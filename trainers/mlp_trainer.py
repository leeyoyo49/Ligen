import torch, optuna, copy
from optuna.logging import set_verbosity, ERROR  # 或 WARNING/CRITICAL

from typing import Dict
from models.mlp import Net, CustomLoss
from utils.timing import ETAMeter, format_hms, wandb_is_enabled
from types import SimpleNamespace
from utils.logger import get_logger

from tqdm.auto import tqdm

set_verbosity(ERROR)

# --- helper: robustly convert single-valued tensors/arrays to float ---
def as_scalar(x):
    """Return a Python float for scalars, 0-d/1x1 tensors/ndarrays; otherwise try float(x)."""
    try:
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().reshape(-1)[0])
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(x, np.ndarray) and x.size == 1:
            return float(x.reshape(-1)[0])
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return x

def dict_to_ns(d: dict):
    return SimpleNamespace(**d)


class MLPTrainer:
    def __init__(self, input_size: int, output_size: int, cfg_mlp: dict, device: str, wb=None, tag="mlp"):
        self.input_size = input_size
        self.output_size = output_size
        self.cfg = dict_to_ns(cfg_mlp)
        self.device = torch.device(device)
        self.wb = wb
        self.tag = tag
        self.best_params: Dict = {}
        self.train_loader = None
        self.test_loader = None
        self.all_loader = None
        self._wb_meta = None  # ← 由 main 注入
        self.log = get_logger(f"trainer.mlp.{tag}")

    def set_wandb_meta(self, meta: Dict):
        """meta keys: project, group, experiment_name, dataset, p, seed"""
        self._wb_meta = dict(meta or {})
        self.log.debug(f"wandb_meta set: {self._wb_meta}")

    def set_loaders(self, all_loader, train_loader, test_loader):
        self.all_loader = all_loader
        self.train_loader = train_loader
        self.test_loader = test_loader
        try:
            self.log.debug(f"loaders set: all={len(all_loader)} train={len(train_loader)} test={len(test_loader)}")
        except Exception:
            self.log.debug("loaders set.")

    def _ensure_xy(self, x: torch.Tensor, y: torch.Tensor):
        """Ensure (x,y) are ordered as (features, targets).
        If a loader mistakenly yields (coords, features) with shapes (B,2),(B,input_size), swap them.
        """
        try:
            if x.ndim == 2 and y.ndim == 2:
                xin, yin = x.size(1), y.size(1)
                if xin != self.input_size and yin == self.input_size:
                    return y, x
        except Exception:
            pass
        return x, y

    # --------------------------
    # Optuna tuning → 每個 trial 開子 run（同 group）
    # --------------------------
    def _objective(self, trial: optuna.Trial):
        hidden = trial.suggest_int("hidden_size", self.cfg.hidden_min, self.cfg.hidden_max, log=True)
        lr     = trial.suggest_float("learning_rate", self.cfg.lr_min, self.cfg.lr_max, log=True)
        drop   = trial.suggest_float("dropout_rate", self.cfg.dropout_min, self.cfg.dropout_max)
        self.log.debug(f"[{self.tag}] trial {trial.number} params: hidden={hidden} lr={lr:.5f} drop={drop:.3f}")

        model = Net(self.input_size, hidden, self.output_size, drop).to(self.device)
        # Sum over batch, then we'll divide by total sample count for true per-sample average
        criterion = CustomLoss(reduction="sum")
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        # ✅ 用 optuna_epochs（原本是 epoch_cnt）
        meter = ETAMeter(self.cfg.optuna_epochs); meter.start()

        trun = None
        if self.wb is not None and wandb_is_enabled() and self._wb_meta is not None:
            meta = self._wb_meta
            project = meta.get("project", "localizer")
            group   = meta.get("group")
            exp     = meta.get("experiment_name", "exp")
            dataset = meta.get("dataset", "ds")
            p       = meta.get("p", -1)
            seed    = meta.get("seed", -1)
            # Tags for discoverability (from meta + trial specific)
            base_tags = list(meta.get("tags", []))
            trial_tags = base_tags + [self.tag, f"trial:{trial.number}"]

            trial_run_name = f"{exp}-{dataset}-p{p}-s{seed}-{self.tag}-t{trial.number}"
            trun = self.wb.init(
                project=project,
                group=group,
                job_type=f"{self.tag}_tune",
                name=trial_run_name,
                tags=trial_tags,
                reinit=True,
                config={
                    "experiment_name": exp,
                    "parent_group": group,
                    "trial_run_name": trial_run_name,
                    "trial_number": trial.number,
                    "dataset": dataset, "p": p, "seed": seed,
                    "mlp_tag": self.tag,
                    "hidden_size": hidden,
                    "learning_rate": lr,
                    "dropout_rate": drop,
                },
            )
            self.wb.define_metric(f"{self.tag}/loss/tune/*", step_metric="epoch")
            self.wb.define_metric(f"{self.tag}/time/*",      step_metric="epoch")

        # ✅ 這個 trial 的進度條
        pbar = tqdm(total=self.cfg.optuna_epochs,
                    desc=f"Optuna t{trial.number} ({self.tag})",
                    leave=False, dynamic_ncols=True)

        # NEW: per-epoch metrics table for this trial
        tune_table = None
        if trun is not None:
            tune_table = self.wb.Table(columns=["epoch", "train_loss", "eval_loss", "eta_sec"])

        try:
            for epoch in range(self.cfg.optuna_epochs):
                # train (per-sample average: sum over batches / total samples)
                model.train(); train_sum, train_n = 0.0, 0
                for x, y in self.train_loader:
                    x, y = self._ensure_xy(x, y)
                    x, y = x.to(self.device).float(), y.to(self.device).float()
                    optim.zero_grad(); loss = criterion(model(x), y); loss.backward(); optim.step()
                    bsz = int(y.size(0))
                    train_sum += float(loss.item())  # loss is summed over batch
                    train_n += bsz
                train_loss = train_sum / max(train_n, 1)

                # eval (per-sample average)
                model.eval(); eval_sum, eval_n = 0.0, 0
                with torch.no_grad():
                    for x, y in self.test_loader:
                        x, y = self._ensure_xy(x, y)
                        x, y = x.to(self.device).float(), y.to(self.device).float()
                        l = criterion(model(x), y).item()  # summed over batch
                        eval_sum += float(l)
                        eval_n += int(y.size(0))
                eval_loss = eval_sum / max(eval_n, 1)

                _, _, remain = meter.update()

                # ✅ 更新進度條提示
                pbar.update(1)
                pbar.set_postfix_str(f"train={train_loss:.4f} eval={eval_loss:.4f} ETA={format_hms(remain)}")

                # Populate the tune table with metrics
                if tune_table is not None:
                    tune_table.add_data(
                        int(epoch),
                        as_scalar(train_loss),
                        as_scalar(eval_loss),
                        as_scalar(remain),
                    )

                # Log to W&B during tuning
                if trun is not None:
                    trun.log({
                        f"{self.tag}/loss/tune/train_loss": as_scalar(train_loss),
                        f"{self.tag}/loss/tune/eval_loss":  as_scalar(eval_loss),
                        f"{self.tag}/time/tune_eta_sec":    as_scalar(remain),
                        "epoch": epoch,
                    })

                trial.report(eval_loss, epoch)
                # Pruning disabled - let all trials run to completion

            pbar.close()
            if trun is not None:
                trun.log({f"{self.tag}/time/tune_time_total": meter.elapsed, "epoch": self.cfg.optuna_epochs - 1})
                # NEW: log the completed table once per trial
                if tune_table is not None:
                    trun.log({f"{self.tag}/tables/tune_metrics": tune_table})
                trun.finish()

            return eval_loss

        except Exception:
            pbar.close()
            if trun is not None:
                trun.finish()
            self.log.exception(f"[{self.tag}] trial {trial.number} failed")
            raise

    def tune(self, study_name: str, n_trials: int):
        self.log.info(f"[{self.tag}] tuning start: study={study_name} trials={n_trials}")
        study = optuna.create_study(direction="minimize", study_name=study_name)
        study.optimize(self._objective, n_trials=n_trials)
        self.best_params = study.best_trial.params
        self.log.info(f"[{self.tag}] tuning done: best_value={study.best_value:.6f} params={self.best_params}")

        # 在「當前 active run（通常是 final run 前後沒有）」不強行寫；summary 交給 main 的 final run
        return study.best_value, self.best_params

    # --------------------------
    # Final training（記在當前 active run：final run）
    # --------------------------
    def train_final(self, save_path: str):
        # Add safety check for best_params
        if not self.best_params:
            raise ValueError("Must call tune() before train_final() - no best parameters available")
            
        model = Net(
            self.input_size,
            self.best_params["hidden_size"],
            self.output_size,
            self.best_params["dropout_rate"],
        ).to(self.device)
        # Use 'sum' reduction consistently with tuning phase
        criterion = CustomLoss(reduction="sum")
        optim = torch.optim.Adam(model.parameters(), lr=self.best_params["learning_rate"])

        meter = ETAMeter(self.cfg.epoch_cnt)
        meter.start()
        self.log.info(f"[{self.tag}] final training: save_path={save_path} params={self.best_params}")

        # W&B setup for final training (log to the active run started by main)
        final_table = None
        if self.wb is not None and wandb_is_enabled():
            try:
                self.wb.define_metric(f"{self.tag}/loss/final/*", step_metric="epoch")
                self.wb.define_metric(f"{self.tag}/time/*", step_metric="epoch")
                final_table = self.wb.Table(columns=["epoch", "train_loss", "eval_loss", "eta_sec"])
            except Exception:
                # Non-fatal: continue even if define_metric/table creation fails
                pass

        # Progress bar only (no W&B per-epoch loss recording during finetuning)
        pbar = tqdm(total=self.cfg.epoch_cnt, desc=f"[MLP] Training {self.tag}", dynamic_ncols=True)

        best_eval = float("inf")
        best_epoch = None
        best_state = None

        for epoch in range(self.cfg.epoch_cnt):
            model.train()
            total_wsum, total_n = 0.0, 0
            for x, y in self.all_loader:
                x, y = self._ensure_xy(x, y)
                x, y = x.to(self.device).float(), y.to(self.device).float()
                optim.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optim.step()
                bsz = int(y.size(0))
                # loss is summed over batch (reduction='sum')
                total_wsum += float(loss.item())
                total_n += bsz

            # compute eval loss over test/eval loader (dataset-level average) for display only
            model.eval()
            eval_avg_loss = None
            if self.test_loader is not None:
                with torch.no_grad():
                    eval_wsum, eval_n = 0.0, 0
                    for x, y in self.test_loader:
                        x, y = self._ensure_xy(x, y)
                        x, y = x.to(self.device).float(), y.to(self.device).float()
                        l = criterion(model(x), y)  # summed over batch (reduction='sum')
                        bsz = int(y.size(0))
                        eval_wsum += float(l.item())
                        eval_n += bsz
                eval_avg_loss = eval_wsum / max(eval_n, 1)

            _, _, remain = meter.update()
            avg_loss = total_wsum / max(total_n, 1)

            pbar.update(1)
            if eval_avg_loss is not None:
                pbar.set_postfix_str(f"train={avg_loss:.4f} eval={eval_avg_loss:.4f} ETA={format_hms(remain)}")
                # track best by evaluation loss
                if eval_avg_loss < best_eval:
                    best_eval = float(eval_avg_loss)
                    # keep a CPU copy of the best weights
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
            else:
                pbar.set_postfix_str(f"loss={avg_loss:.4f} ETA={format_hms(remain)}")

            # W&B per-epoch logging (final training)
            if final_table is not None:
                try:
                    final_table.add_data(
                        int(epoch),
                        as_scalar(avg_loss),
                        as_scalar(eval_avg_loss) if eval_avg_loss is not None else None,
                        as_scalar(remain),
                    )
                except Exception:
                    pass
            if self.wb is not None and wandb_is_enabled():
                try:
                    log_payload = {
                        f"{self.tag}/loss/final/train_loss": as_scalar(avg_loss),
                        f"{self.tag}/time/final_eta_sec": as_scalar(remain),
                        "epoch": epoch,
                    }
                    if eval_avg_loss is not None:
                        log_payload[f"{self.tag}/loss/final/eval_loss"] = as_scalar(eval_avg_loss)
                    self.wb.log(log_payload)
                except Exception:
                    pass

        pbar.close()

        # load best weights before saving, if available
        if best_state is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        torch.save(model, save_path)

        # Final W&B logging: total time, best epoch/metric, and the per-epoch table
        if self.wb is not None and wandb_is_enabled():
            try:
                payload = {f"{self.tag}/time/train_time_sec": as_scalar(meter.elapsed), "epoch": self.cfg.epoch_cnt - 1}
                if best_eval != float("inf"):
                    payload[f"{self.tag}/loss/final/best_eval"] = as_scalar(best_eval)
                if best_epoch is not None:
                    payload[f"{self.tag}/final/best_epoch"] = int(best_epoch)
                self.wb.log(payload)
                if final_table is not None:
                    self.wb.log({f"{self.tag}/tables/final_metrics": final_table})
            except Exception:
                pass
        self.log.info(f"training finished. total_sec={meter.elapsed:.2f}")
        return model

    @torch.no_grad()
    def evaluate_pointwise_rmse(self, model, df, x_cols, y_cols):
        import numpy as np
        device = self.device; total = 0.0
        for i in range(len(df)):
            x = torch.tensor(df.iloc[i][x_cols].values, dtype=torch.float32, device=device).unsqueeze(0)
            y = torch.tensor(df.iloc[i][y_cols].values, dtype=torch.float32, device=device).unsqueeze(0)
            pred = model(x).cpu().numpy(); gt = y.cpu().numpy()
            total += np.sqrt(((pred - gt)**2).sum())
        rmse = float(total / len(df))
        self.log.info(f"[{self.tag}] pointwise RMSE on {len(df)} samples = {rmse:.6f}")
        return rmse