import os
import argparse
import pandas as pd
import time
import shutil

from utils.config import load_config
from utils.timing import ensure_dirs, set_seed, wandb_is_enabled
from dataloader import api_for_kind
from trainers.mlp_trainer import MLPTrainer
from trainers.cgan_trainer import CGANTrainer
from utils.logger import init_logger, get_logger  # <-- added

os.environ["WANDB_SILENT"] = "true"   # 關掉那些「Tracking run / View project / Run summary」橫幅

# --- helper: robustly convert single-valued tensors/arrays to float ---
def as_scalar(x):
    """Return a Python float for scalars, 0-d/1x1 tensors/ndarrays; otherwise try float(x)."""
    try:
        import torch
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


def run(cfg_path: str):
    cfg = load_config(cfg_path)
    exp_name = cfg["experiment"]["name"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    exp_name = f"CGAN-{exp_name}-{ts}"
    
    # initialize logger (store logs under experiment/<name>/logs)
    log_base_dir = cfg["experiment"]["base_dir"]
    logger, log_meta = init_logger(exp_name=exp_name, base_dir=log_base_dir, level=cfg.get("log_level", None))
    log = get_logger("cgan")

    # NEW: report which log files are active
    try:
        import logging
        def _file_handlers(lg):
            paths = []
            for h in getattr(lg, "handlers", []):
                p = getattr(h, "baseFilename", None)
                if p:
                    paths.append(os.path.abspath(p))
            return paths
        root = logging.getLogger()
        active_files = set(_file_handlers(logger)) | set(_file_handlers(root)) | set(_file_handlers(log))
        if active_files:
            log.info(f"Active log files: {sorted(active_files)}")
    except Exception as e:
        log.debug(f"Could not introspect logger handlers: {e}")

    # --- paths/dirs ---
    exp_dirs = {
        "models_dir": os.path.join(cfg["experiment"]["base_dir"], exp_name, "model"),
        "mlp_model_dir": os.path.join(cfg["experiment"]["base_dir"], exp_name, "model"),
        "gen_data_dir": os.path.join(cfg["experiment"]["base_dir"], exp_name, "generated_data"),
    }
    results_path = os.path.join("Results", f"CGAN-{cfg['dataset']['kind']}-{exp_name}-results.csv")
    ensure_dirs(exp_dirs["models_dir"], exp_dirs["mlp_model_dir"], exp_dirs["gen_data_dir"], os.path.dirname(results_path))
    if not os.path.exists(results_path):
        pd.DataFrame(columns=["model","train_size","seed","test_loss_before","test_loss_after"]).to_csv(results_path, index=False)

    # copy the exact config used into the experiment folder for reproducibility
    try:
        cfg_dest = os.path.join(cfg["experiment"]["base_dir"], exp_name, os.path.basename(cfg_path))
        shutil.copyfile(cfg_path, cfg_dest)
        # also keep a canonical name for convenience
        cfg_dest2 = os.path.join(cfg["experiment"]["base_dir"], exp_name, "run_config.yaml")
        shutil.copyfile(cfg_path, cfg_dest2)
        log.info(f"Copied config to {cfg_dest} and {cfg_dest2}")
    except Exception as e:
        log.warning(f"Failed to copy config into experiment folder: {e}")

    # print config
    log.info(f"Loaded config from {cfg_path}")
    log.info(f"Saving results to {results_path}, experiment dirs: {cfg['experiment']['base_dir']}")
    for k, v in cfg.items():
        log.debug(f"  {k}: {v}")

    # --- dataset selection (wifi | spectral) ---
    ds = cfg["dataset"]
    data_api = api_for_kind(ds["kind"])
    df_all = data_api.load_df(ds["path"])
    if "sample_id" in df_all.columns:
        df_all = df_all.drop(columns=["sample_id"])
    x_cols = list(ds["x_cols"]); y_cols = list(ds["y_cols"])
    x_dim  = len(x_cols)

    # --- device ---
    device = cfg.get("device", "auto")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    seed_cnt = cfg.get("seeds", 10)
    seeds = list(range(int(seed_cnt)))

    # --- wandb ---
    USE_WB = bool(cfg.get("wandb", {}).get("enabled", False)) and wandb_is_enabled()
    wb_cfg = cfg.get("wandb", {})
    if USE_WB:
        try:
            import wandb
        except Exception as e:
            log.warning(f"W&B import failed; disabling: {e}")
            USE_WB = False

    train_sizes = [9,8,7,6,5,4,3,2,1]
    for p in train_sizes:
        frac = p / 10.0
        for seed in seeds:
            p_t_time_start = time.time()
            log.sep()
            log.info(f"Seed: {seed}, Train size: {p} (frac={frac:.1f})")
            set_seed(seed)
            train_df, test_df = data_api.split_by_frac(df_all, frac=frac, seed=seed, y_cols=y_cols)

            # 以 group 當“大 trial”識別（原本）＋加入時間戳
            group_name = f"{exp_name}-{ds['kind']}-p{p}-s{seed}"
            # 新增：使用「實驗名」做為更高層 group；原 trial 改成 tag/metadata
            exp_group = exp_name
            trial_id = f"{ds['kind']}-p{p}-s{seed}"
            tags_common = [exp_name, ds['kind'], f"p{p}", f"s{seed}", trial_id]

            # ===== 1) CGAN run =====
            if USE_WB:
                run_cgan = wandb.init(
                    project=wb_cfg.get("project", "localizer"),
                    name=f"{exp_name}-{ds['kind']}-p{p}-s{seed}-CGAN",
                    group=exp_group,                     # <--- top-level group = experiment
                    tags=tags_common + ["cgan"],         # <--- small group info moved to tags
                    job_type="cgan",
                    config={
                        "experiment_name": exp_name,
                        "dataset": ds["kind"], "p": p, "seed": seed,
                        "device": device, "x_dim": x_dim,
                        "mlp": cfg.get("mlp", {}), "cgan": cfg.get("cgan", {}),
                        "trial_id": trial_id              # <--- handy in the UI/table
                    },
                    reinit=True
                )
                wandb.define_metric("cgan/*", step_metric="cgan_step")

            cgan = CGANTrainer(
                coord_dim=len(y_cols),
                feat_dim=x_dim,
                cfg_cgan=cfg.get("cgan", {}),
                device=device,
                wb=(wandb if USE_WB else None),
            )

            ckpt_path = os.path.join(exp_dirs["models_dir"], f"cgan_p{p}_s{seed}.pth")
            pair_loader = data_api.make_pair_loader(train_df, y_cols=y_cols, x_cols=x_cols, batch_size=cfg.get("batch_size", 4096))
            test_loader = data_api.make_pair_loader(test_df, y_cols=y_cols, x_cols=x_cols, batch_size=cfg.get("batch_size", 4096))
            log.info(f"Training CGAN (p={p}, s={seed}) → {ckpt_path}")
            cgan.train(pair_loader, ckpt_path=ckpt_path, resume=True)

            # new: use synth_multi; fallback to legacy synth_repeats if present
            synth_multi = int(cfg["cgan"].get("synth_multi", cfg["cgan"].get("synth_repeats", 1)))
            feats_df, coords_np = cgan.synthesize_grid(synth_multi=synth_multi, columns=x_cols)
            synth_out = feats_df.copy()
            synth_out.insert(0, "positionY", coords_np[:,1])
            synth_out.insert(0, "positionX", coords_np[:,0])
            synth_csv = os.path.join(exp_dirs["gen_data_dir"], f"CGAN_generated_data_p{p}_s{seed}.csv")
            synth_out.to_csv(synth_csv, index=False)

            if USE_WB:
                run_cgan.finish()

            # ===== 2) MLP tuning（每個 trial 都是子 run；同 group） =====
            # real only
            all_loader, tr_loader, te_loader = data_api.make_mlp_dataloaders(
                train_df, x_cols, y_cols, cfg.get("mlp",{}).get("batch_size", 4096), seed)
            mlp1 = MLPTrainer(input_size=x_dim, output_size=2, cfg_mlp=cfg.get("mlp", {}), device=device,
                              wb=(wandb if USE_WB else None), tag="mlp1")
            mlp1.set_loaders(all_loader, tr_loader, test_loader)
            mlp1.set_wandb_meta({
                "project": wb_cfg.get("project", "localizer"),
                "group": exp_group,                    # <--- top-level group
                "experiment_name": exp_name,
                "dataset": ds["kind"], "p": p, "seed": seed,
                "trial_id": trial_id,
                "tags": tags_common + ["mlp", "real_only"]  # <--- keep trial in tags
            })
            best1, bp1 = mlp1.tune(study_name=f"{exp_name}-MLP1-p{p}-s{seed}", n_trials=cfg.get("mlp",{}).get("optuna_trials", 5))

            # real + synthetic
            aug_train = pd.concat([train_df, synth_out], ignore_index=True)
            all_loader2, tr_loader2, te_loader2 = data_api.make_mlp_dataloaders(
                aug_train, x_cols, y_cols, cfg.get("mlp",{}).get("batch_size", 4096), seed)
            mlp2 = MLPTrainer(input_size=x_dim, output_size=2, cfg_mlp=cfg.get("mlp", {}), device=device,
                              wb=(wandb if USE_WB else None), tag="mlp2")
            mlp2.set_loaders(all_loader2, tr_loader2, test_loader)
            mlp2.set_wandb_meta({
                "project": wb_cfg.get("project", "localizer"),
                "group": exp_group,                    # <--- top-level group
                "experiment_name": exp_name,
                "dataset": ds["kind"], "p": p, "seed": seed,
                "trial_id": trial_id,
                "tags": tags_common + ["mlp", "real_plus_synth"]  # <--- keep trial in tags
            })
            best2, bp2 = mlp2.tune(study_name=f"{exp_name}-MLP2-p{p}-s{seed}", n_trials=cfg.get("mlp",{}).get("optuna_trials", 5))

            # ===== 3) Final run（父層：final 訓練與評估） =====
            if USE_WB:
                run_final = wandb.init(
                    project=wb_cfg.get("project", "localizer"),
                    name=f"{exp_name}-{ds['kind']}-p{p}-s{seed}-FINAL",
                    group=exp_group,                      # <--- top-level group
                    tags=tags_common + ["final"],         # <--- small group info as tags
                    job_type="final",
                    config={
                        "experiment_name": exp_name,
                        "dataset": ds["kind"], "p": p, "seed": seed,
                        "device": device, "x_dim": x_dim,
                        "mlp_best_1": bp1, "mlp_best_2": bp2,
                        "trial_id": trial_id
                    },
                    reinit=True
                )
                wandb.define_metric("mlp1/final/*", step_metric="mlp1_final_step")
                wandb.define_metric("mlp2/final/*", step_metric="mlp2_final_step")

            # train final
            m1_path = os.path.join(exp_dirs["mlp_model_dir"], f"MLP_pipeline_1_p{p}_s{seed}")
            m1 = mlp1.train_final(save_path=m1_path)
            test_loss_bef = mlp1.evaluate_pointwise_rmse(m1, test_df, x_cols, y_cols)
            log.perf(f"Loss before synth: {test_loss_bef:.6f}")
            if USE_WB: wandb.log({"eval/test_loss_before": as_scalar(test_loss_bef)})

            m2_path = os.path.join(exp_dirs["mlp_model_dir"], f"MLP_pipeline_2_p{p}_s{seed}")
            m2 = mlp2.train_final(save_path=m2_path)
            test_loss_aft = mlp2.evaluate_pointwise_rmse(m2, test_df, x_cols, y_cols)
            log.perf(f"Loss after synth: {test_loss_aft:.6f}")
            if USE_WB: wandb.log({"eval/test_loss_after": as_scalar(test_loss_aft)})

            # NEW: log single-value metrics as a wandb.Table for better visualization
            if USE_WB:
                try:
                    eval_tbl = wandb.Table(columns=[
                        "experiment","trial_id","dataset","p","seed",
                        "test_loss_before","test_loss_after"
                    ])
                    eval_tbl.add_data(
                        exp_name, trial_id, ds["kind"], int(p), int(seed),
                        as_scalar(test_loss_bef), as_scalar(test_loss_aft)
                    )
                    wandb.log({"eval/summary_table": eval_tbl})
                except Exception as e:
                    log.debug(f"W&B eval table log failed: {e}")

            if USE_WB:
                run_final.finish()

            # ===== 4) append results CSV =====
            row = pd.DataFrame({
                "model": [f"{exp_name}_CGAN_aug_{ds['kind']}_p{p}_s{seed}"],
                "train_size": [p],
                "seed": [seed],
                "test_loss_before": [test_loss_bef],
                "test_loss_after": [test_loss_aft],
            })
            prev = pd.read_csv(results_path) if os.path.exists(results_path) else pd.DataFrame()
            # row 可能是 Series 或 DataFrame，都先轉成 DataFrame 再處理
            row_df = row.to_frame().T if isinstance(row, pd.Series) else row

            def _is_empty_or_all_na(df):
                return df is None or df.empty or df.isna().all().all()

            frames = [df for df in [prev, row_df] if not _is_empty_or_all_na(df)]
            prev = pd.concat(frames, ignore_index=True) if frames else prev
            prev.to_csv(results_path, index=False)
            # log.info(f"Result appended → {results_path}")
            p_t_time_end = time.time()

            log.perf(f"Total time for p={p}, s={seed}: {p_t_time_end - p_t_time_start:.2f} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiment/spectral.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
