import os
import argparse
import pandas as pd
import torch
import time
import shutil

from utils.config import load_config
from utils.timing import ensure_dirs, set_seed, wandb_is_enabled
from dataloader import api_for_kind
from trainers.freegan_trainer import freeganTrainer
from trainers.mlp_trainer import MLPTrainer
from utils.logger import init_logger, get_logger

os.environ["WANDB_SILENT"] = "true"

# --- helper: robustly convert single-valued tensors/arrays to float ---
def as_scalar(x):
    try:
        import torch as _t
        if isinstance(x, _t.Tensor):
            return float(x.detach().cpu().reshape(-1)[0])
    except Exception:
        pass
    try:
        import numpy as _np
        if isinstance(x, _np.ndarray) and x.size == 1:
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
    exp_name = f"freegan-{exp_name}-{ts}"

    # logger + dirs (align with cgan.py)
    base_dir = cfg["experiment"]["base_dir"]
    logger, _meta = init_logger(exp_name=exp_name, base_dir=base_dir, level=cfg.get("log_level", None))
    log = get_logger("freegan")

    # report active log files (align with cgan.py)
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

    exp_dirs = {
        "models_dir": os.path.join(base_dir, exp_name, "model"),
        "mlp_model_dir": os.path.join(base_dir, exp_name, "model"),
        "gen_data_dir": os.path.join(base_dir, exp_name, "generated_data"),
    }
    results_path = os.path.join("Results", f"freegan-{cfg['dataset']['kind']}-{exp_name}-results.csv")
    ensure_dirs(exp_dirs["models_dir"], exp_dirs["mlp_model_dir"], exp_dirs["gen_data_dir"], os.path.dirname(results_path))
    if not os.path.exists(results_path):
        pd.DataFrame(columns=["model","train_size","seed","test_loss_before","test_loss_after"]).to_csv(results_path, index=False)

    # copy the exact config used into the experiment folder for reproducibility (align with cgan_runner)
    try:
        cfg_dest = os.path.join(base_dir, exp_name, os.path.basename(cfg_path))
        shutil.copyfile(cfg_path, cfg_dest)
        cfg_dest2 = os.path.join(base_dir, exp_name, "run_config.yaml")
        shutil.copyfile(cfg_path, cfg_dest2)
        log.info(f"Copied config to {cfg_dest} and {cfg_dest2}")
    except Exception as e:
        log.warning(f"Failed to copy config into experiment folder: {e}")

    # print config (align with cgan.py)
    log.info(f"Loaded config from {cfg_path}")
    log.info(f"Saving results to {results_path}, experiment dirs: {base_dir}")
    for k, v in cfg.items():
        log.debug(f"  {k}: {v}")

    # dataset
    ds = cfg["dataset"]
    data_api = api_for_kind(ds["kind"])
    df_all = data_api.load_df(ds["path"])
    if "sample_id" in df_all.columns:
        df_all = df_all.drop(columns=["sample_id"])
    x_cols = list(ds["x_cols"])
    y_cols = list(ds["y_cols"])
    x_dim  = len(x_cols)

    # device
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    # wandb (align with cgan.py)
    USE_WB = bool(cfg.get("wandb", {}).get("enabled", False)) and wandb_is_enabled()
    wb_cfg = cfg.get("wandb", {})
    if USE_WB:
        try:
            import wandb
        except Exception as e:
            log.warning(f"W&B import failed; disabling: {e}")
            USE_WB = False

    # seeds: accept int (count) or iterable of seeds; default align with cgan (10)
    _seeds_cfg = cfg.get("seeds", 10)
    if isinstance(_seeds_cfg, int):
        seeds = list(range(int(_seeds_cfg)))
    else:
        try:
            seeds = list(_seeds_cfg)
        except Exception:
            seeds = list(range(10))
    train_sizes = [9,8,7,6,5,4,3,2,1]

    tab_cfg = cfg.get("freegan", {})
    mlp_cfg = cfg.get("mlp", {})
    batch_size = int(mlp_cfg.get("batch_size", cfg.get("batch_size", 4096)))

    for p in train_sizes:
        frac = p / 10.0
        for seed in seeds:
            p_t_time_start = time.time()
            log.sep()
            log.info(f"Seed: {seed}, Train size: {p} (frac={frac:.1f})")
            set_seed(int(seed))
            train_df, test_df = data_api.split_by_frac(df_all, frac=frac, seed=int(seed), y_cols=y_cols)
            # align grouping/tags with CGAN
            group_name = f"{exp_name}-{ds['kind']}-p{p}-s{seed}"
            exp_group = exp_name
            trial_id = f"{ds['kind']}-p{p}-s{seed}"
            tags_common = [exp_name, ds['kind'], f"p{p}", f"s{seed}", trial_id]

            # 1) freegan training
            if USE_WB:
                run_tab = wandb.init(
                    project=wb_cfg.get("project", "localizer"),
                    name=f"{exp_name}-{ds['kind']}-p{p}-s{seed}-freegan",
                    group=exp_group,
                    tags=tags_common + ["freegan"],
                    job_type="freegan",
                    config={
                        "experiment_name": exp_name,
                        "dataset": ds["kind"], "p": p, "seed": seed,
                        "device": device, "x_dim": x_dim,
                        "freegan": tab_cfg, "mlp": mlp_cfg,
                        "trial_id": trial_id
                    },
                    reinit=True
                )
                wandb.define_metric("freegan/*", step_metric="freegan_step")

            gan_loader = data_api.make_feature_loader(train_df, x_cols=x_cols, batch_size=batch_size)
            test_loader = data_api.make_pair_loader(test_df, y_cols=y_cols, x_cols=x_cols, batch_size=cfg.get("batch_size", 4096))
            freegan = freeganTrainer(x_dim=x_dim, cfg_freegan=tab_cfg, device=device, wb=(wandb if USE_WB else None))
            ckpt_path = os.path.join(exp_dirs["models_dir"], f"freegan_p{p}_s{seed}.pth")
            log.info(f"Training freegan (p={p}, s={seed}) → {ckpt_path}")
            freegan.train(gan_loader, ckpt_path=ckpt_path, resume=True)

            # support synth_multi like CGAN; fallback to explicit n_synth
            synth_multi = tab_cfg.get("synth_multi", None)
            if synth_multi is not None:
                try:
                    n_synth = int(synth_multi) * int(len(train_df))
                except Exception:
                    n_synth = int(tab_cfg.get("n_synth", 50000))
            else:
                n_synth = int(tab_cfg.get("n_synth", 50000))

            synth_feats = freegan.synthesize(n=n_synth, columns=x_cols)
            synth_csv = os.path.join(exp_dirs["gen_data_dir"], f"freegan_generated_features_p{p}_s{seed}.csv")
            synth_feats.to_csv(synth_csv, index=False)
            log.info(f"Saved synthetic features → {synth_csv}")

            # log synth stats to W&B (align with CGAN style)
            if USE_WB:
                try:
                    wandb.log({
                        "freegan/synth/samples": int(n_synth),
                        "freegan/synth/synth_multi": int(synth_multi) if synth_multi is not None else None,
                        "freegan/synth/train_rows": int(len(train_df)),
                    })
                    synth_tbl = wandb.Table(columns=["samples","synth_multi","train_rows"]) 
                    synth_tbl.add_data(int(n_synth), int(synth_multi) if synth_multi is not None else -1, int(len(train_df)))
                    wandb.log({"freegan/synth/stats_table": synth_tbl})
                except Exception as e:
                    log.debug(f"W&B synth stats log failed: {e}")

            if USE_WB:
                run_tab.finish()

            # 2) MLP tuning: real only
            all_loader, tr_loader, te_loader = data_api.make_mlp_dataloaders(
                train_df, x_cols, y_cols, batch_size=batch_size, seed=int(seed))
            mlp1 = MLPTrainer(input_size=x_dim, output_size=2, cfg_mlp=mlp_cfg, device=device,
                              wb=(wandb if USE_WB else None), tag="mlp1")
            # Align with cgan_runner: use all_loader for train/val and the test_df-based test_loader for eval
            mlp1.set_loaders(all_loader, tr_loader, test_loader)
            mlp1.set_wandb_meta({
                "project": wb_cfg.get("project", "localizer"),
                "group": exp_group,
                "experiment_name": exp_name,
                "dataset": ds["kind"], "p": p, "seed": seed,
                "trial_id": trial_id,
                "tags": tags_common + ["mlp", "real_only"]
            })
            best1, bp1 = mlp1.tune(study_name=f"{exp_name}-freegan-MLP1-p{p}-s{seed}", n_trials=int(mlp_cfg.get("optuna_trials", 5)))

            # 3) Label synth with baseline MLP, then augmented MLP
            mlp1_model_path = os.path.join(exp_dirs["mlp_model_dir"], f"freegan_MLP_1_p{p}_s{seed}")
            m1 = mlp1.train_final(save_path=mlp1_model_path)

            m1.eval()
            with torch.no_grad():
                X_fake = torch.from_numpy(synth_feats[x_cols].values.astype("float32")).to(device)
                preds = m1(X_fake).cpu().numpy()
            synth_labeled = synth_feats.copy()
            synth_labeled.insert(0, "positionY", preds[:, 1])
            synth_labeled.insert(0, "positionX", preds[:, 0])

            aug_train = pd.concat([train_df, synth_labeled], ignore_index=True)
            all_loader2, tr_loader2, te_loader2 = data_api.make_mlp_dataloaders(
                aug_train, x_cols, y_cols, batch_size=batch_size, seed=int(seed))
            mlp2 = MLPTrainer(input_size=x_dim, output_size=2, cfg_mlp=mlp_cfg, device=device,
                              wb=(wandb if USE_WB else None), tag="mlp2")
            mlp2.set_loaders(all_loader2, tr_loader2, test_loader)
            mlp2.set_wandb_meta({
                "project": wb_cfg.get("project", "localizer"),
                "group": exp_group,
                "experiment_name": exp_name,
                "dataset": ds["kind"], "p": p, "seed": seed,
                "trial_id": trial_id,
                "tags": tags_common + ["mlp", "real_plus_synth"]
            })
            best2, bp2 = mlp2.tune(study_name=f"{exp_name}-freegan-MLP2-p{p}-s{seed}", n_trials=int(mlp_cfg.get("optuna_trials", 5)))

            # 4) Final runs + evaluation
            if USE_WB:
                run_final = wandb.init(
                    project=wb_cfg.get("project", "localizer"),
                    name=f"{exp_name}-{ds['kind']}-p{p}-s{seed}-FINAL",
                    group=exp_group,
                    tags=tags_common + ["final"],
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

            m1_final_path = os.path.join(exp_dirs["mlp_model_dir"], f"freegan_MLP_pipeline_1_p{p}_s{seed}")
            m1_final = mlp1.train_final(save_path=m1_final_path)
            test_loss_bef = mlp1.evaluate_pointwise_rmse(m1_final, test_df, x_cols, y_cols)
            log.perf(f"Loss before synth: {test_loss_bef:.6f}")
            if USE_WB: wandb.log({"eval/test_loss_before": as_scalar(test_loss_bef)})

            m2_final_path = os.path.join(exp_dirs["mlp_model_dir"], f"freegan_MLP_pipeline_2_p{p}_s{seed}")
            m2_final = mlp2.train_final(save_path=m2_final_path)
            test_loss_aft = mlp2.evaluate_pointwise_rmse(m2_final, test_df, x_cols, y_cols)
            log.perf(f"Loss after synth: {test_loss_aft:.6f}")
            if USE_WB: wandb.log({"eval/test_loss_after": as_scalar(test_loss_aft)})

            # NEW: log summary table like CGAN
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

            # 5) Append results (robust like cgan.py)
            row = pd.DataFrame({
                "model": [f"{exp_name}_freegan_aug_{ds['kind']}_p{p}_s{seed}"],
                "train_size": [p],
                "seed": [seed],
                "test_loss_before": [test_loss_bef],
                "test_loss_after": [test_loss_aft],
            })
            prev = pd.read_csv(results_path) if os.path.exists(results_path) else pd.DataFrame()
            row_df = row.to_frame().T if isinstance(row, pd.Series) else row

            def _is_empty_or_all_na(df):
                return df is None or df.empty or df.isna().all().all()

            frames = [df for df in [prev, row_df] if not _is_empty_or_all_na(df)]
            prev = pd.concat(frames, ignore_index=True) if frames else prev
            prev.to_csv(results_path, index=False)
            # log.info(f"Result appended → {results_path}")

            # timing summary like cgan.py
            p_t_time_end = time.time()
            log.perf(f"Total time for p={p}, s={seed}: {p_t_time_end - p_t_time_start:.2f} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiment/spectral.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
