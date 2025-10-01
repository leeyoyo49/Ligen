import os, yaml, random, numpy as np, torch

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Defaults
    cfg.setdefault("device", "auto")
    cfg.setdefault("seed", 0)
    cfg.setdefault("batch_size", 4096)
    cfg.setdefault("epochs", 5000)

    # Dataset
    ds = cfg.setdefault("dataset", {})
    ds.setdefault("kind", "wifi")  # 'wifi' or 'spectral'
    if "path" not in ds or "x_cols" not in ds or "y_cols" not in ds:
        raise ValueError("config.yaml must include dataset.kind, dataset.path, dataset.x_cols, dataset.y_cols")

    # Experiment
    exp = cfg.setdefault("experiment", {})
    exp.setdefault("base_dir", "experiment")
    exp.setdefault("name", "default")

    # W&B
    wb = cfg.setdefault("wandb", {})
    wb.setdefault("enabled", False)
    wb.setdefault("project", "localizer")
    wb.setdefault("group", "tabular_gans")
    wb.setdefault("name_prefix", "")

    return cfg

def experiment_dirs(cfg: dict, make: bool = True) -> dict:
    base = cfg["experiment"]["base_dir"]
    name = cfg["experiment"]["name"]
    exp_dir = os.path.join(base, name)
    model_dir = os.path.join(exp_dir, "model")
    gen_dir   = os.path.join(exp_dir, "generated_data")
    if make:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(gen_dir, exist_ok=True)
    return {"exp_dir": exp_dir, "model_dir": model_dir, "gen_dir": gen_dir}

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def pick_device(name: str):
    if name == "cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cpu":  return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_hms(sec: float) -> str:
    sec = int(max(sec, 0)); h = sec // 3600; m = (sec % 3600)//60; s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
