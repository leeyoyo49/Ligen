import os, time, random
import numpy as np, torch

def ensure_dirs(*paths):
    for p in paths: os.makedirs(p, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def format_hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

class ETAMeter:
    def __init__(self, total_steps: int, alpha: float = 0.9):
        self.total = max(int(total_steps), 1)
        self.alpha = alpha; self.step = 0
        self.last_t = None; self.ema = None; self.elapsed = 0.0
    def start(self): import time; self.last_t = time.perf_counter()
    def update(self, n: int = 1):
        import time
        now = time.perf_counter()
        dt = now - (self.last_t or now)
        self.last_t = now; self.elapsed += dt; self.step += n
        self.ema = dt if self.ema is None else self.alpha*self.ema + (1-self.alpha)*dt
        remain = max(self.total - self.step, 0) * self.ema
        return dt, self.ema, remain

def wandb_is_enabled() -> bool:
    return os.getenv("WANDB_DISABLED", "false").lower() not in ("true","1","yes")
