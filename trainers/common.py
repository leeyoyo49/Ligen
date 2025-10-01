import time

def format_hms(sec: float) -> str:
    s = int(max(sec, 0))
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

class ETAMeter:
    """Exponential moving average epoch timer with ETA and total elapsed."""
    def __init__(self, total_steps: int, alpha: float = 0.9):
        self.total = max(int(total_steps), 1)
        self.alpha = alpha
        self.step = 0
        self.last = None
        self.ema = None
        self.elapsed = 0.0

    def start(self):
        self.last = time.perf_counter()

    def update(self, inc: int = 1):
        now = time.perf_counter()
        dt = now - (self.last or now)
        self.last = now
        self.elapsed += dt
        self.step += inc
        self.ema = dt if self.ema is None else self.alpha * self.ema + (1 - self.alpha) * dt
        remain = max(self.total - self.step, 0) * (self.ema or dt)
        return dt, self.ema, remain
