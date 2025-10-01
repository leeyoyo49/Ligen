import logging
import os
import sys
from datetime import datetime

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_COLORS = {
    "DEBUG": "\033[36m",       # Cyan
    "INFO": "\033[32m",        # Green
    "WARNING": "\033[33m",     # Yellow
    "ERROR": "\033[31m",       # Red
    "CRITICAL": "\033[37;41m", # White on Red background
    "PERF": "\033[35m",        # Magenta
    "SEP": "\033[94;1m",       # Bright bold blue
}

class ColorFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, use_color=True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record):
        formatted = super().format(record)
        if self.use_color:
            level = record.levelname
            color = ANSI_COLORS.get(level, "")
            if color:
                return f"{color}{formatted}{ANSI_RESET}"
        return formatted

def _mkdirs(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# ---- Custom log levels ----
PERF_LEVEL = 25
logging.addLevelName(PERF_LEVEL, "PERF")
def perf(self, message, *args, **kws):
    if self.isEnabledFor(PERF_LEVEL):
        self._log(PERF_LEVEL, message, args, **kws)
logging.Logger.perf = perf

SEP_LEVEL = 35
logging.addLevelName(SEP_LEVEL, "SEP")
def sep(self, message: str = "", *args, **kws):
    if self.isEnabledFor(SEP_LEVEL):
        if not message:
            message = "=" * 60
        self._log(SEP_LEVEL, message, args, **kws)
logging.Logger.sep = sep
# ---------------------------

def init_logger(exp_name: str, base_dir: str, level: str | int = None):
    """
    Initialize the ligen logger with:
      - colorized console handler (INFO+)
      - per-run file handler (DEBUG+) at: {base_dir}/{exp_name}/logs/{exp_name}_{ts}.log

    Returns:
      logger: logging.Logger (base: "ligen")
      meta: dict with paths
    """
    logger = logging.getLogger("ligen")
    if getattr(logger, "_is_configured", False):
        return logger, getattr(logger, "_meta", {})

    # Resolve level
    if level is None:
        level = os.getenv("LIGEN_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Build paths
    run_dir = os.path.join(base_dir, exp_name, "logs")
    _mkdirs(run_dir)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log = os.path.join(run_dir, f"{exp_name}_{ts}.log")
    latest_log = os.path.join(run_dir, "latest.log")

    # Base logger
    logger.setLevel(logging.DEBUG)  # capture all; handlers filter
    logger.propagate = False

    # Console handler (color)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(ColorFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        use_color=True
    ))
    logger.addHandler(ch)

    # File handler (no color, full detail)
    fh = logging.FileHandler(run_log, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(process)d | %(name)s | %(filename)s:%(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    # Symlink/copy latest
    try:
        if os.path.islink(latest_log) or os.path.exists(latest_log):
            os.remove(latest_log)
        os.symlink(os.path.basename(run_log), latest_log)
    except Exception:
        # fallback: write a small pointer file if symlink not allowed
        try:
            with open(latest_log, "w", encoding="utf-8") as f:
                f.write(run_log + "\n")
        except Exception:
            pass

    # Capture warnings into logging
    logging.captureWarnings(True)

    meta = {"run_log": run_log, "latest": latest_log, "run_dir": run_dir}
    logger._is_configured = True
    logger._meta = meta

    logger.info(f"Logger initialized. File: {run_log}")
    return logger, meta

def get_logger(name: str | None = None) -> logging.Logger:
    base = logging.getLogger("ligen")
    return base.getChild(name) if name else base
