from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # …/dreamerv2/..

def ensure_dir(p: Union[str, Path]) -> Path:
    """Create the directory (and parents) if needed and return the Path."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def resolve_ckpt(ckpt: Union[str, Path]) -> Path:
    """
    Robustly resolve a checkpoint specification:
      • absolute file / dir
      • relative to cwd
      • relative to project root
      • directory containing many ckpts
    Returns the concrete file path to load.
    """
    p = Path(ckpt)

    # 1) treat pure directory → pick an actual file
    if p.is_dir():
        # preference order: best → latest → highest-numberedubub
        for name in ("models_best.pth", "models_latest.pth"):
            cand = p / name
            if cand.exists():
                return cand
        iters = sorted(p.glob("models_*.pth"))
        if iters:
            return iters[-1]
        raise FileNotFoundError(f"No checkpoints found inside {p}")

    # 2) explicit file that happens to be relative
    if not p.is_absolute():
        p = (Path.cwd() / p)
        if not p.exists():
            p = PROJECT_ROOT / ckpt   # fall-back to repo root

    if p.exists():
        return p
    raise FileNotFoundError(f"Checkpoint not found: {p}")
