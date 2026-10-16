"""Small shared helpers: device, seeding, timing, JSON."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Deadline:
    """Wall-clock budget; training loops poll `expired()` to stop early."""

    def __init__(self, seconds: float | None):
        self.end = None if seconds is None else time.time() + seconds

    def expired(self) -> bool:
        return self.end is not None and time.time() > self.end

    def remaining(self) -> float:
        return float("inf") if self.end is None else max(0.0, self.end - time.time())


def _default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def write_json(path: Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def batches(n: int, bs: int):
    for i in range(0, n, bs):
        yield slice(i, min(n, i + bs))
