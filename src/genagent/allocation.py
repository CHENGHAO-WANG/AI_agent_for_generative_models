"""How many synthetic samples to generate for each class, given the objective and `n`.

- generation / representation / anomaly: `n` is the total, split proportionally to the real classes
- class balancing, augmentation: `n` is a per-class target; classes below `n` are filled up to `n`
- class balancing, fully synthetic: `n` samples per class
"""

from __future__ import annotations

import numpy as np


class AllocationError(ValueError):
    pass


def proportional(n: int, counts: np.ndarray) -> np.ndarray:
    """Largest-remainder split of n proportional to counts (sums exactly to n)."""
    raw = n * counts / counts.sum()
    base = np.floor(raw).astype(int)
    rest = n - base.sum()
    base[np.argsort(-(raw - base))[:rest]] += 1
    return base


def allocate(objective: str, mode: str | None, n: int | None, class_names: list[str] | None,
             class_counts: np.ndarray | None, exceed_policy: str = "error") -> dict:
    """Return {"per_class": {name: k}, "total": int, "n_effective": int, "notes": [...]}.

    exceed_policy applies only to augmentation when n > majority-class size:
    "clamp" (reduce n to the majority size), "keep" (augment every class to n), "error"."""
    notes: list[str] = []
    if n is None or n <= 0:
        return {"per_class": {}, "total": 0, "n_effective": 0, "notes": ["no samples requested"]}
    if class_counts is None:
        if objective == "class_balancing":
            raise AllocationError("class balancing requires labels")
        return {"per_class": {"all": int(n)}, "total": int(n), "n_effective": int(n), "notes": notes}

    counts = np.asarray(class_counts)
    if objective == "class_balancing" and (mode or "augmentation") == "augmentation":
        majority = int(counts.max())
        if n > majority:
            if exceed_policy == "clamp":
                notes.append(f"n={n} exceeds the majority-class size {majority}; clamped to {majority}")
                n = majority
            elif exceed_policy == "keep":
                notes.append(f"n={n} exceeds the majority-class size {majority}; kept, so every class is augmented")
            else:
                raise AllocationError(
                    f"n={n} exceeds the majority-class size ({majority}). Choose: clamp to {majority}, "
                    f"keep n (augment every class), or a new n <= {majority}.")
        per = np.maximum(0, n - counts)
        notes.append(f"augmentation target: every class with fewer than {n} real samples is filled up to {n}")
    elif objective == "class_balancing":
        per = np.full(len(counts), n)
        notes.append(f"fully synthetic balanced set: {n} samples per class")
    else:
        per = proportional(n, counts)
        notes.append("total n split proportionally to the real class distribution")
    return {"per_class": {name: int(k) for name, k in zip(class_names, per)}, "total": int(per.sum()),
            "n_effective": int(n), "notes": notes}


def labels_from_allocation(alloc: dict, class_names: list[str] | None, seed: int = 0) -> np.ndarray | None:
    if class_names is None or "all" in alloc["per_class"]:
        return None
    idx = {c: i for i, c in enumerate(class_names)}
    y = np.concatenate([np.full(k, idx[c], dtype=np.int64) for c, k in alloc["per_class"].items()]) \
        if alloc["per_class"] else np.zeros(0, np.int64)
    return y
