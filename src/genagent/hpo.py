"""Short Optuna search around the agent's initial configuration.

Each trial trains on a subsample with a fraction of the epochs and is scored by a model-agnostic
proxy: MMD between generated samples and real validation data in the shared embedding. This lets
GANs, VAEs, flows and diffusion models be tuned with the same criterion."""

from __future__ import annotations

import math
import time

import numpy as np

from .metrics import median_bandwidth, mmd_rbf
from .utils import Deadline, seed_everything


def _suggest(trial, name, spec):
    t = spec["type"]
    if t == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if t == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    return trial.suggest_categorical(name, spec["choices"])


def proxy_score(model, embedder, X_val, y_val, E_ref_bw, rng, n: int = 1000) -> float:
    idx = rng.choice(len(X_val), min(n, len(X_val)), replace=False)
    Xv = X_val.iloc[idx].reset_index(drop=True) if hasattr(X_val, "iloc") else X_val[idx]
    yv = y_val[idx] if y_val is not None else None
    fake = model.sample(len(idx), yv)
    E_f, E_r = embedder.transform(fake), embedder.transform(Xv)
    if not np.all(np.isfinite(E_f)):
        return float("inf")
    return mmd_rbf(E_r, E_f, E_ref_bw)


def run_search(model_cls, ctx, base_config: dict, space: dict, trials: int, seconds: float,
               train, val, embedder, seed: int = 0, epoch_frac: float = 0.25, data_frac: float = 0.3) -> dict:
    """Return {"best_config", "trials": [...]}; falls back to base_config if no trial completes."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rng = np.random.default_rng(seed)
    n_sub = max(200, int(len(train.X) * data_frac))
    sub = rng.choice(len(train.X), min(n_sub, len(train.X)), replace=False)
    X_sub = train.X.iloc[sub].reset_index(drop=True) if hasattr(train.X, "iloc") else train.X[sub]
    y_sub = train.y[sub] if train.y is not None else None
    bw = median_bandwidth(embedder.transform(X_sub[:1000] if not hasattr(X_sub, "iloc") else X_sub.iloc[:1000]), rng)
    per_trial = seconds / max(trials, 1)
    log = []
    t_end = time.time() + seconds

    def objective(trial):
        cfg = dict(base_config)
        for k, spec in space.items():
            cfg[k] = _suggest(trial, k, spec)
        for k in ("epochs", "diff_steps"):
            if k in cfg:
                cfg[k] = max(1, int(math.ceil(cfg[k] * epoch_frac)))
        seed_everything(seed + trial.number)
        t0 = time.time()
        m = model_cls(cfg, ctx)
        m.fit(X_sub, y_sub, val.X, val.y, Deadline(per_trial * 0.8))
        s = proxy_score(m, embedder, val.X, val.y, bw, rng)
        log.append({"trial": trial.number, "params": {k: cfg[k] for k in space}, "proxy_mmd": s,
                    "seconds": round(time.time() - t0, 1)})
        return s

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    # Enqueue the agent's own initial config first so the search can only improve on it
    study.enqueue_trial({k: base_config[k] for k in space if k in base_config})
    for _ in range(trials):
        if time.time() > t_end:
            break
        try:
            study.optimize(objective, n_trials=1, catch=(RuntimeError, ValueError))
        except Exception as e:  # a failing trial should not abort the run
            log.append({"error": repr(e)})
    done = [t for t in study.trials if t.value is not None and np.isfinite(t.value)]
    best = dict(base_config)
    if done:
        best.update(min(done, key=lambda t: t.value).params)
    return {"best_config": best, "trials": log, "n_completed": len(done)}
