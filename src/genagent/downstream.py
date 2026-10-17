"""Objective-specific downstream analysis of the selected model and its synthetic data.

- class_balancing (augmentation): classifiers trained on real, real with class-weighted loss (the
  non-generative alternative), real + synthetic, and real + synthetic with class weighting, all with the
  same training effort and tested on the real test set over several seeds, with bootstrap intervals
- class_balancing (synthetic_balanced): classifier trained on synthetic only vs real
- representation: latent embeddings of every split, latent UMAP
- anomaly_detection: per-sample anomaly scores, threshold from validation data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import figures as FIG
from .embed import predict_proba, train_classifier
from .utils import write_json

N_SEEDS = 3


def save_synthetic(run, ds, fake, y_fake) -> Path:
    """Save in the input's format: images -> .npz (uint8 NHWC + labels), scRNA -> .h5ad, tabular -> .csv."""
    d = run.f("synthetic", "x").parent
    if ds.modality == "image":
        imgs = (np.clip(fake, 0, 1) * 255).round().astype(np.uint8).transpose(0, 2, 3, 1)
        p = d / "synthetic.npz"
        np.savez_compressed(p, images=imgs, labels=y_fake if y_fake is not None else np.zeros(0))
        FIG.image_grid(fake, y_fake, ds.class_names, run.f("figures", "final_samples.png"),
                       title="final synthetic samples")
    elif ds.modality == "scrna":
        import anndata as ad

        a = ad.AnnData(fake.astype(np.float32))
        a.var_names = ds.meta["var_names"]
        a.obs_names = [f"syn{i}" for i in range(len(fake))]
        if y_fake is not None:
            a.obs[ds.label_name or "label"] = pd.Categorical(np.asarray(ds.class_names)[y_fake])
        p = d / "synthetic.h5ad"
        a.write_h5ad(p)
    else:
        df = fake.copy()
        if y_fake is not None:
            df[ds.label_name or "label"] = np.asarray(ds.class_names)[y_fake]
        p = d / "synthetic.csv"
        df.to_csv(p, index=False)
    return p


def run_downstream(run, ds, plan, objective, model, fake, y_fake) -> dict:
    """Downstream analysis for one objective, using the model selected for it."""
    if objective == "class_balancing":
        if plan["objective"].get("mode") == "synthetic_balanced":
            return _synthetic_only(run, ds, fake, y_fake)
        return _augmentation(run, ds, fake, y_fake)
    if objective == "representation":
        return _representation(run, ds, model)
    if objective == "anomaly_detection":
        return _anomaly(run, ds, plan, model)
    return {"task": "generation", "note": "quality is summarized by the evaluation metrics"}


# ---------------------------------------------------------------- classifiers

CLS_EPOCHS = 5   # image classifier: epochs over the real training set, used to fix the step budget
CLS_BATCH = 256
N_BOOT = 1000    # bootstrap resamples of the test set for confidence intervals


def _fit_predict(ds, emb, Xtr, ytr, Xte, seed, steps, class_weight=False):
    """Image: SmallCNN trained for a fixed number of optimizer steps, so every arm gets the same training
    effort whatever its data size; other modalities: gradient boosting on the shared embedding."""
    k = len(ds.class_names)
    if ds.modality == "image":
        net = train_classifier(Xtr, ytr, k, seed=seed, bs=CLS_BATCH, class_weight=class_weight, steps=steps)
        return predict_proba(net, Xte).argmax(1)
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced" if class_weight else None)
    clf.fit(emb.transform(Xtr), ytr)
    return clf.predict(emb.transform(Xte))


def _report(y, pred, names) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

    rec = recall_score(y, pred, average=None, labels=range(len(names)), zero_division=0)
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float((y == pred).mean()),
            "per_class_recall": {n: float(r) for n, r in zip(names, rec)}}


def _concat(a, b):
    return pd.concat([a, b], ignore_index=True) if hasattr(a, "iloc") else np.concatenate([a, b])


def _bootstrap(y, preds: dict[str, list[np.ndarray]], names, seed: int = 0) -> dict:
    """95% bootstrap intervals over test samples (seed-averaged predictions per resample): balanced accuracy
    and per-class recall per arm, and the paired difference of each arm's balanced accuracy to real_only."""
    rng = np.random.default_rng(seed)
    k = len(names)
    correct = {a: np.stack([p == y for p in ps]).astype(float) for a, ps in preds.items()}
    bal = {a: [] for a in preds}
    rec = {a: [] for a in preds}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        cnt = np.bincount(yb, minlength=k)
        for a, cor in correct.items():
            per_seed = np.stack([np.bincount(yb, weights=c[idx], minlength=k) for c in cor]) / np.maximum(cnt, 1)
            r = per_seed.mean(0)
            r[cnt == 0] = np.nan
            rec[a].append(r)
            bal[a].append(np.nanmean(r))
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
    out = {}
    for a in preds:
        R = np.array(rec[a])
        out[a] = {"balanced_accuracy_ci95": ci(bal[a]),
                  "per_class_recall_ci95": {n: ci(R[:, i]) for i, n in enumerate(names)}}
        if a != "real_only" and "real_only" in preds:
            out[a]["delta_balanced_accuracy_vs_real_only_ci95"] = ci(np.array(bal[a]) - np.array(bal["real_only"]))
    return out


def _summarize(runs: dict[str, list[dict]], names) -> dict:
    out = {}
    for arm, res in runs.items():
        out[arm] = {
            "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in res])),
            "balanced_accuracy_sd": float(np.std([r["balanced_accuracy"] for r in res])),
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in res])),
            "per_class_recall_mean": {n: float(np.mean([r["per_class_recall"][n] for r in res])) for n in names},
        }
    return out


def _run_arms(ds, emb, data: dict, te, seed_base: int = 0):
    """Train each arm's classifier N_SEEDS times; returns reports and raw predictions per arm.
    data: {arm: (X, y, class_weight)}."""
    names = ds.class_names
    steps = CLS_EPOCHS * int(np.ceil(len(ds.splits["train"].y) / CLS_BATCH))
    reports, preds = {}, {}
    for arm, (X, y, cw) in data.items():
        preds[arm] = [_fit_predict(ds, emb, X, y, te.X, seed_base + s, steps, cw) for s in range(N_SEEDS)]
        reports[arm] = [_report(te.y, p, names) for p in preds[arm]]
    return reports, preds, steps


def _augmentation(run, ds, fake, y_fake) -> dict:
    emb = run.embedder()
    tr, te = ds.splits["train"], ds.splits["test"]
    names = ds.class_names
    data = {"real_only": (tr.X, tr.y, False), "real_class_weighted": (tr.X, tr.y, True)}
    if fake is not None and len(y_fake) > 0:
        # Built once and shared by both synthetic arms (each copy of image data is ~1 GB)
        Xa, ya = _concat(tr.X, fake), np.concatenate([tr.y, y_fake])
        data["real_plus_synthetic"] = (Xa, ya, False)
        data["real_plus_synthetic_class_weighted"] = (Xa, ya, True)
    reports, preds, steps = _run_arms(ds, emb, data, te)
    summ = _summarize(reports, names)
    boot = _bootstrap(te.y, preds, names, run.info["seed"])
    for a in summ:
        summ[a].update(boot[a])
    FIG.bars({n: {a: summ[a]["per_class_recall_mean"][n] for a in summ} for n in names},
             run.f("figures", "augmentation_per_class_recall.png"),
             "Per-class recall on real test set (mean over seeds)", "recall")
    base = summ["real_only"]["balanced_accuracy_mean"]
    return {"task": "augmentation",
            "classifier": (f"SmallCNN, {steps} optimizer steps per arm (= {CLS_EPOCHS} epochs of the real data)"
                           if ds.modality == "image" else "HistGradientBoosting on shared embedding"),
            "seeds": N_SEEDS, "n_test_per_class": {n: int((te.y == i).sum()) for i, n in enumerate(names)},
            "arms": summ,
            "delta_balanced_accuracy_vs_real_only": {a: summ[a]["balanced_accuracy_mean"] - base
                                                    for a in summ if a != "real_only"}}


def _synthetic_only(run, ds, fake, y_fake) -> dict:
    emb = run.embedder()
    tr, te = ds.splits["train"], ds.splits["test"]
    data = {"real_only": (tr.X, tr.y, False), "synthetic_only": (fake, y_fake, False)}
    reports, preds, steps = _run_arms(ds, emb, data, te)
    summ = _summarize(reports, ds.class_names)
    boot = _bootstrap(te.y, preds, ds.class_names, run.info["seed"])
    for a in summ:
        summ[a].update(boot[a])
    FIG.bars({n: {a: summ[a]["per_class_recall_mean"][n] for a in summ} for n in ds.class_names},
             run.f("figures", "synthetic_only_per_class_recall.png"), "Per-class recall: real vs synthetic-only training")
    return {"task": "synthetic_balanced", "seeds": N_SEEDS,
            "n_test_per_class": {n: int((te.y == i).sum()) for i, n in enumerate(ds.class_names)}, "arms": summ}


def _representation(run, ds, model) -> dict:
    from .metrics import latent_metrics

    lat = {s: model.encode(sp.X) for s, sp in ds.splits.items()}
    p = run.f("synthetic", "latent_embeddings.npz")
    np.savez_compressed(p, **lat)
    te = ds.splits["test"]
    lab = te.y if te.y is not None else te.eval_labels
    names = ds.class_names if te.y is not None else ds.eval_label_names
    res = {"task": "representation", "latent_file": p.relative_to(run.path).as_posix(), "latent_dim": int(lat["train"].shape[1])}
    if lab is not None:
        res["latent_metrics_test"] = latent_metrics(lat["test"], lab, run.info["seed"])
        # Larger split gives a more readable UMAP
        tr = ds.splits["train"]
        lab_tr = tr.y if tr.y is not None else tr.eval_labels
        FIG.latent_umap(lat["train"], lab_tr, names, run.f("figures", "latent_umap.png"),
                        "UMAP of learned latent space (train split)", run.info["seed"])
    return res


def _anomaly(run, ds, plan, model) -> dict:
    va, te = ds.splits["val"], ds.splits["test"]
    s_val, s_te = model.anomaly_score(va.X), model.anomaly_score(te.X)
    thr = float(np.quantile(s_val, 0.95))
    pd.DataFrame({"index": np.arange(len(s_te)), "anomaly_score": s_te, "flagged": s_te > thr}).to_csv(
        run.f("synthetic", "anomaly_scores_test.csv"), index=False)
    groups = {"validation": s_val, "test": s_te}
    res = {"task": "anomaly_detection", "threshold_val_q95": thr, "flagged_fraction_test": float((s_te > thr).mean())}
    holdout = (plan.get("anomaly") or {}).get("holdout_classes") or []
    if holdout:
        from sklearn.metrics import roc_auc_score

        hid = [ds.class_names.index(c) for c in holdout]
        is_anom = np.isin(te.y, hid)
        res["auroc_holdout_classes"] = float(roc_auc_score(is_anom, s_te))
        res["recall_at_threshold"] = float((s_te[is_anom] > thr).mean())
        groups = {"test normal": s_te[~is_anom], "test held-out classes": s_te[is_anom]}
    FIG.histograms(groups, run.f("figures", "anomaly_scores.png"), "Anomaly scores", "score (higher = more anomalous)")
    write_json(run.f("synthetic", "anomaly_summary.json"), res)
    return res
