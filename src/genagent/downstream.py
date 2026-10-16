"""Objective-specific downstream analysis of the selected model and its synthetic data.

- class_balancing (augmentation): classifier trained on real vs real + synthetic vs a non-generative
  alternative (class-weighted loss), all tested on the real test set over several seeds
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


def run_downstream(run, ds, plan, model, fake, y_fake) -> dict:
    otype = plan["objective"]["type"]
    if otype == "class_balancing":
        if plan["objective"].get("mode") == "synthetic_balanced":
            return {"downstream": _synthetic_only(run, ds, fake, y_fake)}
        return {"downstream": _augmentation(run, ds, fake, y_fake)}
    if otype == "representation":
        return {"downstream": _representation(run, ds, model)}
    if otype == "anomaly_detection":
        return {"downstream": _anomaly(run, ds, plan, model)}
    return {"downstream": {"note": "generation objective: quality is summarized by the evaluation metrics"}}


# ---------------------------------------------------------------- classifiers

def _fit_predict(ds, emb, Xtr, ytr, Xte, seed, class_weight=False):
    """Image: SmallCNN; other modalities: gradient boosting on the shared embedding."""
    k = len(ds.class_names)
    if ds.modality == "image":
        net = train_classifier(Xtr, ytr, k, epochs=5, seed=seed, class_weight=class_weight)
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


def _augmentation(run, ds, fake, y_fake) -> dict:
    emb = run.embedder()
    tr, te = ds.splits["train"], ds.splits["test"]
    names = ds.class_names
    arms = {"real_only": [], "real_class_weighted": [], "real_plus_synthetic": []}
    has_syn = fake is not None and len(y_fake) > 0
    for s in range(N_SEEDS):
        arms["real_only"].append(_report(te.y, _fit_predict(ds, emb, tr.X, tr.y, te.X, s), names))
        arms["real_class_weighted"].append(_report(te.y, _fit_predict(ds, emb, tr.X, tr.y, te.X, s, True), names))
        if has_syn:
            Xa, ya = _concat(tr.X, fake), np.concatenate([tr.y, y_fake])
            arms["real_plus_synthetic"].append(_report(te.y, _fit_predict(ds, emb, Xa, ya, te.X, s), names))
    arms = {k: v for k, v in arms.items() if v}
    summ = _summarize(arms, names)
    FIG.bars({n: {a: summ[a]["per_class_recall_mean"][n] for a in summ} for n in names},
             run.f("figures", "augmentation_per_class_recall.png"),
             "Per-class recall on real test set (mean over seeds)", "recall")
    base = summ["real_only"]["balanced_accuracy_mean"]
    res = {"task": "augmentation", "classifier": "SmallCNN (5 epochs)" if ds.modality == "image"
           else "HistGradientBoosting on shared embedding", "seeds": N_SEEDS, "arms": summ}
    if "real_plus_synthetic" in summ:
        res["delta_balanced_accuracy_vs_real"] = summ["real_plus_synthetic"]["balanced_accuracy_mean"] - base
    res["delta_balanced_accuracy_class_weighted_vs_real"] = summ["real_class_weighted"]["balanced_accuracy_mean"] - base
    return res


def _synthetic_only(run, ds, fake, y_fake) -> dict:
    emb = run.embedder()
    tr, te = ds.splits["train"], ds.splits["test"]
    arms = {"real_only": [], "synthetic_only": []}
    for s in range(N_SEEDS):
        arms["real_only"].append(_report(te.y, _fit_predict(ds, emb, tr.X, tr.y, te.X, s), ds.class_names))
        arms["synthetic_only"].append(_report(te.y, _fit_predict(ds, emb, fake, y_fake, te.X, s), ds.class_names))
    summ = _summarize(arms, ds.class_names)
    FIG.bars({n: {a: summ[a]["per_class_recall_mean"][n] for a in summ} for n in ds.class_names},
             run.f("figures", "synthetic_only_per_class_recall.png"), "Per-class recall: real vs synthetic-only training")
    return {"task": "synthetic_balanced", "seeds": N_SEEDS, "arms": summ}


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
