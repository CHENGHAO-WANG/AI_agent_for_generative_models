"""Evaluation metrics.

General layer (any modality), computed on the shared embedding:
    precision / recall / density / coverage (Naeem et al. 2020), MMD (RBF), C2ST AUC,
    DCR memorization check, TSTR (when labeled).
Specific layer, used when the modality matches:
    image   - FID in classifier/AE feature space, class consistency of conditional samples
    scrna   - gene mean / variance / zero-fraction correlations, Leiden cluster proportions,
              marker-gene expression agreement
    tabular - sdmetrics quality report (column shapes, column pair trends)
`score_candidates` maps metrics to [0, 1] "goodness" and combines them with weights that depend on
the objective and modality.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, pairwise_distances, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- general layer


def _sub(A, n, rng):
    return A if len(A) <= n else A[rng.choice(len(A), n, replace=False)]


def prdc(real: np.ndarray, fake: np.ndarray, k: int = 5) -> dict:
    """Precision/recall (manifold overlap) and density/coverage (Naeem et al. 2020)."""
    nn_r = NearestNeighbors(n_neighbors=k + 1).fit(real)
    r_rad = nn_r.kneighbors(real)[0][:, -1]          # k-NN radius of each real point
    nn_f = NearestNeighbors(n_neighbors=k + 1).fit(fake)
    f_rad = nn_f.kneighbors(fake)[0][:, -1]
    d_fr = pairwise_distances(fake, real)
    inside = d_fr < r_rad[None, :]                   # fake i inside real ball j
    precision = inside.any(1).mean()
    density = inside.sum(1).mean() / k
    recall = (d_fr.T < f_rad[None, :]).any(1).mean()  # real j inside some fake ball
    coverage = (d_fr.min(0) < r_rad).mean()
    return {"precision": float(precision), "recall": float(recall), "density": float(density),
            "coverage": float(coverage)}


def mmd_rbf(a: np.ndarray, b: np.ndarray, bandwidth: float | None = None) -> float:
    """Unbiased squared MMD with an RBF kernel (median heuristic bandwidth)."""
    ab = np.concatenate([a, b])
    d2 = pairwise_distances(ab, metric="sqeuclidean")
    if bandwidth is None:
        bandwidth = np.sqrt(np.median(d2[d2 > 0]) / 2)
    K = np.exp(-d2 / (2 * bandwidth ** 2))
    n, m = len(a), len(b)
    kxx = (K[:n, :n].sum() - np.trace(K[:n, :n])) / (n * (n - 1))
    kyy = (K[n:, n:].sum() - np.trace(K[n:, n:])) / (m * (m - 1))
    kxy = K[:n, n:].mean()
    return float(kxx + kyy - 2 * kxy)


def median_bandwidth(a: np.ndarray, rng) -> float:
    s = _sub(a, 1000, rng)
    d2 = pairwise_distances(s, metric="sqeuclidean")
    return float(np.sqrt(np.median(d2[d2 > 0]) / 2))


def c2st_auc(real: np.ndarray, fake: np.ndarray, seed: int = 0) -> float:
    """Classifier two-sample test: cross-validated AUC of real-vs-synthetic (0.5 = indistinguishable)."""
    X = np.concatenate([real, fake])
    y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
    clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def dcr(train: np.ndarray, holdout: np.ndarray, fake: np.ndarray) -> dict:
    """Distance to closest training record for synthetic vs real holdout samples.

    If synthetic samples are much closer to training points than unseen real data is, the model is
    reproducing training records. Flag when > 3x the expected 5% fall under the holdout's 5th percentile
    or when > 1% are (near-)exact copies."""
    nn = NearestNeighbors(n_neighbors=1).fit(train)
    d_fake = nn.kneighbors(fake)[0][:, 0]
    d_hold = nn.kneighbors(holdout)[0][:, 0]
    q5 = np.quantile(d_hold, 0.05)
    frac_close = float((d_fake < q5).mean())
    exact = float((d_fake < 1e-6 * max(np.median(d_hold), 1e-12)).mean())
    return {"dcr_median_synthetic": float(np.median(d_fake)), "dcr_median_holdout": float(np.median(d_hold)),
            "dcr_ratio": float(np.median(d_fake) / max(np.median(d_hold), 1e-12)),
            "dcr_frac_below_holdout_q05": frac_close, "dcr_exact_copy_frac": exact,
            "memorization_flag": bool(frac_close > 0.15 or exact > 0.01)}


def per_class_checks(E_train, y_train, E_holdout, y_holdout, E_fake, y_fake, class_names,
                     classifier=None, fake_X=None, min_n: int = 20) -> dict:
    """Class consistency and memorization for each class separately, on a class-balanced probe sample.

    Consistency: a classifier trained on real data (the image classifier, else 15-NN in the shared
    embedding) predicts the requested class. Memorization: the share of a class's synthetic samples closer
    to a training sample than the 5th percentile of that class's real held-out samples (pooled held-out
    reference when the class has fewer than `min_n` held-out samples)."""
    if classifier is not None and fake_X is not None:
        from .embed import predict_proba

        pred = predict_proba(classifier, fake_X).argmax(1)
    else:
        pred = KNeighborsClassifier(15).fit(E_train, y_train).predict(E_fake)
    nn = NearestNeighbors(n_neighbors=1).fit(E_train)
    d_fake = nn.kneighbors(E_fake)[0][:, 0]
    d_hold = nn.kneighbors(E_holdout)[0][:, 0]
    q5_pooled = np.quantile(d_hold, 0.05)
    tiny = 1e-6 * max(np.median(d_hold), 1e-12)
    per, flagged = {}, []
    for c, name in enumerate(class_names):
        f = y_fake == c
        if not f.any():
            continue
        h = y_holdout == c
        own = h.sum() >= min_n
        q5 = np.quantile(d_hold[h], 0.05) if own else q5_pooled
        close = float((d_fake[f] < q5).mean())
        exact = float((d_fake[f] < tiny).mean())
        flag = bool(f.sum() >= min_n and (close > 0.15 or exact > 0.01))
        per[name] = {"n_train": int((y_train == c).sum()), "n_heldout": int(h.sum()),
                     "class_consistency": float((pred[f] == c).mean()),
                     "dcr_frac_below_q05": close, "dcr_exact_copy_frac": exact,
                     "dcr_reference": "class" if own else "pooled", "memorization_flag": flag}
        if flag:
            flagged.append(name)
    return {"per_class": per,
            "class_consistency_balanced": float(np.mean([v["class_consistency"] for v in per.values()])),
            "class_consistency_min": float(min(v["class_consistency"] for v in per.values())),
            "memorization_flag_any_class": bool(flagged), "memorization_flagged_classes": flagged}


def tstr(fake_E, fake_y, real_tr_E, real_tr_y, test_E, test_y, seed: int = 0) -> dict:
    """Train-on-synthetic / test-on-real with a linear probe on embeddings; TRTR baseline uses the
    same number of real training samples."""
    def fit_eval(X, y):
        if len(np.unique(y)) < 2:
            return float("nan")
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        clf.fit(X, y)
        return float(balanced_accuracy_score(test_y, clf.predict(test_E)))

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(real_tr_E), min(len(fake_E), len(real_tr_E)), replace=False)
    t = fit_eval(fake_E, fake_y)
    r = fit_eval(real_tr_E[idx], real_tr_y[idx])
    return {"tstr_balanced_acc": t, "trtr_balanced_acc": r, "tstr_ratio": float(t / r) if r > 0 else float("nan")}


def general_metrics(E_train, E_test, E_fake, y_train=None, y_test=None, y_fake=None, seed=0,
                    n_max: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    r = _sub(E_test, n_max, rng)
    f = _sub(E_fake, n_max, rng)
    out = prdc(r, f)
    bw = median_bandwidth(E_train, rng)
    out["mmd"] = mmd_rbf(r, f, bw)
    out["mmd_baseline_real"] = mmd_rbf(r, _sub(E_train, len(r), rng), bw)  # real-vs-real reference
    out["c2st_auc"] = c2st_auc(r, f, seed)
    out.update(dcr(_sub(E_train, 20000, rng), r, f))
    if y_fake is not None and y_test is not None:
        out.update(tstr(E_fake, y_fake, E_train, y_train, E_test, y_test, seed))
    return out


# ---------------------------------------------------------------- specific layer


def fid(a: np.ndarray, b: np.ndarray) -> float:
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def image_metrics(embedder, E_train, E_test, fake_X, E_fake, y_fake, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    out = {"fid_feature_space": fid(E_test, E_fake),
           "fid_baseline_real": fid(E_test, _sub(E_train, len(E_test), rng))}
    if embedder.classifier is not None and y_fake is not None:
        from .embed import predict_proba

        pred = predict_proba(embedder.classifier, fake_X).argmax(1)
        out["class_consistency_acc"] = float((pred == y_fake).mean())
        out["class_consistency_per_class"] = {int(c): float((pred[y_fake == c] == c).mean())
                                              for c in np.unique(y_fake)}
    return out


def _marker_genes(L: np.ndarray, labels: np.ndarray, k: int = 5) -> dict[int, np.ndarray]:
    """Top-k genes by mean log-expression difference between each cluster and the rest."""
    out = {}
    for c in np.unique(labels):
        diff = L[labels == c].mean(0) - L[labels != c].mean(0)
        out[int(c)] = np.argsort(-diff)[:k]
    return out


def scrna_metrics(ds, embedder, fake_X, E_fake) -> dict:
    from .embed import scrna_lognorm

    tr, te = ds.splits["train"], ds.splits["test"]
    Lr, Lf = scrna_lognorm(te.X), scrna_lognorm(fake_X)
    out = {
        "gene_mean_corr": float(np.corrcoef(Lr.mean(0), Lf.mean(0))[0, 1]),
        "gene_var_corr": float(np.corrcoef(Lr.var(0), Lf.var(0))[0, 1]),
        "gene_zero_frac_corr": float(np.corrcoef((te.X == 0).mean(0), (fake_X == 0).mean(0))[0, 1]),
        "libsize_median_real": float(np.median(te.X.sum(1))),
        "libsize_median_synthetic": float(np.median(fake_X.sum(1))),
    }
    if tr.eval_labels is not None:
        # Assign synthetic cells to the reference groups (cell types or clusters) by 15-NN in the shared
        # embedding, then compare group proportions with the training split (largest real sample).
        knn = KNeighborsClassifier(15).fit(embedder.transform(tr.X), tr.eval_labels)
        fl = knn.predict(E_fake)
        k = len(ds.eval_label_names)
        p_train = np.bincount(tr.eval_labels, minlength=k) / len(tr.eval_labels)
        pf = np.bincount(fl, minlength=k) / len(fl)
        # Real-vs-real baseline: held-out real cells assigned the same way. Its TVD to the training
        # proportions is the error of a perfect generator (sampling noise + assignment bias).
        ho = np.concatenate([ds.splits["val"].X, te.X])
        p_ho = np.bincount(knn.predict(embedder.transform(ho)), minlength=k) / len(ho)
        names_g = ds.eval_label_names
        out["cluster_props_reference_train"] = {n: round(float(v), 4) for n, v in zip(names_g, p_train)}
        out["cluster_props_synthetic"] = {n: round(float(v), 4) for n, v in zip(names_g, pf)}
        out["cluster_props_heldout_real_assigned"] = {n: round(float(v), 4) for n, v in zip(names_g, p_ho)}
        out["cluster_tvd"] = float(0.5 * np.abs(p_train - pf).sum())
        out["cluster_tvd_baseline_real"] = float(0.5 * np.abs(p_train - p_ho).sum())
        # Marker agreement: real group means from the training cells' own labels
        Ltr = scrna_lognorm(tr.X)
        markers = _marker_genes(Ltr, tr.eval_labels)
        genes = np.unique(np.concatenate(list(markers.values())))
        Mr = np.stack([Ltr[tr.eval_labels == c][:, genes].mean(0) if (tr.eval_labels == c).any()
                       else np.zeros(len(genes)) for c in range(k)])
        Mf = np.stack([Lf[fl == c][:, genes].mean(0) if (fl == c).any() else np.zeros(len(genes)) for c in range(k)])
        out["marker_expr_corr"] = float(np.corrcoef(Mr.ravel(), Mf.ravel())[0, 1])
        var_names = ds.meta.get("var_names")
        if var_names:
            out["marker_genes"] = {names_g[c]: [var_names[g] for g in gs] for c, gs in markers.items()}
    return out


def tabular_metrics(ds, ctx, fake_df) -> dict:
    from sdmetrics.reports.single_table import QualityReport

    real = ds.splits["test"].X
    meta = {"columns": {c: {"sdtype": "numerical" if t == "numeric" else "categorical"}
                        for c, t in ctx.col_types.items()}}
    real2, fake2 = real.copy(), fake_df.copy()
    for c, t in ctx.col_types.items():
        if t == "categorical":
            real2[c], fake2[c] = real2[c].astype(str), fake2[c].astype(str)
    rep = QualityReport()
    rep.generate(real2, fake2, meta, verbose=False)
    props = rep.get_properties().set_index("Property")["Score"].to_dict()
    return {"sdm_quality_score": float(rep.get_score()),
            "sdm_column_shapes": float(props.get("Column Shapes", np.nan)),
            "sdm_column_pair_trends": float(props.get("Column Pair Trends", np.nan))}


# ---------------------------------------------------------------- objective-specific


def latent_metrics(Z: np.ndarray, labels: np.ndarray, seed=0) -> dict:
    """Representation quality: kNN accuracy and k-means ARI of reference labels in latent space."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    k = len(np.unique(labels))
    knn = cross_val_predict(KNeighborsClassifier(15), Z, labels, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    km = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
    return {"latent_knn_acc": float(balanced_accuracy_score(labels, knn)),
            "latent_kmeans_ari": float(adjusted_rand_score(labels, km)),
            "latent_silhouette": float(silhouette_score(Z, labels)) if k > 1 else float("nan"),
            "latent_dim": int(Z.shape[1])}


# ---------------------------------------------------------------- scoring

GENERAL = ["precision", "recall", "density", "coverage", "c2st", "mmd", "tstr"]
SPECIFIC = {
    "image": ["fid"],
    "scrna": ["gene_mean_corr", "gene_var_corr", "gene_zero_frac_corr", "cluster_tvd", "marker_expr_corr"],
    "tabular": ["sdm_column_shapes", "sdm_column_pair_trends"],
}
OBJECTIVE_WEIGHTS = {
    "generation": {"precision": 1, "recall": 1, "density": 0.5, "coverage": 1, "c2st": 1, "mmd": 1, "tstr": 0.5,
                   "class_consistency": 1, "_specific": 1},
    "class_balancing": {"precision": 1, "recall": 0.5, "density": 0.5, "coverage": 0.5, "c2st": 0.5, "mmd": 0.5,
                        "tstr": 2, "class_consistency": 2, "_specific": 1},
    "representation": {"precision": 0.5, "recall": 0.5, "density": 0.25, "coverage": 0.5, "c2st": 0.5,
                       "mmd": 0.5, "tstr": 0.25, "latent_knn_acc": 3, "latent_kmeans_ari": 3, "_specific": 0.5},
    "anomaly_detection": {"precision": 0.5, "recall": 0.5, "density": 0.25, "coverage": 0.5, "c2st": 0.5,
                          "mmd": 0.5, "anomaly_auroc": 4, "_specific": 0.5},
}
SPECIFIC_MULTIPLIER = 2.0  # modality-specific metrics weigh more when the modality matches


def default_weights(objective: str, modality: str) -> dict:
    base = dict(OBJECTIVE_WEIGHTS.get(objective, OBJECTIVE_WEIGHTS["generation"]))
    spec = base.pop("_specific")
    for m in SPECIFIC[modality]:
        base[m] = base.get(m, spec) * SPECIFIC_MULTIPLIER
    return base


def goodness(m: dict) -> dict:
    """Map raw metrics to [0, 1], higher = better. Distances are scored relative to the real-vs-real baseline."""
    g = {}

    def put(k, v):
        if v is not None and np.isfinite(v):
            g[k] = float(np.clip(v, 0, 1))

    for k in ("precision", "recall", "coverage"):
        put(k, m.get(k))
    if "density" in m:
        put("density", min(m["density"], 1.0))
    if "c2st_auc" in m:
        put("c2st", 1 - 2 * abs(m["c2st_auc"] - 0.5))
    if "mmd" in m:
        b = max(m.get("mmd_baseline_real", 0.0), 1e-4)
        put("mmd", b / max(m["mmd"], b))
    put("tstr", m.get("tstr_ratio"))
    if "fid_feature_space" in m:
        b = max(m["fid_baseline_real"], 1e-6)
        put("fid", b / max(m["fid_feature_space"], b))
    # Class-balanced consistency (per-class probe) when available, so rare classes count equally
    put("class_consistency", m.get("class_consistency_balanced", m.get("class_consistency_acc")))
    for k in ("gene_mean_corr", "gene_var_corr", "gene_zero_frac_corr", "marker_expr_corr"):
        put(k, m.get(k))
    if "cluster_tvd" in m:
        # relative to held-out real cells assigned the same way (floored: tiny baselines are noise)
        b = max(m.get("cluster_tvd_baseline_real", 0.0), 0.02)
        put("cluster_tvd", b / max(m["cluster_tvd"], b))
    for k in ("sdm_column_shapes", "sdm_column_pair_trends", "latent_knn_acc", "anomaly_auroc"):
        put(k, m.get(k))
    put("latent_kmeans_ari", m.get("latent_kmeans_ari"))
    return g


def score_candidates(metrics: dict[str, dict], weights: dict) -> dict:
    """Weighted mean goodness over the metrics each candidate has; memorizing candidates are excluded."""
    out = {}
    for name, m in metrics.items():
        g = goodness(m)
        used = {k: w for k, w in weights.items() if k in g and w > 0}
        tot = sum(used.values())
        score = sum(g[k] * w for k, w in used.items()) / tot if tot else float("nan")
        excluded = bool(m.get("memorization_flag", False))
        out[name] = {"score": round(score, 4), "goodness": {k: round(v, 4) for k, v in g.items()},
                     "weights_used": used, "excluded_memorization": excluded}
    eligible = {k: v for k, v in out.items() if not v["excluded_memorization"] and np.isfinite(v["score"])}
    ranking = sorted(eligible, key=lambda k: -eligible[k]["score"])
    return {"per_candidate": out, "ranking": ranking, "best": ranking[0] if ranking else None}
