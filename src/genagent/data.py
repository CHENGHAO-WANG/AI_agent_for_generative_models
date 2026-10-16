"""Dataset loading, splitting, and profiling.

Every input is normalized into a `Dataset` with one of three modalities:
- "image":   X is float32 (N, C, H, W) in [0, 1]
- "scrna":   X is float32 (N, G) raw counts on highly variable genes
- "tabular": X is a pandas DataFrame of features (label column removed)
Labels, when present, are integer codes in `y` with names in `class_names`.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_CACHE = Path("data_cache")
BUILTIN = ("pathmnist", "pbmc3k")
LABEL_CANDIDATES = ("label", "labels", "y", "target", "class", "cell_type", "celltype")


@dataclass
class Split:
    X: Any
    y: np.ndarray | None = None
    # Labels used only for evaluation (e.g. Leiden clusters of an unlabeled scRNA dataset)
    eval_labels: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.X)


@dataclass
class Dataset:
    name: str
    modality: str
    splits: dict[str, Split]
    class_names: list[str] | None = None
    label_name: str | None = None
    eval_label_names: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def labeled(self) -> bool:
        return self.class_names is not None

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "Dataset":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------- loading

def load_dataset(source: str, label: str | None = None, seed: int = 0) -> Dataset:
    """Load a built-in dataset name or a path. `label`: None = auto-detect, "none" = unlabeled."""
    DATA_CACHE.mkdir(exist_ok=True)
    if source.lower() == "pathmnist":
        ds = _load_medmnist("pathmnist", label)
    elif source.lower() == "pbmc3k":
        ds = _load_pbmc3k(label, seed)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {source}")
        if path.is_dir():
            ds = _load_image_folder(path, label, seed)
        elif path.suffix == ".npz":
            ds = _load_npz(path, label, seed)
        elif path.suffix == ".h5ad":
            ds = _load_h5ad(path, label, seed)
        elif path.suffix in (".csv", ".parquet", ".tsv"):
            ds = _load_table(path, label, seed)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    return ds


def _encode_labels(y_all: np.ndarray) -> tuple[np.ndarray, list[str]]:
    classes, codes = np.unique(np.asarray(y_all).astype(str), return_inverse=True)
    return codes.astype(np.int64), [str(c) for c in classes]


def split_indices(n: int, y: np.ndarray | None, seed: int) -> dict[str, np.ndarray]:
    """Seeded 80/10/10 split, stratified when labels are available and every class is large enough."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(n)
    strat = y if y is not None and np.bincount(y).min() >= 10 else None
    tr, rest = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)
    strat_rest = y[rest] if strat is not None else None
    va, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=strat_rest)
    return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}


def _resolve_label(available: list[str], label: str | None, meta: dict) -> str | None:
    """Pick the label key. Records ambiguity in `meta` so the agent can ask the user."""
    if label is not None:
        if label.lower() == "none":
            return None
        if label not in available:
            raise ValueError(f"Label '{label}' not found; available: {available}")
        return label
    found = [k for k in available if k.lower() in LABEL_CANDIDATES]
    meta["label_candidates"] = found
    meta["label_autodetected"] = True
    if len(found) > 1:
        meta["label_ambiguous"] = True
    return found[0] if found else None


def _load_medmnist(flag: str, label: str | None) -> Dataset:
    import medmnist
    from medmnist import INFO

    info = INFO[flag]
    cls = getattr(medmnist, info["python_class"])
    splits, ys = {}, {}
    for s in ("train", "val", "test"):
        d = cls(split=s, download=True, size=28, root=str(DATA_CACHE))
        imgs = d.imgs if d.imgs.ndim == 4 else d.imgs[..., None]
        splits[s] = np.ascontiguousarray(imgs.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
        ys[s] = d.labels.reshape(-1).astype(np.int64)
    use_labels = label is None or label.lower() != "none"
    names = [info["label"][str(i)] for i in range(len(info["label"]))]
    return Dataset(
        name=flag,
        modality="image",
        splits={s: Split(splits[s], ys[s] if use_labels else None) for s in splits},
        class_names=names if use_labels else None,
        label_name="label" if use_labels else None,
        meta={"source": f"medmnist:{flag}", "official_splits": True, "description": info["description"]},
    )


def _load_pbmc3k(label: str | None, seed: int) -> Dataset:
    import scanpy as sc

    sc.settings.datasetdir = DATA_CACHE
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()
    return _scrna_from_adata(adata, "pbmc3k", label if label is not None else "none", seed, qc=True)


def _load_h5ad(path: Path, label: str | None, seed: int) -> Dataset:
    import anndata as ad

    adata = ad.read_h5ad(path)
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"]
    elif adata.raw is not None:
        adata = adata.raw.to_adata()
    X = adata.X[:200]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    if not (np.all(X >= 0) and np.allclose(X, np.round(X))):
        # Not count data: treat as a numeric table instead
        df = adata.to_df()
        for c in adata.obs.columns:
            df[c] = adata.obs[c].values
        return _table_dataset(df, path.stem, label, seed, source=str(path))
    return _scrna_from_adata(adata, path.stem, label, seed, qc=False)


def _scrna_from_adata(adata, name: str, label: str | None, seed: int, qc: bool) -> Dataset:
    """Standard scanpy QC, 2000 HVGs selected on log-normalized data, raw counts kept."""
    import scanpy as sc

    meta: dict[str, Any] = {"source": name, "official_splits": False}
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    if qc:
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        adata = adata[(adata.obs.n_genes_by_counts < 2500) & (adata.obs.pct_counts_mt < 5)].copy()
    meta["n_cells_after_qc"] = int(adata.n_obs)
    meta["n_genes_after_qc"] = int(adata.n_vars)
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    n_hvg = min(2000, adata.n_vars)
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_hvg)
    adata = adata[:, tmp.var.highly_variable.values].copy()
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float32)

    lab = _resolve_label(list(adata.obs.columns), label, meta)
    y, class_names = (None, None)
    if lab is not None:
        y, class_names = _encode_labels(adata.obs[lab].values)
    idx = split_indices(len(X), y, seed)
    splits = {s: Split(X[i], y[i] if y is not None else None) for s, i in idx.items()}
    meta["var_names"] = list(adata.var_names)
    ds = Dataset(name, "scrna", splits, class_names, lab, meta=meta)
    _add_scrna_eval_clusters(ds, seed)
    return ds


def _add_scrna_eval_clusters(ds: Dataset, seed: int) -> None:
    """Leiden clusters on the training split (standard pipeline); other splits assigned by kNN.

    These clusters are evaluation-only pseudo-labels (cluster proportions, marker genes)."""
    import anndata as ad
    import scanpy as sc
    from sklearn.neighbors import KNeighborsClassifier

    from .embed import scrna_lognorm

    tr = ds.splits["train"].X
    a = ad.AnnData(scrna_lognorm(tr))
    sc.pp.pca(a, n_comps=min(50, a.n_vars - 1), random_state=seed)
    sc.pp.neighbors(a, random_state=seed)
    sc.tl.leiden(a, resolution=0.5, random_state=seed, flavor="igraph", n_iterations=2, directed=False)
    lab = a.obs["leiden"].astype(int).values
    ds.splits["train"].eval_labels = lab
    # Assign other splits via kNN in PCA space fit on train
    pcs = a.varm["PCs"]
    mu = a.X.mean(0)
    knn = KNeighborsClassifier(15).fit(a.obsm["X_pca"], lab)
    for s in ("val", "test"):
        proj = (scrna_lognorm(ds.splits[s].X) - mu) @ pcs
        ds.splits[s].eval_labels = knn.predict(proj)
    ds.eval_label_names = [f"leiden_{i}" for i in range(lab.max() + 1)]


def _load_npz(path: Path, label: str | None, seed: int) -> Dataset:
    z = np.load(path, allow_pickle=False)
    keys = list(z.keys())
    meta: dict[str, Any] = {"source": str(path)}
    if all(f"{s}_images" in keys for s in ("train", "val", "test")):
        # MedMNIST-style file with official splits
        meta["official_splits"] = True
        use_labels = label is None or label.lower() != "none"
        splits = {}
        for s in ("train", "val", "test"):
            img = z[f"{s}_images"]
            y = z[f"{s}_labels"].reshape(-1).astype(np.int64) if use_labels and f"{s}_labels" in keys else None
            splits[s] = Split(_to_nchw(img), y)
        names = None
        if use_labels:
            names = [str(i) for i in range(int(max(sp.y.max() for sp in splits.values())) + 1)]
        return Dataset(path.stem, "image", splits, names, "label" if names else None, meta=meta)
    data_key = max((k for k in keys), key=lambda k: z[k].size)
    lab = _resolve_label([k for k in keys if k != data_key], label, meta)
    X = z[data_key]
    y, names = (None, None)
    if lab is not None:
        y, names = _encode_labels(z[lab].reshape(-1))
    meta["official_splits"] = False
    if X.ndim >= 3:
        X = _to_nchw(X)
        idx = split_indices(len(X), y, seed)
        return Dataset(path.stem, "image", {s: Split(X[i], y[i] if y is not None else None) for s, i in idx.items()},
                       names, lab, meta=meta)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if lab is not None:
        df["__label__"] = np.asarray(names)[y]
    return _table_dataset(df, path.stem, "__label__" if lab else "none", seed, source=str(path))


def _to_nchw(img: np.ndarray) -> np.ndarray:
    """Convert image arrays to float (N, C, H, W) in [0, 1]."""
    if img.ndim == 3:
        img = img[:, None]
    elif img.shape[-1] in (1, 3) and img.shape[1] not in (1, 3):
        img = img.transpose(0, 3, 1, 2)
    img = np.ascontiguousarray(img, dtype=np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return img


def _load_image_folder(path: Path, label: str | None, seed: int) -> Dataset:
    """Folder of images; one subfolder per class gives labels."""
    from PIL import Image

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in exts)
    if not files:
        raise ValueError(f"No images found under {path}")
    sizes = [Image.open(f).size for f in files[:200]]
    side = int(np.median([min(s) for s in sizes]))
    side = max(16, min(64, side)) // 4 * 4  # UNet needs sides divisible by 4
    mode = "RGB" if any(Image.open(f).mode not in ("L", "1") for f in files[:50]) else "L"
    X = np.stack([np.asarray(Image.open(f).convert(mode).resize((side, side))) for f in files])
    use_labels = label is None or label.lower() != "none"
    parents = [f.parent.name for f in files]
    y, names = (None, None)
    if use_labels and len(set(parents)) > 1:
        y, names = _encode_labels(np.array(parents))
    X = _to_nchw(X)
    idx = split_indices(len(X), y, seed)
    meta = {"source": str(path), "official_splits": False, "resized_to": side}
    return Dataset(path.name, "image", {s: Split(X[i], y[i] if y is not None else None) for s, i in idx.items()},
                   names, "folder" if names else None, meta=meta)


def _load_table(path: Path, label: str | None, seed: int) -> Dataset:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    return _table_dataset(df, path.stem, label, seed, source=str(path))


def _table_dataset(df: pd.DataFrame, name: str, label: str | None, seed: int, source: str) -> Dataset:
    meta: dict[str, Any] = {"source": source, "official_splits": False}
    lab = _resolve_label(list(df.columns), label, meta)
    y, names = (None, None)
    if lab is not None:
        y, names = _encode_labels(df[lab].values)
        df = df.drop(columns=[lab])
    # Drop identifier-like columns (unique strings) which cannot be modeled
    for c in list(df.columns):
        if df[c].dtype == object and df[c].nunique() > 0.9 * len(df):
            df = df.drop(columns=[c])
            meta.setdefault("dropped_id_columns", []).append(c)
    idx = split_indices(len(df), y, seed)
    splits = {s: Split(df.iloc[i].reset_index(drop=True), y[i] if y is not None else None) for s, i in idx.items()}
    return Dataset(name, "tabular", splits, names, lab, meta=meta)


# ---------------------------------------------------------------- profiling

def tabular_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Categorical if non-numeric or a numeric column with few distinct integer values."""
    types = {}
    for c in df.columns:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            types[c] = "categorical"
        else:
            nn = s.dropna()
            types[c] = "categorical" if (nn.nunique() <= 10 and np.allclose(nn, np.round(nn))) else "numeric"
    return types


def profile_dataset(ds: Dataset) -> dict[str, Any]:
    tr = ds.splits["train"]
    p: dict[str, Any] = {
        "name": ds.name,
        "modality": ds.modality,
        "split_sizes": {s: len(sp) for s, sp in ds.splits.items()},
        "official_splits": ds.meta.get("official_splits", False),
        "labeled": ds.labeled,
        "label_name": ds.label_name,
    }
    for k in ("label_candidates", "label_ambiguous", "label_autodetected", "dropped_id_columns",
              "n_cells_after_qc", "n_genes_after_qc", "description", "resized_to"):
        if k in ds.meta:
            p[k] = ds.meta[k]
    if ds.labeled:
        counts = np.bincount(tr.y, minlength=len(ds.class_names))
        p["n_classes"] = len(ds.class_names)
        p["class_counts_train"] = {n: int(c) for n, c in zip(ds.class_names, counts)}
        p["imbalance_ratio_max_over_min"] = round(float(counts.max() / max(counts.min(), 1)), 3)
        p["minority_class"] = ds.class_names[int(counts.argmin())]
        p["majority_class_size"] = int(counts.max())
    if ds.modality == "image":
        X = tr.X
        p["image_shape_chw"] = list(X.shape[1:])
        p["pixel_mean_per_channel"] = [round(float(v), 4) for v in X.mean((0, 2, 3), dtype=np.float64)]
        p["pixel_std_per_channel"] = [round(float(v), 4) for v in X.std((0, 2, 3), dtype=np.float64)]
        p["n_features"] = int(np.prod(X.shape[1:]))
        p["missing_values"] = int(np.isnan(X).sum())
    elif ds.modality == "scrna":
        X = tr.X
        lib = X.sum(1)
        p["n_genes"] = int(X.shape[1])
        p["sparsity_fraction_zero"] = round(float((X == 0).mean()), 4)
        p["library_size"] = {"median": float(np.median(lib)), "min": float(lib.min()), "max": float(lib.max())}
        p["mean_count_per_entry"] = round(float(X.mean()), 4)
        p["eval_clusters_leiden"] = len(ds.eval_label_names or [])
        p["eval_cluster_counts_train"] = {n: int(c) for n, c in zip(
            ds.eval_label_names, np.bincount(tr.eval_labels, minlength=len(ds.eval_label_names)))}
        p["missing_values"] = int(np.isnan(X).sum())
    else:
        df = tr.X
        types = tabular_column_types(df)
        p["n_features"] = df.shape[1]
        p["column_types"] = types
        p["n_numeric"] = sum(t == "numeric" for t in types.values())
        p["n_categorical"] = sum(t == "categorical" for t in types.values())
        miss = df.isna().mean()
        p["missing_fraction_by_column"] = {c: round(float(v), 4) for c, v in miss.items() if v > 0}
        p["missing_values"] = int(df.isna().sum().sum())
    return p
