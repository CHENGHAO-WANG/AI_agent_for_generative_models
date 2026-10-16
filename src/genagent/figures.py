"""Figures for the report (matplotlib, saved as PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def image_grid(X: np.ndarray, y: np.ndarray | None, class_names, path: Path, per_class: int = 10,
               title: str = "") -> None:
    """One row per class (or a plain grid when unlabeled)."""
    if y is None:
        rows = [X[i * per_class:(i + 1) * per_class] for i in range(min(8, len(X) // per_class))]
        labels = [""] * len(rows)
    else:
        rows, labels = [], []
        for c in np.unique(y):
            rows.append(X[y == c][:per_class])
            labels.append(class_names[c] if class_names else str(c))
    fig, axes = plt.subplots(len(rows), per_class, figsize=(per_class * 0.8, len(rows) * 0.85 + 0.4))
    axes = np.atleast_2d(axes)
    for r, (imgs, lab) in enumerate(zip(rows, labels)):
        for c in range(per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(imgs):
                img = imgs[c].transpose(1, 2, 0)
                ax.imshow(img.squeeze(), cmap="gray" if img.shape[2] == 1 else None, vmin=0, vmax=1)
        axes[r, 0].text(-2, imgs[0].shape[1] / 2 if len(imgs) else 0, lab[:18], ha="right", va="center", fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def umap_overlay(E_real: np.ndarray, E_fake: np.ndarray, path: Path, title: str = "", seed: int = 0,
                 labels_real: np.ndarray | None = None) -> None:
    """Joint UMAP of real and synthetic embeddings."""
    import umap

    rng = np.random.default_rng(seed)
    r = E_real[rng.permutation(len(E_real))[:3000]]
    f = E_fake[rng.permutation(len(E_fake))[:3000]]
    U = umap.UMAP(random_state=seed, n_neighbors=15).fit_transform(np.concatenate([r, f]))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    ax[0].scatter(U[:len(r), 0], U[:len(r), 1], s=3, c="#4C72B0", alpha=0.5, label="real")
    ax[0].scatter(U[len(r):, 0], U[len(r):, 1], s=3, c="#DD8452", alpha=0.5, label="synthetic")
    ax[0].legend(markerscale=4, fontsize=8)
    ax[0].set_title("real vs synthetic", fontsize=9)
    ax[1].scatter(U[len(r):, 0], U[len(r):, 1], s=3, c="#DD8452", alpha=0.5)
    ax[1].set_title("synthetic only", fontsize=9)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def latent_umap(Z: np.ndarray, labels: np.ndarray, names, path: Path, title: str, seed: int = 0) -> None:
    import umap

    U = umap.UMAP(random_state=seed).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for c in np.unique(labels):
        m = labels == c
        ax.scatter(U[m, 0], U[m, 1], s=4, label=names[c] if names else str(c), alpha=0.7)
    ax.legend(markerscale=3, fontsize=7, bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def bars(values: dict[str, dict[str, float]], path: Path, title: str, ylabel: str = "") -> None:
    """Grouped bars: outer keys = groups (e.g. classes), inner keys = series (e.g. real vs augmented)."""
    groups = list(values)
    series = list(next(iter(values.values())))
    x = np.arange(len(groups))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(max(5, len(groups) * 0.7), 3.5))
    for i, s in enumerate(series):
        ax.bar(x + i * w - 0.4 + w / 2, [values[g][s] for g in groups], w, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels([g[:14] for g in groups], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def histograms(groups: dict[str, np.ndarray], path: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for k, v in groups.items():
        ax.hist(v, bins=50, alpha=0.5, density=True, label=k)
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def tabular_marginals(real, fake, col_types: dict, path: Path, title: str, max_cols: int = 8) -> None:
    cols = list(col_types)[:max_cols]
    fig, axes = plt.subplots(2, (len(cols) + 1) // 2, figsize=(3 * ((len(cols) + 1) // 2), 5))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        if col_types[c] == "numeric":
            ax.hist(real[c].dropna(), bins=30, alpha=0.5, density=True, label="real")
            ax.hist(fake[c].dropna(), bins=30, alpha=0.5, density=True, label="synthetic")
        else:
            r = real[c].astype(str).value_counts(normalize=True)
            f = fake[c].astype(str).value_counts(normalize=True).reindex(r.index).fillna(0)
            x = np.arange(len(r))[:10]
            ax.bar(x - 0.2, r.values[:10], 0.4, label="real")
            ax.bar(x + 0.2, f.values[:10], 0.4, label="synthetic")
            ax.set_xticks(x)
            ax.set_xticklabels([str(i)[:8] for i in r.index[:10]], rotation=45, fontsize=6)
        ax.set_title(c[:20], fontsize=8)
    for ax in axes[len(cols):]:
        ax.axis("off")
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
