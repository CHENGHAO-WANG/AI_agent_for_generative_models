"""Unit tests for allocation, metrics, codec, and data loading."""

import numpy as np
import pandas as pd
import pytest

from genagent.allocation import AllocationError, allocate, labels_from_allocation, proportional
from genagent.data import load_dataset, profile_dataset, tabular_column_types
from genagent.embed import TabularCodec
from genagent import metrics as M

NAMES = ["a", "b", "c"]
COUNTS = np.array([100, 50, 10])


def test_proportional_sums_exactly():
    k = proportional(101, COUNTS)
    assert k.sum() == 101 and k[0] > k[1] > k[2]


def test_generation_allocation_proportional():
    a = allocate("generation", None, 160, NAMES, COUNTS)
    assert a["per_class"] == {"a": 100, "b": 50, "c": 10}


def test_augmentation_fills_to_target():
    a = allocate("class_balancing", "augmentation", 80, NAMES, COUNTS)
    assert a["per_class"] == {"a": 0, "b": 30, "c": 70}


def test_augmentation_exceeding_majority():
    with pytest.raises(AllocationError):
        allocate("class_balancing", "augmentation", 150, NAMES, COUNTS, "error")
    assert allocate("class_balancing", "augmentation", 150, NAMES, COUNTS, "clamp")["per_class"] == \
        {"a": 0, "b": 50, "c": 90}
    assert allocate("class_balancing", "augmentation", 150, NAMES, COUNTS, "keep")["per_class"] == \
        {"a": 50, "b": 100, "c": 140}


def test_synthetic_balanced_per_class():
    a = allocate("class_balancing", "synthetic_balanced", 20, NAMES, COUNTS)
    assert a["total"] == 60 and set(a["per_class"].values()) == {20}


def test_unlabeled_allocation_and_labels():
    a = allocate("generation", None, 50, None, None)
    assert a["per_class"] == {"all": 50}
    assert labels_from_allocation(a, None) is None
    y = labels_from_allocation(allocate("generation", None, 160, NAMES, COUNTS), NAMES)
    assert np.bincount(y).tolist() == [100, 50, 10]


def test_prdc_identical_vs_shifted():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(300, 5)), rng.normal(size=(300, 5))
    same = M.prdc(a, b)
    far = M.prdc(a, b + 10)
    assert same["precision"] > 0.7 and same["coverage"] > 0.7
    assert far["precision"] == 0 and far["coverage"] == 0


def test_mmd_and_c2st_detect_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(300, 4)), rng.normal(size=(300, 4))
    assert M.mmd_rbf(a, b) < M.mmd_rbf(a, b + 1)
    assert M.c2st_auc(a, b) < 0.65 < M.c2st_auc(a, b + 2)


def test_dcr_flags_copies():
    rng = np.random.default_rng(0)
    train, hold = rng.normal(size=(500, 3)), rng.normal(size=(200, 3))
    assert M.dcr(train, hold, train[:200])["memorization_flag"]
    assert not M.dcr(train, hold, rng.normal(size=(200, 3)))["memorization_flag"]


def test_scoring_excludes_memorizers():
    w = M.default_weights("generation", "tabular")
    m = {"good": {"precision": 0.9, "recall": 0.9}, "copy": {"precision": 1, "recall": 1, "memorization_flag": True}}
    s = M.score_candidates(m, w)
    assert s["best"] == "good" and s["per_candidate"]["copy"]["excluded_memorization"]


def test_specific_metrics_weighted_more():
    w = M.default_weights("generation", "scrna")
    assert w["gene_mean_corr"] > w["precision"]


def test_codec_roundtrip():
    df = pd.DataFrame({"x": [0.5, 1.5, np.nan, 4.25] * 5, "k": [1, 2, 3, 1] * 5, "c": ["u", "v", "u", None] * 5})
    t = tabular_column_types(df)
    assert t == {"x": "numeric", "k": "categorical", "c": "categorical"}
    codec = TabularCodec(df, t)
    back = codec.decode(codec.encode(df))
    assert back["c"].tolist()[:4] == ["u", "v", "u", "u"]
    assert np.allclose(back["x"][:4], [0.5, 1.5, 1.5, 4.25])
    assert back["k"].tolist()[:4] == [1, 2, 3, 1]


def test_loaders(image_npz, table_csv, scrna_h5ad):
    im = load_dataset(str(image_npz))
    assert im.modality == "image" and im.labeled and im.splits["train"].X.shape[1:] == (3, 16, 16)
    tb = load_dataset(str(table_csv))
    assert tb.modality == "tabular" and tb.label_name == "label" and "label" not in tb.splits["train"].X
    assert profile_dataset(tb)["missing_values"] > 0
    sc = load_dataset(str(scrna_h5ad))
    assert sc.modality == "scrna" and not sc.labeled and sc.splits["train"].eval_labels is not None
    nolab = load_dataset(str(table_csv), label="none")
    assert not nolab.labeled


def test_augmentation_ratio_cap():
    a = allocate("class_balancing", "augmentation", 100, NAMES, COUNTS, max_synthetic_per_real=3)
    assert a["per_class"] == {"a": 0, "b": 50, "c": 30}
    assert any("capped" in note for note in a["notes"])


def test_split_switches_to_60_20_20_for_rare_classes():
    from genagent.data import split_indices

    y = np.r_[np.zeros(990, int), np.ones(10, int)]   # 1% minority
    meta = {}
    idx = split_indices(len(y), y, 0, meta)
    assert meta["split"].startswith("60/20/20") and len(idx["test"]) == 200
    meta = {}
    split_indices(1000, np.r_[np.zeros(900, int), np.ones(100, int)], 0, meta)
    assert meta["split"].startswith("80/10/10")


def test_per_class_checks_flag_copied_class():
    rng = np.random.default_rng(0)
    E_tr = rng.normal(size=(600, 4))
    y_tr = np.repeat([0, 1, 2], 200)
    E_ho, y_ho = rng.normal(size=(300, 4)), np.repeat([0, 1, 2], 100)
    fake = rng.normal(size=(150, 4))
    y_f = np.repeat([0, 1, 2], 50)
    fake[y_f == 2] = E_tr[y_tr == 2][:50]          # class 2 copies training samples
    res = M.per_class_checks(E_tr, y_tr, E_ho, y_ho, fake, y_f, ["a", "b", "c"])
    assert res["memorization_flagged_classes"] == ["c"] and res["memorization_flag_any_class"]


# ---------------------------------------------------------------- models

def _vector_ctx(d, n, n_classes=0):
    """Context for d-dimensional vectors handled as 1x1xd 'images' (flattened, no rescaling)."""
    from genagent.models import ModelContext

    ctx = ModelContext(modality="image", n_train=n, shape=(1, 1, d), n_features=d)
    if n_classes:
        ctx.n_classes, ctx.class_freq = n_classes, np.full(n_classes, 1 / n_classes)
    return ctx


def _planted(n=20000, d=10, q=2, sigma2=0.1, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d, q))
    X = rng.normal(size=(n, q)) @ W.T + np.sqrt(sigma2) * rng.normal(size=(n, d)) + 3.0
    return X.reshape(n, 1, 1, d).astype(np.float32), W


def test_ppca_recovers_planted_subspace_and_noise():
    from genagent.models.ppca import PPCA
    from genagent.utils import Deadline

    X, W = _planted()
    m = PPCA({"n_components": 2}, _vector_ctx(10, len(X)))
    m.fit(X, None, None, None, Deadline(60))
    assert abs(m.sigma2 - 0.1) < 0.01
    # The fitted loadings span the planted subspace: projecting W onto it loses almost nothing
    P = m.U @ m.U.T
    assert np.linalg.norm(W - P @ W) / np.linalg.norm(W) < 0.02


def test_ppca_log_likelihood_matches_dense_gaussian():
    from scipy.stats import multivariate_normal

    from genagent.models.ppca import PPCA
    from genagent.utils import Deadline

    X, _ = _planted(n=500)
    m = PPCA({"n_components": 3}, _vector_ctx(10, len(X)))
    m.fit(X, None, None, None, Deadline(60))
    C = m.W @ m.W.T + m.sigma2 * np.eye(10)
    ref = multivariate_normal(m.mu, C).logpdf(X[:20].reshape(20, -1))
    assert np.allclose(m.log_likelihood(X[:20]), ref, rtol=1e-6)


def test_ppca_with_one_class_equals_unlabeled():
    from genagent.models.ppca import PPCA
    from genagent.utils import Deadline

    X, _ = _planted(n=500)
    a = PPCA({"n_components": 2}, _vector_ctx(10, len(X)))
    a.fit(X, None, None, None, Deadline(60))
    b = PPCA({"n_components": 2}, _vector_ctx(10, len(X), n_classes=1))
    b.fit(X, np.zeros(len(X), dtype=int), None, None, Deadline(60))
    assert np.allclose(a.log_likelihood(X[:20]), b.log_likelihood(X[:20]))
    assert b.sample(7, np.zeros(7, dtype=int)).shape == (7, 1, 1, 10)


def test_aliases_resolve_with_fixed_config():
    from genagent.models import REGISTRY, resolve_model

    cls, fixed = resolve_model("wgan_gp_scrna")
    assert cls is REGISTRY["gan"] and fixed == {"loss": "wgan_gp"}
    assert resolve_model("ddpm_image") == (REGISTRY["ddpm"], {})
    assert "ddpm_image" not in REGISTRY
    with pytest.raises(KeyError):
        resolve_model("nope")


def test_vector_space_roundtrips_tables_and_maps_counts(table_csv, scrna_h5ad):
    from genagent.models import ModelContext
    from genagent.models.spaces import VectorSpace

    ds = load_dataset(str(table_csv))
    ctx = ModelContext.from_dataset(ds)
    X = ds.splits["train"].X
    vs = VectorSpace(ctx)
    E = vs.fit(X)
    assert E.shape[1] == ctx.codec.dim and len(vs.cat_blocks) == len(ctx.codec.cat)
    back = vs.decode(E, None, np.random.default_rng(0))
    assert (back[ctx.codec.cat].astype(str).to_numpy() == X[ctx.codec.cat].astype(str).to_numpy()).mean() > 0.95
    ds = load_dataset(str(scrna_h5ad))
    ctx = ModelContext.from_dataset(ds)
    vs = VectorSpace(ctx, pca=5)
    E = vs.fit(ds.splits["train"].X)
    counts = vs.decode(E[:4], None, np.random.default_rng(0))
    assert counts.shape == (4, ctx.n_features) and np.all(counts == np.round(counts)) and counts.min() >= 0


@pytest.mark.parametrize("loss", ["vanilla", "hinge", "wgan_gp"])
@pytest.mark.parametrize("modality", ["image", "tabular"])
def test_gan_losses_train_and_sample(loss, modality, image_npz, table_csv):
    from genagent.models import ModelContext
    from genagent.models.gan import GAN
    from genagent.utils import Deadline

    ds = load_dataset(str(image_npz if modality == "image" else table_csv))
    ctx = ModelContext.from_dataset(ds)
    tr = ds.splits["train"]
    m = GAN({"loss": loss, "epochs": 1}, ctx)
    h = m.fit(tr.X, tr.y, None, None, Deadline(60))
    assert np.isfinite(h["d_loss"][0]) and np.isfinite(h["g_loss"][0])
    assert len(m.sample(5)) == 5
