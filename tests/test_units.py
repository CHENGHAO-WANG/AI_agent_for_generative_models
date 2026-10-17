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
