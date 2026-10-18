# Issue log

Unresolved decisions and unsatisfactory implementations to return to, and how each was resolved. Design changes are recorded in `docs/development-log.md`.

Entries 1–7 were found in two `/genagent:run` runs on 2026-10-16/17 (`pathmnist_20261016-185228`, class balancing; `pbmc3k_20261017-000600`, representation + generation; run folders under `D:/PythonProject/gen-agent-runs/runs/`). Entries 8–9 come from `pathmnist_imbalanced_20261017-004129` (class balancing on `data_cache/pathmnist_imbalanced.npz`, imbalance ratio 16.3). Entry 10 was recorded on 2026-10-17 when the scANVI gap was deferred.

## Open

None.

## Resolved

### 1. Representation baseline is circular with the scRNA evaluation labels

- **Issue:** For unlabeled scRNA data, the evaluation labels are Leiden clusters computed on a PCA-50 of the training split, and val/test cells are labelled by 15-NN in that same PCA space (`src/genagent/data.py:206-229`). The representation baseline is `lognorm-pca50`, and latent metrics are scored on the test split, so the baseline's `latent_knn_acc` is close to a 15-NN vote against labels produced by a 15-NN vote in its own space. ARI and silhouette are inflated too.
- **Observed:** pbmc3k baseline kNN 0.757 / ARI 0.607, against 0.698 / 0.470 for the latent-diffusion scVI latent; the latent's silhouette was higher (0.248 vs 0.188). The report could not say whether the learned latent beats PCA.
- **Workaround:** The caveat was logged in the run's decision log and report. Candidate-vs-candidate comparisons remain fair.
- **Unresolved:** Use labels that are independent of the baseline embedding (annotated cell types; see issue 2), or define the reference clusters on an embedding that is not the baseline. Either way, report the baseline comparison as non-conclusive while it is circular.
- **Resolved 2026-10-17:** Representation baselines are unsupervised (scRNA/tabular: PCA), and a comparison whose reference labels were derived from the baseline's own space (Leiden on PCA) is flagged as circular; PBMC 3k now has annotated cell types as evaluation labels (issue 2). Evidence: commit 8238828; [development log, 2026-10-17: Revisions after the first demo runs](../docs/development-log.md#2026-10-17-revisions-after-the-first-demo-runs).

### 2. The pbmc3k built-in ignores the available cell-type annotations

- **Issue:** `_load_pbmc3k` (`src/genagent/data.py:143-149`) loads raw `sc.datasets.pbmc3k()`, which has no labels. `sc.datasets.pbmc3k_processed()` carries the scanpy-tutorial `louvain` annotations (8 cell types) for what is very likely the same 2,638 cells after QC, but it stores scaled values, not counts.
- **Consequence:** It causes issue 1, and it rules out conditional generation by cell type (issue 4).
- **Unresolved:** Whether the built-in should join the tutorial annotations onto the raw counts by barcode (as `cell_type`), offered as a labeled variant or used only as evaluation labels. Check that the toolkit's QC keeps the same barcodes.
- **Resolved 2026-10-17:** The pbmc3k built-in joins the scanpy-tutorial cell types onto the raw counts by barcode as `cell_type`, used as evaluation labels and, with `--label cell_type`, as modeling labels. Evidence: commit 8238828; [development log, 2026-10-17: Revisions after the first demo runs](../docs/development-log.md#2026-10-17-revisions-after-the-first-demo-runs).

### 3. Latent metrics are computed only under the `representation` objective

- **Issue:** `latent_metrics` runs only when `objective.type == "representation"` (`src/genagent/pipeline.py:376-417`). A mixed request ("generate cells and learn a latent space") must choose one objective; under `generation`, latent quality is never scored, even for encoder models.
- **Workaround:** In the pbmc3k run the user was asked and chose representation + generate. That excluded `nsf_flow` from the comparison.
- **Unresolved:** Whether to compute latent metrics for every encoder-capable candidate regardless of objective (weighted 0 when the objective is not representation), or to support secondary objectives in the plan.
- **Resolved 2026-10-17:** Plans take secondary objectives; every candidate is scored on every objective it can serve (latent metrics included), and a model is selected per objective. Evidence: commit 8238828; [development log, 2026-10-17: Revisions after the first demo runs](../docs/development-log.md#2026-10-17-revisions-after-the-first-demo-runs).

### 4. Unconditional scRNA generation distorts cell-state proportions, with no way to correct them

- **Issue:** With `use_labels: false` (the default for representation, and the only option on unlabeled data), sampling is unconditional, and the latent-diffusion prior concentrates mass in the dominant T-cell mode.
- **Observed** (cluster share, real test → synthetic):
  - leiden_0 (T cells): 46.2% → 65.9%
  - B cells: 15.2% → 4.9%
  - NK cells: 4.9% → 0.8%
  - platelets: 0.4% → 0%
  - `cluster_tvd` 0.250. The refinement (15k diffusion steps, width 512) made it worse: 80.7% leiden_0, TVD 0.345.
- **Unresolved:** How to control proportions on unlabeled data, for example by allowing the evaluation clusters (or annotated types, issue 2) as conditioning labels with a proportional allocation, or by sampling from the aggregate posterior instead of the learned prior.
- **Resolved 2026-10-17:** The latent-diffusion prior was repaired (residual FiLM denoiser, EMA, cosine learning rate, no clipping, ancestral sampling): PBMC cluster TVD 0.25-0.35 -> 0.05, against 0.04 for the real-vs-real baseline. Cell types can be used as conditioning labels (issue 2). Evidence: commit 8238828; [development log, 2026-10-17: Revisions after the first demo runs](../docs/development-log.md#2026-10-17-revisions-after-the-first-demo-runs).

### 5. `cluster_tvd` and `marker_expr_corr` are biased toward large clusters and noisy

- **Issue:** Synthetic cells are assigned to clusters by a 15-NN majority vote in PCA space, against the training cells (`src/genagent/metrics.py:179-186`). Ambiguous or blurred cells tend to be voted into the largest neighbouring cluster, which inflates large clusters independently of the model; scVI also showed leiden_0 at 56.1%. The reference proportions come from the 264-cell test split, where leiden_6 is a single cell.
- **Unresolved:** Separate assignment bias from model error, for example by reporting the real-vs-real assignment of held-out cells as a baseline TVD, using train+val proportions as the reference, or using a calibrated classifier instead of majority kNN.
- **Resolved 2026-10-17:** Cluster proportions are scored against training-split proportions and relative to a real-vs-real baseline (held-out real cells assigned the same way). Evidence: commit 8238828; [development log, 2026-10-17: Revisions after the first demo runs](../docs/development-log.md#2026-10-17-revisions-after-the-first-demo-runs).

### 6. `GA report --export` and the run/cache directories resolve against the current working directory

- **Issue:** `export(run, name, out_root="outputs")` (`src/genagent/report.py:244`), `DATA_CACHE = Path("data_cache")` (`src/genagent/data.py:20`) and `runs/` are all relative to the working directory. When the report was built with the run folder as the working directory, the export landed in `runs/pathmnist_20261016-185228/outputs/generated_report_1_20261017-000408/` instead of the top-level `outputs/`.
- **Unresolved:** Anchor `outputs/` (and possibly `data_cache/`) to the runs root or the project root, independent of the working directory. Also, `docs/overview.md` still describes the export path as `outputs/<name>/`, without the timestamp added in 3894da1.
- **Resolved 2026-10-17:** Exports are anchored next to the `runs/` folder, independent of the working directory; the data cache can be set with `GENAGENT_DATA_CACHE`; `docs/overview.md` describes the timestamped export path. Evidence: commit 8238828; [development log, 2026-10-17: Revisions after the first demo runs](../docs/development-log.md#2026-10-17-revisions-after-the-first-demo-runs).

### 7. PathMNIST `finalize` was killed under memory pressure (unconfirmed cause)

- **Observed:** The first `GA finalize` on PathMNIST (25,969 synthetic images, augmentation downstream with 3 seeds) was stopped by Claude Code because the system was critically low on memory; a rerun succeeded. Claude Code's notice says this reflects system-wide pressure, so the toolkit is not necessarily at fault.
- **Possible contributor (not verified):** `_augmentation` rebuilds `_concat(tr.X, fake)` inside the seed loop (`src/genagent/downstream.py:117`), holding a fresh copy of roughly 116k × 2352 values per seed.
- **Unresolved:** Measure finalize's peak memory on PathMNIST; if it is high, build the augmented array once outside the seed loop.
- **Resolved 2026-10-17:** The augmentation arrays are built once outside the seed loop; finalize's peak memory on PathMNIST was measured at 3.8 GB. Evidence: commit 8238828; [development log, 2026-10-17: Revisions after the first demo runs](../docs/development-log.md#2026-10-17-revisions-after-the-first-demo-runs).

### 8. Augmentation allocation has no cap on the per-class synthetic:real ratio

- **Issue:** `allocate` (`src/genagent/allocation.py:26-52`) fills every class below `n` up to `n`, whatever the class's real size; `n_exceeds_policy` only governs `n` above the majority size. On the imbalanced PathMNIST, `n = 12885` gave about 12k synthetic patches each to mucus (801 real), normal colon mucosa (789) and stroma (940), about 15:1.
- **Observed:** Real + synthetic reached 0.750 ± 0.020 balanced accuracy, against 0.782 ± 0.006 for the class-weighted loss, with lower recall on all three reduced classes (mucosa 0.632 vs 0.847, stroma 0.252 vs 0.442). Within one run, a lower ratio or a combination with class weighting could not be tried.
- **Unresolved:** Whether to add a per-class ratio cap to the plan (for example `max_synthetic_per_real`), and whether the downstream analysis should also evaluate class weighting combined with augmentation.
- **Resolved 2026-10-17:** The plan takes an optional per-class cap `max_synthetic_per_real`, and the augmentation analysis adds a fourth arm (real + synthetic with class weighting), with equal training steps across arms. Evidence: commit 8238828; [development log, 2026-10-17: Revisions after the first demo runs](../docs/development-log.md#2026-10-17-revisions-after-the-first-demo-runs).

### 9. Class consistency and DCR are reported only across all classes

- **Issue:** `class_consistency_acc` is one mean over all synthetic samples (`src/genagent/metrics.py:150`), and the DCR near-copy fraction is likewise pooled (`metrics.py:93`). Weak conditioning or memorisation in a few small classes is diluted by the large ones.
- **Observed:** The imbalanced run's DDPM had class consistency 0.873 (0.969 on the full data) and a DCR near-copy fraction of 0.104. The data cannot show whether the drop and the near-copies are concentrated in the ~800-image classes, which each sampled 12k patches.
- **Unresolved:** Report both metrics per class, and possibly flag memorisation per class, so that class-balancing decisions can rely on the minority classes' own numbers.
- **Resolved 2026-10-17:** Class consistency and DCR are reported per class on a class-balanced probe, and a balancing run is flagged for memorization if any class it generates for is flagged. Evidence: commit 8238828; [development log, 2026-10-17: Revisions after the first demo runs](../docs/development-log.md#2026-10-17-revisions-after-the-first-demo-runs).

### 10. No truly class-conditional scVI-family model (scANVI not in the registry)

- **Issue:** `scvi` has no label input; its "conditional" sampling encodes real cells of the requested class and decodes them with fresh noise (`SCVIModel._latent_samples`, `src/genagent/models/scrna.py`). For rare classes this replays a handful of real cells (PBMC 3k: ~12 megakaryocytes in the training split) and cannot produce unseen states. `latent_diffusion_scvi` is the only scVI-based candidate with a learned per-class prior.
- **Candidate:** scANVI (`scvi.model.SCANVI`, scvi-tools) is scVI's semi-supervised extension with a label-dependent latent prior, i.e. a generative model of counts that can sample cells of a requested type. Adding it needs a sampler around its generative internals (label → latent prior → NB decoder) and tests.
- **Unresolved:** Deferred for the 2026-10-19 deadline. Until then the registry and decision guide describe `scvi`'s class conditioning as posterior replay, suitable only as a baseline for class balancing.
- **Resolved 2026-10-17:** `scanvi` added; it samples from scANVI's label-dependent latent prior (scVI, then scANVI with a short KL warm-up). `scvi` keeps posterior replay as its documented class conditioning. Evidence: PR #2 (commit 07af14e); [development log, 2026-10-17: scANVI and an amortized GPLVM encoder](../docs/development-log.md#2026-10-17-scanvi-and-an-amortized-gplvm-encoder).
