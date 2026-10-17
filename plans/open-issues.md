# Open issues

Unresolved defects and unsatisfactory implementations to return to. Entries 1–7 were found in two `/genagent:run` runs on 2026-10-16/17 (`pathmnist_20261016-185228`, class balancing; `pbmc3k_20261017-000600`, representation + generation; run folders under `D:/PythonProject/gen-agent-runs/runs/`).

## 1. Representation baseline is circular with the scRNA evaluation labels

- **Issue:** For unlabeled scRNA data, the evaluation labels are Leiden clusters computed on a PCA-50 of the training split, and val/test cells are labelled by 15-NN in that same PCA space (`src/genagent/data.py:206-229`). The representation baseline is `lognorm-pca50`, and latent metrics are scored on the test split, so the baseline's `latent_knn_acc` is close to a 15-NN vote against labels produced by a 15-NN vote in its own space. ARI and silhouette are inflated too.
- **Observed:** pbmc3k baseline kNN 0.757 / ARI 0.607, against 0.698 / 0.470 for the latent-diffusion scVI latent; the latent's silhouette was higher (0.248 vs 0.188). The report could not say whether the learned latent beats PCA.
- **Workaround:** The caveat was logged in the run's decision log and report. Candidate-vs-candidate comparisons remain fair.
- **Unresolved:** Use labels that are independent of the baseline embedding (annotated cell types; see issue 2), or define the reference clusters on an embedding that is not the baseline. Either way, report the baseline comparison as non-conclusive while it is circular.

## 2. The pbmc3k built-in ignores the available cell-type annotations

- **Issue:** `_load_pbmc3k` (`src/genagent/data.py:143-149`) loads raw `sc.datasets.pbmc3k()`, which has no labels. `sc.datasets.pbmc3k_processed()` carries the scanpy-tutorial `louvain` annotations (8 cell types) for what is very likely the same 2,638 cells after QC, but it stores scaled values, not counts.
- **Consequence:** It causes issue 1, and it rules out conditional generation by cell type (issue 4).
- **Unresolved:** Whether the built-in should join the tutorial annotations onto the raw counts by barcode (as `cell_type`), offered as a labeled variant or used only as evaluation labels. Check that the toolkit's QC keeps the same barcodes.

## 3. Latent metrics are computed only under the `representation` objective

- **Issue:** `latent_metrics` runs only when `objective.type == "representation"` (`src/genagent/pipeline.py:376-417`). A mixed request ("generate cells and learn a latent space") must choose one objective; under `generation`, latent quality is never scored, even for encoder models.
- **Workaround:** In the pbmc3k run the user was asked and chose representation + generate. That excluded `nsf_flow` from the comparison.
- **Unresolved:** Whether to compute latent metrics for every encoder-capable candidate regardless of objective (weighted 0 when the objective is not representation), or to support secondary objectives in the plan.

## 4. Unconditional scRNA generation distorts cell-state proportions, with no way to correct them

- **Issue:** With `use_labels: false` (the default for representation, and the only option on unlabeled data), sampling is unconditional, and the latent-diffusion prior concentrates mass in the dominant T-cell mode.
- **Observed** (cluster share, real test → synthetic):
  - leiden_0 (T cells): 46.2% → 65.9%
  - B cells: 15.2% → 4.9%
  - NK cells: 4.9% → 0.8%
  - platelets: 0.4% → 0%
  - `cluster_tvd` 0.250. The refinement (15k diffusion steps, width 512) made it worse: 80.7% leiden_0, TVD 0.345.
- **Unresolved:** How to control proportions on unlabeled data, for example by allowing the evaluation clusters (or annotated types, issue 2) as conditioning labels with a proportional allocation, or by sampling from the aggregate posterior instead of the learned prior.

## 5. `cluster_tvd` and `marker_expr_corr` are biased toward large clusters and noisy

- **Issue:** Synthetic cells are assigned to clusters by a 15-NN majority vote in PCA space, against the training cells (`src/genagent/metrics.py:179-186`). Ambiguous or blurred cells tend to be voted into the largest neighbouring cluster, which inflates large clusters independently of the model; scVI also showed leiden_0 at 56.1%. The reference proportions come from the 264-cell test split, where leiden_6 is a single cell.
- **Unresolved:** Separate assignment bias from model error, for example by reporting the real-vs-real assignment of held-out cells as a baseline TVD, using train+val proportions as the reference, or using a calibrated classifier instead of majority kNN.

## 6. `GA report --export` and the run/cache directories resolve against the current working directory

- **Issue:** `export(run, name, out_root="outputs")` (`src/genagent/report.py:244`), `DATA_CACHE = Path("data_cache")` (`src/genagent/data.py:20`) and `runs/` are all relative to the working directory. When the report was built with the run folder as the working directory, the export landed in `runs/pathmnist_20261016-185228/outputs/generated_report_1_20261017-000408/` instead of the top-level `outputs/`.
- **Unresolved:** Anchor `outputs/` (and possibly `data_cache/`) to the runs root or the project root, independent of the working directory. Also, `docs/overview.md` still describes the export path as `outputs/<name>/`, without the timestamp added in 3894da1.

## 7. PathMNIST `finalize` was killed under memory pressure (unconfirmed cause)

- **Observed:** The first `GA finalize` on PathMNIST (25,969 synthetic images, augmentation downstream with 3 seeds) was stopped by Claude Code because the system was critically low on memory; a rerun succeeded. Claude Code's notice says this reflects system-wide pressure, so the toolkit is not necessarily at fault.
- **Possible contributor (not verified):** `_augmentation` rebuilds `_concat(tr.X, fake)` inside the seed loop (`src/genagent/downstream.py:117`), holding a fresh copy of roughly 116k × 2352 values per seed.
- **Unresolved:** Measure finalize's peak memory on PathMNIST; if it is high, build the augmented array once outside the seed loop.
