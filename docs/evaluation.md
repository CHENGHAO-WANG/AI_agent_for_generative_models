# Evaluation, scoring, and search proxy

Status: Current

Code: `src/genagent/metrics.py`, `src/genagent/embed.py` (`Embedder`), `src/genagent/hpo.py`, `src/genagent/pipeline.py` (`evaluate`), `src/genagent/downstream.py`.

## Shared embedding

General metrics need a feature space in which Euclidean distance is meaningful for every modality. The `Embedder` is fit once, at `init`, on real training data only:

| Modality | Embedding |
|---|---|
| scRNA | `log1p(counts / library size × 10^4)` → PCA (50 components) |
| tabular | numeric: median-impute + standardize; categorical: mode-impute + one-hot; PCA to 20 dims if wider |
| image, labeled | 128-d penultimate features of a small CNN classifier (GroupNorm, 8 epochs, cosine LR), ≈ 0.90 val / 0.83 test accuracy on PathMNIST |
| image, unlabeled | 64-d bottleneck of a convolutional autoencoder |

Inception features are not used: 28×28 histology is far from ImageNet and the system must not depend on external pretrained networks.

## Evaluation protocol

For each candidate, `evaluate` draws `m = min(|test|, 3000)` test samples; when the model is class-conditional it generates exactly the labels of those test samples, so class composition matches. All general metrics compare these synthetic samples with the real **test** samples (never seen by any generator).

Label-conditioned candidates also get a **per-class probe**: 200 samples of every class, so rare classes are judged on their own rather than diluted by common ones (`metrics.per_class_checks`). For each class: class consistency (the image classifier, else 15-NN on the shared embedding trained on real labels, predicts the requested class) and DCR against that class's own held-out samples (pooled held-out reference when the class has fewer than 20). A class is flagged for memorization under the same rule as the pooled DCR (with at least 20 probe samples); in a class-balancing run, any flagged class excludes the candidate.

Metrics are computed for every objective in the plan (primary and secondary): latent metrics for every encoder-capable candidate when representation is an objective, anomaly metrics when anomaly detection is.

## General metrics

- **Precision / recall / density / coverage** (Naeem et al., 2020), k = 5. Precision: fraction of synthetic points inside the k-NN ball of some real point. Recall: fraction of real points inside some synthetic point's ball. Density: average number of real balls containing a synthetic point, divided by k (robust to real outliers). Coverage: fraction of real points whose ball contains a synthetic point.
- **MMD²** (unbiased, RBF kernel) with the bandwidth set by the median heuristic on real training embeddings, and a real-vs-real reference (`mmd_baseline_real`: test vs an equally sized training subsample) on the same kernel.
- **C2ST AUC**: 5-fold cross-validated gradient-boosting classifier separating real from synthetic. 0.5 = indistinguishable.
- **DCR** (distance to closest record): nearest training-point distances for synthetic samples vs real test samples. `dcr_frac_below_holdout_q05` is the share of synthetic samples closer to the training set than the 5th percentile of real held-out samples (expected ≈ 0.05). `memorization_flag` is set when this exceeds 0.15 or when more than 1 % are near-exact copies.
- **TSTR** (labeled data): logistic regression on embeddings trained on synthetic data, balanced accuracy on the real test split, divided by the same classifier trained on an equal number of real training samples (TRTR).

## Modality-specific metrics

- **Image**: FID computed in the classifier feature space (Fréchet distance between Gaussian fits), with a real-vs-real baseline; class consistency = accuracy of the embedding classifier on conditionally generated samples.
- **scRNA**: Pearson correlations across genes of the mean and variance of log-normalized expression and of the zero fraction (raw counts). Group proportions: synthetic cells are assigned to the reference groups (modeling labels, else annotations such as the `pbmc3k` cell types, else Leiden clusters) by 15-NN in the shared embedding, and compared with the **training split's** proportions by total variation distance. The real-vs-real baseline `cluster_tvd_baseline_real` assigns held-out real cells (validation + test) the same way, so it measures what a perfect generator would score (sampling noise + assignment bias). Marker agreement = correlation of the group × marker-gene mean-expression matrices (top-5 markers per group by mean difference on training data; real means from the training cells' own labels).
- **Tabular**: sdmetrics `QualityReport` (column shapes, column-pair trends).
- **Objective-specific**: representation — 15-NN balanced accuracy (5-fold) and k-means ARI of reference labels (modeling labels, else evaluation labels) in the latent space of the test split, plus silhouette. Baselines are **unsupervised** embeddings scored the same way: images — PCA (50) on pixels and a convolutional autoencoder; scRNA and tabular — the shared PCA embedding. The supervised image-classifier features are not a baseline, since they would win by construction. When the reference labels are Leiden clusters computed on the same PCA space as the scRNA baseline, the baseline is marked `circular_reference_labels` and the comparison is inconclusive. Anomaly detection — AUROC when the plan holds out classes from training.

## Score

Each metric is mapped to a goodness value in [0, 1]:

| Metric | Goodness |
|---|---|
| precision, recall, coverage, latent kNN acc, sdmetrics, anomaly AUROC | value (clipped) |
| class consistency | class-balanced mean from the per-class probe (pooled accuracy when no probe) |
| density | min(density, 1) |
| C2ST | 1 − 2·|AUC − 0.5| |
| MMD | baseline / max(MMD, baseline) |
| FID | baseline / max(FID, baseline) |
| TSTR | TSTR / TRTR (clipped) |
| correlations, ARI | value clipped to [0, 1] |
| cluster TVD | baseline / max(TVD, baseline), baseline = max(real-vs-real TVD, 0.02) |

The score is the weighted mean of available goodness values. Default weights depend on the objective (`OBJECTIVE_WEIGHTS`), and every modality-specific metric's weight is multiplied by 2 when its modality matches (`SPECIFIC_MULTIPLIER`). Each objective has its own score over the candidates that serve it; the **combined** score uses the primary objective's weights plus half (`SECONDARY_WEIGHT`) of each secondary objective's weights, over the candidates serving the primary. The agent may override individual weights in the plan with a logged rationale. Candidates with `memorization_flag` (or, in class balancing, any per-class flag) are excluded from the rankings. Weights and thresholds are heuristics, not calibrated.

**Selection** (`pipeline.select`): each objective gets a model. By default the best combined-score candidate serves every objective it can, unless another candidate beats it on that objective's score by more than `PREFER_ONE_MARGIN` (0.03).

## Search proxy

The Optuna search must compare configurations of any model family on the same footing and quickly, so each trial trains on 30 % of the training data for 25 % of the configured epochs, generates up to 1000 samples matching validation labels, and is scored by MMD² to real validation samples in the shared embedding (median-heuristic bandwidth from training data). The agent's initial configuration is enqueued as the first trial, so the search can only replace it with a configuration that scored better on the proxy. The proxy favors distribution match; it does not see the objective's downstream metric.

## Downstream analysis

- **Augmentation**: four arms, each over 3 seeds, scored on the real test split: real only; real with class-weighted loss (the non-generative alternative); real + synthetic; real + synthetic with class-weighted loss. Images use the small CNN trained for the **same number of optimizer steps** in every arm (5 epochs of the real training set), so arms differ only in their data; other modalities use gradient boosting on the shared embedding. Reported: balanced accuracy (mean ± sd), macro F1, per-class recall, and 95% bootstrap intervals (1,000 resamples of the test set, seed-averaged predictions) for balanced accuracy, per-class recall, and each arm's paired difference from real-only. The synthetic arms share one augmented array (peak memory on PathMNIST ≈ 3.8 GB).
- **Synthetic-balanced**: classifier trained on synthetic only vs real only, with the same intervals.
- **Representation**: latent embeddings of all splits are saved; latent metrics on the test split; UMAP of the training-split latent.
- **Anomaly detection**: scores on validation and test; threshold at the validation 95th percentile; AUROC and recall at the threshold for hold-out classes.

## Limitations

- Metrics in a learned embedding inherit its blind spots (e.g. the classifier features ignore label-irrelevant texture).
- Small test splits make PRDC and C2ST noisy, and rare classes have few test samples even under the 60/20/20 split (PBMC 3k: 7 dendritic cells, 3 megakaryocytes); the bootstrap intervals show this but cannot fix it.
- The search proxy is the same for every objective.
