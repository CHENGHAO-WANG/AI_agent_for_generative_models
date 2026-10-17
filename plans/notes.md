# Project notes: genagent — an AI agent for generative modeling

Agreed 2026-10-16 (grilling session); revised 2026-10-17 after the first demo runs
(see `docs/development-log.md`). Course project; deadline 2026-10-19.
Status: implemented; `docs/overview.md` describes the current state. Items marked
*(2026-10-17)* are being implemented on branch `fix/demo-run-issues`.

## Purpose
Given a dataset and an optional user objective, an agent decides whether and how to
apply generative modeling: it profiles the data, interprets the objective, selects and
tunes generative models, trains them, evaluates the generated data, recommends
downstream use, and writes a report. Requirements: `project_requirements.txt`.
The 4-page course report is out of scope here.

## Architecture
- **Claude Code plugin `genagent`** (decision maker). Main command:
  `/genagent:run <path|pathmnist|pbmc3k> [n] [objective text] [--label <col>|none] [--budget 30m] [--no-questions] [--export <name>]`.
  A skill tells Claude how to drive the toolkit and what to decide at each step.
  No rule-based planner: Claude makes all decisions.
- **Python package `genagent`** (toolkit, CLI): `init`, `plan`, `train`, `evaluate`,
  `refine`, `select`, `finalize`, `report`, `replay`. Tools are deterministic given their config.
- **Decision log** `runs/<id>/decisions.json`: every decision with its rationale.
  `genagent replay runs/<id>` re-executes a run from it without Claude.
- The repo is both the Python package and the plugin (local `marketplace.json`).
- Exports go to `outputs/<name>_<timestamp>/` next to the `runs/` folder, independent of
  the working directory *(2026-10-17)*.

## Decision process (per run)
1. Profile data (modality, size, feature types, labels, class balance, missingness).
2. Interpret objective: one **primary** objective plus optional **secondary** objectives
   *(2026-10-17)*. Optionally ask the user up to 3 multiple-choice questions, only during
   planning and only when the answer materially changes the plan. `--no-questions` accepts defaults.
3. Select 2-3 candidate models from the fixed registry (Claude may not write new
   model code during a run). Every candidate must support the primary objective. A candidate
   that cannot serve a secondary objective (e.g. no encoder for representation) needs the
   user's consent, asked during planning; under `--no-questions` it is included, competes only
   for the objectives it supports, and is flagged *(2026-10-17)*.
4. Claude sets the initial config and search space; Optuna runs a short search
   (~4-6 trials per candidate, reduced epochs/subsample, cheap proxy score,
   ~30% of budget); best config is fully trained.
5. Evaluate all candidates on every objective; rank by weighted score; one refinement
   round allowed if the best fails a quality threshold.
6. Select the final model **per objective** from the same candidates; prefer one model for
   all objectives when it is within a small margin of the best on each *(2026-10-17)*.
7. Generate final samples; run each objective's downstream analysis; render report;
   Claude writes interpretation and recommendations.

Default time budget ~30 min per dataset on an RTX 3060 Ti (8 GB).

## Model registry
| Modality | Models | Source |
|---|---|---|
| Images | class-conditional DDPM | diffusers (`UNet2DModel`, DDPM/DDIM scheduler) |
| Images | conditional conv VAE, conditional DCGAN | from scratch (PyTorch) |
| Images | flows: excluded (poor sample quality; rationale stated by agent) | — |
| scRNA | scVI (NB-VAE) | scvi-tools |
| scRNA | latent diffusion on scVI latents, WGAN-GP on normalized counts | from scratch |
| scRNA | neural spline flow on PCA/scVI latents | zuko |
| Tabular | CTGAN, TVAE | ctgan |
| Tabular | flow | zuko |

Policy: use libraries where possible, otherwise write in PyTorch.
Avoid synthcity and scDiffusion (incompatible numpy/torch pins).
Class conditioning is described honestly per model *(2026-10-17)*: scVI has no label input and
"replays posteriors of real cells of the class" — a weak choice for very rare classes, kept as a
candidate. Training-time rebalancing is offered only where a library exposes it natively
(CTGAN `log_frequency`); no workarounds are written for other models.

## Objectives and the meaning of `n`
- **Generation** (default when no objective is given): `n` = total synthetic samples,
  split proportionally to real class distribution (class-conditional when labeled).
- **Class balancing — augmentation** (default meaning of balancing): `n` = per-class
  target; each class with fewer than `n` real observations is augmented to `n`;
  synthetic data is combined with real data. If `n` exceeds the majority-class size,
  ask: clamp to majority size / keep `n` (augment every class) / enter new value.
  Under `--no-questions`, stop with an error. Optional plan cap `max_synthetic_per_real`
  (no default) limits each class's synthetic:real ratio, so capped classes stay below `n`
  *(2026-10-17)*.
- **Class balancing — fully synthetic** (only when objective asks, e.g. shareable
  dataset): `n` synthetic samples per class.
- **Representation learning, anomaly detection**: `n` optional; if given, generate
  `n` samples split proportionally. Main outputs: latent embedding / anomaly scores.
- **Imputation, uncertainty quantification**: recognized and declined with an
  explanation of what would be used.
- Allocation follows the primary objective; samples come from the model selected for the
  sample-producing objective.

## Evaluation
- **General layer** (any data type) on a shared embedding (standardized PCA for
  tabular/scRNA; classifier features if labeled, else autoencoder features, for images):
  precision/recall/density/coverage, MMD, C2ST AUC, DCR (memorization vs holdout),
  TSTR when labeled.
- **Specific layer**, used when the modality matches: images — FID in classifier
  feature space, sample grids; scRNA — marker genes, gene mean/variance correlation,
  cluster proportions, UMAP overlay; tabular — sdmetrics column shapes and
  pair correlations.
- **Per-class checks** *(2026-10-17)*: class consistency and DCR are reported per class on a
  per-class probe sample; a balancing run is flagged for memorization if any class it
  generates for is flagged.
- **scRNA cluster proportions** *(2026-10-17)*: reference = training-split proportions;
  scored relative to a real-vs-real baseline (held-out real cells assigned the same way).
- **Representation baselines** *(2026-10-17)* are unsupervised (images: pixel PCA and a
  convolutional autoencoder; scRNA/tabular: PCA). When reference labels were derived from
  the baseline's own space (Leiden on PCA), the comparison is flagged as circular.
- Ranking: weighted score; the primary objective's weights plus half of each secondary
  objective's weights; modality upweights its specific metrics. Default weights in code;
  Claude may adjust with logged justification. DCR is a hard guardrail (memorizing models excluded).
- Augmentation evaluated by classifier trained on real, real class-weighted, real+synthetic,
  and real+synthetic class-weighted, with equal training steps across arms, tested on the
  real test set (balanced accuracy, per-class recall with bootstrap 95% intervals), several seeds.

## Data handling
- Official splits when present; otherwise seeded stratified 80/10/10, or 60/20/20 when the
  smallest class is under 2% of the samples *(2026-10-17)*. Test split never seen by generators.
- Input formats: `.npz`, `.h5ad`, `.csv`/`.parquet`, image folders; built-in names
  `pathmnist` (28x28), `pbmc3k`. `--label` optional; auto-detect otherwise; ask if ambiguous.
- PBMC 3k: standard scanpy QC, 2000 HVGs, raw counts. The scanpy-tutorial cell-type
  annotations (8 types, same 2,638 barcodes) are joined as `cell_type` *(2026-10-17)*: used as
  evaluation labels, and as modeling labels with `--label cell_type`. Other unlabeled scRNA
  data falls back to Leiden clusters for evaluation.
- Synthetic data saved in the input's format (plus image grids).

## Outputs
`runs/<id>/`: synthetic data, metrics, `decisions.json`, report (Markdown + HTML).
Demo plan (revised 2026-10-17; the user runs the demos): PBMC 3k with cell-type labels for
class balancing; PathMNIST for generation (primary) with representation (secondary).
The imbalanced PathMNIST variant was dropped.
Report recommendations follow a checklist: suitable uses, uses to avoid, caveats,
suggested synthetic:real ratio.

## Engineering
- Python 3.12, uv with `uv.lock`; CUDA PyTorch from the cu130 index (driver 617).
- pytest unit tests with tiny fixtures; 2-minute end-to-end smoke test via CLI tools.
- Windows: DataLoader `num_workers=0`.

## Open questions
See `plans/open-issues.md`.
