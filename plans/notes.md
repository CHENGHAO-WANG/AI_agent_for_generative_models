# Project notes: genagent — an AI agent for generative modeling

Agreed 2026-10-16 (grilling session). Course project; deadline 2026-10-19.
Status: design agreed, nothing implemented yet. Everything below is intended design
until `docs/overview.md` says otherwise.

## Purpose
Given a dataset and an optional user objective, an agent decides whether and how to
apply generative modeling: it profiles the data, interprets the objective, selects and
tunes generative models, trains them, evaluates the generated data, recommends
downstream use, and writes a report. Requirements: `project_requirements.txt`.
The 4-page course report is out of scope here.

## Architecture
- **Claude Code plugin `genagent`** (decision maker). Main command:
  `/genagent:run <path|pathmnist|pbmc3k> [n] [objective text] [--label <col>|none] [--budget 30m] [--no-questions]`.
  A skill tells Claude how to drive the toolkit and what to decide at each step.
  No rule-based planner: Claude makes all decisions.
- **Python package `genagent`** (toolkit, CLI): `profile`, `train`, `sample`,
  `evaluate`, `render-report`, `replay`. Tools are deterministic given their config.
- **Decision log** `runs/<id>/decisions.json`: every decision with its rationale.
  `genagent replay runs/<id>` re-executes a run from it without Claude.
- The repo is both the Python package and the plugin (local `marketplace.json`).

## Decision process (per run)
1. Profile data (modality, size, feature types, labels, class balance, missingness).
2. Interpret objective; optionally ask the user up to 3 multiple-choice questions,
   only during planning and only when the answer materially changes the plan.
   `--no-questions` accepts defaults.
3. Select 2-3 candidate models from the fixed registry (Claude may not write new
   model code during a run).
4. Claude sets the initial config and search space; Optuna runs a short search
   (~4-6 trials per candidate, reduced epochs/subsample, cheap proxy score,
   ~30% of budget); best config is fully trained.
5. Evaluate all candidates; rank by weighted score; one refinement round allowed if
   the best fails a quality threshold.
6. Generate final samples; render report; Claude writes interpretation and
   recommendations.

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

## Objectives and the meaning of `n`
- **Generation** (default when no objective is given): `n` = total synthetic samples,
  split proportionally to real class distribution (class-conditional when labeled).
- **Class balancing — augmentation** (default meaning of balancing): `n` = per-class
  target; each class with fewer than `n` real observations is augmented to `n`;
  synthetic data is combined with real data. If `n` exceeds the majority-class size,
  ask: clamp to majority size / keep `n` (augment every class) / enter new value.
  Under `--no-questions`, stop with an error.
- **Class balancing — fully synthetic** (only when objective asks, e.g. shareable
  dataset): `n` synthetic samples per class.
- **Representation learning, anomaly detection**: `n` optional; if given, generate
  `n` samples split proportionally. Main outputs: latent embedding / anomaly scores.
- **Imputation, uncertainty quantification**: recognized and declined with an
  explanation of what would be used.

## Evaluation
- **General layer** (any data type) on a shared embedding (standardized PCA for
  tabular/scRNA; classifier features if labeled, else autoencoder features, for images):
  precision/recall/density/coverage, MMD, C2ST AUC, DCR (memorization vs holdout),
  TSTR when labeled.
- **Specific layer**, used when the modality matches: images — FID in classifier
  feature space, sample grids; scRNA — marker genes, gene mean/variance correlation,
  Leiden cluster proportions, UMAP overlay; tabular — sdmetrics column shapes and
  pair correlations.
- Ranking: weighted score; objective sets primary metric, modality upweights its
  specific metrics. Default weights in code; Claude may adjust with logged
  justification. DCR is a hard guardrail (memorizing models excluded).
- Augmentation evaluated by classifier trained on real vs real+synthetic, tested on
  real test set (balanced accuracy, per-class recall), several seeds.

## Data handling
- Official splits when present; otherwise seeded stratified 80/10/10. Test split
  never seen by generators.
- Input formats: `.npz`, `.h5ad`, `.csv`/`.parquet`, image folders; built-in names
  `pathmnist` (28x28), `pbmc3k`. `--label` optional; auto-detect otherwise; ask if ambiguous.
- PBMC 3k: standard scanpy QC, 2000 HVGs, raw counts; unlabeled; Leiden clusters
  for evaluation only.
- Synthetic data saved in the input's format (plus image grids).

## Outputs
`runs/<id>/`: synthetic data, metrics, `decisions.json`, report (Markdown + HTML).
Demos committed under `outputs/`: `generated_report_1` (PathMNIST, class balancing
by augmentation), `generated_report_2` (PBMC 3k, generation + representation learning).
Report recommendations follow a checklist: suitable uses, uses to avoid, caveats,
suggested synthetic:real ratio.

## Engineering
- Python 3.12, uv with `uv.lock`; CUDA PyTorch from the cu130 index (driver 617).
- pytest unit tests with tiny fixtures; 2-minute end-to-end smoke test via CLI tools.
- Windows: DataLoader `num_workers=0`.

## Open questions
None at agreement time.
