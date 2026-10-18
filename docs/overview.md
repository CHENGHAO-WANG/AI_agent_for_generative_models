# Overview

Current state of the implemented system. Intended design and agreed decisions: [plans/notes.md](../plans/notes.md). Metric and scoring details: [evaluation.md](evaluation.md). Model construction: [generative-models.md](generative-models.md). Issues and their resolutions: [plans/issue-log.md](../plans/issue-log.md).

## Components

```
Claude Code plugin (skills/run)            decides: objectives, models, hyperparameters, weights,
   SKILL.md + reference/*.md               refinement, selection, recommendations
        |  calls `genagent <step>` (JSON in / JSON out)
Python toolkit (src/genagent)              computes: loading, profiling, training, metrics, report
        |  reads/writes
Run directory runs/<id>/                   all state; decisions.json + commands.json make it replayable
Exports outputs/<name>_<timestamp>/        report + artifacts, next to runs/
```

- **Plugin** (`.claude-plugin/`, `skills/run/`): one user-invoked skill, `/genagent:run`. `SKILL.md` holds the step sequence; `reference/decision-guide.md` holds the decision heuristics and metric interpretation; `reference/plan-schema.md` the plan format; `reference/report-guide.md` the narrative format and recommendation checklist.
- **Toolkit** (`src/genagent/`):
  - `data.py`: loads built-ins (`pathmnist` via medmnist at 28×28; `pbmc3k` via scanpy with standard QC + 2000 HVGs, raw counts, and the scanpy-tutorial cell types joined by barcode as `cell_type`) and files (`.npz`, `.h5ad`, `.csv`/`.parquet`, image folders). Official splits, or a seeded split stratified on labels (or evaluation labels): 80/10/10, or 60/20/20 when the smallest class is under 2% of samples. Label auto-detection with ambiguity flag; profiling. scRNA evaluation labels: the modeling labels, else annotations (`pbmc3k` cell types), else Leiden clusters (flagged as circular for the PCA baseline). The data cache is `data_cache/` or `$GENAGENT_DATA_CACHE`.
  - `embed.py`: shared embedding for general metrics (scRNA: log-normalized PCA-50; tabular: standardized/one-hot codec + PCA-20; images: features of a CNN classifier trained on the real data, or an autoencoder if unlabeled), the tabular codec, and the image classifier also used downstream (trainable for a fixed number of steps).
  - `models/`: registry of generative models behind one interface (`fit`, `sample(n, y)`, optional `encode`, `anomaly_score`), each declaring how it uses class labels. One model per algorithm; the network and likelihood follow the modality (`spaces.py` vector representations, `nets.py` backbones): `ddpm` (U-Net for images, residual-MLP denoiser for scRNA/tabular; classifier-free guidance, EMA), `cvae` (Bernoulli / negative-binomial / Gaussian+categorical likelihoods), `gan` (`loss` vanilla, hinge or wgan_gp), `ppca` (closed form), `gplvm` (GPyTorch, amortized encoder), `nsf_flow` (zuko; scRNA and tabular). scRNA only: `scvi` (scvi-tools; no label input — class samples replay real cells' posteriors), `scanvi` (scVI followed by scANVI; label-dependent latent prior; needs labels), `latent_diffusion_scvi` (scVI + the shared DDPM core, ancestral sampling). Tabular only: `ctgan` (with `log_frequency`), `tvae`. Former per-modality names resolve as aliases. Details: [generative-models.md](generative-models.md).
  - `hpo.py`: short Optuna search, scored by MMD to validation data in the shared embedding.
  - `metrics.py`: general and modality-specific metrics, per-class consistency and memorization checks, goodness mapping, objective/modality weights, ranking with a memorization guardrail.
  - `allocation.py`: per-class sample counts from objective and `n`, with an optional per-class synthetic:real cap.
  - `pipeline.py`: the steps `init → plan → train → evaluate → [refine → evaluate] → select → finalize`, primary and secondary objectives, per-objective scoring and selection, the decision log and command log.
  - `downstream.py`: per-objective analysis (augmentation: four classifier arms with equal training effort and bootstrap intervals; latent embeddings; anomaly scores) and saving synthetic data in the input's format.
  - `report.py`: Markdown + self-contained HTML report from run files plus the agent's narrative; export to `outputs/<name>_<timestamp>/` next to the `runs/` folder.
  - `cli.py`: `genagent` command (all steps, `status`, `note`, `replay`, `registry`).

## Workflow

1. `init` profiles the data and fits the shared embedding; returns models available for the modality (with default configs, search spaces and class-conditioning descriptions) and flags (ambiguous labels, rare classes, circular evaluation labels).
2. The agent writes a plan: a primary objective and optional secondary objectives, `n`, candidates with configs, search spaces, time fractions, label use, weight changes and rationales. `plan` validates it: every candidate must serve the primary objective; one that cannot serve a secondary objective is rejected until the user approves it (automatic under `--no-questions`). It then computes the allocation from the sample-producing objective.
3. `train` runs the search (30% of each candidate's time) and the full training under a wall-clock deadline (65% of the budget split by time fraction).
4. `evaluate` samples from each candidate (class labels matched to the test split), computes metrics for every objective against the real test split, runs a class-balanced per-class probe for label-conditioned candidates, scores unsupervised representation baselines, and ranks candidates per objective and combined.
5. The agent may use one refinement round, then `select`s a model per objective (default: one model unless another wins an objective by more than 0.03).
6. `finalize` generates the final synthetic data with the sample-producing objective's model and runs each objective's downstream analysis with its own model.
7. The agent writes the narrative; `report` renders it with all factual sections and the decision log.

`replay` re-executes `commands.json` with the stored plan, refinement, and narrative inputs, so a run can be reproduced without the agent.

## Assumptions and limitations

- Images are expected small (≤ 64 px; resized otherwise); the U-Net needs sides divisible by 4.
- scRNA input must contain raw counts (`X`, `layers["counts"]`, or `raw`); otherwise an `.h5ad` is treated as a numeric table.
- Generated scRNA from models on log-normalized expression (`ddpm`, `gan`, `nsf_flow`, `ppca`, `gplvm`) is mapped back to counts by Poisson sampling at empirical library sizes, so its noise model is simpler than the negative binomial of `scvi` and `cvae`.
- `ddpm` and `gan` on tables treat one-hot categories as continuous values (argmax / Gumbel-softmax); `gplvm` trains on at most `max_points` points and models images and scRNA on their top principal components.
- Saved models from runs before the 2026-10-17 registry refactor no longer load (classes moved); `replay` retrains and is unaffected.
- Evaluation sample size is capped at 3000 and uses the real test split as reference; rare classes have few test samples even with the 60/20/20 split (PBMC 3k: 7 dendritic cells, 3 megakaryocytes), which the bootstrap intervals expose.
- Conditional models learn rare classes from few examples; `scvi` has no label input, so its class-specific samples replay real cells. `scanvi` and `cvae` are label-conditioned count models; a class with a handful of cells still gets a weakly determined prior.
- Imputation and uncertainty quantification objectives are recognized and declined.
- Metric weights and thresholds are heuristics documented in the decision guide.
