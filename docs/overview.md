# Overview

Current state of the implemented system. Intended design and agreed decisions: [plans/notes.md](../plans/notes.md). Metric and scoring details: [evaluation.md](evaluation.md).

## Components

```
Claude Code plugin (skills/run)            decides: objective, models, hyperparameters, weights,
   SKILL.md + reference/*.md               refinement, selection, recommendations
        |  calls `genagent <step>` (JSON in / JSON out)
Python toolkit (src/genagent)              computes: loading, profiling, training, metrics, report
        |  reads/writes
Run directory runs/<id>/                   all state; decisions.json + commands.json make it replayable
```

- **Plugin** (`.claude-plugin/`, `skills/run/`): one user-invoked skill, `/genagent:run`. `SKILL.md` holds the step sequence; `reference/decision-guide.md` holds the decision heuristics and metric interpretation; `reference/plan-schema.md` the plan format; `reference/report-guide.md` the narrative format and recommendation checklist.
- **Toolkit** (`src/genagent/`):
  - `data.py`: loads built-ins (`pathmnist` via medmnist at 28×28; `pbmc3k` via scanpy with standard QC + 2000 HVGs, raw counts) and files (`.npz`, `.h5ad`, `.csv`/`.parquet`, image folders); official splits or a seeded stratified 80/10/10; label auto-detection with ambiguity flag; profiling. For unlabeled scRNA, Leiden clusters (train split, kNN-assigned elsewhere) are kept as evaluation-only labels.
  - `embed.py`: shared embedding for general metrics (scRNA: log-normalized PCA-50; tabular: standardized/one-hot codec + PCA-20; images: features of a CNN classifier trained on the real data, or an autoencoder if unlabeled), the tabular codec, and the image classifier also used downstream.
  - `models/`: registry of generative models behind one interface (`fit`, `sample(n, y)`, optional `encode`, `anomaly_score`). Images: `ddpm_image` (diffusers U-Net, classifier-free guidance, EMA), `cvae_image`, `dcgan_image`. scRNA: `scvi` (scvi-tools), `latent_diffusion_scvi`, `wgan_gp_scrna`, `nsf_flow` (zuko, on PCA space). Tabular: `ctgan`, `tvae` (ctgan library), `nsf_flow`.
  - `hpo.py`: short Optuna search, scored by MMD to validation data in the shared embedding.
  - `metrics.py`: general and modality-specific metrics, goodness mapping, objective/modality weights, ranking with a memorization guardrail.
  - `allocation.py`: per-class sample counts from objective and `n`.
  - `pipeline.py`: the steps `init → plan → train → evaluate → [refine → evaluate] → select → finalize`, the decision log and command log.
  - `downstream.py`: objective-specific analysis (augmentation classifier comparison incl. class-weighted alternative; latent embeddings; anomaly scores) and saving synthetic data in the input's format.
  - `report.py`: Markdown + self-contained HTML report from run files plus the agent's narrative; export to `outputs/<name>/`.
  - `cli.py`: `genagent` command (all steps, `status`, `note`, `replay`, `registry`).

## Workflow

1. `init` profiles the data and fits the shared embedding; returns models available for the modality with default configs and search spaces.
2. The agent writes a plan (objective type/mode, `n`, candidates with configs, search spaces, time fractions, weight changes, rationales); `plan` validates it and computes the allocation.
3. `train` runs the search (30% of each candidate's time) and the full training under a wall-clock deadline (65% of the budget split by time fraction).
4. `evaluate` samples from each candidate (class labels matched to the test split), computes metrics against the real test split, writes figures and the ranking.
5. The agent may use one refinement round, then `select`s a model (default: best score).
6. `finalize` generates the final synthetic data from the allocation and runs the downstream analysis.
7. The agent writes the narrative; `report` renders it with all factual sections and the decision log.

`replay` re-executes `commands.json` with the stored plan, refinement, and narrative inputs, so a run can be reproduced without the agent.

## Assumptions and limitations

- Images are expected small (≤ 64 px; resized otherwise); the U-Net needs sides divisible by 4.
- scRNA input must contain raw counts (`X`, `layers["counts"]`, or `raw`); otherwise an `.h5ad` is treated as a numeric table.
- Generated scRNA from the flow and WGAN-GP is mapped back to counts by Poisson sampling at empirical library sizes, so its noise model is simpler than scVI's negative binomial.
- Evaluation sample size is capped at 3000 and uses the real test split as reference; small test splits (e.g. PBMC 3k: 264 cells) make metrics noisy.
- Imputation and uncertainty quantification objectives are recognized and declined.
- Metric weights and thresholds are heuristics documented in the decision guide.
