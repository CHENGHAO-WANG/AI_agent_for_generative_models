# Plan and refinement file formats

## Plan (`GA plan --file`)

```json
{
  "objective": {
    "type": "generation | class_balancing | representation | anomaly_detection | imputation | uncertainty_quantification",
    "secondary": ["representation"],
    "mode": "augmentation | synthetic_balanced",
    "interpretation": "how you read the user's objective(s) and why"
  },
  "n": 686,
  "n_exceeds_policy": "clamp | keep | error",
  "max_synthetic_per_real": 5,
  "generate": true,
  "generate_rationale": "why synthetic samples are (not) produced",
  "use_labels": true,
  "questions": [
    {"question": "...", "options": ["...", "..."], "answer": "...", "answered_by": "user | default"}
  ],
  "candidates": [
    {
      "name": "ldm",
      "model": "latent_diffusion_scvi",
      "config": {"diff_steps": 10000},
      "search_space": "default",
      "hpo_trials": 4,
      "time_fraction": 0.5,
      "use_labels": true,
      "partial_ok": false,
      "rationale": "why this model and these settings for this data and objective"
    }
  ],
  "metric_weights": {"class_balancing": {"tstr": 3}},
  "weights_rationale": "why the defaults were changed (omit both keys to keep defaults)",
  "anomaly": {"holdout_classes": ["mucus"]}
}
```

Field rules (enforced by `GA plan`):

- `objective.type` is the primary objective; `objective.secondary` (optional) lists further objectives from the first four types. Every objective is evaluated and gets a downstream analysis.
- `objective.mode` applies to `class_balancing` only; default `augmentation`. Class balancing needs labels.
- `imputation` / `uncertainty_quantification` are declined: give no candidates; the interpretation should say what would be used instead.
- `n` defaults to the `--n` argument. Its meaning depends on the sample-producing objective (the primary if it is generation or class balancing, else the first such secondary); see the decision guide.
- `n_exceeds_policy` is needed only when augmentation's `n` exceeds the majority-class size; set it from the user's answer.
- `max_synthetic_per_real` (optional, augmentation only, no default) caps each class's synthetic count at this multiple of its real count.
- `use_labels` (plan default; each candidate may override) defaults to true when the sample-producing objective is generation or class balancing on labeled data. A representation candidate usually sets `use_labels: false`, since conditioning on labels lets the latent space ignore class structure. Class balancing requires label-conditioned candidates.
- `candidates[].model`: a registry name valid for the modality (from `init`'s `models_available`). `config` keys must exist in that model's `default_config`; unspecified keys keep their defaults. `search_space`: `"default"`, `{}` (no search), or a dict `{param: {"type": "float"|"int"|"categorical", "low", "high", "log", "choices"}}` over config keys. `hpo_trials` defaults to 5 when a search space is given. `time_fraction`s are normalized to sum to 1.
- Capabilities: generation and class balancing need `sample`, representation `encode`, anomaly detection `anomaly`. Every candidate must serve the primary objective. A candidate that cannot serve a secondary objective is rejected until the user approves it (`partial_ok: true`, recorded in `questions`); under `--no-questions` it is approved automatically. Approved partial candidates compete only for the objectives they serve. Every objective needs at least one candidate.
- `metric_weights`: `{objective: {metric: weight}}` overrides single defaults per objective (a flat `{metric: weight}` applies to the primary). `GA plan` prints the full merged tables.
- `anomaly.holdout_classes` (labeled data only) removes those classes from training so detection can be scored by AUROC on the test set.

Training time per candidate = budget x 0.65 x time_fraction; a search uses 30% of it (each trial: 25% of epochs on 30% of the data, scored by MMD to validation data in the shared embedding). Models stop at their deadline even if epochs remain.

## Refinement (`GA refine --file`), one round only

```json
[
  {"from": "dcgan", "name": "dcgan_refined", "config_overrides": {"lr_g": 1e-4, "epochs": 30},
   "time_scale": 0.5, "rationale": "what failed and why this change should fix it"}
]
```

`time_scale` (default 0.5) multiplies the original candidate's training time. The refined candidate trains without a search, from the original's final configuration plus the overrides, and keeps the original's labels setting and objectives.

## Narrative (`GA report --narrative`)

See [report-guide.md](report-guide.md).
