# Plan and refinement file formats

## Plan (`GA plan --file`)

```json
{
  "objective": {
    "type": "generation | class_balancing | representation | anomaly_detection | imputation | uncertainty_quantification",
    "mode": "augmentation | synthetic_balanced",
    "interpretation": "how you read the user's objective and why"
  },
  "n": 12885,
  "n_exceeds_policy": "clamp | keep | error",
  "generate": true,
  "generate_rationale": "why synthetic samples are (not) produced",
  "use_labels": true,
  "questions": [
    {"question": "...", "options": ["...", "..."], "answer": "...", "answered_by": "user | default"}
  ],
  "candidates": [
    {
      "name": "ddpm",
      "model": "ddpm_image",
      "config": {"epochs": 15, "base_channels": 64},
      "search_space": "default",
      "hpo_trials": 4,
      "time_fraction": 0.5,
      "rationale": "why this model and these settings for this data and objective"
    }
  ],
  "metric_weights": {"tstr": 3},
  "weights_rationale": "why the defaults were changed (omit both keys to keep defaults)",
  "anomaly": {"holdout_classes": ["mucus"]}
}
```

Field rules (enforced by `GA plan`):

- `objective.mode` applies to `class_balancing` only; default `augmentation`. Class balancing needs labels.
- `imputation` / `uncertainty_quantification` are declined: give no candidates; the interpretation should say what would be used instead.
- `n` defaults to the `--n` argument. Its meaning depends on the objective (see the decision guide).
- `n_exceeds_policy` is needed only when augmentation's `n` exceeds the majority-class size; set it from the user's answer.
- `use_labels` defaults to true for generation/class balancing on labeled data and false for representation/anomaly detection (conditioning on labels would let the latent space ignore class structure).
- `candidates[].model`: a registry name valid for the modality (from `init`'s `models_available`). `config` keys must exist in that model's `default_config`; unspecified keys keep their defaults. `search_space`: `"default"`, `{}` (no search), or a dict `{param: {"type": "float"|"int"|"categorical", "low", "high", "log", "choices"}}` over config keys. `hpo_trials` defaults to 5 when a search space is given. `time_fraction`s are normalized to sum to 1.
- `representation` candidates need the `encode` capability; `anomaly_detection` candidates need `anomaly`.
- `metric_weights` entries override single defaults; `GA plan` prints the full merged table.
- `anomaly.holdout_classes` (labeled data only) removes those classes from training so detection can be scored by AUROC on the test set.

Training time per candidate = budget x 0.65 x time_fraction; a search uses 30% of it (each trial: 25% of epochs on 30% of the data, scored by MMD to validation data in the shared embedding). Models stop at their deadline even if epochs remain.

## Refinement (`GA refine --file`), one round only

```json
[
  {"from": "dcgan", "name": "dcgan_refined", "config_overrides": {"lr_g": 1e-4, "epochs": 30},
   "time_scale": 0.5, "rationale": "what failed and why this change should fix it"}
]
```

`time_scale` (default 0.5) multiplies the original candidate's training time. The refined candidate trains without a search, from the original's final configuration plus the overrides.

## Narrative (`GA report --narrative`)

See [report-guide.md](report-guide.md).
