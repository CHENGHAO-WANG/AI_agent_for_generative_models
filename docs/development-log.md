# Development log

## 2026-10-16: Imbalanced PathMNIST appendix demo

- **Change:** added a second class-balancing run on an artificially imbalanced PathMNIST variant (three classes cut to 10% of training images), alongside the natural-data run that becomes `generated_report_1`.
- **Why:** natural PathMNIST is only mildly imbalanced (max/min 1.63), so augmentation may show no gain over real-only or class-weighted training. The natural run demonstrates the agent judging whether generation is worthwhile; the variant demonstrates the case where it should help.
- **Consequences:** `scripts/make_imbalanced_pathmnist.py` builds the variant reproducibly; the `.npz` loader reads an optional `class_names` key. Validation and test splits are unchanged, so both runs are evaluated on the original distribution.

## 2026-10-17: Revisions after the first demo runs

- **Change:** secondary objectives with per-objective model selection; honest per-model class-conditioning descriptions (scVI replays real cells' posteriors); optional per-class synthetic:real cap for augmentation; a fourth augmentation arm (synthetic + class weighting) with equal training steps; bootstrap intervals and a proportion-based 60/20/20 split for rare classes; per-class class consistency and DCR; unsupervised representation baselines; PBMC 3k cell-type annotations as evaluation and optional modeling labels; scRNA cluster proportions against training proportions with a real-vs-real baseline; exports anchored next to `runs/`.
- **Why:** the PathMNIST, PBMC 3k and imbalanced-PathMNIST runs exposed the defects in `plans/open-issues.md` (entries 1-9): a circular representation baseline, mixed objectives forced into one, proportion distortion traced to a defective latent-diffusion prior (samples at ~70% of the latents' spread), an uncapped 15:1 synthetic ratio that lost to class weighting, pooled metrics hiding rare-class failures, and working-directory-dependent export paths.
- **Consequences:** plans gain `objective.secondary`, per-candidate `use_labels` and `partial_ok`, and `max_synthetic_per_real`; `select` records a model per objective; the demo plan changes to PBMC 3k for class balancing and PathMNIST for generation + representation; the imbalanced variant is dropped. scANVI is recorded as the missing class-conditional scVI-family model (open issue 10).
