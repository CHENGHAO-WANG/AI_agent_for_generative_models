# Development log

## 2026-10-16: Imbalanced PathMNIST appendix demo

- **Change:** added a second class-balancing run on an artificially imbalanced PathMNIST variant (three classes cut to 10% of training images), alongside the natural-data run that becomes `generated_report_1`.
- **Why:** natural PathMNIST is only mildly imbalanced (max/min 1.63), so augmentation may show no gain over real-only or class-weighted training. The natural run demonstrates the agent judging whether generation is worthwhile; the variant demonstrates the case where it should help.
- **Consequences:** `scripts/make_imbalanced_pathmnist.py` builds the variant reproducibly; the `.npz` loader reads an optional `class_names` key. Validation and test splits are unchanged, so both runs are evaluated on the original distribution.

## 2026-10-17: Revisions after the first demo runs

- **Change:** secondary objectives with per-objective model selection; honest per-model class-conditioning descriptions (scVI replays real cells' posteriors); optional per-class synthetic:real cap for augmentation; a fourth augmentation arm (synthetic + class weighting) with equal training steps; bootstrap intervals and a proportion-based 60/20/20 split for rare classes; per-class class consistency and DCR; unsupervised representation baselines; PBMC 3k cell-type annotations as evaluation and optional modeling labels; scRNA cluster proportions against training proportions with a real-vs-real baseline; exports anchored next to `runs/`.
- **Why:** the PathMNIST, PBMC 3k and imbalanced-PathMNIST runs exposed the defects in `plans/open-issues.md` (entries 1-9): a circular representation baseline, mixed objectives forced into one, proportion distortion traced to a defective latent-diffusion prior (samples at ~70% of the latents' spread), an uncapped 15:1 synthetic ratio that lost to class weighting, pooled metrics hiding rare-class failures, and working-directory-dependent export paths.
- **Consequences:** plans gain `objective.secondary`, per-candidate `use_labels` and `partial_ok`, and `max_synthetic_per_real`; `select` records a model per objective; the demo plan changes to PBMC 3k for class balancing and PathMNIST for generation + representation; the imbalanced variant is dropped. scANVI is recorded as the missing class-conditional scVI-family model (open issue 10).

## 2026-10-17: One model per algorithm across modalities; PPCA and GPLVM

- **Change:** the per-modality generators `ddpm_image`, `cvae_image`, `dcgan_image` and `wgan_gp_scrna` became `ddpm`, `cvae` and `gan`, each serving images, scRNA and tabular data; `gan` takes `loss` (`vanilla`, `hinge`, `wgan_gp`). Added `ppca` (closed form) and `gplvm` (GPyTorch Bayesian GPLVM), both on all modalities. The old names resolve as aliases. Plan: `plans/2026-10-17-model-registry-refactor.md`.
- **Why:** the algorithms do not depend on the data type; only the network (conv vs MLP) and the representation/likelihood do, as `nsf_flow` already showed. Separating them gives a label-conditioned VAE for tables (TVAE's classes come from rejection sampling) and for scRNA counts (scVI has no label input), and lets the agent compare GAN losses. PPCA adds a fast exact-likelihood linear baseline; GPLVM a non-linear latent model for small data.
- **Consequences:** new shared modules `models/spaces.py` (vector representations) and `models/nets.py` (backbones); latent diffusion reuses the DDPM core. A pre/post comparison on fixed PathMNIST and PBMC 3k subsets reproduced the old image and scRNA models within run-to-run noise (WGAN-GP and latent diffusion bit-identical). The decision guide asks the agent to add a vanilla-loss `gan` baseline whenever it uses `gan`. New dependency: `gpytorch`. Saved `model.pt` files from older runs no longer unpickle (classes moved); `replay` retrains and is unaffected.

