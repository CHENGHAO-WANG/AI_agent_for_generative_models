# Decision guide

Reference for the planning, evaluation, refinement, and selection steps. These are defaults with reasons; depart from them when the profile or results argue otherwise, and log why.

## Interpreting the objective

| User says (examples) | `objective.type` | Notes |
|---|---|---|
| nothing | `generation` | Generate `n` samples that follow the real class distribution. |
| "more data", "synthetic copy", "share data", "privacy" | `generation` | For sharing/privacy, weight DCR and C2ST more. |
| "balance", "rare classes", "minority", "imbalanced" | `class_balancing`, mode `augmentation` | Synthetic data is added to real data. |
| "balanced synthetic dataset", "replace the real data", "shareable balanced set" | `class_balancing`, mode `synthetic_balanced` | Fully synthetic, `n` per class. |
| "embedding", "latent space", "clusters", "cell states", "features" | `representation` | Needs a model with an encoder. |
| "outliers", "anomalies", "abnormal", "QC failures", "novelty" | `anomaly_detection` | Needs a likelihood or reconstruction score. |
| "fill missing", "impute" | `imputation` | Declined; say what would be used (e.g. scVI denoised expression, a VAE conditioned on observed entries). |
| "confidence", "uncertainty", "error bars" | `uncertainty_quantification` | Declined; say what would be used (posterior sampling, ensembles). |

Mixed objectives: pick the one the user's wording puts first and note the others in the interpretation.

**Meaning of `n`:**
- generation, representation, anomaly detection: total synthetic samples, split proportionally across classes (optional for the last two).
- augmentation: per-class target; each class below `n` is filled up to `n`. `n` may not exceed the majority-class size unless the user chooses "keep".
- synthetic_balanced: samples per class.

**Whether to generate.** Generating is a decision, not a default:
- Augmentation when the imbalance ratio (max/min class count) is below ~1.5 has little room to help; still run it if asked, but plan to report the non-generative alternative (class-weighted loss, which `finalize` always evaluates) as the likely better choice.
- Representation and anomaly detection need no samples; generate only if `n` was given.
- Very small data (< ~300 samples): generative models will mostly memorize; prefer simple models, watch DCR, and say so.

## Asking the user

Ask only at the planning step, in one AskUserQuestion batch of at most 3 questions, each with your recommended option first. Ask only when the answer changes the plan:
- `init` flags an ambiguous label column.
- Augmentation `n` exceeds the majority-class size: options "Clamp to <majority>", "Keep n (augment every class)", "Enter a new n". Record the answer in `questions` and set `n_exceeds_policy` / `n`.
- The objective text admits two objective types with different models.
- The budget is clearly too small for the objective (e.g. diffusion on > 50k images in < 10 minutes).

With `--no-questions`: take the recommended default, record it with `"answered_by": "default"`; an augmentation `n` above the majority-class size stops the run with the error from `GA plan` (report it to the user).

## Choosing candidates

Pick 2-3 candidates from different families so the comparison is informative. Starting points:

| Modality | Objective | Recommended candidates | Why |
|---|---|---|---|
| image | generation / class balancing | `ddpm_image`, `dcgan_image`, `cvae_image` | Diffusion is the fidelity reference; GAN is fast and sharp; VAE is the stable fallback. Class-conditional generation matters most for balancing. |
| image | representation | `cvae_image` (+ a `use_labels: false` setting) | Only image model with an encoder. Compare against the embedding baseline. |
| image | anomaly detection | `cvae_image`, `ddpm_image` | ELBO vs denoising error. |
| scrna | generation | `latent_diffusion_scvi`, `scvi`, `nsf_flow` (or `wgan_gp_scrna` for > 10k cells) | Count likelihood (scVI) is the right noise model; latent diffusion fixes scVI's prior mismatch; the flow is a fast likelihood baseline. WGAN-GP is data-hungry. |
| scrna | representation | `scvi`, `latent_diffusion_scvi` | Both give scVI latents; compare with the PCA baseline. |
| scrna | anomaly detection | `scvi`, `nsf_flow` | Reconstruction NLL vs PCA-space likelihood. |
| tabular | any generative | `ctgan`, `tvae`, `nsf_flow` | CTGAN for many/imbalanced categoricals, TVAE for small tables, flow for mostly numeric tables and likelihoods. |

**Time fractions.** Give diffusion the largest share (it is ~5-10x slower per epoch than a GAN or VAE): e.g. DDPM 0.5, GAN 0.3, VAE 0.2 on images. Keep at least a few epochs' worth of time for every candidate.

**Configuration from the profile.**
- Epochs: aim for a few thousand optimizer steps at minimum (steps per epoch = n_train / batch_size); the deadline caps long runs, so overshooting epochs is safe, undershooting wastes budget.
- Small data (< 5k): smaller networks (VAE latent 16, flow hidden 64-128, scVI n_hidden 128), more epochs, keep early stopping.
- Large images (> 50k): DDPM `base_channels` 64, `layers_per_block` 1 (≈ 45 s/epoch on 90k 28×28 images on an RTX 3060 Ti); `guidance_scale` 1.5-3 for class-conditional balancing.
- scRNA: `n_latent` 10 (≤ 5k cells) to 30 (large atlases); flow `n_components` 20-50.
- Search: use `"default"` spaces with 3-5 trials when the candidate has ≥ 5 minutes; give fast models (VAE, flow, small GAN) searches first; skip the search for a candidate whose time is short.

**Metric weights.** Defaults depend on the objective and multiply modality-specific metrics by 2. Change them only for a stated reason, e.g. privacy-oriented sharing: raise `c2st` and keep the DCR guardrail; balancing on images: `class_consistency` and `tstr` matter most.

## Reading the evaluation

All general metrics compare synthetic data with the real **test** split in the shared embedding.

| Metric | Good | Warning sign |
|---|---|---|
| precision (fidelity) | > 0.6 | < 0.3: many off-manifold samples |
| recall / coverage (diversity) | close to precision | much lower than precision: mode collapse (common for GANs) |
| density | ≈ 1 | > 1.5: samples crowd dense regions |
| MMD vs `mmd_baseline_real` | within ~2-3× baseline | > 10× baseline: distribution mismatch |
| C2ST AUC | < 0.7 | > 0.95: trivially distinguishable; synthetic data unfit as a stand-in |
| DCR (`dcr_frac_below_holdout_q05`) | ≈ 0.05 | > 0.15 or exact copies: memorization (auto-excluded) |
| TSTR ratio | > 0.8 | < 0.5: class information lost |
| FID (feature space) vs baseline | within ~2× baseline | > 5× baseline |
| class_consistency_acc (images) | near the classifier's real accuracy (~0.85) | near chance (1 / n_classes): conditioning failed |
| gene mean / var / zero-fraction corr (scRNA) | > 0.95 / > 0.85 / > 0.9 | var corr < 0.7: wrong overdispersion |
| cluster_tvd (scRNA) | < 0.1 | > 0.3: cell-type proportions distorted; check rare clusters |
| marker_expr_corr (scRNA) | > 0.8 | < 0.5: cell identities blurred |
| latent kNN acc / ARI | ≥ the embedding baseline | below baseline: the generative latent adds nothing |

Look at the figures too: blurry but correct samples (VAE) and sharp but repetitive samples (GAN) can score alike.

## Refinement triggers

Use the one refinement round when one of these holds, targeting the cause:
- Best candidate stopped by budget with loss still falling → more time (`time_scale` up to 1.0) or a smaller network.
- GAN with recall ≪ precision (mode collapse) or diverging losses → lower learning rates, more critic steps / lower `lr_d`.
- Class consistency near chance for a conditional model → raise `guidance_scale` (diffusion), train longer.
- VAE with blurry samples and high C2ST → lower `beta`, larger `latent_dim`.
- All candidates fail the objective's primary metric → refine the closest one; say in the report if it still fails.

Do not refine when the best candidate meets the "good" column for its objective's primary metrics; log that instead.

## Selecting the final model

Default: the top of the score ranking. Override when the objective's primary evidence disagrees with the aggregate score (e.g. for augmentation, the candidate with clearly better class consistency and TSTR beats one with better unconditional fidelity), or when figures reveal a failure the metrics miss. Never select a candidate flagged for memorization.
