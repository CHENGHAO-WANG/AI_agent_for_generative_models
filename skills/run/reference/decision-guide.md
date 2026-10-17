# Decision guide

Reference for the planning, evaluation, refinement, and selection steps. These are defaults with reasons; depart from them when the profile or results argue otherwise, and log why.

## Interpreting the objective

| User says (examples) | `objective.type` | Notes |
|---|---|---|
| nothing | `generation` | Generate `n` samples that follow the real class distribution. |
| "more data", "synthetic copy", "share data", "privacy" | `generation` | For sharing/privacy, weight DCR and C2ST more. |
| "balance", "rare classes", "minority", "imbalanced" | `class_balancing`, mode `augmentation` | Synthetic data is added to real data. |
| "balanced synthetic dataset", "replace the real data", "shareable balanced set" | `class_balancing`, mode `synthetic_balanced` | Fully synthetic, `n` per class. |
| "embedding", "latent space", "clusters", "cell states", "features", "structure" | `representation` | Needs a model with an encoder. |
| "outliers", "anomalies", "abnormal", "QC failures", "novelty" | `anomaly_detection` | Needs a likelihood or reconstruction score. |
| "fill missing", "impute" | `imputation` | Declined; say what would be used (e.g. scVI denoised expression, a VAE conditioned on observed entries). |
| "confidence", "uncertainty", "error bars" | `uncertainty_quantification` | Declined; say what would be used (posterior sampling, ensembles). |

**Several goals in one request** ("generate cells and learn a latent space"): make the goal the user's wording leads with the primary `objective.type` and list the others in `objective.secondary`. Every objective is evaluated and gets a downstream analysis; the primary decides the allocation of samples and weighs double in the combined score. Ask which goal is primary only when the wording gives no order and the choice changes the candidates.

**Labels.** `--label` names the modeling labels. Built-in `pbmc3k` carries scanpy-tutorial cell types in `cell_type`: they are always used as evaluation labels, and become modeling labels only with `--label cell_type` (needed for class balancing on cell types).

**Meaning of `n`:**
- generation, representation, anomaly detection: total synthetic samples, split proportionally across classes (optional for the last two).
- augmentation: per-class target; each class below `n` is filled up to `n`. `n` may not exceed the majority-class size unless the user chooses "keep". `max_synthetic_per_real` (optional) caps each class's synthetic:real ratio, so a capped class stays below `n`.
- synthetic_balanced: samples per class.

**Whether to generate.** Generating is a decision, not a default:
- Augmentation when the imbalance ratio (max/min class count) is below ~1.5 has little room to help; still run it if asked, but plan to report the non-generative alternative (class-weighted loss, which `finalize` always evaluates) as the likely better choice.
- Representation and anomaly detection need no samples; generate only if `n` was given.
- Very small data (< ~300 samples): generative models will mostly memorize; prefer simple models, watch DCR, and say so.

**Rare classes in augmentation.** A conditional model learns a class from that class's training examples, with the shared network supplying the rest. With only tens of examples, synthetic samples of the class mostly re-express those examples, and filling the class to `n` can make it 10-15x synthetic. Real + synthetic then often loses to class-weighted training (observed: 0.750 vs 0.782 balanced accuracy at 15:1). For classes with fewer than ~1,000 real samples, set `max_synthetic_per_real` (2-5 is a reasonable range) and say why; `init` flags classes with fewer than 50 training samples, which are too few to evaluate or to learn a distribution from.

## Asking the user

Ask only at the planning step, in one AskUserQuestion batch of at most 3 questions, each with your recommended option first. Ask only when the answer changes the plan:
- `init` flags an ambiguous label column.
- Augmentation `n` exceeds the majority-class size: options "Clamp to <majority>", "Keep n (augment every class)", "Enter a new n". Record the answer in `questions` and set `n_exceeds_policy` / `n`.
- A candidate you want cannot serve a secondary objective (e.g. a flow has no encoder for representation): "Include <candidate>? It would compete only for <objectives>." On yes, set `partial_ok: true`; on no, drop it.
- The objective wording gives no order between two goals that need different candidates.
- The budget is clearly too small for the objective (e.g. diffusion on > 50k images in < 10 minutes).

With `--no-questions`: take the recommended default, record it with `"answered_by": "default"`; partial candidates are included and flagged; an augmentation `n` above the majority-class size stops the run with the error from `GA plan` (report it to the user).

## Choosing candidates

Pick 2-3 candidates from different families so the comparison is informative. Starting points:

| Modality | Objective | Recommended candidates | Why |
|---|---|---|---|
| image | generation / class balancing | `ddpm_image`, `dcgan_image`, `cvae_image` | Diffusion is the fidelity reference; GAN is fast and sharp; VAE is the stable fallback. Class-conditional generation matters most for balancing. |
| image | representation | `cvae_image` with `use_labels: false` | Only image model with an encoder; without labels its latent must encode class structure itself. Compared with unsupervised baselines (pixel PCA, autoencoder). |
| image | anomaly detection | `cvae_image`, `ddpm_image` | ELBO vs denoising error. |
| scrna | generation | `latent_diffusion_scvi`, `scvi`, `nsf_flow` (or `wgan_gp_scrna` for > 10k cells) | Count likelihood (scVI) is the right noise model; latent diffusion learns the latent distribution, fixing scVI's prior mismatch (scVI's N(0, I) prior over-represents the dominant population); the flow is a fast likelihood baseline. WGAN-GP is data-hungry. |
| scrna | class balancing | `latent_diffusion_scvi`, `nsf_flow`, `scvi` as a baseline | Latent diffusion and the flow are truly label-conditioned. `scvi` has no label input: it "replays posteriors of real cells of the class", a weak choice for very rare classes. |
| scrna | representation | `scvi`, `latent_diffusion_scvi` | Both give scVI latents; compare with the PCA baseline. |
| scrna | anomaly detection | `scvi`, `nsf_flow` | Reconstruction NLL vs PCA-space likelihood. |
| tabular | any generative | `ctgan`, `tvae`, `nsf_flow` | CTGAN for many/imbalanced categoricals (`log_frequency: true` samples rare categories more often in training), TVAE for small tables, flow for mostly numeric tables and likelihoods. |

Each model's `conditional` field (from `init`) says how it uses labels; quote it when it matters for the objective. A candidate's `use_labels` overrides the plan's: a representation candidate usually trains without labels even when the generation candidates use them.

**Time fractions.** Give diffusion the largest share (it is ~5-10x slower per epoch than a GAN or VAE): e.g. DDPM 0.5, GAN 0.3, VAE 0.2 on images. Keep at least a few epochs' worth of time for every candidate.

**Configuration from the profile.**
- Epochs: aim for a few thousand optimizer steps at minimum (steps per epoch = n_train / batch_size); the deadline caps long runs, so overshooting epochs is safe, undershooting wastes budget.
- Small data (< 5k): smaller networks (VAE latent 16, flow hidden 64-128, scVI n_hidden 128), more epochs, keep early stopping.
- Large images (> 50k): DDPM `base_channels` 64, `layers_per_block` 1 (≈ 45 s/epoch on 90k 28×28 images on an RTX 3060 Ti); `guidance_scale` 1.5-3 for class-conditional generation.
- scRNA: `n_latent` 10 (≤ 5k cells) to 30 (large atlases); flow `n_components` 20-50; latent diffusion `diff_steps` 6,000-15,000.
- Search: use `"default"` spaces with 3-5 trials when the candidate has ≥ 5 minutes; give fast models (VAE, flow, small GAN) searches first; skip the search for a candidate whose time is short.

**Metric weights.** Defaults depend on the objective and multiply modality-specific metrics by 2; the combined score adds half of each secondary objective's weights. Change them only for a stated reason, e.g. privacy-oriented sharing: raise `c2st` and keep the DCR guardrail.

## Reading the evaluation

General metrics compare synthetic data with the real **test** split in the shared embedding. Per-class checks use a class-balanced probe (200 samples per class), so rare classes are judged on their own numbers.

| Metric | Good | Warning sign |
|---|---|---|
| precision (fidelity) | > 0.6 | < 0.3: many off-manifold samples |
| recall / coverage (diversity) | close to precision | much lower than precision: mode collapse (common for GANs) |
| density | ≈ 1 | > 1.5: samples crowd dense regions |
| MMD vs `mmd_baseline_real` | within ~2-3× baseline | > 10× baseline: distribution mismatch |
| C2ST AUC | < 0.7 | > 0.95: trivially distinguishable; synthetic data unfit as a stand-in |
| DCR (`dcr_frac_below_holdout_q05`) | ≈ 0.05 | > 0.15 or exact copies: memorization (auto-excluded) |
| per-class DCR (`per_class`, `memorization_flagged_classes`) | no class flagged | a flagged class: its synthetic samples copy its training samples; in class balancing the whole candidate is excluded |
| TSTR ratio | > 0.8 | < 0.5: class information lost |
| FID (feature space) vs baseline | within ~2× baseline | > 5× baseline |
| class consistency (`class_consistency_balanced`, `class_consistency_min`, `per_class`) | near the classifier's accuracy on real data | a class near chance (1 / n_classes): conditioning failed for that class |
| gene mean / var / zero-fraction corr (scRNA) | > 0.95 / > 0.85 / > 0.9 | var corr < 0.7: wrong overdispersion |
| `cluster_tvd` vs `cluster_tvd_baseline_real` (scRNA) | within ~2× the real-vs-real baseline | far above it: group proportions distorted; check which groups in the proportions table |
| marker_expr_corr (scRNA) | > 0.8 | < 0.5: cell identities blurred |
| latent kNN acc / ARI | ≥ the unsupervised baselines | below them: the generative latent adds nothing |

A baseline marked `circular_reference_labels` was scored against labels derived from its own embedding (Leiden clusters on PCA); treat that comparison as inconclusive. Look at the figures too: blurry but correct samples (VAE) and sharp but repetitive samples (GAN) can score alike.

**Downstream intervals.** Augmentation results carry bootstrap 95% intervals per arm and per class, and for each arm's difference from real-only. Claim a gain only when the difference interval excludes 0; call a class inconclusive when its recall interval is wider than ~0.5 or its test count is in single digits.

## Refinement triggers

Use the one refinement round when one of these holds, targeting the cause:
- Best candidate stopped by budget with loss still falling → more time (`time_scale` up to 1.0) or a smaller network.
- GAN with recall ≪ precision (mode collapse) or diverging losses → lower learning rates, more critic steps / lower `lr_d`.
- Class consistency near chance for a conditional model, overall or for one class → raise `guidance_scale` (diffusion), train longer.
- VAE with blurry samples and high C2ST → lower `beta`, larger `latent_dim`.
- All candidates fail the objective's primary metric → refine the closest one; say in the report if it still fails.

Do not refine when the best candidate meets the "good" column for its objective's primary metrics; log that instead.

## Selecting the final models

Each objective gets a final model, chosen from the candidates that serve it. Default (`GA select` without options): the best combined-score candidate serves every objective it can, unless another candidate beats it on that objective by more than 0.03, so one model is used where it is nearly as good. Override when the objective's primary evidence disagrees with the score (e.g. for augmentation, clearly better per-class consistency beats better unconditional fidelity) or when figures reveal a failure the metrics miss: `--name` forces one model for all objectives it serves, `--assign <objective>=<candidate>` sets one objective. When different models serve different objectives, say so in the report: synthetic samples then do not come from the model whose latent space is delivered. Never select a candidate flagged for memorization.
