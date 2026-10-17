# Plan: one model per algorithm across data types, plus PPCA and GPLVM

## Context
Today the DDPM, conditional VAE and GAN are each tied to one data type (`ddpm_image`, `cvae_image`, `dcgan_image`, `wgan_gp_scrna`). The algorithm itself doesn't depend on the data type. What ties each model to one type is (a) the network (conv vs MLP) and (b) the data conversion and likelihood. `nsf_flow` already shows the target pattern: one model, with per-modality `_to_space`/`_from_space` (`src/genagent/models/flow.py:44-61`).

Goal:
- `ddpm`, `cvae` and `gan` each work on images, scRNA and tabular data.
- `gan` gets a `loss` setting: `vanilla`, `hinge` or `wgan_gp`.
- Add two models, `ppca` (closed form) and `gplvm` (GPyTorch), both on all three data types.
- The old names stay usable as aliases.

This is a design change to the model registry recorded in `plans/notes.md`. The course deadline is 2026-10-19, so the work is staged so that every stage is a working, committed state.

Agreed with the user:
- New names, with the old names kept as aliases.
- WGAN is a `loss` setting of `gan`, not a separate entry; the vanilla objective is included.
- GPLVM uses GPyTorch.
- DDPM, cVAE and GAN cover all three data types.

## Design

### Shared pieces (new)
- **`src/genagent/models/spaces.py`**, `VectorSpace(modality, kind)`: converts native data to float vectors and back.
  - It takes over `NSFFlow._to_space/_from_space` and reuses `scrna_lognorm`, `scrna_counts_from_lognorm` and `TabularCodec` (`src/genagent/embed.py`), plus `_sample_libsizes` (`base.py:141`).
  - Conversions:
    - scRNA: log-normalized expression (optional PCA). Back to counts by Poisson sampling at sampled library sizes.
    - Tabular: codec vectors. Exposes `cat_blocks` (one slice per one-hot block). Back by `codec.decode`.
    - Images: flattened pixels (used only by PPCA and GPLVM), clipped to [0, 1] on the way back.
  - Optional standardization.
- **`src/genagent/models/nets.py`**: the vector backbones moved out of `scrna.py`: `_MLP`, `_ResBlock`, `_MLPDenoiser`. The image conv nets (`_CVAENet`, `_Gen`, `_Disc`) move here too.
- **Diffusion core** (in `ddpm.py`): `train_denoiser(...)` and `sample_denoiser(...)`, built from the existing loops (classifier-free label drop, EMA with warm-up, deadline checks, DDIM or ancestral sampling).

### Registry entries after the change
| Entry | Images | scRNA | Tabular | Capabilities |
|---|---|---|---|---|
| `ddpm` | diffusers U-Net, DDIM sampling (as now) | `_MLPDenoiser` on standardized log-normalized genes, ancestral DDPM sampling | `_MLPDenoiser` on standardized codec vectors, argmax decode | sample, anomaly (denoising error) |
| `cvae` | conv net, Bernoulli (as now) | MLP encoder on log1p counts; decoder gives gene fractions × library size, NB likelihood with learned per-gene dispersion (scVI-like, but label-conditioned) | MLP; Gaussian (learned per-column σ) for numeric columns + cross-entropy per categorical block; categories sampled from the softmax | sample, encode, anomaly (negative ELBO) |
| `gan` (`loss`: `vanilla`, `hinge`, `wgan_gp`) | conv G/D, projection discriminator; spectral norm unless `wgan_gp` | MLP G/D with one-hot label input, ReLU output on log-normalized expression | MLP G/D; Gumbel-softmax (τ = 0.2) on categorical blocks so fake one-hots look like real ones | sample |
| `ppca` | flattened pixels | log-normalized expression | codec vectors | sample, encode, anomaly (exact log-likelihood) |
| `gplvm` | pixels → PCA-50 | log-normalized → PCA-50 | codec vectors | sample, encode, anomaly (approximate) |
| unchanged | — | `scvi`, `latent_diffusion_scvi` (now uses the shared diffusion core), `nsf_flow` | `ctgan`, `tvae`, `nsf_flow` | |

GAN losses:
- `vanilla`: the original GAN objective with the non-saturating generator loss.
- `hinge`: as in `dcgan_image` today.
- `wgan_gp`: as in `wgan_gp_scrna` today, with no spectral norm and a gradient penalty.
- Defaults are `hinge` for images and `wgan_gp` for scRNA and tabular (both are the configurations tested so far). `loss` can be searched by Optuna as a categorical setting.

### PPCA (`models/ppca.py`, numpy, closed form, Tipping & Bishop 1999)
- **Unlabeled fit:** μ = mean. The eigendecomposition of the sample covariance (via SVD when d > n) gives σ² = mean of the discarded eigenvalues and W = U_q(Λ_q − σ²I)^½.
- **Labeled fit:** class-specific means μ_c with a shared W and σ² fitted to the pooled within-class covariance. This stays well-posed for rare classes (PBMC has 9 megakaryocytes in training).
- **Sampling:** x = μ_c + Wz + σε.
- **`encode`:** posterior mean M⁻¹Wᵀ(x − μ), with M = WᵀW + σ²I.
- **`anomaly_score`:** −log N(x; μ, WWᵀ + σ²I) using the Woodbury identity and the matrix determinant lemma. With labels, it is a log-sum-exp over classes, weighted by `class_freq`.
- **Settings:** `n_components` (searchable).

### GPLVM (`models/gplvm.py`, GPyTorch Bayesian GPLVM)
- **Model:** `gpytorch.models.gplvm.BayesianGPLVM` with `VariationalLatentVariable` (initialized from PCA), shared inducing points, an RBF/ARD kernel, independent outputs and a Gaussian likelihood. Trained on minibatches with `VariationalELBO`, checking the deadline.
- **Size limits:** training uses at most `max_points` (default 5000, stratified) points, because every point gets its own latent parameters.
- **Labels:** not used in training. For sampling, a Gaussian is fitted per class to the learned latent means, with its covariance shrunk toward the pooled covariance. Without labels, one Gaussian is fitted to all latent means, which avoids the prior mismatch scVI has.
- **Sampling:** z from that Gaussian → sample from the GP predictive → add likelihood noise → convert back with `VectorSpace.decode`.
- **`encode` and `anomaly_score` for new data:** the GP is held fixed and latent point estimates are optimized for a fixed number of Adam steps. The anomaly score is the negative predictive log-likelihood at that latent point. Both are documented as approximations.
- **Settings:** `latent_dim`, `n_inducing`, `lr`, `epochs`, `max_points`, `output_pca`.
- **Dependency:** add `gpytorch` with `uv add gpytorch`. If it can't be installed with our torch ≥ 2.14 / numpy 2 pins, stop and ask the user.

### Aliases
- `ALIASES` in `base.py` maps old names to new entries with fixed settings:
  - `ddpm_image` → `ddpm`
  - `cvae_image` → `cvae`
  - `dcgan_image` → `gan` with `{"loss": "hinge"}`
  - `wgan_gp_scrna` → `gan` with `{"loss": "wgan_gp"}`
- A new `resolve_model(name) -> (cls, config_overrides)` handles the lookup. The plan step (`pipeline.py:229`) rewrites the model name to the new one, merges the fixed settings, and logs that the alias was resolved. The other lookups (`pipeline.py:396`, `pipeline.py:635`, `report.py:134`) also go through `resolve_model`, so existing run folders still evaluate.
- `REGISTRY` and `models_for` list only the new names, so `init` shows each model once.
- Limitation: `model.pt` files saved under the old classes will no longer load, because the class paths moved. `replay` is unaffected, since it retrains.

### Data-type-specific defaults
`default_config` and `search_space` branch on `ctx.modality`, as `NSFFlow` does now. Image configs keep their current keys and defaults, so the old image behavior is reproduced.

## Stages (each one commits on topic branch `refactor/model-registry`)
1. `spaces.py` and `nets.py`; `flow.py` and `scrna.py` switch to them. No behavior change; tests pass.
2. `ddpm` and the diffusion core; `latent_diffusion_scvi` uses the core.
3. `cvae` (image path unchanged; MLP/NB and MLP/tabular paths new).
4. `gan` with the three losses; scRNA path equal to the old WGAN-GP.
5. Aliases, `resolve_model` wired into pipeline/report; old classes removed from `image.py`/`scrna.py`.
6. `ppca`.
7. `gplvm` plus the gpytorch dependency.
8. Docs and agent guidance (below).

If time runs short, stop after stage 6 and record GPLVM in `plans/open-issues.md` (with the user's confirmation).

## Docs and guidance to update
- `plans/notes.md`: rewrite the model-registry table and note the alias policy. Add a dated entry to `docs/development-log.md`.
- `docs/generative-models.md` (new, starting as `Status: Draft`, then `Current`): the algorithm × network × data-conversion structure, the vector likelihoods, the GAN losses, the PPCA closed form and class-mean variant, GPLVM training/sampling/encoding and its approximations. Link it from `docs/overview.md`.
- `docs/overview.md`: the models bullet and the limitations.
- `skills/run/reference/decision-guide.md`: rewrite the model-choice table with the new names, when to use `ppca` (a fast, exact-likelihood linear baseline) and `gplvm` (smooth nonlinear manifold; small n), GAN `loss` guidance, and the weak spots: DDPM directly on genes is data-hungry, and tabular DDPM/GAN relax categories crudely compared with CTGAN.
- `skills/run/SKILL.md` and `reference/plan-schema.md` where they name models; `README.md` if it lists models.

## Verification
- `uv run pytest` (unit + smoke).
  - Smoke tests switch to the new names, with one plan that still uses `ddpm_image`/`wgan_gp_scrna` to test aliases.
  - Add tiny smoke flows: tabular (`cvae`, `ddpm`, `gan`, `ppca`) and scRNA (`cvae`, `ddpm`, `gplvm`).
- New unit tests:
  - PPCA recovers a planted subspace and σ² on synthetic Gaussian data.
  - The PPCA log-likelihood equals `scipy.stats.multivariate_normal.logpdf` with the implied covariance.
  - The class-mean PPCA reduces to plain PPCA with one class.
  - Alias resolution merges the fixed settings.
  - `VectorSpace` encode→decode round-trips for tabular data and maps scRNA to count shape.
  - The GAN runs one training step for each of `vanilla`, `hinge` and `wgan_gp` on both networks.
- Regression check on real data (1–3 min budgets, CLI flow):
  - PathMNIST subset with `ddpm`, `cvae` and `gan` (hinge). Compare MMD/FID with the same plan run on `main` before the refactor; they should match within run-to-run noise.
  - PBMC 3k with `cvae`, `ppca`, `gplvm` and `latent_diffusion_scvi`; check that the metrics are reasonable and that per-class sampling works for rare classes.
- `genagent registry` lists each model once, with its modalities and default configs.
