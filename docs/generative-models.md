Status: Current

# Generative models

How the registry's models are built, what they assume, and where their approximations lie. Code: `src/genagent/models/`. Decision heuristics for the agent: `skills/run/reference/decision-guide.md`.

## Structure: algorithm × network × data representation

The algorithm itself (diffusion, VAE, GAN, PPCA, GPLVM, flow) is independent of the data type. What changes with the data is:

1. **Network**: images use conv nets (`nets.py`: `ConvVAENet`, `ConvGenerator`, `ConvDiscriminator`, the diffusers U-Net); scRNA and tabular data are flat vectors and use MLPs (`MLP`, `MLPDenoiser`, `VectorVAENet`).
2. **Representation and likelihood**: `spaces.VectorSpace` maps native samples to float vectors and back:

| Modality | Vector | Back to native |
|---|---|---|
| image | flattened pixels in [0, 1] (PPCA, GPLVM only; DDPM/cVAE/GAN use the image tensor) | clip to [0, 1], reshape |
| scrna | log-normalized expression `log1p(x / lib · 1e4)`, optionally PCA | rescale to a library size drawn from training cells of the same class, Poisson counts (`embed.scrna_counts_from_lognorm`) |
| tabular | `TabularCodec`: standardized numeric columns, then one one-hot block per categorical column | un-standardize, argmax per block |

Each model's `default_config` and `search_space` depend on the modality. Former per-modality names (`ddpm_image`, `cvae_image`, `dcgan_image`, `wgan_gp_scrna`) are aliases (`base.ALIASES`, `base.resolve_model`): `plan` rewrites them to the current model, merges the config they fix (e.g. `loss`), and keeps the old name as the candidate name.

## DDPM (`ddpm.py`)

Epsilon-prediction diffusion with 1000 training timesteps, classifier-free guidance (labels dropped to a null class with probability `p_uncond`) and an EMA of the weights with warm-up. The training and sampling loops (`train_denoiser`, `sample_denoiser`, `denoising_error`) are shared with `latent_diffusion_scvi`.

- **image**: diffusers `UNet2DModel` on pixels scaled to [-1, 1]; epoch-based training; DDIM sampling (`sampling_steps`, default 50).
- **scrna / tabular**: `MLPDenoiser` (residual MLP; the time and label embeddings modulate every block) on standardized vectors (scRNA: each gene centered, all divided by one shared standard deviation; per-gene standardization turned genes expressed in a few cells into values up to 40 SD, and sample FID on PBMC 3k fell from 343 to 150 with the shared scale); step-based training (`diff_steps`) with a cosine learning-rate decay; ancestral DDPM sampling (1000 steps), which matched the spread of the data better than DDIM for vectors; samples are clipped to ±8 standard deviations. The scheduler does not clip to [-1, 1].
- **Anomaly score**: unconditional noise-prediction error averaged over t ∈ {50, 200, 400}.
- **Limitations**: on scRNA the model works on ~2000 log-normalized genes directly and is data-hungry (PBMC 3k, ~1.6k training cells: far behind `latent_diffusion_scvi`); on tables, one-hot blocks are diffused as continuous values and decoded by argmax (a relaxation, unlike TabDDPM's multinomial diffusion).

## Conditional VAE (`cvae.py`)

Encoder q(z | x, c) and decoder p(x | z, c), with the label embedding fed to both; labels are dropped to the null class with probability 0.1 so label-free encoding and scoring stay in distribution. Loss = reconstruction NLL + β · KL.

| Modality | Network | Likelihood p(x \| z, c) |
|---|---|---|
| image | conv | Bernoulli per pixel |
| scrna | MLP on log1p counts | negative binomial with mean = softmax(decoder) · observed library size and learned per-gene inverse dispersion θ (as in scVI, but label-conditioned) |
| tabular | MLP on codec vectors | Gaussian with learned per-column σ on standardized numeric columns + categorical cross-entropy per one-hot block |

Vector models use a linear KL warm-up over the first `kl_warmup` fraction of epochs and keep the checkpoint with the best validation negative ELBO. Sampling: z ~ N(0, I), then a draw from the likelihood (NB as a Gamma-Poisson mixture at a library size from training cells of the class; numeric columns with their σ; categories from the softmax). `encode` returns the posterior mean with the null label; the anomaly score is the negative ELBO (single-sample estimate).

## GAN (`gan.py`)

Conditional GAN with `loss`:

| `loss` | Discriminator loss | Generator loss | Regularization |
|---|---|---|---|
| `vanilla` | −log σ(D(x)) − log(1 − σ(D(G(z)))) | −log σ(D(G(z))) (non-saturating) | spectral norm |
| `hinge` | max(0, 1 − D(x)) + max(0, 1 + D(G(z))) | −D(G(z)) | spectral norm |
| `wgan_gp` | D(G(z)) − D(x) + λ (‖∇D(x̂)‖ − 1)² on interpolates x̂ | −D(G(z)) | gradient penalty; layer norm instead of spectral norm in the MLP critic |

The generator steps once per `n_critic` discriminator steps. Defaults: `hinge` for images, `wgan_gp` for scRNA and tables. The agent adds a `vanilla` candidate as a baseline whenever it uses `gan` (decision guide).

- **image**: DCGAN-style generator (tanh output) and projection discriminator.
- **scrna**: scGAN-style MLPs on log-normalized expression (ReLU output); one-hot label concatenated to both inputs; counts by Poisson sampling.
- **tabular**: MLPs on codec vectors; each one-hot block of the generator output passes through Gumbel-softmax (τ = 0.2), so generated categories look like real one-hots; decoded by argmax.

No likelihood or encoder.

## Probabilistic PCA (`ppca.py`)

Model (Tipping & Bishop, 1999): x = μ_c + W z + σ ε with z ~ N(0, I_q), ε ~ N(0, I_d), so x ~ N(μ_c, C) with C = W Wᵀ + σ² I. Vectors are not re-standardized (pixels and codec vectors share a scale; log-normalized genes keep their variances).

**Closed-form fit.** With residuals R (x − μ, or x − μ_{c(x)} when labeled) and S = Rᵀ R / n with leading eigenpairs (λ_i, u_i), the maximum-likelihood solution is

- σ² = (tr S − Σ_{i≤q} λ_i) / (d − q), the mean of the discarded eigenvalues,
- W = U_q (Λ_q − σ² I)^{1/2} (up to rotation).

S is accumulated in chunks when n ≥ d; otherwise the eigenpairs come from an SVD of R / √n. **Labels**: each class has its own mean μ_c, while W and σ² are fitted to the pooled within-class covariance; this stays well-posed for classes with a handful of samples (a separate PPCA per class would not). A class absent from training uses the overall mean.

**Likelihood.** C has eigenvalues λ_i on u_i and σ² elsewhere, so log N(x; μ, C) = −½ [d log 2π + Σ log λ_i + (d − q) log σ² + (‖r‖² − Σ (1 − σ²/λ_i)(u_iᵀ r)²) / σ²] with r = x − μ, without forming C. With labels the anomaly score is −log Σ_c π_c N(x; μ_c, C), with π_c the training class frequencies. `encode` returns the posterior mean (Wᵀ W + σ² I)⁻¹ Wᵀ (x − μ̄) with the overall mean μ̄. A unit test checks the likelihood against `scipy.stats.multivariate_normal`.

**Limitations**: Gaussian samples around a linear subspace: no non-linear structure, skew, sparsity or discreteness. A fast baseline with an exact likelihood.

## Bayesian GPLVM (`gplvm.py`, GPyTorch)

Model (Titsias & Lawrence, 2010): each output dimension j is an independent GP f_j(z) of a q-dimensional latent z with a shared ARD-RBF kernel, plus Gaussian noise with a per-output variance. Each output has its own sparse variational GP (inducing points initialized at latent positions of random training points). Every training point has a variational posterior q(z_n) = N(m_n, diag(s_n²)) (GPyTorch: s_n = softplus(q_log_sigma)) and a N(0, I) prior. Training maximizes the ELBO on minibatches (`VariationalELBO` plus the latent KL), with latents initialized at standardized principal components.

- **Vectors**: standardized PCA-50 of pixels (images) or of log-normalized expression (scRNA) by default (`output_pca`); standardized codec vectors for tables. Variation outside the kept components is lost.
- **Size**: at most `max_points` training points (default 5000), sampled proportionally per class with at least 20 per class (each point carries its own latent parameters).
- **Sampling**: latents are drawn from a Gaussian approximation of the aggregate posterior: mean and covariance of the m_n plus the mean posterior variance. With labels, a Gaussian per class, with covariance w S_c + (1 − w) S_pooled (plus the mean posterior variance), w = n_c / (n_c + 10). Outputs are drawn from the GP predictive marginal at each latent (mean, variance, noise), then mapped back with `VectorSpace.decode`. Labels are not used in training, so class samples come from the latent region of that class.
- **Encoding and anomaly score of new data** (approximations): there is no amortized encoder. For each new point, z is initialized at the latent of the nearest training point in output space and optimized for `encode_steps` Adam steps to minimize −log N(x; μ(z), diag(var(z) + noise)) − log N(z; 0, I) under the fixed GP. `encode` returns that z; the anomaly score is the minimized value.
- **Persistence**: the GPyTorch model class is built on first use and exposed as the module attribute `_BGPLVM`, so saved models unpickle.

## Other models

- `scvi`, `latent_diffusion_scvi` (`scrna.py`): scvi-tools scVI; latent diffusion trains the shared DDPM core on standardized scVI latents and decodes through scVI's NB decoder. `scvi` has no label input (posterior replay for class samples).
- `nsf_flow` (`flow.py`): zuko neural spline flow on `VectorSpace` vectors (scRNA: PCA of log-normalized expression; tabular: codec with dequantization noise on one-hot blocks).
- `ctgan`, `tvae` (`tabular.py`): the `ctgan` library; TVAE gets classes by rejection sampling on the generated label column.
