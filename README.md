# genagent: an AI agent for generative modeling

Given a dataset and an optional objective, the agent profiles the data, interprets the objective, decides whether and how to apply generative modeling, selects and tunes models (VAE, GAN, normalizing flow, diffusion), trains them, evaluates the synthetic data, recommends how to use it, and writes a report.

It is a **Claude Code plugin** (Claude makes the decisions) on top of a **Python toolkit** (`genagent`, which does the computation). See [docs/overview.md](docs/overview.md) for the architecture.

## Setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), an NVIDIA GPU with a CUDA 13 driver (CPU works but is slow), Claude Code.

```sh
uv sync                          # creates .venv with CUDA PyTorch and all dependencies from uv.lock
uv run pytest                    # optional: unit + smoke tests (~2 min)
```

Install the plugin from this repository (inside Claude Code):

```
/plugin marketplace add <path-to-this-repo>
/plugin install genagent@genagent-marketplace
```

## Usage (single command)

Run Claude Code in this repository and type:

```
/genagent:run <data> [n] [objective text] [--label <col>|none] [--budget <minutes>] [--no-questions] [--export <name>]
```

Examples:

```
/genagent:run pathmnist 12885 balance the tissue classes --export generated_report_1
/genagent:run pbmc3k 2638 generate synthetic cells and learn a latent space of cell states --export generated_report_2
/genagent:run data/my_table.csv 5000
```

- `<data>`: `pathmnist`, `pbmc3k` (downloaded automatically), or a path to `.npz`, `.h5ad`, `.csv`/`.parquet`, or a folder of images with one subfolder per class.
- `n`: number of synthetic samples. For augmentation-style class balancing it is a per-class target (every class with fewer than `n` samples is filled up to `n`).
- Objective: free text; omitted means plain data generation.
- The agent may ask up to three multiple-choice questions while planning; `--no-questions` accepts its defaults.

Each run writes `runs/<dataset>_<timestamp>/` with the synthetic data, metrics, figures, the decision log (`decisions.json`), and `report.md` / `report.html`. `--export <name>` copies the report and artifacts (without model weights) to `outputs/<name>_<YYYYmmdd-HHMMSS>/`, with the report files named `<name>.md` / `<name>.html`; repeated exports never overwrite each other. `<name>` must be a plain name, not a path.

## Re-running without the agent

Every run logs the exact toolkit commands and the agent's plan, so it can be re-executed deterministically without Claude:

```sh
uv run genagent replay runs/<run-id> --dst runs/<run-id>_replay
```

The toolkit commands can also be used directly: `uv run genagent --help`.

## Demo outputs

- `outputs/generated_report_1_<timestamp>/` — PathMNIST, class balancing by augmentation
- `outputs/generated_report_2_<timestamp>/` — PBMC 3k, generation + representation learning
