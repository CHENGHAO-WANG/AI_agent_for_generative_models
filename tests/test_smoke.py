"""End-to-end smoke tests: the full tool sequence on tiny data for each modality, plus replay."""

import json

import pytest

from genagent import cli

NARRATIVE = {
    "summary": "s", "objective_interpretation": "o", "model_choice": "m", "hyperparameters": "h",
    "evaluation": "e", "downstream": "d", "limitations": ["l"],
    "recommendations": {"suitable_uses": ["u"], "avoid": ["a"], "caveats": ["c"], "synthetic_to_real_ratio": "1:1"},
}


def run_cli(*args):
    assert cli.main([str(a) for a in args]) == 0


def write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


def full_flow(tmp_path, data, plan, n=60):
    rd = tmp_path / "run"
    run_cli("init", "--data", data, "--n", n, "--budget", 1, "--run-dir", rd)
    run_cli("plan", "--run", rd, "--file", write(tmp_path, "plan.json", plan))
    run_cli("train", "--run", rd)
    run_cli("evaluate", "--run", rd)
    run_cli("select", "--run", rd, "--rationale", "test")
    run_cli("finalize", "--run", rd)
    run_cli("report", "--run", rd, "--narrative", write(tmp_path, "narr.json", NARRATIVE))
    assert (rd / "report.html").exists() and (rd / "decisions.json").exists()
    return rd


def test_image_augmentation(tmp_path, image_npz):
    plan = {"objective": {"type": "class_balancing", "mode": "augmentation"}, "n": 90,
            "candidates": [{"model": "cvae_image", "config": {"epochs": 2}},
                           {"model": "dcgan_image", "config": {"epochs": 1}, "search_space": "default",
                            "hpo_trials": 1},
                           {"model": "ddpm_image", "config": {"epochs": 1, "sampling_steps": 5,
                                                              "base_channels": 32}}]}
    rd = full_flow(tmp_path, image_npz, plan)
    down = json.loads((rd / "downstream.json").read_text())
    assert down["per_class"]["0"] == 0 and down["n_synthetic"] > 0
    assert "real_plus_synthetic" in down["downstream"]["arms"]
    # Replay reproduces the run from the command log without the agent
    run_cli("replay", rd, "--dst", tmp_path / "replayed")
    assert (tmp_path / "replayed" / "report.md").exists()


def test_tabular_generation(tmp_path, table_csv):
    plan = {"objective": {"type": "generation"},
            "candidates": [{"model": "tvae", "config": {"epochs": 5}},
                           {"model": "ctgan", "config": {"epochs": 5}},
                           {"model": "nsf_flow", "config": {"epochs": 5}}]}
    rd = full_flow(tmp_path, table_csv, plan)
    assert (rd / "synthetic" / "synthetic.csv").exists()


def test_scrna_representation(tmp_path, scrna_h5ad):
    plan = {"objective": {"type": "representation"},
            "candidates": [{"model": "scvi", "config": {"epochs": 5}},
                           {"model": "latent_diffusion_scvi", "config": {"epochs": 5, "diff_steps": 50,
                                                                         "sampling_steps": 10}}]}
    rd = full_flow(tmp_path, scrna_h5ad, plan)
    assert (rd / "synthetic" / "latent_embeddings.npz").exists()


def test_scrna_gan_and_flow_anomaly(tmp_path, scrna_h5ad):
    plan = {"objective": {"type": "anomaly_detection"},
            "candidates": [{"model": "nsf_flow", "config": {"epochs": 5, "n_components": 10}}]}
    rd = full_flow(tmp_path, scrna_h5ad, plan, n=0)
    assert (rd / "synthetic" / "anomaly_scores_test.csv").exists()


def test_declined_objective(tmp_path, table_csv):
    rd = tmp_path / "run"
    run_cli("init", "--data", table_csv, "--budget", 1, "--run-dir", rd)
    run_cli("plan", "--run", rd, "--file", write(tmp_path, "p.json", {"objective": {"type": "imputation"}}))
    run_cli("finalize", "--run", rd)


def test_plan_validation_errors(tmp_path, table_csv):
    rd = tmp_path / "run"
    run_cli("init", "--data", table_csv, "--budget", 1, "--run-dir", rd)
    bad = {"objective": {"type": "generation"}, "candidates": [{"model": "ddpm_image"}]}
    assert cli.main(["plan", "--run", str(rd), "--file", str(write(tmp_path, "b.json", bad))]) == 2
    bad = {"objective": {"type": "class_balancing"}, "n": 10_000, "candidates": [{"model": "tvae"}]}
    assert cli.main(["plan", "--run", str(rd), "--file", str(write(tmp_path, "c.json", bad))]) == 2
