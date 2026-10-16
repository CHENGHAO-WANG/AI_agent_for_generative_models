"""Render the final analysis report (Markdown + self-contained HTML).

Factual sections come from run artifacts (profile, plan, decisions, metrics, downstream results).
Interpretive sections come from `narrative.json`, written by the agent."""

from __future__ import annotations

import base64
import re
import shutil
from pathlib import Path

from .models import REGISTRY
from .utils import read_json, write_json

NARRATIVE_KEYS = ("summary", "objective_interpretation", "model_choice", "hyperparameters", "evaluation",
                  "downstream", "recommendations", "limitations")
REC_KEYS = ("suitable_uses", "avoid", "caveats", "synthetic_to_real_ratio")

GENERAL_COLS = ["precision", "recall", "density", "coverage", "mmd", "mmd_baseline_real", "c2st_auc",
                "tstr_balanced_acc", "trtr_balanced_acc", "dcr_ratio", "dcr_frac_below_holdout_q05"]
SPECIFIC_COLS = {
    "image": ["fid_feature_space", "fid_baseline_real", "class_consistency_acc"],
    "scrna": ["gene_mean_corr", "gene_var_corr", "gene_zero_frac_corr", "cluster_tvd", "marker_expr_corr"],
    "tabular": ["sdm_quality_score", "sdm_column_shapes", "sdm_column_pair_trends"],
}
OBJ_COLS = ["latent_knn_acc", "latent_kmeans_ari", "latent_silhouette", "anomaly_auroc"]


def validate_narrative(nar: dict) -> list[str]:
    missing = [k for k in NARRATIVE_KEYS if not nar.get(k)]
    rec = nar.get("recommendations") or {}
    missing += [f"recommendations.{k}" for k in REC_KEYS if not rec.get(k)]
    return missing


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return "" if v is None else str(v)


def _table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_fmt(c).replace("|", "/") for c in r) + " |" for r in rows]
    return "\n".join(out)


def _bullets(items) -> str:
    if isinstance(items, str):
        return items
    return "\n".join(f"- {i}" for i in items)


def _fig(run_path: Path, rel: str, caption: str) -> str:
    return f"![{caption}]({rel})\n\n*{caption}*\n" if (run_path / rel).exists() else ""


def render(run, narrative_file: str | None = None) -> Path:
    p = run.path
    if narrative_file:
        nar = read_json(narrative_file)
        missing = validate_narrative(nar)
        if missing:
            raise ValueError(f"narrative is missing: {missing}")
        write_json(run.f("inputs", "narrative.json"), nar)
    else:
        nar = run.json("inputs/narrative.json") or {}
    info, prof, plan = run.info, run.json("profile.json"), run.plan()
    metrics, scores = run.json("metrics.json", {}), run.json("scores.json", {})
    down, dec = run.json("downstream.json", {}), run.json("decisions.json", [])
    modality = info["modality"]
    obj = plan["objective"]
    L: list[str] = []
    L.append(f"# Generative modeling report: {info['dataset_name']}\n")
    L.append(_table(["Item", "Value"], [
        ["Data", info["data"]], ["Modality", modality], ["Requested n", info.get("n")],
        ["User objective", info.get("objective_text") or "(none: default to data generation)"],
        ["Interpreted objective", obj["type"] + (f" ({obj['mode']})" if obj.get("mode") else "")],
        ["Selected model", plan.get("selected", "-")], ["Budget (min)", info["budget_minutes"]],
        ["Seed", info["seed"]], ["Created", info["created"]]]))
    L.append("\n## 1. Summary\n\n" + nar.get("summary", ""))

    # -- dataset
    L.append("\n## 2. Dataset profile\n")
    rows = [["Split sizes", prof["split_sizes"]], ["Official splits", prof["official_splits"]],
            ["Labeled", prof["labeled"]]]
    for k in ("n_classes", "imbalance_ratio_max_over_min", "minority_class", "majority_class_size",
              "image_shape_chw", "n_genes", "sparsity_fraction_zero", "library_size", "eval_clusters_leiden",
              "n_features", "n_numeric", "n_categorical", "missing_values", "n_cells_after_qc"):
        if k in prof:
            rows.append([k.replace("_", " "), prof[k]])
    L.append(_table(["Property", "Value"], rows))
    if prof.get("class_counts_train"):
        L.append("\nTraining class counts:\n\n" + _table(["Class", "Count"], list(prof["class_counts_train"].items())))

    # -- objective
    L.append("\n## 3. Objective interpretation\n\n" + nar.get("objective_interpretation", ""))
    qs = plan.get("questions") or []
    if qs:
        L.append("\nQuestions asked during planning:\n\n" + _table(
            ["Question", "Answer", "Answered by"], [[q.get("question"), q.get("answer"), q.get("answered_by", "user")]
                                                    for q in qs]))
    alloc = plan.get("allocation", {})
    if alloc.get("per_class"):
        L.append("\nSynthetic sample allocation:\n\n" + _table(["Class", "Synthetic samples"],
                                                              list(alloc["per_class"].items())))
        L.append("\n" + _bullets(alloc.get("notes", [])))

    # -- models
    L.append("\n## 4. Model selection and hyperparameters\n\n" + nar.get("model_choice", ""))
    rows = []
    for c in plan["candidates"]:
        cls = REGISTRY[c["model"]]
        rows.append([c["name"], cls.family, cls.library, c.get("refined_from", ""), c.get("rationale", "")])
    if rows:
        L.append("\n" + _table(["Candidate", "Family", "Implementation", "Refined from", "Rationale"], rows))
    L.append("\n" + nar.get("hyperparameters", ""))
    rows = []
    for c in plan["candidates"]:
        t = run.json(f"models/{c['name']}/train.json")
        if t is None:
            continue
        hpo = t.get("hpo") or {}
        rows.append([c["name"], _cfg(t["final_config"]), hpo.get("n_completed", 0), t["history"].get("epochs_run"),
                     t["seconds"], t["history"].get("stopped_by_budget")])
    if rows:
        L.append("\n" + _table(["Candidate", "Final configuration", "Search trials", "Epochs", "Train s",
                                "Stopped by budget"], rows))

    # -- evaluation
    L.append("\n## 5. Evaluation\n")
    names = [k for k in metrics if not k.startswith("baseline:")]
    if names:
        cols = [c for c in GENERAL_COLS if any(c in metrics[n] for n in names)]
        L.append("General metrics (shared embedding; real test data as reference):\n\n" +
                 _table(["Metric"] + names, [[c] + [metrics[n].get(c) for n in names] for c in cols]))
        cols = [c for c in SPECIFIC_COLS[modality] + OBJ_COLS if any(c in metrics[n] for n in names)]
        base = [k for k in metrics if k.startswith("baseline:")]
        if cols:
            L.append(f"\n{modality}-specific and objective metrics:\n\n" + _table(
                ["Metric"] + names + base, [[c] + [metrics[n].get(c) for n in names + base] for c in cols]))
        pc = scores.get("per_candidate", {})
        L.append("\nWeighted score (higher is better; memorizing candidates excluded):\n\n" + _table(
            ["Candidate", "Score", "Memorization flag"],
            [[n, pc[n]["score"], pc[n]["excluded_memorization"]] for n in pc]))
        L.append(f"\nRanking: {', '.join(scores.get('ranking', []))}\n")
        L.append("\n" + nar.get("evaluation", "") + "\n")
        for n in names:
            for rel, cap in ((f"figures/samples_{n}.png", f"Samples from {n}"),
                             (f"figures/umap_{n}.png", f"UMAP overlay for {n}"),
                             (f"figures/marginals_{n}.png", f"Marginal distributions for {n}")):
                L.append(_fig(p, rel, cap))

    # -- downstream
    L.append("\n## 6. Final synthetic data and downstream analysis\n")
    if down.get("synthetic_file"):
        L.append(f"Generated **{down['n_synthetic']}** samples with `{plan.get('selected')}` -> "
                 f"`{down['synthetic_file']}`.\n")
    d = down.get("downstream", {})
    if d.get("arms"):
        arms = d["arms"]
        L.append(_table(["Training data", "Balanced acc (mean ± sd)", "Macro F1"],
                        [[a, f"{v['balanced_accuracy_mean']:.4f} ± {v['balanced_accuracy_sd']:.4f}",
                          v["macro_f1_mean"]] for a, v in arms.items()]))
        cls_names = list(next(iter(arms.values()))["per_class_recall_mean"])
        L.append("\nPer-class recall (mean over seeds):\n\n" + _table(
            ["Class"] + list(arms), [[c] + [arms[a]["per_class_recall_mean"][c] for a in arms] for c in cls_names]))
    elif d:
        L.append(_table(["Item", "Value"], [[k, v] for k, v in _flat(d).items()]))
    for rel, cap in (("figures/final_samples.png", "Final synthetic samples"),
                     ("figures/augmentation_per_class_recall.png", "Per-class recall by training data"),
                     ("figures/synthetic_only_per_class_recall.png", "Per-class recall: synthetic-only training"),
                     ("figures/latent_umap.png", "Learned latent space"),
                     ("figures/anomaly_scores.png", "Anomaly score distributions")):
        L.append(_fig(p, rel, cap))
    L.append("\n" + nar.get("downstream", ""))

    # -- recommendations
    rec = nar.get("recommendations", {})
    L.append("\n## 7. Recommendations for downstream use\n")
    L.append("**Suitable uses**\n\n" + _bullets(rec.get("suitable_uses", [])))
    L.append("\n**Uses to avoid**\n\n" + _bullets(rec.get("avoid", [])))
    L.append("\n**Caveats**\n\n" + _bullets(rec.get("caveats", [])))
    L.append("\n**Suggested synthetic : real ratio**\n\n" + _bullets(rec.get("synthetic_to_real_ratio", "")))
    L.append("\n## 8. Limitations\n\n" + _bullets(nar.get("limitations", "")))

    # -- decision log
    L.append("\n## Appendix: decision log\n")
    L.append(_table(["Step", "Decision", "Rationale"],
                    [[e["step"], e["decision"], e.get("rationale", "")] for e in dec]))
    md = "\n".join(L) + "\n"
    (p / "report.md").write_text(md, encoding="utf-8")
    (p / "report.html").write_text(to_html(md, p, info["dataset_name"]), encoding="utf-8")
    return p / "report.md"


def _cfg(c: dict) -> str:
    return ", ".join(f"{k}={_fmt(v)}" for k, v in c.items())


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[prefix + k] = v
    return out


CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2em auto;padding:0 16px;
line-height:1.5;color:#1d1d1f;background:#fff}
table{border-collapse:collapse;margin:.8em 0;font-size:.88em;display:block;overflow-x:auto}
th,td{border:1px solid #d0d0d5;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#f3f3f6} img{max-width:100%} h1{font-size:1.7em} h2{border-bottom:1px solid #ddd;padding-bottom:.2em}
code{background:#f3f3f6;padding:1px 4px;border-radius:3px}
"""


def to_html(md: str, base: Path, title: str) -> str:
    """Markdown -> HTML with figures inlined as base64 so the file is self-contained."""
    import markdown

    body = markdown.markdown(md, extensions=["tables"])

    def inline(m):
        src = m.group(1)
        f = base / src
        if f.exists():
            return f'src="data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}"'
        return m.group(0)

    body = re.sub(r'src="([^"]+\.png)"', inline, body)
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{title} report</title><style>{CSS}</style>"
            f"</head><body>{body}</body></html>")


def export(run, name: str, out_root: str = "outputs") -> Path:
    """Copy the report and its artifacts (not model weights or cached data) to outputs/<name>/."""
    dst = Path(out_root) / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for item in ("figures", "synthetic", "inputs", "eval"):
        if (run.path / item).exists() and item != "eval":
            shutil.copytree(run.path / item, dst / item)
    for f in ("run.json", "profile.json", "plan.json", "decisions.json", "commands.json", "metrics.json",
              "scores.json", "downstream.json"):
        if (run.path / f).exists():
            shutil.copy(run.path / f, dst / f)
    for c in (run.path / "models").glob("*/train.json"):
        (dst / "training").mkdir(exist_ok=True)
        shutil.copy(c, dst / "training" / f"{c.parent.name}.json")
    shutil.copy(run.path / "report.md", dst / f"{name}.md")
    shutil.copy(run.path / "report.html", dst / f"{name}.html")
    return dst
