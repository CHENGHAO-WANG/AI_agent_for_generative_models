"""Render the final analysis report (Markdown + self-contained HTML).

Factual sections come from run artifacts (profile, plan, decisions, metrics, downstream results).
Interpretive sections come from `narrative.json`, written by the agent."""

from __future__ import annotations

import base64
import datetime
import re
import shutil
from pathlib import Path

from .models import resolve_model
from .utils import read_json, write_json

NARRATIVE_KEYS = ("summary", "objective_interpretation", "model_choice", "hyperparameters", "evaluation",
                  "downstream", "recommendations", "limitations")
REC_KEYS = ("suitable_uses", "avoid", "caveats", "synthetic_to_real_ratio")

GENERAL_COLS = ["precision", "recall", "density", "coverage", "mmd", "mmd_baseline_real", "c2st_auc",
                "tstr_balanced_acc", "trtr_balanced_acc", "dcr_ratio", "dcr_frac_below_holdout_q05"]
SPECIFIC_COLS = {
    "image": ["fid_feature_space", "fid_baseline_real"],
    "scrna": ["gene_mean_corr", "gene_var_corr", "gene_zero_frac_corr", "cluster_tvd", "cluster_tvd_baseline_real",
              "marker_expr_corr"],
    "tabular": ["sdm_quality_score", "sdm_column_shapes", "sdm_column_pair_trends"],
}
OBJ_COLS = ["class_consistency_acc", "class_consistency_balanced", "class_consistency_min",
            "memorization_flag_any_class", "latent_knn_acc", "latent_kmeans_ari", "latent_silhouette",
            "circular_reference_labels", "anomaly_auroc"]


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
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return f"[{v[0]:.3g}, {v[1]:.3g}]"
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


def _objective_label(obj: dict) -> str:
    s = obj["type"] + (f" ({obj['mode']})" if obj.get("mode") else "")
    if obj.get("secondary"):
        s += "; secondary: " + ", ".join(obj["secondary"])
    return s


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
    selected = plan.get("selected") or {}
    L: list[str] = []
    L.append(f"# Generative modeling report: {info['dataset_name']}\n")
    L.append(_table(["Item", "Value"], [
        ["Data", info["data"]], ["Modality", modality], ["Requested n", info.get("n")],
        ["User objective", info.get("objective_text") or "(none: default to data generation)"],
        ["Interpreted objective", _objective_label(obj)],
        ["Selected model(s)", "; ".join(f"{o}: {c}" for o, c in selected.items()) or "-"],
        ["Budget (min)", info["budget_minutes"]], ["Seed", info["seed"]], ["Created", info["created"]]]))
    L.append("\n## 1. Summary\n\n" + nar.get("summary", ""))

    # -- dataset
    L.append("\n## 2. Dataset profile\n")
    rows = [["Split sizes", prof["split_sizes"]], ["Official splits", prof["official_splits"]],
            ["Labeled", prof["labeled"]]]
    for k in ("split", "smallest_class_share", "n_classes", "imbalance_ratio_max_over_min", "minority_class",
              "majority_class_size", "image_shape_chw", "n_genes", "sparsity_fraction_zero", "library_size",
              "eval_label_count", "eval_labels_source", "n_features", "n_numeric", "n_categorical",
              "missing_values", "n_cells_after_qc"):
        if k in prof:
            rows.append([k.replace("_", " "), prof[k]])
    L.append(_table(["Property", "Value"], rows))
    if prof.get("class_counts_train"):
        L.append("\nTraining class counts:\n\n" + _table(["Class", "Count"], list(prof["class_counts_train"].items())))
    elif prof.get("eval_label_counts_train"):
        L.append("\nTraining counts of the evaluation-only reference labels:\n\n" +
                 _table(["Group", "Count"], list(prof["eval_label_counts_train"].items())))

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
        cls, _ = resolve_model(c["model"])
        rows.append([c["name"], cls.family, cls.library, ", ".join(c.get("serves", [])),
                     "on" if c.get("use_labels") else "off",
                     cls.conditional if c.get("use_labels") else "", c.get("refined_from", ""), c.get("rationale", "")])
    if rows:
        L.append("\n" + _table(["Candidate", "Family", "Implementation", "Serves", "Labels", "Class conditioning",
                                "Refined from", "Rationale"], rows))
    partial = [c for c in plan["candidates"] if c.get("partial_ok")]
    if partial:
        L.append("\n" + _bullets([f"`{c['name']}` cannot serve every objective; included with approval "
                                  f"({c.get('partial_ok_by', 'user')}) and competes only for "
                                  f"{', '.join(c.get('serves', []))}." for c in partial]))
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
    names = [k for k in metrics if not k.startswith("baseline:") and "error" not in metrics[k]]
    base = [k for k in metrics if k.startswith("baseline:")]
    if names:
        cols = [c for c in GENERAL_COLS if any(c in metrics[n] for n in names)]
        L.append("General metrics (shared embedding; real test data as reference):\n\n" +
                 _table(["Metric"] + names, [[c] + [metrics[n].get(c) for n in names] for c in cols]))
        cols = [c for c in SPECIFIC_COLS[modality] + OBJ_COLS if any(c in metrics[n] for n in names + base)]
        if cols:
            L.append(f"\n{modality}-specific and objective metrics (baselines are unsupervised embeddings scored "
                     "like a latent space):\n\n" + _table(
                         ["Metric"] + names + base, [[c] + [metrics[n].get(c) for n in names + base] for c in cols]))
        L.extend(_per_class_tables(metrics, names))
        L.extend(_cluster_tables(metrics, names))
        L.append(_score_tables(scores))
        L.append("\n" + nar.get("evaluation", "") + "\n")
        for n in names:
            for rel, cap in ((f"figures/samples_{n}.png", f"Samples from {n}"),
                             (f"figures/umap_{n}.png", f"UMAP overlay for {n}"),
                             (f"figures/marginals_{n}.png", f"Marginal distributions for {n}")):
                L.append(_fig(p, rel, cap))

    # -- downstream
    L.append("\n## 6. Final synthetic data and downstream analysis\n")
    if down.get("synthetic_file"):
        L.append(f"Generated **{down['n_synthetic']}** samples with `{down.get('generated_by')}` -> "
                 f"`{down['synthetic_file']}`.\n")
    if down.get("note_allocation"):
        L.append(down["note_allocation"] + "\n")
    for o, d in (down.get("downstream") or {}).items():
        L.append(f"\n### {o.replace('_', ' ').capitalize()} (model: `{selected.get(o, '-')}`)\n")
        L.extend(_downstream_section(d))
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
    to_pdf(md, p, info["dataset_name"], p / "report.pdf")
    return p / "report.md"


def _per_class_tables(metrics: dict, names: list[str]) -> list[str]:
    out = []
    for n in names:
        pc = metrics[n].get("per_class")
        if not pc:
            continue
        out.append(f"\nPer-class checks for `{n}` (class-balanced probe; "
                   "memorization reference: the class's own held-out samples when it has at least 20):\n\n" +
                   _table(["Class", "Train n", "Held-out n", "Class consistency", "DCR share < q05",
                           "Exact copies", "Reference", "Memorization flag"],
                          [[c, v["n_train"], v["n_heldout"], v["class_consistency"], v["dcr_frac_below_q05"],
                            v["dcr_exact_copy_frac"], v["dcr_reference"], v["memorization_flag"]]
                           for c, v in pc.items()]))
    return out


def _cluster_tables(metrics: dict, names: list[str]) -> list[str]:
    with_props = [n for n in names if "cluster_props_synthetic" in metrics[n]]
    if not with_props:
        return []
    ref = metrics[with_props[0]]
    groups = list(ref["cluster_props_reference_train"])
    rows = [[g, ref["cluster_props_reference_train"][g], ref["cluster_props_heldout_real_assigned"][g]] +
            [metrics[n]["cluster_props_synthetic"][g] for n in with_props] for g in groups]
    return ["\nGroup proportions (synthetic cells assigned by 15-NN; held-out real cells assigned the same way "
            "give the real-vs-real baseline):\n\n" +
            _table(["Group", "Train (reference)", "Held-out real (assigned)"] + with_props, rows)]


def _score_tables(scores: dict) -> str:
    pc = scores.get("per_candidate", {})
    per_obj = scores.get("per_objective", {})
    objs = list(per_obj)
    rows = []
    for n in pc:
        rows.append([n, pc[n]["score"]] + [per_obj[o]["per_candidate"].get(n, {}).get("score", "n/a") for o in objs] +
                    [pc[n]["excluded_memorization"]])
    txt = ("\nWeighted scores (higher is better; combined = primary objective's weights + half of each secondary "
           "objective's weights; 'n/a' = the candidate does not serve that objective; memorizing candidates are "
           "excluded from rankings):\n\n" +
           _table(["Candidate", "Combined"] + objs + ["Memorization flag"], rows))
    txt += f"\n\nCombined ranking: {', '.join(scores.get('ranking', []))}"
    for o in objs:
        txt += f"; {o}: {', '.join(per_obj[o].get('ranking', []))}"
    return txt + "\n"


def _downstream_section(d: dict) -> list[str]:
    out = []
    if d.get("arms"):
        arms = d["arms"]
        out.append(f"Classifier: {d.get('classifier', '')}; {d.get('seeds')} seeds; test samples per class: "
                   f"{d.get('n_test_per_class')}.\n")
        out.append(_table(["Training data", "Balanced acc (mean ± sd)", "95% CI", "Δ vs real only (95% CI)",
                           "Macro F1"],
                          [[a, f"{v['balanced_accuracy_mean']:.4f} ± {v['balanced_accuracy_sd']:.4f}",
                            v.get("balanced_accuracy_ci95"), v.get("delta_balanced_accuracy_vs_real_only_ci95", ""),
                            v["macro_f1_mean"]] for a, v in arms.items()]))
        cls_names = list(next(iter(arms.values()))["per_class_recall_mean"])
        out.append("\nPer-class recall, mean over seeds [bootstrap 95% CI]:\n\n" + _table(
            ["Class"] + list(arms),
            [[c] + [f"{arms[a]['per_class_recall_mean'][c]:.3f} {_fmt(arms[a].get('per_class_recall_ci95', {}).get(c))}"
                    for a in arms] for c in cls_names]))
    elif d:
        out.append(_table(["Item", "Value"], [[k, v] for k, v in _flat(d).items()]))
    return out


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


def _inline_images(body: str, base: Path, max_w: float | None = None, max_h: float | None = None) -> str:
    """Replace relative PNG sources by base64 data. With `max_w`/`max_h` (points), also set an
    explicit size that fits the box, for renderers without CSS max-width (the PDF engine)."""

    def inline(m):
        f = base / m.group(1)
        if not f.exists():
            return m.group(0)
        data = f.read_bytes()
        attr = f'src="data:image/png;base64,{base64.b64encode(data).decode()}"'
        if max_w and data[:8] == b"\x89PNG\r\n\x1a\n":
            # PNG header: width and height are big-endian integers at bytes 16-24; 1 px = 0.75 pt at 96 dpi
            w, h = int.from_bytes(data[16:20], "big") * 0.75, int.from_bytes(data[20:24], "big") * 0.75
            s = min(1.0, max_w / w, (max_h or h) / h)
            attr += f' width="{w * s:.0f}" height="{h * s:.0f}"'
        return attr

    return re.sub(r'src="([^"]+\.png)"', inline, body)


def to_html(md: str, base: Path, title: str) -> str:
    """Markdown -> HTML with figures inlined as base64 so the file is self-contained."""
    import markdown

    body = _inline_images(markdown.markdown(md, extensions=["tables"]), base)
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{title} report</title><style>{CSS}</style>"
            f"</head><body>{body}</body></html>")


# Print layout for the PDF: landscape A4 because the metric tables are wide; xhtml2pdf supports
# only a subset of CSS (no max-width, no overflow), so figures get explicit sizes instead. Long tables
# break across pages with a repeated header row. The text font is DejaVu Sans (shipped with matplotlib),
# because the built-in PDF fonts lack glyphs the report uses (Greek letters, math signs, bullets).
PDF_CSS = """
@font-face{font-family:DejaVu;src:url("%(regular)s")}
@font-face{font-family:DejaVu;src:url("%(bold)s");font-weight:bold}
@page{size:a4 landscape;margin:1.4cm}
body{font-family:DejaVu;font-size:9pt;line-height:1.35;color:#1d1d1f}
h1{font-size:17pt} h2{font-size:13pt;border-bottom:1px solid #bbb;padding-bottom:2pt;margin-top:14pt}
h3{font-size:11pt}
table{margin:4pt 0}
th,td{border:0.5pt solid #b0b0b8;padding:2pt 3pt;font-size:7pt;text-align:left;vertical-align:top}
th{background-color:#eeeef2}
code{font-family:DejaVu;background-color:#f3f3f6}
ul{list-style-type:none;margin-left:0;padding-left:8pt}
"""
PDF_CONTENT_PT = (842 - 2 * 40, 595 - 2 * 40 - 40)  # usable width and figure height on landscape A4


def to_pdf(md: str, base: Path, title: str, out: Path) -> Path:
    """Markdown -> PDF (xhtml2pdf) with the figures embedded; raises if the conversion fails."""
    import markdown
    from xhtml2pdf import pisa

    from matplotlib import font_manager

    fonts = {k: Path(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight=w),
                                           fallback_to_default=False)).as_uri()
             for k, w in (("regular", "normal"), ("bold", "bold"))}
    body = _inline_images(markdown.markdown(md, extensions=["tables"]), base, *PDF_CONTENT_PT)
    body = body.replace("<table>", '<table repeat="1">')  # repeat the header row after a page break
    # xhtml2pdf draws list markers from a built-in symbol font that renders "•" as a wrong glyph;
    # its markers are switched off in PDF_CSS and a bullet in the text font is prepended instead
    body = body.replace("<li>", "<li>•&nbsp;&nbsp;")
    html = (f"<html><head><meta charset='utf-8'><title>{title} report</title><style>{PDF_CSS % fonts}</style>"
            f"</head><body>{body}</body></html>")
    with open(out, "wb") as fh:
        status = pisa.CreatePDF(html, dest=fh, encoding="utf-8")
    if status.err:
        raise RuntimeError(f"PDF conversion reported {status.err} error(s)")
    return out


def export(run, name: str, out_root: str | None = None) -> Path:
    """Copy the report and its artifacts (not model weights or cached data) to
    outputs/<name>_<YYYYmmdd-HHMMSS>/. The timestamp keeps repeated exports apart; the report
    files inside keep the plain name (<name>.md / <name>.html / <name>.pdf).

    `outputs/` sits next to the `runs/` folder holding the run (runs/<id> -> outputs/), whatever the
    current working directory is."""
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"export name must be a plain name without path separators, got {name!r}")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(out_root) if out_root else run.path.resolve().parent.parent / "outputs"
    dst = root / f"{name}_{stamp}"
    dst.mkdir(parents=True)  # fails rather than overwrite if the folder already exists
    for item in ("figures", "synthetic", "inputs"):
        if (run.path / item).exists():
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
    if (run.path / "report.pdf").exists():  # absent for reports rendered before PDF output existed
        shutil.copy(run.path / "report.pdf", dst / f"{name}.pdf")
    return dst
