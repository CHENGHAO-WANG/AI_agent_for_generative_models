"""Command-line interface. Every command prints a JSON result on stdout (after a `=== RESULT ===` marker)
so the agent can read it reliably even when libraries print progress messages."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
import warnings
from pathlib import Path

MARKER = "=== RESULT ==="


def _quiet():
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    for name in ("lightning", "lightning.pytorch", "pytorch_lightning", "scvi", "optuna", "numba", "anndata"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _emit(obj) -> None:
    from .utils import _default

    print(MARKER)
    print(json.dumps(obj, indent=2, default=_default))


def _names(s: str | None):
    return [x.strip() for x in s.split(",")] if s else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="genagent", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("registry", help="list available generative models")
    s.add_argument("--modality", choices=["image", "scrna", "tabular"])

    s = sub.add_parser("init", help="load + profile a dataset and create a run directory")
    s.add_argument("--data", required=True, help="path, or built-in name: pathmnist | pbmc3k")
    s.add_argument("--n", type=int, default=None, help="number of synthetic samples (meaning depends on objective)")
    s.add_argument("--objective", default=None, help="free-text user objective")
    s.add_argument("--label", default=None, help="label column/key, 'none' for unlabeled, omit to auto-detect")
    s.add_argument("--budget", type=float, default=30.0, help="time budget in minutes")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-questions", action="store_true")
    s.add_argument("--run-dir", default=None)

    for name, hlp in (("plan", "validate and store the agent's plan JSON"),
                      ("refine", "one refinement round from a JSON spec")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--run", required=True)
        s.add_argument("--file", required=True)

    for name in ("train", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--names", default=None, help="comma-separated candidate names (default: all pending)")

    s = sub.add_parser("select", help="choose the final model")
    s.add_argument("--run", required=True)
    s.add_argument("--name", default=None, help="one model for every objective it serves (default: best "
                   "combined score, replaced per objective only when beaten by more than the margin)")
    s.add_argument("--assign", action="append", default=[], metavar="OBJECTIVE=CANDIDATE",
                   help="model for one objective; repeatable")
    s.add_argument("--rationale", default="")

    s = sub.add_parser("finalize", help="generate final synthetic data and run downstream analysis")
    s.add_argument("--run", required=True)

    s = sub.add_parser("report", help="render report.md/html; optionally export to outputs/<name>")
    s.add_argument("--run", required=True)
    s.add_argument("--narrative", default=None, help="JSON with the agent's interpretive sections")
    s.add_argument("--export", default=None, help="e.g. generated_report_1")

    s = sub.add_parser("note", help="append a free-form decision to the decision log")
    s.add_argument("--run", required=True)
    s.add_argument("--step", required=True)
    s.add_argument("--decision", required=True)
    s.add_argument("--rationale", default="")

    s = sub.add_parser("status", help="show which steps of a run are done")
    s.add_argument("--run", required=True)

    s = sub.add_parser("replay", help="re-execute a finished run from its logged commands, without the agent")
    s.add_argument("src")
    s.add_argument("--dst", default=None)
    s.add_argument("--export", default=None, help="export name for the replayed report")
    return p


def status(run_path: str) -> dict:
    from .pipeline import Run

    run = Run(run_path)
    p = run.path
    plan = run.json("plan.json")
    st = {"run_dir": str(p), "initialized": True, "planned": plan is not None}
    if plan:
        st["candidates"] = {c["name"]: {"trained": (p / "models" / c["name"] / "train.json").exists()}
                            for c in plan["candidates"]}
        st["evaluated"] = list((run.json("metrics.json") or {}).keys())
        st["refinement_done"] = bool(plan.get("refinement_done"))
        st["selected"] = plan.get("selected")
    st["finalized"] = (p / "downstream.json").exists()
    st["reported"] = (p / "report.md").exists()
    return st


def replay(src: str, dst: str | None, export: str | None) -> dict:
    from . import pipeline as P
    from . import report as R
    from .utils import read_json

    srcp = Path(src)
    cmds = read_json(srcp / "commands.json")
    dst = dst or str(srcp) + "_replay"
    run = None
    for c in cmds:
        a = c["args"]
        if c["cmd"] == "init":
            P.init_run(a["data"], a["n"], a["objective_text"], a["label"], a["budget_minutes"], a["seed"],
                       a["no_questions"], run_dir=dst)
            run = P.Run(dst)
        elif c["cmd"] == "plan":
            P.set_plan(run, str(srcp / a["plan_file"]))
        elif c["cmd"] == "train":
            P.train(run, a["names"])
        elif c["cmd"] == "evaluate":
            P.evaluate(run, a["names"])
        elif c["cmd"] == "refine":
            P.refine(run, str(srcp / a["refine_file"]))
        elif c["cmd"] == "select":
            P.select(run, a["name"], a["rationale"], a.get("assign"))
        elif c["cmd"] == "finalize":
            P.finalize(run)
        elif c["cmd"] == "note":
            P.note(run, a["step"], a["decision"], a["rationale"])
        elif c["cmd"] == "report":
            R.render(run, str(srcp / a["narrative_file"]) if a.get("narrative_file") else None)
            run.log_command("report", a)
    out = {"replayed_from": str(srcp), "run_dir": dst, "commands": len(cmds)}
    if export:
        out["exported_to"] = str(R.export(run, export))
    return out


def main(argv=None) -> int:
    _quiet()
    args = build_parser().parse_args(argv)
    from . import pipeline as P

    try:
        if args.cmd == "registry":
            from .models import REGISTRY

            res = [m.describe() for m in REGISTRY.values() if not args.modality or args.modality in m.modalities]
        elif args.cmd == "init":
            res = P.init_run(args.data, args.n, args.objective, args.label, args.budget, args.seed,
                             args.no_questions, run_dir=args.run_dir)
        elif args.cmd == "plan":
            res = P.set_plan(P.Run(args.run), args.file)
        elif args.cmd == "train":
            res = P.train(P.Run(args.run), _names(args.names))
        elif args.cmd == "evaluate":
            res = P.evaluate(P.Run(args.run), _names(args.names))
        elif args.cmd == "refine":
            res = P.refine(P.Run(args.run), args.file)
        elif args.cmd == "select":
            assign = dict(a.split("=", 1) for a in args.assign) or None
            res = P.select(P.Run(args.run), args.name, args.rationale, assign)
        elif args.cmd == "finalize":
            res = P.finalize(P.Run(args.run))
        elif args.cmd == "report":
            from . import report as R

            run = P.Run(args.run)
            path = R.render(run, args.narrative)
            run.log_command("report", {"narrative_file": "inputs/narrative.json"
                                       if (run.path / "inputs" / "narrative.json").exists() else None})
            res = {"report_md": str(path), "report_html": str(path.with_suffix(".html"))}
            if args.export:
                res["exported_to"] = str(R.export(run, args.export))
        elif args.cmd == "note":
            P.note(P.Run(args.run), args.step, args.decision, args.rationale)
            res = {"ok": True}
        elif args.cmd == "status":
            res = status(args.run)
        elif args.cmd == "replay":
            res = replay(args.src, args.dst, args.export)
        else:
            raise SystemExit(f"unknown command {args.cmd}")
    except (P.PlanError, ValueError, FileNotFoundError) as e:
        _emit({"error": str(e), "type": type(e).__name__})
        return 2
    except Exception as e:  # unexpected: show traceback for diagnosis
        _emit({"error": repr(e), "traceback": traceback.format_exc()[-3000:]})
        return 1
    _emit(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
