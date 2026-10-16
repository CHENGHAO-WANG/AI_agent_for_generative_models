"""Run steps exposed by the CLI. Claude (via the plugin skill) calls them in order:

    init -> plan -> train -> evaluate -> [refine -> evaluate] -> select -> finalize -> report

Each step reads and writes files in the run directory, appends its decisions to `decisions.json`,
and records itself in `commands.json` so `replay` can re-execute the run without the agent.
"""

from __future__ import annotations

import datetime
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import metrics as M
from .allocation import AllocationError, allocate, labels_from_allocation
from .data import Dataset, load_dataset, profile_dataset
from .embed import Embedder
from .models import REGISTRY, GenerativeModel, ModelContext, models_for
from .utils import Deadline, read_json, seed_everything, write_json

OBJECTIVES = ("generation", "class_balancing", "representation", "anomaly_detection")
DECLINED = ("imputation", "uncertainty_quantification")
TRAIN_SHARE = 0.65   # fraction of the run budget reserved for training (incl. search)
HPO_SHARE = 0.30     # fraction of each candidate's training time used for the search
N_EVAL_MAX = 3000


class PlanError(ValueError):
    pass


# ================================================================ run bookkeeping

class Run:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not (self.path / "run.json").exists():
            raise FileNotFoundError(f"Not a run directory: {self.path}")
        self.info = read_json(self.path / "run.json")

    # files
    def f(self, *parts) -> Path:
        p = self.path.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def dataset(self) -> Dataset:
        return Dataset.load(self.path / "dataset.pkl")

    def embedder(self) -> Embedder:
        return torch.load(self.path / "embedder.pt", weights_only=False)

    def plan(self) -> dict:
        p = self.path / "plan.json"
        if not p.exists():
            raise PlanError("No plan yet; run `genagent plan` first")
        return read_json(p)

    def save_plan(self, plan: dict) -> None:
        write_json(self.path / "plan.json", plan)

    def json(self, name: str, default=None):
        p = self.path / name
        return read_json(p) if p.exists() else default

    # logs
    def log_decision(self, step: str, decision: str, rationale: str = "", data: Any = None) -> None:
        log = self.json("decisions.json", [])
        log.append({"time": _now(), "step": step, "decision": decision, "rationale": rationale,
                    **({"data": data} if data is not None else {})})
        write_json(self.path / "decisions.json", log)

    def log_command(self, cmd: str, args: dict) -> None:
        log = self.json("commands.json", [])
        log.append({"cmd": cmd, "args": args})
        write_json(self.path / "commands.json", log)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def context_for(ds: Dataset, use_labels: bool, seed: int) -> ModelContext:
    ctx = ModelContext.from_dataset(ds, seed)
    if not use_labels:
        ctx.n_classes, ctx.class_names, ctx.class_freq, ctx.libsizes_by_class = 0, None, None, {}
    return ctx


def _take(X, idx):
    return X.iloc[idx].reset_index(drop=True) if hasattr(X, "iloc") else X[idx]


# ================================================================ init

def init_run(data: str, n: int | None = None, objective: str | None = None, label: str | None = None,
             budget: float = 30.0, seed: int = 0, no_questions: bool = False, runs_root: str = "runs",
             run_dir: str | None = None) -> dict:
    seed_everything(seed)
    t0 = time.time()
    ds = load_dataset(data, label, seed)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(run_dir) if run_dir else Path(runs_root) / f"{ds.name}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    info = {"data": data, "n": n, "objective_text": objective, "label": label, "budget_minutes": budget,
            "seed": seed, "no_questions": no_questions, "created": _now(), "dataset_name": ds.name,
            "modality": ds.modality}
    write_json(path / "run.json", info)
    ds.save(path / "dataset.pkl")
    prof = profile_dataset(ds)
    write_json(path / "profile.json", prof)
    emb = Embedder(ds, seed)
    torch.save(emb, path / "embedder.pt")
    run = Run(path)
    run.log_command("init", {k: info[k] for k in ("data", "n", "objective_text", "label", "budget_minutes",
                                                     "seed", "no_questions")})
    run.log_decision("profile", f"modality={ds.modality}, labeled={ds.labeled}",
                     "determined from file type and data values", {"embedding": emb.kind})

    ctx = ModelContext.from_dataset(ds, seed)
    flags = []
    if prof.get("label_ambiguous"):
        flags.append(f"label column ambiguous: candidates {prof['label_candidates']}")
    if ds.labeled and n and n > prof["majority_class_size"]:
        flags.append(f"n={n} exceeds the majority-class size {prof['majority_class_size']} "
                     "(matters only if the objective is augmentation-style class balancing)")
    if prof.get("missing_values"):
        flags.append(f"{prof['missing_values']} missing values (imputed with median/mode for modeling)")
    return {
        "run_dir": str(path),
        "profile": prof,
        "objectives_supported": list(OBJECTIVES),
        "objectives_declined": list(DECLINED),
        "models_available": [m.describe(ctx) for m in models_for(ds.modality)],
        "budget_minutes": budget,
        "training_seconds_total": round(budget * 60 * TRAIN_SHARE),
        "embedding_for_metrics": emb.kind,
        "flags": flags,
        "seconds_elapsed": round(time.time() - t0, 1),
    }


# ================================================================ plan

def set_plan(run: Run, plan_file: str) -> dict:
    """Validate the agent's plan, compute the per-class allocation, and log every decision."""
    plan = read_json(plan_file)
    ds = run.dataset()
    prof = run.json("profile.json")
    obj = plan.get("objective") or {}
    otype = obj.get("type")
    if otype not in OBJECTIVES + DECLINED:
        raise PlanError(f"objective.type must be one of {OBJECTIVES + DECLINED}")
    mode = obj.get("mode")
    if otype == "class_balancing":
        if not ds.labeled:
            raise PlanError("class_balancing needs labels; the dataset is unlabeled")
        mode = mode or "augmentation"
        if mode not in ("augmentation", "synthetic_balanced"):
            raise PlanError("objective.mode must be 'augmentation' or 'synthetic_balanced'")
    obj["mode"] = mode
    plan["objective"] = obj

    n = plan.get("n", run.info.get("n"))
    plan["n"] = n
    cands = plan.get("candidates", [])
    if otype in DECLINED:
        plan["generate"] = False
        cands = []
    else:
        if not cands:
            raise PlanError("at least one candidate model is required")
        plan.setdefault("generate", bool(n))

    use_labels = plan.get("use_labels")
    if use_labels is None:
        use_labels = ds.labeled and otype in ("generation", "class_balancing")
    if otype == "class_balancing":
        use_labels = True
    plan["use_labels"] = bool(use_labels and ds.labeled)
    ctx = context_for(ds, plan["use_labels"], run.info["seed"])

    seen = set()
    for c in cands:
        cls = REGISTRY.get(c.get("model"))
        if cls is None:
            raise PlanError(f"unknown model '{c.get('model')}'; choose from {sorted(REGISTRY)}")
        if ds.modality not in cls.modalities:
            raise PlanError(f"{cls.name} does not support modality {ds.modality}")
        c.setdefault("name", cls.name)
        if c["name"] in seen:
            raise PlanError(f"duplicate candidate name {c['name']}")
        seen.add(c["name"])
        defaults = cls.default_config(ctx)
        bad = set(c.get("config", {})) - set(defaults)
        if bad:
            raise PlanError(f"{cls.name}: unknown config keys {sorted(bad)}; valid: {sorted(defaults)}")
        c["config"] = {**defaults, **c.get("config", {})}
        space = c.get("search_space")
        if space == "default":
            space = cls.search_space(ctx)
        space = space or {}
        bad = set(space) - set(defaults)
        if bad:
            raise PlanError(f"{cls.name}: search space has unknown keys {sorted(bad)}")
        c["search_space"] = space
        c.setdefault("hpo_trials", 5 if space else 0)
        if otype == "representation" and "encode" not in cls.capabilities:
            raise PlanError(f"{cls.name} has no encoder; not usable for representation learning")
        if otype == "anomaly_detection" and "anomaly" not in cls.capabilities:
            raise PlanError(f"{cls.name} has no anomaly score; not usable for anomaly detection")
    tf = [c.get("time_fraction") for c in cands]
    if cands:
        if any(t is None for t in tf):
            tf = [1.0] * len(cands)
        s = sum(tf)
        for c, t in zip(cands, tf):
            c["time_fraction"] = t / s
    plan["candidates"] = cands

    # Allocation of synthetic samples across classes
    counts = np.array(list(prof["class_counts_train"].values())) if ds.labeled and plan["use_labels"] else None
    names = ds.class_names if counts is not None else None
    policy = plan.get("n_exceeds_policy") or ("error" if run.info.get("no_questions") else None)
    try:
        alloc = allocate(otype if otype in OBJECTIVES else "generation", mode, n if plan["generate"] else None,
                         names, counts, policy or "error")
    except AllocationError as e:
        hint = "" if run.info.get("no_questions") else (
            " Ask the user (clamp / keep / new n) and set plan.n_exceeds_policy or plan.n accordingly.")
        raise PlanError(str(e) + hint) from e
    plan["allocation"] = alloc

    holdout = (plan.get("anomaly") or {}).get("holdout_classes")
    if holdout:
        if not ds.labeled:
            raise PlanError("anomaly.holdout_classes needs labels")
        missing = set(holdout) - set(ds.class_names)
        if missing:
            raise PlanError(f"unknown holdout classes {missing}")

    weights = M.default_weights(otype if otype in OBJECTIVES else "generation", ds.modality)
    weights.update(plan.get("metric_weights") or {})
    plan["metric_weights"] = weights
    plan["validated"] = _now()

    run.save_plan(plan)
    stored = run.f("inputs", "plan_input.json")
    if Path(plan_file).resolve() != stored.resolve():
        shutil.copy(plan_file, stored)
    run.log_command("plan", {"plan_file": "inputs/plan_input.json"})
    for q in plan.get("questions", []):
        run.log_decision("question", f"Q: {q.get('question')} -> A: {q.get('answer')}",
                         f"answered by {q.get('answered_by', 'user')}")
    run.log_decision("objective", f"{otype}" + (f" ({mode})" if mode else ""), obj.get("interpretation", ""))
    run.log_decision("generate", f"generate={plan['generate']}, n={n}, use_labels={plan['use_labels']}",
                     plan.get("generate_rationale", ""), alloc)
    for c in cands:
        run.log_decision("model_selection", f"candidate {c['name']} = {c['model']}", c.get("rationale", ""),
                         {"config": c["config"], "search_space": c["search_space"], "hpo_trials": c["hpo_trials"],
                          "time_fraction": round(c["time_fraction"], 3)})
    run.log_decision("metric_weights", "weights for ranking candidates", plan.get("weights_rationale",
                     "default weights for objective and modality"), weights)
    if otype in DECLINED:
        run.log_decision("decline", f"objective {otype} not performed", obj.get("interpretation", ""))
    return {"plan": plan, "training_seconds_per_candidate":
            {c["name"]: round(run.info["budget_minutes"] * 60 * TRAIN_SHARE * c["time_fraction"]) for c in cands}}


# ================================================================ train

def _train_data(ds: Dataset, plan: dict):
    """Training/validation splits; removes anomaly hold-out classes and labels if not used."""
    tr, va = ds.splits["train"], ds.splits["val"]
    holdout = (plan.get("anomaly") or {}).get("holdout_classes") or []
    Xtr, ytr, Xva, yva = tr.X, tr.y, va.X, va.y
    if holdout:
        hid = [ds.class_names.index(c) for c in holdout]
        mtr, mva = ~np.isin(ytr, hid), ~np.isin(yva, hid)
        Xtr, ytr, Xva, yva = _take(Xtr, np.flatnonzero(mtr)), ytr[mtr], _take(Xva, np.flatnonzero(mva)), yva[mva]
    if not plan["use_labels"]:
        ytr = yva = None
    from .data import Split

    return Split(Xtr, ytr), Split(Xva, yva)


def train(run: Run, names: list[str] | None = None) -> list[dict]:
    from .hpo import run_search

    plan = run.plan()
    ds = run.dataset()
    seed = run.info["seed"]
    ctx = context_for(ds, plan["use_labels"], seed)
    emb = run.embedder()
    tr, va = _train_data(ds, plan)
    out = []
    for c in plan["candidates"]:
        if names and c["name"] not in names:
            continue
        if not names and (run.path / "models" / c["name"] / "train.json").exists():
            continue
        cls = REGISTRY[c["model"]]
        secs = c.get("seconds") or run.info["budget_minutes"] * 60 * TRAIN_SHARE * c["time_fraction"]
        t0 = time.time()
        cfg, hpo = dict(c["config"]), None
        if c["search_space"] and c["hpo_trials"] > 0:
            hpo = run_search(cls, ctx, cfg, c["search_space"], c["hpo_trials"], secs * HPO_SHARE, tr, va, emb, seed)
            cfg = hpo["best_config"]
        seed_everything(seed)
        model = cls(cfg, ctx)
        hist = model.fit(tr.X, tr.y, va.X, va.y, Deadline(max(30.0, secs - (time.time() - t0))))
        d = run.f("models", c["name"], "x").parent
        model.save(d)
        rec = {"candidate": c["name"], "model": c["model"], "final_config": cfg, "history": hist,
               "hpo": hpo, "seconds": round(time.time() - t0, 1), "budget_seconds": round(secs)}
        write_json(d / "train.json", rec)
        if hpo:
            run.log_decision("hyperparameters", f"{c['name']}: search chose {_diff(c['config'], cfg)}",
                             f"Optuna, {hpo['n_completed']} trials, proxy = MMD to validation data", hpo["trials"])
        out.append({k: rec[k] for k in ("candidate", "model", "final_config", "seconds", "budget_seconds")} |
                   {"epochs_run": hist.get("epochs_run"), "stopped_by_budget": hist.get("stopped_by_budget")})
    run.log_command("train", {"names": names})
    return out


def _diff(a: dict, b: dict) -> dict:
    return {k: b[k] for k in b if a.get(k) != b[k]}


# ================================================================ evaluate

def _eval_embeddings(run: Run, ds: Dataset, emb: Embedder):
    p = run.path / "eval" / "real_embeddings.npz"
    if p.exists():
        z = np.load(p)
        return z["train"], z["test"], z["train_idx"]
    rng = np.random.default_rng(run.info["seed"])
    tr = ds.splits["train"]
    idx = np.sort(rng.choice(len(tr), min(10000, len(tr)), replace=False))
    E_tr, E_te = emb.transform(_take(tr.X, idx)), emb.transform(ds.splits["test"].X)
    np.savez(run.f("eval", "real_embeddings.npz"), train=E_tr, test=E_te, train_idx=idx)
    return E_tr, E_te, idx


def _reference_labels(ds: Dataset, split: str):
    sp = ds.splits[split]
    if sp.y is not None:
        return sp.y, ds.class_names
    if sp.eval_labels is not None:
        return sp.eval_labels, ds.eval_label_names
    return None, None


def evaluate(run: Run, names: list[str] | None = None) -> dict:
    from . import figures as FIG

    plan = run.plan()
    ds = run.dataset()
    emb = run.embedder()
    seed = run.info["seed"]
    otype = plan["objective"]["type"]
    E_tr, E_te, tr_idx = _eval_embeddings(run, ds, emb)
    te = ds.splits["test"]
    allm = run.json("metrics.json", {})
    rng = np.random.default_rng(seed)
    eval_idx = np.sort(rng.choice(len(te), min(N_EVAL_MAX, len(te)), replace=False))
    ctx = context_for(ds, plan["use_labels"], seed)

    # Baseline without a generative model (for representation: the PCA/classifier embedding itself)
    if otype == "representation" and "baseline:" + emb.kind not in allm:
        lab, _ = _reference_labels(ds, "test")
        if lab is not None:
            allm["baseline:" + emb.kind] = M.latent_metrics(E_te, lab, seed)

    for c in plan["candidates"]:
        name = c["name"]
        if names and name not in names:
            continue
        mdir = run.path / "models" / name
        if not (mdir / "train.json").exists() or (name in allm and not names):
            continue
        model = GenerativeModel.load(mdir)
        y_req = te.y[eval_idx] if (plan["use_labels"] and te.y is not None) else None
        torch.manual_seed(seed)
        fake = model.sample(len(eval_idx), y_req)
        E_f = emb.transform(fake)
        m: dict[str, Any] = {}
        if not np.all(np.isfinite(E_f)):
            m["error"] = "non-finite samples"
            allm[name] = m
            continue
        m.update(M.general_metrics(E_tr, E_te[eval_idx], E_f, ds.splits["train"].y[tr_idx]
                                   if ds.splits["train"].y is not None else None,
                                   te.y[eval_idx] if te.y is not None else None, y_req, seed))
        if ds.modality == "image":
            m.update(M.image_metrics(emb, E_tr, E_te, fake, E_f, y_req, seed))
            FIG.image_grid(fake, y_req, ds.class_names, run.f("figures", f"samples_{name}.png"),
                           title=f"{name}: synthetic samples")
        elif ds.modality == "scrna":
            m.update(M.scrna_metrics(ds, emb, fake, E_f))
            FIG.umap_overlay(E_te[eval_idx], E_f, run.f("figures", f"umap_{name}.png"),
                             title=f"{name}: real (test) vs synthetic, PCA-UMAP", seed=seed)
        else:
            m.update(M.tabular_metrics(ds, ctx, fake))
            FIG.tabular_marginals(te.X, fake, ctx.col_types, run.f("figures", f"marginals_{name}.png"),
                                  title=f"{name}: marginals")
        if otype == "representation" and "encode" in model.capabilities:
            lab, _ = _reference_labels(ds, "test")
            if lab is not None:
                m.update(M.latent_metrics(model.encode(te.X), lab, seed))
        if otype == "anomaly_detection" and "anomaly" in model.capabilities:
            m.update(_anomaly_eval(model, ds, plan))
        allm[name] = m
    write_json(run.path / "metrics.json", allm)
    scores = M.score_candidates({k: v for k, v in allm.items() if not k.startswith("baseline:") and "error" not in v},
                                plan["metric_weights"])
    write_json(run.path / "scores.json", scores)
    run.log_command("evaluate", {"names": names})
    run.log_decision("evaluation", f"ranking: {scores['ranking']}", "weighted goodness score; "
                     "memorizing candidates excluded",
                     {k: v["score"] for k, v in scores["per_candidate"].items()})
    return {"scores": {k: v["score"] for k, v in scores["per_candidate"].items()}, "ranking": scores["ranking"],
            "excluded_memorization": [k for k, v in scores["per_candidate"].items() if v["excluded_memorization"]],
            "metrics": {k: _compact(v) for k, v in allm.items()}}


def _compact(m: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
            if not isinstance(v, (dict, list))}


def _anomaly_eval(model, ds: Dataset, plan: dict) -> dict:
    from sklearn.metrics import roc_auc_score

    te = ds.splits["test"]
    s = model.anomaly_score(te.X)
    out = {"anomaly_score_test_median": float(np.median(s))}
    holdout = (plan.get("anomaly") or {}).get("holdout_classes") or []
    if holdout:
        hid = [ds.class_names.index(c) for c in holdout]
        is_anom = np.isin(te.y, hid)
        if 0 < is_anom.sum() < len(is_anom):
            out["anomaly_auroc"] = float(roc_auc_score(is_anom, s))
    return out


# ================================================================ refine / select

def refine(run: Run, refine_file: str) -> list[dict]:
    """One refinement round: new candidates derived from trained ones with config overrides."""
    plan = run.plan()
    if plan.get("refinement_done"):
        raise PlanError("the single refinement round has already been used")
    spec = read_json(refine_file)
    items = spec if isinstance(spec, list) else [spec]
    ds = run.dataset()
    ctx = context_for(ds, plan["use_labels"], run.info["seed"])
    names = []
    for it in items:
        base = next((c for c in plan["candidates"] if c["name"] == it["from"]), None)
        if base is None:
            raise PlanError(f"unknown candidate {it['from']}")
        trained = read_json(run.path / "models" / base["name"] / "train.json")
        cls = REGISTRY[base["model"]]
        bad = set(it.get("config_overrides", {})) - set(cls.default_config(ctx))
        if bad:
            raise PlanError(f"unknown config keys {sorted(bad)}")
        name = it.get("name", base["name"] + "_refined")
        secs = run.info["budget_minutes"] * 60 * TRAIN_SHARE * base["time_fraction"] * it.get("time_scale", 0.5)
        plan["candidates"].append({"name": name, "model": base["model"], "refined_from": base["name"],
                                   "config": {**trained["final_config"], **it.get("config_overrides", {})},
                                   "search_space": {}, "hpo_trials": 0, "time_fraction": base["time_fraction"],
                                   "seconds": secs, "rationale": it.get("rationale", "")})
        names.append(name)
        run.log_decision("refinement", f"{name} from {base['name']} with {it.get('config_overrides')}",
                         it.get("rationale", ""))
    plan["refinement_done"] = True
    run.save_plan(plan)
    shutil.copy(refine_file, run.f("inputs", "refine_input.json"))
    run.log_command("refine", {"refine_file": "inputs/refine_input.json"})
    return train(run, names)


def select(run: Run, name: str | None = None, rationale: str = "") -> dict:
    scores = run.json("scores.json")
    if scores is None:
        raise PlanError("evaluate first")
    chosen = name or scores["best"]
    if chosen is None:
        raise PlanError("no eligible candidate (all excluded or failed)")
    if chosen not in scores["per_candidate"]:
        raise PlanError(f"unknown candidate {chosen}")
    plan = run.plan()
    plan["selected"] = chosen
    run.save_plan(plan)
    run.log_command("select", {"name": chosen, "rationale": rationale})
    run.log_decision("final_model", chosen, rationale or "highest weighted score",
                     {"score": scores["per_candidate"][chosen]["score"], "best_by_score": scores["best"]})
    return {"selected": chosen, "score": scores["per_candidate"][chosen]["score"], "best_by_score": scores["best"]}


# ================================================================ finalize

def finalize(run: Run) -> dict:
    """Generate the final synthetic data with the selected model and run the objective's downstream analysis."""
    from . import downstream

    plan = run.plan()
    ds = run.dataset()
    seed = run.info["seed"]
    otype = plan["objective"]["type"]
    out: dict[str, Any] = {"objective": otype, "mode": plan["objective"].get("mode")}
    if otype in DECLINED:
        out["note"] = "objective declined; no model trained"
        write_json(run.path / "downstream.json", out)
        run.log_command("finalize", {})
        return out
    name = plan.get("selected")
    if name is None:
        raise PlanError("select a model first")
    model = GenerativeModel.load(run.path / "models" / name)
    alloc = plan["allocation"]
    fake, y_fake = None, None
    if plan["generate"] and alloc["total"] > 0:
        seed_everything(seed + 1)
        y_fake = labels_from_allocation(alloc, ds.class_names if plan["use_labels"] else None)
        fake = model.sample(alloc["total"], y_fake)
        out["synthetic_file"] = downstream.save_synthetic(run, ds, fake, y_fake).relative_to(run.path).as_posix()
        out["n_synthetic"] = int(alloc["total"])
        out["per_class"] = alloc["per_class"]
    out.update(downstream.run_downstream(run, ds, plan, model, fake, y_fake))
    write_json(run.path / "downstream.json", out)
    run.log_command("finalize", {})
    run.log_decision("finalize", f"generated {out.get('n_synthetic', 0)} samples with {name}", "",
                     {k: v for k, v in out.items() if k in ("per_class", "synthetic_file")})
    return out


# ================================================================ free-form decision log entry

def note(run: Run, step: str, decision: str, rationale: str) -> None:
    run.log_decision(step, decision, rationale)
    run.log_command("note", {"step": step, "decision": decision, "rationale": rationale})
