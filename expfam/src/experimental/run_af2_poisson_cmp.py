"""AF2: automatic selection between Poisson and COM-Poisson (nu = 2) count
attributes, and its effect on recovery of the shared latent Z.

Why AF2 (after AF1, PR #166)
    In AF1 the {0,1} columns made the Bernoulli candidate dominate Poisson for
    every loading (log Bern - log Pois = e^eta - log(1+e^eta) > 0), so the
    selector could not fail.  Poisson vs COM-Poisson(nu=2) on count columns has
    no such pointwise ordering (log CMP2 - log Pois = -log x! + A_P(eta) -
    A_C(eta) changes sign with x), and PR #129 measured an expected
    per-observation KL of only ~0.03-0.06 between them.  Selection here is a
    genuine statistical decision.

Design (frozen in ``protocol()``; written before any AF2 result)
    data   PR #129 generator ``generate_multisupport_data`` (canonical-
           multisupport-v1), n = 75, K = 3, d = 9:
           Gaussian x3 (tau 1), Poisson x3, CMP(nu=2) x3 (tau 2, PR #129
           COUNT calibration value), Y Bernoulli (w0 = -1, w = 1),
           20 independent datasets (seed 1661001 + r)
    masks  identical rule and seeds to M1 / AF1 (``m1.build_masks(r)``):
           test 10 %, dense = 100 % / sparse = 10 % of the training pool
    candidates for count columns: {poisson, cmp_nu2p0}; Gaussian columns are
           gate-decided.  Optimiser route: trust_exact (PR #129 main run)
    methods (same data, train mask, test mask, model seed):
           oracle        true candidate models
           wrong         every count column fitted with the other candidate
                         (Poisson <-> CMP2 swap)
           support_only  PR #129 rule: first admissible candidate (= poisson
                         for every count column)
           auto_first    hybrid selector, START_FIRST (Poisson start; PRIMARY)
           auto_last     hybrid selector, START_LAST (CMP2 start; sensitivity)
    seeds  model seed 1662000 + r (oracle, wrong, support_only, Auto refit);
           Auto exploration seed 1663000 + r; one seed series
    metric Procrustes RMSE(Z) of the joint posterior mode (E1 evaluator, Z0 =
           last, fixed theta-hat, the fit's train mask).  last / mean5 also.

Lineage: E-consistent per-column + PR #129 generic model (experimental
prototype; not adoptable for the manuscript).  PR #129 is NOT merged: its
sources were copied unchanged (blob SHAs recorded) and only an opt-in
train_mask pass-through was added.

Execution integrity: fail-fast; a candidate failure inside exploration stops
the Auto fit (PR #129 rule: a failed candidate never lets another win).
FAILED fits are recorded and never re-run; interrupted fits are re-run on
--resume and logged in attempts.csv.

Usage
    python run_af2_poisson_cmp.py --write-protocol
    python run_af2_poisson_cmp.py --smoke
    python run_af2_poisson_cmp.py --execute [--resume]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
for p in (HERE, HERE.parent, ROOT / "tools" / "research_audit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_misspecification_m1 as m1                                    # noqa: E402
import run_z_point_estimator_e1 as e1                                   # noqa: E402
from run_auto_family_z_recovery import (                                # noqa: E402
    append_jsonl, load_jsonl, sha16, sha256_file, write_csv)

EXPERIMENT = "AF2"
N, K, D = 75, 3, 9
TRUE_MODELS = ["gaussian"] * 3 + ["poisson"] * 3 + ["cmp_nu2p0"] * 3
TAU = [1.0] * 3 + [2.0] * 6
W0, W = -1.0, 1.0
DATA_SEEDS = tuple(1661001 + r for r in range(20))
MODEL_SEED_BASE = 1662000
SEARCH_SEED_BASE = 1663000
REGIMES = tuple(m1.REGIMES)                                # dense, sparse
METHODS = ("oracle", "wrong", "support_only", "auto_first", "auto_last")
AUTO_START = {"auto_first": "START_FIRST", "auto_last": "START_LAST"}
PRIMARY_AUTO = "auto_first"
PRIMARY_ESTIMATOR = "mode"
COUNT_CANDIDATES = ("poisson", "cmp_nu2p0")
SWAP = {"poisson": "cmp_nu2p0", "cmp_nu2p0": "poisson", "gaussian": "gaussian"}
L, EXPLORATION_ITERS, REFIT_ITERS, NEWTON_ALPHA = 5, 8, 8, 0.5
ROUTE = "trust_exact"
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 10000, 20261110
PLANNED_FITS = len(DATA_SEEDS) * len(REGIMES) * len(METHODS)          # 200
PLANNED_EM_RUNS = len(DATA_SEEDS) * len(REGIMES) * (3 + 2 * 2)        # 280
PR129_COMMIT = "4fa6159"
PR129_BLOBS = {
    "family_specs.py": "24d257d5fed588d69843ab5ccbec302c09db6bf3",
    "model_dual_expfam_generic.py": "52e52b05a871ce8e95c54cdeabd321e5223e6691",
    "family_selection_multisupport.py": "1790930fc7822be3833a1c8b1076104e6c79761d",
    "data_generator_multisupport.py": "053ff3e63ca15a50c3f284636f21ac3c1c67545e",
}

OUT_ROOT = ROOT / "expfam" / "results" / "auto_family_z_recovery" / "af2_20261110"
STATE_ROOT = Path(r"D:\tento\kennkyu_artifacts") / "auto_family_z_recovery_af2_20261110"


def registry_and_priority():
    from family_specs import (GROUP_COUNT, PRIMARY_PRIORITY,
                              build_primary_registry)
    reg = build_primary_registry()
    prio = {g: tuple(v) for g, v in PRIMARY_PRIORITY.items()}
    prio[GROUP_COUNT] = COUNT_CANDIDATES            # AF2 candidate set
    return reg, prio


def protocol() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "mode": "CONFIRMATORY (pre-registered before any AF2 result was read)",
        "research_question": (
            "When the attribute family choice is a genuine statistical decision "
            "(Poisson vs COM-Poisson nu=2 count columns), how often does automatic "
            "selection recover the true family, and how much of the Z-recovery loss "
            "from misspecification does it prevent?"),
        "lineage": "E-consistent per-column + PR #129 generic model (experimental prototype; "
                   "not adoptable for the manuscript)",
        "pr129": {"commit": PR129_COMMIT, "merged": False,
                  "copied_unchanged_blobs": PR129_BLOBS,
                  "local_change": "opt-in train_mask pass-through in run_em_generic, "
                                  "run_generic_exploration, run_generic_hybrid"},
        "generator": "data_generator_multisupport.generate_multisupport_data (PR #129, unchanged)",
        "data": {"n": N, "K_true": K, "K_fit": K, "d": D, "true_models": TRUE_MODELS,
                 "tau": TAU, "tau_source": "PR #129: Gaussian tau_real_mixed = 1; COUNT "
                 "calibration value 2.0 (label 'EASY' there; chosen here because the "
                 "count columns then carry a strong latent signal)",
                 "family_y": "bernoulli", "w0": W0, "w": W,
                 "data_seeds": list(DATA_SEEDS)},
        "masks": {"source": "run_misspecification_m1.build_masks(r) (same seeds as M1/AF1)",
                  "test_ratio": m1.TEST_RATIO, "regimes": dict(m1.REGIMES)},
        "candidates": {"count": list(COUNT_CANDIDATES), "real": ["gaussian"],
                       "other_groups": "PR #129 PRIMARY_PRIORITY (not expected to occur)",
                       "route": ROUTE, "tie_rule": "priority order (poisson first)",
                       "failed_candidate_rule": "PR #129: the Auto fit stops (SelectorStop); "
                                                "no other candidate wins by default"},
        "methods": {
            "oracle": "true candidate models",
            "wrong": "Poisson <-> CMP2 swap on every count column",
            "support_only": "PR #129 rule: first admissible candidate (poisson on every count column)",
            "auto_first": "hybrid selector, START_FIRST (Poisson start) -- PRIMARY",
            "auto_last": "hybrid selector, START_LAST (CMP2 start) -- sensitivity",
            "note": "support_only equals the one-directional misspecification "
                    "(CMP2 columns as Poisson); wrong misspecifies all six count columns"},
        "em": {"L": L, "exploration_iters": EXPLORATION_ITERS, "refit_iters": REFIT_ITERS,
               "newton_alpha": NEWTON_ALPHA, "numerics": "consistent / generic", "fail_fast": True},
        "seeds": {"model_seed": f"{MODEL_SEED_BASE} + r (all fits incl. Auto refit)",
                  "search_seed": f"{SEARCH_SEED_BASE} + r (both Auto starts)",
                  "series": 1},
        "primary_metric": "Procrustes RMSE(Z) of the joint posterior mode (E1 evaluator)",
        "primary_contrasts": {
            "delta_wrong": "RMSE(wrong) - RMSE(oracle)",
            "delta_auto": "RMSE(auto_first) - RMSE(oracle)",
            "benefit_wrong": "RMSE(wrong) - RMSE(auto_first)"},
        "key_secondary_contrasts": {
            "delta_support_only": "RMSE(support_only) - RMSE(oracle)",
            "benefit_support_only": "RMSE(support_only) - RMSE(auto_first)",
            "auto_last_minus_auto_first": "RMSE(auto_last) - RMSE(auto_first)"},
        "statistical_unit": "replicate (20 independent datasets per regime)",
        "interval": {"type": "percentile bootstrap of the mean of replicate-level contrasts",
                     "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "secondary_metrics": [
            "true-family recovery per column (count columns, by true family)",
            "gate-decided / score-decided columns", "final margins and scores",
            "start agreement", "candidate optimiser convergence and failures",
            "E-step backtracks", "held-out Y log score / AUC", "RMSE(Z) last / mean5",
            "fit and mode failure rates"],
        "planned_fits": PLANNED_FITS, "planned_em_runs": PLANNED_EM_RUNS,
        "missing_and_failure_rule": (
            "FAILED fits and non-OK modes are kept, never retried or imputed; contrasts only "
            "where both members are OK, exclusions counted next to every summary, together "
            "with failure rates. Success-only accuracy is always reported with the failure rate."),
        "technical_stop_conditions": [
            "a test pair inside any train mask", "data/mask hash differing between methods",
            "Auto refit train mask differing from the cell's train mask",
            "protocol.json differing from this protocol"],
        "warnings_not_stops": ["candidate optimiser non-convergence", "non-OK joint mode",
                               "E-step backtracks", "selector failures (they are results)"],
        "artifacts": {"results": "expfam/results/auto_family_z_recovery/af2_20261110",
                      "states_outside_repo": str(STATE_ROOT)},
    }


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True).encode()).hexdigest()


def generate(r: int):
    from data_generator_multisupport import generate_multisupport_data
    reg, _ = registry_and_priority()
    return generate_multisupport_data(n=N, k=K, seed=DATA_SEEDS[r],
                                      column_specs=[reg[m] for m in TRUE_MODELS],
                                      tau=TAU, w0=W0, w=W)


def planned_fits() -> list[dict[str, Any]]:
    plan = []
    for r, seed in enumerate(DATA_SEEDS):
        for regime in REGIMES:
            for method in METHODS:
                plan.append({"r": r, "replicate": m1.replicate_label(r), "data_seed": seed,
                             "regime": regime, "method": method,
                             "model_seed": MODEL_SEED_BASE + r,
                             "search_seed": SEARCH_SEED_BASE + r if method in AUTO_START else "",
                             "start": AUTO_START.get(method, "")})
    return plan


def fit_key(fit) -> str:
    return f"{fit['replicate']}_{fit['regime']}_{fit['method']}"


def run_fit(fit, ds, masks):
    from family_selection_multisupport import gate_columns, run_generic_hybrid
    from model_dual_expfam_generic import run_em_generic

    reg, prio = registry_and_priority()
    base = {k: fit[k] for k in ("replicate", "data_seed", "regime", "method",
                                "model_seed", "search_seed", "start")}
    train, test = masks[fit["regime"]], masks["test"]
    run = dict(base, fit_key=fit_key(fit), n_train_pairs=m1.n_upper(train),
               n_test_pairs=m1.n_upper(test), x_sha=sha16(ds.X), y_sha=sha16(ds.Y),
               z_true_sha=sha16(ds.Z), train_mask_sha=sha16(train), test_mask_sha=sha16(test),
               train_test_overlap=int(np.triu(train & test, 1).sum()))
    est_rows, col_rows, trace_rows, states = [], [], [], {}
    stage = ""
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        try:
            gates = gate_columns(ds.X, reg, prio)
            run["gate_groups"] = "|".join(g.support_group for g in gates)
            run["n_score_decided"] = sum(g.is_ambiguous for g in gates)
            if fit["method"] in AUTO_START:
                stage = "exploration+refit"
                hyb = run_generic_hybrid(
                    ds.X, ds.Y, k=K, registry=reg, priority=prio, start=fit["start"],
                    L=L, exploration_num_iter=EXPLORATION_ITERS, refit_num_iter=REFIT_ITERS,
                    search_seed=fit["search_seed"], refit_seed=fit["model_seed"],
                    newton_alpha=NEWTON_ALPHA, route=ROUTE, train_mask=train)
                res = hyb["refit"]
                ex = hyb["exploration"]
                assign = list(ex["selected_models"])
                trace = ex["selection_trace"]
                last_it = max((t["iteration"] for t in trace), default=0)
                final = {t["column"]: t for t in trace if t["iteration"] == last_it}
                nchg, first = {}, {}
                for t in trace:
                    if t.get("changed"):
                        nchg[t["column"]] = nchg.get(t["column"], 0) + 1
                        first.setdefault(t["column"], t["iteration"])
                for g in gates:
                    t = final.get(g.column, {})
                    scores = json.loads(t["candidate_scores_json"]) if t.get("candidate_scores_json") else {}
                    grads = json.loads(t["candidate_grad_inf_json"]) if t.get("candidate_grad_inf_json") else {}
                    col_rows.append(dict(
                        base, column=g.column, true_model=TRUE_MODELS[g.column],
                        support_group=g.support_group, admissible="|".join(g.admissible),
                        decided_by=g.decided_by, start_model=hyb["start_models"][g.column],
                        selected_model=assign[g.column],
                        correct=assign[g.column] == TRUE_MODELS[g.column],
                        final_score_poisson=scores.get("poisson", ""),
                        final_score_cmp_nu2p0=scores.get("cmp_nu2p0", ""),
                        final_margin_neg2=t.get("margin_neg2", ""),
                        final_grad_inf_poisson=grads.get("poisson", ""),
                        final_grad_inf_cmp_nu2p0=grads.get("cmp_nu2p0", ""),
                        final_all_converged=t.get("all_candidates_converged", ""),
                        n_changes=nchg.get(g.column, 0) if g.is_ambiguous else "",
                        first_change_iteration=first.get(g.column, "") if g.is_ambiguous else ""))
                for t in trace:
                    trace_rows.append(dict(base, **t))
                refit_mask = np.asarray(res["model"].train_mask, bool)
                run.update({
                    "exploration_estep_backtracks": ex["estep_backtracks"],
                    "n_nonconverged_trace_rows": sum(1 for t in trace
                                                     if t.get("all_candidates_converged") is False),
                    "max_trace_grad_inf": max((float(t.get("max_candidate_grad_inf", 0.0))
                                               for t in trace), default=float("nan")),
                })
            else:
                stage = "fixed_fit"
                if fit["method"] == "oracle":
                    assign = list(TRUE_MODELS)
                elif fit["method"] == "wrong":
                    assign = [SWAP[m] for m in TRUE_MODELS]
                else:                                            # support_only
                    assign = [g.admissible[0] for g in gates]
                res = run_em_generic(ds.X, ds.Y, column_specs=[reg[c] for c in assign],
                                     family_y="bernoulli", k=K, L=L, num_iter=REFIT_ITERS,
                                     seed=fit["model_seed"], newton_alpha=NEWTON_ALPHA,
                                     train_mask=train)
                refit_mask = np.asarray(res["model"].train_mask, bool)
            stage = "evaluation"
            run.update({
                "assignment": "|".join(assign),
                "n_correct_columns": sum(a == t for a, t in zip(assign, TRUE_MODELS)),
                "n_correct_count_columns": sum(a == t for a, t in zip(assign[3:], TRUE_MODELS[3:])),
                "all_correct": assign == TRUE_MODELS,
                "refit_train_mask_sha": sha16(refit_mask),
                "refit_mask_matches": bool(np.array_equal(refit_mask, train)),
                "refit_estep_backtracks": res.get("estep_backtracks", 0),
                "Q_strict": res.get("Q_strict", ""),
            })
            Zs = np.asarray(res["Z_samples"], float)
            w0, w = float(res["w0"]), float(res["w"])
            mode = e1.joint_mode(res["model"], ds.X, ds.Y, Zs[:, :, -1])
            est = e1.point_estimates(Zs, mode["Z_hat"] if mode["Z_hat"] is not None
                                     else np.full_like(Zs[:, :, -1], np.nan))
            for name, Z in est.items():
                ok = bool(np.all(np.isfinite(Z))) and (name != "mode" or mode["status"] == "OK")
                row = dict(base, fit_key=fit_key(fit), estimator=name,
                           estimator_status="OK" if ok else
                           (f"MODE_{mode['status']}" if name == "mode" else "NONFINITE"))
                if np.all(np.isfinite(Z)):
                    row.update(e1.score_estimate(Z, ds.Z, ds.Y, test, w0, w))
                est_rows.append(row)
            run.update({"status": "OK", "error": "", "w0_hat": w0, "w_hat": w,
                        "retry_count": res["retry_count"],
                        "replacement_count": res["replacement_count"],
                        "seed_rescue_count": res["seed_rescue_count"],
                        "em_runtime_s": res["runtime_s"],
                        **{f"mode_{k}": v for k, v in mode.items() if k != "Z_hat"}})
            states = {"Zsamples": Zs, "F": np.asarray(res["F"], float),
                      "sigma": np.asarray(res["sigma"], float),
                      "scalars": np.array([w0, w, float(res["var_z"])]),
                      **({"Zmode": mode["Z_hat"]} if mode["Z_hat"] is not None else {})}
        except Exception as exc:                 # recorded, never retried or replaced
            run.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"[:500],
                        "failed_stage": stage, "traceback_tail": traceback.format_exc()[-800:]})
            if fit["method"] in AUTO_START:
                # every column still counts in the recovery denominator; rows
                # written before the failure are replaced, never duplicated
                # (pre-run audit D5)
                col_rows = []
                for g_col in range(D):
                    col_rows.append(dict(base, column=g_col, true_model=TRUE_MODELS[g_col],
                                         selected_model="", correct=False,
                                         decided_by="", selection_status="FIT_FAILED"))
        run["n_warnings"] = len(wl)
        run["warning_messages"] = " || ".join(sorted(
            {f"{x.category.__name__}: {str(x.message)[:160]}" for x in wl}))[:1000]
    run["wall_runtime_s"] = round(time.perf_counter() - t0, 2)
    return run, est_rows, col_rows, trace_rows, states


def data_audit(cache) -> list[dict[str, Any]]:
    rows = []
    for r, (ds, masks) in sorted(cache.items()):
        dense, sparse, test = masks["dense"], masks["sparse"], masks["test"]
        rows.append({
            "replicate": m1.replicate_label(r), "data_seed": DATA_SEEDS[r],
            "x_sha": sha16(ds.X), "y_sha": sha16(ds.Y), "z_true_sha": sha16(ds.Z),
            "dense_sha": sha16(dense), "sparse_sha": sha16(sparse), "test_sha": sha16(test),
            "n_test": m1.n_upper(test), "n_dense": m1.n_upper(dense), "n_sparse": m1.n_upper(sparse),
            "dense_test_overlap": int(np.triu(dense & test, 1).sum()),
            "sparse_test_overlap": int(np.triu(sparse & test, 1).sum()),
            "sparse_subset_of_dense": bool(np.all(dense[sparse])),
            "x_col_max": "|".join(f"{v:.3g}" for v in ds.X.max(axis=0)),
            "x_col_mean": "|".join(f"{v:.3g}" for v in ds.X.mean(axis=0)),
            "x_col_var_over_mean_counts": "|".join(
                f"{ds.X[:, j].var() / max(ds.X[:, j].mean(), 1e-12):.3g}" for j in range(3, D)),
            "y_density": float(ds.metadata["y_density"]),
            "train_density_sparse": float(ds.Y[np.triu(sparse, 1)].mean()),
        })
    return rows


def runinfo(extra):
    import scipy
    import sklearn
    req = ROOT / "requirements.txt"
    return {"experiment": EXPERIMENT, "script": "expfam/src/experimental/run_af2_poisson_cmp.py",
            "git_head": m1.git("rev-parse", "HEAD"),
            "git_branch": m1.git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_tracked": bool(m1.git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "platform": platform.platform(),
            "requirements_sha256": sha256_file(req) if req.exists() else "",
            "protocol_sha256": protocol_sha256(), "command": " ".join(sys.argv), **extra}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write-protocol", action="store_true")
    g.add_argument("--smoke", action="store_true", help="rep01, dense and sparse, 5 methods")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--state-root", default=str(STATE_ROOT))
    args = ap.parse_args(argv)
    out_root, state_root = Path(args.out_root), Path(args.state_root)

    if args.write_protocol:
        out_root.mkdir(parents=True, exist_ok=True)
        p = out_root / "protocol.json"
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
        p.write_text(json.dumps(protocol(), indent=2), encoding="utf-8")
        write_csv(out_root / "manifest.csv", planned_fits())
        print(f"wrote {p} sha256(canonical)={protocol_sha256()}")
        return 0

    frozen = json.loads((out_root / "protocol.json").read_text(encoding="utf-8"))
    if hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest() != protocol_sha256():
        raise SystemExit("protocol.json on disk differs from protocol(); refusing to run")
    stage = "smoke" if args.smoke else "full"
    out, state_dir = out_root / stage, state_root / stage
    if out.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite {out}")
    if args.resume and (args.smoke or not out.exists()):
        raise SystemExit("--resume only continues an existing --execute output")
    out.mkdir(parents=True, exist_ok=True)
    (state_dir / "fits").mkdir(parents=True, exist_ok=True)

    plan = planned_fits()
    if args.smoke:
        plan = [f for f in plan if f["replicate"] == "rep01"]
    progress = out / "progress.jsonl"
    events = load_jsonl(progress)
    finished = {e["fit_key"] for e in events if e["event"] == "FINISHED"}
    interrupted = sorted({e["fit_key"] for e in events if e["event"] == "STARTED"} - finished)

    def load(name):
        p = out / name
        if not p.exists():
            return []
        with open(p, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    names = ("fit_runs.csv", "fit_estimators.csv", "selection_columns.csv", "selection_trace.csv")
    tables = {n: load(n) for n in names}
    if interrupted:
        bad = set(interrupted)
        for n in tables:
            tables[n] = [r for r in tables[n] if r.get("fit_key", "_".join(
                [r.get("replicate", ""), r.get("regime", ""), r.get("method", "")])) not in bad]
    attempts = load("attempts.csv")
    for k in interrupted:
        attempts.append({"fit_key": k, "event": "INTERRUPTED_BEFORE_FINISH",
                         "action": "re-run on resume with identical seeds (not a failure retry)",
                         "time_utc": datetime.now(timezone.utc).isoformat()})

    t_all = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    cache: dict[int, Any] = {}
    todo = [f for f in plan if fit_key(f) not in finished]
    print(f"{stage}: planned {len(plan)}, finished {len(finished)}, to run {len(todo)}, "
          f"interrupted-to-rerun {len(interrupted)}", flush=True)
    for i, fit in enumerate(todo, 1):
        if fit["r"] not in cache:
            cache[fit["r"]] = (generate(fit["r"]), m1.build_masks(fit["r"]))
        ds, masks = cache[fit["r"]]
        key = fit_key(fit)
        append_jsonl(progress, {"event": "STARTED", "fit_key": key,
                                "time_utc": datetime.now(timezone.utc).isoformat()})
        run, est, cols, trace, st = run_fit(fit, ds, masks)
        if st:
            p = state_dir / "fits" / f"{key}.npz"
            np.savez_compressed(p, **st)
            run["state_file"], run["state_sha256"] = p.name, sha256_file(p)
        tables["fit_runs.csv"].append(run)
        tables["fit_estimators.csv"].extend(est)
        tables["selection_columns.csv"].extend(cols)
        tables["selection_trace.csv"].extend(trace)
        attempts.append({"fit_key": key, "event": run["status"],
                         "action": "first and only execution" if key not in interrupted
                         else "re-run after interruption",
                         "time_utc": datetime.now(timezone.utc).isoformat()})
        for n, rows in tables.items():
            if rows:
                write_csv(out / n, rows)
        write_csv(out / "attempts.csv", attempts)
        append_jsonl(progress, {"event": "FINISHED", "fit_key": key, "status": run["status"],
                                "time_utc": datetime.now(timezone.utc).isoformat()})
        mode = next((r for r in est if r["estimator"] == "mode"), {})
        print(f"[{i}/{len(todo)}] {key} {run['status']} cnt_ok={run.get('n_correct_count_columns', '-')} "
              f"mode={run.get('mode_status', '-')} rmse_mode={mode.get('rmse_Z', float('nan')):.4f} "
              f"t={run['wall_runtime_s']}s {run.get('error', '')[:80]}", flush=True)

    for r in sorted({f["r"] for f in plan}):
        if r not in cache:
            cache[r] = (generate(r), m1.build_masks(r))
    write_csv(out / "data_mask_audit.csv", data_audit(cache))
    prev = json.loads((out / "runinfo.json").read_text(encoding="utf-8")) if (out / "runinfo.json").exists() else {}
    sessions = prev.get("sessions", []) + [{
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_s": round(time.perf_counter() - t_all, 1), "fits_run": len(todo),
        "git_head": m1.git("rev-parse", "HEAD"), "command": " ".join(sys.argv)}]
    (out / "runinfo.json").write_text(json.dumps(runinfo({
        "stage": stage, "planned_fits": len(plan), "finished_fits": len(finished) + len(todo),
        "state_dir": str(state_dir / "fits"), "sessions": sessions,
        "seeds": {"data": "1661001 + r", "model": f"{MODEL_SEED_BASE} + r",
                  "search": f"{SEARCH_SEED_BASE} + r", "bootstrap": BOOTSTRAP_SEED},
    }), indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
