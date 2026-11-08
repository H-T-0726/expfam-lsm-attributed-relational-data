"""AF1: does automatic X-family selection prevent the Z-recovery loss of a
Bernoulli->Poisson misspecification?  (Oracle / Auto / Wrong, M1 data)

Research question
    When misspecifying the observation family of attribute data X degrades
    recovery of the shared latent structure Z, how much of that degradation
    does automatic per-column family selection prevent?

Design (frozen in ``protocol()``; nothing here is tuned on results)
    * data, test mask, training pool, dense/sparse thinning: exactly M1
      (``run_misspecification_m1.generate`` / ``build_masks``), 20 replicates;
    * three methods on the SAME data, train mask and test mask:
        Oracle  true families G3/B6/P3, fixed;
        Wrong   the six Bernoulli columns fitted as Poisson (G3/P6/P3), fixed;
        Auto    Phase 9C hybrid selector (support gate -> per-column score on
                the fixed E-step samples, BFGS loading per candidate,
                8 exploration iterations) then a fresh fixed-family refit
                (8 iterations).  Two pre-registered starts for the {0,1}
                columns: Poisson start (PRIMARY) and Bernoulli start
                (sensitivity).  The train mask is passed to BOTH stages;
    * two model-seed series taken from E1 (A = 1343000 + r, B = 1344000 + r).
      Oracle / Wrong use the model seed exactly as E1; Auto's refit uses the
      same model seed, its exploration a separate pre-registered search seed;
    * Z evaluation: E1's estimators (last, mean5, joint posterior mode from
      Z0 = last at fixed theta-hat with the fit's train mask), Procrustes
      RMSE(Z).  The primary estimator is the joint mode.

Lineage: E-consistent per-column (experimental prototype; not adoptable for
the manuscript).  No E-step, objective, selector, estimator or numerics was
changed; ``family_selection`` only gained an opt-in ``train_mask`` pass-through.

Execution integrity: fail-fast, no retry / replacement / seed rescue.  A failed
fit is recorded as FAILED and never re-run.  A fit that was STARTED but never
finished (process interrupted) is re-run on resume with the same seeds and is
counted in ``attempts.csv``.

Usage
    python run_auto_family_z_recovery.py --write-protocol
    python run_auto_family_z_recovery.py --smoke
    python run_auto_family_z_recovery.py --execute [--resume]
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

# ── Frozen design ─────────────────────────────────────────────────────────
EXPERIMENT = "AF1"
SERIES = {"A": e1.PARTS["A"], "B": e1.PARTS["B"]}         # model seed = base + r
SEARCH_SEED_BASE = {"A": 1345000, "B": 1346000}           # exploration seed = base + r
REGIMES = tuple(m1.REGIMES)                                # ("dense", "sparse")
METHODS = ("oracle", "wrong", "auto_pstart", "auto_bstart")
AUTO_START = {"auto_pstart": "poisson", "auto_bstart": "bernoulli"}
PRIMARY_AUTO = "auto_pstart"
PRIMARY_ESTIMATOR = "mode"
ESTIMATORS = e1.ESTIMATORS
FIXED_FAMILIES = {"oracle": list(m1.ARMS["correct"]), "wrong": list(m1.ARMS["wrong"])}
EXPLORATION_ITERS = 8
REFIT_ITERS = m1.NUM_ITER                                  # 8
L = m1.L                                                   # 5
CANDIDATE_OPTIMIZER = "bfgs"                               # Phase 9C default
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261109
N_REPLICATES = len(m1.DATA_SEEDS)
PLANNED_FITS = len(SERIES) * N_REPLICATES * len(REGIMES) * len(METHODS)   # 320
PLANNED_EM_RUNS = len(SERIES) * N_REPLICATES * len(REGIMES) * (2 + 2 * 2)  # 480

OUT_ROOT = ROOT / "expfam" / "results" / "auto_family_z_recovery" / "af1_20261109"
STATE_ROOT = Path(r"D:\tento\kennkyu_artifacts") / "auto_family_z_recovery_af1_20261109"


def protocol() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "mode": "CONFIRMATORY (pre-registered before any AF1 result was read)",
        "research_question": (
            "When misspecifying the observation family of attribute data X degrades "
            "recovery of the shared latent structure Z, how much of that degradation "
            "does automatic per-column family selection prevent?"),
        "lineage": "E-consistent per-column (experimental prototype; not adoptable for the manuscript)",
        "half_factor": "adopted equations, no extra 1/2 (fixed lineage via DualExpFamLSMMasked)",
        "generator": "canonical-clean-mixed-v1 via run_misspecification_m1.generate (M1, unchanged)",
        "data": {"n": m1.N, "d": m1.D, "K_true": m1.K_TRUE, "K_fit": m1.K_FIT,
                 "family_x_true": list(m1.FAMILY_X_TRUE), "family_y": m1.FAMILY_Y,
                 "w0": m1.W0_TRUE, "w": m1.W_TRUE, "f_scale": m1.F_SCALE,
                 "gaussian_x_variance": m1.SIGMA_X_VAR,
                 "data_seeds": list(m1.DATA_SEEDS), "n_replicates": N_REPLICATES},
        "masks": {"source": "run_misspecification_m1.build_masks(r) (unchanged)",
                  "test_ratio": m1.TEST_RATIO, "regimes": dict(m1.REGIMES),
                  "split_seed": "1341000 + r", "thin_seed": "1342000 + 10 r + i",
                  "rule": "all methods and both Auto stages see the same train mask; "
                          "test pairs are never in any train mask"},
        "methods": {
            "oracle": {"families": FIXED_FAMILIES["oracle"], "fit": "run_em_experimental, fixed"},
            "wrong": {"families": FIXED_FAMILIES["wrong"], "fit": "run_em_experimental, fixed"},
            "auto_pstart": {"selector": "family_selection.run_hybrid_family_selection",
                            "ambiguous_start": "poisson", "role": "PRIMARY Auto"},
            "auto_bstart": {"selector": "family_selection.run_hybrid_family_selection",
                            "ambiguous_start": "bernoulli", "role": "sensitivity (start dependence)"},
            "auto_settings": {"exploration_iters": EXPLORATION_ITERS, "refit_iters": REFIT_ITERS,
                              "candidate_optimizer": CANDIDATE_OPTIMIZER,
                              "candidate_set_for_01_columns": ["bernoulli", "poisson"],
                              "score": "Q_l(m) = mean over the L fixed E-step samples of "
                                       "sum_i log p_m(x_il | f_lm^T z_i), full normaliser, "
                                       "f_lm optimised per candidate",
                              "tie_rule": "bernoulli > poisson > gaussian (exact ties only)"},
            "primary_auto_start_rationale": (
                "The Poisson start begins at the Wrong assignment, so any recovery is "
                "attributable to the selector and not to the initial assignment."),
        },
        "em": {"L": L, "num_iter": REFIT_ITERS, "numerics_mode": "consistent",
               "failure_policy": "fail_fast", "K_fixed": m1.K_FIT},
        "seeds": {"series": {s: f"{b} + r" for s, b in SERIES.items()},
                  "oracle_wrong_seed": "series model seed (identical to E1 Part A / B)",
                  "auto_refit_seed": "series model seed",
                  "auto_search_seed": {s: f"{b} + r" for s, b in SEARCH_SEED_BASE.items()},
                  "note": "the two series and the two starts reuse the same data; they are "
                          "NOT independent datasets"},
        "primary_metric": "Procrustes RMSE(Z) of the joint posterior mode (E1 'mode', Z0 = last)",
        "primary_contrasts": {
            "delta_wrong": "RMSE(wrong) - RMSE(oracle)",
            "delta_auto": "RMSE(auto_pstart) - RMSE(oracle)",
            "benefit": "RMSE(wrong) - RMSE(auto_pstart)"},
        "statistical_unit": "replicate (20 independent datasets per regime)",
        "replicate_aggregation": (
            "per replicate and regime, the contrast is the mean over the seed series in which "
            "both fits and both modes are OK; series results are also reported separately"),
        "interval": {"type": "percentile bootstrap of the mean of replicate-level contrasts",
                     "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "secondary_metrics": [
            "RMSE(Z) for last and mean5", "true-family recovery per column",
            "number of gate-decided and score-decided columns", "final margins and scores",
            "Bernoulli-start vs Poisson-start agreement", "held-out Y mean log score and AUC",
            "gram correlation", "candidate optimiser convergence", "fit and mode failure rates",
            "reproduction of E1 oracle/wrong RMSE (diagnostic)"],
        "planned_fits": PLANNED_FITS, "planned_em_runs": PLANNED_EM_RUNS,
        "missing_and_failure_rule": (
            "FAILED fits and non-OK modes are kept, never retried or imputed. A contrast is "
            "computed only where both members are OK; the number excluded is reported next to "
            "every summary together with the failure rate."),
        "technical_stop_conditions": [
            "a test pair inside any train mask", "data or mask hash differing between methods of a cell",
            "Auto refit train mask differing from the cell's train mask",
            "retry / replacement / seed-rescue count > 0",
            "protocol.json on disk differing from this protocol"],
        "warnings_not_stops": [
            "candidate optimiser non-convergence (recorded; the selection stands)",
            "non-OK joint mode (recorded; excluded pairwise)",
            "E1 reproduction mismatch (diagnostic)"],
        "artifacts": {"results": str(OUT_ROOT.relative_to(ROOT)).replace("\\", "/"),
                      "states_outside_repo": str(STATE_ROOT)},
    }


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True).encode()).hexdigest()


# ── plan ──────────────────────────────────────────────────────────────────

def planned_fits() -> list[dict[str, Any]]:
    plan = []
    for series, base in SERIES.items():
        for r, data_seed in enumerate(m1.DATA_SEEDS):
            for regime in REGIMES:
                for method in METHODS:
                    plan.append({
                        "series": series, "r": r, "replicate": m1.replicate_label(r),
                        "data_seed": data_seed, "regime": regime, "method": method,
                        "model_seed": base + r,
                        "search_seed": (SEARCH_SEED_BASE[series] + r
                                        if method in AUTO_START else ""),
                        "ambiguous_start": AUTO_START.get(method, "")})
    return plan


def fit_key(fit: dict[str, Any]) -> str:
    return f"{fit['series']}_{fit['replicate']}_{fit['regime']}_{fit['method']}"


def sha16(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── one fit ───────────────────────────────────────────────────────────────

def _fixed_fit(fit, ds, train_mask):
    from em_runner import run_em_experimental                            # noqa: E402
    # identical call to run_misspecification_m1.run_fit / E1 run_fit
    return run_em_experimental(
        ds.X, ds.Y, family_x=None, family_y=m1.FAMILY_Y, k=m1.K_FIT, L=L,
        num_iter=REFIT_ITERS, seed=fit["model_seed"], train_mask=train_mask,
        family_x_list=list(FIXED_FAMILIES[fit["method"]]), compute_strict_Q=False,
        validate_support=True, allow_support_mismatch=False,
        numerics_mode="consistent", failure_policy="fail_fast")


def _auto_fit(fit, ds, train_mask, stage_log):
    from family_selection import run_hybrid_family_selection               # noqa: E402

    def hook(kind, status, info):
        stage_log.append(f"{kind}:{status}")
    return run_hybrid_family_selection(
        ds.X, ds.Y, k=m1.K_FIT, ambiguous_start=fit["ambiguous_start"],
        family_y=m1.FAMILY_Y, L=L, exploration_num_iter=EXPLORATION_ITERS,
        refit_num_iter=REFIT_ITERS, search_seed=fit["search_seed"],
        refit_seed=fit["model_seed"], execution_hook=hook,
        candidate_optimizer=CANDIDATE_OPTIMIZER, train_mask=train_mask)


def run_fit(fit, ds, masks):
    """Return (run_row, estimator_rows, column_rows, trace_rows, candidate_rows, states)."""
    base = {k: fit[k] for k in ("series", "replicate", "data_seed", "regime", "method",
                                "model_seed", "search_seed", "ambiguous_start")}
    train, test = masks[fit["regime"]], masks["test"]
    run = dict(base, fit_key=fit_key(fit),
               n_train_pairs=m1.n_upper(train), n_test_pairs=m1.n_upper(test),
               x_sha=sha16(ds.X), y_sha=sha16(ds.Y), z_true_sha=sha16(ds.Z),
               train_mask_sha=sha16(train), test_mask_sha=sha16(test),
               train_test_overlap=int(np.triu(train & test, 1).sum()))
    est_rows, col_rows, trace_rows, cand_rows, states = [], [], [], [], {}
    stage_log: list[str] = []
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        try:
            if fit["method"] in AUTO_START:
                hyb = _auto_fit(fit, ds, train, stage_log)
                res = hyb["refit"]
                families = list(hyb["selected_assignment"])
                gates = {g["column"]: g for g in hyb["gates"]}
                last_it = max((t["iteration"] for t in hyb["selection_trace"]), default=0)
                final = {t["column"]: t for t in hyb["selection_trace"]
                         if t["iteration"] == last_it}
                first_change = {}
                for t in hyb["selection_trace"]:
                    if t["changed"]:
                        first_change.setdefault(t["column"], t["iteration"])
                n_switch = {}
                for t in hyb["selection_trace"]:
                    n_switch[t["column"]] = n_switch.get(t["column"], 0) + int(bool(t["changed"]))
                for j in range(m1.D):
                    g, t = gates[j], final.get(j)
                    col_rows.append(dict(
                        base, column=j, true_family=m1.FAMILY_X_TRUE[j],
                        gate_candidates=g["candidates"], decided_by=g["decided_by"],
                        initial_family=hyb["initial_assignment"][j],
                        selected_family=families[j],
                        correct=families[j] == m1.FAMILY_X_TRUE[j],
                        final_score_bernoulli=t.get("score_bernoulli", "") if t else "",
                        final_score_poisson=t.get("score_poisson", "") if t else "",
                        final_margin_neg2=t["margin_neg2"] if t else "",
                        final_all_candidates_converged=t["all_candidates_converged"] if t else "",
                        n_changes=n_switch.get(j, 0) if t else "",
                        first_change_iteration=first_change.get(j, "") if t else ""))
                for t in hyb["selection_trace"]:
                    trace_rows.append(dict(base, **{k: v for k, v in t.items()
                                                    if k != "selected_loading"}))
                for c in hyb["candidate_rows"]:
                    cand_rows.append(dict(base, iteration=last_it, **c))
                refit_mask = np.asarray(res["model"].train_mask, bool)
                run.update({
                    "selected_assignment": "|".join(families),
                    "n_gate_decided": sum(1 for g in hyb["gates"] if g["decided_by"] == "gate"),
                    "n_score_decided": sum(1 for g in hyb["gates"] if g["decided_by"] == "score"),
                    "n_correct_columns": sum(f == t for f, t in zip(families, m1.FAMILY_X_TRUE)),
                    "all_correct": families == list(m1.FAMILY_X_TRUE),
                    "selected_equals_oracle": families == FIXED_FAMILIES["oracle"],
                    "convergence_gate_status": hyb["convergence_gate"]["status"],
                    "n_nonconverged_selection_rows":
                        len(hyb["convergence_gate"]["non_converged_selection_rows"]),
                    "n_nonconverged_final_candidates":
                        hyb["convergence_gate"]["non_converged_candidate_count"],
                    "exploration_retry_count": hyb["integrity"]["exploration_retry_count"],
                    "refit_train_mask_sha": sha16(refit_mask),
                    "refit_mask_matches": bool(np.array_equal(refit_mask, train)),
                })
            else:
                res = _fixed_fit(fit, ds, train)
                families = list(FIXED_FAMILIES[fit["method"]])
                run.update({"selected_assignment": "|".join(families),
                            "refit_train_mask_sha": sha16(np.asarray(res["model"].train_mask, bool)),
                            "refit_mask_matches": bool(np.array_equal(
                                np.asarray(res["model"].train_mask, bool), train))})
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
            run.update({
                "status": "OK", "error": "", "w0_hat": w0, "w_hat": w,
                "abs_w0_err": abs(w0 - m1.W0_TRUE), "abs_w_err": abs(w - m1.W_TRUE),
                "retry_count": res["retry_count"], "replacement_count": res["replacement_count"],
                "seed_rescue_count": res["seed_rescue_count"],
                "nan_occurred": bool(res["nan_occurred"]), "em_runtime_s": res["runtime_s"],
                **{f"mode_{k}": v for k, v in mode.items() if k != "Z_hat"}})
            states = {"Zsamples": Zs, "F": np.asarray(res["F"], float),
                      "sigma": np.asarray(res["sigma"], float),
                      "scalars": np.array([w0, w, float(res["var_z"])]),
                      **({"Zmode": mode["Z_hat"]} if mode["Z_hat"] is not None else {})}
        except Exception as exc:            # recorded, never retried or replaced
            run.update({"status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}"[:500],
                        "failed_stage": stage_log[-1] if stage_log else "",
                        "traceback_tail": traceback.format_exc()[-800:]})
        run["stages"] = "|".join(stage_log)
        run["n_warnings"] = len(wl)
        run["warning_messages"] = " || ".join(sorted(
            {f"{x.category.__name__}: {str(x.message)[:160]}" for x in wl}))[:1000]
    run["wall_runtime_s"] = round(time.perf_counter() - t0, 2)
    return run, est_rows, col_rows, trace_rows, cand_rows, states


# ── I/O ───────────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    tmp.replace(path)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def data_audit(cache) -> list[dict[str, Any]]:
    rows = []
    for r, (ds, masks) in sorted(cache.items()):
        dense, sparse, test, pool = masks["dense"], masks["sparse"], masks["test"], masks["pool"]
        rows.append({
            "replicate": m1.replicate_label(r), "data_seed": m1.DATA_SEEDS[r],
            "x_sha": sha16(ds.X), "y_sha": sha16(ds.Y), "z_true_sha": sha16(ds.Z),
            "dense_sha": sha16(dense), "sparse_sha": sha16(sparse), "test_sha": sha16(test),
            "n_test": m1.n_upper(test), "n_dense": m1.n_upper(dense), "n_sparse": m1.n_upper(sparse),
            "dense_test_overlap": int(np.triu(dense & test, 1).sum()),
            "sparse_test_overlap": int(np.triu(sparse & test, 1).sum()),
            "sparse_subset_of_dense": bool(np.all(dense[sparse])),
            "dense_equals_pool": bool(np.array_equal(dense, pool)),
            "symmetric": bool(all(np.array_equal(m, m.T) for m in (dense, sparse, test))),
            "no_diagonal": bool(all(not np.any(np.diag(m)) for m in (dense, sparse, test))),
            "pool_plus_test_is_all_pairs": m1.n_upper(pool) + m1.n_upper(test) == m1.N * (m1.N - 1) // 2,
            "x_col_max": "|".join(str(int(v)) if float(v).is_integer() else f"{v:.3f}"
                                  for v in ds.X.max(axis=0)),
            "train_density_dense": float(ds.Y[np.triu(dense, 1)].mean()),
            "train_density_sparse": float(ds.Y[np.triu(sparse, 1)].mean()),
        })
    return rows


def runinfo(extra: dict[str, Any]) -> dict[str, Any]:
    import scipy
    import sklearn
    req = ROOT / "requirements.txt"
    return {
        "experiment": EXPERIMENT,
        "script": "expfam/src/experimental/run_auto_family_z_recovery.py",
        "git_head": m1.git("rev-parse", "HEAD"),
        "git_branch": m1.git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_tracked": bool(m1.git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "platform": platform.platform(),
        "requirements_sha256": sha256_file(req) if req.exists() else "",
        "protocol_sha256": protocol_sha256(),
        "command": " ".join(sys.argv),
        **extra,
    }


# ── main ──────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write-protocol", action="store_true")
    g.add_argument("--smoke", action="store_true",
                   help="series A, rep01, dense and sparse, all 4 methods (8 fits)")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted --execute; FAILED fits are never re-run")
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
    pre = m1.preflight()
    if not pre["pass"]:
        raise SystemExit(f"M1 preflight failed: {pre['failed_checks']}")

    stage = "smoke" if args.smoke else "full"
    out, state_dir = out_root / stage, state_root / stage
    if out.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite {out} (use --resume to continue --execute)")
    if args.resume and (args.smoke or not out.exists()):
        raise SystemExit("--resume only continues an existing --execute output")
    out.mkdir(parents=True, exist_ok=True)
    (state_dir / "fits").mkdir(parents=True, exist_ok=True)

    plan = planned_fits()
    if args.smoke:
        plan = [f for f in plan if f["series"] == "A" and f["replicate"] == "rep01"]
    progress = out / "progress.jsonl"
    events = load_jsonl(progress)
    finished = {e["fit_key"] for e in events if e["event"] == "FINISHED"}
    started = [e["fit_key"] for e in events if e["event"] == "STARTED"]
    interrupted = sorted({k for k in started if k not in finished})

    def load(name):
        p = out / name
        if not p.exists():
            return []
        with open(p, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    tables = {n: load(n) for n in ("fit_runs.csv", "fit_estimators.csv", "selection_columns.csv",
                                    "selection_trace.csv", "candidate_scores.csv")}
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
            cache[fit["r"]] = (m1.generate(fit["data_seed"]), m1.build_masks(fit["r"]))
        ds, masks = cache[fit["r"]]
        key = fit_key(fit)
        append_jsonl(progress, {"event": "STARTED", "fit_key": key,
                                "time_utc": datetime.now(timezone.utc).isoformat()})
        run, est, cols, trace, cand, st = run_fit(fit, ds, masks)
        if st:
            p = state_dir / "fits" / f"{key}.npz"
            np.savez_compressed(p, **st)
            run["state_file"] = p.name
            run["state_sha256"] = sha256_file(p)
        tables["fit_runs.csv"].append(run)
        tables["fit_estimators.csv"].extend(est)
        tables["selection_columns.csv"].extend(cols)
        tables["selection_trace.csv"].extend(trace)
        tables["candidate_scores.csv"].extend(cand)
        attempts.append({"fit_key": key, "event": run["status"],
                         "action": "first and only execution" if key not in interrupted
                         else "re-run after interruption",
                         "time_utc": datetime.now(timezone.utc).isoformat()})
        for name, rows in tables.items():
            if rows:
                write_csv(out / name, rows)
        write_csv(out / "attempts.csv", attempts)
        append_jsonl(progress, {"event": "FINISHED", "fit_key": key, "status": run["status"],
                                "time_utc": datetime.now(timezone.utc).isoformat()})
        mode = next((r for r in est if r["estimator"] == "mode"), {})
        print(f"[{i}/{len(todo)}] {key} {run['status']} "
              f"n_correct={run.get('n_correct_columns', '-')} mode={run.get('mode_status', '-')} "
              f"rmse_mode={mode.get('rmse_Z', float('nan')):.4f} "
              f"t={run['wall_runtime_s']}s", flush=True)

    for r in range(N_REPLICATES):
        if r not in cache and any(f["r"] == r for f in plan):
            cache[r] = (m1.generate(m1.DATA_SEEDS[r]), m1.build_masks(r))
    write_csv(out / "data_mask_audit.csv", data_audit(cache))
    (out / "preflight_m1.json").write_text(json.dumps(pre, indent=2), encoding="utf-8")
    prev = {}
    ri = out / "runinfo.json"
    if ri.exists():
        prev = json.loads(ri.read_text(encoding="utf-8"))
    sessions = prev.get("sessions", []) + [{
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_s": round(time.perf_counter() - t_all, 1), "fits_run": len(todo),
        "git_head": m1.git("rev-parse", "HEAD"), "command": " ".join(sys.argv)}]
    ri.write_text(json.dumps(runinfo({
        "stage": stage, "planned_fits": len(plan),
        "finished_fits": len(finished) + len(todo),
        "state_dir": str(state_dir / "fits"), "sessions": sessions,
        "seeds": {"series_model_seed_base": SERIES, "search_seed_base": SEARCH_SEED_BASE,
                  "bootstrap_seed": BOOTSTRAP_SEED},
    }), indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
