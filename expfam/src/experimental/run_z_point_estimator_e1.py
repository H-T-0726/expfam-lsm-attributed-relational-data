"""Experiment E1 (Issue #141): does M1's misspecification effect depend on the
Z point estimate?

Research question
-----------------
M1 (Issue #134) reported paired delta RMSE_Z = wrong - correct using the last
Monte Carlo sample of the final E-step as the Z estimate.  On the SAME fits
(same theta-hat), how do the dense / sparse deltas and their interaction change
when the Z estimate is

* ``last``  -- Z_samples[:, :, -1] (what M1 reported),
* ``mean5`` -- the mean of the L=5 samples of the final E-step,
* ``mode``  -- the joint mode of ln p(X, Y_train | Z, theta-hat) + ln p(Z),
               found by ``laplace_k_criterion.find_joint_mode`` from Z0 = last
               (the same solver and start rule as Issue #128)?

``mode`` is NOT presumed to be the correct estimate.  Keeping or overturning
M1's conclusion is not a success criterion.

Frozen design (Issue #141; fixed before any E1 fit, do not change after results)
------------------------------------------------------------------------------
Part A  replay of M1: the 80 fits of ``run_misspecification_m1`` with the
        same code, data, masks, model seeds (1343000 + r) and settings.
        Reproduction of M1's recorded last-sample metrics is checked against
        ``expfam/results/misspecification_m1/phase_m1_20261013/fit_results.csv``
        (rule: REPRODUCTION_RULE below).
Part B  the same 80 fits with model seed 1344000 + r (pre-specified in #141).
        The model seed drives BOTH the initial values and the Monte Carlo
        draws, so Part A vs Part B measures run-to-run variation from
        initialisation + MC together, not MC error alone.
Mode    status OK / NOT_STATIONARY / HESSIAN_NOT_PD recorded with max|grad|,
        min eigenvalue of H, iterations, Phi.  A second mode from Z0 = mean5 is
        a pre-registered local-solution diagnostic only.  Non-OK modes are never
        imputed; analyses report complete pairs and completion counts separately.
Primary   : paired delta RMSE_Z (wrong - correct) with ``mode``, per regime,
            and interaction = delta_sparse - delta_dense; Part A, complete pairs.
Secondary : the same with ``last`` and ``mean5``; Gram correlation; held-out
            Bernoulli mean log score and AUC per estimator; mode diagnostics;
            Part A vs Part B variation of the deltas.

States (Z_samples, F, sigma, w0, w, var_z, Z_mode) are written OUTSIDE the
repository; their sha256 is recorded.  Existing M1 artifacts are read only.

Run
---
    python expfam/src/experimental/run_z_point_estimator_e1.py --smoke
    python expfam/src/experimental/run_z_point_estimator_e1.py --execute
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
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

import run_misspecification_m1 as m1                                     # noqa: E402
from eval_utils import upper_pairs_of                                    # noqa: E402
from utils_expfam import calc_rmse, procrustes_rotation                  # noqa: E402

ISSUE = 141
PARTS = {"A": m1.MODEL_SEED_BASE, "B": 1344000}        # model seed = base + r
ESTIMATORS = ("last", "mean5", "mode")
MODE_GRAD_TOL, MODE_MAX_ITER = 1e-8, 200               # #128 Laplace settings
REPRODUCTION_RULE = {
    "metrics": ["rmse_Z", "gram_corr", "heldout_mean_log_score", "heldout_auc",
                "w0_hat", "w_hat"],
    "exact_tolerance": 1e-12,
    "fp_tolerance": 1e-6,
    "rule": ("REPRODUCED if every metric of every Part A fit is within 1e-12 of "
             "M1; REPRODUCED_WITH_FP_DIFFERENCE if within 1e-6 and no paired "
             "delta changes sign (investigate and record, continue); otherwise "
             "NOT_REPRODUCED: Part B and the comparison are not run"),
}
M1_FIT_RESULTS = (ROOT / "expfam" / "results" / "misspecification_m1"
                  / "phase_m1_20261013" / "fit_results.csv")
OUT_DIR = ROOT / "expfam" / "results" / "z_point_estimator_e1" / "e1_20261104"
STATE_DIR = Path(r"D:\tento\kennkyu_artifacts") / "z_point_estimator_e1_20261104"


# ── plan ──────────────────────────────────────────────────────────────────

def planned_fits(parts=("A", "B")) -> list[dict[str, Any]]:
    plan = []
    for part in parts:
        for fit in m1.planned_fits():
            plan.append({**fit, "part": part,
                         "model_seed": PARTS[part] + fit["r"]})
    return plan


def fit_key(fit: dict[str, Any]) -> str:
    return f"{fit['part']}_{fit['replicate']}_{fit['regime']}_{fit['arm']}"


# ── estimators and scores ─────────────────────────────────────────────────

def point_estimates(Z_samples: np.ndarray, Z_mode: np.ndarray) -> dict[str, np.ndarray]:
    return {"last": np.asarray(Z_samples[:, :, -1], float),
            "mean5": np.asarray(Z_samples, float).mean(axis=2),
            "mode": np.asarray(Z_mode, float)}


def score_estimate(Z_est, Z_true, Y, test_mask, w0, w) -> dict[str, float]:
    """M1's scoring (``m1.score_fit``) applied to a given Z estimate."""
    from run_heldout_k_selection_pilot import heldout_bernoulli_mean_log_score  # noqa: E402
    from sklearn.metrics import roc_auc_score

    R, k_min = procrustes_rotation(Z_est, Z_true)
    if k_min != m1.K_TRUE:
        raise RuntimeError(f"k_min={k_min} != {m1.K_TRUE}")
    iu = np.triu_indices(Z_true.shape[0], 1)
    rows, cols = upper_pairs_of(test_mask)
    eta = w0 + w * np.sum(Z_est[rows] * Z_est[cols], axis=1)
    y = Y[rows, cols]
    return {
        "rmse_Z": float(calc_rmse(Z_true[:, :k_min], Z_est[:, :k_min] @ R)),
        "gram_corr": float(np.corrcoef((Z_est @ Z_est.T)[iu],
                                       (Z_true @ Z_true.T)[iu])[0, 1]),
        "heldout_mean_log_score": heldout_bernoulli_mean_log_score(y, eta),
        "heldout_auc": float(roc_auc_score(y, 1.0 / (1.0 + np.exp(-eta)))),
        "z_est_norm_ratio": float(np.linalg.norm(Z_est) / np.linalg.norm(Z_true)),
    }


def joint_mode(model, X, Y, Z0) -> dict[str, Any]:
    """Joint mode at fixed theta-hat with the fit's training mask (no imputation)."""
    import laplace_k_criterion as lk                                     # noqa: E402

    t0 = time.perf_counter()
    try:
        res = lk.laplace_log_observed(model, X, Y, Z0, grad_tol=MODE_GRAD_TOL,
                                      max_iter=MODE_MAX_ITER)
    except Exception as exc:             # recorded as a mode outcome, EM fit kept
        return {"Z_hat": None, "status": f"ERROR: {type(exc).__name__}: {exc}"[:200],
                "grad_inf": float("nan"), "min_eig_H": float("nan"),
                "iterations": -1, "phi": float("nan"),
                "seconds": round(time.perf_counter() - t0, 3)}
    return {"Z_hat": np.asarray(res.Z_hat, float), "status": res.status,
            "grad_inf": float(res.grad_inf), "min_eig_H": float(res.min_eigenvalue_H),
            "iterations": int(res.iterations), "phi": float(res.phi_at_mode),
            "seconds": round(time.perf_counter() - t0, 3)}


# ── one fit ───────────────────────────────────────────────────────────────

def run_fit(fit, ds, masks) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    from em_runner import run_em_experimental                            # noqa: E402

    base = {k: fit[k] for k in ("part", "replicate", "data_seed", "regime", "arm",
                                "model_seed")}
    run = dict(base, n_train_pairs=m1.n_upper(masks[fit["regime"]]),
               n_test_pairs=m1.n_upper(masks["test"]))
    est_rows: list[dict[str, Any]] = []
    states: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        try:
            # identical call to run_misspecification_m1.run_fit
            res = run_em_experimental(
                ds.X, ds.Y, family_x=None, family_y=m1.FAMILY_Y, k=m1.K_FIT, L=m1.L,
                num_iter=m1.NUM_ITER, seed=fit["model_seed"],
                train_mask=masks[fit["regime"]],
                family_x_list=list(m1.ARMS[fit["arm"]]), compute_strict_Q=False,
                validate_support=True, allow_support_mismatch=False,
                numerics_mode="consistent", failure_policy="fail_fast")
            Zs = np.asarray(res["Z_samples"], float)
            if not np.array_equal(Zs[:, :, -1], np.asarray(res["Z_est"], float)):
                raise RuntimeError("Z_est is not the last sample")
            w0, w = float(res["w0"]), float(res["w"])
            m_last = joint_mode(res["model"], ds.X, ds.Y, Zs[:, :, -1])
            m_mean = joint_mode(res["model"], ds.X, ds.Y, Zs.mean(axis=2))
            est = point_estimates(Zs, m_last["Z_hat"] if m_last["Z_hat"] is not None
                                  else np.full_like(Zs[:, :, -1], np.nan))
            for name, Z in est.items():
                if not np.all(np.isfinite(Z)):
                    continue
                sc = score_estimate(Z, ds.Z, ds.Y, masks["test"], w0, w)
                est_rows.append(dict(base, estimator=name, **sc,
                                     mode_status=m_last["status"] if name == "mode" else ""))
            alt = (score_estimate(m_mean["Z_hat"], ds.Z, ds.Y, masks["test"], w0, w)
                   if m_mean["Z_hat"] is not None else {"rmse_Z": float("nan")})
            both = m_last["Z_hat"] is not None and m_mean["Z_hat"] is not None
            run.update({
                "status": "OK", "error": "", "w0_hat": w0, "w_hat": w,
                "retry_count": res["retry_count"],
                "replacement_count": res["replacement_count"],
                "seed_rescue_count": res["seed_rescue_count"],
                "nan_occurred": bool(res["nan_occurred"]),
                "em_runtime_s": res["runtime_s"],
                **{f"mode_{k}": v for k, v in m_last.items() if k != "Z_hat"},
                **{f"mode_from_mean5_{k}": v for k, v in m_mean.items() if k != "Z_hat"},
                "mode_from_mean5_rmse_Z": alt["rmse_Z"],
                "modes_max_abs_diff": (float(np.max(np.abs(m_last["Z_hat"] - m_mean["Z_hat"])))
                                       if both else float("nan")),
                "modes_phi_diff": m_last["phi"] - m_mean["phi"],
            })
            states = {"Zsamples": Zs, "F": np.asarray(res["F"], float),
                      "sigma": np.asarray(res["sigma"], float),
                      "scalars": np.array([w0, w, float(res["var_z"])]),
                      **({"Zmode": m_last["Z_hat"]} if m_last["Z_hat"] is not None else {}),
                      **({"Zmode_from_mean5": m_mean["Z_hat"]} if m_mean["Z_hat"] is not None else {})}
        except Exception as exc:            # recorded, never retried or replaced
            run.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"[:500]})
        run["n_warnings"] = len(wl)
        run["warning_messages"] = " || ".join(sorted(
            {f"{x.category.__name__}: {str(x.message)[:160]}" for x in wl}))[:1000]
    run["wall_runtime_s"] = round(time.perf_counter() - t0, 2)
    return run, est_rows, states


# ── reproduction check (Part A vs recorded M1) ────────────────────────────

def reproduction(runs, est_rows, only_present: bool = False) -> dict[str, Any]:
    """Compare Part A last-sample metrics with M1.

    ``only_present`` restricts the comparison to the fits that were run (smoke);
    the full run compares all 80 and treats a missing fit as NOT_REPRODUCED.
    """
    with M1_FIT_RESULTS.open(encoding="utf-8") as fh:
        m1rows = {(r["replicate"], r["regime"], r["arm"]): r for r in csv.DictReader(fh)}
    if only_present:
        ran = {(r["replicate"], r["regime"], r["arm"]) for r in runs if r["part"] == "A"}
        m1rows = {k: v for k, v in m1rows.items() if k in ran}
    last = {(e["replicate"], e["regime"], e["arm"]): e for e in est_rows
            if e["part"] == "A" and e["estimator"] == "last"}
    runs_a = {(r["replicate"], r["regime"], r["arm"]): r for r in runs if r["part"] == "A"}
    diffs, missing = [], []
    for key, ref in m1rows.items():
        if key not in last:
            missing.append(key)
            continue
        got = {**last[key], "w0_hat": runs_a[key]["w0_hat"], "w_hat": runs_a[key]["w_hat"]}
        for metric in REPRODUCTION_RULE["metrics"]:
            diffs.append({"replicate": key[0], "regime": key[1], "arm": key[2],
                          "metric": metric, "m1": float(ref[metric]),
                          "e1": float(got[metric]),
                          "abs_diff": abs(float(ref[metric]) - float(got[metric]))})
    max_diff = max((d["abs_diff"] for d in diffs), default=math.inf)
    sign_flip = False
    for rep in {k[0] for k in m1rows}:
        for regime in m1.REGIMES:
            if only_present and not ((rep, regime, "wrong") in m1rows
                                     and (rep, regime, "correct") in m1rows):
                continue
            try:
                old = float(m1rows[(rep, regime, "wrong")]["rmse_Z"]) - float(m1rows[(rep, regime, "correct")]["rmse_Z"])
                new = last[(rep, regime, "wrong")]["rmse_Z"] - last[(rep, regime, "correct")]["rmse_Z"]
                sign_flip |= (old > 0) != (new > 0)
            except KeyError:
                sign_flip = True
    if missing:
        verdict = "NOT_REPRODUCED"
    elif max_diff <= REPRODUCTION_RULE["exact_tolerance"]:
        verdict = "REPRODUCED"
    elif max_diff <= REPRODUCTION_RULE["fp_tolerance"] and not sign_flip:
        verdict = "REPRODUCED_WITH_FP_DIFFERENCE"
    else:
        verdict = "NOT_REPRODUCED"
    return {"verdict": verdict, "max_abs_diff": max_diff, "n_compared": len(diffs),
            "missing": [list(k) for k in missing], "paired_sign_flip": sign_flip,
            "rule": REPRODUCTION_RULE, "diffs": diffs}


# ── analysis ──────────────────────────────────────────────────────────────

def dist(values) -> dict[str, Any]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75)),
            "min": float(v.min()), "max": float(v.max()),
            "n_positive": int((v > 0).sum()), "n_negative": int((v < 0).sum())}


def paired(est_rows, runs) -> list[dict[str, Any]]:
    status = {(r["part"], r["replicate"], r["regime"], r["arm"]): r.get("mode_status", "")
              for r in runs}
    idx = {(e["part"], e["replicate"], e["regime"], e["arm"], e["estimator"]): e
           for e in est_rows}
    out = []
    for part in PARTS:
        for r in range(len(m1.DATA_SEEDS)):
            rep = m1.replicate_label(r)
            for est in ESTIMATORS:
                row = {"part": part, "replicate": rep, "estimator": est}
                for regime in m1.REGIMES:
                    c = idx.get((part, rep, regime, "correct", est))
                    w = idx.get((part, rep, regime, "wrong", est))
                    ok = c is not None and w is not None
                    if est == "mode":
                        ok = ok and status[(part, rep, regime, "correct")] == "OK" \
                            and status[(part, rep, regime, "wrong")] == "OK"
                    for metric in ("rmse_Z", "gram_corr", "heldout_mean_log_score",
                                   "heldout_auc"):
                        row[f"delta_{metric}_{regime}"] = (
                            w[metric] - c[metric]) if ok else None
                    row[f"complete_{regime}"] = ok
                    for arm, e in (("correct", c), ("wrong", w)):
                        row[f"rmse_Z_{arm}_{regime}"] = e["rmse_Z"] if e else None
                d, s = row["delta_rmse_Z_dense"], row["delta_rmse_Z_sparse"]
                row["interaction_rmse_Z"] = (s - d) if d is not None and s is not None else None
                out.append(row)
    return out


def summarize(runs, est_rows, pairs) -> dict[str, Any]:
    s: dict[str, Any] = {"issue": ISSUE, "planned_fits": len(planned_fits()),
                         "attempted_fits": len(runs),
                         "ok_fits": sum(r["status"] == "OK" for r in runs),
                         "failed_fits": sum(r["status"] != "OK" for r in runs)}
    s["mode_status_counts"] = {}
    for part in PARTS:
        for regime in m1.REGIMES:
            for arm in m1.ARMS:
                sel = [r.get("mode_status", "FAILED") for r in runs
                       if r["part"] == part and r["regime"] == regime and r["arm"] == arm]
                s["mode_status_counts"][f"{part}_{regime}_{arm}"] = {
                    st: sel.count(st) for st in sorted(set(sel))}
    s["deltas"] = {}
    for part in PARTS:
        for est in ESTIMATORS:
            rows = [p for p in pairs if p["part"] == part and p["estimator"] == est]
            block = {}
            for regime in m1.REGIMES:
                block[regime] = {
                    "complete_pairs": sum(bool(p[f"complete_{regime}"]) for p in rows),
                    **{metric: dist([p[f"delta_{metric}_{regime}"] for p in rows])
                       for metric in ("rmse_Z", "gram_corr", "heldout_mean_log_score",
                                      "heldout_auc")}}
            block["interaction_rmse_Z"] = dist([p["interaction_rmse_Z"] for p in rows])
            s["deltas"][f"{part}_{est}"] = block
    # Part A vs Part B (initialisation + MC) variation of the paired deltas
    s["seed_variation"] = {}
    for est in ESTIMATORS:
        a = {p["replicate"]: p for p in pairs if p["part"] == "A" and p["estimator"] == est}
        b = {p["replicate"]: p for p in pairs if p["part"] == "B" and p["estimator"] == est}
        block = {}
        for q in ("delta_rmse_Z_dense", "delta_rmse_Z_sparse", "interaction_rmse_Z"):
            da = [a[r][q] for r in a if a[r][q] is not None and b[r][q] is not None]
            db = [b[r][q] for r in a if a[r][q] is not None and b[r][q] is not None]
            diff = [abs(x - y) for x, y in zip(da, db)]
            block[q] = {"pairs": len(da), "abs_diff_A_B": dist(diff),
                        "corr_A_B": (float(np.corrcoef(da, db)[0, 1]) if len(da) > 2 else None),
                        "sign_agree": int(sum((x > 0) == (y > 0) for x, y in zip(da, db))),
                        "mean_A": float(np.mean(da)) if da else None,
                        "mean_B": float(np.mean(db)) if db else None}
        s["seed_variation"][est] = block
    # point-estimate inflation within arm x regime (Part A)
    idx = {(e["part"], e["replicate"], e["regime"], e["arm"], e["estimator"]): e
           for e in est_rows}
    st = {(r["part"], r["replicate"], r["regime"], r["arm"]): r.get("mode_status", "")
          for r in runs}
    s["inflation_last_minus_mode_A"] = {}
    for regime in m1.REGIMES:
        for arm in m1.ARMS:
            v = []
            for r in range(len(m1.DATA_SEEDS)):
                k = ("A", m1.replicate_label(r), regime, arm)
                if st.get(k) == "OK":
                    v.append(idx[k + ("last",)]["rmse_Z"] - idx[k + ("mode",)]["rmse_Z"])
            s["inflation_last_minus_mode_A"][f"{regime}_{arm}"] = dist(v)
    s["mode_local_solution"] = dist([r.get("modes_max_abs_diff") for r in runs
                                     if r["status"] == "OK"])
    return s


# ── io ────────────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protocol() -> dict[str, Any]:
    return {
        "issue": ISSUE, "mode": "CHARACTERIZATION (re-evaluation of M1)",
        "base_experiment": "M1 (Issue #134), run_misspecification_m1.py at 31ebcac",
        "m1_frozen_design": m1.protocol(),
        "parts": {"A": "replay, model seed 1343000 + r",
                  "B": "model seed 1344000 + r (pre-specified in #141)"},
        "part_b_limitation": ("model seed drives initial values AND Monte Carlo "
                              "draws; A vs B measures their joint run-to-run "
                              "variation, not MC error alone"),
        "estimators": {"last": "Z_samples[:, :, -1]",
                       "mean5": "mean of the L=5 final E-step samples",
                       "mode": ("joint mode of ln p(X, Y_train | Z, theta_hat) + "
                                "ln p(Z) by laplace_k_criterion, Z0 = last, "
                                f"grad_tol {MODE_GRAD_TOL}, max_iter {MODE_MAX_ITER}")},
        "mode_local_solution_diagnostic": "second mode from Z0 = mean5 (not used in the primary)",
        "mode_non_ok_policy": ("NOT_STATIONARY / HESSIAN_NOT_PD recorded, never "
                               "imputed; primary uses complete pairs (both arms OK "
                               "in that regime); completion counts reported separately"),
        "primary": "Part A, mode, paired delta RMSE_Z (wrong - correct) per regime and interaction",
        "secondary": ["last and mean5 deltas", "Gram correlation", "held-out mean log score",
                      "held-out AUC", "mode diagnostics", "Part A vs B variation"],
        "reproduction_rule": REPRODUCTION_RULE,
        "planned_fits": len(planned_fits()),
        "state_dir": str(STATE_DIR),
        "no_new_gate_or_auditor": True,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true",
                   help="Part A rep01 dense (2 fits) into <out>/smoke; checks reproduction")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    args = ap.parse_args(argv)

    out = Path(args.out_dir) / ("smoke" if args.smoke else "full")
    state_dir = Path(args.state_dir) / ("smoke" if args.smoke else "full")
    for p in (out, state_dir):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    pre = m1.preflight()
    if not pre["pass"]:
        raise SystemExit(f"M1 preflight failed: {pre['failed_checks']}")
    out.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    (out / "protocol.json").write_text(json.dumps(protocol(), indent=2), encoding="utf-8")
    (out / "preflight.json").write_text(json.dumps(pre, indent=2), encoding="utf-8")

    started = datetime.now(timezone.utc).isoformat()
    t_all = time.perf_counter()
    plan = planned_fits(("A",)) if args.smoke else planned_fits()
    if args.smoke:
        plan = [f for f in plan if f["replicate"] == "rep01" and f["regime"] == "dense"]
    runs, est_rows, states = [], [], {}
    cache: dict[int, Any] = {}
    stage_verdict = None
    for i, fit in enumerate(plan, 1):
        if fit["part"] == "B" and stage_verdict is None:
            rep = reproduction(runs, est_rows)
            stage_verdict = rep["verdict"]
            write_csv(out / "reproduction_diffs.csv", rep.pop("diffs"))
            (out / "reproduction.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
            if stage_verdict == "NOT_REPRODUCED":
                print("Part A did not reproduce M1; Part B and the comparison are not run")
                break
        if fit["r"] not in cache:
            cache[fit["r"]] = (m1.generate(fit["data_seed"]), m1.build_masks(fit["r"]))
        ds, masks = cache[fit["r"]]
        run, rows, st = run_fit(fit, ds, masks)
        runs.append(run)
        est_rows.extend(rows)
        for k, v in st.items():
            states[f"{fit_key(fit)}_{k}"] = v
        print(f"[{i}/{len(plan)}] {fit_key(fit)} {run['status']} "
              f"mode={run.get('mode_status', '-')} "
              f"rmse(last/mean5/mode)="
              + "/".join(f"{r['rmse_Z']:.4f}" for r in rows), flush=True)

    if args.smoke:
        rep = reproduction(runs, est_rows, only_present=True)
        rep["note"] = "smoke compares only the fits it ran (rep01 dense)"
        write_csv(out / "reproduction_diffs.csv", rep.pop("diffs"))
        (out / "reproduction.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    np.savez_compressed(state_dir / "states.npz", **states)
    write_csv(out / "fit_runs.csv", runs)
    write_csv(out / "fit_estimators.csv", est_rows)
    pairs = paired(est_rows, runs) if not args.smoke else []
    if pairs:
        write_csv(out / "paired_deltas.csv", pairs)
        (out / "summary.json").write_text(
            json.dumps(summarize(runs, est_rows, pairs), indent=2), encoding="utf-8")
    import numpy, scipy, sklearn                                          # noqa: E401
    (out / "runinfo.json").write_text(json.dumps({
        "issue": ISSUE, "script": "expfam/src/experimental/run_z_point_estimator_e1.py",
        "git_head": m1.git("rev-parse", "HEAD"),
        "git_branch": m1.git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_tracked": bool(m1.git("status", "--porcelain", "--untracked-files=no")),
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_s": round(time.perf_counter() - t_all, 1),
        "python": platform.python_version(), "numpy": numpy.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "platform": platform.platform(),
        "attempted_fits": len(runs), "planned_fits": len(plan),
        "reproduction_verdict": stage_verdict,
        "states_file": str(state_dir / "states.npz"),
        "states_sha256": sha256_file(state_dir / "states.npz"),
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
