"""Experiment E2a (Issue #142): is the sparse-Y run-to-run variation of E1
driven by the initial values or by the Monte Carlo draws?

Research question
-----------------
In E1 (Issue #141) one model seed drove both ``initialize_params`` (initial Z, F)
and the E-step MC generator, so Part A (seed 1343000 + r) vs Part B
(1344000 + r) changed both at once.  With the opt-in ``init_seed`` / ``mc_seed``
of ``em_runner.run_em_experimental``, how do the sparse-Y paired delta RMSE_Z
(wrong - correct, joint mode at the final theta-hat) and the per-fit estimates
change when only one of the two seeds changes?

Frozen design (Issue #142 comment of 2026-11-04 + approval; fixed before any
E2a fit, do not change after results)
-----------------------------------------------------------------------------
Condition  E1 sparse-Y: n=75, d=12, K=3, true X = G3/B6/P3, arms correct
           G3/B6/P3 vs wrong G3/P6/P3, Bernoulli Y, E1/M1 data seeds, train and
           test masks, L=5, 8 EM iterations, numerics 'consistent',
           failure_policy 'fail_fast' (no retry / replacement / seed rescue).
Cells      seed A = 1343000 + r, seed B = 1344000 + r;
           (init, mc) in {AA, AB, BA, BB}.  AA = E1 Part A and BB = E1 Part B
           are REUSED (states + CSV); AB and BA are new: 20 x 2 x 2 = 80 fits.
           correct and wrong share data, masks, init seed and MC seed (their
           EM paths still differ because the likelihoods differ).
Smoke      rep01 sparse, correct + wrong, AA and BB, called with the NEW
           arguments (seed=s, init_seed=s, mc_seed=s): Z_samples, Z_est, F, w0,
           w and every estimator metric must equal the E1 artifact (old call
           ``seed=s``) within 1e-12 (SMOKE_RULE).  Otherwise the full run is
           refused; no seed / tolerance / parameter is changed to force a match.
Primary    joint mode (Z0 = last, grad_tol 1e-8, max_iter 200, as E1) paired
           delta RMSE_Z per cell, and per replicate the contrasts
             MC at fixed init   : D_AB - D_AA (init A), D_BB - D_BA (init B)
             init at fixed MC   : D_BA - D_AA (MC A),   D_BB - D_AB (MC B)
             interaction        : (D_AA + D_BB - D_AB - D_BA) / 2
           summarised by mean, median, IQR and median absolute value.
Secondary  the same contrasts for per-fit RMSE_Z; rotation-only and
           rotation+scale RMSE; Z norm ratio; Gram correlation; held-out log
           score and AUC; last / mean5 estimators; F, w0, w variation and
           solution proximity between cells (shared init vs shared MC);
           mode status and local solution (second mode from Z0 = mean5);
           stability of the correct/wrong delta across the four cells.
Reading    descriptive guide fixed in advance (INTERPRETATION_GUIDE); a
           2x2 with two levels and 20 replicates does not establish a general
           variance decomposition or a cause.

States are written OUTSIDE the repository with their sha256.  E1 artifacts are
read only.

Run
---
    python expfam/src/experimental/run_seed_separation_e2a.py --smoke
    python expfam/src/experimental/run_seed_separation_e2a.py --execute
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import run_z_point_estimator_e1 as e1          # sets sys.path              # noqa: E402
import run_misspecification_m1 as m1                                     # noqa: E402
from utils_expfam import calc_rmse, procrustes_rotation                  # noqa: E402

ISSUE = 142
REGIME = "sparse"
SEED_BASE = {"A": 1343000, "B": 1344000}            # = E1 Part A / Part B
CELLS = {"AA": ("A", "A"), "AB": ("A", "B"), "BA": ("B", "A"), "BB": ("B", "B")}
NEW_CELLS = ("AB", "BA")
REUSED = {"AA": "A", "BB": "B"}                       # cell -> E1 part
SMOKE_RULE = {"tolerance": 1e-12,
              "compared": ["Z_samples", "Z_est", "F", "w0", "w",
                           "rmse_Z", "gram_corr", "heldout_mean_log_score",
                           "heldout_auc", "z_est_norm_ratio"],
              "rule": ("MATCH if every compared quantity of the 4 smoke fits equals "
                       "the E1 artifact within 1e-12; otherwise MISMATCH and the "
                       "full run is refused")}
INTERPRETATION_GUIDE = {
    "quantities": ("mc = median over replicates of mean(|D_AB-D_AA|, |D_BB-D_BA|); "
                   "init = median of mean(|D_BA-D_AA|, |D_BB-D_AB|); mode estimator"),
    "small": "both mc and init < 0.01 (about 1/8 of the E1 sparse effect ~0.08)",
    "dominant": "one is >= 2x the other",
    "comparable": "otherwise",
    "note": "descriptive guide only; not a test and not a general decomposition",
}
E1_DIR = e1.OUT_DIR / "full"
E1_STATES = e1.STATE_DIR / "full" / "states.npz"
E1_STATES_SHA256 = "393a6d31af60a51ddf691716632650c1f72538f7da3dce15c7b544b860df9cd2"
OUT_DIR = e1.ROOT / "expfam" / "results" / "seed_separation_e2a" / "e2a_20261104"
STATE_DIR = Path(r"D:\tento\kennkyu_artifacts") / "seed_separation_e2a_20261104"
METRICS = ("rmse_Z", "rmse_Z_scaled", "gram_corr", "heldout_mean_log_score",
           "heldout_auc", "z_est_norm_ratio")


# ── plan ──────────────────────────────────────────────────────────────────

def seeds_of(cell: str, r: int) -> tuple[int, int]:
    i, m = CELLS[cell]
    return SEED_BASE[i] + r, SEED_BASE[m] + r


def planned_fits(cells=NEW_CELLS) -> list[dict[str, Any]]:
    plan = []
    for cell in cells:
        for fit in m1.planned_fits():
            if fit["regime"] != REGIME:
                continue
            init_seed, mc_seed = seeds_of(cell, fit["r"])
            plan.append({**{k: fit[k] for k in ("r", "replicate", "data_seed",
                                                "regime", "arm")},
                         "cell": cell, "init_seed": init_seed, "mc_seed": mc_seed})
    return plan


def fit_key(fit: dict[str, Any]) -> str:
    return f"{fit['cell']}_{fit['replicate']}_{fit['regime']}_{fit['arm']}"


def e1_key(cell: str, replicate: str, arm: str) -> str:
    return f"{REUSED[cell]}_{replicate}_{REGIME}_{arm}"


# ── scoring (shared by new and reused cells) ──────────────────────────────

def scaled_rmse(Z_est, Z_true) -> tuple[float, float]:
    """RMSE after rotation + one global scale (least squares)."""
    R, _ = procrustes_rotation(Z_est, Z_true)
    A = Z_est @ R
    c = float(np.sum(A * Z_true) / np.sum(A * A))
    return float(calc_rmse(Z_true, c * A)), c


def score_state(Zs, Zmode, w0, w, ds, test_mask) -> dict[str, dict[str, float]]:
    out = {}
    for name, Z in e1.point_estimates(Zs, Zmode).items():
        if not np.all(np.isfinite(Z)):
            continue
        sc = e1.score_estimate(Z, ds.Z, ds.Y, test_mask, w0, w)
        sc["rmse_Z_scaled"], sc["opt_scale"] = scaled_rmse(Z, ds.Z)
        out[name] = sc
    return out


# ── one new fit ───────────────────────────────────────────────────────────

def run_fit(fit, ds, masks):
    from em_runner import run_em_experimental                            # noqa: E402

    base = {k: fit[k] for k in ("cell", "replicate", "data_seed", "regime", "arm",
                                "init_seed", "mc_seed")}
    run = dict(base)
    states: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        try:
            # E1/M1 call; only init_seed / mc_seed added (seed kept = init seed)
            res = run_em_experimental(
                ds.X, ds.Y, family_x=None, family_y=m1.FAMILY_Y, k=m1.K_FIT, L=m1.L,
                num_iter=m1.NUM_ITER, seed=fit["init_seed"],
                train_mask=masks[fit["regime"]],
                family_x_list=list(m1.ARMS[fit["arm"]]), compute_strict_Q=False,
                validate_support=True, allow_support_mismatch=False,
                numerics_mode="consistent", failure_policy="fail_fast",
                init_seed=fit["init_seed"], mc_seed=fit["mc_seed"])
            Zs = np.asarray(res["Z_samples"], float)
            Z_est = np.asarray(res["Z_est"], float)
            w0, w = float(res["w0"]), float(res["w"])
            m_last = e1.joint_mode(res["model"], ds.X, ds.Y, Zs[:, :, -1])
            m_mean = e1.joint_mode(res["model"], ds.X, ds.Y, Zs.mean(axis=2))
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
                "modes_max_abs_diff": (float(np.max(np.abs(m_last["Z_hat"] - m_mean["Z_hat"])))
                                       if both else float("nan")),
            })
            states = {"Zsamples": Zs, "Zest": Z_est, "F": np.asarray(res["F"], float),
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
    return run, states


# ── E1 artifacts (read only) ──────────────────────────────────────────────

def load_e1() -> tuple[Any, dict, dict]:
    sha = e1.sha256_file(E1_STATES)
    if sha != E1_STATES_SHA256:
        raise SystemExit(f"E1 states sha256 mismatch: {sha}")
    with (E1_DIR / "fit_runs.csv").open(encoding="utf-8") as fh:
        runs = {(r["part"], r["replicate"], r["regime"], r["arm"]): r for r in csv.DictReader(fh)}
    with (E1_DIR / "fit_estimators.csv").open(encoding="utf-8") as fh:
        ests = {(r["part"], r["replicate"], r["regime"], r["arm"], r["estimator"]): r
                for r in csv.DictReader(fh)}
    return np.load(E1_STATES), runs, ests


def smoke_compare(new_runs, new_states, e1_states, e1_ests, cache) -> dict[str, Any]:
    rows = []
    for run in new_runs:
        cell, rep, arm = run["cell"], run["replicate"], run["arm"]
        k_new = f"{fit_key(run)}_"
        k_old = f"{e1_key(cell, rep, arm)}_"
        if run["status"] != "OK":
            rows.append({"fit": fit_key(run), "quantity": "status", "abs_diff": float("inf")})
            continue
        old_Zs = e1_states[k_old + "Zsamples"]
        old_sc = e1_states[k_old + "scalars"]
        pairs = {"Z_samples": (new_states[k_new + "Zsamples"], old_Zs),
                 "Z_est": (new_states[k_new + "Zest"], old_Zs[:, :, -1]),
                 "F": (new_states[k_new + "F"], e1_states[k_old + "F"]),
                 "w0": (run["w0_hat"], old_sc[0]), "w": (run["w_hat"], old_sc[1]),
                 "Z_mode": (new_states[k_new + "Zmode"], e1_states[k_old + "Zmode"])}
        for q, (a, b) in pairs.items():
            rows.append({"fit": fit_key(run), "quantity": q,
                         "abs_diff": float(np.max(np.abs(np.asarray(a) - np.asarray(b))))})
        ds, masks = cache[int(rep[3:]) - 1]
        sc = score_state(new_states[k_new + "Zsamples"], new_states[k_new + "Zmode"],
                         run["w0_hat"], run["w_hat"], ds, masks["test"])
        for est, vals in sc.items():
            ref = e1_ests[(REUSED[cell], rep, REGIME, arm, est)]
            for metric in ("rmse_Z", "gram_corr", "heldout_mean_log_score",
                           "heldout_auc", "z_est_norm_ratio"):
                rows.append({"fit": fit_key(run), "quantity": f"{est}:{metric}",
                             "abs_diff": abs(vals[metric] - float(ref[metric]))})
    mx = max(r["abs_diff"] for r in rows) if rows else float("inf")
    return {"verdict": "MATCH" if mx <= SMOKE_RULE["tolerance"] and len(new_runs) == 4 else "MISMATCH",
            "max_abs_diff": mx, "n_fits": len(new_runs), "n_compared": len(rows),
            "rule": SMOKE_RULE, "rows": rows}


# ── analysis ──────────────────────────────────────────────────────────────

def cell_records(new_runs, new_states, e1_states, e1_runs, cache):
    """One record per (replicate, cell, arm) with states-derived metrics."""
    by_new = {(r["cell"], r["replicate"], r["arm"]): r for r in new_runs}
    recs, reuse_check = [], 0.0
    for r in range(len(m1.DATA_SEEDS)):
        rep = m1.replicate_label(r)
        ds, masks = cache[r]
        for cell in CELLS:
            for arm in m1.ARMS:
                if cell in REUSED:
                    k = e1_key(cell, rep, arm) + "_"
                    run = e1_runs[(REUSED[cell], rep, REGIME, arm)]
                    src, st = f"E1_part{REUSED[cell]}", e1_states
                    ok = run["status"] == "OK"
                else:
                    run = by_new.get((cell, rep, arm), {"status": "MISSING"})
                    k = f"{cell}_{rep}_{REGIME}_{arm}_"
                    src, st = "E2a", new_states
                    ok = run["status"] == "OK"
                init_seed, mc_seed = seeds_of(cell, r)
                rec = {"replicate": rep, "cell": cell, "arm": arm, "source": src,
                       "init_seed": init_seed, "mc_seed": mc_seed,
                       "status": run["status"], "mode_status": run.get("mode_status", ""),
                       "modes_max_abs_diff": float(run.get("modes_max_abs_diff", "nan") or "nan")}
                if ok:
                    w0, w = (float(x) for x in st[k + "scalars"][:2])
                    Zmode = st[k + "Zmode"] if (k + "Zmode") in st else \
                        np.full_like(st[k + "Zsamples"][:, :, -1], np.nan)
                    sc = score_state(st[k + "Zsamples"], Zmode, w0, w, ds, masks["test"])
                    rec.update({"w0_hat": w0, "w_hat": w})
                    for est, vals in sc.items():
                        for metric, v in vals.items():
                            rec[f"{est}:{metric}"] = v
                    rec["_F"] = st[k + "F"]
                    rec["_Zmode"] = Zmode
                recs.append(rec)
    return recs


def _dist(values) -> dict[str, Any]:
    d = e1.dist(values)
    v = [x for x in values if x is not None and np.isfinite(x)]
    if v:
        d["median_abs"] = float(np.median(np.abs(v)))
        d["mean_abs"] = float(np.mean(np.abs(v)))
    return d


def contrasts(D: dict[str, float]) -> dict[str, float]:
    return {"mc_at_initA": D["AB"] - D["AA"], "mc_at_initB": D["BB"] - D["BA"],
            "init_at_mcA": D["BA"] - D["AA"], "init_at_mcB": D["BB"] - D["AB"],
            "interaction": (D["AA"] + D["BB"] - D["AB"] - D["BA"]) / 2.0,
            "both_changed_AA_BB": D["BB"] - D["AA"]}


def _complete(recs, rep, est):
    """Cell -> (correct, wrong) records when all 8 fits have a usable estimate."""
    sel = {(x["cell"], x["arm"]): x for x in recs if x["replicate"] == rep}
    ok = all(sel[(c, a)]["status"] == "OK" and f"{est}:rmse_Z" in sel[(c, a)]
             and (est != "mode" or sel[(c, a)]["mode_status"] == "OK")
             for c in CELLS for a in m1.ARMS)
    return sel if ok else None


def per_replicate(recs) -> list[dict[str, Any]]:
    out = []
    for r in range(len(m1.DATA_SEEDS)):
        rep = m1.replicate_label(r)
        for est in e1.ESTIMATORS:
            sel = _complete(recs, rep, est)
            row = {"replicate": rep, "estimator": est, "complete": sel is not None}
            if sel is None:
                out.append(row)
                continue
            for metric in METRICS:
                D = {c: sel[(c, "wrong")][f"{est}:{metric}"] - sel[(c, "correct")][f"{est}:{metric}"]
                     for c in CELLS}
                for c in CELLS:
                    row[f"delta_{metric}_{c}"] = D[c]
                for name, v in contrasts(D).items():
                    row[f"delta_{metric}__{name}"] = v
                for arm in m1.ARMS:
                    V = {c: sel[(c, arm)][f"{est}:{metric}"] for c in CELLS}
                    for c in CELLS:
                        row[f"{metric}_{arm}_{c}"] = V[c]
                    for name, v in contrasts(V).items():
                        row[f"{metric}_{arm}__{name}"] = v
            Dr = [row[f"delta_rmse_Z_{c}"] for c in CELLS]
            row["delta_rmse_Z_sd_over_cells"] = float(np.std(Dr, ddof=1))
            row["delta_rmse_Z_n_positive_cells"] = int(sum(x > 0 for x in Dr))
            out.append(row)
    return out


def proximity(recs) -> list[dict[str, Any]]:
    """Distance between the solutions of two cells (same replicate and arm)."""
    groups = {"shared_init": [("AA", "AB"), ("BA", "BB")],
              "shared_mc": [("AA", "BA"), ("AB", "BB")],
              "shared_none": [("AA", "BB"), ("AB", "BA")]}
    idx = {(x["replicate"], x["cell"], x["arm"]): x for x in recs}
    rows = []
    for r in range(len(m1.DATA_SEEDS)):
        rep = m1.replicate_label(r)
        for arm in m1.ARMS:
            for g, pairs in groups.items():
                for c1, c2 in pairs:
                    a, b = idx[(rep, c1, arm)], idx[(rep, c2, arm)]
                    if "_Zmode" not in a or "_Zmode" not in b or a["mode_status"] != "OK" \
                            or b["mode_status"] != "OK":
                        continue
                    Z1, Z2 = a["_Zmode"], b["_Zmode"]
                    iu = np.triu_indices(Z1.shape[0], 1)
                    R, _ = procrustes_rotation(Z2, Z1)      # Z1 ~ Z2 R  =>  F1 ~ F2 R
                    rows.append({
                        "replicate": rep, "arm": arm, "group": g, "pair": f"{c1}-{c2}",
                        "mode_gram_corr": float(np.corrcoef((Z1 @ Z1.T)[iu], (Z2 @ Z2.T)[iu])[0, 1]),
                        "mode_rmse_between": float(calc_rmse(Z1, Z2 @ R)),
                        "F_rel_diff": float(np.linalg.norm(a["_F"] - b["_F"] @ R) / np.linalg.norm(a["_F"])),
                        "abs_diff_w0": abs(a["w0_hat"] - b["w0_hat"]),
                        "abs_diff_w": abs(a["w_hat"] - b["w_hat"]),
                    })
    return rows


def summarize(recs, reps, prox, new_runs) -> dict[str, Any]:
    s: dict[str, Any] = {"issue": ISSUE, "planned_new_fits": len(planned_fits()),
                         "attempted_new_fits": len(new_runs),
                         "ok_new_fits": sum(r["status"] == "OK" for r in new_runs),
                         "integrity_new_fits": {
                             q: int(sum(int(r.get(q, 0) or 0) for r in new_runs))
                             for q in ("retry_count", "replacement_count", "seed_rescue_count")}}
    s["mode_status_counts"] = {}
    for cell in CELLS:
        for arm in m1.ARMS:
            sel = [x["mode_status"] or x["status"] for x in recs
                   if x["cell"] == cell and x["arm"] == arm]
            s["mode_status_counts"][f"{cell}_{arm}"] = {k: sel.count(k) for k in sorted(set(sel))}
    s["local_solution_modes_max_abs_diff"] = {
        cell: _dist([x["modes_max_abs_diff"] for x in recs if x["cell"] == cell])
        for cell in CELLS}
    s["local_solution_count_gt_1e-6"] = {
        cell: int(sum(x["modes_max_abs_diff"] > 1e-6 for x in recs if x["cell"] == cell))
        for cell in CELLS}
    names = ("mc_at_initA", "mc_at_initB", "init_at_mcA", "init_at_mcB",
             "interaction", "both_changed_AA_BB")
    s["contrasts"] = {}
    for est in e1.ESTIMATORS:
        rows = [p for p in reps if p["estimator"] == est and p["complete"]]
        blk: dict[str, Any] = {"complete_replicates": len(rows)}
        for metric in METRICS:
            m = {"delta_by_cell": {c: _dist([p[f"delta_{metric}_{c}"] for p in rows]) for c in CELLS},
                 "delta_contrasts": {n: _dist([p[f"delta_{metric}__{n}"] for p in rows]) for n in names}}
            for arm in m1.ARMS:
                m[f"{arm}_contrasts"] = {n: _dist([p[f"{metric}_{arm}__{n}"] for p in rows])
                                        for n in names}
            mc = [np.mean([abs(p[f"delta_{metric}__mc_at_initA"]), abs(p[f"delta_{metric}__mc_at_initB"])]) for p in rows]
            ini = [np.mean([abs(p[f"delta_{metric}__init_at_mcA"]), abs(p[f"delta_{metric}__init_at_mcB"])]) for p in rows]
            m["per_replicate_mean_abs_mc"] = _dist(mc)
            m["per_replicate_mean_abs_init"] = _dist(ini)
            m["n_replicates_mc_gt_init"] = int(sum(a > b for a, b in zip(mc, ini)))
            blk[metric] = m
        blk["delta_rmse_Z_sd_over_cells"] = _dist([p["delta_rmse_Z_sd_over_cells"] for p in rows])
        blk["delta_rmse_Z_n_positive_cells_hist"] = {
            str(k): int(sum(p["delta_rmse_Z_n_positive_cells"] == k for p in rows)) for k in range(5)}
        s["contrasts"][est] = blk
    pm = s["contrasts"]["mode"]["rmse_Z"]
    mc_med = pm["per_replicate_mean_abs_mc"].get("median")
    in_med = pm["per_replicate_mean_abs_init"].get("median")
    if mc_med is None or in_med is None:
        label = "NOT_AVAILABLE"
    elif mc_med < 0.01 and in_med < 0.01:
        label = "BOTH_SMALL"
    elif mc_med >= 2 * in_med:
        label = "MC_DOMINANT"
    elif in_med >= 2 * mc_med:
        label = "INIT_DOMINANT"
    else:
        label = "COMPARABLE"
    s["interpretation_guide"] = {**INTERPRETATION_GUIDE, "mc_median": mc_med,
                                 "init_median": in_med, "label": label}
    s["proximity"] = {}
    for g in ("shared_init", "shared_mc", "shared_none"):
        for arm in m1.ARMS:
            sel = [p for p in prox if p["group"] == g and p["arm"] == arm]
            s["proximity"][f"{g}_{arm}"] = {
                q: _dist([p[q] for p in sel]) for q in
                ("mode_gram_corr", "mode_rmse_between", "F_rel_diff", "abs_diff_w0", "abs_diff_w")}
    s["by_cell_arm"] = {}
    for cell in CELLS:
        for arm in m1.ARMS:
            sel = [x for x in recs if x["cell"] == cell and x["arm"] == arm]
            s["by_cell_arm"][f"{cell}_{arm}"] = {
                f"{est}:{metric}": _dist([x.get(f"{est}:{metric}") for x in sel])
                for est in e1.ESTIMATORS for metric in METRICS + ("opt_scale",)}
    return s


def protocol() -> dict[str, Any]:
    return {
        "issue": ISSUE, "mode": "CHARACTERIZATION (attribution of E1 sparse-Y variation)",
        "base": "E1 (Issue #141, PR #145) commit 568b4db; M1 frozen design",
        "m1_frozen_design": m1.protocol(), "regime": REGIME,
        "seed_base": SEED_BASE, "cells": {c: {"init": i, "mc": m} for c, (i, m) in CELLS.items()},
        "reused": {"AA": "E1 Part A", "BB": "E1 Part B",
                   "e1_states": str(E1_STATES), "e1_states_sha256": E1_STATES_SHA256},
        "new_cells": list(NEW_CELLS), "planned_new_fits": len(planned_fits()),
        "seed_separation": ("em_runner.run_em_experimental(init_seed, mc_seed): init_seed "
                            "-> initialize_params only, mc_seed -> E-step rng only; "
                            "defaults to seed (unchanged behaviour)"),
        "arms_share": "data, masks, init seed, MC seed (EM paths differ by likelihood)",
        "smoke_rule": SMOKE_RULE,
        "primary": "mode paired delta RMSE_Z per cell; MC / init contrasts and interaction",
        "estimators": "last / mean5 / mode as E1 (mode Z0 = last, grad_tol 1e-8, max_iter 200)",
        "mode_non_ok_policy": "never imputed; a replicate is complete only if all 8 fits are usable",
        "interpretation_guide": INTERPRETATION_GUIDE,
        "limitation": ("two levels per factor, 20 replicates: no general variance "
                       "decomposition or causal claim"),
        "state_dir": str(STATE_DIR), "no_new_gate_or_auditor": True,
    }


def _strip(recs):
    return [{k: v for k, v in x.items() if not k.startswith("_")} for x in recs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true",
                   help="rep01 sparse, AA and BB via the new arguments (4 fits) vs E1")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    args = ap.parse_args(argv)

    stage = "smoke" if args.smoke else "full"
    out, state_dir = Path(args.out_dir) / stage, Path(args.state_dir) / stage
    for p in (out, state_dir):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    if args.execute:
        sm = Path(args.out_dir) / "smoke" / "smoke_comparison.json"
        if not sm.exists() or json.loads(sm.read_text(encoding="utf-8"))["verdict"] != "MATCH":
            raise SystemExit("smoke MATCH is required before the full run")
    e1_states, e1_runs, e1_ests = load_e1()
    pre = m1.preflight()
    if not pre["pass"]:
        raise SystemExit(f"M1 preflight failed: {pre['failed_checks']}")
    out.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    (out / "protocol.json").write_text(json.dumps(protocol(), indent=2), encoding="utf-8")
    (out / "preflight.json").write_text(json.dumps(pre, indent=2), encoding="utf-8")

    if args.smoke:
        plan = [f for f in planned_fits(("AA", "BB")) if f["replicate"] == "rep01"]
    else:
        plan = planned_fits()
    started = datetime.now(timezone.utc).isoformat()
    t_all = time.perf_counter()
    cache = {r: (m1.generate(m1.DATA_SEEDS[r]), m1.build_masks(r))
             for r in range(len(m1.DATA_SEEDS))}
    runs, states = [], {}
    for i, fit in enumerate(plan, 1):
        ds, masks = cache[fit["r"]]
        run, st = run_fit(fit, ds, masks)
        runs.append(run)
        for k, v in st.items():
            states[f"{fit_key(fit)}_{k}"] = v
        print(f"[{i}/{len(plan)}] {fit_key(fit)} init={fit['init_seed']} mc={fit['mc_seed']} "
              f"{run['status']} mode={run.get('mode_status', '-')} "
              f"w={run.get('w_hat', float('nan')):.4f}", flush=True)

    np.savez_compressed(state_dir / "states.npz", **states)
    write = e1.write_csv
    write(out / "fit_runs.csv", runs)
    extra: dict[str, Any] = {}
    if args.smoke:
        cmp_ = smoke_compare(runs, states, e1_states, e1_ests, cache)
        write(out / "smoke_diffs.csv", cmp_.pop("rows"))
        (out / "smoke_comparison.json").write_text(json.dumps(cmp_, indent=2), encoding="utf-8")
        extra["smoke_verdict"] = cmp_["verdict"]
        print(f"smoke: {cmp_['verdict']} max_abs_diff={cmp_['max_abs_diff']}")
    else:
        recs = cell_records(runs, states, e1_states, e1_runs, cache)
        # reused cells: states-derived metrics must equal the E1 CSV
        diff = 0.0
        for x in recs:
            if x["cell"] in REUSED and x["status"] == "OK":
                for est in e1.ESTIMATORS:
                    ref = e1_ests.get((REUSED[x["cell"]], x["replicate"], REGIME, x["arm"], est))
                    if ref and f"{est}:rmse_Z" in x:
                        diff = max(diff, abs(x[f"{est}:rmse_Z"] - float(ref["rmse_Z"])))
        extra["reused_cells_max_abs_diff_rmse_vs_e1_csv"] = diff
        reps = per_replicate(recs)
        prox = proximity(recs)
        write(out / "cells_long.csv", _strip(recs))
        write(out / "per_replicate.csv", reps)
        write(out / "solution_proximity.csv", prox)
        summ = summarize(recs, reps, prox, runs)
        summ.update(extra)
        (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    import numpy, scipy, sklearn                                          # noqa: E401
    (out / "runinfo.json").write_text(json.dumps({
        "issue": ISSUE, "script": "expfam/src/experimental/run_seed_separation_e2a.py",
        "git_head": m1.git("rev-parse", "HEAD"),
        "git_branch": m1.git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_tracked": bool(m1.git("status", "--porcelain", "--untracked-files=no")),
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_s": round(time.perf_counter() - t_all, 1),
        "python": platform.python_version(), "numpy": numpy.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "platform": platform.platform(), "attempted_fits": len(runs), "planned_fits": len(plan),
        "states_file": str(state_dir / "states.npz"),
        "states_sha256": e1.sha256_file(state_dir / "states.npz"),
        **extra,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
