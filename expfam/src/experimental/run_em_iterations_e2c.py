"""Experiment E2c (Issue #142): do the solutions from different initial values
approach each other when the EM budget is extended from 8 to 40 iterations?

Research question
-----------------
E2a found that, in the E1 sparse-Y condition, the solution location (joint-mode
Z, F) is set mainly by the initial values: with the same MC seed, init A and
init B give clearly different solutions after 8 iterations.  Is that because 8
iterations do not finish the estimation, or do separate solutions remain at 40?

Frozen design (fixed before any E2c fit; do not change after results)
---------------------------------------------------------------------
Condition    E2a sparse-Y (= E1 sparse): n=75, d=12, K=3, G3/B6/P3 vs G3/P6/P3,
             Bernoulli Y, E1 data seeds and masks, L=5, consistent, fail_fast.
Cells        AA (init 1343000+r, MC 1343000+r) and BA (init 1344000+r,
             MC 1343000+r): 20 replicates x 2 inits x 2 arms = 80 new fits,
             num_iter = 40, no new data or seeds.
Checkpoints  8, 16, 24, 32, 40 via the opt-in ``checkpoint_iters`` of
             ``run_em_experimental`` (copies only, no RNG use).  theta after
             the M-step of every iteration is recorded.  At each checkpoint the
             joint mode is computed at that theta (Z0 = last sample of that
             iteration; second mode from Z0 = mean5 as E1).
Reproduction The iteration-8 checkpoint must equal E2a's AA (E1 Part A states)
             and BA (E2a states): Z_samples, F, w0, w and the joint mode within
             1e-12 (REPRO_RULE).  Smoke = rep01, 4 fits; the full run is refused
             without a smoke MATCH; the full run re-checks all 80 fits.  No seed,
             condition or tolerance is changed to obtain a match.
Primary      per replicate and arm, the closeness of the init-A and init-B
             solutions from iteration 8 to 40: Gram correlation of the joint
             modes and their distance after rotation + scale alignment.
Secondary    F / w0 / w differences; RMSE_Z (rotation, rotation+scale) per init;
             paired delta RMSE_Z (wrong - correct) per init; held-out log score
             and AUC; mode status / local solution; per-iteration theta change;
             within-fit distance between consecutive checkpoints (a reference
             for how much one run still moves); Q(theta_t | Z_samples_t) and the
             log joint at the mode (Phi) per checkpoint.
Reading      READING_GUIDE (descriptive, fixed in advance; no single threshold
             decides convergence).  L=5 MC noise remains; agreement is not a
             global optimum and disagreement is not proof of multimodality.

States go OUTSIDE the repository with sha256.  E1 / E2a artifacts are read only.

Run
---
    python expfam/src/experimental/run_em_iterations_e2c.py --smoke
    python expfam/src/experimental/run_em_iterations_e2c.py --execute
"""

from __future__ import annotations

import argparse
import copy
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
import run_seed_separation_e2a as e2a                                    # noqa: E402
from utils_expfam import calc_rmse, procrustes_rotation                  # noqa: E402

ISSUE = 142
REGIME = "sparse"
NUM_ITER = 40
CHECKPOINTS = (8, 16, 24, 32, 40)
CELLS = ("AA", "BA")                         # init A / init B, MC A for both
REPRO_RULE = {"checkpoint": 8, "tolerance": 1e-12,
              "compared": ["Z_samples", "F", "w0", "w", "Z_mode"],
              "reference": {"AA": "E1 Part A states (= E2a AA)", "BA": "E2a BA states"},
              "rule": "MATCH if every compared quantity is within 1e-12, else MISMATCH"}
READING_GUIDE = {
    "approach": ("between-init aligned distance of the modes decreases from 8 to 40 "
                 "in >= 14/20 replicates, in both arms"),
    "settled_ratio": ("rho = median between-init distance at 40 / median within-fit "
                      "distance between checkpoints 32 and 40; rho >= 2 reads as "
                      "'separate solutions, each run moving much less than the gap' (B), "
                      "rho < 2 as 'still moving about as much as the gap' (C)"),
    "accuracy": ("RMSE_Z (mode) lower at 40 than at 8 in >= 14/20 replicates per arm "
                 "reads as improved accuracy (D if solutions do not approach)"),
    "e2a_reference": ("E2a shared-init (MC-only) mode Gram correlation at 8 iterations: "
                      "0.967 correct / 0.954 wrong (descriptive anchor, not a threshold)"),
    "note": "descriptive guide only; not a test",
}
E2A_STATES = e2a.STATE_DIR / "full" / "states.npz"
E2A_STATES_SHA256 = "84749cdb11a1e395b13c7af74e59b66c0a7f631211494127dde82ae71b05cd0d"
OUT_DIR = e1.ROOT / "expfam" / "results" / "em_iterations_e2c" / "e2c_20261104"
STATE_DIR = Path(r"D:\tento\kennkyu_artifacts") / "em_iterations_e2c_20261104"


def planned_fits() -> list[dict[str, Any]]:
    return [f for f in e2a.planned_fits(CELLS)]


def fit_key(fit) -> str:
    return e2a.fit_key(fit)


def ref_key(cell: str, replicate: str, arm: str) -> tuple[str, str]:
    """(which reference file, key prefix) of the 8-iteration E2a state."""
    if cell == "AA":
        return "e1", f"A_{replicate}_{REGIME}_{arm}_"
    return "e2a", f"BA_{replicate}_{REGIME}_{arm}_"


# ── alignment helpers ─────────────────────────────────────────────────────

def aligned_distance(Z1, Z2) -> tuple[float, np.ndarray]:
    """RMSE between Z1 and c * Z2 R (rotation + one scale); returns R too."""
    R, _ = procrustes_rotation(Z2, Z1)
    A = Z2 @ R
    c = float(np.sum(A * Z1) / np.sum(A * A))
    return float(calc_rmse(Z1, c * A)), R


def gram_corr(Z1, Z2) -> float:
    iu = np.triu_indices(Z1.shape[0], 1)
    return float(np.corrcoef((Z1 @ Z1.T)[iu], (Z2 @ Z2.T)[iu])[0, 1])


# ── one fit ───────────────────────────────────────────────────────────────

def checkpoint_eval(model, ck, ds, masks) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from eval_utils import calc_Q_dual_strict_exp                        # noqa: E402

    m = copy.deepcopy(model)
    m.params.update({"F": ck["F"], "sigma": ck["sigma"], "w0": ck["w0"],
                     "w": ck["w"], "var_z": ck["var_z"]})
    Zs = ck["Z_samples"]
    m_last = e1.joint_mode(m, ds.X, ds.Y, Zs[:, :, -1])
    m_mean = e1.joint_mode(m, ds.X, ds.Y, Zs.mean(axis=2))
    try:
        q = float(calc_Q_dual_strict_exp(ds.X, ds.Y, Zs, ck["F"], ck["sigma"],
                                         ck["var_z"], ck["w0"], ck["w"], m))
        q_err = ""
    except Exception as exc:              # diagnostic only, never stops the fit
        q, q_err = float("nan"), f"{type(exc).__name__}: {exc}"[:200]
    Zmode = m_last["Z_hat"] if m_last["Z_hat"] is not None else np.full_like(Zs[:, :, -1], np.nan)
    sc = e2a.score_state(Zs, Zmode, ck["w0"], ck["w"], ds, masks["test"])
    both = m_last["Z_hat"] is not None and m_mean["Z_hat"] is not None
    row = {"w0_hat": ck["w0"], "w_hat": ck["w"], "Q_strict": q, "Q_error": q_err,
           **{f"mode_{k}": v for k, v in m_last.items() if k != "Z_hat"},
           "mode_from_mean5_status": m_mean["status"],
           "modes_max_abs_diff": (float(np.max(np.abs(m_last["Z_hat"] - m_mean["Z_hat"])))
                                  if both else float("nan"))}
    for est, vals in sc.items():
        for metric, v in vals.items():
            row[f"{est}:{metric}"] = v
    st = {"Zsamples": Zs, "F": ck["F"], "sigma": ck["sigma"],
          "scalars": np.array([ck["w0"], ck["w"], ck["var_z"]])}
    if m_last["Z_hat"] is not None:
        st["Zmode"] = m_last["Z_hat"]
    return row, st


def run_fit(fit, ds, masks):
    from em_runner import run_em_experimental                            # noqa: E402

    base = {k: fit[k] for k in ("cell", "replicate", "data_seed", "regime", "arm",
                                "init_seed", "mc_seed")}
    run, ck_rows, traj, states = dict(base), [], [], {}
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        try:
            # E2a call; only num_iter = 40 and checkpoint_iters added
            res = run_em_experimental(
                ds.X, ds.Y, family_x=None, family_y=m1.FAMILY_Y, k=m1.K_FIT, L=m1.L,
                num_iter=NUM_ITER, seed=fit["init_seed"],
                train_mask=masks[fit["regime"]],
                family_x_list=list(m1.ARMS[fit["arm"]]), compute_strict_Q=False,
                validate_support=True, allow_support_mismatch=False,
                numerics_mode="consistent", failure_policy="fail_fast",
                init_seed=fit["init_seed"], mc_seed=fit["mc_seed"],
                checkpoint_iters=set(CHECKPOINTS))
            run.update({"status": "OK", "error": "",
                        "retry_count": res["retry_count"],
                        "replacement_count": res["replacement_count"],
                        "seed_rescue_count": res["seed_rescue_count"],
                        "em_runtime_s": res["runtime_s"]})
            hist = res["theta_history"]
            for h_prev, h in zip([None] + hist[:-1], hist):
                rec = dict(base, iteration=h["iteration"], w0=h["w0"], w=h["w"],
                           F_norm=float(np.linalg.norm(h["F"])))
                if h_prev is not None:
                    rec["rel_change_F"] = float(np.linalg.norm(h["F"] - h_prev["F"])
                                                / np.linalg.norm(h_prev["F"]))
                    rec["abs_change_w0"] = abs(h["w0"] - h_prev["w0"])
                    rec["abs_change_w"] = abs(h["w"] - h_prev["w"])
                    rec["rel_change_sigma"] = float(np.linalg.norm(h["sigma"] - h_prev["sigma"])
                                                    / np.linalg.norm(h_prev["sigma"]))
                traj.append(rec)
            for t in CHECKPOINTS:
                row, st = checkpoint_eval(res["model"], res["checkpoints"][t], ds, masks)
                ck_rows.append(dict(base, iteration=t, **row))
                for k, v in st.items():
                    states[f"{fit_key(fit)}_it{t:02d}_{k}"] = v
            states[f"{fit_key(fit)}_Fhist"] = np.stack([h["F"] for h in hist])
            states[f"{fit_key(fit)}_scalarhist"] = np.array([[h["w0"], h["w"]] for h in hist])
        except Exception as exc:            # recorded, never retried or replaced
            run.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"[:500]})
        run["n_warnings"] = len(wl)
        run["warning_messages"] = " || ".join(sorted(
            {f"{x.category.__name__}: {str(x.message)[:160]}" for x in wl}))[:1000]
    run["wall_runtime_s"] = round(time.perf_counter() - t0, 2)
    return run, ck_rows, traj, states


# ── reproduction of the 8-iteration E2a states ────────────────────────────

def load_refs():
    e1_states, _, _ = e2a.load_e1()                 # checks the E1 sha256
    sha = e1.sha256_file(E2A_STATES)
    if sha != E2A_STATES_SHA256:
        raise SystemExit(f"E2a states sha256 mismatch: {sha}")
    return {"e1": e1_states, "e2a": np.load(E2A_STATES)}


def repro_compare(runs, states, refs) -> dict[str, Any]:
    rows = []
    for run in runs:
        if run["status"] != "OK":
            rows.append({"fit": fit_key(run), "quantity": "status", "abs_diff": float("inf")})
            continue
        src, pre = ref_key(run["cell"], run["replicate"], run["arm"])
        ref = refs[src]
        new = f"{fit_key(run)}_it08_"
        pairs = {"Z_samples": (states[new + "Zsamples"], ref[pre + "Zsamples"]),
                 "F": (states[new + "F"], ref[pre + "F"]),
                 "w0": (states[new + "scalars"][0], ref[pre + "scalars"][0]),
                 "w": (states[new + "scalars"][1], ref[pre + "scalars"][1]),
                 "Z_mode": (states[new + "Zmode"], ref[pre + "Zmode"])}
        for q, (a, b) in pairs.items():
            rows.append({"fit": fit_key(run), "quantity": q,
                         "abs_diff": float(np.max(np.abs(np.asarray(a) - np.asarray(b))))})
    mx = max((r["abs_diff"] for r in rows), default=float("inf"))
    return {"verdict": "MATCH" if mx <= REPRO_RULE["tolerance"] else "MISMATCH",
            "max_abs_diff": mx, "n_fits": len(runs), "n_compared": len(rows),
            "rule": REPRO_RULE, "rows": rows}


# ── analysis ──────────────────────────────────────────────────────────────

def _dist(values) -> dict[str, Any]:
    return e2a._dist(values)


def between_init(ck_rows, states) -> list[dict[str, Any]]:
    idx = {(r["replicate"], r["cell"], r["arm"], r["iteration"]): r for r in ck_rows}
    out = []
    for r in range(len(m1.DATA_SEEDS)):
        rep = m1.replicate_label(r)
        for arm in m1.ARMS:
            for t in CHECKPOINTS:
                a, b = idx.get((rep, "AA", arm, t)), idx.get((rep, "BA", arm, t))
                ka = f"AA_{rep}_{REGIME}_{arm}_it{t:02d}_"
                kb = f"BA_{rep}_{REGIME}_{arm}_it{t:02d}_"
                if a is None or b is None or a["mode_status"] != "OK" or b["mode_status"] != "OK":
                    out.append({"replicate": rep, "arm": arm, "iteration": t, "complete": False})
                    continue
                Za, Zb = states[ka + "Zmode"], states[kb + "Zmode"]
                d, R = aligned_distance(Za, Zb)
                Rr, _ = procrustes_rotation(Zb, Za)
                out.append({
                    "replicate": rep, "arm": arm, "iteration": t, "complete": True,
                    "mode_gram_corr": gram_corr(Za, Zb),
                    "mode_dist_rot_scale": d,
                    "mode_dist_rot": float(calc_rmse(Za, Zb @ Rr)),
                    "F_rel_diff": float(np.linalg.norm(states[ka + "F"] - states[kb + "F"] @ R)
                                        / np.linalg.norm(states[ka + "F"])),
                    "abs_diff_w0": abs(a["w0_hat"] - b["w0_hat"]),
                    "abs_diff_w": abs(a["w_hat"] - b["w_hat"]),
                    "abs_diff_rmse_Z_mode": abs(a["mode:rmse_Z"] - b["mode:rmse_Z"]),
                })
    return out


def within_fit(ck_rows, states) -> list[dict[str, Any]]:
    """Distance of one run's mode between consecutive checkpoints."""
    ok = {(r["replicate"], r["cell"], r["arm"], r["iteration"]) for r in ck_rows
          if r["mode_status"] == "OK"}
    out = []
    for r in range(len(m1.DATA_SEEDS)):
        rep = m1.replicate_label(r)
        for cell in CELLS:
            for arm in m1.ARMS:
                for t0, t1 in zip(CHECKPOINTS[:-1], CHECKPOINTS[1:]):
                    if (rep, cell, arm, t0) not in ok or (rep, cell, arm, t1) not in ok:
                        continue
                    k0 = f"{cell}_{rep}_{REGIME}_{arm}_it{t0:02d}_"
                    k1 = f"{cell}_{rep}_{REGIME}_{arm}_it{t1:02d}_"
                    d, _ = aligned_distance(states[k1 + "Zmode"], states[k0 + "Zmode"])
                    out.append({"replicate": rep, "cell": cell, "arm": arm,
                                "segment": f"{t0}-{t1}", "mode_dist_rot_scale": d,
                                "mode_gram_corr": gram_corr(states[k1 + "Zmode"],
                                                            states[k0 + "Zmode"])})
    return out


def paired_by_checkpoint(ck_rows) -> list[dict[str, Any]]:
    idx = {(r["replicate"], r["cell"], r["arm"], r["iteration"]): r for r in ck_rows}
    out = []
    for r in range(len(m1.DATA_SEEDS)):
        rep = m1.replicate_label(r)
        for cell in CELLS:
            for t in CHECKPOINTS:
                c, w = idx.get((rep, cell, "correct", t)), idx.get((rep, cell, "wrong", t))
                row = {"replicate": rep, "cell": cell, "iteration": t}
                for est in e1.ESTIMATORS:
                    ok = c is not None and w is not None and f"{est}:rmse_Z" in c \
                        and f"{est}:rmse_Z" in w
                    if est == "mode":
                        ok = ok and c["mode_status"] == "OK" and w["mode_status"] == "OK"
                    for metric in e2a.METRICS:
                        row[f"{est}:delta_{metric}"] = (w[f"{est}:{metric}"] - c[f"{est}:{metric}"]) if ok else None
                out.append(row)
    return out


def summarize(runs, ck_rows, traj, btw, wfit, pairs, repro) -> dict[str, Any]:
    s: dict[str, Any] = {
        "issue": ISSUE, "planned_fits": len(planned_fits()), "attempted_fits": len(runs),
        "ok_fits": sum(r["status"] == "OK" for r in runs),
        "integrity": {q: int(sum(int(r.get(q, 0) or 0) for r in runs))
                      for q in ("retry_count", "replacement_count", "seed_rescue_count")},
        "reproduction_it08": {k: v for k, v in repro.items() if k != "rows"}}
    s["mode_status"] = {}
    for t in CHECKPOINTS:
        sel = [r["mode_status"] for r in ck_rows if r["iteration"] == t]
        s["mode_status"][str(t)] = {k: sel.count(k) for k in sorted(set(sel))}
    s["local_solution_gt_1e-6"] = {str(t): int(sum(r["modes_max_abs_diff"] > 1e-6 for r in ck_rows
                                                   if r["iteration"] == t)) for t in CHECKPOINTS}
    s["Q_errors"] = int(sum(bool(r["Q_error"]) for r in ck_rows))
    # primary: between-init closeness per checkpoint
    s["between_init"] = {}
    for arm in m1.ARMS:
        blk = {}
        for t in CHECKPOINTS:
            sel = [b for b in btw if b["arm"] == arm and b["iteration"] == t and b["complete"]]
            blk[str(t)] = {q: _dist([b[q] for b in sel]) for q in
                           ("mode_gram_corr", "mode_dist_rot_scale", "mode_dist_rot", "F_rel_diff",
                            "abs_diff_w0", "abs_diff_w", "abs_diff_rmse_Z_mode")}
        d8 = {b["replicate"]: b for b in btw if b["arm"] == arm and b["iteration"] == 8 and b["complete"]}
        d40 = {b["replicate"]: b for b in btw if b["arm"] == arm and b["iteration"] == 40 and b["complete"]}
        common = sorted(set(d8) & set(d40))
        ch = [d40[r]["mode_dist_rot_scale"] - d8[r]["mode_dist_rot_scale"] for r in common]
        gc = [d40[r]["mode_gram_corr"] - d8[r]["mode_gram_corr"] for r in common]
        blk["change_8_to_40"] = {"replicates": len(common),
                                 "dist_change": _dist(ch),
                                 "n_dist_decreased": int(sum(x < 0 for x in ch)),
                                 "gram_corr_change": _dist(gc),
                                 "n_gram_corr_increased": int(sum(x > 0 for x in gc))}
        s["between_init"][arm] = blk
    # within-fit movement per segment
    s["within_fit"] = {}
    for arm in m1.ARMS:
        s["within_fit"][arm] = {seg: {q: _dist([w[q] for w in wfit if w["arm"] == arm and w["segment"] == seg])
                                      for q in ("mode_dist_rot_scale", "mode_gram_corr")}
                                for seg in ("8-16", "16-24", "24-32", "32-40")}
    # per-iteration theta change
    s["theta_change"] = {}
    for arm in m1.ARMS:
        blk = {}
        for lo, hi in ((2, 8), (9, 16), (17, 24), (25, 32), (33, 40)):
            sel = [x for x in traj if x["arm"] == arm and lo <= x["iteration"] <= hi and "rel_change_F" in x]
            blk[f"{lo}-{hi}"] = {q: _dist([x[q] for x in sel]) for q in
                                 ("rel_change_F", "abs_change_w0", "abs_change_w", "rel_change_sigma")}
        s["theta_change"][arm] = blk
    # accuracy, prediction, objective per checkpoint
    s["by_checkpoint"] = {}
    for cell in CELLS:
        for arm in m1.ARMS:
            blk = {}
            for t in CHECKPOINTS:
                sel = [r for r in ck_rows if r["cell"] == cell and r["arm"] == arm and r["iteration"] == t]
                blk[str(t)] = {q: _dist([r.get(q) for r in sel]) for q in
                               ("mode:rmse_Z", "mode:rmse_Z_scaled", "mode:gram_corr",
                                "mode:heldout_mean_log_score", "mode:heldout_auc",
                                "mode:z_est_norm_ratio", "last:rmse_Z", "mean5:rmse_Z",
                                "Q_strict", "mode_phi", "w0_hat", "w_hat")}
            r8 = {r["replicate"]: r for r in ck_rows if r["cell"] == cell and r["arm"] == arm and r["iteration"] == 8}
            r40 = {r["replicate"]: r for r in ck_rows if r["cell"] == cell and r["arm"] == arm and r["iteration"] == 40}
            com = sorted(set(r8) & set(r40))
            blk["rmse_mode_change_8_to_40"] = _dist([r40[r]["mode:rmse_Z"] - r8[r]["mode:rmse_Z"] for r in com])
            blk["n_rmse_mode_decreased"] = int(sum(r40[r]["mode:rmse_Z"] < r8[r]["mode:rmse_Z"] for r in com))
            blk["logscore_change_8_to_40"] = _dist([r40[r]["mode:heldout_mean_log_score"]
                                                    - r8[r]["mode:heldout_mean_log_score"] for r in com])
            blk["Q_change_8_to_40"] = _dist([r40[r]["Q_strict"] - r8[r]["Q_strict"] for r in com])
            blk["phi_change_8_to_40"] = _dist([r40[r]["mode_phi"] - r8[r]["mode_phi"] for r in com])
            s["by_checkpoint"][f"{cell}_{arm}"] = blk
    # misspecification effect per checkpoint, and init contrast of it
    s["delta_mode_rmse"] = {}
    for t in CHECKPOINTS:
        blk = {}
        for cell in CELLS:
            blk[cell] = _dist([p["mode:delta_rmse_Z"] for p in pairs if p["cell"] == cell and p["iteration"] == t])
        a = {p["replicate"]: p["mode:delta_rmse_Z"] for p in pairs if p["cell"] == "AA" and p["iteration"] == t}
        b = {p["replicate"]: p["mode:delta_rmse_Z"] for p in pairs if p["cell"] == "BA" and p["iteration"] == t}
        com = [r for r in a if a[r] is not None and b.get(r) is not None]
        blk["init_contrast_BA_minus_AA"] = _dist([b[r] - a[r] for r in com])
        blk["sign_agree"] = int(sum((a[r] > 0) == (b[r] > 0) for r in com))
        blk["scaled_" + "AA"] = _dist([p["mode:delta_rmse_Z_scaled"] for p in pairs
                                       if p["cell"] == "AA" and p["iteration"] == t])
        blk["scaled_" + "BA"] = _dist([p["mode:delta_rmse_Z_scaled"] for p in pairs
                                       if p["cell"] == "BA" and p["iteration"] == t])
        s["delta_mode_rmse"][str(t)] = blk
    # descriptive reading
    reading = {}
    for arm in m1.ARMS:
        bi = s["between_init"][arm]
        wf = s["within_fit"][arm]["32-40"]["mode_dist_rot_scale"].get("median")
        b40 = bi["40"]["mode_dist_rot_scale"].get("median")
        reading[arm] = {"n_dist_decreased": bi["change_8_to_40"]["n_dist_decreased"],
                        "rho_between40_over_within32_40": (b40 / wf) if (b40 and wf) else None,
                        "n_rmse_decreased_AA": s["by_checkpoint"][f"AA_{arm}"]["n_rmse_mode_decreased"],
                        "n_rmse_decreased_BA": s["by_checkpoint"][f"BA_{arm}"]["n_rmse_mode_decreased"]}
    approach = all(reading[a]["n_dist_decreased"] >= 14 for a in m1.ARMS)
    rho_ok = all((reading[a]["rho_between40_over_within32_40"] or 0) >= 2 for a in m1.ARMS)
    acc = all(min(reading[a]["n_rmse_decreased_AA"], reading[a]["n_rmse_decreased_BA"]) >= 14
              for a in m1.ARMS)
    label = "A_APPROACH" if approach else ("B_SEPARATE_SETTLED" if rho_ok else "C_SEPARATE_MOVING")
    if not approach and acc:
        label += "+D_ACCURACY_IMPROVES"
    s["reading"] = {**READING_GUIDE, "per_arm": reading, "label": label}
    return s


def protocol() -> dict[str, Any]:
    return {
        "issue": ISSUE, "mode": "CHARACTERIZATION (EM budget 8 -> 40 for init dependence)",
        "base": "E2a (PR #146) commit fa51b4f; E1 (PR #145); M1 frozen design",
        "m1_frozen_design": m1.protocol(), "regime": REGIME,
        "cells": {"AA": "init 1343000+r, MC 1343000+r", "BA": "init 1344000+r, MC 1343000+r"},
        "num_iter": NUM_ITER, "checkpoints": list(CHECKPOINTS),
        "planned_fits": len(planned_fits()),
        "checkpoint_recording": ("em_runner checkpoint_iters (opt-in, copies only, no RNG); "
                                 "theta after each M-step; Z_samples at checkpoints"),
        "mode": "joint mode at theta_t, Z0 = last sample of iteration t, grad_tol 1e-8, max_iter 200",
        "Q": "calc_Q_dual_strict_exp(theta_t | Z_samples_t), post hoc, diagnostic only",
        "reproduction_rule": REPRO_RULE,
        "e2a_states": str(E2A_STATES), "e2a_states_sha256": E2A_STATES_SHA256,
        "primary": "between-init mode Gram correlation and rotation+scale distance, 8 -> 40",
        "reading_guide": READING_GUIDE,
        "limitations": ["L = 5 MC noise remains", "agreement is not a global optimum",
                        "disagreement does not prove multimodality",
                        "one condition, two initial values, 20 replicates"],
        "state_dir": str(STATE_DIR), "no_new_gate_or_auditor": True,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="rep01, AA and BA, both arms (4 fits)")
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
        sm = Path(args.out_dir) / "smoke" / "reproduction_it08.json"
        if not sm.exists() or json.loads(sm.read_text(encoding="utf-8"))["verdict"] != "MATCH":
            raise SystemExit("smoke MATCH is required before the full run")
    refs = load_refs()
    pre = m1.preflight()
    if not pre["pass"]:
        raise SystemExit(f"M1 preflight failed: {pre['failed_checks']}")
    out.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    (out / "protocol.json").write_text(json.dumps(protocol(), indent=2), encoding="utf-8")
    (out / "preflight.json").write_text(json.dumps(pre, indent=2), encoding="utf-8")

    plan = planned_fits()
    if args.smoke:
        plan = [f for f in plan if f["replicate"] == "rep01"]
    started = datetime.now(timezone.utc).isoformat()
    t_all = time.perf_counter()
    cache = {r: (m1.generate(m1.DATA_SEEDS[r]), m1.build_masks(r)) for r in range(len(m1.DATA_SEEDS))}
    runs, ck_rows, traj, states = [], [], [], {}
    for i, fit in enumerate(plan, 1):
        ds, masks = cache[fit["r"]]
        run, rows, tr, st = run_fit(fit, ds, masks)
        runs.append(run)
        ck_rows.extend(rows)
        traj.extend(tr)
        states.update(st)
        rm = "/".join(f"{r['mode:rmse_Z']:.3f}" for r in rows if "mode:rmse_Z" in r)
        print(f"[{i}/{len(plan)}] {fit_key(fit)} {run['status']} "
              f"modes={[r['mode_status'] for r in rows]} rmse_mode(8..40)={rm} "
              f"t={run['wall_runtime_s']}s", flush=True)

    np.savez_compressed(state_dir / "states.npz", **states)
    write = e1.write_csv
    write(out / "fit_runs.csv", runs)
    write(out / "checkpoint_fits.csv", ck_rows)
    write(out / "theta_trajectory.csv", traj)
    repro = repro_compare(runs, states, refs)
    write(out / "reproduction_it08_diffs.csv", repro["rows"])
    (out / "reproduction_it08.json").write_text(
        json.dumps({k: v for k, v in repro.items() if k != "rows"}, indent=2), encoding="utf-8")
    print(f"reproduction at iteration 8: {repro['verdict']} max_abs_diff={repro['max_abs_diff']}")
    if not args.smoke:
        btw = between_init(ck_rows, states)
        wfit = within_fit(ck_rows, states)
        pairs = paired_by_checkpoint(ck_rows)
        write(out / "between_init.csv", btw)
        write(out / "within_fit_segments.csv", wfit)
        write(out / "paired_by_checkpoint.csv", pairs)
        (out / "summary.json").write_text(
            json.dumps(summarize(runs, ck_rows, traj, btw, wfit, pairs, repro), indent=2),
            encoding="utf-8")
    import numpy, scipy, sklearn                                          # noqa: E401
    (out / "runinfo.json").write_text(json.dumps({
        "issue": ISSUE, "script": "expfam/src/experimental/run_em_iterations_e2c.py",
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
        "reproduction_it08_verdict": repro["verdict"],
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
