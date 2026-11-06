"""Separate MCEM parameter movement from Monte Carlo noise (mcem_noise_separation_20261106).

Stacked on PR #156 (mcem_stability_signal_matched_20261106).  Lineage E (numerics_mode="consistent"),
prototype.  Nothing here rewrites PR #156's pre-registered stability verdict; this is a separate,
newly pre-registered study.

Phase 3 (no new MCEM fits): re-analyse the 128 archived RQ1 fit states of PR #156
  (A/B/C/D x rep 1..8 x K_fit 4,5 x start 1,2; n=75, K_GEN=5, d=15, L=5; checkpoints 8..64).
  * at every checkpoint: Laplace joint mode Z_hat(theta_c) (initialised at the archived last draw) ->
    C_Lap and mode-based natural parameters; rotation-invariant Gram F F^T, w0, w.
  * at checkpoints 32 and 64 (fixed-theta probe): R=16 independent E-step chains of length 5 started
    from the archived (theta_c, Z_c) exactly as the algorithm's next E-step would be, each followed by
    the algorithm's M-step -> distribution of the one-step update; the same 16 chains are also used
    as fixed-theta evaluation draws (C_Q with L=5, 20 (4 groups of 4 chains), 80 (all)),
    posterior-mean natural parameters and single-draw sampling noise.
Phase 4 (technical pilot, new fits): MCEM with E-step sample size L in {20, 50}; conditions A and D,
  rep 1..4, K_fit 4 and 5, start 1, the same model seeds as PR #156, 32 iterations.  Distinguishes
  (1) more evaluation draws at a fixed theta (Phase 3) from (2) more E-step draws during estimation.

Usage
  python tools/research_audit/run_mcem_noise_separation.py --gates
  python tools/research_audit/run_mcem_noise_separation.py --phase3 --conds A B
  python tools/research_audit/run_mcem_noise_separation.py --phase4 --L 50 --conds A
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import run_mcem_stability as rm  # noqa: E402  (PR #156 runner: data, constants, helpers)

from em_runner import build_model  # noqa: E402
from em_runner_instrumented import _predictors, _step_diagnostics, run_em_instrumented  # noqa: E402

panel, ss, hist, a1 = rm.panel, rm.ss, rm.hist, rm.a1

EXPERIMENT_ID = "mcem_noise_separation_20261106"
OUT_REL = f"expfam/results/k_selection/{EXPERIMENT_ID}"
OUT_ROOT = rm.ROOT / OUT_REL
STATE_ROOT = Path(r"D:\tento\kennkyu_artifacts") / EXPERIMENT_ID
SRC_STATE_ROOT = rm.STATE_ROOT          # PR #156 archived states (read only)
SRC_RESULTS = rm.OUT_ROOT               # PR #156 results (read only)

N, D = rm.N, rm.D
CONDS = ("A", "B", "C", "D")
REPS = rm.REPS
KS = (4, 5)
STARTS = (1, 2)
CPS_ALL = (8, 16, 24, 32, 48, 64)
CPS_PROBE = (32, 64)
R_CHAINS = 16
L_CHAIN = 5
GROUP_L20 = 4                                       # chains per L=20 group
TAG_PROBE = 7700000
TAG_P4_EVAL = 7710000
P4_LS = (20, 50)
P4_CONDS = ("A", "D")
P4_REPS = (1, 2, 3, 4)
P4_START = 1
P4_MAX_ITER = 32
P4_CPS = (8, 16, 24, 32)
P4_R_STEP = 8
NEWTON_ALPHA = 0.5
NEWTON_MAX_ITER = 10

PREREG: dict[str, Any] = {
    "experiment_id": EXPERIMENT_ID,
    "parent": "mcem_stability_signal_matched_20261106 (PR #156); its pre-registered verdicts are NOT changed",
    "lineage": "E (numerics_mode=consistent), prototype",
    "phase3": {
        "states": "PR #156 RQ1 archived states: A/B/C/D x rep 1..8 x K 4,5 x start 1,2 at checkpoints 8,16,24,32,48,64",
        "no_new_fits": True,
        "laplace_at": list(CPS_ALL),
        "probe_at": list(CPS_PROBE),
        "R_chains": R_CHAINS, "chain_length": L_CHAIN,
        "chain_seed": f"default_rng([{TAG_PROBE}, fit_id, checkpoint, r]), r=0..{R_CHAINS - 1}",
        "chain_start": "archived theta_c and archived last draw Z_c (exactly the state the next E-step starts from)",
        "evaluation_sizes": {"L5": "each chain (16 values)", "L20": "chains 4g..4g+3 (4 values)", "L80": "all chains (1 value)"},
    },
    "phase4": {
        "why": "Phase 3 holds theta fixed and therefore cannot show whether a larger E-step sample size changes "
               "the estimates themselves (the MCEM fixed point / slack).  Decided before any Phase 3 result.",
        "L": list(P4_LS), "conditions": list(P4_CONDS), "replicates": list(P4_REPS), "K_fit": list(KS),
        "start": P4_START, "model_seed": "same as PR #156 (hist.model_seed(5, 75, rep, k, s))",
        "max_iter": P4_MAX_ITER, "checkpoints": list(P4_CPS),
        "probe_at_32": f"one-step replicates R={P4_R_STEP} with the fit's own L; evaluation draws 16 chains x 5 "
                       f"with default_rng([{TAG_P4_EVAL}, fit_id, 32, r])",
        "comparison": "L=5 = PR #156 archived state of the same data/K/start/seed at checkpoint 32 (Phase 3 probe)",
        "fallback": "if the gate projection for Phase 4 exceeds 4 h wall clock with 4 processes, drop L=20 (keep L=50)",
    },
    "definitions": {
        "G": "F F^T (d x d), rotation invariant",
        "rel_change(a,b)": "||a - b||_F / (1e-12 + ||b||_F)",
        "one_step_replicate": "theta' = M-step(E-step chain of length L from (theta_c, Z_c)); g_r = rel_change(G'_r, G_c)",
        "drift_noise_decomposition": "m = mean_r G'_r; s2 = sum_r ||G'_r - m||^2/(R-1); drift2 = max(||m - G_c||^2 - s2/R, 0); "
                                     "drift_share = drift2/(drift2 + s2); same for w0, w (scalars)",
        "eta_diag_runner": "eta_x = Z_last F^T, eta_y = w0 + w z_i^T z_j on the train dyads (the PR #156 diagnostic)",
        "eta_single_draw_noise": "theta fixed at theta_c: rel_change(Z_a F^T, Z_b F^T) for the last draws of independent chains "
                                 "(pairs (0,1),(2,3),...,(14,15)); same for eta_y",
        "eta_posterior_mean": "eta_x_pm = Zbar F^T with Zbar the mean of the 80 fixed-theta draws (= E[eta_x] estimate, linear); "
                              "eta_y_pm = mean over draws of w0 + w z_i^T z_j (NOT the plug-in w0 + w Zbar_i^T Zbar_j, which is also reported)",
        "eta_pm_mc_error": "e = ||eta_pm(half1) - eta_pm(half2)||/(2 ||eta_pm||), halves = chains 0-7 and 8-15",
        "eta_mode": "Laplace joint mode Z_hat(theta) from lk.laplace_log_observed initialised at Z_c; eta_x_mode = Z_hat F^T",
        "slack": "S(4->5) = mean over starts of C(4) - C(5); for L=5 the r-th chains of all four fits are paired",
    },
    "rules": {
        "R1_one_step_decomposition": "per fit at checkpoint 64: drift_share <= 0.2 -> ONE_STEP_NOISE_DOMINATED; >= 0.5 -> "
                                     "ONE_STEP_SYSTEMATIC_DOMINATED; else MIXED (G, w0, w). The systematic one-step component "
                                     "includes mean reversion of a fluctuating state and is NOT by itself evidence of long-run drift "
                                     "(long-run movement is judged only by R5)",
        "R1b_mean_reversion": "cosine between (mean one-step G' - G_c) and (center - G_c), center = mean of G at the other checkpoints "
                              "among 16,24,32,48,64; same sign check for w0 and w (descriptive)",
        "R2_observed_vs_fixed_theta": "ratio = median observed PR#156 diagnostic over iterations 57..64 / median of the 16 one-step replicate "
                                      "values at checkpoint 64; 0.67 <= ratio <= 1.5 -> CONSISTENT_WITH_ONE_STEP_NOISE, "
                                      "> 1.5 -> EXCESS_MOVEMENT, < 0.67 -> LESS_THAN_ONE_STEP (gram, eta_x, eta_y)",
        "R3_eta_diagnostic": "share = median single-draw eta noise at theta_64 / median observed eta diagnostic over 57..64; "
                             ">= 0.8 -> ETA_DIAGNOSTIC_SAMPLING_DOMINATED (x and y separately)",
        "R4_posterior_mean_change": "ratio = rel_change(eta_pm(64), eta_pm(32)) / sqrt(e32^2 + e64^2); > 2 -> PARAMETER_CHANGE_DETECTED",
        "R5_systematic_movement": "per quantity Q in {G, w0, w, eta_x_mode, eta_y_mode}: D(c) = rel_change(Q_64, Q_c). "
                                  "Count fits with D(16) > D(48); SYSTEMATIC_MOVEMENT if two-sided exact binomial p < 0.01 with "
                                  "proportion > 0.5 AND median D(16)/D(48) > 1.1; otherwise NO_DETECTABLE_SYSTEMATIC_MOVEMENT. "
                                  "Fits share datasets (32 datasets) -- reported as a caveat, rule not changed",
        "R6_effect_with_better_evaluation": "paired B-A, C-A, D-A of the L=80 fixed-theta slack at 32 and 64: MAINTAINED if median > 0 and >= 6/8 "
                                            "positive; NOT_MAINTAINED if median <= 0; else INCONCLUSIVE",
        "R7_slack_mc_noise": "sd of the L=5 fixed-theta slack (16 paired values) vs PR #156 |S_fit(64) - S_fit(48)| (descriptive)",
        "R8_phase4": "descriptive pilot, no confirmatory verdict. Prediction (DERIVED, MC scaling): plateau gram_rel_change "
                     "~ 1/sqrt(L) (ratio L50/L5 ~ 0.32); eta single-draw diagnostic does not shrink with L. Slack: "
                     "L_CHANGES_SLACK only if all 4 reps move in the same direction by more than 2 x the combined L=80 MC se; "
                     "else NO_CLEAR_L_EFFECT",
    },
    "failure_rules": "exceptions / non-finite draws -> FAILED row, no retry, no reseed; Laplace not OK -> C_Lap / mode quantities INCOMPLETE, no imputation",
    "disclosure": "Before this registration one technical smoke run of the Phase 3 code (fit A rep1 K5 s1, checkpoints 8..64) "
                  "was executed to check that the code runs and to time it. Its values were seen. After it the R1 labels were "
                  "renamed (thresholds unchanged) and R1b was added. No other result was seen.",
    "budget": {"wall_clock_h": 6.0, "phase3_e_step_draws": 128 * 2 * R_CHAINS * L_CHAIN,
               "phase4_fit_iterations": {"L20": 16 * P4_MAX_ITER, "L50": 16 * P4_MAX_ITER}},
}


# --------------------------------------------------------------------------- helpers
need = rm.need


def prereg_hash() -> str:
    return hashlib.sha256(json.dumps(PREREG, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=rm.ROOT).stdout.strip()


def tree_clean() -> bool:
    out = subprocess.run(["git", "status", "--porcelain", "--", ".", f":(exclude){OUT_REL}"], cwd=rm.ROOT,
                         capture_output=True, text=True).stdout.strip()
    return out == ""


def rel(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / (1e-12 + np.linalg.norm(np.asarray(b))))


def fit_id_of(cond, rep, k, s) -> int:
    return 10000 + 1000 * "ABCD".index(cond) + 100 * rep + 10 * k + s


def num_params(k: int) -> int:
    return k * D - k * (k - 1) // 2


def new_model(k: int, L: int, train):
    m = build_model(N, D, k, L, hist.FAMILY_X, hist.FAMILY_Y, train_mask=train, numerics_mode=hist.NUMERICS_MODE)
    m.initialize_params(true_params=None, seed=0)
    return m


def set_theta(m, th: dict, Z) -> None:
    m.params.update({"Z": np.array(Z, dtype=float).copy(), "F": th["F"], "sigma": th["sigma"],
                     "w0": th["w0"], "w": th["w"], "var_z": th["var_z"]})


def e_step(m, X, Y, th: dict, Z0, L: int, rng) -> np.ndarray:
    """Exactly the E-step of em_runner_instrumented (same draw order, same Newton schedule)."""
    n, k = Z0.shape
    Z = np.array(Z0, dtype=float).copy()
    Zs = np.zeros((n, k, L))
    for s in range(L):
        m.params.update(dict(Z=Z.copy(), F=th["F"], sigma=th["sigma"], w0=th["w0"], w=th["w"]))
        Zn = m.calc_eta_newton(X, Y, rng=rng, max_iter=NEWTON_MAX_ITER, alpha=NEWTON_ALPHA)
        Zs[:, :, s] = Zn
        Z = Zn.copy()
    need(bool(np.all(np.isfinite(Zs))), "non-finite E-step draw")
    return m.scale_Z(Zs)


def m_step(m, X, Y, Zs, th: dict) -> dict:
    """Exactly the M-step of em_runner_instrumented (Bernoulli Y: no sigma_y)."""
    F = m.calc_F(X, Zs)
    sigma = m.calc_sigma(X, Zs, F)
    w0 = float(m.calc_w0(Y, Zs, th["w0"], th["w"], max_iter=50))
    w = float(m.calc_w(Y, Zs, w0, th["w"], max_iter=50))
    return {"F": F, "sigma": sigma, "w0": w0, "w": w, "var_z": th["var_z"]}


def dyads_of(train):
    return np.where(np.triu(np.asarray(train, dtype=bool), 1))


def eta_y_mean_over_draws(Zs, w0, w, rows, cols) -> np.ndarray:
    inner = np.einsum("ijl,ijl->il", Zs[rows], Zs[cols])          # (pairs, L)
    return w0 + w * inner.mean(axis=1)


def laplace_at(m, X, Y, th, Zc, k) -> dict:
    set_theta(m, th, Zc)
    out = panel.laplace_terms(m, X, Y, np.array(Zc, dtype=float), num_params=num_params(k), N=N)
    return out


def load_state(npz, key: str) -> tuple[dict, np.ndarray, np.ndarray]:
    Zs = np.asarray(npz[f"{key}_Zsamples"], float)
    Z = np.asarray(npz[f"{key}_Z"], float)
    w0, w, var_z = (float(v) for v in npz[f"{key}_scalars"])
    th = {"F": np.asarray(npz[f"{key}_F"], float), "sigma": np.asarray(npz[f"{key}_sigma"], float),
          "w0": w0, "w": w, "var_z": var_z}
    return th, Z, Zs


def probe(m, X, Y, train, th: dict, Zc, k: int, *, tag: int, fit_id: int, cp: int, L_step: int, R_step: int,
          eval_tag: int | None) -> tuple[dict, list[dict], dict]:
    """Fixed-theta probe at (theta_c, Z_c).  Returns (summary row, replicate rows, arrays)."""
    rows_d, cols_d = dyads_of(train)
    Gc = th["F"] @ th["F"].T
    ex_prev, ey_prev = _predictors(Zc, th["F"], th["w0"], th["w"], rows_d, cols_d)
    reps, Gs, w0s, ws, chains = [], [], [], [], []
    for r in range(R_step):
        rng = np.random.default_rng([tag, fit_id, cp, r])
        Zs = e_step(m, X, Y, th, Zc, L_step, rng)
        th2 = m_step(m, X, Y, Zs, th)
        ex_now, ey_now = _predictors(Zs[:, :, -1], th2["F"], th2["w0"], th2["w"], rows_d, cols_d)
        dg = _step_diagnostics(th2["F"], th["F"], th2["w"], th["w"], th2["w0"], th["w0"], ex_now, ex_prev, ey_now, ey_prev)
        Gs.append(th2["F"] @ th2["F"].T); w0s.append(th2["w0"]); ws.append(th2["w"])
        reps.append({"r": r, "L_step": L_step, "one_step_gram": dg["gram_rel_change"],
                     "one_step_eta_x": dg["eta_x_rel_change"], "one_step_eta_y": dg["eta_y_rel_change"],
                     "one_step_w0": dg["w0_change"], "one_step_w": dg["w_change"], "w0_new": th2["w0"], "w_new": th2["w"]})
        if eval_tag is None:
            chains.append(Zs)
    if eval_tag is not None:
        for r in range(R_CHAINS):
            rng = np.random.default_rng([eval_tag, fit_id, cp, r])
            chains.append(e_step(m, X, Y, th, Zc, L_CHAIN, rng))
    need(len(chains) == R_CHAINS and all(c.shape[2] == L_CHAIN for c in chains), "evaluation chains")
    # ---- drift / noise decomposition (one-step update) -------------------
    summ: dict[str, Any] = {}
    for nm, vals, ref in (("G", Gs, Gc), ("w0", w0s, th["w0"]), ("w", ws, th["w"])):
        arr = np.array(vals, dtype=float)
        mean = arr.mean(axis=0)
        R = arr.shape[0]
        s2 = float(np.sum((arr - mean) ** 2) / (R - 1))
        dm2 = float(np.sum((mean - ref) ** 2))
        drift2 = max(dm2 - s2 / R, 0.0)
        norm = float(np.linalg.norm(ref)) + 1e-12
        summ[f"{nm}_drift_rel"] = math.sqrt(drift2) / norm
        summ[f"{nm}_noise_rel"] = math.sqrt(s2) / norm
        summ[f"{nm}_drift_share"] = drift2 / (drift2 + s2) if (drift2 + s2) > 0 else float("nan")
    for key in ("one_step_gram", "one_step_eta_x", "one_step_eta_y"):
        summ[f"median_{key}"] = float(np.median([x[key] for x in reps]))
    # ---- fixed-theta evaluation ------------------------------------------
    set_theta(m, th, Zc)
    npar = num_params(k)
    for x in reps:
        x["kind"] = "one_step+eval" if eval_tag is None else "one_step"
    evals = []
    for r, Zs in enumerate(chains):
        cq = panel.cq_terms(m, X, Y, Zs, num_params=npar, n=N)
        e = {"C_Q_L5": cq["C_Q"], "Q_X_L5": cq["Q_X"], "Q_Y_L5": cq["Q_Y"], "Q_Z_L5": cq["Q_Z"]}
        evals.append(e)
        if eval_tag is None:
            reps[r].update(e)
        else:
            reps.append({"r": r, "kind": "eval", "L_step": L_CHAIN, **e})
    for g in range(R_CHAINS // GROUP_L20):
        S = np.concatenate(chains[GROUP_L20 * g: GROUP_L20 * (g + 1)], axis=2)
        cq = panel.cq_terms(m, X, Y, S, num_params=npar, n=N)
        summ[f"C_Q_L20_g{g}"] = cq["C_Q"]
        for q in ("Q_X", "Q_Y", "Q_Z"):
            summ[f"{q}_L20_g{g}"] = cq[q]
    S_all = np.concatenate(chains, axis=2)
    cq = panel.cq_terms(m, X, Y, S_all, num_params=npar, n=N)
    summ.update({"C_Q_L80": cq["C_Q"], "Q_X_L80": cq["Q_X"], "Q_Y_L80": cq["Q_Y"], "Q_Z_L80": cq["Q_Z"]})
    summ["C_Q_L5_sd"] = float(np.std([e["C_Q_L5"] for e in evals], ddof=1))
    for q in ("Q_X", "Q_Y", "Q_Z"):
        summ[f"{q}_L5_sd"] = float(np.std([e[f"{q}_L5"] for e in evals], ddof=1))
    # ---- natural parameters: single draw / posterior mean -----------------
    F = th["F"]
    ex_single, ey_single = [], []
    for a in range(0, R_CHAINS, 2):
        za, zb = chains[a][:, :, -1], chains[a + 1][:, :, -1]
        ex_single.append(rel(za @ F.T, zb @ F.T))
        ea = th["w0"] + th["w"] * np.einsum("ij,ij->i", za[rows_d], za[cols_d])
        eb = th["w0"] + th["w"] * np.einsum("ij,ij->i", zb[rows_d], zb[cols_d])
        ey_single.append(rel(ea, eb))
    summ["median_single_draw_eta_x"] = float(np.median(ex_single))
    summ["median_single_draw_eta_y"] = float(np.median(ey_single))
    h1 = np.concatenate(chains[:R_CHAINS // 2], axis=2)
    h2 = np.concatenate(chains[R_CHAINS // 2:], axis=2)
    zbar, zb1, zb2 = S_all.mean(axis=2), h1.mean(axis=2), h2.mean(axis=2)
    ex_pm = zbar @ F.T
    summ["eta_x_pm_mc_error"] = float(np.linalg.norm(zb1 @ F.T - zb2 @ F.T) / (2 * np.linalg.norm(ex_pm)))
    ey_pm = eta_y_mean_over_draws(S_all, th["w0"], th["w"], rows_d, cols_d)
    ey_pm1 = eta_y_mean_over_draws(h1, th["w0"], th["w"], rows_d, cols_d)
    ey_pm2 = eta_y_mean_over_draws(h2, th["w0"], th["w"], rows_d, cols_d)
    summ["eta_y_pm_mc_error"] = float(np.linalg.norm(ey_pm1 - ey_pm2) / (2 * np.linalg.norm(ey_pm)))
    ey_plug = th["w0"] + th["w"] * np.einsum("ij,ij->i", zbar[rows_d], zbar[cols_d])
    summ["eta_y_plugin_minus_pm_rel"] = rel(ey_plug, ey_pm)
    summ["eta_y_plugin_minus_pm_mean"] = float(np.mean(ey_plug - ey_pm))
    arrays = {"Zbar": zbar, "eta_y_pm": ey_pm, "G_onestep_mean": np.mean(np.array(Gs), axis=0),
              "w0_onestep_mean": np.array(np.mean(w0s)), "w_onestep_mean": np.array(np.mean(ws)),
              # persisted for re-analysis / audit only (operational addendum v2); no effect on any computation
              "eval_chains": S_all, "G_onestep": np.array(Gs), "w0_onestep": np.array(w0s), "w_onestep": np.array(ws)}
    return summ, reps, arrays


def write_rows(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def common_meta(stage: str) -> dict:
    import numpy, scipy, platform  # noqa: E401
    return {"experiment_id": EXPERIMENT_ID, "stage": stage, "git_head": git_head(),
            "branch": subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, cwd=rm.ROOT).stdout.strip(),
            "prereg_sha256": prereg_hash(), "lineage": PREREG["lineage"],
            "environment": {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
                            "platform": platform.platform()}}


def rq1_data():
    prov = a1.load_hist_provenance()
    return rm.rq1_data(prov, ss.w0_c())


def src_npz(cond):
    return np.load(SRC_STATE_ROOT / ("rq1_AB_states.npz" if cond in "AB" else "rq1_CD_states.npz"))


# --------------------------------------------------------------------------- phase 3
def phase3_fit(data, npz, cond, rep, k, s) -> tuple[list[dict], list[dict], list[dict], dict]:
    b = data[rep]; v = b["cond"][cond]
    X, Y, train = v["X"], v["Y"], b["train"]
    fid = fit_id_of(cond, rep, k, s)
    m = new_model(k, L_CHAIN, train)
    rows_d, cols_d = dyads_of(train)
    base = {"experiment_id": EXPERIMENT_ID, "phase": 3, "condition": cond, "replicate": rep, "k_fit": k, "start": s,
            "fit_id": fid, "L_fit": 5}
    state_rows, probe_rows, rep_rows, arrays = [], [], [], {}
    Q: dict[int, dict] = {}
    for cp in CPS_ALL:
        key = f"RQ1_{cond}_r{rep}_k{k}_s{s}_it{cp}"
        th, Zc, Zs = load_state(npz, key)
        need(np.array_equal(Zc, Zs[:, :, -1]), f"Z != last draw at {key}")
        lap = laplace_at(m, X, Y, th, Zc, k)
        zhat = lap.pop("Z_hat")
        ok = lap["lap_status"] == "OK"
        ex_mode = zhat @ th["F"].T
        ey_mode = th["w0"] + th["w"] * np.einsum("ij,ij->i", zhat[rows_d], zhat[cols_d])
        Q[cp] = {"G": th["F"] @ th["F"].T, "w0": np.array(th["w0"]), "w": np.array(th["w"]),
                 "eta_x_mode": ex_mode if ok else None, "eta_y_mode": ey_mode if ok else None}
        row = {**base, "checkpoint": cp, "w0": th["w0"], "w": th["w"], "gram_fro": float(np.linalg.norm(Q[cp]["G"])),
               "lap_status": lap["lap_status"], "C_Lap": lap["C_Lap"] if ok else float("nan"),
               "lap_grad_inf": lap["lap_grad_inf"]}
        state_rows.append(row)
        arrays[f"{key}_Zhat"] = zhat
        if cp in CPS_PROBE:
            t0 = time.time()
            summ, reps, arr = probe(m, X, Y, train, th, Zc, k, tag=TAG_PROBE, fit_id=fid, cp=cp, L_step=L_CHAIN,
                                    R_step=R_CHAINS, eval_tag=None)
            probe_rows.append({**base, "checkpoint": cp, **summ, "probe_seconds": round(time.time() - t0, 2)})
            for x in reps:
                rep_rows.append({**base, "checkpoint": cp, **x})
            for nm, a in arr.items():
                arrays[f"{key}_{nm}"] = a
            arrays[f"{key}_theta_F"] = th["F"]
            Q[cp]["eta_x_pm"] = arr["Zbar"] @ th["F"].T
            Q[cp]["eta_y_pm"] = arr["eta_y_pm"]
    # ---- distances to checkpoint 64 ----------------------------------------
    for row in state_rows:
        cp = row["checkpoint"]
        for nm in ("G", "w0", "w", "eta_x_mode", "eta_y_mode"):
            a, b64 = Q[cp][nm], Q[64][nm]
            row[f"D64_{nm}"] = float("nan") if (a is None or b64 is None or cp == 64) else rel(b64, a)
        prev = [c for c in CPS_ALL if c < cp]
        if prev:
            p = prev[-1]
            for nm in ("G", "eta_x_mode"):
                a, bb = Q[p][nm], Q[cp][nm]
                row[f"Dprev_{nm}"] = float("nan") if (a is None or bb is None) else rel(bb, a)
    for pr in probe_rows:
        cp = pr["checkpoint"]
        others = [c for c in (16, 24, 32, 48, 64) if c != cp]
        key = f"RQ1_{cond}_r{rep}_k{k}_s{s}_it{cp}"
        for nm in ("G", "w0", "w"):
            center = np.mean([Q[c][nm] for c in others], axis=0)
            step = np.asarray(arrays[f"{key}_{nm}_onestep_mean"]) - Q[cp][nm]
            toward = center - Q[cp][nm]
            if nm == "G":
                pr["G_mean_reversion_cos"] = float(np.sum(step * toward) / (np.linalg.norm(step) * np.linalg.norm(toward) + 1e-12))
            else:
                pr[f"{nm}_step_toward_center"] = bool(np.sign(step) == np.sign(toward))
        if cp == 64:
            pr["D_eta_x_pm_32_64"] = rel(Q[64]["eta_x_pm"], Q[32]["eta_x_pm"])
            pr["D_eta_y_pm_32_64"] = rel(Q[64]["eta_y_pm"], Q[32]["eta_y_pm"])
    return state_rows, probe_rows, rep_rows, arrays


def run_phase3(conds) -> int:
    """SUPERSEDED by run_mcem_noise_separation_v2.py (per-fit persistence); kept for the record."""
    stage = "phase3_" + "".join(conds)
    out = OUT_ROOT / stage
    need(tree_clean(), "working tree not clean (outside this experiment's output dir)")
    need((OUT_ROOT / "gates" / "gates.json").exists(), "gates not run")
    g = json.loads((OUT_ROOT / "gates" / "gates.json").read_text(encoding="utf-8"))
    need(g.get("status") == "PASSED" and g.get("prereg_sha256") == prereg_hash(), "gates not passed / prereg changed")
    need(not out.exists(), f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    data = rq1_data()
    t0 = time.time()
    S, P, Rr, ledger, arrays = [], [], [], [], {}
    for cond in conds:
        npz = src_npz(cond)
        for rep in REPS:
            for k in KS:
                for s in STARTS:
                    t1 = time.time()
                    try:
                        a, b, c, arr = phase3_fit(data, npz, cond, rep, k, s)
                        S += a; P += b; Rr += c; arrays.update(arr)
                        status, exc = "OK", ""
                    except Exception as e:  # recorded, not retried
                        status, exc = "FAILED", repr(e)
                    ledger.append({"experiment_id": EXPERIMENT_ID, "phase": 3, "condition": cond, "replicate": rep, "k_fit": k,
                                   "start": s, "fit_id": fit_id_of(cond, rep, k, s), "status": status, "exception": exc,
                                   "seconds": round(time.time() - t1, 1)})
                    print(f"[{stage}] {cond} rep{rep} K{k} s{s} {status} ({time.time() - t0:.0f}s)", flush=True)
    write_rows(out / "state_rows.csv", S)
    write_rows(out / "probe_rows.csv", P)
    write_rows(out / "replicate_rows.csv", Rr)
    write_rows(out / "ledger.csv", ledger)
    npz_path = STATE_ROOT / f"{stage}_arrays.npz"
    np.savez_compressed(npz_path, **arrays)
    info = {**common_meta(stage), "status": "COMPLETED", "fits": len(ledger),
            "failed": sum(x["status"] != "OK" for x in ledger), "wall_clock_seconds": round(time.time() - t0, 1),
            "inputs": {c: {"path": str(SRC_STATE_ROOT / ("rq1_AB_states.npz" if c in "AB" else "rq1_CD_states.npz")),
                           "sha256": sha_file(SRC_STATE_ROOT / ("rq1_AB_states.npz" if c in "AB" else "rq1_CD_states.npz"))}
                       for c in conds},
            "arrays": str(npz_path), "arrays_sha256": sha_file(npz_path)}
    (out / "runinfo.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- phase 4
def run_p4_fit(X, Y, train, k, seed, L):
    snaps, traj = {}, {}

    def cb(rec):
        it = int(rec["iteration"])
        traj[it] = (float(rec["q_strict"]), float(rec["bic"]))
        if it in P4_CPS:
            snaps[it] = {"Zs": np.array(rec["Z_samples"]), "Z": np.array(rec["Z_est"]),
                         "th": {"F": np.array(rec["F"]), "sigma": np.array(rec["sigma"]), "w0": float(rec["w0"]),
                                "w": float(rec["w"]), "var_z": float(rec["var_z"])}, "bic": float(rec["bic"])}
    res = run_em_instrumented(X, Y, family_x=hist.FAMILY_X, family_y=hist.FAMILY_Y, k=k, L=L, max_iter=P4_MAX_ITER,
                              checkpoints=tuple(range(1, P4_MAX_ITER + 1)), seed=seed, train_mask=train,
                              numerics_mode=hist.NUMERICS_MODE, stability_window=rm.STAB_WINDOW, on_nan="fail",
                              on_checkpoint=cb)
    need(res["retry_count"] == 0 and not res["nan_detected"], "unexpected retry")
    return res, snaps, traj


def phase4_fit_id(L: int, cond: str, rep: int, k: int) -> int:
    return 40000 + 1000 * P4_LS.index(L) + 100 * "ABCD".index(cond) + 10 * rep + k


def phase4_fit(data, cond: str, rep: int, k: int, L: int, *, max_iter: int = P4_MAX_ITER):
    """One Phase 4 fit + Laplace at every checkpoint + probe at the last checkpoint.
    ``max_iter`` other than P4_MAX_ITER is for tests only.  Returns (base, C, P, R, IT, arrays)."""
    s = P4_START
    cps = tuple(c for c in P4_CPS if c <= max_iter)
    b = data[rep]; v = b["cond"][cond]
    X, Y, train = v["X"], v["Y"], b["train"]
    seed = hist.model_seed(5, N, rep, k, s)
    fid = phase4_fit_id(L, cond, rep, k)
    base = {"experiment_id": EXPERIMENT_ID, "phase": 4, "condition": cond, "replicate": rep, "k_fit": k,
            "start": s, "fit_id": fid, "L_fit": L, "model_seed": seed}
    C, P, Rr, IT, arrays = [], [], [], [], {}
    snaps, traj = {}, {}

    def cb(rec):
        it = int(rec["iteration"])
        traj[it] = (float(rec["q_strict"]), float(rec["bic"]))
        if it in cps:
            snaps[it] = {"Zs": np.array(rec["Z_samples"]), "Z": np.array(rec["Z_est"]),
                         "th": {"F": np.array(rec["F"]), "sigma": np.array(rec["sigma"]), "w0": float(rec["w0"]),
                                "w": float(rec["w"]), "var_z": float(rec["var_z"])}, "bic": float(rec["bic"])}
    res = run_em_instrumented(X, Y, family_x=hist.FAMILY_X, family_y=hist.FAMILY_Y, k=k, L=L, max_iter=max_iter,
                              checkpoints=tuple(range(1, max_iter + 1)), seed=seed, train_mask=train,
                              numerics_mode=hist.NUMERICS_MODE, stability_window=rm.STAB_WINDOW, on_nan="fail",
                              on_checkpoint=cb)
    need(res["retry_count"] == 0 and not res["nan_detected"], "unexpected retry")
    m = new_model(k, L, train)
    for cp in cps:
        sn = snaps[cp]
        lap = laplace_at(m, X, Y, sn["th"], sn["Z"], k)
        zhat = lap.pop("Z_hat")
        C.append({**base, "checkpoint": cp, "C_Q_fit": sn["bic"], "w0": sn["th"]["w0"], "w": sn["th"]["w"],
                  "lap_status": lap["lap_status"],
                  "C_Lap": lap["C_Lap"] if lap["lap_status"] == "OK" else float("nan"),
                  "lap_grad_inf": lap["lap_grad_inf"]})
        key = f"P4_L{L}_{cond}_r{rep}_k{k}_s{s}_it{cp}"
        for nm, a in (("Zsamples", sn["Zs"]), ("Z", sn["Z"]), ("F", sn["th"]["F"]),
                      ("sigma", sn["th"]["sigma"]), ("Zhat", zhat)):
            arrays[f"{key}_{nm}"] = a
        arrays[f"{key}_scalars"] = np.array([sn["th"]["w0"], sn["th"]["w"], sn["th"]["var_z"]])
    last = cps[-1]
    sn = snaps[last]
    summ, reps, arr = probe(m, X, Y, train, sn["th"], sn["Z"], k, tag=TAG_PROBE + 1000 * L, fit_id=fid,
                            cp=last, L_step=L, R_step=P4_R_STEP, eval_tag=TAG_P4_EVAL)
    P.append({**base, "checkpoint": last, **summ})
    for x in reps:
        Rr.append({**base, "checkpoint": last, **x})
    for nm, a in arr.items():
        arrays[f"P4_L{L}_{cond}_r{rep}_k{k}_s{s}_it{last}_{nm}"] = a
    for it, dg in enumerate(res["iteration_diagnostics"], start=1):
        IT.append({**base, "iteration": it, "C_Q_fit": traj[it][1], **{kk: float(vv) for kk, vv in dg.items()}})
    return base, C, P, Rr, IT, arrays


def run_phase4(L: int, conds) -> int:
    """SUPERSEDED by run_mcem_noise_separation_v2.py (per-fit persistence); kept for the record."""
    need(L in P4_LS, "L not pre-registered")
    stage = f"phase4_L{L}_" + "".join(conds)
    out = OUT_ROOT / stage
    need(tree_clean(), "working tree not clean")
    g = json.loads((OUT_ROOT / "gates" / "gates.json").read_text(encoding="utf-8"))
    need(g.get("status") == "PASSED" and g.get("prereg_sha256") == prereg_hash(), "gates not passed / prereg changed")
    need(L in g.get("phase4_L_allowed", []), f"L={L} not allowed by the gate projection")
    need(not out.exists(), f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    data = rq1_data()
    t0 = time.time()
    C, P, Rr, IT, ledger, arrays = [], [], [], [], [], {}
    for cond in conds:
        for rep in P4_REPS:
            for k in KS:
                s = P4_START
                b = data[rep]; v = b["cond"][cond]
                X, Y, train = v["X"], v["Y"], b["train"]
                seed = hist.model_seed(5, N, rep, k, s)
                fid = 40000 + 1000 * P4_LS.index(L) + 100 * "ABCD".index(cond) + 10 * rep + k
                base = {"experiment_id": EXPERIMENT_ID, "phase": 4, "condition": cond, "replicate": rep, "k_fit": k,
                        "start": s, "fit_id": fid, "L_fit": L, "model_seed": seed}
                t1 = time.time()
                try:
                    res, snaps, traj = run_p4_fit(X, Y, train, k, seed, L)
                    m = new_model(k, L, train)
                    for cp in P4_CPS:
                        sn = snaps[cp]
                        lap = laplace_at(m, X, Y, sn["th"], sn["Z"], k)
                        zhat = lap.pop("Z_hat")
                        C.append({**base, "checkpoint": cp, "C_Q_fit": sn["bic"], "w0": sn["th"]["w0"], "w": sn["th"]["w"],
                                  "lap_status": lap["lap_status"],
                                  "C_Lap": lap["C_Lap"] if lap["lap_status"] == "OK" else float("nan"),
                                  "lap_grad_inf": lap["lap_grad_inf"]})
                        key = f"P4_L{L}_{cond}_r{rep}_k{k}_s{s}_it{cp}"
                        for nm, a in (("Zsamples", sn["Zs"]), ("Z", sn["Z"]), ("F", sn["th"]["F"]),
                                      ("sigma", sn["th"]["sigma"]), ("Zhat", zhat)):
                            arrays[f"{key}_{nm}"] = a
                        arrays[f"{key}_scalars"] = np.array([sn["th"]["w0"], sn["th"]["w"], sn["th"]["var_z"]])
                    sn = snaps[P4_MAX_ITER]
                    summ, reps, arr = probe(m, X, Y, train, sn["th"], sn["Z"], k, tag=TAG_PROBE + 1000 * L, fit_id=fid,
                                            cp=P4_MAX_ITER, L_step=L, R_step=P4_R_STEP, eval_tag=TAG_P4_EVAL)
                    P.append({**base, "checkpoint": P4_MAX_ITER, **summ})
                    for x in reps:
                        Rr.append({**base, "checkpoint": P4_MAX_ITER, **x})
                    for it, dg in enumerate(res["iteration_diagnostics"], start=1):
                        IT.append({**base, "iteration": it, "C_Q_fit": traj[it][1],
                                   **{kk: float(vv) for kk, vv in dg.items()}})
                    status, exc = "OK", ""
                except Exception as e:
                    status, exc = "FAILED", repr(e)
                ledger.append({**base, "status": status, "exception": exc, "fit_iterations": P4_MAX_ITER,
                               "seconds": round(time.time() - t1, 1)})
                print(f"[{stage}] {cond} rep{rep} K{k} {status} ({time.time() - t0:.0f}s)", flush=True)
    write_rows(out / "checkpoint_rows.csv", C)
    write_rows(out / "probe_rows.csv", P)
    write_rows(out / "replicate_rows.csv", Rr)
    write_rows(out / "iteration_rows.csv", IT)
    write_rows(out / "ledger.csv", ledger)
    npz_path = STATE_ROOT / f"{stage}_states.npz"
    np.savez_compressed(npz_path, **arrays)
    info = {**common_meta(stage), "status": "COMPLETED", "fits": len(ledger),
            "failed": sum(x["status"] != "OK" for x in ledger), "wall_clock_seconds": round(time.time() - t0, 1),
            "states": str(npz_path), "states_sha256": sha_file(npz_path)}
    (out / "runinfo.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- gates
def run_gates() -> int:
    out = OUT_ROOT / "gates"
    need(tree_clean(), "working tree not clean")
    need(not out.exists(), f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    (out / "prereg.json").write_text(json.dumps(PREREG, indent=1, ensure_ascii=False), encoding="utf-8")
    gates: dict[str, Any] = {}
    data = rq1_data()
    # G1: e_step / m_step / diagnostics reproduce the runner's iteration 1 bit for bit
    rep, k, s, cond = 1, 5, 1, "A"
    b = data[rep]; v = b["cond"][cond]; X, Y, train = v["X"], v["Y"], b["train"]
    seed = hist.model_seed(5, N, rep, k, s)
    rec = {}
    res = run_em_instrumented(X, Y, family_x=hist.FAMILY_X, family_y=hist.FAMILY_Y, k=k, L=L_CHAIN, max_iter=1,
                              checkpoints=(1,), seed=seed, train_mask=train, numerics_mode=hist.NUMERICS_MODE,
                              stability_window=1, on_nan="fail", on_checkpoint=lambda r: rec.update(
                                  {"Zs": np.array(r["Z_samples"]), "F": np.array(r["F"]), "w0": r["w0"], "w": r["w"]}))
    rng = np.random.default_rng(seed)
    m = build_model(N, D, k, L_CHAIN, hist.FAMILY_X, hist.FAMILY_Y, train_mask=train, numerics_mode=hist.NUMERICS_MODE)
    m.initialize_params(true_params=None, seed=seed)
    y_obs = Y[np.triu(m.train_mask, k=1)]
    dens = float(np.clip(y_obs.mean(), 1e-6, 1 - 1e-6))
    m.params["w0"] = np.log(dens / (1 - dens)); m.params["w"] = 0.5; m.params["F"] *= 0.2
    th0 = {"F": m.params["F"].copy(), "sigma": m.params["sigma"].copy(), "w0": float(m.params["w0"]),
           "w": float(m.params["w"]), "var_z": float(m.params["var_z"])}
    Z0 = m.params["Z"].copy()
    Zs = e_step(m, X, Y, th0, Z0, L_CHAIN, rng)
    th1 = m_step(m, X, Y, Zs, th0)
    rd, cd = dyads_of(train)
    ex1, ey1 = _predictors(Zs[:, :, -1], th1["F"], th1["w0"], th1["w"], rd, cd)
    ex0, ey0 = _predictors(Z0, th0["F"], th0["w0"], th0["w"], rd, cd)
    dg = _step_diagnostics(th1["F"], th0["F"], th1["w"], th0["w"], th1["w0"], th0["w0"], ex1, ex0, ey1, ey0)
    d1 = max(float(np.max(np.abs(Zs - rec["Zs"]))), float(np.max(np.abs(th1["F"] - rec["F"]))),
             abs(th1["w0"] - rec["w0"]), abs(th1["w"] - rec["w"]))
    d2 = max(abs(dg[kk] - res["iteration_diagnostics"][0][kk]) for kk in ("gram_rel_change", "eta_x_rel_change", "eta_y_rel_change"))
    need(d1 == 0.0 and d2 == 0.0, f"G1 failed: state diff {d1}, diag diff {d2}")
    gates["G1_one_step_reproduces_runner"] = f"PASS (state diff {d1}, diagnostic diff {d2})"
    # G2: archived states load; Z == last draw; archived C_Q / C_Lap recomputed
    import pandas as pd
    cr = pd.concat([pd.read_csv(SRC_RESULTS / st / "checkpoint_rows.csv") for st in ("rq1_AB", "rq1_CD")])
    npz = src_npz(cond)
    key = f"RQ1_{cond}_r{rep}_k{k}_s{s}_it64"
    th, Zc, Zs64 = load_state(npz, key)
    need(np.array_equal(Zc, Zs64[:, :, -1]), "G2: Z != last draw")
    m5 = new_model(k, L_CHAIN, train)
    set_theta(m5, th, Zc)
    cq = panel.cq_terms(m5, X, Y, Zs64, num_params=num_params(k), n=N)
    row = cr[(cr.condition == cond) & (cr.replicate == rep) & (cr.k_fit == k) & (cr.start == s) & (cr.checkpoint == 64)].iloc[0]
    lap = laplace_at(m5, X, Y, th, Zc, k)
    dq, dl = abs(cq["C_Q"] - row.C_Q), abs(lap["C_Lap"] - row.C_Lap)
    need(dq < 1e-6 and dl < 1e-6, f"G2 failed: C_Q diff {dq}, C_Lap diff {dl}")
    gates["G2_archived_state_recompute"] = f"PASS (C_Q diff {dq:.1e}, C_Lap diff {dl:.1e}; Laplace deterministic given theta, Z_c)"
    # G3: timing of one probe at checkpoint 64 and one L=50 E-step
    t0 = time.time()
    summ, reps, _ = probe(m5, X, Y, train, th, Zc, k, tag=TAG_PROBE + 999, fit_id=0, cp=0, L_step=L_CHAIN,
                          R_step=R_CHAINS, eval_tag=None)
    t_probe = time.time() - t0
    t0 = time.time()
    e_step(m5, X, Y, th, Zc, 50, np.random.default_rng([TAG_PROBE + 998]))
    t_e50 = time.time() - t0
    per_draw = t_e50 / 50
    phase3_h = 128 * (2 * t_probe + 6 * 0.3) / 3600 / 4
    p4_fit_s = {L: P4_MAX_ITER * (per_draw * L + 0.05) for L in P4_LS}
    p4_probe_s = {L: per_draw * (P4_R_STEP * L + R_CHAINS * L_CHAIN) + 2 for L in P4_LS}
    p4_h = {L: 16 * (p4_fit_s[L] + p4_probe_s[L]) / 3600 / 2 for L in P4_LS}     # 2 processes per L (A, D)
    allowed = list(P4_LS) if max(p4_h.values()) <= 4.0 else [50]
    gates["G3_timing"] = {"probe_seconds": round(t_probe, 1), "seconds_per_draw": round(per_draw, 3),
                          "phase3_projected_h_4procs": round(phase3_h, 2),
                          "phase4_projected_h_parallel": {str(L): round(h, 2) for L, h in p4_h.items()},
                          "note": "technical only; probe values not used"}
    info = {**common_meta("gates"), "status": "PASSED", "gates": gates, "phase4_L_allowed": allowed}
    (out / "gates.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gates", action="store_true")
    g.add_argument("--phase3", action="store_true")
    g.add_argument("--phase4", action="store_true")
    ap.add_argument("--conds", nargs="+", choices=list(CONDS))
    ap.add_argument("--L", type=int)
    a = ap.parse_args(argv)
    if a.gates:
        return run_gates()
    need(a.conds, "--conds required")
    if a.phase3:
        return run_phase3(a.conds)
    need(set(a.conds) <= set(P4_CONDS), "phase 4 conditions are A and D")
    return run_phase4(a.L, a.conds)


if __name__ == "__main__":
    sys.exit(main())
