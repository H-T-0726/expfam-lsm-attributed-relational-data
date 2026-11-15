"""AF8-v3 (exploratory, numerical stability): certified optimisation for loading-integrated family selection.

AF8-v2 (a5cffa1) stopped at its pilot by rule S4 (9/72 main cells with one failed IS draw).  Diagnosis at the
exact failed draws: the inner Newton stalls with ||g||_inf ~ 1e-6 because the Armijo test compares phi values
whose predicted change (Newton decrement lambda^2/2 ~ 1e-14) is below the floating-point resolution of phi
(1 ulp of |phi|~2e3 is 4.5e-13); one pure Newton step then gives ||g|| ~ 1e-14.  At such floor-accepted points
the Laplace value S = phi - 1/2 log|H| is wrong by O(||H^-1 g||) through the log-determinant (3.9e-8 observed),
which divided by the FD step (1e-4) is the outer FD-gradient noise that blocked the outer criterion.

v3 changes ONLY the optimisation / acceptance numerics; the mathematics (a8), data, priors, starts, FD steps,
IS proposal/draws/seeds and E2 as main score are those of v2.  AF8 and AF8-v2 files are not modified.

Acceptance is a CERTIFICATE at the returned point, not an absolute gradient threshold:
  inner/joint (x = vec Z or u):  negH PD (Cholesky)  AND  lambda^2/2 = g^T H^-1 g / 2 <= CERT_DEC
                                 AND |Lap(x + H^-1 g) - Lap(x)| <= CERT_LAP,  Lap = phi - 1/2 log|negH|
  outer (f, K = 3, FD):          FD Hessian of -T PD  AND  lambda_f^2/2 <= OUTER_CERT_DEC
                                 AND T(f + H^-1 g) - T(f) <= OUTER_CERT_DEC  (an extra Newton step does not improve)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_af8_integrated_loading as a8          # noqa: E402  (math only; unchanged)
import run_af8v2_integrated_loading as v2        # noqa: E402  (shared helpers only; unchanged)

OUT = a8.RES / "af8v3_numerical_stability_20261115"
N, K, DU = a8.N, a8.K, a8.DU
LOG2PI = a8.LOG2PI
P, C = a8.P, a8.C
CANDS, REGIMES = a8.CANDS, a8.REGIMES
PRIORS, MAIN_PRIOR = a8.PRIORS, a8.MAIN_PRIOR
V2_HEAD = "a5cffa1b7364988786b984137ffb1055a3ec947f"
EvalFail, safe, chol_logdet = v2.EvalFail, v2.safe, v2.chol_logdet

# ---------------- unchanged from v2 ----------------
INNER_GTOL, JOINT_GTOL = v2.INNER_GTOL, v2.JOINT_GTOL      # aims of the Armijo phase
INNER_MAXITER, JOINT_MAXITER = v2.INNER_MAXITER, v2.JOINT_MAXITER
ARMIJO_C, MIN_STEP, EIG_FLOOR_REL = v2.ARMIJO_C, v2.MIN_STEP, v2.EIG_FLOOR_REL
INNER_STEP_CAP, OUTER_STEP_CAP = v2.INNER_STEP_CAP, v2.OUTER_STEP_CAP
FD_H_GRAD, FD_H_HESS, FD_H_HESS_ALT = v2.FD_H_GRAD, v2.FD_H_HESS, v2.FD_H_HESS_ALT
IS_DRAWS, IS_DF, IS_SEED_BASE = v2.IS_DRAWS, v2.IS_DF, v2.IS_SEED_BASE   # SAME draws as v2
NUMERIC_FLOOR = v2.NUMERIC_FLOOR                                         # 0.01 decision floor
PILOT_REPS = v2.PILOT_REPS

# ---------------- v3 certificates (derivation in mathematical_notes_v3.md) ----------------
TAU_E2 = 1e-3            # budget for the numerical error of one candidate's E2 (= NUMERIC_FLOOR / 10)
CERT_DEC = 1e-12         # inner/joint: lambda^2/2 (local-quadratic objective gap)
CERT_LAP = 1e-10         # inner/joint: |Laplace value change under one more full Newton step|
RESOLUTION_ULPS = 64     # Armijo phase hands over to polish when lambda^2/2 <= 64 ulp(|phi|) (negH PD, uncapped step)
POLISH_MAX = 10          # pure Newton steps, accepted only if negH stays PD and lambda^2 strictly decreases
OUTER_CERT_DEC = 1e-6    # outer: lambda_f^2/2 and the gain of one more outer Newton step
OUTER_CERT_MAXITER = 8
RESCUE_SAME_MODE_TOL = 1e-6   # IS: a cold start that failed certification may be replaced by the certified mode
                              # from Z(f_D) only if its last phi is within this of that mode (same mode)
NM_MAXFEV = 400

STOP_RULES_V3 = {
    "S1_cell_numerical_failure_share_max": 0.05,                 # as v2 (E2 without a certified local max)
    "S2_multistart_disagreement_share_max": 0.05,                # as v2 (among certified starts)
    "S2b_cells_with_fewer_than_2_certified_starts_share_max": 0.05,   # NEW (audit W: single-start AGREE is trivial)
    "S3_main_median_abs_DeltaE2_minus_DeltaIS_max": 0.25,       # as v2
    "S3b_main_median_abs_DeltaE2_minus_DeltaIS_below_half_median_abs_DeltaE2": True,
    "S4_IS_cells_with_failed_draws_share_max": 0.05,             # as v2, failed AFTER the pre-registered rescue
    "S4r_IS_cells_with_any_rescued_draw_share_max": 0.05,        # NEW: rescue may not mask a systematic failure
    "S4b_IS_min_ess": 40.0,                                      # as v2
    "S5_determinism_max_abs_diff": 1e-8,                         # as v2
    "S6_fd_step_sensitivity_median_abs_DeltaE2_max": 0.05,       # as v2
    "S7_fd_derivative_rel_err_max": 1e-4,                        # as v2
    "S8_nelder_mead_gain_over_fD_max": TAU_E2,                   # NEW: FD-free check of the outer maximum
    "S9_fd_noise_implied_gap_max": OUTER_CERT_DEC,               # NEW: (FD-gradient spread over h)^2/(2 lambda_min)
}


# --------------------------------------------------------------------------
# certified Newton
# --------------------------------------------------------------------------

@dataclass
class NewtonResult3:
    status: str               # OK | NOT_CERTIFIED | SADDLE | LINESEARCH_FAIL | NOT_CONVERGED | START_INVALID
    x: np.ndarray
    value: float
    grad_inf: float
    min_eig: float
    iters: int
    lam2_half: float = float("nan")
    lap_step_change: float = float("nan")
    n_polish: int = 0
    n_modified: int = 0
    n_negcurv_escape: int = 0
    n_rejected_trials: int = 0
    phase_exit: str = ""      # how the Armijo phase ended: GTOL | RESOLUTION | LINESEARCH_RESOLUTION | MAXITER
    logdet: float = float("nan")
    message: str = ""


def _lap(v, H):
    pd_ok, ld, me = chol_logdet(H)
    return pd_ok, (v - 0.5 * ld) if pd_ok else float("nan"), ld


def certify(vgh: Callable, x, v, g, H) -> tuple[bool, dict]:
    """certificate at x: negH PD, lambda^2/2 <= CERT_DEC, |Lap(x+p)-Lap(x)| <= CERT_LAP (p full Newton step)."""
    Hs = 0.5 * (H + H.T)
    pd_ok, lap0, ld = _lap(v, Hs)
    if not pd_ok:
        return False, {"why": "negH_not_PD", "min_eig": float(np.linalg.eigvalsh(Hs)[0])}
    p = np.linalg.solve(Hs, g)
    lam2h = 0.5 * float(g @ p)
    info = {"lam2_half": lam2h, "logdet": ld, "min_eig": float(np.linalg.eigvalsh(Hs)[0])}
    try:
        vt, _, Ht = vgh(x + p)
    except EvalFail as exc:
        info["why"] = f"newton_step_invalid: {exc}"
        return False, info
    okt, lapt, _ = _lap(vt, 0.5 * (Ht + Ht.T))
    if not okt:
        info["why"] = "negH_not_PD_after_step"
        return False, info
    info["lap_step_change"] = abs(lapt - lap0)
    ok = lam2h <= CERT_DEC and info["lap_step_change"] <= CERT_LAP
    if not ok:
        info["why"] = "decrement_or_laplace_change_above_certificate"
    return ok, info


def polish(vgh: Callable, x, v, g, H):
    """pure Newton steps (no function-value test): accepted only if the trial is valid, negH stays PD and the
    Newton decrement strictly decreases.  Near a nondegenerate local max this is the quadratically convergent
    iteration whose progress phi cannot resolve; it cannot leave the PD region and never accepts an invalid point."""
    n = 0
    Hs = 0.5 * (H + H.T)
    pd_ok, _, _ = chol_logdet(Hs)
    if not pd_ok:
        return x, v, g, H, n
    lam2 = float(g @ np.linalg.solve(Hs, g))
    for _ in range(POLISH_MAX):
        if lam2 <= 1e-26:
            break
        p = np.linalg.solve(Hs, g)
        try:
            vt, gt, Ht = vgh(x + p)
        except EvalFail:
            break
        Hts = 0.5 * (Ht + Ht.T)
        okt, _, _ = chol_logdet(Hts)
        if not okt:
            break
        lam2t = float(gt @ np.linalg.solve(Hts, gt))
        if not lam2t < lam2:
            break
        x, v, g, H, Hs, lam2 = x + p, vt, gt, Ht, Hts, lam2t
        n += 1
    return x, v, g, H, n


def newton_max3(vgh: Callable, x0, gtol: float, max_iter: int, step_cap: float) -> NewtonResult3:
    """Armijo phase as v2 (modified Newton, rejected trials never accepted, negative-curvature escape, SADDLE),
    WITHOUT the absolute-gradient floor acceptance; then polish; then the certificate decides."""
    x = np.asarray(x0, float).copy()
    try:
        v, g, H = vgh(x)
    except EvalFail as exc:
        return NewtonResult3("START_INVALID", x, float("nan"), float("nan"), float("nan"), 0, message=str(exc))
    n_mod = n_esc = n_rej = 0
    exit_ = "MAXITER"
    it = 0
    for it in range(max_iter + 1):
        gi = float(np.max(np.abs(g)))
        Hs = 0.5 * (H + H.T)
        pd_ok, _, _ = chol_logdet(Hs)
        if gi < gtol:
            if pd_ok:
                exit_ = "GTOL"
                break
            ev, V = np.linalg.eigh(Hs)
            d = V[:, 0]
            moved = False
            for a in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 1e-3):
                for sgn in (1.0, -1.0):
                    xt = x + sgn * a * d * (step_cap if step_cap < 1 else 1.0)
                    try:
                        vt, gt, Ht = vgh(xt)
                    except EvalFail:
                        n_rej += 1
                        continue
                    if np.isfinite(vt) and vt > v + 1e-12 * max(1.0, abs(v)):
                        x, v, g, H, moved = xt, vt, gt, Ht, True
                        break
                if moved:
                    break
            if not moved:
                return NewtonResult3("SADDLE", x, v, gi, float(ev[0]), it, n_modified=n_mod, n_negcurv_escape=n_esc,
                                     n_rejected_trials=n_rej)
            n_esc += 1
            continue
        if it == max_iter:
            break
        if pd_ok:
            p = np.linalg.solve(Hs, g)
            if 0.5 * float(g @ p) <= RESOLUTION_ULPS * np.spacing(abs(v)) and float(np.max(np.abs(p))) <= step_cap:
                exit_ = "RESOLUTION"             # predicted gain below the resolution of phi: Armijo uninformative
                break
        else:
            ev, V = np.linalg.eigh(Hs)
            floor = EIG_FLOOR_REL * max(1.0, float(np.max(np.abs(ev))))
            p = V @ ((V.T @ g) / np.maximum(np.abs(ev), floor))
            n_mod += 1
        slope = float(g @ p)
        if not slope > 0:
            p, slope = g.copy(), float(g @ g)
            n_mod += 1
        cap = float(np.max(np.abs(p)))
        if cap > step_cap:
            p *= step_cap / cap
            slope = float(g @ p)
        a, accepted = 1.0, False
        while a >= MIN_STEP:
            xt = x + a * p
            try:
                vt, gt, Ht = vgh(xt)
                ok = np.isfinite(vt) and vt >= v + ARMIJO_C * a * slope
            except EvalFail:
                ok = False
            if ok:
                x, v, g, H, accepted = xt, vt, gt, Ht, True
                break
            n_rej += 1
            a *= 0.5
        if not accepted:
            if pd_ok:
                exit_ = "LINESEARCH_RESOLUTION"     # decided by the certificate after polishing
                break
            return NewtonResult3("LINESEARCH_FAIL", x, v, gi, float("nan"), it, n_modified=n_mod,
                                 n_negcurv_escape=n_esc, n_rejected_trials=n_rej, phase_exit="LINESEARCH")
    x, v, g, H, npol = polish(vgh, x, v, g, H)
    ok, info = certify(vgh, x, v, g, H)
    gi = float(np.max(np.abs(g)))
    if ok:
        st = "OK"
    elif exit_ == "MAXITER":
        st = "NOT_CONVERGED"
    else:
        st = "NOT_CERTIFIED"
    return NewtonResult3(st, x, v, gi, info.get("min_eig", float("nan")), it, info.get("lam2_half", float("nan")),
                         info.get("lap_step_change", float("nan")), npol, n_mod, n_esc, n_rej, exit_,
                         info.get("logdet", float("nan")), info.get("why", ""))


# --------------------------------------------------------------------------
# deterministic S(f), T(f)
# --------------------------------------------------------------------------

@dataclass
class SVal3:
    ok: bool
    S: float
    Z: np.ndarray | None
    status: str
    min_eig: float = float("nan")
    iters: int = 0
    lam2_half: float = float("nan")
    lap_step_change: float = float("nan")
    phi_last: float = float("nan")
    grad_inf: float = float("nan")
    message: str = ""


class Local3:
    """S(f): certified Laplace over Z from the anchor Zmode (cold start, deterministic), cached by exact f."""

    def __init__(self, J: "a8.JointU", Z_ref: np.ndarray):
        self.J, self.Z_ref = J, np.asarray(Z_ref, float).copy()
        self.cache: dict[bytes, SVal3] = {}
        self.n_eval = 0

    _vgh_Z = v2.Local._vgh_Z

    def solve_from(self, f, Z0) -> SVal3:
        f = np.asarray(f, float)
        res = newton_max3(self._vgh_Z(f), np.asarray(Z0, float).ravel(), INNER_GTOL, INNER_MAXITER, INNER_STEP_CAP)
        if res.status != "OK":
            return SVal3(False, float("nan"), None, "S_" + res.status, res.min_eig, res.iters, res.lam2_half,
                         res.lap_step_change, res.value, res.grad_inf, res.message)
        S = res.value + 0.5 * N * K * LOG2PI - 0.5 * res.logdet
        return SVal3(True, S, res.x.reshape(N, K), "OK", res.min_eig, res.iters, res.lam2_half, res.lap_step_change,
                     res.value, res.grad_inf)

    def S(self, f) -> SVal3:
        f = np.asarray(f, float)
        key = f.tobytes()
        if key not in self.cache:
            self.n_eval += 1
            self.cache[key] = self.solve_from(f, self.Z_ref)
        return self.cache[key]

    def T(self, f) -> tuple[bool, float]:
        s = self.S(f)
        if not s.ok:
            return False, float("nan")
        return True, s.S + a8.log_prior(f, self.J.pv)


# --------------------------------------------------------------------------
# outer: v2 BFGS (unchanged) followed by a certified outer Newton stage
# --------------------------------------------------------------------------

def certify_outer(L: Local3, f0) -> dict:
    f = np.asarray(f0, float).copy()
    ok, t = L.T(f)
    if not ok:
        return {"status": "START_INVALID", "f": f, "T": float("nan")}
    moves = 0
    for it in range(OUTER_CERT_MAXITER):
        try:
            g = v2.fd_grad(L, f)
            Hm = v2.fd_hess(L, f)
        except EvalFail as exc:
            return {"status": "FD_FAIL", "f": f, "T": t, "message": str(exc), "moves": moves}
        ev = np.linalg.eigvalsh(Hm)
        if not ev[0] > 0:
            return {"status": "T_STATIONARY_NOT_MAX", "f": f, "T": t, "min_eig_f": float(ev[0]), "moves": moves}
        p = np.linalg.solve(Hm, g)
        dec = 0.5 * float(g @ p)
        if dec <= OUTER_CERT_DEC:
            okp, tp = L.T(f + p)
            gain = (tp - t) if okp else float("nan")
            if okp and gain > OUTER_CERT_DEC:      # the quadratic model under-states the gap: move and re-certify
                f, t, moves = f + p, tp, moves + 1
                continue
            return {"status": "CERT_OK", "f": f, "T": t, "hess_f": Hm, "min_eig_f": float(ev[0]),
                    "cond_f": float(ev[-1] / ev[0]), "lam2_half_f": dec, "newton_gain_f": gain,
                    "grad_inf": float(np.max(np.abs(g))), "moves": moves}
        nrm = float(np.linalg.norm(p))
        if nrm > OUTER_STEP_CAP:
            p *= OUTER_STEP_CAP / nrm
        a, acc = 1.0, False
        while a >= 1e-6:
            okt, tt = L.T(f + a * p)
            if okt and tt >= t + ARMIJO_C * a * float(g @ p):
                f, t, acc, moves = f + a * p, tt, True, moves + 1
                break
            a *= 0.5
        if not acc:
            return {"status": "OUTER_NOT_CERTIFIED", "f": f, "T": t, "lam2_half_f": dec, "moves": moves}
    return {"status": "OUTER_NOT_CERTIFIED", "f": f, "T": t, "moves": moves}


def method_D_E2(L: Local3, starts: dict) -> dict:
    runs = {}
    for name, f0 in starts.items():
        b = v2.maximise_T(L, f0)                       # unchanged BFGS phase (its status is recorded)
        if b["status"] in ("START_INVALID", "FD_FAIL"):
            runs[name] = {"status": "BFGS_" + b["status"], "valid": False, "bfgs_status": b["status"]}
            continue
        c = certify_outer(L, b["f"])
        c["bfgs_status"], c["valid"] = b["status"], c["status"] == "CERT_OK"
        runs[name] = c
    cl = v2.classify_starts(runs, "T", "f")
    out = {"D_" + k: v for k, v in cl.items()}
    out["D_start_status"] = "|".join(f"{k}:{r['status']}" for k, r in runs.items())
    out["D_bfgs_status"] = "|".join(f"{k}:{r.get('bfgs_status')}" for k, r in runs.items())
    if cl["best"] is None:
        out["E2_status"] = "NO_VALID_LOCAL_MAX"
        return out
    b = runs[cl["best"]]
    f, Hm = b["f"], b["hess_f"]
    s = L.S(f)
    ld = float(np.linalg.slogdet(Hm)[1])
    out.update({"D": b["T"], "f_D": f, "Z_D": s.Z, "D_S_at_fD": s.S, "D_logprior_at_fD": a8.log_prior(f, L.J.pv),
                "D_grad_inf": b["grad_inf"], "D_min_eig_f": b["min_eig_f"], "D_cond_f": b["cond_f"],
                "D_lam2_half_f": b["lam2_half_f"], "D_newton_gain_f": b["newton_gain_f"], "D_cert_moves": b["moves"],
                "E2": b["T"] + 0.5 * K * LOG2PI - 0.5 * ld, "E2_logdet_f": ld, "hess_f": Hm})
    for h in FD_H_HESS_ALT:
        try:
            ev = np.linalg.eigvalsh(v2.fd_hess(L, f, h))
            out[f"E2_h{h:g}"] = b["T"] + 0.5 * K * LOG2PI - 0.5 * float(np.sum(np.log(ev))) if ev[0] > 0 else float("nan")
        except EvalFail:
            out[f"E2_h{h:g}"] = float("nan")
    out["E2_status"] = "MULTIMODAL_NONNEGLIGIBLE" if cl["multistart"] == "MULTIMODAL_NONNEGLIGIBLE" else "OK"
    return out


def outer_checks(L: Local3, f_D, T_D, hess_f) -> dict:
    """main prior only: (i) FD-gradient spread over steps h and the implied objective gap; (ii) Nelder-Mead from f_D
    (derivative-free; invalid T points are infeasible for the simplex and counted)."""
    from scipy.optimize import minimize
    out = {}
    gs = {}
    for h in (5e-5, 1e-4, 2e-4):
        try:
            gs[h] = v2.fd_grad(L, f_D, h)
        except EvalFail:
            gs[h] = None
    if all(g is not None for g in gs.values()):
        arr = np.array(list(gs.values()))
        spread = float(np.max(np.abs(arr - arr.mean(0))) * 2)
        lam_min = float(np.linalg.eigvalsh(hess_f)[0])
        out.update(fd_grad_spread=spread, fd_noise_implied_gap=spread ** 2 / (2 * lam_min),
                   **{f"fd_grad_inf_h{h:g}": float(np.max(np.abs(g))) for h, g in gs.items()})
    else:
        out.update(fd_grad_spread=float("nan"), fd_noise_implied_gap=float("nan"))
    n_bad = [0]

    def negT(f):
        ok, t = L.T(np.asarray(f, float))
        if not ok:
            n_bad[0] += 1
            return np.inf
        return -t
    r = minimize(negT, f_D, method="Nelder-Mead",
                 options={"xatol": 1e-7, "fatol": 1e-10, "maxfev": NM_MAXFEV, "initial_simplex": f_D + 0.05 * np.vstack([np.zeros(K), np.eye(K)])})
    out.update(nm_gain_over_fD=float(-r.fun - T_D), nm_f_dist=float(np.max(np.abs(r.x - f_D))),
               nm_nfev=int(r.nfev), nm_invalid_evals=n_bad[0], nm_success=bool(r.success))
    return out


def method_E(J: "a8.JointU", Z_ref, starts_f: dict, Z_starts: dict) -> dict:
    def vgh(u):
        Z, f = u[:-K].reshape(N, K), u[-K:]
        v = safe(J.psi, Z, f)
        gZ, gf = safe(J.grad_u, Z, f)
        H = safe(J.negH_u, Z, f)
        g = np.concatenate([gZ.ravel(), gf])
        if not (np.isfinite(v) and np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
            raise EvalFail("non-finite psi/grad/H")
        return v, g, H
    runs = {}
    for name, f0 in starts_f.items():
        Z0 = Z_starts.get(name, Z_ref)
        res = newton_max3(vgh, np.concatenate([np.asarray(Z0, float).ravel(), np.asarray(f0, float)]),
                          JOINT_GTOL, JOINT_MAXITER, INNER_STEP_CAP)
        runs[name] = {"status": res.status, "valid": res.status == "OK", "psi": res.value, "u": res.x,
                      "f": res.x[-K:], "min_eig": res.min_eig, "logdet": res.logdet, "lam2_half": res.lam2_half,
                      "lap_step_change": res.lap_step_change}
    cl = v2.classify_starts(runs, "psi", "f")
    out = {"E_" + k: v for k, v in cl.items()}
    out["E_start_status"] = "|".join(f"{k}:{r['status']}" for k, r in runs.items())
    if cl["best"] is None:
        out["E_status"] = "NO_VALID_LOCAL_MAX"
        return out
    b = runs[cl["best"]]
    Z, f = b["u"][:-K].reshape(N, K), b["f"]
    H = J.negH_u(Z, f)
    nk = N * K
    HZZ = 0.5 * (H[:nk, :nk] + H[:nk, :nk].T)
    schur = H[nk:, nk:] - H[nk:, :nk] @ np.linalg.solve(HZZ, H[:nk, nk:])
    out.update({"E": b["psi"] + 0.5 * DU * LOG2PI - 0.5 * b["logdet"], "f_E": f, "E_min_eig": b["min_eig"],
                "E_logdet": b["logdet"], "E_lam2_half": b["lam2_half"], "E_lap_step_change": b["lap_step_change"],
                "E_logdet_ZZ": float(np.linalg.slogdet(HZZ)[1]),
                "E_logdet_schur_f": float(np.linalg.slogdet(0.5 * (schur + schur.T))[1]),
                "E_status": "MULTIMODAL_NONNEGLIGIBLE" if cl["multistart"] == "MULTIMODAL_NONNEGLIGIBLE" else "OK"})
    return out


def method_IS(L: Local3, f_D, hess_f, Z_D, seed, draws=IS_DRAWS) -> dict:
    """same proposal, seed and draws as v2.  Per draw: S from the cold start (definition of S).  If that fails
    certification, ONE deterministic additional mode search from Z(f_D) is made; it replaces the draw's value only
    if certified AND the cold start's last phi is within RESCUE_SAME_MODE_TOL of it (same mode, optimiser-only
    failure).  Otherwise the draw is FAILED (not weight 0) and the IS estimate is invalid.  Every rescue is logged."""
    from scipy.stats import multivariate_t
    q = multivariate_t(loc=f_D, shape=np.linalg.inv(hess_f), df=IS_DF, seed=np.random.default_rng(seed))
    fs = np.atleast_2d(q.rvs(size=draws))
    lw = np.full(draws, np.nan)
    cat = np.empty(draws, dtype=object)
    log: list[dict] = []
    for s in range(draws):
        sv = L.S(fs[s])
        if sv.ok:
            cat[s] = "finite"
            lw[s] = sv.S + a8.log_prior(fs[s], L.J.pv) - q.logpdf(fs[s])
            continue
        w = L.solve_from(fs[s], Z_D)
        same = w.ok and np.isfinite(sv.phi_last) and abs(sv.phi_last - w.phi_last) <= RESCUE_SAME_MODE_TOL
        entry = {"draw": s, "f": "|".join(repr(float(a)) for a in fs[s]), "cold_status": sv.status,
                 "cold_grad_inf": sv.grad_inf, "cold_lam2_half": sv.lam2_half, "cold_phi_last": sv.phi_last,
                 "warm_status": w.status, "warm_phi": w.phi_last, "phi_diff": (w.phi_last - sv.phi_last)}
        if same:
            cat[s] = "rescued"
            lw[s] = w.S + a8.log_prior(fs[s], L.J.pv) - q.logpdf(fs[s])
            entry["outcome"] = "RESCUED_SAME_MODE"
        else:
            cat[s] = "failed"
            entry["outcome"] = "FAILED"
        log.append(entry)
    n_fin = int((cat == "finite").sum())
    n_res = int((cat == "rescued").sum())
    n_fail = int((cat == "failed").sum())
    res = {"IS_draws": draws, "IS_finite": n_fin, "IS_rescued": n_res, "IS_zero_weight": 0, "IS_failed": n_fail,
           "IS_draw_log": json.dumps(log)}
    use = (cat == "finite") | (cat == "rescued")
    if not use.any():
        res.update(IS=float("nan"), IS_status="ALL_FAILED")
        return res
    fin = lw[use]
    est = float(logsumexp(fin) - math.log(draws))
    w_ = np.zeros(draws)
    w_[use] = np.exp(fin - est)
    se = float(np.std(w_, ddof=1) / math.sqrt(draws))
    wf = np.exp(fin - fin.max())
    res.update(IS=est if n_fail == 0 else float("nan"), IS_lower_bound_if_failed=est, IS_mc_se=se,
               IS_ess=float(wf.sum() ** 2 / np.sum(wf ** 2)),
               IS_status="OK" if n_fail == 0 else "PARTIAL_FAIL")
    return res


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------

def make_starts(case, j, cand):
    Za = np.asarray(case["st"]["Zmode"], float)
    starts = {"plugin_anchorZ": a8.fixed_z_start(cand, case["X"][:, j], Za), "zero": np.zeros(K)}
    fl = a8.af6_fstar(case["label"], case["regime"], j, cand)
    if fl is not None:
        starts["af6_fL"] = fl
    return Za, starts


def run_cell(case, j, cand, prior_name, case_index, main: bool) -> dict:
    t0 = time.perf_counter()
    J, v = a8.joint_for(case, j, cand, prior_name)
    Za, starts = make_starts(case, j, cand)
    L = Local3(J, Za)
    row: dict[str, Any] = {"replicate": case["label"], "regime": case["regime"], "column": j, "candidate": cand,
                           "prior": prior_name, "prior_var_f": v}
    try:
        de = method_D_E2(L, starts)
        hess_f, Z_D, f_D = de.pop("hess_f", None), de.pop("Z_D", None), de.pop("f_D", None)
        row.update(de)
        if f_D is not None:
            row["f_D"] = "|".join(repr(float(a)) for a in f_D)
            row["fD_norm"] = float(np.linalg.norm(f_D))
        e_starts, z_starts = dict(starts), {}
        if f_D is not None:
            e_starts["f_D"], z_starts["f_D"] = f_D, Z_D
        J2, _ = a8.joint_for(case, j, cand, prior_name)
        e = method_E(J2, Za, e_starts, z_starts)
        f_E = e.pop("f_E", None)
        row.update(e)
        if f_E is not None:
            row["f_E"] = "|".join(f"{a:.10g}" for a in f_E)
        if main and f_D is not None and row.get("E2_status") in ("OK", "MULTIMODAL_NONNEGLIGIBLE"):
            L2 = Local3(a8.joint_for(case, j, cand, prior_name)[0], Za)
            L2.T(f_D + 0.3)
            L2.T(np.zeros(K))
            ok2, t2 = L2.T(f_D)
            row["determinism_abs_diff"] = abs(t2 - row["D"]) if ok2 else float("nan")
            row.update(outer_checks(L, f_D, row["D"], hess_f))
            row.update(method_IS(L, f_D, hess_f, Z_D, IS_SEED_BASE + case_index))
        row["status"] = "OK" if row.get("E2_status") == "OK" else ("FLAGGED" if row.get("E2_status") else "FAILED")
    except Exception as exc:                         # unexpected: recorded, never retried
        row.update({"status": "EXCEPTION", "error": f"{type(exc).__name__}: {exc}"[:300]})
    row["n_S_evals"] = L.n_eval
    row["n_S_not_ok"] = int(sum(not s.ok for s in L.cache.values()))
    row["seconds"] = round(time.perf_counter() - t0, 2)
    return row


selection_state = v2.selection_state      # unchanged (label-free; not evaluated against labels in the pilot)


# --------------------------------------------------------------------------
# protocol / runinfo / drivers
# --------------------------------------------------------------------------

def protocol() -> dict[str, Any]:
    return {
        "experiment": "AF8-v3",
        "study_type": "exploratory numerical-stability revision + pilot (no selection accuracy, no main experiment). Written AFTER seeing the AF8-v2 pilot numerics; not the AF8 pre-registration",
        "parent": {"AF8_v2_head": V2_HEAD,
                   "AF8_v2_protocol_sha256": "57f0a37935eed694aff639728310101d3dedc6709dcc23b28423882fed6fb777",
                   "AF8_v2_stop": "S4 fired: 9/72 main-prior cells had one failed IS draw (9/28,400 draws). Diagnosis at the exact draws: inner Newton stall with ||g||_inf 1e-6..9e-6 because the Armijo test is decided by floating-point rounding of phi (Newton decrement/2 ~ 1e-14 < 1 ulp of |phi|); 2 cells without a valid outer start because the outer FD gradient carried noise ~1e-5..7e-4 > OUTER_GTOL 1e-5, consistent with O(1e-8) Laplace-value errors of floor-accepted inner points divided by h"},
        "unchanged_from_v2": ["model, theta_{-j}, data, masks, priors (main B_sg2_0.667 + 8)", "log density/gradient/negative Hessian (AF8 JointU)",
                              "starts (plugin_anchorZ, zero, af6_fL; E also from f_D)", "Armijo phase of Newton (aim tolerances inner 1e-9 / joint 1e-8, max iterations 100 / 300, step caps, modified Newton, negative-curvature escape, SADDLE)",
                              "outer BFGS phase (v2.maximise_T)", "FD steps 1e-4 / 1e-3, alternative Hessian steps 5e-4 / 2e-3",
                              "IS proposal (t5 at f_D, inverse FD Hessian), 400 draws, seeds 1689500 + case index (identical draws if f_D identical)",
                              "multistart clustering (same point ||df||_inf <= 1e-3, value spread <= 1e-6, multimodal gap log 20)",
                              "selection states and the 0.01 decision floor", "E2 is the main score; E and IS diagnostics"],
        "changed_numerics": {
            "inner_and_joint_acceptance": f"absolute-gradient floor acceptance (OK_FLOOR: ||g||_inf<=1e-6) REMOVED. After the Armijo phase (ending at gtol, when negH is PD and the predicted gain lambda^2/2 is <= {RESOLUTION_ULPS} ulp of |phi| so that the Armijo comparison of phi values is uninformative, at a line-search failure with negH PD, or at max iterations), up to {POLISH_MAX} pure Newton steps are taken, each accepted only if the trial is finite, negH stays PD (Cholesky) and the Newton decrement strictly decreases (no phi comparison: phi cannot resolve the change). The point is ACCEPTED only with a certificate: negH PD, lambda^2/2 = g^T negH^-1 g / 2 <= {CERT_DEC}, and |Lap(x+p) - Lap(x)| <= {CERT_LAP} where p is the full Newton step and Lap = phi - 1/2 log|negH|. Otherwise NOT_CERTIFIED / NOT_CONVERGED / LINESEARCH_FAIL / SADDLE (a failure, never a value).",
            "outer_acceptance": f"OUTER_GTOL (FD ||g||_inf < 1e-5) is no longer the acceptance criterion. After the v2 BFGS phase (whatever its status, if the point is valid), an outer Newton stage with FD gradient (h=1e-4) and FD Hessian (h=1e-3): if FD Hessian of -T not PD -> T_STATIONARY_NOT_MAX; lambda_f^2/2 > {OUTER_CERT_DEC} -> Armijo Newton step; lambda_f^2/2 <= {OUTER_CERT_DEC} -> evaluate T(f + H^-1 g): if the gain > {OUTER_CERT_DEC} move and repeat, else CERT_OK (at most {OUTER_CERT_MAXITER} rounds). A start is valid only with CERT_OK.",
            "IS": f"same draws as v2. A draw whose cold-start S is not certified gets ONE deterministic additional mode search from Z(f_D); its value is used only if certified and the cold start's last phi is within {RESCUE_SAME_MODE_TOL} (same mode). Otherwise the draw is FAILED: the IS estimate is invalid (NaN; lower bound reported). Rescued draws are logged (draw, f, original status, gradient, decrement, phi difference, outcome) and counted separately (S4r). Zero-weight draws: T is finite wherever S is defined for this model, so the category exists but cannot occur; it is never used for failures.",
            "diagnostics_added": "per certified point: lambda^2/2, Laplace step change, min eigenvalue, condition number of the outer FD Hessian; main prior: FD gradient at h in {5e-5,1e-4,2e-4} and the implied gap spread^2/(2 lambda_min); Nelder-Mead (derivative-free) from f_D with gain over T(f_D)",
        },
        "certificate_rationale": {
            "TAU_E2": f"{TAU_E2} = (decision floor {NUMERIC_FLOOR}) / 10: total numerical error budget per candidate E2",
            "CERT_DEC": f"{CERT_DEC}: phi-gap of the local quadratic model at the Z mode; contributes <= {CERT_DEC} to S",
            "CERT_LAP": f"{CERT_LAP}: S error is first order in dZ = H^-1 g through log|H| (observed 3.9e-8 at a v2 floor point). An S error eps produces FD-gradient noise eps/h = {CERT_LAP / FD_H_GRAD:g} at h=1e-4 and FD-Hessian noise ~4 eps/h^2 = {4 * CERT_LAP / FD_H_HESS ** 2:g} at h=1e-3, i.e. a log-det error <= K * 4e-4 / lambda_min(outer) ~ 2e-4 for lambda_min >= 6 (v2 minimum 6.28) -> E2 error 1e-4 < TAU_E2",
            "OUTER_CERT_DEC": f"{OUTER_CERT_DEC}: T-gap at f_D (quadratic model and actual extra Newton step) enters E2 directly; 1e-6 << TAU_E2. Gradient noise 1e-6 (from CERT_LAP) implies a gap ~ (1e-6)^2/(2*6) ~ 1e-13, so the certificate is not noise-limited",
            "S8": f"Nelder-Mead gain over T(f_D) <= TAU_E2 = {TAU_E2}: an FD-free optimiser must not find a higher T by more than the E2 error budget",
            "S9": f"FD-noise implied gap <= OUTER_CERT_DEC: the FD gradient used for the certificate must be resolved at the level of the certificate",
            "limits": "lambda^2 and the extra-step checks are LOCAL (quadratic model at a PD point). They do not establish a global maximum of a non-concave problem; global questions are addressed only by multistart (f, E) and are not addressed for the Z problem beyond the deterministic cold start and the logged rescue.",
        },
        "multistart_success": "a cell's E2 is valid if >= 1 start is CERT_OK and the certified starts form one cluster (AGREE) or a dominant one; S2b additionally limits cells with < 2 certified starts; invalid starts are counted with their status",
        "stop_rules_v3": STOP_RULES_V3,
        "stop_rules_changes_vs_v2": {
            "S1, S2, S3, S3b, S4b, S5, S6, S7": "same definitions and thresholds",
            "S4": "same threshold and definition (cells with >= 1 failed draw); with v3 a draw is failed if neither the cold start nor the pre-registered same-mode rescue is certified. The v2-definition value (any cold-start failure, i.e. rescued draws counted as failed) is reported as S4_v2_definition AND bounded separately by S4r (same 5%)",
            "new": "S2b (< 2 certified starts), S4r (rescues), S8 (Nelder-Mead), S9 (FD noise). No threshold relaxed",
        },
        "units": "rules S1, S2, S2b are over the 648 cells (cell = dataset x regime x column x candidate x prior); S3/S3b over main-prior candidate pairs with both IS valid (the number excluded is reported); S4, S4r, S4b, S5, S8, S9 over the 72 main-prior cells. The dataset (3) is the statistical unit of any later inference; the pilot rules are numerical acceptance criteria, not inference",
        "regression_first": "before the pilot, the 9 known failed IS draws (exact f regenerated from the v2 seeds) and the 2 cells without a valid outer start are re-evaluated as regression tests",
        "post_hoc": "anything computed after the pilot that is not listed here is labelled exploratory/post-hoc; thresholds are not changed after the pilot",
        "pilot": "rep01-rep03, 648 cells (same as AF8 / v2); numerical quantities only; no selection accuracy, AUC or label summaries",
        "independent_audit": "read-only research-auditor after the pilot",
        "not_claimed": ["selection accuracy", "improvement of family selection", "Bayesian evidence of the whole model",
                        "E2 ~ IS implies closeness to the exact marginal over (Z, f)", "confirmation"],
    }


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def runinfo(extra):
    import scipy
    return {"experiment": "AF8-v3", "script": "expfam/src/experimental/run_af8v3_integrated_loading.py",
            "git_head": a8.git("rev-parse", "HEAD"), "git_branch": a8.git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_tracked": bool(a8.git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(), "python_executable": sys.executable,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "protocol_sha256": protocol_sha256(), "command": " ".join(sys.argv), **extra}


def case_index(r, g, j, cand):
    return (r * 2 + REGIMES.index(g)) * 100 + j * 2 + CANDS.index(cand)      # identical to v2


def run(stage, reps, shard=None):
    out = OUT / stage
    out.mkdir(parents=True, exist_ok=True)
    items = list(a8.cells(reps))
    keys = sorted({(r, g) for r, g, *_ in items})
    tag = ""
    if shard:
        i, n = (int(v) for v in shard.split("/"))
        keep = set(keys[i::n])
        items = [c for c in items if (c[0], c[1]) in keep]
        tag = f"_{i}of{n}"
    (out / f"runinfo{tag}.json").write_text(json.dumps(runinfo({"stage": stage, "shard": shard}), indent=2), encoding="utf-8")
    path = out / f"progress{tag}.jsonl"
    done = set()
    if path.exists():
        done = {(d["replicate"], d["regime"], d["column"], d["candidate"], d["prior"])
                for d in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    case, t0 = None, time.time()
    for r, g, j, cand, pr in items:
        label = f"rep{r + 1:02d}"
        if (label, g, j, cand, pr) in done:
            continue
        if case is None or case["label"] != label or case["regime"] != g:
            case = a8.load_case(r, g)
            case["regime"] = g
        row = run_cell(case, j, cand, pr, case_index(r, g, j, cand), main=(pr == MAIN_PRIOR))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        print(f"  {label} {g} j={j} {cand[:4]} {pr} {row['status']} {row['seconds']}s total {time.time() - t0:.0f}s", flush=True)


def collect(stage):
    rows = []
    for p in sorted((OUT / stage).glob("progress*.jsonl")):
        rows += [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()]
    c = pd.DataFrame(rows)
    c.to_csv(OUT / stage / "candidate_scores.csv", index=False)
    return c


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    sh = argv[argv.index("--shard") + 1] if "--shard" in argv else None
    if "--write-protocol" in argv:
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / "protocol.json"
        if p.exists():
            raise SystemExit("protocol.json exists; refusing to overwrite")
        p.write_text(json.dumps(protocol(), indent=2, ensure_ascii=False), encoding="utf-8")
        print("protocol sha256", protocol_sha256())
    if "--pilot" in argv:
        p = OUT / "protocol.json"
        if not p.exists() or json.loads(p.read_text(encoding="utf-8")) != json.loads(json.dumps(protocol(), ensure_ascii=False)):
            raise SystemExit("frozen protocol.json missing or differs from code; refusing to run")
        run("pilot", PILOT_REPS, sh)
    if "--collect-pilot" in argv:
        collect("pilot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
