"""AF8-v2 (exploratory, numerical repair): loading-integrated family selection with safeguarded optimisation.

AF8 (0643fa5) stopped at its pilot because of optimiser defects (saddle convergence of the joint Newton,
non-finite trial points propagated after a failed line search, LinAlgError escaping from S during BFGS,
sentinel values counted as start disagreement; IS denominator after dropping non-finite draws).  AF8 files
are NOT modified; this module re-uses only its verified mathematics (log density, gradient, full negative
Hessian incl. Z-f cross blocks and prior, CMP2 Bessel normaliser, priors) and replaces every optimiser,
failure path and the IS estimator.

Quantities (theta_{-j} = AF6 auto_first anchor, fixed; m in {poisson, cmp_nu2p0}):
  S_m(f)  = Laplace over Z at fixed f, mode found DETERMINISTICALLY from Z_ref = anchor Zmode
  T_m(f)  = S_m(f) + log p(f | m)
  E2_m    = T_m(f_D) + (K/2) log 2pi - 1/2 log|-d2T/df2 (f_D)|      (MAIN: nested Laplace)
  E_m     = Psi(u_hat) + (228/2) log 2pi - 1/2 log|H_uu|            (diagnostic: joint Laplace)
  IS_m    = log mean_s exp(T(f_s) - log q(f_s)), all draws in the denominator (diagnostic of the outer f integral)

No MCEM, no new data, no selection accuracy in the pilot.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_af8_integrated_loading as a8          # noqa: E402  (math only; AF8 file unchanged)

ROOT = a8.ROOT
OUT = a8.RES / "af8v2_numerical_repair_20261115"
N, K, DU = a8.N, a8.K, a8.DU
LOG2PI = a8.LOG2PI
P, C = a8.P, a8.C
CANDS, COUNT_COLS, REGIMES = a8.CANDS, a8.COUNT_COLS, a8.REGIMES
PRIORS, MAIN_PRIOR = a8.PRIORS, a8.MAIN_PRIOR          # unchanged from AF8
AF8_HEAD = "0643fa550c78127cd02d56886330df64ed9c644d"

# ---------------- frozen numerical settings (v2) ----------------
INNER_GTOL = 1e-9            # Z-mode gradient aim (inf-norm), as AF8
JOINT_GTOL = 1e-8            # joint u-mode aim, as AF8
ACCEPT_GTOL = 1e-6           # accepted at the floating-point floor (AF5/AF8: aim 1e-9, accept 1e-6)
STALL_ITERS = 3              # floor = this many consecutive accepted steps with |dv| <= 1e-12 |v|
OUTER_GTOL = 1e-5            # f-optimisation FD gradient (inf-norm), as AF8
OUTER_MAXITER = 200
INNER_MAXITER = 100
JOINT_MAXITER = 300
ARMIJO_C = 1e-4
MIN_STEP = 1e-12
OUTER_STEP_CAP = 0.5         # max ||f step||_2 per outer iteration
INNER_STEP_CAP = 2.0         # max |Z step|_inf per Newton iteration
FD_H_GRAD = 1e-4
FD_H_HESS = 1e-3
FD_H_HESS_ALT = (5e-4, 2e-3)
EIG_FLOOR_REL = 1e-8         # modified Newton: |lambda| floored at EIG_FLOOR_REL * max|lambda|
SAME_POINT_F_TOL = 1e-3      # two valid starts are the same local maximum if ||f_a - f_b||_inf <= this
SAME_POINT_VALUE_TOL = 1e-6  # ... and their objective values differ by <= this
MULTIMODAL_GAP = math.log(20.0)   # second valid local max within this of the best -> single-mode Laplace questionable
IS_DRAWS, IS_DF = 400, 5
IS_SEED_BASE = 1689500        # v2 IS seeds (AF8 used 1688000); disjoint
NUMERIC_FLOOR = 1e-2
PILOT_REPS = (0, 1, 2)

STOP_RULES_V2 = {
    "S1_cell_numerical_failure_share_max": 0.05,
    "S2_multistart_disagreement_share_max": 0.05,
    "S3_main_median_abs_DeltaE2_minus_DeltaIS_max": 0.25,
    "S3b_main_median_abs_DeltaE2_minus_DeltaIS_below_half_median_abs_DeltaE2": True,
    "S4_IS_cells_with_failed_draws_share_max": 0.05,
    "S4b_IS_min_ess": 40.0,
    "S5_determinism_max_abs_diff": 1e-8,
    "S6_fd_step_sensitivity_median_abs_DeltaE2_max": 0.05,
    "S7_fd_derivative_rel_err_max": 1e-4,
}


# --------------------------------------------------------------------------
# safe evaluation
# --------------------------------------------------------------------------

class EvalFail(Exception):
    """numerical failure of an evaluation (non-finite value, overflow, factorisation failure)."""


def safe(fn: Callable, *a):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            out = fn(*a)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError, ValueError, ZeroDivisionError) as exc:
        raise EvalFail(f"{type(exc).__name__}: {exc}") from exc
    return out


def chol_logdet(H):
    """(is_pd, logdet, min_eig) of a symmetric matrix; min_eig only computed when not PD."""
    Hs = 0.5 * (H + H.T)
    try:
        L = np.linalg.cholesky(Hs)
        return True, 2.0 * float(np.sum(np.log(np.diag(L)))), None
    except np.linalg.LinAlgError:
        ev = np.linalg.eigvalsh(Hs)
        return False, float("nan"), float(ev[0])


# --------------------------------------------------------------------------
# safeguarded Newton for maximisation (generic; unit-tested on toy problems)
# --------------------------------------------------------------------------

@dataclass
class NewtonResult:
    status: str                 # OK | SADDLE | LINESEARCH_FAIL | NOT_CONVERGED | START_INVALID
    x: np.ndarray
    value: float
    grad_inf: float
    min_eig: float
    iters: int
    n_modified: int = 0
    n_negcurv_escape: int = 0
    n_rejected_trials: int = 0
    message: str = ""


def newton_max(vgh: Callable, x0: np.ndarray, gtol: float, max_iter: int, step_cap: float,
               accept_gtol: float = ACCEPT_GTOL) -> NewtonResult:
    """maximise phi; vgh(x) -> (phi, grad, negH) or raises EvalFail.
    - modified Newton: negH replaced by V |Lambda|_floored V^T when not positive definite (always an ascent direction)
    - Armijo backtracking; a trial that fails to evaluate or is non-finite is REJECTED (step halved); the iterate
      never moves to a rejected point; if no step is accepted the result is LINESEARCH_FAIL at the last valid point
    - stops as OK only if ||grad||_inf < gtol AND negH is positive definite (Cholesky); a stationary point with an
      indefinite negH is escaped along the most negative curvature direction, and reported SADDLE if that fails.
    - floating-point floor: if the objective stops changing (|dv| <= 1e-12|v| for STALL_ITERS accepted steps, or the
      line search cannot improve) while ||grad||_inf <= accept_gtol and negH is PD, the result is OK_FLOOR."""
    x = np.asarray(x0, float).copy()
    try:
        v, g, H = vgh(x)
    except EvalFail as exc:
        return NewtonResult("START_INVALID", x, float("nan"), float("nan"), float("nan"), 0, message=str(exc))
    n_mod = n_esc = n_rej = 0
    stall = 0
    for it in range(max_iter + 1):
        gi = float(np.max(np.abs(g)))
        Hs = 0.5 * (H + H.T)
        pd_ok, _, _ = chol_logdet(Hs)
        if pd_ok and gi <= accept_gtol and stall >= STALL_ITERS:
            return NewtonResult("OK_FLOOR", x, v, gi, float(np.linalg.eigvalsh(Hs)[0]), it, n_mod, n_esc, n_rej)
        if gi < gtol:
            if pd_ok:
                me = float(np.linalg.eigvalsh(Hs)[0])
                return NewtonResult("OK", x, v, gi, me, it, n_mod, n_esc, n_rej)
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
                return NewtonResult("SADDLE", x, v, gi, float(ev[0]), it, n_mod, n_esc, n_rej)
            n_esc += 1
            continue
        if it == max_iter:
            break
        if pd_ok:
            p = np.linalg.solve(Hs, g)
        else:
            ev, V = np.linalg.eigh(Hs)
            floor = EIG_FLOOR_REL * max(1.0, float(np.max(np.abs(ev))))
            p = V @ ((V.T @ g) / np.maximum(np.abs(ev), floor))
            n_mod += 1
        slope = float(g @ p)
        if not slope > 0:                       # numerical safety: fall back to steepest ascent
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
                stall = stall + 1 if abs(vt - v) <= 1e-12 * max(1.0, abs(v)) else 0
                x, v, g, H, accepted = xt, vt, gt, Ht, True
                break
            n_rej += 1
            a *= 0.5
        if not accepted:
            if pd_ok and gi <= accept_gtol:
                return NewtonResult("OK_FLOOR", x, v, gi, float(np.linalg.eigvalsh(Hs)[0]), it, n_mod, n_esc, n_rej)
            return NewtonResult("LINESEARCH_FAIL", x, v, gi, float("nan"), it, n_mod, n_esc, n_rej)
    return NewtonResult("NOT_CONVERGED", x, v, float(np.max(np.abs(g))), float("nan"), max_iter, n_mod, n_esc, n_rej)


# --------------------------------------------------------------------------
# deterministic S(f) and T(f)
# --------------------------------------------------------------------------

@dataclass
class SVal:
    ok: bool
    S: float
    Z: np.ndarray | None
    status: str
    min_eig: float = float("nan")
    iters: int = 0
    message: str = ""


class Local:
    """one (case, column, candidate, prior): deterministic S and T with a pure cache keyed by f."""

    def __init__(self, J: "a8.JointU", Z_ref: np.ndarray):
        self.J, self.Z_ref = J, np.asarray(Z_ref, float).copy()
        self.cache: dict[bytes, SVal] = {}
        self.n_eval = 0

    def _vgh_Z(self, f):
        J = self.J
        f = np.asarray(f, float)

        def vgh(zv):
            Z = zv.reshape(N, K)
            J.F[J.j] = f
            v = safe(J.phi, Z)
            g = safe(J.gradZ, Z).ravel()
            H = safe(J.negHZZ, Z)
            if not (np.isfinite(v) and np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
                raise EvalFail("non-finite phi/grad/H")
            return v, g, H
        return vgh

    def S(self, f) -> SVal:
        f = np.asarray(f, float)
        key = f.tobytes()
        if key in self.cache:
            return self.cache[key]
        self.n_eval += 1
        res = newton_max(self._vgh_Z(f), self.Z_ref.ravel(), INNER_GTOL, INNER_MAXITER, INNER_STEP_CAP)
        if res.status not in ("OK", "OK_FLOOR"):
            out = SVal(False, float("nan"), None, "S_" + res.status, res.min_eig, res.iters, res.message)
        else:
            Z = res.x.reshape(N, K)
            self.J.F[self.J.j] = f
            H = self.J.negHZZ(Z)
            pd_ok, ld, me = chol_logdet(H)
            if not pd_ok:
                out = SVal(False, float("nan"), None, "S_HZZ_NOT_PD", me, res.iters)
            else:
                out = SVal(True, res.value + 0.5 * N * K * LOG2PI - 0.5 * ld, Z, "OK", res.min_eig, res.iters)
        self.cache[key] = out
        return out

    def T(self, f) -> tuple[bool, float]:
        s = self.S(f)
        if not s.ok:
            return False, float("nan")
        return True, s.S + a8.log_prior(f, self.J.pv)


# --------------------------------------------------------------------------
# outer maximisation of T over f (K = 3): BFGS with FD gradient, capped steps, Armijo, invalid trials rejected
# --------------------------------------------------------------------------

def fd_grad(L: Local, f, h=FD_H_GRAD):
    g = np.zeros(K)
    for a in range(K):
        e = np.zeros(K)
        e[a] = h
        okp, tp = L.T(f + e)
        okm, tm = L.T(f - e)
        if okp and okm:
            g[a] = (tp - tm) / (2 * h)
        else:
            raise EvalFail(f"FD gradient: T invalid at f +- h e_{a}")
    return g


def fd_hess(L: Local, f, h=FD_H_HESS):
    ok0, t0 = L.T(f)
    if not ok0:
        raise EvalFail("FD Hessian: T invalid at centre")
    Hm = np.zeros((K, K))
    for a in range(K):
        for b in range(a, K):
            ea, eb = np.eye(K)[a] * h, np.eye(K)[b] * h
            if a == b:
                vals = [L.T(f + ea), L.T(f - ea)]
                if not all(v[0] for v in vals):
                    raise EvalFail("FD Hessian: T invalid")
                Hm[a, a] = -(vals[0][1] - 2 * t0 + vals[1][1]) / h ** 2
            else:
                vals = [L.T(f + ea + eb), L.T(f + ea - eb), L.T(f - ea + eb), L.T(f - ea - eb)]
                if not all(v[0] for v in vals):
                    raise EvalFail("FD Hessian: T invalid")
                Hm[a, b] = Hm[b, a] = -(vals[0][1] - vals[1][1] - vals[2][1] + vals[3][1]) / (4 * h * h)
    return Hm


def maximise_T(L: Local, f0) -> dict:
    f = np.asarray(f0, float).copy()
    ok, t = L.T(f)
    if not ok:
        return {"status": "START_INVALID", "f": f, "T": float("nan"), "iters": 0, "message": L.S(f).status}
    try:
        g = fd_grad(L, f)
    except EvalFail as exc:
        return {"status": "FD_FAIL", "f": f, "T": t, "iters": 0, "message": str(exc)}
    B = np.eye(K) * 0.1
    n_rej = 0
    for it in range(OUTER_MAXITER):
        if np.max(np.abs(g)) < OUTER_GTOL:
            return {"status": "GRAD_OK", "f": f, "T": t, "iters": it, "grad_inf": float(np.max(np.abs(g))), "n_rejected": n_rej}
        p = B @ g
        if not float(g @ p) > 0:
            B = np.eye(K) * 0.1
            p = B @ g
        nrm = float(np.linalg.norm(p))
        if nrm > OUTER_STEP_CAP:
            p *= OUTER_STEP_CAP / nrm
        slope = float(g @ p)
        a, acc = 1.0, False
        while a >= 1e-10:
            ft = f + a * p
            okt, tt = L.T(ft)
            if okt and tt >= t + ARMIJO_C * a * slope:
                try:
                    gt = fd_grad(L, ft)
                    acc = True
                    break
                except EvalFail:
                    pass
            n_rej += 1
            a *= 0.5
        if not acc:
            if not np.allclose(B, np.eye(K) * 0.1):
                B = np.eye(K) * 0.1                 # one restart along the gradient
                continue
            return {"status": "LINESEARCH_FAIL", "f": f, "T": t, "iters": it, "grad_inf": float(np.max(np.abs(g))),
                    "n_rejected": n_rej}
        s, y = ft - f, -(gt - g)                     # BFGS on -T
        sy = float(s @ y)
        if sy > 1e-12:
            rho = 1.0 / sy
            I = np.eye(K)
            B = (I - rho * np.outer(s, y)) @ B @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        f, t, g = ft, tt, gt
    return {"status": "NOT_CONVERGED", "f": f, "T": t, "iters": OUTER_MAXITER, "grad_inf": float(np.max(np.abs(g))),
            "n_rejected": n_rej}


def classify_starts(runs: dict, value_key: str, x_key: str) -> dict:
    """valid runs -> distinct local maxima; reports agreement, multimodality and invalid starts."""
    valid = {k: r for k, r in runs.items() if r.get("valid")}
    out = {"n_starts": len(runs), "n_valid_starts": len(valid),
           "invalid_starts": "|".join(f"{k}:{r['status']}" for k, r in runs.items() if not r.get("valid"))}
    if not valid:
        out.update(best=None, n_local_max=0, multistart="NO_VALID_START")
        return out
    order = sorted(valid, key=lambda k: -valid[k][value_key])
    clusters: list[list[str]] = []
    for k in order:
        for cl in clusters:
            r0 = valid[cl[0]]
            if np.max(np.abs(valid[k][x_key] - r0[x_key])) <= SAME_POINT_F_TOL:
                cl.append(k)
                break
        else:
            clusters.append([k])
    best = order[0]
    same_cluster_spread = max(max(valid[k][value_key] for k in cl) - min(valid[k][value_key] for k in cl) for cl in clusters)
    out.update(best=best, n_local_max=len(clusters), same_point_value_spread=float(same_cluster_spread))
    if len(clusters) == 1:
        out["multistart"] = "AGREE" if same_cluster_spread <= SAME_POINT_VALUE_TOL else "SAME_POINT_VALUE_MISMATCH"
        out["gap_second_max"] = float("nan")
    else:
        gap = valid[clusters[0][0]][value_key] - valid[clusters[1][0]][value_key]
        out["gap_second_max"] = float(gap)
        out["multistart"] = "MULTIMODAL_NONNEGLIGIBLE" if gap < MULTIMODAL_GAP else "MULTIMODAL_DOMINANT"
    return out


# --------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------

def method_D_E2(L: Local, starts: dict) -> dict:
    runs = {}
    for name, f0 in starts.items():
        r = maximise_T(L, f0)
        r["valid"] = False
        if r["status"] == "GRAD_OK":
            try:
                Hm = fd_hess(L, r["f"])
                ev = np.linalg.eigvalsh(Hm)
                r["hess_f"], r["min_eig_f"] = Hm, float(ev[0])
                r["valid"] = bool(ev[0] > 0)
                if not r["valid"]:
                    r["status"] = "T_STATIONARY_NOT_MAX"
            except EvalFail as exc:
                r["status"], r["message"] = "HESS_FD_FAIL", str(exc)
        runs[name] = r
    cl = classify_starts(runs, "T", "f")
    out = {"D_" + k: v for k, v in cl.items()}
    out["D_start_status"] = "|".join(f"{k}:{r['status']}" for k, r in runs.items())
    if cl["best"] is None:
        out["E2_status"] = "NO_VALID_LOCAL_MAX"
        return out
    b = runs[cl["best"]]
    f, Hm = b["f"], b["hess_f"]
    s = L.S(f)
    out.update({"D": b["T"], "f_D": f, "Z_D": s.Z, "D_S_at_fD": s.S, "D_logprior_at_fD": a8.log_prior(f, L.J.pv),
                "D_grad_inf": b.get("grad_inf"), "D_iters": b["iters"], "D_min_eig_f": b["min_eig_f"],
                "E2": b["T"] + 0.5 * K * LOG2PI - 0.5 * float(np.linalg.slogdet(Hm)[1]),
                "E2_logdet_f": float(np.linalg.slogdet(Hm)[1]), "hess_f": Hm})
    # FD step sensitivity of E2
    for h in FD_H_HESS_ALT:
        try:
            Ha = fd_hess(L, f, h)
            ev = np.linalg.eigvalsh(Ha)
            out[f"E2_h{h:g}"] = b["T"] + 0.5 * K * LOG2PI - 0.5 * float(np.sum(np.log(ev))) if ev[0] > 0 else float("nan")
        except EvalFail:
            out[f"E2_h{h:g}"] = float("nan")
    out["E2_status"] = "MULTIMODAL_NONNEGLIGIBLE" if cl["multistart"] == "MULTIMODAL_NONNEGLIGIBLE" else "OK"
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
        res = newton_max(vgh, np.concatenate([np.asarray(Z0, float).ravel(), np.asarray(f0, float)]),
                         JOINT_GTOL, JOINT_MAXITER, INNER_STEP_CAP)
        runs[name] = {"status": res.status, "valid": res.status in ("OK", "OK_FLOOR"), "psi": res.value, "u": res.x,
                      "f": res.x[-K:], "min_eig": res.min_eig, "n_modified": res.n_modified,
                      "n_escape": res.n_negcurv_escape}
    cl = classify_starts(runs, "psi", "f")
    out = {"E_" + k: v for k, v in cl.items()}
    out["E_start_status"] = "|".join(f"{k}:{r['status']}" for k, r in runs.items())
    if cl["best"] is None:
        out["E_status"] = "NO_VALID_LOCAL_MAX"
        return out
    b = runs[cl["best"]]
    Z, f = b["u"][:-K].reshape(N, K), b["f"]
    H = J.negH_u(Z, f)
    pd_ok, ld, _ = chol_logdet(H)
    nk = N * K
    HZZ = 0.5 * (H[:nk, :nk] + H[:nk, :nk].T)
    schur = H[nk:, nk:] - H[nk:, :nk] @ np.linalg.solve(HZZ, H[:nk, nk:])
    out.update({"E": b["psi"] + 0.5 * DU * LOG2PI - 0.5 * ld, "f_E": f, "E_min_eig": b["min_eig"],
                "E_logdet": ld, "E_logdet_ZZ": float(np.linalg.slogdet(HZZ)[1]),
                "E_logdet_schur_f": float(np.linalg.slogdet(0.5 * (schur + schur.T))[1]),
                "E_status": "MULTIMODAL_NONNEGLIGIBLE" if cl["multistart"] == "MULTIMODAL_NONNEGLIGIBLE" else "OK"})
    return out


def method_IS(L: Local, f_D, hess_f, seed, draws=IS_DRAWS) -> dict:
    """all draws in the denominator.  Categories: finite (weight exp lw), zero (log T = -inf by mathematics: does not
    occur for this model, kept for completeness), failed (numerical failure of S at the draw: NOT a zero weight)."""
    from scipy.stats import multivariate_t
    q = multivariate_t(loc=f_D, shape=np.linalg.inv(hess_f), df=IS_DF, seed=np.random.default_rng(seed))
    fs = np.atleast_2d(q.rvs(size=draws))
    lw = np.full(draws, np.nan)
    cat = np.empty(draws, dtype=object)
    for s in range(draws):
        ok, t = L.T(fs[s])
        if ok and np.isfinite(t):
            lw[s], cat[s] = t - q.logpdf(fs[s]), "finite"
        else:
            cat[s] = "failed"
    n_fin, n_fail = int((cat == "finite").sum()), int((cat == "failed").sum())
    res = {"IS_draws": draws, "IS_finite": n_fin, "IS_zero_weight": 0, "IS_failed": n_fail}
    if n_fin == 0:
        res.update(IS=float("nan"), IS_status="ALL_FAILED")
        return res
    fin = lw[cat == "finite"]
    est = float(logsumexp(fin) - math.log(draws))          # failed draws excluded from the sum but NOT from the denominator
    w = np.zeros(draws)
    w[cat == "finite"] = np.exp(fin - est)
    se = float(np.std(w, ddof=1) / math.sqrt(draws))        # delta-method SE of the log estimate
    wf = np.exp(fin - fin.max())
    res.update(IS=est if n_fail == 0 else float("nan"), IS_lower_bound_if_failed=est,
               IS_mc_se=se, IS_ess=float(wf.sum() ** 2 / np.sum(wf ** 2)),
               IS_status="OK" if n_fail == 0 else "PARTIAL_FAIL")
    return res


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------

def run_cell(case, j, cand, prior_name, case_index, main: bool) -> dict:
    t0 = time.perf_counter()
    J, v = a8.joint_for(case, j, cand, prior_name)
    Za = np.asarray(case["st"]["Zmode"], float)
    x = case["X"][:, j]
    L = Local(J, Za)
    starts = {"plugin_anchorZ": a8.fixed_z_start(cand, x, Za), "zero": np.zeros(K)}
    fl = a8.af6_fstar(case["label"], case["regime"], j, cand)
    if fl is not None:
        starts["af6_fL"] = fl
    row: dict[str, Any] = {"replicate": case["label"], "regime": case["regime"], "column": j, "candidate": cand,
                           "prior": prior_name, "prior_var_f": v}
    try:
        de = method_D_E2(L, starts)
        hess_f, Z_D = de.pop("hess_f", None), de.pop("Z_D", None)
        f_D = de.pop("f_D", None)
        row.update(de)
        if f_D is not None:
            row["f_D"] = "|".join(f"{a:.10g}" for a in f_D)
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
            # determinism: re-evaluate T(f_D) on a FRESH object after evaluating other points first
            L2 = Local(a8.joint_for(case, j, cand, prior_name)[0], Za)
            L2.T(f_D + 0.3)
            L2.T(np.zeros(K))
            ok2, t2 = L2.T(f_D)
            row["determinism_abs_diff"] = abs(t2 - row["D"]) if ok2 else float("nan")
            row.update(method_IS(L, f_D, hess_f, IS_SEED_BASE + case_index))
        row["status"] = "OK" if row.get("E2_status") == "OK" else ("FLAGGED" if row.get("E2_status") else "FAILED")
    except Exception as exc:                         # unexpected: recorded, never retried
        row.update({"status": "EXCEPTION", "error": f"{type(exc).__name__}: {exc}"[:300]})
    row["n_S_evals"] = L.n_eval
    row["seconds"] = round(time.perf_counter() - t0, 2)
    return row


# --------------------------------------------------------------------------
# selection state (defined now; not computed against labels in the pilot)
# --------------------------------------------------------------------------

def selection_state(pois: dict, cmp_: dict, score="E2") -> tuple[str, str | None, float]:
    def valid(r):
        return r is not None and r.get("status") == "OK" and np.isfinite(r.get(score, np.nan))
    vp, vc = valid(pois), valid(cmp_)
    if not vp and not vc:
        return "UNDECIDED_BOTH_FAILED", None, float("nan")
    if not (vp and vc):
        return "UNDECIDED_ONE_FAILED", None, float("nan")
    d = cmp_[score] - pois[score]
    unc = NUMERIC_FLOOR
    for r in (pois, cmp_):
        alts = [r.get(f"E2_h{h:g}", np.nan) for h in FD_H_HESS_ALT]
        alts = [abs(a - r[score]) for a in alts if np.isfinite(a)]
        if alts:
            unc = max(unc, 2 * max(alts))
    if abs(d) <= unc:
        return "UNCERTAIN_NUMERIC", (C if d > 0 else P), d
    return "DECIDED", (C if d > 0 else P), d


# --------------------------------------------------------------------------
# protocol / runinfo / drivers
# --------------------------------------------------------------------------

def protocol() -> dict[str, Any]:
    return {
        "experiment": "AF8-v2", "study_type": "exploratory numerical repair + pilot (no selection accuracy, no main experiment)",
        "parent": {"AF8_head": AF8_HEAD, "AF8_protocol_sha256": "5c8f4b2eec511de602ffaf28e2cdf15737d2b48a3d82bd61d9ffaac6a13e5815",
                   "AF8_stop": "pre-registered stop rule fired at the pilot (start disagreement share 5.6% > 5%); root causes: joint Newton converging to saddle points, failed line searches accepting non-finite trial points, LinAlgError escaping from S during BFGS (22 Poisson cells on one heavy-tailed column), sentinel 1e300 counted as start disagreement; IS divided by the number of finite draws"},
        "unchanged_from_AF8": ["model, theta_{-j} (AF6 auto_first anchor), data, masks", "log density, gradient, full negative Hessian (AF8 JointU, FD-verified)",
                               "CMP2 Bessel normaliser", "priors: main B_sg2_0.667 and the 8 sensitivity priors", "FD steps 1e-4 (gradient) / 1e-3 (Hessian)",
                               "tolerances: inner 1e-9, joint 1e-8, outer 1e-5"],
        "changed_numerics": {
            "newton": "aim tolerance as AF8 (inner 1e-9, joint 1e-8) and acceptance 1e-6 at the floating-point floor (as AF5/AF8 accepted 1e-6): OK_FLOOR when the objective no longer changes (|dv|<=1e-12|v| for 3 accepted steps or no improving step) with grad <= 1e-6 and negH PD; inner max 100 iterations; modified Newton (|eigenvalue| with floor 1e-8*max|eig| when negH not PD), Armijo c=1e-4, rejected trials never accepted, step caps (Z: 2.0 inf-norm; f: 0.5 2-norm), convergence = grad_inf < tol AND Cholesky(negH) succeeds; stationary non-PD points escaped along the most negative curvature direction, else SADDLE",
            "S_determinism": "every S(f) starts the Z-mode search from the anchor Zmode (no warm start); results cached by exact f (pure function)",
            "outer": "BFGS on T with FD gradient, capped steps, Armijo; trial points with invalid T are rejected (no sentinel values); one restart along the gradient; local max requires FD Hessian of T positive definite",
            "failures": "NaN/Inf/overflow/LinAlgError inside an evaluation raise EvalFail, which is a REJECTED trial (inside searches) or a recorded status (never a value)",
            "IS": "all 400 draws in the denominator; draw categories finite / zero-weight / failed recorded; failed draws make the IS estimate invalid (IS_lower_bound_if_failed reported separately); seeds 1689500+case",
        },
        "main_score": "E2 (nested Laplace): first integrate Z by Laplace -> S(f), then integrate f by Laplace around argmax T. E (joint Laplace) kept as a diagnostic; IS checks only the OUTER f integral (same integrand as E2), it is not a check of the Z integral. The choice of E2 was made before any v2 result and is not based on accuracy; the AF8 pilot (which included rep01-03, datasets that are part of the 40 and whose labels are known from AF6) had shown E2 closer to IS, which is partly by construction — disclosed.",
        "starts": {"f": ["plugin_anchorZ (fixed-Z MLE of the candidate at the anchor Zmode)", "zero", "af6_fL (AF6 optimum, if present)"],
                   "E_extra": "f_D with Z at f_D", "Z_inner": "anchor Zmode always"},
        "start_validity": "valid = optimiser status GRAD_OK (outer) / OK (Newton) AND local-maximum curvature check (FD Hessian of T PD / Cholesky of negH)",
        "multistart_rule": f"valid starts are the same local maximum if ||f_a-f_b||_inf <= {SAME_POINT_F_TOL}; AGREE if one cluster with value spread <= {SAME_POINT_VALUE_TOL}; SAME_POINT_VALUE_MISMATCH otherwise; >1 cluster = distinct local maxima: MULTIMODAL_NONNEGLIGIBLE if the second is within log 20 of the best (score flagged, not used for a decision), MULTIMODAL_DOMINANT otherwise. Invalid starts are counted and reported, never silently dropped.",
        "selection_states": "DECIDED (both candidates OK, |Delta| > max(0.01, 2*FD-step sensitivity)); UNCERTAIN_NUMERIC; UNDECIDED_ONE_FAILED; UNDECIDED_BOTH_FAILED. No fallback to Q (a Q-hybrid would have to be pre-registered separately).",
        "stop_rules_v2": STOP_RULES_V2,
        "stop_rules_changes_vs_AF8": {
            "failed share (AF8 5%)": "kept at 5% (S1); a cell counts as failed if E2 has no valid local maximum",
            "start disagreement (AF8: max-min over ALL starts incl. sentinels > 1e-6, share > 5%)": "kept at 5% (S2) but defined over VALID starts only: share of cells with SAME_POINT_VALUE_MISMATCH or distinct local maxima (any gap). Reason: AF8 counted saddles/sentinels as disagreement. Invalid starts are reported separately (count and reason) and do not disappear",
            "E vs IS (AF8)": "same thresholds now applied to the main score E2 (S3); E vs IS reported as diagnostic",
            "new": "S4 IS failed draws / ESS, S5 determinism, S6 FD-step sensitivity of E2, S7 FD derivative check (AF8 rule fd_rel_err_max was not evaluated in the AF8 pilot)",
        },
        "fd_check": "S7 evaluated on the main prior for the first column of every pilot (dataset, regime): analytic vs central FD gradient and Hessian of Psi at (Z_D + 0.01, f_D + 0.01)",
        "pilot": "rep01-rep03, 648 cells (same as AF8); numerical quantities only; no selection accuracy, AUC or label summaries",
        "independent_audit": "read-only research-auditor after the pilot: formulae vs code, maximum conditions, Hessian blocks, E vs E2, prior normalisation, IS weights/denominator, stop rules, AF8-vs-v2 comparison, claim scope",
        "not_claimed": ["selection accuracy", "improvement of family selection", "Bayesian evidence of the whole model", "confirmation"],
    }


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def runinfo(extra):
    import scipy
    return {"experiment": "AF8-v2", "script": "expfam/src/experimental/run_af8v2_integrated_loading.py",
            "git_head": a8.git("rev-parse", "HEAD"), "git_branch": a8.git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_tracked": bool(a8.git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "protocol_sha256": protocol_sha256(), "command": " ".join(sys.argv), **extra}


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
        ci = (r * 2 + REGIMES.index(g)) * 100 + j * 2 + CANDS.index(cand)
        row = run_cell(case, j, cand, pr, ci, main=(pr == MAIN_PRIOR))
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
        run("pilot", PILOT_REPS, sh)
    if "--collect-pilot" in argv:
        collect("pilot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
