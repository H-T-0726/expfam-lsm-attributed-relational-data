"""AF8 (exploratory): column-wise family selection with the loading f_j INTEGRATED under a proper prior.

Fixed: theta_{-j} = AF6 Auto anchor (other columns' F, families, Gaussian variances, w0, w, var_z), X, Y_train
mask, test mask (unused), dataset.  Varied per candidate m in {poisson, cmp_nu2p0}: family of column j, f_j, Z.

  Phi_m(Z, f)   = log p(X, Y_train, Z | f_j = f, theta_{-j}, m)          (proper densities incl. base measures)
  Psi_m(Z, f)   = Phi_m(Z, f) + log p(f | m, prior)
  S_m(f)        = log-Laplace over Z at fixed f  (= AF5 inner quantity)
  T_m(f)        = S_m(f) + log p(f | m, prior)

  method D  (prior MAP, Z integrated, f optimised):   D_m  = max_f T_m(f)
  method E  (joint Laplace over u = (vec Z, f), 228-dim): E_m = Psi(u_hat) + (228/2) log 2pi - 1/2 log|H_uu|
  method E2 (nested Laplace in f around argmax T):     E2_m = D_m + (K/2) log 2pi - 1/2 log|-d2T/df2|
  method IS (f integrated by importance sampling of exp T, Z by Laplace; main prior only)

Comparators taken unchanged from AF6: Q (B), AF5 Laplace (A), joint-MAP variant (E of AF6, here "J").

No MCEM.  No new data (AF6 datasets regenerated from their seeds; hashes checked against fit_manifest).
CMP(nu=2) normaliser via the modified Bessel function: A(eta) = log I0(u), u = 2 exp(eta/2), with a
cancellation-free 1 - I1/I0 for large u.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, gammaln, i0e, i1e, logsumexp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

ROOT = HERE.parent.parent.parent
RES = ROOT / "expfam" / "results" / "auto_family_z_recovery"
AF6 = RES / "af6_laplace_confirmation_20261114" / "full"
OUT = RES / "af8_integrated_loading_20261115"
N, K = 75, 3
DU = N * K + K
LOG2PI = math.log(2 * math.pi)
P, C = "poisson", "cmp_nu2p0"
CANDS = (P, C)
COUNT_COLS = (3, 4, 5, 6, 7, 8)
REGIMES = ("dense", "sparse")
A2_ZERO = {P: 1.0}          # A''_m(0); CMP2 filled below

# --------------------------------------------------------------------------
# frozen settings (written to protocol.json before any AF8 selection value is inspected)
# --------------------------------------------------------------------------
MAIN_PRIOR = "B_sg2_0.667"
PRIORS = {
    # name: (kind, scale)   kind B: g = sqrt(A''_m(0)) f ~ N(0, s I)  ->  f ~ N(0, s / A''_m(0) I)
    #                       kind A: f ~ N(0, s I) for both candidates
    "B_sg2_0.667": ("B", 2.0 / 3.0),
    "B_sg2_0.167": ("B", 1.0 / 6.0),
    "B_sg2_0.333": ("B", 1.0 / 3.0),
    "B_sg2_1.333": ("B", 4.0 / 3.0),
    "B_sg2_2.667": ("B", 8.0 / 3.0),
    "A_sf2_0.167": ("A", 1.0 / 6.0),
    "A_sf2_0.667": ("A", 2.0 / 3.0),
    "A_sf2_2.667": ("A", 8.0 / 3.0),
    "B_EB": ("B", None),          # empirical: s = mean_l A''_{a_l}(0) ||F_hat_l||^2 / K over anchor count columns l != j
}
FD_STEP_GRAD = 1e-4
FD_STEP_HESS = 1e-3
OUTER_GTOL = 1e-5
OUTER_MAXITER = 200
INNER_TOL = 1e-9
JOINT_TOL = 1e-8
NUMERIC_FLOOR = 1e-2
IS_DRAWS = 400
IS_DF = 5
IS_SEED_BASE = 1688000
BOOT_SEED = 20261118
PILOT_REPS = (0, 1, 2)
STOP_RULES = {
    "fd_rel_err_max": 1e-4,
    "mode_start_disagreement_share_max": 0.05,
    "pilot_median_abs_DeltaE_minus_DeltaIS_max": 0.25,
    "pilot_median_abs_DeltaE_minus_DeltaIS_below_half_median_abs_DeltaE": True,
    "pilot_E_failed_share_max": 0.05,
}


# --------------------------------------------------------------------------
# CMP(nu=2): stable normaliser and derivatives
# --------------------------------------------------------------------------

def _one_minus_r(u):
    """1 - I1(u)/I0(u) without cancellation (asymptotic series for u > 1e3)."""
    u = np.asarray(u, float)
    out = np.empty_like(u)
    big = u > 1e3
    ub = u[big]
    out[big] = 1 / (2 * ub) + 1 / (8 * ub ** 2) + 1 / (8 * ub ** 3) + 25 / (128 * ub ** 4)
    us = u[~big]
    out[~big] = 1.0 - i1e(us) / i0e(us)
    return out


def cmp2_A(eta):
    u = 2.0 * np.exp(np.asarray(eta, float) / 2.0)
    return np.log(i0e(u)) + u


def cmp2_dA(eta):
    u = 2.0 * np.exp(np.asarray(eta, float) / 2.0)
    return 0.5 * u * (1.0 - _one_minus_r(u))


def cmp2_d2A(eta):
    u = 2.0 * np.exp(np.asarray(eta, float) / 2.0)
    omr = _one_minus_r(u)
    return 0.25 * u * u * omr * (2.0 - omr)


A2_ZERO[C] = float(cmp2_d2A(np.array([0.0]))[0])


def col_terms(fam, x, eta, s2=1.0):
    """(log p, d/deta log p, -d2/deta2 log p) with the proper base measure of each family."""
    if fam == "gaussian":
        return (-0.5 * (x - eta) ** 2 / s2 - 0.5 * math.log(2 * math.pi * s2), (x - eta) / s2,
                np.full_like(eta, 1.0 / s2))
    if fam == P:
        m = np.exp(eta)
        return x * eta - m - gammaln(x + 1), x - m, m
    if fam == C:
        return x * eta - 2 * gammaln(x + 1) - cmp2_A(eta), x - cmp2_dA(eta), cmp2_d2A(eta)
    raise ValueError(fam)


# --------------------------------------------------------------------------
# priors
# --------------------------------------------------------------------------

def prior_var(name: str, cand: str, anchor_F=None, anchor_fams=None, j=None) -> float:
    kind, s = PRIORS[name]
    if s is None:
        ls = [l for l in COUNT_COLS if l != j]
        s = float(np.mean([A2_ZERO[anchor_fams[l]] * np.sum(anchor_F[l] ** 2) / K for l in ls]))
    return s / A2_ZERO[cand] if kind == "B" else s


def log_prior(f, v):
    """log N(f; 0, v I_K), normalised; for kind B this equals log N(g; 0, s I) + (K/2) log A''_m(0) (Jacobian)."""
    f = np.asarray(f, float)
    return -0.5 * K * math.log(2 * math.pi * v) - 0.5 * float(f @ f) / v


# --------------------------------------------------------------------------
# joint model in u = (vec Z, f_j)
# --------------------------------------------------------------------------

class JointU:
    def __init__(self, X, Y, train, fams, F, s2, w0, w, var_z, j, prior_v):
        self.X, self.Y = np.asarray(X, float), np.asarray(Y, float)
        up = np.triu(np.asarray(train, bool), 1)
        self.up, self.M = up, (up | up.T).astype(float)
        self.fams, self.F, self.s2 = list(fams), np.array(F, float), np.asarray(s2, float)
        self.w0, self.w, self.var_z = float(w0), float(w), float(var_z)
        self.j, self.pv = int(j), float(prior_v)

    # ---- quantities at fixed f (Z only) ----
    def _cols(self, Z):
        eta = Z @ self.F.T
        out = [col_terms(fam, self.X[:, l], eta[:, l], self.s2[l]) for l, fam in enumerate(self.fams)]
        return eta, out

    def phi(self, Z):
        _, cols = self._cols(Z)
        ey = self.w0 + self.w * (Z @ Z.T)
        return (-0.5 * Z.size * math.log(2 * math.pi * self.var_z) - 0.5 * float(np.sum(Z * Z)) / self.var_z
                + sum(float(np.sum(c[0])) for c in cols)
                + float(np.sum((self.Y * ey - np.logaddexp(0, ey))[self.up])))

    def target_logp(self, Z):
        eta = Z @ self.F[self.j]
        return float(np.sum(col_terms(self.fams[self.j], self.X[:, self.j], eta)[0]))

    def gradZ(self, Z):
        _, cols = self._cols(Z)
        g = -Z / self.var_z
        for l, c in enumerate(cols):
            g = g + np.outer(c[1], self.F[l])
        R = self.M * (self.Y - expit(self.w0 + self.w * (Z @ Z.T)))
        return g + self.w * (R @ Z)

    def negHZZ(self, Z):
        n, k = Z.shape
        _, cols = self._cols(Z)
        A = np.stack([c[2] for c in cols], axis=1)
        p = expit(self.w0 + self.w * (Z @ Z.T))
        Cm = self.M * self.w ** 2 * p * (1 - p)
        R = self.M * (self.Y - p)
        H = np.einsum("ik,ka,ib->iakb", Cm, Z, Z) - self.w * np.einsum("ik,ab->iakb", R, np.eye(k))
        diag = (np.eye(k) / self.var_z)[None] + np.einsum("ij,ja,jb->iab", A, self.F, self.F) \
            + np.einsum("ik,ka,kb->iab", Cm, Z, Z)
        idx = np.arange(n)
        H[idx, :, idx, :] = diag
        return H.reshape(n * k, n * k)

    def modeZ(self, Z0, tol=INNER_TOL, max_iter=200):
        Z, fz = Z0.copy(), self.phi(Z0)
        for it in range(max_iter):
            g = self.gradZ(Z)
            gi = float(np.max(np.abs(g)))
            if gi < tol:
                return Z, it, gi
            H = self.negHZZ(Z)
            step = np.linalg.solve(0.5 * (H + H.T), g.ravel()).reshape(Z.shape)
            a = 1.0
            while a > 1e-10:
                Zt = Z + a * step
                ft = self.phi(Zt)
                if ft >= fz - 1e-12 * abs(fz):
                    break
                a *= 0.5
            Z, fz = Zt, ft
        return Z, max_iter, float(np.max(np.abs(self.gradZ(Z))))

    def S(self, f, Z0):
        """Laplace over Z at fixed f_j = f."""
        self.F[self.j] = np.asarray(f, float)
        Zh, it, gi = self.modeZ(Z0)
        H = self.negHZZ(Zh)
        ev = np.linalg.eigvalsh(0.5 * (H + H.T))
        if gi > 1e-6 or ev[0] <= 0:
            return -math.inf, Zh, gi, float(ev[0])
        ph = self.phi(Zh)
        return ph + 0.5 * Zh.size * LOG2PI - 0.5 * float(np.sum(np.log(ev))), Zh, gi, float(ev[0])

    # ---- joint u = (Z, f) ----
    def psi(self, Z, f):
        self.F[self.j] = np.asarray(f, float)
        return self.phi(Z) + log_prior(f, self.pv)

    def grad_u(self, Z, f):
        self.F[self.j] = np.asarray(f, float)
        gZ = self.gradZ(Z)
        eta = Z @ self.F[self.j]
        r = col_terms(self.fams[self.j], self.X[:, self.j], eta)[1]
        gf = Z.T @ r - np.asarray(f, float) / self.pv
        return gZ, gf

    def negH_u(self, Z, f):
        self.F[self.j] = np.asarray(f, float)
        n, k = Z.shape
        HZZ = self.negHZZ(Z)
        eta = Z @ self.F[self.j]
        _, r, a2 = col_terms(self.fams[self.j], self.X[:, self.j], eta)
        fj = self.F[self.j]
        # block (z_i, f): -d2 Psi / dz_ia df_b = A''_i f_a z_ib - r_i delta_ab
        HZf = (a2[:, None, None] * fj[None, :, None] * Z[:, None, :]) - r[:, None, None] * np.eye(k)[None]
        HZf = HZf.reshape(n * k, k)
        Hff = (Z * a2[:, None]).T @ Z + np.eye(k) / self.pv
        H = np.zeros((n * k + k, n * k + k))
        H[:n * k, :n * k] = HZZ
        H[:n * k, n * k:] = HZf
        H[n * k:, :n * k] = HZf.T
        H[n * k:, n * k:] = Hff
        return H

    def mode_u(self, Z0, f0, tol=JOINT_TOL, max_iter=300):
        Z, f = Z0.copy(), np.asarray(f0, float).copy()
        val = self.psi(Z, f)
        for it in range(max_iter):
            gZ, gf = self.grad_u(Z, f)
            g = np.concatenate([gZ.ravel(), gf])
            gi = float(np.max(np.abs(g)))
            if gi < tol:
                return Z, f, it, gi
            H = self.negH_u(Z, f)
            Hs = 0.5 * (H + H.T)
            try:
                step = np.linalg.solve(Hs, g)
                if not np.all(np.isfinite(step)) or float(step @ g) <= 0:
                    raise np.linalg.LinAlgError
            except np.linalg.LinAlgError:
                step = g / max(1.0, float(np.max(np.abs(np.diag(Hs)))))
            a = 1.0
            while a > 1e-12:
                Zt, ft = Z + a * step[:-K].reshape(Z.shape), f + a * step[-K:]
                vt = self.psi(Zt, ft)
                if np.isfinite(vt) and vt >= val - 1e-12 * abs(val):
                    break
                a *= 0.5
            Z, f, val = Zt, ft, vt
        gZ, gf = self.grad_u(Z, f)
        return Z, f, max_iter, float(max(np.max(np.abs(gZ)), np.max(np.abs(gf))))


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def sha16(a):
    import run_af6_laplace_confirmation as a6
    return a6.af2.sha16(a)


def load_case(r: int, regime: str):
    import run_af6_laplace_confirmation as a6
    ds, masks = a6.dataset(r)
    fm = pd.read_csv(AF6 / "fit_manifest.csv")
    m = fm[(fm.replicate == a6.label(r)) & (fm.regime == regime)].iloc[0]
    assert sha16(ds.X) == m.x_sha and sha16(ds.Y) == m.y_sha and sha16(masks[regime]) == m.train_mask_sha
    with np.load(m.state_file) as z:
        st = {k: np.asarray(z[k]) for k in z.files}
    return {"X": np.asarray(ds.X, float), "Y": np.asarray(ds.Y, float), "train": masks[regime], "st": st,
            "assign": m.assignment.split("|"), "label": a6.label(r)}


def joint_for(case, j, cand, prior_name):
    st = case["st"]
    fams = list(case["assign"])
    fams[j] = cand
    w0, w, var_z = (float(v) for v in st["scalars"])
    v = prior_var(prior_name, cand, np.asarray(st["F"], float), case["assign"], j)
    return JointU(case["X"], case["Y"], case["train"], fams, np.asarray(st["F"], float), np.diag(st["sigma"]),
                  w0, w, var_z, j, v), v


def fixed_z_start(fam, x, Z):
    f = np.zeros(Z.shape[1])
    for _ in range(100):
        eta = Z @ f
        _, g1, g2 = col_terms(fam, x, eta)
        g = Z.T @ g1
        if np.max(np.abs(g)) < 1e-9:
            break
        H = (Z * g2[:, None]).T @ Z
        step = np.linalg.solve(H + 1e-12 * np.eye(len(f)), g)
        t, ll0 = 1.0, np.sum(col_terms(fam, x, eta)[0])
        while t > 1e-8 and np.sum(col_terms(fam, x, Z @ (f + t * step))[0]) < ll0:
            t *= 0.5
        f = f + t * step
    return f


# --------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------

def method_D(J: JointU, Za, starts: dict):
    from scipy.optimize import minimize
    warm = {"Z": Za.copy()}

    def T(f):
        s, Zh, gi, me = J.S(f, warm["Z"])
        if not math.isfinite(s):
            return -1e300
        warm["Z"] = Zh
        return s + log_prior(f, J.pv)

    def nT(f):
        return -T(np.asarray(f, float))

    def jac(f):
        g = np.zeros(K)
        for a in range(K):
            e = np.zeros(K)
            e[a] = FD_STEP_GRAD
            g[a] = (nT(f + e) - nT(f - e)) / (2 * FD_STEP_GRAD)
        return g

    runs = {}
    for name, f0 in starts.items():
        warm["Z"] = Za.copy()
        res = minimize(nT, np.asarray(f0, float), jac=jac, method="BFGS",
                       options={"gtol": OUTER_GTOL, "maxiter": OUTER_MAXITER})
        runs[name] = (-float(res.fun), np.asarray(res.x, float), int(res.nit))
    best = max(runs, key=lambda k: runs[k][0])
    val, f, nit = runs[best]
    warm["Z"] = Za.copy()
    s, Zh, gi, me = J.S(f, warm["Z"])
    warm["Z"] = Zh
    gfd = float(np.max(np.abs(jac(f))))
    vals = [v[0] for v in runs.values()]
    return {"D": val, "f_D": f, "Z_D": Zh, "D_best_start": best, "D_start_disagreement": float(max(vals) - min(vals)),
            "D_fd_grad_inf": gfd, "D_S_at_fD": s, "D_logprior_at_fD": log_prior(f, J.pv), "D_iters": nit,
            "D_inner_grad": gi, "D_min_eig_ZZ": me, "_T": T, "_warm": warm}


def method_E(J: JointU, Za, starts: dict):
    runs = {}
    for name, f0 in starts.items():
        Z, f, it, gi = J.mode_u(Za, f0)
        runs[name] = (J.psi(Z, f), Z, f, it, gi)
    best = max(runs, key=lambda k: runs[k][0])
    val, Z, f, it, gi = runs[best]
    H = J.negH_u(Z, f)
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    vals = [v[0] for v in runs.values()]
    fdiff = max(float(np.max(np.abs(v[2] - f))) for v in runs.values())
    ok = bool(gi < 1e-6 and ev[0] > 0)
    E = val + 0.5 * DU * LOG2PI - 0.5 * float(np.sum(np.log(ev))) if ok else -math.inf
    # Schur split: log|H| = log|H_ZZ| + log|H_ff - H_fZ H_ZZ^-1 H_Zf|
    nk = N * K
    HZZ = 0.5 * (H[:nk, :nk] + H[:nk, :nk].T)
    sZZ, ldZZ = np.linalg.slogdet(HZZ)
    schur = H[nk:, nk:] - H[nk:, :nk] @ np.linalg.solve(HZZ, H[:nk, nk:])
    sS, ldS = np.linalg.slogdet(0.5 * (schur + schur.T))
    J.F[J.j] = f
    return {"E": E, "E_ok": ok, "f_E": f, "Z_E": Z, "E_psi": val, "E_phi": val - log_prior(f, J.pv),
            "E_logprior": log_prior(f, J.pv), "E_logdet": float(np.sum(np.log(ev))) if ok else np.nan,
            "E_logdet_ZZ": ldZZ, "E_logdet_schur_f": ldS, "E_min_eig": float(ev[0]), "E_grad_inf": gi, "E_iters": it,
            "E_best_start": best, "E_start_disagreement": float(max(vals) - min(vals)), "E_start_f_maxdiff": fdiff}


def method_E2(D: dict):
    """nested Laplace in f around argmax T (FD Hessian of T)."""
    T, f0 = D["_T"], D["f_D"]
    h = FD_STEP_HESS
    D["_warm"]["Z"] = D["Z_D"].copy()
    t0 = T(f0)
    Hm = np.zeros((K, K))
    for a in range(K):
        for b in range(a, K):
            ea, eb = np.eye(K)[a] * h, np.eye(K)[b] * h
            if a == b:
                Hm[a, a] = -(T(f0 + ea) - 2 * t0 + T(f0 - ea)) / h ** 2
            else:
                Hm[a, b] = Hm[b, a] = -(T(f0 + ea + eb) - T(f0 + ea - eb) - T(f0 - ea + eb) + T(f0 - ea - eb)) / (4 * h * h)
    ev = np.linalg.eigvalsh(Hm)
    ok = bool(ev[0] > 0)
    E2 = D["D"] + 0.5 * K * LOG2PI - 0.5 * float(np.sum(np.log(ev))) if ok else -math.inf
    return {"E2": E2, "E2_ok": ok, "E2_hess_f": Hm, "E2_min_eig_f": float(ev[0]),
            "E2_logdet_f": float(np.sum(np.log(ev))) if ok else np.nan}


def method_IS(D: dict, E2: dict, seed: int, draws=IS_DRAWS):
    from scipy.stats import multivariate_t
    if not E2["E2_ok"]:
        return {"IS": np.nan, "IS_ess": np.nan, "IS_mc_se": np.nan, "IS_nonfinite": draws}
    T = D["_T"]
    cov = np.linalg.inv(E2["E2_hess_f"])
    q = multivariate_t(loc=D["f_D"], shape=cov, df=IS_DF, seed=np.random.default_rng(seed))
    fs = q.rvs(size=draws)
    lw = np.empty(draws)
    for s in range(draws):
        D["_warm"]["Z"] = D["Z_D"].copy()
        lw[s] = T(fs[s]) - q.logpdf(fs[s])
    fin = np.isfinite(lw) & (lw > -1e299)
    lw = lw[fin]
    m = logsumexp(lw) - math.log(len(lw))
    w = np.exp(lw - lw.max())
    ess = float(w.sum() ** 2 / np.sum(w ** 2))
    # delta-method SE of log mean weight
    wn = np.exp(lw - m)
    se = float(np.std(wn, ddof=1) / math.sqrt(len(lw)))
    return {"IS": float(m), "IS_ess": ess, "IS_mc_se": se, "IS_nonfinite": int(draws - fin.sum())}


# --------------------------------------------------------------------------
# one column
# --------------------------------------------------------------------------

def af6_fstar(rep, regime, j, cand):
    cs = _af6_cs()
    c = cs[(cs.replicate == rep) & (cs.regime == regime) & (cs.column == j) & (cs.candidate == cand) & (cs.status == "OK")]
    return None if c.empty else np.array([float(v) for v in str(c.iloc[0].f_star).split("|")])


_CS = None


def _af6_cs():
    global _CS
    if _CS is None:
        _CS = pd.read_csv(AF6 / "candidate_scores.csv")
    return _CS


def run_cell(case, j, cand, prior_name, case_index, full_extras: bool):
    t0 = time.perf_counter()
    J, v = joint_for(case, j, cand, prior_name)
    Za = np.asarray(case["st"]["Zmode"], float)
    x = case["X"][:, j]
    fl = af6_fstar(case["label"], case["regime"], j, cand)
    starts = {"plugin_anchorZ": fixed_z_start(cand, x, Za), "zero": np.zeros(K)}
    if fl is not None:
        starts["af6_fL"] = fl
    row = {"replicate": case["label"], "regime": case["regime"], "column": j, "candidate": cand, "prior": prior_name,
           "prior_kind": PRIORS[prior_name][0], "prior_var_f": v}
    try:
        D = method_D(J, Za, starts)
        row.update({k: v_ for k, v_ in D.items() if not k.startswith("_") and k not in ("f_D", "Z_D")})
        row["f_D"] = "|".join(f"{a:.10g}" for a in D["f_D"])
        Es = dict(starts, f_D=D["f_D"])
        E = method_E(J, Za, Es)
        row.update({k: v_ for k, v_ in E.items() if k not in ("f_E", "Z_E")})
        row["f_E"] = "|".join(f"{a:.10g}" for a in E["f_E"])
        row["fD_norm"], row["fE_norm"] = float(np.linalg.norm(D["f_D"])), float(np.linalg.norm(E["f_E"]))
        if full_extras:
            E2 = method_E2(D)
            row.update({k: v_ for k, v_ in E2.items() if k != "E2_hess_f"})
            row.update(method_IS(D, E2, IS_SEED_BASE + case_index))
            J.F[j] = E["f_E"]
            row["E_target_logp"] = J.target_logp(E["Z_E"])
        row["status"] = "OK" if (math.isfinite(row["D"]) and row["E_ok"]) else "FAILED_NUMERIC"
    except Exception as exc:                             # recorded, never retried
        row.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"[:300]})
    row["seconds"] = round(time.perf_counter() - t0, 2)
    return row


# --------------------------------------------------------------------------
# protocol / runinfo
# --------------------------------------------------------------------------

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    import subprocess
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def protocol() -> dict[str, Any]:
    return {
        "experiment": "AF8", "study_type": "exploratory (re-uses AF6 datasets and saved Auto anchors; NOT an independent confirmation)",
        "lineage": "lineage E + PR #129 generic model (prototype); AF6 anchors at 75012e8",
        "fixed": "theta_{-j} of the AF6 auto_first anchor (other F, families, Gaussian variances, w0, w, var_z), X, Y_train mask; same for every method",
        "mcem_fits": 0, "new_data": "none (AF6 seeds regenerated; X/Y/mask hashes asserted against AF6 fit_manifest)",
        "methods": {
            "B_Q": "AF6 q_selected (same anchor), unchanged",
            "A_laplace_fopt": "AF6 delta (AF5 S_L: f_j maximises the Z-Laplace objective, no prior), unchanged incl. AF6 fallback for FAILED",
            "J_jointmap_then_laplace": "AF6 delta_jointmap_variant, unchanged",
            "D_prior_map": "max_f [S(f) + log p(f|m)], Z by Laplace, f optimised (no f volume term)",
            "E_joint_laplace": "Laplace over u = (vec Z, f_j), 228 dims, full negative Hessian incl. Z-f cross blocks and prior; selection by sign",
            "E2_nested_laplace": "D + (K/2) log 2pi - 1/2 log|-d2T/df2| (FD Hessian), main prior only: numerical check of E",
            "IS_f": "f integrated by importance sampling (multivariate t df=5 around f_D, cov = inverse FD Hessian, 400 draws), Z by Laplace; main prior only: numerical check of E",
        },
        "priors": {k: {"kind": v[0], "scale": v[1]} for k, v in PRIORS.items()},
        "prior_main": MAIN_PRIOR,
        "prior_rationale": "B: g = sqrt(A''_m(0)) f ~ N(0, s I) with s = tau/K = 2/3, the per-component second moment of the generator's standardised loadings (design knowledge, not labels; the generator's loadings are NOT Gaussian). Sensitivity range fixed here: B x{1/4,1/2,2,4}, A with s in {1/6, 2/3, 8/3}, empirical-Bayes B from the anchor's other count columns. Not tuned on AF6 accuracy.",
        "selection_rule": "CMP2 if Delta > 0 else Poisson; |Delta| <= max(0.01, start disagreement) flagged UNDECIDED_NUMERIC but the sign is still used (as AF6 method A); every column kept",
        "failure_rule": "a FAILED cell (either candidate) -> the method falls back to Q for the main tables; sensitivity: counted wrong. Never re-run.",
        "metrics": ["accuracy on true Poisson", "accuracy on true CMP2", "balanced accuracy", "AUC of Delta (rank, descriptive)",
                    "per-dataset #correct difference vs A, B, D, J (40 datasets, percentile bootstrap 10000)",
                    "dataset-cluster bootstrap (2000) CIs for accuracies, balanced accuracy, AUC and their differences",
                    "transitions A->E, B->E, D->E, J->E", "Delta_E - Delta_D and Delta_E - Delta_A by truth",
                    "prior sensitivity (all metrics for every prior)", "numerical: E vs E2 vs IS, start disagreement, min eigenvalues, FD checks",
                    "computation time"],
        "statistical_unit": "dataset (40); columns and regimes are not independent units; dense and sparse reported separately",
        "pilot": "rep01-rep03 (both regimes, all count columns, main prior + all priors) for the numerical audit only; accuracy is not computed in the pilot",
        "stop_rules": STOP_RULES,
        "numerics": {"FD_STEP_GRAD": FD_STEP_GRAD, "FD_STEP_HESS": FD_STEP_HESS, "OUTER_GTOL": OUTER_GTOL,
                     "INNER_TOL": INNER_TOL, "JOINT_TOL": JOINT_TOL, "IS_DRAWS": IS_DRAWS, "IS_DF": IS_DF,
                     "IS_SEED_BASE": IS_SEED_BASE, "BOOT_SEED": BOOT_SEED,
                     "cmp2": "A = log I0(u), u = 2 e^{eta/2}; 1 - I1/I0 by asymptotic series for u > 1e3"},
        "not_claimed": ["Bayesian evidence of the whole model (theta_{-j} fixed)", "BIC", "independent confirmation",
                        "posterior probability of a family", "Z recovery", "real data"],
    }


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def runinfo(extra):
    import scipy
    return {"experiment": "AF8", "script": "expfam/src/experimental/run_af8_integrated_loading.py",
            "git_head": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_tracked": bool(git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "protocol_sha256": protocol_sha256(), "command": " ".join(sys.argv), **extra}


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------

def cells(reps):
    for r in reps:
        for g in REGIMES:
            for j in COUNT_COLS:
                for cand in CANDS:
                    for pr in PRIORS:
                        yield r, g, j, cand, pr


def run(stage: str, reps, shard=None):
    out = OUT / stage
    out.mkdir(parents=True, exist_ok=True)
    tag = ""
    items = list(cells(reps))
    keys_r = sorted({(r, g) for r, g, *_ in items})
    if shard:
        i, n = (int(v) for v in shard.split("/"))
        keep = set(keys_r[i::n])
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
            case = load_case(r, g)
            case["regime"] = g
        ci = (r * 2 + REGIMES.index(g)) * 100 + j * 2 + CANDS.index(cand)
        row = run_cell(case, j, cand, pr, ci, full_extras=(pr == MAIN_PRIOR))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        print(f"  {label} {g} j={j} {cand[:4]} {pr} {row['status']} {row['seconds']}s total {time.time() - t0:.0f}s",
              flush=True)


def collect(stage: str):
    out = OUT / stage
    rows = []
    for p in sorted(out.glob("progress*.jsonl")):
        rows += [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()]
    c = pd.DataFrame(rows)
    c.to_csv(out / "candidate_scores.csv", index=False)
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
        man = [{"replicate": f"rep{r + 1:02d}", "regime": g, "column": j, "candidate": cand, "prior": pr,
                "case_index": (r * 2 + REGIMES.index(g)) * 100 + j * 2 + CANDS.index(cand),
                "is_seed": (IS_SEED_BASE + (r * 2 + REGIMES.index(g)) * 100 + j * 2 + CANDS.index(cand))
                if pr == MAIN_PRIOR else "", "pilot": r in PILOT_REPS}
               for r, g, j, cand, pr in cells(range(40))]
        pd.DataFrame(man).to_csv(OUT / "manifest.csv", index=False)
        print("protocol sha256", protocol_sha256(), "cells", len(man))
    if "--pilot" in argv:
        run("pilot", PILOT_REPS, sh)
    if "--collect-pilot" in argv:
        collect("pilot")
    if "--full" in argv:
        run("full", range(40), sh)
    if "--collect" in argv:
        collect("full")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
