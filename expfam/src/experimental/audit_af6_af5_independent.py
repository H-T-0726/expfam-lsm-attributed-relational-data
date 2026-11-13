"""AF6 Phase A: independent numerical re-computation of AF5 (no MCEM).

Independent of the AF5 / AF4 / legacy code paths for everything that defines
the score: log joint, gradient, negative Hessian (incl. Y cross blocks), joint
mode (own damped Newton), log-determinant (eigenvalues), candidate score S,
Delta, selection, status rule, headline metrics and CIs.

Only the *inputs* are shared with AF5: the generated data (AF2 generator +
masks; checked against AF2's recorded hashes), the saved anchor parameters
(AF2 npz) and the f_j* that AF5 reports (candidate_scores.csv).

CMP(nu=2) normaliser: sum_k e^{k eta}/(k!)^2 = I_0(2 e^{eta/2}) (modified
Bessel), so A(eta) = log I_0(u), A' = (u/2) I_1/I_0, A'' = (u^2/4)(1 - r^2),
r = I_1/I_0, u = 2 e^{eta/2}.  This is a closed form, not the truncated series
used by family_specs.

Approximation error of Delta (Laplace vs exact marginal at the same f*):
thermodynamic integration on the path
    pi_t(Z) ∝ exp( Phi_P(Z) + t [l_C(x_j | Z f_C) - l_P(x_j | Z f_P)] ),
    log p_C - log p_P = ∫_0^1 E_{pi_t}[ l_C - l_P ] dt,
Gauss-Legendre nodes, whitened HMC per node (Laplace of pi_t as metric).
Only column j enters the integrand; every other term is common to both
candidates, so the estimator targets the *difference* directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, gammaln, i0e, i1e

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_af2_poisson_cmp as af2  # noqa: E402  (data generator + masks only)

ROOT = HERE.parent.parent.parent
RES = ROOT / "expfam" / "results" / "auto_family_z_recovery"
AF5 = RES / "af5_local_laplace_selection_20261113"
AF2_FULL = RES / "af2_20261110" / "full"
AF2_STATE = Path(r"D:\tento\kennkyu_artifacts\auto_family_z_recovery_af2_20261110\full\fits")
AF3_COLS = RES / "af3_selector_rescoring_20261113" / "full" / "column_selection.csv"
OUT = RES / "af6_laplace_confirmation_20261114" / "phase_a"
N, K = 75, 3
LOG2PI = math.log(2 * math.pi)
GL_NODES = 6
HMC_EPS, HMC_STEPS, HMC_WARM, HMC_KEEP, HMC_CHAINS = 0.30, 8, 150, 600, 2
TI_SEED = 1668000


# --------------------------------------------------------------------------
# independent densities
# --------------------------------------------------------------------------

def cmp2_A(eta):
    u = 2.0 * np.exp(0.5 * eta)
    return np.log(i0e(u)) + u


def cmp2_dA(eta):
    u = 2.0 * np.exp(0.5 * eta)
    return 0.5 * u * i1e(u) / i0e(u)


def cmp2_d2A(eta):
    u = 2.0 * np.exp(0.5 * eta)
    r = i1e(u) / i0e(u)
    return 0.25 * u * u * (1.0 - r * r)


def col_terms(fam, x, eta, s2=None):
    """returns (log p, dlogp/deta, -d2logp/deta2) elementwise"""
    if fam == "gaussian":
        return (-0.5 * (x - eta) ** 2 / s2 - 0.5 * math.log(2 * math.pi * s2), (x - eta) / s2,
                np.full_like(eta, 1.0 / s2))
    if fam == "poisson":
        return x * eta - np.exp(eta) - gammaln(x + 1), x - np.exp(eta), np.exp(eta)
    if fam == "cmp_nu2p0":
        return x * eta - 2 * gammaln(x + 1) - cmp2_A(eta), x - cmp2_dA(eta), cmp2_d2A(eta)
    raise ValueError(fam)


class Joint:
    def __init__(self, X, Y, train, fams, F, s2, w0, w, var_z):
        self.X, self.Y = X, Y
        up = np.triu(np.asarray(train, bool), 1)
        self.M = (up | up.T).astype(float)
        self.up = up
        self.fams, self.F, self.s2 = list(fams), np.array(F, float), np.asarray(s2, float)
        self.w0, self.w, self.var_z = float(w0), float(w), float(var_z)

    def parts(self, Z):
        eta = Z @ self.F.T
        lp = np.zeros(len(self.fams))
        for j, fam in enumerate(self.fams):
            lp[j] = np.sum(col_terms(fam, self.X[:, j], eta[:, j], self.s2[j])[0])
        ey = self.w0 + self.w * (Z @ Z.T)
        ly = float(np.sum((self.Y * ey - np.logaddexp(0, ey))[self.up]))
        pr = -0.5 * Z.size * math.log(2 * math.pi * self.var_z) - 0.5 * float(np.sum(Z * Z)) / self.var_z
        return pr, lp, ly

    def phi(self, Z):
        pr, lp, ly = self.parts(Z)
        return pr + float(lp.sum()) + ly

    def grad(self, Z):
        eta = Z @ self.F.T
        g = -Z / self.var_z
        for j, fam in enumerate(self.fams):
            g += np.outer(col_terms(fam, self.X[:, j], eta[:, j], self.s2[j])[1], self.F[j])
        R = self.M * (self.Y - expit(self.w0 + self.w * (Z @ Z.T)))
        return g + self.w * (R @ Z)

    def lg(self, Z):
        """fused (phi, grad) in one pass"""
        eta = Z @ self.F.T
        phi = -0.5 * Z.size * math.log(2 * math.pi * self.var_z) - 0.5 * float(np.sum(Z * Z)) / self.var_z
        g = -Z / self.var_z
        for j, fam in enumerate(self.fams):
            lp, d1, _ = col_terms(fam, self.X[:, j], eta[:, j], self.s2[j])
            phi += float(np.sum(lp))
            g += np.outer(d1, self.F[j])
        ey = self.w0 + self.w * (Z @ Z.T)
        phi += float(np.sum((self.Y * ey - np.logaddexp(0, ey))[self.up]))
        g += self.w * ((self.M * (self.Y - expit(ey))) @ Z)
        return phi, g

    def negH(self, Z):
        n, k = Z.shape
        eta = Z @ self.F.T
        A = np.zeros((n, len(self.fams)))
        for j, fam in enumerate(self.fams):
            A[:, j] = col_terms(fam, self.X[:, j], eta[:, j], self.s2[j])[2]
        p = expit(self.w0 + self.w * (Z @ Z.T))
        C = self.M * self.w ** 2 * p * (1 - p)
        R = self.M * (self.Y - p)
        H = np.zeros((n, k, n, k))
        # off-diagonal blocks: C_ik z_k z_i^T - w R_ik I
        H += np.einsum("ik,ka,ib->iakb", C, Z, Z)
        H -= self.w * np.einsum("ik,ab->iakb", R, np.eye(k))
        diag = (np.eye(k) / self.var_z)[None] + np.einsum("ij,ja,jb->iab", A, self.F, self.F) \
            + np.einsum("ik,ka,kb->iab", C, Z, Z)
        for i in range(n):
            H[i, :, i, :] = diag[i]          # M_ii = 0, so the einsums put 0 there
        return H.reshape(n * k, n * k)

    def mode(self, Z0, tol=1e-9, max_iter=100):
        Z = Z0.copy()
        f = self.phi(Z)
        for it in range(max_iter):
            g = self.grad(Z)
            if np.max(np.abs(g)) < tol:
                return Z, it, float(np.max(np.abs(g)))
            H = self.negH(Z)
            step = np.linalg.solve(0.5 * (H + H.T), g.ravel()).reshape(Z.shape)
            a = 1.0
            while a > 1e-8:
                Zt = Z + a * step
                ft = self.phi(Zt)
                if ft >= f - 1e-12 * abs(f):
                    break
                a *= 0.5
            Z, f = Zt, ft
        return Z, max_iter, float(np.max(np.abs(self.grad(Z))))


def laplace_S(J: Joint, Z0):
    Zh, it, g = J.mode(Z0)
    H = J.negH(Zh)
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    logdet = float(np.sum(np.log(ev)))
    phi = J.phi(Zh)
    return {"S": phi + 0.5 * Zh.size * LOG2PI - 0.5 * logdet, "phi": phi, "logdet": logdet,
            "min_eig": float(ev.min()), "grad_inf": g, "iters": it, "Z": Zh, "H": H}


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

sha16 = af2.sha16          # AF2's recorded hash convention (dtype + shape + bytes)


_cache: dict = {}


def inputs(r, regime, anchor):
    key = (r, regime, anchor)
    if key not in _cache:
        ds, masks = af2.generate(r), af2.m1.build_masks(r)
        fr = pd.read_csv(AF2_FULL / "fit_runs.csv")
        row = fr[(fr.replicate == af2.m1.replicate_label(r)) & (fr.regime == regime) & (fr.method == anchor)].iloc[0]
        with np.load(AF2_STATE / f"{af2.m1.replicate_label(r)}_{regime}_{anchor}.npz") as d:
            st = {k: np.asarray(d[k]) for k in d.files}
        X, Y = np.asarray(ds.X, float), np.asarray(ds.Y, float)
        _cache[key] = dict(X=X, Y=Y, train=masks[regime], test=masks["test"], st=st,
                           assign=row.assignment.split("|"),
                           x_ok=sha16(ds.X) == row.x_sha, y_ok=sha16(ds.Y) == row.y_sha,
                           train_sha_ok=(sha16(masks[regime]) == row.train_mask_sha),
                           overlap=int(np.sum(np.triu(masks[regime], 1) & np.triu(masks["test"], 1))))
    return _cache[key]


def joint_for(r, regime, anchor, j, cand, f):
    d = inputs(r, regime, anchor)
    fams = list(d["assign"])
    fams[j] = cand
    F = np.array(d["st"]["F"], float)
    F[j] = f
    w0, w, var_z = (float(v) for v in d["st"]["scalars"])
    return Joint(d["X"], d["Y"], d["train"], fams, F, np.diag(d["st"]["sigma"]), w0, w, var_z), d


def parse_f(s):
    return np.array([float(v) for v in str(s).split("|")])


# --------------------------------------------------------------------------
# 1. derivative checks + full recomputation of S and Delta
# --------------------------------------------------------------------------

def derivative_checks(rows):
    out = []
    rng = np.random.default_rng(5)
    for _, c in rows.iterrows():
        J, d = joint_for(int(c.replicate[3:]) - 1, c.regime, c.anchor, int(c.column), c.candidate, parse_f(c.f_star))
        Z = np.asarray(d["st"]["Zmode"], float) + 0.05 * rng.standard_normal((N, K))
        g = J.grad(Z)
        h = 1e-5
        idx = rng.choice(N * K, 12, replace=False)
        fd = []
        for q in idx:
            E = np.zeros(N * K)
            E[q] = h
            E = E.reshape(N, K)
            fd.append((J.phi(Z + E) - J.phi(Z - E)) / (2 * h))
        gerr = np.max(np.abs(g.ravel()[idx] - np.array(fd))) / max(1, np.max(np.abs(fd)))
        H = J.negH(Z)
        herr = 0.0
        for q in idx[:6]:
            E = np.zeros(N * K)
            E[q] = h
            E = E.reshape(N, K)
            col = -(J.grad(Z + E) - J.grad(Z - E)).ravel() / (2 * h)
            herr = max(herr, np.max(np.abs(H[:, q] - col)) / max(1, np.max(np.abs(col))))
        # CMP2 closed form vs brute-force series
        eta = np.linspace(-3, 4, 15)
        kk = np.arange(0, 400)
        brute = np.array([np.logaddexp.reduce(kk * e - 2 * gammaln(kk + 1)) for e in eta])
        out.append({"replicate": c.replicate, "regime": c.regime, "anchor": c.anchor, "column": c.column,
                    "candidate": c.candidate, "grad_rel_err": gerr, "hess_rel_err": herr,
                    "hess_symmetry": float(np.max(np.abs(H - H.T))),
                    "y_cross_block_norm": float(np.linalg.norm(H[:K, K:])),
                    "cmp2_closed_form_vs_series": float(np.max(np.abs(cmp2_A(eta) - brute)))})
    return pd.DataFrame(out)


def recompute_scores(cs: pd.DataFrame) -> pd.DataFrame:
    out = []
    t0 = time.time()
    for i, c in enumerate(cs.itertuples()):
        r = int(c.replicate[3:]) - 1
        J, d = joint_for(r, c.regime, c.anchor, int(c.column), c.candidate, parse_f(c.f_star))
        L = laplace_S(J, np.asarray(d["st"]["Zmode"], float))
        out.append({"replicate": c.replicate, "regime": c.regime, "anchor": c.anchor, "column": int(c.column),
                    "candidate": c.candidate, "S_af5": c.S, "S_indep": L["S"], "S_diff": L["S"] - c.S,
                    "phi_af5": c.phi, "phi_indep": L["phi"], "logdet_af5": c.logdet, "logdet_indep": L["logdet"],
                    "min_eig_indep": L["min_eig"], "grad_inf_indep": L["grad_inf"],
                    "x_hash_ok": d["x_ok"], "y_hash_ok": d["y_ok"], "train_hash_ok": d["train_sha_ok"],
                    "train_test_overlap": d["overlap"]})
        if i % 100 == 0:
            print(f"  scores {i}/{len(cs)} {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# 2. thermodynamic integration for Delta at the AF5 optima
# --------------------------------------------------------------------------

def hmc_chain(Jt, Zc, Lc, rng, n_warm, n_keep):
    """whitened HMC on pi_t: Z = Zc + Lc^{-T} u (Lc = chol of the Laplace precision of pi_t)."""
    Li = np.linalg.inv(Lc)          # Z - Zc = Li^T u ; grad_u = Li grad_Z
    LiT = Li.T.copy()

    def state(u):
        Z = Zc + (LiT @ u).reshape(N, K)
        lp, g, D = Jt.lgD(Z)
        return lp, Li @ g.ravel(), D

    u = np.zeros(N * K)
    lp, gu, D = state(u)
    Ds, acc = [], 0
    for it in range(n_warm + n_keep):
        p0 = rng.standard_normal(u.size)
        eps = HMC_EPS * rng.uniform(0.8, 1.2)
        un, p = u.copy(), p0 + 0.5 * eps * gu
        for s_ in range(HMC_STEPS):
            un = un + eps * p
            lpn, gun, Dn = state(un)
            if s_ < HMC_STEPS - 1:
                p = p + eps * gun
        p = p + 0.5 * eps * gun
        logacc = (lpn - 0.5 * p @ p) - (lp - 0.5 * p0 @ p0)
        if np.isfinite(logacc) and math.log(rng.uniform()) < logacc:
            u, lp, gu, D = un, lpn, gun, Dn
            if it >= n_warm:
                acc += 1
        if it >= n_warm:
            Ds.append(D)
    return np.array(Ds), acc / n_keep


def batch_se(x, nb=20):
    b = np.array_split(x, nb)
    m = np.array([v.mean() for v in b])
    return float(m.std(ddof=1) / math.sqrt(nb))


def ti_delta(r, regime, anchor, j, fP, fC, Z0, seed):
    JP, _ = joint_for(r, regime, anchor, j, "poisson", fP)
    return ti_delta_joint(JP, j, fP, fC, Z0, seed)


def ti_delta_joint(JP: Joint, j, fP, fC, Z0, seed):
    """JP: Joint whose column j is Poisson with loading fP (all else = anchor)."""
    xs, ws = np.polynomial.legendre.leggauss(GL_NODES)
    ts, ws = 0.5 * (xs + 1), 0.5 * ws
    rng = np.random.default_rng(seed)
    means, ses, accs = [], [], []
    Zc = Z0
    for t in ts:
        # Laplace of pi_t: mode + Hessian as HMC metric
        Jt = _TJoint(JP, t, j, fP, fC)
        Zc, _, _ = Jt.mode(Zc)
        Ht = Jt.negH(Zc)
        Lc = np.linalg.cholesky(0.5 * (Ht + Ht.T))
        draws, acc = [], []
        for ch in range(HMC_CHAINS):
            d, a = hmc_chain(Jt, Zc, Lc, rng, HMC_WARM, HMC_KEEP)
            draws.append(d)
            acc.append(a)
        allD = np.concatenate(draws)
        means.append(allD.mean())
        ses.append(math.sqrt(np.mean([batch_se(d) ** 2 for d in draws]) / HMC_CHAINS))
        accs.append(float(np.mean(acc)))
    means, ses = np.array(means), np.array(ses)
    return {"delta_ti": float(np.sum(ws * means)), "delta_ti_mc_se": float(math.sqrt(np.sum((ws * ses) ** 2))),
            "ti_integrand_t0": float(means[0]), "ti_integrand_t1": float(means[-1]),
            "ti_integrand_range": float(means.max() - means.min()),
            "hmc_acc_min": float(min(accs)), "hmc_acc_median": float(np.median(accs)),
            "ti_quad_check": float(np.trapezoid(np.r_[means[0], means, means[-1]], np.r_[0, ts, 1]) -
                                   np.sum(ws * means))}


class _TJoint(Joint):
    """pi_t as a Joint-like object (mode / negH) for the HMC metric."""

    def __init__(self, JP: Joint, t, j, fP, fC):
        self.__dict__.update(JP.__dict__)
        self.t, self.j, self.fP, self.fC = t, j, fP, fC

    def _extra(self, Z):
        xj = self.X[:, self.j]
        return col_terms("cmp_nu2p0", xj, Z @ self.fC), col_terms("poisson", xj, Z @ self.fP)

    def phi(self, Z):
        c, p = self._extra(Z)
        return Joint.phi(self, Z) + self.t * float(np.sum(c[0]) - np.sum(p[0]))

    def grad(self, Z):
        c, p = self._extra(Z)
        return Joint.grad(self, Z) + self.t * (np.outer(c[1], self.fC) - np.outer(p[1], self.fP))

    def lgD(self, Z):
        c, p = self._extra(Z)
        phi, g = Joint.lg(self, Z)
        D = float(np.sum(c[0]) - np.sum(p[0]))
        return phi + self.t * D, g + self.t * (np.outer(c[1], self.fC) - np.outer(p[1], self.fP)), D

    def negH(self, Z):
        c, p = self._extra(Z)
        H = Joint.negH(self, Z).reshape(N, K, N, K)
        for i in range(N):
            H[i, :, i, :] += self.t * (c[2][i] * np.outer(self.fC, self.fC) - p[2][i] * np.outer(self.fP, self.fP))
        return H.reshape(N * K, N * K)


# --------------------------------------------------------------------------
# 3. independent status / metric recomputation
# --------------------------------------------------------------------------

def status_rule(row):
    if not np.isfinite(row.delta):
        return "FAILED"
    if abs(row.delta) <= max(1e-2, row.start_dis):
        return "UNDECIDED_NUMERIC"
    if row.is_readable:
        if np.sign(row.delta + row.is_delta_correction) != np.sign(row.delta) or abs(row.delta) < 2 * row.is_delta_correction_se:
            return "UNDECIDED_LAPLACE"
        return "DECIDED"
    return "DECIDED_LAPLACE_UNASSESSED"


def boot(v, n=10000, seed=20261115):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, v.size, (n, v.size))].mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def metrics(sel: pd.DataFrame, cs: pd.DataFrame, rec: pd.DataFrame) -> dict:
    s = sel.copy()
    # independent Delta from independently recomputed S
    piv = rec.pivot_table(index=["replicate", "regime", "anchor", "column"], columns="candidate", values="S_indep")
    s = s.join((piv["cmp_nu2p0"] - piv["poisson"]).rename("delta_indep"), on=["replicate", "regime", "anchor", "column"])
    sd = cs.groupby(["replicate", "regime", "anchor", "column"]).start_disagreement.max().rename("start_dis")
    s = s.join(sd, on=["replicate", "regime", "anchor", "column"])
    s["status_indep"] = [status_rule(r) for r in s.itertuples()]
    s["sel_indep"] = np.where(s.delta_indep > 0, "cmp_nu2p0", "poisson")
    a2 = pd.read_csv(AF2_FULL / "selection_columns.csv")
    a2 = a2[(a2.method == "auto_first") & (a2.decided_by == "score")][["replicate", "regime", "column", "selected_model"]]
    q = pd.read_csv(AF3_COLS)
    q = q[(q.start.astype(str) == "zero") & (q.arm == "Z7")][["replicate", "regime", "column", "selected"]]
    p = s[s.anchor == "auto_first"].merge(a2, on=["replicate", "regime", "column"], how="left", validate="1:1") \
        .merge(q, on=["replicate", "regime", "column"], how="left", validate="1:1")
    res = {"n_rows": len(s), "status_mismatch": int((s.status_indep != s.status).sum()),
           "selection_mismatch": int((s.sel_indep != s.laplace_selected).sum()),
           "delta_max_abs_diff": float((s.delta_indep - s.delta).abs().max()),
           "primary_rows": len(p), "af2_missing": int(p.selected_model.isna().sum()), "q_missing": int(p.selected.isna().sum())}
    for reg in ("dense", "sparse"):
        g = p[p.regime == reg]
        elig = g.status_indep.isin(["DECIDED", "DECIDED_LAPLACE_UNASSESSED", "UNDECIDED_LAPLACE"])
        lap = g.sel_indep == g.true_model
        res[f"{reg}_sign_acc"] = float(lap[elig].mean())
        res[f"{reg}_q_acc"] = float((g.selected == g.true_model).mean())
        res[f"{reg}_af2_acc"] = float((g.selected_model == g.true_model).mean())
        applied = np.where(elig, g.sel_indep, g.selected_model) == g.true_model
        af2c = (g.selected_model == g.true_model).to_numpy()
        res[f"{reg}_corr_vs_af2"] = int((applied & ~af2c).sum())
        res[f"{reg}_harm_vs_af2"] = int((~applied & af2c).sum())
        dd = pd.DataFrame({"r": g.replicate, "d": applied.astype(int) - (g.selected == g.true_model).astype(int)})
        v = dd.groupby("r").d.sum().to_numpy()
        res[f"{reg}_dataset_change_vs_q_mean"] = float(v.mean())
        res[f"{reg}_dataset_change_vs_q_ci"] = boot(v)
    return res, s


def ti_summary(sel: pd.DataFrame) -> None:
    """Laplace vs TI on the Phase A subset; recomputes AF5's sparse comparison with TI signs."""
    t = pd.concat([pd.read_csv(q) for q in sorted(OUT.glob("ti_delta_auto_first*.csv"))])         .drop_duplicates(["replicate", "regime", "column"])
    t["laplace_error"] = t.delta_laplace - t.delta_ti
    t["ti_verified"] = (np.sign(t.delta_ti) == np.sign(t.delta_laplace)) & (t.delta_ti.abs() > 2 * t.delta_ti_mc_se)
    t["lap_correct"] = np.where(t.delta_laplace > 0, "cmp_nu2p0", "poisson") == t.true_model
    t["ti_correct"] = np.where(t.delta_ti > 0, "cmp_nu2p0", "poisson") == t.true_model
    t.to_csv(OUT / "ti_delta_combined.csv", index=False)
    rows = []
    for (reg, rd), g in t.groupby(["regime", "is_readable"]):
        rows.append({"regime": reg, "is_readable": rd, "n": len(g), "sign_agree": int(g.sign_agree.sum()),
                     "ti_verified": int(g.ti_verified.sum()),
                     "laplace_error_median": g.laplace_error.median(), "laplace_error_abs_median": g.laplace_error.abs().median(),
                     "laplace_error_abs_max": g.laplace_error.abs().max(), "ti_mc_se_median": g.delta_ti_mc_se.median(),
                     "ti_mc_se_max": g.delta_ti_mc_se.max(), "is_minus_ti_abs_median": (g.delta_is_corrected - g.delta_ti).abs().median() if rd else np.nan,
                     "acc_laplace": g.lap_correct.mean(), "acc_ti": g.ti_correct.mean(), "hmc_acc_min": g.hmc_acc_min.min()})
    pd.DataFrame(rows).to_csv(OUT / "ti_summary.csv", index=False)
    # sparse: AF5 primary-anchor comparison vs Q (Z7) with Laplace signs replaced by TI signs
    q = pd.read_csv(AF3_COLS)
    q = q[(q.start.astype(str) == "zero") & (q.arm == "Z7")][["replicate", "regime", "column", "selected"]]
    s = sel[(sel.anchor == "auto_first") & (sel.regime == "sparse")].merge(q, on=["replicate", "regime", "column"], validate="1:1")         .merge(t[["replicate", "regime", "column", "delta_ti"]], on=["replicate", "regime", "column"], how="left", validate="1:1")
    out = []
    for name, d in (("laplace", s.delta), ("ti", s.delta_ti)):
        ok = (np.where(d > 0, "cmp_nu2p0", "poisson") == s.true_model).astype(int)
        v = (ok - (s.selected == s.true_model).astype(int)).groupby(s.replicate).sum().to_numpy()
        lo, hi = boot(v)
        out.append({"selection": name, "regime": "sparse", "accuracy": ok.mean(), "q_accuracy": (s.selected == s.true_model).mean(),
                    "mean_D_vs_q": v.mean(), "ci_low": lo, "ci_high": hi,
                    "note": "all 120 columns, sign of Delta (UNDECIDED_NUMERIC included); differs slightly from AF5's applied rule"})
    pd.DataFrame(out).to_csv(OUT / "ti_sparse_vs_q.csv", index=False)
    print(pd.DataFrame(rows).to_string())
    print(pd.DataFrame(out).to_string())


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", action="store_true")
    ap.add_argument("--ti", action="store_true")
    ap.add_argument("--ti-anchor", default="auto_first")
    ap.add_argument("--ti-limit", type=int, default=0)
    ap.add_argument("--ti-summary", action="store_true")
    ap.add_argument("--quad-check", action="store_true", help="GL 6 vs 12 nodes on 3 fixed columns")
    ap.add_argument("--shard", default=None, help="i/n: process every n-th remaining column")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    cs = pd.read_csv(AF5 / "full" / "candidate_scores.csv")
    sel = pd.read_csv(AF5 / "full" / "column_selection.csv")
    if a.scores:
        dc = derivative_checks(cs[(cs.replicate.isin(["rep01", "rep07", "rep13"])) & (cs.column.isin([4, 7]))])
        dc.to_csv(OUT / "indep_derivative_checks.csv", index=False)
        rec = recompute_scores(cs)
        rec.to_csv(OUT / "indep_score_recomputation.csv", index=False)
        res, s = metrics(sel, cs, rec)
        s.to_csv(OUT / "indep_column_status.csv", index=False)
        (OUT / "indep_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        print(dc.describe().T[["max"]])
        print(rec[["S_diff"]].abs().describe())
    if a.ti_summary:
        ti_summary(sel)
    if a.quad_check:
        global GL_NODES
        cpiv = cs.set_index(["replicate", "regime", "anchor", "column", "candidate"])
        rows = []
        for rep, reg, j in (("rep01", "dense", 4), ("rep02", "sparse", 7), ("rep03", "sparse", 4)):
            k = (rep, reg, "auto_first", j)
            fP, fC = parse_f(cpiv.loc[k + ("poisson",), "f_star"]), parse_f(cpiv.loc[k + ("cmp_nu2p0",), "f_star"])
            r = int(rep[3:]) - 1
            Z0 = np.asarray(inputs(r, reg, "auto_first")["st"]["Zmode"], float)
            for nodes in (6, 12):
                GL_NODES = nodes
                ti = ti_delta(r, reg, "auto_first", j, fP, fC, Z0, TI_SEED + 900 + nodes)
                rows.append({"replicate": rep, "regime": reg, "column": j, "gl_nodes": nodes, **ti})
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT / "ti_quadrature_check.csv", index=False)
    if a.ti:
        rows = []
        s = sel[sel.anchor == a.ti_anchor].reset_index(drop=True)
        # subset fixed before any TI value is seen: every sparse column, and the
        # dense columns with |Delta_Laplace| < 2 (defined by Laplace, not by correctness)
        s = s[(s.regime == "sparse") | (s.delta.abs() < 2.0)].reset_index(drop=True)
        if a.ti_limit:
            s = s.head(a.ti_limit)
        cpiv = cs.set_index(["replicate", "regime", "anchor", "column", "candidate"])
        t0 = time.time()
        tag = f"_shard{a.shard.replace('/', 'of')}" if a.shard else ""
        path = OUT / f"ti_delta_{a.ti_anchor}{tag}.csv"
        done = set()
        for q in OUT.glob(f"ti_delta_{a.ti_anchor}*.csv"):
            done |= {(x["replicate"], x["regime"], x["column"]) for x in pd.read_csv(q).to_dict("records")}
        if path.exists():
            rows = pd.read_csv(path).to_dict("records")
        si, sn = (int(v) for v in a.shard.split("/")) if a.shard else (0, 1)
        for i, c in enumerate(s.itertuples()):
            if (c.replicate, c.regime, c.column) in done or i % sn != si:
                continue
            r = int(c.replicate[3:]) - 1
            k = (c.replicate, c.regime, c.anchor, c.column)
            fP, fC = parse_f(cpiv.loc[k + ("poisson",), "f_star"]), parse_f(cpiv.loc[k + ("cmp_nu2p0",), "f_star"])
            d = inputs(r, c.regime, c.anchor)
            idx = (r * 2 + ("dense", "sparse").index(c.regime)) * 10 + int(c.column)
            tt = time.time()
            ti = ti_delta(r, c.regime, c.anchor, int(c.column), fP, fC, np.asarray(d["st"]["Zmode"], float),
                          TI_SEED + idx)
            rows.append({"replicate": c.replicate, "regime": c.regime, "anchor": c.anchor, "column": c.column,
                         "true_model": c.true_model, "status": c.status, "is_readable": c.is_readable,
                         "delta_laplace": c.delta, "delta_is_corrected": c.delta_is_corrected,
                         **ti, "laplace_error": c.delta - ti["delta_ti"],
                         "sign_agree": bool(np.sign(c.delta) == np.sign(ti["delta_ti"])),
                         "seconds": time.time() - tt})
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  TI {i+1}/{len(s)} {c.replicate} {c.regime} j={c.column} lap={c.delta:.3f} "
                  f"ti={ti['delta_ti']:.3f}±{ti['delta_ti_mc_se']:.3f} acc={ti['hmc_acc_median']:.2f} "
                  f"{time.time()-tt:.0f}s total {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
