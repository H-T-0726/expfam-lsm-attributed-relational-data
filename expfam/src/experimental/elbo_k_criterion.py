"""C_ELBO(q_Lap): ELBO-based K-selection evaluator with the Laplace Gaussian as q.

Research evaluator added 2026-11-08 (criterion impact validation).  It does
NOT modify ``laplace_k_criterion`` or any estimator; it only reads the joint
mode Z_hat and the joint negative Hessian H that the existing Laplace
evaluator defines, and evaluates

    q_Lap(Z)          = N(vec Z_hat, H^{-1})            (nK-dimensional)
    L(q_Lap, theta)   = E_q[Phi(Z)] + H(q_Lap)
    H(q_Lap)          = (nK/2)(1 + log 2 pi) - (1/2) log|H|
    C_ELBO(K)         = -2 L(q_Lap, theta) + d_K log N
    R3                = Phi(Z_hat) - nK/2 - E_q[Phi]
    identity          C_ELBO = C_Lap + 2 R3   (same theta, Z_hat, H, penalty)

Phi(Z) = log p(Z, X, Y | theta) with every normalising constant (the same
quantity as ``laplace_k_criterion.complete_log_density``).

How E_q[Phi] is computed (the target quantity is unchanged):
  * prior, X part (Gaussian or Poisson columns) and the linear Y part
    y_ij * E[eta_ij] are Gaussian expectations evaluated in closed form from
    the mean Z_hat and the covariance Sigma = H^{-1};
  * only E[ sum_pairs softplus(eta_ij) ] (Bernoulli Y) is estimated by Monte
    Carlo with INDEPENDENT draws Z = Z_hat + L^{-T} eps, H = L L^T (Cholesky).
    The E-step samples of the MCEM fit are never used.
The Monte Carlo standard error of C_ELBO is 2 * sd(S) / sqrt(M) where S is the
per-draw softplus sum (draws are independent).

If H is not positive definite or the joint mode is not stationary, the
existing Laplace status is returned unchanged and no value is produced (no
ridge, no imputation).

Supported: DualExpFamLSMConsistent with family_x in {gaussian, poisson},
family_y = bernoulli, symmetric pair mask with zero diagonal.

Lineage E (experimental prototype).  C_ELBO is an existing variational lower
bound based criterion, NOT a new BIC and NOT Schwarz BIC.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import gammaln

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import laplace_k_criterion as lk  # noqa: E402

EVALUATOR_VERSION = "elbo-k-criterion-v1"
LN2PI = math.log(2.0 * math.pi)


def _softplus(x):
    return np.logaddexp(0.0, x)


@dataclass
class ElboResult:
    status: str
    n: int
    k: int
    phi_at_mode: float
    logdet_H: float
    entropy_q: float
    e_q_phi: float
    e_q_phi_mcse: float
    elbo: float
    R3: float
    analytic_part: float
    mc_mean_softplus: float
    mc_sd_softplus: float
    n_samples: int
    batch_means: list = field(default_factory=list)
    seconds: float = float("nan")
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == lk.STATUS_OK


def _check(model):
    if model.family != "bernoulli":
        raise lk.UnsupportedLaplacePath("elbo evaluator: Bernoulli Y only")
    if model.family_x not in ("gaussian", "poisson"):
        raise lk.UnsupportedLaplacePath("elbo evaluator: Gaussian/Poisson X only")


def phi_point(model, X, Y, Z) -> float:
    """Phi(Z) written from the formulas (used to cross-check the analytic split)."""
    t = lk._theta(model)
    F, w0, w, vz = t["F"], t["w0"], t["w"], t["var_z"]
    n, k = Z.shape
    out = -0.5 * n * k * math.log(2 * math.pi * vz) - 0.5 * float(np.sum(Z ** 2)) / vz
    eta = Z @ F.T
    if model.family_x == "gaussian":
        s = np.diag(t["sigma"])
        out += float(np.sum(-0.5 * np.log(2 * math.pi * s) - 0.5 * (X - eta) ** 2 / s))
    else:
        out += float(np.sum(X * eta - np.exp(eta) - gammaln(X + 1.0)))
    up = np.triu(model._mask_f > 0, 1)
    e = (w0 + w * (Z @ Z.T))[up]
    out += float(np.sum(Y[up] * e - _softplus(e)))
    return out


def elbo_from_mode(model, X, Y, lap: "lk.LaplaceResult", *, n_samples: int, n_batches: int,
                   seed_entropy) -> ElboResult:
    """ELBO with q = N(Z_hat, H^{-1}) at the (already verified) Laplace mode."""
    _check(model)
    t0 = time.perf_counter()
    n, k = lap.n, lap.k
    if not lap.ok:
        return ElboResult(lap.status, n, k, lap.phi_at_mode, lap.logdet_H, *([float("nan")] * 8), 0,
                          notes=["Laplace status not OK: no value (no ridge, no imputation)"])
    t = lk._theta(model)
    F, w0, w, vz = t["F"], t["w0"], t["w"], t["var_z"]
    Zh = lap.Z_hat
    H = lk.negative_hessian(model, X, Y, Zh)
    H = 0.5 * (H + H.T)
    c, low = cho_factor(H, lower=True)
    logdet = 2.0 * float(np.sum(np.log(np.diag(c))))
    if not np.isclose(logdet, lap.logdet_H, rtol=1e-9, atol=1e-7):
        raise AssertionError(f"logdet mismatch {logdet} vs {lap.logdet_H}")
    Sig = cho_solve((c, low), np.eye(n * k))
    Sig = 0.5 * (Sig + Sig.T)
    S4 = Sig.reshape(n, k, n, k)
    Sii = np.einsum("iaib->iab", S4)                       # (n, k, k)
    # prior
    e_prior = -0.5 * n * k * math.log(2 * math.pi * vz) - 0.5 * (float(np.sum(Zh ** 2)) + float(np.trace(Sig))) / vz
    # X part
    m_eta = Zh @ F.T                                       # (n, d)
    v_eta = np.einsum("la,iab,lb->il", F, Sii, F)          # Var(f_l' z_i)
    if model.family_x == "gaussian":
        s = np.diag(t["sigma"])
        e_x = float(np.sum(-0.5 * np.log(2 * math.pi * s) - 0.5 * ((X - m_eta) ** 2 + v_eta) / s))
    else:
        e_x = float(np.sum(X * m_eta - np.exp(m_eta + 0.5 * v_eta) - gammaln(X + 1.0)))
    # Y linear part
    up = np.triu(model._mask_f > 0, 1)
    tr_ij = np.einsum("iaja->ij", S4)                      # E[z_i'z_j] - zh_i'zh_j
    e_eta = w0 + w * (Zh @ Zh.T + tr_ij)
    e_ylin = float(np.sum(Y[up] * e_eta[up]))
    analytic = e_prior + e_x + e_ylin
    # Monte Carlo: E[sum softplus(eta)] with independent draws from q
    rows, cols = np.where(up)
    U = c if low else c.T                                  # lower Cholesky L, H = L L'
    U = np.tril(U)
    per = n_samples // n_batches
    if per * n_batches != n_samples:
        raise ValueError("n_samples must be a multiple of n_batches")
    vals, bmeans = [], []
    for b in range(n_batches):
        rng = np.random.default_rng(list(seed_entropy) + [b])
        eps = rng.standard_normal((n * k, per))
        D = solve_triangular(U.T, eps, lower=False)        # Cov = (L L')^{-1}
        bv = []
        for m in range(per):
            Zs = Zh + D[:, m].reshape(n, k)
            e = w0 + w * np.einsum("pa,pa->p", Zs[rows], Zs[cols])
            bv.append(float(np.sum(_softplus(e))))
        vals.extend(bv)
        bmeans.append(float(np.mean(bv)))
    vals = np.asarray(vals)
    if not np.all(np.isfinite(vals)):
        return ElboResult("NONFINITE_MC", n, k, lap.phi_at_mode, logdet, *([float("nan")] * 8), len(vals),
                          notes=["non-finite softplus sum"])
    mean_sp = float(vals.mean())
    sd_sp = float(vals.std(ddof=1))
    e_q_phi = analytic - mean_sp
    mcse = sd_sp / math.sqrt(len(vals))
    ent = 0.5 * n * k * (1.0 + LN2PI) - 0.5 * logdet
    elbo = e_q_phi + ent
    r3 = lap.phi_at_mode - 0.5 * n * k - e_q_phi
    return ElboResult(lk.STATUS_OK, n, k, lap.phi_at_mode, logdet, ent, e_q_phi, mcse, elbo, r3, analytic,
                      mean_sp, sd_sp, len(vals), bmeans, time.perf_counter() - t0)


def calc_C_ELBO(res: ElboResult, *, d_K: int, N: float) -> dict:
    pen = float(d_K) * math.log(N)
    return {"C_ELBO": -2.0 * res.elbo + pen if res.ok else float("nan"),
            "C_ELBO_mcse": 2.0 * res.e_q_phi_mcse if res.ok else float("nan"),
            "parameter_term": pen}
