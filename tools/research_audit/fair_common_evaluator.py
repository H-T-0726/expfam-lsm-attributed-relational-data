"""Estimator-independent common evaluator (fair comparison, 2026-11-06).

Written from the formulas, not from any estimator class.  Every function is a
deterministic function of (X, Y, mask, Z, theta).  Densities include all
normalising constants.

Model (canonical, both the paper's Gaussian-X / Bernoulli-Y model and the
ExpFam model with those families):
    z_i ~ N(0, var_z I_K)
    x_il | z_i ~ N(f_l' z_i, s_l)            (Gaussian-X, s_l = Sigma_ll)
    x_il | z_i ~ Poisson(exp(f_l' z_i))      (Poisson-X)
    y_ij | z_i, z_j ~ Bern(sigmoid(w0 + w z_i' z_j)),  i < j, (i, j) in mask
Paper Eq.(16) writes the Y factor as {prod_{i != j} p(y_ij)}^{1/2}; for a
symmetric Y this equals the product over unordered pairs (DERIVED; tested).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

LN2PI = math.log(2.0 * math.pi)


def log_prior(Z, var_z=1.0):
    n, k = Z.shape
    return float(-0.5 * n * k * math.log(2 * math.pi * var_z) - 0.5 * np.sum(Z ** 2) / var_z)


def loglik_x(X, Z, F, family, sigma_diag=None):
    eta = Z @ F.T
    if family == "gaussian":
        s = np.asarray(sigma_diag, float)
        return float(np.sum(-0.5 * (LN2PI + np.log(s)) - 0.5 * (X - eta) ** 2 / s))
    if family == "poisson":
        return float(np.sum(X * eta - np.exp(eta) - gammaln(X + 1.0)))
    raise ValueError(family)


def loglik_y(Y, Z, w0, w, mask=None):
    """Bernoulli over unordered pairs i<j inside the mask (diagonal never used)."""
    n = Z.shape[0]
    up = np.triu(np.ones((n, n), dtype=bool), 1)
    if mask is not None:
        up &= np.asarray(mask, bool)
    eta = (w0 + w * (Z @ Z.T))[up]
    y = Y[up]
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def loglik_y_paper_eq16(Y, Z, w0, w):
    """Paper Eq.(16) literally: 1/2 * sum over ORDERED pairs i != j."""
    n = Z.shape[0]
    S = 1.0 / (1.0 + np.exp(-(w0 + w * (Z @ Z.T))))
    lp = Y * np.log(S) + (1 - Y) * np.log(1 - S)
    np.fill_diagonal(lp, 0.0)
    return float(0.5 * lp.sum())


def loglik_x_paper(X, Z, F, sigma_diag, mu=None):
    """Paper Eq.(16)/(B-1): x_i ~ N(F z_i + mu, Sigma)."""
    mu = np.zeros(X.shape[1]) if mu is None else mu
    s = np.asarray(sigma_diag, float)
    R = X - Z @ F.T - mu
    n, d = X.shape
    return float(-0.5 * n * d * LN2PI - 0.5 * n * np.sum(np.log(s)) - 0.5 * np.sum(R ** 2 / s))


def loglik_x_expfam_gauss(X, Z, F, sigma_diag):
    """Same Gaussian written in exponential-family form: T(x)=x, eta=f'z, phi=s, A(eta)=eta^2/2."""
    s = np.asarray(sigma_diag, float)
    eta = Z @ F.T
    h = -0.5 * X ** 2 / s - 0.5 * (LN2PI + np.log(s))
    return float(np.sum((X * eta - 0.5 * eta ** 2) / s + h))


def data_loglik(X, Y, Z, theta, family_x, mask=None):
    return loglik_x(X, Z, theta["F"], family_x, theta.get("sigma_diag")) + loglik_y(Y, Z, theta["w0"], theta["w"], mask)


def q_terms(X, Y, Zs, theta, family_x, mask=None):
    L = Zs.shape[2]
    qx = [loglik_x(X, Zs[:, :, l], theta["F"], family_x, theta.get("sigma_diag")) for l in range(L)]
    qy = [loglik_y(Y, Zs[:, :, l], theta["w0"], theta["w"], mask) for l in range(L)]
    qz = [log_prior(Zs[:, :, l], theta.get("var_z", 1.0)) for l in range(L)]
    return {"Q_X": float(np.mean(qx)), "Q_Y": float(np.mean(qy)), "Q_Z": float(np.mean(qz)),
            "ll_samples": np.array(qx) + np.array(qy)}


def num_params(k, d, family_x):
    return k * d - k * (k - 1) // 2 + (d if family_x == "gaussian" else 0)


def criteria_abc(X, Y, Zs, Z_last, theta, family_x, mask=None):
    n, d = X.shape
    k = Zs.shape[1]
    p = num_params(k, d, family_x) * math.log(n)
    q = q_terms(X, Y, Zs, theta, family_x, mask)
    ll_last = data_loglik(X, Y, Z_last, theta, family_x, mask)
    L = Zs.shape[2]
    return {"P_theta": p, "Q_X": q["Q_X"], "Q_Y": q["Q_Y"], "Q_Z": q["Q_Z"], "P_Z": -2 * q["Q_Z"],
            "A_CQ": -2 * (q["Q_X"] + q["Q_Y"] + q["Q_Z"]) + p, "B_CQ_minus_PZ": -2 * (q["Q_X"] + q["Q_Y"]) + p,
            "C_cond_last": -2 * ll_last + p, "mcse_B": 2 * float(np.std(q["ll_samples"], ddof=1)) / math.sqrt(L),
            "sd_single_sample_C": 2 * float(np.std(q["ll_samples"], ddof=1))}


# ---------------------------------------------------------------- derivatives (adopted formula)

def grad_zi(X, Y, Z, theta, family_x, i, mask=None, var_z=1.0):
    """d/dz_i of log p(Z, X, Y) with unordered pairs (no extra 1/2)."""
    F, w0, w = theta["F"], theta["w0"], theta["w"]
    zi = Z[i]
    g = -zi / var_z
    eta_x = F @ zi
    if family_x == "gaussian":
        g = g + F.T @ ((X[i] - eta_x) / theta["sigma_diag"])
    else:
        g = g + F.T @ (X[i] - np.exp(eta_x))
    s = 1.0 / (1.0 + np.exp(-(w0 + w * (Z @ zi))))
    r = Y[i] - s
    r[i] = 0.0
    if mask is not None:
        r = r * mask[i]
    return g + w * (Z.T @ r)


def neg_hess_zi(X, Y, Z, theta, family_x, i, mask=None, var_z=1.0):
    F, w0, w = theta["F"], theta["w0"], theta["w"]
    zi = Z[i]
    k = Z.shape[1]
    H = np.eye(k) / var_z
    if family_x == "gaussian":
        H = H + F.T @ (F / theta["sigma_diag"][:, None])
    else:
        H = H + F.T @ (F * np.exp(F @ zi)[:, None])
    s = 1.0 / (1.0 + np.exp(-(w0 + w * (Z @ zi))))
    v = s * (1 - s)
    v[i] = 0.0
    if mask is not None:
        v = v * mask[i]
    return H + w ** 2 * (Z.T * v) @ Z
