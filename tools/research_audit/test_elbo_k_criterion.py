"""Tests for expfam/src/experimental/elbo_k_criterion.py (no MCEM fit).

Run:  python -m pytest -q tools/research_audit/test_elbo_k_criterion.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "expfam/src/experimental", ROOT / "reproduction/src"):
    sys.path.insert(0, str(p))

import elbo_k_criterion as ek  # noqa: E402
import em_runner  # noqa: E402
import laplace_k_criterion as lk  # noqa: E402

LN2PI = math.log(2 * math.pi)


def make(n, d, k, xfam, w0=-0.5, w=0.6, seed=0, frac=0.8):
    rng = np.random.default_rng(seed)
    Zt = rng.standard_normal((n, k))
    F = rng.normal(scale=0.4, size=(d, k))
    if xfam == "gaussian":
        s = np.full(d, 0.5)
        X = Zt @ F.T + rng.standard_normal((n, d)) * np.sqrt(s)
    else:
        s = np.ones(d)
        X = rng.poisson(np.exp(Zt @ F.T)).astype(float)
    P = 1 / (1 + np.exp(-(w0 + w * Zt @ Zt.T)))
    U = np.triu(rng.random((n, n)) < P, 1).astype(float)
    Y = U + U.T
    up = np.argwhere(np.triu(np.ones((n, n), bool), 1))
    keep = rng.random(len(up)) < frac
    M = np.zeros((n, n), bool)
    M[up[keep, 0], up[keep, 1]] = True
    M = M | M.T
    m = em_runner.build_model(n, d, k, 5, xfam, "bernoulli", train_mask=M, numerics_mode="consistent")
    m.initialize_params(true_params=None, seed=0)
    m.params.update({"Z": Zt.copy(), "F": F, "sigma": np.diag(s), "w0": w0, "w": w, "var_z": 1.0})
    return m, X, Y, Zt


@pytest.mark.parametrize("xfam", ["gaussian", "poisson"])
def test_phi_point_matches_complete_log_density(xfam):
    m, X, Y, Zt = make(12, 4, 2, xfam)
    for s in range(3):
        Z = np.random.default_rng(s).standard_normal(Zt.shape)
        assert ek.phi_point(m, X, Y, Z) == pytest.approx(lk.complete_log_density(m, X, Y, Z), abs=1e-8)


@pytest.mark.parametrize("xfam", ["gaussian", "poisson"])
def test_identity_and_entropy(xfam):
    m, X, Y, Zt = make(15, 4, 2, xfam)
    lap = lk.laplace_log_observed(m, X, Y, Zt.copy())
    assert lap.ok
    r = ek.elbo_from_mode(m, X, Y, lap, n_samples=256, n_batches=4, seed_entropy=[1, 2])
    nk = 30
    assert r.entropy_q == pytest.approx(0.5 * nk * (1 + LN2PI) - 0.5 * lap.logdet_H, abs=1e-10)
    c_lap = lk.calc_C_Lap(lap, d_K=7, N=15.0)["C_Lap"]
    c_elbo = ek.calc_C_ELBO(r, d_K=7, N=15.0)["C_ELBO"]
    assert c_elbo == pytest.approx(c_lap + 2 * r.R3, abs=1e-8)


def test_gaussian_w0_only_exact():
    """Gaussian X with w = 0: Phi quadratic in Z -> ELBO = l_Lap = exact marginal, R3 = 0."""
    n, d, k = 10, 4, 2
    m, X, Y, Zt = make(n, d, k, "gaussian", w=0.0)
    lap = lk.laplace_log_observed(m, X, Y, Zt.copy())
    r = ek.elbo_from_mode(m, X, Y, lap, n_samples=64, n_batches=2, seed_entropy=[3])
    F, s = m.params["F"], np.diag(m.params["sigma"])
    C = F @ F.T + np.diag(s)
    sign, ld = np.linalg.slogdet(C)
    lx = float(-0.5 * n * (d * LN2PI + ld) - 0.5 * np.einsum("ia,ab,ib->", X, np.linalg.inv(C), X))
    up = np.triu(m._mask_f > 0, 1)
    w0 = m.params["w0"]
    ly = float(np.sum(Y[up] * w0 - np.logaddexp(0, w0)))
    assert lap.log_observed == pytest.approx(lx + ly, abs=1e-8)
    assert r.elbo == pytest.approx(lx + ly, abs=1e-8)
    assert r.R3 == pytest.approx(0.0, abs=1e-8)
    assert r.mc_sd_softplus == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("xfam", ["gaussian", "poisson"])
def test_semi_analytic_matches_brute_force_mc(xfam):
    """Same target: semi-analytic E_q Phi vs plain MC of Phi with q draws."""
    m, X, Y, Zt = make(10, 3, 2, xfam, w=0.8)
    lap = lk.laplace_log_observed(m, X, Y, Zt.copy())
    r = ek.elbo_from_mode(m, X, Y, lap, n_samples=4000, n_batches=4, seed_entropy=[5])
    H = lk.negative_hessian(m, X, Y, lap.Z_hat)
    Sig = np.linalg.inv(0.5 * (H + H.T))
    rng = np.random.default_rng(99)
    D = rng.multivariate_normal(np.zeros(20), Sig, size=20000)
    vals = np.array([ek.phi_point(m, X, Y, lap.Z_hat + D[j].reshape(10, 2)) for j in range(len(D))])
    se = math.hypot(vals.std(ddof=1) / math.sqrt(len(vals)), r.e_q_phi_mcse)
    assert abs(vals.mean() - r.e_q_phi) < 5 * se


def test_sampling_covariance_is_H_inverse():
    m, X, Y, Zt = make(8, 3, 2, "poisson")
    lap = lk.laplace_log_observed(m, X, Y, Zt.copy())
    H = lk.negative_hessian(m, X, Y, lap.Z_hat)
    H = 0.5 * (H + H.T)
    from scipy.linalg import cho_factor, solve_triangular
    c, low = cho_factor(H, lower=True)
    Lc = np.tril(c)
    eps = np.random.default_rng(0).standard_normal((16, 200000))
    D = solve_triangular(Lc.T, eps, lower=False)
    emp = np.cov(D)
    assert np.max(np.abs(emp - np.linalg.inv(H))) < 0.02 * np.max(np.abs(np.linalg.inv(H))) + 5e-3


def test_elbo_is_below_exact_marginal_tiny():
    """n=3, K=1: exact marginal by 3-D grid quadrature; ELBO <= l_marg (up to MC error)."""
    m, X, Y, Zt = make(3, 2, 1, "poisson", w=0.9, frac=1.0)
    g = np.linspace(-7, 7, 141)
    h = g[1] - g[0]
    A, B, Cc = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([A.ravel(), B.ravel(), Cc.ravel()], 1)
    vals = np.array([ek.phi_point(m, X, Y, p.reshape(3, 1)) for p in pts])
    mx = vals.max()
    l_marg = mx + math.log(np.sum(np.exp(vals - mx)) * h ** 3)
    lap = lk.laplace_log_observed(m, X, Y, Zt.copy())
    r = ek.elbo_from_mode(m, X, Y, lap, n_samples=20000, n_batches=4, seed_entropy=[7])
    assert r.elbo <= l_marg + 4 * r.e_q_phi_mcse


def test_not_ok_status_passes_through():
    m, X, Y, Zt = make(10, 3, 2, "poisson")
    lap = lk.laplace_log_observed(m, X, Y, Zt.copy(), max_iter=0)
    assert not lap.ok
    r = ek.elbo_from_mode(m, X, Y, lap, n_samples=8, n_batches=2, seed_entropy=[1])
    assert not r.ok and math.isnan(r.elbo)
    assert math.isnan(ek.calc_C_ELBO(r, d_K=3, N=10.0)["C_ELBO"])
