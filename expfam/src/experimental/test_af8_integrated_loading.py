"""Tests for AF8 (no MCEM)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit_af6_af5_independent as aud     # noqa: E402
import run_af8_integrated_loading as a8     # noqa: E402


def _series_A(eta, kmax=200000):
    k = np.arange(kmax)
    return float(np.logaddexp.reduce(k * eta - 2 * gammaln(k + 1)))


def test_cmp2_stable_matches_series_and_audit_path():
    eta = np.array([-5.0, -1.0, 0.0, 2.0, 6.0, 12.0, 18.0])
    brute = np.array([_series_A(e) for e in eta])
    assert np.max(np.abs(a8.cmp2_A(eta) - brute) / np.maximum(1, np.abs(brute))) < 1e-12
    safe = np.linspace(-4, 8, 25)
    assert np.max(np.abs(a8.cmp2_dA(safe) - aud.cmp2_dA(safe))) < 1e-9
    assert np.max(np.abs(a8.cmp2_d2A(safe) - aud.cmp2_d2A(safe))) < 1e-9


def test_cmp2_large_eta_finite_and_consistent():
    eta = np.array([20.0, 40.0, 51.8, 60.0, 80.0])
    u = 2 * np.exp(eta / 2)
    A, dA, d2A = a8.cmp2_A(eta), a8.cmp2_dA(eta), a8.cmp2_d2A(eta)
    assert np.all(np.isfinite(A) & np.isfinite(dA) & np.isfinite(d2A)) and np.all(d2A > 0)
    # asymptotics: A' ~ u/2 - 1/4, A'' ~ u/4
    assert np.max(np.abs(dA / (u / 2 - 0.25) - 1)) < 1e-6
    assert np.max(np.abs(d2A / (u / 4) - 1)) < 1e-3
    h = 1e-4
    for e in (20.0, 51.8):
        fd1 = (a8.cmp2_A(np.array([e + h])) - a8.cmp2_A(np.array([e - h])))[0] / (2 * h)
        fd2 = (a8.cmp2_dA(np.array([e + h])) - a8.cmp2_dA(np.array([e - h])))[0] / (2 * h)
        assert abs(fd1 / a8.cmp2_dA(np.array([e]))[0] - 1) < 1e-6
        assert abs(fd2 / a8.cmp2_d2A(np.array([e]))[0] - 1) < 1e-5
    # the series switch at u = 1e3 is continuous
    e0 = 2 * math.log(1e3 / 2)
    lo, hi = a8.cmp2_d2A(np.array([e0 - 1e-9])), a8.cmp2_d2A(np.array([e0 + 1e-9]))
    assert abs(hi[0] / lo[0] - 1) < 1e-9


def test_prior_B_is_jacobian_transformed_gaussian():
    from scipy.stats import multivariate_normal
    s = 2 / 3
    for cand in a8.CANDS:
        v = a8.prior_var("B_sg2_0.667", cand)
        a = a8.A2_ZERO[cand]
        f = np.array([0.3, -1.2, 0.7])
        g = math.sqrt(a) * f
        via_g = multivariate_normal(np.zeros(3), s * np.eye(3)).logpdf(g) + 1.5 * math.log(a)
        assert abs(a8.log_prior(f, v) - via_g) < 1e-12
        assert abs(a8.log_prior(f, v) - multivariate_normal(np.zeros(3), v * np.eye(3)).logpdf(f)) < 1e-12
    assert abs(a8.A2_ZERO["cmp_nu2p0"] - 0.5131105267032117) < 1e-12


def test_joint_gradient_and_full_hessian_by_finite_differences():
    rng = np.random.default_rng(3)
    n, k = 12, 3
    Z = rng.standard_normal((n, k))
    F = rng.standard_normal((5, k)) * 0.6
    fams = ["gaussian", "poisson", "poisson", "cmp_nu2p0", "cmp_nu2p0"]
    eta = Z @ F.T
    X = np.column_stack([eta[:, 0] + rng.standard_normal(n), rng.poisson(np.exp(eta[:, 1])),
                         rng.poisson(np.exp(eta[:, 2])), rng.poisson(np.exp(eta[:, 3] / 2)), rng.poisson(np.exp(eta[:, 4] / 2))])
    Y = (rng.random((n, n)) < 0.4).astype(float)
    Y = np.triu(Y, 1) + np.triu(Y, 1).T
    train = rng.random((n, n)) < 0.7
    for j, fam in ((3, "cmp_nu2p0"), (2, "poisson")):
        J = a8.JointU(X, Y, train, fams, F, np.ones(5), -1.0, 0.9, 1.0, j, 0.5)
        Z0, f0 = Z + 0.1, F[j] + 0.2
        u0 = np.concatenate([Z0.ravel(), f0])
        def split(u):
            return u[:n * k].reshape(n, k), u[n * k:]
        g = np.concatenate([a.ravel() for a in J.grad_u(Z0, f0)])
        H = J.negH_u(Z0, f0)
        h = 1e-6
        for q in range(len(u0)):
            e = np.zeros(len(u0)); e[q] = h
            fd = (J.psi(*split(u0 + e)) - J.psi(*split(u0 - e))) / (2 * h)
            assert abs(fd - g[q]) < 1e-5 * max(1, abs(g[q]))
            gp = np.concatenate([a.ravel() for a in J.grad_u(*split(u0 + e))])
            gm = np.concatenate([a.ravel() for a in J.grad_u(*split(u0 - e))])
            assert np.max(np.abs(-(gp - gm) / (2 * h) - H[:, q])) < 1e-5 * max(1, np.max(np.abs(H[:, q])))
        assert np.max(np.abs(H - H.T)) < 1e-12


def test_protocol_fixed_settings():
    p = a8.protocol()
    assert p["prior_main"] == "B_sg2_0.667" and p["mcem_fits"] == 0
    assert set(p["priors"]) == set(a8.PRIORS)
