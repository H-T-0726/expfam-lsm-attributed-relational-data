"""Zero-EM math checks for the Phase 9H Candidate-B Laplace evaluator.

Small fixed arrays, no fitting, no sampling. Real EM entry points are
replaced by functions that fail the test if reached.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import laplace_k_criterion as lk                                  # noqa: E402
from model_dual_expfam_consistent import (                        # noqa: E402
    DualExpFamLSMConsistent,
    DualExpFamLSMPerColumnConsistent,
)


@pytest.fixture(autouse=True)
def _forbid_em(monkeypatch):
    import em_runner

    def _no_em(*_a, **_k):
        raise AssertionError("EM or an E-step was reached in a zero-EM test")

    monkeypatch.setattr(em_runner, "run_em_experimental", _no_em)
    monkeypatch.setattr(DualExpFamLSMPerColumnConsistent, "calc_eta_newton",
                        _no_em)
    monkeypatch.setattr(DualExpFamLSMConsistent, "calc_eta_newton", _no_em)


def _model(families, family_y, k, F, sigma_diag, w0, w, var_z=1.0, n=None):
    n = n or 6
    model = DualExpFamLSMPerColumnConsistent(
        n=n, d=len(families), k=k, L=1, family_x_list=list(families),
        family_y=family_y)
    model.initialize_params(true_params=None, seed=0)
    model.params.update({"F": np.array(F, float),
                         "sigma": np.diag(np.array(sigma_diag, float)),
                         "w0": float(w0), "w": float(w), "var_z": float(var_z)})
    return model


def _symmetric_binary(rng, n, p=0.4):
    upper = np.triu((rng.random((n, n)) < p).astype(float), k=1)
    return upper + upper.T


def _mixed_case(family_y="bernoulli", n=6, k=2, seed=11):
    """Mixed X (G, B, P, G) with relational Y; w clearly nonzero."""

    rng = np.random.default_rng(seed)
    families = ["gaussian", "bernoulli", "poisson", "gaussian"]
    F = rng.normal(scale=0.7, size=(4, k))
    Z = rng.normal(size=(n, k))
    X = np.column_stack([rng.normal(size=n),
                         (rng.random(n) < 0.5).astype(float),
                         rng.poisson(1.5, size=n).astype(float),
                         rng.normal(size=n)])
    if family_y == "bernoulli":
        Y = _symmetric_binary(rng, n)
    else:
        upper = np.triu(rng.poisson(1.0, size=(n, n)).astype(float), k=1)
        Y = upper + upper.T
    model = _model(families, family_y, k, F, [0.8, 1.0, 1.0, 1.7],
                   w0=-0.3, w=0.9, n=n)
    return model, X, Y, Z


# --------------------------------------------------------------------------
# A. gradient by central differences
# --------------------------------------------------------------------------

# Central differences with step h have truncation error O(h² |Phi'''|) and
# rounding error O(eps |Phi| / h). With h = 1e-5, |Phi| ~ 1e2 and
# eps ~ 2.2e-16 both are ~1e-9, so 1e-6 (absolute, on O(1) gradient entries)
# leaves three orders of margin while still catching any missing term.
GRAD_H = 1e-5
GRAD_TOL = 1e-6


def _fd_gradient(model, X, Y, Z, h=GRAD_H):
    g = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        for a in range(Z.shape[1]):
            plus, minus = Z.copy(), Z.copy()
            plus[i, a] += h
            minus[i, a] -= h
            g[i, a] = (lk.complete_log_density(model, X, Y, plus)
                       - lk.complete_log_density(model, X, Y, minus)) / (2 * h)
    return g


@pytest.mark.parametrize("family_y", ["bernoulli", "poisson"])
def test_gradient_matches_central_differences(family_y):
    model, X, Y, Z = _mixed_case(family_y)
    analytic = lk.gradient(model, X, Y, Z)
    numeric = _fd_gradient(model, X, Y, Z)
    err = np.abs(analytic - numeric)
    max_abs = float(err.max())
    max_rel = float((err / np.maximum(np.abs(numeric), 1.0)).max())
    print(f"[grad {family_y}] max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
          f"(nK={Z.size})")
    assert max_abs < GRAD_TOL and max_rel < GRAD_TOL


# --------------------------------------------------------------------------
# B. full Hessian by central differences of the analytic gradient
# --------------------------------------------------------------------------

# Differencing the analytic gradient: truncation O(h²) ~ 1e-12 and rounding
# O(eps |grad| / h) ~ 1e-10 for h = 1e-6; tolerance 1e-6 as above.
HESS_H = 1e-6
HESS_TOL = 1e-6


def _fd_hessian(model, X, Y, Z, h=HESS_H):
    n, k = Z.shape
    H = np.zeros((n * k, n * k))
    for col in range(n * k):
        plus, minus = Z.copy().ravel(), Z.copy().ravel()
        plus[col] += h
        minus[col] -= h
        gp = lk.gradient(model, X, Y, plus.reshape(n, k)).ravel()
        gm = lk.gradient(model, X, Y, minus.reshape(n, k)).ravel()
        H[:, col] = -(gp - gm) / (2 * h)
    return H


@pytest.mark.parametrize("family_y", ["bernoulli", "poisson"])
def test_full_hessian_matches_differences_and_is_symmetric(family_y):
    model, X, Y, Z = _mixed_case(family_y)
    n, k = Z.shape
    analytic = lk.negative_hessian(model, X, Y, Z)
    numeric = _fd_hessian(model, X, Y, Z)
    err = np.abs(analytic - numeric)
    max_abs = float(err.max())
    max_rel = float((err / np.maximum(np.abs(numeric), 1.0)).max())
    asym = float(np.abs(analytic - analytic.T).max())
    blocks = analytic.reshape(n, k, n, k).transpose(0, 2, 1, 3)
    off = max(float(np.abs(blocks[i, j]).max())
              for i in range(n) for j in range(n) if i != j)
    transpose_gap = max(float(np.abs(blocks[i, j] - blocks[j, i].T).max())
                        for i in range(n) for j in range(n))
    print(f"[hess {family_y}] max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
          f"asym={asym:.1e} max|H_ij| (i!=j)={off:.3f}")
    assert max_abs < HESS_TOL and max_rel < HESS_TOL
    assert off > 1e-2                     # relational off-diagonal blocks exist
    assert asym < 1e-12 and transpose_gap < 1e-12


def test_offdiagonal_block_formula_bernoulli():
    model, X, Y, Z = _mixed_case("bernoulli")
    n, k = Z.shape
    w, w0 = model.params["w"], model.params["w0"]
    H = lk.negative_hessian(model, X, Y, Z).reshape(n, k, n, k)
    i, j = 0, 3
    eta = w0 + w * Z[i] @ Z[j]
    mu = 1.0 / (1.0 + math.exp(-eta))
    expected = (w * w * mu * (1 - mu) * np.outer(Z[j], Z[i])
                - w * (Y[i, j] - mu) * np.eye(k))
    assert np.allclose(H[i, :, j, :], expected, atol=1e-13)


# --------------------------------------------------------------------------
# C. Gaussian X, w = 0: Laplace equals the exact marginal
# --------------------------------------------------------------------------

def _gaussian_case(k, var_z=1.0, scale=1.0, n=8, seed=5):
    rng = np.random.default_rng(seed)
    d = 4
    F = rng.normal(scale=0.9, size=(d, k)) / scale
    sigma_diag = np.array([0.5, 1.0, 1.4, 0.8])
    X = rng.normal(size=(n, d)) * 1.3
    Y = _symmetric_binary(rng, n)
    model = _model(["gaussian"] * d, "bernoulli", k, F, sigma_diag,
                   w0=-0.4, w=0.0, var_z=var_z, n=n)
    return model, X, Y, F, sigma_diag


def _exact_log_pXY(X, Y, F, sigma_diag, w0, var_z=1.0):
    cov = var_z * F @ F.T + np.diag(sigma_diag)
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0
    n, d = X.shape
    quad = np.einsum("il,lm,im->", X, np.linalg.inv(cov), X)
    log_px = -0.5 * (n * d * math.log(2 * math.pi) + n * logdet + quad)
    upper = np.triu_indices(Y.shape[0], k=1)
    log_py = float(np.sum(Y[upper] * w0 - np.logaddexp(0.0, w0)))
    return log_px + log_py


@pytest.mark.parametrize("k", [1, 2, 3])
def test_laplace_is_exact_for_gaussian_x_with_w0(k):
    model, X, Y, F, sd = _gaussian_case(k)
    result = lk.laplace_log_observed(model, X, Y, np.zeros((X.shape[0], k)))
    exact = _exact_log_pXY(X, Y, F, sd, w0=-0.4)
    gap = abs(result.log_observed - exact)
    print(f"[exact K={k}] laplace={result.log_observed:.12f} "
          f"exact={exact:.12f} |diff|={gap:.2e} grad_inf={result.grad_inf:.1e}")
    assert result.ok
    assert gap < 1e-9 * max(1.0, abs(exact))


# --------------------------------------------------------------------------
# D. latent-scale invariance
# --------------------------------------------------------------------------

@pytest.mark.parametrize("s", [0.5, 3.0])
def test_scale_invariance_gaussian_exact_case(s):
    base = _gaussian_case(2)
    scaled = _gaussian_case(2, var_z=s * s, scale=s)
    r0 = lk.laplace_log_observed(base[0], base[1], base[2], np.zeros((8, 2)))
    r1 = lk.laplace_log_observed(scaled[0], scaled[1], scaled[2],
                                 np.zeros((8, 2)))
    assert r0.ok and r1.ok
    assert abs(r0.log_observed - r1.log_observed) < 1e-9 * abs(r0.log_observed)


def _scaled_mixed(model, s):
    out = _model(model.family_x_list, model.family, model.k,
                 model.params["F"] / s, np.diag(model.params["sigma"]),
                 w0=model.params["w0"], w=model.params["w"] / s ** 2,
                 var_z=s * s, n=model.n)
    return out


@pytest.mark.parametrize("s", [0.5, 2.0])
def test_scale_invariance_relational_case(s):
    """Mixed X + Bernoulli Y, w != 0: value and K difference unchanged."""

    values = {}
    for k in (1, 2):
        model, X, Y, Z = _mixed_case("bernoulli", k=k)
        r0 = lk.laplace_log_observed(model, X, Y, 0.1 * Z)
        r1 = lk.laplace_log_observed(_scaled_mixed(model, s), X, Y,
                                     0.1 * s * Z)
        assert r0.ok and r1.ok, (r0.notes, r1.notes)
        gap = abs(r0.log_observed - r1.log_observed)
        print(f"[scale s={s} K={k}] {r0.log_observed:.10f} vs "
              f"{r1.log_observed:.10f} |diff|={gap:.2e}")
        assert gap < 1e-8 * abs(r0.log_observed)
        # the mode maps as Z_hat' = s Z_hat
        assert np.allclose(r1.Z_hat, s * r0.Z_hat, atol=1e-7)
        values[k] = (r0.log_observed, r1.log_observed)
    assert abs((values[2][0] - values[1][0])
               - (values[2][1] - values[1][1])) < 1e-7


# --------------------------------------------------------------------------
# joint mode, failure policy, supported paths, parameter term
# --------------------------------------------------------------------------

def test_joint_mode_is_stationary_and_deterministic():
    model, X, Y, Z = _mixed_case("bernoulli")
    r1 = lk.laplace_log_observed(model, X, Y, 0.1 * Z)
    r2 = lk.laplace_log_observed(model, X, Y, 0.1 * Z)
    assert r1.ok and r1.grad_inf <= 1e-8
    g = lk.gradient(model, X, Y, r1.Z_hat)
    assert float(np.abs(g).max()) <= 1e-8
    assert r1.log_observed == r2.log_observed
    assert r1.min_eigenvalue_H > 0


def test_not_stationary_is_reported_not_valued():
    model, X, Y, Z = _mixed_case("bernoulli")
    r = lk.laplace_log_observed(model, X, Y, 3.0 * Z, max_iter=1)
    assert r.status == lk.STATUS_NOT_STATIONARY
    assert math.isnan(r.log_observed)


def test_indefinite_hessian_is_reported_without_ridge(monkeypatch):
    model, X, Y, _, _ = _gaussian_case(2)
    n, k = 8, 2
    monkeypatch.setattr(lk, "find_joint_mode",
                        lambda *a, **kw: (np.zeros((n, k)), 0, 0.0))
    monkeypatch.setattr(lk, "negative_hessian",
                        lambda *a, **kw: np.diag([-1.0] + [1.0] * (n * k - 1)))
    r = lk.laplace_log_observed(model, X, Y, np.zeros((n, k)))
    assert r.status == lk.STATUS_HESSIAN_NOT_PD
    assert math.isnan(r.log_observed)
    assert r.min_eigenvalue_H == -1.0
    out = lk.calc_C_Lap(r, d_K=7, N=n)
    assert math.isnan(out["C_Lap"])


def test_unsupported_paths_raise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 2))
    Y = _symmetric_binary(rng, 6)
    gaussian_y = _model(["gaussian"] * 2, "gaussian", 2, np.ones((2, 2)),
                        [1, 1], 0.0, 0.5)
    with pytest.raises(lk.UnsupportedLaplacePath):
        lk.laplace_log_observed(gaussian_y, X, Y, np.zeros((6, 2)))
    ok = _model(["gaussian"] * 2, "bernoulli", 2, np.ones((2, 2)), [1, 1],
                0.0, 0.5)
    Y_asym = Y.copy()
    Y_asym[0, 1] = 1 - Y_asym[0, 1]
    with pytest.raises(lk.UnsupportedLaplacePath):
        lk.laplace_log_observed(ok, X, Y_asym, np.zeros((6, 2)))
    from model_dual_expfam_masked import DualExpFamLSMMasked
    legacy = DualExpFamLSMMasked(n=6, d=2, k=2, L=1, family_x="gaussian",
                                 family_y="bernoulli")
    legacy.initialize_params(true_params=None, seed=0)
    with pytest.raises(lk.UnsupportedLaplacePath):
        lk.laplace_log_observed(legacy, X, Y, np.zeros((6, 2)))


def test_parameter_term_is_separate_and_explicit():
    model, X, Y, F, sd = _gaussian_case(2)
    r = lk.laplace_log_observed(model, X, Y, np.zeros((8, 2)))
    d_K = lk.loading_parameter_count(2, 4)
    assert d_K == 2 * 4 - 1
    out = lk.calc_C_Lap(r, d_K=d_K, N=8)
    assert out["integration_term"] == -2.0 * r.log_observed
    assert out["parameter_term"] == d_K * math.log(8)
    assert out["C_Lap"] == out["integration_term"] + out["parameter_term"]
    with pytest.raises(TypeError):
        lk.calc_C_Lap(r, d_K=d_K)                     # N has no default
    with pytest.raises(ValueError):
        lk.calc_C_Lap(r, d_K=d_K, N=1)
