"""Zero-EM focused tests for the Phase 9M diagnostic (Issue #98)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import special_ortho_group

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import laplace_k_criterion as lk                                  # noqa: E402
import theta_stationarity_diagnostic as td                        # noqa: E402
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401
from test_laplace_k_criterion import (                            # noqa: E402
    _exact_log_pXY, _gaussian_case, _mixed_case)


def _entry(model, Z):
    p = model.params
    return {"replicate": "rep01", "k": model.k,
            "selected_assignment": list(model.family_x_list),
            "F": p["F"].tolist(), "sigma_diag": np.diag(p["sigma"]).tolist(),
            "w0": p["w0"], "w": p["w"], "var_z": p["var_z"],
            "Z_est": np.asarray(Z).tolist()}


def _mixed_state():
    model, X, Y, Z = _mixed_case("bernoulli")
    state = td.SavedState(_entry(model, 0.1 * Z), X, Y)
    assert state.base().ok
    return state


@pytest.mark.parametrize("k,d", [(1, 4), (2, 4), (3, 12), (4, 12)])
def test_D_horizontal_dimension(k, d):
    F = np.random.default_rng(k).normal(size=(d, k))
    B_V, B_H = td.split_basis(F)
    assert B_H.shape[1] == k * d - k * (k - 1) // 2
    assert B_V.shape[1] == k * (k - 1) // 2
    assert np.allclose(B_H.T @ B_V, 0, atol=1e-12)


def test_A_finite_rotation_leaves_ell_invariant():
    model, X, Y, Z = _mixed_case("bernoulli")
    r0 = lk.laplace_log_observed(model, X, Y, 0.1 * Z)
    R = special_ortho_group.rvs(2, random_state=3)
    model.params["F"] = model.params["F"] @ R
    r1 = lk.laplace_log_observed(model, X, Y, 0.1 * Z @ R)
    assert r0.ok and r1.ok
    assert abs(r0.log_observed - r1.log_observed) < 1e-9 * abs(r0.log_observed)


def test_B_vertical_derivative_is_zero():
    state = _mixed_state()
    fv = lambda v: state.ell(vertical=v)                      # noqa: E731
    g_v, un = td.central_gradient(fv, state.B_V.shape[1],
                                  np.full(state.B_V.shape[1], td.H_BASE / 2))
    fh = lambda t: state.ell(t)                               # noqa: E731
    g_h, _ = td.central_gradient(fh, state.n_coords, state.step_sizes() / 2)
    assert un == 0
    assert np.max(np.abs(g_v)) < 1e-5 * max(1.0, np.max(np.abs(g_h)))


def test_C_gaussian_exact_case_derivatives_match_closed_form():
    model, X, Y, F, sd = _gaussian_case(2)
    state = td.SavedState(_entry(model, np.zeros((8, 2))), X, Y)
    assert state.base().ok

    def exact(theta):
        nh, ng = state.n_horizontal, len(state.gauss)
        Fv = (state.F0.ravel() + state.B_H @ theta[:nh]).reshape(F.shape)
        var = state.var0 * np.exp(theta[nh:nh + ng])
        return _exact_log_pXY(X, Y, Fv, var, w0=state.w0_0 + theta[nh + ng])

    steps = state.step_sizes() / 2
    coords = [0, 3, state.n_horizontal, state.n_horizontal + 2,
              state.n_horizontal + len(state.gauss)]        # F, F, logvar, logvar, w0
    for i in coords:
        e = np.zeros(state.n_coords)
        e[i] = steps[i]
        d_impl = (state.ell(e) - state.ell(-e)) / (2 * steps[i])
        d_exact = (exact(e) - exact(-e)) / (2 * steps[i])
        assert abs(d_impl - d_exact) < 1e-6 * max(1.0, abs(d_exact)), i


def test_one_step_on_a_concave_quadratic():
    a = np.array([0.3, -0.2])
    f = lambda t: -float(np.sum((t - a) ** 2))                # noqa: E731
    g = 2 * a                                                 # gradient at 0
    out = td.one_step(f, g, f(np.zeros(2)))
    assert out["status"] == "ACCEPTED" and out["accepted_factor"] == 1.0
    assert math.isclose(out["delta_ell_one_step"], float(a @ a), rel_tol=1e-6)


def test_one_step_refuses_non_negative_curvature():
    f = lambda t: float(np.sum(t ** 2))                       # noqa: E731
    out = td.one_step(f, np.array([1.0, 0.0]), 0.0)
    assert out["status"] == "UNAVAILABLE"
    assert out["reason"] == "non-negative curvature"


def test_pairwise_ordering_labels():
    paired = {"datasets": [
        {"replicate": "rep01", "C_Lap": {"2": 110.0, "3": 100.0, "4": 105.0},
         "lap_best_k": 3, "lap_second_k": 4, "lap_gap": 5.0},
        {"replicate": "rep02", "C_Lap": {"2": 110.0, "3": 100.0, "4": 105.0},
         "lap_best_k": 3, "lap_second_k": 4, "lap_gap": 5.0}]}
    steps = [
        {"replicate": "rep01", "k": 3, "status": "ACCEPTED",
         "delta_ell_one_step": 1.0, "C_Lap_one_step_diag": 98.0},
        {"replicate": "rep01", "k": 4, "status": "ACCEPTED",
         "delta_ell_one_step": 4.0, "C_Lap_one_step_diag": 97.0},
        {"replicate": "rep01", "k": 2, "status": "ACCEPTED",
         "delta_ell_one_step": 0.5, "C_Lap_one_step_diag": 109.0},
        {"replicate": "rep02", "k": 3, "status": "NO_ACCEPTED_STEP",
         "delta_ell_one_step": float("nan"),
         "C_Lap_one_step_diag": float("nan")}]
    rows = td.pairwise(paired, steps)
    assert rows[0]["ordering"] == "flipped" and rows[0]["adjusted_gap"] == -1
    assert rows[0]["delta_Lap_23_one_step"] == -10.0 - 2 * (1.0 - 0.5)
    assert rows[1]["ordering"] == "unavailable"
    s = td.summarize([], [], rows)
    assert s["DECISION"] == "STATIONARITY_DIAGNOSTIC_INCONCLUSIVE"
