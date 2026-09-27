"""Zero-EM checks for the Phase 9J importance correction (Issue #92)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import laplace_importance as li                                   # noqa: E402
import laplace_k_criterion as lk                                  # noqa: E402
import run_laplace_is_diagnostic as rd                            # noqa: E402
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401
from test_laplace_k_criterion import _gaussian_case, _mixed_case  # noqa: E402


def _mode_and_H(model, X, Y, Z0):
    res = lk.laplace_log_observed(model, X, Y, Z0)
    assert res.ok
    H = lk.negative_hessian(model, X, Y, res.Z_hat)
    return res, 0.5 * (H + H.T)


def test_gaussian_exact_case_correction_is_zero():
    """Gaussian X, w = 0: Phi is quadratic, so r = 0 for every sample and
    the true correction is 0. Only floating-point residue remains."""

    model, X, Y, _, _ = _gaussian_case(2)
    res, H = _mode_and_H(model, X, Y, np.zeros((8, 2)))
    out = li.importance_correction(model, X, Y, res.Z_hat, H,
                                   seeds=[1, 2, 3], samples_per_batch=256)
    for batch in out["batches"]:
        assert abs(batch["correction"]) < 1e-9
        assert batch["relative_ess"] > 1 - 1e-9
        assert batch["non_finite"] == 0
    assert abs(out["pooled"]["correction"]) < 1e-9


def test_non_gaussian_case_is_finite_normalised_and_deterministic():
    model, X, Y, Z = _mixed_case("bernoulli")
    res, H = _mode_and_H(model, X, Y, 0.1 * Z)
    a = li.importance_correction(model, X, Y, res.Z_hat, H, [7, 8], 300)
    b = li.importance_correction(model, X, Y, res.Z_hat, H, [7, 8], 300)
    assert a["pooled"]["correction"] == b["pooled"]["correction"]
    for batch in a["batches"]:
        assert math.isfinite(batch["correction"])
        assert 1.0 <= batch["ess"] <= 300 + 1e-9
        assert 0 < batch["max_weight_share"] <= 1
    assert a["pooled"]["n_samples"] == 600


def test_sampling_uses_the_cholesky_of_H():
    """Δ = L⁻ᵀ ε has covariance H⁻¹: check on the Gaussian exact case."""

    model, X, Y, _, _ = _gaussian_case(1)
    res, H = _mode_and_H(model, X, Y, np.zeros((8, 1)))
    chol = np.linalg.cholesky(H)
    eps = np.random.default_rng(3).standard_normal((20000, H.shape[0]))
    from scipy.linalg import solve_triangular
    delta = solve_triangular(chol.T, eps.T, lower=False).T
    cov = np.cov(delta.T)
    assert np.allclose(cov, np.linalg.inv(H), atol=0.03)


def test_indefinite_H_is_refused():
    model, X, Y, _, _ = _gaussian_case(1)
    H = -np.eye(8)
    with pytest.raises(np.linalg.LinAlgError):
        li.log_ratio_samples(model, X, Y, np.zeros((8, 1)), H, 4, 0)


def test_frozen_protocol():
    p = rd.PROTOCOL
    assert list(p.k_candidates) == [2, 3]
    assert [(r.label, r.data_seed, r.search_seed, r.refit_seed)
            for r in p.replicates] == [
        ("rep01", 991001, 992001, 993001), ("rep02", 991002, 992002, 993002),
        ("rep03", 991003, 992003, 993003)]
    assert p.starts == (("start_B", "bernoulli"),)
    assert p.expected_em_executions == 12
    assert (rd.N_BATCHES, rd.SAMPLES_PER_BATCH, rd.N_PENALTY) == (8, 512, 75)
    assert rd.batch_seed(1, 2, 1) == 995121
    assert rd.batch_seed(3, 3, 8) == 995338


def test_delta_arithmetic():
    def rec(k, c_lap, cs):
        batches = [{"correction": c} for c in cs]
        return {"replicate": "rep01", "k": k, "laplace_status": "OK",
                "C_Lap": c_lap,
                "importance": {"pooled": {"correction": float(np.mean(cs))},
                               "batches": batches}}

    out = rd.replicate_summary([rec(2, 100.0, [0.1] * 8),
                                rec(3, 99.0, [0.6] * 8)], "rep01")
    assert out["delta_Lap_23"] == -1.0
    assert math.isclose(out["delta_c_23"], 0.5)
    assert math.isclose(out["delta_IS_23"], -2.0)
    assert out["batch_delta_IS_23_summary"]["negative"] == 8
