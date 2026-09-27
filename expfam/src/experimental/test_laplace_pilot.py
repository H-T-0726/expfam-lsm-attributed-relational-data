"""Zero-EM checks for the Phase 9I pilot differences only (Issue #90).

Unchanged behaviour is covered by the Phase 9D/9E and Phase 9H tests.
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
import run_joint_family_k_selection as joint                      # noqa: E402
import run_laplace_pilot as lp                                    # noqa: E402
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401


def test_frozen_differences():
    p = lp.PROTOCOL
    assert [(r.label, r.data_seed, r.search_seed, r.refit_seed)
            for r in p.replicates] == [
        ("rep01", 981001, 982001, 983001), ("rep02", 981002, 982002, 983002),
        ("rep03", 981003, 982003, 983003)]
    assert p.starts == (("start_B", "bernoulli"),)
    assert list(p.k_candidates) == [1, 2, 3, 4, 5]
    assert p.expected_em_executions == 30
    assert p.failure_scope == "dataset"
    for field in joint.JointProtocol.__dataclass_fields__:
        if field not in {"stage", "replicates", "starts", "failure_scope"}:
            assert getattr(p, field) == getattr(joint.PROTOCOL, field), field
    s = lp.LAPLACE_SETTINGS
    assert (s["N"], s["grad_tol"], s["max_iter"]) == (75, 1e-8, 200)
    assert s["restart"] is False and s["jitter_or_ridge"] is False


def test_d_K_formula():
    assert lp.d_K(3, 12, 3) == 3 * 12 - 3 + 3
    assert lp.d_K(1, 12, 2) == 12 + 2


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, model, X, Y, Z0, *, grad_tol, max_iter):
        self.calls.append({"Z0": Z0.copy(), "grad_tol": grad_tol,
                           "max_iter": max_iter})
        n, k = Z0.shape
        return lk.LaplaceResult(status=lk.STATUS_OK, log_observed=-100.0,
                                phi_at_mode=-90.0, logdet_H=5.0,
                                logdet_sign=1.0, min_eigenvalue_H=0.5,
                                grad_inf=1e-10, iterations=3, n=n, k=k,
                                Z_hat=Z0)


def test_driver_uses_refit_Z_est_and_frozen_settings(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(lk, "laplace_log_observed", rec)
    Z_est = np.arange(6.0).reshape(3, 2)
    fake = {"refit": {"model": object(), "Z_est": Z_est, "bic": 10.0,
                      "Q_strict": -4.0, "num_params": 25},
            "selected_assignment": ["gaussian", "bernoulli", "poisson"]}
    rows = []
    driver = lp.make_driver(rows, inner=lambda X, Y, **kw: fake)
    X, Y = np.zeros((3, 3)), np.zeros((3, 3))
    out = driver(X, Y, k=2, search_seed=982002)
    assert out is fake
    np.testing.assert_array_equal(rec.calls[0]["Z0"], Z_est)
    assert (rec.calls[0]["grad_tol"], rec.calls[0]["max_iter"]) == (1e-8, 200)
    [row] = rows
    assert row["replicate"] == "rep02" and row["d_K"] == lp.d_K(2, 3, 1)
    assert row["C_Lap"] == 200.0 + row["d_K"] * math.log(75)


def test_unfrozen_seed_is_rejected():
    with pytest.raises(joint.RunnerStop):
        lp._replicate_for(972001)


def _rows(statuses, c_lap, c_q):
    return [{"replicate": "rep01", "k": k, "laplace_status": s,
             "C_Lap": cl, "C_Q": cq}
            for k, s, cl, cq in zip(range(1, 6), statuses, c_lap, c_q)]


def test_incomplete_dataset_gets_no_argmin():
    rows = _rows(["OK", "OK", "HESSIAN_NOT_PD", "OK", "OK"],
                 [5, 4, float("nan"), 6, 7], [5, 3, 4, 6, 7])
    out = lp.dataset_summary(rows, [], "rep01")
    assert out["c_lap_complete"] is False
    assert out["K_hat_Lap"] is None and out["delta_Lap_23"] is None
    assert out["K_hat_Q"] == 2


def test_complete_dataset_argmin_and_deltas():
    rows = _rows(["OK"] * 5, [5, 4, 3, 6, 7], [5, 3, 4, 6, 7])
    out = lp.dataset_summary(rows, [], "rep01")
    assert out["c_lap_complete"] and out["K_hat_Lap"] == 3
    assert out["delta_Lap_23"] == -1 and out["delta_Q_23"] == 1
