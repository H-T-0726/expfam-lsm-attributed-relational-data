"""Zero-EM checks for the Phase 9K differences only (Issue #94).

The evaluator, the per-refit Laplace evaluation and the completeness rule
are covered by the Phase 9H / 9I tests; the runner by the Phase 9D/9E tests.
"""

from __future__ import annotations

import json
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
import run_lap_vs_cq_20 as pc                                     # noqa: E402
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401


def test_frozen_protocol():
    p = pc.PROTOCOL
    assert [(r.label, r.data_seed, r.search_seed, r.refit_seed)
            for r in p.replicates] == [
        (f"rep{r:02d}", 1001000 + r, 1002000 + r, 1003000 + r)
        for r in range(1, 21)]
    assert p.starts == (("start_B", "bernoulli"),)
    assert list(p.k_candidates) == [1, 2, 3, 4, 5]
    assert p.expected_em_executions == 200
    for field in joint.JointProtocol.__dataclass_fields__:
        if field not in {"stage", "replicates", "starts", "failure_scope"}:
            assert getattr(p, field) == getattr(joint.PROTOCOL, field), field


class _Model:
    def __init__(self):
        self.params = {"F": np.ones((3, 2)), "sigma": np.diag([2.0, 1, 1]),
                       "w0": -1.0, "w": 0.5, "var_z": 1.0}


def test_driver_uses_frozen_laplace_settings_and_keeps_state(monkeypatch):
    seen = {}

    def fake_laplace(model, X, Y, Z0, *, grad_tol, max_iter):
        seen.update(Z0=Z0.copy(), grad_tol=grad_tol, max_iter=max_iter)
        return lk.LaplaceResult(status=lk.STATUS_OK, log_observed=-50.0,
                                phi_at_mode=-40.0, logdet_H=3.0,
                                logdet_sign=1.0, min_eigenvalue_H=0.5,
                                grad_inf=1e-10, iterations=2, n=4, k=2,
                                Z_hat=Z0)

    monkeypatch.setattr(lk, "laplace_log_observed", fake_laplace)
    Z_est = np.arange(8.0).reshape(4, 2)
    fake = {"refit": {"model": _Model(), "Z_est": Z_est, "bic": 1.0,
                      "Q_strict": -1.0, "num_params": 7},
            "selected_assignment": ["gaussian", "bernoulli", "poisson"]}
    rows, states = [], []
    out = pc.make_driver(rows, states, inner=lambda X, Y, **kw: fake)(
        np.zeros((4, 3)), np.zeros((4, 4)), k=2, search_seed=1002007)
    assert out is fake
    np.testing.assert_array_equal(seen["Z0"], Z_est)
    assert (seen["grad_tol"], seen["max_iter"]) == (1e-8, 200)
    [row] = rows
    assert row["replicate"] == "rep07" and row["N"] == 75
    assert row["C_Lap"] == 100.0 + row["d_K"] * math.log(75)
    [state] = states
    assert (state["data_seed"], state["refit_seed"]) == (1001007, 1003007)
    assert state["sigma_diag"] == [2.0, 1.0, 1.0]
    assert state["Z_est"] == Z_est.tolist()
    json.dumps(state)


def test_unfrozen_seed_is_rejected():
    with pytest.raises(joint.RunnerStop):
        pc._replicate_for(982001)


def test_best_second_gap():
    assert pc.best_second({1: 5.0, 2: 3.0, 3: 3.5}) == (2, 3, 0.5)


def _rows(rep, lap, q, statuses=None):
    statuses = statuses or ["OK"] * 5
    return [{"replicate": rep, "k": k, "laplace_status": s, "C_Lap": a,
             "C_Q": b} for k, s, a, b in zip(range(1, 6), statuses, lap, q)]


def test_paired_summary_counts(monkeypatch):
    monkeypatch.setattr(pc, "PROTOCOL", pc.dataclasses.replace(
        pc.PROTOCOL, replicates=pc.PROTOCOL.replicates[:3]))
    rows = (_rows("rep01", [9, 5, 4, 6, 7], [9, 4, 5, 6, 7])      # Lap 3, Q 2
            + _rows("rep02", [9, 5, 4, 6, 7], [9, 5, 4, 6, 7])    # both 3
            + _rows("rep03", [9, 5, float("nan"), 6, 7], [9, 4, 5, 6, 7],
                    ["OK", "OK", "HESSIAN_NOT_PD", "OK", "OK"]))  # Lap n/a
    s = pc.paired_summary(rows, [])
    assert s["c_lap_complete_datasets"] == 2
    assert s["P1_K_hat_Lap"]["counts"][3] == 2
    assert s["P2_K_hat_Q"]["counts"] == {1: 0, 2: 2, 3: 1, 4: 0, 5: 0}
    p = s["P3_paired"]
    assert (p["denominator"], p["both_K3"], p["lap_only_K3"]) == (2, 1, 1)
    assert p["pairs"]["rep03"] == [2, None]
    assert s["P4_margins"]["lap_gap"]["min"] == 1
    assert s["P5_technical"]["HESSIAN_NOT_PD"] == 1
