"""Zero-EM focused tests for the Phase 9N multi-step pilot (Issue #100).

Rotation invariance, vertical-score and exact-Gaussian checks are the
Phase 9M tests (test_theta_stationarity_diagnostic.py) and are not repeated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import multistep_local_pilot as mp                                # noqa: E402
import theta_stationarity_diagnostic as td                        # noqa: E402
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401
from test_theta_stationarity_diagnostic import _mixed_state       # noqa: E402


class Quadratic:
    """f(t) = -1/2 (c + t - a)^T D (c + t - a) with a recentring chart."""

    def __init__(self, diag, a, fail=None, chart_ok=True):
        self.D, self.a = np.asarray(diag, float), np.asarray(a, float)
        self.c = np.zeros_like(self.a)
        self.n_coords = self.a.size
        self.fail, self._chart_ok = fail, chart_ok

    def value(self, t):
        r = self.c + t - self.a
        return -0.5 * float(r @ (self.D * r))

    def chart_ok(self):
        return self._chart_ok

    def steps(self):
        return np.full(self.n_coords, 5e-5)

    def evaluate(self, t):
        if self.fail is not None and self.fail(self.c + t):
            return float("nan"), None
        return self.value(t), None

    def accept(self, t, payload):
        self.c = self.c + t

    def blocks(self, g):
        return {}

    def mode_info(self, payload):
        return {}


def test_1_recentred_chart_keeps_horizontal_dimension():
    state = mp.RecenteredState(_mixed_state(), _mixed_state().base())
    rng = np.random.default_rng(0)
    for _ in range(3):
        theta = 0.05 * rng.normal(size=state.n_coords)
        value, res = state.evaluate(theta)
        assert np.isfinite(value)
        state.accept(theta, res)
        assert state.chart_ok()
        assert state.s.n_horizontal == state.expected_horizontal
        V = td.rotation_tangent(state.s.F0)
        assert np.allclose(state.s.B_H.T @ V, 0, atol=1e-10)
        assert np.isclose(state.evaluate(np.zeros(state.n_coords))[0], value,
                          rtol=0, atol=1e-8)


def test_2_concave_quadratic_converges_monotonically():
    q = Quadratic([1.0, 3.0], [0.4, -0.3])
    out = mp.ascend(q, q.value(np.zeros(2)))
    assert out["status"] == mp.CONVERGED
    assert out["final_grad_inf"] <= mp.SCORE_TOL
    ells = [q.value(-q.c)] + [r["ell_Lap"] for r in out["trajectory"]]
    assert all(b > a for a, b in zip(ells, ells[1:]))


def test_3_trajectory_never_decreases_on_ill_conditioned_quadratic():
    q = Quadratic([1.0, 400.0, 20.0], [1.0, 0.5, -1.0])
    out = mp.ascend(q, q.value(np.zeros(3)))
    steps = [r["step_delta_ell"] for r in out["trajectory"]]
    assert steps and min(steps) > 0
    cum = [r["cumulative_delta_ell"] for r in out["trajectory"]]
    assert all(b >= a for a, b in zip(cum, cum[1:]))


def test_4_statuses_are_deterministic():
    def run(q):
        return mp.ascend(q, q.value(np.zeros(q.n_coords)))

    slow = lambda: Quadratic([1.0, 1e4], [1.0, 1e-4])     # noqa: E731  zigzag
    a, b = run(slow()), run(slow())
    assert a["status"] == mp.MAX_ITER and a["iterations"] == mp.MAX_ACCEPTED
    assert a["trajectory"] == b["trajectory"]
    convex = Quadratic([-1.0, -1.0], [1.0, 1.0])
    assert run(convex)["status"] == mp.CURVATURE_NONNEG
    nan_everywhere = Quadratic([1.0, 1.0], [1.0, 1.0], fail=lambda x: True)
    assert run(nan_everywhere)["status"] == mp.DERIVATIVE_UNAVAILABLE
    rank = Quadratic([1.0, 1.0], [1.0, 1.0], chart_ok=False)
    assert run(rank)["status"] == mp.CHART_RANK_FAILURE
    blocked = Quadratic([1.0, 1.0], [1.0, 1.0],
                        fail=lambda x: np.linalg.norm(x) > 1e-2)
    assert run(blocked)["status"] == mp.NO_ACCEPTED


def _result(status, cums):
    traj = [{"cumulative_delta_ell": c} for c in cums]
    return {"status": status, "iterations": len(cums), "trajectory": traj,
            "cumulative_delta_ell": cums[-1] if cums else 0.0}


def test_5_pair_final_gap_labels():
    pairs = (("repA", 3, 2), ("repB", 3, 2), ("repC", 3, 4))
    c = {("repA", 3): 100.0, ("repA", 2): 110.0,
         ("repB", 3): 100.0, ("repB", 2): 104.0,
         ("repC", 3): 100.0, ("repC", 4): 101.0}
    results = {
        ("repA", 3): _result(mp.MAX_ITER, [1.0] * 20),
        ("repA", 2): _result(mp.CONVERGED, [2.0, 3.0]),         # gap 10 -> 6
        ("repB", 3): _result(mp.CONVERGED, [0.5]),
        ("repB", 2): _result(mp.MAX_ITER, [3.0] * 20),          # gap 4 -> -1
        ("repC", 3): _result(mp.NO_ACCEPTED, [0.1]),
        ("repC", 4): _result(mp.CONVERGED, [0.2])}
    rows, traj = mp.pair_rows(c, results, pairs)
    assert [r["ordering"] for r in rows] == ["retained", "flipped",
                                             "unavailable"]
    assert rows[0]["final_diag_gap"] == 6.0
    assert rows[1]["final_diag_gap"] == -1.0
    assert rows[1]["intermediate_flip_rounds"].startswith("1;")
    a_rounds = [t["diag_gap"] for t in traj if t["replicate"] == "repA"]
    assert a_rounds[0] == 10.0 and a_rounds[-1] == 6.0     # carry forward
    assert mp.decide(rows) == "MULTISTEP_PILOT_INCONCLUSIVE"
    assert mp.decide(rows[:2]) == "MULTISTEP_PILOT_ORDERING_CAN_CHANGE"
    assert mp.decide(rows[:1]) == "MULTISTEP_PILOT_ORDERING_STABLE"
