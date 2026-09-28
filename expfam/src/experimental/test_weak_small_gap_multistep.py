"""Zero-EM focused tests for the Phase 9R wrapper (Issue #108)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import laplace_k_criterion as lk                                  # noqa: E402
import multistep_local_pilot as mp                                # noqa: E402
import theta_stationarity_diagnostic as td                        # noqa: E402
import weak_small_gap_multistep as wm                             # noqa: E402
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401
from test_multistep_local_pilot import Quadratic                  # noqa: E402
from test_theta_stationarity_diagnostic import _mixed_state       # noqa: E402


def test_1_same_phase9q_five_pairs():
    assert wm.PAIRS == (("rep10", 2, 3), ("rep11", 3, 2), ("rep07", 2, 3),
                        ("rep04", 2, 3), ("rep09", 3, 2))
    assert wm.PAIRS == tuple(p[:3] for p in wm.wg.PAIRS)


def test_2_exactly_ten_states():
    assert len(wm.STATES) == len(set(wm.STATES)) == 10


def test_3_weak_source_only():
    assert wm.SOURCE_DIR.parts[-2:] == ("phase9p_20260928", "weak_w")
    states = json.loads((wm.SOURCE_DIR / "fitted_states.json")
                        .read_text("utf-8"))
    assert states["condition"] == "weak_w"


def test_4_phase9n_constants_unchanged():
    assert mp.MAX_ACCEPTED == 20 and mp.SCORE_TOL == 1e-3
    assert mp.EPS_DIR == 1e-3 and td.H_BASE / 2 == 5e-5
    assert mp.STEP_FACTORS == (1.0, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32,
                               1 / 64)


def test_5_first_step_gate_logic():
    q = {"status": "ACCEPTED", "accepted_factor": "1.0",
         "delta_ell_one_step": "0.25", "curvature": "-100.0",
         "t_star": "0.05"}
    row = {"accepted_factor": 1.0, "step_delta_ell": 0.25,
           "curvature": -100.0, "t_star": 0.05}
    assert wm.first_step_check(row, q)["pass"]
    assert not wm.first_step_check(dict(row, accepted_factor=0.5), q)["pass"]
    assert not wm.first_step_check(dict(row, step_delta_ell=0.2501),
                                   q)["pass"]
    assert not wm.first_step_check(None, q)["pass"]


def test_6_two_pass_equals_one_pass_and_caps_at_20():
    slow = lambda: Quadratic([1.0, 1e4], [1.0, 1e-4])            # noqa: E731
    one = mp.ascend(q := slow(), q.value(np.zeros(2)))
    q2 = slow()
    ell0 = q2.value(np.zeros(2))
    first = mp.ascend(q2, ell0, max_accepted=1)
    rest = mp.ascend(q2, first["final_ell"], max_accepted=19)
    both = wm.combine(first, rest, ell0)
    assert both["status"] == one["status"] == mp.MAX_ITER
    assert both["iterations"] == one["iterations"] == 20
    assert [r["iteration"] for r in both["trajectory"]] == list(range(1, 21))
    assert np.allclose([r["cumulative_delta_ell"] for r in both["trajectory"]],
                       [r["cumulative_delta_ell"] for r in one["trajectory"]],
                       rtol=0, atol=1e-12)


def test_7_endpoint_decision_logic():
    row = lambda o: {"ordering": o}                               # noqa: E731
    assert wm.decide([row("retained")] * 5) == \
        "WEAK_SMALL_GAP_MULTISTEP_STABLE"
    assert wm.decide([row("tie")] + [row("retained")] * 4) == \
        "WEAK_SMALL_GAP_MULTISTEP_CAN_CHANGE"
    assert wm.decide([row("flipped"), row("unavailable")]) == \
        "WEAK_SMALL_GAP_MULTISTEP_INCONCLUSIVE"


def test_8_failure_logging_does_not_retry(monkeypatch):
    saved = _mixed_state()
    state = wm.LoggingState(saved, saved.base())
    real = lk.laplace_log_observed
    calls = []

    def patched(*args, **kwargs):
        res = real(*args, **kwargs)
        calls.append(1)
        if len(calls) == 6:              # call 1 is the base; coordinate 2, +h
            res.status = lk.STATUS_NOT_STATIONARY
            res.log_observed = float("nan")
            res.notes = ["forced"]
        return res

    monkeypatch.setattr(wm.lk, "laplace_log_observed", patched)
    out = mp.ascend(state, state.evaluate(np.zeros(state.n_coords))[0])
    calls_after_start = len(calls) - 1
    assert out["status"] == mp.DERIVATIVE_UNAVAILABLE
    assert calls_after_start == 2 * state.n_coords       # no extra call
    rows = wm.failure_rows(state, "rep", saved.k, 1)
    assert len(rows) == 1
    assert (rows[0]["coordinate_index"], rows[0]["side"],
            rows[0]["status"], rows[0]["notes"]) == (2, "+",
                                                     "NOT_STATIONARY",
                                                     "forced")
