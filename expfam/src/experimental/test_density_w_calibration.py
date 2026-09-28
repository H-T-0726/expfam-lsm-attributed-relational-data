"""Zero-EM focused tests for the Phase 9S calibration (Issue #110)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import density_w_calibration as dc                                # noqa: E402
import run_lap_vs_cq_20 as pc                                     # noqa: E402
import run_w_sensitivity as ws                                    # noqa: E402
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401

RES = dc.RESOLUTIONS[1]


def test_1_baseline_recovers_minus_one():
    row = dc.calibrate(*RES)
    assert abs(row["baseline_recovery_error"]) < 1e-12


def test_2_all_conditions_match_target():
    row = dc.calibrate(*RES)
    rule = dc.quadrature(*RES)
    for c, w in dc.W_VALUES.items():
        assert abs(dc.p_bar(row[f"w0_{c}"], w, rule) - row["p_target"]) \
            < 1e-13
    # quadrature is a probability rule and reproduces E[S] = 0, Var(S) = K
    assert abs(rule.weights.sum() - 1) < 1e-13
    assert abs(rule.weights @ rule.s) < 1e-13
    assert abs(rule.weights @ rule.s ** 2 - dc.K_TRUE) < 1e-11


def test_3_root_bracket_and_uniqueness():
    rule = dc.quadrature(*RES)
    target = dc.p_bar(-1.0, 1.0, rule)
    for w in dc.W_VALUES.values():
        lo, hi = dc.bracket(target, w, rule)
        assert dc.p_bar(lo, w, rule) <= target <= dc.p_bar(hi, w, rule)
        assert dc.monotonicity_check(w, rule)["strictly_increasing"]
    assert abs(dc.p_bar(0.0, 0.7, rule) - 0.5) < 1e-14    # symmetry of S


def test_4_repeated_calibration_is_deterministic():
    assert dc.calibrate(*RES) == dc.calibrate(*RES)


def test_5_future_protocol_changes_only_w0_and_stage():
    protos = dc.future_protocols({"weak_w": -0.9, "strong_w": -1.1})
    base = pc.PROTOCOL.as_json()
    for c, p in protos.items():
        p9p = ws.protocol_for(c).as_json()
        assert {k for k in p if p[k] != p9p[k]} == {"w0", "stage"}
        assert {k for k in p if p[k] != base[k]} == {"w", "w0", "stage"}
        assert p["expected_em_executions"] == 200
        assert p["starts"] == [{"label": "start_B",
                                "ambiguous_start": "bernoulli"}]
    assert np.isclose(protos["weak_w"]["w"], 2 ** -0.5)
