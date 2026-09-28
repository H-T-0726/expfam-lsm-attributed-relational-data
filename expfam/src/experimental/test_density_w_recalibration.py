"""Zero-EM focused tests for the Phase 9S2 recalibration (Issue #112)."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import density_w_calibration as dc                                # noqa: E402
import density_w_recalibration as dr                              # noqa: E402
import run_w_sensitivity as ws                                    # noqa: E402
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401

COARSE = dr.PRIMARY_SETTINGS[0]


def test_1_density_normalization_and_second_moment():
    c = dr.density_checks()
    assert c["normalization_error"] <= dr.DENSITY_TOL
    assert c["second_moment_error"] <= dr.DENSITY_TOL
    assert c["pass"]


def test_2_s_k1_endpoint():
    assert dr.s_k1(0.0) == 1.0
    assert abs(float(dr.s_k1(1e-300)) - 1.0) < 1e-15
    assert abs(float(dr.s_k1(1e-8)) - 1.0) < 1e-12
    assert float(dr.s_k1(800.0)) == 0.0 or float(dr.s_k1(800.0)) < 1e-300


def test_3_baseline_recovers_minus_one():
    row = dr.calibrate(dr.Adaptive(COARSE[1], COARSE[2]))
    assert abs(row["baseline_recovery_error"]) <= 1e-12


def test_4_primary_repeated_calibration_deterministic():
    a = dr.calibrate(dr.Adaptive(COARSE[1], COARSE[2]))
    b = dr.calibrate(dr.Adaptive(COARSE[1], COARSE[2]))
    assert a == b


def test_5_fixed_node_crosscheck_deterministic():
    n = dr.CROSSCHECK_NODES[0]
    assert dr.calibrate(dr.FixedNode(n)) == dr.calibrate(dr.FixedNode(n))


def _row(label, shift=0.0, residual=0.0):
    r = {"setting": label, "valid": True, "p_target": 0.3 + shift,
         "baseline_recovery_error": 0.0}
    for c in dr.W_VALUES:
        r[f"w0_{c}"] = -1.0 + shift
        r[f"residual_{c}"] = residual
    return r


def test_6_freeze_rule_requires_every_comparison():
    good = [_row("A", 1e-9), _row("B"), _row("C", 5e-13)]
    assert dr.freeze_rule(good, True)["pass"]
    assert not dr.freeze_rule(good, False)["pass"]             # density
    assert not dr.freeze_rule([_row("A"), _row("B"), _row("C", 2e-12)],
                              True)["pass"]                   # agreement
    assert not dr.freeze_rule([_row("A"), _row("B"),
                               _row("C", 0.0, 2e-12)], True)["pass"]
    one = _row("B")
    one["w0_strong_w"] += 2e-12                                # one key only
    assert not dr.freeze_rule([_row("A"), _row("C"), one], True)["pass"]
    assert not dr.freeze_rule([_row("A")], True)["pass"]


def test_7_future_protocol_changes_only_w0_and_stage():
    protos = dc.future_protocols({"weak_w": -0.88, "strong_w": -1.19})
    for c, p in protos.items():
        p9p = ws.protocol_for(c).as_json()
        assert {k for k in p if p[k] != p9p[k]} == {"w0", "stage"}
        assert p["w"] == ws.W_VALUES[c]
