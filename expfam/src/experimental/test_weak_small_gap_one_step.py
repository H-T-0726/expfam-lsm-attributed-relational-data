"""Zero-EM focused tests for the Phase 9Q wrapper (Issue #106)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import theta_stationarity_diagnostic as td                        # noqa: E402
import weak_small_gap_one_step as wg                              # noqa: E402
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401


def test_1_frozen_five_pairs_in_order():
    assert [p[:3] for p in wg.PAIRS] == [
        ("rep10", 2, 3), ("rep11", 3, 2), ("rep07", 2, 3), ("rep04", 2, 3),
        ("rep09", 3, 2)]
    paired = json.loads((wg.SOURCE_DIR / "paired_summary.json")
                        .read_text("utf-8"))
    wg.check_pairs(paired)                  # matches committed Phase 9P


def test_2_exactly_ten_states():
    assert len(wg.STATES) == len(set(wg.STATES)) == 10
    assert {k for _, k in wg.STATES} == {2, 3}


def test_3_weak_source_only():
    assert wg.SOURCE_DIR.parts[-2:] == ("phase9p_20260928", "weak_w")
    states = json.loads((wg.SOURCE_DIR / "fitted_states.json")
                        .read_text("utf-8"))
    assert states["condition"] == "weak_w"
    assert wg.ws.protocol_for(wg.CONDITION).w == wg.ws.W_VALUES["weak_w"]


def test_4_phase9m_constants_unchanged():
    assert td.H_BASE == 1e-4 and td.EPS_DIR == 1e-3
    assert td.STEP_FACTORS == (1.0, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32,
                               1 / 64)
    assert td.RECONSTRUCTION_RTOL == 1e-9


def test_5_pair_decision_logic():
    def row(o):
        return {"ordering": o}
    assert wg.decide([row("retained")] * 5) == \
        "WEAK_SMALL_GAP_ONE_STEP_STABLE"
    assert wg.decide([row("retained")] * 4 + [row("flipped")]) == \
        "WEAK_SMALL_GAP_ONE_STEP_CAN_CHANGE"
    assert wg.decide([row("tie")] + [row("retained")] * 4) == \
        "WEAK_SMALL_GAP_ONE_STEP_CAN_CHANGE"
    assert wg.decide([row("flipped"), row("unavailable")]) == \
        "WEAK_SMALL_GAP_ONE_STEP_INCONCLUSIVE"
