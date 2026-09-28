"""Zero-EM contract tests for the Phase 9P wrapper (Issue #104)."""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import run_family_selection_pilot as pilot                        # noqa: E402
import run_lap_vs_cq_20 as pc                                     # noqa: E402
import run_w_sensitivity as ws                                    # noqa: E402
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401


def test_1_frozen_w_values():
    assert ws.W_VALUES["weak_w"] == 1 / math.sqrt(2)
    assert ws.W_VALUES["baseline"] == 1.0 == pc.PROTOCOL.w
    assert ws.W_VALUES["strong_w"] == math.sqrt(2)
    assert [round(ws.W_VALUES[c] ** 2 * 3, 12) for c in ws.CONDITIONS] \
        == [1.5, 3.0, 6.0]


def test_2_protocols_differ_only_in_w_and_stage_with_the_20_seed_tuples():
    base = dataclasses.asdict(pc.PROTOCOL)
    for c in ws.NEW_CONDITIONS:
        p = dataclasses.asdict(ws.protocol_for(c))
        diff = {k for k in base if base[k] != p[k]}
        assert diff == {"w", "stage"}
        assert p["w"] == ws.W_VALUES[c]
    reps = ws.protocol_for("weak_w").replicates
    assert [(r.label, r.data_seed, r.search_seed, r.refit_seed)
            for r in reps] == [(f"rep{r:02d}", 1001000 + r, 1002000 + r,
                                1003000 + r) for r in range(1, 21)]


def test_3_zfx_identical_across_w_for_the_same_replicate():
    rep = pc.PROTOCOL.replicates[0]
    data = {c: pilot.build_dataset(ws.protocol_for(c), rep)
            for c in ws.CONDITIONS}
    for c in ws.NEW_CONDITIONS:
        for m in ("Z", "F", "X"):
            assert np.array_equal(getattr(data[c], m),
                                  getattr(data["baseline"], m))
    assert not np.array_equal(data["weak_w"].Y, data["strong_w"].Y)


def test_4_baseline_is_read_only_and_triggers_zero_em(tmp_path):
    with pytest.raises(SystemExit):
        ws.run_condition("baseline", tmp_path)
    data = ws.load_condition(tmp_path, "baseline")     # reads Phase 9K only
    assert data["runinfo"]["git_sha"] == ws.PHASE9K_RUN_SHA
    assert data["paired"]["c_lap_complete_datasets"] == 20


def test_5_new_em_cap_is_400():
    total = sum(ws.protocol_for(c).expected_em_executions
                for c in ws.NEW_CONDITIONS)
    assert total == ws.NEW_EM_CAP == 400


def test_6_no_extra_start_k_or_replicate():
    for c in ws.NEW_CONDITIONS:
        p = ws.protocol_for(c)
        assert p.starts == (("start_B", "bernoulli"),)
        assert p.k_candidates == (1, 2, 3, 4, 5)
        assert len(p.replicates) == 20
        assert p.failure_scope == "dataset"
        assert p.expected_em_executions == 200
