"""Zero-EM focused tests for the Phase 9T wrapper (Issue #113)."""

from __future__ import annotations

import ast
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import run_density_controlled_w as dw                             # noqa: E402
import run_family_selection_pilot as pilot                        # noqa: E402
import run_lap_vs_cq_20 as pc                                     # noqa: E402
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401


def _copy(tmp_path, edit=None, name="future_protocol.json"):
    d = tmp_path / "cal"
    shutil.copytree(dw.CALIBRATION_DIR, d)
    if edit:
        p = d / name
        obj = json.loads(p.read_text("utf-8"))
        edit(obj)
        p.write_text(json.dumps(obj), "utf-8")
    return d


def test_1_refuses_unless_ready_and_frozen(tmp_path):
    assert dw.load_frozen()["calibration"]["DECISION"] == dw.READY
    bad = _copy(tmp_path, lambda o: o.update(DECISION="X"),
                "calibration.json")
    with pytest.raises(SystemExit):
        dw.load_frozen(bad)
    shutil.rmtree(bad)
    bad = _copy(tmp_path, lambda o: o.update(status="EXECUTED"))
    with pytest.raises(SystemExit):
        dw.load_frozen(bad)


def test_2_seeds_k_grid_start():
    for c in dw.NEW_CONDITIONS:
        p = dw.protocol_for(c)
        assert [(r.data_seed, r.search_seed, r.refit_seed)
                for r in p.replicates] == [(1001000 + r, 1002000 + r,
                                            1003000 + r) for r in range(1, 21)]
        assert p.k_candidates == (1, 2, 3, 4, 5)
        assert p.starts == (("start_B", "bernoulli"),)


def test_3_new_em_cap_400():
    assert sum(dw.protocol_for(c).expected_em_executions
               for c in dw.NEW_CONDITIONS) == dw.NEW_EM_CAP == 400


def test_4_baseline_is_historical_and_zero_em(tmp_path):
    with pytest.raises(SystemExit):
        dw.run_condition("baseline", tmp_path)
    assert dw.protocol_for("baseline") is pc.PROTOCOL
    data = dw.ws.load_condition(tmp_path, "baseline")
    assert data["runinfo"]["git_sha"] == dw.ws.PHASE9K_RUN_SHA


def test_5_w0_comes_from_the_artifact(tmp_path):
    frozen = dw.load_frozen()
    for c in dw.NEW_CONDITIONS:
        assert dw.protocol_for(c, frozen).w0 == \
            frozen["future_protocol"]["frozen"][c]["w0"]
    edited = json.loads(json.dumps(frozen))
    edited["future_protocol"]["conditions"]["weak_w"]["w0"] = -0.5
    edited["future_protocol"]["frozen"]["weak_w"]["w0"] = -0.5
    assert dw.protocol_for("weak_w", edited).w0 == -0.5        # not hard-coded
    edited["future_protocol"]["conditions"]["weak_w"]["L"] = 6
    with pytest.raises(SystemExit):                              # field check
        dw.protocol_for("weak_w", edited)


def test_6_zfx_identity_for_one_replicate():
    rep = pc.PROTOCOL.replicates[0]
    data = {c: pilot.build_dataset(dw.protocol_for(c), rep)
            for c in dw.CONDITIONS}
    for c in dw.NEW_CONDITIONS:
        for m in ("Z", "F", "X"):
            assert np.array_equal(getattr(data[c], m),
                                  getattr(data["baseline"], m))


def test_7_no_local_optimization_follow_up():
    tree = ast.parse(Path(dw.__file__).read_text("utf-8"))
    imported = {a.name for n in ast.walk(tree)
                if isinstance(n, ast.Import) for a in n.names}
    assert imported.isdisjoint({"theta_stationarity_diagnostic",
                                "multistep_local_pilot",
                                "weak_small_gap_one_step",
                                "weak_small_gap_multistep",
                                "derivative_failure_classification"})
