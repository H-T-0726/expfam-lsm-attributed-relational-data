"""Zero-EM focused tests for the Phase 9V runner/adapter (Issue #118)."""

from __future__ import annotations

import ast
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import paired_attribute_scale as pa                               # noqa: E402
import run_attribute_scale_sensitivity as rs                      # noqa: E402
import run_family_selection_pilot as pilot                        # noqa: E402
import run_lap_vs_cq_20 as pc                                     # noqa: E402
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401

REP = pc.PROTOCOL.replicates[0]


def _edited(tmp_path, name, edit):
    d = tmp_path / "design"
    shutil.copytree(rs.DESIGN_DIR, d)
    p = d / name
    obj = json.loads(p.read_text("utf-8"))
    edit(obj)
    p.write_text(json.dumps(obj), "utf-8")
    return d


def test_1_prerequisite(tmp_path):
    assert rs.load_design()["design"]["DECISION"] == \
        "ATTRIBUTE_SCALE_DESIGN_READY"
    for name, edit in (
            ("design.json", lambda o: o.update(DECISION="X")),
            ("future_protocol.json", lambda o: o.update(status="EXECUTED")),
            ("future_protocol.json",
             lambda o: o.update(future_base_rerun_required=False)),
            ("future_protocol.json", lambda o: o.update(future_new_em_cap=400))):
        d = _edited(tmp_path, name, edit)
        with pytest.raises(SystemExit):
            rs.load_design(d)
        shutil.rmtree(d)


def test_2_exact_f_scales_from_artifact():
    assert [rs.protocol_for(c).f_scale for c in rs.CONDITIONS] == \
        [1.0, math.sqrt(2.0), 2.0]


def test_3_4_5_seeds_k_grid_start():
    for c in rs.CONDITIONS:
        p = rs.protocol_for(c)
        assert [(r.data_seed, r.search_seed, r.refit_seed)
                for r in p.replicates] == [(1001000 + r, 1002000 + r,
                                            1003000 + r) for r in range(1, 21)]
        assert p.k_candidates == (1, 2, 3, 4, 5)
        assert p.starts == (("start_B", "bernoulli"),)


def test_6_em_cap_600():
    assert sum(rs.protocol_for(c).expected_em_executions
               for c in rs.CONDITIONS) == rs.NEW_EM_CAP == 600


def test_7_adapter_is_bit_identical_and_scoped():
    original = pilot.build_dataset
    with rs.paired_data_source():
        adapted = {c: pilot.build_dataset(rs.protocol_for(c), REP)
                   for c in rs.CONDITIONS}
    assert pilot.build_dataset is original
    for c in rs.CONDITIONS:
        ref = pa.generate(REP.data_seed, c)
        for m in ("Z", "F", "X", "Y"):
            assert np.array_equal(getattr(adapted[c], m), getattr(ref, m))


def test_8_9_pairing_and_scale_relation():
    check = pa.pairing_check(REP.label, REP.data_seed,
                             pa.generate_all(REP.data_seed))
    assert all(check[k] for k in pa.REQUIRED_CHECKS)
    assert check["Z_identical"] and check["Q_identical"]
    assert check["Y_identical"] and check["F_exact_scale"]


def test_10_no_historical_base_reuse():
    assert "baseline" not in rs.CONDITIONS
    with rs.paired_data_source():
        new_base = pilot.build_dataset(rs.protocol_for("base"), REP)
    historical = pilot.build_dataset(pc.PROTOCOL, REP)
    assert not np.array_equal(new_base.Z, historical.Z)
    assert new_base.metadata["generator_version"] == \
        pa.PAIRED_GENERATOR_VERSION


def test_11_no_local_optimization():
    tree = ast.parse(Path(rs.__file__).read_text("utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names}
    assert imported.isdisjoint({"theta_stationarity_diagnostic",
                                "multistep_local_pilot",
                                "weak_small_gap_one_step",
                                "weak_small_gap_multistep",
                                "derivative_failure_classification"})
