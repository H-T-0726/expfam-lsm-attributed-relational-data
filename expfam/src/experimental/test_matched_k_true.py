"""Zero-EM focused tests for the Phase 9X runner (Issue #122)."""

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

import run_family_selection_pilot as pilot                        # noqa: E402
import run_lap_vs_cq_20 as pc                                     # noqa: E402
import run_matched_k_true as mx                                   # noqa: E402
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401


def _edited(tmp_path, edit):
    d = tmp_path / "w"
    shutil.copytree(mx.DESIGN_DIR, d)
    p = d / "future_protocol.json"
    obj = json.loads(p.read_text("utf-8"))
    edit(obj)
    p.write_text(json.dumps(obj), "utf-8")
    return d


def test_1_phase9w_ready_and_frozen(tmp_path):
    f = mx.load_frozen()
    assert f["design"]["DECISION"] == "MATCHED_K_TRUE_DESIGN_READY"
    for edit in (lambda o: o.update(execution_authorized=True),
                 lambda o: o.update(K3_anchor_reusable=False),
                 lambda o: o.update(future_new_em_cap=400),
                 lambda o: o.update(new_K_true=[1, 2, 3, 4])):
        d = _edited(tmp_path, edit)
        with pytest.raises(SystemExit):
            mx.load_frozen(d)
        shutil.rmtree(d)


def test_2_artifact_hash_binding():
    f = mx.load_frozen()
    assert set(f["sha256"]) == set(mx.HASHED)
    for name, digest in f["sha256"].items():
        assert digest == mx._sha(mx.DESIGN_DIR / name)


def test_3_issue122_authorization_record():
    a = mx.authorization_record(mx.load_frozen())
    assert a["issue"] == 122 and a["authorized_by"] == "Human"
    assert a["authorization_scope"] == "Phase 9X only"
    assert a["phase9w_merge_sha"] == mx.PHASE9W_MERGE
    assert a["K3_rerun"] is False and a["new_em_cap"] == 300
    fp = json.loads((mx.DESIGN_DIR / "future_protocol.json").read_text("utf-8"))
    assert fp["execution_authorized"] is False        # 9W artifact untouched


def test_4_5_6_new_k_set_k3_zero_em_cap():
    assert mx.NEW_K_TRUES == (1, 2, 4)
    with pytest.raises(SystemExit):
        mx.run_condition(3, Path("unused"))
    assert sum(mx.protocol_for(k).expected_em_executions
               for k in mx.NEW_K_TRUES) == mx.NEW_EM_CAP == 300


def test_7_8_9_replicates_k_grid_start():
    for k in mx.K_TRUES:
        p = mx.protocol_for(k)
        assert [r.label for r in p.replicates] == list(mx.REPLICATES)
        assert [r.data_seed for r in p.replicates] == \
            list(range(1001001, 1001011))
        assert p.k_candidates == (1, 2, 3, 4, 5)
        assert p.starts == (("start_B", "bernoulli"),)
        assert p.k_true == k


def test_10_generic_exact_under_over():
    assert [mx.category(3, k) for k in (1, 2, 3, 4)] == \
        ["over", "over", "exact", "under"]
    assert [mx.category(k, k) for k in (1, 2, 3, 4)] == ["exact"] * 4
    assert mx.category(1, 2) == "under" and mx.category(5, 4) == "over"
    assert mx.errors(5, 4) == (1, 1) and mx.errors(1, 2) == (-1, 1)
    assert mx.category(None, 2) is None


def test_11_generic_paired_labels():
    p = mx.paired_labels(3, 3, 1)           # K=3 is NOT exact when K_true=1
    assert p["neither_exact"] and not p["both_exact"] and p["same_selected_K"]
    p = mx.paired_labels(1, 2, 1)
    assert p["Lap_exact_only"] and p["different_selected_K"]
    p = mx.paired_labels(3, 4, 4)
    assert p["Q_exact_only"]
    assert mx.paired_labels(None, 2, 2) is None
    assert mx.contrasts({1: 10.0, 2: 12.0, 3: 15.0}, 2) == {
        "delta_true_minus_K1": 2.0, "delta_true_minus_K3": -3.0}


def test_12_k3_anchor_integration_read_only():
    d = mx.load_condition(3, Path("unused"))
    assert d["source"] == "historical_phase9k"
    assert {r["replicate"] for r in d["laplace"]} == set(mx.REPLICATES)
    rows = mx.dataset_rows(3, d["laplace"], d["family"])
    assert len(rows) == 10 and all(r["c_lap_complete"] for r in rows)
    committed = json.loads((mx.PHASE9K_DIR / "paired_summary.json")
                           .read_text("utf-8"))
    by = {x["replicate"]: x for x in committed["datasets"]}
    for r in rows:
        assert r["K_hat_Lap"] == by[r["replicate"]]["K_hat_Lap"]
        assert r["K_hat_Q"] == by[r["replicate"]]["K_hat_Q"]
    rep = pc.PROTOCOL.replicates[0]
    a = pilot.build_dataset(mx.protocol_for(3), rep)
    b = pilot.build_dataset(pc.PROTOCOL, rep)
    assert all(np.array_equal(getattr(a, m), getattr(b, m))
               for m in ("Z", "F", "X", "Y"))


def test_13_phase9k_artifacts_not_written():
    before = {p.name: mx._sha(p) for p in mx.PHASE9K_DIR.iterdir()
              if p.is_file()}
    d = mx.load_condition(3, Path("unused"))
    mx.condition_summary(3, mx.dataset_rows(3, d["laplace"], d["family"]))
    after = {p.name: mx._sha(p) for p in mx.PHASE9K_DIR.iterdir()
             if p.is_file()}
    assert before == after


def test_14_no_local_optimization():
    tree = ast.parse(Path(mx.__file__).read_text("utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names}
    assert imported.isdisjoint({"theta_stationarity_diagnostic",
                                "multistep_local_pilot",
                                "weak_small_gap_one_step",
                                "weak_small_gap_multistep",
                                "derivative_failure_classification"})
