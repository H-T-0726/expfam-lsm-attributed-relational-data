"""Zero-EM checks for the Phase 9E differences only (Issue #81).

Unchanged Phase 9D behaviour is covered by test_joint_family_k_selection.py.
The stub driver and the real-EM guard are reused from there.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import run_family_selection_pilot as pilot                     # noqa: E402
import run_joint_family_k_selection as joint                   # noqa: E402
import run_k_repeatability as rep9e                            # noqa: E402
from test_joint_family_k_selection import Stub, _forbid_em     # noqa: E402,F401


def _run(tmp_path, stub=None):
    out = tmp_path / "run"
    joint.execute(out, protocol=rep9e.PROTOCOL, driver=stub or Stub(),
                  authorization=rep9e.AUTHORIZATION)
    return out, rep9e.analyse(out)


def test_the_frozen_differences():
    p = rep9e.PROTOCOL
    assert [(r.label, r.data_seed, r.search_seed, r.refit_seed)
            for r in p.replicates] == [
        (f"rep{r:02d}", 971000 + r, 972000 + r, 973000 + r)
        for r in range(1, 21)]
    assert p.starts == (("start_B", "bernoulli"),)
    assert list(p.k_candidates) == [1, 2, 3, 4, 5]
    assert p.expected_em_executions == 200
    assert p.failure_scope == "dataset"
    # every other scientific setting is Phase 9D's
    changed = {"stage", "replicates", "starts", "failure_scope"}
    for field in joint.JointProtocol.__dataclass_fields__:
        if field not in changed:
            assert getattr(p, field) == getattr(joint.PROTOCOL, field), field
    assert joint.PROTOCOL.starts == joint.STARTS          # 9D unchanged
    assert joint.PROTOCOL.failure_scope == "run"


def test_one_draw_per_dataset_and_no_phase9d_data(tmp_path, monkeypatch):
    draws = []
    real = pilot.generate_canonical_mixed_data
    monkeypatch.setattr(pilot, "generate_canonical_mixed_data",
                        lambda **kw: (draws.append(kw["seed"]), real(**kw))[1])
    out, summary = _run(tmp_path)
    assert draws == [971000 + r for r in range(1, 21)]
    assert not {961001, 961002, 961003} & set(draws)
    assert summary["datasets"]["denominator"] == 20
    ledger = pilot_rows(out / "execution_ledger.csv")
    assert len(ledger) == 200 and {r["start_label"] for r in ledger} == \
        {"start_B"}


def pilot_rows(path):
    import csv
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_delta_23_and_the_decomposition_identity(tmp_path):
    out, summary = _run(tmp_path)
    rows = pilot_rows(out / "k_repeatability.csv")
    for row in rows:
        d23 = float(row["C_Q_k3"]) - float(row["C_Q_k2"])
        assert float(row["delta_23"]) == d23
        assert math.isclose(d23, float(row["penalty_increase_23"])
                            - float(row["fit_gain_23"]), abs_tol=1e-8)
    assert summary["evidence_checks"]["blocking_findings"] == []
    assert summary["evidence_checks"]["criterion_mismatch"] == []
    assert summary["P2_delta_23"]["negative_prefers_k3"] == 20
    assert summary["P1_k_hat_distribution"]["counts"][3] == 20
    assert rep9e.delta_23({2: 5.0, 3: 5.0}) == 0.0
    assert joint.select_k_hat({2: 5.0, 3: 5.0}) == 2        # tie -> K=2


def test_an_isolated_failure_is_recorded_not_replaced(tmp_path):
    stub = Stub(fail_at=("rep10005", "start_B", 4, "refit"))
    out, summary = _run(tmp_path, stub)
    assert summary["datasets"]["completed"] == 19
    assert summary["datasets"]["denominator"] == 19
    [failed] = summary["datasets"]["incomplete"]
    assert (failed["replicate"], failed["k"], failed["replaced"]) == \
        ("rep05", 4, False)
    assert "rep05" not in summary["P2_delta_23"]["values"]
    ledger = pilot_rows(out / "execution_ledger.csv")
    assert len(ledger) == 198                  # rep05 stopped at K=4 refit
    assert {int(r["seed"]) for r in ledger} <= (
        {972000 + r for r in range(1, 21)} | {973000 + r for r in range(1, 21)})
    runinfo = json.loads((out / "runinfo.json").read_text("utf-8"))
    assert runinfo["run_status"] == "SUCCESS_WITH_INCOMPLETE_DATASETS"


def test_a_systemic_error_stops_the_whole_study(tmp_path):
    def broken(X, Y, **kwargs):
        raise KeyError("programming error")

    with pytest.raises(KeyError):
        joint.execute(tmp_path / "run", protocol=rep9e.PROTOCOL,
                      driver=broken, authorization=rep9e.AUTHORIZATION)
    assert (tmp_path / "run" / "failure.json").is_file()
