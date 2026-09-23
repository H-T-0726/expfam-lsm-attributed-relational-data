"""Zero-fit tests for the pilot runner, the auditor and the fail-fast opt-in.

No EM is executed here.  A module-scoped autouse fixture turns an accidental
MCEM call into a loud failure, and the runner is exercised with the hybrid
driver replaced by a stub, so the artifact pipeline, the frozen protocol and
the auditor are all covered without a single fit.
"""

from __future__ import annotations

import csv
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import audit_family_selection_pilot as auditor              # noqa: E402
import em_runner                                            # noqa: E402
import family_selection                                     # noqa: E402
import run_family_selection_pilot as runner                 # noqa: E402
from data_generator_canonical import f_scale_for_row_norm   # noqa: E402
from em_runner import EMFailFast                            # noqa: E402
from family_selection import (                              # noqa: E402
    PILOT_GATE_BLOCKED,
    PILOT_GATE_PASS,
    SelectorStop,
    pilot_convergence_gate,
    support_gate,
)


@pytest.fixture(autouse=True)
def _forbid_em(monkeypatch):
    """Fail loudly if anything in this module tries to run an EM fit."""

    from model_dual_expfam_consistent import DualExpFamLSMPerColumnConsistent

    def _no_em(*_args, **_kwargs):
        raise AssertionError("an EM fit was attempted inside a zero-fit test")

    monkeypatch.setattr(DualExpFamLSMPerColumnConsistent, "calc_eta_newton", _no_em)


# --------------------------------------------------------------------------
# 1. the fail-fast opt-in on em_runner
# --------------------------------------------------------------------------

def test_failure_policy_defaults_to_legacy():
    """The opt-in must not change the behaviour anything else already relies on."""

    signature = inspect.signature(em_runner.run_em_experimental)
    assert signature.parameters["failure_policy"].default == "legacy"
    assert signature.parameters["numerics_mode"].default == "legacy"


def test_failure_policy_is_validated():
    with pytest.raises(ValueError, match="failure_policy must be"):
        em_runner.run_em_experimental(
            np.zeros((4, 2)), np.zeros((4, 4)), family_x="gaussian",
            family_y="bernoulli", k=1, failure_policy="retry_harder")


def test_fail_fast_exception_is_available_and_specific():
    assert issubclass(EMFailFast, RuntimeError)
    assert EMFailFast is not RuntimeError


def test_hybrid_refit_demands_fail_fast(monkeypatch):
    """The reported refit must run under fail-fast, and be checked afterwards."""

    captured = {}

    def fake_refit(*_args, **kwargs):
        captured.update(kwargs)
        return {"failure_policy": kwargs.get("failure_policy"),
                "retry_count": 0, "replacement_count": 0,
                "seed_rescue_count": 0, "Q_strict": -1.0, "bic": 2.0,
                "num_params": 3, "nan_occurred": False, "q_bic_failed": False}

    def fake_exploration(*_args, **_kwargs):
        return family_selection.ExplorationResult(
            selected_assignment=["bernoulli"], initial_assignment=["bernoulli"],
            selection_trace=[], final_candidate_rows=[], margins={},
            iterations=1, metadata={})

    monkeypatch.setattr(family_selection, "run_em_experimental", fake_refit)
    monkeypatch.setattr(family_selection, "run_family_exploration",
                        fake_exploration)

    X = np.array([[0.0], [1.0], [1.0], [0.0]])
    Y = np.zeros((4, 4))
    result = family_selection.run_hybrid_family_selection(
        X, Y, k=1, ambiguous_start="bernoulli")
    assert captured["failure_policy"] == "fail_fast"
    assert captured["numerics_mode"] == "consistent"
    assert result["integrity"]["retry_count"] == 0
    assert result["integrity"]["replacement_count"] == 0
    assert result["integrity"]["seed_rescue_count"] == 0


@pytest.mark.parametrize("counter", ["retry_count", "replacement_count",
                                     "seed_rescue_count"])
def test_hybrid_rejects_a_refit_that_retried(monkeypatch, counter):
    def fake_refit(*_args, **kwargs):
        payload = {"failure_policy": "fail_fast", "retry_count": 0,
                   "replacement_count": 0, "seed_rescue_count": 0}
        payload[counter] = 1
        return payload

    def fake_exploration(*_args, **_kwargs):
        return family_selection.ExplorationResult(
            selected_assignment=["bernoulli"], initial_assignment=["bernoulli"],
            selection_trace=[], final_candidate_rows=[], margins={},
            iterations=1, metadata={})

    monkeypatch.setattr(family_selection, "run_em_experimental", fake_refit)
    monkeypatch.setattr(family_selection, "run_family_exploration",
                        fake_exploration)

    with pytest.raises(SelectorStop, match=f"{counter}="):
        family_selection.run_hybrid_family_selection(
            np.array([[0.0], [1.0], [1.0], [0.0]]), np.zeros((4, 4)),
            k=1, ambiguous_start="bernoulli")


# --------------------------------------------------------------------------
# 2. the candidate-convergence execution gate
# --------------------------------------------------------------------------

def test_convergence_gate_passes_when_every_candidate_converged():
    trace = [{"column": 2, "iteration": 8, "all_candidates_converged": True}]
    rows = [{"column": 2, "optimiser_converged": True}]
    gate = pilot_convergence_gate(trace, rows)
    assert gate["status"] == PILOT_GATE_PASS
    assert gate["non_converged_selection_rows"] == []


def test_convergence_gate_blocks_on_a_single_non_converged_column():
    trace = [{"column": 2, "iteration": 8, "all_candidates_converged": True},
             {"column": 3, "iteration": 8, "all_candidates_converged": False,
              "selected_family": "poisson", "margin_neg2": 0.4}]
    gate = pilot_convergence_gate(trace, [])
    assert gate["status"] == PILOT_GATE_BLOCKED
    assert len(gate["non_converged_selection_rows"]) == 1
    assert gate["non_converged_selection_rows"][0]["column"] == 3
    # The gate states what must NOT be done in response.
    assert "rerun" in gate["remedy_forbidden"]
    assert "raise the optimiser budget" in gate["remedy_forbidden"]


def test_convergence_gate_also_blocks_from_candidate_rows_alone():
    gate = pilot_convergence_gate([], [{"column": 4,
                                        "optimiser_converged": False}])
    assert gate["status"] == PILOT_GATE_BLOCKED
    assert gate["non_converged_candidate_count"] == 1


# --------------------------------------------------------------------------
# 3. the frozen protocols
# --------------------------------------------------------------------------

def test_smoke_protocol_matches_issue_74_literally():
    p = runner.SMOKE
    assert (p.n, p.d, p.k_true, p.k_fit) == (40, 6, 3, 3)
    assert p.family_x_list == ("gaussian", "gaussian", "bernoulli",
                               "bernoulli", "poisson", "poisson")
    assert p.family_y == "bernoulli"
    assert (p.sigma_x_var, p.w0, p.w) == (1.0, -1.0, 1.0)
    assert p.f_scale == pytest.approx(1.0)
    assert p.f_scale == pytest.approx(f_scale_for_row_norm(0.5, d=6, k=3))
    assert (p.L, p.refit_num_iter) == (5, 8)
    assert [r.data_seed for r in p.replicates] == [941001]
    assert [r.search_seed for r in p.replicates] == [942001]
    assert [r.refit_seed for r in p.replicates] == [943001]
    assert p.expected_em_executions == 4


def test_pilot_protocol_matches_issue_74_literally():
    p = runner.PILOT
    assert (p.n, p.d, p.k_true, p.k_fit) == (75, 12, 3, 3)
    assert list(p.family_x_list) == (["gaussian"] * 3 + ["bernoulli"] * 6
                                     + ["poisson"] * 3)
    assert p.f_scale == pytest.approx(math.sqrt(2.0))
    assert (p.L, p.refit_num_iter) == (5, 8)
    assert [r.data_seed for r in p.replicates] == [951001, 951002, 951003]
    assert [r.search_seed for r in p.replicates] == [952001, 952002, 952003]
    assert [r.refit_seed for r in p.replicates] == [953001, 953002, 953003]
    assert p.expected_em_executions == 12


def test_both_stages_together_stay_within_the_execution_cap():
    total = runner.SMOKE.expected_em_executions + runner.PILOT.expected_em_executions
    assert total == 16
    assert total <= auditor.MAX_TOTAL_EM_EXECUTIONS


def test_protocols_are_frozen_dataclasses():
    with pytest.raises(Exception):
        runner.SMOKE.n = 999                     # type: ignore[misc]
    with pytest.raises(Exception):
        runner.SMOKE.replicates[0].data_seed = 1  # type: ignore[misc]


def test_the_command_line_exposes_no_condition_knobs():
    """A flag for n, K or a seed would let a result choose its own conditions."""

    with pytest.raises(SystemExit):
        runner.main(["--stage", "smoke", "--out", "x", "--n", "10"])
    with pytest.raises(SystemExit):
        runner.main(["--stage", "smoke", "--out", "x", "--seed", "1"])
    with pytest.raises(SystemExit):
        runner.main(["--stage", "smoke", "--out", "x", "--num-iter", "3"])
    with pytest.raises(SystemExit):
        runner.main(["--stage", "invented", "--out", "x"])


def test_both_starts_are_run_on_the_same_data():
    assert runner.STARTS == (("start_B", "bernoulli"), ("start_P", "poisson"))


# --------------------------------------------------------------------------
# 4. dataset construction and provenance rows (generator only, no EM)
# --------------------------------------------------------------------------

def test_build_dataset_follows_the_protocol():
    dataset = runner.build_dataset(runner.SMOKE, runner.SMOKE.replicates[0])
    assert dataset.X.shape == (40, 6)
    assert dataset.metadata["seed"] == 941001
    assert dataset.metadata["family_x_list"] == list(runner.SMOKE.family_x_list)


def test_provenance_rows_flag_an_ambiguous_true_poisson_column():
    protocol = runner.SMOKE
    replicate = protocol.replicates[0]
    dataset = runner.build_dataset(protocol, replicate)
    rows = runner.generator_provenance_rows(protocol, replicate, dataset)
    assert len(rows) == protocol.d
    for row in rows:
        column = row["column"]
        true_family = protocol.family_x_list[column]
        assert row["family_x_true"] == true_family
        values = dataset.X[:, column]
        binary = bool(np.all((values == 0.0) | (values == 1.0)))
        assert row["binary_valued"] == binary
        assert row["ambiguous_true_poisson"] == (true_family == "poisson"
                                                 and binary)


# --------------------------------------------------------------------------
# 5. the artifact pipeline, with the hybrid driver stubbed out
# --------------------------------------------------------------------------

def _fake_hybrid(protocol, dataset, ambiguous_start, search_seed, refit_seed,
                 *, converged=True, correct=True):
    """A synthetic hybrid result with the real gate verdicts."""

    gates = support_gate(dataset.X)
    ambiguous = [g.column for g in gates if g.is_ambiguous]
    initial = [g.candidates[0] if not g.is_ambiguous else ambiguous_start
               for g in gates]
    selected = list(initial)
    candidate_rows = []
    trace = []
    for column in ambiguous:
        truth = protocol.family_x_list[column]
        chosen = truth if correct else (
            "poisson" if truth == "bernoulli" else "bernoulli")
        if chosen not in ("bernoulli", "poisson"):
            chosen = "bernoulli"
        selected[column] = chosen
        for family in ("bernoulli", "poisson"):
            candidate_rows.append({
                "column": column, "candidate_family": family,
                "score": -10.0 if family == chosen else -12.0,
                "neg2_score": 20.0 if family == chosen else 24.0,
                "sigma_sq": "", "optimiser_iterations": 7,
                "optimiser_converged": converged, "loading_norm": 0.5,
            })
        trace.append({
            "iteration": protocol.exploration_num_iter, "column": column,
            "previous_family": ambiguous_start, "selected_family": chosen,
            "margin_neg2": 4.0, "changed": chosen != ambiguous_start,
            "all_candidates_converged": converged,
            "score_bernoulli": -10.0, "score_poisson": -12.0,
        })
    return {
        "gates": [g.as_row() for g in gates],
        "ambiguous_columns": ambiguous,
        "initial_assignment": initial,
        "selected_assignment": selected,
        "selection_trace": trace,
        "candidate_rows": candidate_rows,
        "margins": {c: 4.0 for c in ambiguous},
        "exploration_metadata": {"selector_version": "test"},
        "refit": {"Q_strict": -123.5, "bic": 300.25, "num_params": 21,
                  "nan_occurred": False, "q_bic_failed": False,
                  "runtime_s": 0.1, "w0": -1.0, "w": 0.9,
                  "Z_est": np.asarray(dataset.Z) if hasattr(dataset, "Z")
                  else np.zeros((dataset.X.shape[0], protocol.k_true)),
                  "failure_policy": "fail_fast", "retry_count": 0,
                  "replacement_count": 0, "seed_rescue_count": 0},
        "n_gaussian_x_cols": sum(1 for f in selected if f == "gaussian"),
        "ambiguous_start": ambiguous_start,
        "search_seed": search_seed, "refit_seed": refit_seed,
        "integrity": {"failure_policy": "fail_fast", "retry_count": 0,
                      "replacement_count": 0, "seed_rescue_count": 0,
                      "exploration_retry_count": 0,
                      "exploration_replacement_count": 0,
                      "exploration_seed_rescue_count": 0},
        "convergence_gate": pilot_convergence_gate(trace, candidate_rows),
    }


@pytest.fixture
def approved():
    """The human approval is now recorded in the module itself.

    Issue #74 froze the refit num_iter but not the exploration length; a human
    approved the latter separately (Issue #74 Human Gate comment), so the
    record ships approved.  This fixture stays as the seam the pipeline tests
    declare, and asserts the precondition they rely on rather than faking it.
    """

    assert runner.EXPLORATION_NUM_ITER_APPROVAL["approved"] is True


@pytest.fixture
def stub_hybrid(monkeypatch):
    """Replace the hybrid driver so the pipeline runs without any EM."""

    def _install(*, fail_at=None, fail_on_call=1, **options):
        """Replace the hybrid driver.

        The stub still calls ``execution_hook`` around each of the two EM
        executions, so the ledger sees the same attempt sequence a real run
        would produce.  ``fail_at`` makes the stub raise after signalling that
        an execution has STARTED, which is what a real fit dying mid-run looks
        like from the runner's side.
        """

        calls = {"n": 0}

        # The auditor rightly raises a HIGH finding when the worktree was
        # dirty at run time, since the recorded git SHA would not describe the
        # code that ran. These tests must not depend on the state of the
        # developer working tree, so the recorded flag is pinned; the test
        # that checks a dirty tree blocks progression edits runinfo directly.
        monkeypatch.setattr(runner, "_git_dirty", lambda: False)

        def fake(X, Y, *, k, ambiguous_start, family_y, L,
                 exploration_num_iter, refit_num_iter, search_seed,
                 refit_seed, verbose=False, execution_hook=None):
            protocol = runner.PROTOCOLS[_install.stage]
            calls["n"] += 1
            should_fail = (fail_at is not None
                           and calls["n"] == fail_on_call)

            def notify(kind, status, seed):
                if execution_hook is not None:
                    execution_hook(kind, status, {"seed": seed, "k": k, "L": L,
                                                  "ambiguous_start":
                                                  ambiguous_start})

            notify("exploration", "STARTED", search_seed)
            if should_fail and fail_at == "exploration":
                raise RuntimeError("synthetic exploration failure")
            notify("exploration", "SUCCESS", search_seed)

            notify("refit", "STARTED", refit_seed)
            if should_fail and fail_at == "refit":
                raise RuntimeError("synthetic refit failure")
            notify("refit", "SUCCESS", refit_seed)

            class _D:
                pass

            dataset = _D()
            dataset.X = X
            dataset.Z = np.zeros((X.shape[0], protocol.k_true))
            return _fake_hybrid(protocol, dataset, ambiguous_start,
                                search_seed, refit_seed, **options)

        monkeypatch.setattr(runner, "run_hybrid_family_selection", fake)

    _install.stage = "smoke"
    return _install


def test_execute_writes_every_required_artifact(tmp_path, stub_hybrid, approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    summary = runner.execute("smoke", out)

    for name in runner.ARTIFACT_NAMES:
        assert (out / name).is_file(), name
    # audit_report.json is the auditor's output, not the runner's.
    assert not (out / "audit_report.json").exists()

    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["n"] == 40 and protocol["expected_em_executions"] == 4
    runinfo = json.loads((out / "runinfo.json").read_text(encoding="utf-8"))
    assert runinfo["em_executions"] == 4
    assert runinfo["failure_policy"] == "fail_fast"
    assert summary["convergence_gate"]["status"] == PILOT_GATE_PASS
    assert set(summary["claim_boundary"]) >= set(
        auditor.REQUIRED_CLAIM_BOUNDARY_KEYS)


def test_execute_refuses_to_overwrite_a_recorded_run(tmp_path, stub_hybrid, approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)
    with pytest.raises(runner.RunnerStop, match="never overwritten"):
        runner.execute("smoke", out)


def test_gate_decided_columns_never_appear_in_family_scores(tmp_path, stub_hybrid,
                                                            approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)

    with (out / "support_gate.csv").open(encoding="utf-8") as handle:
        gates = list(csv.DictReader(handle))
    with (out / "family_scores.csv").open(encoding="utf-8") as handle:
        scores = list(csv.DictReader(handle))

    scored = {(r["replicate"], r["start_label"], r["column"]) for r in scores}
    gate_only = {(r["replicate"], r["start_label"], r["column"])
                 for r in gates if r["decided_by"] == "gate"}
    assert scored & gate_only == set()
    assert scored


def test_procrustes_rmse_is_invariant_to_the_arbitrary_rotation():
    """Z is identified only up to an orthogonal transform."""

    rng = np.random.default_rng(11)
    Z_true = rng.standard_normal((30, 3))
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    assert runner.procrustes_rmse_z(Z_true, Z_true) == pytest.approx(0.0,
                                                                    abs=1e-12)
    assert runner.procrustes_rmse_z(Z_true @ rotation, Z_true) == pytest.approx(
        0.0, abs=1e-10)
    assert runner.procrustes_rmse_z(Z_true + 1.0, Z_true) > 0.0


def test_fit_results_carry_a_real_rmse_not_an_empty_column(tmp_path, stub_hybrid,
                                                           approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)
    with (out / "fit_results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        assert row["rmse_Z"] != ""
        assert math.isfinite(float(row["rmse_Z"]))
        assert row["w0_est"] != "" and row["w_est"] != ""


# --------------------------------------------------------------------------
# 6. the auditor, on runs it did not produce
# --------------------------------------------------------------------------

def test_auditor_does_not_import_the_runner():
    """An auditor that shares the runner's constants only checks self-consistency."""

    source = (_HERE / "audit_family_selection_pilot.py").read_text(
        encoding="utf-8")
    assert "run_family_selection_pilot" not in source
    assert "from family_selection" not in source
    assert "import family_selection" not in source
    assert "data_generator" not in source
    # It carries its own transcription of the frozen conditions.
    assert auditor.EXPECTED["smoke"]["n"] == 40
    assert auditor.EXPECTED["pilot"]["family_x_list"].count("bernoulli") == 6


def test_auditor_passes_a_well_formed_run(tmp_path, stub_hybrid, approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)

    report = auditor.audit(out)
    assert report["verdict"] == "PASS", report["findings"]
    assert report["blocker_count"] == 0
    assert report["convergence_gate"] == PILOT_GATE_PASS
    assert (out / "audit_report.json").is_file()
    assert report["score_decided_columns"] > 0
    assert report["gate_decided_columns"] > 0


def test_auditor_passes_the_pilot_stage_too(tmp_path, stub_hybrid, approved):
    stub_hybrid()
    stub_hybrid.stage = "pilot"
    out = tmp_path / "pilot_run"
    runner.execute("pilot", out)
    report = auditor.audit(out)
    assert report["verdict"] == "PASS", report["findings"]


def test_auditor_blocks_a_run_with_a_non_converged_candidate(tmp_path, stub_hybrid,
                                                             approved):
    stub_hybrid(converged=False)
    out = tmp_path / "smoke_run"
    summary = runner.execute("smoke", out)
    assert summary["convergence_gate"]["status"] == PILOT_GATE_BLOCKED

    report = auditor.audit(out)
    assert report["convergence_gate"] == PILOT_GATE_BLOCKED
    assert report["non_converged_candidate_rows"] > 0
    # The gate is an integrity condition, not an accuracy threshold: the run
    # itself is still internally consistent, so the audit does not FAIL on it.
    assert report["verdict"] == "PASS"


def test_auditor_reports_a_mis_selection_without_failing(tmp_path, stub_hybrid, approved):
    """Getting the family wrong is a result, not an integrity violation."""

    stub_hybrid(correct=False)
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)
    report = auditor.audit(out)
    assert report["verdict"] == "PASS", report["findings"]


def test_auditor_reports_missing_artifacts(tmp_path, stub_hybrid, approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)
    (out / "family_scores.csv").unlink()
    report = auditor.audit(out)
    assert report["verdict"] == "FAIL"
    assert any("missing required artifacts" in f["message"]
               for f in report["findings"])


@pytest.mark.parametrize("field,value", [
    ("n", 41),
    ("L", 4),
    ("refit_num_iter", 9),
    ("k_fit", 2),
    ("family_y", "poisson"),
])
def test_auditor_catches_a_tampered_protocol(tmp_path, stub_hybrid, approved, field, value):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)
    path = out / "protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = auditor.audit(out)
    assert report["verdict"] == "FAIL"
    assert any(f["check"] == "protocol" and field in f["message"]
               for f in report["findings"])


def test_auditor_catches_a_changed_seed(tmp_path, stub_hybrid, approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)
    path = out / "protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["replicates"][0]["data_seed"] = 123456
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = auditor.audit(out)
    assert report["verdict"] == "FAIL"
    assert any("data_seed" in f["message"] for f in report["findings"])


def test_auditor_catches_a_retry(tmp_path, stub_hybrid, approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)
    path = out / "fit_results.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["retry_count"] = "1"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = auditor.audit(out)
    assert report["verdict"] == "FAIL"
    assert any("retry_count" in f["message"] for f in report["findings"])


def test_auditor_catches_a_non_finite_score(tmp_path, stub_hybrid, approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)
    path = out / "family_scores.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["score"] = "nan"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = auditor.audit(out)
    assert report["verdict"] == "FAIL"
    assert any("not finite" in f["message"] for f in report["findings"])


def test_auditor_catches_a_missing_run_and_a_duplicate_row(tmp_path, stub_hybrid,
                                                           approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)
    path = out / "fit_results.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    duplicated = [rows[0], rows[0]]              # start_P dropped, start_B twice
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(duplicated)

    report = auditor.audit(out)
    assert report["verdict"] == "FAIL"
    messages = " ".join(f["message"] for f in report["findings"])
    assert "missing rows" in messages
    assert "duplicate rows" in messages


def test_auditor_catches_a_score_row_for_a_gate_decided_column(tmp_path, stub_hybrid,
                                                               approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)

    gate_path = out / "support_gate.csv"
    with gate_path.open(encoding="utf-8", newline="") as handle:
        gate_rows = list(csv.DictReader(handle))
    gate_only = next(r for r in gate_rows if r["decided_by"] == "gate")

    path = out / "family_scores.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    smuggled = dict(rows[0])
    smuggled["column"] = gate_only["column"]
    smuggled["replicate"] = gate_only["replicate"]
    smuggled["start_label"] = gate_only["start_label"]
    rows.append(smuggled)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = auditor.audit(out)
    assert report["verdict"] == "FAIL"
    assert any("gate decided" in f["message"] for f in report["findings"])


def test_auditor_catches_a_summary_that_disagrees_with_its_rows(tmp_path, stub_hybrid,
                                                                approved):
    stub_hybrid(converged=False)
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)
    path = out / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["convergence_gate"]["status"] = PILOT_GATE_PASS   # a flattering edit
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = auditor.audit(out)
    assert report["verdict"] == "FAIL"
    assert any(f["check"] == "convergence_gate" for f in report["findings"])


def test_auditor_writes_its_report_even_when_it_fails(tmp_path, stub_hybrid, approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)
    (out / "summary.json").unlink()
    auditor.audit(out)
    report = json.loads((out / "audit_report.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "FAIL"
    assert "Do not rerun" in report["note"]


# --------------------------------------------------------------------------
# 7. evidence preservation when a run dies part-way through
# --------------------------------------------------------------------------

def _ledger(out):
    with (out / "execution_ledger.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_run_directory_and_protocol_exist_before_any_em(tmp_path, stub_hybrid,
                                                        approved):
    """A run that dies inside the first fit must still have committed these."""

    stub_hybrid(fail_at="exploration", fail_on_call=1)
    out = tmp_path / "smoke_run"
    with pytest.raises(RuntimeError, match="synthetic exploration failure"):
        runner.execute("smoke", out)

    assert (out / "protocol.json").is_file()
    assert (out / "runinfo.json").is_file()
    assert (out / "execution_ledger.csv").is_file()
    assert (out / runner.FAILURE_ARTIFACT).is_file()
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["n"] == 40


def test_partial_exploration_failure_preserves_its_attempt(tmp_path,
                                                           stub_hybrid,
                                                           approved):
    stub_hybrid(fail_at="exploration", fail_on_call=1)
    out = tmp_path / "smoke_run"
    with pytest.raises(RuntimeError):
        runner.execute("smoke", out)

    rows = _ledger(out)
    assert len(rows) == 1                       # one EM execution was started
    assert rows[0]["execution_kind"] == "exploration"
    assert rows[0]["status"] == "FAILED"
    assert rows[0]["seed"] == "942001"

    runinfo = json.loads((out / "runinfo.json").read_text(encoding="utf-8"))
    assert runinfo["run_status"] == "FAILED"
    assert runinfo["em_executions"] == 1        # attempted, not returned


def test_refit_failure_counts_both_attempts(tmp_path, stub_hybrid, approved):
    stub_hybrid(fail_at="refit", fail_on_call=1)
    out = tmp_path / "smoke_run"
    with pytest.raises(RuntimeError, match="synthetic refit failure"):
        runner.execute("smoke", out)

    rows = _ledger(out)
    assert [r["execution_kind"] for r in rows] == ["exploration", "refit"]
    assert [r["status"] for r in rows] == ["SUCCESS", "FAILED"]
    runinfo = json.loads((out / "runinfo.json").read_text(encoding="utf-8"))
    assert runinfo["em_executions"] == 2


def test_a_failure_in_the_second_pair_keeps_the_first_pair(tmp_path,
                                                           stub_hybrid,
                                                           approved):
    """Partial provenance survives: the completed start_B run is still there."""

    stub_hybrid(fail_at="exploration", fail_on_call=2)
    out = tmp_path / "smoke_run"
    with pytest.raises(RuntimeError):
        runner.execute("smoke", out)

    rows = _ledger(out)
    assert len(rows) == 3            # 2 for start_B, 1 started for start_P
    assert rows[-1]["status"] == "FAILED"
    assert rows[-1]["start_label"] == "start_P"
    # The finished pair rows were written rather than discarded.
    assert (out / "fit_results.csv").is_file()
    with (out / "fit_results.csv").open(encoding="utf-8") as handle:
        fits = list(csv.DictReader(handle))
    assert [r["start_label"] for r in fits] == ["start_B"]
    assert (out / "generator_provenance.csv").is_file()
    # No summary: the run never reached a state worth summarising.
    assert not (out / "summary.json").exists()


def test_failure_json_records_what_is_needed_to_stop(tmp_path, stub_hybrid,
                                                     approved):
    stub_hybrid(fail_at="refit", fail_on_call=1)
    out = tmp_path / "smoke_run"
    with pytest.raises(RuntimeError):
        runner.execute("smoke", out)

    failure = json.loads(
        (out / runner.FAILURE_ARTIFACT).read_text(encoding="utf-8"))
    for key in auditor.REQUIRED_FAILURE_KEYS:
        assert key in failure, key
    assert failure["exception_type"] == "RuntimeError"
    assert "synthetic refit failure" in failure["message"]
    assert failure["attempted_em_executions"] == 2
    assert failure["execution_kind"] == "refit"
    assert failure["start_label"] == "start_B"
    assert failure["retry_count"] == 0
    assert failure["replacement_count"] == 0
    assert failure["seed_rescue_count"] == 0
    assert failure["git_sha"]
    assert "Do not rerun" in failure["note"]


def test_a_failed_run_is_never_silently_resumed(tmp_path, stub_hybrid,
                                                approved):
    stub_hybrid(fail_at="exploration", fail_on_call=1)
    out = tmp_path / "smoke_run"
    with pytest.raises(RuntimeError):
        runner.execute("smoke", out)
    # Re-running into the same directory is refused, so a failed stage cannot
    # be quietly retried on top of its own evidence.
    with pytest.raises(runner.RunnerStop, match="never overwritten"):
        runner.execute("smoke", out)


def test_auditor_audits_a_failed_run(tmp_path, stub_hybrid, approved):
    stub_hybrid(fail_at="refit", fail_on_call=1)
    out = tmp_path / "smoke_run"
    with pytest.raises(RuntimeError):
        runner.execute("smoke", out)

    report = auditor.audit(out)
    assert report["run_status"] == "FAILED"
    assert report["attempted_em_executions"] == 2
    # Preserving a failure correctly is not the same as having produced the
    # result the next stage would be built on.
    assert report["progress_eligible"] is False
    assert report["stage_progression"] == "BLOCKED"
    assert report["blocker_count"] == 0
    assert (out / "audit_report.json").is_file()


def test_auditor_flags_a_failed_run_with_no_failure_evidence(tmp_path,
                                                             stub_hybrid,
                                                             approved):
    stub_hybrid(fail_at="refit", fail_on_call=1)
    out = tmp_path / "smoke_run"
    with pytest.raises(RuntimeError):
        runner.execute("smoke", out)
    (out / runner.FAILURE_ARTIFACT).unlink()

    report = auditor.audit(out)
    assert report["verdict"] == "FAIL"
    assert any("no failure.json was preserved" in f["message"]
               for f in report["findings"])


def test_auditor_catches_an_undercounted_attempt(tmp_path, stub_hybrid,
                                                 approved):
    """The counter must mean attempts, so a lower number is a blocker."""

    stub_hybrid(fail_at="refit", fail_on_call=1)
    out = tmp_path / "smoke_run"
    with pytest.raises(RuntimeError):
        runner.execute("smoke", out)

    path = out / "runinfo.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["em_executions"] = 1                 # "only one really finished"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = auditor.audit(out)
    assert report["verdict"] == "FAIL"
    assert any("must mean attempts" in f["message"]
               for f in report["findings"])


# --------------------------------------------------------------------------
# 8. a HIGH finding blocks stage progression
# --------------------------------------------------------------------------

def test_a_clean_run_is_progress_eligible(tmp_path, stub_hybrid, approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)
    report = auditor.audit(out)
    assert report["verdict"] == "PASS"
    assert report["blocker_count"] == 0
    assert report["high_count"] == 0
    assert report["medium_count"] == 0
    assert report["progress_eligible"] is True
    assert report["stage_progression"] == "ALLOWED"
    assert auditor.main(["--run-dir", str(out)]) == 0


def test_q_bic_failure_blocks_progression_without_failing_the_verdict(
        tmp_path, stub_hybrid, approved):
    """A HIGH finding must stop the next stage even though nothing was violated."""

    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)

    path = out / "fit_results.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["q_bic_failed"] = "True"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = auditor.audit(out)
    assert report["verdict"] == "PASS"           # no protocol violation
    assert report["high_count"] >= 1
    assert report["progress_eligible"] is False
    assert report["stage_progression"] == "BLOCKED"
    assert auditor.main(["--run-dir", str(out)]) == 1


def test_a_dirty_worktree_blocks_progression(tmp_path, stub_hybrid, approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)

    path = out / "runinfo.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["git_dirty"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = auditor.audit(out)
    assert report["high_count"] >= 1
    assert report["progress_eligible"] is False
    assert any("dirty" in f["message"] for f in report["findings"])


def test_unexpected_rows_block_progression(tmp_path, stub_hybrid, approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)

    path = out / "generator_provenance.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    extra = dict(rows[0])
    extra["column"] = "99"
    rows.append(extra)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = auditor.audit(out)
    assert report["high_count"] >= 1
    assert report["progress_eligible"] is False
    assert any("unexpected rows" in f["message"] for f in report["findings"])


# --------------------------------------------------------------------------
# 9. the one condition Issue #74 did not freeze
# --------------------------------------------------------------------------

def test_the_human_approval_is_recorded_with_its_provenance():
    """The value is 8 because a human said so, and the record says where."""

    record = runner.EXPLORATION_NUM_ITER_APPROVAL
    assert record["parameter"] == "exploration_num_iter"
    assert record["value"] == 8
    assert record["approved"] is True
    assert record["approved_by"] == "Human"
    assert "Issue #74" in record["approved_in"]
    assert "issuecomment-5795174457" in record["approval_url"]
    # The approval is scoped; it is not a general licence for this value.
    assert "feasibility pilot only" in record["scope"]
    assert runner.SMOKE.exploration_num_iter == record["value"]
    assert runner.PILOT.exploration_num_iter == record["value"]


def test_execute_still_refuses_when_an_approval_is_withdrawn(tmp_path,
                                                             stub_hybrid,
                                                             monkeypatch):
    """The gate is a live check, not a comment that was true once."""

    stub_hybrid()
    monkeypatch.setitem(runner.EXPLORATION_NUM_ITER_APPROVAL, "approved",
                        False)
    out = tmp_path / "smoke_run"
    with pytest.raises(runner.RunnerStop, match="has not been approved"):
        runner.execute("smoke", out)
    assert not out.exists()                      # no EM, no directory


def test_protocol_records_the_approval_state(tmp_path, stub_hybrid, approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    approval, = protocol["human_approvals"]
    assert approval["parameter"] == "exploration_num_iter"
    assert approval["approved"] is True
    assert approval["approved_by"] == "Human"
    assert "issuecomment-5795174457" in approval["approval_url"]


def test_auditor_blocks_progression_on_an_unapproved_condition(tmp_path,
                                                               stub_hybrid,
                                                               approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)

    path = out / "protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["human_approvals"][0]["approved"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = auditor.audit(out)
    assert report["high_count"] >= 1
    assert report["progress_eligible"] is False
    assert any("without a recorded human approval" in f["message"]
               for f in report["findings"])


def test_worktree_state_is_captured_before_the_run_writes_anything(
        tmp_path, stub_hybrid, approved, monkeypatch):
    """A run writing its own artifacts must not report itself as a change.

    "Was the working tree clean" is a question about the code that ran. If it
    were evaluated after the run had created its output directory inside the
    repository, every run would record git_dirty=True and block its own
    progression.
    """

    stub_hybrid()
    calls = {"n": 0}

    def counting_dirty():
        calls["n"] += 1
        return calls["n"] > 1          # clean only on the first call

    # Installed after stub_hybrid, which also pins _git_dirty.
    monkeypatch.setattr(runner, "_git_dirty", counting_dirty)
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)

    runinfo = json.loads((out / "runinfo.json").read_text(encoding="utf-8"))
    assert runinfo["git_dirty"] is False
    assert calls["n"] == 1             # asked once, before the run began

    report = auditor.audit(out)
    assert report["progress_eligible"] is True


# --------------------------------------------------------------------------
# 10. the composite pilot gate
# --------------------------------------------------------------------------

def test_audit_pass_plus_blocked_convergence_is_not_pilot_eligible(
        tmp_path, stub_hybrid, approved):
    """A clean artifact set does not mean the frozen score was optimised.

    The artifact audit and the convergence gate answer different questions,
    and the next EM stage needs both. Automation reads
    pilot_progress_eligible, not progress_eligible.
    """

    stub_hybrid(converged=False)
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)

    report = auditor.audit(out)
    assert report["verdict"] == "PASS"
    assert report["progress_eligible"] is True
    assert report["convergence_gate"] == PILOT_GATE_BLOCKED
    assert report["pilot_progress_eligible"] is False
    assert auditor.main(["--run-dir", str(out)]) == 1


def test_audit_pass_plus_ready_convergence_is_pilot_eligible(
        tmp_path, stub_hybrid, approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)

    report = auditor.audit(out)
    assert report["verdict"] == "PASS"
    assert report["progress_eligible"] is True
    assert report["convergence_gate"] == PILOT_GATE_PASS
    assert report["pilot_progress_eligible"] is True
    assert auditor.main(["--run-dir", str(out)]) == 0


def test_a_high_finding_also_blocks_the_composite_gate(tmp_path, stub_hybrid,
                                                       approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)

    path = out / "runinfo.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["git_dirty"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = auditor.audit(out)
    assert report["high_count"] >= 1
    assert report["progress_eligible"] is False
    assert report["pilot_progress_eligible"] is False


def test_a_failed_run_is_not_pilot_eligible(tmp_path, stub_hybrid, approved):
    stub_hybrid(fail_at="refit", fail_on_call=1)
    out = tmp_path / "smoke_run"
    with pytest.raises(RuntimeError):
        runner.execute("smoke", out)

    report = auditor.audit(out)
    assert report["run_status"] == "FAILED"
    assert report["pilot_progress_eligible"] is False


def test_the_composite_rule_is_stated_in_the_report(tmp_path, stub_hybrid,
                                                    approved):
    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)
    report = auditor.audit(out)
    assert report["pilot_progression_rule"] == (
        "progress_eligible and convergence_gate == READY_FOR_PILOT")


def test_the_gate_recomputation_reads_the_iteration_trace_too(tmp_path,
                                                              stub_hybrid,
                                                              approved):
    """A candidate that failed to converge in an earlier iteration still counts.

    The A-type update selects a family at every EM iteration and each of those
    decisions feeds the next E-step, so reading only the final iteration's
    candidate rows would let an earlier non-convergence disappear from the
    gate. Gate 74-B3 smoke-v2 hit exactly this: every final candidate
    converged while seven intermediate iterations had one that did not.
    """

    stub_hybrid()
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)

    # The run is clean, so both tables agree and the gate is READY.
    assert auditor.audit(out)["convergence_gate"] == PILOT_GATE_PASS

    # Now mark one INTERMEDIATE trace row as not converged, leaving every
    # family_scores row converged.
    path = out / "selection_trace.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["all_candidates_converged"] = "False"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (out / "family_scores.csv").open(encoding="utf-8") as handle:
        scores = list(csv.DictReader(handle))
    assert all(row["optimiser_converged"] == "True" for row in scores)

    report = auditor.audit(out)
    assert report["convergence_gate"] == PILOT_GATE_BLOCKED
    assert report["non_converged_iteration_rows"] == 1
    assert report["pilot_progress_eligible"] is False


def test_the_gate_recomputation_still_reads_the_candidate_rows(tmp_path,
                                                               stub_hybrid,
                                                               approved):
    """The other source must keep working: both tables, not one or the other."""

    stub_hybrid(converged=False)
    out = tmp_path / "smoke_run"
    runner.execute("smoke", out)
    report = auditor.audit(out)
    assert report["convergence_gate"] == PILOT_GATE_BLOCKED
    assert report["non_converged_candidate_rows"] > 0
