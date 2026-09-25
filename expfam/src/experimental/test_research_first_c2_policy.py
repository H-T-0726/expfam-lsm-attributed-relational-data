"""Zero-EM contract tests for the research-first C2 policy (Issue #74).

The pilot stage separates technical validity from the candidate-convergence
diagnostic; the tolerance, the optimiser and the smoke semantics are unchanged.
No EM is executed and no preserved artifact directory is read or written.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import audit_family_selection_pilot as auditor                 # noqa: E402
import family_selection as fs                                  # noqa: E402
import run_family_selection_pilot as runner                    # noqa: E402


@pytest.fixture(autouse=True)
def _forbid_em(monkeypatch):
    import em_runner
    from model_dual_expfam_consistent import DualExpFamLSMPerColumnConsistent

    def _no_em(*_args, **_kwargs):
        raise AssertionError("an EM fit was attempted inside a zero-EM test")

    monkeypatch.setattr(em_runner, "run_em_experimental", _no_em)
    monkeypatch.setattr(DualExpFamLSMPerColumnConsistent, "calc_eta_newton",
                        _no_em)


# --------------------------------------------------------------------------
# what does not change
# --------------------------------------------------------------------------

def test_the_tolerance_and_optimiser_settings_are_unchanged():
    assert fs.CANDIDATE_GRAD_INF_TOL == 1e-8
    assert fs.BFGS_MAXITER == 2000
    assert fs.BFGS_GTOL == 1e-10
    assert fs.PHASE9C_CANDIDATE_OPTIMIZER == "bfgs"
    assert fs.LOADING_INSTALLATION_POLICY == "selected_candidate_loading"


def test_the_smoke_protocol_does_not_carry_the_policy():
    assert "progression_policy" not in runner.SMOKE.as_json()


def test_the_pilot_protocol_declares_the_policy_and_is_the_frozen_c2():
    payload = runner.PILOT.as_json()
    policy = payload["progression_policy"]
    assert policy["policy"] == auditor.RESEARCH_FIRST_POLICY_NAME
    assert policy["convergence_threshold_changed"] is False
    assert policy["approved"] is True
    findings: list = []
    auditor._check_protocol(payload, auditor.EXPECTED["pilot"], findings)
    auditor._check_research_first_policy(payload, findings)
    assert findings == []
    assert payload["expected_em_executions"] == 12
    assert payload["exploration_num_iter"] == 8
    assert payload["refit_num_iter"] == 8
    assert payload["candidate_optimizer"] == auditor.APPROVED_CANDIDATE_OPTIMIZER
    assert payload["ambiguous_loading_installation"] == \
        auditor.APPROVED_LOADING_INSTALLATION
    runinfo = runner.build_runinfo(runner.PILOT, started="t",
                                   git_sha="x", git_dirty=False)
    assert runinfo["failure_policy"] == "fail_fast"
    assert runinfo["expected_em_executions"] == 12


def test_the_historical_gate_is_unchanged():
    trace = [{"all_candidates_converged": False, "iteration": 1, "column": 3}]
    assert fs.pilot_convergence_gate(trace)["status"] == "BLOCKED_FOR_PILOT"


def test_a_pilot_protocol_without_the_policy_is_a_blocker():
    payload = runner.PILOT.as_json()
    payload.pop("progression_policy")
    findings: list = []
    auditor._check_research_first_policy(payload, findings)
    assert findings and findings[0]["severity"] == "BLOCKER"


@pytest.mark.parametrize("key,value", [("maxiter", 5000), ("gtol", 1e-8)])
def test_changed_optimiser_settings_are_a_blocker(key, value):
    payload = runner.PILOT.as_json()
    payload["candidate_optimizer_settings"][key] = value
    findings: list = []
    auditor._check_research_first_policy(payload, findings)
    assert any(key in f["message"] and f["severity"] == "BLOCKER"
               for f in findings)


def test_a_changed_tolerance_is_a_blocker():
    payload = runner.PILOT.as_json()
    payload["candidate_convergence_rule"]["convergence_grad_inf_tol"] = 1e-7
    findings: list = []
    auditor._check_research_first_policy(payload, findings)
    assert any(f["severity"] == "BLOCKER" for f in findings)


# --------------------------------------------------------------------------
# the diagnostic
# --------------------------------------------------------------------------

def _trace_row(bernoulli_ok=True, poisson_ok=True):
    return {"iteration": 2, "column": 4, "selected_family": "bernoulli",
            "margin_neg2": 20.0,
            "all_candidates_converged": bernoulli_ok and poisson_ok,
            "bernoulli_converged": bernoulli_ok,
            "bernoulli_grad_inf": 1e-10 if bernoulli_ok else 4.3e-8,
            "bernoulli_n_iter": 6, "bernoulli_scipy_success": bernoulli_ok,
            "bernoulli_scipy_status": 0 if bernoulli_ok else 2,
            "poisson_converged": poisson_ok,
            "poisson_grad_inf": 1e-10 if poisson_ok else 2e-8,
            "poisson_n_iter": 7, "poisson_scipy_success": poisson_ok,
            "poisson_scipy_status": 0 if poisson_ok else 2}


def test_all_converged_is_reported_as_such():
    diagnostic = fs.candidate_convergence_diagnostic([_trace_row()])
    assert diagnostic["status"] == "ALL_CONVERGED"
    assert diagnostic["candidate_evaluations"] == 2
    assert diagnostic["candidate_warnings"] == 0


def test_a_finite_miss_is_a_warning_not_a_pass():
    diagnostic = fs.candidate_convergence_diagnostic(
        [_trace_row(), _trace_row(poisson_ok=False)])
    assert diagnostic["status"] == "CONVERGENCE_WARNING"
    assert diagnostic["candidate_warnings"] == 1
    assert diagnostic["selection_rows_with_warning"] == 1
    warning = diagnostic["warnings"][0]
    assert warning["candidate_family"] == "poisson"
    assert warning["grad_inf"] == 2e-8
    assert warning["scipy_status"] == 2
    assert diagnostic["convergence_grad_inf_tol"] == 1e-8


def test_a_final_non_converged_candidate_is_a_warning():
    diagnostic = fs.candidate_convergence_diagnostic(
        [_trace_row()], [{"optimiser_converged": False}])
    assert diagnostic["status"] == "CONVERGENCE_WARNING"


# --------------------------------------------------------------------------
# end to end through the runner and auditor, with the driver stubbed
# --------------------------------------------------------------------------

def _row_strings(row):
    return {k: ("True" if v is True else "False" if v is False else v)
            for k, v in row.items()}


def _fake_driver(warn_on=None):
    """Stub of run_hybrid_family_selection. ``warn_on`` = (column, iteration)."""

    def fake(X, Y, *, k, ambiguous_start, family_y, L, exploration_num_iter,
             refit_num_iter, search_seed, refit_seed, verbose=False,
             execution_hook=None):
        for kind, seed in (("exploration", search_seed),
                           ("refit", refit_seed)):
            execution_hook(kind, "STARTED", {"seed": seed})
            execution_hook(kind, "SUCCESS", {"seed": seed})
        gates = fs.support_gate(X)
        ambiguous = [g.column for g in gates if g.is_ambiguous]
        selected = [g.candidates[0] if not g.is_ambiguous else "bernoulli"
                    for g in gates]
        trace, cand = [], []
        for column in ambiguous:
            for iteration in range(1, exploration_num_iter + 1):
                ok = warn_on != (column, iteration)
                row = _trace_row(poisson_ok=ok)
                row.update({
                    "column": column, "iteration": iteration,
                    "score_bernoulli": -10.0, "score_poisson": -12.0,
                    "previous_family": ambiguous_start, "changed": False,
                    "candidate_optimizer": "bfgs",
                    "selected_candidate_family": "bernoulli",
                    "selected_candidate_score": -10.0,
                    "selected_loading": "0.1|0.2|0.3",
                    "selected_loading_sha256": "abc",
                    "installed_loading_sha256": "abc",
                    "selected_loading_installed": True,
                    "loading_installation_policy":
                        "selected_candidate_loading",
                    "bernoulli_optimizer": "bfgs",
                    "poisson_optimizer": "bfgs"})
                trace.append(row)
            for family in ("bernoulli", "poisson"):
                cand.append({"column": column, "candidate_family": family,
                             "score": -10.0, "neg2_score": 20.0,
                             "optimizer": "bfgs", "optimiser_iterations": 9,
                             "optimiser_converged": True,
                             "optimiser_gradient_inf": 1e-10,
                             "loading_norm": 0.5})
        return {
            "gates": [g.as_row() for g in gates],
            "ambiguous_columns": ambiguous,
            "initial_assignment": selected, "selected_assignment": selected,
            "selection_trace": trace, "candidate_rows": cand,
            "margins": {c: 4.0 for c in ambiguous},
            "exploration_metadata": {},
            "refit": {"Q_strict": -1.0, "bic": 2.0, "num_params": 17,
                      "nan_occurred": False, "q_bic_failed": False,
                      "runtime_s": 0.1, "w0": -1.0, "w": 1.0,
                      "Z_est": np.zeros((X.shape[0], 3)),
                      "failure_policy": "fail_fast", "retry_count": 0,
                      "replacement_count": 0, "seed_rescue_count": 0},
            "n_gaussian_x_cols": 3, "ambiguous_start": ambiguous_start,
            "search_seed": search_seed, "refit_seed": refit_seed,
            "integrity": {"failure_policy": "fail_fast", "retry_count": 0,
                          "replacement_count": 0, "seed_rescue_count": 0},
            "convergence_gate": fs.pilot_convergence_gate(trace, cand),
        }

    return fake


def _run_pilot(tmp_path, monkeypatch, warn_on):
    monkeypatch.setattr(runner, "_git_dirty", lambda: False)
    monkeypatch.setattr(runner, "run_hybrid_family_selection",
                        _fake_driver(warn_on))
    out = tmp_path / "pilot"
    summary = runner.execute("pilot", out)
    return out, summary


def test_a_warning_keeps_the_run_valid_and_is_preserved(tmp_path, monkeypatch):
    out, summary = _run_pilot(tmp_path, monkeypatch, warn_on=(3, 2))
    assert summary["candidate_convergence_diagnostic"]["status"] == \
        "CONVERGENCE_WARNING"
    assert summary["convergence_gate"]["status"] == "BLOCKED_FOR_PILOT"

    report = auditor.audit(out)
    assert report["research_first_policy_applied"] is True
    assert report["technical_validity"] == "VALID", report["findings"]
    diagnostic = report["candidate_convergence_diagnostic"]
    assert diagnostic["status"] == "CONVERGENCE_WARNING"
    # one warning per start per replicate, at column 3 iteration 2
    assert diagnostic["candidate_warnings"] == 6
    # The historical composite is not relabelled as a pass.
    assert report["convergence_gate"] == "BLOCKED_FOR_PILOT"
    assert report["pilot_progress_eligible"] is False


def test_a_clean_run_reports_all_converged(tmp_path, monkeypatch):
    out, _ = _run_pilot(tmp_path, monkeypatch, warn_on=None)
    report = auditor.audit(out)
    assert report["technical_validity"] == "VALID", report["findings"]
    assert report["candidate_convergence_diagnostic"]["status"] == \
        "ALL_CONVERGED"


def _rewrite_first_trace_row(out, **changes):
    path = out / "selection_trace.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[0].update(changes)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_a_non_finite_gradient_is_a_technical_blocker(tmp_path, monkeypatch):
    out, _ = _run_pilot(tmp_path, monkeypatch, warn_on=None)
    _rewrite_first_trace_row(out, poisson_grad_inf="nan",
                             poisson_converged="False",
                             all_candidates_converged="False")
    report = auditor.audit(out)
    assert report["technical_validity"] == "INVALID"
    assert any(f["check"] == "candidate_values" for f in report["findings"])


def test_a_selected_loading_mismatch_is_a_technical_blocker(tmp_path,
                                                            monkeypatch):
    out, _ = _run_pilot(tmp_path, monkeypatch, warn_on=None)
    _rewrite_first_trace_row(out, installed_loading_sha256="tampered")
    report = auditor.audit(out)
    assert report["technical_validity"] == "INVALID"


def test_a_misreported_diagnostic_is_a_technical_blocker(tmp_path,
                                                         monkeypatch):
    out, _ = _run_pilot(tmp_path, monkeypatch, warn_on=(3, 2))
    _rewrite_first_trace_row(out)          # no change: still consistent
    import json
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["candidate_convergence_diagnostic"]["status"] = "ALL_CONVERGED"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    report = auditor.audit(out)
    assert report["technical_validity"] == "INVALID"


def test_the_runner_cli_does_not_stop_on_a_warning_for_the_pilot(tmp_path,
                                                                 monkeypatch):
    monkeypatch.setattr(runner, "_git_dirty", lambda: False)
    monkeypatch.setattr(runner, "run_hybrid_family_selection",
                        _fake_driver((3, 2)))
    assert runner.main(["--stage", "pilot", "--out",
                        str(tmp_path / "cli")]) == 0
