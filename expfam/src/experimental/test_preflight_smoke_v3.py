"""Zero-EM tests for the Gate 74-B6 preflight and installation-provenance audit.

No EM is executed and no preserved artifact directory is read or written.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import audit_family_selection_pilot as auditor                 # noqa: E402
import preflight_smoke_v3 as preflight                         # noqa: E402
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
# the preflight
# --------------------------------------------------------------------------

def test_the_preflight_passes_on_the_current_runner():
    report = preflight.run_preflight()
    assert report["status"] == "PREFLIGHT_PASS", report["findings"]
    assert report["em_executions"] == 0
    assert report["observed"]["ambiguous_loading_installation"] == \
        "selected_candidate_loading"
    assert report["observed"]["candidate_optimizer"] == "bfgs"


def test_the_preflight_does_not_read_any_preserved_run():
    body = (_HERE / "preflight_smoke_v3.py").read_text(
        encoding="utf-8").split('"""', 2)[2]
    assert "results/family_selection" not in body
    assert "smoke_2026" not in body


def test_the_frozen_c1_checks_are_the_smoke_v2_ones():
    payload = runner.SMOKE.as_json()
    payload["n"] = 41
    report_findings = preflight.v2.check_protocol(payload)
    assert any(f["field"] == "n" for f in report_findings)


def test_a_missing_installation_declaration_fails():
    payload = runner.SMOKE.as_json()
    payload.pop("ambiguous_loading_installation")
    findings = preflight.check_installation_declaration(payload)
    assert findings and findings[0]["field"] == \
        "ambiguous_loading_installation"


def test_the_old_discard_semantics_fails():
    payload = runner.SMOKE.as_json()
    payload["ambiguous_loading_installation"] = "parent_calc_F"
    assert preflight.check_installation_declaration(payload)


def test_the_probe_detects_a_selector_that_does_not_install(monkeypatch):
    """A declaration is a promise; the probe checks the behaviour."""

    import family_selection as fs
    from model_dual_expfam_consistent import DualExpFamLSMPerColumnConsistent

    def pre_b5_calc_F(self, X, Z_samples):
        self._iteration += 1
        self.reassign_families(self.select_families(X, Z_samples))
        return DualExpFamLSMPerColumnConsistent.calc_F(self, X, Z_samples)

    monkeypatch.setattr(fs.FamilySelectingPerColumnLSM, "calc_F",
                        pre_b5_calc_F)
    findings = preflight.probe_installation_behaviour()
    fields = {f["field"] for f in findings}
    assert "probe.selected_loading_installed" in fields


def test_the_runner_declares_the_semantics_in_protocol_and_runinfo():
    payload = runner.SMOKE.as_json()
    assert payload["ambiguous_loading_installation"] == \
        "selected_candidate_loading"
    runinfo = runner.build_runinfo(runner.SMOKE, started="t")
    assert runinfo["ambiguous_loading_installation"] == \
        "selected_candidate_loading"


def test_the_c2_protocol_carries_the_same_semantics():
    payload = runner.PILOT.as_json()
    assert payload["ambiguous_loading_installation"] == \
        "selected_candidate_loading"
    assert payload["candidate_optimizer"] == "bfgs"


# --------------------------------------------------------------------------
# the installation-provenance audit check
# --------------------------------------------------------------------------

def _row(**overrides):
    row = {
        "start_label": "start_B", "iteration": "1", "column": "2",
        "selected_family": "bernoulli",
        "selected_candidate_family": "bernoulli",
        "selected_candidate_score": "-10.0",
        "selected_loading": "0.1|0.2|0.3",
        "selected_loading_sha256": "abc",
        "installed_loading_sha256": "abc",
        "selected_loading_installed": "True",
        "loading_installation_policy": "selected_candidate_loading",
    }
    for family in ("bernoulli", "poisson"):
        row.update({f"{family}_optimizer": "bfgs", f"{family}_n_iter": "9",
                    f"{family}_converged": "True",
                    f"{family}_grad_inf": "1e-10",
                    f"{family}_scipy_success": "True",
                    f"{family}_scipy_status": "0"})
    row.update(overrides)
    return row


def _protocol():
    return runner.SMOKE.as_json()


def test_a_complete_row_passes():
    findings: list = []
    auditor._check_installation_provenance(_protocol(), [_row()], findings)
    assert findings == []


@pytest.mark.parametrize("missing", [
    "poisson_grad_inf", "bernoulli_scipy_status", "poisson_n_iter",
    "bernoulli_optimizer", "selected_loading_sha256",
    "installed_loading_sha256", "selected_loading_installed",
])
def test_a_missing_field_is_a_blocker(missing):
    findings: list = []
    auditor._check_installation_provenance(_protocol(),
                                           [_row(**{missing: ""})], findings)
    assert any(f["severity"] == "BLOCKER" and missing in f["message"]
               for f in findings)


def test_a_mismatched_installed_digest_is_a_blocker():
    findings: list = []
    auditor._check_installation_provenance(
        _protocol(), [_row(installed_loading_sha256="zzz")], findings)
    assert any("does not match" in f["message"] for f in findings)


def test_an_uninstalled_selection_is_a_blocker():
    findings: list = []
    auditor._check_installation_provenance(
        _protocol(), [_row(selected_loading_installed="False")], findings)
    assert any("was not installed" in f["message"] for f in findings)


def test_a_convergence_flag_contradicting_its_gradient_is_a_blocker():
    findings: list = []
    auditor._check_installation_provenance(
        _protocol(), [_row(poisson_grad_inf="5e-3", poisson_converged="True")],
        findings)
    assert any("implies otherwise" in f["message"] for f in findings)


def test_an_undeclared_policy_is_a_blocker():
    protocol = _protocol()
    protocol["ambiguous_loading_installation"] = None
    findings: list = []
    auditor._check_installation_provenance(protocol, [_row()], findings)
    assert any("declares ambiguous_loading_installation" in f["message"]
               for f in findings)


def test_the_check_is_opt_in():
    import inspect

    parameters = inspect.signature(auditor.audit).parameters
    assert parameters["require_installation_provenance"].default is False


def test_end_to_end_audit_with_the_check(tmp_path, monkeypatch):
    """Runner with a stubbed driver whose trace rows carry full evidence."""

    import numpy as np
    from family_selection import pilot_convergence_gate, support_gate

    monkeypatch.setattr(runner, "_git_dirty", lambda: False)

    def fake(X, Y, *, k, ambiguous_start, family_y, L, exploration_num_iter,
             refit_num_iter, search_seed, refit_seed, verbose=False,
             execution_hook=None):
        for kind, seed in (("exploration", search_seed),
                           ("refit", refit_seed)):
            execution_hook(kind, "STARTED", {"seed": seed})
            execution_hook(kind, "SUCCESS", {"seed": seed})
        gates = support_gate(X)
        ambiguous = [g.column for g in gates if g.is_ambiguous]
        selected = [g.candidates[0] if not g.is_ambiguous else "bernoulli"
                    for g in gates]
        trace, cand = [], []
        for column in ambiguous:
            for iteration in range(1, exploration_num_iter + 1):
                row = _row(column=column, iteration=iteration)
                row = {k: (True if v == "True" else v) for k, v in row.items()}
                row.update({"score_bernoulli": -10.0, "score_poisson": -12.0,
                            "margin_neg2": 4.0, "all_candidates_converged": True,
                            "previous_family": ambiguous_start,
                            "changed": False, "candidate_optimizer": "bfgs"})
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
            "n_gaussian_x_cols": 2, "ambiguous_start": ambiguous_start,
            "search_seed": search_seed, "refit_seed": refit_seed,
            "integrity": {"failure_policy": "fail_fast", "retry_count": 0,
                          "replacement_count": 0, "seed_rescue_count": 0},
            "convergence_gate": pilot_convergence_gate(trace, cand),
        }

    monkeypatch.setattr(runner, "run_hybrid_family_selection", fake)
    out = tmp_path / "run"
    runner.execute("smoke", out)

    report = auditor.audit(out, "bfgs", True)
    assert report["verdict"] == "PASS", report["findings"]
    assert report["installation_provenance_required"] is True
    assert report["pilot_progress_eligible"] is True

    # Now break one row's installation evidence and re-audit.
    path = out / "selection_trace.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[0]["installed_loading_sha256"] = "tampered"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = auditor.audit(out, "bfgs", True)
    assert report["verdict"] == "FAIL"
    assert report["pilot_progress_eligible"] is False
    json.dumps(report)
