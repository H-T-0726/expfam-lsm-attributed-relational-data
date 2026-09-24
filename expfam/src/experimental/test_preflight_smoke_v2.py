"""Zero-EM tests for the Gate 74-B3 differential preflight and its audit hook.

No EM is executed and no preserved artifact directory is read or written.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import audit_family_selection_pilot as auditor                 # noqa: E402
import preflight_smoke_v2 as preflight                         # noqa: E402
import run_family_selection_pilot as runner                    # noqa: E402
from family_selection import (                                 # noqa: E402
    OPTIMIZER_ADAM,
    OPTIMIZER_BFGS,
    PHASE9C_CANDIDATE_OPTIMIZER,
)


@pytest.fixture(autouse=True)
def _forbid_em(monkeypatch):
    """Fail loudly if anything in this module tries to run an EM fit."""

    import em_runner
    from model_dual_expfam_consistent import DualExpFamLSMPerColumnConsistent

    def _no_em(*_args, **_kwargs):
        raise AssertionError("an EM fit was attempted inside a zero-EM test")

    monkeypatch.setattr(em_runner, "run_em_experimental", _no_em)
    monkeypatch.setattr(DualExpFamLSMPerColumnConsistent, "calc_eta_newton",
                        _no_em)


# --------------------------------------------------------------------------
# 1. the preflight transcribes C1 rather than reading the historical smoke
# --------------------------------------------------------------------------

def _module_body(name: str) -> str:
    """Source with the module docstring removed.

    The docstrings deliberately name the preserved directories to say the code
    does not read them, so a check over the whole file would trip over its own
    explanation.
    """

    return (_HERE / name).read_text(encoding="utf-8").split('"""', 2)[2]


def test_the_preflight_does_not_read_the_historical_smoke():
    body = _module_body("preflight_smoke_v2.py")
    assert "smoke_20260923" not in body
    assert "results/family_selection" not in body
    report = preflight.run_preflight()
    assert report["reads_historical_smoke"] is False
    assert report["em_executions"] == 0


def test_the_frozen_c1_literals_match_the_issue_protocol():
    c1 = preflight.FROZEN_C1
    assert (c1["n"], c1["d"], c1["k_true"], c1["k_fit"]) == (40, 6, 3, 3)
    assert c1["family_x_list"] == ["gaussian", "gaussian", "bernoulli",
                                   "bernoulli", "poisson", "poisson"]
    assert c1["family_y"] == "bernoulli"
    assert (c1["sigma_x_var"], c1["w0"], c1["w"], c1["f_scale"]) == \
        (1.0, -1.0, 1.0, 1.0)
    assert (c1["L"], c1["exploration_num_iter"], c1["refit_num_iter"]) == \
        (5, 8, 8)
    assert (c1["data_seed"], c1["search_seed"], c1["refit_seed"]) == \
        (941001, 942001, 943001)
    assert c1["starts"] == [{"label": "start_B", "ambiguous_start": "bernoulli"},
                            {"label": "start_P", "ambiguous_start": "poisson"}]
    assert c1["expected_em_executions"] == 4


def test_the_preflight_passes_on_the_current_runner():
    report = preflight.run_preflight()
    assert report["status"] == "PREFLIGHT_PASS", report["findings"]
    assert report["findings"] == []
    assert report["observed"]["candidate_optimizer"] == OPTIMIZER_BFGS
    assert report["observed"]["ambiguous_columns"] == [2, 3]
    assert report["observed"]["start_assignments"]["start_B"][2] == "bernoulli"
    assert report["observed"]["start_assignments"]["start_P"][2] == "poisson"
    # The gate-decided columns are identical between the two starts.
    start_b = report["observed"]["start_assignments"]["start_B"]
    start_p = report["observed"]["start_assignments"]["start_P"]
    assert [start_b[i] for i in (0, 1, 4, 5)] == \
        [start_p[i] for i in (0, 1, 4, 5)]


# --------------------------------------------------------------------------
# 2. the preflight actually catches a scientific difference
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("n", 41),
    ("d", 7),
    ("k_fit", 2),
    ("family_y", "poisson"),
    ("sigma_x_var", 2.0),
    ("w0", -0.5),
    ("w", 1.5),
    ("f_scale", 1.4142135623730951),
    ("L", 4),
    ("exploration_num_iter", 12),
    ("refit_num_iter", 12),
    ("expected_em_executions", 8),
])
def test_a_changed_scientific_field_fails_the_preflight(field, value):
    payload = runner.SMOKE.as_json()
    payload[field] = value
    findings = preflight.check_protocol(payload)
    assert any(f["field"] == field for f in findings), findings


def test_a_changed_family_pattern_fails_the_preflight():
    payload = runner.SMOKE.as_json()
    payload["family_x_list"] = ["gaussian"] * 6
    findings = preflight.check_protocol(payload)
    assert any(f["field"] == "family_x_list" for f in findings)


@pytest.mark.parametrize("seed_field", ["data_seed", "search_seed",
                                        "refit_seed"])
def test_a_changed_seed_fails_the_preflight(seed_field):
    payload = runner.SMOKE.as_json()
    payload["replicates"][0][seed_field] = 111111
    findings = preflight.check_protocol(payload)
    assert any(f["field"] == seed_field for f in findings)


def test_a_changed_start_policy_fails_the_preflight():
    payload = runner.SMOKE.as_json()
    payload["starts"] = [{"label": "start_B", "ambiguous_start": "bernoulli"},
                         {"label": "start_B2", "ambiguous_start": "bernoulli"}]
    findings = preflight.check_protocol(payload)
    assert any(f["field"] == "starts" for f in findings)


def test_reverting_the_optimizer_fails_the_preflight():
    """The approved difference is checked positively, not merely tolerated."""

    payload = runner.SMOKE.as_json()
    payload["candidate_optimizer"] = OPTIMIZER_ADAM
    findings = preflight.check_protocol(payload)
    assert any(f["field"] == "candidate_optimizer" for f in findings)


@pytest.mark.parametrize("key,value", [
    ("maxiter", 500),
    ("gtol", 1e-6),
    ("method", "L-BFGS-B"),
    ("jac", None),
])
def test_a_changed_bfgs_setting_fails_the_preflight(key, value):
    payload = runner.SMOKE.as_json()
    payload["candidate_optimizer_settings"][key] = value
    findings = preflight.check_protocol(payload)
    assert any(key in f["field"] for f in findings)


def test_a_fallback_solver_fails_the_preflight():
    payload = runner.SMOKE.as_json()
    payload["candidate_optimizer_settings"]["fallback_solvers"] = ["Powell"]
    findings = preflight.check_protocol(payload)
    assert any("fallback_solvers" in f["field"] for f in findings)


def test_a_changed_convergence_threshold_fails_the_preflight():
    payload = runner.SMOKE.as_json()
    payload["candidate_convergence_rule"]["convergence_grad_inf_tol"] = 1e-4
    findings = preflight.check_protocol(payload)
    assert any("convergence_grad_inf_tol" in f["field"] for f in findings)


# --------------------------------------------------------------------------
# 3. generator provenance
# --------------------------------------------------------------------------

def test_a_changed_generator_version_or_rng_order_fails_the_preflight():
    dataset = runner.build_dataset(runner.SMOKE, runner.SMOKE.replicates[0])
    metadata = dict(dataset.metadata)
    assert preflight.check_generator(metadata) == []

    changed = dict(metadata, generator_version="canonical-clean-mixed-v2")
    assert any(f["field"] == "generator_version"
               for f in preflight.check_generator(changed))

    reordered = dict(metadata,
                     rng_consumption_order=["Z", "F", "Y",
                                            "X_columns_ascending"])
    assert any(f["field"] == "rng_consumption_order"
               for f in preflight.check_generator(reordered))


def test_a_changed_generator_seed_fails_the_preflight():
    dataset = runner.build_dataset(runner.SMOKE, runner.SMOKE.replicates[0])
    changed = dict(dataset.metadata, seed=999999)
    assert any(f["field"] == "generator.seed"
               for f in preflight.check_generator(changed))


def test_the_cli_reports_and_can_write_a_report(tmp_path, capsys):
    out = tmp_path / "preflight.json"
    assert preflight.main(["--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "PREFLIGHT_PASS"
    assert payload["gate"] == "74-B3"
    assert "PREFLIGHT_PASS" in capsys.readouterr().out


# --------------------------------------------------------------------------
# 4. the auditor hook for the expected candidate optimizer
# --------------------------------------------------------------------------

def test_the_optimizer_check_is_opt_in(tmp_path, monkeypatch):
    """A run recorded before the field existed is audited on its own terms."""

    import inspect

    parameters = inspect.signature(auditor.audit).parameters
    # Both later checks are opt-in: a run recorded before either field
    # existed is audited on its own terms, not retroactively failed.
    assert parameters["expect_candidate_optimizer"].default is None
    assert parameters["require_installation_provenance"].default is False


def test_the_runner_protocol_declares_the_optimizer():
    payload = runner.SMOKE.as_json()
    assert payload["candidate_optimizer"] == PHASE9C_CANDIDATE_OPTIMIZER
    assert payload["candidate_optimizer_settings"]["maxiter"] == 2000
    assert payload["candidate_optimizer_settings"]["gtol"] == 1e-10
    assert payload["candidate_convergence_rule"][
        "convergence_grad_inf_tol"] == 1e-8
    assert payload["candidate_convergence_rule"][
        "scipy_success_is_criterion"] is False


def test_the_auditor_transcribes_the_approved_optimizer():
    assert auditor.APPROVED_CANDIDATE_OPTIMIZER == "bfgs"
    assert auditor.APPROVED_OPTIMIZER_SETTINGS["maxiter"] == 2000
    assert auditor.APPROVED_OPTIMIZER_SETTINGS["gtol"] == 1e-10
    assert auditor.APPROVED_CONVERGENCE_GRAD_INF_TOL == 1e-8


def test_the_auditor_flags_a_row_whose_convergence_contradicts_its_gradient():
    findings: list = []
    protocol = runner.SMOKE.as_json()
    rows = [{
        "replicate": "rep1", "start_label": "start_B", "column": "2",
        "candidate_family": "bernoulli", "optimizer": "bfgs",
        "optimiser_gradient_inf": "1.0",      # far above 1e-8
        "optimiser_converged": "True",
    }]
    auditor._check_candidate_optimizer(protocol, rows, "bfgs", findings)
    assert any("implies" in f["message"] for f in findings)


def test_the_auditor_flags_the_wrong_optimizer():
    findings: list = []
    protocol = runner.SMOKE.as_json()
    rows = [{
        "replicate": "rep1", "start_label": "start_B", "column": "2",
        "candidate_family": "bernoulli", "optimizer": "adam",
        "optimiser_gradient_inf": "1e-12", "optimiser_converged": "True",
    }]
    auditor._check_candidate_optimizer(protocol, rows, "bfgs", findings)
    assert any("optimizer is" in f["message"] for f in findings)


def test_the_auditor_flags_a_protocol_that_declares_another_optimizer():
    findings: list = []
    protocol = runner.SMOKE.as_json()
    protocol["candidate_optimizer"] = "adam"
    auditor._check_candidate_optimizer(protocol, [], "bfgs", findings)
    assert any("declares candidate_optimizer" in f["message"]
               for f in findings)


def test_prior_artifact_directories_are_never_touched_by_b3_code():
    for module in ("preflight_smoke_v2.py",):
        body = _module_body(module)
        for preserved in ("smoke_20260923", "optimizer_validation_20260923",
                          "optimizer_validation_analytic_jac_20260923",
                          "optimizer_migration_bfgs_20260923"):
            assert preserved not in body
