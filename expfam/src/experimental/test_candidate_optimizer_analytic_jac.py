"""Zero-EM tests for the Gate 74-B1R analytic-jac certification and auditor.

No EM is executed, no RNG is used and no preserved artifact directory is
touched.  A module-scoped autouse fixture turns an accidental MCEM call into a
loud failure.
"""

from __future__ import annotations

import csv
import inspect
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

import audit_candidate_optimizer_analytic_jac as b1r_auditor   # noqa: E402
import validate_candidate_optimizer as b1                      # noqa: E402
import validate_candidate_optimizer_analytic_jac as b1r        # noqa: E402
from family_selection import ADAM_LR, ADAM_MAX_ITER, ADAM_TOL  # noqa: E402


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


@pytest.fixture(scope="module")
def Z_samples():
    return b1r.build_z_samples()


# --------------------------------------------------------------------------
# 1. B1R runs on exactly the B1 inputs
# --------------------------------------------------------------------------

def test_the_frozen_inputs_are_the_b1_inputs(Z_samples):
    assert b1r.BASE_VECTORS is b1.BASE_VECTORS
    assert b1r.CASES is b1.CASES
    assert b1r.INITIALISATIONS is b1.INITIALISATIONS
    assert b1r.SAMPLE_SCALES is b1.SAMPLE_SCALES
    assert b1r.FAMILIES is b1.FAMILIES
    assert np.array_equal(Z_samples, b1.build_z_samples())


def test_every_threshold_is_the_b1_threshold():
    assert b1r.OBJECTIVE_MATCH_TOL == b1.OBJECTIVE_MATCH_TOL
    assert b1r.GRADIENT_MATCH_TOL == b1.GRADIENT_MATCH_TOL
    assert b1r.FD_STEP == b1.FD_STEP
    assert b1r.REFERENCE_GRAD_INF_TOL == b1.REFERENCE_GRAD_INF_TOL
    assert b1r.REFERENCE_SCORE_AGREE_TOL == b1.REFERENCE_SCORE_AGREE_TOL
    assert b1r.REFERENCE_LOADING_AGREE_TOL == b1.REFERENCE_LOADING_AGREE_TOL
    assert b1r.ADAM_GAP_PER_OBS_TOL == b1.ADAM_GAP_PER_OBS_TOL
    assert b1r.ADAM_GRAD_INF_TOL == b1.ADAM_GRAD_INF_TOL
    assert (b1r.BFGS_MAXITER, b1r.BFGS_GTOL) == (b1.BFGS_MAXITER, b1.BFGS_GTOL)


def test_production_adam_settings_are_untouched():
    assert (ADAM_MAX_ITER, ADAM_LR, ADAM_TOL) == (50, 0.01, 1e-6)


def test_the_only_change_is_the_reference_jacobian():
    source = inspect.getsource(b1r.solve_reference_analytic_jac)
    assert "jac=negative_jac" in source
    assert "jac=None" not in source
    assert "_column_gradient" in source
    # No fallback solver anywhere in the module.
    module_source = (_HERE / "validate_candidate_optimizer_analytic_jac.py"
                     ).read_text(encoding="utf-8")
    for forbidden in ("Nelder-Mead", "Powell", "L-BFGS-B", "TNC", "CG",
                      "trust-constr"):
        assert forbidden not in module_source
    assert "default_rng" not in module_source
    assert "smoke_20260923" not in module_source


def test_the_supplied_jacobian_is_the_negative_analytic_gradient(Z_samples):
    """The solver minimises the negative score, so its jac must be negated."""

    from family_selection import _column_gradient

    x = b1r.build_case_column("case_A_balanced_nonseparable")
    loading = np.array([0.31, -0.22, 0.14])
    analytic = _column_gradient(x, Z_samples, loading, "bernoulli", None)

    captured = {}

    def fake_minimize(fun, x0, **kwargs):
        captured["jac"] = kwargs["jac"](loading)
        captured["method"] = kwargs["method"]
        captured["options"] = kwargs["options"]

        class _R:
            x = np.asarray(x0, dtype=float)
            success, status, message, nit, njev = True, 0, "ok", 1, 1
            jac = np.zeros_like(x)

        return _R()

    original = b1r.minimize
    b1r.minimize = fake_minimize
    try:
        b1r.solve_reference_analytic_jac(x, Z_samples, "bernoulli",
                                         [0.0, 0.0, 0.0])
    finally:
        b1r.minimize = original

    assert np.allclose(captured["jac"], -analytic)
    assert captured["method"] == "BFGS"
    assert captured["options"] == {"maxiter": 2000, "gtol": 1e-10}


# --------------------------------------------------------------------------
# 2. the auditor is independent
# --------------------------------------------------------------------------

def test_the_b1r_auditor_imports_neither_validator_nor_selector():
    source = (_HERE / "audit_candidate_optimizer_analytic_jac.py").read_text(
        encoding="utf-8")
    assert "validate_candidate_optimizer" not in source
    assert "import family_selection" not in source
    assert "from family_selection" not in source
    assert "import numpy" not in source


def test_the_auditor_rebuilds_the_frozen_arrays_independently(Z_samples):
    assert b1r_auditor.rebuild_z_samples_digest() == b1.array_digest(Z_samples)
    for case in b1r.CASES:
        column = b1r.build_case_column(case)
        assert b1r_auditor.rebuild_case_digest(case) == b1.array_digest(column)
        assert b1r_auditor.rebuild_case_column(case) == column.tolist()


def test_the_auditor_expects_the_analytic_jacobian():
    assert b1r_auditor.EXPECTED_REFERENCE_SOLVER["jac"] == \
        "analytic_production_gradient"
    assert b1r_auditor.EXPECTED_REFERENCE_SOLVER["finite_difference_jacobian"] \
        is False
    assert b1r_auditor.EXPECTED_ADAM["max_iter"] == ADAM_MAX_ITER


# --------------------------------------------------------------------------
# 3. the pipeline, run for real (still zero EM) into a temporary directory
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def b1r_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("b1r") / "run"
    summary = b1r.execute(out)
    return out, summary


def test_execute_writes_every_artifact(b1r_run):
    out, summary = b1r_run
    for name in b1r.ARTIFACT_NAMES:
        assert (out / name).is_file(), name
    assert summary["em_executions"] == 0
    assert summary["gate"] == "74-B1R"


def test_execute_refuses_to_overwrite(b1r_run):
    out, _ = b1r_run
    with pytest.raises(b1r.ValidationStop, match="never overwritten"):
        b1r.execute(out)


def test_the_regression_guards_still_pass(b1r_run):
    _, summary = b1r_run
    assert summary["objective_checks"]["passed"] == 12
    assert summary["objective_checks"]["max_abs_difference"] <= 1e-10
    assert summary["gradient_checks"]["passed"] == 12
    assert summary["gradient_checks"]["max_abs_difference"] <= 1e-5


def test_the_adam_comparison_runs_only_against_a_certified_reference(b1r_run):
    out, summary = b1r_run
    with (out / "adam_comparison.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if summary["status"] == b1r.STATUS_CERTIFIED:
        assert len(rows) == 12
        assert summary["adam_verdict"] in (b1r.ADAM_CERTIFIED,
                                           b1r.ADAM_NEEDS_HUMAN)
    else:
        # Explicit not-run provenance, never a silently absent table.
        assert len(rows) == 1 and rows[0]["status"] == "NOT_RUN"
        assert rows[0]["reason"]
        assert summary["adam_verdict"] == b1r.ADAM_NOT_RUN


def test_the_converged_flag_is_recorded_but_does_not_certify(b1r_run):
    """B1R must be able to tell an inaccurate optimiser from a strict flag."""

    out, summary = b1r_run
    if summary["status"] != b1r.STATUS_CERTIFIED:
        pytest.skip("the reference was not certified in this environment")
    with (out / "adam_comparison.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all("converged_flag" in row for row in rows)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["converged_flag_is_certification_criterion"] is False
    # The verdict must follow the gap and gradient criteria, not the flag.
    failures = summary["adam_summary"]["failures"]
    expected = b1r.ADAM_CERTIFIED if not failures else b1r.ADAM_NEEDS_HUMAN
    assert summary["adam_verdict"] == expected


def test_the_auditor_passes_the_run(b1r_run):
    out, summary = b1r_run
    report = b1r_auditor.audit(out)
    assert report["verdict"] == "PASS", report["findings"]
    assert report["recomputed_status"] == summary["status"]
    assert report["recomputed_adam_verdict"] == summary["adam_verdict"]
    assert (out / "audit_report.json").is_file()


def test_the_auditor_catches_a_claimed_finite_difference_reference(tmp_path,
                                                                   b1r_run):
    out, _ = b1r_run
    copy = tmp_path / "fd"
    shutil.copytree(out, copy)
    path = copy / "protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["reference_solver"]["jac"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = b1r_auditor.audit(copy)
    assert report["verdict"] == "FAIL"
    assert any("reference solver jac" in f["message"]
               for f in report["findings"])


def test_the_auditor_catches_a_fallback_solver(tmp_path, b1r_run):
    out, _ = b1r_run
    copy = tmp_path / "fallback"
    shutil.copytree(out, copy)
    path = copy / "protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["reference_solver"]["fallback_solvers"] = ["Nelder-Mead"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = b1r_auditor.audit(copy)
    assert report["verdict"] == "FAIL"
    assert any("fallback solver" in f["message"] for f in report["findings"])


def test_the_auditor_catches_a_changed_production_setting(tmp_path, b1r_run):
    out, _ = b1r_run
    copy = tmp_path / "prod"
    shutil.copytree(out, copy)
    path = copy / "protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["adam"]["max_iter"] = 500
    payload["production_optimizer_changed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = b1r_auditor.audit(copy)
    assert report["verdict"] == "FAIL"
    messages = " ".join(f["message"] for f in report["findings"])
    assert "production Adam max_iter" in messages
    assert "must not change the production optimiser" in messages


def test_the_auditor_catches_a_certification_claim_the_rows_do_not_support(
        tmp_path, b1r_run):
    out, _ = b1r_run
    copy = tmp_path / "claim"
    shutil.copytree(out, copy)
    path = copy / "reference_solutions.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    for row in rows:
        row["analytic_grad_inf"] = "1.0"          # clearly above 1e-8
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = b1r_auditor.audit(copy)
    assert report["verdict"] == "FAIL"
    assert any("claim certified" in f["message"] for f in report["findings"])


def test_the_auditor_catches_an_adam_table_without_a_certified_reference(
        tmp_path, b1r_run):
    out, summary = b1r_run
    if summary["status"] == b1r.STATUS_CERTIFIED:
        pytest.skip("the reference certified, so this state cannot arise here")
    copy = tmp_path / "adam"
    shutil.copytree(out, copy)
    path = copy / "adam_comparison.csv"
    path.write_text("case,family,init\na,b,c\n", encoding="utf-8")

    report = b1r_auditor.audit(copy)
    assert report["verdict"] == "FAIL"


def test_the_auditor_catches_a_claimed_em_execution(tmp_path, b1r_run):
    out, _ = b1r_run
    copy = tmp_path / "em"
    shutil.copytree(out, copy)
    path = copy / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["em_executions"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = b1r_auditor.audit(copy)
    assert report["verdict"] == "FAIL"
    assert any("zero-EM" in f["message"] for f in report["findings"])


# --------------------------------------------------------------------------
# 4. the preserved directories stay preserved
# --------------------------------------------------------------------------

def test_b1r_does_not_read_or_write_the_preserved_directories():
    for module in ("validate_candidate_optimizer_analytic_jac.py",
                   "audit_candidate_optimizer_analytic_jac.py"):
        source = (_HERE / module).read_text(encoding="utf-8")
        assert "smoke_20260923" not in source
        assert "optimizer_validation_20260923" not in source


def test_the_b1_auditor_was_not_changed_for_b1r():
    """B1R gets its own auditor so B1's recorded report stays as it was."""

    source = (_HERE / "audit_candidate_optimizer_validation.py").read_text(
        encoding="utf-8")
    assert "74-B1R" not in source
    assert "REFERENCE_V2" not in source
    assert "analytic_production_gradient" not in source


def test_poisson_base_measure_is_still_in_the_shared_objective(Z_samples):
    """The regression guard the B1R reference relies on, restated here."""

    counted = np.arange(24, dtype=np.float64) % 4
    loading = np.array([0.05, -0.02, 0.01])
    with_base = b1r.independent_objective(counted, Z_samples, loading,
                                          "poisson")
    eta = np.column_stack([Z_samples[:, :, s] @ loading for s in range(5)])
    without_base = float(np.sum(counted[:, None] * eta - np.exp(eta))) / 5
    expected_base = -sum(math.lgamma(v + 1.0) for v in counted)
    assert with_base == pytest.approx(without_base + expected_base)
