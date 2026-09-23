"""Zero-EM tests for the Gate 74-B1 validator and its independent auditor.

No EM is executed here, and no smoke artifact is read.  A module-scoped
autouse fixture turns an accidental MCEM call into a loud failure.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import audit_candidate_optimizer_validation as b1_auditor    # noqa: E402
import validate_candidate_optimizer as validator             # noqa: E402
from family_selection import (                               # noqa: E402
    ADAM_LR,
    ADAM_MAX_ITER,
    ADAM_TOL,
    optimise_column_loading,
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


@pytest.fixture(scope="module")
def Z_samples():
    return validator.build_z_samples()


# --------------------------------------------------------------------------
# 1. the frozen design
# --------------------------------------------------------------------------

def test_the_design_matches_the_issue_specification(Z_samples):
    assert validator.BASE_VECTORS[0] == (1.0, 0.2, -0.1)
    assert validator.BASE_VECTORS[-1] == (-0.5, 0.4, -0.8)
    assert len(validator.BASE_VECTORS) == 6
    assert validator.GROUP_REPEATS == 4
    assert validator.SAMPLE_SCALES == (0.96, 0.98, 1.00, 1.02, 1.04)
    assert Z_samples.shape == (24, 3, 5)
    assert np.all(np.isfinite(Z_samples))


def test_samples_are_the_base_design_times_each_scale(Z_samples):
    base = np.repeat(np.asarray(validator.BASE_VECTORS), 4, axis=0)
    for index, scale in enumerate(validator.SAMPLE_SCALES):
        assert np.allclose(Z_samples[:, :, index], base * scale)


def test_the_design_uses_no_rng(Z_samples):
    """Rebuilding twice must give bitwise identical arrays."""

    again = validator.build_z_samples()
    assert np.array_equal(Z_samples, again)
    source = (_HERE / "validate_candidate_optimizer.py").read_text(
        encoding="utf-8")
    assert "default_rng" not in source
    assert "np.random" not in source


def test_case_columns_have_the_specified_counts_and_are_not_separable():
    case_a = validator.build_case_column("case_A_balanced_nonseparable")
    case_b = validator.build_case_column("case_B_sparse_nonseparable")
    assert case_a.shape == (24,) and case_b.shape == (24,)
    assert int(case_a.sum()) == 12
    assert int(case_b.sum()) == 8
    # Every exact design group carries both labels, so no linear rule in Z can
    # separate the column perfectly.
    for column in (case_a, case_b):
        for start in range(0, 24, 4):
            group = set(column[start:start + 4].tolist())
            assert group == {0.0, 1.0}


def test_the_three_initialisations_are_the_frozen_ones():
    assert validator.INITIALISATIONS == {
        "init_zero": (0.0, 0.0, 0.0),
        "init_pos": (0.6, -0.4, 0.3),
        "init_neg": (-0.6, 0.4, -0.3),
    }
    assert validator.FAMILIES == ("bernoulli", "poisson")


def test_no_smoke_artifact_is_referenced():
    source = (_HERE / "validate_candidate_optimizer.py").read_text(
        encoding="utf-8")
    assert "smoke_20260923" not in source
    assert "results/family_selection" not in source


# --------------------------------------------------------------------------
# 2. the independent objective really is independent, and correct
# --------------------------------------------------------------------------

def test_the_independent_objective_does_not_call_the_production_scorer():
    import inspect

    source = inspect.getsource(validator.independent_objective)
    # Everything after the docstring closes. The docstring names those helpers
    # precisely to say it does not use them, so a check over the whole source
    # would trip over its own explanation.
    body = source.split('"""')[2]
    assert "column_log_likelihood" not in body
    assert "logaddexp" not in body
    assert "gammaln" not in body
    assert "log1p" in body and "lgamma" in body


def test_independent_and_production_scores_agree(Z_samples):
    for case in validator.CASES:
        x = validator.build_case_column(case)
        for family in validator.FAMILIES:
            for init in validator.INITIALISATIONS.values():
                loading = np.asarray(init, dtype=np.float64)
                from family_selection import column_log_likelihood

                production = column_log_likelihood(x, Z_samples, loading,
                                                   family)
                independent = validator.independent_objective(
                    x, Z_samples, loading, family)
                assert abs(production - independent) <= \
                    validator.OBJECTIVE_MATCH_TOL


def test_poisson_objective_carries_its_base_measure(Z_samples):
    """On a 0/1 column the term is zero, so check it on a counted column."""

    counted = np.arange(24, dtype=np.float64) % 4          # values 0..3
    loading = np.array([0.05, -0.02, 0.01])
    with_base = validator.independent_objective(counted, Z_samples, loading,
                                                "poisson")
    eta = np.column_stack([Z_samples[:, :, s] @ loading for s in range(5)])
    without_base = float(np.sum(counted[:, None] * eta - np.exp(eta))) / 5
    expected_base = -sum(math.lgamma(v + 1.0) for v in counted)
    assert expected_base < 0.0
    assert with_base == pytest.approx(without_base + expected_base)


def test_analytic_gradient_matches_the_finite_difference(Z_samples):
    from family_selection import _column_gradient

    for case in validator.CASES:
        x = validator.build_case_column(case)
        for family in validator.FAMILIES:
            for init in validator.INITIALISATIONS.values():
                loading = np.asarray(init, dtype=np.float64)
                analytic = _column_gradient(x, Z_samples, loading, family, None)
                numeric = validator.finite_difference_gradient(
                    x, Z_samples, loading, family)
                assert np.max(np.abs(analytic - numeric)) <= \
                    validator.GRADIENT_MATCH_TOL


# --------------------------------------------------------------------------
# 3. the diagnostics channel does not change the optimiser
# --------------------------------------------------------------------------

def test_diagnostics_do_not_change_a_single_returned_bit(Z_samples):
    """The recording channel must be a recording channel and nothing more."""

    x = validator.build_case_column("case_A_balanced_nonseparable")
    for family in validator.FAMILIES:
        for init in validator.INITIALISATIONS.values():
            loading_init = np.asarray(init, dtype=np.float64)
            plain = optimise_column_loading(x, Z_samples, family,
                                            loading_init=loading_init)
            diagnostics: dict = {}
            instrumented = optimise_column_loading(
                x, Z_samples, family, loading_init=loading_init,
                diagnostics=diagnostics)
            assert np.array_equal(plain[0], instrumented[0])
            assert plain[1] == instrumented[1]
            assert plain[2] == instrumented[2]
            assert plain[3] == instrumented[3]
            assert diagnostics["n_iter"] == plain[2]
            assert diagnostics["converged"] == plain[3]
            assert len(diagnostics["steps"]) == plain[2]
            assert math.isfinite(diagnostics["final_gradient_inf"])


def test_production_adam_settings_are_untouched():
    assert (ADAM_MAX_ITER, ADAM_LR, ADAM_TOL) == (50, 0.01, 1e-6)


# --------------------------------------------------------------------------
# 4. the auditor is genuinely independent
# --------------------------------------------------------------------------

def test_the_b1_auditor_imports_neither_validator_nor_selector():
    source = (_HERE / "audit_candidate_optimizer_validation.py").read_text(
        encoding="utf-8")
    assert "validate_candidate_optimizer" not in source
    assert "import family_selection" not in source
    assert "from family_selection" not in source
    assert "import numpy" not in source          # it rebuilds without NumPy


def test_the_auditor_rebuilds_the_frozen_arrays_independently(Z_samples):
    assert b1_auditor.rebuild_z_samples_digest() == \
        validator.array_digest(Z_samples)
    for case in validator.CASES:
        column = validator.build_case_column(case)
        assert b1_auditor.rebuild_case_digest(case) == \
            validator.array_digest(column)
        assert b1_auditor.rebuild_case_column(case) == column.tolist()


def test_the_auditor_transcribed_the_same_thresholds():
    assert b1_auditor.EXPECTED_THRESHOLDS["adam_gap_per_obs_tol"] == \
        validator.ADAM_GAP_PER_OBS_TOL
    assert b1_auditor.EXPECTED_THRESHOLDS["adam_grad_inf_tol"] == \
        validator.ADAM_GRAD_INF_TOL
    assert b1_auditor.EXPECTED_ADAM["max_iter"] == validator.ADAM_MAX_ITER
    assert b1_auditor.EXPECTED_REFERENCE_SOLVER["gtol"] == validator.BFGS_GTOL


# --------------------------------------------------------------------------
# 5. the pipeline, run for real (still zero EM) on a temporary directory
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def validation_run(tmp_path_factory):
    """One real B1 execution into a throwaway directory.

    B1 runs no EM, so exercising it in a test costs nothing and is not the
    recorded run: that one goes to its own immutable directory.
    """

    out = tmp_path_factory.mktemp("b1") / "run"
    summary = validator.execute(out)
    return out, summary


def test_execute_writes_every_artifact(validation_run):
    out, summary = validation_run
    for name in validator.ARTIFACT_NAMES:
        assert (out / name).is_file(), name
    assert summary["em_executions"] == 0
    assert summary["status"] in (validator.STATUS_VALID,
                                 validator.STATUS_REFERENCE_NOT_CERTIFIED,
                                 validator.STATUS_FAIL)
    # The Adam comparison exists only against a certified reference.
    certified = summary["status"] == validator.STATUS_VALID
    for name in validator.CERTIFIED_ARTIFACT_NAMES:
        assert (out / name).is_file() is certified, name


def test_an_uncertified_reference_produces_no_adam_comparison(validation_run):
    """Comparing against an uncertified optimum would not mean anything."""

    out, summary = validation_run
    if summary["status"] == validator.STATUS_VALID:
        pytest.skip("the reference was certified in this environment")
    assert summary["status"] == validator.STATUS_REFERENCE_NOT_CERTIFIED
    assert summary["adam_summary"]["runs"] == 0
    assert summary["adam_verdict"] == validator.ADAM_NEEDS_HUMAN
    assert not (out / "adam_comparison.csv").exists()
    assert summary["reference_certification"]["failures"]


def test_execute_refuses_to_overwrite(validation_run):
    out, _ = validation_run
    with pytest.raises(validator.ValidationStop, match="never overwritten"):
        validator.execute(out)


def test_twelve_adam_runs_when_the_reference_is_certified(validation_run):
    out, summary = validation_run
    if summary["status"] != validator.STATUS_VALID:
        pytest.skip("reference was not certified in this environment")
    with (out / "adam_comparison.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    for row in rows:
        assert int(row["adam_max_iter"]) == ADAM_MAX_ITER
        assert float(row["adam_lr"]) == ADAM_LR
        assert float(row["adam_tol"]) == ADAM_TOL
        assert math.isfinite(float(row["objective_gap_per_observation"]))


def test_the_auditor_passes_the_validation_run(validation_run):
    out, summary = validation_run
    report = b1_auditor.audit(out)
    assert report["verdict"] == "PASS", report["findings"]
    assert report["recomputed_status"] == summary["status"]
    assert report["recomputed_adam_verdict"] == summary["adam_verdict"]
    assert (out / "audit_report.json").is_file()


def test_the_auditor_catches_a_tampered_threshold(tmp_path, validation_run):
    import shutil

    out, _ = validation_run
    copy = tmp_path / "tampered"
    shutil.copytree(out, copy)
    path = copy / "protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["thresholds"]["adam_gap_per_obs_tol"] = 1.0    # a flattering edit
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = b1_auditor.audit(copy)
    assert report["verdict"] == "FAIL"
    assert any("adam_gap_per_obs_tol" in f["message"]
               for f in report["findings"])


def test_the_auditor_catches_a_changed_adam_budget(tmp_path, validation_run):
    import shutil

    out, summary = validation_run
    if summary["status"] != validator.STATUS_VALID:
        pytest.skip("reference was not certified in this environment")
    copy = tmp_path / "budget"
    shutil.copytree(out, copy)
    path = copy / "adam_comparison.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    for row in rows:
        row["adam_max_iter"] = "500"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = b1_auditor.audit(copy)
    assert report["verdict"] == "FAIL"
    assert any("adam_max_iter" in f["message"] for f in report["findings"])


def test_the_auditor_catches_a_case_definition_edit(tmp_path, validation_run):
    import shutil

    out, _ = validation_run
    copy = tmp_path / "cases"
    shutil.copytree(out, copy)
    path = copy / "protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"]["case_B_sparse_nonseparable"][0] = [1, 1, 1, 1]
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = b1_auditor.audit(copy)
    assert report["verdict"] == "FAIL"
    assert any("case_B_sparse_nonseparable" in f["message"]
               for f in report["findings"])


def test_the_auditor_catches_a_flattering_summary(tmp_path, validation_run):
    import shutil

    out, summary = validation_run
    if summary["adam_verdict"] == validator.ADAM_CERTIFIED:
        pytest.skip("nothing to flatter: Adam was certified")
    copy = tmp_path / "summary"
    shutil.copytree(out, copy)
    path = copy / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["adam_verdict"] = validator.ADAM_CERTIFIED
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = b1_auditor.audit(copy)
    assert report["verdict"] == "FAIL"
    assert any("adam_verdict" in f["message"] for f in report["findings"])


def test_the_auditor_catches_a_claimed_em_execution(tmp_path, validation_run):
    import shutil

    out, _ = validation_run
    copy = tmp_path / "em"
    shutil.copytree(out, copy)
    path = copy / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["em_executions"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = b1_auditor.audit(copy)
    assert report["verdict"] == "FAIL"
    assert any("zero-EM" in f["message"] for f in report["findings"])
