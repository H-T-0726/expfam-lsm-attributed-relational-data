"""Zero-EM tests for the Gate 74-B2 candidate-optimizer migration.

No EM is executed, no RNG is used, and no preserved artifact directory is
written to.  A module-scoped autouse fixture turns an accidental MCEM call
into a loud failure.
"""

from __future__ import annotations

import csv
import inspect
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import audit_optimizer_migration as b2_auditor                 # noqa: E402
import family_selection as fs                                  # noqa: E402
import validate_candidate_optimizer as b1                      # noqa: E402
import validate_optimizer_migration as b2                      # noqa: E402


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
    return b1.build_z_samples()


@pytest.fixture(scope="module")
def binary_column():
    return b1.build_case_column("case_A_balanced_nonseparable")


# --------------------------------------------------------------------------
# 1. BFGS objective and Jacobian signs
# --------------------------------------------------------------------------

def test_the_bfgs_route_minimises_the_negative_score_with_a_negated_jacobian(
        Z_samples, binary_column):
    captured = {}

    def fake_minimize(fun, x0, **kwargs):
        probe = np.array([0.31, -0.22, 0.14])
        captured["objective"] = fun(probe)
        captured["jac"] = kwargs["jac"](probe)
        captured["method"] = kwargs["method"]
        captured["options"] = kwargs["options"]

        class _R:
            x = np.asarray(x0, dtype=float)
            success, status, message, nit, njev = True, 0, "ok", 3, 3

        return _R()

    original = fs.minimize
    fs.minimize = fake_minimize
    try:
        fs.optimise_column_loading_bfgs(binary_column, Z_samples, "bernoulli",
                                        loading_init=np.zeros(3))
    finally:
        fs.minimize = original

    probe = np.array([0.31, -0.22, 0.14])
    expected_score = fs.column_log_likelihood(binary_column, Z_samples, probe,
                                              "bernoulli")
    expected_grad = fs._column_gradient(binary_column, Z_samples, probe,
                                        "bernoulli", None)
    assert captured["objective"] == pytest.approx(-expected_score)
    assert np.allclose(captured["jac"], -expected_grad)
    assert captured["method"] == "BFGS"
    assert captured["options"] == {"maxiter": 2000, "gtol": 1e-10}


# --------------------------------------------------------------------------
# 2. the BFGS path returns candidate-specific loadings and strict scores
# --------------------------------------------------------------------------

def test_each_candidate_gets_its_own_loading_and_strict_score(Z_samples,
                                                              binary_column):
    gate, = fs.support_gate(binary_column[:, None])
    records = fs.score_column_candidates(
        binary_column, Z_samples, gate, loading_init=np.zeros(3),
        optimizer=fs.OPTIMIZER_BFGS)
    assert [r.family for r in records] == ["bernoulli", "poisson"]
    bernoulli, poisson = records
    assert not np.allclose(bernoulli.loading, poisson.loading)
    for record in records:
        # The reported score is the production strict score at that loading.
        assert record.score == pytest.approx(fs.column_log_likelihood(
            binary_column, Z_samples, record.loading, record.family))


# --------------------------------------------------------------------------
# 3 and 4. convergence uses the gradient norm, not SciPy success
# --------------------------------------------------------------------------

def test_convergence_is_decided_on_the_gradient_not_on_scipy_success(
        Z_samples, binary_column):
    """A successful solve at a non-stationary point must not count as converged."""

    def fake_minimize(fun, x0, **kwargs):
        class _R:
            # Far from the optimum, but the solver says it is happy.
            x = np.array([2.5, -2.0, 1.5])
            success, status, message, nit, njev = True, 0, "success", 1, 1

        return _R()

    original = fs.minimize
    fs.minimize = fake_minimize
    try:
        loading, _, _, converged, provenance = fs.optimise_column_loading_bfgs(
            binary_column, Z_samples, "bernoulli", loading_init=np.zeros(3))
    finally:
        fs.minimize = original

    assert provenance["scipy_success"] is True
    assert provenance["gradient_inf"] > fs.CANDIDATE_GRAD_INF_TOL
    assert converged is False
    assert np.allclose(loading, [2.5, -2.0, 1.5])


def test_a_real_solve_reaches_the_gradient_criterion(Z_samples, binary_column):
    for family in ("bernoulli", "poisson"):
        loading, _, _, converged, provenance = fs.optimise_column_loading_bfgs(
            binary_column, Z_samples, family, loading_init=np.zeros(3))
        assert np.all(np.isfinite(loading))
        assert provenance["gradient_inf"] <= fs.CANDIDATE_GRAD_INF_TOL
        assert converged is True
        assert provenance["convergence_grad_inf_tol"] == 1e-8


# --------------------------------------------------------------------------
# 5. no retry, fallback or reseed
# --------------------------------------------------------------------------

def test_the_bfgs_route_has_no_retry_fallback_or_reseed_path():
    source = inspect.getsource(fs.optimise_column_loading_bfgs)
    body = source.split('"""')[2]
    for forbidden in ("Nelder-Mead", "Powell", "L-BFGS-B", "TNC",
                      "trust-constr", "default_rng", "for attempt",
                      "for retry", "while True"):
        assert forbidden not in body, forbidden
    # minimize is called exactly once.
    assert body.count("minimize(") == 1


def test_provenance_records_no_fallback_and_no_retries(Z_samples,
                                                       binary_column):
    _, _, _, _, provenance = fs.optimise_column_loading_bfgs(
        binary_column, Z_samples, "poisson", loading_init=np.zeros(3))
    assert provenance["fallback_solvers"] == []
    assert provenance["retries"] == 0


# --------------------------------------------------------------------------
# 6. the historical Adam route is preserved and numerically unchanged
# --------------------------------------------------------------------------

PINNED_ADAM_INPUT = {
    "x": [0.0, 1.0] * 12,
    "loading_init": [0.05, -0.02, 0.01],
}


def test_the_adam_route_is_still_callable_and_unchanged(Z_samples):
    """Pinned deterministic inputs: the smoke and B1/B1R must stay reproducible."""

    x = np.asarray(PINNED_ADAM_INPUT["x"], dtype=np.float64)
    init = np.asarray(PINNED_ADAM_INPUT["loading_init"], dtype=np.float64)

    direct = fs.optimise_column_loading(x, Z_samples, "bernoulli",
                                        loading_init=init)
    routed = fs.optimise_candidate_loading(x, Z_samples, "bernoulli",
                                           loading_init=init,
                                           optimizer=fs.OPTIMIZER_ADAM)
    # Routing through the dispatcher changes nothing about the Adam result.
    assert np.array_equal(direct[0], routed[0])
    assert direct[1] == routed[1]
    assert direct[2] == routed[2]
    assert direct[3] == routed[3]
    # Adam still stops at its own budget under its own step rule.
    assert direct[2] == fs.ADAM_MAX_ITER
    assert direct[3] is False
    assert routed[4]["convergence_rule"] == "step infinity norm below tol"
    assert routed[4]["max_iter"] == 50 and routed[4]["lr"] == 0.01


def test_the_b1_and_b1r_validators_still_use_the_adam_route():
    """Their diagnostics describe Adam; they must not silently become BFGS."""

    for module in ("validate_candidate_optimizer.py",
                   "validate_candidate_optimizer_analytic_jac.py"):
        source = (_HERE / module).read_text(encoding="utf-8")
        assert "optimise_column_loading" in source
        assert "PHASE9C_CANDIDATE_OPTIMIZER" not in source
        assert "optimise_candidate_loading" not in source


# --------------------------------------------------------------------------
# 7. Phase 9C selector provenance
# --------------------------------------------------------------------------

def test_the_selector_chooses_bfgs_by_explicit_configuration(Z_samples):
    from family_selection import FamilySelectingPerColumnLSM

    x = np.column_stack([b1.build_case_column("case_A_balanced_nonseparable")])
    model = FamilySelectingPerColumnLSM(
        gates=fs.support_gate(x), n=24, d=1, k=3, L=5,
        family_x_list=["bernoulli"], family_y="bernoulli")
    assert model.candidate_optimizer == fs.PHASE9C_CANDIDATE_OPTIMIZER
    assert model.candidate_optimizer == fs.OPTIMIZER_BFGS
    # The legacy route is still selectable by name.
    legacy = FamilySelectingPerColumnLSM(
        gates=fs.support_gate(x), n=24, d=1, k=3, L=5,
        family_x_list=["bernoulli"], family_y="bernoulli",
        candidate_optimizer=fs.OPTIMIZER_ADAM)
    assert legacy.candidate_optimizer == fs.OPTIMIZER_ADAM
    with pytest.raises(fs.SelectorStop, match="unknown candidate_optimizer"):
        FamilySelectingPerColumnLSM(
            gates=fs.support_gate(x), n=24, d=1, k=3, L=5,
            family_x_list=["bernoulli"], family_y="bernoulli",
            candidate_optimizer="newton")


def test_the_selection_trace_records_the_optimizer(Z_samples):
    from family_selection import FamilySelectingPerColumnLSM

    x = np.column_stack([b1.build_case_column("case_A_balanced_nonseparable")])
    model = FamilySelectingPerColumnLSM(
        gates=fs.support_gate(x), n=24, d=1, k=3, L=5,
        family_x_list=["bernoulli"], family_y="bernoulli")
    model.initialize_params(true_params=None, seed=7)
    model.select_families(x, Z_samples)
    row, = model.selection_trace
    assert row["candidate_optimizer"] == fs.OPTIMIZER_BFGS
    for record in model.last_candidate_records:
        assert record.as_row()["optimizer"] == fs.OPTIMIZER_BFGS
        assert record.as_row()["optimiser_gradient_inf"] <= 1e-8


def test_exploration_metadata_records_the_frozen_settings():
    source = inspect.getsource(fs.run_family_exploration)
    assert "candidate_optimizer" in source
    assert "candidate_optimizer_settings" in source
    assert "historical_adam_preserved" in source


# --------------------------------------------------------------------------
# 8. the gate consumes the new convergence flag
# --------------------------------------------------------------------------

def test_the_pilot_gate_consumes_the_new_candidate_convergence_flag(
        Z_samples, binary_column):
    gate, = fs.support_gate(binary_column[:, None])
    bfgs = fs.score_column_candidates(binary_column, Z_samples, gate,
                                      loading_init=np.zeros(3),
                                      optimizer=fs.OPTIMIZER_BFGS)
    adam = fs.score_column_candidates(binary_column, Z_samples, gate,
                                      loading_init=np.zeros(3),
                                      optimizer=fs.OPTIMIZER_ADAM)

    bfgs_gate = fs.pilot_convergence_gate(
        [{"column": 0, "iteration": 1,
          "all_candidates_converged": all(r.converged for r in bfgs)}],
        [r.as_row() for r in bfgs])
    adam_gate = fs.pilot_convergence_gate(
        [{"column": 0, "iteration": 1,
          "all_candidates_converged": all(r.converged for r in adam)}],
        [r.as_row() for r in adam])

    assert bfgs_gate["status"] == fs.PILOT_GATE_PASS
    assert adam_gate["status"] == fs.PILOT_GATE_BLOCKED
    assert adam_gate["non_converged_candidate_count"] == len(adam)


# --------------------------------------------------------------------------
# 9. the migration run itself (zero EM)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reference_copy(tmp_path_factory):
    """A COPY of the certified B1R run: the recorded one is never written to."""

    source = b2.DEFAULT_REFERENCE_DIR
    if not source.is_dir():
        pytest.skip("the certified Gate 74-B1R run is not present")
    destination = tmp_path_factory.mktemp("b1r_ref") / "run"
    shutil.copytree(source, destination)
    return destination


@pytest.fixture(scope="module")
def migration_run(tmp_path_factory, reference_copy):
    out = tmp_path_factory.mktemp("b2") / "run"
    summary = b2.execute(out, reference_copy)
    return out, summary


def test_the_migration_run_writes_every_artifact(migration_run):
    out, summary = migration_run
    for name in b2.ARTIFACT_NAMES:
        assert (out / name).is_file(), name
    assert summary["em_executions"] == 0
    assert summary["gate"] == "74-B2"


def test_the_migration_run_refuses_to_overwrite(migration_run, reference_copy):
    out, _ = migration_run
    with pytest.raises(b2.MigrationStop, match="never overwritten"):
        b2.execute(out, reference_copy)


def test_the_reference_must_be_a_certified_b1r_run(tmp_path, reference_copy):
    broken = tmp_path / "broken"
    shutil.copytree(reference_copy, broken)
    path = broken / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "REFERENCE_V2_NOT_CERTIFIED"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(b2.MigrationStop, match="not certified"):
        b2.execute(tmp_path / "out", broken)


def test_the_twelve_comparisons_use_the_frozen_settings(migration_run):
    out, _ = migration_run
    with (out / "migration_comparison.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    for row in rows:
        assert row["optimizer"] == "bfgs"
        assert row["bfgs_method"] == "BFGS"
        assert int(row["bfgs_maxiter"]) == 2000
        assert float(row["bfgs_gtol"]) == 1e-10
        assert float(row["convergence_grad_inf_tol"]) == 1e-8
        assert row["fallback_solvers"] == ""
        assert int(row["retries"]) == 0


def test_the_auditor_passes_the_migration_run(migration_run):
    out, summary = migration_run
    report = b2_auditor.audit(out)
    assert report["verdict"] == "PASS", report["findings"]
    assert report["recomputed_status"] == summary["status"]
    assert (out / "audit_report.json").is_file()


def test_the_auditor_imports_neither_validator_nor_selector():
    source = (_HERE / "audit_optimizer_migration.py").read_text(
        encoding="utf-8")
    assert "validate_optimizer_migration" not in source
    assert "import family_selection" not in source
    assert "from family_selection" not in source
    assert "import numpy" not in source


def test_the_auditor_rebuilds_the_frozen_arrays_independently(Z_samples):
    assert b2_auditor.rebuild_z_samples_digest() == b1.array_digest(Z_samples)
    for case in b1.CASES:
        column = b1.build_case_column(case)
        assert b2_auditor.rebuild_case_digest(case) == b1.array_digest(column)


@pytest.mark.parametrize("field,value", [
    ("phase9c_candidate_optimizer", "adam"),
    ("historical_adam_preserved", False),
    ("production_change_outside_phase9c_selector", True),
])
def test_the_auditor_catches_a_tampered_protocol(tmp_path, migration_run,
                                                 field, value):
    out, _ = migration_run
    copy = tmp_path / f"tampered_{field}"
    shutil.copytree(out, copy)
    path = copy / "protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = b2_auditor.audit(copy)
    assert report["verdict"] == "FAIL"


def test_the_auditor_catches_a_changed_bfgs_setting(tmp_path, migration_run):
    out, _ = migration_run
    copy = tmp_path / "gtol"
    shutil.copytree(out, copy)
    path = copy / "protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["candidate_optimizer_settings"]["gtol"] = 1e-6
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = b2_auditor.audit(copy)
    assert report["verdict"] == "FAIL"
    assert any("gtol" in f["message"] for f in report["findings"])


def test_the_auditor_catches_a_pass_the_rows_do_not_support(tmp_path,
                                                            migration_run):
    out, _ = migration_run
    copy = tmp_path / "claim"
    shutil.copytree(out, copy)
    path = copy / "migration_comparison.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["gradient_inf"] = "1.0"          # far above 1e-8, still claims pass
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = b2_auditor.audit(copy)
    assert report["verdict"] == "FAIL"
    assert any("imply" in f["message"] for f in report["findings"])


def test_the_auditor_catches_a_claimed_em_execution(tmp_path, migration_run):
    out, _ = migration_run
    copy = tmp_path / "em"
    shutil.copytree(out, copy)
    path = copy / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["em_executions"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = b2_auditor.audit(copy)
    assert report["verdict"] == "FAIL"
    assert any("zero-EM" in f["message"] for f in report["findings"])


def test_b2_never_writes_to_the_preserved_directories():
    for module in ("validate_optimizer_migration.py",
                   "audit_optimizer_migration.py"):
        source = (_HERE / module).read_text(encoding="utf-8")
        assert "smoke_20260923" not in source
        assert "optimizer_validation_20260923" not in source
