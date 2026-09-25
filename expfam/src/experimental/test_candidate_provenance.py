"""Zero-EM tests for the Gate 74-B4 candidate provenance hardening.

B4 adds per-candidate diagnostics to selection-trace rows and changes nothing
else.  These tests hold that line: the numbers must be identical with and
without the new fields, no optimiser may run an extra time for them, the
frozen Phase 9C settings must be untouched, and the strict convergence gate
must keep meaning exactly what it meant in Gate 74-B3.

No EM is executed and no preserved artifact directory is read or written.
The provenance is pinned on the deterministic fixed-Z inputs from Gate 74-B1
rather than on mocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import family_selection as fs                                  # noqa: E402
import validate_candidate_optimizer as b1                      # noqa: E402
from family_selection import (                                 # noqa: E402
    FamilySelectingPerColumnLSM,
    PILOT_GATE_BLOCKED,
    PILOT_GATE_PASS,
    pilot_convergence_gate,
    support_gate,
)

CANDIDATE_PROVENANCE_SUFFIXES = ("optimizer", "n_iter", "converged",
                                 "grad_inf", "scipy_success", "scipy_status")


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
    """The deterministic Gate 74-B1 posterior-sample block. No RNG, no EM."""

    return b1.build_z_samples()


@pytest.fixture
def ambiguous_model(Z_samples):
    """A one-column model whose single column is gate-ambiguous."""

    x = np.column_stack([b1.build_case_column("case_A_balanced_nonseparable")])
    model = FamilySelectingPerColumnLSM(
        gates=support_gate(x), n=24, d=1, k=3, L=5,
        family_x_list=["bernoulli"], family_y="bernoulli")
    model.initialize_params(true_params=None, seed=7)
    return model, x


# --------------------------------------------------------------------------
# 1. the fields reflect the underlying records
# --------------------------------------------------------------------------

def test_every_candidate_gets_every_provenance_field(ambiguous_model,
                                                     Z_samples):
    model, x = ambiguous_model
    model.select_families(x, Z_samples)
    row, = model.selection_trace

    for family in ("bernoulli", "poisson"):
        for suffix in CANDIDATE_PROVENANCE_SUFFIXES:
            assert f"{family}_{suffix}" in row, f"{family}_{suffix}"


def test_the_fields_are_exactly_the_candidate_records(ambiguous_model,
                                                      Z_samples):
    model, x = ambiguous_model
    model.select_families(x, Z_samples)
    row, = model.selection_trace
    records = {record.family: record for record in model.last_candidate_records}

    for family, record in records.items():
        assert row[f"{family}_optimizer"] == record.optimizer
        assert row[f"{family}_n_iter"] == record.n_iter
        assert row[f"{family}_converged"] == record.converged
        assert row[f"{family}_grad_inf"] == record.gradient_inf
        assert row[f"{family}_scipy_success"] == \
            record.provenance["scipy_success"]
        assert row[f"{family}_scipy_status"] == \
            record.provenance["scipy_status"]
        # And the score column that was already there still agrees.
        assert row[f"score_{family}"] == record.score


def test_the_provenance_helper_reads_records_without_optimising():
    """It is a projection of records, not a computation over them."""

    import inspect

    source = inspect.getsource(fs._candidate_provenance_fields)
    body = source.split('"""')[2]
    for forbidden in ("minimize(", "optimise_", "column_log_likelihood",
                      "_column_gradient"):
        assert forbidden not in body, forbidden


def test_scipy_success_and_convergence_can_disagree(ambiguous_model,
                                                    Z_samples):
    """Why the rule is the gradient: the solver's own flag is not the verdict.

    On this frozen column the Poisson candidate reaches a gradient well inside
    the criterion while SciPy reports precision loss. Recording both is the
    point of the hardening.
    """

    model, x = ambiguous_model
    model.select_families(x, Z_samples)
    row, = model.selection_trace

    assert row["poisson_grad_inf"] <= fs.CANDIDATE_GRAD_INF_TOL
    assert row["poisson_converged"] is True
    assert row["poisson_scipy_success"] is False
    assert row["poisson_scipy_status"] == 2


# --------------------------------------------------------------------------
# 2 and 3. capture changes no number and adds no optimiser call
# --------------------------------------------------------------------------

def test_the_selection_is_numerically_unchanged_by_capture(ambiguous_model,
                                                           Z_samples):
    """Compare against the records, which the trace row cannot influence."""

    model, x = ambiguous_model
    assignment = model.select_families(x, Z_samples)
    row, = model.selection_trace
    records = {record.family: record for record in model.last_candidate_records}

    chosen, margin = fs.select_from_records(list(records.values()))
    assert assignment[0] == chosen
    assert row["selected_family"] == chosen
    assert row["margin_neg2"] == margin
    for family, record in records.items():
        assert row[f"score_{family}"] == record.score
        assert np.all(np.isfinite(record.loading))


def test_capture_adds_no_optimiser_invocation(monkeypatch, ambiguous_model,
                                              Z_samples):
    """Two ambiguous candidates, two BFGS solves. Provenance adds none."""

    model, x = ambiguous_model
    calls = {"n": 0}
    original = fs.minimize

    def counting_minimize(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fs, "minimize", counting_minimize)
    model.select_families(x, Z_samples)
    assert calls["n"] == 2

    row, = model.selection_trace
    assert row["bernoulli_grad_inf"] is not None
    assert row["poisson_grad_inf"] is not None


def test_scoring_twice_gives_identical_rows(ambiguous_model, Z_samples):
    model, x = ambiguous_model
    model.select_families(x, Z_samples)
    first = dict(model.selection_trace[-1])
    model.select_families(x, Z_samples)
    second = dict(model.selection_trace[-1])
    for key in first:
        assert first[key] == second[key], key


# --------------------------------------------------------------------------
# 4 and 5. the historical route and the frozen settings
# --------------------------------------------------------------------------

def test_the_historical_adam_route_is_still_callable_and_unchanged(Z_samples):
    x = b1.build_case_column("case_A_balanced_nonseparable")
    init = np.array([0.05, -0.02, 0.01])
    direct = fs.optimise_column_loading(x, Z_samples, "bernoulli",
                                        loading_init=init)
    routed = fs.optimise_candidate_loading(x, Z_samples, "bernoulli",
                                           loading_init=init,
                                           optimizer=fs.OPTIMIZER_ADAM)
    assert np.array_equal(direct[0], routed[0])
    assert direct[2] == routed[2] == fs.ADAM_MAX_ITER
    assert direct[3] is routed[3] is False
    assert routed[4]["convergence_rule"] == "step infinity norm below tol"


def test_the_phase9c_settings_are_exactly_frozen():
    assert fs.PHASE9C_CANDIDATE_OPTIMIZER == fs.OPTIMIZER_BFGS
    assert fs.BFGS_METHOD == "BFGS"
    assert fs.BFGS_MAXITER == 2000
    assert fs.BFGS_GTOL == 1e-10
    assert fs.CANDIDATE_GRAD_INF_TOL == 1e-8
    assert (fs.ADAM_MAX_ITER, fs.ADAM_LR, fs.ADAM_TOL) == (50, 0.01, 1e-6)


def test_the_adam_route_provenance_also_appears_in_the_trace(Z_samples):
    """The legacy route must stay inspectable, with its own rule recorded."""

    x = np.column_stack([b1.build_case_column("case_B_sparse_nonseparable")])
    model = FamilySelectingPerColumnLSM(
        gates=support_gate(x), n=24, d=1, k=3, L=5,
        family_x_list=["bernoulli"], family_y="bernoulli",
        candidate_optimizer=fs.OPTIMIZER_ADAM)
    model.initialize_params(true_params=None, seed=7)
    model.select_families(x, Z_samples)
    row, = model.selection_trace

    assert row["candidate_optimizer"] == fs.OPTIMIZER_ADAM
    assert row["bernoulli_optimizer"] == fs.OPTIMIZER_ADAM
    assert row["bernoulli_n_iter"] == fs.ADAM_MAX_ITER
    assert row["bernoulli_converged"] is False
    # Adam has no SciPy result, and the field says so rather than inventing one.
    assert row["bernoulli_scipy_success"] == ""
    assert row["bernoulli_scipy_status"] == ""


# --------------------------------------------------------------------------
# 6, 7 and 8. the gate is unchanged
# --------------------------------------------------------------------------

def test_the_aggregate_flag_still_means_all_candidates_converged(
        ambiguous_model, Z_samples):
    model, x = ambiguous_model
    model.select_families(x, Z_samples)
    row, = model.selection_trace
    expected = all(record.converged
                   for record in model.last_candidate_records)
    assert row["all_candidates_converged"] is expected
    assert row["all_candidates_converged"] == (row["bernoulli_converged"]
                                               and row["poisson_converged"])


def test_the_gate_still_blocks_on_any_false_intermediate_row():
    trace = [
        {"column": 2, "iteration": 1, "all_candidates_converged": True},
        {"column": 2, "iteration": 2, "all_candidates_converged": False,
         "selected_family": "bernoulli", "margin_neg2": 24.0,
         "bernoulli_grad_inf": 1e-12, "poisson_grad_inf": 3.4e-5},
        {"column": 2, "iteration": 3, "all_candidates_converged": True},
    ]
    gate = pilot_convergence_gate(trace, [])
    assert gate["status"] == PILOT_GATE_BLOCKED
    assert len(gate["non_converged_selection_rows"]) == 1
    assert gate["non_converged_selection_rows"][0]["iteration"] == 2


def test_converged_final_candidates_cannot_clear_a_false_intermediate_row():
    """Exactly the Gate 74-B3 shape: final rows fine, an earlier one not."""

    final_rows = [{"column": 2, "candidate_family": "bernoulli",
                   "optimiser_converged": True, "bernoulli_grad_inf": 1e-12},
                  {"column": 2, "candidate_family": "poisson",
                   "optimiser_converged": True, "poisson_grad_inf": 1e-11}]
    trace = [{"column": 2, "iteration": 4, "all_candidates_converged": False},
             {"column": 2, "iteration": 8, "all_candidates_converged": True}]

    assert pilot_convergence_gate([], final_rows)["status"] == PILOT_GATE_PASS
    gate = pilot_convergence_gate(trace, final_rows)
    assert gate["status"] == PILOT_GATE_BLOCKED
    assert gate["non_converged_candidate_count"] == 0


def test_the_gate_function_ignores_the_new_fields():
    """Provenance is observational: the gate reads only the aggregate."""

    import inspect

    source = inspect.getsource(pilot_convergence_gate)
    for suffix in ("_grad_inf", "_scipy_success", "_scipy_status", "_n_iter"):
        assert suffix not in source, suffix
    assert "all_candidates_converged" in source
    assert "optimiser_converged" in source


# --------------------------------------------------------------------------
# 9. nothing here touches EM or the preserved evidence
# --------------------------------------------------------------------------

PRESERVED_RUN_DIRECTORIES = (
    "smoke_2026" "0923",
    "smoke_v2_2026" "0924",
    "optimizer_validation_2026" "0923",
    "optimizer_validation_analytic_jac_2026" "0923",
    "optimizer_migration_bfgs_2026" "0923",
)


def test_the_selector_reads_no_preserved_artifact_directory():
    """The provenance hardening lives in the selector, which reads no run.

    The names are split above so that this file does not contain them as
    literals: a scan of the source is the check, and a check that matches
    itself proves nothing.
    """

    source = (_HERE / "family_selection.py").read_text(encoding="utf-8")
    for preserved in PRESERVED_RUN_DIRECTORIES:
        assert preserved not in source, preserved
    assert "results/family_selection" not in source
    assert "results\family_selection" not in source
