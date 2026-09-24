"""Zero-EM tests for the Gate 74-B5 selected-candidate loading installation.

Gate 74-B4 found that the winning candidate's loading produced the winning
score and was then discarded, while the parent 50-step Adam row was carried
forward instead.  Gate 74-B5 installs the winning loading for ambiguous
columns and changes nothing else.  These tests pin both halves of that
sentence on the deterministic fixed-Z inputs from Gate 74-B1: what must now be
different, and what must stay exactly as it was.

No real E-step, no EM runner, no preserved artifact directory.
"""

from __future__ import annotations

import inspect
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
from model_dual_expfam_consistent import (                     # noqa: E402
    DualExpFamLSMPerColumnConsistent,
)


@pytest.fixture(autouse=True)
def _forbid_em(monkeypatch):
    """Fail loudly if anything in this module tries to run an E-step or EM."""

    import em_runner

    def _no_em(*_args, **_kwargs):
        raise AssertionError("an E-step or EM fit was attempted in a B5 test")

    monkeypatch.setattr(em_runner, "run_em_experimental", _no_em)
    monkeypatch.setattr(DualExpFamLSMPerColumnConsistent, "calc_eta_newton",
                        _no_em)


@pytest.fixture(scope="module")
def Z_samples():
    """The deterministic Gate 74-B1 posterior-sample block. No RNG, no EM."""

    return b1.build_z_samples()


def _mixed_X() -> np.ndarray:
    """Two ambiguous columns and two gate-decided ones, all deterministic."""

    ambiguous_a = b1.build_case_column("case_A_balanced_nonseparable")
    ambiguous_b = b1.build_case_column("case_B_sparse_nonseparable")
    real = np.linspace(-1.5, 1.5, 24)                     # gate: gaussian
    counts = (np.arange(24) % 5).astype(np.float64)       # gate: poisson
    return np.column_stack([ambiguous_a, real, ambiguous_b, counts])


START = ["bernoulli", "gaussian", "poisson", "poisson"]
AMBIGUOUS = (0, 2)
GATE_DECIDED = (1, 3)


def _model(X: np.ndarray, start=START) -> FamilySelectingPerColumnLSM:
    model = FamilySelectingPerColumnLSM(
        gates=support_gate(X), n=X.shape[0], d=X.shape[1], k=3, L=5,
        family_x_list=list(start), family_y="bernoulli")
    model.initialize_params(true_params=None, seed=7)
    return model


def test_the_fixture_has_the_intended_gate_verdicts():
    gates = support_gate(_mixed_X())
    assert [g.column for g in gates if g.is_ambiguous] == list(AMBIGUOUS)
    assert [g.fixed_family for g in gates] == [None, "gaussian", None,
                                               "poisson"]


# --------------------------------------------------------------------------
# 1 and 2. the installed row is the winning record's loading
# --------------------------------------------------------------------------

def test_each_ambiguous_row_is_exactly_the_winning_loading(Z_samples):
    X = _mixed_X()
    model = _model(X)
    captured: dict = {}
    original = model.select_families

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.update(model.current_iteration_winners)
        return result

    model.select_families = spy                     # observe, do not change
    F = model.calc_F(X, Z_samples)

    assert set(captured) == set(AMBIGUOUS)
    for column in AMBIGUOUS:
        assert np.array_equal(F[column, :], captured[column].loading)


def test_the_installed_loading_comes_from_the_score_winning_record(Z_samples):
    X = _mixed_X()
    model = _model(X)
    F = model.calc_F(X, Z_samples)
    rows = {row["column"]: row for row in model.selection_trace}

    for column in AMBIGUOUS:
        row = rows[column]
        records = [r for r in model.last_candidate_records
                   if r.column == column]
        best = max(records, key=lambda r: r.score)
        # Family, score and loading all come from the same record.
        assert row["selected_family"] == best.family
        assert row["selected_candidate_family"] == best.family
        assert row["selected_candidate_score"] == best.score
        assert row["selected_loading_sha256"] == fs.loading_digest(best.loading)
        assert row["installed_loading_sha256"] == \
            fs.loading_digest(F[column, :])
        assert row["selected_loading_installed"] is True
        assert row["loading_installation_policy"] == \
            "selected_candidate_loading"
        # The losing loading is never installed.
        loser, = [r for r in records if r is not best]
        assert not np.array_equal(F[column, :], loser.loading)


def test_the_serialised_loading_round_trips_to_the_installed_row(Z_samples):
    X = _mixed_X()
    model = _model(X)
    F = model.calc_F(X, Z_samples)
    for row in model.selection_trace:
        values = np.array([float(v) for v in row["selected_loading"].split("|")])
        assert np.array_equal(values, F[row["column"], :])


# --------------------------------------------------------------------------
# 3. gate-decided rows are exactly the parent calc_F output
# --------------------------------------------------------------------------

def test_non_ambiguous_rows_equal_the_parent_update(Z_samples):
    X = _mixed_X()
    selecting = _model(X)
    F = selecting.calc_F(X, Z_samples)

    # A control with the same initial parameters and the assignment the
    # selection produced, updated by the parent calc_F alone.
    control = _model(X, start=selecting.family_x_list)
    parent_F = DualExpFamLSMPerColumnConsistent.calc_F(control, X, Z_samples)

    for column in GATE_DECIDED:
        assert np.array_equal(F[column, :], parent_F[column, :]), column
    # And the ambiguous rows are exactly where the two differ.
    for column in AMBIGUOUS:
        assert not np.array_equal(F[column, :], parent_F[column, :]), column


# --------------------------------------------------------------------------
# 4. scores, selection, margin and convergence are unchanged
# --------------------------------------------------------------------------

def test_scoring_is_unchanged_by_installation(Z_samples):
    """Same fixed inputs, with and without the installing calc_F."""

    X = _mixed_X()
    scoring_only = _model(X)
    scoring_only.select_families(X, Z_samples)
    reference = {row["column"]: row for row in scoring_only.selection_trace}

    installing = _model(X)
    installing.calc_F(X, Z_samples)
    installed = {row["column"]: row for row in installing.selection_trace}

    for column in AMBIGUOUS:
        for key in ("selected_family", "margin_neg2", "score_bernoulli",
                    "score_poisson", "all_candidates_converged",
                    "bernoulli_converged", "poisson_converged",
                    "bernoulli_grad_inf", "poisson_grad_inf",
                    "bernoulli_n_iter", "poisson_n_iter",
                    "selected_loading_sha256"):
            assert installed[column][key] == reference[column][key], \
                (column, key)


# --------------------------------------------------------------------------
# 5. one optimiser call per candidate, and no more
# --------------------------------------------------------------------------

def test_installation_adds_no_optimiser_invocation(monkeypatch, Z_samples):
    X = _mixed_X()
    model = _model(X)
    calls = {"n": 0}
    original = fs.minimize

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fs, "minimize", counting)
    model.calc_F(X, Z_samples)
    # Two ambiguous columns, two candidates each.
    assert calls["n"] == 4


def test_the_install_step_does_not_optimise():
    source = inspect.getsource(FamilySelectingPerColumnLSM.calc_F)
    body = source.split('"""')[2]
    for forbidden in ("minimize(", "optimise_", "score_column_candidates"):
        assert forbidden not in body, forbidden


# --------------------------------------------------------------------------
# 6. no stale winner can leak across iterations
# --------------------------------------------------------------------------

def test_winners_are_consumed_by_the_call_that_selected_them(Z_samples):
    X = _mixed_X()
    model = _model(X)
    model.calc_F(X, Z_samples)
    assert model.current_iteration_winners == {}


def test_a_disabled_update_installs_nothing_even_with_a_stale_winner(
        Z_samples):
    X = _mixed_X()
    model = _model(X)
    stale = fs.CandidateRecord(column=0, family="bernoulli", score=0.0,
                               loading=np.array([9.0, 9.0, 9.0]),
                               sigma_sq=None, n_iter=1)
    model.current_iteration_winners = {0: stale}
    model.family_update_enabled = False

    F = model.calc_F(X, Z_samples)
    assert not np.array_equal(F[0, :], stale.loading)
    assert model.current_iteration_winners == {}
    assert model.selection_trace == []


def test_a_stale_winner_is_replaced_by_the_new_selection(Z_samples):
    X = _mixed_X()
    model = _model(X)
    stale = fs.CandidateRecord(column=0, family="bernoulli", score=0.0,
                               loading=np.array([9.0, 9.0, 9.0]),
                               sigma_sq=None, n_iter=1)
    model.current_iteration_winners = {0: stale}

    F = model.calc_F(X, Z_samples)
    row = next(r for r in model.selection_trace if r["column"] == 0)
    assert not np.array_equal(F[0, :], stale.loading)
    assert row["installed_loading_sha256"] == row["selected_loading_sha256"]


def test_each_iteration_installs_its_own_winners(Z_samples):
    """Two consecutive calls: the second installs the second call's records."""

    X = _mixed_X()
    model = _model(X)
    model.calc_F(X, Z_samples)
    model.params["F"] = model.params["F"] + 0.05     # a different warm start
    F2 = model.calc_F(X, Z_samples)

    second = [r for r in model.selection_trace if r["iteration"] == 2]
    assert {r["column"] for r in second} == set(AMBIGUOUS)
    for row in second:
        assert row["installed_loading_sha256"] == row["selected_loading_sha256"]
        assert row["installed_loading_sha256"] == \
            fs.loading_digest(F2[row["column"], :])


# --------------------------------------------------------------------------
# 7. the S1 gate is unchanged
# --------------------------------------------------------------------------

def test_the_aggregate_flag_keeps_its_meaning(Z_samples):
    X = _mixed_X()
    model = _model(X)
    model.calc_F(X, Z_samples)
    for row in model.selection_trace:
        assert row["all_candidates_converged"] == (
            row["bernoulli_converged"] and row["poisson_converged"])


def test_the_s1_gate_still_blocks_on_a_false_intermediate_row():
    trace = [{"column": 0, "iteration": 1, "all_candidates_converged": False},
             {"column": 0, "iteration": 2, "all_candidates_converged": True}]
    final = [{"column": 0, "optimiser_converged": True}]
    assert pilot_convergence_gate([], final)["status"] == PILOT_GATE_PASS
    assert pilot_convergence_gate(trace, final)["status"] == PILOT_GATE_BLOCKED


def test_the_gate_does_not_read_the_installation_fields():
    source = inspect.getsource(pilot_convergence_gate)
    for field in ("selected_loading", "installed_loading",
                  "loading_installation"):
        assert field not in source


# --------------------------------------------------------------------------
# 8. frozen BFGS settings
# --------------------------------------------------------------------------

def test_bfgs_settings_are_frozen():
    assert fs.PHASE9C_CANDIDATE_OPTIMIZER == fs.OPTIMIZER_BFGS
    assert (fs.BFGS_METHOD, fs.BFGS_MAXITER, fs.BFGS_GTOL) == \
        ("BFGS", 2000, 1e-10)
    assert fs.CANDIDATE_GRAD_INF_TOL == 1e-8
    assert fs.LOADING_INSTALLATION_POLICY == "selected_candidate_loading"


# --------------------------------------------------------------------------
# 9. the fresh refit is unchanged and reuses no exploration loading
# --------------------------------------------------------------------------

def test_the_fresh_refit_takes_only_the_assignment(monkeypatch):
    captured: dict = {}

    def fake_refit(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"failure_policy": "fail_fast", "retry_count": 0,
                "replacement_count": 0, "seed_rescue_count": 0}

    def fake_exploration(*_args, **_kwargs):
        return fs.ExplorationResult(
            selected_assignment=["bernoulli"], initial_assignment=["bernoulli"],
            selection_trace=[], final_candidate_rows=[], margins={},
            iterations=1, metadata={})

    monkeypatch.setattr(fs, "run_em_experimental", fake_refit)
    monkeypatch.setattr(fs, "run_family_exploration", fake_exploration)
    fs.run_hybrid_family_selection(
        np.array([[0.0], [1.0], [1.0], [0.0]]), np.zeros((4, 4)),
        k=1, ambiguous_start="bernoulli", refit_seed=943001)

    kwargs = captured["kwargs"]
    assert kwargs["family_x_list"] == ["bernoulli"]
    assert kwargs["seed"] == 943001
    assert kwargs["failure_policy"] == "fail_fast"
    # A fresh refit: no loading, F or parameter state is handed over.
    for forbidden in ("F", "F_init", "loadings", "init_params",
                      "true_params", "params"):
        assert forbidden not in kwargs, forbidden
    assert len(captured["args"]) == 2            # X and Y only


# --------------------------------------------------------------------------
# 10. the historical Adam route is callable and pinned
# --------------------------------------------------------------------------

def test_the_historical_adam_route_is_pinned(Z_samples):
    x = b1.build_case_column("case_A_balanced_nonseparable")
    init = np.array([0.05, -0.02, 0.01])
    direct = fs.optimise_column_loading(x, Z_samples, "bernoulli",
                                        loading_init=init)
    routed = fs.optimise_candidate_loading(x, Z_samples, "bernoulli",
                                           loading_init=init,
                                           optimizer=fs.OPTIMIZER_ADAM)
    assert np.array_equal(direct[0], routed[0])
    assert direct[2] == routed[2] == 50
    assert direct[3] is routed[3] is False


def test_the_adam_selector_also_installs_its_own_winner(Z_samples):
    """The installation rule is about which record won, not which optimiser."""

    X = _mixed_X()
    model = FamilySelectingPerColumnLSM(
        gates=support_gate(X), n=24, d=4, k=3, L=5,
        family_x_list=list(START), family_y="bernoulli",
        candidate_optimizer=fs.OPTIMIZER_ADAM)
    model.initialize_params(true_params=None, seed=7)
    F = model.calc_F(X, Z_samples)
    for row in model.selection_trace:
        assert row["installed_loading_sha256"] == row["selected_loading_sha256"]
        assert row["installed_loading_sha256"] == \
            fs.loading_digest(F[row["column"], :])
