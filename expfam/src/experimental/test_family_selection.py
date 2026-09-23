"""Zero-fit tests for the family-selection machinery (Issue #74, Scope B).

These tests execute NO EM and touch no file under ``expfam/results``.  A
module-scoped autouse fixture makes an accidental MCEM call raise, so "no EM
was run" is enforced rather than asserted in prose.  Everything below runs on
FIXED posterior samples supplied by the test itself.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import gammaln

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from family_selection import (  # noqa: E402
    ADAM_MAX_ITER,
    CANDIDATE_PRIORITY,
    CandidateRecord,
    SelectorStop,
    column_log_likelihood,
    initial_assignment,
    optimise_column_loading,
    score_column_candidates,
    select_from_records,
    support_gate,
)

N, K, L = 24, 3, 4


@pytest.fixture(autouse=True)
def _forbid_em(monkeypatch):
    """Fail loudly if anything in this module tries to run an EM fit."""

    import em_runner
    import family_selection
    from model_dual_expfam_consistent import DualExpFamLSMPerColumnConsistent

    def _no_em(*_args, **_kwargs):
        raise AssertionError("an EM fit was attempted inside a zero-fit test")

    monkeypatch.setattr(em_runner, "run_em_experimental", _no_em)
    monkeypatch.setattr(family_selection, "run_em_experimental", _no_em)
    monkeypatch.setattr(DualExpFamLSMPerColumnConsistent, "calc_eta_newton", _no_em)


@pytest.fixture
def fixed_samples():
    """A fixed (n, K, L) posterior-sample block. No EM produced it."""

    rng = np.random.default_rng(20260923)
    return rng.standard_normal((N, K, L))


# --------------------------------------------------------------------------
# A1. support gate
# --------------------------------------------------------------------------

def test_gate_sends_binary_columns_to_bernoulli_versus_poisson():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(N, 1)).astype(float)
    X[0, 0], X[1, 0] = 0.0, 1.0          # make sure both values occur
    gate, = support_gate(X)
    assert gate.candidates == ("bernoulli", "poisson")
    assert gate.decided_by == "score"
    assert gate.is_ambiguous
    assert gate.fixed_family is None


def test_gate_fixes_counts_above_one_to_poisson():
    X = np.array([[0.0], [1.0], [2.0], [7.0]])
    gate, = support_gate(X)
    assert gate.candidates == ("poisson",)
    assert gate.decided_by == "gate"
    assert gate.fixed_family == "poisson"
    assert not gate.is_ambiguous


@pytest.mark.parametrize("values", [
    [[0.5], [1.0], [2.0]],               # non-integer
    [[-1.0], [0.0], [3.0]],              # negative
    [[-0.25], [0.75], [1.5]],            # both
])
def test_gate_fixes_real_or_negative_columns_to_gaussian(values):
    gate, = support_gate(np.array(values))
    assert gate.candidates == ("gaussian",)
    assert gate.decided_by == "gate"
    assert gate.fixed_family == "gaussian"


def test_gate_keeps_the_two_kinds_of_column_separable():
    """``decided_by`` is what stops gate-decided columns entering an accuracy number."""

    X = np.column_stack([
        np.array([0.0, 1.0, 1.0, 0.0]),          # binary   -> score
        np.array([0.0, 2.0, 3.0, 1.0]),          # counts   -> gate
        np.array([0.1, -2.0, 3.5, 1.0]),         # real     -> gate
    ])
    gates = support_gate(X)
    assert [g.decided_by for g in gates] == ["score", "gate", "gate"]
    assert [g.column for g in gates] == [0, 1, 2]
    ambiguous = [g.column for g in gates if g.is_ambiguous]
    assert ambiguous == [0]
    for gate in gates:
        row = gate.as_row()
        assert row["gate_reason"]
        assert row["n_candidates"] == len(gate.candidates)


def test_gate_rejects_non_finite_input():
    with pytest.raises(SelectorStop, match="non-finite"):
        support_gate(np.array([[0.0], [np.nan]]))


def test_initial_assignment_respects_the_gate():
    X = np.column_stack([
        np.array([0.0, 1.0, 1.0, 0.0]),
        np.array([0.0, 2.0, 3.0, 1.0]),
        np.array([0.1, -2.0, 3.5, 1.0]),
    ])
    gates = support_gate(X)
    assert initial_assignment(gates, "bernoulli") == ["bernoulli", "poisson", "gaussian"]
    assert initial_assignment(gates, "poisson") == ["poisson", "poisson", "gaussian"]
    with pytest.raises(SelectorStop, match="not among its candidates"):
        initial_assignment(gates, "gaussian")


# --------------------------------------------------------------------------
# A2. candidate score: completeness, determinism, finiteness
# --------------------------------------------------------------------------

def test_scores_are_finite_and_deterministic(fixed_samples):
    x = np.array([0.0, 1.0] * (N // 2))
    loading = np.full(K, 0.3)
    first = column_log_likelihood(x, fixed_samples, loading, "bernoulli")
    second = column_log_likelihood(x, fixed_samples, loading, "bernoulli")
    assert math.isfinite(first)
    assert first == second                       # bitwise, not approx


def test_poisson_score_includes_the_base_measure_term(fixed_samples):
    """``-log(x!)`` is computed, not assumed away.

    For a column containing a value above 1 the term is strictly negative, so
    omitting it would shift the Poisson candidate upward against Bernoulli.
    """

    x = np.array([0.0, 1.0, 2.0, 5.0] * (N // 4))
    loading = np.full(K, 0.1)
    eta = np.einsum("nkl,k->nl", fixed_samples, loading)
    without_base = float(np.sum(x[:, None] * eta - np.exp(eta))) / L
    expected_base = -float(np.sum(gammaln(x + 1.0)))
    actual = column_log_likelihood(x, fixed_samples, loading, "poisson")
    assert expected_base < 0.0
    assert actual == pytest.approx(without_base + expected_base)


def test_poisson_base_measure_is_exactly_zero_on_a_binary_column(fixed_samples):
    """0! = 1! = 1, so the term vanishes here -- as a fact about the data.

    The implementation still evaluates it; this test pins the numeric
    consequence so that nobody later "optimises away" the term in general.
    """

    x = np.array([0.0, 1.0] * (N // 2))
    assert float(np.sum(gammaln(x + 1.0))) == 0.0
    loading = np.full(K, 0.1)
    eta = np.einsum("nkl,k->nl", fixed_samples, loading)
    without_base = float(np.sum(x[:, None] * eta - np.exp(eta))) / L
    actual = column_log_likelihood(x, fixed_samples, loading, "poisson")
    assert actual == pytest.approx(without_base)


def test_gaussian_score_keeps_its_normalising_constants(fixed_samples):
    x = np.linspace(-1.0, 1.0, N)
    loading = np.full(K, 0.2)
    eta = np.einsum("nkl,k->nl", fixed_samples, loading)
    residual = x[:, None] - eta
    variance = float(np.mean(residual ** 2))
    expected = float(np.sum(
        -0.5 * residual ** 2 / variance
        - 0.5 * math.log(variance)
        - 0.5 * math.log(2.0 * math.pi))) / L
    assert column_log_likelihood(x, fixed_samples, loading, "gaussian") == \
        pytest.approx(expected)


def test_bernoulli_score_matches_a_direct_logaddexp_evaluation(fixed_samples):
    x = np.array([0.0, 1.0] * (N // 2))
    loading = np.full(K, 0.4)
    eta = np.einsum("nkl,k->nl", fixed_samples, loading)
    expected = float(np.sum(x[:, None] * eta - np.logaddexp(0.0, eta))) / L
    assert column_log_likelihood(x, fixed_samples, loading, "bernoulli") == \
        pytest.approx(expected)


def test_non_finite_inputs_fail_fast(fixed_samples):
    x = np.array([0.0, 1.0] * (N // 2))
    with pytest.raises(SelectorStop, match="unknown family"):
        column_log_likelihood(x, fixed_samples, np.zeros(K), "categorical")
    with pytest.raises(SelectorStop, match="loading_init is not finite"):
        optimise_column_loading(x, fixed_samples, "bernoulli",
                                loading_init=np.full(K, np.nan))


# --------------------------------------------------------------------------
# A2. candidate-specific loadings, equal budget
# --------------------------------------------------------------------------

def test_each_candidate_optimises_its_own_loading(fixed_samples):
    """Bernoulli and Poisson must not share ``f_l``: the links differ."""

    x = np.array([0.0, 1.0] * (N // 2))
    gate, = support_gate(x[:, None])
    init = np.full(K, 0.05)
    records = score_column_candidates(x, fixed_samples, gate, loading_init=init)
    assert [r.family for r in records] == ["bernoulli", "poisson"]
    bernoulli, poisson = records
    assert not np.allclose(bernoulli.loading, poisson.loading)
    # Each moved away from the shared starting point.
    assert not np.allclose(bernoulli.loading, init)
    assert not np.allclose(poisson.loading, init)
    # Equal optimiser budget: neither candidate is allowed more effort.
    assert bernoulli.n_iter <= ADAM_MAX_ITER and poisson.n_iter <= ADAM_MAX_ITER


def test_optimiser_improves_its_own_objective(fixed_samples):
    x = np.array([0.0, 1.0] * (N // 2))
    init = np.full(K, 0.05)
    for family in ("bernoulli", "poisson"):
        before = column_log_likelihood(x, fixed_samples, init, family)
        loading, sigma_sq, _ = optimise_column_loading(
            x, fixed_samples, family, loading_init=init)
        after = column_log_likelihood(x, fixed_samples, loading, family,
                                      sigma_sq=sigma_sq)
        assert after >= before, family


def test_scoring_is_reproducible_for_identical_inputs(fixed_samples):
    x = np.array([0.0, 1.0] * (N // 2))
    gate, = support_gate(x[:, None])
    init = np.full(K, 0.05)
    first = score_column_candidates(x, fixed_samples, gate, loading_init=init)
    second = score_column_candidates(x, fixed_samples, gate, loading_init=init)
    for left, right in zip(first, second):
        assert left.score == right.score
        assert np.array_equal(left.loading, right.loading)


def test_gaussian_candidate_profiles_its_variance(fixed_samples):
    x = np.linspace(-2.0, 2.0, N)
    gate, = support_gate(x[:, None])
    assert gate.candidates == ("gaussian",)
    record, = score_column_candidates(x, fixed_samples, gate,
                                      loading_init=np.zeros(K))
    assert record.sigma_sq is not None and record.sigma_sq > 0.0


# --------------------------------------------------------------------------
# A2. deterministic tie rule and margin
# --------------------------------------------------------------------------

def _record(family: str, score: float) -> CandidateRecord:
    return CandidateRecord(column=0, family=family, score=score,
                           loading=np.zeros(K), sigma_sq=None, n_iter=1)


def test_tie_rule_is_deterministic_and_follows_the_declared_priority():
    tied_one = [_record("bernoulli", -10.0), _record("poisson", -10.0)]
    tied_two = [_record("poisson", -10.0), _record("bernoulli", -10.0)]
    assert select_from_records(tied_one)[0] == "bernoulli"
    assert select_from_records(tied_two)[0] == "bernoulli"
    assert CANDIDATE_PRIORITY.index("bernoulli") < CANDIDATE_PRIORITY.index("poisson")
    # A tie has zero margin, which is what downstream diagnostics should see.
    assert select_from_records(tied_one)[1] == 0.0


def test_margin_is_the_criterion_scale_gap_to_the_runner_up():
    records = [_record("bernoulli", -10.0), _record("poisson", -12.5)]
    chosen, margin = select_from_records(records)
    assert chosen == "bernoulli"
    assert margin == pytest.approx((-2.0 * -12.5) - (-2.0 * -10.0))
    assert margin > 0.0


def test_single_candidate_has_zero_margin():
    chosen, margin = select_from_records([_record("gaussian", -3.0)])
    assert chosen == "gaussian"
    assert margin == 0.0


def test_selection_has_no_discrete_search_penalty():
    """HG-3: a constant added to every candidate cannot change the choice.

    ``sum_l log|M_l|`` is exactly such a constant when the candidate set is
    fixed, which is why it is not used as a selector penalty (design 6.4.1).
    """

    records = [_record("bernoulli", -10.0), _record("poisson", -12.5)]
    shifted = [_record(r.family, r.score - math.log(2)) for r in records]
    assert select_from_records(records)[0] == select_from_records(shifted)[0]
    assert select_from_records(records)[1] == pytest.approx(
        select_from_records(shifted)[1])


# --------------------------------------------------------------------------
# Scope B item 6: the standard lineage is untouched
# --------------------------------------------------------------------------

STANDARD_LINEAGE = (
    "reproduction/src/model.py",
    "expfam/src/model_expfam.py",
    "expfam/src/model_dual_expfam.py",
    "expfam/src/model_dual_expfam_fixed.py",
    "expfam/src/utils_expfam.py",
)

# Existing experimental modules that this work drives but must not modify.
UNTOUCHED_EXPERIMENTAL = (
    "expfam/src/experimental/model_dual_expfam_percolumn.py",
    "expfam/src/experimental/model_dual_expfam_consistent.py",
    "expfam/src/experimental/objective_consistent_numerics.py",
    "expfam/src/experimental/eval_utils.py",
    "expfam/src/experimental/em_runner.py",
    "expfam/src/experimental/data_generator_canonical.py",
)


def _changed_paths() -> list[str] | None:
    repo_root = _HERE.parents[2]
    try:
        merge_base = subprocess.run(
            ["git", "merge-base", "HEAD", "origin/main"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--name-only", merge_base, "HEAD"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line.strip() for line in diff.splitlines() if line.strip()]


def test_standard_lineage_and_reused_modules_are_unchanged():
    changed = _changed_paths()
    if changed is None:
        pytest.skip("git is unavailable or origin/main is not fetched here")
    offending = [path for path in changed
                 if path in STANDARD_LINEAGE or path in UNTOUCHED_EXPERIMENTAL]
    assert offending == [], f"this branch must not modify {offending}"


# --------------------------------------------------------------------------
# A4a. the in-EM update hook, exercised WITHOUT an E-step
# --------------------------------------------------------------------------

def _selecting_model(X, assignment, k=K, L=L):
    from family_selection import FamilySelectingPerColumnLSM

    model = FamilySelectingPerColumnLSM(
        gates=support_gate(X), n=X.shape[0], d=X.shape[1], k=k, L=L,
        family_x_list=assignment, family_y="bernoulli")
    model.initialize_params(true_params=None, seed=7)
    return model


@pytest.fixture
def mixed_columns():
    rng = np.random.default_rng(4242)
    X = np.column_stack([
        rng.integers(0, 2, N).astype(float),      # binary -> ambiguous
        rng.poisson(3.0, N).astype(float) + 2.0,  # counts -> gate
        rng.standard_normal(N),                   # real   -> gate
    ])
    return X


def test_only_ambiguous_columns_are_re_selected(mixed_columns, fixed_samples):
    model = _selecting_model(mixed_columns, ["bernoulli", "poisson", "gaussian"])
    assignment = model.select_families(mixed_columns, fixed_samples)
    assert len(assignment) == 3
    assert assignment[1] == "poisson"             # gate-fixed, untouched
    assert assignment[2] == "gaussian"            # gate-fixed, untouched
    assert {row["column"] for row in model.selection_trace} == {0}
    assert assignment[0] in ("bernoulli", "poisson")


def test_selection_trace_records_both_candidate_scores(mixed_columns, fixed_samples):
    model = _selecting_model(mixed_columns, ["bernoulli", "poisson", "gaussian"])
    model.select_families(mixed_columns, fixed_samples)
    row, = model.selection_trace
    assert "score_bernoulli" in row and "score_poisson" in row
    assert math.isfinite(row["score_bernoulli"])
    assert math.isfinite(row["score_poisson"])
    assert row["margin_neg2"] >= 0.0
    assert row["previous_family"] == "bernoulli"
    assert isinstance(row["changed"], bool)


def test_reassign_families_updates_the_column_index_sets(mixed_columns):
    model = _selecting_model(mixed_columns, ["bernoulli", "poisson", "gaussian"])
    model.reassign_families(["poisson", "poisson", "gaussian"])
    assert model.family_x_list == ["poisson", "poisson", "gaussian"]
    assert sorted(model.columns_of("poisson").tolist()) == [0, 1]
    assert model.columns_of("bernoulli").tolist() == []
    assert model.family_x == "mixed"
    model.reassign_families(["gaussian"] * 3)
    assert model.family_x == "gaussian"           # the all-Gaussian marker


def test_reassign_families_rejects_malformed_assignments(mixed_columns):
    model = _selecting_model(mixed_columns, ["bernoulli", "poisson", "gaussian"])
    with pytest.raises(SelectorStop, match="family_x_list length"):
        model.reassign_families(["gaussian", "gaussian"])
    with pytest.raises(SelectorStop, match="unknown family"):
        model.reassign_families(["categorical", "poisson", "gaussian"])


def test_calc_F_performs_the_update_and_the_flag_disables_it(mixed_columns,
                                                             fixed_samples):
    """``calc_F`` is where the A-type update happens; no E-step is involved."""

    model = _selecting_model(mixed_columns, ["bernoulli", "poisson", "gaussian"])
    loadings = model.calc_F(mixed_columns, fixed_samples)
    assert loadings.shape == (mixed_columns.shape[1], K)
    assert np.all(np.isfinite(loadings))
    assert len(model.selection_trace) == 1
    assert model.selection_trace[0]["iteration"] == 1

    model.family_update_enabled = False
    model.calc_F(mixed_columns, fixed_samples)
    assert len(model.selection_trace) == 1        # unchanged


def test_selecting_model_requires_one_gate_per_column(mixed_columns):
    from family_selection import FamilySelectingPerColumnLSM

    with pytest.raises(SelectorStop, match="one gate per column"):
        FamilySelectingPerColumnLSM(
            gates=support_gate(mixed_columns)[:2],
            n=mixed_columns.shape[0], d=mixed_columns.shape[1], k=K, L=L,
            family_x_list=["bernoulli", "poisson", "gaussian"],
            family_y="bernoulli")
