"""Design-phase tests for the EM convergence-sensitivity experiment.

The proposed design (``reports/k_selection/em_convergence_sensitivity_design_20260905.md``)
rests on three properties of the existing EM runner.  If any of them is false,
the paired comparison against the archived 896-fit baseline is invalid and the
protocol must not be frozen.  They are asserted here so that a later change to
``em_runner`` or to the model lineage breaks a test instead of silently
invalidating the design.

  N1  TRAJECTORY NESTING.  With the same seed, the first ``t`` iterations of a
      run with ``num_iter = T > t`` are bit-identical to a run with
      ``num_iter = t``.  This is what allows one long run to stand in for a
      whole ladder of budgets.
  N2  NON-PERTURBING DIAGNOSTIC.  ``mstep_q_diagnostic=True`` does not change
      the estimate.  This is what allows a per-iteration Q record to be taken
      without changing the numbers being compared.
  N3  DETERMINISM.  Two identical calls agree bit for bit.

These are deliberately tiny (n = 24, d = 6): they are statements about the
control flow and the RNG stream, not about the science.  They run no
production fit and write no artifact.

N1 is not a formality.  The runner draws its generator once, before the
iteration loop (``em_runner`` line ~158), and nothing in the loop depends on
``num_iter`` -- but the NaN guard restarts the WHOLE run with ``seed + 1000``
if any iteration produces a non-finite latent sample, and a longer run has
strictly more iterations in which that can happen.  Nesting therefore holds
only for runs that stay clean, which is why the design requires
``nan_occurred`` to be recorded per budget and treats a NaN-triggered restart
as a pairing failure rather than as a result.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPERIMENTAL = ROOT / "expfam" / "src" / "experimental"
if str(EXPERIMENTAL) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTAL))

import em_runner                                            # noqa: E402
from data_generator_canonical import generate_canonical_data  # noqa: E402

N = 24
D = 6
K_TRUE = 2
K_EST = 4
L = 5
SEED = 7
SHORT = 8          # the frozen baseline budget
LONG = 16          # any longer budget; the claim is about the shared prefix


@pytest.fixture(scope="module")
def problem():
    """A small canonical Poisson-X / Bernoulli-Y instance with a dyad holdout."""

    dataset = generate_canonical_data(
        n=N, d=D, k=K_TRUE, seed=11,
        family_x="poisson", family_y="bernoulli",
        f_scale=0.6, w0=-1.0, w=1.0)
    rng = np.random.default_rng(99)
    upper = rng.random((N, N)) < 0.8
    train = np.triu(upper, 1)
    train = train | train.T
    return dataset, train


def _run(problem, num_iter: int, *, diagnostic: bool):
    dataset, train = problem
    return em_runner.run_em_experimental(
        X=dataset.X, Y=dataset.Y,
        family_x="poisson", family_y="bernoulli",
        k=K_EST, L=L, num_iter=num_iter, seed=SEED,
        train_mask=train, numerics_mode="consistent",
        compute_strict_Q=True, mstep_q_diagnostic=diagnostic)


def test_n3_two_identical_calls_agree_bit_for_bit(problem):
    a = _run(problem, SHORT, diagnostic=False)
    b = _run(problem, SHORT, diagnostic=False)
    assert np.array_equal(a["Z_est"], b["Z_est"])
    assert np.array_equal(a["F"], b["F"])
    assert a["w"] == b["w"] and a["w0"] == b["w0"]
    assert a["Q_strict"] == b["Q_strict"]


def test_n2_mstep_q_diagnostic_does_not_move_the_estimate(problem):
    with_diag = _run(problem, SHORT, diagnostic=True)
    without = _run(problem, SHORT, diagnostic=False)
    assert np.array_equal(with_diag["Z_est"], without["Z_est"])
    assert np.array_equal(with_diag["F"], without["F"])
    assert with_diag["w"] == without["w"]
    assert with_diag["w0"] == without["w0"]
    assert with_diag["Q_strict"] == without["Q_strict"]


def test_n1_long_run_reproduces_the_short_run_prefix(problem):
    short = _run(problem, SHORT, diagnostic=True)
    long_ = _run(problem, LONG, diagnostic=True)

    # The guard the nesting claim is conditional on.
    assert not short["nan_occurred"] and not long_["nan_occurred"]

    hs, hl = short["mstep_q_history"], long_["mstep_q_history"]
    assert len(hs) == SHORT and len(hl) == LONG
    for i in range(SHORT):
        assert hs[i]["iteration"] == hl[i]["iteration"] == i + 1
        assert hs[i]["q_before"] == hl[i]["q_before"], i
        assert hs[i]["q_after"] == hl[i]["q_after"], i


def test_the_runner_exposes_no_per_iteration_record_by_default(problem):
    """Why new instrumentation is needed at all.

    Without ``mstep_q_diagnostic`` the returned dict carries no per-iteration
    quantity, which is exactly why the archived 896-fit artifact cannot answer
    whether num_iter=8 had converged.
    """

    result = _run(problem, SHORT, diagnostic=False)
    assert result["mstep_q_history"] == []
    per_iteration = [key for key in result
                     if "history" in key or "trajectory" in key]
    assert per_iteration == ["mstep_q_history"]


def test_nan_retry_is_invisible_in_the_returned_counters():
    """F-08, restated as a test so the new runner cannot inherit it silently.

    ``nan_count`` is reset at the top of every retry and the loop breaks on the
    first clean pass, so a run whose first attempt hit NaN returns exactly the
    same counters as one that never did.  The new runner must record the retry
    itself, not just the final pass's counter.
    """

    source = (EXPERIMENTAL / "em_runner.py").read_text(encoding="utf-8")
    assert "nan_count = 0" in source
    assert "for retry in range(max_retries + 1):" in source
    assert "rng = np.random.default_rng(seed + retry * 1000)" in source
    # the break that discards the failed attempt's history
    assert "if not nan_occurred:" in source
