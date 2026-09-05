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

N1 is not a formality.  The runner draws its generator once, before the
iteration loop (``em_runner`` line ~158), and nothing in the loop depends on
``num_iter`` -- but the NaN guard restarts the WHOLE run with ``seed + 1000``
if any iteration produces a non-finite latent sample, and a longer run has
strictly more iterations in which that can happen.  Nesting therefore holds
only for runs that stay clean, which is why the design requires
``nan_occurred`` to be recorded per budget and treats a NaN-triggered restart
as a pairing failure rather than as a result.

The measurements themselves run in a CHILD PROCESS
(``tools/research_audit/_em_budget_probe.py``).  ``test_k_true_robustness_sweep``
asserts that the whole research_audit suite never imports the real EM runner
(``"em_runner" not in sys.modules``); importing it here would break 64 Phase 8b
tests.  The subprocess keeps that tripwire literally true while still measuring
the real code, so do not inline the probe.

The probe is tiny (n = 24, d = 6): a statement about control flow and the RNG
stream, not about the science.  It runs no production fit and writes no
artifact.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBE = pathlib.Path(__file__).with_name("_em_budget_probe.py")
EM_RUNNER = ROOT / "expfam" / "src" / "experimental" / "em_runner.py"


@pytest.fixture(scope="module")
def probe() -> dict:
    """Run the out-of-process EM probe once and return its JSON report."""

    completed = subprocess.run(
        [sys.executable, str(PROBE)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout)


def test_the_probe_stays_out_of_this_process():
    """The Phase 8b tripwire this module is arranged around.

    If this fails, the probe has been inlined and
    ``test_S2b_real_adapter_is_never_reached_by_the_test_suite`` -- plus the 63
    Phase 8b tests that run after it -- will fail whenever the suite is run as
    a whole.
    """

    assert "em_runner" not in sys.modules


def test_n3_two_identical_calls_agree_bit_for_bit(probe):
    assert probe["repeated_call_is_identical"]


def test_n2_mstep_q_diagnostic_does_not_move_the_estimate(probe):
    assert probe["diagnostic_is_non_perturbing"]


def test_n1_long_run_reproduces_the_short_run_prefix(probe):
    # the condition the nesting claim is conditional on
    assert not probe["short_nan_occurred"]
    assert not probe["long_nan_occurred"]
    assert probe["history_lengths"] == [probe["short_iterations"],
                                        probe["long_iterations"]]
    assert probe["prefix_is_bit_identical"]


def test_the_runner_exposes_no_per_iteration_record_by_default(probe):
    """Why new instrumentation is needed at all.

    Without ``mstep_q_diagnostic`` the returned dict carries no per-iteration
    quantity, which is exactly why the archived 896-fit artifact cannot answer
    whether num_iter=8 had converged.
    """

    assert probe["plain_history_is_empty"]
    assert probe["per_iteration_keys"] == ["mstep_q_history"]


# ------------------------------------------------- the instrumented runner

def test_i1_instrumented_runner_reproduces_em_runner_at_the_baseline_budget(probe):
    """The property the replication gate will demand of the real cells.

    If the instrumented loop and em_runner disagree here, then any difference
    between checkpoint 8 and checkpoint 40 in production would confound "longer
    budget" with "different code path", and the paired design collapses.
    """

    assert probe["instrumented_matches_em_runner"]


def test_i2_checkpoint_eight_of_a_long_run_equals_a_short_run(probe):
    """Checkpoints ARE the budget ladder, not an approximation of it."""

    assert probe["checkpoint8_of_long_equals_short_run"]
    assert probe["ladder_observed"] == [8, 12, 16, 24, 32, 40]


def test_i3_checkpoint_evaluation_perturbs_nothing(probe):
    """Evaluating at six checkpoints ends where evaluating only at 40 ends.

    Consequence: checkpoint evaluation consumes no randomness and mutates no
    optimiser state.  This is a production gate, not a nicety -- if it failed,
    the ladder itself would be changing the thing it measures.
    """

    assert probe["checkpoints_do_not_perturb_the_run"]


def test_the_instrumented_runner_has_no_adaptive_stopping(probe):
    assert probe["stopping_reason"] == "fixed_budget"
    assert probe["actual_iterations"] == 40
    assert probe["retry_count"] == 0 and probe["nan_detected"] is False


def test_the_instrumented_runner_records_rotation_invariant_diagnostics(probe):
    """Raw parameter change alone must not be the only stability measure.

    The latent space is identified only up to an orthogonal transform, so F can
    move without the fitted distribution moving; F F^T cannot.  Both are
    recorded, per criterion and per window.
    """

    assert probe["diagnostic_keys_present"] == [
        "f_raw_change", "gram_rel_change",
        "max_f_raw_change_last4", "max_gram_rel_change_last4",
        "max_w0_change_last4", "max_w_change_last4",
        "w0_change", "w_change",
    ]


def test_nan_retry_is_invisible_in_the_returned_counters():
    """F-08, restated as a test so the new runner cannot inherit it silently.

    ``nan_count`` is reset at the top of every retry and the loop breaks on the
    first clean pass, so a run whose first attempt hit NaN returns exactly the
    same counters as one that never did.  The new runner must record the retry
    itself, not just the final pass's counter.  Read as source text, so this
    check also stays out of the test process.
    """

    source = EM_RUNNER.read_text(encoding="utf-8")
    assert "nan_count = 0" in source
    assert "for retry in range(max_retries + 1):" in source
    assert "rng = np.random.default_rng(seed + retry * 1000)" in source
    # the break that discards the failed attempt's history
    assert "if not nan_occurred:" in source
