"""Out-of-process probe of the EM runner, for the convergence-sensitivity design.

Run as a script; prints one JSON object on stdout.  Never import this module
from a test.

WHY A SEPARATE PROCESS
----------------------
``test_k_true_robustness_sweep.py`` asserts, as a Phase 8b safety tripwire,
that ``"em_runner" not in sys.modules`` -- the whole research_audit suite must
never pull the real EM runner into the test process.  The convergence-
sensitivity design nevertheless has to measure three properties OF that
runner (trajectory nesting, a non-perturbing diagnostic, determinism).  Doing
that here, in a child process, keeps the tripwire literally true while still
measuring the real code.  Do not "simplify" this by importing em_runner in the
test module: that re-breaks 64 Phase 8b tests.

The problem is deliberately tiny (n = 24, d = 6).  These are statements about
control flow and the RNG stream, not about the science.  Nothing is written to
disk and no production fit is performed.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPERIMENTAL = ROOT / "expfam" / "src" / "experimental"
sys.path.insert(0, str(EXPERIMENTAL))

import em_runner                                             # noqa: E402
from em_runner_instrumented import run_em_instrumented         # noqa: E402
from data_generator_canonical import generate_canonical_data   # noqa: E402

N = 24
D = 6
K_TRUE = 2
K_EST = 4
L = 5
SEED = 7
SHORT = 8          # the frozen baseline budget
LONG = 16          # any longer budget; the claim is about the shared prefix


def _problem():
    dataset = generate_canonical_data(
        n=N, d=D, k=K_TRUE, seed=11,
        family_x="poisson", family_y="bernoulli",
        f_scale=0.6, w0=-1.0, w=1.0)
    rng = np.random.default_rng(99)
    upper = rng.random((N, N)) < 0.8
    train = np.triu(upper, 1)
    train = train | train.T
    return dataset, train


def _run(dataset, train, num_iter: int, *, diagnostic: bool):
    return em_runner.run_em_experimental(
        X=dataset.X, Y=dataset.Y,
        family_x="poisson", family_y="bernoulli",
        k=K_EST, L=L, num_iter=num_iter, seed=SEED,
        train_mask=train, numerics_mode="consistent",
        compute_strict_Q=True, mstep_q_diagnostic=diagnostic)


def _identical(a, b) -> bool:
    return (np.array_equal(a["Z_est"], b["Z_est"])
            and np.array_equal(a["F"], b["F"])
            and a["w"] == b["w"] and a["w0"] == b["w0"]
            and a["Q_strict"] == b["Q_strict"])


def main() -> int:
    dataset, train = _problem()

    short_diag = _run(dataset, train, SHORT, diagnostic=True)
    long_diag = _run(dataset, train, LONG, diagnostic=True)
    short_plain = _run(dataset, train, SHORT, diagnostic=False)
    short_again = _run(dataset, train, SHORT, diagnostic=False)

    hs = short_diag["mstep_q_history"]
    hl = long_diag["mstep_q_history"]
    prefix_matches = (
        len(hs) == SHORT and len(hl) == LONG
        and all(hs[i]["iteration"] == hl[i]["iteration"] == i + 1
                and hs[i]["q_before"] == hl[i]["q_before"]
                and hs[i]["q_after"] == hl[i]["q_after"]
                for i in range(SHORT)))

    report = {
        "short_iterations": SHORT,
        "long_iterations": LONG,
        # N1
        "prefix_is_bit_identical": bool(prefix_matches),
        "short_nan_occurred": bool(short_diag["nan_occurred"]),
        "long_nan_occurred": bool(long_diag["nan_occurred"]),
        "history_lengths": [len(hs), len(hl)],
        # N2
        "diagnostic_is_non_perturbing": bool(_identical(short_diag, short_plain)),
        # N3
        "repeated_call_is_identical": bool(_identical(short_plain, short_again)),
        # why new instrumentation is needed at all
        "plain_history_is_empty": short_plain["mstep_q_history"] == [],
        "per_iteration_keys": sorted(
            key for key in short_plain
            if "history" in key or "trajectory" in key),
        # context, not an assertion target
        "q_strict_short": short_diag["Q_strict"],
        "q_strict_long": long_diag["Q_strict"],
    }
    report.update(_instrumented_probes(dataset, train, short_plain))
    json.dump(report, sys.stdout)
    return 0



LADDER = (8, 12, 16, 24, 32, 40)


def _instrumented(dataset, train, max_iter, checkpoints, **kwargs):
    return run_em_instrumented(
        dataset.X, dataset.Y,
        family_x="poisson", family_y="bernoulli",
        k=K_EST, L=L, max_iter=max_iter, checkpoints=checkpoints,
        seed=SEED, train_mask=train, numerics_mode="consistent", **kwargs)


def _instrumented_probes(dataset, train, reference) -> dict:
    """Properties of the instrumented runner that the design depends on.

    I1  at the baseline budget it reproduces em_runner EXACTLY -- same
        Q_strict, BIC, num_params, Z, F, w, w0.  This is what makes the
        replication gate achievable at all.
    I2  the checkpoint-8 record of a 40-iteration run equals the whole of an
        8-iteration run.  Checkpoints are the budget ladder.
    I3  adding checkpoints does not perturb the run: a 40-iteration run
        evaluated at six checkpoints ends where one evaluated only at 40 ends.
        Checkpoint evaluation therefore consumes no randomness and mutates no
        optimiser state.
    """

    short = _instrumented(dataset, train, 8, (8,))
    s8 = short["checkpoint_records"][-1]
    long_ = _instrumented(dataset, train, 40, LADDER)
    l8 = next(r for r in long_["checkpoint_records"] if r["iteration"] == 8)
    l40 = next(r for r in long_["checkpoint_records"] if r["iteration"] == 40)
    bare = _instrumented(dataset, train, 40, (40,))
    b40 = bare["checkpoint_records"][-1]

    def same_state(x, y) -> bool:
        return (x["q_strict"] == y["q_strict"] and x["bic"] == y["bic"]
                and x["num_params"] == y["num_params"]
                and x["w"] == y["w"] and x["w0"] == y["w0"]
                and np.array_equal(x["Z_est"], y["Z_est"])
                and np.array_equal(x["F"], y["F"]))

    return {
        # I1
        "instrumented_matches_em_runner": bool(
            reference["Q_strict"] == s8["q_strict"]
            and reference["bic"] == s8["bic"]
            and reference["num_params"] == s8["num_params"]
            and reference["w"] == s8["w"] and reference["w0"] == s8["w0"]
            and np.array_equal(reference["Z_est"], s8["Z_est"])
            and np.array_equal(reference["F"], s8["F"])),
        # I2
        "checkpoint8_of_long_equals_short_run": bool(same_state(l8, s8)),
        # I3
        "checkpoints_do_not_perturb_the_run": bool(same_state(l40, b40)),
        "ladder_observed": [r["iteration"] for r in long_["checkpoint_records"]],
        "stopping_reason": long_["stopping_reason"],
        "actual_iterations": long_["actual_iterations"],
        "retry_count": long_["retry_count"],
        "nan_detected": bool(long_["nan_detected"]),
        "diagnostic_keys_present": sorted(
            k for k in l40
            if k.endswith("_change") or k.startswith("max_")),
    }


if __name__ == "__main__":
    raise SystemExit(main())
