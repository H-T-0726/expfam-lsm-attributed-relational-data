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
    json.dump(report, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
