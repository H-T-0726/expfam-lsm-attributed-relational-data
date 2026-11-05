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
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPERIMENTAL = ROOT / "expfam" / "src" / "experimental"
sys.path.insert(0, str(EXPERIMENTAL))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

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
    report.update(_production_callback_probes(dataset, train))
    report.update(_prefix_abort_probe(dataset, train))
    report.update(_auditor_agreement_probe(dataset, train))
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


def _tiny_spec(RUN, **overrides):
    body = dict(RUN.protocol_body_from_contract())
    body.update({"n_grid": [N], "replicates": 1, "candidate_k": [K_EST],
                 "max_iter": 40, "checkpoints": list(LADDER),
                 "baseline_checkpoint": 8, "primary_checkpoint": 40})
    body.update(overrides)
    return RUN.Spec(body)


def _production_callback_probes(dataset, train) -> dict:
    """H-02: exercise the ACTUAL production callback, not a stand-in.

    P1  A long run scored at every checkpoint by the production callback ends
        bit-identically to one scored only at the terminal iteration: same RNG
        endpoint, same Z, F, w, w0, Q and same terminal scores.  Intermediate
        scoring therefore consumes no randomness and mutates no optimiser
        state.
    P2  A callback that tries to write into the state it is handed FAILS.  The
        arrays are read-only copies, so a scoring bug cannot silently perturb
        the estimation path.
    """

    import run_em_convergence_sensitivity as RUN

    spec = _tiny_spec(RUN)
    n = dataset.X.shape[0]
    test = np.triu(~train, 1) & ~np.eye(n, dtype=bool)
    rows, cols = np.where(test)
    y_test = dataset.Y[rows, cols]
    identity = {"fit_index": 1, "k_true": K_TRUE, "n": n, "replicate": 1,
                "k_est": K_EST, "start": 1, "data_seed": 0, "split_seed": 0,
                "model_seed": SEED, "train_mask_hash": "x", "test_mask_hash": "y"}

    def scored(checkpoints):
        collected, states = [], {}
        callback = RUN.make_checkpoint_callback(
            identity=identity, dataset=dataset, y_test=y_test,
            test_rows=rows, test_cols=cols, collected=collected, spec=spec,
            states=states)
        result = _instrumented(dataset, train, 40, checkpoints,
                               on_checkpoint=callback)
        return result, collected, states

    laddered, ladder_rows, ladder_states = scored(LADDER)
    terminal, terminal_rows, _ = scored((40,))

    l40 = next(r for r in ladder_rows if r["checkpoint_iteration"] == 40)
    t40 = terminal_rows[-1]
    end_l = laddered["checkpoint_records"][-1]
    end_t = terminal["checkpoint_records"][-1]

    scores_match = all(
        l40[field] == t40[field]
        for field in ("heldout_mean_log_score", "q_strict", "s2_q_based",
                      "s3_plugin_conditional", "num_params", "w0_hat", "w_hat",
                      "checkpoint_state_digest"))
    state_match = (np.array_equal(end_l["Z_est"], end_t["Z_est"])
                   and np.array_equal(end_l["F"], end_t["F"])
                   and end_l["w"] == end_t["w"] and end_l["w0"] == end_t["w0"]
                   and end_l["q_strict"] == end_t["q_strict"])

    def mutating(record):
        record["Z_est"][0, 0] = 1.0e9

    try:
        _instrumented(dataset, train, 8, (8,), on_checkpoint=mutating)
        mutation_refused = False
    except ValueError:
        mutation_refused = True

    return {
        "production_callback_scores_match": bool(scores_match),
        "production_callback_state_match": bool(state_match),
        "production_callback_rows": len(ladder_rows),
        "production_callback_state_keys": sorted(
            {k.split("_it")[-1][3:] or k for k in ladder_states})[:1] and
        len(ladder_states),
        "callback_mutation_is_refused": bool(mutation_refused),
        "production_callback_columns": sorted(l40.keys()),
    }


def _prefix_abort_probe(dataset, train) -> dict:
    """B2-03: a baseline-checkpoint mismatch must stop the loop, not be logged.

    The callback is given an archive whose row for this fit is deliberately
    wrong.  It must raise from INSIDE the checkpoint, so the EM loop unwinds
    and iteration 9 is never entered -- which is what makes the check worth
    having: catching it after iteration 40 would waste the whole run.
    """

    import run_em_convergence_sensitivity as RUN

    spec = _tiny_spec(RUN)
    n = dataset.X.shape[0]
    test = np.triu(~train, 1) & ~np.eye(n, dtype=bool)
    rows, cols = np.where(test)
    y_test = dataset.Y[rows, cols]
    identity = {"fit_index": 1, "k_true": K_TRUE, "n": n, "replicate": 1,
                "k_est": K_EST, "start": 1, "data_seed": 0, "split_seed": 0,
                "model_seed": SEED, "train_mask_hash": "x", "test_mask_hash": "y"}

    # An archive row that is right in every field except one.
    truthful, _ = [], None
    collected_truth = []
    honest = RUN.make_checkpoint_callback(
        identity=identity, dataset=dataset, y_test=y_test, test_rows=rows,
        test_cols=cols, collected=collected_truth, spec=spec)
    _instrumented(dataset, train, 8, (8,), on_checkpoint=honest)
    truth = collected_truth[-1]
    del truthful

    key = (K_TRUE, n, 1, K_EST, 1)
    archive = {key: {field: repr(truth[field]) if isinstance(truth[field], float)
                     else str(truth[field])
                     for field in RUN.REPLICATION_FIELDS}}
    archive[key]["q_strict"] = repr(float(truth["q_strict"]) + 1.0)

    seen: list[int] = []
    collected: list = []
    callback = RUN.make_checkpoint_callback(
        identity=identity, dataset=dataset, y_test=y_test, test_rows=rows,
        test_cols=cols, collected=collected, spec=spec, archive=archive)
    raised = None
    try:
        _instrumented(dataset, train, 40, LADDER, on_checkpoint=callback,
                      on_iteration=seen.append)
    except RUN.PrefixReplicationError as exc:
        raised = exc

    return {
        "prefix_mismatch_raised": raised is not None,
        "prefix_mismatch_is_dedicated_error": type(raised).__name__
        if raised else None,
        "prefix_max_iteration_entered": max(seen) if seen else None,
        "prefix_iteration_nine_entered": 9 in seen,
        "prefix_rows_kept": len(collected),
        "prefix_problem_fields": [p.split(":")[0] for p in raised.problems]
        if raised else [],
    }


def _auditor_agreement_probe(dataset, train) -> dict:
    """The auditor's restated likelihoods must reproduce the runner's numbers.

    Without this the auditor's "independent recomputation" would be an
    independent computation of something else.
    """

    import run_em_convergence_sensitivity as RUN
    import audit_em_convergence_sensitivity as AUDIT

    body = dict(RUN.protocol_body_from_contract())
    body.update({"n_grid": [N], "replicates": 1, "candidate_k": [2, 3],
                 "max_iter": 8, "checkpoints": [4, 8],
                 "baseline_checkpoint": 4, "primary_checkpoint": 8})
    spec = RUN.Spec(body)
    states, iteration_rows = {}, []
    rows, fit_rows = RUN._run_cells(
        RUN.build_cells(spec), spec=spec, max_iter=8, checkpoints=(4, 8),
        archive_check=False, states=states, iteration_rows=iteration_rows,
        progress_every=10_000)

    regen = AUDIT.Regenerator(body)
    worst = {"s1": 0.0, "q": 0.0, "s2": 0.0, "s3": 0.0}
    for row in rows:
        key = row["state_key"]
        n, rep, k_est = int(row["n"]), int(row["replicate"]), int(row["k_est"])
        ds, tr, _t, ri, ci, y_test = regen.get(n, rep)
        z = np.asarray(states[f"{key}_Z"], dtype=np.float64)
        zs = np.asarray(states[f"{key}_Zsamples"], dtype=np.float64)
        f = np.asarray(states[f"{key}_F"], dtype=np.float64)
        w0, w, var_z = (float(v) for v in states[f"{key}_scalars"])
        params = AUDIT.num_params(k_est, int(body["d"]))
        eta = w0 + w * (z @ z.T)
        got = {
            "s1": AUDIT.heldout_mean_log_score(y_test, eta[ri, ci]),
            "q": AUDIT.q_strict(ds.X, ds.Y, zs, f, var_z, w0, w, tr),
            "s3": AUDIT.s3_plugin_conditional(ds.X, ds.Y, z, f, w0, w, params,
                                              n, tr),
        }
        got["s2"] = -2.0 * got["q"] + params * math.log(n)
        want = {"s1": float(row["heldout_mean_log_score"]),
                "q": float(row["q_strict"]),
                "s2": float(row["s2_q_based"]),
                "s3": float(row["s3_plugin_conditional"])}
        for name in worst:
            worst[name] = max(worst[name],
                              abs(got[name] - want[name])
                              / max(abs(want[name]), 1e-12))
    return {
        "auditor_agreement_worst_relative": worst,
        "auditor_agreement_within_tolerance": all(
            v <= AUDIT.RECOMPUTE_RTOL for v in worst.values()),
        "auditor_agreement_checkpoints": len(rows),
        "auditor_agreement_iteration_rows": len(iteration_rows),
        "auditor_agreement_fits": len(fit_rows),
    }


if __name__ == "__main__":
    raise SystemExit(main())
