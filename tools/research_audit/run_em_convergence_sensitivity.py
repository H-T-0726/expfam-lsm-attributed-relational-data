"""EM convergence-sensitivity sweep -- runner.

Question (unchanged from the design document): does relaxing the fixed
8-iteration optimization budget materially change the selected-K pattern under
the otherwise frozen clean-generator protocol?

Everything except the optimization budget is inherited from the archived clean
true-K sweep: generator, families, d, L, masks, seeds, candidate K, starts, the
criteria, the tie rule and the selector.  Those are imported from
``run_clean_true_k_sweep`` rather than restated, so they cannot drift.

There is NO adaptive stopping.  Every fit runs the full budget and the criteria
are evaluated at a fixed, preregistered checkpoint ladder, so the iteration
count cannot depend on K, on the data, or on a held-out score.  The held-out
score is used for selection only; nothing in this file can stop a fit.

Nothing here decides scope.  ``--replication-gate`` proves the new code path
reproduces the archive; ``--full`` runs production and is fail-closed.  Neither
may be run before the protocol is frozen and a human authorises it under
CLAUDE.md section 6.

Usage
-----
    python tools/research_audit/run_em_convergence_sensitivity.py --plan
    python tools/research_audit/run_em_convergence_sensitivity.py \
        --replication-gate --allow-em
    python tools/research_audit/run_em_convergence_sensitivity.py \
        --full --allow-em --confirm-em-convergence-sweep
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
EXPERIMENTAL = ROOT / "expfam" / "src" / "experimental"
if str(EXPERIMENTAL) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTAL))

# The baseline module is the single source of the frozen settings.  Importing
# it does NOT import any EM runner: run_clean_true_k_sweep imports em_runner
# lazily inside its production function, and so does this module.
import run_clean_true_k_sweep as BASE                        # noqa: E402

# ===========================================================================
# FROZEN PROTOCOL -- DRAFT.  Nothing here may change after the protocol hash is
# recorded, and nothing here may be chosen or altered after seeing a result.
# ===========================================================================

PROTOCOL_VERSION = "em-convergence-protocol-draft-1"
ARTIFACT_VERSION = "em-convergence-artifact-v1"
EXPERIMENT_ID = "em_convergence_sensitivity_20260905"

#: The only thing this experiment changes.
MAX_ITER = 40
CHECKPOINTS = (8, 12, 16, 24, 32, 40)
#: Primary comparison.  The intermediate checkpoints describe the trajectory
#: and are secondary; 40 is a BUDGET, and is never called "converged".
BASELINE_CHECKPOINT = 8
PRIMARY_CHECKPOINT = 40

#: Scope.  K_TRUE=5 only; n is a module constant so that --plan, the runtime
#: estimate and the protocol hash cannot disagree about it.
K_TRUE = 5
N_GRID = (75, 100, 150)
REPLICATES = 8
STABILITY_WINDOW = 4
#: Reported at every level.  No single tau is designated "the" threshold:
#: choosing one after seeing results is exactly what preregistration prevents.
TAU_LADDER = (1e-2, 1e-3, 1e-4)

#: Replication gate cell, fixed by RULE and not by inspection: the first
#: replicate of the smallest scoped n.  Written down before the gate ever ran.
GATE_N = min(N_GRID)
GATE_REPLICATE = 1

BASELINE_RUN_DIR = (ROOT / "expfam" / "results" / "k_selection"
                    / "clean_true_k_asymptotics_20260904")
BASELINE_PROTOCOL_HASH = ("547880a16aef6530cfdf7903c4e32f16062397e0"
                          "bacc0c109d5c77fb9892ccc0")
ARTIFACT_DIR = (ROOT / "expfam" / "results" / "k_selection"
                / EXPERIMENT_ID)

#: Fields the gate compares against the archive.  runtime_s is deliberately
#: absent: it is wall-clock and cannot be reproduced.  Every other field is
#: written by csv from a Python float repr, which round-trips exactly, so the
#: comparison is EXACT -- no tolerance is defined, because none is needed.
GATE_EXACT_FIELDS = (
    "data_seed", "split_seed", "model_seed", "num_params",
    "heldout_mean_log_score", "q_strict", "s2_q_based", "s3_plugin_conditional",
)
GATE_INT_FIELDS = ("data_seed", "split_seed", "model_seed", "num_params")


def frozen_protocol() -> dict[str, Any]:
    """The hashable body.  Inherited values are READ from the baseline."""

    base = json.loads((BASELINE_RUN_DIR / "protocol.json")
                      .read_text(encoding="utf-8"))["protocol"]
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": ("does relaxing the fixed 8-iteration optimization budget "
                     "materially change the selected-K pattern under the "
                     "otherwise frozen clean-generator protocol"),
        "baseline_run": BASELINE_RUN_DIR.name,
        "baseline_protocol_hash": BASELINE_PROTOCOL_HASH,
        "max_iter": MAX_ITER,
        "checkpoints": list(CHECKPOINTS),
        "baseline_checkpoint": BASELINE_CHECKPOINT,
        "primary_checkpoint": PRIMARY_CHECKPOINT,
        "adaptive_stopping": False,
        "stopping_reason_expected": "fixed_budget",
        "stability_window": STABILITY_WINDOW,
        "tau_ladder": list(TAU_LADDER),
        "k_true": K_TRUE,
        "n_grid": list(N_GRID),
        "replicates": REPLICATES,
        "gate_cell": {"k_true": K_TRUE, "n": GATE_N, "replicate": GATE_REPLICATE},
        "on_nan": "fail",
        # inherited, not restated
        "candidate_k": base["candidate_k"],
        "starts": base["starts"],
        "family_x": base["family_x"],
        "family_y": base["family_y"],
        "d": base["d"],
        "L": base["L"],
        "test_ratio": base["test_ratio"],
        "tie_tolerance": base["tie_tolerance"],
        "data_seed_base": base["data_seed_base"],
        "split_seed_base": base["split_seed_base"],
        "model_seed_base": base["model_seed_base"],
        "w0_true": base["w0_true"],
        "w_ref": base["w_ref"],
        "k_ref": base["k_ref"],
        "row_norm_sq_target": base["row_norm_sq_target"],
        "poisson_lambda_max": base["poisson_lambda_max"],
        "numerics_mode": base["numerics_mode"],
        "generator_version": base["generator_version"],
        "criteria": base["criteria"],
    }


def protocol_hash() -> str:
    payload = json.dumps(frozen_protocol(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expected_fit_count() -> int:
    return (len(N_GRID) * REPLICATES
            * len(BASE.CANDIDATE_K) * len(BASE.STARTS))


def expected_checkpoint_rows() -> int:
    return expected_fit_count() * len(CHECKPOINTS)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class HarnessStop(RuntimeError):
    """Refusal.  Never caught inside this module."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessStop(message)


def git_head() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def working_tree_clean() -> bool:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                         capture_output=True, text=True)
    return out.returncode == 0 and out.stdout.strip() == ""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_archive_fits() -> dict[tuple[int, int, int, int, int], dict[str, str]]:
    """Archived baseline rows keyed by (k_true, n, replicate, k_est, start)."""

    path = BASELINE_RUN_DIR / "fit_results.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(int(r["k_true"]), int(r["n"]), int(r["replicate"]),
             int(r["k_est"]), int(r["start"])): r for r in rows}


def read_archive_masks() -> dict[tuple[int, int, int], tuple[str, str]]:
    """Archived train/test mask hashes, from generator_provenance.csv."""

    path = BASELINE_RUN_DIR / "generator_provenance.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(int(r["K_TRUE"]), int(r["n"]), int(r["replicate"])):
            (r["train_mask_hash"], r["test_mask_hash"]) for r in rows}


@dataclass(frozen=True)
class Cell:
    k_true: int
    n: int
    replicate: int


def build_cells(n_values: tuple[int, ...] = N_GRID,
                replicates: int = REPLICATES) -> list[Cell]:
    return [Cell(K_TRUE, n, r)
            for n in n_values for r in range(1, replicates + 1)]


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

CHECKPOINT_COLUMNS = [
    "fit_index", "k_true", "n", "replicate", "k_est", "start",
    "data_seed", "split_seed", "model_seed",
    "checkpoint_iteration",
    "heldout_mean_log_score", "q_strict", "s2_q_based", "s3_plugin_conditional",
    "num_params",
    "gram_rel_change", "f_raw_change", "w_change", "w0_change",
    "max_gram_rel_change_last4", "max_f_raw_change_last4",
    "max_w_change_last4", "max_w0_change_last4",
    *(f"budget_stable_gram_{tau:g}" for tau in TAU_LADDER),
    "w0_hat", "w_hat",
    "cumulative_runtime_s", "nan_iterations_so_far", "attempt_index",
    "checkpoint_state_digest",
]

FIT_COLUMNS = [
    "fit_index", "k_true", "n", "replicate", "k_est", "start",
    "data_seed", "split_seed", "model_seed",
    "max_iter", "actual_iterations", "stopping_reason",
    "retry_count", "retry_seed", "nan_detected", "attempts_used",
    "retry_events", "runtime_s",
    "train_mask_hash", "test_mask_hash",
]


def _state_digest(z_est: np.ndarray, f: np.ndarray, w0: float, w: float) -> str:
    """A cheap, order-fixed digest of the checkpoint state.

    Lets an auditor tell 'the same state was reached' from 'the numbers happen
    to round the same way' without storing whole matrices.
    """

    h = hashlib.sha256()
    for array in (np.ascontiguousarray(z_est, dtype=np.float64),
                  np.ascontiguousarray(f, dtype=np.float64)):
        h.update(array.tobytes())
        h.update(str(array.shape).encode())
    h.update(np.float64(w0).tobytes())
    h.update(np.float64(w).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# the fit loop, shared by the gate and by production
# ---------------------------------------------------------------------------

def _run_cells(cells: list[Cell], *, max_iter: int, checkpoints: tuple[int, ...],
               progress_every: int = 25) -> tuple[list[dict], list[dict]]:
    """Run every (cell, K, start) fit and return (checkpoint rows, fit rows).

    The EM modules are imported HERE, not at module scope: the Phase 8b suite
    asserts that no research_audit test ever pulls a real EM runner into the
    test process, and importing at module scope would break that for every
    test that imports this file.
    """

    from em_runner_instrumented import (                      # noqa: PLC0415
        RetryDetected, run_em_instrumented,
    )

    checkpoint_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    index = 0

    for cell in cells:
        scale = BASE.f_scale_for_row_norm(BASE.ROW_NORM_SQ_TARGET,
                                          d=BASE.D_FEATURES, k=cell.k_true)
        w_true = BASE.w_for_matched_y_signal(BASE.W_REF, k=cell.k_true,
                                             k_ref=BASE.K_REF)
        dataset = BASE.generate_canonical_data(
            n=cell.n, d=BASE.D_FEATURES, k=cell.k_true,
            seed=BASE.data_seed(cell.k_true, cell.n, cell.replicate),
            family_x=BASE.FAMILY_X, family_y=BASE.FAMILY_Y,
            f_scale=scale, w0=BASE.W0_TRUE, w=w_true,
            poisson_lambda_max=BASE.POISSON_LAMBDA_MAX)
        train, test = BASE.build_masks(
            cell.n, BASE.split_seed(cell.k_true, cell.n, cell.replicate))
        test_rows, test_cols = np.where(np.triu(test, 1))
        y_test = dataset.Y[test_rows, test_cols]
        train_hash = BASE.stable_hash(train)
        test_hash = BASE.stable_hash(test)

        for k_est in BASE.CANDIDATE_K:
            for start in BASE.STARTS:
                index += 1
                seed = BASE.model_seed(cell.k_true, cell.n, cell.replicate,
                                       k_est, start)
                identity = {
                    "fit_index": index, "k_true": cell.k_true, "n": cell.n,
                    "replicate": cell.replicate, "k_est": k_est, "start": start,
                    "data_seed": BASE.data_seed(cell.k_true, cell.n, cell.replicate),
                    "split_seed": BASE.split_seed(cell.k_true, cell.n, cell.replicate),
                    "model_seed": seed,
                }

                collected: list[dict[str, Any]] = []

                def on_checkpoint(record: dict[str, Any],
                                  _identity=identity,
                                  _dataset=dataset, _y_test=y_test,
                                  _rows=test_rows, _cols=test_cols,
                                  _collected=collected) -> None:
                    """Score one checkpoint.  Deterministic; consumes no RNG.

                    The held-out score is computed here for SELECTION only.  No
                    branch in this callback can shorten or extend the run.
                    """

                    z_est = record["Z_est"]
                    w0_hat, w_hat = record["w0"], record["w"]
                    eta = w0_hat + w_hat * (z_est @ z_est.T)
                    eta_test = eta[_rows, _cols]
                    _require(bool(np.all(np.isfinite(eta_test))),
                             f"non-finite held-out eta at fit "
                             f"{_identity['fit_index']} checkpoint "
                             f"{record['iteration']}")
                    score = BASE.heldout_bernoulli_mean_log_score(_y_test, eta_test)
                    s3 = BASE.s3_plugin_conditional(
                        record["model"], _dataset.X, _dataset.Y, z_est,
                        record["F"], w0_hat, w_hat, record["num_params"],
                        _identity["n"])
                    _require(math.isfinite(score) and math.isfinite(record["bic"])
                             and math.isfinite(s3),
                             f"non-finite criterion at fit "
                             f"{_identity['fit_index']} checkpoint "
                             f"{record['iteration']}")
                    row = dict(_identity)
                    row.update({
                        "checkpoint_iteration": record["iteration"],
                        "heldout_mean_log_score": score,
                        "q_strict": record["q_strict"],
                        "s2_q_based": record["bic"],
                        "s3_plugin_conditional": s3,
                        "num_params": record["num_params"],
                        "w0_hat": w0_hat, "w_hat": w_hat,
                        "cumulative_runtime_s": record["cumulative_runtime_s"],
                        "nan_iterations_so_far": record["nan_iterations_so_far"],
                        "attempt_index": record["attempt_index"],
                        "checkpoint_state_digest": _state_digest(
                            z_est, record["F"], w0_hat, w_hat),
                    })
                    for key in ("gram_rel_change", "f_raw_change",
                                "w_change", "w0_change"):
                        row[key] = record[key]
                        row[f"max_{key}_last4"] = record[f"max_{key}_last4"]
                    for tau in TAU_LADDER:
                        row[f"budget_stable_gram_{tau:g}"] = bool(
                            record["max_gram_rel_change_last4"] < tau)
                    _collected.append(row)

                t0 = time.perf_counter()
                try:
                    result = run_em_instrumented(
                        dataset.X, dataset.Y,
                        family_x=BASE.FAMILY_X, family_y=BASE.FAMILY_Y,
                        k=k_est, L=BASE.L_SAMPLES,
                        max_iter=max_iter, checkpoints=checkpoints,
                        seed=seed, train_mask=train,
                        numerics_mode=BASE.NUMERICS_MODE,
                        stability_window=STABILITY_WINDOW,
                        on_nan="fail", on_checkpoint=on_checkpoint)
                except RetryDetected as exc:
                    raise HarnessStop(
                        f"fit {index} requested a NaN restart ({exc}); a restart "
                        f"re-seeds and voids pairing with the archived baseline, "
                        f"so this is a fit failure, not something to absorb"
                    ) from exc
                elapsed = time.perf_counter() - t0

                _require(len(collected) == len(checkpoints),
                         f"fit {index} produced {len(collected)} checkpoints, "
                         f"expected {len(checkpoints)}")
                _require(result["stopping_reason"] == "fixed_budget",
                         f"fit {index} stopped for {result['stopping_reason']!r}")
                _require(result["actual_iterations"] == max_iter,
                         f"fit {index} ran {result['actual_iterations']} of "
                         f"{max_iter} iterations")
                checkpoint_rows.extend(collected)

                fit_rows.append({
                    **identity,
                    "max_iter": max_iter,
                    "actual_iterations": result["actual_iterations"],
                    "stopping_reason": result["stopping_reason"],
                    "retry_count": result["retry_count"],
                    "retry_seed": result["retry_seed"],
                    "nan_detected": bool(result["nan_detected"]),
                    "attempts_used": result["attempts_used"],
                    "retry_events": json.dumps(result["retry_events"]),
                    "runtime_s": round(elapsed, 3),
                    "train_mask_hash": train_hash,
                    "test_mask_hash": test_hash,
                })

                if index % progress_every == 0:
                    print(f"  fit {index} K_TRUE={cell.k_true} n={cell.n} "
                          f"r={cell.replicate} K={k_est} start={start} "
                          f"{elapsed:.1f}s", flush=True)

    return checkpoint_rows, fit_rows


# ---------------------------------------------------------------------------
# replication gate
# ---------------------------------------------------------------------------

def compare_to_archive(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare checkpoint-8 rows to the archived 8-iteration fits.

    EXACT equality on every comparable field.  csv writes Python floats with
    repr, which round-trips through float() without loss, so an exact test is
    achievable and no tolerance is defined -- a tolerance here would only be a
    place for a real discrepancy to hide.
    """

    archive = read_archive_fits()
    masks = read_archive_masks()
    mismatches: list[dict[str, Any]] = []
    compared = 0

    for row in rows:
        if int(row["checkpoint_iteration"]) != BASELINE_CHECKPOINT:
            continue
        key = (int(row["k_true"]), int(row["n"]), int(row["replicate"]),
               int(row["k_est"]), int(row["start"]))
        reference = archive.get(key)
        if reference is None:
            mismatches.append({"key": key, "field": "row", "reason": "absent"})
            continue
        compared += 1
        for field in GATE_EXACT_FIELDS:
            want = (int(reference[field]) if field in GATE_INT_FIELDS
                    else float(reference[field]))
            got = row[field]
            if want != got:
                mismatches.append({"key": key, "field": field,
                                   "archive": repr(want), "new": repr(got)})
        want_masks = masks.get(key[:3])
        if want_masks is None:
            mismatches.append({"key": key, "field": "mask_provenance",
                               "reason": "absent"})

    return {"compared_rows": compared, "mismatches": mismatches,
            "match": not mismatches and compared > 0}


def compare_masks_to_archive(fit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    archive = read_archive_masks()
    out: list[dict[str, Any]] = []
    for row in fit_rows:
        key = (int(row["k_true"]), int(row["n"]), int(row["replicate"]))
        want = archive.get(key)
        if want is None:
            out.append({"key": key, "reason": "absent from provenance"})
        elif (row["train_mask_hash"], row["test_mask_hash"]) != want:
            out.append({"key": key, "reason": "mask hash differs",
                        "archive": want,
                        "new": (row["train_mask_hash"], row["test_mask_hash"])})
    return out


def run_replication_gate(*, allow_em: bool) -> dict[str, Any]:
    _require(allow_em, "the replication gate runs EM; pass --allow-em")
    cells = [Cell(K_TRUE, GATE_N, GATE_REPLICATE)]
    started = time.time()
    rows, fit_rows = _run_cells(cells, max_iter=BASELINE_CHECKPOINT,
                                checkpoints=(BASELINE_CHECKPOINT,),
                                progress_every=7)
    comparison = compare_to_archive(rows)
    mask_mismatches = compare_masks_to_archive(fit_rows)
    retries = [r for r in fit_rows if int(r["retry_count"]) != 0]
    verdict = ("PASS" if comparison["match"] and not mask_mismatches
               and not retries else "FAIL")
    report = {
        "gate": "replication",
        "experiment_id": EXPERIMENT_ID,
        "protocol_hash": protocol_hash(),
        "cell": {"k_true": K_TRUE, "n": GATE_N, "replicate": GATE_REPLICATE},
        "cell_selection_rule": ("the first replicate of the smallest scoped n; "
                                "fixed by rule, not chosen by inspection"),
        "expected_fits": len(BASE.CANDIDATE_K) * len(BASE.STARTS),
        "observed_fits": len(fit_rows),
        "compared_rows": comparison["compared_rows"],
        "mismatches": comparison["mismatches"],
        "mask_mismatches": mask_mismatches,
        "fits_with_retries": len(retries),
        "wall_clock_seconds": round(time.time() - started, 1),
        "verdict": verdict,
        "production_authorization": "YES" if verdict == "PASS" else "NO",
        "run_code_sha": git_head(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


# ---------------------------------------------------------------------------
# production
# ---------------------------------------------------------------------------

def run_production(*, allow_em: bool, confirm: bool) -> dict[str, Any]:
    _require(allow_em, "production requires --allow-em")
    _require(confirm, "production requires --confirm-em-convergence-sweep")
    _require(not ARTIFACT_DIR.exists(),
             f"artifact directory already exists; refusing to overwrite or "
             f"resume: {ARTIFACT_DIR}")

    started = time.time()
    run_code_sha = git_head()
    tree_clean_before = working_tree_clean()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)

    write_json(ARTIFACT_DIR / "protocol.json", {
        "protocol": frozen_protocol(),
        "protocol_hash": protocol_hash(),
        "protocol_version": PROTOCOL_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "experiment_id": EXPERIMENT_ID,
    })

    try:
        rows, fit_rows = _run_cells(build_cells(), max_iter=MAX_ITER,
                                    checkpoints=CHECKPOINTS)
    except BaseException as exc:                       # noqa: BLE001
        write_json(ARTIFACT_DIR / "failure.json", {
            "experiment_id": EXPERIMENT_ID,
            "failure": f"{type(exc).__name__}: {exc}",
            "wall_clock_seconds": round(time.time() - started, 1),
            "policy": ("no retry, no replacement, no resume, no seed change, "
                       "no tolerance change; this run is over"),
        })
        raise

    _require(len(fit_rows) == expected_fit_count(),
             f"{len(fit_rows)} fits, expected {expected_fit_count()}")
    _require(len(rows) == expected_checkpoint_rows(),
             f"{len(rows)} checkpoint rows, expected "
             f"{expected_checkpoint_rows()}")

    write_csv(ARTIFACT_DIR / "checkpoint_results.csv", rows, CHECKPOINT_COLUMNS)
    write_csv(ARTIFACT_DIR / "fit_results.csv", fit_rows, FIT_COLUMNS)

    replication = compare_to_archive(rows)
    write_json(ARTIFACT_DIR / "replication_vs_baseline.json", {
        "baseline_run": BASELINE_RUN_DIR.name,
        "baseline_protocol_hash": BASELINE_PROTOCOL_HASH,
        "checkpoint": BASELINE_CHECKPOINT,
        "compared_rows": replication["compared_rows"],
        "mismatches": replication["mismatches"],
        "match": replication["match"],
    })

    elapsed = round(time.time() - started, 1)
    write_json(ARTIFACT_DIR / "runinfo.json", {
        "experiment_id": EXPERIMENT_ID,
        "protocol_hash": protocol_hash(),
        "runner_version": "em-convergence-runner-v1",
        "attempted_fit_count": len(fit_rows),
        "completed_fit_count": len(fit_rows),
        "checkpoint_row_count": len(rows),
        "expected_fit_count": expected_fit_count(),
        "expected_checkpoint_rows": expected_checkpoint_rows(),
        "fits_with_retries": sum(1 for r in fit_rows if int(r["retry_count"])),
        "adaptive_stopping": False,
        "stopping_reasons": sorted({r["stopping_reason"] for r in fit_rows}),
        "wall_clock_seconds": elapsed,
        "run_code_sha": run_code_sha,
        "working_tree_clean_before": tree_clean_before,
        "working_tree_clean_after": working_tree_clean(),
        "notes": ("counters are policy declarations, not tallies: this runner "
                  "has no retry, replacement or resume path, and a NaN restart "
                  "raises instead of restarting"),
    })
    print(json.dumps({"status": "COMPLETE", "fits": len(fit_rows),
                      "checkpoint_rows": len(rows),
                      "wall_clock_seconds": elapsed}, indent=2))
    return {"fits": len(fit_rows), "checkpoint_rows": len(rows)}


# ---------------------------------------------------------------------------
# planning (no EM)
# ---------------------------------------------------------------------------

def plan(n_values: tuple[int, ...] = N_GRID) -> dict[str, Any]:
    """Scope and runtime projection from the ARCHIVE only.  Runs no EM."""

    path = BASELINE_RUN_DIR / "fit_results.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        archive = list(csv.DictReader(handle))
    per_n: dict[int, float] = {}
    for n in n_values:
        scoped = [r for r in archive
                  if int(r["k_true"]) == K_TRUE and int(r["n"]) == n]
        per_n[n] = sum(float(r["runtime_s"]) for r in scoped)
    fits = len(n_values) * REPLICATES * len(BASE.CANDIDATE_K) * len(BASE.STARTS)
    seconds = sum(per_n.values()) * MAX_ITER / BASELINE_CHECKPOINT
    return {
        "n_grid": list(n_values),
        "fits": fits,
        "checkpoint_rows": fits * len(CHECKPOINTS),
        "baseline_seconds_at_8_iterations": {n: round(v, 1)
                                             for n, v in per_n.items()},
        "projected_hours_at_max_iter": round(seconds / 3600, 2),
        "projection_note": ("linear in iterations; conservative, since the "
                            "per-checkpoint evaluation does not scale with "
                            "iterations and the archive's fixed end-of-run "
                            "cost is counted as if it did"),
        "protocol_hash": protocol_hash(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--plan-n", type=int, nargs="+", default=None,
                        help="runtime projection for an alternative n grid")
    parser.add_argument("--replication-gate", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--allow-em", action="store_true")
    parser.add_argument("--confirm-em-convergence-sweep", action="store_true")
    args = parser.parse_args(argv)

    if args.plan or args.plan_n:
        grid = tuple(args.plan_n) if args.plan_n else N_GRID
        print(json.dumps(plan(grid), indent=2, sort_keys=True))
        return 0
    if args.replication_gate:
        report = run_replication_gate(allow_em=args.allow_em)
        return 0 if report["verdict"] == "PASS" else 1
    if args.full:
        run_production(allow_em=args.allow_em,
                       confirm=args.confirm_em_convergence_sweep)
        return 0
    parser.error("choose --plan, --replication-gate or --full")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
