"""EM convergence-sensitivity sweep -- runner.

SCIENTIFIC ESTIMAND (deliberately narrow)
-----------------------------------------
Under the same data, masks, initialization and first-eight-iteration RNG
prefix, how does CONTINUING THE SAME L=5 stochastic MCEM procedure from
iteration 8 to iteration 40 change the selected-K pattern?

That is not "the effect of the optimization budget".  Continuing the procedure
draws fresh Monte-Carlo latent samples at every further iteration, so the
contrast mixes additional optimization with additional Monte-Carlo variation
and the two cannot be separated by this design.  The following claims are
FORBIDDEN in any report built on this experiment:

  * a pure optimization effect
  * "the 8-iteration budget caused the under-selection"
  * "the criterion is exonerated"
  * deterministic convergence, or convergence of any kind
  * consistency

Everything except the number of iterations is inherited from the archived
clean true-K sweep: generator, families, d, L, masks, seeds, candidate K,
starts, the criteria, the tie rule and the selector.  Those come from
``run_clean_true_k_sweep``, not from a second copy here.

There is NO adaptive stopping.  Every fit runs the full budget and the
criteria are evaluated at a fixed preregistered ladder, so the iteration count
cannot depend on K, on the data, or on a held-out score.  Nothing in this file
can stop a fit.

PRODUCTION IS FAIL-CLOSED
-------------------------
``run_production`` starts no fit unless every one of the checks in
``authorize_production`` passes: a committed frozen protocol whose body hashes
to its recorded value, HEAD equal to the approved commit, a clean tree, a
gate receipt that PASSED and is bound to that same protocol hash, code SHA and
environment, and a baseline artifact whose protocol hash is the expected one.
No CLI flag bypasses any of it.

Usage
-----
    python tools/research_audit/run_em_convergence_sensitivity.py --plan
    python tools/research_audit/run_em_convergence_sensitivity.py --emit-draft
    python tools/research_audit/run_em_convergence_sensitivity.py --authorize
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
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
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

# Importing the baseline module pulls in NO EM runner: it imports em_runner
# lazily inside its production function, and so does this module.  The Phase 8b
# suite asserts that no research_audit test drags an EM runner into the test
# process.
import run_clean_true_k_sweep as BASE                        # noqa: E402

# ===========================================================================
# PROTOCOL CONSTANTS -- these BUILD the draft.  Production never reads them:
# it loads the committed frozen protocol file and uses that.
# ===========================================================================

PROTOCOL_VERSION = "em-convergence-protocol-draft-2"
ARTIFACT_VERSION = "em-convergence-artifact-v2"
EXPERIMENT_ID = "em_convergence_sensitivity_20260905"

#: The only thing this experiment changes.
MAX_ITER = 40
CHECKPOINTS = (8, 12, 16, 24, 32, 40)
#: Primary comparison.  The intermediate checkpoints describe the trajectory
#: and are secondary; 40 is a BUDGET and is never called "converged".
BASELINE_CHECKPOINT = 8
PRIMARY_CHECKPOINT = 40

#: Scope, settled by human decision on 2026-09-05: the baseline's WHOLE n grid,
#: so the scope is no longer a subset chosen after seeing which n
#: under-selected.
K_TRUE = 5
N_GRID = (50, 75, 100, 150)
REPLICATES = 8
STABILITY_WINDOW = 4
TAU_LADDER = (1e-2, 1e-3, 1e-4)

#: Gate cell, fixed by RULE: the first replicate of the smallest scoped n.
GATE_N = min(N_GRID)
GATE_REPLICATE = 1
GATE_EXPECTED_FITS = 14

BASELINE_RUN_DIR = (ROOT / "expfam" / "results" / "k_selection"
                    / "clean_true_k_asymptotics_20260904")
BASELINE_PROTOCOL_HASH = ("547880a16aef6530cfdf7903c4e32f16062397e0"
                          "bacc0c109d5c77fb9892ccc0")

DESIGN_DIR = ROOT / "reports" / "k_selection"
DRAFT_PROTOCOL_PATH = DESIGN_DIR / f"{EXPERIMENT_ID}_protocol.DRAFT.json"
#: Written by the freeze commit, not by this runner.
FROZEN_PROTOCOL_PATH = DESIGN_DIR / f"{EXPERIMENT_ID}_protocol.json"
#: Written by a human at authorisation time, not by this runner.
AUTHORIZATION_PATH = DESIGN_DIR / f"{EXPERIMENT_ID}_authorization.json"
GATE_RECEIPT_PATH = DESIGN_DIR / f"{EXPERIMENT_ID}_gate_receipt.json"

ARTIFACT_DIR = (ROOT / "expfam" / "results" / "k_selection" / EXPERIMENT_ID)

#: Compared against the archive.  ``runtime_s`` is deliberately absent: it is
#: wall-clock and cannot be reproduced.  Every other field is written by csv
#: from a Python float repr, which round-trips exactly, so the comparison is
#: EXACT -- no tolerance is defined, because a tolerance would only be
#: somewhere for a real discrepancy to hide.
REPLICATION_FIELDS = (
    "data_seed", "split_seed", "model_seed", "num_params",
    "heldout_mean_log_score", "q_strict", "s2_q_based", "s3_plugin_conditional",
)
REPLICATION_INT_FIELDS = ("data_seed", "split_seed", "model_seed", "num_params")

DIAGNOSTIC_KEYS = ("gram_rel_change", "f_raw_change", "w_change", "w0_change",
                   "eta_x_rel_change", "eta_y_rel_change")


class HarnessStop(RuntimeError):
    """Refusal.  Never caught inside this module."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessStop(message)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def git_head() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def working_tree_clean() -> bool:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                         capture_output=True, text=True)
    return out.returncode == 0 and out.stdout.strip() == ""


def environment_fingerprint() -> dict[str, Any]:
    """What the numbers were produced on.

    Recorded rather than reconstructed: KI-014 exists because reconstructed
    provenance is a guess wearing the costume of a measurement.  A mismatch
    against the gate receipt is a refusal, never a reason to relax a tolerance.
    """

    try:
        import scipy                                          # noqa: PLC0415
        scipy_version = scipy.__version__
    except Exception:                                          # noqa: BLE001
        scipy_version = "absent"
    blas: dict[str, Any]
    try:
        blas = {lib.get("internal_api", lib.get("name", "?")):
                lib.get("version", "?")
                for lib in np.show_config(mode="dicts")  # type: ignore[call-arg]
                .get("Build Dependencies", {}).values()
                if isinstance(lib, dict)}
    except Exception:                                          # noqa: BLE001
        blas = {"status": "unavailable"}
    import os                                                  # noqa: PLC0415
    threads = {name: os.environ.get(name)
               for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                            "VECLIB_MAXIMUM_THREADS")}
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "blas": blas,
        "threads": threads,
    }


def environment_key(fingerprint: dict[str, Any]) -> str:
    """The part of the fingerprint an equality check may use.

    Deliberately narrow.  ``platform.platform()`` carries a build number that
    changes with an OS update, and thread env vars are per-shell; requiring
    those to match would turn every routine change into a refusal without
    telling us anything about numerics.
    """

    return json.dumps({k: fingerprint[k] for k in ("python", "numpy", "scipy",
                                                   "machine")},
                      sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# protocol lifecycle
# ---------------------------------------------------------------------------

def canonical_hash(body: dict[str, Any]) -> str:
    """SHA256 of the body-only canonical JSON.  One definition, used everywhere."""

    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")).hexdigest()


def protocol_body_from_constants() -> dict[str, Any]:
    """Build the body from the constants above.  Used to EMIT the draft.

    Production never calls this: it loads the committed frozen file, so a later
    edit to a constant cannot silently redefine a frozen protocol.
    """

    base = json.loads((BASELINE_RUN_DIR / "protocol.json")
                      .read_text(encoding="utf-8"))["protocol"]
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "estimand": (
            "under the same data, masks, initialization and first-eight-"
            "iteration RNG prefix, how does continuing the same L=5 stochastic "
            "MCEM procedure from iteration 8 to iteration 40 change the "
            "selected-K pattern"),
        "forbidden_claims": [
            "pure optimization effect",
            "the 8-iteration budget caused the under-selection",
            "the criterion is exonerated",
            "deterministic convergence",
            "convergence of any kind",
            "consistency",
        ],
        "residual_confound": (
            "continuing the procedure redraws L=5 latent samples every further "
            "iteration, so additional optimization and additional Monte-Carlo "
            "variation are not separable by this design"),
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
        "diagnostic_keys": list(DIAGNOSTIC_KEYS),
        "k_true": K_TRUE,
        "n_grid": list(N_GRID),
        "replicates": REPLICATES,
        "gate_cell": {"k_true": K_TRUE, "n": GATE_N, "replicate": GATE_REPLICATE},
        "gate_expected_fits": GATE_EXPECTED_FITS,
        "on_nan": "fail",
        "replication_fields": list(REPLICATION_FIELDS),
        "all_fit_checkpoint8_replication": True,
        "primary_outcomes": _primary_outcome_spec(),
        # inherited, read from the baseline artifact and never retyped
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


def _primary_outcome_spec() -> dict[str, Any]:
    """Frozen aggregation.  Written out so no ordering choice is left to later."""

    return {
        "P1": {
            "name": "selected K by budget",
            "definition": (
                "for each (n, replicate) and each checkpoint: average S1 over "
                "the two starts for every candidate K, take the K with the "
                "largest average; tie tolerance 1e-12, ties resolved to the "
                "smallest K"),
            "reported": [
                "replicate-level K8 -> K40 transition",
                "exact / under / over counts at each checkpoint",
                "transition counts by n",
            ],
            "comparison": "checkpoint 8 versus checkpoint 40, paired within replicate",
        },
        "P2": {
            "name": "per-candidate-K S1 change from continuing the procedure",
            "definition": (
                "delta(n, r, K) = 0.5 * sum over start in {1,2} of "
                "[ S1(n, r, K, start, 40) - S1(n, r, K, start, 8) ]"),
            "reported": [
                "all 8 replicate-level deltas for every (n, K), not only a mean",
                "the mean over replicates, described",
            ],
            "inference": (
                "DESCRIPTIVE ONLY. No formal test. Spearman between K and the "
                "mean delta may be shown as a diagnostic; the 7 candidate K are "
                "NOT independent samples and no p-value is reported"),
        },
        "secondary": [
            "winning margin (best minus runner-up in selector space) by checkpoint",
            "start disagreement by checkpoint",
            "budget-stability rates by candidate K and n at every tau",
            "S2 and S3 versions of P1 and P2",
            "cumulative runtime by checkpoint",
        ],
        "wording": (
            "no materiality threshold is defined, so the question says CHANGES, "
            "not 'materially changes'"),
    }


def load_frozen_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the committed frozen protocol and verify its own hash.

    Refuses the DRAFT: a draft has no hash, and running production from a file
    that can still be edited is exactly the failure the freeze exists to stop.
    """

    _require(FROZEN_PROTOCOL_PATH.exists(),
             f"no frozen protocol at {FROZEN_PROTOCOL_PATH}; the protocol is "
             f"still a DRAFT and production is not authorised")
    document = json.loads(FROZEN_PROTOCOL_PATH.read_text(encoding="utf-8"))
    _require(document.get("status") == "FROZEN",
             f"{FROZEN_PROTOCOL_PATH.name} is not marked FROZEN")
    body = document.get("protocol")
    _require(isinstance(body, dict), "the frozen protocol carries no body")
    recorded = document.get("protocol_hash")
    recomputed = canonical_hash(body)
    _require(recorded == recomputed,
             f"frozen protocol body hashes to {recomputed}, file records "
             f"{recorded}")
    return document, body


def expected_fit_count(body: dict[str, Any] | None = None) -> int:
    if body is None:
        return (len(N_GRID) * REPLICATES
                * len(BASE.CANDIDATE_K) * len(BASE.STARTS))
    return (len(body["n_grid"]) * int(body["replicates"])
            * len(body["candidate_k"]) * len(body["starts"]))


def expected_checkpoint_rows(body: dict[str, Any] | None = None) -> int:
    ladder = len(CHECKPOINTS) if body is None else len(body["checkpoints"])
    return expected_fit_count(body) * ladder


# ---------------------------------------------------------------------------
# authorization -- every condition, no bypass
# ---------------------------------------------------------------------------

def authorize_production() -> dict[str, Any]:
    """Decide whether production may start.  Returns a report; never raises.

    Each entry is a named condition with its own verdict, so a refusal says
    which one failed rather than just "no".
    """

    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    # A / B -- an immutable frozen protocol whose body hashes to its record
    body: dict[str, Any] | None = None
    document: dict[str, Any] | None = None
    try:
        document, body = load_frozen_protocol()
        check("A_frozen_protocol_exists", True, str(FROZEN_PROTOCOL_PATH))
        check("B_protocol_body_hash_matches", True,
              str(document["protocol_hash"]))
    except HarnessStop as exc:
        check("A_frozen_protocol_exists", False, str(exc))
        check("B_protocol_body_hash_matches", False, "not evaluated")

    # C / D -- the approved commit, and a clean tree
    approved_sha = None
    if AUTHORIZATION_PATH.exists():
        approval = json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))
        approved_sha = approval.get("approved_run_code_sha")
        if document is not None:
            check("C0_authorization_binds_protocol",
                  approval.get("protocol_hash") == document.get("protocol_hash"),
                  f"authorization records {approval.get('protocol_hash')}")
    else:
        check("C0_authorization_binds_protocol", False,
              f"no authorization record at {AUTHORIZATION_PATH}")
    head = git_head()
    check("C_head_is_the_approved_commit", approved_sha == head and bool(approved_sha),
          f"HEAD={head} approved={approved_sha}")
    check("D_working_tree_clean", working_tree_clean(),
          "uncommitted changes present" if not working_tree_clean() else "")

    # E-I -- the gate receipt
    receipt: dict[str, Any] | None = None
    if GATE_RECEIPT_PATH.exists():
        receipt = json.loads(GATE_RECEIPT_PATH.read_text(encoding="utf-8"))
        check("E_gate_receipt_exists", True, str(GATE_RECEIPT_PATH))
    else:
        check("E_gate_receipt_exists", False,
              f"no gate receipt at {GATE_RECEIPT_PATH}")
    if receipt is not None:
        check("F_gate_verdict_is_pass", receipt.get("verdict") == "PASS",
              str(receipt.get("verdict")))
        check("G_gate_protocol_hash_matches",
              document is not None
              and receipt.get("protocol_hash") == document.get("protocol_hash"),
              f"receipt {receipt.get('protocol_hash')}")
        check("H_gate_code_sha_matches",
              bool(approved_sha) and receipt.get("code_sha") == approved_sha,
              f"receipt {receipt.get('code_sha')}")
        check("I_gate_environment_matches",
              receipt.get("environment_key") == environment_key(
                  environment_fingerprint()),
              "environment differs from the gate; this is a refusal, not a "
              "reason to relax a tolerance")
    else:
        for name in ("F_gate_verdict_is_pass", "G_gate_protocol_hash_matches",
                     "H_gate_code_sha_matches", "I_gate_environment_matches"):
            check(name, False, "no receipt")

    # J -- the baseline artifact is the one the protocol was frozen against
    try:
        baseline = json.loads((BASELINE_RUN_DIR / "protocol.json")
                              .read_text(encoding="utf-8"))
        expected = (body or {}).get("baseline_protocol_hash",
                                    BASELINE_PROTOCOL_HASH)
        check("J_baseline_protocol_hash_matches",
              baseline.get("protocol_hash") == expected,
              f"artifact {baseline.get('protocol_hash')}")
    except Exception as exc:                                   # noqa: BLE001
        check("J_baseline_protocol_hash_matches", False, str(exc))

    failed = [c["check"] for c in checks if not c["ok"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "checks": checks,
        "failed": failed,
        "production_authorization": "NO" if failed else "YES",
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_archive_fits() -> dict[tuple[int, int, int, int, int], dict[str, str]]:
    with (BASELINE_RUN_DIR / "fit_results.csv").open(
            "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(int(r["k_true"]), int(r["n"]), int(r["replicate"]),
             int(r["k_est"]), int(r["start"])): r for r in rows}


def read_archive_masks() -> dict[tuple[int, int, int], tuple[str, str]]:
    with (BASELINE_RUN_DIR / "generator_provenance.csv").open(
            "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(int(r["K_TRUE"]), int(r["n"]), int(r["replicate"])):
            (r["train_mask_hash"], r["test_mask_hash"]) for r in rows}


@dataclass(frozen=True)
class Cell:
    k_true: int
    n: int
    replicate: int


def build_cells(n_values=None, replicates: int = REPLICATES) -> list[Cell]:
    n_values = N_GRID if n_values is None else tuple(n_values)
    return [Cell(K_TRUE, n, r)
            for n in n_values for r in range(1, replicates + 1)]


def _state_digest(z_est: np.ndarray, f: np.ndarray, w0: float, w: float) -> str:
    """Order-fixed digest of the checkpoint state.

    A digest is evidence that the same state was reached, not a defence against
    tampering: anyone who edits the state can recompute it.  The actual defence
    is the archived state itself plus the checkpoint-8 comparison against the
    baseline artifact.
    """

    h = hashlib.sha256()
    for array in (np.ascontiguousarray(z_est, dtype=np.float64),
                  np.ascontiguousarray(f, dtype=np.float64)):
        h.update(array.tobytes())
        h.update(str(array.shape).encode())
    h.update(np.float64(w0).tobytes())
    h.update(np.float64(w).tobytes())
    return h.hexdigest()


CHECKPOINT_COLUMNS = [
    "fit_index", "k_true", "n", "replicate", "k_est", "start",
    "data_seed", "split_seed", "model_seed",
    "train_mask_hash", "test_mask_hash",
    "checkpoint_iteration",
    "heldout_mean_log_score", "q_strict", "s2_q_based", "s3_plugin_conditional",
    "num_params",
    *DIAGNOSTIC_KEYS,
    *(f"max_{key}_last4" for key in DIAGNOSTIC_KEYS),
    *(f"budget_stable_gram_{tau:g}" for tau in TAU_LADDER),
    *(f"budget_stable_eta_y_{tau:g}" for tau in TAU_LADDER),
    "w0_hat", "w_hat",
    "cumulative_runtime_s", "nan_iterations_so_far", "attempt_index",
    "checkpoint_state_digest", "state_key",
]

FIT_COLUMNS = [
    "fit_index", "k_true", "n", "replicate", "k_est", "start",
    "data_seed", "split_seed", "model_seed",
    "max_iter", "actual_iterations", "stopping_reason",
    "retry_count", "retry_seed", "nan_detected", "attempts_used",
    "retry_events", "runtime_s",
    "train_mask_hash", "test_mask_hash",
]


def make_checkpoint_callback(*, identity, dataset, y_test, test_rows, test_cols,
                             collected, states=None):
    """Build the callback production actually uses.

    It is a module-level factory rather than a closure inside the fit loop so
    that the invariance evidence can exercise THIS function and not a stand-in
    written for the test (finding H-02).  It scores one checkpoint: it is
    deterministic, consumes no randomness, and every array it receives is a
    read-only copy, so it cannot write through into the estimation path.  The
    held-out score it computes is used for SELECTION only -- no branch here can
    shorten or extend a run.
    """

    def on_checkpoint(record: dict[str, Any]) -> None:
        z_est = record["Z_est"]
        w0_hat, w_hat = record["w0"], record["w"]
        eta = w0_hat + w_hat * (z_est @ z_est.T)
        eta_test = eta[test_rows, test_cols]
        _require(bool(np.all(np.isfinite(eta_test))),
                 f"non-finite held-out eta at fit {identity['fit_index']} "
                 f"checkpoint {record['iteration']}")
        score = BASE.heldout_bernoulli_mean_log_score(y_test, eta_test)
        s3 = BASE.s3_plugin_conditional(
            record["model"], dataset.X, dataset.Y, z_est, record["F"],
            w0_hat, w_hat, record["num_params"], identity["n"])
        _require(math.isfinite(score) and math.isfinite(record["bic"])
                 and math.isfinite(s3),
                 f"non-finite criterion at fit {identity['fit_index']} "
                 f"checkpoint {record['iteration']}")
        state_key = f"{identity['fit_index']:04d}_it{record['iteration']:03d}"
        row = dict(identity)
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
            "state_key": state_key,
        })
        for key in DIAGNOSTIC_KEYS:
            row[key] = record[key]
            row[f"max_{key}_last4"] = record[f"max_{key}_last4"]
            if not math.isfinite(float(record[key])):
                raise HarnessStop(
                    f"non-finite diagnostic {key} at fit "
                    f"{identity['fit_index']} checkpoint {record['iteration']}")
        for tau in TAU_LADDER:
            row[f"budget_stable_gram_{tau:g}"] = bool(
                record["max_gram_rel_change_last4"] < tau)
            row[f"budget_stable_eta_y_{tau:g}"] = bool(
                record["max_eta_y_rel_change_last4"] < tau)
        collected.append(row)
        if states is not None:
            states[f"{state_key}_Z"] = np.asarray(z_est)
            states[f"{state_key}_F"] = np.asarray(record["F"])
            states[f"{state_key}_wpair"] = np.array([w0_hat, w_hat],
                                                    dtype=np.float64)

    return on_checkpoint


# ---------------------------------------------------------------------------
# the fit loop
# ---------------------------------------------------------------------------

def _run_cells(cells: list[Cell], *, max_iter: int, checkpoints: tuple[int, ...],
               archive_check: bool, baseline_checkpoint: int,
               states: dict[str, np.ndarray] | None = None,
               progress_every: int = 25) -> tuple[list[dict], list[dict]]:
    """Run every (cell, K, start) fit.

    ``archive_check`` compares each fit's baseline-checkpoint row against the
    archived 8-iteration fit AS SOON AS the fit finishes, and raises on the
    first mismatch.  Production sets it: a single disagreement anywhere aborts
    the attempt, because a run whose early checkpoints do not reproduce the
    baseline is not a budget comparison at all, and salvaging the fits that did
    match would be selecting the agreeing subset after the fact.

    The EM module is imported HERE, not at module scope: the Phase 8b suite
    asserts that no research_audit test pulls a real EM runner into the test
    process.
    """

    from em_runner_instrumented import (                       # noqa: PLC0415
        RetryDetected, run_em_instrumented,
    )

    archive = read_archive_fits() if archive_check else {}
    archive_masks = read_archive_masks() if archive_check else {}
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

        if archive_check:
            want = archive_masks.get((cell.k_true, cell.n, cell.replicate))
            _require(want == (train_hash, test_hash),
                     f"mask hashes for K_TRUE={cell.k_true} n={cell.n} "
                     f"replicate={cell.replicate} differ from the archive")

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
                    "train_mask_hash": train_hash, "test_mask_hash": test_hash,
                }
                collected: list[dict[str, Any]] = []
                on_checkpoint = make_checkpoint_callback(
                    identity=identity, dataset=dataset, y_test=y_test,
                    test_rows=test_rows, test_cols=test_cols,
                    collected=collected, states=states)

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
                        f"so this attempt has FAILED") from exc
                elapsed = time.perf_counter() - t0

                _require(len(collected) == len(checkpoints),
                         f"fit {index} produced {len(collected)} checkpoints, "
                         f"expected {len(checkpoints)}")
                _require(result["stopping_reason"] == "fixed_budget",
                         f"fit {index} stopped for {result['stopping_reason']!r}")
                _require(result["actual_iterations"] == max_iter,
                         f"fit {index} ran {result['actual_iterations']} of "
                         f"{max_iter} iterations")
                _require(result["retry_count"] == 0,
                         f"fit {index} recorded {result['retry_count']} retries")

                if archive_check:
                    row = next(r for r in collected
                               if r["checkpoint_iteration"] == baseline_checkpoint)
                    mismatches = _compare_row_to_archive(row, archive)
                    _require(not mismatches,
                             f"fit {index} does not reproduce the archived "
                             f"baseline at checkpoint {baseline_checkpoint}: "
                             f"{mismatches}")

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
                })

                if index % progress_every == 0:
                    print(f"  fit {index} K_TRUE={cell.k_true} n={cell.n} "
                          f"r={cell.replicate} K={k_est} start={start} "
                          f"{elapsed:.1f}s", flush=True)

    return checkpoint_rows, fit_rows


def _compare_row_to_archive(row: dict[str, Any],
                            archive: dict[tuple[int, int, int, int, int],
                                          dict[str, str]]) -> list[str]:
    key = (int(row["k_true"]), int(row["n"]), int(row["replicate"]),
           int(row["k_est"]), int(row["start"]))
    reference = archive.get(key)
    if reference is None:
        return [f"{key}: absent from the archive"]
    out: list[str] = []
    for field in REPLICATION_FIELDS:
        want = (int(reference[field]) if field in REPLICATION_INT_FIELDS
                else float(reference[field]))
        got = (int(row[field]) if field in REPLICATION_INT_FIELDS
               else float(row[field]))
        if want != got:
            out.append(f"{field}: archive {want!r} vs new {got!r}")
    return out


def compare_to_archive(rows: list[dict[str, Any]],
                       baseline_checkpoint: int = BASELINE_CHECKPOINT
                       ) -> dict[str, Any]:
    """Compare every baseline-checkpoint row to the archive.  EXACT."""

    archive = read_archive_fits()
    mismatches: list[dict[str, Any]] = []
    per_fit: list[dict[str, Any]] = []
    for row in rows:
        if int(row["checkpoint_iteration"]) != baseline_checkpoint:
            continue
        problems = _compare_row_to_archive(row, archive)
        per_fit.append({"fit_index": row["fit_index"],
                        "k_est": row["k_est"], "start": row["start"],
                        "match": not problems, "problems": problems})
        for problem in problems:
            mismatches.append({"fit_index": row["fit_index"], "problem": problem})
    return {"compared_rows": len(per_fit), "per_fit": per_fit,
            "mismatches": mismatches,
            "match": not mismatches and len(per_fit) > 0}


# ---------------------------------------------------------------------------
# replication gate
# ---------------------------------------------------------------------------

def run_replication_gate(*, allow_em: bool) -> dict[str, Any]:
    """Run the gate cell and WRITE A RECEIPT.

    The verdict is a committed artifact, not a line of stdout: production takes
    the receipt as a required input, so a PASS that exists only in a terminal
    scrollback cannot authorise anything.
    """

    _require(allow_em, "the replication gate runs EM; pass --allow-em")
    document, body = load_frozen_protocol()

    started = time.time()
    fingerprint = environment_fingerprint()
    code_sha = git_head()
    tree_clean = working_tree_clean()

    rows, fit_rows = _run_cells(
        [Cell(K_TRUE, int(body["gate_cell"]["n"]),
              int(body["gate_cell"]["replicate"]))],
        max_iter=int(body["baseline_checkpoint"]),
        checkpoints=(int(body["baseline_checkpoint"]),),
        archive_check=False,
        baseline_checkpoint=int(body["baseline_checkpoint"]),
        progress_every=7)
    comparison = compare_to_archive(rows, int(body["baseline_checkpoint"]))
    masks_ok = _gate_masks_match(fit_rows)
    retries = [r for r in fit_rows if int(r["retry_count"]) != 0]
    nan_flagged = [r for r in fit_rows if r["nan_detected"]]

    verdict = ("PASS" if comparison["match"] and masks_ok
               and len(fit_rows) == int(body["gate_expected_fits"])
               and not retries and not nan_flagged and tree_clean
               else "FAIL")
    receipt = {
        "receipt_version": "em-convergence-gate-receipt-v1",
        "experiment_id": body["experiment_id"],
        "protocol_hash": document["protocol_hash"],
        "code_sha": code_sha,
        "working_tree_clean": tree_clean,
        "baseline_run": body["baseline_run"],
        "baseline_protocol_hash": body["baseline_protocol_hash"],
        "environment": fingerprint,
        "environment_key": environment_key(fingerprint),
        "gate_cell": body["gate_cell"],
        "gate_cell_selection_rule": (
            "the first replicate of the smallest scoped n, derived as "
            "min(N_GRID); fixed by rule, not chosen by inspection"),
        "expected_fits": int(body["gate_expected_fits"]),
        "actual_fits": len(fit_rows),
        "compared_fields": list(REPLICATION_FIELDS)
                           + ["train_mask_hash", "test_mask_hash"],
        "agreement_rule": "EXACT equality; no tolerance is defined",
        "per_fit": comparison["per_fit"],
        "mismatch_count": len(comparison["mismatches"]),
        "mismatches": comparison["mismatches"],
        "mask_hashes_match": masks_ok,
        "retry_count_total": sum(int(r["retry_count"]) for r in fit_rows),
        "nan_detected_fits": len(nan_flagged),
        "wall_clock_seconds": round(time.time() - started, 1),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "production_authorization_input": True,
    }
    GATE_RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(GATE_RECEIPT_PATH, receipt)
    print(json.dumps({"verdict": verdict, "receipt": str(GATE_RECEIPT_PATH),
                      "mismatch_count": receipt["mismatch_count"]}, indent=2))
    return receipt


def _gate_masks_match(fit_rows: list[dict[str, Any]]) -> bool:
    archive = read_archive_masks()
    for row in fit_rows:
        key = (int(row["k_true"]), int(row["n"]), int(row["replicate"]))
        if archive.get(key) != (row["train_mask_hash"], row["test_mask_hash"]):
            return False
    return True


# ---------------------------------------------------------------------------
# production
# ---------------------------------------------------------------------------

def run_production(*, allow_em: bool, confirm: bool) -> dict[str, Any]:
    """One attempt.  SUCCESS or FAILED -- there is no third outcome and no resume."""

    _require(allow_em, "production requires --allow-em")
    _require(confirm, "production requires --confirm-em-convergence-sweep")

    decision = authorize_production()
    _require(decision["production_authorization"] == "YES",
             "PRODUCTION AUTHORIZATION = NO; failed conditions: "
             + ", ".join(decision["failed"]))
    document, body = load_frozen_protocol()

    _require(not ARTIFACT_DIR.exists(),
             f"artifact directory already exists; refusing to overwrite or "
             f"resume: {ARTIFACT_DIR}")

    attempt_id = f"{EXPERIMENT_ID}-{uuid.uuid4().hex[:12]}"
    started = time.time()
    fingerprint = environment_fingerprint()
    run_code_sha = git_head()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)

    # The artifact carries the frozen protocol's EXACT BYTES, so an auditor can
    # compare the artifact copy with the freeze-commit file directly.
    (ARTIFACT_DIR / "protocol.json").write_bytes(
        FROZEN_PROTOCOL_PATH.read_bytes())
    write_json(ARTIFACT_DIR / "attempt.json", {
        "attempt_id": attempt_id, "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "policy": ("a FAILED attempt is never resumed or completed under the "
                   "same attempt_id; a re-run needs a new attempt_id and new "
                   "human authorization"),
    })
    write_json(ARTIFACT_DIR / "authorization_check.json", decision)

    states: dict[str, np.ndarray] = {}
    try:
        rows, fit_rows = _run_cells(
            build_cells(body["n_grid"], int(body["replicates"])),
            max_iter=int(body["max_iter"]),
            checkpoints=tuple(int(c) for c in body["checkpoints"]),
            archive_check=True,
            baseline_checkpoint=int(body["baseline_checkpoint"]),
            states=states)
        _require(len(fit_rows) == expected_fit_count(body),
                 f"{len(fit_rows)} fits, expected {expected_fit_count(body)}")
        _require(len(rows) == expected_checkpoint_rows(body),
                 f"{len(rows)} checkpoint rows, expected "
                 f"{expected_checkpoint_rows(body)}")
        replication = compare_to_archive(rows, int(body["baseline_checkpoint"]))
        _require(replication["match"]
                 and replication["compared_rows"] == len(fit_rows),
                 "the all-fit checkpoint-8 comparison did not match the archive")
    except BaseException as exc:                               # noqa: BLE001
        write_json(ARTIFACT_DIR / "attempt.json", {
            "attempt_id": attempt_id, "status": "FAILED",
            "failure": f"{type(exc).__name__}: {exc}",
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_clock_seconds": round(time.time() - started, 1),
            "policy": ("no retry, no replacement, no resume, no seed change, "
                       "no tolerance change. Partially completed fits are NOT "
                       "analysable evidence. A re-run requires a new "
                       "attempt_id and new human authorization."),
        })
        write_json(ARTIFACT_DIR / "failure.json", {
            "attempt_id": attempt_id,
            "failure": f"{type(exc).__name__}: {exc}",
        })
        raise

    write_csv(ARTIFACT_DIR / "checkpoint_results.csv", rows, CHECKPOINT_COLUMNS)
    write_csv(ARTIFACT_DIR / "fit_results.csv", fit_rows, FIT_COLUMNS)
    np.savez_compressed(ARTIFACT_DIR / "checkpoint_states.npz", **states)
    write_json(ARTIFACT_DIR / "replication_vs_baseline.json", {
        "baseline_run": body["baseline_run"],
        "baseline_protocol_hash": body["baseline_protocol_hash"],
        "checkpoint": int(body["baseline_checkpoint"]),
        "scope": "ALL fits, not only the gate cell",
        "compared_rows": replication["compared_rows"],
        "mismatch_count": len(replication["mismatches"]),
        "match": replication["match"],
    })

    elapsed = round(time.time() - started, 1)
    write_json(ARTIFACT_DIR / "runinfo.json", {
        "attempt_id": attempt_id,
        "experiment_id": body["experiment_id"],
        "protocol_hash": document["protocol_hash"],
        "runner_version": "em-convergence-runner-v2",
        "attempted_fit_count": len(fit_rows),
        "completed_fit_count": len(fit_rows),
        "checkpoint_row_count": len(rows),
        "expected_fit_count": expected_fit_count(body),
        "expected_checkpoint_rows": expected_checkpoint_rows(body),
        "fits_with_retries": sum(1 for r in fit_rows if int(r["retry_count"])),
        "adaptive_stopping": False,
        "stopping_reasons": sorted({r["stopping_reason"] for r in fit_rows}),
        "wall_clock_seconds": elapsed,
        "run_code_sha": run_code_sha,
        "gate_receipt_protocol_hash": json.loads(
            GATE_RECEIPT_PATH.read_text(encoding="utf-8"))["protocol_hash"],
        "environment": fingerprint,
        "environment_key": environment_key(fingerprint),
        "working_tree_clean_before": True,
        "working_tree_clean_after": working_tree_clean(),
        "notes": ("counters are policy declarations, not tallies: this runner "
                  "has no retry, replacement or resume path, and a NaN restart "
                  "raises instead of restarting"),
    })
    write_json(ARTIFACT_DIR / "attempt.json", {
        "attempt_id": attempt_id, "status": "SUCCESS",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": elapsed,
        "fits": len(fit_rows), "checkpoint_rows": len(rows),
    })
    print(json.dumps({"status": "SUCCESS", "attempt_id": attempt_id,
                      "fits": len(fit_rows), "checkpoint_rows": len(rows),
                      "wall_clock_seconds": elapsed}, indent=2))
    return {"attempt_id": attempt_id, "fits": len(fit_rows),
            "checkpoint_rows": len(rows)}


# ---------------------------------------------------------------------------
# planning (no EM)
# ---------------------------------------------------------------------------

def plan(n_values=None) -> dict[str, Any]:
    """Scope, runtime and storage projection from the ARCHIVE only.  No EM."""

    n_values = N_GRID if n_values is None else tuple(n_values)
    with (BASELINE_RUN_DIR / "fit_results.csv").open(
            "r", encoding="utf-8", newline="") as handle:
        archive = list(csv.DictReader(handle))
    per_n = {n: sum(float(r["runtime_s"]) for r in archive
                    if int(r["k_true"]) == K_TRUE and int(r["n"]) == n)
             for n in n_values}
    fits = len(n_values) * REPLICATES * len(BASE.CANDIDATE_K) * len(BASE.STARTS)
    seconds = sum(per_n.values()) * MAX_ITER / BASELINE_CHECKPOINT

    # Archived state: Z (n x K) and F (d x K) per checkpoint, float64.
    floats = sum(len(CHECKPOINTS) * len(BASE.STARTS) * REPLICATES
                 * (n + BASE.D_FEATURES) * k
                 for n in n_values for k in BASE.CANDIDATE_K)
    return {
        "n_grid": list(n_values),
        "fits": fits,
        "checkpoint_rows": fits * len(CHECKPOINTS),
        "baseline_seconds_at_8_iterations": {n: round(v, 1)
                                             for n, v in per_n.items()},
        "projected_hours_at_max_iter": round(seconds / 3600, 2),
        "projection_note": (
            "linear in iterations. It ignores two opposing effects and is "
            "therefore an ESTIMATE, not a bound: it counts the archive's fixed "
            "end-of-run Q/BIC/S3 cost as if it scaled with iterations (over-"
            "estimate), while the new run pays that cost six times per fit "
            "instead of once (under-estimate). The operational guard is the "
            "2x abort threshold, which is a safety stop and NOT a scientific "
            "inclusion criterion."),
        "archived_state_floats": floats,
        "archived_state_megabytes_uncompressed": round(floats * 8 / 1048576, 2),
        "protocol_hash_of_constants": canonical_hash(protocol_body_from_constants()),
    }


def emit_draft() -> Path:
    """Write the DRAFT protocol from the constants.  Never writes the frozen file."""

    DRAFT_PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = protocol_body_from_constants()
    document = json.loads(DRAFT_PROTOCOL_PATH.read_text(encoding="utf-8")) \
        if DRAFT_PROTOCOL_PATH.exists() else {}
    document.update({
        "status": "DRAFT",
        "status_note": (
            "NOT FROZEN. The body below is generated from the runner's "
            "constants by --emit-draft, so the draft and the runner cannot "
            "drift. It carries no protocol_hash on purpose: a hash would look "
            "like a freeze. Freezing writes "
            f"{FROZEN_PROTOCOL_PATH.name} with status FROZEN and its hash, in "
            "its own commit, after human authorization under CLAUDE.md "
            "section 6."),
        "experiment_id": EXPERIMENT_ID,
        "design_document": ("reports/k_selection/"
                            "em_convergence_sensitivity_design_20260905.md"),
        "protocol": body,
        "would_freeze_to_hash": canonical_hash(body),
        "expected_long_runs": expected_fit_count(),
        "expected_checkpoint_rows": expected_checkpoint_rows(),
        "artifact_schema": {
            "checkpoint_results.csv": CHECKPOINT_COLUMNS,
            "fit_results.csv": FIT_COLUMNS,
            "checkpoint_states.npz": (
                "per checkpoint: <state_key>_Z, <state_key>_F, "
                "<state_key>_wpair, so the auditor can recompute every "
                "checkpoint's scores independently"),
        },
        "production_status": {
            "production_runs_so_far": 0,
            "new_production_fits_so_far": 0,
            "replication_gate_runs_so_far": 0,
            "authorisation": "NOT AUTHORISED",
        },
    })
    write_json(DRAFT_PROTOCOL_PATH, document)
    return DRAFT_PROTOCOL_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--plan-n", type=int, nargs="+", default=None)
    parser.add_argument("--emit-draft", action="store_true")
    parser.add_argument("--authorize", action="store_true",
                        help="report the production authorization decision")
    parser.add_argument("--replication-gate", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--allow-em", action="store_true")
    parser.add_argument("--confirm-em-convergence-sweep", action="store_true")
    args = parser.parse_args(argv)

    if args.emit_draft:
        print(json.dumps({"written": str(emit_draft())}, indent=2))
        return 0
    if args.plan or args.plan_n:
        print(json.dumps(plan(args.plan_n), indent=2, sort_keys=True))
        return 0
    if args.authorize:
        decision = authorize_production()
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0 if decision["production_authorization"] == "YES" else 1
    if args.replication_gate:
        receipt = run_replication_gate(allow_em=args.allow_em)
        return 0 if receipt["verdict"] == "PASS" else 1
    if args.full:
        run_production(allow_em=args.allow_em,
                       confirm=args.confirm_em_convergence_sweep)
        return 0
    parser.error("choose --plan, --emit-draft, --authorize, "
                 "--replication-gate or --full")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
