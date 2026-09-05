"""EM convergence-sensitivity sweep -- runner.

SCIENTIFIC ESTIMAND (deliberately narrow)
-----------------------------------------
Under identical data, masks, initialization and first-eight-iteration RNG
prefix, how does continuing the same L=5 stochastic MCEM procedure from
iteration 8 to iteration 40 change selected K?

Continuing the procedure draws fresh Monte-Carlo latent samples at every
further iteration, so the contrast mixes additional optimization with
additional Monte-Carlo variation and the two are not separable by this design.
These claims are FORBIDDEN in any report built on this experiment:

  * a deterministic optimization effect
  * "8 iterations caused the under-selection"
  * "the criterion is exonerated"
  * convergence, of any kind
  * consistency

WHERE THE SPECIFICATION LIVES
-----------------------------
There is exactly one: the ``protocol`` object of the protocol file under
``reports/k_selection/``.  Production reads every execution-relevant field
from it -- families, d, L, n grid, replicates, candidate K, starts, seeds,
split rule, generator settings, signal parameters, the selector and the tie
rule -- and additionally verifies each against an explicit runner contract
before any EM starts, so a protocol file that quietly changed scope is refused
rather than obeyed.  The hash of the body that was actually used is recorded
in the artifact.

REPLICATION GATE
----------------
The one pre-run check that matters scientifically: run the gate cell to
iteration 8 and require it to reproduce the archived
``clean_true_k_asymptotics_20260904`` fits EXACTLY.  If it does not, the
40-iteration run is not a continuation of the archived 8-iteration run, and
the whole comparison is void.  Production repeats the same comparison for
EVERY fit inside the checkpoint callback, so a fit whose first eight
iterations are not the archived ones never reaches iteration 9.

Usage
-----
    python tools/research_audit/run_em_convergence_sensitivity.py --plan
    python tools/research_audit/run_em_convergence_sensitivity.py --emit-draft
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
import os
import platform
import subprocess
import sys
import time
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
# lazily inside its production function, and so does this module.
import run_clean_true_k_sweep as BASE                        # noqa: E402

EXPERIMENT_ID = "em_convergence_sensitivity_20260905"
PROTOCOL_VERSION = "em-convergence-protocol-draft-3"
ARTIFACT_VERSION = "em-convergence-artifact-v3"
RUNNER_CONTRACT_VERSION = "em-convergence-runner-contract-v1"

BASELINE_RUN_DIR = (ROOT / "expfam" / "results" / "k_selection"
                    / "clean_true_k_asymptotics_20260904")
BASELINE_PROTOCOL_HASH = ("547880a16aef6530cfdf7903c4e32f16062397e0"
                          "bacc0c109d5c77fb9892ccc0")

DESIGN_DIR = ROOT / "reports" / "k_selection"
DRAFT_PROTOCOL_PATH = DESIGN_DIR / f"{EXPERIMENT_ID}_protocol.DRAFT.json"
FROZEN_PROTOCOL_PATH = DESIGN_DIR / f"{EXPERIMENT_ID}_protocol.json"

#: Where the replication gate leaves its result for a human to read.  Nothing
#: in this file reads it back: production repeats the same comparison itself.
RUNTIME_ROOT = ROOT / ".local" / EXPERIMENT_ID

#: Final evidence location, and the directory the run writes into until it
#: finishes.  The staging directory is a sibling so the last step is a single
#: rename on the same filesystem: a run that dies part-way leaves ``.partial``
#: behind and never a half-written directory that looks like a finished one.
#: Cleaning up after a failure is a human's job, deliberately.
ARTIFACT_DIR = ROOT / "expfam" / "results" / "k_selection" / EXPERIMENT_ID
STAGING_DIR = ARTIFACT_DIR.with_name(ARTIFACT_DIR.name + ".partial")

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
    """Refusal.  Raised before or during a run; never swallowed."""


class PrefixReplicationError(RuntimeError):
    """The baseline checkpoint did not reproduce the archived fit.

    Raised from inside the checkpoint callback, so the EM loop unwinds before
    the next iteration begins: a fit whose first eight iterations are not the
    archived ones is not a budget comparison, and continuing it would only
    spend hours producing evidence that has to be thrown away.
    """

    def __init__(self, fit_index: int, iteration: int, problems: list[str]):
        super().__init__(
            f"fit {fit_index} does not reproduce the archived baseline at "
            f"iteration {iteration}: {problems}")
        self.fit_index = fit_index
        self.iteration = iteration
        self.problems = problems


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessStop(message)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def _git(*args: str) -> tuple[int, str]:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return out.returncode, out.stdout.strip()


def git_head() -> str:
    code, value = _git("rev-parse", "HEAD")
    return value if code == 0 else "unknown"


def working_tree_clean() -> bool:
    code, value = _git("status", "--porcelain")
    return code == 0 and value == ""


def _blas_identity() -> dict[str, str]:
    """A normalized BLAS/LAPACK identity, or an explicit unavailability marker.

    Normalized because the raw dict carries absolute paths and build flags that
    differ between machines without saying anything about numerics.
    """

    try:
        config = np.show_config(mode="dicts")            # type: ignore[call-arg]
    except Exception:                                     # noqa: BLE001
        return {"status": "unavailable"}
    out: dict[str, str] = {}
    for section in ("Build Dependencies", "build_dependencies"):
        block = config.get(section) or {}
        for name, info in block.items():
            if isinstance(info, dict):
                out[str(name)] = f"{info.get('name', '?')}/{info.get('version', '?')}"
    return out or {"status": "unavailable"}


def environment_fingerprint() -> dict[str, Any]:
    """What the numbers were produced on.

    Recorded rather than reconstructed: KI-014 exists because reconstructed
    provenance is a guess wearing the costume of a measurement.  Only stable
    numerical identity goes in; timestamps and paths do not.
    """

    try:
        import scipy                                       # noqa: PLC0415
        scipy_version = scipy.__version__
    except Exception:                                       # noqa: BLE001
        scipy_version = "absent"
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy_version,
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "blas": _blas_identity(),
        "threads": {name: os.environ.get(name)
                    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                                 "VECLIB_MAXIMUM_THREADS")},
    }


def environment_key(fingerprint: dict[str, Any]) -> str:
    """Canonical key over the numerically relevant fields.

    Recorded so a later run can be compared with this one at a glance;
    timestamps and local paths are excluded because they differ between runs
    without saying anything about the numbers.
    """

    return json.dumps({key: fingerprint.get(key)
                       for key in ("python", "numpy", "scipy", "platform",
                                   "machine", "blas", "threads")},
                      sort_keys=True, separators=(",", ":"))


def canonical_hash(body: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the specification -- ONE object, read from the frozen protocol
# ---------------------------------------------------------------------------

#: Every execution-relevant field, with the value the runner contract expects.
#: Production reads the value from the frozen protocol AND checks it here, so a
#: frozen file that quietly changed scope is refused rather than obeyed.
RUNNER_CONTRACT: dict[str, Any] = {
    "experiment_id": EXPERIMENT_ID,
    "k_true": 5,
    "n_grid": [50, 75, 100, 150],
    "replicates": 8,
    "candidate_k": [1, 2, 3, 4, 5, 6, 7],
    "starts": [1, 2],
    "family_x": "poisson",
    "family_y": "bernoulli",
    "d": 15,
    "L": 5,
    "max_iter": 40,
    "checkpoints": [8, 12, 16, 24, 32, 40],
    "baseline_checkpoint": 8,
    "primary_checkpoint": 40,
    "adaptive_stopping": False,
    "stopping_reason_expected": "fixed_budget",
    "on_nan": "fail",
    "stability_window": 4,
    "tau_ladder": [0.01, 0.001, 0.0001],
    "test_ratio": 0.20,
    "tie_tolerance": 1e-12,
    "tie_rule": "smallest_k_among_ties",
    "selector": "unweighted_mean_over_starts_then_argmax_in_selector_space",
    "split_rule": "deterministic_dyad_holdout_from_split_seed",
    "numerics_mode": "consistent",
    "generator_version": "canonical-clean-v1",
    "data_seed_base": 810000,
    "split_seed_base": 820000,
    "model_seed_base": 830000,
    "data_seed_formula": "data_seed_base + k_true*10000 + n*10 + replicate",
    "split_seed_formula": "split_seed_base + k_true*10000 + n*10 + replicate",
    "model_seed_formula": ("model_seed_base + k_true*100000 + n*1000 + "
                           "replicate*100 + k_est*10 + start"),
    "w0_true": -1.0,
    "w_ref": 1.0,
    "k_ref": 3,
    "row_norm_sq_target": 0.5,
    "poisson_lambda_max": 1.0e6,
    "baseline_run": BASELINE_RUN_DIR.name,
    "baseline_protocol_hash": BASELINE_PROTOCOL_HASH,
    "all_fit_checkpoint8_replication": True,
    "runtime_abort_rule": "none",
}

EXPECTED_LONG_RUNS = (len(RUNNER_CONTRACT["n_grid"])
                      * RUNNER_CONTRACT["replicates"]
                      * len(RUNNER_CONTRACT["candidate_k"])
                      * len(RUNNER_CONTRACT["starts"]))
EXPECTED_CHECKPOINT_ROWS = EXPECTED_LONG_RUNS * len(RUNNER_CONTRACT["checkpoints"])
EXPECTED_ITERATION_ROWS = EXPECTED_LONG_RUNS * RUNNER_CONTRACT["max_iter"]
GATE_N = min(RUNNER_CONTRACT["n_grid"])
GATE_REPLICATE = 1
GATE_EXPECTED_FITS = (len(RUNNER_CONTRACT["candidate_k"])
                      * len(RUNNER_CONTRACT["starts"]))


@dataclass(frozen=True)
class Spec:
    """Execution parameters, all taken from the frozen protocol body."""

    body: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.body[key]

    @property
    def k_true(self) -> int:
        return int(self.body["k_true"])

    @property
    def n_grid(self) -> tuple[int, ...]:
        return tuple(int(n) for n in self.body["n_grid"])

    @property
    def replicates(self) -> int:
        return int(self.body["replicates"])

    @property
    def candidate_k(self) -> tuple[int, ...]:
        return tuple(int(k) for k in self.body["candidate_k"])

    @property
    def starts(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.body["starts"])

    @property
    def checkpoints(self) -> tuple[int, ...]:
        return tuple(int(c) for c in self.body["checkpoints"])

    @property
    def baseline_checkpoint(self) -> int:
        return int(self.body["baseline_checkpoint"])

    def data_seed(self, n: int, replicate: int) -> int:
        return (int(self.body["data_seed_base"]) + self.k_true * 10000
                + n * 10 + replicate)

    def split_seed(self, n: int, replicate: int) -> int:
        return (int(self.body["split_seed_base"]) + self.k_true * 10000
                + n * 10 + replicate)

    def model_seed(self, n: int, replicate: int, k_est: int, start: int) -> int:
        return (int(self.body["model_seed_base"]) + self.k_true * 100000
                + n * 1000 + replicate * 100 + k_est * 10 + start)


def verify_contract(body: dict[str, Any]) -> list[str]:
    """Every contract field must be present and exactly equal.  Returns problems."""

    problems: list[str] = []
    for key, expected in RUNNER_CONTRACT.items():
        if key not in body:
            problems.append(f"{key}: absent from the frozen protocol")
        elif body[key] != expected:
            problems.append(f"{key}: frozen {body[key]!r} != contract {expected!r}")
    return problems


def protocol_body_from_contract() -> dict[str, Any]:
    """Build the DRAFT body.  Production never calls this."""

    body = dict(RUNNER_CONTRACT)
    body.update({
        "protocol_version": PROTOCOL_VERSION,
        "runner_contract_version": RUNNER_CONTRACT_VERSION,
        "estimand": (
            "under identical data, masks, initialization and first-eight-"
            "iteration RNG prefix, how does continuing the same L=5 stochastic "
            "MCEM procedure from iteration 8 to iteration 40 change selected K"),
        "forbidden_claims": [
            "deterministic optimization effect",
            "8 iterations caused the under-selection",
            "the criterion is exonerated",
            "convergence",
            "consistency",
        ],
        "residual_confound": (
            "continuing the procedure redraws L=5 latent samples every further "
            "iteration, so additional optimization and additional Monte-Carlo "
            "variation are not separable by this design"),
        "expected_long_runs": EXPECTED_LONG_RUNS,
        "expected_checkpoint_rows": EXPECTED_CHECKPOINT_ROWS,
        "expected_iteration_rows": EXPECTED_ITERATION_ROWS,
        "gate_cell": {"k_true": RUNNER_CONTRACT["k_true"], "n": GATE_N,
                      "replicate": GATE_REPLICATE},
        "gate_cell_selection_rule": (
            "the first replicate of the smallest scoped n, derived as "
            "min(n_grid); fixed by rule, not chosen by inspection"),
        "gate_expected_fits": GATE_EXPECTED_FITS,
        "replication_fields": list(REPLICATION_FIELDS),
        "diagnostic_keys": list(DIAGNOSTIC_KEYS),
        "criteria": ["S1_heldout_predictive", "S2_q_based",
                     "S3_plugin_conditional"],
        "runtime_estimate_hours": 6.14,
        "runtime_estimate_note": (
            "operational estimate only. There is NO automatic runtime abort: "
            "runtime is not part of the estimand and must never include or "
            "exclude an individual fit. A run that is interrupted produces no "
            "final artifact and is not analysed in part"),
        "primary_outcomes": _primary_outcome_spec(),
    })
    return body


def _primary_outcome_spec() -> dict[str, Any]:
    return {
        "P1": {
            "name": "selected K by budget",
            "definition": (
                "for each (n, replicate) and each checkpoint: average S1 over "
                "the two starts for every candidate K, take the K with the "
                "largest average; tie tolerance 1e-12, ties resolved to the "
                "smallest K"),
            "start_disagreement": (
                "select K INDEPENDENTLY within start 1 and within start 2, "
                "then compare the two selected K. It is NOT 'the raw scores "
                "differ'"),
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
            "replicate_rows_expected": 4 * 8 * 7,
            "summary_rows_expected": 4 * 7,
            "reported": [
                "replicate-level table: n, replicate, K, delta (224 rows)",
                "summary: n, K, mean delta over the 8 replicates (28 rows)",
            ],
            "inference": (
                "DESCRIPTIVE ONLY. No formal test. Starts are NOT replicates. "
                "Spearman between K and the mean delta may be shown as a "
                "diagnostic; the 7 candidate K are not independent samples and "
                "no p-value is reported"),
        },
        "secondary": [
            "winning margin (best minus runner-up in selector space) by checkpoint",
            "start disagreement by checkpoint",
            "budget-stability rates by candidate K and n at every tau",
            "S2 and S3 versions of P1 and P2",
            "cumulative runtime by checkpoint",
        ],
        "wording": (
            "no materiality threshold is defined, so the question says CHANGES "
            "and the word 'materially' is not used"),
    }


def load_protocol() -> tuple[dict[str, Any], Spec]:
    """Read the protocol the run will execute and check it against the contract.

    A frozen file wins over the draft when both exist.  The returned record is
    what production copies into the artifact: the body it actually used, the
    hash of that body, and where it came from.  The hash is always RECOMPUTED
    here rather than read from the file, so the artifact cannot record a hash
    that does not belong to its own protocol body.
    """

    path = (FROZEN_PROTOCOL_PATH if FROZEN_PROTOCOL_PATH.exists()
            else DRAFT_PROTOCOL_PATH)
    _require(path.exists(),
             f"no protocol file at {FROZEN_PROTOCOL_PATH} or "
             f"{DRAFT_PROTOCOL_PATH}; run --emit-draft first")
    document = json.loads(path.read_text(encoding="utf-8"))
    body = document.get("protocol")
    _require(isinstance(body, dict), f"{path.name} carries no protocol body")
    problems = verify_contract(body)
    _require(not problems,
             f"{path.name} does not satisfy the runner contract: "
             + "; ".join(problems))
    record = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_source": path.name,
        "protocol_source_status": document.get("status"),
        "protocol_hash": canonical_hash(body),
        "protocol": body,
    }
    return record, Spec(body)



# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------

def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file and os.replace, so a kill cannot truncate."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ through an open file object.

    ``np.savez_compressed`` APPENDS ``.npz`` to any filename that does not
    already end in it, so handing it ``<name>.npz.tmp`` silently writes
    ``<name>.npz.tmp.npz`` and the rename that follows raises
    FileNotFoundError -- after every fit has already been computed.  Writing
    through an open handle sidesteps the name mangling entirely.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    import io                                              # noqa: PLC0415
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def read_archive_fits() -> dict[tuple[int, int, int, int, int], dict[str, str]]:
    with (BASELINE_RUN_DIR / "fit_results.csv").open(
            "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[tuple[int, int, int, int, int], dict[str, str]] = {}
    for row in rows:
        key = (int(row["k_true"]), int(row["n"]), int(row["replicate"]),
               int(row["k_est"]), int(row["start"]))
        _require(key not in out,
                 f"the archived baseline has a duplicate row for {key}; the "
                 f"comparison target must be unambiguous")
        out[key] = row
    return out


def read_archive_masks() -> dict[tuple[int, int, int], tuple[str, str]]:
    """Mask provenance, with the same uniqueness demand as the fit rows.

    A duplicated key would silently pick one of two mask pairs as "the"
    archived split, and an ambiguous comparison target is not a target.
    """

    with (BASELINE_RUN_DIR / "generator_provenance.csv").open(
            "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[tuple[int, int, int], tuple[str, str]] = {}
    for row in rows:
        key = (int(row["K_TRUE"]), int(row["n"]), int(row["replicate"]))
        _require(key not in out,
                 f"the archived mask provenance has a duplicate row for {key}")
        out[key] = (row["train_mask_hash"], row["test_mask_hash"])
    return out


def _state_digest(z_est: np.ndarray, f: np.ndarray, w0: float, w: float) -> str:
    """Order-fixed digest of the checkpoint state.

    Evidence that the same state was reached, NOT a defence against tampering:
    anyone who edits the state can recompute it.  The defences are the archived
    state itself, the checkpoint-8 comparison, and the auditor's independent
    recomputation of the scores.
    """

    h = hashlib.sha256()
    for array in (np.ascontiguousarray(z_est, dtype=np.float64),
                  np.ascontiguousarray(f, dtype=np.float64)):
        h.update(array.tobytes())
        h.update(str(array.shape).encode())
    h.update(np.float64(w0).tobytes())
    h.update(np.float64(w).tobytes())
    return h.hexdigest()


IDENTITY_COLUMNS = [
    "fit_index", "k_true", "n", "replicate", "k_est", "start",
    "data_seed", "split_seed", "model_seed",
    "train_mask_hash", "test_mask_hash",
]

CHECKPOINT_COLUMNS = [
    *IDENTITY_COLUMNS,
    "checkpoint_iteration",
    "heldout_mean_log_score", "q_strict", "s2_q_based", "s3_plugin_conditional",
    "num_params",
    *DIAGNOSTIC_KEYS,
    *(f"max_{key}_last4" for key in DIAGNOSTIC_KEYS),
    *(f"budget_stable_gram_{tau:g}" for tau in RUNNER_CONTRACT["tau_ladder"]),
    *(f"budget_stable_eta_y_{tau:g}" for tau in RUNNER_CONTRACT["tau_ladder"]),
    "w0_hat", "w_hat", "var_z",
    "cumulative_runtime_s", "nan_iterations_so_far", "attempt_index",
    "checkpoint_state_digest", "state_key",
]

FIT_COLUMNS = [
    *IDENTITY_COLUMNS,
    "max_iter", "actual_iterations", "stopping_reason",
    "retry_count", "retry_seed", "nan_detected", "attempts_used",
    "retry_events", "runtime_s",
]

ITERATION_COLUMNS = [
    "fit_index", "k_true", "n", "replicate", "k_est", "start", "iteration",
    *DIAGNOSTIC_KEYS,
    "nan_iterations_so_far", "attempt_index",
]


# ---------------------------------------------------------------------------
# the production checkpoint callback
# ---------------------------------------------------------------------------

def make_checkpoint_callback(*, identity, dataset, y_test, test_rows, test_cols,
                             collected, spec: Spec, archive=None, states=None):
    """Build the callback production actually uses.

    A module-level factory rather than a closure inside the fit loop so the
    invariance evidence can exercise THIS function and not a stand-in (finding
    H-02).  It scores one checkpoint: deterministic, consumes no randomness,
    and every array it receives is a read-only copy.

    When ``archive`` is supplied it also performs the baseline-checkpoint
    comparison HERE, before returning, and raises
    :class:`PrefixReplicationError` on any difference.  The EM loop therefore
    never begins iteration 9 of a fit whose first eight iterations are not the
    archived ones (finding B2-03).
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
            "w0_hat": w0_hat, "w_hat": w_hat, "var_z": record["var_z"],
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
        for tau in spec["tau_ladder"]:
            row[f"budget_stable_gram_{tau:g}"] = bool(
                record["max_gram_rel_change_last4"] < tau)
            row[f"budget_stable_eta_y_{tau:g}"] = bool(
                record["max_eta_y_rel_change_last4"] < tau)

        if archive is not None and record["iteration"] == spec.baseline_checkpoint:
            problems = compare_row_to_archive(row, archive)
            if problems:
                raise PrefixReplicationError(identity["fit_index"],
                                             record["iteration"], problems)

        collected.append(row)
        if states is not None:
            states[f"{state_key}_Z"] = np.asarray(z_est)
            states[f"{state_key}_Zsamples"] = np.asarray(record["Z_samples"])
            states[f"{state_key}_F"] = np.asarray(record["F"])
            states[f"{state_key}_sigma"] = np.asarray(record["sigma"])
            states[f"{state_key}_scalars"] = np.array(
                [w0_hat, w_hat, record["var_z"]], dtype=np.float64)

    return on_checkpoint


def compare_row_to_archive(row: dict[str, Any],
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
                       baseline_checkpoint: int = 8) -> dict[str, Any]:
    archive = read_archive_fits()
    mismatches: list[dict[str, Any]] = []
    per_fit: list[dict[str, Any]] = []
    for row in rows:
        if int(row["checkpoint_iteration"]) != baseline_checkpoint:
            continue
        problems = compare_row_to_archive(row, archive)
        per_fit.append({"fit_index": row["fit_index"], "k_est": row["k_est"],
                        "start": row["start"], "match": not problems,
                        "problems": problems})
        for problem in problems:
            mismatches.append({"fit_index": row["fit_index"], "problem": problem})
    return {"compared_rows": len(per_fit), "per_fit": per_fit,
            "mismatches": mismatches,
            "match": not mismatches and len(per_fit) > 0}


# ---------------------------------------------------------------------------
# the fit loop
# ---------------------------------------------------------------------------

def _run_cells(cells, *, spec: Spec, max_iter: int, checkpoints: tuple[int, ...],
               archive_check: bool, states=None, iteration_rows=None,
               progress_every: int = 25):
    """Run every (cell, K, start) fit, reading every parameter from ``spec``.

    The EM module is imported HERE, not at module scope: the Phase 8b suite
    asserts that no research_audit test pulls a real EM runner into the test
    process.
    """

    from em_runner_instrumented import (                    # noqa: PLC0415
        RetryDetected, run_em_instrumented,
    )

    archive = read_archive_fits() if archive_check else None
    archive_masks = read_archive_masks() if archive_check else {}
    checkpoint_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    index = 0

    for n, replicate in cells:
        scale = BASE.f_scale_for_row_norm(float(spec["row_norm_sq_target"]),
                                          d=int(spec["d"]), k=spec.k_true)
        w_true = BASE.w_for_matched_y_signal(float(spec["w_ref"]),
                                             k=spec.k_true,
                                             k_ref=int(spec["k_ref"]))
        dataset = BASE.generate_canonical_data(
            n=n, d=int(spec["d"]), k=spec.k_true,
            seed=spec.data_seed(n, replicate),
            family_x=spec["family_x"], family_y=spec["family_y"],
            f_scale=scale, w0=float(spec["w0_true"]), w=w_true,
            poisson_lambda_max=float(spec["poisson_lambda_max"]))
        train, test = BASE.build_masks(n, spec.split_seed(n, replicate))
        test_rows, test_cols = np.where(np.triu(test, 1))
        y_test = dataset.Y[test_rows, test_cols]
        train_hash = BASE.stable_hash(train)
        test_hash = BASE.stable_hash(test)

        if archive_check:
            want = archive_masks.get((spec.k_true, n, replicate))
            _require(want == (train_hash, test_hash),
                     f"mask hashes for n={n} replicate={replicate} differ from "
                     f"the archive")

        for k_est in spec.candidate_k:
            for start in spec.starts:
                index += 1
                seed = spec.model_seed(n, replicate, k_est, start)
                identity = {
                    "fit_index": index, "k_true": spec.k_true, "n": n,
                    "replicate": replicate, "k_est": k_est, "start": start,
                    "data_seed": spec.data_seed(n, replicate),
                    "split_seed": spec.split_seed(n, replicate),
                    "model_seed": seed,
                    "train_mask_hash": train_hash, "test_mask_hash": test_hash,
                }
                collected: list[dict[str, Any]] = []
                callback = make_checkpoint_callback(
                    identity=identity, dataset=dataset, y_test=y_test,
                    test_rows=test_rows, test_cols=test_cols,
                    collected=collected, spec=spec, archive=archive,
                    states=states)

                t0 = time.perf_counter()
                try:
                    result = run_em_instrumented(
                        dataset.X, dataset.Y,
                        family_x=spec["family_x"], family_y=spec["family_y"],
                        k=k_est, L=int(spec["L"]),
                        max_iter=max_iter, checkpoints=checkpoints,
                        seed=seed, train_mask=train,
                        numerics_mode=spec["numerics_mode"],
                        stability_window=int(spec["stability_window"]),
                        on_nan=spec["on_nan"], on_checkpoint=callback)
                except RetryDetected as exc:
                    raise HarnessStop(
                        f"fit {index} requested a NaN restart ({exc}); a restart "
                        f"re-seeds and voids pairing with the archived "
                        f"baseline, so this attempt has FAILED") from exc
                elapsed = time.perf_counter() - t0

                _require(len(collected) == len(checkpoints),
                         f"fit {index} produced {len(collected)} checkpoints, "
                         f"expected {len(checkpoints)}")
                _require(result["stopping_reason"] == spec["stopping_reason_expected"],
                         f"fit {index} stopped for {result['stopping_reason']!r}")
                _require(result["actual_iterations"] == max_iter,
                         f"fit {index} ran {result['actual_iterations']} of "
                         f"{max_iter} iterations")
                _require(result["retry_count"] == 0,
                         f"fit {index} recorded {result['retry_count']} retries")

                checkpoint_rows.extend(collected)
                if iteration_rows is not None:
                    for step, diagnostics in enumerate(
                            result["iteration_diagnostics"], start=1):
                        iteration_rows.append({
                            **{key: identity[key] for key in
                               ("fit_index", "k_true", "n", "replicate",
                                "k_est", "start")},
                            "iteration": step, **diagnostics,
                            "nan_iterations_so_far": 0,
                            "attempt_index": 0,
                        })
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
                    print(f"  fit {index} n={n} r={replicate} K={k_est} "
                          f"start={start} {elapsed:.1f}s", flush=True)

    return checkpoint_rows, fit_rows


def build_cells(spec: Spec):
    return [(n, r) for n in spec.n_grid
            for r in range(1, spec.replicates + 1)]


# ---------------------------------------------------------------------------
# replication gate
# ---------------------------------------------------------------------------

def run_replication_gate(*, allow_em: bool) -> dict[str, Any]:
    """Run the gate cell to iteration 8 and compare it with the archive.

    Fourteen fits (7 candidate K x 2 starts) for the first replicate of the
    smallest scoped n -- a cell fixed by rule, not chosen by inspection.  The
    verdict is what a human reads before committing six hours; the result file
    is a record, and nothing reads it back.
    """

    _require(allow_em, "the replication gate runs EM; pass --allow-em")
    record, spec = load_protocol()

    started = time.time()
    rows, fit_rows = _run_cells(
        [(GATE_N, GATE_REPLICATE)], spec=spec,
        max_iter=spec.baseline_checkpoint,
        checkpoints=(spec.baseline_checkpoint,),
        archive_check=False, progress_every=7)
    comparison = compare_to_archive(rows, spec.baseline_checkpoint)
    masks_ok = _gate_masks_match(fit_rows)
    passed = (comparison["match"] and masks_ok
              and len(fit_rows) == GATE_EXPECTED_FITS)

    result = {
        "verdict": "PASS" if passed else "FAIL",
        "experiment_id": EXPERIMENT_ID,
        "protocol_hash": record["protocol_hash"],
        "protocol_source": record["protocol_source"],
        "code_sha": git_head(),
        "working_tree_clean": working_tree_clean(),
        "baseline_run": spec["baseline_run"],
        "baseline_protocol_hash": spec["baseline_protocol_hash"],
        "environment": environment_fingerprint(),
        "gate_cell": {"k_true": spec.k_true, "n": GATE_N,
                      "replicate": GATE_REPLICATE},
        "expected_fits": GATE_EXPECTED_FITS,
        "actual_fits": len(fit_rows),
        "compared_fields": list(REPLICATION_FIELDS)
                           + ["train_mask_hash", "test_mask_hash"],
        "agreement_rule": "EXACT equality; no tolerance is defined",
        "per_fit": comparison["per_fit"],
        "mismatch_count": len(comparison["mismatches"]),
        "mismatches": comparison["mismatches"],
        "mask_hashes_match": masks_ok,
        "retry_count_total": sum(int(r["retry_count"]) for r in fit_rows),
        "nan_detected_fits": sum(1 for r in fit_rows if r["nan_detected"]),
        "wall_clock_seconds": round(time.time() - started, 1),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(RUNTIME_ROOT / "replication_gate.json", result)
    print(json.dumps({"verdict": result["verdict"],
                      "mismatch_count": result["mismatch_count"],
                      "mask_hashes_match": masks_ok,
                      "result_file": str(RUNTIME_ROOT / "replication_gate.json")},
                     indent=2))
    return result


def _gate_masks_match(fit_rows) -> bool:
    archive = read_archive_masks()
    return all(archive.get((int(r["k_true"]), int(r["n"]), int(r["replicate"])))
               == (r["train_mask_hash"], r["test_mask_hash"]) for r in fit_rows)


# ---------------------------------------------------------------------------
# production
# ---------------------------------------------------------------------------

def run_production(*, allow_em: bool, confirm: bool) -> dict[str, Any]:
    """The full sweep.  Writes into a staging directory and renames once.

    An interrupted run leaves ``<artifact>.partial`` behind and no final
    directory at all.  That is deliberate: partially completed fits are not
    analysable evidence, and a human decides whether to delete the leftover
    and start again.
    """

    _require(allow_em, "production requires --allow-em")
    _require(confirm, "production requires --confirm-em-convergence-sweep")
    _require(not ARTIFACT_DIR.exists(),
             f"{ARTIFACT_DIR} already exists; results are never overwritten")
    _require(not STAGING_DIR.exists(),
             f"{STAGING_DIR} is left over from an earlier run; inspect and "
             f"delete it before starting another")
    record, spec = load_protocol()

    STAGING_DIR.mkdir(parents=True, exist_ok=False)
    started = time.time()
    fingerprint = environment_fingerprint()
    head = git_head()
    clean_before = working_tree_clean()
    write_json(STAGING_DIR / "protocol.json", record)

    states: dict[str, np.ndarray] = {}
    iteration_rows: list[dict[str, Any]] = []
    rows, fit_rows = _run_cells(
        build_cells(spec), spec=spec, max_iter=int(spec["max_iter"]),
        checkpoints=spec.checkpoints, archive_check=True,
        states=states, iteration_rows=iteration_rows)
    _require(len(fit_rows) == int(spec["expected_long_runs"]),
             f"{len(fit_rows)} fits, expected {spec['expected_long_runs']}")
    _require(len(rows) == int(spec["expected_checkpoint_rows"]),
             f"{len(rows)} checkpoint rows, expected "
             f"{spec['expected_checkpoint_rows']}")
    _require(len(iteration_rows) == int(spec["expected_iteration_rows"]),
             f"{len(iteration_rows)} iteration rows, expected "
             f"{spec['expected_iteration_rows']}")
    replication = compare_to_archive(rows, spec.baseline_checkpoint)
    _require(replication["match"]
             and replication["compared_rows"] == len(fit_rows),
             "the all-fit checkpoint comparison did not match the archive")

    write_csv(STAGING_DIR / "checkpoint_results.csv", rows, CHECKPOINT_COLUMNS)
    write_csv(STAGING_DIR / "fit_results.csv", fit_rows, FIT_COLUMNS)
    write_csv(STAGING_DIR / "iteration_diagnostics.csv", iteration_rows,
              ITERATION_COLUMNS)
    write_npz(STAGING_DIR / "checkpoint_states.npz", states)
    write_json(STAGING_DIR / "replication_vs_baseline.json", {
        "baseline_run": spec["baseline_run"],
        "baseline_protocol_hash": spec["baseline_protocol_hash"],
        "checkpoint": spec.baseline_checkpoint,
        "scope": "ALL fits, not only the gate cell",
        "enforced_inside_the_callback": True,
        "compared_rows": replication["compared_rows"],
        "mismatch_count": len(replication["mismatches"]),
        "match": replication["match"],
    })
    elapsed = round(time.time() - started, 1)
    write_json(STAGING_DIR / "runinfo.json", {
        "experiment_id": EXPERIMENT_ID,
        "protocol_hash": record["protocol_hash"],
        "protocol_source": record["protocol_source"],
        "runner_version": "em-convergence-runner-v4",
        "artifact_version": ARTIFACT_VERSION,
        "attempted_fit_count": len(fit_rows),
        "completed_fit_count": len(fit_rows),
        "checkpoint_row_count": len(rows),
        "iteration_row_count": len(iteration_rows),
        "expected_fit_count": int(spec["expected_long_runs"]),
        "expected_checkpoint_rows": int(spec["expected_checkpoint_rows"]),
        "expected_iteration_rows": int(spec["expected_iteration_rows"]),
        "fits_with_retries": sum(1 for r in fit_rows if int(r["retry_count"])),
        "adaptive_stopping": False,
        "runtime_abort_rule": "none",
        "stopping_reasons": sorted({r["stopping_reason"] for r in fit_rows}),
        "wall_clock_seconds": elapsed,
        "run_code_sha": head,
        "environment": fingerprint,
        "environment_key": environment_key(fingerprint),
        "working_tree_clean_before": clean_before,
        "working_tree_clean_after": working_tree_clean(),
    })

    # The last step, and the only one that creates the final directory.
    _require(not ARTIFACT_DIR.exists(),
             f"{ARTIFACT_DIR} appeared during the run; refusing to publish")
    STAGING_DIR.rename(ARTIFACT_DIR)
    print(json.dumps({"status": "COMPLETE", "artifact_dir": str(ARTIFACT_DIR),
                      "fits": len(fit_rows), "checkpoint_rows": len(rows),
                      "iteration_rows": len(iteration_rows),
                      "wall_clock_seconds": elapsed}, indent=2))
    return {"artifact_dir": str(ARTIFACT_DIR), "fits": len(fit_rows),
            "checkpoint_rows": len(rows)}


# ---------------------------------------------------------------------------
# planning and draft emission (no EM)
# ---------------------------------------------------------------------------

def plan(n_values=None) -> dict[str, Any]:
    """Scope, runtime and storage projection from the ARCHIVE only.  No EM."""

    n_values = tuple(RUNNER_CONTRACT["n_grid"]) if n_values is None \
        else tuple(n_values)
    with (BASELINE_RUN_DIR / "fit_results.csv").open(
            "r", encoding="utf-8", newline="") as handle:
        archive = list(csv.DictReader(handle))
    k_true = RUNNER_CONTRACT["k_true"]
    per_n = {n: sum(float(r["runtime_s"]) for r in archive
                    if int(r["k_true"]) == k_true and int(r["n"]) == n)
             for n in n_values}
    candidate_k = RUNNER_CONTRACT["candidate_k"]
    starts = RUNNER_CONTRACT["starts"]
    ladder = RUNNER_CONTRACT["checkpoints"]
    replicates = RUNNER_CONTRACT["replicates"]
    d = RUNNER_CONTRACT["d"]
    L = RUNNER_CONTRACT["L"]
    fits = len(n_values) * replicates * len(candidate_k) * len(starts)
    seconds = sum(per_n.values()) * RUNNER_CONTRACT["max_iter"] / 8

    # Archived state per checkpoint: Z (n x K), Z_samples (n x K x L),
    # F (d x K), sigma (d x d), scalars (3).
    floats = sum(len(ladder) * len(starts) * replicates
                 * (n * k + n * k * L + d * k + d * d + 3)
                 for n in n_values for k in candidate_k)
    return {
        "n_grid": list(n_values),
        "fits": fits,
        "checkpoint_rows": fits * len(ladder),
        "iteration_rows": fits * RUNNER_CONTRACT["max_iter"],
        "baseline_seconds_at_8_iterations": {n: round(v, 1)
                                             for n, v in per_n.items()},
        "projected_hours_at_max_iter": round(seconds / 3600, 2),
        "projection_note": (
            "linear in iterations, an ESTIMATE and not a bound: it counts the "
            "archive's fixed end-of-run cost as if it scaled with iterations "
            "(over-estimate) while the new run pays that cost once per "
            "checkpoint (under-estimate). It is OPERATIONAL only -- there is no "
            "automatic runtime abort, and runtime never includes or excludes a "
            "fit"),
        "archived_state_floats": floats,
        "archived_state_megabytes_uncompressed": round(floats * 8 / 1048576, 2),
        "protocol_hash_of_contract": canonical_hash(protocol_body_from_contract()),
    }


def emit_draft() -> Path:
    """Write the DRAFT protocol.  Exactly ONE specification, no legacy keys.

    The file is REPLACED, never merged into: a merge is how the previous draft
    ended up carrying a stale ``scope.n`` of [75, 100, 150] beside a current
    ``protocol.n_grid`` of [50, 75, 100, 150] (finding B2-01).
    """

    body = protocol_body_from_contract()
    document = {
        "status": "DRAFT",
        "status_note": (
            "The single specification is the 'protocol' object below; it is "
            "generated from the runner contract by --emit-draft, so no second "
            "copy of any field exists anywhere. The runner verifies every "
            "field against the same contract before any EM starts and records "
            "the hash of the body it used in the artifact. To freeze the "
            f"conditions, copy this file to {FROZEN_PROTOCOL_PATH.name} with "
            "status FROZEN; the runner prefers that file when it exists."),
        "experiment_id": EXPERIMENT_ID,
        "design_document": ("reports/k_selection/"
                            "em_convergence_sensitivity_design_20260905.md"),
        "protocol": body,
        "would_freeze_to_hash": canonical_hash(body),
        "artifact_schema": {
            "checkpoint_results.csv": CHECKPOINT_COLUMNS,
            "fit_results.csv": FIT_COLUMNS,
            "iteration_diagnostics.csv": ITERATION_COLUMNS,
            "checkpoint_states.npz": (
                "per checkpoint: <state_key>_Z, <state_key>_Zsamples, "
                "<state_key>_F, <state_key>_sigma, <state_key>_scalars "
                "(w0, w, var_z) -- everything needed to recompute S1, S3, "
                "Q_strict and S2 independently"),
        },
        "runtime_locations": {
            "replication_gate_result": str(RUNTIME_ROOT.relative_to(ROOT)).replace("\\", "/"),
            "staging": str(STAGING_DIR.relative_to(ROOT)).replace("\\", "/"),
            "final_evidence": str(ARTIFACT_DIR.relative_to(ROOT)).replace("\\", "/"),
            "note": ("the run writes into staging and renames once at the end, "
                     "so an interrupted run leaves no final directory. A "
                     "leftover staging directory is removed by a human"),
        },
        "production_status": {
            "production_runs_so_far": 0,
            "new_production_fits_so_far": 0,
            "replication_gate_runs_so_far": 0,
        },
    }
    write_json(DRAFT_PROTOCOL_PATH, document)
    return DRAFT_PROTOCOL_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--plan-n", type=int, nargs="+", default=None)
    parser.add_argument("--emit-draft", action="store_true")
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
    if args.replication_gate:
        result = run_replication_gate(allow_em=args.allow_em)
        return 0 if result["verdict"] == "PASS" else 1
    if args.full:
        run_production(allow_em=args.allow_em,
                       confirm=args.confirm_em_convergence_sweep)
        return 0
    parser.error("choose --plan, --emit-draft, --replication-gate or --full")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
