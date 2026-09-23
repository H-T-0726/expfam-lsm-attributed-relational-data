"""Gate 74-B2: zero-EM migration check for the Phase 9C candidate optimiser.

A human approved migrating the Phase 9C candidate-family optimiser from the
50-step Adam to the analytic-gradient BFGS configuration that Gate 74-B1R
certified.  This module proves the migrated PRODUCTION path on exactly the
deterministic problems B1R certified, before any new smoke runs.

It is a migration regression test, not a new scientific experiment.  The cases,
the Z design, the three initialisations, the families and the thresholds are
the B1/B1R ones, imported rather than restated.  The certified B1R optimum is
read from that stage's recorded artifact and used ONLY as a frozen comparison
target: nothing is re-derived, and that directory is never written to.

Each of the twelve comparisons must satisfy, on the production path:

    finite loadings and score
    final analytic gradient infinity norm <= 1e-8
    strict score agrees with the certified reference within 1e-8
    loading infinity-distance to the certified reference <= 1e-5

and the Bernoulli-versus-Poisson ordering must agree with the reference in all
six comparisons.

No EM, no RNG, no smoke artifact as input, no fallback solver, no retry, no
post-result tuning.

Protocol: GitHub Issue #74, Gate 74-B2 comment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from family_selection import (                                 # noqa: E402
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPS,
    ADAM_LR,
    ADAM_MAX_ITER,
    ADAM_TOL,
    BFGS_GTOL,
    BFGS_MAXITER,
    BFGS_METHOD,
    CANDIDATE_GRAD_INF_TOL,
    OPTIMIZER_ADAM,
    PHASE9C_CANDIDATE_OPTIMIZER,
    score_column_candidates,
    support_gate,
)
# The frozen B1/B1R design, imported so the migration is checked on exactly
# the problems the reference was certified on.
from validate_candidate_optimizer import (                     # noqa: E402
    BASE_VECTORS,
    CASES,
    FAMILIES,
    GROUP_REPEATS,
    INITIALISATIONS,
    K_DIM,
    N_ROWS,
    N_SAMPLES,
    SAMPLE_SCALES,
    array_digest,
    build_case_column,
    build_z_samples,
)

VALIDATOR_VERSION = "candidate-optimizer-migration-v1"
GATE = "74-B2"

# Frozen comparison tolerances (Issue #74 Gate 74-B2).
SCORE_MATCH_TOL = 1e-8
LOADING_MATCH_TOL = 1e-5

STATUS_READY = "B2_READY_FOR_SMOKE_V2"
STATUS_BLOCKED = "B2_BLOCKED"
STATUS_FAIL = "FAIL"

DEFAULT_REFERENCE_DIR = (
    _HERE.parents[2] / "expfam" / "results" / "family_selection"
    / "optimizer_validation_analytic_jac_20260923")


class MigrationStop(RuntimeError):
    """Fail-fast stop for the migration validator."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MigrationStop(message)


# --------------------------------------------------------------------------
# the certified B1R reference, read as a frozen comparison target
# --------------------------------------------------------------------------

def load_certified_reference(reference_dir: Path) -> dict[str, Any]:
    """Read the certified B1R optimum. Read-only: that directory is immutable."""

    protocol_path = reference_dir / "protocol.json"
    summary_path = reference_dir / "summary.json"
    solutions_path = reference_dir / "reference_solutions.csv"
    for path in (protocol_path, summary_path, solutions_path):
        _require(path.is_file(), f"missing B1R artifact: {path}")

    with protocol_path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    _require(protocol.get("gate") == "74-B1R",
             f"{reference_dir} is not a Gate 74-B1R run "
             f"(gate={protocol.get('gate')!r})")
    _require(summary.get("status") == "REFERENCE_V2_CERTIFIED",
             f"the B1R reference is not certified "
             f"(status={summary.get('status')!r}); there is nothing to "
             f"migrate against")
    _require(protocol.get("z_samples_sha256") == array_digest(build_z_samples()),
             "the B1R run used a different Z design from the one imported here")

    with solutions_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    optima: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("certified", "")).strip().lower() != "true":
            continue
        key = (row["case"], row["family"])
        score = float(row["score"])
        # B1R took the best-scoring of the three starts as the optimum; the
        # three agreed to 1.8e-15, so this is a formality, kept identical.
        if key not in optima or score > optima[key]["score"]:
            optima[key] = {
                "score": score,
                "loading": np.array([float(v) for v in row["loading"].split("|")],
                                    dtype=np.float64),
                "init": row["init"],
                "analytic_grad_inf": float(row["analytic_grad_inf"]),
            }
    expected = {(case, family) for case in CASES for family in FAMILIES}
    _require(set(optima) == expected,
             f"the B1R run does not certify every problem: got "
             f"{sorted(optima)}, expected {sorted(expected)}")

    return {
        "optima": optima,
        "artifact_path": str(reference_dir),
        "git_sha": protocol.get("git_sha"),
        "validator_version": protocol.get("validator_version"),
        "z_samples_sha256": protocol.get("z_samples_sha256"),
        "solver": protocol.get("reference_solver"),
    }


# --------------------------------------------------------------------------
# the migrated production path, on the frozen problems
# --------------------------------------------------------------------------

def migration_rows(Z_samples: np.ndarray, reference: dict[str, Any]
                   ) -> list[dict[str, Any]]:
    """Run the Phase 9C production candidate path on the 12 frozen problems.

    The call goes through ``score_column_candidates`` -- the same entry point
    the selector uses -- rather than reaching for the optimiser directly, so
    what is checked is the path that will actually run.
    """

    optima = reference["optima"]
    rows: list[dict[str, Any]] = []
    for case in CASES:
        x = build_case_column(case)
        gate, = support_gate(x[:, None])
        _require(gate.candidates == ("bernoulli", "poisson"),
                 f"{case}: the frozen column should be gate-ambiguous, got "
                 f"{gate.candidates}")
        for init_name, init in INITIALISATIONS.items():
            records = score_column_candidates(
                x, Z_samples, gate,
                loading_init=np.asarray(init, dtype=np.float64),
                optimizer=PHASE9C_CANDIDATE_OPTIMIZER)
            for record in records:
                optimum = optima[(case, record.family)]
                score_difference = abs(record.score - optimum["score"])
                loading_distance = float(
                    np.max(np.abs(record.loading - optimum["loading"])))
                finite = bool(np.all(np.isfinite(record.loading))
                              and math.isfinite(record.score)
                              and math.isfinite(record.gradient_inf))
                rows.append({
                    "case": case, "family": record.family, "init": init_name,
                    "optimizer": record.optimizer,
                    "bfgs_method": record.provenance.get("method"),
                    "bfgs_maxiter": record.provenance.get("maxiter"),
                    "bfgs_gtol": record.provenance.get("gtol"),
                    "convergence_grad_inf_tol": CANDIDATE_GRAD_INF_TOL,
                    "n_iter": record.n_iter,
                    "finite": finite,
                    "converged": record.converged,
                    "gradient_inf": record.gradient_inf,
                    "scipy_success": record.provenance.get("scipy_success"),
                    "scipy_status": record.provenance.get("scipy_status"),
                    "scipy_nit": record.provenance.get("scipy_nit"),
                    "scipy_njev": record.provenance.get("scipy_njev"),
                    "fallback_solvers": "|".join(
                        record.provenance.get("fallback_solvers", [])),
                    "retries": record.provenance.get("retries"),
                    "loading": "|".join(f"{v:.12e}" for v in record.loading),
                    "score": record.score,
                    "reference_score": optimum["score"],
                    "reference_loading": "|".join(f"{v:.12e}"
                                                  for v in optimum["loading"]),
                    "score_abs_difference": score_difference,
                    "loading_inf_distance": loading_distance,
                    "score_tolerance": SCORE_MATCH_TOL,
                    "loading_tolerance": LOADING_MATCH_TOL,
                    "passed": bool(finite and record.converged
                                   and record.gradient_inf <= CANDIDATE_GRAD_INF_TOL
                                   and score_difference <= SCORE_MATCH_TOL
                                   and loading_distance <= LOADING_MATCH_TOL),
                })
    return rows


def family_ordering_rows(reference: dict[str, Any],
                         rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    optima = reference["optima"]
    ordering = []
    for case in CASES:
        reference_scores = {family: optima[(case, family)]["score"]
                            for family in FAMILIES}
        reference_winner = max(reference_scores, key=reference_scores.get)
        for init_name in INITIALISATIONS:
            migrated_scores = {}
            for family in FAMILIES:
                row, = [r for r in rows if r["case"] == case
                        and r["family"] == family and r["init"] == init_name]
                migrated_scores[family] = row["score"]
            migrated_winner = max(migrated_scores, key=migrated_scores.get)
            ordering.append({
                "case": case, "init": init_name,
                "reference_winner": reference_winner,
                "reference_score_bernoulli": reference_scores["bernoulli"],
                "reference_score_poisson": reference_scores["poisson"],
                "migrated_winner": migrated_winner,
                "migrated_score_bernoulli": migrated_scores["bernoulli"],
                "migrated_score_poisson": migrated_scores["poisson"],
                "ordering_agrees": bool(migrated_winner == reference_winner),
            })
    return ordering


def historical_adam_still_reachable(Z_samples: np.ndarray) -> dict[str, Any]:
    """Confirm the legacy route is still callable and still means what it did.

    The smoke and the B1/B1R diagnostics were produced by Adam. If asking for
    it by name stopped working, or started applying the new gradient rule,
    those artifacts would no longer be reproducible.
    """

    case = next(iter(CASES))
    x = build_case_column(case)
    gate, = support_gate(x[:, None])
    records = score_column_candidates(
        x, Z_samples, gate,
        loading_init=np.asarray(INITIALISATIONS["init_zero"], dtype=np.float64),
        optimizer=OPTIMIZER_ADAM)
    return {
        "reachable": True,
        "optimizer": records[0].optimizer,
        "convergence_rule": records[0].provenance.get("convergence_rule"),
        "max_iter": records[0].provenance.get("max_iter"),
        "lr": records[0].provenance.get("lr"),
        "tol": records[0].provenance.get("tol"),
        "n_iter_observed": [r.n_iter for r in records],
        "converged_observed": [r.converged for r in records],
    }


# --------------------------------------------------------------------------
# provenance and writing
# --------------------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=_HERE, capture_output=True,
                              text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):        # pragma: no cover
        return ""


def build_protocol(Z_samples: np.ndarray,
                   reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "validator_version": VALIDATOR_VERSION,
        "gate": GATE,
        "derived_from_gate": "74-B1R",
        "uses_rng": False,
        "uses_smoke_artifacts": False,
        "em_executions": 0,
        "phase9c_candidate_optimizer": PHASE9C_CANDIDATE_OPTIMIZER,
        "candidate_optimizer_settings": {
            "method": BFGS_METHOD,
            "jac": "analytic_production_gradient",
            "maxiter": BFGS_MAXITER,
            "gtol": BFGS_GTOL,
            "finite_difference_jacobian": False,
            "fallback_solvers": [],
        },
        "candidate_convergence_rule": {
            "rule": "finite and final analytic gradient infinity norm <= tol",
            "convergence_grad_inf_tol": CANDIDATE_GRAD_INF_TOL,
            "scipy_success_is_criterion": False,
        },
        "historical_adam_preserved": True,
        "historical_adam_settings": {
            "max_iter": ADAM_MAX_ITER, "lr": ADAM_LR, "beta1": ADAM_BETA1,
            "beta2": ADAM_BETA2, "eps": ADAM_EPS, "tol": ADAM_TOL,
            "convergence_rule": "step infinity norm below tol",
        },
        "comparison_tolerances": {
            "score_match_tol": SCORE_MATCH_TOL,
            "loading_match_tol": LOADING_MATCH_TOL,
        },
        "certified_reference": {
            "gate": "74-B1R",
            "artifact_path": reference["artifact_path"],
            "git_sha": reference["git_sha"],
            "validator_version": reference["validator_version"],
            "solver": reference["solver"],
            "used_as": "frozen comparison target only; never recomputed, "
                       "never written to",
        },
        "base_vectors": [list(v) for v in BASE_VECTORS],
        "group_repeats": GROUP_REPEATS,
        "sample_scales": list(SAMPLE_SCALES),
        "n": N_ROWS, "k": K_DIM, "L": N_SAMPLES,
        "z_samples_sha256": array_digest(Z_samples),
        "cases": {name: [list(group) for group in groups]
                  for name, groups in CASES.items()},
        "case_column_sha256": {name: array_digest(build_case_column(name))
                               for name in CASES},
        "initialisations": {name: list(value)
                            for name, value in INITIALISATIONS.items()},
        "families": list(FAMILIES),
        "expected_migration_rows": len(CASES) * len(FAMILIES)
        * len(INITIALISATIONS),
        "production_change_outside_phase9c_selector": False,
        "git_sha": _git("rev-parse", "HEAD").strip() or "unavailable",
        "git_dirty": bool(_git("status", "--porcelain").strip()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "note": "Migration regression check for Gate 74-B2 only. It is not a "
                "new scientific experiment, it does not reclassify the Gate "
                "74-B smoke, and it authorises no execution by itself.",
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write an empty artifact: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False,
                  default=_json_default)
        handle.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialise {type(value)!r}")


ARTIFACT_NAMES = (
    "protocol.json",
    "migration_comparison.csv",
    "family_ordering.csv",
    "summary.json",
)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def execute(out_dir: Path,
            reference_dir: Path = DEFAULT_REFERENCE_DIR) -> dict[str, Any]:
    """Run the B2 migration check once and write its artifacts."""

    _require(not out_dir.exists(),
             f"{out_dir} already exists; a recorded run is never overwritten")

    started = datetime.now(timezone.utc).isoformat()
    Z_samples = build_z_samples()
    reference = load_certified_reference(reference_dir)
    protocol = build_protocol(Z_samples, reference)

    rows = migration_rows(Z_samples, reference)
    ordering = family_ordering_rows(reference, rows)
    adam = historical_adam_still_reachable(Z_samples)

    failures: list[str] = []
    for row in rows:
        label = f"{row['case']}/{row['family']}/{row['init']}"
        if not row["finite"]:
            failures.append(f"{label}: a value was not finite")
        if row["gradient_inf"] > CANDIDATE_GRAD_INF_TOL:
            failures.append(f"{label}: gradient inf {row['gradient_inf']:.3e} > "
                            f"{CANDIDATE_GRAD_INF_TOL}")
        if not row["converged"]:
            failures.append(f"{label}: the candidate was not converged")
        if row["score_abs_difference"] > SCORE_MATCH_TOL:
            failures.append(
                f"{label}: score differs from the certified reference by "
                f"{row['score_abs_difference']:.3e} > {SCORE_MATCH_TOL}")
        if row["loading_inf_distance"] > LOADING_MATCH_TOL:
            failures.append(
                f"{label}: loading is {row['loading_inf_distance']:.3e} from "
                f"the certified reference > {LOADING_MATCH_TOL}")
    for row in ordering:
        if not row["ordering_agrees"]:
            failures.append(
                f"{row['case']}/{row['init']}: the migrated path prefers "
                f"{row['migrated_winner']}, the certified reference prefers "
                f"{row['reference_winner']}")
    if not adam["reachable"]:
        failures.append("the historical Adam route is no longer reachable")

    status = STATUS_READY if not failures else STATUS_BLOCKED

    summary = {
        "validator_version": VALIDATOR_VERSION,
        "gate": GATE,
        "status": status,
        "em_executions": 0,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "phase9c_candidate_optimizer": PHASE9C_CANDIDATE_OPTIMIZER,
        "migration_comparison": {
            "rows": len(rows),
            "passed": sum(1 for row in rows if row["passed"]),
            "converged": sum(1 for row in rows if row["converged"]),
            "max_score_abs_difference": max(row["score_abs_difference"]
                                            for row in rows),
            "max_loading_inf_distance": max(row["loading_inf_distance"]
                                            for row in rows),
            "max_gradient_inf": max(row["gradient_inf"] for row in rows),
            "score_tolerance": SCORE_MATCH_TOL,
            "loading_tolerance": LOADING_MATCH_TOL,
            "gradient_tolerance": CANDIDATE_GRAD_INF_TOL,
        },
        "family_ordering": {
            "rows": len(ordering),
            "agreements": sum(1 for row in ordering if row["ordering_agrees"]),
        },
        "historical_adam": adam,
        "certified_reference": protocol["certified_reference"],
        "failures": failures,
        "interpretation": {
            "what_this_is": "a migration regression check on the frozen "
                            "Gate 74-B1R problems, not a new experiment",
            "does_not_reclassify_smoke": True,
            "authorises_no_execution": "smoke-v2 and C2 remain unauthorised "
                                       "regardless of this result",
            "smoke_artifacts_used_as_input": False,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=False)
    _write_json(out_dir / "protocol.json", protocol)
    _write_csv(out_dir / "migration_comparison.csv", rows)
    _write_csv(out_dir / "family_ordering.csv", ordering)
    _write_json(out_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Gate 74-B2 candidate-optimizer migration check.")
    parser.add_argument("--out", required=True, type=Path,
                        help="fresh output directory (must not exist)")
    parser.add_argument("--reference-dir", type=Path,
                        default=DEFAULT_REFERENCE_DIR,
                        help="the certified Gate 74-B1R run, read only")
    args = parser.parse_args(argv)

    summary = execute(args.out, args.reference_dir)
    print(f"status={summary['status']} "
          f"passed={summary['migration_comparison']['passed']}/"
          f"{summary['migration_comparison']['rows']} "
          f"ordering={summary['family_ordering']['agreements']}/"
          f"{summary['family_ordering']['rows']} "
          f"em_executions={summary['em_executions']}")
    print(f"artifacts written to {args.out}")
    for failure in summary["failures"]:
        print(f"  FAILURE: {failure}")
    return 0 if summary["status"] == STATUS_READY else 1


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
