"""Artifact-only auditor for the Gate 74-B2 migration check (Issue #74).

Imports neither the validator nor the selector.  It transcribes the frozen
Gate 74-B2 settings and thresholds itself and rebuilds the frozen arrays from
that transcription to confirm the digests, so it can catch a constant edited
after someone saw a result rather than only confirming that the validator
agrees with itself.

Usage::

    python audit_optimizer_migration.py --run-dir <directory>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Sequence

AUDITOR_VERSION = "candidate-optimizer-migration-auditor-v1"
GATE = "74-B2"

REQUIRED_ARTIFACTS = (
    "protocol.json",
    "migration_comparison.csv",
    "family_ordering.csv",
    "summary.json",
)

# --------------------------------------------------------------------------
# Frozen expectations, transcribed independently from the Issue #74 Gate
# 74-B1 / 74-B2 comments.  Do not replace these with an import.
# --------------------------------------------------------------------------

EXPECTED_BASE_VECTORS = [
    [1.0, 0.2, -0.1],
    [-0.6, 1.0, 0.3],
    [0.4, -0.7, 1.0],
    [-1.0, -0.2, 0.5],
    [0.3, 0.8, -0.9],
    [-0.5, 0.4, -0.8],
]
EXPECTED_GROUP_REPEATS = 4
EXPECTED_SAMPLE_SCALES = [0.96, 0.98, 1.00, 1.02, 1.04]
EXPECTED_N = 24
EXPECTED_K = 3
EXPECTED_L = 5

EXPECTED_CASES = {
    "case_A_balanced_nonseparable": [
        [1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0],
        [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0],
    ],
    "case_B_sparse_nonseparable": [
        [1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0],
        [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
    ],
}
EXPECTED_INITS = {
    "init_zero": [0.0, 0.0, 0.0],
    "init_pos": [0.6, -0.4, 0.3],
    "init_neg": [-0.6, 0.4, -0.3],
}
EXPECTED_FAMILIES = ["bernoulli", "poisson"]

# The approved Phase 9C candidate optimiser.
EXPECTED_OPTIMIZER = "bfgs"
EXPECTED_OPTIMIZER_SETTINGS = {
    "method": "BFGS",
    "jac": "analytic_production_gradient",
    "maxiter": 2000,
    "gtol": 1e-10,
    "finite_difference_jacobian": False,
}
EXPECTED_CONVERGENCE_GRAD_INF_TOL = 1e-8

# The historical route that must stay preserved, with its own rule.
EXPECTED_ADAM_SETTINGS = {
    "max_iter": 50, "lr": 0.01, "beta1": 0.9, "beta2": 0.999,
    "eps": 1e-8, "tol": 1e-6,
    "convergence_rule": "step infinity norm below tol",
}

EXPECTED_SCORE_MATCH_TOL = 1e-8
EXPECTED_LOADING_MATCH_TOL = 1e-5

EXPECTED_MIGRATION_ROWS = 12     # 2 cases x 2 families x 3 initialisations
EXPECTED_ORDERING_ROWS = 6       # 2 cases x 3 initialisations

STATUS_READY = "B2_READY_FOR_SMOKE_V2"
STATUS_BLOCKED = "B2_BLOCKED"


class Finding(dict):
    """One audit finding."""

    def __init__(self, severity: str, check: str, message: str) -> None:
        super().__init__(severity=severity, check=check, message=message)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float8(values: Sequence[float]) -> bytes:
    return b"".join(struct.pack("<d", float(v)) for v in values)


def rebuild_case_column(case: str) -> list[float]:
    column: list[float] = []
    for group in EXPECTED_CASES[case]:
        column.extend(float(value) for value in group)
    return column


def rebuild_z_samples_digest() -> str:
    flat: list[float] = []
    for vector in EXPECTED_BASE_VECTORS:
        for _ in range(EXPECTED_GROUP_REPEATS):
            for coordinate in vector:
                for scale in EXPECTED_SAMPLE_SCALES:
                    flat.append(coordinate * scale)
    return hashlib.sha256(_float8(flat)).hexdigest()


def rebuild_case_digest(case: str) -> str:
    return hashlib.sha256(_float8(rebuild_case_column(case))).hexdigest()


def _check_protocol(protocol: dict[str, Any], findings: list[Finding]) -> None:
    if protocol.get("gate") != GATE:
        findings.append(Finding("BLOCKER", "protocol",
                                f"gate is {protocol.get('gate')!r}, expected "
                                f"{GATE!r}"))
    if protocol.get("uses_rng") is not False:
        findings.append(Finding("BLOCKER", "protocol",
                                "the migration check must not use an RNG"))
    if protocol.get("em_executions") != 0:
        findings.append(Finding(
            "BLOCKER", "protocol",
            f"em_executions is {protocol.get('em_executions')!r}; B2 is a "
            f"zero-EM check"))
    if protocol.get("uses_smoke_artifacts") is not False:
        findings.append(Finding(
            "BLOCKER", "protocol",
            "the migration check must not take the smoke artifacts as input"))
    if protocol.get("historical_adam_preserved") is not True:
        findings.append(Finding(
            "BLOCKER", "protocol",
            "the historical Adam route must remain preserved"))
    if protocol.get("production_change_outside_phase9c_selector") is not False:
        findings.append(Finding(
            "BLOCKER", "protocol",
            "the approval covers the Phase 9C candidate path only"))

    if protocol.get("phase9c_candidate_optimizer") != EXPECTED_OPTIMIZER:
        findings.append(Finding(
            "BLOCKER", "protocol",
            f"the Phase 9C candidate optimiser is "
            f"{protocol.get('phase9c_candidate_optimizer')!r}, expected "
            f"{EXPECTED_OPTIMIZER!r}"))

    settings = protocol.get("candidate_optimizer_settings", {})
    for key, wanted in EXPECTED_OPTIMIZER_SETTINGS.items():
        if settings.get(key) != wanted:
            findings.append(Finding(
                "BLOCKER", "protocol",
                f"candidate optimiser {key} is {settings.get(key)!r}, "
                f"expected {wanted!r}"))
    if settings.get("fallback_solvers"):
        findings.append(Finding(
            "BLOCKER", "protocol",
            f"a fallback solver was configured: "
            f"{settings.get('fallback_solvers')!r}"))

    rule = protocol.get("candidate_convergence_rule", {})
    if rule.get("convergence_grad_inf_tol") != EXPECTED_CONVERGENCE_GRAD_INF_TOL:
        findings.append(Finding(
            "BLOCKER", "protocol",
            f"the candidate convergence threshold is "
            f"{rule.get('convergence_grad_inf_tol')!r}, expected "
            f"{EXPECTED_CONVERGENCE_GRAD_INF_TOL!r}"))
    if rule.get("scipy_success_is_criterion") is not False:
        findings.append(Finding(
            "BLOCKER", "protocol",
            "SciPy success must not be the convergence criterion"))

    adam = protocol.get("historical_adam_settings", {})
    for key, wanted in EXPECTED_ADAM_SETTINGS.items():
        if adam.get(key) != wanted:
            findings.append(Finding(
                "BLOCKER", "protocol",
                f"the preserved Adam {key} is {adam.get(key)!r}, expected "
                f"{wanted!r}"))

    tolerances = protocol.get("comparison_tolerances", {})
    if tolerances.get("score_match_tol") != EXPECTED_SCORE_MATCH_TOL:
        findings.append(Finding(
            "BLOCKER", "protocol",
            f"score_match_tol is {tolerances.get('score_match_tol')!r}, "
            f"expected {EXPECTED_SCORE_MATCH_TOL!r}"))
    if tolerances.get("loading_match_tol") != EXPECTED_LOADING_MATCH_TOL:
        findings.append(Finding(
            "BLOCKER", "protocol",
            f"loading_match_tol is {tolerances.get('loading_match_tol')!r}, "
            f"expected {EXPECTED_LOADING_MATCH_TOL!r}"))

    reference = protocol.get("certified_reference", {})
    if reference.get("gate") != "74-B1R":
        findings.append(Finding(
            "BLOCKER", "protocol",
            f"the comparison target is {reference.get('gate')!r}, expected "
            f"the certified Gate 74-B1R reference"))
    for field in ("artifact_path", "git_sha"):
        if not reference.get(field):
            findings.append(Finding(
                "HIGH", "protocol",
                f"the certified reference does not record {field}"))

    # The frozen design must be the B1 one.
    if protocol.get("base_vectors") != EXPECTED_BASE_VECTORS:
        findings.append(Finding("BLOCKER", "protocol",
                                "base_vectors do not match the frozen design"))
    if protocol.get("sample_scales") != EXPECTED_SAMPLE_SCALES:
        findings.append(Finding("BLOCKER", "protocol",
                                "sample_scales do not match"))
    for field, wanted in (("n", EXPECTED_N), ("k", EXPECTED_K),
                          ("L", EXPECTED_L),
                          ("group_repeats", EXPECTED_GROUP_REPEATS)):
        if protocol.get(field) != wanted:
            findings.append(Finding(
                "BLOCKER", "protocol",
                f"{field} is {protocol.get(field)!r}, expected {wanted!r}"))
    if protocol.get("z_samples_sha256") != rebuild_z_samples_digest():
        findings.append(Finding(
            "BLOCKER", "protocol",
            "the Z-sample digest does not match the independently rebuilt "
            "frozen design"))
    for name, groups in EXPECTED_CASES.items():
        if protocol.get("cases", {}).get(name) != groups:
            findings.append(Finding(
                "BLOCKER", "protocol",
                f"{name} groups do not match the frozen design"))
        if protocol.get("case_column_sha256", {}).get(name) != \
                rebuild_case_digest(name):
            findings.append(Finding(
                "BLOCKER", "protocol",
                f"{name} column digest does not match the independently "
                f"rebuilt digest"))
    if protocol.get("initialisations") != EXPECTED_INITS:
        findings.append(Finding("BLOCKER", "protocol",
                                "initialisations do not match"))
    if protocol.get("families") != EXPECTED_FAMILIES:
        findings.append(Finding("BLOCKER", "protocol", "families do not match"))
    if protocol.get("expected_migration_rows") != EXPECTED_MIGRATION_ROWS:
        findings.append(Finding(
            "BLOCKER", "protocol",
            f"expected_migration_rows is "
            f"{protocol.get('expected_migration_rows')!r}, expected "
            f"{EXPECTED_MIGRATION_ROWS}"))

    for field in ("git_sha", "python_version", "numpy_version",
                  "scipy_version"):
        if not protocol.get(field):
            findings.append(Finding("HIGH", "protocol",
                                    f"{field} is missing or empty"))
    if protocol.get("git_dirty"):
        findings.append(Finding(
            "HIGH", "protocol",
            "the working tree was dirty; the recorded git SHA does not fully "
            "describe the code that ran"))


def _check_coverage(rows: Sequence[dict[str, str]], table: str,
                    keys: Sequence[str], wanted: set,
                    findings: list[Finding]) -> None:
    seen: dict[tuple, int] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        seen[key] = seen.get(key, 0) + 1
    missing = sorted(wanted - set(seen))
    if missing:
        findings.append(Finding("BLOCKER", table, f"missing rows for {missing}"))
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    if duplicates:
        findings.append(Finding("BLOCKER", table,
                                f"duplicate rows for {duplicates}"))
    unexpected = sorted(set(seen) - wanted)
    if unexpected:
        findings.append(Finding("HIGH", table,
                                f"unexpected rows for {unexpected}"))


def audit(run_dir: Path) -> dict[str, Any]:
    """Audit a finished B2 run directory and return the report."""

    findings: list[Finding] = []

    missing = [name for name in REQUIRED_ARTIFACTS
               if not (run_dir / name).is_file()]
    if missing:
        findings.append(Finding("BLOCKER", "artifacts",
                                f"missing required artifacts: {missing}"))
        return _finish(run_dir, findings, None)

    protocol = _read_json(run_dir / "protocol.json")
    summary = _read_json(run_dir / "summary.json")
    _check_protocol(protocol, findings)

    rows = _read_csv(run_dir / "migration_comparison.csv")
    ordering = _read_csv(run_dir / "family_ordering.csv")

    _check_coverage(rows, "migration_comparison", ("case", "family", "init"),
                    {(case, family, init) for case in EXPECTED_CASES
                     for family in EXPECTED_FAMILIES
                     for init in EXPECTED_INITS}, findings)
    _check_coverage(ordering, "family_ordering", ("case", "init"),
                    {(case, init) for case in EXPECTED_CASES
                     for init in EXPECTED_INITS}, findings)

    passes = 0
    for row in rows:
        label = f"{row.get('case')}/{row.get('family')}/{row.get('init')}"
        if row.get("optimizer") != EXPECTED_OPTIMIZER:
            findings.append(Finding(
                "BLOCKER", "migration_comparison",
                f"{label}: optimizer is {row.get('optimizer')!r}, expected "
                f"{EXPECTED_OPTIMIZER!r}"))
        for key, wanted in (("bfgs_method", EXPECTED_OPTIMIZER_SETTINGS["method"]),
                            ("bfgs_maxiter", EXPECTED_OPTIMIZER_SETTINGS["maxiter"]),
                            ("bfgs_gtol", EXPECTED_OPTIMIZER_SETTINGS["gtol"])):
            recorded = row.get(key)
            matches = (recorded == wanted if isinstance(wanted, str)
                       else _as_float(recorded) == wanted)
            if not matches:
                findings.append(Finding(
                    "BLOCKER", "migration_comparison",
                    f"{label}: {key} is {recorded!r}, expected {wanted!r}"))
        if _as_float(row.get("convergence_grad_inf_tol", "")) != \
                EXPECTED_CONVERGENCE_GRAD_INF_TOL:
            findings.append(Finding(
                "BLOCKER", "migration_comparison",
                f"{label}: convergence threshold is "
                f"{row.get('convergence_grad_inf_tol')!r}"))
        if row.get("fallback_solvers"):
            findings.append(Finding(
                "BLOCKER", "migration_comparison",
                f"{label}: a fallback solver was used"))
        if _as_float(row.get("retries", "")) not in (0.0, None):
            findings.append(Finding(
                "BLOCKER", "migration_comparison",
                f"{label}: retries is {row.get('retries')!r}, expected 0"))

        gradient = _as_float(row.get("gradient_inf", ""))
        score_difference = _as_float(row.get("score_abs_difference", ""))
        loading_distance = _as_float(row.get("loading_inf_distance", ""))
        for name, value in (("gradient_inf", gradient),
                            ("score_abs_difference", score_difference),
                            ("loading_inf_distance", loading_distance)):
            if value is None or not math.isfinite(value):
                findings.append(Finding(
                    "BLOCKER", "migration_comparison",
                    f"{label}: {name} is {row.get(name)!r}"))

        recomputed = bool(
            _truthy(row.get("finite", ""))
            and _truthy(row.get("converged", ""))
            and gradient is not None
            and gradient <= EXPECTED_CONVERGENCE_GRAD_INF_TOL
            and score_difference is not None
            and score_difference <= EXPECTED_SCORE_MATCH_TOL
            and loading_distance is not None
            and loading_distance <= EXPECTED_LOADING_MATCH_TOL)
        if recomputed != _truthy(row.get("passed", "")):
            findings.append(Finding(
                "BLOCKER", "migration_comparison",
                f"{label}: the row claims passed="
                f"{row.get('passed')!r} but its values imply {recomputed}"))
        if recomputed:
            passes += 1

    agreements = sum(1 for row in ordering
                     if _truthy(row.get("ordering_agrees", "")))

    recomputed_status = (
        STATUS_READY
        if (passes == EXPECTED_MIGRATION_ROWS
            and agreements == EXPECTED_ORDERING_ROWS
            and protocol.get("historical_adam_preserved") is True)
        else STATUS_BLOCKED)
    if summary.get("status") != recomputed_status:
        findings.append(Finding(
            "BLOCKER", "summary",
            f"summary.json reports status {summary.get('status')!r} but the "
            f"artifact rows imply {recomputed_status!r}"))
    if summary.get("em_executions") != 0:
        findings.append(Finding(
            "BLOCKER", "summary",
            f"summary.json records {summary.get('em_executions')!r} EM "
            f"executions; B2 is a zero-EM check"))
    if not (summary.get("historical_adam", {}) or {}).get("reachable"):
        findings.append(Finding(
            "BLOCKER", "summary",
            "the historical Adam route is not recorded as reachable"))

    report = _finish(run_dir, findings, recomputed_status)
    report["migration_rows"] = len(rows)
    report["migration_passes"] = passes
    report["ordering_rows"] = len(ordering)
    report["ordering_agreements"] = agreements
    _write_report(run_dir, report)
    return report


def _finish(run_dir: Path, findings: Sequence[Finding],
            status: str | None) -> dict[str, Any]:
    counts = {severity: sum(1 for f in findings if f["severity"] == severity)
              for severity in ("BLOCKER", "HIGH", "MEDIUM")}
    report = {
        "auditor_version": AUDITOR_VERSION,
        "run_dir": str(run_dir),
        "gate": GATE,
        "recomputed_status": status,
        "verdict": "FAIL" if counts["BLOCKER"] else "PASS",
        "blocker_count": counts["BLOCKER"],
        "high_count": counts["HIGH"],
        "medium_count": counts["MEDIUM"],
        "finding_count": len(findings),
        "audit_clean": counts["BLOCKER"] == 0 and counts["HIGH"] == 0,
        "findings": list(findings),
        "note": "B2 authorises no execution by itself. smoke-v2 and C2 remain "
                "separate Human Gate decisions.",
    }
    if counts["BLOCKER"] or status is None:
        _write_report(run_dir, report)
    return report


def _write_report(run_dir: Path, report: dict[str, Any]) -> None:
    if not run_dir.is_dir():
        return
    with (run_dir / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit a Gate 74-B2 optimizer-migration run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    report = audit(args.run_dir)
    print(f"verdict={report['verdict']} status={report['recomputed_status']} "
          f"migration={report.get('migration_passes')}/"
          f"{report.get('migration_rows')} "
          f"ordering={report.get('ordering_agreements')}/"
          f"{report.get('ordering_rows')} "
          f"blockers={report['blocker_count']} high={report['high_count']}")
    for finding in report["findings"]:
        print(f"  [{finding['severity']}] {finding['check']}: "
              f"{finding['message']}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
