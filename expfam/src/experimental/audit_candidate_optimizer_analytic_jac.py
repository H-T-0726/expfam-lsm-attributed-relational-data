"""Artifact-only auditor for the Gate 74-B1R run (Issue #74).

Separate from the Gate 74-B1 auditor on purpose.  Re-auditing B1 with a
module that had grown a B1R branch would rewrite the preserved B1 report, and
the two stages certify different things: B1's reference differentiates
numerically, B1R's is handed the analytic gradient.  Each stage gets an
auditor that states exactly what that stage was supposed to do.

Like the B1 auditor, this imports neither validator nor selector.  It
transcribes the frozen design itself and rebuilds the arrays from that
transcription to confirm the digests, so it can catch a constant edited after
someone saw a result rather than only confirming that the validator agrees
with itself.

Usage::

    python audit_candidate_optimizer_analytic_jac.py --run-dir <directory>
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

AUDITOR_VERSION = "candidate-optimizer-analytic-jac-auditor-v1"
GATE = "74-B1R"

REQUIRED_ARTIFACTS = (
    "protocol.json",
    "objective_checks.csv",
    "gradient_checks.csv",
    "reference_solutions.csv",
    "adam_comparison.csv",
    "family_ordering.csv",
    "summary.json",
)

# --------------------------------------------------------------------------
# Frozen expectations, transcribed independently from the Issue #74 Gate
# 74-B1 / 74-B1R comments.  Do not replace these with an import.
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
EXPECTED_CASE_ONES = {"case_A_balanced_nonseparable": 12,
                      "case_B_sparse_nonseparable": 8}

EXPECTED_INITS = {
    "init_zero": [0.0, 0.0, 0.0],
    "init_pos": [0.6, -0.4, 0.3],
    "init_neg": [-0.6, 0.4, -0.3],
}
EXPECTED_FAMILIES = ["bernoulli", "poisson"]

# Unchanged production settings. B1R must not have touched these.
EXPECTED_ADAM = {"max_iter": 50, "lr": 0.01, "beta1": 0.9, "beta2": 0.999,
                 "eps": 1e-8, "tol": 1e-6}

# The one thing B1R changes: the Jacobian handed to the reference solver.
EXPECTED_REFERENCE_SOLVER = {
    "method": "BFGS",
    "jac": "analytic_production_gradient",
    "maxiter": 2000,
    "gtol": 1e-10,
    "finite_difference_jacobian": False,
}

EXPECTED_THRESHOLDS = {
    "objective_match_tol": 1e-10,
    "fd_step": 1e-6,
    "gradient_match_tol": 1e-5,
    "reference_grad_inf_tol": 1e-8,
    "reference_score_agree_tol": 1e-8,
    "reference_loading_agree_tol": 1e-5,
    "adam_gap_per_obs_tol": 1e-6,
    "adam_grad_inf_tol": 1e-4,
}

EXPECTED_ADAM_RUNS = 12          # 2 cases x 2 families x 3 initialisations
EXPECTED_ORDERING_ROWS = 6       # 2 cases x 3 initialisations

STATUS_CERTIFIED = "REFERENCE_V2_CERTIFIED"
STATUS_NOT_CERTIFIED = "REFERENCE_V2_NOT_CERTIFIED"
STATUS_FAIL = "FAIL"
ADAM_CERTIFIED = "CURRENT_ADAM_CERTIFIED_FOR_B1R"
ADAM_NEEDS_HUMAN = "HUMAN_OPTIMIZER_DECISION_REQUIRED"
ADAM_NOT_RUN = "NOT_RUN"


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
    """Rebuild the frozen (n, k, L) block and digest it, as NumPy would."""

    flat: list[float] = []
    for vector in EXPECTED_BASE_VECTORS:
        for _ in range(EXPECTED_GROUP_REPEATS):
            for coordinate in vector:
                for scale in EXPECTED_SAMPLE_SCALES:
                    flat.append(coordinate * scale)
    return hashlib.sha256(_float8(flat)).hexdigest()


def rebuild_case_digest(case: str) -> str:
    return hashlib.sha256(_float8(rebuild_case_column(case))).hexdigest()


def _check_protocol(protocol: dict[str, Any],
                    findings: list[Finding]) -> None:
    if protocol.get("gate") != GATE:
        findings.append(Finding("BLOCKER", "protocol",
                                f"gate is {protocol.get('gate')!r}, expected "
                                f"{GATE!r}"))
    if protocol.get("uses_rng") is not False:
        findings.append(Finding("BLOCKER", "protocol",
                                "the validation must not use an RNG"))
    if protocol.get("em_executions") != 0:
        findings.append(Finding(
            "BLOCKER", "protocol",
            f"em_executions is {protocol.get('em_executions')!r}; B1R is a "
            f"zero-EM validation"))
    if protocol.get("uses_smoke_artifacts") is not False:
        findings.append(Finding(
            "BLOCKER", "protocol",
            "the validation must not take the smoke artifacts as input"))
    if protocol.get("production_optimizer_changed") is not False:
        findings.append(Finding(
            "BLOCKER", "protocol",
            "B1R must not change the production optimiser"))
    if protocol.get("production_convergence_rule_changed") is not False:
        findings.append(Finding(
            "BLOCKER", "protocol",
            "B1R must not change the production convergence rule"))
    if protocol.get("converged_flag_is_certification_criterion") is not False:
        findings.append(Finding(
            "BLOCKER", "protocol",
            "the step-based converged flag must not be a B1R criterion"))

    # The frozen inputs must be the B1 ones, unchanged.
    if protocol.get("base_vectors") != EXPECTED_BASE_VECTORS:
        findings.append(Finding("BLOCKER", "protocol",
                                "base_vectors do not match the frozen design"))
    if protocol.get("group_repeats") != EXPECTED_GROUP_REPEATS:
        findings.append(Finding("BLOCKER", "protocol",
                                "group_repeats does not match"))
    if protocol.get("sample_scales") != EXPECTED_SAMPLE_SCALES:
        findings.append(Finding("BLOCKER", "protocol",
                                "sample_scales do not match"))
    for field, wanted in (("n", EXPECTED_N), ("k", EXPECTED_K),
                          ("L", EXPECTED_L)):
        if protocol.get(field) != wanted:
            findings.append(Finding(
                "BLOCKER", "protocol",
                f"{field} is {protocol.get(field)!r}, expected {wanted!r}"))

    cases = protocol.get("cases", {})
    if set(cases) != set(EXPECTED_CASES):
        findings.append(Finding("BLOCKER", "protocol",
                                f"cases are {sorted(cases)}, expected "
                                f"{sorted(EXPECTED_CASES)}"))
    for name, groups in EXPECTED_CASES.items():
        if cases.get(name) != groups:
            findings.append(Finding(
                "BLOCKER", "protocol",
                f"{name} groups do not match the frozen design"))
        if protocol.get("case_ones_count", {}).get(name) != EXPECTED_CASE_ONES[name]:
            findings.append(Finding(
                "BLOCKER", "protocol",
                f"{name} has "
                f"{protocol.get('case_ones_count', {}).get(name)!r} ones, "
                f"expected {EXPECTED_CASE_ONES[name]}"))
        if protocol.get("case_columns", {}).get(name) != rebuild_case_column(name):
            findings.append(Finding(
                "BLOCKER", "protocol",
                f"{name} column does not match the rebuilt frozen column"))
        recorded = protocol.get("case_column_sha256", {}).get(name)
        if recorded != rebuild_case_digest(name):
            findings.append(Finding(
                "BLOCKER", "protocol",
                f"{name} column digest {recorded!r} does not match the "
                f"independently rebuilt digest"))

    if protocol.get("z_samples_sha256") != rebuild_z_samples_digest():
        findings.append(Finding(
            "BLOCKER", "protocol",
            "the Z-sample digest does not match the independently rebuilt "
            "frozen design"))

    if protocol.get("initialisations") != EXPECTED_INITS:
        findings.append(Finding("BLOCKER", "protocol",
                                "initialisations do not match"))
    if protocol.get("families") != EXPECTED_FAMILIES:
        findings.append(Finding("BLOCKER", "protocol", "families do not match"))

    adam = protocol.get("adam", {})
    for key, wanted in EXPECTED_ADAM.items():
        if adam.get(key) != wanted:
            findings.append(Finding(
                "BLOCKER", "protocol",
                f"the production Adam {key} is {adam.get(key)!r}, expected "
                f"the unchanged {wanted!r}"))

    solver = protocol.get("reference_solver", {})
    for key, wanted in EXPECTED_REFERENCE_SOLVER.items():
        if solver.get(key) != wanted:
            findings.append(Finding(
                "BLOCKER", "protocol",
                f"reference solver {key} is {solver.get(key)!r}, expected "
                f"{wanted!r}"))
    if solver.get("fallback_solvers"):
        findings.append(Finding(
            "BLOCKER", "protocol",
            f"a fallback solver was used: {solver.get('fallback_solvers')!r}"))

    thresholds = protocol.get("thresholds", {})
    for key, wanted in EXPECTED_THRESHOLDS.items():
        recorded = thresholds.get(key)
        if recorded is None or float(recorded) != wanted:
            findings.append(Finding(
                "BLOCKER", "protocol",
                f"threshold {key} is {recorded!r}, expected {wanted!r}"))

    if protocol.get("expected_adam_runs") != EXPECTED_ADAM_RUNS:
        findings.append(Finding(
            "BLOCKER", "protocol",
            f"expected_adam_runs is {protocol.get('expected_adam_runs')!r}, "
            f"expected {EXPECTED_ADAM_RUNS}"))
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


def _expected_triples() -> set:
    return {(case, family, init)
            for case in EXPECTED_CASES
            for family in EXPECTED_FAMILIES
            for init in EXPECTED_INITS}


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


def _check_finite(rows: Sequence[dict[str, str]], table: str,
                  columns: Sequence[str], findings: list[Finding]) -> None:
    for index, row in enumerate(rows):
        for column in columns:
            value = _as_float(row.get(column, ""))
            if value is None:
                continue
            if not math.isfinite(value):
                findings.append(Finding(
                    "BLOCKER", table,
                    f"row {index}: {column} is not finite "
                    f"({row.get(column)!r})"))


def _is_not_run(rows: Sequence[dict[str, str]]) -> bool:
    return len(rows) == 1 and rows[0].get("status") == "NOT_RUN"


def audit(run_dir: Path) -> dict[str, Any]:
    """Audit a finished B1R run directory and return the report."""

    findings: list[Finding] = []

    missing = [name for name in REQUIRED_ARTIFACTS
               if not (run_dir / name).is_file()]
    if missing:
        findings.append(Finding("BLOCKER", "artifacts",
                                f"missing required artifacts: {missing}"))
        return _finish(run_dir, findings, None, ADAM_NOT_RUN)

    protocol = _read_json(run_dir / "protocol.json")
    summary = _read_json(run_dir / "summary.json")
    _check_protocol(protocol, findings)

    objective_rows = _read_csv(run_dir / "objective_checks.csv")
    gradient_rows = _read_csv(run_dir / "gradient_checks.csv")
    reference_rows = _read_csv(run_dir / "reference_solutions.csv")
    adam_rows = _read_csv(run_dir / "adam_comparison.csv")
    ordering_rows = _read_csv(run_dir / "family_ordering.csv")

    triples = _expected_triples()
    _check_coverage(objective_rows, "objective_checks",
                    ("case", "family", "init"), triples, findings)
    _check_coverage(gradient_rows, "gradient_checks",
                    ("case", "family", "init"), triples, findings)
    _check_coverage(reference_rows, "reference_solutions",
                    ("case", "family", "init"), triples, findings)

    _check_finite(objective_rows, "objective_checks",
                  ("production_score", "independent_score", "abs_difference"),
                  findings)
    _check_finite(gradient_rows, "gradient_checks",
                  ("max_abs_difference",), findings)
    _check_finite(reference_rows, "reference_solutions",
                  ("score", "analytic_grad_inf", "score_spread_across_inits",
                   "loading_spread_across_inits"), findings)

    # Regression guards, recomputed from the rows.
    objective_ok = True
    for row in objective_rows:
        difference = _as_float(row.get("abs_difference", ""))
        if difference is None or difference > EXPECTED_THRESHOLDS["objective_match_tol"]:
            objective_ok = False
            findings.append(Finding(
                "BLOCKER", "objective_checks",
                f"{row.get('case')}/{row.get('family')}/{row.get('init')}: "
                f"production and independent scores differ by "
                f"{row.get('abs_difference')!r}"))

    gradient_ok = True
    for row in gradient_rows:
        difference = _as_float(row.get("max_abs_difference", ""))
        if difference is None or difference > EXPECTED_THRESHOLDS["gradient_match_tol"]:
            gradient_ok = False
            findings.append(Finding(
                "BLOCKER", "gradient_checks",
                f"{row.get('case')}/{row.get('family')}/{row.get('init')}: "
                f"analytic gradient differs from the finite difference by "
                f"{row.get('max_abs_difference')!r}"))
        if _as_float(row.get("fd_step", "")) != EXPECTED_THRESHOLDS["fd_step"]:
            findings.append(Finding(
                "BLOCKER", "gradient_checks",
                f"fd_step is {row.get('fd_step')!r}, expected "
                f"{EXPECTED_THRESHOLDS['fd_step']!r}"))

    # Certification, recomputed rather than trusted.
    per_problem: dict[tuple, list[dict[str, str]]] = {}
    for row in reference_rows:
        per_problem.setdefault((row.get("case", ""), row.get("family", "")),
                               []).append(row)
    certified_problems = set()
    for problem, rows in per_problem.items():
        grads = [_as_float(r.get("analytic_grad_inf", "")) for r in rows]
        scores = [_as_float(r.get("score", "")) for r in rows]
        spread_score = _as_float(rows[0].get("score_spread_across_inits", ""))
        spread_loading = _as_float(rows[0].get("loading_spread_across_inits", ""))
        finite_ok = all(_truthy(r.get("finite", "")) for r in rows)
        grad_ok = all(g is not None
                      and g <= EXPECTED_THRESHOLDS["reference_grad_inf_tol"]
                      for g in grads)
        agree_ok = (spread_score is not None
                    and spread_score <= EXPECTED_THRESHOLDS["reference_score_agree_tol"]
                    and spread_loading is not None
                    and spread_loading <= EXPECTED_THRESHOLDS["reference_loading_agree_tol"])
        recomputed_ok = bool(finite_ok and grad_ok and agree_ok)
        claimed = all(_truthy(r.get("certified", "")) for r in rows)
        if recomputed_ok != claimed:
            findings.append(Finding(
                "BLOCKER", "reference_solutions",
                f"{problem}: rows claim certified={claimed} but the recorded "
                f"values imply {recomputed_ok}"))
        if recomputed_ok:
            certified_problems.add(problem)
        if scores and any(s is None for s in scores):
            findings.append(Finding(
                "BLOCKER", "reference_solutions",
                f"{problem}: a score could not be read"))

    reference_ok = len(certified_problems) == 4
    recomputed_status = (
        STATUS_FAIL if not (objective_ok and gradient_ok)
        else (STATUS_CERTIFIED if reference_ok else STATUS_NOT_CERTIFIED))

    if summary.get("status") != recomputed_status:
        findings.append(Finding(
            "BLOCKER", "summary",
            f"summary.json reports status {summary.get('status')!r} but the "
            f"artifact rows imply {recomputed_status!r}"))

    recomputed_adam = ADAM_NOT_RUN
    adam_ran = not _is_not_run(adam_rows)
    ordering_ran = not _is_not_run(ordering_rows)

    if recomputed_status != STATUS_CERTIFIED:
        if adam_ran or ordering_ran:
            findings.append(Finding(
                "BLOCKER", "adam_comparison",
                "the reference was not certified, so no Adam comparison "
                "should have been produced"))
    else:
        if not adam_ran or not ordering_ran:
            findings.append(Finding(
                "BLOCKER", "adam_comparison",
                "the reference was certified but the Adam comparison is "
                "marked NOT_RUN"))
        else:
            _check_coverage(adam_rows, "adam_comparison",
                            ("case", "family", "init"), triples, findings)
            _check_coverage(
                ordering_rows, "family_ordering", ("case", "init"),
                {(case, init) for case in EXPECTED_CASES
                 for init in EXPECTED_INITS}, findings)
            _check_finite(adam_rows, "adam_comparison",
                          ("adam_score", "reference_score", "objective_gap",
                           "objective_gap_per_observation",
                           "loading_inf_distance_to_ref", "adam_grad_inf"),
                          findings)
            for row in adam_rows:
                for key, wanted in (("adam_max_iter", EXPECTED_ADAM["max_iter"]),
                                    ("adam_lr", EXPECTED_ADAM["lr"]),
                                    ("adam_tol", EXPECTED_ADAM["tol"])):
                    value = _as_float(row.get(key, ""))
                    if value is None or value != wanted:
                        findings.append(Finding(
                            "BLOCKER", "adam_comparison",
                            f"{key} is {row.get(key)!r}, expected the "
                            f"unchanged production value {wanted!r}"))

            adam_failures = []
            for row in adam_rows:
                label = (f"{row.get('case')}/{row.get('family')}/"
                         f"{row.get('init')}")
                gap = _as_float(row.get("objective_gap_per_observation", ""))
                grad = _as_float(row.get("adam_grad_inf", ""))
                if gap is None or gap > EXPECTED_THRESHOLDS["adam_gap_per_obs_tol"]:
                    adam_failures.append(f"{label}: gap per observation {gap}")
                if grad is None or grad > EXPECTED_THRESHOLDS["adam_grad_inf_tol"]:
                    adam_failures.append(f"{label}: gradient inf norm {grad}")
            for row in ordering_rows:
                if not _truthy(row.get("ordering_agrees", "")):
                    adam_failures.append(
                        f"{row.get('case')}/{row.get('init')}: ordering "
                        f"disagrees with the certified reference")
            recomputed_adam = (ADAM_CERTIFIED if not adam_failures
                               else ADAM_NEEDS_HUMAN)

    if summary.get("adam_verdict") != recomputed_adam:
        findings.append(Finding(
            "BLOCKER", "summary",
            f"summary.json reports adam_verdict "
            f"{summary.get('adam_verdict')!r} but the artifact rows imply "
            f"{recomputed_adam!r}"))
    if summary.get("em_executions") != 0:
        findings.append(Finding(
            "BLOCKER", "summary",
            f"summary.json records {summary.get('em_executions')!r} EM "
            f"executions; B1R is a zero-EM validation"))

    report = _finish(run_dir, findings, recomputed_status, recomputed_adam)
    report["certified_problems"] = len(certified_problems)
    report["adam_rows"] = len(adam_rows) if adam_ran else 0
    report["ordering_rows"] = len(ordering_rows) if ordering_ran else 0
    report["ordering_agreements"] = (
        sum(1 for row in ordering_rows if _truthy(row.get("ordering_agrees", "")))
        if ordering_ran else 0)
    report["adam_converged_flag_true"] = (
        sum(1 for row in adam_rows if _truthy(row.get("converged_flag", "")))
        if adam_ran else 0)
    _write_report(run_dir, report)
    return report


def _finish(run_dir: Path, findings: Sequence[Finding], status: str | None,
            adam_verdict: str) -> dict[str, Any]:
    counts = {severity: sum(1 for f in findings if f["severity"] == severity)
              for severity in ("BLOCKER", "HIGH", "MEDIUM")}
    report = {
        "auditor_version": AUDITOR_VERSION,
        "run_dir": str(run_dir),
        "gate": GATE,
        "recomputed_status": status,
        "recomputed_adam_verdict": adam_verdict,
        "verdict": "FAIL" if counts["BLOCKER"] else "PASS",
        "blocker_count": counts["BLOCKER"],
        "high_count": counts["HIGH"],
        "medium_count": counts["MEDIUM"],
        "finding_count": len(findings),
        "audit_clean": counts["BLOCKER"] == 0 and counts["HIGH"] == 0,
        "findings": list(findings),
        "note": "B1R certifies nothing about the production optimiser by "
                "itself. Any change to the optimiser or to its convergence "
                "rule is a separate Human Gate decision.",
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
        description="Audit a Gate 74-B1R analytic-jac certification run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    report = audit(args.run_dir)
    print(f"verdict={report['verdict']} status={report['recomputed_status']} "
          f"adam={report['recomputed_adam_verdict']} "
          f"blockers={report['blocker_count']} high={report['high_count']} "
          f"medium={report['medium_count']}")
    for finding in report["findings"]:
        print(f"  [{finding['severity']}] {finding['check']}: "
              f"{finding['message']}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
