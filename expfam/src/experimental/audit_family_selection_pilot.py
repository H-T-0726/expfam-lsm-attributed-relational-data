"""Artifact-only auditor for the Phase 9C family-selection runs (Issue #74).

This module deliberately does NOT import the runner, the selector, the model
classes or the generator.  It reads a finished run directory and checks it
against its OWN literal copy of the frozen Issue #74 conditions.  The
duplication is the point: an auditor that imports the runner's constants can
only confirm that the runner agrees with itself, and would pass unchanged if
someone edited those constants after seeing a result.

What it checks
--------------
* every required artifact is present
* ``protocol.json`` matches the frozen conditions field by field -- n, d, K,
  the family pattern, L, both num_iter values, every seed, and both starts
* the number of EM executions is exactly what the protocol implies
* every table has exactly the rows it should: no missing pair, no duplicate
* every numeric cell is finite
* ``retry_count``, ``replacement_count`` and ``seed_rescue_count`` are 0 and
  the refit ran under ``failure_policy='fail_fast'``
* the candidate-convergence gate is recorded, and its verdict is reported
* the claim-boundary fields are present in ``summary.json``

It writes ``audit_report.json`` into the run directory and exits non-zero when
the verdict is FAIL.  A FAIL is a finding to hand to a human, not something to
fix by rerunning.

Usage::

    python audit_family_selection_pilot.py --run-dir <directory>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

AUDITOR_VERSION = "family-selection-pilot-auditor-v2"

# Present in every run, successful or not: these are written before the first
# EM execution, so their absence means the run never reserved its directory.
ALWAYS_REQUIRED_ARTIFACTS = (
    "protocol.json",
    "runinfo.json",
    "execution_ledger.csv",
)

# Additionally required from a run that claims to have finished.
COMPLETED_RUN_ARTIFACTS = (
    "generator_provenance.csv",
    "support_gate.csv",
    "family_scores.csv",
    "selection_trace.csv",
    "fit_results.csv",
    "summary.json",
)

REQUIRED_ARTIFACTS = ALWAYS_REQUIRED_ARTIFACTS + COMPLETED_RUN_ARTIFACTS

FAILURE_ARTIFACT = "failure.json"

REQUIRED_FAILURE_KEYS = (
    "exception_type", "message", "attempted_em_executions", "replicate",
    "start_label", "execution_kind", "retry_count", "replacement_count",
    "seed_rescue_count", "git_sha",
)

# --------------------------------------------------------------------------
# The frozen expectations, transcribed independently from Issue #74 sections
# C1 and C2.  Do not replace these with an import.
# --------------------------------------------------------------------------

EXPECTED: dict[str, dict[str, Any]] = {
    "smoke": {
        "n": 40,
        "d": 6,
        "k_true": 3,
        "k_fit": 3,
        "family_x_list": ["gaussian", "gaussian", "bernoulli",
                          "bernoulli", "poisson", "poisson"],
        "family_y": "bernoulli",
        "sigma_x_var": 1.0,
        "w0": -1.0,
        "w": 1.0,
        "f_scale": 1.0,                      # f_scale_for_row_norm(0.5, 6, 3)
        "L": 5,
        "refit_num_iter": 8,
        "exploration_num_iter": 8,
        "replicates": [
            {"label": "rep1", "data_seed": 941001,
             "search_seed": 942001, "refit_seed": 943001},
        ],
        "expected_em_executions": 4,
    },
    "pilot": {
        "n": 75,
        "d": 12,
        "k_true": 3,
        "k_fit": 3,
        "family_x_list": ["gaussian"] * 3 + ["bernoulli"] * 6 + ["poisson"] * 3,
        "family_y": "bernoulli",
        "sigma_x_var": 1.0,
        "w0": -1.0,
        "w": 1.0,
        "f_scale": math.sqrt(2.0),           # f_scale_for_row_norm(0.5, 12, 3)
        "L": 5,
        "refit_num_iter": 8,
        "exploration_num_iter": 8,
        "replicates": [
            {"label": "rep1", "data_seed": 951001,
             "search_seed": 952001, "refit_seed": 953001},
            {"label": "rep2", "data_seed": 951002,
             "search_seed": 952002, "refit_seed": 953002},
            {"label": "rep3", "data_seed": 951003,
             "search_seed": 952003, "refit_seed": 953003},
        ],
        "expected_em_executions": 12,
    },
}

EXPECTED_STARTS = [
    {"label": "start_B", "ambiguous_start": "bernoulli"},
    {"label": "start_P", "ambiguous_start": "poisson"},
]

# Total across both stages. Issue #74 caps the whole task at 16.
MAX_TOTAL_EM_EXECUTIONS = 16

REQUIRED_CLAIM_BOUNDARY_KEYS = (
    "score_decides_only", "gate_decides", "do_not_report", "lineage")

# The Phase 9C candidate optimiser a human approved in the Gate 74-B2
# migration, transcribed here rather than imported. Only checked when the
# caller says which optimiser it expects: a run recorded before this field
# existed is audited on its own terms, not retroactively failed.
APPROVED_CANDIDATE_OPTIMIZER = "bfgs"
APPROVED_OPTIMIZER_SETTINGS = {
    "method": "BFGS",
    "jac": "analytic_production_gradient",
    "maxiter": 2000,
    "gtol": 1e-10,
    "finite_difference_jacobian": False,
}
APPROVED_CONVERGENCE_GRAD_INF_TOL = 1e-8

# Gate 74-B5 selected-candidate loading installation, transcribed here rather
# than imported. Checked only when the caller asks for it, for the same reason
# as the optimiser check: an earlier run is audited on its own terms.
APPROVED_LOADING_INSTALLATION = "selected_candidate_loading"
AMBIGUOUS_CANDIDATE_FAMILIES = ("bernoulli", "poisson")
PER_CANDIDATE_PROVENANCE_SUFFIXES = ("optimizer", "n_iter", "converged",
                                     "grad_inf", "scipy_success",
                                     "scipy_status")
INSTALLATION_FIELDS = ("selected_candidate_family", "selected_candidate_score",
                       "selected_loading", "selected_loading_sha256",
                       "loading_installation_policy",
                       "selected_loading_installed",
                       "installed_loading_sha256")


class Finding(dict):
    """One audit finding. A dict so it serialises without ceremony."""

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


def _check_protocol(protocol: dict[str, Any], expected: dict[str, Any],
                    findings: list[Finding]) -> None:
    scalar_fields = ("n", "d", "k_true", "k_fit", "family_y", "L",
                     "refit_num_iter", "exploration_num_iter")
    for field in scalar_fields:
        actual = protocol.get(field)
        if actual != expected[field]:
            findings.append(Finding(
                "BLOCKER", "protocol",
                f"{field}: artifact has {actual!r}, frozen protocol requires "
                f"{expected[field]!r}"))

    for field in ("sigma_x_var", "w0", "w", "f_scale"):
        actual = protocol.get(field)
        if actual is None or not math.isclose(float(actual), expected[field],
                                              rel_tol=1e-12, abs_tol=1e-12):
            findings.append(Finding(
                "BLOCKER", "protocol",
                f"{field}: artifact has {actual!r}, frozen protocol requires "
                f"{expected[field]!r}"))

    if list(protocol.get("family_x_list", [])) != expected["family_x_list"]:
        findings.append(Finding(
            "BLOCKER", "protocol",
            f"family_x_list: artifact has {protocol.get('family_x_list')!r}, "
            f"frozen protocol requires {expected['family_x_list']!r}"))

    actual_reps = protocol.get("replicates", [])
    if len(actual_reps) != len(expected["replicates"]):
        findings.append(Finding(
            "BLOCKER", "protocol",
            f"replicate count: artifact has {len(actual_reps)}, frozen "
            f"protocol requires {len(expected['replicates'])}"))
    else:
        for actual, wanted in zip(actual_reps, expected["replicates"]):
            for key in ("label", "data_seed", "search_seed", "refit_seed"):
                if actual.get(key) != wanted[key]:
                    findings.append(Finding(
                        "BLOCKER", "protocol",
                        f"replicate {wanted['label']} {key}: artifact has "
                        f"{actual.get(key)!r}, frozen protocol requires "
                        f"{wanted[key]!r}"))

    if protocol.get("starts") != EXPECTED_STARTS:
        findings.append(Finding(
            "BLOCKER", "protocol",
            f"starts: artifact has {protocol.get('starts')!r}, frozen "
            f"protocol requires {EXPECTED_STARTS!r}"))

    if protocol.get("expected_em_executions") != expected["expected_em_executions"]:
        findings.append(Finding(
            "BLOCKER", "protocol",
            f"expected_em_executions: artifact has "
            f"{protocol.get('expected_em_executions')!r}, frozen protocol "
            f"requires {expected['expected_em_executions']!r}"))


def _check_runinfo(runinfo: dict[str, Any], expected: dict[str, Any],
                   findings: list[Finding]) -> None:
    executions = runinfo.get("em_executions")
    if executions != expected["expected_em_executions"]:
        findings.append(Finding(
            "BLOCKER", "runinfo",
            f"em_executions: recorded {executions!r}, protocol implies "
            f"{expected['expected_em_executions']!r}"))
    if isinstance(executions, int) and executions > MAX_TOTAL_EM_EXECUTIONS:
        findings.append(Finding(
            "BLOCKER", "runinfo",
            f"em_executions {executions} exceeds the task cap of "
            f"{MAX_TOTAL_EM_EXECUTIONS}"))
    if runinfo.get("failure_policy") != "fail_fast":
        findings.append(Finding(
            "BLOCKER", "runinfo",
            f"failure_policy: recorded {runinfo.get('failure_policy')!r}, "
            f"the protocol requires 'fail_fast'"))
    if runinfo.get("numerics_mode") != "consistent":
        findings.append(Finding(
            "BLOCKER", "runinfo",
            f"numerics_mode: recorded {runinfo.get('numerics_mode')!r}, "
            f"the protocol requires 'consistent'"))
    for field in ("git_sha", "python_version", "started_utc", "finished_utc"):
        if not runinfo.get(field):
            findings.append(Finding(
                "HIGH", "runinfo", f"{field} is missing or empty"))
    if runinfo.get("git_dirty"):
        findings.append(Finding(
            "HIGH", "runinfo",
            "the working tree was dirty at run time; the recorded git_sha "
            "does not fully describe the code that ran"))


def _expected_pairs(expected: dict[str, Any]) -> list[tuple[str, str]]:
    return [(rep["label"], start["label"])
            for rep in expected["replicates"] for start in EXPECTED_STARTS]


def _check_coverage(rows: Sequence[dict[str, str]], keys: Sequence[str],
                    wanted: Iterable[tuple], table: str,
                    findings: list[Finding]) -> None:
    seen: dict[tuple, int] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        seen[key] = seen.get(key, 0) + 1
    wanted_set = set(wanted)
    missing = sorted(wanted_set - set(seen))
    if missing:
        findings.append(Finding(
            "BLOCKER", table, f"missing rows for {missing}"))
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    if duplicates:
        findings.append(Finding(
            "BLOCKER", table, f"duplicate rows for {duplicates}"))
    unexpected = sorted(set(seen) - wanted_set)
    if unexpected:
        findings.append(Finding(
            "HIGH", table, f"unexpected rows for {unexpected}"))


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
                    f"row {index}: {column} is not finite ({row.get(column)!r})"))


def _check_gate_columns(gate_rows: Sequence[dict[str, str]],
                        expected: dict[str, Any],
                        findings: list[Finding]) -> set[tuple[str, str, str]]:
    """Verify the gate verdicts and return the score-decided (rep, start, col)."""

    ambiguous: set[tuple[str, str, str]] = set()
    for row in gate_rows:
        decided_by = row.get("decided_by", "")
        candidates = row.get("candidates", "").split("|")
        true_family = row.get("family_x_true", "")
        if decided_by not in ("gate", "score"):
            findings.append(Finding(
                "BLOCKER", "support_gate",
                f"column {row.get('column')!r}: decided_by is "
                f"{decided_by!r}, expected 'gate' or 'score'"))
            continue
        if decided_by == "score":
            ambiguous.add((row.get("replicate", ""), row.get("start_label", ""),
                           row.get("column", "")))
            if candidates != ["bernoulli", "poisson"]:
                findings.append(Finding(
                    "BLOCKER", "support_gate",
                    f"column {row.get('column')!r}: a score-decided column "
                    f"must offer exactly bernoulli|poisson, got "
                    f"{row.get('candidates')!r}"))
        else:
            if len(candidates) != 1:
                findings.append(Finding(
                    "BLOCKER", "support_gate",
                    f"column {row.get('column')!r}: a gate-decided column "
                    f"must offer exactly one candidate, got "
                    f"{row.get('candidates')!r}"))
            # A Gaussian truth can never be gated to a discrete family.
            if true_family == "gaussian" and candidates != ["gaussian"]:
                findings.append(Finding(
                    "BLOCKER", "support_gate",
                    f"column {row.get('column')!r}: a truly Gaussian column "
                    f"was gated to {candidates!r}"))
    return ambiguous


def _check_candidate_optimizer(protocol: dict[str, Any],
                               score_rows: Sequence[dict[str, str]],
                               expected: str,
                               findings: list[Finding]) -> None:
    """Verify the run used the optimiser the caller says it should have.

    The candidate optimiser is the one intended difference between Gate 74-B
    smoke and smoke-v2, so "only that changed" is worth checking positively:
    the protocol must declare it, the approved settings must be recorded, and
    every scored candidate row must carry that optimiser with a convergence
    flag consistent with the gradient it reached.
    """

    declared = protocol.get("candidate_optimizer")
    if declared != expected:
        findings.append(Finding(
            "BLOCKER", "candidate_optimizer",
            f"protocol.json declares candidate_optimizer {declared!r}, "
            f"expected {expected!r}"))

    if expected == APPROVED_CANDIDATE_OPTIMIZER:
        settings = protocol.get("candidate_optimizer_settings", {}) or {}
        for key, wanted in APPROVED_OPTIMIZER_SETTINGS.items():
            if settings.get(key) != wanted:
                findings.append(Finding(
                    "BLOCKER", "candidate_optimizer",
                    f"candidate optimiser {key} is {settings.get(key)!r}, "
                    f"expected the approved {wanted!r}"))
        if settings.get("fallback_solvers"):
            findings.append(Finding(
                "BLOCKER", "candidate_optimizer",
                f"a fallback solver is recorded: "
                f"{settings.get('fallback_solvers')!r}"))
        rule = protocol.get("candidate_convergence_rule", {}) or {}
        if rule.get("convergence_grad_inf_tol") != \
                APPROVED_CONVERGENCE_GRAD_INF_TOL:
            findings.append(Finding(
                "BLOCKER", "candidate_optimizer",
                f"the candidate convergence threshold is "
                f"{rule.get('convergence_grad_inf_tol')!r}, expected "
                f"{APPROVED_CONVERGENCE_GRAD_INF_TOL!r}"))
        if rule.get("scipy_success_is_criterion") is not False:
            findings.append(Finding(
                "BLOCKER", "candidate_optimizer",
                "SciPy success must not be the convergence criterion"))

    for row in score_rows:
        label = (f"{row.get('replicate')}/{row.get('start_label')}/"
                 f"column {row.get('column')}/{row.get('candidate_family')}")
        if row.get("optimizer") != expected:
            findings.append(Finding(
                "BLOCKER", "candidate_optimizer",
                f"{label}: optimizer is {row.get('optimizer')!r}, expected "
                f"{expected!r}"))
        gradient = _as_float(row.get("optimiser_gradient_inf", ""))
        if gradient is None or not math.isfinite(gradient):
            findings.append(Finding(
                "BLOCKER", "candidate_optimizer",
                f"{label}: gradient infinity norm is "
                f"{row.get('optimiser_gradient_inf')!r}"))
            continue
        if expected == APPROVED_CANDIDATE_OPTIMIZER:
            converged = _truthy(row.get("optimiser_converged", ""))
            should = gradient <= APPROVED_CONVERGENCE_GRAD_INF_TOL
            if converged != should:
                findings.append(Finding(
                    "BLOCKER", "candidate_optimizer",
                    f"{label}: converged={converged} but the gradient "
                    f"{gradient:.3e} implies {should}"))


def _check_installation_provenance(protocol: dict[str, Any],
                                   trace_rows: Sequence[dict[str, str]],
                                   findings: list[Finding]) -> None:
    """Require complete per-candidate and selected-loading evidence.

    Every ambiguous selection-trace row must say, for each candidate, which
    optimiser ran, for how many iterations, whether it converged, the
    gradient it reached and the SciPy status; and it must show that the
    loading installed into F is the winning candidate's own loading. A
    missing field is a BLOCKER: the progression rule needs the evidence, and
    an absent value cannot be read as a passing one.
    """

    declared = protocol.get("ambiguous_loading_installation")
    if declared != APPROVED_LOADING_INSTALLATION:
        findings.append(Finding(
            "BLOCKER", "installation",
            f"protocol.json declares ambiguous_loading_installation "
            f"{declared!r}, expected {APPROVED_LOADING_INSTALLATION!r}"))

    if not trace_rows:
        findings.append(Finding(
            "BLOCKER", "installation",
            "there are no selection-trace rows to carry the evidence"))
    for index, row in enumerate(trace_rows):
        label = (f"trace row {index} ({row.get('start_label')}, iteration "
                 f"{row.get('iteration')}, column {row.get('column')})")
        for family in AMBIGUOUS_CANDIDATE_FAMILIES:
            for suffix in PER_CANDIDATE_PROVENANCE_SUFFIXES:
                key = f"{family}_{suffix}"
                if row.get(key, "") == "":
                    findings.append(Finding(
                        "BLOCKER", "installation",
                        f"{label}: {key} is missing"))
        for key in INSTALLATION_FIELDS:
            if row.get(key, "") == "":
                findings.append(Finding(
                    "BLOCKER", "installation", f"{label}: {key} is missing"))
        if row.get("loading_installation_policy", "") not in (
                "", APPROVED_LOADING_INSTALLATION):
            findings.append(Finding(
                "BLOCKER", "installation",
                f"{label}: loading_installation_policy is "
                f"{row.get('loading_installation_policy')!r}"))
        if row.get("selected_loading_installed", "") != "" and \
                not _truthy(row.get("selected_loading_installed", "")):
            findings.append(Finding(
                "BLOCKER", "installation",
                f"{label}: the selected loading was not installed"))
        selected = row.get("selected_loading_sha256", "")
        installed = row.get("installed_loading_sha256", "")
        if selected and installed and selected != installed:
            findings.append(Finding(
                "BLOCKER", "installation",
                f"{label}: installed loading digest does not match the "
                f"selected candidate's"))
        if row.get("selected_candidate_family", "") and \
                row.get("selected_candidate_family") != row.get("selected_family"):
            findings.append(Finding(
                "BLOCKER", "installation",
                f"{label}: selected_candidate_family "
                f"{row.get('selected_candidate_family')!r} is not the "
                f"selected_family {row.get('selected_family')!r}"))
        for family in AMBIGUOUS_CANDIDATE_FAMILIES:
            gradient = _as_float(row.get(f"{family}_grad_inf", ""))
            if gradient is None or not math.isfinite(gradient):
                continue
            converged = _truthy(row.get(f"{family}_converged", ""))
            if converged != (gradient <= APPROVED_CONVERGENCE_GRAD_INF_TOL):
                findings.append(Finding(
                    "BLOCKER", "installation",
                    f"{label}: {family}_converged={converged} but its "
                    f"gradient {gradient:.3e} implies otherwise"))


def _check_fits(fit_rows: Sequence[dict[str, str]], findings: list[Finding]
                ) -> None:
    for row in fit_rows:
        label = f"{row.get('replicate')}/{row.get('start_label')}"
        if row.get("failure_policy") != "fail_fast":
            findings.append(Finding(
                "BLOCKER", "fit_results",
                f"{label}: failure_policy is {row.get('failure_policy')!r}, "
                f"the protocol requires 'fail_fast'"))
        for counter in ("retry_count", "replacement_count", "seed_rescue_count"):
            value = _as_float(row.get(counter, ""))
            if value is None or value != 0.0:
                findings.append(Finding(
                    "BLOCKER", "fit_results",
                    f"{label}: {counter} is {row.get(counter)!r}, the "
                    f"protocol requires 0"))
        if _truthy(row.get("nan_occurred", "")):
            findings.append(Finding(
                "BLOCKER", "fit_results",
                f"{label}: nan_occurred is true; under fail-fast the run "
                f"should have raised instead of finishing"))
        if _truthy(row.get("q_bic_failed", "")):
            findings.append(Finding(
                "HIGH", "fit_results",
                f"{label}: the Q/BIC computation failed"))


def _check_ledger(rows: Sequence[dict[str, str]], runinfo: dict[str, Any],
                  expected: dict[str, Any], findings: list[Finding]) -> int:
    """The ledger is the record of EM work ATTEMPTED, not work that returned."""

    attempted = len(rows)
    recorded = runinfo.get("em_executions")
    if recorded != attempted:
        findings.append(Finding(
            "BLOCKER", "execution_ledger",
            f"runinfo records {recorded!r} EM executions but the ledger has "
            f"{attempted} attempts; the counter must mean attempts"))
    if attempted > MAX_TOTAL_EM_EXECUTIONS:
        findings.append(Finding(
            "BLOCKER", "execution_ledger",
            f"{attempted} attempted EM executions exceeds the task cap of "
            f"{MAX_TOTAL_EM_EXECUTIONS}"))
    if attempted > expected["expected_em_executions"]:
        findings.append(Finding(
            "BLOCKER", "execution_ledger",
            f"{attempted} attempted EM executions exceeds the "
            f"{expected['expected_em_executions']} this stage implies"))

    sequences = [_as_float(row.get("sequence", "")) for row in rows]
    if sequences != [float(i + 1) for i in range(attempted)]:
        findings.append(Finding(
            "BLOCKER", "execution_ledger",
            f"sequence numbers are not 1..{attempted}: {sequences}"))
    for row in rows:
        status = row.get("status", "")
        if status not in ("STARTED", "SUCCESS", "FAILED"):
            findings.append(Finding(
                "BLOCKER", "execution_ledger",
                f"sequence {row.get('sequence')!r}: status is {status!r}"))
        if row.get("execution_kind") not in ("exploration", "refit"):
            findings.append(Finding(
                "BLOCKER", "execution_ledger",
                f"sequence {row.get('sequence')!r}: execution_kind is "
                f"{row.get('execution_kind')!r}"))
        if not row.get("started_utc"):
            findings.append(Finding(
                "HIGH", "execution_ledger",
                f"sequence {row.get('sequence')!r}: no start timestamp"))
    return attempted


def _check_failure_evidence(failure: dict[str, Any], attempted: int,
                            findings: list[Finding]) -> None:
    missing = [key for key in REQUIRED_FAILURE_KEYS if key not in failure]
    if missing:
        findings.append(Finding(
            "BLOCKER", "failure",
            f"failure.json is missing {missing}"))
    if failure.get("attempted_em_executions") != attempted:
        findings.append(Finding(
            "BLOCKER", "failure",
            f"failure.json records "
            f"{failure.get('attempted_em_executions')!r} attempted "
            f"executions but the ledger has {attempted}"))
    for counter in ("retry_count", "replacement_count", "seed_rescue_count"):
        if failure.get(counter) != 0:
            findings.append(Finding(
                "BLOCKER", "failure",
                f"failure.json records {counter}={failure.get(counter)!r}; "
                f"a failed stage must not have retried or reseeded"))
    if not failure.get("git_sha"):
        findings.append(Finding("HIGH", "failure", "git_sha is missing"))


def _check_approvals(protocol: dict[str, Any],
                     findings: list[Finding]) -> None:
    """Every condition Issue #74 did not freeze needs a recorded approval."""

    approvals = protocol.get("human_approvals")
    if not approvals:
        findings.append(Finding(
            "HIGH", "human_approvals",
            "protocol.json records no human approval for the conditions "
            "Issue #74 left unfrozen (exploration_num_iter)"))
        return
    for approval in approvals:
        name = approval.get("parameter")
        if not approval.get("approved"):
            findings.append(Finding(
                "HIGH", "human_approvals",
                f"{name!r} was set to {approval.get('value')!r} without a "
                f"recorded human approval; Issue #74 did not freeze it"))
        elif not approval.get("approved_by") or not approval.get("approved_in"):
            findings.append(Finding(
                "HIGH", "human_approvals",
                f"{name!r} is marked approved but does not say who approved "
                f"it or where"))


def audit(run_dir: Path,
          expect_candidate_optimizer: str | None = None,
          require_installation_provenance: bool = False) -> dict[str, Any]:
    """Audit a finished run directory and return the report.

    ``expect_candidate_optimizer`` opts into the Gate 74-B3 check that the run
    used a named candidate optimiser. It is off by default so that a run
    recorded before that field existed is audited on its own terms rather than
    retroactively failed.
    """

    findings: list[Finding] = []

    missing = [name for name in ALWAYS_REQUIRED_ARTIFACTS
               if not (run_dir / name).is_file()]
    if missing:
        findings.append(Finding(
            "BLOCKER", "artifacts",
            f"missing artifacts that must exist before any EM runs: {missing}"))
        return _finish(run_dir, None, findings)

    protocol = _read_json(run_dir / "protocol.json")
    runinfo = _read_json(run_dir / "runinfo.json")

    stage = protocol.get("stage")
    if stage not in EXPECTED:
        findings.append(Finding(
            "BLOCKER", "protocol",
            f"stage is {stage!r}; this auditor knows {sorted(EXPECTED)}"))
        return _finish(run_dir, stage, findings)
    expected = EXPECTED[stage]

    _check_protocol(protocol, expected, findings)
    _check_approvals(protocol, findings)

    ledger_rows = _read_csv(run_dir / "execution_ledger.csv")
    attempted = _check_ledger(ledger_rows, runinfo, expected, findings)

    run_status = runinfo.get("run_status", "UNKNOWN")
    failure_path = run_dir / FAILURE_ARTIFACT
    if failure_path.is_file() or run_status == "FAILED":
        # A partial run: audit the EVIDENCE, not the missing results.
        if not failure_path.is_file():
            findings.append(Finding(
                "BLOCKER", "failure",
                "the run is marked FAILED but no failure.json was preserved"))
        else:
            _check_failure_evidence(_read_json(failure_path), attempted,
                                    findings)
        if run_status != "FAILED":
            findings.append(Finding(
                "BLOCKER", "runinfo",
                f"failure.json exists but run_status is {run_status!r}"))
        if attempted >= expected["expected_em_executions"]:
            findings.append(Finding(
                "HIGH", "failure",
                f"a failed run attempted {attempted} executions, which is not "
                f"fewer than the {expected['expected_em_executions']} a "
                f"complete run implies"))
        report = _finish(run_dir, stage, findings, run_status="FAILED")
        report["attempted_em_executions"] = attempted
        report["convergence_gate"] = "NOT_EVALUATED"
        _write_report(run_dir, report)
        return report

    if run_status != "SUCCESS":
        findings.append(Finding(
            "HIGH", "runinfo",
            f"run_status is {run_status!r}; a completed run should say "
            f"SUCCESS"))

    still_missing = [name for name in COMPLETED_RUN_ARTIFACTS
                     if not (run_dir / name).is_file()]
    if still_missing:
        findings.append(Finding(
            "BLOCKER", "artifacts",
            f"missing required artifacts: {still_missing}"))
        return _finish(run_dir, stage, findings, run_status=run_status)

    summary = _read_json(run_dir / "summary.json")
    _check_runinfo(runinfo, expected, findings)

    provenance = _read_csv(run_dir / "generator_provenance.csv")
    gate_rows = _read_csv(run_dir / "support_gate.csv")
    score_rows = _read_csv(run_dir / "family_scores.csv")
    trace_rows = _read_csv(run_dir / "selection_trace.csv")
    fit_rows = _read_csv(run_dir / "fit_results.csv")

    replicate_labels = [rep["label"] for rep in expected["replicates"]]
    pairs = _expected_pairs(expected)

    _check_coverage(
        provenance, ("replicate", "column"),
        [(label, str(column)) for label in replicate_labels
         for column in range(expected["d"])],
        "generator_provenance", findings)
    _check_coverage(
        gate_rows, ("replicate", "start_label", "column"),
        [(rep, start, str(column)) for rep, start in pairs
         for column in range(expected["d"])],
        "support_gate", findings)
    _check_coverage(
        fit_rows, ("replicate", "start_label"), pairs, "fit_results", findings)

    ambiguous = _check_gate_columns(gate_rows, expected, findings)

    # Every score row must belong to a score-decided column, and every
    # score-decided column must have been scored on both candidates.
    score_keys = {(row.get("replicate", ""), row.get("start_label", ""),
                   row.get("column", "")) for row in score_rows}
    for key in sorted(score_keys - ambiguous):
        findings.append(Finding(
            "BLOCKER", "family_scores",
            f"{key}: a score row exists for a column the gate decided; "
            f"gate-decided columns must never be scored"))
    for key in sorted(ambiguous - score_keys):
        findings.append(Finding(
            "BLOCKER", "family_scores",
            f"{key}: a score-decided column has no candidate scores"))
    per_column_candidates: dict[tuple, set[str]] = {}
    for row in score_rows:
        key = (row.get("replicate", ""), row.get("start_label", ""),
               row.get("column", ""))
        per_column_candidates.setdefault(key, set()).add(
            row.get("candidate_family", ""))
    for key, families in sorted(per_column_candidates.items()):
        if families != {"bernoulli", "poisson"}:
            findings.append(Finding(
                "BLOCKER", "family_scores",
                f"{key}: scored candidates are {sorted(families)}, expected "
                f"bernoulli and poisson"))

    for key in sorted({(row.get("replicate", ""), row.get("start_label", ""),
                        row.get("column", "")) for row in trace_rows} - ambiguous):
        findings.append(Finding(
            "BLOCKER", "selection_trace",
            f"{key}: a trace row exists for a column the gate decided"))
    for row in trace_rows:
        iteration = _as_float(row.get("iteration", ""))
        if iteration is None or not (1 <= iteration <= expected["exploration_num_iter"]):
            findings.append(Finding(
                "BLOCKER", "selection_trace",
                f"iteration {row.get('iteration')!r} is outside "
                f"1..{expected['exploration_num_iter']}"))

    _check_finite(score_rows, "family_scores",
                  ("score", "neg2_score", "loading_norm"), findings)
    _check_finite(trace_rows, "selection_trace",
                  ("margin_neg2", "score_bernoulli", "score_poisson"), findings)
    _check_finite(fit_rows, "fit_results", ("Q_strict", "bic"), findings)
    _check_fits(fit_rows, findings)

    # Candidate-convergence gate: an execution-integrity condition, not an
    # accuracy threshold.  Recomputed here from the artifact rather than
    # trusted from summary.json.
    #
    # It has to read BOTH tables. family_scores.csv holds the final
    # iteration's candidates, but the A-type update selects a family at every
    # EM iteration and each of those decisions feeds the next E-step, so a
    # candidate that failed to converge in an earlier iteration still shaped
    # the result. selection_trace.csv is where those iterations are recorded.
    non_converged = [row for row in score_rows
                     if row.get("optimiser_converged", "") != ""
                     and not _truthy(row.get("optimiser_converged", ""))]
    non_converged_iterations = [
        row for row in trace_rows
        if row.get("all_candidates_converged", "") != ""
        and not _truthy(row.get("all_candidates_converged", ""))]
    recomputed_gate = ("BLOCKED_FOR_PILOT"
                       if (non_converged or non_converged_iterations)
                       else "READY_FOR_PILOT")
    reported_gate = (summary.get("convergence_gate", {}) or {}).get("status")
    if reported_gate != recomputed_gate:
        findings.append(Finding(
            "BLOCKER", "convergence_gate",
            f"summary.json reports {reported_gate!r} but the artifact rows "
            f"imply {recomputed_gate!r}"))

    if expect_candidate_optimizer is not None:
        _check_candidate_optimizer(protocol, score_rows,
                                   expect_candidate_optimizer, findings)
    if require_installation_provenance:
        _check_installation_provenance(protocol, trace_rows, findings)

    claim_boundary = summary.get("claim_boundary", {}) or {}
    for key in REQUIRED_CLAIM_BOUNDARY_KEYS:
        if key not in claim_boundary:
            findings.append(Finding(
                "HIGH", "summary",
                f"claim_boundary is missing {key!r}"))

    report = _finish(run_dir, stage, findings, run_status=run_status)
    report["convergence_gate"] = recomputed_gate
    report["non_converged_iteration_rows"] = len(non_converged_iterations)
    # The composite gate. An artifact audit says the run is clean evidence;
    # the convergence gate says the frozen score was actually optimised.  The
    # next EM stage needs BOTH, and automation must read this field rather
    # than progress_eligible, which answers only the first question.
    report["pilot_progress_eligible"] = bool(
        report["progress_eligible"] and recomputed_gate == "READY_FOR_PILOT")
    report["pilot_progression_rule"] = ("progress_eligible and "
                                        "convergence_gate == READY_FOR_PILOT")
    report["expected_candidate_optimizer"] = expect_candidate_optimizer
    report["installation_provenance_required"] = bool(
        require_installation_provenance)
    report["attempted_em_executions"] = attempted
    report["non_converged_candidate_rows"] = len(non_converged)
    report["score_decided_columns"] = len(ambiguous)
    report["gate_decided_columns"] = len(gate_rows) - len(ambiguous)
    _write_report(run_dir, report)
    return report


def _finish(run_dir: Path, stage: str | None,
            findings: Sequence[Finding],
            run_status: str = "UNKNOWN") -> dict[str, Any]:
    """Assemble the report.

    Stage progression requires BLOCKER = 0 AND HIGH = 0.  A HIGH finding --
    a failed Q/BIC computation, a dirty worktree, rows nothing should have
    produced -- means the run is not the clean evidence the next stage would
    be built on, even though the run itself did not violate the protocol.
    Reporting only a PASS/FAIL verdict would let such a run wave the next
    stage through.
    """

    counts = {severity: sum(1 for f in findings if f["severity"] == severity)
              for severity in ("BLOCKER", "HIGH", "MEDIUM")}
    # A run that did not complete is never eligible, however tidy its
    # evidence is: preserving a failure correctly is not the same as having
    # produced the result the next stage would be built on.
    progress_eligible = (counts["BLOCKER"] == 0 and counts["HIGH"] == 0
                         and run_status == "SUCCESS")
    report = {
        "auditor_version": AUDITOR_VERSION,
        "run_dir": str(run_dir),
        "stage": stage,
        "run_status": run_status,
        "verdict": "FAIL" if counts["BLOCKER"] else "PASS",
        "blocker_count": counts["BLOCKER"],
        "high_count": counts["HIGH"],
        "medium_count": counts["MEDIUM"],
        "finding_count": len(findings),
        "progress_eligible": progress_eligible,
        "stage_progression": "ALLOWED" if progress_eligible else "BLOCKED",
        # Overwritten for a completed run once the convergence gate is known;
        # a run that never got that far can never be eligible.
        "pilot_progress_eligible": False,
        "pilot_progression_rule": "progress_eligible and "
                                  "convergence_gate == READY_FOR_PILOT",
        "expected_candidate_optimizer": None,
        "progression_rule": "blocker_count == 0 and high_count == 0 "
                            "and run_status == SUCCESS",
        "findings": list(findings),
        "note": "A FAIL or a blocked progression is a finding to hand to a "
                "human. Do not rerun, reseed, or widen the optimiser budget "
                "in response to it.",
    }
    if stage is None or not progress_eligible:
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
        description="Audit an Issue #74 family-selection run from its artifacts.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expect-candidate-optimizer", default=None,
                        help="require the run to have used this candidate "
                             "optimizer (Gate 74-B3 uses 'bfgs')")
    parser.add_argument("--require-installation-provenance",
                        action="store_true",
                        help="require complete per-candidate provenance and "
                             "Gate 74-B5 selected-loading installation "
                             "evidence on every selection-trace row")
    args = parser.parse_args(argv)

    report = audit(args.run_dir, args.expect_candidate_optimizer,
                   args.require_installation_provenance)
    print(f"verdict={report['verdict']} run_status={report['run_status']} "
          f"blockers={report['blocker_count']} high={report['high_count']} "
          f"medium={report['medium_count']} "
          f"progression={report['stage_progression']} "
          f"gate={report.get('convergence_gate', 'n/a')} "
          f"pilot_progress_eligible={report['pilot_progress_eligible']}")
    for finding in report["findings"]:
        print(f"  [{finding['severity']}] {finding['check']}: {finding['message']}")
    # Exit 0 only when the NEXT EM STAGE may proceed. A HIGH finding blocks it
    # even though the verdict itself is PASS, and so does a blocked
    # convergence gate even though the artifacts are clean.
    return 0 if report["pilot_progress_eligible"] else 1


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
