"""Gate 74-B3 differential preflight: is smoke-v2 the same experiment?

Gate 74-B3 re-runs the frozen C1 smoke with exactly one intended change: the
Phase 9C ambiguous-column candidate optimiser moved from the historical
50-step Adam to the analytic-gradient BFGS configuration Gate 74-B2 certified.
"Exactly one change" is a claim about every other field, so this module checks
them before a single EM call is made.

The frozen C1 values below are transcribed independently from the Issue #74
protocol.  They are deliberately NOT read from ``smoke_20260923``: comparing a
run against the artifact of an earlier run would accept any drift the two
happened to share, and would make a preserved historical artifact an input to
a new execution.

The only semantic difference this preflight permits is candidate-optimiser
provenance, settings and convergence rule.  Everything else -- n, d, K, the
family pattern, family_y, the dispersion, w0, w, f_scale, L, both num_iter
values, all three seeds, the start labels and their ambiguous initial
assignments, the generator version and its RNG consumption order -- must match.

Zero EM.  Run it, read it, and only then run smoke-v2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

PREFLIGHT_VERSION = "smoke-v2-differential-preflight-v1"
GATE = "74-B3"

# --------------------------------------------------------------------------
# Frozen C1, transcribed from the Issue #74 protocol. Not read from any
# artifact directory.
# --------------------------------------------------------------------------

FROZEN_C1: dict[str, Any] = {
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
    "f_scale": 1.0,
    "L": 5,
    "exploration_num_iter": 8,
    "refit_num_iter": 8,
    "data_seed": 941001,
    "search_seed": 942001,
    "refit_seed": 943001,
    "starts": [{"label": "start_B", "ambiguous_start": "bernoulli"},
               {"label": "start_P", "ambiguous_start": "poisson"}],
    "expected_em_executions": 4,
    "generator_version": "canonical-clean-mixed-v1",
    "rng_consumption_order": ["Z", "F", "X_columns_ascending", "Y"],
}

# The one approved difference.
APPROVED_CANDIDATE_OPTIMIZER = "bfgs"
APPROVED_OPTIMIZER_SETTINGS = {
    "method": "BFGS",
    "jac": "analytic_production_gradient",
    "maxiter": 2000,
    "gtol": 1e-10,
    "finite_difference_jacobian": False,
}
APPROVED_CONVERGENCE_GRAD_INF_TOL = 1e-8

# Fields whose difference is allowed, and only these.
ALLOWED_DIFFERENCE_FIELDS = (
    "candidate_optimizer",
    "candidate_optimizer_settings",
    "candidate_convergence_rule",
)


class PreflightStop(RuntimeError):
    """Raised when smoke-v2 is not the frozen C1 experiment."""


def _compare(field: str, actual: Any, expected: Any,
             findings: list[dict[str, Any]]) -> None:
    if actual != expected:
        findings.append({
            "field": field,
            "actual": actual,
            "expected": expected,
            "message": f"{field}: runner has {actual!r}, frozen C1 requires "
                       f"{expected!r}",
        })


def check_protocol(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare a runner protocol payload against the frozen C1 literals."""

    findings: list[dict[str, Any]] = []

    for field in ("n", "d", "k_true", "k_fit", "family_y", "sigma_x_var",
                  "w0", "w", "f_scale", "L", "exploration_num_iter",
                  "refit_num_iter", "expected_em_executions"):
        _compare(field, payload.get(field), FROZEN_C1[field], findings)

    _compare("family_x_list", list(payload.get("family_x_list", [])),
             FROZEN_C1["family_x_list"], findings)
    _compare("starts", payload.get("starts"), FROZEN_C1["starts"], findings)

    replicates = payload.get("replicates", [])
    if len(replicates) != 1:
        findings.append({
            "field": "replicates",
            "actual": len(replicates),
            "expected": 1,
            "message": f"frozen C1 has one replicate, runner has "
                       f"{len(replicates)}",
        })
    else:
        replicate = replicates[0]
        _compare("data_seed", replicate.get("data_seed"),
                 FROZEN_C1["data_seed"], findings)
        _compare("search_seed", replicate.get("search_seed"),
                 FROZEN_C1["search_seed"], findings)
        _compare("refit_seed", replicate.get("refit_seed"),
                 FROZEN_C1["refit_seed"], findings)

    # The one approved difference, checked positively: it must be present and
    # it must be the approved configuration, not merely "something else".
    _compare("candidate_optimizer", payload.get("candidate_optimizer"),
             APPROVED_CANDIDATE_OPTIMIZER, findings)
    settings = payload.get("candidate_optimizer_settings", {})
    for key, wanted in APPROVED_OPTIMIZER_SETTINGS.items():
        _compare(f"candidate_optimizer_settings.{key}", settings.get(key),
                 wanted, findings)
    if settings.get("fallback_solvers"):
        findings.append({
            "field": "candidate_optimizer_settings.fallback_solvers",
            "actual": settings.get("fallback_solvers"),
            "expected": [],
            "message": "a fallback solver is configured",
        })
    rule = payload.get("candidate_convergence_rule", {})
    _compare("candidate_convergence_rule.convergence_grad_inf_tol",
             rule.get("convergence_grad_inf_tol"),
             APPROVED_CONVERGENCE_GRAD_INF_TOL, findings)
    _compare("candidate_convergence_rule.scipy_success_is_criterion",
             rule.get("scipy_success_is_criterion"), False, findings)

    return findings


def check_generator(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare a generated dataset's provenance against the frozen C1 draw."""

    findings: list[dict[str, Any]] = []
    _compare("generator_version", metadata.get("generator_version"),
             FROZEN_C1["generator_version"], findings)
    _compare("rng_consumption_order",
             list(metadata.get("rng_consumption_order", [])),
             FROZEN_C1["rng_consumption_order"], findings)
    _compare("generator.family_x_list",
             list(metadata.get("family_x_list", [])),
             FROZEN_C1["family_x_list"], findings)
    _compare("generator.family_y", metadata.get("family_y"),
             FROZEN_C1["family_y"], findings)
    _compare("generator.n", metadata.get("n"), FROZEN_C1["n"], findings)
    _compare("generator.d", metadata.get("d"), FROZEN_C1["d"], findings)
    _compare("generator.K_true", metadata.get("K_true"),
             FROZEN_C1["k_true"], findings)
    _compare("generator.seed", metadata.get("seed"),
             FROZEN_C1["data_seed"], findings)
    _compare("generator.f_scale", metadata.get("f_scale"),
             FROZEN_C1["f_scale"], findings)
    _compare("generator.w0", metadata.get("w0"), FROZEN_C1["w0"], findings)
    _compare("generator.w", metadata.get("w"), FROZEN_C1["w"], findings)
    _compare("generator.normalization_policy",
             metadata.get("normalization_policy"), "none", findings)
    _compare("generator.link_policy", metadata.get("link_policy"),
             "canonical_no_clipping_fail_fast", findings)
    variances = metadata.get("sigma_x_var", [])
    gaussian_columns = [index for index, family
                        in enumerate(FROZEN_C1["family_x_list"])
                        if family == "gaussian"]
    for index in gaussian_columns:
        recorded = variances[index] if index < len(variances) else None
        _compare(f"generator.sigma_x_var[{index}]", recorded,
                 FROZEN_C1["sigma_x_var"], findings)
    return findings


def check_start_assignments(gates: Sequence[Any],
                            assignments: dict[str, Sequence[str]]
                            ) -> list[dict[str, Any]]:
    """The two starts must differ only on the gate-ambiguous columns."""

    findings: list[dict[str, Any]] = []
    ambiguous = [gate.column for gate in gates if gate.is_ambiguous]
    fixed = [gate.column for gate in gates if not gate.is_ambiguous]
    for start in FROZEN_C1["starts"]:
        label = start["label"]
        assignment = list(assignments.get(label, []))
        if len(assignment) != FROZEN_C1["d"]:
            findings.append({
                "field": f"{label}.length", "actual": len(assignment),
                "expected": FROZEN_C1["d"],
                "message": f"{label}: assignment has {len(assignment)} "
                           f"entries",
            })
            continue
        for column in ambiguous:
            _compare(f"{label}.column{column}", assignment[column],
                     start["ambiguous_start"], findings)
        for column in fixed:
            _compare(f"{label}.column{column}", assignment[column],
                     gates[column].candidates[0], findings)
    return findings


def run_preflight() -> dict[str, Any]:
    """Build the frozen C1 protocol and data, and check both. No EM."""

    import run_family_selection_pilot as runner
    from family_selection import initial_assignment, support_gate

    protocol = runner.SMOKE
    payload = protocol.as_json()
    protocol_findings = check_protocol(payload)

    # Drawing the dataset consumes no EM and is how the generator provenance
    # becomes checkable at all.
    replicate = protocol.replicates[0]
    dataset = runner.build_dataset(protocol, replicate)
    generator_findings = check_generator(dataset.metadata)

    gates = support_gate(dataset.X)
    assignments = {label: initial_assignment(gates, family)
                   for label, family in runner.STARTS}
    start_findings = check_start_assignments(gates, assignments)

    findings = protocol_findings + generator_findings + start_findings
    return {
        "preflight_version": PREFLIGHT_VERSION,
        "gate": GATE,
        "em_executions": 0,
        "reads_historical_smoke": False,
        "frozen_c1": dict(FROZEN_C1),
        "approved_difference": {
            "candidate_optimizer": APPROVED_CANDIDATE_OPTIMIZER,
            "settings": dict(APPROVED_OPTIMIZER_SETTINGS),
            "convergence_grad_inf_tol": APPROVED_CONVERGENCE_GRAD_INF_TOL,
            "allowed_difference_fields": list(ALLOWED_DIFFERENCE_FIELDS),
        },
        "observed": {
            "candidate_optimizer": payload.get("candidate_optimizer"),
            "candidate_optimizer_settings":
                payload.get("candidate_optimizer_settings"),
            "candidate_convergence_rule":
                payload.get("candidate_convergence_rule"),
            "generator_version": dataset.metadata.get("generator_version"),
            "ambiguous_columns": [gate.column for gate in gates
                                  if gate.is_ambiguous],
            "start_assignments": {label: list(value)
                                  for label, value in assignments.items()},
        },
        "findings": findings,
        "status": "PREFLIGHT_PASS" if not findings else "PREFLIGHT_FAIL",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Zero-EM differential preflight for Gate 74-B3 smoke-v2.")
    parser.add_argument("--out", type=Path, default=None,
                        help="optional path to write the preflight report")
    args = parser.parse_args(argv)

    report = run_preflight()
    print(f"status={report['status']} findings={len(report['findings'])} "
          f"em_executions={report['em_executions']}")
    for finding in report["findings"]:
        print(f"  DIFFERENCE: {finding['message']}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"report written to {args.out}")
    return 0 if report["status"] == "PREFLIGHT_PASS" else 1


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
