"""Gate 74-B6 differential preflight: is smoke-v3 the same experiment?

smoke-v3 re-runs the frozen C1 smoke. Relative to smoke-v2 (Gate 74-B3) the
only intended change is the Gate 74-B5 selected-candidate loading
installation for ambiguous exploration rows. The candidate optimiser stays the
approved analytic-gradient BFGS.

This preflight therefore does two things before a single EM call:

1. it runs every frozen-C1 check the smoke-v2 preflight runs -- n, d, K, the
   family pattern, family_y, the dispersion, w0, w, f_scale, L, both num_iter
   values, all three seeds, the starts and their ambiguous initial
   assignments, the generator version and RNG order, and the approved BFGS
   configuration and convergence rule -- reusing those checks, whose frozen
   values are transcribed literals rather than anything read from an earlier
   run's artifact; and
2. it requires, positively, that the runner declares the B5 installation
   semantics, and that the selector actually installs the winning loading on
   a deterministic fixed-Z probe. A declaration alone would be a promise; the
   probe checks the behaviour.

Zero EM. The probe uses the deterministic Gate 74-B1 posterior-sample block,
never an E-step.
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

import preflight_smoke_v2 as v2                               # noqa: E402

PREFLIGHT_VERSION = "smoke-v3-differential-preflight-v1"
GATE = "74-B6"

# The one approved difference from smoke-v2.
APPROVED_LOADING_INSTALLATION = "selected_candidate_loading"
ALLOWED_DIFFERENCE_FROM_SMOKE_V2 = ("ambiguous_loading_installation",)


def check_installation_declaration(payload: dict[str, Any]
                                   ) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    declared = payload.get("ambiguous_loading_installation")
    if declared != APPROVED_LOADING_INSTALLATION:
        findings.append({
            "field": "ambiguous_loading_installation",
            "actual": declared,
            "expected": APPROVED_LOADING_INSTALLATION,
            "message": f"ambiguous_loading_installation: runner has "
                       f"{declared!r}, the approved B5 semantics is "
                       f"{APPROVED_LOADING_INSTALLATION!r}",
        })
    return findings


def probe_installation_behaviour() -> list[dict[str, Any]]:
    """Check on fixed inputs that the winning loading is what gets installed."""

    import numpy as np

    import family_selection as fs
    import validate_candidate_optimizer as b1

    findings: list[dict[str, Any]] = []
    Z_samples = b1.build_z_samples()
    X = np.column_stack([b1.build_case_column("case_A_balanced_nonseparable"),
                         np.linspace(-1.5, 1.5, 24)])
    model = fs.FamilySelectingPerColumnLSM(
        gates=fs.support_gate(X), n=24, d=2, k=3, L=5,
        family_x_list=["bernoulli", "gaussian"], family_y="bernoulli")
    model.initialize_params(true_params=None, seed=7)
    F = model.calc_F(X, Z_samples)

    rows = [row for row in model.selection_trace if row["column"] == 0]
    if len(rows) != 1:
        findings.append({"field": "probe", "actual": len(rows), "expected": 1,
                         "message": "the probe did not produce one trace row "
                                    "for its ambiguous column"})
        return findings
    row = rows[0]
    installed = fs.loading_digest(F[0, :])
    checks = (
        ("probe.selected_loading_installed",
         row.get("selected_loading_installed"), True),
        ("probe.installed_equals_selected",
         row.get("installed_loading_sha256") == row.get("selected_loading_sha256"),
         True),
        ("probe.installed_equals_F_row",
         row.get("installed_loading_sha256") == installed, True),
        ("probe.loading_installation_policy",
         row.get("loading_installation_policy"),
         APPROVED_LOADING_INSTALLATION),
        ("probe.candidate_optimizer", row.get("candidate_optimizer"), "bfgs"),
    )
    for field, actual, expected in checks:
        if actual != expected:
            findings.append({"field": field, "actual": actual,
                             "expected": expected,
                             "message": f"{field}: {actual!r} != {expected!r}"})
    return findings


def run_preflight() -> dict[str, Any]:
    """Frozen-C1 checks, plus the B5 semantics. No EM."""

    import run_family_selection_pilot as runner
    from family_selection import initial_assignment, support_gate

    protocol = runner.SMOKE
    payload = protocol.as_json()

    c1_findings = v2.check_protocol(payload)
    replicate = protocol.replicates[0]
    dataset = runner.build_dataset(protocol, replicate)
    generator_findings = v2.check_generator(dataset.metadata)
    gates = support_gate(dataset.X)
    assignments = {label: initial_assignment(gates, family)
                   for label, family in runner.STARTS}
    start_findings = v2.check_start_assignments(gates, assignments)

    installation_findings = check_installation_declaration(payload)
    probe_findings = probe_installation_behaviour()

    findings = (c1_findings + generator_findings + start_findings
                + installation_findings + probe_findings)
    return {
        "preflight_version": PREFLIGHT_VERSION,
        "gate": GATE,
        "em_executions": 0,
        "reads_historical_smoke": False,
        "frozen_c1": dict(v2.FROZEN_C1),
        "approved_candidate_optimizer": v2.APPROVED_CANDIDATE_OPTIMIZER,
        "approved_difference_from_smoke_v2": {
            "fields": list(ALLOWED_DIFFERENCE_FROM_SMOKE_V2),
            "ambiguous_loading_installation": APPROVED_LOADING_INSTALLATION,
        },
        "observed": {
            "candidate_optimizer": payload.get("candidate_optimizer"),
            "candidate_optimizer_settings":
                payload.get("candidate_optimizer_settings"),
            "ambiguous_loading_installation":
                payload.get("ambiguous_loading_installation"),
            "generator_version": dataset.metadata.get("generator_version"),
            "ambiguous_columns": [g.column for g in gates if g.is_ambiguous],
            "start_assignments": {label: list(value)
                                  for label, value in assignments.items()},
        },
        "findings": findings,
        "status": "PREFLIGHT_PASS" if not findings else "PREFLIGHT_FAIL",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Zero-EM differential preflight for Gate 74-B6 smoke-v3.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_preflight()
    print(f"status={report['status']} findings={len(report['findings'])} "
          f"em_executions={report['em_executions']}")
    for finding in report["findings"]:
        print(f"  DIFFERENCE: {finding['message']}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False,
                      default=str)
            handle.write("\n")
    return 0 if report["status"] == "PREFLIGHT_PASS" else 1


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
