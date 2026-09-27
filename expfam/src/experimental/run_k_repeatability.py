"""Phase 9E (Issue #81): K-selection repeatability under the Phase 9D condition.

Reuses the technically validated Phase 9D joint family + K runner unchanged in
substance; only the scientific differences are set here:

- 20 prospectively frozen new datasets, rep01..rep20, seeds
  971000+r / 972000+r / 973000+r (data / search / refit)
- start_B only (Phase 9D showed start_B and start_P reach identical C_Q)
- dataset-level failure scope: an isolated numerical failure marks that
  dataset INCOMPLETE, is recorded, is never replaced, and the run continues

Primary outputs (Issue #81 ``PHASE9E_RESEARCH_FIRST_REPEATABILITY_FROZEN``):
P1 the K_hat distribution over completed datasets; P2
``delta_23 = C_Q(3) - C_Q(2)``; P3 ``fit_gain_23 = D_2 - D_3`` and
``penalty_increase_23 = [P_Z(3)+P_theta(3)] - [P_Z(2)+P_theta(2)]``, with
``delta_23 = penalty_increase_23 - fit_gain_23``; P4 family recovery at K_hat
(secondary). C_Q is the existing direct criterion (Q-based complete-data /
ICL-type, not Schwarz BIC); nothing here changes it.

Usage (one run, fresh directory)::

    python run_k_repeatability.py --out <fresh directory>
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import audit_family_selection_pilot as audit74                    # noqa: E402
import run_family_selection_pilot as pilot                         # noqa: E402
import run_joint_family_k_selection as joint                       # noqa: E402

WRAPPER_VERSION = "k-repeatability-v1"

PROTOCOL = dataclasses.replace(
    joint.PROTOCOL,
    stage="repeatability",
    replicates=tuple(joint.Replicate(f"rep{r:02d}", 971000 + r, 972000 + r,
                                     973000 + r) for r in range(1, 21)),
    starts=(("start_B", "bernoulli"),),
    failure_scope="dataset",
)

AUTHORIZATION: dict[str, Any] = {
    "gate": "Phase 9E bounded repeatability task",
    "authorized": True,
    "authorized_by": "Human",
    "authorized_in": "Issue #81 PHASE9E_RESEARCH_FIRST_REPEATABILITY_FROZEN "
                     "(Issue body; clarification comment 5856097319)",
}


def delta_23(cq: dict[int, float]) -> float:
    """C_Q(3) - C_Q(2): < 0 prefers K=3, > 0 prefers K=2, 0 -> K=2 (tie rule)."""

    return float(cq[3]) - float(cq[2])


def dataset_rows(cq_rows: Sequence[dict[str, Any]],
                 decomposition_rows: Sequence[dict[str, Any]],
                 joint_rows: Sequence[dict[str, Any]],
                 family_rows: Sequence[dict[str, Any]],
                 completed: Sequence[str]) -> list[dict[str, Any]]:
    """One row per COMPLETED dataset. Incomplete datasets are not in here;
    they are reported separately and never replaced."""

    out = []
    for label in completed:
        cq = {int(r["k"]): float(r["C_Q"]) for r in cq_rows
              if r["replicate"] == label}
        dec = {int(r["k"]): r for r in decomposition_rows
               if r["replicate"] == label}
        path = next(r for r in joint_rows if r["replicate"] == label)
        k_hat = joint.select_k_hat(cq)
        fit_gain = float(dec[2]["D_K"]) - float(dec[3]["D_K"])
        penalty = ((float(dec[3]["P_Z"]) + float(dec[3]["P_theta"]))
                   - (float(dec[2]["P_Z"]) + float(dec[2]["P_theta"])))
        d23 = delta_23(cq)
        cols = [int(c) for c in str(path["score_decided_columns"]).split("|")]
        bernoulli_all_k = all(
            r["selected_family"] == "bernoulli" for r in family_rows
            if r["replicate"] == label and int(r["column"]) in cols)
        seeds = next(r for r in cq_rows if r["replicate"] == label)
        out.append({
            "dataset": label,
            "data_seed": seeds["data_seed"],
            "search_seed": seeds["search_seed"],
            "refit_seed": seeds["refit_seed"],
            "k_hat": k_hat,
            "k_category": joint.k_category(k_hat, PROTOCOL.k_true),
            **{f"C_Q_k{k}": cq[k] for k in sorted(cq)},
            "delta_23": d23,
            "fit_gain_23": fit_gain,
            "penalty_increase_23": penalty,
            "identity_residual": d23 - (penalty - fit_gain),
            "runner_up_gap": joint.runner_up_gap(cq),
            "cq_ordering": "|".join(map(str, joint.cq_ordering(cq))),
            **{f"{part}_k{k}": float(dec[k][part])
               for k in sorted(dec) for part in ("D_K", "P_Z", "P_theta")},
            "selected_families_at_k_hat": path["selected_families_at_k_hat"],
            "bernoulli_to_poisson_at_k_hat":
                int(path["bernoulli_to_poisson_at_k_hat"]),
            "margins_at_k_hat": path["margins_at_k_hat"],
            "cols_3_8_bernoulli_at_every_k": bernoulli_all_k,
        })
    return out


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": min(values),
            "median": statistics.median(values), "max": max(values)}


def summarize(rows: Sequence[dict[str, Any]],
              incomplete: Sequence[dict[str, Any]], planned: int
              ) -> dict[str, Any]:
    completed = len(rows)
    counts = {k: sum(r["k_hat"] == k for r in rows)
              for k in PROTOCOL.k_candidates}
    deltas = [r["delta_23"] for r in rows]
    return {
        "wrapper_version": WRAPPER_VERSION,
        "datasets": {"planned": planned, "completed": completed,
                     "incomplete": list(incomplete),
                     "denominator": completed},
        "P1_k_hat_distribution": {
            "counts": counts,
            "proportions": {k: (c / completed if completed else None)
                            for k, c in counts.items()},
            "exact": sum(r["k_category"] == "exact" for r in rows),
            "under": sum(r["k_category"] == "under" for r in rows),
            "over": sum(r["k_category"] == "over" for r in rows),
            "note": "observed frequency under this one frozen condition; "
                    "not consistency or a general success probability",
        },
        "P2_delta_23": {
            **_stats(deltas),
            "negative_prefers_k3": sum(d < 0 for d in deltas),
            "positive_prefers_k2": sum(d > 0 for d in deltas),
            "zero_tie_to_k2": sum(d == 0 for d in deltas),
            "values": {r["dataset"]: r["delta_23"] for r in rows},
        },
        "P3_decomposition_23": {
            "fit_gain_23": _stats([r["fit_gain_23"] for r in rows]),
            "penalty_increase_23": _stats(
                [r["penalty_increase_23"] for r in rows]),
            "max_abs_identity_residual": max(
                (abs(r["identity_residual"]) for r in rows), default=None),
        },
        "P4_family_at_k_hat": {
            "bernoulli_to_poisson_total": sum(
                r["bernoulli_to_poisson_at_k_hat"] for r in rows),
            "all_six_bernoulli_datasets": sum(
                r["bernoulli_to_poisson_at_k_hat"] == 0 for r in rows),
            "cols_3_8_bernoulli_at_every_k_datasets": sum(
                r["cols_3_8_bernoulli_at_every_k"] for r in rows),
        },
        "claim_boundary": {
            "unit": "independent dataset (start_B only)",
            "criterion": "existing direct C_Q (Q-based complete-data / "
                         "ICL-type; not Schwarz BIC); unchanged",
            "do_not_claim": ["K-selection consistency",
                             "asymptotic validity",
                             "general under-selection bias",
                             "robustness to other conditions",
                             "correctness of the current penalty theory",
                             "real-data effectiveness"],
            "phase9d": "Phase 9D's 3 datasets are historical evidence and "
                       "are not in this denominator",
            "lineage": "E (experimental prototype; not adoptable for the "
                       "manuscript)",
        },
    }


def evidence_checks(run_dir: Path) -> dict[str, Any]:
    """Reuse the existing #74 row-level checks; no new auditor."""

    protocol = json.loads((run_dir / "protocol.json").read_text("utf-8"))
    trace = _read(run_dir / "selection_trace.csv")
    scores = _read(run_dir / "family_scores.csv")
    cq = _read(run_dir / "cq_by_k.csv")
    findings: list = []
    audit74._check_installation_provenance(protocol, trace, findings)
    diagnostic = audit74._recompute_convergence_diagnostic(scores, trace,
                                                           findings)
    log_n = math.log(protocol["n"])
    criterion_mismatch = [
        f"{r['replicate']}/K={r['k']}" for r in cq
        if not math.isclose(float(r["C_Q"]),
                            -2 * float(r["Q_strict"])
                            + float(r["num_params"]) * log_n,
                            rel_tol=1e-12, abs_tol=1e-9)]
    return {
        "blocking_findings": [dict(f) for f in findings
                              if f["severity"] in ("BLOCKER", "HIGH")],
        "criterion_mismatch": criterion_mismatch,
        "integrity": {c: sum(int(r[c]) for r in cq)
                      for c in ("retry_count", "replacement_count",
                                "seed_rescue_count")},
        "candidate_convergence_diagnostic": {
            k: v for k, v in diagnostic.items() if k != "warnings"},
        "candidate_warnings": diagnostic["warnings"],
    }


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def analyse(run_dir: Path) -> dict[str, Any]:
    run_summary = json.loads((run_dir / "summary.json").read_text("utf-8"))
    incomplete = run_summary["datasets"]["incomplete"]
    failed = {i["replicate"] for i in incomplete}
    completed = [r.label for r in PROTOCOL.replicates if r.label not in failed]
    cq = [r for r in _read(run_dir / "cq_by_k.csv")
          if r["replicate"] in completed]
    rows = dataset_rows(cq, _read(run_dir / "cq_decomposition.csv"),
                        _read(run_dir / "joint_selection.csv"),
                        _read(run_dir / "family_by_k.csv"), completed)
    summary = summarize(rows, incomplete, len(PROTOCOL.replicates))
    summary["evidence_checks"] = evidence_checks(run_dir)
    if rows:
        pilot._write_csv(run_dir / "k_repeatability.csv", rows)
    pilot._write_json(run_dir / "repeatability_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Issue #81 Phase 9E repeatability study once.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    joint.execute(args.out, protocol=PROTOCOL, authorization=AUTHORIZATION)
    summary = analyse(args.out)
    p1 = summary["P1_k_hat_distribution"]
    print(f"completed {summary['datasets']['completed']}/"
          f"{summary['datasets']['planned']}; K_hat counts {p1['counts']}")
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
