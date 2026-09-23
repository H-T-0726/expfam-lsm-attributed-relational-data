"""Gate 74-B1R: reference certification with the analytic Jacobian (zero EM).

Gate 74-B1 ended REFERENCE_NOT_CERTIFIED, and for a specific reason: the three
starts agreed on where the optimum is (scores to 2.1e-14, loadings to 2.6e-7),
but the point BFGS returned could not be shown stationary below about 1.2e-7,
because ``jac=None`` locates it by forward differences and that is the
resolution of the difference itself.  B1 also established, on the same twelve
frozen combinations, that the production analytic gradient agrees with a
central difference of an independently written objective to 2.4e-9.

So this stage changes exactly one thing: the reference solver is given that
already-validated analytic gradient instead of differencing numerically.
Everything else is the B1 protocol verbatim -- the same two cases, the same Z
design and sample scales, the same three initial loadings, the same families,
the same thresholds, no RNG, no smoke artifact, no EM.

    If the validated analytic gradient is supplied to BFGS, can the same
    deterministic problems be certified under the original stationarity
    criterion, and if so how close is the unchanged 50-step production Adam to
    that certified optimum?

What this stage does not do
---------------------------
It does not touch the production selector or its Adam settings, it does not
change the production convergence rule, it does not reinterpret B1, and it
uses no fallback solver.  If the reference still fails to certify, that is the
result and the Adam comparison does not run: a gap measured against an
uncertified optimum would not mean anything.

The existing ``converged`` flag is recorded for every Adam run but is NOT a
B1R certification criterion.  That separation is the point of this stage: it
lets "the optimiser is inaccurate" be told apart from "the optimiser is
accurate enough and the step-based flag is too strict".

Protocol: GitHub Issue #74, Gate 74-B1R comment.
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
from scipy.optimize import minimize

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
    _column_gradient,
    column_log_likelihood,
    optimise_column_loading,
)
# The frozen B1 design and its independent checks are REUSED, not restated:
# Issue #74 requires B1R to run on exactly the same deterministic inputs, and
# a second transcription here would be a second thing to keep in step.  The
# independent transcription that matters is the auditor's.
from validate_candidate_optimizer import (                     # noqa: E402
    ADAM_GAP_PER_OBS_TOL,
    ADAM_GRAD_INF_TOL,
    BASE_VECTORS,
    BFGS_GTOL,
    BFGS_MAXITER,
    CASES,
    FAMILIES,
    FD_STEP,
    GRADIENT_MATCH_TOL,
    GROUP_REPEATS,
    INITIALISATIONS,
    K_DIM,
    N_ROWS,
    N_SAMPLES,
    OBJECTIVE_MATCH_TOL,
    REFERENCE_GRAD_INF_TOL,
    REFERENCE_LOADING_AGREE_TOL,
    REFERENCE_SCORE_AGREE_TOL,
    SAMPLE_SCALES,
    array_digest,
    build_case_column,
    build_z_samples,
    gradient_check_rows,
    independent_objective,
    objective_check_rows,
)

VALIDATOR_VERSION = "candidate-optimizer-validation-analytic-jac-v1"
GATE = "74-B1R"

STATUS_CERTIFIED = "REFERENCE_V2_CERTIFIED"
STATUS_NOT_CERTIFIED = "REFERENCE_V2_NOT_CERTIFIED"
STATUS_FAIL = "FAIL"
ADAM_CERTIFIED = "CURRENT_ADAM_CERTIFIED_FOR_B1R"
ADAM_NEEDS_HUMAN = "HUMAN_OPTIMIZER_DECISION_REQUIRED"
ADAM_NOT_RUN = "NOT_RUN"


class ValidationStop(RuntimeError):
    """Fail-fast stop for the B1R validator."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationStop(message)


# --------------------------------------------------------------------------
# reference solver v2: BFGS with the validated analytic gradient
# --------------------------------------------------------------------------

def solve_reference_analytic_jac(x_column: np.ndarray, Z_samples: np.ndarray,
                                 family: str,
                                 init: Sequence[float]) -> dict[str, Any]:
    """BFGS on the independent objective, with the analytic Jacobian supplied.

    The objective is the negative of the same strict per-column score B1 used,
    and the Jacobian is the negative of the production analytic gradient that
    B1 validated against a central difference.  No finite differencing is
    involved anywhere in the solve.
    """

    x = np.asarray(x_column, dtype=np.float64)

    def negative(loading: np.ndarray) -> float:
        return -independent_objective(x, Z_samples, loading, family)

    def negative_jac(loading: np.ndarray) -> np.ndarray:
        return -_column_gradient(x, Z_samples,
                                 np.asarray(loading, dtype=np.float64),
                                 family, None)

    result = minimize(negative, np.asarray(init, dtype=np.float64),
                      method="BFGS", jac=negative_jac,
                      options={"maxiter": BFGS_MAXITER, "gtol": BFGS_GTOL})
    loading = np.asarray(result.x, dtype=np.float64)
    score = independent_objective(x, Z_samples, loading, family)
    analytic = _column_gradient(x, Z_samples, loading, family, None)
    grad_inf = float(np.max(np.abs(analytic)))
    scipy_jac = np.asarray(getattr(result, "jac", np.full_like(loading, np.nan)),
                           dtype=np.float64)
    return {
        "loading": loading,
        "score": float(score),
        "analytic_grad_inf": grad_inf,
        # Now the same quantity by construction, up to the sign; recorded so an
        # auditor can see that the solver was not differencing.
        "scipy_jac_inf": float(np.max(np.abs(scipy_jac))),
        "scipy_success": bool(result.success),
        "scipy_status": int(result.status),
        "scipy_message": str(result.message),
        "scipy_nit": int(result.nit),
        "scipy_njev": int(getattr(result, "njev", -1)),
        "finite": bool(np.all(np.isfinite(loading)) and math.isfinite(score)
                       and math.isfinite(grad_inf)),
    }


def certify_reference_v2(Z_samples: np.ndarray):
    """Certify one reference optimum per (case, family) under the B1 rule."""

    rows: list[dict[str, Any]] = []
    certified: dict[tuple[str, str], dict[str, Any]] = {}
    failures: list[str] = []

    for case in CASES:
        x = build_case_column(case)
        for family in FAMILIES:
            solutions = {name: solve_reference_analytic_jac(
                x, Z_samples, family, init)
                for name, init in INITIALISATIONS.items()}
            scores = [s["score"] for s in solutions.values()]
            loadings = [s["loading"] for s in solutions.values()]
            score_spread = float(max(scores) - min(scores))
            loading_spread = float(max(np.max(np.abs(a - b))
                                       for a in loadings for b in loadings))
            all_finite = all(s["finite"] for s in solutions.values())
            grad_ok = all(s["analytic_grad_inf"] <= REFERENCE_GRAD_INF_TOL
                          for s in solutions.values())
            agree = (score_spread <= REFERENCE_SCORE_AGREE_TOL
                     and loading_spread <= REFERENCE_LOADING_AGREE_TOL)
            ok = bool(all_finite and grad_ok and agree)

            for name, solution in solutions.items():
                rows.append({
                    "case": case, "family": family, "init": name,
                    "loading": "|".join(f"{v:.12e}" for v in solution["loading"]),
                    "score": solution["score"],
                    "analytic_grad_inf": solution["analytic_grad_inf"],
                    "scipy_jac_inf": solution["scipy_jac_inf"],
                    "grad_inf_tolerance": REFERENCE_GRAD_INF_TOL,
                    "finite": solution["finite"],
                    "scipy_success": solution["scipy_success"],
                    "scipy_status": solution["scipy_status"],
                    "scipy_nit": solution["scipy_nit"],
                    "scipy_njev": solution["scipy_njev"],
                    "scipy_message": solution["scipy_message"],
                    "score_spread_across_inits": score_spread,
                    "loading_spread_across_inits": loading_spread,
                    "certified": ok,
                })
            if ok:
                certified[(case, family)] = max(solutions.values(),
                                                key=lambda s: s["score"])
            else:
                reasons = []
                if not all_finite:
                    reasons.append("a solution was not finite")
                if not grad_ok:
                    reasons.append(f"analytic gradient inf norm above "
                                   f"{REFERENCE_GRAD_INF_TOL}")
                if not agree:
                    reasons.append(
                        f"initialisations disagree (score spread "
                        f"{score_spread:.3e}, loading spread "
                        f"{loading_spread:.3e})")
                failures.append(f"{case}/{family}: " + "; ".join(reasons))
    return rows, certified, failures


# --------------------------------------------------------------------------
# the unchanged production Adam, measured against the certified optimum
# --------------------------------------------------------------------------

def adam_comparison_rows(Z_samples: np.ndarray, certified: dict
                         ) -> list[dict[str, Any]]:
    rows = []
    for case in CASES:
        x = build_case_column(case)
        for family in FAMILIES:
            reference = certified[(case, family)]
            for init_name, init in INITIALISATIONS.items():
                diagnostics: dict[str, Any] = {}
                loading, _, n_iter, converged = optimise_column_loading(
                    x, Z_samples, family,
                    loading_init=np.asarray(init, dtype=np.float64),
                    diagnostics=diagnostics)
                score = column_log_likelihood(x, Z_samples, loading, family)
                gap = reference["score"] - score
                rows.append({
                    "case": case, "family": family, "init": init_name,
                    "adam_max_iter": ADAM_MAX_ITER, "adam_lr": ADAM_LR,
                    "adam_beta1": ADAM_BETA1, "adam_beta2": ADAM_BETA2,
                    "adam_eps": ADAM_EPS, "adam_tol": ADAM_TOL,
                    "n_iter": n_iter,
                    # Recorded, but NOT the B1R criterion: separating the two
                    # is how this stage tells an inaccurate optimiser apart
                    # from an over-strict convergence flag.
                    "converged_flag": converged,
                    "adam_loading": "|".join(f"{v:.12e}" for v in loading),
                    "adam_score": score,
                    "reference_score": reference["score"],
                    "objective_gap": gap,
                    "objective_gap_per_observation": gap / N_ROWS,
                    "loading_inf_distance_to_ref": float(
                        np.max(np.abs(loading - reference["loading"]))),
                    "adam_grad_inf": diagnostics["final_gradient_inf"],
                    "final_step_inf": diagnostics["final_step_inf"],
                    "last_objective_change": diagnostics["last_objective_change"],
                    "gap_per_obs_tolerance": ADAM_GAP_PER_OBS_TOL,
                    "grad_inf_tolerance": ADAM_GRAD_INF_TOL,
                })
    return rows


def family_ordering_rows(certified: dict, adam_rows: Sequence[dict[str, Any]]
                         ) -> list[dict[str, Any]]:
    rows = []
    for case in CASES:
        reference_scores = {family: certified[(case, family)]["score"]
                            for family in FAMILIES}
        reference_winner = max(reference_scores, key=reference_scores.get)
        for init_name in INITIALISATIONS:
            adam_scores = {}
            for family in FAMILIES:
                row, = [r for r in adam_rows
                        if r["case"] == case and r["family"] == family
                        and r["init"] == init_name]
                adam_scores[family] = row["adam_score"]
            adam_winner = max(adam_scores, key=adam_scores.get)
            rows.append({
                "case": case, "init": init_name,
                "reference_winner": reference_winner,
                "reference_score_bernoulli": reference_scores["bernoulli"],
                "reference_score_poisson": reference_scores["poisson"],
                "reference_margin": abs(reference_scores["bernoulli"]
                                        - reference_scores["poisson"]),
                "adam_winner": adam_winner,
                "adam_score_bernoulli": adam_scores["bernoulli"],
                "adam_score_poisson": adam_scores["poisson"],
                "adam_margin": abs(adam_scores["bernoulli"]
                                   - adam_scores["poisson"]),
                "ordering_agrees": bool(adam_winner == reference_winner),
            })
    return rows


def not_run_rows(reason: str) -> list[dict[str, Any]]:
    """Explicit not-run provenance, so an absent table is never ambiguous."""

    return [{"status": "NOT_RUN", "reason": reason, "rows": 0}]


# --------------------------------------------------------------------------
# provenance and writing
# --------------------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=_HERE, capture_output=True,
                              text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):        # pragma: no cover
        return ""


def build_protocol(Z_samples: np.ndarray) -> dict[str, Any]:
    return {
        "validator_version": VALIDATOR_VERSION,
        "gate": GATE,
        "derived_from_gate": "74-B1",
        "uses_rng": False,
        "uses_smoke_artifacts": False,
        "em_executions": 0,
        "base_vectors": [list(v) for v in BASE_VECTORS],
        "group_repeats": GROUP_REPEATS,
        "sample_scales": list(SAMPLE_SCALES),
        "n": N_ROWS, "k": K_DIM, "L": N_SAMPLES,
        "z_samples_sha256": array_digest(Z_samples),
        "cases": {name: [list(group) for group in groups]
                  for name, groups in CASES.items()},
        "case_columns": {name: build_case_column(name).tolist()
                         for name in CASES},
        "case_column_sha256": {name: array_digest(build_case_column(name))
                               for name in CASES},
        "case_ones_count": {name: int(build_case_column(name).sum())
                            for name in CASES},
        "initialisations": {name: list(value)
                            for name, value in INITIALISATIONS.items()},
        "families": list(FAMILIES),
        "adam": {"max_iter": ADAM_MAX_ITER, "lr": ADAM_LR,
                 "beta1": ADAM_BETA1, "beta2": ADAM_BETA2,
                 "eps": ADAM_EPS, "tol": ADAM_TOL},
        "production_optimizer_changed": False,
        "production_convergence_rule_changed": False,
        "reference_solver": {
            "library": "scipy.optimize.minimize",
            "method": "BFGS",
            "jac": "analytic_production_gradient",
            "maxiter": BFGS_MAXITER,
            "gtol": BFGS_GTOL,
            "finite_difference_jacobian": False,
            "fallback_solvers": [],
            "changed_from_b1": "jac only: B1 used numerical differentiation "
                               "(jac=None); B1R supplies the production "
                               "analytic gradient that B1 validated",
        },
        "thresholds": {
            "objective_match_tol": OBJECTIVE_MATCH_TOL,
            "fd_step": FD_STEP,
            "gradient_match_tol": GRADIENT_MATCH_TOL,
            "reference_grad_inf_tol": REFERENCE_GRAD_INF_TOL,
            "reference_score_agree_tol": REFERENCE_SCORE_AGREE_TOL,
            "reference_loading_agree_tol": REFERENCE_LOADING_AGREE_TOL,
            "adam_gap_per_obs_tol": ADAM_GAP_PER_OBS_TOL,
            "adam_grad_inf_tol": ADAM_GRAD_INF_TOL,
        },
        "converged_flag_is_certification_criterion": False,
        "expected_adam_runs": len(CASES) * len(FAMILIES) * len(INITIALISATIONS),
        "git_sha": _git("rev-parse", "HEAD").strip() or "unavailable",
        "git_dirty": bool(_git("status", "--porcelain").strip()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "note": "Diagnostic certification thresholds for Gate 74-B1R only. "
                "They are not manuscript claims, they do not reclassify the "
                "Gate 74-B smoke, and they do not reinterpret Gate 74-B1.",
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
    "objective_checks.csv",
    "gradient_checks.csv",
    "reference_solutions.csv",
    "adam_comparison.csv",
    "family_ordering.csv",
    "summary.json",
)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def execute(out_dir: Path) -> dict[str, Any]:
    """Run the whole B1R validation once and write its artifacts."""

    _require(not out_dir.exists(),
             f"{out_dir} already exists; a recorded run is never overwritten")

    started = datetime.now(timezone.utc).isoformat()
    Z_samples = build_z_samples()
    protocol = build_protocol(Z_samples)

    # Regression guards, retained from B1: the reference is not interpreted
    # unless the scorer and its gradient still check out.
    objective_rows = objective_check_rows(Z_samples)
    gradient_rows = gradient_check_rows(Z_samples)
    objective_ok = all(row["passed"] for row in objective_rows)
    gradient_ok = all(row["passed"] for row in gradient_rows)

    reference_rows, certified, reference_failures = certify_reference_v2(
        Z_samples)
    reference_ok = not reference_failures

    if not (objective_ok and gradient_ok):
        status = STATUS_FAIL
    elif reference_ok:
        status = STATUS_CERTIFIED
    else:
        status = STATUS_NOT_CERTIFIED

    adam_rows: list[dict[str, Any]] = []
    ordering_rows: list[dict[str, Any]] = []
    adam_failures: list[str] = []
    adam_verdict = ADAM_NOT_RUN
    not_run_reason = ""

    if status == STATUS_CERTIFIED:
        adam_rows = adam_comparison_rows(Z_samples, certified)
        ordering_rows = family_ordering_rows(certified, adam_rows)
        for row in adam_rows:
            label = f"{row['case']}/{row['family']}/{row['init']}"
            if row["objective_gap_per_observation"] > ADAM_GAP_PER_OBS_TOL:
                adam_failures.append(
                    f"{label}: objective_gap_per_observation "
                    f"{row['objective_gap_per_observation']:.6e} > "
                    f"{ADAM_GAP_PER_OBS_TOL}")
            if row["adam_grad_inf"] > ADAM_GRAD_INF_TOL:
                adam_failures.append(
                    f"{label}: adam_grad_inf {row['adam_grad_inf']:.6e} > "
                    f"{ADAM_GRAD_INF_TOL}")
        for row in ordering_rows:
            if not row["ordering_agrees"]:
                adam_failures.append(
                    f"{row['case']}/{row['init']}: Adam prefers "
                    f"{row['adam_winner']}, the certified reference prefers "
                    f"{row['reference_winner']}")
        adam_verdict = ADAM_CERTIFIED if not adam_failures else ADAM_NEEDS_HUMAN
    else:
        not_run_reason = (
            "the reference was not certified, so there is no optimum to "
            f"measure against (status {status})" if status != STATUS_FAIL
            else "the independent objective or gradient regression guard "
                 "failed, so the reference was not interpreted")

    summary = {
        "validator_version": VALIDATOR_VERSION,
        "gate": GATE,
        "status": status,
        "adam_verdict": adam_verdict,
        "em_executions": 0,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "objective_checks": {
            "total": len(objective_rows),
            "passed": sum(1 for r in objective_rows if r["passed"]),
            "max_abs_difference": max(r["abs_difference"]
                                      for r in objective_rows),
            "tolerance": OBJECTIVE_MATCH_TOL,
        },
        "gradient_checks": {
            "total": len(gradient_rows),
            "passed": sum(1 for r in gradient_rows if r["passed"]),
            "max_abs_difference": max(r["max_abs_difference"]
                                      for r in gradient_rows),
            "tolerance": GRADIENT_MATCH_TOL,
        },
        "reference_certification": {
            "solver_jac": "analytic_production_gradient",
            "problems": len(CASES) * len(FAMILIES),
            "certified": len(certified),
            "failures": reference_failures,
            "max_analytic_grad_inf": max(r["analytic_grad_inf"]
                                         for r in reference_rows),
            "max_score_spread": max(r["score_spread_across_inits"]
                                    for r in reference_rows),
            "max_loading_spread": max(r["loading_spread_across_inits"]
                                      for r in reference_rows),
            "grad_inf_tolerance": REFERENCE_GRAD_INF_TOL,
            "score_spread_tolerance": REFERENCE_SCORE_AGREE_TOL,
            "loading_spread_tolerance": REFERENCE_LOADING_AGREE_TOL,
        },
        "adam_summary": {
            "runs": len(adam_rows),
            "not_run_reason": not_run_reason,
            "converged_flag_true": sum(1 for r in adam_rows
                                       if r["converged_flag"]),
            "hit_budget": sum(1 for r in adam_rows
                              if r["n_iter"] == ADAM_MAX_ITER),
            "max_objective_gap_per_observation": max(
                (r["objective_gap_per_observation"] for r in adam_rows),
                default=None),
            "max_adam_grad_inf": max((r["adam_grad_inf"] for r in adam_rows),
                                     default=None),
            "max_loading_inf_distance_to_ref": max(
                (r["loading_inf_distance_to_ref"] for r in adam_rows),
                default=None),
            "failures": adam_failures,
        },
        "family_ordering": {
            "rows": len(ordering_rows),
            "agreements": sum(1 for r in ordering_rows if r["ordering_agrees"]),
            "reference_winner_by_case": {
                case: max(FAMILIES, key=lambda f: certified[(case, f)]["score"])
                for case in CASES} if reference_ok else {},
        },
        "interpretation": {
            "changed_from_b1": "the reference Jacobian only",
            "converged_flag_recorded_not_certifying":
                "the production step-based converged flag is recorded for "
                "every Adam run and is not a B1R criterion, so an accurate "
                "optimiser with an over-strict flag is distinguishable from "
                "an inaccurate one",
            "no_production_change": "no production optimiser setting and no "
                                    "production convergence rule was changed",
            "does_not_reinterpret_b1": True,
            "smoke_artifacts_used_as_input": False,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=False)
    _write_json(out_dir / "protocol.json", protocol)
    _write_csv(out_dir / "objective_checks.csv", objective_rows)
    _write_csv(out_dir / "gradient_checks.csv", gradient_rows)
    _write_csv(out_dir / "reference_solutions.csv", reference_rows)
    _write_csv(out_dir / "adam_comparison.csv",
               adam_rows if adam_rows else not_run_rows(not_run_reason))
    _write_csv(out_dir / "family_ordering.csv",
               ordering_rows if ordering_rows else not_run_rows(not_run_reason))
    _write_json(out_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Gate 74-B1R analytic-jac reference certification.")
    parser.add_argument("--out", required=True, type=Path,
                        help="fresh output directory (must not exist)")
    args = parser.parse_args(argv)

    summary = execute(args.out)
    print(f"status={summary['status']} adam={summary['adam_verdict']} "
          f"em_executions={summary['em_executions']}")
    print(f"artifacts written to {args.out}")
    return 0 if summary["status"] == STATUS_CERTIFIED else 1


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
