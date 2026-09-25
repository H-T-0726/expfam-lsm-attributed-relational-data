"""Gate 74-B1: deterministic, zero-EM validation of the candidate optimiser.

The Gate 74-B smoke ended BLOCKED_FOR_PILOT because all eight candidate
optimisations stopped at the frozen 50-step Adam budget without satisfying the
step-based convergence rule.  This module answers the one question that
follows, without running a single EM fit and without touching production
settings:

    On fixed, deterministic, non-separable binary-column problems, how close is
    the current 50-step Adam result to an independently converged reference
    optimum, and is the Bernoulli-vs-Poisson ordering stable?

What this is not
----------------
It is not a rerun of the smoke and it does not read the smoke's artifacts: the
cases below are frozen in this file and use no RNG at all.  It does not change
the production optimiser, and it must not be used to pick a solver setting
because that setting reproduces the smoke's selections.  Which family wins is
recorded, not sought.

Independence
------------
The objective and its gradient are checked against implementations written
here from the formulas, NOT by calling ``column_log_likelihood``.  The
reference optimum comes from SciPy BFGS with numerical differentiation, so it
never reuses the production analytic gradient.  There is no fallback solver:
if the reference cannot be certified, that is the result.

Protocol: GitHub Issue #74, Gate 74-B1 comment.
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

VALIDATOR_VERSION = "candidate-optimizer-validation-v1"

# --------------------------------------------------------------------------
# The frozen design.  No RNG anywhere in this module.
# --------------------------------------------------------------------------

BASE_VECTORS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.2, -0.1),
    (-0.6, 1.0, 0.3),
    (0.4, -0.7, 1.0),
    (-1.0, -0.2, 0.5),
    (0.3, 0.8, -0.9),
    (-0.5, 0.4, -0.8),
)
GROUP_REPEATS = 4                       # n = 6 * 4 = 24
SAMPLE_SCALES: tuple[float, ...] = (0.96, 0.98, 1.00, 1.02, 1.04)   # L = 5

N_ROWS = len(BASE_VECTORS) * GROUP_REPEATS
K_DIM = len(BASE_VECTORS[0])
N_SAMPLES = len(SAMPLE_SCALES)

# Each group's four values map to the four repeated rows of that base vector.
# Every group contains both a 0 and a 1, so no case is completely separable.
CASES: dict[str, tuple[tuple[int, ...], ...]] = {
    "case_A_balanced_nonseparable": (
        (1, 1, 1, 0),
        (1, 0, 0, 0),
        (1, 1, 1, 0),
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (1, 1, 0, 0),
    ),
    "case_B_sparse_nonseparable": (
        (1, 1, 0, 0),
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
    ),
}

INITIALISATIONS: dict[str, tuple[float, float, float]] = {
    "init_zero": (0.0, 0.0, 0.0),
    "init_pos": (0.6, -0.4, 0.3),
    "init_neg": (-0.6, 0.4, -0.3),
}

FAMILIES: tuple[str, ...] = ("bernoulli", "poisson")

# --------------------------------------------------------------------------
# Frozen thresholds.  Diagnostic certification thresholds for B1 only: they
# are not manuscript claims and they do not reclassify the smoke.
# --------------------------------------------------------------------------

OBJECTIVE_MATCH_TOL = 1e-10          # production score vs independent score
FD_STEP = 1e-6                       # central difference step
GRADIENT_MATCH_TOL = 1e-5            # analytic gradient vs finite difference
REFERENCE_GRAD_INF_TOL = 1e-8        # certified optimum must be stationary
REFERENCE_SCORE_AGREE_TOL = 1e-8     # across the three initialisations
REFERENCE_LOADING_AGREE_TOL = 1e-5   # across the three initialisations
ADAM_GAP_PER_OBS_TOL = 1e-6          # (Q_ref - Q_adam) / n
ADAM_GRAD_INF_TOL = 1e-4             # Adam solution stationarity

BFGS_MAXITER = 2000
BFGS_GTOL = 1e-10

STATUS_VALID = "B1_VALID"
STATUS_REFERENCE_NOT_CERTIFIED = "REFERENCE_NOT_CERTIFIED"
STATUS_FAIL = "FAIL"
ADAM_CERTIFIED = "CURRENT_ADAM_CERTIFIED_FOR_B1"
ADAM_NEEDS_HUMAN = "HUMAN_OPTIMIZER_DECISION_REQUIRED"


class ValidationStop(RuntimeError):
    """Fail-fast stop for the validator."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationStop(message)


# --------------------------------------------------------------------------
# frozen arrays
# --------------------------------------------------------------------------

def build_z_samples() -> np.ndarray:
    """The frozen (n, k, L) posterior-sample block. Deterministic, no RNG."""

    base = np.repeat(np.asarray(BASE_VECTORS, dtype=np.float64),
                     GROUP_REPEATS, axis=0)
    samples = np.empty((N_ROWS, K_DIM, N_SAMPLES), dtype=np.float64)
    for index, scale in enumerate(SAMPLE_SCALES):
        samples[:, :, index] = base * scale
    return samples


def build_case_column(case: str) -> np.ndarray:
    """The frozen binary column for one case."""

    _require(case in CASES, f"unknown case {case!r}")
    groups = CASES[case]
    _require(len(groups) == len(BASE_VECTORS),
             f"{case}: expected one group per base vector")
    values: list[float] = []
    for group in groups:
        _require(len(group) == GROUP_REPEATS,
                 f"{case}: each group needs {GROUP_REPEATS} values")
        _require(set(group) == {0, 1},
                 f"{case}: every group must contain both a 0 and a 1 so the "
                 f"problem is not completely separable")
        values.extend(float(value) for value in group)
    return np.asarray(values, dtype=np.float64)


def array_digest(array: np.ndarray) -> str:
    """A stable digest so an auditor can confirm the frozen arrays."""

    return hashlib.sha256(
        np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


# --------------------------------------------------------------------------
# independent objective and gradient (NOT via column_log_likelihood)
# --------------------------------------------------------------------------

def _eta(Z_samples: np.ndarray, loading: np.ndarray) -> np.ndarray:
    """eta[i, s] = f^T z_i^(s), built by explicit per-sample matrix products."""

    return np.column_stack([Z_samples[:, :, s] @ loading
                            for s in range(Z_samples.shape[2])])


def independent_objective(x_column: np.ndarray, Z_samples: np.ndarray,
                          loading: np.ndarray, family: str) -> float:
    """The complete per-column log likelihood, written from the formulas.

    Deliberately not a call into the production scorer: if both sides shared
    an implementation, agreeing would prove nothing.  ``log(1 + e^eta)`` is
    evaluated as ``max(eta, 0) + log1p(exp(-|eta|))`` rather than through
    ``logaddexp``, and the Poisson base measure through ``math.lgamma``
    rather than ``scipy.special.gammaln``.
    """

    _require(family in FAMILIES, f"unknown family {family!r}")
    x = np.asarray(x_column, dtype=np.float64)
    eta = _eta(Z_samples, np.asarray(loading, dtype=np.float64))
    n_samples = eta.shape[1]

    if family == "bernoulli":
        softplus = np.maximum(eta, 0.0) + np.log1p(np.exp(-np.abs(eta)))
        total = float(np.sum(x[:, None] * eta - softplus))
        return total / n_samples

    total = float(np.sum(x[:, None] * eta - np.exp(eta)))
    base_measure = -sum(math.lgamma(value + 1.0) for value in x)
    return total / n_samples + base_measure


def finite_difference_gradient(x_column: np.ndarray, Z_samples: np.ndarray,
                               loading: np.ndarray, family: str,
                               step: float = FD_STEP) -> np.ndarray:
    """Central difference of the INDEPENDENT objective."""

    loading = np.asarray(loading, dtype=np.float64)
    gradient = np.zeros_like(loading)
    for index in range(loading.size):
        forward = loading.copy()
        backward = loading.copy()
        forward[index] += step
        backward[index] -= step
        gradient[index] = (
            independent_objective(x_column, Z_samples, forward, family)
            - independent_objective(x_column, Z_samples, backward, family)
        ) / (2.0 * step)
    return gradient


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def objective_check_rows(Z_samples: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for case in CASES:
        x = build_case_column(case)
        for family in FAMILIES:
            for init_name, init in INITIALISATIONS.items():
                loading = np.asarray(init, dtype=np.float64)
                production = column_log_likelihood(x, Z_samples, loading, family)
                independent = independent_objective(x, Z_samples, loading, family)
                difference = abs(production - independent)
                rows.append({
                    "case": case, "family": family, "init": init_name,
                    "production_score": production,
                    "independent_score": independent,
                    "abs_difference": difference,
                    "tolerance": OBJECTIVE_MATCH_TOL,
                    "passed": bool(difference <= OBJECTIVE_MATCH_TOL),
                })
    return rows


def gradient_check_rows(Z_samples: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for case in CASES:
        x = build_case_column(case)
        for family in FAMILIES:
            for init_name, init in INITIALISATIONS.items():
                loading = np.asarray(init, dtype=np.float64)
                analytic = _column_gradient(x, Z_samples, loading, family, None)
                numeric = finite_difference_gradient(x, Z_samples, loading,
                                                     family)
                difference = float(np.max(np.abs(analytic - numeric)))
                rows.append({
                    "case": case, "family": family, "init": init_name,
                    "analytic_gradient": "|".join(f"{v:.12e}" for v in analytic),
                    "finite_difference_gradient":
                        "|".join(f"{v:.12e}" for v in numeric),
                    "max_abs_difference": difference,
                    "fd_step": FD_STEP,
                    "tolerance": GRADIENT_MATCH_TOL,
                    "passed": bool(difference <= GRADIENT_MATCH_TOL),
                })
    return rows


# --------------------------------------------------------------------------
# independent reference optimiser
# --------------------------------------------------------------------------

def solve_reference(x_column: np.ndarray, Z_samples: np.ndarray, family: str,
                    init: Sequence[float]) -> dict[str, Any]:
    """SciPy BFGS on the INDEPENDENT objective, numerically differentiated."""

    def negative(loading: np.ndarray) -> float:
        return -independent_objective(x_column, Z_samples, loading, family)

    result = minimize(negative, np.asarray(init, dtype=np.float64),
                      method="BFGS", jac=None,
                      options={"maxiter": BFGS_MAXITER, "gtol": BFGS_GTOL})
    loading = np.asarray(result.x, dtype=np.float64)
    score = independent_objective(x_column, Z_samples, loading, family)
    analytic = _column_gradient(x_column, Z_samples, loading, family, None)
    grad_inf = float(np.max(np.abs(analytic)))
    scipy_jac = np.asarray(getattr(result, "jac", np.full_like(loading, np.nan)),
                           dtype=np.float64)
    return {
        "loading": loading,
        "score": float(score),
        "analytic_grad_inf": grad_inf,
        # SciPy's own forward-difference gradient at the same point. Recorded
        # as evidence, never used as a certification criterion: the criterion
        # is the production analytic gradient, fixed before this ran.
        "scipy_jac_inf": float(np.max(np.abs(scipy_jac))),
        "scipy_success": bool(result.success),
        "scipy_status": int(result.status),
        "scipy_message": str(result.message),
        "scipy_nit": int(result.nit),
        "finite": bool(np.all(np.isfinite(loading)) and math.isfinite(score)
                       and math.isfinite(grad_inf)),
    }


def certify_reference(Z_samples: np.ndarray):
    """Solve and certify one reference optimum per (case, family).

    Certification does not rest on SciPy's success flag: the solution must be
    finite, the PRODUCTION analytic gradient must vanish there, and the three
    initialisations must land in the same place.
    """

    rows: list[dict[str, Any]] = []
    certified: dict[tuple[str, str], dict[str, Any]] = {}
    failures: list[str] = []

    for case in CASES:
        x = build_case_column(case)
        for family in FAMILIES:
            solutions = {name: solve_reference(x, Z_samples, family, init)
                         for name, init in INITIALISATIONS.items()}
            scores = [s["score"] for s in solutions.values()]
            loadings = [s["loading"] for s in solutions.values()]
            score_spread = float(max(scores) - min(scores))
            loading_spread = float(max(
                np.max(np.abs(a - b))
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
                    "scipy_message": solution["scipy_message"],
                    "score_spread_across_inits": score_spread,
                    "loading_spread_across_inits": loading_spread,
                    "certified": ok,
                })
            if ok:
                best = max(solutions.values(), key=lambda s: s["score"])
                certified[(case, family)] = best
            else:
                reasons = []
                if not all_finite:
                    reasons.append("a solution was not finite")
                if not grad_ok:
                    reasons.append(
                        f"analytic gradient inf norm above "
                        f"{REFERENCE_GRAD_INF_TOL}")
                if not agree:
                    reasons.append(
                        f"initialisations disagree (score spread "
                        f"{score_spread:.3e}, loading spread "
                        f"{loading_spread:.3e})")
                failures.append(f"{case}/{family}: " + "; ".join(reasons))
    return rows, certified, failures


# --------------------------------------------------------------------------
# the current Adam, instrumented but numerically untouched
# --------------------------------------------------------------------------

def adam_comparison_rows(Z_samples: np.ndarray,
                         certified: dict) -> list[dict[str, Any]]:
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


def family_ordering_rows(certified: dict,
                         adam_rows: Sequence[dict[str, Any]]
                         ) -> list[dict[str, Any]]:
    """Which family each procedure prefers, and whether they agree.

    Recorded, not sought: B1 is about how well the optimiser solved the frozen
    problems, not about which family ought to win.
    """

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
                "reference_margin":
                    abs(reference_scores["bernoulli"]
                        - reference_scores["poisson"]),
                "adam_winner": adam_winner,
                "adam_score_bernoulli": adam_scores["bernoulli"],
                "adam_score_poisson": adam_scores["poisson"],
                "adam_margin":
                    abs(adam_scores["bernoulli"] - adam_scores["poisson"]),
                "ordering_agrees": bool(adam_winner == reference_winner),
            })
    return rows


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
        "gate": "74-B1",
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
        "reference_solver": {"library": "scipy.optimize.minimize",
                             "method": "BFGS", "jac": None,
                             "maxiter": BFGS_MAXITER, "gtol": BFGS_GTOL,
                             "fallback_solvers": []},
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
        "expected_adam_runs": len(CASES) * len(FAMILIES) * len(INITIALISATIONS),
        "git_sha": _git("rev-parse", "HEAD").strip() or "unavailable",
        "git_dirty": bool(_git("status", "--porcelain").strip()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "note": "Diagnostic certification thresholds for Gate 74-B1 only. "
                "They are not manuscript claims and do not reclassify the "
                "Gate 74-B smoke.",
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


# Always written.
ARTIFACT_NAMES = (
    "protocol.json",
    "objective_checks.csv",
    "gradient_checks.csv",
    "reference_solutions.csv",
    "summary.json",
)

# Written only when the reference was certified and Adam therefore ran: a
# comparison against an uncertified optimum would not mean anything.
CERTIFIED_ARTIFACT_NAMES = (
    "adam_comparison.csv",
    "family_ordering.csv",
)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def execute(out_dir: Path) -> dict[str, Any]:
    """Run the whole B1 validation once and write its artifacts."""

    _require(not out_dir.exists(),
             f"{out_dir} already exists; a recorded run is never overwritten")

    started = datetime.now(timezone.utc).isoformat()
    Z_samples = build_z_samples()
    protocol = build_protocol(Z_samples)

    objective_rows = objective_check_rows(Z_samples)
    gradient_rows = gradient_check_rows(Z_samples)
    reference_rows, certified, reference_failures = certify_reference(Z_samples)

    objective_ok = all(row["passed"] for row in objective_rows)
    gradient_ok = all(row["passed"] for row in gradient_rows)
    reference_ok = not reference_failures

    adam_rows: list[dict[str, Any]] = []
    ordering_rows: list[dict[str, Any]] = []
    if reference_ok:
        adam_rows = adam_comparison_rows(Z_samples, certified)
        ordering_rows = family_ordering_rows(certified, adam_rows)

    if not reference_ok:
        status = STATUS_REFERENCE_NOT_CERTIFIED
    elif objective_ok and gradient_ok:
        status = STATUS_VALID
    else:
        status = STATUS_FAIL

    adam_verdict = ADAM_NEEDS_HUMAN
    adam_failures: list[str] = []
    if status == STATUS_VALID:
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
                    f"{row['adam_winner']}, the reference prefers "
                    f"{row['reference_winner']}")
        if not adam_failures:
            adam_verdict = ADAM_CERTIFIED

    summary = {
        "validator_version": VALIDATOR_VERSION,
        "gate": "74-B1",
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
            "problems": len(CASES) * len(FAMILIES),
            "certified": len(certified),
            "failures": reference_failures,
            "max_analytic_grad_inf": max(
                (r["analytic_grad_inf"] for r in reference_rows), default=None),
            "max_score_spread": max(
                (r["score_spread_across_inits"] for r in reference_rows),
                default=None),
            "max_loading_spread": max(
                (r["loading_spread_across_inits"] for r in reference_rows),
                default=None),
            "max_scipy_jac_inf": max(
                (r["scipy_jac_inf"] for r in reference_rows), default=None),
            "observation": "Recorded for the human, not acted on here: the "
                           "reference solver differentiates numerically "
                           "(jac=None), so its forward-difference gradient "
                           "resolves to multiples of roughly 1.2e-7. Both the "
                           "production analytic gradient and SciPy's own "
                           "reported jacobian at the returned point are "
                           "compared against the frozen 1e-8 criterion; where "
                           "they exceed it, that is what the artifact says. "
                           "The solver settings and the criterion were frozen "
                           "before this ran and are not adjusted now.",
        },
        "adam_summary": {
            "runs": len(adam_rows),
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
            "b1_asks": "how close the frozen 50-step Adam gets to an "
                       "independently certified optimum, and whether the "
                       "family ordering is stable",
            "b1_does_not_ask": "which family is correct; the ordering is "
                               "recorded, not sought",
            "no_production_change": "this run changes no production optimiser "
                                    "setting and does not reclassify the "
                                    "Gate 74-B smoke",
            "smoke_artifacts_used_as_input": False,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=False)
    _write_json(out_dir / "protocol.json", protocol)
    _write_csv(out_dir / "objective_checks.csv", objective_rows)
    _write_csv(out_dir / "gradient_checks.csv", gradient_rows)
    _write_csv(out_dir / "reference_solutions.csv", reference_rows)
    if adam_rows:
        _write_csv(out_dir / "adam_comparison.csv", adam_rows)
    if ordering_rows:
        _write_csv(out_dir / "family_ordering.csv", ordering_rows)
    _write_json(out_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Gate 74-B1 zero-EM candidate-optimizer validation.")
    parser.add_argument("--out", required=True, type=Path,
                        help="fresh output directory (must not exist)")
    args = parser.parse_args(argv)

    summary = execute(args.out)
    print(f"status={summary['status']} adam={summary['adam_verdict']} "
          f"em_executions={summary['em_executions']}")
    print(f"artifacts written to {args.out}")
    return 0 if summary["status"] == STATUS_VALID else 1


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
