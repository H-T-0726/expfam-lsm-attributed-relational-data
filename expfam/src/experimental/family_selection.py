"""Forward-only per-column family-selection machinery (Phase 9C, Issue #74).

This module implements the design in
``reports/distribution_selection/automatic_family_selection_design_20260923.md``
for a FIXED-K feasibility pilot.  It adds no new model lineage: it drives the
existing objective-consistent per-column class
(``DualExpFamLSMPerColumnConsistent``, lineage E) and leaves every existing
file unchanged.

What this is, and is not
------------------------
Under the three implemented families and the strict support gate of
``support_gate`` below, the candidate score only decides anything for columns
whose observations are all in {0, 1} (Bernoulli vs Poisson).  Gaussian and
multi-valued-count columns are settled by the gate alone, with no score
involved.  So this machinery is a FEASIBILITY PILOT for per-column family
selection, NOT evidence for automatic distribution selection in general, and
gate-decided columns must never be folded into a selector accuracy number.
``support_gate`` records ``decided_by`` per column precisely so that the two
kinds of column can be kept apart downstream.

Frozen Human Gates (Issue #74; this module does not re-decide them)
-------------------------------------------------------------------
HG-1  Scheme C (hybrid): an A-type family update inside the exploration EM,
      then a FRESH refit with the selected assignment held fixed, and that
      refit is what gets reported.  No extra outer cycle in the pilot.
HG-3  No new discrete search-cost penalty.  Candidates are compared on the
      strict complete per-column score.  ``sum_l log|M_l|`` is NOT used as a
      selector penalty: with each candidate set fixed it is constant in the
      assignment and cannot change ``argmin_c`` (design section 6.4.1).
HG-4  No X-side Negative Binomial.
HG-5  No X-column intercept.

Numerics
--------
Only the objective-consistent numerics are used for scoring
(``objective_consistent_numerics``).  The legacy clipped path is never used
for a candidate score: outside the clip interval the legacy score disagrees
with its own objective (KI-015), and a family comparison reads score
DIFFERENCES directly, so a clipped value is not a usable log-likelihood.
Non-finite scores and non-finite gradients are fail-fast; a candidate is never
rescued by retrying with another seed.  A degenerate Gaussian column variance
is rejected rather than floored, and a degenerate observed Y density stops the
exploration rather than being clipped to a usable-looking value.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from em_runner import EMFailFast, run_em_experimental         # noqa: E402
from model_dual_expfam_consistent import (                    # noqa: E402
    DualExpFamLSMPerColumnConsistent,
)
from objective_consistent_numerics import (                   # noqa: E402
    bernoulli_log_likelihood,
    bernoulli_mean,
    poisson_log_likelihood,
    poisson_mean,
)

SELECTOR_VERSION = "family-selection-v1"

VALID_FAMILIES = ("gaussian", "bernoulli", "poisson")

# Deterministic tie rule.  On an EXACT tie between candidate scores the
# candidate appearing first in this tuple wins.  Ties are expected to be rare,
# but "rare" is not "deterministic", and an audit needs the rule written down
# rather than left to dict or sort order.
CANDIDATE_PRIORITY = ("bernoulli", "poisson", "gaussian")

# The candidate optimiser Phase 9C uses, by explicit configuration.
#
# Gate 74-B1R measured the 50-step Adam against an independently certified
# optimum on deterministic problems: it stopped up to 0.126 per observation
# short, with a gradient infinity norm up to 5.4 where the criterion asks for
# 1e-4. That is an optimiser genuinely far from the optimum, not an accurate
# one wearing a strict flag, so a human approved migrating the Phase 9C
# candidate path to the analytic-gradient BFGS configuration B1R certified.
#
# The Adam route below is NOT removed or redefined. Gate 74-B smoke and the
# B1/B1R diagnostics were produced by it and must stay reproducible, so it
# remains reachable by asking for it by name.
OPTIMIZER_ADAM = "adam"
OPTIMIZER_BFGS = "bfgs"
VALID_OPTIMIZERS = (OPTIMIZER_ADAM, OPTIMIZER_BFGS)

PHASE9C_CANDIDATE_OPTIMIZER = OPTIMIZER_BFGS

# Frozen BFGS configuration (Issue #74 Gate 74-B2 approval).
BFGS_METHOD = "BFGS"
BFGS_MAXITER = 2000
BFGS_GTOL = 1e-10

# What the exploration M-step installs for an ambiguous column (Gate 74-B5).
# Gate 74-B4 found that the winning candidate's loading was used only to score
# it and then discarded, so the column carried forward a loading from the
# parent 50-step Adam instead. Phase 9C now installs the loading that actually
# produced the winning score.
LOADING_INSTALLATION_POLICY = "selected_candidate_loading"

# A BFGS candidate counts as converged only on the gradient it actually
# reached. SciPy's success flag is recorded as provenance but does not decide
# this: B1 showed a solver can report success at a point whose gradient is
# orders of magnitude above the criterion.
CANDIDATE_GRAD_INF_TOL = 1e-8

# Existing Adam convention (model_dual_expfam_percolumn._calc_F_adam_weighted).
# Reused verbatim so the legacy candidate route introduces no new tuning knob.
ADAM_MAX_ITER = 50
ADAM_LR = 0.01
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
ADAM_TOL = 1e-6

# Smallest Gaussian column variance this module will score.  It is a REJECTION
# threshold, not a clamp: flooring a degenerate variance would hand back a
# large finite log-density for a column the model cannot actually describe,
# which is the silent repair this module exists to avoid.  The model classes
# clamp at the same magnitude; the selector refuses instead, because a family
# comparison reads score differences directly.
SIGMA_SQ_MIN = 1e-8


class SelectorStop(RuntimeError):
    """Fail-fast stop for the selector.

    Raised instead of clipping, substituting a fallback value, or retrying a
    candidate with a different seed.  A caller must never catch this and retry:
    that would hide exactly the failure the pilot is meant to observe.
    """


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectorStop(message)


def _checked_variance(sigma_sq: float, *, source: str) -> float:
    """Accept a Gaussian column variance, or stop.

    A non-positive, non-finite or degenerate variance means the column cannot
    be described by this family at these parameters.  Clamping it would return
    an arbitrary finite score for that column and let the comparison proceed on
    a number nothing produced.
    """

    value = float(sigma_sq)
    _require(math.isfinite(value),
             f"{source}: Gaussian column variance is not finite ({value})")
    _require(value >= SIGMA_SQ_MIN,
             f"{source}: Gaussian column variance {value} is below the "
             f"minimum this module will score ({SIGMA_SQ_MIN}); the column is "
             f"degenerate at these parameters and the selector refuses to "
             f"substitute a floor")
    return value


# --------------------------------------------------------------------------
# A1. deterministic support gate
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnGate:
    """The gate verdict for one attribute column."""

    column: int
    candidates: tuple[str, ...]
    decided_by: str          # "gate" | "score"
    gate_reason: str

    @property
    def is_ambiguous(self) -> bool:
        return self.decided_by == "score"

    @property
    def fixed_family(self) -> str | None:
        return self.candidates[0] if self.decided_by == "gate" else None

    def as_row(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "candidates": "|".join(self.candidates),
            "n_candidates": len(self.candidates),
            "decided_by": self.decided_by,
            "gate_reason": self.gate_reason,
            "fixed_family": self.fixed_family if self.fixed_family else "",
        }


def support_gate(X: np.ndarray) -> list[ColumnGate]:
    """Decide each column's admissible families from its observed support.

    The gate is deterministic and uses NO likelihood.  It separates the base
    measure question (counting measure vs Lebesgue) from the family question,
    because a continuous density and a discrete PMF are not comparable: the
    Gaussian log-density moves by ``-n log a`` under ``x -> a x`` while a
    Poisson log-PMF does not, so their ordering can be reversed by a change of
    unit (design section 3.3).

    Verdicts
    --------
    all values in {0, 1}            -> candidates ("bernoulli", "poisson"), by score
    all non-negative integers, max>1 -> ("poisson",), by gate
    anything else                    -> ("gaussian",), by gate
    """

    values = np.asarray(X, dtype=np.float64)
    _require(values.ndim == 2, f"X must be 2-D, got shape {values.shape}")
    _require(bool(np.all(np.isfinite(values))), "X contains a non-finite value")

    gates: list[ColumnGate] = []
    for column in range(values.shape[1]):
        col = values[:, column]
        is_binary = bool(np.all((col == 0.0) | (col == 1.0)))
        is_count = bool(np.all(col >= 0.0) and np.all(col == np.floor(col)))
        if is_binary:
            gates.append(ColumnGate(
                column=column,
                candidates=("bernoulli", "poisson"),
                decided_by="score",
                gate_reason="all values in {0,1}; both families share the "
                            "counting measure, so the comparison is legitimate",
            ))
        elif is_count:
            gates.append(ColumnGate(
                column=column,
                candidates=("poisson",),
                decided_by="gate",
                gate_reason="non-negative integers with a value above 1; "
                            "Bernoulli support is violated",
            ))
        else:
            gates.append(ColumnGate(
                column=column,
                candidates=("gaussian",),
                decided_by="gate",
                gate_reason="a non-integer or negative value is present; "
                            "no implemented discrete family admits it",
            ))
    return gates


def initial_assignment(gates: Sequence[ColumnGate], ambiguous_start: str) -> list[str]:
    """Starting assignment: gate-fixed columns, ambiguous columns at ``ambiguous_start``."""

    _require(ambiguous_start in VALID_FAMILIES,
             f"ambiguous_start must be one of {VALID_FAMILIES}")
    assignment: list[str] = []
    for gate in gates:
        if gate.decided_by == "gate":
            assignment.append(gate.candidates[0])
        else:
            _require(ambiguous_start in gate.candidates,
                     f"column {gate.column}: start {ambiguous_start!r} is not "
                     f"among its candidates {gate.candidates}")
            assignment.append(ambiguous_start)
    return assignment


# --------------------------------------------------------------------------
# A2. per-column candidate optimiser and strict score
# --------------------------------------------------------------------------

def _column_eta(Z_samples: np.ndarray, loading: np.ndarray) -> np.ndarray:
    """eta^(s)_i = f_l^T z_i^(s); shape (n, L)."""

    return np.einsum("nkl,k->nl", Z_samples, loading)


def column_log_likelihood(
    x_column: np.ndarray,
    Z_samples: np.ndarray,
    loading: np.ndarray,
    family: str,
    *,
    sigma_sq: float | None = None,
) -> float:
    """Strict COMPLETE per-column score ``(1/L) sum_s sum_i log p(x_il | eta)``.

    "Complete" means every term of the log probability / density is included,
    including the base measure.  A family comparison may not drop a term just
    because it is constant in the parameters: the dropped term generally
    differs between candidates (design section 4).  Concretely:

    * Poisson keeps ``-log(x!)``.  It is computed, never assumed away.  For a
      0/1 column it happens to evaluate to exactly 0 because ``0! = 1! = 1``,
      but that is a property of the data, not a licence to omit the term.
    * Gaussian keeps ``-0.5 log(2 pi)`` and ``-0.5 log sigma^2``.
    * Bernoulli has base measure ``h(x) = 1``, contributing 0.
    """

    _require(family in VALID_FAMILIES, f"unknown family {family!r}")
    x = np.asarray(x_column, dtype=np.float64)
    eta = _column_eta(Z_samples, np.asarray(loading, dtype=np.float64))
    n, n_samples = eta.shape
    _require(x.shape == (n,), f"x column shape {x.shape} != {(n,)}")

    if family == "bernoulli":
        per_sample = bernoulli_log_likelihood(x[:, None], eta)
        total = float(np.sum(per_sample))
    elif family == "poisson":
        per_sample = poisson_log_likelihood(x[:, None], eta)
        # The base measure term.  Constant in f_l, NOT constant across
        # families, so it stays in the comparison.
        base_measure = -float(np.sum(gammaln(x + 1.0))) * n_samples
        total = float(np.sum(per_sample)) + base_measure
    else:                                                    # gaussian
        residual = x[:, None] - eta
        if sigma_sq is None:
            sigma_sq = float(np.mean(residual ** 2))
        sigma_sq = _checked_variance(sigma_sq, source="column_log_likelihood")
        total = float(np.sum(
            -0.5 * residual ** 2 / sigma_sq
            - 0.5 * math.log(sigma_sq)
            - 0.5 * math.log(2.0 * math.pi)
        ))

    score = total / n_samples
    _require(math.isfinite(score),
             f"non-finite candidate score for family {family!r}")
    return score


def _column_gradient(
    x_column: np.ndarray,
    Z_samples: np.ndarray,
    loading: np.ndarray,
    family: str,
    sigma_sq: float | None,
) -> np.ndarray:
    """d/df_l of the L-averaged column log-likelihood.

    For a canonical link the score is ``x - mu(eta)``; the Gaussian column
    additionally divides by its variance.
    """

    x = np.asarray(x_column, dtype=np.float64)
    eta = _column_eta(Z_samples, loading)
    if family == "bernoulli":
        residual = x[:, None] - bernoulli_mean(eta)
    elif family == "poisson":
        residual = x[:, None] - poisson_mean(eta)
    else:                                                    # gaussian
        variance = _checked_variance(
            sigma_sq if sigma_sq is not None
            else float(np.mean((x[:, None] - eta) ** 2)),
            source="_column_gradient")
        residual = (x[:, None] - eta) / variance
    # (1/L) sum_s Z_s^T residual_s
    gradient = np.einsum("nl,nkl->k", residual, Z_samples) / eta.shape[1]
    _require(bool(np.all(np.isfinite(gradient))),
             f"non-finite gradient for family {family!r}")
    return gradient


def optimise_column_loading(
    x_column: np.ndarray,
    Z_samples: np.ndarray,
    family: str,
    *,
    loading_init: np.ndarray,
    max_iter: int = ADAM_MAX_ITER,
    lr: float = ADAM_LR,
    beta1: float = ADAM_BETA1,
    beta2: float = ADAM_BETA2,
    eps: float = ADAM_EPS,
    tol: float = ADAM_TOL,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[np.ndarray, float | None, int, bool]:
    """Optimise ``f_l`` for ONE candidate family on FIXED posterior samples.

    Every candidate gets its own ``f_l``: a different link implies a different
    optimum, so reusing one loading across candidates would compare the
    families at the wrong parameter values (design section 2.1).  Every
    candidate also gets the SAME ``Z_samples``, the same optimiser budget and
    the same tolerance, so a score difference cannot come from unequal effort.

    Both candidates also start from the SAME loading: the incumbent row of the
    current ``F``.  That is the only shared starting point available inside the
    EM, but it is not neutral -- the incumbent family's row is already near ITS
    optimum, while the challenger may need more than ``max_iter`` steps to
    reach its own.  A challenger that stops early scores too low, which biases
    the comparison toward the incumbent and is one concrete mechanism behind
    the path dependence that design section 7 flags for scheme A.  The returned
    ``converged`` flag makes that observable instead of silent: an audit can
    count the candidates that used the whole budget, and a margin backed by a
    non-converged challenger should not be read as a confident decision.

    ``diagnostics`` is an optional dict filled in with per-step quantities --
    gradient and step infinity norms, and the objective before and after each
    step.  It is a RECORDING channel only: nothing it captures feeds back into
    the update, and with it omitted (the default) not a single extra floating
    point operation enters the numerical path.  A test pins that the returned
    values are bitwise identical with and without it.  It exists so that the
    optimiser can be diagnosed without the diagnosis changing what it does.

    Returns ``(loading, sigma_sq, n_iter, converged)``; ``sigma_sq`` is the
    profiled Gaussian variance, and ``None`` for the other families.
    """

    _require(family in VALID_FAMILIES, f"unknown family {family!r}")
    loading = np.array(loading_init, dtype=np.float64).copy()
    _require(loading.shape == (Z_samples.shape[1],),
             f"loading shape {loading.shape} != {(Z_samples.shape[1],)}")
    _require(bool(np.all(np.isfinite(loading))), "loading_init is not finite")

    first_moment = np.zeros_like(loading)
    second_moment = np.zeros_like(loading)
    used_iter = 0
    converged = False
    steps: list[dict[str, float]] | None = None
    if diagnostics is not None:
        steps = []
        diagnostics["steps"] = steps

    for step in range(1, max_iter + 1):
        used_iter = step
        sigma_sq = None
        if family == "gaussian":
            residual = np.asarray(x_column, dtype=np.float64)[:, None] - \
                _column_eta(Z_samples, loading)
            sigma_sq = _checked_variance(float(np.mean(residual ** 2)),
                                         source="optimise_column_loading")
        gradient = _column_gradient(x_column, Z_samples, loading, family, sigma_sq)

        # Same sign convention as _calc_F_adam_weighted: ascend the
        # log-likelihood by descending its negative.
        descent = -gradient
        first_moment = beta1 * first_moment + (1.0 - beta1) * descent
        second_moment = beta2 * second_moment + (1.0 - beta2) * descent ** 2
        first_hat = first_moment / (1.0 - beta1 ** step)
        second_hat = second_moment / (1.0 - beta2 ** step)
        proposal = loading - lr * first_hat / (np.sqrt(second_hat) + eps)
        _require(bool(np.all(np.isfinite(proposal))),
                 f"candidate optimiser produced a non-finite loading for {family!r}")
        converged = bool(np.max(np.abs(proposal - loading)) < tol)
        if steps is not None:
            objective_before = column_log_likelihood(
                x_column, Z_samples, loading, family, sigma_sq=sigma_sq)
            objective_after = column_log_likelihood(
                x_column, Z_samples, proposal, family, sigma_sq=sigma_sq)
            steps.append({
                "step": step,
                "gradient_inf": float(np.max(np.abs(gradient))),
                "step_inf": float(np.max(np.abs(proposal - loading))),
                "objective_before": objective_before,
                "objective_after": objective_after,
                "objective_change": objective_after - objective_before,
            })
        loading = proposal
        if converged:
            break

    sigma_sq = None
    if family == "gaussian":
        residual = np.asarray(x_column, dtype=np.float64)[:, None] - \
            _column_eta(Z_samples, loading)
        sigma_sq = _checked_variance(float(np.mean(residual ** 2)),
                                     source="optimise_column_loading")
    if diagnostics is not None:
        diagnostics["final_gradient_inf"] = float(np.max(np.abs(
            _column_gradient(x_column, Z_samples, loading, family, sigma_sq))))
        diagnostics["final_step_inf"] = (steps[-1]["step_inf"] if steps
                                         else float("nan"))
        diagnostics["last_objective_change"] = (steps[-1]["objective_change"]
                                                if steps else float("nan"))
        diagnostics["n_iter"] = used_iter
        diagnostics["converged"] = converged
    return loading, sigma_sq, used_iter, converged


def optimise_column_loading_bfgs(
    x_column: np.ndarray,
    Z_samples: np.ndarray,
    family: str,
    *,
    loading_init: np.ndarray,
    maxiter: int = BFGS_MAXITER,
    gtol: float = BFGS_GTOL,
) -> tuple[np.ndarray, float | None, int, bool, dict[str, Any]]:
    """Optimise ``f_l`` for ONE candidate family with analytic-gradient BFGS.

    The objective is the negative of the production strict complete score and
    the Jacobian is the negative of the production analytic gradient -- the
    same pair Gate 74-B1 validated against an independently written objective
    and a central difference.  Nothing is differenced numerically here.

    Convergence is decided on the gradient actually reached, not on SciPy
    ``success``: Gate 74-B1 recorded a solver reporting success at a point
    whose gradient was two orders of magnitude above the criterion, so the flag
    is provenance rather than a verdict.

    There is no fallback solver, no retry from another initialisation and no
    reseeding.  A candidate that does not reach the criterion is reported as
    not converged, and the pilot gate is what acts on that.
    """

    _require(family in VALID_FAMILIES, f"unknown family {family!r}")
    x = np.asarray(x_column, dtype=np.float64)
    start = np.array(loading_init, dtype=np.float64).copy()
    _require(start.shape == (Z_samples.shape[1],),
             f"loading shape {start.shape} != {(Z_samples.shape[1],)}")
    _require(bool(np.all(np.isfinite(start))), "loading_init is not finite")

    def _profiled_variance(loading: np.ndarray) -> float | None:
        if family != "gaussian":
            return None
        residual = x[:, None] - _column_eta(Z_samples, loading)
        return _checked_variance(float(np.mean(residual ** 2)),
                                 source="optimise_column_loading_bfgs")

    def negative(loading: np.ndarray) -> float:
        loading = np.asarray(loading, dtype=np.float64)
        return -column_log_likelihood(x, Z_samples, loading, family,
                                      sigma_sq=_profiled_variance(loading))

    def negative_jac(loading: np.ndarray) -> np.ndarray:
        loading = np.asarray(loading, dtype=np.float64)
        return -_column_gradient(x, Z_samples, loading, family,
                                 _profiled_variance(loading))

    result = minimize(negative, start, method=BFGS_METHOD, jac=negative_jac,
                      options={"maxiter": maxiter, "gtol": gtol})
    loading = np.asarray(result.x, dtype=np.float64)
    _require(bool(np.all(np.isfinite(loading))),
             f"BFGS produced a non-finite loading for {family!r}")
    sigma_sq = _profiled_variance(loading)
    gradient_inf = float(np.max(np.abs(
        _column_gradient(x, Z_samples, loading, family, sigma_sq))))
    converged = bool(math.isfinite(gradient_inf)
                     and gradient_inf <= CANDIDATE_GRAD_INF_TOL)
    provenance = {
        "optimizer": OPTIMIZER_BFGS,
        "method": BFGS_METHOD,
        "maxiter": int(maxiter),
        "gtol": float(gtol),
        "convergence_grad_inf_tol": CANDIDATE_GRAD_INF_TOL,
        "gradient_inf": gradient_inf,
        "scipy_success": bool(result.success),
        "scipy_status": int(result.status),
        "scipy_message": str(result.message),
        "scipy_nit": int(result.nit),
        "scipy_njev": int(getattr(result, "njev", -1)),
        "fallback_solvers": [],
        "retries": 0,
    }
    return loading, sigma_sq, int(result.nit), converged, provenance


def optimise_candidate_loading(
    x_column: np.ndarray,
    Z_samples: np.ndarray,
    family: str,
    *,
    loading_init: np.ndarray,
    optimizer: str,
) -> tuple[np.ndarray, float | None, int, bool, dict[str, Any]]:
    """Route a candidate optimisation to the named optimiser.

    ``optimizer`` has no default on purpose.  Phase 9C uses BFGS by explicit
    configuration and the historical Adam route stays reachable by name; a
    default here would let a call site drift between them silently.
    """

    _require(optimizer in VALID_OPTIMIZERS,
             f"unknown optimizer {optimizer!r}; choose from {VALID_OPTIMIZERS}")
    if optimizer == OPTIMIZER_BFGS:
        return optimise_column_loading_bfgs(
            x_column, Z_samples, family, loading_init=loading_init)

    loading, sigma_sq, n_iter, converged = optimise_column_loading(
        x_column, Z_samples, family, loading_init=loading_init)
    # The Adam route keeps its own step-based convergence rule unchanged. The
    # gradient norm is recorded alongside so both routes report the same
    # diagnostic, without redefining what Adam means by converged.
    gradient_inf = float(np.max(np.abs(
        _column_gradient(x_column, Z_samples, loading, family, sigma_sq))))
    provenance = {
        "optimizer": OPTIMIZER_ADAM,
        "method": "adam",
        "max_iter": ADAM_MAX_ITER,
        "lr": ADAM_LR,
        "beta1": ADAM_BETA1,
        "beta2": ADAM_BETA2,
        "eps": ADAM_EPS,
        "tol": ADAM_TOL,
        "convergence_rule": "step infinity norm below tol",
        "gradient_inf": gradient_inf,
        "fallback_solvers": [],
        "retries": 0,
    }
    return loading, sigma_sq, n_iter, converged, provenance


@dataclass
class CandidateRecord:
    """One (column, candidate) evaluation."""

    column: int
    family: str
    score: float
    loading: np.ndarray
    sigma_sq: float | None
    n_iter: int
    converged: bool = True
    optimizer: str = OPTIMIZER_ADAM
    gradient_inf: float = float("nan")
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "candidate_family": self.family,
            "score": self.score,
            "neg2_score": -2.0 * self.score,
            "sigma_sq": "" if self.sigma_sq is None else self.sigma_sq,
            "optimizer": self.optimizer,
            "optimiser_iterations": self.n_iter,
            "optimiser_converged": self.converged,
            "optimiser_gradient_inf": self.gradient_inf,
            "scipy_success": self.provenance.get("scipy_success", ""),
            "loading_norm": float(np.linalg.norm(self.loading)),
        }


def score_column_candidates(
    x_column: np.ndarray,
    Z_samples: np.ndarray,
    gate: ColumnGate,
    *,
    loading_init: np.ndarray,
    optimizer: str,
) -> list[CandidateRecord]:
    """Optimise and score every candidate of one column, under equal effort.

    ``optimizer`` is required rather than defaulted: which optimiser produced a
    candidate score is part of what the score means, and Gate 74-B1R showed the
    two routes land in very different places.
    """

    records: list[CandidateRecord] = []
    for family in gate.candidates:
        loading, sigma_sq, n_iter, converged, provenance = (
            optimise_candidate_loading(x_column, Z_samples, family,
                                       loading_init=loading_init,
                                       optimizer=optimizer))
        score = column_log_likelihood(
            x_column, Z_samples, loading, family, sigma_sq=sigma_sq)
        records.append(CandidateRecord(
            column=gate.column, family=family, score=score,
            loading=loading, sigma_sq=sigma_sq, n_iter=n_iter,
            converged=converged, optimizer=optimizer,
            gradient_inf=provenance["gradient_inf"], provenance=provenance))
    return records


def loading_digest(loading: np.ndarray) -> str:
    """A stable digest of a loading vector, for installation evidence.

    Computed over the exact float64 bytes, so two digests agree only if the
    installed row is bit-for-bit the selected candidate's loading.
    """

    import hashlib

    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(loading, dtype=np.float64)).tobytes()).hexdigest()


def _candidate_provenance_fields(records: Sequence[CandidateRecord]
                                 ) -> dict[str, Any]:
    """Per-candidate diagnostics for one selection-trace row.

    Purely observational: every value is read from a CandidateRecord that has
    already been computed, so adding these fields runs no optimiser and
    changes no number. The aggregate ``all_candidates_converged`` continues to
    carry the gate; these say how far each candidate actually got.
    """

    fields: dict[str, Any] = {}
    for record in records:
        prefix = record.family
        provenance = record.provenance or {}
        fields[f"{prefix}_optimizer"] = record.optimizer
        fields[f"{prefix}_n_iter"] = record.n_iter
        fields[f"{prefix}_converged"] = record.converged
        fields[f"{prefix}_grad_inf"] = record.gradient_inf
        fields[f"{prefix}_scipy_success"] = provenance.get("scipy_success", "")
        fields[f"{prefix}_scipy_status"] = provenance.get("scipy_status", "")
    return fields


def select_from_records(records: Sequence[CandidateRecord]) -> tuple[str, float]:
    """Pick the best candidate and report its margin, with a deterministic tie rule.

    The margin is ``(-2 * score_best) - (-2 * score_runner_up)``, i.e. how much
    worse the runner-up is on the reported criterion scale.  It is 0.0 when a
    column has a single candidate, and negative values are impossible by
    construction.  Recording the margin -- rather than inventing a discrete
    search penalty -- is what HG-3 asks of the pilot.
    """

    _require(len(records) >= 1, "no candidate records to select from")
    for record in records:
        # score_column_candidates cannot produce these, but this function is
        # public and a non-finite score must never reach a selection.
        _require(math.isfinite(record.score),
                 f"column {record.column}: candidate {record.family!r} has a "
                 f"non-finite score ({record.score})")
    best = max(
        records,
        key=lambda record: (record.score,
                            -CANDIDATE_PRIORITY.index(record.family)),
    )
    if len(records) == 1:
        return best.family, 0.0
    runner_up = max((r for r in records if r is not best),
                    key=lambda record: record.score)
    margin = (-2.0 * runner_up.score) - (-2.0 * best.score)
    return best.family, float(margin)


# --------------------------------------------------------------------------
# A4a. the model class that performs the A-type in-EM family update
# --------------------------------------------------------------------------

class FamilySelectingPerColumnLSM(DualExpFamLSMPerColumnConsistent):
    """Per-column model that re-selects ambiguous families inside the M-step.

    The update is placed in ``calc_F`` so that it happens exactly where the
    design says it can: on the FIXED posterior samples of the current E-step,
    where ``Q_Z`` and ``Q_Y`` do not depend on the assignment and therefore
    cancel from the per-column comparison (design section 2.3).  Re-selecting
    before delegating to the parent's ``calc_F`` means the subsequent
    ``calc_sigma``, the X log-likelihood and the next E-step all see the new
    assignment, which is what an A-type update means here.
    """

    def __init__(self, *, gates: Sequence[ColumnGate],
                 candidate_optimizer: str = PHASE9C_CANDIDATE_OPTIMIZER,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _require(len(gates) == self.d,
                 f"need one gate per column: {len(gates)} gates, d={self.d}")
        _require(candidate_optimizer in VALID_OPTIMIZERS,
                 f"unknown candidate_optimizer {candidate_optimizer!r}")
        self.gates = list(gates)
        # Named here rather than inherited from a default deeper down, so the
        # artifact can say which optimiser produced every candidate score.
        self.candidate_optimizer = candidate_optimizer
        # The winning CandidateRecord per ambiguous column for the CURRENT
        # selection only. Replaced wholesale at every select_families call so
        # a record from an earlier iteration can never be installed.
        self.current_iteration_winners: dict[int, CandidateRecord] = {}
        self._current_iteration_trace_rows: list[dict[str, Any]] = []
        self.family_update_enabled = True
        self.selection_trace: list[dict[str, Any]] = []
        self.last_candidate_records: list[CandidateRecord] = []
        self._iteration = 0

    def reassign_families(self, family_x_list: Sequence[str]) -> None:
        """Install a new per-column assignment without rebuilding the model."""

        family_x_list = [str(f) for f in family_x_list]
        _require(len(family_x_list) == self.d,
                 f"family_x_list length {len(family_x_list)} != d={self.d}")
        for family in family_x_list:
            _require(family in VALID_FAMILIES, f"unknown family {family!r}")
        self.family_x_list = family_x_list
        self._col_idx = {
            fam: np.array([j for j, f in enumerate(family_x_list) if f == fam],
                          dtype=int)
            for fam in VALID_FAMILIES
        }
        self._all_gaussian = (len(self._col_idx["gaussian"]) == self.d)
        self.family_x = "gaussian" if self._all_gaussian else "mixed"

    def select_families(self, X: np.ndarray, Z_samples: np.ndarray) -> list[str]:
        """Re-select ambiguous columns on the given fixed posterior samples."""

        assignment = list(self.family_x_list)
        records: list[CandidateRecord] = []
        loadings = self.params["F"]
        # Fresh per-call state: nothing selected in an earlier iteration may
        # survive into this one.
        winners: dict[int, CandidateRecord] = {}
        iteration_rows: list[dict[str, Any]] = []
        for gate in self.gates:
            if not gate.is_ambiguous:
                continue
            column_records = score_column_candidates(
                X[:, gate.column], Z_samples, gate,
                loading_init=loadings[gate.column, :],
                optimizer=self.candidate_optimizer)
            chosen, margin = select_from_records(column_records)
            # The record whose score won. Family, score and loading all come
            # from this one object, which is what Gate 74-B5 requires.
            winner, = [record for record in column_records
                       if record.family == chosen]
            winners[gate.column] = winner
            records.extend(column_records)
            assignment[gate.column] = chosen
            row = {
                "iteration": self._iteration,
                "column": gate.column,
                "candidate_optimizer": self.candidate_optimizer,
                "previous_family": self.family_x_list[gate.column],
                "selected_family": chosen,
                "margin_neg2": margin,
                "changed": chosen != self.family_x_list[gate.column],
                # A margin whose loser never converged is not a confident
                # decision; record it rather than let the margin stand alone.
                # This aggregate keeps its original meaning: the strict gate
                # reads it, and B4 did not change what it means.
                "all_candidates_converged": all(r.converged
                                                for r in column_records),
                **{f"score_{r.family}": r.score for r in column_records},
                # Per-candidate diagnostics. Gate 74-B3 recorded only the
                # aggregate above, so when seven intermediate iterations came
                # back false there was no way to tell how far off they were.
                # Read straight off the records that were already computed:
                # no optimiser runs again for this.
                **_candidate_provenance_fields(column_records),
                # Selected-loading evidence (Gate 74-B5). The installation
                # fields are filled in by calc_F once the row is actually
                # written into F; a row selected outside calc_F stays
                # uninstalled and says so.
                "selected_candidate_family": winner.family,
                "selected_candidate_score": winner.score,
                "selected_loading": "|".join(f"{v:.17g}"
                                             for v in winner.loading),
                "selected_loading_sha256": loading_digest(winner.loading),
                "loading_installation_policy": LOADING_INSTALLATION_POLICY,
                "selected_loading_installed": False,
                "installed_loading_sha256": "",
            }
            self.selection_trace.append(row)
            iteration_rows.append(row)
        self.last_candidate_records = records
        self.current_iteration_winners = winners
        self._current_iteration_trace_rows = iteration_rows
        return assignment

    def calc_F(self, X: np.ndarray, Z_samples: np.ndarray) -> np.ndarray:
        """Select ambiguous families, update F, and install the winners.

        Gate 74-B4 found that the winning candidate's loading was used only to
        score it and then discarded, so an ambiguous column carried forward the
        parent 50-step Adam row instead of the optimum that produced its score.
        Gate 74-B5 closes exactly that gap and nothing else:

        1. score and select exactly as before, on this call's fixed Z_samples;
        2. install the winning family name exactly as before;
        3. run the parent calc_F for the whole of F, so gate-decided rows keep
           their existing behaviour;
        4. overwrite ONLY each ambiguous row with the loading of the winning
           record from this same call, and record that it happened.

        No optimiser runs after selection, losing loadings are never
        installed, and a record from an earlier call cannot be installed
        because the winners are rebuilt on every selection and consumed here.
        """

        if not self.family_update_enabled:
            # No selection this call, so nothing may be installed -- not even
            # a winner left over from an earlier one.
            self.current_iteration_winners = {}
            self._current_iteration_trace_rows = []
            return super().calc_F(X, Z_samples)

        self._iteration += 1
        self.reassign_families(self.select_families(X, Z_samples))
        winners = self.current_iteration_winners
        rows = {row["column"]: row for row in self._current_iteration_trace_rows}

        F = np.array(super().calc_F(X, Z_samples), dtype=np.float64, copy=True)
        for column, winner in winners.items():
            _require(winner.family == self.family_x_list[column],
                     f"column {column}: the installed family "
                     f"{self.family_x_list[column]!r} is not the winning "
                     f"record's family {winner.family!r}")
            _require(bool(np.all(np.isfinite(winner.loading))),
                     f"column {column}: the winning loading is not finite")
            F[column, :] = winner.loading
            row = rows[column]
            row["selected_loading_installed"] = True
            row["installed_loading_sha256"] = loading_digest(F[column, :])

        # Consumed: the next call starts from nothing.
        self.current_iteration_winners = {}
        self._current_iteration_trace_rows = []
        return F

    def __repr__(self) -> str:                               # pragma: no cover
        counts = {f: len(idx) for f, idx in self._col_idx.items() if len(idx)}
        return (f"FamilySelectingPerColumnLSM(n={self.n}, d={self.d}, "
                f"k={self.k}, family_x_cols={counts}, "
                f"family_y='{self.family}')")


# --------------------------------------------------------------------------
# execution-integrity gate: did the candidate scores get optimised at all?
# --------------------------------------------------------------------------

PILOT_GATE_PASS = "READY_FOR_PILOT"
PILOT_GATE_BLOCKED = "BLOCKED_FOR_PILOT"


def pilot_convergence_gate(
    selection_trace: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Block the pilot if any ambiguous column's candidates failed to converge.

    This is NOT an accuracy threshold.  A margin whose loser stopped before
    reaching its own optimum is not a comparison of two fitted families, it is
    a comparison of one fitted family against an unfinished one -- so the
    frozen score was never actually evaluated.  That is an execution-integrity
    condition, and it is checked before the pilot rather than argued about
    afterwards.

    When this blocks, the response is to stop and hand the finding to a human.
    Raising the optimiser budget, changing the seed or the initialisation, or
    rerunning would all be tuning the protocol against an observed result.
    """

    offending = [
        {
            "iteration": row.get("iteration"),
            "column": row.get("column"),
            "selected_family": row.get("selected_family"),
            "margin_neg2": row.get("margin_neg2"),
        }
        for row in selection_trace
        if row.get("all_candidates_converged") is False
    ]
    non_converged_rows = [row for row in candidate_rows
                          if row.get("optimiser_converged") is False]
    blocked = bool(offending) or bool(non_converged_rows)
    return {
        "status": PILOT_GATE_BLOCKED if blocked else PILOT_GATE_PASS,
        "non_converged_selection_rows": offending,
        "non_converged_candidate_count": len(non_converged_rows),
        "checked_selection_rows": len(selection_trace),
        "remedy_forbidden": [
            "raise the optimiser budget",
            "change the seed or the initialisation",
            "change the optimiser",
            "rerun",
        ],
    }


# --------------------------------------------------------------------------
# A4b. the hybrid driver: exploration, then a fresh fixed-family refit
# --------------------------------------------------------------------------

@dataclass
class ExplorationResult:
    """Outcome of the exploration stage (the A-type search)."""

    selected_assignment: list[str]
    initial_assignment: list[str]
    selection_trace: list[dict[str, Any]]
    final_candidate_rows: list[dict[str, Any]]
    margins: dict[int, float]
    iterations: int
    metadata: dict[str, Any] = field(default_factory=dict)


def run_family_exploration(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    k: int,
    gates: Sequence[ColumnGate],
    initial_families: Sequence[str],
    family_y: str = "bernoulli",
    L: int = 5,
    num_iter: int = 8,
    seed: int = 42,
    newton_alpha: float = 0.5,
    verbose: bool = False,
    candidate_optimizer: str = PHASE9C_CANDIDATE_OPTIMIZER,
) -> ExplorationResult:
    """Exploration MCEM with an A-type family update inside each M-step.

    The loop mirrors ``em_runner.run_em_experimental`` -- same informed
    initialisation, the same ``scale_Z`` placement, the same M-step order --
    with two deliberate differences, both required by the Issue #74 protocol:

    * the family assignment is updated inside ``calc_F`` (the A-type step);
    * a non-finite E-step is FAIL-FAST.  ``run_em_experimental`` guards NaNs
      and retries with a halved Newton step; here a numerical failure is the
      observation, so it is raised rather than repaired.

    The reported fit does NOT come from this loop.  It comes from the fresh
    fixed-assignment refit in ``run_hybrid_family_selection``, which uses the
    unmodified ``run_em_experimental``.
    """

    n, d = X.shape
    rng = np.random.default_rng(seed)
    model = FamilySelectingPerColumnLSM(
        gates=gates, n=n, d=d, k=k, L=L,
        family_x_list=list(initial_families), family_y=family_y,
        candidate_optimizer=candidate_optimizer)
    model.initialize_params(true_params=None, seed=seed)

    # Informed init, Y side (mirrors em_runner.run_em_experimental).
    obs_upper = np.triu(model.train_mask, k=1)
    y_obs = Y[obs_upper]
    if family_y == "bernoulli":
        observed_density = float(y_obs.mean())
        # em_runner clips this density so that a degenerate graph still yields
        # a finite w0.  Here a graph with no edges, or with every edge present,
        # carries no relational signal at all, so continuing would produce a
        # meaningless exploration fit from an arbitrary w0.  Stop instead; the
        # clip below then never actually binds on a usable dataset.
        _require(0.0 < observed_density < 1.0,
                 f"observed Y density is {observed_density}; a graph with no "
                 f"edges or with every edge present carries no relational "
                 f"signal, and the pilot refuses to substitute a clipped w0")
        density = float(np.clip(observed_density, 1e-6, 1 - 1e-6))
        model.params["w0"] = np.log(density / (1 - density))
        model.params["w"] = 0.5
    elif family_y == "poisson":
        mean_count = float(y_obs[y_obs > 0].mean()) if np.any(y_obs > 0) else 1.0
        model.params["w0"] = np.log(mean_count + 1e-10)
        model.params["w"] = 0.1
    else:                                                    # gaussian
        model.params["w0"] = float(y_obs.mean())
        model.params["w"] = 0.5
        model.sigma_y = float(max(y_obs.std(), 0.01))

    # Informed init, X side: shrink the rows of non-Gaussian columns so eta
    # starts inside a numerically safe range (mirrors em_runner).
    for column, family in enumerate(initial_families):
        if family in ("bernoulli", "poisson"):
            model.params["F"][column, :] *= 0.2

    Z = model.params["Z"].copy()
    F = model.params["F"].copy()
    sigma = model.params["sigma"].copy()
    w0 = float(model.params["w0"])
    w = float(model.params["w"])

    for iteration in range(1, num_iter + 1):
        Z_samples = np.zeros((n, k, L))
        for sample in range(L):
            model.params.update(dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w))
            Z_new = model.calc_eta_newton(
                X, Y, rng=rng, max_iter=10, alpha=newton_alpha)
            Z_samples[:, :, sample] = Z_new
            Z = Z_new.copy()

        _require(bool(np.all(np.isfinite(Z_samples))),
                 f"exploration E-step produced a non-finite Z at iteration "
                 f"{iteration}; the pilot fails fast instead of repairing it")

        Z_samples = model.scale_Z(Z_samples)
        Z = Z_samples[:, :, -1].copy()

        F = model.calc_F(X, Z_samples)          # <- A-type family update here
        sigma = model.calc_sigma(X, Z_samples, F)
        w0 = float(model.calc_w0(Y, Z_samples, w0, w, max_iter=50))
        w = float(model.calc_w(Y, Z_samples, w0, w, max_iter=50))
        if family_y == "gaussian":
            model.calc_sigma_y(Y, Z_samples, w0, w)
        if verbose:
            print(f"  [exploration iter={iteration}] "
                  f"assignment={model.family_x_list}")

    last_iteration = max((row["iteration"] for row in model.selection_trace),
                         default=0)
    margins = {row["column"]: row["margin_neg2"]
               for row in model.selection_trace
               if row["iteration"] == last_iteration}

    return ExplorationResult(
        selected_assignment=list(model.family_x_list),
        initial_assignment=list(initial_families),
        selection_trace=list(model.selection_trace),
        final_candidate_rows=[r.as_row() for r in model.last_candidate_records],
        margins=margins,
        iterations=num_iter,
        metadata={
            "selector_version": SELECTOR_VERSION,
            "numerics_mode": "consistent",
            "candidate_optimizer": candidate_optimizer,
            "candidate_optimizer_settings": (
                {"method": BFGS_METHOD, "maxiter": BFGS_MAXITER,
                 "gtol": BFGS_GTOL,
                 "convergence_grad_inf_tol": CANDIDATE_GRAD_INF_TOL}
                if candidate_optimizer == OPTIMIZER_BFGS else
                {"max_iter": ADAM_MAX_ITER, "lr": ADAM_LR,
                 "beta1": ADAM_BETA1, "beta2": ADAM_BETA2,
                 "eps": ADAM_EPS, "tol": ADAM_TOL,
                 "convergence_rule": "step infinity norm below tol"}),
            "historical_adam_preserved": True,
            "ambiguous_loading_installation": LOADING_INSTALLATION_POLICY,
            "newton_alpha": float(newton_alpha),
            "L": int(L),
            "num_iter": int(num_iter),
            "seed": int(seed),
            "adam": {"max_iter": ADAM_MAX_ITER, "lr": ADAM_LR,
                     "beta1": ADAM_BETA1, "beta2": ADAM_BETA2,
                     "eps": ADAM_EPS, "tol": ADAM_TOL},
            "tie_rule": list(CANDIDATE_PRIORITY),
            # There is no retry path to count: a non-finite E-step raises.
            "retry_policy": "fail_fast_no_retry",
            "last_selection_iteration": int(last_iteration),
        },
    )


def run_hybrid_family_selection(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    k: int,
    ambiguous_start: str,
    family_y: str = "bernoulli",
    L: int = 5,
    exploration_num_iter: int = 8,
    refit_num_iter: int = 8,
    search_seed: int = 42,
    refit_seed: int = 43,
    verbose: bool = False,
    execution_hook: Any = None,
    candidate_optimizer: str = PHASE9C_CANDIDATE_OPTIMIZER,
) -> dict[str, Any]:
    """Scheme C: A-type exploration, then a fresh fixed-assignment refit.

    The refit is a plain ``run_em_experimental`` call with the selected
    assignment held fixed, so the reported ``Q_strict`` / criterion is NOT
    conditioned on the search path and comes from the same audited runner the
    rest of the repository uses (design section 7).

    Per HG-1 the pilot performs exactly one exploration and one refit; it does
    not iterate further outer cycles, and this function makes no claim about
    how many cycles would be needed for the assignment to stop changing.

    ``execution_hook`` is an optional callable ``hook(kind, status, info)``
    invoked immediately BEFORE each of the two EM executions with status
    ``"STARTED"`` and again after it returns with ``"SUCCESS"``.  ``kind`` is
    ``"exploration"`` or ``"refit"``.  It exists so that a caller can record an
    attempt on disk before the attempt can fail: a run that dies mid-EM must
    still leave evidence of what it had already started, and a counter
    incremented only on success would undercount real EM work.  Default
    ``None`` calls nothing and changes nothing.
    """

    def _notify(kind: str, status: str, seed: int) -> None:
        if execution_hook is not None:
            execution_hook(kind, status, {"seed": int(seed),
                                          "k": int(k),
                                          "L": int(L),
                                          "ambiguous_start": ambiguous_start})

    gates = support_gate(X)
    start = initial_assignment(gates, ambiguous_start)
    _notify("exploration", "STARTED", search_seed)
    exploration = run_family_exploration(
        X, Y, k=k, gates=gates, initial_families=start, family_y=family_y,
        L=L, num_iter=exploration_num_iter, seed=search_seed, verbose=verbose,
        candidate_optimizer=candidate_optimizer)
    _notify("exploration", "SUCCESS", search_seed)

    # failure_policy='fail_fast' is required here, not optional. The legacy
    # policy repairs a non-finite E-step by substituting Z_prev and then
    # retries on a DIFFERENT seed, so a repaired refit would report as clean
    # and the pilot's "retry = replacement = seed rescue = 0" condition would
    # be unverifiable from the artifact.
    _notify("refit", "STARTED", refit_seed)
    refit = run_em_experimental(
        X, Y, family_x="mixed", family_y=family_y, k=k, L=L,
        num_iter=refit_num_iter, seed=refit_seed,
        family_x_list=exploration.selected_assignment,
        compute_strict_Q=True, numerics_mode="consistent",
        failure_policy="fail_fast", verbose=verbose)
    _notify("refit", "SUCCESS", refit_seed)
    _require(refit.get("failure_policy") == "fail_fast",
             "the reported refit did not run under failure_policy='fail_fast'")
    for counter in ("retry_count", "replacement_count", "seed_rescue_count"):
        _require(int(refit.get(counter, -1)) == 0,
                 f"refit reported {counter}={refit.get(counter)}; the pilot "
                 f"requires it to be 0")

    n_gaussian = sum(1 for f in exploration.selected_assignment if f == "gaussian")
    return {
        "gates": [gate.as_row() for gate in gates],
        "ambiguous_columns": [g.column for g in gates if g.is_ambiguous],
        "initial_assignment": exploration.initial_assignment,
        "selected_assignment": exploration.selected_assignment,
        "selection_trace": exploration.selection_trace,
        "candidate_rows": exploration.final_candidate_rows,
        "margins": exploration.margins,
        "exploration_metadata": exploration.metadata,
        "refit": refit,
        "n_gaussian_x_cols": n_gaussian,
        "ambiguous_start": ambiguous_start,
        "candidate_optimizer": candidate_optimizer,
        "search_seed": int(search_seed),
        "refit_seed": int(refit_seed),
        "integrity": {
            "failure_policy": refit.get("failure_policy"),
            "retry_count": int(refit.get("retry_count", 0)),
            "replacement_count": int(refit.get("replacement_count", 0)),
            "seed_rescue_count": int(refit.get("seed_rescue_count", 0)),
            "exploration_retry_count": 0,
            "exploration_replacement_count": 0,
            "exploration_seed_rescue_count": 0,
        },
        "convergence_gate": pilot_convergence_gate(
            exploration.selection_trace, exploration.final_candidate_rows),
    }


__all__ = [
    "SELECTOR_VERSION",
    "CANDIDATE_PRIORITY",
    "SelectorStop",
    "ColumnGate",
    "CandidateRecord",
    "ExplorationResult",
    "FamilySelectingPerColumnLSM",
    "support_gate",
    "initial_assignment",
    "column_log_likelihood",
    "optimise_column_loading",
    "optimise_column_loading_bfgs",
    "optimise_candidate_loading",
    "score_column_candidates",
    "OPTIMIZER_ADAM",
    "OPTIMIZER_BFGS",
    "VALID_OPTIMIZERS",
    "PHASE9C_CANDIDATE_OPTIMIZER",
    "BFGS_METHOD",
    "BFGS_MAXITER",
    "BFGS_GTOL",
    "CANDIDATE_GRAD_INF_TOL",
    "select_from_records",
    "run_family_exploration",
    "run_hybrid_family_selection",
    "pilot_convergence_gate",
    "PILOT_GATE_PASS",
    "_candidate_provenance_fields",
    "LOADING_INSTALLATION_POLICY",
    "loading_digest",
    "PILOT_GATE_BLOCKED",
    "EMFailFast",
]
