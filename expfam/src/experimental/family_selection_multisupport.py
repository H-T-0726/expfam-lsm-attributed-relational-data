"""Multi-support per-column family selection (forward-only, lineage E).

Generalises the Phase 9C machinery (``family_selection.py``, unchanged) from
the fixed {bernoulli, poisson} ambiguity to candidate sets on several base
measures, using the ``FamilySpec`` interface.

Level 0 -- support gate (no likelihood)
    non-negative integers                     -> GROUP_COUNT
    non-integer, all in (0, 1)                -> GROUP_BOUNDED_CONTINUOUS
    non-integer, all > 0, some >= 1           -> GROUP_POSITIVE_CONTINUOUS
    anything else                             -> GROUP_REAL
Level 1 -- within the group, every candidate whose support contains the
    column is ADMISSIBLE; the others have likelihood exactly 0 and are
    recorded as inadmissible (not scored, not silently dropped).  With one
    admissible candidate the column is ``decided_by = gate``; with two or more
    it is ``decided_by = score``.

Scores are the strict COMPLETE per-column log-likelihood averaged over the
shared posterior samples (or the shared true-Z surrogate), each candidate with
its own BFGS-optimised loading.  Candidates within a group have equal free
parameter counts (all nuisances fixed), so no penalty is involved (Phase 9B
HG-3 (iii)); the margin is recorded instead.

For the existing families (gaussian / bernoulli / poisson) the candidate is
optimised and scored by the EXISTING Phase 9C functions, unchanged, so a
Phase 9C configuration run through this module reproduces Phase 9C exactly
(``test_generic_regression.py``).

Failure policy: a candidate whose optimisation or score raises is recorded as
FAILED with its reason.  A column with any failed admissible candidate has NO
selection (status FAILURE) -- the remaining candidates are never promoted to
winner by default.  Inside MCEM exploration such a failure stops the fit
(``SelectorStop``).  Nothing is retried, reseeded or clipped.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

from family_selection import (
    BFGS_GTOL, BFGS_MAXITER, BFGS_METHOD, CANDIDATE_GRAD_INF_TOL,
    SelectorStop, column_log_likelihood, loading_digest,
    optimise_column_loading_bfgs,
)
from family_specs import (
    GROUP_BOUNDED, GROUP_COUNT, GROUP_POSITIVE, GROUP_REAL, FamilySpec,
    GaussianSpec, NotEvaluable, SupportViolation,
)
from model_dual_expfam_generic import (
    DualExpFamLSMGeneric, informed_init, run_em_generic,
)
from em_runner import EMFailFast

SELECTOR_VERSION_MS = "multisupport-family-selection-v1"
LEGACY_FAMILIES = ("gaussian", "bernoulli", "poisson")

# Candidate optimisation routes.
#   "phase9c"     existing families through the unchanged Phase 9C BFGS
#                 functions; other families through analytic-gradient BFGS.
#                 Used to certify that Phase 9C is reproduced.
#   "trust_exact" every non-Gaussian candidate through a Newton trust region
#                 with the exact analytic Hessian (scipy 'trust-exact').  The
#                 canonical candidate objective is concave in the loading, so
#                 the optimum is the same; the trust region keeps trial points
#                 near accepted iterates.  A trial point where the COM-Poisson
#                 series cannot be certified (NotEvaluable) is a rejected step
#                 (objective +inf); accepted iterates and every reported score
#                 are always certified.  Used by the multi-support experiment.
ROUTE_PHASE9C = "phase9c"
ROUTE_TRUST_EXACT = "trust_exact"
TRUST_EXACT_MAXITER = 200
TRUST_EXACT_GTOL = 1e-10


# --------------------------------------------------------------------------
# Level 0 / Level 1 gate
# --------------------------------------------------------------------------

def support_group_of(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise SelectorStop("column contains a non-finite value")
    if np.all(x >= 0.0) and np.all(x == np.floor(x)):
        return GROUP_COUNT
    if np.all((x > 0.0) & (x < 1.0)):
        return GROUP_BOUNDED
    if np.all(x > 0.0):
        return GROUP_POSITIVE
    return GROUP_REAL


@dataclass(frozen=True)
class ColumnGateMS:
    column: int
    support_group: str
    group_candidates: tuple[str, ...]
    admissible: tuple[str, ...]
    inadmissible: tuple[str, ...]

    @property
    def decided_by(self) -> str:
        if len(self.admissible) == 0:
            return "none_admissible"
        return "score" if len(self.admissible) >= 2 else "gate"

    @property
    def is_ambiguous(self) -> bool:
        return self.decided_by == "score"

    def as_row(self) -> dict[str, Any]:
        return {"column": self.column, "support_group": self.support_group,
                "group_candidates": "|".join(self.group_candidates),
                "admissible": "|".join(self.admissible),
                "inadmissible": "|".join(self.inadmissible),
                "n_admissible": len(self.admissible),
                "decided_by": self.decided_by}


def gate_columns(X: np.ndarray, registry: dict[str, FamilySpec],
                 priority: dict[str, Sequence[str]]) -> list[ColumnGateMS]:
    gates = []
    for col in range(X.shape[1]):
        x = X[:, col]
        group = support_group_of(x)
        cands = tuple(priority[group])
        adm = tuple(c for c in cands if registry[c].in_support(x))
        inadm = tuple(c for c in cands if c not in adm)
        gates.append(ColumnGateMS(col, group, cands, adm, inadm))
    return gates


def start_assignment(gates: Sequence[ColumnGateMS], start: str) -> list[str]:
    """START_FIRST / START_LAST, or a named candidate for ambiguous columns."""
    out = []
    for g in gates:
        if not g.admissible:
            raise SelectorStop(f"column {g.column}: no admissible candidate in "
                               f"{g.support_group}")
        if start == "START_FIRST":
            out.append(g.admissible[0])
        elif start == "START_LAST":
            out.append(g.admissible[-1])
        else:
            out.append(start if (g.is_ambiguous and start in g.admissible)
                       else g.admissible[0])
    return out


# --------------------------------------------------------------------------
# candidate optimisation and score
# --------------------------------------------------------------------------

@dataclass
class CandidateRecordMS:
    column: int
    family_name: str
    candidate_model: str
    status: str                       # "OK" | "FAILED"
    score: float = float("nan")
    loading: np.ndarray | None = None
    sigma_sq: float | None = None
    n_iter: int = 0
    converged: bool = False
    gradient_inf: float = float("nan")
    scipy_success: Any = ""
    scipy_status: Any = ""
    optimizer_route: str = ""
    failure_reason: str = ""
    n_obs: int = 0

    def as_row(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "candidate_family": self.family_name,
            "candidate_model": self.candidate_model,
            "candidate_status": self.status,
            "candidate_score": self.score,
            "normalized_score_per_obs": (self.score / self.n_obs
                                         if self.n_obs and self.status == "OK"
                                         else float("nan")),
            "optimizer_success": self.scipy_success,
            "optimizer_status": self.scipy_status,
            "optimizer_iterations": self.n_iter,
            "optimizer_route": self.optimizer_route,
            "gradient_inf_norm": self.gradient_inf,
            "gradient_converged_1e-8": self.converged,
            "loading_norm": (float(np.linalg.norm(self.loading))
                             if self.loading is not None else float("nan")),
            "sigma_sq": "" if self.sigma_sq is None else self.sigma_sq,
            "failure_reason": self.failure_reason,
        }


def _generic_mean_score_and_grad(spec: FamilySpec, x: np.ndarray, Tx: np.ndarray,
                                 Z_samples: np.ndarray, loading: np.ndarray):
    eta = np.einsum("nkl,k->nl", Z_samples, loading)
    L = eta.shape[1]
    lp = spec.log_prob(x[:, None], eta)
    score = float(np.sum(lp)) / L
    resid = Tx[:, None] - spec.mean_statistic(eta)
    grad = np.einsum("nl,nkl->k", resid, Z_samples) / L
    if not (math.isfinite(score) and np.all(np.isfinite(grad))):
        raise FloatingPointError(f"{spec.candidate_model}: non-finite score/gradient")
    return score, grad


def _trust_exact(spec: FamilySpec, x: np.ndarray, Z_samples: np.ndarray,
                 start: np.ndarray):
    Tx = spec.sufficient_statistic(x)
    cache: dict = {"key": None}
    counter = {"nonevaluable": 0}

    def evaluate(f):
        key = np.asarray(f, dtype=np.float64).tobytes()
        if cache["key"] == key:
            return cache["val"]
        f = np.asarray(f, dtype=np.float64)
        try:
            eta = np.einsum("nkl,k->nl", Z_samples, f)
            L = eta.shape[1]
            score = float(np.sum(spec.log_prob(x[:, None], eta))) / L
            mean = spec.mean_statistic(eta)
            var = spec.variance_statistic(eta)
            grad = np.einsum("nl,nkl->k", Tx[:, None] - mean, Z_samples) / L
            hess = -np.einsum("nl,nkl,njl->kj", var, Z_samples, Z_samples) / L
            if not (math.isfinite(score) and np.all(np.isfinite(grad))
                    and np.all(np.isfinite(hess))):
                raise FloatingPointError("non-finite candidate objective")
            val = (score, grad, hess)
        except NotEvaluable:
            counter["nonevaluable"] += 1
            val = None
        cache["key"], cache["val"] = key, val
        return val

    def fun(f):
        v = evaluate(f)
        return math.inf if v is None else -v[0]

    # scipy builds the local model of a TRIAL point before deciding whether
    # to accept it.  At a non-evaluable trial point the objective is +inf, so
    # the actual reduction is -inf and the step is always rejected; the
    # derivative placeholders below can never enter an accepted iterate (the
    # returned point is re-evaluated and must be certified).
    k = Z_samples.shape[1]

    def jac(f):
        v = evaluate(f)
        return np.zeros(k) if v is None else -v[1]

    def hess(f):
        v = evaluate(f)
        return np.eye(k) if v is None else -v[2]

    if evaluate(start) is None:
        raise NotEvaluable("the starting loading is not evaluable")
    res = minimize(fun, start, method="trust-exact", jac=jac, hess=hess,
                   options={"maxiter": TRUST_EXACT_MAXITER,
                            "gtol": TRUST_EXACT_GTOL})
    loading = np.asarray(res.x, dtype=np.float64)
    final = evaluate(loading)
    if final is None:
        raise NotEvaluable("the returned loading is not evaluable")
    return loading, final, res, counter["nonevaluable"]


def score_candidate(x: np.ndarray, Z_samples: np.ndarray, spec: FamilySpec,
                    loading_init: np.ndarray, column: int,
                    route: str = ROUTE_PHASE9C) -> CandidateRecordMS:
    x = np.asarray(x, dtype=np.float64)
    rec = CandidateRecordMS(column=column, family_name=spec.family_name,
                            candidate_model=spec.candidate_model, status="OK",
                            n_obs=int(x.size))
    if route not in (ROUTE_PHASE9C, ROUTE_TRUST_EXACT):
        raise ValueError(f"unknown candidate route {route!r}")
    try:
        if route == ROUTE_TRUST_EXACT and not spec.profiled_variance:
            loading, (score, grad, _), res, n_bad = _trust_exact(
                spec, x, Z_samples, np.array(loading_init, dtype=np.float64))
            ginf = float(np.max(np.abs(grad)))
            rec.optimizer_route = f"trust_exact(nonevaluable_trials={n_bad})"
            rec.loading, rec.n_iter = loading, int(res.nit)
            rec.converged = bool(math.isfinite(ginf) and ginf <= CANDIDATE_GRAD_INF_TOL)
            rec.gradient_inf = ginf
            rec.scipy_success, rec.scipy_status = bool(res.success), int(res.status)
            rec.score = float(score)
        elif spec.candidate_model in LEGACY_FAMILIES:
            # Existing Phase 9C route, unchanged.
            loading, sigma_sq, nit, conv, prov = optimise_column_loading_bfgs(
                x, Z_samples, spec.candidate_model, loading_init=loading_init)
            score = column_log_likelihood(x, Z_samples, loading,
                                          spec.candidate_model, sigma_sq=sigma_sq)
            rec.optimizer_route = "phase9c_existing_bfgs"
            rec.loading, rec.sigma_sq, rec.n_iter = loading, sigma_sq, int(nit)
            rec.converged, rec.gradient_inf = bool(conv), float(prov["gradient_inf"])
            rec.scipy_success, rec.scipy_status = prov["scipy_success"], prov["scipy_status"]
            rec.score = float(score)
        else:
            Tx = spec.sufficient_statistic(x)

            def fun(f):
                s, g = _generic_mean_score_and_grad(spec, x, Tx, Z_samples,
                                                    np.asarray(f, dtype=np.float64))
                return -s, -g

            start = np.array(loading_init, dtype=np.float64).copy()
            res = minimize(fun, start, method=BFGS_METHOD, jac=True,
                           options={"maxiter": BFGS_MAXITER, "gtol": BFGS_GTOL})
            loading = np.asarray(res.x, dtype=np.float64)
            score, grad = _generic_mean_score_and_grad(spec, x, Tx, Z_samples, loading)
            ginf = float(np.max(np.abs(grad)))
            rec.optimizer_route = "generic_analytic_bfgs"
            rec.loading, rec.n_iter = loading, int(res.nit)
            rec.converged = bool(math.isfinite(ginf) and ginf <= CANDIDATE_GRAD_INF_TOL)
            rec.gradient_inf = ginf
            rec.scipy_success, rec.scipy_status = bool(res.success), int(res.status)
            rec.score = float(score)
    except (FloatingPointError, SelectorStop, SupportViolation, ValueError,
            OverflowError) as exc:
        rec.status = "FAILED"
        rec.failure_reason = f"{type(exc).__name__}: {exc}"
        rec.score = float("nan")
    return rec


@dataclass
class ColumnSelection:
    column: int
    status: str                       # "SELECTED" | "FAILURE"
    selected: str = ""
    runner_up: str = ""
    winner_score: float = float("nan")
    runnerup_score: float = float("nan")
    margin_neg2: float = float("nan")
    failure_reason: str = ""
    records: list[CandidateRecordMS] = field(default_factory=list)


def select_column(records: Sequence[CandidateRecordMS],
                  priority: Sequence[str]) -> ColumnSelection:
    col = records[0].column
    failed = [r for r in records if r.status != "OK"]
    if failed:
        return ColumnSelection(col, "FAILURE", failure_reason="; ".join(
            f"{r.candidate_model}: {r.failure_reason}" for r in failed),
            records=list(records))
    order = list(priority)
    best = max(records, key=lambda r: (r.score, -order.index(r.candidate_model)))
    if len(records) == 1:
        return ColumnSelection(col, "SELECTED", selected=best.candidate_model,
                               winner_score=best.score, margin_neg2=0.0,
                               records=list(records))
    others = [r for r in records if r is not best]
    runner = max(others, key=lambda r: (r.score, -order.index(r.candidate_model)))
    margin = (-2.0 * runner.score) - (-2.0 * best.score)
    return ColumnSelection(col, "SELECTED", selected=best.candidate_model,
                           runner_up=runner.candidate_model,
                           winner_score=best.score, runnerup_score=runner.score,
                           margin_neg2=float(margin), records=list(records))


def score_columns(X: np.ndarray, Z_samples: np.ndarray,
                  gates: Sequence[ColumnGateMS], registry: dict[str, FamilySpec],
                  loading_inits: np.ndarray, columns: Sequence[int] | None = None,
                  route: str = ROUTE_PHASE9C) -> list[ColumnSelection]:
    out = []
    cols = [g.column for g in gates] if columns is None else list(columns)
    for g in gates:
        if g.column not in cols:
            continue
        recs = [score_candidate(X[:, g.column], Z_samples, registry[c],
                                loading_inits[g.column], g.column, route=route)
                for c in g.admissible]
        out.append(select_column(recs, g.group_candidates))
    return out


# --------------------------------------------------------------------------
# Experiment A: selector-only on the true Z
# --------------------------------------------------------------------------

def selector_only(X: np.ndarray, Z_true: np.ndarray, registry, priority,
                  route: str = ROUTE_PHASE9C
                  ) -> tuple[list[ColumnGateMS], list[ColumnSelection]]:
    """Every candidate scored on the SAME deterministic true-Z surrogate.

    ``Z_samples = Z_true[:, :, None]`` (L = 1): the selector sees the true
    latent positions, so a selection error here is the selector's own and not
    an MCEM error.  Each candidate starts from the zero loading; the canonical
    candidate objective is concave in the loading, so the start does not
    choose the optimum.
    """
    gates = gate_columns(X, registry, priority)
    Zs = np.asarray(Z_true, dtype=np.float64)[:, :, None]
    inits = np.zeros((X.shape[1], Zs.shape[1]))
    return gates, score_columns(X, Zs, gates, registry, inits, route=route)


# --------------------------------------------------------------------------
# Experiments B/C: A-type exploration inside MCEM, then fixed refit
# --------------------------------------------------------------------------

class FamilySelectingGenericLSM(DualExpFamLSMGeneric):
    """Generic analogue of ``family_selection.FamilySelectingPerColumnLSM``."""

    def __init__(self, *, gates, registry, route=ROUTE_PHASE9C, **kwargs):
        super().__init__(**kwargs)
        self.gates = list(gates)
        self.registry = registry
        self.route = route
        self.selection_trace: list[dict[str, Any]] = []
        self.last_selections: list[ColumnSelection] = []
        self._iteration = 0

    def calc_F(self, X, Z_samples):
        self._iteration += 1
        loadings = self.params["F"]
        amb = [g for g in self.gates if g.is_ambiguous]
        selections = score_columns(X, Z_samples, amb, self.registry, loadings,
                                   route=self.route)
        specs = list(self.column_specs)
        winners = {}
        for g, sel in zip(amb, selections):
            row = {"iteration": self._iteration, "column": g.column,
                   "support_group": g.support_group,
                   "admissible": "|".join(g.admissible),
                   "previous_candidate": specs[g.column].candidate_model,
                   "status": sel.status}
            if sel.status != "SELECTED":
                row["failure_reason"] = sel.failure_reason
                self.selection_trace.append(row)
                self.last_selections = selections
                raise SelectorStop(f"exploration iteration {self._iteration}, "
                                   f"column {g.column}: {sel.failure_reason}")
            winner = [r for r in sel.records if r.candidate_model == sel.selected][0]
            winners[g.column] = winner
            specs[g.column] = self.registry[sel.selected]
            row.update({
                "selected_candidate": sel.selected,
                "selected_family": self.registry[sel.selected].family_name,
                "runner_up": sel.runner_up,
                "changed": sel.selected != row["previous_candidate"],
                "margin_neg2": sel.margin_neg2,
                "winner_score": sel.winner_score,
                "runnerup_score": sel.runnerup_score,
                "all_candidates_converged": all(r.converged for r in sel.records),
                "max_candidate_grad_inf": max(r.gradient_inf for r in sel.records),
                "candidate_scores_json": json.dumps(
                    {r.candidate_model: r.score for r in sel.records}),
                "candidate_grad_inf_json": json.dumps(
                    {r.candidate_model: r.gradient_inf for r in sel.records}),
                "selected_loading_sha256": loading_digest(winner.loading),
            })
            self.selection_trace.append(row)
        self.last_selections = selections
        self.reassign_specs(specs)
        F = np.array(super().calc_F(X, Z_samples), dtype=np.float64, copy=True)
        for col, winner in winners.items():
            F[col, :] = winner.loading
        for row in self.selection_trace[-len(amb):] if amb else []:
            row["selected_loading_installed"] = True
            row["installed_loading_sha256"] = loading_digest(F[row["column"], :])
        return F


def run_generic_exploration(X, Y, *, k, gates, registry, initial_models,
                            family_y="bernoulli", L=5, num_iter=8, seed=42,
                            newton_alpha=0.5, route=ROUTE_PHASE9C) -> dict[str, Any]:
    """Exploration MCEM (A-type updates); mirrors run_family_exploration."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    model = FamilySelectingGenericLSM(
        gates=gates, registry=registry, route=route, n=n, d=d, k=k, L=L,
        column_specs=[registry[c] for c in initial_models], family_y=family_y)
    model.initialize_params(true_params=None, seed=seed)
    informed_init(model, Y, family_y)

    Z = model.params["Z"].copy()
    F = model.params["F"].copy()
    sigma = model.params["sigma"].copy()
    w0 = float(model.params["w0"])
    w = float(model.params["w"])
    for iteration in range(1, num_iter + 1):
        Z_samples = np.zeros((n, k, L))
        for sample in range(L):
            model.params.update(dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w))
            Z_new = model.calc_eta_newton(X, Y, rng=rng, max_iter=10,
                                          alpha=newton_alpha)
            Z_samples[:, :, sample] = Z_new
            Z = Z_new.copy()
        if not np.all(np.isfinite(Z_samples)):
            raise SelectorStop(f"exploration E-step non-finite at iteration "
                               f"{iteration}; fail-fast")
        Z_samples = model.scale_Z(Z_samples)
        Z = Z_samples[:, :, -1].copy()
        F = model.calc_F(X, Z_samples)
        sigma = model.calc_sigma(X, Z_samples, F)
        w0 = float(model.calc_w0(Y, Z_samples, w0, w, max_iter=50))
        w = float(model.calc_w(Y, Z_samples, w0, w, max_iter=50))
        if family_y == "gaussian":
            model.calc_sigma_y(Y, Z_samples, w0, w)
    return {"selected_models": model.candidate_models(),
            "initial_models": list(initial_models),
            "selection_trace": list(model.selection_trace),
            "final_selections": list(model.last_selections),
            "estep_backtracks": int(getattr(model, "estep_backtracks", 0)),
            "iterations": num_iter}


def run_generic_hybrid(X, Y, *, k, registry, priority, start, family_y="bernoulli",
                       L=5, exploration_num_iter=8, refit_num_iter=8,
                       search_seed=42, refit_seed=43, newton_alpha=0.5,
                       route=ROUTE_PHASE9C) -> dict[str, Any]:
    """Scheme C: exploration with A-type updates, then a fresh fixed refit."""
    gates = gate_columns(X, registry, priority)
    init = start_assignment(gates, start)
    exploration = run_generic_exploration(
        X, Y, k=k, gates=gates, registry=registry, initial_models=init,
        family_y=family_y, L=L, num_iter=exploration_num_iter, seed=search_seed,
        newton_alpha=newton_alpha, route=route)
    refit = run_em_generic(X, Y, column_specs=[registry[c] for c in
                                               exploration["selected_models"]],
                           family_y=family_y, k=k, L=L, num_iter=refit_num_iter,
                           seed=refit_seed, newton_alpha=newton_alpha)
    return {"gates": gates, "start_models": init, "exploration": exploration,
            "refit": refit}


__all__ = ["SELECTOR_VERSION_MS", "ROUTE_PHASE9C", "ROUTE_TRUST_EXACT", "support_group_of", "ColumnGateMS",
           "gate_columns", "start_assignment", "CandidateRecordMS",
           "score_candidate", "ColumnSelection", "select_column",
           "score_columns", "selector_only", "FamilySelectingGenericLSM",
           "run_generic_exploration", "run_generic_hybrid", "EMFailFast"]
