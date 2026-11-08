"""Selector-independent difficulty calibration (frozen BEFORE any selector run).

For a true candidate model m with natural parameter eta and a competitor m'
of the same support group, the pointwise projection divergence is

    KL_proj(m, eta -> m') = min_{eta'} KL( p_m(. | eta) || p_m'(. | eta') ).

The competitor may pick a different eta' for every observation, which it
cannot do in the model (its eta' must be linear in z), so this is a LOWER
bound on the per-observation expected log-likelihood ratio the selector can
exploit at the population optimum.  It is infinite when p_m puts mass outside
the competitor's support (e.g. Poisson vs zero-truncated Poisson): such pairs
are separable by the support gate, not by the score.

With the information-matched generator, eta for a column of true model m is
N(0, tau / A''_m(0)) on average over columns.  The group difficulty is

    D_g(tau) = mean_{m in g} min_{m' in g, m' != m} E_eta[ KL_proj(m, eta -> m') ]

and the frozen rule is EASY = argmax_tau D_g(tau), HARD = argmin_tau D_g(tau)
over a pre-declared grid.  Nothing here reads a selector result.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import brentq

from family_specs import (
    GROUP_BOUNDED, GROUP_COUNT, GROUP_POSITIVE, FamilySpec, _cmp_series_block,
)

CALIBRATION_VERSION = "difficulty-calibration-v1"
GH_NODES = 10   # probabilists' Hermite; outermost node 4.86 sd
TAU_GRID = (0.25, 0.5, 1.0, 2.0)


def _discrete_pmf(spec: FamilySpec, eta: float):
    if spec.candidate_model.startswith("cmp_"):
        out = _cmp_series_block(np.array([eta], dtype=float), spec.nu)
        return out["x"], out["probs"][0]
    mean = float(spec.mean_statistic(np.array([eta]))[0])
    var = float(spec.variance_statistic(np.array([eta]))[0])
    M = int(max(60, math.ceil(mean + 60.0 * math.sqrt(var) + 20)))
    if spec.candidate_model.startswith("binomial_"):
        M = spec.n_trials
    x = np.arange(0, M + 1, dtype=float)
    x = x[spec.support_mask(x)]
    p = np.exp(spec.log_prob(x, np.full_like(x, eta)))
    if abs(p.sum() - 1.0) > 1e-9:
        raise FloatingPointError(f"{spec.candidate_model} pmf mass {p.sum()} at eta={eta}")
    return x, p


def _continuous_grid(spec: FamilySpec, eta: float):
    if spec.support_group == GROUP_POSITIVE:
        u = np.linspace(-40.0, 25.0, 65001)
        x = np.exp(u)
        jac = x
    else:
        u = np.linspace(-40.0, 40.0, 80001)
        x = 1.0 / (1.0 + np.exp(-u))
        jac = x * (1.0 - x)
        keep = (x > 0.0) & (x < 1.0)
        u, x, jac = u[keep], x[keep], jac[keep]
    du = u[1] - u[0]
    dens = np.exp(spec.log_prob(x, np.full_like(x, eta))) * jac
    w = dens * du
    mass = float(w.sum())
    if abs(mass - 1.0) > 1e-6:
        raise FloatingPointError(f"{spec.candidate_model} density mass {mass} at eta={eta}")
    return x, w / mass, mass


def _solve_moment(spec: FamilySpec, target: float) -> float:
    """eta' with A'_{m'}(eta') = target (monotone; bracket expanded safely)."""
    def g(e):
        return float(spec.mean_statistic(np.array([e]))[0]) - target
    lo, hi = -1.0, 1.0
    for _ in range(60):
        try:
            glo, ghi = g(lo), g(hi)
        except FloatingPointError:
            raise
        if glo < 0 < ghi:
            return brentq(g, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=500)
        if glo >= 0:
            lo -= (hi - lo)
        if ghi <= 0:
            try:
                g(hi + (hi - lo))
                hi += (hi - lo)
            except FloatingPointError:
                hi = hi + 0.5 * (hi - lo) / 4.0
    raise FloatingPointError(f"cannot bracket the moment equation for "
                             f"{spec.candidate_model}, target={target}")


def kl_projection(true: FamilySpec, eta: float, comp: FamilySpec) -> dict[str, float]:
    if true.support_group == GROUP_COUNT:
        x, p = _discrete_pmf(true, eta)
        keep = p > 0
        x, p = x[keep], p[keep]
        if not np.all(comp.support_mask(x)):
            return {"kl": math.inf, "eta_prime": math.nan}
        logp_true = true.log_prob(x, np.full_like(x, eta))
    else:
        x, p, _ = _continuous_grid(true, eta)
        keep = p > 0
        x, p = x[keep], p[keep]
        logp_true = true.log_prob(x, np.full_like(x, eta))
    T = comp.sufficient_statistic(x)
    target = float(np.sum(p * T))
    eta_p = _solve_moment(comp, target)
    logp_comp = comp.log_prob(x, np.full_like(x, eta_p))
    kl = float(np.sum(p * (logp_true - logp_comp)))
    return {"kl": max(kl, 0.0), "eta_prime": float(eta_p), "kl_raw": kl}


def calibrate(registry: dict[str, FamilySpec], priority: dict[str, Sequence[str]],
              groups: Sequence[str] = (GROUP_COUNT, GROUP_POSITIVE, GROUP_BOUNDED),
              tau_grid: Sequence[float] = TAU_GRID) -> dict[str, Any]:
    nodes, weights = np.polynomial.hermite_e.hermegauss(GH_NODES)
    weights = weights / weights.sum()
    pair_rows, true_rows, group_rows, moment_rows = [], [], [], []
    for group in groups:
        cands = list(priority[group])
        for cm in cands:
            spec = registry[cm]
            mean0 = float(spec.mean_statistic(np.zeros(1))[0])
            var0 = float(spec.variance_statistic(np.zeros(1))[0])
            if group == GROUP_COUNT:
                xg, pg = _discrete_pmf(spec, 0.0)
            else:
                xg, pg, _ = _continuous_grid(spec, 0.0)
            xm = float(np.sum(pg * xg))
            xv = float(np.sum(pg * (xg - xm) ** 2))
            moment_rows.append({"support_group": group, "candidate_model": cm,
                                "family_name": spec.family_name,
                                "A_prime_at_0": mean0, "A_double_prime_at_0": var0,
                                "x_mean_at_eta0": xm, "x_var_at_eta0": xv})
        for tau in tau_grid:
            per_true = []
            for cm in cands:
                spec = registry[cm]
                sd = math.sqrt(tau / spec.info_at_zero())
                comp_vals = {}
                for cm2 in cands:
                    if cm2 == cm:
                        continue
                    comp = registry[cm2]
                    vals = []
                    for z, wt in zip(nodes, weights):
                        vals.append(kl_projection(spec, float(sd * z), comp)["kl"])
                    vals = np.array(vals)
                    ekl = (math.inf if np.any(np.isinf(vals))
                           else float(np.sum(weights * vals)))
                    comp_vals[cm2] = ekl
                    pair_rows.append({"support_group": group, "tau": tau,
                                      "true_candidate_model": cm,
                                      "competitor_candidate_model": cm2,
                                      "eta_sd": sd,
                                      "expected_kl_projection": ekl,
                                      "separable_by_support": bool(math.isinf(ekl))})
                finite = {c: v for c, v in comp_vals.items() if math.isfinite(v)}
                closest = min(finite, key=finite.get) if finite else ""
                min_kl = finite[closest] if finite else math.inf
                per_true.append(min_kl)
                true_rows.append({"support_group": group, "tau": tau,
                                  "true_candidate_model": cm, "eta_sd": sd,
                                  "closest_competitor": closest,
                                  "min_expected_kl_projection": min_kl,
                                  "loglr_lower_bound_n75": 75 * min_kl,
                                  "loglr_lower_bound_n150": 150 * min_kl})
            group_rows.append({"support_group": group, "tau": tau,
                               "D_mean_min_kl": float(np.mean(per_true))})
    selection = {}
    for group in groups:
        rows = [r for r in group_rows if r["support_group"] == group]
        easy = max(rows, key=lambda r: (r["D_mean_min_kl"], r["tau"]))["tau"]
        hard = min(rows, key=lambda r: (r["D_mean_min_kl"], r["tau"]))["tau"]
        selection[group] = {"EASY": easy, "HARD": hard}
    return {"version": CALIBRATION_VERSION, "gh_nodes": GH_NODES,
            "tau_grid": list(tau_grid), "pair_rows": pair_rows,
            "true_rows": true_rows, "group_rows": group_rows,
            "moment_rows": moment_rows, "selection": selection,
            "rule": "EASY = argmax_tau D_g(tau); HARD = argmin_tau D_g(tau); "
                    "D_g = mean over true candidates of the smallest finite "
                    "expected pointwise-projection KL to a same-group competitor"}


__all__ = ["CALIBRATION_VERSION", "TAU_GRID", "kl_projection", "calibrate"]
