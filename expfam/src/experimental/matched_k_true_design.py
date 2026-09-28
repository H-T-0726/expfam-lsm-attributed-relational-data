"""Phase 9W (Issue #120): matched-signal K_true design and calibration.

Zero EM. For K_true in {1, 2, 3, 4} (d = 12, K_ref = 3 anchor):

  X loading energy   f_scale(K) = f_scale_for_row_norm(0.5, d=12, k=K)
                     = sqrt(6/K), so average ||f_l||^2 = f^2 K / d = 0.5
  Y variance         w(K) = w_for_matched_y_signal(1, k=K, k_ref=3)
                     = sqrt(3/K), so Var(w S_K) = w^2 K = 3
  mean edge prob.    w0(K) solves p_bar_K(w0) = p_target = p_bar_3(-1),
                     p_bar_K(b) = E[sigmoid(b + w(K) S_K)], S_K = z_i^T z_j

with the exact density of S_K
  f_K(s) = (|s|/2)^nu K_nu(|s|) / (sqrt(pi) Gamma(K/2)),  nu = (K-1)/2.

Numerical rules, fixed before any run:
  PRIMARY    adaptive QUADPACK on [0, 1] + [1, inf) (split because the K=1
             density K_0(|s|)/pi has a log singularity at 0), settings A/B/C.
  CROSSCHECK exponential map s = e^t, fixed-step trapezoid on
             t in [T_LO, T_HI], steps H_STEPS (not QUADPACK).
  FREEZE     B vs C agree within 1e-12 on p_target and every w0(K); every
             root residual under C <= 1e-12; K3 recovers -1 within 1e-12;
             density normalization and E[S^2] = K within 1e-11. Cross-check:
             its finest two steps agree within 1e-12 and its finest values
             agree with the primary production values within 1e-12.
  K3 ANCHOR  w0(3) = -1 by the definition of the target; the calibrated
             value is only a recovery check.

The canonical generators, estimators and Phase 9K artifacts are untouched.
Lineage E.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq
from scipy.special import expit, gamma, kve

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import laplace_k_criterion as lk                                   # noqa: E402
import run_family_selection_pilot as pilot                         # noqa: E402
import run_lap_vs_cq_20 as pc                                      # noqa: E402
from data_generator_canonical import (                              # noqa: E402
    DEFAULT_POISSON_LAMBDA_MAX,
    GeneratorStop,
    f_scale_for_row_norm,
    w_for_matched_y_signal,
)

DESIGN_VERSION = "matched-k-true-design-v1"
K_TRUES = (1, 2, 3, 4)
CANDIDATE_K = (1, 2, 3, 4, 5)
D = 12
ROW_ENERGY = 0.5
K_REF, W_REF, W0_REF = 3, 1.0, -1.0
FUTURE_REPLICATES = 10
PRIMARY_SETTINGS = (("A", 1e-10, 1e-10), ("B", 1e-12, 1e-12),
                    ("C", 2e-13, 2e-13))
QUAD_LIMIT = 200
T_LO, T_HI = -45.0, 4.5
H_STEPS = (1 / 8, 1 / 16, 1 / 32, 1 / 64)
FREEZE_TOL = 1e-12
DENSITY_TOL = 1e-11
ROOT_XTOL, ROOT_RTOL = 1e-15, 4 * np.finfo(float).eps
SMALL_S = 1e-12
PHASE9K_DIR = (Path(__file__).resolve().parents[2]
               / "results/lap_vs_cq_20/phase9k_20260928")
QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)


# --------------------------------------------------------------------------
# matched parameters
# --------------------------------------------------------------------------

def f_scale(k: int) -> float:
    return f_scale_for_row_norm(ROW_ENERGY, d=D, k=k)


def w_of(k: int) -> float:
    return w_for_matched_y_signal(W_REF, k=k, k_ref=K_REF)


# --------------------------------------------------------------------------
# exact density of S_K
# --------------------------------------------------------------------------

def density(s: np.ndarray | float, k: int) -> np.ndarray:
    """f_K(s) for s >= 0 (symmetric); limit value for 0 < s < SMALL_S, nu > 0."""
    s = np.atleast_1d(np.asarray(s, float))
    nu = (k - 1) / 2
    const = 1.0 / (math.sqrt(math.pi) * gamma(k / 2))
    out = np.empty_like(s)
    small = s < SMALL_S
    big = ~small
    out[big] = const * (s[big] / 2) ** nu * kve(nu, s[big]) * np.exp(-s[big])
    if nu > 0:
        out[small] = const * gamma(nu) / 2
    else:                                   # K = 1: K_0 has a log singularity
        ss = np.maximum(s[small], np.finfo(float).tiny)
        out[small] = const * kve(0, ss) * np.exp(-ss)
    return out


def _pair(b: float, w: float, k: int) -> Callable[[float], float]:
    return lambda s: float((expit(b + w * s) + expit(b - w * s))
                           * density(s, k)[0])


class Adaptive:
    def __init__(self, epsabs: float, epsrel: float):
        self.epsabs, self.epsrel = epsabs, epsrel
        self.max_abserr, self.warnings = 0.0, 0

    def integrate(self, f: Callable[[float], float]) -> float:
        total = 0.0
        for lo, hi in ((0.0, 1.0), (1.0, math.inf)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", IntegrationWarning)
                value, err = quad(f, lo, hi, epsabs=self.epsabs,
                                  epsrel=self.epsrel, limit=QUAD_LIMIT)
            total += value
            self.max_abserr = max(self.max_abserr, err)
            self.warnings += sum(issubclass(c.category, IntegrationWarning)
                                 for c in caught)
        return total

    def p_bar(self, b: float, w: float, k: int) -> float:
        return self.integrate(_pair(b, w, k))


class ExpTrapezoid:
    """Fixed-step trapezoid after s = e^t (independent of QUADPACK)."""

    def __init__(self, h: float):
        t = np.arange(T_LO, T_HI + h / 2, h)
        self.s = np.exp(t)
        self.h = h
        self._dens = {}

    def weights(self, k: int) -> np.ndarray:
        if k not in self._dens:
            self._dens[k] = self.h * self.s * density(self.s, k)
        return self._dens[k]

    def p_bar(self, b: float, w: float, k: int) -> float:
        return float(np.dot(self.weights(k), expit(b + w * self.s)
                            + expit(b - w * self.s)))


def bracket(fn: Callable[[float], float], target: float):
    lo, hi = -1.0, 1.0
    while fn(lo) > target:
        lo *= 2.0
        if lo < -1e3:
            raise ValueError("no lower bracket")
    while fn(hi) < target:
        hi *= 2.0
        if hi > 1e3:
            raise ValueError("no upper bracket")
    return lo, hi


def calibrate(method) -> dict[str, Any]:
    target = method.p_bar(W0_REF, W_REF, K_REF)
    out: dict[str, Any] = {"p_target": target}
    for k in K_TRUES:
        w = w_of(k)
        fn = lambda b, w=w, k=k: method.p_bar(b, w, k)            # noqa: E731
        lo, hi = bracket(fn, target)
        w0 = brentq(lambda b: fn(b) - target, lo, hi, xtol=ROOT_XTOL,
                    rtol=ROOT_RTOL, maxiter=500)
        out[f"w0_K{k}"] = w0
        out[f"residual_K{k}"] = fn(w0) - target
    out["K3_recovery_error"] = out["w0_K3"] - W0_REF
    return out


KEYS = ("p_target",) + tuple(f"w0_K{k}" for k in K_TRUES)


def _diffs(a, b):
    return {k: abs(a[k] - b[k]) for k in KEYS}


def freeze_rule(rows, density_ok: bool) -> dict[str, Any]:
    valid = [r for r in rows if r.get("valid")]
    if len(valid) < 2:
        return {"pass": False, "reason": "fewer than two valid settings",
                "production": None}
    a, b = valid[-2], valid[-1]
    diffs = _diffs(a, b)
    residuals = {k: abs(b[f"residual_K{k}"]) for k in K_TRUES}
    ok = (all(v <= FREEZE_TOL for v in diffs.values())
          and all(v <= FREEZE_TOL for v in residuals.values())
          and abs(b["K3_recovery_error"]) <= FREEZE_TOL and density_ok)
    return {"pass": ok, "compared": [a["setting"], b["setting"]],
            "differences": diffs, "stricter_residuals": residuals,
            "density_ok": density_ok, "production": b if ok else None}


def crosscheck_rule(rows, production) -> dict[str, Any]:
    a, b = rows[-2], rows[-1]
    internal = _diffs(a, b)
    vs = _diffs(b, production) if production else None
    ok = (production is not None
          and all(v <= FREEZE_TOL for v in internal.values())
          and all(v <= FREEZE_TOL for v in vs.values()))
    return {"pass": ok, "internal_finest_two": internal,
            "finest_vs_primary": vs}


def density_checks() -> dict[str, Any]:
    method = Adaptive(1e-14, 1e-13)
    out = {}
    for k in K_TRUES:
        norm = 2 * method.integrate(lambda s: float(density(s, k)[0]))
        m2 = 2 * method.integrate(lambda s: s * s * float(density(s, k)[0]))
        out[f"K{k}"] = {"normalization": norm,
                        "normalization_error": abs(norm - 1),
                        "second_moment": m2,
                        "second_moment_error": abs(m2 - k)}
    out["pass"] = all(out[f"K{k}"]["normalization_error"] <= DENSITY_TOL
                      and out[f"K{k}"]["second_moment_error"] <= DENSITY_TOL
                      for k in K_TRUES)
    out["tolerance"] = DENSITY_TOL
    return out


# --------------------------------------------------------------------------
# protocols, generation, context
# --------------------------------------------------------------------------

def protocol_for(k: int, w0: float, n_replicates: int = FUTURE_REPLICATES):
    return dataclasses.replace(
        pc.PROTOCOL, stage=f"matched_k_true_K{k}", k_true=k,
        f_scale=f_scale(k), w=w_of(k), w0=w0,
        replicates=pc.PROTOCOL.replicates[:n_replicates])


def context_row(k: int, rep: str, ds, w0: float, w: float) -> dict[str, Any]:
    fam = np.array(pc.PROTOCOL.family_x_list)
    g, b, p = (np.flatnonzero(fam == name)
               for name in ("gaussian", "bernoulli", "poisson"))
    eta_x = ds.Z @ ds.F.T
    rn = np.sum(ds.F ** 2, axis=1)
    prob_x = expit(eta_x[:, b]).ravel()
    rate = np.exp(eta_x[:, p]).ravel()
    upper = np.triu_indices(ds.Z.shape[0], k=1)
    eta_y = w0 + w * (ds.Z @ ds.Z.T)[upper]
    prob_y = expit(eta_y)
    row = {"K_true": k, "replicate": rep, "f_scale": f_scale(k), "w": w,
           "w0": w0, "F_rank": int(np.linalg.matrix_rank(ds.F)),
           "row_norm_sq_min": float(rn.min()),
           "row_norm_sq_median": float(np.median(rn)),
           "row_norm_sq_mean": float(rn.mean()),
           "row_norm_sq_max": float(rn.max()),
           "eta_x_sd_overall": float(eta_x.std()),
           "eta_x_sd_gaussian": float(eta_x[:, g].std()),
           "eta_x_sd_bernoulli": float(eta_x[:, b].std()),
           "eta_x_sd_poisson": float(eta_x[:, p].std()),
           "gaussian_snr_mean": float(np.var(eta_x[:, g], axis=0).mean()
                                      / pc.PROTOCOL.sigma_x_var),
           "bernoulli_prob_mean": float(prob_x.mean()),
           "bernoulli_prob_sd": float(prob_x.std()),
           **{f"bernoulli_prob_q{int(round(q*100)):02d}":
              float(np.quantile(prob_x, q)) for q in QUANTILES},
           "bernoulli_share_p_lt_0.05": float(np.mean(prob_x < 0.05)),
           "bernoulli_share_p_gt_0.95": float(np.mean(prob_x > 0.95)),
           "poisson_rate_mean": float(rate.mean()),
           "poisson_rate_median": float(np.median(rate)),
           "poisson_rate_q95": float(np.quantile(rate, 0.95)),
           "poisson_rate_q99": float(np.quantile(rate, 0.99)),
           "poisson_rate_max": float(rate.max()),
           "poisson_safety_margin": float(DEFAULT_POISSON_LAMBDA_MAX
                                          / rate.max()),
           "eta_y_mean": float(eta_y.mean()), "eta_y_sd": float(eta_y.std()),
           "y_prob_mean": float(prob_y.mean()),
           **{f"y_prob_q{int(round(q*100)):02d}":
              float(np.quantile(prob_y, q)) for q in QUANTILES},
           "y_share_p_lt_0.05": float(np.mean(prob_y < 0.05)),
           "y_share_p_gt_0.95": float(np.mean(prob_y > 0.95)),
           "realized_edge_density": float(ds.Y[upper].mean()),
           "finite": bool(all(np.all(np.isfinite(a))
                              for a in (ds.Z, ds.F, ds.X, ds.Y))),
           "supports_ok": bool(
               np.all((ds.X[:, b] == 0) | (ds.X[:, b] == 1))
               and np.all(ds.X[:, p] >= 0)
               and np.all(ds.X[:, p] == np.floor(ds.X[:, p]))
               and np.all((ds.Y == 0) | (ds.Y == 1)))}
    return row


def k3_anchor_compatibility(w0_k3: float) -> dict[str, Any]:
    proto = protocol_for(3, w0_k3, n_replicates=len(pc.PROTOCOL.replicates))
    base = pc.PROTOCOL.as_json()
    mine = proto.as_json()
    field_diff = sorted(k for k in base if base[k] != mine[k])
    prov = {(r["replicate"], int(r["column"])): r
            for r in pc._read(PHASE9K_DIR / "generator_provenance.csv")}
    rows = []
    for rep in pc.PROTOCOL.replicates:
        new = pilot.build_dataset(proto, rep)
        hist = pilot.build_dataset(pc.PROTOCOL, rep)
        committed_ok = all(
            float(prov[(rep.label, c)]["column_mean"])
            == float(np.mean(new.X[:, c])) for c in range(D))
        rows.append({"replicate": rep.label,
                     **{f"{m}_bit_identical": bool(np.array_equal(
                         getattr(new, m), getattr(hist, m)))
                        for m in ("Z", "F", "X", "Y")},
                     "matches_committed_generator_provenance": committed_ok})
    reusable = (field_diff == ["stage"]
                and all(all(v for k, v in r.items() if k != "replicate")
                        for r in rows))
    return {"protocol_fields_differing_from_phase9k": field_diff,
            "replicates": rows,
            "counts": {k: sum(r[k] for r in rows) for k in rows[0]
                       if k != "replicate"},
            "K3_anchor_reusable": reusable}


def k1_boundary_audit() -> dict[str, Any]:
    lap = pc._read(PHASE9K_DIR / "laplace_by_k.csv")
    k1 = [r for r in lap if r["k"] == "1"]
    return {
        "loading_parameter_count": {k: lk.loading_parameter_count(k, D)
                                    for k in CANDIDATE_K},
        "rotation_dimension_K1": 0,
        "notes": {
            "generator": "build_full_rank_loadings(k=1): reduced QR of a "
                         "12x1 Gaussian, rank 1 by construction; checked "
                         "on every K_true=1 dataset (F_rank in context)",
            "identifiability": "O(1) = {+1, -1}: only a discrete sign "
                               "ambiguity, no continuous rotation; the "
                               "loading count 1*12 - 0 = 12 is consistent",
            "candidate_b_at_model_K1": f"Phase 9K evaluated Candidate B at "
                                       f"model K=1 on {len(k1)} refits: "
                                       f"{sum(r['laplace_status'] == 'OK' for r in k1)} OK",
            "joint_runner": "k_category / k_hat_equals_k_true use "
                            "protocol.k_true (generic)",
            "summary_helpers": "run_lap_vs_cq_20.K_TRUE = 3, category() and "
                               "paired_summary's both_K3 / lap_only_K3 counts "
                               "are hard-coded to 3; a future Phase 9X "
                               "runner must compute every outcome relative "
                               "to its condition's K_true (requirement, not "
                               "a model blocker)",
            "procrustes": "procrustes_rmse_z is used only by the Phase 9C "
                          "pilot path, not by the joint runner",
        },
        "blocker": False,
        "future_requirement": "outcomes (exact/under/over, signed error, "
                              "true-K gaps) must use each condition's "
                              "K_true, not the hard-coded K_TRUE = 3",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; never overwritten")
    code_sha, dirty = pilot._git_sha(), pilot._git_dirty()
    args.out.mkdir(parents=True)

    matching = {f"K{k}": {"f_scale": f_scale(k), "w": w_of(k),
                          "avg_row_energy": f_scale(k) ** 2 * k / D,
                          "w2_K": w_of(k) ** 2 * k}
                for k in K_TRUES}
    matching_ok = all(abs(m["avg_row_energy"] - ROW_ENERGY) <= 1e-15
                      and abs(m["w2_K"] - 3.0) <= 1e-15
                      for m in matching.values())
    k3_exact = (f_scale(3) == pc.PROTOCOL.f_scale and w_of(3) == 1.0)

    checks = density_checks()
    primary = []
    for label, ea, er in PRIMARY_SETTINGS:
        m = Adaptive(ea, er)
        row: dict[str, Any] = {"setting": label, "epsabs": ea, "epsrel": er}
        try:
            row.update(calibrate(m), valid=True)
        except Exception as exc:                  # recorded, not changed
            row.update(valid=False, error=f"{type(exc).__name__}: {exc}")
        row.update(max_abserr=m.max_abserr, integration_warnings=m.warnings)
        primary.append(row)
    freeze = freeze_rule(primary, checks["pass"])
    cross = [{"h": h, **calibrate(ExpTrapezoid(h))} for h in H_STEPS]
    cross_rule = crosscheck_rule(cross, freeze["production"])
    prod = freeze["production"]
    pilot._write_csv(args.out / "calibration_by_k.csv", primary)
    pilot._write_csv(args.out / "deterministic_crosscheck.csv", cross)
    pilot._write_json(args.out / "deterministic_crosscheck.json",
                      {"method": f"s = e^t, trapezoid on [{T_LO}, {T_HI}]",
                       "steps": list(H_STEPS), "rows": cross,
                       "rule": cross_rule})

    w0 = ({k: (W0_REF if k == 3 else prod[f"w0_K{k}"]) for k in K_TRUES}
          if prod else None)
    calibration_ok = freeze["pass"] and cross_rule["pass"]
    pilot._write_json(args.out / "calibration_by_k.json", {
        "primary_settings": [list(p) for p in PRIMARY_SETTINGS],
        "split": "[0, 1] + [1, inf)", "quad_limit": QUAD_LIMIT,
        "density_checks": checks, "rows": primary,
        "freeze_rule": {k: v for k, v in freeze.items() if k != "production"},
        "crosscheck_rule": cross_rule,
        "p_target": prod["p_target"] if prod else None,
        "w0": w0, "K3_rule": "w0(3) = -1 by definition; calibrated value "
                             "is a recovery check",
    })

    context, stop = [], None
    if w0 is not None:
        for k in K_TRUES:
            proto = protocol_for(k, w0[k])
            for rep in proto.replicates:
                try:
                    ds = pilot.build_dataset(proto, rep)
                except GeneratorStop as exc:          # recorded, no retry
                    stop = {"K_true": k, "replicate": rep.label,
                            "message": str(exc)}
                    break
                context.append(context_row(k, rep.label, ds, w0[k], w_of(k)))
            if stop:
                break
        pilot._write_csv(args.out / "zero_inference_context.csv", context)
    safe = (w0 is not None and stop is None
            and len(context) == len(K_TRUES) * FUTURE_REPLICATES
            and all(r["finite"] and r["supports_ok"]
                    and r["F_rank"] == r["K_true"] for r in context))

    k1 = k1_boundary_audit()
    pilot._write_json(args.out / "k1_boundary_audit.json", k1)
    anchor = k3_anchor_compatibility(W0_REF)
    pilot._write_json(args.out / "k3_anchor_compatibility.json", anchor)
    reusable = anchor["K3_anchor_reusable"]
    new_k = [k for k in K_TRUES if not (k == 3 and reusable)]
    cap = len(new_k) * FUTURE_REPLICATES * len(CANDIDATE_K) * 2

    ready = (matching_ok and k3_exact and calibration_ok and safe
             and not k1["blocker"])
    decision = ("MATCHED_K_TRUE_DESIGN_READY" if ready else
                "MATCHED_K_TRUE_DESIGN_NEEDS_REVISION")
    pilot._write_json(args.out / "design.json", {
        "design_version": DESIGN_VERSION, "code_sha": code_sha,
        "git_dirty": dirty, "K_true": list(K_TRUES),
        "candidate_K": list(CANDIDATE_K), "matching": matching,
        "matching_ok": matching_ok, "K3_equals_phase9k": k3_exact,
        "p_target": prod["p_target"] if prod else None, "w0": w0,
        "calibration_pass": freeze["pass"],
        "crosscheck_pass": cross_rule["pass"],
        "zero_inference_safe": safe, "generator_stop": stop,
        "K1_blocker": k1["blocker"], "K3_anchor_reusable": reusable,
        "future_new_em_cap": cap, "DECISION": decision,
        "em_executions": 0, "refits": 0,
        "interpretation": "matched-signal K_true sensitivity; matches "
                          "average X loading energy, Y natural-parameter "
                          "variance and population mean edge probability "
                          "only",
    })
    if ready:
        conds = {}
        for k in K_TRUES:
            spec = protocol_for(k, w0[k]).as_json()
            spec["role"] = ("historical anchor: Phase 9K rep01..rep10 "
                            "results reused read-only (not rerun)"
                            if (k == 3 and reusable) else "new run")
            conds[f"K{k}"] = spec
        pilot._write_json(args.out / "future_protocol.json", {
            "phase": "9X (future; NOT RUN)", "status": "FROZEN_NOT_EXECUTED",
            "execution_authorized": False, "design_code_sha": code_sha,
            "p_target": prod["p_target"], "conditions": conds,
            "new_K_true": new_k, "K3_anchor_reusable": reusable,
            "replicates": [r.label for r in
                           pc.PROTOCOL.replicates[:FUTURE_REPLICATES]],
            "candidate_K": list(CANDIDATE_K), "start": "start_B only",
            "future_new_em_cap": cap,
            "data_generator": "run_family_selection_pilot.build_dataset "
                              "(historical canonical mixed generator)",
            "runner_requirement": k1["future_requirement"],
            "primary_outcomes": [
                "K_hat_Lap and K_hat_Q counts per K_true",
                "exact/under/over relative to K_true",
                "signed error K_hat - K_true", "absolute error",
                "best-vs-second gap",
                "criterion difference between K_true and the nearest "
                "under/over candidates", "family assignment",
                "Candidate-B technical status",
                "confusion-style K_true x K_hat table (not an accuracy)"],
            "interpretation": "matched-signal K_true sensitivity; not a "
                              "pure latent-dimension effect",
        })
    print(json.dumps({"calibration": freeze["pass"],
                      "crosscheck": cross_rule["pass"], "safe": safe,
                      "K3_reusable": reusable, "cap": cap,
                      "DECISION": decision}))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
