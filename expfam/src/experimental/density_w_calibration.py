"""Phase 9S (Issue #110): mean-density-controlled relational-w design.

Zero-EM calibration only. For K_true = 3 and z_i, z_j iid N(0, I_K),

    p_bar(b, w) = E[ sigmoid(b + w * S) ],   S = z_i^T z_j.

Given R = ||z_i||, S | R ~ N(0, R^2); with T = R^2 / 2 ~ Gamma(K/2, 1)
and G ~ N(0, 1) independent, S = sqrt(2T) G, so

    p_bar(b, w) = E_{T,G}[ sigmoid(b + w sqrt(2T) G) ],

evaluated by generalized Gauss-Laguerre (T, alpha = K/2 - 1) x
Gauss-Hermite-normal (G) quadrature. The target is the population value
p_target = p_bar(-1, 1) (never an observed density); for each frozen w one
fixed intercept w0(w) solves p_bar(w0, w) = p_target (brentq).

Numerical rules, fixed before any result is seen:
  RESOLUTIONS are evaluated in increasing order; the production resolution
  is the smallest one whose p_target and all three calibrated w0 agree with
  the next finer resolution within AGREEMENT_TOL (absolute). If none does,
  the calibration is not frozen (NEEDS_REVISION).

The canonical generator and all estimators are untouched. Lineage E.

Usage::

    python density_w_calibration.py --out <fresh directory>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, gammaln, roots_genlaguerre, roots_hermitenorm

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import run_family_selection_pilot as pilot                         # noqa: E402
import run_lap_vs_cq_20 as pc                                      # noqa: E402
import run_w_sensitivity as ws                                     # noqa: E402

CALIBRATION_VERSION = "density-w-calibration-v1"
K_TRUE = 3
BASELINE = (-1.0, 1.0)                               # (w0, w)
W_VALUES = dict(ws.W_VALUES)                         # weak / baseline / strong
RESOLUTIONS = ((16, 32), (32, 64), (64, 128), (128, 256))   # (n_T, n_G)
AGREEMENT_TOL = 1e-12
ROOT_XTOL = 1e-15
ROOT_RTOL = 4 * np.finfo(float).eps
MC_SEED = 20260928
MC_SAMPLES = 10_000_000
MC_CHUNK = 1_000_000


@dataclasses.dataclass(frozen=True)
class Rule:
    """Tensor quadrature nodes for E[f(sqrt(2T) G)]."""
    s: np.ndarray            # values of sqrt(2T) * G
    weights: np.ndarray      # probability weights (sum to 1)


def quadrature(n_t: int, n_g: int, k: int = K_TRUE) -> Rule:
    alpha = k / 2 - 1
    t, wt = roots_genlaguerre(n_t, alpha)            # weight t^alpha e^{-t}
    wt = wt / math.exp(gammaln(k / 2))               # -> Gamma(k/2, 1) law
    g, wg = roots_hermitenorm(n_g)                   # weight e^{-g^2/2}
    wg = wg / math.sqrt(2 * math.pi)                 # -> N(0, 1) law
    s = np.sqrt(2 * t)[:, None] * g[None, :]
    return Rule(s=s.ravel(), weights=(wt[:, None] * wg[None, :]).ravel())


def p_bar(b: float, w: float, rule: Rule) -> float:
    return float(np.dot(rule.weights, expit(b + w * rule.s)))


def bracket(target: float, w: float, rule: Rule) -> tuple[float, float]:
    """Deterministic expanding bracket around the root."""
    lo, hi = -1.0, 1.0
    while p_bar(lo, w, rule) > target:
        lo *= 2.0
        if lo < -1e3:
            raise ValueError("no lower bracket")
    while p_bar(hi, w, rule) < target:
        hi *= 2.0
        if hi > 1e3:
            raise ValueError("no upper bracket")
    return lo, hi


def solve_w0(target: float, w: float, rule: Rule) -> float:
    lo, hi = bracket(target, w, rule)
    return brentq(lambda b: p_bar(b, w, rule) - target, lo, hi,
                  xtol=ROOT_XTOL, rtol=ROOT_RTOL, maxiter=500)


def calibrate(n_t: int, n_g: int) -> dict[str, Any]:
    rule = quadrature(n_t, n_g)
    target = p_bar(*BASELINE, rule)
    out: dict[str, Any] = {"n_T": n_t, "n_G": n_g,
                           "weights_sum": float(rule.weights.sum()),
                           "p_target": target}
    for cond, w in W_VALUES.items():
        w0 = solve_w0(target, w, rule)
        out[f"w0_{cond}"] = w0
        out[f"residual_{cond}"] = p_bar(w0, w, rule) - target
    out["baseline_recovery_error"] = out["w0_baseline"] - BASELINE[0]
    return out


def monotonicity_check(w: float, rule: Rule,
                       grid: np.ndarray | None = None) -> dict[str, Any]:
    """Numerical companion to the exact strict-monotonicity argument."""
    grid = np.linspace(-12.0, 12.0, 481) if grid is None else grid
    vals = np.array([p_bar(b, w, rule) for b in grid])
    return {"grid": [float(grid[0]), float(grid[-1]), len(grid)],
            "strictly_increasing": bool(np.all(np.diff(vals) > 0)),
            "p_at_ends": [float(vals[0]), float(vals[-1])]}


def select_production(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    keys = ["p_target"] + [f"w0_{c}" for c in W_VALUES]
    for a, b in zip(rows, rows[1:]):
        if all(abs(a[k] - b[k]) <= AGREEMENT_TOL for k in keys):
            return a
    return None


def mc_crosscheck(calibrated: dict[str, tuple[float, float]]
                  ) -> dict[str, Any]:
    """Fixed-seed latent-prior Monte Carlo; diagnostic only."""
    rng = np.random.default_rng(MC_SEED)
    sums = {c: 0.0 for c in calibrated}
    sq = {c: 0.0 for c in calibrated}
    done = 0
    while done < MC_SAMPLES:
        m = min(MC_CHUNK, MC_SAMPLES - done)
        zi = rng.standard_normal((m, K_TRUE))
        zj = rng.standard_normal((m, K_TRUE))
        s = np.einsum("ij,ij->i", zi, zj)
        for c, (w0, w) in calibrated.items():
            p = expit(w0 + w * s)
            sums[c] += float(p.sum())
            sq[c] += float((p * p).sum())
        done += m
    out = {"method": "iid latent-prior Monte Carlo of sigmoid(w0 + w z_i^T z_j)",
           "seed": MC_SEED, "samples": MC_SAMPLES, "conditions": {}}
    for c in calibrated:
        mean = sums[c] / MC_SAMPLES
        var = sq[c] / MC_SAMPLES - mean * mean
        out["conditions"][c] = {"w0": calibrated[c][0], "w": calibrated[c][1],
                                "estimate": mean,
                                "standard_error": math.sqrt(var / MC_SAMPLES)}
    return out


def future_protocols(w0: dict[str, float]) -> dict[str, Any]:
    """The Phase 9T design: Phase 9P protocol with calibrated w0. Not run."""
    out = {}
    for cond in ("weak_w", "strong_w"):
        p9p = ws.protocol_for(cond)
        proto = dataclasses.replace(p9p, stage=f"density_w_{cond}",
                                    w0=w0[cond])
        out[cond] = proto.as_json()
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; a recorded run is never "
                         f"overwritten")
    rows = [calibrate(nt, ng) for nt, ng in RESOLUTIONS]
    prod = select_production(rows)
    args.out.mkdir(parents=True)
    pilot._write_csv(args.out / "quadrature_convergence.csv", rows)
    if prod is None:
        pilot._write_json(args.out / "calibration.json", {
            "DECISION": "DENSITY_CONTROLLED_W_DESIGN_NEEDS_REVISION",
            "reason": "no successive-resolution agreement"})
        print("no production resolution")
        return 0
    rule = quadrature(prod["n_T"], prod["n_G"])
    mono = {c: monotonicity_check(w, rule) for c, w in W_VALUES.items()}
    calibrated = {c: (prod[f"w0_{c}"], w) for c, w in W_VALUES.items()}
    cross = mc_crosscheck(calibrated)
    for c, v in cross["conditions"].items():
        v["deterministic_p_bar"] = p_bar(*calibrated[c], rule)
        v["difference"] = v["estimate"] - v["deterministic_p_bar"]
        v["difference_in_se"] = v["difference"] / v["standard_error"]
    baseline_ok = abs(prod["baseline_recovery_error"]) <= AGREEMENT_TOL
    residual_ok = all(abs(prod[f"residual_{c}"]) <= AGREEMENT_TOL
                      for c in W_VALUES)
    ready = baseline_ok and residual_ok and all(
        m["strictly_increasing"] for m in mono.values())
    calibration = {
        "calibration_version": CALIBRATION_VERSION,
        "code_sha": pilot._git_sha(), "git_dirty": pilot._git_dirty(),
        "K_true": K_TRUE, "baseline": {"w0": BASELINE[0], "w": BASELINE[1]},
        "estimand": "p_bar(w0, w) = E[sigmoid(w0 + w z_i^T z_j)], "
                    "z_i, z_j iid N(0, I_3) (population mean edge "
                    "probability)",
        "quadrature": "generalized Gauss-Laguerre (T ~ Gamma(3/2, 1), "
                      "alpha = 1/2) x Gauss-Hermite-normal (G ~ N(0,1)), "
                      "S = sqrt(2T) G",
        "resolutions": [list(r) for r in RESOLUTIONS],
        "production_rule": f"smallest resolution agreeing with the next "
                           f"finer one within {AGREEMENT_TOL} (abs) on "
                           f"p_target and all three w0",
        "production_resolution": [prod["n_T"], prod["n_G"]],
        "root_solver": {"method": "brentq", "xtol": ROOT_XTOL,
                        "rtol": ROOT_RTOL,
                        "bracket": "deterministic doubling from [-1, 1]"},
        "p_target": prod["p_target"],
        "calibrated": {c: {"w0": prod[f"w0_{c}"], "w": w,
                           "w2_K_true": w * w * K_TRUE,
                           "p_bar_residual": prod[f"residual_{c}"]}
                       for c, w in W_VALUES.items()},
        "baseline_recovery_error": prod["baseline_recovery_error"],
        "monotonicity_check": mono,
        "DECISION": ("DENSITY_CONTROLLED_W_DESIGN_READY" if ready else
                     "DENSITY_CONTROLLED_W_DESIGN_NEEDS_REVISION"),
        "controls_only": "population mean Bernoulli edge probability; not "
                         "the probability variance, saturation, full "
                         "edge-probability distribution or realized "
                         "finite-sample density",
        "em_executions": 0, "refits": 0,
    }
    pilot._write_json(args.out / "calibration.json", calibration)
    pilot._write_json(args.out / "independent_crosscheck.json", cross)
    w0 = {c: prod[f"w0_{c}"] for c in W_VALUES}
    protocols = future_protocols(w0)
    base = pc.PROTOCOL.as_json()
    pilot._write_json(args.out / "future_protocol.json", {
        "phase": "9T (future; NOT RUN)", "status": "FROZEN_NOT_EXECUTED",
        "execution_authorized": False,
        "calibration_version": CALIBRATION_VERSION,
        "p_target": prod["p_target"],
        "baseline": {"w0": BASELINE[0], "w": BASELINE[1],
                     "source": "historical Phase 9K artifact "
                               "expfam/results/lap_vs_cq_20/"
                               "phase9k_20260928/ (reuse; never rerun)"},
        "conditions": protocols,
        "differs_from_phase9k_protocol_only_in": {
            c: sorted(k for k in base if base[k] != protocols[c][k])
            for c in protocols},
        "differs_from_phase9p_protocol_only_in": {
            c: sorted(k for k in protocols[c]
                      if ws.protocol_for(c).as_json()[k] != protocols[c][k])
            for c in protocols},
        "new_em_cap": 400,
        "preflight_required": "regenerate weak/base/strong datasets without "
                              "inference and require Z, F, X bit-identical "
                              "in 20/20 (as Phase 9P); Y differs",
        "y_context_required": ["population p_target",
                               "per-replicate true mean edge probability",
                               "realized edge density", "eta_Y mean / sd",
                               "probability quantiles",
                               "saturation diagnostics (share of "
                               "probabilities < 0.05 and > 0.95)"],
        "primary_outcomes": ["technical completeness", "K_hat_Lap",
                             "K_hat_Q", "exact/under/over",
                             "paired C_Q vs C_Lap", "delta_23",
                             "best-vs-second gap", "K2->3 decomposition",
                             "Candidate-B technical diagnostics",
                             "family assignment (secondary)"],
        "interpretation": "mean-density-controlled relational-w "
                          "sensitivity; population mean edge probability "
                          "matched by design only",
    })
    print(json.dumps({"production": [prod["n_T"], prod["n_G"]],
                      "p_target": prod["p_target"], **w0,
                      "DECISION": calibration["DECISION"]}))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
