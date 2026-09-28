"""Phase 9S2 (Issue #112): exact-density one-dimensional recalibration of
the Phase 9S mean-density-controlled w design. Zero EM.

Estimand (unchanged from Phase 9S): K = 3, z_i, z_j iid N(0, I_3),
S = z_i^T z_j, p_bar(b, w) = E[sigmoid(b + w S)], p_target = p_bar(-1, 1),
and one fixed w0(w) per frozen w with p_bar(w0(w), w) = p_target.

Exact density (derived in the report): phi_S(t) = (1 + t^2)^(-K/2), and
for K = 3, f_S(s) = |s| K_1(|s|) / pi. By symmetry

    p_bar(b, w) = (1/pi) int_0^inf [sigmoid(b + w s) + sigmoid(b - w s)]
                                    * s K_1(s) ds,

with s K_1(s) -> 1 as s -> 0+ (implemented explicitly, and as
s * k1e(s) * exp(-s) elsewhere for stability).

Everything numerical below is frozen before the run:
  PRIMARY_SETTINGS  adaptive QUADPACK (scipy.integrate.quad) on [0, inf)
  CROSSCHECK_NODES  fixed-node Gauss-Legendre after s = x / (1 - x)
  FREEZE_TOL        1e-12 (unchanged from Phase 9S)
  DENSITY_TOL       1e-11
The production calibration is frozen only if the two strictest valid
primary settings agree within FREEZE_TOL on p_target and all three w0,
every root residual under the stricter setting is <= FREEZE_TOL, and the
density checks pass. The cross-check passes only if its two finest node
counts agree within FREEZE_TOL and its finest values agree with the
production values within FREEZE_TOL. The Monte Carlo audit and the future
Phase 9T protocol are produced only if both deterministic routes pass.

The canonical generator and all estimators are untouched. Lineage E.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq
from scipy.special import expit, k1e, roots_legendre

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import density_w_calibration as dc                                 # noqa: E402
import run_family_selection_pilot as pilot                         # noqa: E402
import run_lap_vs_cq_20 as pc                                      # noqa: E402
import run_w_sensitivity as ws                                     # noqa: E402

RECALIBRATION_VERSION = "density-w-recalibration-v1"
K_TRUE = dc.K_TRUE
BASELINE = dc.BASELINE
W_VALUES = dict(dc.W_VALUES)
PRIMARY_SETTINGS = (("A", 1e-10, 1e-10), ("B", 1e-12, 1e-12),
                    ("C", 2e-13, 2e-13))            # (label, epsabs, epsrel)
QUAD_LIMIT = 200
CROSSCHECK_NODES = (256, 512, 1024, 2048)
FREEZE_TOL = 1e-12
DENSITY_TOL = 1e-11
CHAR_T = (0.5, 1.0, 2.0)
PHASE9S_PROVISIONAL = {"p_target": 0.3314062121315355,
                       "w0_weak_w": -0.878099440538959,
                       "w0_strong_w": -1.1890648128624175}
PHASE9S_DIR = ws.REPO_ROOT / "expfam/results/density_controlled_w_design/phase9s_20260928"


def s_k1(s: np.ndarray | float) -> np.ndarray:
    """s * K_1(s) for s >= 0, with the exact limit 1 at s = 0."""
    s = np.asarray(s, float)
    out = np.ones_like(s)
    pos = s > 0
    out[pos] = s[pos] * k1e(s[pos]) * np.exp(-s[pos])
    return out


def density(s: np.ndarray | float) -> np.ndarray:
    """f_S(s) = |s| K_1(|s|) / pi (K = 3)."""
    return s_k1(np.abs(np.asarray(s, float))) / math.pi


def _integrand(b: float, w: float) -> Callable[[float], float]:
    def f(s: float) -> float:
        return float((expit(b + w * s) + expit(b - w * s))
                     * s_k1(s)) / math.pi
    return f


# --------------------------------------------------------------------------
# primary: adaptive QUADPACK
# --------------------------------------------------------------------------

class Adaptive:
    def __init__(self, epsabs: float, epsrel: float):
        self.epsabs, self.epsrel = epsabs, epsrel
        self.max_abserr = 0.0
        self.warnings = 0

    def integrate(self, f: Callable[[float], float]) -> float:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IntegrationWarning)
            value, abserr = quad(f, 0.0, math.inf, epsabs=self.epsabs,
                                 epsrel=self.epsrel, limit=QUAD_LIMIT)
        self.max_abserr = max(self.max_abserr, abserr)
        self.warnings += sum(issubclass(c.category, IntegrationWarning)
                             for c in caught)
        return value

    def p_bar(self, b: float, w: float) -> float:
        return self.integrate(_integrand(b, w))


# --------------------------------------------------------------------------
# cross-check: fixed-node Gauss-Legendre on s = x / (1 - x)
# --------------------------------------------------------------------------

class FixedNode:
    def __init__(self, n: int):
        x, wx = roots_legendre(n)
        x = 0.5 * (x + 1.0)                          # [0, 1)
        wx = 0.5 * wx
        self.s = x / (1.0 - x)
        self.weights = wx / (1.0 - x) ** 2 * s_k1(self.s) / math.pi

    def p_bar(self, b: float, w: float) -> float:
        return float(np.dot(self.weights, expit(b + w * self.s)
                            + expit(b - w * self.s)))


# --------------------------------------------------------------------------
# root solve (Phase 9S bracket logic)
# --------------------------------------------------------------------------

def bracket(fn: Callable[[float], float], target: float
            ) -> tuple[float, float]:
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
    target = method.p_bar(*BASELINE)
    out: dict[str, Any] = {"p_target": target}
    for cond, w in W_VALUES.items():
        fn = lambda b, w=w: method.p_bar(b, w)                    # noqa: E731
        lo, hi = bracket(fn, target)
        w0 = brentq(lambda b: fn(b) - target, lo, hi, xtol=dc.ROOT_XTOL,
                    rtol=dc.ROOT_RTOL, maxiter=500)
        out[f"w0_{cond}"] = w0
        out[f"residual_{cond}"] = fn(w0) - target
    out["baseline_recovery_error"] = out["w0_baseline"] - BASELINE[0]
    return out


KEYS = ("p_target",) + tuple(f"w0_{c}" for c in W_VALUES)


def max_abs_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {k: abs(a[k] - b[k]) for k in KEYS}


def freeze_rule(rows: Sequence[dict[str, Any]], density_ok: bool
                ) -> dict[str, Any]:
    """Two strictest valid primary settings must agree within FREEZE_TOL."""
    valid = [r for r in rows if r.get("valid")]
    if len(valid) < 2:
        return {"pass": False, "reason": "fewer than two valid settings"}
    a, b = valid[-2], valid[-1]
    diffs = max_abs_diff(a, b)
    residuals = {c: abs(b[f"residual_{c}"]) for c in W_VALUES}
    ok = (all(v <= FREEZE_TOL for v in diffs.values())
          and all(v <= FREEZE_TOL for v in residuals.values())
          and abs(b["baseline_recovery_error"]) <= FREEZE_TOL
          and density_ok)
    return {"pass": ok, "compared": [a["setting"], b["setting"]],
            "differences": diffs, "stricter_residuals": residuals,
            "density_ok": density_ok, "production": b if ok else None}


def crosscheck_rule(rows: Sequence[dict[str, Any]],
                    production: dict[str, Any] | None) -> dict[str, Any]:
    a, b = rows[-2], rows[-1]
    internal = max_abs_diff(a, b)
    vs = max_abs_diff(b, production) if production else None
    ok = (production is not None
          and all(v <= FREEZE_TOL for v in internal.values())
          and all(v <= FREEZE_TOL for v in vs.values()))
    return {"pass": ok, "internal_finest_two": internal,
            "finest_vs_primary": vs}


# --------------------------------------------------------------------------
# density checks (independent of the calibration)
# --------------------------------------------------------------------------

def density_checks() -> dict[str, Any]:
    opts = dict(epsabs=1e-14, epsrel=1e-13, limit=QUAD_LIMIT)
    norm = 2 * quad(lambda s: float(s_k1(s)) / math.pi, 0, math.inf,
                    **opts)[0]
    second = 2 * quad(lambda s: s * s * float(s_k1(s)) / math.pi, 0,
                      math.inf, **opts)[0]
    fourth = 2 * quad(lambda s: s ** 4 * float(s_k1(s)) / math.pi, 0,
                      math.inf, **opts)[0]
    char = {}
    for t in CHAR_T:
        val = 2 * quad(lambda s: float(s_k1(s)) / math.pi, 0, math.inf,
                       weight="cos", wvar=t)[0]
        char[str(t)] = {"numerical": val,
                        "exact": (1 + t * t) ** (-K_TRUE / 2),
                        "abs_error": abs(val - (1 + t * t) ** (-K_TRUE / 2))}
    out = {"normalization": norm, "normalization_error": abs(norm - 1),
           "mean": 0.0, "mean_note": "exactly 0 by symmetry of f_S",
           "second_moment": second, "second_moment_error": abs(second - 3),
           "fourth_moment": fourth,
           # E[S^4] = 3 E[||z_i||^4] = 3 K (K + 2)
           "fourth_moment_exact": 3.0 * K_TRUE * (K_TRUE + 2),
           "characteristic_function": char,
           "tolerance": DENSITY_TOL}
    out["fourth_moment_error"] = abs(fourth - out["fourth_moment_exact"])
    out["pass"] = (out["normalization_error"] <= DENSITY_TOL
                   and out["second_moment_error"] <= DENSITY_TOL)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; a recorded run is never "
                         f"overwritten")
    code_sha, dirty = pilot._git_sha(), pilot._git_dirty()
    args.out.mkdir(parents=True)
    checks = density_checks()
    pilot._write_json(args.out / "density_checks.json", checks)

    primary = []
    for label, epsabs, epsrel in PRIMARY_SETTINGS:
        method = Adaptive(epsabs, epsrel)
        row: dict[str, Any] = {"setting": label, "epsabs": epsabs,
                               "epsrel": epsrel, "limit": QUAD_LIMIT}
        try:
            row.update(calibrate(method), valid=True)
        except Exception as exc:                  # recorded, not changed
            row.update(valid=False, error=f"{type(exc).__name__}: {exc}")
        row.update(max_abserr=method.max_abserr,
                   integration_warnings=method.warnings)
        primary.append(row)
    pilot._write_csv(args.out / "primary_adaptive_convergence.csv", primary)
    freeze = freeze_rule(primary, checks["pass"])

    cross = [{"nodes": n, **calibrate(FixedNode(n))}
             for n in CROSSCHECK_NODES]
    cross_rule = crosscheck_rule(cross, freeze["production"])
    pilot._write_csv(args.out / "fixed_node_crosscheck.csv", cross)
    pilot._write_json(args.out / "fixed_node_crosscheck.json",
                      {"nodes": list(CROSSCHECK_NODES),
                       "map": "s = x / (1 - x), Gauss-Legendre on [0, 1)",
                       "rows": cross, "rule": cross_rule})

    ready = freeze["pass"] and cross_rule["pass"]
    prod = freeze["production"]
    reference = prod if prod else next(
        (r for r in reversed(primary) if r.get("valid")), None)
    comparison = ({k: {"phase9s2": reference[k],
                       "phase9s_provisional": v,
                       "difference": reference[k] - v}
                   for k, v in PHASE9S_PROVISIONAL.items()}
                  if reference else None)
    decision = ("DENSITY_CONTROLLED_W_CALIBRATION_READY" if ready else
                "DENSITY_CONTROLLED_W_CALIBRATION_NEEDS_REVISION")
    calibration = {
        "recalibration_version": RECALIBRATION_VERSION,
        "code_sha": code_sha, "git_dirty": dirty,
        "estimand": "p_bar(w0, w) = E[sigmoid(w0 + w z_i^T z_j)], K=3",
        "density": "f_S(s) = |s| K_1(|s|) / pi",
        "primary_settings": [list(p) for p in PRIMARY_SETTINGS],
        "quad_limit": QUAD_LIMIT, "freeze_tol": FREEZE_TOL,
        "density_tol": DENSITY_TOL,
        "crosscheck_nodes": list(CROSSCHECK_NODES),
        "root_solver": {"method": "brentq", "xtol": dc.ROOT_XTOL,
                        "rtol": dc.ROOT_RTOL,
                        "bracket": "deterministic doubling from [-1, 1] "
                                   "(Phase 9S)"},
        "density_checks_pass": checks["pass"],
        "freeze_rule": {k: v for k, v in freeze.items()
                        if k != "production"},
        "crosscheck_rule": cross_rule,
        "p_target": prod["p_target"] if prod else None,
        "calibrated": ({c: {"w0": prod[f"w0_{c}"], "w": w,
                            "p_bar_residual": prod[f"residual_{c}"]}
                        for c, w in W_VALUES.items()} if prod else None),
        "production_setting": prod["setting"] if prod else None,
        "comparison_to_phase9s_provisional": comparison,
        "DECISION": decision,
        "controls_only": "population mean Bernoulli edge probability; not "
                         "the probability variance, quantiles, saturation, "
                         "full distribution or realized finite-sample "
                         "density",
        "em_executions": 0, "refits": 0,
    }
    if ready:
        w0 = {c: prod[f"w0_{c}"] for c in W_VALUES}
        calibrated = {c: (w0[c], w) for c, w in W_VALUES.items()}
        mc = dc.mc_crosscheck(calibrated)
        for c, v in mc["conditions"].items():
            v["deterministic_p_bar"] = prod["p_target"]
            v["difference"] = prod["p_target"] - v["estimate"]
            v["difference_in_se"] = v["difference"] / v["standard_error"]
        pilot._write_json(args.out / "independent_mc_crosscheck.json", mc)
        protocols = dc.future_protocols(w0)
        base = pc.PROTOCOL.as_json()
        pilot._write_json(args.out / "future_protocol.json", {
            "phase": "9T (future; NOT RUN)", "status": "FROZEN_NOT_EXECUTED",
            "execution_authorized": False,
            "recalibration_version": RECALIBRATION_VERSION,
            "calibration_code_sha": code_sha,
            "p_target": prod["p_target"],
            "frozen": {c: {"w0": w0[c], "w": w} for c, w in W_VALUES.items()},
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
                          if ws.protocol_for(c).as_json()[k]
                          != protocols[c][k]) for c in protocols},
            "new_em_cap": 400,
            "preflight_required": "regenerate weak/base/strong datasets "
                                  "without inference; Z, F, X bit-identical "
                                  "in 20/20 (as Phase 9P); Y differs",
            "y_context_required": [
                "frozen w0 / w", "population p_target",
                "per-replicate true mean edge probability",
                "realized edge density", "eta mean / sd",
                "probability mean / sd", "probability q01 q05 q50 q95 q99",
                "share p < 0.05", "share p > 0.95"],
            "primary_outcomes": ["technical completeness", "K_hat_Lap",
                                 "K_hat_Q", "exact/under/over",
                                 "paired C_Q vs C_Lap", "delta_23",
                                 "best-vs-second gap", "K2->3 decomposition",
                                 "Candidate-B technical diagnostics",
                                 "family assignment (secondary)"],
            "interpretation": "mean-density-controlled relational-w "
                              "sensitivity; population mean edge "
                              "probability matched by design only",
        })
        calibration["mc_crosscheck"] = "independent_mc_crosscheck.json"
        calibration["future_protocol"] = "FROZEN_NOT_EXECUTED"
    else:
        calibration["mc_crosscheck"] = "NOT_RUN (deterministic not READY)"
        calibration["future_protocol"] = "NOT_FROZEN"
    pilot._write_json(args.out / "calibration.json", calibration)
    print(json.dumps({"freeze": freeze["pass"],
                      "crosscheck": cross_rule["pass"],
                      "DECISION": decision}))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
