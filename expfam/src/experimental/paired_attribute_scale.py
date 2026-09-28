"""Phase 9U (Issue #116): component-isolated paired generator for an
attribute-loading-scale design. Zero inference.

Forward-only; the historical canonical generators are untouched. Each
replicate's data_seed seeds one numpy SeedSequence whose spawned children
are dedicated streams (frozen layout, recorded as provenance):

    child 0         Z                      z_i ~ N(0, I_K), drawn once
    child 1         F orientation          Gaussian (d x K) -> QR, sign-fixed
                                           (build_full_rank_loadings), drawn once
    child 2 + l     X column l (l = 0..d-1)
    child 2 + d     Y

For each condition c with loading scale f_c, F_c = Q * f_c with the same Q.
Observation coupling (fixed before the preflight):

    Gaussian   eps_il ~ N(0,1) shared;          X = eta + sqrt(sigma_x_var) eps
    Bernoulli  U_il ~ U[0,1) shared;            X = 1[U < sigmoid(eta)]
    Poisson    U_il ~ U[0,1) shared;            X = F_Pois^{-1}(U; lambda = exp(eta))
               (inverse CDF: smallest x with F(x) >= U; U = 0 -> 0). Every
               entry is checked to satisfy F(x - 1) < U <= F(x); a failure is
               recorded as a coupling defect, never replaced by another method.
    Y          V_ij ~ U[0,1) shared, i < j;     Y = 1[V < sigmoid(w0 + w z_i^T z_j)]

Each condition's marginal law is the canonical model; only the coupling
across conditions is new. Poisson rates use canonical_poisson_rate (no
clipping; lambda_max gate). No inference is imported or run. Lineage E.
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
from scipy.stats import poisson

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_generator_canonical import (                              # noqa: E402
    DEFAULT_POISSON_LAMBDA_MAX,
    GeneratorStop,
    build_full_rank_loadings,
    canonical_poisson_rate,
    canonical_sigmoid,
)

PAIRED_GENERATOR_VERSION = "paired-attribute-scale-v1"
F_SCALES = {"weak": 1.0, "base": math.sqrt(2.0), "strong": 2.0}
CONDITIONS = tuple(F_SCALES)
N, D, K_TRUE = 75, 12, 3
FAMILY_X = ("gaussian",) * 3 + ("bernoulli",) * 6 + ("poisson",) * 3
SIGMA_X_VAR = 1.0
W0, W = -1.0, 1.0
LAMBDA_MAX = DEFAULT_POISSON_LAMBDA_MAX
N_CHILDREN = 2 + D + 1
QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)


@dataclasses.dataclass
class PairedDataset:
    condition: str
    f_scale: float
    Z: np.ndarray
    Q: np.ndarray
    F: np.ndarray
    eta_x: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    poisson_inversion_ok: bool
    max_poisson_rate: float


def streams(data_seed: int) -> list[np.random.SeedSequence]:
    return np.random.SeedSequence(data_seed).spawn(N_CHILDREN)


def stream_provenance(data_seed: int) -> list[dict[str, Any]]:
    names = (["Z", "F_orientation"] + [f"X_column_{l}" for l in range(D)]
             + ["Y"])
    return [{"data_seed": data_seed, "stream": name, "child_index": i,
             "entropy": str(ss.entropy), "spawn_key": list(ss.spawn_key)}
            for i, (name, ss) in enumerate(zip(names, streams(data_seed)))]


def _draws(data_seed: int) -> dict[str, Any]:
    """Every random input, drawn once per replicate from dedicated streams."""
    ch = streams(data_seed)
    z = np.random.default_rng(ch[0]).standard_normal((N, K_TRUE))
    q = build_full_rank_loadings(np.random.default_rng(ch[1]), d=D,
                                 k=K_TRUE, f_scale=1.0)
    base = []
    for l, family in enumerate(FAMILY_X):
        rng = np.random.default_rng(ch[2 + l])
        base.append(rng.standard_normal(N) if family == "gaussian"
                    else rng.random(N))
    m = N * (N - 1) // 2
    v = np.random.default_rng(ch[2 + D]).random(m)
    return {"Z": z, "Q": q, "column_base": base, "V": v}


def poisson_inverse_cdf(u: np.ndarray, rate: np.ndarray
                        ) -> tuple[np.ndarray, bool]:
    x = poisson.ppf(u, rate)
    x = np.where(u == 0.0, 0.0, x)
    ok = bool(np.all(np.isfinite(x)) and np.all(x >= 0))
    if ok:
        upper = poisson.cdf(x, rate)
        lower = np.where(x > 0, poisson.cdf(x - 1, rate), 0.0)
        ok = bool(np.all((lower < u) | (u == 0.0)) and np.all(u <= upper))
    return x.astype(np.float64), ok


def _support_ok(values: np.ndarray, family: str) -> bool:
    if not np.all(np.isfinite(values)):
        return False
    if family == "bernoulli":
        return bool(np.all((values == 0.0) | (values == 1.0)))
    if family == "poisson":
        return bool(np.all(values >= 0.0) and np.all(values == np.floor(values)))
    return True


def generate(data_seed: int, condition: str,
             draws: dict[str, Any] | None = None) -> PairedDataset:
    f_scale = F_SCALES[condition]
    draws = _draws(data_seed) if draws is None else draws
    z, q = draws["Z"], draws["Q"]
    f = q * f_scale
    if int(np.linalg.matrix_rank(f)) != K_TRUE:
        raise GeneratorStop(f"rank(F) != {K_TRUE}")
    eta_x = z @ f.T
    x = np.empty((N, D))
    inversion_ok, max_rate = True, 0.0
    for l, family in enumerate(FAMILY_X):
        base = draws["column_base"][l]
        if family == "gaussian":
            x[:, l] = eta_x[:, l] + math.sqrt(SIGMA_X_VAR) * base
        elif family == "bernoulli":
            x[:, l] = (base < canonical_sigmoid(eta_x[:, l])).astype(float)
        else:
            rate = canonical_poisson_rate(eta_x[:, l], lambda_max=LAMBDA_MAX)
            max_rate = max(max_rate, float(rate.max()))
            x[:, l], ok = poisson_inverse_cdf(base, rate)
            inversion_ok = inversion_ok and ok
        if not _support_ok(x[:, l], family):
            raise GeneratorStop(f"column {l} ({family}) outside its support")
    upper = np.triu_indices(N, k=1)
    eta_y = W0 + W * (z @ z.T)[upper]
    y = np.zeros((N, N))
    y[upper] = (draws["V"] < canonical_sigmoid(eta_y)).astype(float)
    y = y + y.T
    return PairedDataset(condition, f_scale, z, q, f, eta_x, x, y,
                         inversion_ok, max_rate)


def generate_all(data_seed: int) -> dict[str, PairedDataset]:
    draws = _draws(data_seed)
    return {c: generate(data_seed, c, draws) for c in CONDITIONS}


# --------------------------------------------------------------------------
# preflight quantities
# --------------------------------------------------------------------------

def _q(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {f"{prefix}_q{int(round(q * 100)):02d}": float(np.quantile(values, q))
            for q in QUANTILES}


def signal_context(rep: str, ds: PairedDataset) -> dict[str, Any]:
    fam = np.array(FAMILY_X)
    g, b, p = (np.flatnonzero(fam == name)
               for name in ("gaussian", "bernoulli", "poisson"))
    rn = np.sum(ds.F ** 2, axis=1)
    sv = np.linalg.svd(ds.F, compute_uv=False)
    prob = canonical_sigmoid(ds.eta_x[:, b]).ravel()
    rate = np.exp(ds.eta_x[:, p]).ravel()
    gvar = np.var(ds.eta_x[:, g], axis=0)
    upper = np.triu_indices(N, k=1)
    row = {"replicate": rep, "condition": ds.condition, "f_scale": ds.f_scale,
           "singular_values": ";".join(f"{v:.15g}" for v in sv),
           "row_norm_sq_min": float(rn.min()),
           "row_norm_sq_median": float(np.median(rn)),
           "row_norm_sq_max": float(rn.max()),
           "row_norm_sq_mean": float(rn.mean()),
           "eta_sd_overall": float(ds.eta_x.std()),
           **{f"eta_sd_{name}": float(ds.eta_x[:, idx].std())
              for name, idx in (("gaussian", g), ("bernoulli", b),
                                ("poisson", p))},
           **_q(ds.eta_x[:, g].ravel(), "eta_gaussian"),
           **_q(ds.eta_x[:, b].ravel(), "eta_bernoulli"),
           **_q(ds.eta_x[:, p].ravel(), "eta_poisson"),
           "gaussian_signal_var_min": float(gvar.min()),
           "gaussian_signal_var_mean": float(gvar.mean()),
           "gaussian_signal_var_max": float(gvar.max()),
           "gaussian_noise_var": SIGMA_X_VAR,
           "gaussian_snr_mean": float(gvar.mean() / SIGMA_X_VAR),
           "bernoulli_prob_mean": float(prob.mean()),
           "bernoulli_prob_sd": float(prob.std()),
           **_q(prob, "bernoulli_prob"),
           "bernoulli_share_p_lt_0.05": float(np.mean(prob < 0.05)),
           "bernoulli_share_p_gt_0.95": float(np.mean(prob > 0.95)),
           "bernoulli_observed_mean": float(ds.X[:, b].mean()),
           "poisson_rate_mean": float(rate.mean()),
           "poisson_rate_median": float(np.median(rate)),
           "poisson_rate_max": float(rate.max()),
           **_q(rate, "poisson_rate"),
           "poisson_safety_margin": float(LAMBDA_MAX / rate.max()),
           "poisson_theoretical_mean": float(np.mean(np.exp(rn[p] / 2))),
           "poisson_observed_mean": float(ds.X[:, p].mean()),
           "poisson_observed_max": float(ds.X[:, p].max()),
           "poisson_inversion_ok": ds.poisson_inversion_ok,
           "y_edge_density": float(ds.Y[upper].mean())}
    return row


def pairing_check(rep: str, data_seed: int, sets: dict[str, PairedDataset]
                  ) -> dict[str, Any]:
    w, b, s = (sets[c] for c in CONDITIONS)
    again = generate_all(data_seed)
    return {
        "replicate": rep, "data_seed": data_seed,
        "Z_identical": bool(np.array_equal(w.Z, b.Z) and np.array_equal(w.Z, s.Z)),
        "Q_identical": bool(np.array_equal(w.Q, b.Q) and np.array_equal(w.Q, s.Q)),
        "F_exact_scale": all(bool(np.array_equal(sets[c].F, w.Q * F_SCALES[c]))
                             for c in CONDITIONS),
        "F_weak_equals_Q": bool(np.array_equal(w.F, w.Q)),
        "rank_F_3": all(int(np.linalg.matrix_rank(sets[c].F)) == K_TRUE
                        for c in CONDITIONS),
        "Y_identical": bool(np.array_equal(w.Y, b.Y) and np.array_equal(w.Y, s.Y)),
        "X_differs_weak_base": bool(not np.array_equal(w.X, b.X)),
        "X_differs_base_strong": bool(not np.array_equal(b.X, s.X)),
        "finite": all(bool(np.all(np.isfinite(sets[c].X))) for c in CONDITIONS),
        "poisson_inversion_ok": all(sets[c].poisson_inversion_ok
                                    for c in CONDITIONS),
        "max_poisson_rate_strong": s.max_poisson_rate,
        "poisson_safe": all(sets[c].max_poisson_rate <= LAMBDA_MAX
                            for c in CONDITIONS),
        "deterministic_repeat": all(
            bool(np.array_equal(again[c].X, sets[c].X)
                 and np.array_equal(again[c].Y, sets[c].Y))
            for c in CONDITIONS),
    }


REQUIRED_CHECKS = ("Z_identical", "Q_identical", "F_exact_scale",
                   "F_weak_equals_Q", "rank_F_3", "Y_identical", "finite",
                   "poisson_inversion_ok", "poisson_safe",
                   "deterministic_repeat")


def historical_comparison(rep, historical, paired_base) -> dict[str, Any]:
    return {"replicate": rep.label, "data_seed": rep.data_seed,
            **{f"{m}_identical": bool(np.array_equal(getattr(historical, m),
                                                     getattr(paired_base, m)))
               for m in ("Z", "F", "X", "Y")}}


def base_reuse_decision(comparison: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reusable = all(r[f"{m}_identical"] for r in comparison
                   for m in ("Z", "F", "X", "Y"))
    return {"historical_base_reusable": reusable,
            "future_base_rerun_required": not reusable,
            "future_new_em_cap": (2 if reusable else 3) * 20 * 5 * 2}


def main(argv: Sequence[str] | None = None) -> int:
    import run_family_selection_pilot as pilot            # provenance/io only
    import run_lap_vs_cq_20 as pc

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; a recorded run is never "
                         f"overwritten")
    code_sha = pilot._git_sha()
    reps = pc.PROTOCOL.replicates
    provenance, checks, context, history = [], [], [], []
    stop = None
    for rep in reps:
        provenance += stream_provenance(rep.data_seed)
        try:
            sets = generate_all(rep.data_seed)
        except GeneratorStop as exc:                      # recorded, no retry
            stop = {"replicate": rep.label, "message": str(exc)}
            break
        checks.append(pairing_check(rep.label, rep.data_seed, sets))
        context += [signal_context(rep.label, sets[c]) for c in CONDITIONS]
        history.append(historical_comparison(
            rep, pilot.build_dataset(pc.PROTOCOL, rep), sets["base"]))
    args.out.mkdir(parents=True)
    pilot._write_csv(args.out / "stream_provenance.csv", provenance)
    pilot._write_json(args.out / "stream_provenance.json", provenance)
    if checks:
        pilot._write_csv(args.out / "pairing_checks.csv", checks)
        pilot._write_csv(args.out / "attribute_signal_context.csv", context)
        pilot._write_csv(args.out / "historical_base_comparison.csv", history)
    all_pass = (stop is None and len(checks) == len(reps)
                and all(c[k] for c in checks for k in REQUIRED_CHECKS))
    reuse = base_reuse_decision(history) if history else None
    pilot._write_json(args.out / "pairing_checks.json", {
        "required_checks": list(REQUIRED_CHECKS),
        "pass_counts": {k: sum(bool(c[k]) for c in checks)
                        for k in REQUIRED_CHECKS},
        "replicates": len(checks), "generator_stop": stop,
        "all_pass": all_pass})
    pilot._write_json(args.out / "historical_base_comparison.json", {
        "counts": {f"{m}_identical": sum(r[f"{m}_identical"] for r in history)
                   for m in ("Z", "F", "X", "Y")},
        "replicates": len(history), **(reuse or {})})
    decision = ("ATTRIBUTE_SCALE_DESIGN_READY" if all_pass else
                "ATTRIBUTE_SCALE_DESIGN_NEEDS_REVISION")
    energy = {c: {"f_scale": f, "avg_row_norm_sq_derived": f * f * K_TRUE / D}
              for c, f in F_SCALES.items()}
    pilot._write_json(args.out / "design.json", {
        "paired_generator_version": PAIRED_GENERATOR_VERSION,
        "code_sha": code_sha,
        "f_scales": F_SCALES, "loading_energy": energy,
        "common": {"n": N, "d": D, "K_true": K_TRUE,
                   "family_x": list(FAMILY_X), "family_y": "bernoulli",
                   "sigma_x_var": SIGMA_X_VAR, "w0": W0, "w": W,
                   "poisson_lambda_max": LAMBDA_MAX},
        "streams": "SeedSequence(data_seed).spawn(15): 0 Z, 1 F_orientation, "
                   "2..13 X columns 0..11, 14 Y",
        "coupling": {"gaussian": "shared standard-normal noise",
                     "bernoulli": "shared uniform, X = 1[U < sigmoid(eta)]",
                     "poisson": "shared uniform + inverse Poisson CDF "
                                "(scipy.stats.poisson.ppf; U = 0 -> 0), "
                                "entrywise inversion check",
                     "Y": "dedicated stream, shared uniform, "
                          "Y = 1[V < sigmoid(-1 + z_i^T z_j)]"},
        "DECISION": decision, "em_executions": 0, "refits": 0,
        "interpretation": "attribute-loading-scale sensitivity; not a pure "
                          "attribute-information effect",
    })
    if all_pass:
        base_json = pc.PROTOCOL.as_json()
        conditions = {}
        for c, f in F_SCALES.items():
            proto = dataclasses.replace(pc.PROTOCOL,
                                        stage=f"attribute_scale_{c}",
                                        f_scale=f).as_json()
            proto["data_generator"] = (
                f"paired_attribute_scale.generate(data_seed, {c!r}) "
                f"({PAIRED_GENERATOR_VERSION}); NOT "
                f"run_family_selection_pilot.build_dataset")
            conditions[c] = proto
        pilot._write_json(args.out / "future_protocol.json", {
            "phase": "9V (future; NOT RUN)", "status": "FROZEN_NOT_EXECUTED",
            "execution_authorized": False,
            "design_code_sha": code_sha,
            "conditions": conditions,
            "differs_from_phase9k_protocol_only_in": {
                c: sorted(k for k in conditions[c]
                          if base_json.get(k) != conditions[c][k])
                for c in conditions},
            **reuse,
            "baseline_policy": ("rerun all three conditions with the paired "
                                "generator; the historical Phase 9K base is "
                                "not bit-compatible")
            if reuse["future_base_rerun_required"] else
            "historical Phase 9K base reusable",
            "preflight_required": "regenerate all three conditions without "
                                  "inference; Z, Q, Y identical and exact F "
                                  "scale relation in 20/20",
            "primary_outcomes": ["technical completeness", "K_hat_Lap",
                                 "K_hat_Q", "exact/under/over",
                                 "matched selected-K transitions across "
                                 "attribute scale", "delta_23",
                                 "best-vs-second gap", "K2->3 decomposition",
                                 "family-selection behavior",
                                 "Candidate-B technical diagnostics",
                                 "X-signal context by family"],
            "interpretation": "attribute-loading-scale sensitivity with Z "
                              "and realized Y shared; not a pure attribute "
                              "information effect",
        })
    print(json.dumps({"all_pass": all_pass, "stop": stop, "reuse": reuse,
                      "DECISION": decision}))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
