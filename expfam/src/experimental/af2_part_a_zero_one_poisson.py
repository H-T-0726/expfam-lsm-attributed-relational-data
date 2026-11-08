"""AF2 Part A: true-Poisson columns that show only {0,1} -- when it happens and
what the selectors then do.  No MCEM.

1. Pointwise orderings on x in {0,1} (exact, eta grid):
     log Bern - log Pois = e^eta - log(1 + e^eta)                 > 0
     log CMP2 - log Pois = A_P(eta) - A_C2(eta)
                         = e^eta - log sum_x e^{eta x}/(x!)^2     > 0
   so on a {0,1} column the main selector (gate {bernoulli, poisson}) always
   picks Bernoulli, and the PR #129 count selector restricted to
   {poisson, cmp_nu2p0} always picks CMP2, whatever the loading.
2. How often a true-Poisson column shows only {0,1}.  With z_i ~ N(0, I) iid
   and eta_i = b + f^T z_i, the x_i are iid given f and
       P(max_i x_i <= 1 | f) = ( E[(1 + e^eta) e^{-e^eta}] )^n,
       eta ~ N(b, ||f||^2),
   computed by Gauss-Hermite quadrature.  The research model has no
   intercept (b = 0); b < 0 is shown ONLY to locate where the blind spot
   appears (sparse counts, e.g. real data or a model with an intercept).
   A Monte Carlo check reproduces the quadrature.
3. Selector behaviour on simulated columns with the TRUE Z (L = 1): columns
   that are {0,1} (expected: 100 % Bernoulli / CMP2) and columns with a value
   >= 2 (Poisson vs CMP2 decided by the score), plus the angle between the
   loading fitted under the wrong (selected) family and the true loading.

Outputs (CSV) go to expfam/results/auto_family_z_recovery/af2_part_a_20261110/.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = ROOT / "expfam" / "results" / "auto_family_z_recovery" / "af2_part_a_20261110"
SEED = 20261110
NS = (20, 40, 75, 150)
SCALES = (0.0, 0.5, 1.0, 1.4142135623730951, 2.0, 2.449489742783178)
INTERCEPTS = (0.0, -1.0, -2.0, -3.0, -4.0)
SIM_INTERCEPTS = (-2.0, -3.0)
SIM_COLUMNS = 200
SIM_SCALE = 1.0


def write(path: Path, rows):
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def p01_single(b: float, s: float) -> float:
    """E[(1 + e^eta) e^{-e^eta}], eta ~ N(b, s^2), Gauss-Hermite (200 nodes)."""
    x, wts = np.polynomial.hermite_e.hermegauss(200)
    eta = b + s * x
    lam = np.exp(eta)
    return float(np.sum(wts * (1.0 + lam) * np.exp(-lam)) / np.sqrt(2 * np.pi))


def orderings():
    from family_specs import BernoulliSpec, build_primary_registry
    reg = build_primary_registry()
    eta = np.linspace(-10.0, 5.0, 3001)
    rows = []
    for x in (0.0, 1.0):
        xv = np.full_like(eta, x)
        lp = reg["poisson"].log_prob(xv, eta)
        db = BernoulliSpec().log_prob(xv, eta) - lp
        dc = reg["cmp_nu2p0"].log_prob(xv, eta) - lp
        rows.append({"x": int(x), "eta_min": eta[0], "eta_max": eta[-1], "grid_points": eta.size,
                     "min_logBern_minus_logPois": float(db.min()),
                     "min_logCMP2_minus_logPois": float(dc.min()),
                     "max_abs_err_vs_closed_form_bern":
                         float(np.max(np.abs(db - (np.exp(eta) - np.log1p(np.exp(eta))))))})
    return rows


def probability_table():
    rows = []
    for b in INTERCEPTS:
        for s in SCALES:
            q = p01_single(b, s)
            for n in NS:
                rows.append({"intercept_b": b, "loading_norm_s": round(s, 6),
                             "in_research_model": b == 0.0, "n": n,
                             "p_single_obs_le1": q, "p_column_all_01": q ** n,
                             "log10_p_column_all_01": n * np.log10(q)})
    return rows


def mc_check(rng):
    rows = []
    for b, s, n in ((-2.0, 1.0, 75), (-3.0, 1.0, 75), (-3.0, 0.5, 40), (0.0, 0.0, 20)):
        reps = 20000
        eta = b + s * rng.standard_normal((reps, n))
        x = rng.poisson(np.exp(eta))
        rows.append({"intercept_b": b, "loading_norm_s": s, "n": n, "mc_reps": reps,
                     "mc_p_column_all_01": float(np.mean(x.max(axis=1) <= 1)),
                     "quadrature_p_column_all_01": p01_single(b, s) ** n})
    return rows


def selector_simulation(rng):
    import family_selection as fs
    import family_selection_multisupport as fsm
    from family_specs import build_primary_registry
    reg = build_primary_registry()
    rows = []
    n, k = 75, 3
    for b in SIM_INTERCEPTS:
        for c in range(SIM_COLUMNS):
            Z = rng.standard_normal((n, k))
            f = rng.standard_normal(k)
            f *= SIM_SCALE / np.linalg.norm(f)
            x = rng.poisson(np.exp(b + Z @ f)).astype(float)
            Zs = Z[:, :, None]
            row = {"intercept_b": b, "column_id": c, "n_ones": int(np.sum(x == 1)),
                   "max_x": int(x.max()), "all_01": bool(x.max() <= 1)}
            if x.sum() == 0:
                row.update({"main_selected": "NA_all_zero", "count_selected": "NA_all_zero"})
                rows.append(row)
                continue
            # main selector (AF1 path): gate on this single column
            gate = fs.support_gate(x[:, None])[0]
            if gate.is_ambiguous:
                recs = fs.score_column_candidates(x, Zs, gate, loading_init=np.zeros(k),
                                                  optimizer="bfgs")
                chosen, margin = fs.select_from_records(recs)
                lb = [r for r in recs if r.family == "bernoulli"][0].loading
                lp = [r for r in recs if r.family == "poisson"][0].loading
                row.update({"main_selected": chosen, "main_margin_neg2": margin,
                            "cos_bern_loading_true": float(lb @ f / (np.linalg.norm(lb) * np.linalg.norm(f) + 1e-300)),
                            "cos_pois_loading_true": float(lp @ f / (np.linalg.norm(lp) * np.linalg.norm(f) + 1e-300))})
            else:
                row.update({"main_selected": gate.fixed_family + "(gate)"})
            # PR #129 count selector restricted to {poisson, cmp_nu2p0}
            recs = [fsm.score_candidate(x, Zs, reg[m], np.zeros(k), 0, route="trust_exact")
                    for m in ("poisson", "cmp_nu2p0")]
            sel = fsm.select_column(recs, ("poisson", "cmp_nu2p0"))
            row.update({"count_selected": sel.selected or f"FAILURE:{sel.failure_reason[:60]}",
                        "count_margin_neg2": sel.margin_neg2 if sel.selected else ""})
            rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    write(out / "orderings_01.csv", orderings())
    write(out / "p_all01_quadrature.csv", probability_table())
    write(out / "p_all01_mc_check.csv", mc_check(rng))
    sim = selector_simulation(rng)
    write(out / "selector_on_simulated_columns.csv", sim)
    summ = []
    for b in SIM_INTERCEPTS:
        for grp, cond in (("all_01", True), ("has_value_ge2", False)):
            sub = [r for r in sim if r["intercept_b"] == b and r["all_01"] == cond
                   and r["main_selected"] != "NA_all_zero"]
            summ.append({"intercept_b": b, "group": grp, "columns": len(sub),
                         "main_bernoulli": sum(r["main_selected"] == "bernoulli" for r in sub),
                         "main_poisson": sum(r["main_selected"] in ("poisson", "poisson(gate)") for r in sub),
                         "count_poisson": sum(r["count_selected"] == "poisson" for r in sub),
                         "count_cmp_nu2p0": sum(r["count_selected"] == "cmp_nu2p0" for r in sub),
                         "count_failures": sum(str(r["count_selected"]).startswith("FAILURE") for r in sub),
                         "median_cos_bern_loading_true": float(np.median([r["cos_bern_loading_true"] for r in sub
                                                                          if "cos_bern_loading_true" in r]))
                         if any("cos_bern_loading_true" in r for r in sub) else "",
                         "median_cos_pois_loading_true": float(np.median([r["cos_pois_loading_true"] for r in sub
                                                                          if "cos_pois_loading_true" in r]))
                         if any("cos_pois_loading_true" in r for r in sub) else ""})
    write(out / "selector_summary.csv", summ)
    (out / "runinfo.json").write_text(json.dumps({
        "script": "expfam/src/experimental/af2_part_a_zero_one_poisson.py", "seed": SEED,
        "python": platform.python_version(), "numpy": np.__version__,
        "sim_columns_per_intercept": SIM_COLUMNS, "sim_loading_norm": SIM_SCALE,
        "note": "true Z used (L = 1); no MCEM; intercepts b < 0 are outside the research model"},
        indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
