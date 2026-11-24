"""Zero-EM generator diagnostics for the K-selection Exp2-generalisation audit.

Purpose: decide whether each candidate data generator is a valid draw from the
canonical Dual-ExpFam model when K_TRUE varies over {1,3,5,7,9}.  No model is
fitted, no criterion is computed, no K is selected.  Only properties of the
generated data (support, moments, rank, saturation, clipping) are measured.

Generators examined (all code unchanged, imported read-only):
  H   historical  expfam/src/data_generator_expfam.generate_dual_data (defaults)
  PC  paper-code  reproduction/src/data_generator.set_true_params      (G/B only)
  PT  paper-text  Mikawa et al. 2024 Sect.4.1 Steps 1-3 as read in the
                  paper-Fig.3 study (F ~ N(0,10) variance, Sigma=0.1 I, no
                  normalisation, w0 ~ N(0,1), w = 3 N(0,1))              (G/B only)
  CM  canonical-clean-v1 with the matched-signal convention used by the clean
      n-sweep / d-sweep (avg ||f_l||^2 = 0.5, w = 1*sqrt(3/K), w0 = -1,
      sigma_x_var = 1, sigma_y_sd = 1)

Output: one CSV row per (series, generator, K_TRUE, seed) plus a summary CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "expfam" / "src"))
sys.path.insert(0, str(ROOT / "expfam" / "src" / "experimental"))
sys.path.insert(0, str(ROOT / "reproduction" / "src"))

from data_generator_expfam import generate_dual_data  # noqa: E402
from data_generator import set_true_params  # noqa: E402
from data_generator_canonical import (  # noqa: E402
    GeneratorStop, generate_canonical_data, f_scale_for_row_norm,
    w_for_matched_y_signal)

N, D = 150, 15
K_TRUES = (1, 3, 5, 7, 9)
SERIES = {"O": ("gaussian", "bernoulli"), "A": ("poisson", "bernoulli"),
          "B": ("gaussian", "poisson"), "C": ("bernoulli", "gaussian")}
SEEDS = tuple(range(5))
SAT = 5.0          # |eta| > 5  <=>  p < 0.0067 or p > 0.9933
CLIP_HI = 10.0     # historical Poisson clip upper bound (np.clip(eta,-20,10))


def paper_text(k, seed):
    rng = np.random.default_rng(9_260_000 + 100 * k + seed)
    Z = rng.standard_normal((N, k))
    F = rng.normal(0.0, np.sqrt(10.0), (D, k))
    w0 = rng.standard_normal()
    w = 3.0 * rng.standard_normal()
    X = Z @ F.T + np.sqrt(0.1) * rng.standard_normal((N, D))
    return dict(Z=Z, F=F, X=X, w0=w0, w=w)


def eta_x_of(gen, fam_x, Z, F):
    return Z @ F.T


def measure(series, gen, k, seed):
    fx, fy = SERIES[series]
    row = dict(series=series, generator=gen, k_true=k, seed=seed,
               family_x=fx, family_y=fy, status="OK", note="")
    try:
        if gen == "H":
            dd = generate_dual_data(n=N, d=D, k=k, seed=4000 + 37 * k + seed,
                                    family_x=fx, family_y=fy)
            Z, F, X, Y, w0, w = dd["Z"], dd["F"], dd["X"], dd["Y"], dd["w0"], dd["w"]
        elif gen == "PC":
            dd = set_true_params(n=N, d=D, k=k, seed=5000 + 37 * k + seed)
            Z, F, X, Y, w0, w = dd["Z"], dd["F"], dd["X"], dd["Y"], dd["w0"], dd["w"]
        elif gen == "PT":
            dd = paper_text(k, seed)
            Z, F, X, w0, w = dd["Z"], dd["F"], dd["X"], dd["w0"], dd["w"]
            Y = None
        else:  # CM
            ds = generate_canonical_data(
                n=N, d=D, k=k, seed=6000 + 37 * k + seed, family_x=fx, family_y=fy,
                f_scale=f_scale_for_row_norm(0.5, d=D, k=k),
                w0=-1.0, w=w_for_matched_y_signal(1.0, k=k, k_ref=3),
                sigma_x_var=1.0, sigma_y_sd=1.0)
            Z, F, X, Y = ds.Z, ds.F, ds.X, ds.Y
            w0, w = ds.metadata["w0"], ds.metadata["w"]
    except GeneratorStop as e:
        row.update(status="GENERATOR_STOP", note=str(e)[:160])
        return row
    iu = np.triu_indices(N, 1)
    G = Z @ Z.T
    eta_y = (w0 + w * G)[iu]
    eta_x = eta_x_of(gen, fx, Z, F)
    sv = np.linalg.svd(F, compute_uv=False)
    row.update(
        w0=float(w0), w=float(w), w2K=float(w * w * k),
        rank_F=int(np.linalg.matrix_rank(F)), rank_Z=int(np.linalg.matrix_rank(Z)),
        F_cond=float(sv[0] / sv[-1]), mean_f_row_norm_sq=float(np.mean(np.sum(F * F, 1))),
        Z_col_mean_absmax=float(np.max(np.abs(Z.mean(0)))),
        Z_col_var_min=float(Z.var(0).min()), Z_col_var_max=float(Z.var(0).max()),
        X_col_mean_absmax=float(np.max(np.abs(X.mean(0)))),
        X_col_var_min=float(X.var(0).min()), X_col_var_max=float(X.var(0).max()),
        X_exact_zscore=bool(np.allclose(X.mean(0), 0, atol=1e-10) and np.allclose(X.var(0), 1, atol=1e-8)),
        eta_x_sd=float(eta_x.std()), eta_x_max=float(eta_x.max()),
        frac_eta_x_sat=float(np.mean(np.abs(eta_x) > SAT)),
        frac_eta_x_over_clip=float(np.mean(eta_x > CLIP_HI)),
        eta_y_sd=float(eta_y.std()), eta_y_max=float(eta_y.max()), eta_y_min=float(eta_y.min()),
        frac_eta_y_sat=float(np.mean(np.abs(eta_y) > SAT)),
        frac_eta_y_over_clip=float(np.mean(eta_y > CLIP_HI)),
    )
    if fx == "bernoulli":
        row["X_support_ok"] = bool(np.isin(X, (0.0, 1.0)).all())
        row["X_mean"] = float(X.mean())
    elif fx == "poisson":
        row["X_support_ok"] = bool((X >= 0).all() and np.allclose(X, np.round(X)))
        row["X_mean"] = float(X.mean()); row["X_max"] = float(X.max())
    else:
        row["X_support_ok"] = True
    if Y is not None:
        yv = Y[iu]
        row["Y_symmetric_zero_diag"] = bool(np.allclose(Y, Y.T) and np.all(np.diag(Y) == 0))
        row["Y_mean"] = float(yv.mean())
        if fy == "bernoulli":
            row["Y_support_ok"] = bool(np.isin(yv, (0.0, 1.0)).all())
        elif fy == "poisson":
            row["Y_support_ok"] = bool((yv >= 0).all())
            row["Y_max"] = float(yv.max()); row["Y_frac_zero"] = float(np.mean(yv == 0))
            row["lambda_y_max_unclipped_log"] = float(eta_y.max())
        else:
            row["Y_support_ok"] = True
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    rows = []
    for s in SERIES:
        gens = ["H", "CM"] + (["PC", "PT"] if s == "O" else [])
        for g in gens:
            for k in K_TRUES:
                for seed in SEEDS:
                    rows.append(measure(s, g, k, seed))
    keys = sorted({kk for r in rows for kk in r}, key=lambda x: (x not in rows[0], x))
    with open(out / "generator_diagnostics_per_draw.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    # summary: mean over seeds of numeric fields, status counts
    import pandas as pd
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns.drop(["k_true", "seed"], errors="ignore")
    summ = df.groupby(["series", "generator", "k_true"]).agg(
        n_draws=("seed", "size"), n_stop=("status", lambda s: int((s != "OK").sum())),
        **{c: (c, "mean") for c in num}).reset_index()
    summ.to_csv(out / "generator_diagnostics_summary.csv", index=False)
    (out / "runinfo.json").write_text(json.dumps({
        "script": "tools/research_audit/diag_k_exp2_generators.py", "n": N, "d": D,
        "k_trues": K_TRUES, "seeds_per_cell": len(SEEDS), "em_fits": 0,
        "purpose": "zero-EM generator validity diagnostics (no criterion, no K selection)"},
        indent=2))
    print(summ.to_string())


if __name__ == "__main__":
    main()
