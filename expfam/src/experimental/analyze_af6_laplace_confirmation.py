"""Pre-registered analysis for AF6 (fixed before any AF6 data are generated).

Unit: dataset (40).  Every count column of every dataset stays in every
denominator; failed / undecided columns are handled by the rules in
protocol.json["primary_endpoint"], never dropped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

TRUE_MODELS = ["gaussian"] * 3 + ["poisson"] * 3 + ["cmp_nu2p0"] * 3
COUNT_COLS = (3, 4, 5, 6, 7, 8)
REGIMES = ("dense", "sparse")
N_DATASETS = 40
BOOT_N, BOOT_SEED = 10000, 20261116
FINITE_DELTA = ("DECIDED", "DECIDED_LAPLACE_UNASSESSED", "UNDECIDED_LAPLACE", "UNDECIDED_NUMERIC")


def boot_ci(v):
    v = np.asarray(v, float)
    rng = np.random.default_rng(BOOT_SEED)
    m = v[rng.integers(0, v.size, size=(BOOT_N, v.size))].mean(axis=1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def num(c: pd.DataFrame, name: str) -> pd.Series:
    """numeric column, or all-NaN when the column is absent (e.g. no TI run)"""
    if name not in c:
        return pd.Series(np.nan, index=c.index)
    return pd.to_numeric(c[name], errors="coerce")


def frame(stage: Path) -> pd.DataFrame:
    """One row per (dataset, regime, count column) = 40 x 2 x 6 = 480, whatever failed."""
    grid = pd.DataFrame([{"replicate": f"rep{r + 1:02d}", "regime": g, "column": j, "true_model": TRUE_MODELS[j]}
                         for r in range(N_DATASETS) for g in REGIMES for j in COUNT_COLS])
    fits = pd.read_csv(stage / "fit_manifest.csv")
    grid = grid.merge(fits[["replicate", "regime", "status", "assignment"]].rename(columns={"status": "fit_status"}),
                      on=["replicate", "regime"], how="left", validate="m:1")
    grid["fit_status"] = grid.fit_status.fillna("MISSING")
    sel = pd.read_csv(stage / "column_selection.csv", keep_default_na=False, na_values=[""])
    keep = [c for c in sel.columns if c not in ("true_model", "anchor", "anchor_role")]
    c = grid.merge(sel[keep], on=["replicate", "regime", "column"], how="left", validate="1:1")
    c["status"] = c.status.where(c.fit_status == "OK", "FIT_FAILED").fillna("MISSING")
    tm = c.true_model
    fit_ok = c.fit_status == "OK"
    finite = c.status.isin(FINITE_DELTA) & np.isfinite(pd.to_numeric(c.delta, errors="coerce"))
    c["delta"] = pd.to_numeric(c.delta, errors="coerce")
    c["sel_B"] = np.where(fit_ok & (c.q_status == "SELECTED"), c.q_selected, "")
    c["sel_A"] = np.where(finite, np.where(c.delta > 0, "cmp_nu2p0", "poisson"), c.sel_B)
    c["sel_C"] = np.where(fit_ok, c.af2_auto_selected, "")
    c["sel_D"] = np.where(c.status == "DECIDED", c.sel_A, c.sel_B)
    jm = num(c, "delta_jointmap_variant")
    c["sel_E"] = np.where(np.isfinite(jm), np.where(jm > 0, "cmp_nu2p0", "poisson"), c.sel_B)
    ti = num(c, "delta_ti")
    c["ti_available"] = np.isfinite(ti)
    c["sel_TI"] = np.where(c.ti_available, np.where(ti > 0, "cmp_nu2p0", "poisson"), c.sel_A)
    se = num(c, "delta_ti_mc_se")
    c["ti_verified"] = c.ti_available & (np.sign(ti) == np.sign(c.delta)) & (ti.abs() > 2 * se)
    c["laplace_error"] = c.delta - ti
    c["delta_ti"], c["delta_ti_mc_se"], c["ti_hmc_acc_min"] = ti, se, num(c, "ti_hmc_acc_min")
    for m in "ABCDE":
        c[f"ok_{m}"] = c[f"sel_{m}"] == tm
    c["ok_TI"] = c.sel_TI == tm
    return c


def dr(c, a, b, regime, mask=None):
    g = c[c.regime == regime] if mask is None else c[(c.regime == regime) & mask]
    v = (g[f"ok_{a}"].astype(int) - g[f"ok_{b}"].astype(int)).groupby(g.replicate).sum()
    v = v.reindex([f"rep{r + 1:02d}" for r in range(N_DATASETS)], fill_value=0)
    lo, hi = boot_ci(v.to_numpy())
    return {"comparison": f"{a}-{b}", "regime": regime, "datasets": len(v), "mean_D": float(v.mean()),
            "ci_low": lo, "ci_high": hi, "sd_D": float(v.std(ddof=1)), "n_pos": int((v > 0).sum()),
            "n_zero": int((v == 0).sum()), "n_neg": int((v < 0).sum())}


def main(stage: Path) -> None:
    out = stage / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    c = frame(stage)
    c.to_csv(out / "column_annotated.csv", index=False)
    stats = []
    prim = dr(c, "A", "B", "sparse")
    prim.update(endpoint="PRIMARY", success=bool(prim["mean_D"] > 0 and prim["ci_low"] > 0))
    stats.append(prim)
    for a, b, g, lab in [("A", "B", "dense", "secondary"), ("A", "C", "sparse", "secondary"), ("A", "C", "dense", "secondary"),
                         ("D", "B", "sparse", "secondary"), ("D", "B", "dense", "secondary"),
                         ("E", "B", "sparse", "diagnostic"), ("E", "B", "dense", "diagnostic"),
                         ("E", "A", "sparse", "diagnostic"), ("E", "A", "dense", "diagnostic"),
                         ("TI", "B", "sparse", "approximation"), ("TI", "B", "dense", "approximation"),
                         ("B", "C", "sparse", "secondary"), ("B", "C", "dense", "secondary")]:
        s = dr(c, a, b, g)
        s["endpoint"] = lab
        stats.append(s)
    # A-B restricted to TI-verified columns (other columns contribute 0) -- reported separately, never primary
    for g in REGIMES:
        s = dr(c, "A", "B", g, mask=c.ti_verified | ((g == "dense") & ~c.ti_available))
        s.update(endpoint="approximation: TI-verified columns only (+ dense columns outside the TI subset)",
                 comparison="A-B")
        stats.append(s)
    pd.DataFrame(stats).to_csv(stage.parent / "statistical_analysis.csv", index=False)

    acc = []
    for g in REGIMES:
        for tmod in ("all", "poisson", "cmp_nu2p0"):
            sub = c[c.regime == g] if tmod == "all" else c[(c.regime == g) & (c.true_model == tmod)]
            acc.append({"regime": g, "true_model": tmod, "columns": len(sub),
                        **{f"acc_{m}": float(sub[f"ok_{m}"].mean()) for m in ("A", "B", "C", "D", "E", "TI")},
                        "D_differs_from_B": int((sub.sel_D != sub.sel_B).sum())})
    pd.DataFrame(acc).to_csv(out / "accuracy.csv", index=False)

    pair = []
    for g in REGIMES:
        sub = c[c.regime == g]
        for ref in ("B", "C"):
            for m in ("A", "D", "E", "TI"):
                okr, okm = sub[f"ok_{ref}"], sub[f"ok_{m}"]
                pair.append({"regime": g, "reference": ref, "method": m,
                             "corrections": int((~okr & okm).sum()), "harms": int((okr & ~okm).sum()),
                             "harms_poisson_to_cmp": int((okr & ~okm & (sub.true_model == "poisson")).sum()),
                             "harms_cmp_to_poisson": int((okr & ~okm & (sub.true_model == "cmp_nu2p0")).sum()),
                             "ref_wrong": int((~okr).sum()), "both_wrong": int((~okr & ~okm).sum()),
                             "method_wrong": int((~okm).sum()),
                             "method_wrong_equal_anchor_family": int((~okm & (sub[f"sel_{m}"] == sub.anchor_family_j)).sum())})
    pd.DataFrame(pair).to_csv(stage.parent / "paired_comparison.csv", index=False)

    ad = []
    for g in REGIMES:
        sub = c[c.regime == g]
        t = sub[sub.ti_available]
        ad.append({"regime": g, "columns": len(sub),
                   **{f"status_{s}": int((sub.status == s).sum()) for s in
                      ("DECIDED", "DECIDED_LAPLACE_UNASSESSED", "UNDECIDED_LAPLACE", "UNDECIDED_NUMERIC",
                       "UNDECIDED_NOT_CONVERGED", "FAILED", "FIT_FAILED", "MISSING")},
                   "is_readable": int((sub.is_readable.astype(str).str.lower() == "true").sum()),
                   "ti_run": len(t), "ti_verified": int(t.ti_verified.sum()),
                   "ti_sign_agree": int((np.sign(t.delta) == np.sign(t.delta_ti)).sum()),
                   "laplace_error_median": float(t.laplace_error.median()) if len(t) else np.nan,
                   "laplace_error_abs_median": float(t.laplace_error.abs().median()) if len(t) else np.nan,
                   "laplace_error_abs_max": float(t.laplace_error.abs().max()) if len(t) else np.nan,
                   "ti_mc_se_median": float(pd.to_numeric(t.delta_ti_mc_se).median()) if len(t) else np.nan,
                   "ti_mc_se_max": float(pd.to_numeric(t.delta_ti_mc_se).max()) if len(t) else np.nan,
                   "hmc_acc_min": float(pd.to_numeric(t.ti_hmc_acc_min).min()) if len(t) else np.nan,
                   "acc_A_ti_verified": float(t[t.ti_verified].ok_A.mean()) if t.ti_verified.any() else np.nan,
                   "acc_A_not_ti_verified": float(t[~t.ti_verified].ok_A.mean()) if (~t.ti_verified).any() else np.nan})
    pd.DataFrame(ad).to_csv(stage.parent / "approximation_diagnostics.csv", index=False)
    figures(out, c)
    print(json.dumps(prim, indent=2))


def figures(out: Path, c: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, g in zip(axes, REGIMES):
        sub = c[c.regime == g]
        v = (sub.ok_A.astype(int) - sub.ok_B.astype(int)).groupby(sub.replicate).sum()
        vals, cnt = np.unique(v, return_counts=True)
        ax.bar(vals, cnt, color="#4C72B0")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{g}: D_r = correct(Laplace) - correct(Q same anchor)\nmean {v.mean():.2f}", fontsize=9)
        ax.set_xlabel("D_r")
    axes[0].set_ylabel("datasets")
    fig.tight_layout()
    fig.savefig(out / "fig1_dr_distribution.png", dpi=150)
    plt.close(fig)
    t = c[c.ti_available]
    if len(t):
        fig, ax = plt.subplots(figsize=(5, 5))
        for g, col in zip(REGIMES, ("#4C72B0", "#DD8452")):
            s = t[t.regime == g]
            ax.scatter(s.delta_ti, s.delta, s=12, color=col, label=g)
        lim = float(np.nanmax(np.abs(t[["delta", "delta_ti"]].to_numpy()))) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color="gray", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("Delta by thermodynamic integration (at f*)")
        ax.set_ylabel("Delta by Laplace")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig2_laplace_vs_ti.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
