"""AF8 analysis (written before any AF8 selection value is inspected).

--pilot-audit : numerical stop rules on the pilot (no accuracy is computed)
--analyze     : full analysis (exploratory; dataset = statistical unit)
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

import run_af8_integrated_loading as a8     # noqa: E402

P, C = a8.P, a8.C
REPS = [f"rep{r + 1:02d}" for r in range(40)]


def pivot(c: pd.DataFrame) -> pd.DataFrame:
    """one row per (replicate, regime, column, prior) with C - P differences."""
    keys = ["replicate", "regime", "column", "prior"]
    c = c.copy()
    c["ok"] = c.status == "OK"
    vals = [v for v in ("D", "E", "E2", "IS", "IS_mc_se", "IS_ess", "D_start_disagreement", "E_start_disagreement",
                        "E_min_eig", "E_logdet_schur_f", "E_logprior", "E_phi", "D_S_at_fD", "D_logprior_at_fD",
                        "fD_norm", "fE_norm", "seconds") if v in c]
    w = c.pivot_table(index=keys, columns="candidate", values=vals, aggfunc="first")
    ok = c.pivot_table(index=keys, columns="candidate", values="ok", aggfunc="first")
    d = pd.DataFrame(index=w.index)
    d["both_ok"] = ok.reindex(columns=[P, C]).fillna(False).astype(bool).all(axis=1)
    for v in ("D", "E", "E2", "IS", "E_logdet_schur_f", "E_logprior", "E_phi", "D_S_at_fD", "D_logprior_at_fD",
              "fD_norm", "fE_norm"):
        if (v, P) in w and (v, C) in w:
            d[f"Delta_{v}"] = pd.to_numeric(w[v, C]) - pd.to_numeric(w[v, P])
    if ("IS_mc_se", P) in w:
        d["IS_se_delta"] = np.sqrt(pd.to_numeric(w["IS_mc_se", P]) ** 2 + pd.to_numeric(w["IS_mc_se", C]) ** 2)
        d["IS_ess_min"] = np.minimum(pd.to_numeric(w["IS_ess", P]), pd.to_numeric(w["IS_ess", C]))
    num = lambda v: w[v].apply(pd.to_numeric, errors="coerce")       # noqa: E731
    d["start_dis"] = np.maximum(num("D_start_disagreement").max(axis=1), num("E_start_disagreement").max(axis=1))
    d["E_min_eig"] = num("E_min_eig").min(axis=1)
    d["seconds"] = num("seconds").sum(axis=1)
    return d.reset_index()


def pilot_audit() -> dict:
    c = pd.read_csv(a8.OUT / "pilot" / "candidate_scores.csv")
    d = pivot(c)
    m = d[d.prior == a8.MAIN_PRIOR]
    ok = m[m.both_ok]
    res = {"cells": len(c), "failed_cells": int((c.status != "OK").sum()),
           "failed_share": float((c.status != "OK").mean()),
           "start_disagreement_gt_1e-6_share": float((d.start_dis > 1e-6).mean()),
           "E_min_eig_min": float(d.E_min_eig.min()),
           "main_columns": len(m), "main_both_ok": len(ok),
           "median_abs_DeltaE_minus_DeltaIS": float((ok.Delta_E - ok.Delta_IS).abs().median()),
           "max_abs_DeltaE_minus_DeltaIS": float((ok.Delta_E - ok.Delta_IS).abs().max()),
           "median_abs_DeltaE2_minus_DeltaIS": float((ok.Delta_E2 - ok.Delta_IS).abs().median()),
           "max_abs_DeltaE2_minus_DeltaIS": float((ok.Delta_E2 - ok.Delta_IS).abs().max()),
           "median_IS_se_delta": float(ok.IS_se_delta.median()), "min_IS_ess": float(ok.IS_ess_min.min()),
           "median_abs_DeltaE": float(ok.Delta_E.abs().median()),
           "sign_agree_E_IS": int((np.sign(ok.Delta_E) == np.sign(ok.Delta_IS)).sum()),
           "sign_agree_E2_IS": int((np.sign(ok.Delta_E2) == np.sign(ok.Delta_IS)).sum()),
           "seconds_total": float(c.seconds.sum())}
    sr = a8.STOP_RULES
    res["stop"] = {
        "failed_share": res["failed_share"] > sr["pilot_E_failed_share_max"],
        "start_disagreement": res["start_disagreement_gt_1e-6_share"] > sr["mode_start_disagreement_share_max"],
        "E_vs_IS_abs": res["median_abs_DeltaE_minus_DeltaIS"] > sr["pilot_median_abs_DeltaE_minus_DeltaIS_max"],
        "E_vs_IS_relative": res["median_abs_DeltaE_minus_DeltaIS"] > 0.5 * res["median_abs_DeltaE"],
    }
    res["proceed"] = not any(res["stop"].values())
    (a8.OUT / "pilot" / "pilot_numerical_audit.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    d.to_csv(a8.OUT / "pilot" / "pilot_numerics_by_column.csv", index=False)
    print(json.dumps(res, indent=2))
    return res


# --------------------------------------------------------------------------
# full analysis
# --------------------------------------------------------------------------

def auc(x, y):
    from scipy.stats import mannwhitneyu
    x, y = np.asarray(x, float), np.asarray(y, bool)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    if y.all() or (~y).all():
        return np.nan
    return float(mannwhitneyu(x[y], x[~y]).statistic / (y.sum() * (~y).sum()))


def af6_frame() -> pd.DataFrame:
    import analyze_af6_laplace_confirmation as an6
    c = an6.frame(a8.AF6)
    c = c.rename(columns={"sel_A": "sel_A", "sel_B": "sel_B", "sel_E": "sel_J"})
    c["score_A"] = c.delta
    c["score_B"] = pd.to_numeric(c.q_margin_cmp_minus_pois, errors="coerce")
    c["score_J"] = pd.to_numeric(c.delta_jointmap_variant, errors="coerce")
    return c[["replicate", "regime", "column", "true_model", "sel_A", "sel_B", "sel_J", "score_A", "score_B", "score_J"]]


def build(prior: str, d: pd.DataFrame, base: pd.DataFrame, fail_wrong=False) -> pd.DataFrame:
    s = base.merge(d[d.prior == prior], on=["replicate", "regime", "column"], how="left")
    s["both_ok"] = s.both_ok.fillna(False).astype(bool)
    for m in ("D", "E", "E2", "IS"):
        col = f"Delta_{m}"
        if col not in s:
            continue
        x = pd.to_numeric(s[col], errors="coerce")
        fin = s.both_ok & np.isfinite(x)
        fb = "" if fail_wrong else s.sel_B
        s[f"sel_{m}"] = np.where(fin, np.where(x > 0, C, P), fb)
        s[f"score_{m}"] = x.where(fin)
        s[f"undecided_{m}"] = fin & (x.abs() <= np.maximum(a8.NUMERIC_FLOOR, s.start_dis.fillna(0)))
    for m in ("A", "B", "J", "D", "E", "E2", "IS"):
        if f"sel_{m}" in s:
            s[f"ok_{m}"] = s[f"sel_{m}"] == s.true_model
    return s


def metrics(s: pd.DataFrame, m: str) -> dict:
    tc = (s.true_model == C).to_numpy()
    ok = s[f"ok_{m}"].to_numpy()
    accP, accC = ok[~tc].mean(), ok[tc].mean()
    return {"acc_poisson": float(accP), "acc_cmp2": float(accC), "balanced": float((accP + accC) / 2),
            "acc_all": float(ok.mean()), "auc": auc(s[f"score_{m}"], tc) if f"score_{m}" in s else np.nan}


def cluster_boot(s: pd.DataFrame, methods, ref_pairs, B=2000):
    rng = np.random.default_rng(a8.BOOT_SEED)
    idx = {r: np.flatnonzero(s.replicate.to_numpy() == r) for r in REPS}
    draws = []
    for _ in range(B):
        t = s.iloc[np.concatenate([idx[r] for r in rng.choice(REPS, len(REPS))])]
        draws.append({m: metrics(t, m) for m in methods})
    out = {}
    for m in methods:
        for k in ("acc_poisson", "acc_cmp2", "balanced", "auc"):
            v = np.array([d_[m][k] for d_ in draws], float)
            out[(m, k)] = (float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975)))
    for a, b in ref_pairs:
        for k in ("acc_poisson", "acc_cmp2", "balanced", "auc"):
            v = np.array([d_[a][k] - d_[b][k] for d_ in draws], float)
            out[(f"{a}-{b}", k)] = (float(np.nanmean(v)), float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975)))
    return out


def per_dataset(s, a, b):
    v = (s[f"ok_{a}"].astype(int) - s[f"ok_{b}"].astype(int)).groupby(s.replicate).sum().reindex(REPS, fill_value=0)
    rng = np.random.default_rng(a8.BOOT_SEED)
    mb = v.to_numpy()[rng.integers(0, 40, (10000, 40))].mean(1)
    return {"mean_D": float(v.mean()), "ci_low": float(np.quantile(mb, .025)), "ci_high": float(np.quantile(mb, .975)),
            "n_pos": int((v > 0).sum()), "n_zero": int((v == 0).sum()), "n_neg": int((v < 0).sum())}


def analyze():
    out = a8.OUT
    an = out / "full" / "analysis"
    an.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(out / "full" / "candidate_scores.csv")
    d = pivot(c)
    base = af6_frame()
    main = build(a8.MAIN_PRIOR, d, base)
    main.to_csv(out / "selection_results.csv", index=False)
    methods = ["B", "A", "J", "D", "E", "E2", "IS"]
    pairs = [("E", "A"), ("E", "B"), ("E", "D"), ("E", "J"), ("E2", "D"), ("IS", "D"), ("D", "A"), ("IS", "A")]
    stats, paired, trans = [], [], []
    for g in a8.REGIMES:
        s = main[main.regime == g]
        cb = cluster_boot(s, methods, pairs)
        for m in methods:
            r = {"regime": g, "method": m, "prior": a8.MAIN_PRIOR if m in ("D", "E", "E2", "IS") else "",
                 **metrics(s, m)}
            for k in ("acc_poisson", "acc_cmp2", "balanced", "auc"):
                r[f"{k}_ci_low"], r[f"{k}_ci_high"] = cb[(m, k)]
            if f"undecided_{m}" in s:
                r["undecided_numeric"] = int(s[f"undecided_{m}"].sum())
                r["failed_fallback_to_Q"] = int((~s.both_ok).sum())
            stats.append(r)
        for a, b in pairs:
            r = {"regime": g, "comparison": f"{a}-{b}", **per_dataset(s, a, b)}
            for k in ("acc_poisson", "acc_cmp2", "balanced", "auc"):
                r[f"diff_{k}"], r[f"diff_{k}_ci_low"], r[f"diff_{k}_ci_high"] = cb[(f"{a}-{b}", k)]
            oa, ob = s[f"ok_{a}"], s[f"ok_{b}"]
            for t in (P, C):
                tt = s.true_model == t
                r[f"{t}_right_to_wrong"] = int((ob & ~oa & tt).sum())     # b right, a wrong
                r[f"{t}_wrong_to_right"] = int((~ob & oa & tt).sum())
            paired.append(r)
        for t in (P, C):
            tt = s[s.true_model == t]
            trans.append({"regime": g, "true_model": t,
                          "median_DeltaE_minus_DeltaA": float((tt.score_E - tt.score_A).median()),
                          "median_DeltaE_minus_DeltaD": float((tt.score_E - tt.score_D).median()),
                          "median_DeltaD_minus_DeltaA": float((tt.score_D - tt.score_A).median()),
                          "median_DeltaE2_minus_DeltaD": float((tt.score_E2 - tt.score_D).median()),
                          "median_DeltaIS_minus_DeltaE": float((tt.score_IS - tt.score_E).median()),
                          "median_DeltaA": float(tt.score_A.median()), "median_DeltaD": float(tt.score_D.median()),
                          "median_DeltaE": float(tt.score_E.median()), "median_DeltaIS": float(tt.score_IS.median()),
                          "median_Delta_logdet_schur_f_halfneg": float((-0.5 * tt.Delta_E_logdet_schur_f).median()),
                          "median_Delta_logprior_E": float(tt.Delta_E_logprior.median())})
    pd.DataFrame(stats).to_csv(out / "statistical_analysis.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired_comparison.csv", index=False)
    pd.DataFrame(trans).to_csv(an / "score_shift_by_truth.csv", index=False)
    # failure = wrong sensitivity
    fw = build(a8.MAIN_PRIOR, d, base, fail_wrong=True)
    pd.DataFrame([{"regime": g, "method": m, **metrics(fw[fw.regime == g], m)} for g in a8.REGIMES
                  for m in ("D", "E", "E2", "IS")]).to_csv(an / "sensitivity_failure_counted_wrong.csv", index=False)
    # prior sensitivity
    ps = []
    for pr in a8.PRIORS:
        s_all = build(pr, d, base)
        for g in a8.REGIMES:
            s = s_all[s_all.regime == g]
            for m in ("D", "E"):
                tc = s.true_model == C
                ps.append({"prior": pr, "kind": a8.PRIORS[pr][0], "scale": a8.PRIORS[pr][1], "regime": g, "method": m,
                           **metrics(s, m), "failed_columns": int((~s.both_ok).sum()),
                           "median_Delta_true_poisson": float(s.loc[~tc, f"score_{m}"].median()),
                           "median_Delta_true_cmp2": float(s.loc[tc, f"score_{m}"].median()),
                           "selection_changes_vs_main": int((s[f"sel_{m}"].to_numpy()
                                                             != main.loc[main.regime == g, f"sel_{m}"].to_numpy()).sum())})
    pd.DataFrame(ps).to_csv(out / "prior_sensitivity.csv", index=False)
    # numerical summary
    num = []
    for g in a8.REGIMES:
        s = main[(main.regime == g) & main.both_ok]
        num.append({"regime": g, "columns_ok": len(s), "failed_columns": int((main[main.regime == g].both_ok == False).sum()),
                    "median_abs_DeltaE_minus_DeltaIS": float((s.score_E - s.score_IS).abs().median()),
                    "max_abs_DeltaE_minus_DeltaIS": float((s.score_E - s.score_IS).abs().max()),
                    "median_abs_DeltaE2_minus_DeltaIS": float((s.score_E2 - s.score_IS).abs().median()),
                    "max_abs_DeltaE2_minus_DeltaIS": float((s.score_E2 - s.score_IS).abs().max()),
                    "sign_disagree_E_IS": int((np.sign(s.score_E) != np.sign(s.score_IS)).sum()),
                    "sign_disagree_E2_IS": int((np.sign(s.score_E2) != np.sign(s.score_IS)).sum()),
                    "median_IS_se_delta": float(s.IS_se_delta.median()), "max_IS_se_delta": float(s.IS_se_delta.max()),
                    "min_IS_ess": float(s.IS_ess_min.min()), "E_min_eig_min": float(s.E_min_eig.min()),
                    "start_dis_gt_1e-6": int((d[d.regime == g].start_dis > 1e-6).sum()),
                    "start_dis_max": float(d[d.regime == g].start_dis.max()),
                    "cells_failed_all_priors": int(((c.regime == g) & (c.status != "OK")).sum()),
                    "seconds_main_prior_total": float(main[main.regime == g].seconds.sum()),
                    "seconds_all_cells_total": float(c[c.regime == g].seconds.sum())})
    pd.DataFrame(num).to_csv(an / "numerical_summary.csv", index=False)
    figures(main, pd.DataFrame(ps), an)
    print(pd.DataFrame(stats).round(3).to_string())


def figures(main, ps, an):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey="row")
    for i, g in enumerate(a8.REGIMES):
        s = main[main.regime == g]
        for k, (m, lab) in enumerate((("A", "A: Laplace, f optimised (AF6)"), ("D", "D: prior MAP"), ("E", "E: f integrated (joint Laplace)"))):
            ax = axes[i, k]
            for t, col in ((P, "#1f77b4"), (C, "#d62728")):
                x = pd.to_numeric(s.loc[s.true_model == t, f"score_{m}"], errors="coerce").dropna().clip(-20, 20)
                ax.hist(x, bins=40, range=(-20, 20), alpha=.55, color=col, label=f"true {t}")
            ax.axvline(0, color="k", lw=.8)
            ax.set_title(f"{g}: {lab}", fontsize=9)
            if k == 0:
                ax.set_ylabel("columns")
            if i == 1:
                ax.set_xlabel("score difference CMP2 - Poisson (clipped to +-20)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(an / "fig1_delta_by_truth.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, g in zip(axes, a8.REGIMES):
        t = ps[(ps.regime == g) & (ps.method == "E")]
        x = np.arange(len(t))
        ax.plot(x, t.acc_poisson, "o-", label="true Poisson")
        ax.plot(x, t.acc_cmp2, "s-", label="true CMP2")
        ax.plot(x, t.balanced, "k^-", label="balanced")
        ax.set_xticks(x, t.prior, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0.4, 1.02)
        ax.set_title(f"{g}: E (f integrated) by prior")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(an / "fig2_prior_sensitivity.png", dpi=130)
    plt.close(fig)


def pilot_diagnose() -> dict:
    """Root-cause diagnostics for the pilot stop (numerical only; no selection/accuracy)."""
    import traceback
    import warnings
    warnings.filterwarnings("ignore")
    res = {}
    # (a) joint Newton from f = 0: which kind of stationary point?
    case = a8.load_case(0, "sparse")
    case["regime"] = "sparse"
    x, Za = case["X"][:, 3], np.asarray(case["st"]["Zmode"], float)
    rows = []
    for cand in a8.CANDS:
        J, _ = a8.joint_for(case, 3, cand, a8.MAIN_PRIOR)
        starts = {"plugin_anchorZ": a8.fixed_z_start(cand, x, Za), "zero": np.zeros(a8.K)}
        fl = a8.af6_fstar("rep01", "sparse", 3, cand)
        if fl is not None:
            starts["af6_fL"] = fl
        for nm, f0 in starts.items():
            Z, f, it, gi = J.mode_u(Za, f0)
            ev = np.linalg.eigvalsh(J.negH_u(Z, f))
            rows.append({"case": "rep01 sparse col3", "candidate": cand, "start": nm, "f_end": np.round(f, 4).tolist(),
                         "grad_inf": gi, "psi": J.psi(Z, f), "negH_min_eig": float(ev[0]), "negH_n_negative": int((ev < 0).sum())})
    res["a_joint_newton_stationary_points"] = rows
    # (b) where D raises for rep01 sparse col3 Poisson
    J, _ = a8.joint_for(case, 3, P, a8.MAIN_PRIOR)
    try:
        a8.method_D(J, Za, {"plugin_anchorZ": a8.fixed_z_start(P, x, Za), "zero": np.zeros(a8.K)})
        res["b_D_failure"] = "no exception"
    except Exception as exc:
        res["b_D_failure"] = {"exception": f"{type(exc).__name__}: {exc}",
                              "trace": traceback.format_exc(limit=4).splitlines()[-4:]}
    Zt, it, gi = J.modeZ(Za.copy())
    res["b_column_x_max"], res["b_column_x_mean"] = float(x.max()), float(x.mean())
    # (c) the Z-only problem at f = 0 (D's zero start)
    case2 = a8.load_case(1, "sparse")
    case2["regime"] = "sparse"
    J2, _ = a8.joint_for(case2, 8, P, a8.MAIN_PRIOR)
    s, Zh, gi, me = J2.S(np.zeros(a8.K), np.asarray(case2["st"]["Zmode"], float))
    res["c_D_zero_start_rep02_sparse_col8"] = {"S": s, "inner_grad_inf": gi, "min_eig_ZZ": me}
    (a8.OUT / "pilot" / "pilot_failure_diagnosis.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=2, default=str))
    return res


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--pilot-audit" in argv:
        pilot_audit()
    if "--pilot-diagnose" in argv:
        pilot_diagnose()
    if "--analyze" in argv:
        analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
