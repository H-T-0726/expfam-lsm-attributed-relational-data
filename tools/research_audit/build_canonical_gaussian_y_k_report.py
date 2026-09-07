"""
Aggregation, decomposition and plots for the canonical Gaussian-Y K-selection
experiment (2026-09-07).

Reads only `fit_results.csv` (+ generator diagnostics + the historical
fixed-official artifact for a DESCRIPTIVE, explicitly non-comparable table).
Writes CSVs and PNGs. No model is refit here.

Tie rule (declared in the runner, applied identically here):
    smallest_k_among_ties -- ties are additionally written to ties.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
RES = _ROOT / "expfam" / "results" / "canonical_gaussian_y_k_selection_20260907"
FIG = _ROOT / "figures" / "canonical_gaussian_y_k_selection_20260907"
CANDIDATE_K = [1, 2, 3, 4, 5, 6, 7, 8, 9]

C_TRUE, C_ALT, C_PEN, C_PRIOR = "#2196F3", "#FF5722", "#4CAF50", "#9C27B0"


def select_from(sub: pd.DataFrame, score_col: str):
    """Return (selected_k, selected_score, second_k, second_score, tie_ks)."""
    o = sub[sub["success"].astype(bool) & np.isfinite(sub[score_col])]
    if len(o) == 0:
        return None, np.nan, None, np.nan, []
    mn = o[score_col].min()
    ties = sorted(o[o[score_col] == mn]["k_est"].astype(int).tolist())
    sel = min(ties)
    rest = o[o["k_est"] != sel].sort_values(score_col)
    if len(rest):
        return sel, float(mn), int(rest.iloc[0]["k_est"]), float(rest.iloc[0][score_col]), ties
    return sel, float(mn), None, np.nan, ties


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES / "fit_results.csv")

    # =================================================================
    # selection_by_trial.csv + ties.csv + counterfactual diagnostic
    # =================================================================
    sel_rows, tie_rows, cf_rows = [], [], []
    for (cfg, tr), sub in df.groupby(["config_id", "trial"], sort=True):
        k0 = int(sub["K_0"].iloc[0])
        sel, ssc, k2, s2, ties = select_from(sub, "criterion_score")
        tk = sub[sub["k_est"] == k0]
        true_score = float(tk["criterion_score"].iloc[0]) if len(tk) else np.nan
        nfail = int((~sub["success"].astype(bool)).sum())
        sel_rows.append({
            "config_id": cfg, "trial": tr, "data_seed": int(sub["data_seed"].iloc[0]),
            "n": int(sub["n"].iloc[0]), "w_true": float(sub["w_true"].iloc[0]),
            "K_0": k0, "selected_k": sel, "correct": (sel == k0),
            "selected_score": ssc, "true_k_score": true_score,
            "second_best_k": k2, "second_best_score": s2,
            "score_gap_to_second": (s2 - ssc) if np.isfinite(s2) else np.nan,
            "all_candidate_fits_success": nfail == 0,
            "num_failed_candidates": nfail,
            "n_tied_at_min": len(ties),
        })
        if len(ties) > 1:
            tie_rows.append({"config_id": cfg, "trial": tr, "K_0": k0,
                             "tied_k": ",".join(map(str, ties)), "score": ssc,
                             "tie_rule": "smallest_k_among_ties", "resolved_to": sel})
        # counterfactual: same criterion minus the direct latent-prior term
        cf, cfs, cf2, cfs2, cft = select_from(sub, "score_without_direct_prior")
        cf_rows.append({"config_id": cfg, "trial": tr, "K_0": k0, "n": int(sub["n"].iloc[0]),
                        "w_true": float(sub["w_true"].iloc[0]),
                        "selected_k_current_criterion": sel,
                        "selected_k_without_direct_prior": cf,
                        "changed": sel != cf, "cf_selected_score": cfs,
                        "cf_second_best_k": cf2, "cf_gap_to_second": (cfs2 - cfs) if np.isfinite(cfs2) else np.nan,
                        "n_tied_at_min": len(cft)})

    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(RES / "selection_by_trial.csv", index=False)
    pd.DataFrame(tie_rows).to_csv(RES / "ties.csv", index=False)
    pd.DataFrame(cf_rows).to_csv(RES / "diagnostic_counterfactual_selection.csv", index=False)

    # =================================================================
    # aggregate_recovery.csv
    # =================================================================
    agg = []
    for (k0, n, w), sub in sel_df.groupby(["K_0", "n", "w_true"], sort=True):
        counts = sub["selected_k"].value_counts().to_dict()
        cor = sub[sub["correct"]]
        wrg = sub[~sub["correct"]]
        row = {
            "K_0": k0, "n": n, "w": w,
            "num_trials": len(sub),
            "num_complete_trials": int(sub["all_candidate_fits_success"].sum()),
            "recovery_count": int(sub["correct"].sum()),
            "recovery_rate": float(sub["correct"].mean()),
            "selected_k_mean": float(sub["selected_k"].mean()),
            "selected_k_median": float(sub["selected_k"].median()),
            "selected_k_mode": int(sub["selected_k"].mode().iloc[0]),
            "mean_score_gap_when_correct": float(cor["score_gap_to_second"].mean()) if len(cor) else np.nan,
            "mean_score_gap_when_wrong": float(wrg["score_gap_to_second"].mean()) if len(wrg) else np.nan,
        }
        for k in CANDIDATE_K:
            row[f"selection_counts_k{k}"] = int(counts.get(k, 0))
        agg.append(row)
    agg_df = pd.DataFrame(agg)
    agg_df.to_csv(RES / "aggregate_recovery.csv", index=False)

    # =================================================================
    # confusion_matrix.csv  (core arm only, w = 0.5)
    # =================================================================
    core = sel_df[sel_df["w_true"] == 0.5]
    cm = pd.crosstab(core["K_0"], core["selected_k"])
    cm = cm.reindex(columns=CANDIDATE_K, fill_value=0)
    cm.to_csv(RES / "confusion_matrix.csv")

    # =================================================================
    # aggregate_score_by_k.csv
    # =================================================================
    sb = (df[df["success"].astype(bool)]
          .groupby(["K_0", "n", "w_true", "k_est"], sort=True)
          .agg(criterion_mean=("criterion_score", "mean"),
               criterion_sd=("criterion_score", "std"),
               criterion_median=("criterion_score", "median"),
               fit_only_mean=("fit_only_score", "mean"),
               direct_prior_mean=("direct_prior_score", "mean"),
               penalty_mean=("penalty_term", "mean"),
               q_strict_mean=("q_strict", "mean"),
               mean_log_pZ=("mean_log_pZ", "mean"),
               mean_log_pX=("mean_log_pX_given_Z", "mean"),
               mean_log_pY=("mean_log_pY_given_Z", "mean"),
               n_fits=("criterion_score", "size"))
          .reset_index())
    sb.to_csv(RES / "aggregate_score_by_k.csv", index=False)

    # =================================================================
    # penalty_vs_fitgain_by_n.csv  -- mechanism of selection
    #
    # The criterion decomposes EXACTLY as
    #     C_Q(K) = D_K + P_Z(K) + P_theta(K)
    # and (because scale_Z fixes mean(Z^2)=1 over the whole sample stack)
    #     P_Z(K) = n*K*(log 2*pi + 1)      -- deterministic, carries no data
    # so the per-dimension marginal balance at the truth is what decides
    # whether the criterion stops at K_0 or keeps going.
    # =================================================================
    mech = []
    for (k0, n, w), s in sb.groupby(["K_0", "n", "w_true"], sort=True):
        s = s.sort_values("k_est").set_index("k_est")
        if k0 not in s.index or (k0 + 1) not in s.index:
            continue
        d_fit = float(s.loc[k0 + 1, "fit_only_mean"] - s.loc[k0, "fit_only_mean"])
        d_prior = float(s.loc[k0 + 1, "direct_prior_mean"] - s.loc[k0, "direct_prior_mean"])
        d_pen = float(s.loc[k0 + 1, "penalty_mean"] - s.loc[k0, "penalty_mean"])
        mech.append({
            "K_0": k0, "n": n, "w": w,
            "delta_fit_K0_to_K0p1": d_fit,
            "delta_direct_prior": d_prior,
            "delta_explicit_penalty": d_pen,
            "predicted_direct_prior_slope_n_log2pi_plus_1": n * (np.log(2 * np.pi) + 1.0),
            "total_delta_C_Q": d_fit + d_prior + d_pen,
            "stops_at_K0": (d_fit + d_prior + d_pen) > 0,
            "implicit_over_explicit_ratio": d_prior / d_pen if d_pen else np.nan,
            "fit_gain_over_implicit_penalty": -d_fit / d_prior if d_prior else np.nan,
        })
    mech_df = pd.DataFrame(mech)
    mech_df.to_csv(RES / "penalty_vs_fitgain_by_n.csv", index=False)

    # =================================================================
    # historical descriptive comparison (NOT DIRECTLY COMPARABLE)
    # =================================================================
    hist_p = _ROOT / "expfam" / "results" / "fixed_official" / "exp1_k9" / "fixed_exp1_bic_k1to9_bestk_by_trial.csv"
    if hist_p.exists():
        h = pd.read_csv(hist_p)
        hc = h[h["scenario"] == "C"]
        hcounts = hc["best_k_by_trial_bic"].value_counts().to_dict()
        cc = sel_df[(sel_df["K_0"] == 3) & (sel_df["n"] == 150) & (sel_df["w_true"] == 0.5)]
        ccounts = cc["selected_k"].value_counts().to_dict()
        rows = []
        for tag, gen, zpost, ncnt, counts, src in [
            ("historical_fixed_scenario_C", "data_generator_expfam.generate_dual_data",
             "column sample z-score of Z (KI-021 G1)", len(hc), hcounts, str(hist_p.relative_to(_ROOT))),
            ("canonical_gaussian_y_v1", "data_generator_canonical_gaussian_y_k",
             "NONE", len(cc), ccounts, "expfam/results/canonical_gaussian_y_k_selection_20260907/selection_by_trial.csv"),
        ]:
            r = {"arm": tag, "generator": gen, "z_postprocessing": zpost,
                 "K_GEN": 3, "n": 150, "d": 15, "w": 0.5, "sigma_y": 0.1, "w0": 0.0,
                 "family_x": "bernoulli", "family_y": "gaussian",
                 "num_trials": ncnt, "source": src,
                 "NOT_DIRECTLY_COMPARABLE": (
                     "historical arm reports K_hat == K_GEN under a generator whose Z is column "
                     "z-scored, so K_0 is NOT established there (K_0 does not exist under the "
                     "canonical M_K; see fixed_generator_kgen_k0_structural_audit_20260907 sec 12.3). "
                     "The canonical arm reports K_hat vs a theoretically guaranteed K_0 = K_GEN.")}
            for k in CANDIDATE_K:
                r[f"selected_k{k}_count"] = int(counts.get(k, 0))
            rows.append(r)
        pd.DataFrame(rows).to_csv(RES / "historical_vs_canonical_descriptive.csv", index=False)

    # =================================================================
    # PLOTS
    # =================================================================
    core_agg = agg_df[agg_df["w"] == 0.5]

    # 1. recovery rate by K_0 (n=150 core arm)
    a = core_agg[core_agg["n"] == 150].sort_values("K_0")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(a["K_0"].astype(str), a["recovery_rate"], color=C_TRUE, width=0.55)
    for _, r in a.iterrows():
        ax.text(str(int(r["K_0"])), r["recovery_rate"] + 0.02,
                f"{int(r['recovery_count'])}/{int(r['num_trials'])}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_xlabel("$K_0$ (= $K_{GEN}$, theoretically identified)")
    ax.set_ylabel("recovery rate  ($\\hat{K} = K_0$)")
    ax.set_title("Q-based criterion: recovery of the theoretically\nidentified $K_0$  (canonical Gaussian-Y, $n=150$, 10 trials)")
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(FIG / "recovery_rate_by_K0.png", dpi=180); plt.close(fig)

    # 2. selected K distribution
    fig, ax = plt.subplots(figsize=(7.5, 4))
    width = 0.26
    for i, k0 in enumerate(sorted(core_agg["K_0"].unique())):
        r = core_agg[(core_agg["K_0"] == k0) & (core_agg["n"] == 150)]
        if not len(r):
            continue
        vals = [int(r[f"selection_counts_k{k}"].iloc[0]) for k in CANDIDATE_K]
        ax.bar(np.array(CANDIDATE_K) + (i - 1) * width, vals, width=width,
               label=f"$K_0={k0}$")
    ax.set_xticks(CANDIDATE_K); ax.set_xlabel("selected $\\hat{K}$")
    ax.set_ylabel("number of trials"); ax.legend()
    ax.set_title("Distribution of selected $\\hat{K}$  (canonical Gaussian-Y, $n=150$)")
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(FIG / "selected_K_distribution.png", dpi=180); plt.close(fig)

    # 3. score curve K0=3, n=150
    s = sb[(sb["K_0"] == 3) & (sb["n"] == 150) & (sb["w_true"] == 0.5)].sort_values("k_est")
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.errorbar(s["k_est"], s["criterion_mean"], yerr=s["criterion_sd"],
                color=C_TRUE, marker="o", capsize=4, lw=1.8)
    ax.axvline(3, color="black", ls="--", lw=1.0, alpha=0.7, label="$K_0 = 3$")
    ax.set_xticks(CANDIDATE_K); ax.set_xlabel("candidate $K$")
    ax.set_ylabel("Q-based criterion  $C_Q(K)$  (lower = selected)")
    ax.set_title("Q-based criterion vs candidate $K$\n(canonical Gaussian-Y, $K_0=3$, $n=150$, 10 trials, $\\pm$1 sd)")
    ax.legend(); ax.grid(ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(FIG / "score_curve_K0_3_n150.png", dpi=180); plt.close(fig)

    # 4. score decomposition
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(s["k_est"], s["fit_only_mean"], marker="o", color=C_ALT, label="$D_K$ fit-only $-2(\\ln p(X|Z)+\\ln p(Y|Z))$")
    axes[0].plot(s["k_est"], s["direct_prior_mean"], marker="s", color=C_PRIOR, label="$P_Z(K)$ direct prior $-2\\ln p(Z)$")
    axes[0].plot(s["k_est"], s["penalty_mean"], marker="^", color=C_PEN, label="$P_\\theta(K)$ explicit penalty")
    axes[0].set_xticks(CANDIDATE_K); axes[0].set_xlabel("candidate $K$"); axes[0].set_ylabel("contribution")
    axes[0].set_title("Absolute contributions"); axes[0].legend(fontsize=8); axes[0].grid(ls="--", alpha=0.4)
    base = s[s["k_est"] == 1]
    for col, c, m, lab in [("fit_only_mean", C_ALT, "o", "$D_K$ fit-only"),
                           ("direct_prior_mean", C_PRIOR, "s", "$P_Z(K)$ direct prior"),
                           ("penalty_mean", C_PEN, "^", "$P_\\theta(K)$ penalty")]:
        axes[1].plot(s["k_est"], s[col].values - float(base[col].iloc[0]), marker=m, color=c, label=lab)
    axes[1].plot(s["k_est"], s["criterion_mean"].values - float(base["criterion_mean"].iloc[0]),
                 marker="D", color=C_TRUE, lw=2.2, label="$C_Q(K)$ total")
    axes[1].axvline(3, color="black", ls="--", lw=1.0, alpha=0.7)
    axes[1].set_xticks(CANDIDATE_K); axes[1].set_xlabel("candidate $K$")
    axes[1].set_ylabel("change relative to $K=1$")
    axes[1].set_title("Which term moves the selection?"); axes[1].legend(fontsize=8); axes[1].grid(ls="--", alpha=0.4)
    fig.suptitle("Q-based criterion decomposition  $C_Q(K) = D_K + P_Z(K) + P_\\theta(K)$   ($K_0=3$, $n=150$)")
    fig.tight_layout(); fig.savefig(FIG / "score_decomposition_K0_3_n150.png", dpi=180); plt.close(fig)

    # 5. sample size sweep for K0=3
    ns = core_agg[core_agg["K_0"] == 3].sort_values("n")
    if len(ns) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ns["n"], ns["recovery_rate"], marker="o", color=C_TRUE, lw=2, label="recovery rate")
        ax2 = ax.twinx()
        ax2.plot(ns["n"], ns["selected_k_mean"], marker="s", color=C_ALT, ls="--", label="mean $\\hat{K}$")
        ax2.axhline(3, color="black", ls=":", lw=1.0)
        ax.set_ylim(-0.05, 1.05); ax.set_xlabel("$n$")
        ax.set_ylabel("recovery rate", color=C_TRUE); ax2.set_ylabel("mean selected $\\hat{K}$", color=C_ALT)
        ax.set_title("Sample-size sweep  ($K_0 = 3$, canonical Gaussian-Y, 10 trials)")
        ax.grid(ls="--", alpha=0.4)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
        fig.tight_layout(); fig.savefig(FIG / "sample_size_recovery_K0_3.png", dpi=180); plt.close(fig)

    # 6. signal sensitivity
    ws = agg_df[(agg_df["K_0"] == 3) & (agg_df["n"] == 150)].sort_values("w")
    if ws["w"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ws["w"], ws["recovery_rate"], marker="o", color=C_TRUE, lw=2, label="recovery rate")
        ax2 = ax.twinx()
        ax2.plot(ws["w"], ws["selected_k_mean"], marker="s", color=C_ALT, ls="--", label="mean $\\hat{K}$")
        ax2.axhline(3, color="black", ls=":", lw=1.0)
        ax.set_ylim(-0.05, 1.05); ax.set_xlabel("$w$ (Y coupling; $K_0$ identified for every $w \\neq 0$)")
        ax.set_ylabel("recovery rate", color=C_TRUE); ax2.set_ylabel("mean selected $\\hat{K}$", color=C_ALT)
        ax.set_title("Signal sensitivity  ($K_0=3$, $n=150$, 10 trials)")
        ax.grid(ls="--", alpha=0.4)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
        fig.tight_layout(); fig.savefig(FIG / "signal_sensitivity_K0_3.png", dpi=180); plt.close(fig)

    # =================================================================
    # console summary
    # =================================================================
    print("=== aggregate_recovery ===")
    cols = ["K_0", "n", "w", "num_trials", "recovery_count", "recovery_rate",
            "selected_k_mean", "selected_k_mode"]
    print(agg_df[cols].to_string(index=False))
    print("\n=== confusion (core w=0.5) ===")
    print(cm.to_string())
    print("\n=== counterfactual (direct-prior removed) ===")
    cfd = pd.DataFrame(cf_rows)
    print(cfd.groupby(["K_0", "n", "w_true"])[["changed"]].agg(["sum", "size"]).to_string())
    cfsel = cfd.groupby(["K_0", "n", "w_true"])["selected_k_without_direct_prior"].apply(
        lambda x: dict(sorted(x.value_counts().items())))
    print(cfsel.to_string())
    print("\n=== mechanism: marginal balance at K_0 -> K_0+1 ===")
    if len(mech_df):
        print(mech_df[["K_0", "n", "w", "delta_fit_K0_to_K0p1", "delta_direct_prior",
                       "delta_explicit_penalty", "total_delta_C_Q", "stops_at_K0",
                       "implicit_over_explicit_ratio"]].round(2).to_string(index=False))
    print("\nwrote CSVs to", RES)
    print("wrote figures to", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
