"""
Aggregation, decomposition, diagnostics and figures for the per-column
mixed-X K-selection experiment (2026-09-07).

Reads only the experiment's own artifacts. No model is refit here.
Produces every CSV required by the experiment specification plus the figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
RES = _ROOT / "expfam" / "results" / "percolumn_mixed_x_k_selection_20260907"
FIG = _ROOT / "figures" / "percolumn_mixed_x_k_selection_20260907"
CAND_K = [1, 2, 3, 4, 5, 6, 7, 8, 9]
ORDER = ["H-B", "H-G", "H-P", "M-BG", "M-BP", "M-GP", "M-BGP"]
COL = {"homogeneous": "#2196F3", "mixed": "#FF5722"}


def select(sub, col):
    o = sub[sub["success"].astype(bool) & np.isfinite(sub[col])]
    if not len(o):
        return None, np.nan, None, np.nan, []
    mn = o[col].min()
    ties = sorted(o[o[col] == mn]["k_est"].astype(int).tolist())
    sel = min(ties)
    rest = o[o["k_est"] != sel].sort_values(col)
    if len(rest):
        return sel, float(mn), int(rest.iloc[0]["k_est"]), float(rest.iloc[0][col]), ties
    return sel, float(mn), None, np.nan, ties


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES / "fit_results.csv")
    kind = df.groupby("config_id")["kind"].first().to_dict()

    # ---------------- selection_by_trial + counterfactual ----------------
    sel_rows, cf_rows, tie_rows = [], [], []
    for (cfg, tr), sub in df.groupby(["config_id", "trial"], sort=True):
        k0 = int(sub["K_0"].iloc[0])
        s, ss, k2, s2, ties = select(sub, "criterion_score")
        tk = sub[sub["k_est"] == k0]
        sel_rows.append({
            "config_id": cfg, "kind": kind[cfg], "trial": tr,
            "data_seed": int(sub["data_seed"].iloc[0]),
            "n_bernoulli": int(sub["n_bernoulli_x_cols"].iloc[0]),
            "n_gaussian": int(sub["n_gaussian_x_cols"].iloc[0]),
            "n_poisson": int(sub["n_poisson_x_cols"].iloc[0]),
            "K_0": k0, "selected_k": s, "correct": s == k0,
            "selected_score": ss,
            "true_k_score": float(tk["criterion_score"].iloc[0]) if len(tk) else np.nan,
            "second_best_k": k2, "second_best_score": s2,
            "score_gap_to_second": (s2 - ss) if np.isfinite(s2) else np.nan,
            "all_candidate_fits_success": bool(sub["success"].astype(bool).all()),
            "num_failed_candidates": int((~sub["success"].astype(bool)).sum()),
            "n_tied_at_min": len(ties),
            "w_hat_at_K0": float(tk["w_hat"].iloc[0]) if len(tk) else np.nan,
            "sigma_y_hat_at_K0": float(tk["sigma_y_hat"].iloc[0]) if len(tk) else np.nan,
        })
        if len(ties) > 1:
            tie_rows.append({"config_id": cfg, "trial": tr,
                             "tied_k": ",".join(map(str, ties)),
                             "score": ss, "resolved_to": s})
        c, cs, c2, cs2, ct = select(sub, "score_without_direct_prior")
        cf_rows.append({"config_id": cfg, "kind": kind[cfg], "trial": tr, "K_0": k0,
                        "selected_k_current_criterion": s,
                        "selected_k_without_direct_prior": c,
                        "changed": s != c, "cf_selected_score": cs,
                        "cf_second_best_k": c2, "n_tied_at_min": len(ct)})
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(RES / "selection_by_trial.csv", index=False)
    pd.DataFrame(tie_rows).to_csv(RES / "ties.csv", index=False)
    cf = pd.DataFrame(cf_rows)
    cf.to_csv(RES / "diagnostic_counterfactual_selection.csv", index=False)

    # ---------------- aggregate_recovery ----------------
    agg = []
    for cfg, s in sel.groupby("config_id", sort=True):
        cnt = s["selected_k"].value_counts().to_dict()
        cor, wrg = s[s["correct"]], s[~s["correct"]]
        r = {"config_id": cfg, "kind": kind[cfg],
             "n_bernoulli": int(s["n_bernoulli"].iloc[0]),
             "n_gaussian": int(s["n_gaussian"].iloc[0]),
             "n_poisson": int(s["n_poisson"].iloc[0]),
             "num_trials": len(s),
             "num_complete_trials": int(s["all_candidate_fits_success"].sum()),
             "recovery_count": int(s["correct"].sum()),
             "recovery_rate": float(s["correct"].mean()),
             "selected_k_mean": float(s["selected_k"].mean()),
             "selected_k_median": float(s["selected_k"].median()),
             "selected_k_mode": int(s["selected_k"].mode().iloc[0]),
             "mean_gap_when_correct": float(cor["score_gap_to_second"].mean()) if len(cor) else np.nan,
             "mean_gap_when_wrong": float(wrg["score_gap_to_second"].mean()) if len(wrg) else np.nan,
             "mean_w_hat_at_K0": float(s["w_hat_at_K0"].mean()),
             "mean_sigma_y_hat_at_K0": float(s["sigma_y_hat_at_K0"].mean())}
        for k in CAND_K:
            r[f"selection_counts_k{k}"] = int(cnt.get(k, 0))
        agg.append(r)
    agg = pd.DataFrame(agg).set_index("config_id").loc[ORDER].reset_index()
    agg.to_csv(RES / "aggregate_recovery.csv", index=False)

    # ---------------- selected_k_distribution ----------------
    dist = []
    for cfg in ORDER:
        s = sel[sel.config_id == cfg]
        for k in CAND_K:
            dist.append({"config_id": cfg, "kind": kind[cfg], "k": k,
                         "count": int((s["selected_k"] == k).sum()),
                         "total": len(s),
                         "rate": float((s["selected_k"] == k).mean())})
    pd.DataFrame(dist).to_csv(RES / "selected_k_distribution.csv", index=False)

    # ---------------- score_by_k ----------------
    sb = (df[df["success"].astype(bool)]
          .groupby(["config_id", "kind", "k_est"], sort=True)
          .agg(criterion_mean=("criterion_score", "mean"),
               criterion_sd=("criterion_score", "std"),
               criterion_median=("criterion_score", "median"),
               fit_only_mean=("fit_only_score", "mean"),
               direct_prior_mean=("direct_prior_score", "mean"),
               penalty_mean=("penalty_term", "mean"),
               q_strict_mean=("q_strict", "mean"),
               mean_log_pX=("mean_log_pX_given_Z", "mean"),
               mean_log_pY=("mean_log_pY_given_Z", "mean"),
               mean_log_pZ=("mean_log_pZ", "mean"),
               n_fits=("criterion_score", "size")).reset_index())
    sb.to_csv(RES / "score_by_k.csv", index=False)

    # ---------------- score_decomposition (K_0 -> K_0+1 margin) ----------------
    dec = []
    for cfg in ORDER:
        s = sb[sb.config_id == cfg].sort_values("k_est").set_index("k_est")
        k0 = 3
        if k0 + 1 not in s.index:
            continue
        dfit = float(s.loc[k0 + 1, "fit_only_mean"] - s.loc[k0, "fit_only_mean"])
        dpri = float(s.loc[k0 + 1, "direct_prior_mean"] - s.loc[k0, "direct_prior_mean"])
        dpen = float(s.loc[k0 + 1, "penalty_mean"] - s.loc[k0, "penalty_mean"])
        dx = float(-2 * (s.loc[k0 + 1, "mean_log_pX"] - s.loc[k0, "mean_log_pX"]))
        dy = float(-2 * (s.loc[k0 + 1, "mean_log_pY"] - s.loc[k0, "mean_log_pY"]))
        dec.append({"config_id": cfg, "kind": kind[cfg], "K_0": k0,
                    "delta_fit_total": dfit,
                    "delta_fit_X_part": dx, "delta_fit_Y_part": dy,
                    "delta_direct_prior": dpri, "delta_explicit_penalty": dpen,
                    "total_delta_C_Q": dfit + dpri + dpen,
                    "stops_at_K0": (dfit + dpri + dpen) > 0,
                    "implicit_over_explicit": dpri / dpen if dpen else np.nan,
                    "n_gaussian": int(agg[agg.config_id == cfg]["n_gaussian"].iloc[0])})
    dec = pd.DataFrame(dec)
    dec.to_csv(RES / "score_decomposition.csv", index=False)

    # ---------------- mixed vs homogeneous ----------------
    mv = []
    for kd in ("homogeneous", "mixed"):
        a = agg[agg.kind == kd]
        d2 = dec[dec.kind == kd]
        mv.append({"kind": kd, "n_configs": len(a),
                   "total_trials": int(a["num_trials"].sum()),
                   "total_recovered": int(a["recovery_count"].sum()),
                   "recovery_rate": float(a["recovery_count"].sum() / a["num_trials"].sum()),
                   "mean_gap_to_second": float(a["mean_gap_when_correct"].mean()),
                   "mean_delta_fit": float(d2["delta_fit_total"].mean()),
                   "mean_delta_fit_X": float(d2["delta_fit_X_part"].mean()),
                   "mean_delta_fit_Y": float(d2["delta_fit_Y_part"].mean()),
                   "mean_delta_prior": float(d2["delta_direct_prior"].mean()),
                   "mean_delta_penalty": float(d2["delta_explicit_penalty"].mean()),
                   "mean_total_margin": float(d2["total_delta_C_Q"].mean())})
    pd.DataFrame(mv).to_csv(RES / "mixed_vs_homogeneous_comparison.csv", index=False)

    # ---------------- diagnostics passthrough ----------------
    if (RES / "diagnostics.csv").exists():
        pass  # written by the runner

    # =================== FIGURES ===================
    # 1. recovery by family config
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cols = [COL[kind[c]] for c in ORDER]
    ax.bar(ORDER, agg["recovery_rate"], color=cols, width=0.6)
    for i, r in enumerate(agg.itertuples()):
        ax.text(i, r.recovery_rate + 0.02, f"{r.recovery_count}/{r.num_trials}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel("recovery rate  ($\\hat{K}=K_0$)")
    ax.set_xlabel("X family configuration")
    ax.set_title("K$_0$ recovery by per-column X family configuration\n"
                 "(canonical Gaussian-Y fixes $K_0=K_{GEN}=3$; $n=150$, $d=15$, 10 trials)")
    ax.grid(axis="y", ls="--", alpha=0.4)
    h = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("homogeneous", "mixed")]
    ax.legend(h, ["homogeneous control", "mixed"], loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "recovery_by_family_config.png", dpi=180); plt.close(fig)

    # 2. selected K distribution
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.11
    for i, cfg in enumerate(ORDER):
        s = sel[sel.config_id == cfg]
        vals = [int((s["selected_k"] == k).sum()) for k in CAND_K]
        ax.bar(np.array(CAND_K) + (i - 3) * w, vals, width=w, label=cfg,
               color=COL[kind[cfg]], alpha=0.55 + 0.06 * i,
               edgecolor="black", linewidth=0.4)
    ax.axvline(3, color="black", ls="--", lw=1.0, alpha=0.7)
    ax.set_xticks(CAND_K); ax.set_xlabel("selected $\\hat{K}$")
    ax.set_ylabel("number of trials"); ax.legend(ncol=4, fontsize=8)
    ax.set_title("Distribution of selected $\\hat{K}$ by X family configuration ($K_0=3$)")
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(FIG / "selected_K_by_family_config.png", dpi=180); plt.close(fig)

    # 3. score curves H-B vs M-BGP (+ all configs, centred at their own K=3)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for cfg in ("H-B", "M-BGP"):
        s = sb[sb.config_id == cfg].sort_values("k_est")
        axes[0].errorbar(s["k_est"], s["criterion_mean"], yerr=s["criterion_sd"],
                         marker="o", capsize=3, lw=1.7, label=cfg,
                         color=COL[kind[cfg]])
    axes[0].axvline(3, color="black", ls="--", lw=1.0, alpha=0.7, label="$K_0=3$")
    axes[0].set_xticks(CAND_K); axes[0].set_xlabel("candidate $K$")
    axes[0].set_ylabel("$C_Q(K)$ (lower = selected)")
    axes[0].set_title("H-B vs M-BGP (absolute)"); axes[0].legend(); axes[0].grid(ls="--", alpha=0.4)
    for cfg in ORDER:
        s = sb[sb.config_id == cfg].sort_values("k_est")
        base = float(s[s.k_est == 3]["criterion_mean"].iloc[0])
        axes[1].plot(s["k_est"], s["criterion_mean"] - base, marker="o",
                     label=cfg, color=COL[kind[cfg]],
                     ls="-" if kind[cfg] == "mixed" else "--", alpha=0.85)
    axes[1].axvline(3, color="black", ls="--", lw=1.0, alpha=0.7)
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_xticks(CAND_K); axes[1].set_xlabel("candidate $K$")
    axes[1].set_ylabel("$C_Q(K) - C_Q(3)$")
    axes[1].set_title("all configs, centred at $K_0=3$")
    axes[1].legend(ncol=2, fontsize=8); axes[1].grid(ls="--", alpha=0.4)
    fig.suptitle("Q-based criterion vs candidate $K$ under per-column X families")
    fig.tight_layout(); fig.savefig(FIG / "score_curve_by_family_config.png", dpi=180); plt.close(fig)

    # 4. decomposition for M-BGP
    s = sb[sb.config_id == "M-BGP"].sort_values("k_est")
    b = s[s.k_est == 1]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for col, c, m, lab in (("fit_only_mean", "#FF5722", "o", "$D_K$ fit ($X$ and $Y$)"),
                           ("direct_prior_mean", "#9C27B0", "s", "$P_Z(K)$ latent prior"),
                           ("penalty_mean", "#4CAF50", "^", "$P_\\theta(K)$ parameter penalty")):
        ax.plot(s["k_est"], s[col] - float(b[col].iloc[0]), marker=m, color=c, label=lab)
    ax.plot(s["k_est"], s["criterion_mean"] - float(b["criterion_mean"].iloc[0]),
            marker="D", color="#2196F3", lw=2.2, label="$C_Q(K)$ total")
    ax.axvline(3, color="black", ls="--", lw=1.0, alpha=0.7, label="$K_0=3$")
    ax.set_xticks(CAND_K); ax.set_xlabel("candidate $K$")
    ax.set_ylabel("change relative to $K=1$")
    ax.set_title("Criterion decomposition, mixed Bernoulli/Gaussian/Poisson X\n"
                 "M-BGP (5/5/5), $C_Q(K)=D_K+P_Z(K)+P_\\theta(K)$")
    ax.legend(fontsize=9); ax.grid(ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(FIG / "score_decomposition_mixed_BGP.png", dpi=180); plt.close(fig)

    # 5. prior counterfactual
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(ORDER))
    cur = [float(sel[sel.config_id == c]["selected_k"].mean()) for c in ORDER]
    nop = [float(cf[cf.config_id == c]["selected_k_without_direct_prior"].mean()) for c in ORDER]
    ax.bar(x - 0.2, cur, width=0.4, color="#2196F3", label="current criterion")
    ax.bar(x + 0.2, nop, width=0.4, color="#FFC107",
           label="diagnostic: $P_Z$ removed")
    ax.axhline(3, color="black", ls="--", lw=1.0, alpha=0.8, label="$K_0=3$")
    ax.set_xticks(x); ax.set_xticklabels(ORDER)
    ax.set_ylabel("mean selected $\\hat{K}$"); ax.set_xlabel("X family configuration")
    ax.set_title("Latent-prior counterfactual (DIAGNOSTIC ONLY, not a proposed criterion)")
    ax.legend(fontsize=9); ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(FIG / "prior_counterfactual_by_family_config.png", dpi=180); plt.close(fig)

    # =================== console ===================
    print("=== aggregate_recovery ===")
    print(agg[["config_id", "kind", "n_bernoulli", "n_gaussian", "n_poisson",
               "num_trials", "recovery_count", "recovery_rate", "selected_k_mean",
               "mean_gap_when_correct"]].round(3).to_string(index=False))
    print("\n=== score_decomposition (K_0 -> K_0+1) ===")
    print(dec[["config_id", "kind", "delta_fit_total", "delta_fit_X_part",
               "delta_fit_Y_part", "delta_direct_prior", "delta_explicit_penalty",
               "total_delta_C_Q", "implicit_over_explicit"]].round(2).to_string(index=False))
    print("\n=== mixed vs homogeneous ===")
    print(pd.read_csv(RES / "mixed_vs_homogeneous_comparison.csv").round(3).to_string(index=False))
    print("\n=== counterfactual ===")
    g = cf.groupby("config_id")["changed"].agg(["sum", "size"])
    print(g.to_string())
    print("counterfactual selected K counts:",
          dict(sorted(cf["selected_k_without_direct_prior"].value_counts().items())))
    print("\nties:", len(tie_rows))
    print("wrote CSVs ->", RES)
    print("wrote figures ->", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
