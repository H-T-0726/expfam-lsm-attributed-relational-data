"""
Shared library for multi-scenario experiments.

Each experiment function accepts (true_fx, true_fy, scenario_tag)
and produces scenario-tagged CSV + figures in results/.

Scenarios:
  A [P-B]: True X=Poisson,  Y=Bernoulli
  B [G-P]: True X=Gaussian, Y=Poisson
  C [B-G]: True X=Bernoulli, Y=Gaussian
"""

import sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

_SRC = Path(__file__).parent
_RES = _SRC.parent / "results"
_RES.mkdir(exist_ok=True)
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent.parent / "reproduction" / "src"))

from data_generator_expfam import generate_dual_data          # noqa
from utils_expfam import (                                     # noqa
    run_em_dual, calc_Q_dual_strict, calc_bic_dual,
    procrustes_rotation, calc_rmse,
)

warnings.filterwarnings("ignore")

# ── Global constants ──────────────────────────────────────────────────
FAMILIES    = ["gaussian", "bernoulli", "poisson"]
FAM_LABEL   = {"gaussian": "Gaussian", "bernoulli": "Bernoulli", "poisson": "Poisson"}
FAM_SHORT   = {"gaussian": "Gauss", "bernoulli": "Bern", "poisson": "Pois"}

N, D, K_TRUE = 150, 15, 3
N_TRIALS     = 10
L, NITER     = 5, 8
K_LIST       = [1, 2, 3, 4, 5, 6]
N_LIST       = [50, 100, 150, 200, 250, 300]
D_LIST       = [5, 10, 15, 20, 25, 30]

METRICS = ["rmse_Z", "rmse_F", "rmse_sigma", "rmse_Y", "rmse_X", "w0_err", "w_err"]
METRIC_LABELS = {
    "rmse_Z":     "RMSE(Z)", "rmse_F":    "RMSE(F)",
    "rmse_sigma": "RMSE(Sigma)", "rmse_Y":  "RMSE(mu_Y)",
    "rmse_X":     "RMSE(mu_X)", "w0_err": "|w0 err|", "w_err": "|w err|",
}
METRIC_COLORS = {
    "rmse_Z": "#2878b5", "rmse_F": "#f39c12", "rmse_sigma": "#27ae60",
    "rmse_Y": "#e06c75", "rmse_X": "#9b59b6", "w0_err": "#1abc9c", "w_err": "#e67e22",
}
METRIC_MARKERS = {
    "rmse_Z": "o", "rmse_F": "s", "rmse_sigma": "^",
    "rmse_Y": "D", "rmse_X": "v", "w0_err": "p", "w_err": "h",
}


# ─────────────────────────────────────────────────────────────────────
# Exp 1: k variation
# ─────────────────────────────────────────────────────────────────────

def run_exp1(true_fx: str, true_fy: str, scenario_tag: str,
             base_seed: int = 4000) -> pd.DataFrame:
    """BIC + RMSE(Z) vs k. Returns DataFrame."""
    records = []
    total = len(K_LIST) * N_TRIALS
    done  = 0
    for k_est in K_LIST:
        for trial in range(N_TRIALS):
            done += 1
            dseed = base_seed + trial * 100
            mseed = dseed + 50
            data = generate_dual_data(n=N, d=D, k=K_TRUE, seed=dseed,
                                      family_x=true_fx, family_y=true_fy)
            res = run_em_dual(
                X=data["X"], Y=data["Y"], true_params=data,
                family_x=true_fx, family_y=true_fy,
                k=k_est, L=L, num_iter=NITER, seed=mseed,
                compute_strict_Q=True,
            )
            bic, npar = calc_bic_dual(
                Q_strict=res["Q_strict"], k=k_est, n=N, d=D,
                family_x=true_fx, family_y=true_fy,
            )
            records.append({
                "scenario": scenario_tag, "k_est": k_est, "trial": trial,
                "rmse_Z": res["rmse_Z"], "BIC": bic,
                "Q_strict": res["Q_strict"],
            })
            print(f"  [{done:3d}/{total}] Exp1 k={k_est} t={trial}"
                  f"  RMSE(Z)={res['rmse_Z']:.4f}  BIC={bic:.1f}")
    return pd.DataFrame(records)


def plot_exp1(df: pd.DataFrame, true_fx: str, true_fy: str,
              scenario_tag: str) -> None:
    agg = df.groupby("k_est").agg(
        rmse_Z_mean=("rmse_Z", "mean"), rmse_Z_std=("rmse_Z", "std"),
        BIC_mean=("BIC", "mean"),       BIC_std=("BIC", "std"),
    ).reset_index()
    ks = agg["k_est"].values
    bic_best_k = int(agg.loc[agg["BIC_mean"].idxmin(), "k_est"])

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    fig.suptitle(
        f"Scenario {scenario_tag}: Dimension Identification\n"
        f"(True X={FAM_LABEL[true_fx]}, True Y={FAM_LABEL[true_fy]}, "
        f"n={N}, d={D}, k*={K_TRUE})",
        fontsize=12, fontweight="bold",
    )
    ax0 = axes[0]
    ax0.errorbar(ks, agg["rmse_Z_mean"], yerr=agg["rmse_Z_std"],
                 marker="o", color="#2878b5", linewidth=2, markersize=7,
                 capsize=4, elinewidth=1.5, label="RMSE(Z)")
    ax0.axvline(K_TRUE, color="#e06c75", ls="--", lw=1.5, label=f"k*={K_TRUE}")
    ax0.set_xlabel("Estimated k", fontsize=12)
    ax0.set_ylabel("RMSE(Z)", fontsize=12)
    ax0.set_xticks(K_LIST)
    ax0.legend(fontsize=10); ax0.grid(True, alpha=0.4)
    ax0.set_title("RMSE(Z) vs k  [L-shape]", fontsize=11)

    ax1 = axes[1]
    ax1.errorbar(ks, agg["BIC_mean"], yerr=agg["BIC_std"],
                 marker="s", color="#f39c12", linewidth=2, markersize=7,
                 capsize=4, elinewidth=1.5, label="BIC")
    ax1.axvline(K_TRUE, color="#e06c75", ls="--", lw=1.5, label=f"k*={K_TRUE}")
    bic_best_v = float(agg.loc[agg["BIC_mean"].idxmin(), "BIC_mean"])
    ax1.annotate(f"k={bic_best_k}",
                 xy=(bic_best_k, bic_best_v),
                 xytext=(bic_best_k + 0.4, bic_best_v + abs(bic_best_v)*0.02),
                 fontsize=10, color="#e06c75",
                 arrowprops=dict(arrowstyle="->", color="#e06c75"))
    ax1.set_xlabel("Estimated k", fontsize=12)
    ax1.set_ylabel("BIC", fontsize=12)
    ax1.set_xticks(K_LIST)
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.4)
    ax1.set_title("BIC vs k  [V-shape: minimum at k*]", fontsize=11)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(_RES / f"fig_scenario_{scenario_tag}_exp1_k.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Exp1] BIC best k={bic_best_k}  (pass={bic_best_k==K_TRUE})")


# ─────────────────────────────────────────────────────────────────────
# Exp 2 & 3: n and d variation
# ─────────────────────────────────────────────────────────────────────

def run_exp23(true_fx: str, true_fy: str, scenario_tag: str,
              base_seed: int = 5000) -> tuple:
    """Returns (df_n, df_d)."""
    def one_run(n, d, trial):
        dseed = base_seed + trial * 1000 + n + d
        mseed = dseed + 500
        data = generate_dual_data(n=n, d=d, k=K_TRUE, seed=dseed,
                                  family_x=true_fx, family_y=true_fy)
        res = run_em_dual(
            X=data["X"], Y=data["Y"], true_params=data,
            family_x=true_fx, family_y=true_fy,
            k=K_TRUE, L=L, num_iter=NITER, seed=mseed,
        )
        return {m: res[m] for m in METRICS}

    # Exp 2: n variation
    records_n = []
    total = len(N_LIST) * N_TRIALS; done = 0
    for n_val in N_LIST:
        for trial in range(N_TRIALS):
            done += 1
            rec = one_run(n_val, D, trial)
            rec["n"] = n_val; rec["trial"] = trial
            records_n.append(rec)
            print(f"  [{done:3d}/{total}] Exp2 n={n_val} t={trial}"
                  f"  RMSE(Z)={rec['rmse_Z']:.4f}")
    df_n = pd.DataFrame(records_n)

    # Exp 3: d variation
    records_d = []
    total = len(D_LIST) * N_TRIALS; done = 0
    for d_val in D_LIST:
        for trial in range(N_TRIALS):
            done += 1
            rec = one_run(N, d_val, trial)
            rec["d"] = d_val; rec["trial"] = trial
            records_d.append(rec)
            print(f"  [{done:3d}/{total}] Exp3 d={d_val} t={trial}"
                  f"  RMSE(Z)={rec['rmse_Z']:.4f}")
    df_d = pd.DataFrame(records_d)
    return df_n, df_d


def _plot_metrics_panel(df, x_col, x_label, title, out_stem):
    """7-metric error-bar plot saved to out_stem.pdf/.png."""
    agg = df.groupby(x_col).agg(
        **{f"{m}_mean": (m, "mean") for m in METRICS},
        **{f"{m}_std":  (m, "std")  for m in METRICS},
    ).reset_index()
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    for m in ["rmse_Z", "rmse_F", "rmse_sigma", "rmse_Y", "rmse_X"]:
        axes[0].errorbar(agg[x_col], agg[f"{m}_mean"], yerr=agg[f"{m}_std"],
                         marker=METRIC_MARKERS[m], color=METRIC_COLORS[m],
                         linewidth=1.8, markersize=6, capsize=3,
                         label=METRIC_LABELS[m])
    axes[0].set_xlabel(x_label, fontsize=12); axes[0].set_ylabel("RMSE", fontsize=12)
    axes[0].legend(fontsize=10, ncol=2); axes[0].grid(True, alpha=0.4)
    axes[0].set_xticks(agg[x_col].values)
    axes[0].set_title("Structural parameters (Z, F, Sigma, Y, X)", fontsize=11)
    for m in ["w0_err", "w_err"]:
        axes[1].errorbar(agg[x_col], agg[f"{m}_mean"], yerr=agg[f"{m}_std"],
                         marker=METRIC_MARKERS[m], color=METRIC_COLORS[m],
                         linewidth=1.8, markersize=6, capsize=3,
                         label=METRIC_LABELS[m])
    axes[1].set_xlabel(x_label, fontsize=12); axes[1].set_ylabel("Abs. Error", fontsize=12)
    axes[1].legend(fontsize=11); axes[1].grid(True, alpha=0.4)
    axes[1].set_xticks(agg[x_col].values)
    axes[1].set_title("Relational parameters (w0, w)", fontsize=11)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_stem}.png")


def plot_exp23(df_n, df_d, true_fx, true_fy, scenario_tag):
    _plot_metrics_panel(
        df_n, "n", "Sample size n",
        f"Scenario {scenario_tag}: Asymptotic Consistency vs n\n"
        f"(True X={FAM_LABEL[true_fx]}, True Y={FAM_LABEL[true_fy]}, d={D}, k={K_TRUE})",
        str(_RES / f"fig_scenario_{scenario_tag}_exp2_n"),
    )
    _plot_metrics_panel(
        df_d, "d", "Attribute dimension d",
        f"Scenario {scenario_tag}: Scalability vs d\n"
        f"(True X={FAM_LABEL[true_fx]}, True Y={FAM_LABEL[true_fy]}, n={N}, k={K_TRUE})",
        str(_RES / f"fig_scenario_{scenario_tag}_exp3_d"),
    )


# ─────────────────────────────────────────────────────────────────────
# Exp 4: Mismatch matrix + ablation
# ─────────────────────────────────────────────────────────────────────

def run_exp4(true_fx: str, true_fy: str, scenario_tag: str,
             base_seed: int = 6000) -> pd.DataFrame:
    """3x3 mismatch + ablation. Returns DataFrame."""
    records = []

    # Part A: 3x3 grid
    for fx_m in FAMILIES:
        for fy_m in FAMILIES:
            correct = (fx_m == true_fx and fy_m == true_fy)
            for trial in range(N_TRIALS):
                dseed = base_seed + trial * 100
                mseed = dseed + 50
                data = generate_dual_data(n=N, d=D, k=K_TRUE, seed=dseed,
                                          family_x=true_fx, family_y=true_fy)
                res = run_em_dual(
                    X=data["X"], Y=data["Y"], true_params=data,
                    family_x=fx_m, family_y=fy_m,
                    k=K_TRUE, L=L, num_iter=NITER, seed=mseed,
                )
                records.append({
                    "scenario": scenario_tag,
                    "model_fx": fx_m, "model_fy": fy_m,
                    "fix_w": False, "fix_x": False,
                    "condition": f"X={FAM_LABEL[fx_m]}, Y={FAM_LABEL[fy_m]}",
                    "correct": correct, "trial": trial,
                    "rmse_Z": res["rmse_Z"], "rmse_F": res["rmse_F"],
                    "rmse_Y": res["rmse_Y"], "rmse_X": res["rmse_X"],
                })
                print(f"  [Exp4-grid] X={fx_m[:4]:4s} Y={fy_m[:4]:4s} t={trial}"
                      f"  RMSE(Z)={res['rmse_Z']:.4f}"
                      f"{'  [CORRECT]' if correct else ''}")

    # Part B: ablation
    for label, fw, fxa in [("Y-only (fix_x)", False, True),
                            ("X-only (fix_w)", True,  False)]:
        for trial in range(N_TRIALS):
            dseed = base_seed + trial * 100
            mseed = dseed + 50
            data = generate_dual_data(n=N, d=D, k=K_TRUE, seed=dseed,
                                      family_x=true_fx, family_y=true_fy)
            res = run_em_dual(
                X=data["X"], Y=data["Y"], true_params=data,
                family_x=true_fx, family_y=true_fy,
                k=K_TRUE, L=L, num_iter=NITER, seed=mseed,
                fix_w=fw, fix_x=fxa,
            )
            records.append({
                "scenario": scenario_tag,
                "model_fx": true_fx, "model_fy": true_fy,
                "fix_w": fw, "fix_x": fxa,
                "condition": label, "correct": False, "trial": trial,
                "rmse_Z": res["rmse_Z"], "rmse_F": res["rmse_F"],
                "rmse_Y": res["rmse_Y"], "rmse_X": res["rmse_X"],
            })
            print(f"  [Exp4-abl] {label} t={trial}  RMSE(Z)={res['rmse_Z']:.4f}")

    return pd.DataFrame(records)


def plot_exp4(df: pd.DataFrame, true_fx: str, true_fy: str,
              scenario_tag: str) -> dict:
    """Generates heatmap + bar chart. Returns summary dict."""
    agg = df.groupby(["model_fx", "model_fy", "fix_w", "fix_x", "condition"]).agg(
        rmse_Z_mean=("rmse_Z", "mean"),
        rmse_Z_std =("rmse_Z", "std"),
    ).reset_index()

    grid_df   = agg[(~agg["fix_w"]) & (~agg["fix_x"])]
    proposed  = float(grid_df[(grid_df.model_fx == true_fx) &
                               (grid_df.model_fy == true_fy)]["rmse_Z_mean"].iloc[0])
    prop_std  = float(grid_df[(grid_df.model_fx == true_fx) &
                               (grid_df.model_fy == true_fy)]["rmse_Z_std"].iloc[0])

    # ── Figure 4: Heatmap ───────────────────────────────────────────
    heatmap = np.zeros((3, 3))
    for i, fx in enumerate(FAMILIES):
        for j, fy in enumerate(FAMILIES):
            row = grid_df[(grid_df.model_fx == fx) & (grid_df.model_fy == fy)]
            heatmap[i, j] = float(row["rmse_Z_mean"].iloc[0])

    fig4, ax4 = plt.subplots(figsize=(7, 5.5))
    im = ax4.imshow(heatmap, cmap=plt.cm.RdYlGn_r, aspect="auto",
                    vmin=heatmap.min() * 0.9, vmax=heatmap.max() * 1.05)
    plt.colorbar(im, ax=ax4, label="RMSE(Z)")
    ax4.set_xticks(range(3)); ax4.set_yticks(range(3))
    ax4.set_xticklabels([FAM_LABEL[f] for f in FAMILIES], fontsize=11)
    ax4.set_yticklabels([FAM_LABEL[f] for f in FAMILIES], fontsize=11)
    ax4.set_xlabel("Model family for Y", fontsize=12)
    ax4.set_ylabel("Model family for X", fontsize=12)
    ax4.set_title(
        f"Scenario {scenario_tag}: RMSE(Z) Mismatch Heatmap\n"
        f"True data: X={FAM_LABEL[true_fx]}, Y={FAM_LABEL[true_fy]} "
        f"(n={N}, d={D}, k={K_TRUE})",
        fontsize=11, fontweight="bold",
    )
    for i in range(3):
        for j in range(3):
            val = heatmap[i, j]
            is_correct = (FAMILIES[i] == true_fx and FAMILIES[j] == true_fy)
            color = "white" if val > (heatmap.max() * 0.6) else "black"
            text  = f"{val:.3f}"
            if is_correct:
                text += "\n[Proposed]"
            ax4.text(j, i, text, ha="center", va="center", fontsize=10,
                     color=color, fontweight="bold" if is_correct else "normal")
            if is_correct:
                ax4.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                              linewidth=2.5, edgecolor="#2878b5", facecolor="none"))
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig4.savefig(_RES / f"fig_scenario_{scenario_tag}_exp4_heatmap.{ext}",
                     dpi=300, bbox_inches="tight")
    plt.close(fig4)

    # ── Figure 5: Bar chart ─────────────────────────────────────────
    bar_data = [{
        "label":  f"Proposed\n({FAM_SHORT[true_fx]},{FAM_SHORT[true_fy]})",
        "mean":   proposed, "std": prop_std,
        "color":  "#2878b5", "group": "proposed",
    }]
    # X mismatch (Y fixed correct)
    for fx in FAMILIES:
        if fx == true_fx: continue
        row = grid_df[(grid_df.model_fx == fx) & (grid_df.model_fy == true_fy)]
        bar_data.append({
            "label": f"X={FAM_SHORT[fx]}\n(Y={FAM_SHORT[true_fy]})",
            "mean":  float(row["rmse_Z_mean"].iloc[0]),
            "std":   float(row["rmse_Z_std"].iloc[0]),
            "color": "#e06c75", "group": "x_mismatch",
        })
    # Y mismatch (X fixed correct)
    for fy in FAMILIES:
        if fy == true_fy: continue
        row = grid_df[(grid_df.model_fx == true_fx) & (grid_df.model_fy == fy)]
        bar_data.append({
            "label": f"Y={FAM_SHORT[fy]}\n(X={FAM_SHORT[true_fx]})",
            "mean":  float(row["rmse_Z_mean"].iloc[0]),
            "std":   float(row["rmse_Z_std"].iloc[0]),
            "color": "#f39c12", "group": "y_mismatch",
        })
    # Ablation
    abl_df = agg[agg["fix_w"] | agg["fix_x"]]
    for label, fw, fxa in [("No X\n(fix_x)", False, True),
                             ("No Y\n(fix_w)", True,  False)]:
        row = abl_df[(abl_df["fix_w"] == fw) & (abl_df["fix_x"] == fxa)]
        if len(row):
            bar_data.append({
                "label": label,
                "mean":  float(row["rmse_Z_mean"].iloc[0]),
                "std":   float(row["rmse_Z_std"].iloc[0]),
                "color": "#27ae60", "group": "ablation",
            })

    fig5, ax5 = plt.subplots(figsize=(10, 5.5))
    labels = [b["label"] for b in bar_data]
    means  = [b["mean"]  for b in bar_data]
    stds   = [b["std"]   for b in bar_data]
    colors = [b["color"] for b in bar_data]
    x_pos  = np.arange(len(bar_data))
    bars   = ax5.bar(x_pos, means, yerr=stds, capsize=5,
                     color=colors, alpha=0.85, width=0.6,
                     edgecolor="black", linewidth=0.7,
                     error_kw={"elinewidth": 1.5})
    for bar, m, s in zip(bars, means, stds):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + s + 0.004,
                 f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax5.axhline(proposed, color="#2878b5", ls="--", lw=1.5,
                label=f"Proposed={proposed:.3f}")
    ax5.set_xticks(x_pos); ax5.set_xticklabels(labels, fontsize=10)
    ax5.set_ylabel("RMSE(Z)  (lower is better)", fontsize=12)
    ax5.set_title(
        f"Scenario {scenario_tag}: RMSE(Z) Comparison\n"
        f"True: X={FAM_LABEL[true_fx]}, Y={FAM_LABEL[true_fy]}  "
        f"(n={N}, d={D}, k={K_TRUE}, {N_TRIALS} trials)",
        fontsize=11, fontweight="bold",
    )
    ax5.legend(fontsize=10); ax5.grid(True, axis="y", alpha=0.4)
    ax5.set_ylim(0, max(means) * 1.30)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig5.savefig(_RES / f"fig_scenario_{scenario_tag}_exp5_barchart.{ext}",
                     dpi=300, bbox_inches="tight")
    plt.close(fig5)

    # Summary
    print(f"\n  === Scenario {scenario_tag} Mismatch Summary ===")
    print(f"  Proposed ({FAM_SHORT[true_fx]},{FAM_SHORT[true_fy]}): {proposed:.4f}")
    max_ratio = max((b["mean"]/proposed) for b in bar_data[1:])
    worst     = max(bar_data[1:], key=lambda b: b["mean"])
    for b in bar_data[1:]:
        print(f"  {b['label'].replace(chr(10),' '):<25s}: "
              f"{b['mean']:.4f} ({b['mean']/proposed:.2f}x)")
    return {
        "proposed":  proposed,
        "max_ratio": max_ratio,
        "worst_label": worst["label"].replace("\n", " "),
    }


# ─────────────────────────────────────────────────────────────────────
# Master runner for one scenario
# ─────────────────────────────────────────────────────────────────────

def run_scenario(true_fx: str, true_fy: str, scenario_tag: str,
                 seed_offset: int = 0) -> dict:
    """Run all 4 experiments for one scenario. Returns summary dict."""
    t0 = time.perf_counter()
    print(f"\n{'='*65}")
    print(f"SCENARIO {scenario_tag}: True X={FAM_LABEL[true_fx]}, "
          f"True Y={FAM_LABEL[true_fy]}")
    print(f"{'='*65}")

    # Exp 1
    print(f"\n--- Exp 1: k variation ---")
    df1 = run_exp1(true_fx, true_fy, scenario_tag, base_seed=4000+seed_offset)
    df1.to_csv(_RES / f"exp_scenario_{scenario_tag}_exp1_k.csv", index=False)
    plot_exp1(df1, true_fx, true_fy, scenario_tag)

    # Exp 2 & 3
    print(f"\n--- Exp 2&3: n/d variation ---")
    df_n, df_d = run_exp23(true_fx, true_fy, scenario_tag, base_seed=5000+seed_offset)
    df_n.to_csv(_RES / f"exp_scenario_{scenario_tag}_exp2_n.csv", index=False)
    df_d.to_csv(_RES / f"exp_scenario_{scenario_tag}_exp3_d.csv", index=False)
    plot_exp23(df_n, df_d, true_fx, true_fy, scenario_tag)

    # Exp 4
    print(f"\n--- Exp 4: Mismatch + Ablation ---")
    df4 = run_exp4(true_fx, true_fy, scenario_tag, base_seed=6000+seed_offset)
    df4.to_csv(_RES / f"exp_scenario_{scenario_tag}_exp4_mismatch.csv", index=False)
    summary4 = plot_exp4(df4, true_fx, true_fy, scenario_tag)

    elapsed = (time.perf_counter() - t0) / 60
    print(f"\nScenario {scenario_tag} total: {elapsed:.1f} min")
    return {"scenario": scenario_tag, "true_fx": true_fx, "true_fy": true_fy,
            "elapsed_min": elapsed, **summary4}
