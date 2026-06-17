"""
fixed版公式実験結果から修論用主要図を生成するスクリプト。

入力CSV:
  expfam/results/fixed_official/exp2/fixed_exp2_n_sweep_agg.csv
  expfam/results/fixed_official/exp4/fixed_exp4_scen_a_ratios.csv
  expfam/results/fixed_official/exp4/fixed_exp4_scen_b_ratios.csv
  expfam/results/fixed_official/exp4/fixed_exp4_scen_c_ratios.csv

出力:
  expfam/figures/fixed_official/fig_fixed_exp2_n_sweep.png/pdf
  expfam/figures/fixed_official/fig_fixed_exp4_mismatch_max_ratios.png/pdf
  expfam/figures/fixed_official/fig_fixed_exp4_mismatch_all_conditions.png/pdf
  expfam/figures/fixed_official/fig_fixed_exp4_mismatch_plot_values.csv

既存ファイルは上書きしない。
"""

from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

_SRC    = Path(__file__).parent
_ROOT   = _SRC.parent.parent
_RES    = _ROOT / "expfam" / "results" / "fixed_official"
_OUTDIR = _ROOT / "expfam" / "figures" / "fixed_official"
_OUTDIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Helper: save without overwrite
# ─────────────────────────────────────────────────────────────────────

def save_fig(fig, stem):
    for ext in ("png", "pdf"):
        path = _OUTDIR / f"{stem}.{ext}"
        if path.exists():
            print(f"  SKIP (exists): {path}")
        else:
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"  Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Style constants
# ─────────────────────────────────────────────────────────────────────

SCEN_COLORS = {"A": "#2196F3", "B": "#FF5722", "C": "#4CAF50"}
SCEN_MARKERS = {"A": "o", "B": "s", "C": "^"}
SCEN_LABELS = {
    "A": "Scenario A (X=Poisson, Y=Bernoulli)",
    "B": "Scenario B (X=Gaussian, Y=Poisson)",
    "C": "Scenario C (X=Bernoulli, Y=Gaussian)",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────────────
# Figure 1(a): n-sweep
# ─────────────────────────────────────────────────────────────────────

def make_fig_n_sweep():
    agg_path = _RES / "exp2" / "fixed_exp2_n_sweep_agg.csv"
    df = pd.read_csv(agg_path)
    print(f"\n[Fig1a] Loaded: {agg_path}  ({len(df)} rows)")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for sc in ["A", "B", "C"]:
        sub = df[df["scenario"] == sc].sort_values("n")
        ax.errorbar(
            sub["n"], sub["rmse_z_mean"],
            yerr=sub["rmse_z_std"],
            label=SCEN_LABELS[sc],
            color=SCEN_COLORS[sc],
            marker=SCEN_MARKERS[sc],
            linewidth=1.8, markersize=6,
            capsize=4, capthick=1.2,
            elinewidth=1.0,
        )

    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("RMSE($\\mathbf{Z}$)")
    ax.set_title("n-sweep: RMSE(Z) vs. Sample Size\n"
                 "(fixed implementation, 10 trials, $d=15$, $k^*=3$)")
    ax.set_xticks([50, 100, 150, 200, 250, 300])
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    save_fig(fig, "fig_fixed_exp2_n_sweep")
    print("  [Fig1a] done")


# ─────────────────────────────────────────────────────────────────────
# Figure 1(b): mismatch — max ratio per scenario (3×3 grid only)
# ─────────────────────────────────────────────────────────────────────

ABLATION_CONDITIONS = {"Y_only", "X_only"}

FAM_LABEL = {"gaussian": "Gaussian", "bernoulli": "Bernoulli", "poisson": "Poisson"}


def _grid_max(ratio_df):
    """Return row with max ratio_vs_oracle, excluding ablation conditions."""
    grid = ratio_df[
        ~ratio_df["condition_name"].isin(ABLATION_CONDITIONS) &
        (ratio_df["ratio_vs_oracle"] > 1.0)
    ]
    return grid.loc[grid["ratio_vs_oracle"].idxmax()]


def make_fig_mismatch_max():
    ratios = {}
    selected_rows = []
    for sc in ["A", "B", "C"]:
        path = _RES / "exp4" / f"fixed_exp4_scen_{sc.lower()}_ratios.csv"
        df = pd.read_csv(path)
        print(f"\n[Fig1b] Loaded: {path}  ({len(df)} rows)")
        best = _grid_max(df)
        ratios[sc] = best
        selected_rows.append({
            "scenario":           sc,
            "selected_condition": best["condition_name"],
            "est_x":              best["est_x"],
            "est_y":              best["est_y"],
            "ratio_vs_oracle":    best["ratio_vs_oracle"],
            "note": (
                f"3x3 grid max (ablation excluded), "
                f"est_x={best['est_x']}, est_y={best['est_y']}"
            ),
        })

    # Save plot values CSV
    pv_path = _OUTDIR / "fig_fixed_exp4_mismatch_plot_values.csv"
    if pv_path.exists():
        print(f"  SKIP (exists): {pv_path}")
    else:
        pd.DataFrame(selected_rows).to_csv(pv_path, index=False)
        print(f"  Saved: {pv_path}")

    # Bar chart
    scenarios = ["A", "B", "C"]
    heights   = [float(ratios[sc]["ratio_vs_oracle"]) for sc in scenarios]
    colors    = [SCEN_COLORS[sc] for sc in scenarios]
    bar_labels = []
    for sc in scenarios:
        r = ratios[sc]
        bar_labels.append(
            f"Scenario {sc}\n"
            f"({FAM_LABEL[r['est_x']]}-X\n"
            f" {FAM_LABEL[r['est_y']]}-Y)"
        )

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    bars = ax.bar(bar_labels, heights, color=colors, width=0.5,
                  edgecolor="black", linewidth=0.8)

    # Annotate ratio values
    for bar, h in zip(bars, heights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.4,
            f"{h:.2f}×",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    # Reference line at y=1
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Oracle (1.0×)")

    ax.set_ylabel("RMSE(Z) ratio vs. oracle")
    ax.set_title(
        "Distribution Misspecification Impact\n"
        "(max ratio in $3\\times3$ grid, ablation excluded, fixed implementation)"
    )
    ax.set_ylim(0, max(heights) * 1.2)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    save_fig(fig, "fig_fixed_exp4_mismatch_max_ratios")
    print("  [Fig1b-max] done")


# ─────────────────────────────────────────────────────────────────────
# Figure supplementary: all 9 conditions across A/B/C (heatmap style)
# ─────────────────────────────────────────────────────────────────────

def make_fig_mismatch_all():
    """Grouped bar chart: all 9 grid conditions for each scenario."""
    FAMILIES = ["gaussian", "bernoulli", "poisson"]
    FAM_SHORT = {"gaussian": "Gaus", "bernoulli": "Bern", "poisson": "Pois"}

    # Build x-label: est_x × est_y combos in FAMILIES order
    grid_combos = [(ex, ey) for ex in FAMILIES for ey in FAMILIES]
    x_labels = [f"{FAM_SHORT[ex]}-X\n{FAM_SHORT[ey]}-Y" for ex, ey in grid_combos]

    all_data = {}
    for sc in ["A", "B", "C"]:
        path = _RES / "exp4" / f"fixed_exp4_scen_{sc.lower()}_ratios.csv"
        df = pd.read_csv(path)
        # Filter only 3×3 grid
        grid = df[~df["condition_name"].isin(ABLATION_CONDITIONS)].copy()
        # Map to combo order
        combo_map = {(r["est_x"], r["est_y"]): r["ratio_vs_oracle"]
                     for _, r in grid.iterrows()}
        all_data[sc] = [combo_map.get((ex, ey), float("nan"))
                        for ex, ey in grid_combos]

    x = np.arange(len(x_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, sc in enumerate(["A", "B", "C"]):
        offset = (i - 1) * width
        rects = ax.bar(x + offset, all_data[sc], width,
                       label=f"Scen {sc}", color=SCEN_COLORS[sc],
                       edgecolor="black", linewidth=0.6, alpha=0.85)

    # Mark oracle conditions (ratio ≈ 1.0) with a dashed outline
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Estimated distribution family (X-side / Y-side)")
    ax.set_ylabel("RMSE(Z) ratio vs. oracle")
    ax.set_title(
        "All $3\\times3$ Misspecification Conditions\n"
        "(fixed implementation, ablation excluded)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Highlight oracle cells (ratio ≈ 1) with grey background band
    for j, (ex, ey) in enumerate(grid_combos):
        vals = [all_data[sc][j] for sc in ["A", "B", "C"]
                if not np.isnan(all_data[sc][j])]
        if vals and min(vals) < 1.1:
            ax.axvspan(j - 0.5, j + 0.5, color="lightgrey", alpha=0.25, zorder=0)

    fig.tight_layout()
    save_fig(fig, "fig_fixed_exp4_mismatch_all_conditions")
    print("  [Fig1b-all] done")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Output dir: {_OUTDIR}")

    make_fig_n_sweep()
    make_fig_mismatch_max()
    make_fig_mismatch_all()

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
