"""
per-column 検証フェーズの図を作成する。

入力（すべて expfam/results/per_column_family/）:
    single_vs_joint_summary.csv / attribute_ablation_summary.csv /
    noise_check_summary.csv / movielens_mixed_x_agg.csv

出力（figures/per_column_family/、png + pdf）:
    single_vs_joint_rmse_z          条件別 RMSE_Z（seed 別点 + 平均バー）
    single_vs_joint_heldout_ll      条件別 strict held-out test Y 対数尤度
    attribute_ablation_lines        属性追加 ablation の折れ線（RMSE_Z / test ll）
    noise_check_lines               ノイズ列数に対する性能変化
    movielens_mixed_x_test_y_ll     MovieLens pilot の条件別 strict held-out
                                    test Y 対数尤度（横向きドットプロット）

配色は dataviz ルールに従い「役割」で固定:
    per-column（本命 prototype）= blue、単独属性 = aqua、
    全列共通強制（誤指定比較用）= yellow、y_only = muted gray。

実行: python tools/research_audit/plot_per_column_figures.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent.parent
RES_DIR = _ROOT / "expfam" / "results" / "per_column_family"
FIG_DIR = _ROOT / "figures" / "per_column_family"

# dataviz reference palette（役割固定; 光面 #fcfcfb 前提）
C_PERCOL = "#2a78d6"    # slot 1 blue  : per-column（本命）
C_SINGLE = "#1baf7a"    # slot 2 aqua  : 単独属性
C_FORCED = "#eda100"    # slot 3 yellow: 全列共通強制（誤指定比較用）
C_YONLY = "#898781"     # muted        : y_only ベースライン
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

ROLE = {
    "per_column_all": ("per-column (correct spec.)", C_PERCOL),
    "single_gaussian": ("single: Gaussian block", C_SINGLE),
    "single_bernoulli": ("single: Bernoulli block", C_SINGLE),
    "single_poisson": ("single: Poisson block", C_SINGLE),
    "all_gaussian": ("forced all-Gaussian (missp.)", C_FORCED),
    "all_bernoulli": ("forced all-Bernoulli (missp.)", C_FORCED),
    "all_poisson": ("forced all-Poisson (missp.)", C_FORCED),
    "all_bernoulli_binarized": ("binarized all-Bernoulli (missp.)", C_FORCED),
    "y_only": ("Y-only (no X)", C_YONLY),
}
ORDER = ["per_column_all", "single_gaussian", "single_poisson",
         "single_bernoulli", "all_gaussian", "all_poisson",
         "all_bernoulli_binarized", "all_bernoulli", "y_only"]

# MovieLens pilot（movielens_mixed_x_agg.csv）専用の表示名・配色・並び順。
# 役割は既存 ROLE と同じ色定数を再利用（per-column=blue, 全列強制=yellow,
# y_only=gray, 単独属性ブロック=aqua）。
ROLE_ML = {
    "genre_only": ("Genre only", C_SINGLE),
    "y_only": ("Y only", C_YONLY),
    "mixed_all_gaussian": ("Mixed X: forced Gaussian", C_FORCED),
    "mixed_percolumn": ("Mixed X: per-column", C_PERCOL),
    "rating_stats_only": ("Rating statistics only", C_SINGLE),
    "mixed_all_bernoulli": ("Mixed X: forced Bernoulli", C_FORCED),
}
ORDER_ML = ["genre_only", "y_only", "mixed_all_gaussian",
            "mixed_percolumn", "rating_stats_only", "mixed_all_bernoulli"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.dpi": 150,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/per_column_family/{name}.png/.pdf")


def barh_with_points(df, metric, title, xlabel, name, clip=None):
    """条件別比較。clip なし: 0 起点の水平バー + seed 別点。
    clip あり（0 が軸範囲外）: バーは誤解を招くのでドットプロットにする。"""
    conds = [c for c in ORDER if c in df["condition"].unique()]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ypos = np.arange(len(conds))[::-1]
    for y, c in zip(ypos, conds):
        vals = df.loc[df["condition"] == c, metric].to_numpy()
        mean = float(np.mean(vals))
        label, color = ROLE[c]
        if clip is None:
            ax.barh(y, mean, height=0.62, color=color, alpha=0.85, zorder=2)
            ax.scatter(vals, np.full_like(vals, y, dtype=float),
                       s=14, color=INK, alpha=0.65, zorder=3, linewidths=0)
            ax.text(mean, y, f"  {mean:.3f}", va="center", ha="left",
                    fontsize=8, color=INK)
        else:
            vshow = vals[(vals >= clip[0]) & (vals <= clip[1])]
            n_off = len(vals) - len(vshow)
            ax.scatter(vshow, np.full_like(vshow, y, dtype=float),
                       s=16, color=color, alpha=0.55, zorder=3, linewidths=0)
            if clip[0] <= mean <= clip[1]:
                ax.scatter([mean], [y], s=70, color=color, zorder=4,
                           edgecolors="white", linewidths=1.2)
                ax.text(mean, y + 0.32, f"{mean:.3f}", va="bottom",
                        ha="center", fontsize=7.5, color=INK)
            else:
                ax.annotate(f"mean {mean:.1f} (off scale) →" if mean > clip[1]
                            else f"← mean {mean:.1f} (off scale)",
                            xy=(clip[0] if mean < clip[0] else clip[1], y),
                            va="center", fontsize=7.5, color=INK,
                            ha="left" if mean < clip[0] else "right")
            if n_off and not (mean < clip[0] or mean > clip[1]):
                ax.text(clip[0], y - 0.3, f"({n_off} seed off scale)",
                        va="top", ha="left", fontsize=7, color=MUTED)
    ax.set_yticks(ypos)
    ax.set_yticklabels([ROLE[c][0] for c in conds])
    if clip is not None:
        ax.set_xlim(clip)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    save(fig, name)


def fig_single_vs_joint():
    df = pd.read_csv(RES_DIR / "single_vs_joint_summary.csv")
    barh_with_points(
        df, "rmse_Z",
        "Mixed-X synthetic: RMSE(Z) by condition (3 seeds, Procrustes)",
        "RMSE(Z)  (lower is better)",
        "single_vs_joint_rmse_z")
    # all_bernoulli の test ll は −106 まで落ちる trial があるため軸を制限し注記
    barh_with_points(
        df, "test_y_ll",
        "Mixed-X synthetic: strict held-out test Y log-lik. per pair",
        "test log-likelihood / pair  (higher is better)",
        "single_vs_joint_heldout_ll", clip=(-2.2, -2.0))


def fig_ablation():
    df = pd.read_csv(RES_DIR / "attribute_ablation_summary.csv")
    agg = df.groupby(["step", "condition"]).agg(
        rmse=("rmse_Z", "mean"), rmse_sd=("rmse_Z", "std"),
        ll=("test_y_ll", "mean"), ll_sd=("test_y_ll", "std")).reset_index()
    agg = agg.sort_values("step")
    labels = {"y_only": "Y-only", "bern_only": "+Bern(3)",
              "bern_gauss": "+Gauss(3)", "bern_gauss_pois": "+Pois(3)",
              "bern_gauss_pois_noise3": "+noise(3)"}
    x = agg["step"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    for ax, m, sd, ylab, better in (
            (axes[0], "rmse", "rmse_sd", "RMSE(Z)", "lower"),
            (axes[1], "ll", "ll_sd", "test Y log-lik. / pair", "higher")):
        ax.errorbar(x, agg[m], yerr=agg[sd], color=C_PERCOL, linewidth=2,
                    marker="o", markersize=5, capsize=3, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[c] for c in agg["condition"]], fontsize=8)
        ax.set_ylabel(f"{ylab}  ({better} is better)")
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_title("Attribute-addition ablation (per-column, 3 seeds)",
                      loc="left")
    fig.tight_layout()
    save(fig, "attribute_ablation_lines")


def fig_noise():
    df = pd.read_csv(RES_DIR / "noise_check_summary.csv")
    agg = df.groupby("condition").agg(
        n=("n_noise_cols", "first"), fam=("noise_family", "first"),
        rmse=("rmse_Z", "mean"), rmse_sd=("rmse_Z", "std"),
        ll=("test_y_ll", "mean"), ll_sd=("test_y_ll", "std")).reset_index()
    gauss = agg[agg["fam"].isin(["none", "gaussian"])].sort_values("n")
    others = agg[agg["fam"].isin(["bernoulli", "poisson"])]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    for ax, m, sd, ylab, better in (
            (axes[0], "rmse", "rmse_sd", "RMSE(Z)", "lower"),
            (axes[1], "ll", "ll_sd", "test Y log-lik. / pair", "higher")):
        ax.errorbar(gauss["n"], gauss[m], yerr=gauss[sd], color=C_PERCOL,
                    linewidth=2, marker="o", markersize=5, capsize=3,
                    label="+ Gaussian noise cols", zorder=3)
        for _, r in others.iterrows():
            mk = "s" if r["fam"] == "bernoulli" else "^"
            ax.errorbar([r["n"]], [r[m]], yerr=[r[sd]], color=C_FORCED,
                        marker=mk, markersize=6, capsize=3, linestyle="none",
                        label=f"+ {r['fam']} noise (3)", zorder=3)
        ax.set_xlabel("number of noise columns added to 9 informative cols")
        ax.set_ylabel(f"{ylab}  ({better} is better)")
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_title("Noise-attribute check (per-column, 3 seeds)", loc="left")
    h, l = axes[1].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    axes[1].legend(uniq.values(), uniq.keys(), frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "noise_check_lines")


def fig_movielens_mixed_x():
    """MovieLens pilot: 条件別 strict held-out test Y log-likelihood。

    値は全条件が負であり 0 起点の棒グラフは比較を歪めるため、
    横向きドットプロット（平均点 + agg CSV の標準偏差によるエラーバー）にする。
    数値は movielens_mixed_x_agg.csv（集計済み）からのみ読み込み、
    ソースコードへのハードコードは行わない。
    """
    df = pd.read_csv(RES_DIR / "movielens_mixed_x_agg.csv")

    present = list(df["condition"])
    conds = [c for c in ORDER_ML if c in present]
    missing = [c for c in ORDER_ML if c not in present]
    extra = [c for c in present if c not in ORDER_ML]
    for c in missing:
        print(f"[warn] fig_movielens_mixed_x: condition '{c}' not found in "
              f"movielens_mixed_x_agg.csv — skipped")
    if extra:
        print(f"[warn] fig_movielens_mixed_x: unexpected condition(s) in CSV "
              f"not in ORDER_ML (NOT plotted): {extra}")

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ypos = np.arange(len(conds))[::-1]
    for y, c in zip(ypos, conds):
        row = df.loc[df["condition"] == c].iloc[0]
        mean = float(row["test_y_ll_mean"])
        std = row.get("test_y_ll_std", np.nan)
        label, color = ROLE_ML[c]
        if pd.notna(std):
            ax.errorbar([mean], [y], xerr=[float(std)], fmt="o",
                        color=color, ecolor=color, elinewidth=1.3,
                        capsize=4, markersize=8, markeredgecolor="white",
                        markeredgewidth=1.0, zorder=3)
        else:
            ax.scatter([mean], [y], s=70, color=color, zorder=3,
                       edgecolors="white", linewidths=1.0)
        ax.text(mean, y + 0.30, f"{mean:.3f}", va="bottom", ha="center",
                fontsize=8.5, color=INK)

    ax.set_yticks(ypos)
    ax.set_yticklabels([ROLE_ML[c][0] for c in conds])
    ax.set_ylim(min(ypos) - 0.7, max(ypos) + 0.7)
    ax.margins(x=0.15)
    ax.set_xlabel("test Y log-likelihood / pair  (higher is better)")
    ax.set_title("MovieLens: held-out Y prediction by attribute condition",
                 loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "movielens_mixed_x_test_y_ll")


if __name__ == "__main__":
    fig_single_vs_joint()
    fig_ablation()
    fig_noise()
    fig_movielens_mixed_x()
    print("done")
