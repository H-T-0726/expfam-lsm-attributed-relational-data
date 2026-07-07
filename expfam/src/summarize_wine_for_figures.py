"""
Wine fixed-pilot 実験結果をNotion資料用に要約するスクリプト。

既存の expfam/results/real_data/wine_fixed_pilot/ 以下のCSVを読み込むだけで、
モデルの再学習・再実行は一切行わない。

使う結果は fixed版 (run_fixed_real_wine_pilot.py が生成したもの) のみ。
旧版 (expfam/results/wine_dual_results.csv, expfam/results/wine_F.npy) および
元論文再現 (reproduction/results/results_real_wine.csv) は読み込まない
(Notion主結果に混ぜないため)。

出力先 (すべて新規):
  expfam/results/real_data/wine_clean/*.csv
  expfam/figures/real_data/wine_clean/*.png/.pdf

既存の wine_fixed_pilot/ 以下のCSV・図、および既存のモデル実装は変更しない。
"""

import os
import shutil

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

INPUT_RESULTS_DIR = os.path.join(ROOT, "expfam", "results", "real_data", "wine_fixed_pilot")
INPUT_FIG_DIR = os.path.join(ROOT, "expfam", "figures", "real_data", "wine_fixed_pilot")
OUT_CSV_DIR = os.path.join(ROOT, "expfam", "results", "real_data", "wine_clean")
OUT_FIG_DIR = os.path.join(ROOT, "expfam", "figures", "real_data", "wine_clean")

BIC_K1TO9_FILE = "wine_bic_k1to9.csv"
BIC_BESTK_FILE = "wine_bic_bestk.csv"
ABLATION_FILE = "wine_ablation_metrics.csv"
Z_EMBEDDINGS_FILE = "wine_z_embeddings.csv"

BEST_K = 3


def load_inputs():
    loaded = {}
    missing = []
    for fname in [BIC_K1TO9_FILE, BIC_BESTK_FILE, ABLATION_FILE, Z_EMBEDDINGS_FILE]:
        path = os.path.join(INPUT_RESULTS_DIR, fname)
        if os.path.exists(path):
            loaded[fname] = pd.read_csv(path)
        else:
            missing.append(path)
    return loaded, missing


def build_bic_summary(data):
    df = data[BIC_K1TO9_FILE]
    agg = df.groupby("k").agg(
        BIC_mean=("bic", "mean"),
        BIC_std=("bic", "std"),
        success_rate=("success", "mean"),
    ).reset_index()

    interp = {}
    for k in agg["k"]:
        if k == BEST_K:
            interp[k] = "BIC minimum; corresponds to label-derived 3-class structure"
        elif k < BEST_K:
            interp[k] = "underfit relative to BIC minimum"
        else:
            interp[k] = "BIC increasing; added complexity not justified by fit"
    agg["short_interpretation"] = agg["k"].map(interp)

    agg["BIC_mean"] = agg["BIC_mean"].round(2)
    agg["BIC_std"] = agg["BIC_std"].round(2)
    agg["success_rate"] = agg["success_rate"].round(3)

    out_path = os.path.join(OUT_CSV_DIR, "wine_bic_summary_clean.csv")
    agg.to_csv(out_path, index=False)
    return agg


def build_ablation_summary(data):
    df = data[ABLATION_FILE]
    agg = df.groupby("condition").agg(
        BIC_mean=("bic", "mean"),
        BIC_std=("bic", "std"),
        AUC_Y_mean=("auc_y", "mean"),
        AUC_Y_std=("auc_y", "std"),
        AP_Y_mean=("ap_y", "mean"),
        AP_Y_std=("ap_y", "std"),
        silhouette_mean=("silhouette", "mean"),
        silhouette_std=("silhouette", "std"),
    ).reset_index()

    interp = {
        "X+Y": "uses both attributes and relation; best balanced condition",
        "X_only": "cannot reconstruct Y because relation data is not used",
        "Y_only": "very strong because Y is label-derived from class labels",
    }
    agg["short_interpretation"] = agg["condition"].map(interp)

    order = ["X+Y", "X_only", "Y_only"]
    agg["_order"] = agg["condition"].apply(lambda c: order.index(c) if c in order else 99)
    agg = agg.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    for col in ["BIC_mean", "BIC_std", "AUC_Y_mean", "AUC_Y_std", "AP_Y_mean", "AP_Y_std",
                "silhouette_mean", "silhouette_std"]:
        agg[col] = agg[col].round(4)

    out_path = os.path.join(OUT_CSV_DIR, "wine_ablation_summary_clean.csv")
    agg.to_csv(out_path, index=False)
    return agg


def build_use_in_notion_plan():
    rows = [
        dict(item="BIC vs K figure",
             file_path="expfam/figures/real_data/wine_clean/wine_bic_k_summary_clean.png",
             where_to_use="本文：結果1 BICによるK選択",
             importance="必須",
             caution="K=3はラベル由来Yの影響が大きい。「未知構造の発見」とは言わない"),
        dict(item="Ablation bar chart",
             file_path="expfam/figures/real_data/wine_clean/wine_ablation_bar_clean.png",
             where_to_use="本文：結果2 X+Y/X_only/Y_only比較",
             importance="必須",
             caution="Y_onlyが強いのはYがラベル由来のため。「属性Xが不要」とは言わない"),
        dict(item="Z ablation comparison (clean copy)",
             file_path="expfam/figures/real_data/wine_clean/wine_z_ablation_comparison_clean.png",
             where_to_use="トグル・補助（潜在空間の直感的な可視化）",
             importance="推奨（必須ではない）",
             caution="既存図のコピーであり再生成はしていない（X_only/Y_onlyの生Zが保存されていないため再生成不可）"),
        dict(item="wine_bic_summary_clean.csv",
             file_path="expfam/results/real_data/wine_clean/wine_bic_summary_clean.csv",
             where_to_use="本文の表データ源（BIC vs Kの数値表）",
             importance="必須",
             caution="k=1〜9, 5trial平均。fixed版のみ"),
        dict(item="wine_ablation_summary_clean.csv",
             file_path="expfam/results/real_data/wine_clean/wine_ablation_summary_clean.csv",
             where_to_use="本文の表データ源（ablationの数値表）",
             importance="必須",
             caution="k=3固定, 5trial平均。fixed版のみ"),
        dict(item="F heatmap (旧版wine_F.npy由来)",
             file_path="(未作成)",
             where_to_use="使用しない",
             importance="使用不可",
             caution="fixed版のFは保存されていない。旧版Fを使うとfixed版結果と混同するため今回は作成しない"),
        dict(item="旧版 wine_dual_results.csv",
             file_path="expfam/results/wine_dual_results.csv",
             where_to_use="使用しない（必要なら混同注意の脚注のみ）",
             importance="使用不可",
             caution="k=6固定・旧実装(model_dual_expfam.py)。fixed版のk-sweep結果と混同しないこと"),
        dict(item="reproduction/results_real_wine.csv",
             file_path="reproduction/results/results_real_wine.csv",
             where_to_use="使用しない（必要なら混同注意の脚注のみ）",
             importance="使用不可",
             caution="元論文再現(Bernoulli-Y固定モデル)。BICを持たず指標体系が異なるため直接比較不可"),
    ]
    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_CSV_DIR, "wine_use_in_notion_plan.csv")
    out.to_csv(out_path, index=False)
    return out


def make_fig_bic_k_summary(bic_summary):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = bic_summary["k"].values
    means = bic_summary["BIC_mean"].values
    stds = bic_summary["BIC_std"].fillna(0).values

    colors = ["#C44E52" if k == BEST_K else "#4C72B0" for k in ks]
    ax.bar(ks, means, yerr=stds, capsize=4, color=colors, ecolor="gray")

    best_idx = list(ks).index(BEST_K)
    ax.annotate(
        f"min at K={BEST_K}",
        xy=(BEST_K, means[best_idx]),
        xytext=(BEST_K, means[best_idx] + max(means) * 0.08),
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#C44E52",
        arrowprops=dict(arrowstyle="->", color="#C44E52"),
    )

    ax.set_xlabel("K (number of latent dimensions)")
    ax.set_ylabel("BIC (mean over 5 trials, error bars = std)")
    ax.set_title("Wine pilot (fixed): BIC vs K")
    ax.set_xticks(ks)
    fig.text(
        0.5, -0.02,
        "Note: Y is label-derived from wine class labels (same-class indicator), not a measured network.",
        ha="center", fontsize=8, style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_FIG_DIR, f"wine_bic_k_summary_clean.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_fig_ablation_bar(ablation_summary):
    conditions = ablation_summary["condition"].tolist()
    metrics = [
        ("AUC_Y", "AUC_Y_mean", "AUC_Y_std"),
        ("AP_Y", "AP_Y_mean", "AP_Y_std"),
        ("silhouette", "silhouette_mean", "silhouette_std"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(conditions))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for i, (label, mean_col, std_col) in enumerate(metrics):
        means = ablation_summary[mean_col].values
        stds = ablation_summary[std_col].fillna(0).values
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=label, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(f"Wine pilot (fixed): X+Y vs X_only vs Y_only (K={BEST_K})")
    ax.legend()
    fig.text(
        0.5, -0.02,
        "Note: Y_only is strong because Y is label-derived from wine class labels (same-class indicator).",
        ha="center", fontsize=8, style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_FIG_DIR, f"wine_ablation_bar_clean.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def copy_z_ablation_comparison():
    copied = []
    skipped = []
    for ext in ("png", "pdf"):
        src = os.path.join(INPUT_FIG_DIR, f"wine_z_ablation_comparison.{ext}")
        dst = os.path.join(OUT_FIG_DIR, f"wine_z_ablation_comparison_clean.{ext}")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied.append(dst)
        else:
            skipped.append(src)
    return copied, skipped


def main():
    os.makedirs(OUT_CSV_DIR, exist_ok=True)
    os.makedirs(OUT_FIG_DIR, exist_ok=True)

    data, missing = load_inputs()
    print(f"Loaded {len(data)} input files; missing: {missing if missing else 'none'}")

    bic_summary = build_bic_summary(data)
    ablation_summary = build_ablation_summary(data)
    build_use_in_notion_plan()

    make_fig_bic_k_summary(bic_summary)
    make_fig_ablation_bar(ablation_summary)
    copied, skipped = copy_z_ablation_comparison()

    print("BIC summary:\n", bic_summary)
    print("Ablation summary:\n", ablation_summary)
    print(f"Copied Z ablation comparison figure(s): {copied}")
    if skipped:
        print(f"WARNING: source Z ablation figure(s) not found: {skipped}")

    f_npy_fixed = os.path.join(INPUT_RESULTS_DIR, "wine_F.npy")
    if os.path.exists(f_npy_fixed):
        print("Fixed-version F found (unexpected) -- F heatmap NOT implemented in this script.")
    else:
        print("No fixed-version F (.npy) found in wine_fixed_pilot/ -- F heatmap skipped, as intended.")


if __name__ == "__main__":
    main()
