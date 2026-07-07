"""
common_reconstruction_summary.csv (Wine/Cora/MovieLens共通再構成評価のtrial別結果)
を読み込み、Notion資料用の集計CSV・比較表・図を作成するスクリプト。

既存CSVの読み込みのみ。モデル再学習は行わない
(再学習自体は run_common_realdata_reconstruction_eval.py が担当)。

出力先 (すべて新規、同じ common_reconstruction_eval ディレクトリ内):
  expfam/results/real_data/common_reconstruction_eval/common_reconstruction_agg.csv
  expfam/results/real_data/common_reconstruction_eval/common_reconstruction_overview.csv
  expfam/results/real_data/common_reconstruction_eval/use_in_notion_plan.csv
  expfam/figures/real_data/common_reconstruction_eval/common_reconstruction_overview.png/pdf
  expfam/figures/real_data/common_reconstruction_eval/wine_reconstruction_k3_k6.png/pdf
  expfam/figures/real_data/common_reconstruction_eval/cora_reconstruction_k_metrics.png/pdf
  expfam/figures/real_data/common_reconstruction_eval/movielens_reconstruction_k_metrics.png/pdf
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(ROOT, "expfam", "results", "real_data", "common_reconstruction_eval")
FIG_DIR = os.path.join(ROOT, "expfam", "figures", "real_data", "common_reconstruction_eval")

WINE_MAIN_K = {3: "BIC minimum (fixed-version k-sweep)", 6: "K used in the original paper's Wine experiment"}
CORA_MAIN_K = {1: "BIC minimum", 3: "NMI/ARI maximum (best label structure)", 6: "AUC/AP maximum (best link reconstruction, also best in held-out)"}
ML_MAIN_K = {5: "BIC minimum", 8: "Best RMSE_Y/Pearson_Y/AP among tested k"}


def load_summary():
    path = os.path.join(OUT_DIR, "common_reconstruction_summary.csv")
    return pd.read_csv(path)


def build_agg(df):
    success_df = df[df["success"] == True].copy()  # noqa: E712

    agg_rows = []
    for (dataset, k), g in success_df.groupby(["dataset", "k"]):
        x_name = g["X_primary_metric_name"].iloc[0]
        y_name = g["Y_primary_metric_name"].iloc[0]
        row = dict(
            dataset=dataset, k=k, n_trials=len(g),
            success_rate=len(g) / len(df[(df["dataset"] == dataset) & (df["k"] == k)]),
            BIC_mean=g["BIC"].mean(), BIC_std=g["BIC"].std(),
            X_primary_metric_name=x_name,
            X_primary_metric_mean=g["X_primary_metric_value"].mean(),
            X_primary_metric_std=g["X_primary_metric_value"].std(),
            Y_primary_metric_name=y_name,
            Y_primary_metric_mean=g["Y_primary_metric_value"].mean(),
            Y_primary_metric_std=g["Y_primary_metric_value"].std(),
            RMSE_X_mean=g["RMSE_X"].mean(), RMSE_Y_mean=g["RMSE_Y"].mean(),
            BCE_X_mean=g["BCE_X"].mean(), BCE_Y_mean=g["BCE_Y"].mean(),
            AUC_Y_mean=g["AUC_Y"].mean(), AP_Y_mean=g["AP_Y"].mean(),
            Pearson_Y_mean=g["Pearson_Y"].mean(), Spearman_Y_mean=g["Spearman_Y"].mean(),
            Poisson_NLL_mean=g["Poisson_NLL"].mean(),
            silhouette_mean=g["silhouette"].mean(), NMI_mean=g["NMI"].mean(), ARI_mean=g["ARI"].mean(),
            w0_mean=g["w0"].mean(), runtime_mean=g["runtime"].mean(),
        )

        main_k_map = {"wine": WINE_MAIN_K, "cora": CORA_MAIN_K, "movielens": ML_MAIN_K}.get(dataset, {})
        if k in main_k_map:
            row["short_interpretation"] = main_k_map[k]
        else:
            row["short_interpretation"] = "not a main k for this dataset; included for the fuller k-sweep (priority B)"
        agg_rows.append(row)

    out = pd.DataFrame(agg_rows).sort_values(["dataset", "k"]).reset_index(drop=True)
    out.to_csv(os.path.join(OUT_DIR, "common_reconstruction_agg.csv"), index=False)
    return out


def build_overview(agg_df):
    def fmt(g, col, fmt_str="{:.4f}"):
        v = g[col].iloc[0]
        return fmt_str.format(v) if pd.notna(v) else "NA"

    rows = []

    wine_g = agg_df[(agg_df["dataset"] == "wine") & (agg_df["k"].isin(WINE_MAIN_K.keys()))]
    rows.append(dict(
        dataset="Wine", data_role="Gaussian/Bernoulli basic real-data check (paper-aligned setup)",
        family_x="gaussian", family_y="bernoulli", main_k="K=3 and K=6",
        why_main_k="K=3 is the fixed-version BIC minimum; K=6 is the value used in the original paper's Wine experiment",
        X_reconstruction_summary="; ".join(
            f"k={r.k}: RMSE_X={r.RMSE_X_mean:.4f}, Pearson_X={'NA'}" for r in wine_g.itertuples()
        ),
        Y_reconstruction_summary="; ".join(
            f"k={r.k}: BCE_Y={r.BCE_Y_mean:.4f}, AUC_Y={r.AUC_Y_mean:.4f}, AP_Y={r.AP_Y_mean:.4f}" for r in wine_g.itertuples()
        ),
        BIC_summary="; ".join(f"k={r.k}: BIC={r.BIC_mean:.1f}" for r in wine_g.itertuples()),
        factor_interpretation_summary="F heatmap (13 chemical features x k); see wine_factor_top_features.csv",
        additional_eval_summary="Supplementary: BIC k=1..9 sweep, X+Y/X_only/Y_only ablation (existing wine_fixed_pilot results)",
        caution="Y is label-derived (same-class indicator), not an observed network",
    ))

    cora_g = agg_df[(agg_df["dataset"] == "cora") & (agg_df["k"].isin(CORA_MAIN_K.keys()))]
    rows.append(dict(
        dataset="Cora", data_role="Bernoulli/Bernoulli natural binary relation check",
        family_x="bernoulli", family_y="bernoulli", main_k="K=1, K=3, K=6",
        why_main_k="K=1 is the BIC minimum; K=3 is the NMI/ARI maximum (best label structure); K=6 is the AUC/AP maximum (best link reconstruction)",
        X_reconstruction_summary="; ".join(
            f"k={r.k}: BCE_X={r.BCE_X_mean:.4f}" for r in cora_g.itertuples()
        ),
        Y_reconstruction_summary="; ".join(
            f"k={r.k}: BCE_Y={r.BCE_Y_mean:.4f}, AUC_Y={r.AUC_Y_mean:.4f}, AP_Y={r.AP_Y_mean:.4f}" for r in cora_g.itertuples()
        ),
        BIC_summary="; ".join(f"k={r.k}: BIC={r.BIC_mean:.1f}" for r in cora_g.itertuples()),
        factor_interpretation_summary="Word-presence features have no public vocabulary (LINQS Cora); top words reported as feature_index. See cora_factor_top_words.csv",
        additional_eval_summary="Supplementary: held-out link prediction (cora_heldout_link_prediction, existing results)",
        caution="Y is a real observed citation link, not label-derived; best_k differs by criterion (BIC vs AUC/AP vs NMI/ARI)",
    ))

    ml_g = agg_df[(agg_df["dataset"] == "movielens") & (agg_df["k"].isin(ML_MAIN_K.keys()))]
    rows.append(dict(
        dataset="MovieLens", data_role="Bernoulli/Poisson count-relation check",
        family_x="bernoulli", family_y="poisson", main_k="K=5 and K=8",
        why_main_k="K=5 is the BIC minimum; K=8 has the best RMSE_Y/Pearson_Y/AP among tested k",
        X_reconstruction_summary="; ".join(
            f"k={r.k}: BCE_X={r.BCE_X_mean:.4f}" for r in ml_g.itertuples()
        ),
        Y_reconstruction_summary="; ".join(
            f"k={r.k}: Poisson_NLL={r.Poisson_NLL_mean:.4f}, RMSE_Y={r.RMSE_Y_mean:.4f}, Pearson_Y={r.Pearson_Y_mean:.4f}" for r in ml_g.itertuples()
        ),
        BIC_summary="; ".join(f"k={r.k}: BIC={r.BIC_mean:.1f}" for r in ml_g.itertuples()),
        factor_interpretation_summary="F heatmap (19 genres x k); some factors tentatively associate with popularity/rating attributes (see existing factor_interpretation_summary.csv)",
        additional_eval_summary="Supplementary: lift-based popularity-corrected ranking vs popularity/genre-cosine baselines (existing movielens_colike_interpretation results)",
        caution="Y is an in-sample reconstruction target (co-like counts), not a strict held-out prediction",
    ))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "common_reconstruction_overview.csv"), index=False)
    return out


def build_use_in_notion_plan():
    rows = [
        dict(dataset="all", item="Common reconstruction overview figure",
             file_path="expfam/figures/real_data/common_reconstruction_eval/common_reconstruction_overview.png",
             where_to_use="実データ実験 共通方針の説明図（本文冒頭）", importance="必須",
             caution="X/Yの指標はfamilyごとに異なるため、3データセットを単純な値の大小で比較しない"),
        dict(dataset="all", item="Common reconstruction overview table",
             file_path="expfam/results/real_data/common_reconstruction_eval/common_reconstruction_overview.csv",
             where_to_use="実データ実験 共通方針の説明表（本文冒頭）", importance="必須",
             caution="dataset別にX/Y再構成・BIC・factor解釈・補助評価をまとめた一覧"),
        dict(dataset="wine", item="Wine reconstruction k=3/k=6 figure",
             file_path="expfam/figures/real_data/common_reconstruction_eval/wine_reconstruction_k3_k6.png",
             where_to_use="Wine章：主結果（再構成評価）", importance="必須",
             caution="K=3とK=6を直接比較。BIC最小はK=3"),
        dict(dataset="wine", item="Wine F heatmap (k=3, k=6)",
             file_path="expfam/figures/real_data/common_reconstruction_eval/wine_F_heatmap_k3.png / wine_F_heatmap_k6.png",
             where_to_use="Wine章：主結果（factor解釈）", importance="必須",
             caution="13化学成分×factor。回転不定性に注意"),
        dict(dataset="wine", item="Existing BIC k-sweep / ablation (X+Y/X_only/Y_only)",
             file_path="expfam/results/real_data/wine_fixed_pilot/ , expfam/results/real_data/wine_clean/",
             where_to_use="Wine章：補足", importance="補足",
             caution="新しい再構成評価が主結果になったため、既存のBIC/ablationは補足に位置づけを変更"),
        dict(dataset="cora", item="Cora reconstruction k-metrics figure",
             file_path="expfam/figures/real_data/common_reconstruction_eval/cora_reconstruction_k_metrics.png",
             where_to_use="Cora章：主結果（再構成評価）", importance="必須",
             caution="K=1,3,6を中心に、BIC/X再構成/Y再構成の関係を見せる"),
        dict(dataset="cora", item="Cora F heatmap / top words (k=1,3,6)",
             file_path="expfam/figures/real_data/common_reconstruction_eval/cora_F_heatmap_k{1,3,6}.png , "
                       "expfam/results/real_data/common_reconstruction_eval/cora_factor_top_words.csv",
             where_to_use="Cora章：主結果（factor解釈）", importance="必須",
             caution="LINQS Cora公開データには単語名が含まれないため、word_index表記。単語の意味は復元できない"),
        dict(dataset="cora", item="Existing held-out link prediction",
             file_path="expfam/results/real_data/cora_heldout_link_prediction/ , expfam/results/real_data/cora_clean/",
             where_to_use="Cora章：補助評価", importance="補助",
             caution="新しい再構成評価が主結果。held-outは「未知リンクへの汎化」を見る補助評価として残す"),
        dict(dataset="movielens", item="MovieLens reconstruction k-metrics figure",
             file_path="expfam/figures/real_data/common_reconstruction_eval/movielens_reconstruction_k_metrics.png",
             where_to_use="MovieLens章：主結果（再構成評価）", importance="必須",
             caution="K=5,8を中心に、BIC/X再構成/Y再構成(Poisson)の関係を見せる"),
        dict(dataset="movielens", item="MovieLens F heatmap (k=5,8)",
             file_path="expfam/figures/real_data/common_reconstruction_eval/movielens_F_heatmap_k5.png / movielens_F_heatmap_k8.png",
             where_to_use="MovieLens章：主結果（factor解釈）", importance="必須",
             caution="19ジャンル×factor。既存のfactor_interpretation_summary.csvとの整合性を確認した上で使う"),
        dict(dataset="movielens", item="Existing lift-based popularity-corrected ranking",
             file_path="expfam/results/real_data/movielens_colike_interpretation/ , expfam/results/real_data/movielens_colike_clean/",
             where_to_use="MovieLens章：補助評価", importance="補助",
             caution="新しい再構成評価が主結果。ランキング評価はあくまで補助実験として位置づける。item-item baselineは強調しない"),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "use_in_notion_plan.csv"), index=False)
    return out


def make_overview_figure(agg_df):
    datasets = [
        ("wine", WINE_MAIN_K, "BCE_Y_mean", "AUC_Y_mean", "BCE_Y / AUC_Y (Bernoulli Y)"),
        ("cora", CORA_MAIN_K, "BCE_Y_mean", "AUC_Y_mean", "BCE_Y / AUC_Y (Bernoulli Y)"),
        ("movielens", ML_MAIN_K, "Poisson_NLL_mean", "Pearson_Y_mean", "Poisson_NLL / Pearson_Y (Poisson Y)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (ds, main_k, col1, col2, label) in zip(axes, datasets):
        g = agg_df[(agg_df["dataset"] == ds) & (agg_df["k"].isin(main_k.keys()))].sort_values("k")
        x = np.arange(len(g))
        width = 0.35
        ax2 = ax.twinx()
        ax.bar(x - width / 2, g[col1], width, color="#4C72B0", label=col1.replace("_mean", ""))
        ax2.bar(x + width / 2, g[col2], width, color="#DD8452", label=col2.replace("_mean", ""))
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in g["k"]])
        ax.set_title(f"{ds}\n{label}", fontsize=10)
        ax.set_ylabel(col1.replace("_mean", ""), color="#4C72B0")
        ax2.set_ylabel(col2.replace("_mean", ""), color="#DD8452")
    fig.suptitle("Common real-data reconstruction overview\n(metric differs by family -- do not compare raw values across datasets)")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"common_reconstruction_overview.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_wine_figure(agg_df):
    g = agg_df[(agg_df["dataset"] == "wine") & (agg_df["k"].isin(WINE_MAIN_K.keys()))].sort_values("k")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar([str(k) for k in g["k"]], g["BIC_mean"], yerr=g["BIC_std"], capsize=4, color="#4C72B0")
    axes[0].set_title("BIC"); axes[0].set_xlabel("k")
    axes[1].bar([str(k) for k in g["k"]], g["RMSE_X_mean"], color="#55A868")
    axes[1].set_title("RMSE_X (Gaussian X)"); axes[1].set_xlabel("k")
    axes[2].bar([str(k) for k in g["k"]], g["BCE_Y_mean"], color="#C44E52")
    axes[2].set_title("BCE_Y (Bernoulli Y)"); axes[2].set_xlabel("k")
    fig.suptitle("Wine: reconstruction at K=3 (BIC min) vs K=6 (paper reference)")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"wine_reconstruction_k3_k6.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_cora_figure(agg_df):
    g = agg_df[agg_df["dataset"] == "cora"].sort_values("k")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(g["k"], g["BIC_mean"], "o-", color="#4C72B0")
    axes[0].set_title("BIC vs k"); axes[0].set_xlabel("k")
    axes[1].plot(g["k"], g["BCE_X_mean"], "o-", color="#55A868")
    axes[1].set_title("BCE_X vs k (Bernoulli X)"); axes[1].set_xlabel("k")
    axes[2].plot(g["k"], g["AUC_Y_mean"], "o-", color="#C44E52", label="AUC_Y")
    axes[2].plot(g["k"], g["BCE_Y_mean"], "s-", color="#937860", label="BCE_Y")
    axes[2].set_title("Y reconstruction vs k (Bernoulli Y)"); axes[2].set_xlabel("k")
    axes[2].legend(fontsize=8)
    for k in CORA_MAIN_K:
        for ax in axes:
            ax.axvline(k, color="gray", linestyle=":", alpha=0.4)
    fig.suptitle("Cora: reconstruction vs k (main k = 1, 3, 6)")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"cora_reconstruction_k_metrics.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_movielens_figure(agg_df):
    g = agg_df[agg_df["dataset"] == "movielens"].sort_values("k")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(g["k"], g["BIC_mean"], "o-", color="#4C72B0")
    axes[0].set_title("BIC vs k"); axes[0].set_xlabel("k")
    axes[1].plot(g["k"], g["BCE_X_mean"], "o-", color="#55A868")
    axes[1].set_title("BCE_X vs k (Bernoulli X)"); axes[1].set_xlabel("k")
    axes[2].plot(g["k"], g["Poisson_NLL_mean"], "o-", color="#C44E52", label="Poisson_NLL")
    ax2 = axes[2].twinx()
    ax2.plot(g["k"], g["Pearson_Y_mean"], "s--", color="#8172B2", label="Pearson_Y")
    axes[2].set_title("Y reconstruction vs k (Poisson Y)"); axes[2].set_xlabel("k")
    for k in ML_MAIN_K:
        for ax in axes:
            ax.axvline(k, color="gray", linestyle=":", alpha=0.4)
    fig.suptitle("MovieLens: reconstruction vs k (main k = 5, 8)")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"movielens_reconstruction_k_metrics.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df = load_summary()
    agg = build_agg(df)
    build_overview(agg)
    build_use_in_notion_plan()

    make_overview_figure(agg)
    make_wine_figure(agg)
    make_cora_figure(agg)
    make_movielens_figure(agg)

    print(agg[["dataset", "k", "BIC_mean", "X_primary_metric_name", "X_primary_metric_mean",
               "Y_primary_metric_name", "Y_primary_metric_mean", "success_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
