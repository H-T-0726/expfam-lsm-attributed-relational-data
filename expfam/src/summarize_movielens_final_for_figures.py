"""
MovieLens co-like実験の最終Notion素材を整理するスクリプト。

既存の以下のCSVを読み込むだけで、モデルの再学習・再実行は一切行わない:
  expfam/results/real_data/movielens_colike_interpretation/movielens_colike_poisson_agg.csv
  expfam/results/real_data/movielens_colike_interpretation/movielens_colike_factor_interpretation_summary.csv
  expfam/results/real_data/movielens_colike_interpretation/movielens_colike_factor_top_movies.csv
  expfam/results/real_data/movielens_colike_interpretation/movielens_colike_baseline_metrics.csv
  expfam/results/real_data/movielens_colike_interpretation/movielens_colike_bestk_summary.csv

既存の movielens_colike_clean/ は意図的に縮約された「本文用3指標版」であり、
本スクリプトはそれを上書きせず、より完全な指標一式を movielens_final_clean/ に
別途作成する（監査・確認用）。

出力先 (すべて新規):
  expfam/results/real_data/movielens_final_clean/*.csv

図は新規作成しない（既存の movielens_colike_interpretation/ ・
movielens_colike_clean/ の図で必要十分と判断したため。詳細は報告を参照）。

既存のCSV・図・モデル実装は一切変更しない。
"""

import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

INTERP_DIR = os.path.join(ROOT, "expfam", "results", "real_data", "movielens_colike_interpretation")
OUT_CSV_DIR = os.path.join(ROOT, "expfam", "results", "real_data", "movielens_final_clean")

# Three factors judged most interpretable in the prior audit (k=8, representative trial=2).
INTERPRETABLE_FACTORS = [2, 4, 5]
TENTATIVE_LABELS = {
    2: "classic / highly-rated older films",
    4: "broadly liked / high-like-count films",
    5: "high-rating / critically liked classic films",
}


def load_inputs():
    paths = dict(
        poisson_agg=os.path.join(INTERP_DIR, "movielens_colike_poisson_agg.csv"),
        factor_summary=os.path.join(INTERP_DIR, "movielens_colike_factor_interpretation_summary.csv"),
        factor_top_movies=os.path.join(INTERP_DIR, "movielens_colike_factor_top_movies.csv"),
        baseline_metrics=os.path.join(INTERP_DIR, "movielens_colike_baseline_metrics.csv"),
        bestk_summary=os.path.join(INTERP_DIR, "movielens_colike_bestk_summary.csv"),
    )
    data = {}
    missing = []
    for key, path in paths.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            missing.append(path)
    return data, missing


def build_file_inventory():
    rows = [
        dict(file_path="expfam/src/run_fixed_real_movielens_colike_interpretation.py",
             category="script", experiment_type="poisson_colike_main_experiment",
             contains_metrics="yes", contains_figures="yes (generates figures)",
             safe_to_use_in_notion="yes (code, not a result)",
             caution="Main experiment script; generates all CSVs/figures under movielens_colike_interpretation/"),
        dict(file_path="expfam/src/summarize_movielens_colike_for_notion.py",
             category="script", experiment_type="notion_summary_v1",
             contains_metrics="yes", contains_figures="yes",
             safe_to_use_in_notion="yes (code, not a result)",
             caution="Earlier clean-summary script; produced movielens_colike_clean/ (simplified, 3-metric main table)"),
        dict(file_path="expfam/results/real_data/movielens_colike_interpretation/",
             category="poisson_main_experiment", experiment_type="poisson_colike_reconstruction + lift_heldout + factor_interpretation",
             contains_metrics="yes", contains_figures="yes (separate figures dir)",
             safe_to_use_in_notion="yes",
             caution="Full, detailed results (15 CSVs). Source of truth for all aggregate numbers used in this audit"),
        dict(file_path="expfam/figures/real_data/movielens_colike_interpretation/",
             category="poisson_main_experiment", experiment_type="figures (detailed)",
             contains_metrics="n/a", contains_figures="yes",
             safe_to_use_in_notion="yes (some, e.g. poisson_k_metrics; others are detail/toggle-only)",
             caution="poisson_k_metrics.png has 6 panels (BIC,RMSE_Y,Pearson,AP,NMI,runtime) -- most complete k-sweep figure available"),
        dict(file_path="expfam/results/real_data/movielens_colike_clean/",
             category="notion_clean_v1", experiment_type="simplified_main_text_tables",
             contains_metrics="yes (reduced: RMSE_Y/Pearson/BIC only for poisson table)",
             contains_figures="no (CSV only)",
             safe_to_use_in_notion="yes (already designed for Notion main text)",
             caution="Intentionally simplified per earlier task spec; missing MAE/Spearman/AP/NMI/ARI/runtime columns -- "
                     "superseded for audit purposes by movielens_final_clean/, but still usable as-is for the main-text 3-metric table"),
        dict(file_path="expfam/figures/real_data/movielens_colike_clean/",
             category="notion_clean_v1", experiment_type="simplified_main_text_figures",
             contains_metrics="n/a", contains_figures="yes",
             safe_to_use_in_notion="yes",
             caution="main_poisson_k_summary.png (3-panel only), main_k_interpretation_summary.png (3-factor table image), "
                     "supp_lift_baseline_comparison.png (item_item already excluded) -- directly reusable"),
        dict(file_path="reports/movielens_colike_clean/movielens_colike_notion_summary.md",
             category="notion_clean_v1", experiment_type="draft_markdown_report",
             contains_metrics="yes (as prose/tables)", contains_figures="no",
             safe_to_use_in_notion="reference only (user writes Notion body themselves)",
             caution="Earlier draft; user said this task's Notion body will be written independently"),
        dict(file_path="expfam/results/real_data/movielens_final_clean/",
             category="notion_clean_v2_audit", experiment_type="full_metric_tables_for_audit",
             contains_metrics="yes (full schema incl. MAE/Spearman/AP/NMI/ARI/success_rate/runtime)",
             contains_figures="no (CSV only; no new figures created, existing ones judged sufficient)",
             safe_to_use_in_notion="yes",
             caution="New in this task; does not overwrite movielens_colike_clean/"),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_CSV_DIR, "movielens_file_inventory.csv"), index=False)
    return out


def build_main_poisson_table(data):
    df = data["poisson_agg"].copy()
    interp = {
        2: "underfit; weakest reconstruction among tested k",
        3: "improves over k=2 but still below k=5/8 on RMSE/Pearson/AP",
        5: "BIC minimum; balanced complexity vs reconstruction",
        8: "best RMSE_Y/Pearson/AP/BIC among tested k for reconstruction, but BIC is slightly worse than k=5",
    }
    out = pd.DataFrame({
        "k": df["k"],
        "BIC_mean": df["bic_mean"].round(2),
        "RMSE_Y_mean": df["rmse_y_mean"].round(4),
        "MAE_Y_mean": df["mae_y_mean"].round(4),
        "Pearson_mean": df["pearson_mean"].round(4),
        "Spearman_mean": df["spearman_mean"].round(4),
        "high_colike_AP_mean": df["high_colike_ap_mean"].round(4),
        "NMI_mean": df["nmi_mean"].round(4),
        "ARI_mean": df["ari_mean"].round(4),
        "success_rate": df["success_rate"],
        "runtime_mean": df["runtime_mean"].round(2),
        "short_interpretation": df["k"].map(interp),
    })
    out.to_csv(os.path.join(OUT_CSV_DIR, "movielens_main_poisson_table.csv"), index=False)
    return out


def _movies_str(factor_top_movies, factor, side, n=3):
    sub = factor_top_movies[
        (factor_top_movies["factor"] == factor) & (factor_top_movies["side"] == side)
    ].sort_values("rank").head(n)
    return "; ".join(sub["title"].tolist())


def build_k_interpretation_table(data):
    fis = data["factor_summary"]
    ftm = data["factor_top_movies"]

    rows = []
    for f in INTERPRETABLE_FACTORS:
        row = fis[fis["factor"] == f].iloc[0]
        high_movies = _movies_str(ftm, f, "high")
        low_movies = _movies_str(ftm, f, "low")
        rows.append(dict(
            factor=f,
            tentative_label=TENTATIVE_LABELS[f],
            evidence_top_movies=f"high: {high_movies} | low: {low_movies}",
            evidence_correlations=f"{row['strongest_positive_correlations']} (neg: {row['strongest_negative_correlations']})",
            caution="Z has rotation non-identifiability; this is a tentative/suggestive interpretation, not a confirmed semantic axis (k=8, representative trial only).",
        ))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_CSV_DIR, "movielens_main_k_interpretation_table.csv"), index=False)
    return out


def build_supp_lift_baseline_table(data):
    bm = data["baseline_metrics"]
    agg = bm.groupby("method")[[
        "ap_sampled_mean", "auc_sampled_mean", "ap_all_candidates_mean", "auc_all_candidates_mean",
        "precision_at_10_mean", "recall_at_10_mean", "ndcg_at_10_mean", "map_at_10_mean", "hit_rate_at_10_mean",
    ]].first().reset_index()

    interp = {
        "popularity": "weak; popularity alone barely explains lift-defined strong co-like pairs",
        "genre_cosine": "strongest simple baseline, but still far below proposed",
        "popularity_genre": "slightly better than popularity alone, still well below proposed",
        "item_item": "near-random / floor effect by evaluation design (test positives forced to 0 in train score); not a fair generalization measure",
        "proposed_dual_expfam": "highest AP/AUC/NDCG/MAP/hit-rate among all compared methods",
    }
    caution = {
        "popularity": "",
        "genre_cosine": "",
        "popularity_genre": "",
        "item_item": "Do not emphasize in main text -- floor effect from leakage-prevention design (test positives zeroed in train score)",
        "proposed_dual_expfam": "In-sample/zero-filled-edge-hiding evaluation, not strict missing-pair CV",
    }
    order = ["popularity", "genre_cosine", "popularity_genre", "item_item", "proposed_dual_expfam"]

    out = pd.DataFrame({
        "method": agg["method"],
        "AP_sampled_mean": agg["ap_sampled_mean"].round(4),
        "AUC_sampled_mean": agg["auc_sampled_mean"].round(4),
        "AP_all_mean": agg["ap_all_candidates_mean"].round(4),
        "AUC_all_mean": agg["auc_all_candidates_mean"].round(4),
        "precision_at_10_mean": agg["precision_at_10_mean"].round(4),
        "recall_at_10_mean": agg["recall_at_10_mean"].round(4),
        "NDCG_at_10_mean": agg["ndcg_at_10_mean"].round(4),
        "MAP_at_10_mean": agg["map_at_10_mean"].round(4),
        "hit_rate_at_10_mean": agg["hit_rate_at_10_mean"].round(4),
        "short_interpretation": agg["method"].map(interp),
        "caution": agg["method"].map(caution),
    })
    out["_order"] = out["method"].apply(lambda m: order.index(m) if m in order else 99)
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    out.to_csv(os.path.join(OUT_CSV_DIR, "movielens_supp_lift_baseline_table.csv"), index=False)
    return out


def build_use_in_notion_plan():
    rows = [
        dict(item="Poisson k-sweep metrics figure (6-panel)",
             file_path="expfam/figures/real_data/movielens_colike_interpretation/movielens_colike_poisson_k_metrics.png",
             where_to_use="本文：結果1 Poissonカウント関係の再構成", importance="必須",
             caution="BIC/RMSE_Y/Pearson/AP/NMI/runtimeの6パネル。最も網羅的なk-sweep図"),
        dict(item="Poisson main table (movielens_main_poisson_table.csv)",
             file_path="expfam/results/real_data/movielens_final_clean/movielens_main_poisson_table.csv",
             where_to_use="本文：結果1の数値表（フル指標）", importance="必須",
             caution="本文に出すのはRMSE_Y/Pearson/BICの3列で十分（movielens_colike_clean/main_poisson_table.csvでも可）。フル指標は確認・付録用"),
        dict(item="K interpretation table (movielens_main_k_interpretation_table.csv)",
             file_path="expfam/results/real_data/movielens_final_clean/movielens_main_k_interpretation_table.csv",
             where_to_use="本文：結果2 Kの解釈（factor 2,4,5の3つに絞る）", importance="必須",
             caution="回転不定性の注意を必ず添える。全8factorは本文に入れない"),
        dict(item="K interpretation summary image (3-factor)",
             file_path="expfam/figures/real_data/movielens_colike_clean/main_k_interpretation_summary.png",
             where_to_use="本文：結果2 の図", importance="必須",
             caution="既存clean図をそのまま使用可"),
        dict(item="Z by factor (detailed, all 8 factors)",
             file_path="expfam/figures/real_data/movielens_colike_interpretation/movielens_colike_z_by_factor.png",
             where_to_use="トグル・補助（factor詳細）", importance="推奨",
             caution="本文では使わず、詳細を見たい読者向けのトグルに回す"),
        dict(item="Factor correlation heatmap (all 8 factors x attributes)",
             file_path="expfam/figures/real_data/movielens_colike_interpretation/movielens_colike_factor_correlation_heatmap.png",
             where_to_use="トグル・補助（factor詳細）", importance="推奨",
             caution="本文では使わず、詳細を見たい読者向けのトグルに回す"),
        dict(item="Supp lift baseline comparison figure (item-item excluded)",
             file_path="expfam/figures/real_data/movielens_colike_clean/supp_lift_baseline_comparison.png",
             where_to_use="本文後半：結果3 補助ランキング評価", importance="必須",
             caution="item-item baselineは既に除外済み。本文で強調しない方針と一致"),
        dict(item="Supp lift baseline table (movielens_supp_lift_baseline_table.csv)",
             file_path="expfam/results/real_data/movielens_final_clean/movielens_supp_lift_baseline_table.csv",
             where_to_use="本文：結果3の数値表", importance="必須",
             caution="item_item行は表に残すが、本文の文章では強調しない（floor effectの注記つき）"),
        dict(item="Recommendation examples (CSV only, no figure)",
             file_path="expfam/results/real_data/movielens_colike_clean/main_recommendation_examples.csv",
             where_to_use="補足（推薦例として1-2件引用する場合）", importance="任意",
             caution="図は未作成。CSVのみ"),
        dict(item="Y true vs predicted scatter (NOT FOUND)",
             file_path="(does not exist)",
             where_to_use="本文またはトグル候補だったが作成不可", importance="優先度A/Bのギャップとして報告",
             caution="既存保存物には全ペアのY_predが含まれておらず(top-50のみtop_predicted_pairs.csvに存在)、"
                     "代表的な散布図を既存データだけから作ることはできない。新規に作るには再実行が必要"),
        dict(item="Baseline comparison figure WITH item_item (detailed)",
             file_path="expfam/figures/real_data/movielens_colike_interpretation/movielens_colike_baseline_comparison.png",
             where_to_use="使わない（本文では上記item-item除外版を使う）", importance="使用不可（本文では）",
             caution="item_item baselineを含むため、本文用には不適。詳細確認用としてのみ保持"),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_CSV_DIR, "movielens_use_in_notion_plan.csv"), index=False)
    return out


def main():
    os.makedirs(OUT_CSV_DIR, exist_ok=True)

    data, missing = load_inputs()
    print(f"Loaded {len(data)} input files; missing: {missing if missing else 'none'}")

    inventory = build_file_inventory()
    poisson_table = build_main_poisson_table(data)
    k_table = build_k_interpretation_table(data)
    baseline_table = build_supp_lift_baseline_table(data)
    build_use_in_notion_plan()

    print("\nPoisson main table:\n", poisson_table[["k", "BIC_mean", "RMSE_Y_mean", "Pearson_mean", "high_colike_AP_mean"]])
    print("\nK interpretation table:\n", k_table[["factor", "tentative_label"]])
    print("\nBaseline table:\n", baseline_table[["method", "AP_sampled_mean", "NDCG_at_10_mean"]])


if __name__ == "__main__":
    main()
