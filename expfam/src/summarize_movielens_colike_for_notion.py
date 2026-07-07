"""
MovieLens co-like experiment results をNotion・発表用に要約するスクリプト。

既存の実験 (run_fixed_real_movielens_colike_interpretation.py が生成した
expfam/results/real_data/movielens_colike_interpretation/ 以下のCSV) を
読み込むだけで、新しい実験は一切実行しない。

出力先 (すべて新規):
  reports/movielens_colike_clean/movielens_colike_notion_summary.md
  expfam/results/real_data/movielens_colike_clean/*.csv
  expfam/figures/real_data/movielens_colike_clean/*.png/.pdf

既存の movielens_colike_interpretation/ 以下のCSV・図、および
既存のモデル実装 (model_dual_expfam.py 等) は一切変更しない。
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

INPUT_DIR = os.path.join(ROOT, "expfam", "results", "real_data", "movielens_colike_interpretation")
OUT_CSV_DIR = os.path.join(ROOT, "expfam", "results", "real_data", "movielens_colike_clean")
OUT_FIG_DIR = os.path.join(ROOT, "expfam", "figures", "real_data", "movielens_colike_clean")
OUT_REPORT_DIR = os.path.join(ROOT, "reports", "movielens_colike_clean")

REQUIRED_INPUT_FILES = [
    "movielens_colike_data_summary.csv",
    "movielens_colike_poisson_agg.csv",
    "movielens_colike_lift_heldout_agg.csv",
    "movielens_colike_baseline_metrics.csv",
    "movielens_colike_factor_top_movies.csv",
    "movielens_colike_factor_correlations.csv",
    "movielens_colike_factor_interpretation_summary.csv",
    "movielens_colike_recommendation_examples.csv",
]
OPTIONAL_INPUT_FILES = [
    "movielens_colike_poisson_summary.csv",
    "movielens_colike_bestk_summary.csv",
]

# Three factors judged easiest to interpret (used in main_k_interpretation_table /
# figure 2 / the Notion report). Chosen by inspecting
# movielens_colike_factor_interpretation_summary.csv: these are the three factors
# with the strongest single-attribute |pearson_corr| among the best_k=8 fit, and
# they map onto the three candidate axes the requester suggested
# (classic/highly-rated films, popularity, high-rating-rate/acclaimed-classics).
INTERPRETABLE_FACTORS = [2, 4, 5]
BEST_K_FOR_INTERPRETATION = 8
REPRESENTATIVE_TRIAL = 2

QUERY_MOVIE_TITLES_PRIORITY = [
    "Citizen Kane (1941)",
    "Vertigo (1958)",
    "Star Trek IV: The Voyage Home (1986)",
]
QUERY_MOVIE_LIMIT_CASE = "Mary Poppins (1964)"


def load_inputs():
    loaded = {}
    missing = []
    for fname in REQUIRED_INPUT_FILES + OPTIONAL_INPUT_FILES:
        path = os.path.join(INPUT_DIR, fname)
        if os.path.exists(path):
            loaded[fname] = pd.read_csv(path)
        else:
            missing.append(fname)
    return loaded, missing


def build_main_poisson_table(data):
    df = data["movielens_colike_poisson_agg.csv"].copy()
    interp = {
        2: "次元が少なくシンプルだが、再構成誤差(RMSE_Y)は4条件中で最大",
        3: "ジャンル構造との対応が比較的よい",
        5: "BIC最小でバランスがよい",
        8: "再構成性能(RMSE_Y/Pearson)は最良だが複雑（BICはk=5よりやや悪化）",
    }
    out = pd.DataFrame({
        "k": df["k"],
        "RMSE_Y_mean": df["rmse_y_mean"].round(3),
        "Pearson_mean": df["pearson_mean"].round(3),
        "BIC_mean": df["bic_mean"].round(1),
        "short_interpretation": df["k"].map(interp),
    })
    out.to_csv(os.path.join(OUT_CSV_DIR, "main_poisson_table.csv"), index=False)
    return out


def _short_movie_list(factor_top_movies, factor, side, n=3):
    sub = factor_top_movies[
        (factor_top_movies["factor"] == factor) & (factor_top_movies["side"] == side)
    ].sort_values("rank").head(n)
    return "; ".join(sub["title"].tolist())


def build_main_k_interpretation_table(data):
    fis = data["movielens_colike_factor_interpretation_summary.csv"]
    ftm = data["movielens_colike_factor_top_movies.csv"]

    labels = {
        2: "classic / well-regarded films (older release_year, higher avg_rating)",
        4: "popularity-related (high like_count)",
        5: "high-rating / acclaimed-classics related (high high_rating_rate)",
    }
    evidence = {
        2: "release_year r=-0.52; avg_rating r=0.49; high_rating_rate r=0.47",
        4: "log_like_count r=0.75; like_count r=0.73; high_rating_rate r=0.65",
        5: "high_rating_rate r=0.78; avg_rating r=0.77; like_count r=0.70",
    }
    caution = "潜在空間には回転不定性があるため、factorの意味は確定ではない（暫定的な解釈）。"

    rows = []
    for f in INTERPRETABLE_FACTORS:
        rows.append({
            "factor": f,
            "tentative_label": labels[f],
            "evidence_correlation": evidence[f],
            "top_high_movies_short": _short_movie_list(ftm, f, "high"),
            "top_low_movies_short": _short_movie_list(ftm, f, "low"),
            "caution": caution,
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_CSV_DIR, "main_k_interpretation_table.csv"), index=False)
    return out


def build_supp_lift_baseline_table(data):
    bm = data["movielens_colike_baseline_metrics.csv"]
    agg = bm.groupby("method")[["ap_sampled_mean", "ndcg_at_10_mean"]].first().reset_index()

    interp = {
        "popularity": "人気度だけではlift関係をほとんど説明できない（ランダムよりわずかに上）",
        "genre_cosine": "ベースライン中で最強だが、提案手法の半分以下",
        "popularity_genre": "人気度+ジャンルでも提案手法に届かない",
        "item_item": "評価設計上の床効果（test positiveをtrain score上で0にする）でランダム相当。参考値として記録するが本文では除外",
        "proposed_dual_expfam": "全ベースラインに対しAP_sampled/NDCG@10とも最も高い値を示した",
    }
    include_in_main = {
        "popularity": "yes",
        "genre_cosine": "yes",
        "popularity_genre": "yes",
        "item_item": "no",
        "proposed_dual_expfam": "yes",
    }
    out = pd.DataFrame({
        "method": agg["method"],
        "AP_sampled": agg["ap_sampled_mean"].round(3),
        "NDCG_at_10": agg["ndcg_at_10_mean"].round(3),
        "short_interpretation": agg["method"].map(interp),
        "include_in_main": agg["method"].map(include_in_main),
    })
    order = ["popularity", "genre_cosine", "popularity_genre", "item_item", "proposed_dual_expfam"]
    out["_order"] = out["method"].apply(lambda m: order.index(m) if m in order else 99)
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    out.to_csv(os.path.join(OUT_CSV_DIR, "supp_lift_baseline_table.csv"), index=False)
    return out


def _top5_titles(rec_df, query_title, method):
    sub = rec_df[
        (rec_df["query_title"] == query_title) & (rec_df["method"] == method)
    ].sort_values("rank").head(5)
    return " > ".join(sub["recommended_title"].tolist())


def build_main_recommendation_examples(data):
    rec = data["movielens_colike_recommendation_examples.csv"]

    comments = {
        "Citizen Kane (1941)": (
            "提案手法はVertigoやNorth by Northwestなど、ジャンルラベルだけでは出てこない"
            "強い共高評価関係を持つ古典名作を高く評価している。genre_cosineは同じDramaタグの"
            "映画（Boogie Nights, Good Will Huntingなど）を返すのみで、ジャンルを超えた構造を捉えていない。"
        ),
        "Vertigo (1958)": (
            "提案手法はCitizen Kaneなど質の近い名作群を返す一方、genre_cosineはMystery/Thriller"
            "タグを持つだけの映画（Basic Instinctなど）を返し、必ずしも質的な近さを反映していない。"
        ),
        "Star Trek IV: The Voyage Home (1986)": (
            "genre_cosineは同シリーズ作品（Star Trek系列）を機械的に返すのに対し、提案手法は"
            "シリーズ外の映画（Clear and Present Dangerなど）も高くランクしており、ジャンルラベルを"
            "超えた共高評価構造を示す一方、Star Trek VIのような直接の関連作も依然上位に残る。"
        ),
        "Mary Poppins (1964)": (
            "【限界例】提案手法はVertigoやGandhi、Citizen Kaneのような全く異なるジャンルの名作"
            "ドラマを上位推薦しており、子供向けミュージカルとしての直感とは一致しにくい。人気度・"
            "高評価率に関連する因子がジャンルを問わず強く効いている可能性を示す例で、解釈には注意が必要。"
        ),
    }
    use_in_main = {
        "Citizen Kane (1941)": "yes",
        "Vertigo (1958)": "yes",
        "Star Trek IV: The Voyage Home (1986)": "yes",
        "Mary Poppins (1964)": "yes",
    }

    rows = []
    for q in QUERY_MOVIE_TITLES_PRIORITY + [QUERY_MOVIE_LIMIT_CASE]:
        proposed = _top5_titles(rec, q, "proposed_poisson_colike")
        genre = _top5_titles(rec, q, "genre_cosine")
        if not proposed or not genre:
            continue
        rows.append({
            "query_movie": q,
            "proposed_top5": proposed,
            "genre_cosine_top5": genre,
            "comment": comments.get(q, ""),
            "use_in_main": use_in_main.get(q, "yes"),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_CSV_DIR, "main_recommendation_examples.csv"), index=False)
    return out


def build_claims_and_cautions():
    rows = [
        dict(claim_level="strong",
             statement="Bernoulli X / Poisson Y の実データ適用において、全fit(Poisson 12件+Bernoulli 9件=21件)が成功し、NaN・エラーは発生しなかった",
             evidence="movielens_colike_poisson_summary.csv / movielens_colike_lift_heldout_summary.csv で success=True, nan_occurred=False が全件",
             caution="n=100 subsetでの結果。大規模データでの安定性は未確認"),
        dict(claim_level="strong",
             statement="liftで定義した強い共高評価ペアのランキングにおいて、提案手法は4つのベースライン（popularity, genre cosine, popularity+genre, item-item）全てに対しAP_sampled/NDCG@10で上回った",
             evidence="movielens_colike_baseline_metrics.csv: 提案手法 AP_sampled=0.677,NDCG@10=0.408 / 最良ベースライン(genre_cosine) AP_sampled=0.265,NDCG@10=0.138",
             caution="item-itemベースラインは評価設計上の床効果(test positiveをtrain score上で0にする)があるため比較から除外して判断"),
        dict(claim_level="moderate",
             statement="Bernoulli X / Poisson Y の実データ適用例を示した",
             evidence="MovieLens co-like countデータ(n=100,d=19)で全fit成功",
             caution="n=100 subsetの結果であり、MovieLens全体の結論ではない"),
        dict(claim_level="moderate",
             statement="共高評価人数(Y_colike_count)をPoisson関係としてin-sampleで再構成できた",
             evidence="movielens_colike_poisson_agg.csv: Pearson(Y,Y_hat)=0.907(k=2)~0.963(k=8)",
             caution="in-sample reconstructionであり、strict held-outではない"),
        dict(claim_level="moderate",
             statement="BICで比較するとk=5が最もバランスが良い",
             evidence="movielens_colike_poisson_agg.csv: BIC_mean k=2:31809.5, k=3:29435.1, k=5:28667.3(最小), k=8:28812.1",
             caution="BICはモデル選択指標の一つであり、唯一の基準ではない"),
        dict(claim_level="weak",
             statement="Kの一部因子(factor 2,4,5)は、人気度・高評価率・公開年(古さ)と関連する可能性がある",
             evidence="movielens_colike_factor_correlations.csv: |pearson_corr|>=0.47の属性相関が存在",
             caution="回転不定性があり、n=100の探索的な相関分析に基づく暫定的な解釈"),
        dict(claim_level="weak",
             statement="提案手法の推薦例は、ジャンル類似度だけでは説明できない構造を示している可能性がある",
             evidence="Citizen Kane/Vertigo等の推薦例で、提案手法とgenre_cosineのTop5が大きく異なる",
             caution="少数(4件)の質的な事例観察に基づく。一般化はできない"),
        dict(claim_level="do_not_claim",
             statement="ユーザー個人への映画推薦ができた",
             evidence="なし",
             caution="movie-node projectionでありuser-level推薦ではない"),
        dict(claim_level="do_not_claim",
             statement="商用推薦システムとして使える",
             evidence="なし",
             caution="n=100 subset、held-out評価も限定的(zero-filled edge hiding)"),
        dict(claim_level="do_not_claim",
             statement="Kの意味が完全に同定された",
             evidence="なし",
             caution="回転不定性のためZの各次元の意味は確定できない"),
        dict(claim_level="do_not_claim",
             statement="未知映画ペアの共高評価人数を厳密に予測できた",
             evidence="なし",
             caution="Poisson実験はin-sample reconstructionであり、strict held-outではない"),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_CSV_DIR, "claims_and_cautions.csv"), index=False)
    return out


def build_stability_audit(data):
    """Checks, using ONLY existing outputs, whether K-interpretation is stable
    across trials/seeds. Two different things are checked separately because
    they have very different evidence available:

    1. Whole-Z cluster quality (NMI/ARI/silhouette vs primary_genre) IS saved
       per trial in movielens_colike_poisson_summary.csv (3 trials per k) -> checkable.
    2. Per-factor tentative labels / correlations / top movies are saved ONLY
       for the single representative fit (k=8, trial=2) in
       movielens_colike_factor_top_movies.csv / _factor_correlations.csv -> NOT checkable.
    """
    rows = []

    if "movielens_colike_poisson_summary.csv" in data:
        ps = data["movielens_colike_poisson_summary.csv"]
        for k in [5, BEST_K_FOR_INTERPRETATION]:
            sub = ps[ps["k"] == k]
            if len(sub) == 0:
                continue
            nmi_range = sub["nmi_primary_genre"].max() - sub["nmi_primary_genre"].min()
            ari_range = sub["ari_primary_genre"].max() - sub["ari_primary_genre"].min()
            sil_range = sub["silhouette"].max() - sub["silhouette"].min()
            stable = "roughly stable" if (nmi_range < 0.1 and ari_range < 0.05) else "variable across seeds"
            rows.append({
                "factor_or_label": f"k={k}: whole-Z cluster quality vs primary_genre (NMI/ARI/silhouette)",
                "evidence_type": f"{len(sub)} MCEM trials (different seeds), aggregate Z-quality metrics only (not per-factor)",
                "stability_summary": (
                    f"NMI range={nmi_range:.3f}, ARI range={ari_range:.3f}, "
                    f"silhouette range={sil_range:.3f} across trials"
                ),
                "stable_or_not": stable,
                "caution": "これはZ全体の品質指標であり、個々のfactorの意味(tentative_label)の安定性は示さない",
            })

    rows.append({
        "factor_or_label": "per-factor tentative_label / correlation / top movies (factor 0-7, k=8)",
        "evidence_type": "movielens_colike_factor_top_movies.csv / movielens_colike_factor_correlations.csv は representative fit (k=8, trial=2) の1trialのみで計算されている",
        "stability_summary": "not checkable: 他のtrial(0,1)についてfactor単位の相関・上位映画は保存されていない",
        "stable_or_not": "unknown / not checkable with existing outputs",
        "caution": "K解釈はrepresentative fitに基づくため、複数seedでの安定性確認が必要（今回は実施していない）",
    })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_CSV_DIR, "k_interpretation_stability_audit.csv"), index=False)
    return out


def make_fig_poisson_k_summary(main_poisson_table):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ks = main_poisson_table["k"].astype(str)

    axes[0].bar(ks, main_poisson_table["RMSE_Y_mean"], color="#4C72B0")
    axes[0].set_title("RMSE_Y (lower is better)")
    axes[0].set_xlabel("k")

    axes[1].bar(ks, main_poisson_table["Pearson_mean"], color="#55A868")
    axes[1].set_title("Pearson(Y, Y_hat)")
    axes[1].set_xlabel("k")
    axes[1].set_ylim(0, 1.05)

    axes[2].bar(ks, main_poisson_table["BIC_mean"], color="#C44E52")
    axes[2].set_title("BIC (lower is better)")
    axes[2].set_xlabel("k")

    fig.suptitle("MovieLens co-like Poisson fit: k comparison (in-sample reconstruction)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_FIG_DIR, f"main_poisson_k_summary.{ext}"), dpi=150)
    plt.close(fig)


def make_fig_k_interpretation_summary(main_k_table):
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")
    col_labels = ["factor", "tentative label", "top movies (high side)", "key correlation"]
    cell_text = []
    for _, r in main_k_table.iterrows():
        top_movies = r["top_high_movies_short"]
        if len(top_movies) > 60:
            top_movies = top_movies[:57] + "..."
        cell_text.append([str(r["factor"]), r["tentative_label"], top_movies, r["evidence_correlation"]])
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.8)
    for c in range(len(col_labels)):
        table.auto_set_column_width(c)
    ax.set_title("MovieLens co-like (k=8): 3 most interpretable factors (tentative)", pad=20)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_FIG_DIR, f"main_k_interpretation_summary.{ext}"), dpi=150)
    plt.close(fig)


def make_fig_baseline_comparison(supp_table):
    sub = supp_table[supp_table["include_in_main"] == "yes"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sub))
    width = 0.35
    ax.bar(x - width / 2, sub["AP_sampled"], width, label="AP_sampled", color="#4C72B0")
    ax.bar(x + width / 2, sub["NDCG_at_10"], width, label="NDCG@10", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["method"], rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Lift link prediction: proposed vs. simple baselines\n(item-item excluded; see caution in report)")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_FIG_DIR, f"supp_lift_baseline_comparison.{ext}"), dpi=150)
    plt.close(fig)


def render_markdown_table(df, columns=None):
    if columns is None:
        columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def build_markdown_report(data, missing, main_poisson, main_k, supp_baseline, main_rec):
    ds = data.get("movielens_colike_data_summary.csv")

    n = int(ds["n"].iloc[0]) if ds is not None else "?"
    d = int(ds["d"].iloc[0]) if ds is not None else "?"
    n_users = int(ds["n_users_full_ml100k"].iloc[0]) if ds is not None else "?"
    like_thr = int(ds["like_threshold_rating"].iloc[0]) if ds is not None else "?"
    y_mean = ds["Y_colike_count_mean"].iloc[0] if ds is not None else "?"
    y_max = ds["Y_colike_count_max"].iloc[0] if ds is not None else "?"
    y_density = ds["Y_colike_count_density_pos"].iloc[0] if ds is not None else "?"
    min_support = int(ds["Y_lift_binary_min_support"].iloc[0]) if ds is not None else "?"
    lift_thr = ds["Y_lift_binary_lift_threshold"].iloc[0] if ds is not None else "?"
    lift_density = ds["Y_lift_binary_density"].iloc[0] if ds is not None else "?"
    lift_n_pos = int(ds["Y_lift_binary_n_positive"].iloc[0]) if ds is not None else "?"

    md = []
    md.append("# MovieLens co-like recommendation 実験：Notion用整理サマリー\n")

    md.append("## 1. このページで伝えたいこと\n")
    md.append(
        "MovieLens実験では、映画ジャンルXと共高評価人数Yを用いて、Bernoulli X / Poisson Y "
        "の実データ適用を確認した。また、推定された潜在変数Zの一部が、人気度・高評価率・年代・"
        "ジャンル群と関連している可能性を探索的に確認した。\n"
    )

    md.append("## 2. なぜMovieLensを使うのか\n")
    md.append(
        "- Wine実験: X=化学成分（連続値）, Y=同じカテゴリかどうか（Bernoulli）\n"
        "- MovieLens実験: X=ジャンル0/1（Bernoulli）, Y=両方高評価した人数（カウント）\n"
        "- MovieLensでは、Yを単純な「関係の有無」ではなく**Poissonのカウント関係**として扱える点が"
        "Wine実験と異なる新しい検証ポイントになっている。\n"
    )

    md.append("## 3. データ設計\n")
    md.append(
        f"- node = movie\n"
        f"- X = genre multi-hot, family_x = Bernoulli\n"
        f"- Y_colike_count[i,j] = 映画iとjを両方 rating >= {like_thr} で評価したユーザー数\n"
        f"- family_y = Poisson\n"
        f"- subset = genre_stratified_mp100, n = {n}, d = {d}（MovieLens 100k全体のユーザー数 n_users={n_users}）\n"
        f"- Y_colike_count: mean={y_mean:.2f}, max={y_max:.0f}, density(>0)={y_density:.3f}\n"
        f"- 補助実験: Y_lift_binary[i,j] = 1 if count>=min_support and lift>=threshold\n"
        f"  （lift = observed / expected, expected = like_count_i × like_count_j / n_users）\n"
        f"  選択値: min_support={min_support}, lift_threshold={lift_thr} "
        f"（density={lift_density:.3f}, positive_edges={lift_n_pos}）\n"
    )

    md.append("## 4. 主実験：Poisson co-like count\n")
    md.append(
        "本文で見る指標はRMSE_Y・Pearson・BICの3つに絞る。AP/AUC/NDCG/MAPなどの詳細指標は"
        "補助実験・付録扱いとする。\n\n"
        "**この実験はin-sample再構成であり、未知ペアの共高評価人数を予測できたわけではない。**\n\n"
    )
    md.append(render_markdown_table(main_poisson, ["k", "RMSE_Y_mean", "Pearson_mean", "BIC_mean", "short_interpretation"]))
    md.append("")

    md.append("## 5. Kの解釈\n")
    md.append(
        f"best_k_for_interpretation = {BEST_K_FOR_INTERPRETATION}"
        f"（representative trial={REPRESENTATIVE_TRIAL}）。"
        "全8因子のうち、特に解釈しやすい3因子のみを示す。\n\n"
    )
    md.append(render_markdown_table(
        main_k,
        ["factor", "tentative_label", "evidence_correlation", "top_high_movies_short", "top_low_movies_short"],
    ))
    md.append("\n**注意点：潜在空間には回転不定性があるため、factorの意味は確定ではない。**\n")

    md.append("## 6. 補助実験：lift ranking\n")
    md.append(
        "目的：人気度補正後の強い共高評価ペアを、単純な人気度・ジャンル類似度より上位に識別できるかを確認する。\n\n"
        "item-itemベースラインは評価設計上の床効果（test positiveをtrain score上で0にする必要があるため）"
        "があり、公平な比較として説明が難しいため本文からは外す。\n\n"
    )
    main_supp = supp_baseline[supp_baseline["include_in_main"] == "yes"]
    md.append(render_markdown_table(main_supp, ["method", "AP_sampled", "NDCG_at_10", "short_interpretation"]))
    md.append("")

    md.append("## 7. 推薦例\n")
    for _, r in main_rec.iterrows():
        md.append(f"**query: {r['query_movie']}**\n")
        md.append(f"- 提案手法Top5: {r['proposed_top5']}")
        md.append(f"- genre cosine Top5: {r['genre_cosine_top5']}")
        md.append(f"- コメント: {r['comment']}\n")

    md.append("## 8. 今回言えること\n")
    md.append(
        "- Bernoulli X / Poisson Y の実データ適用例を作れた\n"
        "- 共高評価人数をPoisson関係としてin-sampleで再構成できた\n"
        "- BICではk=5がバランス良かった\n"
        "- Kの一部因子は、人気度・高評価率・年代・ジャンル群と関連する可能性が見られた\n"
        "- liftで定義した強い共高評価ペアについて、提案手法はpopularity/genre baselineより高いランキング性能を示した\n"
    )

    md.append("## 9. まだ言えないこと\n")
    md.append(
        "- ユーザー個人への映画推薦ができたわけではない\n"
        "- 商用推薦システムとして使えるとは言えない\n"
        "- Kの意味が完全に同定されたわけではない\n"
        "- Poisson側はin-sample再構成であり、strict held-outではない\n"
        "- lift rankingもzero-filled edge hidingであり、strict missing-pair CVではない\n"
        "- n=100 subsetの結果であり、MovieLens全体の結論ではない\n"
        "- item-item baselineとの公平な比較にはpair mask対応が必要\n"
    )

    md.append("## 10. 次にやるべきこと\n")
    md.append(
        "1. K解釈はrepresentative fitに基づくため、複数seedでの安定性確認が必要\n"
        "2. MovieLens n=200/300への拡大\n"
        "3. pair mask対応によるstrict held-out\n"
        "4. 公平な推薦ベースライン比較\n"
        "5. Negative Binomial Y\n"
    )

    if missing:
        md.append("\n---\n")
        md.append(f"*(注: 以下の入力ファイルが見つからなかったため、該当する内容は省略されています: {', '.join(missing)})*\n")

    return "\n".join(md)


def main():
    os.makedirs(OUT_CSV_DIR, exist_ok=True)
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUT_REPORT_DIR, exist_ok=True)

    data, missing = load_inputs()
    print(f"Loaded {len(data)} input files; missing: {missing if missing else 'none'}")

    main_poisson = build_main_poisson_table(data)
    main_k = build_main_k_interpretation_table(data)
    supp_baseline = build_supp_lift_baseline_table(data)
    main_rec = build_main_recommendation_examples(data)
    build_claims_and_cautions()
    build_stability_audit(data)

    make_fig_poisson_k_summary(main_poisson)
    make_fig_k_interpretation_summary(main_k)
    make_fig_baseline_comparison(supp_baseline)

    report_md = build_markdown_report(data, missing, main_poisson, main_k, supp_baseline, main_rec)
    report_path = os.path.join(OUT_REPORT_DIR, "movielens_colike_notion_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Wrote report to {report_path}")
    print(f"Missing input files: {missing}")


if __name__ == "__main__":
    main()
