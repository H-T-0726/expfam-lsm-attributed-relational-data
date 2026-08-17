"""
MovieLens attribute diagnosis（軽量版）— mixed_percolumn 悪化要因の切り分け。

背景（story diagnostics フェーズ、2026-07-13〜）:
    既存 pilot（run_movielens_mixed_x_percolumn.py, research/per-column-validation
    フェーズ）では mixed_percolumn（genre19 + mean_rating + year + ratings_count
    の22列 per-column）が genre_only より悪化した
    （test_y_ll: mixed_percolumn -3.815 < genre_only -3.423、
     movielens_mixed_x_agg.csv 参照）。

    本スクリプトは、どの属性・どの count 処理が悪化に寄与するかを
    属性を1つずつ足しながら切り分ける診断実験である。
    既存スクリプト・CSV・図・レポートは変更しない（新規ファイルのみ）。

属性構築（既存 run_movielens_mixed_x_percolumn.py の build_attributes() を複製・拡張）:
    genre 19列        : multi-hot（Bernoulli）— 既存 movielens_X_genre.npy
    mean_rating 1列   : ml-100k.zip の u.data から計算し z-score（Gaussian）
    year 1列          : metadata の title から正規表現で抽出し z-score（Gaussian）
    ratings_count 系  : metadata の ratings_count を3通りに処理
        - count_raw : 生値（Poisson、既存 pilot と同じ扱い）
        - count_log : log1p(count) を z-score（Gaussian）
        - count_z   : 生値をそのまま z-score（Gaussian、log変換なし対比用）

⚠ リーク注意（必ず確認・明記）:
    mean_rating と ratings_count は、train/test の pair split より前に
    build_attributes() 内で「全 u.data / 既存メタデータの ratings_count 列」
    から計算される（split 非依存）。Y（共評価カウント行列）も同じ u.data
    由来であり、mean_rating・ratings_count と Y の間には構造的な相関が
    ありうる（情報リークのリスク）。year のみタイトル由来でリークなし。
    今回は train-only 化はしない（診断実験のため、既存 pilot と同じ計算方法を
    踏襲する）。したがって本実験の結果は厳密な汎化性能の証拠ではなく、
    「どの属性・どの処理が悪化に寄与するか」を切り分けるための診断である。

今回のスコープ（軽量版）:
    SPLIT_TRIALS=[0], MODEL_TRIALS=[0]（1 fit/条件）、11条件。
    フル拡張時は SPLIT_TRIALS/MODEL_TRIALS にインデックスを追加するだけでよい。

出力:
    expfam/results/story_diagnostics/movielens_attribute_diagnosis_20260713_smoke.csv
    expfam/results/story_diagnostics/movielens_attribute_diagnosis_20260713_smoke_agg.csv
    expfam/results/story_diagnostics/movielens_attribute_diagnosis_20260713_smoke_runinfo.csv
    figures/story_diagnostics/movielens_attribute_test_y_ll_smoke.png
    figures/story_diagnostics/movielens_count_treatment_comparison_smoke.png

実行: python tools/research_audit/run_movielens_attribute_diagnosis.py
"""

import io
import re
import sys
import time
import zipfile
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "expfam" / "src"))
sys.path.insert(0, str(_ROOT / "expfam" / "src" / "experimental"))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from em_runner import run_em_experimental, predict_mu_y        # noqa: E402
from eval_utils import make_pair_split, heldout_count_metrics  # noqa: E402

DATA_DIR = _ROOT / "expfam" / "data" / "movielens_pilot"
ZIP_PATH = _ROOT / "expfam" / "data" / "ml-100k.zip"
OUT_DIR = _ROOT / "expfam" / "results" / "story_diagnostics"
FIG_DIR = _ROOT / "figures" / "story_diagnostics"

RUN_TAG = "20260713_trials4"
FIG_TAG = "trials4"

K = 3
SPLIT_TRIALS = [0, 1]
MODEL_TRIALS = [0, 1]
TEST_RATIO = 0.2
L, NITER = 5, 8
SPLIT_SEED_BASE = 102000
MODEL_SEED_BASE = 103000
# HIGH_COUNT_THRESHOLD は main() 内で Y の中央値から動的に決定する
# （固定値1にすると全ペアが positive になり AUC/AP が定義不能になるため。
#  この pilot データは genre-stratified top movies のみを含み、Y の最小値が
#  3 で 0 が存在しないことを確認済み）。


def build_attributes():
    """既存 build_attributes() を複製し、count の変換違いを追加。"""
    X_genre = np.load(DATA_DIR / "movielens_X_genre.npy").astype(float)
    Y = np.load(DATA_DIR / "movielens_Y_count.npy").astype(float)
    movie_ids = np.load(DATA_DIR / "movielens_movie_ids.npy")
    meta = pd.read_csv(DATA_DIR / "movielens_movies_metadata.csv")
    meta = meta.set_index("mid").loc[movie_ids]

    # mean rating（u.data: user \t item \t rating \t ts）— 全ログ由来（split非依存）
    with zipfile.ZipFile(ZIP_PATH) as zf:
        with zf.open("ml-100k/u.data") as f:
            ratings = pd.read_csv(io.TextIOWrapper(f, "utf-8"), sep="\t",
                                  names=["user", "item", "rating", "ts"])
    mean_rating = ratings.groupby("item")["rating"].mean()
    mean_rating = mean_rating.reindex(movie_ids).to_numpy()
    if np.any(np.isnan(mean_rating)):
        raise RuntimeError("mean rating missing for some pilot movies")

    # release year（title 末尾の (YYYY)）— タイトル由来、リークなし
    years = np.array([
        float(m.group(1)) if (m := re.search(r"\((\d{4})\)\s*$", t)) else np.nan
        for t in meta["title"]])
    if np.any(np.isnan(years)):
        raise RuntimeError("release year missing for some pilot movies")

    counts = meta["ratings_count"].to_numpy().astype(float)  # 全ログ由来（split非依存）

    def zscore(v):
        return (v - v.mean()) / v.std()

    return dict(
        X_genre=X_genre, Y=Y,
        mean_rating_z=zscore(mean_rating),
        year_z=zscore(years),
        count_raw=counts,
        count_log_z=zscore(np.log1p(counts)),
        count_z=zscore(counts),
        stats=dict(mean_rating_mean=float(np.mean(mean_rating)),
                   year_min=float(years.min()), year_max=float(years.max()),
                   count_mean=float(counts.mean()), count_max=float(counts.max())))


LEAK_CAVEAT = (
    "mean_rating と ratings_count は train/test pair split より前に "
    "build_attributes() 内で全 u.data / 既存メタデータから計算されており、"
    "split に依存しない（leakage risk）。Y（共評価カウント）も同じ u.data 由来。"
    "year のみタイトル由来でリークなし。今回は train-only 化をしていない診断実験であり、"
    "厳密な汎化性能の証拠ではなく悪化要因の切り分けを目的とする。"
)

TRANSFORM_NOTES = {
    "y_only": "X 不使用（fix_x=True）",
    "genre_only": "genre 19列（0/1 生値、Bernoulli 正指定）",
    "genre_year": "genre19(Bernoulli) + year(z-score, Gaussian) = 20列",
    "genre_avg_rating": "genre19(Bernoulli) + mean_rating(z-score, Gaussian) = 20列",
    "genre_count_raw_poisson": "genre19(Bernoulli) + ratings_count(生値, Poisson) = 20列",
    "genre_log_count_gaussian": "genre19(Bernoulli) + log1p(ratings_count)のz-score(Gaussian) = 20列",
    "genre_zscore_count_gaussian": "genre19(Bernoulli) + ratings_count生値のz-score(Gaussian、log変換なし) = 20列",
    "rating_stats_only": "mean_rating(z-score,Gaussian) + year(z-score,Gaussian)"
                         " + ratings_count(生値,Poisson) の3列 per-column",
    "mixed_percolumn_raw": "genre19(Bernoulli) + mean_rating + year(Gaussian) + "
                          "ratings_count(生値,Poisson) = 22列 per-column（既存pilotのmixed_percolumnと同一構成）",
    "mixed_percolumn_log_count": "genre19(Bernoulli) + mean_rating + year(Gaussian) + "
                                "log1p(ratings_count)のz-score(Gaussian) = 22列 per-column"
                                "（countをPoisson生値からGaussian log-zに変更した比較用）",
    "mixed_all_gaussian": "誤指定比較用: mixed_percolumn_rawと同じ22列を生値のまま全列Gaussian強制",
}


def build_conditions(at):
    genre = at["X_genre"]
    conds = {}

    conds["y_only"] = (genre, dict(family_x="gaussian", fix_x=True), {})
    conds["genre_only"] = (genre, dict(family_x="bernoulli"),
                           {"genre": list(range(19))})

    def genre_plus(col, fam_label, block_name):
        X = np.hstack([genre, col.reshape(-1, 1)])
        fam = ["bernoulli"] * 19 + [fam_label]
        return (X, dict(family_x=None, family_x_list=fam),
               {"genre": list(range(19)), block_name: [19]})

    conds["genre_year"] = genre_plus(at["year_z"], "gaussian", "year")
    conds["genre_avg_rating"] = genre_plus(at["mean_rating_z"], "gaussian", "mean_rating")
    conds["genre_count_raw_poisson"] = genre_plus(at["count_raw"], "poisson", "count_raw")
    conds["genre_log_count_gaussian"] = genre_plus(at["count_log_z"], "gaussian", "count_log")
    conds["genre_zscore_count_gaussian"] = genre_plus(at["count_z"], "gaussian", "count_z")

    stats3 = np.column_stack([at["mean_rating_z"], at["year_z"], at["count_raw"]])
    fam3 = ["gaussian", "gaussian", "poisson"]
    conds["rating_stats_only"] = (stats3, dict(family_x=None, family_x_list=fam3),
                                  {"mean_rating": [0], "year": [1], "count_raw": [2]})

    X22_raw = np.hstack([genre, stats3])
    fam22_raw = ["bernoulli"] * 19 + fam3
    blocks22_raw = {"genre": list(range(19)), "mean_rating": [19],
                    "year": [20], "count_raw": [21]}
    conds["mixed_percolumn_raw"] = (X22_raw, dict(family_x=None, family_x_list=fam22_raw),
                                    blocks22_raw)
    conds["mixed_all_gaussian"] = (X22_raw, dict(family_x="gaussian"), blocks22_raw)

    stats3_log = np.column_stack([at["mean_rating_z"], at["year_z"], at["count_log_z"]])
    fam3_log = ["gaussian", "gaussian", "gaussian"]
    X22_log = np.hstack([genre, stats3_log])
    fam22_log = ["bernoulli"] * 19 + fam3_log
    blocks22_log = {"genre": list(range(19)), "mean_rating": [19],
                    "year": [20], "count_log": [21]}
    conds["mixed_percolumn_log_count"] = (X22_log, dict(family_x=None, family_x_list=fam22_log),
                                          blocks22_log)

    return conds


def block_recon_rmse(X_used, col_blocks, res):
    """列グループ（block名→列インデックス）ごとの X 再構成 RMSE。"""
    model = res["model"]
    eta_x = res["Z_est"] @ res["F"].T
    mu_x = model._mean_function_x(eta_x)
    out = {}
    for bname, pos in col_blocks.items():
        resid = X_used[:, pos] - mu_x[:, pos]
        out[f"x_rmse_{bname}"] = float(np.sqrt(np.mean(resid ** 2)))
    return out


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    at = build_attributes()
    n = at["Y"].shape[0]
    conds = build_conditions(at)

    iu = np.triu_indices(n, 1)
    y_upper = at["Y"][iu]
    high_count_threshold = float(np.median(y_upper))
    print(f"=== MovieLens attribute diagnosis ({FIG_TAG}): n={n}, "
          f"{len(conds)} conds x {len(SPLIT_TRIALS)} splits x "
          f"{len(MODEL_TRIALS)} seeds ===")
    print(f"attribute stats: {at['stats']}")
    print(f"Y upper-tri: min={y_upper.min():.0f} median={high_count_threshold:.0f} "
          f"max={y_upper.max():.0f} (high_count_threshold=median)")
    print(f"LEAK CAVEAT: {LEAK_CAVEAT}")

    rows = []
    for split in SPLIT_TRIALS:
        train_mask, test_mask = make_pair_split(
            n, TEST_RATIO, seed=SPLIT_SEED_BASE + split * 100)
        for mt in MODEL_TRIALS:
            for cname, (X_used, kw, col_blocks) in conds.items():
                res = run_em_experimental(
                    X_used, at["Y"], family_y="poisson",
                    k=K, L=L, num_iter=NITER,
                    seed=MODEL_SEED_BASE + split * 10 + mt,
                    train_mask=train_mask, **kw)
                mu_y = predict_mu_y(res)
                m_te = heldout_count_metrics(
                    at["Y"], mu_y, test_mask, "poisson",
                    high_count_threshold=high_count_threshold)
                m_tr = heldout_count_metrics(at["Y"], mu_y, train_mask, "poisson")
                row = {
                    "condition": cname, "split": split, "model_trial": mt,
                    "n_cols_used": 0 if kw.get("fix_x") else X_used.shape[1],
                    "test_y_ll": m_te.get("mean_ll", float("nan")),
                    "test_y_rmse": m_te.get("rmse", float("nan")),
                    "test_spearman": m_te.get("spearman", float("nan")),
                    "test_hc_auc": m_te.get("hc_auc", float("nan")),
                    "test_hc_ap": m_te.get("hc_ap", float("nan")),
                    "train_y_ll": m_tr.get("mean_ll", float("nan")),
                    "w0": res["w0"], "w": res["w"],
                    "nan_occurred": res["nan_occurred"],
                    "runtime_s": res["runtime_s"],
                }
                if col_blocks and not kw.get("fix_x"):
                    row.update(block_recon_rmse(X_used, col_blocks, res))
                rows.append(row)
                print(f"s={split} m={mt} {cname:28s} "
                      f"te_ll={row['test_y_ll']:.3f} "
                      f"te_rmse={row['test_y_rmse']:.2f} "
                      f"auc={row['test_hc_auc']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"movielens_attribute_diagnosis_{RUN_TAG}.csv", index=False)

    agg = df.groupby("condition").agg(
        n_trials=("test_y_ll", "count"),
        test_y_ll_mean=("test_y_ll", "mean"), test_y_ll_std=("test_y_ll", "std"),
        test_y_rmse_mean=("test_y_rmse", "mean"),
        test_spearman_mean=("test_spearman", "mean"),
        test_hc_auc_mean=("test_hc_auc", "mean"),
        test_hc_ap_mean=("test_hc_ap", "mean"),
        n_nan=("nan_occurred", "sum")).reset_index()

    genre_only_ll = float(agg.loc[agg["condition"] == "genre_only", "test_y_ll_mean"].iloc[0])
    agg["test_y_ll_diff_vs_genre_only"] = agg["test_y_ll_mean"] - genre_only_ll
    agg.to_csv(OUT_DIR / f"movielens_attribute_diagnosis_{RUN_TAG}_agg.csv", index=False)
    print("\n=== Aggregated ===")
    print(agg.to_string(index=False))

    def git_head():
        try:
            return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True,
                                  cwd=_ROOT).stdout.strip()
        except Exception:
            return "unknown"

    runinfo = [{
        "script": "tools/research_audit/run_movielens_attribute_diagnosis.py",
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "branch": "research/story-diagnostics",
        "n": n, "k": K, "L": L, "num_iter": NITER,
        "test_ratio": TEST_RATIO,
        "split_trials": len(SPLIT_TRIALS), "model_trials": len(MODEL_TRIALS),
        "split_seed_base": SPLIT_SEED_BASE, "model_seed_base": MODEL_SEED_BASE,
        "high_count_threshold": high_count_threshold,
        "condition": cname, "transform_note": note,
        "leak_caveat": LEAK_CAVEAT,
        "attribute_stats": str(at["stats"]),
        "total_runtime_s": round(time.perf_counter() - t0, 1),
    } for cname, note in TRANSFORM_NOTES.items()]
    pd.DataFrame(runinfo).to_csv(
        OUT_DIR / f"movielens_attribute_diagnosis_{RUN_TAG}_runinfo.csv", index=False)

    make_figures(agg)
    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f} min")


ORDER = ["y_only", "genre_only", "genre_year", "genre_avg_rating",
         "genre_count_raw_poisson", "genre_log_count_gaussian",
         "genre_zscore_count_gaussian", "rating_stats_only",
         "mixed_percolumn_raw", "mixed_percolumn_log_count", "mixed_all_gaussian"]
LABELS = {
    "y_only": "y_only", "genre_only": "genre_only",
    "genre_year": "genre+year", "genre_avg_rating": "genre+avg_rating",
    "genre_count_raw_poisson": "genre+count(raw,Poisson)",
    "genre_log_count_gaussian": "genre+count(log,Gaussian)",
    "genre_zscore_count_gaussian": "genre+count(zscore,Gaussian)",
    "rating_stats_only": "rating_stats_only",
    "mixed_percolumn_raw": "mixed_percolumn(raw count)",
    "mixed_percolumn_log_count": "mixed_percolumn(log count)",
    "mixed_all_gaussian": "mixed_all_gaussian（誤指定）",
}


def make_figures(agg):
    agg_ordered = agg.set_index("condition").reindex(
        [c for c in ORDER if c in agg["condition"].values]).reset_index()

    # 図1: 全条件の test Y ll 比較（横棒）
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    labels = [LABELS.get(c, c) for c in agg_ordered["condition"]]
    colors = ["#C44E52" if c == "genre_only" else
             ("#55A868" if "mixed" in c or c == "rating_stats_only" else "#4C72B0")
             for c in agg_ordered["condition"]]
    ax.barh(labels, agg_ordered["test_y_ll_mean"], color=colors)
    ax.set_xlabel("test Y log-likelihood / pair（高いほど良い）")
    ax.set_title(f"MovieLens attribute diagnosis（{FIG_TAG}, "
                f"{len(SPLIT_TRIALS)*len(MODEL_TRIALS)} fits/条件）")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fname1 = f"movielens_attribute_test_y_ll_{FIG_TAG}.png"
    fig.savefig(FIG_DIR / fname1, dpi=150)
    plt.close(fig)
    print(f"saved {FIG_DIR / fname1}")

    # 図2: count処理の比較のみ抜粋
    count_conds = ["genre_only", "genre_count_raw_poisson",
                  "genre_log_count_gaussian", "genre_zscore_count_gaussian",
                  "mixed_percolumn_raw", "mixed_percolumn_log_count"]
    sub = agg_ordered[agg_ordered["condition"].isin(count_conds)]
    sub = sub.set_index("condition").reindex(count_conds).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([LABELS.get(c, c) for c in sub["condition"]], sub["test_y_ll_mean"],
          color="#4C72B0")
    ax.set_ylabel("test Y log-likelihood / pair（高いほど良い）")
    ax.set_title(f"count属性の処理方法によるtest Y llの違い（{FIG_TAG}）")
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fname2 = f"movielens_count_treatment_comparison_{FIG_TAG}.png"
    fig.savefig(FIG_DIR / fname2, dpi=150)
    plt.close(fig)
    print(f"saved {FIG_DIR / fname2}")


if __name__ == "__main__":
    main()
