"""
MovieLens mixed-X per-column pilot（Priority 6、小規模）。

目的:
    実データでジャンル以外の属性（平均評価・公開年・評価件数）を
    per-column family で追加できるかの実現性確認（pilot）。
    人工実験が主証拠であり、本実験は補助。

属性（n=100 映画、既存 movielens_pilot データから構築、新規データファイルは作らない）:
    genre 19 列        : multi-hot（Bernoulli）— 既存 movielens_X_genre.npy
    mean_rating 1 列   : ml-100k.zip の u.data から計算し z-score（Gaussian）
    year 1 列          : metadata の title から正規表現で抽出し z-score（Gaussian）
    ratings_count 1 列 : metadata の ratings_count 生値（Poisson）

⚠ 既知の注意点（レポートにも記載）:
    - X モデルに切片がない（η = f_l^T z_i）ため、Gaussian 属性は z-score 必須。
      Poisson の ratings_count（平均 ≈ 180）は η≈5 を F と Z だけで
      作る必要があり、モデル仕様上不利。挙動をそのまま正直に記録する。
    - ratings_count は Y（共評価カウント）と同じ評価ログ由来であり、
      情報リークの懸念がある（Y の行和と強相関）。予測改善が出ても
      額面通り受け取れない。
    - mixed_all_gaussian / mixed_all_bernoulli は生値のまま単一 family を
      強制する「比較用の誤指定モデル」（自然なモデルではない）。

条件（k=3、split 2 × model seed 2 = 4 fits/条件）:
    y_only / genre_only / rating_stats_only / mixed_percolumn /
    mixed_all_gaussian / mixed_all_bernoulli

評価: strict held-out（pair mask、test 20%）の test 対数尤度（Poisson, /pair）・
      RMSE・Spearman（run_movielens_strict_heldout.py と同手続き）。

出力:
    expfam/results/per_column_family/movielens_mixed_x_summary.csv
    expfam/results/per_column_family/movielens_mixed_x_agg.csv
    expfam/results/per_column_family/movielens_mixed_x_runinfo.csv

実行: python tools/research_audit/run_movielens_mixed_x_percolumn.py
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

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "expfam" / "src"))
sys.path.insert(0, str(_ROOT / "expfam" / "src" / "experimental"))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from em_runner import run_em_experimental, predict_mu_y        # noqa: E402
from eval_utils import make_pair_split, heldout_count_metrics  # noqa: E402

DATA_DIR = _ROOT / "expfam" / "data" / "movielens_pilot"
ZIP_PATH = _ROOT / "expfam" / "data" / "ml-100k.zip"
OUT_DIR = _ROOT / "expfam" / "results" / "per_column_family"

K = 3
SPLIT_TRIALS = [0, 1]
MODEL_TRIALS = [0, 1]
TEST_RATIO = 0.2
L, NITER = 5, 8
SPLIT_SEED_BASE = 92000
MODEL_SEED_BASE = 93000


def build_attributes():
    """既存データ + zip から属性行列を構築（ファイルは書き出さない）。"""
    X_genre = np.load(DATA_DIR / "movielens_X_genre.npy").astype(float)
    Y = np.load(DATA_DIR / "movielens_Y_count.npy").astype(float)
    movie_ids = np.load(DATA_DIR / "movielens_movie_ids.npy")
    meta = pd.read_csv(DATA_DIR / "movielens_movies_metadata.csv")
    meta = meta.set_index("mid").loc[movie_ids]          # npy と同順に整列

    # mean rating（u.data: user \t item \t rating \t ts）
    with zipfile.ZipFile(ZIP_PATH) as zf:
        with zf.open("ml-100k/u.data") as f:
            ratings = pd.read_csv(io.TextIOWrapper(f, "utf-8"), sep="\t",
                                  names=["user", "item", "rating", "ts"])
    mean_rating = ratings.groupby("item")["rating"].mean()
    mean_rating = mean_rating.reindex(movie_ids).to_numpy()
    if np.any(np.isnan(mean_rating)):
        raise RuntimeError("mean rating missing for some pilot movies")

    # release year（title 末尾の (YYYY)）
    years = np.array([
        float(m.group(1)) if (m := re.search(r"\((\d{4})\)\s*$", t)) else np.nan
        for t in meta["title"]])
    if np.any(np.isnan(years)):
        raise RuntimeError("release year missing for some pilot movies")

    counts = meta["ratings_count"].to_numpy().astype(float)

    def zscore(v):
        return (v - v.mean()) / v.std()

    return dict(
        X_genre=X_genre, Y=Y,
        mean_rating_z=zscore(mean_rating),
        year_z=zscore(years),
        ratings_count=counts,
        stats=dict(mean_rating_mean=float(np.mean(mean_rating)),
                   year_min=float(years.min()), year_max=float(years.max()),
                   count_mean=float(counts.mean()),
                   count_max=float(counts.max())))


TRANSFORM_NOTES = {
    "y_only": "X 不使用（fix_x=True）",
    "genre_only": "genre 19列（0/1 生値、Bernoulli 正指定）",
    "rating_stats_only": "mean_rating(z-score, Gaussian) + year(z-score, Gaussian)"
                         " + ratings_count(生値, Poisson) の 3 列 per-column",
    "mixed_percolumn": "genre 19列(Bernoulli) + 上記 3 列 = 22 列 per-column",
    "mixed_all_gaussian": "誤指定比較用: 同じ 22 列を生値のまま全列 Gaussian 強制",
    "mixed_all_bernoulli": "誤指定比較用: 同じ 22 列を生値のまま全列 Bernoulli 強制"
                           "（z-score 値・カウントに Bernoulli 尤度を適用する誤用の模擬）",
}


def build_conditions(at):
    stats3 = np.column_stack([at["mean_rating_z"], at["year_z"],
                              at["ratings_count"]])
    fam3 = ["gaussian", "gaussian", "poisson"]
    X22 = np.hstack([at["X_genre"], stats3])
    fam22 = ["bernoulli"] * 19 + fam3
    return {
        "y_only": (at["X_genre"], dict(family_x="gaussian", fix_x=True)),
        "genre_only": (at["X_genre"], dict(family_x="bernoulli")),
        "rating_stats_only": (stats3, dict(family_x=None,
                                           family_x_list=fam3)),
        "mixed_percolumn": (X22, dict(family_x=None, family_x_list=fam22)),
        "mixed_all_gaussian": (X22, dict(family_x="gaussian")),
        "mixed_all_bernoulli": (X22, dict(family_x="bernoulli")),
    }


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    at = build_attributes()
    n = at["Y"].shape[0]
    conds = build_conditions(at)
    print(f"=== MovieLens mixed-X pilot: n={n}, "
          f"{len(conds)} conds x {len(SPLIT_TRIALS)} splits x "
          f"{len(MODEL_TRIALS)} seeds ===")
    print(f"attribute stats: {at['stats']}")

    rows = []
    for split in SPLIT_TRIALS:
        train_mask, test_mask = make_pair_split(
            n, TEST_RATIO, seed=SPLIT_SEED_BASE + split * 100)
        for mt in MODEL_TRIALS:
            for cname, (X_used, kw) in conds.items():
                res = run_em_experimental(
                    X_used, at["Y"], family_y="poisson",
                    k=K, L=L, num_iter=NITER,
                    seed=MODEL_SEED_BASE + split * 10 + mt,
                    train_mask=train_mask, **kw)
                mu_y = predict_mu_y(res)
                m_te = heldout_count_metrics(at["Y"], mu_y, test_mask,
                                             "poisson")
                m_tr = heldout_count_metrics(at["Y"], mu_y, train_mask,
                                             "poisson")
                rows.append({
                    "condition": cname, "split": split, "model_trial": mt,
                    "n_cols_used": X_used.shape[1] if not kw.get("fix_x")
                                   else 0,
                    "test_y_ll": m_te.get("mean_ll", float("nan")),
                    "test_y_rmse": m_te.get("rmse", float("nan")),
                    "test_spearman": m_te.get("spearman", float("nan")),
                    "train_y_ll": m_tr.get("mean_ll", float("nan")),
                    "w0": res["w0"], "w": res["w"],
                    "nan_occurred": res["nan_occurred"],
                    "runtime_s": res["runtime_s"],
                })
                print(f"s={split} m={mt} {cname:20s} "
                      f"te_ll={rows[-1]['test_y_ll']:.3f} "
                      f"te_rmse={rows[-1]['test_y_rmse']:.2f} "
                      f"sp={rows[-1]['test_spearman']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "movielens_mixed_x_summary.csv", index=False)

    agg = df.groupby("condition").agg(
        test_y_ll_mean=("test_y_ll", "mean"), test_y_ll_std=("test_y_ll", "std"),
        test_y_rmse_mean=("test_y_rmse", "mean"),
        test_spearman_mean=("test_spearman", "mean"),
        n_nan=("nan_occurred", "sum")).reset_index()
    agg.to_csv(OUT_DIR / "movielens_mixed_x_agg.csv", index=False)
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
        "script": "tools/research_audit/run_movielens_mixed_x_percolumn.py",
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "branch": "research/per-column-validation",
        "n": n, "k": K, "L": L, "num_iter": NITER,
        "test_ratio": TEST_RATIO,
        "split_trials": len(SPLIT_TRIALS), "model_trials": len(MODEL_TRIALS),
        "split_seed_base": SPLIT_SEED_BASE, "model_seed_base": MODEL_SEED_BASE,
        "condition": cname, "transform_note": note,
        "leak_caveat": "ratings_count は Y（共評価カウント）と同じ評価ログ由来"
                       "（リーク懸念、レポート参照）",
        "attribute_stats": str(at["stats"]),
        "total_runtime_s": round(time.perf_counter() - t0, 1),
    } for cname, note in TRANSFORM_NOTES.items()]
    pd.DataFrame(runinfo).to_csv(OUT_DIR / "movielens_mixed_x_runinfo.csv",
                                 index=False)
    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
