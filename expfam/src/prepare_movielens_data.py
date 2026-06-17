"""
MovieLens 100K データ準備スクリプト — モデル学習なし、データ分析のみ。

入力: expfam/data/ml-100k.zip
処理:
  1. u.data / u.item / u.genre を読み込み
  2. 3つの subset 候補を作成・比較
     A: top_rated_100       — 評価数上位 100 本
     B: mid_popular_100     — 評価数 30〜200 の 100 本
     C: genre_stratified_mp — 主要 10 ジャンルから各 10 本、評価数 30〜200 優先
  3. 各 subset で X (genre multi-hot) / Y_count (co-rated count) を計算
  4. density / mean / var / overdispersion / threshold 密度を集計
  5. 最良 subset の npy を保存

出力:
  expfam/data/movielens_pilot/
  expfam/results/real_data/movielens_data_prep/
  expfam/figures/real_data/movielens_data_prep/
"""

import sys
import zipfile
import io
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

_SRC  = Path(__file__).parent
_ROOT = _SRC.parent.parent

ZIP_PATH = _ROOT / "expfam" / "data" / "ml-100k.zip"
OUT_DATA = _ROOT / "expfam" / "data" / "movielens_pilot"
OUT_PREP = _ROOT / "expfam" / "results" / "real_data" / "movielens_data_prep"
FIG_DIR  = _ROOT / "expfam" / "figures" / "real_data" / "movielens_data_prep"

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]
# ジャンルインデックス
GENRE_IDX = {g: i for i, g in enumerate(GENRES)}

TARGET_GENRES = [
    "Drama", "Comedy", "Action", "Thriller", "Romance",
    "Adventure", "Crime", "Sci-Fi", "Horror", "Mystery",
]
PER_GENRE   = 10
N_TARGET    = len(TARGET_GENRES) * PER_GENRE    # 100
MIN_RATINGS = 30
MAX_RATINGS = 200
THRESHOLDS  = [1, 5, 10, 20, 30, 50]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def load_movielens():
    """ZIP から u.data / u.item を読み込む。"""
    assert ZIP_PATH.exists(), f"Not found: {ZIP_PATH}"

    with zipfile.ZipFile(ZIP_PATH) as zf:
        # ratings
        with zf.open("ml-100k/u.data") as f:
            raw = f.read().decode("latin-1")
        ratings_rows = []
        for line in raw.strip().split("\n"):
            p = line.strip().split("\t")
            if len(p) >= 3:
                ratings_rows.append((int(p[0]), int(p[1]), int(p[2])))
        ratings_df = pd.DataFrame(ratings_rows, columns=["uid", "mid", "rating"])

        # items (movie meta + genre flags)
        with zf.open("ml-100k/u.item") as f:
            raw = f.read().decode("latin-1")
        item_rows = []
        for line in raw.strip().split("\n"):
            p = line.split("|")
            if len(p) < 24:
                continue
            mid   = int(p[0])
            title = p[1]
            flags = [int(p[i + 5]) for i in range(19)]
            item_rows.append([mid, title] + flags)
        cols = ["mid", "title"] + GENRES
        items_df = pd.DataFrame(item_rows, columns=cols)

    print(f"Ratings: {len(ratings_df)}  "
          f"Users: {ratings_df['uid'].nunique()}  "
          f"Movies: {ratings_df['mid'].nunique()}")
    print(f"Items in u.item: {len(items_df)}")
    print(f"Rating range: {ratings_df['rating'].min()} to {ratings_df['rating'].max()}")
    return ratings_df, items_df


# ─────────────────────────────────────────────────────────────────────
# Subset selection
# ─────────────────────────────────────────────────────────────────────

def select_top_rated(items_df, rpm, n=100):
    """Strategy A: 評価数上位 n 本。"""
    rpm_s  = rpm.sort_values(ascending=False)
    valid  = [m for m in rpm_s.index if m in set(items_df["mid"])]
    chosen = valid[:n]
    return items_df[items_df["mid"].isin(chosen)].copy()


def select_mid_popular(items_df, rpm, min_r=MIN_RATINGS, max_r=MAX_RATINGS, n=100):
    """Strategy B: 評価数 min_r〜max_r の中から n 本（評価数降順）。"""
    mask   = (rpm >= min_r) & (rpm <= max_r)
    rpm_m  = rpm[mask].sort_values(ascending=False)
    valid  = [m for m in rpm_m.index if m in set(items_df["mid"])]
    chosen = valid[:n]
    return items_df[items_df["mid"].isin(chosen)].copy()


def select_genre_stratified(items_df, rpm,
                             target_genres=TARGET_GENRES,
                             per_genre=PER_GENRE,
                             min_r=MIN_RATINGS, max_r=MAX_RATINGS):
    """
    Strategy C: 主要ジャンルから各 per_genre 本。
    - 各ジャンル内で min_r〜max_r の映画を評価数降順でソート
    - 同映画は 1 回だけ選ぶ
    - 足りない場合は範囲を 20〜300 に拡張
    """
    selected_ids = set()
    genre_counts = {}

    for genre in target_genres:
        gi    = GENRE_IDX[genre]
        # そのジャンルを持つ映画
        genre_movies = items_df[items_df[genre] == 1]["mid"].tolist()

        # 評価数でフィルタ（mid_popular 条件、既選択を除く）
        for min_r_try, max_r_try in [
            (min_r, max_r),
            (20, 300),
            (10, 500),
        ]:
            cands = [
                m for m in genre_movies
                if m not in selected_ids
                and m in rpm.index
                and min_r_try <= rpm[m] <= max_r_try
            ]
            cands_sorted = sorted(cands, key=lambda m: rpm[m], reverse=True)
            # 評価数でほぼ中央付近を優先（上位 top 40%〜60% 相当を目指す）
            # ただし pilot なので簡単にメディアン付近を取る
            # 実装: 上位から PER_GENRE 本
            chosen_g = cands_sorted[:per_genre]
            if len(chosen_g) > 0:
                for m in chosen_g:
                    selected_ids.add(m)
                genre_counts[genre] = len(chosen_g)
                break
        else:
            genre_counts[genre] = 0

    print(f"  Genre-stratified selection counts: {genre_counts}")
    return items_df[items_df["mid"].isin(selected_ids)].copy(), genre_counts


# ─────────────────────────────────────────────────────────────────────
# X and Y computation
# ─────────────────────────────────────────────────────────────────────

def compute_X(subset_df):
    """X = genre multi-hot (n × 19)."""
    X = subset_df[GENRES].values.astype(np.float32)
    return X


def compute_Y_count(subset_df, ratings_df):
    """
    Y_count[i,j] = 共通評価ユーザー数 (対称行列, 対角=0)。
    """
    movie_ids  = subset_df["mid"].tolist()
    n          = len(movie_ids)
    mid_to_idx = {m: i for i, m in enumerate(movie_ids)}

    # movie -> users set
    movie_users = defaultdict(set)
    for _, row in ratings_df[ratings_df["mid"].isin(set(movie_ids))].iterrows():
        movie_users[row["mid"]].add(row["uid"])

    Y = np.zeros((n, n), dtype=np.int32)
    for i, mi in enumerate(movie_ids):
        for j in range(i + 1, n):
            mj = movie_ids[j]
            cnt = len(movie_users[mi] & movie_users[mj])
            Y[i, j] = cnt
            Y[j, i] = cnt
    return Y


def primary_genre_label(subset_df):
    """各映画の primary genre（最初の 1 が立つジャンルのインデックス）。"""
    labels = []
    for _, row in subset_df.iterrows():
        flags = [row[g] for g in GENRES]
        idx   = next((i for i, v in enumerate(flags) if v == 1), 0)
        labels.append(idx)
    return np.array(labels, dtype=int)


# ─────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────

def y_count_stats(Y, rpm_subset):
    """Y_count 行列の統計を辞書で返す。"""
    n     = len(Y)
    upper = Y[np.triu_indices(n, k=1)]

    density_pos   = float((upper > 0).mean())
    mean_all      = float(upper.mean())
    mean_nonzero  = float(upper[upper > 0].mean()) if (upper > 0).any() else 0.0
    std_nonzero   = float(upper[upper > 0].std())  if (upper > 0).any() else 0.0
    max_count     = int(upper.max())
    var_all       = float(upper.var())
    var_nonzero   = float(upper[upper > 0].var())  if (upper > 0).any() else 0.0
    vom_all       = var_all    / mean_all      if mean_all > 0 else float("nan")
    vom_nonzero   = var_nonzero / mean_nonzero if mean_nonzero > 0 else float("nan")

    quantiles = {
        "q50": float(np.percentile(upper, 50)),
        "q75": float(np.percentile(upper, 75)),
        "q90": float(np.percentile(upper, 90)),
        "q95": float(np.percentile(upper, 95)),
        "q99": float(np.percentile(upper, 99)),
    }

    # threshold densities
    thr_dens = {}
    for t in THRESHOLDS:
        d = float((upper >= t).mean())
        thr_dens[t] = d

    # ratings_per_movie stats for subset
    rpm_vals = rpm_subset.values
    return {
        "n":              n,
        "n_pairs":        len(upper),
        "density_pos":    density_pos,
        "mean_all":       mean_all,
        "mean_nonzero":   mean_nonzero,
        "std_nonzero":    std_nonzero,
        "max_count":      max_count,
        "var_over_mean_all":     vom_all,
        "var_over_mean_nonzero": vom_nonzero,
        "quantiles":      quantiles,
        "threshold_density": thr_dens,
        "rpm_mean":  float(rpm_vals.mean()),
        "rpm_std":   float(rpm_vals.std()),
        "rpm_min":   int(rpm_vals.min()),
        "rpm_max":   int(rpm_vals.max()),
    }


def genre_dist(subset_df):
    """ジャンル別本数。"""
    dist = {}
    for g in GENRES:
        dist[g] = int(subset_df[g].sum())
    return dist


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────

def save_fig(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = FIG_DIR / f"{stem}.{ext}"
        if p.exists():
            print(f"  SKIP (exists): {p}")
        else:
            fig.savefig(p, dpi=200, bbox_inches="tight")
            print(f"  Saved: {p}")
    plt.close(fig)


def make_y_count_hist(Y_by_strategy, strat_labels):
    n_s  = len(strat_labels)
    fig, axes = plt.subplots(1, n_s, figsize=(6.0 * n_s, 4.5))
    if n_s == 1:
        axes = [axes]

    for ax, (strat, Y), label in zip(axes, Y_by_strategy.items(), strat_labels):
        n = len(Y)
        upper = Y[np.triu_indices(n, k=1)].flatten()
        ax.hist(upper[upper > 0], bins=40, color="#2196F3", alpha=0.75, edgecolor="white")
        ax.set_xlabel("Co-rated count (Y > 0 only)", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.set_title(f"{strat}\n(n={n}, density_pos={float((upper>0).mean()):.3f})", fontsize=9)
        ax.axvline(float(upper.mean()), color="#FF5722", linestyle="--",
                   linewidth=1.4, label=f"mean={upper.mean():.1f}")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("MovieLens Co-rated Count Distribution by Subset Strategy", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "movielens_y_count_hist_by_strategy")


def make_threshold_density_figure(thr_df):
    """strategy × threshold → density の折れ線図。"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors  = ["#2196F3", "#FF5722", "#4CAF50"]
    strategies = thr_df["strategy"].unique()

    for strat, color in zip(strategies, colors):
        sdf = thr_df[thr_df["strategy"] == strat]
        ths = sdf["threshold"].values
        ds  = sdf["density"].values
        ax.plot(ths, ds, "o-", color=color, linewidth=2, markersize=7, label=strat)

    ax.axhline(0.05, color="gray", linestyle=":", linewidth=1.2, label="density=0.05")
    ax.axhline(0.20, color="gray", linestyle="--", linewidth=1.2, label="density=0.20")
    ax.set_xlabel("Threshold T (Y_binary = 1 if count ≥ T)")
    ax.set_ylabel("Y_binary density")
    ax.set_title("Threshold vs. Y_binary Density by Subset Strategy", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "movielens_threshold_density_by_strategy")


def make_genre_distribution_figure(genre_df):
    """strategy × genre の本数バー比較。"""
    strategies = genre_df["strategy"].unique()
    genres_show = [g for g in GENRES if g != "unknown"]
    x    = np.arange(len(genres_show))
    width = 0.25
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    for si, (strat, color) in enumerate(zip(strategies, colors)):
        row  = genre_df[genre_df["strategy"] == strat].iloc[0]
        vals = [int(row.get(g, 0)) for g in genres_show]
        ax.bar(x + si * width, vals, width, label=strat, color=color, alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([g[:8] for g in genres_show], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Count")
    ax.set_title("Genre Distribution by Subset Strategy", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "movielens_genre_distribution_by_strategy")


# ─────────────────────────────────────────────────────────────────────
# Save npy arrays
# ─────────────────────────────────────────────────────────────────────

def save_subset_arrays(subset_df, Y, strategy_name):
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    X      = compute_X(subset_df)
    labels = primary_genre_label(subset_df)
    ids    = np.array(subset_df["mid"].tolist())

    for fname, arr in [
        ("movielens_X_genre.npy",             X),
        ("movielens_Y_count.npy",              Y.astype(np.float32)),
        ("movielens_movie_ids.npy",            ids),
        ("movielens_primary_genre_labels.npy", labels),
    ]:
        p = OUT_DATA / fname
        if not p.exists():
            np.save(p, arr)
            print(f"  Saved: {p}  shape={arr.shape}")

    # threshold binary
    for t in [10, 20]:
        Y_bin = (Y >= t).astype(np.float32)
        np.fill_diagonal(Y_bin, 0.0)
        p = OUT_DATA / f"movielens_Y_binary_t{t}.npy"
        if not p.exists():
            np.save(p, Y_bin)
            print(f"  Saved: {p}  density={(Y_bin[np.triu_indices(len(Y_bin),k=1)]>0).mean():.4f}")

    # metadata CSV
    p = OUT_DATA / "movielens_movies_metadata.csv"
    if not p.exists():
        meta = subset_df[["mid", "title"] + GENRES].copy()
        meta["primary_genre"] = [GENRES[l] for l in labels]
        meta["ratings_count"] = [len(set()) for _ in range(len(meta))]   # placeholder
        meta.to_csv(p, index=False)
        print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT_PREP.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────
    print("=== Loading MovieLens 100K ===")
    ratings_df, items_df = load_movielens()

    rpm = ratings_df.groupby("mid")["rating"].count().rename("n_ratings")

    # ratings summary
    print(f"\nRatings/user: min={ratings_df.groupby('uid').size().min()} "
          f"max={ratings_df.groupby('uid').size().max()} "
          f"mean={ratings_df.groupby('uid').size().mean():.1f}")
    print(f"Ratings/movie: min={rpm.min()} max={rpm.max()} mean={rpm.mean():.1f} "
          f"median={rpm.median():.0f}")

    # ratings metadata CSV
    p = OUT_PREP / "movielens_ratings_metadata.csv"
    if not p.exists():
        meta = pd.DataFrame({
            "total_users":   [ratings_df["uid"].nunique()],
            "total_movies":  [ratings_df["mid"].nunique()],
            "total_ratings": [len(ratings_df)],
            "rating_min":    [int(ratings_df["rating"].min())],
            "rating_max":    [int(ratings_df["rating"].max())],
            "rating_mean":   [float(ratings_df["rating"].mean())],
            "ratings_per_user_mean":  [float(ratings_df.groupby("uid").size().mean())],
            "ratings_per_user_max":   [int(ratings_df.groupby("uid").size().max())],
            "ratings_per_movie_mean": [float(rpm.mean())],
            "ratings_per_movie_median": [float(rpm.median())],
            "ratings_per_movie_max":  [int(rpm.max())],
        })
        meta.to_csv(p, index=False)
        print(f"\nSaved: {p}")

    # ── Subsets ────────────────────────────────────────────────────
    print("\n=== Building subsets ===")

    sub_A = select_top_rated(items_df, rpm, n=N_TARGET)
    print(f"Strategy A (top_rated_100): n={len(sub_A)}")

    sub_B = select_mid_popular(items_df, rpm, MIN_RATINGS, MAX_RATINGS, N_TARGET)
    print(f"Strategy B (mid_popular_100): n={len(sub_B)}")

    sub_C, gc = select_genre_stratified(items_df, rpm, TARGET_GENRES, PER_GENRE, MIN_RATINGS, MAX_RATINGS)
    print(f"Strategy C (genre_stratified): n={len(sub_C)}")

    subsets = {
        "top_rated_100":          sub_A,
        "mid_popular_100":        sub_B,
        "genre_stratified_mp100": sub_C,
    }

    # ── Y_count ───────────────────────────────────────────────────
    print("\n=== Computing Y_count matrices (may take a minute) ===")
    Y_by_strategy = {}
    stats_by_strategy = {}
    for name, sub in subsets.items():
        print(f"  {name} ...", end=" ", flush=True)
        Y = compute_Y_count(sub, ratings_df)
        Y_by_strategy[name] = Y
        rpm_sub = rpm[rpm.index.isin(sub["mid"].tolist())]
        st = y_count_stats(Y, rpm_sub)
        stats_by_strategy[name] = st
        n = len(sub)
        upper = Y[np.triu_indices(n, k=1)]
        print(f"density_pos={st['density_pos']:.4f}  "
              f"mean_all={st['mean_all']:.2f}  "
              f"var/mean={st['var_over_mean_all']:.1f}  "
              f"max={st['max_count']}")

    # ── Subset summary CSV ─────────────────────────────────────────
    subset_rows = []
    for name, sub in subsets.items():
        st = stats_by_strategy[name]
        gd = genre_dist(sub)
        t_dens = {f"density_t{t}": st["threshold_density"][t] for t in THRESHOLDS}
        n_genres_per = sub[GENRES].sum(axis=1)
        # adoption judgement
        if st["density_pos"] > 0.90:
            adopt = "no (too dense)"
        elif st["density_pos"] < 0.10:
            adopt = "no (too sparse)"
        elif st["var_over_mean_all"] > 50:
            adopt = "caution (high overdispersion)"
        else:
            adopt = "yes"

        row = {
            "strategy":               name,
            "n":                      st["n"],
            "d_genre":                19,
            "rpm_mean":               round(st["rpm_mean"], 1),
            "rpm_std":                round(st["rpm_std"], 1),
            "rpm_min":                st["rpm_min"],
            "rpm_max":                st["rpm_max"],
            "genre_count_per_movie_mean": round(float(n_genres_per.mean()), 2),
            "Y_density_pos":          round(st["density_pos"], 4),
            "Y_mean_all_pairs":       round(st["mean_all"], 2),
            "Y_mean_nonzero":         round(st["mean_nonzero"], 2),
            "Y_max":                  st["max_count"],
            "var_over_mean_all":      round(st["var_over_mean_all"], 2),
            "var_over_mean_nonzero":  round(st["var_over_mean_nonzero"], 2),
            "q50": st["quantiles"]["q50"], "q75": st["quantiles"]["q75"],
            "q90": st["quantiles"]["q90"], "q95": st["quantiles"]["q95"],
            "adopt": adopt,
        }
        row.update(t_dens)
        subset_rows.append(row)

    ss_df = pd.DataFrame(subset_rows)
    out_ss = OUT_PREP / "movielens_subset_summary.csv"
    if not out_ss.exists():
        ss_df.to_csv(out_ss, index=False)
        print(f"\nSaved: {out_ss}")

    # ── Threshold density CSV ──────────────────────────────────────
    thr_rows = []
    for name in subsets:
        st = stats_by_strategy[name]
        for t in THRESHOLDS:
            d = st["threshold_density"][t]
            thr_rows.append({
                "strategy":         name,
                "threshold":        t,
                "density":          round(d, 4),
                "positive_edges":   round(d * st["n"] * (st["n"] - 1) / 2),
                "random_AP_baseline": round(d / (d + (1 - d)), 4) if d < 1.0 else 1.0,
            })
    thr_df = pd.DataFrame(thr_rows)
    out_thr = OUT_PREP / "movielens_threshold_density.csv"
    if not out_thr.exists():
        thr_df.to_csv(out_thr, index=False)
        print(f"Saved: {out_thr}")

    # ── Genre summary CSV ──────────────────────────────────────────
    genre_rows = []
    for name, sub in subsets.items():
        gd  = genre_dist(sub)
        row = {"strategy": name}
        row.update(gd)
        genre_rows.append(row)
    genre_df = pd.DataFrame(genre_rows)
    out_genre = OUT_PREP / "movielens_genre_summary.csv"
    if not out_genre.exists():
        genre_df.to_csv(out_genre, index=False)
        print(f"Saved: {out_genre}")

    # ── Figures ───────────────────────────────────────────────────
    print("\n=== Figures ===")
    make_y_count_hist(Y_by_strategy, list(subsets.keys()))
    make_threshold_density_figure(thr_df)
    make_genre_distribution_figure(genre_df)

    # ── Recommended subset selection ───────────────────────────────
    print("\n=== Subset recommendation ===")

    # Strategy C を primary candidate とする（genre balance + mid popular）
    # ただし density と var/mean を確認して最終判定
    rec_name = "genre_stratified_mp100"
    rec_st   = stats_by_strategy[rec_name]
    rec_sub  = subsets[rec_name]

    # best threshold for Bernoulli Y (target: 0.05 〜 0.20)
    thr_cands = {t: abs(d - 0.12) for t, d in rec_st["threshold_density"].items()
                 if 0.03 <= d <= 0.30}
    best_thr  = min(thr_cands, key=thr_cands.get) if thr_cands else 10

    fam_y_poisson  = "yes" if rec_st["var_over_mean_all"] < 30 else "caution"
    fam_y_bernoulli = "yes"

    rec_df = pd.DataFrame([{
        "recommended_strategy":   rec_name,
        "n":                      rec_st["n"],
        "d_genre":                19,
        "Y_count_mean_all":       round(rec_st["mean_all"], 2),
        "Y_count_density_pos":    round(rec_st["density_pos"], 4),
        "var_over_mean_all":      round(rec_st["var_over_mean_all"], 2),
        "recommended_threshold":  best_thr,
        "threshold_density":      round(rec_st["threshold_density"][best_thr], 4),
        "family_y_poisson":       fam_y_poisson,
        "family_y_bernoulli":     fam_y_bernoulli,
        "family_x":               "bernoulli",
        "note": (
            "Primary genre is a convenience label (multi-label data). "
            f"Poisson Y: var/mean={rec_st['var_over_mean_all']:.1f} (overdispersion expected). "
            f"Recommended threshold for Bernoulli Y: {best_thr}."
        ),
    }])
    out_rec = OUT_PREP / "movielens_recommended_subset.csv"
    if not out_rec.exists():
        rec_df.to_csv(out_rec, index=False)
        print(f"Saved: {out_rec}")

    # ── Save npy arrays for recommended subset ─────────────────────
    print(f"\n=== Saving npy arrays for {rec_name} ===")
    save_subset_arrays(rec_sub, Y_by_strategy[rec_name], rec_name)

    # Update movies_metadata with actual rating counts
    p = OUT_DATA / "movielens_movies_metadata.csv"
    if p.exists():
        meta = pd.read_csv(p)
        rpm_dict = rpm.to_dict()
        meta["ratings_count"] = meta["mid"].map(rpm_dict).fillna(0).astype(int)
        meta.to_csv(p, index=False)

    # ── Final report ───────────────────────────────────────────────
    print("\n=== Summary ===")
    for name, sub in subsets.items():
        st = stats_by_strategy[name]
        print(f"\n  [{name}]  n={st['n']}")
        print(f"    rpm: mean={st['rpm_mean']:.1f}  min={st['rpm_min']}  max={st['rpm_max']}")
        print(f"    Y_count: density_pos={st['density_pos']:.4f}  "
              f"mean_all={st['mean_all']:.2f}  var/mean={st['var_over_mean_all']:.2f}")
        print(f"    Threshold densities: " +
              "  ".join(f"t={t}:{st['threshold_density'][t]:.4f}" for t in THRESHOLDS))

    print(f"\n=== Recommended: {rec_name} ===")
    print(f"  n={rec_st['n']}, density_pos={rec_st['density_pos']:.4f}, "
          f"mean_all={rec_st['mean_all']:.2f}, var/mean={rec_st['var_over_mean_all']:.2f}")
    print(f"  Recommended threshold: {best_thr} (density={rec_st['threshold_density'][best_thr]:.4f})")
    print(f"  family_y=poisson: {fam_y_poisson}")
    print(f"  family_y=bernoulli (t={best_thr}): {fam_y_bernoulli}")

    print("\n  NOTE: primary_genre_labels は便宜的ラベル。"
          "各映画は複数ジャンルを持つ可能性があり、NMI/ARI の解釈は限定的。")


if __name__ == "__main__":
    main()
