"""
Step 4 -- Real Data Validation: MovieLens 100K Co-Rating Experiment.

Demonstrates that the Poisson model outperforms the Bernoulli baseline
on real-world count-valued relational data.

Dataset: MovieLens 100K (https://grouplens.org/datasets/movielens/100k/)
  Y_ij = number of users who co-rated movie pair (i, j)
           -> natural count variable, Poisson-distributed
  X_i  = genre feature vector (18 binary genres), z-score normalised

Key insight: Top-n popular movies have near-full co-rating density
(almost all pairs share at least one common viewer). Bernoulli trained
on binarised Y (all-1) learns Z almost entirely from X attributes, while
Poisson trained on count Y learns both attribute structure AND count
magnitude. This difference reveals itself when predicting STRONG
connections (Y > threshold, e.g., top 30th percentile of counts).

Experiment:
  1. Build co-rating matrix Y, select top-n movies by popularity
  2. Hold out 20% of upper-triangle pairs as test set
  3. Fit Bernoulli baseline (binarised Y_train) and Poisson (count Y_train)
     with k selected by quick BIC scan
  4. Evaluate on Y_test:
     - AUC-ROC  (link strength: Y_ij > HIGH_THRESHOLD, top 30%)
     - Spearman (ranking: score vs actual count rank)
     - RMSE     (Poisson: lambda vs count; Bernoulli: scaled prob vs count)

Output:
    expfam/results/real_movielens_results.csv
    expfam/results/real_movielens_bic.csv
    expfam/results/real_movielens_plot.png
    expfam/results/GEMINI_REPORT_STEP4.md
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import urllib.request
import zipfile
import io

_ROOT = Path(__file__).parent.parent.parent
_SRC  = Path(__file__).parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from utils_expfam import run_em, calc_bic, calc_Q_strict             # noqa
from model_expfam import ExpFamLatentStructuralModel                  # noqa

OUTPUT_DIR = _ROOT / "expfam" / "results"
DATA_DIR   = _ROOT / "expfam" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
N_NODES       = 300        # top-n movies by popularity
L             = 10
NUM_ITER      = 20         # more iterations for real data
K_SCAN        = [2, 3, 4, 5]   # BIC k scan
TEST_FRAC     = 0.20       # hold-out fraction
HIGH_PCT      = 70         # Y > percentile(HIGH_PCT) = "strong link" for AUC
SEED          = 2024

ML_URL  = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_PATH = DATA_DIR / "ml-100k.zip"


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------
def download_movielens():
    if ML_PATH.exists():
        print(f"  Using cached: {ML_PATH}")
        return
    print(f"  Downloading MovieLens 100K from {ML_URL} ...")
    try:
        urllib.request.urlretrieve(ML_URL, ML_PATH)
        print(f"  Downloaded: {ML_PATH} ({ML_PATH.stat().st_size // 1024} KB)")
    except Exception as e:
        raise RuntimeError(
            f"Download failed: {e}\n"
            f"Please manually download {ML_URL} to {ML_PATH}"
        )


def load_movielens():
    download_movielens()
    with zipfile.ZipFile(ML_PATH) as zf:
        # Ratings: userId, movieId, rating, timestamp (tab-separated)
        with zf.open("ml-100k/u.data") as f:
            ratings = pd.read_csv(
                f, sep="\t", header=None,
                names=["userId", "movieId", "rating", "timestamp"]
            )
        # Movie metadata: movieId | title | ... | 18 genre flags
        genre_cols = [
            "unknown", "Action", "Adventure", "Animation", "Childrens",
            "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
            "FilmNoir", "Horror", "Musical", "Mystery", "Romance",
            "SciFi", "Thriller", "War", "Western"
        ]
        meta_cols = (
            ["movieId", "title", "release_date", "video_release_date", "imdb_url"]
            + genre_cols
        )
        with zf.open("ml-100k/u.item") as f:
            movies = pd.read_csv(
                f, sep="|", header=None, names=meta_cols,
                encoding="latin-1"
            )
    return ratings, movies[["movieId"] + genre_cols]


def build_corating_matrix(ratings, movie_ids):
    """Build Y[i,j] = # users who rated both movie i and movie j."""
    # Pivot: rows=movieId, cols=userId, values=1/0
    r = ratings[ratings.movieId.isin(movie_ids)].copy()
    pivot = r.pivot_table(index="movieId", columns="userId",
                          values="rating", aggfunc="count", fill_value=0)
    pivot = pivot.loc[movie_ids]           # ensure row order
    # Co-rating count: Y = pivot @ pivot.T  (integer-valued)
    Y = (pivot.values @ pivot.values.T).astype(np.float64)
    np.fill_diagonal(Y, 0.0)              # zero diagonal
    return Y


def build_attribute_matrix(movies, movie_ids, genre_cols):
    """X[i] = genre feature vector, z-score normalised."""
    sub = movies.set_index("movieId").loc[movie_ids]
    X = sub[genre_cols].values.astype(np.float64)
    # z-score per column (column mean=0, std=1)
    mu  = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    X = (X - mu) / std
    return X


# -----------------------------------------------------------------------
# Train / Test split
# -----------------------------------------------------------------------
def make_train_test_split(Y, test_frac, seed):
    """
    Hold out `test_frac` of upper-triangle pairs as test set.
    Returns Y_train (test positions zeroed) and boolean test_mask.
    """
    n = Y.shape[0]
    rng = np.random.default_rng(seed)

    upper_idx = np.argwhere(np.triu(np.ones((n, n), dtype=bool), k=1))
    n_test = int(len(upper_idx) * test_frac)
    chosen = rng.choice(len(upper_idx), size=n_test, replace=False)

    test_mask = np.zeros((n, n), dtype=bool)
    for ci in chosen:
        i, j = upper_idx[ci]
        test_mask[i, j] = True
        test_mask[j, i] = True

    Y_train = Y.copy()
    Y_train[test_mask] = 0.0
    return Y_train, test_mask


# -----------------------------------------------------------------------
# BIC-based k selection (Poisson model on Y_train_pois)
# -----------------------------------------------------------------------
def select_k_by_bic(X, Y, k_list, L, num_iter_bic, seed, n_nodes, d):
    print("  BIC k scan (Poisson, num_iter=10):")
    bic_results = []
    fake_params = {"var_f": 5.0}   # no true Z/F for real data
    for k in k_list:
        res = run_em(
            X, Y, true_params=fake_params,
            family="poisson", k=k,
            L=L, num_iter=num_iter_bic,
            seed=seed, compute_strict_Q=True,
            sigma_y_init=1.0,
        )
        bic, num_params = calc_bic(res["Q_strict"], k=k, n=n_nodes, d=d)
        bic_results.append({"k": k, "Q_strict": res["Q_strict"],
                             "BIC": bic, "num_params": num_params})
        print(f"    k={k}  Q_strict={res['Q_strict']:.1f}  BIC={bic:.1f}")

    df_bic = pd.DataFrame(bic_results)
    best_k = int(df_bic.loc[df_bic.BIC.idxmin(), "k"])
    print(f"  Best k by BIC: {best_k}")
    return best_k, df_bic


# -----------------------------------------------------------------------
# Evaluation metrics
# -----------------------------------------------------------------------
def sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def evaluate(Z_est, w0, w, family, Y_full, test_mask, high_threshold):
    """Compute AUC, RMSE(count), Spearman on test pairs.

    AUC-ROC uses Y > high_threshold as positive label (strong link detection).
    This is meaningful even when density ≈ 1 (near all pairs co-rated).
    """
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    eta = float(w0) + float(w) * (Z_est @ Z_est.T)

    if family == "bernoulli":
        score = sigmoid(eta)
    else:  # poisson
        score = np.exp(np.clip(eta, -20, 10))

    # Upper-triangle test pairs only
    upper_test = np.triu(test_mask, k=1)
    y_count  = Y_full[upper_test]              # true counts
    y_strong = (y_count > high_threshold).astype(int)  # strong link label
    y_exist  = (y_count > 0).astype(int)       # existence label
    y_pred   = score[upper_test]

    # AUC-ROC: strong link detection task (Y > high_threshold)
    if y_strong.sum() == 0 or y_strong.sum() == len(y_strong):
        auc_strong = float("nan")
    else:
        auc_strong = roc_auc_score(y_strong, y_pred)

    # AUC-ROC: existence task (Y > 0)
    if y_exist.sum() == 0 or y_exist.sum() == len(y_exist):
        auc_exist = float("nan")
    else:
        auc_exist = roc_auc_score(y_exist, y_pred)

    # RMSE (count task): compare predicted score to actual count
    # For fair comparison, normalise Bernoulli prediction to count scale
    if family == "bernoulli":
        scale = float(np.mean(y_count)) / max(float(np.mean(y_pred)), 1e-8)
        y_pred_scaled = y_pred * scale
    else:
        y_pred_scaled = y_pred
    rmse_count = float(np.sqrt(np.mean((y_pred_scaled - y_count) ** 2)))

    # Spearman (ranking task): scale-invariant, most fair comparison
    sp_corr = float(spearmanr(y_pred, y_count).statistic)

    return {
        "auc_strong": auc_strong,
        "auc_exist":  auc_exist,
        "rmse_count": rmse_count,
        "spearman":   sp_corr,
        "n_test_pairs":    int(len(y_count)),
        "n_strong_pairs":  int(y_strong.sum()),
        "n_exist_pairs":   int(y_exist.sum()),
        "high_threshold":  float(high_threshold),
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  Step 4 -- Real Data: MovieLens 100K Co-Rating")
    print("=" * 65)

    t_total = time.time()

    # ---- Load data ----
    print("\n[1] Loading MovieLens 100K ...")
    ratings, movies = load_movielens()
    genre_cols = [c for c in movies.columns if c != "movieId"]
    print(f"  {len(ratings)} ratings, {movies.movieId.nunique()} movies, "
          f"{ratings.userId.nunique()} users")

    # ---- Select top-n movies by total co-rating degree ----
    print(f"\n[2] Building co-rating matrix (top {N_NODES} movies) ...")
    # Degree = number of ratings per movie
    movie_degree = ratings.groupby("movieId").size().sort_values(ascending=False)
    top_movies = movie_degree.index[:N_NODES].tolist()
    Y_full = build_corating_matrix(ratings, top_movies)
    X      = build_attribute_matrix(movies, top_movies, genre_cols)

    n, d = X.shape
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    Y_upper    = Y_full[upper_mask]
    n_nonzero  = int((Y_upper > 0).sum())
    density    = n_nonzero / len(Y_upper)

    print(f"  n={n} movies, d={d} genre features")
    print(f"  Y: max={int(Y_full.max())}, mean={Y_upper.mean():.2f}, "
          f"density={density:.3f} ({n_nonzero}/{len(Y_upper)} nonzero pairs)")

    # ---- Train / Test split ----
    print(f"\n[3] Train/Test split (test_frac={TEST_FRAC}) ...")
    Y_train, test_mask = make_train_test_split(Y_full, TEST_FRAC, SEED)
    Y_train_bern = (Y_train > 0).astype(np.float64)  # binarised for Bernoulli
    n_test_pairs = int(np.triu(test_mask, k=1).sum())
    print(f"  Train pairs: {int(upper_mask.sum()) - n_test_pairs}, "
          f"Test pairs: {n_test_pairs}")

    # Compute high-link threshold from training data (positive counts only)
    Y_train_upper = Y_train[upper_mask]
    Y_train_pos   = Y_train_upper[Y_train_upper > 0]
    high_threshold = float(np.percentile(Y_train_pos, HIGH_PCT)) if len(Y_train_pos) > 0 else 1.0
    print(f"  High-link threshold (>{HIGH_PCT}th pct of nonzero train counts): {high_threshold:.1f}")

    # ---- k selection via BIC ----
    print(f"\n[4] k selection by BIC (Poisson, k_scan={K_SCAN}) ...")
    best_k, df_bic = select_k_by_bic(
        X, Y_train, K_SCAN, L=L, num_iter_bic=10,
        seed=SEED, n_nodes=n, d=d
    )
    bic_csv = OUTPUT_DIR / "real_movielens_bic.csv"
    df_bic.to_csv(bic_csv, index=False)

    fake_params = {"var_f": 5.0}

    # ---- Fit Bernoulli (baseline) ----
    print(f"\n[5] Fitting Bernoulli baseline (k={best_k}, num_iter={NUM_ITER}) ...")
    t0 = time.time()
    res_bern = run_em(
        X, Y_train_bern, true_params=fake_params,
        family="bernoulli", k=best_k,
        L=L, num_iter=NUM_ITER, seed=SEED,
    )
    t_bern = time.time() - t0
    Z_bern = res_bern["_Z_samples"][:, :, -1]
    print(f"  Done in {t_bern:.1f}s  Q_final={res_bern['Q_final']:.1f}  "
          f"w0={res_bern['w0_est']:.3f}  w={res_bern['w_est']:.3f}")

    # ---- Fit Poisson (proposed) ----
    print(f"\n[6] Fitting Poisson proposed (k={best_k}, num_iter={NUM_ITER}) ...")
    t0 = time.time()
    res_pois = run_em(
        X, Y_train, true_params=fake_params,
        family="poisson", k=best_k,
        L=L, num_iter=NUM_ITER, seed=SEED,
    )
    t_pois = time.time() - t0
    Z_pois = res_pois["_Z_samples"][:, :, -1]
    print(f"  Done in {t_pois:.1f}s  Q_final={res_pois['Q_final']:.1f}  "
          f"w0={res_pois['w0_est']:.3f}  w={res_pois['w_est']:.3f}")

    # ---- Evaluate on test set ----
    print("\n[7] Evaluating on test set ...")
    eval_bern = evaluate(Z_bern, res_bern["w0_est"], res_bern["w_est"],
                         "bernoulli", Y_full, test_mask, high_threshold)
    eval_pois = evaluate(Z_pois, res_pois["w0_est"], res_pois["w_est"],
                         "poisson",   Y_full, test_mask, high_threshold)

    print(f"\n  {'Metric':<20} {'Bernoulli':>12} {'Poisson':>12} {'Winner':>10}")
    print(f"  {'-'*56}")
    for metric, higher_better in [("auc_strong", True), ("rmse_count", False),
                                   ("spearman", True)]:
        bv, pv = eval_bern[metric], eval_pois[metric]
        if higher_better:
            winner = "Poisson" if pv > bv else "Bernoulli"
        else:
            winner = "Poisson" if pv < bv else "Bernoulli"
        print(f"  {metric:<20} {bv:>12.4f} {pv:>12.4f} {winner:>10}")

    print(f"\n  Test set: {eval_pois['n_test_pairs']} pairs "
          f"({eval_pois['n_strong_pairs']} strong [Y>{high_threshold:.1f}], "
          f"{eval_pois['n_exist_pairs']} nonzero)")

    # ---- Save results CSV ----
    results = pd.DataFrame([
        {"model": "Bernoulli", "k": best_k, "num_iter": NUM_ITER,
         "auc_strong": eval_bern["auc_strong"], "auc_exist": eval_bern["auc_exist"],
         "rmse_count": eval_bern["rmse_count"], "spearman": eval_bern["spearman"],
         "high_threshold": high_threshold,
         "w0": res_bern["w0_est"], "w": res_bern["w_est"],
         "Q_final": res_bern["Q_final"], "elapsed_s": t_bern},
        {"model": "Poisson", "k": best_k, "num_iter": NUM_ITER,
         "auc_strong": eval_pois["auc_strong"], "auc_exist": eval_pois["auc_exist"],
         "rmse_count": eval_pois["rmse_count"], "spearman": eval_pois["spearman"],
         "high_threshold": high_threshold,
         "w0": res_pois["w0_est"], "w": res_pois["w_est"],
         "Q_final": res_pois["Q_final"], "elapsed_s": t_pois},
    ])
    results_csv = OUTPUT_DIR / "real_movielens_results.csv"
    results.to_csv(results_csv, index=False)

    # ---- Plots ----
    _make_plots(Z_bern, Z_pois, res_bern, res_pois, eval_bern, eval_pois,
                Y_full, test_mask, df_bic, best_k, n, d, density, high_threshold)

    # ---- Gemini report ----
    total = time.time() - t_total
    _write_report(eval_bern, eval_pois, res_bern, res_pois,
                  n, d, density, Y_upper, best_k, total, high_threshold)

    print(f"\n  Total time: {total:.1f}s ({total/60:.1f} min)")
    print(f"  Saved: {results_csv}")
    print(f"  Saved: {bic_csv}")
    print("\n[experiment_real_movielens DONE]")


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------
def _make_plots(Z_bern, Z_pois, res_bern, res_pois, eval_bern, eval_pois,
                Y_full, test_mask, df_bic, best_k, n, d, density, high_threshold):
    from sklearn.metrics import roc_curve
    from scipy.stats import spearmanr

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    upper_test = np.triu(test_mask, k=1)
    y_count  = Y_full[upper_test]
    y_strong = (y_count > high_threshold).astype(int)  # strong-link label for ROC

    colors = {"bernoulli": "tab:blue", "poisson": "tab:orange"}

    # ---- BIC curve ----
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(df_bic.k, df_bic.BIC, "o-", color="tab:purple", linewidth=2)
    ax0.axvline(best_k, color="gray", linestyle="--", linewidth=1.5,
                label=f"best k={best_k}")
    ax0.set_xlabel("k", fontsize=11); ax0.set_ylabel("BIC", fontsize=11)
    ax0.set_title("BIC vs k (Poisson, train)", fontsize=12)
    ax0.legend(fontsize=9); ax0.grid(True, alpha=0.3)
    ax0.set_xticks(df_bic.k.tolist())

    # ---- ROC curves (strong-link detection: Y > high_threshold) ----
    ax1 = fig.add_subplot(gs[0, 1])
    for fam, Z_est, w0, w, res in [
        ("bernoulli", Z_bern, res_bern["w0_est"], res_bern["w_est"], res_bern),
        ("poisson",   Z_pois, res_pois["w0_est"], res_pois["w_est"], res_pois),
    ]:
        eta = float(w0) + float(w) * (Z_est @ Z_est.T)
        score = sigmoid(eta) if fam == "bernoulli" else np.exp(np.clip(eta, -20, 10))
        fpr, tpr, _ = roc_curve(y_strong, score[upper_test])
        auc_v = eval_bern["auc_strong"] if fam == "bernoulli" else eval_pois["auc_strong"]
        label = f"{'Bernoulli' if fam=='bernoulli' else 'Poisson'} (AUC={auc_v:.3f})"
        ax1.plot(fpr, tpr, color=colors[fam], linewidth=2, label=label)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax1.set_xlabel("FPR", fontsize=11); ax1.set_ylabel("TPR", fontsize=11)
    ax1.set_title(f"ROC: Strong-link (Y>{high_threshold:.0f})", fontsize=12)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # ---- Predicted vs actual count scatter ----
    for col_i, (fam, Z_est, w0, w) in enumerate([
        ("bernoulli", Z_bern, res_bern["w0_est"], res_bern["w_est"]),
        ("poisson",   Z_pois, res_pois["w0_est"], res_pois["w_est"]),
    ]):
        ax = fig.add_subplot(gs[0, 2] if fam == "bernoulli" else gs[1, 0])
        eta = float(w0) + float(w) * (Z_est @ Z_est.T)
        score = sigmoid(eta) if fam == "bernoulli" else np.exp(np.clip(eta, -20, 10))
        pred_vals = score[upper_test]
        sp = eval_bern["spearman"] if fam=="bernoulli" else eval_pois["spearman"]
        ax.scatter(pred_vals, y_count, alpha=0.1, s=3,
                   color=colors[fam], rasterized=True)
        ax.set_xlabel("Predicted score", fontsize=10)
        ax.set_ylabel("Actual co-rating count", fontsize=10)
        title = f"{'Bernoulli' if fam=='bernoulli' else 'Poisson'}: Spearman={sp:.3f}"
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

    # ---- Bar chart comparison ----
    ax_bar = fig.add_subplot(gs[1, 1])
    metrics_display = ["AUC-ROC\n(strong)", "Spearman"]
    bern_vals = [eval_bern["auc_strong"], eval_bern["spearman"]]
    pois_vals  = [eval_pois["auc_strong"],  eval_pois["spearman"]]
    x = np.arange(len(metrics_display))
    ax_bar.bar(x - 0.2, bern_vals, 0.35, label="Bernoulli",
               color="tab:blue", alpha=0.8, edgecolor="black")
    ax_bar.bar(x + 0.2, pois_vals, 0.35, label="Poisson",
               color="tab:orange", alpha=0.8, edgecolor="black")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(metrics_display, fontsize=10)
    ax_bar.set_ylabel("Score (higher = better)", fontsize=10)
    ax_bar.set_title("AUC-ROC & Spearman Comparison", fontsize=11)
    ax_bar.legend(fontsize=9); ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.set_ylim(0, 1.05)

    # ---- Q convergence ----
    ax_q = fig.add_subplot(gs[1, 2])
    ax_q.plot(range(1, len(res_bern["Q_history"]) + 1),
              res_bern["Q_history"], "-", color="tab:blue",
              linewidth=2, label="Bernoulli")
    ax_q.plot(range(1, len(res_pois["Q_history"]) + 1),
              res_pois["Q_history"], "-", color="tab:orange",
              linewidth=2, label="Poisson")
    ax_q.set_xlabel("EM iteration", fontsize=10)
    ax_q.set_ylabel("Q (log-likelihood)", fontsize=10)
    ax_q.set_title("Q Convergence", fontsize=11)
    ax_q.legend(fontsize=9); ax_q.grid(True, alpha=0.3)

    plt.suptitle(
        f"MovieLens 100K Co-Rating Experiment  "
        f"(n={n}, d={d}, k={best_k}, density={density:.3f})\n"
        f"Bernoulli baseline vs Poisson proposed -- Link & Count Prediction",
        fontsize=12
    )
    plot_path = OUTPUT_DIR / "real_movielens_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path}")


# -----------------------------------------------------------------------
# Gemini report
# -----------------------------------------------------------------------
def _write_report(eval_bern, eval_pois, res_bern, res_pois,
                  n, d, density, Y_upper, best_k, total, high_threshold):
    lines = ["# Geminiへの報告書 (Report from Claude -- Step 4)\n\n"]

    # Determine winners
    auc_winner  = "Poisson" if eval_pois["auc_strong"] > eval_bern["auc_strong"] else "Bernoulli"
    rmse_winner = "Poisson" if eval_pois["rmse_count"] < eval_bern["rmse_count"] else "Bernoulli"
    sp_winner   = "Poisson" if eval_pois["spearman"] > eval_bern["spearman"] else "Bernoulli"
    poisson_wins_all = (auc_winner == sp_winner == "Poisson"
                        and rmse_winner == "Poisson")

    lines.append("## 1. 現在のステータスと結論\n\n")
    status = "Step 4 完走。"
    if poisson_wins_all:
        status += ("MovieLens 100K 実データにおいて、**Poissonモデルが全3指標で"
                   "Bernoulliベースラインを上回ることを実証完了**。")
    else:
        winners = f"AUC: {auc_winner}, RMSE: {rmse_winner}, Spearman: {sp_winner}"
        status += f"MovieLens 100K 実データで比較完了。指標別結果: {winners}。"
    lines.append(status + "\n")

    lines.append("\n## 2. データセットの概要\n\n")
    lines.append(
        f"| 項目 | 値 |\n|------|----|\n"
        f"| データセット | MovieLens 100K (co-rating matrix) |\n"
        f"| ノード数 n | {n} 映画 (上位{n}件, 最多評価順) |\n"
        f"| 属性次元 d | {d} (18ジャンル one-hot, z-score正規化) |\n"
        f"| ペア数 | {len(Y_upper):,} (上三角) |\n"
        f"| 非ゼロペア密度 | {density:.3f} ({int((Y_upper>0).sum()):,}/{len(Y_upper):,}) |\n"
        f"| Y最大値 | {int(Y_upper.max())} (co-rating count) |\n"
        f"| Y平均値 | {Y_upper.mean():.2f} |\n"
        f"| 選択 k | {best_k} (BICで自動選択) |\n"
        f"| num_iter | {NUM_ITER} |\n"
    )

    lines.append("\n## 3. 予測性能評価 (Test Set Metrics)\n\n")
    lines.append(
        "| 指標 | Bernoulli (baseline) | Poisson (proposed) | 優位 | 改善幅 |\n"
        "|------|---------------------|-------------------|------|--------|\n"
    )
    def delta_str(b, p, higher_better):
        diff = p - b
        pct = abs(diff) / max(abs(b), 1e-8) * 100
        arrow = "+" if (diff > 0) == higher_better else "-"
        return f"{arrow}{abs(diff):.4f} ({pct:.1f}%)"

    lines.append(
        f"| AUC-ROC (strong, Y>{high_threshold:.0f}) | {eval_bern['auc_strong']:.4f} | {eval_pois['auc_strong']:.4f} "
        f"| **{auc_winner}** | {delta_str(eval_bern['auc_strong'], eval_pois['auc_strong'], True)} |\n"
    )
    lines.append(
        f"| RMSE (count) | {eval_bern['rmse_count']:.4f} | {eval_pois['rmse_count']:.4f} "
        f"| **{rmse_winner}** | {delta_str(eval_bern['rmse_count'], eval_pois['rmse_count'], False)} |\n"
    )
    lines.append(
        f"| Spearman corr | {eval_bern['spearman']:.4f} | {eval_pois['spearman']:.4f} "
        f"| **{sp_winner}** | {delta_str(eval_bern['spearman'], eval_pois['spearman'], True)} |\n"
    )

    lines.append("\n### パラメータ推定値\n\n")
    lines.append(
        f"| モデル | w0 est | w est | Q_final |\n"
        f"|--------|--------|-------|----------|\n"
        f"| Bernoulli | {res_bern['w0_est']:.4f} | {res_bern['w_est']:.4f}"
        f" | {res_bern['Q_final']:.1f} |\n"
        f"| Poisson   | {res_pois['w0_est']:.4f} | {res_pois['w_est']:.4f}"
        f" | {res_pois['Q_final']:.1f} |\n"
    )

    lines.append("\n## 4. 数理的・実装的な懸念点 (Roadblocks)\n\n")
    lines.append(
        "- **Train/Test split**: テストペアを Y_train で 0 として扱うため、"
        "両モデルに同等の biasが導入される。比較の公平性は保たれている。\n"
        "- **RMSE (count) の比較**: Bernoulli の予測スコアは [0,1]、"
        "Poisson の予測 lambda は実際のカウントスケール [0, Y_max]。"
        "そのため絶対的な RMSE 値はスケールが異なる。"
        "Spearman 相関はスケール不変のため最も公正な指標。\n"
        "- **k 選択**: BIC を Poisson モデルで実施し、同じ k を Bernoulli にも適用。\n"
    )

    lines.append("\n## 5. Claudeからの次の一手 (Next Step) の提案\n\n")
    lines.append(
        "人工データ実験 (Step 3.x) と実データ検証 (Step 4) がすべて完了した。\n\n"
        "**Step 5: 論文執筆 (LaTeX化)**\n\n"
        "提案する論文構成:\n"
        "1. Introduction: 指数族による統一フレームワークの動機\n"
        "2. Model: ExpFam LSM の数式定義、勾配・精度行列の一般化\n"
        "3. Algorithm: Monte Carlo EM, Newton-Laplace E-step, Adam M-step\n"
        "4. Experiments (Synthetic): Step 3.7 (L字型), 3.8 (全パラメータ vs n/d)\n"
        "5. Experiments (Real): Step 4 (MovieLens 100K, AUC/Spearman 比較)\n"
        "6. Conclusion: 汎用フレームワークの意義と今後の拡張 (NB, GP等)\n\n"
        "すべての Figure/Table データは `expfam/results/` に揃っている。\n"
    )

    lines.append(f"\n実行時間: {total:.1f}s ({total/60:.1f} min)\n\n")
    lines.append("*Claude Code -- Step 4 Report*\n")

    report_path = OUTPUT_DIR / "GEMINI_REPORT_STEP4.md"
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"  Saved: {report_path}")


if __name__ == "__main__":
    main()
