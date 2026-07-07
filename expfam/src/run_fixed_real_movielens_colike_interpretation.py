"""
MovieLens co-like recommendation and latent factor interpretation experiment.

Two experiments on the existing genre_stratified_mp100 movie subset (n=100, d=19):

  Experiment A (Poisson):
    Y_colike_count[i,j] = # users who rated BOTH movie i and movie j >= 4 (rating>=4 = "high rating").
    family_x=bernoulli (genre multi-hot), family_y=poisson.
    In-sample count reconstruction + K-interpretation analysis (NOT strict held-out:
    the model API has no pair mask, see model_dual_expfam_fixed.py).

  Experiment B (Bernoulli lift link recommendation):
    Y_lift_binary[i,j] = 1 if observed co-like count is large AND popularity-adjusted
    "lift" over the popularity-expected co-like count is large (i.e. the pair is liked
    together more than popularity alone would predict).
    family_x=bernoulli, family_y=bernoulli, held-out link prediction.
    The model has no pair-mask support, so held-out positives are hidden by zeroing them
    in Y_train ("zero_filled_edge_hiding_no_pair_mask"), NOT a strict missing-edge CV.

Baselines (popularity product / genre cosine / popularity+genre / train item-item score)
are evaluated on the SAME test positive/negative candidates as the proposed method.

IMPORTANT — scope limits (see report for full list):
  - Poisson count modeling is in-sample reconstruction, not held-out count prediction.
  - Bernoulli lift link prediction is zero-filled edge hiding, not strict pair-mask CV.
  - K-dimension interpretation is tentative/suggestive (Z has rotation non-identifiability).
  - This is an n=100 movie subset, not a conclusion about MovieLens as a whole.

Output:
  expfam/results/real_data/movielens_colike_interpretation/
  expfam/figures/real_data/movielens_colike_interpretation/
"""

import re
import sys
import time
import zipfile
import warnings
import traceback
from pathlib import Path
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    roc_auc_score, average_precision_score, silhouette_score,
    normalized_mutual_info_score, adjusted_rand_score,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

_SRC  = Path(__file__).parent
_ROOT = _SRC.parent.parent

sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from model_dual_expfam_fixed import DualExpFamLSMFixed
from utils_expfam import calc_Q_dual_strict, calc_bic_dual

# ─────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────

ZIP_PATH = _ROOT / "expfam" / "data" / "ml-100k.zip"
DATA_DIR = _ROOT / "expfam" / "data" / "movielens_pilot"
OUT_DIR  = _ROOT / "expfam" / "results" / "real_data" / "movielens_colike_interpretation"
FIG_DIR  = _ROOT / "expfam" / "figures" / "real_data" / "movielens_colike_interpretation"

FAMILY_X = "bernoulli"

LIKE_THRESHOLD = 4   # rating >= 4 => "high rating" / "like"

MIN_SUPPORT_CANDIDATES   = [5, 10, 20]
LIFT_THRESHOLD_CANDIDATES = [1.2, 1.5, 2.0, 3.0]
LIFT_EPS = 1e-9

# Experiment A (Poisson co-like count)
K_LIST_A   = [2, 3, 5, 8]
N_TRIALS_A = 3
SEED_BASE_A = 21000   # + k*100 + trial*10

# Experiment B (Bernoulli lift link prediction, held-out)
K_LIST_B        = [3, 5, 8]
SPLIT_TRIALS_B  = [0, 1, 2]
MODEL_TRIALS_B  = [0]
TEST_EDGE_RATIO = 0.2
NEG_RATIO       = 5
RANDOM_AP_SAMPLED = 1.0 / (1.0 + NEG_RATIO)
EVALUATION_MODE = "zero_filled_edge_hiding_no_pair_mask"
SPLIT_SEED_BASE_B = 24000   # + split_trial*100
MODEL_SEED_BASE_B = 25000   # + k*100 + split_trial*10 + model_trial

TOP_K_QUERY = 10

BEST_K_INTERPRETATION_CANDIDATES = [5, 8]

L, NITER = 5, 8

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]

PRIORITY_QUERY_TITLES = [
    "Toy Story", "Star Wars", "Fargo", "Pulp Fiction", "Godfather",
    "Twelve Monkeys", "Independence Day", "Scream", "Return of the Jedi", "Contact",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def load_raw_ratings():
    """Load full u.data ratings (uid, mid, rating) from the ml-100k zip."""
    assert ZIP_PATH.exists(), f"Not found: {ZIP_PATH}"
    with zipfile.ZipFile(ZIP_PATH) as zf:
        with zf.open("ml-100k/u.data") as f:
            raw = f.read().decode("latin-1")
    rows = []
    for line in raw.strip().split("\n"):
        p = line.strip().split("\t")
        if len(p) >= 3:
            rows.append((int(p[0]), int(p[1]), int(p[2])))
    return pd.DataFrame(rows, columns=["uid", "mid", "rating"])


def load_subset():
    """Load the existing genre_stratified_mp100 subset (X, ids, labels, metadata)."""
    X            = np.load(DATA_DIR / "movielens_X_genre.npy").astype(np.float64)
    movie_ids    = np.load(DATA_DIR / "movielens_movie_ids.npy")
    genre_labels = np.load(DATA_DIR / "movielens_primary_genre_labels.npy")
    meta         = pd.read_csv(DATA_DIR / "movielens_movies_metadata.csv")

    # Defensive re-alignment: ensure meta row order matches movie_ids order.
    if not np.array_equal(meta["mid"].values, movie_ids):
        meta = meta.set_index("mid").loc[movie_ids].reset_index()

    titles = meta["title"].tolist()
    return X, movie_ids, genre_labels, meta, titles


def parse_release_year(title):
    m = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(m.group(1)) if m else np.nan


# ─────────────────────────────────────────────────────────────────────
# Co-like count + lift construction
# ─────────────────────────────────────────────────────────────────────

def compute_movie_attributes(ratings_df, movie_ids, like_threshold=LIKE_THRESHOLD):
    """Per-movie like_count / rating_count / avg_rating / high_rating_rate, plus
    per-movie set of users who rated it >= like_threshold (for co-like counting)."""
    sub = ratings_df[ratings_df["mid"].isin(set(movie_ids.tolist()))]
    like_users = defaultdict(set)
    rating_count = {}
    avg_rating = {}
    like_count = {}

    grouped = sub.groupby("mid")
    for mid, g in grouped:
        rating_count[mid] = int(len(g))
        avg_rating[mid]   = float(g["rating"].mean())
        like_users[mid]   = set(g.loc[g["rating"] >= like_threshold, "uid"].tolist())
        like_count[mid]   = len(like_users[mid])

    n_users = int(ratings_df["uid"].nunique())
    attrs = pd.DataFrame({
        "mid": movie_ids,
        "rating_count": [rating_count.get(m, 0) for m in movie_ids],
        "avg_rating":   [avg_rating.get(m, np.nan) for m in movie_ids],
        "like_count":   [like_count.get(m, 0) for m in movie_ids],
    })
    attrs["high_rating_rate"] = attrs["like_count"] / attrs["rating_count"].replace(0, np.nan)
    attrs["log_rating_count"] = np.log1p(attrs["rating_count"])
    attrs["log_like_count"]   = np.log1p(attrs["like_count"])
    return attrs, like_users, n_users


def compute_Y_colike_count(movie_ids, like_users):
    n = len(movie_ids)
    Y = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        ui = like_users.get(movie_ids[i], set())
        for j in range(i + 1, n):
            uj = like_users.get(movie_ids[j], set())
            c = len(ui & uj)
            Y[i, j] = c
            Y[j, i] = c
    return Y


def compute_lift_matrix(Y_colike_count, like_count_arr, n_users):
    n = len(like_count_arr)
    expected = np.outer(like_count_arr, like_count_arr).astype(float) / max(n_users, 1)
    np.fill_diagonal(expected, 0.0)
    lift = Y_colike_count / np.maximum(expected, LIFT_EPS)
    return expected, lift


def cosine_sim_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    Xn = X / norms
    return Xn @ Xn.T


# ─────────────────────────────────────────────────────────────────────
# Lift threshold audit + selection
# ─────────────────────────────────────────────────────────────────────

def audit_lift_candidates(Y_colike_count, lift, like_count_arr, genre_sim, titles):
    n = Y_colike_count.shape[0]
    upper = np.triu_indices(n, k=1)
    n_pairs_total = len(upper[0])
    pop_product_all = (like_count_arr[upper[0]] * like_count_arr[upper[1]]).astype(float)
    genre_sim_all = genre_sim[upper]
    support_all = Y_colike_count[upper]
    lift_all = lift[upper]

    mean_pop_all = float(pop_product_all.mean())
    mean_genre_all = float(genre_sim_all.mean())

    rows = []
    for min_support, threshold in product(MIN_SUPPORT_CANDIDATES, LIFT_THRESHOLD_CANDIDATES):
        mask = (support_all >= min_support) & (lift_all >= threshold)
        n_pos = int(mask.sum())
        density = n_pos / n_pairs_total

        if n_pos > 0:
            sel_support = support_all[mask]
            sel_lift = lift_all[mask]
            sel_pop = pop_product_all[mask]
            sel_genre = genre_sim_all[mask]
            mean_support = float(sel_support.mean())
            median_support = float(np.median(sel_support))
            mean_lift = float(sel_lift.mean())
            pop_bias_mean = float(sel_pop.mean())
            pop_bias_ratio = pop_bias_mean / mean_pop_all if mean_pop_all > 0 else float("nan")
            genre_check_mean = float(sel_genre.mean())

            order = np.argsort(-sel_lift)[:3]
            sel_i = upper[0][mask]
            sel_j = upper[1][mask]
            top_examples = "; ".join(
                f"{titles[sel_i[o]]} / {titles[sel_j[o]]} (lift={sel_lift[o]:.2f}, count={sel_support[o]:.0f})"
                for o in order
            )
        else:
            mean_support = median_support = mean_lift = float("nan")
            pop_bias_mean = pop_bias_ratio = genre_check_mean = float("nan")
            top_examples = ""

        rows.append({
            "decision_type": "lift_binary_candidate",
            "min_support": min_support,
            "lift_threshold": threshold,
            "positive_edges": n_pos,
            "density": density,
            "mean_support": mean_support,
            "median_support": median_support,
            "mean_lift": mean_lift,
            "popularity_bias_mean_product": pop_bias_mean,
            "popularity_bias_ratio_vs_all_pairs": pop_bias_ratio,
            "genre_cosine_mean_positive": genre_check_mean,
            "genre_cosine_mean_all_pairs": mean_genre_all,
            "top_positive_examples": top_examples,
        })
    return pd.DataFrame(rows)


def select_lift_setting(audit_df):
    """Auto-select (min_support, lift_threshold) closest to target density=0.10,
    preferring candidates with positive_edges>=200 and density in [0.05,0.15]."""
    df = audit_df.copy()
    df["score"] = (df["density"] - 0.10).abs()
    in_range = df[(df["density"] >= 0.05) & (df["density"] <= 0.15)]
    preferred = in_range[in_range["positive_edges"] >= 200]

    if len(preferred) > 0:
        chosen = preferred.sort_values("score").iloc[0]
        reason = "density in [0.05,0.15] and positive_edges>=200; picked closest to target density 0.10"
    elif len(in_range) > 0:
        chosen = in_range.sort_values("score").iloc[0]
        reason = "density in [0.05,0.15] but positive_edges<200 for all such candidates; picked closest to target density 0.10"
    else:
        chosen = df.sort_values("score").iloc[0]
        reason = "no candidate had density in [0.05,0.15]; picked global closest to target density 0.10 (fallback)"

    return int(chosen["min_support"]), float(chosen["lift_threshold"]), reason, chosen


def decide_high_colike_threshold(Y_colike_count):
    """High-colike threshold for Experiment A binary eval: target ~top-10% density."""
    n = Y_colike_count.shape[0]
    upper = Y_colike_count[np.triu_indices(n, k=1)]
    rows = []
    for pct in [80, 85, 90, 95]:
        thr = float(np.ceil(np.percentile(upper, pct)))
        density = float((upper >= thr).mean())
        rows.append({
            "decision_type": "high_colike_threshold_candidate",
            "percentile": pct, "threshold": thr, "density": density,
        })
    audit = pd.DataFrame(rows)
    # pick percentile=90 candidate (closest to a "top 10%" threshold by construction)
    chosen = audit[audit["percentile"] == 90].iloc[0]
    return float(chosen["threshold"]), float(chosen["density"]), audit


# ─────────────────────────────────────────────────────────────────────
# Z metrics (shared)
# ─────────────────────────────────────────────────────────────────────

def compute_z_metrics(Z_est, labels):
    k_z  = Z_est.shape[1]
    n_cl = len(np.unique(labels))
    if k_z == 1:
        Z_2d = np.hstack([Z_est, np.zeros((len(Z_est), 1))])
    elif k_z == 2:
        Z_2d = Z_est.copy()
    else:
        Z_2d = PCA(n_components=2).fit_transform(Z_est)
    try:
        sil = float(silhouette_score(Z_2d, labels))
    except Exception:
        sil = float("nan")
    try:
        km  = KMeans(n_clusters=max(n_cl, 2), random_state=42, n_init=10).fit_predict(Z_2d)
        nmi = float(normalized_mutual_info_score(labels, km))
        ari = float(adjusted_rand_score(labels, km))
    except Exception:
        nmi = ari = float("nan")
    return sil, nmi, ari, Z_2d


# ─────────────────────────────────────────────────────────────────────
# Experiment A: Poisson co-like count EM
# ─────────────────────────────────────────────────────────────────────

def run_em_poisson(X, Y, k, seed, L=5, num_iter=8):
    n, d = X.shape
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_count = float(Y[upper_mask].mean())

    nan_count_total = 0
    error_msg = ""

    for retry in range(3):
        newton_alpha = 0.5 / (2 ** retry)
        w_init = 0.1 / (2 ** retry)

        rng   = np.random.default_rng(seed + retry * 1000)
        model = DualExpFamLSMFixed(n=n, d=d, k=k, L=L, family_x=FAMILY_X, family_y="poisson")
        model.initialize_params(true_params=None, seed=seed + retry * 1000)

        model.params["w0"] = float(np.log(max(mean_count, 1e-10)))
        model.params["w"]  = w_init
        model.params["F"] *= 0.2

        Z     = model.params["Z"].copy()
        F     = model.params["F"].copy()
        sigma = model.params["sigma"].copy()
        w0    = float(model.params["w0"])
        w     = float(model.params["w"])
        var_z = float(model.params["var_z"])
        Z_prev, nan_count = Z.copy(), 0

        try:
            for _ in range(1, num_iter + 1):
                Z_samples = np.zeros((n, k, L))
                for l in range(L):
                    model.params.update(dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w))
                    Z_new = model.calc_eta_newton(X, Y, rng=rng, max_iter=10, alpha=newton_alpha)
                    Z_samples[:, :, l] = Z_new
                    Z = Z_new.copy()

                if np.any(np.isnan(Z_samples)) or np.any(np.isinf(Z_samples)):
                    nan_count += 1
                    Z_samples = np.stack([Z_prev] * L, axis=2)
                    Z = Z_prev.copy()

                Z_samples = model.scale_Z(Z_samples)
                Z_prev    = Z.copy()
                Z         = Z_samples[:, :, -1].copy()
                F         = model.calc_F(X, Z_samples)
                sigma     = model.calc_sigma(X, Z_samples, F)
                w0        = float(model.calc_w0(Y, Z_samples, w0, w, max_iter=50))
                w         = float(model.calc_w(Y, Z_samples, w0, w, max_iter=50))
        except Exception as e:
            error_msg = str(e)
            nan_count += 1

        nan_count_total += nan_count
        if nan_count == 0:
            break

    model.params.update({"F": F, "sigma": sigma, "w0": w0, "w": w})

    try:
        Q_strict  = calc_Q_dual_strict(X, Y, Z_samples, F, sigma, var_z, w0, w, model)
        bic, npar = calc_bic_dual(Q_strict, k, n, d, FAMILY_X, "poisson")
    except Exception:
        Q_strict, bic, npar = float("nan"), float("nan"), 0

    Z_est = Z_samples[:, :, -1]
    eta_y = float(w0) + float(w) * (Z_est @ Z_est.T)
    mu_y  = np.clip(model._mean_function(eta_y), 0.0, 1e6)

    return {
        "model": model, "Z_est": Z_est, "F": F, "sigma": sigma,
        "w0": float(w0), "w": float(w), "mu_y": mu_y,
        "Q_strict": Q_strict, "bic": bic, "num_params": npar,
        "nan_occurred": nan_count_total > 0, "nan_count": nan_count_total,
        "error_message": error_msg,
    }


def evaluate_poisson_fit(res, X, Y, high_colike_threshold):
    n = Y.shape[0]
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    y_upper = Y[upper_mask]
    y_hat   = res["mu_y"][upper_mask]

    residuals = y_upper - y_hat
    rmse_y = float(np.sqrt(np.mean(residuals ** 2)))
    mae_y  = float(np.mean(np.abs(residuals)))

    try:
        pearson_corr, _ = pearsonr(y_upper, y_hat)
        pearson_corr = float(pearson_corr)
    except Exception:
        pearson_corr = float("nan")
    try:
        spearman_corr, _ = spearmanr(y_upper, y_hat)
        spearman_corr = float(spearman_corr)
    except Exception:
        spearman_corr = float("nan")

    y_binary = (y_upper >= high_colike_threshold).astype(int)
    try:
        high_colike_auc = float(roc_auc_score(y_binary, y_hat))
    except Exception:
        high_colike_auc = float("nan")
    try:
        high_colike_ap = float(average_precision_score(y_binary, y_hat))
    except Exception:
        high_colike_ap = float("nan")

    mu_x = res["model"]._mean_function_x(res["Z_est"] @ res["F"].T)
    rmse_x = float(np.sqrt(np.mean((X - mu_x) ** 2)))

    return {
        "rmse_y": rmse_y, "mae_y": mae_y,
        "pearson_corr": pearson_corr, "spearman_corr": spearman_corr,
        "high_colike_auc": high_colike_auc, "high_colike_ap": high_colike_ap,
        "high_colike_threshold": high_colike_threshold,
        "rmse_x": rmse_x,
    }


# ─────────────────────────────────────────────────────────────────────
# Experiment B: Bernoulli lift link prediction EM
# ─────────────────────────────────────────────────────────────────────

def run_em_bernoulli(X, Y, k, seed, L=5, num_iter=8):
    n, d = X.shape
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    density = float(np.clip(Y[upper_mask].mean(), 1e-6, 1 - 1e-6))

    nan_count_total = 0
    error_msg = ""

    for retry in range(3):
        newton_alpha = 0.5 / (2 ** retry)

        rng   = np.random.default_rng(seed + retry * 1000)
        model = DualExpFamLSMFixed(n=n, d=d, k=k, L=L, family_x=FAMILY_X, family_y="bernoulli")
        model.initialize_params(true_params=None, seed=seed + retry * 1000)

        model.params["w0"] = float(np.log(density / (1.0 - density)))
        model.params["w"]  = 0.5
        model.params["F"] *= 0.2

        Z     = model.params["Z"].copy()
        F     = model.params["F"].copy()
        sigma = model.params["sigma"].copy()
        w0    = float(model.params["w0"])
        w     = float(model.params["w"])
        var_z = float(model.params["var_z"])
        Z_prev, nan_count = Z.copy(), 0

        try:
            for _ in range(1, num_iter + 1):
                Z_samples = np.zeros((n, k, L))
                for l in range(L):
                    model.params.update(dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w))
                    Z_new = model.calc_eta_newton(X, Y, rng=rng, max_iter=10, alpha=newton_alpha)
                    Z_samples[:, :, l] = Z_new
                    Z = Z_new.copy()

                if np.any(np.isnan(Z_samples)) or np.any(np.isinf(Z_samples)):
                    nan_count += 1
                    Z_samples = np.stack([Z_prev] * L, axis=2)
                    Z = Z_prev.copy()

                Z_samples = model.scale_Z(Z_samples)
                Z_prev    = Z.copy()
                Z         = Z_samples[:, :, -1].copy()
                F         = model.calc_F(X, Z_samples)
                sigma     = model.calc_sigma(X, Z_samples, F)
                w0        = float(model.calc_w0(Y, Z_samples, w0, w, max_iter=50))
                w         = float(model.calc_w(Y, Z_samples, w0, w, max_iter=50))
        except Exception as e:
            error_msg = str(e)
            nan_count += 1

        nan_count_total += nan_count
        if nan_count == 0:
            break

    model.params.update({"F": F, "sigma": sigma, "w0": w0, "w": w})

    try:
        Q_strict  = calc_Q_dual_strict(X, Y, Z_samples, F, sigma, var_z, w0, w, model)
        bic, npar = calc_bic_dual(Q_strict, k, n, d, FAMILY_X, "bernoulli")
    except Exception:
        Q_strict, bic, npar = float("nan"), float("nan"), 0

    Z_est = Z_samples[:, :, -1]
    eta_y = float(w0) + float(w) * (Z_est @ Z_est.T)
    mu_y  = model._mean_function(eta_y)

    return {
        "model": model, "Z_est": Z_est, "F": F, "sigma": sigma,
        "w0": float(w0), "w": float(w), "mu_y": mu_y,
        "Q_strict": Q_strict, "bic": bic, "num_params": npar,
        "nan_occurred": nan_count_total > 0, "nan_count": nan_count_total,
        "error_message": error_msg,
    }


def split_positive_edges(Y, test_ratio, seed):
    rng = np.random.default_rng(seed)
    pos_i, pos_j = np.where(np.triu(Y > 0.5, k=1))
    pos_pairs = list(zip(pos_i.tolist(), pos_j.tolist()))
    n_test = max(1, int(len(pos_pairs) * test_ratio))
    perm = rng.permutation(len(pos_pairs))
    test_pairs  = [pos_pairs[perm[i]] for i in range(n_test)]
    train_pairs = [pos_pairs[perm[i]] for i in range(n_test, len(pos_pairs))]
    Y_train = Y.copy()
    for (i, j) in test_pairs:
        Y_train[i, j] = 0.0
        Y_train[j, i] = 0.0
    return Y_train, train_pairs, test_pairs


def sample_negatives(Y, pos_pairs, ratio, seed):
    rng = np.random.default_rng(seed)
    pos_set = {(i, j) for i, j in pos_pairs} | {(j, i) for i, j in pos_pairs}
    n = Y.shape[0]
    negs = [(i, j) for i in range(n) for j in range(i + 1, n)
            if (i, j) not in pos_set and Y[i, j] < 0.5]
    n_neg = min(int(len(pos_pairs) * ratio), len(negs))
    idx = rng.choice(len(negs), size=n_neg, replace=False)
    return [negs[i] for i in idx]


def eval_scores_sampled(score_matrix, pos_pairs, neg_pairs):
    rows = [p[0] for p in pos_pairs] + [p[0] for p in neg_pairs]
    cols = [p[1] for p in pos_pairs] + [p[1] for p in neg_pairs]
    y_true = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    y_scr  = [float(score_matrix[r, c]) for r, c in zip(rows, cols)]
    try:
        auc = float(roc_auc_score(y_true, y_scr))
        ap  = float(average_precision_score(y_true, y_scr))
    except Exception:
        auc = ap = float("nan")
    return auc, ap


def eval_scores_global(score_matrix, Y_original, test_pairs):
    """Global pooled ranking: test positives vs ALL true negatives (Y_original<0.5)."""
    n = score_matrix.shape[0]
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    neg_mask = upper_mask & (Y_original < 0.5)
    neg_i, neg_j = np.where(neg_mask)
    pos_scores = np.array([score_matrix[i, j] for i, j in test_pairs]) if test_pairs else np.array([])
    neg_scores = score_matrix[neg_i, neg_j]
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    try:
        auc = float(roc_auc_score(labels, scores))
        ap  = float(average_precision_score(labels, scores))
    except Exception:
        auc = ap = float("nan")
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    random_baseline = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else float("nan")
    if n_pos > 0:
        order = np.argsort(-scores)
        topk_labels = labels[order[:n_pos]]
        precision_at_k = float(topk_labels.sum() / n_pos)
        recall_at_k = float(topk_labels.sum() / n_pos)
    else:
        precision_at_k = recall_at_k = float("nan")
    return {
        "test_auc_all_candidates": auc, "test_ap_all_candidates": ap,
        "all_candidate_random_ap_baseline": random_baseline,
        "test_precision_at_K": precision_at_k, "test_recall_at_K": recall_at_k,
        "n_candidates": int(n_pos + n_neg), "n_candidate_neg": int(n_neg),
    }


def apk(rel_full, num_relevant_total, k):
    if num_relevant_total == 0:
        return float("nan")
    hits, score = 0, 0.0
    for i, rel in enumerate(rel_full[:k], start=1):
        if rel:
            hits += 1
            score += hits / i
    denom = min(num_relevant_total, k)
    return score / denom if denom > 0 else 0.0


def ndcg_at_k(rel_full, num_relevant_total, k):
    rel_topk = rel_full[:k]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(rel_topk))
    ideal_hits = min(num_relevant_total, k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return float(dcg / idcg) if idcg > 0 else float("nan")


def eval_query_level(score_matrix, Y_train_zero_filled, test_pos_pairs, titles, movie_ids, k_at=TOP_K_QUERY):
    n = score_matrix.shape[0]
    test_pos_by_query = defaultdict(set)
    for (i, j) in test_pos_pairs:
        test_pos_by_query[i].add(j)
        test_pos_by_query[j].add(i)

    rows = []
    for qi in range(n):
        test_pos_set = test_pos_by_query.get(qi, set())
        num_test_pos = len(test_pos_set)
        if num_test_pos == 0:
            continue
        candidates = [j for j in range(n) if j != qi and Y_train_zero_filled[qi, j] < 0.5]
        if not candidates:
            continue
        scores = np.array([score_matrix[qi, j] for j in candidates])
        order  = np.argsort(-scores)
        ranked = [candidates[idx] for idx in order]
        rel    = [1 if j in test_pos_set else 0 for j in ranked]

        topk = ranked[:k_at]
        rel_topk = rel[:k_at]
        precision = float(sum(rel_topk) / k_at)
        recall    = float(sum(rel_topk) / num_test_pos)
        ndcg      = ndcg_at_k(rel, num_test_pos, k_at)
        ap        = apk(rel, num_test_pos, k_at)
        hit       = float(1.0 if sum(rel_topk) > 0 else 0.0)

        rec_titles = [titles[j] for j in topk]
        hit_titles = [titles[j] for j, r in zip(topk, rel_topk) if r]
        rec_scores = [float(scores[order[idx]]) for idx in range(min(k_at, len(order)))]

        rows.append({
            "query_movie_id": int(movie_ids[qi]), "query_title": titles[qi],
            "number_of_test_positive_items": num_test_pos,
            "precision_at_10": precision, "recall_at_10": recall,
            "ndcg_at_10": ndcg, "ap_at_10": ap, "hit_rate_at_10": hit,
            "recommended_titles_top10": " | ".join(rec_titles),
            "hit_titles_top10": " | ".join(hit_titles),
            "model_score_top10": " | ".join(f"{s:.4f}" for s in rec_scores),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────

def build_baseline_scores(like_count_arr, X, Y_colike_count, test_pairs_for_split):
    n = len(like_count_arr)
    pop = np.outer(like_count_arr, like_count_arr).astype(float)
    np.fill_diagonal(pop, 0.0)

    genre = cosine_sim_matrix(X)
    np.fill_diagonal(genre, 0.0)

    upper = np.triu_indices(n, k=1)
    pop_z = np.zeros_like(pop)
    genre_z = np.zeros_like(genre)
    pop_vals = pop[upper]
    genre_vals = genre[upper]
    pop_mu, pop_sd = pop_vals.mean(), pop_vals.std() if pop_vals.std() > 0 else 1.0
    genre_mu, genre_sd = genre_vals.mean(), genre_vals.std() if genre_vals.std() > 0 else 1.0
    combo_vals = (pop_vals - pop_mu) / pop_sd + (genre_vals - genre_mu) / genre_sd
    combo = np.zeros_like(pop)
    combo[upper] = combo_vals
    combo[(upper[1], upper[0])] = combo_vals

    item_item = Y_colike_count.copy()
    for (i, j) in test_pairs_for_split:
        item_item[i, j] = 0.0
        item_item[j, i] = 0.0

    return {
        "popularity": pop,
        "genre_cosine": genre,
        "popularity_genre": combo,
        "item_item": item_item,
    }


# ─────────────────────────────────────────────────────────────────────
# Query movie auto-selection
# ─────────────────────────────────────────────────────────────────────

def select_query_movies(meta, titles, target_n=8):
    priority_idx = []
    for pt in PRIORITY_QUERY_TITLES:
        for i, t in enumerate(titles):
            if pt.lower() in t.lower():
                priority_idx.append(i)
                break

    if len(priority_idx) >= target_n:
        return priority_idx[:target_n]

    order = meta["ratings_count"].values.argsort()[::-1]
    chosen = list(priority_idx)
    seen_genres = {meta["primary_genre"].iloc[i] for i in chosen}
    for i in order:
        if len(chosen) >= target_n:
            break
        if i in chosen:
            continue
        g = meta["primary_genre"].iloc[i]
        if g not in seen_genres:
            chosen.append(int(i))
            seen_genres.add(g)
    for i in order:
        if len(chosen) >= target_n:
            break
        if i not in chosen:
            chosen.append(int(i))
    return chosen[:target_n]


def recommend_topN(score_matrix, query_idx, exclude_idx_set, n_top=10):
    n = score_matrix.shape[0]
    candidates = [j for j in range(n) if j != query_idx and j not in exclude_idx_set]
    scores = np.array([score_matrix[query_idx, j] for j in candidates])
    order = np.argsort(-scores)[:n_top]
    return [(candidates[i], float(scores[i])) for i in order]


# ─────────────────────────────────────────────────────────────────────
# Figure helpers
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


def make_data_distribution_figure(Y_colike_count, like_count_arr, lift, chosen_density):
    n = Y_colike_count.shape[0]
    upper = np.triu_indices(n, k=1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

    axes[0].hist(Y_colike_count[upper], bins=40, color="#2196F3", alpha=0.8, edgecolor="white")
    axes[0].set_title("Y_colike_count distribution")
    axes[0].set_xlabel("co-like count")

    axes[1].hist(like_count_arr, bins=30, color="#4CAF50", alpha=0.8, edgecolor="white")
    axes[1].set_title("like_count (rating>=4) distribution")
    axes[1].set_xlabel("like_count per movie")

    finite_lift = lift[upper]
    finite_lift = finite_lift[np.isfinite(finite_lift)]
    axes[2].hist(np.clip(finite_lift, 0, np.percentile(finite_lift, 99)), bins=40,
                 color="#FF9800", alpha=0.8, edgecolor="white")
    axes[2].set_title("lift distribution (clipped at p99)")
    axes[2].set_xlabel("lift")

    axes[3].bar(["selected\nY_lift_binary"], [chosen_density], color="#9C27B0", alpha=0.8)
    axes[3].set_ylim(0, max(0.3, chosen_density * 1.5))
    axes[3].set_title("Selected Y_lift_binary density")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.25)
    fig.suptitle("MovieLens co-like data distribution", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "movielens_colike_data_distribution")


def make_poisson_k_metrics_figure(agg_df):
    ks = agg_df["k"].values
    metrics = [
        ("bic_mean", "BIC", "#607D8B"),
        ("rmse_y_mean", "RMSE_Y", "#F44336"),
        ("pearson_mean", "Pearson(Y,Y_hat)", "#2196F3"),
        ("high_colike_ap_mean", "high_colike AP", "#4CAF50"),
        ("nmi_mean", "NMI (primary genre)", "#9C27B0"),
        ("runtime_mean", "runtime (s)", "#FF9800"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.8 * len(metrics), 4.0))
    for ax, (col, label, color) in zip(axes, metrics):
        ax.plot(ks, agg_df[col].values, "o-", color=color, linewidth=2, markersize=7)
        ax.set_xlabel("k")
        ax.set_title(label, fontsize=9)
        ax.set_xticks(ks.tolist())
        ax.grid(True, linestyle="--", alpha=0.25)
    fig.suptitle("MovieLens co-like Poisson — Metrics vs k", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "movielens_colike_poisson_k_metrics")


def make_baseline_comparison_figure(baseline_df):
    methods = baseline_df["method"].unique().tolist()
    metrics = ["ap_sampled_mean", "ap_all_candidates_mean", "ndcg_at_10_mean", "map_at_10_mean"]
    labels = ["AP (sampled)", "AP (all-candidates)", "NDCG@10", "MAP@10"]

    x = np.arange(len(methods))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2196F3", "#F44336", "#4CAF50", "#9C27B0"]
    for mi, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [baseline_df[baseline_df["method"] == m][metric].mean() for m in methods]
        ax.bar(x + mi * width, vals, width, label=label, color=color, alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("score")
    ax.set_title("MovieLens co-like — proposed vs baselines (mean over splits)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "movielens_colike_baseline_comparison")


def make_factor_correlation_heatmap(corr_df, k_val):
    pivot = corr_df.pivot(index="factor", columns="attribute", values="pearson_corr")
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * pivot.shape[1]), max(4, 0.6 * pivot.shape[0])))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=75, ha="right", fontsize=6)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f"factor {i}" for i in pivot.index], fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson corr")
    ax.set_title(f"Factor x attribute correlation (k={k_val})", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "movielens_colike_factor_correlation_heatmap")


def make_z_by_factor_figure(Z_est, genre_labels, k_val):
    if Z_est.shape[1] > 2:
        Z_2d = PCA(n_components=2).fit_transform(Z_est)
        pca_used = True
    else:
        Z_2d = Z_est
        pca_used = False
    unique_labels = sorted(np.unique(genre_labels).tolist())
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for ci, lbl in enumerate(unique_labels):
        mask = genre_labels == lbl
        gname = GENRES[lbl] if lbl < len(GENRES) else str(lbl)
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], color=cmap(ci), label=gname,
                   s=40, alpha=0.8, edgecolors="none")
    ax.set_xlabel("PC1" if pca_used else "Z1")
    ax.set_ylabel("PC2" if pca_used else "Z2")
    ax.set_title(f"MovieLens co-like — Z by primary genre (k={k_val})\n"
                 "NOTE: primary genre is approximate", fontsize=9)
    ax.legend(fontsize=6, ncol=2, framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "movielens_colike_z_by_factor")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print("=== Loading data ===")
    ratings_df = load_raw_ratings()
    X, movie_ids, genre_labels, meta, titles = load_subset()
    n, d = X.shape
    print(f"  subset n={n}, d={d}  (reused genre_stratified_mp100)")

    attrs, like_users, n_users = compute_movie_attributes(ratings_df, movie_ids)
    attrs["release_year"] = [parse_release_year(t) for t in titles]
    like_count_arr = attrs["like_count"].values.astype(float)
    print(f"  n_users(full ml-100k)={n_users}")
    print(f"  like_count: mean={like_count_arr.mean():.1f} min={like_count_arr.min():.0f} max={like_count_arr.max():.0f}")

    Y_colike_count = compute_Y_colike_count(movie_ids, like_users)
    expected, lift = compute_lift_matrix(Y_colike_count, like_count_arr, n_users)
    genre_sim = cosine_sim_matrix(X)

    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    colike_density_pos = float((Y_colike_count[upper_mask] > 0).mean())
    colike_mean = float(Y_colike_count[upper_mask].mean())
    print(f"  Y_colike_count: density_pos={colike_density_pos:.4f} mean={colike_mean:.2f} "
          f"max={Y_colike_count.max():.0f}")

    # ── Lift threshold audit ───────────────────────────────────────
    print("\n=== Lift threshold audit ===")
    lift_audit_df = audit_lift_candidates(Y_colike_count, lift, like_count_arr, genre_sim, titles)
    min_support, lift_threshold, lift_reason, chosen_row = select_lift_setting(lift_audit_df)
    print(f"  Selected: min_support={min_support}, lift_threshold={lift_threshold}")
    print(f"  Reason: {lift_reason}")
    print(f"  -> positive_edges={int(chosen_row['positive_edges'])}, density={chosen_row['density']:.4f}")

    high_colike_threshold, high_colike_density, hc_audit_df = decide_high_colike_threshold(Y_colike_count)
    print(f"  high_colike_threshold={high_colike_threshold:.0f} (density={high_colike_density:.4f})")

    threshold_audit_df = pd.concat([lift_audit_df, hc_audit_df], ignore_index=True)
    threshold_audit_path = OUT_DIR / "movielens_colike_threshold_audit.csv"
    threshold_audit_df.to_csv(threshold_audit_path, index=False)
    print(f"Saved: {threshold_audit_path}")

    Y_lift_binary = ((Y_colike_count >= min_support) & (lift >= lift_threshold)).astype(np.float64)
    np.fill_diagonal(Y_lift_binary, 0.0)
    lift_density = float(Y_lift_binary[upper_mask].mean())
    lift_n_positive = int(Y_lift_binary[upper_mask].sum())
    print(f"  Y_lift_binary: density={lift_density:.4f} n_positive={lift_n_positive}")

    # ── Data summary CSV ────────────────────────────────────────────
    data_summary = pd.DataFrame([{
        "subset_strategy": "genre_stratified_mp100", "n": n, "d": d,
        "family_x": FAMILY_X, "n_users_full_ml100k": n_users,
        "like_threshold_rating": LIKE_THRESHOLD,
        "Y_colike_count_density_pos": colike_density_pos,
        "Y_colike_count_mean": colike_mean,
        "Y_colike_count_max": float(Y_colike_count.max()),
        "Y_lift_binary_min_support": min_support,
        "Y_lift_binary_lift_threshold": lift_threshold,
        "Y_lift_binary_density": lift_density,
        "Y_lift_binary_n_positive": lift_n_positive,
        "lift_selection_reason": lift_reason,
        "high_colike_threshold": high_colike_threshold,
        "high_colike_density": high_colike_density,
    }])
    data_summary_path = OUT_DIR / "movielens_colike_data_summary.csv"
    data_summary.to_csv(data_summary_path, index=False)
    print(f"Saved: {data_summary_path}")

    make_data_distribution_figure(Y_colike_count, like_count_arr, lift, lift_density)

    # ─────────────────────────────────────────────────────────────
    # Experiment A: Poisson co-like count
    # ─────────────────────────────────────────────────────────────
    total_a = len(K_LIST_A) * N_TRIALS_A
    print(f"\n=== Experiment A (Poisson): k={K_LIST_A}, {N_TRIALS_A} trials = {total_a} fits ===")

    a_rows = []
    a_fits = {}   # (k, trial) -> res
    fit_count = 0
    for k_val in K_LIST_A:
        for trial in range(N_TRIALS_A):
            fit_count += 1
            seed = SEED_BASE_A + k_val * 100 + trial * 10
            print(f"\n[A {fit_count:2d}/{total_a}] k={k_val} trial={trial} seed={seed}")
            t0 = time.perf_counter()
            try:
                res = run_em_poisson(X, Y_colike_count, k_val, seed, L=L, num_iter=NITER)
                elapsed = time.perf_counter() - t0
                ev = evaluate_poisson_fit(res, X, Y_colike_count, high_colike_threshold)
                sil, nmi, ari, _ = compute_z_metrics(res["Z_est"], genre_labels)
                success, err_msg = True, res.get("error_message", "")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                traceback.print_exc()
                res = {"Z_est": None, "bic": float("nan"), "Q_strict": float("nan"),
                       "w0": float("nan"), "w": float("nan"),
                       "nan_occurred": True, "nan_count": 999}
                ev = {"rmse_y": float("nan"), "mae_y": float("nan"),
                      "pearson_corr": float("nan"), "spearman_corr": float("nan"),
                      "high_colike_auc": float("nan"), "high_colike_ap": float("nan"),
                      "high_colike_threshold": high_colike_threshold, "rmse_x": float("nan")}
                sil = nmi = ari = float("nan")
                success, err_msg = False, str(e)

            a_fits[(k_val, trial)] = res
            print(f"  BIC={res['bic']:.1f}  RMSE_Y={ev['rmse_y']:.3f}  Pearson={ev['pearson_corr']:.3f}"
                  f"  hc_AP={ev['high_colike_ap']:.4f}  NMI={nmi:.4f}  [{elapsed:.1f}s]"
                  f"  w0={res['w0']:.3f} w={res['w']:.3f}")

            a_rows.append({
                "k": k_val, "trial": trial, "seed": seed,
                "bic": res["bic"], "q_strict": res["Q_strict"],
                "rmse_y": ev["rmse_y"], "mae_y": ev["mae_y"],
                "pearson_corr": ev["pearson_corr"], "spearman_corr": ev["spearman_corr"],
                "high_colike_ap": ev["high_colike_ap"], "high_colike_auc": ev["high_colike_auc"],
                "high_colike_threshold": ev["high_colike_threshold"],
                "rmse_x": ev["rmse_x"],
                "nmi_primary_genre": nmi, "ari_primary_genre": ari, "silhouette": sil,
                "w0": res["w0"], "w": res["w"],
                "success": success, "nan_occurred": res["nan_occurred"],
                "runtime_seconds": round(elapsed, 2), "error_message": err_msg,
            })

    a_df = pd.DataFrame(a_rows)
    a_summary_path = OUT_DIR / "movielens_colike_poisson_summary.csv"
    a_df.to_csv(a_summary_path, index=False)
    print(f"\nSaved: {a_summary_path}")

    agg_rows = []
    for k_val in K_LIST_A:
        sub = a_df[a_df["k"] == k_val]
        n_ok = int(sub["success"].sum())
        row = {"k": k_val, "n_trials": len(sub), "n_success": n_ok,
               "success_rate": n_ok / len(sub) if len(sub) else float("nan")}
        for col, out_name in [
            ("bic", "bic_mean"), ("rmse_y", "rmse_y_mean"), ("mae_y", "mae_y_mean"),
            ("pearson_corr", "pearson_mean"), ("spearman_corr", "spearman_mean"),
            ("high_colike_ap", "high_colike_ap_mean"),
            ("nmi_primary_genre", "nmi_mean"), ("ari_primary_genre", "ari_mean"),
            ("runtime_seconds", "runtime_mean"),
        ]:
            vals = sub[col].dropna().astype(float)
            row[out_name] = float(vals.mean()) if len(vals) else float("nan")
        agg_rows.append(row)
    a_agg_df = pd.DataFrame(agg_rows)
    a_agg_path = OUT_DIR / "movielens_colike_poisson_agg.csv"
    a_agg_df.to_csv(a_agg_path, index=False)
    print(f"Saved: {a_agg_path}")

    make_poisson_k_metrics_figure(a_agg_df)

    # ── Best-k for interpretation ────────────────────────────────────
    cand_df = a_agg_df[a_agg_df["k"].isin(BEST_K_INTERPRETATION_CANDIDATES)].copy()
    if len(cand_df) > 0 and cand_df["pearson_mean"].notna().any():
        best_k_for_interpretation = int(cand_df.loc[cand_df["pearson_mean"].idxmax(), "k"])
    else:
        best_k_for_interpretation = BEST_K_INTERPRETATION_CANDIDATES[0]
    print(f"\nbest_k_for_interpretation = {best_k_for_interpretation} "
          f"(chosen among {BEST_K_INTERPRETATION_CANDIDATES} by Pearson(Y,Y_hat))")

    sub_best = a_df[(a_df["k"] == best_k_for_interpretation) & a_df["success"]]
    if len(sub_best) > 0:
        best_trial = int(sub_best.loc[sub_best["pearson_corr"].idxmax(), "trial"])
    else:
        best_trial = 0
    best_res = a_fits[(best_k_for_interpretation, best_trial)]
    print(f"  representative fit: k={best_k_for_interpretation} trial={best_trial}")

    bestk_summary = pd.DataFrame([{
        "best_k_for_interpretation": best_k_for_interpretation,
        "candidates": str(BEST_K_INTERPRETATION_CANDIDATES),
        "selection_rule": "max mean Pearson(Y,Y_hat) among candidate k in {5,8}",
        "representative_trial": best_trial,
        "bic": best_res["bic"], "w0": best_res["w0"], "w": best_res["w"],
    }])
    bestk_path = OUT_DIR / "movielens_colike_bestk_summary.csv"
    bestk_summary.to_csv(bestk_path, index=False)
    print(f"Saved: {bestk_path}")

    # ─────────────────────────────────────────────────────────────
    # K-interpretation analysis (on best_k_for_interpretation fit)
    # ─────────────────────────────────────────────────────────────
    print("\n=== K-interpretation analysis ===")
    Z_best = best_res["Z_est"]
    k_best = Z_best.shape[1]

    # 1. factor top/bottom movies
    top_movie_rows = []
    for r in range(k_best):
        zvals = Z_best[:, r]
        order_high = np.argsort(-zvals)[:10]
        order_low  = np.argsort(zvals)[:10]
        for side, order in [("high", order_high), ("low", order_low)]:
            for rank, idx in enumerate(order, start=1):
                top_movie_rows.append({
                    "k": best_k_for_interpretation, "trial": best_trial, "factor": r,
                    "side": side, "rank": rank,
                    "movie_id": int(movie_ids[idx]), "title": titles[idx],
                    "genres": "|".join(g for g in GENRES if meta[g].iloc[idx] == 1),
                    "primary_genre": meta["primary_genre"].iloc[idx],
                    "z_value": float(zvals[idx]),
                    "rating_count": int(attrs["rating_count"].iloc[idx]),
                    "like_count": int(attrs["like_count"].iloc[idx]),
                    "avg_rating": float(attrs["avg_rating"].iloc[idx]),
                    "high_rating_rate": float(attrs["high_rating_rate"].iloc[idx]),
                })
    top_movies_df = pd.DataFrame(top_movie_rows)
    top_movies_path = OUT_DIR / "movielens_colike_factor_top_movies.csv"
    top_movies_df.to_csv(top_movies_path, index=False)
    print(f"Saved: {top_movies_path}")

    # 2. factor x attribute correlations
    attr_cols = ["rating_count", "log_rating_count", "like_count", "log_like_count",
                 "avg_rating", "high_rating_rate"]
    attr_matrix = {c: attrs[c].values.astype(float) for c in attr_cols}
    if attrs["release_year"].notna().any():
        attr_matrix["release_year"] = attrs["release_year"].values.astype(float)
    for g in GENRES:
        attr_matrix[f"genre_{g}"] = meta[g].values.astype(float)

    category_map = {
        "rating_count": "popularity-related", "log_rating_count": "popularity-related",
        "like_count": "popularity-related", "log_like_count": "popularity-related",
        "avg_rating": "high-rating-related", "high_rating_rate": "high-rating-related",
        "genre_Animation": "animation/family-related", "genre_Children's": "animation/family-related",
        "genre_Musical": "animation/family-related",
        "genre_Sci-Fi": "sci-fi/action-related", "genre_Action": "sci-fi/action-related",
        "genre_Adventure": "sci-fi/action-related", "genre_War": "sci-fi/action-related",
        "genre_Romance": "romance/drama-related", "genre_Drama": "romance/drama-related",
        "genre_Comedy": "romance/drama-related",
    }

    corr_rows = []
    for r in range(k_best):
        zvals = Z_best[:, r]
        pcorrs = {}
        for attr, vals in attr_matrix.items():
            mask = np.isfinite(vals)
            if mask.sum() < 3 or np.std(vals[mask]) == 0:
                pc, sc = float("nan"), float("nan")
            else:
                try:
                    pc, _ = pearsonr(zvals[mask], vals[mask])
                except Exception:
                    pc = float("nan")
                try:
                    sc, _ = spearmanr(zvals[mask], vals[mask])
                except Exception:
                    sc = float("nan")
            pcorrs[attr] = (float(pc), float(sc))
        abs_order = sorted(pcorrs.keys(), key=lambda a: abs(pcorrs[a][0]) if np.isfinite(pcorrs[a][0]) else -1,
                            reverse=True)
        rank_map = {a: i + 1 for i, a in enumerate(abs_order)}
        for attr, (pc, sc) in pcorrs.items():
            if np.isfinite(pc) and abs(pc) >= 0.25:
                hint = "suggestive: " + category_map.get(attr, "weak/no clear interpretation")
            else:
                hint = "suggestive: weak/no clear interpretation"
            corr_rows.append({
                "factor": r, "attribute": attr, "pearson_corr": pc, "spearman_corr": sc,
                "abs_pearson_rank": rank_map[attr], "interpretation_hint": hint,
            })
    corr_df = pd.DataFrame(corr_rows)
    corr_path = OUT_DIR / "movielens_colike_factor_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"Saved: {corr_path}")

    make_factor_correlation_heatmap(corr_df, best_k_for_interpretation)
    make_z_by_factor_figure(Z_best, genre_labels, best_k_for_interpretation)

    # 3. factor interpretation summary
    interp_rows = []
    for r in range(k_best):
        sub_c = corr_df[corr_df["factor"] == r].copy()
        sub_c_valid = sub_c[sub_c["pearson_corr"].notna()]
        top_pos = sub_c_valid.sort_values("pearson_corr", ascending=False).head(3)
        top_neg = sub_c_valid.sort_values("pearson_corr", ascending=True).head(3)
        max_abs = sub_c_valid["pearson_corr"].abs().max() if len(sub_c_valid) else 0.0

        if max_abs >= 0.5:
            confidence = "medium"
        elif max_abs >= 0.3:
            confidence = "low-medium"
        else:
            confidence = "low"

        # Label by the SINGLE strongest |pearson| correlate (not a magnitude-blind majority
        # vote over top_pos+top_neg attribute names), so a few weak same-category negative
        # correlations can't outvote one dominant positive correlation in a different category.
        sub_c_ranked = sub_c[sub_c["pearson_corr"].notna()].sort_values("abs_pearson_rank")
        if len(sub_c_ranked) > 0 and abs(sub_c_ranked.iloc[0]["pearson_corr"]) >= 0.25:
            top1 = sub_c_ranked.iloc[0]
            cat = category_map.get(top1["attribute"], "unclear/mixed")
            direction = "positively" if top1["pearson_corr"] > 0 else "negatively"
            tentative_label = (
                f"tentative: {cat} ({direction} associated with {top1['attribute']}, "
                f"r={top1['pearson_corr']:.2f})"
            )
        else:
            tentative_label = "tentative: unclear/mixed (no attribute with |pearson_corr|>=0.25)"

        top_high = top_movies_df[(top_movies_df["factor"] == r) & (top_movies_df["side"] == "high")]["title"].head(5).tolist()
        top_low  = top_movies_df[(top_movies_df["factor"] == r) & (top_movies_df["side"] == "low")]["title"].head(5).tolist()

        interp_rows.append({
            "factor": r, "tentative_label": tentative_label,
            "strongest_positive_correlations": "; ".join(
                f"{a}({v:.2f})" for a, v in zip(top_pos["attribute"], top_pos["pearson_corr"])),
            "strongest_negative_correlations": "; ".join(
                f"{a}({v:.2f})" for a, v in zip(top_neg["attribute"], top_neg["pearson_corr"])),
            "top_high_movies": " | ".join(top_high),
            "top_low_movies": " | ".join(top_low),
            "interpretation_confidence": confidence,
            "caution": ("Z has rotation non-identifiability; this label is suggestive/tentative, "
                        "not a confirmed semantic axis."),
        })
    interp_df = pd.DataFrame(interp_rows)
    interp_path = OUT_DIR / "movielens_colike_factor_interpretation_summary.csv"
    interp_df.to_csv(interp_path, index=False)
    print(f"Saved: {interp_path}")

    # 4. top predicted pairs (Poisson mu_y)
    mu_y_best = best_res["mu_y"]
    w0_best, w_best = best_res["w0"], best_res["w"]
    iu, ju = np.triu_indices(n, k=1)
    pred_vals = mu_y_best[iu, ju]
    order = np.argsort(-pred_vals)[:50]

    pair_rows = []
    for rank, idx in enumerate(order, start=1):
        i, j = int(iu[idx]), int(ju[idx])
        contribs = w_best * Z_best[i, :] * Z_best[j, :]
        order_c = np.argsort(-np.abs(contribs))
        f1, f2 = int(order_c[0]), int(order_c[1]) if k_best > 1 else int(order_c[0])
        pair_rows.append({
            "rank": rank, "movie_i": int(movie_ids[i]), "title_i": titles[i],
            "genres_i": "|".join(g for g in GENRES if meta[g].iloc[i] == 1),
            "movie_j": int(movie_ids[j]), "title_j": titles[j],
            "genres_j": "|".join(g for g in GENRES if meta[g].iloc[j] == 1),
            "observed_colike_count": float(Y_colike_count[i, j]),
            "predicted_colike_count": float(mu_y_best[i, j]),
            "lift": float(lift[i, j]),
            "like_count_i": int(attrs["like_count"].iloc[i]),
            "like_count_j": int(attrs["like_count"].iloc[j]),
            "genre_cosine": float(genre_sim[i, j]),
            "model_score": float(mu_y_best[i, j]),
            "top_contributing_factor_1": f1, "top_contributing_factor_2": f2,
            "contribution_factor_1": float(contribs[f1]), "contribution_factor_2": float(contribs[f2]),
        })
    top_pairs_df = pd.DataFrame(pair_rows)
    top_pairs_path = OUT_DIR / "movielens_colike_top_predicted_pairs.csv"
    top_pairs_df.to_csv(top_pairs_path, index=False)
    print(f"Saved: {top_pairs_path}")

    # 6. pair factor contributions (for the same top-50 pairs, all factors)
    label_by_factor = dict(zip(interp_df["factor"], interp_df["tentative_label"]))
    contrib_rows = []
    for idx in order:
        i, j = int(iu[idx]), int(ju[idx])
        contribs = w_best * Z_best[i, :] * Z_best[j, :]
        rank_order = np.argsort(-np.abs(contribs))
        rank_map = {int(f): pos + 1 for pos, f in enumerate(rank_order)}
        for r in range(k_best):
            contrib_rows.append({
                "movie_i": int(movie_ids[i]), "title_i": titles[i],
                "movie_j": int(movie_ids[j]), "title_j": titles[j],
                "factor": r, "z_i_factor": float(Z_best[i, r]), "z_j_factor": float(Z_best[j, r]),
                "factor_contribution": float(contribs[r]),
                "absolute_contribution_rank": rank_map[r],
                "tentative_factor_label": label_by_factor.get(r, ""),
            })
    contrib_df = pd.DataFrame(contrib_rows)
    contrib_path = OUT_DIR / "movielens_colike_pair_factor_contributions.csv"
    contrib_df.to_csv(contrib_path, index=False)
    print(f"Saved: {contrib_path}")

    # ─────────────────────────────────────────────────────────────
    # Experiment B: Bernoulli lift link prediction (held-out)
    # ─────────────────────────────────────────────────────────────
    total_b = len(K_LIST_B) * len(SPLIT_TRIALS_B) * len(MODEL_TRIALS_B)
    print(f"\n=== Experiment B (Bernoulli lift): k={K_LIST_B}, splits={SPLIT_TRIALS_B}, "
          f"models={MODEL_TRIALS_B} = {total_b} fits ===")
    print(f"  evaluation_mode={EVALUATION_MODE}, test_edge_ratio={TEST_EDGE_RATIO}, "
          f"neg_ratio={NEG_RATIO}, sampled_random_AP_baseline={RANDOM_AP_SAMPLED:.4f}")

    split_cache = {}
    for split_trial in SPLIT_TRIALS_B:
        split_seed = SPLIT_SEED_BASE_B + split_trial * 100
        Y_train, train_pos, test_pos = split_positive_edges(Y_lift_binary, TEST_EDGE_RATIO, split_seed)
        neg_seed_tr = split_seed + 50
        neg_seed_te = split_seed + 60
        train_neg = sample_negatives(Y_lift_binary, train_pos, NEG_RATIO, neg_seed_tr)
        test_neg  = sample_negatives(Y_lift_binary, test_pos, NEG_RATIO, neg_seed_te)
        split_cache[split_trial] = dict(Y_train=Y_train, train_pos=train_pos, test_pos=test_pos,
                                         train_neg=train_neg, test_neg=test_neg)

    b_rows = []
    b_fits = {}
    query_rows_all = []
    fit_count = 0
    for k_val in K_LIST_B:
        for split_trial in SPLIT_TRIALS_B:
            sc = split_cache[split_trial]
            for model_trial in MODEL_TRIALS_B:
                fit_count += 1
                model_seed = MODEL_SEED_BASE_B + k_val * 100 + split_trial * 10 + model_trial
                print(f"\n[B {fit_count:2d}/{total_b}] k={k_val} split={split_trial} "
                      f"model={model_trial} seed={model_seed}")
                t0 = time.perf_counter()
                try:
                    res = run_em_bernoulli(X, sc["Y_train"], k_val, model_seed, L=L, num_iter=NITER)
                    elapsed = time.perf_counter() - t0
                    mu_y = res["mu_y"]
                    train_auc, train_ap = eval_scores_sampled(mu_y, sc["train_pos"], sc["train_neg"])
                    test_auc, test_ap = eval_scores_sampled(mu_y, sc["test_pos"], sc["test_neg"])
                    glob = eval_scores_global(mu_y, Y_lift_binary, sc["test_pos"])
                    qrows = eval_query_level(mu_y, sc["Y_train"], sc["test_pos"], titles, movie_ids)
                    sil, nmi, ari, _ = compute_z_metrics(res["Z_est"], genre_labels)
                    success, err_msg = True, res.get("error_message", "")
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    traceback.print_exc()
                    res = {"bic": float("nan"), "Q_strict": float("nan"), "num_params": 0,
                           "w0": float("nan"), "w": float("nan"), "Z_est": None,
                           "nan_occurred": True, "nan_count": 999}
                    train_auc = train_ap = test_auc = test_ap = float("nan")
                    glob = {"test_auc_all_candidates": float("nan"), "test_ap_all_candidates": float("nan"),
                            "all_candidate_random_ap_baseline": float("nan"),
                            "test_precision_at_K": float("nan"), "test_recall_at_K": float("nan")}
                    qrows = []
                    sil = nmi = ari = float("nan")
                    success, err_msg = False, str(e)

                b_fits[(k_val, split_trial, model_trial)] = res
                if qrows:
                    qdf_fit = pd.DataFrame(qrows)
                    q_agg = qdf_fit[["precision_at_10", "recall_at_10", "ndcg_at_10",
                                      "ap_at_10", "hit_rate_at_10"]].mean()
                    for qr in qrows:
                        qr2 = dict(qr)
                        qr2.update({"k": k_val, "split_trial": split_trial, "method": "proposed_dual_expfam"})
                        query_rows_all.append(qr2)
                else:
                    q_agg = pd.Series({"precision_at_10": float("nan"), "recall_at_10": float("nan"),
                                        "ndcg_at_10": float("nan"), "ap_at_10": float("nan"),
                                        "hit_rate_at_10": float("nan")})

                test_ap_over_random = test_ap / RANDOM_AP_SAMPLED if np.isfinite(test_ap) else float("nan")

                print(f"  BIC={res['bic']:.1f}  te_AUC_sampled={test_auc:.4f}  te_AP_sampled={test_ap:.4f}"
                      f"  te_AP_all={glob['test_ap_all_candidates']:.4f}  NDCG@10={q_agg['ndcg_at_10']:.4f}"
                      f"  NMI={nmi:.4f}  [{elapsed:.1f}s]")

                b_rows.append({
                    "k": k_val, "split_trial": split_trial, "model_trial": model_trial,
                    "model_seed": model_seed, "evaluation_mode": EVALUATION_MODE,
                    "n_train_pos": len(sc["train_pos"]), "n_test_pos": len(sc["test_pos"]),
                    "n_train_neg": len(sc["train_neg"]), "n_test_neg": len(sc["test_neg"]),
                    "sampled_random_ap_baseline": RANDOM_AP_SAMPLED,
                    "train_auc_sampled": train_auc, "train_ap_sampled": train_ap,
                    "test_auc_sampled": test_auc, "test_ap_sampled": test_ap,
                    "test_ap_over_sampled_random": test_ap_over_random,
                    "test_auc_all_candidates": glob["test_auc_all_candidates"],
                    "test_ap_all_candidates": glob["test_ap_all_candidates"],
                    "all_candidate_random_ap_baseline": glob["all_candidate_random_ap_baseline"],
                    "test_precision_at_K": glob["test_precision_at_K"],
                    "test_recall_at_K": glob["test_recall_at_K"],
                    "test_precision_at_10": q_agg["precision_at_10"],
                    "test_recall_at_10": q_agg["recall_at_10"],
                    "test_ndcg_at_10": q_agg["ndcg_at_10"],
                    "test_map_at_10": q_agg["ap_at_10"],
                    "test_hit_rate_at_10": q_agg["hit_rate_at_10"],
                    "bic": res["bic"], "q_strict": res["Q_strict"], "num_params": res.get("num_params", 0),
                    "nmi": nmi, "ari": ari, "silhouette": sil,
                    "w0": res["w0"], "w": res["w"],
                    "nan_occurred": res["nan_occurred"], "success": success,
                    "runtime_seconds": round(elapsed, 2), "error_message": err_msg,
                })

    b_df = pd.DataFrame(b_rows)
    b_summary_path = OUT_DIR / "movielens_colike_lift_heldout_summary.csv"
    b_df.to_csv(b_summary_path, index=False)
    print(f"\nSaved: {b_summary_path}")

    query_df = pd.DataFrame(query_rows_all)
    query_path = OUT_DIR / "movielens_colike_query_recommendation_metrics.csv"
    query_df.to_csv(query_path, index=False)
    print(f"Saved: {query_path}")

    b_agg_rows = []
    for k_val in K_LIST_B:
        sub = b_df[b_df["k"] == k_val]
        row = {"k": k_val, "n_splits": len(sub)}
        for col, out_name in [
            ("test_ap_sampled", "test_ap_sampled_mean"), ("test_auc_sampled", "test_auc_sampled_mean"),
            ("test_ap_all_candidates", "test_ap_all_candidates_mean"),
            ("test_auc_all_candidates", "test_auc_all_candidates_mean"),
            ("test_ap_over_sampled_random", "ap_over_random_mean"),
            ("test_precision_at_10", "precision_at_10_mean"), ("test_recall_at_10", "recall_at_10_mean"),
            ("test_ndcg_at_10", "ndcg_at_10_mean"), ("test_map_at_10", "map_at_10_mean"),
            ("test_hit_rate_at_10", "hit_rate_at_10_mean"),
        ]:
            vals = sub[col].dropna().astype(float)
            row[out_name] = float(vals.mean()) if len(vals) else float("nan")
        row["sampled_random_ap_baseline"] = RANDOM_AP_SAMPLED
        row["all_candidate_random_ap_baseline_mean"] = float(sub["all_candidate_random_ap_baseline"].dropna().mean())
        b_agg_rows.append(row)
    b_agg_df = pd.DataFrame(b_agg_rows)
    b_agg_path = OUT_DIR / "movielens_colike_lift_heldout_agg.csv"
    b_agg_df.to_csv(b_agg_path, index=False)
    print(f"Saved: {b_agg_path}")

    # ─────────────────────────────────────────────────────────────
    # Baselines (same test pos/neg per split as Experiment B)
    # ─────────────────────────────────────────────────────────────
    print("\n=== Baselines ===")
    baseline_rows = []
    for split_trial in SPLIT_TRIALS_B:
        sc = split_cache[split_trial]
        scores = build_baseline_scores(like_count_arr, X, Y_colike_count, sc["test_pos"])
        for method, score_matrix in scores.items():
            auc_s, ap_s = eval_scores_sampled(score_matrix, sc["test_pos"], sc["test_neg"])
            glob = eval_scores_global(score_matrix, Y_lift_binary, sc["test_pos"])
            qrows = eval_query_level(score_matrix, sc["Y_train"], sc["test_pos"], titles, movie_ids)
            if qrows:
                qdf = pd.DataFrame(qrows)
                q_agg = qdf[["precision_at_10", "recall_at_10", "ndcg_at_10", "ap_at_10", "hit_rate_at_10"]].mean()
            else:
                q_agg = pd.Series({c: float("nan") for c in
                                    ["precision_at_10", "recall_at_10", "ndcg_at_10", "ap_at_10", "hit_rate_at_10"]})
            baseline_rows.append({
                "method": method, "k": np.nan, "split_trial": split_trial,
                "test_auc_sampled": auc_s, "test_ap_sampled": ap_s,
                "test_auc_all_candidates": glob["test_auc_all_candidates"],
                "test_ap_all_candidates": glob["test_ap_all_candidates"],
                "precision_at_10": q_agg["precision_at_10"], "recall_at_10": q_agg["recall_at_10"],
                "ndcg_at_10": q_agg["ndcg_at_10"], "map_at_10": q_agg["ap_at_10"],
                "hit_rate_at_10": q_agg["hit_rate_at_10"],
            })
            print(f"  split={split_trial} {method:16s} AP_sampled={ap_s:.4f} "
                  f"AP_all={glob['test_ap_all_candidates']:.4f} NDCG@10={q_agg['ndcg_at_10']:.4f}")

        for k_val in K_LIST_B:
            row_b = b_df[(b_df["k"] == k_val) & (b_df["split_trial"] == split_trial)]
            if len(row_b) == 0:
                continue
            row_b = row_b.iloc[0]
            baseline_rows.append({
                "method": "proposed_dual_expfam", "k": k_val, "split_trial": split_trial,
                "test_auc_sampled": row_b["test_auc_sampled"], "test_ap_sampled": row_b["test_ap_sampled"],
                "test_auc_all_candidates": row_b["test_auc_all_candidates"],
                "test_ap_all_candidates": row_b["test_ap_all_candidates"],
                "precision_at_10": row_b["test_precision_at_10"], "recall_at_10": row_b["test_recall_at_10"],
                "ndcg_at_10": row_b["test_ndcg_at_10"], "map_at_10": row_b["test_map_at_10"],
                "hit_rate_at_10": row_b["test_hit_rate_at_10"],
            })

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_agg = baseline_df.groupby("method").agg(
        ap_sampled_mean=("test_ap_sampled", "mean"),
        auc_sampled_mean=("test_auc_sampled", "mean"),
        ap_all_candidates_mean=("test_ap_all_candidates", "mean"),
        auc_all_candidates_mean=("test_auc_all_candidates", "mean"),
        precision_at_10_mean=("precision_at_10", "mean"),
        recall_at_10_mean=("recall_at_10", "mean"),
        ndcg_at_10_mean=("ndcg_at_10", "mean"),
        map_at_10_mean=("map_at_10", "mean"),
        hit_rate_at_10_mean=("hit_rate_at_10", "mean"),
    ).reset_index()
    baseline_merged = baseline_df.merge(baseline_agg, on="method", how="left")
    baseline_path = OUT_DIR / "movielens_colike_baseline_metrics.csv"
    baseline_merged.to_csv(baseline_path, index=False)
    print(f"\nSaved: {baseline_path}")

    make_baseline_comparison_figure(baseline_agg)

    # ─────────────────────────────────────────────────────────────
    # Recommendation examples (Poisson best_k_for_interpretation model)
    # ─────────────────────────────────────────────────────────────
    print("\n=== Recommendation examples ===")
    query_indices = select_query_movies(meta, titles, target_n=8)
    print(f"  query movies: {[titles[i] for i in query_indices]}")

    item_item_full = Y_colike_count.copy()
    rec_scores = {
        "proposed_poisson_colike": mu_y_best,
        "popularity": np.outer(like_count_arr, like_count_arr),
        "genre_cosine": genre_sim,
        "item_item_full_colike": item_item_full,
    }

    rec_rows = []
    for qi in query_indices:
        for method, score_matrix in rec_scores.items():
            top10 = recommend_topN(score_matrix, qi, {qi}, n_top=10)
            for rank, (j, score) in enumerate(top10, start=1):
                contribs = w_best * Z_best[qi, :] * Z_best[j, :]
                top_factor = int(np.argmax(np.abs(contribs)))
                rec_rows.append({
                    "query_movie_id": int(movie_ids[qi]), "query_title": titles[qi],
                    "query_genres": "|".join(g for g in GENRES if meta[g].iloc[qi] == 1),
                    "method": method, "rank": rank,
                    "recommended_movie_id": int(movie_ids[j]), "recommended_title": titles[j],
                    "recommended_genres": "|".join(g for g in GENRES if meta[g].iloc[j] == 1),
                    "observed_colike_count": float(Y_colike_count[qi, j]),
                    "lift": float(lift[qi, j]),
                    "score": float(score),
                    "hit_in_test_if_applicable": "",
                    "top_contributing_factor": top_factor if method == "proposed_poisson_colike" else "",
                    "contribution_explanation": (
                        f"factor {top_factor} contributes {contribs[top_factor]:.3f} to w*z_i.z_j"
                        if method == "proposed_poisson_colike" else ""
                    ),
                })
    rec_df = pd.DataFrame(rec_rows)
    rec_path = OUT_DIR / "movielens_colike_recommendation_examples.csv"
    rec_df.to_csv(rec_path, index=False)
    print(f"Saved: {rec_path}")

    # ─────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_start
    print(f"\n=== Done in {elapsed_total/60:.1f} min ===")
    print(f"Results: {OUT_DIR}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
