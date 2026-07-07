"""
共通の実データ再構成評価 (Wine / Cora / MovieLens)。

目的:
  推定された Z, F, w0, w を用いて、観測された X, Y をどれくらい再構成
  できるかを、3つの実データに共通の評価軸として計算する。
  (元論文に合わせるためではなく、潜在構造モデルの実データ評価として
   自然な共通軸を作るため。)

モデル式 (model_dual_expfam_fixed.py を直接確認して採用):
  eta_X = Z @ F.T                       (n x d)
  X_hat = model._mean_function_x(eta_X)
  eta_Y = w0 + w * (Z @ Z.T)             (n x n, scalar w)
  Y_hat = model._mean_function(eta_Y)

既存コード (model_dual_expfam.py, model_dual_expfam_fixed.py, utils_expfam.py,
exp_scenario_lib.py) は変更しない。既存の結果・図も変更しない。

データの前処理は、既存スクリプトと同一にしている:
  Wine   : expfam/src/run_fixed_real_wine_pilot.py の load_wine_data() と同一
  Cora   : expfam/src/run_fixed_real_cora_balanced_k_sweep.py の
           parse_cora()/build_balanced_degree()/build_subset() と同一
  MovieLens : expfam/src/run_fixed_real_movielens_colike_interpretation.py の
           load_subset()/load_raw_ratings()/compute_movie_attributes()/
           compute_Y_colike_count() と同一 (genre_stratified_mp100 subset)

出力:
  expfam/results/real_data/common_reconstruction_eval/common_reconstruction_summary.csv
  expfam/results/real_data/common_reconstruction_eval/wine_factor_top_features.csv
  expfam/results/real_data/common_reconstruction_eval/cora_factor_top_words.csv
  expfam/results/real_data/common_reconstruction_eval/movielens_factor_top_genres.csv
  expfam/figures/real_data/common_reconstruction_eval/wine_F_heatmap_k{3,6}.png/pdf
  expfam/figures/real_data/common_reconstruction_eval/cora_F_heatmap_k{1,3,6}.png/pdf
  expfam/figures/real_data/common_reconstruction_eval/movielens_F_heatmap_k{5,8}.png/pdf
"""

import re
import sys
import time
import traceback
import warnings
import zipfile
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.special import gammaln
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, silhouette_score,
    normalized_mutual_info_score, adjusted_rand_score,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

_SRC = Path(__file__).parent
_ROOT = _SRC.parent.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from model_dual_expfam_fixed import DualExpFamLSMFixed  # noqa: E402
from utils_expfam import calc_Q_dual_strict, calc_bic_dual  # noqa: E402

OUT_DIR = _ROOT / "expfam" / "results" / "real_data" / "common_reconstruction_eval"
FIG_DIR = _ROOT / "expfam" / "figures" / "real_data" / "common_reconstruction_eval"

L, NITER = 5, 8
N_TRIALS = 3

WINE_K_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9]
WINE_MAIN_K = [3, 6]

CORA_K_LIST = [1, 2, 3, 4, 5, 6]
CORA_MAIN_K = [1, 3, 6]
CORA_PER_CLASS = 40
CORA_D_SUBSET = 50

ML_K_LIST = [2, 3, 5, 8]
ML_MAIN_K = [5, 8]
ML_LIKE_THRESHOLD = 4

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


# ════════════════════════════════════════════════════════════════════
# Data loaders (reusing existing preprocessing exactly)
# ════════════════════════════════════════════════════════════════════

def load_wine_data():
    wine_raw = load_wine()
    X = StandardScaler().fit_transform(wine_raw.data)
    labels = wine_raw.target
    n, d = X.shape
    Y = (labels[:, None] == labels[None, :]).astype(float)
    np.fill_diagonal(Y, 0)
    feature_names = list(wine_raw.feature_names)
    return dict(X=X, Y=Y, labels=labels, n=n, d=d,
                family_x="gaussian", family_y="bernoulli",
                feature_names=feature_names, n_classes=3)


def parse_cora():
    data_dir = _ROOT / "expfam" / "data" / "cora"
    content_path = data_dir / "cora.content"
    cites_path = data_dir / "cora.cites"

    node_ids, features, labels = [], [], []
    label_set = []
    with open(content_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            node_ids.append(parts[0])
            features.append([int(x) for x in parts[1:-1]])
            lbl = parts[-1]
            if lbl not in label_set:
                label_set.append(lbl)
            labels.append(lbl)

    node_ids = np.array(node_ids)
    X_full = np.array(features, dtype=np.float32)
    label_idx = np.array([label_set.index(l) for l in labels], dtype=int)

    edges = set()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    with open(cites_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            if u in id_to_idx and v in id_to_idx:
                i, j = id_to_idx[u], id_to_idx[v]
                if i != j:
                    edges.add((min(i, j), max(i, j)))

    degree = np.zeros(len(node_ids), dtype=int)
    for (i, j) in edges:
        degree[i] += 1
        degree[j] += 1

    return X_full, label_idx, label_set, edges, degree


def build_balanced_degree(label_idx, n_classes, per_class, degree):
    selected = []
    for c in range(n_classes):
        idx_c = np.where(label_idx == c)[0]
        k_c = min(per_class, len(idx_c))
        order = np.argsort(-degree[idx_c])
        chosen = idx_c[order[:k_c]]
        selected.extend(chosen.tolist())
    return np.array(selected)


def load_cora_data():
    X_full, label_idx, label_set, edges, degree = parse_cora()
    n_classes = len(label_set)
    subset_idx = build_balanced_degree(label_idx, n_classes, CORA_PER_CLASS, degree)
    subset_set = {v: i for i, v in enumerate(subset_idx)}
    n_sub = len(subset_idx)

    X_sub_full = X_full[subset_idx, :]
    col_sums = X_sub_full.sum(axis=0)
    top_cols = np.argsort(-col_sums)[:CORA_D_SUBSET]
    X_sub = X_sub_full[:, top_cols]

    Y_sub = np.zeros((n_sub, n_sub), dtype=float)
    for (u, v) in edges:
        if u in subset_set and v in subset_set:
            i, j = subset_set[u], subset_set[v]
            Y_sub[i, j] = 1.0
            Y_sub[j, i] = 1.0
    np.fill_diagonal(Y_sub, 0.0)

    labels_sub = label_idx[subset_idx]
    return dict(X=X_sub, Y=Y_sub, labels=labels_sub, n=n_sub, d=CORA_D_SUBSET,
                family_x="bernoulli", family_y="bernoulli",
                feature_names=[f"word_index_{int(c)}" for c in top_cols],
                n_classes=n_classes)


def load_movielens_data():
    data_dir = _ROOT / "expfam" / "data" / "movielens_pilot"
    zip_path = _ROOT / "expfam" / "data" / "ml-100k.zip"

    X = np.load(data_dir / "movielens_X_genre.npy").astype(np.float64)
    movie_ids = np.load(data_dir / "movielens_movie_ids.npy")
    genre_labels = np.load(data_dir / "movielens_primary_genre_labels.npy")

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("ml-100k/u.data") as f:
            raw = f.read().decode("latin-1")
    rows = []
    for line in raw.strip().split("\n"):
        p = line.strip().split("\t")
        if len(p) >= 3:
            rows.append((int(p[0]), int(p[1]), int(p[2])))
    ratings_df = pd.DataFrame(rows, columns=["uid", "mid", "rating"])

    sub = ratings_df[ratings_df["mid"].isin(set(movie_ids.tolist()))]
    like_users = defaultdict(set)
    for mid, g in sub.groupby("mid"):
        like_users[mid] = set(g.loc[g["rating"] >= ML_LIKE_THRESHOLD, "uid"].tolist())

    n = len(movie_ids)
    Y = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        ui = like_users.get(movie_ids[i], set())
        for j in range(i + 1, n):
            uj = like_users.get(movie_ids[j], set())
            c = len(ui & uj)
            Y[i, j] = c
            Y[j, i] = c

    n_classes = len(np.unique(genre_labels))
    return dict(X=X, Y=Y, labels=genre_labels, n=n, d=X.shape[1],
                family_x="bernoulli", family_y="poisson",
                feature_names=GENRES, n_classes=n_classes)


# ════════════════════════════════════════════════════════════════════
# Generic MCEM runner (reuses the pattern from existing fixed-pilot scripts)
# ════════════════════════════════════════════════════════════════════

def run_em_common(X, Y, family_x, family_y, k, L=5, num_iter=8, seed=42, max_retries=2):
    n, d = X.shape
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    nan_count = 0

    for retry in range(max_retries + 1):
        newton_alpha = 0.5 / (2 ** retry)
        rng = np.random.default_rng(seed + retry * 1000)
        model = DualExpFamLSMFixed(n=n, d=d, k=k, L=L, family_x=family_x, family_y=family_y)
        model.initialize_params(true_params=None, seed=seed + retry * 1000)

        if family_y == "bernoulli":
            density = float(np.clip(Y[upper_mask].mean(), 1e-6, 1 - 1e-6))
            model.params["w0"] = np.log(density / (1 - density))
            model.params["w"] = 0.5
        elif family_y == "poisson":
            upper = Y[upper_mask]
            mean_cnt = float(upper[upper > 0].mean()) if np.any(upper > 0) else 1.0
            model.params["w0"] = np.log(mean_cnt + 1e-10)
            model.params["w"] = 0.1 / (2 ** retry)
        else:  # gaussian
            upper_vals = Y[upper_mask]
            model.params["w0"] = float(upper_vals.mean())
            model.params["w"] = 0.5
            model.sigma_y = float(max(upper_vals.std(), 0.01))

        if family_x in ("bernoulli", "poisson"):
            model.params["F"] *= 0.2

        Z = model.params["Z"].copy()
        F = model.params["F"].copy()
        sigma = model.params["sigma"].copy()
        w0 = model.params["w0"]
        w = model.params["w"]
        var_z = model.params["var_z"]
        Z_prev, nan_count = Z.copy(), 0

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
            Z_prev = Z.copy()
            Z = Z_samples[:, :, -1].copy()

            F = model.calc_F(X, Z_samples)
            sigma = model.calc_sigma(X, Z_samples, F)
            w0 = model.calc_w0(Y, Z_samples, w0, w, max_iter=50)
            w = model.calc_w(Y, Z_samples, w0, w, max_iter=50)
            if family_y == "gaussian":
                model.calc_sigma_y(Y, Z_samples, w0, w)

        if nan_count == 0:
            break

    model.params.update({"F": F, "sigma": sigma, "w0": w0, "w": w})
    Q_strict = calc_Q_dual_strict(X, Y, Z_samples, F, sigma, var_z, w0, w, model)
    bic, num_params = calc_bic_dual(Q_strict, k, n, d, family_x, family_y)

    Z_est = Z_samples[:, :, -1]
    return dict(model=model, Z=Z_est, F=F, sigma=sigma, w0=float(w0), w=float(w),
                Q_strict=Q_strict, bic=bic, num_params=num_params,
                nan_occurred=nan_count > 0, nan_count=nan_count, upper_mask=upper_mask)


# ════════════════════════════════════════════════════════════════════
# Reconstruction metrics
# ════════════════════════════════════════════════════════════════════

def eval_X_reconstruction(model, X, Z, F, family_x):
    eta_x = Z @ F.T
    mu_x = model._mean_function_x(eta_x)
    x_flat = X.flatten()
    mu_flat = mu_x.flatten()

    out = dict(X_primary_metric_name="NA", X_primary_metric_value=np.nan,
               RMSE_X=np.nan, MAE_X=np.nan, BCE_X=np.nan,
               AUC_X=np.nan, AP_X=np.nan, accuracy_X=np.nan, Pearson_X=np.nan)

    rmse = float(np.sqrt(np.mean((x_flat - mu_flat) ** 2)))
    mae = float(np.mean(np.abs(x_flat - mu_flat)))
    out["RMSE_X"] = rmse
    out["MAE_X"] = mae

    if family_x == "gaussian":
        try:
            pear = float(pearsonr(x_flat, mu_flat)[0])
        except Exception:
            pear = float("nan")
        out["Pearson_X"] = pear
        out["X_primary_metric_name"] = "RMSE_X"
        out["X_primary_metric_value"] = rmse
    elif family_x == "bernoulli":
        p = np.clip(mu_flat, 1e-10, 1.0 - 1e-10)
        bce = float(-np.mean(x_flat * np.log(p) + (1 - x_flat) * np.log(1 - p)))
        out["BCE_X"] = bce
        out["accuracy_X"] = float(np.mean((p >= 0.5).astype(float) == x_flat))
        try:
            if len(np.unique(x_flat)) > 1:
                out["AUC_X"] = float(roc_auc_score(x_flat, mu_flat))
                out["AP_X"] = float(average_precision_score(x_flat, mu_flat))
        except Exception:
            pass
        out["X_primary_metric_name"] = "BCE_X"
        out["X_primary_metric_value"] = bce
    elif family_x == "poisson":
        try:
            pear = float(pearsonr(x_flat, mu_flat)[0])
        except Exception:
            pear = float("nan")
        out["Pearson_X"] = pear
        out["X_primary_metric_name"] = "RMSE_X"
        out["X_primary_metric_value"] = rmse

    return out


def eval_Y_reconstruction(model, Y, Z, w0, w, family_y, upper_mask, high_colike_percentile=90):
    eta_y = w0 + w * (Z @ Z.T)
    mu_y = model._mean_function(eta_y)
    y_true = Y[upper_mask]
    y_pred = mu_y[upper_mask]
    eta_true = eta_y[upper_mask]

    out = dict(Y_primary_metric_name="NA", Y_primary_metric_value=np.nan,
               RMSE_Y=np.nan, MAE_Y=np.nan, BCE_Y=np.nan, AUC_Y=np.nan, AP_Y=np.nan,
               random_AP_Y=np.nan, density_Y=np.nan, Pearson_Y=np.nan, Spearman_Y=np.nan,
               Poisson_NLL=np.nan, Poisson_Deviance=np.nan, high_colike_AP=np.nan)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    out["RMSE_Y"] = rmse
    out["MAE_Y"] = mae

    if family_y == "bernoulli":
        density = float(y_true.mean())
        out["density_Y"] = density
        out["random_AP_Y"] = density
        p = np.clip(y_pred, 1e-10, 1.0 - 1e-10)
        bce = float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
        out["BCE_Y"] = bce
        try:
            if len(np.unique(y_true)) > 1:
                out["AUC_Y"] = float(roc_auc_score(y_true, y_pred))
                out["AP_Y"] = float(average_precision_score(y_true, y_pred))
        except Exception:
            pass
        out["Y_primary_metric_name"] = "BCE_Y"
        out["Y_primary_metric_value"] = bce

    elif family_y == "poisson":
        try:
            pear = float(pearsonr(y_true, y_pred)[0])
        except Exception:
            pear = float("nan")
        try:
            spear = float(spearmanr(y_true, y_pred)[0])
        except Exception:
            spear = float("nan")
        out["Pearson_Y"] = pear
        out["Spearman_Y"] = spear

        eta_c = np.clip(eta_true, -20, 10)
        lam = np.exp(eta_c)
        nll = float(np.mean(lam - y_true * eta_c + gammaln(y_true + 1)))
        out["Poisson_NLL"] = nll

        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.where(y_true > 0, y_true * np.log(np.maximum(y_true, 1e-10) / np.maximum(lam, 1e-10)), 0.0)
        deviance = float(np.mean(2.0 * (log_ratio - (y_true - lam))))
        out["Poisson_Deviance"] = deviance

        thr = float(np.percentile(y_true, high_colike_percentile))
        high_label = (y_true >= thr).astype(int)
        try:
            if len(np.unique(high_label)) > 1:
                out["high_colike_AP"] = float(average_precision_score(high_label, y_pred))
        except Exception:
            pass

        out["Y_primary_metric_name"] = "Poisson_NLL"
        out["Y_primary_metric_value"] = nll

    return out


def compute_label_metrics(Z, labels, n_classes):
    k = Z.shape[1]
    if k == 1:
        Z_2d = np.hstack([Z, np.zeros((len(Z), 1))])
    elif k == 2:
        Z_2d = Z.copy()
    else:
        Z_2d = PCA(n_components=2).fit_transform(Z)
    try:
        sil = float(silhouette_score(Z_2d, labels))
    except Exception:
        sil = float("nan")
    try:
        km = KMeans(n_clusters=n_classes, random_state=42, n_init=10).fit_predict(Z_2d)
        nmi = float(normalized_mutual_info_score(labels, km))
        ari = float(adjusted_rand_score(labels, km))
    except Exception:
        nmi = ari = float("nan")
    return sil, nmi, ari


def w_summary_str(w):
    # w is scalar in this model (eta_Y = w0 + w * Z@Z.T)
    return f"scalar={float(w):.6f}"


# ════════════════════════════════════════════════════════════════════
# F heatmap + top-feature CSV helpers
# ════════════════════════════════════════════════════════════════════

def save_F_heatmap(F, feature_names, title, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d, k = F.shape
    fig_h = max(4.0, 0.22 * d)
    fig, ax = plt.subplots(figsize=(1.0 * k + 2.5, fig_h))
    vmax = np.max(np.abs(F)) if F.size else 1.0
    im = ax.imshow(F, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"f{j}" for j in range(k)])
    ax.set_yticks(range(d))
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7, label="F value")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def top_features_per_factor(F, feature_names, dataset, k, top_n=5):
    d, kk = F.shape
    rows = []
    for j in range(kk):
        col = F[:, j]
        order = np.argsort(-np.abs(col))[:top_n]
        for rank, idx in enumerate(order, start=1):
            rows.append(dict(
                dataset=dataset, k=k, factor=j, rank=rank,
                feature_name=feature_names[idx], F_value=float(col[idx]),
            ))
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════
# Per-dataset driver
# ════════════════════════════════════════════════════════════════════

def run_dataset(dataset_name, data, k_list, main_k_list, seed_base, top_feature_records):
    X, Y, labels = data["X"], data["Y"], data["labels"]
    family_x, family_y = data["family_x"], data["family_y"]
    n, d = data["n"], data["d"]
    feature_names = data["feature_names"]
    n_classes = data["n_classes"]

    rows = []
    best_by_k = {}  # k -> (bic, fit_result) for the lowest-BIC trial, used for F heatmaps

    for k in k_list:
        for trial in range(N_TRIALS):
            seed = seed_base + k * 100 + trial * 10
            t0 = time.perf_counter()
            try:
                fit = run_em_common(X, Y, family_x, family_y, k, L=L, num_iter=NITER, seed=seed)
                success = True
                err_msg = ""
            except Exception as e:
                traceback.print_exc()
                fit = None
                success = False
                err_msg = str(e)
            runtime = time.perf_counter() - t0

            if not success:
                rows.append(dict(dataset=dataset_name, k=k, trial=trial, success=False,
                                  family_x=family_x, family_y=family_y, n=n, d=d,
                                  runtime=runtime, nan_occurred=True, error_message=err_msg))
                print(f"  [{dataset_name}] k={k} t={trial} FAILED: {err_msg}")
                continue

            model = fit["model"]
            Z, F, w0, w = fit["Z"], fit["F"], fit["w0"], fit["w"]
            upper_mask = fit["upper_mask"]

            x_metrics = eval_X_reconstruction(model, X, Z, F, family_x)
            y_metrics = eval_Y_reconstruction(model, Y, Z, w0, w, family_y, upper_mask)
            sil, nmi, ari = compute_label_metrics(Z, labels, n_classes)

            row = dict(
                dataset=dataset_name, k=k, trial=trial, success=True,
                family_x=family_x, family_y=family_y, n=n, d=d,
                BIC=fit["bic"], Q_strict=fit["Q_strict"],
                **x_metrics, **y_metrics,
                silhouette=sil, NMI=nmi, ARI=ari,
                w0=w0, w_summary=w_summary_str(w),
                runtime=runtime, nan_occurred=fit["nan_occurred"], error_message="",
                pair_scope="upper_triangle (i<j), symmetric Y, diagonal excluded",
            )
            rows.append(row)
            print(f"  [{dataset_name}] k={k} t={trial} BIC={fit['bic']:.1f} "
                  f"{x_metrics['X_primary_metric_name']}={x_metrics['X_primary_metric_value']:.4f} "
                  f"{y_metrics['Y_primary_metric_name']}={y_metrics['Y_primary_metric_value']:.4f} "
                  f"[{runtime:.1f}s]")

            if k in main_k_list:
                if k not in best_by_k or fit["bic"] < best_by_k[k][0]:
                    best_by_k[k] = (fit["bic"], F.copy(), trial)

    # F heatmaps + top-feature CSV for main K's (lowest-BIC trial among the trials run)
    for k, (bic_val, F_best, trial_idx) in best_by_k.items():
        save_F_heatmap(
            F_best, feature_names,
            title=f"{dataset_name} F heatmap (k={k}, best trial={trial_idx}, BIC={bic_val:.1f})",
            stem=f"{dataset_name}_F_heatmap_k{k}",
        )
        top_feature_records.append(top_features_per_factor(F_best, feature_names, dataset_name, k))

    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    top_feature_records = []
    all_rows = []

    print("=== Loading data ===")
    wine_data = load_wine_data()
    print(f"Wine: n={wine_data['n']}, d={wine_data['d']}")
    cora_data = load_cora_data()
    print(f"Cora: n={cora_data['n']}, d={cora_data['d']}")
    ml_data = load_movielens_data()
    print(f"MovieLens: n={ml_data['n']}, d={ml_data['d']}")

    print("\n=== Wine ===")
    wine_df = run_dataset("wine", wine_data, WINE_K_LIST, WINE_MAIN_K, seed_base=31000,
                           top_feature_records=top_feature_records)
    all_rows.append(wine_df)

    print("\n=== Cora ===")
    cora_df = run_dataset("cora", cora_data, CORA_K_LIST, CORA_MAIN_K, seed_base=32000,
                           top_feature_records=top_feature_records)
    all_rows.append(cora_df)

    print("\n=== MovieLens ===")
    ml_df = run_dataset("movielens", ml_data, ML_K_LIST, ML_MAIN_K, seed_base=33000,
                         top_feature_records=top_feature_records)
    all_rows.append(ml_df)

    summary_df = pd.concat(all_rows, ignore_index=True)
    summary_path = OUT_DIR / "common_reconstruction_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path} ({len(summary_df)} rows)")

    top_features_df = pd.concat(top_feature_records, ignore_index=True) if top_feature_records else pd.DataFrame()
    for ds, fname in [("wine", "wine_factor_top_features.csv"),
                       ("cora", "cora_factor_top_words.csv"),
                       ("movielens", "movielens_factor_top_genres.csv")]:
        sub = top_features_df[top_features_df["dataset"] == ds] if len(top_features_df) else top_features_df
        sub.to_csv(OUT_DIR / fname, index=False)
        print(f"Saved: {OUT_DIR / fname} ({len(sub)} rows)")

    elapsed = (time.perf_counter() - t_start) / 60
    print(f"\nTotal runtime: {elapsed:.1f} min")
    print(f"Success rate overall: {summary_df['success'].mean():.2%}")


if __name__ == "__main__":
    main()
