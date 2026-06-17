"""
MovieLens Poisson Y pilot script.

Movie-node projection: Y = co-rating count (Poisson Y), X = genre multi-hot (Bernoulli X).
novel combination: X=Bernoulli, Y=Poisson — verifies DualExpFamLSMFixed on real count data.

Data  : expfam/data/movielens_pilot/
n=100, d=19, k=[2,3,5], 3 trials = 9 fits total

High-count binary eval: Y_count >= 80 treated as positive (density ~0.069)

NOTE on genre labels: primary_genre is assigned as the FIRST genre flag per movie.
MovieLens 100K movies have on average 1.72 genre tags, so primary genre is
a convenience label, not an exclusive category. NMI/ARI should be interpreted
with this multi-label caveat in mind.

Output:
  expfam/results/real_data/movielens_poisson_pilot/
  expfam/figures/real_data/movielens_poisson_pilot/
"""

import sys
import time
import traceback
import warnings
from pathlib import Path

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

DATA_DIR = _ROOT / "expfam" / "data" / "movielens_pilot"
OUT_DIR  = _ROOT / "expfam" / "results" / "real_data" / "movielens_poisson_pilot"
FIG_DIR  = _ROOT / "expfam" / "figures" / "real_data" / "movielens_poisson_pilot"

FAMILY_X = "bernoulli"
FAMILY_Y = "poisson"
K_LIST   = [2, 3, 5]
N_TRIALS = 3
SEED_BASE = 9000   # seed = SEED_BASE + k*100 + trial*10
L, NITER  = 5, 8

HIGH_COUNT_THRESHOLD = 80   # Y_count >= 80 → positive for binary eval

# Standard MovieLens 100K genre names (from u.genre, index = position in file)
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
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
# EM runner (Poisson Y adapted)
# ─────────────────────────────────────────────────────────────────────

def run_em_fixed(X, Y, family_x, family_y, k, L=5, num_iter=8, seed=42):
    n, d = X.shape
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_count = float(Y[upper_mask].mean())

    nan_count_total = 0
    error_msg = ""

    for retry in range(3):
        newton_alpha = 0.5 / (2 ** retry)
        # Halve w each retry for Poisson stability
        w_init = 0.1 / (2 ** retry)

        rng   = np.random.default_rng(seed + retry * 1000)
        model = DualExpFamLSMFixed(n=n, d=d, k=k, L=L,
                                   family_x=family_x, family_y=family_y)
        model.initialize_params(true_params=None, seed=seed + retry * 1000)

        # Informed init for Poisson Y: w0 = log(mean_count)
        if family_y == "poisson":
            model.params["w0"] = float(np.log(max(mean_count, 1e-10)))
            model.params["w"]  = w_init
        # Bernoulli X: shrink F to avoid early saturation
        if family_x in ("bernoulli", "poisson"):
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
                    model.params.update(dict(Z=Z.copy(), F=F, sigma=sigma,
                                            w0=w0, w=w))
                    Z_new = model.calc_eta_newton(
                        X, Y, rng=rng, max_iter=10, alpha=newton_alpha)
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
        bic, npar = calc_bic_dual(Q_strict, k, n, d, family_x, family_y)
    except Exception:
        Q_strict = float("nan")
        bic      = float("nan")
        npar     = 0

    Z_est = Z_samples[:, :, -1]

    # Y predictions (clip to avoid extreme values)
    eta_y   = float(w0) + float(w) * (Z_est @ Z_est.T)
    mu_y    = np.clip(model._mean_function(eta_y), 0.0, 1e5)
    y_upper = Y[upper_mask].astype(float)
    y_hat   = mu_y[upper_mask]

    # Count regression metrics
    residuals  = y_upper - y_hat
    rmse_y     = float(np.sqrt(np.mean(residuals ** 2)))
    mae_y      = float(np.mean(np.abs(residuals)))

    try:
        pcorr, _ = pearsonr(y_upper, y_hat)
        pearson_corr = float(pcorr)
    except Exception:
        pearson_corr = float("nan")

    try:
        scorr, _ = spearmanr(y_upper, y_hat)
        spearman_corr = float(scorr)
    except Exception:
        spearman_corr = float("nan")

    try:
        safe_hat = np.maximum(y_hat, 0.0)
        rmse_log1p = float(np.sqrt(np.mean(
            (np.log1p(y_upper) - np.log1p(safe_hat)) ** 2
        )))
    except Exception:
        rmse_log1p = float("nan")

    # High-count binary eval (Y_count >= threshold → positive)
    y_binary    = (y_upper >= HIGH_COUNT_THRESHOLD).astype(int)
    hc_density  = float(y_binary.mean())
    random_ap_b = hc_density  # random classifier AP = positive rate

    try:
        hc_auc = float(roc_auc_score(y_binary, y_hat))
    except Exception:
        hc_auc = float("nan")
    try:
        hc_ap  = float(average_precision_score(y_binary, y_hat))
    except Exception:
        hc_ap  = float("nan")

    # X reconstruction
    mu_x   = model._mean_function_x(Z_est @ F.T)
    rmse_x = float(np.sqrt(np.mean((X - mu_x) ** 2)))
    try:
        x_acc = float(np.mean((mu_x > 0.5).astype(int) == X.astype(int)))
    except Exception:
        x_acc = float("nan")

    return {
        "Q_strict": Q_strict, "bic": bic, "num_params": npar,
        "rmse_y": rmse_y, "mae_y": mae_y,
        "pearson_corr": pearson_corr, "spearman_corr": spearman_corr,
        "rmse_log1p_y": rmse_log1p,
        "hc_density": hc_density, "random_ap_baseline": random_ap_b,
        "hc_auc": hc_auc, "hc_ap": hc_ap,
        "rmse_x": rmse_x, "x_binary_acc": x_acc,
        "mean_y": float(y_upper.mean()), "mean_y_hat": float(y_hat.mean()),
        "max_y": float(y_upper.max()), "max_y_hat": float(y_hat.max()),
        "w0": float(w0), "w": float(w),
        "Z_est": Z_est, "y_upper": y_upper, "y_hat": y_hat,
        "nan_occurred": nan_count_total > 0, "nan_count": nan_count_total,
        "error_message": error_msg,
    }


# ─────────────────────────────────────────────────────────────────────
# Z metrics
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
        km  = KMeans(n_clusters=n_cl, random_state=42, n_init=10).fit_predict(Z_2d)
        nmi = float(normalized_mutual_info_score(labels, km))
        ari = float(adjusted_rand_score(labels, km))
    except Exception:
        nmi = ari = float("nan")
    return sil, nmi, ari, Z_2d


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


def make_k_metrics_figure(agg_df):
    ks = agg_df["k"].values

    metrics = [
        ("bic_mean",          "bic_std",           "BIC",               "#607D8B"),
        ("rmse_y_mean",       "rmse_y_std",         "RMSE Y_count",      "#F44336"),
        ("mae_y_mean",        "mae_y_std",          "MAE Y_count",       "#FF9800"),
        ("pearson_corr_mean", "pearson_corr_std",   "Pearson(Y, Y_hat)", "#2196F3"),
        ("hc_ap_mean",        "hc_ap_std",          "High-count AP",     "#4CAF50"),
        ("nmi_mean",          "nmi_std",            "NMI (genre)",       "#9C27B0"),
    ]
    n_m   = len(metrics)
    fig, axes = plt.subplots(1, n_m, figsize=(4.0 * n_m, 4.0))

    for ax, (mu_col, sd_col, label, color) in zip(axes, metrics):
        mu = agg_df[mu_col].values
        sd = agg_df[sd_col].values if sd_col in agg_df.columns else np.zeros_like(mu)
        ax.errorbar(ks, mu, yerr=sd, fmt="o-", color=color,
                    linewidth=2, markersize=7, capsize=4)
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=9)
        ax.set_xticks(ks.tolist())
        ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle(
        f"MovieLens Poisson Y pilot — Metrics vs k (mean +/- std)\n"
        f"(n=100, d=19, X=Bern, Y=Pois, {N_TRIALS} trials/k)",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, "movielens_poisson_k_metrics")


def make_y_scatter_figure(y_upper, y_hat, k_val, trial_idx, pearson_corr):
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(y_upper, y_hat, alpha=0.35, s=10, color="#2196F3", edgecolors="none")

    lo = min(y_upper.min(), y_hat.min())
    hi = max(y_upper.max(), y_hat.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="y=x (perfect)")
    ax.axhline(y_hat.mean(), color="gray", linestyle=":", linewidth=1.0,
               label=f"mean Y_hat={y_hat.mean():.1f}")
    ax.axvline(y_upper.mean(), color="gray", linestyle=":", linewidth=1.0,
               label=f"mean Y={y_upper.mean():.1f}")

    ax.set_xlabel("Y_count (true)")
    ax.set_ylabel("Y_hat (predicted)")
    ax.set_title(
        f"Y_count true vs predicted (k={k_val}, trial={trial_idx})\n"
        f"Pearson r={pearson_corr:.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "movielens_poisson_y_true_vs_pred")


def make_z_genre_figure(Z_est, genre_labels, genre_names, k_val):
    k_z  = Z_est.shape[1]
    pca_used = k_z > 2
    if k_z == 1:
        Z_2d = np.hstack([Z_est, np.zeros((len(Z_est), 1))])
    elif k_z == 2:
        Z_2d = Z_est.copy()
    else:
        Z_2d = PCA(n_components=2).fit_transform(Z_est)

    unique_labels = sorted(np.unique(genre_labels).tolist())
    n_ul = len(unique_labels)
    cmap = plt.cm.get_cmap("tab20", n_ul)

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    for ci, lbl in enumerate(unique_labels):
        mask = genre_labels == lbl
        gname = genre_names[lbl] if lbl < len(genre_names) else str(lbl)
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                   color=cmap(ci), label=gname,
                   s=35, alpha=0.80, edgecolors="none")

    ax.set_xlabel("PC1" if pca_used else "Z1", fontsize=9)
    ax.set_ylabel("PC2" if pca_used else "Z2", fontsize=9)
    ax.set_title(
        f"MovieLens — Latent Z by primary genre (k={k_val})\n"
        "NOTE: primary genre is approximate (movies have avg 1.72 genre tags)",
        fontsize=9,
    )
    ax.legend(fontsize=6, loc="best", framealpha=0.7, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "movielens_poisson_z_by_genre")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────
    X           = np.load(DATA_DIR / "movielens_X_genre.npy").astype(np.float64)
    Y_count     = np.load(DATA_DIR / "movielens_Y_count.npy").astype(np.float64)
    genre_labels = np.load(DATA_DIR / "movielens_primary_genre_labels.npy")
    movie_ids   = np.load(DATA_DIR / "movielens_movie_ids.npy")

    n, d = X.shape
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    Y_upper    = Y_count[upper_mask]

    # High-count binary mask (computed in-script, not saved)
    Y_binary_t80 = (Y_count >= HIGH_COUNT_THRESHOLD).astype(np.float64)
    hc_density   = float(Y_binary_t80[upper_mask].mean())

    # Y_count stats
    y_mean    = float(Y_upper.mean())
    y_var     = float(Y_upper.var())
    y_max     = float(Y_upper.max())
    var_mean  = y_var / max(y_mean, 1e-10)
    density_pos = float((Y_upper > 0).mean())

    print("=== MovieLens data summary ===")
    print(f"  n={n}, d={d}, family_x={FAMILY_X}, family_y={FAMILY_Y}")
    print(f"  Y_count: mean={y_mean:.2f}, var/mean={var_mean:.2f}, max={y_max:.0f}")
    print(f"  Y_count density_pos={density_pos:.4f} (all pairs have count>0)")
    print(f"  high_count (>={HIGH_COUNT_THRESHOLD}): density={hc_density:.4f}")
    print(f"  random_AP_baseline = {hc_density:.4f}")
    print(f"  unique primary genres: {len(np.unique(genre_labels))}")

    # ── Data summary CSV ───────────────────────────────────────────
    ds_path = OUT_DIR / "movielens_poisson_data_summary.csv"
    if not ds_path.exists():
        ds_row = {
            "n": n, "d": d, "family_x": FAMILY_X, "family_y": FAMILY_Y,
            "subset_strategy": "genre_stratified_mp100",
            "y_count_mean": y_mean, "y_count_var_over_mean": var_mean,
            "y_count_max": y_max, "y_count_density_pos": density_pos,
            "high_count_threshold": HIGH_COUNT_THRESHOLD,
            "high_count_density": hc_density,
            "random_ap_baseline": hc_density,
            "n_unique_primary_genres": int(len(np.unique(genre_labels))),
        }
        pd.DataFrame([ds_row]).to_csv(ds_path, index=False)
        print(f"Saved: {ds_path}")

    # ── k-sweep ────────────────────────────────────────────────────
    total_fits = len(K_LIST) * N_TRIALS
    print(f"\n=== Poisson Y pilot: k={K_LIST}, {N_TRIALS} trials = {total_fits} fits ===")

    t0           = time.perf_counter()
    summary_rows = []
    z_store      = {}    # {(k, trial): res}
    fit_count    = 0

    for k_val in K_LIST:
        for trial in range(N_TRIALS):
            fit_count += 1
            seed = SEED_BASE + k_val * 100 + trial * 10
            print(f"\n[{fit_count:2d}/{total_fits}] k={k_val}  trial={trial}  seed={seed}")

            t_k = time.perf_counter()
            try:
                res     = run_em_fixed(X, Y_count, FAMILY_X, FAMILY_Y, k=k_val,
                                       L=L, num_iter=NITER, seed=seed)
                elapsed = time.perf_counter() - t_k
                sil, nmi, ari, _ = compute_z_metrics(res["Z_est"], genre_labels)
                success  = True
                err_msg  = res.get("error_message", "")
            except Exception as e:
                elapsed = time.perf_counter() - t_k
                traceback.print_exc()
                res     = {"Q_strict": float("nan"), "bic": float("nan"),
                           "num_params": 0, "rmse_y": float("nan"),
                           "mae_y": float("nan"), "pearson_corr": float("nan"),
                           "spearman_corr": float("nan"), "rmse_log1p_y": float("nan"),
                           "hc_auc": float("nan"), "hc_ap": float("nan"),
                           "hc_density": hc_density, "random_ap_baseline": hc_density,
                           "rmse_x": float("nan"), "x_binary_acc": float("nan"),
                           "mean_y": y_mean, "mean_y_hat": float("nan"),
                           "max_y": y_max, "max_y_hat": float("nan"),
                           "w0": float("nan"), "w": float("nan"),
                           "Z_est": None, "y_upper": Y_upper,
                           "y_hat": np.full_like(Y_upper, float("nan")),
                           "nan_occurred": True, "nan_count": 999,
                           "error_message": str(e)}
                sil = nmi = ari = float("nan")
                success  = False
                err_msg  = str(e)

            z_store[(k_val, trial)] = res

            print(f"  BIC={res['bic']:.1f}  Q={res['Q_strict']:.1f}"
                  f"  RMSE_Y={res['rmse_y']:.2f}"
                  f"  Pearson={res['pearson_corr']:.3f}"
                  f"  hc_AP={res['hc_ap']:.4f}"
                  f"  NMI={nmi:.4f}"
                  f"  [{elapsed:.1f}s]"
                  f"  w0={res['w0']:.3f}  w={res['w']:.3f}"
                  f"  max_Y_hat={res['max_y_hat']:.1f}")

            summary_rows.append({
                "k": k_val, "trial": trial, "seed": seed,
                "bic": res["bic"], "q_strict": res["Q_strict"],
                "num_params": res["num_params"],
                "rmse_y": res["rmse_y"], "mae_y": res["mae_y"],
                "pearson_corr": res["pearson_corr"],
                "spearman_corr": res["spearman_corr"],
                "rmse_log1p_y": res["rmse_log1p_y"],
                "hc_density": res["hc_density"],
                "random_ap_baseline": res["random_ap_baseline"],
                "hc_auc": res["hc_auc"], "hc_ap": res["hc_ap"],
                "rmse_x": res["rmse_x"], "x_binary_acc": res["x_binary_acc"],
                "mean_y": res["mean_y"], "mean_y_hat": res["mean_y_hat"],
                "max_y": res["max_y"], "max_y_hat": res["max_y_hat"],
                "silhouette": sil, "nmi": nmi, "ari": ari,
                "w0": res["w0"], "w": res["w"],
                "nan_occurred": res["nan_occurred"],
                "nan_count": res["nan_count"],
                "success": success,
                "runtime_s": round(elapsed, 1),
                "error_message": err_msg,
            })

    elapsed_total = time.perf_counter() - t0
    print(f"\nDone in {elapsed_total/60:.1f} min  ({fit_count}/{total_fits} fits)")

    # ── Save summary CSV ───────────────────────────────────────────
    sum_df = pd.DataFrame(summary_rows)
    sum_path = OUT_DIR / "movielens_poisson_summary.csv"
    if not sum_path.exists():
        sum_df.to_csv(sum_path, index=False)
        print(f"Saved: {sum_path}")

    # ── Aggregation ────────────────────────────────────────────────
    num_cols = [
        "bic", "q_strict", "rmse_y", "mae_y", "pearson_corr", "spearman_corr",
        "rmse_log1p_y", "hc_auc", "hc_ap", "rmse_x", "x_binary_acc",
        "silhouette", "nmi", "ari", "runtime_s",
    ]
    agg_rows = []
    for k_val in K_LIST:
        sub = sum_df[sum_df["k"] == k_val]
        n_ok = int(sub["success"].sum())
        row  = {"k": k_val, "n_trials": len(sub), "n_success": n_ok}
        for col in num_cols:
            vals = sub[col].dropna().astype(float)
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"]  = float(vals.std())  if len(vals) else float("nan")
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    agg_path = OUT_DIR / "movielens_poisson_agg.csv"
    if not agg_path.exists():
        agg_df.to_csv(agg_path, index=False)
        print(f"Saved: {agg_path}")

    # ── Best k ────────────────────────────────────────────────────
    def best_k(metric_col, minimize=True):
        vals = agg_df[metric_col].values
        mask = ~np.isnan(vals)
        if not mask.any():
            return float("nan")
        idx = np.argmin(vals[mask]) if minimize else np.argmax(vals[mask])
        return int(agg_df["k"].values[np.where(mask)[0][idx]])

    bestk_row = {
        "best_k_by_BIC":        best_k("bic_mean",          minimize=True),
        "best_k_by_RMSE_Y":     best_k("rmse_y_mean",       minimize=True),
        "best_k_by_Pearson":    best_k("pearson_corr_mean",  minimize=False),
        "best_k_by_hc_AP":      best_k("hc_ap_mean",        minimize=False),
        "best_k_by_NMI":        best_k("nmi_mean",          minimize=False),
    }
    bestk_path = OUT_DIR / "movielens_poisson_bestk.csv"
    if not bestk_path.exists():
        pd.DataFrame([bestk_row]).to_csv(bestk_path, index=False)
        print(f"Saved: {bestk_path}")

    # ── Console aggregation ─────────────────────────────────────────
    print("\n=== Aggregated results (mean over trials) ===")
    for _, row in agg_df.iterrows():
        print(f"  k={int(row.k)}: BIC={row.bic_mean:.1f}+/-{row.bic_std:.1f}"
              f"  RMSE_Y={row.rmse_y_mean:.2f}"
              f"  Pearson={row.pearson_corr_mean:.3f}"
              f"  hc_AP={row.hc_ap_mean:.4f}"
              f"  NMI={row.nmi_mean:.4f}"
              f"  ARI={row.ari_mean:.4f}"
              f"  w0={sum_df[sum_df.k==row.k].w0.mean():.3f}"
              f"  [{row.runtime_s_mean:.0f}s]")

    print("\n=== Best k ===")
    for name, val in bestk_row.items():
        print(f"  {name}={val}")

    # ── Figures ─────────────────────────────────────────────────────
    print("\n=== Figures ===")

    # 1. k metrics
    make_k_metrics_figure(agg_df)

    # 2. Y true vs pred — best trial by Pearson for best_k_by_Pearson
    bk_pearson = int(bestk_row["best_k_by_Pearson"]) if not np.isnan(bestk_row["best_k_by_Pearson"]) else K_LIST[0]
    sub_bk = sum_df[(sum_df["k"] == bk_pearson) & sum_df["success"]]
    if len(sub_bk) > 0:
        best_trial_idx = int(sub_bk["pearson_corr"].idxmax())
        best_trial_row = sub_bk.loc[best_trial_idx]
        best_trial     = int(best_trial_row["trial"])
        best_pearson   = float(best_trial_row["pearson_corr"])
        best_res       = z_store.get((bk_pearson, best_trial))
        if best_res is not None and best_res.get("Z_est") is not None:
            make_y_scatter_figure(
                best_res["y_upper"], best_res["y_hat"],
                bk_pearson, best_trial, best_pearson,
            )

    # 3. Z by genre — same best k/trial
    bk_nmi = int(bestk_row["best_k_by_NMI"]) if not np.isnan(bestk_row["best_k_by_NMI"]) else K_LIST[0]
    sub_nmi = sum_df[(sum_df["k"] == bk_nmi) & sum_df["success"]]
    if len(sub_nmi) > 0:
        best_nmi_idx  = int(sub_nmi["nmi"].idxmax())
        best_nmi_trial = int(sub_nmi.loc[best_nmi_idx, "trial"])
        best_res_nmi  = z_store.get((bk_nmi, best_nmi_trial))
        if best_res_nmi is not None and best_res_nmi.get("Z_est") is not None:
            make_z_genre_figure(
                best_res_nmi["Z_est"], genre_labels, GENRES, bk_nmi
            )

    print("\n=== Done ===")
    print(f"Results: {OUT_DIR}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
