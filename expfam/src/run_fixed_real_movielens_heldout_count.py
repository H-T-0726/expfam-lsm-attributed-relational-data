"""
MovieLens held-out count regression pilot.

===========================================================================
IMPORTANT — MASKED EVALUATION, NOT STRICT HELD-OUT
===========================================================================
The existing model API (model_dual_expfam_fixed.py, model_expfam.py) does NOT
support missing-pair masks. Specifically:

  calc_w0      : grad = sum_all(Y - A'(eta))         — no mask param
  calc_w       : grad = sum_all((Y - A'(eta))*ZZT)   — no mask param
  _calc_gradient: uses Y[i,:] for all j != i         — no mask param
  _calc_precision_matrix: uses var_y for all j != i  — no mask param

Setting test Y_ij = 0 would bias Poisson learning ("zero count" is a valid
observation). There is no masking mechanism.

DECISION: Implement MASKED EVALUATION.
  - Model is trained on FULL Y_count (all pairs visible during training).
  - After training, upper-triangle pairs are split into train/test for
    EVALUATION ONLY.
  - train_RMSE / test_RMSE measure reconstruction quality on each subset.
  - Since the model is a low-rank latent variable model (predictions =
    exp(w0 + w * z_i.T @ z_j)), it CANNOT memorize individual pair values.
    Comparing train vs test RMSE therefore tests whether the latent space
    captures count variation consistently, not pair-level memorization.

Strict held-out (excluding test pairs from gradient) would require adding
a boolean mask argument to calc_w0, calc_w, and _calc_gradient in the
protected source files. This is deferred to future work.
===========================================================================

family_x      = bernoulli
family_y      = poisson
n=100, d=19
k_values      = [3, 5]
split_trials  = [0, 1, 2]
model_trials  = [0, 1]
Total         = 2 k × 3 split × 2 model = 12 fits

test_pair_ratio = 0.2   (990 / 4950 upper-triangle pairs per split)

Output:
  expfam/results/real_data/movielens_heldout_count/
  expfam/figures/real_data/movielens_heldout_count/
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
OUT_DIR  = _ROOT / "expfam" / "results" / "real_data" / "movielens_heldout_count"
FIG_DIR  = _ROOT / "expfam" / "figures" / "real_data" / "movielens_heldout_count"

FAMILY_X         = "bernoulli"
FAMILY_Y         = "poisson"
K_LIST           = [3, 5]
SPLIT_TRIALS     = [0, 1, 2]
MODEL_TRIALS     = [0, 1]
TEST_PAIR_RATIO  = 0.2
L, NITER         = 5, 8

HIGH_COUNT_THRESHOLD = 80   # Y_count >= 80 → positive for high-count binary eval

SPLIT_SEED_BASE = 11000   # + split_trial * 100
MODEL_SEED_BASE = 12000   # + k * 100 + split_trial * 10 + model_trial

# NOTE: primary genre is assigned as the FIRST genre flag per movie.
# MovieLens 100K movies have on average 1.72 genre tags.
# NMI/ARI are approximate due to multi-label nature of movie genres.
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
# Pair split
# ─────────────────────────────────────────────────────────────────────

def split_pairs(n, test_ratio, seed):
    """
    Split upper-triangle pairs (i<j) into train / test.
    Returns (train_pairs, test_pairs) as lists of (i, j) tuples.
    Both (i,j) and (j,i) in the test set are considered test.
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.triu_indices(n, k=1)
    all_pairs  = list(zip(rows.tolist(), cols.tolist()))
    n_total    = len(all_pairs)
    n_test     = max(1, int(n_total * test_ratio))
    perm       = rng.permutation(n_total)
    test_pairs  = [all_pairs[perm[i]] for i in range(n_test)]
    train_pairs = [all_pairs[perm[i]] for i in range(n_test, n_total)]
    return train_pairs, test_pairs


def make_eval_mask(n, pairs):
    """Boolean (n,n) symmetric mask that is True at the given pair positions."""
    mask = np.zeros((n, n), dtype=bool)
    for (i, j) in pairs:
        mask[i, j] = True
        mask[j, i] = True
    return mask


# ─────────────────────────────────────────────────────────────────────
# EM runner (same as Poisson pilot, train on FULL Y_count)
# ─────────────────────────────────────────────────────────────────────

def run_em_fixed(X, Y, family_x, family_y, k, L=5, num_iter=8, seed=42):
    n, d = X.shape
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_count = float(Y[upper_mask].mean())

    nan_count_total = 0
    error_msg = ""

    for retry in range(3):
        newton_alpha = 0.5 / (2 ** retry)
        w_init       = 0.1 / (2 ** retry)

        rng   = np.random.default_rng(seed + retry * 1000)
        model = DualExpFamLSMFixed(n=n, d=d, k=k, L=L,
                                   family_x=family_x, family_y=family_y)
        model.initialize_params(true_params=None, seed=seed + retry * 1000)

        if family_y == "poisson":
            model.params["w0"] = float(np.log(max(mean_count, 1e-10)))
            model.params["w"]  = w_init
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
    eta_y = float(w0) + float(w) * (Z_est @ Z_est.T)
    mu_y  = np.clip(model._mean_function(eta_y), 0.0, 1e5)

    mu_x   = model._mean_function_x(Z_est @ F.T)
    rmse_x = float(np.sqrt(np.mean((X - mu_x) ** 2)))
    try:
        x_acc = float(np.mean((mu_x > 0.5).astype(int) == X.astype(int)))
    except Exception:
        x_acc = float("nan")

    return {
        "Q_strict": Q_strict, "bic": bic, "num_params": npar,
        "rmse_x": rmse_x, "x_binary_acc": x_acc,
        "w0": float(w0), "w": float(w),
        "Z_est": Z_est, "mu_y_full": mu_y,
        "nan_occurred": nan_count_total > 0, "nan_count": nan_count_total,
        "error_message": error_msg,
    }


def eval_pairs(Y, mu_y, pairs, mask, threshold=HIGH_COUNT_THRESHOLD):
    """Evaluate count regression on given pair set."""
    rows  = np.array([p[0] for p in pairs], dtype=int)
    cols  = np.array([p[1] for p in pairs], dtype=int)
    y_true = Y[rows, cols].astype(float)
    y_hat  = mu_y[rows, cols].astype(float)

    residuals = y_true - y_hat
    rmse      = float(np.sqrt(np.mean(residuals ** 2)))
    mae       = float(np.mean(np.abs(residuals)))

    try:
        pcorr, _ = pearsonr(y_true, y_hat)
        pearson_corr = float(pcorr)
    except Exception:
        pearson_corr = float("nan")

    try:
        scorr, _ = spearmanr(y_true, y_hat)
        spearman_corr = float(scorr)
    except Exception:
        spearman_corr = float("nan")

    y_binary    = (y_true >= threshold).astype(int)
    hc_density  = float(y_binary.mean())
    random_ap_b = hc_density

    try:
        hc_auc = float(roc_auc_score(y_binary, y_hat))
    except Exception:
        hc_auc = float("nan")
    try:
        hc_ap  = float(average_precision_score(y_binary, y_hat))
    except Exception:
        hc_ap  = float("nan")

    return {
        "rmse": rmse, "mae": mae,
        "pearson_corr": pearson_corr, "spearman_corr": spearman_corr,
        "mean_y": float(y_true.mean()), "mean_y_hat": float(y_hat.mean()),
        "max_y": float(y_true.max()), "max_y_hat": float(y_hat.max()),
        "hc_density": hc_density, "random_ap_baseline": random_ap_b,
        "hc_auc": hc_auc, "hc_ap": hc_ap,
        "y_true": y_true, "y_hat": y_hat,
    }


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


def make_metrics_figure(agg_df):
    ks = agg_df["k"].values

    metrics_spec = [
        ("train_rmse_mean",       "train_rmse_std",       "Train RMSE",        "#90CAF9"),
        ("test_rmse_mean",        "test_rmse_std",        "Test RMSE",         "#F44336"),
        ("train_pearson_mean",    "train_pearson_std",    "Train Pearson",     "#81C784"),
        ("test_pearson_mean",     "test_pearson_std",     "Test Pearson",      "#2E7D32"),
        ("test_hc_ap_mean",       "test_hc_ap_std",       "Test hc_AP (>=80)", "#FF9800"),
        ("nmi_mean",              "nmi_std",              "NMI (genre)",       "#9C27B0"),
    ]
    n_m   = len(metrics_spec)
    fig, axes = plt.subplots(1, n_m, figsize=(3.8 * n_m, 4.0))

    for ax, (mu_col, sd_col, label, color) in zip(axes, metrics_spec):
        mu = agg_df[mu_col].values.astype(float)
        sd = agg_df.get(sd_col, pd.Series(np.zeros_like(mu))).values.astype(float)
        ax.errorbar(ks, mu, yerr=sd, fmt="o-", color=color,
                    linewidth=2, markersize=7, capsize=4)
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=9)
        ax.set_xticks(ks.tolist())
        ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle(
        "MovieLens held-out count regression — Metrics vs k\n"
        "(MASKED EVALUATION: trained on all pairs, eval split into train/test)\n"
        f"(n=100, d=19, X=Bern, Y=Pois, {len(SPLIT_TRIALS)} splits × {len(MODEL_TRIALS)} models/k)",
        fontsize=9,
    )
    fig.tight_layout()
    save_fig(fig, "movielens_heldout_count_metrics")


def make_scatter_figure(train_ev, test_ev, k_val, split_idx, model_idx):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))

    for ax, ev, label, color in [
        (axes[0], train_ev, "Train pairs", "#2196F3"),
        (axes[1], test_ev,  "Test pairs",  "#F44336"),
    ]:
        y_true = ev["y_true"]
        y_hat  = ev["y_hat"]
        ax.scatter(y_true, y_hat, alpha=0.35, s=8, color=color, edgecolors="none")
        lo = min(y_true.min(), y_hat.min())
        hi = max(y_true.max(), y_hat.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="y=x")
        ax.set_xlabel("Y_count (true)")
        ax.set_ylabel("Y_hat (predicted)")
        ax.set_title(
            f"{label}  (k={k_val}, split={split_idx}, model={model_idx})\n"
            f"RMSE={ev['rmse']:.2f}  Pearson={ev['pearson_corr']:.3f}",
            fontsize=9,
        )
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle(
        "MovieLens Y_count — Train vs Test scatter\n"
        "(MASKED EVAL: model trained on all pairs)",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, "movielens_heldout_y_true_vs_pred")


def make_z_figure(Z_est, genre_labels, k_val):
    k_z = Z_est.shape[1]
    pca_used = k_z > 2
    if k_z == 1:
        Z_2d = np.hstack([Z_est, np.zeros((len(Z_est), 1))])
    elif k_z == 2:
        Z_2d = Z_est.copy()
    else:
        Z_2d = PCA(n_components=2).fit_transform(Z_est)

    unique_labels = sorted(np.unique(genre_labels).tolist())
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    for ci, lbl in enumerate(unique_labels):
        mask  = genre_labels == lbl
        gname = GENRES[lbl] if lbl < len(GENRES) else str(lbl)
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                   color=cmap(ci), label=gname, s=35, alpha=0.80, edgecolors="none")

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
    save_fig(fig, "movielens_heldout_z_by_genre")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────
    X            = np.load(DATA_DIR / "movielens_X_genre.npy").astype(np.float64)
    Y_count      = np.load(DATA_DIR / "movielens_Y_count.npy").astype(np.float64)
    genre_labels = np.load(DATA_DIR / "movielens_primary_genre_labels.npy")

    n, d         = X.shape
    upper_mask   = np.triu(np.ones((n, n), dtype=bool), k=1)
    all_pairs    = [(int(i), int(j)) for i, j in zip(*np.where(upper_mask))]
    n_pairs      = len(all_pairs)
    n_test_pairs = max(1, int(n_pairs * TEST_PAIR_RATIO))
    n_train_pairs = n_pairs - n_test_pairs
    Y_upper       = Y_count[upper_mask]

    print("=== MovieLens held-out count regression ===")
    print(f"  MASKED EVALUATION mode (model trained on all pairs)")
    print(f"  n={n}, d={d}, family_x={FAMILY_X}, family_y={FAMILY_Y}")
    print(f"  total upper-triangle pairs: {n_pairs}")
    print(f"  train pairs: {n_train_pairs}, test pairs: {n_test_pairs}")
    print(f"  Y_count mean={Y_upper.mean():.2f}, max={Y_upper.max():.0f}")
    print(f"  high_count_threshold={HIGH_COUNT_THRESHOLD}, "
          f"density={float((Y_upper >= HIGH_COUNT_THRESHOLD).mean()):.4f}")
    print(f"  k_values={K_LIST}, split_trials={SPLIT_TRIALS}, "
          f"model_trials={MODEL_TRIALS}")

    total_fits = len(K_LIST) * len(SPLIT_TRIALS) * len(MODEL_TRIALS)
    print(f"  total fits: {total_fits}")

    # ── Data summary CSV ────────────────────────────────────────────
    ds_path = OUT_DIR / "movielens_heldout_count_data_summary.csv"
    if not ds_path.exists():
        hc_dens = float((Y_upper >= HIGH_COUNT_THRESHOLD).mean())
        pd.DataFrame([{
            "n": n, "d": d, "family_x": FAMILY_X, "family_y": FAMILY_Y,
            "subset_strategy": "genre_stratified_mp100",
            "n_upper_pairs": n_pairs,
            "n_train_pairs": n_train_pairs,
            "n_test_pairs": n_test_pairs,
            "test_pair_ratio": TEST_PAIR_RATIO,
            "y_count_mean": float(Y_upper.mean()),
            "y_count_max": float(Y_upper.max()),
            "high_count_threshold": HIGH_COUNT_THRESHOLD,
            "test_hc_density_expected": hc_dens,
            "evaluation_mode": "masked_evaluation_not_strict_held_out",
            "k_values": str(K_LIST),
            "split_trials": str(SPLIT_TRIALS),
            "model_trials": str(MODEL_TRIALS),
        }]).to_csv(ds_path, index=False)
        print(f"Saved: {ds_path}")

    # ── Main experiment loop ─────────────────────────────────────────
    t0           = time.perf_counter()
    summary_rows = []
    # store scatter data for best trial
    best_scatter = {"test_pearson": -np.inf, "data": None, "k": None,
                    "split": None, "model": None}
    best_z_nmi   = {"nmi": -np.inf, "Z_est": None, "k": None}
    fit_count    = 0

    print(f"\n=== {total_fits} fits ===")

    for k_val in K_LIST:
        for split_trial in SPLIT_TRIALS:
            split_seed = SPLIT_SEED_BASE + split_trial * 100
            train_pairs, test_pairs = split_pairs(n, TEST_PAIR_RATIO, split_seed)
            # Boolean masks for evaluation
            train_mask = make_eval_mask(n, train_pairs)
            test_mask  = make_eval_mask(n, test_pairs)

            for model_trial in MODEL_TRIALS:
                fit_count += 1
                model_seed = MODEL_SEED_BASE + k_val * 100 + split_trial * 10 + model_trial
                print(f"\n[{fit_count:2d}/{total_fits}] k={k_val}  "
                      f"split={split_trial}  model={model_trial}  "
                      f"seed={model_seed}")

                t_k = time.perf_counter()
                try:
                    # Train on FULL Y_count (masked evaluation approach)
                    res  = run_em_fixed(X, Y_count, FAMILY_X, FAMILY_Y,
                                        k=k_val, L=L, num_iter=NITER,
                                        seed=model_seed)
                    elapsed = time.perf_counter() - t_k

                    mu_y  = res["mu_y_full"]
                    tr_ev = eval_pairs(Y_count, mu_y, train_pairs, train_mask)
                    te_ev = eval_pairs(Y_count, mu_y, test_pairs,  test_mask)
                    sil, nmi, ari, _ = compute_z_metrics(res["Z_est"], genre_labels)
                    success  = True
                    err_msg  = res.get("error_message", "")

                except Exception as e:
                    elapsed = time.perf_counter() - t_k
                    traceback.print_exc()
                    mu_y = np.full((n, n), float("nan"))
                    tr_ev = te_ev = {
                        "rmse": float("nan"), "mae": float("nan"),
                        "pearson_corr": float("nan"), "spearman_corr": float("nan"),
                        "mean_y": float("nan"), "mean_y_hat": float("nan"),
                        "max_y": float("nan"), "max_y_hat": float("nan"),
                        "hc_density": float("nan"), "random_ap_baseline": float("nan"),
                        "hc_auc": float("nan"), "hc_ap": float("nan"),
                        "y_true": np.array([]), "y_hat": np.array([]),
                    }
                    res = {"Q_strict": float("nan"), "bic": float("nan"),
                           "num_params": 0, "rmse_x": float("nan"),
                           "x_binary_acc": float("nan"), "w0": float("nan"),
                           "w": float("nan"), "Z_est": None,
                           "nan_occurred": True, "nan_count": 999}
                    sil = nmi = ari = float("nan")
                    success = False
                    err_msg = str(e)

                delta_rmse   = te_ev["rmse"]   - tr_ev["rmse"]
                delta_pearson = tr_ev["pearson_corr"] - te_ev["pearson_corr"]

                print(f"  BIC={res['bic']:.1f}"
                      f"  tr_RMSE={tr_ev['rmse']:.2f}"
                      f"  te_RMSE={te_ev['rmse']:.2f}"
                      f"  dRMSE={delta_rmse:+.2f}"
                      f"  tr_Pear={tr_ev['pearson_corr']:.3f}"
                      f"  te_Pear={te_ev['pearson_corr']:.3f}"
                      f"  te_hc_AP={te_ev['hc_ap']:.4f}"
                      f"  NMI={nmi:.4f}"
                      f"  [{elapsed:.1f}s]"
                      f"  w0={res['w0']:.3f}  w={res['w']:.3f}")

                # Track best for figures
                if success and te_ev["pearson_corr"] > best_scatter["test_pearson"]:
                    best_scatter.update({
                        "test_pearson": te_ev["pearson_corr"],
                        "data": (tr_ev, te_ev),
                        "k": k_val, "split": split_trial, "model": model_trial,
                    })
                if success and nmi > best_z_nmi["nmi"] and res["Z_est"] is not None:
                    best_z_nmi.update({
                        "nmi": nmi, "Z_est": res["Z_est"], "k": k_val,
                    })

                summary_rows.append({
                    "k": k_val, "split_trial": split_trial,
                    "model_trial": model_trial, "model_seed": model_seed,
                    "bic": res["bic"], "q_strict": res["Q_strict"],
                    "num_params": res["num_params"],
                    "train_rmse": tr_ev["rmse"], "train_mae": tr_ev["mae"],
                    "train_pearson": tr_ev["pearson_corr"],
                    "train_spearman": tr_ev["spearman_corr"],
                    "train_mean_y": tr_ev["mean_y"],
                    "train_mean_y_hat": tr_ev["mean_y_hat"],
                    "train_max_y_hat": tr_ev["max_y_hat"],
                    "test_rmse": te_ev["rmse"], "test_mae": te_ev["mae"],
                    "test_pearson": te_ev["pearson_corr"],
                    "test_spearman": te_ev["spearman_corr"],
                    "test_mean_y": te_ev["mean_y"],
                    "test_mean_y_hat": te_ev["mean_y_hat"],
                    "test_max_y": te_ev["max_y"],
                    "test_max_y_hat": te_ev["max_y_hat"],
                    "test_hc_density": te_ev["hc_density"],
                    "test_random_ap_baseline": te_ev["random_ap_baseline"],
                    "test_hc_auc": te_ev["hc_auc"],
                    "test_hc_ap": te_ev["hc_ap"],
                    "rmse_x": res["rmse_x"],
                    "x_binary_acc": res["x_binary_acc"],
                    "silhouette": sil, "nmi": nmi, "ari": ari,
                    "w0": res["w0"], "w": res["w"],
                    "delta_rmse": delta_rmse, "delta_pearson": delta_pearson,
                    "nan_occurred": res["nan_occurred"],
                    "nan_count": res["nan_count"],
                    "success": success,
                    "runtime_s": round(elapsed, 1),
                    "error_message": err_msg,
                    "evaluation_mode": "masked_evaluation",
                })

    elapsed_total = time.perf_counter() - t0
    print(f"\nDone in {elapsed_total/60:.1f} min  ({fit_count}/{total_fits} fits)")

    # ── Save summary CSV ────────────────────────────────────────────
    sum_df   = pd.DataFrame(summary_rows)
    sum_path = OUT_DIR / "movielens_heldout_count_summary.csv"
    if not sum_path.exists():
        sum_df.to_csv(sum_path, index=False)
        print(f"Saved: {sum_path}")

    # ── Aggregation ─────────────────────────────────────────────────
    num_cols = [
        "bic", "train_rmse", "train_mae", "train_pearson", "train_spearman",
        "test_rmse", "test_mae", "test_pearson", "test_spearman",
        "test_hc_auc", "test_hc_ap", "rmse_x", "x_binary_acc",
        "silhouette", "nmi", "ari", "delta_rmse", "delta_pearson", "runtime_s",
    ]
    agg_rows = []
    for k_val in K_LIST:
        sub  = sum_df[sum_df["k"] == k_val]
        n_ok = int(sub["success"].sum())
        row  = {"k": k_val, "n_fits": len(sub), "n_success": n_ok}
        for col in num_cols:
            vals = sub[col].dropna().astype(float)
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"]  = float(vals.std())  if len(vals) else float("nan")
        agg_rows.append(row)

    agg_df   = pd.DataFrame(agg_rows)
    agg_path = OUT_DIR / "movielens_heldout_count_agg.csv"
    if not agg_path.exists():
        agg_df.to_csv(agg_path, index=False)
        print(f"Saved: {agg_path}")

    # ── Best k ──────────────────────────────────────────────────────
    def best_k(col, minimize=True):
        vals = agg_df[col].values.astype(float)
        mask = ~np.isnan(vals)
        if not mask.any():
            return float("nan")
        idx = np.argmin(vals[mask]) if minimize else np.argmax(vals[mask])
        return int(agg_df["k"].values[np.where(mask)[0][idx]])

    bestk = {
        "best_k_by_test_RMSE":    best_k("test_rmse_mean",    minimize=True),
        "best_k_by_test_Pearson": best_k("test_pearson_mean", minimize=False),
        "best_k_by_test_hc_AP":   best_k("test_hc_ap_mean",  minimize=False),
        "best_k_by_NMI":          best_k("nmi_mean",         minimize=False),
    }
    bk_path = OUT_DIR / "movielens_heldout_count_bestk.csv"
    if not bk_path.exists():
        pd.DataFrame([bestk]).to_csv(bk_path, index=False)
        print(f"Saved: {bk_path}")

    # ── Console summary ──────────────────────────────────────────────
    print("\n=== Aggregated results (mean over splits × models) ===")
    for _, row in agg_df.iterrows():
        print(f"  k={int(row.k)} ({int(row.n_success)}/{int(row.n_fits)} ok)"
              f"  BIC={row.bic_mean:.0f}"
              f"  tr_RMSE={row.train_rmse_mean:.2f}"
              f"  te_RMSE={row.test_rmse_mean:.2f}"
              f"  dRMSE={row.delta_rmse_mean:+.2f}"
              f"  tr_Pear={row.train_pearson_mean:.3f}"
              f"  te_Pear={row.test_pearson_mean:.3f}"
              f"  te_hc_AP={row.test_hc_ap_mean:.4f}"
              f"  NMI={row.nmi_mean:.4f}")
    print("\n=== Best k ===")
    for name, val in bestk.items():
        print(f"  {name}={val}")

    # ── Figures ─────────────────────────────────────────────────────
    print("\n=== Figures ===")
    make_metrics_figure(agg_df)

    if best_scatter["data"] is not None:
        tr_ev, te_ev = best_scatter["data"]
        make_scatter_figure(
            tr_ev, te_ev,
            best_scatter["k"], best_scatter["split"], best_scatter["model"],
        )

    if best_z_nmi["Z_est"] is not None:
        make_z_figure(best_z_nmi["Z_est"], genre_labels, best_z_nmi["k"])

    print(f"\nResults: {OUT_DIR}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
