"""
MovieLens Bernoulli threshold=80 pilot.

Y_binary[i,j] = 1 if Y_count[i,j] >= 80 else 0   (density ~0.069)

Motivation: Poisson Y experiment used all-positive counts.
Here we binarize Y_count to obtain a sparse network (density ~0.069)
suitable for Bernoulli link prediction and comparison with Cora.

Part 1 — In-sample reconstruction:
  family_x=bernoulli, family_y=bernoulli
  k=[2,3,5], 3 trials = 9 fits

Part 2 — Held-out link prediction:
  test_edge_ratio=0.2 of positive edges → Y_train sets test positives to 0
  neg_ratio=5  →  random_AP_baseline = 1/6 ≈ 0.167
  k=[3,5], split_trials=[0,1,2], model_trials=[0,1] = 12 fits

NOTE: Setting test_positive to 0 in Y_train is an approximation for Bernoulli Y
(the model treats them as "no link"). This is the same approach used in the Cora
held-out experiment and is standard for binary link prediction.

NOTE (genre labels): primary genre is the first genre flag per movie.
MovieLens 100K movies have on average 1.72 genre tags. NMI/ARI are approximate.

Output:
  expfam/results/real_data/movielens_bernoulli_t80_pilot/
  expfam/figures/real_data/movielens_bernoulli_t80_pilot/
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
OUT_DIR  = _ROOT / "expfam" / "results" / "real_data" / "movielens_bernoulli_t80_pilot"
FIG_DIR  = _ROOT / "expfam" / "figures" / "real_data" / "movielens_bernoulli_t80_pilot"

FAMILY_X  = "bernoulli"
FAMILY_Y  = "bernoulli"
THRESHOLD = 80

# Part 1: in-sample
IS_K_LIST   = [2, 3, 5]
IS_TRIALS   = 3
IS_SEED_BASE = 13000   # + k*100 + trial*10

# Part 2: held-out
HO_K_LIST      = [3, 5]
HO_SPLIT_TRIALS = [0, 1, 2]
HO_MODEL_TRIALS = [0, 1]
HO_TEST_RATIO   = 0.2
HO_NEG_RATIO    = 5
HO_RANDOM_AP    = 1.0 / (1.0 + HO_NEG_RATIO)   # 1/6 ≈ 0.167
HO_SPLIT_BASE   = 14000   # + split_trial*100
HO_MODEL_BASE   = 15000   # + k*100 + split_trial*10 + model_trial

L, NITER = 5, 8

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
# EM runner (Bernoulli Y, Bernoulli X)
# ─────────────────────────────────────────────────────────────────────

def run_em_fixed(X, Y, family_x, family_y, k, L=5, num_iter=8, seed=42):
    n, d = X.shape
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    density    = float(np.clip(Y[upper_mask].mean(), 1e-6, 1 - 1e-6))

    nan_count_total = 0
    error_msg = ""

    for retry in range(3):
        newton_alpha = 0.5 / (2 ** retry)

        rng   = np.random.default_rng(seed + retry * 1000)
        model = DualExpFamLSMFixed(n=n, d=d, k=k, L=L,
                                   family_x=family_x, family_y=family_y)
        model.initialize_params(true_params=None, seed=seed + retry * 1000)

        # Informed init for Bernoulli Y
        if family_y == "bernoulli":
            model.params["w0"] = float(np.log(density / (1.0 - density)))
            model.params["w"]  = 0.5
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

    Z_est   = Z_samples[:, :, -1]
    eta_y   = float(w0) + float(w) * (Z_est @ Z_est.T)
    mu_y    = model._mean_function(eta_y)
    y_upper = Y[upper_mask].astype(int)
    y_scr   = mu_y[upper_mask]

    try:
        auc_y = float(roc_auc_score(y_upper, y_scr))
        ap_y  = float(average_precision_score(y_upper, y_scr))
    except Exception:
        auc_y = ap_y = float("nan")

    mu_x   = model._mean_function_x(Z_est @ F.T)
    rmse_x = float(np.sqrt(np.mean((X - mu_x) ** 2)))
    try:
        x_acc = float(np.mean((mu_x > 0.5).astype(int) == X.astype(int)))
    except Exception:
        x_acc = float("nan")

    return {
        "Q_strict": Q_strict, "bic": bic, "num_params": npar,
        "auc_y": auc_y, "ap_y": ap_y,
        "rmse_x": rmse_x, "x_binary_acc": x_acc,
        "w0": float(w0), "w": float(w),
        "Z_est": Z_est, "mu_y": mu_y,
        "nan_occurred": nan_count_total > 0, "nan_count": nan_count_total,
        "error_message": error_msg,
    }


# ─────────────────────────────────────────────────────────────────────
# Held-out helpers
# ─────────────────────────────────────────────────────────────────────

def split_positive_edges(Y, test_ratio, seed):
    """Split positive edges; Y_train sets test positives to 0."""
    rng = np.random.default_rng(seed)
    pos_i, pos_j = np.where(np.triu(Y > 0.5, k=1))
    pos_pairs    = list(zip(pos_i.tolist(), pos_j.tolist()))
    n_test       = max(1, int(len(pos_pairs) * test_ratio))
    perm         = rng.permutation(len(pos_pairs))
    test_pairs   = [pos_pairs[perm[i]] for i in range(n_test)]
    train_pairs  = [pos_pairs[perm[i]] for i in range(n_test, len(pos_pairs))]
    Y_train = Y.copy()
    for (i, j) in test_pairs:
        Y_train[i, j] = 0.0
        Y_train[j, i] = 0.0
    return Y_train, train_pairs, test_pairs


def sample_negatives(Y, pos_pairs, ratio, seed):
    rng       = np.random.default_rng(seed)
    pos_set   = {(i, j) for i, j in pos_pairs} | {(j, i) for i, j in pos_pairs}
    n         = Y.shape[0]
    n_neg     = int(len(pos_pairs) * ratio)
    negs      = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if (i, j) not in pos_set and Y[i, j] < 0.5]
    n_neg     = min(n_neg, len(negs))
    idx       = rng.choice(len(negs), size=n_neg, replace=False)
    return [negs[i] for i in idx]


def eval_link_pred(mu_y, pos_pairs, neg_pairs):
    rows = [p[0] for p in pos_pairs] + [p[0] for p in neg_pairs]
    cols = [p[1] for p in pos_pairs] + [p[1] for p in neg_pairs]
    y_true = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    y_scr  = mu_y[rows, cols].tolist()
    try:
        auc = float(roc_auc_score(y_true, y_scr))
        ap  = float(average_precision_score(y_true, y_scr))
    except Exception:
        auc = ap = float("nan")
    return auc, ap


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


def make_is_metrics_figure(agg_df, density, random_ap):
    ks = agg_df["k"].values
    metrics_spec = [
        ("bic_mean",       "bic_std",       "BIC",          "#607D8B"),
        ("auc_y_mean",     "auc_y_std",     "Y AUC",        "#2196F3"),
        ("ap_y_mean",      "ap_y_std",      "Y AP",         "#F44336"),
        ("nmi_mean",       "nmi_std",       "NMI (genre)",  "#9C27B0"),
        ("ari_mean",       "ari_std",       "ARI (genre)",  "#FF9800"),
        ("silhouette_mean","silhouette_std","Silhouette",   "#4CAF50"),
    ]
    n_m   = len(metrics_spec)
    fig, axes = plt.subplots(1, n_m, figsize=(3.8 * n_m, 4.0))
    for ax, (mu_col, sd_col, label, color) in zip(axes, metrics_spec):
        mu = agg_df[mu_col].values.astype(float)
        sd = agg_df[sd_col].values.astype(float) if sd_col in agg_df.columns else np.zeros_like(mu)
        ax.errorbar(ks, mu, yerr=sd, fmt="o-", color=color,
                    linewidth=2, markersize=7, capsize=4)
        if label == "Y AP":
            ax.axhline(random_ap, color="gray", linestyle=":", linewidth=1.2,
                       label=f"random={random_ap:.3f}")
            ax.legend(fontsize=6)
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=9)
        ax.set_xticks(ks.tolist())
        ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle(
        f"MovieLens Bernoulli t=80 — In-sample Metrics vs k (mean+/-std)\n"
        f"(n=100, d=19, X=Bern, Y=Bern, density={density:.4f}, {IS_TRIALS} trials/k)",
        fontsize=9,
    )
    fig.tight_layout()
    save_fig(fig, "movielens_bernoulli_t80_k_metrics")


def make_z_figure(Z_est, genre_labels, k_val, title_suffix=""):
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
        f"MovieLens Bernoulli t=80 — Z by primary genre (k={k_val}){title_suffix}\n"
        "NOTE: primary genre is approximate (movies have avg 1.72 genre tags)",
        fontsize=9,
    )
    ax.legend(fontsize=6, loc="best", framealpha=0.7, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "movielens_bernoulli_t80_z_by_genre")


def make_heldout_metrics_figure(ho_agg_df, random_ap):
    ks = ho_agg_df["k"].values
    metrics_spec = [
        ("train_auc_mean", "train_auc_std", "Train AUC",    "#90CAF9"),
        ("test_auc_mean",  "test_auc_std",  "Test AUC",     "#1565C0"),
        ("train_ap_mean",  "train_ap_std",  "Train AP",     "#EF9A9A"),
        ("test_ap_mean",   "test_ap_std",   "Test AP",      "#B71C1C"),
        ("nmi_mean",       "nmi_std",       "NMI (genre)",  "#9C27B0"),
    ]
    n_m   = len(metrics_spec)
    fig, axes = plt.subplots(1, n_m, figsize=(3.8 * n_m, 4.0))
    for ax, (mu_col, sd_col, label, color) in zip(axes, metrics_spec):
        mu = ho_agg_df[mu_col].values.astype(float)
        sd = ho_agg_df.get(sd_col, pd.Series(np.zeros_like(mu))).values.astype(float)
        ax.errorbar(ks, mu, yerr=sd, fmt="o-", color=color,
                    linewidth=2, markersize=7, capsize=4)
        if "AP" in label:
            ax.axhline(random_ap, color="gray", linestyle=":", linewidth=1.2,
                       label=f"random={random_ap:.3f}")
            ax.legend(fontsize=6)
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=9)
        ax.set_xticks(ks.tolist())
        ax.grid(True, linestyle="--", alpha=0.25)

    n_fits = len(HO_SPLIT_TRIALS) * len(HO_MODEL_TRIALS)
    fig.suptitle(
        f"MovieLens Bernoulli t=80 — Held-out Link Prediction (mean+/-std)\n"
        f"(test_edge_ratio=0.2, neg_ratio={HO_NEG_RATIO}, "
        f"{n_fits} fits/k, random_AP={random_ap:.3f})",
        fontsize=9,
    )
    fig.tight_layout()
    save_fig(fig, "movielens_bernoulli_t80_heldout_metrics")


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

    # Compute Y_binary_t80 in-script (not saved — no overwrite risk)
    t80_path = DATA_DIR / "movielens_Y_binary_t80.npy"
    if t80_path.exists():
        Y_binary = np.load(t80_path).astype(np.float64)
        print("  Loaded movielens_Y_binary_t80.npy")
    else:
        Y_binary = (Y_count >= THRESHOLD).astype(np.float64)
        print(f"  Computed Y_binary_t80 in-script (threshold={THRESHOLD})")

    n, d = X.shape
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    density    = float(Y_binary[upper_mask].mean())
    n_positive = int(Y_binary[upper_mask].sum())
    random_ap_is = density   # in-sample random baseline

    print("=== MovieLens Bernoulli t=80 data summary ===")
    print(f"  n={n}, d={d}, threshold={THRESHOLD}")
    print(f"  Y_density={density:.4f}  n_positive={n_positive}")
    print(f"  random_AP_baseline (in-sample)={random_ap_is:.4f}")
    print(f"  random_AP_baseline (held-out)={HO_RANDOM_AP:.4f}")

    # ── Data summary CSV ─────────────────────────────────────────────
    ds_path = OUT_DIR / "movielens_bernoulli_t80_data_summary.csv"
    if not ds_path.exists():
        pd.DataFrame([{
            "n": n, "d": d, "threshold": THRESHOLD,
            "family_x": FAMILY_X, "family_y": FAMILY_Y,
            "y_density": density, "n_positive": n_positive,
            "n_upper_pairs": int(upper_mask.sum()),
            "random_ap_insample": random_ap_is,
            "heldout_test_edge_ratio": HO_TEST_RATIO,
            "heldout_neg_ratio": HO_NEG_RATIO,
            "random_ap_heldout": HO_RANDOM_AP,
        }]).to_csv(ds_path, index=False)
        print(f"Saved: {ds_path}")

    # ─────────────────────────────────────────────────────────────────
    # PART 1: In-sample k-sweep
    # ─────────────────────────────────────────────────────────────────
    total_is = len(IS_K_LIST) * IS_TRIALS
    print(f"\n=== PART 1: In-sample k-sweep: k={IS_K_LIST}, "
          f"{IS_TRIALS} trials = {total_is} fits ===")

    t0        = time.perf_counter()
    is_rows   = []
    best_z    = {"nmi": -np.inf, "Z_est": None, "k": None}
    fit_count = 0

    for k_val in IS_K_LIST:
        for trial in range(IS_TRIALS):
            fit_count += 1
            seed = IS_SEED_BASE + k_val * 100 + trial * 10
            print(f"\n[IS {fit_count:2d}/{total_is}] k={k_val}  "
                  f"trial={trial}  seed={seed}")

            t_k = time.perf_counter()
            try:
                res     = run_em_fixed(X, Y_binary, FAMILY_X, FAMILY_Y,
                                       k=k_val, L=L, num_iter=NITER, seed=seed)
                elapsed = time.perf_counter() - t_k
                sil, nmi, ari, _ = compute_z_metrics(res["Z_est"], genre_labels)
                success  = True
                err_msg  = res.get("error_message", "")
            except Exception as e:
                elapsed = time.perf_counter() - t_k
                traceback.print_exc()
                res = {"Q_strict": float("nan"), "bic": float("nan"),
                       "num_params": 0, "auc_y": float("nan"), "ap_y": float("nan"),
                       "rmse_x": float("nan"), "x_binary_acc": float("nan"),
                       "w0": float("nan"), "w": float("nan"),
                       "Z_est": None, "nan_occurred": True,
                       "nan_count": 999, "error_message": str(e)}
                sil = nmi = ari = float("nan")
                success = False
                err_msg = str(e)

            print(f"  BIC={res['bic']:.1f}  Q={res['Q_strict']:.1f}"
                  f"  AUC={res['auc_y']:.4f}  AP={res['ap_y']:.4f}"
                  f"  NMI={nmi:.4f}  ARI={ari:.4f}"
                  f"  [{elapsed:.1f}s]"
                  f"  w0={res['w0']:.3f}  w={res['w']:.3f}")

            if success and nmi > best_z["nmi"] and res["Z_est"] is not None:
                best_z.update({"nmi": nmi, "Z_est": res["Z_est"], "k": k_val})

            is_rows.append({
                "k": k_val, "trial": trial, "seed": seed,
                "bic": res["bic"], "q_strict": res["Q_strict"],
                "num_params": res["num_params"],
                "auc_y": res["auc_y"], "ap_y": res["ap_y"],
                "random_ap_baseline": random_ap_is,
                "rmse_x": res["rmse_x"], "x_binary_acc": res["x_binary_acc"],
                "silhouette": sil, "nmi": nmi, "ari": ari,
                "w0": res["w0"], "w": res["w"],
                "nan_occurred": res["nan_occurred"],
                "nan_count": res["nan_count"],
                "success": success,
                "runtime_s": round(elapsed, 1),
                "error_message": err_msg,
            })

    # Save in-sample CSV
    is_df    = pd.DataFrame(is_rows)
    is_path  = OUT_DIR / "movielens_bernoulli_t80_summary.csv"
    if not is_path.exists():
        is_df.to_csv(is_path, index=False)
        print(f"Saved: {is_path}")

    # Aggregate
    is_num_cols = [
        "bic", "q_strict", "auc_y", "ap_y", "rmse_x", "x_binary_acc",
        "silhouette", "nmi", "ari", "runtime_s",
    ]
    is_agg_rows = []
    for k_val in IS_K_LIST:
        sub  = is_df[is_df["k"] == k_val]
        n_ok = int(sub["success"].sum())
        row  = {"k": k_val, "n_trials": len(sub), "n_success": n_ok}
        for col in is_num_cols:
            vals = sub[col].dropna().astype(float)
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"]  = float(vals.std())  if len(vals) else float("nan")
        is_agg_rows.append(row)

    is_agg_df  = pd.DataFrame(is_agg_rows)
    agg_path   = OUT_DIR / "movielens_bernoulli_t80_agg.csv"
    if not agg_path.exists():
        is_agg_df.to_csv(agg_path, index=False)
        print(f"Saved: {agg_path}")

    # Best k (in-sample)
    def best_k_col(col, df=is_agg_df, minimize=True):
        vals = df[col].values.astype(float)
        mask = ~np.isnan(vals)
        if not mask.any():
            return float("nan")
        idx = np.argmin(vals[mask]) if minimize else np.argmax(vals[mask])
        return int(df["k"].values[np.where(mask)[0][idx]])

    # Console
    print("\n=== In-sample aggregated (mean over trials) ===")
    for _, row in is_agg_df.iterrows():
        print(f"  k={int(row.k)}: BIC={row.bic_mean:.1f}"
              f"  AUC={row.auc_y_mean:.4f}"
              f"  AP={row.ap_y_mean:.4f}"
              f"  NMI={row.nmi_mean:.4f}"
              f"  ARI={row.ari_mean:.4f}"
              f"  [{row.runtime_s_mean:.0f}s]")

    # ─────────────────────────────────────────────────────────────────
    # PART 2: Held-out link prediction
    # ─────────────────────────────────────────────────────────────────
    total_ho = len(HO_K_LIST) * len(HO_SPLIT_TRIALS) * len(HO_MODEL_TRIALS)
    print(f"\n=== PART 2: Held-out link prediction: k={HO_K_LIST}, "
          f"splits={HO_SPLIT_TRIALS}, models={HO_MODEL_TRIALS} = {total_ho} fits ===")
    print(f"  test_edge_ratio={HO_TEST_RATIO}, neg_ratio={HO_NEG_RATIO}, "
          f"random_AP_baseline={HO_RANDOM_AP:.4f}")
    print("  NOTE: test_positive set to 0 in Y_train (Bernoulli approximation)")

    ho_rows   = []
    fit_count = 0

    for k_val in HO_K_LIST:
        for split_trial in HO_SPLIT_TRIALS:
            split_seed = HO_SPLIT_BASE + split_trial * 100
            Y_train, train_pos, test_pos = split_positive_edges(
                Y_binary, HO_TEST_RATIO, split_seed)

            # Negatives: sample from pairs where Y_binary = 0 in original
            neg_seed_tr = HO_SPLIT_BASE + split_trial * 100 + 50
            neg_seed_te = HO_SPLIT_BASE + split_trial * 100 + 60
            train_neg   = sample_negatives(Y_binary, train_pos, HO_NEG_RATIO, neg_seed_tr)
            test_neg    = sample_negatives(Y_binary, test_pos,  HO_NEG_RATIO, neg_seed_te)

            for model_trial in HO_MODEL_TRIALS:
                fit_count += 1
                model_seed = HO_MODEL_BASE + k_val * 100 + split_trial * 10 + model_trial
                print(f"\n[HO {fit_count:2d}/{total_ho}] k={k_val}  "
                      f"split={split_trial}  model={model_trial}  "
                      f"seed={model_seed}")
                print(f"  train_pos={len(train_pos)}  test_pos={len(test_pos)}"
                      f"  train_neg={len(train_neg)}  test_neg={len(test_neg)}")

                t_k = time.perf_counter()
                try:
                    res     = run_em_fixed(X, Y_train, FAMILY_X, FAMILY_Y,
                                           k=k_val, L=L, num_iter=NITER,
                                           seed=model_seed)
                    elapsed = time.perf_counter() - t_k
                    mu_y    = res["mu_y"]
                    train_auc, train_ap = eval_link_pred(mu_y, train_pos, train_neg)
                    test_auc,  test_ap  = eval_link_pred(mu_y, test_pos,  test_neg)
                    sil, nmi, ari, _ = compute_z_metrics(res["Z_est"], genre_labels)
                    success  = True
                    err_msg  = res.get("error_message", "")
                except Exception as e:
                    elapsed = time.perf_counter() - t_k
                    traceback.print_exc()
                    res = {"Q_strict": float("nan"), "bic": float("nan"),
                           "num_params": 0, "rmse_x": float("nan"),
                           "w0": float("nan"), "w": float("nan"),
                           "Z_est": None, "nan_occurred": True,
                           "nan_count": 999, "error_message": str(e)}
                    train_auc = train_ap = test_auc = test_ap = float("nan")
                    sil = nmi = ari = float("nan")
                    success  = False
                    err_msg  = str(e)

                print(f"  BIC={res['bic']:.1f}"
                      f"  tr_AUC={train_auc:.4f}  tr_AP={train_ap:.4f}"
                      f"  te_AUC={test_auc:.4f}   te_AP={test_ap:.4f}"
                      f"  NMI={nmi:.4f}"
                      f"  [{elapsed:.1f}s]"
                      f"  w0={res['w0']:.3f}  w={res['w']:.3f}")

                ho_rows.append({
                    "k": k_val, "split_trial": split_trial,
                    "model_trial": model_trial, "model_seed": model_seed,
                    "n_train_pos": len(train_pos), "n_test_pos": len(test_pos),
                    "n_train_neg": len(train_neg), "n_test_neg": len(test_neg),
                    "random_ap_baseline": HO_RANDOM_AP,
                    "bic": res["bic"], "q_strict": res["Q_strict"],
                    "train_auc": train_auc, "train_ap": train_ap,
                    "test_auc": test_auc, "test_ap": test_ap,
                    "test_ap_over_random": (test_ap / HO_RANDOM_AP
                                            if not np.isnan(test_ap) else float("nan")),
                    "rmse_x": res["rmse_x"],
                    "silhouette": sil, "nmi": nmi, "ari": ari,
                    "w0": res["w0"], "w": res["w"],
                    "nan_occurred": res["nan_occurred"],
                    "nan_count": res["nan_count"],
                    "success": success,
                    "runtime_s": round(elapsed, 1),
                    "error_message": err_msg,
                })

    ho_df = pd.DataFrame(ho_rows)

    ho_sum_path = OUT_DIR / "movielens_bernoulli_t80_heldout_summary.csv"
    if not ho_sum_path.exists():
        ho_df.to_csv(ho_sum_path, index=False)
        print(f"Saved: {ho_sum_path}")

    # Aggregate held-out
    ho_num_cols = [
        "bic", "train_auc", "train_ap", "test_auc", "test_ap",
        "test_ap_over_random", "rmse_x", "silhouette", "nmi", "ari", "runtime_s",
    ]
    ho_agg_rows = []
    for k_val in HO_K_LIST:
        sub  = ho_df[ho_df["k"] == k_val]
        n_ok = int(sub["success"].sum())
        row  = {"k": k_val, "n_fits": len(sub), "n_success": n_ok}
        for col in ho_num_cols:
            vals = sub[col].dropna().astype(float)
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"]  = float(vals.std())  if len(vals) else float("nan")
        ho_agg_rows.append(row)

    ho_agg_df = pd.DataFrame(ho_agg_rows)
    ho_agg_path = OUT_DIR / "movielens_bernoulli_t80_heldout_agg.csv"
    if not ho_agg_path.exists():
        ho_agg_df.to_csv(ho_agg_path, index=False)
        print(f"Saved: {ho_agg_path}")

    # Combined best k CSV
    bestk_row = {
        "best_k_by_BIC_insample":   best_k_col("bic_mean",   is_agg_df, minimize=True),
        "best_k_by_AP_insample":    best_k_col("ap_y_mean",  is_agg_df, minimize=False),
        "best_k_by_AUC_insample":   best_k_col("auc_y_mean", is_agg_df, minimize=False),
        "best_k_by_NMI_insample":   best_k_col("nmi_mean",   is_agg_df, minimize=False),
        "best_k_by_ARI_insample":   best_k_col("ari_mean",   is_agg_df, minimize=False),
        "best_k_by_test_AP_heldout": best_k_col("test_ap_mean",  ho_agg_df, minimize=False),
        "best_k_by_test_AUC_heldout": best_k_col("test_auc_mean", ho_agg_df, minimize=False),
    }
    bk_path = OUT_DIR / "movielens_bernoulli_t80_bestk.csv"
    if not bk_path.exists():
        pd.DataFrame([bestk_row]).to_csv(bk_path, index=False)
        print(f"Saved: {bk_path}")

    elapsed_total = time.perf_counter() - t0
    print(f"\nTotal done in {elapsed_total/60:.1f} min")

    # Console held-out summary
    print("\n=== Held-out aggregated (mean over splits x models) ===")
    for _, row in ho_agg_df.iterrows():
        print(f"  k={int(row.k)} ({int(row.n_success)}/{int(row.n_fits)} ok)"
              f"  tr_AUC={row.train_auc_mean:.4f}"
              f"  te_AUC={row.test_auc_mean:.4f}"
              f"  tr_AP={row.train_ap_mean:.4f}"
              f"  te_AP={row.test_ap_mean:.4f}"
              f"  AP/rand={row.test_ap_over_random_mean:.2f}x"
              f"  NMI={row.nmi_mean:.4f}")

    print("\n=== Best k ===")
    for name, val in bestk_row.items():
        print(f"  {name}={val}")

    # ── Figures ─────────────────────────────────────────────────────
    print("\n=== Figures ===")
    make_is_metrics_figure(is_agg_df, density, random_ap_is)
    if best_z["Z_est"] is not None:
        make_z_figure(best_z["Z_est"], genre_labels, best_z["k"])
    make_heldout_metrics_figure(ho_agg_df, HO_RANDOM_AP)

    print(f"\nResults: {OUT_DIR}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
