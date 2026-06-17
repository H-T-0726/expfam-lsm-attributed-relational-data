"""
Wine dataset — fixed版 real-data pilot 実験スクリプト。

データ概要:
  sklearn の load_wine() を使用
  n=178 サンプル、d=13 連続属性、3クラス（class_0/1/2）
  X = StandardScaler().fit_transform(wine.data)  → family_x=gaussian
  Y = (label_i == label_j).astype(float), 対角=0  → family_y=bernoulli

  注意: Y はクラスラベルから構成した「類似関係行列」であり、
        実測ネットワークではない。

実験内容:
  Pilot 1: BIC k=1〜9 (5試行)
  Pilot 2: X+Y / X_only / Y_only ablation at best_k (5試行)
  Pilot 3: Z 可視化 (2D scatter, 3クラス色分け)

評価指標（真のZなし）:
  BIC, Y reconstruction AUC/AP, silhouette score, Z 2D visualization

実装: DualExpFamLSMFixed (Y-side 0.5 なし)

出力:
  expfam/results/real_data/wine_fixed_pilot/
  expfam/figures/real_data/wine_fixed_pilot/

実行例:
  cd expfam/src
  python run_fixed_real_wine_pilot.py
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

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, silhouette_score
)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

_SRC  = Path(__file__).parent
_ROOT = _SRC.parent.parent

sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from model_dual_expfam_fixed import DualExpFamLSMFixed     # noqa: E402
from utils_expfam import calc_Q_dual_strict, calc_bic_dual  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────

OUT_DIR  = _ROOT / "expfam" / "results" / "real_data" / "wine_fixed_pilot"
FIG_DIR  = _ROOT / "expfam" / "figures" / "real_data" / "wine_fixed_pilot"

FAMILY_X = "gaussian"
FAMILY_Y = "bernoulli"
K_LIST   = list(range(1, 10))   # k=1〜9
N_TRIALS = 5
L, NITER = 5, 8

# Seed規則: BIC=1000+, ablation=2000+
BIC_BASE     = 1000
ABLATION_BASE = 2000

CLASS_COLORS = ["#2196F3", "#FF5722", "#4CAF50"]
CLASS_NAMES  = ["class_0", "class_1", "class_2"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

# ─────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────

def load_wine_data():
    wine_raw = load_wine()
    X = StandardScaler().fit_transform(wine_raw.data)
    labels = wine_raw.target
    n, d = X.shape
    Y = (labels[:, None] == labels[None, :]).astype(float)
    np.fill_diagonal(Y, 0)

    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    density = float(Y[upper_mask].mean())
    class_counts = [int(np.sum(labels == c)) for c in range(3)]

    print(f"Wine dataset: n={n}, d={d}")
    print(f"Classes: {class_counts} (total {sum(class_counts)})")
    print(f"Y density (same-class pairs in upper triangle): {density:.4f}")
    print(f"Note: Y constructed from class labels, NOT a real network.")
    print()

    return X, Y, labels, n, d


# ─────────────────────────────────────────────────────────────────────
# Core EM runner (real data — no true_params)
# ─────────────────────────────────────────────────────────────────────

def run_em_fixed_wine(X, Y, family_x, family_y, k,
                      L=5, num_iter=8, seed=42,
                      fix_w=False, fix_x=False):
    """
    MC-EM using DualExpFamLSMFixed for real Wine data.
    真のZがないためProcrustes/rmse_Zは計算しない。
    返値: Q_strict, BIC, Y_AUC, Y_AP, rmse_X, silhouette, Z_est
    """
    n, d = X.shape
    max_retries = 2

    labels_for_sil = None  # set from outside if needed

    for retry in range(max_retries + 1):
        newton_alpha = 0.5 / (2 ** retry)
        rng = np.random.default_rng(seed + retry * 1000)

        model = DualExpFamLSMFixed(
            n=n, d=d, k=k, L=L,
            family_x=family_x, family_y=family_y,
        )
        # 実データ用: true_params=None → var_f=10.0 (default)
        model.initialize_params(true_params=None, seed=seed + retry * 1000)

        # Informed init: Y側
        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if family_y == "bernoulli":
            density = float(np.clip(Y[upper_mask].mean(), 1e-6, 1 - 1e-6))
            model.params["w0"] = np.log(density / (1 - density))
            model.params["w"]  = 0.5
        elif family_y == "poisson":
            upper = Y[upper_mask]
            mean_cnt = float(upper[upper > 0].mean()) if np.any(upper > 0) else 1.0
            model.params["w0"] = np.log(mean_cnt + 1e-10)
            model.params["w"]  = 0.1 / (2 ** retry)
        else:  # gaussian
            upper_vals = Y[upper_mask]
            model.params["w0"] = float(upper_vals.mean())
            model.params["w"]  = 0.5
            model.sigma_y = float(max(upper_vals.std(), 0.01))

        # Informed init: X側
        if family_x in ("bernoulli", "poisson"):
            model.params["F"] *= 0.2

        # Ablation: fix_w (X_only) / fix_x (Y_only)
        if fix_w:
            model.params["w"] = 0.0
        if fix_x:
            model.params["F"] = np.zeros((d, k))

        Z     = model.params["Z"].copy()
        F     = model.params["F"].copy()
        sigma = model.params["sigma"].copy()
        w0    = model.params["w0"]
        w     = model.params["w"]
        var_z = model.params["var_z"]

        Z_prev    = Z.copy()
        nan_count = 0

        for _ in range(1, num_iter + 1):
            # E-step
            Z_samples = np.zeros((n, k, L))
            for l in range(L):
                model.params.update(
                    dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w)
                )
                Z_new = model.calc_eta_newton(
                    X, Y, rng=rng, max_iter=10, alpha=newton_alpha
                )
                Z_samples[:, :, l] = Z_new
                Z = Z_new.copy()

            if np.any(np.isnan(Z_samples)) or np.any(np.isinf(Z_samples)):
                nan_count += 1
                Z_samples = np.stack([Z_prev] * L, axis=2)
                Z = Z_prev.copy()

            Z_samples = model.scale_Z(Z_samples)
            Z_prev = Z.copy()
            Z = Z_samples[:, :, -1].copy()

            # M-step
            if not fix_x:
                F     = model.calc_F(X, Z_samples)
                sigma = model.calc_sigma(X, Z_samples, F)
            w0 = model.calc_w0(Y, Z_samples, w0, w, max_iter=50)
            if not fix_w:
                w = model.calc_w(Y, Z_samples, w0, w, max_iter=50)
            if family_y == "gaussian":
                model.calc_sigma_y(Y, Z_samples, w0, w)

        nan_occurred = nan_count > 0
        if not nan_occurred:
            break

    model.params.update({"F": F, "sigma": sigma, "w0": w0, "w": w})
    Q_strict = calc_Q_dual_strict(X, Y, Z_samples, F, sigma, var_z, w0, w, model)
    bic, num_params = calc_bic_dual(
        Q_strict=Q_strict, k=k, n=n, d=d,
        family_x=family_x, family_y=family_y,
    )

    # Y 再構成指標
    Z_est  = Z_samples[:, :, -1]
    eta_y  = float(w0) + float(w) * (Z_est @ Z_est.T)
    mu_y   = model._mean_function(eta_y)
    y_true = Y[upper_mask].astype(int)
    y_score = mu_y[upper_mask]

    try:
        auc_y = float(roc_auc_score(y_true, y_score))
        ap_y  = float(average_precision_score(y_true, y_score))
    except Exception:
        auc_y = float("nan")
        ap_y  = float("nan")

    # X 再構成精度
    mu_x  = model._mean_function_x(Z_est @ F.T)
    rmse_x = float(np.sqrt(np.mean((X - mu_x) ** 2)))

    return {
        "Q_strict":    Q_strict,
        "bic":         bic,
        "num_params":  num_params,
        "auc_y":       auc_y,
        "ap_y":        ap_y,
        "rmse_x":      rmse_x,
        "w0":          float(w0),
        "w":           float(w),
        "Z_est":       Z_est,
        "nan_occurred": nan_occurred,
        "nan_count":   nan_count,
    }


# ─────────────────────────────────────────────────────────────────────
# Helper: silhouette score for Z
# ─────────────────────────────────────────────────────────────────────

def compute_silhouette(Z_est, labels):
    k = Z_est.shape[1]
    if k == 1:
        Z_2d = np.hstack([Z_est, np.zeros((len(Z_est), 1))])
    elif k == 2:
        Z_2d = Z_est
    else:
        Z_2d = PCA(n_components=2).fit_transform(Z_est)
    try:
        sil = float(silhouette_score(Z_2d, labels))
    except Exception:
        sil = float("nan")
    return sil, Z_2d


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────

def save_fig(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{stem}.{ext}"
        if path.exists():
            print(f"  SKIP (exists): {path}")
        else:
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"  Saved: {path}")
    plt.close(fig)


def make_bic_figure(bic_df, best_k):
    bic_mean = bic_df.groupby("k")["bic"].mean()
    bic_std  = bic_df.groupby("k")["bic"].std()

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.errorbar(
        bic_mean.index, bic_mean.values,
        yerr=bic_std.values,
        marker="o", color="#2196F3", linewidth=1.8, markersize=6,
        capsize=4, capthick=1.2, elinewidth=1.0,
        label="BIC (mean ± std, 5 trials)",
    )
    ax.axvline(x=best_k, color="red", linestyle="--", linewidth=1.2,
               label=f"Best k={best_k}")
    ax.set_xlabel("Number of latent dimensions $k$")
    ax.set_ylabel("BIC")
    ax.set_title(
        f"Wine Pilot: BIC vs. k (family_x=Gaussian, family_y=Bernoulli)\n"
        f"n={bic_df['n'].iloc[0]}, d={bic_df['d'].iloc[0]}, "
        f"{bic_df['trial'].nunique()} trials"
    )
    ax.set_xticks(K_LIST)
    ax.legend(framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, "wine_bic_k1to9")


def _z2d_scatter(ax, Z_2d, labels, title, pca_used=False):
    for c in range(3):
        mask = labels == c
        ax.scatter(
            Z_2d[mask, 0], Z_2d[mask, 1],
            c=CLASS_COLORS[c], label=CLASS_NAMES[c],
            s=30, alpha=0.75, edgecolors="none",
        )
    ax.set_title(title, fontsize=10)
    xlabel = "PC1" if pca_used else "Z₁"
    ylabel = "PC2" if pca_used else "Z₂"
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=7, loc="best", framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.3)


def make_z_figure(Z_est, labels, best_k, stem="wine_z_xy"):
    k = Z_est.shape[1]
    pca_used = k > 2
    if k == 1:
        Z_2d = np.hstack([Z_est, np.zeros((len(Z_est), 1))])
    elif k == 2:
        Z_2d = Z_est
    else:
        Z_2d = PCA(n_components=2).fit_transform(Z_est)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    _z2d_scatter(
        ax, Z_2d, labels,
        title=f"Wine Pilot: Z visualization (X+Y, k={best_k}"
              + (", PCA 2D)" if pca_used else ")"),
        pca_used=pca_used,
    )
    fig.tight_layout()
    save_fig(fig, stem)


def make_z_ablation_figure(z_dict, labels, best_k):
    """3-panel: X+Y / X_only / Y_only."""
    conditions = ["X+Y", "X_only", "Y_only"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, cond in zip(axes, conditions):
        Z_est = z_dict.get(cond)
        if Z_est is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(cond)
            continue
        k = Z_est.shape[1]
        pca_used = k > 2
        if k == 1:
            Z_2d = np.hstack([Z_est, np.zeros((len(Z_est), 1))])
        elif k == 2:
            Z_2d = Z_est
        else:
            Z_2d = PCA(n_components=2).fit_transform(Z_est)
        _z2d_scatter(ax, Z_2d, labels,
                     title=f"{cond} (k={best_k})",
                     pca_used=pca_used)

    fig.suptitle(
        "Wine Pilot: Z visualization comparison (Ablation)\n"
        f"family_x=Gaussian, family_y=Bernoulli, k={best_k}",
        fontsize=11,
    )
    fig.tight_layout()
    save_fig(fig, "wine_z_ablation_comparison")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X, Y, labels, n, d = load_wine_data()

    t0 = time.perf_counter()

    # ────────────────────────────────────────────────────────────────
    # Pilot 1: BIC k=1〜9
    # ────────────────────────────────────────────────────────────────
    total_bic = len(K_LIST) * N_TRIALS
    print(f"=== Pilot 1: BIC k=1〜9  ({len(K_LIST)} k × {N_TRIALS} trials = {total_bic} fits) ===")

    bic_rows = []
    done_bic = 0

    for k_est in K_LIST:
        for trial in range(N_TRIALS):
            done_bic += 1
            seed = BIC_BASE + k_est * 100 + trial * 10
            try:
                res = run_em_fixed_wine(
                    X, Y, FAMILY_X, FAMILY_Y, k=k_est,
                    L=L, num_iter=NITER, seed=seed,
                    fix_w=False, fix_x=False,
                )
                success = True
                err_msg = ""
            except Exception as e:
                traceback.print_exc()
                res = {
                    "Q_strict": float("nan"), "bic": float("nan"),
                    "num_params": -1, "auc_y": float("nan"),
                    "ap_y": float("nan"), "rmse_x": float("nan"),
                    "w0": float("nan"), "w": float("nan"),
                    "nan_occurred": True, "nan_count": -1,
                }
                success = False
                err_msg = str(e)

            elapsed = (time.perf_counter() - t0) / 60
            print(
                f"  [{done_bic:3d}/{total_bic}] k={k_est} t={trial}"
                f"  BIC={res['bic']:.1f}"
                f"  AUC={res['auc_y']:.4f}"
                f"  AP={res['ap_y']:.4f}"
                f"  [{elapsed:.1f} min]"
            )

            bic_rows.append({
                "k":           k_est,
                "trial":       trial,
                "seed":        seed,
                "n":           n,
                "d":           d,
                "family_x":    FAMILY_X,
                "family_y":    FAMILY_Y,
                "q_strict":    res["Q_strict"],
                "bic":         res["bic"],
                "num_params":  res["num_params"],
                "auc_y":       res["auc_y"],
                "ap_y":        res["ap_y"],
                "rmse_x":      res["rmse_x"],
                "w0":          res["w0"],
                "w":           res["w"],
                "success":     success,
                "error_message": err_msg,
                "note": "wine_fixed_pilot BIC",
            })

    bic_df = pd.DataFrame(bic_rows)
    out_bic = OUT_DIR / "wine_bic_k1to9.csv"
    if not out_bic.exists():
        bic_df.to_csv(out_bic, index=False)
        print(f"\nSaved: {out_bic}  ({len(bic_df)} rows)")
    else:
        print(f"\nSKIP (exists): {out_bic}")

    # BIC best k
    bic_mean_per_k = bic_df.groupby("k")["bic"].mean()
    best_k = int(bic_mean_per_k.idxmin())
    print(f"\n=== BIC k=1〜9 summary (mean BIC) ===")
    for k_val, bic_val in bic_mean_per_k.items():
        marker = "  ← BEST" if k_val == best_k else ""
        print(f"  k={k_val}: BIC={bic_val:.1f}{marker}")
    print(f"\nBest k = {best_k}")

    bestk_df = pd.DataFrame([{
        "best_k": best_k,
        "bic_mean_at_best_k": float(bic_mean_per_k[best_k]),
        "family_x": FAMILY_X,
        "family_y": FAMILY_Y,
        "n_trials": N_TRIALS,
        "note": "wine_fixed_pilot BIC best k",
    }])
    out_bestk = OUT_DIR / "wine_bic_bestk.csv"
    if not out_bestk.exists():
        bestk_df.to_csv(out_bestk, index=False)
        print(f"Saved: {out_bestk}")
    else:
        print(f"SKIP (exists): {out_bestk}")

    make_bic_figure(bic_df, best_k)

    # ────────────────────────────────────────────────────────────────
    # Pilot 2: Ablation (X+Y / X_only / Y_only) at best_k
    # ────────────────────────────────────────────────────────────────
    ablation_configs = [
        ("X+Y",    False, False),
        ("X_only", True,  False),
        ("Y_only", False, True),
    ]
    total_abl = len(ablation_configs) * N_TRIALS
    print(f"\n=== Pilot 2: Ablation at k={best_k}  "
          f"({len(ablation_configs)} conds × {N_TRIALS} trials = {total_abl} fits) ===")

    abl_rows = []
    done_abl = 0
    z_best_by_cond = {}  # 可視化用: 各条件のbest trial Z

    for cond_name, fix_w, fix_x in ablation_configs:
        z_best_auc = -1.0
        z_best = None

        for trial in range(N_TRIALS):
            done_abl += 1
            seed = ABLATION_BASE + {"X+Y": 0, "X_only": 100, "Y_only": 200}[cond_name] + trial * 10
            try:
                res = run_em_fixed_wine(
                    X, Y, FAMILY_X, FAMILY_Y, k=best_k,
                    L=L, num_iter=NITER, seed=seed,
                    fix_w=fix_w, fix_x=fix_x,
                )
                sil, _ = compute_silhouette(res["Z_est"], labels)
                success = True
                err_msg = ""

                if res["auc_y"] > z_best_auc:
                    z_best_auc = res["auc_y"]
                    z_best = res["Z_est"].copy()

            except Exception as e:
                traceback.print_exc()
                res = {
                    "Q_strict": float("nan"), "bic": float("nan"),
                    "num_params": -1, "auc_y": float("nan"),
                    "ap_y": float("nan"), "rmse_x": float("nan"),
                    "w0": float("nan"), "w": float("nan"),
                    "nan_occurred": True, "nan_count": -1,
                }
                sil = float("nan")
                success = False
                err_msg = str(e)

            elapsed = (time.perf_counter() - t0) / 60
            print(
                f"  [{done_abl:3d}/{total_abl}] {cond_name:8s} t={trial}"
                f"  BIC={res['bic']:.1f}"
                f"  AUC={res['auc_y']:.4f}"
                f"  AP={res['ap_y']:.4f}"
                f"  sil={sil:.4f}"
                f"  [{elapsed:.1f} min]"
            )

            abl_rows.append({
                "condition":   cond_name,
                "fix_w":       fix_w,
                "fix_x":       fix_x,
                "k":           best_k,
                "trial":       trial,
                "seed":        seed,
                "n":           n,
                "d":           d,
                "family_x":    FAMILY_X,
                "family_y":    FAMILY_Y,
                "q_strict":    res["Q_strict"],
                "bic":         res["bic"],
                "num_params":  res["num_params"],
                "auc_y":       res["auc_y"],
                "ap_y":        res["ap_y"],
                "rmse_x":      res["rmse_x"],
                "silhouette":  sil,
                "w0":          res["w0"],
                "w":           res["w"],
                "success":     success,
                "error_message": err_msg,
                "note": f"wine_fixed_pilot ablation {cond_name}",
            })

        z_best_by_cond[cond_name] = z_best

    abl_df = pd.DataFrame(abl_rows)
    out_abl = OUT_DIR / "wine_ablation_metrics.csv"
    if not out_abl.exists():
        abl_df.to_csv(out_abl, index=False)
        print(f"\nSaved: {out_abl}  ({len(abl_df)} rows)")
    else:
        print(f"\nSKIP (exists): {out_abl}")

    # Ablation集計表
    print("\n=== Ablation summary (mean across trials) ===")
    agg = abl_df.groupby("condition")[
        ["bic", "auc_y", "ap_y", "silhouette"]
    ].mean().round(4)
    print(agg.to_string())

    # ────────────────────────────────────────────────────────────────
    # Z embeddings CSV (best trial X+Y)
    # ────────────────────────────────────────────────────────────────
    Z_xy_best = z_best_by_cond.get("X+Y")
    z_emb_rows = []
    if Z_xy_best is not None:
        k_actual = Z_xy_best.shape[1]
        pca_used = k_actual > 2
        if k_actual == 1:
            Z_2d = np.hstack([Z_xy_best, np.zeros((n, 1))])
        elif k_actual == 2:
            Z_2d = Z_xy_best
        else:
            Z_2d = PCA(n_components=2).fit_transform(Z_xy_best)

        for i in range(n):
            row = {
                "sample_id":  i,
                "label":      int(labels[i]),
                "class_name": CLASS_NAMES[int(labels[i])],
                "z_pc1":      float(Z_2d[i, 0]),
                "z_pc2":      float(Z_2d[i, 1]),
                "mode":       "X+Y",
                "k":          best_k,
                "pca_used":   pca_used,
                "note": "wine_fixed_pilot X+Y best AUC trial",
            }
            for j in range(k_actual):
                row[f"z_{j+1}"] = float(Z_xy_best[i, j])
            z_emb_rows.append(row)

    z_emb_df = pd.DataFrame(z_emb_rows)
    out_zemb = OUT_DIR / "wine_z_embeddings.csv"
    if not out_zemb.exists() and len(z_emb_rows) > 0:
        z_emb_df.to_csv(out_zemb, index=False)
        print(f"Saved: {out_zemb}  ({len(z_emb_df)} rows)")
    else:
        print(f"SKIP (exists or empty): {out_zemb}")

    # ────────────────────────────────────────────────────────────────
    # Pilot 3: Z visualization
    # ────────────────────────────────────────────────────────────────
    print("\n=== Pilot 3: Z visualization ===")

    if Z_xy_best is not None:
        make_z_figure(Z_xy_best, labels, best_k, stem="wine_z_xy")
    else:
        print("  WARNING: No Z_est from X+Y — skipping wine_z_xy figure")

    make_z_ablation_figure(z_best_by_cond, labels, best_k)

    # ────────────────────────────────────────────────────────────────
    # Final summary
    # ────────────────────────────────────────────────────────────────
    elapsed_total = (time.perf_counter() - t0) / 60
    total_fits = total_bic + total_abl
    print(f"\nDone in {elapsed_total:.1f} min  ({total_fits} fits total)")

    print("\n=== BIC best k ===")
    print(f"  best_k = {best_k}  (BIC mean = {bic_mean_per_k[best_k]:.1f})")

    print("\n=== Ablation mean summary ===")
    for cond in ["X+Y", "X_only", "Y_only"]:
        sub = abl_df[abl_df["condition"] == cond]
        print(f"  {cond:8s}: AUC={sub['auc_y'].mean():.4f}  "
              f"AP={sub['ap_y'].mean():.4f}  "
              f"sil={sub['silhouette'].mean():.4f}  "
              f"BIC={sub['bic'].mean():.1f}")

    success_rate = abl_df["success"].mean()
    bic_success = bic_df["success"].mean()
    print(f"\n=== Success rate: BIC={bic_success:.0%}, Ablation={success_rate:.0%} ===")


if __name__ == "__main__":
    main()
