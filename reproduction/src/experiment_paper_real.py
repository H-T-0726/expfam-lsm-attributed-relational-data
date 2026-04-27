"""
Real Data Experiment: Wine dataset.

Uses sklearn Wine dataset (n=178, d=13).
- X: standardized features (mean=0, std=1)
- Y: binary relation matrix (y_ij=1 if same class, 0 otherwise)
- k=6 (selected by BIC per paper)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, Any, Optional
import time

import sys
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

from model import LatentStructuralModel


OUTPUT_DIR = Path("c:/研究2/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINE_FEATURE_NAMES = [
    "alcohol", "malic acid", "ash", "alcalinity of ash",
    "magnesium", "total phenols", "flavanoids",
    "nonflavanoid phenols", "proanthocyanins",
    "color intensity", "hue",
    "OD280/OD315", "proline"
]


def load_and_prepare_data():
    """Load Wine dataset and construct X and Y."""
    wine = load_wine()
    X_raw = wine.data          # (178, 13)
    labels = wine.target       # (178,) values in {0, 1, 2}

    # Standardize X: mean=0, std=1 per feature
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)   # (178, 13)

    # Build symmetric binary relation matrix Y
    # y_ij = 1 if same class, 0 otherwise; diagonal = 0
    n = len(labels)
    Y = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j and labels[i] == labels[j]:
                Y[i, j] = 1.0

    print(f"  n={n}, d={X.shape[1]}")
    print(f"  Class counts: {np.bincount(labels)}")
    print(f"  Y density: {np.mean(Y):.4f}  "
          f"(expected ~{sum(c*(c-1) for c in np.bincount(labels)) / (n*(n-1)):.4f})")
    return X, Y, labels


def calc_rmse(true: np.ndarray, est: np.ndarray) -> float:
    return np.sqrt(np.mean((true - est) ** 2))


def calc_log_likelihood(X, Y, Z, F, sigma, var_z, w0, w):
    n, k = Z.shape
    d = X.shape[1]

    ln_p_Z = (
        -(n * k / 2) * np.log(2 * np.pi)
        - (n * k / 2) * np.log(var_z)
        - (1 / (2 * var_z)) * np.sum(Z ** 2)
    )

    residual = X - Z @ F.T
    sigma_diag = np.diag(sigma)
    ln_p_X = (
        -(n * d / 2) * np.log(2 * np.pi)
        - (n / 2) * np.sum(np.log(sigma_diag))
        - 0.5 * np.sum(residual ** 2 / sigma_diag)
    )

    ZZT = Z @ Z.T
    logits = w0 + w * ZZT
    S = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    S = np.clip(S, 1e-10, 1 - 1e-10)

    ln_p_Y = Y * np.log(S) + (1 - Y) * np.log(1 - S)
    np.fill_diagonal(ln_p_Y, 0.0)
    ln_p_Y = 0.5 * np.sum(ln_p_Y)

    return ln_p_Z + ln_p_X + ln_p_Y


def run_em(X, Y, k, num_iter=30, L=10, init_seed=42, verbose=True):
    """Run EM algorithm on real data (no true params)."""
    n, d = X.shape

    model = LatentStructuralModel(n=n, d=d, k=k, L=L)
    model.initialize_params(true_params=None, seed=init_seed)

    # Informed initialization from Y density
    y_density = np.mean(Y)
    w0_init = np.log(y_density / (1 - y_density + 1e-10))
    model.params['w0'] = w0_init
    model.params['w'] = 0.5

    rng = np.random.default_rng(init_seed)

    Z = model.params['Z'].copy()
    F = model.params['F'].copy()
    sigma = model.params['sigma'].copy()
    w0 = model.params['w0']
    w = model.params['w']
    var_z = model.params['var_z']

    best_log_likelihood = -np.inf
    best_state = None

    for iteration in range(1, num_iter + 1):
        # E-Step
        Z_samples = np.zeros((n, k, L))
        for l in range(L):
            model.params['Z'] = Z.copy()
            model.params['F'] = F
            model.params['sigma'] = sigma
            model.params['w0'] = w0
            model.params['w'] = w

            Z_new = model.calc_eta_newton(X, Y, rng=rng, max_iter=10, alpha=0.5)
            Z_samples[:, :, l] = Z_new
            Z = Z_new.copy()

        Z_samples = model.scale_Z(Z_samples)
        Z = Z_samples[:, :, -1].copy()

        # M-Step
        F = model.calc_F(X, Z_samples)
        sigma = model.calc_sigma(X, Z_samples, F)
        w0 = model.calc_w0(Y, Z_samples, w0, w, max_iter=50)
        w = model.calc_w(Y, Z_samples, w0, w, max_iter=50)

        log_likelihood = calc_log_likelihood(X, Y, Z, F, sigma, var_z, w0, w)

        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_state = {
                'Z': Z.copy(), 'F': F.copy(), 'sigma': sigma.copy(),
                'w0': w0, 'w': w
            }

        if verbose and (iteration % 5 == 0 or iteration == 1):
            print(f"  Iter {iteration:3d}: logL={log_likelihood:10.2f}  "
                  f"w0={w0:.4f}  w={w:.4f}")

    return best_state, best_log_likelihood


def plot_F_heatmap(F: np.ndarray, feature_names: list, save_path: Path):
    """Plot F (d x k) as heatmap with feature names on y-axis."""
    d, k = F.shape

    # Normalize columns for display (each latent factor)
    F_norm = F / (np.max(np.abs(F), axis=0, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Symmetric colormap centered at 0
    vmax = np.max(np.abs(F_norm))
    im = ax.imshow(
        F_norm,
        aspect='auto',
        cmap='RdBu_r',
        vmin=-vmax, vmax=vmax
    )

    # Axes labels
    ax.set_yticks(range(d))
    ax.set_yticklabels(feature_names, fontsize=11)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"z{i+1}" for i in range(k)], fontsize=12)
    ax.set_xlabel("Latent Dimension", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(r"Estimated Loading Matrix $\hat{F}$" + f"  (k={k})", fontsize=13)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized loading", fontsize=10)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, d, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("=" * 60)
    print("Real Data Experiment: Wine Dataset")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────
    print("\n[1] Loading Wine dataset...")
    X, Y, labels = load_and_prepare_data()
    n, d = X.shape

    # ── Settings ──────────────────────────────────────
    k = 6
    L = 10
    num_iter = 30
    n_trials = 3

    print(f"\n[2] Running EM  (k={k}, L={L}, num_iter={num_iter}, n_trials={n_trials})")

    best_overall_state = None
    best_overall_logL = -np.inf

    for trial in range(n_trials):
        init_seed = 42 + trial * 1000
        print(f"\n  --- Trial {trial + 1}/{n_trials}  (seed={init_seed}) ---")
        start = time.time()

        state, logL = run_em(
            X, Y, k,
            num_iter=num_iter, L=L,
            init_seed=init_seed, verbose=True
        )
        elapsed = time.time() - start
        print(f"  Best logL={logL:.2f}   ({elapsed:.1f}s)")

        if logL > best_overall_logL:
            best_overall_logL = logL
            best_overall_state = state

    # ── Evaluation ────────────────────────────────────
    print("\n[3] Evaluation")
    Z_est = best_overall_state['Z']
    F_est = best_overall_state['F']
    w0_est = best_overall_state['w0']
    w_est = best_overall_state['w']

    # RMSE(X)
    X_recon = Z_est @ F_est.T
    rmse_X = calc_rmse(X, X_recon)

    # RMSE(Y)
    ZZT = Z_est @ Z_est.T
    logits = w0_est + w_est * ZZT
    Y_pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    np.fill_diagonal(Y_pred, 0.0)
    rmse_Y = calc_rmse(Y, Y_pred)

    print(f"\n  Results (paper: RMSE(X)=0.7924, RMSE(Y)=0.1415, w0~=-1.18, w~=1.72)")
    print(f"  {'RMSE(X)':<12}: {rmse_X:.4f}")
    print(f"  {'RMSE(Y)':<12}: {rmse_Y:.4f}")
    print(f"  {'w0':<12}: {w0_est:.4f}  (paper: ~= -1.18)")
    print(f"  {'w':<12}: {w_est:.4f}  (paper: ~=  1.72)")
    print(f"  {'Best logL':<12}: {best_overall_logL:.2f}")

    # ── Visualisation ─────────────────────────────────
    print("\n[4] Plotting F heatmap...")
    save_path = OUTPUT_DIR / "paper_experiment_real_F.png"
    plot_F_heatmap(F_est, WINE_FEATURE_NAMES, save_path)

    # ── Summary table: F values ───────────────────────
    print("\n[5] Estimated F (loading matrix):")
    header = f"{'Feature':<25} " + "".join(f"{'z'+str(i+1):>8}" for i in range(k))
    print(header)
    print("-" * len(header))
    for i, name in enumerate(WINE_FEATURE_NAMES):
        row = f"{name:<25} " + "".join(f"{F_est[i, j]:8.4f}" for j in range(k))
        print(row)

    # ── Save CSV ──────────────────────────────────────
    print("\n[6] Saving CSV...")
    import pandas as pd
    rows = [{'metric': 'RMSE_X', 'value': rmse_X},
            {'metric': 'RMSE_Y', 'value': rmse_Y},
            {'metric': 'w0', 'value': w0_est},
            {'metric': 'w', 'value': w_est},
            {'metric': 'log_likelihood', 'value': best_overall_logL}]
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(OUTPUT_DIR / "results_real_wine.csv", index=False)
    print(f"  CSV saved: {OUTPUT_DIR / 'results_real_wine.csv'}")

    print("\n" + "=" * 60)
    print("Real Data Experiment completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
