"""
Paper Experiment 3: Effect of n and d on estimation accuracy.

Case 1: Varying n with fixed d=15, k*=3
Case 2: Varying d with fixed n=150, k*=3

Evaluates RMSE of parameter estimates and Q function values.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple
import time

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_generator import set_true_params
from model import LatentStructuralModel


def calc_rmse(true: np.ndarray, est: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((true - est) ** 2))


def calc_correlation_matrix(A: np.ndarray, B: np.ndarray) -> float:
    """Calculate correlation between Gram matrices."""
    A_gram = A @ A.T
    B_gram = B @ B.T
    a_flat = A_gram.flatten()
    b_flat = B_gram.flatten()
    return np.corrcoef(a_flat, b_flat)[0, 1]


def procrustes_rotation(F_est: np.ndarray, F_true: np.ndarray):
    """
    Solve orthogonal Procrustes: find R s.t. F_est @ R ~= F_true.
    Uses min(k_est, k_true) columns. Returns R and k_min.
    """
    k_min = min(F_est.shape[1], F_true.shape[1])
    M = F_est[:, :k_min].T @ F_true[:, :k_min]
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    return R, k_min


def calc_log_likelihood(X, Y, Z, F, sigma, var_z, w0, w):
    """Calculate complete-data log-likelihood."""
    n, k = Z.shape
    d = X.shape[1]

    # ln p(Z)
    ln_p_Z = (
        -(n * k / 2) * np.log(2 * np.pi)
        - (n * k / 2) * np.log(var_z)
        - (1 / (2 * var_z)) * np.sum(Z ** 2)
    )

    # ln p(X|Z)
    residual = X - Z @ F.T
    sigma_diag = np.diag(sigma)
    ln_p_X = (
        -(n * d / 2) * np.log(2 * np.pi)
        - (n / 2) * np.sum(np.log(sigma_diag))
        - 0.5 * np.sum(residual ** 2 / sigma_diag)
    )

    # ln p(Y|Z)
    ZZT = Z @ Z.T
    logits = w0 + w * ZZT
    S = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    S = np.clip(S, 1e-10, 1 - 1e-10)

    ln_p_Y = Y * np.log(S) + (1 - Y) * np.log(1 - S)
    np.fill_diagonal(ln_p_Y, 0.0)
    ln_p_Y = 0.5 * np.sum(ln_p_Y)

    return ln_p_Z + ln_p_X + ln_p_Y


def run_em(n, d, k, X, Y, true_params, num_iter=30, L=10, init_seed=42):
    """
    Run EM algorithm and return final metrics.

    Returns
    -------
    dict
        Dictionary with Q, RMSE values, and estimated parameters.
    """
    model = LatentStructuralModel(n=n, d=d, k=k, L=L)
    model.initialize_params(true_params=true_params, seed=init_seed)

    # Informed initialization
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

        # Scale Z
        Z_samples = model.scale_Z(Z_samples)
        Z = Z_samples[:, :, -1].copy()

        # M-Step
        F = model.calc_F(X, Z_samples)
        sigma = model.calc_sigma(X, Z_samples, F)
        w0 = model.calc_w0(Y, Z_samples, w0, w, max_iter=50)
        w = model.calc_w(Y, Z_samples, w0, w, max_iter=50)

        # Track best
        log_likelihood = calc_log_likelihood(X, Y, Z, F, sigma, var_z, w0, w)
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_state = {
                'Z': Z.copy(), 'F': F.copy(), 'sigma': sigma.copy(),
                'w0': w0, 'w': w
            }

        if iteration % 10 == 0:
            print(f"      Iter {iteration}: logL = {log_likelihood:.1f}")

    # Compute metrics using best state
    Z_est = best_state['Z']
    F_est = best_state['F']
    sigma_est = best_state['sigma']
    w0_est = best_state['w0']
    w_est = best_state['w']

    # Y prediction RMSE
    ZZT = Z_est @ Z_est.T
    logits_est = w0_est + w_est * ZZT
    Y_pred = 1.0 / (1.0 + np.exp(-np.clip(logits_est, -500, 500)))
    np.fill_diagonal(Y_pred, 0.0)

    # X reconstruction RMSE
    X_recon = Z_est @ F_est.T

    # Procrustes-aligned RMSE for F and Z
    # true_params here is the one passed for THIS specific (n, d) iteration —
    # so F_true/Z_true are always the correct ground truth for this run.
    R, k_min = procrustes_rotation(F_est, true_params['F'])
    F_rot = F_est[:, :k_min] @ R
    Z_rot = Z_est[:, :k_min] @ R
    F_true_sub = true_params['F'][:, :k_min]
    Z_true_sub = true_params['Z'][:, :k_min]

    metrics = {
        'Q': best_log_likelihood,
        'rmse_X': calc_rmse(true_params['X'], X_recon),
        'rmse_Y': calc_rmse(true_params['Y'], Y_pred),
        'rmse_w0': np.abs(true_params['w0'] - w0_est),
        'rmse_w': np.abs(true_params['w'] - w_est),
        'rmse_sigma': calc_rmse(np.diag(true_params['sigma']), np.diag(sigma_est)),
        'rmse_F_rot': calc_rmse(F_true_sub, F_rot),
        'rmse_Z_rot': calc_rmse(Z_true_sub, Z_rot),
    }

    return metrics


class Experiment3Runner:
    """Run Experiment 3: Effect of n and d on estimation accuracy."""

    def __init__(
        self,
        k_true: int = 3,
        L: int = 10,
        num_iter: int = 30,
        n_trials: int = 2,
        data_seed: int = 1980,
        output_dir: str = "results"
    ):
        self.k_true = k_true
        self.L = L
        self.num_iter = num_iter
        self.n_trials = n_trials
        self.data_seed = data_seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_case1: Dict[int, Dict[str, float]] = {}  # n -> metrics
        self.results_case2: Dict[int, Dict[str, float]] = {}  # d -> metrics

    def run_case1(self, n_list: List[int] = None, d_fixed: int = 15):
        """Case 1: Varying n with fixed d."""
        if n_list is None:
            n_list = [50, 100, 150, 200, 250, 300]

        print("=" * 60)
        print(f"Case 1: Varying n (d={d_fixed}, k*={self.k_true})")
        print("=" * 60)

        for n in n_list:
            print(f"\n  n = {n}")

            true_params = set_true_params(
                n=n, d=d_fixed, k=self.k_true,
                seed=self.data_seed
            )
            X = true_params['X']
            Y = true_params['Y']

            print(f"    True w0={true_params['w0']:.4f}, w={true_params['w']:.4f}, "
                  f"Y density={np.mean(Y):.4f}")

            best_metrics = None
            best_Q = -np.inf

            for trial in range(self.n_trials):
                print(f"    Trial {trial + 1}/{self.n_trials}...")
                start_time = time.time()

                init_seed = 42 + trial * 1000
                metrics = run_em(
                    n, d_fixed, self.k_true, X, Y, true_params,
                    num_iter=self.num_iter, L=self.L, init_seed=init_seed
                )
                elapsed = time.time() - start_time

                print(f"      Q={metrics['Q']:.1f}, RMSE(X)={metrics['rmse_X']:.4f}, "
                      f"RMSE(Y)={metrics['rmse_Y']:.4f}, time={elapsed:.1f}s")

                if metrics['Q'] > best_Q:
                    best_Q = metrics['Q']
                    best_metrics = metrics

            self.results_case1[n] = best_metrics

    def run_case2(self, d_list: List[int] = None, n_fixed: int = 150):
        """Case 2: Varying d with fixed n."""
        if d_list is None:
            d_list = [5, 10, 15, 20, 25, 30]

        print("\n" + "=" * 60)
        print(f"Case 2: Varying d (n={n_fixed}, k*={self.k_true})")
        print("=" * 60)

        for d in d_list:
            print(f"\n  d = {d}")

            true_params = set_true_params(
                n=n_fixed, d=d, k=self.k_true,
                seed=self.data_seed
            )
            X = true_params['X']
            Y = true_params['Y']

            print(f"    True w0={true_params['w0']:.4f}, w={true_params['w']:.4f}, "
                  f"Y density={np.mean(Y):.4f}")

            best_metrics = None
            best_Q = -np.inf

            for trial in range(self.n_trials):
                print(f"    Trial {trial + 1}/{self.n_trials}...")
                start_time = time.time()

                init_seed = 42 + trial * 1000
                metrics = run_em(
                    n_fixed, d, self.k_true, X, Y, true_params,
                    num_iter=self.num_iter, L=self.L, init_seed=init_seed
                )
                elapsed = time.time() - start_time

                print(f"      Q={metrics['Q']:.1f}, RMSE(X)={metrics['rmse_X']:.4f}, "
                      f"RMSE(Y)={metrics['rmse_Y']:.4f}, time={elapsed:.1f}s")

                if metrics['Q'] > best_Q:
                    best_Q = metrics['Q']
                    best_metrics = metrics

            self.results_case2[d] = best_metrics

    def plot_case1(self, save: bool = True):
        """Plot Case 1 results: varying n."""
        if not self.results_case1:
            print("No Case 1 results to plot.")
            return

        n_vals = sorted(self.results_case1.keys())
        metrics_keys = [
            ('rmse_X', 'RMSE(X)', 'tab:blue'),
            ('rmse_Y', 'RMSE(Y)', 'tab:orange'),
            ('rmse_w0', '|error(w0)|', 'tab:green'),
            ('rmse_w', '|error(w)|', 'tab:red'),
            ('rmse_F_rot', 'RMSE(F) aligned', 'tab:purple'),
            ('rmse_Z_rot', 'RMSE(Z) aligned', 'tab:brown'),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Experiment 3 - Case 1: Varying n (d=15, k*={self.k_true})',
                     fontsize=14, fontweight='bold')

        for idx, (key, label, color) in enumerate(metrics_keys):
            ax = axes[idx // 3, idx % 3]
            values = [self.results_case1[n][key] for n in n_vals]
            ax.plot(n_vals, values, 'o-', color=color, linewidth=2, markersize=8)
            ax.set_xlabel('n (number of data points)')
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "paper_experiment_3_n.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filepath}")
        plt.close()

    def plot_case2(self, save: bool = True):
        """Plot Case 2 results: varying d."""
        if not self.results_case2:
            print("No Case 2 results to plot.")
            return

        d_vals = sorted(self.results_case2.keys())
        metrics_keys = [
            ('rmse_X', 'RMSE(X)', 'tab:blue'),
            ('rmse_Y', 'RMSE(Y)', 'tab:orange'),
            ('rmse_w0', '|error(w0)|', 'tab:green'),
            ('rmse_w', '|error(w)|', 'tab:red'),
            ('rmse_F_rot', 'RMSE(F) aligned', 'tab:purple'),
            ('rmse_Z_rot', 'RMSE(Z) aligned', 'tab:brown'),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Experiment 3 - Case 2: Varying d (n=150, k*={self.k_true})',
                     fontsize=14, fontweight='bold')

        for idx, (key, label, color) in enumerate(metrics_keys):
            ax = axes[idx // 3, idx % 3]
            values = [self.results_case2[d][key] for d in d_vals]
            ax.plot(d_vals, values, 'o-', color=color, linewidth=2, markersize=8)
            ax.set_xlabel('d (dimensionality)')
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "paper_experiment_3_d.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filepath}")
        plt.close()

    def print_results_table(self):
        """Print results in tabular format."""
        if self.results_case1:
            print("\n" + "=" * 90)
            print("Case 1: Varying n (d=15, k*=3)")
            print("=" * 90)
            header = (f"{'n':>6} | {'RMSE(X)':>9} | {'RMSE(Y)':>9} | "
                      f"{'|err w0|':>9} | {'|err w|':>9} | "
                      f"{'RMSE(F)':>9} | {'RMSE(Z)':>9} | {'RMSE(Sig)':>9} | {'Q':>12}")
            print(header)
            print("-" * len(header))
            for n in sorted(self.results_case1.keys()):
                m = self.results_case1[n]
                print(f"{n:6d} | {m['rmse_X']:9.4f} | {m['rmse_Y']:9.4f} | "
                      f"{m['rmse_w0']:9.4f} | {m['rmse_w']:9.4f} | "
                      f"{m['rmse_F_rot']:9.4f} | {m['rmse_Z_rot']:9.4f} | "
                      f"{m['rmse_sigma']:9.4f} | {m['Q']:12.1f}")

        if self.results_case2:
            print("\n" + "=" * 90)
            print("Case 2: Varying d (n=150, k*=3)")
            print("=" * 90)
            header = (f"{'d':>6} | {'RMSE(X)':>9} | {'RMSE(Y)':>9} | "
                      f"{'|err w0|':>9} | {'|err w|':>9} | "
                      f"{'RMSE(F)':>9} | {'RMSE(Z)':>9} | {'RMSE(Sig)':>9} | {'Q':>12}")
            print(header)
            print("-" * len(header))
            for d in sorted(self.results_case2.keys()):
                m = self.results_case2[d]
                print(f"{d:6d} | {m['rmse_X']:9.4f} | {m['rmse_Y']:9.4f} | "
                      f"{m['rmse_w0']:9.4f} | {m['rmse_w']:9.4f} | "
                      f"{m['rmse_F_rot']:9.4f} | {m['rmse_Z_rot']:9.4f} | "
                      f"{m['rmse_sigma']:9.4f} | {m['Q']:12.1f}")

    def save_csv(self):
        """Save Case 1 and Case 2 results to separate CSV files."""
        import pandas as pd

        if self.results_case1:
            rows = [{'n': n, 'k_true': self.k_true, **self.results_case1[n]}
                    for n in sorted(self.results_case1.keys())]
            df = pd.DataFrame(rows)
            p = self.output_dir / "results_exp3_case1_n.csv"
            df.to_csv(p, index=False)
            print(f"  CSV saved: {p}")

        if self.results_case2:
            rows = [{'d': d, 'k_true': self.k_true, **self.results_case2[d]}
                    for d in sorted(self.results_case2.keys())]
            df = pd.DataFrame(rows)
            p = self.output_dir / "results_exp3_case2_d.csv"
            df.to_csv(p, index=False)
            print(f"  CSV saved: {p}")

    def save_csv_aligned(self):
        """Save results with Procrustes-aligned metrics to new CSV files."""
        import pandas as pd

        aligned_cols = [
            'Q', 'rmse_X', 'rmse_Y', 'rmse_w0', 'rmse_w',
            'rmse_F_rot', 'rmse_Z_rot', 'rmse_sigma',
        ]

        if self.results_case1:
            rows = []
            for n in sorted(self.results_case1.keys()):
                m = self.results_case1[n]
                rows.append({'n': n, 'k_true': self.k_true,
                             **{c: m[c] for c in aligned_cols}})
            df = pd.DataFrame(rows)
            p = self.output_dir / "results_exp3_case1_n_aligned.csv"
            df.to_csv(p, index=False)
            print(f"  CSV saved: {p}")

        if self.results_case2:
            rows = []
            for d in sorted(self.results_case2.keys()):
                m = self.results_case2[d]
                rows.append({'d': d, 'k_true': self.k_true,
                             **{c: m[c] for c in aligned_cols}})
            df = pd.DataFrame(rows)
            p = self.output_dir / "results_exp3_case2_d_aligned.csv"
            df.to_csv(p, index=False)
            print(f"  CSV saved: {p}")


def main():
    """Run Experiment 3."""
    print("=" * 60)
    print("Paper Experiment 3: Effect of n and d")
    print("=" * 60)

    runner = Experiment3Runner(
        k_true=3,
        L=10,
        num_iter=30,
        n_trials=2,
        data_seed=1980,
        output_dir="c:/研究2/results"
    )

    # Case 1: Varying n
    runner.run_case1(
        n_list=[50, 100, 150, 200, 250, 300],
        d_fixed=15
    )

    # Case 2: Varying d
    runner.run_case2(
        d_list=[5, 10, 15, 20, 25, 30],
        n_fixed=150
    )

    # Plot results
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)
    runner.plot_case1(save=True)
    runner.plot_case2(save=True)

    # Print tables
    runner.print_results_table()

    # Save CSV
    runner.save_csv()
    runner.save_csv_aligned()

    print("\n" + "=" * 60)
    print("Experiment 3 completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
