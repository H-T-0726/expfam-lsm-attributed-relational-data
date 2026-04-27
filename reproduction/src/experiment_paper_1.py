"""
Paper Experiment 1: Parameter estimation when k is varied.

Reproduces Experiment 1 from:
Mikawa et al., "A study on latent structural models for binary
relational data with attribute information", NOLTA 2024.

Setting:
- True latent dimension k* = 3
- Data: n=150, d=15
- Vary estimated k from 1 to 6
- Record Q function and RMSE for each k
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
import time

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_generator import set_true_params
from model import LatentStructuralModel


def calc_rmse(true: np.ndarray, est: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((true - est) ** 2))


def procrustes_rotation(F_est: np.ndarray, F_true: np.ndarray):
    """
    Solve orthogonal Procrustes problem: find R s.t. F_est @ R ~= F_true.
    Uses min(k_est, k_true) columns for cross-dimension comparison.

    Returns R (k_min x k_min) and k_min.
    """
    k_min = min(F_est.shape[1], F_true.shape[1])
    M = F_est[:, :k_min].T @ F_true[:, :k_min]   # k_min x k_min
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    return R, k_min


def calc_correlation_matrix(A: np.ndarray, B: np.ndarray) -> float:
    """Calculate correlation using Gram matrices."""
    A_gram = A @ A.T
    B_gram = B @ B.T
    a_flat = A_gram.flatten()
    b_flat = B_gram.flatten()
    corr = np.corrcoef(a_flat, b_flat)[0, 1]
    return corr


class Experiment1Runner:
    """
    Run Experiment 1: Varying k while keeping true k* = 3.
    """

    def __init__(
        self,
        n: int = 150,
        d: int = 15,
        k_true: int = 3,
        k_range: List[int] = None,
        L: int = 10,
        num_iter: int = 10,
        num_trials: int = 3,
        data_seed: int = 1980,
        output_dir: str = "results"
    ):
        """
        Initialize Experiment 1 runner.

        Parameters
        ----------
        n : int
            Number of data points
        d : int
            Dimensionality of observed data
        k_true : int
            True latent dimension for data generation
        k_range : List[int]
            List of k values to test
        L : int
            Number of Monte Carlo samples
        num_iter : int
            Number of EM iterations per trial
        num_trials : int
            Number of trials with different initializations
        data_seed : int
            Random seed for data generation
        output_dir : str
            Directory to save results
        """
        self.n = n
        self.d = d
        self.k_true = k_true
        self.k_range = k_range or [1, 2, 3, 4, 5, 6]
        self.L = L
        self.num_iter = num_iter
        self.num_trials = num_trials
        self.data_seed = data_seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results: Dict[int, List[Dict[str, Any]]] = {}
        self.true_params: Dict[str, Any] = None
        self.X: np.ndarray = None
        self.Y: np.ndarray = None

    def generate_data(self):
        """Generate data with true k*."""
        print("=" * 60)
        print(f"Generating Data with k* = {self.k_true}")
        print("=" * 60)

        self.true_params = set_true_params(
            n=self.n, d=self.d, k=self.k_true, seed=self.data_seed
        )
        self.X = self.true_params['X']
        self.Y = self.true_params['Y']

        print(f"  n={self.n}, d={self.d}, k*={self.k_true}")
        print(f"  True w0: {self.true_params['w0']:.4f}")
        print(f"  True w: {self.true_params['w']:.4f}")
        print(f"  Y density: {np.mean(self.Y):.4f}")

    def run_single_trial(
        self,
        k: int,
        trial: int,
        init_seed: int
    ) -> Dict[str, Any]:
        """Run a single trial for given k."""
        # Initialize model
        model = LatentStructuralModel(n=self.n, d=self.d, k=k, L=self.L)
        model.initialize_params(true_params=self.true_params, seed=init_seed)

        # Use informed initialization for w0, w
        y_density = np.mean(self.Y)
        w0_init = np.log(y_density / (1 - y_density + 1e-10))
        model.params['w0'] = w0_init
        model.params['w'] = 0.5

        rng = np.random.default_rng(init_seed)

        # Storage for this trial
        Q_history = []
        best_metrics = None
        best_Q = -np.inf

        Z = model.params['Z'].copy()
        F = model.params['F'].copy()
        sigma = model.params['sigma'].copy()
        w0 = model.params['w0']
        w = model.params['w']
        var_z = model.params['var_z']

        for iteration in range(1, self.num_iter + 1):
            # E-Step: Generate Z samples
            Z_samples = np.zeros((self.n, k, self.L))

            for l in range(self.L):
                model.params['Z'] = Z.copy()
                model.params['F'] = F
                model.params['sigma'] = sigma
                model.params['w0'] = w0
                model.params['w'] = w

                Z_new = model.calc_eta_newton(
                    self.X, self.Y, rng=rng, max_iter=10, alpha=0.5
                )
                Z_samples[:, :, l] = Z_new
                Z = Z_new.copy()

            # Scale Z
            Z_samples = model.scale_Z(Z_samples)
            Z = Z_samples[:, :, -1].copy()

            # M-Step
            F = model.calc_F(self.X, Z_samples)
            sigma = model.calc_sigma(self.X, Z_samples, F)
            w0 = model.calc_w0(self.Y, Z_samples, w0, w, max_iter=50)
            w = model.calc_w(self.Y, Z_samples, w0, w, max_iter=50)

            # Calculate Q function
            Q = self._calc_Q(Z_samples, F, sigma, var_z, w0, w)
            Q_history.append(Q)

            # Calculate metrics
            metrics = self._calc_metrics(Z_samples, F, sigma, w0, w, Q)

            if Q > best_Q:
                best_Q = Q
                best_metrics = metrics.copy()

        # Store final results
        best_metrics['Q_history'] = Q_history
        best_metrics['trial'] = trial

        return best_metrics

    def _calc_Q(
        self,
        Z_samples: np.ndarray,
        F: np.ndarray,
        sigma: np.ndarray,
        var_z: float,
        w0: float,
        w: float
    ) -> float:
        """Calculate Q function."""
        n, k, L = Z_samples.shape
        d = self.X.shape[1]
        Q = 0.0

        for l in range(L):
            Z_l = Z_samples[:, :, l]

            # ln p(Z)
            ln_p_Z = (
                -(n * k / 2) * np.log(2 * np.pi)
                - (n / 2) * k * np.log(var_z)
                - (1 / (2 * var_z)) * np.sum(Z_l ** 2)
            )

            # ln p(X|Z)
            residual = self.X - Z_l @ F.T
            sigma_diag = np.diag(sigma)
            ln_p_X = (
                -(n * d / 2) * np.log(2 * np.pi)
                - (n / 2) * np.sum(np.log(sigma_diag))
                - 0.5 * np.sum(residual ** 2 / sigma_diag)
            )

            # ln p(Y|Z)
            ZZT = Z_l @ Z_l.T
            logits = w0 + w * ZZT
            S = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
            S = np.clip(S, 1e-10, 1 - 1e-10)

            ln_p_Y = self.Y * np.log(S) + (1 - self.Y) * np.log(1 - S)
            np.fill_diagonal(ln_p_Y, 0.0)
            ln_p_Y = 0.5 * np.sum(ln_p_Y)

            Q += ln_p_Z + ln_p_X + ln_p_Y

        return Q / L

    def _calc_metrics(
        self,
        Z_samples: np.ndarray,
        F: np.ndarray,
        sigma: np.ndarray,
        w0: float,
        w: float,
        Q: float
    ) -> Dict[str, float]:
        """Calculate metrics."""
        Z_est = Z_samples[:, :, -1]
        k = Z_est.shape[1]

        metrics = {
            'Q': Q,
            'rmse_w0': np.abs(self.true_params['w0'] - w0),
            'rmse_w': np.abs(self.true_params['w'] - w),
            'rmse_sigma': calc_rmse(
                np.diag(self.true_params['sigma']), np.diag(sigma)
            ),
            'w0': w0,
            'w': w,
        }

        # RMSE for Y
        ZZT = Z_est @ Z_est.T
        logits = w0 + w * ZZT
        Y_pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        np.fill_diagonal(Y_pred, 0.0)
        metrics['rmse_Y'] = calc_rmse(self.true_params['Y'], Y_pred)

        # MAE for Y (as in paper)
        Y_diff = np.abs(self.true_params['Y'] - Y_pred)
        np.fill_diagonal(Y_diff, 0.0)
        n_pairs = self.n * (self.n - 1)
        metrics['mae_Y'] = np.sum(Y_diff) / n_pairs

        # RMSE for X
        X_recon = Z_est @ F.T
        metrics['rmse_X'] = calc_rmse(self.true_params['X'], X_recon)

        # ── Procrustes-aligned RMSE for F and Z ──────────────────
        # Use min(k, k_true) columns so all k values are comparable
        R, k_min = procrustes_rotation(F, self.true_params['F'])
        F_rot = F[:, :k_min] @ R
        Z_rot = Z_est[:, :k_min] @ R

        F_true_sub = self.true_params['F'][:, :k_min]
        Z_true_sub = self.true_params['Z'][:, :k_min]

        metrics['rmse_F'] = calc_rmse(F_true_sub, F_rot)
        metrics['rmse_Z'] = calc_rmse(Z_true_sub, Z_rot)

        # Gram-matrix correlation (kept for comparison)
        if k == self.k_true:
            metrics['corr_Z'] = calc_correlation_matrix(
                Z_est, self.true_params['Z']
            )
            metrics['corr_F'] = calc_correlation_matrix(
                F, self.true_params['F']
            )
        else:
            Z_norms_est = np.linalg.norm(Z_est, axis=1)
            Z_norms_true = np.linalg.norm(self.true_params['Z'], axis=1)
            metrics['corr_Z'] = np.corrcoef(Z_norms_est, Z_norms_true)[0, 1]
            F_norms_est = np.linalg.norm(F, axis=1)
            F_norms_true = np.linalg.norm(self.true_params['F'], axis=1)
            metrics['corr_F'] = np.corrcoef(F_norms_est, F_norms_true)[0, 1]

        return metrics

    def run_experiment(self):
        """Run the full experiment."""
        print("\n" + "=" * 60)
        print("Running Experiment 1: Varying k")
        print("=" * 60)

        total_runs = len(self.k_range) * self.num_trials
        run_count = 0

        for k in self.k_range:
            print(f"\n--- k = {k} ---")
            self.results[k] = []

            for trial in range(1, self.num_trials + 1):
                run_count += 1
                init_seed = 42 + trial * 100

                print(f"  Trial {trial}/{self.num_trials} "
                      f"(overall: {run_count}/{total_runs})...", end=" ")

                start_time = time.time()
                metrics = self.run_single_trial(k, trial, init_seed)
                elapsed = time.time() - start_time

                self.results[k].append(metrics)
                print(f"Q={metrics['Q']:.1f}, "
                      f"RMSE_Y={metrics['rmse_Y']:.4f}, "
                      f"time={elapsed:.1f}s")

    def aggregate_results(self) -> Dict[str, Dict[int, Any]]:
        """Aggregate results across trials."""
        aggregated = {
            'Q_mean': {}, 'Q_std': {}, 'Q_best': {},
            'mae_Y_mean': {}, 'mae_Y_std': {},
            'rmse_Y_mean': {}, 'rmse_Y_std': {},
            'rmse_X_mean': {}, 'rmse_X_std': {},
            'rmse_F_mean': {}, 'rmse_F_std': {},
            'rmse_Z_mean': {}, 'rmse_Z_std': {},
            'rmse_sigma_mean': {},
            'rmse_w0_mean': {},
            'rmse_w_mean': {},
            'corr_Z_mean': {},
            'corr_F_mean': {},
        }

        for k in self.k_range:
            trials = self.results[k]

            def _m(key): return [t[key] for t in trials]

            Q_values = _m('Q')
            aggregated['Q_mean'][k] = np.mean(Q_values)
            aggregated['Q_std'][k] = np.std(Q_values)
            aggregated['Q_best'][k] = np.max(Q_values)
            aggregated['mae_Y_mean'][k] = np.mean(_m('mae_Y'))
            aggregated['mae_Y_std'][k] = np.std(_m('mae_Y'))
            aggregated['rmse_Y_mean'][k] = np.mean(_m('rmse_Y'))
            aggregated['rmse_Y_std'][k] = np.std(_m('rmse_Y'))
            aggregated['rmse_X_mean'][k] = np.mean(_m('rmse_X'))
            aggregated['rmse_X_std'][k] = np.std(_m('rmse_X'))
            aggregated['rmse_F_mean'][k] = np.mean(_m('rmse_F'))
            aggregated['rmse_F_std'][k] = np.std(_m('rmse_F'))
            aggregated['rmse_Z_mean'][k] = np.mean(_m('rmse_Z'))
            aggregated['rmse_Z_std'][k] = np.std(_m('rmse_Z'))
            aggregated['rmse_sigma_mean'][k] = np.mean(_m('rmse_sigma'))
            aggregated['rmse_w0_mean'][k] = np.mean(_m('rmse_w0'))
            aggregated['rmse_w_mean'][k] = np.mean(_m('rmse_w'))
            aggregated['corr_Z_mean'][k] = np.nanmean(_m('corr_Z'))
            aggregated['corr_F_mean'][k] = np.nanmean(_m('corr_F'))

        return aggregated

    def plot_results(self, save: bool = True):
        """Generate plots similar to paper Figure 2."""
        print("\n" + "=" * 60)
        print("Generating Plots")
        print("=" * 60)

        agg = self.aggregate_results()

        k_values = list(self.k_range)

        # Create figure with subplots (similar to paper Fig. 2)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Q function
        ax = axes[0, 0]
        Q_mean = [agg['Q_mean'][k] for k in k_values]
        Q_std = [agg['Q_std'][k] for k in k_values]
        ax.errorbar(k_values, Q_mean, yerr=Q_std, fmt='o-', capsize=5,
                    linewidth=2, markersize=8)
        ax.axvline(x=self.k_true, color='r', linestyle='--', alpha=0.7,
                   label=f'True k*={self.k_true}')
        ax.set_xlabel('k (latent dimension)', fontsize=12)
        ax.set_ylabel('Q function', fontsize=12)
        ax.set_title('(a) Q function', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: RMSE of Y
        ax = axes[0, 1]
        rmse_Y_mean = [agg['mae_Y_mean'][k] for k in k_values]
        rmse_Y_std = [agg['mae_Y_std'][k] for k in k_values]
        ax.errorbar(k_values, rmse_Y_mean, yerr=rmse_Y_std, fmt='o-', capsize=5,
                    linewidth=2, markersize=8, color='green')
        ax.axvline(x=self.k_true, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('k', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title('(c) MAE between $y_{ij}$ and $\\hat{y}_{ij}$', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot 3: RMSE of F
        ax = axes[0, 2]
        corr_F = [agg['corr_F_mean'][k] for k in k_values]
        ax.plot(k_values, corr_F, 'o-', linewidth=2, markersize=8, color='purple')
        ax.axvline(x=self.k_true, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('k', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_title('(d) Correlation of F', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot 4: RMSE of w0
        ax = axes[1, 0]
        rmse_w0 = [agg['rmse_w0_mean'][k] for k in k_values]
        ax.plot(k_values, rmse_w0, 'o-', linewidth=2, markersize=8, color='orange')
        ax.axvline(x=self.k_true, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('k', fontsize=12)
        ax.set_ylabel('|w0 - w0*|', fontsize=12)
        ax.set_title('(e) Error of $w_0$', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot 5: RMSE of w
        ax = axes[1, 1]
        rmse_w = [agg['rmse_w_mean'][k] for k in k_values]
        ax.plot(k_values, rmse_w, 'o-', linewidth=2, markersize=8, color='brown')
        ax.axvline(x=self.k_true, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('k', fontsize=12)
        ax.set_ylabel('|w - w*|', fontsize=12)
        ax.set_title('(f) Error of $w$', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot 6: RMSE of X
        ax = axes[1, 2]
        rmse_X = [agg['rmse_X_mean'][k] for k in k_values]
        rmse_X_std = [agg['rmse_X_std'][k] for k in k_values]
        ax.errorbar(k_values, rmse_X, yerr=rmse_X_std, fmt='o-', capsize=5,
                    linewidth=2, markersize=8, color='magenta')
        ax.axvline(x=self.k_true, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('k', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('RMSE of X reconstruction', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'Experiment 1: Varying k (True k*={self.k_true})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            filepath = self.output_dir / "paper_experiment_1.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filepath}")

        plt.close()

    def print_results_table(self):
        """Print results table similar to paper Table II."""
        print("\n" + "=" * 60)
        print("Results Table (similar to Paper Table II)")
        print("=" * 60)

        agg = self.aggregate_results()

        print(f"\n{'k':>4} | {'Q':>12} | {'MAE_Y':>8} | {'RMSE_X':>8} | "
              f"{'|w0-w0*|':>8} | {'|w-w*|':>8} | {'Corr_F':>8}")
        print("-" * 75)

        for k in self.k_range:
            print(f"{k:>4} | {agg['Q_mean'][k]:>12.1f} | "
                  f"{agg['mae_Y_mean'][k]:>8.4f} | "
                  f"{agg['rmse_X_mean'][k]:>8.4f} | "
                  f"{agg['rmse_w0_mean'][k]:>8.4f} | "
                  f"{agg['rmse_w_mean'][k]:>8.4f} | "
                  f"{agg['corr_F_mean'][k]:>8.4f}")

        print("-" * 75)

        print(f"\n{'k':>4} | {'RMSE_F':>8} | {'RMSE_Z':>8} | {'RMSE_Sig':>8}"
              f"  (Procrustes-aligned)")
        print("-" * 40)

        for k in self.k_range:
            print(f"{k:>4} | {agg['rmse_F_mean'][k]:>8.4f} | "
                  f"{agg['rmse_Z_mean'][k]:>8.4f} | "
                  f"{agg['rmse_sigma_mean'][k]:>8.4f}")

        print("-" * 40)

        best_k_Q = max(self.k_range, key=lambda k: agg['Q_mean'][k])
        best_k_Y = min(self.k_range, key=lambda k: agg['mae_Y_mean'][k])
        best_k_F = min(self.k_range, key=lambda k: agg['rmse_F_mean'][k])

        print(f"\nBest k by Q function: k={best_k_Q}")
        print(f"Best k by MAE(Y): k={best_k_Y}")
        print(f"Best k by RMSE(F) [aligned]: k={best_k_F}")
        print(f"True k*: k={self.k_true}")

    def save_csv(self, filepath: str = None):
        """Save aggregated results to CSV (base metrics)."""
        import pandas as pd
        if filepath is None:
            filepath = self.output_dir / "results_exp1_varying_k.csv"

        agg = self.aggregate_results()
        rows = []
        for k in self.k_range:
            rows.append({
                'k': k, 'k_true': self.k_true,
                'Q_mean': agg['Q_mean'][k], 'Q_std': agg['Q_std'][k],
                'Q_best': agg['Q_best'][k],
                'mae_Y_mean': agg['mae_Y_mean'][k],
                'mae_Y_std': agg['mae_Y_std'][k],
                'rmse_X_mean': agg['rmse_X_mean'][k],
                'rmse_X_std': agg['rmse_X_std'][k],
                'rmse_w0_mean': agg['rmse_w0_mean'][k],
                'rmse_w_mean': agg['rmse_w_mean'][k],
                'corr_F_mean': agg['corr_F_mean'][k],
                'corr_Z_mean': agg['corr_Z_mean'][k],
            })
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"  CSV saved: {filepath}")

    def save_csv_aligned(self, filepath: str = None):
        """Save aggregated results with Procrustes-aligned RMSE to CSV."""
        import pandas as pd
        if filepath is None:
            filepath = self.output_dir / "results_exp1_varying_k_aligned.csv"

        agg = self.aggregate_results()
        rows = []
        for k in self.k_range:
            rows.append({
                'k': k, 'k_true': self.k_true,
                'Q_mean': agg['Q_mean'][k], 'Q_std': agg['Q_std'][k],
                'Q_best': agg['Q_best'][k],
                'mae_Y_mean': agg['mae_Y_mean'][k],
                'mae_Y_std': agg['mae_Y_std'][k],
                'rmse_X_mean': agg['rmse_X_mean'][k],
                'rmse_X_std': agg['rmse_X_std'][k],
                'rmse_F_mean': agg['rmse_F_mean'][k],
                'rmse_F_std': agg['rmse_F_std'][k],
                'rmse_Z_mean': agg['rmse_Z_mean'][k],
                'rmse_Z_std': agg['rmse_Z_std'][k],
                'rmse_sigma_mean': agg['rmse_sigma_mean'][k],
                'rmse_w0_mean': agg['rmse_w0_mean'][k],
                'rmse_w_mean': agg['rmse_w_mean'][k],
                'corr_F_mean': agg['corr_F_mean'][k],
                'corr_Z_mean': agg['corr_Z_mean'][k],
            })
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"  CSV saved: {filepath}")


def main():
    """Run Experiment 1."""
    print("=" * 60)
    print("Paper Experiment 1: Varying k")
    print("=" * 60)

    runner = Experiment1Runner(
        n=150,
        d=15,
        k_true=3,
        k_range=[1, 2, 3, 4, 5, 6],
        L=10,
        num_iter=10,
        num_trials=3,  # Reduced for faster execution
        data_seed=1980,
        output_dir="c:/研究2/results"
    )

    # Generate data
    runner.generate_data()

    # Run experiment
    runner.run_experiment()

    # Plot results
    runner.plot_results(save=True)

    # Print table
    runner.print_results_table()

    # Save CSV (base + aligned)
    runner.save_csv()
    runner.save_csv_aligned()

    print("\n" + "=" * 60)
    print("Experiment 1 completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
