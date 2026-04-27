"""
Experiment Runner for Latent Structural Model.

This module implements the full Monte Carlo EM algorithm and
experimental evaluation based on the NOLTA 2024 paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import time

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_generator import set_true_params
from model import LatentStructuralModel


def calc_rmse(true: np.ndarray, est: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((true - est) ** 2))


def calc_correlation_matrix(A: np.ndarray, B: np.ndarray) -> float:
    """
    Calculate correlation between two matrices using their Gram matrices.

    For Z: compare Z @ Z^T (inner products between data points)
    For F: compare F @ F^T (factor covariance structure)
    """
    A_gram = A @ A.T
    B_gram = B @ B.T

    # Flatten and compute correlation
    a_flat = A_gram.flatten()
    b_flat = B_gram.flatten()

    # Pearson correlation
    corr = np.corrcoef(a_flat, b_flat)[0, 1]
    return corr


def rotate_to_match(Z_est: np.ndarray, Z_true: np.ndarray) -> np.ndarray:
    """
    Apply Procrustes rotation to align estimated Z with true Z.

    This accounts for rotational indeterminacy in factor models.
    """
    # Procrustes rotation: find R that minimizes ||Z_est @ R - Z_true||
    U, _, Vt = np.linalg.svd(Z_true.T @ Z_est)
    R = Vt.T @ U.T
    return Z_est @ R


class ExperimentRunner:
    """
    Run experiments for the Latent Structural Model.
    """

    def __init__(
        self,
        n: int = 150,
        d: int = 15,
        k: int = 3,
        L: int = 11,
        num_iter: int = 10,
        seed: int = 1980,
        output_dir: str = "results"
    ):
        """
        Initialize experiment runner.

        Parameters
        ----------
        n : int
            Number of data points
        d : int
            Dimensionality of observed data
        k : int
            Dimensionality of latent space
        L : int
            Number of Monte Carlo samples
        num_iter : int
            Number of EM iterations
        seed : int
            Random seed for data generation
        output_dir : str
            Directory to save results
        """
        self.n = n
        self.d = d
        self.k = k
        self.L = L
        self.num_iter = num_iter
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for results
        self.history: List[Dict[str, Any]] = []
        self.true_params: Optional[Dict[str, Any]] = None
        self.model: Optional[LatentStructuralModel] = None
        self.X: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None

    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate synthetic data."""
        print("=" * 60)
        print("Generating Synthetic Data")
        print("=" * 60)

        self.true_params = set_true_params(
            n=self.n, d=self.d, k=self.k, seed=self.seed
        )
        self.X = self.true_params['X']
        self.Y = self.true_params['Y']

        print(f"  n={self.n}, d={self.d}, k={self.k}")
        print(f"  X shape: {self.X.shape}")
        print(f"  Y shape: {self.Y.shape}")
        print(f"  True w0: {self.true_params['w0']:.4f}")
        print(f"  True w: {self.true_params['w']:.4f}")
        print(f"  Y density: {np.mean(self.Y):.4f}")

        return self.X, self.Y, self.true_params

    def initialize_model(
        self,
        init_seed: int = 42,
        use_informed_init: bool = False
    ) -> LatentStructuralModel:
        """
        Initialize the model.

        Parameters
        ----------
        init_seed : int
            Random seed for initialization
        use_informed_init : bool
            If True, initialize w0 and w closer to reasonable values
            (not true values, but better starting point)
        """
        print("\n" + "=" * 60)
        print("Initializing Model")
        print("=" * 60)

        self.model = LatentStructuralModel(
            n=self.n, d=self.d, k=self.k, L=self.L
        )
        self.model.initialize_params(
            true_params=self.true_params, seed=init_seed
        )

        if use_informed_init:
            # Use more reasonable initial values for w0, w
            # Based on Y density, estimate reasonable w0
            y_density = np.mean(self.Y)
            # sigmoid(w0) ≈ density when w*z^Tz is small
            w0_init = np.log(y_density / (1 - y_density + 1e-10))
            self.model.params['w0'] = w0_init
            # w should be positive for positive correlation
            self.model.params['w'] = 0.5
            print(f"  Using informed initialization")

        print(f"  L (MC samples): {self.L}")
        print(f"  Initial w0: {self.model.params['w0']:.4f}")
        print(f"  Initial w: {self.model.params['w']:.4f}")

        return self.model

    def calc_Q_function(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z_samples: np.ndarray,
        F: np.ndarray,
        sigma: np.ndarray,
        var_z: float,
        w0: float,
        w: float
    ) -> float:
        """
        Calculate the Q function (expected log-likelihood).

        Q = (1/L) Σ_l [ln p(Z^(l)) + ln p(X|Z^(l)) + ln p(Y|Z^(l))]
        """
        n, k, L = Z_samples.shape
        d = X.shape[1]

        Q = 0.0

        for l in range(L):
            Z_l = Z_samples[:, :, l]

            # ln p(Z) = -nk/2 ln(2π) - nk/2 ln(σ²_z) - 1/(2σ²_z) Σ||z_i||²
            # Note: var_z = 1.0 by design, so the second term is 0
            ln_p_Z = (
                -(n * k / 2) * np.log(2 * np.pi)
                - (n * k / 2) * np.log(var_z)
                - (1 / (2 * var_z)) * np.sum(Z_l ** 2)
            )

            # ln p(X|Z) = -nd/2 ln(2π) - n/2 Σ ln(σ²_m) - 1/2 Σ (x - Fz)^T Σ^{-1} (x - Fz)
            residual = X - Z_l @ F.T
            sigma_diag = np.diag(sigma)
            ln_p_X = (
                -(n * d / 2) * np.log(2 * np.pi)
                - (n / 2) * np.sum(np.log(sigma_diag))
                - 0.5 * np.sum(residual ** 2 / sigma_diag)
            )

            # ln p(Y|Z) = 1/2 Σ_{i≠j} [y_ij ln s_ij + (1-y_ij) ln(1-s_ij)]
            ZZT = Z_l @ Z_l.T
            logits = w0 + w * ZZT
            S = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
            S = np.clip(S, 1e-10, 1 - 1e-10)  # Avoid log(0)

            ln_p_Y = Y * np.log(S) + (1 - Y) * np.log(1 - S)
            np.fill_diagonal(ln_p_Y, 0.0)
            ln_p_Y = 0.5 * np.sum(ln_p_Y)

            Q += ln_p_Z + ln_p_X + ln_p_Y

        return Q / L

    def calc_metrics(
        self,
        Z_samples: np.ndarray,
        F: np.ndarray,
        sigma: np.ndarray,
        w0: float,
        w: float,
        Q: float
    ) -> Dict[str, float]:
        """Calculate RMSE and other metrics."""
        # Use last sample of Z
        Z_est = Z_samples[:, :, -1]

        # Rotate Z to match true Z
        Z_rotated = rotate_to_match(Z_est, self.true_params['Z'])

        # RMSE for various parameters
        metrics = {
            'Q': Q,
            'rmse_w0': np.abs(self.true_params['w0'] - w0),
            'rmse_w': np.abs(self.true_params['w'] - w),
            'rmse_sigma': calc_rmse(
                np.diag(self.true_params['sigma']), np.diag(sigma)
            ),
            'corr_Z': calc_correlation_matrix(Z_est, self.true_params['Z']),
            'corr_F': calc_correlation_matrix(F, self.true_params['F']),
        }

        # RMSE for Y prediction
        ZZT = Z_est @ Z_est.T
        logits = w0 + w * ZZT
        Y_pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        np.fill_diagonal(Y_pred, 0.0)
        metrics['rmse_Y'] = calc_rmse(self.true_params['Y'], Y_pred)

        # RMSE for X reconstruction
        X_recon = Z_est @ F.T
        metrics['rmse_X'] = calc_rmse(self.true_params['X'], X_recon)

        return metrics

    def fit(self, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Run the Monte Carlo EM algorithm.

        Returns
        -------
        List[Dict[str, Any]]
            History of metrics at each iteration
        """
        print("\n" + "=" * 60)
        print("Running Monte Carlo EM Algorithm")
        print("=" * 60)

        rng = np.random.default_rng(42)
        self.history = []

        # Get initial parameters
        Z = self.model.params['Z'].copy()
        F = self.model.params['F'].copy()
        sigma = self.model.params['sigma'].copy()
        w0 = self.model.params['w0']
        w = self.model.params['w']
        var_z = self.model.params['var_z']

        print(f"\nStarting EM iterations (num_iter={self.num_iter}, L={self.L})")
        print("-" * 60)

        for iteration in range(1, self.num_iter + 1):
            start_time = time.time()

            # ============ E-Step ============
            # Generate L samples of Z from posterior
            Z_samples = np.zeros((self.n, self.k, self.L))

            for l in range(self.L):
                # Update model params for this iteration
                self.model.params['Z'] = Z.copy()
                self.model.params['F'] = F
                self.model.params['sigma'] = sigma
                self.model.params['w0'] = w0
                self.model.params['w'] = w

                # Sample Z from posterior using Newton + Laplace
                Z_new = self.model.calc_eta_newton(
                    self.X, self.Y, rng=rng, max_iter=10, alpha=0.5
                )
                Z_samples[:, :, l] = Z_new
                Z = Z_new.copy()  # Use for next sample

            # Scale Z samples
            Z_samples = self.model.scale_Z(Z_samples)
            Z = Z_samples[:, :, -1].copy()

            # ============ M-Step ============
            # Update F
            F = self.model.calc_F(self.X, Z_samples)

            # Update sigma
            sigma = self.model.calc_sigma(self.X, Z_samples, F)

            # Update w0
            w0 = self.model.calc_w0(self.Y, Z_samples, w0, w, max_iter=50)

            # Update w
            w = self.model.calc_w(self.Y, Z_samples, w0, w, max_iter=50)

            # ============ Evaluate ============
            # Calculate Q function
            Q = self.calc_Q_function(
                self.X, self.Y, Z_samples, F, sigma, var_z, w0, w
            )

            # Calculate metrics
            metrics = self.calc_metrics(Z_samples, F, sigma, w0, w, Q)
            metrics['iteration'] = iteration
            metrics['w0'] = w0
            metrics['w'] = w

            elapsed = time.time() - start_time
            metrics['time'] = elapsed

            self.history.append(metrics)

            if verbose:
                print(f"Iter {iteration:2d} | "
                      f"Q={Q:10.2f} | "
                      f"w0={w0:7.4f} (true={self.true_params['w0']:7.4f}) | "
                      f"w={w:7.4f} (true={self.true_params['w']:7.4f}) | "
                      f"corr_Z={metrics['corr_Z']:.4f} | "
                      f"time={elapsed:.2f}s")

        # Store final parameters
        self.model.params['Z'] = Z
        self.model.params['F'] = F
        self.model.params['sigma'] = sigma
        self.model.params['w0'] = w0
        self.model.params['w'] = w

        print("-" * 60)
        print("EM algorithm completed.")

        return self.history

    def plot_results(self, save: bool = True) -> None:
        """Generate and save result plots."""
        print("\n" + "=" * 60)
        print("Generating Plots")
        print("=" * 60)

        if not self.history:
            print("No results to plot. Run fit() first.")
            return

        # Extract history
        iterations = [h['iteration'] for h in self.history]
        Q_values = [h['Q'] for h in self.history]
        w0_values = [h['w0'] for h in self.history]
        w_values = [h['w'] for h in self.history]
        corr_Z = [h['corr_Z'] for h in self.history]
        corr_F = [h['corr_F'] for h in self.history]
        rmse_Y = [h['rmse_Y'] for h in self.history]
        rmse_X = [h['rmse_X'] for h in self.history]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Q function
        ax = axes[0, 0]
        ax.plot(iterations, Q_values, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Q function')
        ax.set_title('Q Function vs Iteration')
        ax.grid(True, alpha=0.3)

        # Plot 2: w0 convergence
        ax = axes[0, 1]
        ax.plot(iterations, w0_values, 'b-o', linewidth=2, markersize=6, label='Estimated')
        ax.axhline(y=self.true_params['w0'], color='r', linestyle='--',
                   linewidth=2, label=f"True ({self.true_params['w0']:.4f})")
        ax.set_xlabel('Iteration')
        ax.set_ylabel('w0')
        ax.set_title('w0 Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: w convergence
        ax = axes[0, 2]
        ax.plot(iterations, w_values, 'b-o', linewidth=2, markersize=6, label='Estimated')
        ax.axhline(y=self.true_params['w'], color='r', linestyle='--',
                   linewidth=2, label=f"True ({self.true_params['w']:.4f})")
        ax.set_xlabel('Iteration')
        ax.set_ylabel('w')
        ax.set_title('w Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Correlation of Z
        ax = axes[1, 0]
        ax.plot(iterations, corr_Z, 'g-o', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Correlation')
        ax.set_title('Correlation of Z (Gram matrix)')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        # Plot 5: RMSE of Y
        ax = axes[1, 1]
        ax.plot(iterations, rmse_Y, 'r-o', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE of Y Prediction')
        ax.grid(True, alpha=0.3)

        # Plot 6: RMSE of X
        ax = axes[1, 2]
        ax.plot(iterations, rmse_X, 'm-o', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE of X Reconstruction')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "convergence_plots.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filepath}")

        plt.close()

        # Additional plot: Scatter plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Scatter: True Z vs Estimated Z (using Gram matrices)
        Z_est = self.model.params['Z']
        Z_true = self.true_params['Z']

        ax = axes[0]
        ZZ_true = (Z_true @ Z_true.T).flatten()
        ZZ_est = (Z_est @ Z_est.T).flatten()
        ax.scatter(ZZ_true, ZZ_est, alpha=0.3, s=10)
        ax.plot([ZZ_true.min(), ZZ_true.max()],
                [ZZ_true.min(), ZZ_true.max()], 'r--', linewidth=2)
        ax.set_xlabel('True Z @ Z^T')
        ax.set_ylabel('Estimated Z @ Z^T')
        ax.set_title(f'Z Inner Products (corr={corr_Z[-1]:.4f})')
        ax.grid(True, alpha=0.3)

        # Scatter: True Y vs Predicted Y
        ax = axes[1]
        Y_true = self.true_params['Y']
        ZZT = Z_est @ Z_est.T
        logits = self.model.params['w0'] + self.model.params['w'] * ZZT
        Y_pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))

        # Get upper triangular elements
        mask = np.triu(np.ones_like(Y_true, dtype=bool), k=1)
        y_true_flat = Y_true[mask]
        y_pred_flat = Y_pred[mask]

        ax.scatter(y_true_flat, y_pred_flat, alpha=0.3, s=10)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2)
        ax.set_xlabel('True Y')
        ax.set_ylabel('Predicted P(Y=1)')
        ax.set_title(f'Y Prediction (RMSE={rmse_Y[-1]:.4f})')
        ax.grid(True, alpha=0.3)

        # Scatter: True F vs Estimated F (Gram matrices)
        ax = axes[2]
        F_est = self.model.params['F']
        F_true = self.true_params['F']

        FF_true = (F_true @ F_true.T).flatten()
        FF_est = (F_est @ F_est.T).flatten()
        ax.scatter(FF_true, FF_est, alpha=0.5, s=20)
        ax.plot([FF_true.min(), FF_true.max()],
                [FF_true.min(), FF_true.max()], 'r--', linewidth=2)
        ax.set_xlabel('True F @ F^T')
        ax.set_ylabel('Estimated F @ F^T')
        ax.set_title(f'F Structure (corr={corr_F[-1]:.4f})')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "scatter_plots.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filepath}")

        plt.close()

    def print_summary(self) -> None:
        """Print summary of results."""
        print("\n" + "=" * 60)
        print("Experiment Summary")
        print("=" * 60)

        if not self.history:
            print("No results. Run fit() first.")
            return

        final = self.history[-1]

        print(f"\nData:")
        print(f"  n={self.n}, d={self.d}, k={self.k}")
        print(f"  L={self.L}, num_iter={self.num_iter}")

        print(f"\nTrue Parameters:")
        print(f"  w0 = {self.true_params['w0']:.4f}")
        print(f"  w  = {self.true_params['w']:.4f}")

        print(f"\nEstimated Parameters:")
        print(f"  w0 = {final['w0']:.4f} (error: {final['rmse_w0']:.4f})")
        print(f"  w  = {final['w']:.4f} (error: {final['rmse_w']:.4f})")

        print(f"\nFinal Metrics:")
        print(f"  Q function:    {final['Q']:.2f}")
        print(f"  Corr(Z):       {final['corr_Z']:.4f}")
        print(f"  Corr(F):       {final['corr_F']:.4f}")
        print(f"  RMSE(Y):       {final['rmse_Y']:.4f}")
        print(f"  RMSE(X):       {final['rmse_X']:.4f}")
        print(f"  RMSE(sigma):   {final['rmse_sigma']:.4f}")

        total_time = sum(h['time'] for h in self.history)
        print(f"\nTotal time: {total_time:.2f}s")


def main():
    """Run the main experiment."""
    print("=" * 60)
    print("Latent Structural Model for Binary Relational Data")
    print("Experiment Runner")
    print("=" * 60)

    # Create experiment runner
    runner = ExperimentRunner(
        n=150,
        d=15,
        k=3,
        L=10,
        num_iter=15,
        seed=1980,
        output_dir="c:/研究2/results"
    )

    # Generate data
    runner.generate_data()

    # Initialize model with informed initialization
    runner.initialize_model(init_seed=42, use_informed_init=True)

    # Run EM algorithm
    runner.fit(verbose=True)

    # Plot results
    runner.plot_results(save=True)

    # Print summary
    runner.print_summary()

    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
