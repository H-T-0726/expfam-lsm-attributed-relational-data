"""
Paper Experiment 2: Identifying the true dimension using BIC.

Reproduces Experiment 2 from:
Mikawa et al., "A study on latent structural models for binary
relational data with attribute information", NOLTA 2024.

Setting:
- True latent dimension k* in {1, 3, 5, 7, 9}
- Data: n=150, d=15
- Estimated k from 1 to 10
- Calculate BIC for each combination
- Verify that BIC is minimized at k = k*
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


class Experiment2Runner:
    """
    Run Experiment 2: Model selection using BIC.
    """

    def __init__(
        self,
        n: int = 150,
        d: int = 15,
        k_true_list: List[int] = None,
        k_range: List[int] = None,
        L: int = 10,
        num_iter: int = 10,
        n_trials: int = 1,
        data_seed: int = 1980,
        output_dir: str = "results"
    ):
        """
        Initialize Experiment 2 runner.

        Parameters
        ----------
        n : int
            Number of data points
        d : int
            Dimensionality of observed data
        k_true_list : List[int]
            List of true latent dimensions to test
        k_range : List[int]
            Range of k values to estimate
        L : int
            Number of Monte Carlo samples
        num_iter : int
            Number of EM iterations
        n_trials : int
            Number of random initialization trials (best BIC is selected)
        data_seed : int
            Base random seed for data generation
        output_dir : str
            Directory to save results
        """
        self.n = n
        self.d = d
        self.k_true_list = k_true_list or [1, 3, 5, 7, 9]
        self.k_range = k_range or list(range(1, 11))
        self.L = L
        self.num_iter = num_iter
        self.n_trials = n_trials
        self.data_seed = data_seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage: {k_true: {k: {'BIC': ..., 'Q': ..., ...}}}
        self.results: Dict[int, Dict[int, Dict[str, Any]]] = {}

    def calc_BIC(
        self,
        log_likelihood: float,
        k: int,
        n: int,
        d: int
    ) -> float:
        """
        Calculate Bayesian Information Criterion (BIC).

        Based on paper Eq. (26):
        BIC = -2 ln L + ((k+1)d - k(k-1)/2) ln n

        Parameters
        ----------
        log_likelihood : float
            Log-likelihood (or Q function value)
        k : int
            Number of latent dimensions
        n : int
            Number of data points
        d : int
            Dimensionality of observed data

        Returns
        -------
        float
            BIC value (lower is better)
        """
        # Number of free parameters
        # F: d x k parameters
        # sigma: d parameters (diagonal)
        # w0: 1 parameter
        # w: 1 parameter
        # But paper uses: (k+1)d - k(k-1)/2
        num_params = (k + 1) * d - k * (k - 1) / 2

        BIC = -2 * log_likelihood + num_params * np.log(n)
        return BIC

    def calc_log_likelihood(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        F: np.ndarray,
        sigma: np.ndarray,
        var_z: float,
        w0: float,
        w: float
    ) -> float:
        """
        Calculate log-likelihood.

        ln L = ln p(X|Z,F,Σ) + ln p(Y|Z,w0,w) + ln p(Z)
        """
        n, k = Z.shape
        d = X.shape[1]

        # ln p(Z) = -nk/2 ln(2π) - nk/2 ln(σ²_z) - 1/(2σ²_z) Σ||z_i||²
        # Note: var_z = 1.0 by design, so the second term is 0
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

    def run_single_experiment(
        self,
        k_true: int,
        k_est: int,
        X: np.ndarray,
        Y: np.ndarray,
        true_params: Dict[str, Any],
        init_seed: int
    ) -> Dict[str, Any]:
        """Run a single experiment for given k_true and k_est."""
        # Initialize model
        model = LatentStructuralModel(n=self.n, d=self.d, k=k_est, L=self.L)
        model.initialize_params(true_params=true_params, seed=init_seed)

        # Use informed initialization
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
        best_params = None

        for iteration in range(1, self.num_iter + 1):
            # E-Step
            Z_samples = np.zeros((self.n, k_est, self.L))

            for l in range(self.L):
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

            # Calculate log-likelihood
            log_likelihood = self.calc_log_likelihood(
                X, Y, Z, F, sigma, var_z, w0, w
            )

            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_params = {
                    'Z': Z.copy(),
                    'F': F.copy(),
                    'sigma': sigma.copy(),
                    'w0': w0,
                    'w': w
                }

        # Calculate BIC using best log-likelihood
        BIC = self.calc_BIC(best_log_likelihood, k_est, self.n, self.d)

        return {
            'BIC': BIC,
            'log_likelihood': best_log_likelihood,
            'params': best_params
        }

    def run_experiment(self):
        """Run the full experiment."""
        print("=" * 60)
        print("Running Experiment 2: Model Selection by BIC")
        print(f"  num_iter={self.num_iter}, n_trials={self.n_trials}")
        print("=" * 60)

        total_runs = len(self.k_true_list) * len(self.k_range)
        run_count = 0

        for k_true in self.k_true_list:
            print(f"\n{'='*60}")
            print(f"Generating data with k* = {k_true}")
            print("=" * 60)

            # Generate data with this k_true
            true_params = set_true_params(
                n=self.n, d=self.d, k=k_true,
                seed=self.data_seed + k_true * 100
            )
            X = true_params['X']
            Y = true_params['Y']

            print(f"  True w0: {true_params['w0']:.4f}")
            print(f"  True w: {true_params['w']:.4f}")
            print(f"  Y density: {np.mean(Y):.4f}")

            self.results[k_true] = {}

            for k_est in self.k_range:
                run_count += 1

                print(f"  k={k_est:2d} ({run_count}/{total_runs})...", end=" ")

                start_time = time.time()

                # Run multiple trials and select best BIC
                best_result = None
                for trial in range(self.n_trials):
                    init_seed = 42 + k_est * 10 + trial * 1000
                    result = self.run_single_experiment(
                        k_true, k_est, X, Y, true_params, init_seed
                    )
                    if best_result is None or result['BIC'] < best_result['BIC']:
                        best_result = result

                elapsed = time.time() - start_time

                self.results[k_true][k_est] = best_result
                print(f"BIC={best_result['BIC']:10.1f}, "
                      f"logL={best_result['log_likelihood']:10.1f}, "
                      f"time={elapsed:.1f}s")

    def find_best_k(self) -> Dict[int, int]:
        """Find the k that minimizes BIC for each k_true."""
        best_k = {}
        for k_true in self.k_true_list:
            BIC_values = {k: self.results[k_true][k]['BIC'] for k in self.k_range}
            best_k[k_true] = min(BIC_values, key=BIC_values.get)
        return best_k

    def plot_results(self, save: bool = True):
        """Generate BIC plot similar to paper Figure 3."""
        print("\n" + "=" * 60)
        print("Generating Plots")
        print("=" * 60)

        fig, ax = plt.subplots(figsize=(10, 7))

        colors = ['blue', 'green', 'red', 'purple', 'orange']
        markers = ['o', 's', '^', 'D', 'v']

        for i, k_true in enumerate(self.k_true_list):
            k_values = list(self.k_range)
            BIC_values = [self.results[k_true][k]['BIC'] for k in k_values]

            ax.plot(k_values, BIC_values,
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    linewidth=2, markersize=8,
                    label=f'k* = {k_true}')

            # Mark the minimum BIC
            min_idx = np.argmin(BIC_values)
            ax.scatter([k_values[min_idx]], [BIC_values[min_idx]],
                       s=200, color=colors[i % len(colors)],
                       marker='*', zorder=5, edgecolors='black')

        ax.set_xlabel('Number of dimensions (k)', fontsize=14)
        ax.set_ylabel('BIC', fontsize=14)
        ax.set_title('Experiment 2: BIC for Different True Dimensions k*',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(self.k_range)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "paper_experiment_2.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filepath}")

        plt.close()

    def print_results_table(self):
        """Print results table."""
        print("\n" + "=" * 60)
        print("BIC Results Table")
        print("=" * 60)

        # Header
        header = "k*\\k |"
        for k in self.k_range:
            header += f" {k:>8} |"
        print(header)
        print("-" * len(header))

        # Data rows
        for k_true in self.k_true_list:
            row = f" {k_true:>2} |"
            BIC_values = [self.results[k_true][k]['BIC'] for k in self.k_range]
            min_BIC = min(BIC_values)

            for k in self.k_range:
                BIC = self.results[k_true][k]['BIC']
                # Mark minimum with *
                if BIC == min_BIC:
                    row += f" *{BIC:>6.0f}*|"
                else:
                    row += f" {BIC:>7.0f} |"
            print(row)

        print("-" * len(header))

        # Summary
        print("\nSummary: Best k (by BIC) for each k*")
        print("-" * 40)
        best_k = self.find_best_k()
        correct = 0
        for k_true in self.k_true_list:
            match = "OK" if best_k[k_true] == k_true else "MISS"
            if best_k[k_true] == k_true:
                correct += 1
            print(f"  k* = {k_true}: Best k = {best_k[k_true]} [{match}]")

        accuracy = correct / len(self.k_true_list) * 100
        print(f"\nAccuracy: {correct}/{len(self.k_true_list)} = {accuracy:.1f}%")

    def save_csv(self, filepath: str = None):
        """Save BIC results table to CSV."""
        import pandas as pd
        if filepath is None:
            filepath = Path(self.output_dir) / "results_exp2_bic.csv"

        rows = []
        best_k = self.find_best_k()
        for k_true in self.k_true_list:
            for k in self.k_range:
                r = self.results[k_true][k]
                rows.append({
                    'k_true': k_true,
                    'k_est': k,
                    'BIC': r['BIC'],
                    'log_likelihood': r['log_likelihood'],
                    'best_k': best_k[k_true],
                    'correct': int(best_k[k_true] == k_true),
                })
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"  CSV saved: {filepath}")


def main():
    """Run Experiment 2."""
    print("=" * 60)
    print("Paper Experiment 2: Model Selection by BIC")
    print("=" * 60)

    runner = Experiment2Runner(
        n=150,
        d=15,
        k_true_list=[1, 3, 5, 7, 9],
        k_range=list(range(1, 11)),
        L=10,
        num_iter=30,
        n_trials=3,
        data_seed=1980,
        output_dir="c:/研究2/results"
    )

    # Run experiment
    runner.run_experiment()

    # Plot results
    runner.plot_results(save=True)

    # Print table
    runner.print_results_table()

    # Save CSV
    runner.save_csv()

    print("\n" + "=" * 60)
    print("Experiment 2 completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
