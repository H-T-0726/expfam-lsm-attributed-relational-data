"""
Data Generator for Latent Structural Model for Binary Relational Data.

This module implements the data generation logic from setTrueParams.m
Based on: Mikawa et al., "A study on latent structural models for binary
relational data with attribute information", NOLTA 2024.
"""

import numpy as np
from typing import Dict, Any, Tuple


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute sigmoid function with numerical stability."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def normalize_zscore(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Normalize array to zero mean and unit variance along specified axis.

    Parameters
    ----------
    X : np.ndarray
        Input array
    axis : int
        Axis along which to normalize (0 for columns, 1 for rows)

    Returns
    -------
    np.ndarray
        Normalized array
    """
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True, ddof=0)
    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def set_true_params(
    n: int,
    d: int,
    k: int,
    seed: int = 1980,
    var_f: float = 5.0,
    uniq: float = 0.1,
    w_scale: float = 3.0
) -> Dict[str, Any]:
    """
    Generate true parameters and synthetic data for the latent structural model.

    This function generates:
    - Latent variables Z ~ N(0, I), then normalized
    - Factor loading matrix F ~ N(0, var_f), with row normalization
    - Observed data X ~ N(ZF^T, Sigma), then z-score normalized
    - Binary relational data Y ~ Bernoulli(sigmoid(w0 + w * Z @ Z^T))

    Parameters
    ----------
    n : int
        Number of data points (objects)
    d : int
        Dimensionality of observed data X
    k : int
        Dimensionality of latent variable Z
    seed : int, optional
        Random seed for reproducibility (default: 1980)
    var_f : float, optional
        Variance for generating F (default: 5.0)
    uniq : float, optional
        Uniqueness value for diagonal of Sigma (default: 0.1)
    w_scale : float, optional
        Scale factor for w (default: 3.0)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'X': Observed data matrix (n x d), normalized
        - 'Y': Binary relational data matrix (n x n), symmetric with 0 diagonal
        - 'Z': Latent variable matrix (n x k), normalized
        - 'F': Factor loading matrix (d x k)
        - 'sigma': Diagonal covariance matrix (d x d)
        - 'w0': Bias parameter for Y generation
        - 'w': Weight parameter for Y generation
        - 'var_z': Variance of Z (fixed to 1.0)
        - 'var_f': Variance used to generate F

    Notes
    -----
    - var_z is fixed to 1.0 for identifiability
    - X is normalized (zero mean, unit variance per dimension) as mu_x is omitted
    - Y is symmetric with zero diagonal: Y_ij = Y_ji, Y_ii = 0
    """
    rng = np.random.default_rng(seed)

    # Fixed parameters
    var_z = 1.0  # Fixed for identifiability

    # Diagonal covariance matrix Sigma = diag(uniq, uniq, ..., uniq)
    sigma = np.diag(np.full(d, uniq))

    # Generate latent variables Z ~ N(0, var_z * I)
    Z = rng.normal(0, np.sqrt(var_z), size=(n, k))
    # Normalize Z (zero mean, unit variance per column)
    Z = normalize_zscore(Z, axis=0)

    # Generate factor loading matrix F ~ N(0, var_f)
    F = rng.normal(0, np.sqrt(var_f), size=(d, k))

    # Normalize F rows for factor analysis constraint
    # F_i = (F_i / ||F_i||) * sqrt(1 - sigma_ii)
    for i in range(d):
        norm_fi = np.linalg.norm(F[i, :])
        if norm_fi > 0:
            F[i, :] = (F[i, :] / norm_fi) * np.sqrt(1.0 - sigma[i, i])

    # Generate w0 and w
    w0 = rng.standard_normal()
    w = w_scale * rng.standard_normal()

    # Generate X ~ N(Z @ F^T, Sigma)
    # Each row x_i ~ N(F @ z_i, Sigma)
    mean_X = Z @ F.T  # (n, d)
    # Add noise: epsilon_i ~ N(0, Sigma)
    noise = rng.multivariate_normal(np.zeros(d), sigma, size=n)
    X = mean_X + noise

    # Normalize X (zero mean, unit variance per dimension)
    X = normalize_zscore(X, axis=0)

    # Generate Y (binary relational data)
    # Compute logits: L_ij = w0 + w * z_i^T z_j
    ZZT = Z @ Z.T  # (n, n)
    logits = w0 + w * ZZT
    probs = sigmoid(logits)

    # Generate upper triangular part only (i < j)
    # np.triu with k=1 excludes diagonal
    uniform_samples = rng.random(size=(n, n))
    Y_full = (uniform_samples < probs).astype(np.float64)
    Y_upper = np.triu(Y_full, k=1)

    # Make symmetric: Y = Y_upper + Y_upper^T
    Y = Y_upper + Y_upper.T

    # Ensure diagonal is 0 (should already be, but explicit)
    np.fill_diagonal(Y, 0)

    return {
        'X': X,
        'Y': Y,
        'Z': Z,
        'F': F,
        'sigma': sigma,
        'w0': w0,
        'w': w,
        'var_z': var_z,
        'var_f': var_f,
    }


def generate_data(
    n: int,
    d: int,
    k: int,
    seed: int = 1980
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Convenience function to generate data matching MATLAB interface.

    Parameters
    ----------
    n : int
        Number of data points
    d : int
        Dimensionality of observed data
    k : int
        Dimensionality of latent space
    seed : int, optional
        Random seed (default: 1980)

    Returns
    -------
    X : np.ndarray
        Observed data matrix (n x d)
    Y : np.ndarray
        Binary relational data matrix (n x n)
    params : Dict[str, Any]
        Dictionary of all parameters
    """
    params = set_true_params(n, d, k, seed)
    return params['X'], params['Y'], params


if __name__ == "__main__":
    print("=" * 60)
    print("Data Generator Self-Validation")
    print("=" * 60)

    # Test parameters (matching paper's Experiment 1)
    n, d, k = 150, 15, 3
    seed = 1980

    print(f"\nGenerating data with n={n}, d={d}, k={k}, seed={seed}")
    params = set_true_params(n, d, k, seed)

    X = params['X']
    Y = params['Y']
    Z = params['Z']
    F = params['F']

    print(f"\nGenerated shapes:")
    print(f"  X: {X.shape}")
    print(f"  Y: {Y.shape}")
    print(f"  Z: {Z.shape}")
    print(f"  F: {F.shape}")
    print(f"  w0: {params['w0']:.4f}")
    print(f"  w: {params['w']:.4f}")

    # Validation tests
    print("\n" + "-" * 40)
    print("Validation Tests")
    print("-" * 40)

    # Test 1: X normalization (mean ~ 0, variance ~ 1)
    X_mean = np.mean(X, axis=0)
    X_var = np.var(X, axis=0, ddof=0)

    print(f"\n[Test 1] X normalization:")
    print(f"  Mean per dimension (should be ~0):")
    print(f"    min={X_mean.min():.2e}, max={X_mean.max():.2e}")
    print(f"  Variance per dimension (should be ~1):")
    print(f"    min={X_var.min():.4f}, max={X_var.max():.4f}")

    assert np.allclose(X_mean, 0, atol=1e-10), "FAILED: X mean is not close to 0"
    assert np.allclose(X_var, 1, atol=1e-10), "FAILED: X variance is not close to 1"
    print("  [PASSED] X is properly normalized")

    # Test 2: Y symmetry
    print(f"\n[Test 2] Y symmetry:")
    is_symmetric = np.allclose(Y, Y.T)
    print(f"  Y == Y^T: {is_symmetric}")
    assert is_symmetric, "FAILED: Y is not symmetric"
    print("  [PASSED] Y is symmetric")

    # Test 3: Y diagonal is 0
    print(f"\n[Test 3] Y diagonal:")
    diag_Y = np.diag(Y)
    all_zero = np.allclose(diag_Y, 0)
    print(f"  All diagonal elements are 0: {all_zero}")
    assert all_zero, "FAILED: Y diagonal is not 0"
    print("  [PASSED] Y diagonal is all zeros")

    # Test 4: Y is binary
    print(f"\n[Test 4] Y binary values:")
    unique_values = np.unique(Y)
    is_binary = np.all(np.isin(unique_values, [0, 1]))
    print(f"  Unique values in Y: {unique_values}")
    assert is_binary, "FAILED: Y contains non-binary values"
    print("  [PASSED] Y contains only 0 and 1")

    # Test 5: Z normalization
    print(f"\n[Test 5] Z normalization:")
    Z_mean = np.mean(Z, axis=0)
    Z_var = np.var(Z, axis=0, ddof=0)
    print(f"  Mean per factor: {Z_mean}")
    print(f"  Variance per factor: {Z_var}")
    assert np.allclose(Z_mean, 0, atol=1e-10), "FAILED: Z mean is not close to 0"
    assert np.allclose(Z_var, 1, atol=1e-10), "FAILED: Z variance is not close to 1"
    print("  [PASSED] Z is properly normalized")

    # Additional statistics
    print("\n" + "-" * 40)
    print("Additional Statistics")
    print("-" * 40)

    # Y density (proportion of 1s)
    n_possible_edges = n * (n - 1) // 2
    n_edges = int(np.sum(np.triu(Y, k=1)))
    density = n_edges / n_possible_edges
    print(f"\nY statistics:")
    print(f"  Number of edges: {n_edges}")
    print(f"  Possible edges: {n_possible_edges}")
    print(f"  Density: {density:.4f}")

    # F row norms
    F_row_norms = np.linalg.norm(F, axis=1)
    expected_norm = np.sqrt(1 - params['sigma'][0, 0])
    print(f"\nF row norms:")
    print(f"  Expected: {expected_norm:.4f}")
    print(f"  Actual: min={F_row_norms.min():.4f}, max={F_row_norms.max():.4f}")

    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED!")
    print("=" * 60)
