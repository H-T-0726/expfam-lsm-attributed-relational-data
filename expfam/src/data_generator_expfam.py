"""
Data Generators — Exponential Family Relational Data.

Provides generators for three families used in the ExpFam generality proof:
    generate_bernoulli_data : y_ij ~ Bernoulli(sigmoid(w0 + w * z_i^T z_j))
    generate_poisson_data   : y_ij ~ Poisson(exp(w0 + w * z_i^T z_j))
    generate_gaussian_data  : y_ij ~ N(w0 + w * z_i^T z_j, sigma_y^2)

Attribute data generation is shared:
    z_i ~ N(0, I)  [normalized]
    x_i ~ N(F z_i, Sigma)  [z-score normalized]

Y is symmetric with zero diagonal (upper triangle sampled, then mirrored).
"""

import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

_BASE = Path(__file__).parent.parent.parent / "reproduction" / "src"
sys.path.insert(0, str(_BASE))
from data_generator import normalize_zscore  # noqa: E402


def generate_poisson_data(
    n: int,
    d: int,
    k: int,
    seed: int = 1980,
    var_f: float = 5.0,
    uniq: float = 0.1,
    w0_true: float = 0.0,
    w_true: float = 0.5,
) -> Dict[str, Any]:
    """
    Generate synthetic Poisson relational data.

    Parameters
    ----------
    n         : number of nodes
    d         : attribute dimensionality
    k         : true latent dimensionality
    seed      : random seed
    var_f     : variance for F generation
    uniq      : diagonal value of Sigma (uniqueness)
    w0_true   : true bias for Poisson rate (eta = w0 + w * z_i^T z_j)
    w_true    : true weight; kept small so lambda stays reasonable

    Returns
    -------
    dict with keys: X, Y, Z, F, sigma, w0, w, var_z, var_f,
                    lambda_mean, lambda_max, count_mean, count_max,
                    density (fraction of pairs with y > 0)
    """
    rng = np.random.default_rng(seed)
    var_z = 1.0

    sigma = np.diag(np.full(d, uniq))

    # --- Latent variables Z ~ N(0, I), normalised column-wise ---
    Z = rng.normal(0.0, np.sqrt(var_z), size=(n, k))
    Z = normalize_zscore(Z, axis=0)

    # --- Factor loadings F (row-normalised) ---
    F = rng.normal(0.0, np.sqrt(var_f), size=(d, k))
    for i in range(d):
        norm_fi = np.linalg.norm(F[i, :])
        if norm_fi > 0:
            F[i, :] = (F[i, :] / norm_fi) * np.sqrt(1.0 - sigma[i, i])

    # --- Attribute data X ~ N(Z F^T, Sigma), z-score normalised ---
    noise = rng.multivariate_normal(np.zeros(d), sigma, size=n)
    X = Z @ F.T + noise
    X = normalize_zscore(X, axis=0)

    # --- Poisson rate matrix lambda_ij = exp(w0 + w * z_i^T z_j) ---
    ZZT = Z @ Z.T                              # (n, n)
    eta = w0_true + w_true * ZZT               # (n, n)
    eta_clipped = np.clip(eta, -20, 10)        # clip for stability
    lam = np.exp(eta_clipped)                  # (n, n) Poisson rates

    # --- Sample Y (upper triangle only, then symmetrise) ---
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    counts = rng.poisson(lam[upper_mask])      # vectorised sampling

    Y = np.zeros((n, n), dtype=np.float64)
    Y[upper_mask] = counts
    Y = Y + Y.T                                # symmetric
    np.fill_diagonal(Y, 0.0)

    # Diagnostics
    upper_vals = Y[upper_mask]
    n_pairs = upper_mask.sum()

    return {
        "X": X,
        "Y": Y,
        "Z": Z,
        "F": F,
        "sigma": sigma,
        "w0": w0_true,
        "w": w_true,
        "var_z": var_z,
        "var_f": var_f,
        # diagnostics
        "lambda_mean": float(lam[upper_mask].mean()),
        "lambda_max": float(lam[upper_mask].max()),
        "count_mean": float(upper_vals.mean()),
        "count_max": float(upper_vals.max()),
        "density": float((upper_vals > 0).sum() / n_pairs),
    }


def _generate_base(n, d, k, seed, var_f=5.0, uniq=0.1):
    """Shared latent structure: Z, F, X (common to all families)."""
    from data_generator import normalize_zscore  # noqa
    rng = np.random.default_rng(seed)
    var_z = 1.0
    sigma = np.diag(np.full(d, uniq))

    Z = rng.normal(0.0, np.sqrt(var_z), size=(n, k))
    Z = normalize_zscore(Z, axis=0)

    F = rng.normal(0.0, np.sqrt(var_f), size=(d, k))
    for i in range(d):
        norm_fi = np.linalg.norm(F[i, :])
        if norm_fi > 0:
            F[i, :] = (F[i, :] / norm_fi) * np.sqrt(1.0 - sigma[i, i])

    noise = rng.multivariate_normal(np.zeros(d), sigma, size=n)
    X = Z @ F.T + noise
    X = normalize_zscore(X, axis=0)

    return rng, Z, F, X, sigma, var_z


def generate_bernoulli_data(
    n: int, d: int, k: int, seed: int = 1980,
    var_f: float = 5.0, uniq: float = 0.1,
    w0_true: float = -1.0, w_true: float = 1.5,
) -> Dict[str, Any]:
    """
    Generate synthetic Bernoulli relational data (NOLTA 2024 setting).

    Default w0=-1.0, w=1.5 gives moderate link density (~25-35%).
    """
    rng, Z, F, X, sigma, var_z = _generate_base(n, d, k, seed, var_f, uniq)

    ZZT = Z @ Z.T
    eta = w0_true + w_true * ZZT
    prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))

    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    links = rng.binomial(1, prob[upper_mask]).astype(np.float64)

    Y = np.zeros((n, n), dtype=np.float64)
    Y[upper_mask] = links
    Y = Y + Y.T
    np.fill_diagonal(Y, 0.0)

    upper_vals = Y[upper_mask]
    return {
        "X": X, "Y": Y, "Z": Z, "F": F, "sigma": sigma,
        "w0": w0_true, "w": w_true,
        "var_z": var_z, "var_f": var_f,
        "density": float(upper_vals.mean()),
    }


def generate_gaussian_data(
    n: int, d: int, k: int, seed: int = 1980,
    var_f: float = 5.0, uniq: float = 0.1,
    w0_true: float = 0.0, w_true: float = 0.5,
    sigma_y_true: float = 0.1,
) -> Dict[str, Any]:
    """
    Generate synthetic Gaussian relational data (SMC 2022 extension).

    y_ij ~ N(w0 + w * z_i^T z_j, sigma_y^2)

    Default sigma_y_true=0.1 gives low noise continuous observations.
    """
    rng, Z, F, X, sigma, var_z = _generate_base(n, d, k, seed, var_f, uniq)

    ZZT = Z @ Z.T
    eta = w0_true + w_true * ZZT  # (n, n) true mean

    # Sample upper triangle, symmetrise
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    noise_vals = rng.normal(0.0, sigma_y_true, size=int(upper_mask.sum()))

    Y = np.zeros((n, n), dtype=np.float64)
    Y[upper_mask] = eta[upper_mask] + noise_vals
    Y = Y + Y.T
    np.fill_diagonal(Y, 0.0)

    upper_vals = Y[upper_mask]
    return {
        "X": X, "Y": Y, "Z": Z, "F": F, "sigma": sigma,
        "w0": w0_true, "w": w_true,
        "sigma_y": sigma_y_true,
        "var_z": var_z, "var_f": var_f,
        "y_mean": float(upper_vals.mean()),
        "y_std": float(upper_vals.std()),
        "y_min": float(upper_vals.min()),
        "y_max": float(upper_vals.max()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dual-ExpFam data generator
# ──────────────────────────────────────────────────────────────────────────────

# Default true parameters per family
_Y_DEFAULTS = {
    "bernoulli": dict(w0=-1.0, w=1.5),
    "poisson":   dict(w0=0.0,  w=0.5),
    "gaussian":  dict(w0=0.0,  w=0.5, sigma_y=0.1),
}


def generate_dual_data(
    n: int,
    d: int,
    k: int,
    seed: int = 1980,
    family_x: str = "gaussian",
    family_y: str = "bernoulli",
    # X-side params
    var_f: float = 5.0,
    uniq: float = 0.1,
    sigma_x_true: float = 0.1,   # Gaussian X noise σ
    # Y-side params (defaults filled from _Y_DEFAULTS if not given)
    w0_true: float = None,
    w_true:  float = None,
    sigma_y_true: float = 0.1,   # Gaussian Y noise σ
) -> Dict[str, Any]:
    """
    Generate synthetic data for Dual-ExpFam LSM.

    Both attribute X and relational Y can follow independent exponential
    family distributions (gaussian / bernoulli / poisson).

    Shared latent structure:
        z_i ~ N(0, I_k)      (column-normalised)
        F   ~ row-normalised (same as _generate_base)

    X generation (no z-score normalisation for discrete families):
        Gaussian  : x_i ~ N(F z_i, σ_x^2 I), then z-score normalised
        Bernoulli : x_{ij} ~ Bernoulli(σ(f_j^T z_i))
        Poisson   : x_{ij} ~ Poisson(exp(f_j^T z_i))

    Y generation (upper triangle, then symmetrised):
        Bernoulli : y_ij ~ Bernoulli(σ(w0 + w z_i^T z_j))
        Poisson   : y_ij ~ Poisson(exp(w0 + w z_i^T z_j))
        Gaussian  : y_ij ~ N(w0 + w z_i^T z_j, σ_y^2)

    Returns
    -------
    dict with keys: X, Y, Z, F, sigma, w0, w, var_z, var_f,
                    family_x, family_y, and family-specific diagnostics.
    """
    from data_generator import normalize_zscore  # noqa

    valid = ("gaussian", "bernoulli", "poisson")
    if family_x not in valid:
        raise ValueError(f"family_x must be one of {valid}, got '{family_x}'")
    if family_y not in valid:
        raise ValueError(f"family_y must be one of {valid}, got '{family_y}'")

    # Fill Y-side defaults
    if w0_true is None:
        w0_true = _Y_DEFAULTS[family_y]["w0"]
    if w_true is None:
        w_true = _Y_DEFAULTS[family_y]["w"]

    rng   = np.random.default_rng(seed)
    var_z = 1.0

    # ── Shared latent structure ─────────────────────────────────────
    Z = rng.normal(0.0, np.sqrt(var_z), size=(n, k))
    Z = normalize_zscore(Z, axis=0)

    sigma_x = np.diag(np.full(d, uniq))   # X noise covariance (Gaussian X)
    F = rng.normal(0.0, np.sqrt(var_f), size=(d, k))
    for j in range(d):
        nrm = np.linalg.norm(F[j, :])
        if nrm > 0:
            F[j, :] = (F[j, :] / nrm) * np.sqrt(1.0 - sigma_x[j, j])

    # ── Generate X ─────────────────────────────────────────────────
    eta_x_full = Z @ F.T   # (n, d)  natural parameters for X

    if family_x == "gaussian":
        noise = rng.multivariate_normal(np.zeros(d), sigma_x, size=n)
        X = eta_x_full + noise
        X = normalize_zscore(X, axis=0)   # z-score normalise (same as existing generators)
    elif family_x == "bernoulli":
        prob_x = 1.0 / (1.0 + np.exp(-np.clip(eta_x_full, -500, 500)))
        X = rng.binomial(1, prob_x).astype(np.float64)
        # No normalisation: X ∈ {0,1}^{n×d}
    else:   # poisson
        lam_x = np.exp(np.clip(eta_x_full, -20, 10))
        X = rng.poisson(lam_x).astype(np.float64)
        # No normalisation: X ∈ Z_{≥0}^{n×d}

    # ── Generate Y (upper triangle, symmetrised) ────────────────────
    ZZT      = Z @ Z.T
    eta_y    = w0_true + w_true * ZZT    # (n, n)
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    Y = np.zeros((n, n), dtype=np.float64)

    if family_y == "bernoulli":
        prob_y = 1.0 / (1.0 + np.exp(-np.clip(eta_y, -500, 500)))
        Y[upper_mask] = rng.binomial(1, prob_y[upper_mask]).astype(np.float64)
    elif family_y == "poisson":
        lam_y = np.exp(np.clip(eta_y, -20, 10))
        Y[upper_mask] = rng.poisson(lam_y[upper_mask]).astype(np.float64)
    else:   # gaussian
        noise_y = rng.normal(0.0, sigma_y_true, size=int(upper_mask.sum()))
        Y[upper_mask] = eta_y[upper_mask] + noise_y

    Y = Y + Y.T
    np.fill_diagonal(Y, 0.0)

    # ── Diagnostics ─────────────────────────────────────────────────
    upper_Y = Y[upper_mask]
    upper_X = X.flatten()

    result = {
        # Data
        "X": X, "Y": Y, "Z": Z, "F": F,
        "sigma": sigma_x,
        # True parameters
        "w0": w0_true, "w": w_true,
        "var_z": var_z, "var_f": var_f,
        # Family labels
        "family_x": family_x,
        "family_y": family_y,
        # Y diagnostics
        "y_mean":  float(upper_Y.mean()),
        "y_std":   float(upper_Y.std()),
        "y_min":   float(upper_Y.min()),
        "y_max":   float(upper_Y.max()),
        # X diagnostics
        "x_mean":  float(upper_X.mean()),
        "x_std":   float(upper_X.std()),
    }
    if family_y == "bernoulli":
        result["y_density"] = float(upper_Y.mean())
    if family_y == "poisson":
        result["y_density"]    = float((upper_Y > 0).mean())
        result["y_count_max"]  = float(upper_Y.max())
    if family_y == "gaussian":
        result["sigma_y"] = sigma_y_true
    if family_x == "bernoulli":
        result["x_density"] = float((X > 0).mean())
    if family_x == "poisson":
        result["x_count_max"] = float(X.max())

    return result


if __name__ == "__main__":
    data = generate_poisson_data(n=150, d=15, k=3, seed=1980)
    print("Poisson data generation check:")
    print(f"  X : {data['X'].shape}, mean={data['X'].mean():.4f}")
    print(f"  Y : {data['Y'].shape}, max={int(data['count_max'])}, "
          f"mean={data['count_mean']:.3f}, density={data['density']:.3f}")
    print(f"  Z : {data['Z'].shape}")
    print(f"  w0={data['w0']:.3f}, w={data['w']:.3f}")
    print(f"  lambda mean={data['lambda_mean']:.3f}, max={data['lambda_max']:.3f}")
    assert np.allclose(data["Y"], data["Y"].T), "Y not symmetric"
    assert np.allclose(np.diag(data["Y"]), 0), "Y diagonal not zero"
    print("  [PASSED] Y is symmetric with zero diagonal")
