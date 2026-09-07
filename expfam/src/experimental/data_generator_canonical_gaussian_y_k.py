"""
Canonical Gaussian-Y data generator for the K_0 = K_GEN recovery experiment (2026-09-07).

PURPOSE
-------
Generate data that literally follows the canonical model for which the
Gaussian-Y population identifiability theorem holds:

    z_i     ~ iid N(0, I_K)                       <-- NO post-processing
    eta^X   = Z F^T                               (no bias)
    x_il    ~ Bernoulli( sigmoid(f_l^T z_i) )     <-- NO post-processing
    eta^Y   = w0 + w z_i^T z_j
    y_ij    = eta^Y_ij + eps_ij,  eps ~ iid N(0, sigma_y^2),  i < j
    Y       symmetric, zero diagonal              <-- NO post-processing

Under `w != 0` the marginal law of a single dyad y_ij determines
(w0, K, w^2, sigma_y^2) uniquely, hence K_0 = K_GEN is a theoretical
ground truth for this generator.
See `reports/identifiability/gaussian_y_k_identifiability_adversarial_audit_20260907.md`.

WHAT THIS FILE IS NOT
---------------------
This is NOT `expfam/src/data_generator_expfam.py::generate_dual_data`.
That historical generator applies `normalize_zscore(Z, axis=0)` after drawing Z
(KI-021 G1), which destroys the `z_i ~ iid N(0, I_K)` premise of the theorem.
This module performs NO standardization of Z, X, or Y at any point.

The historical generator is NOT modified by this file.

F CONVENTION
------------
`F_raw ~ iid N(0, var_f)` with each row rescaled to `||f_l|| = sqrt(1 - uniq)`.
This matches the historical convention so that the X-side signal scale is
comparable, and it is a *choice of the model parameter F*, not a deviation from
the canonical model: F is a free parameter of Theta_K and any particular value
is admissible. Row rescaling by a positive diagonal preserves rank(F).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

import numpy as np
from scipy.special import expit

GENERATOR_ID = "canonical-gaussian-y-k-v1"

# Post-processing operations that are forbidden in this generator.
# Kept as a literal so tests and the theory gate can assert against the source.
FORBIDDEN_POSTPROCESSING = (
    "normalize_zscore",
    "standardize",
    "whiten",
    "scale_Z",
)


def _sha256_array(a: np.ndarray) -> str:
    """Stable content hash of a float array (C-contiguous, float64)."""
    b = np.ascontiguousarray(a, dtype=np.float64)
    return hashlib.sha256(b.tobytes()).hexdigest()


def generate_canonical_gaussian_y(
    n: int,
    d: int,
    k: int,
    seed: int,
    *,
    w0_true: float = 0.0,
    w_true: float = 0.5,
    sigma_y_true: float = 0.1,
    var_f: float = 5.0,
    uniq: float = 0.1,
) -> Dict[str, Any]:
    """
    Draw one canonical Bernoulli-X / Gaussian-Y dataset.

    Parameters
    ----------
    n, d, k       : nodes, attribute dimension, TRUE latent dimension (K_GEN)
    seed          : data seed; the whole dataset is a deterministic function of it
    w0_true       : Y intercept
    w_true        : Y coupling scalar (must be != 0 for the theorem to apply)
    sigma_y_true  : Y noise STANDARD DEVIATION (not variance)
    var_f         : variance used to draw F_raw
    uniq          : row norm of F is sqrt(1 - uniq)

    Returns
    -------
    dict with X, Y, Z, F, sigma, w0, w, sigma_y, var_z, var_f, and diagnostics.
    `var_f` is returned because `LatentStructuralModel.initialize_params`
    reads `true_params["var_f"]`; returning 5.0 keeps estimator initialisation
    byte-identical to the historical fixed official Exp1 setting.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if sigma_y_true < 0.0:
        raise ValueError(f"sigma_y_true must be >= 0, got {sigma_y_true}")

    rng = np.random.default_rng(seed)
    var_z = 1.0

    # ---- Z: iid N(0, I_k).  NO STANDARDIZATION OF ANY KIND. ----------------
    Z = rng.normal(0.0, 1.0, size=(n, k))

    # ---- F: row-rescaled Gaussian (positive diagonal => rank preserved) -----
    sigma_x = np.diag(np.full(d, uniq))
    F = rng.normal(0.0, np.sqrt(var_f), size=(d, k))
    for j in range(d):
        nrm = np.linalg.norm(F[j, :])
        if nrm > 0:
            F[j, :] = (F[j, :] / nrm) * np.sqrt(1.0 - uniq)

    # ---- X: Bernoulli, canonical logit link, no clipping, no normalisation --
    eta_x = Z @ F.T                      # (n, d)
    prob_x = expit(eta_x)                # numerically stable exact sigmoid
    X = rng.binomial(1, prob_x).astype(np.float64)

    # ---- Y: Gaussian additive noise on the upper triangle -------------------
    eta_y = w0_true + w_true * (Z @ Z.T)          # (n, n)
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    n_pairs = int(upper_mask.sum())

    Y = np.zeros((n, n), dtype=np.float64)
    noise_y = rng.normal(0.0, sigma_y_true, size=n_pairs)
    Y[upper_mask] = eta_y[upper_mask] + noise_y
    Y = Y + Y.T
    np.fill_diagonal(Y, 0.0)

    upper_Y = Y[upper_mask]

    # ---- diagnostics (descriptive only; never used as gates) ---------------
    sv_Z = np.linalg.svd(Z, compute_uv=False)
    sv_F = np.linalg.svd(F, compute_uv=False)
    z_col_mean = Z.mean(axis=0)
    z_col_var = Z.var(axis=0, ddof=0)
    z_fro_sq = float(np.sum(Z ** 2))

    return {
        # data
        "X": X, "Y": Y, "Z": Z, "F": F,
        "sigma": sigma_x,
        # true parameters
        "w0": w0_true, "w": w_true, "sigma_y": sigma_y_true,
        "var_z": var_z, "var_f": var_f, "uniq": uniq,
        # labels
        "family_x": "bernoulli", "family_y": "gaussian",
        "generator_id": GENERATOR_ID,
        "K_GEN": k, "n": n, "d": d, "seed": seed,
        # generator diagnostics
        "rank_Z": int(np.linalg.matrix_rank(Z)),
        "sigma_min_Z": float(sv_Z.min()),
        "sigma_max_Z": float(sv_Z.max()),
        "rank_F": int(np.linalg.matrix_rank(F)),
        "sigma_min_F": float(sv_F.min()),
        "sigma_max_F": float(sv_F.max()),
        "condition_number_F": float(sv_F.max() / sv_F.min()) if sv_F.min() > 0 else float("inf"),
        "z_col_mean_max_abs": float(np.max(np.abs(z_col_mean))),
        "z_col_var_min": float(z_col_var.min()),
        "z_col_var_max": float(z_col_var.max()),
        "z_frobenius_sq": z_fro_sq,
        "z_frobenius_sq_over_nK": float(z_fro_sq / (n * k)),
        "x_mean": float(X.mean()),
        "y_mean": float(upper_Y.mean()),
        "y_variance": float(upper_Y.var(ddof=0)),
        "n_pairs": n_pairs,
        "dataset_hash_X": _sha256_array(X),
        "dataset_hash_Y": _sha256_array(Y),
        "dataset_hash_Z": _sha256_array(Z),
    }


def first_latent_draw(n: int, k: int, seed: int) -> np.ndarray:
    """
    Reference implementation of the FIRST draw the generator makes.

    Tests assert that `generate_canonical_gaussian_y(...)["Z"]` equals this
    exactly. If any standardization were ever inserted after the draw, this
    equality would fail. This is a code-path test, not a distributional test.
    """
    return np.random.default_rng(seed).normal(0.0, 1.0, size=(n, k))
