"""
Paired per-column-X / canonical-Gaussian-Y generator (2026-09-07).

PURPOSE
-------
Vary ONLY the per-column family composition of X while holding the latent
structure and the whole Y channel fixed, so that differences in K-selection
cannot be attributed to a different Z, F or Y.

    z_i     ~ iid N(0, I_K)                        <- NO post-processing
    F       ~ row-rescaled Gaussian, ||f_l|| = sqrt(1 - uniq)
    eta^X   = Z F^T
    x_il    ~ family_x_list[l] with natural parameter eta^X_il
    eta^Y   = w0 + w z_i^T z_j
    y_ij    = eta^Y_ij + eps_ij,  eps ~ iid N(0, sigma_y^2),  i < j

Because Y is canonical Gaussian with w != 0, the single-dyad marginal of Y
determines K uniquely, so K_0 = K_GEN regardless of how X's columns are
composed. See gaussian_y_k_identifiability_adversarial_audit_20260907.md.

PAIRED DESIGN (the reason this file exists rather than reusing the core one)
---------------------------------------------------------------------------
The canonical core generator draws Z, F, X, Y from ONE rng stream in that
order, so changing X's family would shift the stream and perturb Y. Here two
independent streams are used:

    rng_core = default_rng(seed)                 -> Z, F, Y      (family-independent)
    rng_x    = default_rng(seed + X_STREAM_OFFSET) -> X columns  (family-dependent)

Consequence: for a fixed (n, d, K, seed) every family configuration receives
BIT-IDENTICAL Z, F and Y. Only X differs. This is asserted at run time by
hashing Z, F and Y.

NO STANDARDIZATION of Z, X or Y is performed anywhere.

X-side parameterization follows the existing historical convention and
introduces nothing new:
    Gaussian columns : x ~ N(eta, sigma_x2), sigma_x2 = uniq  (= 0.1)
                       matches ||f_l||^2 = 1 - uniq so signal+noise = 1
    Bernoulli columns: x ~ Bernoulli(sigmoid(eta))
    Poisson columns  : x ~ Poisson(exp(eta))    -- NO clipping
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Sequence

import numpy as np
from scipy.special import expit

GENERATOR_ID = "percolumn-x-canonical-gaussian-y-v1"
X_STREAM_OFFSET = 500_000
VALID_FAMILIES = ("gaussian", "bernoulli", "poisson")


def _sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def make_family_list(spec: Sequence[tuple[str, int]], d: int) -> list[str]:
    """Blocked column assignment, e.g. [("bernoulli",8),("gaussian",7)]."""
    out: list[str] = []
    for fam, cnt in spec:
        if fam not in VALID_FAMILIES:
            raise ValueError(f"bad family {fam}")
        out.extend([fam] * cnt)
    if len(out) != d:
        raise ValueError(f"family spec sums to {len(out)}, expected d={d}")
    return out


def generate_percolumn_gaussian_y(
    n: int,
    d: int,
    k: int,
    seed: int,
    family_x_list: Sequence[str],
    *,
    w0_true: float = 0.0,
    w_true: float = 0.5,
    sigma_y_true: float = 0.1,
    var_f: float = 5.0,
    uniq: float = 0.1,
) -> Dict[str, Any]:
    """One dataset. Z, F and Y depend only on (n, d, k, seed) -- not on families."""
    if len(family_x_list) != d:
        raise ValueError(f"family_x_list length {len(family_x_list)} != d={d}")
    for f in family_x_list:
        if f not in VALID_FAMILIES:
            raise ValueError(f"Unsupported family '{f}'")

    # ---- stream 1: latent structure and Y (family-independent) -------------
    rng_core = np.random.default_rng(seed)
    var_z = 1.0
    Z = rng_core.normal(0.0, 1.0, size=(n, k))          # NO standardization

    F = rng_core.normal(0.0, np.sqrt(var_f), size=(d, k))
    for j in range(d):
        nrm = np.linalg.norm(F[j, :])
        if nrm > 0:
            F[j, :] = (F[j, :] / nrm) * np.sqrt(1.0 - uniq)

    eta_y = w0_true + w_true * (Z @ Z.T)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    n_pairs = int(upper.sum())
    Y = np.zeros((n, n), dtype=np.float64)
    Y[upper] = eta_y[upper] + rng_core.normal(0.0, sigma_y_true, size=n_pairs)
    Y = Y + Y.T
    np.fill_diagonal(Y, 0.0)

    # ---- stream 2: X (family-dependent, cannot disturb stream 1) -----------
    rng_x = np.random.default_rng(seed + X_STREAM_OFFSET)
    eta_x = Z @ F.T                                     # (n, d)
    sigma_x2 = float(uniq)
    X = np.empty((n, d), dtype=np.float64)
    for l, fam in enumerate(family_x_list):
        e = eta_x[:, l]
        if fam == "gaussian":
            X[:, l] = e + rng_x.normal(0.0, np.sqrt(sigma_x2), size=n)
        elif fam == "bernoulli":
            X[:, l] = rng_x.binomial(1, expit(e)).astype(np.float64)
        else:                                            # poisson, NO clipping
            X[:, l] = rng_x.poisson(np.exp(e)).astype(np.float64)

    # ---- diagnostics (descriptive only; never used as gates) ---------------
    counts = {f: int(sum(1 for x in family_x_list if x == f)) for f in VALID_FAMILIES}
    pois_cols = [l for l, f in enumerate(family_x_list) if f == "poisson"]
    sv_Z = np.linalg.svd(Z, compute_uv=False)
    sv_F = np.linalg.svd(F, compute_uv=False)
    upper_Y = Y[upper]

    return {
        "X": X, "Y": Y, "Z": Z, "F": F,
        "sigma": np.diag(np.full(d, sigma_x2)),
        "w0": w0_true, "w": w_true, "sigma_y": sigma_y_true,
        "var_z": var_z, "var_f": var_f, "uniq": uniq,
        "family_x_list": list(family_x_list),
        "family_y": "gaussian",
        "n_gaussian_x_cols": counts["gaussian"],
        "n_bernoulli_x_cols": counts["bernoulli"],
        "n_poisson_x_cols": counts["poisson"],
        "generator_id": GENERATOR_ID,
        "K_GEN": k, "n": n, "d": d, "seed": seed,
        # generator diagnostics
        "rank_Z": int(np.linalg.matrix_rank(Z)),
        "sigma_min_Z": float(sv_Z.min()), "sigma_max_Z": float(sv_Z.max()),
        "rank_F": int(np.linalg.matrix_rank(F)),
        "sigma_min_F": float(sv_F.min()), "sigma_max_F": float(sv_F.max()),
        "eta_x_min": float(eta_x.min()), "eta_x_max": float(eta_x.max()),
        "eta_x_abs_max": float(np.abs(eta_x).max()),
        # Poisson safety: the consistent numerics RAISE (never clip) on overflow,
        # and the legacy clip window is [-20, 10]; record how close we get.
        "poisson_eta_max": float(eta_x[:, pois_cols].max()) if pois_cols else np.nan,
        "poisson_eta_min": float(eta_x[:, pois_cols].min()) if pois_cols else np.nan,
        "poisson_eta_outside_legacy_clip": int(
            np.sum((eta_x[:, pois_cols] < -20) | (eta_x[:, pois_cols] > 10))
        ) if pois_cols else 0,
        "poisson_count_max": float(X[:, pois_cols].max()) if pois_cols else np.nan,
        "poisson_count_mean": float(X[:, pois_cols].mean()) if pois_cols else np.nan,
        "x_mean": float(X.mean()),
        "y_mean": float(upper_Y.mean()), "y_variance": float(upper_Y.var(ddof=0)),
        "n_pairs": n_pairs,
        "dataset_hash_X": _sha256_array(X),
        "dataset_hash_Y": _sha256_array(Y),
        "dataset_hash_Z": _sha256_array(Z),
        "dataset_hash_F": _sha256_array(F),
    }
