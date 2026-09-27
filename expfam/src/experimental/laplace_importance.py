"""Phase 9J (Issue #92): importance-sampling correction to the Laplace integral.

With Phi(Z) = log p(Z, X, Y | theta, K), joint mode Z_hat and
H = −∇²Phi(Z_hat), the Laplace-Gaussian proposal q = N(Z_hat, H⁻¹) gives the
exact identity

    log p(X, Y | theta, K) = laplace_log_observed + log E_q[exp r(Z)],
    r(Z) = Phi(Z) − Phi(Z_hat) + ½ Δᵀ H Δ,   Δ = Z − Z_hat.

This module estimates c = log E_q exp(r) by importance sampling, in log
space, with weight diagnostics. It is a diagnostic of the Laplace
approximation only; it is not a criterion and changes nothing else.

Sampling: H = L Lᵀ (Cholesky), ε ~ N(0, I), Δ = solve(Lᵀ, ε). H⁻¹ is never
formed. H must be positive definite; no ridge or jitter is added.

A sample whose Phi evaluation raises FloatingPointError (the consistent
numerics raise instead of clipping) or returns a non-finite value is counted
in ``non_finite`` and given zero weight (r = −inf). The count is reported.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import laplace_k_criterion as lk                                  # noqa: E402

IS_VERSION = "laplace-importance-v1"


def log_ratio_samples(model, X, Y, Z_hat: np.ndarray, H: np.ndarray,
                      n_samples: int, seed: int) -> dict[str, Any]:
    """Draw n_samples from N(Z_hat, H⁻¹) and return r for each."""

    n, k = Z_hat.shape
    chol = np.linalg.cholesky(H)                   # raises if H is not PD
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n_samples, n * k))
    delta = solve_triangular(chol.T, eps.T, lower=False).T      # Δ = L⁻ᵀ ε
    phi_hat = lk.complete_log_density(model, X, Y, Z_hat)
    r = np.empty(n_samples)
    non_finite = 0
    for s in range(n_samples):
        d = delta[s]
        try:
            phi = lk.complete_log_density(model, X, Y,
                                          Z_hat + d.reshape(n, k))
        except FloatingPointError:
            phi = float("nan")
        if not math.isfinite(phi):
            non_finite += 1
            r[s] = -math.inf
        else:
            r[s] = phi - phi_hat + 0.5 * float(d @ H @ d)
    return {"r": r, "non_finite": non_finite, "seed": int(seed)}


def weight_summary(r: np.ndarray) -> dict[str, Any]:
    """logmeanexp correction and normalised-weight diagnostics."""

    size = r.size
    finite = r[np.isfinite(r)]
    if finite.size == 0:
        return {"correction": -math.inf, "ess": 0.0, "relative_ess": 0.0,
                "max_weight_share": float("nan"), "r_min": float("nan"),
                "r_median": float("nan"), "r_max": float("nan"),
                "n_samples": int(size)}
    lse = float(logsumexp(r))
    w = np.exp(r - lse)
    ess = float(1.0 / np.sum(w ** 2))
    return {"correction": lse - math.log(size), "ess": ess,
            "relative_ess": ess / size,
            "max_weight_share": float(np.max(w)),
            "r_min": float(np.min(finite)),
            "r_median": float(np.median(finite)),
            "r_max": float(np.max(finite)), "n_samples": int(size)}


def importance_correction(model, X, Y, Z_hat: np.ndarray, H: np.ndarray,
                          seeds: list[int], samples_per_batch: int
                          ) -> dict[str, Any]:
    """Per-batch and pooled corrections over fixed independent batches."""

    batches, pooled_r = [], []
    for index, seed in enumerate(seeds, start=1):
        drawn = log_ratio_samples(model, X, Y, Z_hat, H, samples_per_batch,
                                  seed)
        summary = weight_summary(drawn["r"])
        summary.update({"batch_index": index, "seed": seed,
                        "non_finite": drawn["non_finite"]})
        batches.append(summary)
        pooled_r.append(drawn["r"])
    pooled = weight_summary(np.concatenate(pooled_r))
    pooled["non_finite"] = sum(b["non_finite"] for b in batches)
    corrections = np.array([b["correction"] for b in batches])
    return {"batches": batches, "pooled": pooled,
            "batch_correction_stats": {
                "mean": float(np.mean(corrections)),
                "sd": float(np.std(corrections, ddof=1))
                if len(corrections) > 1 else float("nan"),
                "min": float(np.min(corrections)),
                "median": float(np.median(corrections)),
                "max": float(np.max(corrections))},
            "is_version": IS_VERSION}
