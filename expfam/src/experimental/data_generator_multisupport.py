"""Forward-only multi-support synthetic generator (canonical model, per-column spec).

Same sampling law and rules as ``data_generator_canonical_mixed`` (unchanged):

    z_i            ~ N(0, I_K)                       (never normalised)
    F              = Q diag(f_scale) row-scaled, rank K by construction
    x_il | z_i     ~ ExpFam_{m_l}(eta_il = f_l^T z_i)   canonical, no intercept
    y_ij | z_i,z_j ~ Bernoulli(sigmoid(w0 + w z_i^T z_j)), i < j

RNG consumption order is frozen: Z, F, X column by column (ascending), Y.

Information-matched loadings
----------------------------
Families have very different natural-parameter scales (A''(0) is 1/12 for the
Continuous Bernoulli and ~2.2 for COM-Poisson nu=0.5).  Using one f_scale for
all columns would hand some families almost no latent signal.  Each row is
therefore scaled by ``1 / sqrt(A''_{m_l}(0))``, so that the Fisher information
about eta at eta = 0 carried by every column is the same target ``tau`` on
average:  E_l[ ||f_l||^2 ] * A''_{m_l}(0) = tau.  ``tau`` is the difficulty
knob (frozen per support group by ``difficulty_calibration`` from distribution
distances only).

No seed rescue: any unsafe draw raises ``GeneratorStop``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from data_generator_canonical import (
    GeneratorStop, build_full_rank_loadings, canonical_sigmoid,
)
from family_specs import FamilySpec, FAMILY_SPECS_VERSION

MULTISUPPORT_GENERATOR_VERSION = "canonical-multisupport-v1"
RNG_ORDER = ("Z", "F", "X_columns_ascending", "Y")


@dataclass
class MultiSupportDataset:
    Z: np.ndarray
    F: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def generate_multisupport_data(*, n: int, k: int, seed: int,
                               column_specs: Sequence[FamilySpec],
                               tau: float | Sequence[float],
                               w0: float = -1.0, w: float = 1.0
                               ) -> MultiSupportDataset:
    specs = list(column_specs)
    d = len(specs)
    if d < k:
        raise GeneratorStop(f"d >= K required: d={d}, K={k}")
    tau_vec = np.broadcast_to(np.asarray(tau, dtype=np.float64), (d,)).copy()
    if not np.all(tau_vec > 0):
        raise GeneratorStop("tau must be positive")
    rng = np.random.default_rng(seed)

    Z = rng.standard_normal((n, k))
    base = build_full_rank_loadings(rng, d=d, k=k, f_scale=1.0)
    info0 = np.array([s.info_at_zero() for s in specs])
    if not np.all(np.isfinite(info0) & (info0 > 0)):
        raise GeneratorStop("A''(0) must be finite and positive for every spec")
    # average ||q_l||^2 = K/d, so this scale gives E_l ||f_l||^2 A''(0) = tau
    row_scale = np.sqrt(tau_vec * d / k / info0)
    F = base * row_scale[:, None]
    if int(np.linalg.matrix_rank(F)) != k:
        raise GeneratorStop("rank(F) != K after row scaling")

    eta = Z @ F.T
    X = np.empty((n, d))
    for col, spec in enumerate(specs):
        try:
            X[:, col] = spec.sample(eta[:, col], rng)
        except FloatingPointError as exc:
            raise GeneratorStop(f"column {col} ({spec.candidate_model}): {exc}") from exc
        if not spec.in_support(X[:, col]):
            raise GeneratorStop(f"column {col} left its support")

    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    eta_y = (w0 + w * (Z @ Z.T))[upper]
    Y = np.zeros((n, n))
    Y[upper] = rng.binomial(1, canonical_sigmoid(eta_y)).astype(np.float64)
    Y = Y + Y.T
    np.fill_diagonal(Y, 0.0)
    density = float(Y[upper].mean())
    if not (0.0 < density < 1.0):
        raise GeneratorStop(f"degenerate Y density {density}")

    meta = {
        "generator_version": MULTISUPPORT_GENERATOR_VERSION,
        "family_specs_version": FAMILY_SPECS_VERSION,
        "n": n, "d": d, "K_true": k, "seed": int(seed),
        "rng_consumption_order": list(RNG_ORDER),
        "true_candidate_models": [s.candidate_model for s in specs],
        "true_family_names": [s.family_name for s in specs],
        "true_support_groups": [s.support_group for s in specs],
        "tau": [float(t) for t in tau_vec],
        "info_at_zero": [float(v) for v in info0],
        "row_scale": [float(v) for v in row_scale],
        "f_row_norms_sq": [float(v) for v in np.sum(F ** 2, axis=1)],
        "eta_min": [float(v) for v in eta.min(axis=0)],
        "eta_max": [float(v) for v in eta.max(axis=0)],
        "family_y": "bernoulli", "w0": float(w0), "w": float(w),
        "y_density": density,
        "link_policy": "canonical natural parameter eta = f^T z; no intercept",
        "normalization_policy": "none",
    }
    return MultiSupportDataset(Z=Z, F=F, X=X, Y=Y, metadata=meta)


__all__ = ["MULTISUPPORT_GENERATOR_VERSION", "MultiSupportDataset",
           "generate_multisupport_data", "GeneratorStop"]
