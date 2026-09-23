"""Forward-only per-column mixed-X canonical synthetic generator.

``data_generator_canonical.generate_canonical_data`` takes a SCALAR
``family_x`` and therefore cannot produce the heterogeneous attribute matrix
that the Phase 9 family-selection pilot needs.  This module adds a mixed-X
draw WITHOUT touching that file: ``generate_canonical_data`` stays
byte-identical, and its results stay valid as observations on the data it
produced.

The sampling law is the same canonical model, now with a per-column family::

    z_i             ~ N(0, I_K)
    f_l             free, rank(F) = K by construction
    x_il | z_i      ~ ExpFam_{c_l}( eta^X_il = f_l^T z_i )     c_l = family_x_list[l]
    y_ij | z_i,z_j  ~ ExpFam_Y( eta^Y_ij = w0 + w z_i^T z_j ), i < j

Design rules inherited from the canonical generator (each maps to a finding in
``reports/identifiability/canonical_clean_generator_spec_20260904.md``):

* G1  Z is drawn iid N(0, I_K) and NEVER normalised afterwards.
* G2  F is a free parameter shared by every column; rank(F) = K is guaranteed
      by construction, never by reseeding.
* G3  Gaussian columns are N(f_l^T z, sigma_x_var_l) and are NEVER z-scored.
* G4  Poisson columns use the canonical exp link with NO clipping.  An unsafe
      rate stops the generator instead of being silently truncated.
* G5  The declared per-column X dispersion is actually used and recorded.
* G7  ``sigma_x_var`` is a VARIANCE, ``sigma_y_sd`` is a STANDARD DEVIATION.

Rules specific to the mixed draw:

* M1  Z and F are shared by all columns.  The per-column family changes only
      the observation law, never the latent structure.
* M2  The RNG is consumed in the frozen order Z, F, then X COLUMN BY COLUMN in
      ascending column index, then Y.  ``rng_consumption_order`` records this,
      so the same seed and configuration reproduce the same arrays bit for bit.
* M3  Seed rescue is forbidden.  If a configuration is unsafe the generator
      raises ``GeneratorStop``; a caller must never catch it and retry with a
      different seed.

This module performs NO inference and imports no model class.

Design: ``reports/distribution_selection/automatic_family_selection_design_20260923.md``
Protocol: GitHub Issue #74 (Phase 9C), Scope A3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from data_generator_canonical import (  # noqa: E402
    DEFAULT_POISSON_LAMBDA_MAX,
    VALID_FAMILIES,
    GeneratorStop,
    build_full_rank_loadings,
    canonical_poisson_rate,
    canonical_sigmoid,
    poisson_y_moment_existence,
)

MIXED_GENERATOR_VERSION = "canonical-clean-mixed-v1"

# X is consumed one column at a time, in ascending column index.  Spelling the
# per-column step out here (rather than a bare "X") is what makes the mixed
# draw reproducible: a reader can tell from the metadata alone in which order
# the column draws happened.
MIXED_RNG_CONSUMPTION_ORDER = ("Z", "F", "X_columns_ascending", "Y")

REQUIRED_MIXED_METADATA_KEYS = (
    "generator_version", "family_x_list", "family_y", "n", "d", "K_true",
    "F_rank", "f_scale", "f_row_norms_sq", "sigma_x_var", "sigma_y_sd",
    "w0", "w", "seed", "rng_consumption_order", "link_policy",
    "normalization_policy", "diagonal_policy", "poisson_lambda_max",
    "moment_existence", "expected_x_mean", "family_x_counts",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeneratorStop(message)


def _finite(array: np.ndarray, name: str) -> np.ndarray:
    if not np.all(np.isfinite(array)):
        raise GeneratorStop(f"{name} contains a non-finite value")
    return array


@dataclass
class MixedCanonicalDataset:
    """A literal draw from the canonical model with per-column X families."""

    Z: np.ndarray
    F: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"Z": self.Z, "F": self.F, "X": self.X, "Y": self.Y,
                "metadata": dict(self.metadata)}


def generate_canonical_mixed_data(
    *,
    n: int,
    d: int,
    k: int,
    seed: int,
    family_x_list: Sequence[str],
    family_y: str = "bernoulli",
    f_scale: float = 0.6,
    singular_values: Sequence[float] | None = None,
    sigma_x_var: float | Sequence[float] = 1.0,
    w0: float = -1.0,
    w: float = 1.0,
    sigma_y_sd: float = 1.0,
    poisson_lambda_max: float = DEFAULT_POISSON_LAMBDA_MAX,
    allow_infinite_variance: bool = False,
) -> MixedCanonicalDataset:
    """Draw (Z, F, X, Y) from the canonical model with a per-column X family.

    Parameters
    ----------
    family_x_list
        Length-``d`` sequence over ``{gaussian, bernoulli, poisson}``.  Column
        ``l`` is drawn from ``family_x_list[l]``.
    sigma_x_var
        VARIANCE of the Gaussian-X noise, scalar or length ``d``.  Only the
        entries of Gaussian columns are used; entries of non-Gaussian columns
        are recorded as ``None`` so that nothing implies a dispersion the draw
        did not use.
    sigma_y_sd
        STANDARD DEVIATION of the Gaussian-Y noise.

    Raises
    ------
    GeneratorStop
        On any unsafe or malformed configuration.  Never caught-and-retried:
        that would be seed rescue.
    """

    family_x_list = [str(f) for f in family_x_list]
    _require(len(family_x_list) == d,
             f"family_x_list length {len(family_x_list)} != d={d}")
    for index, family in enumerate(family_x_list):
        _require(family in VALID_FAMILIES,
                 f"family_x_list[{index}]={family!r} is not one of {VALID_FAMILIES}")
    _require(family_y in VALID_FAMILIES,
             f"family_y must be one of {VALID_FAMILIES}")
    _require(n >= 2, f"n must be at least 2 to have a dyad: n={n}")
    _require(d >= 1, f"d must be positive: d={d}")
    _require(d >= k, f"d >= K is required: d={d}, K={k}")

    sigma_x_vector = np.broadcast_to(
        np.asarray(sigma_x_var, dtype=np.float64), (d,)).copy()
    gaussian_columns = [l for l, f in enumerate(family_x_list) if f == "gaussian"]
    _require(bool(np.all(sigma_x_vector[gaussian_columns] > 0.0)),
             f"every Gaussian column needs a strictly positive variance: {sigma_x_var}")

    moment_existence = poisson_y_moment_existence(w)
    if family_y == "poisson":
        _require(moment_existence["mean_finite"],
                 f"canonical Poisson-Y needs |w| < 1 for a finite mean; w={w}")
        if not allow_infinite_variance:
            _require(
                moment_existence["variance_finite"],
                f"canonical Poisson-Y needs |w| < 1/2 for a finite variance; "
                f"w={w}. Pass allow_infinite_variance=True only deliberately.",
            )

    rng = np.random.default_rng(seed)

    # --- 1. Z: iid N(0, I_K).  NEVER normalised (G1). ---------------------
    latent = rng.standard_normal((n, k))
    _finite(latent, "Z")

    # --- 2. F: shared across columns, full rank by construction (G2, M1). --
    loadings = build_full_rank_loadings(
        rng, d=d, k=k, f_scale=f_scale, singular_values=singular_values)
    row_norms_sq = np.sum(loadings ** 2, axis=1)

    # --- 3. X: one column at a time, ascending column index (M2). ---------
    eta_x = latent @ loadings.T
    _finite(eta_x, "eta_x")

    attributes = np.empty((n, d), dtype=np.float64)
    for column in range(d):
        family = family_x_list[column]
        eta_column = eta_x[:, column]
        if family == "gaussian":
            noise = rng.standard_normal(n) * math.sqrt(float(sigma_x_vector[column]))
            attributes[:, column] = eta_column + noise      # NOT z-scored (G3)
        elif family == "bernoulli":
            attributes[:, column] = rng.binomial(
                1, canonical_sigmoid(eta_column)).astype(np.float64)
        else:                                               # poisson
            rate = canonical_poisson_rate(eta_column, lambda_max=poisson_lambda_max)
            attributes[:, column] = rng.poisson(rate).astype(np.float64)
    _finite(attributes, "X")

    # --- 4. Y: upper triangle only, symmetrised for storage. --------------
    gram = latent @ latent.T
    eta_y_full = w0 + w * gram
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    eta_y = eta_y_full[upper]
    _finite(eta_y, "eta_y")

    relations = np.zeros((n, n), dtype=np.float64)
    if family_y == "bernoulli":
        relations[upper] = rng.binomial(1, canonical_sigmoid(eta_y)).astype(np.float64)
    elif family_y == "poisson":
        rate_y = canonical_poisson_rate(eta_y, lambda_max=poisson_lambda_max)
        relations[upper] = rng.poisson(rate_y).astype(np.float64)
    else:                                                   # gaussian
        _require(float(sigma_y_sd) > 0.0,
                 f"sigma_y_sd must be strictly positive: {sigma_y_sd}")
        relations[upper] = eta_y + rng.standard_normal(eta_y.shape) * float(sigma_y_sd)
    relations = relations + relations.T
    np.fill_diagonal(relations, 0.0)
    _finite(relations, "Y")

    dataset = MixedCanonicalDataset(
        Z=latent, F=loadings, X=attributes, Y=relations,
        metadata={
            "generator_version": MIXED_GENERATOR_VERSION,
            "family_x_list": list(family_x_list),
            "family_x_counts": {fam: int(family_x_list.count(fam))
                                for fam in VALID_FAMILIES},
            "family_y": family_y,
            "n": int(n), "d": int(d), "K_true": int(k),
            "F_rank": int(np.linalg.matrix_rank(loadings)),
            "f_scale": float(f_scale),
            "singular_values": (None if singular_values is None
                                else [float(v) for v in singular_values]),
            "f_row_norms_sq": [float(v) for v in row_norms_sq],
            # Only Gaussian columns actually used a variance.  Recording None
            # elsewhere keeps the metadata from implying a dispersion that the
            # draw never consumed.
            "sigma_x_var": [float(sigma_x_vector[l]) if family_x_list[l] == "gaussian"
                            else None for l in range(d)],
            "sigma_y_sd": (float(sigma_y_sd) if family_y == "gaussian" else None),
            "w0": float(w0), "w": float(w),
            "seed": int(seed),
            "rng_consumption_order": list(MIXED_RNG_CONSUMPTION_ORDER),
            "link_policy": "canonical_no_clipping_fail_fast",
            "normalization_policy": "none",
            "diagonal_policy": "Y_ii is outside the observation model; stored as 0",
            "poisson_lambda_max": float(poisson_lambda_max),
            "moment_existence": moment_existence,
            "allow_infinite_variance": bool(allow_infinite_variance),
            # For a canonical Poisson column, E[X_l] = exp(||f_l||^2 / 2)
            # (theory audit 7.1).  Undefined for the other families.
            "expected_x_mean": [
                float(math.exp(row_norms_sq[l] / 2.0))
                if family_x_list[l] == "poisson" else None
                for l in range(d)
            ],
        },
    )
    validate_mixed_canonical_dataset(dataset)
    return dataset


def validate_mixed_canonical_dataset(dataset: MixedCanonicalDataset) -> None:
    """Fail-fast structural and per-column support validation."""

    meta = dataset.metadata
    n, d, k = int(meta["n"]), int(meta["d"]), int(meta["K_true"])
    family_x_list = list(meta["family_x_list"])

    _require(dataset.Z.shape == (n, k), f"Z shape {dataset.Z.shape} != {(n, k)}")
    _require(dataset.F.shape == (d, k), f"F shape {dataset.F.shape} != {(d, k)}")
    _require(dataset.X.shape == (n, d), f"X shape {dataset.X.shape} != {(n, d)}")
    _require(dataset.Y.shape == (n, n), f"Y shape {dataset.Y.shape} != {(n, n)}")
    for name, array in (("Z", dataset.Z), ("F", dataset.F),
                        ("X", dataset.X), ("Y", dataset.Y)):
        _finite(array, name)

    _require(int(np.linalg.matrix_rank(dataset.F)) == k,
             "rank(F) != K in the produced dataset")
    _require(bool(np.array_equal(dataset.Y, dataset.Y.T)), "Y is not symmetric")
    _require(bool(np.all(np.diag(dataset.Y) == 0.0)), "Y has a nonzero diagonal")

    _require(len(family_x_list) == d,
             f"family_x_list length {len(family_x_list)} != d={d}")
    for column, family in enumerate(family_x_list):
        _require_column_support(dataset.X[:, column], family, f"X[:, {column}]")

    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    _require_column_support(dataset.Y[upper], meta["family_y"], "Y")

    for column, family in enumerate(family_x_list):
        if family == "gaussian":
            variance = meta["sigma_x_var"][column]
            _require(variance is not None and variance > 0.0,
                     f"Gaussian column {column} needs a positive sigma_x_var")
    if meta["family_y"] == "gaussian":
        _require(meta["sigma_y_sd"] is not None and meta["sigma_y_sd"] > 0.0,
                 "Gaussian-Y requires a strictly positive sigma_y_sd")

    _require(d >= k, f"d >= K violated in the produced dataset: d={d}, K={k}")

    if meta["family_y"] == "poisson" and not meta["allow_infinite_variance"]:
        _require(bool(meta["moment_existence"]["variance_finite"]),
                 "Poisson-Y variance is not finite and it was not allowed")

    missing = [key for key in REQUIRED_MIXED_METADATA_KEYS if key not in meta]
    _require(not missing, f"metadata is missing keys: {missing}")


def _require_column_support(values: np.ndarray, family: str, name: str) -> None:
    flat = np.asarray(values, dtype=np.float64).ravel()
    if family == "bernoulli":
        _require(bool(np.all((flat == 0.0) | (flat == 1.0))),
                 f"{name} is Bernoulli but has a value outside {{0, 1}}")
    elif family == "poisson":
        _require(bool(np.all(flat >= 0.0)),
                 f"{name} is Poisson but has a negative value")
        _require(bool(np.all(flat == np.floor(flat))),
                 f"{name} is Poisson but has a non-integer value")
    # gaussian: any finite real is in support; finiteness already checked.


__all__ = [
    "MIXED_GENERATOR_VERSION",
    "MIXED_RNG_CONSUMPTION_ORDER",
    "REQUIRED_MIXED_METADATA_KEYS",
    "MixedCanonicalDataset",
    "generate_canonical_mixed_data",
    "validate_mixed_canonical_dataset",
]
