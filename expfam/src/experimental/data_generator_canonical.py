"""Forward-only canonical clean synthetic generator.

This module generates data that is a LITERAL draw from the canonical model

    z_i            ~ N(0, I_K)
    x_il | z_i     ~ ExpFam_X( eta^X_il = f_l^T z_i )
    y_ij | z_i,z_j ~ ExpFam_Y( eta^Y_ij = w0 + w z_i^T z_j ),   i < j

with no post-hoc normalisation, no row-normalised F, no hidden clipping and
explicit dispersion semantics.

It is NOT a replacement for ``expfam/src/data_generator_expfam.py``.  That
historical generator is left byte-identical on purpose; its results remain
valid as observations on the data it produced.  What this module adds is a
generator whose sampling law matches the model the estimator assumes, so that
"well-specified" is a statement about the data and not only about the code.

Specification: ``reports/identifiability/canonical_clean_generator_spec_20260904.md``
Theory:        ``reports/identifiability/true_k_identifiability_hardened_20260904.md``

Design rules enforced here (each maps to a finding in the theory audit):

* G1  Z is drawn iid N(0, I_K) and NEVER normalised afterwards.
* G2  F is a free parameter; its rows are NOT renormalised to a fixed norm.
      rank(F) = K is guaranteed by construction, never by reseeding.
* G3  Gaussian X is N(F z, Sigma_X) and is NEVER z-scored afterwards.
* G4  Poisson uses the canonical exp link with NO clipping.  Unsafe rates stop
      the generator instead of being silently truncated.
* G5  The declared X dispersion is actually used, and is recorded in metadata.
* G7  Dispersion semantics are encoded in the argument names:
      ``sigma_x_var`` is a VARIANCE, ``sigma_y_sd`` is a STANDARD DEVIATION.

This module performs NO inference and imports no model class.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

GENERATOR_VERSION = "canonical-clean-v1"
VALID_FAMILIES = ("gaussian", "bernoulli", "poisson")
RNG_CONSUMPTION_ORDER = ("Z", "F", "X", "Y")
DEFAULT_POISSON_LAMBDA_MAX = 1.0e6


class GeneratorStop(RuntimeError):
    """Fail-fast stop.

    Raised instead of clipping, truncating, reseeding or otherwise silently
    repairing an unsafe configuration.  A caller must never catch this and
    retry with a different seed: that is seed rescue.
    """


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeneratorStop(message)


def _finite(array: np.ndarray, name: str) -> np.ndarray:
    if not np.all(np.isfinite(array)):
        raise GeneratorStop(f"{name} contains a non-finite value")
    return array


# --------------------------------------------------------------------------
# canonical links -- stable evaluation only, never a change of model
# --------------------------------------------------------------------------

def canonical_sigmoid(eta: np.ndarray) -> np.ndarray:
    """sigmoid(eta), evaluated by the branch that cannot overflow.

    Algebraically identical to 1/(1+exp(-eta)) everywhere; the two branches
    exist only so that exp() is never called on a large positive argument.
    This is NOT clipping: no value of eta is altered.
    """

    eta = _finite(np.asarray(eta, dtype=np.float64), "Bernoulli eta")
    out = np.empty_like(eta)
    positive = eta >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-eta[positive]))
    exp_eta = np.exp(eta[~positive])
    out[~positive] = exp_eta / (1.0 + exp_eta)
    return out


def canonical_poisson_rate(eta: np.ndarray, *, lambda_max: float) -> np.ndarray:
    """exp(eta) with NO clipping; unsafe values stop the generator.

    ``lambda_max`` is a gate, not a ceiling: exceeding it raises rather than
    truncates, so a configuration that would need clipping is rejected at the
    design stage instead of silently producing non-canonical data.
    """

    eta = _finite(np.asarray(eta, dtype=np.float64), "Poisson eta")
    log_max = math.log(np.finfo(np.float64).max)
    largest = float(np.max(eta)) if eta.size else -math.inf
    _require(
        largest <= log_max,
        f"Poisson exp(eta) would overflow float64: max eta={largest}, "
        f"log(finfo.max)={log_max}. The generator refuses to clip.",
    )
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            rate = np.exp(eta)
        except FloatingPointError as exc:      # pragma: no cover - guarded above
            raise GeneratorStop(f"Poisson exp(eta) failed: {exc}") from exc
    _finite(rate, "Poisson rate")
    observed_max = float(np.max(rate)) if rate.size else 0.0
    _require(
        observed_max <= lambda_max,
        f"Poisson rate exceeds the configured gate: max lambda={observed_max} "
        f"> lambda_max={lambda_max}. The generator refuses to clip; lower "
        f"f_scale, |w| or |w0| instead.",
    )
    return rate


# --------------------------------------------------------------------------
# parameter construction
# --------------------------------------------------------------------------

def build_full_rank_loadings(rng: np.random.Generator, *, d: int, k: int,
                             f_scale: float,
                             singular_values: Sequence[float] | None = None,
                             ) -> np.ndarray:
    """F in R^{d x K} with rank exactly K, guaranteed by construction.

    A Gaussian matrix is orthonormalised by a reduced QR and then scaled.  The
    column space is K-dimensional whenever the scaling is nonzero, so the rank
    does not depend on luck.  If the check still fails the generator STOPS --
    it never redraws with another seed.
    """

    _require(d >= k, f"d >= K is required for a full-rank F: d={d}, K={k}")
    _require(k >= 1, f"K must be positive: K={k}")
    _require(f_scale > 0.0, f"f_scale must be positive: {f_scale}")

    if singular_values is None:
        values = np.full(k, float(f_scale), dtype=np.float64)
    else:
        values = np.asarray(singular_values, dtype=np.float64)
        _require(values.shape == (k,),
                 f"singular_values must have length K={k}, got {values.shape}")
        _require(bool(np.all(values > 0.0)),
                 "every singular value must be strictly positive")

    gaussian = rng.standard_normal((d, k))
    q, r = np.linalg.qr(gaussian, mode="reduced")
    # Fix the sign convention deterministically so the construction is a
    # function of the seed alone.
    diag = np.diag(r).copy()
    _require(bool(np.all(diag != 0.0)),
             "QR produced a zero pivot; the loading construction is degenerate")
    q = q * np.sign(diag)[np.newaxis, :]
    loadings = q * values[np.newaxis, :]
    _finite(loadings, "F")

    rank = int(np.linalg.matrix_rank(loadings))
    _require(
        rank == k,
        f"rank(F)={rank} != K={k} after construction. The generator STOPS; "
        f"redrawing with another seed would be seed rescue.",
    )
    return loadings


def f_scale_for_row_norm(target_row_norm_sq: float, *, d: int, k: int) -> float:
    """f_scale giving an AVERAGE squared row norm of ``target_row_norm_sq``.

    With F = Q diag(f_scale) and Q having orthonormal columns, sum_l ||q_l||^2
    = K, so the average squared row norm is f_scale^2 K / d.  Solving for
    f_scale keeps the per-column attribute signal comparable across K_TRUE
    instead of letting it grow with K.

    For canonical Poisson-X this directly fixes the average expected count,
    since E[X_l] = exp(||f_l||^2 / 2) (theory audit 7.1).
    """

    _require(target_row_norm_sq > 0.0,
             f"target_row_norm_sq must be positive: {target_row_norm_sq}")
    _require(d >= k >= 1, f"need d >= K >= 1: d={d}, K={k}")
    return float(math.sqrt(target_row_norm_sq * d / k))


def w_for_matched_y_signal(w_ref: float, *, k: int, k_ref: int) -> float:
    """w_K = w_ref sqrt(K_ref / K), which holds w^2 K constant.

    Var(z_i . z_j) = K (theory audit 9.2), so Var(w S) = w^2 K.  Without this
    rescaling a larger K_TRUE would automatically carry a stronger relational
    signal, confounding "does the criterion recover K" with "is the signal
    louder".  This is the same variance-matching convention as the Phase 8b
    sensitivity estimand.
    """

    _require(k >= 1 and k_ref >= 1, f"K and K_ref must be positive: {k}, {k_ref}")
    return float(w_ref) * math.sqrt(float(k_ref) / float(k))


def poisson_y_moment_existence(w: float) -> dict[str, Any]:
    """E[Y^r] < inf iff |r w| < 1 for the canonical Poisson-Y.

    See the theory audit, proposition P6.  The mean is finite iff |w| < 1 and
    the variance iff |w| < 1/2; the historical default w = 0.5 sits exactly on
    the variance-divergence boundary.
    """

    absolute = abs(float(w))
    return {
        "abs_w": absolute,
        "mean_finite": absolute < 1.0,
        "variance_finite": absolute < 0.5,
        "highest_finite_moment_order": (
            None if absolute == 0.0 else int(math.floor(1.0 / absolute - 1e-12))
        ),
    }


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalDataset:
    """A literal draw from the canonical model, plus its provenance."""

    Z: np.ndarray
    F: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"Z": self.Z, "F": self.F, "X": self.X, "Y": self.Y,
                "metadata": dict(self.metadata)}


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------

def generate_canonical_data(
    *,
    n: int,
    d: int,
    k: int,
    seed: int,
    family_x: str = "poisson",
    family_y: str = "bernoulli",
    f_scale: float = 0.6,
    singular_values: Sequence[float] | None = None,
    sigma_x_var: float | Sequence[float] = 1.0,
    w0: float = -1.0,
    w: float = 1.0,
    sigma_y_sd: float = 1.0,
    poisson_lambda_max: float = DEFAULT_POISSON_LAMBDA_MAX,
    allow_infinite_variance: bool = False,
) -> CanonicalDataset:
    """Draw (Z, X, Y) literally from the canonical model.

    Parameters whose name carries their semantics
    ---------------------------------------------
    sigma_x_var
        VARIANCE of the Gaussian-X noise (the diagonal of Sigma_X), scalar or
        length-d.  This matches the estimator, whose Gaussian-X term divides
        the squared residual by ``diag(params["sigma"])``.
    sigma_y_sd
        STANDARD DEVIATION of the Gaussian-Y noise.  This matches the model
        classes, which store ``self.sigma_y`` as a standard deviation and
        square it at use.

    The RNG is consumed in the frozen order Z, F, X, Y, so the same seed and
    configuration reproduce the same arrays bit for bit.
    """

    _require(family_x in VALID_FAMILIES, f"family_x must be one of {VALID_FAMILIES}")
    _require(family_y in VALID_FAMILIES, f"family_y must be one of {VALID_FAMILIES}")
    _require(n >= 2, f"n must be at least 2 to have a dyad: n={n}")
    _require(d >= 1, f"d must be positive: d={d}")
    _require(d >= k, f"d >= K is required: d={d}, K={k}")

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

    # --- 1. Z: iid N(0, I_K).  NEVER normalised (G1). --------------------
    latent = rng.standard_normal((n, k))
    _finite(latent, "Z")

    # --- 2. F: free parameter, full rank by construction (G2). -----------
    loadings = build_full_rank_loadings(
        rng, d=d, k=k, f_scale=f_scale, singular_values=singular_values)
    row_norms_sq = np.sum(loadings ** 2, axis=1)

    # --- 3. X --------------------------------------------------------------
    eta_x = latent @ loadings.T
    _finite(eta_x, "eta_x")

    sigma_x_vector: np.ndarray | None = None
    sigma_x_matrix: np.ndarray | None = None
    if family_x == "gaussian":
        sigma_x_vector = np.broadcast_to(
            np.asarray(sigma_x_var, dtype=np.float64), (d,)).copy()
        _require(bool(np.all(sigma_x_vector > 0.0)),
                 f"sigma_x_var must be strictly positive: {sigma_x_var}")
        sigma_x_matrix = np.diag(sigma_x_vector)
        noise = rng.standard_normal((n, d)) * np.sqrt(sigma_x_vector)[np.newaxis, :]
        attributes = eta_x + noise                    # NOT z-scored (G3)
    elif family_x == "bernoulli":
        attributes = rng.binomial(1, canonical_sigmoid(eta_x)).astype(np.float64)
    else:                                             # poisson
        rate_x = canonical_poisson_rate(eta_x, lambda_max=poisson_lambda_max)
        attributes = rng.poisson(rate_x).astype(np.float64)
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
    else:                                             # gaussian
        _require(float(sigma_y_sd) > 0.0,
                 f"sigma_y_sd must be strictly positive: {sigma_y_sd}")
        relations[upper] = eta_y + rng.standard_normal(eta_y.shape) * float(sigma_y_sd)
    relations = relations + relations.T
    np.fill_diagonal(relations, 0.0)
    _finite(relations, "Y")

    dataset = CanonicalDataset(
        Z=latent, F=loadings, X=attributes, Y=relations,
        metadata={
            "generator_version": GENERATOR_VERSION,
            "family_x": family_x,
            "family_y": family_y,
            "n": int(n), "d": int(d), "K_true": int(k),
            "F_rank": int(np.linalg.matrix_rank(loadings)),
            "f_scale": float(f_scale),
            "singular_values": (None if singular_values is None
                                else [float(v) for v in singular_values]),
            "f_row_norms_sq": [float(v) for v in row_norms_sq],
            "Sigma_X": (None if sigma_x_matrix is None
                        else sigma_x_matrix.tolist()),
            "sigma_x_var": (None if sigma_x_vector is None
                            else [float(v) for v in sigma_x_vector]),
            "sigma_y_sd": (float(sigma_y_sd) if family_y == "gaussian" else None),
            "w0": float(w0), "w": float(w),
            "seed": int(seed),
            "rng_consumption_order": list(RNG_CONSUMPTION_ORDER),
            "link_policy": "canonical_no_clipping_fail_fast",
            "normalization_policy": "none",
            "diagonal_policy": "Y_ii is outside the observation model; stored as 0",
            "poisson_lambda_max": float(poisson_lambda_max),
            "moment_existence": moment_existence,
            "allow_infinite_variance": bool(allow_infinite_variance),
            # Theory audit (7.1): for canonical Poisson-X, E[X_l] = exp(||f_l||^2 / 2)
            "expected_x_mean": ([float(math.exp(v / 2.0)) for v in row_norms_sq]
                                if family_x == "poisson" else None),
        },
    )
    validate_canonical_dataset(dataset)
    return dataset


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

REQUIRED_METADATA_KEYS = (
    "generator_version", "family_x", "family_y", "n", "d", "K_true", "F_rank",
    "f_scale", "f_row_norms_sq", "Sigma_X", "sigma_x_var", "sigma_y_sd",
    "w0", "w", "seed", "rng_consumption_order", "link_policy",
    "normalization_policy", "diagonal_policy", "poisson_lambda_max",
    "moment_existence", "expected_x_mean",
)


def validate_canonical_dataset(dataset: CanonicalDataset) -> None:
    """Fail-fast structural and support validation (spec section 2.10)."""

    meta = dataset.metadata
    n, d, k = int(meta["n"]), int(meta["d"]), int(meta["K_true"])

    # V1 / V2 / V3 / V4 -- shape and finiteness
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

    # V5 -- family support
    _require_support(dataset.X, meta["family_x"], "X")
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    _require_support(dataset.Y[upper], meta["family_y"], "Y")

    # V6 -- dispersion positivity
    if meta["family_x"] == "gaussian":
        _require(meta["sigma_x_var"] is not None and
                 all(v > 0.0 for v in meta["sigma_x_var"]),
                 "Gaussian-X requires strictly positive sigma_x_var")
    if meta["family_y"] == "gaussian":
        _require(meta["sigma_y_sd"] is not None and meta["sigma_y_sd"] > 0.0,
                 "Gaussian-Y requires a strictly positive sigma_y_sd")

    # V7 -- d >= K
    _require(d >= k, f"d >= K violated in the produced dataset: d={d}, K={k}")

    # V9 -- Poisson-Y moment existence
    if meta["family_y"] == "poisson" and not meta["allow_infinite_variance"]:
        _require(bool(meta["moment_existence"]["variance_finite"]),
                 "Poisson-Y variance is not finite and it was not allowed")

    # V10 -- metadata completeness
    missing = [key for key in REQUIRED_METADATA_KEYS if key not in meta]
    _require(not missing, f"metadata is missing keys: {missing}")


def _require_support(values: np.ndarray, family: str, name: str) -> None:
    flat = np.asarray(values, dtype=np.float64).ravel()
    if family == "bernoulli":
        _require(bool(np.all((flat == 0.0) | (flat == 1.0))),
                 f"{name} is Bernoulli but has a value outside {{0, 1}}")
    elif family == "poisson":
        _require(bool(np.all(flat >= 0.0)), f"{name} is Poisson but has a negative value")
        _require(bool(np.all(flat == np.floor(flat))),
                 f"{name} is Poisson but has a non-integer value")
    # gaussian: any finite real is in support; finiteness already checked.


__all__ = [
    "GENERATOR_VERSION",
    "VALID_FAMILIES",
    "RNG_CONSUMPTION_ORDER",
    "DEFAULT_POISSON_LAMBDA_MAX",
    "GeneratorStop",
    "CanonicalDataset",
    "canonical_sigmoid",
    "canonical_poisson_rate",
    "build_full_rank_loadings",
    "f_scale_for_row_norm",
    "w_for_matched_y_signal",
    "poisson_y_moment_existence",
    "generate_canonical_data",
    "validate_canonical_dataset",
]
