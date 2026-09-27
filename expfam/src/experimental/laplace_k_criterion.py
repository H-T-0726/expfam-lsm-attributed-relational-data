"""Phase 9H (Issue #88): post-hoc Laplace evaluator for Candidate B.

Candidate B (Phase 9G) approximates the observed-data log likelihood with the
latent coordinates integrated out, at FIXED fitted parameters theta:

    log p(X, Y | theta, K) = log ∫ exp Phi(Z) dZ
                           ≈ Phi(Z_hat) + (nK/2) log(2π) − ½ log|H(Z_hat)|

    Phi(Z) = log p(Z) + log p(X | Z, theta) + log p(Y | Z, theta)
    Z_hat  = joint mode: ∇_Z Phi(Z_hat) = 0   (NOT an E-step sample)
    H      = −∇²_Z Phi(Z_hat)                 (full nK × nK; Y couples nodes)

and the parameter term is kept SEPARATE:

    C_Lap(K) = −2 · laplace_log_observed + d_K · log(N)

This module adds no model. Phi is the existing strict complete-data density
``eval_utils.calc_Q_dual_strict_exp`` evaluated at a single Z (all
normalisation constants, Poisson factorials, the prior with its variance).
Gradient and Hessian use the model's own mean/curvature hooks
(``_mean_function_x``, ``_variance_function_x``, ``_mean_function``,
``_variance_function``).

Nothing here fits, samples or changes the estimator: it is a deterministic
function of (model parameters, X, Y, starting Z).

Supported paths (anything else raises ``UnsupportedLaplacePath``):
- model: ``DualExpFamLSMConsistent`` or ``DualExpFamLSMPerColumnConsistent``
- X columns: gaussian (dispersion = model sigma diagonal), bernoulli, poisson
- Y: bernoulli or poisson with canonical eta_ij = w0 + w z_iᵀ z_j,
  symmetric Y and symmetric pair mask with zero diagonal
Not supported: Gaussian Y, NB, legacy (non-consistent) models.

Lineage E (experimental prototype; not adoptable for the manuscript).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from eval_utils import calc_Q_dual_strict_exp                    # noqa: E402
from model_dual_expfam_consistent import (                       # noqa: E402
    DualExpFamLSMConsistent,
    DualExpFamLSMPerColumnConsistent,
)

EVALUATOR_VERSION = "laplace-k-criterion-v1"
SUPPORTED_Y = ("bernoulli", "poisson")

STATUS_OK = "OK"
STATUS_NOT_STATIONARY = "NOT_STATIONARY"
STATUS_HESSIAN_NOT_PD = "HESSIAN_NOT_PD"


class UnsupportedLaplacePath(ValueError):
    """The model/family combination is outside the verified evaluator."""


# --------------------------------------------------------------------------
# theta and support
# --------------------------------------------------------------------------

def _theta(model) -> dict[str, Any]:
    p = model.params
    return {"F": np.asarray(p["F"], float), "sigma": np.asarray(p["sigma"], float),
            "w0": float(p["w0"]), "w": float(p["w"]),
            "var_z": float(p["var_z"])}


def _check_supported(model, X: np.ndarray, Y: np.ndarray) -> None:
    if not isinstance(model, (DualExpFamLSMConsistent,
                              DualExpFamLSMPerColumnConsistent)):
        raise UnsupportedLaplacePath(
            f"{type(model).__name__} is not a supported model; use the "
            f"objective-consistent classes")
    if model.family not in SUPPORTED_Y:
        raise UnsupportedLaplacePath(
            f"family_y={model.family!r} is not supported (supported: "
            f"{SUPPORTED_Y})")
    if isinstance(model, DualExpFamLSMConsistent) and \
            model.family_x not in ("gaussian", "bernoulli", "poisson"):
        raise UnsupportedLaplacePath(f"family_x={model.family_x!r}")
    if not np.array_equal(Y, Y.T):
        raise UnsupportedLaplacePath("Y must be symmetric")
    mask = model._mask_f
    if not np.array_equal(mask, mask.T) or np.any(np.diag(mask) != 0):
        raise UnsupportedLaplacePath("pair mask must be symmetric with a "
                                     "zero diagonal")
    if X.shape != (model.n, model.d) or Y.shape != (model.n, model.n):
        raise UnsupportedLaplacePath("X/Y shapes do not match the model")


def _x_dispersion(model, sigma: np.ndarray) -> np.ndarray:
    """phi_l per X column: Gaussian variance for Gaussian columns, else 1."""

    phi = np.ones(model.d)
    if isinstance(model, DualExpFamLSMPerColumnConsistent):
        gauss = model._col_idx["gaussian"]
    else:
        gauss = np.arange(model.d) if model.family_x == "gaussian" else []
    if len(gauss):
        phi[gauss] = np.diag(sigma)[gauss]
    return phi


# --------------------------------------------------------------------------
# Phi, gradient, Hessian
# --------------------------------------------------------------------------

def complete_log_density(model, X, Y, Z: np.ndarray) -> float:
    """Phi(Z) = log p(Z, X, Y | theta), all constants included."""

    t = _theta(model)
    return float(calc_Q_dual_strict_exp(
        X, Y, Z[:, :, None], t["F"], t["sigma"], t["var_z"], t["w0"],
        t["w"], model))


def _scores(model, X, Y, Z):
    t = _theta(model)
    F, w0, w = t["F"], t["w0"], t["w"]
    eta_x = Z @ F.T
    phi = _x_dispersion(model, t["sigma"])
    score_x = (X - model._mean_function_x(eta_x)) / phi          # (n, d)
    curv_x = model._variance_function_x(eta_x) / phi              # (n, d)
    eta_y = w0 + w * (Z @ Z.T)
    mask = model._mask_f
    score_y = (Y - model._mean_function(eta_y)) * mask            # φ_Y = 1
    curv_y = model._variance_function(eta_y) * mask
    return t, score_x, curv_x, score_y, curv_y


def gradient(model, X, Y, Z: np.ndarray) -> np.ndarray:
    """∇_Z Phi, shape (n, K).

    prior: −z_i / var_z;  X: Fᵀ s_i;  Y: w Σ_j R_ij z_j (the ½ over ordered
    pairs cancels against the two orders in which z_i appears).
    """

    t, s_x, _, r_y, _ = _scores(model, X, Y, Z)
    return -Z / t["var_z"] + s_x @ t["F"] + t["w"] * (r_y @ Z)


def negative_hessian(model, X, Y, Z: np.ndarray) -> np.ndarray:
    """H = −∇²_Z Phi, shape (nK, nK), index (i, a) -> i*K + a.

    H_ii = I/var_z + Fᵀ diag(c_i) F + w² Σ_j C_ij z_j z_jᵀ
    H_ij = w² C_ij z_j z_iᵀ − w R_ij I          (i ≠ j), H_ji = H_ijᵀ
    """

    t, _, c_x, r_y, c_y = _scores(model, X, Y, Z)
    F, w = t["F"], t["w"]
    n, k = Z.shape
    blocks = (w * w) * c_y[:, :, None, None] * Z[None, :, :, None] \
        * Z[:, None, None, :]                                     # [i,j,a,b]
    blocks -= w * r_y[:, :, None, None] * np.eye(k)[None, None]
    diag = (np.eye(k)[None] / t["var_z"]
            + np.einsum("la,il,lb->iab", F, c_x, F)
            + (w * w) * np.einsum("ij,ja,jb->iab", c_y, Z, Z))
    idx = np.arange(n)
    blocks[idx, idx] = diag                                       # C_ii=R_ii=0
    return blocks.transpose(0, 2, 1, 3).reshape(n * k, n * k)


# --------------------------------------------------------------------------
# joint mode and Laplace integral
# --------------------------------------------------------------------------

@dataclass
class LaplaceResult:
    status: str
    log_observed: float                 # nan unless status == OK
    phi_at_mode: float
    logdet_H: float
    logdet_sign: float
    min_eigenvalue_H: float
    grad_inf: float
    iterations: int
    n: int
    k: int
    Z_hat: np.ndarray = field(repr=False)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


def find_joint_mode(model, X, Y, Z0: np.ndarray, *, grad_tol: float = 1e-8,
                    max_iter: int = 200) -> tuple[np.ndarray, int, float]:
    """Deterministic joint mode of Phi by damped full Newton.

    Newton direction H⁻¹∇Phi with backtracking on Phi (monotone ascent).
    Where H is not positive definite at an iterate the direction falls back
    to the gradient; this is optimisation mechanics only and does not alter
    the evaluated quantity, which is checked at the returned point.
    Returns (Z, iterations, final max|∇Phi|).
    """

    Z = np.array(Z0, dtype=float, copy=True)
    n, k = Z.shape
    value = complete_log_density(model, X, Y, Z)
    for it in range(1, max_iter + 1):
        g = gradient(model, X, Y, Z)
        g_inf = float(np.max(np.abs(g)))
        if g_inf <= grad_tol:
            return Z, it - 1, g_inf
        H = negative_hessian(model, X, Y, Z)
        try:
            step = np.linalg.solve(H, g.ravel())
            if float(step @ g.ravel()) <= 0.0:
                raise np.linalg.LinAlgError("not an ascent direction")
        except np.linalg.LinAlgError:
            step = g.ravel().copy()
        step = step.reshape(n, k)
        alpha = 1.0
        while alpha > 1e-12:
            trial = Z + alpha * step
            try:
                trial_value = complete_log_density(model, X, Y, trial)
            except FloatingPointError:
                trial_value = -math.inf
            if trial_value >= value:
                Z, value = trial, trial_value
                break
            alpha *= 0.5
        else:
            break                                   # no ascent possible
    g = gradient(model, X, Y, Z)
    return Z, max_iter, float(np.max(np.abs(g)))


def laplace_log_observed(model, X, Y, Z0: np.ndarray, *,
                         grad_tol: float = 1e-8,
                         max_iter: int = 200) -> LaplaceResult:
    """Laplace approximation to log p(X, Y | theta, K) at fixed theta.

    Returns a structured result. The value is reported only when the joint
    stationarity condition holds (max|∇Phi| ≤ grad_tol) AND H is positive
    definite at the mode; no ridge or jitter is ever added to H.
    """

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    _check_supported(model, X, Y)
    Z_hat, iters, g_inf = find_joint_mode(model, X, Y, Z0, grad_tol=grad_tol,
                                          max_iter=max_iter)
    n, k = Z_hat.shape
    phi = complete_log_density(model, X, Y, Z_hat)
    H = negative_hessian(model, X, Y, Z_hat)
    H = 0.5 * (H + H.T)          # floating-point symmetrisation only
    sign, logdet = np.linalg.slogdet(H)
    min_eig = float(np.linalg.eigvalsh(H)[0])
    notes: list[str] = []
    if g_inf > grad_tol:
        status = STATUS_NOT_STATIONARY
        notes.append(f"max|grad Phi|={g_inf:.3e} > {grad_tol:.1e}")
    elif sign <= 0 or min_eig <= 0.0:
        status = STATUS_HESSIAN_NOT_PD
        notes.append(f"slogdet sign={sign}, min eigenvalue={min_eig:.3e}")
    else:
        status = STATUS_OK
    value = (phi + 0.5 * n * k * math.log(2.0 * math.pi) - 0.5 * logdet
             if status == STATUS_OK else float("nan"))
    return LaplaceResult(status=status, log_observed=float(value),
                         phi_at_mode=phi, logdet_H=float(logdet),
                         logdet_sign=float(sign), min_eigenvalue_H=min_eig,
                         grad_inf=g_inf, iterations=iters, n=n, k=k,
                         Z_hat=Z_hat, notes=notes)


# --------------------------------------------------------------------------
# parameter term (kept separate)
# --------------------------------------------------------------------------

def loading_parameter_count(k: int, d: int) -> int:
    """K-dependent loading count K d − K(K−1)/2 (Phase 9G convention).

    Other parameters (Gaussian-X dispersions, w0, w) are the caller's to
    add; they do not depend on K when the family assignment is fixed.
    """

    return k * d - k * (k - 1) // 2


def calc_C_Lap(result: LaplaceResult, *, d_K: int, N: float
               ) -> dict[str, Any]:
    """C_Lap = −2 · laplace_log_observed + d_K · log(N).

    ``d_K`` and ``N`` are required from the caller. N = n is a working
    convention (Phase 9G), not a proven sample size; nothing here chooses it.
    """

    if N <= 1:
        raise ValueError("N must exceed 1")
    integration = -2.0 * result.log_observed
    penalty = float(d_K) * math.log(N)
    return {"status": result.status,
            "C_Lap": integration + penalty if result.ok else float("nan"),
            "integration_term": integration,
            "parameter_term": penalty, "d_K": int(d_K), "N": float(N),
            "evaluator_version": EVALUATOR_VERSION}
