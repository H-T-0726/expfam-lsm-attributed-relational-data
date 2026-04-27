"""
Shared utilities for ExpFam experiments.

Provides:
- calc_rmse, procrustes_rotation
- calc_Q_no_fact   : Q without factorial (for convergence monitoring)
- calc_Q_strict    : Q with -ln(y!) correction (for BIC)
- calc_bic         : BIC = -2*Q_strict + num_params*ln(n)
- run_em           : EM runner with NaN fallback (Y-only ExpFam)
- calc_Q_dual      : Q function for Dual-ExpFam (X + Y)
- run_em_dual      : EM runner for DualExpFamLSM (X + Y both ExpFam)
"""

import numpy as np
from scipy.special import gammaln
from pathlib import Path
import sys
import time

_ROOT = Path(__file__).parent.parent.parent
_SRC  = Path(__file__).parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from model_expfam import ExpFamLatentStructuralModel   # noqa
from model_dual_expfam import DualExpFamLSM            # noqa
from data_generator_expfam import generate_poisson_data  # noqa


# ──────────────────────────────────────────────────────────────────────
# Basic helpers
# ──────────────────────────────────────────────────────────────────────

def calc_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def procrustes_rotation(A_est: np.ndarray, A_true: np.ndarray):
    """Optimal rotation R minimising ||A_est[:,:k_min] @ R - A_true[:,:k_min]||."""
    k_min = min(A_est.shape[1], A_true.shape[1])
    M = A_est[:, :k_min].T @ A_true[:, :k_min]
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt, k_min


# ──────────────────────────────────────────────────────────────────────
# Q functions
# ──────────────────────────────────────────────────────────────────────

def _lnpZ_lnpX(X, Z_l, F, sigma, var_z):
    """ln p(Z_l) + ln p(X|Z_l)."""
    n, k = Z_l.shape
    d = X.shape[1]
    sd = np.diag(sigma)

    lnpZ = (-(n * k / 2) * np.log(2 * np.pi)
            - (n * k / 2) * np.log(var_z)
            - (1 / (2 * var_z)) * np.sum(Z_l ** 2))

    resid = X - Z_l @ F.T
    lnpX = (-(n * d / 2) * np.log(2 * np.pi)
            - (n / 2) * np.sum(np.log(sd))
            - 0.5 * np.sum(resid ** 2 / sd))

    return lnpZ + lnpX


def calc_Q_no_fact(X, Y, Z_samples, F, sigma, var_z, w0, w, model):
    """Q without factorial term (used for convergence monitoring)."""
    n, k, L = Z_samples.shape
    Q = 0.0
    for l in range(L):
        Z_l = Z_samples[:, :, l]
        base = _lnpZ_lnpX(X, Z_l, F, sigma, var_z)
        lnpY = model.calc_log_likelihood_Y(Y, Z_samples[:, :, l:l+1], w0, w)
        Q += base + lnpY
    return Q / L


def calc_Q_strict(X, Y, Z_samples, F, sigma, var_z, w0, w, model):
    """
    Q with strict Poisson log-likelihood: includes -ln(y!) correction.

    For Bernoulli: identical to calc_Q_no_fact (no change).
    For Poisson  : adds  -sum_{i<j} ln(y_ij!)  which is constant w.r.t. theta.
                   Required for meaningful absolute BIC values.
    """
    Q_base = calc_Q_no_fact(X, Y, Z_samples, F, sigma, var_z, w0, w, model)
    if model.family == "poisson":
        upper_mask = np.triu(np.ones_like(Y, dtype=bool), k=1)
        factorial_corr = -float(np.sum(gammaln(Y[upper_mask] + 1)))
        return Q_base + factorial_corr
    return Q_base


def calc_bic(Q_strict: float, k: int, n: int, d: int) -> tuple:
    """
    BIC = -2 * Q_strict + num_params * ln(n)

    num_params = (k+1)*d - k*(k-1)//2
        (free params in F after rotational constraint, plus diagonal Sigma,
         matches Mikawa et al. 2024 formula)
    """
    num_params = (k + 1) * d - k * (k - 1) // 2
    bic = -2.0 * Q_strict + num_params * np.log(n)
    return bic, num_params


# ──────────────────────────────────────────────────────────────────────
# EM runner
# ──────────────────────────────────────────────────────────────────────

def run_em(
    X: np.ndarray,
    Y: np.ndarray,
    true_params: dict,
    family: str,
    k: int,
    L: int = 10,
    num_iter: int = 10,
    seed: int = 42,
    verbose: bool = False,
    compute_strict_Q: bool = False,
    sigma_y_init: float = 1.0,
    fix_w: bool = False,
) -> dict:
    """
    Run Monte Carlo EM with automatic NaN fallback.

    Fallback strategy (up to 2 retries):
      - Halve Newton alpha (0.5 → 0.25 → 0.125)
      - Halve initial w for Poisson

    Gaussian: sigma_y estimated each M-step via model.calc_sigma_y().

    Returns dict with rmse_Z, rmse_X, Q_final, Q_strict (if requested),
    w0_est, w_est, sigma_y_est (Gaussian only), nan_occurred.
    """
    n, d = X.shape
    max_retries = 2

    for retry in range(max_retries + 1):
        newton_alpha = 0.5 / (2 ** retry)
        w_init = 0.1 / (2 ** retry) if family == "poisson" else 0.5

        rng = np.random.default_rng(seed + retry * 1000)
        model = ExpFamLatentStructuralModel(
            n=n, d=d, k=k, L=L, family=family,
            sigma_y=sigma_y_init,
        )
        model.initialize_params(true_params=true_params, seed=seed + retry * 1000)

        # Informed init
        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if family == "bernoulli":
            density = float(np.clip(np.mean(Y), 1e-6, 1 - 1e-6))
            model.params["w0"] = np.log(density / (1 - density))
            model.params["w"] = 0.5
        elif family == "poisson":
            upper = Y[upper_mask]
            mean_cnt = float(upper[upper > 0].mean()) if np.any(upper > 0) else 1.0
            model.params["w0"] = np.log(mean_cnt + 1e-10)
            model.params["w"] = w_init
        else:  # gaussian
            upper_vals = Y[upper_mask]
            model.params["w0"] = float(upper_vals.mean())
            model.params["w"] = 0.5
            # Init sigma_y from residual spread
            model.sigma_y = float(max(upper_vals.std(), 0.01))

        Z = model.params["Z"].copy()
        F = model.params["F"].copy()
        sigma = model.params["sigma"].copy()
        w0 = model.params["w0"]
        w = model.params["w"]
        var_z = model.params["var_z"]

        # Ablation: fix w=0 (block relational signal from Y)
        if fix_w:
            w = 0.0
            model.params["w"] = 0.0

        Z_prev = Z.copy()
        nan_count = 0
        Q_history = []

        for iteration in range(1, num_iter + 1):
            # E-step
            Z_samples = np.zeros((n, k, L))
            for l in range(L):
                model.params.update(
                    dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w)
                )
                Z_new = model.calc_eta_newton(
                    X, Y, rng=rng, max_iter=10, alpha=newton_alpha
                )
                Z_samples[:, :, l] = Z_new
                Z = Z_new.copy()

            # NaN guard
            if np.any(np.isnan(Z_samples)) or np.any(np.isinf(Z_samples)):
                nan_count += 1
                if verbose:
                    print(f"  [NaN iter={iteration} retry={retry}] "
                          "Resetting to Z_prev.")
                Z_samples = np.stack([Z_prev] * L, axis=2)
                Z = Z_prev.copy()

            Z_samples = model.scale_Z(Z_samples)
            Z_prev = Z.copy()
            Z = Z_samples[:, :, -1].copy()

            # M-step
            F = model.calc_F(X, Z_samples)
            sigma = model.calc_sigma(X, Z_samples, F)
            w0 = model.calc_w0(Y, Z_samples, w0, w, max_iter=50)
            if not fix_w:
                w = model.calc_w(Y, Z_samples, w0, w, max_iter=50)
            if family == "gaussian":
                model.calc_sigma_y(Y, Z_samples, w0, w)

            Q = calc_Q_no_fact(X, Y, Z_samples, F, sigma, var_z, w0, w, model)
            Q_history.append(Q)

            if verbose:
                print(f"    iter={iteration:2d} Q={Q:.2f} "
                      f"w0={w0:.4f} w={w:.4f}")

        nan_occurred = nan_count > 0

        # Stop retrying if no NaN
        if not nan_occurred:
            break
        if retry < max_retries and verbose:
            print(f"  [FALLBACK] Retry {retry+1}: "
                  f"newton_alpha={newton_alpha/2:.4f}")

    # ---- Basic metrics ----
    Z_est = Z_samples[:, :, -1]

    # RMSE(X): always computable (latent reconstruction of observed attributes)
    rmse_X = calc_rmse(X, Z_est @ F.T)

    # Metrics requiring ground-truth parameters (synthetic data only)
    has_truth = (true_params is not None and "Z" in true_params
                 and true_params["Z"] is not None)

    if has_truth:
        R, k_min = procrustes_rotation(Z_est, true_params["Z"])
        Z_rot    = Z_est[:, :k_min] @ R
        rmse_Z   = calc_rmse(true_params["Z"][:, :k_min], Z_rot)

        # RMSE(F): same Procrustes rotation
        F_aligned = F[:, :k_min] @ R
        rmse_F    = calc_rmse(true_params["F"][:, :k_min], F_aligned)

        # RMSE(Sigma)
        rmse_sigma = calc_rmse(np.diag(sigma), np.diag(true_params["sigma"]))

        # RMSE(Y): predicted mean vs true mean (upper triangle)
        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        eta_est  = float(w0) + float(w) * (Z_est @ Z_est.T)
        mu_est   = model._mean_function(eta_est)
        eta_true = float(true_params["w0"]) + float(true_params["w"]) * (
            true_params["Z"] @ true_params["Z"].T)
        mu_true  = model._mean_function(eta_true)
        rmse_Y   = calc_rmse(mu_est[upper_mask], mu_true[upper_mask])

        w0_err = abs(float(w0) - float(true_params.get("w0", 0.0)))
        w_err  = abs(float(w)  - float(true_params.get("w",  0.5)))
    else:
        # Real data: no ground truth available
        R = None; k_min = k
        rmse_Z = rmse_F = rmse_sigma = rmse_Y = float("nan")
        w0_err = w_err = float("nan")

    Q_strict_val = None
    if compute_strict_Q:
        Q_strict_val = calc_Q_strict(
            X, Y, Z_samples, F, sigma, var_z, w0, w, model
        )

    return {
        "family": family,
        "n": n, "k": k,
        "Q_final": Q_history[-1] if Q_history else float("nan"),
        "Q_strict": Q_strict_val,
        "Q_history": Q_history,
        # Z and X (basic)
        "rmse_Z": rmse_Z,
        "rmse_X": rmse_X,
        # Full parameter metrics
        "rmse_F":     rmse_F,
        "rmse_sigma": rmse_sigma,
        "rmse_Y":     rmse_Y,
        "w0_err":     w0_err,
        "w_err":      w_err,
        # Estimated values
        "w0_est": float(w0),
        "w_est":  float(w),
        "sigma_y_est": float(model.sigma_y),
        "nan_occurred": nan_occurred,
        "nan_count": nan_count,
        "_Z_samples": Z_samples,
        "_F": F,
        "_sigma": sigma,
        "_var_z": var_z,
    }


# ──────────────────────────────────────────────────────────────────────
# Dual-ExpFam Q function
# ──────────────────────────────────────────────────────────────────────

def _lnpZ(Z_l: np.ndarray, var_z: float) -> float:
    """ln p(Z_l) under isotropic Gaussian prior."""
    n, k = Z_l.shape
    return float(
        -(n * k / 2.0) * np.log(2.0 * np.pi * var_z)
        - (1.0 / (2.0 * var_z)) * np.sum(Z_l ** 2)
    )


def calc_Q_dual(
    X: np.ndarray,
    Y: np.ndarray,
    Z_samples: np.ndarray,
    F: np.ndarray,
    sigma: np.ndarray,
    var_z: float,
    w0: float,
    w: float,
    model: "DualExpFamLSM",
) -> float:
    """
    Q function for Dual-ExpFam LSM.

    Q = (1/L) Σ_l [ ln p(Z_l) + ln p(X|Z_l, F; family_x)
                               + ln p(Y|Z_l, w0, w; family_y) ]

    Note: Poisson Y omits -Σln(y_ij!); Poisson X omits -Σln(x_ij!).
    These constants are excluded from monitoring (consistent with calc_Q_no_fact).
    """
    n, k, L = Z_samples.shape
    Q = 0.0
    for l in range(L):
        Z_l  = Z_samples[:, :, l]
        lnpZ = _lnpZ(Z_l, var_z)
        lnpX = model.calc_log_likelihood_X(X, Z_samples[:, :, l:l+1], F)
        lnpY = model.calc_log_likelihood_Y(Y, Z_samples[:, :, l:l+1], w0, w)
        Q += lnpZ + lnpX + lnpY
    return Q / L


def calc_Q_dual_strict(
    X: np.ndarray,
    Y: np.ndarray,
    Z_samples: np.ndarray,
    F: np.ndarray,
    sigma: np.ndarray,
    var_z: float,
    w0: float,
    w: float,
    model: "DualExpFamLSM",
) -> float:
    """
    Strict Q for BIC: includes factorial corrections for Poisson X and Y.

    Adds -Σln(y_ij!) if family_y='poisson'
    Adds -Σln(x_ij!) if family_x='poisson'
    """
    Q_base = calc_Q_dual(X, Y, Z_samples, F, sigma, var_z, w0, w, model)
    upper_mask = np.triu(np.ones((model.n, model.n), dtype=bool), k=1)
    corr = 0.0
    if model.family == "poisson":     # family_y
        corr -= float(np.sum(gammaln(Y[upper_mask] + 1)))
    if model.family_x == "poisson":
        corr -= float(np.sum(gammaln(X + 1)))
    return Q_base + corr


# ──────────────────────────────────────────────────────────────────────
# Dual-ExpFam EM runner
# ──────────────────────────────────────────────────────────────────────

def calc_bic_dual(
    Q_strict: float, k: int, n: int, d: int,
    family_x: str = "gaussian", family_y: str = "bernoulli",
) -> tuple:
    """
    BIC for Dual-ExpFam LSM.

    num_params:
        F (rotational constraint): k*d - k*(k-1)//2
        Sigma: d  if family_x='gaussian', else 0
        sigma_y: 1  if family_y='gaussian', else 0
        (w0, w are treated as implicit per NOLTA 2024 convention)
    """
    f_params     = k * d - k * (k - 1) // 2
    sigma_x_p    = d if family_x == "gaussian" else 0
    sigma_y_p    = 1 if family_y == "gaussian" else 0
    num_params   = f_params + sigma_x_p + sigma_y_p
    bic = -2.0 * Q_strict + num_params * np.log(n)
    return bic, num_params


def run_em_dual(
    X: np.ndarray,
    Y: np.ndarray,
    true_params: dict,
    family_x: str,
    family_y: str,
    k: int,
    L: int = 10,
    num_iter: int = 10,
    seed: int = 42,
    verbose: bool = False,
    compute_strict_Q: bool = False,
    sigma_y_init: float = 1.0,
    fix_w: bool = False,
    fix_x: bool = False,
) -> dict:
    """
    Run Monte Carlo EM for DualExpFamLSM.

    Supports all 9 combinations of family_x × family_y.
    Fallback strategy (up to 2 retries): halve newton_alpha each retry.

    Returns dict with rmse_Z, rmse_X, rmse_F, rmse_sigma, rmse_Y,
    w0_err, w_err, Q_final, Q_strict (optional), and estimated values.
    """
    n, d = X.shape
    max_retries = 2

    for retry in range(max_retries + 1):
        newton_alpha = 0.5 / (2 ** retry)

        rng = np.random.default_rng(seed + retry * 1000)
        model = DualExpFamLSM(
            n=n, d=d, k=k, L=L,
            family_x=family_x,
            family_y=family_y,
            sigma_y=sigma_y_init,
        )
        model.initialize_params(true_params=true_params, seed=seed + retry * 1000)

        # ── Informed initialisation: Y side ──────────────────────────
        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if family_y == "bernoulli":
            density = float(np.clip(np.mean(Y), 1e-6, 1 - 1e-6))
            model.params["w0"] = np.log(density / (1 - density))
            model.params["w"]  = 0.5
        elif family_y == "poisson":
            upper = Y[upper_mask]
            mean_cnt = float(upper[upper > 0].mean()) if np.any(upper > 0) else 1.0
            model.params["w0"] = np.log(mean_cnt + 1e-10)
            model.params["w"]  = 0.1 / (2 ** retry)
        else:   # gaussian Y
            upper_vals = Y[upper_mask]
            model.params["w0"] = float(upper_vals.mean())
            model.params["w"]  = 0.5
            model.sigma_y = float(max(upper_vals.std(), 0.01))

        # ── Informed initialisation: X side ──────────────────────────
        # Scale F so that eta_x = F z_i stays in a numerically safe range.
        if family_x == "bernoulli":
            # Keep eta_x small → sigmoid near 0.5 → balanced binary attributes
            model.params["F"] *= 0.2
        elif family_x == "poisson":
            # Keep eta_x small → exp(eta_x) near 1 → Poisson mean ~ 1
            model.params["F"] *= 0.2

        Z = model.params["Z"].copy()
        F = model.params["F"].copy()
        sigma = model.params["sigma"].copy()
        w0    = model.params["w0"]
        w     = model.params["w"]
        var_z = model.params["var_z"]

        if fix_w:
            w = 0.0
            model.params["w"] = 0.0

        # fix_x ablation: set F=0 → Term 2 of gradient/Hessian = 0
        # → Z is learned purely from Y relational signal (Term 3 only)
        if fix_x:
            F = np.zeros((d, k))
            model.params["F"] = np.zeros((d, k))

        Z_prev    = Z.copy()
        nan_count = 0
        Q_history = []

        for iteration in range(1, num_iter + 1):
            # ── E-step ───────────────────────────────────────────────
            Z_samples = np.zeros((n, k, L))
            for l in range(L):
                model.params.update(dict(Z=Z.copy(), F=F, sigma=sigma,
                                         w0=w0, w=w))
                Z_new = model.calc_eta_newton(
                    X, Y, rng=rng, max_iter=10, alpha=newton_alpha
                )
                Z_samples[:, :, l] = Z_new
                Z = Z_new.copy()

            # NaN guard
            if np.any(np.isnan(Z_samples)) or np.any(np.isinf(Z_samples)):
                nan_count += 1
                if verbose:
                    print(f"  [NaN iter={iteration} retry={retry}] Resetting.")
                Z_samples = np.stack([Z_prev] * L, axis=2)
                Z = Z_prev.copy()

            Z_samples = model.scale_Z(Z_samples)
            Z_prev = Z.copy()
            Z = Z_samples[:, :, -1].copy()

            # ── M-step ───────────────────────────────────────────────
            if not fix_x:
                F     = model.calc_F(X, Z_samples)        # ← DualExpFamLSM override
                sigma = model.calc_sigma(X, Z_samples, F) # ← DualExpFamLSM override
            # (if fix_x: keep F=0, sigma=identity)
            w0    = model.calc_w0(Y, Z_samples, w0, w, max_iter=50)
            if not fix_w:
                w = model.calc_w(Y, Z_samples, w0, w, max_iter=50)
            if family_y == "gaussian":
                model.calc_sigma_y(Y, Z_samples, w0, w)

            Q = calc_Q_dual(X, Y, Z_samples, F, sigma, var_z, w0, w, model)
            Q_history.append(Q)

            if verbose:
                print(f"    iter={iteration:2d}  Q={Q:.2f}  "
                      f"w0={w0:.4f}  w={w:.4f}")

        nan_occurred = nan_count > 0
        if not nan_occurred:
            break
        if retry < max_retries and verbose:
            print(f"  [FALLBACK] Retry {retry+1}: newton_alpha={newton_alpha/2:.4f}")

    # ── Basic metrics ─────────────────────────────────────────────────
    Z_est  = Z_samples[:, :, -1]
    # RMSE(X): reconstruction via estimated F
    mu_x_est = model._mean_function_x(Z_est @ F.T)   # (n, d)
    rmse_X = calc_rmse(X, mu_x_est)

    has_truth = (
        true_params is not None
        and "Z" in true_params
        and true_params["Z"] is not None
    )

    if has_truth:
        R, k_min = procrustes_rotation(Z_est, true_params["Z"])
        Z_rot    = Z_est[:, :k_min] @ R
        rmse_Z   = calc_rmse(true_params["Z"][:, :k_min], Z_rot)

        F_aligned = F[:, :k_min] @ R
        rmse_F    = calc_rmse(true_params["F"][:, :k_min], F_aligned)

        rmse_sigma = calc_rmse(np.diag(sigma), np.diag(true_params["sigma"]))

        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        eta_y_est  = float(w0) + float(w) * (Z_est @ Z_est.T)
        mu_y_est   = model._mean_function(eta_y_est)
        eta_y_true = float(true_params["w0"]) + float(true_params["w"]) * (
            true_params["Z"] @ true_params["Z"].T)
        mu_y_true  = model._mean_function(eta_y_true)
        rmse_Y     = calc_rmse(mu_y_est[upper_mask], mu_y_true[upper_mask])

        w0_err = abs(float(w0) - float(true_params.get("w0", 0.0)))
        w_err  = abs(float(w)  - float(true_params.get("w",  0.5)))
    else:
        R = None; k_min = k
        rmse_Z = rmse_F = rmse_sigma = rmse_Y = float("nan")
        w0_err = w_err = float("nan")

    Q_strict_val = None
    if compute_strict_Q:
        Q_strict_val = calc_Q_dual_strict(
            X, Y, Z_samples, F, sigma, var_z, w0, w, model
        )

    return {
        "family_x": family_x,
        "family_y": family_y,
        "n": n, "k": k,
        "Q_final":  Q_history[-1] if Q_history else float("nan"),
        "Q_strict": Q_strict_val,
        "Q_history": Q_history,
        # Z and X
        "rmse_Z":     rmse_Z,
        "rmse_X":     rmse_X,
        # Full parameter metrics
        "rmse_F":     rmse_F,
        "rmse_sigma": rmse_sigma,
        "rmse_Y":     rmse_Y,
        "w0_err":     w0_err,
        "w_err":      w_err,
        # Estimated values
        "w0_est":      float(w0),
        "w_est":       float(w),
        "sigma_y_est": float(model.sigma_y),
        "nan_occurred": nan_occurred,
        "nan_count":    nan_count,
        "_Z_samples": Z_samples,
        "_F":   F,
        "_sigma": sigma,
        "_var_z": var_z,
    }
