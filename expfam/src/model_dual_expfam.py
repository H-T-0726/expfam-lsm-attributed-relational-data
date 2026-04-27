"""
Dual Exponential Family Latent Structural Model.

Extends ExpFamLatentStructuralModel so that BOTH the attribute data X
and the relational data Y can follow independent exponential family
distributions.

Generative model:
    z_i ~ N(0, sigma_z^2 * I_k)
    x_{ij} ~ ExpFam_X(eta_{ij}^X = f_j^T z_i)      j = 1..d
    y_{ij} ~ ExpFam_Y(eta_{ij}^Y = w0 + w z_i^T z_j)  i < j

E-step gradient (z_i):
    Term1 : -(1/sigma_z^2) z_i                                         [Z prior]
    Term2 : (1/phi_X) F^T [T_X(x_i) - A_X'(F z_i)]                   [X: NEW]
    Term3 : (w/2phi_Y) sum_{j!=i} [T_Y(y_ij) - A_Y'(eta_ij^Y)] z_j   [Y: inherited]

E-step Hessian (precision matrix):
    Term1 : (1/sigma_z^2) I
    Term2 : (1/phi_X) F^T diag[A_X''(F z_i)] F                        [X: NEW]
    Term3 : (w^2/2phi_Y) sum_{j!=i} A_Y''(eta_ij^Y) z_j z_j^T        [Y: inherited]

M-step:
    F      : closed-form (Gaussian X) or Adam (Bernoulli / Poisson X)
    Sigma  : closed-form MLE (Gaussian X only); identity otherwise
    w0, w  : Adam (inherited)
    sigma_y: MLE (Gaussian Y only, inherited)

Backward compatibility:
    DualExpFamLSM(family_x='gaussian', family_y='bernoulli')
    behaves identically to ExpFamLatentStructuralModel(family='bernoulli').
"""

import numpy as np
import sys
from pathlib import Path

_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))
from model_expfam import ExpFamLatentStructuralModel  # noqa: E402


class DualExpFamLSM(ExpFamLatentStructuralModel):
    """
    Dual Exponential Family Latent Structural Model.

    Parameters
    ----------
    n, d, k, L : int
        Same as base class.
    family_x : str
        Distribution for attribute data X.
        One of 'gaussian', 'bernoulli', 'poisson'. Default 'gaussian'.
    family_y : str
        Distribution for relational data Y.
        One of 'gaussian', 'bernoulli', 'poisson'. Default 'bernoulli'.
    sigma_y : float
        Initial sigma_y for Gaussian Y (updated each M-step). Default 1.0.
    """

    VALID_FAMILIES = ("gaussian", "bernoulli", "poisson")

    def __init__(
        self,
        n: int,
        d: int,
        k: int,
        L: int = 10,
        family_x: str = "gaussian",
        family_y: str = "bernoulli",
        sigma_y: float = 1.0,
    ):
        if family_x not in self.VALID_FAMILIES:
            raise ValueError(
                f"Unsupported family_x '{family_x}'. "
                f"Choose from {self.VALID_FAMILIES}."
            )
        if family_y not in self.VALID_FAMILIES:
            raise ValueError(
                f"Unsupported family_y '{family_y}'. "
                f"Choose from {self.VALID_FAMILIES}."
            )
        # parent stores family_y as self.family (used by _mean_function / _variance_function)
        super().__init__(n=n, d=d, k=k, L=L, family=family_y, sigma_y=sigma_y)
        self.family_x = family_x

    # ------------------------------------------------------------------
    # X-side link functions
    # ------------------------------------------------------------------

    def _mean_function_x(self, eta_x: np.ndarray) -> np.ndarray:
        """
        A_X'(eta) — conditional mean E[T_X(X) | eta] for the X model.

        eta_x : (d,) or (n, d) array of natural parameters.
        """
        if self.family_x == "gaussian":
            return eta_x.copy()
        if self.family_x == "bernoulli":
            return self._sigmoid(eta_x)
        # poisson
        return np.exp(np.clip(eta_x, -20, 10))

    def _variance_function_x(self, eta_x: np.ndarray) -> np.ndarray:
        """
        A_X''(eta) / phi_X — effective curvature weight for the X model.

        Used in the Hessian (precision matrix) Term 2.
        For Gaussian X: handled separately via sigma^{-1} diagonal.
        """
        if self.family_x == "bernoulli":
            s = self._sigmoid(eta_x)
            return np.clip(s * (1.0 - s), 1e-8, None)
        if self.family_x == "poisson":
            return np.clip(np.exp(np.clip(eta_x, -20, 10)), 1e-8, None)
        # gaussian: caller should use sigma directly; this path is a fallback
        return np.ones_like(eta_x)

    # ------------------------------------------------------------------
    # E-step: gradient override
    # ------------------------------------------------------------------

    def _calc_gradient(
        self, X, Y, Z, F, sigma, var_z, w0, w, i
    ) -> np.ndarray:
        """
        Generalised gradient of -ln f(z_i | X, Y).

        Term 1  :  -(1/sigma_z^2) z_i
        Term 2  :  (1/phi_X) F^T [T_X(x_i) - A_X'(F z_i)]   [X: ExpFam]
        Term 3  :  (w/2phi_Y) Z_{-i}^T [T_Y(y_{i,-i}) - A_Y'(eta_Y)]   [Y: inherited]
        """
        z_i = Z[i, :]
        x_i = X[i, :]

        # ── Term 1: Z prior ──────────────────────────────────────────
        term1 = -(1.0 / var_z) * z_i

        # ── Term 2: X likelihood (ExpFam generalised) ────────────────
        eta_x_i = F @ z_i                          # (d,)
        mu_x_i  = self._mean_function_x(eta_x_i)   # A_X'(eta), (d,)
        residual_x = x_i - mu_x_i                  # T_X(x) - A_X'(eta)

        if self.family_x == "gaussian":
            # phi_X = sigma_j^2  →  divide residual element-wise
            sigma_inv_diag = 1.0 / np.maximum(np.diag(sigma), 1e-8)
            term2 = F.T @ (residual_x * sigma_inv_diag)
        else:
            # phi_X = 1  →  no division
            term2 = F.T @ residual_x

        # ── Term 3: Y likelihood (family_y, reuse parent logic) ──────
        eta_y    = w0 + w * (Z @ z_i)              # (n,)
        mu_y     = self._mean_function(eta_y)       # A_Y'(eta), uses self.family = family_y
        residual_y = Y[i, :] - mu_y                # T_Y(y) - A_Y'(eta)
        residual_y[i] = 0.0                        # exclude self
        if self.family == "gaussian":               # family_y == 'gaussian'
            residual_y = residual_y / max(self.sigma_y ** 2, 1e-8)
        term3 = 0.5 * w * (Z.T @ residual_y)

        return -(term1 + term2 + term3)

    # ------------------------------------------------------------------
    # E-step: precision matrix override
    # ------------------------------------------------------------------

    def _calc_precision_matrix(
        self, Z, F, sigma, var_z, w0, w, i
    ) -> np.ndarray:
        """
        Generalised precision matrix A_i = -Hessian of ln f(z_i | X, Y).

        Term 1  :  (1/sigma_z^2) I
        Term 2  :  (1/phi_X) F^T diag[A_X''(F z_i)] F   [X: ExpFam]
        Term 3  :  (w^2/2phi_Y) Z^T diag[A_Y''(eta_Y)] Z  [Y: inherited]
        """
        z_i = Z[i, :]
        k   = Z.shape[1]

        # ── Term 1 ───────────────────────────────────────────────────
        term1 = (1.0 / var_z) * np.eye(k)

        # ── Term 2: X curvature ──────────────────────────────────────
        eta_x_i = F @ z_i   # (d,)

        if self.family_x == "gaussian":
            # A_X'' = 1, phi_X = sigma_j^2  →  F^T Sigma^{-1} F
            sigma_inv_diag = 1.0 / np.maximum(np.diag(sigma), 1e-8)
            # Efficient: F^T diag(1/sigma_j^2) F
            term2 = F.T @ (F * sigma_inv_diag[:, None])
        else:
            var_x_i = self._variance_function_x(eta_x_i)  # A_X''(eta)/phi_X, (d,)
            # F^T diag(var_x_i) F
            term2 = F.T @ (F * var_x_i[:, None])

        # ── Term 3: Y curvature (inherited) ──────────────────────────
        eta_y   = w0 + w * (Z @ z_i)               # (n,)
        var_y   = self._variance_function(eta_y)    # A_Y''(eta)/phi_Y, (n,)
        var_y[i] = 0.0
        term3 = 0.5 * (w ** 2) * (Z.T @ np.diag(var_y) @ Z)

        return term1 + term2 + term3

    # ------------------------------------------------------------------
    # M-step: F update
    # ------------------------------------------------------------------

    def calc_F(self, X: np.ndarray, Z_samples: np.ndarray) -> np.ndarray:
        """
        Update F.

        Gaussian X : closed-form (inherited from LatentStructuralModel).
        Other X    : Adam gradient ascent on Q_X.
        """
        if self.family_x == "gaussian":
            return super().calc_F(X, Z_samples)
        return self._calc_F_adam(X, Z_samples)

    def _calc_F_adam(
        self,
        X: np.ndarray,
        Z_samples: np.ndarray,
        max_iter: int = 50,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """
        Adam gradient ascent for F under non-Gaussian X.

        Gradient of Q_X w.r.t. F (shape d×k):
            ∇_F Q_X = (1/L) Σ_l (X - A_X'(Z_l F^T))^T Z_l

        This is the full-matrix update (all j simultaneously), which is
        more efficient than per-feature updates.
        """
        n, k, L = Z_samples.shape
        F = self.params["F"].copy()     # initialise from current estimate
        m = np.zeros_like(F)           # Adam 1st moment
        v = np.zeros_like(F)           # Adam 2nd moment

        for t in range(1, max_iter + 1):
            # ── gradient ──────────────────────────────────────────────
            grad = np.zeros_like(F)
            for l in range(L):
                Z_l    = Z_samples[:, :, l]          # (n, k)
                eta_x  = Z_l @ F.T                   # (n, d)
                mu_x   = self._mean_function_x(eta_x) # (n, d)
                residual = X - mu_x                  # T_X(x) - A_X'(eta)
                grad  += residual.T @ Z_l            # (d, k)
            grad /= L

            # ── Adam step (maximise → negate grad for Adam) ───────────
            g  = -grad
            m  = beta1 * m + (1.0 - beta1) * g
            v  = beta2 * v + (1.0 - beta2) * g ** 2
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)
            F_new = F - lr * m_hat / (np.sqrt(v_hat) + eps)

            if np.max(np.abs(F_new - F)) < tol:
                F = F_new
                break
            F = F_new

        return F

    # ------------------------------------------------------------------
    # M-step: Sigma update
    # ------------------------------------------------------------------

    def calc_sigma(
        self, X: np.ndarray, Z_samples: np.ndarray, F: np.ndarray
    ) -> np.ndarray:
        """
        Update Sigma (X-side dispersion).

        Gaussian X : closed-form MLE (inherited).
        Other X    : Sigma is not applicable; return identity.
        """
        if self.family_x == "gaussian":
            return super().calc_sigma(X, Z_samples, F)
        return np.eye(self.d)

    # ------------------------------------------------------------------
    # X log-likelihood (for Q function monitoring)
    # ------------------------------------------------------------------

    def calc_log_likelihood_X(
        self,
        X: np.ndarray,
        Z_samples: np.ndarray,
        F: np.ndarray,
    ) -> float:
        """
        Compute (1/L) Σ_l E[ln p(X | Z_l, F)] for the current family_x.

        Note: F is passed explicitly (not taken from self.params) so that
        the Q function can be evaluated with the latest M-step estimate.

        Gaussian  : omits the normalising constant -nd/2 ln(2π).
        Bernoulli : T(x) = x, base measure h(x) = 1.
        Poisson   : omits -Σ ln(x_ij!) (constant w.r.t. F).
        """
        n, k, L = Z_samples.shape
        ll = 0.0

        for l in range(L):
            Z_l   = Z_samples[:, :, l]   # (n, k)
            eta_x = Z_l @ F.T            # (n, d)

            if self.family_x == "gaussian":
                sigma      = self.params["sigma"]
                sigma_diag = np.maximum(np.diag(sigma), 1e-8)
                resid      = X - eta_x    # (n, d)
                # ln p = -0.5*(resid^2/sigma_j^2) - 0.5*ln(sigma_j^2) - 0.5*ln(2pi)
                # Include ln(2pi) constant to be consistent with _lnpZ_lnpX
                # and to ensure BIC values are comparable across Q functions.
                ln_p = (-0.5 * resid ** 2 / sigma_diag
                        - 0.5 * np.log(sigma_diag)
                        - 0.5 * np.log(2.0 * np.pi))
            elif self.family_x == "bernoulli":
                S    = self._sigmoid(np.clip(eta_x, -500, 500))
                S    = np.clip(S, 1e-10, 1.0 - 1e-10)
                ln_p = X * np.log(S) + (1.0 - X) * np.log(1.0 - S)
            else:   # poisson
                eta_c = np.clip(eta_x, -20, 10)
                ln_p  = X * eta_c - np.exp(eta_c)   # omits -ln(x!)

            ll += float(np.sum(ln_p))

        return ll / L

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        sigma_y_str = (
            f", sigma_y={self.sigma_y:.4f}" if self.family == "gaussian" else ""
        )
        return (
            f"DualExpFamLSM("
            f"n={self.n}, d={self.d}, k={self.k}, "
            f"family_x='{self.family_x}', family_y='{self.family}'"
            f"{sigma_y_str})"
        )
