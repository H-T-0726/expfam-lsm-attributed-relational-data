"""
Exponential Family Latent Structural Model.

Extends LatentStructuralModel (reproduction/src/model.py) via Template Method
pattern to support arbitrary exponential family distributions for relational Y.

Supported families:
    'bernoulli' : Y_ij ~ Bernoulli(sigmoid(eta_ij))   [NOLTA 2024 original]
    'poisson'   : Y_ij ~ Poisson(exp(eta_ij))         [new extension]
    'gaussian'  : Y_ij ~ N(eta_ij, sigma_y^2)         [SMC 2022 extension]

Mathematical framework (GLM canonical link):
    p(y; eta) = h(y) exp(eta T(y) - A(eta)) / phi
    gradient term3 : (1/2) sum_{j!=i} [T(y_ij) - A'(eta_ij)] / phi  * w z_j
    precision term3: (1/2) sum_{j!=i}  A''(eta_ij) / phi             * w^2 z_j z_j^T

Dispersion parameter phi:
    Bernoulli / Poisson : phi = 1   (no dispersion)
    Gaussian            : phi = sigma_y^2  (estimated in M-step)
"""

import numpy as np
from typing import Optional, Dict, Any
import sys
from pathlib import Path

# Import base class from reproduction directory
_BASE = Path(__file__).parent.parent.parent / "reproduction" / "src"
sys.path.insert(0, str(_BASE))
from model import LatentStructuralModel  # noqa: E402


class ExpFamLatentStructuralModel(LatentStructuralModel):
    """Exponential family generalisation of LatentStructuralModel."""

    def __init__(self, n: int, d: int, k: int, L: int = 10,
                 family: str = "bernoulli", sigma_y: float = 1.0):
        super().__init__(n=n, d=d, k=k, L=L)
        if family not in ("bernoulli", "poisson", "gaussian"):
            raise ValueError(f"Unsupported family '{family}'. "
                             "Choose 'bernoulli', 'poisson', or 'gaussian'.")
        self.family = family
        # Dispersion parameter sigma_y: used only for Gaussian.
        # sigma_y is updated each M-step via calc_sigma_y().
        self.sigma_y = float(sigma_y)

    # ------------------------------------------------------------------
    # Template methods: A'(eta) and A''(eta)
    # ------------------------------------------------------------------

    def _mean_function(self, eta: np.ndarray) -> np.ndarray:
        """A'(eta) — conditional mean E[T(Y)|eta]."""
        if self.family == "bernoulli":
            return self._sigmoid(eta)
        if self.family == "poisson":
            # lambda = exp(eta), clip per Gemini Q3 feedback (lambda_max ~ 22026)
            return np.exp(np.clip(eta, -20, 10))
        # gaussian: identity link, A'(eta) = eta
        return eta.copy()

    def _variance_function(self, eta: np.ndarray) -> np.ndarray:
        """
        A''(eta) / phi — effective precision weight for Newton Hessian.

        Bernoulli : A''(eta) = s(1-s),          phi=1
        Poisson   : A''(eta) = exp(eta),         phi=1
        Gaussian  : A''(eta) = 1,  phi=sigma_y^2 → weight = 1/sigma_y^2
        """
        if self.family == "bernoulli":
            s = self._sigmoid(eta)
            return np.clip(s * (1.0 - s), 1e-8, None)
        if self.family == "poisson":
            return np.clip(np.exp(np.clip(eta, -20, 10)), 1e-8, None)
        # gaussian: constant 1/sigma_y^2
        return np.full_like(eta, 1.0 / max(self.sigma_y ** 2, 1e-8))

    # ------------------------------------------------------------------
    # E-step overrides
    # ------------------------------------------------------------------

    def _calc_gradient(self, X, Y, Z, F, sigma, var_z, w0, w, i):
        """
        Gradient of -ln f(Z|X,Y) w.r.t. z_i.

        General form (Eq.23 generalised):
          -(1/sigma_z^2) z_i  +  F^T Sigma^{-1}(x_i - F z_i)
          + (1/2) sum_{j!=i} [T(y_ij) - A'(eta_ij)] / phi  * w z_j

        phi = 1 for Bernoulli/Poisson; phi = sigma_y^2 for Gaussian.
        """
        n, k = Z.shape
        z_i = Z[i, :]
        x_i = X[i, :]

        # Term 1
        term1 = -(1.0 / var_z) * z_i

        # Term 2
        sigma_inv = np.diag(1.0 / np.diag(sigma))
        term2 = F.T @ sigma_inv @ (x_i - F @ z_i)

        # Term 3: (T(y) - A'(eta)) / phi
        eta = w0 + w * (Z @ z_i)          # (n,)
        mu = self._mean_function(eta)      # A'(eta), (n,)
        residual = Y[i, :] - mu            # T(y) - A'(eta)
        residual[i] = 0.0                  # exclude self
        if self.family == "gaussian":
            residual = residual / max(self.sigma_y ** 2, 1e-8)
        term3 = 0.5 * w * (Z.T @ residual)

        return -(term1 + term2 + term3)

    def _calc_precision_matrix(self, Z, F, sigma, var_z, w0, w, i):
        """
        Precision matrix A_i = -Hessian of ln f(Z|X,Y) at z_i.

        General form (Eq.22 generalised):
          (1/sigma_z^2) I  +  F^T Sigma^{-1} F
          + (1/2) sum_{j!=i} A''(eta_ij) w^2 z_j z_j^T
        """
        n, k = Z.shape
        z_i = Z[i, :]

        # Term 1
        term1 = (1.0 / var_z) * np.eye(k)

        # Term 2
        sigma_inv = np.diag(1.0 / np.diag(sigma))
        term2 = F.T @ sigma_inv @ F

        # Term 3: replace s(1-s) with A''(eta)
        eta = w0 + w * (Z @ z_i)          # (n,)
        var_fn = self._variance_function(eta)  # A''(eta), (n,)
        var_fn[i] = 0.0
        term3 = 0.5 * (w ** 2) * (Z.T @ np.diag(var_fn) @ Z)

        return term1 + term2 + term3

    # ------------------------------------------------------------------
    # M-step overrides
    # ------------------------------------------------------------------

    def _phi(self) -> float:
        """Dispersion parameter phi: 1.0 for Bernoulli/Poisson, sigma_y^2 for Gaussian."""
        if self.family == "gaussian":
            return max(self.sigma_y ** 2, 1e-8)
        return 1.0

    def calc_w0(self, Y, Z_samples, w0_init, w,
                max_iter=50, alpha=0.01,
                beta1=0.9, beta2=0.999, epsilon=1e-8, tol=1e-8):
        """Adam update for w0. Gradient: -(1/phi) * sum(T(y) - A'(eta))."""
        n, k, L = Z_samples.shape
        phi = self._phi()
        w0 = w0_init
        m = v = 0.0

        for t in range(1, max_iter + 1):
            w0_prev = w0
            grad_sum = 0.0
            for l in range(L):
                Z_l = Z_samples[:, :, l]
                eta = w0 + w * (Z_l @ Z_l.T)
                diff = Y - self._mean_function(eta)
                np.fill_diagonal(diff, 0.0)
                grad_sum += np.sum(diff)

            grad = -grad_sum / (2.0 * L * phi)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            w0 = w0 - alpha * m_hat / (np.sqrt(v_hat) + epsilon)

            if abs(w0 - w0_prev) < tol:
                break

        return w0

    def calc_w(self, Y, Z_samples, w0, w_init,
               max_iter=50, alpha=0.01,
               beta1=0.9, beta2=0.999, epsilon=1e-8, tol=1e-8):
        """Adam update for w. Gradient: -(1/phi) * sum((T(y) - A'(eta)) * z_i^T z_j)."""
        n, k, L = Z_samples.shape
        phi = self._phi()
        w = w_init
        m = v = 0.0

        for t in range(1, max_iter + 1):
            w_prev = w
            grad_sum = 0.0
            for l in range(L):
                Z_l = Z_samples[:, :, l]
                ZZT = Z_l @ Z_l.T
                eta = w0 + w * ZZT
                diff = Y - self._mean_function(eta)
                np.fill_diagonal(diff, 0.0)
                grad_sum += np.sum(diff * ZZT)

            grad = -grad_sum / (2.0 * L * phi)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            w = w - alpha * m_hat / (np.sqrt(v_hat) + epsilon)

            if abs(w - w_prev) < tol:
                break

        return w

    def calc_sigma_y(self, Y: np.ndarray, Z_samples: np.ndarray,
                     w0: float, w: float) -> float:
        """
        M-step update for Gaussian dispersion sigma_y (MLE).

        sigma_y^2 = mean_{l} mean_{i<j} (y_ij - eta_ij)^2

        Updates self.sigma_y in-place and returns new value.
        Only meaningful for family='gaussian'.
        """
        n, k, L = Z_samples.shape
        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        num_pairs = int(upper_mask.sum())

        ss = 0.0
        for l in range(L):
            Z_l = Z_samples[:, :, l]
            eta = w0 + w * (Z_l @ Z_l.T)
            resid_sq = (Y - eta) ** 2
            ss += float(np.sum(resid_sq[upper_mask]))

        sigma_sq = ss / (L * num_pairs)
        self.sigma_y = float(np.sqrt(max(sigma_sq, 1e-6)))
        return self.sigma_y

    # ------------------------------------------------------------------
    # Log-likelihood for Y (for Q function computation)
    # ------------------------------------------------------------------

    def calc_log_likelihood_Y(self, Y: np.ndarray, Z_samples: np.ndarray,
                               w0: float, w: float) -> float:
        """
        Compute (1/L) sum_l E[ln p(Y|Z_l)] for the current family.

        Note: Poisson omits -ln(Y!) (constant w.r.t. theta).
              Gaussian omits constants -n*ln(2*pi)/2.
        """
        n, k, L = Z_samples.shape
        ll = 0.0
        for l in range(L):
            Z_l = Z_samples[:, :, l]
            eta = w0 + w * (Z_l @ Z_l.T)

            if self.family == "bernoulli":
                S = self._sigmoid(np.clip(eta, -500, 500))
                S = np.clip(S, 1e-10, 1 - 1e-10)
                ln_p = Y * np.log(S) + (1 - Y) * np.log(1 - S)
            elif self.family == "poisson":
                eta_c = np.clip(eta, -20, 10)
                ln_p = Y * eta_c - np.exp(eta_c)
            else:  # gaussian
                sig2 = max(self.sigma_y ** 2, 1e-8)
                ln_p = -0.5 * (Y - eta) ** 2 / sig2 - 0.5 * np.log(sig2)

            np.fill_diagonal(ln_p, 0.0)
            ll += 0.5 * np.sum(ln_p)

        return ll / L

    def __repr__(self):
        extra = f", sigma_y={self.sigma_y:.4f}" if self.family == "gaussian" else ""
        return (f"ExpFamLatentStructuralModel("
                f"n={self.n}, d={self.d}, k={self.k}, "
                f"L={self.L}, family='{self.family}'{extra})")
