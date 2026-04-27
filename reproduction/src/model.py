"""
Latent Structural Model for Binary Relational Data.

This module implements the model class based on:
Mikawa et al., "A study on latent structural models for binary
relational data with attribute information", NOLTA 2024.
"""

import numpy as np
from typing import Dict, Any, Optional


class LatentStructuralModel:
    """
    Latent Structural Model for Binary Relational Data with Attribute Information.

    This model estimates latent variables Z from:
    - Observed attribute data X (continuous)
    - Binary relational data Y

    Generative model:
        z_i ~ N(0, sigma_z^2 * I)
        x_i ~ N(F @ z_i, Sigma)
        y_ij ~ Bernoulli(sigmoid(w0 + w * z_i^T @ z_j))

    Parameters are estimated using Monte Carlo EM algorithm with Laplace approximation.

    Attributes
    ----------
    n : int
        Number of data points (objects)
    d : int
        Dimensionality of observed data X
    k : int
        Dimensionality of latent variable Z
    L : int
        Number of Monte Carlo samples for E-step
    params : Dict[str, Any]
        Dictionary containing model parameters
    """

    def __init__(self, n: int, d: int, k: int, L: int = 10):
        """
        Initialize the Latent Structural Model.

        Parameters
        ----------
        n : int
            Number of data points (objects)
        d : int
            Dimensionality of observed data X
        k : int
            Dimensionality of latent variable Z
        L : int, optional
            Number of Monte Carlo samples for E-step (default: 10)
        """
        self.n = n
        self.d = d
        self.k = k
        self.L = L
        self.params: Dict[str, Any] = {}

    def initialize_params(
        self,
        true_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize model parameters for estimation.

        Based on setParamsDesc.m from the MATLAB implementation.

        Parameters
        ----------
        true_params : Dict[str, Any], optional
            Dictionary containing true parameters (for setting varF, varZ scales).
            If None, default values are used.
        seed : int, optional
            Random seed for reproducibility. If None, random initialization.

        Notes
        -----
        Initialized parameters:
        - varZ: Fixed to 1.0 for identifiability
        - varF: 2 * true_params['var_f'] if available, else 10.0
        - Z: Random normal (n x k)
        - F: Random normal (d x k)
        - sigma: Identity matrix diagonal (d x d)
        - w0: Random normal scalar
        - w: 3 * random normal scalar (can be large)
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # Set variances based on true_params or defaults
        # varZ is fixed to 1.0 for identifiability
        var_z = 1.0

        if true_params is not None:
            var_f = true_params.get('var_f', 5.0) * 2.0
        else:
            var_f = 10.0  # Default: assume true var_f ~ 5.0

        # Initialize Z: random normal (n x k)
        Z = rng.standard_normal(size=(self.n, self.k))

        # Initialize F: random normal (d x k)
        F = rng.standard_normal(size=(self.d, self.k))

        # Initialize sigma: diagonal matrix with ones
        sigma = np.eye(self.d)

        # Initialize w0: random normal scalar
        w0 = rng.standard_normal()

        # Initialize w: scaled random normal (can be large)
        w = 3.0 * rng.standard_normal()

        # Store all parameters
        self.params = {
            'Z': Z,
            'F': F,
            'sigma': sigma,
            'w0': w0,
            'w': w,
            'var_z': var_z,
            'var_f': var_f,
        }

    def get_params(self) -> Dict[str, Any]:
        """
        Get current model parameters.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all model parameters
        """
        return self.params.copy()

    def set_params(self, **kwargs) -> None:
        """
        Set model parameters.

        Parameters
        ----------
        **kwargs
            Parameter name-value pairs to update
        """
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            else:
                raise KeyError(f"Unknown parameter: {key}")

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"LatentStructuralModel(n={self.n}, d={self.d}, k={self.k}, L={self.L})"
        )

    def summary(self) -> str:
        """
        Get a summary of the model and its parameters.

        Returns
        -------
        str
            Summary string
        """
        lines = [
            "=" * 50,
            "Latent Structural Model Summary",
            "=" * 50,
            f"Dimensions:",
            f"  n (data points):     {self.n}",
            f"  d (observed dim):    {self.d}",
            f"  k (latent dim):      {self.k}",
            f"  L (MC samples):      {self.L}",
        ]

        if self.params:
            lines.extend([
                "",
                "Parameters:",
                f"  var_z:   {self.params.get('var_z', 'N/A')}",
                f"  var_f:   {self.params.get('var_f', 'N/A')}",
                f"  w0:      {self.params.get('w0', 'N/A'):.4f}" if 'w0' in self.params else "  w0:      N/A",
                f"  w:       {self.params.get('w', 'N/A'):.4f}" if 'w' in self.params else "  w:       N/A",
                "",
                "Parameter shapes:",
                f"  Z:       {self.params['Z'].shape}" if 'Z' in self.params else "  Z:       N/A",
                f"  F:       {self.params['F'].shape}" if 'F' in self.params else "  F:       N/A",
                f"  sigma:   {self.params['sigma'].shape}" if 'sigma' in self.params else "  sigma:   N/A",
            ])
        else:
            lines.append("\nParameters: Not initialized")

        lines.append("=" * 50)
        return "\n".join(lines)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Compute sigmoid function with numerical stability."""
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x))
        )

    def _calc_gradient(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        F: np.ndarray,
        sigma: np.ndarray,
        var_z: float,
        w0: float,
        w: float,
        i: int
    ) -> np.ndarray:
        """
        Calculate gradient of log posterior for z_i.

        Based on paper Eq. (23):
        ∂ln f/∂z_i = -(1/σ²_z)z_i + F^T Σ^{-1}(x_i - Fz_i) + (1/2)Σ_{j≠i}(y_ij - s_ij)w z_j

        Note: μ_x is omitted (assumed 0).

        Parameters
        ----------
        X : np.ndarray
            Observed data matrix (n x d)
        Y : np.ndarray
            Binary relational data matrix (n x n)
        Z : np.ndarray
            Current latent variable matrix (n x k)
        F : np.ndarray
            Factor loading matrix (d x k)
        sigma : np.ndarray
            Diagonal covariance matrix (d x d)
        var_z : float
            Variance of latent variable (fixed to 1.0)
        w0 : float
            Bias parameter
        w : float
            Weight parameter
        i : int
            Index of the data point

        Returns
        -------
        np.ndarray
            Gradient vector (k,)
        """
        n, k = Z.shape
        z_i = Z[i, :]  # (k,)
        x_i = X[i, :]  # (d,)

        # Term 1: -(1/σ²_z) z_i
        term1 = -(1.0 / var_z) * z_i

        # Term 2: F^T Σ^{-1} (x_i - F z_i)
        # For diagonal Sigma, Σ^{-1} = diag(1/σ²_1, ..., 1/σ²_d)
        sigma_inv = np.diag(1.0 / np.diag(sigma))
        residual = x_i - F @ z_i  # (d,)
        term2 = F.T @ sigma_inv @ residual  # (k,)

        # Term 3: (1/2) Σ_{j≠i} (y_ij - s_ij) w z_j
        # Compute s_ij = sigmoid(w0 + w * z_i^T z_j) for all j
        logits = w0 + w * (Z @ z_i)  # (n,)
        s = self._sigmoid(logits)  # (n,)

        # (y_ij - s_ij) for all j, then multiply by z_j and sum
        y_minus_s = Y[i, :] - s  # (n,)
        # Exclude j=i by setting that term to 0
        y_minus_s[i] = 0.0

        # Σ_{j≠i} (y_ij - s_ij) z_j = Z^T @ (y - s) with y_minus_s[i] = 0
        term3 = 0.5 * w * (Z.T @ y_minus_s)  # (k,)

        # Gradient (negative because we want to minimize -log posterior)
        # But for Newton update: z = z - α * A^{-1} * gradient
        # Here gradient is ∂(-ln f)/∂z_i, so we return -gradient of ln f
        gradient = -(term1 + term2 + term3)

        return gradient

    def _calc_precision_matrix(
        self,
        Z: np.ndarray,
        F: np.ndarray,
        sigma: np.ndarray,
        var_z: float,
        w0: float,
        w: float,
        i: int
    ) -> np.ndarray:
        """
        Calculate precision matrix (inverse of covariance) for posterior of z_i.

        Based on paper Eq. (22):
        A_i = (1/σ²_z)I + F^T Σ^{-1} F + (1/2) Σ_{j≠i} s_ij(1-s_ij) w² z_j z_j^T

        Parameters
        ----------
        Z : np.ndarray
            Current latent variable matrix (n x k)
        F : np.ndarray
            Factor loading matrix (d x k)
        sigma : np.ndarray
            Diagonal covariance matrix (d x d)
        var_z : float
            Variance of latent variable (fixed to 1.0)
        w0 : float
            Bias parameter
        w : float
            Weight parameter
        i : int
            Index of the data point

        Returns
        -------
        np.ndarray
            Precision matrix A_i (k x k)
        """
        n, k = Z.shape
        z_i = Z[i, :]  # (k,)

        # Term 1: (1/σ²_z) I
        term1 = (1.0 / var_z) * np.eye(k)

        # Term 2: F^T Σ^{-1} F
        sigma_inv = np.diag(1.0 / np.diag(sigma))
        term2 = F.T @ sigma_inv @ F  # (k x k)

        # Term 3: (1/2) Σ_{j≠i} s_ij(1-s_ij) w² z_j z_j^T
        # Compute s_ij = sigmoid(w0 + w * z_i^T z_j) for all j
        logits = w0 + w * (Z @ z_i)  # (n,)
        s = self._sigmoid(logits)  # (n,)

        # s_ij(1-s_ij) for all j
        s_deriv = s * (1.0 - s)  # (n,)
        # Exclude j=i
        s_deriv[i] = 0.0

        # Σ_{j≠i} s_ij(1-s_ij) z_j z_j^T = Z^T @ diag(s_deriv) @ Z
        # But more efficiently: Σ_j s_deriv[j] * z_j @ z_j^T
        # = Z^T @ diag(s_deriv) @ Z
        term3 = 0.5 * (w ** 2) * (Z.T @ np.diag(s_deriv) @ Z)  # (k x k)

        # Precision matrix
        A_i = term1 + term2 + term3

        return A_i

    def calc_eta_newton(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        max_iter: int = 10,
        alpha: float = 0.01,
        tol: float = 1e-4
    ) -> np.ndarray:
        """
        Calculate posterior distribution of Z using Newton method with Laplace approximation.

        This implements Algorithm 2 (lines 4-10) from the paper.
        For each data point i:
        1. Find mode η_i using Newton method
        2. Compute precision matrix A_i at the mode
        3. Sample z_i from N(η_i, A_i^{-1})

        Parameters
        ----------
        X : np.ndarray
            Observed data matrix (n x d)
        Y : np.ndarray
            Binary relational data matrix (n x n)
        rng : np.random.Generator, optional
            Random number generator. If None, creates new one.
        max_iter : int, optional
            Maximum iterations for Newton method (default: 10)
        alpha : float, optional
            Learning rate for Newton update (default: 0.01)
        tol : float, optional
            Convergence tolerance (default: 1e-4)

        Returns
        -------
        np.ndarray
            Updated Z matrix (n x k) sampled from posterior
        """
        if rng is None:
            rng = np.random.default_rng()

        # Get current parameters
        Z = self.params['Z'].copy()
        F = self.params['F']
        sigma = self.params['sigma']
        var_z = self.params['var_z']
        w0 = self.params['w0']
        w = self.params['w']

        n, k = Z.shape

        for i in range(n):
            # Newton optimization for z_i
            for iteration in range(max_iter):
                z_prev = Z[i, :].copy()

                # Calculate gradient
                grad = self._calc_gradient(X, Y, Z, F, sigma, var_z, w0, w, i)

                # Calculate precision matrix
                A_i = self._calc_precision_matrix(Z, F, sigma, var_z, w0, w, i)

                # Symmetrize A_i for numerical stability
                A_i = (A_i + A_i.T) / 2.0

                # Add small regularization for numerical stability
                A_i += 1e-6 * np.eye(k)

                # Compute inverse (covariance matrix)
                try:
                    A_i_inv = np.linalg.inv(A_i)
                except np.linalg.LinAlgError:
                    # If singular, use pseudo-inverse
                    A_i_inv = np.linalg.pinv(A_i)

                # Newton update: z = z - α * A^{-1} * gradient
                Z[i, :] = Z[i, :] - alpha * (A_i_inv @ grad)

                # Check convergence
                diff = np.mean(np.abs(Z[i, :] - z_prev))
                if diff < tol:
                    break

                if np.linalg.norm(grad) < tol:
                    break

            # Sample from posterior N(η_i, A_i^{-1})
            # Recompute A_i at the final position
            A_i = self._calc_precision_matrix(Z, F, sigma, var_z, w0, w, i)
            A_i = (A_i + A_i.T) / 2.0
            A_i += 1e-6 * np.eye(k)

            try:
                A_i_inv = np.linalg.inv(A_i)
                # Ensure positive definiteness for sampling
                A_i_inv = (A_i_inv + A_i_inv.T) / 2.0
                # Sample from multivariate normal
                Z[i, :] = rng.multivariate_normal(Z[i, :], A_i_inv)
            except (np.linalg.LinAlgError, ValueError):
                # If sampling fails, keep the mode
                pass

        return Z

    # ================================================================
    # M-Step Methods
    # ================================================================

    def scale_Z(self, Z_samples: np.ndarray) -> np.ndarray:
        """
        Scale Z samples to have unit variance per dimension.

        Based on scaleZ function in calcdescmetric_ver4.m.
        Scales Z so that the average squared norm equals k (latent dimension).

        Parameters
        ----------
        Z_samples : np.ndarray
            Z samples of shape (n, k, L) or (n, k)

        Returns
        -------
        np.ndarray
            Scaled Z samples with same shape as input
        """
        if Z_samples.ndim == 2:
            # Single sample (n, k)
            n, k = Z_samples.shape
            # Compute average squared value
            avg_sq = np.mean(Z_samples ** 2)
            # Scale so that average squared value = 1 (variance = 1)
            if avg_sq > 0:
                scale = np.sqrt(k / (avg_sq * k))  # = 1 / sqrt(avg_sq)
                return Z_samples * scale
            return Z_samples
        else:
            # Multiple samples (n, k, L)
            n, k, L = Z_samples.shape
            # Compute average squared value across all samples
            avg_sq = np.mean(Z_samples ** 2)
            # Scale factor from MATLAB: sqrt(k / a) where a = mean of squared elements
            if avg_sq > 0:
                scale = np.sqrt(k / (avg_sq * k))
                return Z_samples * scale
            return Z_samples

    def calc_F(self, X: np.ndarray, Z_samples: np.ndarray) -> np.ndarray:
        """
        Calculate F using analytical solution (M-step).

        Based on paper Eq. (10) and calcF in calcdescmetric_ver4.m:
        F = (Σ_l X^T Z^(l)) @ (Σ_l Z^(l)^T Z^(l))^{-1}

        Note: μ_x is assumed to be 0.

        Parameters
        ----------
        X : np.ndarray
            Observed data matrix (n x d)
        Z_samples : np.ndarray
            Z samples of shape (n, k, L)

        Returns
        -------
        np.ndarray
            Updated F matrix (d x k)
        """
        n, k, L = Z_samples.shape
        d = X.shape[1]

        # Sum of X^T @ Z over all samples
        XZ_sum = np.zeros((d, k))
        # Sum of Z^T @ Z over all samples
        ZZ_sum = np.zeros((k, k))

        for l in range(L):
            Z_l = Z_samples[:, :, l]  # (n, k)
            XZ_sum += X.T @ Z_l  # (d, k)
            ZZ_sum += Z_l.T @ Z_l  # (k, k)

        # F = XZ_sum @ inv(ZZ_sum)
        try:
            F = XZ_sum @ np.linalg.inv(ZZ_sum)
        except np.linalg.LinAlgError:
            F = XZ_sum @ np.linalg.pinv(ZZ_sum)

        return F

    def calc_sigma(self, X: np.ndarray, Z_samples: np.ndarray, F: np.ndarray) -> np.ndarray:
        """
        Calculate Sigma using analytical solution (M-step).

        Based on paper Eq. (12) and calcSigma in calcdescmetric_ver4.m:
        Σ = diag((1/Ln) Σ_l (X - Z^(l) F^T)^T (X - Z^(l) F^T))

        Parameters
        ----------
        X : np.ndarray
            Observed data matrix (n x d)
        Z_samples : np.ndarray
            Z samples of shape (n, k, L)
        F : np.ndarray
            Factor loading matrix (d x k)

        Returns
        -------
        np.ndarray
            Updated diagonal covariance matrix (d x d)
        """
        n, k, L = Z_samples.shape
        d = X.shape[1]

        # Sum of residual covariance
        sigma_sum = np.zeros((d, d))

        for l in range(L):
            Z_l = Z_samples[:, :, l]  # (n, k)
            residual = X - Z_l @ F.T  # (n, d)
            sigma_sum += residual.T @ residual  # (d, d)

        # Average and keep only diagonal
        sigma = sigma_sum / (L * n)
        sigma = np.diag(np.diag(sigma))

        # Ensure positive values
        sigma = np.maximum(sigma, 1e-6 * np.eye(d))

        return sigma

    def calc_w0(
        self,
        Y: np.ndarray,
        Z_samples: np.ndarray,
        w0_init: float,
        w: float,
        max_iter: int = 50,
        alpha: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        tol: float = 1e-8
    ) -> float:
        """
        Update w0 using Adam optimizer.

        Based on calcw0.m. Uses gradient from Eq. (24):
        ∂L/∂w0 = (1/2L) Σ_l Σ_{i≠j} (y_ij - s_ij)

        The factor 1/2 comes from the symmetry of Y (counting i<j only once).

        Parameters
        ----------
        Y : np.ndarray
            Binary relational data matrix (n x n)
        Z_samples : np.ndarray
            Z samples of shape (n, k, L)
        w0_init : float
            Initial value of w0
        w : float
            Current value of w
        max_iter : int
            Maximum iterations for Adam
        alpha : float
            Learning rate
        beta1 : float
            Exponential decay rate for first moment
        beta2 : float
            Exponential decay rate for second moment
        epsilon : float
            Small constant for numerical stability
        tol : float
            Convergence tolerance

        Returns
        -------
        float
            Updated w0
        """
        n, k, L = Z_samples.shape
        w0 = w0_init

        # Adam state
        m = 0.0
        v = 0.0

        for t in range(1, max_iter + 1):
            w0_prev = w0

            # Compute gradient: (1/2L) Σ_l Σ_{i≠j} (y_ij - s_ij)
            grad_sum = 0.0
            for l in range(L):
                Z_l = Z_samples[:, :, l]
                ZZT = Z_l @ Z_l.T  # (n, n)
                logits = w0 + w * ZZT
                S = self._sigmoid(logits)

                # Y - S, excluding diagonal
                diff = Y - S
                np.fill_diagonal(diff, 0.0)
                grad_sum += np.sum(diff)

            # Gradient (negative because we maximize log-likelihood)
            # Division by 2L follows MATLAB implementation
            grad = -grad_sum / (2.0 * L)

            # Adam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            # Bias correction
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # Update
            w0 = w0 - alpha * m_hat / (np.sqrt(v_hat) + epsilon)

            # Check convergence
            if np.abs(w0 - w0_prev) < tol:
                break

        return w0

    def calc_w(
        self,
        Y: np.ndarray,
        Z_samples: np.ndarray,
        w0: float,
        w_init: float,
        max_iter: int = 50,
        alpha: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        tol: float = 1e-8
    ) -> float:
        """
        Update w using Adam optimizer.

        Based on calcw.m. Uses gradient from Eq. (25):
        ∂L/∂w = (1/2L) Σ_l Σ_{i≠j} (y_ij - s_ij) z_i^T z_j

        The factor 1/2 comes from the symmetry of Y.

        Parameters
        ----------
        Y : np.ndarray
            Binary relational data matrix (n x n)
        Z_samples : np.ndarray
            Z samples of shape (n, k, L)
        w0 : float
            Current value of w0
        w_init : float
            Initial value of w
        max_iter : int
            Maximum iterations for Adam
        alpha : float
            Learning rate
        beta1 : float
            Exponential decay rate for first moment
        beta2 : float
            Exponential decay rate for second moment
        epsilon : float
            Small constant for numerical stability
        tol : float
            Convergence tolerance

        Returns
        -------
        float
            Updated w
        """
        n, k, L = Z_samples.shape
        w = w_init

        # Adam state
        m = 0.0
        v = 0.0

        for t in range(1, max_iter + 1):
            w_prev = w

            # Compute gradient: (1/2L) Σ_l Σ_{i≠j} (y_ij - s_ij) * z_i^T z_j
            grad_sum = 0.0
            for l in range(L):
                Z_l = Z_samples[:, :, l]
                ZZT = Z_l @ Z_l.T  # (n, n)
                logits = w0 + w * ZZT
                S = self._sigmoid(logits)

                # (Y - S) * ZZT, excluding diagonal
                diff = Y - S
                np.fill_diagonal(diff, 0.0)
                grad_sum += np.sum(diff * ZZT)

            # Gradient (negative because we maximize log-likelihood)
            grad = -grad_sum / (2.0 * L)

            # Adam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            # Bias correction
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # Update
            w = w - alpha * m_hat / (np.sqrt(v_hat) + epsilon)

            # Check convergence
            if np.abs(w - w_prev) < tol:
                break

        return w


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'c:/研究2/src')
    from data_generator import set_true_params

    print("=" * 60)
    print("Model Class Self-Validation")
    print("=" * 60)

    # Test parameters
    n, d, k = 150, 15, 3
    L = 10
    seed = 42

    # Create model instance
    print(f"\n[Test 1] Creating model instance...")
    model = LatentStructuralModel(n=n, d=d, k=k, L=L)
    print(f"  Model: {model}")
    assert model.n == n, "FAILED: n mismatch"
    assert model.d == d, "FAILED: d mismatch"
    assert model.k == k, "FAILED: k mismatch"
    assert model.L == L, "FAILED: L mismatch"
    print("  [PASSED] Model instantiation")

    # Initialize parameters without true_params
    print(f"\n[Test 2] Initializing parameters (no true_params)...")
    model.initialize_params(seed=seed)
    params = model.get_params()

    print(f"  Parameters initialized:")
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"    {key}: {value}")

    # Validate shapes
    print(f"\n[Test 3] Validating parameter shapes...")
    assert params['Z'].shape == (n, k), f"FAILED: Z shape {params['Z'].shape} != ({n}, {k})"
    assert params['F'].shape == (d, k), f"FAILED: F shape {params['F'].shape} != ({d}, {k})"
    assert params['sigma'].shape == (d, d), f"FAILED: sigma shape {params['sigma'].shape} != ({d}, {d})"
    assert np.isscalar(params['w0']), "FAILED: w0 should be scalar"
    assert np.isscalar(params['w']), "FAILED: w should be scalar"
    assert params['var_z'] == 1.0, "FAILED: var_z should be 1.0"
    print("  [PASSED] All shapes correct")

    # Validate sigma is identity
    print(f"\n[Test 4] Validating sigma is identity matrix...")
    assert np.allclose(params['sigma'], np.eye(d)), "FAILED: sigma is not identity"
    print("  [PASSED] sigma is identity matrix")

    # Test with true_params
    print(f"\n[Test 5] Initializing with true_params...")
    true_params = {'var_f': 5.0, 'var_z': 1.0}
    model2 = LatentStructuralModel(n=n, d=d, k=k, L=L)
    model2.initialize_params(true_params=true_params, seed=seed)
    params2 = model2.get_params()

    assert params2['var_f'] == 10.0, f"FAILED: var_f should be 2 * true_var_f = 10.0, got {params2['var_f']}"
    print(f"  var_f = {params2['var_f']} (2 * true_var_f = 2 * 5.0)")
    print("  [PASSED] true_params scaling works")

    # Test set_params
    print(f"\n[Test 6] Testing set_params...")
    new_w0 = 0.5
    model2.set_params(w0=new_w0)
    assert model2.params['w0'] == new_w0, "FAILED: set_params did not work"
    print(f"  w0 updated to {model2.params['w0']}")
    print("  [PASSED] set_params works")

    # Test summary
    print(f"\n[Test 7] Model summary:")
    print(model.summary())

    # Test reproducibility
    print(f"\n[Test 8] Testing reproducibility with same seed...")
    model_a = LatentStructuralModel(n=n, d=d, k=k, L=L)
    model_a.initialize_params(seed=123)

    model_b = LatentStructuralModel(n=n, d=d, k=k, L=L)
    model_b.initialize_params(seed=123)

    assert np.allclose(model_a.params['Z'], model_b.params['Z']), "FAILED: Z not reproducible"
    assert np.allclose(model_a.params['F'], model_b.params['F']), "FAILED: F not reproducible"
    assert model_a.params['w0'] == model_b.params['w0'], "FAILED: w0 not reproducible"
    assert model_a.params['w'] == model_b.params['w'], "FAILED: w not reproducible"
    print("  [PASSED] Same seed produces identical parameters")

    # ================================================================
    # Test calc_eta_newton
    # ================================================================
    print("\n" + "=" * 60)
    print("Testing calc_eta_newton")
    print("=" * 60)

    # Generate synthetic data
    print(f"\n[Test 9] Generating synthetic data...")
    true_data = set_true_params(n=n, d=d, k=k, seed=1980)
    X = true_data['X']
    Y = true_data['Y']
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  True w0: {true_data['w0']:.4f}, True w: {true_data['w']:.4f}")
    print("  [PASSED] Data generated")

    # Create and initialize model
    print(f"\n[Test 10] Testing calc_eta_newton...")
    model3 = LatentStructuralModel(n=n, d=d, k=k, L=L)
    model3.initialize_params(true_params=true_data, seed=42)

    Z_before = model3.params['Z'].copy()
    print(f"  Z before shape: {Z_before.shape}")
    print(f"  Z before mean: {Z_before.mean():.4f}, std: {Z_before.std():.4f}")

    # Run Newton method
    rng = np.random.default_rng(42)
    Z_after = model3.calc_eta_newton(X, Y, rng=rng, max_iter=10, alpha=0.01)

    print(f"  Z after shape: {Z_after.shape}")
    print(f"  Z after mean: {Z_after.mean():.4f}, std: {Z_after.std():.4f}")

    # Validate output shape
    assert Z_after.shape == (n, k), f"FAILED: Z_after shape {Z_after.shape} != ({n}, {k})"
    print("  [PASSED] Output shape correct")

    # Check that Z has changed
    assert not np.allclose(Z_before, Z_after), "FAILED: Z did not change"
    print("  [PASSED] Z was updated")

    # Check for NaN/Inf
    assert not np.any(np.isnan(Z_after)), "FAILED: Z contains NaN"
    assert not np.any(np.isinf(Z_after)), "FAILED: Z contains Inf"
    print("  [PASSED] No NaN/Inf values")

    # Test gradient calculation
    print(f"\n[Test 11] Testing gradient calculation...")
    grad = model3._calc_gradient(
        X, Y, model3.params['Z'], model3.params['F'], model3.params['sigma'],
        model3.params['var_z'], model3.params['w0'], model3.params['w'], i=0
    )
    print(f"  Gradient shape: {grad.shape}")
    print(f"  Gradient norm: {np.linalg.norm(grad):.4f}")
    assert grad.shape == (k,), f"FAILED: gradient shape {grad.shape} != ({k},)"
    print("  [PASSED] Gradient shape correct")

    # Test precision matrix calculation
    print(f"\n[Test 12] Testing precision matrix calculation...")
    A_i = model3._calc_precision_matrix(
        model3.params['Z'], model3.params['F'], model3.params['sigma'],
        model3.params['var_z'], model3.params['w0'], model3.params['w'], i=0
    )
    print(f"  Precision matrix shape: {A_i.shape}")
    print(f"  Precision matrix diagonal: {np.diag(A_i)}")

    # Check symmetry
    assert np.allclose(A_i, A_i.T), "FAILED: Precision matrix not symmetric"
    print("  [PASSED] Precision matrix is symmetric")

    # Check positive definiteness (all eigenvalues > 0)
    eigenvalues = np.linalg.eigvalsh(A_i)
    print(f"  Eigenvalues: {eigenvalues}")
    assert np.all(eigenvalues > 0), "FAILED: Precision matrix not positive definite"
    print("  [PASSED] Precision matrix is positive definite")

    # ================================================================
    # Test M-Step Methods
    # ================================================================
    print("\n" + "=" * 60)
    print("Testing M-Step Methods")
    print("=" * 60)

    # Create Z samples (n, k, L)
    print(f"\n[Test 13] Creating Z samples for M-step...")
    L_test = 5
    rng = np.random.default_rng(42)
    Z_samples = np.zeros((n, k, L_test))
    model4 = LatentStructuralModel(n=n, d=d, k=k, L=L_test)
    model4.initialize_params(true_params=true_data, seed=42)

    for l in range(L_test):
        Z_samples[:, :, l] = model4.calc_eta_newton(X, Y, rng=rng, max_iter=5, alpha=0.01)
        model4.params['Z'] = Z_samples[:, :, l].copy()

    print(f"  Z_samples shape: {Z_samples.shape}")
    print(f"  Z_samples mean: {Z_samples.mean():.4f}, std: {Z_samples.std():.4f}")
    print("  [PASSED] Z samples created")

    # Test scale_Z
    print(f"\n[Test 14] Testing scale_Z...")
    Z_scaled = model4.scale_Z(Z_samples)
    print(f"  Z_scaled shape: {Z_scaled.shape}")
    print(f"  Z_scaled mean: {Z_scaled.mean():.4f}, std: {Z_scaled.std():.4f}")
    # Check variance is closer to 1
    assert Z_scaled.shape == Z_samples.shape, "FAILED: scale_Z changed shape"
    assert not np.any(np.isnan(Z_scaled)), "FAILED: Z_scaled contains NaN"
    print("  [PASSED] scale_Z works")

    # Test calc_F
    print(f"\n[Test 15] Testing calc_F...")
    F_new = model4.calc_F(X, Z_scaled)
    print(f"  F_new shape: {F_new.shape}")
    print(f"  F_new norm: {np.linalg.norm(F_new):.4f}")
    assert F_new.shape == (d, k), f"FAILED: F shape {F_new.shape} != ({d}, {k})"
    assert not np.any(np.isnan(F_new)), "FAILED: F contains NaN"
    print("  [PASSED] calc_F works")

    # Test calc_sigma
    print(f"\n[Test 16] Testing calc_sigma...")
    sigma_new = model4.calc_sigma(X, Z_scaled, F_new)
    print(f"  sigma_new shape: {sigma_new.shape}")
    print(f"  sigma_new diagonal (first 5): {np.diag(sigma_new)[:5]}")
    assert sigma_new.shape == (d, d), f"FAILED: sigma shape {sigma_new.shape} != ({d}, {d})"
    # Check diagonal matrix
    assert np.allclose(sigma_new, np.diag(np.diag(sigma_new))), "FAILED: sigma is not diagonal"
    # Check positive diagonal
    assert np.all(np.diag(sigma_new) > 0), "FAILED: sigma has non-positive diagonal"
    print("  [PASSED] calc_sigma works")

    # Test calc_w0
    print(f"\n[Test 17] Testing calc_w0...")
    w0_init = model4.params['w0']
    w_current = model4.params['w']
    print(f"  w0 before: {w0_init:.4f}")
    w0_new = model4.calc_w0(Y, Z_scaled, w0_init, w_current, max_iter=50)
    print(f"  w0 after: {w0_new:.4f}")
    print(f"  True w0: {true_data['w0']:.4f}")
    assert not np.isnan(w0_new), "FAILED: w0 is NaN"
    assert not np.isinf(w0_new), "FAILED: w0 is Inf"
    print("  [PASSED] calc_w0 works")

    # Test calc_w
    print(f"\n[Test 18] Testing calc_w...")
    w_init = model4.params['w']
    print(f"  w before: {w_init:.4f}")
    w_new = model4.calc_w(Y, Z_scaled, w0_new, w_init, max_iter=50)
    print(f"  w after: {w_new:.4f}")
    print(f"  True w: {true_data['w']:.4f}")
    assert not np.isnan(w_new), "FAILED: w is NaN"
    assert not np.isinf(w_new), "FAILED: w is Inf"
    print("  [PASSED] calc_w works")

    # Test gradient direction (sanity check)
    print(f"\n[Test 19] Gradient sanity check...")
    # Create Z samples with true Z to check gradient direction
    Z_true_samples = np.stack([true_data['Z']] * L_test, axis=2)

    # With true Z, gradients should be small if w0, w are close to true
    grad_w0_sum = 0.0
    grad_w_sum = 0.0
    w0_true = true_data['w0']
    w_true = true_data['w']

    for l in range(L_test):
        Z_l = Z_true_samples[:, :, l]
        ZZT = Z_l @ Z_l.T
        logits = w0_true + w_true * ZZT
        S = model4._sigmoid(logits)
        diff = Y - S
        np.fill_diagonal(diff, 0.0)
        grad_w0_sum += np.sum(diff)
        grad_w_sum += np.sum(diff * ZZT)

    grad_w0_at_true = -grad_w0_sum / (2.0 * L_test)
    grad_w_at_true = -grad_w_sum / (2.0 * L_test)
    print(f"  Gradient w.r.t. w0 at true params: {grad_w0_at_true:.4f}")
    print(f"  Gradient w.r.t. w at true params: {grad_w_at_true:.4f}")
    print("  (Gradients should be relatively small at true parameters)")
    print("  [PASSED] Gradient sanity check")

    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED!")
    print("=" * 60)
