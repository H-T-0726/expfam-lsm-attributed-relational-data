"""
DualExpFamLSMFixed — Y-side 0.5 係数を除去した修正版。

数学的根拠：
  Q_Y = Σ_{i<j} [T(y_ij) η_ij - A(η_ij)]  （上三角のみで定義）

  ∂Q_Y/∂z_i を計算すると:
    Σ_{j>i} [T(y_ij) - A'(η_ij)] w z_j
  + Σ_{j<i} [T(y_ji) - A'(η_ji)] w z_j       （y_ij=y_ji, η_ij=η_jiより）
  = w Σ_{j≠i} [T(y_ij) - A'(η_ij)] z_j       → 1/2 なし

  同様に Hessian の Y 側:
  -∂²Q_Y/∂z_i² = w² Σ_{j≠i} A''(η_ij) z_j z_j^T  → 1/2 なし

元の model_dual_expfam.py との差異:
  _calc_gradient      L.159: `0.5 * w * ...`  → `w * ...`
  _calc_precision_matrix L.200: `0.5 * (w**2) * ...`  → `(w**2) * ...`

互換性:
  既存コード (model_dual_expfam.py) は一切変更しない。
  本ファイルを import して DualExpFamLSMFixed を使うことで切り替える。
"""

import numpy as np
import sys
from pathlib import Path

_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))
from model_dual_expfam import DualExpFamLSM  # noqa: E402


class DualExpFamLSMFixed(DualExpFamLSM):
    """
    Y 側 Term3 の 1/2 係数を除去した修正版 DualExpFamLSM。

    オリジナル (DualExpFamLSM) との違いは _calc_gradient と
    _calc_precision_matrix の Term3 のみ。その他の挙動は同一。
    """

    # ------------------------------------------------------------------
    # E-step: gradient (Y-side 0.5 removed)
    # ------------------------------------------------------------------

    def _calc_gradient(self, X, Y, Z, F, sigma, var_z, w0, w, i) -> np.ndarray:
        """
        Gradient of -ln f(z_i | X, Y).

        Term 1  :  -(1/sigma_z^2) z_i
        Term 2  :  (1/phi_X) F^T [T_X(x_i) - A_X'(F z_i)]
        Term 3  :  w * Σ_{j≠i} [T_Y(y_ij) - A_Y'(η_ij)] z_j   ← 1/2 なし (fixed)
        """
        z_i = Z[i, :]
        x_i = X[i, :]

        # ── Term 1: Z prior ──────────────────────────────────────────
        term1 = -(1.0 / var_z) * z_i

        # ── Term 2: X likelihood ────────────────────────────────────
        eta_x_i = F @ z_i
        mu_x_i  = self._mean_function_x(eta_x_i)
        residual_x = x_i - mu_x_i

        if self.family_x == "gaussian":
            sigma_inv_diag = 1.0 / np.maximum(np.diag(sigma), 1e-8)
            term2 = F.T @ (residual_x * sigma_inv_diag)
        else:
            term2 = F.T @ residual_x

        # ── Term 3: Y likelihood (1/2 removed) ──────────────────────
        eta_y      = w0 + w * (Z @ z_i)
        mu_y       = self._mean_function(eta_y)
        residual_y = Y[i, :] - mu_y
        residual_y[i] = 0.0
        if self.family == "gaussian":
            residual_y = residual_y / max(self.sigma_y ** 2, 1e-8)
        term3 = w * (Z.T @ residual_y)          # ← 0.5 なし

        return -(term1 + term2 + term3)

    # ------------------------------------------------------------------
    # E-step: precision matrix (Y-side 0.5 removed)
    # ------------------------------------------------------------------

    def _calc_precision_matrix(self, Z, F, sigma, var_z, w0, w, i) -> np.ndarray:
        """
        Precision matrix A_i = -Hessian of ln f(z_i | X, Y).

        Term 1  :  (1/sigma_z^2) I
        Term 2  :  (1/phi_X) F^T diag[A_X''(F z_i)] F
        Term 3  :  w^2 * Σ_{j≠i} A_Y''(η_ij) z_j z_j^T   ← 1/2 なし (fixed)
        """
        z_i = Z[i, :]
        k   = Z.shape[1]

        # ── Term 1 ───────────────────────────────────────────────────
        term1 = (1.0 / var_z) * np.eye(k)

        # ── Term 2: X curvature ──────────────────────────────────────
        eta_x_i = F @ z_i

        if self.family_x == "gaussian":
            sigma_inv_diag = 1.0 / np.maximum(np.diag(sigma), 1e-8)
            term2 = F.T @ (F * sigma_inv_diag[:, None])
        else:
            var_x_i = self._variance_function_x(eta_x_i)
            term2 = F.T @ (F * var_x_i[:, None])

        # ── Term 3: Y curvature (1/2 removed) ────────────────────────
        eta_y    = w0 + w * (Z @ z_i)
        var_y    = self._variance_function(eta_y)
        var_y[i] = 0.0
        term3 = (w ** 2) * (Z.T @ np.diag(var_y) @ Z)   # ← 0.5 なし

        return term1 + term2 + term3

    def __repr__(self) -> str:
        sigma_y_str = (
            f", sigma_y={self.sigma_y:.4f}" if self.family == "gaussian" else ""
        )
        return (
            f"DualExpFamLSMFixed("
            f"n={self.n}, d={self.d}, k={self.k}, "
            f"family_x='{self.family_x}', family_y='{self.family}'"
            f"{sigma_y_str})"
        )
