"""
DualExpFamLSMMasked — pair mask（strict held-out）対応の Dual-ExpFam LSM。

背景（KI-012）:
    既存 API（model_dual_expfam_fixed.py / model_expfam.py）は欠測ペアの
    マスクを持たず、held-out ペアを Y=0 に置くと Poisson/Bernoulli では
    「カウント 0 を観測した」ことになり学習が汚染される。
    本クラスは E-step（gradient / precision）と M-step（calc_w0 / calc_w /
    calc_sigma_y）および Y 対数尤度を、観測ペア集合（train_mask=True）に
    制限する。

数学的定義:
    観測ペア集合 O ⊂ {(i,j): i<j} に対し
      Q_Y = Σ_{(i,j)∈O} [T(y_ij) η_ij − A(η_ij)] / φ_Y
    ∂Q_Y/∂z_i = w Σ_{j: (i,j)∈O} [T(y_ij) − A'(η_ij)]/φ_Y · z_j   （1/2 なし: fixed 版準拠）
    −∂²Q_Y/∂z_i² = w² Σ_{j: (i,j)∈O} A''(η_ij)/φ_Y · z_j z_j^T

互換性:
    train_mask=None（全ペア観測）のとき DualExpFamLSMFixed と数値的に同一
    （test_experimental_models.py で検証）。既存クラスは変更しない。
"""

import numpy as np
import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent          # expfam/src
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent.parent / "reproduction" / "src"))
from model_dual_expfam_fixed import DualExpFamLSMFixed  # noqa: E402


class DualExpFamLSMMasked(DualExpFamLSMFixed):
    """
    pair mask 対応 Dual-ExpFam LSM（fixed 版継承、E-step 0.5 なし）。

    Parameters
    ----------
    train_mask : (n, n) bool array or None
        True = 学習に使う観測ペア。対称でなければならない。
        対角は強制的に False。None なら全ペア観測（fixed 版と同一挙動）。
    """

    def __init__(self, n, d, k, L=10,
                 family_x="gaussian", family_y="bernoulli",
                 sigma_y=1.0, train_mask=None):
        super().__init__(n=n, d=d, k=k, L=L,
                         family_x=family_x, family_y=family_y,
                         sigma_y=sigma_y)
        self.set_train_mask(train_mask)

    # ------------------------------------------------------------------
    # Mask handling
    # ------------------------------------------------------------------

    def set_train_mask(self, train_mask):
        if train_mask is None:
            tm = np.ones((self.n, self.n), dtype=bool)
        else:
            tm = np.asarray(train_mask, dtype=bool).copy()
            if tm.shape != (self.n, self.n):
                raise ValueError(
                    f"train_mask shape {tm.shape} != ({self.n}, {self.n})")
            if not np.array_equal(tm, tm.T):
                raise ValueError("train_mask must be symmetric")
        np.fill_diagonal(tm, False)
        self.train_mask = tm
        self._mask_f = tm.astype(np.float64)

    def n_train_pairs(self) -> int:
        """観測（学習）ペア数 |O|（上三角）。"""
        return int(np.triu(self.train_mask, k=1).sum())

    # ------------------------------------------------------------------
    # Y-side hooks（NB 版がオーバーライドする）
    # ------------------------------------------------------------------

    def _y_score_estep(self, y, mu):
        """E-step 用スコア dl/dη（1/φ_Y 込み）。array-generic。"""
        resid = y - mu
        if self.family == "gaussian":
            resid = resid / max(self.sigma_y ** 2, 1e-8)
        return resid

    def _y_residual_mstep(self, y, mu):
        """M-step (calc_w0/calc_w) 用残差。φ_Y は呼び出し側の /(2Lφ) が扱う。"""
        return y - mu

    # ------------------------------------------------------------------
    # E-step overrides（fixed 版 + mask）
    # ------------------------------------------------------------------

    def _calc_gradient(self, X, Y, Z, F, sigma, var_z, w0, w, i):
        z_i = Z[i, :]
        x_i = X[i, :]

        # Term 1: Z prior
        term1 = -(1.0 / var_z) * z_i

        # Term 2: X likelihood
        eta_x_i = F @ z_i
        mu_x_i = self._mean_function_x(eta_x_i)
        residual_x = x_i - mu_x_i
        if self.family_x == "gaussian":
            sigma_inv_diag = 1.0 / np.maximum(np.diag(sigma), 1e-8)
            term2 = F.T @ (residual_x * sigma_inv_diag)
        else:
            term2 = F.T @ residual_x

        # Term 3: Y likelihood（mask 適用、対角は mask=False で自動除外）
        eta_y = w0 + w * (Z @ z_i)
        mu_y = self._mean_function(eta_y)
        residual_y = self._y_score_estep(Y[i, :], mu_y) * self._mask_f[i, :]
        term3 = w * (Z.T @ residual_y)

        return -(term1 + term2 + term3)

    def _calc_precision_matrix(self, Z, F, sigma, var_z, w0, w, i):
        z_i = Z[i, :]
        k = Z.shape[1]

        term1 = (1.0 / var_z) * np.eye(k)

        eta_x_i = F @ z_i
        if self.family_x == "gaussian":
            sigma_inv_diag = 1.0 / np.maximum(np.diag(sigma), 1e-8)
            term2 = F.T @ (F * sigma_inv_diag[:, None])
        else:
            var_x_i = self._variance_function_x(eta_x_i)
            term2 = F.T @ (F * var_x_i[:, None])

        eta_y = w0 + w * (Z @ z_i)
        var_y = self._variance_function(eta_y) * self._mask_f[i, :]
        term3 = (w ** 2) * (Z.T @ np.diag(var_y) @ Z)

        return term1 + term2 + term3

    # ------------------------------------------------------------------
    # M-step overrides（mask 適用）
    # ------------------------------------------------------------------

    def calc_w0(self, Y, Z_samples, w0_init, w,
                max_iter=50, alpha=0.01,
                beta1=0.9, beta2=0.999, epsilon=1e-8, tol=1e-8):
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
                diff = self._y_residual_mstep(Y, self._mean_function(eta))
                diff = diff * self._mask_f
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
                diff = self._y_residual_mstep(Y, self._mean_function(eta))
                diff = diff * self._mask_f
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

    def calc_sigma_y(self, Y, Z_samples, w0, w):
        """Gaussian-Y dispersion の MLE（観測ペアのみ）。"""
        n, k, L = Z_samples.shape
        obs_upper = np.triu(self.train_mask, k=1)
        num_pairs = int(obs_upper.sum())
        if num_pairs == 0:
            return self.sigma_y

        ss = 0.0
        for l in range(L):
            Z_l = Z_samples[:, :, l]
            eta = w0 + w * (Z_l @ Z_l.T)
            resid_sq = (Y - eta) ** 2
            ss += float(np.sum(resid_sq[obs_upper]))

        sigma_sq = ss / (L * num_pairs)
        self.sigma_y = float(np.sqrt(max(sigma_sq, 1e-6)))
        return self.sigma_y

    # ------------------------------------------------------------------
    # Y log-likelihood（mask 適用、正規化定数の扱いは基底クラス準拠）
    # ------------------------------------------------------------------

    def calc_log_likelihood_Y(self, Y, Z_samples, w0, w):
        """
        (1/L) Σ_l E[ln p(Y_O | Z_l)]（観測ペアのみ）。

        基底クラス同様、Poisson は −ln(y!)、Gaussian は −(1/2)ln(2π) を省略。
        厳密な値が必要な場合は eval_utils.calc_Q_dual_strict_exp を使うこと。
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

            ln_p = ln_p * self._mask_f      # mask（対角 False 含む）
            ll += 0.5 * np.sum(ln_p)        # 両方向和 /2 = 上三角和（基底準拠）

        return ll / L

    def __repr__(self):
        return (f"DualExpFamLSMMasked(n={self.n}, d={self.d}, k={self.k}, "
                f"family_x='{self.family_x}', family_y='{self.family}', "
                f"n_train_pairs={self.n_train_pairs()})")
