"""
DualExpFamLSMPerColumn — 列ごとに X の分布族を指定できる最小プロトタイプ。

動機（Q4 / KI 系）:
    現行実装は family_x が全列共通のスカラー文字列であり
    （model_dual_expfam.py L.85）、実データの混在属性
    （例: 年齢=Gaussian、会員フラグ=Bernoulli、購買数=Poisson）を
    正しく扱えない。本クラスは family_x_list（長さ d のリスト）で
    列ごとの指数型分布族を指定可能にする。

数理:
    x_il ~ ExpFam_{c(l)}( η_il = f_l^T z_i )   c(l) ∈ {gaussian, bernoulli, poisson}
    E-step Term2（勾配）: F^T [ w ⊙ (T(x_i) − A'_{c}(F z_i)) ]
        w_l = 1/σ_l²（Gaussian 列）、1（それ以外）
    E-step Term2（曲率）: F^T diag[c_l] F
        c_l = 1/σ_l²（Gaussian 列）、A''_{c(l)}(η_l)（それ以外）
    M-step F: 全列 Gaussian なら閉形式（継承）、混在なら重み付き Adam
    M-step Σ: Gaussian 列のみ MLE、他列は 1 固定

制約（プロトタイプの割り切り、設計書参照）:
    - family_y / pair mask は DualExpFamLSMMasked を継承（NB は未結合）
    - BIC の num_params は eval_utils.calc_bic_exp(family_x='mixed',
      n_gaussian_x_cols=...) を使うこと
    - utils_expfam.calc_bic_dual / calc_Q_dual_strict は使用不可
      （mixed を知らない）。eval_utils.calc_Q_dual_strict_exp を使う。
"""

import numpy as np
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from model_dual_expfam_masked import DualExpFamLSMMasked  # noqa: E402

_VALID = ("gaussian", "bernoulli", "poisson")


class DualExpFamLSMPerColumn(DualExpFamLSMMasked):
    """
    列ごとの X 分布族対応 Dual-ExpFam LSM（fixed 系列 + pair mask 対応）。

    Parameters
    ----------
    family_x_list : list[str]  長さ d。各列の分布族。
    """

    def __init__(self, n, d, k, L=10,
                 family_x_list=None, family_y="bernoulli",
                 sigma_y=1.0, train_mask=None):
        if family_x_list is None:
            family_x_list = ["gaussian"] * d
        if len(family_x_list) != d:
            raise ValueError(f"family_x_list length {len(family_x_list)} != d={d}")
        for f in family_x_list:
            if f not in _VALID:
                raise ValueError(f"Unsupported family '{f}' in family_x_list")

        # 親は有効な family_x を要求するので placeholder を渡し、直後に上書き
        super().__init__(n=n, d=d, k=k, L=L,
                         family_x="gaussian", family_y=family_y,
                         sigma_y=sigma_y, train_mask=train_mask)
        self.family_x_list = list(family_x_list)
        self._col_idx = {
            fam: np.array([j for j, f in enumerate(family_x_list) if f == fam],
                          dtype=int)
            for fam in _VALID
        }
        self._all_gaussian = (len(self._col_idx["gaussian"]) == d)
        # 'mixed' マーカー（全列同一でも per-column 経路を通す）
        self.family_x = "gaussian" if self._all_gaussian else "mixed"

    def columns_of(self, fam: str) -> np.ndarray:
        return self._col_idx[fam]

    # ------------------------------------------------------------------
    # X-side link functions（(d,) / (n,d) 両対応）
    # ------------------------------------------------------------------

    def _mean_function_x(self, eta_x):
        out = np.empty_like(np.asarray(eta_x, dtype=float))
        g, b, p = (self._col_idx["gaussian"], self._col_idx["bernoulli"],
                   self._col_idx["poisson"])
        if len(g):
            out[..., g] = eta_x[..., g]
        if len(b):
            out[..., b] = self._sigmoid(eta_x[..., b])
        if len(p):
            out[..., p] = np.exp(np.clip(eta_x[..., p], -20, 10))
        return out

    def _variance_function_x(self, eta_x):
        """A''(η)（族ごと）。Gaussian 列は 1（σ 重みは呼び出し側で扱う）。"""
        out = np.empty_like(np.asarray(eta_x, dtype=float))
        g, b, p = (self._col_idx["gaussian"], self._col_idx["bernoulli"],
                   self._col_idx["poisson"])
        if len(g):
            out[..., g] = 1.0
        if len(b):
            s = self._sigmoid(eta_x[..., b])
            out[..., b] = np.clip(s * (1.0 - s), 1e-8, None)
        if len(p):
            out[..., p] = np.clip(np.exp(np.clip(eta_x[..., p], -20, 10)),
                                  1e-8, None)
        return out

    def _x_weight_vector(self, sigma):
        """勾配用の列重み w_l = 1/σ_l²（Gaussian）or 1（その他）。"""
        wvec = np.ones(self.d)
        g = self._col_idx["gaussian"]
        if len(g):
            sd = np.maximum(np.diag(sigma)[g], 1e-8)
            wvec[g] = 1.0 / sd
        return wvec

    # ------------------------------------------------------------------
    # E-step overrides（Term2 を列重み付きに）
    # ------------------------------------------------------------------

    def _calc_gradient(self, X, Y, Z, F, sigma, var_z, w0, w, i):
        z_i = Z[i, :]
        x_i = X[i, :]

        term1 = -(1.0 / var_z) * z_i

        eta_x_i = F @ z_i
        residual_x = x_i - self._mean_function_x(eta_x_i)
        term2 = F.T @ (residual_x * self._x_weight_vector(sigma))

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
        curv = self._variance_function_x(eta_x_i) * self._x_weight_vector(sigma)
        term2 = F.T @ (F * curv[:, None])

        eta_y = w0 + w * (Z @ z_i)
        var_y = self._variance_function(eta_y) * self._mask_f[i, :]
        term3 = (w ** 2) * (Z.T @ np.diag(var_y) @ Z)

        return term1 + term2 + term3

    # ------------------------------------------------------------------
    # M-step overrides
    # ------------------------------------------------------------------

    def calc_F(self, X, Z_samples):
        if self._all_gaussian:
            return super().calc_F(X, Z_samples)   # 閉形式（継承）
        return self._calc_F_adam_weighted(X, Z_samples)

    def _calc_F_adam_weighted(self, X, Z_samples,
                              max_iter=50, lr=0.01,
                              beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-6):
        """
        混在列用 Adam。Gaussian 列の勾配は 1/σ_l² で重み付ける
        （既存 _calc_F_adam は重みなし = σ_l=1 相当のため補正）。
        """
        n, k, L = Z_samples.shape
        F = self.params["F"].copy()
        sigma = self.params["sigma"]
        wvec = self._x_weight_vector(sigma)
        m = np.zeros_like(F)
        v = np.zeros_like(F)

        for t in range(1, max_iter + 1):
            grad = np.zeros_like(F)
            for l in range(L):
                Z_l = Z_samples[:, :, l]
                eta_x = Z_l @ F.T
                residual = (X - self._mean_function_x(eta_x)) * wvec[None, :]
                grad += residual.T @ Z_l
            grad /= L

            g = -grad
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * g ** 2
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)
            F_new = F - lr * m_hat / (np.sqrt(v_hat) + eps)
            if np.max(np.abs(F_new - F)) < tol:
                return F_new
            F = F_new
        return F

    def calc_sigma(self, X, Z_samples, F):
        """Gaussian 列のみ σ_l² を MLE、他列は 1 固定。"""
        if self._all_gaussian:
            return super().calc_sigma(X, Z_samples, F)
        n, k, L = Z_samples.shape
        sigma_diag = np.ones(self.d)
        g = self._col_idx["gaussian"]
        if len(g):
            ss = np.zeros(len(g))
            for l in range(L):
                resid = X[:, g] - (Z_samples[:, :, l] @ F.T)[:, g]
                ss += np.mean(resid ** 2, axis=0)
            sigma_diag[g] = np.maximum(ss / L, 1e-6)
        return np.diag(sigma_diag)

    # ------------------------------------------------------------------
    # X log-likelihood（列ごと、Gaussian は ln2π 込み = 親規約準拠）
    # ------------------------------------------------------------------

    def calc_log_likelihood_X(self, X, Z_samples, F):
        n, k, L = Z_samples.shape
        sigma = self.params["sigma"]
        g, b, p = (self._col_idx["gaussian"], self._col_idx["bernoulli"],
                   self._col_idx["poisson"])
        ll = 0.0
        for l in range(L):
            eta_x = Z_samples[:, :, l] @ F.T
            if len(g):
                sd = np.maximum(np.diag(sigma)[g], 1e-8)
                resid = X[:, g] - eta_x[:, g]
                ll += float(np.sum(-0.5 * resid ** 2 / sd
                                   - 0.5 * np.log(sd)
                                   - 0.5 * np.log(2.0 * np.pi)))
            if len(b):
                S = self._sigmoid(np.clip(eta_x[:, b], -500, 500))
                S = np.clip(S, 1e-10, 1.0 - 1e-10)
                ll += float(np.sum(X[:, b] * np.log(S)
                                   + (1.0 - X[:, b]) * np.log(1.0 - S)))
            if len(p):
                eta_c = np.clip(eta_x[:, p], -20, 10)
                ll += float(np.sum(X[:, p] * eta_c - np.exp(eta_c)))
        return ll / L

    def __repr__(self):
        counts = {f: len(idx) for f, idx in self._col_idx.items() if len(idx)}
        return (f"DualExpFamLSMPerColumn(n={self.n}, d={self.d}, k={self.k}, "
                f"family_x_cols={counts}, family_y='{self.family}')")
