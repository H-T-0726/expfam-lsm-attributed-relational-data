"""
DualExpFamLSMNB — Negative Binomial (NB2) 関係データ Y 対応モデル。

動機（KI-012）:
    MovieLens 共評価カウント Y は var/mean ≈ 9.9 の過分散を示し、
    Poisson（var/mean = 1）の仮定が破れている。NB2 は
        E[y] = μ,  Var[y] = μ + μ²/r
    により過分散を表現できる（r→∞ で Poisson に退化）。

数理的整理:
    NB2, log link:  μ_ij = exp(η_ij),  η_ij = w0 + w z_i^T z_j
    log p(y|μ,r) = ln Γ(y+r) − ln Γ(r) − ln Γ(y+1)
                   + y ln(μ/(μ+r)) + r ln(r/(μ+r))

    score:      ∂ln p/∂η = (y − μ) · r/(μ+r)
    curvature:  −∂²ln p/∂η² = r μ (y+r)/(μ+r)²
    Fisher 情報（E[y]=μ を代入）: μ r/(μ+r)      ← Newton/Laplace で使用
    r→∞: score → (y−μ), Fisher → μ（Poisson に一致）

    注意: r を固定した NB2 は自然パラメータ η'=ln(μ/(μ+r)) に関して
    指数型分布族だが、本実装は既存実装との整合のため log link
    （η=ln μ、非正準リンク）を用いる。Newton の重みには Fisher 情報
    （期待情報量）を使うため、Hessian は常に半正定値で安定。

dispersion r の扱い:
    本実装は r を **固定パラメータ** とする（コンストラクタで指定）。
    r の推定は eval_utils.moment_estimate_nb_r（Pearson 残差モーメント法）
    を推奨。プロファイル尤度推定は今後の課題。
    BIC では r を 1 パラメータとして数える（eval_utils.calc_bic_exp）。

実装上の注意:
    - 親クラスへの family_y は 'poisson' を渡す（exp リンク・初期化ロジックを
      再利用するため）。self.family == 'poisson' のままなので、
      utils_expfam.calc_Q_dual_strict は本モデルに使用しないこと
      （Poisson の階乗補正が二重・誤適用になる）。
      代わりに eval_utils.calc_Q_dual_strict_exp を使う。
    - family_y_label = 'nb' で判別する。
"""

import numpy as np
import sys
from pathlib import Path
from scipy.special import gammaln

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from model_dual_expfam_masked import DualExpFamLSMMasked  # noqa: E402


class DualExpFamLSMNB(DualExpFamLSMMasked):
    """
    NB2-Y（固定 dispersion r）+ pair mask 対応 Dual-ExpFam LSM。

    Parameters
    ----------
    nb_r : float
        NB2 dispersion（> 0）。Var[y] = μ + μ²/r。大きいほど Poisson に近い。
    train_mask : (n, n) bool or None
        観測ペアマスク（DualExpFamLSMMasked と同じ）。
    """

    def __init__(self, n, d, k, L=10,
                 family_x="gaussian", nb_r=10.0,
                 sigma_y=1.0, train_mask=None):
        if not np.isfinite(nb_r) or nb_r <= 0:
            raise ValueError(f"nb_r must be a positive finite float, got {nb_r}")
        # exp リンク・Poisson 系初期化を再利用するため family_y='poisson' で親を初期化
        super().__init__(n=n, d=d, k=k, L=L,
                         family_x=family_x, family_y="poisson",
                         sigma_y=sigma_y, train_mask=train_mask)
        self.nb_r = float(nb_r)
        self.family_y_label = "nb"

    # ------------------------------------------------------------------
    # Y-side link functions
    # ------------------------------------------------------------------
    # _mean_function は親（poisson: exp(clip(η))）をそのまま使う（μ = exp η）

    def _variance_function(self, eta):
        """Fisher 情報重み: μ r/(μ+r)（r→∞ で Poisson の μ に一致）。"""
        mu = np.exp(np.clip(eta, -20, 10))
        return np.clip(mu * self.nb_r / (mu + self.nb_r), 1e-8, None)

    def _y_score_estep(self, y, mu):
        """NB2 score: (y − μ) r/(μ+r)。"""
        return (y - mu) * self.nb_r / (mu + self.nb_r)

    def _y_residual_mstep(self, y, mu):
        return (y - mu) * self.nb_r / (mu + self.nb_r)

    # ------------------------------------------------------------------
    # Y log-likelihood（NB は正規化定数まで含む厳密値を返す）
    # ------------------------------------------------------------------

    def calc_log_likelihood_Y(self, Y, Z_samples, w0, w):
        """
        (1/L) Σ_l E[ln p(Y_O | Z_l)]（観測ペアのみ、**全定数込み**）。

        注: 基底 Poisson 実装が −ln(y!) を省略するのと異なり、NB は
        ln Γ(y+r) が r（モデル比較で動く量）を含むため全項を含める。
        Poisson との対数尤度比較は eval_utils の full-constant 版で行うこと。
        """
        n, k, L = Z_samples.shape
        r = self.nb_r
        ll = 0.0
        for l in range(L):
            Z_l = Z_samples[:, :, l]
            eta = np.clip(w0 + w * (Z_l @ Z_l.T), -20, 10)
            mu = np.exp(eta)
            ln_p = (gammaln(Y + r) - gammaln(r) - gammaln(Y + 1)
                    + Y * (eta - np.log(mu + r))
                    + r * (np.log(r) - np.log(mu + r)))
            ln_p = ln_p * self._mask_f
            ll += 0.5 * np.sum(ln_p)
        return ll / L

    def __repr__(self):
        return (f"DualExpFamLSMNB(n={self.n}, d={self.d}, k={self.k}, "
                f"family_x='{self.family_x}', family_y='nb', "
                f"nb_r={self.nb_r:.3f}, n_train_pairs={self.n_train_pairs()})")
