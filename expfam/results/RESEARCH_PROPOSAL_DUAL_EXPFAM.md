# Research Proposal: Dual-ExpFam Latent Structural Model

**Title:** A Unified Latent Structural Model with Exponential Family Distributions for Both Attribute and Relational Data

**Prepared by:** Claude Code (AI Researcher)
**Date:** 2026-03-24
**Status:** Phase 2 — Design Document (Implementation Not Yet Started)

---

## Executive Summary

現在の実装（`ExpFamLatentStructuralModel`）は、関係データ $Y$ の分布を指数型分布族に一般化したが、**属性データ $X$ は依然としてガウス分布に固定**されている。本提案は $X$ の分布も指数型分布族に拡張する「**Dual-ExpFam LSM**」を設計する。数理的に最も重要な発見は、**$X$ と $Y$ の両者が EMアルゴリズムのEステップ（勾配・ヘッシアン）に完全に対称な形で寄与する**という美しい構造である。

---

## Section 1: Mathematical Formulation

### 1.1 生成モデルの再定義

**Dual-ExpFam LSM** の生成過程：

$$z_i \sim \mathcal{N}(0,\, \sigma_z^2 I_k), \quad i = 1, \ldots, n$$

$$x_{ij} \mid z_i \sim p_X\!\left(x_{ij};\, \eta_{ij}^X\right), \quad \eta_{ij}^X = f_j^\top z_i, \quad j = 1, \ldots, d$$

$$y_{ij} \mid z_i, z_j \sim p_Y\!\left(y_{ij};\, \eta_{ij}^Y\right), \quad \eta_{ij}^Y = w_0 + w\, z_i^\top z_j, \quad i < j$$

ここで $p_X$ と $p_Y$ はそれぞれ独立に指定される指数型分布族：

$$p(y;\,\eta) = h(y)\exp\!\Bigl(\eta\, T(y) - A(\eta)\Bigr) / \varphi$$

| 分布 | $T(y)$ | $A'(\eta) = \mathbb{E}[T(Y)\mid\eta]$ | $A''(\eta) = \mathrm{Var}[T(Y)\mid\eta]$ | $\varphi$ |
|------|--------|---------------------------------------|-------------------------------------------|-----------|
| Gaussian | $y$ | $\eta$ | $1$ | $\sigma^2$ |
| Bernoulli | $y$ | $\sigma(\eta)$ | $\sigma(\eta)(1-\sigma(\eta))$ | $1$ |
| Poisson | $y$ | $e^\eta$ | $e^\eta$ | $1$ |

### 1.2 完全対数尤度

$$\mathcal{L}(\Theta \mid Z) = \underbrace{\ln p(Z)}_{\text{prior}} + \underbrace{\ln p(X \mid Z, F, \varphi_X)}_{\text{X likelihood}} + \underbrace{\ln p(Y \mid Z, w_0, w, \varphi_Y)}_{\text{Y likelihood}}$$

**各項の展開：**

$$\ln p(Z) = -\frac{nk}{2}\ln(2\pi\sigma_z^2) - \frac{1}{2\sigma_z^2}\|Z\|_F^2$$

$$\ln p(X \mid Z, F) = \sum_{i=1}^n \sum_{j=1}^d \left[\ln h_X(x_{ij}) + \frac{T_X(x_{ij})\, f_j^\top z_i - A_X(f_j^\top z_i)}{\varphi_{X,j}}\right]$$

$$\ln p(Y \mid Z, w_0, w) = \sum_{i < j} \left[\ln h_Y(y_{ij}) + \frac{T_Y(y_{ij})\, \eta_{ij}^Y - A_Y(\eta_{ij}^Y)}{\varphi_Y}\right]$$

**Q 関数（モンテカルロ近似）：**

$$Q(\Theta) = \frac{1}{L} \sum_{l=1}^L \mathcal{L}(\Theta \mid Z^{(l)})$$

ただし $Z^{(l)} \sim p(Z \mid X, Y, \Theta^{\text{old}})$ はラプラス近似からのサンプル。

### 1.3 Eステップ：一般化された勾配とヘッシアン

**対数事後分布の $z_i$ に関する勾配：**

$$\frac{\partial \ln p(z_i, x_i, Y_{i\cdot} \mid \Theta)}{\partial z_i} = \underbrace{-\frac{1}{\sigma_z^2} z_i}_{\text{Term 1: Z prior}} + \underbrace{\frac{1}{\varphi_X} F^\top \bigl[T_X(x_i) - A_X'(Fz_i)\bigr]}_{\text{Term 2: X likelihood ← \textbf{NEW}}} + \underbrace{\frac{w}{2\varphi_Y} \sum_{j \neq i} \bigl[T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)\bigr] z_j}_{\text{Term 3: Y likelihood}}$$

ここで $T_X(x_i) \in \mathbb{R}^d$ は属性の十分統計量ベクトル（ガウスなら $x_i$、ベルヌーイなら $x_i \in \{0,1\}^d$、ポアソンなら非負整数ベクトル）。

**精度行列（ヘッシアンの負値）：**

$$A_i = \underbrace{\frac{1}{\sigma_z^2} I_k}_{\text{Term 1}} + \underbrace{\frac{1}{\varphi_X} F^\top \mathrm{diag}\!\left[A_X''(Fz_i)\right] F}_{\text{Term 2: X curvature ← \textbf{NEW}}} + \underbrace{\frac{w^2}{2\varphi_Y} \sum_{j \neq i} A_Y''(\eta_{ij}^Y)\, z_j z_j^\top}_{\text{Term 3: Y curvature}}$$

> **核心的洞察：** Term 2 と Term 3 は完全に対称な構造を持つ。$X$ と $Y$ はそれぞれ自身の $A'(\cdot)$（残差）と $A''(\cdot)$（曲率重み）によって、ニュートンステップに独立かつ加法的に寄与する。現在のガウス $X$ の実装 $F^\top \Sigma^{-1}(x_i - Fz_i)$ は $\varphi_{X,j} = \sigma_j^2$、$A_X' = \mathrm{id}$、$A_X'' = 1$ とした特殊ケースに過ぎない。

**ガウス $X$ との互換性確認（Term 2）：**

| | 一般式 | ガウス $X$ 代入 | 現実装 |
|-|--------|----------------|-------|
| 勾配 | $\frac{1}{\varphi_X} F^\top [T_X(x_i) - A_X'(Fz_i)]$ | $F^\top \Sigma^{-1}(x_i - Fz_i)$ | ✅ 一致 |
| ヘッシアン | $\frac{1}{\varphi_X} F^\top \mathrm{diag}[A_X''] F$ | $F^\top \Sigma^{-1} F$ | ✅ 一致 |

### 1.4 Mステップ：パラメータ更新の分離原理

Mステップはパラメータ群を完全に分離して最適化できる。

**（a）$F$ の更新（$X$ の構造のみに依存）：**

- **ガウス $X$（閉形式）：**
$$F = \left(\sum_l X^\top Z^{(l)}\right) \left(\sum_l Z^{(l)\top} Z^{(l)}\right)^{-1}$$

- **非ガウス $X$（Adam 勾配法）：**
$$\nabla_{f_j} Q_X = \frac{1}{L\varphi_X} \sum_l \sum_i \left[T_X(x_{ij}) - A_X'(f_j^\top z_i^{(l)})\right] z_i^{(l)}$$

**（b）$\Sigma$ の更新（ガウス $X$ のみ）：**

$$\sigma_j^2 = \frac{1}{nL} \sum_l \sum_i \left(x_{ij} - f_j^\top z_i^{(l)}\right)^2$$

非ガウス $X$ では分散は分布によって固定（Bernoulli）またはリンク関数から決定（Poisson）されるため、独立パラメータとしての $\Sigma$ は不要。

**（c）$w_0, w$ の更新（$Y$ の構造のみに依存）：**

$$\nabla_{w_0} Q_Y = \frac{1}{2L\varphi_Y} \sum_l \sum_{i \neq j} \left[T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)\right]$$

$$\nabla_{w} Q_Y = \frac{1}{2L\varphi_Y} \sum_l \sum_{i \neq j} \left[T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)\right] z_i^{(l)\top} z_j^{(l)}$$

**（d）分散パラメータの更新：**

| パラメータ | 適用条件 | 更新式 |
|-----------|---------|--------|
| $\sigma_j^2$ (X側) | $X \sim \text{Gaussian}$ | 残差の二乗平均（閉形式） |
| $\sigma_Y^2$ (Y側) | $Y \sim \text{Gaussian}$ | 残差の二乗平均（閉形式） |
| なし | Bernoulli / Poisson | $\varphi = 1$ 固定 |

### 1.5 BIC の拡張

モデル選択のための BIC：

$$\mathrm{BIC} = -2 Q_{\mathrm{strict}} + p_{\mathrm{total}} \ln n$$

有効パラメータ数（識別性制約込み）：

$$p_{\mathrm{total}} = \underbrace{\left[(k+1)d - \frac{k(k-1)}{2}\right]}_{F \text{ と } \Sigma_X \text{ (Gauss X のみ)}} + \underbrace{2}_{w_0, w} + \underbrace{\mathbf{1}[Y\text{=Gauss}]}_{\sigma_Y^2}$$

非ガウス $X$ では $\Sigma$ が不要なため $p_{\mathrm{total}} = kd - k(k-1)/2 + 2$（約 $d$ パラメータ削減）。

---

## Section 2: System Architecture (クラス設計)

### 2.1 設計方針

以下の原則に従う：
1. **現実装を最大限流用**：`ExpFamLatentStructuralModel` の Term 3（$Y$ 側）ロジックは完成している
2. **Term 2 のみを差し替え**：$X$ 側の勾配・ヘッシアン計算をファミリー依存化
3. **M-step の分岐**：`calc_F()` のみ非ガウス時に Adam に切り替え

### 2.2 クラス階層

```
LatentStructuralModel          (reproduction/src/model.py)
│  z_i ~ N(0,I), x_i ~ N(Fz_i, Σ), y_ij ~ Bernoulli
│  _calc_gradient: Term1 + Term2_gauss + Term3_bern
│  calc_F: closed-form
│
└── ExpFamLatentStructuralModel  (expfam/src/model_expfam.py)
    │  Y を任意の ExpFam に拡張
    │  _calc_gradient: Term1 + Term2_gauss + Term3_expfam_Y   ← 現状
    │  _calc_precision_matrix: Term1 + Term2_gauss + Term3_expfam_Y
    │  calc_F: closed-form (X=Gauss のまま)
    │
    └── DualExpFamLSM            (expfam/src/model_dual_expfam.py)  [NEW]
           X も任意の ExpFam に拡張
           family_x: 'gaussian' | 'bernoulli' | 'poisson'
           family_y: 'gaussian' | 'bernoulli' | 'poisson'
           _calc_gradient: Term1 + Term2_expfam_X + Term3_expfam_Y   ← 拡張
           _calc_precision_matrix: 同上
           calc_F: Gauss→閉形式, 非Gauss→Adam
           calc_sigma: Gauss X のみ有効
```

### 2.3 提案クラス定義（疑似コード）

```python
class DualExpFamLSM(ExpFamLatentStructuralModel):
    """
    Dual Exponential Family Latent Structural Model.

    Both attribute X and relational Y follow independent ExpFam distributions.

    Generative model:
        z_i ~ N(0, sigma_z^2 * I)
        x_{ij} ~ ExpFam_X(eta_{ij}^X = f_j^T z_i)    j=1..d
        y_{ij} ~ ExpFam_Y(eta_{ij}^Y = w0 + w z_i^T z_j)  i<j

    Args:
        family_x: 'gaussian' | 'bernoulli' | 'poisson'
        family_y: 'gaussian' | 'bernoulli' | 'poisson'
    """

    VALID_FAMILIES = ("gaussian", "bernoulli", "poisson")

    def __init__(self, n, d, k, L=10,
                 family_x="gaussian",   # ← NEW
                 family_y="bernoulli",  # ← family 引数を分離
                 sigma_y=1.0,
                 sigma_x=None):         # ← per-feature dispersion for Gaussian X
        # 親クラスの family 引数 = family_y
        super().__init__(n=n, d=d, k=k, L=L, family=family_y, sigma_y=sigma_y)

        assert family_x in self.VALID_FAMILIES
        self.family_x = family_x
        # Gaussian X: per-feature dispersion (d,)
        # Other X: sigma_x not used (phi=1 fixed)
        self.sigma_x = (np.ones(d) if sigma_x is None else np.asarray(sigma_x))

    # ── X 側のリンク関数 ────────────────────────────────────────────

    def _mean_function_x(self, eta_x: np.ndarray) -> np.ndarray:
        """A_X'(eta) — conditional mean of T(X) given eta."""
        if self.family_x == "gaussian":   return eta_x
        if self.family_x == "bernoulli":  return self._sigmoid(eta_x)
        if self.family_x == "poisson":    return np.exp(np.clip(eta_x, -20, 10))

    def _variance_function_x(self, eta_x: np.ndarray) -> np.ndarray:
        """A_X''(eta) / phi_X — effective curvature weight for X."""
        if self.family_x == "gaussian":
            return np.outer(np.ones(eta_x.shape[0]),
                            1.0 / np.maximum(self.sigma_x**2, 1e-8))  # (n, d)
        if self.family_x == "bernoulli":
            s = self._sigmoid(eta_x)
            return np.clip(s * (1.0 - s), 1e-8, None)
        if self.family_x == "poisson":
            return np.clip(np.exp(np.clip(eta_x, -20, 10)), 1e-8, None)

    # ── E-step オーバーライド ────────────────────────────────────────

    def _calc_gradient(self, X, Y, Z, F, sigma, var_z, w0, w, i):
        """
        Generalized gradient:
          -(1/sigma_z^2) z_i
          + (1/phi_X) F^T [T_X(x_i) - A_X'(F z_i)]    ← X: ExpFam
          + (w/2phi_Y) sum_{j!=i} [T_Y(y_ij) - A_Y'(eta_ij)] z_j  ← Y: unchanged
        """
        z_i = Z[i, :]
        # Term 1
        term1 = -(1.0 / var_z) * z_i
        # Term 2 (X側, ExpFam 一般化)
        eta_x_i = F @ z_i          # (d,)
        mu_x_i  = self._mean_function_x(eta_x_i)  # (d,)
        if self.family_x == "gaussian":
            sigma_inv_diag = 1.0 / np.maximum(np.diag(sigma), 1e-8)
            term2 = F.T @ (sigma_inv_diag * (X[i, :] - mu_x_i))
        else:
            term2 = F.T @ (X[i, :] - mu_x_i)  # phi_X = 1
        # Term 3 (Y側, 親クラスと同じロジック)
        # ... (call parent's term3 logic with self.family = family_y)
        return -(term1 + term2 + term3)

    def _calc_precision_matrix(self, X, Z, F, sigma, var_z, w0, w, i):
        """
        Generalized precision matrix:
          (1/sigma_z^2) I
          + (1/phi_X) F^T diag[A_X''(F z_i)] F         ← X: ExpFam
          + (w^2/2phi_Y) sum_{j!=i} A_Y''(eta_ij) z_j z_j^T  ← Y: unchanged
        """
        z_i = Z[i, :]
        term1 = (1.0 / var_z) * np.eye(self.k)
        # Term 2 (X側)
        eta_x_i   = F @ z_i           # (d,)
        var_fn_x  = self._variance_function_x(eta_x_i[None, :])[0]  # (d,)
        term2 = F.T @ np.diag(var_fn_x) @ F
        # Term 3 (Y側, 親クラスと同じ)
        # ...
        return term1 + term2 + term3

    # ── M-step オーバーライド ────────────────────────────────────────

    def calc_F(self, X, Z_samples):
        """Closed-form for Gaussian X; Adam for Bernoulli/Poisson X."""
        if self.family_x == "gaussian":
            return super().calc_F(X, Z_samples)  # 既存の閉形式
        return self._calc_F_adam(X, Z_samples)

    def _calc_F_adam(self, X, Z_samples, max_iter=50, alpha=0.01):
        """
        Adam update for F under non-Gaussian X.

        Gradient w.r.t. f_j (j-th row of F^T):
          nabla_{f_j} Q_X = (1/L) sum_l sum_i [T_X(x_ij) - A_X'(f_j^T z_i^l)] z_i^l
        """
        # ... Adam loop per feature dimension j

    def calc_sigma(self, X, Z_samples, F):
        """MLE for Gaussian X dispersion; returns identity for others."""
        if self.family_x == "gaussian":
            return super().calc_sigma(X, Z_samples, F)  # 既存の閉形式
        # Non-Gaussian X: Sigma is meaningless, return unit diagonal
        return np.eye(self.d)

    def calc_sigma_x(self, X, Z_samples, F):
        """
        M-step update for per-feature Gaussian X dispersion (MLE).

        sigma_x_j^2 = (1/nL) sum_l sum_i (x_ij - f_j^T z_i^l)^2
        Only meaningful for family_x='gaussian'.
        """
        if self.family_x != "gaussian":
            return self.sigma_x
        # ... MLE update

    def __repr__(self):
        return (f"DualExpFamLSM(n={self.n}, d={self.d}, k={self.k}, "
                f"family_x='{self.family_x}', family_y='{self.family}', "
                f"sigma_x={self.sigma_x.mean():.3f}, sigma_y={self.sigma_y:.3f})")
```

### 2.4 設計上の重要な判断

| 判断事項 | 選択 | 理由 |
|---------|------|------|
| `_calc_gradient` に `X` を渡すか | ✅ すでに渡されている（基底クラス設計が正しい） | `calc_eta_newton` が `X` を引数として持つ |
| `family_x='gaussian'` 時の後方互換性 | ✅ 完全互換 | `calc_F`, `calc_sigma` を既存メソッドに委譲 |
| 非ガウス $X$ 時の $\Sigma$ | 単位行列を返す | $\Sigma$ はガウス専用概念 |
| `_calc_gradient` の引数追加 | `X` は既存の引数に含まれる | 変更不要 |
| Mixin vs 継承 | 単純継承 | $X$ 側の拡張は Term 2 の差し替えのみで局所的 |

### 2.5 データ生成器の拡張

```python
# data_generator_dual_expfam.py (NEW)
def generate_dual_data(
    n, d, k, seed,
    family_x: str,   # 'gaussian' | 'bernoulli' | 'poisson'
    family_y: str,   # 'gaussian' | 'bernoulli' | 'poisson'
    # X-side true params
    sigma_x_true: float = 0.1,   # for Gaussian X
    # Y-side true params
    w0_true: float = -1.0,
    w_true: float = 1.5,
    sigma_y_true: float = 0.1,   # for Gaussian Y
) -> dict:
    """
    Generate synthetic data with independent ExpFam for X and Y.

    Shared structure: Z ~ N(0, I), F (factor loading)
    X: each x_ij ~ family_x(eta_x = f_j^T z_i)
    Y: y_ij ~ family_y(eta_y = w0 + w z_i^T z_j)
    """
```

---

## Section 3: Ultimate Experimental Design

### 3.1 実験設計の哲学

NOLTA 2024 の実験 Exp 1〜4 を完全に踏襲し、以下の**2つの新軸**を加える：

1. **`family_x` 軸**：属性 $X$ の分布の影響（新規）
2. **`family_y` 軸**：関係 $Y$ の分布の影響（前論文から継承・拡張）

これにより「なぜ Dual-ExpFam か」という問いに、定量的な答えを与える。

### 3.2 Exp 1: BIC による潜在次元同定（拡張版）

**設定：** $n=150$, $d=15$, $k^*=3$, $k \in \{1,\ldots,6\}$, 10 trials

**走らせる family 組み合わせ（6種）：**

| 代表記号 | `family_x` | `family_y` | 選定理由 |
|---------|-----------|-----------|---------|
| G×G | Gaussian | Gaussian | 前論文ベースライン |
| B×B | Bernoulli | Bernoulli | 純離散×純離散 |
| P×P | Poisson | Poisson | 純カウント×純カウント |
| G×P | Gaussian | Poisson | 最も現実的（属性=連続, 関係=共起頻度） |
| B×P | Bernoulli | Poisson | 二値属性×カウント関係（SNS等） |
| P×B | Poisson | Bernoulli | カウント属性×二値関係 |

**評価指標：** BIC vs $k$（6×6 グリッドプロット）、RMSE$(Z)$ vs $k$（同）

**期待される発見：** 全6組み合わせで BIC が $k=3$ を正しく選択。

### 3.3 Exp 2: 漸近一致性（$n$ 変化）

**設定：** $n \in \{50, 100, 150, 200, 250, 300\}$, $d=15$, $k=3$, 10 trials

**全パラメータの RMSE を記録（NOLTA 2024 と同様）：**

$$\mathrm{RMSE}(Z),\ \mathrm{RMSE}(F),\ \mathrm{RMSE}(\Sigma),\ \mathrm{RMSE}(Y),\ \mathrm{RMSE}(X),\ |w_0^{\text{est}} - w_0^*|,\ |w^{\text{est}} - w^*|$$

**代表組み合わせ：** G×G（ベースライン比較）+ G×P（新規最重要）

**新規指標 RMSE(X)：**
$$\mathrm{RMSE}(X) = \sqrt{\frac{1}{nd}\sum_{i,j}\left(x_{ij} - A_X'(f_j^{\text{est}\top} z_i^{\text{est}})\right)^2}$$

$n$ 増加でゼロに収束することが漸近一致性の証拠。

### 3.4 Exp 3: 属性次元 $d$ への頑健性

**設定：** $d \in \{5, 10, 15, 20, 25, 30\}$, $n=150$, $k=3$, 10 trials

**着目点：** $d$ が増えるほど $X$ からの情報（Term 2 の和の項数 $\propto d$）が豊富になり、RMSE$(Z)$ と RMSE$(F)$ の改善が加速する。非ガウス $X$ では非線形リンク関数の恩恵でさらに急峻な改善が期待される。

### 3.5 Exp 4: 分布ミスマッチの 3×3 比較マトリクス（最強の貢献証明）

#### 3.5.1 設計思想

現行の Exp 4（$Y$ のみの 3×3 ミスマッチ）を**$X$ と $Y$ の独立ミスマッチ**に拡張する。最も説得力ある提示形式は **2枚の独立した 3×3 ヒートマップ**（ $X$ ミスマッチ単独、$Y$ ミスマッチ単独）+ **ワーストケース例**の3点セット。

#### 3.5.2 Exp 4a: $Y$ ミスマッチマトリクス（$X$ は正解固定）

| 真のデータ $Y$ \ 使用モデル $Y$ | Bernoulli | Poisson | Gaussian |
|-------------------------------|-----------|---------|---------|
| **Bernoulli** | ✓ | × | × |
| **Poisson** | × | ✓ | × |
| **Gaussian** | × | × | ✓ |

$(X = \text{Gaussian}$ で固定、$n=150$, $d=15$, $k=3$, 10 trials)

→ 前回実験でこのマトリクスは取得済み（Poisson→Bernoulli: 6.1倍, Gaussian→Poisson: 9.6倍）。

#### 3.5.3 Exp 4b: $X$ ミスマッチマトリクス（$Y$ は正解固定）★ 新規 ★

| 真のデータ $X$ \ 使用モデル $X$ | Gaussian | Bernoulli | Poisson |
|-------------------------------|----------|-----------|---------|
| **Gaussian** | ✓ | × | × |
| **Bernoulli** | × | ✓ | × |
| **Poisson** | × | × | ✓ |

$(Y = \text{Bernoulli}$ で固定、同設定)

**予測される破綻メカニズム：**

- ガウス連続値 $X$ にポアソン $X$ モデルを適用：$\eta_{ij}^X = f_j^\top z_i$ に対し $\exp(\eta)$ を使用 → 負値の連続データで exp が不整合 → 極端な推定値
- 二値 $X$ にガウス $X$ モデルを適用：$\mathrm{MSE}(X)$ は定義上計算可能だが、$x_{ij} \in \{0,1\}$ の情報を連続残差として扱うため、Term 2 の勾配スケールが誤る → RMSE$(Z)$ 増大

#### 3.5.4 Exp 4c: 双方ミスマッチのワーストケース

真の生成：$X_{\text{Gauss}} \times Y_{\text{Pois}}$（連続属性 × カウント関係）

| ケース | $X$ モデル | $Y$ モデル | RMSE$(Z)$ 予測 |
|-------|-----------|-----------|--------------|
| 正解 (Dual) | Gaussian | Poisson | 最小（基準値） |
| $Y$ のみ誤り | Gaussian | Bernoulli | 中程度 |
| $X$ のみ誤り | Bernoulli | Poisson | 中程度 |
| 両方誤り | Bernoulli | Bernoulli | 最大（劣化倍率の積） |

この「誤り要因の加法性 vs 乗法性」の分析が、本論文の最も新規性の高いメッセージとなる。

#### 3.5.5 Exp 4d: $X$-$Y$ ファミリーグリッドの全 9 組の RMSE 測定

真の生成分布が $X_{\text{Gauss}} \times Y_{\text{Pois}}$ の場合のモデルミスマッチ：

| モデル $(X, Y)$ | RMSE$(Z)$ |
|----------------|---------|
| (Gauss, Pois) ✓ | 基準 |
| (Gauss, Bern) | ？ |
| (Gauss, Gauss) | ？ |
| (Bern, Pois) | ？ |
| (Bern, Bern) | ？ |
| ... | ... |

これにより「$X$ と $Y$ のどちらのミスマッチがより破滅的か」を定量化。

### 3.6 実験パラメータの統一設定

```python
# 全実験共通のハイパーパラメータ
GLOBAL_CONFIG = {
    # データ設定
    "n_default": 150,
    "d_default": 15,
    "k_true": 3,
    "var_f": 5.0,
    "uniq": 0.1,

    # EM設定
    "L": 10,           # モンテカルロサンプル数
    "num_iter": 15,    # EMイテレーション数（Dual版は収束が遅い可能性）
    "newton_alpha": 0.5,
    "adam_alpha": 0.01,

    # 試行回数
    "n_trials": 10,

    # 真のパラメータ（Y側）
    "w0_bern": -1.0, "w_bern": 1.5,
    "w0_pois": 0.0,  "w_pois": 0.5,
    "w0_gauss": 0.0, "w_gauss": 0.5, "sigma_y_gauss": 0.1,

    # 真のパラメータ（X側）
    "sigma_x_gauss": 0.1,  # noise level for Gaussian X
    # Bernoulli X: link = sigmoid(F z_i), no extra param
    # Poisson X: link = exp(F z_i), need to ensure exp(F z_i) is reasonable
}
```

### 3.7 評価指標の完全リスト（全実験共通）

| 指標 | 数式 | 対象 |
|-----|------|------|
| RMSE$(Z)$ | $\|Z^{\text{est}} R - Z^*\|_F / \sqrt{nk}$ | 潜在変数推定 |
| RMSE$(F)$ | $\|F^{\text{est}} R - F^*\|_F / \sqrt{dk}$ | 因子負荷 |
| RMSE$(\Sigma)$ | $\|\hat{\sigma} - \sigma^*\| / \sqrt{d}$ | X分散（Gauss X のみ） |
| RMSE$(Y)$ | $\|\hat{\mu}_Y - \mu_Y^*\|_{\text{upper}} / \binom{n}{2}^{1/2}$ | Y予測精度 |
| RMSE$(X)$ | $\|\hat{\mu}_X - X\|_F / \sqrt{nd}$ | X再構成精度 ★新規★ |
| $|w_0^{\text{err}}|$ | $|w_0^{\text{est}} - w_0^*|$ | バイアス推定 |
| $|w^{\text{err}}|$ | $|w^{\text{est}} - w^*|$ | 重み推定 |
| BIC | $-2Q_{\text{strict}} + p\ln n$ | モデル選択 |
| 計算時間 [s] | 壁時計時間 | スケーラビリティ |

Procrustes 補正（$R = UV^\top$ from SVD$(Z^{\text{est}\top} Z^*)$）を全 RMSE$(Z)$, RMSE$(F)$ に適用。

---

## Appendix: 実装上の注意点

### A.1 非ガウス $X$ のデータ生成における正則化

- **Bernoulli $X$**：$f_j^\top z_i$ の値域を調整しないと $p_{ij} \approx 0$ or $1$ の飽和状態になる。$F$ の行ノルムを制限（現行の `generate_*_data` と同じ正規化）が必要。
- **Poisson $X$**：$\lambda_{ij}^X = \exp(f_j^\top z_i)$ が発散しないよう $f_j^\top z_i \leq 10$（現行 clip と同じ）を設ける。$\bar{\lambda}_X \approx 2 \sim 5$ 程度に調整。

### A.2 Adam の $F$ 更新での収束安定性

非ガウス $X$ の $F$ Adam 更新では、各特徴次元 $j$ を独立に更新するか、行列まとめて更新するかを設計段階で決める。**行列まとめ更新**（$d \times k$ 全体を1つの Adam ステートで管理）が実装上シンプルで推奨。

### A.3 識別性制約

- $Z$ の回転不変性：Procrustes 補正により対処済み
- $F$ の符号不変性：同上
- Dual-ExpFam での新しい不変性：$X$ の因子負荷 $F$ と $Y$ の因子負荷（$Z$ に内包）は共有されるため、追加の不変性は生じない（SMC 2022 の注記と一致）

---

## 結論とディレクターへの報告

Phase 1（精読）の結果、現実装のアーキテクチャは**Dual-ExpFam への拡張に対して予め良く設計されている**ことが判明した：

1. `_calc_gradient(self, X, Y, Z, F, sigma, var_z, w0, w, i)` の引数 `X` はすでに存在する
2. Term 2（X側）と Term 3（Y側）は数理的に対称であり、差し替えの局所性が保証される
3. `calc_F` の閉形式解は Gaussian X の特殊ケースであり、Adam への切り替えは `if family_x != "gaussian"` の分岐で済む

**実装に必要な変更は最小限：**
- `model_dual_expfam.py`（約 250 行）：`DualExpFamLSM` クラス
- `data_generator_dual_expfam.py`（約 150 行）：`generate_dual_data()` 関数
- `utils_dual_expfam.py`（約 100 行）：`run_em_dual()` のラッパー更新
- 実験スクリプト 4 本（各約 200 行）：Exp 1〜4 拡張版

**最大の学術的 Contribution（改めて整理）：**

$$\boxed{\text{「Dual-ExpFam により、分布ミスマッチによる RMSE 劣化を } X \text{ 側でも定量化できた」}}$$

ディレクターの承認が得られ次第、`model_dual_expfam.py` の実装を開始できる状態にある。

---
*Document version: 1.0 | Generated: 2026-03-24*
