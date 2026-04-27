# Gemini へのプロンプト（新規実験の方向性を相談）

---

## 背景・完了済みの作業

Mikawa et al. (NOLTA 2024) の論文「A study on latent structural models for binary relational data」を Python で完全再現しました。

**モデルの概要:**
- 観測データ: ノード属性 `X ∈ ℝ^{n×d}`、関係データ `Y ∈ {0,1}^{n×n}`（二値行列）
- 生成モデル:
  ```
  z_i  ~ N(0, I_k)                             [潜在変数]
  x_i  ~ N(F z_i, Σ)                           [属性の生成]
  y_ij ~ Bernoulli(σ(w₀ + w · z_i^T z_j))     [関係の生成]
  ```
- 推定: モンテカルロ EM アルゴリズム（E-step: ラプラス近似 + ニュートン法、M-step: Adam）

**再現実験の結果（論文 Tables II–IV と比較済み）:**
- Experiment 1: 因子数 k の選択（BIC） → 真の k=3 を正しく選択
- Experiment 2: サンプル数 n / 次元 d の変化に対する RMSE
- Experiment 3: ワインデータへの実データ適用

---

## 次に取り組みたい新規実験

**目標:** 関係データ Y の分布を Bernoulli から**指数型分布族全般**に拡張する。

### 数理的定式化

指数型分布族の標準形:
```
p(y; η) = h(y) · exp( η·T(y) − A(η) )
```

拡張された生成モデル:
```
y_ij ~ ExpFam( η_ij = w₀ + w · z_i^T z_j )
```

**E-step（ニュートン法の勾配・精度行列）は分布によらず統一形式:**
```
勾配:   ∂/∂z_i = −(1/σ²_z)z_i + F^TΣ^{-1}(x_i−Fz_i) + (1/2)Σ_{j≠i} [T(y_ij)−A'(η_ij)] · w · z_j

精度行列: A_i = (1/σ²_z)I + F^TΣ^{-1}F + (1/2)Σ_{j≠i} A''(η_ij) · w² · z_j z_j^T
```

| 分布 | T(y) | A'(η)（平均） | A''(η)（分散） | 適用例 |
|------|------|--------------|----------------|--------|
| Bernoulli | y | σ(η) | σ(η)(1−σ(η)) | 二値グラフ（既存） |
| Poisson | y | exp(η) | exp(η) | 購買回数・共著数 |
| Gamma | y | exp(η)* | exp(2η)/α* | 取引量・相関強度 |
| Neg. Binomial | y | r·eη/(1−eη) | r·eη/(1−eη)² | スパースカウント |

*log-link 使用時

**M-step は既存の Adam をそのまま流用**（残差 `y−σ(η)` を `T(y)−A'(η)` に変えるだけ）。

### 想定する実装構造
```
src/
├── model.py                 # 既存（Bernoulli、変更なし）
├── model_expfam.py          # 新規: 指数型分布族基底クラス
│   ├── _sufficient_stat(y)  # T(y)
│   ├── _log_partition(eta)  # A(η)
│   ├── _mean_function(eta)  # A'(η)
│   └── _variance_function(eta) # A''(η)
└── distributions/
    ├── poisson.py
    └── negative_binomial.py
```

---

## Gemini へのお願い

以下の点について、アドバイスと実装方針をください:

### Q1: 数式の妥当性確認
上記の E-step 勾配・精度行列の一般化式は数学的に正しいか？
特に以下を確認してほしい:
- `(1/2)` 係数の扱い（i<j の対称性から来る）
- Poisson の場合の `η_ij = w₀ + w·z_i^T z_j` が `z_i` についての微分で `w·z_j` になる理由
- `A''(η_ij) · w² · z_j z_j^T` の正定値性の保証

### Q2: 最初の PoC（概念実証）として何を実装すべきか
- 合成データでの Poisson モデル（n=150, d=15, k=3）を推奨しているが、他に良いアプローチはあるか？
- Bernoulli モデルとの比較実験の設計はどうすべきか？

### Q3: 数値安定性の懸念点
Poisson では `exp(η_ij)` が発散しうる。以下の対策で十分か？
```python
eta_ij = np.clip(eta_ij, -20, 20)   # λ in [2e-9, 4.85e8]
var_fn = np.minimum(exp(eta_ij), 1e6)
```

### Q4: 論文としての新規性・貢献のアピール方法
- この拡張が KDD/NeurIPS レベルの貢献として成立するために、何が必要か？
- 比較すべき既存手法（ベースライン）として何が適切か？
- 実データは Amazon Co-Purchase、DBLP Co-authorship、Gene Co-expression を想定しているが、より適切なデータセットはあるか？

---

## 添付する参考ファイル

（以下のコードを参照してください）

### 既存モデルのコア実装（`model.py` の E-step 抜粋）

```python
def calc_eta_newton(self, X, Y, F, sigma, w0, w, num_iter=30, alpha=0.5):
    """E-step: ニュートン法で事後分布の最頻値 η_i を求める"""
    n, d = X.shape
    k = F.shape[1]
    eta = np.zeros((n, k))

    for _ in range(num_iter):
        grad = np.zeros((n, k))
        hess = np.zeros((n, k, k))

        # 属性データ項
        FtSigInv = F.T @ sigma_inv
        grad += (X @ sigma_inv @ F) - (eta @ FtSigInv.T @ FtSigInv)

        # 関係データ項（Bernoulli）
        eta_dot = eta @ eta.T          # z_i^T z_j の行列
        s = sigmoid(w0 + w * eta_dot)  # A'(η) = σ(η)
        residual = Y - s               # T(y) - A'(η)

        grad += (1/2) * (residual + residual.T) @ eta * w

        # 精度行列: A''(η) · w² · z_j z_j^T の和
        var_fn = s * (1 - s)           # A''(η) = σ(η)(1-σ(η))
        for i in range(n):
            A_i = (1/sigma_z2) * np.eye(k) + FtSigInv @ F
            for j in range(n):
                if i != j:
                    A_i += (1/2) * var_fn[i,j] * w**2 * np.outer(eta[j], eta[j])
            # ニュートン更新: η_i ← η_i + α · A_i^{-1} · grad_i
            eta[i] += alpha * np.linalg.solve(A_i, grad[i])

    return eta
```

このコードの `s = sigmoid(...)` と `var_fn = s * (1 - s)` を、他の分布の `A'(η)` と `A''(η)` に置き換えるだけで全分布に対応できる設計を確認してほしい。
