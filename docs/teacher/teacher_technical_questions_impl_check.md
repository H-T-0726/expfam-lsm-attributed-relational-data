# 先生技術的コメント 実装照合チェックシート

確認日：2026-05-08  
確認対象：expfam/src/model_dual_expfam.py, model_expfam.py, utils_expfam.py, reproduction/src/model.py

---

## Q1. 指数型分布族の式はスカラーでOKか

**先生コメント：**「これは全てスカラーでOK? このあとはベクトルとして表現している気もするので，検討が必要です」

### 結論：スカラー表記でOK。ただし補足文が1文必要。

### 実装上の扱い

**関係データ y_ij：**  
`_calc_gradient` (`model_dual_expfam.py` L.153):
```python
eta_y = w0 + w * (Z @ z_i)   # (n,) — j番目の成分 eta_y[j] がスカラー η_ij^Y
```
`y_ij` は (i,j) ペアごとの1スカラー観測値として扱われている。✓

**属性データ x_il：**  
`_calc_gradient` (`model_dual_expfam.py` L.140-141):
```python
eta_x_i = F @ z_i            # (d,) — l番目の成分 eta_x_i[l] が η_il^X
mu_x_i  = self._mean_function_x(eta_x_i)  # element-wise A_X'(η)
```
`eta_x_i` は (d,) ベクトルで、要素ごとに指数型分布族のリンク関数を適用している。  
つまり「各属性次元 l ごとにスカラー η_il^X に対して1変量の指数型分布族を適用」している。

`_variance_function_x` (`model_dual_expfam.py` L.104-117):
```python
# bernoulli: s = sigmoid(eta_x)  → element-wise
# poisson:   exp(eta_x)          → element-wise
# gaussian:  ones_like(eta_x)    → element-wise
```
すべて element-wise（成分ごと）の計算であり，スカラー kernel の (d,) ベクトル化版。

### 先行研究[1]との関係

先行研究[1]（Gaussian X のみ）:  
`x_i ~ N(F z_i, Σ)` と**ベクトル**で書かれている（`reproduction/src/model.py` Docstring L.23）。  
しかし Σ は対角行列（`calc_sigma` L.582: `np.diag(np.diag(sigma))`）なので，実質的には：

$$x_{il} \mid \mathbf{z}_i \sim \mathcal{N}(\mathbf{f}_l^\top \mathbf{z}_i, \sigma_l^2) \quad \text{独立}$$

すなわち「各次元独立なガウス分布の積」と等価。提案手法はこれを各次元の観測分布一般化に拡張している。

### 原稿での推奨表現

2.3節に以下の1文を追加すれば実装と一致する：

> 「本稿では各観測値を1変量の指数型分布族として扱う．属性情報 $x_{il}$ については各属性次元 $l$ ごとに独立に適用する．」

または 3.1節の生成モデルに：

> 「$\mathrm{ExpFam}_X$ は各属性次元に独立に適用し，$x_{il} \mid \mathbf{z}_i \sim \mathrm{ExpFam}_X(\mathbf{f}_l^\top \mathbf{z}_i)$ とする．」

### 先生に説明するなら

「ご指摘の通りです。2.3節の式は1変量の指数型分布族の定義式であり，y_ij はスカラー，x_il は属性次元ごとのスカラーとしてこれを適用しています。先行研究[1]の Gaussian X も対角共分散を仮定しているため，各次元独立なガウス分布の積として解釈でき，同じ"次元ごとに適用"という形になります。原稿に "各属性次元ごとに適用する" という補足を1文加えます。」

---

## Q2. x の成分を個別に指定しているのか

**先生コメント：**「ここもなのだけど，xの成分を個別に指定するの？従来は x（ベクトル）に対して確率分布を指定していたと思うのだけど…（時間がないので今回はこのままでもいいけど，疑義がつきそう）」

### 結論：実装は各次元独立に指数型分布族を適用。従来手法の対角Gaussian も各次元独立なので矛盾しない。先生への説明は可能。

### 実装上の扱い

**Gaussian-X（先行研究[1]との比較）：**  
提案実装 (`model_dual_expfam.py` L.186-190):
```python
if self.family_x == "gaussian":
    sigma_inv_diag = 1.0 / np.maximum(np.diag(sigma), 1e-8)
    term2 = F.T @ (F * sigma_inv_diag[:, None])   # = F^T Sigma^{-1} F
```
これは `F^T diag(1/σ_l^2) F` であり，先行研究[1]の `F^T Σ^{-1} F` と完全に同一。

先行研究[1]の Sigma は対角行列（`reproduction/src/model.py` L.582: `sigma = np.diag(np.diag(sigma))`）なので，  
`x_i ~ N(F z_i, Σ)` with diagonal Σ ＝ `x_il | z_i ~ N(f_l^T z_i, σ_l^2)` each independently。

**Bernoulli-X / Poisson-X：**  
提案実装 (`model_dual_expfam.py` L.191-194):
```python
else:
    var_x_i = self._variance_function_x(eta_x_i)   # A_X''(η_il), shape (d,)
    term2 = F.T @ (F * var_x_i[:, None])            # = F^T diag(A_X''(F m_i)) F
```
`eta_x_i = F @ z_i` (shape (d,)) に対して，`_variance_function_x` が element-wise に適用される。  
これは「各属性次元 l ごとに独立な指数型分布族 x_il ~ ExpFam_X(f_l^T z_i)」の Hessian と等価。

M-step の F 勾配 (`_calc_F_adam` L.249-252):
```python
eta_x  = Z_l @ F.T                    # (n, d)
mu_x   = self._mean_function_x(eta_x) # (n, d), element-wise
residual = X - mu_x                   # T_X(x) - A_X'(eta), element-wise
grad  += residual.T @ Z_l             # (d, k)
```
残差も element-wise → 各次元独立な尤度の勾配そのもの。

### 従来手法との関係

| 観点 | 先行研究[1] Gaussian-X | 提案手法 Bernoulli/Poisson-X |
|------|----------------------|---------------------------|
| モデル記法 | $\mathbf{x}_i \sim \mathcal{N}(\mathbf{F}\mathbf{z}_i, \boldsymbol{\Sigma})$ | $x_{il} \sim \mathrm{ExpFam}_X(\mathbf{f}_l^\top \mathbf{z}_i)$ |
| Σの構造 | 対角行列（成分ごと独立）| 分散関数 $A_X''(\eta_l)$（成分ごと） |
| 等価な成分表示 | $x_{il} \sim \mathcal{N}(\mathbf{f}_l^\top \mathbf{z}_i, \sigma_l^2)$ | $x_{il} \sim \mathrm{ExpFam}_X(\mathbf{f}_l^\top \mathbf{z}_i)$ |
| 成分間の独立性 | ✓（対角Σ） | ✓（設計上独立） |

従来手法も対角共分散を使っているため，実質的に「各次元ごとに分布を指定」している。提案手法はその"分布の種類"をベルヌーイ・ポアソンに一般化したもの。

### 原稿での推奨表現

3.1節の生成モデルで以下のように明示する：

> 「属性情報については，各属性次元 $l$ が独立に $x_{il} \mid \mathbf{z}_i \sim \mathrm{ExpFam}_X(\eta_{il}^X)$ に従うとする．先行研究[1]が仮定するガウス分布（対角共分散）も，これを各次元の独立なガウス分布として解釈できる特殊ケースである．」

### 先生に説明するなら

「ご指摘ありがとうございます。先行研究[1]は $\mathbf{x}_i \sim \mathcal{N}(\mathbf{F}\mathbf{z}_i, \boldsymbol{\Sigma})$ と書いていますが，実装では Σ を対角行列として推定しています（reproduction/src/model.py）。対角Σのガウス分布は各次元独立なガウスの積と等価なため，実質的に "次元ごとに分布を指定" しています。提案手法はこの "各次元独立" という構造を保ちつつ，分布の種類をベルヌーイ・ポアソンに拡張しています。原稿で "先行研究[1]と同様に成分間の独立性を仮定し，各次元の分布をExpFamに一般化" と書くことで疑義を減らせます。」

---

## Q3. j≠i なら 1/2 は不要では？

**先生コメント：**「j≠iとしているのであれば分母の2はいらないのではないかと思うんだけど．どうだろう？」

### 結論：A — 実装・先行研究[1]が (1/2)Σ_{i≠j} の慣行を採用しているため，1/2 は必要かつ一貫している。

### 実装上の根拠

**[根拠1] Y 対数尤度の定義（最重要）**

`calc_log_likelihood_Y` (`model_expfam.py` L.266-268):
```python
np.fill_diagonal(ln_p, 0.0)
ll += 0.5 * np.sum(ln_p)      # ← (1/2)Σ_{i≠j} の慣行
```
`ln_p` は n×n の全ペア行列（対角=0）。`0.5 * sum` = `(1/2)Σ_{i≠j}`。  
**Y 対数尤度を (1/2)Σ_{i≠j} と定義しているため，そのヘッセ行列にも (1/2) が付く。**

**[根拠2] 精度行列の Term3**

提案実装 (`model_dual_expfam.py` L.200):
```python
term3 = 0.5 * (w ** 2) * (Z.T @ np.diag(var_y) @ Z)
```
先行研究[1]再現実装 (`reproduction/src/model.py` L.353):
```python
term3 = 0.5 * (w ** 2) * (Z.T @ np.diag(s_deriv) @ Z)
```
先行研究[1]のコメント（L.306-307）:
```
Based on paper Eq. (22):
A_i = (1/σ²_z)I + F^T Σ^{-1} F + (1/2) Σ_{j≠i} s_ij(1-s_ij) w² z_j z_j^T
```
**先行研究[1]の論文 Eq.(22) に 1/2 が明記されている。**

**[根拠3] M-step 勾配の除算**

先行研究[1]再現実装 (`reproduction/src/model.py` L.663):
```python
grad = -grad_sum / (2.0 * L)
```
コメント（L.607）: "The factor 1/2 comes from the symmetry of Y (counting i<j only once)."

提案実装 (`model_expfam.py` L.168):
```python
grad = -grad_sum / (2.0 * L * phi)
```
`grad_sum = Σ_l Σ_{i≠j} [T_Y(y_ij) - A_Y'(η_ij^Y)]`（全ペア対称和）  
`/ (2*L)` で正しい `Σ_{i<j} / L` と等価にする。

### Y 対数尤度の二つの記法の対応

数学的に等価な2つの記法：

| 記法 | Y 対数尤度 | 精度行列 Term3 | M-step 勾配 |
|------|-----------|---------------|------------|
| **慣行 A（実装で採用）** | $\frac{1}{2}\sum_{i \neq j}\ln p_Y(y_{ij})$ | $\frac{(w^Y)^2}{2}\sum_{j\neq i} A_Y''(\eta_{ij}^Y)\mathbf{z}_j\mathbf{z}_j^\top$ | $\frac{1}{2L}\sum_{i\neq j}[\cdots]$ |
| **慣行 B（先生が示す形）** | $\sum_{i < j}\ln p_Y(y_{ij})$ | $(w^Y)^2\sum_{j\neq i} A_Y''(\eta_{ij}^Y)\mathbf{z}_j\mathbf{z}_j^\top$ | $\frac{1}{L}\sum_{i<j}[\cdots]$ |

**両慣行は数学的に等価。**

現在の実装は慣行 A。この場合 Hessian に 1/2 が必要。

慣行 B を採用するには：
- 精度行列 Term3 から 1/2 を除く
- M-step 勾配の `/ (2L)` を `/ L` に変更し，和を `Σ_{i<j}` に変更
- `calc_log_likelihood_Y` の `0.5 * np.sum(ln_p)` を upper_mask で計算し直す

どちらも正しいが，**先行研究[1] Eq.(22)(23) が慣行 A を採用しているため，整合性のため慣行 A を維持することを強く推奨する。**

### 原稿の式をどう書くべきか

**現状維持（推奨）：** `(w^Y)^2/2 * Σ_{j≠i} A_Y''(η_ij^Y) z_j z_j^T` と書き，
本文に「先行研究[1]と同一の慣行（Y 側の対数尤度を $(1/2)\sum_{i\neq j}$ で計算）に従う」と一文加える。

**代替案（1/2 を消す場合）：** `(w^Y)^2 * Σ_{j>i} A_Y''(η_ij^Y) z_j z_j^T` と書く（不等号の向きを変える）。ただしこの場合は M-step の勾配式も `Σ_{i<j}/L` に変更が必要で，先行研究[1]との対応記載が複雑になる。

### 先生に説明するなら

「ご指摘ありがとうございます。先行研究[1]（式(22)）が $\frac{1}{2}\sum_{j\neq i}$ の和として精度行列を定義しており，これは Y 側の対数尤度を $\frac{1}{2}\sum_{i\neq j}\ln p_Y(y_{ij})$（対称行列を二重計上して 1/2 で割る）として実装しているためです（model_expfam.py L.267）。もし $\sum_{i<j}$ 表記に変えるなら 1/2 は不要ですが，M ステップの勾配式との記法統一も必要になります。現原稿は先行研究[1]と同一の慣行に従っているため，1/2 があることを明記する方向で進めます。」

---

## Q4. Σ はパラメータとしてあるのか

**先生コメント：**「Σってパラメータとしてある？」

### 結論：Gaussian-X の場合のみ推定される条件付きパラメータ。Bernoulli/Poisson-X では Σ は推定されず，単位行列を返すのみ。θ に常に含めるのは不正確。

### 実装上の扱い

**`calc_sigma` (`model_dual_expfam.py` L.274-285):**
```python
def calc_sigma(self, X, Z_samples, F):
    if self.family_x == "gaussian":
        return super().calc_sigma(X, Z_samples, F)   # 解析解で推定
    return np.eye(self.d)                             # identity を返すだけ（推定しない）
```

**`calc_bic_dual` (`utils_expfam.py` L.398-403):**
```python
f_params  = k * d - k * (k - 1) // 2
sigma_x_p = d if family_x == "gaussian" else 0      # Gaussian-X のみ d パラメータ
sigma_y_p = 1 if family_y == "gaussian" else 0
num_params = f_params + sigma_x_p + sigma_y_p
```
BIC の計算においても，Σ のパラメータ数は Gaussian-X のみカウント。

**精度行列における Σ の使用 (`model_dual_expfam.py` L.186-194):**
```python
if self.family_x == "gaussian":
    sigma_inv_diag = 1.0 / np.maximum(np.diag(sigma), 1e-8)
    term2 = F.T @ (F * sigma_inv_diag[:, None])   # F^T Sigma^{-1} F を使う
else:
    var_x_i = self._variance_function_x(eta_x_i)  # A_X''(η) を使う（Σ 不要）
    term2 = F.T @ (F * var_x_i[:, None])
```

### 分布族ごとの扱い

| ExpFam_X | Σ の推定 | 精度行列 Term2 | BIC への寄与 |
|---------|---------|--------------|------------|
| ガウス分布 | ✓ 推定（解析解）| $\mathbf{F}^\top \boldsymbol{\Sigma}^{-1} \mathbf{F}$ | d パラメータ |
| ベルヌーイ | ✗ 不使用（identity を返す）| $\mathbf{F}^\top \mathrm{diag}(A_X''(\mathbf{F}\mathbf{m}_i)) \mathbf{F}$ | 0 |
| ポアソン | ✗ 不使用（identity を返す）| $\mathbf{F}^\top \mathrm{diag}(A_X''(\mathbf{F}\mathbf{m}_i)) \mathbf{F}$ | 0 |

注意: `params["sigma"]` は全 family_x に対してメモリ上に存在するが，非 Gaussian では M-step で更新されず identity のまま。推定されるパラメータとは言えない。

### 原稿での推奨表現

**現在の原稿 `θ = {F, Σ, w_0^Y, w^Y}` という記法は Gaussian-X 専用であり，不正確。**

2通りの修正案：

**案1（推奨）：θを条件付きで定義**
> パラメータ集合 $\boldsymbol{\theta} = \{\mathbf{F},\ w_0^Y,\ w^Y\}$ とする（$\mathrm{ExpFam}_X$ がガウス分布のとき，属性データの対角分散行列 $\boldsymbol{\Sigma}$ も含む）．

**案2（簡略化）：論文で主に扱うシナリオに合わせる**
Scenario A のように非 Gaussian-X を主な提案として強調するなら，Σ は括弧書き補足として扱う。

**V_X の式における Σ の記述は問題なし：**
> $\mathbf{V}_X(\mathbf{m}_i) = \boldsymbol{\Sigma}^{-1}$（ガウス分布）または $\mathrm{diag}(A_X''(\mathbf{F}\mathbf{m}_i))$（ベルヌーイ・ポアソン）

この書き方では Σ が "Gaussian-X の場合のみ使用" という文脈で正確に使われている。

### 先生に説明するなら

「Σ は属性データにガウス分布を使う場合（Gaussian-X）のみ推定される対角分散行列です（model_dual_expfam.py: calc_sigma が Gaussian-X のときのみ解析解で更新）。ベルヌーイ・ポアソンでは Σ は存在せず，代わりに各指数型分布族の分散関数 $A_X''(\eta)$ を精度行列の計算に使います。したがって θ の定義を『$\mathrm{ExpFam}_X$ がガウスのとき Σ を含む』と条件付きに修正します。」

---

## 確認できなかったこと

- `jima_abst_稗田天人ver3_m.docx`：.docx形式のため直接読み込み不可。Word上の式と上記の実装照合結果を手動で対照することを推奨。
- 先行研究 PDF（`paper/A_study_on_latent_structural_models_for_binary_rel.pdf`）：PDFレンダリングツール不可のため直接確認不可。ただし再現実装コード（`reproduction/src/model.py`）がEq.(22)/(23)の内容を明示的にコメントで記載しており，間接的に確認済み。
