# パラメータ推定 正しい式集（3.2 節用）

確認日：2026-05-08  
実装照合：model_dual_expfam.py + utils_expfam.py + reproduction/src/model.py

---

## 1. 生成モデル（完全データ尤度）

### 1.1 潜在変数の事前分布

$$\mathbf{z}_i \sim \mathcal{N}(\mathbf{0},\ \sigma_z^2 \mathbf{I}_k), \quad \sigma_z^2 = 1 \quad (i = 1,\ldots,n)$$

**実装根拠：** `var_z = 1.0`（固定、推定しない。`initialize_params` L.99）

---

### 1.2 関係データの生成モデル（提案: Y 側）

$$y_{ij} \mid \mathbf{z}_i, \mathbf{z}_j, w_0^Y, w^Y \sim \mathrm{ExpFam}_Y\!\left(\eta_{ij}^Y\right), \quad
\eta_{ij}^Y = w_0^Y + w^Y \mathbf{z}_i^\top \mathbf{z}_j \quad (i < j) \tag{A1}$$

**実装根拠：** `eta_y = w0 + w * (Z @ z_i)`（`_calc_gradient` L.153, `model_dual_expfam.py`）

**先行研究[1]との対応：** [1]では Bernoulli のみ。形式は同一（スカラー w_0, w）。

**従来の原稿 eq(4) との差異：**  
原稿は $\eta_{ij}^Y = \mathbf{z}_i^\top \mathbf{W}_Y \mathbf{z}_j$（行列 W_Y）と書いていたが、
**実装はスカラー。必ず式(A1)の形に修正すること。**

---

### 1.3 属性データの生成モデル（提案: X 側）

$$x_{il} \mid \mathbf{z}_i, \mathbf{F} \sim \mathrm{ExpFam}_X\!\left(\eta_{il}^X\right), \quad
\eta_{il}^X = \mathbf{f}_l^\top \mathbf{z}_i \tag{A2}$$

ここで $\mathbf{f}_l \in \mathbb{R}^k$ は荷重行列 $\mathbf{F} \in \mathbb{R}^{d \times k}$ の第 $l$ 行ベクトルを列ベクトルとして転置したもの。
行列表記では $\boldsymbol{\eta}_i^X = \mathbf{F} \mathbf{z}_i \in \mathbb{R}^d$。

**実装根拠：** `eta_x_i = F @ z_i`（`_calc_gradient` L.141, `model_dual_expfam.py`）

**バイアス項について：** 従来の原稿 eq(5) は $\eta_{il}^X = w_{0l} + \mathbf{z}_i^\top \mathbf{w}_l$ と書いているが、
実装では $w_{0l} = 0$（バイアスなし）。先行研究[1]も同様に省略（model.py L.229: "Note: μ_x is omitted"）。

---

### 1.4 完全データ対数尤度

$$\log p(\mathbf{X}, \mathbf{Y}, \mathbf{Z} \mid \boldsymbol{\theta}) =
\underbrace{\sum_{i=1}^n \log p(\mathbf{z}_i)}_{\text{Z事前分布}}
+ \underbrace{\sum_{i=1}^n \sum_{l=1}^d \log p_X(x_{il} \mid \eta_{il}^X)}_{\text{X尤度}}
+ \underbrace{\sum_{i < j} \log p_Y(y_{ij} \mid \eta_{ij}^Y)}_{\text{Y尤度}} \tag{A3}$$

**注意：** Y の和は $i < j$（対称な Y の二重計上を避けるため）。実装もこの慣行に従う。

---

## 2. パラメータ集合

$$\boldsymbol{\theta} = \{\mathbf{F},\ \boldsymbol{\Sigma},\ w_0^Y,\ w^Y\}$$

| 記号 | 説明 | 形状 | 推定法 |
|------|------|------|--------|
| $\mathbf{F}$ | 属性荷重行列 | $d \times k$ | 解析解（Gauss-X）/ Adam（非Gauss-X）|
| $\boldsymbol{\Sigma}$ | 属性ノイズ分散（対角行列） | $d \times d$ | 解析解（Gauss-X のみ）|
| $w_0^Y$ | 関係データのバイアス | スカラー | Adam |
| $w^Y$ | 関係データの重み | スカラー | Adam |
| $\sigma_z^2 = 1$ | 潜在変数の事前分散 | スカラー | 固定（推定しない）|
| $\sigma_y^2$ | Y側分散（Gauss-Y のみ） | スカラー | 解析解（Gauss-Y のみ）|

---

## 3. EMアルゴリズム：Q関数

$$Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{\mathrm{old}}) = \mathbb{E}_{p(\mathbf{Z} \mid \mathbf{X}, \mathbf{Y}, \boldsymbol{\theta}^{\mathrm{old}})}\!\left[\log p(\mathbf{X}, \mathbf{Y}, \mathbf{Z} \mid \boldsymbol{\theta})\right] \tag{A4}$$

モンテカルロ近似（MCサンプル数 L）：

$$\hat{Q}(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{\mathrm{old}}) = \frac{1}{L} \sum_{l=1}^L \log p\!\left(\mathbf{X}, \mathbf{Y}, \mathbf{Z}^{(l)} \mid \boldsymbol{\theta}\right),
\quad \mathbf{Z}^{(l)} \sim q(\mathbf{Z}) \tag{A5}$$

**実装根拠：** `calc_Q_dual` 関数（`utils_expfam.py` L.324），`L` は `DualExpFamLSM.__init__` の引数（デフォルト L=10）。

---

## 4. E ステップ：ラプラス近似

各 $i$ について事後分布 $p(\mathbf{z}_i \mid \mathbf{X}, \mathbf{Y}, \mathbf{Z}_{-i}, \boldsymbol{\theta}^{\mathrm{old}})$ を
ガウス分布で近似する（ラプラス近似）。

**モード（事後平均の近似）：**
$$\mathbf{m}_i = \arg\max_{\mathbf{z}_i} \log p(\mathbf{z}_i \mid \mathbf{X}, \mathbf{Y}, \mathbf{Z}_{-i}, \boldsymbol{\theta}^{\mathrm{old}}) \quad \text{（ニュートン法）} \tag{A6}$$

**精度行列：**

$$\mathbf{A}_i = \underbrace{\frac{1}{\sigma_z^2} \mathbf{I}_k}_{\text{Z事前分布}}
+ \underbrace{\mathbf{F}^\top \mathbf{V}_X \mathbf{F}}_{\text{X尤度寄与}}
+ \underbrace{\frac{(w^Y)^2}{2} \sum_{j \neq i} A_Y''\!\left(\eta_{ij}^Y\right) \mathbf{z}_j \mathbf{z}_j^\top}_{\text{Y尤度寄与}} \tag{A7}$$

ここで $\mathbf{V}_X$ は X 分布族に依存：

$$\mathbf{V}_X = \begin{cases}
\boldsymbol{\Sigma}^{-1} & (\mathrm{ExpFam}_X = \text{ガウス分布}) \\
\mathrm{diag}\!\left(A_X''(\mathbf{F} \mathbf{m}_i)\right) & (\mathrm{ExpFam}_X = \text{ベルヌーイ・ポアソン})
\end{cases} \tag{A8}$$

**近似事後分布（ラプラス近似）：**
$$q_i(\mathbf{z}_i) = \mathcal{N}\!\left(\mathbf{m}_i,\ \mathbf{A}_i^{-1}\right) \tag{A9}$$

**実装根拠：**
- `_calc_precision_matrix`（`model_dual_expfam.py` L.167）で式(A7)(A8)を計算
- `calc_eta_newton`（`model.py` L.360）でニュートン法によりモードを探索しサンプリング

**先行研究[1]との対応：**
[1]の式(22)（精度行列）の一般化。[1]では X = Gaussian のみなので:
$$\mathbf{A}_i^{[1]} = \frac{1}{\sigma_z^2} \mathbf{I}_k + \mathbf{F}^\top \boldsymbol{\Sigma}^{-1} \mathbf{F} + \frac{w^2}{2} \sum_{j \neq i} s_{ij}(1-s_{ij}) \mathbf{z}_j \mathbf{z}_j^\top$$
提案手法では第2項・第3項を指数型分布族全般に拡張（式(A7)(A8)）。

---

## 5. E ステップ：対数事後分布の勾配

ニュートン法に使用する勾配（先行研究[1] Eq.(23) の一般化）：

$$\nabla_{\mathbf{z}_i} \log p(\mathbf{z}_i \mid \cdot) = -\frac{1}{\sigma_z^2} \mathbf{z}_i
+ \mathbf{F}^\top \mathbf{V}_X^{1/2} \left[T_X(\mathbf{x}_i) - A_X'(\mathbf{F}\mathbf{z}_i)\right]
+ \frac{w^Y}{2} \sum_{j \neq i} \left[T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)\right] \mathbf{z}_j \tag{A10}$$

ただし Gaussian X では $\mathbf{F}^\top \mathbf{V}_X^{1/2}[\cdots] = \mathbf{F}^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}_i - \mathbf{F}\mathbf{z}_i)$ となる。

**1/2 係数について（重要）：**
式(A10)の第3項の 1/2 は先行研究[1]と同じ慣行（[1] Eq.(23)）であり、実装も一致。
これは Y 側の対数尤度 $\sum_{i<j}$ を微分する際に生じる係数で、M ステップの勾配計算とも整合する。

---

## 6. M ステップ

$$\boldsymbol{\theta}^{\mathrm{new}} = \arg\max_{\boldsymbol{\theta}} \hat{Q}(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{\mathrm{old}}) \tag{A11}$$

**Gaussian X の F 解析解（先行研究[1] Eq.(10) と同一）：**

$$\mathbf{F}^{\mathrm{new}} = \left(\sum_{l=1}^L \mathbf{X}^\top \mathbf{Z}^{(l)}\right) \left(\sum_{l=1}^L {\mathbf{Z}^{(l)}}^\top \mathbf{Z}^{(l)}\right)^{-1} \tag{A12}$$

**非Gaussian X の F 勾配（Adam を適用）：**

$$\nabla_{\mathbf{F}} \hat{Q}_X = \frac{1}{L} \sum_{l=1}^L \left[\mathbf{X} - A_X'\!\left(\mathbf{Z}^{(l)} \mathbf{F}^\top\right)\right]^\top \mathbf{Z}^{(l)} \tag{A13}$$

**w_0^Y, w^Y の勾配（Adam を適用、先行研究[1] Eq.(24)(25) の一般化）：**

$$\frac{\partial \hat{Q}}{\partial w_0^Y} = \frac{1}{2L} \sum_{l=1}^L \sum_{i \neq j} \left[T_Y(y_{ij}) - A_Y'\!\left(\eta_{ij}^Y\right)\right] \tag{A14}$$

$$\frac{\partial \hat{Q}}{\partial w^Y} = \frac{1}{2L} \sum_{l=1}^L \sum_{i \neq j} \left[T_Y(y_{ij}) - A_Y'\!\left(\eta_{ij}^Y\right)\right] \mathbf{z}_i^\top \mathbf{z}_j \tag{A15}$$

---

## 7. 2ページ予稿に入れる式の優先順位

### 必須（2ページ予稿で省略不可）

1. **式(A7)の精度行列 A_i**（3項構造）：本研究の技術的貢献の核心
   - 先行研究[1]は第2項が Gaussian X 専用の F^T Σ^{-1} F のみ
   - 提案手法は F^T V_X F（V_X が分布族依存）で一般化
   - この式1本で「何を一般化したか」が伝わる

### 採用推奨（先生コメント対応に有効）

2. **式(A5)のQ関数 MC 近似**：「結局何をしているか」の答えになる
3. **式(A9)のラプラス近似 q_i(z_i)**：MCEM + Laplace の明示

### 省略可（本文の文章で代替可能）

4. 式(A10)の勾配：「先行研究[1] Eq.(23) を指数型分布族に一般化」と文章で済む
5. 式(A12)(A13)のM-step：「解析解またはAdamで最大化」と文章で済む
6. 式(A14)(A15)：省略可

### Wordに貼る際の補足

- ベクトル・行列は太字（LaTeX の \mathbf{·}）
- スカラーは斜体（w_0^Y, w^Y）
- 式番号は (4), (5), ... を原稿の続き番号で振る
- η の上付き添字（Y, X）はスーパースクリプト
- A_X'', A_Y'' は分散関数（LaTeX では A_X'' または A_X^{''}）

---

## 8. 記号一覧（原稿全体で統一すべき）

| 記号 | 意味 | 注意点 |
|------|------|--------|
| $\mathbf{z}_i \in \mathbb{R}^k$ | オブジェクト $i$ の潜在ベクトル | |
| $\mathbf{F} \in \mathbb{R}^{d \times k}$ | 属性荷重行列 | 先行研究[1]も同名 |
| $\boldsymbol{\Sigma}$ | 属性データ分散（対角行列, Gauss-X のみ） | 対角行列 $\mathrm{diag}(\sigma_1^2,\ldots,\sigma_d^2)$ |
| $w_0^Y, w^Y$ | 関係データのスカラーパラメータ | **行列 W_Y ではない** |
| $\eta_{ij}^Y$ | 関係データの自然パラメータ | $w_0^Y + w^Y \mathbf{z}_i^\top \mathbf{z}_j$ |
| $\eta_{il}^X$ | 属性データの自然パラメータ | $\mathbf{f}_l^\top \mathbf{z}_i$ （バイアスなし）|
| $A_X'(\cdot),\ A_Y'(\cdot)$ | X, Y の分布族の平均関数 | = 各分布族の期待値関数 |
| $A_X''(\cdot),\ A_Y''(\cdot)$ | X, Y の分布族の分散関数 | 精度行列の重みとして使用 |
| $L$ | MCサンプル数 | **R ではなく L**（先行研究[1]・実装ともに L） |
| $\mathbf{m}_i$ | $z_i$ 事後分布のモード | ニュートン法で求める |
| $\mathbf{A}_i$ | $z_i$ 事後分布の精度行列 | $= -\nabla^2_{z_i} \log p(z_i|\cdot)$ |
