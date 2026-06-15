# NotebookLM用 研究理解資料

**作成日:** 2026-05-18  
**作成根拠:** Phase 0〜1D の監査結果（CSV照合済み・コード確認済み）  
**注意:** 推測による補完は行っていない。未確認事項は「未確認」と明記している。

---

## 1. この資料の目的

本資料は、NotebookLM に Dual-ExpFam LSM の研究内容を正確に理解させるために作成した。
以下の点を特に意識して書いている。

- 既存研究（NOLTA 2024）と本研究（Dual-ExpFam LSM）を混同しない
- 原稿で採用している数式と、現行 Python 実装の状態を混同しない
- 実験結果はCSV照合済みの値だけを使う
- 「確認済み」「注意点」「未確認」を明示する

---

## 2. 研究の一言要約

**属性データ X と関係データ Y の両方を指数型分布族で統一的に扱える潜在構造モデル（Dual-ExpFam LSM）を提案し、人工データ実験で有効性を示した。**

既存研究（NOLTA 2024）が X=Gaussian・Y=Bernoulli に固定していた分布族の制約を解除し、
ポアソン・ガウス・ベルヌーイなど多様な分布族の組み合わせを一つの枠組みで扱えるようにした。

---

## 3. 研究背景

ソーシャルネットワーク・購買履歴・推薦システムなどでは、
「対象間の関係を表す行列（関係データ）」と「各対象の属性（属性データ）」が同時に観測される。
このようなデータから共通の低次元潜在変数を推定するのが属性情報付き潜在構造モデルである。

しかし従来手法（NOLTA 2024）では分布族が固定されており、
カウントデータ・連続値関係データなどに適用する際には再定式化が必要だった。

---

## 4. 対象とするデータ

### 4.1 関係データ

$n$ 個のオブジェクト間の関係を表す行列 $\mathbf{Y} = (y_{ij})$。
$y_{ij}$ の分布族はデータ種別に応じて選択する：
- 二値関係（友人か否か）: Bernoulli 分布
- カウント関係（コメント数）: Poisson 分布
- 連続値関係: Gaussian 分布

$\mathbf{Y}$ は対称行列で対角成分は存在しない（自己関係なし）。上三角のみ観測し、$i < j$ で扱う。

### 4.2 属性データ

各オブジェクト $i$ が持つ $d$ 次元の属性ベクトルをまとめた行列 $\mathbf{X} = (x_{il})$（$1 \le l \le d$）。
属性の分布族も同様に選択する：
- 連続属性: Gaussian 分布
- 二値属性（所属有無）: Bernoulli 分布
- カウント属性（アクセス数）: Poisson 分布

### 4.3 属性情報付き関係データ

$\mathbf{Y}$ と $\mathbf{X}$ の両方から共通の $k$ 次元潜在変数行列
$\mathbf{Z} \in \mathbb{R}^{n \times k}$ を推定する問題を扱う。

---

## 5. 既存研究 NOLTA 2024

### 5.1 既存研究の位置づけ

Mikawa et al., "A study on latent structural models for binary relational data with attribute information,"
NOLTA, IEICE, vol. 15, no. 2, pp. 335–353, 2024.

本研究の直接の出発点であり、提案手法の特殊ケースとして包含される。
Python 再現実装: `reproduction/src/model.py`（クラス名: `LatentStructuralModel`）
MATLAB 原典実装: `Mato Lab Program/calcEtaNewton.m` 等

### 5.2 既存研究の生成モデル

$$z_i \sim \mathcal{N}(0, \sigma_z^2 I_k), \quad \sigma_z^2 = 1 \text{（固定）}$$

$$y_{ij} \mid z_i, z_j \sim \mathrm{Bernoulli}\!\left(\sigma(w_0 + w\, z_i^\top z_j)\right) \quad (i < j)$$

$$x_{il} \mid z_i \sim \mathcal{N}(f_l^\top z_i,\ \sigma_l^2)$$

分布族は **Y = Bernoulli、X = Gaussian に固定**。バイアス項なし（$\eta_{il}^X = f_l^\top z_i$）。

### 5.3 既存研究の推定方法

- Monte Carlo EM（MCEM）アルゴリズム
- E-step: Laplace 近似により $q_i(z_i) = \mathcal{N}(m_i, A_i^{-1})$ を求め、$L$ 個のMCサンプルを生成
- M-step: F の解析解更新、$w_0, w$ の Adam 更新
- 潜在次元数 $k$ の選択: BIC

### 5.4 既存研究の限界

- X の分布族が Gaussian に固定されており、カウントデータや二値属性には適用できない
- Y の分布族が Bernoulli に固定されており、連続値・カウント関係データには適用できない

### 5.5 既存研究内の注意点（重要）

**NOLTA 2024 PDF の E-step 式と MATLAB 実装の間に不整合がある。**

- NOLTA 2024 PDF Eq.(22)(23): E-step の精度行列・勾配の Y 側 Term3 に **1/2 あり**
- MATLAB `calcEtaNewton.m` の `calcAi` 関数 (L.56-63): 精度行列 Y 側 Term3 に **1/2 なし**
- MATLAB `calcGrad` 関数 (L.43-49): 勾配 Y 側 Term3 に **1/2 なし**

本研究では、数学的再導出と MATLAB 確認に基づき **1/2 なしを正しいとして採用**する（詳細は §14.1）。

---

## 6. 本研究 Dual-ExpFam LSM

### 6.1 本研究の目的

X と Y の分布族をともに指数型分布族に一般化し、分析者が柔軟に指定できる
属性情報付き潜在構造モデルを提案・実験で検証すること。

### 6.2 基本アイデア

既存研究のモデル構造（MCEM + Laplace 近似）を保ちつつ、
GLM（一般化線形モデル）の枠組みで X・Y 側の分布族を抽象化する。
任意の指数型分布族は以下の形で表現できる：

$$p(x \mid \eta) = h(x)\exp\{\eta\, T(x) - A(\eta)\}$$

- $\eta$: 自然パラメータ
- $T(x)$: 十分統計量
- $A(\eta)$: 対数分配関数（$A'(\eta)$ = 期待値関数、$A''(\eta)$ = 分散関数）

### 6.3 既存研究から継承した部分

- 潜在変数の事前分布: $z_i \sim \mathcal{N}(0, I_k)$（$\sigma_z^2 = 1$ 固定）
- 関係データの線形予測子: $\eta_{ij}^Y = w_0^Y + w^Y z_i^\top z_j$（スカラー $w_0^Y, w^Y$）
- 属性データのバイアスなし: $\eta_{il}^X = f_l^\top z_i$
- MCEM + Laplace 近似の枠組み
- BIC による潜在次元数の選択

### 6.4 本研究で拡張した部分

- **X 側の分布族**: Gaussian 固定 → Gaussian・Bernoulli・Poisson から任意指定
- **Y 側の分布族**: Bernoulli 固定 → Gaussian・Bernoulli・Poisson から任意指定
- E-step 精度行列 Term2 の一般化: $F^\top \Sigma^{-1} F$（Gaussian のみ）→ $F^\top V_X F$（任意族）
- 非 Gaussian X での F 更新: 解析解 → Adam 勾配上昇
- Gaussian Y での分散 $\sigma_y^2$ の M-step 推定

### 6.5 本研究の新規性

既存研究[1]の精度行列の第2項 $F^\top \Sigma^{-1} F$ を $F^\top V_X(m_i) F$ に一般化した点が核心。
$V_X$ が X の分布族に依存することで、Bernoulli-X・Poisson-X 等にも対応できる。

---

## 7. 数式モデル

ここで $A'(\eta)$ は指数型分布族における平均関数，$A''(\eta)$ は分散関数を表す。

### 7.1 記号定義

| 記号 | 意味 | 形状 | 備考 |
|------|------|------|------|
| $n$ | オブジェクト数 | スカラー | 実験: 150 |
| $d$ | 属性次元数 | スカラー | 実験: 15 |
| $k$ | 潜在次元数 | スカラー | 真値: 3、探索: 1〜6 |
| $z_i$ | オブジェクト $i$ の潜在ベクトル | $(k,)$ | |
| $\mathbf{Z}$ | 潜在変数行列 | $(n, k)$ | |
| $x_i$ | オブジェクト $i$ の属性ベクトル | $(d,)$ | |
| $\mathbf{X}$ | 属性データ行列 | $(n, d)$ | |
| $y_{ij}$ | オブジェクト $i, j$ 間の関係 | スカラー | $i < j$ のみ |
| $\mathbf{F}$ | 属性荷重行列 | $(d, k)$ | 推定パラメータ |
| $\boldsymbol{\Sigma}$ | 属性データ分散（対角） | $(d, d)$ | Gaussian-X のみ推定 |
| $w_0^Y$ | 関係データバイアス | スカラー | 推定パラメータ |
| $w^Y$ | 関係データ重み | スカラー | 推定パラメータ |
| $\sigma_y^2$ | Y 側分散 | スカラー | Gaussian-Y のみ推定 |
| $L$ | MC サンプル数 | スカラー | 実験: 5 |

### 7.2 潜在変数の事前分布

$$z_i \sim \mathcal{N}(0, I_k), \quad i = 1, \ldots, n$$

$\sigma_z^2 = 1$ に固定（識別性確保）。推定しない。

### 7.3 属性データ X の生成モデル

$$x_{il} \mid z_i \sim \mathrm{ExpFam}_X(\eta_{il}^X), \quad \eta_{il}^X = f_l^\top z_i$$

バイアス項なし。行列表記: $\boldsymbol{\eta}_i^X = \mathbf{F} z_i \in \mathbb{R}^d$。

分布族ごとの $A_X'(\eta)$（期待値関数）と $A_X''(\eta)$（分散関数）：

| $\mathrm{ExpFam}_X$ | $A_X'(\eta)$（期待値） | $A_X''(\eta)$（分散） |
|--------------------|---------------------|---------------------|
| Gaussian | $\eta$（恒等写像） | $1$（/$\sigma_j^2$ で除す） |
| Bernoulli | $\sigma(\eta) = 1/(1+e^{-\eta})$ | $\sigma(\eta)(1-\sigma(\eta))$ |
| Poisson | $\exp(\eta)$ | $\exp(\eta)$ |

### 7.4 関係データ Y の生成モデル

$$y_{ij} \mid z_i, z_j \sim \mathrm{ExpFam}_Y(\eta_{ij}^Y), \quad \eta_{ij}^Y = w_0^Y + w^Y z_i^\top z_j \quad (i < j)$$

$w_0^Y, w^Y \in \mathbb{R}$ はスカラー（行列ではない）。Y は対称行列（$y_{ij} = y_{ji}$）。

### 7.5 同時分布

$$p(\mathbf{X}, \mathbf{Y}, \mathbf{Z} \mid \boldsymbol{\theta}) = \prod_{i=1}^n p(z_i) \cdot \prod_{i=1}^n \prod_{l=1}^d p_X(x_{il} \mid \eta_{il}^X) \cdot \prod_{i < j} p_Y(y_{ij} \mid \eta_{ij}^Y)$$

パラメータ: $\boldsymbol{\theta} = \{F, w_0^Y, w^Y\}$（+ Gaussian-X のとき $\boldsymbol{\Sigma}$、Gaussian-Y のとき $\sigma_y^2$）

### 7.6 対数尤度（完全データ）

$$\log p(\mathbf{X}, \mathbf{Y}, \mathbf{Z} \mid \boldsymbol{\theta})
= \underbrace{\sum_{i=1}^n \log p(z_i)}_{\text{Z 事前分布}}
+ \underbrace{\sum_{i=1}^n \sum_{l=1}^d \log p_X(x_{il} \mid \eta_{il}^X)}_{\text{X 尤度}}
+ \underbrace{\sum_{i < j} \log p_Y(y_{ij} \mid \eta_{ij}^Y)}_{\text{Y 尤度（片側和）}}$$

Y 尤度を **片側和** $\sum_{i < j}$ で書くことが重要（対称性により $\sum_{i \neq j}$ の半分に等しい）。

### 7.7 Q 関数（Monte Carlo 近似）

$$\hat{Q}(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{\mathrm{old}}) = \frac{1}{L} \sum_{l=1}^L \log p\!\left(\mathbf{X}, \mathbf{Y}, \mathbf{Z}^{(l)} \mid \boldsymbol{\theta}\right), \quad \mathbf{Z}^{(l)} \sim q(\mathbf{Z})$$

### 7.8 E-step の勾配（原稿採用式）

$$\nabla_{z_i} \log p(z_i \mid \cdot) =
\underbrace{-z_i}_{\text{Term1: Z事前}}
+ \underbrace{F^\top V_X[x_i - A_X'(Fz_i)]}_{\text{Term2: X尤度}}
+ \underbrace{w^Y \sum_{j \neq i} [T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)] z_j}_{\text{Term3: Y尤度}}$$

$$V_X = \begin{cases}
\Sigma^{-1} & (\mathrm{ExpFam}_X = \text{Gaussian}) \\
\mathrm{diag}(A_X''(Fz_i)) & (\text{Bernoulli/Poisson-X})
\end{cases}$$

**Term3 の係数 $w^Y$（1/2 なし）。** 理由: Y 尤度を $\sum_{i<j}$ で定義すると、
$z_i$ に関する微分で $j > i$ の項と $j < i$ の項が合算されて $\sum_{j \neq i}$ になる。
数学的証明: `docs/math_notes/half_factor_math_explanation.md` 参照。

### 7.9 E-step の精度行列（原稿採用式・Eq.(6)）

$$A_i = \underbrace{I_k}_{\text{Term1}}
+ \underbrace{F^\top V_X(m_i) F}_{\text{Term2}}
+ \underbrace{(w^Y)^2 \sum_{j \neq i} A_Y''(\eta_{ij}^Y)\, z_j z_j^\top}_{\text{Term3（1/2 なし）}}$$

- Term1: Z 事前分布の寄与（$\sigma_z^2 = 1$）
- Term2: X 側の寄与（$V_X$ は X の分布族に依存）
- Term3: Y 側の寄与（**1/2 なし**、これが原稿採用式）

ラプラス近似: $q_i(z_i) = \mathcal{N}(m_i, A_i^{-1})$、$m_i$ はニュートン法で求めるモード。

### 7.10 M-step の更新

**F の更新:**
- Gaussian-X: 解析解（既存研究と同一形式）
  $F^{\mathrm{new}} = \left(\sum_l X^\top Z^{(l)}\right)\left(\sum_l {Z^{(l)}}^\top Z^{(l)}\right)^{-1}$
- 非 Gaussian-X: Adam 勾配上昇
  $\nabla_F \hat{Q}_X = \frac{1}{L}\sum_l [X - A_X'(Z^{(l)} F^\top)]^\top Z^{(l)}$

**$w_0^Y, w^Y$ の更新（Adam、正しい 1/2L あり）:**

$$\frac{\partial \hat{Q}}{\partial w_0^Y} = \frac{1}{2L} \sum_l \sum_{i \neq j} [T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)]$$

$\sum_{i \neq j}$ の全和を2で割ることは $\sum_{i < j}$ に換算するための正しい操作。

**$\boldsymbol{\Sigma}$ の更新:** Gaussian-X のみ解析解 MLE。非 Gaussian-X では $I_d$ に固定。  
**$\sigma_y^2$ の更新:** Gaussian-Y のみ解析解 MLE。

---

## 8. 推定アルゴリズム

### 8.1 EMアルゴリズム

E-step と M-step を交互に繰り返す。
- E-step: 現在のパラメータ $\boldsymbol{\theta}^{\mathrm{old}}$ のもとで潜在変数 $\mathbf{Z}$ の事後分布を推定
- M-step: 推定した $\mathbf{Z}$ のサンプルを用いて $\hat{Q}$ を $\boldsymbol{\theta}$ について最大化

### 8.2 Monte Carlo EM（MCEM）

$\mathbf{Z}$ の事後分布は解析的に求めにくいため、Laplace 近似 + MC サンプリングで近似する。
各オブジェクト $i$ について $z_i$ を $q_i(z_i) = \mathcal{N}(m_i, A_i^{-1})$ からサンプリングし $L$ 個のサンプルを生成する。

### 8.3 Laplace 近似

$z_i$ の条件付き事後分布 $p(z_i \mid X, Y, Z_{-i}, \boldsymbol{\theta}^{\mathrm{old}})$ を
ガウス分布 $\mathcal{N}(m_i, A_i^{-1})$ で近似する。

- $m_i$: ニュートン法による対数事後分布のモード（最大化点）
- $A_i$: 対数事後分布の負のヘッセ行列（精度行列）= §7.9 の式

### 8.4 Adam による更新

非 Gaussian-X での F 更新、および $w_0^Y, w^Y$ の更新に Adam を使用。
設定: 学習率 $\alpha = 0.01$、$\beta_1 = 0.9$、$\beta_2 = 0.999$、最大 50 反復。

### 8.5 実装上の流れ（1 EM 反復）

```
for i = 1 to n:
    m_i = Newton法でモードを求める（_calc_gradient と _calc_precision_matrix を使用）
    Z_l = N(m_i, A_i^{-1}) からサンプリング
Z_samples = Z^(1), ..., Z^(L)  （shape: n × k × L）

# M-step
F      = calc_F(X, Z_samples)           # Gauss-X: 解析解、他: Adam
Sigma  = calc_sigma(X, Z_samples, F)    # Gauss-X: 解析解、他: I_d
w0     = calc_w0(Y, Z_samples, w0, w)   # 常に Adam
w      = calc_w(Y, Z_samples, w0, w)    # 常に Adam
sigma_y = calc_sigma_y(Y, Z_samples, w0, w)  # Gauss-Y のみ
```

---

## 9. 実装との対応

### 9.1 主要ファイル

| ファイル | 役割 |
|---------|------|
| `reproduction/src/model.py` | ベースクラス `LatentStructuralModel`（既存研究再現） |
| `expfam/src/model_expfam.py` | `ExpFamLatentStructuralModel`（Y 側を ExpFam に拡張） |
| `expfam/src/model_dual_expfam.py` | `DualExpFamLSM`（X・Y 両方を ExpFam に拡張、提案手法本体） |
| `expfam/src/utils_expfam.py` | Q 関数・BIC・RMSE・Procrustes・EM 実行 |
| `expfam/src/data_generator_expfam.py` | 人工データ生成 |
| `expfam/src/exp_scenario_lib.py` | 実験共通設定・実験関数 |

### 9.2 クラス構造（継承）

```
LatentStructuralModel          （reproduction/src/model.py）
└── ExpFamLatentStructuralModel （expfam/src/model_expfam.py）
    └── DualExpFamLSM           （expfam/src/model_dual_expfam.py）← 提案手法本体
```

`DualExpFamLSM` は `_calc_gradient` と `_calc_precision_matrix` を完全にオーバーライドして X・Y 両方の ExpFam に対応している。

### 9.3 数式とコードの対応表

| 数式 | 実装場所 | 行番号 |
|------|---------|-------|
| 精度行列 Term1（Z事前） | `model_dual_expfam.py::_calc_precision_matrix` | L.181 |
| 精度行列 Term2（Gauss-X） | 同上 | L.186-190 |
| 精度行列 Term2（非Gauss-X） | 同上 | L.192-194 |
| 精度行列 Term3（Y側） | 同上 | L.200 |
| 勾配 Term2（Gauss-X） | `model_dual_expfam.py::_calc_gradient` | L.144-147 |
| 勾配 Term2（非Gauss-X） | 同上 | L.149-150 |
| 勾配 Term3（Y側） | 同上 | L.159 |
| F 更新（Gauss-X 解析解） | `DualExpFamLSM.calc_F` → parent | — |
| F 更新（非Gauss-X Adam） | `DualExpFamLSM._calc_F_adam` | L.219-268 |
| w0/w 更新（Adam） | `model_expfam.py::calc_w0, calc_w` | L.149-210 |
| Q 関数（Dual） | `utils_expfam.py::calc_Q_dual` | L.324-352 |
| BIC（Dual） | `utils_expfam.py::calc_bic_dual` | L.386-404 |
| Procrustes + RMSE(Z) | `utils_expfam.py::run_em_dual` | L.555-557 |

### 9.4 現行 Python 実装に残る注意点

**⚠ E-step の Y 側 Term3 に spurious な 0.5 が残存している。**

| 場所 | 内容 | 原稿式との差 |
|-----|------|------------|
| `model_dual_expfam.py` L.159 | `term3 = 0.5 * w * (Z.T @ residual_y)` | `w` の前に余分な `0.5` |
| `model_dual_expfam.py` L.200 | `term3 = 0.5 * (w**2) * (...)` | `w**2` の前に余分な `0.5` |
| `model_expfam.py` L.109, 135 | 同じ `0.5` が存在（DualExpFamLSMでは使われない） | 同上 |

- 0.5 が掛かっているのは Y 側 Term3 のみ。Term1・Term2 は原稿式と一致している。
- そのため「posterior 全体の Newton 方向が正しい」とは断定できない。
  Y 側が支配的な場合には近似的に打ち消される可能性があるが、全体としては原稿式と一致しない。
- Laplace 近似のサンプリング分散は Y 側が支配的な方向で大きくなりやすい。
- **修正は修論フェーズで対応予定。現時点では触らない。**

M-step の `/2L`（w0・w 更新の `grad / (2 * L * phi)`）は**正しい**。
Q 関数 Y 側の `0.5 * sum(log p)` も**正しい**。

---

## 10. 実験設定

### 10.1 実験の目的

提案手法 Dual-ExpFam LSM の有効性を人工データで検証する。
具体的には以下を確認する：
1. 正しい分布族を指定すれば潜在変数 Z を精度よく推定できるか
2. 情報量基準（BIC）が真の潜在次元 $k^* = 3$ を選択できるか
3. サンプル数 $n$ 増加に伴い推定精度が改善するか
4. 分布族の誤指定がどの程度 RMSE(Z) を悪化させるか
5. 先行研究と同一条件で同等の精度が得られるか

### 10.2 Scenario A/B/C

| シナリオ | 真の $\mathrm{ExpFam}_X$ | 真の $\mathrm{ExpFam}_Y$ | 略称 |
|---------|------------------------|------------------------|------|
| **A** | Poisson | Bernoulli | P-B |
| **B** | Gaussian | Poisson | G-P |
| **C** | Bernoulli | Gaussian | B-G |

**共通設定（`expfam/src/exp_scenario_lib.py` L.40-45 より確認）:**
- $n = 150$（オブジェクト数）
- $d = 15$（属性次元数）
- $k^* = 3$（真の潜在次元数）
- 10 試行（trial 0〜9）の平均を評価
- MC サンプル数 $L = 5$
- EM 反復数: 8

### 10.3 k-sweep 実験（Exp1）

$k \in \{1, 2, 3, 4, 5, 6\}$ で RMSE(Z) を評価し、BIC による次元選択を確認する。

### 10.4 n-sweep 実験（Exp2）

$n \in \{50, 100, 150, 200, 250, 300\}$ で RMSE(Z) の推移を確認する。
k を真値 $k^* = 3$ に固定して評価。

### 10.5 mismatch 実験（Exp4）

正解の分布族指定を基準として、意図的に誤った分布族を指定した場合の RMSE(Z) の悪化倍率を評価する。
比較する条件：

1. Proposed: 正しい分布族（$\mathrm{ExpFam}_X$ = 真の族、$\mathrm{ExpFam}_Y$ = 真の族）
2. X-side misspec.: Y 側を正解に固定し、X 側のみを別の分布族に変更
3. Y-side misspec.: X 側を正解に固定し、Y 側のみを別の分布族に変更
4. Fixed Gauss-X/Bern-Y: 先行研究固定条件（X=Gaussian, Y=Bernoulli）を各シナリオに適用
5. Y-only (fix_x): X 情報を完全に無視し（F=0 固定）、Y 側のみで Z を推定
6. X-only (fix_w): Y 情報を完全に無視し（w=0 固定）、X 側のみで Z を推定

### 10.6 Control 条件

先行研究と同一の設定（$\mathrm{ExpFam}_X$ = Gaussian、$\mathrm{ExpFam}_Y$ = Bernoulli）で
提案手法と先行研究再現実装の RMSE(Z) を比較する。
設定: $n = 150, d = 15$、5 試行。

### 10.7 評価指標

- **RMSE(Z)**: 潜在変数推定精度（主指標）。Procrustes 回転後に計算。
- **RMSE(F)**: 荷重行列推定精度
- **RMSE(Y)**: 関係データ再構成精度
- **RMSE(X)**: 属性データ再構成精度
- **BIC**: $-2\hat{Q}_{\mathrm{strict}} + \{kd - k(k-1)/2 + [\text{Gauss-X: }d] + [\text{Gauss-Y: }1]\} \ln n$

**Procrustes 回転:** 潜在変数は回転に対して不変なため、RMSE(Z) 計算前に最適回転行列 $R$ を適用する。
$\min_R \|Z_{\mathrm{est}} R - Z_{\mathrm{true}}\|$（$R$ は直交行列）。
実装: `utils_expfam.py::procrustes_rotation`。全実験で適用済み（`run_em_dual` L.555-557）。

---

## 11. 実験結果

### 11.1 k-sweep 結果（CSV 照合済み）

RMSE(Z) の 10 試行平均（`exp_scenario_{A,B,C}_exp1_k.csv` より）:

| $k$ | Scen. A (P-B) | Scen. B (G-P) | Scen. C (B-G) |
|-----|:-----------:|:-----------:|:-----------:|
| 1 | 0.9526 | 1.0627 | 0.9979 |
| 2 | 0.7655 | 0.6510 | 0.5764 |
| **3** | **0.2784** | **0.1817** | **0.0284** |
| 4 | 0.5050 | 0.4360 | 0.2989 |
| 5 | 0.7068 | 0.5375 | 0.3879 |
| 6 | 0.6917 | 0.6025 | 0.4177 |

**全シナリオで $k = 3$ のとき RMSE(Z) が最小。**
原稿掲載値（A: 0.278, B: 0.182, C: 0.028）と一致（小数 3 桁丸め）。

### 11.2 n-sweep 結果（CSV 照合済み）

RMSE(Z) の 10 試行平均（`exp_scenario_{A,B,C}_exp2_n.csv` より）:

| $n$ | Scen. A | Scen. B | Scen. C |
|-----|:------:|:------:|:------:|
| 50 | 0.4056 | 0.1901 | 0.0530 |
| 100 | 0.3194 | 0.1914 | 0.0350 |
| 150 | 0.2785 | 0.1703 | 0.0292 |
| 200 | 0.2469 | 0.1682 | 0.0248 |
| 250 | 0.2245 | 0.1352 | 0.0219 |
| 300 | 0.2076 | 0.1312 | 0.0202 |

**削減率（n=50→300）:** A: 48.8%、B: 31.0%、C: 62.0%（CSV 照合済み）

**注意: Scen. B の n=50→100 で一時的な増加**（0.1901→0.1914）。原稿 L.81 に記載済み。

### 11.3 mismatch 結果（CSV 照合済み）

各シナリオの最大悪化倍率（`exp_scenario_{A,B,C}_exp4_mismatch.csv` より）:

| シナリオ | 最大悪化倍率 | 条件 | 条件の種別 |
|---------|:----------:|------|----------|
| A | **3.41×** | X=Bernoulli, Y=Bernoulli | X-only 誤指定（Y=Bern は正解） |
| B | **7.35×** | X=Poisson, Y=Bernoulli | **両方** 誤指定 |
| C | **41.5×** | X=Gaussian, Y=Poisson | **両方** 誤指定 |

**重要な注意:** Scen. B と C の最大値は「X・Y 両方を誤指定」した条件から来ている。
この条件は図 1(b) に独立したバーとして表示されていない（§11.6 参照）。

### 11.4 Control 条件の結果（CSV 照合済み）

$k = 3$、5 試行（`comparison_control_exp1.csv` より）:

| モデル | RMSE(Z) 平均 | RMSE(Z) 標準偏差 |
|-------|:-----------:|:--------------:|
| 先行研究（baseline） | 0.1793 | 0.0144 |
| 提案手法（dual_expfam） | 0.1799 | 0.0155 |
| 差（絶対値） | **0.0006** | — |

差 0.0006 < 0.001。提案手法は先行研究と実質同等の精度を維持している。

### 11.5 BIC による潜在次元選択（CSV 照合済み）

10 試行平均 BIC が最小となる $k$（`exp1_full_{A,B,C}.csv` より）:

| シナリオ | BIC 最小 $k$ | $k=3$ vs $k=4$ の BIC 差 |
|---------|:-----------:|:-----------------------:|
| A | **3** ✓ | 487（余裕あり） |
| B | **3** ✓ | 180（比較的小さい） |
| C | **3** ✓ | 484（余裕あり、BIC は負値） |

**注意: Scen. B の差が 180 と小さい。** 外れ試行では $k = 4$ が選ばれる可能性がある。
「全シナリオで BIC が $k^* = 3$ を選択」という主張は「10 試行平均 BIC」が最小の $k$ が 3 であるという意味。

### 11.6 図 1(a)(b) の説明

**図 1(a) `figures/fig1a_n_sweep_color.pdf/png`（2026-05-07 版）:**
n-sweep の結果。3 シナリオの RMSE(Z) を $n = 50$ を基準 1.0 に正規化した折れ線グラフ。

**図 1(b) `figures/fig1b_misspecification_color.pdf/png`（2026-05-07 版）:**
分布族誤指定の影響。シナリオごとに 3 種のバーを表示：

| バー色 | 内容 | Scen. A | Scen. B | Scen. C |
|-------|------|:-------:|:-------:|:-------:|
| 水色（斜め線） | X-side misspec. の最大倍率 | 3.4× | 3.0× | 3.7× |
| 橙（クロス） | Y-side misspec. の最大倍率 | 1.3× | 6.5× | 15.7× |
| 灰（水平線） | Fixed Gauss-X/Bern-Y | 2.5× | 6.5× | **23.6×** |

**図の最大表示値: Scen. C の灰色バー = 23.6×（先行研究固定条件）**

**原稿本文（L.83）の「最大 41.5 倍」との関係:**
- 23.6×：Fixed Gauss-X/Bern-Y 条件（図の灰色バー）
- 41.5×：X=Gaussian, Y=Poisson（**両方誤指定**、図中に対応するバーなし）
- 41.5× は CSV 全条件中の最大値として事実上正確だが、図から直接読み取れない。
- 図は代表 3 条件を示すものであり、全誤指定条件を網羅していない。

### 11.7 Scenario C の慎重な解釈

Scenario C（真の X=Bernoulli, Y=Gaussian）の mismatch 結果（CSV より）:

| 条件 | RMSE(Z) | 対 Proposed 比 |
|-----|:------:|:----------:|
| Proposed (Bern, Gauss) | 0.0287 | 1.00× |
| X=Poisson（X 誤指定、Y 正解） | 0.0284 | ≈ 0.99× |
| Y-only (fix_x=True) | 0.0286 | ≈ 0.99× |
| X=Gaussian（X 誤指定、Y 正解） | 0.1055 | 3.67× |
| X-only (fix_w=True) | 1.1024 | **38.4×** |
| Y=Bernoulli（Y 誤指定） | 0.4523 | 15.75× |

**観察:** Y-only 条件（X 情報を完全に除去）でも RMSE(Z) ≈ 0.0286 と Proposed とほぼ同等。
一方 X-only 条件では RMSE(Z) = 1.102 と壊滅的に悪化。

**可能性の高い解釈:** Gaussian Y は内積 $z_i^\top z_j$ を連続値で直接観測するため（観測数 $n(n-1)/2 = 11{,}175$ 個）、
Bernoulli X（観測数 $nd = 2{,}250$ 個、2 値）に比べて $Z$ の復元に強く寄与した可能性がある。
また fix_x=True のとき RMSE(X) = 0.500（Proposed の 0.456 より悪化）であり、
X 情報を除去すると X の予測精度は下がるが Z の推定精度はほぼ変わらないことを示している。

**断定しないこと:** Term2（X 側）と Term3（Y 側）の勾配ノルム比較は未実施。
「X 情報が使われていない」可能性は否定できない。
「Y=Gaussian が Z 推定に強く寄与した可能性がある」という表現に留める。

---

## 12. 本研究から言えること

以下は CSV・コード確認に基づいて主張できる内容：

1. **正しい分布族の指定が重要:** 誤指定により RMSE(Z) が最大 41.5 倍悪化（全条件中）
2. **BIC が $k^* = 3$ を正しく選択:** 全シナリオで 10 試行平均 BIC が最小の $k$ は 3
3. **$n$ 増加に伴い精度が改善:** 全シナリオで削減率 31〜62%
4. **先行研究と同等の精度（Control 条件）:** RMSE(Z) 差 0.0006 < 0.001
5. **指数型分布族への一般化は機能する:** 3 つのシナリオで提案手法が有効に動作

---

## 13. 主張しすぎると危険なこと

1. **「X 情報が Z 推定に有効に活用されている」:** Scen. C で Y-only ≈ Proposed のため要確認
2. **「Newton 方向は正しい」:** Y 側 Term3 のみ 0.5 が掛かっており、全体では断定不可
3. **「Scen. B で BIC が常に k=3 を選ぶ」:** BIC 差 180 は小さく、試行によっては k=4 の可能性あり
4. **「図から 41.5 倍を確認できる」:** 図の最大バーは 23.6×（灰色）であり、41.5× は図に表示なし
5. **「先行研究との差がゼロ」:** 差は 0.0006（ゼロではない）、「実質同等」が適切な表現

---

## 14. 現時点の矛盾・注意点

### 14.1 1/2 係数問題

**数学的正解（`docs/math_notes/half_factor_math_explanation.md` で確定）:**
- E-step 勾配 Term3: $w^Y \sum_{j \neq i}(\ldots)$（1/2 なし）
- E-step 精度行列 Term3: $(w^Y)^2 \sum_{j \neq i}(\ldots)$（1/2 なし）
- M-step の $/2L$: 正しい（$\sum_{i \neq j}$ を $\sum_{i < j}$ に換算するため）

**資料間の矛盾（整理済み）:**

| 資料 | 精度行列 Term3 | E-step 勾配 Term3 | 位置づけ |
|-----|:---:|:---:|------|
| 原稿 Eq.(6) | 1/2 なし ✓ | — | **正しい** |
| MATLAB calcAi | 1/2 なし ✓ | 1/2 なし ✓ | **正しい** |
| NOLTA 2024 PDF Eq.(22)(23) | **1/2 あり** | **1/2 あり** | 論文記載（本研究の再導出・MATLAB実装とは不一致） |
| Python 実装 L.200, L.159 | **0.5 あり** | **0.5 あり** | NOLTA PDF を踏襲した不整合 |
| `docs/math_notes/legacy/parameter_estimation_corrected_formulas.md` | 1/2 あり（旧） | 1/2 あり（旧） | **中間段階の矛盾資料**（採用不可） |

### 14.2 Python 実装と原稿式の違い

| 箇所 | 実装（現状） | 原稿採用式 | 差異 |
|-----|------------|----------|------|
| `model_dual_expfam.py` L.200 | `0.5 * (w**2) * (...)` | $(w^Y)^2 \sum_{j\neq i}(\ldots)$ | 0.5 余分 |
| `model_dual_expfam.py` L.159 | `0.5 * w * (...)` | $w^Y \sum_{j\neq i}(\ldots)$ | 0.5 余分 |
| `model_expfam.py` L.135, 109 | 同じ 0.5 あり | 同上 | DualExpFamLSM では未使用 |

修正予定: 修論フェーズ。現時点では触らない。

### 14.3 図 1(b) の 23.6 倍と本文 41.5 倍

- 図中最大値: **23.6×**（Scen. C、灰色バー = Fixed Gauss-X/Bern-Y 条件）
- 本文記載: **41.5×**（CSV 全条件中の最大、X=Gaussian + Y=Poisson の両方誤指定条件）
- 41.5× は図に表示されていない
- 詳細: `docs_for_notebooklm/03_figure_consistency_check.md` 参照

### 14.4 Scenario C の X-mismatch

- Y-only ≈ Proposed という現象の原因は未特定
- 「Y=Gaussian が Z 推定に強く寄与した可能性」が有力だが断定できない
- Term2 vs Term3 のノルム比較が未実施
- 追加検査が必要: E-step 中の ||Term2|| vs ||Term3|| の計測

---

## 15. NotebookLM に投入する際の推奨ファイル

| ファイル | 理由 |
|---------|------|
| `CLAUDE.md`（root） | 確定事項・方針の基準 |
| `conference_submission_final_draft.md` | 完成版原稿（数式・実験条件・数値の一次ソース） |
| `docs_for_notebooklm/NOTEBOOKLM_RESEARCH_BRIEF.md` | 本資料 |
| `docs_for_notebooklm/01_formula_code_audit.md` | 数式とコードの整合性監査結果 |
| `docs_for_notebooklm/02_experiment_result_verification.md` | 実験結果の照合結果 |
| `docs_for_notebooklm/03_figure_consistency_check.md` | 図と本文の整合性確認 |
| `docs/math_notes/half_factor_math_explanation.md` | 1/2 不要の数学的証明 |
| `docs/math_notes/half_factor_literature_code_check.md` | MATLAB vs Python 照合表 |
| `expfam/references/baseline_metrics.md` | 先行研究との比較数値 |

---

## 16. NotebookLM に投入しない方がよい、または注意して扱うファイル

| ファイル | 理由 |
|---------|------|
| `docs/math_notes/legacy/parameter_estimation_corrected_formulas.md` | E-step の 1/2 が現在の確定事項と矛盾する中間段階の文書 |
| `expfam/CLAUDE.md` | 旧 Gemini セッション向け。root の CLAUDE.md が優先 |
| `expfam/results/GEMINI_REPORT_*.md` | AI 生成レポート。CSV・コードで確認できない内容は採用しない |
| `expfam/results/archive/` 配下 | 旧実験結果。現行結果と混同しないこと |
| `paper_writing_examples/*.pdf` | 他学会の参考例。本研究の内容ではない |

---

## 17. NotebookLM に質問するとよい問い

- 「提案手法（Dual-ExpFam LSM）が先行研究（NOLTA 2024）と違う点はどこか？」
- 「精度行列の式を説明してほしい。特に Term2 と Term3 の違いは？」
- 「1/2 係数問題とは何か？原稿の式と Python 実装はどう違うか？」
- 「Scenario A/B/C のそれぞれで何を示しているか？」
- 「mismatch 実験から何が言えるか？特に Scen. C の Y-only 条件について」
- 「図 1(b) の 23.6 倍と原稿テキストの 41.5 倍の違いは何か？」
- 「BIC による潜在次元選択はどのシナリオで確実か？」
- 「先行研究の MATLAB 実装と本研究の Python 実装の相違点は？」

---

## 18. 未確認事項・今後の課題

| 項目 | 状況 | 優先度 |
|-----|------|------|
| E-step 中の \|\|Term2\|\| vs \|\|Term3\|\| の定量比較 | 未実施（追加実験要） | 高 |
| 1/2 修正版実装との RMSE 比較実験 | 未実施 | 中 |
| 図 1(b) の Word キャプションが「23.6 倍」か「41.5 倍」か | .docx はリポジトリ外 | 中 |
| MATLAB calcGrad の w 乗算欠如の影響 | 先行研究側の問題 | 低 |
| Scen. B で全 10 試行が k=3 を選ぶか（BIC 差 180 の安定性） | 試行別 BIC 未確認 | 低 |
