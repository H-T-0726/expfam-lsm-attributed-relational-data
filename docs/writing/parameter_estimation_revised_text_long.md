# 3.2 パラメータ推定（丁寧版・3ページ相当）

先生コメント「式もないし，結局何をしているのか伝わらない」への対応。
式を入れて，MCEM + ラプラス近似の流れが読者に伝わる版。
最終的な 2 ページ版は parameter_estimation_revised_text_short.md を使う。

---

## 3.2 パラメータ推定

提案モデルのパラメータ集合を

$$\boldsymbol{\theta} = \{\mathbf{F},\ \boldsymbol{\Sigma},\ w_0^Y,\ w^Y\}$$

とする（$\mathbf{F} \in \mathbb{R}^{d \times k}$：属性荷重行列，
$\boldsymbol{\Sigma} = \mathrm{diag}(\sigma_1^2,\ldots,\sigma_d^2)$：属性データ分散（$\mathrm{ExpFam}_X$=ガウスのみ），
$w_0^Y, w^Y \in \mathbb{R}$：関係データのスカラーパラメータ）．
潜在変数の事前分散 $\sigma_z^2$ は識別性のため $\sigma_z^2 = 1$ と固定する[1]．

---

### 3.2.1 EMアルゴリズムの枠組み

潜在変数 $\mathbf{Z}$ の周辺尤度

$$p(\mathbf{X}, \mathbf{Y} \mid \boldsymbol{\theta}) = \int p(\mathbf{X}, \mathbf{Y}, \mathbf{Z} \mid \boldsymbol{\theta})\, d\mathbf{Z}$$

は一般に解析的に計算できない．そこで，EM アルゴリズムに基づく近似推定を行う[1]．
EM アルゴリズムは，以下の Q 関数をパラメータ $\boldsymbol{\theta}$ について最大化することを繰り返す：

$$Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{\mathrm{old}}) = \mathbb{E}_{p(\mathbf{Z} \mid \mathbf{X}, \mathbf{Y}, \boldsymbol{\theta}^{\mathrm{old}})}\!\left[\log p(\mathbf{X}, \mathbf{Y}, \mathbf{Z} \mid \boldsymbol{\theta})\right] \tag{4}$$

ここで $p(\mathbf{Z} \mid \mathbf{X}, \mathbf{Y}, \boldsymbol{\theta}^{\mathrm{old}})$ は現在のパラメータ推定値 $\boldsymbol{\theta}^{\mathrm{old}}$ のもとでの Z の事後分布である．

---

### 3.2.2 Eステップ：ラプラス近似とモンテカルロサンプリング

式(4)の期待値を計算するためには，事後分布 $p(\mathbf{Z} \mid \mathbf{X}, \mathbf{Y}, \boldsymbol{\theta}^{\mathrm{old}})$
が必要である．各 $\mathbf{z}_i$ の条件付き事後分布を独立に近似する．

**ラプラス近似：**
各オブジェクト $i$ について，対数事後分布のモードを

$$\mathbf{m}_i = \arg\max_{\mathbf{z}_i} \log p(\mathbf{z}_i \mid \mathbf{X}, \mathbf{Y}, \mathbf{Z}_{-i}, \boldsymbol{\theta}^{\mathrm{old}}) \tag{5}$$

とし（ニュートン法で探索），事後分布をモード周りのガウス分布で近似する：

$$q_i(\mathbf{z}_i) = \mathcal{N}(\mathbf{m}_i,\ \mathbf{A}_i^{-1}) \tag{6}$$

ここで $\mathbf{A}_i$ は対数事後分布の負のヘッセ行列（精度行列）であり，
生成モデルの3つの項（Z事前分布・X尤度・Y尤度）に対応して：

$$\mathbf{A}_i = \underbrace{\mathbf{I}_k}_{\text{Z事前}}
+ \underbrace{\mathbf{F}^\top \mathbf{V}_X(\mathbf{m}_i) \mathbf{F}}_{\text{X尤度の寄与}}
+ \underbrace{\frac{(w^Y)^2}{2} \sum_{j \neq i} A_Y''(\eta_{ij}^Y)\, \mathbf{z}_j \mathbf{z}_j^\top}_{\text{Y尤度の寄与}} \tag{7}$$

ただし $\sigma_z^2 = 1$ を代入した．$\mathbf{V}_X(\mathbf{m}_i)$ は $\mathrm{ExpFam}_X$ の分布族に応じて定まる：

$$\mathbf{V}_X(\mathbf{m}_i) = \begin{cases}
\boldsymbol{\Sigma}^{-1} & (\mathrm{ExpFam}_X = \text{ガウス分布}) \\
\mathrm{diag}\!\left(A_X''(\mathbf{F} \mathbf{m}_i)\right) & (\mathrm{ExpFam}_X = \text{ベルヌーイ・ポアソン})
\end{cases} \tag{8}$$

ここで $A_X''(\cdot),\ A_Y''(\cdot)$ は各分布族の分散関数（log-partition function の2次導関数）であり，
ベルヌーイ分布では $A''(\eta) = \sigma(\eta)(1-\sigma(\eta))$，
ポアソン分布では $A''(\eta) = e^\eta$，
ガウス分布では $A''(\eta) = 1$ となる[2]．

**式(7)の意義：**
先行研究[1]では X = ガウス分布のみを扱うため，精度行列の第2項は常に $\mathbf{F}^\top \boldsymbol{\Sigma}^{-1} \mathbf{F}$ である．
提案手法では $\mathbf{V}_X$ が分布族に依存するため，
非ガウス属性データ（ベルヌーイ・ポアソン）に対しても精度行列を正しく計算できる．
これが本研究における E ステップの一般化の核心である．

**ニュートン法の更新式：**
モード探索に用いる対数事後分布の勾配は（先行研究[1] Eq.(23) の一般化）：

$$\nabla_{\mathbf{z}_i} \log p(\mathbf{z}_i \mid \cdot) = -\mathbf{z}_i
+ \mathbf{F}^\top \mathbf{U}_X^{-1}[T_X(\mathbf{x}_i) - A_X'(\mathbf{F}\mathbf{z}_i)]
+ \frac{w^Y}{2} \sum_{j \neq i} [T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)]\, \mathbf{z}_j \tag{9}$$

ここで $T_X(\cdot),\ T_Y(\cdot)$ は十分統計量（各分布族で $T(x) = x$），
$A_X'(\cdot),\ A_Y'(\cdot)$ は平均関数（ベルヌーイ: $\sigma(\eta)$，ポアソン: $e^\eta$，ガウス: $\eta$）である．

（Gaussian X では $\mathbf{F}^\top \mathbf{U}_X^{-1}[\cdots] = \mathbf{F}^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}_i - \mathbf{F}\mathbf{z}_i)$）

**モンテカルロサンプリング：**
式(6)の近似分布 $q_i(\mathbf{z}_i)$ から $L$ 個のサンプルを生成し，Q 関数を近似する：

$$\hat{Q}(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{\mathrm{old}}) = \frac{1}{L} \sum_{l=1}^L \log p(\mathbf{X}, \mathbf{Y}, \mathbf{Z}^{(l)} \mid \boldsymbol{\theta}) \tag{10}$$

ここで $\mathbf{Z}^{(l)} = [z_1^{(l)},\ldots,z_n^{(l)}]^\top$ であり，各 $z_i^{(l)} \sim q_i(z_i)$ である．
この手法をモンテカルロ EM（MCEM）+ラプラス近似と呼ぶ[1]．

---

### 3.2.3 Mステップ：パラメータの最大化

E ステップで得た $L$ 個のサンプル $\{\mathbf{Z}^{(l)}\}_{l=1}^L$ を用いて，
$\hat{Q}(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{\mathrm{old}})$ を $\boldsymbol{\theta}$ について最大化する（M ステップ）．

**荷重行列 F の更新：**

$\mathrm{ExpFam}_X$ = ガウス分布のとき，F の更新式は先行研究[1] Eq.(10) と同一の解析解となる：

$$\mathbf{F}^{\mathrm{new}} = \left(\sum_{l=1}^L \mathbf{X}^\top \mathbf{Z}^{(l)}\right)\!\left(\sum_{l=1}^L {\mathbf{Z}^{(l)}}^\top \mathbf{Z}^{(l)}\right)^{-1} \tag{11}$$

$\mathrm{ExpFam}_X$ = ベルヌーイまたはポアソンのとき，解析解が存在しないため，
以下の勾配を用いた Adam 勾配上昇法で F を更新する：

$$\nabla_{\mathbf{F}} \hat{Q}_X = \frac{1}{L} \sum_{l=1}^L \left[\mathbf{X} - A_X'(\mathbf{Z}^{(l)} \mathbf{F}^\top)\right]^\top \mathbf{Z}^{(l)} \tag{12}$$

**属性分散 Σ の更新（Gaussian X のみ）：**

$$\boldsymbol{\Sigma}^{\mathrm{new}} = \mathrm{diag}\!\left(\frac{1}{nL}\sum_{l=1}^L (\mathbf{X} - \mathbf{Z}^{(l)} \mathbf{F}^\top)^\top (\mathbf{X} - \mathbf{Z}^{(l)} \mathbf{F}^\top)\right) \tag{13}$$

**関係パラメータ $w_0^Y, w^Y$ の更新（先行研究[1] Eqs.(24)(25) の一般化）：**

$$\frac{\partial \hat{Q}}{\partial w_0^Y} = \frac{1}{2L} \sum_{l=1}^L \sum_{i \neq j}\!\left[T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)\right] \tag{14}$$

$$\frac{\partial \hat{Q}}{\partial w^Y} = \frac{1}{2L} \sum_{l=1}^L \sum_{i \neq j}\!\left[T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)\right] \mathbf{z}_i^\top \mathbf{z}_j \tag{15}$$

式(14)(15)を用いた Adam 勾配上昇法で $w_0^Y, w^Y$ を更新する．
先行研究[1]では $T_Y(y) = y$（ベルヌーイの十分統計量）であり，
式(14)(15)はそのときに文献[1] Eqs.(24)(25) と一致する．

---

### 3.2.4 潜在次元数の選択

潜在次元数 $k$ の選択には情報量基準 BIC を用いる[1]：

$$\mathrm{BIC}(k) = -2\hat{Q}_{\mathrm{strict}} + p_k \ln n \tag{16}$$

ここで $\hat{Q}_{\mathrm{strict}}$ はポアソン分布を含む場合に $-\sum \ln(y!)$ 補正を加えた厳密なQ値，
$p_k$ は自由パラメータ数（回転不定性を考慮して $p_k = kd - k(k-1)/2 + [\text{Gauss-X}:d] + [\text{Gauss-Y}:1]$）である．
$\hat{Q}(\cdot)$ を最小化する $k$ を選択する．

潜在変数行列 $\mathbf{Z}$ は直交変換に対して不定であるため，
推定精度の評価にはプロクルステス回転を適用した誤差指標 $\mathrm{RMSE}(\mathbf{Z})$ を用いる[1]．

---

## 節全体の字数（日本語文字数のみ概算）

| 小節 | 推定文字数 |
|------|----------|
| 3.2.1 EMの枠組み | 約150字 |
| 3.2.2 Eステップ | 約400字 |
| 3.2.3 Mステップ | 約350字 |
| 3.2.4 BIC次元選択 | 約120字 |
| **合計** | **約1020字** |

2ページ予稿には多すぎる（3節全体で1000字程度が目安）。短縮版は _short.md を参照。

---

## 先行研究[1]の対応式まとめ

| 提案手法の式番号 | 先行研究[1]の対応 | 違い |
|----------------|-----------------|------|
| 式(4) Q関数 | 先行研究[1] EM 目的関数 | 記法同一 |
| 式(7) 精度行列 | 先行研究[1] Eq.(22) | 第2項を V_X で一般化 |
| 式(9) 勾配 | 先行研究[1] Eq.(23) | 第2項を ExpFam_X に一般化 |
| 式(11) F 解析解 | 先行研究[1] Eq.(10) | Gaussian X で同一 |
| 式(14)(15) w 勾配 | 先行研究[1] Eqs.(24)(25) | T_Y, A_Y' で一般化 |

---

## 注意事項（先生に指摘されやすい点）

1. **「ラプラス近似」という言葉を使ってよいか？**
   → 使ってよい。実装が `calc_eta_newton`（ニュートン法でモードを探索）→
   `multivariate_normal(Z[i,:], A_i_inv)`（精度行列の逆行列から正規分布サンプリング）
   という正確なラプラス近似であることを実装で確認済み（model.py L.446-460）。

2. **「先行研究[1]のパラメータ推定法を一般化した」と言えるか？**
   → 言える。E ステップの精度行列（式(7)）と勾配（式(9)）の第2項が
   Gaussian X 専用から ExpFam_X 一般に拡張されていることを実装で確認済み。
   M ステップも非Gaussian X では Adam に切り替えることで一般化（式(12)）。

3. **式(9)の 1/2 係数の根拠は？**
   → 先行研究[1] Eq.(23) と実装（_calc_gradient term3 L.159）が同じ 1/2 を使用。
   これは Y 側の対数尤度を評価する際に使う慣行（1/2 Σ_{i≠j} として実装）。
   先生に聞かれた場合は「先行研究[1]と同一の慣行に従った」と答えてよい。
