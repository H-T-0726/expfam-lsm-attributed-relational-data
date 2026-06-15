# 精度行列の 1/2 係数：数学的整理

調査日：2026-05-08

---

## 結論

**先生の指摘は数学的に正しい。**  
`sum_{j≠i}` を使う場合，精度行列の Y 側の項に 1/2 は**不要**である。

---

## 1. 二つの等価な記法

Y が対称行列（y_ij = y_ji）の場合，以下の二つは**完全に等価**：

**記法 A：** Q 関数の Y 側を対称和として書く

$$\ln p(Y \mid Z) = \frac{1}{2} \sum_{i \neq j} \ln p_Y(y_{ij} \mid \eta_{ij}^Y)$$

**記法 B：** Q 関数の Y 側を片側和として書く

$$\ln p(Y \mid Z) = \sum_{i < j} \ln p_Y(y_{ij} \mid \eta_{ij}^Y)$$

対称性（y_ij = y_ji，η_ij^Y = η_ji^Y）により，記法 A = 記法 B（値が等しい）。

---

## 2. z_i の条件付き事後分布への寄与

**ラプラス近似の核心：** z_i を更新するとき，z_{-i} を固定して  

$$\ln p(z_i \mid \mathbf{X}, \mathbf{Y}, \mathbf{Z}_{-i}, \boldsymbol{\theta})$$

の Y 側寄与を求める。

### 記法 A を使った場合

$$\frac{1}{2} \sum_{a \neq b} \ln p_Y(y_{ab} \mid \eta_{ab}^Y)$$

のうち z_i に依存する項（z_{-i} 固定）：

- a=i の項：$\frac{1}{2} \sum_{j \neq i} \ln p_Y(y_{ij} \mid w_0^Y + w^Y \mathbf{z}_i^\top \mathbf{z}_j)$
- b=i の項：$\frac{1}{2} \sum_{j \neq i} \ln p_Y(y_{ji} \mid w_0^Y + w^Y \mathbf{z}_j^\top \mathbf{z}_i)$  
  $= \frac{1}{2} \sum_{j \neq i} \ln p_Y(y_{ij} \mid w_0^Y + w^Y \mathbf{z}_i^\top \mathbf{z}_j)$（対称性より）

合計：
$$\frac{1}{2}\sum_{j\neq i}(\cdots) + \frac{1}{2}\sum_{j\neq i}(\cdots) = \sum_{j \neq i} \ln p_Y(y_{ij} \mid \eta_{ij}^Y) \quad \leftarrow \mathbf{1/2 \text{ が消える}}$$

### 記法 B を使った場合

$$\sum_{a < b} \ln p_Y(y_{ab} \mid \eta_{ab}^Y)$$

のうち z_i に依存する項：

- a=i の項：$\sum_{j > i} \ln p_Y(y_{ij} \mid \eta_{ij}^Y)$
- b=i の項：$\sum_{j < i} \ln p_Y(y_{ji} \mid \eta_{ji}^Y) = \sum_{j < i} \ln p_Y(y_{ij} \mid \eta_{ij}^Y)$

合計：
$$\sum_{j > i}(\cdots) + \sum_{j < i}(\cdots) = \sum_{j \neq i} \ln p_Y(y_{ij} \mid \eta_{ij}^Y) \quad \leftarrow \mathbf{1/2 \text{ なし}}$$

**どちらの記法を使っても，z_i の条件付き事後分布への Y 側寄与は Σ_{j≠i}（1/2 なし）になる。**

---

## 3. 精度行列（ヘッセ行列）の導出

z_i に関する Y 側の項：$\sum_{j \neq i} \ln p_Y(y_{ij} \mid w_0^Y + w^Y \mathbf{z}_i^\top \mathbf{z}_j)$

2 階微分（ヘッセ行列）：

$$\frac{\partial^2}{\partial \mathbf{z}_i^2} \left[ \sum_{j \neq i} \ln p_Y(y_{ij} \mid \eta_{ij}^Y) \right] = -\sum_{j \neq i} A_Y''(\eta_{ij}^Y) \cdot (w^Y)^2 \cdot \mathbf{z}_j \mathbf{z}_j^\top$$

精度行列 A_i = 負のヘッセ行列：

$$\boxed{\mathbf{A}_i = \frac{1}{\sigma_z^2}\mathbf{I}_k + \mathbf{F}^\top \mathbf{V}_X(\mathbf{m}_i)\mathbf{F} + (w^Y)^2 \sum_{j \neq i} A_Y''(\eta_{ij}^Y)\, \mathbf{z}_j \mathbf{z}_j^\top}$$

**1/2 なし。これが数学的に正しい式。**

---

## 4. 勾配（ニュートン法）の導出

$$\frac{\partial}{\partial \mathbf{z}_i} \left[ \sum_{j \neq i} \ln p_Y(y_{ij} \mid \eta_{ij}^Y) \right] = w^Y \sum_{j \neq i} \left[T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)\right] \mathbf{z}_j$$

**1/2 なし，w^Y の乗算あり。**

---

## 5. M ステップ勾配の 1/2 は正しい

Q 関数を最大化する M ステップの勾配（w_0^Y について）：

$$\frac{\partial \hat{Q}}{\partial w_0^Y} = \frac{1}{L}\sum_{l=1}^L \sum_{i<j} \left[T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)\right]$$

計算上は `(Y - S)` の全要素 (i≠j) を合計して /2 で割る：

$$= \frac{1}{2L} \sum_{l=1}^L \sum_{i \neq j} \left[T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)\right]$$

**M ステップの /2L は正しい。** 対称行列の全和を片側和に換算するためのもの。

---

## 6. 現在の Python 実装の 1/2 がなぜ問題か

Python 精度行列：`term3 = 0.5 * w^2 * Σ_{j≠i}(...)` → `A_i = (1/2) * A_i_correct`  
Python 勾配：`term3 = 0.5 * w * Σ_{j≠i}(...)` → `grad = (1/2) * grad_correct`

**ニュートン更新方向への影響：**

$$\mathbf{A}_i^{-1} \cdot \nabla = \left(\frac{1}{2}\mathbf{A}_{i,\text{correct}}\right)^{-1} \cdot \frac{1}{2}\nabla_{\text{correct}} = 2\mathbf{A}_{i,\text{correct}}^{-1} \cdot \frac{1}{2}\nabla_{\text{correct}} = \mathbf{A}_{i,\text{correct}}^{-1} \cdot \nabla_{\text{correct}}$$

→ **ニュートン更新の方向は正しい**（1/2 が打ち消しあう）。

**サンプリング分散への影響：**

$$\text{共分散行列} = \mathbf{A}_i^{-1} = \frac{1}{2}\mathbf{A}_{i,\text{correct}} \bigg)^{-1} = 2 \cdot \mathbf{A}_{i,\text{correct}}^{-1}$$

→ **サンプリング分散が 2 倍に膨らむ**（MCEM のサンプルにより多くのノイズが乗る）。

実験結果への実質的影響：L=10 サンプルで平均されるため，定性的な結論は変わらないが，  
量的には収束が遅くなる可能性がある。

---

## 7. 各記法で統一した場合の式の対応

### 原稿で使うべき式（案A: 1/2 を除去）

$$\mathbf{A}_i = \mathbf{I}_k + \mathbf{F}^\top \mathbf{V}_X(\mathbf{m}_i)\mathbf{F} + (w^Y)^2 \sum_{j \neq i} A_Y''(\eta_{ij}^Y)\, \mathbf{z}_j \mathbf{z}_j^\top$$

対応する M ステップ勾配（変更なし）：

$$\frac{\partial \hat{Q}}{\partial w_0^Y} = \frac{1}{L}\sum_{l=1}^L \sum_{i < j} \left[T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)\right]$$

または等価な対称和表現（コードで使いやすい形）：

$$= \frac{1}{2L}\sum_{l=1}^L \sum_{i \neq j} \left[T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)\right]$$

### 現在の Python 実装（1/2 あり）との関係

原稿の式（1/2 なし）と Python 実装（1/2 あり）は，**ニュートン更新方向は同じ** だが，  
論文の式としては正確でない。論文を先生に説明する際は 1/2 を除く方が正確。

---

## 8. 原稿の推奨

**推奨：案 A（1/2 を除去する）**

根拠：
1. 数学的に正しい
2. MATLAB 原典実装（calcAi，L.56-63）と一致する
3. 先生の指摘に正面から応答できる

式：
$$\mathbf{A}_i = \mathbf{I}_k + \mathbf{F}^\top \mathbf{V}_X(\mathbf{m}_i)\mathbf{F} + (w^Y)^2 \sum_{j \neq i} A_Y''(\eta_{ij}^Y)\, \mathbf{z}_j \mathbf{z}_j^\top$$

本文に追加すべき注記（もし先行研究との整合を書く場合）：
> 「Y 側の対数尤度は先行研究[1]と同様に対称性を考慮して $\frac{1}{2}\sum_{i\neq j}$ として定義する（式(A5)）が，$z_i$ の条件付き事後分布はこの対称和の両側からの寄与を合計するため，精度行列には $\sum_{j\neq i}$ のみが現れ，1/2 は不要となる．」
