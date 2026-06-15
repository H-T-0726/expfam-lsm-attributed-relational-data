# 原稿の精度行列の修正方針

作成日：2026-05-08  
結論：**1/2 を除去する（案 A）**

---

## 最終推奨：案 A（1/2 を除去）

### 根拠

| 観点 | 1/2 あり | 1/2 なし |
|------|---------|---------|
| 数学的導出 | × | ✓ |
| MATLAB calcAi | × | ✓ |
| 先生の直感 | × | ✓ |
| Python 実装との整合 | ✓ | × |
| 先行研究 Eq.(22) との整合 | ? (PDF 未確認) | ? |

数学的正しさ・MATLAB 原典・先生の指摘が全て「1/2 なし」を支持する。  
Python 実装との不整合は残るが，ニュートン方向への影響はなく，  
論文としての正確性を優先すべき。

---

## 変更箇所

### (1) conference_submission_final_draft.md の式 (6)

**変更前：**

$$\mathbf{A}_i = \mathbf{I}_k + \mathbf{F}^\top \mathbf{V}_X(\mathbf{m}_i)\,\mathbf{F} + \frac{(w^Y)^2}{2} \sum_{j \neq i} A_Y''(\eta_{ij}^Y)\, \mathbf{z}_j \mathbf{z}_j^\top \tag{6}$$

**変更後：**

$$\mathbf{A}_i = \mathbf{I}_k + \mathbf{F}^\top \mathbf{V}_X(\mathbf{m}_i)\,\mathbf{F} + (w^Y)^2 \sum_{j \neq i} A_Y''(\eta_{ij}^Y)\, \mathbf{z}_j \mathbf{z}_j^\top \tag{6}$$

### (2) 式 (6) 直後の本文（現在の注記を変更）

**変更前：**
「なお，第3項の $1/2$ は先行研究[1]と同一の慣行に従う．」

**変更後（この文を削除し，以下に置き換え）：**
不要になるため削除する。または必要に応じて：
「先行研究[1]の精度行列（Gaussian-X 専用）における $\mathbf{F}^\top\boldsymbol{\Sigma}^{-1}\mathbf{F}$ を，$\mathbf{F}^\top\mathbf{V}_X(\mathbf{m}_i)\mathbf{F}$ に一般化した点が本研究の貢献である．」

### (3) parameter_estimation_corrected_formulas.md の式 (A7)

**変更前：**

$$\mathbf{A}_i = \underbrace{\frac{1}{\sigma_z^2} \mathbf{I}_k}_{\text{Z事前}}
+ \underbrace{\mathbf{F}^\top \mathbf{V}_X \mathbf{F}}_{\text{X尤度寄与}}
+ \underbrace{\frac{(w^Y)^2}{2} \sum_{j \neq i} A_Y''\!\left(\eta_{ij}^Y\right) \mathbf{z}_j \mathbf{z}_j^\top}_{\text{Y尤度寄与}}$$

**変更後：**

$$\mathbf{A}_i = \underbrace{\frac{1}{\sigma_z^2} \mathbf{I}_k}_{\text{Z事前}}
+ \underbrace{\mathbf{F}^\top \mathbf{V}_X \mathbf{F}}_{\text{X尤度寄与}}
+ \underbrace{(w^Y)^2 \sum_{j \neq i} A_Y''\!\left(\eta_{ij}^Y\right) \mathbf{z}_j \mathbf{z}_j^\top}_{\text{Y尤度寄与}}$$

### (4) parameter_estimation_revised_text_short.md の各バージョン

同様に `(w^Y)^2 / 2` を `(w^Y)^2` に変更する。

---

## 案 B（1/2 を残す場合）のリスク

先生の指摘が正しいのに 1/2 を維持しようとした場合：

### 残せる説明

「先行研究[1]の Eq.(22) と同一の記法に従う」と書けばよいように見えるが…

### 問題点

1. **数学的に不正確**：z_i の条件付き事後分布の Hessian は Σ_{j≠i} であり，1/2 は余分
2. **MATLAB 原典と不一致**：calcAi は 1/2 なし
3. **先生を説得できない**：「先行研究にある」は「なぜあるのか」への回答にならない
4. **若干の説明増加**：「(1/2)Σ_{i≠j} の慣行を採用しているため」という補足が必要になるが，その補足自体が数学的に正しくない

**案 B は推奨しない。**

---

## 案 C（精度行列を省略する）のリスク

先生の「何をしているか伝わらない」というコメントへの対応で精度行列を省略すると，  
今度は「1/2 問題を回避するために式を省いた」と受け取られる可能性がある。  
精度行列を入れる方針はそのまま維持し，式を正しく修正する方が良い。

---

## Python 実装との不整合について

原稿を修正（1/2 除去）しても，Python 実装（1/2 あり）は変わらない。

この不整合の実害：
- ニュートン方向：影響なし（勾配と精度行列の両方に 1/2 があるので打ち消しあう）
- サンプリング：共分散が 2 倍（より広いサンプリング），収束に若干の影響あり
- 実験結果：定性的結論は変わらない（1/2 は方向に影響しない）

**2 ページ予稿の範囲では実装修正を行わず，論文の式だけ正しくすることで十分。**

修論や次の発表では Python 実装の 1/2 も修正すると，理論とコードが完全に一致する。

---

## 修正後の最終的な精度行列の式（確定版）

$$\mathbf{A}_i = \mathbf{I}_k + \mathbf{F}^\top \mathbf{V}_X(\mathbf{m}_i)\,\mathbf{F} + (w^Y)^2 \sum_{j \neq i} A_Y''(\eta_{ij}^Y)\, \mathbf{z}_j \mathbf{z}_j^\top$$

$$\mathbf{V}_X(\mathbf{m}_i) = \begin{cases}
\boldsymbol{\Sigma}^{-1} & (\mathrm{ExpFam}_X = \text{ガウス分布，先行研究[1]と同一形}) \\
\mathrm{diag}(A_X''(\mathbf{F}\mathbf{m}_i)) & (\mathrm{ExpFam}_X = \text{ベルヌーイ・ポアソン，本研究の一般化})
\end{cases}$$

---

## 先行研究 [1] との整合（補足）

MATLAB 原典実装（calcAi，calcEtaNewton.m L.56）：1/2 なし ✓  
先行研究 Python 再現コード（model.py L.353）：1/2 あり（PDF 未確認につき論文の内容は不確定）  
本稿の推奨：1/2 なし（数学的正解・MATLAB 一致）

もし先行研究論文に 1/2 が明記されているなら，そちらは記法上の誤りの可能性があることを  
修論等の場で先生と確認する価値がある。
