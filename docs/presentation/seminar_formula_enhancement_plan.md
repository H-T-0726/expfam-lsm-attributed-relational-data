# 数式強化版 差分計画書

**作成日：** 2026-05-19  
**目的：** ゼミ発表用 Notion 資料（seminar_notion_full.md）の数式を強化するための差分案  
**注意：** Notion ページはまだ更新しない。本書は差分案・確認書のみ。

---

## 1. 現状の数式カバレッジ整理

現在の `seminar_notion_full.md` での数式状況：

| 章 | 現状 | 評価 |
|---|------|------|
| §2 先行研究 | 生成モデル式（z_i, x_il, y_ij）あり | △ 同時分布・Q関数なし |
| §5 提案手法概要 | 比較表・図のみ。数式ほぼなし | ✗ θのパラメータ集合なし |
| §6 指数型分布族 | 一般形・A'/A'' 表あり | △ A'(η) の意味付けは十分 |
| §7 提案モデルの数式 | 生成モデル式・精度行列 Eq.(6) あり | △ 同時分布・M-step更新式なし（Q関数は§8本文へ） |
| §8 推定アルゴリズム | フロー図のみ。数式なし | ✗ BIC式なし（Q関数を本文に追加推奨） |
| §9 実装との対応 | ファイル一覧のみ | △ BIC num_params 式なし |
| §11 実験設定 | 設定値の表のみ | △ BIC式があるとよい |
| §12 実験結果 | 数値表のみ | ○（数値資料として十分） |

---

## 2. 追加すべき数式の整理表

| 追加先 | 数式 | 本文/トグル | 目的 | 根拠ファイル | 注意点 |
|-------|------|:----------:|------|------------|-------|
| §5 | $\boldsymbol{\theta} = \{F,\, w_0^Y,\, w^Y\}$（Gauss-X のとき $+\Sigma$、Gauss-Y のとき $+\sigma_y^2$） | 本文 | 推定対象パラメータを明示 | `conference_submission_final_draft.md` §3.3、`CLAUDE.md`（生成モデル節） | w0/w がスカラーであることを強調 |
| §6 | 指数型分布族の一般形：$p(x \mid \eta) = h(x)\exp\{\eta\, T(x) - A(\eta)\}$ | 本文（すでにあり） | — | `conference_submission_final_draft.md` §2.3 | 現状記載済み。確認のみ |
| §6 | Gaussian-X での V_X：$V_X = \Sigma^{-1}$（対角） | 本文（表で） | 精度行列との接続を明示 | `conference_submission_final_draft.md` Eq.(6)、`model_dual_expfam.py` L.186-190 | 非Gauss-Xでは $\mathrm{diag}(A_X''(Fm_i))$ |
| §7 | 同時分布：$p(X,Y,Z\mid\theta) = \prod_i p(z_i)\cdot\prod_{i,l}p_X(x_{il}\mid\eta_{il}^X)\cdot\prod_{i<j}p_Y(y_{ij}\mid\eta_{ij}^Y)$ | トグル | 尤度の構造を明示 | `NOTEBOOKLM_RESEARCH_BRIEF.md` §7.5 | Y尤度が **片側和** ($i<j$) であることを注記 |
| **§8** | Q関数（MCEM基本形）：$\hat{Q}(\boldsymbol{\theta}\mid\boldsymbol{\theta}^{\mathrm{old}}) \simeq \frac{1}{L}\sum_{l=1}^L \log p(X,Y,Z^{(l)}\mid\boldsymbol{\theta}), \; Z^{(l)}\sim q(Z)$ | **本文** | MCEMの目的関数を本文で明示（推定アルゴリズムの説明の核） | `utils_expfam.py` L.68-77（`calc_Q_no_fact`）、`NOTEBOOKLM_RESEARCH_BRIEF.md` §7.7 | Poisson の factorial 補正・BIC用 $\hat{Q}_\mathrm{strict}$ は §8 のトグルに分ける |
| **§8** (トグル) | Q_strict（BIC用）：$\hat{Q}_\mathrm{strict} = \hat{Q} - \sum_{i<j}\ln(y_{ij}!)$（Poisson-Y）または $- \sum_{il}\ln(x_{il}!)$（Poisson-X） | トグル | BIC計算のための factorial 補正を説明 | `utils_expfam.py` L.355-378（`calc_Q_dual_strict`） | Bernoulli/Gaussian では補正不要。比較のため一貫して適用 |
| §7 | E-step 勾配（原稿採用式・V_X を勾配側で一律使用しない形）：$\nabla_{z_i}\ln p(z_i\mid\cdot) = -z_i + F^\top\{T_X(x_i)-A_X'(Fz_i)\} + w^Y\sum_{j\neq i}\{T_Y(y_{ij})-A_Y'(\eta_{ij}^Y)\}z_j$ | トグル（すでにあり） | 既存。注意書きを修正 | `NOTEBOOKLM_RESEARCH_BRIEF.md` §7.8、`model_dual_expfam.py` L.123-161 | **Gaussian-X では分散パラメータによる重み付けが入る。精度行列側の $V_X(m_i)$ とは役割が異なるため、勾配式では $V_X$ を一律に書かない。** |
| §7 | M-step 実装上の勾配（w0）：実装では `grad_sum / (2.0 * L * phi)` の形で計算。/2L は $\sum_{i\neq j}$ を $\sum_{i<j}$ に換算するため正しい | トグル | /2L が正しい根拠を示す。符号規約は実装に依存 | `model_expfam.py` L.149-178（`calc_w0`）、L.168 | /2L の意味を説明。Q関数最大化かどうかは実装符号に依存するため断定しない |
| §7 | M-step 実装上の勾配（w）：実装では `grad_sum / (2.0 * L * phi)` の形で計算。内積 $z_i^\top z_j$ を乗じた和として対称性を利用 | トグル | w更新の対称性と/2Lを示す。符号規約は実装に依存 | `model_expfam.py` L.180-210（`calc_w`）、L.200 | φ = $\sigma_y^2$（Gauss-Y のみ）。他は φ=1 |
| §8 | BIC 式：$\mathrm{BIC} = -2\hat{Q}_\mathrm{strict} + p \ln n$、ここで $p = kd - \tfrac{k(k-1)}{2} + [\text{d if Gauss-X}] + [\text{1 if Gauss-Y}]$ | **本文** | BICの計算根拠を示す | `utils_expfam.py` L.386-404（`calc_bic_dual`）、`NOTEBOOKLM_RESEARCH_BRIEF.md` §10.7 | w0/w はパラメータ数に含まれない（NOLTA 2024 慣行と一致） |
| §8 | σ_y² 更新（Gauss-Y のとき）：$\hat{\sigma}_y^2 = \frac{1}{L}\sum_l \frac{1}{n(n-1)/2}\sum_{i<j}(y_{ij}-\eta_{ij}^Y)^2$ | トグル | Gauss-Y の M-step 解析解 | `model_expfam.py` L.212-（`calc_sigma_y`） | 非Gauss-Y では推定しない |
| §11 | BIC パラメータ数の内訳表：Scenario A(Pois-X,Bern-Y) / B(Gauss-X,Pois-Y) / C(Bern-X,Gauss-Y) | トグル | k=3のときの num_params を具体化 | `utils_expfam.py` L.399-403 | n=150,d=15,k=3 のとき: A: 45-3+0+0=42 / B: 45-3+15+0=57 / C: 45-3+0+1=43 |

---

## 3. 根拠確認済みの主要数式

以下はすべて根拠ファイルから直接確認した式。

### 3.1 先行研究の生成モデル（確認済み）

根拠：`conference_submission_final_draft.md` §2.2 Eq.(1)(2)

$$z_i \sim \mathcal{N}(0, I_k)$$

$$x_{il} \mid z_i \sim \mathcal{N}(f_l^\top z_i,\ \sigma_l^2) \qquad \text{（バイアスなし）}$$

$$y_{ij} \mid z_i, z_j \sim \mathrm{Bernoulli}\!\left(\sigma(w_0 + w\, z_i^\top z_j)\right) \qquad (i < j)$$

### 3.2 提案手法の生成モデル（確認済み）

根拠：`conference_submission_final_draft.md` §3.1 Eq.(4)、§3.2 Eq.(5)

$$x_{il} \mid z_i \sim \mathrm{ExpFam}_X(\eta_{il}^X), \qquad \eta_{il}^X = f_l^\top z_i$$

$$y_{ij} \mid z_i, z_j \sim \mathrm{ExpFam}_Y(\eta_{ij}^Y), \qquad \eta_{ij}^Y = w_0^Y + w^Y z_i^\top z_j \qquad (i < j)$$

$w_0^Y, w^Y \in \mathbb{R}$ はスカラー（行列ではない）。

### 3.3 推定パラメータ集合（確認済み）

根拠：`conference_submission_final_draft.md` §3.3、`CLAUDE.md`（生成モデル節）

$$\boldsymbol{\theta} = \{F,\, w_0^Y,\, w^Y\}$$

Gaussian-X のときのみ対角分散行列 $\Sigma$ を追加で推定。  
Gaussian-Y のときのみ $\sigma_y^2$ を追加で推定。

### 3.4 指数型分布族の一般形（確認済み）

根拠：`conference_submission_final_draft.md` §2.3 Eq.(3)

$$p(x \mid \eta) = h(x)\exp\!\left\{\eta\, T(x) - A(\eta)\right\}$$

### 3.5 精度行列 Eq.(6)（確認済み）

根拠：`conference_submission_final_draft.md` §3.3 Eq.(6)

$$A_i = \underbrace{I_k}_{\text{Term1}} + \underbrace{F^\top V_X(m_i) F}_{\text{Term2}} + \underbrace{(w^Y)^2 \sum_{j \neq i} A_Y''(\eta_{ij}^Y)\, z_j z_j^\top}_{\text{Term3（**1/2 なし**、原稿採用式）}}$$

$$V_X = \begin{cases} \Sigma^{-1} & (\mathrm{ExpFam}_X = \text{Gaussian}) \\ \mathrm{diag}(A_X''(Fm_i)) & (\text{Bernoulli / Poisson-X}) \end{cases}$$

### 3.6 Q 関数（MCEM近似）（確認済み）

根拠：`utils_expfam.py` L.68-77（`calc_Q_no_fact`）、`NOTEBOOKLM_RESEARCH_BRIEF.md` §7.7

**§8 本文に出す式：**

$$\hat{Q}(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{\mathrm{old}}) \simeq \frac{1}{L}\sum_{l=1}^L \log p\!\left(X, Y, Z^{(l)} \mid \boldsymbol{\theta}\right), \qquad Z^{(l)} \sim q(Z)$$

**§8 トグルに入れる補足（BIC用）：**

BIC 計算のために $\hat{Q}_\mathrm{strict}$ を使う：Poisson-Y のとき $-\sum_{i<j}\ln(y_{ij}!)$、Poisson-X のとき $-\sum_{il}\ln(x_{il}!)$ を $\hat{Q}$ に加算（`utils_expfam.py` L.355-378）。Bernoulli/Gaussian には補正不要。

### 3.7 BIC（確認済み）

根拠：`utils_expfam.py` L.386-404（`calc_bic_dual`）

$$\mathrm{BIC} = -2\hat{Q}_\mathrm{strict} + p \ln n$$

$$p = \underbrace{kd - \frac{k(k-1)}{2}}_{\text{F の自由パラメータ数（回転拘束後）}} + \underbrace{[d \text{ if Gauss-X}]}_{\Sigma \text{ の対角成分数}} + \underbrace{[1 \text{ if Gauss-Y}]}_{\sigma_y^2}$$

w0/w はパラメータ数から除外（NOLTA 2024 Eq.(26) 慣行と一致）。

### 3.8 M-step w0・w 更新（実装確認済み）

根拠：`model_expfam.py` L.168, L.200（`calc_w0`・`calc_w`）

Python 実装では以下の形で勾配が計算され、Adam に渡される：

```python
# calc_w0 L.168
grad = -grad_sum / (2.0 * L * phi)

# calc_w  L.200
grad = -grad_sum / (2.0 * L * phi)
```

ここで `grad_sum` は $\sum_{l=1}^L\sum_{i\neq j}[T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)]$（w0）または同 $\times z_i^\top z_j$（w）。  
$\phi = \sigma_y^2$（Gaussian-Y のとき）、$\phi = 1$（Bernoulli / Poisson-Y のとき）。

**/ 2L の意味：** $\sum_{i\neq j} / (2L) = \sum_{i<j} / L$ に等価。Q 関数の Y 尤度を $\frac{1}{2}\sum_{i\neq j}$ で定義した場合の自然な換算であり、正しい。

**符号規約について：** 実装上の `grad` の符号はQ関数最大化に向いているかどうかは、Adam の更新式（`w = w - alpha * m_hat / ...`）の符号と合わせて判断する必要がある。Notion 資料では「実装上の /2L は正しい」の一点に絞り、勾配の向きは断定しない。

### 3.9 σ_y² 更新（Gauss-Y のとき）（確認済み）

根拠：`model_expfam.py` L.212-（`calc_sigma_y` の docstring）

$$\hat{\sigma}_y^2 = \frac{1}{L}\sum_{l=1}^L \frac{1}{n(n-1)/2}\sum_{i<j}\!\left(y_{ij} - \eta_{ij}^Y\right)^2$$

---

## 4. 1/2 係数問題の整理（追加時に使う表現）

### 使うべき表現

> NOLTA 2024 PDF Eq.(22)(23) には E-step 精度行列・勾配の Y 側 Term3 に 1/2 が含まれているが、本研究の再導出および MATLAB 実装（`calcEtaNewton.m` の `calcAi`・`calcGrad` 関数）とは不一致であり、本研究では **1/2 なしの式を原稿採用式として整理する**。

### 各資料での状況（根拠確認済み）

| 資料 | 精度行列 Term3 | E-step 勾配 Term3 | 根拠 |
|-----|:---:|:---:|------|
| 原稿 Eq.(6) | **1/2 なし** ✓ | — | `conference_submission_final_draft.md` |
| MATLAB `calcAi`（L.56-63） | **1/2 なし** ✓ | — | `half_factor_literature_code_check.md` §2-A |
| MATLAB `calcGrad`（L.43-49） | — | **1/2 なし** ✓ | `half_factor_literature_code_check.md` §2-B |
| NOLTA 2024 PDF Eq.(22)(23) | 1/2 あり（監査資料では1/2ありとして整理。本資料単体ではPDF本文を直接再確認していない） | 1/2 あり | `01_formula_code_audit.md`・`NOTEBOOKLM_RESEARCH_BRIEF.md`（過去の監査資料での整理） |
| Python `model_dual_expfam.py` L.200 | **0.5 あり** ✗ | — | `model_dual_expfam.py` L.200 |
| Python `model_dual_expfam.py` L.159 | — | **0.5 あり** ✗ | `model_dual_expfam.py` L.159 |

**NOLTA 2024 PDF Eq.(22)(23) の 1/2 について：** これまでの監査資料（`01_formula_code_audit.md`・`NOTEBOOKLM_RESEARCH_BRIEF.md`）では 1/2 あり として整理されている。ただし、本資料単体では PDF 本文を直接再確認していないため、発表資料では「本研究の再導出および MATLAB 実装とは不一致」と慎重に述べる。

### M-step と Q 関数の 1/2 は正しい（誤解防止）

| 箇所 | 1/2 の有無 | 正しいか | 根拠 |
|-----|:---:|:---:|------|
| Q 関数 Y 側の `0.5 * sum(ln_p)` | あり | **✓ 正しい** | `model_expfam.py` L.267 |
| M-step の `/2L`（w0/w 更新） | あり | **✓ 正しい** | `model_expfam.py` L.168, 200 |
| MATLAB `calcp_Y` の `sum/2` | あり | **✓ 正しい** | `half_factor_literature_code_check.md` §2-D |
| E-step 精度行列 Term3（Python L.200） | あり | **✗ 不整合** | `model_dual_expfam.py` L.200 |
| E-step 勾配 Term3（Python L.159） | あり | **✗ 不整合** | `model_dual_expfam.py` L.159 |

---

## 5. 未確認の式（推測で補完しない）

以下は現時点で根拠ファイルから直接確認できなかった式：

| 式 | 状況 | 今後の確認方法 |
|---|------|-------------|
| F の解析解（Gaussian-X のとき） | コードで「parent クラスへ委譲」とあるが式自体は未確認 | `reproduction/src/model.py` の `calc_F` を読む |
| Procrustes 回転の最適化式 | 「直交行列 R を最適化」とのみ記載 | `utils_expfam.py` L.38-43 の SVD 部分から確認可能 |
| NOLTA 2024 PDF Eq.(22)(23) の 1/2 | 監査資料では1/2ありとして整理済み。本資料単体ではPDF本文を直接再確認していない。発表資料では「MATLAB実装・本研究の再導出とは不一致」と慎重に述べる | 過去の監査資料を参照。必要なら PDF ビューワで再確認 |

---

## 6. 本文に出す式 vs トグルに入れる式

### 本文に出す式（Q関数配置統一後の確定版）

| 章 | 式 | 状態 |
|---|---|------|
| §2 | 先行研究の生成モデル（z_i, x_il, y_ij） | ✅ 現状記載済み |
| §5 | $\boldsymbol{\theta} = \{F, w_0^Y, w^Y\}$（Gauss-X なら +Σ、Gauss-Y なら +σ_y²） | ⬜ **追加推奨** |
| §6 | 指数型分布族の一般形 $p(x\mid\eta) = h(x)\exp\{\eta T(x) - A(\eta)\}$ | ✅ 現状記載済み |
| §7 | 提案手法の生成モデル（eq4・eq5） | ✅ 現状記載済み |
| §7 | 精度行列 $A_i = I_k + F^\top V_X(m_i)F + (w^Y)^2\sum_{j\neq i}A_Y''\,z_jz_j^\top$ | ✅ 現状記載済み |
| **§8** | **MCEM Q 関数（基本形）：$\hat{Q}(\theta\mid\theta^{\mathrm{old}}) \simeq \frac{1}{L}\sum_{l=1}^L\log p(X,Y,Z^{(l)}\mid\theta)$** | ⬜ **追加推奨**（§7 ではなく §8 本文） |

### トグルに入れる式（Q関数配置統一後の確定版）

| 章 | 式 | 状態 |
|---|---|------|
| §7 | 同時分布 $p(X,Y,Z\mid\theta)$（Y尤度は片側和 i<j） | ⬜ 追加推奨 |
| §7 | E-step 勾配（Term1/2/3、V_X を一律使用しない形・注意書き修正） | ✅ 現状トグルにあり（注意書きを修正） |
| §7 | M-step w0/w 更新（/2L の意味・符号規約は実装依存と注記） | ⬜ 追加推奨 |
| §7 | 1/2係数問題の整理表（MATLAB/原稿/Python の比較） | ✅ 現状トグルにあり |
| **§8** | **Q_strict（BIC用）：$\hat{Q}_\mathrm{strict} = \hat{Q} - \sum_{i<j}\ln(y_{ij}!)$（Poisson-Y）等** | ⬜ **追加推奨**（§8 トグル） |
| §8 | BIC 自由パラメータ数の内訳（Scenario A/B/C 別の具体値） | ⬜ 追加推奨 |
| §8 | σ_y² 更新式（Gauss-Y のときのみ） | ⬜ 追加推奨 |

---

## 7. 次フェーズ案（修正後）

### Phase A（今すぐ可能）：ChatGPT に差分案を確認

本書の §2（追加すべき数式の整理表）・§3（根拠確認済みの式）・§6（本文/トグル区分）を ChatGPT に貼り、以下を確認する：
- E-step 勾配の V_X を外した書き方は数学的に正しいか
- M-step w0/w の /2L の説明で十分か。符号規約の注記は適切か
- Q 関数を本文に出す判断は妥当か
- 1/2 係数問題の表現は慎重に書けているか
- 追加すべき式の漏れはないか

### Phase B（ChatGPT 確認後）：数式強化版 Markdown を作成

**`docs/presentation/seminar_notion_full_formula_rich.md`** として新規作成する。  
**既存の `seminar_notion_full.md` は上書きしない。**

差分の適用方針（Q関数配置統一後）：
- §5 に $\boldsymbol{\theta} = \{F, w_0^Y, w^Y\}$ の一文を追加（本文）
- §7 の同時分布トグルを新規追加（Q関数は §7 ではなく §8 へ）
- §7 の E-step 勾配トグルの注意書きを修正（V_X を勾配側で一律使用しない旨）
- §7 の M-step w0/w トグルを新規追加（/2L の説明・符号規約は実装依存と注記）
- **§8 の本文に Q 関数（基本形）を追加**（推定アルゴリズムの説明の核として）
- §8 の本文に BIC 式を追加
- §8 のトグルに Q_strict・BIC num_params 内訳・σ_y² 更新式を追加

### Phase C（Phase B 完成後）：Notion に別ページとして作成

**既存の Notion 本文ページ（`3641d35ae5f8816fb892d16986ecf1b4`）は上書きしない。**

`seminar_notion_full_formula_rich.md` を使い、**「数式強化版：Dual-ExpFam LSM ゼミ発表メモ」** として同じ親ページ（`3641d35ae5f880b2aa46ee2853a031ee`）配下に**別ページ**として新規作成する。

---

## 8. 追加しないと決めた式（理由付き）

| 式 | 追加しない理由 |
|---|-------------|
| Laplace 近似の詳細（ニュートン法の更新式） | 符号規約が Python 実装と整合しているか断定できない。「run_em_dual に従う」と記述で十分 |
| F の解析解（Gaussian-X） | 根拠ファイルで式自体を未確認。「先行研究 Eq.(10) と同一」と記述で十分 |
| Procrustes の最適化式 | 実験評価上の技術的詳細。ゼミ発表で深入りする必要なし |

> **根拠ファイル：** `conference_submission_final_draft.md`、`NOTEBOOKLM_RESEARCH_BRIEF.md` §7、`utils_expfam.py`、`model_expfam.py`、`model_dual_expfam.py`、`half_factor_math_explanation.md`、`half_factor_literature_code_check.md`
