# 数式・コード整合性監査

**作成日:** 2026-05-18  
**監査対象:** CLAUDE.md（root）・原稿・Python実装・MATLAB実装・先生対応メモ

---

## 1. このファイルの目的

提案手法 Dual-ExpFam LSM に関して、以下の資料間の整合性を1行ずつ確認し、
NotebookLM投入資料の「正しい式の基準」を確定する。

- 学会予稿（`conference_submission_final_draft.md`）
- 数学的証明（`docs/math_notes/half_factor_math_explanation.md`）
- MATLAB実装（`calcEtaNewton.m`）
- Python実装（`model_expfam.py`, `model_dual_expfam.py`, `utils_expfam.py`）
- 先生対応メモ（`docs/math_notes/legacy/parameter_estimation_corrected_formulas.md`）

---

## 2. 結論サマリー

| 確認項目 | 結論 |
|---------|------|
| 精度行列 Term3 の 1/2 | **原稿・MATLABは1/2なし（正しい）。Python実装は0.5あり（不整合）** |
| E-step 勾配 Term3 の 1/2 | 同上（Python L.109/L.159 に0.5あり、原稿・MATLABは1/2なし） |
| M-step 勾配の /2L | **正しい**（Σ_{i≠j}→Σ_{i<j}の換算として数学的に正当） |
| Newton方向への影響 | 1/2はY側Term3のみ。Term1・Term2は正しく実装。**完全な打ち消しは成立しない** |
| Laplace近似の影響 | Y側が支配的な方向でサンプリング分散が大きくなりやすい |
| BIC計算 | w0/wを非カウント（NOLTA 2024 Eq.(26)確認済み、w0/wは式上で明示なし）。Sigma/sigma_yは条件付き計上 |
| RMSE(Z) + Procrustes | **全実験で正しく適用**（run_em_dual L.555 / run_em L.249） |
| Scenario C X-mismatch | Y=Gaussian が Z 復元を支配、Xの貢献は小さい（実装バグと断定できない） |

---

## 3. 原稿で採用する正しい数式

### 精度行列（原稿 Eq.(6)、`docs/math_notes/half_factor_math_explanation.md` の確定式）

$$\mathbf{A}_i = \mathbf{I}_k + \mathbf{F}^\top \mathbf{V}_X(\mathbf{m}_i)\mathbf{F} + (w^Y)^2 \sum_{j \neq i} A_Y''\!\left(\eta_{ij}^Y\right) \mathbf{z}_j \mathbf{z}_j^\top$$

$$\mathbf{V}_X = \begin{cases}
\boldsymbol{\Sigma}^{-1} & (\mathrm{ExpFam}_X = \text{ガウス}) \\
\mathrm{diag}\!\left(A_X''(\mathbf{F}\mathbf{m}_i)\right) & (\text{ベルヌーイ・ポアソン})
\end{cases}$$

**σ²_z = 1 固定のため第1項は I_k のみ（原稿通り）**

### E-step 勾配（原稿では省略、`docs/math_notes/half_factor_math_explanation.md` Section 4 より）

$$\nabla_{z_i}\ln p(z_i|\cdot) = -\mathbf{z}_i + \mathbf{F}^\top \mathbf{V}_X[\mathbf{x}_i - A_X'(\mathbf{F}\mathbf{z}_i)] + w^Y\sum_{j \neq i}\left[T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)\right]\mathbf{z}_j$$

**1/2 なし。** M-step勾配（w0/w更新）の /2L とは別物。

---

## 4. 1/2係数問題の整理

### 4-1. 数学的根拠（`docs/math_notes/half_factor_math_explanation.md` より）

Y側の対数尤度を記法Aで書くと `(1/2)Σ_{i≠j}(...)` だが、z_i を固定して微分すると
「a=i の項」と「b=i の項」が両方 z_i に依存し合計すると 1/2 が消えて `Σ_{j≠i}` になる。
記法B（`Σ_{i<j}`）でも同様の計算で `Σ_{j≠i}` が得られる。
**どちらの記法を使っても精度行列・E-step勾配に 1/2 は出現しない（数学的に確定）。**

### 4-2. 各資料の 1/2 の有無

| 資料 | 箇所 | 1/2の有無 | 正誤 |
|-----|------|---------|-----|
| **NOLTA 2024 PDF Eq.(22)** | 精度行列 Term3 | **あり（1/2）** | ⚠ 再導出で不要と判明（後述） |
| **NOLTA 2024 PDF Eq.(23)** | E-step 勾配 Term3 | **あり（1/2）** | ⚠ 同上 |
| 原稿 Eq.(6) | 精度行列 Term3 | **なし** | ✓ 再導出・MATLAB確認に基づき採用 |
| `docs/math_notes/half_factor_math_explanation.md` Section 3 | 精度行列 Term3 | **なし** | ✓ 正しい（数学的に確定） |
| `MATLAB calcEtaNewton.m` L.56-63（calcAi） | 精度行列 Term3 | **なし** | ✓ 正しい |
| `MATLAB calcEtaNewton.m` L.43-49（calcGrad） | E-step 勾配 Term3 | **なし** | ✓ 正しい（w乗算欠如はあるが1/2は正しく不在） |
| `MATLAB calcw0.m` L.44 | M-step 勾配 | あり（`/2L`） | ✓ 正しい |
| `Python model_expfam.py` L.135 | 精度行列 Term3 | **0.5あり** | **× 不正**（NOLTA PDF式をそのまま踏襲） |
| `Python model_expfam.py` L.109 | E-step 勾配 Term3 | **0.5あり** | **× 不正**（同上） |
| `Python model_dual_expfam.py` L.200 | 精度行列 Term3（オーバーライド） | **0.5あり** | **× 不正** |
| `Python model_dual_expfam.py` L.159 | E-step 勾配 Term3（オーバーライド） | **0.5あり** | **× 不正** |
| `Python model_expfam.py` L.168, 200 | M-step 勾配（w0, w） | あり（`/2L`） | ✓ 正しい |
| `Python model_expfam.py` L.205 | Q関数Y側 | 0.5あり（`0.5*sum(ln_p)`） | ✓ 正しい |
| `docs/math_notes/legacy/parameter_estimation_corrected_formulas.md` Eq.(A7) | 精度行列 Term3 | **w²/2あり** | ⚠ 不整合（後述） |
| `docs/math_notes/legacy/parameter_estimation_corrected_formulas.md` Eq.(A10) | E-step 勾配 Term3 | **w/2あり** | ⚠ 不整合（後述） |

### 4-3. 先行研究PDFとMATLABの内部矛盾（重要な前提）

**NOLTA 2024 PDF Eq.(22)(23) に 1/2 が存在することを確認済み。**
一方、同じ研究グループのMATLAB実装（`calcEtaNewton.m` calcAi）には 1/2 が**ない**。
すなわち、先行研究の論文式とMATLABの間にもともと不整合が存在する。

本研究では `docs/math_notes/half_factor_math_explanation.md` の再導出とMATLAB確認に基づき、
**1/2なし（精度行列・E-step勾配ともに）を正しい式として採用し、原稿 Eq.(6) に反映済み。**

> NOLTA 2024のPDF式（1/2あり）とは異なるが、本研究では独立した数学的再導出とMATLAB実装の確認に基づき
> 1/2なしを採用している。Python実装が0.5を引き継いでいるのは、NOLTA 2024 PDF式をベースに実装したため。

### 4-4. `docs/math_notes/legacy/parameter_estimation_corrected_formulas.md` との矛盾

このファイル（確認日2026-05-08）の Eq.(A7) は精度行列に `(w^Y)^2/2` を含む（1/2あり）。
同 Eq.(A10) は E-step 勾配に `w^Y/2` を含む（1/2あり）。
さらに Section 5 に「これは先行研究[1] Eq.(23)と同じ慣行であり実装も一致」と明記されている。

これは `docs/math_notes/half_factor_math_explanation.md`（「1/2は不要」）および `conference_submission_final_draft.md` Eq.(6)（「1/2なし」）と**直接矛盾する**。

「先行研究Eq.(23)と同じ慣行」という記述は、NOLTA 2024 PDF の 1/2 を踏襲した中間段階の判断を反映している。
後から `docs/math_notes/half_factor_math_explanation.md` の再導出によって 1/2 が不要と確定し、原稿 Eq.(6) で修正済み。

**NotebookLM 用資料では `docs/math_notes/half_factor_math_explanation.md` と原稿 Eq.(6) を正として参照し、
`docs/math_notes/legacy/parameter_estimation_corrected_formulas.md` の Eq.(A7)(A10) は中間段階の矛盾資料として分類すること。**

### 4-5. Newton方向への影響（慎重な分析）

Python実装の精度行列と勾配の誤り：

```
A_wrong    = Term1 + Term2 + 0.5 * Term3_Y    （Term1, Term2は正しく実装）
grad_wrong = Term1_grad + Term2_grad + 0.5 * Term3_grad_Y
```

Newton更新方向：
```
-A_wrong^{-1} * grad_wrong
≠ -A_correct^{-1} * grad_correct  （一般に）
```

**CLAUDE.md の「Newton方向は正しい」という記述は不正確。**

正確な説明：
- Term3（Y側）のみに0.5が乗っており、Term1とTerm2には0.5が乗っていない
- Y側が支配的な場合（Term3 >> Term1 + Term2）は近似的に打ち消しが働くが、完全ではない
- Z事前分布（Term1）や X 寄与（Term2）が無視できない場合、Newton方向にも系統的なズレが生じる
- ズレの大きさはシナリオごとに異なる（Y分布族・データ量・w値による）

Laplace近似のサンプリング分散：
```
共分散 = A_wrong^{-1} = (Term1 + Term2 + 0.5*Term3_Y)^{-1}
                     ≠ A_correct^{-1}
```
Y側が支配的な方向でサンプリング分散が大きくなりやすい（2倍と断定はできない）。

---

## 5. Python実装に残る不整合

| ファイル | 行番号 | 内容 | 正誤 | 修正方針 |
|---------|-------|------|------|--------|
| `model_expfam.py` | L.109 | `term3 = 0.5 * w * (Z.T @ residual)` | **×** | `0.5` → 削除 |
| `model_expfam.py` | L.135 | `term3 = 0.5 * (w**2) * (...)` | **×** | `0.5` → 削除 |
| `model_dual_expfam.py` | L.159 | `term3 = 0.5 * w * (Z.T @ residual_y)` | **×** | `0.5` → 削除 |
| `model_dual_expfam.py` | L.200 | `term3 = 0.5 * (w**2) * (...)` | **×** | `0.5` → 削除 |
| `model_expfam.py` | L.168 | `grad = -grad_sum / (2.0 * L * phi)` | ✓ | 変更不要（M-step正しい） |
| `model_expfam.py` | L.200 | `grad = -grad_sum / (2.0 * L * phi)` | ✓ | 変更不要（M-step正しい） |

**注意:** `model_expfam.py` の `_calc_gradient` と `_calc_precision_matrix` は
`DualExpFamLSM` から **呼ばれない**（DualExpFamLSM が両者をオーバーライドしている）。
ただし `ExpFamLatentStructuralModel` としての使用（Y側のみ拡張の古いコード）では影響する。

**修正対応:** CLAUDE.md にある通り、修論フェーズで対応。現時点でコード修正禁止。

---

## 6. MATLAB実装との比較

| 比較対象 | MATLAB（calcEtaNewton.m） | Python（model_dual_expfam.py） |
|---------|--------------------------|-------------------------------|
| 精度行列 Term1 | `(1/varZ)*eye(k)` | `(1.0/var_z)*np.eye(k)` | 整合 |
| 精度行列 Term2 | `(F'/Sigma)*F` | `F.T@(F*sigma_inv_diag[:,None])` | 整合（Gauss-X） |
| 精度行列 Term3 | `w^2 * Z'*sig*Z`（**1/2なし**） | `0.5*(w**2)*(Z.T@diag(var_y)@Z)` | **不整合** |
| E-step 勾配 Term1 | `(1/varZ)*Z[i,:]'` | `-(1.0/var_z)*z_i` | 整合 |
| E-step 勾配 Term2 | `(F'/Sigma)*(x_i-F*z_i)` | `F.T@(residual_x*sigma_inv_diag)` | 整合 |
| E-step 勾配 Term3 | `(Y[i,:]-S)*Z`（**1/2なし**） | `0.5*w*(Z.T@residual_y)` | **不整合**（MATLABはwも欠如） |
| M-step 勾配（w0/w） | `Σ_{i≠j}/(2*L)` | `Σ_{i≠j}/(2*L*phi)` | 整合（phiは各族の分散） |
| Q関数 Y側 | `(1/2)*Σ_{i≠j}(...)` | `0.5*sum(ln_p)` | 整合 |

**MATLABのcalcAi自己項除去バグ（L.61）:** `sig(ind,ind)*(1-sig(ind,ind))*w^2` を引いているが
正しくは `sig(ind,ind)*w^2` のみ引けばよい。1/2係数問題とは独立したバグ。

---

## 7. Dual-ExpFam LSMのE-step対応表

### 7-1. 勾配（`model_dual_expfam.py::_calc_gradient`）

| 項 | 理論上の式 | Python実装（ファイル L.行） | 実装内容 | 整合性 | 注意点 |
|----|-----------|--------------------------|---------|-------|------|
| Term1（Z事前分布） | $-\mathbf{z}_i / \sigma_z^2$ | `model_dual_expfam.py` L.137 | `-(1.0/var_z)*z_i` | ✓ | σ²_z=1固定 |
| Term2（Gauss-X） | $\mathbf{F}^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}_i - \mathbf{F}\mathbf{z}_i)$ | L.144-147 | `F.T@(residual_x*sigma_inv_diag)` | ✓ | |
| Term2（非Gauss-X） | $\mathbf{F}^\top [T_X(\mathbf{x}_i) - A_X'(\mathbf{F}\mathbf{z}_i)]$ | L.149-150 | `F.T@residual_x` | ✓ | phi_X=1 |
| Term3（Y側） | $w^Y\sum_{j\neq i}[T_Y-A_Y']z_j$（1/2なし） | L.159 | `0.5*w*(Z.T@residual_y)` | **×** | 1/2 spurious |
| Gaussian Y の phi | 分母に sigma_y² | L.157-158 | `residual_y/sigma_y^2` | ✓ | |

**`_calc_gradient` の返り値：** `-(term1 + term2 + term3)`  
→ この関数は「負の対数事後分布の勾配（Newtonで最小化する方向）」を返す設計。

### 7-2. 精度行列（`model_dual_expfam.py::_calc_precision_matrix`）

| 項 | 理論上の式 | Python実装（ファイル L.行） | 実装内容 | 整合性 | 注意点 |
|----|-----------|--------------------------|---------|-------|------|
| Term1 | $\mathbf{I}_k / \sigma_z^2$ | L.181 | `(1.0/var_z)*np.eye(k)` | ✓ | |
| Term2（Gauss-X） | $\mathbf{F}^\top \boldsymbol{\Sigma}^{-1} \mathbf{F}$ | L.188-190 | `F.T@(F*sigma_inv_diag[:,None])` | ✓ | |
| Term2（非Gauss-X） | $\mathbf{F}^\top \mathrm{diag}(A_X'')\mathbf{F}$ | L.192-194 | `F.T@(F*var_x_i[:,None])` | ✓ | |
| Term3（Y側） | $(w^Y)^2\sum_{j\neq i}A_Y''(\eta)z_jz_j^\top$（1/2なし） | L.200 | `0.5*(w**2)*(Z.T@diag(var_y)@Z)` | **×** | 1/2 spurious |

---

## 8. M-stepとパラメータ更新の対応表

| パラメータ | 更新方法 | 実装（ファイル：関数） | 整合性 | 備考 |
|-----------|---------|---------------------|-------|-----|
| F（Gauss-X） | 解析解（NOLTA 2024 Eq.(10)同一） | `DualExpFamLSM.calc_F` → parent継承 | ✓ | |
| F（非Gauss-X） | Adam勾配上昇 | `DualExpFamLSM._calc_F_adam` L.219-268 | ✓ | `∇_F = (1/L)Σ_l (X-A_X'(ZF^T))^T Z_l` |
| Sigma（Gauss-X） | 解析解MLE | `DualExpFamLSM.calc_sigma` → parent継承 | ✓ | |
| Sigma（非Gauss-X） | 固定 = I | L.285: `return np.eye(self.d)` | ✓ | Sigma不要な族 |
| w0^Y | Adam（/2L 正しい） | `ExpFamLatentStructuralModel.calc_w0` L.149-178 | ✓ | phi_Y=sigma_y²（Gauss-Y） |
| w^Y | Adam（/2L 正しい） | `ExpFamLatentStructuralModel.calc_w` L.180-210 | ✓ | 同上 |
| sigma_y（Gauss-Y） | 解析解MLE | `ExpFamLatentStructuralModel.calc_sigma_y` L.212- | ✓ | |

**注:** `calc_log_likelihood_Y`（Q関数Y側）は `DualExpFamLSM` でオーバーライドされず、
`ExpFamLatentStructuralModel` 経由で `model_expfam.py` L.267 の実装を使用。
`0.5 * sum(ln_p)` で `(1/2)Σ_{i≠j}` を計算しており **正しい**。

---

## 9. BIC計算の監査

### `calc_bic_dual`（`utils_expfam.py` L.386-404）

```python
f_params   = k * d - k * (k - 1) // 2    # F: 回転拘束後の自由パラメータ数
sigma_x_p  = d if family_x == 'gaussian' else 0   # Σ (Gauss-X のみ)
sigma_y_p  = 1 if family_y == 'gaussian' else 0   # sigma_y (Gauss-Y のみ)
num_params = f_params + sigma_x_p + sigma_y_p
BIC = -2 * Q_strict + num_params * ln(n)
```

**NOLTA 2024 Eq.(26) の確認済み式：**

$$\mathrm{BIC} = -2\ln\hat{L} + \left\{(k+1)d - \frac{k(k-1)}{2}\right\}\ln n$$

ここで $(k+1)d - k(k-1)/2 = kd - k(k-1)/2 + d$ = F自由パラメータ数 + Σ（対角）。  
**w0, w は Eq.(26) の num_params に明示されていない**（式上では含まれていない）。

| 項目 | 扱い | 整合性 |
|-----|------|-------|
| F（回転拘束あり） | k*d - k*(k-1)//2 | ✓ NOLTA 2024 Eq.(26)と一致 |
| Σ（Gauss-X） | dパラメータ計上 | ✓ NOLTA 2024 Eq.(26)の「+d」に対応 |
| sigma_y（Gauss-Y） | 1パラメータ計上 | ✓（本研究で追加、NOLTA 2024にはGauss-Yなし） |
| w0, w^Y | **カウントしない** | ⚠ NOLTA 2024 Eq.(26)でも明示なし（慣行として一致） |

**w0, w をBICパラメータ数から除外する扱い** は NOLTA 2024 Eq.(26) で明示的に含まれていないことと一致する。
ただし、w0/w を除外することが意図的な設計判断か省略かは論文本文からは断定できない。
（`calc_bic` Y-onlyモデルも同じ慣行で、w0/wはカウントしない。）

**Q_strict について：**
- Poisson-Y: `-Σln(y_ij!)` を追加（`calc_Q_dual_strict` L.375）
- Poisson-X: `-Σln(x_il!)` を追加（L.378）
- Gaussian/Bernoulli: 正規化定数の扱いは定数なので BIC 比較には影響なし

---

## 10. RMSE計算とProcrustesの監査

### Procrustes回転の適用状況

| 場所 | 行番号 | 適用 |
|-----|-------|------|
| `run_em_dual`（`utils_expfam.py`） | L.555-557 | ✓ `procrustes_rotation(Z_est, true_params["Z"])` |
| `run_em`（`utils_expfam.py`） | L.249-251 | ✓ 同上 |

`exp_scenario_lib.py` の各実験関数は `run_em_dual` を呼び出しており、
返り値の `rmse_Z` はすでにProcrustes回転後の値。**全実験でProcrustesが正しく適用されている。**

### `procrustes_rotation` の実装（`utils_expfam.py` L.38-43）

```python
k_min = min(A_est.shape[1], A_true.shape[1])
M = A_est[:, :k_min].T @ A_true[:, :k_min]
U, _, Vt = np.linalg.svd(M)
return U @ Vt, k_min
```

回転行列 R = U @ Vt（直交行列）を返し、RMSE(Z) は `||Z_est[:,:k_min] @ R - Z_true[:,:k_min]||` で計算。
**実装は正しい（符号の不定性のみ除去する正規直交変換）。**

---

## 11. Scenario CのX-mismatch問題

### 11-1. Scenario Cの設定

- true X = Bernoulli, true Y = Gaussian
- 実験設定: n=150, d=15, k*=3, 10試行

### 11-2. mismatch実験のRMSE(Z)（`exp_scenario_C_exp4_mismatch.csv` より集計）

| 条件 | model_fx | model_fy | fix_w | fix_x | RMSE(Z)平均 | RMSE(Z)標準偏差 |
|-----|---------|---------|------|------|------------|---------------|
| **Proposed** | bernoulli | **gaussian** | False | False | **0.0287** | 0.0014 |
| Y-only（X無視） | bernoulli | gaussian | False | **True** | **0.0286** | 0.0013 |
| X=Poisson（誤指定） | **poisson** | gaussian | False | False | **0.0284** | 0.0013 |
| X=Gaussian（誤指定） | **gaussian** | gaussian | False | False | 0.1055 | 0.1951 |
| X-only（Y無視） | bernoulli | gaussian | **True** | False | 1.1024 | 0.0528 |
| Y=Bernoulli（誤指定） | bernoulli | **bernoulli** | False | False | 0.4523 | 0.0046 |
| Y=Poisson（誤指定） | bernoulli | **poisson** | False | False | 0.3204 | 0.0119 |

### 11-3. 現象の解釈

**観察:** fix_x（Y-only）が Proposed とほぼ同等（0.0286 vs 0.0287）。
X=Poisson 誤指定でも同等（0.0284）。

**仮説（データの性質から）：**
Y_ij ~ N(w0 + 0.5 * z_i^T z_j, sigma_y^2) は内積 z_i^T z_j を連続値で直接観測する。
- Y 観測数: n*(n-1)/2 = 150*149/2 = **11,175 個**（連続値）
- X 観測数: n*d = 150*15 = **2,250 個**（2値）

Gaussian Y は高い情報量で Z の内積構造を拘束するため、
Bernoulli X の追加情報は Z 回復に対してほぼ冗長になりやすい。

**反証すべき仮説（実装バグの可能性）：**
`fix_x=True` は F=0 に固定する実装（`utils_expfam.py` L.486-488）。
M-step でも `if not fix_x: F = model.calc_F(...)` によりFが更新されない（L.519-521）。
これは意図通り。fix_x=True で RMSE(Z) が良好なのは、
Gaussian Y だけで Z が回復できている、というデータの性質を反映していると解釈できる。

**断定できないこと:** X side が実際に計算に寄与しているかどうか（量的な分析）。

### 11-4. この問題の研究上の意義

Scenario C での主張は「Y=Gaussianを正しく指定すると、Y=Bernoulli等の誤指定より大幅に良い」。
X-mismatch が軽微なことは、Y 分布族の正確な指定が X 分布族の正確な指定より重要、
という主張の補強になり得る（論文でも言及済み: "ExpFam_Y が推定に支配的な寄与をする"）。

---

## 12. NotebookLM用資料での書き方

### 正式採用する式（根拠: 原稿 Eq.(6) + `docs/math_notes/half_factor_math_explanation.md`）

精度行列: **1/2 なし**（$(w^Y)^2 \sum_{j\neq i}$）  
E-step 勾配: **1/2 なし**（$w^Y \sum_{j\neq i}$）  
M-step 勾配 (w0/w): **1/2あり正しい**（$\frac{1}{2L}\sum_{i\neq j}$）

### 実装との関係を説明する際の注意書き

> 「NOLTA 2024 PDF Eq.(22)(23) には E-step 精度行列・勾配の Term3 に 1/2 が記載されている。
> Python 実装（model_expfam.py, model_dual_expfam.py）はこの PDF 式を踏襲して 0.5 が残存している（L.135, L.200）。
> 一方、同研究グループの MATLAB 実装（calcAi）には 1/2 がなく、本研究の再導出でも 1/2 は不要と確定した。
> 本研究の最終原稿 Eq.(6) は 1/2 なしを採用している。
> Python 実装の 0.5 は Newton 方向に Y 側が支配的な場合の系統的ズレを生じさせるが、
> 現段階では修論フェーズでの修正予定として保留中。」

### `docs/math_notes/legacy/parameter_estimation_corrected_formulas.md` の位置づけ

このファイルは参照禁止ではないが、Eq.(A7)(A10) の 1/2 の記述が最新の確定事項と矛盾する。
NotebookLM に投入する場合は矛盾資料として分類し、原稿 Eq.(6) と比較させるプロンプトを設定すること。

---

## 13. 修正すべきだが今は触らないもの

CLAUDE.md の方針に従い、以下の修正は修論フェーズまで保留：

| 対象 | 内容 | 影響 |
|-----|------|------|
| `model_expfam.py` L.109, 135 | `0.5 *` を削除 | E-step の Y 側 Term3 が正しくなる |
| `model_dual_expfam.py` L.159, 200 | `0.5 *` を削除 | 同上 |
| `model_dual_expfam.py` docstring L.16, 21 | `w/2phi_Y` → `w/phi_Y` に修正 | ドキュメントと原稿が一致する |

---

## 14. 未確認事項

| 未確認項目 | 根拠 | 優先度 |
|-----------|------|------|
| ~~先行研究 Eq.(22)(23) に 1/2 があるかどうか~~ | **確認済み：NOLTA 2024 Eq.(22)(23) に 1/2 あり** | 解決済み |
| w0, w をBICカウントしない根拠の明示 | NOLTA 2024 Eq.(26) では式上に含まれていないことを確認。意図的除外か省略かは不明 | 低（実用上は問題なし） |
| E-step中の ‖Term2‖ vs ‖Term3‖ の比較 | 未実施（要追加実験） | 中 |
| Scenario CでFが実際に更新されているかの確認 | ログ確認未実施 | 中 |
| MATLAB calcGrad の w 乗算欠如が実験結果に与える影響 | 先行研究側の問題 | 低 |

---

## 15. 次にやるべき追加検査

以下の追加検査をすることで、Scenario C X-mismatch 問題と1/2影響を定量化できる。

### 追加検査 A（優先高）: Term2 vs Term3 ノルム比較

E-step の各反復で `||Term2||` と `||Term3||` を記録し、シナリオ間で比較する。
→ Scenario C で Term3 >> Term2 なら「Y支配」の定量的証拠になる。

### 追加検査 B（優先中）: 1/2修正版との比較実験

`model_dual_expfam.py` の L.159, L.200 から `0.5 *` を削除したバージョンで
同じ実験条件（Scenario A, n=150, k=3, 10試行）を実行し、RMSE(Z) の変化を確認する。
→ 1/2の有無が結論に影響するかどうかを直接確認できる。

### 追加検査 C（優先中）: Scenario C で X-only条件の RMSE(X) を確認

fix_x=True のとき RMSE(X) が大きくなるはずで、「Xが予測できていない状態でも RMSE(Z) が良い」という
現象を定量確認することで、「Y が Z を単独復元できている」という解釈の裏付けになる。

### 追加検査 D（優先低）: sigma_y が小さいかどうかの確認

Gaussian Y の sigma_y（推定値）が小さければ Y が高精度で z_i^T z_j を観測しており、
Y 支配の説明が強化される。`log_scenario_C.txt` の sigma_y の推定値ログを確認する。
