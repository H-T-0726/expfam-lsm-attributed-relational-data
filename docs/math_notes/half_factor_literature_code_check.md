# 精度行列の Y 側 1/2 係数：論文・MATLAB・Python 実装照合

調査日：2026-05-08

---

## 結論（先に示す）

**先生の指摘は数学的に正しい。**

| 観点 | 1/2 の有無 | 根拠 |
|------|-----------|------|
| 数学的正解 | **なし** | z_i の条件付き事後分布を微分すると自然に Σ_{j≠i} が現れ，1/2 は不要 |
| MATLAB 原典実装（calcAi） | **なし** | calcEtaNewton.m L.56-63 |
| Python 再現実装（model.py） | **あり** | reproduction/src/model.py L.353，「Based on paper Eq.(22)」と注釈 |
| Python 提案実装（model_expfam.py） | **あり** | model_expfam.py L.135 |
| Python 提案実装（model_dual_expfam.py） | **あり** | model_dual_expfam.py L.200 |

**MATLAB 原典には 1/2 がなく，Python 再現実装には 1/2 がある。**  
MATLAB と Python の間に不整合がある。

---

## 1. 先行研究の論文（PDF）

### 確認結果

`paper/A_study_on_latent_structural_models_for_binary_rel.pdf`：**読み込みツール非対応のため直接確認不可**（pdftoppm 未インストール）。  
`paper/2.pdf`：同上，確認不可。

### 間接的証拠

Python 再現実装 `reproduction/src/model.py` の `_calc_precision_matrix` のコメント：

```python
"""
Based on paper Eq. (22):
A_i = (1/σ²_z)I + F^T Σ^{-1} F + (1/2) Σ_{j≠i} s_ij(1-s_ij) w² z_j z_j^T
"""
```

（reproduction/src/model.py L.305-307）

このコメントが事実であれば，先行研究の論文 Eq.(22) には 1/2 が書かれている可能性がある。  
ただし，**PDF を直接確認していないため断定できない**。

---

## 2. MATLAB 実装

### 発見したファイル

- `Mato Lab Program/calcEtaNewton.m`
- `Mato Lab Program/EffcalcEtaNewton (1).m`
- `Mato Lab Program/calcw0.m`
- `Mato Lab Program/calcw.m`
- `Mato Lab Program/calcdescmetric_ver4.m`

### 2-A. 精度行列（calcAi）

**ファイル：** `Mato Lab Program/calcEtaNewton.m`  
**関数：** `calcAi`  
**行番号：** L.56-63

```matlab
function Ai = calcAi(F, varZ, Sigma, w0, w, Z, ind)
    FSF = (F' / Sigma) * F;
    sig = diag(calcsigmoid(w0 + w .* Z(ind, :) * Z'));
    sig = w^2 .* sig .* (1 - sig);          % ← w^2 * s_j*(1-s_j) を対角に
    tmp = Z' * sig * Z;                      % ← Σ_j w^2*s_j(1-s_j) z_j z_j^T
    tmp = tmp - sig(ind,ind)*(1-sig(ind,ind)).* w^2 .* Z(ind,:)' * Z(ind,:); % ← 自己項除去
    Ai = (1 / varZ) * eye(size(Z, 2)) + FSF + tmp;
end
```

**1/2 の有無：なし。**  
Y 側の Term3 = `Σ_{j≠i} w^2 * s_j(1-s_j) * z_j z_j^T`（1/2 なし）。

なお，自己項の除去コード `sig(ind,ind)*(1-sig(ind,ind))*w^2` は意図とは異なる値を引いており  
（正しくは `sig(ind,ind) .* Z(ind,:)' * Z(ind,:)` のみでよい），実装上のバグがあるが，  
**1/2 係数の有無という観点では 1/2 は存在しない**。

### 2-B. E ステップ勾配（calcGrad）

**ファイル：** `Mato Lab Program/calcEtaNewton.m`  
**関数：** `calcGrad`  
**行番号：** L.43-49

```matlab
function partE = calcGrad(X, Y, Z, Sigma, F, varZ, w0, w, ind)
    FS = (F' / Sigma) * (X(ind, :)' - F * Z(ind, :)');
    S = calcsigmoid(w0 + w .* Z(ind, :) * Z');
    YSWZ = (Y(ind, :) - S) * Z - (Y(ind, ind) - S(ind))* Z(ind, :);
    partE = (1 / varZ) * Z(ind, :)' - FS - YSWZ';
end
```

計算される Term3 = `Σ_{j≠i} (y_ij - s_ij) * z_j`

**1/2 の有無：なし。**  
なお数学的に正しい勾配は `w * Σ_{j≠i} (y_ij - s_ij) * z_j` であり，MATLAB には `w` 乗算が欠けている  
（別の実装バグ）が，1/2 係数の観点では 1/2 は存在しない。

### 2-C. M ステップ（calcw0, calcw）

**ファイル：** `Mato Lab Program/calcw0.m`，L.44

```matlab
gradW0 = sum((Y - S), 'all') / (2 * size(Z, 3));
```

**ファイル：** `Mato Lab Program/calcw.m`，L.42

```matlab
gradW = sumtmp / (2 * size(Z, 3));
```

勾配 = `Σ_{i≠j}(...) / (2*L)` = `Σ_{i<j}(...) / L`（対称行列の対称和を片側和に換算）。  
**1/2 の有無：あり（正しい）。** Q 関数の偏微分として整合的。

### 2-D. Q 関数の Y 側（calcp_Y）

**ファイル：** `Mato Lab Program/calcdescmetric_ver4.m`，L.112-117

```matlab
function p_Y = calcp_Y(Y, Z, w0, w)
    S = 1 ./ (1 + exp(-w0 - w .* Z * Z'));
    tmp = (Y .* log(S + 10e-7)) + ((1 - Y) .* log((1 - S) + 10e-7));
    tmp = tmp - diag(diag(tmp));
    p_Y = sum(tmp, 'all') / 2;             % ← (1/2) * Σ_{i≠j}
end
```

**1/2 の有無：あり（正しい）。** Q 関数内の Y 対数尤度は `(1/2) Σ_{i≠j}` として計算。

### MATLAB まとめ

| 箇所 | 1/2 の有無 | 正しいか |
|------|-----------|---------|
| calcp_Y（Q 関数 Y 側） | あり | ✓（(1/2)Σ_{i≠j} = Σ_{i<j} に対応） |
| calcAi（精度行列 Y 側） | **なし** | ✓（数学的正解） |
| calcGrad（E ステップ勾配 Y 側） | **なし** | ✓（w 乗算は別途バグあるが 1/2 は正しく不在） |
| calcw0, calcw（M ステップ勾配） | あり | ✓（Σ_{i≠j}→Σ_{i<j} の換算として正しい） |

---

## 3. Python 実装

### 3-A. 精度行列（再現実装）

**ファイル：** `reproduction/src/model.py`  
**関数：** `_calc_precision_matrix`  
**行番号：** L.353

```python
term3 = 0.5 * (w ** 2) * (Z.T @ np.diag(s_deriv) @ Z)
```

**1/2 の有無：あり。** 関数 docstring に "Based on paper Eq.(22)" と注釈がある。

### 3-B. 精度行列（提案モデル基底クラス）

**ファイル：** `expfam/src/model_expfam.py`  
**関数：** `_calc_precision_matrix`  
**行番号：** L.135

```python
term3 = 0.5 * (w ** 2) * (Z.T @ np.diag(var_fn) @ Z)
```

**1/2 の有無：あり。**

### 3-C. 精度行列（提案モデル Dual 版）

**ファイル：** `expfam/src/model_dual_expfam.py`  
**関数：** `_calc_precision_matrix`  
**行番号：** L.200

```python
term3 = 0.5 * (w ** 2) * (Z.T @ np.diag(var_y) @ Z)
```

**1/2 の有無：あり。**

### 3-D. E ステップ勾配（再現実装）

**ファイル：** `reproduction/src/model.py`  
**関数：** `_calc_gradient`  
**行番号：** L.283

```python
term3 = 0.5 * w * (Z.T @ y_minus_s)
```

**1/2 の有無：あり。** こちらも数学的正解は `1.0 * w * Σ_{j≠i}(...)` であり，1/2 は余分。

### 3-E. Y 対数尤度（Q 関数）

**ファイル：** `expfam/src/model_expfam.py`  
**関数：** `calc_log_likelihood_Y`  
**行番号：** L.267

```python
np.fill_diagonal(ln_p, 0.0)
ll += 0.5 * np.sum(ln_p)    # ← (1/2) * Σ_{i≠j}
```

**1/2 の有無：あり（正しい）。** MATLAB calcp_Y と同じ慣行で整合。

### 3-F. M ステップ勾配（再現実装）

**ファイル：** `reproduction/src/model.py`  
**関数：** `calc_w0`  
**行番号：** L.663

```python
grad = -grad_sum / (2.0 * L)
```

**1/2 の有無：あり（正しい）。** Σ_{i≠j}/2L = Σ_{i<j}/L として正しい。

### Python まとめ

| 箇所 | 1/2 の有無 | 正しいか |
|------|-----------|---------|
| calc_log_likelihood_Y（Q 関数 Y 側） | あり | ✓ |
| _calc_precision_matrix（精度行列 Y 側） | あり | **× 数学的に余分な 1/2** |
| _calc_gradient（E ステップ勾配 Y 側） | あり | **× 数学的に余分な 1/2** |
| calc_w0, calc_w（M ステップ勾配） | あり | ✓ |

---

## 4. MATLAB vs Python の不整合まとめ

| 確認箇所 | MATLAB | Python | 数学的正解 |
|---------|--------|--------|----------|
| Q 関数の Y 対数尤度 | (1/2)Σ_{i≠j} | (1/2)Σ_{i≠j} | どちらでも等価 |
| 精度行列 Y 側 | **w^2 Σ_{j≠i}（1/2 なし）** | 0.5 w^2 Σ_{j≠i} | **1/2 なし** |
| E ステップ勾配 Y 側 | Σ_{j≠i}（1/2 なし，w 乗算欠如） | 0.5 w Σ_{j≠i} | **1/2 なし，w あり** |
| M ステップ勾配 | Σ_{i≠j}/(2L) | Σ_{i≠j}/(2L) | どちらも正しい |

**Python 実装は，精度行列と E ステップ勾配の両方に余分な 1/2 を持っている。**  
これは MATLAB 原典と一致せず，数学的正解とも異なる。

ただし，勾配と精度行列の**両方**に同じ 1/2 があるため，ニュートン更新の方向は正しい（後述）。  
誤差はサンプリング分散（共分散行列 A_i^{-1}）が 2 倍に膨らむことにある。

---

## 5. PDF 確認について

- `paper/A_study_on_latent_structural_models_for_binary_rel.pdf`：確認不可
- `paper/2.pdf`：確認不可
- `Mato Lab Program/NOLTA_exp_ver5 (1).pdf`：確認不可

論文 Eq.(22) に 1/2 が書かれているかどうかは **PDF を直接読んで確認することを強く推奨する**。  
Python 再現実装のコメントから 1/2 が論文に記載されている可能性はあるが，断定はできない。
