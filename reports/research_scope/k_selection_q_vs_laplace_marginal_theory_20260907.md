# 潜在変数 Z の扱いによる K 依存項の変化 — Q-based / conditional / Laplace-marginal の数式比較

**作成日:** 2026-09-07
**種別:** READ-ONLY 理論監査。**コード変更なし・実験実行なし・Issue/PR 作成なし・criterion の採用判断なし。**
**新規作成物:** 本ファイル 1 件のみ。既存 CSV / figure / historical report は一切変更していない。
**ブランチ:** `docs/notion-k-selection-story-20260906`

**前提文書:** `reports/research_scope/q_based_criterion_decomposition_audit_20260907.md`（2026-09-07 修正版）。
同文書の実装監査 [CODE]・一次 CSV 監査 [ARTIFACT]・原論文確認 [PAPER] の結果を前提とし、本監査では覆さない。

**本監査が行うこと:** 潜在変数 `Z` を
**(A) complete-data Q として評価する / (B) exact posterior で期待値を取る / (C) conditional に plug-in する /
(D) observed likelihood として積分する** の 4 通りで扱ったときに、
**K 依存項が数式上どう変わるか**を導出する。

**本監査が行わないこと:** どの criterion を採用すべきかの判断。standard Schwarz BIC が正しいという前提。
current Q-based が誤りであるという前提。

---

## 1. Executive summary

### 1.1 中心的な導出結果 [DERIVED]

`m = nK` とし、`ℓ_cond,K(Z,θ) := log p_K(D|Z,θ)`、`Ẑ_K` を `h_K(Z;θ) = log p_K(D,Z|θ)` の mode、
`H_K = −∇²_Z h_K(Ẑ_K;θ)` とする。observed likelihood を `Z` について Laplace 近似すると:

```
−2 ℓ_obs,K(θ)  ≈  −2 ℓ_cond,K(Ẑ_K, θ)  +  ‖Ẑ_K‖²_F  +  log det H_K
```

**Gaussian prior の正規化定数 `−(m/2)log 2π` は、Laplace 体積因子 `(2π)^{m/2}` と厳密に打ち消し合い、
`m log 2π` は最終式に一切残らない**（§10 で途中式を省略せず導出）。

これに対し current Q-based criterion は:

```
ΔC_Q(K→K+1) = Δ D_fit + n·log 2π + n + (d−K)·log n
                          └─ 275.6816 ─┘ └150┘   └ 60.128 ┘        (n=150, d=15, K=3)
```

**両者は「同じ量の別表現」ではない。** 対応関係は次のとおりである。

| current Q-based の項 | Laplace-marginal での対応物 |
|---|---|
| `+nK·log 2π`（Gaussian 正規化） | **存在しない（体積因子と相殺して消える）** |
| `+nK`（`scale_Z` による latent norm の固定） | `+‖Ẑ_K‖²_F`（**固定されず、データ依存**） |
| **（対応項なし）** | `+log det H_K`（posterior volume / curvature） |
| `+p̂ log n` | **段階が違う。** これは `θ` についての第 2 段階 Laplace から出る形（§12） |

### 1.2 425.6816/K の内訳 [DERIVED]

```
n·(1 + log 2π) = n·log 2π  +  n
425.6815600    = 275.6815600 + 150.0000000        (n = 150)
```

| 成分 | 値 (n=150) | 出所 | どの基準に現れるか |
|---|---:|---|---|
| `n·log 2π` | **275.6816** | Gaussian prior の**正規化定数**。`Z` の値に依存しない | **A: あり／B: あり／C: なし／D: 相殺して消える** |
| `n` | **150.0000** | latent norm `‖Z‖²_F` が `scale_Z` により `nK` に固定されること | **A: 固定値／B: `E_{q*}‖Z‖²` で可変／C: なし／D: `‖Ẑ_K‖²` で可変** |

> **したがって「425.6816/K は Gaussian prior から必然的に生じる」は誤りである。**
> **275.6816/K は `log p(Z)` を criterion に含める限り必然だが、
> 残り 150/K（= n/K）は `scale_Z` との組合せで初めて決定論化する** [DERIVED]。

### 1.3 内部整合性チェック [DERIVED]

§7（B を global Gaussian posterior で近似したケース）と §9-10（D の Laplace 近似）は
**独立に導出したうえで同一の式に到達する**（§10.4）。符号ミスがないことの確認になっている。

### 1.4 本監査が新たに導いた構造的事実 [DERIVED]

**fixed 系列の per-node `A_i` は、大域 Hessian `H_K` の `(i,i)` 対角ブロックと厳密に一致する**（§9.3）。
一方 `H_K` には非ゼロの非対角ブロック `−w r_ij I_K + w² V_Y(η_ij) z_j z_i^T` があり、
その第 1 項は残差 `r_ij` の符号により**不定符号**である。したがって:

- `H_K ≠ blockdiag(A_1,…,A_n)`
- Fischer の不等式より **`log det H_K ≤ Σ_i log det A_i`**（`H_K ≻ 0` の下で）
- `A_i ⪰ I` は成り立つが、**`H_K ⪰ I_m` は保証されない**

（historical 0.5 系列の `A_i` は対角ブロックと一致しない。これは KI-001 の既存判断と整合する
独立な観察であり、KI-001 の判断を変更するものではない。§9.3.2）

---

## 2. Scope

### 2.1 対象

- **基本拡張**（X 全体に 1 family、Y 全体に 1 family）の joint model。
- 4 基準 A / B / C / D の **K 依存項の数式比較**のみ。
- per-column は §15 で**拡張可能性の指摘だけ**を行う。

### 2.2 非対象

- criterion の採用・変更・提案
- 新実験、コード変更
- clean / S1 / K_PRED 系列（本監査の中心にしない）
- fixed 30/30 を consistency の証明として扱うこと

### 2.3 Evidence labels

| ラベル | 意味 |
|---|---|
| `[CODE]` | 実コードの file path + 行番号 |
| `[ARTIFACT]` | 一次 CSV の path + 値 |
| `[PAPER]` | 原論文 PDF から直接確認した式・記述 |
| `[DERIVED]` | 本監査が自分で行った数学的導出 |
| `[ASSUMPTION]` | 明示した理想化（Gaussian posterior 等）。成立の証明はない |
| `[INFERENCE]` | 一般論からの推測 |
| `[UNRESOLVED]` | 証明も反証もできない |

### 2.4 記号

| 記号 | 意味 |
|---|---|
| `D = (X, Y)` | 観測データ。`X ∈ R^{n×d}`、`Y ∈ R^{n×n}`（上三角のみ使用） |
| `Z ∈ R^{n×K}` | 潜在変数行列。`m := nK` |
| `θ` | `{F, w_0^Y, w^Y}` ＋ Gaussian-X なら `Σ_X`、Gaussian-Y なら `σ_Y²` |
| `ℓ_cond,K(Z,θ)` | `log p_K(D\|Z,θ) = log p(X\|Z,θ_X) + log p(Y\|Z,θ_Y)` |
| `h_K(Z;θ)` | `log p_K(D,Z\|θ) = log p_K(Z) + ℓ_cond,K(Z,θ)` |
| `Ẑ_K` | `h_K(·;θ)` の mode |
| `H_K` | `−∇²_Z h_K(Ẑ_K;θ)`（`m×m`） |
| `A_i` | コード上の per-node precision（`_calc_precision_matrix`）。`K×K` |
| `p̂ = P_K` | `calc_bic_dual` の parameter count |

---

## 3. Common joint model

### 3.1 joint 密度 [CODE + PAPER]

基本拡張の生成モデル（root `CLAUDE.md` §1、原論文 Fig.1 と Eq.(16) 周辺）:

```
p_K(D, Z | θ) = p_K(Z) · p_K(X | Z, θ_X) · p_K(Y | Z, θ_Y)
```

- `z_i ~ N(0, I_K)` i.i.d.（実装は `var_z = 1.0` に固定。`reproduction/src/model.py:99`）[CODE]
- `x_il ~ ExpFam_X(η^X_il = f_l^T z_i)`（バイアスなし）
- `y_ij ~ ExpFam_Y(η^Y_ij = w_0 + w z_i^T z_j)`、`i < j`

### 3.2 log prior [DERIVED]

`var_z = 1`、`m = nK` として

```
log p_K(Z) = −(m/2)·log(2π) − (1/2)·‖Z‖²_F                                    …(3.1)
```

- 第 1 項 `−(m/2)log 2π` は **`Z` の値に依存しない正規化定数**。`K` に線形。
- 第 2 項 `−(1/2)‖Z‖²_F` は **`Z` に依存する energy 項**。
- 事前分布のもとでは `E_prior[‖Z‖²_F] = m = nK`（各成分が `N(0,1)`）[DERIVED]。

### 3.3 conditional log-likelihood [CODE]

```
ℓ_cond,K(Z,θ) = Σ_{i,l} log p_{X}(x_il | η^X_il) + Σ_{i<j} log p_{Y}(y_ij | η^Y_ij)      …(3.2)
```

Y 側が `Σ_{i<j}`（ユニークペア和）であることはコードで確認済み
（`model_expfam.py:266-267` が非対角全和に `0.5` を掛けて `Σ_{i<j}` にしている）[CODE]。
原論文 Eq.(16) の `{∏_{i≠j}}^{1/2}` と同じ規約である [PAPER]。

### 3.4 重要な制約 — `scale_Z` を勝手に持ち込まない

**`scale_Z`（`model.py:468-504`）は実装上の事実であり、A でのみ導入する。**
B / C / D は生成モデル (3.1) のみを前提とし、**`‖Z‖²_F = nK` を仮定しない** [CODE + DERIVED]。

---

## 4. Current Q-based criterion（A）

### 4.1 定義 [CODE]

```
C_Q(K) = −2·Q_strict,K + P_K·log n                                  utils_expfam.py:403

Q_strict,K = (1/L) Σ_{l=1}^{L} [ log p_K(Z^(l)) + log p(X|Z^(l),θ) + log p(Y|Z^(l),θ) ] + corr
                                                                    utils_expfam.py:345-352, :372-379
```

`corr` は Poisson の階乗補正（K 非依存定数）[CODE]。

### 4.2 `scale_Z` による決定論化（既存監査の再利用）[CODE + DERIVED]

`scale_Z` は `Z_samples` 配列**全体**の平均二乗を 1 に正規化する（`model.py:499-503`）。したがって

```
(1/L) Σ_l ‖Z^(l)‖²_F = nK                     （L 平均について厳密。実数演算で厳密、float64 で 1±O(10⁻¹⁶)）
```

(3.1) は `‖Z‖²_F` についてアフィンなので L 平均と交換でき、

```
lnpZ_avg = −(m/2)·log 2π − (1/2)·nK = −(m/2)·(1 + log 2π)
−2·lnpZ_avg = nK·(1 + log 2π)                                                  …(4.1)
```

### 4.3 K 差分 [DERIVED]

```
D_fit(K) := −2·[ (1/L)Σ_l ℓ_cond,K(Z^(l),θ̂_K) + corr ]        （A の当てはまり項）

C_Q(K) = D_fit(K) + nK(1 + log 2π) + P_K·log n

ΔC_Q(K) := C_Q(K+1) − C_Q(K)
         = Δ D_fit + n·(1 + log 2π) + (d − K)·log n                            …(4.2)
```

`ΔP_K = [d(K+1) − (K+1)K/2] − [dK − K(K−1)/2] = d − K` [DERIVED]。

### 4.4 数値（n=150, d=15, K=3→4）[DERIVED]

| 項 | 値 |
|---|---:|
| `n·log 2π` | 275.6816 |
| `n` | 150.0000 |
| `n·(1 + log 2π)` | **425.6816** |
| `(d−K)·log n = 12 × 5.0106` | **60.1276** |
| **合計（`Δ D_fit` を除く K 追加 cost）** | **485.8092** |

> **呼称の固定:** この 485.81 を「正しい penalty」と呼ばない。
> **`current implementation 上の K 追加 cost`** と呼ぶ。
> 「正しさ」は §11-§13 で 4 基準を比較したうえでも決着しない（§18）。

### 4.5 A における `q` の位置づけ（前提文書からの引き継ぎ）

`Z^(l)` は「各 `p(z_i | Z_{−i}, D, θ)` の Laplace 近似を条件付き分布として用いた Gauss–Seidel 逐次
サンプリングの `scale_Z` 適用後の出力」であり、exact posterior でも per-node Laplace の積でもない
（前提文書 §8.3）[CODE]。**本監査では A の `q` の entropy を評価しない。**

---

## 5. Ideal exact-posterior EM Q（B）

### 5.1 定義 [ASSUMPTION]

**理想化する:** `q_old,K(Z) = p_K(Z | D, θ_old)` を **exact posterior** とする。
これは current implementation とは別物である（§4.5）[CODE]。

```
Q_K(θ, θ_old) = E_{q_old,K}[ log p_K(D,Z|θ) ]                                  …(5.1)
```

### 5.2 2 項分解 [DERIVED]

(3.1) を代入して

```
Q_K(θ,θ_old) = E_{q_old}[ ℓ_cond,K(Z,θ) ] + E_{q_old}[ log p_K(Z) ]
             = E_{q_old}[ ℓ_cond,K(Z,θ) ] − (m/2)·log 2π − (1/2)·E_{q_old}[ ‖Z‖²_F ]     …(5.2)
```

`−2` 倍すると

```
−2 Q_K = −2 E_{q_old}[ℓ_cond,K] + m·log 2π + E_{q_old}[‖Z‖²_F]                  …(5.3)
```

### 5.3 A との差 — 決定的な 1 点 [DERIVED]

(5.3) と (4.1)+(4.3) を比べる。

| 成分 | A（current） | B（ideal exact-EM Q） |
|---|---|---|
| Gaussian 正規化 | `+ m·log 2π` = `+nK·log 2π` | `+ m·log 2π` = `+nK·log 2π`。**同一** |
| latent norm | `+ nK`（`scale_Z` により**定数**） | `+ E_{q_old}[‖Z‖²_F]`。**posterior 依存の確率量。一般に `nK` ではない** |
| 当てはまり | `L=5` の単一実現平均 | exact posterior 期待値 |

> **したがって「B でも 425.6816/K が固定 cost になる」とは言えない。**
> **B で自動的に固定されるのは 275.6816/K（正規化部分）だけである** [DERIVED]。

`E_{q_old}[‖Z‖²_F]` の値は事前分布では `nK` だが（§3.2）、posterior では一般に異なる。
posterior が縮小（shrinkage）すれば `nK` より小さくなりうるし、
データが `Z` を大きく引っ張れば大きくなりうる。**どちらになるかは未解決** [UNRESOLVED]。

### 5.4 「Q の中の `−(m/2)log 2π` だけを取り出して O(nK) penalty と解釈してよいか」

**単独では解釈してよくない** [DERIVED]。理由は §6 で示す:
`Q` と observed likelihood の関係には posterior entropy `H(q)` が必ず入り、
**`H(q)` にも同じ型の正規化項が現れうる**からである。
ただし一般の posterior は Gaussian とは限らないので、**「必ず相殺する」とは言わない**（§6.4・§7）。

---

## 6. Exact variational identity

### 6.1 恒等式 [DERIVED]

`q` を `p_θ(Z|D)` に絶対連続な任意の密度、`θ` を任意のパラメータとする。

```
log p_θ(D) = E_q[log p_θ(D)]                                  （被積分関数が Z に依存しない）
           = E_q[ log p_θ(D,Z) − log p_θ(Z|D) ]               （ベイズの定理）
           = E_q[log p_θ(D,Z)] − E_q[log q(Z)] + E_q[log q(Z) − log p_θ(Z|D)]
```

`Q(θ;q) := E_q[log p_θ(D,Z)]`、`H(q) := −E_q[log q]`、`KL := KL(q ‖ p_θ(Z|D))` と置いて

> **`log p_θ(D) = Q(θ;q) + H(q) + KL(q ‖ p_θ(Z|D))`**   …(★)

**(★) は任意の `q`・任意の `θ` について EXACT である**（前提文書 §9.2 で確認済み。
`q` を作った `θ_old` と評価に使う `θ_new` が異なっても恒等式は壊れない）[DERIVED]。

### 6.2 `θ = θ_old` の特別な場合 [DERIVED]

`q = p_{θ_old}(Z|D)` を代入すると `KL = 0` なので

> **`log p_{θ_old}(D) = Q(θ_old, θ_old) + H( p_{θ_old}(Z|D) )`**   …(6.1)

すなわち

```
−2 Q(θ_old,θ_old) = −2 log p_{θ_old}(D) + 2·H( p_{θ_old}(Z|D) )                …(6.2)
```

> **complete-data Q を使うことは、observed log-likelihood に `+2H(posterior)` を加えることと厳密に同値である。**
> `H` は微分エントロピーなので**負にもなりうる**。posterior が集中していれば `H < 0` となり、
> このとき `Q > log p(D)` である [DERIVED]。

### 6.3 K 依存性の所在 [DERIVED]

(6.2) の右辺で `K` に依存するのは `log p(D)` と `H(posterior)` の両方である。
`H` は `m = nK` 次元分布の微分エントロピーなので、**一般に `O(nK)` のスケールを持つ**。

### 6.4 「必ず相殺する」とは言わない [UNRESOLVED]

`Q` に含まれる `−(m/2)log 2π` と `H(posterior)` の正規化成分が相殺するかどうかは、
**posterior の形に依存する。** 一般の posterior は Gaussian ではない。
Gaussian を仮定した特別ケースを §7 で別扱いする。

---

## 7. Gaussian-posterior special case

### 7.1 仮定 [ASSUMPTION]

```
p_K(Z | D, θ) ≈ N( Ẑ_K , H_K^{-1} ) ,     m = nK
```

**これは理想化である。** 本モデルの posterior が Gaussian である保証はない
（`η^Y_ij = w_0 + w z_i^T z_j` は `Z` について双線形であり、`h_K` は `Z` について凹ではない。§9.4）[DERIVED]。

**`scale_Z` の条件 `‖Z‖²_F = nK` はここでは仮定しない。** 使う場合は §7.4 で別ケースとして明示する。

### 7.2 entropy [DERIVED]

`N(μ,Σ)` の微分エントロピーは `(1/2) log{ (2πe)^m det Σ }` なので

```
H(q) = (m/2)·log(2πe) + (1/2)·log det(H_K^{-1})
     = (m/2)·(1 + log 2π) − (1/2)·log det H_K                                   …(7.1)
```

### 7.3 (6.1) への代入 — 途中式を省略せず [DERIVED]

`q = N(Ẑ_K, H_K^{-1})` のもとで

```
E_q[ ‖Z‖²_F ] = ‖Ẑ_K‖²_F + tr( H_K^{-1} )                                       …(7.2)
```

（`E‖Z‖² = ‖μ‖² + tr Σ`。）これを (5.2) に入れて

```
Q(θ,θ) = E_q[ℓ_cond] − (m/2)log 2π − (1/2)‖Ẑ_K‖²_F − (1/2) tr(H_K^{-1})          …(7.3)
```

(6.1) に (7.1) と (7.3) を足す:

```
log p(D) ≈ E_q[ℓ_cond] − (m/2)log 2π − (1/2)‖Ẑ_K‖² − (1/2)tr(H_K^{-1})
                       + (m/2)(1 + log 2π) − (1/2) log det H_K
```

`−(m/2)log 2π + (m/2)log 2π = 0` なので **Gaussian 正規化はここで消える**:

```
log p(D) ≈ E_q[ℓ_cond] + m/2 − (1/2)‖Ẑ_K‖² − (1/2)tr(H_K^{-1}) − (1/2) log det H_K   …(7.4)
```

`−2` 倍:

```
−2 log p(D) ≈ −2 E_q[ℓ_cond] + ‖Ẑ_K‖² + tr(H_K^{-1}) − m + log det H_K            …(7.5)
```

### 7.4 追加仮定を置いた場合（別ケースとして明示）[ASSUMPTION]

**追加仮定:** `E_q[‖Z‖²_F] = ‖Ẑ_K‖² + tr(H_K^{-1}) = m`（＝ `scale_Z` が A で強制している条件と同型）。

このとき (7.5) は

```
−2 log p(D) ≈ −2 E_q[ℓ_cond] + m − m + log det H_K = −2 E_q[ℓ_cond] + log det H_K   …(7.6)
```

> **この追加仮定の下でのみ、`nK(1 + log 2π)` の全体が entropy 側と打ち消し合い、
> 残るのは curvature 項 `log det H_K` だけになる** [DERIVED under ASSUMPTION]。
>
> **しかし A にはこの entropy 項が存在しない。**
> A は `scale_Z` で `E‖Z‖² = nK` の側だけを強制し、相殺相手である `H(q)` を criterion に持っていない。
> **これが A と「entropy まで含めた基準」との構造的な差である** [DERIVED]。

（前提文書 §10.2 の block-diagonal surrogate による同型の代数と整合する。
そこでは `Σ_q = blockdiag(A_i^{-1})` を仮定して `−(1/2)Σ_i log det A_i` が残った。
本節は global Gaussian 版であり、`log det H_K` が残る。両者の関係は §9.3 で扱う。）

---

## 8. Original-paper conditional criterion（C）

### 8.1 定義 [PAPER]

原論文 Eq.(16):

```
L = ∏_{i=1}^{n} p(x_i | F, z_i, μ_x, Σ) · { ∏_{i≠j} p(y_ij | z_i, z_j, w_0, w) }^{1/2}
```

すなわち

```
ℓ_cond,K = log p_K(D | Z, θ) = log p(X|Z,θ_X) + log p(Y|Z,θ_Y)                   …(8.1)
```

**`p(Z)` を直接含まず、`Z` を積分もしない。**

Eq.(26):

```
BIC = −2 log L + [ (k+1)d − k(k−1)/2 ] log n                                     [PAPER]
```

本文が "the true dimension k* affects the likelihood function of L from Eq. (16)" と明記しており、
**Eq.(26) の当てはまり項は Eq.(18) の `Q` ではなく Eq.(16) の `L` である** [PAPER]。

### 8.2 どの `Z` を代入するか [UNRESOLVED]

**原論文本文からは特定できない。**（最終 `Z` の plug-in か、`L` サンプル平均か、別の推定量か。）
前提文書 §12.5 の限定をそのまま保持する。本監査もこれを解かない。

### 8.3 K 依存項 [DERIVED]

C には次のいずれも**直接は入らない**:

- `log p(Z)`（したがって `m log 2π` も `‖Z‖²` も入らない）
- posterior entropy
- Laplace determinant

したがって C の K 依存項は **conditional fit の改善** と **明示的 parameter penalty** の 2 つだけである。

```
Δ[−2 ℓ_cond,K + P_K log n] = Δ[−2 ℓ_cond] + (d − K)·log n                        …(8.2)
```

### 8.4 「K を増やせば必ず改善する」とは言わない [UNRESOLVED]

- `sup_{Z,θ} ℓ_cond,K` は `K` について非減少である（`K+1` 次元の `Z` は `K` 次元を部分集合として含みうる）
  ——ただしこれは `Z` と `θ` を自由に最大化した場合の話である [INFERENCE]。
- **実際に評価されるのは最大化点ではなく EM が返した `Ẑ` である。**
  さらに `{M_K}` が入れ子であるかは既存監査でも**スケッチ段階**にとどまる
  （`k_selection_theory_audit_20260822.md` §9.4）。
- **したがって「conditional fit は必ず単調改善する」と断定しない。**

### 8.5 有限標本証拠としての `S_cf` [ARTIFACT + DERIVED]

fixed 系列（n=150, d=15, K_GEN=3, K=1..9, 10 trials × 3 シナリオ）の一次 CSV
`fixed_exp1_bic_k1to9_summary.csv` を恒等分解した結果:

- `S_cf(K) := −2·(Q_strict − lnpZ_det) + P_K log n`（Z prior を除いた診断量）は
  **A/B/C 全シナリオ・全ステップで K について単調減少**し、
  argmin は **trial 単位 30/30 で範囲上端（K=8 または 9）**へ移る。

> **この事実の使い方 [DERIVED]:**
> 「**conditional-like な fit に明示 parameter penalty だけを付けた診断量では、
> この有限標本条件で K=3 を止められなかった**」という**有限標本証拠**としてのみ使う。
>
> **`S_cf` は C そのものではない。** 相違点 3 つ:
> (i) 罰則が Eq.(26) の `(k+1)d − k(k−1)/2` ではない（Poisson-X / Bernoulli-X で不一致）、
> (ii) Gaussian-Y の `−0.5 log 2π`（ペアごと）が欠落している（K 非依存だが絶対値が違う）、
> (iii) 代入する `Z` が EM のサンプル平均であり、原論文の手続きは未確定（§8.2）。
> **「原論文 BIC の再現」と呼んではならない。**

---

## 9. Laplace approximation of observed likelihood（D）

### 9.1 定義 [DERIVED]

```
ℓ_obs,K(θ) := log p_K(D|θ) = log ∫ exp{ h_K(Z;θ) } dZ ,     h_K(Z;θ) = log p_K(D,Z|θ)
```

`Ẑ_K := argmax_Z h_K(Z;θ)`、`H_K := −∇²_Z h_K(Ẑ_K;θ)`（`m×m`、mode では正定値）、`m = nK`。

### 9.2 Laplace 近似 [DERIVED]

`h_K` を `Ẑ_K` のまわりで 2 次まで展開する。mode では `∇h_K(Ẑ_K) = 0` なので

```
h_K(Z) ≈ h_K(Ẑ_K) − (1/2)(Z − Ẑ_K)^T H_K (Z − Ẑ_K)
```

Gaussian 積分 `∫ exp{−(1/2) u^T H u} du = (2π)^{m/2} (det H)^{−1/2}` より

```
∫ exp{h_K(Z)} dZ ≈ exp{ h_K(Ẑ_K) } · (2π)^{m/2} · (det H_K)^{−1/2}
```

対数をとって

> **`ℓ_obs,K(θ) ≈ h_K(Ẑ_K;θ) + (m/2)·log(2π) − (1/2)·log det H_K`**   …(9.1)

### 9.3 大域 Hessian `H_K` の構造 [DERIVED]

`h_K(Z) = −(m/2)log2π − (1/2)Σ_i‖z_i‖² + Σ_{i,l} log p_X(x_il|η^X_il) + Σ_{i<j} log p_Y(y_ij|η^Y_ij)`。

指数型分布族の標準形 `log p(y|η) = [T(y)η − A(η)]/φ + c` を使う。
`r_ij := (y_ij − A_Y'(η^Y_ij))/φ_Y`、`V_Y(η) := A_Y''(η)/φ_Y`、`V_X` も同様とする。

**勾配:**

```
∂h_K/∂z_i = −z_i + F^T [ (x_i − A_X'(F z_i)) / φ_X ] + w · Σ_{j≠i} r_ij · z_j        …(9.2)
```

（`Σ_{i<j}` を `z_i` で微分すると、ペア `{i,j}` は 1 回ずつ現れるので `Σ_{j≠i}`。**extra 1/2 は出ない。**）

**対角ブロック `(i,i)`:** `η^Y_ij` は `z_j` を固定すれば `z_i` について**線形**なので、
2 階微分に残差項は現れない。

```
[H_K]_{ii} = I_K + F^T diag( V_X(F z_i) ) F + w² · Σ_{j≠i} V_Y(η^Y_ij) · z_j z_j^T      …(9.3)
```

**非対角ブロック `(i,j)`, `i ≠ j`:** (9.2) の第 3 項を `z_j` で微分すると

```
∂/∂z_j [ w r_ij z_j ] = w r_ij I_K + w z_j · (∂r_ij/∂z_j)^T
                      = w r_ij I_K − w² V_Y(η^Y_ij) · z_j z_i^T

⇒ [H_K]_{ij} = − w r_ij I_K + w² V_Y(η^Y_ij) · z_j z_i^T                              …(9.4)
```

#### 9.3.1 コードの `A_i` との関係 [CODE + DERIVED]

(9.3) を `model_dual_expfam_fixed.py:101-113` と比較する:

```python
term1 = (1.0 / var_z) * np.eye(k)                       # :101   → I_K
term2 = F.T @ (F * var_x_i[:, None])                    # :107   → F^T diag(V_X) F
term3 = (w ** 2) * (Z.T @ np.diag(var_y) @ Z)           # :113   → w² Σ_{j≠i} V_Y z_j z_j^T
```

> **fixed 系列の `A_i` は、大域 Hessian `H_K` の `(i,i)` 対角ブロックと厳密に一致する** [DERIVED]。
> 同様に fixed 系列の gradient（`:77` の `term3 = w * (Z.T @ residual_y)`）は (9.2) 第 3 項と厳密に一致する。

#### 9.3.2 historical 0.5 系列との関係 [CODE + DERIVED]

historical 系列（`model_dual_expfam.py:159`, `:200`）は同じ位置に `0.5` を持つため、
(9.2)(9.3) とは一致しない。

> **これは KI-001 の既存の判断（採用式＝1/2 なしを正とする）と整合する独立な観察である。**
> **本監査は KI-001 の判断を変更しない。** また root `CLAUDE.md` §5 の限定
> 「0.5 が掛かるのは Y 側項のみで Z 事前分布項・X 側項には掛かっていないため、
> Newton 方向が全体として正しいとは断定できない」も解除しない。

#### 9.3.3 `H_K ≠ blockdiag(A_i)` と Fischer の不等式 [DERIVED]

(9.4) の第 1 項 `−w r_ij I_K` は残差 `r_ij` の符号により**不定符号**である。したがって:

1. **`H_K` は blockdiag ではない。** `log det H_K ≠ Σ_i log det A_i`。
2. **`A_i ⪰ I_K` は成り立つ**（(9.3) の第 2・第 3 項は半正定値、`var_z = 1`）が、
   **`H_K ⪰ I_m` は保証されない**（不定符号の非対角項があるため）。
   したがって **`log det H_K ≥ 0` も保証されない。**
3. mode では `H_K ≻ 0` なので **Fischer の不等式**（正定値行列の行列式は対角ブロックの行列式の積以下）が使えて

> **`log det H_K ≤ Σ_i log det A_i`**   …(9.5)

（前提文書 §10.3 の診断量 `s_laplace_post = S_cf + Σ_i log det A_i^post` に現れる `Σ_i log det A_i` は、
この意味で `log det H_K` の**上界側**の量である。ただし `A_i^post` は mode `Ẑ_K` ではなくサンプル `Z^(l)` で
評価されており、fit 項も別物なので、**`s_laplace_post` が D の近似であるとは言えない** [UNRESOLVED]。）

### 9.4 Laplace 近似が妥当かどうか [UNRESOLVED]

以下はいずれも未解決であり、**D を「正しい基準」として扱わない理由**である。

| # | 論点 | 状態 |
|---|---|---|
| 1 | 積分次元 `m = nK` が `n` とともに増える | 通常の Laplace 近似は妥当な漸近近似ではない可能性（Shun & McCullagh 1995。`k_selection_theory_audit_20260822.md` §8.5 経由）[UNRESOLVED] |
| 2 | `h_K` は `Z` について凹ではない | `η^Y_ij` が双線形。mode の一意性・多峰性は未検証 [UNRESOLVED] |
| 3 | `θ` 固定での `Z` 積分に `O(K)` 対称性はあるか | `θ` を固定すると `log p(X\|Z)` が回転不変でないので、`Z` のみの積分では零モードは生じない [DERIVED]。ただし `rank F < K` のときは別（§17-U6） |
| 4 | `H_K ≻ 0` の成立範囲 | mode では定義上成立。それ以外では保証なし [DERIVED] |

---

## 10. Cancellation of Gaussian normalization

### 10.1 完全導出（途中式を省略しない）[DERIVED]

(9.1) を `−2` 倍する:

```
−2 ℓ_obs,K(θ) ≈ −2 h_K(Ẑ_K;θ) − m·log(2π) + log det H_K                          …(10.1)
```

`h_K` の定義に (3.1) を代入:

```
h_K(Ẑ_K;θ) = ℓ_cond,K(Ẑ_K,θ) + log p_K(Ẑ_K)
           = ℓ_cond,K(Ẑ_K,θ) − (m/2)·log(2π) − (1/2)·‖Ẑ_K‖²_F
```

したがって

```
−2 h_K(Ẑ_K;θ) = −2 ℓ_cond,K(Ẑ_K,θ) + m·log(2π) + ‖Ẑ_K‖²_F                        …(10.2)
```

(10.2) を (10.1) に代入:

```
−2 ℓ_obs,K(θ) ≈ [ −2 ℓ_cond,K(Ẑ_K,θ) + m·log(2π) + ‖Ẑ_K‖²_F ] − m·log(2π) + log det H_K
```

`+ m log(2π) − m log(2π) = 0`。よって

> ### **`−2 ℓ_obs,K(θ) ≈ −2 ℓ_cond,K(Ẑ_K,θ) + ‖Ẑ_K‖²_F + log det H_K`**   …(10.3)
>
> **Gaussian prior の正規化定数 `m log 2π` は、Laplace 体積因子 `(2π)^{m/2}` と厳密に相殺し、
> 最終式に一切残らない** [DERIVED]。

### 10.2 符号の再確認 [DERIVED]

| ステップ | 符号 |
|---|---|
| `∫ e^{−(1/2)u^T H u} du = (2π)^{m/2}(det H)^{−1/2}` | `log` をとると `+(m/2)log2π − (1/2)log det H` |
| `×(−2)` | `−m log2π + log det H` |
| `log p(Ẑ) = −(m/2)log2π − (1/2)‖Ẑ‖²` を `×(−2)` | `+m log2π + ‖Ẑ‖²` |
| 和 | `m log2π − m log2π = 0`、`+‖Ẑ‖² + log det H` |

**符号ミスなし。** §10.4 の独立チェックも一致する。

### 10.3 「marginalize すれば penalty が消える」とは言わない [DERIVED]

(10.3) の右辺に残る `‖Ẑ_K‖²_F` と `log det H_K` は**どちらも `K` に依存する**。

- `‖Ẑ_K‖²_F`: `Ẑ_K ∈ R^{n×K}` なので次元とともに項数が増える。`O(nK)` のスケール。
- `log det H_K`: `H_K` は `m×m = nK×nK`。固有値の対数和なので `O(nK)` のスケール。

> **消えたのは Gaussian 正規化定数 `m log 2π` だけである。**
> **K 依存の complexity 自体は `‖Ẑ_K‖²` と `log det H_K` として残る。**

### 10.4 §7（Gaussian posterior 版 B）との独立整合チェック [DERIVED]

(7.5) は

```
−2 log p(D) ≈ −2 E_q[ℓ_cond] + ‖Ẑ_K‖² + tr(H_K^{-1}) − m + log det H_K
```

であった。ここで同じ Gaussian 近似の下で `E_q[ℓ_cond]` を 2 次まで展開する:

```
E_q[ℓ_cond] ≈ ℓ_cond(Ẑ_K) + (1/2)·tr( H_K^{-1} ∇²_Z ℓ_cond(Ẑ_K) )
```

`ℓ_cond = h_K − log p_K(Z)` かつ `∇²_Z log p_K(Z) = −I_m` なので
`∇²_Z ℓ_cond(Ẑ_K) = ∇²_Z h_K(Ẑ_K) + I_m = −H_K + I_m`。したがって

```
E_q[ℓ_cond] ≈ ℓ_cond(Ẑ_K) + (1/2)·tr( H_K^{-1}(−H_K + I_m) )
            = ℓ_cond(Ẑ_K) + (1/2)( −m + tr H_K^{-1} )

⇒ −2 E_q[ℓ_cond] ≈ −2 ℓ_cond(Ẑ_K) + m − tr H_K^{-1}
```

(7.5) に代入:

```
−2 log p(D) ≈ [ −2 ℓ_cond(Ẑ_K) + m − tr H_K^{-1} ] + ‖Ẑ_K‖² + tr H_K^{-1} − m + log det H_K
            = −2 ℓ_cond(Ẑ_K) + ‖Ẑ_K‖² + log det H_K
```

> **(10.3) と完全に一致する。** 2 つの独立な経路（entropy 経由と Laplace 積分経由）が同じ式に到達しており、
> **符号ミスがないことの内部チェックになっている** [DERIVED]。

---

## 11. K-dependent term comparison

### 11.1 K→K+1 差分の構造 [DERIVED]

**A（current Q-based）:**

```
Δ C_Q = Δ D_fit + n·log 2π + n + (d − K)·log n                                   …(4.2)
```

**B（ideal exact-EM Q、追加の罰則なし）:**

```
Δ[−2 Q_K] = Δ[ −2 E_{q*}[ℓ_cond] ] + n·log 2π + Δ E_{q*}[‖Z‖²_F]                 …(5.3)
```

**C（conditional、Eq.(26) の罰則付き）:**

```
Δ[−2 ℓ_cond + P_K log n] = Δ[−2 ℓ_cond] + (d − K)·log n                          …(8.2)
```

**D（Laplace-marginal、`Z` 積分のみ。`θ` 罰則は §12 で別段階）:**

```
Δ[−2 ℓ_obs] ≈ Δ[−2 ℓ_cond(Ẑ)] + Δ‖Ẑ_K‖²_F + Δ log det H_K                       …(11.1)
                    └ fit ┘      └ latent norm ┘  └ posterior volume / curvature ┘
```

### 11.2 4 基準の項別対照表 [DERIVED]

| 基準 | Gaussian prior normalization | latent norm | entropy / volume | explicit θ penalty |
|---|---|---|---|---|
| **A** current Q-based | **あり。** `+nK·log 2π`（n=150 で 275.6816/K）。決定論的 | **あり。** `+nK`（n=150 で 150/K）。**`scale_Z` により決定論化** | **なし**（明示項なし。posterior geometry は sampled `Z` 経由で fit に間接影響） | `+P̂_K·log n`。`(d−K)log n` |
| **B** ideal exact-EM Q | **あり。** `+nK·log 2π`。決定論的 | **あり。** `+E_{q*}[‖Z‖²_F]`。**データ依存・非決定論** | `Q` 単体にはなし。**observed likelihood との関係では `H(q*)` が必須**（(6.1)） | 内在しない（付けるなら別途） |
| **C** conditional (Eq.16) | **なし** | **なし** | **なし** | 原論文 Eq.(26): `[(k+1)d − k(k−1)/2]log n` |
| **D** Laplace observed (`Z` 積分) | **相殺して消える**（§10.1） | **あり。** `+‖Ẑ_K‖²_F`。**データ依存・非決定論** | **あり。** `+log det H_K` | **第 2 段階の話**（§12）。`Z` 積分からは出ない |

### 11.3 A と D の差を 1 行で [DERIVED]

```
A の K 追加 cost（fit を除く）:  n·log 2π + n            = 定数 425.6816   (n=150)
D の K 追加 cost（fit を除く）:  Δ‖Ẑ_K‖²_F + Δ log det H_K = データ依存
```

- **A の第 1 項に対応するものが D には存在しない**（相殺して消えている）。
- **A の第 2 項は D では固定されない**（`scale_Z` がないので `‖Ẑ_K‖²` は自由）。
- **D の curvature 項に対応するものが A には存在しない。**

> **したがって A の `nK(1+log 2π)` と D の `‖Ẑ_K‖² + log det H_K` は、
> 「同じ complexity の別表現」ではない** [DERIVED]。

### 11.4 両者が数値的に一致するための条件 [DERIVED]

A の cost が D の `Z` 段階 complexity と一致するには、`K` あたり

```
‖Ẑ_{K+1}‖² − ‖Ẑ_K‖² ≈ n          かつ        Δ log det H_K ≈ n·log 2π
```

が必要である。第 2 条件は「`H_K` の固有値の対数平均が `log 2π ≈ 1.8379`、すなわち固有値の幾何平均が
`2π ≈ 6.2832` である」ことを意味する。

> **この条件が成り立つ理論的理由は存在しない** [DERIVED]。
> **成り立つかどうかは測定されていない** [UNRESOLVED]。
> （`log det H_K` はどの実験でも記録されていない。前提文書 §13.3。）

---

## 12. Two-stage marginalization: Z vs θ

### 12.1 2 段階を絶対に混同しない [DERIVED]

```
第 1 段階（Z の周辺化）:   p(D | θ, K) = ∫ p(D, Z | θ, K) dZ
第 2 段階（θ の周辺化）:   p(D | K)    = ∫ p(D | θ, K) · π(θ | K) dθ
```

- **§9-§10 の Laplace 近似は第 1 段階のみである。** 積分次元は `m = nK`。
- **Schwarz BIC の `p_K log N` は第 2 段階の Laplace 近似から出る概念である。** 積分次元は `dim θ`。

第 2 段階を正則条件の下で Laplace 近似すると（一般論）:

```
−2 log p(D|K) ≈ −2 log p(D | θ̂_K, K) + p_K·log N + O(1)                          …(12.1)
```

`N` は実効標本数。**本モデルで `N` が何であるべきかは未解決**（`n` か `n(n−1)/2` か、その中間か。
`k_selection_theory_audit_20260822.md` §8.3、Krivitsky & Kolaczyk 2015 経由）[UNRESOLVED]。

### 12.2 current code が両段階に対して何をしているか [CODE + DERIVED]

`C_Q(K) = −2 Q_strict + P̂_K log n` は次を同時に含んでいる:

| 成分 | どの段階の量か |
|---|---|
| `−2 Q_strict`（`log p(Z)` を含む complete-data 量） | **第 1 段階より手前**（`Z` を積分していない）。(6.2) より `−2 log p(D\|θ) + 2H(posterior)` に相当 |
| `+P̂_K log n` | **第 2 段階**の形（parameter-count × log n） |

> **すなわち current criterion は、第 1 段階を実行しないまま第 2 段階の形式の罰則を付けている** [DERIVED]。
> 第 1 段階固有の K 依存項（`‖Ẑ_K‖²`、`log det H_K`）は含まれず、
> 代わりに `Z` を積分しないことに固有の項（`nK(1+log 2π)`）が入っている。
>
> **これが「正しい 2 段階近似に対応するか」は未証明である** [UNRESOLVED]。

### 12.3 第 2 段階の罰則についても正則条件は満たされない [UNRESOLVED]

(12.1) の導出には `θ` の識別可能性と Fisher 情報の非退化が必要だが、本モデルでは

- `O(K)` 回転による大域的非識別性（`F` に対する回転自由度）
- `rank F < K` のときの階数欠損
- `w = 0` での Y 側 Fisher 情報の消失

が既知である（`k_selection_theory_audit_20260822.md` §9-§10）。
`− k(k−1)/2` はこの回転自由度に対応する次元 `dim O(K)` と数値が一致するが、
**原論文はこの帰属を書いていない**（前提文書 §7.5）[PAPER + UNRESOLVED]。

---

## 13. Meaning of the current 425.6816/K term

### 13.1 Q1. 425.6816/K は Gaussian prior から必然的に生じるのか、`scale_Z` との組合せで生じるのか？

**両方である。ただし成分ごとに答えが違う** [DERIVED]。

```
425.6815600 = 275.6815600 (= n·log 2π)  +  150.0000000 (= n)
```

| 成分 | 出所 | `scale_Z` に依存するか |
|---|---|---|
| `n·log 2π` = **275.6816** | Gaussian prior の**正規化定数** `−(m/2)log 2π`。`Z` の値に依存しない | **依存しない。** `log p_K(Z)` を criterion に含め `var_z` を固定する限り、A でも B でも必ず現れる |
| `n` = **150.0000** | latent norm `−(1/2)‖Z‖²_F` | **`scale_Z` に依存する。** `(1/L)Σ_l‖Z^(l)‖² = nK` が強制されて初めて決定論的な `n/K` になる |

> **「425.6816/K は Gaussian prior から必然」は誤り。**
> **正しくは「275.6816/K は prior の正規化定数から必然、150/K は `scale_Z` との組合せで生じる」** [DERIVED]。

### 13.2 Q2. ideal exact-EM Q でも同じ 425.6816/K が固定 cost になるのか？

**ならない** [DERIVED]。

(5.3) より B の K 依存 prior 寄与は

```
+ n·log 2π  +  Δ E_{q*}[‖Z‖²_F]
```

- 第 1 項 `n·log 2π = 275.6816` は **A と同一で固定**。
- 第 2 項は **posterior 依存の確率量**であり、`n = 150` に固定される理由はない。
  事前分布のもとでは `E[‖Z‖²] = nK` だが、posterior では一般に異なる（§5.3）。

さらに B を observed likelihood と比較するには (6.1) により `H(q*)` が必要であり、
`H(q*)` にも `K` 依存がある（§6.3）。
**「Q の中の `−(m/2)log 2π` だけを取り出して penalty と読む」ことは、B においても正当化されない** [DERIVED]。

### 13.3 Q3. observed likelihood を Laplace 近似すると Gaussian normalization はどうなるか？

**Laplace 体積因子 `(2π)^{m/2}` と厳密に相殺して消える** [DERIVED、§10.1 で完全導出]。

```
−2 ℓ_obs,K ≈ −2 ℓ_cond,K(Ẑ_K) + ‖Ẑ_K‖²_F + log det H_K
```

`m log 2π` は最終式に現れない。§10.4 で独立経路からも確認した。

### 13.4 Q4. Laplace marginal では K の complexity は主に何として残るか？

**`‖Ẑ_K‖²_F` と `log det H_K` の 2 つである**（`Z` 段階）。
`θ`-level parameter penalty は**別段階**（第 2 段階）から出る（§12）。

| 候補 | 判定 |
|---|---|
| `‖Ẑ_K‖²` | **残る。** latent norm / prior energy。`O(nK)` スケール、データ依存 |
| `log det H_K` | **残る。** posterior volume / curvature。`O(nK)` スケール、データ依存。`log det H_K ≤ Σ_i log det A_i`（(9.5)） |
| `θ`-level parameter penalty | **`Z` 積分からは出ない。** 第 2 段階の量（§12.1） |
| Gaussian normalization `m log 2π` | **残らない**（相殺、§10.1） |
| その他 | 積分次元が `n` とともに増えるため Laplace 近似自体の妥当性が未解決（§9.4-1）[UNRESOLVED] |

### 13.5 Q5. current の 425.6816/K を observed marginal likelihood の Laplace complexity の代理と解釈できる理論的根拠はあるか？

**現時点で理論的根拠はない** [DERIVED + UNRESOLVED]。

根拠がないことの構造的理由 3 点:

1. **`n·log 2π` 部分（275.6816/K）は D では相殺されて存在しない。**
   D に対応物のない項が A に入っている。
2. **`n` 部分（150/K）は D では `Δ‖Ẑ_K‖²` に対応するが、A では `scale_Z` により固定され、
   D では固定されない。** 固定値がデータ依存量の代理になる保証はない。
3. **D の `log det H_K` に対応する項が A には存在しない**（前提文書 §11.1 項 5）。

**一致が成り立つ条件は §11.4 に明示した**（`Δ‖Ẑ_K‖² ≈ n` かつ `H_K` の固有値の幾何平均 `≈ 2π`）。
**この条件が成り立つ理論的理由はなく、成否は測定もされていない** [UNRESOLVED]。

> **ただし「代理として不適である」ことを示したわけでもない。**
> 示したのは「**代理であるという主張を支える導出が存在しない**」ことである。

### 13.6 Q6. fixed 30/30 はどの基準についての証拠なのか？

**A についての、n=150・d=15・K_GEN=3 という 1 点での有限標本証拠である** [ARTIFACT + DERIVED]。

| 基準 | fixed 30/30 は証拠になるか |
|---|---|
| **A** current Q-based | **なる。** ただし n の 1 点・K_GEN の 1 点・3 family 組合せに限る |
| **B** ideal exact-EM Q | **ならない。** exact posterior は使われていない（§4.5） |
| **C** conditional | **ならない。** 派生診断量 `S_cf` が「明示 penalty だけでは止まらなかった」ことを示すだけで、`S_cf` は C そのものではない（§8.5） |
| **D** Laplace observed | **ならない。** `‖Ẑ_K‖²` も `log det H_K` も測定されていない |

**consistency の証拠ではない**（n を振った K 選択実験は fixed 系列に存在しない）。

---

## 14. What fixed 30/30 establishes

### 14.1 確定していること [ARTIFACT + DERIVED]

1. `fixed_exp1_bic_k1to9_bestk_frequency.csv`: A/B/C 全シナリオで `k=3` が 10/10（合計 30/30）。
   条件は n=150, d=15, K_GEN=3, K=1..9, L=5, num_iter=8, 全 270 fits `success=True`。
2. 恒等分解により、**K > 3 側で過大 K を止めている主要な direct 項は Z prior 由来の
   `n(1+log 2π) = 425.6816` である**（前提文書 §11.2-§11.3）。
   K ≥ 3 の全ステップ・全 3 シナリオで `Δ penalty`(35〜60) < `|Δ(−2 fit)|`(73〜181) < 425.6816。
3. `S_cf`（Z prior 除去）の argmin は **trial 単位 30/30 で範囲上端**へ移る。

### 14.2 確定していないこと [UNRESOLVED]

- **どの基準が「正しい」かについては何も言っていない。** §13.6 のとおり A についての証拠にすぎない。
- **425.6816 が D の complexity の良い代理であるかどうかは検証されていない**（§13.5）。
- **n 依存性は測定されていない。**
- **consistency は示していない。**

### 14.3 呼称の維持

`KNOWN_ISSUES.md` の既存表現（「明示した人工設定において、歴史的に `BIC` と呼ばれてきた Q ベース基準が
`K = 3` を選択した」「Schwarz BIC の妥当性・一般的な true-K recovery・model-selection consistency の
いずれも意味しない」）を維持する。本監査はこれを強めない。

---

## 15. Per-column implications

**本節は拡張可能性の指摘のみであり、詳細導出は行わない。**
**per-column は `experimental/model_dual_expfam_percolumn.py` の prototype であり、
本文採用不可である**（root `CLAUDE.md` §3）[CODE]。

### 15.1 joint density は成立する [DERIVED]

列 `l` の family `g_l` が既知なら

```
log p(X | Z) = Σ_{l=1}^{d} log p_{g_l}( X_{·l} | η_{·l} ) ,     η_il = f_l^T z_i        …(15.1)
```

これは列について因子分解された正当な確率モデルである。
**「family が混在しているから確率モデルが成立しない」わけではない** [DERIVED]。

### 15.2 Laplace Hessian への寄与は加法的 [DERIVED]

(9.3) の X 側項は、per-column では

```
[H_K]_{ii} ⊃ Σ_{l=1}^{d} V_{g_l}(η_il) · f_l f_l^T = F^T diag( V_{g_1}(η_i1), …, V_{g_d}(η_id) ) F
```

となる。**各列の `A''_{g_l}(η_il)/φ_{g_l}` が加算的に寄与するだけであり、構造は変わらない。**

### 15.3 したがって per-column で変わるもの [DERIVED]

- `log det H_K`（curvature 項）の値
- conditional fit `ℓ_cond,K` の構造と**スケール**

### 15.4 別問題として残るもの [UNRESOLVED]

**family ごとに base measure が異なるため、異なる family 構成の間で fit 値の絶対値を比較することは
別途の正当化を要する。** これは `reports/research_scope/basic_vs_percolumn_bic_scope_audit_20260906.md` の
scope に属し、**本監査では扱わない。**

---

## 16. Established facts

すべて [CODE] / [ARTIFACT] / [PAPER] / [DERIVED] のいずれかで裏づけられる。

| # | 内容 | 根拠 |
|---|---|---|
| F1 | `log p_K(Z) = −(nK/2)log 2π − (1/2)‖Z‖²_F`（`var_z=1`） | [CODE] `utils_expfam.py:318-321` |
| F2 | 事前分布のもとで `E[‖Z‖²_F] = nK` | [DERIVED] §3.2 |
| F3 | A の K 追加 cost は `Δ D_fit + n·log 2π + n + (d−K)log n` | [CODE + DERIVED] §4.3 |
| F4 | `425.6815600 = 275.6815600 + 150.0000000`（n=150）。前者は prior 正規化、後者は `scale_Z` 由来 | [DERIVED] §13.1 |
| F5 | 恒等式 `log p_θ(D) = Q(θ;q) + H(q) + KL` は任意の `q`・任意の `θ` で EXACT | [DERIVED] §6.1 |
| F6 | `θ = θ_old` かつ `q = ` exact posterior のとき `KL = 0`、`log p(D) = Q + H(posterior)` | [DERIVED] §6.2 |
| F7 | したがって `−2Q = −2 log p(D) + 2H(posterior)`。`H` は負にもなりうるので `Q > log p(D)` もありうる | [DERIVED] §6.2 |
| F8 | Laplace 近似: `ℓ_obs,K ≈ h_K(Ẑ_K) + (m/2)log 2π − (1/2)log det H_K` | [DERIVED] §9.2 |
| F9 | **`−2 ℓ_obs,K ≈ −2 ℓ_cond,K(Ẑ_K) + ‖Ẑ_K‖²_F + log det H_K`。`m log 2π` は厳密に相殺する** | [DERIVED] §10.1 |
| F10 | §7（Gaussian posterior 経由）と §10（Laplace 積分経由）は独立に同一式へ到達する | [DERIVED] §10.4 |
| F11 | Gaussian posterior 仮定の下では `H(q) = (m/2)(1+log 2π) − (1/2)log det H_K` であり、`Q` の `−(m/2)log2π` と正規化部分が相殺する | [DERIVED under ASSUMPTION] §7.3 |
| F12 | さらに `E_q‖Z‖² = m` を仮定すると `nK(1+log2π)` 全体が相殺し `log det H_K` だけが残る | [DERIVED under ASSUMPTION] §7.4 |
| F13 | **fixed 系列の `A_i` は大域 Hessian `H_K` の `(i,i)` 対角ブロックと厳密に一致する** | [CODE + DERIVED] §9.3.1 |
| F14 | `H_K` の非対角ブロックは `−w r_ij I_K + w² V_Y(η_ij) z_j z_i^T` であり、第 1 項は不定符号 | [DERIVED] §9.3 |
| F15 | したがって `H_K ≠ blockdiag(A_i)`、`H_K ⪰ I_m` は保証されない、`log det H_K ≥ 0` も保証されない | [DERIVED] §9.3.3 |
| F16 | Fischer の不等式より `log det H_K ≤ Σ_i log det A_i`（`H_K ≻ 0` の下で） | [DERIVED] §9.3.3 |
| F17 | 原論文 Eq.(26) の当てはまり項は Eq.(16) の `L` であり `p(Z)` を含まない | [PAPER] §8.1 |
| F18 | C には prior 項・entropy・determinant のいずれも直接入らない | [DERIVED] §8.3 |
| F19 | fixed 系列で `S_cf` は全ステップ単調減少し、argmin は trial 単位 30/30 で範囲上端へ移る | [ARTIFACT + DERIVED] §8.5 |
| F20 | Schwarz の `p_K log N` は `θ` についての第 2 段階 Laplace から出る。`Z` 積分からは出ない | [DERIVED] §12.1 |
| F21 | current criterion は第 1 段階を実行しないまま第 2 段階の形式の罰則を付けている | [CODE + DERIVED] §12.2 |
| F22 | per-column でも joint density は成立し、Hessian への寄与は列について加法的 | [DERIVED] §15.1-15.2 |

---

## 17. Assumptions / unresolved points

### 17.1 明示した仮定 [ASSUMPTION]

| # | 仮定 | 使った箇所 | 成立の保証 |
|---|---|---|---|
| S1 | `q_old,K = ` exact posterior | §5, §6.2 | **なし。** current implementation は exact posterior を使わない（§4.5） |
| S2 | `p(Z\|D,θ) ≈ N(Ẑ_K, H_K^{-1})`（global Gaussian posterior） | §7 | **なし。** `h_K` は `Z` について凹ではない（§9.4-2） |
| S3 | `E_q[‖Z‖²_F] = m`（`scale_Z` と同型の追加条件） | §7.4 のみ | **A の実装事実であり、B/D には持ち込んでいない** |
| S4 | Laplace 近似が `m = nK` 次元で妥当 | §9, §10 | **なし。** 積分次元が `n` とともに増える（§9.4-1） |
| S5 | `Ẑ_K` が一意な mode で `H_K ≻ 0` | §9, §10 | mode では定義上成立。一意性・多峰性は未検証 |

### 17.2 未解決 [UNRESOLVED]

| # | 問い |
|---|---|
| U1 | `Δ‖Ẑ_K‖²_F` と `Δ log det H_K` の実際の値・オーダー（どの実験でも未記録） |
| U2 | §11.4 の一致条件（`Δ‖Ẑ‖² ≈ n` かつ `H_K` 固有値の幾何平均 `≈ 2π`）が成り立つか |
| U3 | `m = nK` 次元での Laplace 近似の誤差評価 |
| U4 | `h_K` の mode の一意性・多峰性 |
| U5 | 第 2 段階の実効標本数 `N`（`n` か `n(n−1)/2` か中間か） |
| U6 | `rank F < K` のときの `H_K` の退化 |
| U7 | `{M_K}` の入れ子性（既存監査でもスケッチ段階） |
| U8 | 特異モデルにおける第 2 段階罰則の正しい形（`p_K/2` が正しい係数か） |
| U9 | 原論文 Experiment 2 がどの `Z` を Eq.(16) に代入したか |
| U10 | current の `q̄` の density / entropy（前提文書 §8.3.1 判定 4・6） |

### 17.3 standard BIC へ飛ばない [DERIVED]

**D が A より概念的に observed distribution へ近いとしても、
「standard Schwarz BIC が正しい」とは結論しない。** 少なくとも次が未解決である:

回転非識別性 / 階数欠損 / 特異潜在モデル / ネットワーク依存 / 実効標本数 / パラメータ次元 / 入れ子性
（U5-U8、`k_selection_theory_audit_20260822.md` §9-§10。Drton & Plummer 2017 は
"all the classical hidden or latent variable models are singular" と述べており、
特異モデルでは Schwarz の**導出**が無効になる。ただし同監査 §10.3 のとおり、
**特異性は一致性という結論を自動的に否定するものでもない**）。

**本監査の範囲は「`Z` を積分したら K 依存項がどう変わるか」までである。**

---

## 18. Provisional theoretical conclusions

**本節は理論的整理であり、criterion の採用判断ではない。**

### 18.1 結論 5 点

1. **`Z` の扱い方を変えると、K 依存項の「型」自体が変わる。**
   `Z` を complete-data として代入する（A・B）と Gaussian prior の正規化 `nK log 2π` が criterion に残るが、
   `Z` を積分する（D）とそれは Laplace 体積因子と**厳密に相殺して消える**（F9、§10.1）。
   **これは近似の粗さの問題ではなく、量の定義が違うことによる構造差である。**

2. **current の 425.6816/K は 2 成分に厳密に分解できる。**
   `275.6816/K`（`= n log 2π`）は prior の正規化定数から**必然的に**生じ、
   `150/K`（`= n`）は **`scale_Z` との組合せで初めて決定論化する**（F4、§13.1）。
   **「Gaussian prior だから必然」でも「`scale_Z` だけの産物」でもない。**

3. **`Z` を積分しても K の complexity は消えない。形が変わるだけである。**
   Laplace-marginal では `‖Ẑ_K‖²_F`（latent norm / prior energy）と
   `log det H_K`（posterior volume / curvature）として残る（F9、§10.3）。
   どちらも `O(nK)` スケールだが、**A の固定係数 `1 + log 2π` と違い、係数がデータ依存である。**

4. **A の cost が D の complexity の代理であるという導出は存在しない。**
   A には D にない項（`nK log 2π`）があり、D には A にない項（`log det H_K`）がある。
   一致条件は §11.4 に明示したが、**理論的理由はなく、測定もされていない**（F4、§13.5）。
   **ただし「代理として不適である」ことを示したわけでもない。** 示したのは根拠の不在である。

5. **current criterion は 2 段階の周辺化のうち第 1 段階を飛ばしている。**
   `−2 Q_strict` は `Z` 積分**前**の量、`+p̂ log n` は `θ` 積分（第 2 段階）の形である（F20・F21、§12）。
   第 1 段階固有の K 依存項（`‖Ẑ_K‖²`、`log det H_K`）は含まれず、
   代わりに「積分しないこと」に固有の項が入っている。
   **この組合せが正しい 2 段階近似に対応するかは未証明である。**

### 18.2 次に数式だけで進められること（実験を要さない）

| # | 課題 | 必要なもの |
|---|---|---|
| T1 | `log det H_K` の K 依存オーダーを、(9.3)(9.4) の構造から解析的に評価する | 追加実験不要。行列解析 |
| T2 | `H_K ≻ 0` の十分条件（残差項 `w r_ij` の大きさに対する条件）を書く | 追加実験不要 |
| T3 | `rank F < K` のときの `H_K` の退化と `Ẑ_K` の非一意性を整理する | 追加実験不要 |
| T4 | 第 2 段階について、`O(K)` 商の上で正則条件がどこまで回復するかを整理する | 追加実験不要 |

**本監査はこれらを行っていない。** また **T1-T4 は criterion の採用判断ではない。**

---

## 付録 A. 数値の再確認

```python
import math
n, d, K = 150, 15, 3
l2pi = math.log(2*math.pi)          # 1.8378770664093453
n*l2pi                              # 275.6815599614018   ← Gaussian prior normalization
n*1                                 # 150                 ← latent norm (scale_Z 由来)
n*(1+l2pi)                          # 425.6815599614018   ← A の Z prior 由来 K 追加 cost
(d-K)*math.log(n)                   # 60.127623529155066  ← 明示罰則の K 追加 cost
n*(1+l2pi) + (d-K)*math.log(n)      # 485.80918349055685
2*math.pi                           # 6.283185307179586   ← §11.4 の一致条件に現れる値
```

## 付録 B. 参照文書との関係

| 文書 | 関係 |
|---|---|
| `reports/research_scope/q_based_criterion_decomposition_audit_20260907.md`（修正版） | **前提。** 実装監査・一次 CSV 監査・原論文確認の結果を引き継ぎ、覆さない |
| 同 §10.2（block-diagonal surrogate） | 本監査 §7 は global Gaussian 版であり、同型の代数。`Σ_i log det A_i` と `log det H_K` の関係は本監査 §9.3.3 で (9.5) として与えた |
| 同 §9.4（5 段階表） | 本監査の A（段階 5）と B（段階 2/3）の区別に対応 |
| `reports/k_selection_theory/k_selection_theory_audit_20260822.md` | §8.3（実効標本数）・§9（識別可能性）・§10（特異性）を引用。**書き換えていない** |
| `KNOWN_ISSUES.md` KI-001 / KI-010 | **いずれの判断も変更しない。** §9.3.2 は KI-001 と整合する独立観察を追加しただけ |
| `EXPERIMENT_REGISTRY.md` | 追記していない（新実験を行っていないため） |
| `reports/research_scope/basic_vs_percolumn_bic_scope_audit_20260906.md` | §15.4 の family 間比較問題はそちらの scope。本監査では扱わない |
