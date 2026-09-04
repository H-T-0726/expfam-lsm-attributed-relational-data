# 真の潜在次元 K の定義と識別可能性 — 硬化版理論監査

**作成日:** 2026-09-04
**種別:** 理論監査（コード変更なし・EM 実行なし）
**baseline main:** `7e335602999977060208ce37ac8cdff8fedfa66e`
**数値検証:** `tools/research_audit/verify_identifiability_identities.py`（71 checks、verdict **PASS**、failure 0）

## Evidence label 凡例

| ラベル | 意味 |
|---|---|
| `[PRIMARY_SOURCE]` | 原論文そのもの |
| `[CONFIRMED_IN_REPOSITORY]` | 一次コードで確認 |
| `[DERIVED]` | 本監査で独立に導出（証明または証明スケッチ付き） |
| `[NUMERIC_CHECK]` | 上記スクリプトによる独立数値確認 |
| `[EMPIRICAL_EXISTING]` | 過去の既存 artifact |
| `[HYPOTHESIS]` | 仮説 |
| `[UNRESOLVED]` | 未解決 |

---

## 1. Executive verdict

先生の指摘 1〜5 に対する現時点の到達点。

| # | 指摘 | 到達点 |
|---|---|---|
| 1 | 生成モデルとして本当に成立しているか | **成立する（条件付き）**。canonical model は finite `n,d,K`、finite parameter、Gaussian 分散 > 0 の下で proper。ただし **Poisson-Y は proper でもモーメントが発散しうる**（§11）。さらに**現行 historical generator は canonical model の literal generator ではない**（§13、`[CONFIRMED_IN_REPOSITORY]`） |
| 2 | 「真の K」とは何か | **定義を確定した**。`K* = min{K : P0 ∈ M_K}` = 観測分布を表現できる最小潜在次元（§2）。**generator で Z を何列作ったか**とは別概念 |
| 3 | K_TRUE を設定しただけでは識別可能とは限らない | **正しい**。識別不能になる具体的条件を列挙し（§5）、Bernoulli-X で **反例を構成**した（§8） |
| 4 | 複数分布へ拡張したとき BIC の理論的性質は自明でない | **正しい**。現行 `calc_bic_dual` は Q3（観測データ周辺尤度）ではない（`[EMPIRICAL_EXISTING]` KI-010、`reports/k_selection_theory/paper_bic_reproduction_alignment_20260904.md`）。さらに **Gaussian-Y では M_K が入れ子でないことを証明**したので（§12）、入れ子性を前提とする漸近論はそのままでは使えない |
| 5 | 有限標本で選べることと n→∞ の一致は別 | **正しい**。本監査は population identifiability までしか到達しておらず、**推定量の consistency は `[UNRESOLVED]`**（§16、U6） |

**新しく証明できた主要命題は 3 つ**（§14）:

- **P1**: canonical Poisson-X では、population moment から `FF^T` を復元でき、`d ≥ K` かつ `rank(F)=K` なら `K` は population identifiable。
- **P2**: canonical Gaussian-Y では、**単一 dyad の周辺分布だけ**から `(K, w², σ_y²)` が決まる（`w ≠ 0` のとき）。
- **P3**: その帰結として **Gaussian-Y では `M_K ⊄ M_{K+1}`**（`w ≠ 0`）。すなわち K の族は入れ子でない。

**証明できなかった主要事項**（§16）: Bernoulli-X / Bernoulli-Y の一般 identifiability、Gaussian-X の Σ unknown の場合、Poisson-Y の一般 identifiability、そして **すべての criterion の n→∞ consistency**。

---

## 2. 「真の K」の定義

### 2.1 定義

パラメータ空間 `Theta_K`（`F ∈ R^{d×K}`, `w0, w ∈ R`, および family 依存の dispersion）に対し

```
p_{theta,K}(X, Y) = ∫ p(Z) p_{theta,K}(X | Z) p_{theta,K}(Y | Z) dZ
M_K = { p_{theta,K}(X, Y) : theta ∈ Theta_K }
```

真の観測分布を `P0` として

```
K* = min { K : P0 ∈ M_K }
```

と定義する。**`K*` は「観測分布 `P0` を表現できる最小の潜在次元」**である。`[DERIVED]`

### 2.2 この定義が意味を持つ条件

`min` が well-defined であるためには次で十分である。`[DERIVED]`

1. 候補集合 `{K : P0 ∈ M_K}` が空でない（少なくとも 1 つの K が `P0` を表現する）。
2. `K` は正整数上を動くので、空でない部分集合には最小元が必ず存在する（整列性）。

**重要:** この定義は `M_K ⊆ M_{K+1}`（入れ子性）を**必要としない**。`min` は整列性だけから存在する。したがって §12 で示すように族が入れ子でなくても `K*` の定義自体は有効である。`[DERIVED]`

ただし入れ子でない場合、`{K : P0 ∈ M_K}` は**単一の点**になりうる（Gaussian-Y ではまさにそうなる、§12）。そのとき `K*` は「最小」というより「唯一」の K である。

### 2.3 generator の K_TRUE と `K*` の区別

**必ず区別する。** `[DERIVED]`

| 記号 | 意味 |
|---|---|
| `K_TRUE` | synthetic generator が `Z` を何列作ったかという**手続き上の数** |
| `K*` | そうして作られた**観測分布**の最小潜在次元 |

一般に `K* ≤ K_TRUE` であり、等号は自明ではない。等号が破れる具体例:

- `rank(F) = r < K_TRUE` かつ `w = 0` のとき、X は `r` 次元の潜在構造しか持たず Y は潜在に依存しないので `K* ≤ r < K_TRUE`。
- Bernoulli-X で `d = 1` のとき、`K_TRUE` が何であっても観測分布は `Bern(1/2)^n` に一致するため `K*` は `K_TRUE` を復元しない（§8）。

**「人工データで K_TRUE を 5 に設定したから真の K は 5 である」とは書けない。** 実験計画では `K* = K_TRUE` となるよう **construction で保証**する必要がある（Phase 3 の clean generator 仕様の目的）。

---

## 3. Properness と identifiability の分離

混同しやすいので明示的に分ける。`[DERIVED]`

| 概念 | 主張内容 | 成立条件 |
|---|---|---|
| **properness** | `p(Z, X, Y)` が全確率 1 の確率測度である | finite `n, d, K`、finite parameters、Gaussian 分散 > 0、canonical link |
| **finite moments** | `E[Y^r] < ∞` 等 | family 依存。**Poisson-Y では追加条件が要る**（§11） |
| **identifiability** | 異なる `theta` が異なる `P_theta` を与える（または `K` が `P0` から決まる） | はるかに強い条件。§6〜§12 |

**properness は identifiability を含意しない。** Poisson-Y は `|w| ≥ 1/2` でも proper だが分散は無限大であり、Bernoulli-X は常に proper だが `d=1` では K を識別できない。

逆に **finite moments も identifiability を含意しない**。モーメントが有限でも、モーメント列が分布を一意に決めるとは限らない（モーメント問題）。本監査の識別性主張はすべて「モーメントから特定のパラメータ関数を復元できる」形であり、これは identifiability の**十分**条件として使っている（復元できれば異なるパラメータは異なるモーメントを持つ）。

---

## 4. 不変性（invariance）

`Z` の prior を `N(0, I_K)` に固定した canonical model において、観測分布を変えない変換を列挙する。`[DERIVED]`

`Q ∈ O(K)`（直交行列）に対し `z → Qz` とすると:

- prior 不変: `Qz ~ N(0, Q I Q^T) = N(0, I_K)`。
- X 側: `eta^X = f_l^T z`。`f_l → Q f_l` とすれば `(Qf_l)^T (Qz) = f_l^T z` で不変。すなわち `F → FQ^T`。
- Y 側: `eta^Y = w0 + w z_i^T z_j`。`(Qz_i)^T(Qz_j) = z_i^T z_j` で不変。

したがって **`O(K)` 回転は観測分布を変えない**。含まれる部分群:

| 変換 | 説明 |
|---|---|
| **rotation** | `F → FQ^T`, `Q ∈ SO(K)` |
| **sign flip** | 各潜在座標の符号反転（`Q = diag(±1)`、`det = ±1`） |
| **permutation** | 潜在座標の並べ替え（置換行列も直交行列） |

**scale は自由でない。** `z → cz` は prior を `N(0, c²I)` に変えるため、prior を `N(0,I)` に固定した時点で潜在スケールは固定される。これが「`N(0,I)` prior による scale fixation」である。`[DERIVED]`

ただし **X 側だけを見ると `F` と潜在スケールの間に残る不定性**があり、Y 側の `w` がそれを結び付けている:
`(F, w)` と `(cF, w/c²)` は `z → z/c` の再パラメータ化で同じ `eta^X`, `eta^Y` を与える — が、これも prior が `N(0,I)` に固定されているため許されない。よって **prior 固定下では `F` と `w` は個別に意味を持つ**。`[DERIVED]`

**重要:** これらの不変性は `F` の識別可能性を `FF^T` のレベルまでしか許さない。したがって **`K` の識別に使えるのは `F` そのものではなく `FF^T`（Gram 行列）とその rank である**。以降の議論はすべてこの形をとる。

---

## 5. 退化ケース — 「K が存在する」と「K を識別できる」の分離

`[DERIVED]`

| ケース | `K` は「存在」するか | 観測分布から識別できるか |
|---|---|---|
| `F = 0` | する（generator は K 列作った） | **不可**。X は潜在に依存しない。Y だけが情報を持つので `K*` は Y 側の識別性に完全に依存する |
| `w = 0` | する | **Y から不可**。`eta^Y = w0` は定数で潜在に依存しない。Gaussian-Y では `κ_4 = κ_6 = 0` となり §9.4 の復元公式が `0/0` になる |
| `F = 0` かつ `w = 0` | する | **完全に不可**。`X, Y` は潜在に依存せず `K* = 0`（あるいは K は全く決まらない） |
| `rank(F) = r < K` | する | **X からは高々 `r` まで**。`FF^T` の rank は `r` なので X 側は `r` しか見えない。残り `K-r` 次元は Y 側の `z_i^T z_j` にのみ現れる |
| `d < K` | する | **X からは高々 `d` まで**。`FF^T ∈ R^{d×d}` の rank は `d` 以下 |
| `w` が非常に小さい | する | **population では識別可（`w ≠ 0` なら）だが有限標本では実質不可**。復元公式 `K = κ_4/(6w⁴)` は `w→0` で数値的に不安定（分母が `w⁴`）。これは identifiability の問題ではなく **estimability / 情報量**の問題 |
| Bernoulli-X, `d = 1` | する | **不可**（§8 に反例） |
| Bernoulli-Y, `w0 = 0` | する | **edge density からは不可**（§10.2） |

**このセクションの要点:** 「潜在次元 K を持つ生成過程が存在する」ことと「観測分布から K が決まる」ことは別である。実験計画では **識別可能な領域に条件を置くこと自体が設計判断**であり、その判断を明示しなければならない。

---

## 6. Gaussian-X

### 6.1 モデルと population 共分散

```
z ~ N(0, I_K),  x | z ~ N(F z, Sigma)
```

周辺分布は `x ~ N(0, F F^T + Sigma)` であり

```
Cov(X) = F F^T + Sigma
```

`[DERIVED]`（証明: `Cov(x) = E[Cov(x|z)] + Cov(E[x|z]) = Sigma + F Cov(z) F^T = Sigma + FF^T`）

### 6.2 Sigma が既知の場合

`FF^T = Cov(X) − Sigma` が population から一意に決まる。したがって

```
rank(Cov(X) − Sigma) = rank(F F^T) = rank(F)
```

`rank(F) = K` かつ `d ≥ K` ならば **`K` は population identifiable**。`[DERIVED]`

### 6.3 Sigma が未知（対角）の場合 — 因子分析の識別問題

`Cov(X) = FF^T + Psi`（`Psi` 対角）という分解は**一般に一意でない**。これは古典的な因子分析の識別問題そのものである。

パラメータ数の必要条件（Ledermann 型）:

- 観測側の自由度: `d(d+1)/2`
- モデル側: `FF^T` の自由度 `dK − K(K−1)/2` ＋ `Psi` の `d`

必要条件は `d(d+1)/2 ≥ dK − K(K−1)/2 + d`、整理して `(d−K)² ≥ d + K`。`[DERIVED]`

具体例: `d = 2, K = 1` では `(2−1)² = 1 < 3` で**必要条件を満たさない** → 識別不能。

**したがって「Gaussian-X なら K は必ず識別できる」とは書かない。** `Sigma` 既知、または `Sigma` 対角で Ledermann 型条件＋generic 条件が成り立つ場合に限られる。一般の十分条件の完全な整理は `[UNRESOLVED]`。

---

## 7. Poisson-X — **P1: 証明された識別可能性**

### 7.1 モデル

```
z ~ N(0, I_K),   X_l | z ~ Poisson( exp(f_l^T z) ),  l = 1..d
```

各列は `z` の下で条件付き独立。canonical（unclipped）exp link を仮定。

### 7.2 一次モーメント

`f_l^T z ~ N(0, ||f_l||²)` なので Gaussian の MGF より

```
E[X_l] = E[ E[X_l | z] ] = E[ exp(f_l^T z) ] = exp( ||f_l||² / 2 )
```

したがって

```
||f_l||² = 2 log E[X_l]                                        ... (7.1)
```

`[DERIVED]` `[NUMERIC_CHECK]`（`poisson_x_gram_recovery`, rel err 0.0017）

### 7.3 交差二次モーメント（`l ≠ m`）

`l ≠ m` では `X_l ⊥ X_m | z` なので

```
E[X_l X_m] = E[ E[X_l|z] E[X_m|z] ]
           = E[ exp( (f_l + f_m)^T z ) ]
           = exp( ||f_l + f_m||² / 2 )
```

`||f_l+f_m||² = ||f_l||² + ||f_m||² + 2 f_l^T f_m` を代入して (7.1) と組み合わせると

```
f_l^T f_m = log( E[X_l X_m] / ( E[X_l] E[X_m] ) )              ... (7.2)
```

`[DERIVED]` `[NUMERIC_CHECK]`

**`l ≠ m` に限る理由:** `E[X_l²]` には Poisson 自身の分散項（`E[λ_l] + E[λ_l²]`）が混ざるため、対角成分は (7.1) の一次モーメント経由で取るほうが clean である。実際 (7.1) と (7.2) だけで Gram 行列全体が決まる。

### 7.4 命題 P1

> **命題 P1.** canonical Poisson-X（unclipped exp link）において、`d ≥ K` かつ `rank(F) = K` ならば、`X` の population 一次・交差二次モーメントから Gram 行列 `G = FF^T` が一意に決まり、
> ```
> K = rank(G)
> ```
> により **`K` は population identifiable** である。

**証明.** (7.1) が `G` の対角、(7.2) が非対角をすべて与えるので `G` は population moments の関数として一意に決まる。`G = FF^T` で `rank(G) = rank(F) = K`。∎ `[DERIVED]`

**モーメント有限性:** Gaussian の MGF は全実数で有限なので、`E[X_l]`, `E[X_l X_m]` は**任意の有限 `F` について常に有限**である。Poisson-Y（§11）と異なり、モーメント存在の追加条件は不要。`[DERIVED]`

### 7.5 P1 が要求する仮定（省略不可）

1. **canonical unclipped link**。`exp` に hard clip があると (7.1)(7.2) は崩れる。historical generator は `np.clip(eta, -20, 10)` を持つ（§13）。
2. `d ≥ K`。
3. `rank(F) = K`。
4. population moments（有限標本での rank 判定は別問題）。

**`K = rank(G)` は population の主張である。** 有限標本では `Ĝ` の固有値は 0 にならず、rank 判定には閾値が要る。**閾値を結果を見て決めてはいけない**ため、本研究では Phase 8 の C4 を「selected K を作る criterion」ではなく **structural spectrum diagnostic** として保存する。`[DERIVED]`

---

## 8. Bernoulli-X — **反例あり**

### 8.1 一次モーメントは情報を持たない

```
X_l | z ~ Bernoulli( sigmoid(f_l^T z) )
```

`U = f_l^T z ~ N(0, ||f_l||²)` は 0 対称。`sigmoid(u) + sigmoid(−u) = 1` より、任意の対称確率変数 `U` について `E[sigmoid(U)] = 1/2`。したがって

```
E[X_l] = 1/2      （K, f_l によらず）
```

`[DERIVED]` `[NUMERIC_CHECK]`（`bernoulli_x_mean`、K=1,3,5 × 4 列すべて rel err < 0.01）

### 8.2 反例（`d = 1`）

> **反例 C1.** `d = 1` の canonical Bernoulli-X において、任意の `K ≥ 1` と任意の `f_1 ∈ R^K \ {0}` に対し、観測分布は
> ```
> X_1 ~ Bernoulli(1/2)   （各行 iid）
> ```
> で**完全に一致する**。したがって `K` は観測分布から識別できない。

**証明.** §8.1 より `P(X_1 = 1) = E[sigmoid(f_1^T z)] = 1/2`。`d=1` なので観測は各行 1 ビットで、行は iid。よって観測分布は `K, f_1` に依存しない。∎ `[DERIVED]`

これは先生の指摘 3（「K_TRUE を設定しただけでは識別可能とは限らない」）の**具体的な裏付け**である。

### 8.3 `d > 1` の場合 — 必要条件のカウンティング

`d > 1` では pairwise モーメント `E[X_l X_m]` が情報を持つ。これは `(||f_l||², ||f_m||², f_l^T f_m)` の関数、すなわち Gram 行列 `G` の 3 成分の関数である。

パラメータ数の**必要**条件:

- 未知数: `G = FF^T`（rank ≤ K の PSD）の自由度 `dK − K(K−1)/2`
- pairwise モーメントの本数: `d(d−1)/2`（対角は §8.1 より情報ゼロ）

pairwise モーメントだけで `G` を決めるには少なくとも

```
d(d−1)/2  ≥  dK − K(K−1)/2                                     ... (8.1)
```

が必要。`[DERIVED]`

例: `K=3, d=6` では `15 ≥ 15` で境界ちょうど。`K=5, d=6` では `15 < 20` で **pairwise だけでは足りない**。`K=1, d=3` では `3 ≥ 3` で境界。

**(8.1) は必要条件にすぎない。** 満たされていても写像の単射性は別問題であり、満たされない場合は 3 次以上のモーメント（`E[X_l X_m X_p]` 等）が必要になる。

### 8.4 判定

- **「Bernoulli-X では K は識別できない」とは書かない。** 反例は `d = 1` に限る。
- **「Bernoulli-X なら K は識別できる」とも書かない。**
- `d > 1` での一般の識別可能性条件は **`[UNRESOLVED]`**。

---

## 9. Gaussian-Y — **P2: 単一 dyad からの完全復元**

### 9.1 `S = z_i^T z_j` の MGF

`z_i, z_j` は独立に `N(0, I_K)`。`z_j` で条件づけると `S | z_j ~ N(0, ||z_j||²)` なので

```
M_S(t) = E[ exp(t S) ] = E_{z_j}[ exp( t² ||z_j||² / 2 ) ]
```

`||z_j||² ~ chi²_K` の MGF は `(1−2u)^{−K/2}`（`u < 1/2`）。`u = t²/2` を代入して

```
M_S(t) = (1 − t²)^{−K/2},      |t| < 1                          ... (9.1)
```

`[DERIVED]` `[NUMERIC_CHECK]`（`S_mgf`, K=1,3,5 × t=0.3,0.5、rel err ≤ 0.011）

### 9.2 キュムラント

`kappa(t) = log M_S(t) = −(K/2) log(1 − t²) = (K/2) Σ_{m≥1} t^{2m}/m`

`kappa_r = r! × [t^r 係数]` より

```
kappa_2(S) = 2!  · (K/2)(1/1) = K
kappa_4(S) = 4!  · (K/2)(1/2) = 6K
kappa_6(S) = 6!  · (K/2)(1/3) = 120K
```

**奇数次キュムラントはすべて 0**（`kappa(t)` が `t` の偶関数）。`[DERIVED]` `[NUMERIC_CHECK]`（`S_kappa_{2,4,6}`、K=1,3,5、rel err ≤ 0.030）

**候補式 `kappa_2 = K`, `kappa_4 = 6K`, `kappa_6 = 120K` はすべて確認された。**

### 9.3 Y のキュムラント

```
Y_ij = w0 + w S + eps,     eps ~ N(0, sigma_y²)  独立
```

独立和のキュムラントは加法的、`eps` は `r ≥ 3` でキュムラント 0、スケーリングは `kappa_r(wS) = w^r kappa_r(S)`。したがって

```
kappa_2(Y) = K w² + sigma_y²
kappa_3(Y) = 0
kappa_4(Y) = 6 K w⁴
kappa_6(Y) = 120 K w⁶
```

`[DERIVED]`。**候補式 `kappa_4(Y) = 6Kw⁴`, `kappa_6(Y) = 120Kw⁶` を確認。**

### 9.4 復元公式と命題 P2

`w ≠ 0` のとき `kappa_4 ≠ 0` で

```
w²        = kappa_6(Y) / ( 20 · kappa_4(Y) )                    ... (9.2)
K         = kappa_4(Y) / ( 6 w⁴ )                               ... (9.3)
sigma_y²  = kappa_2(Y) − K w²                                   ... (9.4)
w0        = E[Y]                                                ... (9.5)
```

検算: (9.2) 右辺 `= 120Kw⁶ / (20·6Kw⁴) = w²` ✓、(9.3) 右辺 `= 6Kw⁴/(6w⁴) = K` ✓。`[DERIVED]` `[NUMERIC_CHECK]`（K=3, w=0.8, w0=0.4, σ_y=0.5、8M draws: ŵ²=0.6379 vs 0.64、K̂=3.019 vs 3、σ̂_y²=0.2438 vs 0.25）

> **命題 P2.** canonical Gaussian-Y において `w ≠ 0` ならば、**単一 dyad `(i,j)` の周辺分布だけ**から `(w0, w², K, sigma_y²)` が一意に決まる。すなわち `K` は population identifiable である。

**証明.** (9.5), (9.2), (9.3), (9.4) を順に適用すればよい。各ステップは population キュムラントの関数であり、`w ≠ 0` のとき well-defined。∎ `[DERIVED]`

### 9.5 何が識別**されない**か

| 項目 | 状況 |
|---|---|
| **`w` の符号** | **識別されない。** (9.1) は `t` の偶関数なので `S` は 0 対称であり、`wS` と `(−w)S` は同分布。単一 dyad 周辺からは `w²` までしか決まらない `[DERIVED]` |
| `w = 0` | `kappa_4 = kappa_6 = 0` で (9.2)(9.3) が `0/0`。`Y` は `N(w0, sigma_y²)` に退化し **K は識別不能** `[DERIVED]` |
| `F` | Y だけからは決まらない（§4 の `O(K)` 不変性、および `F` は Y に現れない） |
| `sigma_y` 未知の影響 | (9.4) で決まるので**問題にならない**。`w ≠ 0` である限り `sigma_y` が未知でも K の識別は妨げられない `[DERIVED]` |

**単一 dyad marginal vs network joint distribution:** P2 は**単一 dyad の周辺分布だけ**で成立するので、network 全体の同時分布を使う必要がない。これは Gaussian-Y の特別に強い性質であり、Bernoulli-Y（§10）とは対照的である。`[DERIVED]`

---

## 10. Bernoulli-Y — `[UNRESOLVED]`

### 10.1 モデル

```
Y_ij | Z ~ Bernoulli( sigmoid( w0 + w z_i^T z_j ) ),   i < j
```

### 10.2 単一 edge の周辺分布は情報が乏しい

`S` は 0 対称（§9.2）なので、`w0 = 0` のとき §8.1 と同じ議論で

```
P(Y_ij = 1) = E[ sigmoid(w S) ] = 1/2      （K, w によらず）
```

`[DERIVED]`

すなわち **`w0 = 0` では edge density は K の情報を全く持たない**。`w0 ≠ 0` なら `P(Y_ij=1) = E[sigmoid(w0 + wS)]` は `(w0, w², K)` に依存するが、これは 1 本の方程式であり 3 未知数を決められない。

**したがって Gaussian-Y と違い、Bernoulli-Y では単一 dyad 周辺だけでは K を識別できない。** `[DERIVED]`

### 10.3 network joint distribution — K 情報がどこに入るか

K の情報は**ノードを共有する複数 edge の同時分布**に入る。`Y_ij` と `Y_ik` は `z_i` を共有するため独立でない。

| motif | population 確率 | K 依存性 |
|---|---|---|
| edge | `E[sigmoid(w0 + w S_ij)]` | `w0 ≠ 0` のときのみ、`(w0,w²,K)` の 1 関数 |
| 2-star（`j–i–k`） | `E[ sigmoid(w0+wS_ij) · sigmoid(w0+wS_ik) ]`、`z_i` 共有 | `z_i` を通じた相関に K が入る |
| triangle | `E[ Π sigmoid(w0+w S) ]`、3 ノードすべて共有 | さらに高次 |

`z_i` で条件づけると `S_ij = z_i^T z_j` と `S_ik = z_i^T z_k` は独立で、それぞれ `N(0, ||z_i||²)`。したがって 2-star 確率は

```
E_{z_i}[ g(||z_i||²)² ],   g(v) = E_{U~N(0,v)}[ sigmoid(w0 + wU) ]
```

と書け、`||z_i||² ~ chi²_K` を通じて **K は `chi²_K` の分布形として入る**。`[DERIVED]`

これは「K の情報は `||z_i||²` の分布に入っている」という構造的な理解を与える。edge marginal では `||z_i||²` が積分されて潰れるが、共有ノードを持つ motif では潰れない。

### 10.4 判定

- 上の構造から **K が joint distribution に情報を残すことは確か**だが、`g` が sigmoid の Gaussian 平均で閉形式を持たないため、`chi²_K` の族が motif 確率の族として単射かどうかを示せていない。
- **一般 identifiability theorem は `[UNRESOLVED]`。**
- **「Bernoulli-Y では K は識別できない」とは書かない**（示していない）。
- **「Bernoulli-Y でも K は識別できる」とも書かない**（示していない）。

---

## 11. Poisson-Y — モーメント存在条件

### 11.1 モデルと `E[λ^r]`

```
lambda_ij = exp( w0 + w S ),   Y_ij | Z ~ Poisson(lambda_ij)
```

(9.1) より

```
E[lambda^r] = E[ exp( r w0 + r w S ) ] = exp(r w0) · M_S(r w)
            = exp(r w0) · ( 1 − r² w² )^{−K/2}                  ... (11.1)
```

`[DERIVED]`。**候補式 `E[λ^r] = exp(r w0) M_S(rw)` を確認。**

`M_S` は `|t| < 1` でのみ有限なので

```
E[lambda^r] < ∞   ⟺   | r w | < 1                               ... (11.2)
```

`[DERIVED]`。**候補条件 `|rw| < 1` を確認。**

### 11.2 Y のモーメント

Poisson の階乗モーメントは `E[Y(Y−1)···(Y−r+1)] = E[λ^r]` なので、`Y` の `r` 次モーメントの有限性は `E[λ^r]` の有限性と同値である。したがって

| 量 | 有限となる条件 |
|---|---|
| `E[Y]`（平均） | `|w| < 1` |
| `Var(Y)`（分散、`E[Y²]` 要） | **`|w| < 1/2`** |
| `E[Y³]` | `|w| < 1/3` |
| `E[Y^r]` | `|w| < 1/r` |

`[DERIVED]` `[NUMERIC_CHECK]`（`poisson_y_moment_existence`、K=1,3,5 × w グリッド）

**候補条件「mean finite: `|w| < 1`」「variance finite: `|w| < 1/2`」を確認。**

### 11.3 **historical default `w = 0.5` は分散発散の境界そのもの**

`[CONFIRMED_IN_REPOSITORY]` `expfam/src/data_generator_expfam.py` の `_Y_DEFAULTS`:

```python
"poisson":   dict(w0=0.0,  w=0.5),
```

`w = 0.5` を (11.2) に入れると `|2w| = 1.0`、すなわち **`< 1` を満たさない**。`E[λ²] = (1 − 4·0.25)^{−K/2} = 0^{−K/2} = ∞`。

> **観察 O1.** canonical（unclipped）Poisson-Y に historical default `w = 0.5` を入れると、**平均は有限だが分散は発散する**（境界ちょうど）。

`[DERIVED]`

**これは properness の問題ではない。** 分布自体は proper であり、サンプルは生成できる。問題は

1. 二次モーメントが存在しないため、**分散・相関・CLT 的な議論が使えない**。
2. 有限標本の標本分散は `n` とともに発散し、**安定しない**。

historical generator が実際には有限のデータを出しているのは、`np.clip(eta, -20, 10)` の hard clip が上側を切っているためである（§13）。すなわち **historical Poisson-Y のデータは、clip がなければ二次モーメントを持たない分布からの clip 済みサンプルである。** `[CONFIRMED_IN_REPOSITORY]` + `[DERIVED]`

**含意:** 本研究の clean generator では Poisson-Y を使う場合、`|w| < 1/2`（分散有限）を **generator gate として強制**すべきである。Phase 3 仕様に反映する。

### 11.4 階乗モーメントからの識別可能性（部分的）

`|w| < 1/3` なら `a_r = E[λ^r]`（`r=1,2,3`）がすべて有限。`b_r = log a_r = r w0 − (K/2) log(1 − r² w²)` から `w0` を消去すると

```
R(w²) = ( 2 b_1 − b_2 ) / ( 3 b_1 − b_3 )
```

は `w²` のみの関数になる。`R` が `(0, 1/9)` 上で狭義単調なら `w²` が決まり、続いて `K`, `w0` が決まる。

`[NUMERIC_CHECK]`: グリッド 4000 点で **狭義単調**（`R` は `0.3333 → 0.0313` に単調減少）。

> **判定.** 数値的には単調だが**解析的な単調性の証明はしていない**。したがって
> 「`|w| < 1/3` の canonical Poisson-Y では `(w0, w², K)` が階乗モーメントから決まる」は
> **`[HYPOTHESIS]` + 数値的裏付けあり、証明は `[UNRESOLVED]`**。

---

## 12. `M_K` と `M_{K+1}` の入れ子性 — **P3**

### 12.1 素朴な埋め込みが失敗する理由

K 次元モデルを K+1 次元で表そうとして `z'_i = (z_i, u_i)`、`u_i ~ N(0,1)` iid とする。

- **X 側**: `F' = [F, 0]` とすれば `eta^X = F' z' = F z` で完全に一致する。**X 側だけなら埋め込める。** `[DERIVED]`
- **Y 側**: `eta^Y = w0 + w (z_i^T z_j + u_i u_j)`。余分な項 `w u_i u_j` を消すには `w = 0` にするしかないが、`w` は**全潜在座標に共通のスカラー**なので、`u` 成分だけを止めることができない。`[DERIVED]`

**この観察だけでは「非入れ子」の証明にならない**（別のパラメータ `(F', w0', w')` が補償する可能性を排除していない）。Gaussian-Y ではそれを排除できる。

### 12.2 命題 P3（Gaussian-Y）

> **命題 P3.** canonical Gaussian-Y において、`w ≠ 0` なる `P ∈ M_K` は `M_{K+1}` に属さない。したがって
> ```
> M_K ⊄ M_{K+1}      （Gaussian-Y、w ≠ 0）
> ```

**証明.** `P ∈ M_K` を `(w0, w, sigma_y)`、`w ≠ 0` で与えられるものとする。§9.3 より
`kappa_4(P) = 6 K w⁴`, `kappa_6(P) = 120 K w⁶`。

いま `P ∈ M_{K+1}` と仮定し、その表現を `(w0', w', sigma_y')` とする。同じ §9.3 を `K+1` に適用すると
`kappa_4(P) = 6(K+1) w'⁴`, `kappa_6(P) = 120(K+1) w'⁶`。

(9.2) より `w'² = kappa_6/(20 kappa_4) = 120Kw⁶/(20·6Kw⁴) = w²`。
これを (9.3) に入れると `K+1 = kappa_4/(6 w'⁴) = 6Kw⁴/(6w⁴) = K`。

`K + 1 = K` は矛盾。よって `P ∉ M_{K+1}`。∎ `[DERIVED]` `[NUMERIC_CHECK]`（`km1_nesting_gaussian_y`、K ∈ {1,2,3,5,7} × w ∈ {0.3,0.8,1.5} すべてで `recovered_K = K`、`≠ K+1`）

### 12.3 P3 の含意 — model selection にとって重大

`[DERIVED]`

1. **`M_K` の族は入れ子ではない**（Gaussian-Y の場合、`w ≠ 0` の部分で互いに素）。
2. したがって **「K を大きくすれば必ず当てはまりが良くなる」という単調性は population レベルでは成立しない**。
3. **入れ子性を前提とする漸近論（尤度比検定の `chi²` 近似、Schwarz BIC の標準的導出、Wilks の定理）はそのままでは適用できない。** これは先生の指摘 4 に対する具体的な理由付けである。
4. ただし**有限標本の最尤当てはまりは K とともに単調に改善しうる**（パラメータ数が増えるため）。population の非入れ子性と有限標本の当てはまり単調性は別物である。

### 12.4 他の family での入れ子性

| family | 判定 |
|---|---|
| Gaussian-Y, `w ≠ 0` | **非入れ子（証明済み、P3）** |
| Gaussian-Y, `w = 0` | 退化。すべての K が同じ `N(w0, sigma_y²)` を与えるので、この点では全 `M_K` が交わる |
| X 側のみ（Y なし） | **入れ子**（`F' = [F, 0]` で埋め込める）`[DERIVED]` |
| Bernoulli-Y | **`[UNRESOLVED]`**。§12.1 の障害は同じだが、補償パラメータの不存在を示せていない |
| Poisson-Y | **`[UNRESOLVED]`**。§11.4 の階乗モーメント論法が使えれば P3 と同型の証明になるが、単調性が未証明 |

**「K と K+1 は完全に non-nested」とは書かない。** 証明できたのは Gaussian-Y の `w ≠ 0` の場合だけである。

---

## 13. historical generator と canonical model の乖離 `[CONFIRMED_IN_REPOSITORY]`

`expfam/src/data_generator_expfam.py` を直接読んで確認した事実のみを記す。**historical の数値結果を無効化するものではない**（§16）。

| # | 箇所 | 実装 | canonical model との差 |
|---|---|---|---|
| G1 | `_generate_base` L.122-123, `generate_dual_data` L.282-283 | `Z = rng.normal(...); Z = normalize_zscore(Z, axis=0)` | **`Z` は iid `N(0,I_K)` ではない。** 列ごとに標本平均 0・標本 SD 1 に強制されるため行間に依存が入る。§9 の `S` の分布論（`M_S(t) = (1−t²)^{−K/2}`）は厳密には成立しない |
| G2 | `_generate_base` L.127-129, `generate_dual_data` L.288-290 | `F[i,:] = F[i,:]/||F[i,:]|| * sqrt(1 − sigma[i,i])` | **`F` は自由パラメータでない。** 全行が `||f_l||² = 1 − uniq`（既定 0.9）に固定される。Poisson-X なら (7.1) より `E[X_l] = exp(0.45)` が全列共通になる |
| G3 | `_generate_base` L.132-133, `generate_dual_data` L.297-298 | `X = Z @ F.T + noise; X = normalize_zscore(X, axis=0)` | **Gaussian-X は `N(Fz, Sigma)` ではない。** 生成後に列 z-score されるため §6.1 の `Cov(X) = FF^T + Sigma` が成立しない。返り値の `F`・`sigma` と最終的な `X` が literal に対応しない |
| G4 | `generate_poisson_data` L.80, `generate_dual_data` L.304 (X), L.319 (Y) | `np.clip(eta, -20, 10)` | **Poisson の hard clip。** `lambda ≤ e^10 ≈ 22026` に silent に切られる。§7 の (7.1)(7.2) と §11 の (11.1) はいずれも unclipped link を前提とするため、clip が効く領域では成立しない |
| G5 | `generate_dual_data` L.233 | `sigma_x_true: float = 0.1` | **宣言されているが一度も使われない。** Gaussian-X の雑音共分散は `sigma_x = np.diag(np.full(d, uniq))` で `uniq` から作られる。`sigma_x_true` を変えても生成データは変わらない |
| G6 | Bernoulli 各所 | `np.clip(eta, -500, 500)` | **数値的に無害。** `sigmoid(±500)` は倍精度で既に 0/1 に飽和しており、モデルを変えない。G4 とは性質が異なるので同列に扱わない |
| G7 | Gaussian-Y | `rng.normal(0.0, sigma_y_true, ...)` | numpy の第 2 引数は **標準偏差**。`sigma_y` = SD で `CLAUDE.md` の規約と整合。**問題なし** |

### 13.1 判定

- **historical の数値結果を削除・無効化しない。** それらは「当該 generator が生成したデータ上での観測」として有効である。
- ただし **「canonical model から well-specified に生成した実験」という強い解釈は取れない。** G1・G3 により、推定器が仮定するモデルと生成過程が一致していない（mild misspecification）。
- **G5 は API の不整合**であり、`sigma_x_true` を指定した過去の呼び出しがあれば、その意図と実際の生成条件が食い違っている。historical 実験の再解釈が必要かは個別確認が要る（本監査の範囲外、`[UNRESOLVED]`）。
- **historical generator は変更しない**（Phase 3 で forward-only の別モジュールを作る）。

---

## 14. 証明された命題の一覧

| ID | 命題 | 条件 | 根拠 |
|---|---|---|---|
| **P1** | Poisson-X で `FF^T` が population moments から復元でき、`K = rank(FF^T)` は population identifiable | canonical unclipped link、`d ≥ K`、`rank(F)=K` | §7.4 `[DERIVED]` `[NUMERIC_CHECK]` |
| **P2** | Gaussian-Y で単一 dyad 周辺から `(w0, w², K, sigma_y²)` が一意に決まる | `w ≠ 0` | §9.4 `[DERIVED]` `[NUMERIC_CHECK]` |
| **P3** | Gaussian-Y で `M_K ⊄ M_{K+1}`（族は入れ子でない） | `w ≠ 0` | §12.2 `[DERIVED]` `[NUMERIC_CHECK]` |
| **P4** | `O(K)` 回転・符号反転・座標置換は観測分布を変えない。`N(0,I)` prior によりスケールは固定 | — | §4 `[DERIVED]` |
| **P5** | Gaussian-X で `Sigma` 既知なら `K = rank(Cov(X) − Sigma)` | `rank(F)=K`, `d ≥ K` | §6.2 `[DERIVED]` |
| **P6** | Poisson-Y のモーメント有限性は `E[Y^r] < ∞ ⟺ |w| < 1/r` | canonical unclipped link | §11.2 `[DERIVED]` `[NUMERIC_CHECK]` |
| **P7** | Bernoulli-X / Bernoulli-Y（`w0=0`）の一次モーメントは K の情報を持たない（常に 1/2） | `Z` が 0 対称 | §8.1, §10.2 `[DERIVED]` `[NUMERIC_CHECK]` |
| **O1** | historical Poisson-Y の default `w = 0.5` は分散発散の境界そのもの | — | §11.3 `[DERIVED]` + `[CONFIRMED_IN_REPOSITORY]` |

---

## 15. 反例

| ID | 内容 | 節 |
|---|---|---|
| **C1** | Bernoulli-X, `d = 1`: 任意の `K`・任意の `f_1 ≠ 0` で観測分布が `Bern(1/2)` に一致。K は識別不能 | §8.2 |
| **C2** | `w = 0`（任意の family-Y）: `Y` が潜在に依存せず、Y 側から K の情報が完全に消える | §5 |
| **C3** | Gaussian-X, `d=2, K=1`, `Sigma` 未知対角: Ledermann 型必要条件 `(d−K)² ≥ d+K` を満たさず識別不能 | §6.3 |
| **C4** | Bernoulli-Y, `w0 = 0`: edge density が `K, w` によらず 1/2 | §10.2 |

---

## 16. 未解決事項 `[UNRESOLVED]`

| ID | 内容 | 何が足りないか |
|---|---|---|
| U1 | Bernoulli-X（`d > 1`）の一般 identifiability | pairwise モーメント写像の単射性。カウンティング条件 (8.1) は必要条件にすぎない |
| U2 | Bernoulli-Y の一般 identifiability | motif 確率の族が `chi²_K` について単射かどうか。`g(v) = E[sigmoid(w0+wU)]`, `U~N(0,v)` に閉形式がない |
| U3 | Gaussian-X（`Sigma` 未知対角）の十分条件の完全な整理 | 因子分析の識別可能性の一般理論の適用。generic identifiability と everywhere identifiability の区別 |
| U4 | Poisson-Y の identifiability | §11.4 の `R(w²)` 単調性の解析的証明 |
| U5 | Bernoulli-Y / Poisson-Y の `M_K` 非入れ子性 | P3 と同型の論法が使えるか |
| U6 | **すべての criterion の n→∞ consistency** | 本監査は population identifiability までしか到達していない。推定量の一致性は別問題であり、非入れ子性（P3）により標準的な BIC 漸近論も使えない。**先生の指摘 5 は未解決のまま** |
| U7 | 有限標本での rank 判定閾値 | P1 は population の主張。標本 Gram の固有値から K を決める閾値を、結果を見ずに事前に決める方法が確立していない |
| U8 | G5（`sigma_x_true` 未使用）が historical 実験の解釈に与える影響 | どの過去実験が `sigma_x_true` を明示指定したかの網羅確認 |
| U9 | `K* = K_TRUE` を保証する construction 条件の完全性 | Phase 3 で「full-rank F・`w≠0`・`d≥K`」を課すが、これが `K*=K_TRUE` の**十分**条件であることは family ごとに要確認 |

---

## 17. model selection への含意

`[DERIVED]` を明示的に述べる。

1. **入れ子性を前提とする漸近論は使えない**（P3）。Schwarz BIC の標準的導出は、モデルが入れ子であることそのものより「真のモデルが候補族に含まれ、局所漸近正規性が成り立つ」ことを使うが、`M_K` が互いに素な多様体の族である場合、K を跨いだ比較の漸近的正当化は自明でない。

2. **現行 `calc_bic_dual` は観測データ周辺尤度（Q3）を使っていない**（`[EMPIRICAL_EXISTING]` KI-010、`reports/k_selection_theory/paper_bic_reproduction_alignment_20260904.md`）。したがって「Schwarz BIC」と呼ばない、という既存の運用は本監査でも支持される。

3. **held-out 予測スコアは `K*` を推定していない。** held-out score は「未知の Y を予測するのに最適な K」を選ぶ。P2 が示すように `K*` は分布の高次キュムラント（4 次・6 次）に符号化されており、**予測損失の最小化とは別の量**である。両者が一致する保証はない。`[DERIVED]`

   → これは Phase 8 の C1 の解釈上、決定的に重要である。**「held-out で K_TRUE が選ばれた」ことは「K_TRUE が識別された」ことを意味しない。**

4. **識別可能性が成り立たない領域では、どんな criterion も K を当てられない。** C1（Bernoulli-X, d=1）や C2（w=0）では、あらゆる推定手続きが原理的に失敗する。実験でこれらの領域を避けるのは設計判断であり、明示すべきである。

5. **Poisson-Y を使う実験では `|w| < 1/2` を課さないと、分散が定義されない分布の上で二乗誤差的な議論をすることになる**（O1）。

---

## 18. Claim ledger

### 書いてよい（条件を明示すれば）

| 主張 | 必須の限定 |
|---|---|
| canonical model は proper probability model として定義できる | finite `n,d,K`、finite parameters、Gaussian 分散 > 0。**Poisson-Y はモーメント有限性が別条件**（P6） |
| Poisson-X では、明示した条件下で `K` は population identifiable である | canonical unclipped link、`d ≥ K`、`rank(F)=K`、population moments。**有限標本の rank 判定は別問題**（U7） |
| Gaussian-Y では、明示した条件下で `K` は単一 dyad 周辺から population identifiable である | `w ≠ 0`。**`w` の符号は識別されない** |
| Gaussian-Y では `M_K` は入れ子でない | `w ≠ 0`。**他 family は未証明**（U5） |
| Bernoulli-X では `d = 1` のとき `K` は識別不能である（反例あり） | `d = 1` に限る。**`d > 1` の一般結論ではない** |
| historical generator は canonical model の literal generator ではない | G1〜G5 の具体的差分を挙げる。**historical 結果を無効とは言わない** |
| historical Poisson-Y の default `w=0.5` は分散発散の境界にある | canonical unclipped link での話。**historical データは clip 済み** |

### 書いてはいけない

- 「BIC は理論的に正しい」
- 「held-out なら真の K を必ず選べる」
- 「Poisson-X なら常に K が識別可能」（unclipped link・`d≥K`・`rank(F)=K` が要る）
- 「Bernoulli では K は識別できない」（反例は `d=1` のみ）
- 「K-selection consistency を証明した」（U6）
- 「n を増やせば必ず K_TRUE に収束する」（U6）
- 「clean generator で全問題を解決した」
- 「historical generator のせいで過去結果は無効」
- 「K と K+1 は完全に non-nested」（Gaussian-Y `w≠0` のみ証明）
- 「真の K は generator の K_TRUE である」（§2.3）

---

## 19. 数値検証の再現

```
python tools/research_audit/verify_identifiability_identities.py --out <path>
```

- seed `20260904` 固定、tolerance は実行前に `TOLERANCES` として宣言（**結果を見て変更していない**）。
- 2026-09-04 実行: **71 checks / failure 0 / verdict PASS**。
- `--fast` は smoke 用の小サンプル版であり、`gaussian_y_sigma_y2`（差分推定のため MC 誤差が増幅される）は fast では宣言済み tolerance を超える。**本サイズでは PASS**（rel err 0.025）。tolerance は変更していない。
