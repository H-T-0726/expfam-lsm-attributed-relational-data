# 真の潜在次元 K の定義と識別可能性 — 硬化版理論監査

**作成日:** 2026-09-04
**種別:** 理論監査（コード変更なし・EM 実行なし）
**baseline main:** `7e335602999977060208ce37ac8cdff8fedfa66e`
**数値検証:** `tools/research_audit/verify_identifiability_identities.py`
（**81 result rows / failure 0 / verdict PASS**。ただし内訳は **独立に反証可能な Monte-Carlo・数値照合が 41 行、構成上必ず PASS する解析評価・算術整合行が 40 行**。§19 参照）
**独立敵対レビュー:** `reports/identifiability/true_k_identifiability_review_20260904.md`
（Reviewer A 数理 / Reviewer B 統計。BLOCKER 3・HIGH 12 を受けて本文を改訂済み）

## Evidence label 凡例

| ラベル | 意味 |
|---|---|
| `[PRIMARY_SOURCE]` | 原論文そのもの |
| `[CONFIRMED_IN_REPOSITORY]` | 一次コードで確認 |
| `[DERIVED]` | 本監査で独立に導出（証明または証明スケッチ付き） |
| `[NUMERIC_CHECK]` | 上記スクリプトによる**独立な**数値確認（乱数を引いて解析値と照合したもの。式を式自身に代入し直す算術整合は含めない） |
| `[ARITHMETIC_CONSISTENCY]` | 主張式を式自身に代入して整合を見ただけの確認。独立証拠ではない |
| `[DEFINITION]` | 定義。導出ではない |
| `[ASSERTION]` | 主張のみ。証明も数値裏付けもない |
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
| 4 | 複数分布へ拡張したとき BIC の理論的性質は自明でない | **正しい**。現行 `calc_bic_dual` は Q3（観測データ周辺尤度）ではない（`[EMPIRICAL_EXISTING]` KI-010、`reports/k_selection_theory/paper_bic_reproduction_alignment_20260904.md`）。さらに **Gaussian-Y（`w≠0`）では M_K が入れ子でないことを証明**した（§12）。**ただし非入れ子が壊すのは尤度比検定の χ² 近似・Wilks の定理であって Schwarz BIC の導出ではない。**BIC が使えない理由は別にあり（潜在変数モデルの特異性・境界パラメータ・有効標本数の未定義、§17.1・§16b）、本監査はその特異性の定量（RLCT）に到達していない |
| 5 | 有限標本で選べることと n→∞ の一致は別 | **正しい**。本監査は population identifiability までしか到達しておらず、**推定量の consistency は `[UNRESOLVED]`**（§16、U6） |

**新しく証明できた主要命題は 3 つ**（§14）:

- **P1**: canonical Poisson-X では、population moment から `FF^T` を復元でき、`d ≥ K` かつ `rank(F)=K` なら `K` は population identifiable。
- **P2**: canonical Gaussian-Y では、**単一 dyad の周辺分布だけ**から `(K, w², σ_y²)` が決まる（`w ≠ 0` のとき）。
- **P3**: その帰結として **Gaussian-Y では `M_K ⊄ M_{K+1}`**（`w ≠ 0`）。すなわち K の族は入れ子でない。
- **P8**（敵対レビューで新たに得られた）: **`w` の符号は三角形（`n ≥ 3`）から識別できる。**`E[S_ij S_ik S_jk] = K` より Gaussian-Y の三角形 3 次同時中心モーメントは `w³K` であり、その符号が `w` の符号を決める。

**重要な適用範囲:** P2・P3 は **Gaussian-Y 限定**である。Phase 7e / 8b が実行したのは `family_y = bernoulli` であり、そこでは識別性も非入れ子性も **`[UNRESOLVED]`** のままである（§10.4・§12.4）。

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

と定義する。`[DEFINITION]`

**仮定 M-closed（省略不可）:** この定義は `P0 ∈ ⋃_K M_K`、すなわち **`P0` が候補族のどれかで厳密に表現できる**
ことを前提とする。これが破れると `{K : P0 ∈ M_K} = ∅` となり **`K*` は存在しない**。
実データは一般に M-closed ではない（本リポジトリの Wine / Cora / MovieLens は KI-011・KI-012・KI-018 で
誤指定が記録されている）。したがって **「実データで `K*` を推定している」とは書けない**。
誤指定下の代替目標（KL 射影次元）は §2.4 に挙げるが、その存在には `M_K` の閉性が要り、本監査では未確立（U11）。

**`M_K` は `n` で添字づけられる。** `X ∈ R^{n×d}`, `Y ∈ R^{n×n}` なので厳密には `M_K^{(n)}`・`K*^{(n)}` である。
P2・P3 は `n` に依らない単一 dyad 周辺から証明しているため `n` を跨いで移送できるが、それは
`z_i` が iid でモデルが projective であることに依存する。**historical generator の列 z-score（G1）はこの projectivity を壊す。**

### 2.2 この定義が意味を持つ条件

`min` が well-defined であるためには次で十分である。`[DERIVED]`

1. 候補集合 `{K : P0 ∈ M_K}` が空でない（少なくとも 1 つの K が `P0` を表現する）。
2. `K` は **非負整数** `{0, 1, 2, ...}` 上を動くので、空でない部分集合には最小元が必ず存在する（整列性）。
   `K = 0` は `F ∈ R^{d×0}`・`eta^X ≡ 0`・`eta^Y ≡ w0` の退化モデルとして well-defined であり、
   §5 の `F=0` かつ `w=0` の行で `K* = 0` と書くためには添字集合にこれを含める必要がある。

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

### 2.4 目標次元の代替定義（`K*` だけが選択肢ではない）

`[DEFINITION]`。本監査の各命題が実際にどれを確立したかを明示する。

| 記号 | 定義 | 本監査で確立したか |
|---|---|---|
| `K*` | `min{K : P0 ∈ M_K}`（M-closed 前提） | P2・P3 が Gaussian-Y で確立 |
| `K^rank` | population Gram `FF^T` の階数 | **P1・P5 が確立するのはこちら**。`X` 周辺しか見ないので、`Y` にしか現れない潜在次元は数えない（§5 の `rank(F)=r<K` 行） |
| `K°` | KL 射影次元 `min argmin_K inf_{θ∈Θ_K} KL(P0‖P_θ)` | **未確立（U11）。** 誤指定（M-open）で well-defined になりうる唯一の目標で、§17.3 の予測スコアに対応するのもこれ |
| 数値階数 | 有限標本 Gram の閾値付き階数 | **未確立（U7）。** 推定 Gram は PSD ですらない（§7.5） |

**`K^rank ≤ K*` であり等号は自明でない。** P1 を `K*` の主張として引用してはいけない。

---

## 3. Properness と identifiability の分離

混同しやすいので明示的に分ける。`[DERIVED]`

| 概念 | 主張内容 | 成立条件 |
|---|---|---|
| **properness** | `p(Z, X, Y)` が全確率 1 の確率測度である | finite `n, d, K`、finite parameters、Gaussian 分散 > 0、canonical link |
| **finite moments** | `E[Y^r] < ∞` 等 | family 依存。**Poisson-Y では追加条件が要る**（§11） |
| **parameter identifiability（`O(K)` を法として）** | `P_θ1 = P_θ2 ⟹ θ2 = θ1·Q`（`Q ∈ O(K)`） | §4 が示すとおり `O(K)` を法とせずに書くと**本モデルでは決して成立しない** |
| **order identifiability（`K` の識別）** | `P0 ∈ M_K ∩ M_{K'} ⟹ K = K'` | parameter identifiability とは**別問題**。P2・P3 はこちら |
| **functional identifiability** | `P_θ1 = P_θ2 ⟹ g(θ1) = g(θ2)`（特定の関数 `g` について） | P1・P5 が確立するのは `g = FF^T` および `g = rank` についてのみ |
| **local / global** | 近傍で一意か、大域で一意か | Ledermann 型のパラメータ数条件は **generic 点での local 識別**の必要条件 |
| **generic / everywhere** | パラメータ空間のほとんど至るところか、全域か | P1 の `rank(F)=K` は **generic 条件**。§5 の退化集合は空でない |
| **estimability** | 有限標本から推定できるか | identifiability とは**別**。§5 の「`w` が非常に小さい」行はこちら |

**properness は identifiability を含意しない。** Poisson-Y は `|w| ≥ 1/2` でも proper だが分散は無限大であり、Bernoulli-X は常に proper だが `d=1` では K を識別できない。

逆に **finite moments も identifiability を含意しない**。モーメントが有限でも、モーメント列が分布を一意に決めるとは限らない（モーメント問題）。

**本監査の論法の正確な射程:** すべての識別性主張は「population モーメント写像から特定の関数 `g(θ)` を復元できる」形である。
これは **`g(θ)` の functional identifiability の十分条件**であって、**`θ` そのものの identifiability ではない**。
`g` を共有する異なる `θ` については何も言えない。さらにこの論法は、用いるモーメントが
**真値だけでなく候補パラメータ空間全体で有限**であることを要する（Poisson-X は §7.4、Poisson-Y は §11.4 でこれを満たす）。

本文で **「population identifiable」** と書くときは「厳密な population 分布に対する識別可能性」を指し、
**有限標本からの推定可能性（estimability）は含意しない**。

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
`z → z/c`（すなわち `z = c z'`）とすると `eta^X = f^T z = (cf)^T z'`、`eta^Y = w z_i^T z_j = (c²w) z_i'^T z_j'` なので、対応する組は **`(cF, c²w)`** である（`(cF, w/c²)` ではない）— が、これも prior が `N(0,I)` に固定されているため許されない。よって **prior 固定下では `F` と `w` は個別に意味を持つ**。`[DERIVED]`

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
| Bernoulli-X, `d = 1` | する | **X 周辺からは不可**（§8 の反例 C1）。**Y 側が informative なら joint では識別されうる**（例: Gaussian-Y, `w≠0` は P2 により `K` を決める） |
| Bernoulli-Y, `w0 = 0` | する | **edge density からは不可**（§10.2）。共有ノードを持つ motif は情報を持つが一般結論は `[UNRESOLVED]` |
| `w = 0` かつ Bernoulli-X `d = 1` | する | **joint でも不可**。X 側・Y 側とも `K` の情報を持たない |

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

**証明.** (7.1) が `G` の対角、(7.2) が非対角をすべて与えるので `G` は population moments の関数として一意に決まる。

ここで **競合表現に対する量化が必要である**（これがないと識別可能性の証明にならない）。同じ `P0` の X 周辺を
与える任意の canonical Poisson-X 表現 `(K', F')` もまた (7.1)(7.2) を満たすので `F'F'^T = G`。ゆえに
`K' ≥ rank(F') = rank(G)`。逆に `G` の階数分解をとれば `K' = rank(G)` の表現が実在する。したがって

```
K* (X 周辺の最小潜在次元) = rank(G)
```

であり、真値が `rank(F) = K` を満たすとき `K* = K`。∎ `[DERIVED]`

**注意（敵対レビュー A-F08）:** 量化なしに `rank(G) = rank(F) = K` と書くだけでは不十分である。
`(K+1, F' = [F, 0])` は X 周辺を完全に再現するので、**最小性の規約（§2.1）を経由しない限り `K` そのものは
X 側から識別されない**。上の一段がその橋渡しである。

**モーメント有限性:** Gaussian の MGF は全実数で有限なので、`E[X_l]`, `E[X_l X_m]` は**任意の有限 `F` について常に有限**である。Poisson-Y（§11）と異なり、モーメント存在の追加条件は不要。`[DERIVED]`

### 7.5 P1 が要求する仮定（省略不可）

1. **canonical unclipped link**。`exp` に hard clip があると (7.1)(7.2) は崩れる。historical generator は `np.clip(eta, -20, 10)` を持つ（§13）。
2. `d ≥ K`。
3. `rank(F) = K`。
4. population moments（有限標本での rank 判定は別問題）。

**`K = rank(G)` は population の主張である。** 有限標本では事情が大きく異なる。本監査の数値確認では、
`d=6, K=3`, 4×10⁶ draws で推定 Gram `Ĝ` の固有値は

```
eig(G)  = [1.8152, 0.8916, 0.4334,  2.8e-17, -1.1e-16, -1.3e-16]
eig(Ĝ)  = [1.8176, 0.8918, 0.4332,  9.2e-04,  5.1e-04, -2.0e-03]
```

であり、**`Ĝ` は PSD 錐の外に出る（最小固有値 −2.0e−03）**。したがって閾値なしの `rank(Ĝ)` は `d` になり、
`K` を返さない。固有値ギャップ比は `λ_3/λ_4 ≈ 469` と大きいが、**この比を K 判定の閾値として事前に固定する
根拠を本監査は持たない**。`[NUMERIC_CHECK]`

**結論:** 有限標本の rank 閾値を事前に固定する方法は確立していない（U7）。本監査は Gram spectrum を
**診断量としてのみ**扱い、**K 選択の criterion としては提案しない**。

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

> **反例 C1（X 周辺に限る）.** `d = 1` の canonical Bernoulli-X において、任意の `K ≥ 1` と
> 任意の `f_1 ∈ R^K \ {0}` に対し、**`X` の周辺分布**は
> ```
> X_1 ~ Bernoulli(1/2)   （各行 iid）
> ```
> で**完全に一致する**。したがって **`X` の周辺分布からは** `K` を識別できない。

**証明.** §8.1 より `P(X_1 = 1) = E[sigmoid(f_1^T z)] = 1/2`。`d=1` なので X の観測は各行 1 ビットで、行は iid。
よって **X 周辺分布**は `K, f_1` に依存しない。∎ `[DERIVED]`

**射程の限定（敵対レビュー BLOCKER、両レビュアーが独立に指摘）:** §2.1 の `M_K` は **joint** `(X, Y)` の
分布族であり、`P0` は joint の観測分布である。C1 が示したのは **X 周辺**の縮退だけで、`Y` については何も
言っていない。実際、`family_y = gaussian` かつ `w ≠ 0` なら **P2 により `K` は Y 側だけから識別される**ので、
`family_x = bernoulli, d = 1, family_y = gaussian, w ≠ 0` という配置では **`K` は joint で識別可能**である。

**一般則:** 周辺分布で証明した**肯定的**識別性は joint に移送できる（観測が増えて困ることはない）。
**否定的**な非識別性は移送できない。P1 は前者なので安全、C1・C3 は後者なので周辺限定である。

これは先生の指摘 3（「K_TRUE を設定しただけでは識別可能とは限らない」）の**具体的な裏付け**であるが、
**「Bernoulli-X を使うと K が識別できない」ではなく「X チャネルだけでは K の情報が取れない配置が実在する」**
という限定付きの裏付けである。

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

`[DERIVED]` `[NUMERIC_CHECK]`（`S_mgf`, K=1,3,5 × t=0.3,0.5、**rel err ≤ 0.0011**）

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

**証明.** (9.5), (9.2), (9.3), (9.4) を順に適用すればよい。各ステップは population キュムラントの関数であり、`w ≠ 0` のとき well-defined。

**競合表現への量化**（P1 と同じ一段。初版が欠いていた。敵対レビュー F-10）: 同じ dyad 周辺を
与える任意の canonical Gaussian-Y 表現 `(K', w0', w', σ_y')` も同じキュムラント恒等式を満たすので、
(9.2) より `w'² = w²`、(9.3) より `K' = K`。ゆえに表現は一意である（§12.2 と同じ論法）。∎ `[DERIVED]`

### 9.5 何が識別**されない**か

| 項目 | 状況 |
|---|---|
| **`w` の符号** | **単一 dyad 周辺からは識別されない。** (9.1) は `t` の偶関数なので `S` は 0 対称であり、`wS` と `(−w)S` は同分布。**ただし `n ≥ 3` の joint からは識別される（§9.6 の P8）。** `[DERIVED]` |
| `w = 0` | `kappa_4 = kappa_6 = 0` で (9.2)(9.3) が `0/0`。`Y` は `N(w0, sigma_y²)` に退化し **K は識別不能** `[DERIVED]` |
| `F` | Y だけからは決まらない（§4 の `O(K)` 不変性、および `F` は Y に現れない） |
| `sigma_y` 未知の影響 | (9.4) で決まるので**問題にならない**。`w ≠ 0` である限り `sigma_y` が未知でも K の識別は妨げられない `[DERIVED]` |

**単一 dyad marginal vs network joint distribution:** P2 は**単一 dyad の周辺分布だけ**で成立するので、network 全体の同時分布を使う必要がない。Bernoulli-Y（§10）とは対照的である。`[DERIVED]`

**ただしこれは population の主張である。** 実際に観測するのは network 1 個であり、`n(n−1)/2` 個の dyad は
`n` 個の潜在ベクトルを共有するため **独立ではない**。したがって `κ_4`・`κ_6` の標本版は iid 平均ではなく、
その一致性・収束率は本監査では示していない（§16b、U6）。

### 9.6 P8 — `w` の符号は三角形から識別される

> **命題 P8.** canonical Gaussian-Y において `n ≥ 3` ならば、三角形 `(i,j,k)` の 3 次同時中心モーメント
> ```
> E[ (Y_ij − w0)(Y_ik − w0)(Y_jk − w0) ] = w³ K
> ```
> が成立し、`w ≠ 0` のときその符号は **`w` の符号**を決める。

**証明.** `z_i` で条件づけずに、まず `z_i` について期待をとる。`E[z_i z_i^T] = I_K` なので
```
E[S_ij S_ik | z_j, z_k] = E[ (z_j^T z_i)(z_i^T z_k) | z_j,z_k ] = z_j^T I z_k = S_jk
```
ゆえに `E[S_ij S_ik S_jk] = E[S_jk²] = Var(S) = κ_2(S) = K`（§9.2）。
`Y_ab − w0 = w S_ab + ε_ab` で `ε` は平均 0・独立なので交差項はすべて消え、
3 次同時中心モーメントは `w³ · E[S_ij S_ik S_jk] = w³ K`。`K > 0` より符号は `sign(w³) = sign(w)`。∎
`[DERIVED]` `[NUMERIC_CHECK]`（`triangle_third_moment_of_S` / `triangle_identifies_sign_of_w`、K=1,3,5 × w=±0.8、9 行すべて rel err ≤ 0.0056 かつ符号一致）

**由来:** この命題は本監査の初版にはなく、**敵対レビュー（Reviewer A, F-01）が初版の「`w` の符号は識別されない」
という誤った断定を反証する過程で得られた**。初版は単一 dyad で示した対称性を joint の主張として書いていた。
本監査で独立に再導出・数値確認したうえで採用する。

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

`[DERIVED]` `[ARITHMETIC_CONSISTENCY]`（`poisson_y_moment_existence` の 21 行は主張式を式自身に代入して
評価しているだけで、`pass` は定数 `True`。**独立な数値証拠ではない**）

**`E[Y^r] < ∞ ⟺ E[λ^r] < ∞` の一行証明**（初版は無証明で主張していた）: 第 2 種 Stirling 数 `S(r,j) ≥ 0` により
`Y^r = Σ_{j=1}^r S(r,j) Y^{(j)}`（`Y^{(j)}` は下降階乗）で、`E[Y^{(j)}] = E[λ^j]`。すべて非負項の和であり、
Lyapunov 不等式 `E[λ^j] ≤ (E[λ^r])^{j/r}`（`j ≤ r`）より `j = r` の項が支配的。ゆえに同値。`[DERIVED]`

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

**ここで初版は誤った因果を書いていた（敵対レビュー A-F03 で反証）。** 正しくは次のとおり。

1. **実現サンプルは clip の有無によらず a.s. 有限である。** `S = z_i^T z_j` は a.s. 有限なので `λ = exp(w0+wS)` も
   a.s. 有限で、`Y | λ ~ Poisson(λ)` も a.s. 有限。発散するのは **population の 2 次モーメント**であって、
   個々の実現値ではない。観測される症状は「標本分散が `n` とともに安定しない」ことである。
2. **`np.clip(eta, −20, 10)` は historical の設定ではほぼ発動しない。** `w0=0, w=0.5` では `η_y = 0.5 S` で、
   `P(η_y > 10)` は Reviewer A の 2×10⁷ dyad の見積もりで K=1: 0、K=3: 5e−08、K=5: 0 のオーダーである。
   すなわち clip は**数値ガードとして事実上不活性**であり、データが有限であることを説明しない。

したがって正しい記述は: **historical Poisson-Y の default `w = 0.5` は、canonical unclipped model の下で
population の分散が発散する境界そのものである。clip はその事実を隠しても救ってもいない。**
`[CONFIRMED_IN_REPOSITORY]` + `[DERIVED]`

**さらに historical データには第 2 の乖離がある（敵対レビュー B-19）:** (11.1) は `‖z_j‖² ~ χ²_K` に依存するが、
historical generator は `Z` を列 z-score している（§13 G1）ので、そもそも `χ²_K` が成立しない。
**O1 は canonical model についての主張であり、historical データの分布についての主張ではない。**

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

`K + 1 = K` は矛盾。よって `P ∉ M_{K+1}`。∎ `[DERIVED]`

**`w' = 0` の場合の処理**（初版が飛ばしていた一行、A-F15）: (9.2) の適用には `κ_4 ≠ 0` すなわち `w' ≠ 0` が要る。
`w' = 0` の表現なら `κ_4(P) = 0` になるが、仮定より `κ_4(P) = 6Kw⁴ ≠ 0`（`w ≠ 0`）。よって `w' = 0` は起こらない。

**数値的裏付けの限界（A-F06）:** `km1_nesting_gaussian_y` の 15 行は `κ_4 = 6Kw⁴`・`κ_6 = 120Kw⁶` を
**代入したうえで**復元式がそれを戻すことを確認しているだけであり、乱数を一切引いていない。
したがって P3 に付くのは `[ARITHMETIC_CONSISTENCY]` であって `[NUMERIC_CHECK]` ではない。
**P3 に独立な数値証拠は存在しない。** ただし P3 が依拠する §9.2・§9.3 のキュムラント自体は独立に確認されている。

### 12.3 P3 の含意 — model selection にとって重大

`[DERIVED]`

1. **`M_K` の族は入れ子ではない**。正確には `{P ∈ M_K : w ≠ 0}` が `M_{K+1}` と交わらない、である。
   **`M_K` と `M_{K+1}` 自体は互いに素ではない**: `w = 0` の切片では `F' = [F, 0]`, `w' = 0` により
   同じ joint が両方に属する（A-F11）。「互いに素」と書いてはいけない。
2. したがって **「K を大きくすれば必ず当てはまりが良くなる」という単調性は population レベルでは成立しない**。
   ただしこの形の主張は入れ子族でも自明に成り立つ（`K = K*` で KL は 0 になり、それ以上改善しない）ので、
   P3 から実際に得られる強い主張は「`K+1` で厳密に悪化する」であり、それには `inf_{Q ∈ M_{K+1}} KL(P0‖Q) > 0`、
   すなわち `M_{K+1}` の閉性が要る。`P0 ∉ M_{K+1}` は `P0 ∉ closure(M_{K+1})` を含意しない。**閉性は未確立（U11）。**
3. **非入れ子が壊すのは尤度比検定の `χ²` 近似と Wilks の定理である。** `K` vs `K+1` の LRT は入れ子性を要するので使えない。
   **しかし Schwarz BIC の導出は入れ子性を前提としない**（モデルごとの Laplace 近似であり、非入れ子モデルの
   比較に BIC を使うのは標準的である）。初版はこれを混同していた（A-F04・B-B04）。
   **BIC がここで正当化されない本当の理由は §17.1 に移した。**
4. **有限標本の最尤当てはまりが `K` とともに単調に改善する保証もない。** 初版は「パラメータ数が増えるため」
   単調に改善しうると書いたが、`sup_{Θ_{K+1}} L ≥ sup_{Θ_K} L` はまさに `M_K ⊆ M_{K+1}` から出る性質であり、
   P3 がそれを否定した以上この理由付けは使えない（B-B06）。**X 側のみの部分族は入れ子（§12.4）なので
   X 由来の当てはまりは単調だが、Y を含む joint の当てはまりの単調性は `[UNRESOLVED]`。**

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
| **P1** | Poisson-X で `FF^T` が population moments から復元でき、**X 周辺の**最小潜在次元が `rank(FF^T)` に一致する | canonical unclipped link、`rank(F)=K`（これが `d ≥ K` を含意）。**generic 条件**であり全域ではない | §7.4 `[DERIVED]` `[NUMERIC_CHECK]` |
| **P2** | Gaussian-Y で単一 dyad 周辺から `(w0, w², K, sigma_y²)` が一意に決まる | `w ≠ 0` | §9.4 `[DERIVED]` `[NUMERIC_CHECK]` |
| **P3** | Gaussian-Y で `M_K ⊄ M_{K+1}`（族は入れ子でない） | `w ≠ 0`。**`M_K ∩ M_{K+1} ≠ ∅`（`w=0` 切片で交わる）** | §12.2 `[DERIVED]` `[ARITHMETIC_CONSISTENCY]`（独立数値証拠なし） |
| **P8** | Gaussian-Y で `w` の**符号**は三角形（`n≥3`）から識別される。`E[(Y_ij−w0)(Y_ik−w0)(Y_jk−w0)] = w³K` | `w ≠ 0`, `n ≥ 3` | §9.6 `[DERIVED]` `[NUMERIC_CHECK]` |
| **P4** | `O(K)` 回転・符号反転・座標置換は観測分布を変えない。`N(0,I)` prior によりスケールは固定 | — | §4 `[DERIVED]` |
| **P5** | Gaussian-X で `Sigma` 既知なら **X 周辺の**最小潜在次元 `= rank(Cov(X) − Sigma)` | `rank(F)=K`（`d ≥ K` を含意）。**generic** | §6.2 `[DERIVED]` |
| **P6** | Poisson-Y のモーメント有限性は `E[Y^r] < ∞ ⟺ |w| < 1/r` | canonical unclipped link | §11.2 `[DERIVED]` `[ARITHMETIC_CONSISTENCY]`（独立数値証拠なし） |
| **P7** | Bernoulli-X / Bernoulli-Y（`w0=0`）の一次モーメントは K の情報を持たない（常に 1/2） | `Z` が 0 対称 | §8.1, §10.2 `[DERIVED]` `[NUMERIC_CHECK]` |
| **O1** | historical Poisson-Y の default `w = 0.5` は分散発散の境界そのもの | — | §11.3 `[DERIVED]` + `[CONFIRMED_IN_REPOSITORY]` |

---

## 15. 反例

| ID | 内容 | 節 |
|---|---|---|
| **C1** | Bernoulli-X, `d = 1`: 任意の `K`・任意の `f_1 ≠ 0` で **X 周辺**が `Bern(1/2)` に一致。**X 周辺のみの反例で、joint の反例ではない**（`w≠0` の Gaussian-Y なら P2 により識別される） | §8.2 |
| **C2** | `w = 0`（任意の family-Y）: `Y` が潜在に依存せず、Y 側から K の情報が完全に消える | §5 |
| **C3** | Gaussian-X, `d=2, K=1`, `Sigma` 未知対角: Ledermann 型必要条件 `(d−K)² ≥ d+K` を満たさない。**`[COUNTING_ARGUMENT]`（観測同値なパラメータ対を明示していない）かつ X 周辺のみ** | §6.3 |
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
| U9 | `K* = K_TRUE` を保証する construction 条件の完全性 | clean generator は「full-rank F・`w≠0`・`d≥K`」を課すが、これが `K*=K_TRUE` の**十分**条件であることは family ごとに要確認 |
| U10 | held-out 予測スコアの population 選択対象 | proper scoring rule なら M-closed 下で `K*` に一致しうるが、Phase 7e/8b が使うのは推定 `ẑ` を代入した **plug-in raw-eta score** で proper ではない。その target は特定されていない（§17.3） |
| U11 | `M_K` の閉性、および誤指定下の pseudo-true `K` | `P0 ∉ M_{K+1}` は `P0 ∉ closure(M_{K+1})` を含意しない。KL 射影次元の存在にも閉性が要る（§2.1・§12.3-2） |
| U12 | 候補集合 `{K : P0 ∈ M_K}` が連結か | 非入れ子族では `{2,5}` のような非連結集合が原理的に否定できていない。その場合 `min` は導出値ではなく**選択規約**になる |

---

## 16b. 漸近レジームの未確定性

`[ASSERTION]` ではなく、指摘の構造だけを述べる。**本監査はここに解答を持たない。**

先生の指摘 5（「有限データでたまたま K を選べることと `n → ∞` で真のモデルを選べることは別」）に答えるには、
まず **何が無限大に行くのか**を決める必要がある。本モデルではそれ自体が確定していない。

| 論点 | 内容 |
|---|---|
| **有効標本数が 3 通りある** | ノード数 `n`／dyad 数 `n(n−1)/2`／X の要素数 `nd`。これらは `n` のオーダーが違う。現行 `calc_bic_dual` の penalty は `p̂ · log n` で **ノード数**を使う（`[CONFIRMED_IN_REPOSITORY]`）が、Y チャネルが供給する情報は `n(n−1)/2` dyad 分である |
| **潜在変数が `n` とともに増える** | `Z ∈ R^{n×K}` は `nK` 個の座標を持ち、`n` と一緒に増える。これは古典的 iid 設定ではなく **incidental parameter / Neyman–Scott 型**の状況である。`Q_strict` は `Z` を完全データとして扱う一方、penalty `p̂` は構造パラメータだけを数え `Z` を数えない |
| **dyad は独立でない** | P1・P2 の有限標本版は、`n` 個の潜在ベクトルを共有する **従属な** dyad の平均である。network 1 個からの `κ_4`・`κ_6` 推定の一致性・収束率は本監査では示していない |
| **モデルが特異である** | §4 の `O(K)` 不変性により写像は単射でなく、Fisher 情報は `K(K−1)/2` 次元の軌道方向に退化する。さらに `rank(F) < K` や `w = 0` の縮退集合（§5）上でも退化する。本モデルの RLCT は未知（`RESEARCH_MASTER.md` §14 U5） |
| **境界パラメータ** | `Σ_X ⪰ 0`、`σ_y² ≥ 0`、因子分析の `Ψ ⪰ 0`。`K` 比較はしばしば境界上で起きる |

**したがって本監査は「`n → ∞` で何が起きるか」について何も証明していない。** これは U6 に集約する。

---

## 17. model selection への含意

**この節の主張はすべて「明示した条件下で」であり、`n → ∞` の一致性は一切含まない。**

### 17.1 BIC が正当化されない理由は非入れ子性ではない

初版はここを誤っていた（A-F04・B-B04）。整理する。

| 手法 | 非入れ子で壊れるか | 本モデルで使えるか |
|---|---|---|
| 尤度比検定の `χ²` 近似・Wilks の定理 | **壊れる** | `K` vs `K+1` には使えない（P3、Gaussian-Y `w≠0`） |
| Schwarz BIC の導出 | **壊れない**（モデルごとの Laplace 近似で、非入れ子比較に BIC を使うのは標準的） | **別の理由で正当化されない**（下記） |

**Schwarz BIC がここで正当化されない実際の理由:**

1. **潜在変数モデルは特異である。** §4 の `O(K)` 不変性で Fisher 情報が退化し、`rank(F)<K` や `w=0` の
   縮退集合（§5）上でも退化する。Laplace 近似の前提（局所漸近正規性・情報行列の正則性）が成立しない。
   特異モデルでは `(p/2)log n` の項が RLCT に置き換わるが、**本モデルの RLCT は未知**（`RESEARCH_MASTER.md` §14 U5）。
2. **境界パラメータ**（§16b）。
3. **有効標本数が未定義**（§16b）。`calc_bic_dual` は `log n` にノード数を使う。

**なお `calc_bic_dual` の実装事実**（`[CONFIRMED_IN_REPOSITORY]` `expfam/src/utils_expfam.py`）:
`num_params = kd − k(k−1)/2 + [d if X gaussian] + [1 if Y gaussian]`、`bic = −2 Q_strict + num_params · log n`。

- `−k(k−1)/2` は §4 の `O(K)` 軌道の次元をすでに引いている。§6.3 の Ledermann 計数と整合する。
- **`w0` と `w` は `num_params` に数えられていない**（NOLTA 2024 の慣行としてコメントされている）。
  ところが P2 は **Gaussian-Y では `K` の情報をもっぱら `w` が運ぶ**ことを示した。
  `K` の唯一の担い手をパラメータ数から慣行で外していることは、`K` 選択の文脈では検討に値する（A-F14）。
  **本監査はこれを「誤り」とは断定しない。** penalty の設計問題として記録する。

### 17.2 現行 `calc_bic_dual` は観測データ周辺尤度ではない

`[EMPIRICAL_EXISTING]` KI-010、`reports/k_selection_theory/paper_bic_reproduction_alignment_20260904.md`。
`Q_strict` は Q2（complete joint）の MC 平均であって Q3（観測データ周辺尤度）ではない。
したがって「Schwarz BIC」と呼ばない、という既存の運用は本監査でも支持される。

### 17.3 held-out 予測スコアが何を選ぶかは `[UNRESOLVED]`

**初版はここで「held-out score は `K*` を推定していない」と断定していたが、その論法は無効であり、
結論も population レベルでは逆の可能性が高い**（A-F05・B-B07）。訂正する。

- **キュムラント論法は非論理的だった。** 「`K*` は 4 次・6 次キュムラントに符号化されている」ことは
  「予測損失の最小化が別の量である」ことを何ら含意しない。
- **厳密に proper な scoring rule を population で評価すれば、M-closed の下で argmin は `K*` に一致しうる。**
  対数損失の population 期待値は `H(P0) + KL(P0‖Q)` で `KL = 0` の点、すなわち `P0` 自身で最大になる。
  P3 により `M_K` は（`w≠0` 部分で）互いに交わらないので、それを達成するのは `K = K*` だけである。
- **実際に成立しうる乖離の理由は別にある**:
  1. **Phase 7e / 8b が使うのは plug-in raw-eta score** である（`y·η − logaddexp(0,η)`、`η = ŵ0 + ŵ ẑ_i^T ẑ_j`）。
     推定した `ẑ` を代入した plug-in であって、モデルの予測分布に対する proper scoring rule **ではない**。
     その population target は特定されていない。
  2. **有限標本の bias–variance trade-off。**
  3. **M-open（誤指定）ではそもそも `K*` が存在せず**（§2.1）、予測目標だけが well-defined になる。

**判定: `[UNRESOLVED]`（U10）。** 「held-out は `K*` を推定している」とも「していない」とも書かない。

**ただし次は独立に正しく、維持する:**
> **「held-out で `K_TRUE` が選ばれた」ことは「`K_TRUE` が識別された」ことを意味しない。**
> 選択（selection）と識別（identification）は別概念であり、これは §17.3 の論法に依存しない。

### 17.4 識別可能性が成り立たない領域での挙動

**初版の「どんな criterion も K を当てられない」は言い過ぎだった**（B-B17）。訂正する。

正しくは: **観測分布に `K` を区別する情報が存在しない場合、その情報を使って `K` を当てることは
どんな手続きにもできない。** しかし手続きが特定の `K` を出力すること自体は妨げられない。
たとえば frozen Phase 7e/8b の **smallest-K tie rule** は、データが無情報なら `K = 1` を選ぶ。
その出力が `K_TRUE = 1` と一致したとしても、それは **tie rule の反映であってデータからの識別ではない**。

該当領域: `w = 0`（C2）、および X 側・Y 側がともに無情報な配置（§5）。
**C1（Bernoulli-X, `d=1`）は X 周辺のみの反例なので、ここには単独では該当しない。**

### 17.5 実験設計への含意

1. **`K*` と `K_TRUE` は別物である**（§2.3）。人工データの「真値一致」は `K_TRUE`（generator の手続き上の設定）
   との一致であって `K*` との一致ではない。`K* < K_TRUE` の可能性は family ごとに検証が要る（U9）。
   **したがって under-selection を直ちに criterion の失敗と読んではならない。**
2. **P1 の前提は historical generator では満たされていない。** Phase 7e / 8b は
   `generate_dual_data`（`family_x=poisson`）を使っており、そこには Poisson の hard clip（G4）と
   `Z` の列 z-score（G1）がある。**P1 は当該実験の well-specification を保証しない。**
   本研究の clean generator はこの 2 点を除去する。
3. **P2・P3 は Gaussian-Y 限定である。** Phase 7e / 8b が実行したのは `family_y = bernoulli` であり、
   そこでは識別性（U2）も非入れ子性（U5）も未解決である。
4. **Poisson-Y を使う実験では `|w| < 1/2` を課さないと、分散が定義されない分布の上で
   二乗誤差的な議論をすることになる**（O1）。
5. **Gram spectrum を K 判定に使うなら閾値問題が避けられない**（U7、§7.5）。推定 Gram は PSD ですらない。

## 18. Claim ledger

**改訂版。** 敵対レビューで撤回した主張は「書いてはいけない」に移してある。

### 書いてよい（条件を明示すれば）

| 主張 | 必須の限定 |
|---|---|
| canonical model は proper probability model として定義できる | finite `n,d,K`、finite parameters、Gaussian 分散 > 0。**Poisson-Y はモーメント有限性が別条件**（P6） |
| canonical Poisson-X では、明示した条件下で **X 周辺の最小潜在次元**が `rank(FF^T)` に一致する | canonical unclipped link、`rank(F)=K`、population moments。**generic 条件**。**`K*` ではなく `K^rank` の主張**（§2.4）。**有限標本の rank 判定は別問題で、推定 Gram は PSD ですらない**（U7・§7.5）。**この前提は Phase 7e/8b の生成器では満たされていない**（§13 G1・G4） |
| canonical Gaussian-Y では、明示した条件下で `(w0, w², K, σ_y²)` が単一 dyad 周辺から population identifiable である | `w ≠ 0`。**単一 dyad 周辺からは `w` の符号は決まらない**（→ P8） |
| canonical Gaussian-Y では `w` の符号が三角形（`n ≥ 3`）から識別される | `w ≠ 0`。`E[(Y_ij−w0)(Y_ik−w0)(Y_jk−w0)] = w³K` |
| canonical Gaussian-Y では `{P ∈ M_K : w≠0}` は `M_{K+1}` と交わらない | `w ≠ 0`。**`M_K` と `M_{K+1}` 自体は `w=0` 切片で交わる**。**他 family は未証明**（U5） |
| 非入れ子性は尤度比検定の `χ²` 近似・Wilks の定理を無効にする | **Schwarz BIC の導出は非入れ子でも壊れない。** BIC がここで正当化されないのは特異性・境界・有効標本数による（§17.1） |
| canonical Bernoulli-X では `d = 1` のとき **X 周辺分布**から `K` を識別できない（反例あり） | `d = 1` に限り、**かつ X 周辺のみ**。**joint の反例ではない**（`w≠0` の Gaussian-Y なら P2 で識別される） |
| historical generator は canonical model の literal generator ではない | G1〜G5 の具体的差分を挙げる。**historical 結果を無効とは言わない** |
| canonical unclipped Poisson-Y では `E[Y^r] < ∞ ⟺ \|w\| < 1/r` であり、historical default `w=0.5` は分散発散の境界にある | canonical model についての主張。**historical データは clip 済みかつ `Z` が z-score 済みで `χ²_K` が成立しないので、historical データの分布についての主張ではない** |
| 選択（selection）と識別（identification）は別概念である | 「held-out で `K_TRUE` が選ばれた」は「`K_TRUE` が識別された」を意味しない |

### 書いてはいけない

- 「BIC は理論的に正しい」
- 「BIC は非入れ子だから使えない」（**理由が誤り**。§17.1）
- 「held-out なら真の K を必ず選べる」
- 「held-out score は `K*` を推定している」**および**「していない」（どちらも未証明。U10）
- 「Poisson-X なら常に K が識別可能」（unclipped link・`rank(F)=K` の generic 条件が要る）
- 「Poisson-X の結果は `K*` についての主張である」（確立したのは `K^rank`。§2.4）
- 「Bernoulli では K は識別できない」（反例は `d=1` かつ **X 周辺のみ**）
- 「`w` の符号は識別されない」（**単一 dyad 限定**。三角形からは識別される）
- 「`M_K` と `M_{K+1}` は互いに素」（`w=0` 切片で交わる）
- 「識別不能な領域ではどんな criterion も K を出力しない」（tie rule により出力は出る）
- 「clip があるから historical Poisson-Y のデータが有限である」（実現値は clip なしでも a.s. 有限）
- 「K-selection consistency を証明した」（U6・§16b）
- 「n を増やせば必ず K_TRUE に収束する」（U6）
- 「clean generator で全問題を解決した」
- 「historical generator のせいで過去結果は無効」
- 「K と K+1 は完全に non-nested」（Gaussian-Y `w≠0` のみ、しかも `w=0` 切片で交わる）
- 「真の K は generator の K_TRUE である」（§2.3）
- 「実データにおいて `K*` を推定している」（M-closed が成立しない。§2.1）
- 「Phase 8b の真値一致率は `K*` recovery である」（`K_TRUE` との一致である。§17.5）
- 「P2・P3 は Phase 7e/8b（`family_y=bernoulli`）に適用される」（**Gaussian-Y 限定**）
- 「81 checks PASS だから 81 件の独立証拠がある」（独立は 41 件。§19）

### 引用時の lineage 明記（root `CLAUDE.md` §3 / KI-002）

Phase 7e・Phase 8b はいずれも **lineage E**（`DualExpFamLSMConsistent`、experimental prototype、
**本文採用不可**）である。本監査からそれらの結果に言及するときは必ずこの tag を付ける。

---

## 19. 数値検証の再現と、その証拠力の内訳

```
python tools/research_audit/verify_identifiability_identities.py --out <path>
```

- seed `20260904` 固定、tolerance は実行前に `TOLERANCES` として宣言（**結果を見て変更していない**）。
- 2026-09-04 改訂後の実行: **81 result rows / failure 0 / verdict PASS**。

**内訳（敵対レビュー A-F06・B-B10 を受けて明示する）:**

| 区分 | 行数 | check |
|---|---:|---|
| **独立に反証可能**（乱数を引いて解析値と照合、または数値グリッド） | **41** | `S_mgf` 6 / `S_kappa_{2,4,6}` 9 / `gaussian_y_*` 3 / `poisson_x_gram_recovery` 1 / `bernoulli_x_mean` 12 / `poisson_y_factorial_moment_monotonicity` 1 / `triangle_*` 9 |
| **構成上必ず PASS**（解析式の自己評価・算術整合・診断出力） | **40** | `poisson_y_moment_existence` 21 / `km1_nesting_gaussian_y` 15 / `poisson_x_eigen_*` 2 / `poisson_x_rank_true_by_construction` 1 / `poisson_x_estimated_gram_is_not_psd` 1 |

**したがって「81 checks PASS」を独立証拠 81 件と読んではいけない。**
とくに **P3 と P6 には独立な数値証拠がない**（それぞれ `km1_nesting_gaussian_y` と
`poisson_y_moment_existence` に依拠しており、どちらも主張式を式自身に代入している）。
両命題の `[NUMERIC_CHECK]` ラベルは §14 で `[ARITHMETIC_CONSISTENCY]` に降格した。

**tolerance の検出力（A-F12、未対応の限界として記録）:** 宣言 tolerance は達成精度より 6〜18 倍緩い。
たとえば `cumulant_6` の tol 0.35 は `κ_6 = 120K` を `90K` や `150K` と区別できない。
tolerance は結果を見て変更しない方針を守るため**今回は据え置き**、
「検出力が低い」という事実を限界として記録する。次回の実験では MC 散らばりの実測（例 5·sd）から設定する。

**改訂で修正したスクリプトのバグ:**
- `poisson_x_rank` は `gram_true`（構成上 rank K）を検査しており、推定 Gram を一度も見ていなかった（A-F07）。
  推定 Gram の閾値なし階数・最小固有値・固有値ギャップを報告する行に置き換えた。
- `_cumulants_from_sample` の 6 次式が `−10 μ₃²` 項を 0 に固定していた（A-F16）。
  対称変数では population で 0 だが標本では 0 でないため、汎用ヘルパーとして復元した。

### 19.1 `--fast` について

`--fast` は smoke 用の小サンプル版であり、`gaussian_y_sigma_y2`（差分推定のため MC 誤差が増幅される）は
fast では宣言済み tolerance を超える。**本サイズでは PASS**。tolerance は変更していない。

---

## 20. 改訂履歴（敵対レビュー対応）

本文書は 2026-09-04 に初版を作成し、同日の独立敵対レビュー
（`reports/identifiability/true_k_identifiability_review_20260904.md`、Reviewer A 数理 / Reviewer B 統計）
を受けて改訂した。**初版の主張のうち次は誤りであり、撤回・訂正した。**

| # | 初版の誤り | 訂正 |
|---|---|---|
| 1 | 反例 C1 を joint model の非識別性として書いた（自身の P2 と矛盾） | X 周辺限定に rescope。§8.2・§5・§15・§17.4・§18 |
| 2 | 「`w` の符号は識別されない」と断定 | 単一 dyad 限定。**三角形からは識別される（新命題 P8）** |
| 3 | 「clip があるから historical データが有限」 | 実現値は clip なしでも a.s. 有限。clip は当該設定でほぼ不活性 |
| 4 | 「Schwarz BIC の導出は入れ子性を前提とする」 | 前提としない。BIC が使えない理由は特異性・境界・有効標本数（§17.1） |
| 5 | 「held-out score は `K*` を推定していない」と断定 | 論法が無効。`[UNRESOLVED]`（U10）に降格 |
| 6 | 「71 checks PASS」を独立証拠として提示 | 41 独立 / 40 構成上 PASS に内訳を明示。P3・P6 の `[NUMERIC_CHECK]` を降格 |
| 7 | `poisson_x_rank` が推定 Gram を検査していなかった | スクリプト修正 |
| 8 | P1 の証明に競合表現への量化がなかった | 最小性の一段を追加 |
| 9 | `(cF, w/c²)` | `(cF, c²w)` |
| 10 | 「`M_K` は互いに素」 | `w=0` 切片で交わる。正しくは `{P ∈ M_K : w≠0} ∩ M_{K+1} = ∅` |
| 11 | 「どんな criterion も K を当てられない」 | tie rule により出力自体は出る。識別と選択の区別（§17.4） |
| 12 | M-closed 仮定が明示されていなかった | §2.1 に明示。実データでは `K*` が存在しない |
| 13 | 漸近レジーム・有効標本数・incidental parameter が欠落 | §16b を新設 |
| 14 | `S_mgf` の rel err を 0.011 と記載 | 実測 **0.0011** |
| 15 | `K` の添字集合と `K*=0` が不整合 | 非負整数に統一 |

**レビューで指摘されたが、本監査では対応しきれなかった事項**は U10・U11・U12 および
§19 の tolerance 検出力として記録した。
