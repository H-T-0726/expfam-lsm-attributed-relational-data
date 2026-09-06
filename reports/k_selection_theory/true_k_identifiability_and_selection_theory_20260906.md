# 真の潜在次元 K と選択基準の理論整理 — 先生相談前メモ

**作成日:** 2026-09-06
**種別:** THEORY REVIEW ONLY（新規実験なし・コード変更なし）
**目的:** 先生からの 6 つの指摘に対し、**何が証明済みで、何が条件付きで、どこから先が本当の研究課題なのか**を
一次証拠に基づいて固定する。**分からないことを埋めない。**

## この文書の位置づけ

**新しい定理を作った文書ではない。** 既存の一次資料を先生の指摘の形に組み直し、
2026-09-06 に実施した 2 つの実験結果を有限標本の観察として追加したものである。

一次資料:

| 内容 | ファイル |
|---|---|
| 識別可能性の理論監査（敵対レビュー済み） | `reports/identifiability/true_k_identifiability_hardened_20260904.md` |
| K 選択理論マップ | `reports/identifiability/k_selection_theory_map_20260905.md` |
| 原論文 / MATLAB / 再現 / 現行の BIC 対応監査 | `reports/k_selection_theory/paper_bic_reproduction_alignment_20260904.md` |
| canonical generator 仕様 | `reports/identifiability/canonical_clean_generator_spec_20260904.md` |
| 研究内容の正本 | `RESEARCH_MASTER.md` §17・§18 |
| 事故台帳 | `KNOWN_ISSUES.md` KI-002・KI-010・KI-019・KI-020・KI-021 |

## 証拠ラベル

| ラベル | 意味 |
|---|---|
| **[PROVED]** | 数式として証明した。条件は明示する |
| **[DERIVED]** | 既存の定義・コード・一次証拠から直接導ける |
| **[EMPIRICAL]** | 有限の実験で観測しただけ。証明ではない |
| **[UNRESOLVED]** | まだ言えない |

---

## 1. Research question

先生の指摘を研究上の問いに直す。

| # | 先生の指摘 | 研究上の問い |
|---|---|---|
| A | 真の潜在次元をどう決めるのか。複数の分布へ拡張すると BIC が満たすべき理論的性質が変わるのではないか | 何を推定対象の「K」と定めるか。現行基準は何を近似しているか。Schwarz BIC の前提はこのモデルで成立するか |
| B | 定式化できるだけでなく、生成モデルとして矛盾なく成立しているか | 各 family の条件付き分布は proper か。joint は正規化可能か。モーメントは有限か |
| C | 真の潜在次元が 1 でも、現実には真のモデルは分からない。観測データからどう当てるのか | 観測分布から K は識別可能か。family ごとにどこまで言えるか |
| D | 実験だけでなく理屈から考える必要がある | 有限標本の観察と理論的主張の境界をどこに引くか |
| E | Y が疎・欠測なら未知の関係の予測に使えるのではないか | sparse-Y と missing-Y を分けたとき、何が言えるか。予測に最適な K は生成モデルの K と一致するか |
| F | データ数が増えたとき・無限大へ近づいたとき何が言えるか | どの漸近レジームを採るか。一致性は証明できるか |

**この文書が答えるのは「何が分かっていて、どこから先が研究課題か」であって、6 問すべてに答えを出すことではない。**

---

## 2. Model definition

コードと canonical documentation から確認した現行モデル。`[DERIVED]` `[CONFIRMED_IN_REPOSITORY]`

```
z_i        ~ N(0, I_K)                          i = 1..n
eta^X_il   = f_l^T z_i                          バイアスなし
eta^Y_ij   = w0 + w z_i^T z_j                   i < j、w0, w はスカラー
x_il | z_i ~ ExpFam_X( eta^X_il )
y_ij | z_i,z_j ~ ExpFam_Y( eta^Y_ij )
```

joint:

```
p(Z, X, Y | theta, K)
  = prod_i p(z_i) × prod_{i,l} p_X(x_il | z_i, theta_X) × prod_{i<j} p_Y(y_ij | z_i, z_j, theta_Y)
```

`theta = { F, w0, w }` ＋ Gaussian-X なら `Sigma_X`（対角）＋ Gaussian-Y なら `sigma_y^2`。
対象 family は Gaussian / Bernoulli / Poisson。

**コードとの対応（一次確認）:**

- `expfam/src/experimental/data_generator_canonical.py`:
  `eta_x = latent @ loadings.T`（309 行）、`gram = latent @ latent.T; eta_y_full = w0 + w * gram`（330-331 行）。
  `link_policy = "canonical_no_clipping_fail_fast"`、`normalization_policy = "none"`。
- `expfam/src/experimental/model_dual_expfam_consistent.py`（lineage E）が推定側。
- `w0, w` は**スカラー**であって行列 `W_Y` ではない（root `CLAUDE.md` §1）。
- X は per-component（列ごとに因子分解）。標準の `DualExpFamLSM` では `family_x` は全列共通。

### 2.1 historical generator は canonical model ではない `[CONFIRMED_IN_REPOSITORY]`

**混同しない。** `expfam/src/data_generator_expfam.py`（historical、H-D）は canonical model と以下で乖離する（KI-021）:

| # | historical の挙動 | canonical model の前提 |
|---|---|---|
| G1 | `Z` を生成後に列 z-score する | `z_i ~ N(0, I_K)` iid。z-score 後は `‖z_j‖² ~ χ²_K` が成立しない |
| G2 | `F` の各行を `‖f_j‖² = 1 − uniq = 0.9` に正規化する | `F` に正規化制約なし |
| G3 | Gaussian-X を生成後に z-score する | `x_i ~ N(F z_i, Sigma_X)` のまま |
| G4 | Poisson の `eta` を `[−20, 10]` で clip する | unclipped canonical link |

**含意:** 後述の識別可能性の命題（P1・P2・P6）は **canonical model についての主張**であり、
historical generator が生成したデータの分布についての主張ではない。
2026-09-06 の historical bridge 実験（§10）はこの欠陥のある generator の**挙動**を測ったのであって、
その正しさを支持するものではない。

---

## 3. Three meanings of K

**この研究で最も事故を起こしてきた区別である。** 4 つに分ける。`[DERIVED]`

| 記号 | 定義 | 本研究での状態 |
|---|---|---|
| `K_GEN` (= `K_TRUE`) | 人工データ生成時に人間が指定した潜在次元。`Z` を何列作ったかという**手続き上の数** | 定義のみ。**推定対象ではない** |
| **`K*`** | `min{ K : P0 ∈ M_K }`。`M_K = { p_{theta,K}(X,Y) : theta ∈ Theta_K }`、`p_{theta,K}(X,Y) = ∫ p(Z) p(X\|Z) p(Y\|Z) dZ` | **定義を確定**（M-closed 前提）。**推定対象の候補** |
| `K^rank` | population Gram `FF^T` の階数。**X 周辺しか見ない** | **P1・P5 が確立するのはこちら** |
| `K_PRED` | 有限標本で予測性能を最大化する次元（held-out score で選ばれる K） | 手続きとしては定義済み。**population target は `[UNRESOLVED]`（U10）** |

**関係:** `K^rank ≤ K* ≤ K_GEN`。**どの不等号も等号とは限らない。** `[DERIVED]`

### 3.1 `K_GEN = K*` と言える条件

`[UNRESOLVED]`（U9）

`K* ≤ K_GEN` は自明（generator の構成が `M_{K_GEN}` の元を作るため）。
**逆向き（`K* ≥ K_GEN`、すなわち等号）は family ごとに証明が要る。**

現時点で言えるのは:

- **Gaussian-Y かつ `w ≠ 0`:** P2 により単一 dyad の周辺分布から `K` が決まるので、**`K* = K_GEN`** `[PROVED]`（条件付き）。
- **Poisson-X かつ `rank(F) = K_GEN`（generic）かつ unclipped link:** P1 により `K^rank = K_GEN`。
  `K^rank ≤ K*` なので **`K* ≥ K_GEN` が従い、`K* = K_GEN`** `[PROVED]`（条件付き）。
- **Bernoulli-Y・Bernoulli-X・Poisson-Y:** `[UNRESOLVED]`。

**重要:** 現在の主要実験（clean true-K sweep、EM continuation）は **X=Poisson / Y=Bernoulli** である。
X 側の P1 が効くので `K* = K_GEN` は**条件付きで言える**が、これは `rank(F)=K` と unclipped link を要求する。
`canonical-clean-v1` はこの 2 条件を満たすよう設計されている（`rank(F)=K_TRUE` を全 64 セルで確認済み）。
**Y 側（Bernoulli-Y）は依然として未解決である。**

### 3.2 `K_PRED = K*` と言える保証

`[UNRESOLVED]`（U10）

**保証はない。** 理由:

1. 厳密に proper な scoring rule であれば、M-closed 下で population 最適が真の分布に一致しうる。
   しかし現行 **S1 は推定した `ẑ` を代入した plug-in raw-eta score であって proper scoring rule ではない**。
2. 有限標本では bias–variance trade-off が別に効く。予測に最適な K が `K*` より小さくなることは
   一般の統計モデルで普通に起きる。
3. **選択（selection）≠ 識別（identification）。** 「held-out で `K_GEN` が選ばれた」は
   「`K_GEN` が識別された」を意味しない。

さらに手続き上の交絡: **データに K を区別する情報がない場合でも、smallest-K tie rule により手続きは `K=1` を出力する。**
clean sweep で `K_TRUE=1` が全 `n` で 4/4 だったのは、この**下限効果と交絡している**。

---

## 4. Generative-model validity（先生の指摘 B）

**properness と moment existence を厳密に分ける。** `[DERIVED]`

### 4.1 Properness

| 概念 | 主張 | 成立条件 |
|---|---|---|
| **properness** | `p(Z, X, Y)` が全確率 1 の確率測度である | finite `n, d, K`、finite parameters、Gaussian 分散 > 0、canonical link |
| **finite moments** | `E[Y^r] < ∞` など | family 依存。**Poisson-Y では追加条件が要る** |

**判定: [PROVED]（条件付き）— 有限パラメータの下で、6 つの family 組合せすべてで joint は proper である。**

理由は 2 段階で、どちらも初等的である:

1. **各条件付き分布が proper。** 有限の `eta` に対し
   Gaussian `N(mu, sigma^2)`（`sigma^2 > 0`）、Bernoulli `sigma(eta) ∈ (0,1)`、
   Poisson `exp(eta) ∈ (0, ∞)` はいずれも全質量 1 の分布である。
   canonical link は `eta ∈ R` を各 family の有効パラメータ空間へ写す全単射なので、
   `eta` が有限であれば分布は必ず定義される。
2. **joint が正規化可能。** 条件付き密度は `Z` の各値で 1 に積分され、`p(Z)` は proper な Gaussian なので、
   Fubini により `∫ p(Z) [∫ p(X|Z) p(Y|Z) dX dY] dZ = ∫ p(Z) dZ = 1`。
   `X` と `Y` は `Z` を与えたとき条件付き独立であり、積は有限個（`nd + n(n−1)/2` 個）の因子からなる。

`eta` の有限性は `theta` が有限であることから従う: `eta^X_il = f_l^T z_i` と `eta^Y_ij = w0 + w z_i^T z_j` は
有限値の連続関数であり、`z_i` は a.s. 有限である。

| family | properness | 備考 |
|---|---|---|
| Gaussian-X | **proper**（`Sigma_X ≻ 0`） | 境界 `Sigma_X = 0` は退化 |
| Bernoulli-X | **proper**（常に） | `sigma(eta) ∈ (0,1)` は `eta` が有限なら常に成立 |
| Poisson-X | **proper**（常に） | `exp(eta) < ∞` が有限 `eta` で成立 |
| Gaussian-Y | **proper**（`sigma_y^2 > 0`） | |
| Bernoulli-Y | **proper**（常に） | |
| Poisson-Y | **proper**（常に） | **ただしモーメントは別問題。§4.2** |

**properness は identifiability を含意しない。** Bernoulli-X は常に proper だが `d=1` では K を識別できない（§5.4）。

### 4.2 Moment existence — Poisson-Y に条件が要る

**[PROVED] P6:** canonical（unclipped）Poisson-Y で

```
S = z_i^T z_j,   M_S(t) = (1 − t^2)^{−K/2}   (|t| < 1)
lambda_ij = exp(w0 + w S)
E[lambda^r] = exp(r w0) · M_S(r w) = exp(r w0) (1 − r^2 w^2)^{−K/2}
E[lambda^r] < ∞  ⟺  |r w| < 1
```

Poisson の階乗モーメント `E[Y(Y−1)···(Y−r+1)] = E[lambda^r]` と、
第 2 種 Stirling 数による `Y^r = Σ_j S(r,j) Y^{(j)}`（全項非負）＋ Lyapunov 不等式により
**`E[Y^r] < ∞ ⟺ E[lambda^r] < ∞`**。したがって

| 量 | 有限となる条件 |
|---|---|
| `E[Y]` | `\|w\| < 1` |
| `Var(Y)` | **`\|w\| < 1/2`** |
| `E[Y^r]` | `\|w\| < 1/r` |

**これは properness の問題ではない。** 分布は proper でサンプルも a.s. 有限である。
発散するのは **population のモーメント**であって個々の実現値ではない。
症状は「標本分散が `n` とともに安定しない」ことであり、分散・相関・CLT 的な議論が使えなくなる。

**観察 O1 `[CONFIRMED_IN_REPOSITORY]`:** historical の `_Y_DEFAULTS["poisson"] = dict(w0=0.0, w=0.5)` は
`|2w| = 1.0` で、**分散発散の境界そのもの**である。
ただし historical generator は `Z` を列 z-score しているので `‖z_j‖² ~ χ²_K` が成立せず、
**O1 は canonical model についての主張であって historical データの分布についての主張ではない**（§2.1）。

**Gaussian-Y・Bernoulli-Y にはこの種の条件は不要。** Bernoulli-Y は `Y ∈ {0,1}` で全モーメントが有限、
Gaussian-Y は `E[Y] = w0`、`Var(Y) = K w^2 + sigma_y^2`（P2 の `kappa_2 = K` から）で常に有限。

**含意:** clean generator で Poisson-Y を使う場合、**`|w| < 1/2` を generator gate として強制すべき**である。

### 4.3 モーメント有限でも identifiability は従わない

`[DERIVED]` モーメントが有限でも、モーメント列が分布を一意に決めるとは限らない（モーメント問題）。
本研究の識別性の議論はすべて「population モーメント写像から特定の関数 `g(theta)` を復元できる」形であり、
**`g(theta)` の functional identifiability の十分条件**であって **`theta` そのものの identifiability ではない**。

---

## 5. Identifiability（先生の指摘 C）

### 5.0 出発点

**異なる `K` で同じ `p(X,Y)` を生成できるなら、データが無限にあっても `K` を区別できない。** `[DERIVED]`

これは統計的手続きの問題ではなく、モデル族の構造の問題である。
どんな推定量・どんな基準を使っても、`P0 ∈ M_K ∩ M_{K'}` で `K ≠ K'` なら区別は不可能である。
したがって **「n を増やせば当たる」と言う前に、まず識別可能性を確認しなければならない。**

### 5.1 共通の不定性

**[PROVED] P4 — `O(K)` 不変性。** `Q ∈ O(K)` に対し `z → Qz`、`F → F Q^T` とすると:

- prior 不変: `Qz ~ N(0, Q I Q^T) = N(0, I_K)`
- X 側: `(Q f_l)^T (Q z) = f_l^T z`
- Y 側: `(Q z_i)^T (Q z_j) = z_i^T z_j`

回転・符号反転・座標置換がすべて含まれる。**観測分布は変わらない。**

**スケールは自由でない。** `z → cz` は prior を `N(0, c^2 I)` に変えるので、
`N(0,I)` prior を固定した時点で潜在スケールは固定される。
（X 側だけ見ると `(cF, c^2 w)` の組が対応するが、prior 固定によりこれも許されない。）

**含意:** `F` そのものは識別できない。**識別に使えるのは `FF^T`（Gram）とその rank である。**
`K` は `FF^T` のレベルでは識別されうる。

**[PROVED] 退化ケース — K が「存在する」ことと「識別できる」ことは別:**

| ケース | 観測分布から K を識別できるか |
|---|---|
| `F = 0` | X は潜在に依存しない。Y 側の識別性に完全に依存 |
| `w = 0` | `eta^Y = w0` は定数。**Y から不可**。Gaussian-Y では `kappa_4 = kappa_6 = 0` で復元公式が `0/0` |
| `F = 0` かつ `w = 0` | **完全に不可**。`X, Y` は潜在に依存しない |
| `rank(F) = r < K` | X からは高々 `r` まで。残り `K−r` 次元は Y 側にのみ現れる |
| `d < K` | X からは高々 `d` まで |
| `w` が非常に小さい | population では識別可（`w ≠ 0` なら）だが**有限標本では実質不可**。`K = kappa_4/(6 w^4)` は `w→0` で不安定。これは estimability の問題 |

### 5.2 Poisson-X — **[PROVED] P1**

モデル: `X_l | z ~ Poisson(exp(f_l^T z))`、`z ~ N(0, I_K)`。

`f_l^T z ~ N(0, ‖f_l‖^2)` と対数正規のモーメント母関数から:

```
E[X_l] = E[ exp(f_l^T z) ] = exp( ‖f_l‖^2 / 2 )                                    ... (1)
```

`l ≠ m` について、`X_l ⊥ X_m | z` なので

```
E[X_l X_m] = E[ E[X_l|z] E[X_m|z] ] = E[ exp( (f_l + f_m)^T z ) ]
           = exp( ‖f_l + f_m‖^2 / 2 )
           = exp( ( ‖f_l‖^2 + ‖f_m‖^2 ) / 2 + f_l^T f_m )                          ... (2)
```

(1)(2) より

```
‖f_l‖^2  = 2 log E[X_l]
f_l^T f_m = log { E[X_l X_m] / ( E[X_l] E[X_m] ) }                                  ... (3)
```

したがって **`FF^T` の全成分が population moment から復元できる**（対角は (1)、非対角は (3)）。

**命題 P1:** unclipped exp link かつ `rank(F) = K`（generic 条件）かつ `d ≥ K` のとき、
`rank(FF^T) = K` であり、**X 周辺の最小潜在次元は `rank(FF^T)` に一致する**。

**省略できない仮定:**

1. **unclipped link.** clip すると (1)(2) の等式が壊れる。historical generator は clip する（§2.1 G4）。
2. **`rank(F) = K`（generic）.** `rank(F) < K` なら `FF^T` の rank は `K` 未満。
3. **`d ≥ K`.** `FF^T ∈ R^{d×d}` の rank は `d` 以下。
4. **population moment.** 有限標本の推定 Gram には適用されない（§5.6）。
5. **候補パラメータ空間全体でモーメントが有限**であること（Poisson-X は満たす）。

**限定（省略不可）:** P1 が確立するのは **`K^rank`** であって `K*` ではない。
`K^rank ≤ K*` は成り立つが、**等号は自明ではない**。
「Poisson-X なら常に K が識別可能」とは**書かない**。

### 5.3 Gaussian-X — **[PROVED] P5（Σ 既知）／[UNRESOLVED] U3（Σ 未知）**

`x_i = F z_i + eps_i`、`eps_i ~ N(0, Sigma)` のとき

```
Cov(X) = F F^T + Sigma
```

**Σ 既知なら** `FF^T = Cov(X) − Sigma` が直接得られ、`rank(F) = K`（generic）の下で
**X 周辺の最小潜在次元 = `rank(Cov(X) − Sigma)`**。`[PROVED] P5`

**Σ 未知（対角）は因子分析の古典的識別問題**で、本研究では未解決（U3）。
パラメータ数のカウンティングから得られる**必要条件**は

```
(d − K)^2 ≥ d + K
```

（観測共分散の自由度 `d(d+1)/2` と、パラメータ数 `dK − K(K−1)/2 + d` の比較から）。
これは **generic 点での local 識別の必要条件**であって十分条件ではない。
例: `d=2, K=1` は `(2−1)^2 = 1 < 3 = d+K` で必要条件を満たさない。

### 5.4 Bernoulli-X — **[UNRESOLVED] U1**

**[PROVED] P7 — 一次モーメントは無情報.** `z ~ N(0,I_K)` は 0 対称なので `f_l^T z` も 0 対称。
`sigma(-u) = 1 − sigma(u)` より

```
E[X_l] = E[ sigma(f_l^T z) ] = 1/2      （K, f_l によらず）
```

**反例 C1（`d = 1`）:** `d=1` では観測は `X_1 ∈ {0,1}` のみで、その分布は `E[X_1] = 1/2` により
**`K` と `f_1` に依存しない**。したがって **X 周辺からは `K` を識別できない**。

**重要な限定:** これは **X 周辺の反例であって joint の反例ではない。**
Y 側が informative なら joint では識別されうる（例: Gaussian-Y かつ `w ≠ 0` なら P2 が `K` を決める）。

**`d > 1` の一般的な K identifiability は [UNRESOLVED]（U1）。**
高次モーメントは無情報ではない（`E[X_l X_m]` は `f_l^T f_m` に依存する）が、
`FF^T` を復元する閉形式は得られていない。
**「Bernoulli では K は識別できない」とは書かない。**

### 5.5 Y 側

#### 5.5.1 Gaussian-Y — **[PROVED] P2・P3・P8**

`S = z_i^T z_j` の MGF は、`z_j` を固定すると `S | z_j ~ N(0, ‖z_j‖^2)` で `E[e^{tS} | z_j] = exp(t^2 ‖z_j‖^2 / 2)`、
`‖z_j‖^2 ~ chi^2_K` の MGF を代入して

```
M_S(t) = (1 − t^2)^{−K/2}          (|t| < 1)
```

キュムラントは `kappa_2 = K`、`kappa_4 = 6K`、`kappa_6 = 120K`。
`Y_ij = w0 + w S + eps` の中心キュムラントは `w^r kappa_r`（`eps` は Gaussian なので 4 次以上に寄与しない）。よって

```
w^2 = kappa_6^Y / (20 kappa_4^Y),    K = kappa_4^Y / (6 w^4),    sigma_y^2 = kappa_2^Y − K w^2
```

**命題 P2:** `w ≠ 0` のとき **単一 dyad の周辺分布から `(w0, w^2, K, sigma_y^2)` が完全に決まる。**

**識別されないもの:** `w` の符号は単一 dyad からは決まらない（`w^2` しか出ない）。

**命題 P8:** `E[(Y_ij − w0)(Y_ik − w0)(Y_jk − w0)] = w^3 K` により、
**三角形（`n ≥ 3`）から `w` の符号が識別される。**

**命題 P3（非入れ子性）:** `{P ∈ M_K : w ≠ 0}` は `M_{K+1}` と交わらない。
（P2 により `K` が周辺分布の関数として一意に決まるため。）
**ただし `M_K` と `M_{K+1}` 自体は `w = 0` 切片で交わる。** 「互いに素」は**偽**。

#### 5.5.2 Bernoulli-Y — **[UNRESOLVED] U2（実験で実際に使っている family）**

**単一 edge の周辺分布と network 全体の依存構造を分ける。**

- **単一 edge:** `P(Y_ij = 1) = E[ sigma(w0 + w S) ]`。これは `(w0, w, K)` の 1 つのスカラー関数にすぎず、
  異なる `(w0, w, K)` が同じ edge density を与えうる。
  特に `w0 = 0` では `S` の 0 対称性と `sigma(-u) = 1 − sigma(u)` から `P(Y_ij=1) = 1/2` となり、
  **edge density は完全に無情報**（P7 の Y 版）。
- **network 全体:** `K` の情報は**共有ノードを持つ motif**（三角形、2-star、共通近傍数など）に入る。
  `Y_ij` と `Y_ik` は `z_i` を共有するので独立でない。この依存構造が `K` を決めるかは未証明。

**判定: [UNRESOLVED]。** 一般の K identifiability は証明されていない。

**これが最も重要な未解決点である。** 主要実験（clean true-K sweep、EM continuation、historical bridge）は
すべて **Y = Bernoulli** で走っている。**強い理論結果 P2・P3 は Gaussian-Y 限定であり、これらの実験を正当化しない。**

#### 5.5.3 Poisson-Y — moment existence と identifiability を分ける

- **moment existence: [PROVED] P6**（§4.2）。`E[Y^r] < ∞ ⟺ |w| < 1/r`。
- **identifiability: [UNRESOLVED] U4。**

部分的な結果: `|w| < 1/3` なら `a_r = E[lambda^r]`（`r = 1,2,3`）がすべて有限で、
`b_r = log a_r = r w0 − (K/2) log(1 − r^2 w^2)` から `w0` を消去すると

```
R(w^2) = ( 2 b_1 − b_2 ) / ( 3 b_1 − b_3 )
```

が `w^2` のみの関数になる。`R` が `(0, 1/9)` 上で狭義単調なら `w^2`、続いて `K`, `w0` が決まる。

**グリッド 4000 点の数値確認では狭義単調（`0.3333 → 0.0313`）だが、解析的な単調性は証明していない。**
したがってこれは **[HYPOTHESIS] + 数値的裏付けあり、証明は [UNRESOLVED]**。

### 5.6 identifiability と estimability は別

`[DERIVED]` population で識別可能でも、有限標本から推定できるとは限らない。

**一次証拠:** clean true-K sweep（896 fits）の Poisson-X Gram 診断（S4）は、
**全 64 セルで推定 Gram の最小固有値が負**（中央値 −1.80〜−0.52）であり、
推定 Gram は **PSD 錐の外**にある。閾値なし階数は常に `d = 15`。
事前に固定できる rank 閾値がないため、**S4 は selected K を作らない**（U7）。

**P1 は population moment についての主張であり、有限標本の推定 Gram には適用されない。**

---

## 6. Criterion / BIC taxonomy（先生の指摘 A）

### 6.1 3 つの尤度を絶対に混同しない `[DERIVED]`

| 記号 | 式 | `p(Z)` | `Z` を積分 | 呼び名 |
|---|---|:---:|:---:|---|
| **Q1** | `ln p(X, Y \| Z, theta)` | なし | しない | conditional / plug-in likelihood |
| **Q2** | `ln p(Z, X, Y \| theta)` | **あり** | しない | complete joint log density |
| **Q3** | `ln p(X, Y \| theta) = ln ∫ p(Z,X,Y\|theta) dZ` | あり | **する** | **observed-data marginal likelihood** |

**Schwarz BIC が近似するのは Q3 だけである。**

### 6.2 各基準が何を評価しているか `[CONFIRMED_IN_REPOSITORY]` / `[SUPPORTED_BY_PRIMARY_SOURCE]`

| 系統 | 当てはまり項 | 型 | `p(Z)` | MC 平均 | 位置づけ |
|---|---|:---:|:---:|:---:|---|
| **原論文 Eq.(26)** | Eq.(16) の `ln L` | **Q1** | なし | **本文から特定不能** | 論文上は `BIC` と記載。**Q3 ではない** |
| MATLAB `calcdescmetric_ver4.m` | `ln p(X\|Z)+ln p(Y\|Z)+ln p(Z)` の平均 | **Q2 の MC 平均** | あり | あり | Q 関数を計算していることは確認済み。paper Exp.2 に実際に代入されたかは `[UNRESOLVED]` |
| Python 再現 `experiment_paper_2.py` | 1 個の `Z` に対する `ln p(Z)+ln p(X\|Z)+ln p(Y\|Z)` | **Q2 plug-in** | あり | **なし** | コメント上は Eq.(26) 再現だが当てはまり項は Eq.(16) と一致しない |
| historical `calc_bic_dual` | `Q_strict` | **Q2 の MC 近似** | あり | あり | **legacy Q ベース基準。Schwarz BIC と呼ばない** |
| 本研究 **S1** | held-out plug-in raw-eta mean log score | **尤度ではない** | — | — | predictive criterion。population target は `[UNRESOLVED]`（U10） |
| 本研究 **S2** | `−2 Q_strict + p log n` | **Q2** | あり | あり | Q ベース基準（`calc_bic_dual` と同型） |
| 本研究 **S3** | `Ẑ` を代入、`p(Z)` なし、積分なし | **Q1** | なし | なし | **本研究が定義した基準。原論文 Eq.(26) ではない** |
| standard Schwarz BIC | — | **Q3** | `Z` を積分 | — | 上記のいずれとも別 |

### 6.3 答えるべき 3 つの問い

**Q. historical に「BIC」と呼ばれていたものは standard Schwarz BIC か。**
**A. 違う。** `calc_bic_dual` の当てはまり項は `Q_strict`（Q2 の MC 近似）であり、
観測データ周辺尤度 Q3 ではない。`[CONFIRMED_IN_REPOSITORY]`
**関数名 `calc_bic_dual`・CSV 列名 `BIC`・過去結果の呼称は変更しない**（過去の結果を引用可能に保つため）。
**呼び方だけを変える: Q ベース complete-data criterion / ICL 型。**（KI-010）

**Q. 原論文 Eq.(26) は何を使っているか。**
**A. Eq.(16) の `ln L`、すなわち Q1 型である。** `[SUPPORTED_BY_PRIMARY_SOURCE]`
論文は `BIC` と記載しているが、**Q3 ではない**。
さらに**評価手続き（どの `Z` を代入するか、MC 平均をとるか）は本文から特定不能**であり、
そこは `[UNRESOLVED]` である。
**「原論文 Eq.(26) は standard Schwarz BIC である」とは書かない。**

**Q. 現行 S1 は BIC の一種か。**
**A. 違う。** S1 は held-out dyad 上の予測スコアであり、尤度でも情報量基準でもない。
**BIC とは別種の predictive criterion である。**

### 6.4 Q1 型基準への警告（有限標本の観察）

`[EMPIRICAL]` clean true-K sweep で **S3（Q1 型）は 64 セル中 3 しか `K_TRUE` に一致せず、
ほぼ全セルで候補上限 `K = 7` を選んだ**（over-selection 61/64）。

理由は構造的である: `p(Z)` を含めず `Z` を積分しない基準は、
**潜在次元を増やすほど代入した `Ẑ` への当てはまりが良くなり、`p log n` の罰則が追いつかない。**

**これは Q1 型基準への警告であって、原論文 Eq.(26) の評価ではない**（評価手続きが特定不能なため）。
**「S3 の失敗は原論文 BIC の失敗である」とは書かない。**

---

## 7. Consistency requirements（先生の指摘 A・D）

### 7.1 standard Schwarz BIC が通常要求するもの

`[DERIVED]`

1. 真のモデルが候補族に含まれる（M-closed）
2. identifiability（パラメータ写像が単射）
3. regularity（Fisher 情報が正則、LAN が成立）
4. 固定次元のパラメータ化（`n` とともにパラメータ数が増えない）
5. 尤度の適切な漸近挙動（Laplace 近似が有効）
6. 有効標本数が定義されている

### 7.2 本モデルでどれが成立しないか

**よくある誤り:「モデル族が入れ子でないから BIC が使えない」。これは誤りである。**
Schwarz BIC の導出はモデルごとの Laplace 近似であり、非入れ子モデルの比較に BIC を使うのは標準的である。
**非入れ子性（P3）が無効にするのは尤度比検定の χ² 近似と Wilks の定理**であって、BIC の導出ではない。

**BIC がこのモデルで正当化されない実際の理由は 3 つ。** `[DERIVED]`

| # | 理由 | 詳細 |
|---|---|---|
| **1** | **モデルが特異（singular）** | `O(K)` 不変性により Fisher 情報が `K(K−1)/2` 次元の軌道方向に退化する。加えて `rank(F) < K`・`w = 0` の縮退集合上でも退化する。特異モデルでは `(p/2) log n` が **RLCT（実対数閾値）**に置き換わるが、**本モデルの RLCT は未知** |
| **2** | **境界パラメータ** | `Sigma_X ⪰ 0`、`sigma_y^2 ≥ 0`、因子分析の `Psi ⪰ 0`。`K` の比較はしばしば境界上で起きる |
| **3** | **有効標本数が未定義** | ノード数 `n` / dyad 数 `n(n−1)/2` / X 要素数 `nd` の 3 通りがあり、オーダーが違う。`calc_bic_dual` は `log n` に**ノード数**を使うが、Y チャネルが供給する情報は `n(n−1)/2` dyad 分である |

**さらに 4 つ目:** 潜在変数 `Z ∈ R^{n×K}` は `nK` 個の座標を持ち **`n` とともに増える**。
これは古典的 iid 設定ではなく **incidental parameter / Neyman–Scott 型**の状況である。
`Q_strict` は `Z` を完全データとして扱う一方、罰則 `p̂` は構造パラメータだけを数え `Z` を数えない。

**実装事実 `[CONFIRMED_IN_REPOSITORY]`:**
`num_params = kd − k(k−1)/2 + [d if X gaussian] + [1 if Y gaussian]`。
`−k(k−1)/2` は `O(K)` 軌道の次元を既に引いている。**`w0` と `w` は数えられていない**（原論文の慣行）。
Gaussian-Y では `K` の情報をもっぱら `w` が運ぶことが P2 から判明したので検討に値するが、
**`K` に依存しない定数なので順位には影響しない。「誤り」とは断定しない。**

### 7.3 結論

**「BIC consistency が証明済み」ではない。「証明されていない」でもなく、「前提が確認されていない」。**

**未確認の条件を列挙する:**

| # | 未確認の条件 |
|---|---|
| C1 | M-closed（実データでは一般に成立しない。その場合 `K*` は存在しない） |
| C2 | Bernoulli-Y の identifiability（U2）— **実験で使っている family** |
| C3 | 本モデルの RLCT（特異モデルなので `p/2` は使えない） |
| C4 | 有効標本数の定義（`n` / `n(n−1)/2` / `nd`） |
| C5 | incidental parameter `Z` の扱い |
| C6 | dyad の従属性の下での漸近論 |
| C7 | 境界パラメータでの Laplace 近似 |
| C8 | `M_K` の閉性（U11）。`P0 ∉ M_{K+1}` は `P0 ∉ closure(M_{K+1})` を含意しない |

**そもそも現行の S2 / `calc_bic_dual` は Q3 を使っていないので、
上の条件がすべて満たされたとしても Schwarz の議論は適用できない。**

---

## 8. Asymptotic regimes（先生の指摘 F）

**まず「何が無限大に行くのか」を決めなければならない。本研究ではそれ自体が確定していない。** `[UNRESOLVED]`（U6）

| レジーム | 内容 | 状態 |
|---|---|---|
| **A** | `n → ∞`、`d` fixed、`K` fixed | 最も標準的だが、`Z ∈ R^{n×K}` が `n` とともに増える incidental parameter 問題が残る |
| **B** | `n → ∞`、Y dense（全 `n(n−1)/2` dyad を観測） | Y の情報量は `O(n^2)`。X の情報量は `O(nd)`。**2 チャネルのオーダーが違う** |
| **C** | `n → ∞`、Y sparse（観測 dyad が `o(n^2)`） | 疎性の速度をどう置くかで結論が変わる。ネットワーク統計の標準的な難所 |
| **D** | `d` も増える | 高次元因子分析の領域。`d/n → gamma` などの設定が必要 |

**Y について必ず明記すべきこと `[DERIVED]`:**

> 観測ペア数は `O(n^2)` だが、`z_i` を `n−1` 本の dyad が共有するため、
> **単純に `n^2` 個の iid observation として扱うことはできない。**

具体的に: `Y_ij` と `Y_ik` は `z_i` を共有するので独立でない。
P1・P2 の有限標本版は「`n` 個の潜在ベクトルを共有する**従属な** dyad の平均」であり、
**network 1 個からの `kappa_4`・`kappa_6` 推定の一致性・収束率は本研究では示していない。**

**したがって `P(K̂ = K*) → 1` のような一致性を議論するには、まず**

1. どのレジーム（A/B/C/D）を採るか
2. 有効標本数を何と定義するか
3. どの family で識別可能性が成立しているか

**の 3 点を固定しなければならない。現時点ではどれも固定されていない。**

**本研究は `n → ∞` で何が起きるかについて何も証明していない。**
**「n を増やせば `K_TRUE` に収束する」とは書かない。**

---

## 9. Sparse / missing Y（先生の指摘 E）

### 9.1 2 つを分ける `[DERIVED]`

| | 意味 | 統計的な扱い |
|---|---|---|
| **sparse Y** | edge は**観測されている**が、その値の多くが 0（疎なネットワーク） | 欠測ではない。`y_ij = 0` は情報である |
| **missing Y** | dyad の一部が**観測されていない** | 欠測。observed 部分から `Z` を推定し、未観測 dyad を予測できる |

**混同すると「疎だから予測できる」という誤った推論になる。**

### 9.2 missing Y の使い方

observed `X, Y_obs` → `Z` 推定 → 未観測 `Y_ij` 予測、という使い方は**構造的に可能**である。
`Y` の欠測が `Z` と独立（MCAR/MAR）なら、observed dyad の尤度だけで `Z` の事後は定義できる。

**現行の held-out S1 はこの設定に近い。** clean true-K sweep は `test_ratio = 0.20` で
dyad を train/test に分割し、train で推定した `ẑ` を使って test dyad の予測スコアを計算している。

### 9.3 ただし予測に最適な K と `K*` は一致するとは限らない

`[UNRESOLVED]`（U10、§3.2 と同じ）

**`K_PRED = K*` の保証はない。**
S1 は proper scoring rule ではなく、有限標本では bias–variance trade-off が別に効く。
**「held-out なら真の K を選べる」とは書かない。**

### 9.4 既存の関連証拠 `[EMPIRICAL]`

| 実験 | 設定 | 観察 | 限定 |
|---|---|---|---|
| Issue #27 sparse-Y | 人工データ、`y_obs_rate = 0.1`、n=80, d=9, K_TRUE=3、lineage E+F | 複数属性を同時に使う条件が比較条件より RMSE_Z を改善（`y_only` 比 +0.5122、10/10）。dense-Y では +0.0087〜+0.0510 まで縮小 | **prototype・本文採用不可。**「一般に per-column が優れる」「実データでも優れる」とは書かない |
| Issue #33 MovieLens | 実データ、user-disjoint split 30 回、K=3 固定 | `Delta_s` 平均 +0.004248、正 23/30、経験的 p10–p90 が **0 をまたぐ** | **「有効性を確認」「statistically significant」とは書かない。** 30 splits は同じ 943 users を再利用しており independent replicate ではない |

**含意:** sparse/missing Y の予測利用は**方向としては支持されているが、効果量は小さく、実データでは split 間変動に埋もれる。**
**この用途で `K` を選ぶことの理論的正当化は別問題であり、未解決である。**

---

## 10. Existing empirical evidence

**以下はすべて `[EMPIRICAL]` であり、理論の証明ではない。有限標本での観察である。**
**consistency の証明には使わない。**

### 10.1 K_TRUE = 5、X=Poisson / Y=Bernoulli の n 依存

| n | OLD historical（lineage historical、5 trials、Q ベース基準、candidate K 1..10） | CURRENT clean 8 iter（lineage E、8 replicates、S1、candidate K 1..7） | CURRENT clean 40 iter |
|---|---|---|---|
| 50 | **1/5** | 2/8 | 2/8 |
| 75 | **4/5** | 0/8 | 1/8 |
| 100 | **5/5** | 4/8 | 6/8 |
| 150 | **5/5** | 8/8 | 7/8 |

**一次証拠:**
OLD = `expfam/results/k_selection/historical_pb_k5_n_sweep_20260906/`（n=150 は `exp2_bic_A.csv` と 50/50 bit-identical）。
CURRENT = `expfam/results/k_selection/clean_true_k_asymptotics_20260904/` および
`expfam/results/k_selection/em_convergence_sensitivity_20260905/`。

**系列が違う。同じ表に入れているのは n 依存の向きを見るためだけであり、成功率を統計的に比較していない**（KI-002）。
trial 数・criterion・signal 構成・generator・training mask がすべて異なる。

### 10.2 固定 K=3 での選択成功

| 実験 | lineage | 設定 | 結果 |
|---|---|---|---|
| `fixed_official/exp1_k9` | **C（fixed、0.5 なし）** | n=150、candidate K 1..9、10 trials × 3 scenario（A: P/B、B: G/P、C: B/G） | **30/30** で K=3 を選択 |
| `exp2_bic_{A,B,C}.csv` の `k_true=3` | **historical** | n=150、candidate K 1..10、5 trials × 3 scenario | **15/15**（各 5/5） |

**この 2 つは別の実験である。「30/30」は fixed 系列の値であり、historical 系列の値ではない**（KI-002）。

### 10.3 これらから言えること・言えないこと

**言えること:**

1. 有限標本では **K 選択は条件依存**である（family・n・criterion・generator によって結果が変わる）。
2. **small-n で悪化する例がある**（OLD historical でも n=150 の 5/5 が n=50 で 1/5 まで下がる）。
3. **Poisson だから K 選択不能ではない**（n が十分なら OLD・CURRENT とも 5/5・8/8 を達成している）。
4. **8 iteration 固定だけが原因ではない**（40 反復へ継続しても一貫して改善しなかった。EM continuation 実験）。
5. **Q1 型基準（S3）は over-selection する**（64 セル中 61 で候補上限を選択）。

**言えないこと:**

- consistency（`n → ∞` で `K̂ → K*`）の証明
- 「n を増やせば `K_TRUE` に収束する」（真値一致数は単調でない。CURRENT S1 は n=75 でいったん 0/8 に下がる）
- 「どの criterion が正しいか」（S1 vs S2 の一致は 64 中 44、S1 vs S3 は 2、三者一致 0）
- 「sample size が唯一の原因である」（signal・generator・criterion・Y hold-out の寄与は分離していない）

**特記:** OLD が n=75 で 4/5、CURRENT が 0/8 と最も大きく食い違う。
**ここが sample size 以外の差が効いていることを示す最も明確な手掛かりだが、原因は切り分けていない。**

---

## 11. What is established / unresolved

### 11.1 先生の 6 指摘への回答表

| 先生の指摘 | 現時点で答えられること | 条件付きで言えること | 未解決 |
|---|---|---|---|
| **1. 真の K / BIC** | 「K」には `K_GEN`・`K*`・`K^rank`・`K_PRED` の 4 つがあり、`K^rank ≤ K* ≤ K_GEN`。現行 `calc_bic_dual`・S2 は **Q2 ベースで Schwarz BIC ではない**。原論文 Eq.(26) は **Q1 型**で Q3 ではない。Q1 型は構造的に over-selection する（S3 で 61/64） | BIC が正当化されない理由は**非入れ子性ではなく**、①モデルの特異性 ②境界パラメータ ③有効標本数未定義 の 3 つ | Schwarz BIC の前提のうち C1–C8 が未確認。本モデルの **RLCT が未知**。有効標本数の定義も未定 |
| **2. 生成モデル成立性** | **[PROVED]** 有限パラメータの下で 6 family 組合せすべてで joint は proper。各条件付き分布が proper かつ Fubini で正規化可能 | Poisson-Y は **`\|w\| < 1/r` で `E[Y^r] < ∞`**（分散は `\|w\| < 1/2`）。properness とモーメント有限性は別問題 | モーメント有限でも identifiability は従わない（モーメント問題） |
| **3. 真のモデルをどう当てるか** | 異なる `K` が同じ `p(X,Y)` を作るなら**データが無限でも区別不能**。まず識別可能性を確認しなければならない。**[PROVED]** Poisson-X（P1）、Gaussian-X Σ 既知（P5）、Gaussian-Y（P2・P8）、Gaussian-Y の非入れ子性（P3） | P1 は unclipped link・`rank(F)=K`・`d ≥ K`・population moment を要求し、確立するのは **`K^rank`** であって `K*` ではない | **Bernoulli-Y（U2）が未解決 — 実験で実際に使っている family。** Bernoulli-X `d>1`（U1）、Gaussian-X Σ 未知（U3）、Poisson-Y（U4）も未解決 |
| **4. 理論から考える** | 3 つの尤度（Q1/Q2/Q3）の区別、4 つの K の区別、6 つの identifiability の意味、properness と moment の分離、identifiability と estimability の分離 — これらはすべて固定した | 実験結果は「有限標本での観察」としてのみ使う。理論の証明には使わない | `M_K` の閉性（U11）、誤指定下の pseudo-true `K`（`K°`）、候補集合の連結性（U12） |
| **5. sparse / missing Y** | sparse（0 が多い）と missing（未観測）は**別**。missing なら observed から `Z` を推定して未観測 dyad を予測でき、現行 held-out S1 はこの設定に近い | 人工データでは sparse-Y で複数属性同時利用が RMSE_Z を改善（prototype）。実データ MovieLens では平均は正だが split 間変動に埋もれる | **予測に最適な `K_PRED` が `K*` と一致する保証はない（U10）。** S1 は proper scoring rule ではない |
| **6. n → ∞** | どのレジーム（A: n→∞ / B: dense / C: sparse / D: d も増加）を採るかを**まず決めなければならない**。Y は `O(n^2)` ペアあるが `z_i` を共有するので **iid ではない** | — | **どの基準についても一致性は証明していない（U6）。** 有効標本数の定義、incidental parameter `Z` の扱い、dyad の従属性下での漸近論がいずれも未確定 |

### 11.2 Claim Ledger

#### [PROVED]

| # | 主張 | 条件 |
|---|---|---|
| 1 | 有限パラメータの下で canonical joint `p(Z,X,Y)` は proper。6 family 組合せすべて | finite `n,d,K`、finite parameters、Gaussian 分散 > 0、canonical link |
| 2 | **P4** `O(K)` 回転・符号反転・座標置換は観測分布を変えない。`N(0,I)` prior がスケールを固定する | — |
| 3 | **P1** Poisson-X で `‖f_l‖²=2 log E[X_l]`、`f_l·f_m = log(E[X_lX_m]/(E[X_l]E[X_m]))` により `FF^T` が population moment から復元でき、**X 周辺の最小潜在次元 = `rank(FF^T)`** | unclipped exp link、`rank(F)=K`（generic）、`d ≥ K`、population。**`K^rank` の主張であって `K*` ではない** |
| 4 | **P5** Gaussian-X で `Σ` 既知なら **X 周辺の**最小潜在次元 = `rank(Cov(X) − Σ)` | `rank(F)=K`（generic） |
| 5 | **P2** Gaussian-Y で `M_S(t)=(1−t²)^{−K/2}` → `κ₂=K, κ₄=6K, κ₆=120K` → 単一 dyad から `(w0, w², K, σ_y²)` が決まる | `w ≠ 0`。**`w` の符号は決まらない** |
| 6 | **P8** Gaussian-Y で `E[(Y_ij−w0)(Y_ik−w0)(Y_jk−w0)] = w³K` により `w` の符号が三角形から識別される | `w ≠ 0`、`n ≥ 3` |
| 7 | **P3** Gaussian-Y で `{P ∈ M_K : w≠0} ∩ M_{K+1} = ∅`（族は入れ子でない） | `w ≠ 0`。**`M_K` と `M_{K+1}` 自体は `w=0` 切片で交わる** |
| 8 | **P6** Poisson-Y で `E[Y^r] < ∞ ⟺ \|w\| < 1/r`（平均 `\|w\|<1`、分散 `\|w\|<1/2`） | unclipped link |
| 9 | **P7** Bernoulli-X および `w0=0` の Bernoulli-Y では一次モーメントが `K` の情報を持たない（常に 1/2） | `Z` が 0 対称 |
| 10 | 反例: Bernoulli-X `d=1` では **X 周辺から** `K` を識別できない | **joint の反例ではない** |

#### [DERIVED]

| # | 主張 |
|---|---|
| 1 | `K^rank ≤ K* ≤ K_GEN` で、どの不等号も等号とは限らない |
| 2 | Poisson-X + `rank(F)=K` + unclipped の下では `K* = K_GEN`（P1 と `K^rank ≤ K*` から） |
| 3 | Gaussian-Y + `w≠0` の下では `K* = K_GEN`（P2 から） |
| 4 | 現行 `calc_bic_dual` / S2 は Q2 を使っており Q3 を使っていない。**Schwarz BIC ではない** |
| 5 | 原論文 Eq.(26) の当てはまり項 Eq.(16) は Q1 型であり Q3 ではない |
| 6 | BIC が正当化されない理由は非入れ子性ではなく、①特異性 ②境界 ③有効標本数未定義 |
| 7 | Q1 型基準は潜在次元とともに当てはまりが単調改善するので over-selection しやすい |
| 8 | Y は `O(n²)` ペアあるが `z_i` を共有するので iid ではない |
| 9 | properness は identifiability を含意しない。finite moments も identifiability を含意しない |
| 10 | historical generator は canonical model と 4 点で乖離する（Z の z-score、F 行正規化、X の z-score、η clipping） |

#### [EMPIRICAL]

| # | 観察 | 一次証拠 |
|---|---|---|
| 1 | OLD historical P/B `K_TRUE=5` の per-trial 選択は n=50 で 1/5、n=75 で 4/5、n=100 で 5/5、n=150 で 5/5 | `historical_pb_k5_n_sweep_20260906/` |
| 2 | n=150 の 50 セルは `exp2_bic_A.csv` と BIC・rmse_Z が bit-identical | 同上 `replication_gate.json` |
| 3 | CURRENT clean P/B `K_TRUE=5`（S1）は 8 iter で 2/8, 0/8, 4/8, 8/8、40 iter で 2/8, 1/8, 6/8, 7/8 | `clean_true_k_asymptotics_20260904/`、`em_convergence_sensitivity_20260905/` |
| 4 | 8→40 反復へ継続しても K 選択は一貫して 5 へ改善しなかった。挙動は n 依存かつ checkpoint 間で非単調 | `em_convergence_sensitivity_20260905/audit_report.json` |
| 5 | S3（Q1 型）は 64 セル中 3 しか一致せず、61 セルで候補上限 K=7 を選択 | `clean_true_k_asymptotics_20260904/` |
| 6 | 推定 Poisson-X Gram は全 64 セルで最小固有値が負（PSD ですらない）。閾値なし階数は常に d=15 | 同上 `gram_spectrum.csv` |
| 7 | fixed 系列 `exp1_k9` は K_TRUE=3 を 3 scenario × 10 trials で 30/30 選択 | `fixed_official/exp1_k9/` |
| 8 | criterion 一致は S1 vs S2 44/64、S1 vs S3 2/64、S2 vs S3 0/64、三者一致 0/64 | `clean_true_k_asymptotics_20260904/` |

#### [UNRESOLVED]

| ID | 内容 |
|---|---|
| **U2** | **Bernoulli-Y の一般 K identifiability — 実験で実際に使っている family。最重要** |
| U1 | Bernoulli-X（`d>1`）の identifiability |
| U3 | Gaussian-X（Σ 未知対角）の十分条件 |
| U4 | Poisson-Y の identifiability（`R(w²)` の解析的単調性が未証明） |
| U5 | Bernoulli-Y / Poisson-Y の非入れ子性 |
| **U6** | **どの基準についても `n→∞` の一致性。漸近レジーム自体が未確定** |
| U7 | 有限標本の rank 閾値（推定 Gram は PSD ですらない） |
| U9 | clean construction で `K* = K_GEN` か（family ごと。Bernoulli-Y で未解決） |
| **U10** | **held-out plug-in score（S1）の population target。`K_PRED = K*` の保証なし** |
| U11 | `M_K` の閉性、誤指定下の pseudo-true `K`（`K°`） |
| U12 | 候補集合 `{K : P0 ∈ M_K}` の連結性 |
| — | 本モデルの **RLCT**（特異モデルなので `p/2` は使えない） |
| — | **有効標本数の定義**（`n` / `n(n−1)/2` / `nd`） |
| — | 原論文 Eq.(26) の評価手続き（どの `Z`、MC 平均か）— 本文から特定不能 |
| — | OLD が n=75 で 4/5、CURRENT が 0/8 と食い違う原因（signal / generator / criterion / Y hold-out が交絡） |

### 11.3 絶対に書いてはいけないこと

- 「BIC は理論的に正しい」／「BIC は非入れ子だから使えない」（**後者は理由が誤り**）
- 「held-out なら真の K を選べる」／「S1 は `K*` を推定している」
- 「Poisson-X なら常に K が識別可能」／「Bernoulli では K は識別できない」
- 「K 選択の一致性を証明した」／「n を増やせば `K_TRUE` に収束する」
- 「実データで `K*` を推定した」
- 「S3 の失敗は原論文 BIC の失敗である」
- 「8 反復が under-selection の原因だった」／「選択基準が正しい／誤っている」
- 「sample size が過去と現在の差の唯一の原因である」
- 「historical generator が正しい」／「canonical generator が悪い」

---

## 12. Advisor discussion points

### A. 30 秒説明

> 「真の潜在次元 K」には少なくとも 4 つの意味があり、混同していました。
> 生成時に指定した `K_GEN`、観測分布を表現できる最小次元 `K*`、X 側の Gram の階数 `K^rank`、
> 予測が最良になる `K_PRED` です。関係は `K^rank ≤ K* ≤ K_GEN` で、等号は自明ではありません。
> **生成モデルとしての成立性（properness）は 6 つの family 組合せすべてで証明できました。**
> **識別可能性は Poisson-X と Gaussian-Y では証明できましたが、
> 実験で実際に使っている Bernoulli-Y では未解決です。**
> また、これまで「BIC」と呼んでいたものは観測データの周辺尤度を使っておらず、
> **Schwarz BIC ではありません。**
> `n → ∞` の一致性は、どの漸近レジームを採るかすら決まっていないので、証明していません。

### B. 2 分説明

> **まずモデルが生成モデルとして成立するかを確認しました。**
> 有限パラメータなら各条件付き分布は proper で、`Z` を積分しても全質量 1 になるので、
> Gaussian / Bernoulli / Poisson の X 側 3 通り × Y 側 3 通りすべてで joint は proper です。
> ただし **properness とモーメントの有限性は別問題**で、Poisson-Y では `E[Y^r] < ∞` に `|w| < 1/r` が要ります。
> 分散が有限なのは `|w| < 1/2` のときだけで、**古い generator の既定値 `w = 0.5` はちょうどその境界**でした。
>
> **次に「真の K」を定義し直しました。** 一番大事なのは、
> **異なる K が同じ観測分布を作るなら、データが無限にあっても K は区別できない**という点です。
> なので基準の議論の前に識別可能性を確認する必要があります。
> Poisson-X では population のモーメントから `FF^T` を復元でき、その階数が X 周辺の最小次元になります。
> Gaussian-Y では `z_i^T z_j` のキュムラントから、**単一の dyad だけで `K` が決まります**。
> **しかし実験で使っている Bernoulli-Y は未解決です。** 単一 edge の density はスカラー 1 個で情報が足りず、
> `K` の情報は共有ノードを持つ motif に入りますが、それが `K` を決めるかは証明していません。
>
> **BIC については、混同を 3 つに整理しました。**
> `Z` を代入するだけの Q1、`p(Z)` を掛けるが積分しない Q2、`Z` を積分する Q3 です。
> **Schwarz BIC が近似するのは Q3 だけ**ですが、現行実装も原論文の式も Q3 ではありません。
> さらにこのモデルは `O(K)` 回転で観測分布が変わらない**特異モデル**なので、
> 罰則 `(p/2) log n` の理論的根拠がそもそも成り立たず、代わりに必要な RLCT は未知です。
> 有効標本数もノード数 `n` か dyad 数 `n(n−1)/2` か決まっていません。
>
> **実験は有限標本の観察としてのみ使っています。**
> 古い条件でも `n` を小さくすると `K=5` の選択は 5/5 から 1/5 に落ちました。
> 反復数を 8 から 40 に増やしても、選択は一貫しては改善しませんでした。
> **どちらも「n が効く」「反復数だけではない」という有限標本の観察であって、一致性の証明ではありません。**

### C. 先生に聞くべき質問（5 つ）

これらは自分で調べれば答えが出る種類の質問ではなく、**研究方針の判断**を仰ぐものである。

**Q1. 修論としてどこまで理論を求めますか。**
現状、識別可能性は Poisson-X と Gaussian-Y で証明できていますが、
**実験で使っている Bernoulli-Y は未解決**です。
(a) Bernoulli-Y の識別可能性を証明することを修論の主目標に据えるか、
(b) 証明済みの family（Gaussian-Y）へ実験を移して理論と実験を揃えるか、
(c) 識別可能性は未解決と明記したうえで有限標本の挙動整理を主結果とするか。
**この 3 つで修論の形が大きく変わります。**

**Q2. どの「K」を研究対象にしますか。**
`K*`（観測分布の最小次元）は M-closed を仮定しないと定義できず、実データでは一般に存在しません。
`K_PRED`（予測最良次元）なら誤指定下でも定義できますが、`K*` と一致する保証がありません。
**推定対象を `K*` に置くか `K_PRED` に置くかで、必要な理論も評価指標も変わります。**

**Q3. 次の切り分けをどこに置きますか。**
古い条件と現在の条件は `n=75` で最も食い違いました（4/5 対 0/8）。
違いは ①signal 強度（`w=1.5` 固定 vs `w=sqrt(3/K)`）②generator（Z 標準化・F 行正規化・η clipping の有無）
③criterion（Q ベース vs held-out S1）④Y の hold-out（全ペア学習 vs 20% hold-out）の 4 つです。
**どれを先に分離すべきでしょうか。** 私は ① と ④ が最も直接効くと見ていますが、これは推測です。

**Q4. 基準をどう位置づけますか。**
現行の Q ベース基準は Schwarz BIC ではなく、このモデルは特異なので標準的な BIC 理論が適用できません。
(a) 特異モデルの理論（RLCT / sBIC / WBIC）へ踏み込むか、
(b) 予測基準（held-out）に一本化して「BIC の代替」として位置づけるか、
(c) 既存基準の限界を明示するにとどめるか。
**修論の分量と残り時間を踏まえてご判断いただきたいです。**

**Q5. `n → ∞` をどこまで議論すべきですか。**
一致性を論じるには漸近レジーム（Y が dense か sparse か、`d` を固定するか）と
有効標本数の定義（`n` か `n(n−1)/2` か）をまず決める必要があり、
潜在変数 `Z` が `n` とともに増える incidental parameter 問題も残ります。
**修論では「レジームを固定して問題を定式化するところまで」で十分でしょうか、
それとも一致性の証明まで求められますか。**

---

## 改訂履歴

| 日付 | 内容 |
|---|---|
| 2026-09-06 | 初版。新規実験なし。既存の一次資料（`true_k_identifiability_hardened_20260904.md`、`k_selection_theory_map_20260905.md`、`paper_bic_reproduction_alignment_20260904.md`、`RESEARCH_MASTER.md` §17-18、`KNOWN_ISSUES.md`）を先生の 6 指摘の形に再構成し、2026-09-06 の EM continuation 実験と historical bridge 実験を有限標本の観察として追加した |
