# K 選択理論マップ — 何が証明済みで何が未解決か

**作成日:** 2026-09-05
**目的:** 修論で K 選択を論じるために必要な概念を区別し、
**それぞれが本研究で「証明済み」なのか「未解決」なのかを一枚に固定する。**
**新しい定理を無理に作らない。** 未解決は未解決と書く。

一次資料: `true_k_identifiability_hardened_20260904.md`（敵対レビュー済み）、
`paper_bic_reproduction_alignment_20260904.md`、`clean_true_k_results_20260905.md`、
`KNOWN_ISSUES.md` KI-010・KI-019・KI-020・KI-021。

---

## 1. まず区別する — 3 つの「尤度」

**混同がこの研究で最も事故を起こしてきた場所である。**

| 記号 | 式 | `p(Z)` | `Z` を積分 | 呼び名 |
|---|---|:---:|:---:|---|
| **Q1** | `ln p(X, Y \| Z, θ)` | なし | しない | conditional / plug-in likelihood |
| **Q2** | `ln p(Z, X, Y \| θ)` | **あり** | しない | complete joint log density |
| **Q3** | `ln p(X, Y \| θ) = ln ∫ p(Z,X,Y\|θ)dZ` | あり | **する** | **observed-data marginal likelihood** |

**Schwarz BIC が近似するのは Q3 だけである。**

### 本研究に登場する基準の対応

| 基準 | 当てはまり項 | 位置づけ | 状態 |
|---|---|---|---|
| **原論文 Eq.(26)** | Eq.(16) の `ln L` = **Q1** | 論文上は `BIC` と定義。**評価手続き（どの `Z` か、MC 平均か）は本文から特定不能** | `[UNRESOLVED]` |
| MATLAB `calcdescmetric_ver4.m` | `ln p(X\|Z)+ln p(Y\|Z)+ln p(Z)` の平均 = **Q2 の MC 平均** | Q 関数を計算していることは確認済み。paper Exp.2 に実際に代入されたかは未確定 | `[UNRESOLVED]` |
| 現行 `calc_bic_dual` | `Q_strict` = **Q2 の MC 平均** | **Q3 ではない。「Schwarz BIC」と呼ばない** | `[CONFIRMED_IN_REPOSITORY]` |
| 本研究 **S2** | 同上（`−2 Q_strict + p log n`） | **Q ベース基準** | 同上 |
| 本研究 **S3** | **Q1**（`Ẑ` を代入、`p(Z)` なし、積分なし） | **本研究が定義した基準。原論文の基準ではない** | 定義は明確 |
| 本研究 **S1** | held-out 予測スコア（尤度ではない） | plug-in raw-eta mean log score | population target は `[UNRESOLVED]`（U10） |

**含意:** Q1 と Q2 は `Z` を積分していないので、**罰則を `p log n` にしても Schwarz の議論は成立しない。**
実験でも **S3（Q1 型）は 64 中 3 しか当たらず、ほぼ全セルで候補上限 `K=7` を選んだ。**
潜在次元を増やすほど代入した `Ẑ` への当てはまりが良くなり、罰則が追いつかないためである。

---

## 2. 「K」に 4 つの候補がある

| 記号 | 定義 | 本研究で確立したか |
|---|---|---|
| `K_TRUE` | generator が `Z` を何列作ったかという**手続き上の数** | 定義のみ。**推定対象ではない** |
| **`K*`** | `min{K : P0 ∈ M_K}`（観測分布を表現できる最小潜在次元） | **定義を確定**（§2）。M-closed 前提 |
| `K^rank` | population Gram `FF^T` の階数（**X 周辺のみを見る**） | **P1・P5 が確立するのはこちら** |
| `K°` | KL 射影次元 `min argmin_K inf_{θ∈Θ_K} KL(P0‖P_θ)` | **未確立（U11）。** 誤指定下で唯一 well-defined になりうる |

**関係:** `K^rank ≤ K* ≤ K_TRUE`。**どの不等号も等号とは限らない。**

**実験の「真値一致」は `K_TRUE` との一致である。** `K*` とも `K^rank` とも違う。

---

## 3. identifiability の 6 つの意味

| 概念 | 主張の形 | 本研究の状態 |
|---|---|---|
| **parameter identifiability（`O(K)` を法として）** | `P_θ1 = P_θ2 ⟹ θ2 = θ1·Q, Q ∈ O(K)` | `O(K)` を法とせずには**本モデルでは決して成立しない**（§4） |
| **order identifiability** | `P0 ∈ M_K ∩ M_{K'} ⟹ K = K'` | **Gaussian-Y `w≠0` で証明（P3）。** 他 family は未解決（U5） |
| **functional identifiability** | 特定の関数 `g(θ)` が決まる | **P1（`g = FF^T`）・P2（`g = (w0,w²,K,σ_y²)`）で証明** |
| **local / global** | 近傍で一意か大域で一意か | Ledermann 型条件は **generic 点での local** 識別の必要条件 |
| **generic / everywhere** | ほとんど至るところか全域か | P1 の `rank(F)=K` は **generic**。退化集合（§5）は空でない |
| **estimability** | 有限標本から推定できるか | **identifiability とは別。** 推定 Gram は PSD ですらない（U7） |

---

## 4. family 別の到達点

| family | 何が言えるか | 状態 |
|---|---|---|
| **Poisson-X** | `‖f_l‖²=2 log E[X_l]`、`f_l·f_m=log(E[X_lX_m]/(E[X_l]E[X_m]))` で `FF^T` 復元。**X 周辺の最小次元 = `rank(FF^T)`** | **証明済み（P1）。** unclipped link・`rank(F)=K`（generic）・population が必要 |
| **Gaussian-X（Σ 既知）** | `FF^T = Cov(X) − Σ` から階数 | **証明済み（P5）** |
| **Gaussian-X（Σ 未知対角）** | 因子分析の識別問題。必要条件 `(d−K)² ≥ d+K` | **未解決（U3）。** `d=2,K=1` は必要条件を満たさない |
| **Bernoulli-X** | `E[X_l]=1/2` で一次モーメントは無情報。`d=1` は **X 周辺の反例** | **`d>1` は未解決（U1）** |
| **Gaussian-Y** | `M_S(t)=(1−t²)^{−K/2}` → `κ_2=K, κ_4=6K, κ_6=120K` → **単一 dyad で `(w0,w²,K,σ_y²)`**。`w` の符号は**三角形**から | **証明済み（P2・P8）** |
| **Bernoulli-Y** | `w0=0` では edge density が無情報。K の情報は共有ノード motif に入る | **未解決（U2）。実験で使っている family** |
| **Poisson-Y** | `E[Y^r]<∞ ⟺ \|w\|<1/r`。既定 `w=0.5` は分散発散の境界 | **モーメント条件は証明済み（P6）。識別可能性は未解決（U4）** |

---

## 5. model nesting

| 主張 | 状態 |
|---|---|
| X 側のみの部分族は入れ子（`F'=[F,0]`） | **証明済み** |
| Gaussian-Y `w≠0` で `{P ∈ M_K : w≠0} ∩ M_{K+1} = ∅` | **証明済み（P3）** |
| `M_K` と `M_{K+1}` が互いに素 | **偽。** `w=0` 切片で交わる |
| Bernoulli-Y / Poisson-Y の非入れ子性 | **未解決（U5）** |
| `M_K` が閉集合か | **未解決（U11）。** `P0 ∉ M_{K+1}` は `P0 ∉ closure(M_{K+1})` を含意しない |

**含意:** 非入れ子が無効にするのは **尤度比検定の χ² 近似と Wilks の定理**である。

---

## 6. BIC consistency — なぜ言えないのか

**よくある誤り:** 「モデル族が入れ子でないから BIC が使えない」。
**これは誤りである。** Schwarz BIC の導出はモデルごとの Laplace 近似であり、
非入れ子モデルの比較に BIC を使うのは標準的である。

**BIC がこのモデルで正当化されない実際の理由は 3 つ。**

| # | 理由 | 詳細 |
|---|---|---|
| 1 | **モデルが特異（singular）** | `O(K)` 回転で観測分布が不変 → Fisher 情報が `K(K−1)/2` 次元の軌道方向に退化。加えて `rank(F)<K`・`w=0` の縮退集合上でも退化。特異モデルでは `(p/2)log n` が **RLCT** に置き換わるが、**本モデルの RLCT は未知**（`RESEARCH_MASTER.md` §14 U5） |
| 2 | **境界パラメータ** | `Σ_X ⪰ 0`、`σ_y² ≥ 0`、因子分析の `Ψ ⪰ 0`。K 比較はしばしば境界上で起きる |
| 3 | **有効標本数が未定義** | ノード数 `n` / dyad 数 `n(n−1)/2` / X 要素数 `nd` の 3 通り。`calc_bic_dual` は `log n` に**ノード数**を使う。さらに `Z ∈ R^{n×K}` は `n` とともに増える **incidental parameter** で、`Q_strict` は `Z` を完全データ扱いする一方、罰則 `p̂` は `Z` を数えない |

**追加の実装事実**（`expfam/src/utils_expfam.py`、`[CONFIRMED_IN_REPOSITORY]`）:
`num_params = kd − k(k−1)/2 + [d if X gaussian] + [1 if Y gaussian]`。
`−k(k−1)/2` は `O(K)` 軌道の次元を既に引いている。
**`w0` と `w` は数えられていない**（原論文の慣行）。
Gaussian-Y では `K` の情報をもっぱら `w` が運ぶことが判明したので検討に値するが、
**`K` に依存しない定数なので順位には影響しない。「誤り」とは断定しない。**

---

## 7. held-out predictive selection

| 論点 | 状態 |
|---|---|
| S1 の定義 | `y·η − logaddexp(0,η)` を held-out dyad 上で平均、2 start 非加重平均、tie 1e-12、最小 K |
| **S1 の population target** | **未解決（U10）** |
| なぜ未解決か | 厳密に proper な scoring rule なら M-closed 下で `K*` に一致しうる。しかし S1 は**推定 `ẑ` を代入した plug-in** で proper scoring rule ではない |
| 有限標本 | bias–variance trade-off が別に効く |
| **確実に言えること** | **選択（selection）≠ 識別（identification）。** 「held-out で `K_TRUE` が選ばれた」は「`K_TRUE` が識別された」を意味しない |

**さらに:** データに K を区別する情報がない場合でも、
**smallest-K tie rule により手続きは `K=1` を出力する。**
それが `K_TRUE=1` と一致しても、それはデータからの識別ではない。
（実験の `K_TRUE=1` が全 `n` で 4/4 なのは、この下限効果と交絡している。）

---

## 8. regular vs singular latent-variable model

| 性質 | 本モデル |
|---|---|
| パラメータ写像の単射性 | **なし**（`O(K)` 不変性、§4） |
| Fisher 情報の正則性 | **なし**（軌道方向に退化。退化集合上でさらに退化） |
| 局所漸近正規性 | **成立しない** |
| 標準的な BIC / AIC / LRT 漸近論 | **そのままでは適用不可** |
| RLCT | **未知** |
| 潜在変数の個数 | `nK`、**`n` とともに増える**（incidental parameter） |
| 観測の独立性 | **dyad は独立でない**（`z_i` を `n−1` 本の dyad が共有） |

---

## 9. 一枚まとめ

### 証明済み

- `K*` の定義（M-closed 前提、入れ子性不要）
- Poisson-X: `FF^T` の population 復元、**X 周辺**の最小次元 = `rank(FF^T)`（P1）
- Gaussian-X（Σ 既知）: 階数から `K`（P5）
- Gaussian-Y: 単一 dyad から `(w0,w²,K,σ_y²)`（P2）、三角形から `w` の符号（P8）
- Gaussian-Y: `{P ∈ M_K : w≠0} ∩ M_{K+1} = ∅`（P3）
- Poisson-Y: `E[Y^r]<∞ ⟺ \|w\|<1/r`（P6）
- 反例: Bernoulli-X `d=1`（**X 周辺のみ**）、`w=0`、Gaussian-X `d=2,K=1`（counting）、Bernoulli-Y `w0=0` の edge density
- 現行 `calc_bic_dual` は Q3 を使っていない

### 未解決

| ID | 内容 |
|---|---|
| U1 | Bernoulli-X（`d>1`）の識別可能性 |
| **U2** | **Bernoulli-Y の識別可能性 — 実験で使っている family** |
| U3 | Gaussian-X（Σ 未知）の十分条件 |
| U4 | Poisson-Y の識別可能性 |
| U5 | Bernoulli-Y / Poisson-Y の非入れ子性 |
| **U6** | **どの基準についても `n→∞` の一致性** |
| U7 | 有限標本の rank 閾値（推定 Gram は PSD ですらない） |
| U9 | clean construction で `K* = K_TRUE` か |
| U10 | held-out plug-in score の population target |
| U11 | `M_K` の閉性、誤指定下の pseudo-true `K` |
| U12 | `{K : P0 ∈ M_K}` の連結性 |
| — | 本モデルの RLCT |
| — | 有効標本数の定義 |

### 絶対に書いてはいけない

- 「BIC は理論的に正しい」／「BIC は非入れ子だから使えない」（**理由が誤り**）
- 「held-out なら真の K を選べる」／「held-out は `K*` を推定している（していない）」
- 「Poisson-X なら常に K が識別可能」
- 「Bernoulli では K は識別できない」
- 「K 選択の一致性を証明した」／「`n` を増やせば `K_TRUE` に収束する」
- 「実データで `K*` を推定した」
- 「S3 の失敗は原論文 BIC の失敗である」
