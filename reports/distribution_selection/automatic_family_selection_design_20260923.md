# 属性分布族の自動選択 — 連続潜在 MCEM 上での定式化設計

- 日付: 2026-09-23
- Issue: #73（parent #71 / Phase 9B、Phase 9 の主研究）
- branch: `design/73-automatic-family-selection`
- baseline: `origin/main = f5a0e8f17a50b7b41371fb2495facd960091978a`
- scope: **THEORY / DESIGN ONLY**。model implementation 0 行、EM fit 0 件、synthetic run 0 件、artifact 生成 0 件。

**この文書は設計案であり、正式採用はしていない。** 方式の採用・実験条件の freeze は Human Gate。

---

## 0. この設計が解こうとしている問題

現在 `family_x`（全列共通）または `family_x_list`（per-column prototype）は**人間が指定**する。
これを、観測 `X, Y` から列ごとに自動で決める形へ拡張したい。

本文書の結論を先に書く。

1. **列ごとの family 比較は、固定した posterior `q_t(Z)` の下で well-posed に定義できる**（§2）。
   `Q_Z` と `Q_Y` は `c_l` に依存しないので比較から厳密に消える。
2. **ただし比較できるのは「同じ基底測度の上にある候補どうし」だけである**（§3）。
   count データを Gaussian として扱うかどうかは **family の選択ではなく representation の選択**であり、
   log-likelihood の大小では決められない。これは KI-018 の未解明問題と同じ構造である。
3. **現行モデルには X 列の intercept がない**ため、family 比較が「分布形の適合」ではなく
   「位置（平均水準）の適合」に強く汚染される（§5）。
   人工データ（モデルから生成）では問題にならないが、実データでは支配的な交絡になりうる。
4. 方式は **C（hybrid）を推奨候補**とする（§7）。採用は Human Gate。
5. #74 は **K 固定・matched generator・support gate 内の同一測度比較**に限定した
   **feasibility pilot** とする（§8.2）。現行 3 family ＋厳密 gate では score が非自明に働くのは
   **0/1 列の Bernoulli vs Poisson だけ**であり、自動分布選択一般の実証にはならない。
   実行前に **HG-1（方式）と HG-3（penalty 方針）を Human が freeze する必要がある**（§12）。
6. `Σ_l log|M_l|` は `M_l` 固定なら assignment に関して定数であり、
   **selector penalty として機能しない**（§6.4.1）。離散 search cost は未解決のまま残る。

---

## 1. 現在のモデルと MCEM（守るべき前提）

```
z_i  ~ N(0, I_K)                                （連続潜在ベクトル。latent class ではない）
y_ij ~ ExpFam_Y( η_ij^Y = w_0^Y + w^Y z_i^T z_j )   i < j
x_il ~ ExpFam_X( η_il^X = f_l^T z_i )              バイアスなし
```

- `w_0^Y, w^Y` はスカラー。
- X の尤度は列 `l` ごとに因子分解する（root `CLAUDE.md` §1）。
- **`z_i ∈ {1,…,K}` の latent class model ではない。** したがって latent-class 用の
  Structural EM をそのまま移植しない。以下はすべて現行 MCEM の posterior sample
  `Z^(s), s = 1,…,L` の上で再導出する。
- E-step は Newton 系サンプラで `Z^(s)` を得て `scale_Z` で正規化する
  （`utils_expfam.py` L.496–516）。`var_z = 1` 固定。

本設計は上記生成モデルを**変更しない**。追加するのは列ごとの family 指示変数だけである。

---

## 2. family 変数 `c_l` と MCEM 上の candidate score

### 2.1 記法

列 `l ∈ {1,…,d}` に対し family 変数

```
c_l ∈ M_l   （M_l は列 l の候補集合。§3 の support gate が決める）
```

を導入し、

```
x_il | z_i, c_l = m  ~  ExpFam_m( η_il = f_l^T z_i ;  ψ_lm )
```

とする。`ψ_lm` は **family m の下での列 l のパラメータ**で、現行実装では

```
ψ_lm = { f_l^(m) ∈ R^K }                       m ∈ {bernoulli, poisson}
ψ_lm = { f_l^(m) ∈ R^K , σ_l^2 > 0 }           m = gaussian
```

**`f_l` は family 非依存の共通量ではない。** link が違えば最適な `f_l` も違うので、
候補 family ごとに `f_l^(m)` を別々に最適化する必要がある。これは後の計算量議論（§7）に効く。

### 2.2 complete-data 対数尤度の列分解

現行モデルでは、`Z` を与えたとき complete-data 対数尤度が

```
log p(Z, X, Y | θ, c)
  = log p(Z)                                   … c にも F にも依存しない
  + Σ_{l=1}^{d} Σ_{i=1}^{n} log p_{c_l}(x_il | f_l^T z_i ; ψ_l)   … 列ごとに分離
  + Σ_{i<j} log p_Y(y_ij | w_0^Y + w^Y z_i^T z_j)                 … c に依存しない
```

と厳密に分かれる。これが本設計の土台である。

### 2.3 candidate score

反復 `t` の posterior sample `Z^(s) ~ q_t(Z)`, `s = 1,…,L` を固定して

```
Q_l(m, ψ_lm | q_t)  =  (1/L) Σ_{s=1}^{L} Σ_{i=1}^{n} log p_m( x_il | f_l^{(m)T} z_i^{(s)} ; ψ_lm )
```

と定義する。§2.2 より、`q_t` を固定した M-step の目的関数は

```
Q(θ, c | q_t) = Q_Z(q_t) + Σ_l Q_l(c_l, ψ_l | q_t) + Q_Y(w_0, w | q_t)
```

であり、**`Q_Z` と `Q_Y` は `c` に依存しない**。したがって列 `l` における family 比較では
両者が厳密に打ち消え、

```
Δ_l(m, m') = Q_l(m, ψ̂_lm | q_t) − Q_l(m', ψ̂_lm' | q_t)          （＋ §6 の penalty 差）
```

だけが問題になる。**これが「continuous-Z MCEM 上で family 比較が well-posed である」という主張の内容**である。
ここまでは代数的事実であり、追加仮定を必要としない。

### 2.4 monotonicity について（限定つき）

`q_t` を固定したまま `(c, θ)` について penalized-Q

```
G(θ, c | q_t) = −2 Q(θ, c | q_t) + Pen(c)
```

を減少させる更新は、**同じ `q_t` の下では**必ず `G` を改善する（M-step の定義そのもの）。
しかし

- `q_t` 自体が `c^{(t)}` と `θ^{(t)}` に依存する（E-step の precision
  `A_i = I_K + F^T V_X(m_i) F + (w^Y)^2 Σ_j V_Y z_j z_j^T` は全列の family に依存する）、
- `Z^(s)` は MC sample であり `Q` は確率的近似である、
- 現行 `Q_strict` は観測データ周辺尤度ではない（#72 / KI-010）、

ため、**「EM 全体として observed-data 的な意味で単調に改善する」とは書かない。**
書いてよいのは「固定した `q_t` の下での coordinate ascent（ECM 型）である」までである。
これは UNRESOLVED（§10 U3）。

---

## 3. support gate — 何と何を比較してよいか

### 3.1 二段構えにする

**「観測型（データ型）まで完全自動」と「観測型を与えた上で family 自動選択」を分ける。**

| 段 | 内容 | 本設計での扱い |
|---|---|---|
| Level 0 | 列 `l` の観測がどの基底測度に乗るか（counting measure か Lebesgue か）を決める | **決定的な support gate のみ**。尤度比較を使わない |
| Level 1 | 同一の基底測度の上で family を選ぶ | **candidate score で自動選択**（§2.3） |

Level 0 を尤度比較でやろうとしてはいけない。理由は §3.3。

### 3.2 決定的 support gate（Level 0）

観測列 `x_{·l}` から機械的に決まる admissible 集合:

| 観測 | 厳密に admissible な family | 備考 |
|---|---|---|
| すべて `{0,1}` | `{bernoulli, poisson}` | 0/1 は Poisson の support にも含まれる。**同一測度なので比較は正当** |
| すべて非負整数（1 より大を含む） | `{poisson}`（将来 `{poisson, nb}`） | counting measure 上 |
| 非整数 or 負値を含む | `{gaussian}` | Lebesgue 上 |

**現行の実装済み 3 family だけでは、厳密な gate をかけた時点で
Level 1 に真の選択肢が残るのは「0/1 列の Bernoulli vs Poisson」だけになる。**
これは本設計の重要な帰結であり、隠さずに書く。意味のある Level 1 の選択肢を増やすには

- counting measure 上: Poisson vs **Negative Binomial**（`experimental/model_dual_expfam_nb.py` が存在。ただし Y 側 NB であり X 側 NB は未実装）
- Lebesgue 上: Gaussian vs（未実装の裾の重い連続 family）

のいずれかが必要になる。**これは新規実装を伴うので Human Gate**（§11 HG-4）。

### 3.3 なぜ continuous density と discrete PMF を横並びにできないか

- discrete PMF は counting measure に関する density で、値は確率そのもの（`≤ 1`、`log ≤ 0`）。
- continuous density は Lebesgue measure に関する density で、`x` の測定単位に依存する。
  単位を `x → a·x` と取り替えると Gaussian の log-density は列全体で `−n log a` だけ動くが、
  Poisson の log-PMF は何も変わらない。
  **つまり「Gaussian の score が Poisson より大きい」は単位の取り方で反転させられる。**
- したがって `Σ_i log N(x_il ; …)` と `Σ_i log Pois(x_il ; …)` の大小比較は
  **model selection statement として意味を持たない**。

**帰結:** 「count 列を raw のまま Poisson で扱うか、log 変換して Gaussian で扱うか」は
**family 選択ではなく representation 選択**である。これは尤度でも本 criterion でも決められない。
判定するなら共通スケール上の held-out 予測スコア（あるいは人間の判断）による。

これは KI-018 の構造とそのまま一致する。Issue #33 の
`mixed_train_raw_poisson − mixed_train_log` は **representation の比較**であり、
`−0.100274` / **29/30 splits で悪化**という結果は held-out 予測スコアで得られている。
**「intercept 欠如が原因」「curvature が原因」とは書かない**（KI-018）。
本設計もこの representation 問題を解決しない。

---

## 4. family comparison に必要な「完全な」log probability / density

parameter optimization では定数として落としてよい項でも、**family comparison では落とせない**。
落としてよいのは「比較する全候補で共通の定数」だけであり、base measure `h(x)` は候補ごとに違う。

| family | 比較に必要な完全式 | 現行実装の状態 |
|---|---|---|
| Bernoulli | `x·η − log(1 + e^η)`（`h(x) = 1` なので追加項なし） | **完全**。`model_dual_expfam.py` L.324–327 / `objective_consistent_numerics.bernoulli_log_likelihood` |
| Poisson | `x·η − e^η` **− log(x!)** | `calc_log_likelihood_X` は `−log(x!)` を**含まない**。`calc_Q_dual_strict`（L.375–376）および `eval_utils.calc_Q_dual_strict_exp`（L.217–223、per-column 版あり）が後段で加える |
| Gaussian | `−½(x−η)²/σ_l² − ½ log σ_l² − ½ log 2π` | **完全**（`model_dual_expfam.py` L.321–323 は `−½ log 2π` を含む。ただし docstring L.303 は「省く」と書いており実装と不一致） |
| （将来）NB | `log Γ(x+r) − log Γ(r) − log(x!) + …` | X 側 NB 未実装 |

**設計上の必須ルール:**

1. family 比較スコアには **必ず strict 版**（`calc_Q_dual_strict_exp` 系）を使う。
   `calc_Q_dual` / `calc_log_likelihood_X` を直接比較に使わない。
2. `−log(x!)` は `ψ` に依存しない定数だが、**Poisson と Bernoulli を比べるときに残る**。
   0/1 列では `x! = 1` なので `−log(x!) = 0` となり、この 1 ケースに限っては偶然消える。
   1 より大きい値を含む列では消えない。**「定数だから落とす」という理由付けをしない。**
3. Gaussian の `−½ log 2π` も同様。列内で定数だが family 間では残る。
4. Y 側 Gaussian の `−½ log 2π` は標準系列で欠落している（#72 report（PR #76）§5-1）。
   本設計は X 側 family 選択なので `Q_Y` は比較で消えるが、
   **`C_Q` の絶対値を系列横断で並べない**（KI-002）。

### 4.1 numerical clipping（KI-015）

`experimental/model_dual_expfam_percolumn.py` の legacy numerics は
Poisson の `η` を `[−20, 10]` に clip し、Bernoulli の確率を floor する。
KI-015 の一次確認により、**clip 区間の外側では報告される score が目的関数と一致しない**
（決定論的反例: `eta=11.5, x=3` で実装 score `−22023.465794806718`、有限差分は `0.0`）。

family 比較は「複数候補の score の大小」を直接使うため、
**clip された領域に落ちた候補の score は比較に使えない。**

したがって本設計は **`numerics_mode = "consistent"` 系列
（`experimental/model_dual_expfam_consistent.py`、lineage E）を前提とする。**
`objective_consistent_numerics` の `bernoulli_log_likelihood` / `poisson_log_likelihood` は
clip せず、非有限値を例外にする。legacy numerics での family 比較は設計に含めない。

なお lineage E は **experimental prototype であり本文採用不可**（root `CLAUDE.md` §3）。

---

## 5. X 列 intercept の欠如 — family 比較への構造的交絡

現行モデルは `η_il = f_l^T z_i` でバイアス項を持たない。`z_i ~ N(0, I_K)` なので
`η_il` の（モデル上の）中心は 0 である。すると各 family の「既定の水準」は

| family | `η = 0` での平均 |
|---|---|
| Gaussian | `0` |
| Bernoulli | `0.5` |
| Poisson | `1` |

に固定される。列 `l` の経験平均がこの水準から離れていると、
`f_l` は**水準合わせのために使われる**（`‖f_l‖` が大きくなり、同時に `η` の分散も増える）。

**帰結:** family 比較スコアの差 `Δ_l(m, m')` は「分布形の適合」だけでなく
「その family の既定水準に列の平均がどれだけ近いか」を強く含む。
平均 10 の count 列は Poisson でも `η ≈ 2.3` を `f_l^T z_i` だけで作る必要があり、
分布形が正しくてもスコアが落ちる。

**限定（KI-018 を踏まえて誠実に書く）:**
Issue #28 §9.3 の通り、intercept 欠如 / raw scale / Poisson 曲率 / X 側過分散 の 4 要因は
既存条件では交絡しており分離されていない。したがって
**「intercept 欠如が family 誤選択の原因である」とは書かない。**
ここで書けるのは「intercept がないので、family 比較は位置適合と交絡した量になる」という
**モデル構造から直接従う事実**までである。

**設計上の対応（2 択。採用は Human Gate）:**

- **(a) 最小 pilot を matched generator に限定する**（推奨）。データを現行モデル
  （intercept なし、`η = f_l^T z_i`）から literal に生成すれば、
  **intercept omission による model misspecification は生じない**（真値がモデル内にある）。
  #74 の最小 pilot はこれで足りる。
  **ただし「交絡が消える」のではない。** family score が位置適合を含むこと自体は
  matched generator でも残り、列の経験平均が family の既定水準から離れているほど
  誤選択しやすいという性質はそのまま効く。§8.5 でこれを観測対象にする。
- **(b) 列ごとの intercept `b_l` を導入する**（`η_il = b_l + f_l^T z_i`）。
  これは **生成モデルの変更**であり root `CLAUDE.md` §6 の Human Gate。
  `θ` と `p_K`（`+d`）も変わり、過去の全結果と非互換になる。**本 Issue では設計しない。**

---

## 6. family-specific parameter count と penalty

### 6.1 各 family の追加パラメータ

| family | 列あたり追加パラメータ | 個数 |
|---|---|---|
| Gaussian | `σ_l²`（dispersion） | 1 |
| Bernoulli | なし | 0 |
| Poisson | なし | 0 |
| （将来）NB | `r_l` | 1 |

`f_l ∈ R^K` は **どの family でも共通に K 個**であり、全体では `F` ブロックとして
`k·d − k(k−1)/2` に含まれる（回転制約込み）。**family 選択によって `F` ブロックの数え方は変わらない。**

### 6.2 現行 criterion への接続（#72 の結果を使う）

#72 の確認より

```
C_Q(K, c) = D_{K,c} + P_Z(K) + P_θ(K, c)
P_Z(K)    = n·K·(1 + ln 2π)        … family に依存しない（決定論的・K に線形）
P_θ(K, c) = p_{K,c} · ln n
p_{K,c}   = k·d − k(k−1)/2  +  Σ_l 1{c_l = gaussian}  +  1{family_y = gaussian}
```

- **`d·1{family_x = gaussian}` を `Σ_l 1{c_l = gaussian}` に置き換えるのが唯一の変更点**である。
- これは `experimental/eval_utils.calc_bic_exp(family_x='mixed', n_gaussian_x_cols=…)`（L.232–256）に
  **既に実装されている**。新しい criterion を作る必要はない。
- 対応する Q は `eval_utils.calc_Q_dual_strict_exp`（L.186–229）が per-column Poisson 階乗補正を持つ。
- `utils_expfam.calc_bic_dual` / `calc_Q_dual_strict` は `mixed` を知らないので
  **使用不可**（`model_dual_expfam_percolumn.py` L.22–26 に明記）。

### 6.3 二重計上について（**限定つき**）

**現在 `p_{K,c}` が数えている連続パラメータブロックの間では、重複計上は起きない。** 理由:

1. `P_Z(K)` は `c` に依存しない（`Q_Z` は `F` にも `c` にも依存しない）。
2. `P_θ` の family 依存部分 `Σ_l 1{c_l = gaussian}` は `K` に依存しない。
3. `F` ブロック `k·d − k(k−1)/2` は `K` と `d` だけの関数で、`c` に依存しない。

つまり `P_θ(K, c) = [K の関数] + [c の関数] + const` と加法的に分離しており、
**連続パラメータの自由度**を二度数えることはない。

**この主張はここまでである。** `c` を data から選んだこと自体の
**離散的な search cost（model multiplicity）は `p_{K,c}` にまったく入っていない**（§6.4）。
したがって「K penalty と family penalty の二重計上は起きない」を、
「選択手続き全体として罰則が適切である」という意味に読まないこと。
離散 search cost は **UNRESOLVED のまま残る**。

### 6.4 **数えられていない離散自由度**（重要・UNRESOLVED）

`p_{K,c}` は **`c` を data から選んだこと自体のコストを含んでいない。**
`Π_l |M_l|` 通りの assignment から 1 つを選んでおきながら離散的な罰則が 0 である。

#### 6.4.1 `Σ_l log|M_l|` は selector penalty として機能しない（訂正）

素朴な候補として「MDL 的な選択コスト `Σ_l log|M_l|` を `C_Q` に加える」が考えられるが、
**これは assignment `c` の選択結果を変えない。**
各列の候補集合 `M_l` が `c` に依存せず固定である限り `Σ_l log|M_l|` は `c` に関する定数であり、

```
argmin_c [ C_Q(K, c) + Σ_l log|M_l| ]  =  argmin_c C_Q(K, c)
```

が厳密に成立する。したがってこの項は
「より柔軟な family への偏り」を補正しない。

この項が意味を持ちうるのは、**「family を自動選択する model class 全体」と
「family を事前固定した model class」を比べる model prior / code length** としてであり、
**assignment 間の selector penalty としてではない。** 両者を混同しない。

#### 6.4.2 残る選択肢

離散 search の multiplicity を本当に補正したいなら、次のいずれかを**別途設計**する必要がある。

- (i) 明示的な model prior（`c` に依存する項）を置いた extended criterion
- (ii) family 選択を held-out データで行い、`C_Q` は selection に使わない
- (iii) 補正を置かず、raw criterion で選択し bias の可能性を limitation として明記する

いずれも**理論決定であり Human Gate**（§11 HG-3）。本設計では決めない。

**#74 の feasibility pilot では新しい penalty を発明しない。** (iii) の raw criterion で選択し、
選択が僅差だったか大差だったかを **score margin** として記録する（§8.4）。
margin の分布が分かってから補正の要否を議論するのが順序として正しい。

---

## 7. 3 方式の比較

記号: `d` 列、候補数 `|M|`（列あたり）、EM 反復 `T`、MC sample `L`、full fit 1 回のコストを `C_fit`。

### 方式 A — EM 反復内部で family を更新

各反復で E-step 後に、列ごとに全候補の `ψ_lm` を最適化して penalized score で `c_l` を更新する。

```
for t in 1..T:
    Z^(1..L) ← E-step(θ^(t), c^(t))
    for l in 1..d:
        for m in M_l:
            ψ̂_lm ← argmax_ψ Q_l(m, ψ | q_t)
        c_l^(t+1) ← argmin_m [ −2 Q_l(m, ψ̂_lm | q_t) + Pen_l(m) ]
    θ^(t+1) ← 残りの M-step（w_0, w, σ_Y）
```

### 方式 B — 候補 family ごとに outer-loop で fit して比較

family 配置 `c` を固定した完全な MCEM を複数回走らせ、最終 `C_Q` で比較する。

### 方式 C — hybrid

A を inner engine として candidate assignment `ĉ` を得たのち、
**`ĉ` を固定して標準の MCEM をもう一度最初から走らせ、その fit の `C_Q` を報告値とする。**
`ĉ` が変わらなくなるまで繰り返す。**何周で安定するかは未実験であり、本文書は周回数を見積もらない。**
反復上限と停止規則は #74 で観測してから決める。

### 比較表

| 軸 | A（EM 内更新） | B（outer-loop 比較） | C（hybrid） |
|---|---|---|---|
| current MCEM との整合性 | 高。`q_t` 固定下で `Q_Z`, `Q_Y` が厳密に消え、M-step の列分解をそのまま使う | 高。各 fit は現行 `run_em_dual` 契約そのまま | 高。最終 fit が現行契約の通常 fit になる |
| 数理の明確さ | 中。coordinate ascent（ECM 型）としては明確だが、`q_t` が `c` に依存するため大域的性質は言えない | **最高**。各候補が独立した通常の推定。比較は最終 criterion のみ | 中〜高。探索段階と報告段階を分離でき、報告値の由来が明確 |
| 実装の難易度 | 中。per-column の `ψ_lm` 最適化ループと選択ロジックが新規。model class 自体は per-column 版で足りる | 低（単純）だが**配置の列挙が必要** | 中。A ＋ 再 fit の配線 |
| 計算量 | `O(T · d · |M| · 列 M-step)`。**full fit は 1 回**。列ごとの最適化は `Z` 固定なので安価 | **`|M|^d` 通りの配置を全探索すると爆発**。`d = 15`, `|M| = 2` で 32768 fit。greedy / coordinate に落とすと実質 C になる | A の探索コスト ＋ 固定 `ĉ` での再 fit。**再 fit 回数は未測定**（§7 方式 C） |
| Monte Carlo noise への強さ | **弱い**。`Δ_l` が小さい列で反復ごとに assignment が振動しうる。`q_t` が incumbent family の下で作られるため path dependence（早期固定）も起きる | **強い**。各候補が独立に収束した後で比較する。MC noise は各 fit 内に閉じる | 中。A の振動は残るが、最終報告値は固定 `ĉ` での通常 fit なので再現性が高い |
| 修士研究としての実現可能性 | 高（計算は軽い）が、振動・path dependence の診断と安定化規則が余分に要る | **`d` が小さい場合のみ**。本 repository の実データ規模の `d` では非現実的 | **高**。既存の実験 harness・registry 規約にそのまま載る |
| provenance 適合性 | 低〜中。報告する `C_Q` が探索経路に条件づいた量になる | 高 | **高**。最終 artifact が「固定 family での通常 fit」なので既存の runinfo / registry 形式で記録できる |

### 推奨候補（**採用しない。Human Gate**）

**C（hybrid）を推奨候補とする。**

理由:

1. B は `|M|^d` の列挙が現実的でなく、greedy 化すると結局 C と同型になる。
2. A 単独だと、報告する `C_Q(K, ĉ)` が「探索経路に条件づいた量」になり、
   本 repository の provenance 規約（結果は承認済み script から、系列を混ぜない）と相性が悪い。
3. C は探索（A）と報告（固定 `ĉ` での通常 fit）を分離するので、
   最終数値が既存の `run_em_dual` / `calc_bic_exp` 経路そのままになり、監査可能性が高い。
4. 計算量は A の探索コスト＋固定 `ĉ` での再 fit であり、修士研究の規模に収まる見込み。
   **ただし必要な再 fit 回数は未測定**なので、総コストは #74 で実測する。

**ただし C は EM 全体の単調性を保証しない**（再 fit で目的関数が下がりうる）。
その場合の扱い（`ĉ` を採るか、`C_Q` が小さい方を採るか）は決めていない。UNRESOLVED（§10 U4）。

---

## 8. 最小 prototype 設計（#74 で実装するもの・**まだ freeze しない**）

### 8.1 何を使い、何を触らないか

| 項目 | 内容 |
|---|---|
| model class | `experimental/model_dual_expfam_consistent.DualExpFamLSMPerColumnConsistent`（lineage E、`numerics_mode = "consistent"`） |
| scoring | `experimental/eval_utils.calc_Q_dual_strict_exp` ＋ `calc_bic_exp(family_x='mixed', n_gaussian_x_cols=…)` |
| 生成器 | `experimental/data_generator_canonical.generate_canonical_data` |
| 新規に書くもの | **family 選択の wrapper script 1 本のみ** |
| **触らないもの** | `model_dual_expfam*.py` 各種、`utils_expfam.py`、`eval_utils.py`、既存 artifact |

**確認済みの gap:** `generate_canonical_data` の `family_x` は**スカラー**であり
per-column 指定に対応していない（`data_generator_canonical.py` L.246–262）。
混在列の人工データを作るには per-column 対応の生成経路が要る。
これは #74 の実装事項であり、**#73 では書かない**。既存生成器を書き換えるのではなく
forward-only の追加とすること（KI-015 の legacy 保存方針と同じ）。

### 8.2 #74 の位置づけ — **feasibility pilot であって、自動分布選択一般の実証ではない**

#74 で調べるのは 1 つだけである。

> **K を真値に固定した状態で、family-selection machinery が真の family assignment を
> 列ごとに回収できるか。**

K 選択は行わない。実データは使わない。held-out も使わない。

#### claim boundary（**これを守らないと過剰主張になる**）

§3.2 の厳密 support gate を適用すると、列は次の 3 種に分かれる。

| 列の観測 | 決まり方 | score が働くか |
|---|---|---|
| 非整数 or 負値を含む | **gate だけで Gaussian に確定** | 働かない |
| 2 以上を含む非負整数 | **gate だけで Poisson に確定**（現行候補集合では） | 働かない |
| 0/1 のみ | `{bernoulli, poisson}` が残る | **ここだけ score が働く** |

したがって **現行の実装済み 3 family ＋ 厳密 support gate の下では、
candidate score による非自明な family 選択が起きるのは 0/1 列の Bernoulli vs Poisson だけ**である。

**#74 で書いてよいこと:**

- 「support gate ＋ candidate score という machinery が、matched generator 上で
  意図どおり動作し、gate 決定列と score 決定列を分離して集計できた」
- 「0/1 列における Bernoulli vs Poisson の score 選択が、どの程度・どの margin で真値を回収したか」

**#74 で書いてはいけないこと:**

- 「属性確率分布の自動選択を実証した」「自動分布選択一般が機能する」
- 「family 自動選択が人手指定より優れる」
- gate だけで決まった列を含めた「family recovery rate」を提案手法の性能として提示すること
- 実データでも同様に動く、という含意

**一般的な自動分布選択を主張したいなら、同一 support 内に追加候補が最低 1 つ必要**である
（例: counting measure 上の Poisson vs 別の count family）。これは新規実装を伴い Human Gate（HG-4）。
**#74 で scope を広げない。**

### 8.3 手順（案。**HG-1 で方式が確定してから確定する**）

以下は推奨候補 C を前提に書いた**案**であり、方式の採用は HG-1 として未了である。
HG-1 の結論によって手順 3–4 は変わる。

1. 既知の `c_true ∈ {gaussian, bernoulli, poisson}^d` で人工データを生成（現行モデルから literal に）。
2. 各列に **決定的 support gate**（§3.2）を適用し `M_l` を得る。
   - gate の出力自体を記録する。Gaussian 列は gate だけで決まるはずで、
     **gate で決まった列と score で決まった列を混同して「回収率」を出さない**。
3. `c` を初期値（例: gate の既定値）から開始し、方式 A で反復更新して `ĉ` を得る。
4. `ĉ` を固定して通常 MCEM を再 fit（方式 C）。`C_Q(K_true, ĉ)` を記録。
5. 比較のため `c_true` 固定 fit と、全列単一 family の fit も記録。

### 8.4 記録する量

| 量 | 目的 |
|---|---|
| 列ごとの `ĉ_l` と `c_true,l` の混同行列 | **主要指標**。「exact recovery rate」は gate 決定列を除いた列だけで計算する |
| 各列・各候補の `Q_l(m)` と penalty 込みの差 `Δ_l`（生値） | 補正の要否を後から議論できるよう raw のまま保存する |
| 反復ごとの `c^(t)` の履歴 | 振動 / path dependence の有無（§7 の A のリスク） |
| **score margin** `min_{m ≠ ĉ_l} [ −2Q_l(m) − (−2Q_l(ĉ_l)) ]`（penalty 込み） | **新しい penalty を発明せず**、選択が僅差か大差かを raw criterion のまま診断する（§6.4.2） |
| seed をまたいだ `ĉ` の一致率 | MC noise への感度 |

### 8.5 予想される失敗モードを**あらかじめ指標にする**

- **0/1 値しか取らない Poisson 列**は support gate 上 Bernoulli としても admissible であり、
  原理的に区別しにくい。これは bug ではなく識別性の問題。
  `c_true = poisson` かつ `max_i x_il ≤ 1` の列は**別カテゴリとして集計する**。
  §8.2 の通り、score が働くのはまさにこのケースなので、**pilot の主戦場はここである**。
- 列の経験平均が family の既定水準（§5）から遠いほど誤選択しやすい。
  これは matched generator でも残る性質である（§5(a)）。
  各列の経験平均を記録し、誤選択との関係を**観測する**（原因と断定しない — KI-018）。

### 8.6 freeze しない項目（Human が決める）

`n` / `d` / `K_true` / `c_true` の構成比 / `f_scale` / `sigma_x_var` / `w_0^Y` / `w^Y` /
`family_y` / `L` / `num_iter` / replicate 数 / seed 集合。
加えて **HG-1（方式 A/B/C）と HG-3（penalty 方針）は実行前に Human が freeze すること**（§12）。
**本文書はこれらを決定しない。** `prepare-experiment` の pre-flight を通してから実行すること。

---

## 9. K 選択との将来接続（#75、本 Issue では着手しない）

#72 の結果と §6 を合わせると

```
C_Q(K, c) = D_{K,c} + P_Z(K) + P_θ(K, c)
```

で `P_Z` は `c` 非依存、`P_θ` は `K` 部分と `c` 部分に加法分離する（§6.3。
**連続パラメータブロック間の話であり、離散 search cost を含まない**）。
したがって #75 の joint search は形式的には

```
(K̂, ĉ) = argmin_{K, c} C_Q(K, c)
```

と書ける。**ただし次の 2 点が未解決のまま残る。**

1. 各 `K` で `c` を最適化してから `K` を比較すると、`C_Q` は `c` について profile された量になる。
   この profiling の自由度が `p_{K,c}` に入っていない（§6.4 と同じ問題が K 方向にも波及する）。
2. `P_Z(K) ≈ 2.8379 nK` と `P_θ` が同じ向きの K 罰則として二重に効いている（#72 report（PR #76）§5-4）。
   これは現行 criterion の仕様であり、**勝手に統合・削除しない**。

---

## 10. UNRESOLVED

| ID | 内容 |
|---|---|
| U1 | **cross-measure 比較は原理的に尤度で決められない**（§3.3）。count を Gaussian として扱う／log 変換するは representation 選択であり、本設計では解決しない。KI-018 と同じ未解明領域 |
| U2 | 実装済み 3 family に厳密 support gate をかけると、Level 1 の実質的選択肢が「0/1 列の Bernoulli vs Poisson」だけになる（§3.2）。意味のある選択肢を増やすには X 側 NB 等の新規実装が要る |
| U3 | 方式 A の反復が何に収束するか未証明（§2.4）。`q_t` が `c` に依存するため、固定 `q_t` 下の coordinate ascent としてしか述べられない |
| U4 | 方式 C の再 fit で目的関数が悪化した場合の扱いが未定（§7） |
| U5 | family assignment の**離散 search cost** が `p_{K,c}` に入っていない（§6.4）。素朴な `Σ_l log|M_l|` は `c` に関する定数なので selector penalty として機能しない（§6.4.1）。有効な補正（model prior / extended criterion / held-out）は未設計 |
| U6 | X 列 intercept がないため family score が位置適合を含む（§5）。matched generator では intercept omission による model misspecification は生じないが、**位置適合の寄与自体は matched generator でも残る**。**原因の断定はしない**（KI-018） |
| U7 | `Q_X` を family score に再利用する際、Gaussian-X の `Σ_X` が `F` より 1 M-step 古い（#72 report（PR #76）§5-3）。family 比較は score の差を直接使うため、このラグの影響が K 比較より直接的に効く可能性がある。**未測定** |
| U8 | per-column 混在の人工データ生成経路が存在しない（§8.1）。#74 で forward-only に追加する必要がある |
| U9 | lineage E（experimental / consistent）は **本文採用不可**。family 自動選択を修論本文の提案手法にするには、どの系列で正式化するかの決定が要る |

---

## 11. Human Gate（人間の判断が必要。Claude / Codex は実行しない）

| ID | 判断事項 |
|---|---|
| HG-1 | 方式 A / B / C のどれを正式採用するか（本文書の推奨候補は C。**未採用**） |
| HG-2 | #74 の実験条件の freeze（§8.6 の全項目） |
| HG-3 | family 選択の離散 search cost をどう扱うか（§6.4.2 の (i) model prior / (ii) held-out / (iii) 補正なし＋limitation 明記）。**#74 は (iii) ＋ score margin 診断で足りる**が、その承認は必要 |
| HG-4 | 候補集合を X 側 NB 等へ拡張するか（新規実装を伴う。§3.2 / U2） |
| HG-5 | X 列 intercept `b_l` を導入するか（**生成モデルの変更**。root `CLAUDE.md` §6） |
| HG-6 | cross-measure / representation 選択（raw count vs log 変換）を研究範囲に含めるか（§3.3 / U1） |
| HG-7 | どの lineage で正式化するか（U9） |

---

## 12. Decision

## `READY_FOR_MINIMAL_PROTOTYPE`（**実行前提つき**）

この decision が意味するのは「**設計として最小 prototype を書き下せる状態にある**」ことであり、
「**今すぐ #74 を実行してよい**」ことではない。両者を混同しない。

### 設計として揃ったもの

- family 比較を continuous-Z MCEM 上で well-posed に定義する式が得られた（§2.3）。
  `Q_Z` と `Q_Y` が比較から厳密に消えるという代数的事実に立脚しており、追加仮定を要しない。
- 比較してよい範囲（support gate）と、比較に必要な完全な log probability / density の
  項が特定できた（§3, §4）。必要な strict 版・consistent numerics は **既に repository に存在する**。
- parameter count の接続先が確定した（`calc_bic_exp(family_x='mixed', n_gaussian_x_cols=…)`、§6.2）。
  連続パラメータブロック間で重複計上がないことも確認した（§6.3。**離散 search cost は未解決**）。
- #74 が何を調べ、何を主張してはいけないかの claim boundary が確定した（§8.2）。

### **実行前に Human が freeze しなければならないもの（precondition）**

| 前提 | 内容 | なぜ pilot を止めるか |
|---|---|---|
| **HG-1** | 方式 A / B / C の採用 | §8.3 の手順 3–4 が方式に依存する。未決のまま実装すると推奨候補 C を既成事実化してしまう |
| **HG-3** | penalty 方針 | §6.4.2 の (i)/(ii)/(iii) のどれを取るかで選択規則が変わる。**pilot 自体は (iii) raw criterion ＋ score margin 診断で足りる**（§8.4）が、それでよいという承認は必要 |
| **HG-2** | 実験条件の freeze | §8.6。`prepare-experiment` の pre-flight を通すこと |

HG-4〜HG-7 は #74 の実行を止めない（それぞれ候補集合の拡張・モデル変更・研究範囲・正式化系列の話であり、
feasibility pilot の外側にある）。ただし **HG-4 が未決である以上、#74 の結果から
「自動分布選択一般」を主張することはできない**（§8.2 の claim boundary）。

**この decision は「設計が正式採用された」という意味ではない。**
**#74 は自動開始しない。**

---

## 13. Codex / 次セッション向け review checklist

本設計を独立に検証する場合、次の順で確認すれば足りる。**過去 Issue の網羅読みは不要。**

### 一次証拠の再照合（コードを読むだけ）

- [ ] `Q_Z` と `Q_Y` が `c_l` に依存しないこと（`utils_expfam._lnpZ`、`model_expfam.calc_log_likelihood_Y` に `family_x` が現れないこと）。§2.3 の根拠。
- [ ] `calc_log_likelihood_X` が列ごとの和で書かれていること（`model_dual_expfam.py` L.307–334、per-column 版 L.215–237）。§2.2 の根拠。
- [ ] Poisson-X の `−log(x!)` が `calc_log_likelihood_X` にはなく `calc_Q_dual_strict` / `calc_Q_dual_strict_exp` にあること。§4 の根拠。
- [ ] Gaussian-X の `−½ log 2π` が**実装には入っており** docstring と食い違っていること（L.303 vs L.321–323）。
- [ ] `calc_bic_exp` が `family_x='mixed'` で Gaussian 列数だけ数えること（`eval_utils.py` L.247–250）。§6.2 の根拠。
- [ ] `objective_consistent_numerics` の `poisson_log_likelihood` が `−log(x!)` を**含まない**こと（L.83–95）。strict 補正との二重計上がないことの確認。
- [ ] `model_dual_expfam_nb.py` の NB が **Y 側のみ**で、`family_x` は親へ素通しであること（L.63–69）。§3.2 の根拠。
- [ ] `data_generator_canonical.generate_canonical_data` の `family_x` が**スカラー**であること（L.246–262）。§8.1 の gap。

### 主張の過剰さチェック

- [ ] 「EM 全体が単調に改善する」と書いていないか（§2.4 は固定 `q_t` 下の coordinate ascent までに限定）。
- [ ] 「intercept 欠如が family 誤選択の原因」と書いていないか（KI-018。§5 は構造的交絡の指摘までに限定）。
- [ ] 「per-column / family 自動選択が一般に優れる」と書いていないか（KI-016 H 項）。
- [ ] 方式 C を「採用した」と書いていないか（推奨候補まで。HG-1）。
- [ ] 実験条件を freeze していないか（§8.6）。
- [ ] 「#74 で自動分布選択一般を実証できる」と読めないか（§8.2 の claim boundary）。
- [ ] `Σ_l log|M_l|` を selector penalty として扱っていないか（§6.4.1）。
- [ ] 「二重計上は起きない」を無限定に書いていないか（§6.3 は連続パラメータブロック間に限定）。
- [ ] 方式 C の周回数など、未実験の量を見積もっていないか（§7）。
- [ ] lineage E（experimental / consistent）を本文採用可のように書いていないか（root `CLAUDE.md` §3、U9）。
- [ ] 異なる系列の数値を並べていないか（KI-002。本文書は数値比較を一切していない）。

### #72 側（PR #76）と合わせて見るとき

- [ ] `C_Q(K, c)` の `P_Z` が `c` 非依存、`P_θ` が `K` 部分と `c` 部分に加法分離すること（§6.3）。**連続パラメータブロック間で**重複計上がないことの根拠であり、離散 search cost を含む主張ではない。
- [ ] `Q_X` を family score に使うとき、Gaussian-X の `Σ_X` が `F` より 1 M-step 古い件（U7）が未測定のまま残っていること。

---

## 14. Validation

| 項目 | 結果 |
|---|---|
| 新規 EM fit | 0 |
| synthetic run | 0 |
| model implementation 変更 | 0 行 |
| scientific code 変更 | 0 行 |
| results / artifacts 変更 | 0 件 |
| 追加ファイル | 本 report 1 件のみ |
| `git diff --check` | clean |

参照した一次証拠:
`expfam/src/model_dual_expfam.py`（L.274–334）/
`expfam/src/model_expfam.py`（L.241–269）/
`expfam/src/utils_expfam.py`（L.315–404, L.478–566）/
`expfam/src/experimental/model_dual_expfam_percolumn.py`（L.1–27, L.196–241）/
`expfam/src/experimental/model_dual_expfam_consistent.py`（L.1–30, L.110–170）/
`expfam/src/experimental/objective_consistent_numerics.py`（L.44–95）/
`expfam/src/experimental/eval_utils.py`（L.186–256）/
`expfam/src/experimental/data_generator_canonical.py`（L.1–60, L.246–290）/
`KNOWN_ISSUES.md`（KI-015 / KI-018 / H 項）/ `RESEARCH_MASTER.md`（§12.5b, §13 D 項）/
root `CLAUDE.md` / #72 の report `reports/k_selection_theory/cq_decomposition_minimal_check_20260923.md`（branch `audit/72-k-criterion-minimal-check` / PR #76。**本 branch には存在しない**）
