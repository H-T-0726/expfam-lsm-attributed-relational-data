# fixed artificial-data generator における `K_GEN = K_0` の構造条件監査

**作成日:** 2026-09-07
**種別:** READ-ONLY 構造監査（コード変更なし・実験実行なし・generator 再実行なし・新規 simulation なし）
**branch:** `docs/notion-k-selection-story-20260906`（作業前 status: `?? reports/research_scope/` のみ）
**対象:** fixed official Exp1 / K-selection 系列（`K_GEN = 3`, `n = 150`, `d = 15`, scenario A/B/C）
**implementation lineage:** fixed 系列（`DualExpFamLSMFixed`、0.5 なし。root `CLAUDE.md` §3、KI-002）

---

## Evidence labels

| ラベル | 意味 |
|---|---|
| `[CODE]` | 一次コードを直接読んで確認した事実 |
| `[ARTIFACT]` | 既存の結果 CSV から読み取った事実 |
| `[DERIVED]` | 本監査で独立に導出した（証明を本文に置く） |
| `[INFERENCE]` | 既存の派生文書・別条件の推定からの推論。一次証拠ではない |
| `[UNRESOLVED]` | 未解決。証明も反証もしていない |

**本監査は数値実験を一切実行していない。** Z・F の realized 値は repository に保存されていないため
（§5D・§6.3 で確認）、rank に関する主張は生成分布からの `[DERIVED]` であり `[ARTIFACT]` ではない。

---

## 1. Executive summary

fixed generator（`expfam/src/data_generator_expfam.py` の `generate_dual_data`）を一次コードで追跡した結果、

**(a) 素朴な低次元退化は見つからなかった。**
`rank(Z) = 3`・`rank(F) = 3` はいずれも生成分布の下で probability 1 で成立し（§5・§6）、
row normalization は正の対角行列を左から掛ける操作であって **rank を保存する**（§6.2、`[DERIVED]`）。
`w` は A/B/C すべてで非零（1.5 / 0.5 / 0.5）であり、3 つの latent direction すべてが
`eta^Y_ij = w0 + w Σ_{r=1}^{3} z_ir z_jr` に入る（§7、`[CODE]`）。
Poisson clip は A の X 側・B の Y 側ともほぼ確実に不活性である（§9、`[DERIVED]` + `[INFERENCE]`）。

**(b) しかし `K_GEN = K_0` は「未証明」ではなく、B と C では前提そのものが壊れている。**
`K_0 = min{K : p_* ∈ M_K}` は `p_* ∈ ⋃_K M_K`（M-closed）を要求する。本監査は

- **Scenario B（Gaussian-X）**: 生成された `X` が全列で標本平均ちょうど 0・標本二乗平均ちょうど 1 を満たす
  measure-zero 集合の上に確率 1 で乗るため、**`p_*^B` は `M_K` のどの元とも互いに特異**である。
  よって `{K : p_*^B ∈ M_K} = ∅` であり **`K_0` は存在しない**（§12.2、`[DERIVED]`）。
- **Scenario C（Gaussian-Y）**: `T = Σ_{i<j} Y_ij` が厳密に `N(−112.5, 111.75)` になるのに対し、
  `M_K` のどの元でもこの `T` は「非 Gaussian な二次形式 + Gaussian」の畳み込みになる。
  Cramér の分解定理から `w' = 0` が強制され、それは分散条件と矛盾する。
  よって **`p_*^C ∉ M_K`（∀K）であり `K_0` は存在しない**（§12.3、`[DERIVED]`）。
- **Scenario A**: 同じ構造的原因（`Z` の列 z-score）は存在するが、`X`・`Y` とも離散で
  観測レベルの特異性が作れないため **`[UNRESOLVED]`**（§12.4）。

この (b) は KI-021 の G1・G3 を本監査で独立に再導出し、その帰結を `K_0` の定義側まで押し進めたものである。
KI-021 は「canonical model の literal generator ではない」と述べていた。本監査が付け加えるのは
**「そのずれは、B と C では `K_0` を well-defined にする M-closed 前提を実際に破る水準である」**という点である。

**(c) したがって fixed 30/30 は「true-K recovery」として読めない。**
ただし読み方が変わるのは「何を回収したか」の解釈だけであり、**既存の数値結果は一切無効化されない**
（`RESEARCH_MASTER.md` §14・KI-010 の既存 limitation がそのまま正しい）。§13・§19 Q10 を参照。

---

## 2. Scope and definitions

### 2.1 監査対象

| 項目 | 値 | 出所 |
|---|---|---|
| runner（k=1..6） | `expfam/src/run_fixed_official_exp1_bic_full.py` | `EXPERIMENT_REGISTRY.md:53` |
| runner（k=7..9） | `expfam/src/run_fixed_official_exp1_bic_k9_extension.py` | 同上 |
| generator | `expfam/src/data_generator_expfam.py:223` `generate_dual_data` | `[CODE]` |
| model | `expfam/src/model_dual_expfam_fixed.py:33` `DualExpFamLSMFixed` | `[CODE]` |
| criterion | `expfam/src/utils_expfam.py:386` `calc_bic_dual`（Q ベース。Schwarz BIC ではない、KI-010） | `[CODE]` |
| 主結果 CSV | `expfam/results/fixed_official/exp1_k9/fixed_exp1_bic_k1to9_bestk_by_trial.csv` | `[ARTIFACT]` |

`n = 150`, `d = 15`, `K_GEN = 3`, `L = 5`, `num_iter = 8`, 候補 `K = 1..9`, 各シナリオ 10 trial。
`run_fixed_official_exp1_bic_full.py:47,49,50` / `run_fixed_official_exp1_bic_k9_extension.py:59,61,64`。`[CODE]`

### 2.2 中心にしないもの

clean generator（`experimental/data_generator_canonical.py`）、S1/S2/S3、held-out、per-column、
`K_PRED` は本監査の対象外。historical 旧 0.5 系列は §4.4 の「同一 generator である」という
一点でのみ参照する。

### 2.3 用語

`K_GEN`（generator が `Z` を何列作ったか）／`K_0`（観測分布の最小潜在次元）／`K_hat`（criterion の選択値）
を厳密に区別する。この区別は本監査の独自定義ではなく、
`reports/identifiability/true_k_identifiability_hardened_20260904.md` §2.3 の `K_TRUE` / `K*` と同じものである
（同文書は `K*` 表記、本文書は依頼に合わせて `K_0` 表記）。

---

## 3. `K_GEN` vs `K_0` vs `K_hat`

```
M_K = { p_{theta,K}(X, Y) = ∫ p(Z) p_{theta,K}(X|Z) p_{theta,K}(Y|Z) dZ : theta ∈ Theta_K }
      ただし p(Z) は z_i ~ iid N(0, I_K)

K_0 = min { K : p_* ∈ M_K }
```

| 記号 | 本件での値・状態 |
|---|---|
| `K_GEN` | **3**（`run_*` の `K_TRUE = 3` が `generate_dual_data(k=3)` に渡る）`[CODE]` |
| `K_0` | **A: `[UNRESOLVED]` / B: 存在しない / C: 存在しない**（§12） |
| `K_hat` | **3**（A/B/C × 10 trial = 30/30）`[ARTIFACT]` |

**重要:** `K_0` は `M_K` の定義に依存する。上の `M_K` は
`true_k_identifiability_hardened_20260904.md` §2.1 の canonical 定義であり、
**推定側 `DualExpFamLSMFixed` が実際に仮定している族**でもある（§8.1 で確認）。
別の定義（生成 pipeline 自身を族とみなす読み方）を採ると §12 の結論は変わる。§12.5 に併記する。

---

## 4. Exact fixed generator code path

### 4.1 call chain

```
run_fixed_official_exp1_bic_full.py:219-222         (k = 1..6)
run_fixed_official_exp1_bic_k9_extension.py:221-224 (k = 7..9)
        seed_data = 4000 + SEED_OFFSETS[scen] + trial*100
        │  A: 4000,4100,...,4900   B: 5000,...,5900   C: 6000,...,6900
        ▼
data_generator_expfam.py:223  generate_dual_data(n=150, d=15, k=3, seed=seed_data,
                                                 family_x=…, family_y=…)
        │   ※ var_f / uniq / w0_true / w_true / sigma_y_true はすべて既定値のまま
        ▼
  :272-276  w0_true, w_true ← _Y_DEFAULTS[family_y]        (:216-220)
  :278      rng = np.random.default_rng(seed)              ← 単一 rng を以降で共有
  :282      Z   = rng.normal(0, 1, (150, 3))
  :283      Z   = normalize_zscore(Z, axis=0)              → reproduction/src/data_generator.py:22-42
  :285      sigma_x = diag(0.1 * ones(15))                 (uniq = 0.1)
  :286      F   = rng.normal(0, sqrt(5.0), (15, 3))        (var_f = 5.0)
  :287-290  F[j,:] ← F[j,:]/‖F[j,:]‖ * sqrt(1 − 0.1)       ← 行正規化（‖f_l‖² = 0.9）
  :293      eta_x = Z @ F.T                                (150×15、バイアスなし)
        ├─ :295-298  gaussian : X = eta_x + N(0, sigma_x); X = normalize_zscore(X, axis=0)
        ├─ :299-301  bernoulli: X = Binomial(1, sigmoid(clip(eta_x, ±500)))
        └─ :303-306  poisson  : X = Poisson(exp(clip(eta_x, -20, 10)))
  :309-310  eta_y = w0_true + w_true * (Z @ Z.T)
        ├─ :315-317  bernoulli: Y_upper = Binomial(1, sigmoid(clip(eta_y, ±500)))
        ├─ :318-320  poisson  : Y_upper = Poisson(exp(clip(eta_y, -20, 10)))
        └─ :321-323  gaussian : Y_upper = eta_y_upper + N(0, sigma_y_true=0.1)
  :325-326  Y = Y + Y.T ; fill_diagonal(Y, 0)
```

`[CODE]`。**historical docs ではなく、上記 2 runner が実際に import・呼び出す関数を追跡した結果である。**

### 4.2 シナリオ別の真パラメータ

| Scen | `family_x` | `family_y` | `w0` | `w` | `sigma_y` | 出所 |
|---|---|---|---|---|---|---|
| A | poisson | bernoulli | −1.0 | **1.5** | — | `:217` `_Y_DEFAULTS["bernoulli"]` |
| B | gaussian | poisson | 0.0 | **0.5** | — | `:218` `_Y_DEFAULTS["poisson"]` |
| C | bernoulli | gaussian | 0.0 | **0.5** | 0.1 | `:219` `_Y_DEFAULTS["gaussian"]` / `:237` |

両 runner とも `w0_true` / `w_true` / `sigma_y_true` / `var_f` / `uniq` を**明示指定していない**ので、
上表はすべて既定値経由である。`run_fixed_official_exp1_bic_full.py:219-222` /
`run_fixed_official_exp1_bic_k9_extension.py:221-224`。`[CODE]`

### 4.3 seed

`seed_data = 4000 + {A:0, B:1000, C:2000} + trial*100`、`seed_model = seed_data + 50`。
両 runner で同一規約（`_full.py:55-56,215-216` / `_k9_extension.py:66-67,217-218`）。
生成は `default_rng(seed_data)` 一本で Z → F → X → Y を順に引くため、
**同一 `(scenario, trial)` の 9 個の候補 K は完全に同じ `(X, Y, Z, F)` を共有する**。`[CODE]`
これは K 比較として望ましい設計だが、trial 間の独立性は 10 個しかないことを意味する。

### 4.4 historical 系列との関係（差の説明が必要な場合のみ）

`generate_dual_data` は fixed 系列専用の関数ではなく、**historical 旧 0.5 系列と同一の関数**である
（`EXPERIMENT_REGISTRY.md:470` の historical P/B K5 n-sweep も同じ `generate_dual_data` を使う）。
fixed / historical の差は**推定側（`DualExpFamLSMFixed` vs `run_em_dual`）にあり、生成側にはない**。`[CODE]`
したがって §5-§12 の生成側の所見は historical にもそのまま当てはまる。
**結果 CSV・数値は決して混在させない**（KI-002）。

---

## 5. Z generation and rank

### 5.A 生成前

`data_generator_expfam.py:282` — `Z_raw = rng.normal(0.0, sqrt(1.0), size=(150, 3))`。
各要素 iid `N(0,1)`、すなわち `z_i ~ N(0, I_3)` iid。**この段階では canonical model と一致する。** `[CODE]`

### 5.B standardization の正確な内容

`:283` `Z = normalize_zscore(Z, axis=0)` → `reproduction/src/data_generator.py:38-42`:

```
mean = mean(X, axis=0)          # 列平均
std  = std(X, axis=0, ddof=0)   # 列標準偏差（ddof=0）
std  = where(std == 0, 1.0, std)
return (X - mean) / std
```

したがって

```
Z = M Z_raw D^{-1},   M = I_150 − (1/150) 1 1^T,   D = diag(列 SD)
```

- **列ごとの mean centering を行う。** 行ごとではない。
- **列ごとの variance normalization を行う。** ddof=0 なので結果の列標本分散はちょうど 1。
- **whitening ではない。** 単純な列 z-score であり、**latent coordinates 間の標本相関は除去しない**。
  `(1/n) Z^T Z` は対角がちょうど 1 になるが、非対角は 0 にならない。`[CODE]` `[DERIVED]`

**厳密に成立する制約（浮動小数点誤差を除く）:**

```
Σ_i z_ir = 0            (r = 1,2,3)      … 3 本
Σ_i z_ir² = n = 150     (r = 1,2,3)      … 3 本
⟹ ‖Z‖_F² = n·K = 450   （厳密）
```

`[DERIVED]`。この 6 本の制約が §12 の中心的な論拠になる。

### 5.C rank(Z) = 3 の証明

> **主張 Z1.** `rank(Z) = 3` が probability 1 で成立する。`[DERIVED]`

**証明.** `rank(M Z_raw) < 3` ⟺ ある `v ≠ 0` で `M Z_raw v = 0` ⟺ `det(Z_raw^T M Z_raw) = 0`。
`Z_raw ↦ det(Z_raw^T M Z_raw)` は `Z_raw` の要素の多項式であり、恒等的に 0 ではない
（`rank(M) = n−1 = 149 ≥ 3` なので `M` の像の中に 1 次独立な 3 本を取れる）。
恒等的に 0 でない多項式の零集合は `R^{150×3}` で Lebesgue 測度 0。`Z_raw` は密度を持つので確率 0。
次に `D` は正定値対角である（列 SD = 0 ⟺ 列が定数、これも確率 0。なお `std == 0` の分岐が
`1.0` に置換するので零除算は起きず、その場合は当該列がそのまま 0 列として残る）。
可逆な対角行列を右から掛けても rank は変わらない。ゆえに `rank(Z) = rank(M Z_raw) = 3` a.s. ∎

**standardization 後も rank は保たれる**（centering は rank を落としうる射影だが、
`n−1 = 149 ≫ K = 3` なので落ちるのは測度 0 の集合上だけ。scaling は正対角なので常に保たれる）。`[DERIVED]`

### 5.D 既存 artifact からの確認可否

`expfam/results/fixed_official/` 以下に `Z_true` を保存した npy/npz/csv は**存在しない**
（`exp1/` は `*_agg.csv` / `*_bestk.csv` / `*_summary.csv` の 3 本のみ、
`exp1_k9/` は 7 本の集計 CSV のみ）。repository 全体でも `.venv` 外に `.npy` / `.npz` は無い。`[ARTIFACT]`

したがって **realized な 30 個の `Z` について rank / 特異値を数値確認することはできない。**
再生成は禁止条件に触れるため実施しない。`rank(Z) = 3` は **`[DERIVED]`（生成分布の主張）** であり、
**realized レベルは `[UNRESOLVED]`** である。ただし主張 Z1 の零集合は測度 0 であり、
コード上 rank を落とす経路（列の zero 化・重複列の代入・上書き）は存在しない。`[CODE]`

---

## 6. F generation, normalization and rank

### 6.1 生成の完全な内容

`data_generator_expfam.py:286-290`:

```python
F = rng.normal(0.0, np.sqrt(var_f), size=(d, k))     # var_f = 5.0, (15, 3)
for j in range(d):
    nrm = np.linalg.norm(F[j, :])
    if nrm > 0:
        F[j, :] = (F[j, :] / nrm) * np.sqrt(1.0 - sigma_x[j, j])   # sqrt(0.9)
```

| 項目 | 内容 |
|---|---|
| shape | `(15, 3)` = `(d, K_GEN)`。`d ≥ K` を満たす |
| entry distribution | iid `N(0, 5.0)` |
| normalization | **row normalization のみ**。全行を `‖f_l‖ = sqrt(0.9)` に固定 |
| column normalization | **なし** |
| zeroing | **なし**（`nrm > 0` ガードは零除算回避であり、零化ではない） |
| clipping | **なし** |
| rescaling | 上記 row normalization 以外なし |
| 生成後の上書き | **なし**（`F` は `:290` 以降変更されない） |

`[CODE]`

### 6.2 row normalization は rank を保存する

> **主張 F1.** `rank(F) = rank(F_raw)`。`[DERIVED]`

**証明.** `nrm > 0` がすべての行で成り立つとき（下記 F2 より a.s.）、

```
F = D_F F_raw,   D_F = diag( sqrt(0.9) / ‖F_raw[j,:]‖ )_{j=1..15}
```

`D_F` は正の対角成分のみを持つので可逆。可逆行列を左から掛けても rank は変わらない。∎

**これは依頼文 §7 が提示した `F_new = D F_old`（`D` 可逆対角）という仮説そのものであり、
コード上その通りであることが確認された。** 行正規化は rank を壊していない。

> **主張 F2.** `rank(F) = 3` が probability 1 で成立する。`[DERIVED]`

**証明.** `F_raw` は `R^{15×3}` 上に密度を持つ iid Gaussian。`rank(F_raw) < 3` は
`det(F_raw^T F_raw) = 0` と同値で、これは恒等的に 0 でない多項式の零集合、測度 0。
また `‖F_raw[j,:]‖ = 0` は確率 0 なので F1 が適用でき、`rank(F) = 3` a.s. ∎

- **zero column なし**（a.s.）／**exact duplicate・線形従属列なし**（a.s.）／
  **near-zero singular value:** `σ_min(F) > 0` a.s. だが、**realized な値は artifact から確認できない**（§6.3）。
- **rank-deficient normalization なし**（F1）。

### 6.3 既存 artifact からの確認可否

`F_true` を保存した artifact は **存在しない**（§5.D と同じ理由）。
したがって realized な `exact/numerical rank`・`singular values`・`condition number`・`σ_min` は
**`[UNRESOLVED]`** である。新規 `F` の生成は行っていない。

`clean_true_k_asymptotics_20260904/gram_spectrum.csv` は存在するが、これは
**canonical clean generator（`canonical-clean-v1`）の artifact であり、fixed generator のものではない**
（`EXPERIMENT_REGISTRY.md:417`）。系列が違うので本監査では `F` の証拠に使わない（KI-002）。

### 6.4 row normalization の副作用（rank とは別問題）

全行が `‖f_l‖² = 0.9` に固定されるため、**`F` は自由パラメータではない**（KI-021 G2 を独立に再確認）。
これは rank を落とさないが、`Theta_K` の特別な点を選んでいることを意味する。
とくに Poisson-X では全列で `E[X_l]` が共通値になる。`[CODE]` `[DERIVED]`
**これは識別可能性を壊す方向の制約ではない**（自由度を減らすだけ）が、
「generic な `F` について確認した」とは書けない。

---

## 7. Y-side `w` and inner-product structure

| 問い | 回答 | 根拠 |
|---|---|---|
| `w` の生成方法 | 生成されない。`_Y_DEFAULTS` の定数（`:216-220`）が `:272-276` で代入される | `[CODE]` |
| A/B/C すべてで `w ≠ 0` か | **YES**。A: 1.5、B: 0.5、C: 0.5 | `[CODE]` |
| `w` は候補 K で変わるか | **変わらない。** `generate_dual_data` は `k=K_GEN=3` 固定で 1 回だけ呼ばれ、候補 `k_est` は EM 側のみに渡る（`_full.py:219-227`） | `[CODE]` |
| `w0` は K の情報を持つか | **持たない。** `eta^Y` の定数オフセットであり `Z` に依存しない。単独では latent dimension の情報を含まない | `[DERIVED]` |
| 全 K coordinate が入るか | **入る。** `:310` `eta_y = w0_true + w_true * (Z @ Z.T)`、`(Z Z^T)_{ij} = Σ_{r=1}^{3} z_ir z_jr` | `[CODE]` |
| 方向別 parameter はあるか | **ない。** `w` はスカラーで全 latent direction に共通に作用する。`w_r` のような per-direction 係数は存在しない | `[CODE]` |

**確認: `w ≠ 0` から `K` の識別可能性は結論しない。** これは C3 の条件確認にすぎない。

### 7.1 `S = z_i^T z_j` の分布形（canonical model の場合）

`z_i, z_j ~ iid N(0, I_K)` のとき

```
M_S(t) = (1 − t²)^{−K/2}   (|t| < 1)
kappa_2(S) = K,  kappa_4(S) = 6K,  kappa_6(S) = 120K,  奇数次はすべて 0
E[S] = 0,  Var(S) = K
E[S⁴] = kappa_4 + 3 kappa_2² = 6K + 3K²
standardized kurtosis = E[S⁴]/Var(S)² = 3 + 6/K       ⟸ 超過尖度 6/K
```

`K = 3` では超過尖度 = 2。`[DERIVED]`
（同一式は `true_k_identifiability_hardened_20260904.md` §9.1-9.2 で
`[NUMERIC_CHECK]`（rel err ≤ 0.030）済み。本監査は独立に再導出した。）

**含意（限定付き）:** `K` は `S` に **scale としてだけでなく distribution shape として**入る
（超過尖度 `6/K` は `w` の再スケールで消せない）。
**ただしこれをもって Bernoulli/Poisson/Gaussian-Y の観測分布で `K` が識別可能とは結論しない。**
Gaussian-Y については既存の命題 P2/P3 がこれを完成させているが（§8.1・§12.3）、
**その前提（`‖z_j‖² ~ chi²_K`）は fixed generator では成立しない**（§5.B の列 z-score）。`[DERIVED]`

---

## 8. Family-specific nondegeneracy

### 8.1 推定側モデルの link（`M_K` の実体確認）

| family | 生成側 | 推定側（`M_K` の実体） | 一致 |
|---|---|---|---|
| Gaussian-X | `eta + N(0, diag(0.1))`、**その後 z-score** | `N(eta, diag(sigma_l²))`、z-score なし（`model_dual_expfam.py:314-323`） | **不一致** |
| Bernoulli-X | `sigmoid(clip(eta, ±500))` | `sigmoid(clip(eta, ±500))`（`:324-327`） | 一致 |
| Poisson-X | `exp(clip(eta, −20, 10))` | `exp(clip(eta, −20, 10))`（`:328-330`、平均関数は `:102`） | 一致 |
| Bernoulli-Y | `sigmoid(clip(eta, ±500))` | 同（`model_expfam.py:255-258`） | 一致 |
| Poisson-Y | `exp(clip(eta, −20, 10))` | 同（`model_expfam.py:259-261`） | 一致 |
| Gaussian-Y | `N(eta, 0.1²)` | `N(eta, sigma_y²)`（`model_expfam.py:262-264`） | 一致 |

`[CODE]`。**重要な帰結が 2 つある。**

1. **Poisson の clip は生成側と推定側で同一である。** すなわち clip は
   generator/estimator の誤指定ではなく、**実装された `M_K` そのものが clipped link を持つ**。
   その代わり、canonical unclipped link を前提とする命題 P1 は **この `M_K` には適用できない**（KI-021 G4 と整合）。
2. **Gaussian-X だけが構造的に不一致である。** 生成側の追加 z-score（`:298`）に対応する操作が
   推定側に存在しない。§12.2 の根拠になる。

なお推定側の `scale_Z`（`reproduction/src/model.py:468-503`）は
**全要素共通のスカラー倍**であり、生成側の**列ごと** z-score とは別物である。
推定側が生成側の制約を復元しているわけではない。`[CODE]`

### 8.2 各シナリオの非退化性（signal が観測へ反映されるか）

`‖f_l‖ = sqrt(0.9) ≈ 0.9487`、`Z` は列 SD ちょうど 1、`Var(S) ≈ K = 3`（近似、§5.B の制約により厳密ではない）。

| Scen | X 側 | Y 側 |
|---|---|---|
| **A** | `lambda = exp(f_l^T z_i)`、`sd(eta_x) ≈ 0.95` → `lambda` はおよそ `exp(±2)` に広がる。定数化しない | `sigmoid(−1 + 1.5 S)`、`sd(1.5S) ≈ 2.60` → 確率が 0/1 に張り付く構造ではない |
| **B** | `X = Zf + N(0,0.1)`、信号分散 ≈ 0.9 対 雑音分散 0.1 → 列 z-score 後も信号比 0.9 が保存。潰れない | `lambda = exp(0.5 S)`、`sd(0.5S) ≈ 0.87` → 定数化しない |
| **C** | `sigmoid(f_l^T z_i)`（**切片なし**）、`sd(eta) ≈ 0.95` → `p` は概ね 0.2〜0.8 | `N(0.5 S, 0.1²)`、平均部分の sd ≈ 0.866 対 雑音 sd 0.1 → **分散比 75**。3 シナリオ中で最強の signal |

`[DERIVED]`（`[CODE]` の定数から）。

**確認できない点:** realized な `y_density` / `x_count_max` / `lambda_max` は
`generate_dual_data` が返り値に含めているにもかかわらず（`:343-361`）、
**両 runner が CSV に保存していない**（`_full.py:263-283` の row dict に含まれない）。
したがって realized なデータの退化（Bernoulli 確率がほぼ一定、Poisson が clip 境界に張り付く等）を
**artifact から確認することはできない。`[UNRESOLVED]`**

**注意（P7、既存命題）:** Scenario C の Bernoulli-X は切片を持たないため、
`Z` が 0 対称なら `E[X_l] = 1/2` が `K` によらず成立する
（`true_k_identifiability_hardened_20260904.md` §8.1、`[NUMERIC_CHECK]` 済み）。
**X の一次モーメントは `K` の情報を持たない。** これは「signal が弱い」ではなく
「そのモーメントに情報がない」という構造的事実である。

**「signal が弱い」と「population identifiability がない」を混同しない。** §11 で分離する。

---

## 9. Poisson clipping audit

clip が発動しうるのは **A の X 側**（`:304`）と **B の Y 側**（`:319`）の 2 箇所のみ。
C には Poisson がないので clip はない（`clip(±500)` は sigmoid の飽和域であり数値的に無害、KI-021 G6）。

### 9.1 Scenario A の X 側

`eta_x,il = f_l^T z_i`、`|eta_x| ≤ ‖f_l‖ ‖z_i‖ = 0.9487 ‖z_i‖`。
§5.B より `Σ_i ‖z_i‖² = 450` が**厳密に**成立するので

```
max_i ‖z_i‖² ≤ 450   ⟹   |eta_x| ≤ 0.9487 · sqrt(450) = 20.13     （決定論的上界）
```

- **上側 clip（+10）が発動する条件:** `‖z_i‖² ≥ (10/0.9487)² = 111.1`。
  すなわち **1 ノードが latent energy 総量 450 の 24.7% を単独で占める**必要がある（平均持ち分は 0.67%）。
  生成分布の下では `‖z_i‖²` はおよそ `chi²_3` なので `P ≈ 6e−24`。
  A の全エントリ数は 10 trial × 150 × 15 = 22,500 なので期待発動数は `≈ 1e−19`。
- **下側 clip（−20）が発動する条件:** `‖z_i‖² ≥ 444.4`、総量の 98.8%。
  他の 149 ノードがほぼ原点になる必要があり、列 SD = 1 の制約と両立しない。

**判定: 決定論的には排除できないが、ほぼ確実に不活性。** `[DERIVED]`
realized データでの確認は artifact 不在のため **`[UNRESOLVED]`**（§8.2）。

### 9.2 Scenario B の Y 側

`eta_y = 0.5 S`。clip は `S > 20` または `S < −40` で発動。canonical では `sd(S) = sqrt(3) = 1.73` なので
`S > 20` は 11.5 sd に相当する。

既存の推定として、`true_k_identifiability_hardened_20260904.md` §11.3 が
Reviewer A による 2×10⁷ dyad の見積り `P(eta_y > 10) ≈ 5e−08`（K=3, w=0.5）を記録している。
B の総 dyad 数は 10 trial × 11,175 = 111,750 なので期待発動数 `≈ 0.006`。
**`[INFERENCE]`** — これは canonical（z-score なし）`Z` に対する別文書の推定であり、
fixed generator の realized データについての一次証拠ではない。

### 9.3 clip と population identifiability の関係（区別すべき点）

**clip がほぼ不活性であることは、clip が無害であることを意味しない。**
clip 領域の確率は極小だが厳密に 0 ではないので、clipped link の population 分布は
unclipped canonical link のそれと厳密には異なる。
その結果 **命題 P1（Poisson-X の `FF^T` モーメント復元）の前提は形式的に破れる**
（§8.1 の通り、`M_K` 自体も clipped link を持つため、P1 は `M_K` の主張として使えない）。
これは KI-021 が既に記録している事実の再確認である。`[DERIVED]`

### 9.4 Poisson-Y の分散発散（B）

`w = 0.5` は canonical unclipped Poisson-Y で `E[lambda²] = ∞` の境界そのものである
（既存の観察 O1、`hardened` §11.3）。
ただし同文書自身が注意している通り、**fixed generator は `Z` を列 z-score しているので
`‖z_j‖² ~ chi²_K` が成立せず、O1 の式は fixed データの分布には literal に適用できない。**
本監査もこの点を維持する。`[INFERENCE]`

---

## 10. Joint X+Y structural activity

依頼文 §11 の 4 ケース分類に対する判定。

| Case | 条件 | fixed A/B/C の該当 |
|---|---|---|
| Case 1 | `rank(F) = K` かつ `w ≠ 0` | **A・B・C すべてが該当**（§6.2 F2、§7）`[DERIVED]` + `[CODE]` |
| Case 2 | `rank(F) < K` かつ `w ≠ 0` | 該当なし |
| Case 3 | `rank(F) = K` かつ `w = 0` | 該当なし |
| Case 4 | `rank(F) < K` かつ `w = 0` | 該当なし |

**各 latent direction `r = 1,2,3` の activity（C5）:**

- X 側: `F[:, r] ≠ 0` a.s.（`rank(F) = 3` より各列は非零）。`eta_x = Z F^T` は全 3 列を使う。`[DERIVED]`
- Y 側: `w z_ir z_jr` の項が全 `r` について存在する。ただし **係数はスカラー `w` で共通**であり、
  方向別に強度を変える parameter は存在しない。`[CODE]`

**したがって 3 方向すべてが X・Y の双方を通じて観測生成に影響する。**
**ただし Case 1 に該当することは十分性を意味しない**（§12・§16）。

---

## 11. Population identifiability vs finite-sample recoverability

両者を混同しないための分類。

| 生成側の要素 | 主に何へ効くか | 理由 |
|---|---|---|
| `Z` の列 z-score（G1） | **identifiability の前提（M-closedness）** | 生成 `Z` の法則がモデルの事前分布と異なる。§12 の中心 |
| `X` の列 z-score（G3、B のみ） | **identifiability の前提（M-closedness）** | 観測 `X` を measure-zero 集合に押し込む。§12.2 |
| `F` の行正規化（G2） | **finite-sample ease**（identifiability は不変） | rank 保存（F1）。全列の SNR を均一化するので推定は容易になる |
| `w` の大きさ（A: 1.5、B/C: 0.5） | **主に finite-sample ease** | `w ≠ 0` なら population 情報は存在。大きさは推定の容易さを決める |
| `sigma_y = 0.1`（C） | **finite-sample ease** | Y の SNR 分散比 75。C の BIC 分離が突出して大きい理由（§13） |
| Poisson clip（G4） | **identifiability の前提**（極小だが厳密に非零） | §9.3 |
| `n = 150`, 10 trial | **finite-sample recoverability のみ** | population の主張は一切含まない |

**`w != 0` でも非常に小さければ population では情報があっても finite sample では実質不可**という
依頼文の指摘は正しく、fixed では逆に `w` が大きめ（A で 1.5）で `sigma_y` が小さい（C で 0.1）ため、
**強い signal が 30/30 を助けている可能性が高い。ただしそれは population identifiability の証拠ではない。**

---

## 12. Lower-dimensional collapse checks / M-closedness

### 12.1 明白な低次元縮約構造の探索（C6）

| 縮約経路 | fixed generator での判定 |
|---|---|
| `rank(F)` 低下 | **なし**（F2、a.s.）`[DERIVED]` |
| `w = 0` | **なし**（1.5 / 0.5 / 0.5）`[CODE]` |
| `rank(Z)` 低下 | **なし**（Z1、a.s.）`[DERIVED]` |
| latent coordinate が常に 0 | **なし**（列 SD = 1 の制約により、どの列も恒等的に 0 になれない）`[DERIVED]` |
| normalization で 1 方向が消える | **なし**（列 z-score も行正規化も可逆な対角作用、F1・Z1）`[DERIVED]` |
| generator が一部列しか使わない | **なし**（`eta_x = Z @ F.T` と `Z @ Z.T` がどちらも全 3 列を使う）`[CODE]` |
| K-independent sufficient statistic のみで生成 | **なし**（`eta` は `Z`・`F` に依存）`[CODE]` |

**結論: `K_0 < K_GEN` を強制する明白な構造的退化は見つからなかった。** `[DERIVED]`

### 12.2 Scenario B — `K_0` は存在しない

> **主張 B1.** Scenario B の `p_*^B` は、`M_K` のどの元とも互いに特異である。
> したがって `{K : p_*^B ∈ M_K} = ∅` であり、**`K_0` は定義されない。** `[DERIVED]`

**証明.** `data_generator_expfam.py:298` により、生成された `X ∈ R^{150×15}` は全列 `l` について

```
Σ_i X_il = 0        かつ        (1/150) Σ_i X_il² = 1
```

を厳密に満たす。この集合 `S ⊂ R^{150×15}` は 2d = 30 本の等式で定義されるので
Lebesgue 測度 0（余次元 30）。生成側は `p_*^B(X ∈ S) = 1`。

一方、`M_K` の任意の元では `X | Z ~ N(Z F^T, Sigma)`、`Sigma = diag(sigma_l²)` かつ `sigma_l² > 0`
（推定側 `model_dual_expfam.py:314-323` が実際に仮定する形。`sigma_diag = maximum(diag(sigma), 1e-8)` により
正値が保証されている）。条件付き分布が `R^{150×15}` 上で絶対連続なので、`Z` について積分しても
`p(X ∈ S) = ∫ P(X ∈ S | Z) dp(Z) = 0`。

（`sigma_l² = 0` を許す極限を考えても救われない。その場合 `X = Z F^T` は rank ≤ 3 だが、
生成側の `X` は雑音により a.s. rank 15 である。）

`p_*^B(X ∈ S) = 1` と `p(X ∈ S) = 0` は両立しないので `p_*^B ≠ p` for all `p ∈ ⋃_K M_K`。∎

**含意:** Scenario B について `K_GEN = K_0` は「未証明」ではなく **ill-posed** である。
X 側の生成過程がモデル族の外にあるため、`K_0` を定義する `min` の対象集合が空になる。

（浮動小数点では列平均は厳密に 0 ではなく `O(1e-17)` である。上の証明はコードの厳密算術意味論に対するもので、
浮動小数点上でも「モデル族が説明できない量の集中」という結論の実質は変わらない。）

### 12.3 Scenario C — `K_0` は存在しない

> **主張 C1.** Scenario C の `p_*^C` は `M_K` のどの元とも一致しない。
> したがって **`K_0` は定義されない。** `[DERIVED]`

**証明.** 統計量 `T = Σ_{i<j} Y_ij` を取る。

**(i) 生成側での `T` の分布。** §5.B の厳密制約から

```
Σ_{i<j} z_i^T z_j = ½( ‖Σ_i z_i‖² − Σ_i ‖z_i‖² ) = ½( 0 − 450 ) = −225   （厳密、確率 1）
```

`w0 = 0, w = 0.5, sigma_y = 0.1`、dyad 数 `11175 = 150·149/2` なので

```
T = 0.5·(−225) + Σ_{i<j} eps_ij = −112.5 + N(0, 11175·0.01)
  ⟹  T ~ N(−112.5, 111.75)      （厳密に Gaussian、Z に依存する変動はゼロ）
```

**(ii) `M_K` 側での `T` の分布。** `p ∈ M_K` を `(w0', w', sigma_y')` とすると

```
T' = 11175·w0' + w'·Q + E,     Q = Σ_{i<j} z_i^T z_j,   E ~ N(0, 11175 sigma_y'²)
```

で `Q`（`Z` の関数）と `E` は独立。`Q = Σ_{r=1}^{K} q_r`、`q_r = ½ u_r^T (J_n − I_n) u_r`、`u_r ~ N(0, I_n)`。
`½(J−I)` の固有値は `(n−1)/2`（1 重）と `−1/2`（`n−1` 重）なので、二次形式のキュムラント公式
`kappa_m = 2^{m−1}(m−1)! tr(A^m)` より

```
tr(A³) = ((n−1)/2)³ + (n−1)(−1/2)³ = (149³ − 149)/8 = 413475
kappa_3(Q) = K · 2²·2! · 413475 = 3 · 8 · 413475 = 9,923,400  ≠ 0
```

すなわち `Q` は Gaussian ではない。

**(iii) 矛盾。** (i) より `T` は厳密に Gaussian。`T' = const + w'Q + E` で `w'Q` と `E` は独立なので、
**Cramér の分解定理**（独立な 2 変数の和が正規なら各々が正規）より `w'Q` は正規でなければならない。
`Q` は非退化かつ非正規（`kappa_3 ≠ 0`）なので `w' = 0` が強制される。

`w' = 0` のとき `M_K` の Y は `Y_ij ~ iid N(w0', sigma_y'²)` となり、
`Var(T') = 11175·sigma_y'²`。これを `Var(T) = 111.75` に合わせると `sigma_y'² = 0.01`。
しかしそのとき単一 dyad の分散は `Var(Y'_ij) = 0.01` であるのに対し、生成側は
`Var(Y_ij) = w² Var(S) + sigma_y² ≈ 0.25·3 + 0.01 = 0.76`。両立しない。∎

**含意:** Scenario C も `K_GEN = K_0` は ill-posed。
**しかも C は Gaussian-Y であり、既存命題 P2/P3 が `K` の population identifiability を
証明できていた唯一の family である**（`hardened` §9.4・§12.2）。
その P2/P3 は `‖z_j‖² ~ chi²_K` すなわち iid `N(0, I_K)` prior を前提としており、
**fixed generator の列 z-score がまさにその前提を壊している。**
すなわち **「本来なら証明が届いたはずの唯一のシナリオで、generator 側の後処理が証明の適用を阻んでいる」。**

### 12.4 Scenario A — `[UNRESOLVED]`

A では `X` が Poisson（離散・格子上に台を持つ）、`Y` が Bernoulli（`{0,1}`）で、
**生成後の連続変換が一切ない**。したがって §12.2 の特異性論法も §12.3 の Cramér 論法も使えない
（`T = Σ_{i<j} Y_ij` は整数値でありモデル側も整数値）。

一方、`K_0` を well-defined にする M-closed 前提が成立する積極的根拠もない。
生成側の `Z` の法則（余次元 6 の制約付き）とモデルの iid `N(0,I_3)` prior は別物であり、
`Z ↦ p(X,Y|Z)` は単射ではない（`O(K)` 不変性、`hardened` §4）ので、
**周辺分布が偶然一致する可能性を排除も証明もできない。**

**判定: `p_*^A ∈ ⋃_K M_K` かどうかは `[UNRESOLVED]`。したがって `K_0^A` の存在も未確定であり、
`K_GEN = K_0` は当然 `[UNRESOLVED]`。**

### 12.5 `M_K` の定義を変えた場合（別読み）

`M_K` を「canonical model の族」ではなく **「fixed 生成 pipeline 自身を `K` でパラメータ化した族」**
（列 z-score・行正規化・clip を族の定義に含める）と取り直すと、§12.2・§12.3 の特異性・Cramér 論法は消える。
その読み方の下では A/B/C いずれも `p_* ∈ M̃_{K_GEN}` が構成的に成立するので `K_0 ≤ 3` は自明になり、
**残る問いは「`K < 3` の pipeline が同じ joint 分布を作れないか」だけになる。**
これは §16 の未解決事項そのものであり、**本監査ではどの family についても証明していない。**

**どちらの読み方を採るかは研究上の判断であり、Human Gate に属する。** 本監査は決めない。

---

## 13. Relationship to fixed 30/30

### 13.1 既存事実（一次 artifact）

`expfam/results/fixed_official/exp1_k9/fixed_exp1_bic_k1to9_bestk_by_trial.csv`：
30 行すべて `best_k_by_trial_bic = 3`, `selected_correctly = True`。
`fixed_exp1_bic_k1to9_summary.csv` は 270 行（3 scen × 9 k × 10 trial）、`success` が全 True。`[ARTIFACT]`

trial 平均の criterion 値（`fixed_exp1_bic_k1to9_agg.csv`、`bic_mean`）`[ARTIFACT]`：

| k | A | B | C |
|---|---|---|---|
| 1 | 20939.2 | 44015.7 | 6134.9 |
| 2 | 18544.2 | 37689.4 | −1967.3 |
| **3** | **16459.6** | **32224.0** | **−36228.4** |
| 4 | 16796.5 | 32528.6 | −35902.7 |
| 9 | 18584.2 | 34274.2 | −34308.7 |

（C の k=2→3 の落差 −34261 は A の −2085・B の −5465 と桁が違う。§8.2 の `sigma_y = 0.1` による
極端に高い Y-SNR と整合するが、**これは記述的観察であり因果の証明ではない**。）

### 13.2 本監査から新しく言えること（限定）

**言える:**

- fixed generator には `rank(F) < 3`・`rank(Z) < 3`・`w = 0`・family generator の定数化といった
  **明白な低次元退化は存在しない**（§12.1）。したがって 30/30 は
  「`K_GEN = 3` が形式上だけ存在して観測生成に使われていない」という単純なケースの産物ではない。
- 3 つの latent direction はすべて X 側・Y 側の双方を通じて観測に入る（§10）。

**言えない（新たに判明した制限）:**

- **30/30 を「真の最小潜在次元 `K_0 = 3` を回収した」とは書けない。**
  B・C では `K_0` がそもそも存在せず（§12.2・§12.3）、A では存在自体が未確定（§12.4）だからである。
- 30/30 は **`K_hat = K_GEN` の一致**であって `K_hat = K_0` の一致ではない。

**変わらないこと:** `RESEARCH_MASTER.md` §9・§14 および `KNOWN_ISSUES.md`「今すぐ主張してよいこと」の
既存の書き方（Q ベース基準／Schwarz BIC ではない／一般的 true-K recovery ではない／
model-selection consistency ではない）は**すべて依然として正しく、修正を要しない**。
本監査は限定条件を 1 本追加するだけである。既存の CSV・図・記録は一切変更しない。

---

## 14. Condition-by-condition verdict

| 条件 | fixed で確認 | 根拠 | `K_GEN = K_0` への意味 |
|---|---|---|---|
| Z full rank（`rank(Z)=3`） | **CONFIRMED（生成分布として a.s.）** / realized は UNKNOWN | `[CODE]` + `[DERIVED]`（主張 Z1）。artifact 不在 | necessary structural check。十分性なし |
| F full rank（`rank(F)=3`） | **CONFIRMED（a.s.）** / realized は UNKNOWN | `[CODE]` + `[DERIVED]`（主張 F1・F2）。artifact 不在 | necessary structural check |
| row normalization が rank を保つ | **CONFIRMED** | `[DERIVED]`（`F = D_F F_raw`、`D_F` 正対角） | 退化経路を 1 本潰した |
| `w != 0`（A/B/C） | **CONFIRMED**（1.5 / 0.5 / 0.5） | `[CODE]`（`_Y_DEFAULTS`） | necessary。`w≠0` から識別性は出ない |
| Y が全 3 方向を含む | **CONFIRMED** | `[CODE]`（`w * (Z @ Z.T)`、方向別係数なし） | necessary |
| X family nondegenerate | **LIKELY**（構造として非定数） / realized 診断は UNKNOWN | `[DERIVED]` + `[UNRESOLVED]`（density 未保存） | necessary。C は `E[X_l]=1/2`（P7）で一次モーメントに情報なし |
| Y family nondegenerate | **LIKELY** / realized 診断は UNKNOWN | 同上 | necessary |
| Poisson clip が潰していない | **LIKELY（ほぼ確実に不活性）** / realized は UNKNOWN | `[DERIVED]`（決定論的上界 20.13）+ `[INFERENCE]` | clip 領域は確率非零なので unclipped 前提の P1 は不適用 |
| no obvious lower-dimensional collapse | **CONFIRMED** | `[DERIVED]`（§12.1 の 7 経路すべて否定） | 素朴な `K_0 < K_GEN` を排除 |
| M-closed（`p_* ∈ ⋃_K M_K`） | **A: UNRESOLVED / B: VIOLATED / C: VIOLATED** | `[DERIVED]`（§12.2 特異性、§12.3 Cramér） | **B・C では `K_0` が存在しない** |
| `K' < K_GEN` で same `p(X,Y)` impossible | **UNRESOLVED（全シナリオ）** | 証明なし | ultimate identifiability condition |

判定語: CONFIRMED / LIKELY / NOT ESTABLISHED / VIOLATED / UNRESOLVED。

---

## 15. What is established

1. fixed official Exp1 / K-selection の generator call chain を file+line で確定した（§4）。
   両 runner は `generate_dual_data` を `w0/w/sigma_y/var_f/uniq` すべて既定値で呼ぶ。
2. `Z` は列 z-score され、`Σ_i z_ir = 0` と `Σ_i z_ir² = 150` が厳密に成立する（§5.B）。
   whitening ではないので latent 座標間の標本相関は残る。
3. `rank(Z) = 3`・`rank(F) = 3` は生成分布の下で probability 1（§5.C・§6.2）。
4. `F` の行正規化は正の対角行列を左から掛ける操作であり **rank を保存する**（§6.2）。
5. `w ≠ 0` が A/B/C すべてで成立し、`w` はスカラーとして全 3 方向へ共通に作用する（§7）。
6. `S = z_i^T z_j` の超過尖度は canonical model で `6/K` であり、`K` は shape として入る（§7.1）。
7. Poisson clip は A の X 側・B の Y 側とも実質的に不活性（§9.1・§9.2）。
   ただし推定側 `M_K` も同じ clip を持つ（§8.1）。
8. §12.1 の 7 つの低次元縮約経路はいずれも該当しない（§12.1）。
9. **Scenario B の `p_*^B` は `M_K` のどの元とも互いに特異であり、`K_0` は存在しない**（§12.2）。
10. **Scenario C の `p_*^C` は `M_K` のどの元とも一致せず、`K_0` は存在しない**（§12.3）。
11. fixed 30/30 は一次 CSV で確認済み（§13.1）。

---

## 16. What is not established

1. **`forall K < 3, p_{K_GEN}(X,Y) notin M_K`** — どのシナリオでも証明していない。`[UNRESOLVED]`
2. **Scenario A の M-closedness**（`p_*^A ∈ ⋃_K M_K` か）`[UNRESOLVED]`
3. realized な 30 個の `Z`・`F` の rank・特異値・条件数（artifact 不在、再生成禁止）`[UNRESOLVED]`
4. realized データの family 退化診断（`y_density` / `x_count_max` 等が CSV に保存されていない）`[UNRESOLVED]`
5. Bernoulli-X（`d>1`）の一般 identifiability（既存 U1）／Bernoulli-Y（U2）／
   Gaussian-X の `Sigma` 未知（U3）／Poisson-Y（U4）。いずれも既存 `[UNRESOLVED]`
6. `Sigma` 未知対角の因子分析非一意性: `d=15, K=3` では Ledermann 型必要条件
   `(d−K)² = 144 ≥ d+K = 18` を満たすが、**必要条件にすぎず単射性は別問題**（既存 `hardened` §6.3）
7. `M_K` の閉性・誤指定下の pseudo-true `K`（既存 U11）
8. 任意 criterion の `n → ∞` consistency（既存 U6）

---

## 17. What remains for a true `K_GEN = K_0` proof

依存関係の順に並べる。上から順に埋めない限り下は意味を持たない。

1. **`K_0` を well-defined にすること。**
   現状の canonical `M_K` の下では B・C で `{K : p_* ∈ M_K} = ∅`。
   選択肢は (a) generator を canonical に合わせる（列 z-score と X の z-score を外す）、
   (b) `M_K` の定義を pipeline ごと取り直す（§12.5）、
   (c) 誤指定下の代替目標 `K°`（KL 射影次元、既存 U11）へ切り替える。
   **どれを採るかは Human Gate。**
2. **`p_* ∈ M_{K_GEN}` の構成的証明**（M-closed の肯定側）。
3. **`forall K < K_GEN` で `p_* ∉ M_K`** の証明。family ごとに必要:
   - Gaussian-Y: 既存 P2/P3 が使える（`w≠0`）が、**iid `N(0,I_K)` prior が要る**。
   - Bernoulli-Y: 既存 U2。motif 確率の族が `chi²_K` について単射かどうか。
   - Poisson-Y: 既存 U4。`R(w²)` の解析的単調性。`|w| < 1/3` の条件も要る（B の `w=0.5` は外）。
   - Poisson-X: 既存 P1 が使えるが **unclipped link が要る**（現行 `M_K` は clipped、§8.1）。
   - Bernoulli-X: 既存 U1。
   - Gaussian-X: `Sigma` 未知対角の識別条件、既存 U3。
4. **有限標本での rank 判定閾値**（既存 U7）。3 が population で解けても、
   `K_hat` が `K_0` に近づくかは別問題（既存 U6、consistency）。

---

## 18. Provisional conclusion

> fixed generator では、`Z` と `F` の rank deficiency、`w = 0`、
> family generator の定数化、Poisson clip による潰れといった
> **明白な低次元退化はいずれも確認されなかった**。
> `rank(Z) = rank(F) = 3` は生成分布の下で probability 1 で成立し、
> 行正規化は正の対角作用なので rank を保存する。
> `w` は A/B/C すべてで非零で、3 つの latent direction すべてが X 側・Y 側の双方に入る。
> したがって **`K_GEN = 3` が形式上だけ存在して観測生成に使われていない、という単純なケースではない。**
>
> 一方、`K < 3` の別 parameterization が同一の joint observational distribution `p(X,Y)` を
> 生成できないことは **証明されていない**。
>
> **さらに本監査は、それ以前の段階で前提が壊れていることを見つけた。**
> Scenario B では生成された `X` が全列で標本平均 0・標本二乗平均 1 を厳密に満たす measure-zero 集合に
> 確率 1 で乗るため、`p_*^B` は `M_K` のどの元とも互いに特異である。
> Scenario C では `Σ_{i<j} Y_ij` が厳密に Gaussian になるのに対し `M_K` 側では
> 非 Gaussian 二次形式が混ざるため、Cramér の分解定理により一致しえない。
> いずれも `{K : p_* ∈ M_K} = ∅` を意味し、**`K_0` はそもそも定義されない。**
> Scenario A については離散観測のため同様の論法が使えず、M-closedness 自体が未確定である。
>
> したがって fixed の `K_GEN = 3` は **strongly nondegenerate simulation target** とは位置づけられるが、
> **`K_0 = 3` の identifiability proof は未完であるどころか、B・C では `K_0` の定義が
> 現行の `M_K` の下で成立していない。**
> この乖離の原因は generator の後処理（`Z` の列 z-score、`X` の列 z-score）であり、
> **これは KI-021 が既に記録している G1・G3 の帰結を `K_0` の定義側まで辿ったものである。**

---

## 19. Q1-Q10 answers

**Q1. fixed generator の `Z` は `K_GEN = 3` の 3 方向を実際に持っているか？**
**YES（生成分布として probability 1）。** `[DERIVED]` 主張 Z1（§5.C）。
`Z_raw` は iid `N(0,I_3)`、列 z-score は `M Z_raw D^{-1}` で `M` の rank は 149、`D` は正対角。
rank が落ちるのは測度 0 の集合上のみ。
**ただし realized な 30 個の `Z` については artifact が存在せず数値確認していない（`[UNRESOLVED]`）。**

**Q2. `F` は `K_GEN = 3` を full rank で X 側へ伝えているか？**
**YES（a.s.）。** `[DERIVED]` 主張 F2（§6.2）。`F_raw ~ N(0,5)` の `15×3`、`d ≥ K` を満たし、
`rank(F_raw) = 3` a.s.。`eta_x = Z F^T` は 3 列すべてを使う（`[CODE]` `:293`）。
realized な特異値・条件数は `[UNRESOLVED]`（artifact 不在）。

**Q3. row normalization 等は rank を壊していないか？**
**壊していない。** `[DERIVED]` 主張 F1（§6.2）。
`F = D_F F_raw`、`D_F = diag(sqrt(0.9)/‖F_raw[j,:]‖)` は正の対角行列で可逆なので `rank(F) = rank(F_raw)`。
`Z` 側の列 z-score も `M`（centering）と正対角 `D`（scaling）の合成で、centering が rank を落とすのは測度 0。
**ただし行正規化は全行を `‖f_l‖² = 0.9` に固定するので `F` は自由パラメータではない**（§6.4、KI-021 G2）。

**Q4. Y 側 `w` は全 scenario で nonzero か？**
**YES。** `[CODE]` A: `w = 1.5`（`_Y_DEFAULTS["bernoulli"]`）、B: `0.5`、C: `0.5`。
両 runner は `w_true` を明示指定しないので既定値がそのまま使われる（§4.2）。

**Q5. Y は全 3 latent directions を生成式に含むか？**
**YES。** `[CODE]` `:310` `eta_y = w0_true + w_true * (Z @ Z.T)`、
`(ZZ^T)_{ij} = Σ_{r=1}^{3} z_ir z_jr`。
**`w` はスカラーであり方向別 parameter は存在しない**（全方向に共通に作用する）。

**Q6. Poisson clip 等によって latent predictor が明らかに低次元・定数へ潰れていないか？**
**潰れていない（ほぼ確実に不活性）。** `[DERIVED]` + `[INFERENCE]`（§9）。
A の X 側は `|eta_x| ≤ 20.13` という決定論的上界を持ち、`+10` 到達には 1 ノードが
latent energy の 24.7% を占める必要がある（`P ≈ 6e−24`）。
B の Y 側は既存推定で `P(eta_y > 10) ≈ 5e−08`、期待発動数 ≈ 0.006。
**realized データでの直接確認は `y_density` / `x_count_max` が CSV に保存されていないため `[UNRESOLVED]`。**
なお clip は生成側・推定側の双方に同一に入っているので誤指定ではないが、
その代わり **unclipped canonical link を前提とする P1 はこの `M_K` に適用できない**（§9.3）。

**Q7. fixed generator に `K_0 < K_GEN` となる明白な構造的退化は見つかったか？**
**見つからなかった。** `[DERIVED]`（§12.1）。
rank 低下・`w=0`・恒等的に 0 の座標・正規化による方向消失・未使用列・
K 非依存十分統計量のいずれも該当しない。
**ただし「`K_0 < K_GEN` が無い」ことと「`K_0 = K_GEN` である」ことは別である。**
実際に見つかったのはより深刻な問題で、**B・C では `K_0` がそもそも存在しない**（§12.2・§12.3）。

**Q8. 逆に、`K_GEN = K_0` を数学的に証明できたか？**
**できていない。** `[UNRESOLVED]`。
それどころか、Scenario B・C については **`K_0` の定義が現行 `M_K` の下で成立しない**ことを
本監査が導出した（§12.2 特異性論法、§12.3 Cramér 論法）。Scenario A は M-closedness 自体が未確定。

**Q9. 証明できないなら、何が最後に足りないか？**
4 段階（§17）。要点だけ言えば:
(1) **`K_0` を well-defined にする決定**（generator を canonical へ寄せる／`M_K` の定義を取り直す／
代替目標 `K°` へ切り替える）— これは Human Gate。
(2) `p_* ∈ M_{K_GEN}` の構成的証明。
(3) `forall K < K_GEN, p_* ∉ M_K` の family 別証明。Gaussian-Y は既存 P2/P3 が使えるが
**iid `N(0,I_K)` prior が要る**（列 z-score が阻む）、Poisson-X は **unclipped link が要る**、
Bernoulli-X / Bernoulli-Y / Poisson-Y は既存 U1・U2・U4 のまま。
(4) 有限標本の rank 閾値（U7）と criterion の consistency（U6）。

**Q10. fixed 30/30 を今後どの表現で記述するのが最も正確か？**

現行の canonical 表現（`RESEARCH_MASTER.md` §14）に、**generator 側の限定条件を 1 本追加する**のが正確である。

> 「明示した fixed 系列の人工設定（`n = 150`, `d = 15`, 生成時の潜在次元 `K_GEN = 3`、
> シナリオ A/B/C 各 10 trial）において、歴史的に `BIC` と呼ばれてきた Q ベース基準が、
> 候補 `K = 1,…,9` から `K = 3` を 30/30 trial で選択した。
> これは **`K_hat = K_GEN` の一致**であって、観測分布の最小潜在次元 `K_0` の回収ではない。
> 使用した generator は `Z` を列 z-score し（Scenario B ではさらに `X` を列 z-score する）ため、
> **`K_0` を定義する M-closed 前提は Scenario B・C では成立せず、Scenario A では未確定である。**
> Schwarz BIC の妥当性・一般的な true-K recovery・model-selection consistency のいずれも意味しない。」

**避けるべき表現:** 「真の潜在次元を回収した」「`K_0 = 3` を正しく選んだ」「true-K recovery」
「BIC で正しい `K` を選べる」。
**維持すべき既存表現:** 「Q ベース基準」（Schwarz BIC と呼ばない、KI-010）、
関数名 `calc_bic_dual`・CSV 列名 `BIC`・過去結果の呼称は変更しない。

---

## 付録: source priority と claimability

本監査は root `CLAUDE.md` §4 の source priority に従い、以下の順で証拠を採った。

1. 一次コード（`data_generator_expfam.py`, `run_fixed_official_exp1_bic_*.py`,
   `model_dual_expfam.py`, `model_expfam.py`, `model_dual_expfam_fixed.py`,
   `reproduction/src/data_generator.py`, `reproduction/src/model.py`, `utils_expfam.py`）
   と一次結果 CSV（`fixed_official/exp1_k9/*.csv`）。
2. canonical docs（`CLAUDE.md`, `KNOWN_ISSUES.md` KI-010/KI-021, `EXPERIMENT_REGISTRY.md:53`,
   `RESEARCH_MASTER.md` §9/§14）。
3. 凍結報告 `reports/identifiability/true_k_identifiability_hardened_20260904.md`
   — **既存の命題 P1/P2/P3/P7/O1 と未解決事項 U1-U12 の参照にのみ使用**。
   数値主張はこの文書からは引かず、`[INFERENCE]` として明示した箇所（§9.2・§9.4）を除き
   すべて本監査で独立に導出した。

### claimability / evidence status（2 軸）

| 主張 | claimability | evidence status | 一次証拠 |
|---|---|---|---|
| fixed 系列で候補 K=1..9 から K=3 を 30/30 trial 選択 | **QUALIFIED ONLY**（Q ベース基準・Schwarz BIC ではない・一般的 true-K recovery ではない・consistency ではない） | **VERIFIED** | `fixed_official/exp1_k9/fixed_exp1_bic_k1to9_bestk_by_trial.csv`（30 行） |
| fixed generator の `rank(Z) = rank(F) = K_GEN` | ALLOWED（「生成分布の下で a.s.」を付す） | **DERIVED**（§5.C・§6.2、realized は未確認） | `data_generator_expfam.py:282-290` |
| A/B/C すべてで `w ≠ 0` | ALLOWED | **VERIFIED** | `data_generator_expfam.py:216-220, 272-276` |
| Scenario B・C で `K_0` が存在しない | ALLOWED（本監査の導出であることを明示） | **DERIVED**（§12.2・§12.3） | `data_generator_expfam.py:283, 298, 322-323` |
| `K_GEN = K_0` | **NOT ALLOWED**（`KNOWN_ISSUES.md`「人工データの `K_TRUE` が真の潜在次元 `K*` である」） | **UNRESOLVED** | — |
| fixed 30/30 が true-K recovery である | **NOT ALLOWED**（`RESEARCH_MASTER.md` §14、KI-010） | **UNRESOLVED** | — |

**implementation lineage:** 本監査で引用した数値はすべて **fixed 系列**（`DualExpFamLSMFixed`、0.5 なし）。
old 0.5 系列・`experimental/` prototype の数値は一切混ぜていない（KI-002）。
