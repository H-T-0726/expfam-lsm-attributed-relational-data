# Phase 8a: K_TRUE 可変 held-out K-selection robustness — design (pre-registration)

**日付:** 2026-09-01
**Issue:** #47
**branch:** `design/47-k-true-robustness`
**base main SHA:** `fdef75cbb7e87b472c9c6eae0508479f5a391803`
**種別:** DESIGN ONLY（EM fit 0 件・実験結果なし・canonical docs 変更なし）
**改訂:** 2026-09-01 — Codex independent review（`FIX_DESIGN_BEFORE_HUMAN_GATE` → `APPROVE_DESIGN_FOR_HUMAN_GATE`）
の findings HIGH-01 / HIGH-02 / HIGH-03 / MEDIUM-01 / MEDIUM-02 / LOW-01 / F-01 を反映

**Human Gate frozen:** 2026-09-01
**Decision source:** GitHub Issue #47 — Human Gate Decision comment（2026-09-01, by H-T-0726）
**Decision type:** **HUMAN GATE DECISION**（AI recommendation ではない）

> 本レポートは **結果を見る前に** 何を変え・何を固定し・何を成功/失敗と呼ばないかを凍結するための
> pre-registration である。
> **比較可能性判断の primary evidence は解析式であり、シミュレーション由来の数値ではない。**
>
> **§16 の H1–H4 は 2026-09-01 の Human Gate により確定済みである（PENDING ではない）。**
> 検討されたが採用されなかった代替案は `ALTERNATIVES CONSIDERED (NOT SELECTED)` として
> 履歴のために残してある。**current config は §9 の frozen table を読むこと。**

---

## 1. Executive summary

Phase 7e（Issue #43 / PR #44）の held-out plug-in K-selection protocol を
`K_TRUE ∈ {1,2,4,5}` へ拡張する設計監査を行った。結論は次のとおり。

| 監査項目 | 結果 |
|---|---|
| K_TRUE の parameterization | 追跡完了。generator 側は `generate_dual_data(k=...)` の 1 引数、harness 側は module 定数 `K_TRUE=3` の 6 参照。**scientific semantics を変えずに変数化できる** |
| Phase 7e K_TRUE=3 anchor の再利用 | **REUSABLE_ANCHOR**（§3.3） |
| K_TRUE=1 boundary | **blocker なし・特殊 hack 不要**（§7） |
| seed 設計 | Phase 7e と衝突しない deterministic block を提示（§10） |
| fit budget | **336 unique new fits + 42 共通 anchor = 378 unique total**（§11） |
| **signal comparability** | fixed-`w` 単独では不成立（`Var(η^Y) = w² K c_n` が `K_TRUE` に比例、§5・§6）。**Human Gate により A（fixed-`w`）と B（variance-matched）を別 estimand として事前登録することで解決** |

### Human Gate 決定（2026-09-01・Issue #47）

| gate | 決定 | 内容 |
|---|---|---|
| **H1** | **`A+B`** | Option A（`w = 1.5` fixed）と Option B（`w_K = 1.5·√(3/K_TRUE)`）を**両方**事前登録して実行する。**結果を見てどちらかを drop してはならない** |
| **H2** | **`CRN`** | A/B 間で **data-generation RNG** と **model-initialization RNG** を common-random-number design で対応させる。**H2 は pair-index mask を管理しない** |
| **H3** | **`H3-a`** | **A = primary / B = pre-registered sensitivity**。role は implementation 前に freeze 済み、role switching 禁止 |
| **H4** | **`S-c`** | Phase 7e `K_TRUE=3` の保存済み pair-index mask を reference とし、新規 `K_TRUE ∈ {1,2,4,5}` にも同一 mask を適用。**`K_TRUE=3` は再実行しない** |

**decision gate は `A: IMPLEMENT_K_TRUE_ROBUSTNESS_HARNESS_NEXT` へ更新（§15）。**

> **重要: `A` は full experiment の実行許可ではない。**
> 次に許可されるのは **implementation と zero-EM validation のみ**であり、
> **336 fits の full run はまだ許可されていない**（§15.1）。

Codex independent review の科学的推奨であった
**R3: `PRE_REGISTER_A_AND_B_AS_SEPARATE_ESTIMANDS`**（§8.6）は、
**独立レビューの推奨として記録され、その後 2026-09-01 の Human Gate により
H1=`A+B` として正式に採用された。** 両者は別個の記録である。

---

## 2. Research question

> Phase 7e で凍結した held-out plug-in K-selection protocol は、generator の `K_TRUE` を変えたとき、
> 有限標本でどのような selected-K pattern を示すか。

これは 2026-07-14 ゼミ指摘のうち
「真のモデルが 1 次元など別の潜在次元だった場合にも、選択基準がそれを捉えられるのか」への
**finite-sample empirical response** である。

`n → ∞` の consistency は本 Issue の対象外（§13）。

---

## 3. Phase 7e compatibility matrix

### 3.1 一次確認した source

| # | path | 役割 | 確認内容 |
|---|---|---|---|
| 1 | `tools/research_audit/run_heldout_k_selection_pilot.py` | Phase 7e harness | 定数・manifest・seed・score・selector・fit boundary |
| 2 | `expfam/src/data_generator_expfam.py` | generator | `generate_dual_data` の Z / F / X / Y 生成式 |
| 3 | `reproduction/src/data_generator.py` | `normalize_zscore` | 列 z-score（ddof=0） |
| 4 | `expfam/src/experimental/eval_utils.py` | `make_pair_split` / `calc_bic_exp` | split の K 非依存性・`num_params` |
| 5 | `expfam/src/experimental/em_runner.py` | `run_em_experimental` / `build_model` | informed init・retry・`scale_Z` |
| 6 | `reproduction/src/model.py` | `initialize_params` / `scale_Z` | 推定側 Z scale 規約 |
| 7 | `tools/research_audit/audit_heldout_full_pilot.py` | 独立 audit | 42 行・K_TRUE=3 の hard-code 範囲 |
| 8 | `tools/research_audit/test_heldout_k_selection_pilot.py` | 120 tests | 42 / 3 replicate の固定 assertion 範囲 |
| 9 | `expfam/results/k_selection/heldout_full_pilot_20260824/` | 一次成果物 | manifest / fit_results / selection / aggregate / runinfo |
| 10 | `reports/k_selection_theory/heldout_k_selection_full_pilot_report_20260824.md` | frozen report | 固定条件表・interpretation boundary |
| 11 | `reports/k_selection_theory/heldout_k_selection_full_pilot_provenance_addendum_20260831.md` | forward correction | exactly-once の表現限界 |
| 12 | Issue #43 / PR #44 | 実行承認記録 | RUN_CODE_SHA・review verdict |

### 3.2 1 対 1 比較（Phase 7e 実測値 vs 本 design 提案）

| 因子 | Phase 7e（`runinfo.json` 実測） | Phase 8a 提案 | 一致 |
|---|---|---|:--:|
| model lineage | `DualExpFamLSMConsistent`（**本文採用不可 prototype**） | 同一 | 同一 |
| `family_x` / `family_y` | `poisson` / `bernoulli` | 同一 | 同一 |
| `n` / `d` / `L` / `num_iter` | 75 / 15 / 5 / 8 | 同一 | 同一 |
| `numerics_mode` | `consistent` | 同一 | 同一 |
| `test_ratio` | 0.20（expected_test_pairs = 555） | 0.20 | 同一 |
| candidate K | `{1,…,7}` | `{1,…,7}` | 同一 |
| starts / K | `{1,2}` | `{1,2}` | 同一 |
| dataset replicates | `{1,2,3}` | `{1,2,3}` | 同一 |
| primary score | held-out Bernoulli raw-eta mean log score | 同一 | 同一 |
| selector | 2-start mean → max → tie ≤1e-12 → smallest K | 同一 | 同一 |
| `w0_true` | `-1.0`（`_Y_DEFAULTS["bernoulli"]`） | `-1.0` | 同一 |
| `var_f` / `uniq` | `5.0` / `0.1` | 同一 | 同一 |
| split mask の構成 | `make_pair_split(75, 0.2, 42000+replicate)` | **H4=S-c により同一**（新規 `K_TRUE` にも anchor mask を適用、§10.4） | **同一** |
| `w_true` | `1.5`（同上） | **H1=A+B により 2 estimand**: A は `1.5` 固定（K_TRUE=3 で Phase 7e と一致）、B は `1.5·√(3/K_TRUE)`（K_TRUE=3 で `1.5`） | **K_TRUE=3 では同一** |
| `K_TRUE` | 3 | `{1,2,4,5}` + anchor 3 | 意図的操作 |

**Human Gate（2026-09-01）により未決定項目は解消した。** `w_true` は estimand ごとに確定し、
split mask は H4=S-c により Phase 7e anchor と同一である。
**`K_TRUE=3` では A・B とも `w=1.5`・anchor mask となり、Phase 7e と完全に一致する。**

### 3.3 anchor 再利用の根拠

`fit_results.csv` に記録された replicate 単位の 4 hash
（`x_hash` / `training_y_hash` / `train_mask_hash` / `test_mask_hash`）を、
現 HEAD 上で `generate_dual_data` + `make_pair_split` + `make_training_y_values` から
再計算して照合したところ、**3 replicate すべてで committed 値と一致した**。

これは新しい科学的測定ではなく、**repository に committed された hash を
決定的に再導出しただけの検証**であり、誰でも同じ手順で再現できる。

code provenance:

- `runinfo.json` の `run_code_sha = b9311e64a7b36c0a8a9704fff0ee7b38efe36a8a` は **git commit SHA** であり、
  `git merge-base --is-ancestor b9311e6 HEAD` → 真。
- `git diff b9311e6 HEAD --` に対し
  `run_heldout_k_selection_pilot.py` / `data_generator_expfam.py` / `eval_utils.py` /
  `reproduction/src/data_generator.py` の 4 file はすべて **差分空**。
- `score_config_hash` を現 HEAD で再計算した値は `fit_results.csv` の記録値と一致
  （score protocol の同一性）。

→ **Phase 7e K_TRUE=3 の入力データ・split・score 契約・実行コードは現 HEAD で完全に再現する。**

**verdict: `REUSABLE_ANCHOR`**（案 B を採る場合も `w_3 = 1.5` なので成立。§8.3）

**execution date と code SHA が新規 run と異なることは、それ自体では scientific confound ではない。**
必要なのは provenance の分離であり、それは §11 の lineage 列で担保する。

**H4 の S-c（§10.4）を採る場合も、`K_TRUE=3` を再実行しない。**
Phase 7e anchor の split seed・保存済み `train_mask_hash` / `test_mask_hash`・
関連コード semantics を **reference として読むだけ**で、
新規 `K_TRUE` の mask をその reference へ揃える。

---

## 4. K_TRUE dependency map

`K_TRUE=3` は **generator 引数 1 箇所** と **harness module 定数 1 箇所（参照 6 箇所）** にのみ存在する。

| stage | file / function | K_TRUE dependency | 変数化可能? | scientific semantics 変更? |
|---|---|---|:--:|---|
| module 定数 | `run_heldout_k_selection_pilot.py:39` `K_TRUE = 3` | 定義元。L.1973 / 2079 / 2097 / 2148 / 2639 / 2647 / 3278 が参照 | 可 | なし（値の供給元のみ） |
| Z 生成 | `generate_dual_data` `Z = rng.normal(...,(n,k))` → `normalize_zscore(axis=0)` | shape `(n,K_TRUE)` | 可 | **あり — §5.1**（`Σ_i‖z_i‖²/n = K_TRUE`） |
| F 生成 | 同 `F = rng.normal(0,√var_f,(d,k))` → 行正規化 `‖f_l‖=√(1−uniq)` | shape `(d,K_TRUE)` | 可 | 行ノルムは K 不変。**mean loading energy は K 依存 — §5.4** |
| X の η | 同 `eta_x_full = Z @ F.T` | `(n,d)`、K に依らない shape | 可 | 分散水準は K 不変。**分布形状の完全不変は主張しない — §6.3** |
| X 生成 | 同 Poisson `rng.poisson(exp(clip(η,-20,10)))` | 上記経由のみ | 可 | 同上 |
| Y の η | 同 `eta_y = w0 + w * (Z @ Z.T)` | `Var(z_i^Tz_j) = K_TRUE·c_n` | 可 | **あり — §6.2（本 design の blocker）** |
| Y 生成 | 同 `rng.binomial(1, σ(η))` upper→対称化 | 上記経由のみ | 可 | あり（density / entropy / 形状が K 依存） |
| split / masks | `eval_utils.make_pair_split(n, test_ratio, seed)` | **K を引数に取らない** | — | **なし**（K 完全不変） |
| train/test 構成 | `make_training_y_values` / `make_score_only_target` | Y の値のみ経由 | — | なし |
| fit（候補 K） | `_full_fit_config` → `FrozenFitConfig.k_est = row.k` | `k_est` は candidate K であり K_TRUE と独立 | — | なし |
| 推定側 Z scale | `model.scale_Z` → `mean(Ẑ²)=1`（`E‖ẑ_i‖²=k_est`） | `k_est` 依存 | — | なし（生成側規約と整合、§6.4） |
| scoring | `heldout_raw_eta_pairs` + `heldout_bernoulli_mean_log_score` | **K_TRUE を参照しない**（`w0, w, Ẑ` のみ） | — | なし |
| selector | `select_k_from_two_starts` | **K_TRUE を参照しない** | — | なし |
| 集計 | `_aggregate_across_replicates`（L.2639 / 2647） | `counts.get(K_TRUE,0)` / `true_k=K_TRUE` | 可 | なし（記述ラベル生成のみ） |
| artifact schema | 各 CSV / `runinfo.json` | **`K_TRUE` 列が存在しない** | 要拡張 | なし（列追加は記録の拡張） |

**「定数を引数化すれば済む」わけではない箇所は 2 つ:**

1. **artifact schema** — 現行 CSV は `K_TRUE` 列を持たない（§11・implementation plan §4）。
2. **generator の signal scale** — §5・§6。これは引数化では解けない。

---

## 5. Generator mathematical audit（**primary evidence**）

CLAUDE.md §1 の確定生成モデル:

```
z_i  ~ N(0, I_k)
y_ij ~ ExpFam_Y( η_ij^Y = w_0^Y + w^Y z_i^T z_j )   i < j
x_il ~ ExpFam_X( η_il^X = f_l^T z_i )
```

`generate_dual_data` の実装はこれに**有限標本正規化**を加えている。
以下はすべて generator ensemble 上の**解析的帰結**であり、シミュレーション由来ではない。

### 5.1 Z（列 z-score）

```
Z~ ~ N(0,1)^{n×k}
z_{ic} = (z~_{ic} − mean_i z~_{ic}) / sd_i z~_{ic}      (ddof = 0)
```

ddof=0 の列 z-score により、**各実現において厳密に**

```
Σ_i z_{ic} = 0            (∀c)
(1/n) Σ_i z_{ic}² = 1     (∀c)
⇒ (1/n) Σ_i ‖z_i‖² = k                                  … (5.1)  厳密・確率的でない
```

**重要:** この制約により、**z-score 後の `Z` の列は厳密には Gaussian ではない。**
各列は「和 0・二乗和 n」の超球面（次元 `n−2`）上の一様分布に従う。
以降の高次モーメントはこの分布から導出する（Gaussian を仮定しない）。

### 5.2 内積の 1 次モーメント（厳密）

`m = n − 1` とおく。列中心化から、各実現で
`Σ_{i≠j} z_{ic} z_{jc} = (Σ_i z_{ic})² − Σ_i z_{ic}² = −n`。
順序対は `n(n−1)` 組で交換可能なので、ensemble 上

```
E[ z_{ic} z_{jc} ] = − 1/m        (i ≠ j)
⇒ E[ z_i^T z_j ] = − K / (n−1)                          … (5.2)  厳密
```

### 5.3 内積の分散（**厳密・有限 n 補正あり**）

`P = I − (1/n)11ᵀ`、`g ~ N(0,I_n)`、`v = Pg ~ N(0,P)` とすると
`z_{·c} = √n · v/‖v‖` と書ける。方向 `v/‖v‖` と半径 `‖v‖` は独立なので

```
E[z_{ic}² z_{jc}²] = n² · E[v_i² v_j²] / E[‖v‖⁴]
```

`Var(v_i) = m/n`、`Cov(v_i,v_j) = −1/n`、Gaussian の 4 次モーメントより
`E[v_i²v_j²] = (m/n)² + 2(1/n)² = (m²+2)/n²`。
`‖v‖² ~ χ²_m` より `E[‖v‖⁴] = m(m+2)`。よって

```
E[ z_{ic}² z_{jc}² ] = (m² + 2) / ( m (m+2) )            … (5.3a)  厳密
Var( z_{ic} z_{jc} ) = (m² + 2)/( m(m+2) ) − 1/m²        … (5.3b)  厳密
```

列は互いに独立なので

```
Var( z_i^T z_j ) = K · c_n ,
   c_n := (m² + 2)/( m(m+2) ) − 1/m² ,   m = n − 1      … (5.3)  厳密
```

`n = 75`（`m = 74`）では `c_n = 25331/26011 ≈ 0.973857`。

**したがって無修飾に `Var(z_i^T z_j) = K` と書かない。**
大標本近似を使う場合は `≈ K` または `K(1 + O(1/n))` と明記する。

### 5.4 Y 側自然パラメータ

```
η_ij^Y = w_0 + w · z_i^T z_j
E[η^Y]   = w_0 − w K/(n−1)
Var(η^Y) = w² · K · c_n                                 … (5.4)  厳密
sd(η^Y)  = |w| √(K c_n)
```

**`w` を固定すると `Var(η^Y) ∝ K_TRUE`。これが本 design の中心的発見である。**

さらに `z_i^T z_j` は `K` 個の独立項の和なので、**分散だけでなく分布形状（尖度）も `K` に依存する**
（`K=1` では積型で裾が重く、`K` の増大とともに中心極限的に正規へ近づく）。
`σ(·)` は非線形なので、Y の density と条件付きエントロピーは
分散だけでなく高次モーメントにも依存する。

### 5.5 X 側自然パラメータ

行正規化 `‖f_l‖ = √(1−uniq)` は **k に依らず全ての l に課される**。よって

```
‖F‖_F² = d (1 − uniq)                                   … (5.5)  厳密・k 不変
```

`F` を条件づけたとき、`z_i` の各座標が平均 0・分散 1・座標間無相関であることから

```
Var( f_l^T z_i | F ) = ‖f_l‖² = 1 − uniq                … (5.6)  分散水準の主張・k 不変
```

**(5.6) は分散水準の主張であって、`η^X` の分布が `k` に対して厳密不変であることを意味しない。**
`z_i` は厳密 Gaussian ではなく（§5.1）、`f_l^T z_i` の高次モーメント・分布形状には
`k` 依存が残り得る。

Poisson-X の平均については、`f_l^T z_i` を Gaussian と近似すれば
`E[λ_il] ≈ exp((1−uniq)/2) = exp(0.45)` となる。
**これは Gaussian approximation であり、厳密値ではない。**

一方 `F^T F` は k×k で `tr(F^T F) = d(1−uniq)` 固定なので

```
mean loading energy := tr(F^T F)/K = d(1−uniq)/K  ∝ 1/K … (5.7)
```

**(5.7) は loading energy（`F` の二次形式のトレース平均）についての主張であり、
Fisher information についての主張ではない。** Poisson-X の条件付き Fisher information は

```
I_{X,i}(z_i) = F^T diag( exp(F z_i) ) F                 … (5.8)
```

であり、`diag(exp(F z_i))` が `F` と `z_i` に依存するため、
**`I_{X,i}` が `1/K` でスケールすることは現時点で証明されていない（未解決）。**

---

## 6. Signal comparability analysis

### 6.1 判断の根拠は解析式に一本化する

比較可能性の判断は **§5 の解析式のみ**に基づく。すなわち

```
(5.1)  Σ_i‖z_i‖²/n = K                （厳密）
(5.2)  E[z_i^T z_j] = −K/(n−1)        （厳密）
(5.3)  Var(z_i^T z_j) = K c_n         （厳密）
(5.4)  Var(η^Y) = w² K c_n            （厳密）
(5.5)  ‖F‖_F² = d(1−uniq)             （厳密・k 不変）
(5.7)  tr(F^TF)/K = d(1−uniq)/K       （厳密、loading energy）
案 B:  w_K² K = const                 （代数的に一定）
```

> **設計時に untracked な scratch diagnostic（repository 外・tracked script なし・
> seed manifest なし・machine-readable output なし）を実行し、
> 上記解析式と定性的に整合することを確認した。
> ただしこれは primary evidence ではなく、数値効果量として引用しない。**
> Issue #47 は design only であり、この diagnostic を正式 artifact 化しない。

### 6.2 Y 側 — **K 不変ではない**

(5.4) より、`w` を固定すると `Var(η^Y) = w² K c_n` は `K_TRUE` に比例して増大する。
その帰結として、`K_TRUE` を変えると

- Y の edge density が変化する（`E[σ(η^Y)]` は `η^Y` の分布に依存し、`w_0 < 0` では
  分散の増大とともに増加する）
- 1 dyad あたりの条件付きエントロピー `H(y_ij | Z)` が変化する
- `z_i^T z_j` の**分布形状（高次モーメント）**も `K` に依存する（§5.4）

すなわち **oracle 水準の irreducible な score 水準そのものが `K_TRUE` によって変化する**
（真の `(Z, w_0, w)` を与えても達成できない誤差の下限が水準ごとに異なる）。

> **安全な結論（本 design の公式表現）:**
> 固定 `w` では `K_TRUE` に応じて oracle 水準の irreducible score level も変化するため、
> **task difficulty / response uncertainty が `K_TRUE` 間で同一ではない。**
> ただし、その K 間 score-level 差を candidate-K selection margin の大きさと
> **直接比較して confounding magnitude を定量化しない。**

**理由（明示的に記録する）:** K_TRUE 間の oracle absolute score 水準差と、
同一 dataset・同一 K_TRUE 内の candidate-K 間 selection margin は
**異なる統計量であり、比を取って「何倍の交絡」と解釈することは不適切である。**
generator comparability failure の根拠は
**`Var(η^Y) ∝ K_TRUE` と Y density / entropy / 分布形状の変化そのもの**であって、
そのような倍率比較ではない。

### 6.3 X 側 — 分散水準は不変、loading energy は K 依存

- `‖F‖_F² = d(1−uniq)` は **厳密に k 不変**（5.5）。
- `Var(f_l^T z_i | F) = 1 − uniq` という **分散水準の主張は k 不変**（5.6）。
- **ただし「X の周辺分布が `K_TRUE` に厳密不変」とは主張しない。**
  z-score 後の `Z` は厳密 Gaussian ではなく（§5.1）、
  高次モーメント・分布形状には `k` 依存が残り得る。
  Poisson 平均 `exp(0.45)` は **Gaussian approximation** である。
- **mean loading energy `tr(F^TF)/K = d(1−uniq)/K` は `1/K` で低下する**（5.7）。
- **これが Poisson-X の Fisher information 全体の `1/K` scaling を意味するかは未解決**（5.8）。

この項目の位置づけは §14 の C3 で整理する。

### 6.4 推定側の scale 規約は整合している

`reproduction/src/model.py:scale_Z` は `mean(Ẑ²)=1`、すなわち `E‖ẑ_i‖² = k_est` に正規化する。
これは generator の (5.1) と**同じ規約**であり、`k_est = k_true` のとき
生成側と推定側の潜在スケールが一致する。ここに K 依存の不整合はない。

### 6.5 verdict

> **固定 `w` のもとで `K_TRUE` を変えると、変わるのは「潜在次元」だけではない。
> `Var(η^Y) = w² K c_n` により Y の interaction variance が同時に変わり、
> density・entropy・分布形状も変化する。**

したがって現行 frozen generator semantics（fixed `w`）上の K_TRUE sweep は
**「潜在次元だけを変えた比較」ではない。**

とりわけ `K_TRUE=1` セルは「最も低次元」であると同時に「最も Y interaction variance が小さい」
セルになる。ここで under-selection が観測されても、
**「選択基準が低次元を捉えられない」のか「Y の応答不確実性が大きい水準だから」なのかを
分離できない。**

**SIGNAL COMPARABILITY: FAIL（fixed-`w` semantics のまま sweep する場合）**

---

## 7. K_TRUE = 1 boundary analysis

**Codex verdict: NO BLOCKER / hack 不要。** 以下は generator コードと採用式から導かれる
構造的事実であり、性能の優劣を主張するものではない。

| 項目 | K_TRUE=1 での状態 | 根拠 |
|---|---|---|
| `F.shape` / rank | `(d,1)` = (15,1) / rank 1 | 行正規化後、全行が非零 |
| 行ノルム | 全行 `√(1−uniq) = √0.9` | 生成コードの行正規化（k 共通の規則） |
| `F` の値域 | **`±√0.9` の 2 値のみ**（符号のみ変化） | k=1 では方向が符号に退化 |
| `F^T F` | スカラー `‖F‖_F² = d(1−uniq) = 13.5` | (5.5) |
| `Z` の規約 | 列分散 1、`Σ_i z_i²/n = 1` | (5.1) |
| 回転群 | `O(1) = {±1}`（離散）、`k(k−1)/2 = 0` | 連続回転自由度なし |
| sign ambiguity | `z_i^T z_j` は `Z → −Z` に対し**厳密に不変** | 内積の双線形性 |
| score | 上記より **符号反転に完全不変** → 補正不要 | `η̂ = ŵ_0 + ŵ ẑ_i^Tẑ_j` |
| `num_params = k·d − k(k−1)/2` | `15 − 0 = 15`（well-defined） | `calc_bic_exp`（**K 選択には未使用**） |

### 7.1 k=1 特有の構造（**blocker ではないが報告する**）

`k=1` では行正規化により `F` の全 15 行が `±√0.9` となり、
**X の全列が符号を除いて同一の信号 `η_il^X = ±√0.9 · z_i` を運ぶ。**
`k≥2` では行が球面上の異なる方向を向くのに対し、`k=1` では方向が符号に退化する。

- **規則そのものは k 共通**（「全行のノルムを `√(1−uniq)` に揃える」）であり、
  k=1 で規則を変えているわけではない。したがって **K_TRUE=1 専用 hack は不要**。
- ただしこれは **K_TRUE=1 に固有の構造**であり、interpretation では
  他水準と同列に扱わず stratify する（§14 C4）。

### 7.2 Procrustes / parameter count は本 protocol に発生しない

Phase 7e の `fit_results.csv` の列を一次確認した結果、
**Procrustes・latent-space metric・parameter count は 1 列も記録されておらず、
selection にも一切使われていない**（列は score / `Q_strict` / 失敗フラグ / hash 群のみ）。
K 選択には BIC も `Q_strict` も使わない（frozen score config、`predict_mu_y=False`）。
したがって `k(k−1)/2` 補正の問題は本 protocol には**発生しない**。

### 7.3 candidate `k_est = 1` の実績

Phase 7e の 42 fits には `k_est=1` の 6 fit が含まれ、`fit_results.csv` 上で
いずれも `retry=0 / warnings=0 / q_failure=False / nan_occurred=False /
finite_state=True / fit_status=clean`。**推定側 k=1 経路は clean に走った実績がある。**

### 7.4 verdict

**`K_TRUE = 1` は特殊 hack なしで定義可能。blocker ではない。**
ただし §7.1 の構造は K_TRUE=1 固有であり、集計時に stratify する。

**注記:** 条件数などの数値的性質を根拠に「k=1 が他次元より有利／最良」と述べない。
異なる `K_TRUE` は異なる推定問題であり、そのような異次元間の性能比較は本 design では行わない。

---

## 8. Rescaling alternatives と estimand の定義

**結果を見て決めてはならない。以下は design 段階の理論的整理であり、本 Issue では採用を確定しない。**

### 8.1 Option A — fixed `w`（Phase 7e generator semantics をそのまま維持）

```
w = 1.5  (全 K_TRUE で固定)
```

**estimand:**

> original Phase 7e generator family（`w_0=−1, w=1.5, n=75, d=15, var_f=5, uniq=0.1`）において
> `K_TRUE` を変化させたときの、**end-to-end finite-sample selector behavior**。

**必須 limitation（report に必ず併記する）:**

- **dimension effect alone ではない。**
- Y interaction variance `Var(η^Y) = w² K c_n` も `K_TRUE` 依存。
- Y の density / entropy / 分布形状（difficulty）も `K_TRUE` 依存。

| 観点 | 内容 |
|---|---|
| Phase 7e anchor 互換 | **完全互換**（`w_true` を渡さない = Phase 7e と同一呼び出し） |
| generator meaning | 「相互作用係数 `w` が固定された系」。`w` を系の物理的性質と見なす立場では自然 |
| pros | 実装変更が最小。anchor が無条件に使える。generator を一切触らない |
| cons | 「1 次元の真値を捉えられるか」というゼミ指摘に対し、次元効果と応答不確実性が分離されない |

### 8.2 Option B — variance-matched（`w_K = w_ref √(k_ref/K)`）

(5.4) より `Var(η^Y) = w² K c_n` なので、`w_K = w_ref √(k_ref/K)` とすれば
`w_K² K = w_ref² k_ref` が**代数的に一定**となり、`Var(η^Y) = w_ref² k_ref c_n` が K 不変になる。
`w_ref = 1.5`, `k_ref = 3`。

```
w_K = 1.5 · sqrt(3 / K_TRUE)
```

**estimand:**

> Y natural-parameter variance を `K=3` reference へ **ensemble level で match** した
> generator family における、**finite-sample selector behavior**。

**必須 limitation（report に必ず併記する）:**

- **variance matching only**（一致するのは 2 次モーメントのみ）。
- `z_i^T z_j` の **higher moments は `K_TRUE` 依存**（§5.4）。
- Bernoulli の非線形リンクを通した後の **density / entropy は完全には一致しない**。
- **dimension effect alone ではない。**

| 観点 | 内容 |
|---|---|
| Phase 7e anchor 互換 | **`k_ref=3` により `w_3 = 1.5` となり K_TRUE=3 は Phase 7e と同一（§8.3）** |
| generator meaning | 「`w` ではなく `Var(η^Y)` が系の固定量」。次元を上げても各 dyad の interaction variance を保つ立場 |
| pros | Y 側の支配的な K 依存（分散）を除去。anchor を壊さない |
| cons | `w_true` が `K_TRUE` の関数になる（`w` が定数でなくなる）。generator semantics の変更であり Human Gate |

### 8.3 Option B が anchor を壊さない理由（解析的根拠）

`generate_dual_data` のコードを読むと、RNG の消費順序は `Z → F → X → Y` であり、
`w_true` は `Z`・`F`・`X` の draw に一切影響しない。`w_true` は
`eta_y = w0 + w * (Z @ Z.T)` を通じて `prob` にのみ入る。

`k_ref = 3` のとき `w_3 = 1.5 · √(3/3) = 1.5` は既定値 `_Y_DEFAULTS["bernoulli"]["w"] = 1.5` と一致する。
したがって `prob` 配列が同一となり、同一 RNG 状態からの `rng.binomial` の出力も同一になる。

この帰結（`X`/`Y`/`Z`/`F` がすべて既定呼び出しと一致すること）は、
Phase 7e の data seed 41001–41003 上で決定的に確認した。

**→ Option B を `k_ref=3` で採用しても `K_TRUE=3` anchor は不変。
decision `C`（anchor 使用不可）は発生しない。**

### 8.4 Option B の残差（**解析的に必ず残るもの**）

`w_K² K = const` は **2 次モーメントのみ**を一致させる。

- `z_i^T z_j` は `K` 個の独立項の和であり、**その分布形状（尖度など高次モーメント）は
  `K` に依存する**（`K=1` は積型で裾が重く、`K` 増大とともに正規に近づく）。
- `σ(·)` は非線形なので、Y の density と `H(y|Z)` は分散だけでなく高次モーメントに依存する。
- したがって **variance matching では density / entropy を完全には一致させられない。**

**Option B でも「潜在次元だけが変わる」とは断定できない。**
到達できる最良の表現は §8.2 の estimand 文言のとおりである。

### 8.5 どちらも `K_TRUE` 単独操作ではない

案 A・案 B のいずれも、`K_TRUE` を変えると同時に何かが変わる。
違いは**何を固定した族に沿って比較するか**である。
これは技術的欠陥ではなく **estimand の選択**であり、人間が決めるべき科学的判断である。

### 8.6 Strategy comparison（**Human Gate 2026-09-01: Strategy 3 = SELECTED**）

| Strategy | 状態 | 内容 | pros | cons |
|---|---|---|---|---|
| **1. A only** | *alternative considered — NOT SELECTED* | fixed `w` のみ実行 | 実装最小・anchor 無条件互換・新規 168 fits | 次元効果と Y 応答不確実性が分離されない。ゼミ指摘への回答としては限定的 |
| **2. B only** | *alternative considered — NOT SELECTED* | variance-matched のみ実行 | Y 分散の K 依存を除去。ゼミ指摘に近い | Phase 7e が採用した generator family そのものの挙動は測れない。高次モーメント残差は残る |
| **3. A + B（separate estimands）** | **SELECTED（Human Gate 2026-09-01, H1）** | 両方を別 estimand として事前登録。**A/B の hierarchy は H3-a（A primary + B pre-registered sensitivity）に確定済み（§16 H3）** | **A/B contrast により、Y natural-parameter variance を `K=3` 基準へ match するための w-scaling rule に対して selected-K pattern がどの程度 sensitive かを記述できる**（§13 の解釈境界を参照）。どちらの estimand も結果前に freeze される | fit 数が倍（§11）。集計・report が 2 系統になる。**A/B 差は Y variance だけの isolated causal contribution を意味しない** |
| **4. generator redesign** | *alternative considered — NOT SELECTED* | 新しい生成規約を設計（例: `d` を `K` とともに増やす等） | 原理的にはより清潔な比較が可能 | 本 Issue の scope 外。Phase 7e anchor を捨てることになる。設計・検証コストが大きい |

> **INDEPENDENT REVIEW RECOMMENDATION（Codex）: R3 = `PRE_REGISTER_A_AND_B_AS_SEPARATE_ESTIMANDS`（Strategy 3）**
>
> **HUMAN GATE DECISION（2026-09-01, Issue #47）: H1 = `A+B`（Strategy 3）— SELECTED**

独立レビューの推奨（R3）と Human Gate の決定は別個の記録である。
R3 は推奨であり、実際に scope を確定したのは 2026-09-01 の Human Gate decision である（§16 H1）。

---

## 9. Frozen configuration（**CURRENT — Human Gate 2026-09-01 反映済み**）

**これが current config である。UNDECIDED 項目は存在しない。**
検討されたが採用されなかった代替案は §8.6 / §10.4 / §10.7 / §16 に
`ALTERNATIVES CONSIDERED (NOT SELECTED)` として履歴のために残してある。

```text
--- HUMAN GATE FROZEN (2026-09-01, GitHub Issue #47) ---------------------
ESTIMANDS             = A+B                                              (H1)
PRIMARY_ESTIMAND      = A
SENSITIVITY_ESTIMAND  = B
HIERARCHY             = H3_A        A primary + B pre-registered sensitivity (H3)
RANDOM_DESIGN         = CRN         data_seed / model_seed のみを支配          (H2)
MASK_DESIGN           = S_C         Phase 7e anchor-aligned pair-index mask    (H4)

Option A  w_true      = 1.5                          (fixed)
Option B  w_true(K)   = 1.5 * sqrt(3 / K_TRUE)       (w_K^2 * K = 1.5^2 * 3)
          → K_TRUE=3 では A・B とも w = 1.5（既存 anchor を共有できる根拠）

--- FROZEN EXPERIMENTAL FACTORS ------------------------------------------
issue                 = 47 (design) / 次 implementation Issue (execution)
phase                 = 8a
model lineage         = DualExpFamLSMConsistent   (experimental prototype・本文採用不可)
numerics_mode         = consistent
family_x              = poisson
family_y              = bernoulli
n                     = 75
d                     = 15
L                     = 5
num_iter              = 8
test_ratio            = 0.20        (expected_test_pairs = 555)
var_f                 = 5.0
uniq                  = 0.1
w0_true               = -1.0
K_TRUE (new)          = {1, 2, 4, 5}
K_TRUE (anchor)       = 3           (Phase 7e 既存 42 fits を再利用・再実行しない)
K_TRUE (final grid)   = {1, 2, 3, 4, 5}
candidate K           = {1, 2, 3, 4, 5, 6, 7}
starts per K          = {1, 2}
dataset replicates    = {1, 2, 3}
primary score         = held-out Bernoulli raw-eta mean log score
                        eta_ij = w0 + w * z_i^T z_j
                        s_ij   = y_ij*eta_ij - logaddexp(0, eta_ij)
                        fit score = mean over held-out upper test pairs
selector              = 2-start mean -> max K score -> tie <= 1e-12 -> smallest K
tie_tolerance         = 1e-12       (roundoff 保護のみ。統計的同等性閾値ではない)
使用しない             = predict_mu_y / threshold / probability clipping / BIC / Q_strict

--- FROZEN SEED / MASK POLICY --------------------------------------------
data_seed             = H2=CRN のため A・B で対応（estimand offset なし）
model_seed            = H2=CRN のため A・B で対応（estimand offset なし）
split_seed            = H4=S_C: 42000 + replicate（Phase 7e anchor と同一）
                        ※ H2 は split_seed を支配しない（§10.7.1）

--- FROZEN BUDGET --------------------------------------------------------
new fits              = 4 K_TRUE x 3 rep x 7 cand K x 2 starts x 2 estimands = 336
existing anchor       = 42 (Phase 7e K_TRUE=3、A・B で共有・再実行しない)
unique total          = 378
per-estimand matrix   = 168 new + 42 anchor = 210-fit equivalent (15 cells)
```

### primary outputs（1 estimand あたり K_TRUE × replicate = 15 セル）

各セルにつき: `selected K` / `tie candidate set` / `best score` / `second-best score` /
`margin`（統計的有意差ではない） / `selected K − K_TRUE` / `|selected K − K_TRUE|` /
`under | exact | over` の記述ラベル。

集計: K_TRUE 別 selected-K counts / K_TRUE 別 exact 数（分母 3） /
15 セルの confusion-style table / signed error / absolute error。

**`3 replicates` なので百分率を一般性能として扱わない。
`exact-selection rate` を出す場合も descriptive only と明記する。**

---

## 10. Seed convention

### 10.1 Phase 7e（凍結・変更しない）

```
data_seed  = 41000 + replicate                             -> {41001, 41002, 41003}
split_seed = 42000 + replicate                             -> {42001, 42002, 42003}
model_seed = 43000 + replicate*1000 + K*10 + start         -> 44011 … 46072 (42 個)
```

### 10.2 Phase 8a 提案（新規 `K_TRUE ∈ {1,2,4,5}` 専用ブロック）

```
data_seed  = 51000 + 100*K_TRUE + replicate
model_seed = 530000 + 10000*K_TRUE + 1000*replicate + 10*K + start
split_seed = 42000 + replicate        <-- H4 = S_C (SELECTED, Human Gate 2026-09-01)
```

`data_seed` と `model_seed` は `(K_TRUE, replicate, candidate_K, start)` から一意に決まる。

| 役割 | Phase 7e 範囲 | Phase 8a 範囲 | 個数 |
|---|---|---|---:|
| data | 41001–41003 | 51101–51503 | 12 |
| split（S-a） | 42001–42003 | 52101–52503 | 12 |
| split（S-b） | 42001–42003 | 52001–52003 | 3 |
| split（S-c） | 42001–42003 | **42001–42003 を意図的に再利用**（§10.4） | 3 |
| model | 44011–46072 | 541011–583072 | 168 |

### 10.3 衝突検査（design 段階で全数実施済み・fit は実行しない）

| 検査 | 結果 |
|---|---|
| 新規 model seed 個数 / 一意性 | 168 個・168 distinct → **一意** |
| Phase 7e 全 seed ∩ Phase 8a 全 seed（**data / model のみ**） | **空集合** |
| Phase 8a 内 role 間重複（data∩split, data∩model, split∩model） | すべて空（S-a / S-b） |
| Phase 7e 内 role 間重複 | すべて空 |

**S-c の split seed のみ、Phase 7e の値（42001–42003）を意図的に再利用する。**
これは accidental collision ではなく **pre-registered common-mask design** である（§10.5）。
data seed と model seed は S-c でも Phase 7e と交差しない。

例（`K_TRUE, rep, K, start → data, model`）:

```
1 1 1 1 -> 51101 541011      5 1 1 1 -> 51501 581011
1 1 7 2 -> 51101 541072      5 1 7 2 -> 51501 581072
1 3 1 1 -> 51103 543011      5 3 1 1 -> 51503 583011
1 3 7 2 -> 51103 543072      5 3 7 2 -> 51503 583072
```

### 10.4 split mask 構成 — **H4 = S-c に確定（Human Gate 2026-09-01）**

`make_pair_split(n, test_ratio, seed)` は `K` を引数に取らないため、
`(n, test_ratio, split_seed)` が同じなら **`K_TRUE` に依らず同一の pair-index mask** が得られる。
これを踏まえ、H4 の選択肢は次の 3 案である。

**前版の誤りの訂正:** 前版は S-b を「全 `K_TRUE` で同一の held-out pair index 集合」「C6 を統制できる」と
記載していたが、これは**誤りである**。S-b の `52000 + replicate` が揃えるのは
**新規 `K_TRUE ∈ {1,2,4,5}` の間だけ**であり、
Phase 7e の `K_TRUE=3` anchor は `42000 + replicate` で生成された別の mask を持つ。
本改訂で S-b の記述を限定表現へ訂正し、全水準を揃える案を **S-c** として新設した。

#### S-a: K_TRUE-specific independent masks

```
split_seed = 52000 + 100*K_TRUE + replicate        -> 52101 … 52503 (12 個)
```

- 各 `K_TRUE` / replicate で**別の** split mask。
- mask-level Monte Carlo variation を `K_TRUE` 間で共有しない。
- Phase 7e anchor との自然な独立性を維持する。

**limitation:** `K_TRUE` 間の selected-K 差に **mask-level variation も含まれる。C6 は残る。**

#### S-b: NEW-K shared mask only

```
新規 K_TRUE ∈ {1,2,4,5} : split_seed = 52000 + replicate   -> 52001 … 52003 (3 個)
Phase 7e K_TRUE=3 anchor : split_seed = 42000 + replicate   （変更しない）
```

- **新規 4 水準（`K_TRUE ∈ {1,2,4,5}`）の間でのみ**同一 pair-index mask を共有する。
- **`K_TRUE=3` anchor とは mask が異なる。**

**正しい表現は `partial mask alignment among new K_TRUE conditions` である。**
**「全 K_TRUE 共通」とは書かない。「C6 を完全に control する」とも書かない。**

**limitation:** `K_TRUE=3` anchor との比較だけが非対称になる。

#### S-c: Full anchor-aligned shared mask

Phase 7e の `K_TRUE=3` が実際に使用した

```
split_seed = 42000 + replicate
```

が生成する pair-index mask を、新規 `K_TRUE ∈ {1,2,4,5}` にも**意図的に**使用する。

**成立の前提（実装前に確認すること）:**

- `n = 75` が同一
- `test_ratio = 0.20` が同一
- `make_pair_split` が `K_TRUE` を参照しない（§4 で確認済み）
- Phase 7e の mask construction semantics が現 HEAD でも同一
  （§3.3 の `train_mask_hash` / `test_mask_hash` 一致で確認済み）

これらが満たされる場合、replicate `r` について
**`K_TRUE = 1,2,3,4,5` のすべてで同一の pair-index mask を使用できる。**

**重要:** `K_TRUE=3` を再実行しない。
**既存 Phase 7e anchor の mask を reference とし、新規 `K_TRUE` の mask をその reference へ揃える。**
実装時には最低限、replicate ごとに

```
split_mask_hash(new K_TRUE, rep) == split_mask_hash(phase7e K3 anchor, rep)
```

を **full fit より前の zero-EM provenance gate** で assert する（implementation plan §3.4）。

**S-c を採用する場合でも、次のようには呼ばない:**

- ~~same dataset~~
- ~~paired statistical replicate~~
- ~~identical Y~~ / ~~identical Z~~

**同じなのは pair-index held-out mask だけである。**
underlying の `Z` / `F` / `X` / `Y` は `K_TRUE` ごとに別の generator realization である。

#### H4 trade-off table

| option | K1/2/4/5 alignment | K3 anchor alignment | advantage | limitation |
|---|:--:|:--:|---|---|
| **S-a** | none | no | より独立な generator + mask realization。水準間の偶然が伝播しない | `K_TRUE` comparison に mask variation が混ざる |
| **S-b** | yes | no | 新規 4 水準間では mask-level variation を揃えられる（partial alignment） | 中間案だが **`K_TRUE=3` anchor との比較だけ非対称**になる |
| **S-c** | yes | yes | **mask-level variation を全 true-K grid（1–5）で揃えられる** | intentional split-seed reuse／同一 mask は同一データを意味しない／anchor protocol への依存が強くなる／mask-specific behavior を全 true-K で共有する |

**Human Gate（2026-09-01, GitHub Issue #47）により `H4 = S-c` が SELECTED。**
S-a / S-b は *alternatives considered — NOT SELECTED* として履歴のために残す。

**S-c 採用に伴う運用条件（frozen）:**

- **`K_TRUE = 3` を再実行しない。** Phase 7e artifact の保存済み `split_mask_hash` /
  split seed / provenance を **read-only reference** として使用する。
- 新規 `K_TRUE ∈ {1,2,4,5}` の mask を、その reference へ揃える。
- **full fit の前に §10.4 の zero-EM provenance gate を必ず通す**
  （implementation plan §3.4 の MC1–MC4）。
- 一致しない場合は STOP。seed rescue / Phase 7e rerun / tolerance relaxation /
  post-hoc mask replacement / failed cell drop / **new reference mask generation** をすべて禁止する。

### 10.5 Seed policy（S-c と整合するよう整理）

| 役割 | policy |
|---|---|
| **data_seed** | `K_TRUE` / replicate ごとに一意。**意図しない collision は禁止。** |
| **model_seed** | `K_TRUE` / replicate / candidate K / start ごとに一意。**意図しない collision は禁止。** |
| **split seed / mask** | **H4 に依存する（一律の一意性要求は課さない）。** S-a: `K_TRUE`-specific で一意。S-b: 新規 `K_TRUE` 間で**意図的に共有**、K3 anchor とは非共有。S-c: 同一 replicate について `K_TRUE = 1…5` で**意図的に共有**。 |

**S-b / S-c における split seed の共有は accidental seed collision ではなく、
`pre-registered common-mask design` として扱う。**
したがって「全 seed は役割内でも完全 unique でなければならない」という一律要求は
split seed には適用しない（data / model seed には引き続き適用する）。

manifest には最低限、次の意味を持つ列を設ける（exact field name は既存 schema との整合を見て決めてよい）:

| 意味 | 内容 |
|---|---|
| split seed | 実際に使用した seed 値 |
| split mask hash | 生成された mask の hash |
| mask design | `S-a` / `S-b` / `S-c` のいずれか |
| mask group id | 同一 mask を共有する cell 群の識別子 |
| anchor mask hash | Phase 7e K3 anchor の mask hash（S-c では一致を要求、S-a/S-b では参照値として記録） |
| intentional seed reuse | split seed の共有が意図的であることの明示フラグ |

### 10.6 設計判断の記録

- **data seed は `K_TRUE` に依存させる。** 同一 seed を別 `K_TRUE` で使うと、
  `Z`/`F` が同一 RNG stream の先頭から reshape されて生成され、水準間に人為的相関が入る。
- **seed rescue / replacement は禁止**（§12）。
  S-c で mask hash が一致しない場合、**seed を差し替えて合わせることも、
  Phase 7e を再実行して合わせることも禁止**である。STOP して人間へ返す。

### 10.7 A + B を実行する場合の random-number design（**human gate H2**）

Strategy 3（A+B）を採る場合、A と B の間で乱数をどう扱うかは独立の設計判断である。

#### 10.7.1 H2 と H4 の責任範囲（**凍結**）

**H2 と H4 は管理対象が重ならない。split seed / split mask については H4 の決定が H2 に優先する。**

| gate | 管理する対象 | 管理し**ない**対象 |
|---|---|---|
| **H2** | Option A ↔ B 間の **data-generation RNG**（`data_seed`）と **model-initialization RNG**（`model_seed`）の関係 | **pair-index held-out mask を管理しない** |
| **H4** | **held-out pair-index mask の構成のみ**（`split_seed` / mask） | data / model RNG を管理しない |

#### 10.7.2 H2 の 2 択（**Human Gate 2026-09-01: `CRN` = SELECTED**）

| 設計 | 内容 | pros | cons |
|---|---|---|---|
| **common random numbers (CRN)** | A と B で、事前登録された範囲の **data random draws** と **model random draws** を対応させる（違いは `w_true` のみ） | A/B 差が **`w` の違いのみ**に帰属する。分散の共通成分が相殺され、差の解釈が最も明確 | A と B が統計的に独立でない。片方の偶然が両方に伝播する |
| **independent random blocks** | A と B に **`data_seed` と `model_seed` の別 block** を割り当てる | data / model RNG について A と B が独立。片方の偶然が他方に伝播しない | A/B 差に乱数変動が上乗せされ、`w` の寄与が見えにくい |

**CRN を採る場合でも split mask は H4 の S-a / S-b / S-c 定義に従う。
「CRN だから split seed も必ず同一」とはしない。**

**INDEPENDENT を採る場合、estimand-specific offset / 別 block を適用する対象は
`data_seed` と `model_seed` のみである。`split_seed` / pair-index mask に
estimand-specific offset を自動適用しない。split は H4 が単独で支配する。**

**注:** CRN を採る場合でも、`w_true` は RNG 消費順序に影響しない（§8.3）ため、
A と B は `Z`・`F`・`X` を共有し `Y` の draw のみが変わる。
これは CRN として自然な構成である。

#### 10.7.3 H2 × H4 の組み合わせ（**frozen: `CRN` × `S-c`**）

**H2 と H4 は scientific decision として直交するが、
これは「randomness の全成分が A/B 間で独立になる」という意味ではない。**

両者は独立に選べる。例えば `H2 = CRN` かつ `H4 = S-a` は可能であり、
`H2 = independent` かつ `H4 = S-c` も可能である。

**`H2 = INDEPENDENT` かつ `H4 = S-c` は明示的に合法である。**
この場合の independent は **data-generation RNG と model-initialization RNG に限る。
pair-index mask は H4 = S-c に従って A/B 間でも共有される。**
すなわち **A/B are independent in data/model RNG, but share the held-out pair-index mask by H4 = S-c.**

**この構成を `fully independent experiments` や `all random numbers independent` と呼ばない。**

| 組み合わせ | data RNG (A↔B) | model RNG (A↔B) | pair-index mask |
|---|---|---|---|
| `INDEPENDENT` × `S-a` | independent | independent | H4 に従い `K_TRUE`-specific（A/B 間でも別）。**より広い意味で独立性は高いが、それは H4=S-a の帰結であり、H2 が mask independence を保証するわけではない** |
| `INDEPENDENT` × `S-b` | independent | independent | H4=S-b の規則に従う（新規 `K_TRUE` 間のみ共有。共有範囲の定義は §10.4 から変更しない） |
| `INDEPENDENT` × `S-c` | independent | independent | **Phase 7e anchor-aligned common mask を A/B 間でも共有** |
| `CRN` × `S-a` / `S-b` / `S-c` | 対応 | 対応 | H4 の定義に従う |

**結果を見てから決めない。§16 の H2 / H4 で freeze する。**

---

## 11. Fit budget（**FROZEN — H1 = A+B**）

### 11.1 current budget（Human Gate 2026-09-01）

`K_TRUE=3` では `w_3 = 1.5` が A と B で一致するため（§8.3）、
**Phase 7e の 42 fits は A と B の共通 anchor として使用できる。**

```
4 new K_TRUE ({1,2,4,5})
  x 3 replicates
  x 7 candidate K
  x 2 starts
  x 2 estimands (A, B)
------------------------------------------------------------------
new fits           : 336
existing anchor    : 42   (Phase 7e K_TRUE=3、A・B で共有・再実行しない)
unique total fits  : 336 + 42                = 378
各 estimand の matrix : 168 new + 42 anchor    = 210-fit equivalent（それぞれ 15 セル）
```

**`210 × 2 = 420` unique fits ではない。** anchor 42 は A・B で共有され、二重に数えない。

### 11.2 検討された代替（*NOT SELECTED*）

| Strategy | 状態 | unique new fits | unique total fits |
|---|---|---:|---:|
| 1（A only） | *NOT SELECTED* | 168 | 210 |
| 2（B only） | *NOT SELECTED* | 168 | 210 |
| **3（A+B）** | **SELECTED** | **336** | **378** |
| 4（redesign） | *NOT SELECTED* | 未定（別 Issue） | 未定 |

### 11.3 実行許可の状態

**budget が確定していることは full run の実行許可を意味しない。**
336 fits の実行は、implementation → static/adversarial tests →
zero-EM provenance gate → smoke → independent review → **明示的な人間の承認**
を経たあとにのみ許可される（§15.1）。

### 11.4 execution lineage の分離（必須）

| lineage | K_TRUE | fits | RUN_CODE_SHA | 出力先 |
|---|---|---:|---|---|
| Phase 7e（既存） | 3 | 42 | `b9311e64a7b36c0a8a9704fff0ee7b38efe36a8a` | `expfam/results/k_selection/heldout_full_pilot_20260824/` |
| Phase 8a（新規） | 1, 2, 4, 5 | 168 / estimand | 実行時に記録 | `expfam/results/k_selection/k_true_robustness_<estimand>_<YYYYMMDD>/` |

**別 recorded execution lineage である。**
集約表・図には必ず `lineage` / `run_code_sha` / `artifact_dir` 列を持たせ、
KI-002 に従い「どの系列の数値か」を明示する。
Phase 7e の artifact ディレクトリには**一切書き込まない**。

**execution date と code SHA の差だけでは scientific confound ではない（§14 C7）。**
必要なのは provenance の分離である。

---

## 12. Failure policy

Phase 7e の global-stop philosophy を継承する。

### 12.1 BLOCKING（1 件でもあれば full run に進まない / 中断し部分結果を採用しない）

- **generator configuration failure（deterministic / algebraic のみ）** — §12.3
- manifest mismatch（行数・key 集合・順序・seed 規約・`w_true` 値）
- seed collision（§10.3 の検査に不合格）
- leakage / held-out target contamination
- missing / duplicate (estimand, K_TRUE, replicate, K, start)
- warning（`warnings` 非空）
- retry（`internal_retry > 0`）
- NaN / 非有限
- Q failure（`q_failure=True`）
- score mismatch（独立再計算との不一致）
- provenance mismatch（hash 群の不一致）
- unexpected artifact

### 12.2 禁止行為（failure 後）

seed rescue / seed replacement / tolerance relaxation / candidate K range 変更 /
replicate drop / failed-cell drop / 部分結果の報告。

失敗時は **全体を停止し、原因を記録して人間へ返す。**

### 12.3 sample-level 統計量を blocking gate にしない（**HIGH-03**）

Option B では `w_K² K = const` は**代数的に**一定だが、
有限データセットで観測される標本統計量（標本 `sd(η^Y)` など）は**確率変動する**。
正しい generator であっても標本レベルの閾値を超えうるため、
**標本統計量に閾値を課して full run を停止させてはならない**（false failure になる）。

| 種別 | 項目 | 扱い |
|---|---|---|
| **BLOCKING（deterministic / algebraic）** | configured `w_K` が `1.5·√(3/K_TRUE)` と一致 | 停止条件 |
| | `w_K² · K_TRUE` が数値許容内で全 `K_TRUE` 一定 | 停止条件 |
| | `K_TRUE = 3` で `w_3 = 1.5` | 停止条件 |
| | generator formula / parameter mapping に予期しない分岐がない | 停止条件 |
| | manifest に expected `w_K` が保存されている | 停止条件 |
| **RECORD ONLY** | 標本 `sd(η^Y)` | 記録するが停止させない |
| | Y density | 記録するが停止させない |
| | 条件付きエントロピー | 記録するが停止させない |
| | oracle score | 記録するが停止させない |
| | 高次モーメント診断 | 記録するが停止させない |

Option A では variance 一定の条件を課さない（分散が動くことを承知で選んだ semantics であるため）。
ただし **RECORD ONLY 項目は必ず artifact に記録し、report に limitation として転記する。**

---

## 13. Interpretation boundary

### この実験が答えられること

- `K_TRUE = 1/2/4/5` で selected K がどこに出るか（記述）
- under-selection / over-selection の傾向が見えるか
- `K_TRUE = 3` の Phase 7e 結果が特殊に見えるか
- candidate range の端（K_TRUE=1 は下端、K_TRUE=5 は上端 7 に近い）で挙動が変わるか
- （Strategy 3 の場合）A と B の差が見えるか

### この実験が答えられないこと（**書いてはいけない**）

- `selector is consistent` / asymptotic consistency
- `P(K̂ = K_TRUE) → 1`
- `true K is recovered in general` / `true K generally recovered`
- `n → ∞` で真の K を選ぶ
- `held-out criterion is theoretically valid`
- `held-out criterion is better than BIC` / `BIC is invalid because this selector works better`
- real-data latent dimension validity

### n → ∞ との境界（明記事項）

**本 K_TRUE sweep は `n = 75` 固定の finite-sample empirical robustness のみである。**
`n` は 1 水準しかなく、標本サイズは一切変化しない。
ゼミ指摘のうち「データ数が無限大のときに何を保証できるか」は
**asymptotic theory の別 Issue** であり、本 sweep の有限標本結果を
その証拠として流用しない。

### 追加の境界（本 design 固有）

- **held-out score の水準を `K_TRUE` 間で比較しない。** §6.2 のとおり oracle 水準の
  irreducible score level 自体が `K_TRUE` で変化する。比較してよいのは
  **同一 K_TRUE・同一 replicate 内の candidate K 間**のみ。
  `best score` / `margin` の `K_TRUE` 間比較は行わない。
- **K_TRUE 間の score-level 差と candidate-K selection margin の比を取らない。**
  異なる統計量であり、比率を confounding magnitude として解釈しない。
- `margin` は統計的有意差ではない（Phase 7e と同じ）。
- 3 replicates の百分率を一般性能として扱わない。
- 異なる `K_TRUE` は異なる推定問題であり、水準間の「性能の優劣」を述べない。
- **A/B contrast の解釈を限定する。** A/B 差から言えるのは
  「Y natural-parameter variance を `K=3` 基準へ match するための **w-scaling rule** に対して
  selected-K pattern がどの程度 sensitive か」までである。
  **A/B の結果差を「Y variance だけの isolated causal contribution」と解釈しない。**
  `w_K` を変えると `η^Y` の分布は分散だけでなく全体が変化し（§8.4）、
  Option B は variance matching only だからである。
  なお「A と B の generator 呼び出しで異なる scientific manipulation は `w` rule のみである」
  という **mechanical statement** は正しく、これは書いてよい。
- **A+B を実行する場合、`H3` で freeze した A/B の役割（H3-a の primary / sensitivity、
  または H3-b の co-equal）を結果に応じて入れ替えない。**
  favorable な option だけを本文採用することを result を見て決めない（§16 H3 Reporting freeze）。
- model lineage は `DualExpFamLSMConsistent`（experimental prototype）であり、
  **本文採用不可**（CLAUDE.md §3）。

---

## 14. Remaining confounders — triage

Codex independent review の分類を反映した最終 triage。

| # | 項目 | 分類 | 内容と根拠 |
|---|---|---|---|
| **C1** | Y signal strength（`Var(η^Y) = w² K c_n`） | **MUST FIX BEFORE SWEEP** | fixed-`w` では `K_TRUE` に比例。**対処は「A/B の estimand を結果を見る前に freeze すること」**（§8・§16 H1）。generator を直す必要はなく、何を比較しているかを事前に確定すればよい |
| **C2** | inner-product distribution shape（`z_i^Tz_j` の高次モーメントの K 依存） | **ACCEPTABLE LIMITATION** | 分散一致では除去不能（§8.4）。estimand の limitation として明記すれば許容 |
| **C3** | X coordinate information | **NOT ACTUALLY A CONFOUNDER** | **mean loading energy `tr(F^TF)/K = d(1−uniq)/K` は `1/K` で低下する**が、これは固定 `d` のもとで潜在自由度を増やすことに伴う **model difficulty の一部**であり、外的交絡ではない。**Poisson-X の Fisher information が `1/K` でスケールすることは未証明**（§5.5 (5.8)）であり、断定しない |
| **C4** | K_TRUE=1 の `F` 退化 / 特殊構造 | **MUST CONTROL / STRATIFY** | blocker ではない（§7）。集計時に K_TRUE=1 を stratify し、`F` が `±√0.9` に退化する構造を report に明記する |
| **C5** | candidate-range edge effect（K_TRUE=1 は下端、5 は上端 7 に近い） | **MUST CONTROL / STRATIFY** | 設計上の意図（Issue #47）。over / under selection の非対称性を水準別に stratify して報告する |
| **C6** | held-out pair mask variation（水準間で held-out pair 集合が異なりうる） | **CONTROLLED（H4 = S-c）** | Human Gate により **S-c = fully aligned at pair-index-mask level across `K_TRUE = {1,2,3,4,5}`**（A・B 両 estimand で共通）。統制されるのは **pair-index mask のみ**であり、`Z`/`F`/`X`/`Y` は水準ごとに別の generator realization である。**「全 variation を control する」とは書かない。** S-a / S-b は *NOT SELECTED*（§10.4） |
| **C7** | anchor と新規セルが別 execution lineage | **NOT ACTUALLY A CONFOUNDER** | code diff 空・data hash 一致・score config 一致（§3.3）。**ただし `lineage` / `run_code_sha` / `artifact_dir` の provenance 列は必須**（§11.4） |
| **C8** | replicate 数 3 | **ACCEPTABLE LIMITATION** | descriptive only。百分率を一般性能として扱わない（§13） |

---

## 15. Decision gate

**`A: IMPLEMENT_K_TRUE_ROBUSTNESS_HARNESS_NEXT`**

**（2026-09-01 の Human Gate decision により、旧
`B: GENERATOR_COMPARABILITY_MUST_BE_FIXED_BEFORE_SWEEP` から更新）**

### 15.1 `A` が許可するもの・許可しないもの

> **`A` は full experiment の実行許可ではない。**
> **`A != authorization for full experiment`**

| | 状態 |
|---|---|
| **許可される** | harness の implementation / static tests / adversarial tests / **zero-EM validation**（`--validate-only`・`--config-gate`・`--record-diagnostics`） |
| **まだ許可されない** | **336 fits の full run**、smoke を超える EM 実行、canonical docs 更新、`EXPERIMENT_REGISTRY.md` への追記 |

**次段階の順序（各段階で人間へ返す）:**

```
implementation
  -> static / adversarial tests
  -> zero-EM provenance gate (MC1-MC4 を含む)
  -> smoke
  -> independent review
  -> explicit human approval
  -> full 336-fit run
```

### 15.2 なぜ B から A へ移れるか

旧 gate `B` の未確定部分は、2026-09-01 の Human Gate decision（Issue #47）により解消した。

| 旧 B の理由 | 解消状況 |
|---|---|
| Option A と Option B が別 estimand であり、どちらを実行するか人間が freeze していない | **解消。** H1 = `A+B`、H3 = `H3-a`（A primary / B pre-registered sensitivity）として両方を事前登録（§16） |
| fixed-`w` では `Var(η^Y) = w² K c_n` が `K_TRUE` 依存 | **estimand 定義として明示的に受け入れた。** Option A の必須 limitation として記載（§8.1・§16） |
| Option B でも高次モーメント等が `K_TRUE` 依存 | **estimand 定義に limitation として含めて事前登録した**（§8.2・§8.4） |

**generator は変更していない。** 解決したのは「何を比較しているか」の確定であり、
generator semantics の書き換えではない。

### 15.3 変わらない判断

- `C: PHASE7E_K3_CANNOT_BE_REUSED_AS_ANCHOR` は引き続き該当しない。
  code diff 空・stored hash 一致・score config 一致、かつ `K_TRUE=3` では A・B とも `w = 1.5`（§3.3・§8.3）。
- `D: K_TRUE_SWEEP_NOT_JUSTIFIED` も引き続き該当しない。
- **理由として使用しないもの（維持）:** `K_TRUE` 間の oracle score 水準差と
  candidate-K selection margin の比。異なる統計量であり、confounding magnitude の
  定量化として不適切である（§6.2）。

### 15.4 Acceptance criteria（本 design Issue の完了条件）

- [x] K_TRUE variable path が generator から score まで監査済み（§4）
- [x] `K_TRUE = {1,2,3,4,5}` の rationale を文書化（§2・Issue #47）
- [x] Phase 7e `K_TRUE=3` anchor 互換性を EM 実行前に確立（§3.3 `REUSABLE_ANCHOR`）
- [x] candidate K は `1..7` のまま（§9）
- [x] signal comparability を明示的に監査（§5・§6）
- [x] `K_TRUE=1` boundary を明示的に監査（§7）
- [x] deterministic かつ衝突のない seed 規約を提示（§10）
- [x] fit budget を確定（**336 new + 42 anchor = 378 unique**、§11）
- [x] 実験を一切実行していない（EM fits = 0、§17）
- [x] model / result / canonical docs を変更していない（§17）
- [x] interpretation boundary が consistency / general recovery claim を禁止（§13）
- [x] independent design review 完了（Codex: `APPROVE_DESIGN_FOR_HUMAN_GATE`）
- [x] **Human Gate decision（H1–H4）が確定・記録済み（2026-09-01, Issue #47、§16）**
- [x] decision gate を 1 つ選択（**`A`**、ただし §15.1 のとおり full run の許可ではない）

## 16. Human Gate decisions（**FROZEN 2026-09-01**）

**Decision source:** GitHub Issue #47 — Human Gate Decision comment（2026-09-01, by H-T-0726）
**Decision type:** **HUMAN GATE DECISION**（AI / independent-review recommendation ではない）

**H1–H4 はすべて確定済みである。PENDING の項目はない。**
**結果を見てから変更することはできない。**

| gate | 決定 | 状態 |
|---|---|---|
| **H1** | `A+B` | **SELECTED** |
| **H2** | `CRN` | **SELECTED** |
| **H3** | `H3-a`（A primary + B pre-registered sensitivity） | **SELECTED** |
| **H4** | `S-c`（Phase 7e anchor-aligned shared pair-index mask） | **SELECTED** |

---

### H1 — estimands: **`A+B`（SELECTED）**

Option A と Option B を**両方**事前登録して実行する。

| estimand | `w_true` | 内容 |
|---|---|---|
| **Option A** | `w = 1.5`（fixed） | original Phase 7e generator family |
| **Option B** | `w_K = 1.5·√(3 / K_TRUE)` | Y natural-parameter variance matched |

**A と B は別の estimand である。結果に基づいてどちらかを drop / 昇格してはならない。**

*ALTERNATIVES CONSIDERED — NOT SELECTED:* `A only` / `B only` / `generator redesign`（§8.6）

### H2 — A/B の random-number relationship: **`CRN`（SELECTED）**

A/B 間で **data-generation RNG** と **model-initialization RNG** を
pre-registered common-random-number design で対応させる。

**H2 は pair-index held-out mask を管理しない。
split seed / pair-index mask は H4 のみが管理する（§10.7.1）。**

**CRN を `all random numbers are identical` と書かない。**
一致するのは事前登録された data / model の random draws であり、
`w_true` が異なる以上 Y の draw は A と B で異なる（§8.3）。

*ALTERNATIVE CONSIDERED — NOT SELECTED:* `independent random blocks`（§10.7.2）

### H3 — scientific hierarchy: **`H3-a`（SELECTED）**

**A primary + B pre-registered sensitivity**

| estimand | role | scientific question |
|---|---|---|
| **Option A**（`w = 1.5` fixed） | **primary** | original Phase 7e generator family において、`K_TRUE` を変えたとき held-out K selector が finite sample でどう振る舞うか |
| **Option B**（`w_K = 1.5·√(3/K_TRUE)`） | **pre-registered sensitivity** | Y natural-parameter variance の `K` 依存を ensemble level で緩和した場合、selected-K pattern がどう変わるか |

**必須条件（frozen）:**

- **role は implementation 前に freeze 済みである。**
- **role switching を禁止する。**
- **A が悪く B が良かった場合でも、B を primary へ昇格しない。**
- **B が悪かった場合でも、sensitivity result を隠さない。**
- **A/B 両方を pre-registration どおり報告する。**

*ALTERNATIVE CONSIDERED — NOT SELECTED:* `H3-b`（A/B co-equal separate estimands）

### H4 — held-out pair-index mask: **`S-c`（SELECTED）**

Phase 7e `K_TRUE=3` で保存済みの split seed / split mask を reference とし、
新規 `K_TRUE ∈ {1,2,4,5}` にも**同一の pair-index mask**を適用する。

```
Phase 7e : split_seed = 42000 + replicate   （reference・変更しない）
新規     : K_TRUE = {1,2,4,5} にも同じ pair-index mask を適用
```

**必須条件（frozen）:**

- **`K_TRUE = 3` を再実行しない。**
- Phase 7e artifact の保存済み `split_mask_hash` / split seed / provenance を
  **read-only reference** として使用する。
- **full fit の前に zero-EM gate（implementation plan §3.4 MC1–MC4）を必ず通す。**
- **同一 mask は same dataset を意味しない。** 各 `K_TRUE` の `Z` / `F` / `X` / `Y` は
  その `K_TRUE` に応じた generator output であり、揃っているのは held-out pair index のみである。

*ALTERNATIVES CONSIDERED — NOT SELECTED:* `S-a` / `S-b`（§10.4）

---

### Reporting freeze（H3 に付随・frozen）

**H3 の decision は implementation / smoke / full run のいずれよりも前に freeze 済みである。**
以下を禁止する。

- result を見た後の **primary の変更**
- result を見た後の **sensitivity の昇格**
- **favorable な option だけを本文採用することを experiment result を見て決めること**
- **A/B role の post-hoc reinterpretation**

### H2 + H4 の相互作用（frozen config での意味）

frozen config は `H2 = CRN` かつ `H4 = S-c` である。責任範囲は分離されている（§10.7.1）。

- **H2 = CRN:** A/B 間で data / model random draws を対応させる。
- **H4 = S-c:** Phase 7e anchor-aligned pair-index mask を、全 `K_TRUE`・両 estimand で共有する。

したがって A と B は data/model RNG で対応し、かつ anchor-aligned mask を共有する。
**ただしこれを `same dataset` とは書かない。**
各 `K_TRUE` の `Z` / `F` / `X` / `Y` はその `K_TRUE` に応じた generator output である。

## 17. Validation

| 項目 | 値 |
|---|---|
| **EM fits executed** | **0** |
| `run_em_experimental` 呼び出し | 0 |
| K_TRUE sweep 実行 | なし |
| 新規 generator simulation / probe | **なし**（本改訂では 1 件も実行していない） |
| Phase 7e 再実行 | なし |
| 変更した model / scientific code | なし |
| 変更した result CSV / runinfo / Phase 7e artifact | なし |
| 変更した canonical docs | なし |
| main への直接変更 | なし（`design/47-k-true-robustness` で作業） |
| commit / push / PR / Issue 変更 | なし |
| repository へ書き込んだファイル | 本 report と implementation plan の 2 件のみ |

**比較可能性判断の primary evidence は §5 の解析式であり、
シミュレーション由来の数値効果量に依存していない。**
