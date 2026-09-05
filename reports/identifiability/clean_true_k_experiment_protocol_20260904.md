# clean true-K n-sweep 実験プロトコル（事前登録・凍結）

**作成日:** 2026-09-04
**baseline main:** `7e335602999977060208ce37ac8cdff8fedfa66e`
**protocol hash:** `547880a16aef6530cfdf7903c4e32f16062397e0bacc0c109d5c77fb9892ccc0`
**runner:** `tools/research_audit/run_clean_true_k_sweep.py`
**machine-readable:** `expfam/results/k_selection/clean_true_k_asymptotics_20260904/protocol.json`（実行時に生成）

**本プロトコルは実行開始前に凍結された。結果を見てから変更しない。**

---

## 1. PRIMARY QUESTION

> **canonical clean generator による well-specified な有限標本設定において、
> `n` を増加させたとき、事前定義した各 K-selection criterion の selected-K パターンは
> どのように変化するか。**

**これは consistency theorem ではない。** 本実験は記述的な有限標本観測であり、
`n → ∞` の一致性については何も示さない（理論監査 §16b・U6）。

---

## 2. 事前に確定している解釈上の限定

理論監査（`true_k_identifiability_hardened_20260904.md`）と敵対レビューの結論を、
**結果を見る前に**ここへ固定しておく。

| # | 限定 | 根拠 |
|---|---|---|
| L1 | **`K_TRUE` と `K*` は別物。** 本実験の「真値一致」は `K_TRUE`（generator の設定）との一致であって `K*` との一致ではない。`K* < K_TRUE` の可能性は family ごとに未検証 | §2.3・U9 |
| L2 | **`family_y = bernoulli` は理論的に未解決領域である。** P2（識別性）・P3（非入れ子性）はいずれも **Gaussian-Y 限定**で、Bernoulli-Y では識別性（U2）も非入れ子性（U5）も未証明 | §10.4・§12.4 |
| L3 | **S1（held-out）が何を選んでいるかは未解決。** plug-in raw-eta score は proper scoring rule ではなく、その population target は特定されていない | U10・§17.3 |
| L4 | **S2 を「Schwarz BIC」と呼ばない。** `Q_strict` は観測データ周辺尤度ではない | KI-010・KI-019 |
| L5 | **S3 は原論文 Eq.(26) ではない。** 原論文の評価手続きは本文から特定不能なので、alignment は主張しない | `paper_bic_reproduction_alignment_20260904.md` |
| L6 | **S4 は criterion ではない。** 有限標本の rank 閾値を事前に固定する根拠がないため、selected K を作らない | U7・§7.5 |
| L7 | **有効標本数が未定義。** ノード数 `n` / dyad 数 `n(n−1)/2` / X 要素数 `nd` のどれが効くかは未確定。S2・S3 の penalty は `log n`（ノード数）を使う | §16b |
| L8 | **lineage E（experimental prototype）である。本文採用不可** | root `CLAUDE.md` §3 |

---

## 3. 生成モデル（canonical clean generator）

`expfam/src/experimental/data_generator_canonical.py`（`generator_version = canonical-clean-v1`）。
**historical generator は使わない。**

| 項目 | 値 |
|---|---|
| `family_x` / `family_y` | `poisson` / `bernoulli` |
| `Z` | iid `N(0, I_K)`、**正規化なし** |
| `F` | reduced QR による構成で `rank(F) = K` を保証。**行正規化なし・seed rescue なし** |
| `d` | 15 |
| `w0_true` | −1.0 |
| link | canonical。**clip なし・unsafe なら FAIL FAST** |

### 3.1 K_TRUE 間の信号整合（交絡回避のための設計判断）

`K_TRUE` を変えると X 側・Y 側の信号強度が自動的に変わってしまうため、両方を固定する。

```
f_scale = sqrt( row_norm_sq_target * d / K )     → 平均 ||f_l||^2 = 0.5（K によらず一定）
w_K     = w_ref * sqrt( K_ref / K )              → w^2 K = 1.0 * 3（K によらず一定）
```

- `row_norm_sq_target = 0.5`、`w_ref = 1.0`、`K_ref = 3`。
- 根拠: `Var(z_i^T z_j) = K`（理論監査 §9.2）、`E[X_l] = exp(||f_l||²/2)`（§7.1）。
- **`row_norm_sq_target = 0.5` は runtime benchmark と同時に、結果を見る前に選んだ**
  （X の平均カウント ≈ 1.3、最大 ≈ 20–36、ゼロ率 ≈ 37%、Y density ≈ 0.33 で K 間に整合）。

**この整合自体が設計判断である。** `f_scale` を固定して信号が K とともに増える設計とは
別の結果になりうる。本実験は前者を選んだ。

---

## 4. 実験グリッドと fit 予算

| 項目 | 値 |
|---|---|
| `K_TRUE` | **1, 3, 5**（`K_TRUE = 5` を primary focus） |
| `n` | **50, 75, 100, 150** |
| candidate `K` | 1, …, 7 |
| starts | 1, 2 |
| replicates | `K_TRUE=1`: 4 / `K_TRUE=3`: 4 / **`K_TRUE=5`: 8** |
| **TIER** | **A** |
| cells | 64 |
| **fits** | **896**（K_TRUE=1: 224 / 3: 224 / 5: 448） |

### 4.1 TIER の決め方（runtime のみ）

事前に定義した 3 つの TIER から、**wall-clock 推定が約 3.5〜4 時間以内に収まる最大の TIER** を選んだ。
選択結果や科学的結果は**一切参照していない**。

| n | 実測 平均秒/fit |
|---:|---:|
| 50 | 4.684 |
| 75 | 7.237 |
| 100 | 9.980 |
| 150 | 16.629 |

| TIER | fits | 推定 wall-clock |
|---|---:|---:|
| **A（採用）** | **896** | **2.40 h** |
| B | 616 | 1.65 h |
| C | 504 | 1.35 h |

benchmark は `k_est ∈ {1,4,7}` で計測（平均 4 は `1..7` の平均と一致）。
**TIER は protocol.json へ記録した後は固定する。**

---

## 5. EM 設定

| 項目 | 値 |
|---|---|
| lineage | `DualExpFamLSMConsistent`（objective-consistent、`numerics_mode="consistent"`） |
| **旧 0.5 lineage** | **使用しない** |
| `L` | 5 |
| `num_iter` | 8 |
| `compute_strict_Q` | True |

`L = 5`・`num_iter = 8` は Phase 7e / 8b と同一であり、比較可能性のために踏襲する
（ただし generator が異なるので数値を同じ表に並べない）。

---

## 6. Y holdout

- **test_ratio = 0.20**、dyad（上三角ペア）単位。
- split は `(K_TRUE, n, replicate)` ごとに決定論的 seed で固定。
- **同一 cell 内では candidate K・start をまたいで同一 mask を使う。**
- train / test の重なりがないこと、mask が対称かつ対角 False であることを実行時に検証する。

---

## 7. seed 規約（すべて cell index の純関数）

```
data_seed  = 810000 + K_TRUE*10000 + n*10 + replicate
split_seed = 820000 + K_TRUE*10000 + n*10 + replicate
model_seed = 830000 + K_TRUE*100000 + n*1000 + replicate*100 + K*10 + start
```

model seed の一意性は manifest 構築時に検証する。

---

## 8. 選択基準（同一の fit 証拠から複数を保存する）

### S1 — held-out predictive（PRIMARY）

Phase 7e / 8b と**同一定義**（`heldout_bernoulli_mean_log_score` を再利用）。

```
eta_ij      = w0_hat + w_hat * z_i^T z_j        （推定 Z を代入した plug-in）
score_ij    = y_ij * eta_ij - logaddexp(0, eta_ij)
fit score   = held-out test dyad 上の平均
Sbar(K)     = 2 start の非加重平均
tie 候補    = max_K Sbar(K) - Sbar(K) <= 1e-12
selected K  = tie 候補の最小 K
```

**これは「未知 Y の予測に最適な K」を評価する。** 最小の真の潜在次元 `K*` と一致する保証はない（L3）。

### S2 — Q-based criterion（SECONDARY）

`-2 Q_strict + p log n`（`em_runner` の返す `bic` = `calc_bic_exp`）。

- `p = kd − k(k−1)/2 + [d if X gaussian] + [1 if Y gaussian]`。本実験では `p = 15k − k(k−1)/2`。
- **`w0, w` は数えられていない**（NOLTA 2024 の慣行）。理論監査 §17.1 に記録済み。
- **「Schwarz BIC」「true BIC」「correct BIC」と呼ばない**（L4）。
- 罰則付き逸脱度なので**小さいほど良い**。凍結 selector には符号を反転して渡す。

### S3 — plug-in conditional criterion（SECONDARY）

```
-2 * [ ln p(X | Z_final, F) + ln p(Y_train | Z_final, w0, w) ] + p log n
```

- `ln p(Z)` を**含めず**、`Z` を**積分しない**。これが S2 との違い。
- **原論文 Eq.(26)/Eq.(16) ではない。** 原論文の評価手続き（どの `Z` か、MC 平均をとるか）は
  本文から特定不能であり `[UNRESOLVED]`（L5）。本基準は**本モジュール内で完結に定義した第 3 の基準**であって、
  paper alignment は一切主張しない。**BIC と呼ばない。**
- 小さいほど良い。符号反転して selector に渡す。

### S4 — Poisson-X Gram spectrum（構造診断・criterion ではない）

理論監査 P1 の population 恒等式を標本モーメントで置き換えた推定 Gram

```
G_hat[l,l] = 2 log mean(X_l)
G_hat[l,m] = log( mean(X_l X_m) / (mean(X_l) mean(X_m)) )
```

の**固有値・最小固有値・閾値なし階数・固有値ギャップ比**を保存する。

- **selected K を作らない。** rank 閾値を事前に固定する根拠がないため（L6）。
- **結果を見て閾値を決めることを禁止する。**
- 推定 Gram は PSD 錐の外に出ることが理論監査 §7.5 で確認済み。負の固有値もそのまま記録する。

---

## 9. 失敗時の扱い（FAIL CLOSED）

fit が 1 つでも失敗したら **run 全体を停止**し、`failure.json` に
`attempted_fit_count` / `completed_fit_count` / `failed_fit_index` / `error` /
`retry_count=0` / `replacement_fits_executed=0` / `run_code_sha` を保存する。

**禁止:** retry / 別 seed での再試行 / 失敗 cell だけの再実行 / replacement fit /
threshold 緩和 / tolerance 緩和 / replicate 削除 / 都合の良い結果だけの報告 /
結果を見た後のプロトコル変更 / 黙った resume・partial reuse。

科学的なコードバグが判明した場合は、現在の run を FAILED として保存し、修正・レビューののち
**新しい attempt id で fit 1 から全体を再実行**する。

artifact ディレクトリが既に存在する場合、runner は **overwrite も resume も拒否して停止する。**

---

## 10. 成果物

```
expfam/results/k_selection/clean_true_k_asymptotics_20260904/
  protocol.json              凍結プロトコルと hash
  manifest.csv               896 行（fit_index 1..896）
  generator_provenance.csv   64 cell の生成 provenance と mask hash
  fit_results.csv            896 行の per-fit 記録（S1/S2/S3 の生値）
  selection_matrix.csv       64 cell x 3 criterion = 192 行
  gram_spectrum.csv          64 行の S4 診断
  summary.json               criterion 別・cell 別の集計
  runinfo.json               実行 provenance
```

---

## 11. 事前に決めた分析（結果を見る前）

- `K_TRUE` × `n` × criterion ごとに selected K の一覧、真値一致数、mean selected K、
  under / over の内訳を報告する。
- **PRIMARY focus は `K_TRUE = 5`** における `n = 50 → 150` の変化。
- **criterion 間で selected K が食い違う場合、それをそのまま重要な結果として報告する。**

### 事前に禁止した書き方

改善しても「K-selection consistency を証明した」と書かない。
改善しなくても「K は識別不可能」と書かない。
いずれも有限標本の記述的証拠にとどめる（理論監査 §18 の claim ledger に従う）。
