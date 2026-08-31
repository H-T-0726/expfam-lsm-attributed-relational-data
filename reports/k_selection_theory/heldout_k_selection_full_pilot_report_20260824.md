# Phase 7e full held-out K-selection pilot

本レポートは `expfam/results/k_selection/heldout_full_pilot_20260824/` に保存された machine-readable artifact から自動生成したものである。数値の手作業転記は行っていない。

## 1. 目的

Phase 7c (Issue #39) で設計し Phase 7d (Issue #41) で実装・falsification した leakage-safe held-out K-selection protocol を、候補 K = 1,...,7 と 3 dataset replicate へ拡張して **exactly once** 実行し、各 replicate における selected K を記述的に測定する。

**「K=3 が選ばれること」は成功条件ではない。** 目的は frozen protocol 下での選択挙動の測定であり、結果を見た後に seed / replicate / tolerance / score / K range / start count / split / preprocessing / failure rule を変更しない。

## 2. 固定条件

| 項目 | 値 |
|---|---|
| model lineage | `DualExpFamLSMConsistent`（objective-consistent experimental prototype、**本文採用不可**） |
| `family_x` | `poisson` |
| `family_y` | `bernoulli` |
| `K_TRUE` | 3 |
| `n` | 75 |
| `d` | 15 |
| `L` | 5 |
| `num_iter` | 8 |
| `numerics_mode` | `consistent` |
| `test_ratio` | 0.2 |
| candidate K | [1, 2, 3, 4, 5, 6, 7] |
| starts | [1, 2] |
| dataset replicates | [1, 2, 3] |
| total fits | 42 |

### seed convention

```
data_seed  = 41000 + replicate
split_seed = 42000 + replicate
model_seed = 43000 + replicate*1000 + K*10 + start
```

- data seeds: [41001, 41002, 41003]
- split seeds: [42001, 42002, 42003]
- model seeds: 44011 ... 46072（42 個、全て一意）

### primary score

```
eta_ij = w0 + w * z_i^T z_j
log score_ij = y_ij * eta_ij - logaddexp(0, eta_ij)
fit score = held-out upper test pairs 上の mean log score
```

`predict_mu_y` / probability clipping / threshold / rounding は使用していない。K 選択に BIC や `Q_strict` は使用していない。この score は plug-in であり、parameter・Z の不確実性を積分していない（posterior predictive・marginal likelihood・ELBO ではない）。

### selector

```
Sbar_r(K) = (start1 score + start2 score) / 2
tie candidate : max_K Sbar_r(K) - Sbar_r(K) <= 1e-12
selected K    : tie candidates の smallest K
```

`1e-12` は roundoff tie protection のみであり、統計的 equivalence threshold ではない。

## 3. リーク防止・provenance設計

- Design A（transductive dyad holdout）。node 集合は train/test で同一。held-out は Y の dyad のみ。
- split guard は **PAIR-MASK TOPOLOGY ONLY**。Y 値・prevalence・weighted degree・fit 品質を一切参照しない。
- **全 3 replicate の split を EM 開始前に生成・validate** し、全 PASS 後にのみ first EM fit を許可した。
- fit 側には `X` と `TrainingYValues`（train upper pairs のみ）しか渡らない。masked cell には finite かつ Bernoulli support 内の canary 値 0 を置く（`NaN * 0` を避けるため NaN は禁止）。
- `ScoreOnlyTarget` は各 replicate の **14 fit がすべて clean 完了しstored count/order gate を PASS した後に 1 回だけ**生成される。未完了 replicate の target は生成されない。
- replicate 内では x_hash / training_y_hash / train_mask_hash / test_mask_hash / fit_provenance_hash / target_topology_hash / score_target_hash / preprocessing_hash / score_config_hash がすべて一致することをexpected 側を再構築したうえで要素ごとに検証する（uniform corruption も検出される）。

### per-replicate provenance

| replicate | x_hash | train_mask_hash | test_mask_hash | score_target_hash |
|---|---|---|---|---|
| 1 | `7b70d89f1ee1` | `2ed450a0e792` | `387bda3b6122` | `150e6c43d371` |
| 2 | `f9838a7e47c4` | `ed37ddf1607a` | `cb94a1861b00` | `eda298e315fa` |
| 3 | `565ba209d1f7` | `dd38a6dc0c72` | `3e56e119afbb` | `77bb6de60558` |

（完全なハッシュは `runinfo.json` を参照。）

## 4. 実行情報

- issue: #43
- branch: `experiment/full-heldout-k-selection-pilot`
- RUN_CODE_SHA: `b9311e64a7b36c0a8a9704fff0ee7b38efe36a8a`
- base main SHA: `a11406ca5e93c216bd4faa875fdbe0ca73c406c6`
- command: `python tools/research_audit/run_heldout_k_selection_pilot.py --full --allow-em --confirm-full-pilot`
- UTC start / finish: 2026-08-23T18:23:52.927690+00:00 / 2026-08-23T18:28:59.333196+00:00
- local start: 2026-08-24T03:23:52.927702+09:00
- Python: 3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026, 13:03:48) [MSC v.1944 64 bit (AMD64)]
- NumPy: 2.3.5
- platform: Windows-11-10.0.26200-SP0
- expected fit count: 42
- **actual EM fit count: 42**
- score targets created: 3
- score rows: 42
- failure state: `none`

### per-fit hard gate

- internal retry 合計: 0
- warning 合計: 0
- Q failure 件数: 0
- NaN / nonfinite 件数: 0
- `fit_status != clean` の件数: 0

## 5. replicate別結果

| replicate | selected K | K=1 | K=2 | K=3 | K=4 | K=5 | K=6 | K=7 | best | second best | margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **3** | -0.596873 | -0.510742 | -0.460118 | -0.488560 | -0.551112 | -0.544203 | -0.576647 | -0.460118 | -0.488560 | 0.028442 |
| 2 | **3** | -0.592694 | -0.514030 | -0.469892 | -0.478528 | -0.505084 | -0.532329 | -0.559970 | -0.469892 | -0.478528 | 0.008636 |
| 3 | **5** | -0.589456 | -0.535913 | -0.485493 | -0.484102 | -0.483358 | -0.551123 | -0.577337 | -0.483358 | -0.484102 | 0.000744 |

margin は best と second-best の 2-start mean log score の差である。**統計的有意差ではない。**

- replicate 1: tie candidates = {3}
- replicate 2: tie candidates = {3}
- replicate 3: tie candidates = {5}

## 6. 集約結果

- n_replicates = 3
- K_TRUE = 3
- selected K counts = 3:2, 5:1
- K_TRUE selected count = 2
- **descriptive pilot recovery rate = 0.6667（2 / 3）**

この recovery rate は **3 replicate だけの記述的 pilot 結果**であり、統計的一貫性や一般的な true-K recovery を意味しない。

### K別 2-start mean log score の記述統計（3 replicate）

| K | mean | sample sd | min | max |
|---:|---:|---:|---:|---:|
| 1 | -0.593008 | 0.003719 | -0.596873 | -0.589456 |
| 2 | -0.520228 | 0.013682 | -0.535913 | -0.510742 |
| 3 | -0.471834 | 0.012799 | -0.485493 | -0.460118 |
| 4 | -0.483730 | 0.005027 | -0.488560 | -0.478528 |
| 5 | -0.513184 | 0.034596 | -0.551112 | -0.483358 |
| 6 | -0.542552 | 0.009505 | -0.551123 | -0.532329 |
| 7 | -0.571318 | 0.009834 | -0.577337 | -0.559970 |

## 7. 解釈

### 事実（artifact から直接読み取れること）

- frozen 42-row manifest どおりに EM fit が 42 回実行され、retry・warning・Q failure・NaN はいずれも 0 件だった。
- replicate 1 の selected K は 3 だった。
- replicate 2 の selected K は 3 だった。
- replicate 3 の selected K は 5 だった。
- 3 replicate 中 2 replicate で K = 3 が選択された。

### 解釈（推測を含む）

- 本 pilot は 1 つの synthetic 条件（family_x=poisson, family_y=bernoulli, n=75, d=15, K_TRUE=3）における選択挙動の記述である。
- held-out plug-in log score は K に対して単調ではなく、候補 K の間で有限標本上の予測性能を比較しているにすぎない。`K_hat_pred` は generative `K_TRUE` と一致する保証を持たない。

### 主張してはいけないこと

- 一般に true K を回復する
- K 選択の consistency を証明した
- BIC（`calc_bic_dual` / Q-based complete-data criterion）より優れる
- Phase 7b の C1/C2/C3 より優れる
- 実データでの妥当性
- 漸近的性質
- 修士論文・予稿レベルの最終結論

## 8. 制約

- **replication unit は独立生成 dataset replicate であり、本 pilot はわずか 3 個**である。recovery rate の分母は 3 であり、信頼区間を伴う推定量ではない。
- held-out dyad は node を共有するため独立ではない。held-out pair 数は独立標本サイズではない。
- score は plug-in であり、parameter・Z の不確実性を積分していない。
- 候補 K のモデルは回転不定性を持ち、操作的アルゴリズム上で入れ子とは限らない。
- MCEM の近似（L=5）と有限反復（num_iter=8）が予測ランキングに影響しうる。
- 使用した lineage は `experimental/` の objective-consistent prototype であり、`CLAUDE.md` §3 により **本文採用不可**である。
- transductive dyad holdout であり、新規 node に対する inductive 一般化（Design B）は現行 API では未サポートである。

## 9. 次の判断

- self-audit verdict: **PASS**
- BLOCKER 0 / HIGH 0 / MEDIUM 0 / LOW 0
- 独立再計算との最大差分: mean score 0.0, aggregate 1.734723475976807e-18

## 10. Post-pilot decision

### self-audit 結果（保存 artifact のみから独立再計算）

`tools/research_audit/audit_heldout_full_pilot.py` は pilot harness の selector を
import せず、Issue #43 に書かれた seed convention と selector rule だけから
manifest・selector・集約を再計算する。

- verdict: **PASS**
- BLOCKER 0 / HIGH 0 / MEDIUM 0 / LOW 0
- fit rows = 42、manifest rows = 42、duplicate keys = 0、missing keys = []
- retry 0 / warning 0 / Q failure 0 / NaN 0
- 独立再計算した selected K = {replicate 1: 3, replicate 2: 3, replicate 3: 5}
- 独立再計算した tie candidates = {replicate 1: [3], replicate 2: [3], replicate 3: [5]}
- 独立再計算した selected-K counts = {'3': 2, '5': 1}
- 独立再計算した K_TRUE selected count = 2
- 独立再計算した descriptive recovery rate = 0.6666666666666666
- runtime 値との最大差分: 2-start mean score **0.0**、K別集約 **1.734723475976807e-18**

### 非 blocking note

K 別集約統計の最大差分 `1.734723475976807e-18` は 0 ではない。これは
NumPy の pairwise summation と audit 側の純 Python 逐次加算の丸め順序差であり、
frozen tolerance `1e-12` を大きく下回る。**selector に入力される 2-start mean score
の差分は厳密に 0.0 であり**、selected K・tie set・
selected-K counts・recovery rate はいずれも完全一致した。この note は選択結果に影響しない。

### 判断

**A: REPORT_HELDOUT_PILOT_TO_ADVISOR**

理由:

- frozen 42-row manifest どおりに 42 fit がちょうど 1 回だけ実行され、全 fit が clean
  （internal retry 0 / warning 0 / Q failure 0 / NaN 0 / 非有限 0）だった。
- EM 開始前に 3 replicate すべての split を生成・topology guard 済みで、
  score target は各 replicate の 14 clean fit 完了後に 1 回ずつ、計 3 回だけ生成された。
- provenance gate（replicate 内 hash 不変性、manifest 完全性、seed convention）が
  独立再計算で PASS した。
- 保存 artifact（manifest / fit_results / replicate_selection / aggregate_summary /
  score_by_k / runinfo.json / runinfo.md / stdout.log）が揃っている。
- self-audit に BLOCKER・HIGH が 0 件。
- 本レポートは解釈境界を維持しており、consistency・BIC 優越・manuscript conclusion を
  主張していない。

**K=3 が選択された回数は、この A/B/C 判定の条件に用いていない。**

### 助言を求めたい点（結果そのものではなく設計上の論点）

- replicate 3 では selected K = 5、best と second-best の margin は
  `0.000744` であり、replicate 1 の `0.028442` より 1 桁以上小さい。
  この margin は統計的有意差ではなく、**現行 protocol は margin の大小を選択規則に
  一切使っていない**。margin をどう扱うか（報告のみに留めるか、将来の設計に組み込むか）は
  未決定である。
- replication unit が 3 個しかないため、recovery rate は記述値にとどまる。
  replicate 数を増やす場合、**本 pilot の結果を見た後に増やすことは事後的な設計変更**に
  あたるため、別 Issue で事前登録し直す必要がある。

---

## 付録: 本レポートの生成方法

```
python tools/research_audit/audit_heldout_full_pilot.py > audit.json
python tools/research_audit/build_heldout_full_pilot_report.py --audit-json audit.json
```

§1-§9 は上記コマンドで artifact から自動生成した。§10 は判断（A/B/C）のみ人手で選び、
記載した数値はすべて `audit.json` からプログラム的に埋め込んでいる。
