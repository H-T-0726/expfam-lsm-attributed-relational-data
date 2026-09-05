# EM 収束感度実験 — 設計・事前登録ドラフト（2026-09-05）

**status: REMEDIATION ROUND 2（Codex 第 2 次レビュー NO-GO への対応後）。protocol は DRAFT のまま、production・replication gate・freeze・pilot はいずれも実行していない。fit は 0 件。**

> **仕様の正本はこの文書ではない。**
> 実験仕様の単一の source of truth は
> `reports/k_selection/em_convergence_sensitivity_20260905_protocol.DRAFT.json`
> の **`protocol` オブジェクトだけ**であり、それは runner の contract から
> `--emit-draft` で生成される。**§1〜§16 は各ラウンド時点の設計記録**であって、
> 当時の scope（3 点 `n`、336 runs、2016 rows、5.40 h）や当時の estimand の言い回しを
> 含む。**それらは現行値ではない。**現行値は §18 と DRAFT protocol を見ること。
> 歴史的記録は書き換えず、前へ向かって上書きする（`.claude/rules/historical-records.md`）。

runner・auditor・test は実装済み（§16）。**fit は 0 件。**

| | |
|---|---|
| experiment_id | `em_convergence_sensitivity_20260905` |
| machine-readable draft（**仕様の正本**） | `reports/k_selection/em_convergence_sensitivity_20260905_protocol.DRAFT.json` |
| baseline（対照） | `expfam/results/k_selection/clean_true_k_asymptotics_20260904/`（896 fits、`num_iter=8`） |
| lineage | **E（`DualExpFamLSMConsistent`、experimental prototype）。本文採用不可** |
| PRODUCTION RUNS THIS PHASE | **0** |
| NEW PRODUCTION FITS THIS PHASE | **0** |

本書の数値は `tools/research_audit/analyze_em_budget_baseline.py`（読み取り専用）が
既存 artifact から生成する。手で書いた数値ではない。

---

## 1. Research question

> **Does relaxing the fixed 8-iteration optimization budget materially change the
> selected-K pattern under the otherwise frozen clean-generator protocol?**

「固定 8 iteration という最適化予算を緩和したとき、それ以外の条件を固定したまま
selected K がどの程度変化するか」。

**これは次のいずれでもない。**

- K-selection consistency の証明でも検定でもない
- true-K identifiability 理論の追加証明ではない
- 新しい criterion の提案ではない
- `K=5` を選ばせるための tuning ではない

`K=5` の一致数が増えなかったとしても、それは**失敗ではなく結果**である（§8 Case B/C/D）。

---

## 2. Motivation — 何が未解決なのか

clean true-K experiment の限界 10（敵対レビュー F-02）が、そのまま本実験の動機である。

> EM を全 K で `num_iter=8` / `L=5` の固定予算で打ち切っており、収束判定も収束診断も
> 記録していない。候補 K が大きいほど自由パラメータが多く、同一予算では相対的に
> 未収束になりやすい。したがって「誤りが under-selection に偏る」ことも
> 「`n` とともに改善する」ことも、criterion の性質ではなく最適化予算の `K` 依存性で
> 説明できる。**本実験はこの 2 つを分離していない。**

自由パラメータ数は artifact から直接読める（一次証拠）。

| candidate K | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `num_params` | 15 | 29 | 42 | 54 | 65 | 75 | 84 |

`K=7` は `K=1` の **5.6 倍**のパラメータを、**同じ 8 反復**で当てにいっている。

### 2.1 交絡が実際に効きうる形（既存 artifact の形状）

`K_TRUE=5` の候補 K 別平均スコア（`analyze_em_budget_baseline.py` 出力）。

S1 held-out mean log score（**高いほど良い**）:

| | K=1 | K=2 | K=3 | K=4 | K=5 | K=6 | K=7 | argmax |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| n=50 | −0.61729 | −0.61472 | −0.61775 | −0.63622 | −0.62441 | −0.66292 | −0.67090 | K=2 |
| n=75 | −0.61626 | −0.60182 | −0.59355 | −0.59444 | −0.60239 | −0.62812 | −0.64562 | K=3 |
| n=100 | −0.61265 | −0.58625 | −0.57703 | −0.56363 | −0.56083 | −0.57985 | −0.60273 | K=5 |
| n=150 | −0.61406 | −0.58614 | −0.56768 | −0.55078 | −0.54326 | −0.55685 | −0.57057 | K=5 |

`Q_strict`（高いほど良い）も同型で、argmax は n=50→K=3、n=75→K=4、n=100→K=5、n=150→K=5。

**曲線は必ずずピークの右側で落ちる。** その落ち方が

- (i) criterion が複雑さを正しく罰している（＝ criterion の有限標本特性）のか
- (ii) 高 K の当てはめが 8 反復では終わっていない（＝ optimization underfitting）のか

を、**既存 artifact は区別できない**（§3-A）。本実験はここだけを狙う。

### 2.2 既存の関連事実（再確認済み、一次証拠）

- 896 fits で `nan_occurred` **0 行**、`q_bic_failed` **0 行**
- 総 runtime **8821.7 s**、`n` に強く依存（平均 4.81 / 7.47 / 10.14 / 16.96 s）
- 候補 K 依存は弱い（8.63 → 10.45 s）。**K を増やしても計算時間はほぼ増えない**
  ＝ 8 反復という予算が K に対して等しく「浅い」ことの傍証

---

## 3. 設計上の質問への回答（コードと artifact から）

### A. `num_iter=8` で収束しているかを、現在の artifact だけから判定できるか

**判定できない。** `fit_results.csv` の列は

```
fit_index, k_true, n, replicate, k_est, start, data_seed, split_seed, model_seed,
heldout_mean_log_score, q_strict, s2_q_based, s3_plugin_conditional, num_params,
nan_occurred, nan_count, q_bic_failed, failure_reason, runtime_s
```

で、**per-iteration の量が 1 つもない**（`analyze_em_budget_baseline.py` の
`per_iteration_columns` は `NONE`）。`summary.json` / `runinfo.json` にもない。

production sweep は `run_em_experimental(..., compute_strict_Q=True)` を
`mstep_q_diagnostic` **なし**で呼んでいる（`run_clean_true_k_sweep.py` L547-552）。
したがって記録されているのは最終反復の値だけで、**軌跡の傾きが分からない**。

**欠けている diagnostic（本実験で新設が必要なもの）**

| 欲しい量 | 現在 | 備考 |
|---|---|---|
| 反復ごとの objective / Q | ✗（`mstep_q_diagnostic` を有効にすれば取れるが未使用） | §3-B |
| パラメータ変化ノルム | ✗ | 新規 instrumentation |
| held-out スコアの軌跡 | ✗ | checkpoint でのみ取る（§3-E の leakage 回避） |
| 収束フラグ / 停止理由 / 実反復数 | ✗ | 新規 |
| 反復ごとの runtime | ✗ | 新規 |
| 内部 retry の発生 | ✗（**構造的に不可能**） | F-08、§3-B |

**「8 反復で収束していた／していなかった」は、現時点でどちらも主張できない。**

### B. 現在の EM 実装から反復ごとに取れる情報

`expfam/src/experimental/em_runner.py` を読んで分類した。

**すでにコード上取れるもの**

- `mstep_q_history`（`mstep_q_diagnostic=True` のとき）:
  各反復の `{iteration, q_before, q_after, q_diff, decreased}`。
  **同一 `Z_samples`・同一 `train_mask`** で M-step 前後の strict Q を計算する。
  - **推定経路を一切変えない**ことを実測で確認した
    （`test_em_budget_nesting.py::test_n2_...`、`Z_est`/`F`/`w`/`w0`/`Q_strict` が bit-identical）。
  - **ただし反復をまたいだ Q 列は単調目的関数ではない**。E-step ごとに MC サンプルが
    変わるため。docstring 自身がそう明記している。M-step 単調性の検査にのみ使える。
- `verbose=True` の `w0` / `w`（stdout のみ、artifact 化されていない）
- 実行後の集約値: `Q_strict`, `bic`, `num_params`, `nan_occurred`, `nan_count`,
  `runtime_s`, `q_bic_failed`, `failure_reason`

**新規 instrumentation が必要なもの**

- パラメータ変化ノルム（`F`, `w`, `w0` の反復間差分）
- gradient norm — **取れない**。E-step の Newton solve
  (`reproduction/src/model.py::calc_eta_newton`) は勾配を返さない。返すように変えると
  歴史的 lineage と共有しているモデルクラスを変更することになるため、**本実験では採用しない**。
- `Z` の変化量、反復ごとの held-out スコア、反復ごとの runtime、収束フラグ、停止理由
- **内部 retry の可視化（F-08）**

**F-08 を一次コードで再確認した。** `em_runner.py` L155-163, L205, L281-283:

```python
for retry in range(max_retries + 1):
    newton_alpha = 0.5 / (2 ** retry)
    rng = np.random.default_rng(seed + retry * 1000)
    ...
    nan_count = 0                      # ← retry ごとにリセット
    ...
    nan_occurred = nan_count > 0
    if not nan_occurred:
        break                          # ← 最初に成功した attempt で抜ける
```

retry 0 で NaN が出て retry 1 が clean だった場合、返るのは `nan_count=0`,
`nan_occurred=False` で、**一度も NaN が出なかった run と区別できない**。
これは `test_em_budget_nesting.py::test_nan_retry_is_invisible_in_the_returned_counters`
としてテストに固定した。**新 runner はこれを継承してはならない**（§9）。

### C. `max_iter` を増やすだけで公平な比較になるか

**RNG 軌跡の観点では、なる。ただし無条件ではない。**

構造の確認（`em_runner.py`）: `rng = np.random.default_rng(seed + retry*1000)` は
**反復ループの外で 1 度だけ**生成され、以後 `calc_eta_newton(..., rng=rng)` が
逐次消費する。ループ内に `num_iter` に依存する分岐は存在しない。したがって
反復 `t` の時点の RNG 状態は反復 `1..t−1` にしか依存しない。

**仮定せずに実測した**（`test_em_budget_nesting.py::test_n1_...`、n=24 の小規模決定論テスト）:

- `num_iter=8` と `num_iter=16` を同一 seed で実行し、最初の 8 反復の
  `q_before` / `q_after` が **bit-identical** であることを確認
- 同一呼び出しの再現性（N3）も確認

**ただし成立条件が 1 つある。** NaN ガードは run 全体を `seed+1000` で**やり直す**。
長い run は NaN を踏む反復が多い分だけ restart 確率が高く、restart した run の
先頭 8 反復は baseline のそれでは**ない**。よって

> **nesting は「NaN restart が起きなかった run」についてのみ成立する。**

これは設計上の必須要件になる（§9 で `attempts_used > 1` を pairing failure として記録）。
baseline 896 fits では `nan_occurred` が 0 行なので、baseline 側は clean である。

### D. 「8 / 20 / 50 を別々に回す」 vs 「1 回の long run を checkpoint 評価」

**結果を見る前に、checkpoint 方式を選択する。**

| 観点 | 別々に回す | long run + checkpoint | 判定 |
|---|---|---|---|
| 計算量 | 8+12+16+24+32+40 = **132 反復/fit** | **40 反復/fit** | checkpoint（3.3×） |
| RNG trajectory | §3-C により一致するが、**一致は前提**になる | 同一ストリームなので**構造的に一致**。ずれようがない | checkpoint |
| 比較公平性 | 予算間で prefix が同じ「はず」 | prefix が同じ**である** | checkpoint |
| 実装侵襲性 | 既存 runner をそのまま呼べる（低） | 反復ループを持つ新モジュールが要る（中） | 別々に回す |
| 再現性 | fit ごとに 6 本の run。NaN restart が予算ごとに独立に起きうる | fit ごとに 1 本。restart は 1 回検出すれば全 checkpoint に効く | checkpoint |

**checkpoint 方式を採る。**唯一の弱点である「実装侵襲性」＝ baseline と別のコード経路に
なることは、§6 の **replication gate**（checkpoint 8 が archive と bit-identical で
あることを要求する）で潰す。逆に言えば、この gate を通らない限り production は認可しない。

criteria の評価（`calc_Q_dual_strict_exp` / held-out score / `s3_plugin_conditional`）は
いずれも **rng を受け取らず決定論的**であることをコードで確認済み。したがって
checkpoint 評価を挟んでも RNG ストリームは乱れない。

### E. adaptive convergence stopping を使うか、使うなら何で定義するか

**使わない。** 本実験の停止規則は「事前登録した固定反復数 40」であり、
adaptive stopping は**存在しない**。理由は 2 つある。

1. **stopping rule が K に有利／不利になる余地を、設計から消すため。**
   全候補 K が厳密に同じ反復数を受け取るなら、「停止規則が高 K を優遇した」という
   批判は成立しない。adaptive にした瞬間、収束量の K 依存性がそのまま
   予算の K 依存性に化ける — それは今まさに切り分けようとしている交絡である。
2. **data leakage を構造的に排除するため。** held-out スコアを停止にも選択にも使えば
   leakage になる。停止規則が無ければ、その経路自体が存在しない。

収束は**測るが、止めるためには使わない**。事前に定義できる量だけを使う。

- `param_rel_change` = `max( ‖ΔF‖_F/(‖F‖_F+1e-12), |Δw|/(|w|+1e-12), |Δw0|/(|w0|+1e-12) )`
  — スケール不変で、実装から直接取れる。`1e-12` は float64 の相対精度から決めた定数で、
  結果を見て決めたものではない。
- 「τ で budget-stable」= checkpoint 直前 4 反復の `param_rel_change` の最大値 < τ。
  **τ ∈ {1e-2, 1e-3, 1e-4} を全部報告する。**「これが収束閾値だ」と 1 つを選ぶことはしない
  （結果を見てから選ぶのを防ぐ唯一の方法は、選ばないこと）。

**採用しない候補と理由**

- Q の反復間差分 — E-step ごとに MC サンプルが変わるので単調列ではない（§3-B）。
  収束判定には使えない。
- gradient norm — 実装が返さない（§3-B）。
- held-out スコアの停滞 — leakage。

**declared limitation:** `L=5` は毎反復サンプルし直すため `param_rel_change` は
0 に収束しない。どんな τ も「MC ジッタの大きさ」を同時に測っており、しかもその
ジッタ自体が K に依存しうる。**本設計はこれを解決しない。** τ を 3 つ報告するのは、
この事実を隠さないためでもある。

---

## 4. 変えるもの／固定するもの

**変えるのは optimization budget だけ。**

固定（baseline から継承、`protocol.json` の値を読み出して使い、再入力しない）:
`canonical-clean-v1` generator / `family_x=poisson` / `family_y=bernoulli` /
`K_TRUE` / `n` / `d=15` / signal scale / `w` scaling / candidate K=1..7 /
train-test mask / data seed / split seed / model seed / starts=2 / `L=5` /
S1・S2・S3 の定義 / tie tolerance `1e-12` / 最小 K の tie rule /
2 start 非加重平均の集約 / パラメータ化 / `numerics_mode="consistent"`。

seed 式は baseline runner のものをそのまま使う:

```
data_seed  = DATA_SEED_BASE  + k_true*10000  + n*10   + replicate
split_seed = SPLIT_SEED_BASE + k_true*10000  + n*10   + replicate
model_seed = MODEL_SEED_BASE + k_true*100000 + n*1000 + replicate*100 + k_est*10 + start
```

**artifact から再現できない要素は無い**（seed base は `protocol.json` に、
mask は `build_masks(n, split_seed)` で決定論的に再構成でき、mask hash が
`manifest.csv` に記録されている）。推測に頼る箇所はない。

---

## 5. Scope — なぜこれが最小か

**336 EM runs**（= 3 n × 8 replicate × 7 candidate K × 2 start）。
baseline の 896 fits を 40 反復で全部やり直せば 3.3 倍以上になるが、そうしない。

| 軸 | 採用 | 事前登録した理由 |
|---|---|---|
| `K_TRUE` | **5 のみ** | 交絡が問題になっているのは `K_TRUE=5` の under-selection。`K_TRUE=1` は 3 重に交絡していて識別に使えず、`K_TRUE=3` は replicate が 4 しかない |
| `n` | **75, 100, 150** | under-selection 最強（S1 0/8）→ 遷移域（4/8）→ ほぼ回復（8/8）を 1 本で貫く 3 点 |
| `n=50` | **除外** | baseline の S1 一致は 2/8 で、**n=75 の 0/8 より弱い**。最極端ではないので、4 点目を足しても対比が増えない |
| replicate | **全 8**（間引かない） | 間引かないので cherry-picking 規則が要らない。これが最も安全 |
| candidate K | **1..7 全部** | 削れば「都合の悪い K を落とした」になる。削らない |
| start | **2 とも** | start 不一致を予算の関数として見るため |

**scope が result-informed であることを明示する。** `K_TRUE=5` とこの 3 つの `n` を
選んだのは baseline の結果を見たからである。これは避けられない（問いがその結果に
ついてのものだから）が、その代わりに本実験は**独立検証ではなく targeted follow-up**
であり、規則は新しいデータが 1 つも存在しない今この時点で固定され、どんな結果でも
変更しない。

---

## 6. Primary comparison と pairing

**baseline arm**: archive の `num_iter=8` の結果。**再実行しない。**
**relaxed arm**: 同一データ・同一 mask・同一初期化で 40 反復の 1 本 run、
checkpoint `{8, 12, 16, 24, 32, 40}` で評価。

### Replication gate（pilot、production 前の必須関門）

| | |
|---|---|
| 目的 | 新 runner が baseline を checkpoint 8 で**完全再現**することの証明。これが通れば、以後の差は「予算の差」であって「コード経路の差」ではない |
| 範囲 | `K_TRUE=5`, `n=75`, replicate 1, 全 7 K × 2 start = **14 fits**、`max_iter=8` |
| 所要 | 約 **103 秒**（7.36 s/fit × 14） |
| 合格条件 | 14 行すべてで `heldout_mean_log_score` / `q_strict` / `s2_q_based` / `s3_plugin_conditional` / `num_params` が archive と**完全一致**。mask hash も一致 |
| 不合格時 | **BLOCKER。production 不認可。** 許容誤差を緩めない、近似比較に切り替えない、非 paired 設計に逃げない |
| freeze rule | protocol は pilot の**前**に hash して凍結する。pilot は PASS / FAIL しか返せず、protocol の項目を変更できない。変更が必要になったら新 version として凍結し直し、pilot を最初からやり直す |

### Primary outcomes

- **P1 — selected K の変化。** 各 (criterion, n, replicate) の selected K を
  checkpoint ごとに算出。selector は凍結のまま。checkpoint 8 と 40 を replicate 内で対応づけ。
- **P2 — 候補 K ごとの追加反復による score gain。**
  `Δ_K = mean S1 score(K, 40) − mean S1 score(K, 8)`。
  **`Δ_K` が K とともに増えるか**が、2 つの説明を最も直接分ける量である。
  表全体と、各 `n` での K と `Δ_K` の Spearman 相関を報告する。
  **7 点しかないので記述統計であり、検定も p 値の閾値も置かない。**

### Secondary outcomes

exact / under / over 件数（over は個別に名指し）、winning margin、start 不一致、
候補 K・`n` 別の budget-stable 率、`Q_strict` と S2 の軌跡、
checkpoint ごとの累積 runtime、S2 / S3 版の P1・P2。

---

## 7. Runtime 見積もりと pilot

`analyze_em_budget_baseline.py` による（既存 artifact のみ）。

baseline の当該 scope（336 fits）は 8 反復で **3886.7 s**。反復数に線形と仮定した投影
（末尾の Q/BIC/S3 計算は反復数に比例しないので、**線形仮定は過大評価側**＝保守的）:

| `max_iter` | 16 | 24 | 32 | **40** | 50 |
|---|---:|---:|---:|---:|---:|
| 投影時間 | 2.16 h | 3.24 h | 4.32 h | **5.40 h** | 6.75 h |

**40 を採る。** baseline の 5 倍の予算で、`K=7`（84 パラメータ）に対しても
`K=1`（15 パラメータ）の 8 反復相当より十分深い。50 との差 1.35 h に見合う
追加情報は無いと事前に判断した。

checkpoint 6 回分の評価コストは別途上乗せとして見込む（各回 `Q_strict` 1 回・
held-out 1 回・S3 1 回、いずれも決定論的）。

**abort 規則**: 実測が投影の 2 倍を超えたら停止して `failure.json` を書く。resume しない。

---

## 8. Interpretation matrix（結果を見る前に確定）

| Case | パターン | 読み方 | 禁止 |
|---|---|---|---|
| **A** | `Δ_K` が K とともに増え、selected K が `K_TRUE` 側へ動く | **fixed-budget optimization confounding の証拠。**過去の under-selection を criterion 固有と解釈してはならない | 「criterion は正しかった」と書かない。言えるのは「8 反復の結果は予算律速だった」だけ |
| **B** | objective / 収束診断は改善するが selected K はほぼ不変 | under-selection は 8 反復打ち切りだけでは説明しにくい | **criterion 原因と断定しない。**`L`・MC 床・初期化は依然として未制御 |
| **C** | 高 K も 8 反復時点ですでに budget-stable | F-02 の optimization explanation はこの設定では弱まる | `K_TRUE=5` とこの 3 つの `n` を超えて一般化しない |
| **D** | 長い最適化で selected K が不安定化 / over-selection 増加 | optimization と criterion の相互作用。**相互作用として報告する** | どの方向にも「改善」という語を使わない |

**4 つとも科学的結果である。** `K=5` が増えなかったから失敗、とはしない。
どの結果も protocol 変更・再実行・seed 変更・arm 追加の理由にならない。

---

## 9. 新しい artifact 要件

> **この節は 2026-09-05 の設計時点の列案である。実装された正本のスキーマは
> §17.4（M-04）で述べたとおり `--emit-draft` が生成する DRAFT protocol の
> `artifact_schema` であり、テストが `CHECKPOINT_COLUMNS` / `FIT_COLUMNS` との
> 完全一致を強制する。当時の案には `q_before/q_after/q_diff/mstep_decreased`
> が含まれていたが、これらは保存しないと決めたので実装には無い。**

**checkpoint 行**（当時の案、fit × checkpoint = 2016 行）:
`fit_index, k_true, n, replicate, k_est, start, data_seed, split_seed, model_seed,
checkpoint_iteration, heldout_mean_log_score, q_strict, s2_q_based,
s3_plugin_conditional, num_params, param_rel_change, budget_stable_1e-2/1e-3/1e-4,
q_before, q_after, q_diff, mstep_decreased, cumulative_runtime_s,
nan_iterations_so_far, attempt_index`

**fit 行**（当時の案、336 行）:
`attempts_used, retry_events, stopping_reason, actual_iterations, nan_occurred,
nan_count_total_across_attempts, numerical_warnings, train_mask_hash,
test_mask_hash, generator_version, run_code_sha, working_tree_clean`

### F-08 の解決（本実験で初めて可能になる）

新 runner は反復ループを自分で持つので、NaN 事象と restart を
`retry_events = [{attempt, iteration, kind}, ...]` として全部記録できる。
`attempts_used > 1` の fit は **pairing failure** として明示的に記録し、
黙って吸収しない（§3-C の nesting 条件が破れているため）。

**取らないもの（理由つき）**

- gradient norm — E-step が返さない。返させるには歴史的 lineage と共有している
  モデルクラスの変更が要るので、**本実験では取らない**
- 反復ごとの held-out スコア — checkpoint でのみ取る。毎反復の held-out トレースは
  停止シグナルへの転用を誘発し、leakage になる

---

## 10. Failure policy

以下のいずれかで**即停止**、`failure.json` を書いて終了。retry しない、resume しない、
seed を変えない、tolerance を緩めない、protocol を変えない。

- replication gate 不合格
- 非有限の criterion 値
- `attempts_used > 1`（NaN restart）が発生 — 当該 fit を pairing failure として記録
- 実行時間が投影の 2 倍超
- artifact ディレクトリが既に存在

---

## 11. Claim boundary

- lineage **E（prototype）**、`EXPERIMENT_REGISTRY` は `原稿採用 ✗`。
  **manuscript evidence への昇格は Human Gate**（root `CLAUDE.md` §6）
- 一致は **`K_TRUE`** との一致であって `K*` とのものではない
- 1 つの合成設定、`K_TRUE=5`、3 つの `n`、Poisson-X / Bernoulli-Y、信号 1 水準
- **確立しないもの**: 任意の criterion の consistency、Bernoulli-Y の識別可能性（U2）・
  非入れ子性（U5）、40 反復の到達点が MLE であること、`L=5` の MC 誤差が無視できること

---

## 12. 実装計画（まだ実装しない）

1. `expfam/src/experimental/em_runner_instrumented.py` — 反復ループを所有し、
   checkpoint 評価フックと `retry_events` を持つ新モジュール。
   **`em_runner.py` は変更しない**（歴史的 lineage）。
2. `tools/research_audit/run_em_convergence_sensitivity.py` — sweep runner。
   `--replication-gate` で pilot、`--full` で production（`--allow-em` と確認フラグ必須、
   artifact ディレクトリ存在時は fail-closed）。
3. `tools/research_audit/audit_em_convergence_sensitivity.py` — 独立監査。
   protocol hash 再計算、checkpoint 8 と archive の完全一致、fit 数・checkpoint 数の
   密度検査、runtime 会計、`attempts_used` 検査。
4. テスト。`test_em_budget_nesting.py` は**本フェーズで作成済み・6 件 pass**。

### 12.1 テストプロセスに関する制約（設計中に実測で判明）

Phase 8b のテストは `assert "em_runner" not in sys.modules` という
**プロセス全体のトリップワイヤ**を持つ（`test_k_true_robustness_sweep.py`
`test_S2b_real_adapter_is_never_reached_by_the_test_suite`）。研究監査スイート全体が
実 EM runner を一度も import しないことを保証するための防御であり、正しい。

本フェーズで最初に書いた nesting テストは `em_runner` を直接 import していたため、
**スイート全体を走らせると Phase 8b の 64 件が連鎖失敗した**（単体では 1194 件 pass）。
ガードを弱めるのではなく、こちらを**子プロセス化**して解決した
（`tools/research_audit/_em_budget_probe.py` を `subprocess` で起動し JSON を受け取る）。

**これは新 runner とその sweep runner にもそのまま効く制約である。**
`run_em_convergence_sensitivity.py` のテストは、実 runner をテストプロセスへ
import してはならない。設計に織り込む。

**この順序で、production はステップ 3 の gate 通過と人間の認可の後。**

---

## 13. 独立敵対レビュー（設計フェーズ）

Reviewer A（科学設計・統計）と Reviewer B（再現性・実装・leakage）の 2 名視点。
**BLOCKER 0 / HIGH 0 を満たすまで freeze 候補にしない。**

### Reviewer A — 科学設計・統計

| ID | severity | 指摘 | 対応 | 状態 |
|---|---|---|---|---|
| A-1 | **HIGH** | **scope が result-informed。**`K_TRUE=5` と 3 つの `n` は baseline を見て選ばれており、これを黙っていれば事前登録の体裁だけの後付けになる | §5 に「result-informed である」ことを**明示**し、targeted follow-up であって独立検証ではないと限定。protocol JSON にも `scope_is_result_informed: true` を置いた | **解消** |
| A-2 | **HIGH** | **`Δ_K` の単調性を「検定」と書けば誤り。**K は 7 点、replicate 8、相関の有意性を主張できない | P2 を**記述統計**と明記。Spearman は要約統計として報告し、p 値の閾値を置かないと事前登録 | **解消** |
| A-3 | MEDIUM | **`L=5` 固定が新しい交絡になる。**反復を増やしても MC 誤差は減らない。「40 反復で収束」は「MC ジッタの中に入った」に過ぎないかもしれない | §3-E と protocol の `monte_carlo_floor_caveat` に declared limitation として記載。τ を 3 つ報告。**解決はしない**と明言。L 感度は将来の別実験 | **限界として受理** |
| A-4 | MEDIUM | Case A を「criterion は正しかった」と読む誘惑がある | §8 の Case A に禁止文を明記 | **解消** |
| A-5 | MEDIUM | `max_iter=40` の根拠が弱い。40 で収束していなければ Case C と Case B が区別できない | 40 は「baseline の 5 倍」という事前根拠で選択。**40 でも budget-stable にならない fit があれば、それ自体を結果として報告する**（Case B/C の判別不能を隠さない）と §8 に含意。protocol の `tau_ladder` で可視化される | **解消** |
| A-6 | LOW | n=50 除外の理由は「弱いから」でよいのか | S1 一致 2/8 > n=75 の 0/8 という**一次証拠**に基づく。事前登録済み | **解消** |

### Reviewer B — 再現性・実装・leakage

| ID | severity | 指摘 | 対応 | 状態 |
|---|---|---|---|---|
| B-1 | **BLOCKER** | **新 runner は baseline と別のコード経路。**checkpoint 8 が baseline と一致する保証がなく、「予算の差」と「実装の差」が交絡する | §6 の **replication gate** を production の必須前提条件にした。14 fits で完全一致を要求し、不合格なら production 不認可 | **解消** |
| B-2 | **HIGH** | **nesting を仮定していないか。**「同じ seed なら prefix は同じ」は検証なしでは仮定 | `test_em_budget_nesting.py::test_n1_...` で `num_iter=8` と `16` の最初の 8 反復が **bit-identical** であることを実測（5 tests pass）。さらに**成立条件（NaN restart が無いこと）**を §3-C に明記 | **解消** |
| B-3 | **HIGH** | **NaN restart が pairing を静かに壊す。**長い run ほど restart 確率が高く、restart 後の prefix は baseline のものではない | `attempts_used > 1` を **pairing failure** として artifact に記録し、§10 で failure policy に含めた。F-08 の可視化（`retry_events`）を新 runner の要件にした | **解消** |
| B-4 | **HIGH** | **held-out スコアが停止と選択の両方に使われれば leakage** | **停止規則を置かない**設計にした（§3-E）。leakage 経路が構造的に存在しない。反復ごとの held-out トレースも取らない（§9） | **解消** |
| B-5 | MEDIUM | **stopping rule が K に有利／不利にならないか** | 停止規則が無く、全 K が厳密に同じ反復数を受け取る。収束量は測るが止めるのに使わない | **解消** |
| B-6 | MEDIUM | **収束基準自体が K 依存ではないか** | `param_rel_change` は相対量なのでスケールには不変だが、**MC ジッタの K 依存性までは消せない**。A-3 と同じ限界として declared。選択には一切使わないので、少なくとも結果を歪めない | **限界として受理** |
| B-7 | MEDIUM | `mstep_q_diagnostic` を有効にすると推定が変わるのではないか | 実測で bit-identical を確認（N2、5 tests pass）。ただし **`q_before/q_after` は同一 E-step 上の比較であり、反復間の Q 列は単調目的ではない**ことを §3-B に明記 | **解消** |
| B-8 | MEDIUM | replicate 間引きが cherry-picking になる | **間引かない**（全 8 replicate、全 7 K、全 2 start）。規則が要らないのが最も安全 | **解消** |
| B-9 | LOW | DRAFT protocol に hash が無い | **意図的**。hash を置けば freeze に見える。freeze は別 commit（§6 freeze rule、JSON の `status_note`） | **解消** |
| B-11 | **HIGH** | **新しいテストが Phase 8b の安全トリップワイヤを壊した。**`em_runner` をテストプロセスに import したため、スイート全体で 64 件が連鎖失敗した（設計中に実測。単体実行では露見しない） | ガードは弱めず、測定を**子プロセス**へ移した（§12.1）。制約として新 runner の実装計画にも明記 | **解消** |
| B-10 | LOW | runtime 投影が線形仮定 | 末尾の Q/BIC/S3 は反復数に比例しないので**線形は過大評価＝保守側**。script の docstring と §7 に明記。abort 閾値は投影の 2 倍 | **解消** |

**結果: BLOCKER 0 / HIGH 0。**
（B-11 は設計中に自分のテストが引き起こした実害であり、スイート全体を走らせなければ見えなかった。単体実行だけで済ませないこと。）
未解決として受理したのは A-3 / B-6（`L=5` の MC 床と収束量の K 依存性）で、
いずれも**本設計が解決しないと明示した限界**であり、選択には使わない量に限定されている。

---

## 14. この設計が「最小」である理由

- baseline 896 fits のうち、**交絡が問題になる 336 fits 相当だけ**を対象にした
- 予算の梯子を**別々の run にせず 1 本の run の checkpoint**にして、計算量を 3.3 分の 1 にした
- **新しい criterion も新しい generator も新しい family も導入しない**
- 変えるのは `num_iter` **1 つだけ**
- replicate も候補 K も start も**削っていない**ので、選択規則を作る必要がない

---

## 15. この段階で実行していないこと

- production 実験、新規 EM fit、pilot の実行（`replication gate` も**まだ走らせていない**）
- 既存 artifact の変更・再生成・削除
- protocol の freeze（本書と JSON は **DRAFT**）
- 新 runner / 新 auditor の実装
- push / PR / merge / Issue 操作

**§3 の検証に使った EM は、`n=24` の決定論的ユニットテスト 1 種類のみ**
（`test_em_budget_nesting.py`）。production fit ではなく、artifact も書かない。


---

## 16. 実装フェーズ（2026-09-05、設計レビュー承認後）

設計レビューで承認された内容を実装した。**protocol は DRAFT のままで、
production も replication gate も pilot も実行していない（fit 0 件）。**

### 16.1 成果物

| 種別 | ファイル |
|---|---|
| instrumented EM | `expfam/src/experimental/em_runner_instrumented.py` |
| sweep runner | `tools/research_audit/run_em_convergence_sensitivity.py` |
| independent auditor | `tools/research_audit/audit_em_convergence_sensitivity.py` |
| tests | `tools/research_audit/test_em_convergence_sensitivity.py`（30 件）、`test_em_budget_nesting.py`（11 件）、`_em_budget_probe.py`（子プロセス） |

**`expfam/src/experimental/em_runner.py` は byte 単位で不変。** 既存の共有実装は
書き換えず、反復ループを持つ別モジュールを新設した（レビュー §4 の指示どおり
experimental wrapper / instrumentation path を優先）。

### 16.2 Primary comparison の確定

**checkpoint 8 対 checkpoint 40 が primary。** 12/16/24/32 は trajectory を記述する
secondary diagnostic であり、**40 反復を「converged」とは呼ばない**。
protocol JSON の `intermediate_checkpoints_are_secondary` に明記した。

### 16.3 収束診断の hardening（レビュー §2）

raw parameter change だけを唯一の指標にしない。**潜在空間は直交変換までしか
決まらない**ので、`F` 自体は分布が動かなくても大きく動きうる。回転不変な
Gram 変化を主指標にした。

| 列 | 定義 | 役割 |
|---|---|---|
| `gram_rel_change` | `‖F_tF_tᵀ − F_{t−1}F_{t−1}ᵀ‖_F / (1 + ‖F_{t−1}F_{t−1}ᵀ‖_F)` | **主指標。**回転不変 |
| `f_raw_change` | `‖F_t − F_{t−1}‖_F / (1 + ‖F_{t−1}‖_F)` | 保存はする。**単独で「収束」と認定しない**（回転を変化として数える） |
| `w_change` | `\|Δw\| / (1 + \|w_{t−1}\|)` | mixed absolute-relative |
| `w0_change` | `\|Δw0\| / (1 + \|w0_{t−1}\|)` | 同上 |

**分母を `1 + |x|` にした理由**: 純粋な相対変化は パラメータ が 0 近傍を通過すると
発散する。`w` と `w0` にとってそれは異常ではなく日常なので、mixed
absolute-relative scaling で有界にした。閾値ではなくスケーリングの選択であり、
結果を見る前に固定してある。

4 つとも**個別の列**として artifact に保存し、直近 4 反復の最大値も
`max_*_last4` として保存する。budget-stable 判定は `gram_rel_change` で行い、
τ ∈ {1e-2, 1e-3, 1e-4} を**全部**列として持つ。

**すべて descriptive budget-stability diagnostics であり、formal convergence proof
ではない。** `L=5` を毎反復引き直すので、どれも固定点で 0 に収束しない。
auditor はこれを LOW finding として毎回明示的に記録する。

### 16.4 Replication gate（レビュー §3）

| 項目 | 内容 |
|---|---|
| セル選択規則 | **最小の scoped `n` の replicate 1**。コード上 `GATE_N = min(N_GRID)` として導出され、結果を見て選び直せない |
| 対象 | `K_TRUE=5, n=75, replicate=1`、7 K × 2 start = **14 fits**、`max_iter=8` |
| 比較フィールド | `data_seed` / `split_seed` / `model_seed` / `num_params` / `heldout_mean_log_score`(S1) / `q_strict` / `s2_q_based`(S2) / `s3_plugin_conditional`(S3) / `train_mask_hash` / `test_mask_hash` |
| selected-K 再計算に必要な値 | S1・S2・S3 の raw 値と `num_params` は**凍結セレクタの入力そのもの**。ここが一致すれば selected K は構成上再現される |
| 一致規則 | **全フィールド完全一致（tolerance なし）** |
| 一致規則の根拠 | csv は Python float を repr で書き、`float()` で無損失に往復する。つまり bit-identical 比較が**技術的に可能**である。可能なのに tolerance を置けば、それは実際の食い違いが隠れる場所にしかならない |
| 除外 | `runtime_s` のみ（wall-clock で再現不能） |
| 不合格時 | **PRODUCTION AUTHORIZATION = NO** で停止。比較を緩めない、tolerance に切り替えない、非 paired 設計に逃げない |
| production 後 | auditor が**最終 scope の全 fit** について checkpoint 8 を旧 artifact と再照合（`replication_vs_baseline`、不一致は BLOCKER） |

テストで、archive の値を 1 ulp ずらしただけで gate が落ちることを確認済み
（`test_gate_comparison_detects_a_mutated_value`）。

### 16.5 NaN / internal retry（レビュー §4）

新 runner は production で `on_nan="fail"`。NaN 潜在サンプルを検出したら
**restart せずに `RetryDetected` を送出**し、sweep がそれを fit failure として
扱う。restart は `seed+1000` で再シードするので、その run の先頭反復はもはや
baseline が辿ったものではなく、当該 fit の pairing は無効になる。**隠さず
失敗として扱うのが正しい扱い**であって、不都合だから消すのではない。

artifact には `retry_event` / `retry_count` / `retry_seed` / `stopping_reason` /
`nan_detected` / `attempts_used` / `actual_iterations` を記録する。**F-08 は
この runner では解決済み。**

### 16.6 Checkpoint invariance（レビュー §5、production gate）

子プロセス probe で実測し、テストに固定した。

| ID | 主張 | 結果 |
|---|---|---|
| **I1** | instrumented runner は baseline 予算で `em_runner` を**完全再現**（`Q_strict`・BIC・`num_params`・`Z`・`F`・`w`・`w0`） | ✅ |
| **I2** | 40 反復 run の checkpoint 8 は 8 反復 run 全体と一致 | ✅ |
| **I3** | checkpoint を 6 個入れた 40 反復 run は、40 だけで評価した run と**同じ終点** | ✅ |

I3 が「RNG を消費しない・optimizer/model state を書き換えない・trajectory を
変えない」の実測的な証明である。criteria 評価関数がいずれも `rng` を受け取らない
決定論的関数であることもコードで確認済み。**これは production gate であり、
落ちたら production しない。**

### 16.7 独立 auditor（レビュー §9）

`audit_em_convergence_sensitivity.py` は **runner を import しない**。
seed 式・mask 構成・S2 恒等式・セレクタ・期待件数をすべて protocol body から
独立に再記述しており、runner のバグが自分自身を承認できないようにしてある
（テストで import が無いことを検査）。

検査項目: 期待 fit / checkpoint 件数、`fit_index` の密性と重複、checkpoint ladder の
一致、seed 式の再計算、mask hash の再構成、非有限値、`s2 = −2·q_strict + p·log n`、
`num_params` が K のみの関数であること、retry / NaN メタデータの整合、
**checkpoint 8 と旧 artifact の完全一致**、raw 値からの selected K 独立再計算、
runtime 会計、protocol hash の再計算、**adaptive stopping が存在しないこと**
（`stopping_reason` が全件 `fixed_budget`、`actual_iterations == max_iter`）。

### 16.8 テスト（レビュー §10）

| 要求 | テスト | 結果 |
|---|---|---|
| 8 vs 40 prefix nesting | `test_n1_...`, `test_i2_...` | ✅ |
| checkpoint evaluation RNG invariance | `test_i3_...` | ✅ |
| checkpoint state nonmutation | `test_i3_...`, `test_i1_...` | ✅ |
| replication gate detects mutation | `test_gate_comparison_detects_a_mutated_value` | ✅ |
| auditor detects missing/duplicate checkpoint | `test_auditor_detects_a_missing_checkpoint` / `..._a_duplicate_checkpoint` | ✅ |
| auditor detects altered score | `test_auditor_detects_an_altered_score` | ✅ |
| auditor detects retry metadata mismatch | `test_auditor_detects_retry_metadata_mismatch` | ✅ |
| em_runner safety tripwire remains green | `test_no_em_runner_enters_this_process`, `test_the_runner_imports_em_lazily` | ✅ |

加えて auditor の改竄検出を 8 種（S2 恒等式・隠れた NaN・adaptive stopping・
短い run・偽装 seed・偽装 mask hash・改竄 protocol body・不可能な runtime 会計）
固定した。auditor の fixture は **checkpoint 8 に archive の実値**を使うので、
replication チェックが正しい理由で通る。

### 16.9 `n=50` を足した場合の追加 runtime（レビュー §6）

**既存 runtime artifact のみ**から算出（`run_em_convergence_sensitivity.py --plan-n`）。

| scope | fits | checkpoint rows | 投影時間 |
|---|---:|---:|---:|
| `n = {75, 100, 150}`（現案） | 336 | 2016 | **5.40 h** |
| `n = {50, 75, 100, 150}` | 448 | 2688 | **6.14 h** |
| **差分（n=50 の追加分）** | **+112** | **+672** | **+0.74 h（+14%）** |

**推奨: `n = {50, 75, 100, 150}` を採用する。** +0.74 h で
「result-informed n selection」批判が実質的に消える — 4 点にすれば baseline の
`n` グリッド**全体**になり、「どの `n` が under-selection したかを見てから
部分集合を選んだ」ではなくなるからである。A-1 は限界の宣言でしか対処できて
いなかったが、これは**設計で消せる**。

**ただし scope は勝手に変更しない。**DRAFT は `{75,100,150}` のままで、
4 点案を `scope.alternative_under_review` に併記した。**protocol freeze 前の
human decision とする。**

### 16.10 pilot 規則（レビュー §7）

gate は **runtime 確認・replication 確認・instrumentation correctness のみ**に使う。
**gate の selected K や K recovery を見て `max_iter` / checkpoints / candidate K /
replicate / `n` / stopping rule を変更してはならない。** runtime が事前 abort
閾値を超えた場合は**自動縮小せず human review へ戻す**。

### 16.11 実装フェーズの敵対レビュー（第 2 巡）

| ID | severity | 指摘 | 対応 | 状態 |
|---|---|---|---|---|
| C-1 | **BLOCKER** | 新 runner が baseline と数値的に一致する保証がない | probe I1 で `em_runner` との**完全一致**を実測し、テストに固定。実セルでは replication gate が再証明する | **解消** |
| C-2 | **HIGH** | checkpoint 評価が trajectory を変えていないか | I3 で 6 checkpoint 版と 1 checkpoint 版の**終点一致**を実測 | **解消** |
| C-3 | **HIGH** | 既存 `em_runner` を改造すれば全歴史結果の意味が変わる | **byte 不変**。別モジュールで mirror | **解消** |
| C-4 | **HIGH** | auditor が runner を import すれば自己承認になる | import 禁止をテストで強制。seed 式・mask・セレクタを独立再記述 | **解消** |
| C-5 | MEDIUM | `f_raw_change` だけを見ると回転を収束の失敗と誤読する | `gram_rel_change` を主指標にし、`f_raw_change` 単独使用を禁止と明記 | **解消** |
| C-6 | MEDIUM | `n` scope が result-informed のまま | §16.9 で 4 点案の追加コストを提示し**推奨**。決定は human gate | **提示済み・決定待ち** |
| C-7 | MEDIUM | `checkpoint_state_digest` は改竄検出に足りるか | 足りない。**state が一致したことの補助**であって改竄防止ではないと明記。実際の防御は checkpoint 8 の archive 照合 | **限界として受理** |
| C-8 | LOW | `L=5` の MC 床は依然未解決 | A-3 / B-6 と同じ。解決しないと宣言済み | **限界として受理** |

**結果: BLOCKER 0 / HIGH 0。**

### 16.12 このフェーズでも実行していないこと

- production 実験、新規 production fit、**replication gate の実行**、pilot の実行
- protocol の freeze（DRAFT のまま。`protocol_hash` を JSON に置いていないのも意図的）
- 既存 artifact の変更・再生成・削除
- push / PR / merge / Issue 操作

**実行した EM は `n=24` の決定論的 probe（子プロセス）1 種類のみ。**
production fit ではなく、artifact も書かない。


---

## 17. Codex 独立レビュー（NO-GO）への対応 — remediation phase

**EXECUTIVE VERDICT = NO-GO。** 科学設計は QUALIFIED で成立しているが、
production governance・auditor completeness・claim boundary に欠陥があるという判定。

**12 件すべてを一次コードで独立に再確認し、REJECT は 0 件、全件 ACCEPT した。**
以下は finding ごとの確認結果と対応。

### 17.1 BLOCKER

| ID | 判定 | 一次確認 | 修正 | 検証 |
|---|---|---|---|---|
| **B-01** scope mismatch | **ACCEPT** | DRAFT は `n={75,100,150}`、`alternative_under_review` に 4 点案が併記され、**scope の単一の source of truth が存在しなかった**。`analyze_em_budget_baseline.py` の `PROPOSED_N` も 3 点で独立に持たれていた | human decision に従い **4 点へ統一**。runner・DRAFT・design doc・runtime 分析・tests・auditor 期待値がすべて `n={50,75,100,150}` / **448 runs** / **2688 rows**。DRAFT は `--emit-draft` で runner の定数から**生成**するので二重定義が消えた | `test_scope_is_the_four_point_grid_everywhere`、`test_the_draft_on_disk_matches_the_runner_constants` |
| **B-02** production が freeze/gate/SHA/clean-tree を強制しない | **ACCEPT** | `run_production` は `allow_em` / `confirm` / ディレクトリ非存在の 3 つしか要求せず、**protocol を定数から再生成**し、clean tree を*記録するだけ*で要求せず、gate receipt も approved SHA も見ていなかった | `authorize_production()` を新設。**A–J の 10 条件**を全部満たさない限り fit を 1 件も開始しない。CLI フラグでは迂回できない | `test_the_approved_world_authorizes` ＋ 条件ごとの refusal テスト 11 件 |
| **B-03** failure policy が fail-closed でない | **ACCEPT** | 件数検査の `_require` が `try` の**外**にあり、失敗時に `failure.json` が書かれず artifact が中途半端に残った。checkpoint-8 照合は**記録するだけ**で、不一致でも COMPLETE artifact ができた | `attempt_id` と `attempt.json`（RUNNING → SUCCESS / FAILED）を導入。件数検査・全 fit 照合を `try` の**内側**へ移動。**最初の不一致で attempt 全体を FAIL**、残りの fit を続行せず COMPLETE artifact を作らない。**部分成功 fit は解析対象にしない**と明記 | `test_auditor_refuses_a_failed_attempt` / `..._an_interrupted_attempt` |

### 17.2 HIGH

| ID | 判定 | 一次確認 | 修正 | 検証 |
|---|---|---|---|---|
| **H-01** stochastic MC draws との交絡 | **ACCEPT** | 追加反復は毎回 `L=5` を引き直す。「optimization budget の効果」は分離できない | **estimand を狭くした**（§17.3）。禁止 claim を protocol body に列挙 | `test_the_estimand_is_narrow_and_names_its_forbidden_claims` |
| **H-02** production callback が invariance test に入っていない | **ACCEPT** | probe I3 は `on_checkpoint=None`。**実際の scoring は一度も検査されていなかった** | callback を module-level factory `make_checkpoint_callback` に切り出し、probe が**production と同じ関数**で 40 反復 × 6 checkpoint と terminal-only を比較 | RNG 終点・`Z`・`F`・`w`・`w0`・`Q`・終端スコア・state digest がすべて一致 |
| **H-03** latent/predictor movement を見ていない | **ACCEPT** | 診断は `F` / `w` / `w0` のみ。`F Fᵀ` は回転不変だが**潜在位置がどこへ行ったかは言わない** | `eta_x_rel_change`（`Z Fᵀ`）と `eta_y_rel_change`（固定 **train** dyad 上の `w0 + w zᵢᵀzⱼ`）を追加。**test dyad は 1 つも使わない**ので診断に held-out 情報が原理的に入らない | `test_the_instrumented_runner_records_rotation_invariant_diagnostics` |
| **H-04** checkpoint identity と fit identity が bind されていない | **ACCEPT** | auditor は `fit_index` の集合一致と ladder しか見ておらず、**identity 列を入れ替えても通った** | 各 checkpoint 行の 10 個の identity 列を**その fit 行と完全一致**で照合 | `test_auditor_detects_a_swapped_checkpoint_identity` / `..._seed` |
| **H-05** convergence 列 / later S1,S3 の検証不足 | **ACCEPT** | auditor が一度も参照しない列が **checkpoint 15 個 ＋ `retry_seed`** あった（機械的に列挙して確認） | 診断の有限性・非負性・窓最大値の代数・budget_stable フラグ・state digest・`retry_seed` を検証。**checkpoint state（Z,F,w0,w）を NPZ に保存**し digest と突き合わせ。later S1/S3 は artifact だけからは再計算できないので、**その限界を LOW finding として毎回出力**する | 該当 mutation test 8 件 |
| **H-06** primary outcome aggregation が未固定 | **ACCEPT** | P2 の「平均」の取り方（start 平均が先か replicate 平均が先か）が未定義だった | P1/P2 を protocol body に**式で凍結**。P2 は `delta(n,r,K) = 0.5·Σ_start [S1(40) − S1(8)]`、全 8 replicate を提示、**記述のみ・検定なし・p 値なし**。Spearman は diagnostic 限定、7 点を独立標本としない | `test_primary_outcomes_are_frozen_with_their_aggregation_order` |
| **H-07** 14-fit gate が global equivalence の証明にならない | **ACCEPT** | その通り。1 セルの一致は 448 fit の一致を含意しない | production は**全 448 fit** について checkpoint 8 到達時点で archive と照合し、1 項目でも不一致なら即 attempt FAIL。auditor も**全 fit**を再照合（`replication_coverage` が `compared == len(fits)` を要求） | `test_replication_comparison_is_exact` ほか |

### 17.3 Scientific estimand（レビュー §2）

**Primary estimand を狭くした。**

> Under the same data, masks, initialization, and first-eight-iteration RNG
> prefix, how does continuing the same `L=5` stochastic MCEM procedure from
> iteration 8 to iteration 40 change the selected-K pattern?

「同一データ・mask・初期値・最初の 8 反復の RNG prefix を固定した上で、
同じ `L=5` stochastic MCEM を 8 反復から 40 反復まで継続すると、
selected K がどのように変化するか」。

**禁止 claim（protocol body の `forbidden_claims` に列挙）**

- pure optimization effect
- 8 iteration caused the under-selection
- criterion was exonerated
- deterministic convergence
- consistency

**H-01 は設計上の残存交絡として明示**：追加反復は毎回 `L=5` を引き直すので、
**追加の最適化と追加の MC 変動は本設計では分離できない**（`residual_confound`）。
`materially changes` は materiality threshold を事前定義できないので**削除し、
単に changes とした**。

### 17.4 MEDIUM

| ID | 判定 | 対応 |
|---|---|---|
| **M-01** clean-path equivalence のみ | **ACCEPT** | 「line-for-line mirror」という過大表現を撤回。正確に **clean-path mirror for the frozen Poisson-X / Bernoulli-Y configuration** とし、**NaN / retry / exception path は historical runner と同一ではない**こと、production は最初の NaN で fail するので**科学的証拠は clean path のみ**であることを module docstring に明記 |
| **M-02** runtime projection | **ACCEPT** | 「線形は保守側」は誤り。**逆向きの 2 効果**（archive の固定終端コストを反復比例と数える＝過大、新 run は終端コストを fit あたり 6 回払う＝過小）を明記し、**bound ではなく estimate** と書き直した。2× 閾値は **operational safety stop であって scientific inclusion criterion ではない**と明記 |
| **M-03** environment provenance 不足 | **ACCEPT** | `environment_fingerprint()` を新設（Python / NumPy / SciPy / platform / machine / processor / BLAS / thread env vars）。gate receipt と production runinfo の両方に記録。**environment mismatch は refusal であって tolerance へ逃げる理由ではない**。等価判定は `python/numpy/scipy/machine` に限定（OS ビルド番号や shell 変数まで一致要求すると無意味な refusal が増えるため） |
| **M-04** design/protocol/schema drift | **ACCEPT** | design §9 は保存していない `q_before/q_after/q_diff/mstep_decreased` を列挙していた（**文書にあって実装にない**）。DRAFT を runner 定数から `--emit-draft` で**生成**する方式に変え、`artifact_schema` に実際の列を持たせ、テストで `CHECKPOINT_COLUMNS` / `FIT_COLUMNS` と**完全一致**を要求。以後 drift はテストが落ちる |

### 17.5 Production authorization（レビュー §3）

`authorize_production()` は 10 条件を個別に判定し、1 つでも欠ければ
`production_authorization = NO` を返して exit 1。**CLI フラグでは迂回できない。**

| | 条件 |
|---|---|
| A | immutable frozen protocol file が存在する |
| B | body-only canonical JSON の SHA256 が記録値と一致する |
| C0 | authorization record が同じ protocol hash に紐づいている |
| C | 現在の HEAD == approved `run_code_sha` |
| D | working tree clean |
| E | gate receipt が存在する |
| F | receipt の verdict == PASS |
| G | receipt の protocol_hash == frozen protocol_hash |
| H | receipt の code_sha == approved SHA |
| I | receipt の environment fingerprint が一致する |
| J | baseline artifact の protocol hash が期待値と一致する |

現時点の実行結果は **`production_authorization = NO`**（A・B・C0・C・D・E・F・G・H・I が未成立、J のみ成立）。

### 17.6 Gate receipt（レビュー §4）

`reports/k_selection/em_convergence_sensitivity_20260905_gate_receipt.json` を
機械可読で書く設計にした（**まだ実行していないので存在しない**）。
`receipt_version` / `experiment_id` / `protocol_hash` / `code_sha` /
`baseline_protocol_hash` / `environment` / `environment_key` / `gate_cell` /
`expected_fits=14` / `actual_fits` / `compared_fields` / `per_fit` /
`mismatch_count` / `mismatches` / `verdict` / `timestamp_utc` /
`retry_count_total` / `nan_detected_fits`。**production はこれを必須入力にする。**

### 17.7 Attempt lifecycle（レビュー §6）

`attempt_id` を導入。attempt は **SUCCESS か FAILED の 2 値**。
attempt failure の条件: replication mismatch / checkpoint-8 mismatch / NaN /
internal retry の発生 / 非有限スコア / checkpoint 欠落 / protocol・hash 不一致 /
dirty tree / SHA 不一致 / 中断 / runtime hard guard。

**FAILED attempt を resume して同じ attempt として完成させない。**
再実行するなら **new attempt_id ＋ new human authorization**。
auditor は `attempt.json` の status が `SUCCESS` でなければ **BLOCKER**。

### 17.8 Checkpoint state archival（レビュー §9）

checkpoint ごとに `Z` / `F` / `w0` / `w` を `checkpoint_states.npz`
（`savez_compressed`）へ保存し、CSV には identity ＋ `state_key` ＋
`checkpoint_state_digest` を置く。auditor は state を読んで digest と
パラメータを再計算する。

**容量見積もり（archive のみから算出）: 1,169,280 floats = 8.92 MB 非圧縮**
（`--plan` の `archived_state_megabytes_uncompressed`）。圧縮後はこれ以下。
既存 artifact 合計 1.36 MB に対して大きいが、**later checkpoint を独立検証
可能にする唯一の手段**なので採用を推奨する。過大と判断される場合は human
review へ戻す（digest だけ残す代替は、**later S1/S3 の独立再計算を放棄する**
ことを意味する）。

**残る限界**: S1 と S3 は state だけからは再計算できない（データ行列が要る。
再生成は可能だが artifact には含まれない）。auditor はこれを **LOW finding
として毎回明示**する。

### 17.9 未解決として残すもの

| 項目 | 理由 |
|---|---|
| `L=5` の MC 床 | 本設計は解決しない。診断はすべて descriptive |
| later checkpoint の S1 / S3 の完全独立再計算 | データ行列が artifact に無い。state・digest・S2 恒等式・診断代数までは検証する |
| `checkpoint_state_digest` の改竄耐性 | digest は「同じ state に到達した」ことの証拠であって改竄防止ではない（state を編集すれば digest も再計算できる）。実際の防御は archive との checkpoint-8 照合 |

### 17.10 対応後の残件

**BLOCKER 0 / HIGH 0 / MEDIUM 0 / LOW 3**（LOW はいずれも上表の宣言済み限界）。

### 17.11 このフェーズでも実行していないこと

- **replication gate の実行**（frozen protocol が無いので構造的に不可能。テストで確認済み）
- production 実験、新規 production fit、pilot
- protocol の freeze
- 既存 artifact の変更・再生成・削除
- push / PR / merge / Issue 操作

**実行した EM は `n=24` の決定論 probe（子プロセス）のみ。**


---

## 18. Codex 第 2 次レビュー（NO-GO）への対応 — remediation round 2

**7 件すべてを一次コードで再確認し、REJECT は 0 件、全件 ACCEPT した。**
以下は finding ごとの確認結果と対応。**現行の scope と estimand はこの節が正本の要約であり、
機械可読な正本は DRAFT protocol の `protocol` オブジェクトである。**

### 18.0 現行仕様（単一の source of truth からの要約）

| 項目 | 値 |
|---|---|
| `K_TRUE` | 5 |
| `n` | **50, 75, 100, 150** |
| replicates | 8 |
| candidate K | 1..7 |
| starts | 2 |
| `L` | 5 |
| `max_iter` | 40 |
| checkpoints | 8, 12, 16, 24, 32, 40 |
| **EXPECTED LONG RUNS** | **448** |
| **EXPECTED CHECKPOINT ROWS** | **2688** |
| **EXPECTED ITERATION ROWS** | **17920** |
| gate cell | `n = min(n_grid) = 50`, replicate 1, K=1..7, start 1,2 = **14 fits** |
| runtime | **約 6.14 h（operational estimate のみ。自動 abort なし）** |
| 状態保存 | 52.05 MB 非圧縮（`Z`, `Z_samples`, `F`, `sigma`, scalars） |

**Estimand（唯一の許容表現）**

> Under identical data, masks, initialization and first-eight-iteration RNG
> prefix, how does continuing the same `L=5` stochastic MCEM procedure from
> iteration 8 to 40 change selected K?

**禁止**: deterministic optimization effect / 8 iterations caused under-selection /
criterion exonerated / convergence / consistency。

### 18.1 BLOCKER

| ID | 判定 | 一次確認 | 修正 | 検証 |
|---|---|---|---|---|
| **B2-01** protocol の source of truth が 1 つでない | **ACCEPT** | DRAFT を機械的に調べたところ、`protocol.n_grid = [50,75,100,150]` の**隣に** legacy な `scope.n = [75,100,150]` と `scope.em_runs = 336` が残っていた。`emit_draft` が既存文書へ `update()` していたため。legacy top-level キーは **21 個** | `emit_draft` を**全面置換**方式へ変更。DRAFT の top-level キーを 9 個に固定し、実験仕様は `protocol` のみ。runner は `RUNNER_CONTRACT` から body を生成し、production は**凍結ファイルから全 execution field を読む**（option A）**うえで** contract と突き合わせて 1 つでも違えば拒否（option B も併用） | `test_the_draft_holds_exactly_one_specification`、execution field 31 個の mutation test |
| **B2-02** authorization / receipt chain が偽造可能かつ自己矛盾 | **ACCEPT** | `approved_run_code_sha` は**自分自身を含む commit** と一致せねばならず原理的に成立しない。gate receipt は tracked file で、commit すれば HEAD が動いて条件 C が壊れる | **trust anchor を human 作成の annotated git tag** に変更（tag 作成は HEAD も working tree も動かさない）。receipt は **`.local/research_gates/`（gitignore 済み）へ**。production は verdict だけでなく **receipt の全フィールド**を検証し、`environment_key` は receipt の environment オブジェクトから**再計算**して比較する | `test_the_approved_world_authorizes` ＋ 偽造 15 パターンの refusal test |
| **B2-03** 全 fit の checkpoint-8 照合が iteration 40 の後 | **ACCEPT** | `_run_cells` は `run_em_instrumented` の**戻り値**を照合していた。prefix の性質を最後に確かめるのは prefix check ではない | 照合を**callback の中**へ移動し、不一致で `PrefixReplicationError` を送出。EM ループは巻き戻り、**iteration 9 に入らない** | 子プロセス probe で `prefix_max_iteration_entered = 8`、`prefix_iteration_nine_entered = False`、`prefix_rows_kept = 0` を実測 |
| **B2-04** attempt / failure lifecycle が不完全 | **ACCEPT** | `ABANDONED` が無く、staging も無く、書き込みは非 atomic、停止した RUNNING の扱いが未定義 | attempt 専用 staging `.local/research_attempts/<exp>/<attempt_id>/`、状態機械 RUNNING → SUCCESS / FAILED / ABANDONED（**終端からの遷移は拒否**）、`os.replace` による atomic write、SUCCESS のときだけ最終 artifact へ export | 遷移テスト 4 件、auditor は SUCCESS 以外を BLOCKER |

### 18.2 HIGH

| ID | 判定 | 一次確認 | 修正 | 検証 |
|---|---|---|---|---|
| **H2-01** checkpoint state 不足 / later score を独立再計算していない | **ACCEPT** | NPZ は `Z, F, w0, w` のみで、**Q の計算に使った L 個の潜在サンプルが無い**ため Q は原理的に再計算不能だった | `Z_samples`（n×K×L）と `sigma`・`var_z` を追加保存（**52.05 MB 非圧縮**）。auditor は Poisson-X / Bernoulli-Y の対数尤度・`Q_strict`・S2 恒等式・S1・S3 を**すべて自前に書き直し**、再生成したデータと保存 state から**全 checkpoint を再計算**する | 実 fit との突き合わせで **S1・Q・S2・S3 の最大相対差 0.0**。later score の改竄 4 種が **audit FAILURE** |
| **H2-02** start disagreement が誤り / P2 未実装 | **ACCEPT** | 旧実装は `v[0] != v[1]`（**生スコアが違うか**）で、ほぼ常に真。P2 は仕様だけで実装が無かった | start ごとに**独立に selector を回して選ばれた K を比較**する定義へ修正。P2 を実装し、`audit_p2_replicate.csv`（224 行）と `audit_p2_summary.csv`（28 行）を出力 | スコアが違っても同じ K を選ぶ fixture で **FALSE**、違う K を選ぶ fixture で **TRUE**。P2 は手計算値との**厳密一致**を検証 |
| **H2-03** DRAFT / design に legacy scope・claim が残存 | **ACCEPT** | 上記 B2-01 のほか、design §§5-16 に 336 / 2016 / 5.40 h / gate `n=75` が残っていた | DRAFT は全面置換で legacy 0。design 冒頭に**「§1〜§16 は各ラウンド時点の設計記録であり現行値ではない」** banner を追加し、現行値は本節に集約。新規ファイル群には stale 用語・数値が存在しないことをテストで強制 | `test_current_files_carry_no_stale_scope_or_claim`（5 ファイル）、`test_the_draft_states_the_allowed_estimand_verbatim` |

### 18.3 runtime 規則（レビュー §6）

**自動 2× abort を科学 protocol から削除した。** runtime は estimand の一部ではなく、
**個々の K fit を含めたり除いたりする根拠に絶対にしない**。
`runtime_abort_rule = "none"` を protocol に明記し、auditor も検査する。
6.14 h は **operational estimate のみ**。手動・プロセス中断が起きた場合は
**attempt 全体が FAILED / ABANDONED** になり、部分的な科学解析はしない。

### 18.4 callback immutability の限定（レビュー §14）

callback には依然として live な `model` オブジェクトが渡る（S3 の評価に必要）。
配列（`Z_est` / `Z_samples` / `F` / `sigma`）は **read-only コピー**なので書き込みは
`ValueError` になり、これは実測済み。ただし **`model` 経由の変更まで構造的に
封じてはいない**ので、immutability は **qualified** として記載する。
実際の担保は「production と同じ callback を使った trajectory 一致テスト」である。

### 18.5 対応後の残件

**BLOCKER 0 / HIGH 0 / MEDIUM 0 / LOW 3。**

| LOW | 内容 |
|---|---|
| 1 | `L=5` の MC 床。診断はすべて descriptive で、固定点でも 0 に収束しない |
| 2 | auditor は `data_generator_canonical` を import する。**生成器自身の欠陥は捕まえられない**（他はすべて自前実装）。ただし最強の数値検査である archive との checkpoint-8 完全一致は生成器の正しさに依存しない |
| 3 | `checkpoint_state_digest` は改竄防止ではない。state を編集すれば digest も再計算できる。実際の防御は archive 照合と auditor の再計算 |

### 18.6 このラウンドでも実行していないこと

- freeze commit / freeze tag の作成
- replication gate の実行（frozen protocol も tag も無いので構造的に不可能）
- production、新規 production fit
- push / PR / merge / Issue 操作

**実行した EM は `n=24` の決定論 probe（子プロセス）のみ。**
