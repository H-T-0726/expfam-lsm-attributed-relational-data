# Phase 8b K_TRUE robustness full sweep — Attempt 2（成功実行の最終アーカイブ）

本レポートは `expfam/results/k_selection/k_true_robustness_full_attempt2_20260904/` の machine-readable artifact から自動生成した（`tools/research_audit/build_k_true_robustness_report.py`）。数値の手作業転記は行っていない。
各セルの `selected_k` は per-fit の生スコアから frozen Phase 7e selector で**再導出**し、`selection_matrix.csv` および `full_summary.json` と一致することを確認したうえで出力している（不一致なら生成が中断する）。

**このアーカイブ工程で実行した real EM fit は 0 である。**

## 1. 目的

Phase 7e (Issue #43) で凍結した leakage-safe held-out K-selection protocol を、
generator の `K_TRUE` を {1, 2, 4, 5} へ拡張して **exactly once** 実行し、各 replicate で選択される K を記述的に測定する。

**「真の K が選ばれること」は成功条件ではない。** 目的は frozen protocol 下での選択挙動の測定であり、結果を見た後に seed / replicate / tolerance / score / K range / start count / split / preprocessing / failure rule を変更しない。

## 2. 本実験で使った K 選択基準（誤記防止のため明示）

```
eta_ij      = w0 + w * z_i^T z_j
log score   = y_ij * eta_ij - logaddexp(0, eta_ij)
fit score   = held-out upper-triangle test pair 上の mean log score
Sbar(K)     = (start1 score + start2 score) / 2       # 非加重 2-start 平均
tie 候補    = max_K Sbar(K) - Sbar(K) <= 1e-12
selected K  = tie 候補のうち最小の K                   # frozen tie rule
```

- `score_config_hash` = `aae47803a349add9823cd36ed45cc9c2d81bbee435af4c9d64cbb6359de0fa8b`
- `frozen_config_hash` = `8ed6389bc69e4c79910ae40e75b863195fa2a8a4b31d0fb8ab65703c293bcfba`
- `1e-12` は roundoff tie protection のみであり、統計的 equivalence threshold ではない。

**Phase 8b の K 選択に次のものは一切使っていない:**
`Q_strict` / EM の Q 関数基準 / ICL-type complete-data criterion / Schwarz BIC / marginal likelihood / posterior predictive / ELBO。

この score は **plug-in** であり、parameter・Z の不確実性を積分していない。`predict_mu_y` / probability clipping / threshold / rounding は使用していない。

> 歴史的文脈（本実験の基準ではない）: `calc_bic_dual` は観測データの周辺尤度ではなく
> `Q_strict` を使う Q-based complete-data criterion / ICL-type であり、これを
> 「Schwarz BIC」と呼ばない（KI-010、`RESEARCH_MASTER.md` §12.6）。この論点は
> **legacy の基準に関するものであって、Phase 7e/8b の held-out 予測スコアとは
> 別物**である。両者を同一視しない（KI-019）。

## 3. 固定条件

| 項目 | 値 |
|---|---|
| model lineage | `DualExpFamLSMConsistent`（objective-consistent experimental prototype、**本文採用不可**） |
| `family_x` / `family_y` | `poisson` / `bernoulli` |
| `n` / `d` / `L` / `num_iter` | 75 / 15 / 5 / 8 |
| `numerics_mode` / `test_ratio` | `consistent` / 0.20 |
| new `K_TRUE` grid | [1, 2, 4, 5] |
| anchor `K_TRUE` | 3（Phase 7e artifact の **READ-ONLY 再利用**） |
| candidate K | [1, 2, 3, 4, 5, 6, 7] |
| starts / replicates | [1, 2] / [1, 2, 3] |
| mask_design / random_design / hierarchy | `S_C` / `CRN` / `H3_A` |
| 新規 fit 数 | **336**（A 168 / B 168） |

### estimand（A と B は分けて報告する）

| estimand | role | `w_true` の定義 | K_TRUE=1 | K_TRUE=2 | K_TRUE=4 | K_TRUE=5 |
|---|---|---|---|---|---|---|
| A | primary | `w = 1.5`（K_TRUE によらず固定） | 1.5 | 1.5 | 1.5 | 1.5 |
| B | sensitivity | `w_K = 1.5 * sqrt(3 / K_TRUE)`（`w_K^2 * K` を ensemble で一致させる） | 2.598076211353316 | 1.8371173070873834 | 1.299038105676658 | 1.161895003862225 |

`w0_true = -1.0`。seed base: data 51000 / model 530000 / anchor split 42000。

## 4. Provenance

| 役割 | SHA |
|---|---|
| role 1: approved scientific baseline | `68c78e1191889609dead05ea5a9fb11525ce92e2` |
| role 2: reviewed full-execution main | `ddc9b0b4c38da995fedf43ceef12f17dfb4db353` |
| role 3: runtime `run_code_sha` | `ef85b4c921546129b8d4f7440f8a09a41aa652e5` |
| Phase 7e anchor `run_code_sha` | `b9311e64a7b36c0a8a9704fff0ee7b38efe36a8a` |

- frozen protocol hash: `2d19c5fe6edadd0823925ed7dd051cb27837bccf51d5102e0bcee53271654eb9`（protocol origin: Issue #49）
- execution issue: #59 / execution attempt: `phase8b-full-attempt-2`
- prior aborted attempt: `phase8b-full-attempt-1`（reason: `operator_interrupt`、artifact: `expfam/results/k_selection/k_true_robustness_full_20260902`）
- `partial_results_reused` = `False`（Attempt 1 の部分結果は一切再利用していない）
- 実行時刻: `2026-09-04T00:07:50.578516+00:00` → `2026-09-04T00:48:21.764528+00:00`（UTC）
- `working_tree_clean_before_execution` = `True`

### artifact SHA-256

| ファイル | SHA-256 | bytes |
|---|---|---:|
| `audit_report.json` | `88f37ee1d02160c42ba36396c2d19149c1e7b9f0f33186a58bb042228181131c` | 694 |
| `authorization.json` | `6cdfa2ad4d8cf8ab092a27edde1c44103db689410495a424ef262ac01f29e7cb` | 1563 |
| `config_gate.csv` | `8507b0071db10a0d27f8ed9a1e2c23ba3104db48f346652b12995cb8c74cb956` | 6416 |
| `full_fit_results.csv` | `59beae9a46beb64d7b6d96b2d55ff822e1dea3d2cfdb185056b65bc379e398eb` | 138995 |
| `full_summary.json` | `ec14625eaac6e618984a104a381032518ac37ba931609459605f91e8d9cc0119` | 10480 |
| `leakage_gate.csv` | `0a47fccbc95523d8279092bb7ecd837f6833a625f919cdafc987bf951f2211af` | 152787 |
| `manifest.csv` | `14c084e5588c7ea288ce7fb6e433b7645c727062ed9d265f29a76f4762c4f4d0` | 68359 |
| `mask_provenance.csv` | `dd122535a6681db2d95558a47c9ee3e93963b223ef1f5acdf20256ff48b70036` | 7352 |
| `runinfo.json` | `0f0a0d6fc5e58f1074d4d8b46a6ff1e9d403dea2a7bb6e57713c47fb6a4cadbe` | 2357 |
| `selection_matrix.csv` | `7f1190d29e0e0f4e58511f9ed5b3e7711245625bdb6e1970102848a54fac4543` | 3867 |

## 5. 実行 contract（artifact から再計算）

| 項目 | 値 |
|---|---|
| `attempted_fit_count` | 336 |
| `clean_fit_calls` | 336 |
| `scored_rows` | 336 |
| `full_fit_results.csv` data 行数 | 336 |
| A 行 / B 行 | 168 / 168 |
| global `fit_index` | 1..336（重複 0 / 欠番 0） |
| K_TRUE=3 新規行 | 0 |
| `internal_retry` 非ゼロ行 | 0 |
| `replacement_fits_executed` | 0 |
| `phase7e_rerun_count` | 0 |
| `canary_fits_executed` / `smoke_fits_executed` | 0 / 0 |
| `finite_state` 全 True | True |
| `q_failure` 全 False | True |
| `nan_occurred` 全 False | True |
| `warning_count` 全 0 | True |
| `failure.json` | 不在（失敗 run ではない） |

seed rescue・tolerance 緩和・replacement fit・retry・Phase7e rerun・canary rerun・smoke rerun はいずれも **0**。frozen `partial_failure_policy` は `stop_immediately, no_replacement_fit, no_retry, no_seed_rescue, no_tolerance_change, preserve_partial_evidence, no_completed_summary, no_audit_pass, rerun_requires_a_new_human_gate`。

### gate

- `config_gate.csv`: 103 件すべて `passed=True`
- `leakage_gate.csv`: 336 行すべて `pre_fit_passed` / `post_fit_passed` = True、`fit_boundary_status` = `clean`（`boundary_version` = `phase8b-leakage-boundary-v1`）
- `mask_provenance.csv`: 24 セルすべて `anchor_match=True`、`mask_design` = `S_C`、split seed [42001, 42002, 42003]（Phase 7e と意図的に共有）

## 6. 独立監査

`tools/research_audit/audit_k_true_robustness_sweep.py --mode full`（artifact のみを読み、harness の selector も authorization も import しない）。

| 項目 | 値 |
|---|---|
| verdict | **PASS** |
| BLOCKER / HIGH / MEDIUM | 0 / 0 / 0 |
| findings | `[]` |
| audit_version | `phase8b-full-audit-v1` |
| 監査対象ファイル | 9 件 |

auditor は role 1 / role 2 / protocol hash / execution attempt id / 336 / 168 / Phase 7e anchor 42 を **自前の literal として独立に保持**しており、runner の定数を読み込まない。

## 7. 結果 — replicate 単位の selected K

**K_TRUE=3 の行は Phase 7e anchor の READ-ONLY 再利用であり、A と B は同一の証拠を参照している。6 個の独立実験ではない。**

### A（primary）

| K_TRUE | r1 | r2 | r3 | 真値一致 | lineage |
|---:|---:|---:|---:|:---:|---|
| 1 | **1** | **1** | **1** | 3/3 | `phase8a_new` |
| 2 | **2** | 3 | **2** | 2/3 | `phase8a_new` |
| 3 | **3** | **3** | 5 | 2/3 | `phase7e_anchor`（READ-ONLY 再利用・A/B 共有） |
| 4 | **4** | **4** | **4** | 3/3 | `phase8a_new` |
| 5 | 4 | 4 | **5** | 1/3 | `phase8a_new` |

### B（sensitivity）

| K_TRUE | r1 | r2 | r3 | 真値一致 | lineage |
|---:|---:|---:|---:|:---:|---|
| 1 | **1** | **1** | **1** | 3/3 | `phase8a_new` |
| 2 | **2** | 3 | **2** | 2/3 | `phase8a_new` |
| 3 | **3** | **3** | 5 | 2/3 | `phase7e_anchor`（READ-ONLY 再利用・A/B 共有） |
| 4 | **4** | **4** | **4** | 3/3 | `phase8a_new` |
| 5 | 3 | 4 | **5** | 1/3 | `phase8a_new` |

### recovery count（新規グリッド `K_TRUE in {1,2,4,5}` のみ）

| K_TRUE | A | B | 合算 |
|---:|:---:|:---:|:---:|
| 1 | 3/3 | 3/3 | 6/6 |
| 2 | 2/3 | 2/3 | 4/6 |
| 4 | 3/3 | 3/3 | 6/6 |
| 5 | 1/3 | 1/3 | 2/6 |
| **合計** | **9/12** | **9/12** | **18/24** |

Phase 7e anchor（`K_TRUE=3`、A/B 共有の単一証拠）: selected K = [3, 3, 5]、真値一致 **2/3**。

### 選択方向の内訳（新規グリッド、各 12 セル）

| estimand | exact | over-selection | under-selection |
|---|:---:|:---:|:---:|
| A | 9 | 1 | 2 |
| B | 9 | 1 | 2 |

### A / B の差

| K_TRUE | replicate | A | B |
|---:|---:|---:|---:|
| 5 | 1 | 4 | 3 |

新規グリッド 12 セル中、A と B で選択が異なるのは **1 セルのみ**。

**この 1 セルの差から信号強度スケーリングの一般的効果を推論しない。**

## 8. 選択マージン（記述のみ）

`margin` = 選択された K の 2-start 平均スコア − 次点 K の 2-start 平均スコア。値が小さいほど、その replicate では上位候補が僅差であったことを意味する。**これは原因の説明ではない。**

| estimand | K_TRUE | replicate | selected K | best mean score | margin |
|---|---:|---:|---:|---:|---:|
| A | 1 | 1 | 1 | -0.531069 | 0.041482 |
| A | 1 | 2 | 1 | -0.505641 | 0.025183 |
| A | 1 | 3 | 1 | -0.519225 | 0.042449 |
| A | 2 | 1 | 2 | -0.488623 | 0.028966 |
| A | 2 | 2 | 3 | -0.452133 | 0.010964 |
| A | 2 | 3 | 2 | -0.497071 | 0.051406 |
| A | 4 | 1 | 4 | -0.472542 | 0.035495 |
| A | 4 | 2 | 4 | -0.421853 | 0.034462 |
| A | 4 | 3 | 4 | -0.510354 | 0.008902 |
| A | 5 | 1 | 4 | -0.525364 | 0.004474 |
| A | 5 | 2 | 4 | -0.456218 | 0.027100 |
| A | 5 | 3 | 5 | -0.432541 | 0.058851 |
| B | 1 | 1 | 1 | -0.511381 | 0.034350 |
| B | 1 | 2 | 1 | -0.453276 | 0.038882 |
| B | 1 | 3 | 1 | -0.439876 | 0.023104 |
| B | 2 | 1 | 2 | -0.475124 | 0.036145 |
| B | 2 | 2 | 3 | -0.416563 | 0.002866 |
| B | 2 | 3 | 2 | -0.460430 | 0.030865 |
| B | 4 | 1 | 4 | -0.471446 | 0.046993 |
| B | 4 | 2 | 4 | -0.481811 | 0.017873 |
| B | 4 | 3 | 4 | -0.480006 | 0.028951 |
| B | 5 | 1 | 3 | -0.565335 | 0.009312 |
| B | 5 | 2 | 4 | -0.482753 | 0.006553 |
| B | 5 | 3 | 5 | -0.488501 | 0.046723 |

全 24 セルで tie 候補は 1 個のみであり、tie rule が発動したセルは存在しない。

## 9. 解釈（有限標本の記述的結果）

> 凍結した held-out 予測スコアによる K 選択では、`K_TRUE=1` および `K_TRUE=4` では
> 3 反復すべてで真値が選択され、`K_TRUE=2` では 2/3、`K_TRUE=5` では 1/3 で真値が
> 選択された。`K_TRUE=5` では候補集合に 5 より大きい K も含まれる一方、選択結果は
> 低い K 側に寄る傾向が観測された。ただし各条件 3 反復のみであり、本結果は有限標本に
> おける記述的結果として解釈する。

この傾向は A（primary、`w` 固定）と B（sensitivity、`w_K` スケーリング）でほぼ同一であった（差は 1 セルのみ）。

`K_TRUE=5` の under-selection について**原因は述べない**。§8 の margin 表は選択の僅差さを記述するだけであり、原因の同定には追加の解析が必要である。

## 10. 主張境界

### 書いてよい

- frozen 実験内での有限標本 selected-K パターン
- replicate 単位の変動
- 記述的な over-selection / under-selection
- A primary と B sensitivity の比較（分けて報告する）
- 明示された recovery count

### 書いてはいけない

- consistency / asymptotic consistency
- universal K recovery / K-selection consistency
- Schwarz BIC / BIC consistency / Schwarz-BIC success
- 理論保証（theoretical guarantee）
- 本合成設定を超える一般化
- 「Phase 8b は Q_strict / ICL-type / BIC で K を選んだ」（**事実として誤り**）
- 「K_TRUE=3 について A と B が独立に 6 セル分の証拠を与える」（同一 anchor の共有参照）

この実験が答えている問いは次の 1 つだけである:

> generator の `K_TRUE` が変わったとき、凍結された Phase 7e held-out plug-in K-selection protocol は有限標本でどのような selected-K パターンを示すか。

加えて lineage E（objective-consistent experimental prototype）は**本文採用不可**である（root `CLAUDE.md` §3）。

## 11. 証拠の数え方

| 区分 | fit 数 | 由来 |
|---|---:|---|
| Attempt 2 で新規実行 | 336 | 本 artifact |
| Phase 7e anchor（READ-ONLY 再利用） | 42 | `expfam/results/k_selection/heldout_full_pilot_20260824` |
| **統合ユニーク証拠** | **378** | — |

**336 + 42 = 378 であって 420 ではない。**
A と B が同じ anchor 42 fits を参照するため、anchor を 2 回数えてはならない。

## 12. Attempt 1（provenance のみ・科学的使用不可）

`expfam/results/k_selection/k_true_robustness_full_20260902/` を削除せず保全している。

| 項目 | 値 |
|---|---|
| status | `FAILED` |
| 位置づけ | **ABORTED_BY_OPERATOR_INTERRUPT / provenance only / no scientific use** |
| `attempted_fit_count` | 3 |
| `clean_fit_calls` | 2 |
| `scored_rows` | 0 |
| `failed_fit_index` | 3 |
| `retry_count` / `replacement_fits_executed` | 0 / 0 |
| `run_code_sha` | `1946953ffc7e7db586dda2933c9a25a6f0235d07` |
| `artifact_version` | `phase8b-full-artifact-v1` |

| ファイル | SHA-256 | bytes |
|---|---|---:|
| `authorization.json` | `e11216ff56ac107f5ae6b5b7d7996182a047d57412098dd354313cc88c925ba7` | 1278 |
| `config_gate.csv` | `8507b0071db10a0d27f8ed9a1e2c23ba3104db48f346652b12995cb8c74cb956` | 6416 |
| `failure.json` | `543bd103c4d890fa387c4625c235afa9306905b2823d30fc3a31b6256dfeb9ed` | 773 |
| `full_fit_results.csv` | `56333c91d9d346db4975246cc25ac43cae3982fd7207b2194326bd7561912586` | 343 |
| `leakage_gate.csv` | `538dc0b0fcb61f34699779ef1fa46075484747ac2ee754956e72debd7d0f1cec` | 243 |
| `manifest.csv` | `14c084e5588c7ea288ce7fb6e433b7645c727062ed9d265f29a76f4762c4f4d0` | 68359 |
| `mask_provenance.csv` | `dd122535a6681db2d95558a47c9ee3e93963b223ef1f5acdf20256ff48b70036` | 7352 |

Attempt 1 の 2 clean fits は Attempt 2 に **一切再利用していない**（`partial_results_reused = False`）。Attempt 1 の数値を科学的主張の根拠にしない。

---

**このアーカイブ工程で実行した real EM fit = 0。**

