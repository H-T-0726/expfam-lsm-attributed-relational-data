# Phase 9C C2 research-first exploratory feasibility pilot — research summary（2026-09-25）

- 位置づけ: **EXPLORATORY / FEASIBILITY CHARACTERIZATION**（Issue #74 Human Gate 2026-09-25, comment 5831839025）
- 系列: **E（experimental prototype; 本文採用不可）** — `expfam/src/experimental/family_selection.py`（per-column consistent numerics）
- 一次データ: `expfam/results/family_selection/pilot_c2_20260925/`（`summary.json`, `selection_trace.csv`, `family_scores.csv`, `fit_results.csv`, `audit_report.json`）
- 実行コード: `8fac4af`（git_dirty = False）。C2 は **1 回だけ**実行。rerun / reseed / retune なし。

---

## 1. 研究上の問い

Phase 9C の family 選択機構（support gate ＋ ambiguous 列の Bernoulli/Poisson score 比較、
Scheme C hybrid: exploration 中の A-type 更新 ＋ fresh fixed-family refit）は、凍結済み C2 条件で

- 0/1 列（真は Bernoulli）に対して、どの family を選ぶか
- 開始点（start_B / start_P）や replicate によって選択が揺れるか
- 数値上の限界（候補最適化の収束 warning）がどこで起きるか

を観察する。**一般的な自動分布選択の確証的証拠を得ることは目的ではない。**

## 2. 何をしたか

- 凍結 C2: n=75, d=12, K_true=K_fit=3, X 列 = Gaussian×3（列0–2）・Bernoulli×6（列3–8）・Poisson×3（列9–11）,
  family_y=bernoulli, sigma_x_var=1, w0=−1, w=1, f_scale=√2, L=5, exploration/refit num_iter=8,
  3 replicates（seeds 95100r/95200r/95300r）× 2 starts。
- 候補最適化: analytic-gradient BFGS（maxiter 2000, gtol 1e-10）。収束 diagnostic の閾値 grad_inf ≤ 1e-8 は**不変**。
- ambiguous 列には勝者候補の loading を install（Gate 74-B5）。
- 事前の zero-EM preflight: contract tests（299 passed, 3 skipped）＋ auditor の独立 frozen C2 expectation（findings 0）。
- 今回の policy 変更は「役割」だけ: 候補収束は **WARNING diagnostic** として記録し、progression blocker にはしない。
  technical validity = audit BLOCKER 0・HIGH 0・run_status SUCCESS。

## 3. 一次結果（OBSERVED）

### Technical validity

| 項目 | 値 |
|---|---|
| technical_validity | **VALID** |
| 実 EM 実行 | **12 / 12** SUCCESS（ledger 12 行, runinfo em_executions 12） |
| audit | PASS, BLOCKER 0, HIGH 0, MEDIUM 0 |
| retry / replacement / seed rescue | 0 / 0 / 0 |
| selected-loading provenance | 全 288 selection 行で完備・一致（audit で検査） |

### Family 選択（score が決める ambiguous 列 = 列3–8 のみ）

| replicate | start | ambiguous 列の選択 | 真 | start 間一致 |
|---|---|---|---|---|
| rep1 | start_B / start_P | Bernoulli ×6 / Bernoulli ×6 | Bernoulli ×6 | 一致 |
| rep2 | start_B / start_P | Bernoulli ×6 / Bernoulli ×6 | Bernoulli ×6 | 一致 |
| rep3 | start_B / start_P | Bernoulli ×6 / Bernoulli ×6 | Bernoulli ×6 | 一致 |

- **Bernoulli → Poisson 誤選択: 0 / 36**（3 rep × 2 start × 6 列）。
- **start 一致: 3 / 3 replicate**（最終 12 列 assignment が完全一致）。
- **ambiguous_true_poisson: 0 件**。真 Poisson 列（9–11）はどの replicate でも 2 以上のカウントを含み、gate が Poisson に決定した
  （gate 決定列なので selector の正解率には含めない）。
- 最終 margin（−2 score 単位, Bernoulli 優位）: min **45.69**, max **65.50**, 平均 54.72。
  replicate 内の start 間差は最大 2.21（rep2 列7: 54.60 vs 56.81）。
- 選択の軌跡: 全 288 selection 行で Bernoulli が選ばれた。変化は **start_P の iteration 1 のみ**（6 列 × 3 rep = 18 件, Poisson → Bernoulli）で、
  以後は一度も変わらない。trace 全体の margin は 45.50–68.91。

### 候補収束 diagnostic（WARNING, 閾値不変）

| 項目 | 値 |
|---|---|
| status | **CONVERGENCE_WARNING** |
| 候補評価 | 576（288 行 × 2 候補） |
| warning（grad_inf > 1e-8） | **64 / 576**（60 / 288 行） |
| 最終 iteration 候補の非収束 | 8 / 72 |
| SciPy status | 64 件すべて status 2（precision loss）; maxiter 到達は 0 |
| BFGS 反復数 | 5–15 |
| grad_inf | min 1.04e-8, median 4.32e-8, max 1.30e-6（>1e-7: 19 件, >1e-6: 1 件） |
| 候補 family | Poisson（敗者側）47, Bernoulli（勝者側）17 |
| 分布 | 6 run すべてに発生（5–17 件/run）、全 iteration（1–8）に分散 |
| warning 行の margin | 最小 45.75 |

最大値 1.30e-6 は rep3 start_P iteration 1 列5 の Poisson 候補（margin 51.30）。

### Fresh fixed-family refit（副次）

| replicate | Q_strict | 履歴 criterion field（`bic`） | 自由パラメータ数 | Procrustes RMSE(Z) |
|---|---|---|---|---|
| rep1 | −2528.600 | 5212.629 | 36 | 0.422 |
| rep2 | −2458.043 | 5071.515 | 36 | 0.430 |
| rep3 | −2523.145 | 5201.720 | 36 | 0.448 |

各 replicate で start_B / start_P の refit 値は一致（選択 assignment と refit seed が同じため）。
`bic` 列は Q-based complete-data criterion（ICL-type）であり、Schwarz BIC ではない（KI-010）。

## 4. 言えること

- 凍結 C2 条件・この 3 replicate では、Phase 9C 機構は**技術的に最後まで動き**（12/12, integrity 0, provenance 完備）、
  真 Bernoulli の 0/1 列 36 件すべてで Bernoulli を選んだ。開始点に依らず同じ結論に収束した。
- start_P から始めても、Poisson の割り当ては最初の選択で Bernoulli に入れ替わり、以後安定した。
- 候補収束 warning は頻繁（約 11%）に起きるが、すべて SciPy precision loss による早期停止で、反復予算の不足ではない。
  warning が起きた行でも margin は 45 以上あった。

## 5. まだ言えないこと・限界

- **一般的な自動分布選択の有効性**、**人手指定より優れること**、**実データでの有効性**は言えない（synthetic・1 条件・3 replicate）。
- gate が決めた Gaussian / Poisson 列は selector の成績ではない。selector が実際に判断したのは Bernoulli/Poisson の 0/1 列だけで、
  しかも真が Bernoulli の場合しか観測していない（ambiguous_true_poisson は 0 件で、**0/1 に見える真 Poisson 列での挙動は未観測**）。
- margin は大きいが、この条件（f_scale=√2, n=75）が判別の易しい側にある可能性がある。より弱い信号や小さい n での挙動は未検証。
- 収束 warning が score・margin・選択に与える影響は**測っていない**。（HYPOTHESIS: 勾配 ≤1.3e-6 の点で score の誤差は margin 45 に比べて
  極めて小さいと予想されるが、本 run では検証していない。）
- 本 run は exploratory であり、確証的 evidence に昇格しない。系列 E（prototype）で、本文採用不可。
- 歴史的 artifact の verdict（smoke-v2 / smoke-v3 の `BLOCKED_FOR_PILOT`）は変更していない。本 run でも historical S1 値は
  `BLOCKED_FOR_PILOT` として並記され、`pilot_progress_eligible = false` のまま（research-first policy では progression 判定に使わない）。

## 6. 次の Human / 指導教員の判断

1. この結果（機構は動く・易しい条件では安定に正しく選ぶ）で Phase 9C の feasibility を十分とみなすか。
2. 判別が難しい条件（真 Poisson が 0/1 に見える列、弱い信号、小さい n）を future run で確認するか。確認するなら、事前に research question・条件・seed を固定する。
3. 収束 warning の score への影響を測る必要があるか（研究判断が変わる場合に限る）。
4. #75 や K 探索に進むかどうか（本タスクでは開始していない）。
