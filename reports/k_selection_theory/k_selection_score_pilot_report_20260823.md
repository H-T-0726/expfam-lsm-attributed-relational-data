# Phase 7b K-selection score diagnostic pilot

## 1. Purpose and scope

Issue #37 Phase 7b は、同一の fitted models 上で3種類の latent-dimension selection score の定義、挙動、argmin、および post-hoc curvature quantity を比較する diagnostic pilot である。

使用したモデルは objective-consistent な experimental/prototype lineage (`DualExpFamLSMConsistent`) であり、manuscript-approved ではない。本実験は paper Experiment 2 の再現でも、K-selection consistency study でもない。

## 2. Score definitions

区別する量は次の3つである。

- Q1: `log p(X,Y | Z,theta)`。conditional / plug-in quantity で、Zを積分しない。
- Q2: `log p(Z,X,Y | theta)`。現在の `Q_strict` がsampling procedure下で平均する対象。
- Q3: `log p(X,Y | theta)`。observed-data marginal likelihood。C1/C2/C3のいずれもQ3ではない。

候補scoreは次のとおり。

- C1: `bic_impl = -2 Q_strict + p_hat log n`。descriptive baseline。
- C2: `S_cf = -2(Q_strict - lnpZ_det) + p_hat log n`、`lnpZ_det = -(nk/2)(1+log(2*pi))`。counterfactual diagnostic。
- C3: `S_laplace_post = S_cf + (1/L) sum_l sum_i logdet A_i_post`。final scaled samplesから再計算したpost-hoc curvature diagnostic。

C1をcorrect/Schwarz BIC、C2またはC3をcorrected/true BIC、C3をELBO、marginal likelihoodまたはその近似・boundとは呼ばない。

## 3. Design

| Item | Setting |
|---|---|
| `family_x` / `family_y` | poisson / bernoulli |
| `K_TRUE` | 3 |
| candidate `k` | 1, 2, 3, 4, 5, 6, 7 |
| `n` | 75, 150 |
| trials | 3 (1-based) |
| `d` | 15 |
| `L` | 5 |
| `num_iter` | 8 |
| numerics | `consistent` |
| total fits | 42 |

- Fixed execution commit: `eaad30a2301a90acea2fe9a6ea149dc0056753de`
- Runner SHA-256: `7c03ce92cdf6c2b7ea43e4939febabf797204a525c13710aeafe29ca8aa6d2bb`

## 4. Integrity

Primary summary/runinfoの監査結果は次のとおり。

- 42/42 fits completed and recorded; `(n, trial, k_est)` は42件すべてunique。
- 全fitで `numerics_mode = consistent`。
- max `lnpZ_abs_err = 2.273736754432321e-13`（gate: `< 1e-6`）。
- internal retry 0、`nan_occurred` 0、total `nan_count` 0、`q_bic_failed` 0、warnings 0。
- non-positive slogdetは `0 / 23625` evaluations。
- 各 `(n,trial)` 内で全candidate kのX hashおよびY hashが一致し、full summaryのhashは64 hex。
- smoke/full共通3 fits `(n=75, trial=1, k=2,3,4)` のseedと主要数値は完全一致した。

## 5. Main diagnostic result

| Score | Selection | Frequency | Status |
|---|---:|---:|---|
| C1 `bic_impl` | k=3 | 6/6 | interior |
| C2 `S_cf` | k=7 | 6/6 | range boundary |
| C3 `S_laplace_post` | k=3 | 6/6 | interior |

C2は6群中5群でcandidate kに対してstrictly decreasingだった。例外は `n=75, trial=3` で、`k=4 -> 5` に一時的な上昇（`+15.878`）がある。それでも同群のminimumはcandidate range上限の `k=7` である。したがって「C2は全6群で単調減少」とは主張しない。

C1/C3は全6群でk=3を選んだが、Phase 7bは両者のいずれかをtheoretically correct criterionとする根拠を与えない。

## 6. Curvature diagnostic

Trial-meanの `sum_log_det_A_post` は、n=75、n=150の両方でcandidate kとともに増加した。

| n | k=3から7へのC2 change | curvature change | C3 change |
|---:|---:|---:|---:|
| 75 | -116.010 | +443.048 | +327.038 |
| 150 | -340.995 | +972.739 | +631.743 |

このpilotではpost-hoc curvature termがC2の高K方向への低下を上回った。これはpost-hoc diagnosticとして観測された挙動であり、正しいpenaltyまたはcorrected BICを意味しない。

## 7. Claim boundary

Allowed:

- このscenario、candidate range、seedで観測されたselection behavior。
- score definitionによって選択Kが異なったこと。
- post-hoc curvature diagnosticが両nでKとともに増加したこと。
- 固定設定・seedにおけるsmoke/full reproducibility。

Prohibited:

- C1がcorrect/Schwarz BICであるという主張。
- C2/C3がcorrectedまたはtrue BICであるという主張。
- C3がELBO、marginal likelihood、その近似またはboundであるという主張。
- K-selection consistency、`n -> infinity` recovery、generalization、real-data validity。
- paper Experiment 2 reproduction。

## 8. Independent audit

Independent audit verdictは **A: RESULTS_VALID — READY_FOR_DECISION**。FindingはBLOCKER 0、HIGH 0、MEDIUM 0だった。

LOW findingsは、(1) C2のstrict decreaseは5/6群のみであること、(2) trial-average score figureだけでは `n=75, trial=3` のtrial-level例外が見えないこと、の2点である。本reportではC2を全群単調減少とは表現せず、例外とboundary minimumを明記した。

## 9. Decision

Formal decisionは **C: DESIGN_HELDOUT_K_SELECTION_NEXT**。

同じfitsでもC1/C3とC2が異なるKを選び、C1/C3をcorrect criterionとする理論的根拠もPhase 7bでは得られなかった。このため、in-sample Q-based diagnosticsを拡大する前に、held-out predictive K-selectionの設計可能性を検討する。これは「C3が正しい」ことを理由とするdecisionではない。

Decision Cは、少なくとも次の場合に撤回する。

- dyadic dependenceを保った公平なheld-out splitを設計できない。
- X/Y双方で情報漏洩を避けたprediction targetを定義できない。
- K間で同一prediction targetを公平に比較できない。
- theoretical auditでheld-out riskがKを識別しないと判明する。
- small design validationでsplit/seedの軽微な変更だけによりK-selectionが解釈不能なほど不安定になる。

## 10. Primary artifacts

- Runner: `tools/research_audit/run_k_selection_score_pilot.py`
- Summary: `expfam/results/k_selection/k_selection_score_pilot_20260823_summary.csv`
- Aggregate: `expfam/results/k_selection/k_selection_score_pilot_20260823_agg.csv`
- Selection: `expfam/results/k_selection/k_selection_score_pilot_20260823_selection.csv`
- Runinfo: `expfam/results/k_selection/k_selection_score_pilot_20260823_runinfo.csv`
- Score curves: `figures/k_selection/k_selection_score_pilot_20260823_score_curves.png`
- Curvature figure: `figures/k_selection/k_selection_score_pilot_20260823_log_det_A.png`
