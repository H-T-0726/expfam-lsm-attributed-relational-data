# Phase 9N — 最小 5 マージンでの有界な多ステップ局所最適化パイロット（2026-09-28）

- 位置づけ: **ZERO-EM PILOT**（Issue #100）。EM・E-step・refit・family 選択・新しいデータなし。
- 対象: Phase 9K の保存済み状態（`expfam/results/lap_vs_cq_20/phase9k_20260928/fitted_states.json`）のうち、
  元の C_Lap の最良 vs 次点の差が小さい 5 ペア（事前固定）の 10 状態だけ。
- 結果: `expfam/results/multistep_local/phase9n_20260928/`
  （`protocol.json`, `trajectory.csv`, `per_state_final.csv`, `pair_summary.csv`, `pair_trajectory.csv`, `final_theta.json`, `summary.json`）
- 実行コード: `c16be84`（`expfam/src/experimental/multistep_local_pilot.py`、実行時 clean）。1 回だけ実行。Phase 9K・9M の artifact は変更していない。
- 系列 E（experimental prototype; 本文採用不可）

---

## 1. 研究上の問い

実装の目的関数 `ℓ_Lap(θ, K) = laplace_log_observed(θ, K)` を固定手順で多ステップ局所改善したとき、
Phase 9K の最小 5 マージンで元の「最良 vs 次点」の順序が変わるか。これは refit でも新しい基準でもなく、MLE を保証しない。

## 2. 何をしたか（事前固定）

- **ペア**: rep10 (3 vs 2, 22.03), rep04 (3 vs 2, 29.15), rep19 (3 vs 2, 40.70), rep09 (3 vs 4, 42.99), rep18 (3 vs 4, 43.44)。
  実行時に Phase 9K の `paired_summary.json` の最良・次点と一致することを確認した。
- **再構成**: Phase 9M と同じ（data seed から X, Y を再生成、保存 θ、保存 Z_est → joint mode）。10/10 で `laplace_log_observed` が committed 値とビット単位で一致。
- **θ 座標**: 現在の F で作った水平座標（回転の接空間 `{F A : Aᵀ = −A}` の直交補空間）、Gaussian 列の対数分散、w0、w（加法）。
  **採用したステップごとに F・分散・w0・w を新しい中心とし、水平 chart を新しい F から作り直した**（次元 `Kd − K(K−1)/2` を毎回確認）。
- **Z の扱い**: 各反復内のすべての評価は、その反復の現在の採用済み joint mode から開始。採用した評価の mode を次の反復の開始点にした。restart・ridge・jitter・許容値変更なし。
- **勾配**: Phase 9M の主の中心差分（h = 5e-5; w0・w は `5e-5·max(1,|現在値|)`）だけ。h と h/2 の比較はなし。
- **1 反復**: `‖g‖∞ ≤ 1e-3` なら CONVERGED_SCORE。そうでなければ `u = g/‖g‖₂`、`eps_dir = 1e-3` の方向曲率（負を要求）、
  `t* = ‖g‖₂/|曲率|`、係数 1 … 1/64 で最初に有限かつ ℓ が増えたものを採用。
- **上限**: 1 状態あたり 20 回の採用ステップ。勾配は終点でも評価し、そこで許容値以下なら CONVERGED_SCORE、そうでなければ MAX_ITER_20。
  CONVERGED_SCORE と MAX_ITER_20 だけを正常な有界終了とし、それ以外はそのペアを unavailable とする。
- **診断値**: `C_Lap_multistep_diag(K) = C_Lap_Phase9K(K) − 2·cumulative_Δℓ(K)`（罰則は不変）。**新しい基準ではない**。

## 3. 結果

### 状態ごと

| rep | K | 停止 status | 採用ステップ | 累積 Δℓ | ‖g‖₂ 初期 → 終点 | ‖g‖∞ 初期 → 終点 |
|---|---|---|---|---|---|---|
| rep10 | 2 | MAX_ITER_20 | 20 | 0.446 | 5.97 → 1.11 | 2.57 → 0.946 |
| rep10 | 3 | MAX_ITER_20 | 20 | 0.901 | 18.4 → 1.25 | 13.7 → 0.916 |
| rep04 | 2 | **DERIVATIVE_UNAVAILABLE** | 4 | 0.153 | 7.67 → — | 5.30 → — |
| rep04 | 3 | MAX_ITER_20 | 20 | 0.515 | 6.74 → 0.652 | 2.78 → 0.381 |
| rep19 | 2 | MAX_ITER_20 | 20 | 0.406 | 7.79 → 1.06 | 5.02 → 0.706 |
| rep19 | 3 | MAX_ITER_20 | 20 | 0.692 | 8.69 → 0.917 | 4.21 → 0.469 |
| rep09 | 3 | **DERIVATIVE_UNAVAILABLE** | 14 | 0.517 | 7.08 → — | 4.35 → — |
| rep09 | 4 | **DERIVATIVE_UNAVAILABLE** | 6 | 0.982 | 17.5 → — | 12.6 → — |
| rep18 | 3 | MAX_ITER_20 | 20 | 0.271 | 5.55 → 0.427 | 2.81 → 0.252 |
| rep18 | 4 | MAX_ITER_20 | 20 | 0.899 | 11.9 → 1.28 | 7.75 → 0.806 |

- 停止 status: CONVERGED_SCORE 0、**MAX_ITER_20 7**、**DERIVATIVE_UNAVAILABLE 3**、その他 0（NO_ACCEPTED_STEP, CURVATURE_NONNEGATIVE, CHART_RANK_FAILURE, OBJECTIVE_UNAVAILABLE はいずれも 0）。
- DERIVATIVE_UNAVAILABLE の 3 状態は、いずれも勾配の 1 成分で、摂動した点の joint mode の評価が OK にならなかった（どの座標かは記録していない。救済していない）。
  失敗までに採用したステップは有効で、その累積 Δℓ は上表のとおり。
- **CONVERGED_SCORE に達した状態はない。MAX_ITER_20 の 7 状態は収束していない**（終点の ‖g‖∞ は 0.25–0.95）。
- 採用された係数はすべて 1、方向曲率は −328.7 〜 −75.0（すべて負）、chart の次元は毎回一致。
  採用した評価の Z-mode の grad_inf は最大 9.4e-9、H の最小固有値は最小 0.497。
- 1 ステップあたりの改善は反復とともに小さくなり（最後のステップ 7e-4 〜 0.03 程度）、累積 Δℓ は 0.15〜0.98 にとどまった。
  K が大きい側（rep10 の K=3、rep09/rep18 の K=4）の方が累積改善が大きい。

### ペアごと（最良 vs 次点、criterion 単位）

| rep | 最良 / 次点 | 元の差 | 終点の診断差 | 判定 |
|---|---|---|---|---|
| rep10 | 3 / 2 | 22.03 | 22.94 | retained |
| rep04 | 3 / 2 | 29.15 | —（ラウンド 4 まで 29.64） | **unavailable**（K=2 が DERIVATIVE_UNAVAILABLE） |
| rep19 | 3 / 2 | 40.70 | 41.27 | retained |
| rep09 | 3 / 4 | 42.99 | —（ラウンド 6 まで 41.95） | **unavailable**（K=3, K=4 とも DERIVATIVE_UNAVAILABLE） |
| rep18 | 3 / 4 | 43.44 | 42.19 | retained |

- **retained 3 / flipped 0 / unavailable 2**。
- 同期ラウンドごとの差（`pair_trajectory.csv`）では、**途中の反転は 1 つもなかった**。途中の差の最小値は rep10 22.03（ラウンド 0）、
  rep09 41.95（両側が利用可能だった最後のラウンド 6）、rep18 42.19。rep10 の差は 22.03 → 22.59（ラウンド 1）→ 22.94（ラウンド 20）と単調に増えた。
- unavailable の 2 ペアでも、利用可能だったラウンドでの差の変化は 1.1 以下（rep04 +0.49、rep09 −1.04）だった。これは記述であり、判定には使わない。

## 4. DECISION

## **DECISION: MULTISTEP_PILOT_INCONCLUSIVE**

5 ペアのうち 2 ペア（rep04, rep09）で片側または両側が DERIVATIVE_UNAVAILABLE で止まり、事前に決めた規則でペアが unavailable になった。
利用可能な 3 ペアではいずれも順序は保たれた（flip 0）が、それは STABLE の条件（5 ペアすべて available）を満たさない。
また、利用可能な 6 状態はすべて MAX_ITER_20 で、**収束していない**。

## 5. 解釈（INTERPRETATION）

- 1 回目のステップの Δℓ は 10 状態すべてで Phase 9M の 1 ステップの値と一致した（同じ手順の再現の確認）。
  累積 Δℓ は 0.15–0.98（criterion 単位で最大 1.96）で、同じ状態の 1 回目のステップの 1.7–2.8 倍だった。
  改善は最初の数ステップに集中し、その後はステップごとに小さくなった。
- 利用可能な 3 ペアでは、差の変化は −1.26（rep18）〜 +0.91（rep10）で、元の差（22–43）より 1 桁以上小さかった。
- ただし勾配はまだ 0 ではなく（終点の ‖g‖∞ 0.25–0.95）、この方法（1 方向・スコア方向・固定の曲率）で 20 ステップ以降にどれだけ改善が残るかは分からない。
- DERIVATIVE_UNAVAILABLE は、局所最適化を続けると、ある方向の小さな摂動で joint mode の評価（Phase 9H の許容値と Hessian 正定値の条件）が OK にならない点に到達しうることを示す。
  実装の `ℓ_Lap` は、そのような点の近くでは有限差分で評価しにくい。

## 6. CAN SAY / CANNOT SAY

**CAN SAY**
- Phase 9K の最小 5 マージンで、固定した多ステップの局所スコア方向最適化（最大 20 ステップ）を行ったところ、
  3 ペア（rep10, rep19, rep18）では終点で順序が保たれ、2 ペア（rep04, rep09）は途中で勾配の成分が評価できなくなり unavailable になった。
- 同期ラウンドごとに見ても、利用可能なラウンドで反転はなかった。
- 各状態の累積 Δℓ は 0.15–0.98 で、利用可能な 3 ペアの差の変化は元の差より 1 桁以上小さかった。

**CANNOT SAY**
- 真の MLE、`ℓ_Lap` の局所最適、大域最適に到達したこと（MAX_ITER_20 は未収束、CONVERGED_SCORE は 0）。
- 完全に最適化した Candidate B の結果、診断値が新しい基準であること、新しい recovery rate。
- 8 回の MCEM 反復で一般に十分であること、一致性、優越性。
- rep04・rep09 の順序が保たれること（unavailable）。
- Phase 9K の結果を書き換えること（Phase 9K は凍結済みの過去の結果のまま）。

## 7. 最大の UNRESOLVED と次の Human 判断

**最大の UNRESOLVED**: 3 状態で DERIVATIVE_UNAVAILABLE になった原因（摂動した点で joint mode が NOT_STATIONARY になったのか、Hessian が正定値でなくなったのか、
どの座標か）が記録されておらず、rep04・rep09 の順序は判定できない。また、利用可能な状態もすべて未収束で、20 ステップ以降の残りの改善の大きさは分からない。

**次の Human 判断**: unavailable の原因を（新しい最適化なしに）保存された終点の近くで記述する診断を行うか、
あるいは本パイロットの結果（3/5 retained、2/5 unavailable、flip 0）で局所最適化の問いを閉じるか。
いずれも本 Issue の範囲外で、他の Phase 9K 状態への拡張や別の最適化手法には新しい Human Gate が必要。
