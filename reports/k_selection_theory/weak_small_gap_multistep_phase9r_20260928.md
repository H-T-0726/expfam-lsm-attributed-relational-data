# Phase 9R — weak_w の小さな C_Lap 差に対する有界な多ステップ局所診断（2026-09-28）

- 位置づけ: **BOUNDED MULTI-STEP LOCAL DIAGNOSTIC**（Issue #108）。EM・E-step・refit・family 選択・新しいデータなし。
- 対象: Phase 9Q と同じ 5 ペア / 10 状態。**開始点は Phase 9P weak_w の元の保存状態**（`expfam/results/relational_w_sensitivity/phase9p_20260928/weak_w/fitted_states.json`）。
  Phase 9Q の artifact は 1 ステップ目の照合にだけ使った。
- 結果: `expfam/results/weak_small_gap_multistep/phase9r_20260928/`
  （`protocol.json`, `first_step_gate.csv`, `trajectory.csv`, `per_state_final.csv`, `pair_summary.csv/json`, `pair_trajectory.csv`, `final_theta.json`, `failure_metadata.csv/json`）
- 実行コード: `42eb147`（`expfam/src/experimental/weak_small_gap_multistep.py`、実行時 clean）。1 回だけ実行。
  中身は Phase 9N の `multistep_local_pilot.py` をそのまま使った。Phase 9P・9Q の artifact は変更していない。
- 系列 E（experimental prototype; 本文採用不可）

---

## 1. 研究上の問い

Phase 9Q で 1 ステップでは 5 ペアすべての順序が保たれた。同じ元の θ̂ から Phase 9N と同じ有界な多ステップ手順（最大 20 回の採用ステップ）を繰り返すと、有界な終点で順序が変わるか。
refit・新しい基準・MLE の判定・大域最適化ではない。

## 2. 何をしたか

- **再構成**: weak_w の protocol で X, Y を再生成し、保存された θ・Z_est から Candidate B を再評価。10/10 で committed 値と一致。
- **手順**（Phase 9N と同じ）: 各反復で、現在の F から回転の接空間と水平補空間を作り直し（次元 `Kd − K(K−1)/2` を毎回確認）、
  主の中心差分（h = 5e-5; w0・w は `5e-5·max(1,|現在値|)`）で勾配を求め、`‖g‖∞ ≤ 1e-3` なら CONVERGED_SCORE。
  そうでなければスコア方向に `eps_dir = 1e-3` の曲率で `t*` を求め、係数 1 … 1/64 で最初に ℓ が増えたものを採用。採用した評価の Z mode を次の反復の開始点にする。
  最大 20 回の採用ステップ。勾配は終点でも評価する（Phase 9N と同じ）。
- **Phase 9Q の 1 ステップ目の照合**: まず 10 状態すべてを 1 回の採用ステップまで進め、Phase 9Q と比べた。すべて一致した場合だけ残りの反復に進んだ。
  続きの最初に 1 ステップ目の点の勾配をもう一度（同じ入力で決定的に）計算しており、1 回目と完全に一致したことを 10/10 で記録した。
- **失敗の記録**: 手順がすでに行った Candidate B の評価の構造化結果を保存しただけで、再評価・救済はしていない。
- **診断値**: `C_Lap_multistep_diag(K) = C_Lap_Phase9P_weak(K) − 2·cumulative_Δℓ(K)`。**新しい基準ではない**。

## 3. 結果

### Phase 9Q の 1 ステップ目の照合: **PASS（10/10）**

status（ACCEPTED）、採用係数、Δℓ、曲率、t* がすべて一致した（Δℓ はビット単位で同じ値）。

### 状態ごと

| rep | K | 停止 status | 採用ステップ | 累積 Δℓ | ‖g‖₂ 初期 → 終点 | ‖g‖∞ 初期 → 終点 |
|---|---|---|---|---|---|---|
| rep10 | 2 | **DERIVATIVE_UNAVAILABLE**（20 ステップ後の終点で） | 20 | 0.432 | 6.34 → — | 2.77 → — |
| rep10 | 3 | MAX_ITER_20 | 20 | 0.729 | 9.24 → 1.28 | 4.95 → 0.739 |
| rep11 | 3 | **DERIVATIVE_UNAVAILABLE** | 7 | 0.661 | 8.65 → — | 5.11 → — |
| rep11 | 2 | **DERIVATIVE_UNAVAILABLE** | 17 | 0.535 | 7.01 → — | 4.05 → — |
| rep07 | 2 | MAX_ITER_20 | 20 | 0.539 | 8.75 → 0.914 | 4.64 → 0.505 |
| rep07 | 3 | MAX_ITER_20 | 20 | 0.898 | 10.1 → 1.73 | 4.96 → 0.994 |
| rep04 | 2 | MAX_ITER_20 | 20 | 0.628 | 9.16 → 2.18 | 5.88 → 1.64 |
| rep04 | 3 | MAX_ITER_20 | 20 | 0.700 | 6.97 → 1.10 | 2.70 → 0.547 |
| rep09 | 3 | **DERIVATIVE_UNAVAILABLE**（20 ステップ後の終点で） | 20 | 0.699 | 7.16 → — | 3.17 → — |
| rep09 | 2 | MAX_ITER_20 | 20 | 0.243 | 4.14 → 0.831 | 1.87 → 0.650 |

- 停止 status: CONVERGED_SCORE 0、**MAX_ITER_20 6（収束していない）**、**DERIVATIVE_UNAVAILABLE 4**、その他 0。
- rep10 K=2 と rep09 K=3 は 20 回のステップをすべて採用したあと、終点での勾配の評価（収束の確認）で 1 成分が利用不可になった。
  凍結した Phase 9N の規則では、終点の勾配が評価できない状態は DERIVATIVE_UNAVAILABLE であり、ここでもそのまま扱った（事後に MAX_ITER_20 へ読み替えない）。

### 失敗の詳細（構造化ログ、追加評価なし）

| 状態 | 失敗した反復 | 座標 | 側 | Candidate B status | max\|∇Φ\| | mode 反復 | H の最小固有値 | logdet 符号 / 値 |
|---|---|---|---|---|---|---|---|---|
| rep10 K=2 | 21（終点） | F_horizontal[19] | + | NOT_STATIONARY | 1.21e-8 | 200 | 1.89 | +1 / 334.4 |
| rep11 K=3 | 8 | F_horizontal[30] | − | NOT_STATIONARY | 3.63e-8 | 200 | 0.96 | +1 / 445.7 |
| rep11 K=2 | 18 | F_horizontal[22] | − | NOT_STATIONARY | 1.12e-8 | 200 | 1.82 | +1 / 316.5 |
| rep09 K=3 | 21（終点） | w | + | NOT_STATIONARY | 1.38e-8 | 200 | 0.79 | +1 / 457.6 |

4 件とも、1 座標の片側だけで、Candidate B が NOT_STATIONARY（joint-mode solver が反復上限 200 に達し、`max|∇Φ|` が許容値 1e-8 をわずかに上回る）、H は正定値のままだった。
Phase 9O で分類した Phase 9N の失敗と同じ型である（ただし今回は 3 件が F の水平座標、1 件が w）。例外・非有限値・Hessian の正定値性の喪失はなかった。

### ペアごと（元の最良の向き、criterion 単位）

| rep | 最良 / 次点 | 元の差 | 終点の診断差 | 判定 | K=3 側の累積改善が大きいか |
|---|---|---|---|---|---|
| rep10 | 2 / 3 | 1.536 | —（ラウンド 20 で 0.942） | **unavailable**（K=2 が DERIVATIVE_UNAVAILABLE） | yes（0.729 vs 0.432） |
| rep11 | 3 / 2 | 4.537 | —（ラウンド 7 まで、4.930） | **unavailable**（両側 DERIVATIVE_UNAVAILABLE） | yes（0.661 vs 0.535） |
| rep07 | 2 / 3 | 5.264 | 4.546 | retained | yes（0.898 vs 0.539） |
| rep04 | 2 / 3 | 7.023 | 6.879 | retained | yes（0.700 vs 0.628） |
| rep09 | 3 / 2 | 10.702 | —（ラウンド 20 で 11.615） | **unavailable**（K=3 が DERIVATIVE_UNAVAILABLE） | yes（0.699 vs 0.243） |

- **retained 2 / flipped 0 / tie 0 / unavailable 3**。
- **途中の反転・同点: なし**。どのラウンドでも差は正のままだった（`pair_trajectory.csv`）。
- rep10 の同期ラウンドの差: 1.536（0）→ 1.328（1）→ 1.047（5）→ 1.004（7）→ 0.973（10）→ 0.951（15）→ 0.942（20）。
  縮み方はラウンドとともに小さくなった（ラウンド 15→20 で 0.009）。
- 5 ペアすべてで、K=3 の側の累積改善が K=2 の側より大きかった。

## 4. DECISION

## **DECISION: WEAK_SMALL_GAP_MULTISTEP_INCONCLUSIVE**

3 ペア（rep10, rep11, rep09）で片側または両側が DERIVATIVE_UNAVAILABLE で止まり、事前の規則で unavailable になった。
利用可能な 2 ペア（rep07, rep04）は終点で順序が保たれた。利用可能な状態はすべて MAX_ITER_20 で、**収束していない**。

## 5. 解釈（INTERPRETATION）

- 多ステップでも、どのペアでもどのラウンドでも順序は反転しなかった。最も小さい rep10 では、20 ラウンドで差が 1.536 → 0.942 まで縮んだが、縮み方は小さくなっていた。
  ただし rep10 は K=2 側の終点の勾配が評価できず、事前の規則では unavailable である。ラウンド 20 の差 0.942 は記述であり、判定には使わない。
- 累積の改善は K=3 の側で一貫して大きく、差は K=3 寄りに動き続けた（Phase 9Q の 1 ステップと同じ向き）。
- 失敗の型は Phase 9O と同じで、joint-mode solver が許容値 1e-8 のわずか上で 200 反復止まるものだった。局所最適化を続けると、この型の失敗が一定の割合で起きる。

## 6. CAN SAY / CANNOT SAY

**CAN SAY**
- 同じ 5 つの weak_w の小さな差のペアについて、凍結した Phase 9N 型の 20 ステップの局所診断では、2 ペア（rep07, rep04）が終点で順序を保ち、3 ペアは Candidate B の NOT_STATIONARY による勾配の欠損で unavailable になった。
- どのペアでも、利用可能なラウンドで順序の反転・同点はなかった。rep10 の差はラウンド 20 で 0.942 だった（K=2 側は 20 ステップ採用後の終点の勾配だけが評価できなかった）。
- 5 ペアすべてで、K=3 の側の累積改善が大きかった。
- 最初のステップは Phase 9Q と完全に一致した。

**CANNOT SAY**
- MAX_ITER_20 の状態が MLE・局所最適であること（CONVERGED_SCORE は 0）。大域最適。完全に最適化した Candidate B。
- rep10・rep11・rep09 の終点での順序（事前の規則では unavailable）。
- 20 ステップ以降に順序が変わらないこと。
- Phase 9P の K の数の書き換え、新しい recovery rate、一般的な頑健性・一致性・優越性。

## 7. 最大の UNRESOLVED と次の Human 判断

**最大の UNRESOLVED**: rep10 は 20 ステップで差 0.942 まで縮み、最小の差であることに変わりはないが、終点が unavailable で、状態も収束していない（CONVERGED_SCORE 0）。
この局所手順では、joint-mode solver の NOT_STATIONARY（許容値 1e-8 のわずか上で 200 反復）が繰り返し起き、判定を妨げている。
これを評価の定義（許容値・反復上限）の問題として扱うかどうかは、frozen spec の変更にあたる。

**次の Human 判断**: (a) joint-mode solver の停止条件を事前に固定して変えた別の設計にするか、(b) 局所最適化の問いをここで閉じ、
「1 ステップ・20 ステップの範囲で反転は観察されなかったが、小さな差は K=3 寄りに縮む」という記述にとどめるか。いずれも新しい Human Gate が必要。
