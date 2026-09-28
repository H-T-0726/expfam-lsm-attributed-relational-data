# Phase 9Y — K3 → K4 の境界の読み取り専用の分解（2026-09-29）

- 位置づけ: **POST-HOC EXPLORATORY MECHANISM CHARACTERIZATION**（Issue #124）。Phase 9X の K_true = 4 で見えた過小選択の境界について、
  すでに commit 済みの値だけから基準の差を既存の成分に分解する。分析の内容は Issue #124 で計算の前に固定された。**原因の証明ではない**。
- **EM 0、refit 0、family 選択 0、Candidate B の再評価 0、θ の最適化 0**。新しい dataset・seed・K_true もない。
- 結果: `expfam/results/k34_boundary_decomposition/phase9y_20260929/`
  （`source_provenance.json`, `ktrue4_rows.csv`, `ktrue4_summary.json`, `ktrue3_context_rows.csv`, `ktrue3_context_summary.json`, `reconstruction_checks.json`）
- 実行コード: `03dd8da`（`expfam/src/experimental/k34_boundary_decomposition.py`、実行時 clean）。推論・評価のコードは import していない（テストで確認）。
- 系列 E（experimental prototype; 本文採用不可）

---

## 1. 問い

Phase 9X の K_true = 4（rep01..rep10）では、C_Lap が K4 を 8/10、K3 を 2/10、C_Q が K4 を 1/10、K3 以下を 9/10 選んだ。
既存の fit について、モデルの K = 3 と K = 4 の基準の差が、既存のどの成分の大きさの組み合わせで正負になったかを記述する。

## 2. 入力（読み取りのみ）

各ファイルを Phase 9X の merge commit `1cb316bdde7b05b9c17c73eafede0d2ce367e536` の内容とバイト単位で照合してから読んだ（SHA-256）:

| 入力 | SHA-256（merge 時点） |
|---|---|
| Phase 9X K4 `laplace_by_k.csv` | `7962718437372967c55ed3558f07e4590efabd6df5f63a11f380264ece82e800` |
| Phase 9X K4 `cq_decomposition.csv` | `727f2f5b3d4281f7a88d9ea9bf215fb5b41d10c61c358277ead34ce9b5a0a704` |
| Phase 9X `combined/dataset_rows.csv`（K_hat） | `61929c202263ea057e551417f1054dd7b6cd55bc820f62c0d64eaadb054339cf` |
| Phase 9K `laplace_by_k.csv`（K_true = 3 の比較用） | `d46c40622db10a2151ea3af568cc8dbf907beccd570b743b04fbf6abec4a934e` |
| Phase 9K `cq_decomposition.csv`（同） | `b4c1b9c52e7b01a31b433b7b0798b0efcef231c87713875dfee10f495bdf5931` |

## 3. 分解の定義（既存の定義をそのまま K3 → K4 に一般化）

符号: **Δ34 = C(K=4) − C(K=3)**。基準は小さい方が良いので、Δ34 < 0 なら K4 の方が良い。

**C_Lap**（Phase 9P/9T の `run_w_sensitivity.decomposition` と同じ定義）:
- `−2 ℓ_Lap = D_mode + V`、`D_mode = −2 φ(Ẑ) − ‖Ẑ‖² − nK ln 2π`（joint mode でのデータの当てはまり）、`V = ‖Ẑ‖² + log|H|`（Laplace の体積）、
  `C_Lap = −2 ℓ_Lap + d_K ln N`（`laplace_k_criterion.calc_C_Lap`）。
- `Δ34_Lap = −fit_gain_Lap_34 + volume_increment_34 + param_increment_34`、
  `fit_gain_Lap_34 = D_mode(3) − D_mode(4)`、`volume_increment_34 = V(4) − V(3)`、`param_increment_34 = [d_4 − d_3] ln N`。
- 使った列（`laplace_by_k.csv`）: `phi_at_mode`, `Z_hat_sq_norm`, `logdet_H`, `parameter_term`, `C_Lap`, `n_gaussian_selected`。
- 注: Issue の本文は volume を「logdet_H の差」と書いているが、同じ Issue が求める既存の定義では V に ‖Ẑ‖² も含まれる。既存の定義に従い、
  内訳として Δlog|H| と Δ‖Ẑ‖² も別に記録した（両者の和が volume_increment）。

**C_Q**（`run_joint_family_k_selection.cq_decomposition` と同じ定義。Schwarz BIC とは呼ばない）:
- `C_Q = D_K + P_Z + P_θ`、`D_K = −2(Q_X + Q_Y)`、`P_Z = −2 Q_Z`、`P_θ = num_params · ln n`。
- `Δ34_Q = −fit_gain_Q_34 + P_Z_increment_34 + param_increment_34`、`fit_gain_Q_34 = D_K(3) − D_K(4)`。
- 使った列（`cq_decomposition.csv`）: `D_K`, `P_Z`, `P_θ`, `num_params`。直接の値は `laplace_by_k.csv` の `C_Q`（refit の `bic`）。

**K4 が K3 に勝つ条件**と余裕（正なら K4 が良い）:
- C_Q: `fit_gain_Q_34 > P_Z_increment_34 + param_increment_34`、`margin_Q = fit_gain_Q_34 − (P_Z_increment_34 + param_increment_34)`
- C_Lap: `fit_gain_Lap_34 > volume_increment_34 + param_increment_34`、`margin_Lap = fit_gain_Lap_34 − (volume_increment_34 + param_increment_34)`

## 4. 事前の理論的な文脈（DERIVED、観察の結果ではない）

- **パラメータの増分**: 実際の `d_K = K d − K(K−1)/2 + n_gaussian_selected`（`run_laplace_pilot.d_K`、テストで同一性を確認）から、
  K3 → K4 の loading の数は 33 → 42 で **+9**（Gaussian の列の数はすべての refit で 3 のまま）。C_Lap（N = 75）でも C_Q（n = 75）でも増分は `9 ln 75 = 38.857`。
  C_Q の `num_params` は d_K と一致した（10/10）。
- **P_Z の増分**: `P_Z = −2Q_Z = nK ln(2π var_z) + mean_l ‖Z_l‖²/var_z`。var_z = 1 と scale_Z の規則（mean_l ‖Z_l‖² = nK）のもとで `nK(1 + ln 2π)` となり、
  次元を 1 つ増やすごとに `n(1 + ln 2π) = 212.8408`（n = 75）。commit 済みの値でも 10/10 で 212.84 だった。
- したがって C_Q で K4 が K3 に勝つには、MC 平均の fit gain が一定の **251.70**（212.84 + 38.86）を超える必要がある。
  C_Lap では、その閾値にあたる `volume_increment + 38.86` が fit ごとに変わる。
- これは基準の構造であり、過小選択の原因の証明ではない。

## 5. 再構成の確認

| 条件 | 利用可能 | C_Lap の最大の残差 | C_Q の最大の残差 |
|---|---|---|---|
| K_true = 4 | 10/10 | **1.6e-12** | **1.4e-12** |
| K_true = 3（比較用） | 10/10 | 5.4e-13 | 1.4e-12 |

すべて許容値 1e-10 以下。C_Lap の parameter_term の差は実際の d_K から計算した値と、C_Q の P_θ の差は num_params から計算した値と一致した。

## 6. K_true = 4 の replicate ごとの値

| rep | K_hat Lap / Q | Δ34_Lap | fit_gain_Lap | volume（Δlog\|H\| + Δ‖Ẑ‖²） | param | margin_Lap | Δ34_Q | fit_gain_Q | P_Z | param | margin_Q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rep01 | 4 / 2 | −15.06 | 258.55 | 204.63 (134.41 + 70.22) | 38.86 | 15.06 | 47.44 | 204.26 | 212.84 | 38.86 | −47.44 |
| rep02 | 4 / 2 | −21.11 | 268.58 | 208.61 (138.56 + 70.05) | 38.86 | 21.11 | 46.81 | 204.89 | 212.84 | 38.86 | −46.81 |
| rep03 | 3 / 1 | **1.13** | 219.97 | 182.24 (115.14 + 67.10) | 38.86 | **−1.13** | 92.77 | 158.93 | 212.84 | 38.86 | −92.77 |
| rep04 | 4 / 3 | −11.73 | 243.93 | 193.34 (118.91 + 74.42) | 38.86 | 11.73 | 89.66 | 162.04 | 212.84 | 38.86 | −89.66 |
| rep05 | 4 / 3 | −46.04 | 312.87 | 227.97 (155.83 + 72.15) | 38.86 | 46.04 | 8.01 | 243.69 | 212.84 | 38.86 | −8.01 |
| rep06 | 3 / 2 | **9.58** | 208.16 | 178.89 (111.79 + 67.09) | 38.86 | **−9.58** | 100.01 | 151.69 | 212.84 | 38.86 | −100.01 |
| rep07 | 4 / 2 | −8.25 | 245.51 | 198.40 (122.43 + 75.97) | 38.86 | 8.25 | 85.97 | 165.73 | 212.84 | 38.86 | −85.97 |
| rep08 | 4 / 3 | −8.96 | 250.96 | 203.13 (125.48 + 77.66) | 38.86 | 8.96 | 93.25 | 158.45 | 212.84 | 38.86 | −93.25 |
| rep09 | 4 / 4 | −86.58 | 377.13 | 251.70 (178.33 + 73.37) | 38.86 | 86.58 | **−66.31** | 318.01 | 212.84 | 38.86 | **66.31** |
| rep10 | 4 / 2 | −30.45 | 276.30 | 206.99 (129.72 + 77.27) | 38.86 | 30.45 | 88.30 | 163.40 | 212.84 | 38.86 | −88.30 |

残差は C_Lap で 8.5e-14〜1.6e-12、C_Q で 3.7e-13〜1.4e-12（`ktrue4_rows.csv`）。
C_Q の K_hat は K3 と K4 の比較だけでは決まらない（rep01 などは K2 を選んだ）。ここでは K3 と K4 の境界だけを分解している。

## 7. K_true = 4 の要約（min / median / max; 負 / 正）

| 成分 | 値 |
|---|---|
| Δ34_Lap | −86.58 / −13.40 / 9.58（8 / 2） |
| fit_gain_Lap_34 | 208.16 / 254.75 / 377.13 |
| volume_increment_34 | 178.89 / 203.88 / 251.70 |
| 　うち Δlog\|H\| | 111.79 / 127.60 / 178.33 |
| 　うち Δ‖Ẑ‖² | 67.09 / 72.76 / 77.66 |
| param_increment（両基準） | 38.86（一定） |
| **margin_Lap** | **−9.58 / 13.40 / 86.58（負 2 / 正 8）** |
| Δ34_Q | −66.31 / 87.13 / 100.01（1 / 9） |
| fit_gain_Q_34 | 151.69 / 164.57 / 318.01 |
| P_Z_increment_34 | 212.84（一定） |
| **margin_Q** | **−100.01 / −87.13 / 66.31（負 9 / 正 1）** |

記述:
- C_Lap では fit gain（median 254.8）が「体積の増分 + 38.86」（median 242.7）をわずかに上回り、余裕は 8/10 で正、median 13.4。
  余裕が負の 2 件（rep03 −1.13、rep06 −9.58）は、fit gain が最も小さい 2 件（220.0、208.2）だった。
- C_Q では fit gain（median 164.6）が一定の閾値 251.70 を 9/10 で下回り、余裕の median は −87.1。閾値を超えたのは rep09（fit gain 318.0）だけで、rep05（243.7）は −8.0 で届かなかった。
- 同じ fit で、C_Q の fit gain（MC 平均）は C_Lap の fit gain（joint mode）より median で約 90 小さく、C_Lap の体積の増分（median 203.9）は C_Q の P_Z の増分（212.84）より小さかった。

## 8. 選ばれた側による記述的な group（探索的、確認的ではない）

group の median だけを示す。検定・回帰・相関の探索・分類器・閾値の調整はしていない。

| group | n | fit_gain_Lap | volume | margin_Lap | fit_gain_Q | P_Z | margin_Q |
|---|---|---|---|---|---|---|---|
| Lap_K4_over_K3 | 8 | 263.56 | 205.81 | 18.09 | 184.99 | 212.84 | −66.70 |
| Lap_K3_over_K4（rep03, rep06） | 2 | 214.07 | 180.56 | −5.35 | 155.31 | 212.84 | −96.39 |
| Q_K4_over_K3（rep09） | 1 | 377.13 | 251.70 | 86.58 | 318.01 | 212.84 | 66.31 |
| Q_K3_over_K4 | 9 | 250.96 | 203.13 | 11.73 | 163.40 | 212.84 | −88.30 |

## 9. K_true = 3 の比較（Phase 9K rep01..rep10、context のみ）

同じ seed の label でも同じ dataset ではない。対応のある推論や K_true の因果効果の主張はしない。median だけを並べる。

| 成分（median） | K_true = 3 | K_true = 4 |
|---|---|---|
| Δ34_Lap | 52.08（正 10/10、K3 が良い） | −13.40 |
| fit_gain_Lap_34 | 144.86 | 254.75 |
| volume_increment_34（Δlog\|H\| / Δ‖Ẑ‖²） | 156.43（87.34 / 69.33） | 203.88（127.60 / 72.76） |
| margin_Lap | −52.08（負 10/10） | 13.40 |
| Δ34_Q | 163.61 | 87.13 |
| fit_gain_Q_34 | 88.09 | 164.57 |
| P_Z_increment_34 / param | 212.84 / 38.86 | 212.84 / 38.86 |
| margin_Q | −163.61（負 10/10） | −87.13 |

K_true = 3 では K4 を加えても両基準とも K3 が良く（過大選択なし）、K_true = 4 では C_Lap の余裕の中央値が正に移った一方、C_Q の余裕はまだ負だった。

## 10. DECISION

## **DECISION: K34_BOUNDARY_DECOMPOSITION_CHARACTERIZED**

K_true = 4 の 10 replicate すべてで、両基準の K3 → K4 の差を commit 済みの成分から許容値 1e-10 以内（最大 1.6e-12）で再構成でき、事前に決めた表をすべて作れた。
この判定は、分解が仮説を支持するかどうかによらない。

## 11. CAN SAY / CANNOT SAY

**CAN SAY**
- Phase 9X の既存の fit では、K3 → K4 の C_Lap の差は「joint mode での fit gain」と「体積の増分 + 9 ln 75」の差で決まり、fit gain が上回ったのが 8/10、下回ったのが 2/10（rep03, rep06、fit gain が最も小さい 2 件）だった。
- 同じ fit で、K3 → K4 の C_Q の差は「MC 平均の fit gain」と一定の「P_Z の増分 212.84 + 9 ln 75 = 251.70」の差で決まり、fit gain がこの一定の閾値を超えたのは 1/10 だけだった。
- 同じ fit で、C_Q の fit gain は C_Lap の fit gain より小さく（median で約 90）、C_Lap の体積の増分は C_Q の P_Z の増分より小さかった。

**CANNOT SAY**
- 過小選択の因果の原因。P_Z が原因であることの証明、1 次元あたりの信号の希釈が原因であることの証明（fit gain がなぜその大きさなのかは分解からは分からない）。
- 基準の優越性、一致性、一般の recovery、この固定の設計の外への一般化。
- K_true = 3 と 4 の違いを K_true の効果とすること。

## 12. 最大の UNRESOLVED

C_Q で K4 が勝てない直接の形は「fit gain < 一定の 251.70」だが、fit gain が小さい理由（MC 平均の fit と joint mode での fit の違い、1 次元あたりの信号の大きさ、MCEM の 8 反復の fit の誤差など）は、
commit 済みの成分の分解からは識別できない。C_Lap の 2 件の境界（rep03 −1.13、rep06 −9.58）も、fit の最適化の誤差の範囲にあるかどうかは分からない（本 Issue では局所最適化をしていない）。
次の段階は新しい Human Gate が必要である。
