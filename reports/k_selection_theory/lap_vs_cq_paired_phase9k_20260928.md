# Phase 9K — C_Lap と C_Q の frozen 20-dataset paired characterization（2026-09-28）

- 位置づけ: **CONDITION-SPECIFIC PAIRED CHARACTERIZATION**（Issue #94）。一致性・一般的な優劣・頑健性の研究ではない。
- 系列: **E（experimental prototype; 本文採用不可）**
- 一次データ: `expfam/results/lap_vs_cq_20/phase9k_20260928/`
  （`paired_summary.json`, `laplace_by_k.csv`, `cq_by_k.csv`, `cq_decomposition.csv`, `family_by_k.csv`, `fitted_states.json`, `execution_ledger.csv`）
- 実行コード: `bbf45e4`（git_dirty = False）。20 dataset を **1 回だけ**実行。rerun / 置換 / 追加 replicate なし。

---

## 1. 研究上の問い

1 つの固定条件から事前に固定した新しい 20 dataset で、同じ refit に対して
(1) Candidate B の `C_Lap` はどの K を選ぶか、(2) 現行 `C_Q` はどの K を選ぶか、(3) 両者の選択はどこで一致・不一致になるか、
(4) `C_Lap` の最良と次点の差はどの程度か、(5) Candidate B は 100 個の K 別 refit で技術的に評価できるか。

## 2. 何をしたか

- 条件: n=75, d=12, K_true=3, X = G×3 / B×6 / P×3, Bernoulli-Y, σ²_x=1, w0=−1, w=1, f_scale=√2, L=5, 8/8 反復, K∈{1..5}, start_B のみ。
- seeds: rep r = 1001000+r / 1002000+r / 1003000+r（r=1..20）。dataset は 1 回だけ生成し、全 K で共有。
- 各 (dataset, K): 既存の exploration → assignment 固定 → fresh refit → 既存の `C_Q` → **同じ refit** で Candidate B（Phase 9I と同一設定:
  θ = refit 最終値、Z0 = refit の Z_est、grad_tol 1e-8、max_iter 200、restart・ridge・jitter なし）。
- `C_Lap = −2·laplace_log_observed + d_K ln 75`（N = 75 は working convention）、`d_K = Kd − K(K−1)/2 + n_gaussian_selected`。
  **IS 補正は使っていない**（Phase 9J で重みが強く退化したため）。
- K_hat_Lap は 5 つの K すべてが status OK の dataset だけで計算する（部分集合での argmin はしない）。

## 3. 結果（OBSERVED）

### 実行の健全性と P5（技術的な評価可能性）

| 項目 | 値 |
|---|---|
| 実 EM 実行 | **200 / 200** SUCCESS、成功した refit 100 / 100 |
| retry / replacement / seed rescue | 0 / 0 / 0 |
| Candidate B status | **OK 100** / NOT_STATIONARY 0 / HESSIAN_NOT_PD 0 / 評価エラー 0 |
| C_Lap-complete dataset | **20 / 20** |
| joint-mode Newton 反復 | 4–50（最大は rep13・K=1 の 50。次は 18, 14, 13） |
| 最終 max\|∇Phi\| | ≤ 9.1e-9 |
| H の最小固有値 | 0.057 – 6.53。K 別の最小: K=1 1.85, K=2 0.68, K=3 0.45, K=4 0.22, **K=5 0.057** |
| log\|H\| の K 別平均 | 197.3, 374.6, 530.3, 616.1, 690.5 |
| 候補収束 warning（diagnostic） | 1131 / 9600 |
| selected assignment | 100 refit すべてで G×3 / B×6 / P×3 |

### P1 / P2 — K の選択分布（denominator = 20）

| | K=1 | K=2 | K=3 | K=4 | K=5 | exact / under / over |
|---|---|---|---|---|---|---|
| **K_hat_Lap** | 0 | 1 | **19** | 0 | 0 | 19 / 1 / 0 |
| **K_hat_Q** | 1 | 5 | **14** | 0 | 0 | 14 / 6 / 0 |

この 1 条件・20 dataset での観測頻度であり、一般的な回復確率ではない。Phase 9E（C_Q で 17/20）とは別の dataset で、合算しない。

### P3 — paired の結果

| 区分 | dataset 数 |
|---|---|
| 両方 K=3 | 14 |
| C_Lap のみ K=3 | 5（rep04, rep07, rep10, rep12, rep19） |
| C_Q のみ K=3 | 0 |
| どちらも K=3 でない | 1（rep20: 両方 K=2） |
| 同じ K | 15 |
| 異なる K | 5 |

dataset ごとの (K_hat_Q, K_hat_Lap):
rep01 (3,3), rep02 (3,3), rep03 (3,3), rep04 (2,3), rep05 (3,3), rep06 (3,3), rep07 (1,3), rep08 (3,3), rep09 (3,3), rep10 (2,3),
rep11 (3,3), rep12 (2,3), rep13 (3,3), rep14 (3,3), rep15 (3,3), rep16 (3,3), rep17 (3,3), rep18 (3,3), rep19 (2,3), rep20 (2,2)。

### P4 — 差の大きさ

| 量 | min | median | max | 負 / 正 |
|---|---|---|---|---|
| delta_Lap_23 = C_Lap(3) − C_Lap(2) | −150.28 | −75.86 | +52.32 | 19 / 1 |
| delta_Q_23 = C_Q(3) − C_Q(2) | −130.24 | −3.65 | +157.07 | 14 / 6 |
| C_Lap の最良と次点の差 | **22.03** | **49.16** | **58.44** | — |
| C_Q の最良と次点の差 | 1.96 | 34.77 | 130.24 | — |

C_Lap の差が小さい順の 5 dataset: rep10 22.03（最良 3・次点 2）、rep04 29.15（3・2）、rep19 40.70（3・2）、rep09 42.99（3・4）、rep18 43.44（3・4）。
C_Lap の次点は 20 dataset 中 15 で K=4、5 で K=2（rep20 は最良 2・次点 3）。
C_Q は 5 dataset（rep05 −1.96, rep11 −2.09, rep15 −2.14, rep06 −2.87, rep09 −4.42）で delta_Q_23 の絶対値が 5 未満だった。
閾値は設けていない（記述のみ）。

C_Lap の K 曲線の全値は `paired_summary.json` の `datasets[*].C_Lap`、C_Q は `datasets[*].C_Q` にある。

### P6 — 分解（K=2 → 3、20 dataset）

| 量 | min | median | max |
|---|---|---|---|
| C_Lap: fit gain `D_mode(2) − D_mode(3)`（joint mode での当てはまり） | 150.8 | 347.8 | 447.1 |
| C_Lap: Laplace 体積の増分 `Δ(‖Ẑ‖² + log\|H\|)` | 160.0 | 231.5 | 255.3 |
| C_Q: fit gain `D_K(2) − D_K(3)`（MC サンプル平均） | 98.9 | 259.7 | 386.3 |
| C_Q: P_Z の増分 | 212.84（一定） | | |
| 両者共通: P_θ の増分 | 43.17（一定） | | |
| fit gain の差（C_Lap − C_Q） | **49.1** | **77.3** | **101.9** |

Candidate B の各 K の分解（`laplace_log_observed`, `integration_term`, `parameter_term`, `phi_at_mode`, `logdet_H`, `‖Ẑ‖²`, `d_K`, `N`）は
`laplace_by_k.csv`、C_Q の分解（`D_K`, `P_Z`, `P_theta`, `Q_strict`）は `cq_decomposition.csv` / `cq_by_k.csv` にある。

### P7 — family（secondary）

100 refit すべてで列 3–8 は Bernoulli、gate 決定列は Gaussian×3 / Poisson×3。K_hat_Q・K_hat_Lap のどちらでも列 3–8 は Bernoulli×6。

## 4. 観測されたこと（OBSERVED / INTERPRETATION）

- **OBSERVED**: Candidate B は 100 個すべての K 別 refit で停留点と正定値 H に到達し、20 dataset すべてで C_Lap-complete だった。
  ただし H の最小固有値は K とともに小さくなり、K=5 で最小 0.057 だった。
- **OBSERVED**: この条件の 20 dataset で、C_Lap は 19 で K=3、1 で K=2 を選び、C_Q は 14 で K=3、5 で K=2、1 で K=1 を選んだ。
  選択が異なった 5 dataset はすべて「C_Q が K<3、C_Lap が K=3」で、その逆（C_Q のみ K=3）はなかった。過大選択はどちらにもなかった。
- **OBSERVED**: C_Lap の最良と次点の差は 22.0–58.4 で、C_Q（2.0–130.2）より範囲が狭かった。
- **INTERPRETATION（分解の記述）**: K=2→3 の Z 関連の増分は、C_Lap の Laplace 体積（160–255、median 231.5）と C_Q の P_Z（212.84）で同じ程度だった。
  一方、joint mode での fit gain は MC サンプル平均での fit gain を**全 20 dataset で**上回った（差 49–102）。
  Phase 9I の 3 dataset で見た「違いは主に当てはまり項の評価点による」という記述と整合する。一般的な機構としては確認していない。

## 5. CAN SAY / CANNOT SAY

**CAN SAY**
- この 1 条件の事前固定 20 dataset で、C_Lap と C_Q はそれぞれ上の K 選択分布（C_Lap 19/1/0、C_Q 14/6/0 の exact/under/over）を示した。
- 同じ refit 上で、15 dataset で同じ K、5 dataset で異なる K（すべて C_Q が K<3、C_Lap が K=3）だった。
- Candidate B はこの規模で 100/100 の refit について技術的に評価できた。

**CANNOT SAY**
- 一致性、一般的な優劣、一般的な回復率、頑健性、他の n/d/K_true/信号への一般化。
- N = 75 の理論的な正しさ、厳密な周辺尤度、Laplace 誤差の上界、実データでの有効性。
- **小さな差（C_Lap の最小 22.0）と、非 Gaussian 事後での Laplace 近似誤差との大小関係は UNRESOLVED**（Phase 9J の IS は重みが退化しており、誤差の推定に使えない）。
- Phase 9E / 9I / 9J の頻度とは合算しない。

## 6. fitted state

`fitted_states.json`（commit 済みの primary artifact）: 100 refit それぞれの replicate、K、data/search/refit seed、family assignment、F、
Gaussian 分散（sigma 対角）、w0、w、var_z、最終 Z_est、code SHA `bbf45e4`、evaluator version。浮動小数点は往復可能な repr（float64 と完全一致）。
X, Y は保存しておらず、data seed から `run_family_selection_pilot.build_dataset(run_lap_vs_cq_20.PROTOCOL, replicate)` で完全に再生成できる。

## 7. 次の Human 判断

1. この 1 条件で C_Lap と C_Q の K 選択分布が異なった（特に C_Q の under 選択 6/20 に対して C_Lap 1/20）ことを踏まえ、
   他の条件（n, d, K_true, 信号強度）へ事前固定で広げるか。
2. C_Lap の差（最小 22）が非 Gaussian Laplace 誤差より大きいと言える根拠が要るか。要るなら、保存した fitted state 上で、より事後に合う推定法を事前固定で使う。
3. N と d_K の慣行（working convention）の理論的な扱い。
いずれも新しい frozen protocol と Human Gate が必要（本タスクでは開始していない）。
