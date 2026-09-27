# Phase 9I — Candidate B（Laplace）の frozen 3-dataset pilot（2026-09-28）

- 位置づけ: **FROZEN SMALL PILOT / CHARACTERIZATION**（Issue #90）。一致性・頑健性の研究ではない。
- 系列: **E（experimental prototype; 本文採用不可）**
- 一次データ: `expfam/results/laplace_pilot/phase9i_20260928/`
  （`laplace_by_k.csv`, `laplace_pilot_summary.json`, `cq_by_k.csv`, `cq_decomposition.csv`, `family_by_k.csv`, `execution_ledger.csv`）
- 実行コード: `84182a4`（git_dirty = False）。3 dataset を **1 回だけ**実行。rerun / reseed / 追加 replicate なし。

---

## 1. 研究上の問い

Phase 9D/9E と同じ条件の新しい 3 dataset で、(1) Candidate B の `C_Lap` を K=1..5 すべてで評価できるか、
(2) `C_Lap` はどの K を選ぶか、(3) **同じ refit** での現行 `C_Q` と K の順序がどう違うか。どちらが優れているかは問わない。

## 2. 何をしたか

- 条件: n=75, d=12, K_true=3, X = G×3 / B×6 / P×3, family_y=Bernoulli, σ²_x=1, w0=−1, w=1, f_scale=√2, L=5, exploration/refit 8/8, K∈{1..5}, start_B のみ。
- seeds: rep01–rep03 = 981000+r / 982000+r / 983000+r。dataset は 1 回だけ生成し、全 K で共有。
- 各 (dataset, K): 既存の family-selection exploration → assignment 固定 → fresh refit（Phase 9D/9E の runner をそのまま使用）→
  refit の `C_Q`（既存の値）→ **同じ refit の θ で Candidate B を事後評価**（Z0 = refit の Z_est、grad_tol 1e-8、max_iter 200、restart・ridge・jitter なし）。
- `C_Lap = −2·laplace_log_observed + d_K ln N`、`N = 75`（working convention; 理論的に確定した sample size ではない）、
  `d_K = Kd − K(K−1)/2 + n_gaussian_selected(K)`、w0/w は数えない。

## 3. 結果（OBSERVED）

### 実行の健全性

| 項目 | 値 |
|---|---|
| 実 EM 実行 | **30 / 30** SUCCESS（15 refit すべて成功） |
| retry / replacement / seed rescue | 0 / 0 / 0 |
| 候補収束 warning（diagnostic） | 165 / 1440 候補評価 |
| selected assignment | 全 15 refit で G×3 / B×6 / P×3（n_gaussian_selected = 3） |

### P0 — Candidate B の評価可能性

| status | 件数 |
|---|---|
| OK | **15 / 15** |
| NOT_STATIONARY | 0 |
| HESSIAN_NOT_PD | 0 |
| 評価エラー | 0 |
| C_Lap-complete dataset | **3 / 3** |

joint mode の Newton 反復は 4–11 回、最終 `max|∇Phi|` は 2.4e-15 – 7.6e-9（すべて ≤ 1e-8）、
H の最小固有値は 0.235 – 6.0（すべて正）、`log|H|` は 205.7（K=1）から 740.9（K=5）。

### P1/P2 — 同じ refit での C_Q と C_Lap

| rep | K | C_Q | C_Lap | laplace_log_observed |
|---|---|---|---|---|
| rep01 | 1 | 5204.03 | 5188.60 | −2561.92 |
| rep01 | 2 | **5114.69** | 5060.93 | −2474.34 |
| rep01 | 3 | 5189.74 | **5059.41** | −2451.99 |
| rep01 | 4 | 5353.18 | 5105.76 | −2455.73 |
| rep01 | 5 | 5509.62 | 5149.91 | −2460.54 |
| rep02 | 1 | 5149.95 | 5166.15 | −2550.69 |
| rep02 | 2 | **5011.14** | 4999.99 | −2443.87 |
| rep02 | 3 | 5079.44 | **4988.01** | −2416.29 |
| rep02 | 4 | 5266.00 | 5041.22 | −2423.47 |
| rep02 | 5 | 5420.83 | 5088.63 | −2429.90 |
| rep03 | 1 | 5414.61 | 5411.40 | −2673.32 |
| rep03 | 2 | 5316.81 | 5275.41 | −2581.58 |
| rep03 | 3 | **5173.35** | **5103.66** | −2474.12 |
| rep03 | 4 | 5328.26 | 5144.17 | −2474.94 |
| rep03 | 5 | 5540.89 | 5207.30 | −2489.24 |

| rep | K_hat_Q | K_hat_Lap | delta_Q_23 = C_Q(3)−C_Q(2) | delta_Lap_23 = C_Lap(3)−C_Lap(2) |
|---|---|---|---|---|
| rep01 | 2 | 3 | +75.05 | **−1.53** |
| rep02 | 2 | 3 | +68.30 | −11.98 |
| rep03 | 3 | 3 | −143.46 | −171.74 |

- `C_Lap` は 3 dataset すべてで K=3 を選んだ。`C_Q` は rep01・rep02 で K=2、rep03 で K=3 を選んだ。
- rep01 の `C_Lap` は K=2 と K=3 の差が 1.53 しかない（tolerance band は設けていない。記述のみ）。
- `C_Lap` の K=4, 5 は K=3 より 40–104 大きく、過大選択はなかった。
- `laplace_log_observed`（θ̂ での観測尤度の Laplace 近似）は rep01・rep02 で K=3 が最大、rep03 では K=3 と K=4 がほぼ同じ（−2474.12 vs −2474.94）。

### P3 — Candidate B の分解（K=2 → 3）

`C_Lap = D_mode + Vol + P_θ`、`D_mode = −2[ℓ_X + ℓ_Y](Ẑ)`、`Vol = ‖Ẑ‖² + log|H|`。
`C_Q = D_K + P_Z + P_θ`（D_K は MC サンプル平均、P_Z = nK(1+ln 2π)）。

| rep | C_Lap: fit gain (D_mode) | Vol 増分 | C_Q: fit gain (D_K) | P_Z 増分 | P_θ 増分（共通） |
|---|---|---|---|---|---|
| rep01 | 251.73 | 207.02 | 180.96 | 212.84 | 43.17 |
| rep02 | 262.66 | 207.51 | 187.72 | 212.84 | 43.17 |
| rep03 | 484.06 | 269.14 | 399.48 | 212.84 | 43.17 |

各 K の値（`Vol` の内訳 `‖Ẑ‖²` ≈ 72K、`log|H|`）は `laplace_by_k.csv` にある。

### P4 — family（secondary）

全 15 refit・全 K で列 3–8 は Bernoulli、gate 決定列は Gaussian×3 / Poisson×3。K_hat_Q と K_hat_Lap のどちらでも列 3–8 は Bernoulli×6。

## 4. 観測されたこと（OBSERVED / INTERPRETATION）

- **OBSERVED**: Candidate B は、実際の規模（nK ≤ 375）で 15 refit すべてについて停留点と正定値 H に到達し、評価できた。
- **OBSERVED**: 同じ refit でも、C_Q と C_Lap は 3 dataset 中 2 つで異なる K を選んだ（C_Q: 2, 2, 3、C_Lap: 3, 3, 3）。
- **INTERPRETATION（分解の記述）**: K=2→3 での Z 関連の増分は、C_Lap の `Vol`（207.0, 207.5, 269.1）と C_Q の `P_Z`（212.84）で**大きさが近かった**。
  2 つの基準の違いは主に**当てはまり項をどこで評価するか**にあり、joint mode での fit gain（251.7, 262.7）は MC サンプル平均での fit gain（181.0, 187.7）より大きかった。
  rep01・rep02 では、この差が delta の符号を変えた。
  これは本 pilot の 3 dataset での分解の記述であり、一般的な機構として確認したものではない。

## 5. 言えないこと

- **C_Lap が C_Q より優れている・改善した・正しい**とは言えない。3 dataset の記述であり、比較の優劣は研究質問ではない。
- C_Lap が K_true を**一致的に**回復するとは言えない。
- 3 dataset からの頻度（C_Lap: 3/3、C_Q: 1/3 で K=3）は推定値として扱えない。Phase 9E（C_Q で 17/20 が K=3）の頻度と比べてもいけない（別の dataset・別の規模）。
- rep01 の C_Lap の差 1.53 は小さく、Laplace 近似の誤差（Phase 9H §10 の UNRESOLVED; 非 Gaussian 事後）でどちらにも動きうる大きさかどうかは分からない。
- N = 75、d_K の慣行、θ̂ が 8 反復 MCEM の出力であることの影響は評価していない。
- 系列 E の prototype であり、本文の証拠にしない。

## 6. 次の Human 判断

1. この pilot（評価可能性 15/15、C_Lap は 3/3 で K=3、同じ refit でも C_Q と順序が異なる）を踏まえて、Candidate B を事前固定の より大きな比較に進めるか。
2. 進める場合、rep01 のような小さな差を解釈するために、非 Gaussian 事後での Laplace 誤差の大きさを先に理論・数値で見積もるか。
3. いずれも新しい frozen protocol と Human Gate が必要（本タスクでは開始していない）。
