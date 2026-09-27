# Phase 9D Gate 75-B — joint family + K selection, research summary（2026-09-27）

- 位置づけ: **CONFIRMATORY WITHIN ONE FROZEN SYNTHETIC CONDITION**（Issue #75 `PHASE9D_PROTOCOL_FROZEN`）
- 実行承認: Issue #75 Human Gate comment 5855763104
- 系列: **E（experimental / objective-consistent / per-column prototype; 本文採用不可）**
- 一次データ: `expfam/results/joint_family_k_selection/gate75b_20260927/`
  （`joint_selection.csv`, `cq_by_k.csv`, `family_by_k.csv`, `cq_decomposition.csv`, `selection_trace.csv`, `summary.json`, `audit_report.json`）
- 実行コード: `67e0edf`（git_dirty = False）。frozen protocol を **1 回だけ**実行。rerun / resume / reseed / redraw なし。

---

## 1. 研究上の問い

観測 X, Y と事前に与えた K 候補 `{1,…,5}` だけから、merged #74 の family 選択機構と既存の `C_Q(K)` を 1 本の監査可能な
pipeline として接続し、(1) score が決める X 列の family と (2) 潜在次元 K を、真の family・真の K を使わずに選べるか。

## 2. 何をしたか

- 凍結条件: n=75, d=12, K_true=3, X = Gaussian×3（列0–2）/ Bernoulli×6（列3–8）/ Poisson×3（列9–11）, family_y=bernoulli,
  σ²_x=1, w0=−1, w=1, f_scale=√2, L=5, exploration/refit num_iter 8/8。
- 3 replicate（data/search/refit seed 96100r/96200r/96300r）× 2 start × K∈{1..5}。replicate ごとにデータを 1 回だけ生成し、全 K・両 start で共有。
- 各 (replicate, start, K) で #74 pipeline をゼロから実行（support gate → B/P 候補 score → exploration → assignment 固定 → fresh refit）。K 間の warm start・family 持ち越しなし。
- `K_hat = argmin_K C_Q(K)`（start ごと。完全同値なら小さい K）。`C_Q(K) = −2 Q_strict + p_K ln n`,
  `p_K = Kd − K(K−1)/2 + (選択後の Gaussian 列数)`。これは Q-based complete-data / ICL-type 基準であり **Schwarz BIC ではない**。
- 候補最適化は analytic-gradient BFGS（maxiter 2000, gtol 1e-10）。grad_inf ≤ 1e-8 は **diagnostic（WARNING）**。

## 3. 一次結果（OBSERVED）

### Technical validity

| 項目 | 値 |
|---|---|
| technical_validity | **VALID** |
| 実 EM 実行 | **60 / 60** SUCCESS |
| 独立 audit | PASS, BLOCKER 0, HIGH 0, WARNING 1（候補収束 diagnostic） |
| retry / replacement / seed rescue | 0 / 0 / 0 |
| データ生成 | replicate ごと 1 回。ambiguous_true_poisson 0 件 |
| decomposition 整合 | `|D_K + P_Z + P_θ − C_Q|` 最大 1.8e-12 |

### P1 — K recovery

| path | K_hat | 区分 | C_Q(1) | C_Q(2) | C_Q(3) | C_Q(4) | C_Q(5) | runner-up gap | 順位 |
|---|---|---|---|---|---|---|---|---|---|
| rep1 / start_B | **2** | under | 5382.625 | **5333.020** | 5341.604 | 5517.493 | 5737.582 | 8.584 | 2,3,1,4,5 |
| rep1 / start_P | **2** | under | 5382.625 | **5333.020** | 5341.604 | 5517.493 | 5737.582 | 8.584 | 2,3,1,4,5 |
| rep2 / start_B | **3** | exact | 5303.367 | 5193.320 | **5152.722** | 5349.842 | 5531.293 | 40.598 | 3,2,1,4,5 |
| rep2 / start_P | **3** | exact | 5303.367 | 5193.320 | **5152.722** | 5349.842 | 5531.293 | 40.598 | 3,2,1,4,5 |
| rep3 / start_B | **3** | exact | 5583.194 | 5517.173 | **5441.703** | 5629.297 | 5802.094 | 75.470 | 3,2,1,4,5 |
| rep3 / start_P | **3** | exact | 5583.194 | 5517.173 | **5441.703** | 5629.297 | 5802.094 | 75.470 | 3,2,1,4,5 |

- **exact 4 / under 2 / over 0**（6 path; start 間はデータ共有のため独立ではない。replicate 単位では exact 2 / under 1）。
- 過大選択（K=4, 5）は 1 件もない。K=4, 5 はどの replicate でも K=1 よりも C_Q が大きい。
- rep1 の under は差 8.58（K=2 vs K=3）と小さい。rep2 / rep3 の K=3 選択の差は 40.6 / 75.5。

### P2 — 選択 K での family recovery（score が決める列 3–8 のみ）

- 6 path すべてで列 3–8 は **Bernoulli ×6**。**Bernoulli → Poisson 誤選択 0 / 36**、**all-six exact 6 / 6**。
- K_hat での最終 margin（−2 score 単位, Bernoulli 優位）: 45.61–58.87。
- K_hat 以外も含め、**全 30 (replicate, start, K) で列 3–8 は Bernoulli**、gate 決定列は Gaussian×3・Poisson×3。
  全 1440 selection 行で Bernoulli が選ばれ、変化は start_P の iteration 1（Poisson → Bernoulli, 90 件）のみ。全 K の最終 margin 45.34–64.23。

### P3 — joint exact recovery

- **4 / 6 path**（rep2, rep3 の両 start）。rep1 の 2 path は family は完全だが K_hat=2 のため joint exact ではない。

### P4 — start stability

| replicate | K_hat 一致 | K_hat での assignment 一致 | C_Q 順位一致 |
|---|---|---|---|
| rep1 | yes | yes | yes |
| rep2 | yes | yes | yes |
| rep3 | yes | yes | yes |

補足（OBSERVED）: 全 (replicate, K) で start_B と start_P の `C_Q` は**ビット単位で一致**した。両 start が同じ assignment に到達し、
fresh refit は同じ assignment・同じ refit seed で実行されるため。したがって start 一致は「選択された family が一致した」ことの帰結であり、
K 選択について独立な追加情報ではない。

### Diagnostic — C_Q decomposition（#72; 選択には使っていない）

start_B（start_P と同値）:

| rep | K | D_K | P_Z | P_θ | C_Q |
|---|---|---|---|---|---|
| rep1 | 1 | 5105.02 | 212.84 | 64.76 | 5382.625 |
| rep1 | 2 | 4795.08 | 425.68 | 112.25 | 5333.020 |
| rep1 | 3 | 4547.65 | 638.52 | 155.43 | 5341.604 |
| rep1 | 4 | 4471.84 | 851.36 | 194.29 | 5517.493 |
| rep1 | 5 | 4444.55 | 1064.20 | 228.83 | 5737.582 |
| rep2 | 1 | 5025.76 | 212.84 | 64.76 | 5303.367 |
| rep2 | 2 | 4655.38 | 425.68 | 112.25 | 5193.320 |
| rep2 | 3 | 4358.77 | 638.52 | 155.43 | 5152.722 |
| rep2 | 4 | 4304.19 | 851.36 | 194.29 | 5349.842 |
| rep2 | 5 | 4238.26 | 1064.20 | 228.83 | 5531.293 |
| rep3 | 1 | 5305.59 | 212.84 | 64.76 | 5583.194 |
| rep3 | 2 | 4979.24 | 425.68 | 112.25 | 5517.173 |
| rep3 | 3 | 4647.75 | 638.52 | 155.43 | 5441.703 |
| rep3 | 4 | 4583.65 | 851.36 | 194.29 | 5629.297 |
| rep3 | 5 | 4509.06 | 1064.20 | 228.83 | 5802.094 |


- `P_Z(K) = nK(1 + ln 2π) = 212.84 K` はデータ非依存の K 比例項（#72 の結論どおり）。
- rep1 の K=2 → 3: D_K の減少 247.43 に対し、罰則の増加は P_Z 212.84 + P_θ 43.18 = 256.02。差 8.58 で K=2 が選ばれた。
  rep2 / rep3 の同じ区間では D_K の減少は 296.6 / 331.5 で、罰則増加 256.02 を上回った。
- K=3 → 4 では、D_K の減少（54.6–75.8）が罰則増加（212.84 + 38.86 = 251.70）を大きく下回った。

### 候補収束 diagnostic（WARNING）

| 項目 | 値 |
|---|---|
| grad_inf > 1e-8 | **305 / 2880** 候補評価（289 / 1440 selection 行） |
| 最終 iteration 候補の非収束 | 36 / 360 |
| SciPy status | 305 件すべて 2（precision loss）; maxiter 到達 0; BFGS 反復最大 16 |
| grad_inf | min 1.00e-8, median 3.80e-8, max 1.38e-6（>1e-7: 83, >1e-6: 2） |
| 候補 family | Poisson 198, Bernoulli 107 |
| K 別件数 | K=1: 37, 2: 56, 3: 61, 4: 64, 5: 87 |

warning 行を含め selection 行の margin は最小 44.91。

### Fresh refit（副次）

Q_strict と p_K は `cq_by_k.csv` に全 30 行を記録（p_K = 15, 26, 36, 45, 53; Gaussian 列数は全行で選択 assignment から 3）。
refit runtime 8.65–10.79 s / K。

## 4. 言えること

凍結条件・この K 候補・3 replicate の下で、統合 pipeline は事前に定めた規則どおりに family と K を選び、

- score が決める真 Bernoulli 列は、全 K・全 path で Bernoulli に選ばれた（誤選択 0）
- K は 6 path 中 4 path で K_true=3 を選び（replicate 単位で 2 / 3）、残り 2 path（rep1）は K=2 を選んだ。過大選択はなかった
- joint exact は 4 / 6 path

を観測した。技術的には 60/60 実行・integrity 0・監査 BLOCKER/HIGH 0 で有効。

## 5. まだ言えないこと・限界

- **K 選択の一致性**、**一般的な family 選択の妥当性**、**joint 選択の一般的信頼性**、**実データでの有効性**、
  **人手指定への優越**、**n/d/信号強度/候補グリッドへの頑健性**は言えない（1 条件・3 replicate）。
- 6 path は 3 データの 2 start で、独立な 6 標本ではない。信頼区間は付けない。start 一致は assignment 一致の帰結で、K について独立情報にならない。
- rep1 の under 選択は差 8.58 と小さく、replicate 間変動の範囲にあるのか系統的傾向なのかは、この 3 replicate からは判断できない。
- （INTERPRETATION / HYPOTHESIS）rep1 では、データ非依存の `P_Z(K)` が罰則増加の大部分（212.84 / 256.02）を占め、D_K の改善をわずかに上回った。
  #72 が指摘した「P_Z と P_θ が同方向の K 罰則として二重に効く」性質が、この条件で under 側に寄与した可能性がある。
  ただし本 run はこれを検証する設計ではなく、基準の修正は #75 の範囲外（frozen）。
- family search の離散的な選択は p_K で別途罰せられていない（#75 で凍結済みの既知の限界）。この条件では全 K で同じ assignment が選ばれたため、K 比較への影響は観測されていない。
- 候補収束 warning の score・選択への影響は測っていない（margin は最小 44.9）。
- 系列 E の prototype であり、本文証拠に昇格しない。

## 6. 次の Human / 指導教員の判断

1. この結果（family は安定に正しく、K は 3 replicate 中 2 で正解・1 で K=2）で Phase 9D の目的を満たしたとみなすか。
2. rep1 型の under 選択（P_Z と D_K の差が小さいケース）を、どう扱うか。
   追加 replicate / 条件で確かめるか、P_Z の扱いを理論課題として別 phase で検討するか。いずれも新しい frozen protocol と Human Gate が必要。
3. PR #80 の merge、Issue #75 の close（いずれも本タスクでは実施していない）。
