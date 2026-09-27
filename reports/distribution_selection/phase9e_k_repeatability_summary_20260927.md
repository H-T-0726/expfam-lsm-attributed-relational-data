# Phase 9E — K-selection repeatability under the Phase 9D condition（2026-09-27）

- 位置づけ: **CHARACTERIZATION**（Issue #81 `PHASE9E_RESEARCH_FIRST_REPEATABILITY_FROZEN`）
- 系列: **E（experimental / objective-consistent / per-column prototype; 本文採用不可）**
- 一次データ: `expfam/results/k_repeatability/phase9e_20260927/`
  （`k_repeatability.csv`, `repeatability_summary.json`, `cq_by_k.csv`, `cq_decomposition.csv`, `family_by_k.csv`, `selection_trace.csv`, `execution_ledger.csv`）
- 実行コード: `56e3cfc`（git_dirty = False）。20 dataset を **1 回だけ**実行。rerun / reseed / redraw / 置換なし。

---

## 1. 研究上の問い

Phase 9D と同じ synthetic 条件・同じ joint family+K pipeline で、**独立に生成した dataset だけを変えたとき**、
現行の `C_Q(K)` が K=2 と K=3 をどの程度選び分けるか。
特に「Phase 9D rep1 の K=2 選択は単発の変動か、同条件で繰り返し起こる現象か」を記述する。

## 2. 何をしたか

- 条件は Phase 9D と同一（n=75, d=12, K_true=3, X = G×3 / B×6 / P×3, family_y=bernoulli, σ²_x=1, w0=−1, w=1, f_scale=√2, L=5,
  exploration/refit 8/8, K∈{1..5}, BFGS maxiter 2000 / gtol 1e-10, grad_inf ≤ 1e-8 は diagnostic）。
- 新規 20 dataset（rep01–rep20, seed 971000+r / 972000+r / 973000+r）。dataset ごとに 1 回だけ生成し、K=1..5 で共有。**start_B のみ**。
- 各 K を独立に fit（support gate → exploration → assignment 固定 → fresh refit → direct `C_Q(K)`）。K 間の warm start・family 持ち越しなし。
- 実装は Phase 9D runner の最小拡張（start の設定化と dataset 単位の失敗記録）と、集計用の小さな wrapper のみ。基準・選択器・モデルは不変。
- `C_Q(K) = −2 Q_strict + p_K ln n`（Q-based complete-data / ICL-type。**Schwarz BIC ではない**）、`K_hat = argmin_K C_Q(K)`（完全同値なら小さい K）。

## 3. 一次結果（OBSERVED）

### 実行の健全性

| 項目 | 値 |
|---|---|
| completed datasets | **20 / 20**（incomplete 0） |
| 実 EM 実行 | **200 / 200** SUCCESS |
| retry / replacement / seed rescue | 0 / 0 / 0 |
| C_Q = −2Q + p_K ln n の再計算 | 全 100 行一致 |
| selected/installed loading provenance（#74 checks 再利用） | blocking finding 0 |
| `delta_23 = penalty_increase_23 − fit_gain_23` | 最大残差 3.1e-12 |
| ambiguous true-Poisson | 0 件（score 決定列は全 dataset で列 3–8） |

### P1 — K_hat の分布（denominator = 20 completed datasets）

| K_hat | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| count | 0 | **3** | **17** | 0 | 0 |
| proportion | 0.00 | 0.15 | 0.85 | 0.00 | 0.00 |

- **exact 17 / under 3 / over 0**。K=2 を選んだのは rep01, rep03, rep10。
- これはこの 1 条件で観測された頻度であり、K 選択の一致性や一般的な成功確率ではない。
- 過大選択はなかった。全 dataset で `C_Q(4) − C_Q(3) ≥ 138.56`、`C_Q(2) − C_Q(1) ≤ −19.41`。
  最良・次点は 16 dataset が (3, 2)、3 dataset が (2, 3)、1 dataset（rep13）が (3, 4)。

### P2 — K=2 vs K=3 の差 `delta_23 = C_Q(3) − C_Q(2)`

| dataset | delta_23 | dataset | delta_23 |
|---|---|---|---|
| rep01 | **+92.80** | rep11 | −128.12 |
| rep02 | −6.84 | rep12 | −71.51 |
| rep03 | **+30.85** | rep13 | −156.44 |
| rep04 | −2.99 | rep14 | −53.36 |
| rep05 | −17.33 | rep15 | −15.46 |
| rep06 | −89.82 | rep16 | −35.78 |
| rep07 | −183.52 | rep17 | −71.99 |
| rep08 | −124.55 | rep18 | −44.79 |
| rep09 | −2.07 | rep19 | −105.73 |
| rep10 | **+93.20** | rep20 | −44.53 |

- min **−183.52**, median **−44.66**, max **+93.20**。negative（K=3 優位）**17**、positive（K=2 優位）**3**、zero **0**。
- K=3 を選んだ 17 dataset のうち 3 つ（rep09 −2.07, rep04 −2.99, rep02 −6.84）は差が 10 未満だった。
  K=2 を選んだ 3 つは +30.85, +92.80, +93.20。（tolerance band は設けていない。記述のみ。）

### P3 — K=2 → 3 の分解

| 量 | min | median | max |
|---|---|---|---|
| `fit_gain_23 = D_2 − D_3` | 162.81 | 300.68 | 439.54 |
| `penalty_increase_23` | 256.0157 | 256.0157 | 256.0157 |

- `penalty_increase_23` は**全 dataset で同一の定数 256.02**（= `P_Z` の増分 n(1+ln 2π) = 212.84 と `P_θ` の増分 10 ln 75 = 43.18 の和）。
  全 dataset・全 K で選択 assignment が同じ（Gaussian 列 3）ため、p_K の差もデータに依存しなかった。
- したがってこの条件では、K=2 vs 3 の判定は **`fit_gain_23` が 256.02 を超えるかどうか**だけで決まった。
- `fit_gain_23` の分布: 平均 302.9, 標準偏差 73.5。**256.02 未満は 3 / 20**（162.81, 163.21, 225.16 → いずれも K=2 選択）。
  256.02 をわずかに超えた dataset（258.09, 259.01, 262.85）が 3 つあり、これらが P2 の差 10 未満の 3 dataset に対応する。

### P4（secondary）— selected K での family

- 列 3–8 は全 20 dataset の K_hat で **Bernoulli ×6**、**Bernoulli → Poisson 誤選択 0 / 120**。
  K_hat での margin は 45.0–73.7。
- **全 20 dataset・全 K で列 3–8 は Bernoulli のまま**（全 4800 selection 行で Bernoulli）。trace margin の最小は 42.67。

### 候補収束 diagnostic（WARNING）

| 項目 | 値 |
|---|---|
| grad_inf > 1e-8 | **1057 / 9600** 候補評価（1006 / 4800 selection 行）; 最終 iteration 候補の非収束 142 / 1200 |
| K 別件数（各 K 1920 評価） | K=1: 99, 2: 144, 3: 207, 4: 277, 5: 330 |
| grad_inf | min 1.00e-8, median 3.56e-8, max 1.08e-6（>1e-7: 234, >1e-6: 1） |
| SciPy status | 全件 2（precision loss）; maxiter 到達 0; BFGS 反復最大 32 |
| 候補 family | Poisson 712, Bernoulli 345 |

warning は K とともに増えたが、family 選択は全行で同じで、primary 結果を壊している証拠は見つからなかった。

## 4. 解釈（INTERPRETATION — 因果の断定はしない）

1. **観測**: この条件では K=3 が多数（17/20）だが、K=2 が一定数（3/20）起こる。K=1, 4, 5 は選ばれない。
2. **Phase 9D rep1 との関係**: Phase 9D rep1 は `fit_gain_23 = 247.43`（delta_23 = +8.58）だった。これは Phase 9E の分布
   （162.81–439.54, 20 件中 3 件が 256.02 未満）の下側に入るが、外れ値ではない（rep03 の 225.16 と rep02 の 262.85 の間）。
   **rep1 型の K=2 選択は、同条件で一定の頻度で再現する現象**として観測された。
3. **K=2/3 境界**: 現行 `C_Q` では、K を 2→3 に上げるときの罰則増加がデータによらず 256.02 で一定であり、そのうち 212.84 は
   データ非依存の `P_Z` 増分である。K=2 か 3 かは、dataset ごとの fit 改善 `D_2 − D_3` がこの一定値を超えるかで分かれ、
   fit 改善の dataset 間ばらつき（sd ≈ 73）が境界（256.02）をまたいでいる。
   これは現行基準の挙動の記述であり、`P_Z` を不適切と結論するものではない。

## 5. まだ言えないこと・限界

- **K 選択の一致性**、**漸近的妥当性**、**一般的な under-selection bias**、**他条件（n, d, 信号強度, K grid）への頑健性**、
  **現行罰則理論の正しさ**、**実データでの有効性**は言えない。1 条件・20 dataset の観測頻度である。
- 20 dataset からの 3/20 は粗い推定であり、頻度の区間推定や条件依存性はこの run では扱っていない。
- `fit_gain_23` のばらつきが何に由来するか（データの信号の実現値、MC-EM の L=5 近似、8 反復の refit など）は分離していない。
- 候補収束 warning の score への影響は測っていない。
- 系列 E の prototype であり、本文証拠に昇格しない。

## 6. 次に研究判断が必要な点

1. この頻度（K=2 が 20 中 3）を、現行 `C_Q` の挙動として受け入れて先へ進むか、K 選択の理論（特に `P_Z` と `P_θ` の二重罰則という #72 の論点）を別 phase で扱うか。
   後者なら新しい frozen protocol と Human Gate が必要（本タスクでは criterion を一切変更していない）。
2. `fit_gain_23` のばらつきの由来（信号実現値 vs 推定の近似誤差）を区別する必要があるか。研究判断が変わる場合に限って検討する。

---

## Appendix（参考; 事前に計画した合算ではない）

Phase 9D（Issue #75）の 3 dataset は historical evidence として別に保持している。参考までに raw count を並べると:

| | K=2 | K=3 | 計 |
|---|---|---|---|
| Phase 9D（3 dataset, start 共通） | 1 | 2 | 3 |
| Phase 9E（20 dataset, start_B） | 3 | 17 | 20 |

2 つは別々に計画された実験であり、23 dataset を事前に計画したものではない。Phase 9E の primary 結果は上の 20 dataset のみで計算している。
