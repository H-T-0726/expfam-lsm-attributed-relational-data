# Phase 9W — 信号を揃えた K_true 設計と較正（2026-09-28）

- 位置づけ: **THEORY / DESIGN / ZERO-EM CALIBRATION AND COMPATIBILITY AUDIT ONLY**（Issue #120）。EM・refit・family 選択・K 選択・新しい fitted state での Candidate B の評価なし。
- 結果: `expfam/results/matched_k_true_design/phase9w_20260928/`
  （`design.json`, `calibration_by_k.json/csv`, `deterministic_crosscheck.json/csv`, `zero_inference_context.csv`, `k1_boundary_audit.json`, `k3_anchor_compatibility.json`, `future_protocol.json`）
- 実行コード: `4c22d75`（`expfam/src/experimental/matched_k_true_design.py`、実行時 clean）。数値の設定はすべて実行前にコードで固定し、1 回だけ実行した。
  historical な generator・推定器・Candidate B・C_Q・Phase 9K の artifact は変更していない。
- 系列 E（experimental prototype; 本文採用不可）

---

## 1. 問い

これまでの Candidate B / C_Q の特徴づけはほぼすべて K_true = 3 だった。K_true ∈ {1, 2, 3, 4} で、X の信号・Y の信号・edge 密度が
K_true とともに変わらないよう主要な要約を揃えた有限標本の設計（将来の Phase 9X）を凍結できるか。

K_true = 5 は加えない（候補 K = 1..5 の境界になり、Phase 9K の anchor との比較が崩れる）。K_true = 4 でも過大側の候補 K = 5 が 1 つ残る。

## 2. 揃える量（DERIVED、実装で確認）

**X の loading energy**: `F = Q · f_scale`（Q は正規直交な列）なので `Σ_l ‖f_l‖² = f_scale² K`、平均は `f_scale² K / d`。
Phase 9K（K = 3, f_scale = √2, d = 12）では 0.5。これを固定すると `f_scale(K) = √(0.5 d / K) = √(6/K)`。
既存の `f_scale_for_row_norm(0.5, d=12, k=K)` と同じ定義で、4 つの K すべてで `f_scale² K / d = 0.5`（誤差 ≤ 1.1e-16）。

**Y の分散**: `S_K = z_iᵀ z_j` について `E[S_K] = 0`、`Var(S_K) = Σ_k E[z_ik²] E[z_jk²] = K`。
Phase 9K（K = 3, w = 1）では `Var(w S) = 3`。これを固定すると `w(K) = √(3/K)`。既存の `w_for_matched_y_signal(1, k=K, k_ref=3)` と同じ定義で、`w(K)² K = 3`（誤差 ≤ 4.4e-16）。

**平均 edge 確率**: 分散を揃えても S_K の分布の形は K で違うので、`p̄_K(b) = E[σ(b + w(K) S_K)]` は揃わない。
target を Phase 9K の母集団の値 `p_target = p̄_3(−1)` とし、各 K で固定の `w0(K)` を `p̄_K(w0(K)) = p_target` で決める（観察された Y や replicate ごとの Z は使わない）。
K = 3 では target の定義から w0 = −1。

## 3. 一般の K での厳密な密度

Phase 9S2 と同じ導出（`φ_{S_K}(t) = (1 + t²)^{−K/2}` の逆 Fourier 変換）で

`f_K(s) = (|s|/2)^ν K_ν(|s|) / (√π Γ(K/2))`、ν = (K−1)/2。

K = 1 では `K_0(|s|)/π`（s = 0 に対数の特異点、可積分）、K = 3 では `|s| K_1(|s|)/π`。s → 0 の極限（ν > 0 で Γ(ν)/2 · 定数）は明示的に扱った。

| K | 正規化の誤差 | E[S²] − K の誤差 |
|---|---|---|
| 1 | 1.1e-16 | 2.2e-16 |
| 2 | 2.2e-16 | 0 |
| 3 | 2.2e-16 | 8.9e-16 |
| 4 | 0 | 0 |

事前の許容値（1e-11）を満たした。

## 4. 決定的な較正（実行前に固定）

- **主**: adaptive QUADPACK を [0, 1] と [1, ∞) に分けて使う（K = 1 の s = 0 の対数特異点が事前に分かっているため）。設定 A（1e-10）、B（1e-12）、C（2e-13）。根は Phase 9S2 と同じ brentq と bracket。
- **独立の cross-check**: `s = e^t` と変換し、t ∈ [−45, 4.5] で固定幅の台形則（h = 1/8, 1/16, 1/32, 1/64）。QUADPACK とは独立で、K = 1 の端点の特異点も扱える。
- **固定の規則**（Phase 9S2 と同じ 1e-12）: B と C の差がすべて ≤ 1e-12、C での根の残差 ≤ 1e-12、K = 3 の w0 の復元誤差 ≤ 1e-12、密度の確認に合格。
  cross-check は、最も細かい 2 つの差 ≤ 1e-12、かつ最も細かい値と主の値の差 ≤ 1e-12。

| 設定 | p_target | w0(K1) | w0(K2) | w0(K3) | w0(K4) | 最大の abserr | 警告 |
|---|---|---|---|---|---|---|---|
| A | 0.33140621213147148 | −0.9305782473107929 | −0.9779597638182868 | −1 | −1.012763421579947 | 7.7e-11 | 0 |
| B | 0.3314062121314677 | −0.9305782473108158 | −0.9779597638183110 | −1 | −1.012763421579973 | 8.9e-13 | 0 |
| C | 0.3314062121314677 | −0.9305782473108135 | −0.9779597638183112 | −1 | −1.012763421579973 | 1.6e-13 | 0 |

- **B と C の差**: p_target 0、w0(K1) 2.3e-15、w0(K2) 2.2e-16、w0(K3) 0、w0(K4) 0。C での残差 ≤ 1.7e-16。K = 3 は厳密に −1 を復元。**PASS**。
- **cross-check**: 最も細かい 2 つの差 ≤ 3.3e-16、主との差 ≤ 1.9e-15（w0(K1)）。**PASS**。
- p_target は Phase 9S2（0.33140621213146765）と 5e-17 以内で一致した。

### 固定した条件

| K_true | f_scale | w | w0 |
|---|---|---|---|
| 1 | √6 = 2.449489742783178 | √3 = 1.7320508075688772 | **−0.9305782473108135** |
| 2 | √3 = 1.7320508075688772 | √(3/2) = 1.224744871391589 | **−0.9779597638183112** |
| 3 | √2 = 1.4142135623730951 | 1 | **−1**（Phase 9K の anchor） |
| 4 | √(3/2) = 1.224744871391589 | √(3/4) = 0.8660254037844386 | **−1.0127634215799726** |

p_target = 0.3314062121314677。K が小さいほど S_K の分布の裾が重く（同じ分散で尖度が大きい）、同じ平均確率にするには w0 を 0 寄りにする必要がある、と数値から読める（導出はしていない）。

## 5. 推論なしの生成と安全（K_true 1..4 × rep01..rep10、40 dataset）

既存の canonical mixed generator（`run_family_selection_pilot.build_dataset`）だけを使った。**40/40 で generator の停止なし・有限・support が正しい・rank(F) = K_true**。

| 量（median; min–max） | K1 | K2 | K3 | K4 |
|---|---|---|---|---|
| 行の ‖f_l‖²（最小 / 最大の median） | 0.0028 / 1.74 | 0.028 / 1.58 | 0.074 / 1.03 | 0.16 / 0.95 |
| η^X の SD（全体） | 0.71 | 0.70 | 0.70 | 0.70 |
| Gaussian の記述的 SNR | 0.38（0.05–1.37） | 0.57（0.18–1.11） | 0.56（0.15–0.75） | 0.47（0.31–0.65） |
| Bernoulli の確率の SD | 0.164 | 0.146 | 0.153 | 0.151 |
| Poisson の率の q99 / 最大（median） | 4.2 / 7.5 | 4.9 / 7.4 | 4.4 / 10.1 | 4.4 / 6.7 |
| Poisson の安全の余裕（最小） | 4.8e4 | 2.5e4 | 6.6e4 | 6.5e4 |
| η^Y の SD | 1.74 | 1.70 | 1.66 | 1.66 |
| 真の edge 確率の平均 | 0.336 | 0.330 | 0.335 | 0.330 |
| 真の確率の q01 / q99 | 0.002 / 0.983 | 0.004 / 0.968 | 0.005 / 0.966 | 0.005 / 0.962 |
| p < 0.05 / p > 0.95 の割合 | 0.097 / 0.024 | 0.101 / 0.017 | 0.100 / 0.016 | 0.109 / 0.014 |
| 実現した edge 密度 | 0.339 | 0.328 | 0.333 | 0.332 |

平均の量（η^X の SD、平均確率、実現密度）は K によらずほぼ揃った。揃っていないもの: **行ごとの loading energy の偏り**（K = 1 では 1 本の列が 12 行に energy を配るため、行の ‖f_l‖² が 0.00002 から 3.49 まで広がり、ほとんど信号のない列がある）、
Gaussian の SNR のばらつき、Y の確率の裾（K が小さいほど両端が少し重い）。

## 6. K_true = 1 の境界の監査（静的、EM なし）

- **生成**: `build_full_rank_loadings(k=1)` は 12×1 の reduced QR で、rank 1 を構成で保証。10/10 で rank(F) = 1。
- **識別性**: O(1) = {+1, −1} で、連続な回転はなく符号の不定だけ。loading の数 `K d − K(K−1)/2 = 12` と一致（K = 1..5 で 12, 23, 33, 42, 50）。
- **Candidate B**: モデルの K = 1 では Phase 9K ですでに 20/20 の refit で評価され、すべて OK だった。
  テストでは、K_true = 1 のデータに K = 1 のモデル（初期値、未 fit）で `laplace_log_observed` を呼び、構造化された結果（Ẑ は 75×1）が返ることを確認した。
- **joint runner**: `k_category` と `k_hat_equals_k_true` は `protocol.k_true` を使っており、K によらない。
- **要約の関数**: `run_lap_vs_cq_20.K_TRUE = 3`、`category()`、`paired_summary` の both_K3 / lap_only_K3 などは 3 に固定されている。
  モデルの blocker ではないが、**将来の Phase 9X の runner は、すべての結果（exact/under/over、符号つき誤差、真の K との差）を各条件の K_true に対して計算しなければならない**。これを future protocol の要件として凍結した。
- `procrustes_rmse_z` は Phase 9C の pilot の経路だけで使われ、joint runner では使われない。
- **K = 1 だけの特別な修正は不要**（blocker なし）。

## 7. K3 の anchor との互換性

K_true = 3 の protocol（f_scale = √2、w = 1、w0 = −1、同じ seed）は Phase 9K の protocol と stage 以外の全項目で一致した。
同じ 20 の data seed で推論なしに再生成し、Phase 9K の生成と比べた:

| 配列 | bit 単位で一致 |
|---|---|
| Z / F / X / Y | 20/20 / 20/20 / 20/20 / 20/20 |
| commit 済みの `generator_provenance.csv`（列平均） | 20/20 |

**K3 historical anchor reusable: YES**。将来の Phase 9X では、K_true = 3 は Phase 9K の rep01..rep10 の結果を読み取り専用で使い、再実行しない。

## 8. 将来の Phase 9X protocol

`future_protocol.json`: **FROZEN_NOT_EXECUTED**（`execution_authorized: false`）。

- K_true: 1, 2, 4 を新しく実行、3 は Phase 9K の anchor を再利用。
- replicate: **rep01..rep10**（Phase 9K と同じ seed の系列）。20 には増やさない。
- 候補 K = 1..5、start_B のみ、探索・refit 8/8、同じ family 選択・C_Q・Candidate B、データは historical な canonical mixed generator。
- **新しい EM の上限: 300**（3 × 10 × 5 × 2）。
- runner の要件: 結果はすべて各条件の K_true に対して計算する。
- 主な結果: K_hat_Lap / K_hat_Q の数、exact/under/over、符号つき誤差、絶対誤差、最良 vs 次点の差、K_true と隣の候補との基準値の差、family 割当、Candidate B の技術的な状態、K_true × K_hat の表（一般の正答率ではない）。

## 9. DECISION

## **DECISION: MATCHED_K_TRUE_DESIGN_READY**

X の loading energy と Y の分散の規則を確認し、一般の K の w0 の較正が変更していない 1e-12 の基準で固定でき、独立の決定的な cross-check も一致し、
40 の推論なしの dataset はすべて安全で、K_true = 1 に blocker はなく、K3 の anchor の再利用（YES）と future protocol・EM の上限（300）を凍結した。

## 10. 解釈の範囲

揃えたのは、平均の X loading energy、Y の自然パラメータの分散、母集団の平均 edge 確率の 3 つだけである。
S_K の高次の積率、行ごとの loading energy の分布、Bernoulli の応答の形、Poisson の裾、確率の分布の形は K_true によって変わる。
将来の結果は「matched-signal K_true sensitivity」と呼び、純粋な潜在次元の効果とは呼ばない。

## 11. CAN SAY / CANNOT SAY

**CAN SAY**
- K_true = 1..4 について、平均の X loading energy（0.5）、Y の自然パラメータの分散（3）、母集団の平均 edge 確率（0.3314062121314677）を K_true = 3 の baseline に揃えた有限標本の設計を凍結した。
- 固定の w0 は 2 つの独立な決定的な積分経路で 1e-12 より細かく一致した。
- K_true = 3 の条件は Phase 9K と bit 単位で同じデータを作り、その結果を再利用できる。

**CANNOT SAY**
- 純粋な潜在次元の効果、分布全体の一致。
- 一致性・漸近的な主張、一般の true-K recovery、基準の優越性、実データでの妥当性。
- K 選択に関すること（本 Phase では推論をしていない）。

## 12. 最大の UNRESOLVED

平均の信号を揃えても、K_true = 1 では行ごとの loading energy が非常に偏る（ほとんど信号のない列がある）。
将来の Phase 9X で K_true = 1 の K 選択が他と違っても、それが潜在次元そのものによるのか、この偏りや Y の確率の裾の違いによるのかは、この設計では分けられない。
