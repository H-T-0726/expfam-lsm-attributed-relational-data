# Phase 9T — 平均 edge 確率を揃えた relational-w 感度（2026-09-28）

- 位置づけ: **CONDITION-SPECIFIC MEAN-DENSITY-CONTROLLED RELATIONAL-W SENSITIVITY**（Issue #113）。
  Phase 9S2 の DECISION が READY であることを条件に、事前に承認された実験。頑健性の証明でも、基準の優劣の検定でもない。
- 前提（Phase 9S2, Issue #112, PR #114）: `DENSITY_CONTROLLED_W_CALIBRATION_READY`、較正コード `248b11a`、`future_protocol.json` は `FROZEN_NOT_EXECUTED`。
  本 branch は Phase 9S2 の最終 commit `10ebcaf` から始めた。
- 条件: Phase 9P と同じ（n=75, d=12, K_true=3, X = G3/B6/P3, Y = Bernoulli, f_scale=√2, L=5, 8/8 反復, K=1..5, start_B, 同じ 20 seed, C_Q, Candidate B, N=75）で、
  w0 だけを Phase 9S2 の凍結値にした。w0 は `future_protocol.json` から読み、生成した protocol がその JSON と全項目で一致することを確認した（手入力・再較正なし）。
  baseline（w0 = −1, w = 1）は Phase 9K の記録をそのまま読んだ（**baseline の EM は 0 回追加**）。
- 結果: `expfam/results/density_controlled_w_sensitivity/phase9t_20260928/`
  （`preflight/`, `weak_w/`, `strong_w/`, `combined/`, `phase9p_comparison/`）
- 実行コード: `2ef6ef3`（`expfam/src/experimental/run_density_controlled_w.py`）。weak と strong を各 1 回、順に実行（いずれも git clean、run SUCCESS）。
- 系列 E（experimental prototype; 本文採用不可）
- **数え方**: 同じ 20 replicate の系列（同じ seed・Z・F・X）を 3 条件で見たもので、60 個の独立な dataset ではない。

---

## 1. 研究上の問い

Phase 9P は w0 = −1 を固定して w を変えたため、平均 edge 確率も変わっていた。
母集団の平均 Bernoulli edge 確率を baseline（p_target = 0.33140621213146765）に揃えたうえで w を変えたとき、C_Lap と C_Q の K 選択の振る舞いはどうなるか。

## 2. 事前の確認（EM の前）

- Phase 9S2 の READY、freeze 規則と cross-check の PASS、`future_protocol.json` の `FROZEN_NOT_EXECUTED`、weak/strong の凍結 w0 の存在を確認した（SHA-256 は `preflight/preflight.json`）。
- **パイプラインの比較可能性: PASS**。Phase 9P の実行コード `2a99133` から本 branch までの差分は、Phase 9Q/9R/9S/9S2 のファイル・報告書・結果の**追加だけ**で、既存ファイルの変更は 0 件。
- **Z/F/X の同一性: 20/20**（controlled weak / baseline / controlled strong の間でビット単位で一致）。
- 凍結値: weak (w0, w) = (−0.8780994405393426, 1/√2)、baseline (−1, 1)、strong (−1.1890648128923011, √2)。

## 3. 密度と飽和の文脈（真の生成値から、記述のみ）

| 条件 | 有限 replicate の真の平均確率（min/median/max） | 実現 edge 密度 | η の平均（median） | η の SD（median） | 確率の SD（median） | q01 / q50 / q99（median） | p < 0.05 の割合 | p > 0.95 の割合 |
|---|---|---|---|---|---|---|---|---|
| weak | 0.320 / 0.329 / 0.339 | 0.308 / 0.329 / 0.344 | −0.882 | 1.16 | 0.202 | 0.020 / 0.294 / 0.903 | 0.042 | 0.003 |
| baseline | 0.314 / 0.328 / 0.342 | 0.313 / 0.333 / 0.341 | −1.006 | 1.63 | 0.249 | 0.005 / 0.269 / 0.968 | 0.100 | 0.016 |
| strong | 0.307 / 0.326 / 0.346 | 0.306 / 0.328 / 0.354 | −1.197 | 2.31 | 0.295 | 0.001 / 0.234 / 0.994 | 0.188 | 0.041 |

- 母集団の平均確率は設計上 3 条件で同じ（0.3314）。有限の Z では真の平均確率は 0.307〜0.346 でばらつき、実現密度も一致しない。
- **揃っていないもの**: 確率の SD、分位、飽和（p > 0.95 の割合は 0.003 → 0.016 → 0.041、p < 0.05 は 0.042 → 0.100 → 0.188）。w が大きいほど確率は両端に広がる。

## 4. 結果（条件ごと）

### 技術的な完全性

| 条件 | EM 試行 / 成功 | 不完全な dataset | Candidate B OK / NOT_STATIONARY / HESSIAN_NOT_PD / error | C_Lap 完全 |
|---|---|---|---|---|
| controlled weak | 200 / 200 | 0 | 100 / 0 / 0 / 0 | 20 |
| baseline（Phase 9K） | 200 / 200 | 0 | 100 / 0 / 0 / 0 | 20 |
| controlled strong | 200 / 200 | 0 | 100 / 0 / 0 / 0 | 20 |

新しい EM: 計画 400、試行 400、成功 400。retry・置き換え・seed の救済は 0。

### 選ばれた K

| 条件 | K_hat_Lap（K1..K5） | exact / under / over | K_hat_Q（K1..K5） | exact / under / over |
|---|---|---|---|---|
| controlled weak | 0 / 3 / 17 / 0 / 0 | 17 / 3 / 0 | 8 / 11 / 1 / 0 / 0 | 1 / 19 / 0 |
| baseline | 0 / 1 / 19 / 0 / 0 | 19 / 1 / 0 | 1 / 5 / 14 / 0 / 0 | 14 / 6 / 0 |
| controlled strong | 0 / 0 / 20 / 0 / 0 | 20 / 0 / 0 | 0 / 1 / 19 / 0 / 0 | 19 / 1 / 0 |

過大選択（K ≥ 4）は 0。

### 同じ条件内の C_Q と C_Lap

| 条件 | 両方 K=3 | C_Lap だけ | C_Q だけ | どちらも K≠3 | 同じ K | 異なる K |
|---|---|---|---|---|---|---|
| controlled weak | 1 | 16 | 0 | 3 | 2 | 18 |
| baseline | 14 | 5 | 0 | 1 | 15 | 5 |
| controlled strong | 19 | 1 | 0 | 0 | 19 | 1 |

### 差（min / median / max; 負の数 / 正の数）

| 条件 | delta_Lap_23 | delta_Q_23 | C_Lap の最良 vs 次点の差 | C_Q の最良 vs 次点の差 |
|---|---|---|---|---|
| controlled weak | −66.0 / −23.7 / 45.1（17 / 3） | −17.5 / 70.7 / 173.2（2 / 18） | **0.20** / 27.6 / 59.3 | 1.35 / 30.1 / 105.8 |
| baseline | −150.3 / −75.9 / 52.3（19 / 1） | −130.2 / −3.7 / 157.1（14 / 6） | 22.0 / 49.2 / 58.4 | 1.96 / 34.8 / 130.2 |
| controlled strong | −290.2 / −163.1 / −23.0（20 / 0） | −346.5 / −134.4 / 28.8（19 / 1） | 23.0 / 55.8 / 69.4 | 23.4 / 134.4 / 199.8 |

C_Lap の最小の差: controlled weak は rep04 **0.20**（最良 K=3 / 次点 K=2）、rep11 1.93、rep07 3.49、rep10 7.58、rep14 9.83。controlled strong は rep20 23.0。
事前の規則に従い、これらの小さな差に局所最適化の診断は行っていない。

## 5. w をまたいだ対応（controlled weak → baseline → controlled strong）

- **C_Lap**: 3 条件で同じ K 17/20（すべて 3→3→3）。変わったのは rep07・rep10（2→3→3）、rep20（2→2→3）。
  weak→baseline: same 18 / under→exact 2。baseline→strong: same 19 / under→exact 1。
- **C_Q**: 3 条件で同じ K 1/20。主な経路は 2→3→3（8）、1→3→3（5）、2→2→3（3）。
  weak→baseline: same 5 / under→exact 13 / under→under 2。baseline→strong: same 15 / under→exact 5。
- どちらも exact→under、exact→over、over→exact は 0。

差の baseline からの変化（median）: delta_Lap_23 は weak へ +50.8（符号の変化 2）、strong へ −84.8（1）。delta_Q_23 は weak へ +78.2（12）、strong へ −122.5（5）。
C_Lap の差は weak へ −21.2、strong へ +9.0。C_Q の差は weak へ +11.9、strong へ +107.8。

## 6. K=2 → 3 の分解（median; min–max は `combined/combined_summary.json`）

| 量 | controlled weak | baseline | controlled strong |
|---|---|---|---|
| C_Lap: joint mode での fit gain | 262.3 | 347.8 | 472.4 |
| C_Lap: Laplace 体積の増分 | 199.0 | 231.5 | 263.7 |
| C_Q: MC 平均での fit gain | 185.3 | 259.7 | 390.4 |
| C_Q: P_Z の増分 | 212.84 | 212.84 | 212.84 |
| 共通: P_θ の増分 | 43.17 | 43.17 | 43.17 |
| fit gain の差（C_Lap − C_Q） | 71.7 | 77.3 | 78.9 |

Phase 9P と同じく、fit gain と C_Lap の体積の増分は w とともに増え、C_Q の P_Z の増分は一定、fit gain の差は条件によらずほぼ同じ（median 72–79）だった。

## 7. 技術的な挙動（Candidate B、条件 × K）

新しい 200 refit はすべて OK。joint mode の反復は median 4–9（最大 17）、最後の grad_inf は最大 9.7e-9（≤ 1e-8）、H の最小固有値はすべて正で、
最小値は K=5 で 0.039（controlled weak）/ 0.057（baseline）/ 0.145（controlled strong）。family selection の convergence warning（診断のみ）は K とともに増えた（K=5 の median 16–17）。

## 8. family の挙動（二次的な観察）

3 条件 × 5 K × 20 dataset のすべての refit で、選ばれた family 割当が真の割当（G3/B6/P3）と一致した。

## 9. Phase 9P（w0 = −1、平均密度は制御なし）との対応比較

同じ w・同じ Z/F/X で、w0 だけが −1 から凍結値に変わった比較。

| 量（Phase 9P → Phase 9T） | weak | strong |
|---|---|---|
| 有限 replicate の真の平均確率（median） | 0.308 → 0.329（+0.022） | 0.352 → 0.326（−0.026） |
| 実現 edge 密度（median） | 0.306 → 0.329（+0.022） | 0.352 → 0.328（−0.025） |
| K_hat_Lap の変化 | same 19、under→exact 1（rep04: 2→3） | same 20 |
| K_hat_Q の変化 | same 17、under→exact 1（rep18: 2→3）、exact→under 1（rep15: 3→2）、under→under 1（rep06: 1→2） | same 20 |
| delta_Lap_23 の変化（median, 範囲） | +1.4（−19.0〜+22.1）、符号の変化 1（rep04: 7.02 → −0.20） | +7.8（−35.8〜+36.1）、符号の変化 0 |
| delta_Q_23 の変化 | +2.0（−34.7〜+52.4）、符号の変化 2 | +6.9（−88.9〜+36.1）、符号の変化 0 |
| C_Lap の差の変化 | −0.8（−22.1〜+19.0） | −0.9（−11.3〜+14.3） |
| C_Q の差の変化 | +3.7（−48.3〜+25.9） | +7.1（−34.4〜+39.9） |
| C_Lap の fit gain / 体積の増分の変化（median） | −3.5 / −2.6 | −8.7 / +0.7 |
| C_Q の fit gain の変化（median） | −2.0 | −6.9 |

- K=3 の数は、C_Lap で Phase 9P の 16 → 19 → 20 に対し Phase 9T は 17 → 19 → 20、C_Q では両方とも 1 → 14 → 19 だった。
- つまり平均 edge 確率を揃えても、w による K 選択のパターン（weak で K ≤ 2 が増え、特に C_Q で顕著）はほとんど変わらなかった。
  replicate ごとの差の変化の中央値は小さく（数単位）、範囲は ±数十だった。
- これを「密度の因果効果」とは呼ばない。w0 を変えると実現密度だけでなく η と確率の分布の位置全体が変わるためである。

## 10. DECISION

## **DECISION: DENSITY_CONTROLLED_W_SENSITIVITY_CHARACTERIZED**

両方の新しい条件が凍結した Phase 9S2 の protocol で 1 回ずつ完了し（EM 400/400 成功、不完全な dataset 0、Candidate B 200/200 OK）、
Z/F/X の同一性と baseline の再利用を確認したうえで、事前に決めた集計をすべて報告できた。この判定は K=3 の頻度によらない。

## 11. 解釈（INTERPRETATION）

- この条件では、母集団の平均 edge 確率を揃えても、w を弱めると K ≤ 2 が増え（C_Q で顕著、C_Lap では少し）、強めると両基準とも K=3 に集まった。
  Phase 9P で見えた w-sensitivity のパターンは、平均密度の違いだけでは説明されなかった。
- 残る違いは、η の広がりと確率の分布の形（飽和）であり、本設計はこれらを揃えていない。
  したがって、残ったパターンを「純粋な信号の強さの効果」とは言えず、「平均密度を揃えたときの w-sensitivity」である。
- K=2→3 の分解は Phase 9P と同じ傾向で、C_Q の P_Z の増分が一定である一方、C_Lap の体積の増分は w とともに変わった。

## 12. CAN SAY / CANNOT SAY

**CAN SAY**
- この固定条件で、母集団の平均 Bernoulli edge 確率を baseline に揃えた weak / baseline / strong では、
  C_Lap の K=3 は 17 → 19 → 20、C_Q の K=3 は 1 → 14 → 19 で、過大選択は 0 だった。
- 同じ w で w0 = −1（Phase 9P）と比べると、K の選択は weak で C_Lap 1 件・C_Q 3 件、strong で 0 件だけが変わり、条件ごとの K=3 の数はほぼ同じだった。
- controlled weak の rep04 では C_Lap の差が 0.20 と非常に小さかった（最良 K=3）。

**CANNOT SAY**
- 純粋な信号の効果、密度・飽和の完全な制御、密度の因果効果。
- 一般的な頑健性、一致性、優越性、一般的な recovery 確率。
- 他の n/d/K_true・X の信号・family 構成・実データへの一般化。
- 60 個の独立な dataset としての数え方、Phase 9K/9P の結果の書き換え。

## 13. 最大の UNRESOLVED と次の Human 判断

**最大の UNRESOLVED**: 平均密度を揃えても残る w-sensitivity が、η の広がりそのものによるのか、確率の飽和（分布の形）によるのかは、この設計では分けられない。
また controlled weak の rep04（差 0.20）は、Phase 9M〜9R で見た局所最適化の誤差（1 ステップで最大 2.8）より小さく、その順序は θ̂ の最適化誤差の範囲にある。
事前の規則に従い、これには追加の診断を行っていない。

**次の Human 判断**: (a) 飽和の形も揃える別の設計を考えるか、(b) weak の小さな差の扱いを決めるか、(c) relational-w の感度の特徴づけをここで閉じるか。いずれも新しい Human Gate が必要。
