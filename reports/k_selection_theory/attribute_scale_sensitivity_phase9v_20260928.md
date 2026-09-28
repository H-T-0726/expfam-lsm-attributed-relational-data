# Phase 9V — Y を固定した属性 loading scale の感度（2026-09-28）

- 位置づけ: **CONDITION-SPECIFIC ATTRIBUTE-LOADING-SCALE SENSITIVITY**（Issue #118）。Phase 9U で凍結した対応設計の本実験。
  純粋な属性の情報の効果ではなく、頑健性の証明や基準の優劣の検定でもない。
- 前提: Phase 9U（`d53771e`）の `ATTRIBUTE_SCALE_DESIGN_READY`、`future_protocol.json` = `FROZEN_NOT_EXECUTED`、`future_base_rerun_required = true`、`future_new_em_cap = 600` を確認した
  （SHA-256 は `preflight/preflight.json`）。
- データ: Phase 9U の `paired_attribute_scale.py` だけを使った（Z と実現した Y を 3 条件で共有）。historical な Phase 9K/9T の baseline は使っていない。
- 条件: weak f_scale = 1、base √2、strong 2（平均 loading energy 0.25 / 0.50 / 1.00）。n = 75, d = 12, K_true = 3, X = G3/B6/P3, Y = Bernoulli, w0 = −1, w = 1。
  推論は Phase 9K/9T と同じ（8/8 反復、K = 1..5、start_B、同じ family 選択・C_Q・Candidate B、N = 75、同じ 20 の seed）。
- 結果: `expfam/results/attribute_scale_sensitivity/phase9v_20260928/`（`preflight/`, `weak/`, `base/`, `strong/`, `combined/`）
- 実行コード: `063e240`（`expfam/src/experimental/run_attribute_scale_sensitivity.py`）。3 条件を各 1 回、順に実行（いずれも git clean）。
- 系列 E（experimental prototype; 本文採用不可）
- **数え方**: 同じ 20 の対応した replicate（Z・Q・Y・search/refit seed を共有）を 3 つの loading scale で見たもので、60 個の独立な dataset ではない。

---

## 1. 研究上の問い

Z と実現した Y を共有したまま、共通の属性 loading energy を half / base / double に変えたとき、C_Lap と C_Q の K 選択はどう変わるか。

## 2. data adapter と事前の確認（EM の前）

- **adapter**: `joint.execute` の間だけ `run_family_selection_pilot.build_dataset` を差し替え、protocol の f_scale（厳密一致）から
  `paired_attribute_scale.generate(data_seed, condition)` を呼んで配列をそのまま渡す。終了後に元に戻す（テストで確認）。再抽出・正規化・列の並べ替え・family の変更はしない。
  historical な generator と pipeline のファイルは変更していない。
- **preflight**（推論なし、20 replicate × 3 条件）:

| 確認 | 合格 |
|---|---|
| Z の同一性 / Q（F の向き）の同一性 / `F_c = Q·f_scale_c` の厳密な関係 / rank(F) = 3 | 20/20 |
| **Y の同一性（SHA-256 が 3 条件で一致）** | **20/20** |
| 有限・support・Poisson の逆関数の確認・Poisson の安全（strong の最大率 349.07） | 20/20 |
| 同じ seed での再生成の一致 | 20/20 |
| **adapter が渡す Z/F/X/Y が generator の出力とビット単位で一致** | 全件 |
| 各条件の protocol が凍結した JSON と全項目で一致 | 3/3 |

replicate ごとの Y の edge 密度（3 条件で共通）: 0.313〜0.352。

## 3. 実行と技術的な完全性

| 条件 | EM 試行 / 成功 | 不完全な dataset | Candidate B OK / その他 | C_Lap 完全 |
|---|---|---|---|---|
| weak | 200 / 200 | 0 | 100 / 0 | 20 |
| base | 200 / 200 | 0 | 100 / 0 | 20 |
| strong | 193 / 192 | **1（rep04）** | 96 / 0 | 19 |

- 新しい EM: 計画 600、試行 593、成功 592。retry・置き換え・seed の救済は 0。
- **strong rep04**: K = 1 は完了、K = 2 の探索 EM で `FloatingPointError: Poisson exp(eta) would overflow ... max eta=1120.2`（推定中のモデルの Poisson の線形予測子が float64 の上限を超えた）。
  既存の research-first の規則どおり dataset を不完全として記録し、K = 3〜5 は実行していない（200 − 7 = 193）。
  データ自体は preflight の安全条件を満たしていた（rep04 strong の真の率の最大は 29.7）。部分的な K の格子から K_hat は作っていない。

## 4. 結果（条件ごと）

### 選ばれた K

| 条件 | K_hat_Lap（K1..K5） | exact / under / over | K_hat_Q（K1..K5） | exact / under / over |
|---|---|---|---|---|
| weak | 0 / 0 / 20 / 0 / 0 | 20 / 0 / 0 | 0 / 9 / 11 / 0 / 0 | 11 / 9 / 0 |
| base | 0 / 0 / 20 / 0 / 0 | 20 / 0 / 0 | 0 / 5 / 15 / 0 / 0 | 15 / 5 / 0 |
| strong（19） | 0 / 0 / 18 / **1** / 0 | 18 / 0 / **1** | 0 / 1 / 18 / 0 / 0 | 18 / 1 / 0 |

strong rep06 で C_Lap が K = 4 を選んだ（次点 K = 3 との差 13.65）。本シリーズの C_Lap で初めての過大選択である。事前の規則に従い追加の診断はしていない。

### 同じ条件内の C_Q と C_Lap

| 条件 | 両方 K=3 | C_Lap だけ | C_Q だけ | どちらも K≠3 | 同じ K | 異なる K |
|---|---|---|---|---|---|---|
| weak | 11 | 9 | 0 | 0 | 11 | 9 |
| base | 15 | 5 | 0 | 0 | 15 | 5 |
| strong（19） | 17 | 1 | 1 | 0 | 17 | 2 |

### 差（min / median / max; 負 / 正）

| 条件 | delta_Lap_23 | delta_Q_23 | C_Lap の最良 vs 次点 | C_Q の最良 vs 次点 |
|---|---|---|---|---|
| weak | −125.7 / −73.3 / −18.8（20 / 0） | −85.9 / −5.1 / 58.5（11 / 9） | 18.8 / 45.6 / 59.9 | 0.45 / 39.2 / 85.9 |
| base | −203.7 / −105.4 / −42.4（20 / 0） | −154.7 / −59.5 / 43.3（15 / 5） | 39.8 / 51.6 / 68.2 | 0.67 / 59.5 / 154.7 |
| strong（19） | −270.9 / −151.2 / −65.9（19 / 0） | −269.1 / −129.4 / 12.6（18 / 1） | 13.7 / 51.1 / 76.4 | 12.6 / 129.4 / 208.3 |

C_Lap の最小の 5 つの差（記述のみ）: weak rep07 18.8、rep18 30.3、rep06 31.5、rep19 34.4、rep13 36.8（すべて 3 vs 2）；
base rep18 39.8、rep06 41.7、rep07 42.4、rep19 42.6、rep17 44.7；strong rep06 13.7（4 vs 3）、rep19 35.9、rep18 40.8、rep02 45.7、rep13 48.2。

## 5. scale をまたいだ対応（weak → base → strong）

- **C_Lap**: 3 条件で同じ K 18/20（3→3→3）。rep06 は 3→3→4（exact→over）、rep04 は 3→3→NA（strong が不完全）。weak→base はすべて same。
- **C_Q**: 3 条件で同じ K 11/20（3→3→3 が 10、2→2→2 が 1）。2→3→3 が 4、2→2→3 が 4、rep04 は 3→3→NA。
  weak→base: same 16 / under→exact 4。base→strong: same 15 / under→exact 4 / unavailable 1。exact→under は 0。
- **差の対応した変化（median）**: delta_Lap_23 は weak→base −27.0（20/20 で負）、base→strong −51.0（19/19 で負）。delta_Q_23 は −39.3（20/20）、−67.3（19/19）。
  符号の変化: delta_Lap_23 は 0 / 0、delta_Q_23 は 4 / 4。
  C_Lap の差: weak→base +4.9、base→strong −1.3。C_Q の差: +37.6、+57.5。

## 6. K=2 → 3 の分解（median）

| 量 | weak | base | strong（19） |
|---|---|---|---|
| C_Lap: joint mode での fit gain | 345.3 | 391.9 | 458.8 |
| C_Lap: Laplace 体積の増分 | 231.9 | 239.4 | 258.2 |
| C_Q: MC 平均での fit gain | 261.1 | 315.5 | 385.4 |
| C_Q: P_Z の増分 | 212.84 | 212.84 | 212.84 |
| 共通: P_θ の増分 | 43.17 | 43.17 | 43.17 |
| fit gain の差（C_Lap − C_Q） | 77.5 | 77.4 | 76.2 |

対応した変化（median）: C_Lap の fit gain は weak→base +36.9、base→strong +65.8、体積の増分は +8.7、+15.9、C_Q の fit gain は +39.3、+67.3。
fit gain の差（C_Lap − C_Q）の変化は +0.1、−1.2 でほぼ 0。
記述: loading energy を上げると両基準の fit gain が同程度に増え、C_Q では固定の P_Z（212.84）+ P_θ（43.17）を超える dataset が増えた。
C_Lap の体積の増分は Phase 9P/9T の w の変化ほどは変わらなかった（232 → 258）。

## 7. family の挙動

3 条件 × 5 K のすべての refit（weak 100、base 100、strong 96）で、選ばれた family 割当が真の割当（G3/B6/P3）と一致した。条件間での割当の変化は 0（weak→base 100 件、base→strong 96 件で比較）。

## 8. X の信号の文脈（median; Phase 9U と同じ量）

| 量 | weak | base | strong |
|---|---|---|---|
| 平均 ‖f_l‖² | 0.25 | 0.50 | 1.00 |
| η^X の SD（全体 / G / B / P） | 0.50 / 0.52 / 0.48 / 0.50 | 0.70 / 0.73 / 0.68 / 0.70 | 1.00 / 1.04 / 0.96 / 1.00 |
| Gaussian の記述的 SNR | 0.26 | 0.52 | 1.04 |
| Bernoulli の確率の SD | 0.112 | 0.150 | 0.195 |
| Bernoulli の p > 0.95 の割合 | 0 | 0 | 0.003 |
| Poisson の率の平均 / q99 / 最大 | 1.13 / 3.5 / 4.6 | 1.28 / 5.8 / 8.8 | 1.65 / 12.1 / 21.5 |
| Poisson の安全の余裕（最小） | 5.4e4 | 1.6e4 | 2.9e3 |

replicate ごとの文脈の変化（事前に固定した 5 量: η^X の SD、Gaussian の SNR、Bernoulli の SD、Bernoulli の飽和、Poisson の q99）と、選ばれた K の経路・delta_23 の変化を
`combined/context_alignment.csv` に並べた。回帰・閾値探索・事後の変数探索は行っていない。
記述として、C_Lap が K = 4 を選んだ strong rep06 は、strong の中で Poisson の真の率の最大が 2 番目に大きい（286.0）replicate だったが、これを原因とは主張しない。

## 9. Candidate B の技術的な挙動

296 refit（weak 100、base 100、strong 96）はすべて OK。joint mode の反復は median 4–9（最大 39）、最後の grad_inf は最大 9.8e-9（≤ 1e-8）、
H の最小固有値はすべて正で、最小値は K = 5 で 0.112（weak）/ 0.128（base）/ 0.123（strong）。convergence warning（診断のみ）は K とともに増えた（K = 5 の median 14–19）。

## 10. DECISION

## **DECISION: ATTRIBUTE_SCALE_SENSITIVITY_PARTIAL**

3 条件とも凍結した protocol で 1 回ずつ実行し、対応の不変量（Z・Q・Y、F のスケール関係、adapter）はすべて成り立った。
strong の rep04 が推定中の Poisson の数値的な overflow で本当に不完全になり（救済・置き換えなし）、strong の集計は 19 dataset、対応の比較は該当する組だけで行った。
それ以外の事前に決めた集計はすべて報告できた。この判定は K = 3 の頻度によらない。

## 11. 解釈（INTERPRETATION）

- Z と実現した Y を共有したまま loading energy を half → base → double にすると、C_Q の K = 3 は 11 → 15 → 18（/19）と増え、
  C_Lap はどの条件でもほぼ K = 3 だった（20 → 20 → 18/19、strong で過大選択 1）。
- delta_23 は両基準とも loading energy とともに一貫して K = 3 寄りに動き（C_Lap は符号の変化 0、C_Q は 4 と 4）、
  両基準の fit gain の差（約 77）は条件によらずほぼ一定だった。C_Q の K 選択の変化は、固定の P_Z に対して fit gain が増えることと対応している。
- Y 側（Phase 9P/9T）で見た C_Q の大きな感度（K = 3 が 1 → 14 → 19）に比べ、X 側の loading scale の感度は C_Q でも小さかった（11 → 15 → 18）。
  ただし 2 つの操作は大きさの尺度が違い、直接の比較はできない。
- loading energy を上げると、推定中の Poisson の数値的な失敗（rep04）や C_Lap の過大選択（rep06）が現れた。いずれも 1 件であり、原因は分からない。

## 12. CAN SAY / CANNOT SAY

**CAN SAY**
- この固定の対応した合成データの設定で、Z と実現した Y を共有し、属性の loading energy を half / base / double にしたとき、
  C_Lap の K = 3 は 20 / 20 / 18（19 中）、C_Q の K = 3 は 11 / 15 / 18（19 中）だった。
- delta_23 は両基準とも loading energy とともに K = 3 寄りに動き、fit gain の差（C_Lap − C_Q）はほぼ一定だった。
- strong で 1 dataset が推定中の Poisson の overflow で不完全になり、1 dataset で C_Lap が K = 4 を選んだ。
- family 割当はすべての refit で真の割当と一致した。

**CANNOT SAY**
- 純粋な属性の情報の効果、family によらない信号の効果（f_scale は Gaussian の SNR、Bernoulli の飽和、Poisson の率の裾を同時に変える）。
- どの family の変化が K 選択の変化を起こしたか。
- 一般的な頑健性・一致性・優越性・recovery 確率、他の n/d/K_true・family 構成・実データへの一般化。
- 60 個の独立な dataset としての数え方。

## 13. 最大の UNRESOLVED と次の Human 判断

**最大の UNRESOLVED**: strong で現れた 2 つの事象（rep04 の推定中の Poisson の overflow と rep06 の C_Lap の過大選択）が、
Poisson の率の裾が大きくなることに関係するのかどうか。この設計では family ごとの寄与を分けられず、事前の規則に従い追加の ablation や診断は行っていない。

**次の Human 判断**: (a) family ごとに scale を分ける設計（例えば Poisson の列だけを固定する）を考えるか、(b) rep06/rep04 の事象の記述的な診断をするか、
(c) ここで属性 scale の感度の特徴づけを閉じるか。いずれも新しい Human Gate が必要。
