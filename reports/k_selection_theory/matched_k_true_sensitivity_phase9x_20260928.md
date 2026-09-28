# Phase 9X — 信号を揃えた K_true の感度（2026-09-29）

- 位置づけ: **CONDITION-SPECIFIC MATCHED-SIGNAL K_true SENSITIVITY**（Issue #122）。Phase 9W で凍結した設計の本実験。
  純粋な潜在次元の効果でも、一致性・漸近の検証でも、一般の true-K recovery の推定でもない。
- 前提: Phase 9W の merge `ba399d5779375540e3b50aa397d81598fbec736a`。`MATCHED_K_TRUE_DESIGN_READY`、`future_protocol.json` は `FROZEN_NOT_EXECUTED` / `execution_authorized: false`、
  `K3_anchor_reusable: true`、`future_new_em_cap: 300`、候補 K 1..5、rep01..rep10、start_B のみ、を確認した。
- **Phase 9X の実行の権限**: Phase 9W の artifact は書き換えず（`execution_authorized: false` のまま）、別の記録 `preflight/execution_authorization.json` を作った
  （issue 122、authorized_by Human、scope Phase 9X only、Phase 9W の merge SHA、Phase 9W の `design.json` / `future_protocol.json` / `calibration_by_k.json` / `k3_anchor_compatibility.json` の SHA-256、
  新しい K_true {1, 2, 4}、K3 rerun false、EM の上限 300、許可しない事項の一覧）。
- 結果: `expfam/results/matched_k_true_sensitivity/phase9x_20260928/`（`preflight/`, `K1/`, `K2/`, `K4/`, `K3_anchor/`, `combined/`）
- 実行コード: `7ac620d`（`expfam/src/experimental/run_matched_k_true.py`）。K_true 1, 2, 4 を各 1 回、順に実行（いずれも git clean、run SUCCESS）。
- 系列 E（experimental prototype; 本文採用不可）

---

## 1. 研究上の問い

平均の X loading energy（0.5）、Y の自然パラメータの分散（3）、母集団の平均 edge 確率（Phase 9K の K3 baseline）を揃えたとき、
現在の C_Lap と C_Q は K_true = 1, 2, 3, 4 でどの K を選ぶか。

## 2. 条件（Phase 9W の `future_protocol.json` から読み込み、手入力なし）

| K_true | f_scale | w | w0 | 役割 |
|---|---|---|---|---|
| 1 | √6 | √3 | −0.9305782473108135 | 新しい実行 |
| 2 | √3 | √(3/2) | −0.9779597638183112 | 新しい実行 |
| 3 | √2 | 1 | −1 | Phase 9K の rep01..rep10（読み取りのみ、新しい EM 0） |
| 4 | √(3/2) | √(3/4) | −1.0127634215799726 | 新しい実行 |

共通: n = 75, d = 12, X = G3/B6/P3, Y = Bernoulli, σ²_x = 1, L = 5, 8/8 反復, 候補 K = 1..5, start_B のみ, 同じ family 選択・C_Q・Candidate B, N = 75, IS なし, 局所最適化なし。
replicate は rep01..rep10（data seed 1001001..1001010 など）だけ。各条件の protocol は凍結した JSON と全項目で一致することを確認した（provenance のための 3 つのキーを除く）。

## 3. 事前の確認（EM の前、推論なし）

- 新しい条件の 30 dataset: 有限、support、rank(F) = K_true、Poisson の安全（最大率 39.7）、凍結値の厳密な一致がすべて 30/30。
- **K3 anchor**: Phase 9W の K3 の protocol で再生成した Z/F/X/Y は Phase 9K と 10/10 でビット単位で一致。Phase 9K の必要な artifact（laplace_by_k、family_by_k、cq_decomposition、ledger、summary、runinfo、fitted_states）がそろい、
  rep01..rep10 × K1..5 の 50 行がすべてある。SHA-256 を記録した。
- **K_true によらない結果の計算**: Phase 9K の `K_TRUE = 3` を使う関数（`category`、`paired_summary` の both_K3 など）は使わず、各条件の K_true に対して計算する関数を新しく書いた。
  preflight の自己確認と focused tests（K_true = 1 で K = 3 が exact にならないことなど）で確認した。
  また、この関数で Phase 9K の rep01..rep10 から計算した K_hat_Lap / K_hat_Q は、commit 済みの Phase 9K の値と一致した。

## 4. 実行

| K_true | EM 試行 / 成功（範囲内） | 新しい EM | 完全な dataset | Candidate B OK |
|---|---|---|---|---|
| 1 | 100 / 100 | 100 | 10 | 50/50 |
| 2 | 100 / 100 | 100 | 10 | 50/50 |
| 3（anchor） | 100 / 100（Phase 9K の記録） | **0** | 10 | 50/50 |
| 4 | 100 / 100 | 100 | 10 | 50/50 |

新しい EM: 計画 300、試行 300、成功 300。retry・置き換え・seed の救済は 0。

## 5. 選ばれた K（K_true × K_hat、10 dataset ずつ）

**C_Lap**

| K_true | K1 | K2 | K3 | K4 | K5 | NA | exact / under / over |
|---|---|---|---|---|---|---|---|
| 1 | 10 | 0 | 0 | 0 | 0 | 0 | 10 / 0 / 0 |
| 2 | 0 | 10 | 0 | 0 | 0 | 0 | 10 / 0 / 0 |
| 3 | 0 | 0 | 10 | 0 | 0 | 0 | 10 / 0 / 0 |
| 4 | 0 | 0 | 2 | 8 | 0 | 0 | 8 / 2 / 0 |

**C_Q**

| K_true | K1 | K2 | K3 | K4 | K5 | NA | exact / under / over |
|---|---|---|---|---|---|---|---|
| 1 | 10 | 0 | 0 | 0 | 0 | 0 | 10 / 0 / 0 |
| 2 | 1 | 9 | 0 | 0 | 0 | 0 | 9 / 1 / 0 |
| 3 | 1 | 2 | 7 | 0 | 0 | 0 | 7 / 3 / 0 |
| 4 | 1 | 5 | 3 | 1 | 0 | 0 | 1 / 9 / 0 |

- 過大選択は両基準ともどの K_true でも 0。これを 1 つの「正答率」や一般の recovery 確率としてはまとめない。
- K_true = 3 の値は Phase 9K の rep01..rep10 だけであり、Phase 9K の 20 replicate の集計（C_Q の K3 は 14/20 など）とは混ぜていない。

### 誤差（K_hat − K_true）

| K_true | C_Lap の符号つき誤差（min / median / max） | C_Lap の絶対誤差の最大 | C_Q の符号つき誤差 | C_Q の絶対誤差（median / max） |
|---|---|---|---|---|
| 1 | 0 / 0 / 0 | 0 | 0 / 0 / 0 | 0 / 0 |
| 2 | 0 / 0 / 0 | 0 | −1 / 0 / 0 | 0 / 1 |
| 3 | 0 / 0 / 0 | 0 | −2 / 0 / 0 | 0 / 2 |
| 4 | −1 / 0 / 0 | 1 | −3 / −2 / 0 | 2 / 3 |

### 同じ条件内の C_Lap と C_Q

| K_true | both_exact | Lap_exact_only | Q_exact_only | neither_exact | same_selected_K | different_selected_K |
|---|---|---|---|---|---|---|
| 1 | 10 | 0 | 0 | 0 | 10 | 0 |
| 2 | 9 | 1 | 0 | 0 | 9 | 1 |
| 3 | 7 | 3 | 0 | 0 | 7 | 3 |
| 4 | 1 | 7 | 0 | 2 | 1 | 9 |

## 6. 真の K を基準にした隣の候補との差

`ΔC(true, k) = C(K_true) − C(k)`。**負なら真の K の基準値の方が小さい（良い）**。

| K_true | 比較 | C_Lap（min / median / max; 負 / 正） | C_Q（min / median / max; 負 / 正） |
|---|---|---|---|
| 1 | K1 vs K2 | −87.9 / −72.6 / −54.3（10 / 0） | −190.9 / −181.9 / −142.1（10 / 0） |
| 2 | K2 vs K1 | −437.2 / −230.3 / −52.1（10 / 0） | −457.7 / −210.8 / 5.5（9 / 1） |
| 2 | K2 vs K3 | −73.2 / −56.4 / −41.9（10 / 0） | −222.4 / −178.0 / −122.2（10 / 0） |
| 3 | K3 vs K2 | −119.1 / −74.4 / −22.0（10 / 0） | −99.2 / −3.7 / 71.2（7 / 3） |
| 3 | K3 vs K4 | −58.4 / −52.1 / −27.1（10 / 0） | −192.7 / −163.6 / −136.8（10 / 0） |
| 4 | K4 vs K3 | −86.6 / −13.4 / 9.6（8 / 2） | −66.3 / 87.1 / 100.0（1 / 9） |
| 4 | K4 vs K5 | −64.4 / −42.6 / −31.4（10 / 0） | −204.7 / −152.8 / −132.7（10 / 0） |

- 過大側（真の K と K+1）では、両基準とも全 40 dataset で真の K の方が良かった。
- 過小側（真の K と K−1）で差が小さくなり、特に K_true = 4 の K4 vs K3 で C_Q は 9/10 で K3 の方が良く、C_Lap も 2/10 で K3 の方が良かった。

### 最良 vs 次点の差（C_Lap の最小の 5 つ、記述のみ）

| K_true | C_Lap の差（min / median / max） | 最小の 5 つ（replicate, 差, 最良 / 次点） |
|---|---|---|
| 1 | 54.3 / 72.6 / 87.9 | rep07 54.3 (1/2), rep02 67.0, rep05 67.7, rep04 70.4, rep10 72.1 |
| 2 | 41.9 / 56.4 / 73.2 | rep04 41.9 (2/3), rep07 42.7, rep03 53.9, rep02 54.5, rep01 55.6 |
| 3 | 22.0 / 47.4 / 58.4 | rep10 22.0 (3/2), rep04 29.2, rep09 43.0 (3/4), rep08 43.6, rep05 45.6 |
| 4 | **1.13** / 13.4 / 50.3 | **rep03 1.13 (3/4)**, rep07 8.25 (4/3), rep08 8.96 (4/3), rep06 9.58 (3/4), rep04 11.7 (4/3) |

C_Q の差: K_true = 1 は 142–191、2 は 5.5–222、3 は 2.0–99、4 は 8.0–79。
K_true = 4 では C_Lap の差が他より小さく、rep03 は 1.13（最良 K3）だった。事前の規則に従い、局所最適化の診断は行っていない。

## 7. family 選択

4 つの K_true × 5 つのモデルの K × 10 dataset のすべての refit（200）で、選ばれた family 割当は真の割当（G3/B6/P3）と一致した。誤りはなかった。

## 8. 信号を揃えた文脈（Phase 9W の推論なしの文脈、median）

| 量 | K1 | K2 | K3 | K4 |
|---|---|---|---|---|
| 行の ‖f_l‖²（最小 / 最大） | 0.0028 / 1.74 | 0.028 / 1.58 | 0.074 / 1.03 | 0.16 / 0.95 |
| η^X の SD | 0.71 | 0.70 | 0.70 | 0.70 |
| Gaussian の記述的 SNR | 0.38 | 0.57 | 0.56 | 0.47 |
| Bernoulli の確率の SD | 0.164 | 0.146 | 0.153 | 0.151 |
| Poisson の率の q99 | 4.2 | 4.9 | 4.4 | 4.4 |
| η^Y の SD | 1.74 | 1.70 | 1.66 | 1.66 |
| Y の確率の q01 / q99 | 0.002 / 0.983 | 0.004 / 0.968 | 0.005 / 0.966 | 0.005 / 0.962 |
| 実現した edge 密度 | 0.339 | 0.328 | 0.333 | 0.332 |

平均の量は K_true によらずほぼ揃ったが、行ごとの loading energy の偏り（K1 で最大）と Y の確率の裾（K が小さいほど少し重い）は違う。解釈の限定にだけ使い、回帰や原因の探索はしていない。

## 9. K=2 → 3 の分解（Phase 9P/9T と同じ定義、median）

| 量 | K_true 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| C_Lap の fit gain | 136.5 | 152.0 | 349.5 | 328.0 |
| C_Lap の体積の増分 | 155.8 | 168.8 | 231.6 | 219.2 |
| C_Q の fit gain | 70.7 | 78.0 | 259.7 | 246.8 |
| C_Q の P_Z の増分 / P_θ の増分 | 212.84 / 43.17 | 同じ | 同じ | 同じ |
| fit gain の差（C_Lap − C_Q） | 68.8 | 69.8 | 79.6 | 74.7 |

これは K = 2 → 3 の比較だけであり、K_true = 4 での過小選択に関わる K = 3 → 4 の分解は事前に決めた量に含めていない。

## 10. Candidate B の技術的な挙動

新しい 150 refit と anchor の 50 refit はすべて OK。joint mode の反復は median 4–9（最大 49）、最後の grad_inf は最大 9.1e-9（≤ 1e-8）、
H の最小固有値はすべて正で、各 K_true の中での最小値は 0.097（K_true 1、モデルの K = 5）/ 0.159（2、K = 4）/ 0.057（3、K = 5）/ 0.184（4、K = 5）。convergence warning（診断のみ）はモデルの K とともに増えた。

## 11. DECISION

## **DECISION: MATCHED_K_TRUE_SENSITIVITY_CHARACTERIZED**

3 つの新しい K_true が凍結した設計で 1 回ずつ完了し（EM 300/300 成功、全 30 dataset が完全）、K3 の anchor を読み取りのみで統合でき、事前に決めた集計をすべて報告できた。この判定は exact の数によらない。

## 12. 解釈（INTERPRETATION）

- この信号を揃えた設定の rep01..rep10 では、C_Lap は K_true = 1, 2, 3 で 10/10、K_true = 4 で 8/10 の dataset で真の K を選び、外れた 2 件は 1 つ小さい K（K3）だった。
- C_Q は K_true が大きいほど小さい K を選ぶ傾向が強く（exact は 10 → 9 → 7 → 1）、K_true = 4 では 9/10 が過小（K1〜K3）だった。
- 両基準とも過大選択は 0 で、過大側の隣（K+1）との差は常に真の K の方が良かった。基準の差は主に過小側（K−1 との比較）に現れた。
- K_true = 4 では C_Lap の差も小さくなり（median 13.4、最小 1.13）、選択は K3 との境界に近かった。

## 13. CAN SAY / CANNOT SAY

**CAN SAY**
- この固定の信号を揃えた合成データの設定で、rep01..rep10 に対して、C_Lap は K_true = 1/2/3/4 で真の K を 10/10/10/8 回、C_Q は 10/9/7/1 回選び、両基準とも過大選択は 0 だった。
- 過小選択は K_true が大きいほど増え、C_Q で顕著だった。

**CANNOT SAY**
- 純粋な潜在次元の効果（揃えたのは平均の X loading energy、Y の分散、平均 edge 確率だけで、高次の積率や family ごとの信号の形は K_true で違う）。
- 一致性・漸近的な妥当性、一般の true-K recovery 確率、基準の優越性、この設定の外での頑健性、実データの潜在次元についての妥当性。
- 10 dataset の数を一般の確率として読むこと。

## 14. 最大の UNRESOLVED と次の Human 判断

**最大の UNRESOLVED**: K_true = 4 での過小選択（C_Q で 9/10、C_Lap で 2/10）と C_Lap の小さな差（rep03 1.13）が、
次元が増えるほど 1 次元あたりの信号が弱くなる設計（平均の energy を固定しているので各次元の分は K に反比例する）によるのか、
基準の罰則（C_Q の P_Z は次元ごとに一定の 212.84）によるのかは、この設計と事前の集計では分けられない。K = 3 → 4 の分解も事前に決めていない。

**次の Human 判断**: (a) K = 3 → 4 の分解を事前に決めて既存の artifact から記述するか、(b) 別の信号の揃え方（例: 次元ごとの energy を固定）を設計するか、
(c) ここで K_true の感度の特徴づけを閉じるか。いずれも新しい Human Gate が必要。
