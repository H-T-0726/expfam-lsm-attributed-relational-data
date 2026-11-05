# ① 以前の実験と canonical 実験の対応：生成器・推定器・K 選択基準・成功率（2026-11-06）

- **位置づけ**：研究ブランチ `research/k-selection-causal-analysis-20261106` の記録。**本文採用は未決定**（系列 E は prototype、昇格は Human Gate）。
- **数値の出所**：`expfam/results/k_selection/historical_canonical_causal_20261106/existing_reanalysis/`（スクリプト `tools/research_audit/analyze_k_causal_existing.py`、MCEM 0 件）。どの値も一次 CSV・保存状態から再計算した。
- **ラベル**：FACT（コード・一次 CSV で確認）／DERIVED（導出）／OBSERVED（実験結果）／INTERPRETATION／HYPOTHESIS。
- **系列を混ぜない（KI-002）**：以下の表は系列ごとに分けて示す。合算した成功率は作らない。

## 1. 照合点の再検証（FACT、一次 CSV から再計算）

| 照合点（依頼文） | 一次資料からの再計算 | 判定 | 出所 |
|---|---|---|---|
| 初期 Dual-ExpFam（0.5 系列）A 10/10、B 9/10、C 10/10 で K=3 | A 10/10、**B 9/10（残り 1 件は K=4 の過大選択）**、C 10/10 | 一致 | `expfam/results/exp_scenario_{A,B,C}_exp1_k.csv`（trial ごとの argmin） |
| fixed 版 A/B/C 合計 30/30 | 30/30（候補 K 1..9） | 一致 | `expfam/results/fixed_official/exp1_k9/fixed_exp1_bic_k1to9_bestk_by_trial.csv` |
| paper-type＋系列 E：C_Q 25/25 | 25/25（K_GEN ∈ {1,3,5,7,9} × 5） | 一致 | `paper_condition_k_criteria_20261006/full/selection.csv` |
| 同：C_Lap は完了 19/19、6 件未完了 | 完了 19 件すべて正解、INCOMPLETE 6 件（K_GEN=1:1、3:3、7:1、9:1） | 一致 | 同上 |
| canonical K_GEN=5、n=75、d=15：C_Q 0/8、C_Lap 3/8 | 0/8、3/8 | 一致 | `clap_clean_nsweep_20261006/full/selection.csv` |
| 同 n=100：C_Q 1/8、C_Lap 8/8 | 1/8、8/8 | 一致 | 同上 |
| 同 n=150：C_Q 7/8、C_Lap 8/8 | 7/8、8/8 | 一致 | 同上 |
| d-sweep d=15→30→60：C_Q 0/8→3/8→5/8 | 0→3→5（C_Lap 3→3→6） | 一致 | `dsweep_k_selection_20261105/full/selection.csv` |
| Phase 9X K_GEN=4：C_Q 1/10、C_Lap 8/10 | 1/10、8/10 | 一致 | `matched_k_true_sensitivity/phase9x_20260928/combined/confusion_{Q,Lap}.csv` |

訂正が必要な照合点はなかった。ただし「初期 B 9/10」の誤りは**過大選択**である（canonical での誤りがほぼ過小選択であるのと方向が違う）。

## 2. 条件の対応表（FACT。値はコードから転記）

全項目の機械可読版は `existing_reanalysis/historical_canonical_condition_matrix.csv`。

| 項目 | 初期 Dual-ExpFam（0.5）／fixed Exp1 | paper-type #139 B | canonical-clean-v1（A1・d-sweep・R2） |
|---|---|---|---|
| X / Y | A Pois/Bern、B Gauss/Pois、C Bern/Gauss | Gauss / Bern | Pois / Bern |
| K_GEN、候補 K | 3、1..6（fixed は 1..9） | 1,3,5,7,9、1..10 | 5（A1 は 1,3,5）、1..7 |
| n、d | 150、15 | 150、15 | 50〜150、15（30・60） |
| Z | N(0,I) の後に**列ごとに z-score** | 生成器による | N(0,I)、**正規化しない** |
| F | N(0,5) の各行を **‖f_l‖² = 0.9 に固定** | var_f=5、uniq=0.1 | reduced QR、特異値をすべて f_scale、**行ノルム² の平均 0.5** |
| 1 方向あたりの F エネルギー（FᵀF の固有値） | 乱数で決まる。Scenario A の 10 データでは平均 6.4 / 4.4 / 2.7（最小 1.5） | 制御なし | すべて 0.5·d/K（K=5, d=15 で **1.5**、K=3 で 2.5） |
| w0、w | Bern-Y：−1、**1.5**（w²K = 6.75） | データごとに乱数 | −1、√(3/K)（K=5 で **0.775**、w²K = 3） |
| X の後処理 | Gaussian-X のみ z-score | X を z-score（K\* は定義されない） | なし |
| 観測マスク | なし（全ペア） | なし | ペアの 20% を test に分ける（train 80%） |
| L、反復、start | 5、8、**1**（trial ごとに 1 回） | 5、8、2 | 5、8、2 |
| 選択規則 | trial ごとの argmin | 2 start 平均の argmin | 2 start 平均の argmin |
| 推定器系列 | **0.5**（`model_dual_expfam.py`）／fixed（0.5 なし） | E | E |
| 失敗時 | Newton 刻みを半分にして最大 2 回リトライ | fail fast | on_nan=fail、リトライなし |

- FACT：歴史的な `run_em_dual` は `true_params` を受け取るが、使うのは var_f のスケール（初期 F の分散）だけで、真の Z・F を初期値に渡していない（`reproduction/src/model.py:63-120`）。**真値リークによる成功ではない。**
- FACT：歴史的な Bernoulli-Y Scenario A の 10 データでは、辺密度の平均は 0.357、平均 p(1−p) は 0.131 だった。canonical（n=75）は 0.340、0.155。

## 3. 選択余裕の大きさの違い（OBSERVED）

| 系列 | 比較 | slack の中央値 | slack / n |
|---|---|---:|---:|
| 初期 Scenario A（0.5、K_GEN=3、n=150） | C_Q(2) − C_Q(3) | 1,927 | 12.8 |
| 初期 Scenario B | 同 | 5,359 | 35.7 |
| 初期 Scenario C | 同 | 34,223 | 228（Gaussian-Y、σ_y=0.1） |
| canonical K_GEN=5、n=150、d=15 | C_Q(4) − C_Q(5) | +154 | 1.02 |
| canonical K_GEN=5、n=75、d=15 | 同 | −129 | −1.72 |

歴史的条件では、正しい次元を加えたときの余裕が 1 ノードあたり 12〜228 あり、C_Q の「入場料」2.84/ノード（P_Z）とパラメータ罰則を大きく上回っていた。canonical K_GEN=5 では余裕が 1 ノードあたり ±2 程度しかなく、境界の近くにある。

## 4. 違いの整理（何が交絡しているか）

歴史的条件と canonical 条件は、次の点が**同時に**違う。このため、歴史的成功と canonical の過小選択を比べるだけでは、どの違いが効いたかは特定できない（INTERPRETATION）。

1. K_GEN（3 対 5）
2. 1 方向あたりの X 信号（F エネルギー約 4.5 平均、乱数で不均一 対 1.5 で一様）
3. 1 方向あたりの Y 信号（w² = 2.25 対 0.6）
4. Z の標本正規化の有無
5. 観測マスク（全ペア 対 80%）
6. 推定器系列（0.5／fixed 対 E）と start 数
7. X の family（Scenario B・C は Poisson-X でない）

本ブランチの統制実験（レポート③）は、このうち **2 と 3 だけ**を、同じ canonical モデル・推定器・データの上で変えた。

## 5. 旧 0.5 係数について（Q2 への答えの根拠）

- OBSERVED：同じ歴史的生成条件で、0.5 系列は 29/30（B の 1 件が過大）、fixed 系列は 30/30 で K=3 を選んだ。
- INTERPRETATION：歴史的条件のように選択余裕が 1 ノードあたり 10 以上ある領域では、Y 側の 0.5 の有無は K の選択結果を変えなかった。これは「0.5 係数が K 選択の成功を生んだのではない」ことを示唆する。
- 限定（KI-001）：0.5 系列の結果については、**Newton 方向が全体として正しいとは断定できない**。また、境界付近（canonical K_GEN=5 のような条件）で 0.5 が選択を変えるかは検証していない。

## 6. ここから言えないこと

- 「歴史的条件では K の識別が一般に容易」：歴史的条件は K_GEN=3、n=150 の 1 条件、各 10 データだけである。
- 「canonical 生成器は不適切」：canonical 生成器は真のモデルからの文字通りの生成であり、K\*=K_GEN の導出（canonical docs では未承認）が成り立つ条件として作られている。歴史的生成器は Z の標本正規化と Gaussian-X の z-score のため、K\* の扱いが異なる。
