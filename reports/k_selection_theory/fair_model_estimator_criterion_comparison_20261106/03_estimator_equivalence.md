# 03 推定器の同等性（Phase B と E1、2026-11-06）

対象：旧 Python 再現の推定器（OLD、`reproduction/src/model.py` を adapter から呼ぶ。コードは変更しない）と系列 E（`em_runner_instrumented`、consistent）。系列 E は prototype（本文採用不可）。数値の出所は `expfam/results/k_selection/fair_model_estimator_criterion_comparison_20261106/` 以下の CSV で、ファイル名だけを書く。

## 1. 手順の対応（FACT、`math_code_correspondence.csv`・`implementation_differences.csv`）

| 段階 | OLD | E | 同じか |
|---|---|---|---|
| 初期の Z・F・Σ | `initialize_params(seed)` | 同じメソッドを継承 | 同じ seed なら一致 |
| 初期の w0・w | 元のスクリプトは n×n 全体（対角を含む）の平均の logit、w=0.5 | 上三角の辺密度の logit、w=0.5 | E1 の adapter で E の定義にそろえた（I5） |
| E-step の対象 | ノードごとのラプラス条件付き。**Y の score と曲率に 1/2**（I1） | 同じ方式。1/2 なし。マスク対応 | **違う** |
| Newton | 刻み 0.5、最大 10 回 | 同じ | 同じ |
| Gibbs の順序・L 標本の連鎖 | i = 1..n を順に | 同じ | 同じ |
| `scale_Z` | L 個の標本をまとめて縮尺 | 継承 | 同じ |
| F・Σ の更新 | 閉じた式 | 同じ形の閉じた式 | 同じ |
| w0・w の更新 | Adam（lr 0.01、50 回）、マスクなし | Adam（lr 0.01、50 回）、マスクつき | E1（全ペア）では同じ目的関数 |
| 乱数 | `default_rng(seed)` | 同じ | seed は同じ。消費順は違いうる |
| 選択の規則（元の Python Exp.2） | 30 反復の最良の plug-in・3 初期値の最良 | 最終状態・2 start の平均 | E1 では共通の規則にそろえ、元の規則は `PY_hist` として記録 |

**DERIVED**：OLD の E-step は、Y の情報を半分にした「焼き戻し」条件付き事後分布を目標にしている。有限差分との相対誤差は、勾配が 0.33、精度行列が 0.093（`equivalence_numeric_checks.csv`）。E の E-step は有限差分と一致する（1.7e−9・2.3e−8）。

## 2. 同じ状態での E-step の比較（OBSERVED、`estep_same_state.csv`）

E1 の E の状態（K = K_GEN、rep 1、start 1）で θ と Z を固定し、各ノード i について両方の目的関数の条件付きモードと共分散 A_i⁻¹ を計算した（fit はしない）。

| 状態 | ŵ | モードの相対差（中央値・最大） | 共分散のトレース比 OLD/E（中央値・範囲） |
|---|---:|---|---|
| G_old K_GEN=5 | 3.89 | 0.031・0.110 | 1.38（1.25〜1.62） |
| G_old K_GEN=9 | −2.38 | 0.044・0.091 | 1.48（1.36〜1.64） |
| G_canon K_GEN=5 | 1.11 | 0.041・0.247 | 1.25（1.14〜1.38） |
| G_canon K_GEN=9 | 0.81 | 0.061・0.136 | 1.27（1.19〜1.34） |

OLD の事後分布は E より 25〜48% 広く、モードは 3〜6% ずれる。Y の結合が強いほど（G_old、|w| が大きい）ずれは大きい。

## 3. 同じデータ・同じ初期状態・同じ seed での推定（E1、OBSERVED）

公平性：`MATCHED_MODEL; MATCHED_CRITERION; ONE_FACTOR_COMPARISON`（推定器の全体。内部の違いは I1〜I5）。n=150、d=15、Gaussian-X、全ペア観測、K_CAND = 1..10、データ 2 つ × 2 start。

**K の選択**（`full_k_selection.csv`、`estimator_agreement_e1.csv`）

| 基準 | OLD と E で K_HAT が同じ | 補足 |
|---|---|---|
| A C_Q | 8/8 | 両推定器とも 8/8 データで K_HAT = K_GEN |
| E C_Lap | 6/8 | 残りの 2 つは Laplace が NOT_STATIONARY で INCOMPLETE。完全な組では 6/6 が一致し、すべて K_GEN |
| B C_Q − P_Z | 0/8 | E は 8/8 が過大（K=10）。OLD は K_GEN=9 の 4/4 と G_canon K5 の 1/2 が正解 |
| C 最後の標本での plug-in | 1/8 | E は 8/8 が過大。OLD は 6/8 が正解（G_old 4/4、G_canon 2/4） |
| D 結合モードでの plug-in | 6/8 | 両者とも完全な組はすべて K=10（過大） |
| PY_hist（OLD のみ、元の Python の規則） | — | 8/8 が正解 |

**推定量**（`estimator_pairs_e1.csv`、K = K_GEN、2 start の平均）

| データ | 真の w | ŵ OLD | ŵ E |
|---|---:|---:|---:|
| G_canon K5 rep 1・2 | 1.16 | 1.02・1.06 | 1.11・1.14 |
| G_canon K9 rep 1・2 | 0.87 | 0.70・0.79 | 0.81・0.89 |
| G_old K5 rep 1・2 | 5.03・−5.93 | 3.43・−2.97 | 3.71・−2.84 |
| G_old K9 rep 1・2 | −2.78・4.90 | −1.78・2.33 | −2.38・3.44 |

- OLD の |ŵ| は、8 データ中 7 つで E より小さい（G_old K5 rep 2 だけ逆）。G_old（Z と X を z-score）では真の w の尺度が推定の尺度と違うので、真値との比較は G_canon だけで行う。G_canon では E のほうが真値に近い。
- C_Q（中央値、E − OLD）：−401（G_canon K5）、−476（G_canon K9）、−597（G_old K5）、−651（G_old K9）。C_Lap は −42〜−143。どの K でも E の状態のほうが C_Q・C_Lap が小さく、K による差の形はほぼ同じ（`fig_E1_estimator_pairs.png`）。
- 回復（`parameter_recovery.csv`、K = K_GEN）：ZZᵀ の相関は OLD 0.946〜0.980、E 0.953〜0.983。η_Y の相対 RMSE は、8 つ中 7 つの条件で E のほうが小さい（G_canon K9 rep 1 は OLD 0.345・E 0.302、G_old K5 rep 2 だけ逆）。
- 計算時間（中央値）：OLD 12.5 秒、E 15.1 秒。

## 4. 判定

| 層 | 判定 |
|---|---|
| アルゴリズムの骨格（初期化、Gibbs の順序、Newton の設定、`scale_Z`、M-step） | 同じ |
| E-step の目標分布 | **違う**（OLD は Y の情報が半分。事後分布は 1.25〜1.48 倍広い） |
| E1 での C_Q による K 選択 | **同じ結果**（8/8）。n=150 のこの条件では、推定器の違いは C_Q の K 選択を変えなかった（OBSERVED、各条件 2 データ） |
| Q_Z を除いた基準（B・C） | 推定器によって結果が変わる（OLD は正解、E は過大のことがある）。**推定器と基準の交互作用**（OBSERVED）。理由は未検証（HYPOTHESIS：OLD の事後分布が広いと、K を増やしたときの Q1 の増え方が小さくなる） |

- **INTERPRETATION**：1/2 は「選択を成功させる原因」ではなかった。少なくとも n=150・全ペア観測・d=15 では、両推定器とも C_Q で K_GEN（5 と 9）を選んだ。
- **限界**：各条件 2 データ。n=75 や部分観測、Poisson-X での OLD と E の比較は行っていない（OLD は Poisson-X と観測マスクに対応していない）。MATLAB の推定器は実行していない（PR #158）。
