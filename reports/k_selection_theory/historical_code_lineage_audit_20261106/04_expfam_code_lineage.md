# 04 自作の指数型分布族モデルの系列と、歴史的条件・canonical 条件の比較（2026-11-06）

系列は混ぜない（KI-002）。§2 の表は系列ごとの結果を 1 行ずつ並べた**参照用の一覧**で、成功率を合算・比較するものではない。0.5 系列・Python 再現（1/2 あり）の行には「Newton 方向が全体として正しいとは断定できない」（KI-001）が付く。完全な比較表は `historical_vs_expfam.csv`、ファイルの由来は `code_lineage_inventory.csv`。

## 1. 系列と時期（FACT、git 履歴）

| 系列 | 主なファイル | リポジトリへの追加 | 主な用途 |
|---|---|---|---|
| 0.5 系列（初期 ExpFam） | `expfam/src/model_dual_expfam.py`、`data_generator_expfam.py`、`utils_expfam.py`、`run_exp2_bic_v2.py` | 7c3040f（2026-04-27、一括） | 学会予稿・初期実験（exp2_bic A/B/C を含む） |
| fixed 系列 | `model_dual_expfam_fixed.py` | 05bc062（2026-06-16） | 1/2 を除いた版。実データ段階 |
| 系列 E | `experimental/model_dual_expfam_consistent.py`（2026-08-21）、`data_generator_canonical.py`（2026-09-04）、`laplace_k_criterion.py`（2026-09-28）、`k_criterion_panel.py`（2026-10-06） | 個別の commit | 現在の K 選択研究（#139、PR #155〜#157） |

## 2. 歴史的な成功と canonical の失敗の数値（OBSERVED、`historical_k_selection_summary.csv`）

| 系列・条件 | X/Y | n | 基準 | K\*=5 の正解 | K\*=5 の過小側の余裕（中央値、最小） | K\*=9 の正解 | K\*=9 の過小側の余裕（中央値、最小） |
|---|---|---:|---|---|---|---|---|
| 論文 Fig. 3 | Gauss/Bern | 150 | 論文の BIC | 1/1 曲線 | 386 | 1/1 曲線 | 57 |
| Python 再現 | Gauss/Bern | 150 | plug-in Q2 の最良反復 | 1/1 | 3,495 | 1/1 | 1,846 |
| #139 paper-type（E） | Gauss/Bern | 150 | C_Q | 5/5 | 3,255（1,085） | 5/5 | 1,734（1,618） |
| 初期 ExpFam Scenario A（0.5） | **Pois/Bern** | 150 | calc_bic_dual | **5/5** | 1,095（770） | **5/5** | 236（90） |
| 初期 ExpFam Scenario B（0.5） | Gauss/Pois | 150 | calc_bic_dual | 4/5（1 件は過大） | 3,540 | 3/5（2 件は過大） | 2,189 |
| 初期 ExpFam Scenario C（0.5） | Bern/Gauss | 150 | calc_bic_dual | 5/5 | 29,895 | 5/5 | 26,889 |
| canonical-clean-v1（E） | Pois/Bern | 75 | C_Q | **0/8** | **−129（−164）** | — | — |
| 同 | Pois/Bern | 75 | C_Lap | 3/8 | −11（−49） | — | — |
| 同 | Pois/Bern | 100 | C_Q | 1/8 | −50 | — | — |
| 同 | Pois/Bern | 150 | C_Q | 7/8 | 154（−90） | — | — |

- **同じ Poisson-X/Bernoulli-Y でも、歴史的な生成器（初期 ExpFam Scenario A、n=150）では K\*=1..9 の 25/25 が正解だった。** K\*=5 の余裕は 1 ノードあたり 7.3、canonical の n=150 は 1.0、n=75 は −1.7（OBSERVED）。
- 初期 ExpFam の誤りは **Scenario B の過大選択**（K\*=5 で 1 件、7 で 2 件、9 で 2 件）で、canonical の過小選択とは向きが逆。

## 3. 信号の強さ（DERIVED＋OBSERVED、`signal_strength_summary.csv`、K=5 の中央値）

1 ノード・1 方向あたりの Fisher 情報で比べる。X の情報は Gaussian なら eig(FᵀΣ⁻¹F)、Poisson なら eig(Σ_l λ_il f_l f_lᵀ)。Y の情報は eig(w² Σ_j s_ij(1−s_ij) z_j z_jᵀ) で、canonical は train ペアの割合 0.8 を掛けた。真の Z・F で計算した値で、推定値ではない。

| 生成器（K=5） | n | FᵀF の固有値（平均） | X の情報 | w² | 辺密度 | Y の情報 | 合計 |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical-clean-v1 | 75 | 1.50 | 2.07 | 0.60 | 0.34 | 5.10 | **7.2** |
| canonical-clean-v1 | 150 | 1.50 | 2.07 | 0.60 | 0.34 | 10.4 | 12.5 |
| 初期 ExpFam Scenario A（Poisson-X） | 150 | 2.70 | 4.14 | 2.25 | 0.38 | 32.1 | **36.2** |
| MATLAB / Python 再現型（Gaussian-X） | 150 | 2.70 | 27.0 | 18.8 | 0.45 | 105 | **132** |
| 論文本文の生成器（F ~ N(0,10)） | 150 | 151 | 1,508 | 記載なし | — | — | — |

- 注意：真のパラメータで計算した値で、推定値ではない。canonical の行は実験に使ったデータではなく、同じ定義から新しい seed で生成したデータで計算した。
- 歴史的な Poisson-X 条件と canonical（n=75）の比は、X 2.0 倍、Y 6.3 倍、合計 5.0 倍。MATLAB/Python 型では合計 18 倍。**情報量の差が最も大きいのは Y 側（w² と n）である**（DERIVED）。
- **ただし、情報量の差の大きさは原因としての効果の大きさを意味しない。** PR #155 の一要因比較（n=75）では、F のエネルギーだけを 1.8 倍にした B が C_Q 3/8・C_Lap 6/8、w² だけを 1.8 倍にした C が C_Q 2/8・C_Lap 4/8 で、F を強めた効果は w を強めた効果と同程度以上だった（`expfam/results/k_selection/historical_canonical_causal_20261106/analysis/selection_frequency.csv`）。
- K=9 の初期 ExpFam Scenario A は、FᵀF の平均固有値が 1.50 で canonical の K=5 と同じだが、Y の情報は 26.2 あり、K\*=9 を 5/5 で選んだ（余裕の最小 90）。

## 4. 同時に変わっている条件（INTERPRETATION）

歴史的な成功（Scenario A、n=150）と canonical の失敗（n=75）の間では、次の 8 つが**同時に**変わっている。

1. n（150 → 75）
2. w（1.5 → 0.775）。w² は 3.75 分の 1
3. F の 1 方向あたりのエネルギー（平均 2.7、乱数で不均一 → 1.5、均一）
4. Poisson の rate（平均 exp(0.45) → exp(0.25)）
5. Z の標本 z-score の有無
6. 観測マスク（全ペア → 80%）
7. 推定器（0.5 系列、刻みの半減リトライ、1 start → 系列 E、2 start 平均）
8. 基準（calc_bic_dual → C_Q。どちらも Q2 の MC 平均）

このうち一要因の比較は既にある：n だけ（canonical C_Q：n=75 で 0/8、100 で 1/8、150 で 7/8）、F だけ（PR #155 B、n=75：C_Q 3/8、C_Lap 6/8）、w だけ（C：C_Q 2/8、C_Lap 4/8）、両方（D：C_Q 3/8、C_Lap 7/8）。D でも w² は 1.08 で、歴史的条件（2.25）の半分以下。
