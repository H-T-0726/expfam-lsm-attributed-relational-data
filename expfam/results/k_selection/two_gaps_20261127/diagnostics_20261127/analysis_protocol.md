# 分析 protocol：Experiment A・B の事後の原因分析（2026-11-27、計算の前に記述）

## 位置づけ
- **事後の探索的分析（POST HOC, EXPLORATORY）**。A・B の本実験の結果（`results_report.md`、commit `1ef9096`）を見た後に計画した。
- ここで得た数値は、本実験の primary 結果を置き換えるものでも、confirmatory evidence でもない。
- 新しいデータ生成・fit・最適化・数値積分・モデル実装は**行わない**。
- 使う材料は、保存済みの NPZ（760 個）、per_fit.csv、既存の生成関数で seed から決定的に再構成した学習データ（X、Y、真の Z・F、mask）。
  - 再構成したデータは、per_fit.csv の hash（x_hash、y_hash、test_hash、z_hash、f_hash）と一致することを確認したうえで使う。
- 系列は E-scalar（prototype）。C_Q は Schwarz BIC ではなく、C_Lap は厳密な周辺尤度ではない。

## 保存状態から復元できるもの・できないもの（コードの確認による）

| 量 | A | B | 備考 |
|---|---|---|---|
| 推定 F、w0、w | ○ | ○ | NPZ |
| Z の最後の MC サンプル（`Z_last`）、L サンプルの平均（`Z_mean`） | ○ | ○ | NPZ |
| Gaussian-X の推定分散（`sigma` の対角） | — | ○（G01・GM） | NPZ。`calc_log_likelihood_X` は `diag(sigma)` を分散として使う |
| var_z | — | ○ | NPZ |
| L=10 個の MC サンプル全体 | × | × | 保存されていない |
| 反復ごとの Q の履歴 | × | × | 保存は最後の相対変化（`q_rel_change_last`）と、M-step で Q が減った回数だけ |
| 事後予測分布（MC サンプルでの平均） | × | — | サンプルが無いので計算しない |

## Experiment A（同じ 20 データ、正指定 Bernoulli と誤指定 Poisson の対応比較）

**予測の定義**
- **主**：既存の held-out 評価と同じ plug-in 予測。$p_{ij} = \sigma(\hat w_0 + \hat w\, \hat z_i^\top \hat z_j)$ で、$\hat z$ = `Z_last`。test ペアは上三角。
- **副**：`Z_mean` を使った plug-in 予測。これは事後予測分布ではない。
- 参照 1：定数予測。各データ・各観測率の**学習ペアの正例率**を確率にする（モデルなし）。
- 参照 2：真のパラメータ（真の Z、w0、w）での予測。真の予測確率で、達成可能な基準の目安。

**fit ごとの集計**
- 予測
  - test の正例率
  - 正例・負例それぞれの予測確率の平均
  - 正例・負例それぞれの平均対数スコア
  - 全体の平均対数スコア（既存の `heldout_log_score` と一致することを確認する）
  - Brier score、AUC（Mann–Whitney）、AP（平均適合率）
- 校正
  - 予測確率の平均 − 正例率
  - 校正の傾き（y を logit(p) にロジスティック回帰したときの係数）
  - ECE（10 等幅 bin）
- パラメータ
  - $\hat w$、$\hat w_0$、$\lVert\hat F\rVert_F$
  - $\hat z_i^\top \hat z_j$ の SD、線形予測子 $\hat\eta$ の SD
  - 真の $\eta$ との相関
  - Gram 相関（真の Z と）、RMSE_Z（既存の値と一致することを確認する）
  - 予測確率の SD

**集計**
- 独立な単位は 20 データ。各（データ、観測率、family）で、2 初期値の値を**算術平均**する（本実験 protocol と同じ）。
- 対応差は「正指定 − 誤指定」。対数スコア・AUC・AP は大きいほど良く、Brier は小さいほど良い。
- 報告するもの
  - 平均、中央値
  - 正指定のほうが良いデータの数
  - データ単位の bootstrap 95% 区間（10,000 回、seed 20261128）
- 全体の対数スコアの差は、正例と負例の寄与に分解する：$\Delta = \pi\,\Delta_{\rm pos} + (1-\pi)\,\Delta_{\rm neg}$。$\pi$ は test の正例率。

## Experiment B

**3.1 推定分散**
- G01・GM の全 fit について、15 列の推定分散の平均・最小・最大と、真の分散との比。
- 条件 × 候補 K ごとに、データの中央値（2 初期値の平均）。
- 初期値間の差（|t0 − t1|）の中央値。

**3.2 推定後の情報行列**（真の情報と区別する）
- Gaussian：$\hat F^\top \mathrm{diag}(1/\hat\sigma_l^2)\hat F$。
- Poisson：次の 2 つを両方報告する。
  - (a) $\hat F^\top \mathrm{diag}(\overline{e^{\hat f_l^\top \hat z_i}})\hat F$：推定 Z の行の平均。
  - (b) $\hat F^\top \mathrm{diag}(e^{\lVert \hat f_l\rVert^2/2})\hat F$：$z \sim N(0,I)$ での期待値。
- Y 側（3 条件共通の式）：$\hat w^2\, \frac1n \sum_i \sum_{j\ne i} p_{ij}(1-p_{ij})\, \hat z_j \hat z_j^\top$。推定値と `Z_last` で計算する。
- 報告するもの：trace、最小・最大固有値、条件数、trace / K（1 次元あたり）。条件 × K ごとに、データの中央値を出す。
- 回転に依存しない量（固有値）だけを使う。

**3.3 C_Q の項分解**
- `Z_last` の 1 サンプルで、既存のモデル関数（`calc_log_likelihood_X`、`calc_log_likelihood_Y`、strict の定数補正）を使い、次を計算する。
  - $\ln p(Z)$、$\ln p(X\mid Z)$、$\ln p(Y\mid Z)$
  - 罰則 $p\ln n$
  - 1 サンプル版の $C_Q^{(1)} = -2(\ln p(Z)+\ln p(X|Z)+\ln p(Y|Z)) + p\ln n$
- 保存されている C_Q（L=10 サンプルの平均）との差を**MC 残差**として報告する。項分解は 1 サンプル近似であることを明記する。
- K=9 との差 $C(K)-C(9)$ を、項ごとに条件 × K でデータの中央値として出す。
  - 選択は保存されている C_Q で行う（変えない）。
  - 1 サンプル版で選んだ K が保存 C_Q の選択と一致するかを記録する。
- C_Lap は、per_fit.csv に保存済みの項（`lap_fit_mode`、`lap_zhat_sq`、`lap_logdet_H`、`lap_penalty`）で同様に分解する。新しい計算はしない。

**3.4 収束**
- 保存済みの `q_rel_change_last` と `mstep_q_decreases` の分布を、条件 × K で出す。
- 2 初期値間の C_Q の差と、K 間の C_Q の差を比べる。
- 履歴が無いため、収束を判定はしない。

## 整合性の確認
- 再構成したデータの hash が per_fit.csv と一致すること。
- 再計算した `heldout_log_score`、RMSE_Z、gram 相関が per_fit.csv と一致すること（A）。
- 1 サンプル版の項の合計と、保存 C_Q との差の大きさ（B）。
- 別プロセスの確認スクリプトで、出力 CSV の集計を、分析スクリプトを import せずに再計算する。

## 書かないこと
- 「w の縮小が予測改善の原因である」と断定しない。
- 「Gaussian と Poisson の情報量を一致させた」「Poisson 分布そのものが K 選択失敗の原因」と書かない。
- 「C_Lap が常に優れている」と書かない。
