# Frozen protocol — Experiment A（2026-11-27、本実験の fit 前に凍結）

**種類**：探索的（EXPLORATORY）。結果を見た後に confirmatory evidence と呼ばない。
**系列**：lineage E-scalar（`DualExpFamLSMConsistent`、`numerics_mode="consistent"`、`failure_policy="fail_fast"`）。全列共通 family のモデル構造だが、**fixed 系列（`DualExpFamLSMFixed`）そのものではない**。prototype・本文採用は Human Gate。

## Research Question
X の全列が同じ分布族（Bernoulli）のとき、関係データ Y の学習観測量が少ないと、X の分布誤指定（Bernoulli→Poisson）による潜在構造 Z の推定誤差の悪化は大きくなるか。

## 仮説と対立する結果
- H_A：$I>0$（sparse のほうが誤指定の悪影響が大きい）。列ごとの family の M1（G3/B6/P3）で観測された向き。
- 対立：$I\le 0$、または $I$ が 0 付近でデータ間で符号がそろわない。どちらも研究結果として報告する。

## 固定条件
- n=150、d=15、K_TRUE=3、候補 K は 3 のみ（K 選択はしない）。
- 生成器：`data_generator_canonical.generate_canonical_data`（canonical-clean-v1、正規化・clip なし）。真の X は全 15 列 Bernoulli、Y は Bernoulli。
- 信号：K-exp2 の K=3 の規則（regime a と b は K=3 で一致）：F = √2.5·Q（Q は直交列）、w = 1.0、w0 = K-exp2 の凍結較正値（母集団 edge 確率 0.20、w0 = −1.96727…）。
- 推定：`run_em_experimental`、L=10、MCEM 10 反復、K=3、informed init（既存の規則）、retry・seed 置換・hidden repair なし。
- Y：上三角ペアの 10%（1,118 ペア）を test として先に確保し、どの学習にも使わない。残り（10,057 ペア）が dense の学習 mask。sparse はその中の無作為な 10%（1,006 ペア）で、dense の部分集合。学習に使わないペアは mask で E-step・M-step・strict Q から除外される（0 として学習しない）。技術検証で、非学習ペアの Y を反転しても勾配・strict Q・M-step（w0）が変わらないことを確認済み。

## 変化させる条件
- Y の観測量：dense / sparse。
- X の推定 family：A1 Bernoulli（正指定）／A2 Poisson（誤指定、{0,1} 上で尤度が定義できる support-compatible な誤指定）。

## seed
- data_seed = 31,261,127,000 + rep（rep = 0..19）。pilot は +5,000,000、rep 100・101。
- test/pool 分割 seed = data_seed + 700,000（`eval_utils.make_pair_split`、test_ratio 0.10）。sparse の抽出 seed = data_seed + 800,000。
- init_seed = 10·data_seed + t、mc_seed = 10·data_seed + 5 + t（t = 0, 1）。**4 条件（2 観測率 × 2 family）で同じ init/mc seed を使う対応設計**。

## 実行数
20 データ × 2 観測率 × 2 family × 2 初期値 = **160 fit**。pilot（別 seed）は 8 fit + 同一 seed 再現 1 fit = 9 fit（実施済み、9/9 OK、再現一致）。pilot の結果は本実験に混ぜない。

## Primary Metric と集計
- fit ごとに Procrustes RMSE_Z（`Z_est` = 最後の MC サンプル、`utils_expfam.procrustes_rotation`）。
- **初期値の集約**：各（データ, 観測率, family）で 2 初期値の RMSE_Z の**算術平均**。
- $\Delta_{r,\text{rate}} = \overline{\mathrm{RMSE}}(\text{Poisson}) - \overline{\mathrm{RMSE}}(\text{Bernoulli})$、$I_r = \Delta_{r,\text{sparse}} - \Delta_{r,\text{dense}}$。
- **独立な実験単位は 20 個の生成データ**（初期値ではない）。
- 報告：平均、中央値、SD、正の数・負の数、bootstrap 95% 区間（10,000 回、seed 20261127、データ単位の再標本化）、t 分布 95% 区間、両側 sign test の p 値（記述用。仮説検定の結論には使わない）。
- 副次：Z の L サンプル平均の RMSE_Z、gram 相関、test ペアの Bernoulli 対数スコア（正 − 誤）。

## 失敗の扱い
- fit が FAILED の場合、その（データ, 観測率, family）は欠測。そのデータの $\Delta$・$I$ は計算せず INCOMPLETE とし、**分母は 20 のまま**報告する。再実行・seed 変更はしない。

## 実行上限・再開
- 4 worker、BLAS 1 スレッド、利用可能メモリ < 1,500 MB が 2 回続いたら新しい fit を出さない。
- 再開：`python tools/research_audit/run_two_gaps_20261127.py execute A expfam/results/k_selection/two_gaps_20261127/expA/run --protocol expfam/results/k_selection/two_gaps_20261127/expA/protocol/protocol.json --resume`（fit_id が per_fit.csv にあれば完了扱い）。

## 既存実験との違い
- M1・E1・MX2 は列ごとの family（G3/B6/P3 など、E-3）。本実験は**全列が同じ family**（E-scalar）。データ生成（K=3 の K-exp2 規則、n=150）も違うので、M1 と数値を並べて比較しない。

## 成果物
`expfam/results/k_selection/two_gaps_20261127/expA/`（tech_check、pilot、protocol、run、analysis）。NPZ は git 管理外（SHA-256 を per_fit.csv に記録）。
