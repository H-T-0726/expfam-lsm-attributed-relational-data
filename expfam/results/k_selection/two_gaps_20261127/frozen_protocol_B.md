# Frozen protocol — Experiment B（2026-11-27、本実験の fit 前に凍結）

**種類**：探索的（EXPLORATORY）。confirmatory evidence と呼ばない。
**系列**：lineage E-scalar（`DualExpFamLSMConsistent`、`numerics_mode="consistent"`、`failure_policy="fail_fast"`）。prototype・本文採用は Human Gate。C_Q は Schwarz BIC ではない。C_Lap は厳密な周辺尤度ではない。

## Research Question
K-exp2 の regime b、K_TRUE=9 で、Gaussian-X / Bernoulli-Y は C_Q 10/10、Poisson-X / Bernoulli-Y は C_Q 0/10 だった。この差は、Gaussian-X（条件付き分散 0.1）の潜在変数に関する情報量が Poisson-X より大きかったためか。

## 仮説と対立する結果（どれも研究結果として報告する）
1. 情報量（trace）をそろえると Gaussian も過小選択する（情報量の説明と整合）。
2. そろえても Gaussian は成功する（情報量の trace だけでは説明できない）。
3. 初期値・最適化の不安定性が結果を支配する（2 初期値の選択の不一致が多い）。
4. C_Q と C_Lap で傾向が違う。
5. INCOMPLETE が多く結論を出せない。

## 固定条件
- n=150、d=15、K_TRUE=9、候補 K=1..10、Y Bernoulli、10 データ、2 初期値、MCEM 10 反復、L=10。
- 信号：K-exp2 の regime b の規則：F = s·Q（Q は直交列、s² = 0.5d/K → F'F = s²I、行ノルム² の平均 0.5）、w = √(3/K)、w0 = K-exp2 の凍結較正値（母集団 edge 確率 0.20、w0 = −2.0229…）。
- **データ生成（本実験用の新しい関数、既存生成器は変更しない）**：`rng = default_rng(seed)` で Z（n×K、iid N(0,1)）、`build_full_rank_loadings`（canonical-clean-v1 と同じ関数）で F。Y は独立な stream `default_rng([seed, 11])`、Gaussian の雑音 E は `default_rng([seed, 22])`、Poisson-X は `default_rng([seed, 33])`。したがって**同じ replicate の 3 条件で Z・F・Y・w・w0 は完全に同一**で、X だけが違う（技術検証で一致を確認済み）。Gaussian の 2 条件は同じ E を使う（X = ZF' + √var·E）。正規化・clip はしない。
- **注意**：この生成は K-exp2 本実験とは乱数の使い方が違う（K-exp2 は系列ごとに seed が違い、X を Y より先に引く）ため、K-exp2 のデータの再現ではない。

## 変化させる条件（X だけ）
- B1 `G01`：Gaussian-X、条件付き分散 0.1（従来条件）。
- B2 `GM`：Gaussian-X、分散 $\sigma^2_{\rm matched} = \mathrm{tr}(F^\top F)/\mathrm{tr}(\mathbb E_z[\mathcal I_P])$、$\mathbb E_z[\mathcal I_P] = F^\top\mathrm{diag}(e^{\lVert f_l\rVert^2/2})F$。replicate ごとに真の F から計算。技術検証の値：0.758〜0.773。
- B3 `P`：Poisson-X。

**情報量の対応は trace の一致だけ**で、行列全体の一致ではない。技術検証（`tech_check_v2/information_diagnostics.csv`）で、Gaussian 側は $F^\top F/\sigma^2$ が等方（条件数 1.00）、Poisson 側は条件数 1.14〜1.17 と、方向ごとの違いが残ることを記録した。G01 の trace は 75、GM と P は約 9.8。ほかの family 由来の違い：Gaussian は各列の分散を推定し（C_Q の罰則 p に +d）、Poisson は分散が平均で決まる。z ごとの情報は Poisson では $e^{f^\top z}$ で変わる（期待値で合わせただけ）。「完全に公平な分布比較」「分布の純粋な因果効果」とは主張しない。

## 推定と基準（K-exp2 と同じ）
- `run_em_experimental(X, Y, family_x, "bernoulli", K, L=10, num_iter=10, numerics_mode="consistent", failure_policy="fail_fast", init_seed, mc_seed, compute_strict_Q=True, mstep_q_diagnostic=True)`。
- C_Q：`calc_bic_exp`（p = Kd − K(K−1)/2 + [Gaussian-X のとき d]）。
- C_Lap：凍結評価器 `run_k_exp2_clap_comparison.evaluate_state`（laplace_k_criterion v1、Z0 = Z_last、grad_tol 1e-8、max_iter 200、d_K ln n）。**許容値の変更・別の開始点・max_iter の延長はしない**。

## seed
- data_seed = 32,261,127,000 + rep（rep 0..9）。pilot は +5,000,000、rep 100。
- init_seed = 10·data_seed + t、mc_seed = 10·data_seed + 5 + t（t = 0, 1）。**3 条件で同じ init/mc seed**（対応設計）。

## 実行数
3 条件 × 10 データ × 10 候補 K × 2 初期値 = **600 fit**。pilot（別 seed、1 データ × 3 条件 × 10 K × 初期値 1 = 30 fit ＋ 同一 seed 再現 1 fit = 31 fit）。pilot の結果は本実験に混ぜない。

## Primary Metric と集計
- 選択規則（K-exp2 と同じ）：データごとに 2 初期値の基準値の算術平均をとり argmin。差が 1e-9 以内の同点は小さい K。
- C_Lap は、そのデータの 20 評価のうち 1 つでも status が OK でなければ INCOMPLETE。**分母は 10 のまま**。fit が FAILED の場合も INCOMPLETE。
- Primary：各条件・各基準の正解・過小・過大・INCOMPLETE の数。
- 対応比較：同じ replicate の 3 条件の選択 K（`paired_by_replicate.csv`）。
- Secondary：選択 K − 9、基準曲線（C(K) − C(9) の中央値）、2 初期値の選択の不一致、fit の収束診断（M-step の Q 減少、最後の相対変化）、情報量の診断。
- family の違う条件の間で C_Q・C_Lap の絶対値は比較しない。

## 失敗の扱い・実行上限・再開
- 再実行・seed 変更・hidden repair はしない。
- 4 worker、BLAS 1 スレッド、メモリ下限 1,500 MB。締め切り 2026-11-27T07:40+09:00 を過ぎたら新しい fit を出さない。B pilot の実測から本実験が締め切り内に終わらないと判断した場合、本実験は開始しない（時間・メモリ・技術的成立性だけで判断。A の結果には依存させない）。
- 再開：`python tools/research_audit/run_two_gaps_20261127.py execute B expfam/results/k_selection/two_gaps_20261127/expB/run --protocol expfam/results/k_selection/two_gaps_20261127/expB/protocol/protocol.json --resume --deadline <ISO>`。

## 成果物
`expfam/results/k_selection/two_gaps_20261127/expB/`。NPZ は git 管理外（SHA-256 を per_fit.csv に記録）。
