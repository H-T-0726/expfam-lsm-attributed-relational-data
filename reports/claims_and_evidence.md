# 主張と根拠の対応表

作成日：2026-05-31

| 主張 | 根拠 | ファイルパス | 図表 | 強さ | 注意点 |
|-----|------|------------|------|------|-------|
| 提案手法は指数型分布族への拡張を実現した | コードの実装 | expfam/src/model_dual_expfam.py | - | 強い | Categorical は未実装 |
| BIC が全3シナリオで k*=3 を正確に選択した | Exp1 CSV 確認済み | exp_scenario_{A/B/C}_exp1_k.csv | fig_scenario_{A/B/C}_exp1_k.pdf | 強い | 各シナリオ10試行のみ |
| n 増加に伴い全シナリオで RMSE(Z) が改善した | Exp2 CSV 確認済み | exp_scenario_{A/B/C}_exp2_n.csv | figures/fig1a_n_sweep_color.png | 強い | Scen.B に n=100 付近で一時変動あり |
| n=50→300 で RMSE(Z) が 49% 削減（Scen.A） | CSV計算済み | exp_scenario_A_exp2_n.csv | 同上 | 強い | Scen.Bは31%、Cは62% |
| 先行研究と同条件で RMSE(Z) 差 < 0.001 | 比較 CSV 確認済み | reproduction/results/comparison/comparison_main_table.csv | - | 強い | 5試行のみ |
| 誤った分布族指定が RMSE(Z) を最大 3.41 倍悪化させる（Scen.A） | Exp4 CSV 確認済み（Bern-Bern条件 0.949 vs oracle 0.278） | exp_scenario_A_exp4_mismatch.csv | figures/fig1b_misspecification_color.png | 強い | 1条件・10試行の平均 |
| 誤指定の影響は最大 7.35 倍（Scen.B） | 原稿記載。CSV での直接確認は条件特定が必要 | exp_scenario_B_exp4_mismatch.csv | 同上 | 中程度 | mismatch_fixed では7.80×、multi-scenarioでは~6.3〜7.3×でばらつきあり |
| 誤指定の影響は最大 41.5 倍（Scen.C） | 原稿記載のみ。対応CSVが特定できていない | CLAUDE.md L.83、exp_scenario_C_exp4_mismatch.csv要確認 | 図中に対応バーなし | 弱い | mismatch_fixed では38.97×（異なる実験）。根拠CSV未特定 |
| 図1(b)の最大値は 23.6×（灰色バー） | CLAUDE.md に記載あり | CLAUDE.md、figures/figure_color_split_report.md | figures/fig1b_misspecification_color.png | 強い | Scen.C の Fixed Gauss-X/Bern-Y 条件 |
| X の誤指定は X を使わないより悪い（Scen.A） | Exp4 ablation（fix_x条件）vs X誤指定条件の比較 | exp_scenario_A_exp4_mismatch.csv（Y-only条件 0.348 vs Gauss-X条件 0.714） | figures/fig1b_misspecification_color.png | 強い | Scen.Aのみ確認 |
| d 増加で X の情報量が増し RMSE(Z) が改善する（Scen.A/B） | Exp3 CSV 参照（GEMINI_REPORT_MULTI_SCENARIO.md） | expfam/results/GEMINI_REPORT_MULTI_SCENARIO.md | fig_scenario_{A/B}_exp3_d.pdf | 中程度 | Scen.C は平坦。CSV 直接未精査 |
| Scen.C の Y=Gaussian が推定を支配している | Exp4 ablation（No X ≈ 提案手法） | expfam/results/GEMINI_REPORT_MULTI_SCENARIO.md（No X ablation 1.00×） | - | 中程度 | 解釈であり、理論的証明なし |
| Categorical 分布は未実装 | コード確認 | expfam/src/model_dual_expfam.py L.61（VALID_FAMILIES） | - | 強い | - |
| 精度行列 Y 側 Term3 に 0.5 係数が残っている | CLAUDE.md・コード | CLAUDE.md、model_dual_expfam.py L.200 | - | 強い | 結果への影響は未定量化 |
| 実データへの適用は未実施（本格的なもの） | ファイル検索 | run_wine_dual.py が存在するが詳細未確認 | - | 強い | wine実験の結果CSVはあるが詳細未評価 |
