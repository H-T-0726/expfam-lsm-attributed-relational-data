# 実験一覧表

作成日：2026-05-31

## 共通条件
- n=150（基本）, d=15, k*=3, 10試行
- L=5 MCサンプル, 8 EM反復（expfam/src/exp_scenario_lib.py L.42-43）
- 評価指標：RMSE(Z)（Procrustes回転適用）を主指標
- 人工データのみ

---

| 実験名 | 目的 | 分布条件（真） | 比較対象 | 評価指標 | 結果CSV | 図ファイル | 主な結果 | 発表で使える結論 | 注意点 |
|-------|------|--------------|---------|---------|--------|---------|--------|----------------|-------|
| Exp1-A BIC次元選択 | BICがk*=3を選ぶか | Scen.A: X=Pois, Y=Bern | k=1,2,3,4,5,6 | RMSE(Z), BIC | exp_scenario_A_exp1_k.csv | fig_scenario_A_exp1_k.pdf | k=3でRMSE(Z)=0.278最小、BIC最小 | BICによる次元自動選択が成功 | 10試行のみ |
| Exp1-B BIC次元選択 | 同上 | Scen.B: X=Gauss, Y=Pois | 同上 | 同上 | exp_scenario_B_exp1_k.csv | fig_scenario_B_exp1_k.pdf | k=3でRMSE(Z)=0.182最小 | 同上 | 同上 |
| Exp1-C BIC次元選択 | 同上 | Scen.C: X=Bern, Y=Gauss | 同上 | 同上 | exp_scenario_C_exp1_k.csv | fig_scenario_C_exp1_k.pdf | k=3でRMSE(Z)=0.028最小（BIC負値） | 同上 | BICが負値（Gauss Y正規化定数の影響） |
| Exp2-A n-sweep | nによる漸近一致性 | Scen.A | n=50,100,150,200,250,300 | RMSE(Z)+6指標 | exp_scenario_A_exp2_n.csv | fig_scenario_A_exp2_n.pdf | 0.406→0.208（-49%） | n増加で精度向上 | 単調減少（変動なし） |
| Exp2-B n-sweep | 同上 | Scen.B | 同上 | 同上 | exp_scenario_B_exp2_n.csv | fig_scenario_B_exp2_n.pdf | 0.190→0.131（-31%） | 同上 | n=50→100で一時変動あり |
| Exp2-C n-sweep | 同上 | Scen.C | 同上 | 同上 | exp_scenario_C_exp2_n.csv | fig_scenario_C_exp2_n.pdf | 0.053→0.020（-62%） | 同上 | Scen.C全体がScen.Aより小さい（Y=Gauss支配） |
| Exp3-A d-sweep | d変化の影響 | Scen.A | d=5,10,15,20,25,30 | RMSE(Z)+6指標 | exp_scenario_A_exp3_d.csv | fig_scenario_A_exp3_d.pdf | 0.322→0.236（-27%） | d増加で精度向上 | - |
| Exp3-B d-sweep | 同上 | Scen.B | 同上 | 同上 | exp_scenario_B_exp3_d.csv | fig_scenario_B_exp3_d.pdf | 0.200→0.146（-27%） | 同上 | - |
| Exp3-C d-sweep | 同上 | Scen.C | 同上 | 同上 | exp_scenario_C_exp3_d.csv | fig_scenario_C_exp3_d.pdf | 0.029→0.029（ほぼ平坦） | d変化の影響が小さい場合がある | Y=Gauss支配のため。原稿記載済み |
| Exp4-A 分布ミスマッチ | 誤指定の影響 | Scen.A | 3×3全分布組み合わせ+ablation | RMSE(Z) | exp_scenario_A_exp4_mismatch.csv | fig_scenario_A_exp4_heatmap.pdf | 最大3.41×（Bern-Bern条件） | 誤指定が精度を最大3倍悪化 | 最大値は1条件、試行間変動大 |
| Exp4-B 分布ミスマッチ | 同上 | Scen.B | 同上 | 同上 | exp_scenario_B_exp4_mismatch.csv | fig_scenario_B_exp4_heatmap.pdf | 最大約7.35×（Gauss-Bern条件） | 同上（最大7倍） | Scen.Bの7.35×はCSVによって数値が異なる可能性 |
| Exp4-C 分布ミスマッチ | 同上 | Scen.C | 同上 | 同上 | exp_scenario_C_exp4_mismatch.csv | fig_scenario_C_exp4_heatmap.pdf | 最大~38×〜41.5×（条件により異なる） | 同上（最大38〜41倍） | 41.5×の根拠CSV要確認（CLAUDE.md参照） |
| Control比較 | 先行研究と同等か | X=Gauss, Y=Bern（従来手法相当） | Mikawa 2024 Python再現 | RMSE(Z/F/Y/X) | reproduction/results/comparison/comparison_main_table.csv | なし | RMSE(Z)差 < 0.001 | 先行研究と同等の精度 | 5試行のみ |
| ミスマッチ固定版 | 別実験での誤指定検証 | Scen.A/B/C | 3×3組み合わせ | RMSE(Z), 悪化倍率 | distribution_mismatch_fixed/mismatch_fixed_summary.csv | distribution_mismatch_fixed/heatmap_ratio_scenario_*.pdf | Scen.C最大38.97×（Pois-Bern条件） | 参考値（別実験） | 上記Exp4と試行設定が異なる可能性あり |

---

## 注：発表の提出図に対応する実験

| 提出図 | 対応実験 | 備考 |
|-------|---------|------|
| fig1a_n_sweep_color.png | Exp2-A/B/C（n=50基準正規化） | figures/に提出用として作成済み |
| fig1b_misspecification_color.png | Exp4-A/B/C（悪化倍率バー） | X-only/Y-only/Fixed条件のみ抽出 |
