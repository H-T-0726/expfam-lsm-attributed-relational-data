# MovieLens attribute diagnosis（trials4 拡張版）— 報告

作成日: 2026-07-13
ブランチ: `research/story-diagnostics`
スクリプト: `tools/research_audit/run_movielens_attribute_diagnosis.py`（`SPLIT_TRIALS=[0,1]`, `MODEL_TRIALS=[0,1]`, 4 fits/条件）
結果: `expfam/results/story_diagnostics/movielens_attribute_diagnosis_20260713_trials4{.,_agg.,_runinfo.}csv`
図: `figures/story_diagnostics/movielens_{attribute_test_y_ll,count_treatment_comparison}_trials4.png`

**位置づけ**: 2026-07-13のsmoke版（1 fit/条件、`movielens_attribute_diagnosis_report_20260713_smoke.md`）で見えた傾向が seed/split 依存でないかを確認する拡張実験。条件は smoke と同一で、fit数のみ1→4（SPLIT_TRIALS×MODEL_TRIALSの2×2）に拡張した。既存の smoke 出力・図は上書きしていない（新ファイル名 `_trials4` を使用）。

## 見方

**test Y log-likelihood / pair（test_y_ll）は大きいほど良い。** ただし常に負の値になる。図のバーは0から負の値へ伸びるため、バーが短い（0に近い）ほど良い。

## 1. リーク注意（必須・再掲）

`mean_rating`と`ratings_count`は、train/testのpair splitより前に、`build_attributes()`内で全`u.data`・既存メタデータから計算されている（split非依存）。Y（共評価カウント）も同じ`u.data`由来であり、構造的な相関がありうる（leakage risk）。今回もtrain-only化はしていない。**したがって本実験の目的は「悪化要因の診断」であり、「実データでの最終性能の主張」ではない**（厳密な汎化性能の証拠として扱わない）。

## 2. 実行結果サマリ

- 11条件 × 2 split × 2 model seed = **44 fits**。CSV行数: 生データ45行(header+44)、agg12行、runinfo12行。
- **NaN・発散は44 fit中0件**。

## 3. 結果表（mean ± std, n_trials=4、`_agg.csv`より）

| 条件 | test_y_ll (mean±std) | test_y_rmse | test_hc_auc | genre_onlyとの差 |
|---|---:|---:|---:|---:|
| y_only | −3.452 ± 0.027 | 7.49 | 0.969 | −0.034 |
| **genre_only** | **−3.417 ± 0.017** | 7.29 | 0.970 | 0.000 |
| genre+year | −3.419 ± 0.008 | 7.29 | 0.971 | −0.002 |
| genre+avg_rating | −3.419 ± 0.009 | 7.29 | 0.972 | −0.002 |
| genre+count(raw,Poisson) | −3.792 ± 0.057 | 9.12 | 0.949 | **−0.374** |
| genre+count(log,Gaussian) | −3.415 ± 0.012 | 7.26 | 0.971 | **+0.002** |
| genre+count(zscore,Gaussian) | −3.416 ± 0.016 | 7.27 | 0.971 | +0.002 |
| rating_stats_only | −3.779 ± 0.056 | 9.04 | 0.950 | **−0.362** |
| mixed_percolumn(raw count) | −3.788 ± 0.052 | 9.08 | 0.949 | **−0.371** |
| mixed_percolumn(log count) | **−3.414 ± 0.014** | 7.30 | 0.971 | **+0.003** |
| mixed_all_gaussian（誤指定） | −3.438 ± 0.018 | 7.41 | 0.971 | −0.021 |

## 4. 指定された比較

- **genre_only vs genre+count(raw,Poisson)**: −3.417±0.017 vs −3.792±0.057（**差 −0.374、std(0.017/0.057)を大きく超える明確な悪化**）。smoke版の−0.314とほぼ同じ方向・規模で再現。
- **genre_only vs genre+count(log,Gaussian)**: −3.417±0.017 vs −3.415±0.012（**差 +0.002、std範囲内で実質差なし**）。悪化は解消。
- **genre_only vs genre+count(zscore,Gaussian)**: −3.417±0.017 vs −3.416±0.016（**差 +0.002、同様に実質差なし**）。log変換の有無に関わらずGaussian z-score化で悪化は解消。
- **mixed_percolumn(raw count) vs mixed_percolumn(log count)**: −3.788±0.052 vs −3.414±0.014（**差 +0.374、std差を大きく超える改善**）。count列の処理を変えるだけで劇的に改善することが4 fitsでも再現。
- **rating_stats_only vs mixed_percolumn(raw count)**: −3.779±0.056 vs −3.788±0.052（**差 −0.009、std範囲内でほぼ同水準**）。genre19列を追加してもraw countによる悪化はほぼ変わらず、genre自体は悪化の原因ではないことを再確認。
- **genre_only vs mixed_percolumn(log count)**: −3.417±0.017 vs −3.414±0.014（**差 +0.003、std範囲内で実質同等**）。count処理を直せば、22列のmixed_percolumnはgenre_onlyと同等になる（「明確に上回る」とまでは言えない）。

## 5. smoke版との再現性確認

| 傾向 | smoke（1 fit） | trials4（4 fits） | 再現性 |
|---|---|---|---|
| raw count Poissonで大きく悪化 | −0.314 | −0.374 | 確認（同方向・同程度） |
| log/zscore Gaussianで悪化解消 | −0.003 | +0.002 | 確認（std範囲内で実質ゼロ） |
| mixed_percolumn_rawの悪化 ≈ rating_stats_only | −0.319 vs −0.309 | −0.371 vs −0.362 | 確認（両者ほぼ一致） |
| mixed_percolumn_log_countがgenre_only同等以上 | +0.017 | +0.003 | 確認（ただしsmokeより差は縮小し、std(0.014)内） |

**4 fitsに拡張しても、smoke版で見えた4つの傾向はすべて同じ方向・同程度の大きさで再現された。** ただし「mixed_percolumn_log_countがgenre_onlyを上回る」という点は、smoke(+0.017)からtrials4(+0.003)で差が縮小しており、std(0.014)と同程度の大きさになったため、**「上回る」ではなく「同等になる」と表現するのが適切**。

## 6. 発表で使える主張

**言ってよい主張:**
- 「genreにratings_countを生値Poissonで1列足すだけで、22列mixed_percolumnとほぼ同じ大きさの悪化（4 fits平均で−0.37前後）が再現された。この傾向はsmoke（1 fit）とtrials4（4 fits）で一貫している」
- 「count属性をGaussian・z-score（log変換の有無に関わらず）で扱うと、悪化はstdの範囲内まで解消される」
- 「count処理を直したmixed_percolumn（log count）は、genre_onlyと同等の性能になる（明確に上回るとまでは言えない）」
- 「mean_rating・ratings_countはtrain/test split前の全ログ由来であり、leakage riskがある。本結果は悪化要因の切り分け診断であり、実データでの最終性能の主張ではない」

**言いすぎな主張（避ける）:**
- 「mixed_percolumn_log_countがgenre_onlyより明確に優れている」（差+0.003はstd 0.014より小さく、誤差範囲内）
- 「per-column familyがMovieLensで有効であることを示した」（mixed_all_gaussianも−3.438とgenre_onlyに近く、per-column family正指定であることの寄与はこの実験だけでは分離できていない。改善の主因はcount列の変換であり、familyの列ごと指定そのものではない）
- 「リークの影響がないことを確認した」（train-only化はしておらず、リークの有無・大きさ自体は未検証のまま）
- 「4 fitsで十分な統計的検証が完了した」（seed数はまだ少なく、より多くのsplit/model seedでの確認が望ましい）

## 7. 次のステップ

- 傾向は十分安定して見えるため、次はcomplementary blocks実験、またはX側切片の導入（今後の課題の優先1位）への着手を検討する。
- block-wise gradient / precision診断は今回も見送った。
- `EXPERIMENT_REGISTRY.md`への追記は、正式採用を決定してから行う（今回もまだ実施していない）。
