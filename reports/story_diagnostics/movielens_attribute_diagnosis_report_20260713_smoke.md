# MovieLens attribute diagnosis（軽量版・smoke）— 報告

作成日: 2026-07-13
ブランチ: `research/story-diagnostics`
スクリプト: `tools/research_audit/run_movielens_attribute_diagnosis.py`（`SPLIT_TRIALS=[0]`, `MODEL_TRIALS=[0]`, 1 fit/条件）
結果: `expfam/results/story_diagnostics/movielens_attribute_diagnosis_20260713_smoke{.,_agg.,_runinfo.}csv`
図: `figures/story_diagnostics/movielens_{attribute_test_y_ll,count_treatment_comparison}_smoke.png`

**位置づけ: これは1 fit/条件の smoke run である。** 傾向確認が目的であり、正式な結論・汎化性能の主張ではない。

## 見方

- **test Y log-likelihood / pair（test_y_ll）は大きいほど良い**。ただし常に負の値になる。図のバーは0から負の値へ伸びるため、**バーが短い（0に近い）ほど良い**。
- **test Y RMSE は小さいほど良い**。
- AUC/AP は「Y（共評価カウント）が中央値(42)以上かどうか」の二値分類として計算（生の閾値1では全ペアがpositiveになりAUC計算不能だったため、中央値を採用）。

## 1. 目的

既存 pilot（`run_movielens_mixed_x_percolumn.py`、`research/per-column-validation`フェーズ）では `mixed_percolumn`（genre19 + mean_rating + year + ratings_count の22列 per-column）が `genre_only` より悪化した（test_y_ll: −3.815 < −3.423、`movielens_mixed_x_agg.csv`）。本実験は、genreに属性を1つずつ足しながら、**どの属性・どのcount処理が悪化の原因か**を切り分ける診断実験である。

## 2. リーク確認（必須）

`build_attributes()`を確認した結果:
- **`mean_rating`と`ratings_count`は、train/testのpair split より前に、全`u.data`・既存メタデータから計算されている（split非依存）**。Y（共評価カウント行列）も同じ`u.data`由来であり、構造的な相関がありうる（leakage risk）。
- **`year`のみタイトル文字列由来でリークなし**。
- 今回はtrain-only化を実装していない（ユーザ指示どおり診断実験として許容）。**したがって本実験の結果は厳密な汎化性能の証拠ではなく、どの属性・どの処理が悪化に寄与するかを切り分けるための診断結果である。**

## 3. 実行結果サマリ

- 11条件 × 1 split × 1 model seed = **11 fits**。CSV行数: 生データ12行(header+11)、agg12行、runinfo12行。
- **NaN・発散は11 fit中0件**。

## 4. 結果表（`_agg.csv`より、test_y_ll_diff_vs_genre_only列を含む）

| 条件 | test_y_ll | test_y_rmse | test_hc_auc | genre_onlyとの差 |
|---|---:|---:|---:|---:|
| y_only | −3.478 | 7.65 | 0.969 | −0.060 |
| **genre_only** | **−3.418** | 7.29 | 0.972 | 0.000 |
| genre + year | −3.423 | 7.33 | 0.973 | −0.005 |
| genre + avg_rating | −3.420 | 7.31 | 0.973 | −0.002 |
| genre + count(raw, Poisson) | −3.733 | 8.90 | 0.953 | **−0.314** |
| genre + count(log, Gaussian) | −3.422 | 7.30 | 0.972 | −0.003 |
| genre + count(zscore, Gaussian) | −3.421 | 7.28 | 0.972 | −0.003 |
| rating_stats_only | −3.727 | 8.80 | 0.956 | −0.309 |
| mixed_percolumn（raw count） | −3.737 | 8.85 | 0.954 | **−0.319** |
| mixed_percolumn（log count） | **−3.401** | 7.31 | 0.972 | **+0.017** |
| mixed_all_gaussian（誤指定） | −3.423 | 7.35 | 0.973 | −0.005 |

## 5. 指定された比較

- **genre_only vs mixed_percolumn_raw**: −3.418 vs −3.737（**差 −0.319、明確に悪化**）。既存pilotの悪化を再現。
- **genre_only vs genre+rating_count_raw_poisson**: −3.418 vs −3.733（**差 −0.314**）。genreに`ratings_count`を生値Poissonで1列足しただけで、mixed_percolumn全体の悪化（−0.319）とほぼ同じ大きさの悪化が生じている。**→ 悪化の主因はratings_countの生値Poisson処理であり、他の属性（year, avg_rating）や属性数の多さではない。**
- **genre_only vs genre+log_count_gaussian**: −3.418 vs −3.422（**差 −0.003、ほぼ差なし**）。
- **genre_only vs genre+zscore_count_gaussian**: −3.418 vs −3.421（**差 −0.003、ほぼ差なし**）。log変換の有無に関わらず、countをGaussianとしてz-score化するだけで悪化はほぼ消える。
- **rating_stats_only vs mixed_percolumn_raw**: −3.727 vs −3.737（**差 −0.010、ほぼ同程度に悪い**）。genre19列を追加しても、rating_stats_only（3列のみ、count生値Poisson込み）の時点で既に大部分の悪化が生じており、genre自体が悪化を引き起こしているわけではない。
- **mixed_percolumn_raw vs mixed_percolumn_log_count**: −3.737 vs −3.401（**差 +0.336、劇的に改善**）。22列構成はそのままに、count列だけをPoisson生値からlog-Gaussian z-scoreへ変更するだけで、genre_onlyを上回る結果になった（この1 fitでは最良）。

## 6. 暫定的な結論

- **count属性を入れると悪化するか**: 条件付きでYes。**「count属性を入れること」自体ではなく、「count属性をX側切片なしのPoisson生値として入れること」が悪化の主因**である。genre+countの2ブロックだけでも、22列の完全なmixed_percolumnとほぼ同じ大きさの悪化が再現された。
- **log_count / zscore_countで改善するか**: Yes。どちらの変換でもgenre_only水準まで回復し、22列構成（mixed_percolumn_log_count）ではむしろgenre_onlyよりわずかに良くなった（この1 fitでは）。log変換の有無自体は今回の差にはほとんど寄与せず、**「Gaussian・z-score化して切片なしモデルのスケール制約に合わせること」が本質**と考えられる。
- year・avg_rating単体の追加はgenre_onlyとほぼ差がない（−0.005, −0.002）。悪化の主犯はcountの扱いに限定される。
- X block reconstruction RMSEでも、count_rawブロックのRMSEが27〜28（生スケール、平均154に対し約18%）と他ブロック（0.29〜0.68程度、こちらは無次元・z-score/0-1スケール）とは全く異なるスケール感であることが確認でき、切片なしモデルでの生カウント表現がスケール上不利であることと整合する。

## 7. 発表で使える主張

**言ってよい主張:**
- 「genreにratings_countを生値Poissonで1列足すだけで、22列mixed_percolumnとほぼ同じ大きさの悪化が smoke run（1 fit）で再現された」
- 「count属性をGaussian・z-score（log変換の有無に関わらず）で扱うと、悪化はほぼ解消される」
- 「mean_rating・yearの追加自体はgenre_onlyとの差がほとんどない」
- 「mean_rating・ratings_countはtrain/test split前の全ログ由来であり、leakage riskがある。本結果は厳密な汎化性能の証拠ではなく悪化要因の切り分け診断である」

**言いすぎな主張（避ける）:**
- 「count属性のGaussian処理がMovieLensで有効であることを証明した」（1 fitのsmoke runであり、seed・split依存性は未確認）
- 「per-column familyがMovieLensで有効であることを示した」（mixed_percolumn_log_countが良かったのはcount処理を変えた効果であり、per-column family自体の効果ではない。mixed_all_gaussian（誤指定、全列Gaussian強制）も同水準の−3.423であり、per-column family正指定であることの寄与は今回の結果からは分離できていない）
- 「リークの影響がないことを確認した」（train-only化はしておらず、リークの有無・大きさ自体は未検証）

## 8. 次のステップ

- SPLIT_TRIALS/MODEL_TRIALSを増やし（例: 各2〜3）、この傾向がseed・split依存でないか確認する。
- block-wise gradient / precision診断は今回見送った（重ければ後回しの指示どおり）。次段階で余裕があれば追加する。
- `EXPERIMENT_REGISTRY.md`への追記は、結果を確認し採用を決定してから行う（今回もまだ実施していない）。
