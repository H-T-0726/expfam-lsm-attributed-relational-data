# story diagnostics フェーズ 統合レポート

作成日: 2026-07-13
ブランチ: `research/story-diagnostics`
位置づけ: per-column family（`DualExpFamLSMPerColumn`, prototype）の検証結果（`research/per-column-validation`フェーズ）を受けて実施した3本の追加診断実験（Y sparsity stress test、MovieLens attribute diagnosis 2回）を、発表ストーリーとして統合したレポート。**complementary blocks実験には今回まだ着手していない**（設計メモのみ`story_diagnostics_next_plan_20260713.md`に保持）。

参照した一次資料:
- 前回フェーズ: `reports/per_column_family/single_vs_joint_per_column_report_20260711.md`, `per_column_math_audit_20260711.md`, `per_column_final_summary_20260711.md`, `expfam/results/per_column_family/single_vs_joint_agg.csv`, `movielens_mixed_x_agg.csv`
- 今回フェーズ: `reports/story_diagnostics/y_sparsity_stress_report_20260713_trials10.md`, `expfam/results/story_diagnostics/y_sparsity_stress_20260713_trials10_agg.csv`, `reports/story_diagnostics/movielens_attribute_diagnosis_report_20260713_trials4.md`, `expfam/results/story_diagnostics/movielens_attribute_diagnosis_20260713_trials4_agg.csv`

---

## 1. 前回まで

- 元論文・先行研究（Mikawa et al. 2024）は、属性データX＝Gaussian固定、関係データY＝Bernoulli固定の潜在構造モデルだった。
- dual-ExpFam LSMでは、XとYの分布をそれぞれ**指数型分布族（exponential family）**に一般化し、Gaussian・Bernoulli・Poissonなどを共通の枠組み（η=線形結合、A'(η)=予測平均、A''(η)=曲率）で切り替えられるようにした。数式監査（31/31 PASS）でこの拡張の実装正しさを確認済み。
- ただし、この時点でもX側はまだ「**X全体で1つのfamilyを選ぶ**」形のままだった。実データの属性行列には二値（ジャンル）・連続値（評価）・カウント（評価件数）が同時に混在するため、X全体を1つのfamilyで扱うのは不自然な場合がある、という課題が残っていた。

## 2. 今回の拡張（per-column family）と最初の結果

- 属性列ごとにfamily `g_l` を指定できる**per-column family（prototype、`DualExpFamLSMPerColumn`）**を実装し、人工データ実験・MovieLens pilotで検証した（`research/per-column-validation`フェーズ）。
- **最初の結果**:
  - 人工mixed-Xデータ（`single_vs_joint`実験）では、per_column_all（RMSE_Z 0.235±0.016, test_y_ll −2.047±0.012）はall_gaussian（0.234±0.018, −2.048±0.014）や単独属性モデル（single_gaussian 0.243等）とほぼ同等で、**改善幅は誤差範囲内と小さかった**。
  - MovieLens pilotでは、mixed_percolumn（genre19+mean_rating+year+ratings_countの22列per-column、test_y_ll −3.815）がgenre_only（−3.423、最良）より**悪化**した。
- この「改善幅が小さい」「MovieLensで悪化した」という2つの結果について、既存レポートは仮説（「Y情報が濃い設定だから」「X切片なし+カウント属性の曲率支配」）を述べていたが、**仮説自体を検証する追加実験は行われていなかった**。これが今回の story diagnostics フェーズの出発点。

## 3. 追加診断1: Y sparsity stress test

**目的**: 人工データで改善幅が小さかったのは「Y側の情報がすでに濃い」設定だったからではないか、という仮説を検証する。

**手法**: 同じ人工mixed-Xデータ生成（`generate()`）を使い、Yの学習観測率（train pair割合）を1.0→0.5→0.2→0.1と下げていき、per_column_all / single_gaussian / all_gaussian / y_onlyの4条件を比較（trials=10、160 fits、NaN・発散なし）。

**結果**（`y_sparsity_stress_20260713_trials10_agg.csv`より、RMSE_Z / test_y_ll、mean±std）:

| y_obs_rate | y_only (RMSE_Z) | all_gaussian | per_column_all |
|---:|---:|---:|---:|
| 1.0 | 0.308±0.014 | 0.233±0.019 | **0.221±0.020** |
| 0.1 | 1.176±0.086 | 0.769±0.133 | **0.343±0.026** |

- **Yが十分にある場合（y_obs_rate=1.0）は、モデル間の差は小さい**（RMSE_Z 0.221〜0.235）。既存single_vs_jointの結果と整合。
- **Yの観測率を下げると、y_onlyとall_gaussian（誤指定）は大きく悪化する**（RMSE_Zでそれぞれ約3.8倍、約3.3倍に悪化）。
- **per_column_allは、RMSE_Zとtest Y llの両方で、全y_obs_rateにわたり最良かつ最も安定（std最小）** だった。
- **これにより、Yが疎な場面ではXがZ推定を補助する可能性がある**ことが、trials=10（seed依存でない再現性込み）で示された。

## 4. 追加診断2: MovieLens attribute diagnosis

**目的**: MovieLens pilotでmixed_percolumnがgenre_onlyより悪化した原因を、属性・count処理ごとに切り分ける。

**手法**: genreに属性を1つずつ足す11条件を構築し、count属性の扱い（raw Poisson / log-Gaussian / zscore-Gaussian）を比較（trials4=4 fits/条件、44 fits、NaN・発散なし）。

**結果**（`movielens_attribute_diagnosis_20260713_trials4_agg.csv`より、test_y_ll、mean±std）:

| 条件 | test_y_ll | genre_onlyとの差 |
|---|---:|---:|
| genre_only | −3.417±0.017 | 0 |
| genre+count(raw,Poisson) | −3.792±0.057 | **−0.374** |
| genre+count(log,Gaussian) | −3.415±0.012 | +0.002 |
| genre+count(zscore,Gaussian) | −3.416±0.016 | +0.002 |
| rating_stats_only | −3.779±0.056 | −0.362 |
| mixed_percolumn(raw count) | −3.788±0.052 | −0.371 |
| mixed_percolumn(log count) | −3.414±0.014 | +0.003 |

- **raw countをPoissonとして入れると大きく悪化する**（genre_onlyから−0.374、std0.017/0.057を超える明確な差）。
- **mixed_percolumn_rawの悪化は、genre+count(raw,Poisson)の悪化（−0.374）とほぼ同じ大きさ・同じ方向**（mixed_percolumn_raw: −0.371）。genre・mean_rating・yearの追加自体はgenre_onlyとの差がほぼゼロであり、**22列構成の悪化はcount列1つの扱いでほぼ説明できる**。
- **countをlog変換またはz-scoreしてGaussianとして扱うと、悪化はstdの範囲内まで解消する**（+0.002、実質差なし）。
- **mixed_percolumn(log count)はgenre_onlyとほぼ同等**（差+0.003、std0.014の範囲内）。
- **したがって、MovieLensでの悪化は「属性ごとに分布を変える発想そのもの」が原因ではなく、「raw count＋Poisson＋切片なし＋スケール未調整」の組み合わせが主因である可能性が高い**（mixed_all_gaussianという全列Gaussian強制の誤指定モデルもgenre_onlyに近い−3.438であり、per-column family自体を悪者にする根拠はない）。

## 5. 注意点

- **MovieLensのmean_rating・ratings_countは、train/testのpair splitより前に、全`u.data`から計算されている（split非依存）**。Y（共評価カウント）も同じ`u.data`由来であり、構造的な相関がありうる（leakage risk）。今回はtrain-only化を行っていない。**したがって、本実験の結果は厳密な汎化性能の証拠ではなく、悪化要因を切り分けるための診断として扱う。**
- **mixed_percolumn(log count)がgenre_onlyを「明確に上回った」とは言わない**。差(+0.003)はstd(0.014)より小さく、「同等になった」という表現が適切。
- **人工データ実験（Y sparsity stress）も1つの生成設定（n=80, d=9, k*=2, gauss3+bern3+pois3, Poisson-Y）のみでの確認**であり、他の設定への一般化は慎重に扱う。

## 6. 結論

- **属性ごとに分布を変えるだけでは不十分**である。人工データ（Yが十分にある通常条件）でもMovieLens（元のraw count構成）でも、per-column family単体で明確な改善が示されたわけではない。
- **ただし、Yが疎な人工条件では有効性が見えた**（Y sparsity stress test、trials=10で再現）。Y側の情報が乏しいほど、Xを正しく統合したモデルが安定して良い結果を示す。
- **MovieLensでは、count属性の扱い（raw Poisson vs Gaussian変換）が悪化要因として明確に見えた**（attribute diagnosis、trials4で再現）。per-column family自体ではなく、切片なしモデルでの生カウント表現が問題だった可能性が高い。
- **次はX側切片、スケーリング、属性ブロック重みが必要**。この3点は今回の2実験どちらの診断結果からも共通して示唆されており、次フェーズの実装優先項目とする。

---

## 発表用の短いまとめ（そのまま使える）

> 前回の発表では、XとYを指数型分布族に拡張し、GaussianやBernoulli・Poissonを共通の枠組みで扱えるようにしました。しかしX側はまだ「属性全体で1つのfamily」を選ぶ形のままで、実データでは二値・連続値・カウントが混在するという課題がありました。
>
> そこで今回、属性列ごとに観測分布を指定できるper-column familyを実装しました。ただし最初の結果は、人工データでは改善幅が小さく、MovieLensの実データではむしろ悪化するというものでした。
>
> この結果を掘り下げるため、2つの追加診断実験を行いました。1つ目は、Yの情報量を意図的に減らす実験です。Yが十分にある通常条件ではモデル間の差は小さいのですが、Yを疎にしていくと、per_column_allは他のモデル（Xを使わないモデルや、familyを誤って指定したモデル）より安定して良い結果を示しました。つまり、Yの情報が乏しい場面でこそ、属性ごとに正しく分布を指定した統合が効いてくる可能性があります。
>
> 2つ目は、MovieLensでの悪化要因を属性ごとに切り分ける実験です。その結果、悪化の主因は「属性ごとに分布を変える」という発想そのものではなく、「評価件数という属性を、切片のないモデルで生の値のままPoissonとして扱ったこと」にあることが分かりました。この属性をlog変換やz-scoreでGaussianとして扱うよう変えるだけで、悪化はほぼ解消し、ジャンルのみを使うモデルと同程度の性能になりました。
>
> まとめると、属性ごとに分布を変えるだけでは十分ではありませんが、Yの情報が乏しい場面での有効性や、実データでの悪化の具体的な原因（count属性の扱い）は明らかになりました。次のステップとして、属性側にも切片を導入すること、カウント属性のスケーリング方法を整理すること、属性ブロックごとの重み付けを検討することが必要だと考えています。

なお、MovieLensのデータには、属性と関係データが同じ評価ログに由来するという構造上の注意点があり、今回の結果は悪化要因の切り分けとして扱っています。厳密な予測性能の証拠として一般化はしていません。
