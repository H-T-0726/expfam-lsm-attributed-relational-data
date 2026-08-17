# 修論・発表用研究ストーリー：過分散診断、strict評価、共有Z検証

作成日: 2026-07-10  
対象成果: `research/overdispersion-z-ablation` から `main` に取り込まれた研究成果

## 1. 研究の中心主張

属性情報付き関係データの潜在空間モデルでは、分布族を単に増やすことよりも、**潜在構造を考慮した診断、リークのない評価、仮定のablationを一体として行うこと**が重要である。

本研究は、(i) 周辺的なカウント過分散と条件付きのモデル誤指定を区別し、(ii) strict pair-mask評価で予測性能を測り、(iii) 共有潜在変数Zと属性Xの分布族仮定を検証する実験手順を提示する。NB2-Yとper-column familyは、この手順を検証するためのexperimental/prototype実装であり、stableモデルへの正式統合を主張しない。

## 2. 従来研究との差分

| 観点 | 従来 | 今回 |
|---|---|---|
| 過分散の判断 | 周辺var/meanを中心に解釈しうる | 潜在構造を入れた後の条件付きPearson過分散とPPCを併用 |
| 評価 | 学習pairを含みうる評価 | pair maskで学習pairとstrict held-out pairを分離 |
| Poisson誤指定 | 離散的な3×3 family mismatchが中心 | NB2-Y人工データで過分散強度に対する用量反応を評価 |
| 共有Z | 共有を前提にした比較が中心 | Y-only/X-onlyとのablationでデータ依存性を検証 |
| 属性Xのfamily | 全列共通family | 列別familyを指定できるprototypeと混在属性デモ |
| 既存結果 | 看板数値の出所追跡が不足 | 約41.5倍の根拠CSV・条件をread-only監査で特定 |

## 3. 今回追加された貢献

1. **条件付き過分散の診断手順**
   - MovieLensで、周辺var/meanと条件付きPearson過分散、plug-in PPCを分けて報告した。
2. **strict held-out評価基盤**
   - pair maskをE-step、M-step、尤度計算に適用し、held-out pairのYを変更しても学習が不変であることをテストした。
3. **NB2-Yのexperimental拡張と数式監査**
   - log linkとFisher情報量重みを用いるE-stepのexperimental実装を追加し、尤度、score、数値微分、Poisson極限、held-out尤度の比較可能性を監査した。
4. **誤指定の評価指標に関する証拠**
   - 人工NB2-Yで、Poisson誤指定の影響がheld-out尤度とw推定誤差には現れる一方、RMSE(Z)だけでは見えない場合を示した。
5. **共有Z仮定のablation**
   - MovieLensでproposed、Y-only、X-onlyをstrict評価で比較し、共有による利得がデータ依存であることを検証した。
6. **混在属性への拡張可能性**
   - per-column family prototypeにより、列ごとにGaussian/Bernoulli/Poissonを指定する最小実装と人工デモを用意した。
7. **既存ミスマッチ結果の再現可能性強化**
   - 約41.5倍の数値の根拠を `exp_scenario_C_exp4_mismatch.csv` のX=Gaussian/Y=Poisson条件として特定した。

## 4. 強く言ってよい主張

- 潜在構造モデルでは、**周辺var/meanだけでPoisson条件付きモデルの妥当性を判断することは不十分**である。MovieLensと統制人工データの双方で、周辺過分散と条件付き過分散が異なることを確認した。
- pair maskによりstrict held-out評価を実装でき、従来のfull-pair評価が楽観的になりうる量をMovieLensで定量化した。
- NB2-Yのexperimental実装について、基準尤度、`scipy.stats.nbinom.logpmf`、数値微分、Poisson極限、テストリークなしを確認した。
- 人工NB2-Yでは、条件付き過分散が強いとPoisson誤指定はheld-out尤度とw推定を悪化させる。
- 既存の約41.5倍悪化結果は、根拠CSVと条件を特定して再計算一致を確認した。

## 5. 弱めに言うべき主張

- MovieLensの周辺var/meanは大きいが、潜在構造を条件づけると残差過分散は小さい。したがって「MovieLensでPoissonが不適切」とは結論しない。
- NB2-Yは**experimentalなNB2拡張**であり、正準ExpFamの正式実装ではない。MovieLensでのNB改善は小さく、低いr_hatを持つ実データでの検証が必要である。
- 共有Zの利得はデータ依存である。MovieLensで弱かったことは、共有Z仮定を一般に否定する根拠ではない。
- per-column familyは人工混在属性で確認した**prototype**であり、実データでの広範な有効性や自動family選択を主張しない。
- RMSE(Z)の逆転に関する事後分散・過信の機構は解釈段階であり、posterior coverageでの検証が必要である。

## 6. 言ってはいけない主張

- 「周辺var/meanが大きいのでMovieLensではPoissonを棄却できる」
- 「NB2-YはMovieLensで大幅な性能改善を達成した」
- 「NB2-Yは正準指数型分布族のstableな正式実装である」
- 「共有Zは一般に不要、または誤りである」
- 「per-column familyは常に必要、または実用化済みである」
- 「潜在変数ZがXとYの因果原因である」
- 「約41.5倍の最悪条件ラベルはすべての実装・試行で不変である」

## 7. 修論の章構成案

1. **序論**: 属性情報付き関係データ、潜在空間モデル、研究課題（分布誤指定・評価リーク・共有表現）。
2. **基礎モデルと既存3×3検証**: Dual-ExpFam LSM、既存ミスマッチ実験、41.5倍結果の位置づけと監査。
3. **診断と評価の方法論**: 周辺対条件付き過分散、pair mask、strict held-out尤度、評価指標の設計。
4. **過分散下のPoisson誤指定**: 人工NB2-Yにおける誤指定の用量反応、NB2-Y experimental拡張と数式監査。
5. **MovieLens実データ評価**: 過分散診断、strict対full評価、NB比較の限定的な解釈。
6. **共有Z仮定の検証**: proposed/Y-only/X-only ablation、データ依存性、適用前検査という位置づけ。
7. **混在属性への拡張可能性**: per-column family prototype、人工デモ、限界。
8. **結論と今後の課題**: 診断→評価→ablationという手順、未解決事項、実データ検証の展望。

## 8. 各章で使う図・表・CSV・レポート

| 章 | 図・表 | CSV | 主なレポート |
|---|---|---|---|
| 2 | 3×3ミスマッチの倍率表 | `expfam/results/mismatch_audit/mismatch_audit_{summary,old05_conditions}.csv` | `reports/mismatch_audit/mismatch_audit_report_20260708.md` |
| 3 | pair maskの模式図、評価プロトコル表 | `expfam/results/overdispersion/movielens_strict_heldout_runinfo.csv` | `reports/overdispersion/pair_mask_design_20260708.md` |
| 4 | held-out ll、w誤差、RMSE(Z)の用量反応図 | `expfam/results/overdispersion/poisson_misspecification_{summary,agg}.csv` | `reports/overdispersion/poisson_misspecification_report_20260708.md`、`reports/overdispersion/nb_math_audit_20260708.md` |
| 5 | `figures/overdispersion/movielens_y_distribution.*`、`movielens_mean_variance.*`、`movielens_strict_heldout_comparison_k{3,5}.*` | `expfam/results/overdispersion/movielens_{overdispersion_diagnostics,ppc_summary,strict_heldout_summary}.csv` | `reports/overdispersion/movielens_overdispersion_diagnostics_20260708.md` |
| 6 | `figures/shared_z_ablation/movielens_shared_z_ablation.*` | `expfam/results/shared_z_ablation/movielens_shared_z_ablation_{summary,agg}.csv` | `reports/shared_z_ablation/shared_z_ablation_report_20260708.md` |
| 7 | per-column比較表（必要ならCSVから図化） | `expfam/results/per_column_family/per_column_demo_{summary,agg}.csv` | `reports/research_direction/per_column_family_design_20260708.md` |
| 8 | 貢献・限界・次実験の一覧表 | 全runinfo CSV | `reports/research_direction/overdispersion_shared_z_research_summary_20260708.md`、`pre_commit_full_audit_20260708.md` |

## 9. 先生に説明するための3分要約

本研究では、属性情報付き関係データの潜在空間モデルを、実データでどのように診断し評価すべきかを整理しました。従来は、カウント関係データで分散が平均より大きいとPoissonが不適切だと解釈しやすい問題がありました。しかしMovieLensでは、周辺の分散/平均比は大きくても、潜在構造を入れた条件付き診断では残差過分散は小さくなりました。つまり、周辺過分散だけで分布族を決めるのは危険です。

次に、学習に使った関係pairで評価してしまう楽観を避けるため、pair maskによるstrict held-out評価を実装しました。これにより、従来評価がどの程度楽観的かを定量化できます。人工NB2-Yデータでは、真に条件付き過分散がある場合、Poisson誤指定はheld-out尤度と関係係数推定を悪化させることを確認しました。そこでNB2-Yも実装しましたが、これはlog linkとFisher型近似を使うexperimentalな拡張であり、MovieLensで大幅改善したとは主張しません。

さらに、XとYで潜在変数Zを共有する仮定をY-only/X-onlyと比較しました。共有の利得はデータ依存であり、MovieLensの結果だけで共有Zを一般に否定するものではありません。属性Xについても、列別familyを扱うprototypeを用意し、混在属性では全列共通familyが不安定になりうることを人工データで確認しました。

結論として、提案は単一の新モデルを押し出すことではなく、潜在構造を考慮した診断、strict評価、ablationを組み合わせ、モデル仮定をデータごとに検証する研究手順です。

## 10. 次に追加実験するなら何を優先すべきか

1. **Coraのstrict shared-Z ablation**
   - 共有Zが有効になりうる、属性と関係の意味的重なりが強い実データで検証する。共有Zの結論をデータ依存として支える最優先実験。
2. **posterior coverage実験**
   - Poisson/NBでのRMSE(Z)逆転の解釈を、推定区間の較正で直接検証する。
3. **低r_hatの実カウントデータでのNB検証**
   - MovieLensで改善が小さい理由を補い、NB2-Yが実データで有用となる条件を調べる。
4. **k×family同時選択実験**
   - kを増やして異質性を吸収することと、NBで残差過分散を表現することのトレードオフを体系化する。
5. **observed-Hessianとの比較とmask-aware BICの整理**
   - 実装上・モデル選択上の未解決事項を縮小する。ただし上記の実証優先タスクの後に扱う。
