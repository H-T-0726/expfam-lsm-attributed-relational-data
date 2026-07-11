# per-column family 検証フェーズ 最終サマリ

作成日: 2026-07-11
ブランチ: `research/per-column-validation`（main 54513a3 から分岐、コミットなし）
位置づけ: per-column family は **prototype**。本フェーズはその価値と限界を
ゼミで説明できるレベルまで検証するもの。正式手法としての主張はしない。

## 中心の問い

> 「ジャンルだけを Bernoulli、平均評価だけを Gaussian、評価件数だけを Poisson と
> して、別々にモデルを回せばよいのでは？ なぜ per-column family で複数属性を
> 同時に入れる必要があるのか？」

## 一言の答え

**別々に回すのは「各属性の単独効果を見る ablation」として有用。
per-column family は、異なる型の複数属性を同時に使って 1 つの共通 Z を
推定するための prototype であり、人工 mixed-X では単独属性・全列共通強制の
いずれとも同等以上だった。ただし属性を増やせば必ず良いわけではなく、
実データ（MovieLens pilot）ではスケールの大きいカウント属性が Z 推定を
支配して悪化する、という prototype の限界も確認された。**

## 実施内容と成果物

| 項目 | スクリプト | 結果 |
|---|---|---|
| 数式監査 | `tools/research_audit/audit_per_column_math.py` | `per_column_math_audit_summary.csv`（31/31 PASS）|
| 単独 vs 同時統合（9条件×3seed） | `run_per_column_single_vs_joint.py` | `single_vs_joint_{summary,agg,runinfo}.csv` |
| 属性追加 ablation（5条件×3seed） | `run_per_column_attribute_ablation.py` | `attribute_ablation_*.csv` |
| ノイズ属性チェック（6条件×3seed） | `run_per_column_noise_check.py` | `noise_check_*.csv` |
| MovieLens mixed-X pilot（6条件×4fit） | `run_movielens_mixed_x_percolumn.py` | `movielens_mixed_x_*.csv` |
| 図 | `plot_per_column_figures.py` | `figures/per_column_family/`（4 組）|

詳細レポート:
`per_column_math_audit_20260711.md` / `single_vs_joint_per_column_report_20260711.md` /
`per_column_ablation_report_20260711.md` / 本ファイル。
全 84 fits と数式監査 31 項目で NaN・発散リトライ・監査 FAIL なし。既存コード・CSV・図の変更なし
（EXPERIMENT_REGISTRY.md への行追記のみ）。

## 12 の問いへの回答

### 1. 数式実装は妥当か → **妥当（監査範囲で）**

31 チェック全 PASS。列ごと family の対数尤度和 `log p(x_i|z_i) = Σ_l log p_{g_l}(x_il|η_il)`、
E-step 勾配 `Σ_l w_l (x_il − A′_l) f_l`・precision `Σ_l w_l A″_l f_l f_l^T` が、
独立実装の数値微分（Y 3 family × mask あり/なし）・列ごと素朴計算・scipy と一致。
全列同一 family では既存スカラーモデルと**厳密一致（差 0.0）**。
Gaussian は mean link（=canonical）、σ_l² は Gaussian 列のみ推定、**X に切片なし**。
未監査: M-step Adam の収束性、EM 全体の統計的性質。

### 2. 別々に属性を使って回すだけで十分か → **不十分（ただし ablation としては有用）**

人工 mixed-X で単独属性は per-column 同時統合に一貫して劣る
（RMSE_Z: joint 0.235 vs 単独 0.243〜0.321、test Y ll: −2.047 vs −2.050〜−2.078）。
さらに (a) どの属性が「当たり」かは事前に不明（Gaussian 単独は僅差、Bernoulli 単独は
y_only とほぼ同じ）、(b) 別々に回すと条件ごとに別の Z が得られ、
「1 つの共通潜在空間」という LSM の目的は果たせない。

### 3. per-column 同時統合は単独属性と比べて何を改善するか

- Z 回復と held-out Y 予測が全単独条件と同等以上（改善幅は本設定で 3〜28%）。
- 3 ブロックすべての X を各分布の自然なスケールで再構成できる唯一の条件
  （単独モデルの再構成性能を犠牲にしない: gauss 0.297/bern 0.450/pois 1.394）。
- 1 つの Z で属性と関係データを同時に説明する解釈が可能になる。

### 4. 全列共通 family と比べて何を改善するか

壊れ方が 3 通り観察された:
- all_bernoulli（生値強制）: 1/3 trial で崩壊（RMSE_Z 1.75、test ll −106）。
- all_poisson: Gaussian ブロック再構成 5 倍悪化、RMSE_Z 劣化。
- all_gaussian: **Z・Y ではほぼ無害**（quasi-likelihood 的頑健性、既存知見と整合）
  だが、非 Gaussian ブロックの X 再構成が 1.5〜1.8 倍悪化し、確率・カウントの
  定義域も守られない。
- 加えて、誤指定 family の「尤度」は確率モデルとして無効なので
  **BIC 等のモデル比較が無意味になる**（all_bernoulli の BIC が最小に見える例を確認）。
  各列の family 指定が妥当である場合、per-column 化は混在属性を列ごとの
  データ型に対応した尤度で扱えるため、モデル比較を意味あるものにできる。

### 5. 複数属性を入れると Z 推定は安定するのか → **情報のある属性なら改善、保証はない**

ablation で Y-only → +Gauss で RMSE_Z −22%（3/3 trial で方向一致）。
ただし情報の薄い Bernoulli ブロック追加は無効果。「安定する」は属性の情報量次第。

### 6. Y の strict held-out 予測は改善するのか → **人工では小幅改善、実データでは悪化もある**

人工: joint が最良（−2.047 vs y_only −2.079、+0.03/pair 程度の小幅）。
MovieLens pilot: genre_only −3.423 に対し mixed_percolumn は **−3.815 に悪化**（問 10 参照）。

### 7. 属性を増やせば必ず良いのか → **良くない（2 実験で確認）**

情報の薄い属性は足しても変わらず（+Bern: 0.295→0.296）、
ノイズ属性は最良で横ばい・seed によっては +13% 悪化。

### 8. ノイズ属性を入れると悪化するのか → **平均では微悪化〜横ばい、seed 依存で明確な悪化**

family 正指定でも情報がなければ価値ゼロ。Gaussian ノイズが最も悪影響が出やすい
（1/σ̂² 重みで勾配寄与が大きいため。ブロック重み診断と整合）。
bern/pois ノイズはほぼ無害。用量反応は 3 seeds では確定できず。

### 9. MovieLens でこの方向性を試す価値はあるのか → **価値はあるが、前処理・モデル拡張が先**

pilot（n=100、genre19 Bern + 平均評価/公開年 z-score Gauss + 評価件数 Pois、
strict held-out）の結果:

| 条件 | test Y ll (/pair) | test RMSE | Spearman |
|---|---|---|---|
| genre_only | **−3.423** | 7.41 | 0.936 |
| y_only | −3.454 | 7.60 | 0.932 |
| mixed_all_gaussian（誤指定比較用） | −3.455 | 7.58 | 0.933 |
| mixed_percolumn | −3.815 | 9.34 | 0.894 |
| rating_stats_only | −3.816 | 9.36 | 0.894 |
| mixed_all_bernoulli（誤指定比較用） | −4.169 | 10.96 | 0.908 |

**mixed_percolumn ≈ rating_stats_only** = 評価統計ブロックが genre 19 列を
完全に支配した。機構はモデル仕様から説明できる: X に切片がないため
平均 154 の評価件数（Poisson）は η≈5 を F と Z で作る必要があり、
曲率 A″ の比較では、その 1 列の A″=e^η≈150 が genre 列（A″≤0.25）の
19 列合計を 30 倍以上上回る場合があり、カウント属性が Z 推定を強く支配した
可能性を説明できる。ただし実際の precision 項は A″ f_l f_l^T に依存するため、
これは曲率ベースの診断である。皮肉なことに all_gaussian 強制は σ̂² が
大きく推定されて自動減衰するため無害だった。
→ **「family を正しくする」だけでは不十分で、切片（offset）・スケーリング・
ブロック間バランスが per-column 実用化の前提条件**（今後の課題）。
なお評価件数は Y と同じ評価ログ由来でリーク懸念があるが、
それでも悪化した（リークで有利になる以前の問題）。

### 10. ゼミで言ってよい主張

1. 現行実装の全列共通 family では混在属性を正しく扱えず、誤った強制は
   崩壊しうる（all_bernoulli）。per-column 化は列単位の重み付けとして
   既存 MCEM+Laplace 枠に自然に入り、数値監査済みの prototype がある。
2. 人工 mixed-X では、per-column 同時統合は単独属性・全列強制のいずれとも
   同等以上で、全属性を自然なスケールで再構成できる唯一の条件だった。
3. 別々に回す実験は単独効果の ablation として有用だが、共通 Z は得られない。
4. 属性を増やせば良いわけではない: 情報の薄い属性は無効果、ノイズは悪化しうる。
5. 実データではスケールの大きいカウント属性が Z を支配して悪化した。
   切片・スケーリングの扱いが per-column 実用化の課題として特定できた。

### 11. 言いすぎな主張（言ってはいけない）

- 「per-column は全列 Gaussian 強制より Z・Y の精度が高い」（人工で差は誤差内。
  Gaussian 強制は Z・Y にはかなり頑健）。
- 「同時統合で大きく改善」（人工での改善幅は小さく、Y 情報が濃い設定に依存）。
- 「MovieLens で per-column が有効」（pilot では悪化。有効性は未実証）。
- 「per-column は完成した手法」（prototype。切片なし・ブロック支配・
  family 選択手続きが未解決）。
- ノイズの用量反応や小さい差（±1 std 内）の断定。

### 12. 次にやるべきこと

1. **X 側切片（列 offset）の導入検討**: η_il = μ_l + f_l^T z_i。
   実データのカウント・非中心属性に必須であることが pilot で判明。
   （実装は本フェーズの範囲外として未着手）
2. カウント属性の前処理規約（log 変換 + Gaussian、または offset 付き Poisson）の比較。
3. ブロック重み・列数不均衡の扱い: 今回は診断（勾配ノルム・llX 記録）のみ。
   weighting の導入は理論的正当化を先に（勝手に導入しない方針を維持）。
4. seed 5 以上への拡張（特に差が小さい項目の確認）。
5. 列ごと family をデータから選ぶ手続き（設計書の指摘どおり組合せ爆発が課題）。
6. NB2-Y との結合（現状 NotImplementedError）は必要になってから。

## 再現性

すべての実験に runinfo CSV（日時・git HEAD・ブランチ・seed 系・設定・変換方法・
実行時間）を保存。seed 系は実験間で分離（81000〜93000）。
既存テスト（pytest experimental 18 件 + test_dual_expfam.py）は全 PASS のまま。
