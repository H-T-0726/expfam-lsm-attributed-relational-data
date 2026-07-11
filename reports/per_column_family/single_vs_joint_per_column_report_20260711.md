# 単独属性 vs per-column 同時統合 比較レポート（人工 mixed-X）

作成日: 2026-07-11
ブランチ: `research/per-column-validation`
スクリプト: `tools/research_audit/run_per_column_single_vs_joint.py`
結果: `expfam/results/per_column_family/single_vs_joint_{summary,agg,runinfo}.csv`

## 目的（ゼミの問いへの直接回答）

> 「ジャンルだけを Bernoulli、平均評価だけを Gaussian、評価件数だけを Poisson として、
> 別々にモデルを回せばよいのでは？ なぜ per-column family で複数属性を
> 同時に入れる必要があるのか？」

## 設定

- 共通の Z_true（n=80, k*=2）から Gaussian 3列 + Bernoulli 3列 + Poisson 3列の
  混在属性 X と Poisson-Y（w0=1.2, w=0.3）を生成（3 trials、data/model/split seed 分離）
- Y は strict held-out（pair mask、test 20%）: test ペアは学習から完全除外し、
  plug-in の test 対数尤度（/pair、全定数込み）と RMSE で評価
- L=5, EM 8 反復。全 27 fits で NaN なし

### 条件と変換（runinfo に同内容を記録）

| 条件 | X | family | 位置づけ |
|---|---|---|---|
| single_gaussian / single_bernoulli / single_poisson | 該当 3 列のみ | 正指定（スカラー） | 単独属性 ablation |
| per_column_all | 9 列 | 列ごと正指定 | **prototype（本命）** |
| all_gaussian / all_bernoulli / all_poisson | 9 列生値のまま | 単一 family 強制 | **比較用の誤指定モデル**（現行の全列共通 family 制約の模擬。自然なモデルではない） |
| all_bernoulli_binarized | Gaussian 列: 列中央値超→1、Poisson 列: >0→1 | 全列 Bernoulli | **比較用の誤指定モデル**（明示的に情報を捨てる変換） |
| y_only | 不使用（F=0 固定） | — | ベースライン |

## 結果（3 trials 平均 ± 標準偏差）

| 条件 | RMSE_Z | test Y ll (/pair) | test Y RMSE | X再構成: gauss / bern / pois |
|---|---|---|---|---|
| **per_column_all** | **0.235 ± 0.016** | **−2.047 ± 0.012** | **2.066** | **0.297 / 0.450 / 1.394** |
| all_gaussian | 0.234 ± 0.018 | −2.048 ± 0.014 | 2.079 | 0.300 / 0.679 / 2.555 |
| single_gaussian | 0.243 ± 0.017 | −2.050 ± 0.017 | 2.078 | 0.300 / — / — |
| all_bernoulli_binarized | 0.292 ± 0.016 | −2.063 ± 0.014 | 2.119 | （二値化後スケール、他と比較不可） |
| single_poisson | 0.294 ± 0.013 | −2.067 ± 0.005 | 2.129 | — / — / 1.350 |
| all_poisson | 0.296 ± 0.021 | −2.068 ± 0.022 | 2.111 | 1.474 / 0.689 / 1.365 |
| single_bernoulli | 0.321 ± 0.018 | −2.078 ± 0.017 | 2.159 | — / 0.451 / — |
| y_only | 0.328 ± 0.022 | −2.079 ± 0.020 | 2.160 | — |
| all_bernoulli | 0.797 ± 0.822 | −36.9 ± 60.3 | 507.5 | 0.968 / 0.466 / 2.716 |

（X 再構成は「学習に使った X」に対する mean スケール RMSE。binarized 条件は
二値化後の X に対する値なので、生値条件と数値を直接比較してはいけない。）

## 読み取り

### 1. 「別々に回せばよいのでは？」→ 部分的にはよいが、同時統合が一貫して上

- 単独属性モデルはどれも動くが、**Z 回復も held-out Y 予測も per-column 同時統合が
  一貫して同等以上**（RMSE_Z: 0.235 vs 0.243〜0.328、test ll: −2.047 vs −2.050〜−2.079）。
- 単独属性の中でも情報量に差がある（この生成設定では Gaussian ブロックが
  最も情報が多く single_gaussian ≈ joint に近い。Bernoulli 単独が最弱で
  y_only とほぼ同水準）。つまり「単独でどこまで行けるか」は
  **どの属性を選んだかに強く依存**し、事前にはわからない。
- 別々に回す価値は「各属性の単独効果を見る ablation」として明確にある。
  ただし別々に回すと **Z が条件ごとに別物**になり、「1 つの共通潜在空間で
  属性と関係データを同時に説明する」という LSM の目的は達成できない。

### 2. per-column 同時統合で何が変わるか

- **1 つの共通 Z に全属性の情報が集約される**: joint は全単独条件を Z・Y 両指標で
  上回り、かつ 3 ブロックすべての X 再構成が単独モデルと同水準
  （gauss 0.297 vs 0.300、bern 0.450 vs 0.451、pois 1.394 vs 1.350）。
  = 各属性の説明力を犠牲にせず統合できている。
- 改善幅は本設定では小さい（RMSE_Z で 3〜28% 程度）。Y 側の情報が既に多い
  （n=80 の全ペア）ため、X の追加寄与が相対的に小さい設定であることに注意。

### 3. 全列共通 family 強制で何が悪くなるか（3 つの異なる壊れ方）

1. **all_bernoulli（生値のまま強制）**: trial 0 で崩壊（RMSE_Z 1.75、test ll −106）。
   3 trial 平均でも最悪。誤 family の quasi-likelihood 勾配が Z 推定を壊しうる。
2. **all_poisson**: 崩壊はしないが Gaussian ブロックの再構成が 5 倍悪化
   （0.30→1.47）し、RMSE_Z も劣化（0.296）。
3. **all_gaussian**: Z と Y ではほぼ無害（既存デモ・過分散フェーズの
   「Gaussian は比較的安全なデフォルト」と整合）。**ただし** Bernoulli/Poisson
   ブロックの X 再構成は 1.5〜1.8 倍悪化（0.679 / 2.555）。予測値が確率・カウントの
   定義域を守らない（負のカウント等）ため、X 側の解釈・生成モデルとしての利用は
   できない。
4. **BIC の落とし穴**: all_bernoulli の BIC（4884〜10231）は per_column_all
   （11574〜11717）より小さく見えるが、これは非二値データに対する Bernoulli
   「尤度」が確率モデルとして無効なため。**誤指定 family 間の BIC 比較は無意味**
   である。各列の family 指定が妥当である場合、per-column 化は混在属性を
   列ごとのデータ型に対応した尤度で扱えるため、BIC などのモデル比較を
   意味あるものにできる。

### 4. binarized 変換（明示的に情報を捨てる場合）

all_bernoulli_binarized は崩壊せず安定（0.292）だが、joint より一貫して劣る。
「変換すれば単一 family でも動く」は正しいが、連続値・カウントの情報を捨てる
コストを払う。

## 言ってよい主張 / 言いすぎな主張（この実験からの範囲）

- **言ってよい**:
  - 混在属性では、per-column 正指定の同時統合が、単独属性・全列強制のいずれとも
    同等以上で、全ブロックの X を自然なスケールで再構成できる唯一の条件だった。
  - 誤った単一 family の強制は、族の選び方次第で Z 推定が崩壊しうる
    （all_bernoulli）。Gaussian 強制は Z・Y にはほぼ無害だが X 側の解釈を失う。
  - 別々に回す実験は ablation として有用だが、共通 Z の推定という目的は
    果たせない。
- **言いすぎ**:
  - 「per-column は全列 Gaussian 強制より Z 推定精度が高い」（差は誤差範囲内）。
  - 「同時統合すれば大きく改善する」（本設定の改善幅は小さい。Y 情報が濃い
    設定であることに依存）。
  - 「実データでも同様」（人工データ 1 設定 × 3 seeds の結果にすぎない）。
