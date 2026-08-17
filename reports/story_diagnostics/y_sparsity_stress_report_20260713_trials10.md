# Y sparsity stress test（trials=10 拡張版）— 報告

作成日: 2026-07-13
ブランチ: `research/story-diagnostics`
スクリプト: `tools/research_audit/run_y_sparsity_stress.py`（`TRIALS=10`, `RUN_TAG="trials10"`）
結果: `expfam/results/story_diagnostics/y_sparsity_stress_20260713_trials10{.,_agg.,_runinfo.}csv`
図: `figures/story_diagnostics/y_sparsity_{rmse_z,test_y_ll}_trials10.png`

**位置づけ**: 2026-07-13の軽量版（trials=2, `y_sparsity_stress_report_20260713.md`）で見えた傾向が seed 依存でないかを確認するための拡張実験。条件・y_obs_rateは軽量版と同一（`y_only`, `single_gaussian`, `per_column_all`, `all_gaussian` × `y_obs_rate∈{1.0,0.5,0.2,0.1}`）で、trial数のみ2→10に拡張した。既存の `y_sparsity_stress_20260713.csv` 系ファイル・図は上書きしていない（新ファイル名 `_trials10` を使用）。

## 見方

- **RMSE_Z は小さいほど良い**（真のZとの誤差、Procrustes整合後）。
- **test Y log-likelihood / pair（test_y_ll）は大きいほど良い**。ただし常に負の値になる（Poisson尤度の対数なので）。−2.0 より −1.9 の方が良い、という向き。

## 1. 実行結果サマリ

- 条件×y_obs_rate×trial = 4×4×10 = **160 fits**。CSV行数: 生データ161行（header+160）、agg17行（header+16）。
- **NaN・発散は160 fit中0件**（`nan_occurred` 列がすべて `False`）。

## 2. RMSE_Z（mean ± std, n_trials=10）

| y_obs_rate | y_only | single_gaussian | per_column_all | all_gaussian |
|---:|---:|---:|---:|---:|
| 1.0 | 0.308 ± 0.014 | 0.232 ± 0.020 | **0.221 ± 0.020** | 0.233 ± 0.019 |
| 0.5 | 0.460 ± 0.041 | 0.279 ± 0.024 | **0.266 ± 0.022** | 0.331 ± 0.109 |
| 0.2 | 0.857 ± 0.094 | 0.364 ± 0.034 | **0.320 ± 0.025** | 0.732 ± 0.126 |
| 0.1 | 1.176 ± 0.086 | 0.426 ± 0.060 | **0.343 ± 0.026** | 0.769 ± 0.133 |

## 3. test Y log-likelihood / pair（mean ± std, n_trials=10）

| y_obs_rate | y_only | single_gaussian | per_column_all | all_gaussian |
|---:|---:|---:|---:|---:|
| 1.0 | −2.037 ± 0.047 | −2.007 ± 0.039 | **−2.003 ± 0.042** | −2.010 ± 0.044 |
| 0.5 | −2.122 ± 0.060 | −2.025 ± 0.039 | **−2.021 ± 0.044** | −2.043 ± 0.064 |
| 0.2 | −2.382 ± 0.117 | −2.072 ± 0.054 | **−2.052 ± 0.054** | −2.177 ± 0.045 |
| 0.1 | −2.618 ± 0.133 | −2.114 ± 0.067 | **−2.068 ± 0.059** | −2.191 ± 0.066 |

（太字＝各行の最良値。両指標とも全 y_obs_rate で `per_column_all` が最良。）

## 4. per_column_all との差分（`*_agg.csv` の `*_diff_vs_percolumn` 列より）

### RMSE_Z の差分（正の値 = per_column_all より悪い）

| y_obs_rate | y_only − per_column_all | single_gaussian − per_column_all | all_gaussian − per_column_all |
|---:|---:|---:|---:|
| 1.0 | +0.086 | +0.010 | +0.011 |
| 0.5 | +0.194 | +0.013 | +0.065 |
| 0.2 | +0.537 | +0.044 | +0.412 |
| 0.1 | +0.833 | +0.083 | +0.426 |

### test_y_ll の差分（負の値 = per_column_all より悪い）

| y_obs_rate | y_only − per_column_all | single_gaussian − per_column_all | all_gaussian − per_column_all |
|---:|---:|---:|---:|
| 1.0 | −0.034 | −0.004 | −0.006 |
| 0.5 | −0.101 | −0.004 | −0.022 |
| 0.2 | −0.330 | −0.020 | −0.125 |
| 0.1 | −0.550 | −0.046 | −0.123 |

## 5. 個別比較

- **per_column_all vs y_only**: y_obs_rate=1.0では差は小さい（RMSE_Z差+0.086）が、y_obs_rateが下がるにつれ単調に差が拡大し、0.1では+0.833（per_column_allの約3.4倍のRMSE）まで開く。test_y_ll差も−0.034→−0.550と単調拡大。**Yが疎になるほどXの寄与が明確に大きくなる、という傾向がn=10で確認できた。**
- **per_column_all vs single_gaussian**: 差は常に小さい（RMSE_Z差 +0.010〜+0.083）が、y_obs_rateが下がるにつれ緩やかに拡大している（1.0:+0.010→0.1:+0.083）。単独属性でもある程度Zは推定できるが、複数属性統合がわずかに上回り続ける。
- **per_column_all vs all_gaussian**: y_obs_rate=1.0では差はごく小さい（+0.011、既存 single_vs_joint の結果と整合）が、0.2〜0.1で急激に差が拡大する（+0.412〜+0.426）。**family誤指定（all_gaussian）のコストは、Y情報が豊富なときは隠れているが、Y情報が乏しくなると顕在化する。**
- **y_obs_rateが下がるほど差が広がるか**: RMSE_Z・test_y_ll・全3比較のいずれも、y_obs_rate=1.0→0.1にかけて**単調に、あるいはほぼ単調に差が拡大**している（all_gaussianのRMSE_Z差のみ0.2→0.1でわずかに横ばい：+0.412→+0.426）。全体として「Yが疎になるほどXの寄与とfamily正指定の重要性が増す」という傾向は明確。

## 6. per_column_all は本当に安定して良いか

n=10に拡張したことで標準偏差（std）も評価できるようになった。

- **平均が最良なだけでなく、ばらつき（std）も小さい**: y_obs_rate=0.1でのRMSE_Z std は per_column_all=0.026 に対し、all_gaussian=0.133、y_only=0.086、single_gaussian=0.060。per_column_allは平均・分散の両面で最も安定している。
- test_y_ll でも同様の傾向（y_obs_rate=0.1で per_column_all std=0.059 に対し all_gaussian std=0.066、y_only std=0.133）。
- trials=2の軽量版で見えた傾向（差が広がる、all_gaussianが特に悪化する）は、trials=10でも同じ方向・同程度の大きさで再現されており、**seed依存のノイズではなく安定した傾向であると言える**（ただし人工データ1設定のみでの確認であることには注意）。

## 7. 発表で使える主張

**言ってよい主張:**
- 「Yの学習観測率を下げるほど、per_column_allとy_onlyの差、per_column_allとall_gaussian（誤指定）の差がともに拡大する傾向が、trials=10の実験で確認できた（trials=2の軽量版と同じ方向・同程度の大きさ）」
- 「per_column_allは、平均だけでなくtrial間のばらつきも小さく、Yが疎な条件でより安定して良い」
- 「family誤指定（all_gaussian）のコストは、Y情報が豊富なときは目立たないが、Y情報が乏しいと顕在化する」

**言いすぎな主張（避ける）:**
- 「per_column_allが一般に最も安定した手法である」（人工データ1設定・1つの生成パラメータでの結果であり、他の設定への一般化は未確認）
- 「per_column_allがsingle_gaussianより明確に優れている」（差は小さく、diffは+0.01〜+0.08程度でありオーバーラップの余地がある）
- 「Yが疎な実データでも同様の効果が出る」（本実験は人工データのみ。実データ（MovieLens等）での検証は未実施）

## 8. 次のステップ

- フル条件（`single_bernoulli`, `single_poisson`, `all_bernoulli`）への拡張は、今回のtrials=10で傾向が十分安定したと判断できるため優先度は下がったが、必要なら同じ構造(`ACTIVE_CONDITIONS`定数変更)で対応可能。
- 実験2（complementary blocks）・実験3（MovieLens attribute diagnosis）は `story_diagnostics_next_plan_20260713.md` の設計メモを参照し、着手判断待ち。
- `EXPERIMENT_REGISTRY.md` への追記は、正式採用が決まってから行う（今回もまだ実施していない）。
