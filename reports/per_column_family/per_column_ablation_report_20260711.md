# per-column 属性追加 ablation + ノイズ属性チェック レポート

作成日: 2026-07-11
ブランチ: `research/per-column-validation`
スクリプト: `tools/research_audit/run_per_column_attribute_ablation.py`,
`tools/research_audit/run_per_column_noise_check.py`
結果: `expfam/results/per_column_family/attribute_ablation_{summary,agg,runinfo}.csv`,
`noise_check_{summary,agg,runinfo}.csv`
図: `figures/per_column_family/attribute_ablation_lines.*`, `noise_check_lines.*`

## 目的

「属性を増やせば必ず良いのか」に答える。
(1) 意味のある属性ブロックを 1 つずつ追加したときの変化（ablation）、
(2) Z_true と無関係なノイズ属性を追加したときの変化（noise check）。

設定は single-vs-joint と同じ人工 mixed-X（n=80, k*=2, Poisson-Y,
strict held-out test 20%, L=5, EM 8 反復, 3 seeds）。seed 系は実験ごとに分離
（ablation: 84000/85000/86000/87000、noise: 88000/89000/90000/91000）。
全 33 fits で NaN なし。

## 実験 1: 属性追加 ablation（すべて per-column 正指定）

| step | 条件 | 列数 | RMSE_Z | test Y ll (/pair) |
|---|---|---|---|---|
| 0 | Y-only | 0 | 0.295 ± 0.032 | −2.040 ± 0.043 |
| 1 | + Bernoulli 3列 | 3 | 0.296 ± 0.027 | −2.042 ± 0.047 |
| 2 | + Gaussian 3列 | 6 | **0.231 ± 0.040** | **−2.016 ± 0.051** |
| 3 | + Poisson 3列 | 9 | **0.229 ± 0.041** | **−2.015 ± 0.041** |
| 4 | + ノイズ 3列（N(0,1), Z と無関係, gaussian 正指定） | 12 | 0.231 ± 0.041 | −2.015 ± 0.040 |

読み取り:

1. **属性追加の効果は属性の情報量に依存し、単調改善ではない。**
   Bernoulli ブロック追加（step 0→1）は RMSE_Z・test ll とも改善なし
   （このデータ生成では二値属性 1 列あたりの情報量が小さい。数式監査の
   ブロック重み診断で見た「Bernoulli の曲率 A″≤1/4 で勾配寄与が小さい」と整合）。
2. **情報の濃い Gaussian ブロックの追加（step 1→2）が改善のほぼすべて**
   （RMSE_Z −22%、test ll +0.026/pair）。Poisson 追加はわずかな追加改善。
3. **ノイズ 3 列の追加（step 3→4）は改善ゼロ**（微悪化〜横ばい）。
   「列を増やす」こと自体に価値はない。
4. 3 seeds の標準偏差は条件間の差と同程度の項目もある。step 2 の改善は
   3 trial すべてで方向が一致しており頑健だが、step 3→4 の微差は
   seed 差の範囲内。

## 実験 2: ノイズ属性チェック（informative 9 列 + ノイズ列、すべて正指定）

ノイズ列は Z_true と無関係（gaussian: N(0,1) / bernoulli: p=0.5 /
poisson: rate=2.0）で、**family 指定自体は正しい**。
「family が正しくても情報がなければ意味がない」ことを見る。

| 条件 | ノイズ列数 | RMSE_Z | test Y ll (/pair) |
|---|---|---|---|
| no_noise | 0 | 0.223 ± 0.014 | −2.038 ± 0.046 |
| + bern_noise3 | 3 | 0.224 ± 0.014 | −2.038 ± 0.046 |
| + pois_noise3 | 3 | 0.224 ± 0.014 | −2.038 ± 0.046 |
| + gauss_noise3 | 3 | 0.233 ± 0.025 | −2.045 ± 0.036 |
| + gauss_noise6 | 6 | 0.223 ± 0.015 | −2.038 ± 0.046 |
| + gauss_noise12 | 12 | 0.235 ± 0.021 | −2.046 ± 0.038 |

読み取り:

1. **ノイズ属性はどの条件でも改善しない。** 平均では横ばい〜微悪化。
2. **悪化は seed 依存で現れる**: trial 1 では gauss_noise3 / gauss_noise12 で
   RMSE_Z が 0.227→0.258 / 0.257（+13%）と明確に悪化した一方、
   trial 0/2 ではほぼ不変。ノイズ列数に対する平均の変化は単調でなく
   （noise6 ≈ no_noise）、3 seeds では用量反応を確定できない。
3. **Gaussian ノイズが最も悪影響が出やすい**（bern/pois ノイズはほぼ無害）。
   これは数式監査のブロック重み診断と整合する: Gaussian 列は重み 1/σ̂_l² で
   E-step 勾配への寄与が大きく、Bernoulli ノイズは曲率上限 1/4、
   Poisson ノイズも推定 F が縮小すれば寄与が減るため。
4. 実行安定性への影響はなし（NaN 0、リトライ 0）。

## 結論（「属性を増やせば必ず良いのか」への回答）

- **良くならない。** 改善するのは「Z に関係する情報を持つ属性を適切な family で
  入れた場合」だけであり、(a) 情報の薄い属性は足しても変わらず、
  (b) 無関係な属性は足しても最良で横ばい、seed によっては明確に悪化する。
- per-column family は「異なる型の属性を同時に入れるための仕組み」であって、
  「入れた属性が有益であることを保証する仕組み」ではない。
  どの属性を入れるかの選択は依然として重要な別問題
  （列単位の周辺診断による事前絞り込み等、設計書の今後の課題と同じ位置づけ）。

## 限界

- 3 seeds の pilot 規模。差の小さい項目（step 3→4、ノイズ用量反応）は
  seed を増やさないと確定できない。
- 人工データ 1 生成設定のみ。Y の情報が濃い（全ペア観測 × n=80）設定なので、
  X の寄与が相対的に小さめに出る方向のバイアスがある。
- ノイズの「強度」（分散スケールや Z との弱相関）は振っていない。
