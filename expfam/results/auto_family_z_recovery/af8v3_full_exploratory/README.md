# AF8-v3 final：loading 積分型 family 選択の 40 データ探索実験（EXPLORATORY）

> **prototype（lineage E + PR #129 の generic model）。正式手法でも manuscript evidence でもない。**
> 40 データは AF6 のデータで、AF6〜AF8 の解析ですでに使われている。**独立な確認（confirmatory）ではない。**
> 基準は Bayesian evidence でも BIC でもない（θ_{−j} は AF6 auto_first anchor に固定）。

## 1. 何をしたか

- 問い（protocol で事前に固定）：loading f_j を積分する（prior ＋ nested Laplace、E2）と、Poisson と CMP(ν=2) の
  2 方向の誤選択のバランスが、Q と AF6 Laplace（f を最適化、prior なし）に比べて改善するか。総合正解率だけでは成功としない。
- データ：AF6 の rep01–rep40（n=75、K=3、Gaussian×3・Poisson×3・CMP2×3、dense-Y / sparse-Y）。count 列 6 本 × 2 regime × 40 = 480 列。
  MCEM の再学習・新しいデータ生成はしていない（hash は `manifest.csv` で AF6 と一致）。
- 主手法 E2、主 prior `B_sg2_0.667`。どちらもこの 40 データの正解率を見る前に `protocol.json` で固定し、変更していない。
  ただし経緯として（独立監査 W4）：AF8 v1 の主スコアは E で、E2 への変更は AF8 の pilot（rep01–03、40 データに含まれ、AF6 でラベルが既知）の数値結果を見た後の v2 で行った
  （`af8v2_numerical_repair_20261115/protocol.json` で開示済み）。prior も、AF6 で Q と B がこのデータでどう誤選択するかが分かった後に設計した。
  正解率は見ていないが、同じデータに対する分岐の自由度（forking paths）があったことは、データの再利用とは別の限界である。
- 比較：A Q、B AF6 Laplace、C joint-MAP Laplace、D prior つき MAP（積分なし）、E joint Laplace、IS（E2 の外側近似の診断のみ）、
  E2|Q（E2 が判定できない列だけ Q に置き換える、別名の副次手法。今回は E2 が全列 DECIDED のため E2 と同一）。
- 統計単位は 40 データ。dataset-cluster bootstrap（10,000 回、AUC は 2,000 回、seed 20261118）の 95% パーセンタイル区間。
  列・regime は独立な単位として扱っていない。多重比較の補正はしておらず、区間は記述的なものである。
  protocol の文面は bootstrap 10,000 回だが、AUC だけはコード（実行前に commit）で 2,000 回にしている（独立監査 N1、文面との相違）。
  AUC は有限な Δ の列だけで計算する（B は 477 列、C は 475 列、他は 480 列）ため、AUC の差は列の集合がわずかに異なる（N3）。

## 2. 手順と gate（FACT）

| 段階 | commit | 結果 |
|---|---|---|
| 修正コード・テスト・protocol | `e8943d3` | 13 テスト通過（AF8 系 42 テスト通過） |
| re-pilot（rep01–03、648 セル、正解率なし） | `fe98694` | S0–S9 すべて通過、v3 pilot とビット単位で同一 |
| 40 データ本実験（8,640 セル） | 実行 HEAD `fe98694` | S0–S9（S7 は pilot のみ）すべて通過 → 正解率を計算 |

数値監査の詳細は `numerical_audit.md`、独立監査は `independent_audit.md`。

## 3. 主な結果（OBSERVED、主 prior、pooled 480 列、分母は各真モデルの全 240 列）

| 手法 | 真 Poisson 正解 | 真 CMP2 正解 | balanced [95% CI] | P→C 誤選択 | C→P 誤選択 | AUC | 失敗 |
|---|---|---|---|---|---|---|---|
| A Q | 234 (0.975) | 188 (0.783) | 0.879 [0.852, 0.906] | 6 | 52 | 0.961 | 0 |
| B AF6 Laplace | 200 (0.833) | 228 (0.950) | 0.892 [0.860, 0.923] | 37 | 12 | 0.973 | 3 |
| C joint-MAP Laplace | 213 (0.888) | 213 (0.888) | 0.888 [0.860, 0.915] | 22 | 26 | 0.959 | 5 |
| D prior MAP | 236 (0.983) | 187 (0.779) | 0.881 [0.852, 0.910] | 3 | 52 | 0.980 | 0 |
| **E2（主）** | **223 (0.929)** | **216 (0.900)** | **0.915 [0.888, 0.940]** | **17** | **24** | **0.980** | 0 |
| E | 228 (0.950) | 211 (0.879) | 0.915 [0.890, 0.938] | 12 | 29 | 0.975 | 0 |
| IS（診断） | 223 | 216 | 0.915 | 17 | 24 | 0.980 | 0 |

判定不能（NUMERIC_UNDECIDED）は D が 2 列、C が 1 列、他は 0（表の正解に含まれない）。
regime 別（E2）：dense は balanced 0.963（P 0.967、C 0.958）、sparse は 0.867（P 0.892、C 0.842）。
B の失敗 3 列と C の失敗 5 列は AF6 での失敗で、正解には数えていない（Q による置き換えなし）。

**対比較（E2 − 比較手法、pooled、差 [95% CI]、移行は列数）**

| 比較 | 真 Poisson 正解率 | 真 CMP2 正解率 | balanced | AUC | 移行（P：誤→正 / 正→誤、C：誤→正 / 正→誤） |
|---|---|---|---|---|---|
| E2 − A Q | −0.046 [−0.071, −0.025] | +0.117 [+0.075, +0.162] | +0.035 [+0.015, +0.058] | +0.018 [+0.009, +0.031] | P 0 / 11、C 28 / 0 |
| E2 − B AF6 Laplace | +0.096 [+0.058, +0.137] | −0.050 [−0.083, −0.021] | +0.023 [−0.004, +0.050] | +0.006 [+0.003, +0.010] | P 23 / 0、C 0 / 12 |
| E2 − C joint-MAP | +0.042 [+0.008, +0.075] | +0.013 [−0.013, +0.042] | +0.027 [+0.006, +0.048] | +0.020 [+0.011, +0.030] | P 15 / 5、C 6 / 3 |
| E2 − D prior MAP | −0.054 [−0.083, −0.029] | +0.121 [+0.079, +0.167] | +0.033 [+0.010, +0.058] | −0.001 [−0.002, +0.000] | P 0 / 13、C 29 / 0 |
| D − B（prior の効果） | +0.150 [+0.104, +0.196] | −0.171 [−0.225, −0.121] | −0.010 [−0.050, +0.029] | +0.007 [+0.003, +0.011] | P 36 / 0、C 0 / 41 |
| E2 − E | −0.021 [−0.042, 0.000] | +0.021 [0.000, +0.046] | 0.000 [−0.017, +0.017] | +0.005 [+0.002, +0.008] | P 2 / 7、C 6 / 1 |

**prior 感度（E2、pooled balanced）**：B_sg2 = 0.167 / 0.333 / 0.667 / 1.333 / 2.667 で 0.840 / 0.892 / 0.915 / 0.902 / 0.890、
A_sf2 = 0.167 / 0.667 / 2.667 で 0.681 / 0.900 / 0.902、B_EB で 0.915。主 prior からの選択の変化は 13〜146 列（判定不能になった列も「変化」に数える）。
prior の尺度を大きくすると P→C が増え C→P が減る、という単調な境界の移動が見られる。

## 4. 解釈（INTERPRETATION）

- E2 は AF6 Laplace と比べて P→C の誤りを 37 → 17 に減らしたが、CMP2 の正解も 228 → 216 に減った。
  移行は一方向だけ（P は誤→正 23・正→誤 0、C は正→誤 12・誤→正 0）で、判定の順位づけよりも**判定境界の平行移動**として振る舞っている。
  balanced の改善 +0.023 の区間は 0 を含む。
- 分解：f 座標で見ると、prior（D − B）が境界を Poisson 側へ大きく動かし（P +36、C −41）、積分（E2 − D、−½ log|H_f| の差）が CMP2 側へ戻す（P −13、C +29）。
  積分は AUC をほぼ変えない（−0.001）。AUC の小さな改善（E2 − B で +0.006）は主に D 段階で生じている。
  ただしこの「prior の効果」と「積分の効果」の割り振りは座標に依存する（独立監査 W3）。B 系 prior は g = √A''_m(0) f の上で定義され、
  A''_CMP2(0) = 0.513 なので、f 上の MAP（D）には候補間で (K/2) log A''_CMP2(0) = −1.00 nat の Jacobian の差が全列で CMP2 に不利に入る。
  E2 は座標の変換で変わらないが D は変わるため、g 座標で比べると約 1 nat が「prior」から「積分」へ移る。移行の向きは事実だが、配分は手法そのものの性質ではない。
- 2 方向の誤選択のバランスは prior の尺度で決まり、主 prior は事前に固定したものだが、今回のデータでは B 系 prior のうち balanced が最大（B_EB と同値）の値に当たっている。
  別の妥当な prior では Q・B と同程度以下になる。
- E と E2 は dense では選択が完全に一致し、sparse で 16 列が入れ替わる（真 Poisson 9 列・真 CMP2 7 列、正解数は正味 0）。近似の違いは sparse で無視できない。
- E2 と IS は全列で同じ選択をした。外側の Gaussian 近似の誤差は選択に影響していない（Z の Laplace 誤差は未検証）。

## 5. 言えないこと

- 独立データでの改善、実データでの有効性、一般性（すべて未検証）。
- E2 が厳密な周辺尤度に近いこと。
- prior の選び方に依存しない改善。
- 多重比較を考慮した有意性。

## 6. ファイル

| ファイル | 内容 |
|---|---|
| `protocol.json` | 結果を見る前に固定した protocol（commit `e8943d3`） |
| `manifest.csv` | 80 ケースの seed・hash（AF6 と照合） |
| `full/candidate_scores.csv` | 8,640 セルの全スコア・証明書（= candidate_scores） |
| `full/full_pairs.csv`, `full/full_start_certificates.csv`, `full/full_is_draw_log.csv` | 対ごとの Δ、開始点ごとの証明書、IS の draw ログ |
| `full/full_numerical_audit.json`, `full/runinfo_*.json` | 停止規則の評価と実行情報 |
| `full/full_vs_repilot_rep01_03_summary.json` | full の rep01–03 と re-pilot の比較（commit `d5227f0` の script） |
| `selection_results.csv` | 480 列 × 手法の Δ・状態・選択 |
| `accuracy.csv` | 手法 × subset の正解率・誤選択率・状態率・AUC と CI |
| `paired_comparison.csv` | 対比較の差・CI・移行数 |
| `statistical_analysis.csv` | データごとの正解数の差（平均・CI・正/0/負のデータ数） |
| `prior_sensitivity.csv` | 9 prior × D/E2/E |
| `approximation_diagnostics.csv` | E・E2・IS の差、分解、曖昧な列 |
| `fig1_delta_by_truth.png`, `fig2_prior_sensitivity.png`, `fig3_E_vs_E2.png` | 図 |
| `runinfo.json` | 解析の実行情報（入力ファイルの sha256） |
| `pilot/` | re-pilot（rep01–03） |
| `numerical_audit.md`, `independent_audit.md` | 数値監査・独立監査 |

## 7. 再現コマンド（repository root で実行）

```sh
.venv/Scripts/python.exe expfam/src/experimental/run_af8v3_full_integrated_loading.py --write-protocol   # protocol.json（commit 済み）
.venv/Scripts/python.exe expfam/src/experimental/run_af8v3_full_integrated_loading.py --pilot --shard i/6   # i = 0..5
.venv/Scripts/python.exe expfam/src/experimental/run_af8v3_full_integrated_loading.py --collect-pilot
.venv/Scripts/python.exe expfam/src/experimental/analyze_af8v3_full_integrated_loading.py --pilot-audit
.venv/Scripts/python.exe expfam/src/experimental/analyze_af8v3_full_integrated_loading.py --manifest
OMP_NUM_THREADS=1 .venv/Scripts/python.exe expfam/src/experimental/run_af8v3_full_integrated_loading.py --full --shard i/12   # i = 0..11
.venv/Scripts/python.exe expfam/src/experimental/run_af8v3_full_integrated_loading.py --collect-full
.venv/Scripts/python.exe expfam/src/experimental/analyze_af8v3_full_integrated_loading.py --full-audit
.venv/Scripts/python.exe expfam/src/experimental/analyze_af8v3_full_integrated_loading.py --analyze
.venv/Scripts/python.exe expfam/src/experimental/analyze_af8v3_full_integrated_loading.py --compare-full-repilot   # commit d5227f0 以降
```
