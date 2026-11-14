# AF6: Laplace 分布選択法の独立監査と、新しい seed による再現性の確認

- 系列: lineage E＋PR #129 の generic model（prototype、本文には採用不可）。Dual-ExpFam LSM の n=75、K=3、X は Gaussian×3 / Poisson×3 / CMP(ν=2)×3、Y は Bernoulli。
- 種類: Phase A は AF5 の独立監査（新しいデータなし、MCEM 0 回）。Phase B は事前登録した確認実験で、新しいデータ 40 個を使う。
- commit:
  - Phase A: `ed7807f`
  - Phase B の protocol、コード、test の凍結: `9e7775a`（protocol sha256 `eea15292…e71f`）。データの生成より前に commit した。
  - 本実行の runinfo はすべて git_head が `9e7775a` で、tracked の変更はない。
- 主要な結果（OBSERVED）:
  - **主要評価項目は達成しなかった。**
  - sparse で、データセットごとの正解列数の差 D_r（Laplace − 同じ anchor の Q）の平均は **+0.225、95% CI [−0.075, 0.525]**（40 データ）。
  - 成功の条件（平均 > 0 かつ CI の下限 > 0）を満たさない。

## 1. Phase A（AF5 の独立監査）の要約

詳細は `independent_audit.md` と `phase_a/` にある。

- 判定は WARNING で、BLOCKER はない。
- 独立な実装（log joint、勾配、完全な Hessian、Newton、固有値による log det、CMP2 の閉形式）で、AF5 の 1440 個の score を差 8e-10 以内で再現した。状態、選択、指標はすべて一致した。
- AF4 の CMP2 の階乗の問題は、AF5 には入っていない。
- 138 列で熱力学的積分（TI）を行い、Laplace 近似の誤差を見た。
  - sparse では Δ の符号の一致が 118/120、偏りの中央値は −0.12。
  - TI の符号で選んだ場合、sparse の Q との差は +0.35 [−0.05, 0.70]。
- AF5 README の誤記は `independent_audit.md` §4 に記録した。AF5 の成果物は変更していない。

## 2. 条件と独立性（FACT）

- 生成と推定の条件は AF2/AF5 と同じである（`protocol.json` の `data`、`mcem`）。
- 新しいデータは 40 個（rep01–rep40）。
  - data seed は 1681001+r。そのほかの seed は `seed_manifest.csv` にある。
  - seed の値は AF1–AF5 と重ならない（`seed_disjointness.json`）。
- 生成後の X/Y のハッシュは、40 個すべてが AF2 の 20 データと異なり、40 個どうしも互いに異なる（`verification/data_independence.csv`）。
- 統計の単位はデータセット（40）である。
- **MCEM はデータセット × regime ごとに 1 回で、合計 80 回の fit を新たに行った。** 中身は AF2 の Auto 探索（auto_first、START_FIRST）と refit である。
  - 80 回すべて OK だった。retry、置き換え、seed rescue はいずれも 0。
  - 事後の mode 診断では、rep13 dense の 1 件が NOT_STATIONARY（grad 2.4e-7、200 反復）だった。fit の状態は OK なので、そのまま使った。
- 手法 A〜E はすべてこの同じ fit を使う。Oracle や Wrong の追加 fit は行っていない。

## 3. 手法（protocol で固定）

| 種類 | 手法 | 内容 |
|---|---|---|
| 主 | A Laplace | AF5 の `run_column` を変更せずに使う。Δ が有限なら sign(Δ) で選び、それ以外は B に従う。 |
| 主 | B Q | 同じ Auto anchor の Zsamples を使う（AF3 の Z7）。 |
| 副 | C | AF2 の Auto が選んだもの。 |
| 副 | D 保守的 | AF5 の状態が DECIDED なら Laplace、それ以外は B。 |
| 診断 | E joint MAP | sign(delta_jointmap_variant)。昇格はしない。 |
| 近似の診断 | TI | sparse の全列と、dense で \|Δ\|<2 の列が対象。TI の符号で選ぶ。 |

## 4. 結果（OBSERVED、240 列 / regime の全列。判定できない列も含む）

出典は `statistical_analysis.csv`、`full/analysis/accuracy.csv`、`paired_comparison.csv`、`approximation_diagnostics.csv`。

### 4.1 正解率（count 列。分母は regime ごとに 240）

| regime | 真の分布 | A Laplace | B Q | C AF2 | D 保守的 | E joint MAP | TI |
|---|---|---|---|---|---|---|---|
| dense | 全体 | 0.946 | 0.946 | 0.946 | 0.942 | 0.942 | 0.942 |
| dense | Poisson | 0.900 | 0.975 | 0.975 | 0.900 | 0.917 | 0.892 |
| dense | CMP2 | 0.992 | 0.917 | 0.917 | 0.983 | 0.967 | 0.992 |
| sparse | 全体 | 0.850 | 0.813 | 0.817 | 0.833 | 0.858 | 0.854 |
| sparse | Poisson | 0.792 | 0.975 | 0.983 | 0.842 | 0.900 | 0.775 |
| sparse | CMP2 | 0.908 | 0.650 | 0.650 | 0.825 | 0.817 | 0.933 |

### 4.2 D_r（データセットごとの正解列数の差、40 データ、percentile bootstrap 10000、seed 20261116）

| 比較 | regime | 平均 | 95% CI | +/0/− | 位置づけ |
|---|---|---|---|---|---|
| **A−B** | **sparse** | **+0.225** | **[−0.075, 0.525]** | 16/14/10 | **主要（未達成）** |
| A−B | dense | 0.000 | [−0.200, 0.200] | 8/25/7 | 副次 |
| A−C | sparse | +0.200 | [−0.125, 0.500] | 16/13/11 | 副次 |
| A−C | dense | 0.000 | [−0.200, 0.200] | 8/25/7 | 副次 |
| D−B | sparse | +0.125 | [−0.125, 0.350] | 12/21/7 | 副次 |
| D−B | dense | −0.025 | [−0.225, 0.175] | 7/26/7 | 副次 |
| E−B | sparse | +0.275 | [0.050, 0.500] | 14/21/5 | 診断（確認的な主張には使わない） |
| E−B | dense | −0.025 | [−0.200, 0.150] | 5/30/5 | 診断 |
| TI−B | sparse | +0.250 | [−0.050, 0.525] | 15/17/8 | 近似の診断 |
| B−C | sparse | −0.025 | [−0.075, 0.000] | 0/39/1 | 副次 |

- 多重比較の補正はしていない。主要評価項目は 1 つで、それ以外は記述のためである（protocol の `multiplicity`）。
- 「TI で確かめた列だけ」の A−B は、全列のときと一致した（sparse +0.225、dense 0.000）。確かめられなかった列（sparse 7 列、dense 2 列）では、A と B の正誤がすべて同じだったためである。

### 4.3 訂正と副作用（B を基準にした A）

- sparse: 訂正（B が誤り、A が正しい）が 31 列、副作用（B が正しい、A が誤り）が 22 列。**副作用 22 列はすべて Poisson → CMP2 の向き**である。
- dense: 訂正 9 列、副作用 9 列。副作用 9 列はすべて Poisson → CMP2。
- TI の符号で選んだ場合も同じ傾向になる（sparse で訂正 34、副作用 24、副作用はすべて Poisson → CMP2）。
- A が誤った列のうち、誤りの選択が anchor の分布と一致した割合は、sparse 13/36、dense 4/13。

### 4.4 判定状態と失敗

- 状態の内訳:
  - sparse: DECIDED 164、DECIDED_LAPLACE_UNASSESSED 70、UNDECIDED_LAPLACE 5、FAILED 1。
  - dense: それぞれ 227、10、1、2。
  - UNDECIDED_NUMERIC と NOT_CONVERGED は 0。
- FAILED の 3 列（rep09 dense の列 3、rep15 dense の列 3、rep15 sparse の列 3。`full/failure_registry.csv`）について:
  - 原因は、AF5 の CMP 級数評価が η≈51 で項数の上限（400 万）を超えたことである。エラー文は `full/candidate_scores.csv` の `error` 列と `full/select_progress_*.jsonl` にある。真の分布はいずれも Poisson。
  - protocol に従って A は B に戻した（3 列とも B は正しい）。
  - 再生成や除外はしていない。
- sparse の状態別に見た A の正解（Poisson / CMP2）。事後の記述集計で、出典は `verification/posthoc_status_by_true_family.csv`:

  | 状態 | 真 Poisson の誤り | 真 Poisson の正解 | 真 CMP2 の誤り | 真 CMP2 の正解 |
  |---|---|---|---|---|
  | DECIDED | 19 | 62 | 8 | 75 |
  | IS で未評価 | 5 | 30 | 1 | 34 |

### 4.5 Laplace 近似の誤差（TI、`approximation_diagnostics.csv`、`fig2_laplace_vs_ti.png`）

- sparse では TI を 239 列で実行した。
  - 符号の一致は 234 列、2·SE で確かめられた列は 232 列。
  - Δ_L − Δ_TI の中央値は −0.125、|誤差| の最大は 0.65。MC SE の中央値は 0.048。
- dense では TI を 37 列で実行し、符号の一致は 36 列だった。誤差の中央値は −0.007。
- HMC の受理率の最小値は 0.87。
- TI は f\* での周辺尤度を測るもので、f についての最大化（profile）を含まない。θ_{−j} は anchor に固定している。

### 4.6 計算時間（`verification/compute_time.csv`、`full/runinfo_*.json`）

- fit の時間は、1 回あたり平均 40 秒、合計 0.89 時間（5 並列）。
- 選択（AF5 Laplace、IS、Q、TI）の時間:
  - 合計 10.2 時間（dense 2.7 時間、sparse 7.5 時間）で、5 並列の実時間は約 2.1 時間。
  - protocol の見積もり（直列で約 8 時間）を上回ったが、条件は変えていない。

## 5. 独立した検証（step 10）

`verify_af6_results.py` を使い、出力は `verification/` に置いた。AF5 のコードではなく、Phase A の独立実装の経路で計算した。

- 957 個の OK の candidate score を再計算した。|ΔS| の最大は 7.8e-10、Hessian の最小固有値は 0.93 以上、勾配は 1e-9 以下。
- 480 列について、状態、選択、主要評価項目を独自に再計算した。
  - 状態の不一致 0、選択の不一致 0。
  - sparse の D_r は +0.225 [−0.075, 0.525]、dense は 0.000 [−0.2, 0.2]。報告値と一致した。
- データの独立性と来歴: X/Y はすべて新しい。生成し直したハッシュ、mask、npz の sha256 は fit_manifest とすべて一致した。
- 限界: Q（AF3 の `score_candidate`）は再実装していない。B は保存した選択を使った。
- 読み取り専用の独立監査（research-auditor）の結果は §7 にある。

## 6. 解釈（INTERPRETATION / HYPOTHESIS）

- AF5 の sparse での改善（+0.40 [0.05, 0.70]、AF2 と同じ 20 データ）は、新しい 40 データで事前に決めた基準では再現しなかった。
  - 点推定は同じ向き（+0.225）で、効果の大きさは約半分である。
  - power_note の検出力は、真の平均が 0.4 なら 0.92、0.2 なら 0.39 だった。
  - 観測された SD は 1.00 で、計画時の 0.75 より大きい。この SD で真の平均が 0.225 なら、正規近似の検出力は約 0.30 しかない。
  - したがってこの結果は、効果がないことも、小さい正の効果があることも否定できない。
- 改善の中身は入れ替えである。
  - 真 CMP2 列の正解率は大きく上がる（sparse 0.65→0.91）。
  - 一方、真 Poisson 列の正解率は下がる（0.975→0.79）。
  - その差し引きで、正味は +9 列 / 240 列になる。
- この入れ替えは、TI の符号で選んでも同じように起こる（Poisson 0.775、CMP2 0.933）。
  - **したがって、Poisson → CMP2 の副作用は、f\* と anchor の θ のもとでの Z の積分に対する Laplace 近似の誤差ではない。** f\* と anchor の θ に条件づけた周辺尤度という基準そのものの性質と考えられる（HYPOTHESIS）。
  - ただし、f を profile 化したことや θ を固定したことの影響は、まだ切り分けていない。
  - Laplace の偏り（−0.12）は、むしろ Poisson 寄りである。
- 保守的な規則 D は副作用を 16 列まで減らしたが、訂正も 21 列に減り、D−B は +0.125 [−0.125, 0.35] だった。
- joint MAP（E）の sparse での CI は 0 を含まない。しかしこれは診断であり、事前に昇格しないと決めていた。確認的な証拠にはしない。

## 7. 独立監査（research-auditor、読み取り専用）

- **判定: BLOCKER なし。** 主要評価項目の未達成は、CSV と protocol から正しく導かれている。
- 確認された点:
  - runinfo 11 件はすべて `9e7775a` で、tracked の変更はなく、protocol の sha も一致する。
  - progress の jsonl に再実行の痕跡はない。
  - A、B、失敗時の規則、480 列の保持、bootstrap は protocol どおりである。
  - 失敗した 3 列は除外も再生成もしていない。
- 監査後に確認したこと: `git diff 9e7775a` で runner、analyzer、監査モジュール、AF5 runner に差分がないことを確かめた（解析は `--analyze` のため runinfo がない）。
- WARNING とその対応:
  - (1) protocol からの逸脱を書いた節がない → §8 に追加した。
  - (2) 計算時間と状態 × 真の分布の表を出すスクリプトがない → `verify_af6_results.py --tables` で `verification/` に出力し、出典を差し替えた。
- NOTE とその対応:
  - Δ=0 のときは Poisson を選ぶ。E は jointmap が非有限なら B に戻る。どちらも protocol に明記されていないが、該当する列はない、または影響は無視できる。
  - FAILED を「誤り」として数えた場合、sparse の平均は 0.200 になる。結論は変わらない。
  - 照合した seed 集合には AF3/AF4 の IS/TI の seed を含めていない。値域が異なるので重なりはないと考えられるが、厳密には UNVERIFIED。
  - 観測された SD に基づく検出力を §6 に追記した。

## 8. protocol からの逸脱（DEVIATIONS）

1. X/Y のハッシュを AF2 データと比べる処理は、protocol では「analysis」で行うと書いた。実際には、結果を見た後に追加した `verify_af6_results.py` で行った。結果（すべて新しいデータ）は判定に影響しない。
2. 副次評価項目の計算時間は、事前登録した解析スクリプトには含まれていない。事後に `verification/compute_time.csv` として出力した。
3. 計算時間は見積もり（直列で約 8 時間）を超えた（選択だけで 10.2 時間）。設定は変えていない。
4. 状態 × 真の分布の表（§4.4）は、事後の記述集計である。

## 9. 主張しないこと

- Laplace 法を新しい分布選択アルゴリズムとして採用すること。
- 真の周辺尤度のもとで改善すること（TI は f\* と固定した θ での値）。
- Z の回復が改善すること（分布の正解率と Z の回復は別の問題で、今回は評価していない）。
- 実データへの一般化。
- Bayesian evidence や BIC と呼ぶこと。

## 10. 成果物

| ファイル | 内容 |
|---|---|
| `protocol.json`, `seed_manifest.csv`, `seed_disjointness.json` | 凍結した protocol（`9e7775a`） |
| `independent_audit.md`, `phase_a/` | Phase A の監査 |
| `full/fit_manifest.csv`, `full/fits_progress_*.jsonl` | 80 回の fit（状態ファイルは `D:\tento\kennkyu_artifacts\af6_laplace_confirmation_20261114\fits\`、sha256 は manifest にある） |
| `full/candidate_scores.csv`, `full/column_selection.csv`, `full/select_progress_*.jsonl`, `full/af2_auto_columns.csv` | 候補の score と列ごとの選択 |
| `full/failure_registry.csv` | 失敗 3 件 |
| `full/runinfo_*.json` | stage と shard ごとの runinfo |
| `statistical_analysis.csv`, `paired_comparison.csv`, `approximation_diagnostics.csv` | 事前登録した解析 |
| `full/analysis/` | 列の注記、正解率、`fig1_dr_distribution.png`、`fig2_laplace_vs_ti.png` |
| `verification/` | 独立な再計算、データの独立性、事後の記述表（`posthoc_status_by_true_family.csv`、`compute_time.csv`） |

## 11. 再現のコマンド（root で、Python 3.13、OMP_NUM_THREADS=2）

```
python expfam/src/experimental/run_af6_laplace_confirmation.py --fits   --shard i/5   # i=0..4
python expfam/src/experimental/run_af6_laplace_confirmation.py --select --shard i/5   # i=0..4
python expfam/src/experimental/run_af6_laplace_confirmation.py --collect
python expfam/src/experimental/run_af6_laplace_confirmation.py --analyze
python expfam/src/experimental/verify_af6_results.py
python expfam/src/experimental/verify_af6_results.py --tables
python -m pytest expfam/src/experimental/test_af6_laplace_confirmation.py
```

環境（python、numpy、scipy、platform、requirements の sha256）は `full/runinfo_*.json` に記録している。
