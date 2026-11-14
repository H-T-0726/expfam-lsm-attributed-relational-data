# AF7：AF6 で真 Poisson 列を CMP2 と誤選択した原因の分解（探索的・事後解析）

- 系列は lineage E＋PR #129 の generic model（prototype、本文には採用不可）。
- **すべて探索的な事後解析である。**
  - 新規の MCEM は 0 回、新しいデータや seed も使っていない。
  - AF1〜AF6 の成果物は変更していない。
  - 正解ラベルは記述（正解率・AUC・遷移）にだけ使い、閾値は調整していない。
- 基にした実行は AF6 `75012e8`（`af6_laplace_confirmation_20261114/full/`）と AF5 `a598bb5`。
- スクリプトは `expfam/src/experimental/analyze_af7_cmp_bias.py`。

## 何を計算したか

| 部分 | 入力 | 内容 |
|---|---|---|
| Part 1（`p1_*`） | AF5/AF6 の保存 CSV のみ | 列ごとの Δ_L、Δ_J、Δ_profile = Δ_L − Δ_J、Δφ_J、Hessian 項、対象列とそれ以外の列への分解、基準どうしの遷移 |
| Part 2（`p2_*`） | AF6 の保存状態（npz）と、生成し直した AF6 データ | 固定した Z（真の Z、anchor の mode）での列尤度比、Pearson 分散比、Z の移動量、データの特徴、真の η での KL |
| Part 3（`p3_*`） | AF6 データと真の θ | anchor を真の θ に置き換えて、同じ局所基準（L、J、φ_J）を計算 |
| summary | Part 1〜3 | 基準ごとの正解率、AUC（データセット単位の cluster bootstrap 2000 回、seed 20261117） |

記号の定義は、スクリプト冒頭の docstring にある。

- S_J と φ_J は、AF5 の joint MAP 版の f_J（Φ(Ẑ(f), f) を最大にする f）で評価した値である。
- h_J = S_J − φ_J − (nK/2) log 2π。

## 主な数値（AF6、sparse、`summary_criteria_accuracy_auc.csv`）

| 基準 | 真 Poisson の正解率 | 真 CMP2 の正解率 | AUC [95% CI] |
|---|---|---|---|
| 真の Z で固定（f は列ごとに最尤） | 0.950 | 1.000 | 0.993 [0.984, 0.999] |
| anchor の mode Z で固定（plug-in） | 0.300 | 1.000 | 0.857 [0.816, 0.895] |
| Q（同じ anchor） | 0.975 | 0.647 | 0.930 [0.899, 0.955] |
| φ_J（joint MAP の log joint、Hessian なし） | 0.076 | 1.000 | 0.611 [0.543, 0.683] |
| J（f_J での Laplace、AF6 の E） | 0.898 | 0.815 | 0.919 [0.892, 0.943] |
| L（Laplace、AF6 の A） | 0.797 | 0.908 | 0.947 [0.925, 0.967] |
| TI（f_L での Z 積分） | 0.780 | 0.933 | 0.947 [0.926, 0.967] |
| L（真の θ を anchor） | 0.824 | 0.950 | 0.968 [0.950, 0.983] |
| J（真の θ を anchor） | 0.908 | 0.817 | 0.941 [0.918, 0.963] |

- dense の値も同じファイルにある。L の AUC は 0.991、Q は 0.988 である。
- dense の TI は |Δ|<2 の 37 列に限られ、比較できない。
- Part 3 で失敗したのは 2 件（rep09 sparse の列 3。AF6 でも η≈51 で失敗した列）で、再実行はしていない。

## 成果物

| ファイル | 内容 |
|---|---|
| `p1_column_decomposition.csv`, `p1_summary_by_truth.csv`, `p1_transitions.csv` | Part 1 |
| `p2_column_diagnostics.csv` | Part 2 |
| `p3_true_theta_candidates.csv`, `p3_true_theta_columns.csv`, `p3_true_theta_progress_*.jsonl` | Part 3 |
| `summary_criteria_accuracy_auc.csv`, `summary_features_by_L_correct.csv`, `summary_column_table.csv` | 集計 |
| `runinfo_summary.json` | 環境（summary を実行した時点。各 Part は同じ head で実行し、そのときスクリプトは未 commit） |

## 再現（repository root、OMP_NUM_THREADS=2）

```
python expfam/src/experimental/analyze_af7_cmp_bias.py --part1
python expfam/src/experimental/analyze_af7_cmp_bias.py --part2
python expfam/src/experimental/analyze_af7_cmp_bias.py --part3 --shard i/5   # i=0..4
python expfam/src/experimental/analyze_af7_cmp_bias.py --part3-collect --summary
```

Part 2 と Part 3 は、AF6 の fit 状態（`D:\tento\kennkyu_artifacts\af6_laplace_confirmation_20261114\fits\`）を使う。

解釈は、AF7 の報告（チャット）で FACT / DERIVED / OBSERVED / HYPOTHESIS を区別して示した。この README には主要な数値だけを載せている。
