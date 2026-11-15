# AF8-v3：内側・外側の最適化を「証明書」で受理する数値安定化と再 pilot（探索的）

- 系列：lineage E＋PR #129 の prototype。manuscript evidence ではなく、正式手法でもない。
  親は AF8-v2（`a5cffa1`、PR #172、停止）で、AF8 と AF8-v2 の成果物は変更していない。
- この protocol は **v2 の pilot 結果を見た後に作った改訂版** である。AF8 の事前登録と同じものとして扱わない。
- 目的：v2 が停止した数値的な原因（φ の丸めで Armijo が止まる、FD 勾配の雑音、IS の draw の失敗）を数理的に修正し、
  同じ 648 セル（rep01–03）で数値面だけの pilot をやり直す。
  - 分布選択の精度は評価していない。正解率・AUC・ラベルとの対応は計算していない。
- 主なスコア：E2（nested Laplace）。E（joint Laplace）と IS は診断用。データ、prior、開始点、IS の draw と seed は v2 と同じ。
- この量は Bayesian evidence でも BIC でもない（θ_{−j} を固定している）。
- 変えたのは、最適化の終わり方と受理の基準だけである。根拠と閾値は `mathematical_notes_v3.md` にある（pilot 前に commit）。

## 結果（OBSERVED）

- 停止規則 S1–S9（S2b、S3b、S4b、S4r を含む）はすべて通過した。閾値は v2 から緩めていない。
- S4 を v2 と同じ定義で数えた値（cold start で 1 回でも失敗した draw を含むセルの割合）は、0/72 だった（v2 は 9/72）。
- IS の draw の失敗は 0/28,800、救済した draw も 0 だった。

数値と v2 との比較は `numerical_audit_v3.md` にある。40 データの本実験は行っていない。行うかどうかは人間が判断する。

## ファイル

| ファイル | 内容 |
|---|---|
| `protocol.json` | pilot 前に凍結した protocol（sha256 c853c009…、`e183c45`） |
| `mathematical_notes_v3.md` | 証明書型の受理の根拠、閾値、限界（pilot 前に作成） |
| `regression/known_failed_draws_v2.csv` | v2 で失敗した 9 draw を、v2 の seed から完全に同じ形で再生成したもの |
| `regression/known_failures_v3_regression.csv` | 9 draw と、外側で失敗した 2 セルを v3 で再評価した回帰結果（pilot の前に実行） |
| `regression/v2_failure_summary.json` | v2 の失敗の要約 |
| `pilot/candidate_scores.csv` | 648 セルの D、E2、E、IS、証明書の量、開始点の状態、時間 |
| `pilot/pilot_pairs_v3.csv` | Δ = M_C − M_P（E2、E、IS、刻み幅を変えた E2）と、ラベルを使わない判定状態 |
| `pilot/pilot_numerical_audit_v3.json` | 停止規則と診断量 |
| `pilot/pilot_fd_check_v3.csv` | S7：FD の勾配と Hessian の検証 |
| `pilot/pilot_is_draw_log_v3.csv` | 救済した draw、または失敗した draw の記録。今回は該当がなく、ヘッダーだけである |
| `pilot/v2_vs_v3_comparison.csv` | v2 と v3 の比較。定義の違いは列名に書いてある |
| `pilot/runinfo_*of5.json`、`pilot/progress_*of5.jsonl` | 実行の記録。git_head e183c45、tracked のファイルに未 commit の変更なし、`.venv` の Python 3.13.14 |
| `numerical_audit_v3.md` | 数値監査と独立監査の結果 |

## 再現

```
.venv/Scripts/python.exe expfam/src/experimental/analyze_af8v3_integrated_loading.py --v2-summary
.venv/Scripts/python.exe expfam/src/experimental/analyze_af8v3_integrated_loading.py --make-fixtures
.venv/Scripts/python.exe expfam/src/experimental/analyze_af8v3_integrated_loading.py --regression
.venv/Scripts/python.exe expfam/src/experimental/run_af8v3_integrated_loading.py --pilot --shard i/5   # i = 0..4
.venv/Scripts/python.exe expfam/src/experimental/run_af8v3_integrated_loading.py --collect-pilot
.venv/Scripts/python.exe expfam/src/experimental/analyze_af8v3_integrated_loading.py --pilot-audit
.venv/Scripts/python.exe -m pytest expfam/src/experimental/test_af8v3_integrated_loading.py
```

環境変数は OMP_NUM_THREADS=2 と PYTHONIOENCODING=utf-8。pilot の後に、コード・protocol・閾値は変更していない。
