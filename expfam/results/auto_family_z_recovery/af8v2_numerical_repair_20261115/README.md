# AF8-v2：loading を積分した分布選択の数値最適化の修正と再 pilot（探索的、停止）

- 系列：lineage E＋PR #129 の prototype。manuscript evidence ではない。親は AF8（`0643fa5`）で、AF8 の成果物は変更していない。
- 目的：AF8 の pilot が停止した数値的な原因（鞍点への収束、BFGS の LinAlgError、Z の問題が凹でない）を修正し、同じ 648 セル（rep01–03）で数値面だけの pilot をやり直す。
  分布選択の精度を上げることは目的ではない。正解率・AUC・ラベルとの対応は計算していない。
- 主なスコア：E2（nested Laplace）。E（joint Laplace）と IS は診断用。prior は AF8 と同じ（主：B_sg2_0.667、ほかに感度分析用の 8 種）。
- この量は Bayesian evidence でも BIC でもない（θ_{−j} を固定している）。

## 結果

**OBSERVED:** S4（IS の draw が失敗したセル 12.5% > 5%）が作動した。事前に登録した規則によって、本実験の数値条件は満たしていない。40 データの本実験は行っていない。

ほかの規則（S1–S3、S4b、S5–S7）は通過した。詳細と原因は `numerical_audit_v2.md` を参照。

## ファイル

| ファイル | 内容 |
|---|---|
| `protocol.json` | pilot の前に凍結した protocol（sha256 57f0a379…、`ac0fb25`） |
| `mathematical_notes_v2.md` | E2 を主とすることについての数理メモ（pilot の前に作成） |
| `pilot/candidate_scores.csv` | 648 セルの D、E2、E、IS、開始点の状態、多峰性、決定性、時間 |
| `pilot/pilot_pairs_v2.csv` | Δ = M_C − M_P（E2、E、IS、刻み幅を変えた E2）と、ラベルを使わない判定状態 |
| `pilot/pilot_numerical_audit_v2.json` | 停止規則 S1–S7 と診断量 |
| `pilot/pilot_fd_check_v2.csv` | S7：FD の勾配と Hessian の検証 |
| `pilot/pilot_failure_diagnosis_v2.csv` | 停止原因の診断（失敗 draw、有効な外側の開始点がないセル） |
| `pilot/af8_vs_v2_comparison.csv` | AF8 と v2 の数値の比較（定義の違いは `numerical_audit_v2.md` §4） |
| `pilot/runinfo_*of5.json`、`pilot/progress_*of5.jsonl` | 実行の記録（git_head ac0fb25、tracked のファイルに未 commit の変更なし） |
| `numerical_audit_v2.md` | 数値監査と独立監査の結果 |

## 再現

```
python expfam/src/experimental/run_af8v2_integrated_loading.py --pilot --shard i/5   # i = 0..4
python expfam/src/experimental/run_af8v2_integrated_loading.py --collect-pilot
python expfam/src/experimental/analyze_af8v2_integrated_loading.py --pilot-audit
python expfam/src/experimental/analyze_af8v2_integrated_loading.py --pilot-diagnose
python -m pytest expfam/src/experimental/test_af8v2_integrated_loading.py
```

環境変数は OMP_NUM_THREADS=2 と PYTHONIOENCODING=utf-8。解析コードは pilot の後に 3 か所を修正した（JSON のキー、S1 の定義を protocol の文言に合わせたこと、診断の追加）。その結果も同じ commit に含めている。
