# 実行記録

- 作成日：2026-11-25。
- branch `experiment/k-selection-exp2-generalization-20261125`。worktree `D:/tento/kennkyu-kexp2`。基点 `origin/main` = `1b5030b2417a314927fb3a60d4d9016307f28d3b`（fetch 後に再確認）。
- **本実験：未実行。** protocol は未凍結（`preregistered_protocol.md` §0 の D1〜D6 が未決定）。protocol の SHA-256 もまだない。
- 環境：Python は `D:/tento/kennkyu/.venv`（Windows 11、12 論理 CPU）。

## 1. 実行したもの（すべて技術的な確認。K 選択の結果を含まない）

| # | 内容 | script | 出力 | MCEM | 基準の計算 |
|---|---|---|---|---|---|
| 1 | 生成器診断（4 系列 × 生成器 2〜4 種 × K_TRUE 5 × 5 draw = 150 draw） | `tools/research_audit/diag_k_exp2_generators.py` | `expfam/results/k_selection/k_exp2_generalization_20261125/generator_diagnostics/` | 0 | なし |
| 2 | fixed 系列での true_params の漏洩確認（4 family 組 × 4 variant、n=40, d=6, K=2, L=2, 3 反復） | `tools/research_audit/check_true_params_leak_fixed.py` | `.../true_params_leak_check/` | 16 fit（コード経路テスト） | なし（Q_strict の一致比較だけ） |
| 3 | 実行時間の計測（canonical matched Series O、K_TRUE=9 の 1 データ、候補 K = 1・10、lineage E と fixed、L=10、10 反復） | `tools/research_audit/timing_k_exp2_smoke.py` | `.../timing_smoke/timing_smoke.json` | 4 fit | **なし**（時間・有限性だけを記録） |

## 2. 結果（技術的なもの）

- 生成器：`mathematical_validity.md` を参照。canonical matched の Series B は 25/25 draw で生成器が停止した（Poisson-Y の分散発散）。
- true_params：4 variant すべてで Q_strict がビット単位で一致した（漏洩なし）。
- 実行時間（1 プロセス、BLAS のスレッド数は既定）：

  | 系列 | K=1 | K=10 |
  |---|---|---|
  | E | 93.2 s | 98.2 s |
  | fixed | 79.9 s | 89.8 s |

  4 fit とも有限で終了した。

## 3. 計算予算の見積もり（概算）

1 fit を約 95 s とする。8 worker で並列にしたとき、1 worker あたりの速度が同じと仮定した概算である（BLAS スレッドの競合で遅くなりうる）。

| 構成 | fit 数 | CPU 時間 | 8 並列の wall 時間 |
|---|---|---|---|
| primary（4 系列、R=10、T=2） | 4,000 | 約 106 h | 約 13〜16 h |
| ＋ PT 生成器の Series O（D5 b） | 5,000 | 約 132 h | 約 17〜20 h |
| ＋ 信号規則の感度分析（D3 c） | 9,000 | 約 238 h | 約 30〜36 h |
| C_Lap の評価 | 同じ fit 上（追加の MCEM なし） | fit あたり数秒〜数十秒（K=10 で nK = 1500 の Hessian） | — |

予算を減らす場合の影響：
- T=2 → 1：初期値の影響を測れなくなる（研究問いの 9 が答えられない）。fit 数は半分になる。
- R=10 → 5：K_TRUE ごとの正解率の精度が下がる（10 データで 1 件 = 10%）。
- L や反復数を減らす：元論文の条件（10・10）との対応が崩れる。

## 4. 本実験を始めるときの手順（判断の後）

1. D1〜D6 を決める。
2. runner を実装して test する。
   - 生成器の unit test
   - true_params を使わない初期化
   - 候補 K の間でデータが一致すること
   - K_TRUE = 1・9、候補 K = 1・10 の境界
   - Q の再計算
   - 保存と再読込
   - seed の再現性
   - fail_fast
3. `protocol.json` を凍結して commit し、SHA-256 をここに記録する。
4. 実行し、事前に決めた解析を行う。

---

## 5. 追記（2026-11-25、研究者の決定 D1〜D6 の後）

### 5.1 実装と test

| 種類 | ファイル |
|---|---|
| 設計規則（新規） | `expfam/src/experimental/k_exp2_design.py` |
| runner（新規） | `tools/research_audit/run_k_exp2_generalization.py` |
| 事前に決めた解析（新規） | `tools/research_audit/analyze_k_exp2_generalization.py` |
| test（新規） | `tools/research_audit/test_k_exp2_generalization.py` |

test は **54 件すべて合格**。次を確認した。
- Poisson-Y で \|w\| < 1/4、母集団平均が 1（解析式と独立な数値積分）
- Bernoulli-Y の w0 の 3 経路の一致と、MC による確認
- rank F と行ノルム、regime 間での Z・Q の共有
- 生成の決定性と、候補 K によらないこと
- 台とモーメント、Poisson-X の exp link
- 真値を渡さないこと、fail_fast の記録、非有限の Q
- CSV の往復と再開の集合
- K=1・10 の実際の EM、選択規則・同点・INCOMPLETE、Clopper–Pearson

解析スクリプトは、偽の 8,000 fit のデータだけで最後まで動作を確認した。

### 5.2 preflight（fit なし）

`expfam/results/k_selection/k_exp2_generalization_20261125/preflight/` に出力した。

- manifest は 8,000、fit_id は一意。
- data seed はデータごとに一意で、init・mc の seed とは重ならない。
- 400 データすべてで次を確認した。
  - rank F = K_TRUE
  - 行ノルムが目標値と一致
  - Z・Q が regime 間で共有されている
- 較正表は `calibration.json`。

### 5.3 技術 smoke（main と別の seed：data seed + 5,000,000）

`.../smoke/` に出力した。

- 構成：4 系列 × {K_TRUE=1, 9} × {候補 K=1, 10}（regime a）。K_TRUE=9・K=10 については regime b も加えた。計 20 fit、8 worker。
- 結果：20/20 が OK。retry・置換・seed 救済は 0。C_Q の再計算差は 0。
- 実行時間（8 並列で実行中の 1 fit）：平均で K=1 が 54.4 s、K=10 が 66.1 s、最大 85.0 s。wall 175 s。
- **見積もり：8,000 fit で CPU 約 134 h、8 worker で wall 約 16.7 h**（上限 45 h を下回る）。
- 空きメモリは約 6.9 GB、D ドライブの空きは 831 GB。
- **smoke の C_Q の値と選択 K は見ていない**（per_fit.csv には残っている）。

### 5.4 凍結

- `.../protocol/protocol.json`：SHA-256 `1a6a4484d3e2abd7eaf1abc651b126bfe6f7c472ff89e402719646059bfdc161`
- `.../protocol/manifest.csv`：8,000 行
- 凍結の commit は、この追記を含む commit（`git log` で確認する）。execute は、その commit の上で working tree が clean であることを確認してから開始する。

### 5.5 最初の execute の中断と再凍結（fit 0 件）

- 2026-11-25 01:01 に `c1755de` から execute を開始した。runinfo を書く段階で、protocol のパスを処理する行の例外（`Path.relative_to`）で停止した。**fit は 1 件も実行していない**（`per_fit.csv` は作られていない）。
- 開始の記録は `main_aborted_before_first_fit_20261125T0101/run_events.jsonl` に残した（空の `states/` だけを削除した）。
- 修正したのは runner の 1 行だけ（`proto_path.as_posix()`）。設計・seed・基準・解析・manifest は変えていない。
- protocol v1 との差は、runner の code hash と凍結時刻だけである。
- **再凍結**：`.../protocol_v2/protocol.json`、SHA-256 `14757d4a161712859601d52aba0e86308903a723f29baee0a6b3d5118ec5a7ad`。**本実験はこの v2 で実行する。** v1（`protocol/`）は記録として残し、使わない。
- test は 54 件すべて合格（再実行）。

### 5.6 本実験の実行（protocol v2）

- 開始は 2026-11-25 01:02:58、git head `d4e5eb9`、protocol v2（SHA-256 `14757d4a…`）。終了は 19:22:46（wall 65,988 s = 18.3 h。見積もり 16.7 h）。
- 再開なし（`run_events.jsonl` は start と finished_batch の 2 行）。
- **8,000 / 8,000 fit が OK。** FAILED 0、欠損 0、retry・置換・seed 救済・NaN は 0。`failures.csv` はヘッダーだけ。
- CPU 合計 146.4 h。1 fit は平均 65.9 s（40.7〜158.4 s）。
- 事前に決めた解析を `.../analysis/` に出力した。解析スクリプトとは別のコードで `per_fit.csv` から選択を独立に集計し直し、40 セルすべてで一致した。
- fit の状態 `main/states/`（8,000 npz、132 MB）は gitignore の対象で、commit していない。
- 結果は `results_report.md`。
