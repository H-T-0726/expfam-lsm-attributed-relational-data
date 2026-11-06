# 02 運用上の追記 v2：メモリ不足からの再開（2026-11-06）

- 事前登録（`87cee59`、内容 hash `f2f49e5d…`）とゲート（`a2bf055`）は**変更しない**。この追記は、実行・保存・再開・計算資源の制御だけを変える。条件・L・seed・replicate・評価方法・判定規則は変えない（PREREG の hash は不変）。
- 正本：`expfam/results/k_selection/mcem_noise_separation_20261106/operational_addendum_v2.json`。本番の再実行の前に commit する。

## 1. 何が起きたか（FACT）

- 最初の実行では 8 プロセスを同時に動かした（Phase 3 A/B/C/D、Phase 4 L20・L50 × A/D）。約 1,000 秒後、システムのメモリが極端に不足したため Claude Code が停止した。
- 出力をステージの最後にまとめて書く設計だったので、**保存された結果は 0 件**。8 つの旧ステージディレクトリ（`phase3_*`、`phase4_*`）は空のまま（各 0 ファイル）で、記録として残し、使わない。

## 2. 原因（OBSERVED：測定）

| 測定（Phase 3 の 1 fit、A rep1 K5 s1） | BLAS 1 スレッド | BLAS 12 スレッド（既定） |
|---|---:|---:|
| private memory（コミット量） | 104 MB | **811 MB** |
| RSS のピーク | 114 MB | 119 MB |
| 1 fit の時間 | 31.5 s | 31.6 s |
| C_Q_L80・drift_share | 同一（ビット単位） | 同一 |

- OpenBLAS にスレッド数の制限がなく、1 プロセスあたり 12 スレッド分のバッファ（約 0.8 GB）を確保していた。8 プロセスで約 6.5 GB のコミットに加え、6 コアに 96 の BLAS スレッドが集中していた。
- この問題の行列は小さく、スレッドを 1 にしても速度も結果も変わらない。
- 当時、他のアプリがどれだけメモリを使っていたかは記録がない。不足の全量をこの原因だけで説明できるかは確かめられない（INCONCLUSIVE）。

## 3. 変更点

| 項目 | 内容 |
|---|---|
| runner | `tools/research_audit/run_mcem_noise_separation_v2.py`（実行の制御だけ）。fit ごとの計算は従来どおり `run_mcem_noise_separation.phase3_fit` / `phase4_fit` |
| 計算モジュールの変更 | (1) `probe()` が保存用に評価連鎖・1 ステップ更新の G・w0・w も返す。(2) `phase3_fit()` が probe 時点の F も保存する。(3) Phase 4 の 1 fit 分の処理を、そのまま `phase4_fit()` に移した。**計算・乱数の消費順は変えていない**（テスト T5・T6 で確認） |
| fit 単位の保存 | `exec_v2/<stage>/fits/<fit_key>/` に CSV と manifest.json、外部 `D:\tento\kennkyu_artifacts\mcem_noise_separation_20261106\exec_v2\<stage>\<fit_key>.npz` に状態。一時ファイルに書いて fsync し、`os.replace` で確定する。manifest は最後に書き、全ファイルの sha256、設定 hash（事前登録、計算コードの hash、seed・K・start・L・反復数、入力データの hash、スレッド設定）、status、時間、ピークメモリを記録する |
| 再開 | 全 fit を検証し、OK と FAILED（設定 hash と checksum が一致するもの）は飛ばす。**FAILED は再試行も seed の変更もしない**。それ以外は INCOMPLETE とし、残ったファイルは削除せずに改名して退避（`*.incomplete-<時刻>`）してから、その fit を 1 反復目から実行し直す。推定用の乱数状態を保存していないので、**fit の途中から再開したとは主張しない** |
| 状態の区別 | OK / FAILED / INCOMPLETE / NOT_RUN |
| スレッド | 実験プロセスの中だけ、OpenBLAS・OpenMP・MKL・NumExpr・vecLib を 1 スレッドに固定 |
| メモリの監視 | 各 fit の開始前に利用可能な物理メモリを確認する。3.0 GB 未満なら 30 秒ごとに最大 10 分待ち、回復しなければ新しい fit を始めずに止める（LOW_MEMORY、ステージは PARTIAL）。メモリは 30 秒ごとに CSV に記録する |
| 並列数 | Phase 3 は 1 プロセスで始める。最初の 2 fit で、ピーク RSS が 500 MB 未満かつ利用可能メモリが 6 GB 超のときに限り、最大 3 プロセスまで増やす。4 以上は使わない。Phase 4 は Phase 3 の保存が完了してから、L=20 を 1 プロセス、その後 L=50 を 1 プロセスだけで実行する（ほかの重い実験と同時に動かさない） |
| 時間の上限 | 起動ごとに 6 時間を超えたら新しい fit を始めない。未実行の fit は NOT_RUN とし、ステージを PARTIAL と報告する。結果を見て条件を省くことはしない |

## 4. 本番前のテスト

`tools/research_audit/test_resume_mcem_noise_separation.py` が、実際の fit を使って次を確認する。結果は `exec_v2/resilience_tests/report.json` に保存し、すべて PASS したあとで本番を始める。

1. 1 fit が完了すると、CSV・状態ファイル・有効な manifest が残る。
2. 実行中のプロセスを強制終了しても、完了した fit は有効なまま残り、中断した fit は INCOMPLETE と判定される。
3. 再実行では、検証済みの fit は飛ばし、未完了の fit だけを退避してから実行し直す。
4. 書きかけの一時ファイル、checksum の不一致、途中で切れた manifest、設定の不一致は、完了として扱わない。
5. 同じ seed と条件で実行し直すと、保存された行と配列を完全に再現する。
6. スナップショットの取得と保存は、MCEM の推定経路を変えない（8 反復時点の状態がビット単位で一致）。
7. Phase 4 の fit を runner 経由で保存でき、保存した状態から C_Q を再計算すると保存値と一致する。
8. Phase 3 の評価連鎖から C_Q(L=80) を再計算すると、保存値と一致する。
9. 集約の結果に重複がなく、未実行の fit は NOT_RUN として残る。
