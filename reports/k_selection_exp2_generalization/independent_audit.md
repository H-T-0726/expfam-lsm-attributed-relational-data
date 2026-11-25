# 8,000 fit の独立監査

- 作成日：2026-11-25。対象 commit `b8f2e3a`（本実験の結果）。対象は `expfam/results/k_selection/k_exp2_generalization_20261125/`。
- 監査スクリプト：`tools/research_audit/audit_k_exp2_independent.py`。本実験の後に、この監査のために書いた。
  - 事前に決めた解析スクリプト（`analyze_k_exp2_generalization.py`）を import しない。
  - 集計に設計モジュール（`k_exp2_design.py`）を使わない。
  - 定数は凍結した `protocol.json` から読み、C_Q の定義と選択規則をこのスクリプトの中で書き直した。
  - 設計モジュールを使うのは、データの再生成のスポットチェックだけである（生成器の再現性を確かめるため）。
- 出力：`.../independent_audit/audit_checks.json`、`independent_selection.csv`、`independent_cell_counts.csv`。
- **結論：監査スクリプトの 31 項目すべてに合格し、NPZ の完全な SHA-256 の照合（§4）も合格した。数値の不一致も、provenance の問題もない。**
- 下の表では、スクリプトの 2 項目（全 fit が OK、`failures.csv` が空）を 1 行（#10）にまとめた。#31 はスクリプトの外で行った確認である。

## 1. 確認項目と結果

| # | 確認 | 結果 |
|---|---|---|
| 1 | protocol の SHA-256 が runinfo の値と一致 | 合格（`14757d4a…ad`） |
| 2 | manifest の hash が protocol の値と一致（凍結時の定義＝行の JSON の SHA-256 を、このスクリプトで書き直して計算） | 合格 |
| 3 | 実行時の commit `d4e5eb9` のコードが、凍結した 13 個の code hash と一致 | 合格（改行コードについては §2） |
| 4 | 現在の作業ツリーのコードが凍結時と同じ | 合格 |
| 5 | runinfo：8 worker、BLAS 1 スレッド、8,000 fit の計画 | 合格 |
| 6 | `run_events.jsonl` が start と finished_batch の 2 行だけ（再開なし）。同じ protocol hash と head | 合格 |
| 7 | manifest が、protocol の格子（4 系列 × 2 regime × 5 K_TRUE × 10 データ × 10 K × 2 初期値）と seed 規則に一致 | 合格（8,000 行） |
| 8 | per_fit の seed が manifest と一致 | 合格（不一致 0） |
| 9 | 8,000 fit がちょうど 1 回ずつ記録されている | 合格 |
| 10 | 全 fit の状態が OK、`failures.csv` は空 | 合格 |
| 11 | retry・置換・seed 救済・NaN の合計が 0 | 合格 |
| 12 | family の列が系列の定義と一致 | 合格 |
| 13 | 各 fit の反復数がちょうど 10（`mstep_q_history.csv`） | 合格 |
| 14 | Q_strict が 10 反復目の M-step 後の Q とビット単位で一致 | 合格（8,000/8,000） |
| 15 | 生成パラメータ（w、w0、F の大きさ）が protocol の規則と一致（許容差 10⁻¹²） | 合格 |
| 16 | Poisson-Y で \|w\| < 1/4 | 合格 |
| 17 | パラメータ数 p = Kd − K(K−1)/2 + d·[G-X] + [G-Y] | 合格 |
| 18 | C_Q = −2 Q_strict + p ln n | 合格（差の最大 0.0。8,000 件すべてビット単位で一致） |
| 19 | 1 つのデータの 20 fit で X・Y が同一 | 合格（400 データ） |
| 20 | 選択（2 初期値の非加重平均、差 10⁻⁹ 以内は小さい K）が、事前に決めた解析と一致 | 合格（初期値ごとの選択も含めて不一致 0） |
| 21 | 10⁻⁹ 以内の同点がない | 合格（同点規則は一度も使われていない） |
| 22 | regime a：200/200 正解 | 合格 |
| 23 | regime b：176/200 正解（過小 24） | 合格 |
| 24 | 過小 24、過大 0 | 合格 |
| 25 | 不完全なデータがない | 合格（400 データ） |
| 26 | K_TRUE = 3 の regime a・b が同一 | 合格（4 系列 × 10 データ × 20 fit で、C_Q・データ hash・w・F の大きさがすべて一致） |
| 27 | 重複を除いた数（下記） | 記録 |
| 28 | fit の状態（npz）が全 fit に存在 | 合格（8,000/8,000） |
| 29 | データの再生成のスポットチェック（8 データ） | 合格（X・Y の hash が一致） |
| 30 | `mstep_q_decreases` の合計 | 1（記述。ゲートにしない） |
| 31 | NPZ の完全な SHA-256（§4。スクリプトの外） | 合格（8,000 件の先頭 16 桁が per_fit と一致） |

## 2. 改行コードについて（provenance 上の注意。問題ではない）

- 凍結した code hash は、Windows の作業ツリー上のバイト列（`core.autocrlf=true` で CRLF に変換されたもの）から計算されている。
- git の blob は LF で保存されている。そのため、blob をそのまま hash すると 12 ファイルで値が違う。
- LF を CRLF に変換して hash すると、12 ファイルすべてが一致した。
- `k_exp2_design.py` は作業ツリーでも LF のままなので、blob をそのまま hash して一致した。
- 監査スクリプトは、両方の hash を計算して、どちらで一致したかを記録している。
- **注意**：別の OS や別の autocrlf 設定で checkout すると、凍結した code hash と一致しない。その場合も、内容は同じことを、この 2 通りの比較で確認できる。

## 3. 重複の扱い（K_TRUE = 3）

- regime a・b の K_TRUE = 3 は、設計上同一の条件である（w = 1 = √(3/3)、√(d/6) = √(0.5d/3)）。結果もビット単位で一致した。
- したがって、**異なる条件は 400 − 40 = 360 である。**
- 正解数を独立な証拠として数える場合は、次のように書く。
  - regime a：200/200
  - regime b（K_TRUE = 3 を除く）：136/160
- 「regime b で 176/200」は、regime a と同じ 40 データを含む数である。

## 4. fit の状態（NPZ）の保全

| 項目 | 値 |
|---|---|
| 場所 | `.../main/states/`（gitignore。PUBLIC リポジトリには入れない） |
| 件数・容量 | 8,000 ファイル、120,044,393 bytes |
| 完全な SHA-256 の一覧 | `.../main/states_sha256_manifest.csv`（git に入れる。per_fit の先頭 16 桁と全件一致） |
| 退避用アーカイブ | `D:/tento/research_backups/k_exp2_generalization_20261125/k_exp2_main_states_20261125.zip`（無圧縮、121,890,191 bytes） |
| アーカイブの SHA-256 | `26e6d8b51d4afda15781ad65a65ecb2f77da348d4cb069c6eff04c04749a6c6d` |
| アーカイブの確認 | 8,000 メンバーすべてを読み戻し、SHA-256 が一覧と一致 |
| **保全状況** | **ローカルのみ（作業ツリーと同じ D ドライブ）。別の媒体へのコピーは、まだ行っていない** |
