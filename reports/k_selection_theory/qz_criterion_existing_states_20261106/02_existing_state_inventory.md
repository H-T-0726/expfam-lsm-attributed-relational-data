# 02 既存の推定状態の棚卸し（2026-11-06）

一次表：`expfam/results/k_selection/qz_criterion_existing_states_20261106/available_saved_states.csv`、`fits_scored_by_series.csv`。状態ファイルは `D:\tento\kennkyu_artifacts\` にあり、読むだけで変更していない。

## 1. 全候補 K で再評価できた状態（FACT）

| 研究 | X/Y | K_GEN | n | d | 候補 K | 保存されている潜在状態 | 使った基準 |
|---|---|---|---|---|---|---|---|
| canonical A1 n-sweep（#139） | Pois/Bern | 1, 3, 5 | 50, 75, 100, 150 | 15 | 1..7 | 5 標本、最後の標本、F、w0・w、Ẑ | A〜E（全部を状態から再計算） |
| PR #155 信号強度 B/C/D | Pois/Bern | 5 | 75（B は 150 も） | 15 | 1..7 | 5 標本、最後の標本、F、σ、w0・w（Ẑ なし） | A〜C は再計算、D・E は保存値 |
| d-sweep | Pois/Bern | 5 | 75 | 30, 60 | 1..7 | 同上 | 同上 |
| R2 | Pois/Bern | 5 | 50, 100, 150 | 30 | 1..7 | 同上 | 同上 |
| R1（L=10、8・16 反復） | Pois/Bern | 5 | 75 | 15, 30, 60 | 1..7 | 同上 | 同上 |
| #139 paper-type | Gauss/Bern | 1, 3, 5, 7, 9 | 150 | 15 | 1..10 | 5 標本、F、σ、Ẑ | 同じ panel コードで保存済みの値（再計算はしていない） |

- 再計算した状態の数：n-sweep 896、PR #155 448、d-sweep・R2 672、R1 1,344（合計 3,360 fit）。paper-type は保存値 500 fit。
- 各状態の data（X、Y、学習用ペアのマスク）は、元の実験と同じ生成関数と seed で作り直した。元の実験が保存したハッシュとの照合は、各生成関数の中で行われる（`run_clap_clean_nsweep.make_cell` など）。
- すべての canonical の状態で、保存されている Z_last は Z_samples[:, :, −1] と一致した（`integrity_checks.json`）。

## 2. 局所比較だけに使える状態

| 研究 | 候補 K | 使い方 |
|---|---|---|
| PR #156 RQ1（64 反復、A/B/C/D、n=75） | 4, 5 | checkpoint 8〜64 の保存行から A・B・D・E の 4→5 の余裕だけ（C は計算していない）。**K̂ は計算しない** |
| PR #157 の固定 θ プローブ | 4, 5 | 使っていない |

## 3. 使わなかった状態

- PR #156 RQ2（K_GEN ± 1 だけ、別の生成器）。
- `em_convergence_sensitivity_20260905`（この研究では状態と条件の対応をとっていない）。
- `z_point_estimator_e1`・`seed_separation_e2a`・`em_iterations_e2c`（K を変えていない）。

## 4. Ẑ（Laplace の結合モード）の扱い

- n-sweep と paper-type だけ Ẑ が保存されている。n-sweep では、Ẑ から再計算した −2 ln p(X, Y | Ẑ) と保存値 `lap_fit_mode` の差は最大 1.8e−12 だった。
- それ以外の研究は Ẑ が保存されていないので、D は保存値 `lap_fit_mode` + p_K ln n を使った。Laplace の計算が OK でなかった (K, start) を含むデータセットは、D と E を INCOMPLETE とし、補完していない。
