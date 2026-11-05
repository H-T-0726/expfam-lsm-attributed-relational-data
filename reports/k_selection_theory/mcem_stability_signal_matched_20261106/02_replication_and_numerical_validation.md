# 02 再現性と数値的検証（2026-11-06）

- **位置づけ**：系列 E（`numerics_mode="consistent"`、prototype）。修士論文への採用は未決定。
- 一次資料：`expfam/results/k_selection/mcem_stability_signal_matched_20261106/{gates,rq1_AB,rq1_CD,rq2,analysis}/`。fit 状態（外部 NPZ、commit しない）は `D:\tento\kennkyu_artifacts\mcem_stability_signal_matched_20261106\{rq1_AB,rq1_CD,rq2}_states.npz`。sha256 は各ステージの `runinfo.json`（rq1_AB `9652f082…`、rq1_CD `d2f1d909…`、rq2 `61a1a7ea…`）。再生成は `python tools/research_audit/run_mcem_stability.py --rq1 --conds A B`（ほかのステージも同様）で行う。
- 事前登録：`gates/prereg.json`。内容 hash `4059f064…` を全ステージの runinfo に記録した。prereg を commit したあとで本番 fit を始めた（ancestry の確認は監査 07 節）。

## 1. ゲート（本番の前に実施。FACT）

| ゲート | 結果 |
|---|---|
| G1 RQ1 データが PR #155 と同じか | PASS（32 条件 × データセットで X/Y/Z の hash が一致） |
| G2 RQ2 データ | PASS（rank 5、FᵀF の固有値がすべて 2.5、Poisson rate、台、平均辺密度の較正） |
| G3a 再実行で EMC の A rep1 K5 s1 を再現できるか | PASS（8・16・24・32 反復で状態の差が 0.0） |
| G3b 独立 MC 評価が推定経路に影響しないか | PASS（16 反復時点の状態が、評価ありとなしでビット単位で一致） |
| G3c パイロットの評価 | Laplace 6/6 が OK、恒等式の誤差の最大 5.7e-13 |
| G4 計算時間 | 1.353 s/fit-iteration。計画 11,344 fit-iteration ≤ 上限 12,000。RQ2 の実施条件を満たした |

## 2. 本番の実行（FACT）

| ステージ | fit | 反復 | fit-iteration | 状態 | 壁時計 |
|---|---:|---:|---:|---|---:|
| RQ1 A・B | 64 | 64 | 4,096 | COMPLETED | 5,677 s |
| RQ1 C・D | 64 | 64 | 4,096 | COMPLETED | 5,699 s |
| RQ2 | 96 | 32 | 3,072 | COMPLETED | 4,644 s |
| ゲート | — | — | 80 | PASS | — |
| 合計 | 224 | — | **11,344** | — | 並列 3 プロセスで約 1.6 時間 |

- fit の失敗（例外・NaN による再開要求）：**0 件**。リトライも seed の取り直しもない。
- **8 反復時点の再現**：RQ1 の 128 fit すべてで、8 反復時点の状態が保存状態と一致した（**128/128 MATCH**、REPLICATION_FAIL 0）。A は EMC の状態、B/C/D は PR #155 の状態と照合した。したがって、RQ1 の 64 反復の軌跡は、PR #155 で評価した fit そのものの継続である（同じ seed で 1 反復目から再実行。最初の 8 反復は同じ乱数 prefix を消費する）。
- 事前登録では 8 反復時点の **C_Q の一致**も確認対象としていたが、本番のコードは状態の配列だけを照合していた。独立レビューの指摘を受けて追加で照合し、128/128 fit で C_Q が PR #155 の `fit_criteria.csv` の値と**完全に一致**した（差の最大 0.0）。
- ゲート G3b は 16 反復時点の状態だけを比べていた（8 反復時点の評価の影響しか排除しない）。追加で、A rep1 K5 s1 を独立評価**なし**で 64 反復まで再実行し、本番（評価あり）の 64 反復時点の状態と比べた。Z 標本・F・σ・スカラーのすべてで差は 0.0（64 fit-iteration を追加。合計 11,408 ≤ 12,000）。
- 上の 2 つの追加確認の出力：`expfam/results/k_selection/mcem_stability_signal_matched_20261106/review_followup/`（`tools/research_audit/review_followup_mcem_stability.py`、事前登録外）。
- RQ2 は新しいデータなので照合対象がない（NOT_APPLICABLE）。

## 3. 数値的な検証（OBSERVED）

| 確認 | 結果 |
|---|---|
| C_Q の分解の閉包 C_Q = −2(Q_X+Q_Y+Q_Z) + p_K ln n | 誤差の最大 2.2e-12 |
| slack の会計 S_Q = 2ΔQ_X + 2ΔQ_Y − n(1+ln 2π) − (d−K) ln n | 誤差の最大 1.4e-12 |
| Laplace の恒等式 C_Q − C_Lap − 2H_Lap = termA + 2R3 | fit ごとの誤差の最大 1.6e-12 |
| Laplace の状態 | 768 checkpoint 中 767 が OK。**1 件 NOT_STATIONARY**（RQ1 A rep6 K4 s1、16 反復、grad_inf = 2.1e-7）。この dataset・checkpoint の C_Lap は INCOMPLETE として扱い、補完していない（C_Lap の対応比較は 16 反復だけ 7 対） |
| 重複キー | 0 |

- 8 反復時点の値は PR #155 の報告値と一致した：C_Q slack の中央値 A/B/C/D = −128.7/−66.3/−81.8/−12.6、対応差の中央値 B−A +78.6、C−A +35.3、D−A +129.1、C_Lap で K=5 を選ぶ件数 3/6/4/7。
- 独立 MC 評価（L_EVAL=20、推定とは別の乱数 `[7600000, model_seed, checkpoint]`）の slack の MC 標準誤差は、64 反復で中央値 約 6（条件により 5.99〜6.26）。

## 4. この節から言えないこと

- 再現の一致は、**同じコード・同じ環境での**ビット単位の一致である。別の BLAS・別の環境での一致は確かめていない。
- 恒等式・会計の誤差が小さいことは、C_Q・C_Lap が**定義どおりに計算されている**ことを示すだけで、基準として正しいことは示さない。
