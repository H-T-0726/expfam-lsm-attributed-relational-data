# K 選択基準の同一 fit 比較 — 実験計画（Issue #128, 2026-10-06）

- 位置づけ: **EXPERIMENT PLAN**（Human Gate 2026-10-06 で承認済みの scope を、実行前の固定事項として記録する）。
- 系列: lineage E（objective-consistent prototype, `numerics_mode="consistent"`）。
  **現時点では experimental prototype であり、修論本文への採用は人間がまだ決定していない（MANUSCRIPT_ADOPTION_STATUS = NOT_DECIDED_BY_HUMAN）。**
- 主題は「候補 K の中から、どの基準を使えば K_TRUE を選べるか」だけ。分布族の自動選択は扱わない（family は生成時の真値に固定）。
- ラベル: FACT / DERIVED / OBSERVED / INTERPRETATION / HYPOTHESIS / UNRESOLVED。

## 1. 問い

真の潜在次元 K_TRUE が既知の人工データで、**同じデータ・同じ候補 K・同じ fitted model** の上で、
基準ごとにどの K が選ばれるか、候補 K に対する基準の曲線がどう違うか。
主目的は K_TRUE の回復であり、予測の最大化ではない。

## 2. 比較する基準（名称は固定）

| 名前 | 役割 | 定義 | 呼んではいけない名前 |
|---|---|---|---|
| `C_Q` | 主 | `−2 Q_strict + p_K ln n = D_K + P_Z + P_θ`（Q-based complete-data / ICL-type） | Schwarz BIC |
| `C_Lap` | 主 | `−2 ℓ_Lap + d_K ln N = fit_mode + volume + P_θ`（Laplace-evidence candidate、Phase 9 Candidate B） | true / correct BIC、exact evidence |
| `S1` | 参考（A のみ） | held-out 関係データの plug-in mean log score（予測目的。K_TRUE 回復の理論保証なし） | — |
| `S3` | 診断（A のみ） | 最終 Z での条件付き尤度 + 罰則（既存値） | 原論文 BIC |
| `C_cond_MC` / `C_cond_last` / `C_cond_mode` | 診断（B のみ） | `−2 ℓ_data(Z*) + p_K ln n`、Z* = E-step サンプル平均 / 最終サンプル / joint mode | 原論文 BIC（Z の評価点を原論文から一意に特定できない） |

- **FACT**: どの基準も同じ `p_K`（A: `Kd − K(K−1)/2`、B: `Kd − K(K−1)/2 + d`）と `ln n` を使う。基準間の違いは当てはまり項・Z の扱いだけ。
- **DERIVED**: `C_cond_MC = D_K + P_θ = C_Q − P_Z`、`C_cond_mode = fit_mode + P_θ`（`k_criterion_panel.py` のテストで確認）。

## 3. 実験 A1: canonical clean n-sweep に C_Lap を追加

| 項目 | 値 |
|---|---|
| 元実験 | `expfam/results/k_selection/clean_true_k_asymptotics_20260904/`（protocol hash `547880a1…`） |
| 条件 | K_TRUE ∈ {1,3,5}、n ∈ {50,75,100,150}、候補 K 1..7、replicate 4/4/8、start 1,2、Poisson-X / Bernoulli-Y、d=15、L=5、8 反復、test_ratio 0.20 |
| seed | 既存の data / split / model seed 関数をそのまま使う |
| K_TRUE=5 の状態 | 外部 npz（`em_convergence_sensitivity_20260905/checkpoint_states.npz`, sha256 `5e4390f1…af84`）の checkpoint 8 を read-only 利用。対応表は commit `dad67b1` の `fit_results.csv` |
| K_TRUE=1,3 の状態 | 既存 seed で fit を再現（`failure_policy="fail_fast"`） |
| 照合（再現） | heldout / q_strict / S2(C_Q) / S3 / num_params が履歴 CSV と **bit 単位で一致**。1 件でも不一致なら停止 |
| 照合（npz） | 同じ 4 値が rtol 1e-9 / atol 1e-6 以内（監査済み `audit_em_convergence_sensitivity.py` の許容値を継承） |
| C_Lap | θ̂ = fit の最終値、Z0 = fit の Z_est、grad_tol 1e-8、max_iter 200、ridge / jitter / restart なし、N = n、Y は fit と同じ学習ペア |
| 選択規則 | 2 start の平均、tie 1e-12 以内は小さい K（既存規則）。補助に start ごとの選択 |
| INCOMPLETE | C_Lap がどこかの (K, start) で非 OK のデータセットは C_Lap 選択を INCOMPLETE にする（補完なし） |
| 出力 | `expfam/results/k_selection/clap_clean_nsweep_20261006/{pilot,full}/`、状態は `D:\tento\kennkyu_artifacts\clap_clean_nsweep_20261006\` |
| n=300 | **実行しない**（既存 model seed 式で (K1,n300)=(K3,n100)、(K3,n300)=(K5,n100) が衝突する。新しい seed 規則も導入しない） |

**A のパイロット gate**: K3/n50/r1（再現 14 fits）と K5/n50/r1（npz 14 fits）で、再現一致、C_Lap がすべて OK で有限、選択の再構成、schema、provenance、履歴 hash の不変。

## 4. 実験 B: 原論文型 generator での基準比較（原論文 Experiment 2 の再現ではない）

| 項目 | 値 |
|---|---|
| generator | `reproduction/src/data_generator.py::set_true_params`（変更なし、var_f=5, uniq=0.1, w_scale=3。w0・w はデータセットごとにランダム） |
| 条件 | n=150、d=15、K_TRUE ∈ {1,3,5,7,9}、候補 K 1..10、Gaussian-X / Bernoulli-Y、全ペア（held-out なし） |
| 推定 | `DualExpFamLSMConsistent`、L=5、8 反復、2 start（平均）、`failure_policy="fail_fast"` |
| replicate | R = 5（500 fits）。R=10 への拡張用 replicate 6..10 の seed は今ここで固定するが、本 Issue では実行しない |
| data seed | `1980 + 100·K_TRUE + 1000·(rep−1)`（rep 1 は historical reproduction と同じ seed） |
| model seed | `2000000 + 100000·K_TRUE + 1000·rep + 10·K + start`（全組で重複なしをテストで確認） |
| 失敗 | EM 例外・非有限 C_Q は FAILED として記録し、そのデータセットは全基準 INCOMPLETE。C_Lap 非 OK は C_Lap と C_cond_mode のみ INCOMPLETE。retry / replacement / seed 変更なし |
| B のパイロット gate | K_TRUE=3, rep 1（20 fits）で FAILED なし、C_Q / C_cond_MC / C_cond_last が有限、C_Q の再構成、C_Lap 20/20 OK。満たさなければ人間の判断を待つ |
| 出力 | `expfam/results/k_selection/paper_condition_k_criteria_20261006/{pilot,full}/` |

**historical reproduction との違い（report に必ず書く）**: 推定器（LatentStructuralModel ではなく consistent model）、L=10 / 30 反復 / 3 試行の最良ではなく L=5 / 8 反復 / 2 start の平均、反復中の最良値ではなく最終状態、各 K_TRUE 1 データセットではなく 5 データセット。

## 5. 保存するもの

各 fit: 全基準の値、C_Q の分解（Q_X, Q_Y, Q_Z, D_K, P_Z, P_θ）、C_Lap の分解（fit_mode, volume, penalty）と Laplace の診断（status, grad_inf, min eig, logdet, 反復数, 時間）、seed、retry/replacement 数。
各データセット × 基準: selected K、exact / under / over、tie 候補、最良と次点の差、start ごとの選択。
各データセット × (K → K+1): C_Q の fit gain と ΔP_Z, ΔP_θ、C_Lap の fit gain と Δvolume, Δpenalty（診断用。原因の証明ではない）。
provenance: protocol hash、git head、branch、Python / NumPy / SciPy、platform、時刻、入力 hash、状態ファイルの sha256、履歴成果物の hash（前後）。

## 6. 監査

runner も model コードも import しない独立 auditor（`audit_clap_clean_nsweep.py`、B 用も同様）で、
fit 数・候補 K の網羅・seed・再現一致・C_Q と C_Lap の閉包（C_Lap は Laplace の生成分から再計算）・selected K・exact/under/over・margin・start ごとの選択・summary の数・履歴成果物の hash・retry / replacement が 0 であることを再計算する。

## 7. 言い方の規則

- 「この固定条件の R データセット中 x 個で K = K_TRUE を選んだ」と書く。accuracy、recovery rate、一般的な優越性、一致性、「K 選択を解決した」とは書かない。
- C_Q が過小選択し C_Lap が K_TRUE を選んでも、「P_Z が原因」とは書かない。分解から言えるのは、各項が選択境界にどれだけ寄与していたかまで。
