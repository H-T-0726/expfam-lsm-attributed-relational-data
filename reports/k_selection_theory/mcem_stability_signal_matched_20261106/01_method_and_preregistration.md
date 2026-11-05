# 01 方法と事前登録（2026-11-06）

- **位置づけ**：系列 E（`numerics_mode="consistent"`、prototype）。**修士論文への採用は未決定**（Human Gate）。PR #155（`research/k-selection-causal-analysis-20261106`、head `72f33cf`）に積み重ねる stacked な研究ブランチ `research/k-selection-mcem-stability-20261106`。
- **事前登録の正本**：`expfam/results/k_selection/mcem_stability_signal_matched_20261106/gates/prereg.json`（コード `tools/research_audit/run_mcem_stability.py` の `PREREG` と同一内容。内容 hash は gates.json と runinfo.json に記録）。本番 fit の前に commit する。
- C_Q（`calc_bic_dual`、CSV 列 `BIC`）は Q-based complete-data / ICL 型で **Schwarz BIC ではない**。C_Lap は周辺尤度のラプラス近似で **厳密な周辺尤度ではない**。

## 1. 研究問い

- **RQ1（最優先）**：PR #155 で 8 反復時点に観察した「F・w を強めると C_Q の 4→5 選択余裕が増える」効果は、同じ MCEM fit を 64 反復まで続けても維持されるか。また、その推定は数値的に安定しているか。
- **RQ2（余力があれば）**：1 方向あたりの信号（FᵀF の固有値 2.5、w² = 1）を K_GEN=3 と K_GEN=5 でそろえても、K_GEN=5 のほうが局所的に選びにくいか。

## 2. Phase 1 の確認結果（FACT）

| 確認事項 | 結果 |
|---|---|
| 8 反復の推定経路を再現できるか | できる。PR #155 で、同じ seed の再実行が EMC の checkpoint 8 状態をビット単位で再現することを確認済み（`signal_strength/gates/replication.json`） |
| 8 反復から乱数状態を引き継いで再開できるか | **できない**。保存状態（Z 標本・F・σ・w0・w・var_z）に推定用 RNG の状態が含まれていない |
| そのため採る方法 | 同じ seed で 1 反復目から 64 反復まで再実行する。最初の 8 反復は同じ乱数 prefix を消費するので、8 反復時点は保存状態と一致するはずで、これを全 fit でゲートとして照合する（A は EMC の 8・16・24・32 反復の状態とも照合） |
| 途中 checkpoint で C_Q・C_Lap を評価できるか | できる。instrumented runner の `on_checkpoint` で状態を deep copy し、反復後に評価する（推定経路は変えない。ゲート G3b で確認） |
| 既存の 40 反復実験（EMC、`em_convergence_sensitivity_20260905`）との違い | EMC は条件 A（A1 データ）のみ、K=1..7、40 反復、checkpoint 8/12/16/24/32/40。A の n=75 では 4→5 の C_Q slack の中央値が −119〜−137（0/8 で正）、C_Lap slack が約 −11.5（3/8 で正）で、8〜40 反復でほぼ一定だった。**今回新しいのは**、(1) B/C/D の継続、(2) 64 反復までの延長、(3) 推定とは別の乱数による独立 MC 評価、(4) 各 checkpoint での X/Y/Z 分解と Laplace 項、(5) RQ2 |

## 3. RQ1 の設計

| 項目 | 値 |
|---|---|
| データ | PR #155 の A/B/C/D（n=75、K_GEN=5、d=15、rep 1..8）を再生成し、PR #155 の provenance の hash と照合 |
| fit | K_fit ∈ {4, 5} × start {1, 2} × 4 条件 × 8 rep = 128 fit、L=5、64 反復、PR #155 と同じ model seed |
| checkpoint | 全評価：8, 16, 24, 32, 48, 64。C_Q とステップ診断：全反復 |
| 評価 | C_Q（Q_X / Q_Y / Q_Z）、C_Lap（joint mode、Hessian、ln\|H\|、体積項、2H_Lap、残差、termA、R3）、独立 MC 評価の C_Q（L_EVAL=20、seed `[7600000, model_seed, checkpoint]`） |
| 局所比較 | S(4→5) = C(4) − C(5)（2 start 平均）。**K=1..7 からの K̂ ではない** |
| 仮説 | B−A、C−A、D−A の C_Q slack 対応差（checkpoint 64）が正 |
| 判定 | MAINTAINED：中央値 > 0 かつ 6/8 以上が正。NOT_MAINTAINED：中央値 ≤ 0。それ以外 INCONCLUSIVE。C_Lap にも同じ規則（副次）。全 checkpoint で記述的にも報告 |
| 安定性（fit） | checkpoint c までの直近 8 反復で gram_rel_change・eta_x_rel_change・eta_y_rel_change の最大がすべて ≤ 0.01 なら STABLE。64 反復時点で満たさなければ NOT_STABLE（閾値は緩めない） |
| 安定性（データセット） | SIGN_STABLE：S(4→5) の符号が 32・48・64 で同じ。SLACK_STABLE：\|S(64) − S(48)\| ≤ 2 × 独立評価による S の MC 標準誤差 |
| 注意 | L=5 の確率的 MCEM は、決定論的 EM の意味では収束しない。STABLE は数値的安定性のラベルで、収束や一致性ではない |

## 4. RQ2 の設計

| 項目 | 値 |
|---|---|
| データ | rep ごとに Z5 ~ N(0, I5)、F5 = Q·√2.5（FᵀF の固有値はすべて 2.5）、Z3 = Z5 の最初の 3 列、F3 = F5 の最初の 3 列（固有値 2.5）。X5 ~ Pois、X3 は X5 からの厳密なカップリング。w = 1（両 K）、w0 は K ごとに母集団の平均辺密度を canonical A（0.33465）に合わせて較正。Y3 は Y5 からの厳密なカップリング。マスクは共有 |
| 共有できないもの | K=3 と K=5 は生成分布が違うので、観測データは同一ではない（共有するのは潜在方向・乱数・マスク）。**総信号量は K=5 が K=3 の 5/3 倍**（方向あたりをそろえた帰結） |
| fit | K_GEN ∈ {3, 5} × K_fit ∈ {K_GEN−1, K_GEN, K_GEN+1} × start {1, 2} × 8 rep = 96 fit、32 反復、checkpoint 8/16/24/32 |
| 主指標 | rep ごとの対応差 Δ = S_Q(4→5 \| K_GEN=5) − S_Q(2→3 \| K_GEN=3)（checkpoint 32） |
| 判定 | K_HARDER：中央値 Δ < 0 かつ 6/8 以上が負。NOT_HARDER：中央値 Δ ≥ 0。それ以外 INCONCLUSIVE |
| 注意 | 罰則 (d−K) ln n は 2→3 で 13 ln 75、4→5 で 11 ln 75（差 8.6）。Z 事前コストは同じ n(1+ln 2π) |

## 5. 予算と打ち切り規則（結果を見る前に固定）

- fit-iteration：RQ1 8,192 ＋ RQ2 3,072 ＋ゲート 80 = 11,344 ≤ 上限 12,000。
- **全候補 K の補助評価（B/C/D を K=1..7 で 64 反復）は実施しない**。それだけで 15,360 fit-iteration が必要で、上限を超えるため。A の全候補 K は EMC と checkpoint 再評価で 40 反復まで既にある。
- 壁時計の上限は 7 時間。ゲートの実測で、並列 3 プロセスの見込みが 6 時間を超える場合は RQ2 を開始しない。上限に達したステージは PARTIAL と記録する。

## 6. 失敗時の扱い

- fit の例外・NaN の再開要求は FAILED として記録し、リトライも seed の取り直しもしない。
- Laplace が OK でないときは、その dataset・checkpoint の C_Lap を INCOMPLETE とし、補完しない。
- 8 反復時点が再現できない fit は REPLICATION_FAIL とし、「同じ fit の継続」の主張から除外する（CSV には残す）。
