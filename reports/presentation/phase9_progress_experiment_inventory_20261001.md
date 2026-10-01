# 進捗発表資料のための実験 inventory と資料用の表（2026-10-01）

- 位置づけ: source audit（`phase9_progress_source_audit_20261001.md`）の付属文書。**read-only**。
- 全数値は primary artifact（`expfam/results/...` の CSV / JSON）から再集計して report と一致を確認したもの。
  各表に primary path・seed set・dataset subset・条件を併記する。
- すべて **lineage E（experimental prototype; 本文採用不可、MANUSCRIPT_ADOPTION_STATUS: NOT_DECIDED_BY_HUMAN）**。
- 基準の呼称: `C_Q` = Q-based complete-data / ICL-type（Schwarz BIC ではない）、`C_Lap` = Candidate B（Laplace 近似 + d_K ln N、厳密な周辺尤度ではない）。
- Phase 番号は provenance 用。資料の主構成には使わない。

## 0. 共通条件（特記なし）

| 項目 | 値 | source |
|---|---|---|
| n / d | 75 / 12 | 各 `protocol.json` |
| 真の X family | Gaussian×3（列 0–2）/ Bernoulli×6（列 3–8）/ Poisson×3（列 9–11） | 同上 |
| Y | Bernoulli（`η = w0 + w z_iᵀ z_j`） | 同上 |
| 生成器 | canonical model から literal に生成（`canonical-clean-mixed-v1`）。`F = Q · f_scale`（Q は正規直交列）、平均行エネルギー `mean_l ‖f_l‖² = f_scale² K / d` | `data_generator_canonical.f_scale_for_row_norm` |
| σ²_x | 1 | 同上 |
| MCEM | L = 5、exploration / refit 8 / 8 反復、`numerics_mode = consistent`、`failure_policy = fail_fast` | 同上 |
| 候補最適化 | analytic-gradient BFGS（maxiter 2000、gtol 1e-10）、`grad_inf ≤ 1e-8` は WARNING diagnostic | 同上 |
| 候補 K | {1, 2, 3, 4, 5} | 同上 |
| retry / replacement / seed rescue | すべて 0 | 各 `runinfo.json` / summary |

---

## 1. 実験 inventory

### X1. 分布選択 pilot（provenance: Phase 9C / Issue #74）

| 項目 | 内容 |
|---|---|
| research question | K を真値に固定したとき、support gate + 候補 score + hybrid の機構が、score で決める 0/1 列の family を回収できるか |
| why needed | 列ごとの family を人が指定している（`family_x` 全列共通 / `family_x_list` per-column prototype）。自動化の機構が意図どおり動くかを最初に確かめる必要があった |
| data condition | 共通条件、K_true = K_fit = 3、w0 = −1、w = 1、f_scale = √2 |
| what changed | 開始 family（start_B: 曖昧列を Bernoulli から / start_P: Poisson から） |
| held fixed | データ（replicate ごと 1 回生成）、K |
| replicates | 3（data seed 951001–951003、search 952001–3、refit 953001–3）× 2 start |
| K grid | 固定 K = 3（K 選択なし） |
| family setting | 自動（support gate + score） |
| criterion | 列ごとの strict 完全対数確率（E5）。C_Q は refit の副次記録のみ |
| primary metric | score 決定列（列 3–8、真 Bernoulli）での選択、B→P 誤選択数、start 間一致 |
| result | 36/36 Bernoulli、誤選択 0、start 一致 3/3 replicate（Table A） |
| interpretation | 凍結した易しい側の条件で、機構が技術的に最後まで動き、真 Bernoulli の 0/1 列を正しく選んだ（**feasibility**） |
| limitation | 0/1 に見える真 Poisson 列は 0 件で未観測、1 条件 3 replicate、gate 決定列は selector の成績ではない、候補最適化の収束 WARNING 64/576 の影響は未測定 |
| primary artifact | `expfam/results/family_selection/pilot_c2_20260925/` |
| report | `reports/distribution_selection/phase9c_c2_research_first_pilot_summary_20260925.md` |
| 注 | `EXPERIMENT_REGISTRY.md` に行がない（source audit C-11）。`convergence_gate = BLOCKED_FOR_PILOT` は historical label（C-07） |

### X2. family + K 同時選択（provenance: Phase 9D / Issue #75）

| 項目 | 内容 |
|---|---|
| research question | 観測 X, Y と K 候補だけから、family と K を 1 本の監査可能な pipeline で選べるか |
| why needed | family を自動にしても K は人が決めていた。family と K は互いに依存する（loading も Gaussian 列数も K ごとに変わりうる）ので、K ごとに family 選択をやり直す必要がある |
| data condition | X1 と同じ条件、別の dataset |
| what changed | K ∈ {1..5}、start（B / P） |
| held fixed | データ（replicate ごとに 1 回、全 K・両 start で共有） |
| replicates | 3（data seed 961001–961003）× 2 start × 5 K = 30 fit（EM 60） |
| K grid | 1..5 |
| family setting | 各 K で独立に自動選択（持ち越しなし） |
| criterion | C_Q（`calc_bic_exp`） |
| primary metric | K̂（path 単位・replicate 単位）、K̂ での family、joint exact |
| result | K̂ = 2, 3, 3（replicate 単位、exact 2/3）。path 単位 exact 4/6、joint exact 4/6。K̂ での score 決定列 36/36 Bernoulli（全 30 (rep, start, K) でも 180/180） |
| interpretation | pipeline は規則どおり動き、family は安定に正しく、K は 1 replicate で K=2 を選んだ |
| limitation | 6 path は 3 データの 2 start で独立ではない（start 一致は assignment 一致の帰結）。離散探索の罰則なし |
| primary artifact | `expfam/results/joint_family_k_selection/gate75b_20260927/` |
| report | `reports/distribution_selection/phase9d_gate75b_joint_family_k_summary_20260927.md` |

### X2b（補助）. K 選択の再現性（provenance: Phase 9E / Issue #81、C_Q のみ）

| 項目 | 内容 |
|---|---|
| research question | X2 の K = 2 は単発か、同条件で繰り返し起こるか |
| condition | X2 と同じ、新しい 20 dataset（seed 971000+r）、start_B のみ |
| result | C_Q の K̂: K2 3 / K3 17（/20）。K2→K3 の罰則増分は全 dataset で 256.0157（定数）、fit gain は 162.81〜439.54（median 300.68）、256.02 未満が 3/20 |
| 役割 | K 基準を検討する根拠の一つ（資料の slide 10）。**9K の 20 dataset とは別の seed の組**（C-03） |
| primary | `expfam/results/k_repeatability/phase9e_20260927/k_repeatability.csv` |
| report | `reports/distribution_selection/phase9e_k_repeatability_summary_20260927.md` |

### X3. ベースライン比較（provenance: Phase 9K / Issue #94）

| 項目 | 内容 |
|---|---|
| research question | 同じ fit の上で、C_Lap と C_Q はどの K を選ぶか（固定条件の特徴づけ） |
| why needed | Candidate B の実装確認（X-impl）と 3 dataset pilot の後、選択の違いを 20 dataset で記述する |
| data condition | 共通条件、K_true = 3、w0 = −1、w = 1、f_scale = √2 |
| what changed | 基準のみ（同じ refit の上で両基準を計算） |
| replicates | 20（data seed 1001001–1001020、search 1002000+r、refit 1003000+r）、start_B のみ |
| technical | EM 200/200、Candidate B 100/100 OK |
| primary metric | K̂ の分布（K1..K5）、exact / under / over |
| result | C_Lap 0/1/19/0/0、C_Q 1/5/14/0/0（Table C） |
| interpretation | この条件では C_Q の誤りは過小選択で、C_Lap は 19/20 で K = 3 |
| limitation | 1 条件・20 dataset。同条件・別 seed（X2b）では C_Q は 17/20 |
| primary artifact | `expfam/results/lap_vs_cq_20/phase9k_20260928/`（`laplace_by_k.csv`, `paired_summary.json`） |
| report | `reports/k_selection_theory/lap_vs_cq_paired_phase9k_20260928.md` |

### X4. relational（Y 側）信号の感度（provenance: Phase 9P / 9S / 9S2 / 9T）

| 項目 | 内容 |
|---|---|
| research question | Y 側の信号 w を弱める / 強めると、両基準の選択はどう動くか |
| why needed | ベースラインだけでは、違いが信号の強さにどう依存するか分からない |
| data condition | X3 と同じ Z / F / X（20/20 で同一）、Y だけを再生成 |
| what changed | w = 1/√2, 1, √2（w²K_true = 1.5 / 3 / 6）。uncontrolled 版（9P）は w0 = −1 固定、density-controlled 版（9T）は母集団の平均 edge 確率を baseline（0.3314）に揃える w0 を較正（weak −0.8780994405393426、strong −1.1890648128923011、1e-12 基準） |
| held fixed | Z / F / X、seed の組（1001000+r）、baseline は X3 を再利用 |
| replicates | 各条件 20 |
| primary metric | K̂ の分布 |
| result | Table D |
| interpretation | w を弱めると K ≤ 2 が増え（C_Q で顕著）、強めると両基準とも K = 3 に集まった。平均 edge 確率を揃えてもパターンはほぼ残った |
| limitation | 揃えたのは母集団の平均 edge 確率だけで、確率の分布・飽和は揃っていない（**純粋な関係信号の効果ではない**）。uncontrolled 版では edge 密度 median も 0.306 / 0.333 / 0.352 と変わった。controlled weak rep04 は C_Lap の差 0.20 と非常に小さい |
| primary artifact | `expfam/results/relational_w_sensitivity/phase9p_20260928/`、`expfam/results/density_controlled_w_recalibration/phase9s2_20260928/`、`expfam/results/density_controlled_w_sensitivity/phase9t_20260928/` |
| report | `relational_w_sensitivity_phase9p_20260928.md`、`density_controlled_w_recalibration_phase9s2_20260928.md`、`density_controlled_w_sensitivity_phase9t_20260928.md` |

### X5. attribute（X 側）loading の感度（provenance: Phase 9U / 9V）

| 項目 | 内容 |
|---|---|
| research question | X 側の loading の強さを変えると、両基準の選択はどう動くか |
| data condition | 成分分離の対応生成器: Z と実現した Y を条件間で共有（Z・Q・Y の一致 20/20）、F = Q · f_scale |
| what changed | f_scale = 1 / √2 / 2（平均行エネルギー 0.25 / 0.50 / 1.00） |
| held fixed | Z、Q、Y、seed ラベル（1001000+r）。**ただし 9K とは別のデータ系列**（C-04）なので base も新しく実行 |
| replicates | 各条件 20（strong は 19 完了） |
| technical | EM 600 計画、593 試行、592 成功。strong rep04 は K = 2 の exploration で Poisson の exp(η) が overflow（max η = 1120.2）し不完全として記録、retry なし |
| result | Table E |
| interpretation | loading を強めると C_Q の K = 3 が 11 → 15 → 18（/19）と増え、C_Lap はどの条件でもほぼ K = 3 |
| limitation | f_scale は Gaussian の SNR、Bernoulli の飽和、Poisson の裾を同時に変える（**純粋な属性情報の効果ではない**）。strong rep06 で C_Lap が K = 4（過大選択 1 件）。DECISION は `ATTRIBUTE_SCALE_SENSITIVITY_PARTIAL` |
| primary artifact | `expfam/results/attribute_scale_design/phase9u_20260928/`、`expfam/results/attribute_scale_sensitivity/phase9v_20260928/` |
| report | `attribute_scale_design_phase9u_20260928.md`、`attribute_scale_sensitivity_phase9v_20260928.md` |

### X6. 信号を揃えた K_true の感度（provenance: Phase 9W / 9X）

| 項目 | 内容 |
|---|---|
| research question | 真の次元 K_true を変えたとき、両基準はどう選ぶか |
| why needed | X3〜X5 はすべて K_true = 3。K_true を変えると信号の強さも同時に変わるので、平均的な信号を揃えた設計が必要 |
| matched quantities | 平均 X loading エネルギー 0.5（f_scale²K/d）、Y の自然パラメータの分散 w²K = 3、母集団の平均 edge 確率 0.3314（X3 の K3 baseline）。w0 は 1e-12 基準で較正 |
| what changed | K_true = 1, 2, 3, 4（f_scale = √6, √3, √2, √(3/2)、w = √3, √(3/2), 1, √(3/4)、w0 = −0.9305782473108135, −0.9779597638183112, −1, −1.0127634215799726） |
| replicates | 各 K_true 10（rep01..rep10）。K_true = 3 は **X3 の rep01..rep10 を読み取りのみで再利用**（新規 EM 0） |
| technical | 新規 EM 300/300 成功、Candidate B 150/150 OK |
| result | Table F |
| interpretation | 過大選択は両基準とも 0。過小選択は K_true が大きいほど増え、C_Q で顕著 |
| limitation | 各セル 10 dataset・1 つの固定設計。一般の recovery 確率ではない。K_true = 4 の C_Lap は best と次点の差 median 13.4、最小 1.13 と境界に近い。純粋な潜在次元の効果ではない（揃えたのは 3 つの平均量だけ） |
| primary artifact | `expfam/results/matched_k_true_design/phase9w_20260928/`、`expfam/results/matched_k_true_sensitivity/phase9x_20260928/` |
| report | `matched_k_true_design_phase9w_20260928.md`、`matched_k_true_sensitivity_phase9x_20260928.md` |

### X7. K3 → K4 境界の分解（provenance: Phase 9Y、read-only）

| 項目 | 内容 |
|---|---|
| research question | K_true = 4 の既存の fit で、K = 3 と 4 の基準の差は、どの成分の大きさの組み合わせで正負になったか |
| what changed | なし（新しい fit なし。commit 済みの値の分解のみ） |
| data | X6 の K_true = 4（rep01..rep10）。比較文脈として X3 の rep01..rep10（K_true = 3） |
| technical | 10/10 で再構成（残差 ≤ 1.6e-12、許容 1e-10）。EM 0 |
| result | Table G |
| interpretation | C_Q は fit gain が一定の閾値 251.70 を超えたのが 1/10。C_Lap は体積の増分がデータ依存で、余裕が正 8/10 |
| limitation | **算術の分解であり原因の識別ではない**。fit gain の大きさの理由（MC 平均 vs joint mode、次元あたりの信号、MCEM の誤差）は未解決。C_Q の K̂ のうち K1・K2 を選んだ 6 dataset は K3 vs K4 の比較では説明されない |
| primary artifact | `expfam/results/k34_boundary_decomposition/phase9y_20260929/` |
| report | `reports/k_selection_theory/k34_boundary_decomposition_phase9y_20260929.md` |

### X-impl（補助）. Candidate B の実装確認と限界の診断

| provenance | 内容 | 一次の結果 | primary |
|---|---|---|---|
| 9H（#88） | 評価関数の zero-EM 検証（小データ） | 勾配・同時 Hessian が有限差分と ~1e-9（report の測定値）、Gaussian-X・w=0 の厳密周辺尤度と 1.4e-14、尺度不変 | `test_laplace_k_criterion.py`（17 tests）、report |
| 9I（#90） | 3 dataset pilot（seed 981001–3） | 15/15 OK。C_Lap K3 3/3、C_Q K3 1/3 | `expfam/results/laplace_pilot/phase9i_20260928/` |
| 9J（#92） | IS による Laplace 近似の診断（K=2,3） | 相対 ESS 0.4〜12%、厳密な周辺尤度は得られていない | `expfam/results/laplace_is_diagnostic/phase9j_20260928/` |
| 9M〜9R | θ̂ の停留性と局所最適化 | θ̂ は ℓ_Lap の停留点ではない。1 ステップでは順序 20/20 保持、多ステップは INCONCLUSIVE（joint-mode が許容値のわずか上で停止） | 各 artifact（synthesis §5） |

これらは資料の主 slide にはせず、limitation（slide 18）と補足（appendix）で使う。

---

## 2. 資料用の表（再構成）

### Table A — 分布選択 pilot（K 固定 = 3）

- 条件: 共通条件、K_true = K_fit = 3、w0 = −1、w = 1、f_scale = √2、data seed 951001–951003、2 start。
- source: `expfam/results/family_selection/pilot_c2_20260925/`（`support_gate.csv`, `family_scores.csv`, `selection_trace.csv`, `summary.json`）

**A-1. 列の決まり方（support gate）**

| 列（真の family） | 観測された値 | 決まり方 | 6 run × 列数 |
|---|---|---|---|
| 0–2（Gaussian） | 非整数・負値を含む | gate → Gaussian 固定 | 18 |
| 9–11（Poisson） | 2 以上を含む非負整数 | gate → Poisson 固定 | 18 |
| 3–8（Bernoulli） | {0, 1} のみ | **score で Bernoulli vs Poisson** | **36** |

**A-2. score で決まった列の結果**

| replicate | start_B の選択 | start_P の選択 | 真 | start 間一致 |
|---|---|---|---|---|
| rep1 | Bernoulli ×6 | Bernoulli ×6 | Bernoulli ×6 | 一致 |
| rep2 | Bernoulli ×6 | Bernoulli ×6 | Bernoulli ×6 | 一致 |
| rep3 | Bernoulli ×6 | Bernoulli ×6 | Bernoulli ×6 | 一致 |
| 計 | | | | **36/36 Bernoulli、B→P 誤選択 0** |

- 最終 margin（−2 score 単位、Bernoulli 優位）: min 45.69 / mean 54.72 / max 65.50。
- 選択の軌跡: 288 selection 行すべてで Bernoulli。変化は start_P の iteration 1 の 18 件（Poisson → Bernoulli）だけ。
- 候補最適化の収束 WARNING: 64/576 候補評価（全て SciPy precision loss、maxiter 到達 0）。
- **takeaway**: 易しい側の 1 条件で、機構は真 Bernoulli の 0/1 列を開始点によらず正しく選んだ。gate で決まった 36 列は selector の成績に含めない。

### Table B — family + K 同時選択（C_Q）

- 条件: X1 と同じ、data seed 961001–961003（Table A とは別 dataset）、2 start × K 1..5。
- source: `expfam/results/joint_family_k_selection/gate75b_20260927/`（`joint_selection.csv`, `cq_decomposition.csv`, `family_by_k.csv`）

**B-1. C_Q(K)（start_B。start_P はビット単位で同一）**

| replicate | C_Q(1) | C_Q(2) | C_Q(3) | C_Q(4) | C_Q(5) | K̂ | 次点との差 | K̂ での列 3–8 |
|---|---|---|---|---|---|---|---|---|
| rep1 | 5382.625 | **5333.020** | 5341.604 | 5517.493 | 5737.582 | **2** | 8.584 | Bernoulli ×6 |
| rep2 | 5303.367 | 5193.320 | **5152.722** | 5349.842 | 5531.293 | **3** | 40.598 | Bernoulli ×6 |
| rep3 | 5583.194 | 5517.173 | **5441.703** | 5629.297 | 5802.094 | **3** | 75.470 | Bernoulli ×6 |

**B-2. 集計（単位を混同しない）**

| 単位 | 指標 | 値 |
|---|---|---|
| dataset（replicate） | K̂ = K_true | **2/3**（K̂ = 2, 3, 3） |
| pipeline path（rep × start） | K̂ = K_true | 4/6（start 間はデータ共有で独立ではない） |
| pipeline path | joint exact（family と K の両方） | 4/6 |
| score 決定列（K̂ で） | Bernoulli 選択 | 36/36（6 path × 6 列）、B→P 0 |
| score 決定列（全 K） | Bernoulli 選択 | 180/180（30 (rep, start, K) × 6 列） |

**B-3. rep1 の K2 → K3（C_Q の分解、`cq_decomposition.csv`）**

| 量 | 値 |
|---|---|
| fit gain D_2 − D_3 | 247.43 |
| P_Z の増分 | 212.84 |
| P_θ の増分（10 ln 75） | 43.17 |
| 罰則の増分の合計 | 256.02 |
| 結果 | 247.43 < 256.02 → K = 2（差 8.58） |

- **takeaway**: family は全 K で正しく選ばれたが、1/3 dataset で K = 2。差は小さく、罰則の増分の大部分はデータ非依存の P_Z だった（**原因とは書かない**）。

### Table C — ベースライン（20 dataset、K_true = 3）

- 条件: 共通条件、K_true = 3、w0 = −1、w = 1、f_scale = √2、data seed 1001001–1001020、start_B。EM 200/200、C_Lap 100/100 OK。
- source: `expfam/results/lap_vs_cq_20/phase9k_20260928/`（`laplace_by_k.csv`, `paired_summary.json`）

| 基準 | K̂=1 | K̂=2 | K̂=3 | K̂=4 | K̂=5 | exact / under / over |
|---|---|---|---|---|---|---|
| C_Lap | 0 | 1 | **19** | 0 | 0 | 19 / 1 / 0 |
| C_Q | 1 | 5 | **14** | 0 | 0 | 14 / 6 / 0 |

対応（同じ dataset の上で）: 両方 K3 14、C_Lap だけ K3 5、C_Q だけ K3 0、両方 K≠3 1（rep20、両方 K = 2）。

- **takeaway**: 同じ fit でも基準によって選ぶ K が違い、C_Q の誤りは過小選択だった（この 1 条件の 20 dataset）。
- **注**: 同条件・別 seed（971000+r）では C_Q は 17/20（X2b）。この 20 版と Table F の K_true=3 行（rep01..rep10 の部分集合）を混ぜない。

### Table D — Y 側（relational w）の感度

- 条件: Table C と同じ Z / F / X（20/20 同一）、Y だけ再生成、seed 1001000+r、各 20 dataset。baseline 行は Table C の再掲。
- source: `phase9p_20260928/{weak_w,strong_w}/laplace_by_k.csv`、`phase9t_20260928/{weak_w,strong_w}/laplace_by_k.csv`、`phase9k_20260928/`

| 設計 | 条件 | w | w0 | w²K | C_Lap K1..K5 | C_Q K1..K5 | C_Lap K3 | C_Q K3 |
|---|---|---|---|---|---|---|---|---|
| 平均 edge 確率を揃えない | weak | 1/√2 | −1 | 1.5 | 0/4/16/0/0 | 9/10/1/0/0 | 16/20 | 1/20 |
| （9P） | baseline | 1 | −1 | 3 | 0/1/19/0/0 | 1/5/14/0/0 | 19/20 | 14/20 |
| | strong | √2 | −1 | 6 | 0/0/20/0/0 | 0/1/19/0/0 | 20/20 | 19/20 |
| 母集団の平均 edge 確率を 0.3314 に揃える | controlled weak | 1/√2 | −0.8781 | 1.5 | 0/3/17/0/0 | 8/11/1/0/0 | **17/20** | **1/20** |
| （9T） | baseline | 1 | −1 | 3 | 0/1/19/0/0 | 1/5/14/0/0 | **19/20** | **14/20** |
| | controlled strong | √2 | −1.1891 | 6 | 0/0/20/0/0 | 0/1/19/0/0 | **20/20** | **19/20** |

- 揃えない設計の実現 edge 密度 median: 0.306 / 0.333 / 0.352（`phase9p/preflight/relational_signal_context.csv`）。
- **takeaway**: 検討した条件では、w を弱めると過小選択が増え（C_Q で顕著）、強めると両基準とも K = 3 に集まった。平均 edge 確率を揃えてもほぼ同じだった。
- **言えないこと**: 純粋な関係信号の効果（確率の分布・飽和は揃っていない）。controlled weak rep04 の C_Lap の差は 0.20。

### Table E — X 側（attribute loading）の感度

- 条件: 対応生成器（Z と Y を条件間で共有、Z/Q/Y 一致 20/20）、seed ラベル 1001000+r、**Table C とは別のデータ系列**（C-04）。
- source: `phase9v_20260928/{weak,base,strong}/laplace_by_k.csv`、`combined/combined_summary.json`

| 条件 | f_scale | 平均行エネルギー | 完了 dataset | C_Lap K1..K5 | C_Q K1..K5 | C_Lap K3 | C_Q K3 |
|---|---|---|---|---|---|---|---|
| weak | 1 | 0.25 | 20 | 0/0/20/0/0 | 0/9/11/0/0 | 20/20 | 11/20 |
| base | √2 | 0.50 | 20 | 0/0/20/0/0 | 0/5/15/0/0 | 20/20 | 15/20 |
| strong | 2 | 1.00 | **19**（rep04 不完全） | 0/0/18/**1**/0 | 0/1/18/0/0 | 18/19 | 18/19 |

- strong rep04: K = 2 の exploration で Poisson の exp(η) が overflow（ledger: max η = 1120.2）。retry なしで不完全として記録。
- strong rep06（この実験の rep06）: C_Lap が K = 4（過大選択）、C_Q は K = 3。
- **takeaway**: 検討した条件では loading を強めると C_Q の K = 3 が 11 → 15 → 18 と増え、C_Lap はほぼ K = 3 のまま。
- **言えないこと**: 純粋な属性情報の効果（f_scale は Gaussian の SNR・Bernoulli の飽和・Poisson の裾を同時に変える）。「Poisson は一般に有害」（1 件の数値的失敗）。

### Table F — 信号を揃えた K_true の感度

- 条件: 平均 X loading エネルギー 0.5、Y の自然パラメータ分散 w²K = 3、母集団の平均 edge 確率 0.3314 を K_true 間で揃えた設計。各セル 10 dataset（rep01..rep10、seed 1001000+r）。**K_true = 3 は Table C の rep01..rep10 の部分集合**（新規 EM 0）。
- source: `expfam/results/matched_k_true_sensitivity/phase9x_20260928/combined/{confusion_Lap.csv, confusion_Q.csv, combined_summary.json}`、`K3_anchor/anchor_provenance.json`

**F-1. exact（主表）**

| K_true | f_scale | w | C_Lap exact | C_Q exact |
|---|---|---|---|---|
| 1 | √6 | √3 | 10/10 | 10/10 |
| 2 | √3 | √(3/2) | 10/10 | 9/10 |
| 3（Table C の rep01..10） | √2 | 1 | 10/10 | 7/10 |
| 4 | √(3/2) | √(3/4) | **8/10** | **1/10** |

**F-2. 選ばれた K の分布（別表として保存）**

| K_true | C_Lap K̂=1/2/3/4/5 | C_Q K̂=1/2/3/4/5 |
|---|---|---|
| 1 | 10/0/0/0/0 | 10/0/0/0/0 |
| 2 | 0/10/0/0/0 | 1/9/0/0/0 |
| 3 | 0/0/10/0/0 | 1/2/7/0/0 |
| 4 | 0/0/2/8/0 | 1/5/3/1/0 |

- 過大選択: 両基準とも 0。K_true = 4 の C_Lap の best と次点の差: median 13.40、min 1.13、max 50.26。
- **takeaway**: この固定設計では、K_true が大きいほど過小選択が増え、C_Q で顕著だった。C_Lap も K_true = 4 で境界に近づいた。
- **言えないこと**: 一般の recovery 確率（各セル 10 dataset・1 設計）、純粋な潜在次元の効果。

### Table G — K_true = 4 の K3 → K4 境界の分解

- 条件: Table F の K_true = 4（rep01..rep10）の既存 fit。新しい fit 0。比較文脈は Table C の rep01..rep10（K_true = 3）。
- source: `expfam/results/k34_boundary_decomposition/phase9y_20260929/{ktrue4_summary.json, ktrue4_rows.csv, ktrue3_context_summary.json, reconstruction_checks.json}`
- 符号: margin = fit gain − (罰則側の増分)。**正なら K4 が K3 より良い。**

**G-1. 成分（K_true = 4、10 dataset の median。() は min〜max）**

| 基準 | fit gain（K3→K4） | 罰則側の増分 | うちパラメータ罰則 | margin median | margin > 0 |
|---|---|---|---|---|---|
| C_Lap | 254.75（208.16〜377.13） | 体積 Δ(‖Ẑ‖² + log|H|) 203.88（178.89〜251.70）+ 38.86 | 9 ln 75 = 38.86 | **+13.40** | **8/10** |
| C_Q | 164.57（151.69〜318.01） | P_Z 増分 212.84（一定）+ 38.86 = **251.70（一定の閾値）** | 9 ln 75 = 38.86 | **−87.13** | **1/10** |

- 体積の内訳（median、和は median の和ではない）: Δ log|H| 127.60、Δ‖Ẑ‖² 72.76。
- 再構成残差: C_Lap 1.56e-12、C_Q 1.45e-12（許容 1e-10）。

**G-2. dataset ごと（K_true = 4、`ktrue4_rows.csv`）**

| rep | C_Lap K̂ | C_Q K̂ | fit_gain_Lap | volume 増分 | margin_Lap | fit_gain_Q | margin_Q |
|---|---|---|---|---|---|---|---|
| rep01 | 4 | 2 | 258.5 | 204.6 | +15.06 | 204.3 | −47.44 |
| rep02 | 4 | 2 | 268.6 | 208.6 | +21.11 | 204.9 | −46.81 |
| rep03 | 3 | 1 | 220.0 | 182.2 | −1.13 | 158.9 | −92.77 |
| rep04 | 4 | 3 | 243.9 | 193.3 | +11.73 | 162.0 | −89.66 |
| rep05 | 4 | 3 | 312.9 | 228.0 | +46.04 | 243.7 | −8.01 |
| rep06 | 3 | 2 | 208.2 | 178.9 | −9.58 | 151.7 | −100.01 |
| rep07 | 4 | 2 | 245.5 | 198.4 | +8.25 | 165.7 | −85.97 |
| rep08 | 4 | 3 | 251.0 | 203.1 | +8.96 | 158.4 | −93.25 |
| rep09 | 4 | 4 | 377.1 | 251.7 | +86.58 | 318.0 | +66.31 |
| rep10 | 4 | 2 | 276.3 | 207.0 | +30.45 | 163.4 | −88.30 |

**G-3. 比較文脈（K_true = 3、Table C の rep01..10）**: margin は C_Lap 10/10 負（median −52.08）、C_Q 10/10 負（median −163.61）。どちらも K3 を K4 より好む。

- **takeaway**: C_Q では K4 が勝つには MC 平均の fit gain が一定の 251.70 を超える必要があり、超えたのは 1/10。C_Lap では罰則側（体積）がデータ依存で、8/10 で fit gain が上回った。同じ fit で fit gain の値自体も 2 基準で違う（MC 平均 vs joint mode）。
- **言えないこと**: 過小選択の原因（P_Z が原因・信号の希釈が原因とは書かない）。C_Q の K̂ が K1・K2 だった 6 dataset は、K3 vs K4 の比較だけでは説明されない。
