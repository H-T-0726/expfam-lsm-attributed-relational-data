# 進捗発表資料のための source audit（2026-10-01）

- 位置づけ: **THEORY / AUDIT（read-only）**。進捗発表資料の前段として、一次情報を監査し、数式・実験条件・結果・主張の境界を確定する。
- **EM 0、新しい実験 0、artifact の再生成 0、parameter tuning 0、criterion 変更 0。** データ処理は既存 CSV / JSON の読み取りと再集計のみ。
- branch: `docs/phase9-progress-material-audit`（新規 worktree `D:/tento/kennkyu-phase9-progress`）
- baseline: `origin/main = ee4be23ab9613c5cc454d8cc1af49c1d2d3551bf`（Phase 9 synthesis の merge commit そのもの。main はこれより進んでいない）
- 系列: 本資料が扱う Phase 9 の分布選択・K 選択はすべて **lineage E（experimental prototype; 本文採用不可）**。
  クラス系譜は `model_dual_expfam_fixed`（1/2 なし系列）→ `_masked` → `_percolumn` → `_consistent`（`expfam/src/experimental/model_dual_expfam_consistent.py` の import で確認）。
- **MANUSCRIPT_ADOPTION_STATUS: NOT_DECIDED_BY_HUMAN**（`RESEARCH_MASTER.md` §19、synthesis §14）。
- ラベル: FACT / DERIVED / OBSERVED / INTERPRETATION / HYPOTHESIS / DECISION / UNRESOLVED（`docs/research_operating_policy.md`）。

同時に作成した 3 文書:

- `reports/presentation/phase9_progress_equation_inventory_20261001.md`（数式 inventory）
- `reports/presentation/phase9_progress_experiment_inventory_20261001.md`（実験 inventory と資料用 Table A〜G）
- `reports/presentation/phase9_progress_storyboard_20261001.md`（storyboard・図案・story 自己レビュー）

---

## 1. source hierarchy（本監査で採用した優先順位）

| 優先度 | 種類 | 本監査での扱い |
|---|---|---|
| 1 | 実装コード / primary CSV / JSON / artifact | 数値はすべてここから再集計して確定。report の数値はここと照合した |
| 2 | canonical documents（`CLAUDE.md`, `docs/research_operating_policy.md`, `RESEARCH_MASTER.md`, `KNOWN_ISSUES.md`, `EXPERIMENT_REGISTRY.md`） | 主張の境界（ALLOWED / QUALIFIED / NOT ALLOWED）・呼称・lineage の正本 |
| 3 | merged research reports（`reports/distribution_selection/*`, `reports/k_selection_theory/*`） | 設計意図・定義・解釈の出典。数値は 1 と照合済みのものだけ使う |
| 4 | Issue / PR 本文 | 今回は直接読んでいない（report に引用された Human Gate comment ID のみ記録） |
| 5 | 過去の AI メモ・説明文（`docs/presentation/seminar_notion_*`, `reports/slide_materials.md`, `reports/figures_for_slides/`, `docs_for_notebooklm/*` 等） | **使用しない**（Phase 9 以前の内容・派生資料。KI-007） |

---

## 2. repository inventory（今回の資料に関係する範囲の分類）

| 分類 | 主なファイル / ディレクトリ | 本資料での役割 |
|---|---|---|
| model（canonical 生成モデル・推定） | `expfam/src/model_dual_expfam*.py`、`reproduction/src/model.py`（`scale_Z`, `var_z=1`） | 背景・式 E1〜E3 |
| model（Phase 9 で使った系列） | `expfam/src/experimental/model_dual_expfam_consistent.py`（`DualExpFamLSMPerColumnConsistent`）、`em_runner.py`、`objective_consistent_numerics.py` | lineage E。全 Phase 9 の fit |
| distribution selection | `expfam/src/experimental/family_selection.py`、`run_family_selection_pilot.py`、`audit_family_selection_pilot.py` | 式 E4〜E6、support gate、hybrid |
| K selection（C_Q） | `expfam/src/experimental/eval_utils.py`（`calc_Q_dual_strict_exp`, `calc_bic_exp`）、`run_joint_family_k_selection.py`（`select_k_hat`, `cq_decomposition`） | 式 E7〜E10 |
| Laplace criterion（C_Lap） | `expfam/src/experimental/laplace_k_criterion.py`、`run_laplace_pilot.py`（`d_K`）、`test_laplace_k_criterion.py` | 式 E11 |
| synthetic generator | `data_generator_canonical.py`（`f_scale_for_row_norm`）、`data_generator_canonical_mixed.py`、`paired_attribute_scale.py`、`matched_k_true_design.py`、`density_w_recalibration.py` | 実験条件 |
| boundary decomposition | `k34_boundary_decomposition.py` | 式 E12〜E13 |
| result artifacts（primary） | `expfam/results/{family_selection, joint_family_k_selection, k_repeatability, laplace_pilot, laplace_is_diagnostic, lap_vs_cq_20, relational_w_sensitivity, density_controlled_w_*, attribute_scale_*, matched_k_true_*, k34_boundary_decomposition}/` | 全数値 |
| reports | `reports/distribution_selection/*.md`、`reports/k_selection_theory/*phase9*`, `cq_*`, `laplace_*`, `k_true_criterion_design_comparison_20260928.md`, `phase9_k_selection_synthesis_20260929.md` | 定義・解釈 |
| canonical docs | §1 の 5 文書 | 主張の境界 |
| historical / superseded（本資料では数値を使わない） | Phase 7e / 8b の held-out K 選択（`reports/k_selection_theory/heldout_*`, `k_true_robustness_*`）、clean true-K n-sweep（RESEARCH_MASTER §17）、学会予稿（0.5 系列）、`family_selection/smoke_*`, `optimizer_*`（9C 前の検証段階）、`docs/presentation/seminar_notion_*` | 別基準・別条件・別系列。混ぜない（KI-002, KI-019） |

---

## 3. 監査したファイル

### 3.1 canonical documents（5）

| ファイル | 読んだ範囲 | 確認したこと |
|---|---|---|
| `CLAUDE.md` | 全体 | 確定生成モデル、1/2 係数の 5 系統、lineage、BIC 呼称制限（§5） |
| `docs/research_operating_policy.md` | 全体 | evidence label、claim ladder、synthetic / real の役割 |
| `RESEARCH_MASTER.md` | §1〜§6、§19 | 研究目的・従来手法・提案手法・Phase 9 claim ledger |
| `KNOWN_ISSUES.md` | D（KI-010）、J（KI-019）、Q、R | C_Q の呼称、2 つの K 基準の混同防止、Phase 9 の「まだ主張してはいけないこと」 |
| `EXPERIMENT_REGISTRY.md` | Phase 9 節（L.439〜463） | 実験 → スクリプト → artifact の対応 |

### 3.2 reports（14）

| ファイル | 読み方 |
|---|---|
| `reports/distribution_selection/automatic_family_selection_design_20260923.md` | 全文 |
| `reports/distribution_selection/phase9c_c2_research_first_pilot_summary_20260925.md` | 全文 |
| `reports/distribution_selection/phase9d_gate75b_joint_family_k_summary_20260927.md` | 全文 |
| `reports/distribution_selection/phase9e_k_repeatability_summary_20260927.md` | 全文 |
| `reports/k_selection_theory/cq_theoretical_clarification_20260927.md` | §1〜§4 |
| `reports/k_selection_theory/k_true_criterion_design_comparison_20260928.md` | 見出し・§3 |
| `reports/k_selection_theory/laplace_evaluator_zero_em_verification_20260928.md` | 全文 |
| `reports/k_selection_theory/k34_boundary_decomposition_phase9y_20260929.md` | §1〜§4 |
| `reports/k_selection_theory/phase9_k_selection_synthesis_20260929.md` | 全文 |
| `lap_vs_cq_paired_phase9k`, `relational_w_sensitivity_phase9p`, `density_controlled_w_sensitivity_phase9t`, `attribute_scale_sensitivity_phase9v`, `matched_k_true_sensitivity_phase9x` の各 report | 結果表・DECISION 行を grep し primary と突き合わせ（全一致） |

### 3.3 code（12）

`expfam/src/experimental/family_selection.py`（support gate L.204–254、strict score L.284–335、選択規則 L.707–734、A-type 更新 L.792–856、exploration L.1048–1191、hybrid L.1194–1294）、
`run_joint_family_k_selection.py`（L.1–300: protocol、`select_k_hat`、`cq_decomposition`）、
`eval_utils.py`（L.186–256: `calc_Q_dual_strict_exp`, `calc_bic_exp`）、
`laplace_k_criterion.py`（全体）、`run_laplace_pilot.py`（`d_K`, N）、`test_laplace_k_criterion.py`（tolerance）、
`reproduction/src/model.py`（`scale_Z` L.468–504、`var_z = 1.0` L.99）、
`model_dual_expfam_consistent.py` / `_percolumn.py` / `_masked.py`（class 系譜）、
`data_generator_canonical.py`（`f_scale_for_row_norm` L.172–）、`paired_attribute_scale.py`（loading energy の定義 L.322）。

**source files audited: 31（canonical 5 / reports 14 / code 12）。**

### 3.4 primary artifacts（16 セット、再集計スクリプトは scratchpad の read-only `recount.py`。repository には追加していない）

| 実験 | primary artifact | 再確認した方法 |
|---|---|---|
| 分布選択 pilot | `expfam/results/family_selection/pilot_c2_20260925/`（`summary.json`, `support_gate.csv`, `family_scores.csv`, `selection_trace.csv`, `fit_results.csv`, `runinfo.json`, `generator_provenance.csv`） | gate 区分・最終選択・margin・trace を CSV から再集計 |
| family + K 同時選択 | `expfam/results/joint_family_k_selection/gate75b_20260927/`（`joint_selection.csv`, `cq_by_k.csv`, `family_by_k.csv`, `cq_decomposition.csv`, `summary.json`, `runinfo.json`, `audit_report.json`） | K_hat・family・P_Z / P_θ・残差・warm start を再集計 |
| K 再現性 | `expfam/results/k_repeatability/phase9e_20260927/k_repeatability.csv` | K_hat 分布・fit_gain_23 |
| Laplace pilot | `expfam/results/laplace_pilot/phase9i_20260928/` | per-K CSV から K_hat 再計算 |
| Laplace IS 診断 | `expfam/results/laplace_is_diagnostic/phase9j_20260928/is_summary.json` | 構造のみ（資料では限界の根拠として report 値を引用） |
| ベースライン 20 dataset | `expfam/results/lap_vs_cq_20/phase9k_20260928/`（`laplace_by_k.csv`, `cq_by_k.csv`, `paired_summary.json`, `protocol.json`） | per-K CSV から両基準の K_hat を再計算 |
| relational w | `expfam/results/relational_w_sensitivity/phase9p_20260928/{weak_w,strong_w,preflight}/` | 同上 + edge 密度 median |
| w0 較正 | `expfam/results/density_controlled_w_recalibration/phase9s2_20260928/calibration.json` | p_target と w0 |
| density-controlled w | `expfam/results/density_controlled_w_sensitivity/phase9t_20260928/{weak_w,strong_w,combined,preflight}/` | 同上 + controlled weak rep04 の gap |
| attribute scale | `expfam/results/attribute_scale_sensitivity/phase9v_20260928/{weak,base,strong,combined,preflight}/` | 同上 + rep04 ledger + rep06 + Z/Y 共有 |
| matched K_true 設計 | `expfam/results/matched_k_true_design/phase9w_20260928/{design.json,calibration_by_k.csv}` | matched 量・w0 |
| matched K_true 実行 | `expfam/results/matched_k_true_sensitivity/phase9x_20260928/{K1,K2,K4,combined,K3_anchor}/` | per-K CSV から再計算 + confusion + anchor 由来 |
| K3→K4 分解 | `expfam/results/k34_boundary_decomposition/phase9y_20260929/`（`ktrue4_rows.csv`, `ktrue4_summary.json`, `ktrue3_context_summary.json`, `reconstruction_checks.json`） | 各成分の median・正負の数・残差 |

---

## 4. conflicts / different protocol（勝手に直していない）

| ID | 種別 | 内容 | primary の判断 | 資料での扱い |
|---|---|---|---|---|
| C-01 | **CONFLICT（呼称・関数の帰属）** | synthesis §2 と `RESEARCH_MASTER.md` §19 は Phase 9 の C_Q を「`calc_bic_dual`」と書くが、Phase 9 の実際の計算は `eval_utils.calc_Q_dual_strict_exp` + `eval_utils.calc_bic_exp(family_x='mixed', n_gaussian_x_cols=…)`（`run_joint_family_k_selection.py` L.12–19, L.137–147）。設計 report §6.2 は `calc_bic_dual` は mixed を扱えないため使用不可と明記 | 式の形 `−2 Q_strict + p_K ln n` と ICL-type という性質は同じ。**関数は `calc_bic_exp`** | 資料では「現行の Q 型基準 C_Q（実装: `calc_bic_exp`、標準系列では `calc_bic_dual`）」と書く。canonical docs は書き換えない |
| C-02 | DIFFERENT_PROTOCOL | 分布選択 pilot（data seed 951001–951003）と family + K 同時選択（961001–961003）はどちらも「score 決定列 36/36」だが**別の dataset** | 別実験 | 2 つの 36/36 を合算しない（72/72 と書かない） |
| C-03 | DIFFERENT_PROTOCOL | C_Q の K=3 は Phase 9E（seed 971000+r）で 17/20、Phase 9K（seed 1001000+r）で 14/20 | 同じ条件・別 seed の組 | 矛盾ではない。資料では 9K の 14/20 だけを主表に使い、17/20 は「seed の組で変わる」例としてのみ |
| C-04 | DIFFERENT_PROTOCOL | attribute scale の base（C_Q 15/20、C_Lap 20/20）は名目上 9K と同じ条件・同じ seed ラベル（1001001…）だが、成分分離の対応生成器による**別のデータ系列**（Z/F/X/Y の一致 0/20、synthesis §8） | 別 dataset | **9V base を 9K baseline として表示しない**。Table C と Table E を混ぜない |
| C-05 | DIFFERENT_PROTOCOL | matched K_true の K_true=3 は 9K の rep01..rep10 の部分集合（C_Lap 10/10、C_Q 7/10）。9K の 20 dataset 版は 19/20、14/20 | `K3_anchor/anchor_provenance.json` の note | 20 版と 10 subset を同じ表に並べない。Table F の K_true=3 行に「9K rep01..10」と明記 |
| C-06 | NOTE（同名 replicate） | 「rep06」は 2 つある: attribute scale strong の rep06（C_Lap が K=4 を選んだ過大選択）と、matched K_true=4 の rep06（C_Lap が K=3 を選んだ過小選択） | 別 dataset | 資料で rep 番号を出すときは必ず実験名を併記 |
| C-07 | NOTE（historical label） | 分布選択 pilot の `summary.json` `convergence_gate.status` と `fit_results.csv` 列 `convergence_gate` は `BLOCKED_FOR_PILOT` | `convergence_gate_role: diagnostic (research-first C2 policy)`、`audit_report` は technical_validity VALID | 資料では「候補最適化の収束 WARNING（diagnostic）」と書き、BLOCKED を結果の失敗と読ませない |
| C-08 | NOTE（定義の言い換え） | 9Y の Issue 本文は volume を「logdet の差」と書いたが、実装・既存定義は `Δ(‖Ẑ‖² + log|H|)` | report §3 が内訳も記録 | 資料は `Δ(‖Ẑ‖² + log|H|)` を使う |
| C-09 | NOTE（数値の出所） | Candidate B 検証の「~1e-9」「1.4e-14」は report の測定値。テストコードが固定するのは許容値（勾配・Hessian 1e-6、Gaussian 厳密一致 1e-9 相対） | 実測値は report にのみ記録 | 資料では「有限差分と ~1e-9 で一致（検証 report の測定値、テスト許容値 1e-6）」と出所を併記 |
| C-10 | NOTE（浮動小数点表記） | p_target が 0.33140621213146765（9S2 / 9T）と 0.3314062121314677（9W）で表記が違う | 差 ≲ 1e-16（同一値の丸め表記） | 資料では 0.3314 と丸めて書く |
| C-11 | REGISTRY_GAP | `EXPERIMENT_REGISTRY.md` の Phase 9 表は 9D から始まり、分布選択 pilot（`family_selection/pilot_c2_20260925/`）の行がない | artifact と report は存在 | 本 task では registry を変更しない。Human への報告事項 |
| C-13 | NOTE（丸め） | 9D report §Diagnostic は K2→K3 の P_θ 増分を「43.18」と書くが、`10 ln 75 = 43.17488`（`cq_decomposition.csv` の 155.42957 − 112.25469 も同じ） | 43.17 が正しい丸め。合計 256.0157 は一致 | 資料では 43.17（または 10 ln 75）と書く。report は書き換えない |
| C-12 | NOTE（9P preflight） | `relational_w_sensitivity/.../preflight/preflight.json` は `git_dirty: true` | report L.28 が理由（preflight 出力 dir が未追跡）を記録。weak/strong の runinfo は dirty=False | 影響なし |

**SOURCE CONFLICT: YES**（C-01 の 1 件。数値の矛盾ではなく関数名の帰属。数値の CONFLICT は 0 件）。
それ以外は DIFFERENT_PROTOCOL / NOTE / REGISTRY_GAP であり、数値は全て primary と一致した。

---

## 5. FACT / DERIVED / OBSERVED の確定一覧

### 5.1 モデル・アルゴリズム（FACT: code）

| # | 内容 | source |
|---|---|---|
| F-1 | `z_i ~ N(0, I_K)`、`η_il^X = f_l^T z_i`（X 切片なし）、`η_ij^Y = w0 + w z_i^T z_j`（スカラー w0, w） | `CLAUDE.md` §1、`RESEARCH_MASTER.md` §4 |
| F-2 | MCEM の E-step はノードごとの Newton + Laplace サンプリング、L=5 本を `scale_Z` で全要素の平均二乗 1 に一括スケール、`var_z = 1` 固定 | `reproduction/src/model.py` L.99, L.468–504、`family_selection.py` L.1125–1144 |
| F-3 | support gate: 全値 {0,1} → 候補 {bernoulli, poisson}（score で決定）／非負整数で 1 より大を含む → poisson 固定／それ以外 → gaussian 固定。尤度を使わない | `family_selection.support_gate` L.204–254 |
| F-4 | 候補 score は「完全な」列ごとの対数確率の L サンプル平均。Poisson は `−log(x!)` を保持、Gaussian は `−½ log σ²` と `−½ log 2π` を保持（σ² はプロファイル MLE）、Bernoulli は基底測度 1 | `family_selection.column_log_likelihood` L.284–335 |
| F-5 | 候補ごとに loading `f_l^(m)` を analytic-gradient BFGS（maxiter 2000, gtol 1e-10）で別々に最適化。収束 `grad_inf ≤ 1e-8` は diagnostic | `family_selection.py` L.100–118, L.640–668 |
| F-6 | 選択規則: score 最大の候補、完全同値なら (bernoulli, poisson, gaussian) の順。margin = (−2 score_次点) − (−2 score_最良) ≥ 0。離散探索の罰則なし | `select_from_records` L.707–734、HG-3 L.27–30 |
| F-7 | Scheme C（hybrid）: exploration EM の各 M-step（`calc_F` 内）で曖昧列の family を再選択 → 最終 assignment を固定 → **同じ assignment での新規 refit**（`run_em_experimental`, `failure_policy='fail_fast'`）を報告値にする | `run_hybrid_family_selection` L.1194–1294 |
| F-8 | 同時選択: 各 (replicate, start, K) で support gate から refit までをゼロから実行。K 間の warm start・family 持ち越しなし（`cq_by_k.csv` の `warm_start_source` は 30/30 行で `none`）。`K_hat = argmin_K C_Q(K)`、完全同値は小さい K | `run_joint_family_k_selection.py` L.8–27, L.215–221、artifact |
| F-9 | `C_Q = −2 Q_strict + p_K ln n`、`p_K = K d − K(K−1)/2 + n_gaussian_x_cols + 1{Y=Gaussian}`（w0, w は数えない） | `eval_utils.calc_bic_exp` L.232–256 |
| F-10 | `C_Q = D_K + P_Z + P_θ` の分解は refit の状態から再計算され、直接値との差は最大 1.8e-12（9D） | `cq_decomposition` L.244–288、`gate75b/cq_decomposition.csv` |
| F-11 | Candidate B の評価: `ℓ_Lap = Φ(Ẑ) + (nK/2) ln 2π − ½ ln|H|`、Ẑ は減衰 full Newton による joint mode、`max|∇Φ| ≤ 1e-8` かつ H 正定値でなければ値を返さない（ridge / jitter なし）、`C_Lap = −2 ℓ_Lap + d_K ln N` | `laplace_k_criterion.py` L.199–312 |
| F-12 | `d_K = K d − K(K−1)/2 + n_gaussian_selected`、N = 75（working convention）。θ̂ は refit の最終値、Ẑ の初期値は refit の `Z_est` | `run_laplace_pilot.d_K` L.81–82、`phase9k/paired_summary.json` `laplace_settings` |
| F-13 | Bernoulli Y の現条件で C_Q の `num_params` と C_Lap の `d_K` は一致（10/10） | `phase9y/reconstruction_checks.json` |

### 5.2 DERIVED

| # | 内容 | 根拠 |
|---|---|---|
| D-1 | `q_t` を固定すると `Q_Z` と `Q_Y` は family 変数 c に依存しないので、列ごとの family 比較から厳密に消える | 設計 report §2.2–2.3 |
| D-2 | 異なる基底測度（counting vs Lebesgue）の候補は対数尤度の大小で比べられない（単位変換で Gaussian の log-density だけが `−n log a` 動く） | 設計 report §3.3、`support_gate` docstring |
| D-3 | 現行 3 family + support gate の下で非自明な score 比較が起きるのは **0/1 列の Bernoulli vs Poisson だけ** | 設計 report §3.2, §8.2 |
| D-4 | `scale_Z` と `var_z = 1` のもとで `P_Z = −2 Q_Z = nK(1 + ln 2π)`、n = 75 で 1 次元あたり 75(1 + ln 2π) = 212.8407799807 | `cq_theoretical_clarification` §2, §4、本監査で数値計算 |
| D-5 | C_Q は Schwarz BIC ではない（観測データの周辺尤度を使わない）。「観測データ型 BIC + 2 × 潜在変数の事後エントロピー」と同型なので ICL-type。ただし連続 Z の微分エントロピーは尺度依存 | `cq_theoretical_clarification` §3–4、KI-010 |
| D-6 | `P_Z` の 1 次元あたりの増分 `n(1 + ln 2π)` は観測データから識別できない慣行（事前分散 1）で決まり、Z の尺度を σ に変えると C_Q は `nK ln σ²` ずれる | 同 §4 |
| D-7 | var_z = 1 で `−2 ℓ_Lap = −2[ℓ_X(Ẑ) + ℓ_Y(Ẑ)] + ‖Ẑ‖² + ln|H(Ẑ)|`（`(nK/2) ln 2π` は事前の定数と相殺）。よって実装の `C_Lap` は E11 と厳密に同値 | 検証 report §1、本監査で `phase9k` rep01 K=1 の列値で数値確認（`integration_term` 5261.2709 を再現） |
| D-8 | Laplace 値と K 間の差は潜在尺度の線形変換で不変 | 設計比較 report §3.2、検証 report §7 |
| D-9 | K3 → K4 の loading 増分は 33 → 42 の +9、パラメータ罰則の増分は `9 ln 75 = 38.857393`。C_Q で K4 が K3 に勝つ閾値は 212.8408 + 38.8574 = **251.6982** | 9Y report §4、本監査で数値計算 |
| D-10 | 9D/9E 条件の K2 → K3 では C_Q の罰則増分が `212.84 + 10 ln 75 = 256.0157` で一定 | 9E report §P3、`k_repeatability.csv` |

### 5.3 OBSERVED（primary から再集計して一致を確認したもの）

数値と条件は experiment inventory の Table A〜G に完全な形で載せた。ここでは確認の結果だけを列挙する。

| # | 観測 | primary | 再集計の結果 |
|---|---|---|---|
| O-1 | 分布選択 pilot: score 決定列（真 Bernoulli）36/36 で Bernoulli、B→P 誤選択 0、start 一致 3/3 replicate、最終 margin 45.69〜65.50 | `pilot_c2_20260925/` | 一致 |
| O-2 | family + K 同時選択: K_hat = 2, 3, 3（replicate 単位）、path 単位 exact 4/6、joint exact 4/6、K_hat での score 決定列 36/36 Bernoulli | `gate75b_20260927/` | 一致 |
| O-3 | K 再現性（C_Q のみ）: K2 3、K3 17（/20） | `phase9e_20260927/k_repeatability.csv` | 一致 |
| O-4 | ベースライン 20: C_Lap 0/1/19/0/0、C_Q 1/5/14/0/0 | `phase9k_20260928/laplace_by_k.csv` | 一致（100/100 OK） |
| O-5 | relational w（非制御）: weak C_Lap 0/4/16/0/0, C_Q 9/10/1/0/0；strong C_Lap 0/0/20/0/0, C_Q 0/1/19/0/0 | `phase9p_20260928/{weak_w,strong_w}/` | 一致 |
| O-6 | density-controlled w: weak C_Lap 0/3/17/0/0, C_Q 8/11/1/0/0；strong C_Lap 0/0/20/0/0, C_Q 0/1/19/0/0 | `phase9t_20260928/{weak_w,strong_w}/` | 一致 |
| O-7 | attribute scale: weak C_Lap 0/0/20/0/0, C_Q 0/9/11/0/0；base C_Lap 0/0/20/0/0, C_Q 0/5/15/0/0；strong（19 完了）C_Lap 0/0/18/1/0, C_Q 0/1/18/0/0 | `phase9v_20260928/{weak,base,strong}/` | 一致 |
| O-8 | matched K_true: C_Lap exact 10/10/10/8、C_Q exact 10/9/7/1（K_true = 1/2/3/4、K_true=3 は 9K rep01..10） | `phase9x_20260928/` | 一致 |
| O-9 | K3→K4 分解（K_true=4）: margin 正は C_Lap 8/10、C_Q 1/10。再構成残差 ≤ 1.6e-12 | `phase9y_20260929/` | 一致 |

---

## 6. 外部参考資料

**EXTERNAL_PDF_STATUS: NOT_AVAILABLE_TO_CLAUDE**

「関係データの属性分布を自動選択する指数型分布族モデルの定式化 — Distribution-Adaptive Structural EM の基本設計」は、
repository（`paper/` には原論文 PDF と別の IEEE 論文 `paper/2.pdf`〔metadata title: "Learning and Estimation of Latent Structural Models Based on between-Data Metrics"〕のみ）
にも、確認したローカルパス（`D:/tento`, `Downloads`, `Documents`, `Desktop` の深さ 4 まで）にも見つからなかった。**内容は推測しない。**

repository 内で確認できる事実だけを書く:

- 設計 report §1 は「`z_i ∈ {1,…,K}` の latent class model ではない。したがって latent-class 用の Structural EM をそのまま移植しない。以下はすべて現行 MCEM の posterior sample `Z^(s)` の上で再導出する」と明記している（FACT）。
- したがって資料での正しい書き方は「**Structural EM 型の分布族選択の考え方を参考にし、continuous-Z MCEM の posterior samples を用いる形へ再定式化した**」であり、「資料の手法をそのまま実装した」とは書かない。
- 参考資料がどの式を含むかは本監査では確認していないので、資料では参考資料の式を引用しない。

---

## 7. 主張の境界（資料で使うラベル付きの整理）

### ALLOWED（そのまま書いてよい）

| claim | label | source |
|---|---|---|
| 現行 C_Q は Schwarz BIC ではなく Q-based complete-data / ICL-type criterion | FACT / DERIVED | KI-010、`cq_theoretical_clarification` |
| scale_Z・var_z = 1 のもとで P_Z = nK(1 + ln 2π)、n = 75 で 1 次元あたり 212.84 | DERIVED | 同上、9D/9Y artifact |
| 列ごとの family 選択 prototype（support gate + strict score + hybrid）を定義・実装した | FACT | 設計 report、`family_selection.py` |
| K ごとに family 選択をやり直す family + K 同時選択 pipeline を実装した | FACT | `run_joint_family_k_selection.py` |
| Candidate B（C_Lap）を定義・実装し、評価関数は有限差分・Gaussian 厳密解と一致した | FACT | 9H report、`laplace_k_criterion.py` |
| 固定した人工データ条件で、両基準の選択パターンを観測した | OBSERVED | Table C〜F |
| 信号を揃えた K_true の設計で、K_true が大きいほど過小選択が増えた（特に C_Q） | OBSERVED | Table F |
| K3→K4 の差は既存成分から算術的に分解でき、閾値に対する余裕は C_Lap 8/10、C_Q 1/10 が正 | OBSERVED（算術） | Table G |

### QUALIFIED ONLY（限定語つきでのみ）

| claim | 必須の限定 |
|---|---|
| C_Lap は検討したいくつかの固定条件で C_Q より true K を選んだ dataset が多かった | n=75, d=12, G3/B6/P3, Y Bernoulli, MCEM 8 反復, start_B のみ, 条件ごと 10〜20 dataset。例外（K_true=4 の 2 件の過小、attribute strong の 1 件の過大）を併記 |
| X / Y の信号を強めると、検討した条件では選択が K=3 に寄った | Y 側は平均 edge 確率だけを揃えた（分布・飽和は未制御）、X 側は Z・Y を共有した対応設計で strong の 1 dataset が不完全。純粋な信号効果ではない |

### NOT ALLOWED（書いてはいけない）

C_Lap は一般に優れる／分布選択は一般に解決した／K 選択は解決した／Candidate B は一致性をもつ／一般の recovery 確率／
P_Z が過小選択の原因／信号の希釈が原因／実データでの K の妥当性／Poisson は一般に有害／lineage E は自動的に本文採用／
C_Q を「Schwarz BIC」、C_Lap を「厳密な周辺尤度」と呼ぶ／Gaussian・Bernoulli・Poisson を全部完全自動判定した。

### UNRESOLVED（synthesis Z9-U1〜U8 をそのまま継承）

K3→K4 の fit gain の大きさの理由（U1）、8 反復 MCEM と θ̂ の非停留性の影響（U2）、一致性（U3）、他条件への一般化（U4）、
別の信号の揃え方（U5）、実データでの K（U6）、ℓ_Lap と厳密周辺尤度の差（U7）、NOT_STATIONARY の扱い（U8）。
分布選択側: 真 Poisson が 0/1 に見える列での挙動は未観測、離散探索コストは未罰則、X 切片なしによる位置適合の交絡、cross-measure（representation）選択は対象外。

---

## 8. provenance

| 項目 | 値 |
|---|---|
| 監査時 HEAD | `ee4be23ab9613c5cc454d8cc1af49c1d2d3551bf` |
| 監査で生成・変更した artifact | 0（本 4 文書のみ追加） |
| 一時再集計スクリプト | session scratchpad の `recount.py`（per-K `laplace_by_k.csv` の C_Lap / C_Q 列から argmin を再計算。repository 外・read-only） |
| EM / refit / Candidate B 再評価 | 0 / 0 / 0 |
| 既存 report・canonical docs の変更 | 0 |
