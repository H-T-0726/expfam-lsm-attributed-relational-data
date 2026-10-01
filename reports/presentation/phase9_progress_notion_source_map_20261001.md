# Notion 進捗資料 — source map（2026-10-01）

- 対象ページ: `reports/presentation/phase9_progress_notion_draft_20261001.md`
- 目的: Notion 本文の各節・各数値が、どの report・primary artifact・code に由来するかを辿れるようにする。
- 優先順位: primary artifact / code > canonical docs > report。本文の数値はすべて Phase 1 の監査で primary から再集計して一致を確認済み
  （`reports/presentation/phase9_progress_source_audit_20261001.md` §3.4・§5.3）。
- C_Q の実装名: Phase 9 の mixed-family 実験の C_Q は `eval_utils.calc_Q_dual_strict_exp` + `eval_utils.calc_bic_exp`（source audit C-01）。`calc_bic_dual` ではない。
- provenance 用の Phase 番号はこの文書にだけ記す（Notion 本文には出さない）。
- パスの略記: report は特記なしで `reports/k_selection_theory/`、artifact は `expfam/results/`、code は `expfam/src/experimental/`。

## 1. 節ごとの対応

| Notion の節 | 内容 | report | primary artifact | code | provenance |
|---|---|---|---|---|---|
| §1.1 | 生成モデル E1〜E3 | `RESEARCH_MASTER.md` §2・§4、`CLAUDE.md` §1 | — | `reproduction/src/model.py`（`var_z = 1.0`、`scale_Z`） | canonical |
| §1.2 | 指数型分布族への拡張 | `RESEARCH_MASTER.md` §3 | — | `expfam/src/model_dual_expfam*.py` | canonical |
| §2 | 今回の課題 | `reports/distribution_selection/automatic_family_selection_design_20260923.md` §0 | — | — | 9B |
| §3.1〜3.2 | family 変数、score、選択規則 | 同 design report §1〜§2・§4 | — | `family_selection.py`（`column_log_likelihood`, `select_from_records`, `FamilySelectingPerColumnLSM`） | 9B / 9C |
| §3.3 | support gate | 同 design report §3・§8.2 | `family_selection/pilot_c2_20260925/support_gate.csv` | `family_selection.support_gate` | 9B / 9C |
| §3.4 | hybrid（探索 → 固定 → refit） | 同 design report §7 | — | `family_selection.run_family_exploration`, `run_hybrid_family_selection`、`em_runner.run_em_experimental` | 9C |
| §4 | 分布選択の実験（36/36） | `reports/distribution_selection/phase9c_c2_research_first_pilot_summary_20260925.md` | `family_selection/pilot_c2_20260925/`（`summary.json`, `family_scores.csv`, `selection_trace.csv`, `support_gate.csv`, `runinfo.json`） | `run_family_selection_pilot.py` | 9C（#74） |
| §5.1〜5.3 | 分布 + K 同時選択 | `reports/distribution_selection/phase9d_gate75b_joint_family_k_summary_20260927.md` | `joint_family_k_selection/gate75b_20260927/`（`joint_selection.csv`, `cq_by_k.csv`, `family_by_k.csv`, `cq_decomposition.csv`） | `run_joint_family_k_selection.py`（`select_k_hat`, `cq_decomposition`） | 9D（#75） |
| §5.4 | 再現性確認（17/20） | `reports/distribution_selection/phase9e_k_repeatability_summary_20260927.md` | `k_repeatability/phase9e_20260927/k_repeatability.csv` | `run_k_repeatability.py` | 9E（#81） |
| §6 | C_Q の式・分解・P_Z・呼称 | `cq_decomposition_minimal_check_20260923.md`, `cq_theoretical_clarification_20260927.md`、`KNOWN_ISSUES.md` KI-010 | `joint_family_k_selection/gate75b_20260927/cq_decomposition.csv` | `eval_utils.calc_Q_dual_strict_exp`, `eval_utils.calc_bic_exp`, `run_joint_family_k_selection.cq_decomposition`, `reproduction/src/model.py` `scale_Z` | #72 / 9F（#84） |
| §7.1 | Candidate B を選んだ理由 | `k_true_criterion_design_comparison_20260928.md` §1〜§3・§12 | — | — | 9G（#86） |
| §7.2〜7.4 | C_Lap の式・各項・C_Q との違い | 同上 §3、`laplace_theta_penalty_theory_20260928.md` | `lap_vs_cq_20/phase9k_20260928/paired_summary.json`（`laplace_settings`） | `laplace_k_criterion.py`（`laplace_log_observed`, `calc_C_Lap`）、`run_laplace_pilot.d_K` | 9G / 9H / 9L |
| §7.5 | 実装確認 | `laplace_evaluator_zero_em_verification_20260928.md`, `laplace_pilot_phase9i_20260928.md`, `laplace_is_diagnostic_phase9j_20260928.md`, `theta_stationarity_phase9m_20260928.md`, `multistep_local_pilot_phase9n_20260928.md` | `laplace_pilot/phase9i_20260928/`, `laplace_is_diagnostic/phase9j_20260928/is_summary.json`, `theta_stationarity/phase9m_20260928/` | `test_laplace_k_criterion.py` | 9H / 9I / 9J / 9M / 9N |
| §8 | baseline 比較（19/20 vs 14/20） | `lap_vs_cq_paired_phase9k_20260928.md` | `lap_vs_cq_20/phase9k_20260928/`（`laplace_by_k.csv`, `paired_summary.json`） | `run_lap_vs_cq_20.py` | 9K（#94） |
| §9.1 | Y 側の信号 | `relational_w_sensitivity_phase9p_20260928.md`, `density_controlled_w_recalibration_phase9s2_20260928.md`, `density_controlled_w_sensitivity_phase9t_20260928.md` | `relational_w_sensitivity/phase9p_20260928/`, `density_controlled_w_recalibration/phase9s2_20260928/calibration.json`, `density_controlled_w_sensitivity/phase9t_20260928/` | `run_w_sensitivity.py`, `density_w_recalibration.py`, `run_density_controlled_w.py` | 9P / 9S2 / 9T |
| §9.2 | X 側の信号 | `attribute_scale_design_phase9u_20260928.md`, `attribute_scale_sensitivity_phase9v_20260928.md` | `attribute_scale_design/phase9u_20260928/`, `attribute_scale_sensitivity/phase9v_20260928/`（strong の `execution_ledger.csv` に停止の記録） | `paired_attribute_scale.py`, `run_attribute_scale_sensitivity.py` | 9U / 9V |
| §10 | K_true を変える | `matched_k_true_design_phase9w_20260928.md`, `matched_k_true_sensitivity_phase9x_20260928.md` | `matched_k_true_design/phase9w_20260928/`（`design.json`, `calibration_by_k.csv`）、`matched_k_true_sensitivity/phase9x_20260928/`（`combined/`, `K3_anchor/anchor_provenance.json`） | `matched_k_true_design.py`, `run_matched_k_true.py`, `data_generator_canonical.f_scale_for_row_norm` | 9W / 9X |
| §11 | K3→K4 の分解 | `k34_boundary_decomposition_phase9y_20260929.md` | `k34_boundary_decomposition/phase9y_20260929/`（`ktrue4_rows.csv`, `ktrue4_summary.json`, `ktrue3_context_summary.json`, `reconstruction_checks.json`） | `k34_boundary_decomposition.py` | 9Y（#124） |
| §12〜§15 | 分かったこと・限界・現在位置・今後 | `phase9_k_selection_synthesis_20260929.md` §11〜§15、`RESEARCH_MASTER.md` §19、`KNOWN_ISSUES.md` Q・R | — | — | 9Z（#126） |

## 2. 本文の主要数値と出所

| 数値（本文の位置） | 値 | primary artifact | 確認方法 |
|---|---|---|---|
| score 決定列（§4.2） | 36/36、誤選択 0/36、開始点一致 3/3 | `family_selection/pilot_c2_20260925/family_scores.csv`, `support_gate.csv`, `summary.json` | gate 区分と最終選択を再集計 |
| margin（§4.2） | 45.69〜65.50 | 同 `family_scores.csv`（`neg2_score`） | 再計算 |
| 選ばれた K（§5.3） | 2, 3, 3／path 4/6／joint 4/6／36/36 | `joint_family_k_selection/gate75b_20260927/joint_selection.csv`, `family_by_k.csv` | 再集計 |
| rep1 の C_Q（§5.3） | 5333.020 / 5341.604、差 8.584 | 同 `joint_selection.csv` | 直接読み取り |
| K2→K3 の分解（§5.4） | 247.43 / 256.02 / 212.84 / 43.17 | 同 `cq_decomposition.csv` | 再計算（43.17 = 10 ln 75。9D report の「43.18」は丸めの差: source audit C-13） |
| 再現性（§5.4） | K=3 17/20、K=2 3/20、fit gain 162.81〜439.54 | `k_repeatability/phase9e_20260927/k_repeatability.csv` | 再集計 |
| P_Z（§6.3） | 212.84 / 次元 | `gate75b_20260927/cq_decomposition.csv`、`k34_boundary_decomposition/phase9y_20260929/ktrue4_summary.json` | 75(1 + ln 2π) = 212.8408 と一致 |
| 分解の再構成誤差（§6.2） | ≤ 1.8e-12 | `gate75b_20260927/cq_decomposition.csv`（`abs_diff`） | 最大値 |
| 実装確認（§7.5、App. C） | ~1e-9、1.4e-14 | report のみ（`laplace_evaluator_zero_em_verification_20260928.md`）。テストは許容値 1e-6 / 相対 1e-9 を固定 | source audit C-09 |
| IS 診断（§7.5） | 相対 ESS 0.4〜12% | `laplace_is_diagnostic/phase9j_20260928/is_summary.json`、synthesis §5 | report 値 |
| baseline（§8.2） | C_Lap 0/1/19/0/0、C_Q 1/5/14/0/0、対応 14/5/0/1 | `lap_vs_cq_20/phase9k_20260928/laplace_by_k.csv`, `paired_summary.json` | per-K 値から argmin を再計算 |
| Y 側（§9.1） | C_Lap 17/19/20、C_Q 1/14/19（揃えた版）／揃えない版 16/19/20・1/14/19 | `density_controlled_w_sensitivity/phase9t_20260928/{weak_w,strong_w}/laplace_by_k.csv`、`relational_w_sensitivity/phase9p_20260928/{weak_w,strong_w}/laplace_by_k.csv` | 再計算 |
| Y 側の w0（§9.1） | −0.8780994405393426 / −1.1890648128923011、p = 0.3314 | `density_controlled_w_recalibration/phase9s2_20260928/calibration.json`、`phase9t/.../protocol.json` | 直接読み取り |
| 揃えない版の edge 密度（§9.1 toggle） | median 0.306 / 0.333 / 0.352 | `relational_w_sensitivity/phase9p_20260928/preflight/relational_signal_context.csv` | 再計算 |
| X 側（§9.2） | C_Lap 20/20・20/20・18/19、C_Q 11/20・15/20・18/19、strong 19 完了 | `attribute_scale_sensitivity/phase9v_20260928/{weak,base,strong}/laplace_by_k.csv`、`combined/combined_summary.json` | 再計算 |
| X 側の停止と過大選択（§9.2） | strong の 1 dataset 停止、C_Lap K=4 1 件 | 同 `strong/execution_ledger.csv`、`strong/paired_summary.json` | 直接読み取り |
| K_true（§10.3） | C_Lap 10/10/10/8、C_Q 10/9/7/1 | `matched_k_true_sensitivity/phase9x_20260928/combined/confusion_Lap.csv`, `confusion_Q.csv`、`K1`/`K2`/`K4` の `laplace_by_k.csv` | 再計算 |
| 揃えた量（§10.2） | 0.5、$w^2K=3$、0.3314、w0 4 値 | `matched_k_true_design/phase9w_20260928/design.json`, `calibration_by_k.csv` | 直接読み取り |
| K3→K4（§11） | 251.70、164.57、254.75、203.88、1/10、8/10 | `k34_boundary_decomposition/phase9y_20260929/ktrue4_summary.json`, `ktrue4_rows.csv` | 直接読み取り |
| K_true=3 の比較文脈（§11.3） | 両基準とも 10/10 で K3 が勝つ | 同 `ktrue3_context_summary.json` | 直接読み取り |

## 3. 混ぜてはいけない数値の組（本文で分けて扱っている）

| 組 | 理由 | 本文での扱い |
|---|---|---|
| §4 の 36/36 と §5 の 36/36 | 別の dataset（seed 951001–3 と 961001–3） | 合算しない |
| §8 の 20 dataset（19/20・14/20）と §10 の K_true=3（10/10・7/10） | 後者は前者の rep01..rep10 の部分集合 | §10 で「rep01..rep10」と明記 |
| §8 の 14/20 と §5.4 の 17/20 | 同じ条件・別の seed の組 | 確率として読まない例として注記 |
| §8 の baseline と §9.2 の base | 名前・seed ラベルは同じだが別の生成方法・別データ | §9.2 で比べないと明記 |
| §9.2 strong の rep06（C_Lap K=4）と §11 の rep06（C_Lap K=3） | 別の dataset | 本文では §11 の toggle にだけ rep 番号を出す |

## 4. 外部資料

- 「Distribution-Adaptive Structural EM の基本設計」の PDF: **EXTERNAL_PDF_STATUS: NOT_AVAILABLE_TO_CLAUDE**（Phase 1 source audit §6）。
- 本文 §3.1 の記述（Structural EM 型の考え方を参考に、continuous-Z MCEM へ再定式化）は、repository の design report §1 の記述の範囲に限定している。PDF からの引用・数式はない。
