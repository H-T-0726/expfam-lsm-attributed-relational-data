# 02 元論文と再現

[← README](README.md)

## 1. この実験で知りたかったこと

- 一般化した実装は、先行研究（Mikawa et al. 2024）と同じ条件で、先行研究の再現実装と同じ結果を出すか。
- 元論文の E-step の式と、本研究の採用式はどこが違うか。

## 2. 実験前に分かっていたこと

- 元論文のモデルは、$X$ が Gaussian、$Y$ が Bernoulli の場合である。
- 研究室の MATLAB コードと Python 再現（`reproduction/src/model.py`）があった。
- 本研究のモデルで $X$=Gaussian、$Y$=Bernoulli、$\mu_x=0$、$\sigma_z^2=1$ とすると、元論文のモデルと完全データ対数密度が一致する（#160、数値差 0、DERIVED + OBSERVED。[報告](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/bada1d2/reports/k_selection_theory/fair_model_estimator_criterion_comparison_20261106/10_advisor_summary_ja.md)）。

## 3. 実験条件

| 項目 | 値 |
|---|---|
| シナリオ | Control_Gauss_Bern（先行研究条件） |
| $K_{TRUE}$ | 3 |
| 推定次元 $k_{est}$ | 1 から順に |
| 試行数 | 各 $k_{est}$ で 5 試行 |
| 実装 | `DualExpFamLSM`（系列 B、0.5 あり）と `reproduction/src/model.py`（0.5 あり） |

## 4. 比較した方法

一般化実装（$X$=Gaussian、$Y$=Bernoulli に設定）と、先行研究の Python 再現。

## 5. 数式と評価指標

**1/2 係数の 5 系統**（[`AGENTS.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/AGENTS.md) §2、FACT）：

| 系統 | Y 側の 1/2 |
|---|:---:|
| 元論文の印刷式（Eq.19/20/22/23、Appendix A-1/A-3/A-5） | あり |
| 旧 0.5 Python（`model_expfam.py` / `model_dual_expfam.py`） | あり |
| 本研究の独立再導出・採用式（unique undirected pair の条件付き） | **extra 1/2 なし** |
| fixed Python（`model_dual_expfam_fixed.py`） | なし |
| MATLAB `calcAi` | なし |

採用式の勾配と精度行列は、分散関数 $V(\eta)=A''(\eta)/\phi$ を通じて分布族に依存する（[`RESEARCH_MASTER.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/RESEARCH_MASTER.md) §6）。
$i$ の位置 $z_i$ に関する Y 側の項は、$i$ を含むペア $j\neq i$ の和で、$z_i$ について微分するときに対称な 2 つの項が 1 つにまとまるので、extra 1/2 は付かない（[導出ノート](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/docs/math_notes/half_factor_math_explanation.md)、DERIVED）。

**元論文の印刷式には 1/2 がある**（2026-08-18 に研究者本人が PDF を確認、[一次確認](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/docs/math_notes/half_factor_primary_source_confirmation_20260818.md)）。採用式との差は独立な再導出に基づく意図的な差である。

評価指標は Procrustes RMSE(Z)。

## 6. 結果の表

| $k_{est}$ | 2 実装の RMSE(Z) 平均 | 差の絶対値 | 確認 |
|---:|---|---:|---|
| 3 | 0.17934 / 0.17988 | 0.00054 | VERIFIED（A-01、A-02-k3） |
| 4 | — | 0.00026 | VERIFIED（A-02-k4） |
| 6 | — | 0.02362 | VERIFIED（A-02-k6） |

## 7. 結果の図

本章の比較は 5 試行の表だけで、図は作っていない。

## 8. その結果から分かること

- OBSERVED：先行研究条件の $k_{est}=3$・4 で、一般化実装は再現実装とほぼ同じ RMSE(Z) を出した。
- INTERPRETATION：Gaussian-X / Bernoulli-Y に設定した一般化実装は、先行研究の再現実装と同じように動く。

## 9. 分からないこと

- **採用式（1/2 なし）が正しいことは、この比較では示されない。** 比べた 2 つはどちらも 0.5 系統である。
- $k_{est}=6$ では差が 0.024 あり、すべての $K$ で一致するわけではない。
- 元論文の Exp.2（$K$ の選択）の計算手順は、本文から特定できない（[07 章](07_latent_dimension_selection.md)）。
- 研究室の MATLAB コードには Exp.2 を実行する部分が見つからず、残っているバッチは毎回 $K=3$ を推定していた（#158 の報告、REPORT_ONLY）。

## 10. 公平性・科学的妥当性

- 比較の評価：**A（限定付き）**。同じデータ・同じ条件で 2 つの実装を比べている（[FC-01](data/comparison_fairness_matrix.csv)）。
- 限定：5 試行だけ。両方とも 0.5 系統なので、Newton 方向が全体として正しいとは断定できない（KI-001）。

## 11. 関連する他の実験

- 0.5 の有無の補助検証 `half_factor_check`（条件対応表がなく、本監査では未再計算）。
- 旧推定器（OLD）と系列 E を、同じデータ・同じ初期値で比べた #161・#162（[07 章](07_latent_dimension_selection.md)）。

## 12. 一次データとリンク

- [`reproduction/results/comparison/comparison_main_table.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/reproduction/results/comparison/comparison_main_table.csv)（main）
- 原論文 PDF：[`paper/A_study_on_latent_structural_models_for_binary_rel.pdf`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/paper/A_study_on_latent_structural_models_for_binary_rel.pdf)
- 理論監査：[`reports/theory_audit/theory_audit_report_20260718.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/reports/theory_audit/theory_audit_report_20260718.md)
- MATLAB・Python の歴史的監査（#158、未 merge）：[`historical_code_lineage_audit_20261106/`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/bada1d2/reports/k_selection_theory/historical_code_lineage_audit_20261106)
