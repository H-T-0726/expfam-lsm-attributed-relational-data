# Evidence matrix：研究上の主張と一次資料の対応

## 区分

| 区分 | 意味 |
|---|---|
| **R-data** | 再現済み（データ）：独立な新しいデータで、同じ向きの結果が再び得られた |
| **R-seed** | 再現済み（同じデータ・別の seed）：同じデータで、初期値と MC の seed だけを変えた再実行で、同じ向きの結果が得られた |
| **R-calc** | 再現済み（計算）：同じ結果を独立な実装・再計算で再現した。新しいデータでの再現ではない |
| **S** | 独立に確認されていない：1 回の実行、または手法を作ったのと同じデータの上の結果 |
| **P** | 事後解析：結果を見た後に追加した解析、または探索的な分解 |
| **T** | 理論：数式から導いたもの。条件つき |
| **N** | 否定または未達成：事前に決めた基準で支持されなかった |

## その他の列と注意

- **採用**：本文に採用できるかどうか。[`RESEARCH_MASTER.md` §14 以降の Claim Ledger](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md) の区分（ALLOWED / QUALIFIED / NOT ALLOWED / 未決定）に従う。
- **lineage を混ぜない（KI-002）**：系列 B・C・E の数値を、1 つの比較にしない。
- **リンク**：すべて commit 固定。PR #168 は `856d475`、PR #129 は `4fa6159`、それ以外は `f71c04d`。

---

## A. 本来の研究（指数型分布族への拡張）

| # | 主張 | 区分 | 系列 | 一次資料 | 採用 | 限定 |
|---|---|---|---|---|---|---|
| A1 | X 側と Y 側を、G/B/P から独立に指定できる実装がある | FACT（code） | B | [`model_dual_expfam.py`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/src/model_dual_expfam.py)、`test_dual_expfam.py` | ALLOWED | 実装レベルの一般化で、性能の主張ではない。Categorical は未実装。**この実装は Y 側に 0.5 が残る（KI-001）**ので、「Newton 方向が全体として正しいとは断定できない」を付記する。採用式（1/2 なし）は lineage C 以降の実装 |
| A2 | 先行研究と同じ条件で、再現実装と一致する（RMSE(Z) の差 < 0.001。**k_est = 3 と 4 だけ**で、ほかの K では最大 0.024） | R-calc | B | [`comparison_main_table.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reproduction/results/comparison/comparison_main_table.csv) | ALLOWED | 5 試行だけ。比べた 2 つ（`model_dual_expfam.py` と `reproduction/src/model.py`）はどちらも 0.5 を含む系統（[§6.1](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md)）。そのため、**採用式の検証ではなく、実装の同等性の確認**である |
| A3 | n=50→300 で RMSE(Z) が下がる（A −49.3%、B −41.2%、C −58.6%） | S（3 シナリオ） | C | [`fixed_exp2_n_sweep_improvement.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/fixed_official/exp2/fixed_exp2_n_sweep_improvement.csv) | ALLOWED | 明示した生成設定の中だけ。fixed 系列は Y 側の extra 1/2 がない（採用式と一致）。ただし η の clip（X 側、Y 側、strict-Q の目的関数）は legacy のまま（PC-001/002 は F で測ったもので、C については同じコードからの推論）。旧 0.5 系列（49%、31%、62%）と同じ表に置かない |
| A4 | 分布の誤指定で RMSE(Z) が悪化する（最悪 A 4.34×、B 9.04×、C 40.37×） | S（系列 B でも同じ向き） | C | [`fixed_official/exp4/`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/fixed_official/exp4) | ALLOWED | 倍率の内訳（OBSERVED、[§5.2](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reports/advisor_feedback/distribution_misspecification_synthesis_20261013.md)）：最大の B 9.04× と C 40.37× は X と Y の両側を誤指定したセル。Y 側だけでは最大 7.9× と 19.6×、X 側だけでは最大 4.34×（A、support-invalid）。「主に Y 側による」は解釈で、X と Y の組み合わせの影響もある。旧 0.5 系列の最悪比（3.41×、7.35×、41.45×）とは誤指定の条件が違うので、差を「0.5 除去の効果」と読まない（KI-003、§15） |
| A5 | support-compatible な X の誤指定（B→P）で、Z の推定誤差が増える。dense +0.020（15/20）、sparse +0.066（17/20） | **R-seed**（E1 Part B：同じデータ・別の seed で sparse を再現。mode で +0.087、20/20） | E | [M1](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/misspecification_m1/phase_m1_20261013)、[E1](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/z_point_estimator_e1/e1_20261104) | 未決定（prototype） | 1 つの固定条件。dense の last 推定は再現しなかった（A +0.020、B +0.002） |
| A6 | 誤指定の悪影響は Y の情報が少ないほど大きい（B→P の場合） | R-seed（M1 と E1） | E | 同上、[PR #168 の結果](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/856d475/reports/misspecification_extended/results_20261110.md) | 未決定 | **誤指定の種類によって交互作用の向きが逆になる**（#168：B6→G6 や all-Gaussian では Y=100% のほうが大きい）。一般則ではない |
| A7 | dense-Y では all-Gaussian でも正しい family とほぼ同じで、Y を 1 割にすると明確に負ける | S（2 系列で同じ向き） | F / E | [`distribution_misspecification_synthesis_20261013.md` §6](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reports/advisor_feedback/distribution_misspecification_synthesis_20261013.md) | QUALIFIED | 系列と生成器によって大きさが違う（+0.12〜+0.43） |
| A8 | sparse-Y・補完的な属性 block の人工設定では、per-column の同時利用で RMSE_Z が改善する（y_only 比 +0.51、single_gaussian 比 +0.20） | S | E+F | `complementary_blocks_consistent_20260821_paired.csv`（[§12.3](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md)） | QUALIFIED | prototype。dense では +0.009 まで縮む。機構は未確立（#31） |
| A9 | MovieLens（user-disjoint）では、平均の向きは正（+0.0042、23/30）で、変動に比べて小さい | S | E+F | [§12.5](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md) | QUALIFIED | 30 の split は独立な replicate ではない。CI は計算していない |
| A10 | 識別可能性の命題：<br>P1：canonical で clip のない Poisson-X で、rank F = K（generic）なら、FFᵀ を population モーメントから復元でき、X 周辺の K^rank が決まる。<br>P2：canonical Gaussian-Y で w ≠ 0 なら、**1 つの dyad の周辺分布**から K、w²、σ_y² が決まる。<br>P3：w ≠ 0 の部分では族は入れ子でない | T | — | [§17.3](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md)、[`true_k_identifiability_hardened_20260904.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reports/identifiability/true_k_identifiability_hardened_20260904.md) | ALLOWED（前提を明記） | **すべて population の主張**。P2 は母集団の周辺分布についての主張で、**1 つの観測値から K を推定できるという意味ではない**。有限標本での推定・一致性は未証明（U6、U7）。**実験で使う Bernoulli-Y は未解決**（U2）。前提の一覧は `mathematical_open_questions.md` §1.1 |
| A11 | C_Q は Schwarz BIC ではなく、ICL 型の基準である | T / FACT（code） | B/C/E | KI-010、§17.4 | ALLOWED | 関数名・CSV の列名は変更しない |
| A12 | C_Lap は、一部の固定条件で C_Q より真の K を多く選んだ（9K：19/20 対 14/20） | S | E | [§19](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md) | QUALIFIED・本文採用は未決定 | 一般の優越ではない。seed の組で変わる |

## B. 分布の自動選択（AF。すべて lineage E の prototype で、本文採用は未決定）

### B0. 4 つの段階の区別

| 段階 | 該当する結果 |
|---|---|
| ① 選択が正解したか | AF1：全列（解析的にほぼ保証）。AF2 Auto：dense 0.900、sparse 0.767。AF6 の Q：0.946、0.813。PR #129：真の Z で COUNT 0.984 |
| ② 合理的な比較手法より改善したか | **選択の正解率**：Laplace（AF5）は同じデータで Q より sparse +0.40 [0.05, 0.70]。E2（AF8-v3）は AF6 Laplace より balanced +0.023 [−0.004, +0.050]（CI は 0 を含む）。<br>**Z の推定**：Auto は Support-only より小さく改善（B5（**主指標の joint mode だけ**。副指標の last と mean5 では、sparse の CI は 0 を含む）） |
| ③ 新しいデータで改善が再現したか | Laplace の Q に対する改善：AF6 の新しい 40 データで点推定は同じ向き（+0.225）だが CI は 0 を含む。<br>Z の推定の改善：新しいデータでは再評価していない。<br>Q の正解率の水準：AF2→AF6 で近い（再現ではなく参考） |
| ④ 事前登録した成功条件を達成したか | AF1：主対比どおり（Benefit > 0）。AF2：主対比（Benefit = Wrong − Auto）が正で CI は 0 を含まない（成功の閾値は protocol で定めていない）。**AF6：主要評価項目は未達成**。AF5・AF8-v3 は探索的で、成功条件は定めていない |


| # | 主張 | 区分 | 一次資料 | 限定 |
|---|---|---|---|---|
| B1 | {0,1} 列では、どの η でも log Bern − log Pois = e^η − log(1+e^η) > 0 | T | [AF1 README §0](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af1_20261109/README.md) | 同じ η での比較。最適化が届いている場合 |
| B2 | Bernoulli vs Poisson の自動選択は 160/160 fit で全列正解し、Wrong による悪化（sparse 0.0816 [0.0642, 0.0991]）を回避した | S（事前登録の CONFIRMATORY。主対比どおり。結果は B1 からほぼ決まる） | [AF1 `full/`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/auto_family_z_recovery/af1_20261109/full) | Auto = Oracle は refit の seed を共有した設計上の性質 |
| B3 | {0,1} 列では CMP2 も Poisson に勝つ。切片なしの n=75 では、{0,1} だけの真 Poisson 列の確率は ≤ 1.01e-10 | T ＋数値計算 | [AF2 Part A](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/auto_family_z_recovery/af2_part_a_20261110) | 負の切片を仮定すると頻繁に起き、選択と交絡する |
| B4 | Poisson vs CMP2 の自動選択（Q スコアによる Auto）の正解は dense 108/120（0.900）、sparse 92/120（0.767）で、誤りはほぼ CMP2→Poisson | S（事前登録の 1 回の実行。正解率の閾値は事前に定めていない） | [AF2 README](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af2_20261110/README.md) | 1 条件・20 データ。AF2 の Q の sparse 正解率は 0.767（92/120）、AF6 の新しいデータでは 0.813 で、近い水準だった（参考） |
| B5 | Auto は Support-only（count 列をすべて Poisson）より RMSE(Z) が小さい：dense 0.0051 [0.0023, 0.0079]（15/20）、sparse 0.0059 [0.0013, 0.0109]（13/20）。Wrong に対しては dense 0.0238、sparse 0.0542。Oracle には届かない（sparse で 0.0116） | S（事前登録の主対比・副対比。CI は 0 を含まないが差は小さい（**主指標の joint mode だけ**。副指標の last と mean5 では、sparse の CI は 0 を含む）。成功の閾値は事前に定めていない） | 同上 | 1 条件・20 データ。新しいデータでの Z の再評価はしていない。PR #129（別の設計）でも、Auto − Support-only の RMSE(Z) は −0.009〜−0.043 だった（3 手法がすべて成功したデータに限る。Support-only に RMSE > 1 の外れ値が 4 fit ある。[PR #129 の報告 §3.5](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/4fa6159/reports/distribution_selection/multisupport_family_selection_report_20261006.md)） |
| B6 | Q スコアは「Z を作った分布」に引き寄せられる（Oracle の Z では 60/60、Support-only の Z では 0.533） | S（統制した比較、同じ 20 データ） | [AF3 README](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af3_selector_rescoring_20261113/README.md) | 機構 (a)〜(c) は HYPOTHESIS で、定理ではない |
| B7 | 割当て全体の Laplace は大きな誤りは退けるが、1 列の違いは区別できない | S | [AF4 README](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af4_laplace_model_comparison_20261113/README.md) | θ̂ の差と Laplace の誤差（O(1)）が重なる |
| B8 | 列ごとの Laplace 選択は sparse で Q を上回る（+0.40 [0.05, 0.70]） | S → **N**（AF6 で未達成） | [AF5 README](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af5_local_laplace_selection_20261113/README.md) | 手法を作ったのと同じ 20 データ |
| B9 | 新しい 40 データの事前登録確認：sparse の A−B は +0.225 [−0.075, 0.525]（未達成） | **N**（confirmatory）＋ R-calc（957 スコアを独立に再現） | [AF6 README](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af6_laplace_confirmation_20261114/README.md) | 検出力は約 0.30。効果がないことも示していない |
| B10 | Laplace 型基準の副作用は、すべて Poisson→CMP2 の向き（sparse 22 列）。TI でも同じなので、Z の Laplace 誤差によるものではない | S（AF6 の中の観察） | 同上 | f の profile 化と θ の固定の影響は切り分けていない |
| B11 | 判定境界のずれは、profile 型の基準で真の分布によらず中央値 +1.0 CMP2 側。真の θ を anchor にしても L の AUC は 0.968 | **P** | [AF7 README](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af7_cmp_bias_analysis_20261114/README.md) | AF7 のチャット報告の 3 点は AF8 の監査で訂正した（[`mathematical_derivation.md` §0](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af8_integrated_loading_20261115/mathematical_derivation.md)） |
| B12 | 積分した基準は prior（測度）に依存し、prior B は平坦極限でも −1.00 nat の候補間の差を残す | T | 同上 §1、§1.1 | A''_CMP2(0) = 0.5131（コードで確認） |
| B13 | E2 は数値的に安定して評価できる（8,640 セル、S0–S9 を通過、|ΔE2−ΔIS| の median 0.017） | FACT（数値監査）＋独立監査 | [AF8-v3 final `numerical_audit.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af8v3_full_exploratory/numerical_audit.md) | Z の Laplace 誤差は検証していない。IS は E2 と同じ被積分関数を使う |
| B14 | E2 − AF6 Laplace：balanced +0.023 [−0.004, +0.050]、P→C 37→17、CMP2 の正解 228→216 | S（AF6 のデータを再利用）、**探索的** | [AF8-v3 final README](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af8v3_full_exploratory/README.md) | 主手法の選び方に forking paths がある（監査 W4）。多重比較の補正なし |
| B15 | E2 の balanced は prior の尺度で 0.68〜0.915 と変わる | S | 同上、`prior_sensitivity.csv` | 事前に指定した 9 つの prior |
| B16 | prior による縮小と積分の効果の配分（D−B、E2−D） | **P**＋座標に依存 | 同上 §4、`independent_audit.md` W3 | 手法そのものの性質ではない |

## C. 再現の状況のまとめ

| 状況 | 該当 |
|---|---|
| 新しいデータで同じ向きが再現した | （該当なし。Q の正解率の水準が AF2 と AF6 で近いのは参考） |
| 同じデータ・別の seed で再現した | A5・A6（M1 → E1、sparse の joint mode） |
| 新しいデータで再現しなかった、または未達成 | B8 → B9（AF5 の改善は AF6 で未達成）。A5 の dense の last 推定 |
| 計算の独立な再現だけ | A2、B9（AF6 のスコア）、B13（AF8-v3 の独立監査） |
| 事前登録した 1 回の実行（独立な再現はない） | B2、B4、B5（AF1・AF2） |
| 同じデータの上で手法を設計・評価したもの | B6〜B8（AF2 の 20 データ）、B14〜B16（AF6 の 40 データ） |
| 事後解析 | B11、B16、AF2 の事後表、AF6 §4.4 の状態別の集計 |
| 理論（条件つき） | A10、A11、B1、B3、B12 |

## D. 書いてはいけないこと（AF に関する追加分、既存の NOT ALLOWED に加える提案）

既存の Claim Ledger に正式に追加するかどうかは、研究者が判断する。以下は提案である。

- 「自動分布選択によって Z の推定が一般に改善する」：AF1 は解析的に決まるケース。AF2 と PR #129 の Support-only に対する改善は、1 条件ずつの小さな差で、新しいデータでは再評価していない。AF5〜AF8 は Z を評価していない。限定付きの記述（B5）は可能。
- 「Laplace による分布選択は Q より優れている」：AF6 の主要評価項目は未達成。
- 「loading を積分すると分布選択が改善する」：AF8-v3 は境界の移動で、prior に依存する（NO-GO）。
- 「E2 は周辺尤度・Bayesian evidence・BIC である」：θ_{−j} を固定しており、Z も Laplace 近似。
- 「AF8-v3 の 40 データの結果は確認的な証拠である」：AF6 のデータの再利用で、forking paths もある。
