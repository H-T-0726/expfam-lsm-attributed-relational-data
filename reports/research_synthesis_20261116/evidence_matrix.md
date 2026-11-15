# Evidence matrix：研究上の主張と一次資料の対応

## 区分

| 区分 | 意味 |
|---|---|
| **R-data** | 再現済み（データ）：独立な新しいデータまたは seed 系列で、同じ向きの結果が再び得られた |
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
| A1 | X 側と Y 側を、G/B/P から独立に指定できる実装がある | FACT（code） | B | [`model_dual_expfam.py`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/src/model_dual_expfam.py)、`test_dual_expfam.py` | ALLOWED | 実装レベルの一般化で、性能の主張ではない。Categorical は未実装 |
| A2 | 先行研究と同じ条件で、再現実装と一致する（RMSE(Z) の差 < 0.001） | R-calc | B | [`comparison_main_table.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reproduction/results/comparison/comparison_main_table.csv) | ALLOWED | 5 試行だけ |
| A3 | n=50→300 で RMSE(Z) が下がる（A −49.3%、B −41.2%、C −58.6%） | S（3 シナリオ） | C | [`fixed_exp2_n_sweep_improvement.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/fixed_official/exp2/fixed_exp2_n_sweep_improvement.csv) | ALLOWED | 明示した生成設定の中だけ。旧 0.5 系列（49%、31%、62%）と同じ表に置かない |
| A4 | 分布の誤指定で RMSE(Z) が悪化する（最悪 A 4.34×、B 9.04×、C 40.37×） | S（系列 B でも同じ向き） | C | [`fixed_official/exp4/`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/fixed_official/exp4) | ALLOWED | 大きな倍率は主に Y 側の誤指定による。X 側だけの最大は 4.34×（support-invalid） |
| A5 | support-compatible な X の誤指定（B→P）で、Z の推定誤差が増える。dense +0.020（15/20）、sparse +0.066（17/20） | **R-data**（E1 Part B で sparse を再現：mode で +0.087、20/20） | E | [M1](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/misspecification_m1/phase_m1_20261013)、[E1](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/z_point_estimator_e1/e1_20261104) | 未決定（prototype） | 1 つの固定条件。dense の last 推定は再現しなかった（A +0.020、B +0.002） |
| A6 | 誤指定の悪影響は Y の情報が少ないほど大きい（B→P の場合） | R-data（M1 と E1） | E | 同上、[PR #168 の結果](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/856d475/reports/misspecification_extended/results_20261110.md) | 未決定 | **誤指定の種類によって交互作用の向きが逆になる**（#168：B6→G6 や all-Gaussian では Y=100% のほうが大きい）。一般則ではない |
| A7 | dense-Y では all-Gaussian でも正しい family とほぼ同じで、Y を 1 割にすると明確に負ける | S（2 系列で同じ向き） | F / E | [`distribution_misspecification_synthesis_20261013.md` §6](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reports/advisor_feedback/distribution_misspecification_synthesis_20261013.md) | QUALIFIED | 系列と生成器によって大きさが違う（+0.12〜+0.43） |
| A8 | sparse-Y・補完的な属性 block の人工設定では、per-column の同時利用で RMSE_Z が改善する（y_only 比 +0.51、single_gaussian 比 +0.20） | S | E+F | `complementary_blocks_consistent_20260821_paired.csv`（[§12.3](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md)） | QUALIFIED | prototype。dense では +0.009 まで縮む。機構は未確立（#31） |
| A9 | MovieLens（user-disjoint）では、平均の向きは正（+0.0042、23/30）で、変動に比べて小さい | S | E+F | [§12.5](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md) | QUALIFIED | 30 の split は独立な replicate ではない。CI は計算していない |
| A10 | Poisson-X で FFᵀ をモーメントから復元できる。Gaussian-Y では 1 つの dyad から K が決まる。族は入れ子でない（P1、P2、P3） | T | — | [§17.3](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md) | ALLOWED（前提を明記） | population の主張。**Bernoulli-Y は未解決** |
| A11 | C_Q は Schwarz BIC ではなく、ICL 型の基準である | T / FACT（code） | B/C/E | KI-010、§17.4 | ALLOWED | 関数名・CSV の列名は変更しない |
| A12 | C_Lap は、一部の固定条件で C_Q より真の K を多く選んだ（9K：19/20 対 14/20） | S | E | [§19](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md) | QUALIFIED・本文採用は未決定 | 一般の優越ではない。seed の組で変わる |

## B. 分布の自動選択（AF。すべて lineage E の prototype で、本文採用は未決定）

| # | 主張 | 区分 | 一次資料 | 限定 |
|---|---|---|---|---|
| B1 | {0,1} 列では、どの η でも log Bern − log Pois = e^η − log(1+e^η) > 0 | T | [AF1 README §0](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af1_20261109/README.md) | 同じ η での比較。最適化が届いている場合 |
| B2 | Bernoulli vs Poisson の自動選択は 160/160 fit で全列正解し、Wrong による悪化（sparse 0.0816）を回避した | S（結果は B1 からほぼ決まる） | [AF1 `full/`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/auto_family_z_recovery/af1_20261109/full) | Auto = Oracle は refit の seed を共有した設計上の性質 |
| B3 | {0,1} 列では CMP2 も Poisson に勝つ。切片なしの n=75 では、{0,1} だけの真 Poisson 列の確率は ≤ 1.01e-10 | T ＋数値計算 | [AF2 Part A](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/auto_family_z_recovery/af2_part_a_20261110) | 負の切片を仮定すると頻繁に起き、選択と交絡する |
| B4 | Poisson vs CMP2 の Q 選択の正解は dense 108/120、sparse 92/120 で、誤りはほぼ CMP2→Poisson | S | [AF2 README](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af2_20261110/README.md) | 1 条件・20 データ。AF2 の Q の sparse 正解率は 0.767（92/120）、AF6 の新しいデータでは 0.813 で、近い水準だった（参考） |
| B5 | Auto の RMSE(Z) の Support-only に対する上積みは小さい（dense 0.0051、sparse 0.0059） | S | 同上 | 1 条件。PR #129 でも Support-only に対して −0.009〜−0.043 と小さい（別の設計） |
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
| 新しいデータ・seed で同じ向きが再現した | A5・A6（M1 → E1、sparse の joint mode） |
| 新しいデータで再現しなかった、または未達成 | B8 → B9（AF5 の改善は AF6 で未達成）。A5 の dense の last 推定 |
| 計算の独立な再現だけ | A2、B9（AF6 のスコア）、B13（AF8-v3 の独立監査） |
| 同じデータの上で手法を設計・評価したもの | B4〜B8、B14〜B16 |
| 事後解析 | B11、B16、AF2 の事後表、AF6 §4.4 の状態別の集計 |
| 理論（条件つき） | A10、A11、B1、B3、B12 |

## D. 書いてはいけないこと（AF に関する追加分、既存の NOT ALLOWED に加える提案）

既存の Claim Ledger に正式に追加するかどうかは、研究者が判断する。以下は提案である。

- 「自動分布選択によって Z の推定が改善された」：AF1 は自明なケース。AF2 の上積みは小さく、AF5〜AF8 は Z を評価していない。
- 「Laplace による分布選択は Q より優れている」：AF6 の主要評価項目は未達成。
- 「loading を積分すると分布選択が改善する」：AF8-v3 は境界の移動で、prior に依存する（NO-GO）。
- 「E2 は周辺尤度・Bayesian evidence・BIC である」：θ_{−j} を固定しており、Z も Laplace 近似。
- 「AF8-v3 の 40 データの結果は確認的な証拠である」：AF6 のデータの再利用で、forking paths もある。
