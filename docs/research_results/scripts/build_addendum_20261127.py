"""Addendum (2026-11-27): editorial records for the two-gaps Experiment A/B and their follow-up analyses.

Like build_tables.py, these rows are records written by the auditor (they contain judgement); they are not
computed from primary data. Numbers quoted here are verified separately by verify_addendum_20261127.py
(result_ids point to data/addendum_20261127_primary_results.csv).

The 2026-11-27 snapshot tables (experiment_inventory.csv 88 rows, comparison_fairness_matrix.csv 32 rows,
claims_evidence_matrix.csv 36 rows, source_coverage.csv, primary_results_index.csv 96 items) are NOT modified.

Counting rule: Experiment A and Experiment B are 2 experiments (new EM fits). The post-hoc diagnostics
(2f71e61) and the forward correction (c035df7) analyse the same saved fits; they are recorded as analyses
attached to A/B (counts_as_new_experiment = no) and are not counted as additional experiments.

Usage (repository root): python docs/research_results/scripts/build_addendum_20261127.py
"""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
GH = "https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob"
TG = "expfam/results/k_selection/two_gaps_20261127"

INVENTORY = [
    {"experiment_id": "TG-A", "counts_as_new_experiment": "yes", "domain": "B",
     "name": "Experiment A：Y の観測量 × X の分布誤指定（全列共通 family）",
     "question": "X の全列が Bernoulli のとき、学習 Y が少ないほど Bernoulli→Poisson の誤指定による Z の悪化は大きいか",
     "type": "実験（探索的、protocol は fit 前に凍結）", "lineage": "E-scalar（DualExpFamLSMConsistent、prototype）",
     "where": "experiment/k-exp2-two-gaps-20261127（未 merge）", "commit": "1ef9096",
     "conditions": "n=150、d=15、K_TRUE=3、全列 Bernoulli-X・Bernoulli-Y、dense（学習 Y 100%）/ sparse（10%）、20 データ × 2 初期値、L=10、MCEM 10 反復",
     "comparators": "正指定 Bernoulli-X vs 誤指定 Poisson-X（同じデータ・mask・seed）",
     "metrics": "primary：I = Δ_sparse − Δ_dense（Procrustes RMSE_Z、Z_last）。副次：gram 相関、Z_mean の RMSE_Z、held-out 対数スコア",
     "run_count": "160 fit（160/160 OK）", "outcome_status": "完了",
     "primary_result": "I = +0.211（20/20 正）、Δ_dense +0.012、Δ_sparse +0.223。sparse では正指定でも RMSE_Z ≈ 1.02・gram 相関 ≈ 0.24",
     "uncertainty": "bootstrap 95% [0.179, 0.242]（20 データ）", "evidence": f"{TG}/expA/analysis/primary_results.csv",
     "comparability": "正指定 vs 誤指定 A、dense vs sparse A（限定付き）、M1・MX2 と C、fixed Exp4 と D",
     "adoption": "Human Gate（prototype、探索的）",
     "limitations": "1 信号設計・1 種類の誤指定・1 sparse 率・10 反復固定。sparse では正指定自身の回復が弱い",
     "completeness": "CONFIRMED", "result_ids": "TG-A-01;TG-A-02;TG-A-03;TG-A-04;TG-A-05;TG-A-06;TG-A-07;TG-A-08"},
    {"experiment_id": "TG-B", "counts_as_new_experiment": "yes", "domain": "F",
     "name": "Experiment B：Fisher 情報の trace をそろえた Gaussian-X と Poisson-X の K 選択",
     "question": "K-exp2 regime b・K_TRUE=9 の C_Q の差（Gaussian 10/10、Poisson 0/10）は、潜在情報量の差で説明できるか",
     "type": "実験（探索的、protocol は fit 前に凍結）", "lineage": "E-scalar（prototype）",
     "where": "experiment/k-exp2-two-gaps-20261127（未 merge）", "commit": "1ef9096",
     "conditions": "n=150、d=15、K_TRUE=9、候補 K=1..10、regime b の信号、10 データ × 2 初期値、3 条件（G01 分散 0.1、GM 分散 ≈0.76、P Poisson）で Z・F・Y を共有、L=10、10 反復",
     "comparators": "G01 vs GM（分散だけ違う）、GM vs P（期待 Fisher の trace をそろえた）、C_Q vs C_Lap（同じ fit）",
     "metrics": "選ばれた K の正解・過小・過大・INCOMPLETE（C_Lap は 20 評価中 1 つでも非 OK なら INCOMPLETE、分母 10）",
     "run_count": "600 fit（600/600 OK）、C_Lap 600 評価（OK 598、NOT_STATIONARY 2）", "outcome_status": "完了",
     "primary_result": "C_Q：G01 10/10、GM 0/10（過小 10）、P 0/10（過小 10）。C_Lap：G01 9/10（INCOMPLETE 1）、GM 8/10（過小 2）、P 9/10（INCOMPLETE 1）",
     "uncertainty": "10 データ、CI なし", "evidence": f"{TG}/expB/analysis/primary_results.csv",
     "comparability": "G01 vs GM A、GM vs P B（trace のみ一致）、C_Q vs C_Lap B、K-exp2 本実験と C",
     "adoption": "Human Gate（prototype、探索的）",
     "limitations": "trace だけを一致させた。1 つの K_TRUE・信号規則、10 データ、10 反復固定",
     "completeness": "CONFIRMED", "result_ids": "TG-B-01;TG-B-02;TG-B-03;TG-B-04;TG-B-05;TG-B-06;TG-B-07;TG-B-08"},
    {"experiment_id": "TG-DIAG", "counts_as_new_experiment": "no（TG-A・TG-B の保存済み fit の事後分析）", "domain": "B;F",
     "name": "A・B の原因分析（保存済み状態のみ、新しい fit なし）",
     "question": "A で誤指定が Z を悪くしたのに Y の予測を良くしたのはなぜか。B で C_Q が過小選択したのはなぜか",
     "type": "事後の探索的分析（protocol は計算前に commit：7447ef0）", "lineage": "E-scalar（prototype）",
     "where": "audit/two-gaps-diagnostics-20261127（未 merge）", "commit": "2f71e61",
     "conditions": "NPZ 760 個（SHA-256 一致）、seed から再構成したデータ（hash 一致）", "comparators": "—",
     "metrics": "予測の分解（AUC・AP・ECE・校正の傾き・定数予測）、推定分散・推定情報量、C_Q の 1 サンプル項分解",
     "run_count": "0 fit", "outcome_status": "完了（一部訂正あり → TG-CORR）",
     "primary_result": "A の逆転は順位付けでなく予測確率の広がりに伴う。B の推定後の X 情報（1 次元あたり）G01 9.25・GM 1.37・P 1.17",
     "uncertainty": "bootstrap（A）、記述（B）", "evidence": f"{TG}/diagnostics_20261127/diagnostic_report.md",
     "comparability": "—", "adoption": "Human Gate",
     "limitations": "165 行の不等号の符号が誤り（TG-CORR で訂正）。項分解は 1 サンプル近似",
     "completeness": "CONFIRMED", "result_ids": "TG-A-09;TG-A-10;TG-A-11;TG-A-12;TG-A-13;TG-A-15;TG-A-16;TG-B-09;TG-B-10;TG-B-17;TG-C-02;TG-C-03"},
    {"experiment_id": "TG-CORR", "counts_as_new_experiment": "no（TG-DIAG の訂正）", "domain": "B;F",
     "name": "原因分析の数理的な最終訂正（forward correction）",
     "question": "C_Q の K 増加条件・X+Y の改善量・項分解・校正の表現・Z_mean・C_Lap の式と実装は正しいか",
     "type": "訂正（THEORY / AUDIT）", "lineage": "E-scalar（prototype）",
     "where": "audit/two-gaps-diagnostics-correction-20261127（未 merge）", "commit": "c035df7",
     "conditions": "per_fit.csv・NPZ（var_z・Z_mean）・既存の診断 CSV", "comparators": "—",
     "metrics": "2Δℓ（保存 C_Q から厳密に復元）、隣接条件と C_Q の上下の一致、C_Lap の閉じ方",
     "run_count": "0 fit", "outcome_status": "完了",
     "primary_result": "正しい条件は 2Δℓ > n(1+ln2π)+(d−K+1)ln n（270/270 一致）。K=8→9 の 2Δℓ 中央値 G01 901.83・GM 311.63・P 290.30、必要量 460.756",
     "uncertainty": "厳密な恒等式（数値誤差 3e-12）", "evidence": f"{TG}/diagnostics_20261127/forward_correction_20261127/forward_correction.md",
     "comparability": "—", "adoption": "Human Gate",
     "limitations": "ℓ は最後の E-step の MC サンプルでの平均（観測データ尤度ではない）。X と Y の厳密な分離はできない",
     "completeness": "CONFIRMED", "result_ids": "TG-A-14;TG-A-16;TG-B-11;TG-B-12;TG-B-13;TG-B-14;TG-B-15;TG-B-16;TG-C-01"},
]

FAIRNESS = [
    ("FC-TG-01", "A：同じデータでの正指定 Bernoulli-X vs 誤指定 Poisson-X", "TG-A", "A", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes",
     "160/160 OK", "事前（protocol 凍結）", "family だけが違う（独立集計で確認）", "同じデータで誤指定は Z の回復を悪くした", "誤指定の一般的な効果"),
    ("FC-TG-02", "A：dense vs sparse（交互作用 I）", "TG-A", "A（限定付き）", "yes", "yes", "部分集合", "Y の量だけ", "yes", "yes", "no（意図した操作）", "yes",
     "160/160 OK", "事前", "sparse の学習 mask は dense の部分集合。MC・初期化の経路も変わる", "この設定で、Y が少ないと誤指定の悪影響が大きかった", "「Y が少ないほど誤指定の悪影響が大きい」という一般法則"),
    ("FC-TG-03", "A と M1（#134）・MX2（PR #168）", "TG-A;D-M1;D-MX2", "C", "no", "no", "no", "no", "no（E-scalar vs E-3）", "similar", "no", "yes",
     "—", "—", "列ごとの family・n=75・d=12 と、全列共通・n=150・d=15", "向きが同じだったという記述", "数値の横並び比較"),
    ("FC-TG-04", "A と fixed Exp4（系列 C）", "TG-A;B-FIXED-EXP4", "D", "no", "no", "no", "no", "no", "no", "no", "partly",
     "—", "—", "系列・生成器・Y の観測量の操作の有無が違う", "なし（別の実験として並記のみ）", "倍率や RMSE の数値比較"),
    ("FC-TG-05", "A：Z_last plug-in と Z_mean plug-in の予測", "TG-DIAG;TG-CORR", "B", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes",
     "—", "事後", "同じ fit だが Z_mean は縮んでいる（二乗平均 0.60・0.42）。どちらも事後予測分布ではない", "予測の作り方で逆転が消えた", "事後予測分布でも逆転が消える／残る"),
    ("FC-TG-06", "B：同じ replicate の G01 vs GM", "TG-B", "A", "yes（Z・F・Y 共有）", "yes", "n/a", "X の分散だけ", "yes", "yes", "no（意図した操作）", "yes",
     "C_Lap INCOMPLETE は分母に残す", "事前", "X の雑音の分散だけが違う", "X の情報量を下げると C_Q の成否が入れ替わった", "—"),
    ("FC-TG-07", "B：GM vs P（期待 Fisher の trace をそろえた）", "TG-B", "B", "yes（Z・F・Y 共有）", "yes", "n/a", "family が違う", "yes", "yes", "trace のみ", "yes",
     "同上", "事前", "方向ごとの情報・z ごとの変動・分散推定（罰則 +d）・尤度の形が違う", "trace をそろえると C_Q の結果は似た", "Poisson 分布そのものが原因／分布族は無関係"),
    ("FC-TG-08", "B：同じ fit での C_Q vs C_Lap", "TG-B", "B", "yes", "yes", "n/a", "yes", "yes", "yes", "yes", "基準が違う",
     "INCOMPLETE は C_Lap 側だけ", "事前（選択規則）", "FC-14 と同じ理由。θ̂ は C_Lap の停留点ではなく単一モード", "同じ推定結果で選ぶ K が違った", "C_Lap が正しい基準"),
    ("FC-TG-09", "B と K-exp2 本実験（O-b-9、A-b-9）", "TG-B;F-KEXP2-MAIN", "C", "no（別データ）", "no", "n/a", "同じ設計規則", "yes", "yes", "—", "yes",
     "—", "—", "乱数の使い方とデータが違う", "C_Q の差が再現した", "C_Lap の正解数の横並び比較"),
    ("FC-TG-10", "B：family の違う条件間で C_Q・C_Lap の絶対値を比べる", "TG-B", "D", "—", "—", "—", "—", "—", "—", "—", "no",
     "—", "—", "尤度の正規化・罰則の項数（C_Q は Gaussian で +d、C_Lap の d_K は分散を含まない）が違う", "選ばれた K だけを比べる", "基準値の絶対値の比較"),
    ("FC-TG-11", "B：K=8→9 の 2Δℓ（厳密）と 1 サンプルの X・Y 項", "TG-CORR", "B", "yes", "yes", "n/a", "yes", "yes", "yes", "yes", "近似 vs 厳密",
     "—", "事後", "X・Y の分離は Z_last 1 サンプルの近似、中央値は足し合わせられない", "X+Y の合計は厳密に言える", "X と Y の寄与の正確な配分"),
]
FAIR_COLS = ["comparison_id", "comparison", "experiments", "grade", "same_data", "same_seed", "same_split_mask",
             "same_conditions", "same_lineage", "same_budget", "same_information", "same_metric", "failure_handling",
             "prereg_vs_posthoc", "reason", "allowed_reading", "forbidden_reading"]

CLAIMS = [
    ("CL-TG-01", "B", "X の全列が Bernoulli の設定で、学習 Y を 10% にすると Bernoulli→Poisson の誤指定による RMSE_Z の悪化が大きくなった（I = +0.211、20/20）", "OBSERVED", "限定付き",
     "E-scalar", "1ef9096 expA/analysis", "TG-A-02;TG-A-03;TG-A-04", "sparse では正指定でも回復が弱い（RMSE_Z ≈ 1.02、gram ≈ 0.24）。1 設定・1 種類の誤指定", "一般法則として書く"),
    ("CL-TG-02", "B", "sparse-Y では、正指定のほうが予測の順位付け（AUC・AP）で 20/20 データとも良かった", "OBSERVED", "限定付き",
     "E-scalar", "2f71e61 prediction_diagnostics.csv", "TG-A-09;TG-A-10", "Z_last による plug-in 予測", "—"),
    ("CL-TG-03", "B", "sparse-Y の Z_last plug-in 予測では、対数スコアと ECE は誤指定のほうが良く（19/20、20/20）、予測確率の SD は小さかった。校正の傾きは正指定のほうが 1 に近かった（18/20）", "OBSERVED", "限定付き",
     "E-scalar", "2f71e61・c035df7", "TG-A-11;TG-A-12;TG-A-14;TG-A-15", "両モデルとも定数予測より悪い（20/20）。Z_mean では逆転が消える", "「誤指定のほうが校正が良い」「誤指定のほうが予測モデルとして優れている」"),
    ("CL-TG-04", "B", "Z_mean による plug-in 予測では、対数スコアの逆転が消えた（正指定が良いデータ 14/20）", "OBSERVED", "限定付き",
     "E-scalar", "2f71e61・c035df7", "TG-A-16", "Z_mean は縮んでいる。事後予測分布ではない", "事後予測分布でも逆転が消える"),
    ("CL-TG-05", "F", "期待 Fisher 情報の trace を Poisson（約 9.8）にそろえた Gaussian-X（GM）は、C_Q で 0/10（すべて過小）だった。G01（trace 75）は 10/10", "OBSERVED", "限定付き",
     "E-scalar", "1ef9096 expB/analysis", "TG-B-02;TG-B-04;TG-B-06;TG-B-08", "trace だけの一致。1 つの K_TRUE", "Poisson 分布そのものが失敗の原因／分布族は無関係"),
    ("CL-TG-06", "F", "scale_Z と var_z = 1 のもとでは、保存 C_Q の Z の事前密度の項は nK(1+ln2π) に等しく、C_Q(K) < C_Q(K−1) ⇔ 2Δℓ > n(1+ln2π)+(d−K+1)ln n", "DERIVED", "言える（この実装条件）",
     "E-scalar", "c035df7 forward_correction.md §1、コード", "TG-B-11;TG-B-14", "E-scalar の実装条件（scale_Z、var_z=1）。他の系列は別途確認", "C_Q を Schwarz BIC と呼ぶ"),
    ("CL-TG-07", "F", "K=8→9 の X+Y の改善 2Δℓ（10 データ中央値）は G01 901.83、GM 311.63、P 290.30 で、必要量 460.756 を超えたのは G01 だけ（10/0/0）", "OBSERVED", "限定付き",
     "E-scalar", "c035df7 cq_exact_increment_summary.csv", "TG-B-12;TG-B-13", "ℓ は最後の E-step の MC サンプルでの平均。2 初期値平均の中央値", "X 側・Y 側の正確な配分"),
    ("CL-TG-08", "F", "C_Q の過小選択は、1 次元あたりの改善が K に比例する固定のコストに届かないことと整合する。推定後の X 情報は GM と P で同程度（1.37・1.17）", "INTERPRETATION", "限定付き",
     "E-scalar", "2f71e61・c035df7", "TG-B-10;TG-B-12;TG-B-15", "探索的・事後。隣接条件と大域 argmin は 29/30 で一致（1 データは非単峰）", "因果の証明"),
    ("CL-TG-09", "F", "同じ fit で C_Lap は GM 8/10（過小 2）、P 9/10（INCOMPLETE 1）、G01 9/10（INCOMPLETE 1）", "OBSERVED", "限定付き",
     "E-scalar", "1ef9096 expB/analysis", "TG-B-03;TG-B-05;TG-B-07", "C_Lap は厳密な周辺尤度ではない。θ̂ は MCEM の値、単一モード。INCOMPLETE は分母に残す", "C_Lap が正しい・常に優れる・過大選択しない"),
    ("CL-TG-10", "F", "10 反復で MCEM が収束したか", "UNRESOLVED", "言えない",
     "E-scalar", "Q の履歴は保存されていない", "", "保存は最後の相対変化（中央値 2e-4〜1.9e-3）と Q の減少回数（0）だけ", "収束した"),
]
CLAIM_COLS = ["claim_id", "domain", "claim", "label", "claimability", "lineage", "evidence", "result_ids",
              "required_qualifiers", "forbidden_overclaim"]


def write(name, cols, recs):
    with open(DATA / name, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in recs:
            w.writerow([r[c] for c in cols] if isinstance(r, dict) else r)


def main():
    cols = ["experiment_id", "counts_as_new_experiment"] + [c for c in INVENTORY[0] if c not in ("experiment_id", "counts_as_new_experiment")]
    write("addendum_20261127_inventory.csv", cols, INVENTORY)
    write("addendum_20261127_fairness.csv", FAIR_COLS, FAIRNESS)
    write("addendum_20261127_claims.csv", CLAIM_COLS, CLAIMS)
    print(len(INVENTORY), "inventory rows (new experiments:",
          sum(r["counts_as_new_experiment"] == "yes" for r in INVENTORY), ")",
          len(FAIRNESS), "comparisons", len(CLAIMS), "claims")


if __name__ == "__main__":
    main()
