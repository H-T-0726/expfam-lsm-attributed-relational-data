"""Read-only re-aggregation of key numbers from primary artifacts.

Every input is read with ``git show <commit>:<path>``, so the script works on any
clone that has fetched the listed commits (see REFS). Nothing is fitted, no
model is imported, and no existing artifact is written.

Output: docs/research_results/data/primary_results_index.csv
  status = VERIFIED   recomputed value equals the reported value (within the stated rounding)
           MISMATCH   recomputed value differs (difference recorded, source file NOT modified)
           UNVERIFIED the input could not be read or the quantity could not be recomputed

Usage (from the repository root):
    python docs/research_results/scripts/verify_primary_numbers.py
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

REFS = {
    "MAIN": "1b5030b",      # origin/main (PR #154)
    "KEXP2": "9d12dc4",     # experiment/k-selection-exp2-generalization-20261125 (unmerged)
    "PF3": "bada1d2",       # research/paper-fig3-condition-criteria-20261109 (#155-#165 stack, unmerged)
    "THESIS": "8a66894",    # docs/thesis-adoption-20261124 (AF1-AF8, CORA-SH1; unmerged)
    "MX2": "856d475",       # experiment/misspecification-extended (PR #168, unmerged)
    "PCMIX": "d610a8d",     # experiment/per-column-mixed-x-k-selection-20260907 (unmerged)
    "HPB": "600108d",       # research/historical-pb-k5-n-sweep-20260906 (unmerged)
}
KX = "expfam/results/k_selection/k_exp2_generalization_20261125"
OUT = Path(__file__).resolve().parents[1] / "data" / "primary_results_index.csv"
ROWS: list[dict] = []


def show(ref: str, path: str) -> str:
    return subprocess.run(["git", "show", f"{REFS[ref]}:{path}"], check=True,
                          capture_output=True).stdout.decode("utf-8")


def rows(ref: str, path: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(show(ref, path))))


def num(x) -> float:
    s = str(x).strip()
    if s.startswith("np.float64(") and s.endswith(")"):
        s = s[len("np.float64("):-1]
    return float(s)


def record(rid, domain, exp, quantity, reported, reported_in, ref, path, method, recomputed, tol=None):
    """reported/recomputed may be numbers or strings. tol=None means exact string equality."""
    if recomputed is None:
        status, rec = "UNVERIFIED", ""
    else:
        if tol is None:
            ok = str(reported) == str(recomputed)
        else:
            ok = abs(float(reported) - float(recomputed)) <= tol
        status = "VERIFIED" if ok else "MISMATCH"
        rec = recomputed
    ROWS.append(dict(result_id=rid, domain=domain, experiment_id=exp, quantity=quantity,
                     reported_value=reported, reported_in=reported_in, recomputed_value=rec,
                     status=status, source_commit=REFS[ref], source_path=path, method=method))


def guarded(fn):
    def run():
        try:
            fn()
        except Exception as e:  # keep going; the failed check is recorded as UNVERIFIED
            ROWS.append(dict(result_id=fn.__name__, domain="", experiment_id="", quantity=f"ERROR: {e!r}"[:200],
                             reported_value="", reported_in="", recomputed_value="", status="UNVERIFIED",
                             source_commit="", source_path="", method=""))
    return run


# ---------------------------------------------------------------- A: original model / reproduction
@guarded
def a_reproduction():
    p = "reproduction/results/comparison/comparison_main_table.csv"
    r = [x for x in rows("MAIN", p) if x["scenario"].startswith("Control")]
    by = defaultdict(dict)
    for x in r:
        by[int(x["k_est"])][x["model"]] = num(x["rmse_Z_mean"])
    models = sorted(by[3])
    vals3 = sorted(by[3].values())
    record("A-01", "A", "Control比較（先行研究再現）", f"k_est=3 RMSE(Z) 平均（{'/'.join(models)}）",
           "0.17934/0.17988", "claim_adoption_matrix.md 一次確認表", "MAIN", p,
           "Control シナリオの k_est=3 の 2 モデルの rmse_Z_mean", f"{vals3[0]:.5f}/{vals3[1]:.5f}")
    for k, rep in [(3, 0.00054), (4, 0.00026), (6, 0.0236)]:
        v = list(by[k].values())
        record(f"A-02-k{k}", "A", "Control比較（先行研究再現）", f"k_est={k} の 2 モデルの RMSE(Z) 差の絶対値",
               rep, "claim_adoption_matrix.md SY-2", "MAIN", p, "|差|", round(abs(v[0] - v[1]), 5),
               tol=0.00006 if k != 6 else 0.0006)


# ---------------------------------------------------------------- B: exponential-family extension (B and C lineages)
@guarded
def b_fixed():
    p = "expfam/results/fixed_official/exp1_k9/fixed_exp1_bic_k1to9_bestk_by_trial.csv"
    r = rows("MAIN", p)
    record("B-01", "B", "fixed official Exp1 (k1-9)", "C_Q（列名 BIC）が K=3 を選んだ trial 数 / 全 trial",
           "30/30", "claim_adoption_matrix.md SY-6", "MAIN", p, "best_k_by_trial_bic==3 を数える",
           f"{sum(x['best_k_by_trial_bic'] == '3' for x in r)}/{len(r)}")
    p = "expfam/results/fixed_official/exp2/fixed_exp2_n_sweep_improvement.csv"
    for x in rows("MAIN", p):
        rep = {"A": 0.4928, "B": 0.4124, "C": 0.5865}[x["scenario"]]
        rec = (num(x["rmse_z_at_n_min"]) - num(x["rmse_z_at_n_max"])) / num(x["rmse_z_at_n_min"])
        record(f"B-02-{x['scenario']}", "B", "fixed official Exp2 (n-sweep)",
               f"Scen.{x['scenario']} の RMSE(Z) 相対減少率 n=50→300", rep, "SY-3 / registry", "MAIN", p,
               "(rmse_n50 − rmse_n300)/rmse_n50 を列から再計算", round(rec, 4), tol=0.00006)
    p = "expfam/results/fixed_official/exp3/fixed_exp3_d_sweep_agg.csv"
    r = rows("MAIN", p)
    a = {int(x["d"]): num(x["rmse_z_mean"]) for x in r if x["scenario"] == "A"}
    b = {int(x["d"]): num(x["rmse_z_mean"]) for x in r if x["scenario"] == "B"}
    c = [num(x["rmse_z_mean"]) for x in r if x["scenario"] == "C"]
    record("B-03-A", "B", "fixed official Exp3 (d-sweep)", "Scen.A RMSE(Z) d 最小→最大",
           "0.2639→0.2046", "SY-4", "MAIN", p, "d の最小・最大での平均",
           f"{a[min(a)]:.4f}→{a[max(a)]:.4f}")
    bd = min(b, key=lambda k: b[k])
    record("B-03-B", "B", "fixed official Exp3 (d-sweep)", "Scen.B RMSE(Z) が最小になる d と値",
           "d=15:0.1401", "SY-4", "MAIN", p, "argmin", f"d={bd}:{b[bd]:.4f}")
    record("B-03-C", "B", "fixed official Exp3 (d-sweep)", "Scen.C RMSE(Z) の範囲",
           "0.0231–0.0238", "SY-4", "MAIN", p, "min–max", f"{min(c):.4f}–{max(c):.4f}")
    for s, rep in [("a", 4.3414), ("b", 9.0405), ("c", 40.3706)]:
        p = f"expfam/results/fixed_official/exp4/fixed_exp4_scen_{s}_ratios.csv"
        r = rows("MAIN", p)
        m = max(num(x["ratio_vs_oracle"]) for x in r if x["fix_w"] == "False" and x["fix_x"] == "False")
        record(f"B-04-{s.upper()}", "B", "fixed official Exp4 (誤指定)",
               f"Scen.{s.upper()} 誤指定の RMSE(Z) 最大倍率（ablation 行を除く）", rep, "SY-5", "MAIN", p,
               "fix_w=False かつ fix_x=False の ratio_vs_oracle の最大", round(m, 4), tol=0.00006)
        if s == "c":
            record("B-04-Call", "B", "fixed official Exp4 (誤指定)", "Scen.C 全行（ablation 含む）の最大倍率",
                   46.8637, "claim_adoption_matrix.md 一次確認表", "MAIN", p, "全行の最大",
                   round(max(num(x["ratio_vs_oracle"]) for x in r), 4), tol=0.00006)


@guarded
def b_old05():
    for s, rep in [("A", 48.8), ("B", 31.0), ("C", 62.0)]:
        p = f"expfam/results/exp_scenario_{s}_exp2_n.csv"
        r = rows("MAIN", p)
        m50 = mean(num(x["rmse_Z"]) for x in r if int(float(x["n"])) == 50)
        m300 = mean(num(x["rmse_Z"]) for x in r if int(float(x["n"])) == 300)
        record(f"B-05-{s}", "B", f"Scen.{s} Exp2（旧 0.5 系列、予稿）", "RMSE(Z) 相対減少率 n=50→300（%）",
               rep, "registry 行（予稿 49/31/62%）", "MAIN", p, "trial 平均から再計算（%）",
               round(100 * (m50 - m300) / m50, 1), tol=0.55 if s == "A" else 0.6)
    for s, rep in [("A", 3.41), ("B", 7.35), ("C", 41.45)]:
        p = f"expfam/results/exp_scenario_{s}_exp4_mismatch.csv"
        r = rows("MAIN", p)
        g = defaultdict(list)
        for x in r:
            g[(x["condition"], x["fix_w"], x["fix_x"], x["correct"])].append(num(x["rmse_Z"]))
        oracle = [v for (c, fw, fx, ok), v in g.items() if ok == "True" and fw == "False" and fx == "False"]
        o = mean(oracle[0])
        ratios = {c: mean(v) / o for (c, fw, fx, ok), v in g.items() if fw == "False" and fx == "False"}
        best = max(ratios, key=lambda k: ratios[k])
        record(f"B-06-{s}", "B", f"Scen.{s} Exp4（旧 0.5 系列、予稿）", "誤指定の最大倍率（ablation を除く）",
               rep, "registry / mismatch_audit_summary.csv", "MAIN", p,
               "正指定条件の平均 RMSE(Z) に対する各条件の平均の比の最大", round(ratios[best], 2), tol=0.006)


# ---------------------------------------------------------------- D: per-column / mixed X
@guarded
def d_percolumn():
    p = "expfam/results/story_diagnostics/complementary_blocks_consistent_20260821_paired.csv"
    r = rows("MAIN", p)
    for rate, comp, rep, cnt in [("0.1", "y_only", 0.5122, 10), ("0.1", "single_bernoulli", 0.4218, 10),
                                 ("0.1", "single_poisson", 0.3889, 10), ("0.1", "single_gaussian", 0.2030, 9),
                                 ("1.0", "single_gaussian", 0.0087, 9), ("1.0", "y_only", 0.0510, 10),
                                 ("1.0", "single_bernoulli", 0.0530, 10), ("1.0", "single_poisson", 0.0490, 10)]:
        x = next(x for x in r if x["y_obs_rate"] == rate and x["comparator"] == comp)
        record(f"D-01-{rate}-{comp}", "D", "complementary blocks（#27）",
               f"y_obs_rate={rate} per_column_all vs {comp} の ΔRMSE_Z 平均（有利 trial 数/10）",
               f"{rep:+.4f} ({cnt}/10)", "SY-8 / RESEARCH_MASTER §12.3（dense の 3 行は独立監査の指摘で追加）", "MAIN", p, "paired CSV の行を読む",
               f"{num(x['delta_rmse_mean']):+.4f} ({x['n_favoring_per_column_rmse']}/10)")
    p = "expfam/results/story_diagnostics/matched_latent_coverage_ablation_20260821_interaction.csv"
    r = [x for x in rows("MAIN", p) if x["comparator"] == "single_gaussian" and x["is_primary"] == "True"]
    I = [num(x["I"]) for x in r]
    record("D-02", "D", "matched latent-coverage（#31）", "primary の交互作用 I の平均（正の trial 数）",
           "+0.114 (9/10)", "SY-9", "MAIN", p, "trial ごとの I を平均",
           f"{mean(I):+.3f} ({sum(v > 0 for v in I)}/{len(I)})")


@guarded
def d_m1_e1():
    p = "expfam/results/misspecification_m1/phase_m1_20261013/paired_rmse_z.csv"
    r = rows("MAIN", p)
    for reg, rep in [("dense", "+0.0200 (15/20)"), ("sparse", "+0.0662 (17/20)")]:
        d = [num(x[f"delta_RMSE_Z_{reg}"]) for x in r]
        record(f"D-03-{reg}", "D", "M1 Bernoulli→Poisson（#134）", f"{reg}-Y の ΔRMSE_Z（誤−正）平均（正の数/20）",
               rep, "MS-1", "MAIN", p, "replicate ごとの差を平均", f"{mean(d):+.4f} ({sum(v > 0 for v in d)}/{len(d)})")
    p = "expfam/results/z_point_estimator_e1/e1_20261104/full/paired_deltas.csv"
    r = rows("MAIN", p)
    sel = [num(x["delta_rmse_Z_sparse"]) for x in r if x["part"] == "B" and x["estimator"] == "mode"]
    record("D-04", "D", "E1（#141）", "Part B・joint mode・sparse の ΔRMSE_Z 平均（正の数）",
           "+0.0871 (20/20)", "MS-1", "MAIN", p, "part=B, estimator=mode",
           f"{mean(sel):+.4f} ({sum(v > 0 for v in sel)}/{len(sel)})")
    sel = [num(x["delta_rmse_Z_dense"]) for x in r if x["part"] == "B" and x["estimator"] == "last"]
    record("D-05", "D", "E1（#141）", "Part B・last・dense の ΔRMSE_Z 平均（正の数）",
           "+0.0021 (9/20)", "claim_adoption_matrix.md 一次確認表", "MAIN", p, "part=B, estimator=last",
           f"{mean(sel):+.4f} ({sum(v > 0 for v in sel)}/{len(sel)})")


@guarded
def d_mx2():
    p = "expfam/results/misspecification_extended/phase_20261110/paired_rmse_z.csv"
    r = [x for x in rows("MX2", p) if x["valid"] == "True" and x["arm"] == "bp6"]
    for yr, rep in [("0.1", 0.082), ("1.0", 0.012)]:
        d = [num(x["delta_rmse"]) for x in r if abs(num(x["y_rate"]) - float(yr)) < 1e-9]
        record(f"D-06-{yr}", "D", "MX2 誤指定の拡張（PR #168）", f"B6→P6、Y 観測率 {yr} の ΔRMSE_Z 平均（全 seed 系列）",
               rep, "AF 総括 evidence_matrix A6 / claim matrix MS-2", "MX2", p,
               f"valid=True・arm=bp6 の平均（n={len(d)}）", round(mean(d), 3), tol=0.0015)


@guarded
def d_pcmix_k():
    p = "expfam/results/percolumn_mixed_x_k_selection_20260907/selection_by_trial.csv"
    r = rows("PCMIX", p)
    record("D-07", "D", "per-column mixed-X K 選択（2026-09-07）", "全 config で K_0=3 を選んだ trial 数",
           "70/70", "aggregate_recovery.csv", "PCMIX", p, "trial 単位の correct を数える",
           f"{sum(x['correct'] == 'True' for x in r)}/{len(r)}")


# ---------------------------------------------------------------- E: family selection
@guarded
def e_af():
    p = "expfam/results/auto_family_z_recovery/af1_20261109/full/selection_columns.csv"
    r = [x for x in rows("THESIS", p) if x["method"].startswith("auto")]
    fits = {(x["series"], x["replicate"], x["regime"], x["method"]) for x in r}
    bad = sum(x["correct"] != "True" for x in r)
    record("E-01", "E", "AF1", "Auto の列選択の誤り数（Auto の fit 数）", "0 (160 fit)",
           "AF 総括 B2", "THESIS", p, "method が auto* の行で correct!=True を数える", f"{bad} ({len(fits)} fit)")
    p = "expfam/results/auto_family_z_recovery/af6_laplace_confirmation_20261114/statistical_analysis.csv"
    x = next(x for x in rows("THESIS", p) if x["endpoint"] == "PRIMARY")
    record("E-02", "E", "AF6（事前登録の確認）", "PRIMARY A−B sparse の平均差 [95% CI]、success",
           "+0.225 [-0.075, 0.525] False", "AF 総括 B9", "THESIS", p, "PRIMARY 行を読む（集計済み CSV）",
           f"{num(x['mean_D']):+.3f} [{num(x['ci_low']):.3f}, {num(x['ci_high']):.3f}] {x['success']}")


# ---------------------------------------------------------------- F: K selection
def argmin_k(d: dict, tol=1e-9):
    m = min(d.values())
    return min(k for k, v in d.items() if v <= m + tol)


@guarded
def f_early():
    p = "expfam/results/k_selection/heldout_full_pilot_20260824/replicate_selection.csv"
    r = rows("MAIN", p)
    sel = {}
    for x in r:
        sel[int(x["replicate"])] = int(x["selected_k"])
    record("F-01", "F", "Phase 7e held-out pilot", "replicate 1–3 の選択 K（K_TRUE=3）", "3,3,5",
           "KS-1 / RESEARCH_MASTER §12.7", "MAIN", p, "replicate ごとの selected_k",
           ",".join(str(sel[k]) for k in sorted(sel)))
    p = "expfam/results/k_selection/k_true_robustness_full_attempt2_20260904/selection_matrix.csv"
    r = [x for x in rows("MAIN", p) if x["lineage"] == "phase8a_new"]
    record("F-02", "F", "Phase 8b K_TRUE robustness（Attempt 2）", "新規グリッド（K_TRUE=1,2,4,5）での exact 数 / セル数",
           "18/24", "KS-1", "MAIN", p, "lineage=phase8a_new（K_TRUE=3 の Phase 7e anchor 6 行を除く）で label==exact を数える",
           f"{sum(x['label'] == 'exact' for x in r)}/{len(r)}")
    p = "expfam/results/k_selection/clean_true_k_asymptotics_20260904/selection_matrix.csv"
    r = rows("MAIN", p)
    for c, rep in [("S1", "2.62→5.00"), ("S2", "1.75→4.88")]:
        v = {n: mean(int(x["selected_k"]) for x in r if x["criterion"] == c and x["K_TRUE"] == "5" and x["n"] == str(n))
             for n in (50, 150)}
        record(f"F-03-{c}", "F", "clean true-K n-sweep", f"K_TRUE=5 の {c} 選択 K の平均 n=50→150",
               rep, "KS-2", "MAIN", p, "replicate 平均", f"{v[50]:.2f}→{v[150]:.2f}")


@guarded
def f_phase9():
    p = "expfam/results/k_repeatability/phase9e_20260927/k_repeatability.csv"
    r = rows("MAIN", p)
    record("F-04", "F", "Phase 9E K repeatability", "C_Q が K=3 を選んだデータ数 / 20", "17/20",
           "registry Phase 9 節", "MAIN", p, "k_hat==3 を数える",
           f"{sum(x['k_hat'] == '3' for x in r)}/{len(r)}")
    p = "expfam/results/lap_vs_cq_20/phase9k_20260928/laplace_by_k.csv"
    r = rows("MAIN", p)
    q, l = defaultdict(dict), defaultdict(dict)
    for x in r:
        q[x["replicate"]][int(x["k"])] = num(x["C_Q"])
        if x["laplace_status"] == "OK":
            l[x["replicate"]][int(x["k"])] = num(x["C_Lap"])
    cq = sum(argmin_k(v) == 3 for v in q.values())
    cl = sum(len(v) == 5 and argmin_k(v) == 3 for v in l.values())
    record("F-05", "F", "Phase 9K Lap vs Q（20 データ）", "K=3 を選んだ数 C_Lap / C_Q", "19/20 ; 14/20",
           "KS-3 / §19", "MAIN", p, "replicate ごとに C_Lap・C_Q の argmin を再計算",
           f"{cl}/{len(l)} ; {cq}/{len(q)}")
    p = "expfam/results/k_selection/dsweep_k_selection_20261105/full/selection.csv"
    r = rows("MAIN", p)
    out = []
    for d in ("15", "30", "60"):
        out.append(str(sum(x["label"] == "exact" for x in r if x["d"] == d and x["criterion"] == "C_Q")))
    record("F-06", "F", "d-sweep（n=75, K_GEN=5）", "C_Q の正解数 d=15/30/60（各 8）", "0/3/5",
           "KS-4", "MAIN", p, "criterion=C_Q, label=exact", "/".join(out))


@guarded
def f_pf3():
    base = "expfam/results/k_selection/paper_fig3_condition_criteria_20261109/selection_per_trial.csv"
    r = rows("PF3", base)
    per = [str(sum(x["outcome"] == "exact" for x in r if x["criterion"] == "CQ" and x["K_GEN"] == str(k)))
           for k in (1, 3, 5, 7, 9)]
    tot = sum(map(int, per))
    record("F-07", "F", "原論文 Fig.3 近似条件（#165）", "C_Q の試行ごとの正解数 K_GEN=1/3/5/7/9（合計/50）",
           "10/7/6/5/1 (29/50)", "06_summary_ja.md", "PF3", base, "criterion=CQ, outcome=exact",
           f"{'/'.join(per)} ({tot}/50)")
    under = sum(x["outcome"] == "under" for x in r if x["criterion"] == "CQ")
    record("F-08", "F", "原論文 Fig.3 近似条件（#165）", "C_Q の過小選択数", "0",
           "06_summary_ja.md", "PF3", base, "criterion=CQ, outcome=under", str(under))
    lap_over = sum(x["outcome"] == "over" for x in r if x["criterion"] == "CLap")
    record("F-08b", "F", "原論文 Fig.3 近似条件（#165）", "C_Lap の過大選択数（評価可能 44 試行中）", "20",
           "06_summary_ja.md（誤りはすべて K_GEN+1 への過大選択）", "PF3", base, "criterion=CLap, outcome=over",
           str(lap_over))
    for c, rep in [("CLap", "24/44"), ("CELBO", "26/44")]:
        rr = [x for x in r if x["criterion"] == c and x["outcome"] in ("exact", "under", "over")]
        record(f"F-09-{c}", "F", "原論文 Fig.3 近似条件（#165）", f"{c} の正解数 / 評価可能な試行", rep,
               "06_summary_ja.md", "PF3", base, f"criterion={c}, outcome∈{{exact,under,over}}",
               f"{sum(x['outcome'] == 'exact' for x in rr)}/{len(rr)}")
        allr = [x for x in r if x["criterion"] == c]
        inc = {}
        for x in allr:
            if x["outcome"] not in ("exact", "under", "over"):
                inc[x["K_GEN"]] = inc.get(x["K_GEN"], 0) + 1
        record(f"F-09b-{c}", "F", "原論文 Fig.3 近似条件（#165）", f"{c} の正解数 / 全試行（評価不能の内訳 K_GEN:数）",
               "UNSTATED", "（報告書は評価可能な試行を分母にしている。参考として全試行の分母で記録）", "PF3", base,
               "全 50 試行を分母にする", None)
        ROWS[-1]["recomputed_value"] = (f"{sum(x['outcome'] == 'exact' for x in allr)}/{len(allr)} (評価不能 "
                                        + ",".join(f"{k}:{v}" for k, v in sorted(inc.items(), key=lambda t: int(t[0]))) + ")")
        ROWS[-1]["status"] = "RECOMPUTED_ONLY"


@guarded
def f_criterion_impact():
    p = "expfam/results/k_selection/criterion_impact_validation_20261108/selection_by_dataset.csv"
    r = rows("PF3", p)
    for ph, c, rep in [("1", "CQ", "17/15/0"), ("2", "CQ", "23/9/0"), ("1", "CLap", "21/6/0"),
                       ("2", "CLap", "27/3/0"), ("2", "CELBO", "25/5/0")]:
        rr = [x for x in r if x["phase"] == ph and x["criterion"] == c]
        e, u, o = (sum(x["outcome"] == t for x in rr) for t in ("exact", "under", "over"))
        record(f"F-10-p{ph}-{c}", "F", "基準の統制比較（#164）", f"Phase {ph} {c} 正解/過小/過大",
               rep, "10_advisor_summary_ja.md", "PF3", p, "selection_by_dataset を数える", f"{e}/{u}/{o}")


def _sel_from_fits(fits, key_ds, key_k, key_score):
    agg = defaultdict(lambda: defaultdict(list))
    for x in fits:
        agg[x[key_ds]][int(x[key_k])].append(num(x[key_score]))
    return {ds: argmin_k({k: mean(v) for k, v in kk.items()}) for ds, kk in agg.items()}


@guarded
def f_kexp2_main():
    p = f"{KX}/main/per_fit.csv"
    r = [x for x in rows("KEXP2", p) if x["status"] == "OK"]
    record("F-11", "F", "K-exp2 本実験（8,000 fit）", "status=OK の fit 数", "8000/8000",
           "results_report.md", "KEXP2", p, "status を数える", f"{len(r)}/8000")
    sel = _sel_from_fits(r, "dataset_id", "k", "C_Q")
    meta = {x["dataset_id"]: (x["series"], x["regime"], int(x["k_true"])) for x in r}
    cell = defaultdict(lambda: [0, 0, 0])
    for ds, k in sel.items():
        s, g, kt = meta[ds]
        cell[(s, g, kt)][0 if k == kt else (1 if k < kt else 2)] += 1
    a = sum(v[0] for (s, g, kt), v in cell.items() if g == "a")
    record("F-12", "F", "K-exp2 本実験", "regime a の正解データ数 / 200", "200/200", "claim_ledger.md 命題 A",
           "KEXP2", p, "2 初期値の C_Q 平均の argmin を per_fit から再計算", f"{a}/200")
    for key, rep in [(("A", "b", 7), "1/9/0"), (("A", "b", 9), "0/10/0"), (("B", "b", 9), "5/5/0")]:
        v = cell[key]
        record(f"F-13-{''.join(map(str, key))}", "F", "K-exp2 本実験", f"{key[0]}-{key[1]}-{key[2]} 正解/過小/過大",
               rep, "claim_ledger.md 命題 B", "KEXP2", p, "同上", f"{v[0]}/{v[1]}/{v[2]}")
    b_other = sum(v[0] for (s, g, kt), v in cell.items() if g == "b" and (s, g, kt) not in
                  {("A", "b", 7), ("A", "b", 9), ("B", "b", 9)})
    record("F-14", "F", "K-exp2 本実験", "regime b の残り 17 セルの正解数 / 170", "170/170",
           "claim_ledger.md 命題 B", "KEXP2", p, "同上", f"{b_other}/170")
    record("F-15", "F", "K-exp2 本実験", "過大選択の総数（400 データ）", "0",
           "claim_ledger.md T3", "KEXP2", p, "同上", str(sum(v[2] for v in cell.values())))


@guarded
def f_kexp2_conv():
    p = f"{KX}/convergence_sensitivity/run/checkpoint_scores.csv"
    r = rows("KEXP2", p)
    for cell, rep in [(("B", "9"), "5/8/7"), (("A", "9"), "0/0/0"), (("O", "9"), "10/10/10")]:
        out = []
        for ck in ("10", "20", "40"):
            rr = [x for x in r if x["series"] == cell[0] and x["k_true"] == cell[1] and x["checkpoint"] == ck]
            sel = _sel_from_fits(rr, "dataset_id", "k", "C_Q")
            out.append(str(sum(k == int(cell[1]) for k in sel.values())))
        record(f"F-16-{cell[0]}", "F", "K-exp2 収束感度（600 fit）", f"{cell[0]}-b-9 の C_Q 正解数 10/20/40 反復",
               rep, "convergence_sensitivity_results.md", "KEXP2", p,
               "checkpoint ごとに 2 初期値平均の argmin", "/".join(out))


def _lap_selection(rows_, key_score="C_Lap", status_key="lap_status"):
    """Per dataset: INCOMPLETE if any of the 20 evaluations is not OK; else argmin of the 2-init mean."""
    by = defaultdict(list)
    for x in rows_:
        by[x["dataset_id"]].append(x)
    out = {}
    for ds, xs in by.items():
        if any(x[status_key] != "OK" for x in xs):
            out[ds] = None
        else:
            out[ds] = _sel_from_fits(xs, "dataset_id", "k", key_score)[ds]
    return out


@guarded
def f_kexp2_clap():
    p = f"{KX}/clap_comparison/run/evaluations.csv"
    r = rows("KEXP2", p)
    record("F-17", "F", "K-exp2 C_Q と C_Lap の比較（6,600 評価）", "評価行数", "6600",
           "clap_comparison_results.md", "KEXP2", p, "行数", str(len(r)))
    main10 = [x for x in r if x["source"] == "main10"]
    sel = _lap_selection(main10)
    record("F-18", "F", "K-exp2 C_Q と C_Lap の比較", "main10 の C_Lap INCOMPLETE データ数 / 300", "47/300",
           "followup_abc_results_20261126.md §B", "KEXP2", p, "20 評価のうち 1 つでも非 OK なら INCOMPLETE",
           f"{sum(v is None for v in sel.values())}/{len(sel)}")
    kt_all = {x["dataset_id"]: int(x["k_true"]) for x in main10}
    qsel = {}
    by = defaultdict(list)
    for x in main10:
        by[x["dataset_id"]].append(x)
    for ds, xs in by.items():
        qsel[ds] = _sel_from_fits(xs, "dataset_id", "k", "C_Q")[ds]
    def tally(d):
        e = sum(1 for k, v in d.items() if v == kt_all[k]); u = sum(1 for k, v in d.items() if v is not None and v < kt_all[k])
        o = sum(1 for k, v in d.items() if v is not None and v > kt_all[k]); i = sum(v is None for v in d.values())
        return f"{e}/{u}/{o}/{i}"
    record("F-18c", "F", "K-exp2 C_Q と C_Lap の比較", "main10 全 300 データ（series O/A/B、C は対象外）の正解/過小/過大/INCOMPLETE：C_Q ; C_Lap",
           "UNSTATED", "（全体集計は報告書に単独の行がない。参考）", "KEXP2", p, "dataset ごとに再計算", None)
    ROWS[-1]["recomputed_value"] = f"{tally(qsel)} ; {tally(sel)}"
    ROWS[-1]["status"] = "RECOMPUTED_ONLY"
    # K_TRUE=3 datasets are shared by regime a and b (same data); count them once
    rb3 = {x["dataset_id"] for x in main10 if x["regime"] == "b" and x["k_true"] == "3"}
    same = all(qsel[d] == qsel[d.replace("-b-", "-a-")] and sel[d] == sel[d.replace("-b-", "-a-")] for d in rb3)
    q2 = {k: v for k, v in qsel.items() if k not in rb3}
    l2 = {k: v for k, v in sel.items() if k not in rb3}
    record("F-18d", "F", "K-exp2 C_Q と C_Lap の比較", "main10 の重複除去後（regime b の K_TRUE=3 を除く 270 データ）の正解/過小/過大/INCOMPLETE：C_Q ; C_Lap（a/b の K_TRUE=3 の選択が一致するか）",
           "INCOMPLETE 39/270", "clap_comparison_results.md L.61（47/300、K_TRUE=3 の重複を除くと 39/270）", "KEXP2", p,
           "regime b の K_TRUE=3 データを除いて数える", None)
    ROWS[-1]["recomputed_value"] = f"{tally(q2)} ; {tally(l2)} (n={len(q2)}; a/b 一致={same})"
    ROWS[-1]["status"] = "VERIFIED" if sum(v is None for v in l2.values()) == 39 and len(l2) == 270 else "MISMATCH"
    print("series in main10:", sorted({x["series"] for x in main10}))
    inc_small = sum(1 for ds, v in sel.items() if v is None and kt_all[ds] in (1, 3, 5))
    record("F-18b", "F", "K-exp2 C_Q と C_Lap の比較", "main10 の INCOMPLETE 47 データのうち K_TRUE∈{1,3,5} の数", "39",
           "followup_abc_results_20261126.md §B（選択バイアス）", "KEXP2", p, "INCOMPLETE データの K_TRUE を数える", str(inc_small))
    for src in ("main10", "conv40"):
        rr = [x for x in r if x["source"] == src and x["series"] == "A" and x["k_true"] == "9" and x["regime"] == "b"]
        s = _lap_selection(rr)
        record(f"F-19-{src}", "F", "K-exp2 C_Q と C_Lap の比較", f"A-b-9 の C_Lap 正解数（{src}）", "6/10",
               "claim_ledger.md §6", "KEXP2", p, "同上", f"{sum(v == 9 for v in s.values())}/{len(s)}")
    rr = [x for x in r if x["source"] == "conv40" and x["series"] == "B" and x["k_true"] == "9" and x["regime"] == "b"]
    s40 = _lap_selection(rr)
    record("F-19b", "F", "K-exp2 C_Q と C_Lap の比較", "B-b-9 の C_Lap 正解数（conv40、INCOMPLETE）", "8/10 (INCOMPLETE 2)",
           "clap_comparison/analysis/summary_by_cell.csv（conv40, B-b-9）", "KEXP2", p, "INCOMPLETE 規則つきの argmin",
           f"{sum(v == 9 for v in s40.values())}/{len(s40)} (INCOMPLETE {sum(v is None for v in s40.values())})")
    rr = [x for x in r if x["source"] == "main10" and x["series"] == "A" and x["k_true"] == "7" and x["regime"] == "b"]
    s = _lap_selection(rr)
    record("F-20", "F", "K-exp2 C_Q と C_Lap の比較", "A-b-7 の C_Lap 正解数（main10）", "10/10",
           "claim_ledger.md §6", "KEXP2", p, "同上", f"{sum(v == 7 for v in s.values())}/{len(s)}")
    over = 0
    kt = {x["dataset_id"]: int(x["k_true"]) for x in main10}
    over = sum(1 for ds, v in sel.items() if v is not None and v > kt[ds])
    record("F-21", "F", "K-exp2 C_Q と C_Lap の比較", "main10 の C_Lap 過大選択データ数（完了分のみ）", "UNSTATED",
           "（報告書に単独の数値なし。参考として記録）", "KEXP2", p, "完了データで k_Lap > K_TRUE", None)
    ROWS[-1]["recomputed_value"] = str(over)
    ROWS[-1]["status"] = "RECOMPUTED_ONLY"


@guarded
def f_kexp2_mixedx():
    p = f"{KX}/mixedx_highk_20261126/run/per_fit.csv"
    r = [x for x in rows("KEXP2", p) if x["phase"] == "main"]
    record("F-22", "F", "K-exp2 追加 C：混合分布 X 高次元（600 fit）", "status=OK の fit 数", "600/600",
           "followup_abc_results_20261126.md §C", "KEXP2", p, "status を数える",
           f"{sum(x['status'] == 'OK' for x in r)}/{len(r)}")
    q = _sel_from_fits(r, "dataset_id", "k", "C_Q")
    l = _lap_selection(r)
    kt = {x["dataset_id"]: int(x["k_true"]) for x in r}
    for k, repq, repl in [(5, "10/10", "6/10 (INCOMPLETE 4)"), (7, "6/10", "10/10 (INCOMPLETE 0)"),
                          (9, "0/10", "9/10 (INCOMPLETE 0)")]:
        dq = [ds for ds in q if kt[ds] == k]
        record(f"F-23-{k}", "F", "K-exp2 追加 C", f"K_TRUE={k} の C_Q 正解数", repq, "§C", "KEXP2", p,
               "2 初期値平均の argmin", f"{sum(q[ds] == k for ds in dq)}/{len(dq)}")
        record(f"F-24-{k}", "F", "K-exp2 追加 C", f"K_TRUE={k} の C_Lap 正解数（INCOMPLETE）", repl, "§C", "KEXP2", p,
               "INCOMPLETE 規則つきの argmin",
               f"{sum(l[ds] == k for ds in dq)}/{len(dq)} (INCOMPLETE {sum(l[ds] is None for ds in dq)})")
    record("F-25", "F", "K-exp2 追加 C", "過大選択の数 C_Q / C_Lap", "0 / 0", "§C", "KEXP2", p, "同上",
           f"{sum(q[ds] > kt[ds] for ds in q)} / {sum(l[ds] is not None and l[ds] > kt[ds] for ds in l)}")


@guarded
def f_gaussian_y_and_hpb():
    p = "expfam/results/canonical_gaussian_y_k_selection_20260907/selection_by_trial.csv"
    r = rows("PCMIX", p)
    record("F-26", "F", "canonical Gaussian-Y K 回復（2026-09-07）", "全 config の正解 trial 数", "70/70",
           "aggregate_recovery.csv（全行 recovery_rate 1.0）", "PCMIX", p, "correct を数える",
           f"{sum(x['correct'] == 'True' for x in r)}/{len(r)}")
    p = "expfam/results/k_selection/historical_pb_k5_n_sweep_20260906/summary.json"
    s = json.loads(show("HPB", p))["by_n"]
    out = []
    for n in ("50", "75", "100", "150"):
        if n in s:
            out.append(f"n={n}:{sum(k == 5 for k in s[n]['selected_k_per_trial'])}/{len(s[n]['selected_k_per_trial'])}")
    record("F-27", "F", "historical P/B K_TRUE=5 n-sweep", "trial ごとの正解数（n 別）",
           "n=50:1/5", "summary.json（by_n.50.exact=1）", "HPB", p, "selected_k_per_trial から数える",
           out[0] if out else None)
    ROWS[-1]["quantity"] += "（全 n：" + "、".join(out) + "）"


# ---------------------------------------------------------------- C: real data
@guarded
def c_real():
    p = "expfam/results/real_data/wine_clean/wine_bic_summary_clean.csv"
    r = rows("MAIN", p)
    best = min(r, key=lambda x: num(x["BIC_mean"]))
    record("C-01", "C", "Wine fixed pilot（clean）", "C_Q（列名 BIC）平均が最小の k と値", "k=3:6324.77",
           "RD-1", "MAIN", p, "argmin", f"k={best['k']}:{num(best['BIC_mean']):.2f}")
    p = "expfam/results/real_data/cora_clean/cora_heldout_summary_clean.csv"
    r = {x["k"]: x for x in rows("MAIN", p)}
    record("C-02", "C", "旧 Cora held-out（transductive）", "test_AP k=3 / k=6 / random", "0.4319/0.4631/0.1667",
           "RD-2", "MAIN", p, "clean CSV を読む（集計済み）",
           f"{r['3']['test_AP_mean']}/{r['6']['test_AP_mean']}/{r['3']['random_AP_baseline_mean']}")
    p = "expfam/results/real_data/movielens_userdisjoint/movielens_userdisjoint_20260822_paired.csv"
    r = rows("MAIN", p)
    d = [num(x["value"]) for x in r if x["contrast"] == "delta_primary"]
    record("C-03", "C", "MovieLens user-disjoint（#33）", "delta_primary 平均（正の split 数/30）",
           "+0.004248 (23/30)", "RD-5", "MAIN", p, "split ごとの値を平均", f"{mean(d):+.6f} ({sum(v > 0 for v in d)}/{len(d)})")
    names = sorted({x["contrast"] for x in r})
    rp = next((n for n in names if "rawpoisson" in n), None)
    if rp:
        d = [num(x["value"]) for x in r if x["contrast"] == rp]
        record("C-04", "C", "MovieLens user-disjoint（#33）", f"{rp} 平均（正の split 数/30）",
               "-0.100274 (1/30)", "RD-6", "MAIN", p, "split ごとの値を平均", f"{mean(d):+.6f} ({sum(v > 0 for v in d)}/{len(d)})")
    p = "expfam/results/overdispersion/movielens_overdispersion_diagnostics.csv"
    x = rows("MAIN", p)[0]
    record("C-05", "C", "MovieLens 過分散診断", "co_rating_count の周辺 var/mean", 9.89,
           "registry 過分散節", "MAIN", p, "var/mean を再計算", round(num(x["var"]) / num(x["mean"]), 2), tol=0.006)
    p = "expfam/results/real_data/cora_strict_heldout_family_main_20261124/heldout_scores.csv"
    r = [x for x in rows("THESIS", p) if x["estimator"] == "mc_predictive"]
    sc = {(x["arm"], x["K"], x["split_index"], x["init_seed"]): num(x["log_score"]) for x in r}
    for K, a, b, rep in [("6", "BX", "GX", "+1.059e-04 (13/15)"), ("3", "BX", "GX", "+1.136e-04 (13/15)"),
                         ("6", "BX", "GXc", "-4.4e-05 (3/15)")]:
        keys = [(s, i) for (arm, k, s, i) in sc if arm == a and k == K]
        diffs = [sc[(a, K, s, i)] - sc[(b, K, s, i)] for s, i in keys if (b, K, s, i) in sc]
        fmt = f"{mean(diffs):+.3e} ({sum(v > 0 for v in diffs)}/{len(diffs)})"
        if rep.startswith("-4.4e-05"):
            fmt = f"{mean(diffs):+.1e} ({sum(v > 0 for v in diffs)}/{len(diffs)})".replace("e-0", "e-0")
        record(f"C-06-K{K}-{b}", "C", "CORA-SH1 strict held-out（2026-11-24）",
               f"K={K} {a}−{b} の mc_predictive log score 差（正の対/15）", rep, "CO-1・CO-2", "THESIS", p,
               "split×seed の対で差を取り平均", fmt.replace("+-", "-"))


def main() -> int:
    for fn in (a_reproduction, b_fixed, b_old05, d_percolumn, d_m1_e1, d_mx2, d_pcmix_k, e_af,
               f_early, f_phase9, f_pf3, f_criterion_impact, f_kexp2_main, f_kexp2_conv, f_kexp2_clap,
               f_kexp2_mixedx, f_gaussian_y_and_hpb, c_real):
        fn()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["result_id", "domain", "experiment_id", "quantity", "reported_value", "reported_in",
            "recomputed_value", "status", "source_commit", "source_path", "method"]
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(ROWS)
    counts = defaultdict(int)
    for x in ROWS:
        counts[x["status"]] += 1
    print(dict(counts))
    for x in ROWS:
        if x["status"] != "VERIFIED":
            print(x["result_id"], x["status"], x["quantity"], "| reported", x["reported_value"], "| recomputed", x["recomputed_value"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
