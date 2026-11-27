"""Addendum (2026-11-27): read-only re-aggregation of the two-gaps Experiment A/B numbers.

Separate from verify_primary_numbers.py: the 96-item snapshot (data/primary_results_index.csv) is NOT
touched. Every input is read with ``git show <commit>:<path>`` from fixed commits (REFS). No model is
imported, nothing is fitted, no existing file is written.

Output: docs/research_results/data/addendum_20261127_primary_results.csv
  VERIFIED        recomputed value equals the value reported in the cited document (stated rounding)
  MISMATCH        differs (difference recorded)
  RECOMPUTED_ONLY no single reported value exists; recorded for reference

Usage (repository root; the commits must be fetched):
    python docs/research_results/scripts/verify_addendum_20261127.py
"""
from __future__ import annotations

import csv
import io
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

REFS = {
    "TG": "1ef9096",     # experiment/k-exp2-two-gaps-20261127: Exp A/B main results (unmerged)
    "DIAG": "2f71e61",   # audit/two-gaps-diagnostics-20261127: post-hoc diagnostics (unmerged)
    "CORR": "c035df7",   # audit/two-gaps-diagnostics-correction-20261127: forward correction (unmerged)
}
TGP = "expfam/results/k_selection/two_gaps_20261127"
DG = TGP + "/diagnostics_20261127"
FC = DG + "/forward_correction_20261127"
OUT = Path(__file__).resolve().parents[1] / "data" / "addendum_20261127_primary_results.csv"
ROWS: list[dict] = []
N, D = 150, 15


def rows(ref, path):
    out = subprocess.run(["git", "show", f"{REFS[ref]}:{path}"], check=True, capture_output=True).stdout
    return list(csv.DictReader(io.StringIO(out.decode("utf-8"))))


def record(rid, exp, quantity, reported, reported_in, ref, path, method, recomputed, tol=None):
    if reported in ("", None):
        status = "RECOMPUTED_ONLY"
    elif tol is None:
        status = "VERIFIED" if str(reported) == str(recomputed) else "MISMATCH"
    else:
        rep = [float(x) for x in str(reported).split("/")]
        rec = [float(x) for x in str(recomputed).split("/")]
        status = "VERIFIED" if len(rep) == len(rec) and all(abs(a - b) <= tol for a, b in zip(rep, rec)) else "MISMATCH"
    ROWS.append({"result_id": rid, "domain": "TG", "experiment_id": exp, "quantity": quantity,
                 "reported_value": reported, "reported_in": reported_in, "recomputed_value": recomputed,
                 "status": status, "source_commit": REFS[ref], "source_path": path, "method": method})


def f(x):
    return float(x)


def two_init_mean(rs, keys, col):
    g = defaultdict(list)
    for r in rs:
        g[tuple(r[k] for k in keys)].append(f(r[col]))
    assert all(len(v) == 2 for v in g.values()), "expected 2 inits per cell"
    return {k: mean(v) for k, v in g.items()}


def exp_a():
    p = TGP + "/expA/run/per_fit.csv"
    pf = [r for r in rows("TG", p) if r["phase"] == "main"]
    ok = sum(r["status"] == "OK" for r in pf)
    record("TG-A-01", "TG-A", "本実験の fit 数（OK / 全体）", "160/160", "results_report.md A-1", "TG", p,
           "status 列を数えた", f"{ok}/{len(pf)}")
    rm = two_init_mean(pf, ("dataset_id", "rate", "family_x_fit"), "rmse_Z_last")
    ds = sorted({k[0] for k in rm})
    dd = [rm[(d, "dense", "poisson")] - rm[(d, "dense", "bernoulli")] for d in ds]
    dsp = [rm[(d, "sparse", "poisson")] - rm[(d, "sparse", "bernoulli")] for d in ds]
    ii = [b - a for a, b in zip(dd, dsp)]
    m = "データごとに 2 初期値の RMSE_Z（Z_last）を平均し、誤指定 − 正指定"
    record("TG-A-02", "TG-A", "Δ_dense の平均（RMSE_Z、Poisson − Bernoulli）・正の数", "0.0116/17",
           "results_report.md A-2", "TG", p, m, f"{mean(dd):.4f}/{sum(x > 0 for x in dd)}", 5e-5)
    record("TG-A-03", "TG-A", "Δ_sparse の平均・正の数", "0.2229/20", "results_report.md A-2", "TG", p, m,
           f"{mean(dsp):.4f}/{sum(x > 0 for x in dsp)}", 5e-5)
    record("TG-A-04", "TG-A", "交互作用 I = Δ_sparse − Δ_dense の平均・正の数（primary）", "0.2113/20",
           "results_report.md A-2", "TG", p, m, f"{mean(ii):.4f}/{sum(x > 0 for x in ii)}", 5e-5)
    lv = "/".join(f"{mean(rm[(d, r, fam)] for d in ds):.3f}" for r in ("dense", "sparse") for fam in ("bernoulli", "poisson"))
    record("TG-A-05", "TG-A", "RMSE_Z の水準 dense B/P・sparse B/P", "0.383/0.395/1.024/1.247",
           "results_report.md A-2", "TG", p, "2 初期値平均の 20 データ平均", lv, 5e-4)
    gm = defaultdict(list)
    for r in pf:
        gm[(r["rate"], r["family_x_fit"])].append(f(r["gram_corr_last"]))
    record("TG-A-06", "TG-A", "gram 相関（真の Z と、fit の平均）sparse B/P", "0.241/0.061",
           "results_report.md A-3", "TG", p,
           "fit の平均。報告値の正指定 0.241 は 0.24046 を丸めた値で、四捨五入では 0.240（丸めの差 0.0005。主張は変わらない）",
           f"{mean(gm[('sparse', 'bernoulli')]):.3f}/{mean(gm[('sparse', 'poisson')]):.3f}", 5e-4)
    hs = two_init_mean(pf, ("dataset_id", "rate", "family_x_fit"), "heldout_log_score")
    for rid, rate, rep in (("TG-A-07", "sparse", "-0.0334/1"), ("TG-A-08", "dense", "0.0025/15")):
        dl = [hs[(d, rate, "bernoulli")] - hs[(d, rate, "poisson")] for d in ds]
        record(rid, "TG-A", f"{rate} の held-out 対数スコアの差（正 − 誤）の平均・正指定が良いデータ数", rep,
               "results_report.md A-3", "TG", p, "Z_last による plug-in 予測、2 初期値平均",
               f"{mean(dl):.4f}/{sum(x > 0 for x in dl)}", 5e-5)

    q = DG + "/prediction_diagnostics.csv"
    pr = rows("DIAG", q)
    pfi = {r["fit_id"]: r for r in pf}
    worst = max(abs(f(r["plugin_last_log_score"]) - f(pfi[r["fit_id"]]["heldout_log_score"])) for r in pr)
    record("TG-A-08b", "TG-A", "原因分析の plug-in 対数スコア = 本実験の heldout_log_score（160 fit の最大差）", "0",
           "analysis_protocol.md（一致を確認すると記載）", "DIAG", q, "fit_id で突き合わせ", f"{worst:.0e}" if worst else "0")
    sp = [r for r in pr if r["rate"] == "sparse"]

    def cm(col):
        return two_init_mean(sp, ("dataset_id", "family_x_fit"), col)
    for rid, col, rep, rin, better in (
            ("TG-A-09", "plugin_last_auc", "0.579/0.517/20", "diagnostic_report.md 1-1", 1),
            ("TG-A-10", "plugin_last_ap", "0.266/0.223/20", "diagnostic_report.md 1-1", 1),
            ("TG-A-11", "plugin_last_log_score", "-0.573/-0.540/1", "diagnostic_report.md 1-1", 1),
            ("TG-A-12", "plugin_last_ece", "0.100/0.070/0", "diagnostic_report.md 1-1（正指定が小さいデータ数）", -1)):
        c = cm(col)
        b = [c[(d, "bernoulli")] for d in ds]
        pp = [c[(d, "poisson")] for d in ds]
        nb = sum((x - y) * better > 0 for x, y in zip(b, pp))
        record(rid, "TG-A", f"sparse {col}：正指定・誤指定の平均・正指定が良いデータ数", rep, rin, "DIAG", q,
               "Z_last plug-in、2 初期値平均、20 データ", f"{mean(b):.3f}/{mean(pp):.3f}/{nb}", 5e-4)
    c, k = cm("plugin_last_log_score"), cm("const_log_score")
    nb = sum(c[(d, "bernoulli")] < k[(d, "bernoulli")] for d in ds)
    npo = sum(c[(d, "poisson")] < k[(d, "poisson")] for d in ds)
    record("TG-A-13", "TG-A", "sparse：定数予測の対数スコア・定数予測より悪いデータ数（正/誤）", "-0.509/20/20",
           "diagnostic_report.md 1-1", "DIAG", q, "定数予測 = 学習ペアの正例率",
           f"{mean(k[(d, 'bernoulli')] for d in ds):.3f}/{nb}/{npo}", 5e-4)
    s = cm("plugin_last_calib_slope")
    closer = sum(abs(1 - s[(d, "bernoulli")]) < abs(1 - s[(d, "poisson")]) for d in ds)
    record("TG-A-14", "TG-A", "sparse 校正の傾き 正/誤 の平均・正指定が 1 に近いデータ数", "0.218/0.087/18",
           "forward_correction.md §4", "DIAG", q, "y を logit(p) にロジスティック回帰した係数",
           f"{mean(s[(d, 'bernoulli')] for d in ds):.3f}/{mean(s[(d, 'poisson')] for d in ds):.3f}/{closer}", 5e-4)
    sd = cm("plugin_last_p_sd")
    record("TG-A-15", "TG-A", "sparse 予測確率の SD 正/誤", "0.177/0.104", "diagnostic_report.md 1-1", "DIAG", q,
           "test ペアの予測確率の SD", f"{mean(sd[(d, 'bernoulli')] for d in ds):.3f}/{mean(sd[(d, 'poisson')] for d in ds):.3f}", 5e-4)
    zm = cm("plugin_mean_log_score")
    nb = sum(zm[(d, "bernoulli")] > zm[(d, "poisson")] for d in ds)
    gb = sum(zm[(d, "bernoulli")] > k[(d, "bernoulli")] for d in ds)
    gp = sum(zm[(d, "poisson")] > k[(d, "bernoulli")] for d in ds)
    record("TG-A-16", "TG-A", "Z_mean plug-in：正指定が良いデータ数 / 定数予測に勝つデータ数（正・誤）", "14/15/10",
           "diagnostic_report.md 1-1、forward_correction.md §5", "DIAG", q, "Z_mean（L サンプル平均）での plug-in。事後予測ではない",
           f"{nb}/{gb}/{gp}")


def exp_b():
    p = TGP + "/expB/run/per_fit.csv"
    pf = [r for r in rows("TG", p) if r["phase"] == "main"]
    st = defaultdict(int)
    for r in pf:
        st[r["lap_status"]] += 1
    record("TG-B-01", "TG-B", "本実験の fit 数 OK / C_Lap OK / NOT_STATIONARY", "600/598/2", "results_report.md B-1",
           "TG", p, "status・lap_status を数えた",
           f"{sum(r['status'] == 'OK' for r in pf)}/{st['OK']}/{st['NOT_STATIONARY']}")
    # selection rule (frozen protocol B): mean of 2 inits, argmin, ties within 1e-9 -> smaller K;
    # C_Lap INCOMPLETE if any of the 20 evaluations is not OK; denominator stays 10
    g = defaultdict(list)
    for r in pf:
        g[(r["cond"], r["dataset_id"])].append(r)
    sel = {}
    for (cond, ds), rs in g.items():
        out = {}
        for crit in ("C_Q", "C_Lap"):
            if crit == "C_Lap" and any(r["lap_status"] != "OK" for r in rs):
                out[crit] = "INCOMPLETE"
                continue
            v = {k: mean(f(r[crit]) for r in rs if int(r["k"]) == k) for k in range(1, 11)}
            best = min(v.values())
            out[crit] = min(k for k in v if v[k] - best <= 1e-9)
        sel[(cond, ds)] = out
    rep = {("G01", "C_Q"): "10/0/0/0", ("G01", "C_Lap"): "9/0/0/1", ("GM", "C_Q"): "0/10/0/0",
           ("GM", "C_Lap"): "8/2/0/0", ("P", "C_Q"): "0/10/0/0", ("P", "C_Lap"): "9/0/0/1"}
    i = 2
    for (cond, crit), rv in rep.items():
        ks = [sel[(c, d)][crit] for (c, d) in sel if c == cond]
        cnt = (sum(k == 9 for k in ks), sum(k != "INCOMPLETE" and k < 9 for k in ks),
               sum(k != "INCOMPLETE" and k > 9 for k in ks), sum(k == "INCOMPLETE" for k in ks))
        record(f"TG-B-{i:02d}", "TG-B", f"{cond} {crit}：正解/過小/過大/INCOMPLETE（10 データ）", rv,
               "results_report.md B-2", "TG", p, "凍結 protocol の規則で選び直した（2 初期値平均の argmin、INCOMPLETE は分母に残す）",
               "/".join(map(str, cnt)))
        i += 1
    q = TGP + "/expB/tech_check_v2/information_diagnostics.csv"
    info = [r for r in rows("TG", q) if r["pilot"] == "False"]
    record("TG-B-08", "TG-B", "真の期待 Fisher 情報の trace：G01 / GM の最小 / GM の最大 / P と GM の最大相対差",
           "75/9.70/9.90/", "results_report.md B-2（相対差は < 1e-12）", "TG", q, "非 pilot の 10 replicate",
           f"{max(f(r['G01_trace']) for r in info):.0f}/{min(f(r['GM_trace']) for r in info):.2f}/"
           f"{max(f(r['GM_trace']) for r in info):.2f}/" + f"{max(f(r['trace_match_relerr']) for r in info):.1e}")
    ROWS[-1]["status"] = "VERIFIED" if (ROWS[-1]["recomputed_value"].startswith("75/9.70/9.90/")
                                        and float(ROWS[-1]["recomputed_value"].split("/")[3]) < 1e-12) else "MISMATCH"
    # estimated parameters at K=9 (diagnostics)
    q2 = DG + "/parameter_diagnostics.csv"
    pd = [r for r in rows("DIAG", q2) if r["exp"] == "B" and int(r["k"]) == 9]

    def med_cond(col, cond):
        sub = [r for r in pd if r["cond"] == cond]
        return median(two_init_mean(sub, ("dataset_id",), col).values())
    record("TG-B-09", "TG-B", "K=9 の推定分散（15 列平均）の中央値 G01 / GM", "0.0949/0.711",
           "diagnostic_report.md 2-1", "DIAG", q2, "2 初期値平均の 10 データ中央値",
           f"{med_cond('sigma2_mean', 'G01'):.4f}/{med_cond('sigma2_mean', 'GM'):.3f}", 5e-4)
    record("TG-B-10", "TG-B", "K=9 の推定 X 情報の 1 次元あたり（trace/K）G01 / GM / P", "9.25/1.37/1.17",
           "diagnostic_report.md 2-2", "DIAG", q2, "推定 F・分散（Poisson は推定 Z の行平均）、2 初期値平均の中央値",
           "/".join(f"{med_cond('xinfo_est_trace_per_dim', c):.2f}" for c in ("G01", "GM", "P")), 5e-3)
    # exact X+Y increment from Q_strict
    worst = 0.0
    for r in pf:
        k = int(r["k"])
        pk = k * D - k * (k - 1) // 2 + (D if r["cond"] != "P" else 0)
        worst = max(worst, abs(f(r["C_Q"]) - (-2 * f(r["Q_strict"]) + pk * math.log(N))), abs(int(r["num_params"]) - pk))
    record("TG-B-11", "TG-B", "C_Q = −2Q_strict + p_K ln n（p_K = Kd − K(K−1)/2 (+d)）の最大誤差", "0",
           "forward_correction.md §1", "TG", p, "600 fit で式と num_params を照合（許容 1e-6）", f"{worst:.1e}", 1e-6)
    ell = two_init_mean(pf, ("cond", "dataset_id", "k"), "Q_strict")
    ell = {key: v + N * int(key[2]) * (1 + math.log(2 * math.pi)) / 2 for key, v in ell.items()}
    cq = two_init_mean(pf, ("cond", "dataset_id", "k"), "C_Q")
    cost9 = N * (1 + math.log(2 * math.pi)) + (D - 9 + 1) * math.log(N)
    meds, fav = [], []
    agree = tot = 0
    first_eq = 0
    for cond in ("G01", "GM", "P"):
        dss = sorted({k[1] for k in ell if k[0] == cond})
        v = [2 * (ell[(cond, d, "9")] - ell[(cond, d, "8")]) for d in dss]
        meds.append(f"{median(v):.2f}")
        fav.append(str(sum(x > cost9 for x in v)))
        for d in dss:
            fl = None
            for k in range(2, 11):
                c = N * (1 + math.log(2 * math.pi)) + (D - k + 1) * math.log(N)
                two = 2 * (ell[(cond, d, str(k))] - ell[(cond, d, str(k - 1))])
                agree += (two > c) == (cq[(cond, d, str(k))] < cq[(cond, d, str(k - 1))])
                tot += 1
                if fl is None and not two > c:
                    fl = k - 1
            first_eq += (fl if fl is not None else 10) == sel[(cond, d)]["C_Q"]
    record("TG-B-12", "TG-B", "K=8→9 の X+Y の改善 2Δℓ（10 データ中央値）G01/GM/P と必要量", "901.83/311.63/290.30/460.756",
           "forward_correction.md §2", "TG", p, "ℓ = Q_strict + nK(1+ln2π)/2 を 2 初期値平均、差の中央値",
           "/".join(meds) + f"/{cost9:.3f}", 5e-3)
    record("TG-B-13", "TG-B", "K=8→9 で 2Δℓ が必要量を超えたデータ数 G01/GM/P", "10/0/0", "forward_correction.md §2",
           "TG", p, "同上", "/".join(fav))
    record("TG-B-14", "TG-B", "隣接条件 2Δℓ > 必要量 と C_Q(K) < C_Q(K−1) の一致（全組）", "270/270",
           "forward_correction.md §1", "TG", p, "3 条件 × 10 データ × K=2..10", f"{agree}/{tot}")
    record("TG-B-15", "TG-B", "「隣接条件が最初に破れる手前の K」= 大域的 argmin のデータ数", "29/30",
           "forward_correction.md §2", "TG", p, "保存 C_Q の argmin と比較", f"{first_eq}/30")
    worst = 0.0
    for r in pf:
        if r["lap_status"] == "OK":
            k = int(r["k"])
            lo = f(r["lap_phi_at_mode"]) + N * k / 2 * math.log(2 * math.pi) - 0.5 * f(r["lap_logdet_H"])
            cl = -2 * lo + (k * D - k * (k - 1) // 2) * math.log(N)
            worst = max(worst, abs(cl - f(r["C_Lap"])))
    record("TG-B-16", "TG-B", "C_Lap = −2[Φ(Ẑ) + (nK/2)ln2π − ½ln|H|] + d_K ln n の最大誤差（598 OK 評価）", "0",
           "forward_correction.md §6", "TG", p, "保存された Φ・ln|H| から計算（d_K は Gaussian の分散を含まない）",
           f"{worst:.1e}", 1e-6)
    gaps = []
    for cond in ("GM", "P"):
        for (c, d), s in sel.items():
            if c == cond:
                gaps.append(cq[(c, d, "9")] - cq[(c, d, str(s["C_Q"]))])
    record("TG-B-17", "TG-B", "GM・P で C_Q(9) − C_Q(選ばれた K) の最小・最大", "190/595", "diagnostic_report.md 2-4",
           "TG", p, "2 初期値平均", f"{min(gaps):.0f}/{max(gaps):.0f}", 1.0)


def correction_checks():
    p = FC + "/independent_check_correction.csv"
    rs = rows("CORR", p)
    record("TG-C-01", "TG-CORR", "forward correction の別プロセス確認（何を確認したかは 13 章 §9）", "17/17",
           "forward_correction.md 冒頭", "CORR", p, "pass 列を数えた", f"{sum(r['pass'] == 'True' for r in rs)}/{len(rs)}")
    p = DG + "/independent_check.csv"
    rs = rows("DIAG", p)
    record("TG-C-02", "TG-DIAG", "原因分析の別プロセス確認", "9/9", "diagnostic_report.md 冒頭", "DIAG", p,
           "pass 列を数えた", f"{sum(r['pass'] == 'True' for r in rs)}/{len(rs)}")
    p = DG + "/consistency_checks.csv"
    rs = rows("DIAG", p)
    record("TG-C-03", "TG-DIAG", "原因分析スクリプト内の整合性検査", "11/11", "diagnostic_report.md 冒頭", "DIAG", p,
           "pass 列を数えた", f"{sum(r['pass'] == 'True' for r in rs)}/{len(rs)}")
    p = TGP + "/independent_tally/independent_tally.csv"
    rs = rows("TG", p)
    col = "pass" if "pass" in rs[0] else ("ok" if "ok" in rs[0] else list(rs[0])[-1])
    record("TG-C-04", "TG-A/TG-B", "本実験の独立集計", "25/25", "results_report.md 冒頭", "TG", p,
           f"{col} 列を数えた", f"{sum(r[col] in ('True', 'PASS', 'OK', '1') for r in rs)}/{len(rs)}")


def main():
    exp_a()
    exp_b()
    correction_checks()
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ROWS[0]))
        w.writeheader()
        w.writerows(ROWS)
    c = defaultdict(int)
    for r in ROWS:
        c[r["status"]] += 1
    print(dict(c), len(ROWS))
    for r in ROWS:
        if r["status"] != "VERIFIED":
            print(r["result_id"], r["status"], r["reported_value"], "|", r["recomputed_value"])


if __name__ == "__main__":
    main()
