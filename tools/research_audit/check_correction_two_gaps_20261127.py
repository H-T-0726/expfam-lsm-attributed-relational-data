"""Separate-process check of forward_correction_20261127 (signs, constants, parameter counts, units).

Imports none of: correct_two_gaps_diagnostics_20261127, diagnose_two_gaps_20261127, run_two_gaps_20261127.
The exact X+Y term is derived here from Q_strict (not from C_Q) with an independently written
parameter count, so the C_Q identity, the penalty count and the sign of the K-increase condition are
each tested rather than assumed.
Usage: python tools/research_audit/check_correction_two_gaps_20261127.py
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "expfam/results/k_selection/two_gaps_20261127"
DIAG = BASE / "diagnostics_20261127"
OUT = DIAG / "forward_correction_20261127"
n, d = 150, 15
res = []


def rd(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def chk(name, ok, detail=""):
    res.append({"check": name, "pass": bool(ok), "detail": str(detail)[:400]})


def n_params(k, gaussian_x):
    # loadings with rotation constraint + diagonal Gaussian-X variances; w0, w not counted (calc_bic_exp)
    return k * d - k * (k - 1) // 2 + (d if gaussian_x else 0)


def main():
    pf = [r for r in rd(BASE / "expB/run/per_fit.csv") if r["status"] == "OK"]
    chk("B: 600 OK fits, 10 datasets x 10 K x 2 inits per condition", len(pf) == 600 and all(
        sum(1 for r in pf if r["cond"] == c) == 200 for c in ("G01", "GM", "P")), len(pf))

    # constants
    c1 = n * (1 + math.log(2 * math.pi))
    chk("constant n(1+ln 2pi) = 425.68", abs(c1 - 425.6816) < 1e-4, c1)
    chk("cost(8->9) = n(1+ln2pi) + (d-9+1) ln n = 460.756", round(c1 + 7 * math.log(n), 3) == 460.756,
        c1 + 7 * math.log(n))

    # parameter count and C_Q identity
    worst_p = max(abs(int(r["num_params"]) - n_params(int(r["k"]), r["cond"] != "P")) for r in pf)
    chk("num_params == Kd - K(K-1)/2 (+d for Gaussian-X)", worst_p == 0, worst_p)
    worst = max(abs(float(r["C_Q"]) - (-2 * float(r["Q_strict"]) + n_params(int(r["k"]), r["cond"] != "P") * math.log(n)))
                for r in pf)
    chk("C_Q == -2 Q_strict + p ln n", worst < 1e-6, worst)
    vz = [float(np.load(BASE / "expB/run/states" / r["state_file"])["var_z"]) for r in pf]
    chk("var_z == 1 in all 600 B states (premise of the exact lnpZ constant)", all(v == 1.0 for v in vz), set(vz))

    # exact l(K) from Q_strict: l = Q - lnpZ, lnpZ = -(nK/2)(1 + ln 2pi)
    g = defaultdict(list)
    for r in pf:
        k = int(r["k"])
        g[(r["cond"], r["dataset_id"], k)].append(
            (float(r["Q_strict"]) + n * k * (1 + math.log(2 * math.pi)) / 2, float(r["C_Q"])))
    cell = {key: (np.mean([a for a, _ in v]), np.mean([b for _, b in v]), len(v)) for key, v in g.items()}
    chk("aggregation unit: each (cond, dataset, K) cell has exactly 2 inits", all(c[2] == 2 for c in cell.values()))
    agree_new = agree_old = total = 0
    med = {}
    for cond in ("G01", "GM", "P"):
        dss = sorted({k[1] for k in cell if k[0] == cond})
        assert len(dss) == 10
        for k in range(2, 11):
            cost = c1 + (d - k + 1) * math.log(n)
            vals = []
            for ds in dss:
                two_dl = 2 * (cell[(cond, ds, k)][0] - cell[(cond, ds, k - 1)][0])
                dec = cell[(cond, ds, k)][1] < cell[(cond, ds, k - 1)][1]
                agree_new += (two_dl > cost) == dec
                agree_old += (-two_dl > cost) == dec        # inequality as printed in diagnostic_report.md l.165
                total += 1
                vals.append(two_dl)
            med[(cond, k)] = float(np.median(vals))
    chk("corrected condition 2*dl > n(1+ln2pi)+(d-K+1)ln n <=> C_Q(K) < C_Q(K-1), all dataset pairs",
        agree_new == total, f"{agree_new}/{total}")
    chk("printed condition -2*dl > cost (report l.165) is wrong (records its agreement count)",
        agree_old < total, f"{agree_old}/{total} agree")
    targets = {"G01": 901.83, "GM": 311.63, "P": 290.30}
    chk("K=8->9 median 2*dl (10 datasets, 2-init mean) matches G01 901.83 / GM 311.63 / P 290.30",
        all(round(med[(c, 9)], 2) == t for c, t in targets.items()),
        {c: round(med[(c, 9)], 4) for c in targets})
    summ = {(r["cond"], int(r["k_to"])): r for r in rd(OUT / "cq_exact_increment_summary.csv")}
    worst = max(abs(med[key] - float(summ[key]["two_delta_ell_XY_exact_median"])) for key in med)
    chk("correction CSV medians reproduce (Q_strict route vs C_Q route)", worst < 1e-6, worst)
    # pooled-fit median differs -> unit matters
    pooled = {}
    for cond in ("G01", "GM", "P"):
        v = []
        for r in pf:
            if r["cond"] == cond and int(r["k"]) == 9:
                prev = [x for x in pf if x["cond"] == cond and x["dataset_id"] == r["dataset_id"]
                        and x["t"] == r["t"] and int(x["k"]) == 8][0]
                v.append(2 * ((float(r["Q_strict"]) + n * 9 * (1 + math.log(2 * math.pi)) / 2)
                              - (float(prev["Q_strict"]) + n * 8 * (1 + math.log(2 * math.pi)) / 2)))
        pooled[cond] = round(float(np.median(v)), 2)
    chk("unit check: median over 20 per-init increments (not the protocol unit) recorded for contrast", True, pooled)

    # 1-sample decomposition: median of sums != sum of medians
    terms = rd(DIAG / "cq_term_decomposition.csv")
    tg = defaultdict(list)
    for t in terms:
        tg[(t["cond"], t["dataset_id"], int(t["k"]))].append(t)
    m = lambda key, c: float(np.mean([float(x[c]) for x in tg[key]]))  # noqa: E731
    det = {}
    for cond in ("G01", "GM", "P"):
        dss = sorted({k[1] for k in tg if k[0] == cond})
        parts = {c: [m((cond, ds, 9), c) - m((cond, ds, 8), c) for ds in dss]
                 for c in ("m2_lnpX", "m2_lnpY", "m2_lnpZ", "penalty", "C_Q_1sample")}
        som = sum(float(np.median(parts[c])) for c in ("m2_lnpX", "m2_lnpY", "m2_lnpZ", "penalty"))
        mos = float(np.median(parts["C_Q_1sample"]))
        det[cond] = (round(som, 1), round(mos, 1))
        s = summ[(cond, 9)]
        assert abs(som - float(s["sum_of_1s_term_medians"])) < 1e-6 and abs(mos - float(s["median_of_1s_total"])) < 1e-6
    chk("8->9: sum of 1-sample term medians != median of 1-sample totals (reproduced)",
        all(abs(a - b) > 1 for a, b in det.values()), det)

    # C_Lap: own formula
    ok = [r for r in pf if r["lap_status"] == "OK"]
    worst = 0.0
    for r in ok:
        k = int(r["k"])
        lo = float(r["lap_phi_at_mode"]) + n * k / 2 * math.log(2 * math.pi) - 0.5 * float(r["lap_logdet_H"])
        cl = -2 * lo + (k * d - k * (k - 1) // 2) * math.log(n)
        worst = max(worst, abs(cl - float(r["C_Lap"])) / abs(cl))
    chk("C_Lap == -2[Phi(Zhat) + (nK/2)ln2pi - (1/2)ln|H|] + (Kd-K(K-1)/2) ln n (rel.)", worst < 1e-12, worst)
    chk("C_Lap status: 598 OK, 2 NOT_STATIONARY (G01 1, P 1); GM all OK",
        len(ok) == 598 and sum(1 for r in pf if r["cond"] == "GM" and r["lap_status"] == "OK") == 200,
        len(ok))

    # A: separated calibration measures, recomputed from prediction_diagnostics + NPZ
    pred = rd(DIAG / "prediction_diagnostics.csv")
    pfa = {r["fit_id"]: r for r in rd(BASE / "expA/run/per_fit.csv")}
    ga = defaultdict(list)
    for f in pred:
        if f["rate"] == "sparse":
            ga[(f["dataset_id"], f["family_x_fit"])].append(f)
    dss = sorted({k[0] for k in ga})
    mm = lambda ds, fam, c: float(np.mean([float(x[c]) for x in ga[(ds, fam)]]))  # noqa: E731
    cnt = {}
    for c, better in (("plugin_last_log_score", 1), ("plugin_last_ece", -1), ("plugin_last_p_sd", 0),
                      ("plugin_last_auc", 1)):
        cnt[c] = sum(mm(ds, "bernoulli", c) > mm(ds, "poisson", c) for ds in dss)
    cnt["slope_bern_closer_to_1"] = sum(abs(1 - mm(ds, "bernoulli", "plugin_last_calib_slope"))
                                        < abs(1 - mm(ds, "poisson", "plugin_last_calib_slope")) for ds in dss)
    cnt["zmean_ls_bern_gt_const"] = sum(mm(ds, "bernoulli", "plugin_mean_log_score") > mm(ds, "bernoulli", "const_log_score") for ds in dss)
    cnt["zmean_ls_pois_gt_const"] = sum(mm(ds, "poisson", "plugin_mean_log_score") > mm(ds, "bernoulli", "const_log_score") for ds in dss)
    expect = {"plugin_last_log_score": 1, "plugin_last_ece": 20, "plugin_last_p_sd": 20, "plugin_last_auc": 20,
              "slope_bern_closer_to_1": 18, "zmean_ls_bern_gt_const": 15, "zmean_ls_pois_gt_const": 10}
    chk("A sparse: Bern>Pois counts (log score 1, ECE 20 [=Pois better], SD 20, AUC 20); slope closer to 1: Bern 18; "
        "Z_mean plug-in beats constant: Bern 15, Pois 10", cnt == expect, cnt)
    zm = defaultdict(list)
    for f in pred:
        if f["rate"] == "sparse":
            st = np.load(BASE / "expA/run/states" / pfa[f["fit_id"]]["state_file"])
            zm[f["family_x_fit"]].append(float((st["Z_mean"] ** 2).mean()))
    chk("A sparse: mean square of Z_mean < 1 (shrunk; Bern ~0.60, Pois ~0.42)",
        all(v < 1 for v in zm["bernoulli"] + zm["poisson"]),
        {k: round(float(np.mean(v)), 3) for k, v in zm.items()})

    with open(OUT / "independent_check_correction.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "pass", "detail"])
        w.writeheader()
        w.writerows(res)
    print(f"{sum(r['pass'] for r in res)}/{len(res)} pass")
    for r in res:
        print(("ok   " if r["pass"] else "FAIL ") + r["check"] + " | " + r["detail"])


if __name__ == "__main__":
    main()
