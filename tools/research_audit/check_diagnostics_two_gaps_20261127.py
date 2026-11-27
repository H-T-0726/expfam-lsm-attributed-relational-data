"""Independent cross-check of diagnostics_20261127 outputs (separate process).

Does NOT import diagnose_two_gaps_20261127.py or run_two_gaps_20261127.py.
  1. 2-init dataset means and paired summaries (A) recomputed from prediction_diagnostics.csv /
     parameter_diagnostics.csv.
  2. For 8 A fits: regenerate data with the library generator + pair split, recompute plug-in
     log score / Brier / AUC / AP with sklearn and own code from the NPZ, compare.
  3. B: Gaussian variances and estimated X-information traces recomputed from NPZ; C_Q term sums;
     per-K deltas vs K=9 recomputed from cq_term_decomposition.csv.
Usage: python tools/research_audit/check_diagnostics_two_gaps_20261127.py
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "expfam/results/k_selection/two_gaps_20261127"
OUT = BASE / "diagnostics_20261127"
sys.path.insert(0, str(ROOT / "expfam/src"))
sys.path.insert(0, str(ROOT / "expfam/src/experimental"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
res = []


def rd(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def chk(name, ok, detail=""):
    res.append({"check": name, "pass": bool(ok), "detail": str(detail)[:300]})


def main():
    # ---- 1. A aggregation
    fits = rd(OUT / "prediction_diagnostics.csv")
    par = {r["fit_id"]: r for r in rd(OUT / "parameter_diagnostics.csv")}
    g = defaultdict(list)
    for f in fits:
        g[(f["dataset_id"], f["rate"], f["family_x_fit"])].append(f)
    cells = {r["dataset_id"] + r["rate"] + r["family_x_fit"]: r for r in rd(OUT / "a_dataset_condition_means.csv")}
    worst = 0.0
    for (ds, rate, fam), v in g.items():
        c = cells[ds + rate + fam]
        for m in ("plugin_last_log_score", "plugin_last_auc", "plugin_mean_log_score", "const_log_score"):
            worst = max(worst, abs(np.mean([float(x[m]) for x in v]) - float(c[m])))
        worst = max(worst, abs(np.mean([float(par[x["fit_id"]]["w_hat"]) for x in v]) - float(c["w_hat"])))
    chk("A: 2-init means (dataset x rate x family) reproduce", worst < 1e-12, worst)
    summ = {(r["rate"], r["metric"]): r for r in rd(OUT / "a_paired_summary.csv")}
    worst, cnt_bad = 0.0, []
    for rate in ("dense", "sparse"):
        for m, hib in (("plugin_last_log_score", 1), ("plugin_last_auc", 1), ("plugin_mean_log_score", 1),
                       ("plugin_last_brier", -1), ("w_hat", 0), ("rmse_Z_last", -1)):
            ds = sorted({k[0] for k in g})
            diff = np.array([float(cells[d + rate + "bernoulli"][m]) - float(cells[d + rate + "poisson"][m]) for d in ds])
            s = summ[(rate, m)]
            worst = max(worst, abs(diff.mean() - float(s["mean_diff_bern_minus_pois"])),
                        abs(np.median(diff) - float(s["median_diff"])))
            if hib and int(s["n_bern_better"]) != int((diff * hib > 0).sum()):
                cnt_bad.append((rate, m))
    chk("A: paired mean / median differences reproduce", worst < 1e-12, worst)
    chk("A: counts of datasets where the correct family is better reproduce", not cnt_bad, cnt_bad)

    # ---- 2. A: independent recomputation of predictions for 8 fits
    import run_k_exp2_clap_comparison as rc            # frozen K-exp2 helper (design table only)
    from data_generator_canonical import generate_canonical_data
    from eval_utils import make_pair_split
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    kd = rc.kd
    w0 = float(rc.bern_table()[("a", 3)]["w0"])
    pf = {r["fit_id"]: r for r in rd(BASE / "expA/run/per_fit.csv")}
    pick = [f for f in fits if f["dataset_id"] in ("A-r000", "A-r013")]
    worst = 0.0
    for f in pick:
        r = pf[f["fit_id"]]
        seed = int(r["data_seed"])
        ds = generate_canonical_data(n=150, d=15, k=3, seed=seed, family_x="bernoulli", family_y="bernoulli",
                                     f_scale=kd.f_scale_of("a", 3), sigma_x_var=1.0, w0=w0, w=kd.w_of("O", "a", 3))
        _, test = make_pair_split(150, 0.10, seed=seed + 700_000)
        st = np.load(BASE / "expA/run/states" / r["state_file"])
        i, j = np.where(np.triu(test, 1))
        y = ds.Y[i, j]
        eta = float(st["w0"]) + float(st["w"]) * np.sum(st["Z_last"][i] * st["Z_last"][j], 1)
        p = 1 / (1 + np.exp(-eta))
        ls = np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))
        for mine, theirs in ((ls, f["plugin_last_log_score"]), (brier_score_loss(y, p), f["plugin_last_brier"]),
                             (roc_auc_score(y, p), f["plugin_last_auc"]),
                             (average_precision_score(y, p), f["plugin_last_ap"]),
                             (y.mean(), f["test_pos_rate"])):
            worst = max(worst, abs(mine - float(theirs)))
    chk("A: 8 fits re-predicted independently (log score, Brier, AUC, AP via sklearn)", worst < 1e-9, worst)

    # ---- 3. B
    terms = rd(OUT / "cq_term_decomposition.csv")
    worst = max(abs(float(t["m2_lnpZ"]) + float(t["m2_lnpX"]) + float(t["m2_lnpY"]) + float(t["penalty"])
                    - float(t["C_Q_1sample"])) for t in terms)
    chk("B: -2(lnpZ + lnpX + lnpY) + penalty = C_Q_1sample", worst < 1e-6, worst)
    pfb = {r["fit_id"]: r for r in rd(BASE / "expB/run/per_fit.csv")}
    worst = max(abs(float(t["C_Q_saved"]) - float(pfb[t["fit_id"]]["C_Q"])) for t in terms)
    chk("B: C_Q_saved = per_fit C_Q", worst == 0.0, worst)
    pb = [r for r in par.values() if r["exp"] == "B"]
    worst = 0.0
    for r in pb:
        st = np.load(BASE / "expB/run/states" / pfb[r["fit_id"]]["state_file"])
        if r["family_x_fit"] == "gaussian":
            sv = np.diag(st["sigma"])
            F = st["F"]
            tr = float(np.sum(F ** 2 / sv[:, None]))          # trace(F' diag(1/s) F)
            worst = max(worst, abs(sv.mean() - float(r["sigma2_mean"])), abs(tr - float(r["xinfo_est_trace"])))
        else:
            F, Z = st["F"], st["Z_last"]
            a = np.exp(Z @ F.T).mean(0)
            worst = max(worst, abs(float(np.sum(a[:, None] * F ** 2)) - float(r["xinfo_est_trace"])) / float(r["xinfo_est_trace"]))
    chk("B: estimated variances and X-information traces recomputed from NPZ", worst < 1e-9, worst)
    cm = defaultdict(list)
    for t in terms:
        cm[(t["cond"], t["dataset_id"], int(t["k"]))].append(t)
    dsum = {(r["cond"], int(r["k"])): r for r in rd(OUT / "cq_term_delta_summary.csv")}
    worst = 0.0
    for cond in ("G01", "GM", "P"):
        dss = sorted({k[1] for k in cm if k[0] == cond})
        for k in range(1, 11):
            for c in ("m2_lnpX", "m2_lnpY", "m2_lnpZ", "C_Q_saved"):
                v = [np.mean([float(x[c]) for x in cm[(cond, d, k)]]) - np.mean([float(x[c]) for x in cm[(cond, d, 9)]])
                     for d in dss]
                worst = max(worst, abs(np.median(v) - float(dsum[(cond, k)][f"d_{c}_vs9_median"])))
    chk("B: per-K term deltas vs K=9 (median of 2-init means) reproduce", worst < 1e-9, worst)
    # derived constant: n(1 + ln 2pi) per latent dimension
    inc = [float(dsum[(c, k)]["inc_m2_lnpZ_from_k-1_median"]) for c in ("G01", "GM", "P") for k in range(2, 11)]
    chk("B: median increment of -2 lnpZ per dimension ~ n(1+ln 2pi)=425.7 (recorded)", True,
        f"min {min(inc):.1f} max {max(inc):.1f}; derived {150 * (1 + math.log(2 * math.pi)):.1f}")

    with open(OUT / "independent_check.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "pass", "detail"])
        w.writeheader()
        w.writerows(res)
    print(f"{sum(r['pass'] for r in res)}/{len(res)} pass")
    for r in res:
        print(("ok   " if r["pass"] else "FAIL ") + r["check"] + " | " + r["detail"])


if __name__ == "__main__":
    main()
