"""Forward correction of diagnostics_20261127 (two-gaps Exp A/B) -- no new fits.

Reads only saved per_fit.csv files, saved NPZ states (var_z, Z_mean) and the existing
diagnostics CSVs. Writes into diagnostics_20261127/forward_correction_20261127/ (new files only).

B (exact, not a 1-sample approximation):
  C_Q = -2 Q_strict + p ln n,  Q_strict = (1/L) sum_l [lnpZ_l + lnpX_l + lnpY_l] + strict corrections.
  scale_Z rescales all L samples jointly so that (1/L) sum_l ||Z_l||_F^2 = nK, and var_z = 1, hence
  -2 (1/L) sum_l lnpZ_l = nK(1 + ln 2pi)  exactly.  Therefore
  -2 l(K) := -2 [(1/L) sum_l (lnpX_l + lnpY_l) + corrections] = C_Q - p ln n - nK(1 + ln 2pi)
  and with Delta l = l(K) - l(K-1):  C_Q(K) < C_Q(K-1)  <=>  2 Delta l > n(1+ln 2pi) + (d-K+1) ln n.
A: calibration-type measures kept separate (ECE, slope, SD of p, log score) + mean square of Z_mean.
C_Lap: stored-term closure, penalty count and the Laplace identity, from per_fit.csv only.
Usage: python tools/research_audit/correct_two_gaps_diagnostics_20261127.py
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
N, D = 150, 15
LN_N = math.log(N)
ZCONST = N * (1 + math.log(2 * math.pi))          # 425.68... per latent dimension


def rd(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def wr(p, rows):
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def cost(k):
    """C_Q cost of going from K-1 to K latent dimensions."""
    return ZCONST + (D - k + 1) * LN_N


def part_b():
    pf = [r for r in rd(BASE / "expB/run/per_fit.csv") if r["status"] == "OK"]
    assert len(pf) == 600
    fits = []
    for r in pf:
        k = int(r["k"])
        st = np.load(BASE / "expB/run/states" / r["state_file"])
        var_z = float(st["var_z"])
        assert var_z == 1.0, (r["fit_id"], var_z)
        p = int(r["num_params"])
        pen = p * LN_N
        m2z = N * k * (1 + math.log(2 * math.pi))
        cq = float(r["C_Q"])
        fits.append({"fit_id": r["fit_id"], "cond": r["cond"], "dataset_id": r["dataset_id"], "t": r["t"],
                     "k": k, "num_params": p, "penalty": pen, "var_z": var_z, "C_Q": cq,
                     "m2_lnpZ_exact": m2z, "m2_ell_XY_exact": cq - pen - m2z,
                     "Q_strict": float(r["Q_strict"]),
                     "Q_minus_lnpZ_minus_ell_check": float(r["Q_strict"]) + m2z / 2 + (cq - pen - m2z) / 2})
    wr(OUT / "cq_exact_xy_recovery_per_fit.csv", fits)

    # 2-init mean per (cond, dataset, k)
    g = defaultdict(list)
    for f in fits:
        g[(f["cond"], f["dataset_id"], f["k"])].append(f)
    cell = {key: {c: float(np.mean([x[c] for x in v])) for c in ("C_Q", "m2_ell_XY_exact", "num_params")}
            for key, v in g.items()}
    # 1-sample term decomposition (existing CSV), 2-init mean
    terms = rd(DIAG / "cq_term_decomposition.csv")
    tg = defaultdict(list)
    for t in terms:
        tg[(t["cond"], t["dataset_id"], int(t["k"]))].append(t)
    tcell = {key: {c: float(np.mean([float(x[c]) for x in v]))
                   for c in ("m2_lnpX", "m2_lnpY", "m2_lnpZ", "penalty", "C_Q_1sample")}
             for key, v in tg.items()}
    sel = {}
    for cond in ("G01", "GM", "P"):
        for ds in sorted({k[1] for k in cell if k[0] == cond}):
            sel[(cond, ds)] = min(range(1, 11), key=lambda k: cell[(cond, ds, k)]["C_Q"])

    per_ds, summ = [], []
    for cond in ("G01", "GM", "P"):
        dss = sorted({k[1] for k in cell if k[0] == cond})
        for k in range(2, 11):
            rows = []
            for ds in dss:
                a, b = cell[(cond, ds, k - 1)], cell[(cond, ds, k)]
                ta, tb = tcell[(cond, ds, k - 1)], tcell[(cond, ds, k)]
                two_dl = a["m2_ell_XY_exact"] - b["m2_ell_XY_exact"]
                d_cq = b["C_Q"] - a["C_Q"]
                dp = b["num_params"] - a["num_params"]
                row = {"cond": cond, "dataset_id": ds, "k_from": k - 1, "k_to": k,
                       "two_delta_ell_XY_exact": two_dl, "cost": cost(k), "delta_num_params": dp,
                       "delta_C_Q_saved": d_cq,
                       "identity_resid": d_cq - (cost(k) - two_dl),
                       "increase_favoured": two_dl > cost(k),
                       "inc_m2_lnpX_1s": tb["m2_lnpX"] - ta["m2_lnpX"],
                       "inc_m2_lnpY_1s": tb["m2_lnpY"] - ta["m2_lnpY"],
                       "inc_m2_lnpZ_1s": tb["m2_lnpZ"] - ta["m2_lnpZ"],
                       "inc_penalty": tb["penalty"] - ta["penalty"],
                       "inc_C_Q_1sample": tb["C_Q_1sample"] - ta["C_Q_1sample"],
                       "selected_k_saved_C_Q": sel[(cond, ds)]}
                rows.append(row)
            per_ds += rows

            def med(c):
                return float(np.median([r[c] for r in rows]))
            parts = ("inc_m2_lnpX_1s", "inc_m2_lnpY_1s", "inc_m2_lnpZ_1s", "inc_penalty")
            summ.append({"cond": cond, "k_from": k - 1, "k_to": k, "n_datasets": len(rows),
                         "two_delta_ell_XY_exact_median": med("two_delta_ell_XY_exact"),
                         "two_delta_ell_XY_exact_min": min(r["two_delta_ell_XY_exact"] for r in rows),
                         "two_delta_ell_XY_exact_max": max(r["two_delta_ell_XY_exact"] for r in rows),
                         "cost": cost(k),
                         "n_datasets_increase_favoured": sum(r["increase_favoured"] for r in rows),
                         "delta_C_Q_saved_median": med("delta_C_Q_saved"),
                         "cost_minus_median_two_delta_ell": cost(k) - med("two_delta_ell_XY_exact"),
                         "max_abs_identity_resid": max(abs(r["identity_resid"]) for r in rows),
                         "one_sample_two_delta_ell_XY_median": -float(np.median(
                             [r["inc_m2_lnpX_1s"] + r["inc_m2_lnpY_1s"] for r in rows])),
                         **{f"{c}_median": med(c) for c in parts},
                         "sum_of_1s_term_medians": sum(med(c) for c in parts),
                         "median_of_1s_total": med("inc_C_Q_1sample"),
                         "median_of_saved_total": med("delta_C_Q_saved")})
    wr(OUT / "cq_exact_increment_per_dataset.csv", per_ds)
    wr(OUT / "cq_exact_increment_summary.csv", summ)
    return summ


def part_a():
    pf = {r["fit_id"]: r for r in rd(BASE / "expA/run/per_fit.csv")}
    pred = rd(DIAG / "prediction_diagnostics.csv")
    zms = {}
    for f in pred:
        st = np.load(BASE / "expA/run/states" / pf[f["fit_id"]]["state_file"])
        zms[f["fit_id"]] = (float(np.mean(st["Z_last"] ** 2)), float(np.mean(st["Z_mean"] ** 2)))
    g = defaultdict(list)
    for f in pred:
        g[(f["dataset_id"], f["rate"], f["family_x_fit"])].append(f)
    cols = ("plugin_last_log_score", "plugin_last_ece", "plugin_last_calib_slope", "plugin_last_p_sd",
            "plugin_last_calib_in_large", "plugin_last_auc", "plugin_mean_log_score", "plugin_mean_ece",
            "plugin_mean_calib_slope", "plugin_mean_p_sd", "plugin_mean_auc", "const_log_score")
    cell = {}
    for key, v in g.items():
        c = {m: float(np.mean([float(x[m]) for x in v])) for m in cols}
        c["Z_last_meansq"] = float(np.mean([zms[x["fit_id"]][0] for x in v]))
        c["Z_mean_meansq"] = float(np.mean([zms[x["fit_id"]][1] for x in v]))
        cell[key] = c
    out = []
    for rate in ("dense", "sparse"):
        dss = sorted({k[0] for k in cell if k[1] == rate})
        for m in cols + ("Z_last_meansq", "Z_mean_meansq"):
            b = np.array([cell[(d, rate, "bernoulli")][m] for d in dss])
            p = np.array([cell[(d, rate, "poisson")][m] for d in dss])
            row = {"rate": rate, "metric": m, "n_datasets": len(dss),
                   "bernoulli_mean": b.mean(), "poisson_mean": p.mean(),
                   "bernoulli_median": float(np.median(b)), "poisson_median": float(np.median(p)),
                   "n_bern_gt_pois": int((b > p).sum())}
            if m.endswith("log_score") and m != "const_log_score":
                cst_b = np.array([cell[(d, rate, "bernoulli")]["const_log_score"] for d in dss])
                row["n_bern_better_than_const"] = int((b > cst_b).sum())
                row["n_pois_better_than_const"] = int((p > cst_b).sum())
            if m.endswith("calib_slope"):
                row["n_bern_closer_to_1"] = int((np.abs(1 - b) < np.abs(1 - p)).sum())
            out.append(row)
    keys = list(out[0]) + ["n_bern_better_than_const", "n_pois_better_than_const", "n_bern_closer_to_1"]
    for r in out:
        for k in keys:
            r.setdefault(k, "")
    wr(OUT / "a_calibration_measures_separated.csv", [{k: r[k] for k in keys} for r in out])
    return out


def part_clap():
    pf = [r for r in rd(BASE / "expB/run/per_fit.csv") if r["status"] == "OK"]
    ok = [r for r in pf if r["lap_status"] == "OK"]
    f = lambda r, c: float(r[c])  # noqa: E731
    rows = []

    def add(name, vals, note=""):
        rows.append({"check": name, "n": len(vals), "max_abs": max(vals) if vals else "", "note": note})
    add("d_K == K d - K(K-1)/2", [abs(int(r["d_K"]) - (int(r["k"]) * D - int(r["k"]) * (int(r["k"]) - 1) // 2)) for r in pf])
    add("lap_penalty == d_K ln n (n=150)", [abs(f(r, "lap_penalty") - int(r["d_K"]) * LN_N) for r in pf])
    add("C_Lap == -2 lap_log_observed + lap_penalty", [abs(f(r, "C_Lap") - (-2 * f(r, "lap_log_observed") + f(r, "lap_penalty"))) for r in ok])
    add("C_Lap == fit_mode + zhat_sq + logdet_H + penalty",
        [abs(f(r, "C_Lap") - (f(r, "lap_fit_mode") + f(r, "lap_zhat_sq") + f(r, "lap_logdet_H") + f(r, "lap_penalty"))) for r in ok])
    add("fit_mode == -2 (phi_at_mode - ln p(Z_hat)), ln p = -(nK/2)ln2pi - zhat_sq/2",
        [abs(f(r, "lap_fit_mode") + 2 * (f(r, "lap_phi_at_mode") + N * int(r["k"]) / 2 * math.log(2 * math.pi) + f(r, "lap_zhat_sq") / 2)) for r in pf if r["lap_phi_at_mode"]])
    add("log_observed == phi + (nK/2)ln2pi - logdet_H/2",
        [abs(f(r, "lap_log_observed") - (f(r, "lap_phi_at_mode") + N * int(r["k"]) / 2 * math.log(2 * math.pi) - f(r, "lap_logdet_H") / 2)) for r in ok])
    diffs = defaultdict(set)
    for r in pf:
        diffs[r["cond"]].add(int(r["num_params"]) - int(r["d_K"]))
    for c, s in sorted(diffs.items()):
        add(f"num_params(C_Q) - d_K(C_Lap), cond {c}", [0.0], f"values {sorted(s)} (constant in K: does not change the argmin)")
    st = defaultdict(int)
    for r in pf:
        st[(r["cond"], r["lap_status"])] += 1
    add("lap_status counts", [0.0], "; ".join(f"{c}:{s}={n}" for (c, s), n in sorted(st.items())))
    add("lap logdet sign == 1 for OK", [abs(f(r, "lap_logdet_sign") - 1) for r in ok])
    wr(OUT / "clap_formula_check.csv", rows)
    return rows


def main():
    OUT.mkdir(exist_ok=False)
    summ = part_b()
    for s in summ:
        if s["k_to"] == 9:
            print(f"{s['cond']} 8->9: 2dl median {s['two_delta_ell_XY_exact_median']:.2f} cost {s['cost']:.3f} "
                  f"favoured {s['n_datasets_increase_favoured']}/10 dCQ median {s['delta_C_Q_saved_median']:.2f} "
                  f"sum1s {s['sum_of_1s_term_medians']:.1f} med1s {s['median_of_1s_total']:.1f}")
    part_a()
    for r in part_clap():
        print(r)


if __name__ == "__main__":
    main()
