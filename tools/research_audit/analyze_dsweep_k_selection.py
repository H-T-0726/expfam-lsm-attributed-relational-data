"""Pre-registered analysis of the d-sweep (written before any d=30/60 fit).

Reads expfam/results/k_selection/dsweep_k_selection_20261105/full/{fit_criteria,selection}.csv
and writes full/analysis/{per_replicate.csv, by_d.csv, residual_by_k.csv, summary.json}.
Verdict rules are the ones frozen in run_dsweep_k_selection.PREREG.
"""

from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import run_dsweep_k_selection as R                          # noqa: E402

FULL = R.OUT_ROOT / "full"
OUT = FULL / "analysis"
D = R.D_LEVELS
LN_N = math.log(R.N)
C2 = R.C2


def f(x):
    return float(x) if x not in ("", None) else float("nan")


def p15(k: int) -> int:
    return k * R.D0 - k * (k - 1) // 2


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"output exists: {OUT}")
    rows = list(csv.DictReader(open(FULL / "fit_criteria.csv", encoding="utf-8")))
    sel = list(csv.DictReader(open(FULL / "selection.csv", encoding="utf-8")))
    prov = list(csv.DictReader(open(FULL / "data_provenance.csv", encoding="utf-8")))
    idx = {(int(r["d"]), int(r["replicate"]), int(r["k_est"]), int(r["start"])): r for r in rows}
    S = {(int(r["d"]), int(r["replicate"]), r["criterion"]): r for r in sel}

    def m(d, rep, k, col):
        return st.mean(f(idx[(d, rep, k, s)][col]) for s in R.hist.STARTS)

    def ok(d, rep, ks):
        return all(idx[(d, rep, k, s)]["lap_status"] == "OK" for k in ks for s in R.hist.STARTS)

    per = []
    for d in D:
        for rep in R.REPLICATES:
            row = {"d": d, "replicate": rep}
            for crit in ("C_Q", "C_Lap"):
                x = S[(d, rep, crit)]
                row[f"{crit}_status"] = x["status"]
                row[f"{crit}_selected"] = x["selected_k"]
                row[f"{crit}_margin"] = x["margin"]
            # C_Q decomposition, slack = C(4) - C(5)
            row["slack_Q"] = m(d, rep, 4, "C_Q") - m(d, rep, 5, "C_Q")
            row["Q_fit_gain"] = m(d, rep, 4, "D_K") - m(d, rep, 5, "D_K")
            row["dP_Z"] = m(d, rep, 5, "P_Z") - m(d, rep, 4, "P_Z")
            row["dP_theta"] = m(d, rep, 5, "P_theta") - m(d, rep, 4, "P_theta")
            row["slack_Q_fixed_penalty"] = row["slack_Q"] + row["dP_theta"] - (p15(5) - p15(4)) * LN_N
            fq = {k: m(d, rep, k, "C_Q") - m(d, rep, k, "P_theta") + p15(k) * LN_N
                  for k in R.hist.CANDIDATE_K}
            row["C_Q_fixed_penalty_selected"] = min(fq, key=fq.get)
            row["lap45_complete"] = ok(d, rep, (4, 5))
            if row["lap45_complete"]:
                row["slack_Lap"] = m(d, rep, 4, "C_Lap") - m(d, rep, 5, "C_Lap")
                row["Lap_fit_gain"] = m(d, rep, 4, "lap_fit_mode") - m(d, rep, 5, "lap_fit_mode")
                row["dvolume"] = m(d, rep, 5, "lap_volume") - m(d, rep, 4, "lap_volume")
                dld = m(d, rep, 5, "lap_logdet_H") - m(d, rep, 4, "lap_logdet_H")
                row["dlogdetH_per_n"] = dld / R.N
                row["two_dH_Lap"] = R.N * C2 - dld
                row["d_logdet_prior_X"] = m(d, rep, 5, "logdet_prior_X_hat") - m(d, rep, 4, "logdet_prior_X_hat")
                row["d_logdet_Y_increment"] = m(d, rep, 5, "logdet_Y_increment_hat") - m(d, rep, 4, "logdet_Y_increment_hat")
                row["d_resid"] = m(d, rep, 5, "residual") - m(d, rep, 4, "residual")
                row["d_termA"] = m(d, rep, 5, "termA") - m(d, rep, 4, "termA")
                row["d_2R3"] = 2 * (m(d, rep, 5, "R3") - m(d, rep, 4, "R3"))
                row["identity_check"] = row["slack_Q"] - (row["slack_Lap"] - row["two_dH_Lap"] - row["d_resid"])
                row["slack_Lap_fixed_penalty"] = row["slack_Lap"] + row["dP_theta"] - (p15(5) - p15(4)) * LN_N
                row["xfisher_median_eig_K5"] = m(d, rep, 5, "xfisher_median_eig_hat")
                row["logdetH_per_n_K5"] = m(d, rep, 5, "lap_logdet_H") / R.N
                row["logdet_prior_X_per_n_K5"] = m(d, rep, 5, "logdet_prior_X_hat") / R.N
            if all(idx[(d, rep, k, s)]["lap_status"] == "OK" for k in R.hist.CANDIDATE_K for s in R.hist.STARTS):
                fl = {k: m(d, rep, k, "C_Lap") - m(d, rep, k, "P_theta") + p15(k) * LN_N
                      for k in R.hist.CANDIDATE_K}
                row["C_Lap_fixed_penalty_selected"] = min(fl, key=fl.get)
                sur = {k: m(d, rep, k, "C_Lap") + R.N * k * C2 - m(d, rep, k, "lap_logdet_H")
                       for k in R.hist.CANDIDATE_K}
                row["surrogate_selected"] = min(sur, key=sur.get)
            for col in ("gram_rel_change", "eta_x_rel_change", "eta_y_rel_change"):
                row[f"{col}_K5"] = m(d, rep, 5, col)
            pv = [p for p in prov if int(p["d"]) == d and int(p["replicate"]) == rep][0]
            for col in ("oracle_logdetH_per_n", "oracle_logdet_prior_X_per_n",
                        "oracle_xfisher_median_eig", "FtF_diag", "gate_fail_prob_union_bound"):
                row[col] = f(pv[col])
            per.append(row)

    def med(d, col, cond=lambda r: True):
        v = [r[col] for r in per if r["d"] == d and col in r and cond(r)
             and isinstance(r[col], (int, float)) and not math.isnan(r[col])]
        return (st.median(v), len(v)) if v else (float("nan"), 0)

    by_d = []
    for d in D:
        b = {"d": d}
        for crit in ("C_Q", "C_Lap"):
            xs = [r for r in per if r["d"] == d]
            comp = [r for r in xs if r[f"{crit}_status"] == "COMPLETE"]
            b[f"{crit}_complete"] = len(comp)
            b[f"{crit}_exact"] = sum(int(r[f"{crit}_selected"]) == R.K_TRUE for r in comp)
            b[f"{crit}_under"] = sum(int(r[f"{crit}_selected"]) < R.K_TRUE for r in comp)
            b[f"{crit}_over"] = sum(int(r[f"{crit}_selected"]) > R.K_TRUE for r in comp)
            b[f"{crit}_selected"] = "".join(r[f"{crit}_selected"] or "i" for r in xs)
        both = [r for r in per if r["d"] == d and r["C_Lap_status"] == "COMPLETE"]
        b["disagree"] = sum(r["C_Q_selected"] != r["C_Lap_selected"] for r in both)
        b["disagree_den"] = len(both)
        b["C_Q_fixed_penalty_exact"] = sum(r["C_Q_fixed_penalty_selected"] == R.K_TRUE for r in per if r["d"] == d)
        fx = [r for r in per if r["d"] == d and "C_Lap_fixed_penalty_selected" in r]
        b["C_Lap_fixed_penalty_exact"] = f"{sum(r['C_Lap_fixed_penalty_selected'] == R.K_TRUE for r in fx)}/{len(fx)}"
        b["surrogate_vs_C_Q_flips"] = f"{sum(r['surrogate_selected'] != int(r['C_Q_selected']) for r in fx)}/{len(fx)}"
        for col in ("slack_Q", "Q_fit_gain", "dP_Z", "dP_theta", "slack_Q_fixed_penalty",
                    "slack_Lap", "Lap_fit_gain", "dvolume", "two_dH_Lap", "dlogdetH_per_n",
                    "d_logdet_prior_X", "d_logdet_Y_increment", "d_resid", "d_termA", "d_2R3",
                    "slack_Lap_fixed_penalty", "xfisher_median_eig_K5", "logdetH_per_n_K5",
                    "logdet_prior_X_per_n_K5", "gram_rel_change_K5", "eta_x_rel_change_K5",
                    "eta_y_rel_change_K5", "oracle_logdetH_per_n", "oracle_xfisher_median_eig",
                    "FtF_diag", "gate_fail_prob_union_bound"):
            b[f"med_{col}"], b[f"n_{col}"] = med(d, col)
        b["max_abs_identity_check"] = max((abs(r["identity_check"]) for r in per
                                           if r["d"] == d and "identity_check" in r), default=float("nan"))
        b["med_abs_d_resid"] = st.median(
            [abs(r["d_resid"]) for r in per if r["d"] == d and "d_resid" in r] or [float("nan")])
        b["med_abs_two_dH_Lap"] = st.median(
            [abs(r["two_dH_Lap"]) for r in per if r["d"] == d and "two_dH_Lap" in r] or [float("nan")])
        b["slack_Q_positive"] = sum(r["slack_Q"] > 0 for r in per if r["d"] == d)
        b["slack_Lap_positive"] = f"{sum(r['slack_Lap'] > 0 for r in per if r['d'] == d and 'slack_Lap' in r)}/{med(d, 'slack_Lap')[1]}"
        by_d.append(b)
    B = {b["d"]: b for b in by_d}

    # residual K-dependence
    res_k = []
    for d in D:
        for k in R.hist.CANDIDATE_K:
            v = [(f(r["residual"]), f(r["termA"]), 2 * f(r["R3"])) for r in rows
                 if int(r["d"]) == d and int(r["k_est"]) == k and r["lap_status"] == "OK"]
            res_k.append({"d": d, "k_est": k, "n_ok": len(v),
                          "med_residual": st.median(x[0] for x in v),
                          "med_termA": st.median(x[1] for x in v),
                          "med_2R3": st.median(x[2] for x in v)})

    # paired, replicates complete at both compared d
    def paired(col, d1, d2):
        a = {r["replicate"]: r[col] for r in per if r["d"] == d1 and col in r}
        b = {r["replicate"]: r[col] for r in per if r["d"] == d2 and col in r}
        common = sorted(set(a) & set(b))
        return {"n": len(common), "increase": sum(b[i] > a[i] for i in common),
                "median_change": st.median([b[i] - a[i] for i in common]) if common else float("nan")}

    pairs = {f"{col}:{d1}->{d2}": paired(col, d1, d2)
             for col in ("slack_Q", "two_dH_Lap", "slack_Q_fixed_penalty", "slack_Lap")
             for d1, d2 in ((15, 30), (30, 60), (15, 60))}

    def verdict_mono(col, refute_rule):
        m15, m30, m60 = (B[d][f"med_{col}"] for d in D)
        if refute_rule(m15, m60):
            return "REFUTED"
        return "SUPPORTED" if m15 < m30 < m60 else "PARTIAL"

    h1 = verdict_mono("slack_Q", lambda a, c: c <= a)
    h3 = verdict_mono("slack_Q_fixed_penalty", lambda a, c: c <= a)
    t15, t30, t60 = (B[d]["med_two_dH_Lap"] for d in D)
    h2a = "REFUTED" if t60 >= t15 else ("SUPPORTED" if t15 > t30 > t60 else "PARTIAL")
    h2b = "REFUTED" if t60 <= 0 else "SUPPORTED"
    h2c = "REFUTED" if any(B[d]["med_abs_d_resid"] > B[d]["med_abs_two_dH_Lap"] for d in D) else "SUPPORTED"
    summary = {"prereg_sha256": R.prereg_hash(),
               "verdicts": {"H1": h1, "H2a": h2a, "H2b": h2b, "H2c": h2c, "H3": h3},
               "medians": {d: {"slack_Q": B[d]["med_slack_Q"], "two_dH_Lap": B[d]["med_two_dH_Lap"],
                               "slack_Q_fixed_penalty": B[d]["med_slack_Q_fixed_penalty"],
                               "med_abs_d_resid": B[d]["med_abs_d_resid"]} for d in D},
               "paired": pairs}
    OUT.mkdir(parents=True)
    R.write_rows(OUT / "per_replicate.csv", per)
    R.write_rows(OUT / "by_d.csv", by_d)
    R.write_rows(OUT / "residual_by_k.csv", res_k)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
