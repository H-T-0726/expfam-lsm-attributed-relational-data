"""Pre-registered analysis of R1 / R2 (written and committed before any production fit).

Reads the primary CSVs of the d-sweep, the checkpoint re-evaluation and R1 / R2,
recomputes every selection with one rule, and writes
expfam/results/k_selection/k_selection_robustness_20261105/analysis/
  conditions.csv, per_replicate.csv, residual_by_k.csv, paired.csv, summary.json
Verdicts follow run_k_selection_robustness.PREREG exactly.  Missing stages
(e.g. n=150 not run) are reported as NOT_RUN, never imputed.
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
import run_k_selection_robustness as R                      # noqa: E402

OUT = R.OUT_ROOT / "analysis"
KS = tuple(R.hist.CANDIDATE_K)
STARTS = tuple(R.hist.STARTS)
C2 = R.C2


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load(path: Path):
    return R.load_csv(path) if path.exists() else None


def conditions():
    """(label, n, d, est, rows) with rows keyed by (replicate, k, start)."""
    conds = []
    ds = load(R.DSWEEP_FULL / "fit_criteria.csv")
    for d in (15, 30, 60):
        conds.append(("n75_d%d_L5it8" % d, 75, d, "L5it8",
                      [r for r in ds if int(r["d"]) == d]))
    e1 = load(R.EXP1_FULL / "fit_criteria.csv")
    conds.append(("n75_d15_L5it16", 75, 15, "L5it16",
                  [{**r, "fit_status": "REUSED"} for r in e1 if int(r["n"]) == 75 and int(r["checkpoint"]) == 16]))
    r1 = load(R.OUT_ROOT / "r1" / "fit_criteria.csv")
    if r1 is not None:
        for d in (15, 30, 60):
            for cp, est in ((8, "L10it8"), (16, "L10it16")):
                conds.append((f"n75_d{d}_{est}", 75, d, est,
                              [r for r in r1 if int(r["d"]) == d and int(r["checkpoint"]) == cp]))
    for n in (50, 100, 150):
        r2 = load(R.OUT_ROOT / f"r2_n{n}" / "fit_criteria.csv")
        if r2 is None:
            continue
        for d in (15, 30):
            conds.append((f"n{n}_d{d}_L5it8", n, d, "L5it8", [r for r in r2 if int(r["d"]) == d]))
    return conds


def per_rep(label, n, d, est, rows):
    out = []
    reps = sorted({int(r["replicate"]) for r in rows})
    for rep in reps:
        by = {(int(r["k_est"]), int(r["start"])): r for r in rows if int(r["replicate"]) == rep}
        if len(by) != len(KS) * len(STARTS):
            raise SystemExit(f"{label} rep {rep}: {len(by)} fits")
        failed = any(r.get("fit_status") == "FAILED_NAN" for r in by.values())
        lap_ok = (not failed) and all(r["lap_status"] == "OK" for r in by.values())
        m = lambda k, col: st.mean(f(by[(k, s)].get(col)) for s in STARTS)
        row = {"condition": label, "n": n, "d": d, "est": est, "replicate": rep,
               "failed_fit": failed, "lap_complete": lap_ok}
        for crit, ok in (("C_Q", not failed), ("C_Lap", lap_ok)):
            if ok:
                means = {k: m(k, crit) for k in KS}
                srt = sorted(means.values())
                row[f"{crit}_selected"] = min(means, key=means.get)
                row[f"{crit}_margin"] = srt[1] - srt[0]
            else:
                row[f"{crit}_selected"] = ""
        if not failed:
            row["slack_Q"] = m(4, "C_Q") - m(5, "C_Q")
            row["Q_fit_gain"] = m(4, "D_K") - m(5, "D_K")
            row["dP_Z"] = m(5, "P_Z") - m(4, "P_Z")
            row["dP_theta"] = m(5, "P_theta") - m(4, "P_theta")
        if (not failed) and all(by[(k, s)]["lap_status"] == "OK" for k in (4, 5) for s in STARTS):
            dld = m(5, "lap_logdet_H") - m(4, "lap_logdet_H")
            row.update({"slack_Lap": m(4, "C_Lap") - m(5, "C_Lap"),
                        "Lap_fit_gain": m(4, "lap_fit_mode") - m(5, "lap_fit_mode"),
                        "dvolume": m(5, "lap_volume") - m(4, "lap_volume"),
                        "two_dH_Lap": n * C2 - dld, "two_dH_Lap_per_n": (n * C2 - dld) / n,
                        "dlogdetH_per_n": dld / n,
                        "d_logdet_prior_X": m(5, "logdet_prior_X_hat") - m(4, "logdet_prior_X_hat"),
                        "d_logdet_Y_increment": m(5, "logdet_Y_increment_hat") - m(4, "logdet_Y_increment_hat"),
                        "d_termA": m(5, "termA") - m(4, "termA"),
                        "d_2R3": 2 * (m(5, "R3") - m(4, "R3"))})
            resid = {k: m(k, "C_Q") - m(k, "C_Lap") - (n * k * C2 - m(k, "lap_logdet_H")) for k in (4, 5)}
            row["d_resid"] = resid[5] - resid[4]
            row["identity_check"] = row["slack_Q"] - (row["slack_Lap"] - row["two_dH_Lap"] - row["d_resid"])
        if lap_ok:
            sur = {k: m(k, "C_Lap") + n * k * C2 - m(k, "lap_logdet_H") for k in KS}
            row["surrogate_selected"] = min(sur, key=sur.get)
        if not failed:
            for col in ("gram_rel_change", "eta_x_rel_change", "eta_y_rel_change"):
                row[f"{col}_K5"] = m(5, col)
        out.append(row)
    return out


def med(rows, col):
    v = [r[col] for r in rows if col in r and isinstance(r[col], (int, float)) and not math.isnan(r[col])]
    return (st.median(v), len(v)) if v else (float("nan"), 0)


METRICS = ("slack_Q", "Q_fit_gain", "dP_Z", "dP_theta", "slack_Lap", "Lap_fit_gain", "dvolume",
           "two_dH_Lap", "two_dH_Lap_per_n", "dlogdetH_per_n", "d_logdet_prior_X",
           "d_logdet_Y_increment", "d_resid", "d_termA", "d_2R3", "gram_rel_change_K5",
           "eta_x_rel_change_K5", "eta_y_rel_change_K5", "C_Q_margin", "C_Lap_margin")


def summarise(label, n, d, est, reps):
    c = {"condition": label, "n": n, "d": d, "est": est, "datasets": len(reps),
         "failed_fit_datasets": sum(r["failed_fit"] for r in reps)}
    for crit in ("C_Q", "C_Lap"):
        comp = [r for r in reps if r[f"{crit}_selected"] != ""]
        c[f"{crit}_complete"] = len(comp)
        c[f"{crit}_exact"] = sum(r[f"{crit}_selected"] == R.K_TRUE for r in comp)
        c[f"{crit}_under"] = sum(r[f"{crit}_selected"] < R.K_TRUE for r in comp)
        c[f"{crit}_over"] = sum(r[f"{crit}_selected"] > R.K_TRUE for r in comp)
        c[f"{crit}_selected"] = "".join(str(r[f"{crit}_selected"]) or "i" for r in reps)
    both = [r for r in reps if r["C_Q_selected"] != "" and r["C_Lap_selected"] != ""]
    c["disagree"] = f"{sum(r['C_Q_selected'] != r['C_Lap_selected'] for r in both)}/{len(both)}"
    c["lap_less_than_q"] = sum(r["C_Lap_selected"] < r["C_Q_selected"] for r in both)
    sur = [r for r in reps if "surrogate_selected" in r]
    c["surrogate_flips"] = f"{sum(r['surrogate_selected'] != r['C_Q_selected'] for r in sur)}/{len(sur)}"
    for col in METRICS:
        c[f"med_{col}"], c[f"n_{col}"] = med(reps, col)
    ic = [abs(r["identity_check"]) for r in reps if "identity_check" in r]
    c["max_abs_identity_check"] = max(ic) if ic else float("nan")
    c["med_abs_d_resid"] = st.median([abs(r["d_resid"]) for r in reps if "d_resid" in r] or [float("nan")])
    return c


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"output exists: {OUT}")
    conds = conditions()
    per, summ = [], {}
    for label, n, d, est, rows in conds:
        reps = per_rep(label, n, d, est, rows)
        per += reps
        summ[label] = summarise(label, n, d, est, reps)

    # residual K-dependence
    res_k = []
    for label, n, d, est, rows in conds:
        for k in KS:
            v = [r for r in rows if int(r["k_est"]) == k and r.get("lap_status") == "OK" and r.get("termA")]
            if v:
                res_k.append({"condition": label, "k_est": k, "n_ok": len(v),
                              "med_residual": st.median(f(r["C_Q"]) - f(r["C_Lap"]) - (n * k * C2 - f(r["lap_logdet_H"])) for r in v),
                              "med_termA": st.median(f(r["termA"]) for r in v),
                              "med_2R3": st.median(2 * f(r["R3"]) for r in v)})

    # paired comparisons (same datasets; only replicates usable in both)
    P = {(r["condition"], r["replicate"]): r for r in per}

    def paired(a, b, col):
        common = [rep for rep in range(1, 9) if (a, rep) in P and (b, rep) in P
                  and col in P[(a, rep)] and col in P[(b, rep)]]
        ch = [P[(b, rep)][col] - P[(a, rep)][col] for rep in common]
        same = [rep for rep in range(1, 9) if (a, rep) in P and (b, rep) in P
                and P[(a, rep)]["C_Q_selected"] != "" and P[(b, rep)]["C_Q_selected"] != ""]
        return {"from": a, "to": b, "metric": col, "n_pairs": len(common),
                "increase": sum(x > 0 for x in ch), "median_change": st.median(ch) if ch else float("nan"),
                "C_Q_same_K": f"{sum(P[(a, r)]['C_Q_selected'] == P[(b, r)]['C_Q_selected'] for r in same)}/{len(same)}"}

    pairs = []
    for d in (15, 30, 60):
        for a, b in ((f"n75_d{d}_L5it8", f"n75_d{d}_L10it8"), (f"n75_d{d}_L10it8", f"n75_d{d}_L10it16")):
            if a in summ and b in summ:
                for col in ("slack_Q", "slack_Lap", "two_dH_Lap"):
                    pairs.append(paired(a, b, col))
    if "n75_d15_L5it16" in summ:
        for col in ("slack_Q", "slack_Lap", "two_dH_Lap"):
            pairs.append(paired("n75_d15_L5it8", "n75_d15_L5it16", col))
    for n in (50, 75, 100, 150):
        a, b = f"n{n}_d15_L5it8", f"n{n}_d30_L5it8"
        if a in summ and b in summ:
            for col in ("slack_Q", "slack_Lap", "two_dH_Lap"):
                pairs.append(paired(a, b, col))

    def g(label, col):
        return summ[label][f"med_{col}"] if label in summ else float("nan")

    verdicts = {}
    if "n75_d15_L10it16" in summ:
        ok1 = all(g(f"n75_d60_{e}", "slack_Q") > g(f"n75_d15_{e}", "slack_Q") for e in ("L10it8", "L10it16"))
        verdicts["R1-H1"] = "SUPPORTED" if ok1 else "REFUTED"
        ok2 = all(g(f"n75_d60_{e}", "two_dH_Lap") < g(f"n75_d15_{e}", "two_dH_Lap")
                  and g(f"n75_d60_{e}", "two_dH_Lap") > 0 for e in ("L10it8", "L10it16"))
        verdicts["R1-H2"] = "SUPPORTED" if ok2 else "REFUTED"
        u = summ["n75_d15_L10it16"]
        verdicts["R1-H3"] = "SUPPORTED" if u["C_Q_under"] >= 5 else "REFUTED"
    else:
        verdicts.update({"R1-H1": "NOT_RUN", "R1-H2": "NOT_RUN", "R1-H3": "NOT_RUN"})
    done_n = [n for n in (50, 75, 100, 150) if f"n{n}_d30_L5it8" in summ and f"n{n}_d15_L5it8" in summ]
    new_n = [n for n in done_n if n != 75]
    if new_n:
        verdicts["R2-H1"] = ("SUPPORTED" if all(g(f"n{n}_d30_L5it8", "slack_Q") > g(f"n{n}_d15_L5it8", "slack_Q")
                                                for n in done_n) else "REFUTED")
        verdicts["R2-H2"] = ("SUPPORTED" if all(g(f"n{n}_d30_L5it8", "two_dH_Lap") < g(f"n{n}_d15_L5it8", "two_dH_Lap")
                                                for n in done_n) else "REFUTED")
        largest = max(done_n)
        verdicts["R2-H3"] = ("SUPPORTED" if g(f"n{largest}_d15_L5it8", "two_dH_Lap_per_n")
                             < g("n50_d15_L5it8", "two_dH_Lap_per_n") else "REFUTED")
        verdicts["R2_completed_n"] = done_n
    else:
        verdicts.update({"R2-H1": "NOT_RUN", "R2-H2": "NOT_RUN", "R2-H3": "NOT_RUN"})

    OUT.mkdir(parents=True)
    R.write_rows(OUT / "conditions.csv", list(summ.values()))
    R.write_rows(OUT / "per_replicate.csv", per)
    R.write_rows(OUT / "residual_by_k.csv", res_k)
    R.write_rows(OUT / "paired.csv", pairs)
    (OUT / "summary.json").write_text(json.dumps({"prereg_sha256": R.prereg_hash(), "verdicts": verdicts,
                                                  "conditions": list(summ)}, indent=2, default=float),
                                      encoding="utf-8")
    print(json.dumps(verdicts, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
