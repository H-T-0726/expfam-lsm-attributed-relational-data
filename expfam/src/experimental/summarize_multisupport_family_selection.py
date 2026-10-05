"""Derived summary tables for a multi-support family-selection run.

Reads ONLY the live primary tables of the run directory and (re)writes the
derived tables below.  Every row of the primary tables is used -- failures are
kept in the denominators and counted separately; nothing is filtered to
successful runs.

Derived outputs (tables/):
  family_recovery_summary.csv           recovery by true family
  candidate_model_recovery_summary.csv  recovery by true candidate model
  confusion_matrix_family.csv           long format: true x selected (+FAILURE)
  confusion_matrix_candidate.csv
  rmse_summary.csv                      RMSE(Z) by method
  rmse_paired_gaps.csv                  per dataset: Auto - Oracle, Auto - SupportOnly
  start_agreement.csv                   per dataset: START_FIRST vs START_LAST
  margin_summary.csv                    score margins (score-decided columns)
  convergence_summary.csv               candidate optimiser diagnostics
summary.json at the run root.

    python summarize_multisupport_family_selection.py --run-dir DIR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from run_multisupport_family_selection import (  # noqa: E402
    append_rows, atomic_write_json, read_csv,
)

SCOPES = {"all_columns": lambda r: True,
          "score_decided_only": lambda r: r.get("decided_by") == "score"}


def _b(v) -> bool:
    return str(v) == "True"


def _f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _write(path: Path, rows):
    if path.exists():
        path.unlink()
    if rows:
        append_rows(path, rows)


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def recovery(sel_rows, level):
    truth_key = "true_family" if level == "family" else "true_candidate_model"
    corr_key = "correct_family" if level == "family" else "correct_candidate_model"
    out = []
    groupings = [
        ("by_condition", lambda r: (r["experiment"], r["stage"], r["method"],
                                    r["group_design"], r["difficulty"], r["n"])),
        ("pooled_difficulty_n", lambda r: (r["experiment"], r["stage"], r["method"],
                                           r["group_design"], "ALL", "ALL")),
        ("by_difficulty", lambda r: (r["experiment"], r["stage"], r["method"],
                                     r["group_design"], r["difficulty"], "ALL")),
        ("by_n", lambda r: (r["experiment"], r["stage"], r["method"],
                            r["group_design"], "ALL", r["n"])),
    ]
    for scope, keep in SCOPES.items():
        for gname, gkey in groupings:
            acc = defaultdict(lambda: [0, 0, 0])
            for r in sel_rows:
                if not keep(r) and r.get("selection_status") != "FAILURE":
                    continue
                if scope == "score_decided_only" and r.get("selection_status") == "FAILURE":
                    # a failed case has no gate record; it stays in the
                    # all-columns scope only and is counted there
                    continue
                k = gkey(r)
                sg = r.get("true_support_group", "")
                for tk in ((sg, r[truth_key]), (sg, "ALL"), ("ALL", "ALL")):
                    a = acc[k + tk]
                    a[0] += 1
                    a[1] += _b(r[corr_key])
                    a[2] += r.get("selection_status") == "FAILURE"
            for key, (n_ev, n_ok, n_fail) in sorted(acc.items()):
                lo, hi = _wilson(n_ok, n_ev)
                out.append({"scope": scope, "grouping": gname,
                            "experiment": key[0], "stage": key[1], "method": key[2],
                            "group_design": key[3], "difficulty": key[4], "n": key[5],
                            "support_group": key[6],
                            ("true_family" if level == "family" else "true_candidate_model"): key[7],
                            "n_evaluated": n_ev, "n_correct": n_ok, "n_failed": n_fail,
                            "recovery_rate": n_ok / n_ev if n_ev else float("nan"),
                            "wilson95_low": lo, "wilson95_high": hi})
    return out


def confusion(sel_rows, level):
    tk = "true_family" if level == "family" else "true_candidate_model"
    sk = "selected_family" if level == "family" else "selected_candidate_model"
    acc = defaultdict(int)
    for r in sel_rows:
        sel = r[sk] if r.get("selection_status") == "SELECTED" else "FAILURE"
        for diff, n in ((r["difficulty"], r["n"]), ("ALL", "ALL")):
            acc[(r["experiment"], r["stage"], r["method"], r["group_design"],
                 r.get("true_support_group", ""), diff, n, r[tk], sel)] += 1
    return [{"experiment": k[0], "stage": k[1], "method": k[2], "group_design": k[3],
             "support_group": k[4], "difficulty": k[5], "n": k[6],
             f"true_{level if level == 'family' else 'candidate_model'}": k[7],
             f"selected_{level if level == 'family' else 'candidate_model'}": k[8],
             "count": v} for k, v in sorted(acc.items())]


def _dataset_id(r):
    return (r["experiment"], r["stage"], r["group_design"], r["difficulty"], r["n"],
            r["replicate"])


def rmse_tables(fit_rows):
    summ = defaultdict(list)
    status = defaultdict(lambda: [0, 0])
    by_ds = defaultdict(dict)
    for r in fit_rows:
        key = (r["experiment"], r["stage"], r["group_design"], r["difficulty"], r["n"],
               r["method"])
        status[key][0] += 1
        v = _f(r["rmse_Z"]) if r["status"] == "SUCCESS" else None
        if v is None:
            status[key][1] += 1
        else:
            summ[key].append(v)
        by_ds[_dataset_id(r)][r["method"]] = v
    rows = []
    for key in sorted(status):
        vals = np.array(summ.get(key, []), dtype=float)
        rows.append({"experiment": key[0], "stage": key[1], "group_design": key[2],
                     "difficulty": key[3], "n": key[4], "method": key[5],
                     "n_fits": status[key][0], "n_failed": status[key][1],
                     "rmse_mean": float(vals.mean()) if vals.size else float("nan"),
                     "rmse_sd": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
                     "rmse_median": float(np.median(vals)) if vals.size else float("nan")})
    gaps = []
    for ds, m in sorted(by_ds.items()):
        for auto in ("AUTO_START_FIRST", "AUTO_START_LAST"):
            a = m.get(auto)
            gaps.append({"experiment": ds[0], "stage": ds[1], "group_design": ds[2],
                         "difficulty": ds[3], "n": ds[4], "replicate": ds[5],
                         "auto_method": auto, "rmse_auto": a,
                         "rmse_oracle": m.get("ORACLE"),
                         "rmse_support_only": m.get("SUPPORT_ONLY"),
                         "gap_auto_minus_oracle": (a - m["ORACLE"]) if a is not None and m.get("ORACLE") is not None else None,
                         "gap_auto_minus_support_only": (a - m["SUPPORT_ONLY"]) if a is not None and m.get("SUPPORT_ONLY") is not None else None,
                         "complete_pair": a is not None and m.get("ORACLE") is not None and m.get("SUPPORT_ONLY") is not None})
    return rows, gaps


def start_agreement(fit_rows):
    by_ds = defaultdict(dict)
    for r in fit_rows:
        if r["method"].startswith("AUTO_"):
            by_ds[_dataset_id(r)][r["method"]] = r
    rows = []
    for ds, m in sorted(by_ds.items()):
        a, b = m.get("AUTO_START_FIRST"), m.get("AUTO_START_LAST")
        both = bool(a and b and a["status"] == "SUCCESS" and b["status"] == "SUCCESS")
        fa = a["final_assignment"].split("|") if both else []
        fb = b["final_assignment"].split("|") if both else []
        rows.append({"experiment": ds[0], "stage": ds[1], "group_design": ds[2],
                     "difficulty": ds[3], "n": ds[4], "replicate": ds[5],
                     "both_succeeded": both,
                     "full_assignment_agree": both and fa == fb,
                     "n_columns": len(fa), "n_columns_agree": sum(x == y for x, y in zip(fa, fb)),
                     "same_refit_Q": both and a["final_Q"] == b["final_Q"]})
    return rows


def margin_summary(sel_rows):
    acc = defaultdict(list)
    for r in sel_rows:
        if r.get("decided_by") != "score" or r.get("selection_status") != "SELECTED":
            continue
        v = _f(r["score_margin"])
        if v is None:
            continue
        for diff in (r["difficulty"], "ALL"):
            acc[(r["experiment"], r["stage"], r["method"], r["true_support_group"],
                 r["true_candidate_model"], diff, _b(r["correct_candidate_model"]))].append(v)
    rows = []
    for k, vals in sorted(acc.items()):
        v = np.array(vals)
        rows.append({"experiment": k[0], "stage": k[1], "method": k[2], "support_group": k[3],
                     "true_candidate_model": k[4], "difficulty": k[5], "selection_correct": k[6],
                     "n": v.size, "margin_min": v.min(), "margin_q25": np.quantile(v, 0.25),
                     "margin_median": np.median(v), "margin_q75": np.quantile(v, 0.75),
                     "margin_max": v.max(), "n_margin_below_2": int((v < 2).sum())})
    return rows


def convergence_summary(diag_rows):
    acc = defaultdict(lambda: [0, 0, 0, []])
    for r in diag_rows:
        k = (r["experiment"], r["stage"], r["candidate_model"])
        a = acc[k]
        a[0] += 1
        a[1] += r["status"] != "OK"
        g = _f(r["gradient_inf"])
        if g is not None:
            a[3].append(g)
            a[2] += g > 1e-8
    rows = []
    for k, (n, nf, nw, g) in sorted(acc.items()):
        g = np.array(g) if g else np.array([np.nan])
        rows.append({"experiment": k[0], "stage": k[1], "candidate_model": k[2],
                     "n_candidate_evaluations": n, "n_failed": nf,
                     "n_grad_inf_above_1e-8_WARNING": nw,
                     "grad_inf_median": float(np.nanmedian(g)),
                     "grad_inf_max": float(np.nanmax(g))})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    a = ap.parse_args(argv)
    run, t = a.run_dir, a.run_dir / "tables"
    sel = read_csv(t / "selection_results.csv")
    fits = read_csv(t / "fit_results.csv")
    diag = read_csv(t / "optimization_diagnostics.csv")
    fam = recovery(sel, "family")
    cand = recovery(sel, "candidate")
    _write(t / "family_recovery_summary.csv", fam)
    _write(t / "candidate_model_recovery_summary.csv", cand)
    _write(t / "confusion_matrix_family.csv", confusion(sel, "family"))
    _write(t / "confusion_matrix_candidate.csv", confusion(sel, "candidate"))
    rs, gaps = rmse_tables(fits)
    _write(t / "rmse_summary.csv", rs)
    _write(t / "rmse_paired_gaps.csv", gaps)
    sa = start_agreement(fits)
    _write(t / "start_agreement.csv", sa)
    _write(t / "margin_summary.csv", margin_summary(sel))
    _write(t / "convergence_summary.csv", convergence_summary(diag))

    def pick(rows, **kw):
        return [r for r in rows if all(str(r.get(k)) == str(v) for k, v in kw.items())]

    headline = {}
    for exp, meth in (("A", "SELECTOR_ONLY"), ("B", "AUTO_START_FIRST"),
                      ("B", "AUTO_START_LAST"), ("C", "AUTO_START_FIRST"),
                      ("C", "AUTO_START_LAST")):
        for scope in SCOPES:
            for r in pick(fam, scope=scope, grouping="pooled_difficulty_n", experiment=exp,
                          stage="main", method=meth, true_family="ALL"):
                headline[f"{exp}|{meth}|{scope}|{r['group_design']}|{r['support_group']}"] = {
                    "n_evaluated": r["n_evaluated"], "n_correct": r["n_correct"],
                    "n_failed": r["n_failed"], "recovery_rate": r["recovery_rate"]}
    gap_stats = {}
    for g in ("COUNT", "POSITIVE", "BOUNDED", "MIXED"):
        for auto in ("AUTO_START_FIRST", "AUTO_START_LAST"):
            rows = [r for r in gaps if r["stage"] == "main" and r["group_design"] == g
                    and r["auto_method"] == auto and r["complete_pair"]]
            if rows:
                go = np.array([r["gap_auto_minus_oracle"] for r in rows])
                gs = np.array([r["gap_auto_minus_support_only"] for r in rows])
                gap_stats[f"{g}|{auto}"] = {"n_datasets": len(rows),
                                            "mean_gap_vs_oracle": float(go.mean()),
                                            "mean_gap_vs_support_only": float(gs.mean()),
                                            "n_auto_better_than_support_only": int((gs < 0).sum())}
    summary = {
        "run_dir": run.name,
        "primary_tables_rows": {"selection_results": len(sel), "fit_results": len(fits),
                                "optimization_diagnostics": len(diag)},
        "fits_failed": sum(1 for r in fits if r["status"] != "SUCCESS"),
        "selection_rows_failed": sum(1 for r in sel if r.get("selection_status") == "FAILURE"),
        "headline_family_recovery_main": headline,
        "rmse_gap_main": gap_stats,
        "start_agreement_main": {
            g: {"n": len([r for r in sa if r["stage"] == "main" and r["group_design"] == g]),
                "full_agree": sum(1 for r in sa if r["stage"] == "main" and r["group_design"] == g
                                  and r["full_assignment_agree"])}
            for g in ("COUNT", "POSITIVE", "BOUNDED", "MIXED")},
        "note": "derived from the live primary tables; failures kept in denominators",
    }
    atomic_write_json(run / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("primary_tables_rows", "fits_failed",
                                              "selection_rows_failed")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
