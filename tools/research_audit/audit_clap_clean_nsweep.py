"""Independent artifact-only auditor for Issue #128 experiment A1.

Imports neither the runner nor any model code.  Every constant is restated
here as a literal, and every derived quantity (seeds, parameter counts,
criterion closure, C_Lap from its raw Laplace components, selections, labels,
margins, per-cell counts) is recomputed from the CSV values alone.

Usage:
    python audit_clap_clean_nsweep.py [--stage full|pilot] [--write-report]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "expfam/results/k_selection/clap_clean_nsweep_20261006"
HIST = ROOT / "expfam/results/k_selection/clean_true_k_asymptotics_20260904"

# --- independent literals --------------------------------------------------
D = 15
K_TRUE = (1, 3, 5)
N_GRID = (50, 75, 100, 150)
CAND = (1, 2, 3, 4, 5, 6, 7)
STARTS = (1, 2)
REPS = {1: 4, 3: 4, 5: 8}
REPLAY = {1, 3}
TIE = 1e-12
NPZ_RTOL, NPZ_ATOL = 1e-9, 1e-6
CLOSE = 1e-6
GRAD_TOL = 1e-8
LOWER = {"C_Q": True, "C_Lap": True, "S1": False, "S3": True}
HIST_NAME = {"C_Q": "S2", "S1": "S1", "S3": "S3"}
HIST_FILES = ("fit_results.csv", "selection_matrix.csv", "manifest.csv",
              "generator_provenance.csv", "protocol.json", "runinfo.json",
              "summary.json", "audit_report.json", "gram_spectrum.csv")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(x: str) -> float:
    return float(x) if x not in ("", None) else float("nan")


class Audit:
    def __init__(self) -> None:
        self.findings: list[dict[str, str]] = []
        self.checks = 0

    def check(self, cond: bool, severity: str, msg: str) -> None:
        self.checks += 1
        if not cond:
            self.findings.append({"severity": severity, "finding": msg})


def select(values: dict[int, float], lower: bool):
    s = {k: (-v if lower else v) for k, v in values.items()}
    best = max(s.values())
    ties = sorted(k for k, v in s.items() if best - v <= TIE)
    k = min(ties)
    rest = sorted((v for kk, v in s.items() if kk != k), reverse=True)
    return k, ties, best - rest[0]


def lab(k: int, t: int) -> str:
    return "exact" if k == t else ("under" if k < t else "over")


def run(stage: str) -> dict:
    a = Audit()
    d = RUN_ROOT / stage
    rows = read(d / "fit_criteria.csv")
    sel = read(d / "selection.csv")
    summary = json.loads((d / "summary.json").read_text("utf-8"))
    info = json.loads((d / "runinfo.json").read_text("utf-8"))
    hist_rows = {(int(r["k_true"]), int(r["n"]), int(r["replicate"]),
                  int(r["k_est"]), int(r["start"])): r
                 for r in read(HIST / "fit_results.csv")}
    hist_sel = {(r["criterion"], int(r["K_TRUE"]), int(r["n"]),
                 int(r["replicate"])): int(r["selected_k"])
                for r in read(HIST / "selection_matrix.csv")}

    # ---- run count / completeness ----------------------------------------
    if stage == "full":
        cells = [(t, n, r) for t in K_TRUE for n in N_GRID
                 for r in range(1, REPS[t] + 1)]
    else:
        cells = [(3, 50, 1), (5, 50, 1)]
    expected = {(t, n, r, k, s) for (t, n, r) in cells for k in CAND
                for s in STARTS}
    keys = [(int(x["k_true"]), int(x["n"]), int(x["replicate"]),
             int(x["k_est"]), int(x["start"])) for x in rows]
    a.check(len(keys) == len(set(keys)), "BLOCKER", "duplicate fit rows")
    a.check(set(keys) == expected, "BLOCKER",
            f"fit key set != expected ({len(keys)} vs {len(expected)})")
    a.check(info.get("status") == "COMPLETED", "BLOCKER", "runinfo not COMPLETED")
    a.check(info.get("completed_rows") == len(expected), "BLOCKER",
            "runinfo completed_rows mismatch")
    for key in ("retry_count", "replacement_fits_executed",
                "seed_rescue_count", "tolerance_relaxations"):
        a.check(info.get(key) == 0, "BLOCKER", f"runinfo {key} != 0")
    a.check(info.get("resumed") is False, "BLOCKER", "run was resumed")
    a.check(not (d / "failure.json").exists(), "BLOCKER", "failure.json present")

    # ---- states archive hash ----------------------------------------------
    sp = Path(info["states_file"])
    a.check(sp.is_file() and sha(sp) == info["states_sha256"], "HIGH",
            "states archive missing or sha256 differs")

    # ---- historical immutability ------------------------------------------
    now = {n: sha(HIST / n) for n in HIST_FILES}
    a.check(info["historical_hashes_before"] == now
            and info["historical_hashes_after"] == now, "BLOCKER",
            "historical artifacts changed")

    # ---- per-row checks -----------------------------------------------------
    by_cell = defaultdict(dict)
    max_npz = defaultdict(float)
    for x in rows:
        t, n, r, k, s = (int(x["k_true"]), int(x["n"]), int(x["replicate"]),
                         int(x["k_est"]), int(x["start"]))
        by_cell[(t, n, r)][(k, s)] = x
        h = hist_rows[(t, n, r, k, s)]
        # seeds from independent literals
        a.check(int(x["data_seed"]) == 810000 + t * 10000 + n * 10 + r,
                "BLOCKER", f"data seed {t,n,r}")
        a.check(int(x["split_seed"]) == 820000 + t * 10000 + n * 10 + r,
                "BLOCKER", f"split seed {t,n,r}")
        a.check(int(x["model_seed"]) == 830000 + t * 100000 + n * 1000
                + r * 100 + k * 10 + s, "BLOCKER", f"model seed {t,n,r,k,s}")
        a.check(int(x["model_seed"]) == int(h["model_seed"]), "BLOCKER",
                "model seed differs from historical")
        npar = k * D - k * (k - 1) // 2
        a.check(int(x["num_params"]) == npar == int(h["num_params"]),
                "BLOCKER", f"num_params {t,n,r,k,s}")
        src = x["state_source"]
        a.check(src == ("replay" if t in REPLAY else "npz_readonly"),
                "BLOCKER", f"state source {t,n,r,k,s}")
        pairs = (("S1", "heldout_mean_log_score"), ("C_Q", "s2_q_based"),
                 ("S3", "s3_plugin_conditional"), ("q_strict", "q_strict"))
        for mine, theirs in pairs:
            v, hv = f(x[mine]), float(h[theirs])
            a.check(f(x[f"hist_{mine}"]) == hv, "BLOCKER",
                    f"hist_{mine} not the historical value {t,n,r,k,s}")
            if src == "replay":
                a.check(v == hv, "BLOCKER",
                        f"replay {mine} not bit-identical {t,n,r,k,s}")
            else:
                a.check(math.isclose(v, hv, rel_tol=NPZ_RTOL,
                                     abs_tol=NPZ_ATOL), "BLOCKER",
                        f"npz {mine} beyond tolerance {t,n,r,k,s}")
                max_npz[mine] = max(max_npz[mine], abs(v - hv))
        for c in ("retry_count", "replacement_count", "seed_rescue_count"):
            a.check(int(x[c]) == 0, "BLOCKER", f"{c} != 0 {t,n,r,k,s}")
        a.check(x["nan_occurred"] == "False", "BLOCKER", "nan_occurred")
        # C_Q closure and the P_Z identity
        pth = npar * math.log(n)
        a.check(abs(f(x["P_theta"]) - pth) <= CLOSE, "HIGH", "P_theta")
        a.check(abs(f(x["D_K"]) + f(x["P_Z"]) + f(x["P_theta"])
                    - f(x["C_Q"])) <= CLOSE, "HIGH", f"C_Q closure {t,n,r,k,s}")
        a.check(abs(f(x["C_Q"]) - (-2 * f(x["q_strict"]) + pth)) <= CLOSE,
                "HIGH", "C_Q != -2 q_strict + p ln n")
        pz = n * k * (1 + math.log(2 * math.pi))
        a.check(math.isclose(f(x["P_Z"]), pz, rel_tol=1e-9), "MEDIUM",
                f"P_Z != nK(1+ln2pi) at {t,n,r,k,s}")
        # C_Lap from raw Laplace components
        st = x["lap_status"]
        if st == "OK":
            lap = (f(x["lap_phi_at_mode"]) + 0.5 * n * k * math.log(2 * math.pi)
                   - 0.5 * f(x["lap_logdet_H"]))
            a.check(abs(-2 * lap + pth - f(x["C_Lap"])) <= CLOSE, "HIGH",
                    f"C_Lap from components {t,n,r,k,s}")
            a.check(abs(f(x["lap_fit_mode"]) + f(x["lap_volume"])
                        + f(x["lap_penalty"]) - f(x["C_Lap"])) <= CLOSE,
                    "HIGH", f"C_Lap closure {t,n,r,k,s}")
            a.check(f(x["lap_grad_inf"]) <= GRAD_TOL
                    and f(x["lap_min_eig_H"]) > 0
                    and f(x["lap_logdet_sign"]) > 0, "HIGH",
                    f"OK status without stationarity/PD {t,n,r,k,s}")
        else:
            a.check(math.isnan(f(x["C_Lap"])), "HIGH",
                    f"non-OK status carries a value {t,n,r,k,s}")

    # ---- selections --------------------------------------------------------
    sel_map = {(x["criterion"], int(x["k_true"]), int(x["n"]),
                int(x["replicate"])): x for x in sel}
    a.check(len(sel_map) == 4 * len(cells) == len(sel), "BLOCKER",
            "selection row count")
    recount = defaultdict(lambda: defaultdict(lambda: {
        "exact": 0, "under": 0, "over": 0, "incomplete": 0, "selected": {}}))
    for (t, n, r) in cells:
        fits = by_cell[(t, n, r)]
        for crit in LOWER:
            row = sel_map[(crit, t, n, r)]
            complete = (crit != "C_Lap" or all(
                fits[(k, s)]["lap_status"] == "OK" for k in CAND
                for s in STARTS))
            cellc = recount[crit][f"K{t}/n{n}"]
            if not complete:
                a.check(row["status"] == "INCOMPLETE" and row["selected_k"]
                        == "", "BLOCKER", f"incomplete rule {crit,t,n,r}")
                cellc["incomplete"] += 1
                cellc["selected"][r] = None
                continue
            means = {k: (f(fits[(k, 1)][crit]) + f(fits[(k, 2)][crit])) / 2
                     for k in CAND}
            k_sel, ties, margin = select(means, LOWER[crit])
            a.check(row["status"] == "COMPLETE"
                    and int(row["selected_k"]) == k_sel, "BLOCKER",
                    f"selected K {crit,t,n,r}: {row['selected_k']} vs {k_sel}")
            a.check(row["label"] == lab(k_sel, t), "BLOCKER", "label")
            a.check(abs(f(row["margin"]) - margin) <= 1e-9, "MEDIUM",
                    f"margin {crit,t,n,r}")
            a.check(json.loads(row["tie_candidates"]) == ties, "MEDIUM",
                    "ties")
            for s in STARTS:
                ks = select({k: f(fits[(k, s)][crit]) for k in CAND},
                            LOWER[crit])[0]
                a.check(int(row[f"selected_start{s}"]) == ks, "MEDIUM",
                        f"per-start selection {crit,t,n,r,s}")
            if crit in HIST_NAME:
                a.check(k_sel == hist_sel[(HIST_NAME[crit], t, n, r)],
                        "BLOCKER", f"historical selection {crit,t,n,r}")
            cellc[lab(k_sel, t)] += 1
            cellc["selected"][r] = k_sel

    # ---- summary.json recount ---------------------------------------------
    for crit, cellmap in recount.items():
        for cell, c in cellmap.items():
            s = summary[crit]["by_cell"][cell]
            reps = sorted(c["selected"])
            a.check(s["exact"] == c["exact"] and s["under"] == c["under"]
                    and s["over"] == c["over"]
                    and s["incomplete"] == c["incomplete"]
                    and s["selected_k"] == [c["selected"][r] for r in reps],
                    "BLOCKER", f"summary {crit} {cell}")

    sev = defaultdict(int)
    for x in a.findings:
        sev[x["severity"]] += 1
    verdict = "PASS" if not (sev["BLOCKER"] or sev["HIGH"]) else "FAIL"
    totals = {crit: {k: sum(c[k] for c in recount[crit].values())
                     for k in ("exact", "under", "over", "incomplete")}
              for crit in recount}
    return {"auditor": "audit_clap_clean_nsweep.py (artifact-only; imports no "
                       "runner or model code)",
            "stage": stage, "verdict": verdict, "checks": a.checks,
            "severity_counts": dict(sev), "findings": a.findings[:200],
            "fit_rows": len(rows), "cells": len(cells),
            "npz_max_absdiff_recomputed": dict(max_npz),
            "laplace_status_counts": {
                st: sum(x["lap_status"] == st for x in rows)
                for st in sorted({x["lap_status"] for x in rows})},
            "recount_totals": totals}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full", choices=("full", "pilot"))
    ap.add_argument("--write-report", action="store_true")
    args = ap.parse_args(argv)
    rep = run(args.stage)
    text = json.dumps(rep, indent=2, sort_keys=True)
    print(text)
    if args.write_report:
        out = RUN_ROOT / args.stage / "audit_report.json"
        if out.exists():
            raise SystemExit(f"{out} exists; not overwriting")
        out.write_text(text, encoding="utf-8")
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
