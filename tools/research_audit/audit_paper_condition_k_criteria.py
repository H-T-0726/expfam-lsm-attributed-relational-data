"""Independent artifact-only auditor for Issue #128 experiment B.

Imports neither the runner nor any model / generator code.  Every constant is
restated here as a literal, and every derived quantity (seeds, parameter
counts, protocol hash, criterion closures, C_Lap from its raw Laplace
components, the C_cond identities, selections, labels, margins, per-start
selections, increment decomposition, per-K_TRUE counts) is recomputed from the
CSV / JSON values alone.

Usage:
    python audit_paper_condition_k_criteria.py [--stage full|pilot]
                                               [--write-report]
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
RUN_ROOT = ROOT / "expfam/results/k_selection/paper_condition_k_criteria_20261006"
GENERATOR = ROOT / "reproduction/src/data_generator.py"

# --- independent literals (Issue #128 plan §4) -----------------------------
N = 150
D = 15
K_TRUE = (1, 3, 5, 7, 9)
CAND = tuple(range(1, 11))
STARTS = (1, 2)
REPS = (1, 2, 3, 4, 5)
PILOT_CELLS = ((3, 1),)
L_SAMPLES = 5
NUM_ITER = 8
TIE = 1e-12
CLOSE = 1e-6
GRAD_TOL = 1e-8
CRITERIA = ("C_Q", "C_Lap", "C_cond_MC", "C_cond_last", "C_cond_mode")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(x) -> float:
    return float(x) if x not in ("", None) else float("nan")


class Audit:
    def __init__(self) -> None:
        self.findings: list[dict[str, str]] = []
        self.checks = 0

    def check(self, cond: bool, severity: str, msg: str) -> None:
        self.checks += 1
        if not cond:
            self.findings.append({"severity": severity, "finding": msg})


def select(values: dict[int, float]):
    """Minimum; ties within TIE -> smallest K; margin to the runner-up."""
    s = {k: -v for k, v in values.items()}
    best = max(s.values())
    ties = sorted(k for k, v in s.items() if best - v <= TIE)
    k = min(ties)
    rest = sorted((v for kk, v in s.items() if kk != k), reverse=True)
    return k, ties, best - rest[0], -best


def lab(k: int, t: int) -> str:
    return "exact" if k == t else ("under" if k < t else "over")


def data_seed(t: int, r: int) -> int:
    return 1980 + 100 * t + 1000 * (r - 1)


def model_seed(t: int, r: int, k: int, s: int) -> int:
    return 2_000_000 + 100_000 * t + 1000 * r + 10 * k + s


def npar(k: int) -> int:
    return (k + 1) * D - k * (k - 1) // 2


def run(stage: str) -> dict:
    a = Audit()
    d = RUN_ROOT / stage
    rows = read(d / "fit_criteria.csv")
    sel = read(d / "selection.csv")
    dec = read(d / "increment_decomposition.csv")
    prov = read(d / "dataset_provenance.csv")
    curves = read(d / "criterion_curves.csv")
    summary = json.loads((d / "summary.json").read_text("utf-8"))
    info = json.loads((d / "runinfo.json").read_text("utf-8"))
    proto = json.loads((d / "protocol.json").read_text("utf-8"))

    cells = (list(PILOT_CELLS) if stage == "pilot"
             else [(t, r) for t in K_TRUE for r in REPS])

    # ---- protocol ------------------------------------------------------------
    p = proto["protocol"]
    a.check(hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
            == proto["protocol_hash"] == info["protocol_hash"], "BLOCKER",
            "protocol hash does not recompute / differs from runinfo")
    for key, val in (("n", N), ("d", D), ("k_true_grid", list(K_TRUE)),
                     ("candidate_k", list(CAND)), ("starts", list(STARTS)),
                     ("replicates", list(REPS)), ("L", L_SAMPLES),
                     ("num_iter", NUM_ITER), ("family_x", "gaussian"),
                     ("family_y", "bernoulli"), ("heldout", False),
                     ("numerics_mode", "consistent"),
                     ("is_paper_experiment2_reproduction", False)):
        a.check(p.get(key) == val, "BLOCKER", f"protocol {key} = {p.get(key)}")
    a.check(p.get("generator_kwargs") == {"var_f": 5.0, "uniq": 0.1,
                                          "w_scale": 3.0},
            "BLOCKER", "generator kwargs")
    a.check(info.get("cells") == [list(c) for c in cells], "BLOCKER",
            "runinfo cells != expected cells")

    # ---- run bookkeeping -----------------------------------------------------
    expected = {(t, r, k, s) for (t, r) in cells for k in CAND for s in STARTS}
    keys = [(int(x["k_true"]), int(x["replicate"]), int(x["k_est"]),
             int(x["start"])) for x in rows]
    a.check(len(keys) == len(set(keys)), "BLOCKER", "duplicate fit rows")
    a.check(set(keys) == expected, "BLOCKER",
            f"fit key set != expected ({len(keys)} vs {len(expected)})")
    a.check(info.get("status") == "COMPLETED", "BLOCKER", "runinfo not COMPLETED")
    a.check(info.get("attempted_fits") == len(expected)
            == info.get("completed_rows"), "BLOCKER", "runinfo fit counts")
    for key in ("retry_count", "replacement_fits_executed",
                "seed_rescue_count", "tolerance_relaxations"):
        a.check(info.get(key) == 0, "BLOCKER", f"runinfo {key} != 0")
    a.check(info.get("resumed") is False, "BLOCKER", "run was resumed")
    a.check(info.get("working_tree_clean_before_execution") is True, "BLOCKER",
            "working tree not clean before execution")
    a.check(not (d / "failure.json").exists(), "BLOCKER", "failure.json present")
    a.check(not (d / "fit_criteria_partial.csv").exists(), "BLOCKER",
            "partial CSV present")
    a.check(info.get("failed_fits") == sum(x["fit_status"] != "OK"
                                           for x in rows),
            "BLOCKER", "runinfo failed_fits")
    lap_counts = defaultdict(int)
    for x in rows:
        lap_counts[x["lap_status"]] += 1
    a.check(info.get("laplace_status_counts") == dict(lap_counts), "HIGH",
            "runinfo laplace_status_counts")
    sp = Path(info["states_file"])
    a.check(sp.is_file() and sha(sp) == info["states_sha256"], "HIGH",
            "states archive missing or sha256 differs")
    a.check(sha(GENERATOR) == info.get("generator_sha256"), "BLOCKER",
            "generator file changed since the run")

    # ---- dataset provenance -------------------------------------------------
    pmap = {(int(x["k_true"]), int(x["replicate"])): x for x in prov}
    a.check(set(pmap) == set(cells) and len(prov) == len(cells), "BLOCKER",
            "dataset provenance cells")
    for (t, r), x in pmap.items():
        a.check(int(x["data_seed"]) == data_seed(t, r), "BLOCKER",
                f"provenance data seed {t,r}")
        dens = f(x["y_density"])
        a.check(0.0 < dens < 1.0, "MEDIUM", f"degenerate Y density {t,r}")
    a.check(len({x["x_sha256"] for x in prov}) == len(prov)
            and len({x["y_sha256"] for x in prov}) == len(prov), "HIGH",
            "duplicate datasets")

    # ---- per-row checks -----------------------------------------------------
    by_cell = defaultdict(dict)
    for x in rows:
        t, r, k, s = (int(x["k_true"]), int(x["replicate"]), int(x["k_est"]),
                      int(x["start"]))
        tag = (t, r, k, s)
        by_cell[(t, r)][(k, s)] = x
        a.check(int(x["data_seed"]) == data_seed(t, r), "BLOCKER",
                f"data seed {tag}")
        a.check(int(x["model_seed"]) == model_seed(t, r, k, s), "BLOCKER",
                f"model seed {tag}")
        a.check(int(x["num_params"]) == npar(k), "BLOCKER", f"num_params {tag}")
        for c in ("retry_count", "replacement_count", "seed_rescue_count"):
            a.check(x.get(c, "0") in ("0", ""), "BLOCKER", f"{c} != 0 {tag}")
        if x["fit_status"] != "OK":
            continue
        pth = npar(k) * math.log(N)
        a.check(abs(f(x["P_theta"]) - pth) <= CLOSE, "HIGH", f"P_theta {tag}")
        a.check(abs(f(x["D_K"]) + f(x["P_Z"]) + f(x["P_theta"])
                    - f(x["C_Q"])) <= CLOSE, "HIGH", f"C_Q closure {tag}")
        a.check(abs(f(x["C_Q"]) - (-2 * f(x["q_strict"]) + pth)) <= CLOSE,
                "HIGH", f"C_Q != -2 q_strict + p ln n {tag}")
        a.check(abs(f(x["D_K"]) + 2 * (f(x["Q_X"]) + f(x["Q_Y"]))) <= CLOSE
                and abs(f(x["P_Z"]) + 2 * f(x["Q_Z"])) <= CLOSE, "HIGH",
                f"D_K / P_Z from Q terms {tag}")
        # C_cond identities (plan §2 DERIVED)
        a.check(abs(f(x["C_cond_MC"]) - (f(x["D_K"]) + pth)) <= CLOSE, "HIGH",
                f"C_cond_MC != D_K + P_theta {tag}")
        for nm, ll in (("C_cond_MC", "cond_ll_MC"),
                       ("C_cond_last", "cond_ll_last")):
            a.check(abs(f(x[nm]) - (-2 * f(x[ll]) + pth)) <= CLOSE, "HIGH",
                    f"{nm} != -2 ll + p ln n {tag}")
        st = x["lap_status"]
        if st == "OK":
            lap = (f(x["lap_phi_at_mode"]) + 0.5 * N * k * math.log(2 * math.pi)
                   - 0.5 * f(x["lap_logdet_H"]))
            a.check(abs(-2 * lap + pth - f(x["C_Lap"])) <= CLOSE, "HIGH",
                    f"C_Lap from components {tag}")
            a.check(abs(f(x["lap_fit_mode"]) + f(x["lap_volume"])
                        + f(x["lap_penalty"]) - f(x["C_Lap"])) <= CLOSE,
                    "HIGH", f"C_Lap closure {tag}")
            a.check(f(x["lap_grad_inf"]) <= GRAD_TOL
                    and f(x["lap_min_eig_H"]) > 0
                    and f(x["lap_logdet_sign"]) > 0, "HIGH",
                    f"OK status without stationarity/PD {tag}")
            a.check(abs(f(x["C_cond_mode"]) - (f(x["lap_fit_mode"]) + pth))
                    <= CLOSE, "HIGH", f"C_cond_mode != fit_mode + P_theta {tag}")
        else:
            a.check(math.isnan(f(x["C_Lap"])) and math.isnan(f(x["C_cond_mode"])),
                    "HIGH", f"non-OK Laplace status carries a value {tag}")

    # ---- selections ---------------------------------------------------------
    sel_map = {(x["criterion"], int(x["k_true"]), int(x["replicate"])): x
               for x in sel}
    a.check(len(sel_map) == len(CRITERIA) * len(cells) == len(sel), "BLOCKER",
            "selection row count")
    recount = defaultdict(lambda: defaultdict(lambda: {
        "exact": 0, "under": 0, "over": 0, "incomplete": 0, "selected": {}}))
    curve_map = {(x["criterion"], int(x["k_true"]), int(x["replicate"]),
                  int(x["k_est"])): x for x in curves}
    for (t, r) in cells:
        fits = by_cell[(t, r)]
        failed = any(x["fit_status"] != "OK" for x in fits.values())
        for crit in CRITERIA:
            row = sel_map[(crit, t, r)]
            c = recount[crit][f"K{t}"]
            complete = (not failed) and all(
                math.isfinite(f(fits[(k, s)][crit])) for k in CAND
                for s in STARTS)
            if not failed:
                for k in CAND:
                    cv = curve_map.get((crit, t, r, k))
                    m = (f(fits[(k, 1)][crit]) + f(fits[(k, 2)][crit])) / 2
                    a.check(cv is not None and (
                        (math.isnan(m) and math.isnan(f(cv["mean_value"])))
                        or abs(f(cv["mean_value"]) - m) <= 1e-9), "MEDIUM",
                        f"curve mean {crit,t,r,k}")
            if not complete:
                a.check(row["status"] == "INCOMPLETE" and row["selected_k"] == "",
                        "BLOCKER", f"incomplete rule {crit,t,r}")
                c["incomplete"] += 1
                c["selected"][r] = None
                continue
            means = {k: (f(fits[(k, 1)][crit]) + f(fits[(k, 2)][crit])) / 2
                     for k in CAND}
            k_sel, ties, margin, best = select(means)
            a.check(row["status"] == "COMPLETE"
                    and int(row["selected_k"]) == k_sel, "BLOCKER",
                    f"selected K {crit,t,r}: {row['selected_k']} vs {k_sel}")
            a.check(row["label"] == lab(k_sel, t), "BLOCKER",
                    f"label {crit,t,r}")
            a.check(abs(f(row["margin"]) - margin) <= 1e-9, "MEDIUM",
                    f"margin {crit,t,r}")
            a.check(abs(f(row["best_value"]) - best) <= 1e-9, "MEDIUM",
                    f"best value {crit,t,r}")
            a.check(json.loads(row["tie_candidates"]) == ties, "MEDIUM",
                    f"ties {crit,t,r}")
            for s in STARTS:
                ks = select({k: f(fits[(k, s)][crit]) for k in CAND})[0]
                a.check(int(row[f"selected_start{s}"]) == ks, "MEDIUM",
                        f"per-start selection {crit,t,r,s}")
            c[lab(k_sel, t)] += 1
            c["selected"][r] = k_sel

    # ---- increment decomposition --------------------------------------------
    dmap = {(int(x["k_true"]), int(x["replicate"]), int(x["k_from"])): x
            for x in dec}
    n_dec = 0
    for (t, r) in cells:
        fits = by_cell[(t, r)]
        if any(x["fit_status"] != "OK" for x in fits.values()):
            continue

        def m(k, col):
            return (f(fits[(k, 1)][col]) + f(fits[(k, 2)][col])) / 2
        for k in CAND[:-1]:
            n_dec += 1
            x = dmap.get((t, r, k))
            if x is None:
                a.check(False, "HIGH", f"missing decomposition {t,r,k}")
                continue
            gain = -(m(k + 1, "D_K") - m(k, "D_K"))
            dpz = m(k + 1, "P_Z") - m(k, "P_Z")
            dpt = m(k + 1, "P_theta") - m(k, "P_theta")
            a.check(abs(f(x["CQ_fit_gain"]) - gain) <= 1e-6
                    and abs(f(x["CQ_dPZ"]) - dpz) <= 1e-6
                    and abs(f(x["CQ_dPtheta"]) - dpt) <= 1e-6
                    and abs(f(x["CQ_slack"]) - (gain - dpz - dpt)) <= 1e-6,
                    "MEDIUM", f"C_Q decomposition {t,r,k}")
            lap_ok = all(fits[(kk, s)]["lap_status"] == "OK"
                         for kk in (k, k + 1) for s in STARTS)
            a.check((x["Lap_complete"] == "True") == lap_ok, "MEDIUM",
                    f"Lap_complete flag {t,r,k}")
            if lap_ok:
                lg = -(m(k + 1, "lap_fit_mode") - m(k, "lap_fit_mode"))
                dv = m(k + 1, "lap_volume") - m(k, "lap_volume")
                dp = m(k + 1, "lap_penalty") - m(k, "lap_penalty")
                a.check(abs(f(x["Lap_fit_gain"]) - lg) <= 1e-6
                        and abs(f(x["Lap_dvolume"]) - dv) <= 1e-6
                        and abs(f(x["Lap_slack"]) - (lg - dv - dp)) <= 1e-6,
                        "MEDIUM", f"C_Lap decomposition {t,r,k}")
    a.check(len(dec) == n_dec, "MEDIUM", "decomposition row count")

    # ---- summary.json recount ----------------------------------------------
    for crit in CRITERIA:
        for cell, c in recount[crit].items():
            s = summary[crit]["by_k_true"][cell]
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
              for crit in CRITERIA}
    return {"auditor": "audit_paper_condition_k_criteria.py (artifact-only; "
                       "imports no runner, model or generator code)",
            "stage": stage, "verdict": verdict, "checks": a.checks,
            "severity_counts": dict(sev), "findings": a.findings[:200],
            "fit_rows": len(rows), "cells": len(cells),
            "failed_fits": sum(x["fit_status"] != "OK" for x in rows),
            "laplace_status_counts": dict(sorted(lap_counts.items())),
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
