"""Independent tally of the convergence-sensitivity run (600 fits).

Written before the run, separately from run_k_exp2_convergence_sensitivity.py
(which it does NOT import). Reads only CSV/JSON outputs, the frozen sensitivity
protocol, and the main run's manifest / per_fit.csv. Re-implements:
  * C_Q(t) = -2 q_after[t] + p ln n,  p = K d - K(K-1)/2 + d [Gaussian X] + [Gaussian Y]
  * selection: per dataset and checkpoint, argmin over K of the mean C_Q of the two
    inits; ties within 1e-9 -> smallest K
and compares with the runner's checkpoint_scores.csv and the pre-specified
analysis outputs.

Usage: python tools/research_audit/tally_k_exp2_convergence_independent.py
           --run RUN_DIR --analysis ANALYSIS_DIR --protocol PROTOCOL_JSON --out OUT_DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "expfam/results/k_selection/k_exp2_generalization_20261125"
N, D = 150, 15
FAMILIES = {"O": ("gaussian", "bernoulli"), "A": ("poisson", "bernoulli"),
            "B": ("gaussian", "poisson"), "C": ("bernoulli", "gaussian")}
CPS = (10, 20, 40)


def rd(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def n_params(k, series):
    fx, fy = FAMILIES[series]
    return k * D - k * (k - 1) // 2 + (D if fx == "gaussian" else 0) + (fy == "gaussian")


def c_q(q, k, series):
    import numpy as np  # the frozen rule multiplies by np.log(n); keep the same float path
    return -2.0 * q + n_params(k, series) * np.log(N)


def pick(bar, tol=1e-9):
    m = min(bar.values())
    return min(k for k, v in bar.items() if v - m <= tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run, ana, out = ROOT / a.run, ROOT / a.analysis, ROOT / a.out
    proto_p = ROOT / a.protocol
    if out.exists():
        raise SystemExit(f"STOP: {out} exists")
    checks = {}
    proto = json.loads(proto_p.read_text(encoding="utf-8"))
    runinfo = json.loads((run / "runinfo.json").read_text(encoding="utf-8"))
    checks["protocol_sha256_matches_runinfo"] = sha(proto_p) == runinfo["protocol_sha256"]
    man = rd(proto_p.parent / "manifest.csv")
    checks["manifest_sha256_matches_protocol"] = \
        sha(proto_p.parent / "manifest.csv") == proto["manifest_sha256"]
    checks["seed_manifest_is_frozen_manifest"] = \
        sha(run / "seed_manifest.csv") == proto["manifest_sha256"]
    main_man = {m["fit_id"]: m for m in rd(MAIN / "protocol_v2" / "manifest.csv")}
    checks["seeds_equal_main_manifest"] = all(
        all(m[c] == main_man[m["fit_id"]][c] for c in ("data_seed", "init_seed", "mc_seed",
                                                         "series", "regime", "k_true", "rep",
                                                         "k", "t"))
        for m in man)
    checks["num_iter_40_all"] = all(m["num_iter"] == "40" for m in man)
    fits = rd(run / "per_fit.csv")
    ids = [r["fit_id"] for r in fits]
    checks["each_planned_fit_recorded_once"] = sorted(ids) == sorted(m["fit_id"] for m in man) \
        and len(ids) == len(set(ids))
    st = Counter(r["status"] for r in fits)
    failures = rd(run / "failures.csv")
    checks["failures_csv_matches_per_fit"] = \
        sorted(f["fit_id"] for f in failures) == sorted(r["fit_id"] for r in fits
                                                        if r["status"] != "OK")
    checks["data_equal_main_all"] = all(r["data_equal_main"] == "True" for r in fits)
    ok = {r["fit_id"]: r for r in fits if r["status"] == "OK"}
    # history
    hist = defaultdict(dict)
    for h in rd(run / "convergence_history.csv"):
        prev = hist[h["fit_id"]].get(int(h["iteration"]))
        if prev is not None and prev != h["q_after"]:
            raise SystemExit("STOP: differing duplicate history rows")
        hist[h["fit_id"]][int(h["iteration"])] = h["q_after"]
    checks["history_40_iterations_per_ok_fit"] = all(
        sorted(hist[f]) == list(range(1, 41)) for f in ok)
    # C_Q recomputation and comparison with checkpoint_scores.csv and main
    main_pf = {r["fit_id"]: r for r in rd(MAIN / "main" / "per_fit.csv")}
    cps = {}
    for c in rd(run / "checkpoint_scores.csv"):
        cps[(c["fit_id"], int(c["checkpoint"]))] = c["C_Q"]
    mism_cp, mism_main, cq = [], [], {}
    for f, r in ok.items():
        k = int(r["k"])
        for t in CPS:
            v = float(c_q(float(hist[f][t]), k, r["series"]))
            cq[(f, t)] = v
            if repr(v) != cps.get((f, t)):
                mism_cp.append((f, t))
        if repr(cq[(f, 10)]) != main_pf[f]["C_Q"]:
            mism_main.append(f)
    checks["C_Q_recomputed_equals_checkpoint_scores_bitwise"] = not mism_cp
    checks["C_Q_at_10_equals_main_per_fit_bitwise"] = not mism_main
    checks["final_Q_equals_q_after_40_all"] = all(
        r["final_q_equals_last_q_after"] == "True" for r in ok.values())
    # selection
    by_ds = defaultdict(dict)
    meta = {}
    for f, r in ok.items():
        for t in CPS:
            by_ds[(r["dataset_id"], t)][(int(r["k"]), int(r["t"]))] = cq[(f, t)]
    for r in fits:
        meta[r["dataset_id"]] = (r["series"], r["regime"], int(r["k_true"]), int(r["rep"]))
    sel_rows = []
    for ds, (s, rg, kt, rep) in sorted(meta.items()):
        for t in CPS:
            d = by_ds.get((ds, t), {})
            if len(d) != 20:
                sel_rows.append(dict(dataset_id=ds, cell=f"{s}-{rg}-{kt}", checkpoint=t,
                                     selected_k="", outcome="INCOMPLETE", init0_k="",
                                     init1_k=""))
                continue
            bar = {k: (d[(k, 0)] + d[(k, 1)]) / 2.0 for k in range(1, 11)}
            sk = pick(bar)
            sel_rows.append(dict(dataset_id=ds, cell=f"{s}-{rg}-{kt}", checkpoint=t,
                                 selected_k=sk,
                                 outcome="exact" if sk == kt else ("under" if sk < kt
                                                                   else "over"),
                                 init0_k=pick({k: d[(k, 0)] for k in range(1, 11)}),
                                 init1_k=pick({k: d[(k, 1)] for k in range(1, 11)})))
    ana_rows = {(r["dataset_id"], int(r["checkpoint"])): r
                for r in rd(ana / "selected_k_by_dataset.csv")}
    diff = [r for r in sel_rows
            if (str(r["selected_k"]), r["outcome"], str(r["init0_k"]), str(r["init1_k"]))
            != tuple(ana_rows[(r["dataset_id"], r["checkpoint"])][c]
                     for c in ("selected_k", "outcome", "init0_k", "init1_k"))]
    checks["selection_equals_prespecified_analysis"] = not diff
    counts = Counter((r["cell"], r["checkpoint"], r["outcome"]) for r in sel_rows)
    summ = rd(ana / "summary_by_cell_checkpoint.csv")
    checks["cell_counts_equal_prespecified_summary"] = all(
        int(s[o]) == counts[(s["cell"], int(s["checkpoint"]), o)]
        for s in summ for o in ("exact", "under", "over"))
    ties = 0
    for (ds, t), d in by_ds.items():
        if len(d) == 20:
            v = sorted((d[(k, 0)] + d[(k, 1)]) / 2.0 for k in range(1, 11))
            ties += (v[1] - v[0]) <= 1e-9
    checks["no_ties_within_1e-9"] = ties == 0
    # states
    sm = {r["state_file"]: r["sha256"] for r in rd(run / "states_sha256_manifest.csv")}
    checks["state_sha256_manifest_matches_per_fit"] = all(
        sm.get(r["state_file"]) == r["state_sha256"] for r in ok.values())
    checks["states_present_on_disk_hash_ok"] = all(
        (run / "states" / r["state_file"]).exists()
        and sha(run / "states" / r["state_file"]) == r["state_sha256"] for r in ok.values())
    out.mkdir(parents=True)
    with open(out / "independent_selection.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sel_rows[0]))
        w.writeheader()
        w.writerows(sel_rows)
    cell_counts = sorted({(c, t) for c, t, _ in counts})
    with open(out / "independent_cell_counts.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "checkpoint", "exact", "under", "over", "incomplete"])
        for c, t in cell_counts:
            w.writerow([c, t, *(counts[(c, t, o)] for o in ("exact", "under", "over",
                                                            "INCOMPLETE"))])
    res = {"checks": checks, "all_ok": all(checks.values()), "status_counts": dict(st),
           "n_fits": len(fits), "mismatch_checkpoint_scores": mism_cp[:20],
           "mismatch_main_c_q": mism_main[:20], "selection_diff": diff[:20]}
    (out / "tally_checks.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
