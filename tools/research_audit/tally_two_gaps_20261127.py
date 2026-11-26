"""Independent tally for the two-gaps experiments (A and B), 2026-11-27.

Does NOT import the runner.  Recomputes, from per_fit.csv only (plus files on disk):
  A: per-dataset 2-init means, Delta, I and their summaries; compares with analysis CSVs.
  B: selections (argmin of 2-init mean, tie 1e-9 -> smaller K; C_Lap INCOMPLETE rule),
     outcome counts per condition; shared Z/F/Y hashes; compares with analysis CSVs.
  Both: fit counts, statuses, C_Q = -2Q + p ln n, NPZ sha256, protected-file sha256.
Usage: python tools/research_audit/tally_two_gaps_20261127.py
"""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "expfam/results/k_selection/two_gaps_20261127"
OUT = BASE / "independent_tally"
checks = []


def rd(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fl(s):
    s = str(s).strip()
    return float(s) if s else float("nan")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check(name, ok, detail=""):
    checks.append({"check": name, "pass": bool(ok), "detail": str(detail)[:400]})


def common(exp, run, proto, n_expected):
    rows = [r for r in rd(run / "per_fit.csv") if r["phase"] == "main"]
    ids = [r["fit_id"] for r in rows]
    check(f"{exp}: fit count = manifest", len(rows) == n_expected and len(set(ids)) == len(ids),
          f"{len(rows)} rows, {len(set(ids))} unique, expected {n_expected}")
    man = {r["fit_id"] for r in rd(proto.parent / "manifest.csv")}
    check(f"{exp}: fit ids = frozen manifest", set(ids) == man)
    ok = [r for r in rows if r["status"] == "OK"]
    check(f"{exp}: no retry/replacement", all(r["retry_count"] == "0" and r["replacement_count"] == "0" for r in ok))
    bad = [r["fit_id"] for r in ok if abs(-2 * fl(r["Q_strict"]) + int(r["num_params"]) * math.log(150) - fl(r["C_Q"])) > 1e-6]
    check(f"{exp}: C_Q = -2Q + p ln n", not bad, bad[:5])
    miss = []
    for r in ok:
        p = run / "states" / r["state_file"]
        if not p.exists() or sha(p) != r["state_sha256"]:
            miss.append(r["fit_id"])
    check(f"{exp}: NPZ sha256", not miss, miss[:5])
    return rows


def tally_A():
    run, proto = BASE / "expA/run", BASE / "expA/protocol/protocol.json"
    rows = common("A", run, proto, 160)
    cell = defaultdict(list)
    for r in rows:
        if r["status"] == "OK":
            cell[(r["dataset_id"], r["rate"], r["family_x_fit"])].append(fl(r["rmse_Z_last"]))
    # same data / test mask across the 4 conditions of a dataset; sparse subset sizes
    by = defaultdict(list)
    for r in rows:
        by[r["dataset_id"]].append(r)
    same = all(len({(r["x_hash"], r["y_hash"], r["test_hash"]) for r in v}) == 1 for v in by.values())
    check("A: same X, Y, test mask within dataset", same)
    seeds = all(len({(r["init_seed"], r["mc_seed"]) for r in v if r["t"] == t}) == 1 for v in by.values() for t in ("0", "1"))
    check("A: same init/mc seed across rate x family (paired)", seeds)
    I, dd, dsp = [], [], []
    for ds in sorted(by):
        try:
            m = {(rt, fa): float(np.mean(cell[(ds, rt, fa)])) for rt in ("dense", "sparse") for fa in ("bernoulli", "poisson")
                 if len(cell[(ds, rt, fa)]) == 2}
            d_dense = m[("dense", "poisson")] - m[("dense", "bernoulli")]
            d_sparse = m[("sparse", "poisson")] - m[("sparse", "bernoulli")]
        except KeyError:
            continue
        dd.append(d_dense)
        dsp.append(d_sparse)
        I.append(d_sparse - d_dense)
    pr = {r["column"]: r for r in rd(BASE / "expA/analysis/primary_results.csv")}
    for col, v in (("I_rmse_Z_last", I), ("delta_rmse_Z_last_dense", dd), ("delta_rmse_Z_last_sparse", dsp)):
        a = pr[col]
        ok = (int(a["n"]) == len(v) and abs(float(a["mean"]) - np.mean(v)) < 1e-12
              and abs(float(a["median"]) - np.median(v)) < 1e-12 and int(a["n_pos"]) == sum(x > 0 for x in v))
        check(f"A: {col} (n, mean, median, n_pos)", ok,
              f"tally n={len(v)} mean={np.mean(v):.6g} median={np.median(v):.6g} pos={sum(x > 0 for x in v)}")
    return {"I_mean": float(np.mean(I)), "I_median": float(np.median(I)), "I_pos": int(sum(x > 0 for x in I)),
            "n": len(I), "delta_dense_mean": float(np.mean(dd)), "delta_sparse_mean": float(np.mean(dsp))}


def sel(d):
    m = min(d.values())
    return min(k for k, v in d.items() if v <= m + 1e-9)


def tally_B():
    run, proto = BASE / "expB/run", BASE / "expB/protocol/protocol.json"
    rows = common("B", run, proto, 600)
    rep = defaultdict(list)
    for r in rows:
        rep[r["replicate_id"]].append(r)
    shared = all(len({(r["z_hash"], r["f_hash"], r["y_hash"]) for r in v}) == 1 for v in rep.values())
    check("B: Z, F, Y shared across the 3 conditions", shared)
    xdiff = all(len({r["x_hash"] for r in v}) == 3 for v in rep.values())
    check("B: X differs between the 3 conditions", xdiff)
    by = defaultdict(list)
    for r in rows:
        by[r["dataset_id"]].append(r)
    out = {}
    counts = defaultdict(lambda: defaultdict(int))
    for ds, v in by.items():
        cond = v[0]["cond"]
        for crit, col, st in (("C_Q", "C_Q", None), ("C_Lap", "C_Lap", "lap_status")):
            okrows = [r for r in v if r["status"] == "OK" and (st is None or r[st] == "OK")]
            if len(okrows) != 20:
                out[(ds, crit)] = None
                counts[(cond, crit)]["INCOMPLETE"] += 1
                continue
            acc = defaultdict(list)
            for r in okrows:
                acc[int(r["k"])].append(fl(r[col]))
            k = sel({kk: float(np.mean(x)) for kk, x in acc.items()})
            out[(ds, crit)] = k
            counts[(cond, crit)]["exact" if k == 9 else ("under" if k < 9 else "over")] += 1
    pr = rd(BASE / "expB/analysis/primary_results.csv")
    for a in pr:
        c = counts[(a["cond"], a["criterion"])]
        ok = all(int(a[o]) == c[o] for o in ("exact", "under", "over", "INCOMPLETE"))
        check(f"B: {a['cond']} {a['criterion']} outcome counts", ok, dict(c))
    an = {r["dataset_id"]: r for r in rd(BASE / "expB/analysis/per_dataset_selection.csv")}
    mism = [(ds, crit) for (ds, crit), k in out.items()
            if str(k if k is not None else "") != an[ds]["k_" + ("Q" if crit == "C_Q" else "Lap")]]
    check("B: per-dataset selected K matches analysis", not mism, mism[:5])
    return {f"{c}|{k}": dict(v) for (c, k), v in counts.items()}


def main():
    OUT.mkdir(exist_ok=True)
    res = {}
    if (BASE / "expA/analysis/primary_results.csv").exists():
        res["A"] = tally_A()
    if (BASE / "expB/analysis/primary_results.csv").exists():
        res["B"] = tally_B()
    proto = json.loads((BASE / "expA/protocol/protocol.json").read_text(encoding="utf-8"))
    changed = [f for f, h in proto["protected_sha256"].items() if sha(ROOT / f) != h]
    check("earlier results unchanged (protected sha256)", not changed, changed)
    with open(OUT / "independent_tally.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "pass", "detail"])
        w.writeheader()
        w.writerows(checks)
    (OUT / "tally_summary.json").write_text(json.dumps({"results": res, "n_checks": len(checks),
                                                       "n_pass": sum(c["pass"] for c in checks)},
                                                      indent=2, default=str), encoding="utf-8")
    print(f"{sum(c['pass'] for c in checks)}/{len(checks)} checks pass")
    for c in checks:
        if not c["pass"]:
            print("FAIL", c)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
