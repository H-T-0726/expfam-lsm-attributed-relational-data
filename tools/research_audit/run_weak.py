"""Runner for weak_protocol.py (stage W: OLD and lineage E on the PR #160 E2 weak n=75 data, all pairs).

Data: run_fair_comparison.e2_data(75, rep, 'weak', 'gaussian') with the training mask replaced by all
pairs; X / Y hashes must equal the PR #160 E2 fit JSONs (STOP otherwise).  Estimators: the PR #160
adapters run_fair_comparison.fit_old / fit_E (unchanged).  Before each fit the initial state of BOTH
estimators is rebuilt from the seed (run_n75.init_state, no fitting) and hashed; any difference STOPs.
Each fit is saved atomically (npz, then JSON with the npz sha256, written last); valid fits are skipped
on resume; FAILED fits are never retried.  One process, BLAS threads 1, memory guard 3 GB.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "reproduction/src", ROOT / "expfam/src/experimental", ROOT / "tools/research_audit"):
    sys.path.insert(0, str(p))

import weak_protocol as wp  # noqa: E402
import run_fair_comparison as rf  # noqa: E402
import run_n75 as rn  # noqa: E402

RESULTS = ROOT / wp.RESULTS_REL
STATES = Path(wp.STATE_ROOT)
P160_FITS = ROOT / wp.PR160_REL / "e2" / "fits"
N = wp.COMMON["n"]


def need(c, m):
    if not c:
        raise SystemExit("STOP: " + m)


def p160_key(rep, k, s):
    return f"e2_gaussian_weak_n75_r{rep}_k{k}_s{s}"


def weak_data(rep, mask="all_pairs"):
    d = rf.e2_data(N, rep, "weak", "gaussian")
    xs, ys = rf.sha_arr(d["X"]), rf.sha_arr(d["Y"])
    for k in wp.COMMON["K_CAND"]:
        for s in wp.COMMON["starts"]:
            j = json.loads((P160_FITS / f"{p160_key(rep, k, s)}.json").read_text(encoding="utf-8"))
            need(j["x_sha256"] == xs and j["y_sha256"] == ys and j["data_seed"] == d["data_seed"],
                 f"PR #160 weak data not reproduced for rep {rep}")
            if mask == "mask_80":
                need(j["train_sha256"] == rf.sha_arr(d["train"]), f"PR #160 train mask not reproduced for rep {rep}")
    if mask == "all_pairs":
        d["train"], d["test"] = rf.all_pairs(N), None
        d["note"] += "; all pairs observed (PR #155 test pairs used as training pairs, no held-out)"
    else:
        d["note"] += "; PR #155 80% training mask, 20% held-out"
    return d


def stage_fits(stage="W"):
    out = []
    for rep in wp.COMMON["replicates"]:
        for k in wp.COMMON["K_CAND"]:
            for s in wp.COMMON["starts"]:
                for est in wp.STAGES[stage]["estimators"]:
                    out.append(dict(stage=stage, key=f"{stage}_r{rep}_k{k}_s{s}_{est}", rep=rep, k=k, s=s, estimator=est,
                                    mask="all_pairs", seed=rf.hist.model_seed(5, N, rep, k, s)))
    return out


def valid(f) -> bool:
    j = RESULTS / f["stage"] / "fits" / f"{f['key']}.json"
    if not j.exists():
        return False
    row = json.loads(j.read_text(encoding="utf-8"))
    if row.get("status") != "OK":
        return row.get("status") == "FAILED"
    npz = STATES / f["stage"] / f"{f['key']}.npz"
    return npz.exists() and rf.sha_file(npz) == row.get("npz_sha256")


def init_pair(Dd, k, seed):
    hO = rn.init_hash(rn.init_state("OLD", Dd["X"], Dd["Y"], Dd["train"], k, seed))
    hE = rn.init_hash(rn.init_state("E", Dd["X"], Dd["Y"], Dd["train"], k, seed))
    return hO, hE


def run_stage(stage="W"):
    fits = stage_fits(stage)
    need(len(fits) == wp.FIT_BUDGET["total_max"], "fit list size != budget")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "tools", "expfam/src", "reproduction/src"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    need(not dirty, "code tree dirty: commit before running")
    code_sha = {p: rf.sha_file(ROOT / "tools/research_audit" / p)
                for p in ("run_weak.py", "weak_protocol.py", "run_fair_comparison.py", "run_n75.py")}
    log = RESULTS / stage / "run_log.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    done = 0
    with log.open("a", encoding="utf-8") as lg:
        lg.write(f"{time.strftime('%H:%M:%S')} start {stage} fits={len(fits)} head={head} protocol={wp.protocol_hash()[:12]} "
                 f"code={ {k: v[:12] for k, v in code_sha.items()} }\n")
        for f in fits:
            if valid(f):
                continue
            waited = 0
            while rf.avail_gb() < wp.COMMON["mem_min_gb"]:
                if waited > 600:
                    lg.write(f"{time.strftime('%H:%M:%S')} STOP low memory\n")
                    return
                time.sleep(30)
                waited += 30
            if f["rep"] not in cache:
                cache = {f["rep"]: weak_data(f["rep"])}
            Dd = cache[f["rep"]]
            hO, hE = init_pair(Dd, f["k"], f["seed"])
            need(all(hO[k] == hE[k] for k in hO), f"initial states differ for {f['key']}")
            row = dict(f, protocol_hash=wp.protocol_hash(), git_head=head, code_sha=code_sha, data_seed=Dd["data_seed"],
                       x_sha256=rf.sha_arr(Dd["X"]), y_sha256=rf.sha_arr(Dd["Y"]), train_sha256=rf.sha_arr(Dd["train"]),
                       note=Dd["note"], **hO, init_identical_OLD_E=True,
                       env=dict(python=sys.version.split()[0], numpy=np.__version__, threads=os.environ["OMP_NUM_THREADS"]))
            t0 = time.perf_counter()
            try:
                if f["estimator"] == "OLD":
                    st = rf.fit_old(Dd["X"], Dd["Y"], f["k"], f["seed"], L=wp.COMMON["L"], iters=wp.COMMON["em_iterations"])
                else:
                    st = rf.fit_E(Dd["X"], Dd["Y"], Dd["train"], f["k"], f["seed"], "gaussian")
                row["npz_sha256"] = rf.save_npz_atomic(STATES / stage / f"{f['key']}.npz", st)
                row["status"], row["exception"] = "OK", ""
            except (FloatingPointError, SystemExit, Exception) as e:
                row["status"], row["exception"], row["npz_sha256"] = "FAILED", f"{type(e).__name__}: {e}"[:300], ""
            row["runtime_s"] = round(time.perf_counter() - t0, 2)
            row["avail_gb"] = round(rf.avail_gb(), 2)
            row["rss_mb"] = round(psutil.Process().memory_info().rss / 1e6, 1)
            rf.atomic_write_bytes(RESULTS / stage / "fits" / f"{f['key']}.json",
                                  json.dumps(row, indent=1, default=str).encode("utf-8"))
            lg.write(f"{time.strftime('%H:%M:%S')} {row['status']} {f['key']} {row['runtime_s']}s avail={row['avail_gb']}GB\n")
            lg.flush()
            done += 1
        lg.write(f"{time.strftime('%H:%M:%S')} end {stage} new={done}\n")


def precheck():
    """No fit.  Data identity with PR #160 + initial-state hashes for every planned fit and for the
    PR #160 80% fits of the same seed -> dataset_and_initialization_hashes.csv."""
    import pandas as pd
    out = RESULTS / "dataset_and_initialization_hashes.csv"
    need(not out.exists(), f"refusing to overwrite {out}")
    rows = []
    for rep in wp.COMMON["replicates"]:
        Da, D8 = weak_data(rep, "all_pairs"), weak_data(rep, "mask_80")
        need(np.allclose(Da["Y"], Da["Y"].T) and np.all(np.diag(Da["Y"]) == 0), "Y not symmetric / diag")
        need(not np.any(D8["train"] & D8["test"]), "PR #160 mask overlaps its held-out pairs")
        iu = np.triu_indices(N, 1)
        for k in wp.COMMON["K_CAND"]:
            for s in wp.COMMON["starts"]:
                seed = rf.hist.model_seed(5, N, rep, k, s)
                j = json.loads((P160_FITS / f"{p160_key(rep, k, s)}.json").read_text(encoding="utf-8"))
                need(j["seed"] == seed, "seed rule differs from PR #160")
                hO, hE = init_pair(Da, k, seed)
                h8 = rn.init_hash(rn.init_state("E", D8["X"], D8["Y"], D8["train"], k, seed))
                r = dict(rep=rep, k=k, s=s, seed=seed, pr160_key=p160_key(rep, k, s), data_seed=Da["data_seed"],
                         x_sha256=rf.sha_arr(Da["X"]), y_sha256=rf.sha_arr(Da["Y"]), x_y_equal_pr160=True,
                         train_sha256_allpairs=rf.sha_arr(Da["train"]), train_sha256_pr160=j["train_sha256"],
                         n_pairs_allpairs=int(np.triu(Da["train"], 1).sum()), n_pairs_pr160_train=int(np.triu(D8["train"], 1).sum()),
                         y_density_all=float(Da["Y"][iu].mean()), y_density_pr160_train=float(D8["Y"][np.triu(D8["train"], 1)].mean()),
                         w_true=float(Da["w"]), w0_true=float(Da["w0"]), F_row_norm2_mean=float(np.mean(np.sum(Da["F"] ** 2, 1))),
                         x_noise_var_min=float(Da["gaussian_noise_var"].min()), x_noise_var_max=float(Da["gaussian_noise_var"].max()))
                for kk in hO:
                    r[f"{kk}_OLD_all"] = hO[kk][:16]
                    r[f"{kk}_E_all"] = hE[kk][:16]
                    r[f"{kk}_E_80"] = h8[kk][:16]
                    r[f"{kk}_OLD_eq_E_all"] = hO[kk] == hE[kk]
                    r[f"{kk}_E_all_eq_E_80"] = hE[kk] == h8[kk]
                rows.append(r)
    df = pd.DataFrame(rows)
    eq = [c for c in df if c.endswith("_OLD_eq_E_all")]
    need(df[eq].all().all(), "initial states of OLD and E differ")
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n")
    e8 = [c for c in df if c.endswith("_E_all_eq_E_80")]
    print(len(df), "seeds x reps; OLD == E (all pairs):", bool(df[eq].all().all()),
          "| E all vs E 80% equal per component:", {c.replace("_E_all_eq_E_80", ""): int(df[c].sum()) for c in e8})


def status(stage="W"):
    c = {"OK": 0, "FAILED": 0, "NOT_RUN": 0}
    for f in stage_fits(stage):
        j = RESULTS / stage / "fits" / f"{f['key']}.json"
        if not j.exists() or not valid(f):
            c["NOT_RUN"] += 1
            continue
        c[json.loads(j.read_text(encoding="utf-8"))["status"]] += 1
    print(stage, len(stage_fits(stage)), c)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status", "precheck"])
    a = ap.parse_args()
    {"precheck": precheck, "run": run_stage, "status": status}[a.cmd]()
