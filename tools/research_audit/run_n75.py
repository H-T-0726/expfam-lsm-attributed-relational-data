"""Runner for n75_protocol.py (stages B, C, D).

Estimators are the PR #160 adapters run_fair_comparison.fit_old / fit_E (unchanged).
Before each fit the initial state of BOTH estimators is rebuilt from the seed
(no fitting) and hashed; if Z, F, sigma, w0, w differ the stage STOPs (seed /
initialisation policy).  Each fit is saved atomically (npz, then JSON with the
npz sha256, written last); valid fits are skipped on resume; FAILED fits are
never retried.  One process, BLAS threads 1, memory guard 3 GB.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "reproduction/src", ROOT / "expfam/src/experimental", ROOT / "tools/research_audit"):
    sys.path.insert(0, str(p))

import n75_protocol as npc  # noqa: E402
import run_fair_comparison as rf  # noqa: E402
from data_generator import set_true_params  # noqa: E402
from model import LatentStructuralModel  # noqa: E402
import em_runner  # noqa: E402

RESULTS = ROOT / npc.RESULTS_REL
STATES = Path(npc.STATE_ROOT)
N, D, KG = npc.COMMON["n"], npc.COMMON["d"], npc.COMMON["K_GEN"]


def need(c, m):
    if not c:
        raise SystemExit("STOP: " + m)


# ---------------------------------------------------------------- data
def canon75(rep):
    big = rf.e1_data(1, KG, rep)                      # PR #160 G_canon n=150 generator, same seed rule
    idx = np.arange(N)
    return dict(X=big["X"][idx], Y=big["Y"][np.ix_(idx, idx)], Z=big["Z"][idx], F=big["F"], w0=big["w0"], w=big["w"],
                train=rf.all_pairs(N), test=None, data_seed=big["data_seed"], x_family="gaussian", g=1,
                note="G_canon n=75: first 75 nodes of the PR #160 n=150 dataset")


def gold75(rep):
    seed = 1980 + 100 * KG + 1000 * (rep - 1)
    p = set_true_params(n=N, d=D, k=KG, seed=seed)
    return dict(X=p["X"], Y=p["Y"], Z=p["Z"], F=p["F"], w0=p["w0"], w=p["w"], train=rf.all_pairs(N), test=None,
                data_seed=seed, x_family="gaussian", g=0, note="G_old n=75: Z and X z-scored over 75 nodes")


def mask80(dseed):
    up = np.argwhere(np.triu(np.ones((N, N), dtype=bool), 1))
    rng = np.random.default_rng([dseed, npc.TAG_MASK])
    keep = rng.choice(len(up), size=int(round(0.8 * len(up))), replace=False)
    sel = up[np.sort(keep)]
    m = np.zeros((N, N), dtype=bool)
    m[sel[:, 0], sel[:, 1]] = True
    m = m | m.T
    test = ~m
    np.fill_diagonal(test, False)
    return m, test


def stage_data(stage, rep):
    if stage == "C":
        return gold75(rep)
    d = canon75(rep)
    if stage == "D":
        d["train"], d["test"] = mask80(d["data_seed"])
        need(not np.any(d["train"] & d["test"]), "mask overlaps held-out pairs")
        need(int(np.triu(d["train"], 1).sum()) == 2220, "mask size")
        d["note"] += "; 80% training pairs"
    return d


# ---------------------------------------------------------------- initial states (no fit)
def init_state(est, X, Y, train, k, seed):
    n, d = X.shape
    if est == "OLD":
        m = LatentStructuralModel(n=n, d=d, k=k, L=npc.COMMON["L"])
        m.initialize_params(true_params=None, seed=seed)
        up = np.triu(np.ones((n, n), dtype=bool), 1)
    else:
        m = em_runner.build_model(n, d, k, npc.COMMON["L"], "gaussian", "bernoulli", train_mask=train,
                                  numerics_mode="consistent")
        m.initialize_params(true_params=None, seed=seed)
        up = np.triu(m.train_mask, 1)
    dens = float(np.clip(Y[up].mean(), 1e-6, 1 - 1e-6))
    return {"Z": m.params["Z"], "F": m.params["F"], "sigma": m.params["sigma"], "w0": math.log(dens / (1 - dens)), "w": 0.5}


def init_hash(st):
    return {f"init_{k}_sha": rf.sha_arr(np.atleast_1d(np.asarray(v, float))) for k, v in st.items()}


# ---------------------------------------------------------------- fits
def stage_fits(stage):
    cfg = npc.STAGES[stage]
    g = 0 if stage == "C" else 1
    out = []
    for rep in cfg["replicates"]:
        for k in npc.COMMON["K_CAND"]:
            for s in npc.COMMON["starts"]:
                for est in cfg["estimators"]:
                    out.append(dict(stage=stage, key=f"{stage}_r{rep}_k{k}_s{s}_{est}", g=g, rep=rep, k=k, s=s,
                                    estimator=est, mask=cfg["mask"], seed=npc.model_seed(g, rep, k, s)))
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


def run_stage(stage):
    fits = stage_fits(stage)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "tools", "expfam/src", "reproduction/src"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    need(not dirty, "code tree dirty: commit before running")
    code_sha = {p: rf.sha_file(ROOT / "tools/research_audit" / p) for p in ("run_n75.py", "run_fair_comparison.py", "n75_protocol.py")}
    log = RESULTS / stage / "run_log.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    done = 0
    with log.open("a", encoding="utf-8") as lg:
        lg.write(f"{time.strftime('%H:%M:%S')} start {stage} fits={len(fits)} head={head} protocol={npc.protocol_hash()[:12]} "
                 f"code={ {k: v[:12] for k, v in code_sha.items()} }\n")
        for f in fits:
            if valid(f):
                continue
            waited = 0
            while rf.avail_gb() < npc.COMMON["mem_min_gb"]:
                if waited > 600:
                    lg.write(f"{time.strftime('%H:%M:%S')} STOP low memory\n")
                    return
                time.sleep(30)
                waited += 30
            if f["rep"] not in cache:
                cache.clear()
                cache[f["rep"]] = stage_data(stage, f["rep"])
            Dd = cache[f["rep"]]
            # initial-state agreement (both estimators, from the seed alone)
            hO = init_hash(init_state("OLD", Dd["X"], Dd["Y"], Dd["train"], f["k"], f["seed"]))
            hE = init_hash(init_state("E", Dd["X"], Dd["Y"], Dd["train"], f["k"], f["seed"]))
            same = {k: hO[k] == hE[k] for k in hO}
            if stage != "D":
                need(all(same.values()), f"initial states differ for {f['key']}: {same}")
            else:          # Phase D: only w0 may differ (training-pair density); Z, F, sigma, w must match
                need(all(v for k, v in same.items() if k != "init_w0_sha"), f"initial Z/F/sigma/w differ for {f['key']}")
            row = dict(f, protocol_hash=npc.protocol_hash(), git_head=head, code_sha=code_sha, data_seed=Dd["data_seed"],
                       x_sha256=rf.sha_arr(Dd["X"]), y_sha256=rf.sha_arr(Dd["Y"]), train_sha256=rf.sha_arr(Dd["train"]),
                       note=Dd["note"], **{f"{k}": v for k, v in (hE if f["estimator"] == "E" else hO).items()},
                       init_identical_OLD_E=bool(all(same.values())))
            t0 = time.perf_counter()
            try:
                if f["estimator"] == "OLD":
                    st = rf.fit_old(Dd["X"], Dd["Y"], f["k"], f["seed"], L=npc.COMMON["L"], iters=npc.COMMON["em_iterations"])
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
    """No fit.  Data checks + initial-state hashes for every planned fit -> dataset_and_initialization_hashes.csv."""
    import pandas as pd
    out = RESULTS / "dataset_and_initialization_hashes.csv"
    need(not out.exists(), f"refusing to overwrite {out}")
    rows = []
    for stage in npc.STAGES:
        cache = {}
        for f in stage_fits(stage):
            if f["rep"] not in cache:
                cache = {f["rep"]: stage_data(stage, f["rep"])}
                Dd = cache[f["rep"]]
                if stage in ("B", "D"):
                    big = rf.e1_data(1, KG, f["rep"])
                    need(np.array_equal(big["X"][:N], Dd["X"]) and np.array_equal(big["Y"][:N, :N], Dd["Y"])
                         and np.array_equal(big["F"], Dd["F"]), "nested subsample mismatch")
                need(np.allclose(Dd["Y"], Dd["Y"].T) and np.all(np.diag(Dd["Y"]) == 0) or stage == "C", "Y not symmetric / diag")
            Dd = cache[f["rep"]]
            hO = init_hash(init_state("OLD", Dd["X"], Dd["Y"], Dd["train"], f["k"], f["seed"]))
            hE = init_hash(init_state("E", Dd["X"], Dd["Y"], Dd["train"], f["k"], f["seed"]))
            r = dict(stage=stage, key=f["key"], estimator=f["estimator"], rep=f["rep"], k=f["k"], s=f["s"], seed=f["seed"],
                     data_seed=Dd["data_seed"], x_sha256=rf.sha_arr(Dd["X"]), y_sha256=rf.sha_arr(Dd["Y"]),
                     train_sha256=rf.sha_arr(Dd["train"]), n_train_pairs=int(np.triu(Dd["train"], 1).sum()),
                     y_density_all=float(Dd["Y"][np.triu_indices(N, 1)].mean()), w_true=float(Dd["w"]), w0_true=float(Dd["w0"]))
            for kk in hO:
                r[f"{kk}_OLD"] = hO[kk][:16]
                r[f"{kk}_E"] = hE[kk][:16]
                r[f"{kk}_equal"] = hO[kk] == hE[kk]
            rows.append(r)
    df = pd.DataFrame(rows)
    eq = [c for c in df if c.endswith("_equal")]
    bad = df[(df.stage != "D") & ~df[eq].all(axis=1)]
    need(bad.empty, f"initial states differ: {bad.key.tolist()[:5]}")
    badD = df[(df.stage == "D") & ~df[[c for c in eq if "w0" not in c]].all(axis=1)]
    need(badD.empty, "Phase D: Z/F/sigma/w differ")
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n")
    print(len(df), "planned fits; init identical (B, C):", bool(df[df.stage != "D"][eq].all().all()),
          "| D w0 differs:", int((~df[df.stage == "D"].init_w0_sha_equal).sum()))


def status(stage):
    c = {"OK": 0, "FAILED": 0, "NOT_RUN": 0}
    fits = stage_fits(stage)
    for f in fits:
        j = RESULTS / stage / "fits" / f"{f['key']}.json"
        if not j.exists() or not valid(f):
            c["NOT_RUN"] += 1
            continue
        c[json.loads(j.read_text(encoding="utf-8"))["status"]] += 1
    print(stage, len(fits), c)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status", "precheck"])
    ap.add_argument("--stage", choices=list(npc.STAGES))
    a = ap.parse_args()
    if a.cmd == "precheck":
        precheck()
    else:
        run_stage(a.stage) if a.cmd == "run" else status(a.stage)
