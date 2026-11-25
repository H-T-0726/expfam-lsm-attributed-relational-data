"""Convergence-sensitivity runner for the K-selection Exp2-generalisation run.

EXPLORATORY. Planned on 2026-11-25 AFTER the main 8,000-fit results were seen
(under-selection in B-b-9 and A-b-7/9). It does not change, re-run or overwrite
the main run. It asks one question: does the C_Q selection, and the C_Q gap
between candidate K, change when the same MCEM run is continued from 10 to 20
and 40 iterations?

What "continued" means here (exact definition, see the design doc):
  * The saved NPZ states hold theta and Z (last sample, sample mean) but NOT the
    numpy Generator state nor the L Z-samples, so an exact resume from the NPZ is
    impossible. This runner never claims to resume.
  * Instead every fit is a NEW deterministic run with the frozen seeds of the
    main run (same data_seed, init_seed, mc_seed, L, numerics) and
    num_iter = 40. The EM loop consumes random numbers in a fixed order that
    does not depend on num_iter, so iterations 1..10 of this run are expected to
    be bit-identical to the main 10-iteration fit. That is a checked gate:
    C_Q at checkpoint 10 must equal the main per_fit.csv C_Q bitwise.
  * C_Q at checkpoint t is calc_bic_exp(q_after[t]) where q_after[t] is the
    strict Q of the M-step diagnostic at iteration t (same Z samples, updated
    theta). In the main run Q_strict == q_after[10] bitwise for all 8,000 fits,
    so this is the frozen evaluation rule applied at earlier iterations.

Modes:
  --tech-check OUT   technical verification only (no new selection result):
                     (1) main-data reproduction with num_iter=10,
                     (2) prefix identity 10/20/40 on smoke-seed data,
                     (3) runtime of 40-iteration fits.
  --freeze OUT --cells B-b-9,O-b-9,A-b-9   write protocol.json + manifest.csv
  --execute --protocol P OUT [--resume]    run a frozen protocol
  --analyze RUN --out OUT                  pre-specified analysis

Human Gate: --freeze/--execute are for after the researcher approves the cell
set. Nothing here was executed beyond --tech-check at the time of writing.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in ("expfam/src", "expfam/src/experimental"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

import k_exp2_design as kd  # noqa: E402

MAIN_BASE = ROOT / "expfam/results/k_selection/k_exp2_generalization_20261125"
MAIN_PROTOCOL = MAIN_BASE / "protocol_v2" / "protocol.json"
MAIN_PROTOCOL_SHA256 = "14757d4a161712859601d52aba0e86308903a723f29baee0a6b3d5118ec5a7ad"
NUM_ITER_MAX = 40
CHECKPOINTS = (10, 20, 40)
N_WORKERS = 8
SMOKE_SEED_OFFSET = 5_000_000          # same offset as the main technical smoke
PROPOSED_CELLS = (("B", "b", 9), ("O", "b", 9), ("A", "b", 9))
CODE_FILES = (
    "expfam/src/experimental/em_runner.py",
    "expfam/src/experimental/eval_utils.py",
    "expfam/src/experimental/k_exp2_design.py",
    "expfam/src/experimental/data_generator_canonical.py",
    "expfam/src/experimental/matched_k_true_design.py",
    "expfam/src/experimental/model_dual_expfam_consistent.py",
    "expfam/src/experimental/model_dual_expfam_masked.py",
    "expfam/src/model_dual_expfam_fixed.py",
    "expfam/src/model_dual_expfam.py",
    "expfam/src/model_expfam.py",
    "reproduction/src/model.py",
    "tools/research_audit/run_k_exp2_convergence_sensitivity.py",
)
CP_FIELDS = ["fit_id", "dataset_id", "series", "regime", "k_true", "rep", "k", "t",
             "checkpoint", "Q", "C_Q", "num_params", "main_C_Q", "repro_equal"]
FIT_FIELDS = ["fit_id", "dataset_id", "series", "regime", "k_true", "rep", "k", "t",
              "data_seed", "init_seed", "mc_seed", "num_iter", "status", "failure_reason",
              "x_hash", "y_hash", "data_equal_main", "final_q_equals_last_q_after",
              "repro10_equal", "mstep_q_decreases", "retry_count", "replacement_count",
              "nan_count", "runtime_s", "state_file", "state_sha256"]


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(p: Path):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def bern_table() -> dict:
    proto = json.loads(MAIN_PROTOCOL.read_text(encoding="utf-8"))
    return {(key.split("-")[0], int(c["k"])): c for key, c in proto["calibration"].items()}


def parse_cells(text: str):
    out = []
    for tok in text.split(","):
        s, r, kt = tok.strip().split("-")
        out.append((s, r, int(kt)))
    return tuple(out)


def jobs_for_cells(cells, num_iter=NUM_ITER_MAX, seed_offset=0, main_rows=None,
                   ks=None, ts=None, reps=None):
    """Jobs from the frozen main manifest (seeds copied, never re-derived).

    seed_offset > 0 gives smoke jobs on different data (fit ids prefixed SMOKE-),
    with init/mc seeds derived by the frozen rule from the offset data seed."""
    man = read_csv(MAIN_BASE / "protocol_v2" / "manifest.csv")
    main_rows = main_rows or {}
    jobs = []
    for m in man:
        key = (m["series"], m["regime"], int(m["k_true"]))
        if key not in cells:
            continue
        k, t, rep = int(m["k"]), int(m["t"]), int(m["rep"])
        if (ks and k not in ks) or (ts and t not in ts) or (reps and rep not in reps):
            continue
        j = dict(fit_id=m["fit_id"], dataset_id=m["dataset_id"], series=m["series"],
                 regime=m["regime"], k_true=int(m["k_true"]), rep=rep, k=k, t=t,
                 family_x=m["family_x"], family_y=m["family_y"],
                 data_seed=int(m["data_seed"]), init_seed=int(m["init_seed"]),
                 mc_seed=int(m["mc_seed"]), num_iter=num_iter)
        if seed_offset:
            ds = j["data_seed"] + seed_offset
            j.update(fit_id="SMOKE-" + j["fit_id"], dataset_id="SMOKE-" + j["dataset_id"],
                     data_seed=ds, init_seed=10 * ds + t, mc_seed=10 * ds + 5 + t)
        else:
            r = main_rows.get(m["fit_id"])
            if r is not None:
                j.update(main_C_Q=r["C_Q"], main_x_hash=r["x_hash"], main_y_hash=r["y_hash"])
        jobs.append(j)
    return jobs


def checkpoint_cq(hist, k, family_x, family_y, checkpoints):
    """C_Q at each checkpoint from the M-step strict-Q history (frozen rule)."""
    from eval_utils import calc_bic_exp
    out = []
    for c in checkpoints:
        if c > len(hist):
            continue
        h = hist[c - 1]
        assert int(h["iteration"]) == c
        q = float(h["q_after"])
        cq, npar = calc_bic_exp(q, k, kd.N, kd.D, family_x, family_y)
        out.append((c, q, float(cq), int(npar)))
    return out


def fit_one(job, table, state_dir):
    from em_runner import run_em_experimental
    t0 = time.perf_counter()
    row = {f: "" for f in FIT_FIELDS}
    row.update({k: job[k] for k in ("fit_id", "dataset_id", "series", "regime", "k_true",
                                    "rep", "k", "t", "data_seed", "init_seed", "mc_seed",
                                    "num_iter")})
    spec = kd.DatasetSpec(job["series"], job["regime"], job["k_true"], job["rep"],
                          kd.w_of(job["series"], job["regime"], job["k_true"]),
                          kd.w0_of(job["series"], job["regime"], job["k_true"], table),
                          kd.f_scale_of(job["regime"], job["k_true"]), job["data_seed"])
    ds = kd.generate(spec)
    X, Y = ds.X, ds.Y
    row.update(x_hash=kd.array_hash(X), y_hash=kd.array_hash(Y))
    if "main_x_hash" in job:
        row["data_equal_main"] = (row["x_hash"], row["y_hash"]) == (job["main_x_hash"],
                                                                    job["main_y_hash"])
    cps = []
    try:
        r = run_em_experimental(
            X, Y, job["family_x"], job["family_y"], job["k"], L=kd.L_MC,
            num_iter=job["num_iter"], numerics_mode="consistent", failure_policy="fail_fast",
            init_seed=job["init_seed"], mc_seed=job["mc_seed"], compute_strict_Q=True,
            mstep_q_diagnostic=True)
        hist = r.get("mstep_q_history") or []
        cps = checkpoint_cq(hist, job["k"], job["family_x"], job["family_y"],
                            tuple(c for c in CHECKPOINTS if c <= job["num_iter"])
                            if job["num_iter"] in CHECKPOINTS else (job["num_iter"],))
        q = float(r["Q_strict"])
        finite_theta = all(np.all(np.isfinite(np.asarray(r[key], float)))
                           for key in ("F", "sigma", "w0", "w"))
        ok = (np.isfinite(q) and np.isfinite(float(r["bic"])) and not r["q_bic_failed"]
              and finite_theta and r["retry_count"] == 0 and r["replacement_count"] == 0
              and len(hist) == job["num_iter"]
              and all(np.isfinite(c[2]) for c in cps))
        row.update(status="OK" if ok else "FAILED",
                   failure_reason="" if ok else "non-finite or integrity",
                   final_q_equals_last_q_after=(q == float(hist[-1]["q_after"])) if hist else False,
                   mstep_q_decreases=sum(bool(h["decreased"]) for h in hist),
                   retry_count=r["retry_count"], replacement_count=r["replacement_count"],
                   nan_count=r["nan_count"])
        if "main_C_Q" in job:
            c10 = [c for c in cps if c[0] == 10]
            row["repro10_equal"] = bool(c10) and repr(c10[0][2]) == job["main_C_Q"]
        row["_hist"] = [{"fit_id": job["fit_id"], **h} for h in hist]
        if state_dir and ok:
            sp = Path(state_dir) / f"{job['fit_id']}.npz"
            np.savez_compressed(sp, F=r["F"], sigma=r["sigma"], w0=r["w0"], w=r["w"],
                                var_z=r["var_z"], sigma_y=r["sigma_y_est"],
                                Z_last=r["Z_est"], Z_mean=r["Z_samples"].mean(axis=2))
            row["state_file"] = sp.name
            row["state_sha256"] = sha256_file(sp)
    except Exception as exc:  # recorded once; never retried
        row.update(status="FAILED", failure_reason=f"{type(exc).__name__}: {str(exc)[:300]}")
        row["_hist"] = []
    row["_cps"] = [dict(fit_id=job["fit_id"], dataset_id=job["dataset_id"],
                        series=job["series"], regime=job["regime"], k_true=job["k_true"],
                        rep=job["rep"], k=job["k"], t=job["t"], checkpoint=c, Q=repr(q),
                        C_Q=repr(cq), num_params=npar, main_C_Q=job.get("main_C_Q", ""),
                        repro_equal=(repr(cq) == job["main_C_Q"]) if (c == 10 and "main_C_Q" in job) else "")
                   for c, q, cq, npar in cps]
    row["runtime_s"] = round(time.perf_counter() - t0, 2)
    return row


def _worker(args):
    return fit_one(*args)


class Writers:
    def __init__(self, out: Path, append: bool):
        mode = "a" if append else "w"
        self.files = {}
        self.w = {}
        for name, fields in (("per_fit.csv", FIT_FIELDS), ("checkpoints.csv", CP_FIELDS),
                             ("mstep_q_history.csv", ["fit_id", "iteration", "q_before",
                                                      "q_after", "q_diff", "decreased"])):
            p = out / name
            new = not p.exists() or not append
            f = open(p, mode, encoding="utf-8", newline="")
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            self.files[name], self.w[name] = f, w

    def write(self, row):
        self.w["per_fit.csv"].writerow(row)
        for c in row["_cps"]:
            self.w["checkpoints.csv"].writerow(c)
        for h in row["_hist"]:
            self.w["mstep_q_history.csv"].writerow(h)
        for f in self.files.values():
            f.flush()

    def close(self):
        for f in self.files.values():
            f.close()


def run_jobs(jobs, out: Path, state_dir, append=False, events=None):
    table = bern_table()
    wr = Writers(out, append)
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    done = 0
    rows = []
    try:
        with ctx.Pool(N_WORKERS, maxtasksperchild=20) as pool:
            for row in pool.imap_unordered(_worker, [(j, table, state_dir) for j in jobs]):
                wr.write(row)
                rows.append(row)
                done += 1
                if done % 8 == 0 or done == len(jobs):
                    el = time.perf_counter() - t0
                    print(f"[{now()}] {done}/{len(jobs)} elapsed {el / 3600:.2f} h", flush=True)
    finally:
        wr.close()
    return rows


def code_hashes():
    return {p: sha256_file(ROOT / p) for p in CODE_FILES}


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def git_clean():
    s = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                       capture_output=True, text=True, check=True).stdout
    return s.strip() == ""


def main_per_fit():
    return {r["fit_id"]: r for r in read_csv(MAIN_BASE / "main" / "per_fit.csv")}


# ── technical verification ─────────────────────────────────────────────
def tech_check(out: Path):
    """No new selection result is produced or printed.

    (1) reproduction: main data, num_iter=10, 4 fits of B-b-9 (K = 1, 9, 10; both inits
        for K = 9) -> C_Q must equal main per_fit.csv bitwise (known values only).
    (2) prefix identity: smoke-seed data (offset 5,000,000; never used for selection),
        cells B-b-9, O-b-9, A-b-9, K in {1, 10}, t = 0, num_iter in {10, 20, 40}:
        the 40-iteration history must start with the 10/20-iteration histories bitwise,
        and the checkpoint C_Q of the 40 run must equal the final C_Q of the 10/20 runs.
    (3) runtime of the 40-iteration fits with 8 workers.
    """
    if out.exists():
        raise SystemExit(f"STOP: {out} exists")
    out.mkdir(parents=True)
    mrows = main_per_fit()
    rep_jobs = []
    for k, t in ((1, 0), (9, 0), (9, 1), (10, 0)):
        rep_jobs += jobs_for_cells((("B", "b", 9),), num_iter=10, main_rows=mrows,
                                   ks={k}, ts={t}, reps={1})
    pre_jobs = []
    for n_it in (40, 20, 10):
        pre_jobs += jobs_for_cells(PROPOSED_CELLS, num_iter=n_it, seed_offset=SMOKE_SEED_OFFSET,
                                   ks={1, 10}, ts={0}, reps={0})
    for j in pre_jobs:
        j["fit_id"] += f"-it{j['num_iter']}"
    jobs = rep_jobs + pre_jobs
    started = now()
    rows = run_jobs(jobs, out, None)
    res: dict = {"started": started, "finished": now(), "git_head": git_head(),
                 "code_sha256": code_hashes(), "n_jobs": len(jobs)}
    by = {r["fit_id"]: r for r in rows}
    res["all_status_OK"] = all(r["status"] == "OK" for r in rows)
    res["failures"] = [(r["fit_id"], r["failure_reason"]) for r in rows if r["status"] != "OK"]
    rr = [by[j["fit_id"]] for j in rep_jobs]
    res["repro10"] = {"n": len(rr),
                      "data_equal_main": all(r["data_equal_main"] is True for r in rr),
                      "c_q_equal_main_bitwise": all(r["repro10_equal"] is True for r in rr)}
    hist = defaultdict(list)
    for r in rows:
        hist[r["fit_id"]] = r["_hist"]
    cpq = {r["fit_id"]: {c["checkpoint"]: c["C_Q"] for c in r["_cps"]} for r in rows}
    pre = []
    for j in pre_jobs:
        if j["num_iter"] != 40:
            continue
        base = j["fit_id"][: -len("-it40")]
        h40 = hist[j["fit_id"]]
        for n_it in (10, 20):
            fid = f"{base}-it{n_it}"
            hs = hist[fid]
            same_hist = len(hs) == n_it and all(
                (a["q_before"], a["q_after"]) == (b["q_before"], b["q_after"])
                for a, b in zip(hs, h40[:n_it]))
            same_cq = cpq[fid].get(n_it) == cpq[j["fit_id"]].get(n_it)
            pre.append({"fit": base, "num_iter": n_it, "history_prefix_bitwise": same_hist,
                        "checkpoint_C_Q_equals_final_C_Q_bitwise": same_cq,
                        "final_q_equals_last_q_after": by[fid]["final_q_equals_last_q_after"]})
    res["prefix"] = {"pairs": pre, "all_ok": all(p["history_prefix_bitwise"]
                                                  and p["checkpoint_C_Q_equals_final_C_Q_bitwise"]
                                                  and p["final_q_equals_last_q_after"]
                                                  for p in pre)}
    rt = [r["runtime_s"] for r in rows if r["num_iter"] == 40]
    rt_k = {k: [r["runtime_s"] for r in rows if r["num_iter"] == 40 and r["k"] == k]
            for k in (1, 10)}
    res["runtime_40iter"] = {"n": len(rt), "mean_s": float(np.mean(rt)),
                             "max_s": float(np.max(rt)),
                             "mean_by_k": {k: float(np.mean(v)) for k, v in rt_k.items()}}
    m = res["runtime_40iter"]["mean_s"]
    res["budget_estimate"] = {
        f"{len(c)}_cells": {"fits": 200 * len(c),
                            "cpu_h": 200 * len(c) * m / 3600,
                            "wall_h_8_workers": 200 * len(c) * m / 3600 / N_WORKERS}
        for c in (PROPOSED_CELLS[:1], PROPOSED_CELLS[:2], PROPOSED_CELLS)}
    res["all_ok"] = bool(res["all_status_OK"] and res["repro10"]["data_equal_main"]
                         and res["repro10"]["c_q_equal_main_bitwise"] and res["prefix"]["all_ok"])
    (out / "tech_check.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("all_ok", "all_status_OK", "repro10",
                                          "runtime_40iter", "budget_estimate")}, indent=2))
    print("prefix all_ok:", res["prefix"]["all_ok"])
    return 0 if res["all_ok"] else 1


# ── freeze / execute ───────────────────────────────────────────────────
def freeze(out: Path, cells):
    if out.exists():
        raise SystemExit(f"STOP: {out} exists")
    if sha256_file(MAIN_PROTOCOL) != MAIN_PROTOCOL_SHA256:
        raise SystemExit("STOP: main protocol hash mismatch")
    out.mkdir(parents=True)
    jobs = jobs_for_cells(cells, main_rows=main_per_fit())
    man = [{k: j[k] for k in ("fit_id", "dataset_id", "series", "regime", "k_true", "rep",
                              "k", "t", "family_x", "family_y", "data_seed", "init_seed",
                              "mc_seed", "num_iter")} for j in jobs]
    with open(out / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0]))
        w.writeheader()
        w.writerows(man)
    proto = {
        "experiment_id": "k_exp2_convergence_sensitivity",
        "status": "EXPLORATORY (planned after the main results were seen)",
        "main_protocol_sha256": MAIN_PROTOCOL_SHA256,
        "cells": ["-".join(map(str, c)) for c in cells],
        "num_iter": NUM_ITER_MAX, "checkpoints": list(CHECKPOINTS), "L": kd.L_MC,
        "seeds": "copied from the main manifest (no new seeds)",
        "definition": "new deterministic run with the frozen seeds; not a resume; "
                      "C_Q(t) = calc_bic_exp(q_after[t]); gate: C_Q(10) == main C_Q bitwise",
        "selection": "per dataset and checkpoint: mean of C_Q over 2 inits, argmin, "
                     "ties within 1e-9 -> smallest K (same as main)",
        "no_adaptive_changes": "no extra iterations, seeds, cells or checkpoints after results",
        "n_fits": len(man), "workers": N_WORKERS, "blas_threads": 1,
        "code_sha256": code_hashes(),
        "manifest_sha256": sha256_file(out / "manifest.csv"),
        "frozen_at": now(),
    }
    (out / "protocol.json").write_text(json.dumps(proto, indent=2, sort_keys=True),
                                       encoding="utf-8")
    print(f"frozen {len(man)} fits -> {out}")


def execute(proto_path: Path, out: Path, resume: bool):
    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    if code_hashes() != proto["code_sha256"]:
        raise SystemExit("STOP: code hash mismatch")
    if not git_clean():
        raise SystemExit("STOP: git working tree not clean")
    man_p = proto_path.parent / "manifest.csv"
    if sha256_file(man_p) != proto["manifest_sha256"]:
        raise SystemExit("STOP: manifest hash mismatch")
    if out.exists() and not resume:
        raise SystemExit(f"STOP: {out} exists (use --resume)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "states").mkdir(exist_ok=True)
    cells = parse_cells(",".join(proto["cells"]))
    jobs = jobs_for_cells(cells, main_rows=main_per_fit())
    if [j["fit_id"] for j in jobs] != [m["fit_id"] for m in read_csv(man_p)]:
        raise SystemExit("STOP: manifest does not match the job list")
    done = set()
    if resume and (out / "per_fit.csv").exists():
        done = {r["fit_id"] for r in read_csv(out / "per_fit.csv")}
    todo = [j for j in jobs if j["fit_id"] not in done]
    ev = {"time": now(), "event": "start", "already_recorded": len(done),
          "to_run": len(todo), "git_head": git_head(),
          "protocol_sha256": sha256_file(proto_path)}
    with open(out / "run_events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")
    if not (out / "runinfo.json").exists():
        (out / "runinfo.json").write_text(json.dumps({
            "protocol": proto_path.as_posix(), "protocol_sha256": ev["protocol_sha256"],
            "git_head_at_start": ev["git_head"], "python": sys.version,
            "platform": platform.platform(), "numpy": np.__version__,
            "workers": N_WORKERS, "blas_threads": 1}, indent=2), encoding="utf-8")
    t0 = time.perf_counter()
    run_jobs(todo, out, out / "states", append=bool(done))
    with open(out / "run_events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now(), "event": "finished_batch", "ran": len(todo),
                            "wall_s": round(time.perf_counter() - t0, 1)}) + "\n")


# ── pre-specified analysis ─────────────────────────────────────────────
def select(cqbar: dict, tol=1e-9) -> int:
    mn = min(cqbar.values())
    return min(k for k, v in cqbar.items() if v - mn <= tol)


def analyze(run: Path, out: Path):
    """Per dataset and checkpoint: selected K (2-init mean), per-init K, outcome,
    gap G(t) = C_Q_bar(K_TRUE) - min_{K != K_TRUE} C_Q_bar(K) (negative = K_TRUE wins),
    transitions 10 -> 20 -> 40. A dataset is INCOMPLETE at a checkpoint if any of its
    20 fits is missing / FAILED. Gate: every checkpoint-10 C_Q equals the main C_Q."""
    if out.exists():
        raise SystemExit(f"STOP: {out} exists")
    fits = {r["fit_id"]: r for r in read_csv(run / "per_fit.csv")}
    cps = read_csv(run / "checkpoints.csv")
    out.mkdir(parents=True)
    rows = analyze_rows(cps, fits)
    gate = all(c["repro_equal"] == "True" for c in cps if int(c["checkpoint"]) == 10)
    with open(out / "selection_by_checkpoint.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cell = defaultdict(Counter)
    for r in rows:
        cell[(r["series"], r["regime"], r["k_true"], r["checkpoint"])][r["outcome"]] += 1
    with open(out / "summary_by_cell_checkpoint.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "regime", "k_true", "checkpoint", "exact", "under", "over",
                    "incomplete"])
        for c in sorted(cell):
            w.writerow([*c, *(cell[c][o] for o in ("exact", "under", "over", "INCOMPLETE"))])
    trans = Counter()
    by = defaultdict(dict)
    for r in rows:
        by[r["dataset_id"]][r["checkpoint"]] = r["selected_k"]
    for ds, d in by.items():
        trans[(ds.rsplit("-", 1)[0], tuple(d.get(c) for c in CHECKPOINTS))] += 1
    with open(out / "transitions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "k10", "k20", "k40", "n_datasets"])
        for (c, ks), n in sorted(trans.items(), key=lambda x: (x[0][0], str(x[0][1]))):
            w.writerow([c, *ks, n])
    (out / "status.json").write_text(json.dumps({
        "repro_gate_checkpoint10_equals_main": gate, "n_fits": len(fits),
        "n_ok": sum(r["status"] == "OK" for r in fits.values()),
        "n_dataset_checkpoints": len(rows),
        "n_incomplete": sum(r["outcome"] == "INCOMPLETE" for r in rows)}, indent=2),
        encoding="utf-8")


def analyze_rows(cps, fits):
    ok = {f for f, r in fits.items() if r["status"] == "OK"}
    d = defaultdict(dict)
    meta = {}
    for c in cps:
        if c["fit_id"] not in ok:
            continue
        key = (c["dataset_id"], int(c["checkpoint"]))
        d[key][(int(c["k"]), int(c["t"]))] = float(c["C_Q"])
        meta[c["dataset_id"]] = (c["series"], c["regime"], int(c["k_true"]), int(c["rep"]))
    all_ds = {r["dataset_id"]: (r["series"], r["regime"], int(r["k_true"]), int(r["rep"]))
              for r in fits.values()}
    rows = []
    for ds, (s, rg, kt, rep) in sorted(all_ds.items()):
        for cp in CHECKPOINTS:
            dct = d.get((ds, cp), {})
            ks = sorted({k for k, _ in dct})
            base = dict(dataset_id=ds, series=s, regime=rg, k_true=kt, rep=rep, checkpoint=cp)
            if len(dct) != 20 or ks != list(kd.K_CANDIDATES):
                rows.append({**base, "selected_k": "", "outcome": "INCOMPLETE", "init0_k": "",
                             "init1_k": "", "margin": "", "gap_ktrue_vs_best_other": ""})
                continue
            bar = {k: (dct[(k, 0)] + dct[(k, 1)]) / 2.0 for k in ks}
            sel = select(bar)
            srt = sorted(bar.values())
            other = min(v for k, v in bar.items() if k != kt)
            rows.append({**base, "selected_k": sel,
                         "outcome": "exact" if sel == kt else ("under" if sel < kt else "over"),
                         "init0_k": select({k: dct[(k, 0)] for k in ks}),
                         "init1_k": select({k: dct[(k, 1)] for k in ks}),
                         "margin": srt[1] - srt[0],
                         "gap_ktrue_vs_best_other": bar[kt] - other})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech-check", metavar="OUT")
    ap.add_argument("--freeze", metavar="OUT")
    ap.add_argument("--cells", default=",".join("-".join(map(str, c)) for c in PROPOSED_CELLS))
    ap.add_argument("--execute", metavar="OUT")
    ap.add_argument("--protocol")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--analyze", metavar="RUN")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.tech_check:
        return tech_check(ROOT / a.tech_check)
    if a.freeze:
        return freeze(ROOT / a.freeze, parse_cells(a.cells))
    if a.execute:
        if not a.protocol:
            raise SystemExit("--execute needs --protocol")
        return execute(ROOT / a.protocol, ROOT / a.execute, a.resume)
    if a.analyze:
        return analyze(ROOT / a.analyze, ROOT / a.out)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
