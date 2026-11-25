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
  --execute OUT --protocol P [--resume]    run a frozen protocol
  --analyze RUN --out OUT                  pre-specified analysis + figures

Human Gate: the researcher approved the 3-cell, 600-fit run on 2026-11-25
(decision D-B). Analysis and figures were written and tested on fake data before
the protocol was frozen and before any 20/40-iteration score of the main data.
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
HIST_FIELDS = ["fit_id", "iteration", "q_before", "q_after", "q_diff", "decreased"]
CELL_COLORS = {"B-b-9": "#2a78d6", "O-b-9": "#eb6834", "A-b-9": "#1baf7a"}  # fixed per cell
RESUME_RULE = ("a fit is done iff its fit_id has a row in per_fit.csv (any status); done fits "
               "are never re-run; FAILED fits are never retried or re-seeded; not-done fits "
               "are run with their frozen seeds; orphan history/checkpoint rows of not-done "
               "fits (crash between writes) are kept and must be bitwise identical to the "
               "re-run rows (checked by --analyze)")


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
        for name, fields in (("checkpoint_scores.csv", CP_FIELDS),
                             ("convergence_history.csv", HIST_FIELDS),
                             ("per_fit.csv", FIT_FIELDS)):
            p = out / name
            new = not p.exists() or not append
            f = open(p, mode, encoding="utf-8", newline="")
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            self.files[name], self.w[name] = f, w

    def write(self, row):
        # history and checkpoint rows first, per_fit last: a per_fit row implies the
        # fit's other rows are on disk (the resume rule keys on per_fit only)
        for c in row["_cps"]:
            self.w["checkpoint_scores.csv"].writerow(c)
        for h in row["_hist"]:
            self.w["convergence_history.csv"].writerow(h)
        self.files["checkpoint_scores.csv"].flush()
        self.files["convergence_history.csv"].flush()
        self.w["per_fit.csv"].writerow(row)
        self.files["per_fit.csv"].flush()

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
        "no_adaptive_changes": "no extra iterations, seeds, cells, checkpoints, init rules "
                               "or criterion changes after results",
        "researcher_decisions_20261125": {
            "D-A": "lineage E results are a candidate for the thesis body; Human Gate NOT "
                   "passed; not treated as adopted",
            "D-B": "run B-b-9, O-b-9, A-b-9 (600 fits) convergence sensitivity",
            "D-C": "C_Lap is a separate future exploratory analysis; not run here"},
        "criterion": "C_Q = -2 Q_strict + p ln n (Q-based complete-data / ICL-type; NOT "
                     "Schwarz BIC); its ln p(Z) term adds n(1+ln 2pi) per latent dimension",
        "resume_rule": RESUME_RULE,
        "failure_rule": "a failed fit is recorded once (failures.csv) and never replaced; a "
                        "dataset with any missing/FAILED fit is INCOMPLETE at that checkpoint",
        "stop_rules": ["any checkpoint-10 C_Q differs bitwise from the main per_fit C_Q",
                       "any regenerated X/Y hash differs from the main run",
                       "code hash mismatch / dirty tree / output exists without --resume"],
        "analysis_plan": {
            "primary": "per cell and checkpoint (10, 20, 40): number of datasets whose "
                       "selected K equals K_TRUE = 9 (Clopper-Pearson 95% CI), under/over",
            "secondary": ["transitions of selected K 10->20->40 per dataset",
                          "per-init selected K and init disagreement per checkpoint",
                          "gap C_Q_bar(9) - C_Q_bar(selected K) and G(t) = C_Q_bar(9) - "
                          "min_{K!=9} C_Q_bar(K) for t = 1..40",
                          "per-K C_Q_bar improvement 10->20->40 and mean per-iteration "
                          "improvement over the last 3 iterations before each checkpoint",
                          "numerical anomalies: failures, M-step strict-Q decreases, NaN"],
            "unit": "dataset (inits are not independent evidence; checkpoints of one "
                    "dataset are not independent evidence)",
            "not_claimed": "40-iteration convergence, consistency or optimality of C_Q"},
        "outputs": ["per_fit.csv", "checkpoint_scores.csv", "convergence_history.csv",
                    "failures.csv", "seed_manifest.csv", "runinfo.json", "run_events.jsonl",
                    "states/ (git-ignored) + states_sha256_manifest.csv"],
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
    if not (out / "seed_manifest.csv").exists():
        (out / "seed_manifest.csv").write_bytes(man_p.read_bytes())
    t0 = time.perf_counter()
    run_jobs(todo, out, out / "states", append=bool(done))
    fits = read_csv(out / "per_fit.csv")
    with open(out / "failures.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fit_id", "status", "failure_reason"])
        for r in fits:
            if r["status"] != "OK":
                w.writerow([r["fit_id"], r["status"], r["failure_reason"]])
    with open(out / "states_sha256_manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state_file", "bytes", "sha256"])
        for p in sorted((out / "states").glob("*.npz")):
            w.writerow([p.name, p.stat().st_size, sha256_file(p)])
    with open(out / "run_events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now(), "event": "finished_batch", "ran": len(todo),
                            "recorded": len(fits),
                            "ok": sum(r["status"] == "OK" for r in fits),
                            "wall_s": round(time.perf_counter() - t0, 1)}) + "\n")


# ── pre-specified analysis ─────────────────────────────────────────────
def select(cqbar: dict, tol=1e-9) -> int:
    mn = min(cqbar.values())
    return min(k for k, v in cqbar.items() if v - mn <= tol)


def clopper_pearson(x: int, n: int, a: float = 0.05):
    from scipy.stats import beta
    lo = 0.0 if x == 0 else float(beta.ppf(a / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - a / 2, x + 1, n - x))
    return lo, hi


def _write(path: Path, rows, fields=None):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _dedupe(rows, key):
    """Orphan rows from an interrupted write must be bitwise identical to re-run rows."""
    seen = {}
    for r in rows:
        k = tuple(r[c] for c in key)
        if k in seen:
            if seen[k] != r:
                raise SystemExit(f"STOP: duplicate rows differ for {k}")
            continue
        seen[k] = r
    return list(seen.values())


def _cell(r) -> str:
    return f"{r['series']}-{r['regime']}-{r['k_true']}"


def analyze(run: Path, out: Path, figures: bool = True):
    """Pre-specified analysis (frozen with the protocol; see analysis_plan).

    Per dataset and checkpoint: selected K (2-init mean, ties within 1e-9 -> smallest K),
    per-init K, outcome, gaps. Per cell x K x iteration: C_Q_bar trajectories from the
    M-step strict-Q history (C_Q(t) = calc_bic_exp(q_after[t]), the frozen rule). A
    dataset is INCOMPLETE at a checkpoint / iteration if any of its 20 fits is missing
    or FAILED. Gate: every checkpoint-10 C_Q equals the main per_fit C_Q bitwise."""
    if out.exists():
        raise SystemExit(f"STOP: {out} exists")
    fits = {r["fit_id"]: r for r in read_csv(run / "per_fit.csv")}
    cps = _dedupe(read_csv(run / "checkpoint_scores.csv"), ("fit_id", "checkpoint"))
    cps = [c for c in cps if c["fit_id"] in fits]
    hp = run / "convergence_history.csv"
    hist = _dedupe(read_csv(hp), ("fit_id", "iteration")) if hp.exists() else []
    hist = [h for h in hist if h["fit_id"] in fits]
    out.mkdir(parents=True)
    rows = analyze_rows(cps, fits)
    gate = bool(cps) and all(c["repro_equal"] == "True"
                             for c in cps if int(c["checkpoint"]) == 10)
    _write(out / "selected_k_by_dataset.csv", rows)
    cell = defaultdict(list)
    for r in rows:
        cell[(r["series"], r["regime"], r["k_true"], r["checkpoint"])].append(r)
    summ = []
    for c in sorted(cell):
        rs = cell[c]
        cnt = Counter(r["outcome"] for r in rs)
        comp = [r for r in rs if r["outcome"] != "INCOMPLETE"]
        lo, hi = clopper_pearson(cnt["exact"], len(comp)) if comp else ("", "")
        gaps = [r["gap_k9_minus_selected"] for r in comp]
        gs = [r["gap_ktrue_vs_best_other"] for r in comp]
        summ.append(dict(cell="-".join(map(str, c[:3])), series=c[0], regime=c[1],
                         k_true=c[2], checkpoint=c[3], n_complete=len(comp),
                         exact=cnt["exact"], under=cnt["under"], over=cnt["over"],
                         incomplete=cnt["INCOMPLETE"], exact_ci95_lo=lo, exact_ci95_hi=hi,
                         init_disagree=sum(r["init_disagree"] for r in comp),
                         init_both_exact=sum(r["init0_k"] == r["init1_k"] == c[2]
                                             for r in comp),
                         median_selected_k=float(np.median([r["selected_k"] for r in comp]))
                         if comp else "",
                         median_gap_ktrue_minus_selected=float(np.median(gaps)) if gaps else "",
                         median_G=float(np.median(gs)) if gs else ""))
    _write(out / "summary_by_cell_checkpoint.csv", summ)
    trans = Counter()
    by = defaultdict(dict)
    for r in rows:
        by[(r["dataset_id"], _cell(r))][r["checkpoint"]] = r["selected_k"]
    for (ds, c), d in by.items():
        trans[(c, tuple(d.get(cp) for cp in CHECKPOINTS))] += 1
    _write(out / "transitions.csv",
           [dict(cell=c, k10=ks[0], k20=ks[1], k40=ks[2], n_datasets=n)
            for (c, ks), n in sorted(trans.items(), key=lambda x: (x[0][0], str(x[0][1])))],
           ["cell", "k10", "k20", "k40", "n_datasets"])
    traj = trajectories(hist, fits) if hist else {}
    if traj and traj["by_dataset"]:
        _write(out / "cq_trajectory_by_dataset.csv", traj["by_dataset"])
        _write(out / "cq_trajectory_median_by_cell_k.csv", traj["median"])
    if traj and traj["improvement"]:
        _write(out / "per_k_improvement.csv", traj["improvement"])
        _write(out / "per_k_improvement_median.csv", traj["improvement_median"])
    dec = Counter()
    for r in fits.values():
        dec[_cell(r)] += int(r["mstep_q_decreases"] or 0)
    status = {
        "repro_gate_checkpoint10_equals_main": gate,
        "data_equal_main_all": all(r["data_equal_main"] == "True" for r in fits.values()),
        "n_fits": len(fits), "n_ok": sum(r["status"] == "OK" for r in fits.values()),
        "n_failed": sum(r["status"] != "OK" for r in fits.values()),
        "n_dataset_checkpoints": len(rows),
        "n_incomplete": sum(r["outcome"] == "INCOMPLETE" for r in rows),
        "history_rows": len(hist),
        "final_q_equals_last_q_after_all": all(
            r["final_q_equals_last_q_after"] == "True" for r in fits.values()
            if r["status"] == "OK"),
        "retry_total": sum(int(r["retry_count"] or 0) for r in fits.values()),
        "replacement_total": sum(int(r["replacement_count"] or 0) for r in fits.values()),
        "nan_total": sum(int(r["nan_count"] or 0) for r in fits.values()),
        "mstep_q_decreases_by_cell": dict(dec),
    }
    (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if figures and rows and traj and traj["by_dataset"]:
        make_figures(out, summ, rows, traj)
    return status


def analyze_rows(cps, fits):
    ok = {f for f, r in fits.items() if r["status"] == "OK"}
    d = defaultdict(dict)
    for c in cps:
        if c["fit_id"] not in ok:
            continue
        d[(c["dataset_id"], int(c["checkpoint"]))][(int(c["k"]), int(c["t"]))] = float(c["C_Q"])
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
                             "init1_k": "", "init_disagree": "", "margin": "",
                             "gap_k9_minus_selected": "", "gap_ktrue_vs_best_other": "",
                             "cq_bar_ktrue": "", "cq_bar_selected": ""})
                continue
            bar = {k: (dct[(k, 0)] + dct[(k, 1)]) / 2.0 for k in ks}
            sel = select(bar)
            srt = sorted(bar.values())
            other = min(v for k, v in bar.items() if k != kt)
            i0 = select({k: dct[(k, 0)] for k in ks})
            i1 = select({k: dct[(k, 1)] for k in ks})
            rows.append({**base, "selected_k": sel,
                         "outcome": "exact" if sel == kt else ("under" if sel < kt else "over"),
                         "init0_k": i0, "init1_k": i1, "init_disagree": int(i0 != i1),
                         "margin": srt[1] - srt[0],
                         "gap_k9_minus_selected": bar[kt] - bar[sel],
                         "gap_ktrue_vs_best_other": bar[kt] - other,
                         "cq_bar_ktrue": bar[kt], "cq_bar_selected": bar[sel]})
    return rows


def trajectories(hist, fits):
    """C_Q_bar(K, t) for t = 1..NUM_ITER_MAX from q_after (frozen C_Q rule at every t)."""
    from eval_utils import calc_bic_exp
    ok = {f: r for f, r in fits.items() if r["status"] == "OK"}
    q = defaultdict(dict)
    for h in hist:
        if h["fit_id"] in ok:
            q[h["fit_id"]][int(h["iteration"])] = float(h["q_after"])
    cq = defaultdict(dict)          # (ds, k, t) -> {iter: C_Q}
    meta = {r["dataset_id"]: (_cell(r), int(r["k_true"])) for r in fits.values()}
    for fid, r in ok.items():
        k = int(r["k"])
        fx, fy = kd.SERIES[r["series"]]
        for it, qa in q[fid].items():
            cq[(r["dataset_id"], k, int(r["t"]))][it] = float(
                calc_bic_exp(qa, k, kd.N, kd.D, fx, fy)[0])
    its = range(1, NUM_ITER_MAX + 1)
    bar = {}
    for ds in meta:
        for k in kd.K_CANDIDATES:
            a, b = cq.get((ds, k, 0), {}), cq.get((ds, k, 1), {})
            for it in its:
                if it in a and it in b:
                    bar[(ds, k, it)] = (a[it] + b[it]) / 2.0
    by_ds = []
    for ds, (cell, kt) in sorted(meta.items()):
        for it in its:
            vals = {k: bar.get((ds, k, it)) for k in kd.K_CANDIDATES}
            if any(v is None for v in vals.values()):
                continue
            other = min(v for k, v in vals.items() if k != kt)
            sel = select(vals)
            for k, v in vals.items():
                by_ds.append(dict(cell=cell, dataset_id=ds, k=k, iteration=it, cq_bar=v,
                                  cq_bar_minus_ktrue=v - vals[kt],
                                  G=vals[kt] - other, selected_k=sel))
    grp = defaultdict(list)
    for r in by_ds:
        grp[(r["cell"], r["k"], r["iteration"])].append(r)
    med = [dict(cell=c, k=k, iteration=it, n_datasets=len(rs),
                median_cq_bar=float(np.median([r["cq_bar"] for r in rs])),
                median_cq_bar_minus_ktrue=float(np.median([r["cq_bar_minus_ktrue"]
                                                           for r in rs])),
                median_G=float(np.median([r["G"] for r in rs])))
           for (c, k, it), rs in sorted(grp.items())]
    imp = []
    for ds, (cell, kt) in sorted(meta.items()):
        for k in kd.K_CANDIDATES:
            v = {it: bar.get((ds, k, it)) for it in its}
            if any(x is None for x in v.values()):
                continue
            imp.append(dict(cell=cell, dataset_id=ds, k=k,
                            impr_10_to_20=v[10] - v[20], impr_20_to_40=v[20] - v[40],
                            impr_10_to_40=v[10] - v[40],
                            rate_last3_at10=(v[7] - v[10]) / 3.0,
                            rate_last3_at20=(v[17] - v[20]) / 3.0,
                            rate_last3_at40=(v[37] - v[40]) / 3.0))
    g2 = defaultdict(list)
    for r in imp:
        g2[(r["cell"], r["k"])].append(r)
    cols = ("impr_10_to_20", "impr_20_to_40", "impr_10_to_40", "rate_last3_at10",
            "rate_last3_at20", "rate_last3_at40")
    imp_med = [dict(cell=c, k=k, n_datasets=len(rs),
                    **{f"median_{x}": float(np.median([r[x] for r in rs])) for x in cols})
               for (c, k), rs in sorted(g2.items())]
    return {"by_dataset": by_ds, "median": med, "improvement": imp,
            "improvement_median": imp_med}


def make_figures(out: Path, summ, rows, traj):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8984", "text.color": "#0b0b0b",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    order = ("B-b-9", "O-b-9", "A-b-9")
    cells = [c for c in order if any(s["cell"] == c for s in summ)]
    cells += sorted({s["cell"] for s in summ} - set(cells))
    color = {c: CELL_COLORS.get(c, "#6250d6") for c in cells}
    # 1: exact count by checkpoint (95% CP CI)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for c in cells:
        ss = sorted([s for s in summ if s["cell"] == c and s["n_complete"]],
                    key=lambda s: s["checkpoint"])
        if not ss:
            continue
        x = [s["checkpoint"] for s in ss]
        y = [s["exact"] for s in ss]
        lo = [s["exact"] - s["exact_ci95_lo"] * s["n_complete"] for s in ss]
        hi = [s["exact_ci95_hi"] * s["n_complete"] - s["exact"] for s in ss]
        ax.errorbar(x, y, yerr=[lo, hi], marker="o", ms=6, lw=2, capsize=3,
                    color=color[c], label=c)
        ax.annotate(c, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color="#52514e")
    ax.set_xticks(list(CHECKPOINTS))
    ax.set_xlabel("MCEM iteration (checkpoint)")
    ax.set_ylabel("datasets with selected K = K_TRUE")
    ax.set_ylim(-0.5, 10.5)
    ax.set_xlim(6, 50)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(1.0, 0.0))
    ax.set_title("C_Q selection = K_TRUE (mean of 2 inits; 95% CP CI)")
    fig.tight_layout()
    fig.savefig(out / "fig_exact_by_checkpoint.png", dpi=150)
    plt.close(fig)
    bd = traj["by_dataset"]
    # 2: per-dataset G(t)
    fig, axs = plt.subplots(1, len(cells), figsize=(4 * len(cells), 3.4), squeeze=False)
    for ax, c in zip(axs[0], cells):
        g = defaultdict(dict)
        for r in bd:
            if r["cell"] == c and r["k"] == 1:
                g[r["dataset_id"]][r["iteration"]] = r["G"]
        for ds, d in sorted(g.items()):
            it = sorted(d)
            ax.plot(it, [d[i] for i in it], lw=1.2, color=color[c], alpha=0.8)
        ax.axhline(0, color="#52514e", lw=1)
        for cp in CHECKPOINTS:
            ax.axvline(cp, color="#b9b8b2", lw=0.8, ls=":")
        ax.set_yscale("symlog", linthresh=50)
        ax.set_title(f"{c}: G(t), one line per dataset")
        ax.set_xlabel("MCEM iteration")
    axs[0][0].set_ylabel("G(t) = C̄_Q(K_TRUE) − min other C̄_Q\n(< 0: K_TRUE selected)")
    fig.tight_layout()
    fig.savefig(out / "fig_gap_trajectories.png", dpi=150)
    plt.close(fig)
    # 3: median C_Q_bar(K) - C_Q_bar(K_TRUE) per K (ordinal blue ramp)
    med = traj["median"]
    ramp = ["#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#104281"]
    fig, axs = plt.subplots(1, len(cells), figsize=(4.3 * len(cells), 3.6), squeeze=False)
    for ax, c in zip(axs[0], cells):
        kt = int(c.split("-")[2])
        for k in kd.K_CANDIDATES:
            if k == kt:
                continue
            rs = [r for r in med if r["cell"] == c and r["k"] == k]
            if not rs:
                continue
            ax.plot([r["iteration"] for r in rs], [r["median_cq_bar_minus_ktrue"] for r in rs],
                    lw=1.6, color=ramp[k - 1], ls="-" if k < kt else "--", label=f"K={k}")
            ax.annotate(str(k), (rs[-1]["iteration"], rs[-1]["median_cq_bar_minus_ktrue"]),
                        xytext=(3, 0), textcoords="offset points", fontsize=7,
                        va="center", color="#52514e")
        ax.axhline(0, color="#52514e", lw=1)
        ax.set_yscale("symlog", linthresh=50)
        ax.set_title(f"{c}: median C̄_Q(K) − C̄_Q(K_TRUE)")
        ax.set_xlabel("MCEM iteration")
    axs[0][0].set_ylabel("< 0: K beats K_TRUE")
    axs[0][-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(out / "fig_cq_trajectories.png", dpi=150)
    plt.close(fig)
    # 4: per-init selected K by checkpoint
    fig, axs = plt.subplots(1, len(cells), figsize=(4 * len(cells), 3.4), sharey=True,
                            squeeze=False)
    for ax, c in zip(axs[0], cells):
        kt = int(c.split("-")[2])
        rs = [r for r in rows if _cell(r) == c and r["outcome"] != "INCOMPLETE"]
        for j, cp in enumerate(CHECKPOINTS):
            sub = sorted([r for r in rs if r["checkpoint"] == cp], key=lambda r: r["rep"])
            for i, r in enumerate(sub):
                x0 = j * 12 + i
                ax.plot([x0, x0], [r["init0_k"], r["init1_k"]], color="#b9b8b2", lw=1)
                ax.plot(x0, r["init0_k"], "o", ms=4, mfc="white", mec=color[c])
                ax.plot(x0, r["init1_k"], "s", ms=4, mfc="white", mec=color[c])
                ax.plot(x0, r["selected_k"], "D", ms=4, color=color[c])
        ax.set_xticks([4.5, 16.5, 28.5], [f"t={cp}" for cp in CHECKPOINTS])
        ax.axhline(kt, color="#52514e", lw=0.8, ls=":")
        ax.set_title(f"{c}: selected K per dataset")
        ax.set_yticks(range(1, 11))
    axs[0][0].set_ylabel("selected K (◆ mean, ○ init 0, □ init 1)")
    fig.tight_layout()
    fig.savefig(out / "fig_init_dependence.png", dpi=150)
    plt.close(fig)


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
        st = analyze(ROOT / a.analyze, ROOT / a.out)
        print(json.dumps(st, indent=2))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
