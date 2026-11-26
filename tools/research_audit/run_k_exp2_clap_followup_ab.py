"""Follow-up numerical-sensitivity experiments A and B for C_Lap (no MCEM).

EXPLORATORY, post hoc (planned 2026-11-26 after the frozen 6,600-evaluation
C_Q-vs-C_Lap comparison, commit 0fb5824). Nothing of that run is changed: its
protocol, manifest, evaluations.csv and analyses are read only. The frozen
evaluator code (run_k_exp2_clap_comparison.py, laplace_k_criterion.py) is imported
and called unchanged.

Experiment A (start-point sensitivity): for A-b-9, A-b-7, B-b-9, O-b-9 (main10) and
A-b-9, B-b-9, O-b-9 (conv40), every candidate K, every dataset, both inits:
C_Lap from Z0 = Z_mean (the saved mean of the L E-step samples). theta_hat, N, d_K,
grad_tol 1e-8, max_iter 200 are identical to the Z_last baseline (official rows).
No "best of the two" is ever formed.

Experiment B (iteration-cap sensitivity): the 52 NOT_STATIONARY evaluations are
first reproduced with max_iter 200 (must equal the official record bitwise) and
then evaluated ONCE with max_iter 400 (Z_last, same theta_hat, same grad_tol 1e-8).
The 5 HESSIAN_NOT_PD evaluations are reproduced and characterised (eigenvalues at
the returned point); they are never rescued. No ridge/jitter, seed change, tolerance
relaxation, candidate exclusion or per-dataset rule.

Modes:
  --freeze A|B OUT                 protocol.json + manifest.csv (before any result)
  --tech-b OUT                     max_iter 400 == 200 on converged evaluations
  --execute A|B OUT --protocol P
  --analyze A|B RUN --out OUT
Lineage E prototype; C_Lap is a Laplace approximation, not an exact marginal
likelihood; C_Q is not Schwarz BIC.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import statistics as stats  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_k_exp2_clap_comparison as rc  # noqa: E402  (frozen; read/called only)

ROOT = rc.ROOT
CC = rc.BASE / "clap_comparison"
OFFICIAL_PROTOCOL = CC / "protocol" / "protocol.json"
OFFICIAL_PROTOCOL_SHA256 = "ae62f0413ec0b2ad5196bdd945847808fe3defe2f687fcbcd0dc103050f53432"
OFFICIAL_EV = CC / "run" / "evaluations.csv"
OFFICIAL_SEL = CC / "analysis" / "selection_by_dataset.csv"
A_CELLS = {"main10": ("A-b-9", "A-b-7", "B-b-9", "O-b-9"),
           "conv40": ("A-b-9", "B-b-9", "O-b-9")}
B_MAX_ITER = 400
WORKERS = 4
SAME_MODE_TOL = 1e-6
CODE_FILES = rc.CODE_FILES + ("tools/research_audit/run_k_exp2_clap_followup_ab.py",)
need, Stop = rc.need, rc.Stop


def cell(r):
    return f'{r["series"]}-{r["regime"]}-{r["k_true"]}'


def code_hashes():
    return {f: rc.sha256_file(ROOT / f) for f in CODE_FILES}


def manifest_rows():
    rows = rc.read_csv(OFFICIAL_PROTOCOL.parent / "manifest.csv")
    for r in rows:
        for k in ("k_true", "rep", "k", "t", "data_seed", "num_params"):
            r[k] = int(r[k])
    return rows


def official_rows():
    return {(r["source"], r["fit_id"]): r for r in rc.read_csv(OFFICIAL_EV)}


# --------------------------------------------------------------------------
# freeze
# --------------------------------------------------------------------------

def freeze(exp: str, out: Path):
    need(not out.exists(), f"output exists: {out}")
    need(rc.sha256_file(OFFICIAL_PROTOCOL) == OFFICIAL_PROTOCOL_SHA256, "official protocol")
    man = manifest_rows()
    off = official_rows()
    if exp == "A":
        sel = [r for r in man if cell(r) in A_CELLS[r["source"]]]
        need(len(sel) == 1400, f"A size {len(sel)}")
    else:
        sel = [r for r in man if off[(r["source"], r["fit_id"])]["lap_status"] != "OK"]
        need(len(sel) == 57, f"B size {len(sel)}")
    out.mkdir(parents=True)
    rc.write_csv(out / "manifest.csv", sel, rc.MANIFEST_FIELDS)
    common = {
        "frozen_at": rc.now(), "git_head_at_freeze": rc.git("rev-parse", "HEAD"),
        "status": "FROZEN_NOT_EXECUTED",
        "nature": "EXPLORATORY numerical-sensitivity analysis, planned after the "
                  "6,600-evaluation C_Q-vs-C_Lap results were seen; not confirmatory",
        "official_run": {"protocol_sha256": OFFICIAL_PROTOCOL_SHA256,
                         "evaluations_sha256": rc.sha256_file(OFFICIAL_EV),
                         "selection_sha256": rc.sha256_file(OFFICIAL_SEL),
                         "commit": "0fb5824", "unchanged": True},
        "inputs_sha256": {k: rc.sha256_file(p) for k, p in rc.INPUTS.items()},
        "code_sha256": code_hashes(), "n_evaluations": len(sel),
        "manifest_sha256": rc.manifest_hash(sel), "workers": WORKERS, "blas_threads": 1,
        "evaluator": "laplace_k_criterion v1 via run_k_exp2_clap_comparison.evaluate_state "
                     "(unchanged); N = n = 150; d_K = Kd - K(K-1)/2; grad_tol 1e-8",
        "forbidden": ["ridge/jitter", "seed change", "tolerance relaxation",
                      "candidate-K exclusion", "per-dataset success rules",
                      "choosing the better of two runs", "overwriting official results"],
        "failure_rule": "status recorded once (OK / NOT_STATIONARY / HESSIAN_NOT_PD / "
                        "ERROR); a dataset with any non-OK evaluation is INCOMPLETE under "
                        "the rule being analysed; denominator stays 10",
        "selection_rule": "argmin_K of the 2-init unweighted mean; ties <= 1e-9 -> smaller K",
    }
    if exp == "A":
        proto = {**common, "experiment_id": "k_exp2_clap_followup_A_start_point",
                 "question": "is C_Lap (and its K selection) stable when the joint-mode "
                             "search starts from Z_mean instead of Z_last?",
                 "cells": A_CELLS, "Z0": "Z_mean (saved mean of the L E-step samples)",
                 "baseline": "official Z_last rows (evaluations.csv), not recomputed",
                 "max_iter": rc.MAX_ITER,
                 "metrics": ["d C_Lap (Z_mean - Z_last)", "d Phi at mode", "grad_inf",
                             "min eig H", "same-mode flag |dPhi|,|dlogdet| <= 1e-6",
                             "selection agreement", "A-b-9 ranks of K=7,8,9",
                             "C_Lap(8)-C_Lap(9) under each start",
                             "INCOMPLETE under each start"],
                 "interpretation": ["the two starts are reported side by side; neither is "
                                    "called better; disagreement is evidence of mode "
                                    "non-uniqueness / start dependence of C_Lap",
                                    "no combined (best-of) selection is formed"]}
    else:
        proto = {**common, "experiment_id": "k_exp2_clap_followup_B_iteration_cap",
                 "question": "are the C_Lap INCOMPLETE datasets due to the Newton "
                             "iteration cap or to a deeper numerical instability?",
                 "targets": "the 52 NOT_STATIONARY and 5 HESSIAN_NOT_PD official rows",
                 "procedure": ["reproduce each target with max_iter 200 (Z_last): status, "
                               "grad_inf and Phi must equal the official record bitwise "
                               "(STOP otherwise)",
                               "NOT_STATIONARY: evaluate ONCE with max_iter 400, Z_last, "
                               "grad_tol 1e-8; record status/value",
                               "HESSIAN_NOT_PD: no rescue; record the eigen-spectrum of H at "
                               "the returned point (n negative, 5 smallest)"],
                 "tech_check": "max_iter 400 leaves evaluations that converged within 200 "
                               "iterations bitwise unchanged (smoke states + 40 official OK "
                               "rows); hence the 'B rule' equals a uniform max_iter = 400 "
                               "rule applied to every evaluation",
                 "max_iter": B_MAX_ITER,
                 "analysis": ["converged / still failing counts", "INCOMPLETE datasets under "
                              "the B rule vs official", "failures at K > K_TRUE",
                              "exact/under/over under the B rule (reported next to, never "
                              "replacing, the official counts)",
                              "selection bias: K_TRUE and C_Q-outcome mix of complete vs "
                              "INCOMPLETE datasets"]}
    rc.write_json(out / "protocol.json", proto)
    print("protocol sha256", rc.sha256_file(out / "protocol.json"))


# --------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------

def job_A(job):
    table = rc.bern_table()
    ds = rc.regenerate(job["series"], job["regime"], job["k_true"], job["rep"],
                       job["data_seed"], table)
    need(rc.kd.array_hash(ds.X) == job["x_hash"] and rc.kd.array_hash(ds.Y) == job["y_hash"],
         f"data {job['dataset_id']}")
    rows = []
    for e in job["evals"]:
        path = rc.STATE_DIRS[job["source"]] / e["state_file"]
        need(rc.sha256_file(path) == e["state_sha256"], f"sha {path.name}")
        st = rc.load_state(path, e["k"])
        Z0 = st["Z_mean"]
        model = rc.restore_model(job["series"], e["k"], st, Z0)
        r = {"source": job["source"], "fit_id": e["fit_id"], "dataset_id": job["dataset_id"],
             "series": job["series"], "regime": job["regime"], "k_true": job["k_true"],
             "rep": job["rep"], "k": e["k"], "t": e["t"], "start": "Z_mean"}
        r.update(rc.evaluate_state(model, ds.X, ds.Y, Z0, e["k"]))
        rows.append(r)
    return rows


def _eig_summary(model, X, Y, Z0, max_iter):
    import laplace_k_criterion as lk
    res = lk.laplace_log_observed(model, X, Y, Z0, grad_tol=rc.GRAD_TOL, max_iter=max_iter)
    H = lk.negative_hessian(model, X, Y, res.Z_hat)
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    return {"n_negative_eig": int(np.sum(ev < 0)), "smallest_eigs": json.dumps(
        [float(x) for x in ev[:5]]), "nK": int(ev.size)}


def job_B(job):
    table = rc.bern_table()
    off = job["official"]
    ds = rc.regenerate(job["series"], job["regime"], job["k_true"], job["rep"],
                       job["data_seed"], table)
    need(rc.kd.array_hash(ds.X) == job["x_hash"] and rc.kd.array_hash(ds.Y) == job["y_hash"],
         f"data {job['dataset_id']}")
    path = rc.STATE_DIRS[job["source"]] / job["state_file"]
    need(rc.sha256_file(path) == job["state_sha256"], f"sha {path.name}")
    st = rc.load_state(path, job["k"])
    Z0 = st["Z_last"]
    rep200 = rc.evaluate_state(rc.restore_model(job["series"], job["k"], st, Z0),
                               ds.X, ds.Y, Z0, job["k"], max_iter=rc.MAX_ITER)
    same = all(str(rep200.get(c, "")) == str(off[c]) for c in
               ("lap_status", "lap_grad_inf", "lap_phi_at_mode", "lap_logdet_H",
                "lap_iterations"))
    need(same, f"reproduction at max_iter 200 differs from official {job['fit_id']}")
    row = {k: job[k] for k in ("source", "fit_id", "dataset_id", "series", "regime",
                               "k_true", "rep", "k", "t")}
    row.update({"official_status": off["lap_status"], "official_grad_inf": off["lap_grad_inf"],
                "official_min_eig_H": off["lap_min_eig_H"], "repro200_equal": True})
    if off["lap_status"] == "NOT_STATIONARY":
        e400 = rc.evaluate_state(rc.restore_model(job["series"], job["k"], st, Z0),
                                 ds.X, ds.Y, Z0, job["k"], max_iter=B_MAX_ITER)
        row.update({f"it400_{c}": e400.get(c, "") for c in (
            "lap_status", "C_Lap", "lap_grad_inf", "lap_min_eig_H", "lap_iterations",
            "lap_phi_at_mode", "lap_logdet_H", "lap_zhat_sq", "lap_fit_mode", "lap_penalty",
            "lap_log_observed", "lap_error", "lap_seconds")})
    else:
        row.update(_eig_summary(rc.restore_model(job["series"], job["k"], st, Z0),
                                ds.X, ds.Y, Z0, rc.MAX_ITER))
    return [row]


def make_jobs(exp, man, off):
    if exp == "B":
        jobs = []
        for r in man:
            j = dict(r)
            j["official"] = off[(r["source"], r["fit_id"])]
            jobs.append(j)
        return jobs
    by = {}
    for r in man:
        key = (r["source"], r["dataset_id"])
        j = by.setdefault(key, {k: r[k] for k in ("source", "dataset_id", "series", "regime",
                                                 "k_true", "rep", "data_seed", "x_hash",
                                                 "y_hash")} | {"evals": []})
        j["evals"].append({k: r[k] for k in ("fit_id", "k", "t", "state_file",
                                             "state_sha256")})
    return [by[k] for k in sorted(by)]


def execute(exp: str, proto_path: Path, out: Path):
    from concurrent.futures import ProcessPoolExecutor
    import psutil
    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    need(proto["code_sha256"] == code_hashes(), "code sha256")
    need(not rc.git("status", "--porcelain"), "working tree must be clean")
    need(rc.sha256_file(OFFICIAL_EV) == proto["official_run"]["evaluations_sha256"],
         "official evaluations changed")
    man = rc.read_csv(proto_path.parent / "manifest.csv")
    for r in man:
        for k in ("k_true", "rep", "k", "t", "data_seed", "num_params"):
            r[k] = int(r[k])
    need(rc.manifest_hash(man) == proto["manifest_sha256"], "manifest sha256")
    need(not out.exists(), f"output exists: {out}")
    need(psutil.virtual_memory().available / 2 ** 20 > 3000, "available memory < 3 GB")
    out.mkdir(parents=True)
    jobs = make_jobs(exp, man, official_rows())
    t0 = time.time()
    rc.append_jsonl(out / "run_events.jsonl", {"event": "start", "time": rc.now(),
                                               "git_head": rc.git("rev-parse", "HEAD"),
                                               "protocol_sha256": rc.sha256_file(proto_path),
                                               "jobs": len(jobs), "workers": WORKERS})
    rows = []
    fn = job_A if exp == "A" else job_B
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for res in ex.map(fn, jobs):
            rows.extend(res)
    rc.write_csv(out / "per_evaluation.csv", rows)
    rc.append_jsonl(out / "run_events.jsonl", {"event": "finished", "time": rc.now(),
                                               "rows": len(rows),
                                               "wall_s": round(time.time() - t0, 1)})
    rc.write_json(out / "runinfo.json", {"protocol": str(proto_path.relative_to(ROOT)),
                                         "environment": rc.environment(), "rows": len(rows),
                                         "wall_s": round(time.time() - t0, 1)})
    print(f"{exp}: {len(rows)} rows in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------
# technical check for B (no new science: converged evaluations must not change)
# --------------------------------------------------------------------------

def _tech_worker(args):
    kind, payload = args
    if kind == "smoke":
        r = payload
        table = rc.bern_table()
        k = int(r["k"])
        st = rc.load_state(rc.SMOKE_DIR / "smoke_states" / r["state_file"], k)
        d = rc.regenerate(r["series"], r["regime"], int(r["k_true"]), int(r["rep"]),
                          int(r["data_seed"]), table)
        a = rc.evaluate_state(rc.restore_model(r["series"], k, st, st["Z_last"]), d.X, d.Y,
                              st["Z_last"], k, max_iter=200)
        b = rc.evaluate_state(rc.restore_model(r["series"], k, st, st["Z_last"]), d.X, d.Y,
                              st["Z_last"], k, max_iter=400)
        return {"id": r["fit_id"], "status200": a["lap_status"], "status400": b["lap_status"],
                "equal": all(a.get(c) == b.get(c) for c in ("lap_status", "C_Lap",
                             "lap_iterations", "lap_grad_inf", "lap_logdet_H"))}
    r = payload
    table = rc.bern_table()
    k = int(r["k"])
    st = rc.load_state(rc.STATE_DIRS[r["source"]] / r["state_file"], k)
    d = rc.regenerate(r["series"], r["regime"], int(r["k_true"]), int(r["rep"]),
                      int(r["data_seed"]), table)
    b = rc.evaluate_state(rc.restore_model(r["series"], k, st, st["Z_last"]), d.X, d.Y,
                          st["Z_last"], k, max_iter=400)
    off = r["official"]
    return {"id": f'{r["source"]}:{r["fit_id"]}', "status200": off["lap_status"],
            "status400": b["lap_status"],
            "equal": all(str(b.get(c)) == str(off[c]) for c in (
                "lap_status", "C_Lap", "lap_iterations", "lap_grad_inf", "lap_logdet_H"))}


def tech_b(out: Path):
    from concurrent.futures import ProcessPoolExecutor
    need(not out.exists(), f"output exists: {out}")
    smoke = [r for r in rc.read_csv(rc.SMOKE_DIR / "per_fit.csv")
             if r["series"] in rc.SERIES_IN and r["status"] == "OK"]
    off = official_rows()
    man = {(r["source"], r["fit_id"]): r for r in manifest_rows()}
    ok_keys = sorted(k for k, v in off.items() if v["lap_status"] == "OK")
    sample = ok_keys[::max(1, len(ok_keys) // 40)][:40]          # deterministic stride
    args = [("smoke", r) for r in smoke] + [
        ("main", {**man[k], "official": off[k]}) for k in sample]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(_tech_worker, args))
    out.mkdir(parents=True)
    rc.write_csv(out / "tech_check_rows.csv", res)
    checks = {"max_iter400_equals_200_on_smoke_states": all(r["equal"] for r in res[:len(smoke)]),
              "max_iter400_equals_official_on_40_OK_rows": all(r["equal"] for r in res[len(smoke):]),
              "n_smoke": len(smoke), "n_official_sample": len(sample)}
    checks["all_ok"] = (checks["max_iter400_equals_200_on_smoke_states"]
                        and checks["max_iter400_equals_official_on_40_OK_rows"])
    rc.write_json(out / "tech_check.json", {"created": rc.now(), "checks": checks,
                                            "git_head": rc.git("rev-parse", "HEAD")})
    print(json.dumps(checks, indent=2))


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def _select(f, col):
    ks = range(1, 11)
    if any(f[(k, t)]["status"] != "OK" for k in ks for t in (0, 1)):
        return None, {}
    bar = {k: (float(f[(k, 0)][col]) + float(f[(k, 1)][col])) / 2.0 for k in ks}
    return rc.pick(bar), bar


def _rank(bar, k):
    order = sorted(bar, key=lambda x: (bar[x], x))
    return order.index(k) + 1


def analyze_A(run: Path, out: Path):
    need(not out.exists(), f"output exists: {out}")
    off = official_rows()
    rows = rc.read_csv(run / "per_evaluation.csv")
    need(len(rows) == 1400, "A rows")
    per = []
    for r in rows:
        o = off[(r["source"], r["fit_id"])]
        both = r["lap_status"] == "OK" and o["lap_status"] == "OK"
        dphi = float(r["lap_phi_at_mode"]) - float(o["lap_phi_at_mode"]) \
            if r.get("lap_phi_at_mode") and o["lap_phi_at_mode"] else float("nan")
        dld = float(r["lap_logdet_H"]) - float(o["lap_logdet_H"]) \
            if r.get("lap_logdet_H") and o["lap_logdet_H"] else float("nan")
        per.append({**{k: r[k] for k in ("source", "fit_id", "dataset_id", "series", "regime",
                                         "k_true", "rep", "k", "t")},
                    "status_Zlast": o["lap_status"], "status_Zmean": r["lap_status"],
                    "C_Lap_Zlast": o["C_Lap"], "C_Lap_Zmean": r["C_Lap"],
                    "d_C_Lap": (float(r["C_Lap"]) - float(o["C_Lap"])) if both else "",
                    "d_phi_at_mode": dphi, "d_logdet_H": dld,
                    "same_mode": bool(abs(dphi) <= SAME_MODE_TOL and abs(dld) <= SAME_MODE_TOL),
                    "grad_inf_Zlast": o["lap_grad_inf"], "grad_inf_Zmean": r.get("lap_grad_inf"),
                    "min_eig_Zlast": o["lap_min_eig_H"], "min_eig_Zmean": r.get("lap_min_eig_H"),
                    "iter_Zlast": o["lap_iterations"], "iter_Zmean": r.get("lap_iterations")})
    g = defaultdict(dict)
    for p in per:
        g[(p["source"], p["dataset_id"])][(int(p["k"]), int(p["t"]))] = p
    sel, curves = [], defaultdict(lambda: defaultdict(list))
    for (src, ds), f in sorted(g.items()):
        p0 = f[(1, 0)]
        kt = int(p0["k_true"])
        fl = {k: {"status": v["status_Zlast"], "c": v["C_Lap_Zlast"]} for k, v in f.items()}
        fm = {k: {"status": v["status_Zmean"], "c": v["C_Lap_Zmean"]} for k, v in f.items()}
        kl, bl = _select(fl, "c")
        km, bm = _select(fm, "c")
        row = {"source": src, "dataset_id": ds, "cell": cell(p0), "k_true": kt, "rep": p0["rep"],
               "k_Lap_Zlast": kl if kl else "INCOMPLETE", "k_Lap_Zmean": km if km else "INCOMPLETE",
               "agree": (kl == km) if (kl and km) else "",
               "n_same_mode_evals": sum(v["same_mode"] for v in f.values()),
               "max_abs_d_C_Lap": max([abs(v["d_C_Lap"]) for v in f.values()
                                       if v["d_C_Lap"] != ""], default="")}
        for lab, bar in (("Zlast", bl), ("Zmean", bm)):
            if bar:
                for k in (7, 8, 9):
                    row[f"rank_K{k}_{lab}"] = _rank(bar, k)
                row[f"gap_8_minus_9_{lab}"] = bar[8] - bar[9]
                for k in range(1, 11):
                    curves[(src, cell(p0), k)][lab].append(bar[k] - bar[kt])
        sel.append(row)
    summ = []
    by = defaultdict(list)
    for s in sel:
        by[(s["source"], s["cell"])].append(s)
    pc = defaultdict(list)
    for p in per:
        pc[(p["source"], cell(p))].append(p)
    for (src, c), v in sorted(by.items()):
        kt = v[0]["k_true"]
        pv = pc[(src, c)]
        dd = [abs(p["d_C_Lap"]) for p in pv if p["d_C_Lap"] != ""]
        summ.append({"source": src, "cell": c, "n_datasets": len(v),
                     "exact_Zlast": sum(s["k_Lap_Zlast"] == kt for s in v),
                     "exact_Zmean": sum(s["k_Lap_Zmean"] == kt for s in v),
                     "incomplete_Zlast": sum(s["k_Lap_Zlast"] == "INCOMPLETE" for s in v),
                     "incomplete_Zmean": sum(s["k_Lap_Zmean"] == "INCOMPLETE" for s in v),
                     "under_Zlast": sum(isinstance(s["k_Lap_Zlast"], int) and s["k_Lap_Zlast"] < kt
                                        for s in v),
                     "under_Zmean": sum(isinstance(s["k_Lap_Zmean"], int) and s["k_Lap_Zmean"] < kt
                                        for s in v),
                     "over_Zlast": sum(isinstance(s["k_Lap_Zlast"], int) and s["k_Lap_Zlast"] > kt
                                       for s in v),
                     "over_Zmean": sum(isinstance(s["k_Lap_Zmean"], int) and s["k_Lap_Zmean"] > kt
                                       for s in v),
                     "selection_agree": sum(s["agree"] is True for s in v),
                     "both_complete": sum(s["agree"] != "" for s in v),
                     "evals": len(pv), "evals_same_mode": sum(p["same_mode"] for p in pv),
                     "evals_both_OK": len(dd),
                     "median_abs_d_C_Lap": stats.median(dd) if dd else "",
                     "max_abs_d_C_Lap": max(dd) if dd else "",
                     "status_pairs": json.dumps(dict(Counter(
                         f'{p["status_Zlast"]}->{p["status_Zmean"]}' for p in pv)))})
    curve_rows = [{"source": a, "cell": b, "k": k,
                   "median_CLap_minus_ktrue_Zlast": stats.median(d["Zlast"]) if d["Zlast"] else "",
                   "median_CLap_minus_ktrue_Zmean": stats.median(d["Zmean"]) if d["Zmean"] else ""}
                  for (a, b, k), d in sorted(curves.items())]
    out.mkdir(parents=True)
    rc.write_csv(out / "per_evaluation_compared.csv", per)
    rc.write_csv(out / "selection_by_dataset.csv", sel)
    rc.write_csv(out / "summary_by_cell.csv", summ)
    rc.write_csv(out / "criterion_curves.csv", curve_rows)
    checks = {"rows": len(per), "official_unchanged": rc.sha256_file(OFFICIAL_EV) == json.loads(
        (run.parent / "protocol" / "protocol.json").read_text(encoding="utf-8")
    )["official_run"]["evaluations_sha256"]}
    rc.write_json(out / "checks.json", checks)
    _fig_A(sel, out)
    print(json.dumps(checks))


def _fig_A(sel, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = [s for s in sel if s["cell"] == "A-b-9"]
    if not v:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, src in zip(axes, ("main10", "conv40")):
        w = sorted((s for s in v if s["source"] == src), key=lambda s: int(s["rep"]))
        x = np.arange(len(w))
        a = [s.get("gap_8_minus_9_Zlast", np.nan) for s in w]
        b = [s.get("gap_8_minus_9_Zmean", np.nan) for s in w]
        a = [np.nan if x_ == "" else x_ for x_ in a]
        b = [np.nan if x_ == "" else x_ for x_ in b]
        ax.bar(x - 0.2, a, 0.38, color="#2a78d6", label="start Z_last")
        ax.bar(x + 0.2, b, 0.38, color="#eb6834", label="start Z_mean")
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xticks(x, [f"r{s['rep']}" for s in w], fontsize=8)
        ax.set_title(f"A-b-9 ({src}): C_Lap(8) - C_Lap(9)", fontsize=9)
    axes[0].set_ylabel(">0: K=9 better")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_ab9_gap8_9_by_start.png", dpi=150)
    plt.close(fig)


def analyze_B(run: Path, out: Path):
    need(not out.exists(), f"output exists: {out}")
    off = official_rows()
    rows = rc.read_csv(run / "per_evaluation.csv")
    need(len(rows) == 57 and all(r["repro200_equal"] == "True" for r in rows), "B rows")
    b400 = {(r["source"], r["fit_id"]): r for r in rows}
    # B rule: every evaluation at max_iter 400 == official for converged rows (tech check),
    # it400 result for the NOT_STATIONARY rows, HESSIAN_NOT_PD stays (deterministic).
    g = defaultdict(dict)
    for key, o in off.items():
        st, c = o["lap_status"], o["C_Lap"]
        if key in b400 and o["lap_status"] == "NOT_STATIONARY":
            st, c = b400[key]["it400_lap_status"], b400[key]["it400_C_Lap"]
        g[(o["source"], o["dataset_id"])][(int(o["k"]), int(o["t"]))] = {
            "status": st, "c": c, "o_status": o["lap_status"], "o_c": o["C_Lap"],
            "C_Q": o["C_Q"], "k_true": int(o["k_true"]), "cell": cell(o)}
    sel = []
    for (src, ds), f in sorted(g.items()):
        p0 = f[(1, 0)]
        kt = p0["k_true"]
        ko, _ = _select({k: {"status": v["o_status"], "c": v["o_c"]} for k, v in f.items()}, "c")
        kb, _ = _select({k: {"status": v["status"], "c": v["c"]} for k, v in f.items()}, "c")
        kq, _ = _select({k: {"status": "OK", "c": v["C_Q"]} for k, v in f.items()}, "c")
        lab = lambda k: "INCOMPLETE" if k is None else ("exact" if k == kt else   # noqa: E731
                                                         ("under" if k < kt else "over"))
        sel.append({"source": src, "dataset_id": ds, "cell": p0["cell"], "k_true": kt,
                    "k_Q": kq, "outcome_Q": lab(kq),
                    "k_Lap_official": ko if ko else "INCOMPLETE", "outcome_official": lab(ko),
                    "k_Lap_Brule": kb if kb else "INCOMPLETE", "outcome_Brule": lab(kb)})
    summ = []
    by = defaultdict(list)
    for s in sel:
        by[(s["source"], s["cell"])].append(s)
    for (src, c), v in sorted(by.items()):
        co = Counter(s["outcome_official"] for s in v)
        cb = Counter(s["outcome_Brule"] for s in v)
        summ.append({"source": src, "cell": c, "n": len(v),
                     **{f"official_{k}": co[k] for k in ("exact", "under", "over", "INCOMPLETE")},
                     **{f"Brule_{k}": cb[k] for k in ("exact", "under", "over", "INCOMPLETE")},
                     "newly_complete": sum(1 for s in v if s["outcome_official"] == "INCOMPLETE"
                                           and s["outcome_Brule"] != "INCOMPLETE")})
    # selection-bias description
    bias = []
    for src in ("main10", "conv40"):
        for rule in ("official", "Brule"):
            v = [s for s in sel if s["source"] == src]
            comp = [s for s in v if s[f"outcome_{rule}"] != "INCOMPLETE"]
            inc = [s for s in v if s[f"outcome_{rule}"] == "INCOMPLETE"]
            for lab, grp in (("complete", comp), ("incomplete", inc)):
                bias.append({"source": src, "rule": rule, "group": lab, "n": len(grp),
                             "k_true_dist": json.dumps(dict(sorted(Counter(
                                 s["k_true"] for s in grp).items()))),
                             "C_Q_outcome_dist": json.dumps(dict(Counter(
                                 s["outcome_Q"] for s in grp)))})
    fail = Counter()
    for r in rows:
        st = r.get("it400_lap_status") or r["official_status"]
        fail[(r["official_status"], st, "K>K_TRUE" if int(r["k"]) > int(r["k_true"]) else
              ("K=K_TRUE" if r["k"] == r["k_true"] else "K<K_TRUE"))] += 1
    out.mkdir(parents=True)
    rc.write_csv(out / "selection_by_dataset.csv", sel)
    rc.write_csv(out / "summary_by_cell.csv", summ)
    rc.write_csv(out / "selection_bias.csv", bias)
    rc.write_csv(out / "failure_transitions.csv", [
        {"official": a, "after_B": b, "k_vs_ktrue": c, "count": n}
        for (a, b, c), n in sorted(fail.items())])
    tot = {src: {rule: dict(Counter(s[f"outcome_{rule}"] for s in sel if s["source"] == src))
                 for rule in ("official", "Brule")} for src in ("main10", "conv40")}
    checks = {"rows": len(rows), "all_repro200_equal": True,
              "NS_converged_at_400": sum(1 for r in rows if r["official_status"] ==
                                         "NOT_STATIONARY" and r["it400_lap_status"] == "OK"),
              "NS_total": sum(1 for r in rows if r["official_status"] == "NOT_STATIONARY"),
              "it400_status": dict(Counter(r.get("it400_lap_status") for r in rows
                                           if r["official_status"] == "NOT_STATIONARY")),
              "totals": tot}
    rc.write_json(out / "checks.json", checks)
    print(json.dumps(checks, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", nargs=2, metavar=("EXP", "OUT"))
    ap.add_argument("--tech-b", metavar="OUT")
    ap.add_argument("--execute", nargs=2, metavar=("EXP", "OUT"))
    ap.add_argument("--protocol")
    ap.add_argument("--analyze", nargs=2, metavar=("EXP", "RUN"))
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.freeze:
        return freeze(a.freeze[0], ROOT / a.freeze[1]) or 0
    if a.tech_b:
        return tech_b(ROOT / a.tech_b) or 0
    if a.execute:
        return execute(a.execute[0], ROOT / a.protocol, ROOT / a.execute[1]) or 0
    if a.analyze:
        fn = analyze_A if a.analyze[0] == "A" else analyze_B
        return fn(ROOT / a.analyze[1], ROOT / a.out) or 0
    ap.error("no mode")


if __name__ == "__main__":
    raise SystemExit(main())
