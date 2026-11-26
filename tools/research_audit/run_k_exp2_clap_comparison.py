"""C_Q vs C_Lap on the SAME saved fits of the K-selection Exp2-generalisation runs.

EXPLORATORY, post hoc. Planned on 2026-11-26 AFTER the 8,000-fit main run and the
600-fit convergence-sensitivity run were seen (C_Q under-selection in A-b-7, A-b-9,
B-b-9). No MCEM is run, no fit is changed, no C_Q value is recomputed or replaced.

Each evaluation is a deterministic function of one saved state (theta_hat and the
last E-step sample Z_last) and the regenerated observed data (X, Y):

    Phi(Z)   = ln p(Z) + ln p(X | Z, theta_hat) + ln p(Y | Z, theta_hat)   (strict constants)
    Z_hat    = joint mode of Phi, damped full Newton from Z0 = Z_last
    H        = -grad^2_Z Phi(Z_hat)   (nK x nK, no ridge / jitter)
    ln p_Lap = Phi(Z_hat) + (nK/2) ln 2 pi - 1/2 ln|H|            (status OK only)
    C_Lap(K) = -2 ln p_Lap + d_K ln N,   d_K = K d - K(K-1)/2,  N = n = 150

and with var_z = 1 the exact split
    C_Lap = fit_mode + ||Z_hat||^2 + ln|H| + d_K ln N,
    fit_mode = -2 [ln p(X|Z_hat) + ln p(Y|Z_hat)]  (= -2 (Phi - ln p(Z_hat))).

Sources (frozen in the protocol): main run (10 iterations; series O, A, B -> 6,000
states) and the convergence-sensitivity run at iteration 40 (600 states). Series C
(Gaussian Y) is outside the verified Laplace evaluator and is excluded.

Modes:
  --tech-check OUT                       technical verification on SMOKE states only
                                         (different data seeds; no main-data score)
  --freeze OUT                           protocol.json + manifest.csv
  --execute OUT --protocol P [--resume]  run a frozen protocol (NOT run in the prep task)
  --analyze RUN --out OUT                pre-specified analysis + figures

Lineage E (experimental prototype). MANUSCRIPT_ADOPTION_STATUS: NOT_DECIDED_BY_HUMAN.
C_Q is a Q-based complete-data (ICL-type) criterion, NOT Schwarz BIC. C_Lap is a
Laplace approximation, NOT an exact marginal likelihood.
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
import platform  # noqa: E402
import statistics as stats  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in ("expfam/src", "expfam/src/experimental", "reproduction/src"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

import k_exp2_design as kd  # noqa: E402

# --------------------------------------------------------------------------
# frozen settings
# --------------------------------------------------------------------------

BASE = ROOT / "expfam/results/k_selection/k_exp2_generalization_20261125"
MAIN_PROTOCOL = BASE / "protocol_v2" / "protocol.json"
MAIN_PROTOCOL_SHA256 = "14757d4a161712859601d52aba0e86308903a723f29baee0a6b3d5118ec5a7ad"
CONV_PROTOCOL = BASE / "convergence_sensitivity" / "protocol" / "protocol.json"
INPUTS = {   # read-only; sha256 recorded at freeze and re-checked at execute
    "main_per_fit": BASE / "main" / "per_fit.csv",
    "main_states_sha256_manifest": BASE / "main" / "states_sha256_manifest.csv",
    "main_selection_per_dataset": BASE / "analysis" / "selection_per_dataset.csv",
    "conv_per_fit": BASE / "convergence_sensitivity" / "run" / "per_fit.csv",
    "conv_checkpoint_scores": BASE / "convergence_sensitivity" / "run" / "checkpoint_scores.csv",
    "conv_states_sha256_manifest":
        BASE / "convergence_sensitivity" / "run" / "states_sha256_manifest.csv",
    "conv_selected_k_by_dataset":
        BASE / "convergence_sensitivity" / "analysis" / "selected_k_by_dataset.csv",
}
STATE_DIRS = {"main10": BASE / "main" / "states",
              "conv40": BASE / "convergence_sensitivity" / "run" / "states"}
SOURCE_CHECKPOINT = {"main10": 10, "conv40": 40}
SERIES_IN = ("O", "A", "B")
GRAD_TOL = 1e-8                 # Phase 9H defaults, unchanged
MAX_ITER = 200
N_PENALTY = kd.N                # N = n (working convention of Phase 9G)
Z0_RULE = "Z_last"              # last E-step sample of the saved state; nothing else
TIE_TOL = 1e-9
CLOSURE_RTOL = 1e-9
CLOSURE_ATOL = 1e-6
FOCUS_CELLS = ("A-b-9", "A-b-7", "B-b-9", "O-b-9")
PZ_PER_DIM = kd.N * (1.0 + math.log(2.0 * math.pi))
LN_N = math.log(kd.N)
SMOKE_DIR = BASE / "smoke"
CODE_FILES = (
    "tools/research_audit/run_k_exp2_clap_comparison.py",
    "expfam/src/experimental/laplace_k_criterion.py",
    "expfam/src/experimental/eval_utils.py",
    "expfam/src/experimental/em_runner.py",
    "expfam/src/experimental/k_exp2_design.py",
    "expfam/src/experimental/data_generator_canonical.py",
    "expfam/src/experimental/matched_k_true_design.py",
    "expfam/src/experimental/model_dual_expfam_consistent.py",
    "expfam/src/experimental/model_dual_expfam_masked.py",
    "expfam/src/model_dual_expfam_fixed.py",
    "expfam/src/model_dual_expfam.py",
    "expfam/src/model_expfam.py",
    "reproduction/src/model.py",
)
# filled in at freeze from the technical check (measured, recorded; never changes numerics)
DEFAULT_WORKERS = 4
MEM_STOP_MB = 1500              # stop dispatching new jobs if available memory stays below
MEM_STOP_CONSECUTIVE = 2

EVAL_FIELDS = [
    "source", "checkpoint", "fit_id", "dataset_id", "series", "regime", "k_true", "rep",
    "k", "t", "C_Q", "num_params", "state_file", "state_sha256_ok", "data_equal",
    "lap_status", "lap_error", "C_Lap", "lap_log_observed", "lap_phi_at_mode",
    "lap_fit_mode", "lap_zhat_sq", "lap_logdet_H", "lap_logdet_sign", "lap_min_eig_H",
    "lap_grad_inf", "lap_iterations", "lap_penalty", "d_K", "lap_closure_absdiff",
    "z_shift_rms", "lap_seconds", "worker_pid",
]
STATUS_OK = "OK"
STATUS_ERROR = "ERROR"          # exception inside the evaluator (recorded once)


class Stop(RuntimeError):
    """Input/provenance gate failed: stop the whole run (never a per-fit failure)."""


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise Stop(msg)


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


def write_csv(p: Path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def append_csv(p: Path, rows, fields):
    new = not p.exists()
    with open(p, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def write_json(p: Path, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def code_hashes() -> dict:
    return {f: sha256_file(ROOT / f) for f in CODE_FILES}


def environment() -> dict:
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
            "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS")}


def bern_table() -> dict:
    proto = json.loads(MAIN_PROTOCOL.read_text(encoding="utf-8"))
    return {(key.split("-")[0], int(c["k"])): c for key, c in proto["calibration"].items()}


# --------------------------------------------------------------------------
# data, state, model
# --------------------------------------------------------------------------

def regenerate(series, regime, k_true, rep, data_seed, table):
    spec = kd.DatasetSpec(series, regime, k_true, rep, kd.w_of(series, regime, k_true),
                          kd.w0_of(series, regime, k_true, table),
                          kd.f_scale_of(regime, k_true), data_seed)
    return kd.generate(spec)


STATE_KEYS = ("F", "sigma", "w0", "w", "var_z", "sigma_y", "Z_last", "Z_mean")


def load_state(path: Path, k: int) -> dict:
    with np.load(path, allow_pickle=False) as z:
        need(set(z.files) == set(STATE_KEYS), f"NPZ keys {path.name}: {z.files}")
        st = {key: np.array(z[key], dtype=np.float64) for key in STATE_KEYS}
    need(st["F"].shape == (kd.D, k), f"F shape {path.name}")
    need(st["sigma"].shape == (kd.D, kd.D), f"sigma shape {path.name}")
    need(st["Z_last"].shape == (kd.N, k) and st["Z_mean"].shape == (kd.N, k),
         f"Z shape {path.name}")
    need(all(np.all(np.isfinite(v)) for v in st.values()), f"non-finite state {path.name}")
    need(float(st["var_z"]) == 1.0, f"var_z != 1 {path.name}")
    return st


def restore_model(series: str, k: int, st: dict, Z0: np.ndarray):
    from em_runner import build_model
    fx, fy = kd.SERIES[series]
    model = build_model(kd.N, kd.D, k, kd.L_MC, fx, fy, numerics_mode="consistent")
    model.initialize_params(true_params=None, seed=0)   # creates the dict; all overwritten
    model.params.update({"F": st["F"].copy(), "sigma": st["sigma"].copy(),
                         "w0": float(st["w0"]), "w": float(st["w"]),
                         "var_z": float(st["var_z"]), "Z": Z0.copy()})
    return model


def evaluate_state(model, X, Y, Z0: np.ndarray, k: int, *, grad_tol=GRAD_TOL,
                   max_iter=MAX_ITER) -> dict:
    """One C_Lap evaluation. Never retried; any exception -> status ERROR."""
    import laplace_k_criterion as lk
    t0 = time.perf_counter()
    d_k = k * kd.D - k * (k - 1) // 2
    out = {"lap_status": "", "lap_error": "", "C_Lap": "", "d_K": d_k,
           "lap_penalty": repr(d_k * math.log(N_PENALTY))}
    try:
        res = lk.laplace_log_observed(model, X, Y, Z0, grad_tol=grad_tol, max_iter=max_iter)
    except Exception as exc:          # recorded once, never retried, no alternative start
        out.update(lap_status=STATUS_ERROR, lap_error=f"{type(exc).__name__}: {str(exc)[:200]}",
                   lap_seconds=round(time.perf_counter() - t0, 3))
        return out
    n = Z0.shape[0]
    zsq = float(np.sum(res.Z_hat ** 2))
    ln_prior = -(n * k / 2.0) * math.log(2.0 * math.pi) - 0.5 * zsq       # var_z = 1
    fit_mode = -2.0 * (res.phi_at_mode - ln_prior)
    out.update(lap_status=res.status, lap_phi_at_mode=repr(res.phi_at_mode),
               lap_fit_mode=repr(fit_mode), lap_zhat_sq=repr(zsq),
               lap_logdet_H=repr(res.logdet_H), lap_logdet_sign=res.logdet_sign,
               lap_min_eig_H=repr(res.min_eigenvalue_H), lap_grad_inf=repr(res.grad_inf),
               lap_iterations=res.iterations,
               z_shift_rms=repr(float(np.sqrt(np.mean((res.Z_hat - Z0) ** 2)))))
    if res.ok and not all(math.isfinite(v) for v in (res.log_observed, res.phi_at_mode,
                                                     res.logdet_H, res.grad_inf, zsq)):
        out.update(lap_status=STATUS_ERROR, lap_error="non-finite value with status OK",
                   lap_seconds=round(time.perf_counter() - t0, 3))
        return out
    if res.ok:
        c_lap = -2.0 * res.log_observed + d_k * math.log(N_PENALTY)
        recomposed = fit_mode + zsq + res.logdet_H + d_k * math.log(N_PENALTY)
        diff = abs(recomposed - c_lap)
        need(diff <= CLOSURE_ATOL + CLOSURE_RTOL * abs(c_lap),
             f"C_Lap closure {diff!r}")               # code bug -> STOP, not a fit failure
        out.update(C_Lap=repr(c_lap), lap_log_observed=repr(res.log_observed),
                   lap_closure_absdiff=repr(diff))
    out["lap_seconds"] = round(time.perf_counter() - t0, 3)
    return out


def run_dataset_job(job: dict) -> list[dict]:
    """All 20 evaluations of one (source, dataset). Gates raise Stop."""
    table = bern_table()
    ds = regenerate(job["series"], job["regime"], job["k_true"], job["rep"],
                    job["data_seed"], table)
    X, Y = ds.X, ds.Y
    data_equal = (kd.array_hash(X) == job["x_hash"] and kd.array_hash(Y) == job["y_hash"])
    need(data_equal, f"regenerated data differ {job['dataset_id']}")
    rows = []
    for e in job["evals"]:
        path = STATE_DIRS[job["source"]] / e["state_file"]
        need(path.exists(), f"missing state {path}")
        need(sha256_file(path) == e["state_sha256"], f"state sha256 {path.name}")
        st = load_state(path, e["k"])
        Z0 = st["Z_last"]
        model = restore_model(job["series"], e["k"], st, Z0)
        row = {"source": job["source"], "checkpoint": SOURCE_CHECKPOINT[job["source"]],
               "dataset_id": job["dataset_id"], "series": job["series"],
               "regime": job["regime"], "k_true": job["k_true"], "rep": job["rep"],
               "fit_id": e["fit_id"], "k": e["k"], "t": e["t"], "C_Q": e["C_Q"],
               "num_params": e["num_params"], "state_file": e["state_file"],
               "state_sha256_ok": True, "data_equal": True, "worker_pid": os.getpid()}
        row.update(evaluate_state(model, X, Y, Z0, e["k"]))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# manifest / freeze
# --------------------------------------------------------------------------

def build_manifest() -> list[dict]:
    main = {r["fit_id"]: r for r in read_csv(INPUTS["main_per_fit"])}
    msha = {r["state_file"]: r["sha256"] for r in read_csv(INPUTS["main_states_sha256_manifest"])}
    conv = {r["fit_id"]: r for r in read_csv(INPUTS["conv_per_fit"])}
    csha = {r["state_file"]: r["sha256"] for r in read_csv(INPUTS["conv_states_sha256_manifest"])}
    cq40 = {r["fit_id"]: r for r in read_csv(INPUTS["conv_checkpoint_scores"])
            if r["checkpoint"] == "40"}
    rows = []
    for fid, r in sorted(main.items()):
        if r["series"] not in SERIES_IN:
            continue
        need(r["status"] == "OK", f"main fit not OK {fid}")
        need(msha[r["state_file"]].startswith(r["state_sha256"]), f"sha prefix {fid}")
        rows.append(dict(source="main10", fit_id=fid, dataset_id=r["dataset_id"],
                         series=r["series"], regime=r["regime"], k_true=int(r["k_true"]),
                         rep=int(r["rep"]), k=int(r["k"]), t=int(r["t"]),
                         data_seed=int(r["data_seed"]), x_hash=r["x_hash"], y_hash=r["y_hash"],
                         state_file=r["state_file"], state_sha256=msha[r["state_file"]],
                         C_Q=r["C_Q"], num_params=int(r["num_params"])))
    for fid, r in sorted(conv.items()):
        need(r["status"] == "OK", f"conv fit not OK {fid}")
        need(csha[r["state_file"]] == r["state_sha256"], f"conv sha {fid}")
        c = cq40[fid]
        rows.append(dict(source="conv40", fit_id=fid, dataset_id=r["dataset_id"],
                         series=r["series"], regime=r["regime"], k_true=int(r["k_true"]),
                         rep=int(r["rep"]), k=int(r["k"]), t=int(r["t"]),
                         data_seed=int(r["data_seed"]), x_hash=r["x_hash"], y_hash=r["y_hash"],
                         state_file=r["state_file"], state_sha256=r["state_sha256"],
                         C_Q=c["C_Q"], num_params=int(c["num_params"])))
    need(sum(r["source"] == "main10" for r in rows) == 6000, "main10 count")
    need(sum(r["source"] == "conv40" for r in rows) == 600, "conv40 count")
    return rows


def manifest_jobs(rows) -> list[dict]:
    jobs = {}
    for r in rows:
        key = (r["source"], r["dataset_id"])
        j = jobs.setdefault(key, {k: r[k] for k in ("source", "dataset_id", "series", "regime",
                                                   "k_true", "rep", "data_seed", "x_hash",
                                                   "y_hash")} | {"evals": []})
        need(all(j[k] == r[k] for k in ("data_seed", "x_hash", "y_hash")), f"data id {key}")
        j["evals"].append({k: r[k] for k in ("fit_id", "k", "t", "state_file", "state_sha256",
                                             "C_Q", "num_params")})
    for j in jobs.values():
        need(len(j["evals"]) == 20, f"job size {j['dataset_id']}")
        j["evals"].sort(key=lambda e: (e["k"], e["t"]))
    return [jobs[k] for k in sorted(jobs)]


MANIFEST_FIELDS = ["source", "fit_id", "dataset_id", "series", "regime", "k_true", "rep", "k",
                   "t", "data_seed", "x_hash", "y_hash", "state_file", "state_sha256", "C_Q",
                   "num_params"]


def manifest_hash(rows) -> str:
    return hashlib.sha256(json.dumps([{k: str(r[k]) for k in MANIFEST_FIELDS} for r in rows],
                                     sort_keys=True).encode()).hexdigest()


def freeze(out: Path, workers: int, tech_check: Path):
    need(not out.exists(), f"output exists: {out}")
    need(sha256_file(MAIN_PROTOCOL) == MAIN_PROTOCOL_SHA256, "main protocol sha256")
    tc = json.loads((tech_check / "tech_check.json").read_text(encoding="utf-8"))
    need(tc.get("all_ok") is True, "technical check not all OK")
    rows = build_manifest()
    out.mkdir(parents=True)
    write_csv(out / "manifest.csv", rows, MANIFEST_FIELDS)
    proto = {
        "experiment_id": "k_exp2_clap_comparison_20261126",
        "status": "FROZEN_NOT_EXECUTED",
        "frozen_at": now(), "git_head_at_freeze": git("rev-parse", "HEAD"),
        "nature": ("EXPLORATORY post-hoc comparison planned after the 8,000-fit and 600-fit "
                   "C_Q results were seen; no MCEM; same saved fits for both criteria"),
        "lineage": "E scalar (DualExpFamLSMConsistent, numerics_mode=consistent); not adopted "
                   "for manuscript (Human Gate)",
        "sources": {"main10": "main run, iteration-10 states, series O/A/B (6,000)",
                    "conv40": "convergence-sensitivity run, iteration-40 states (600)"},
        "excluded": "series C (Gaussian Y): outside the verified Laplace evaluator",
        "n_evaluations": len(rows), "n_dataset_jobs": len(manifest_jobs(rows)),
        "manifest_sha256": manifest_hash(rows),
        "input_sha256": {k: sha256_file(p) for k, p in INPUTS.items()},
        "main_protocol_sha256": MAIN_PROTOCOL_SHA256,
        "conv_protocol_sha256": sha256_file(CONV_PROTOCOL),
        "code_sha256": code_hashes(),
        "criterion": {
            "C_Lap": "C_Lap(K) = -2 ln p_Lap + d_K ln N; ln p_Lap = Phi(Z_hat) + (nK/2) ln 2pi "
                     "- 1/2 ln|H|; Z_hat = joint mode of Phi from Z0; H = -grad^2 Phi(Z_hat)",
            "d_K": "K d - K(K-1)/2 (loading count; Gaussian-X dispersions, w0, w are "
                   "K-invariant and not added; K-differences equal those of p_K)",
            "N": N_PENALTY, "Z0": Z0_RULE, "grad_tol": GRAD_TOL, "max_iter": MAX_ITER,
            "evaluator": "expfam/src/experimental/laplace_k_criterion.py "
                         "(laplace-k-criterion-v1), unchanged",
            "C_Q": "copied from the recorded runs (main per_fit C_Q; conv checkpoint_scores "
                   "checkpoint 40); never recomputed or replaced",
        },
        "selection_rule": {
            "primary": "per (source, dataset): argmin_K of the unweighted mean over the two "
                       "inits of the criterion; ties within 1e-9 -> smallest K; identical rule "
                       "for C_Q and C_Lap",
            "per_init": "argmin_K within one init; only if that init's 10 evaluations are OK",
        },
        "failure_rule": {
            "statuses": ["OK", "NOT_STATIONARY", "HESSIAN_NOT_PD", "ERROR"],
            "per_evaluation": "recorded once; never retried; no alternative start (Z_mean, "
                              "random, other checkpoint); no ridge/jitter; no tolerance or "
                              "max_iter change; no candidate K dropped",
            "dataset": "if any of the 20 C_Lap evaluations of a (source, dataset) is not OK, "
                       "its C_Lap selection is INCOMPLETE; denominator stays 10; INCOMPLETE is "
                       "reported next to the counts; C_Q selection is reported for every dataset",
            "stop": ["state sha256 mismatch", "regenerated X/Y hash mismatch",
                     "NPZ keys/shape/non-finite/var_z != 1", "C_Lap closure failure",
                     "code sha256 mismatch", "dirty working tree", "input sha256 mismatch",
                     "C_Q selection differs from the recorded pre-specified analyses"],
        },
        "execution": {
            "workers": workers, "blas_threads": 1, "unit_of_work": "one (source, dataset) = 20 "
            "evaluations; rows appended only when the whole job finished",
            "resume_rule": "a (source, dataset) job is done iff its 20 rows are in "
                           "evaluations.csv; done jobs are never re-run; worker count does not "
                           "change any number (each evaluation is deterministic)",
            "memory_rule": f"before dispatching each job, if available memory < {MEM_STOP_MB} "
                           f"MB on {MEM_STOP_CONSECUTIVE} consecutive checks, stop dispatching, "
                           "finish running jobs, record STOPPED_MEMORY (resumable)",
            "technical_check": str(tech_check.relative_to(ROOT)),
        },
        "analysis_plan": {
            "outputs": ["selection_by_dataset.csv", "summary_by_cell.csv",
                        "paired_crosstab.csv", "criterion_curves_median.csv",
                        "term_steps_median.csv", "focus_terms_by_dataset.csv",
                        "lap_status_counts.csv", "lap_diagnostics_by_cell_k.csv",
                        "checks.json", "fig_exact_by_criterion.png",
                        "fig_criterion_curves_focus.png", "fig_term_steps_focus.png"],
            "focus_cells": list(FOCUS_CELLS),
            "primary_question": "A-b-9 (main10 and conv40): on the same theta_hat, in how many "
                                "of the 10 datasets does the C_Lap selection differ from C_Q, "
                                "and how many select K_TRUE (exact count, Clopper-Pearson CI)",
            "control": "O-b-9 (C_Q 10/10): does C_Lap keep 10/10; regime a: C_Lap over-selection",
            "checks": ["C_Q selection == recorded selections (STOP otherwise)",
                       "K_TRUE=3 regime a vs b: C_Lap rows identical (design-identical)"],
            "interpretation_rules": [
                "differences between C_Q and C_Lap are attributed to the criterion only "
                "jointly: evaluation point (E-step MC samples vs joint mode), prior term "
                "(nK(1+ln 2pi) vs ||Z_hat||^2), volume term ln|H| (absent in C_Q) and the "
                "K-invariant penalty offset; no single term is named as THE cause",
                "C_Lap selecting K_TRUE more often does not show that the C_Q prior term is "
                "the only cause of under-selection, nor that C_Lap is correct or consistent",
                "C_Lap and C_Q agreeing on K~5 in A-b-9 is reported as consistent with "
                "limited information under this theta_hat, not as proof",
                "theta_hat is the MCEM estimate (Q-based), not the maximiser of ln p_Lap",
                "INCOMPLETE datasets are never imputed; selection counts keep denominator 10",
                "exploratory: no confirmatory claim, no threshold chosen after the run",
            ],
        },
    }
    write_json(out / "protocol.json", proto)
    print("protocol sha256", sha256_file(out / "protocol.json"))


# --------------------------------------------------------------------------
# execute (frozen; not run during preparation)
# --------------------------------------------------------------------------

def done_jobs(path: Path) -> set:
    if not path.exists():
        return set()
    c = Counter((r["source"], r["dataset_id"]) for r in read_csv(path))
    bad = [k for k, v in c.items() if v != 20]
    need(not bad, f"partial job rows in evaluations.csv: {bad[:3]}")
    return set(c)


def execute(proto_path: Path, out: Path, resume: bool):
    import psutil
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    need(proto["code_sha256"] == code_hashes(), "code sha256 differs from protocol")
    need(not git("status", "--porcelain"), "working tree must be clean")
    need(all(sha256_file(p) == proto["input_sha256"][k] for k, p in INPUTS.items()),
         "input sha256")
    rows = read_csv(proto_path.parent / "manifest.csv")
    for r in rows:
        for k in ("k_true", "rep", "k", "t", "data_seed", "num_params"):
            r[k] = int(r[k])
    need(manifest_hash(rows) == proto["manifest_sha256"], "manifest sha256")
    if out.exists():
        need(resume, f"output exists (use --resume): {out}")
    else:
        out.mkdir(parents=True)
    ev_path = out / "evaluations.csv"
    done = done_jobs(ev_path)
    jobs = [j for j in manifest_jobs(rows) if (j["source"], j["dataset_id"]) not in done]
    event = {"event": "start", "time": now(), "git_head": git("rev-parse", "HEAD"),
             "protocol_sha256": sha256_file(proto_path), "jobs_done": len(done),
             "jobs_todo": len(jobs), "workers": proto["execution"]["workers"]}
    append_jsonl(out / "run_events.jsonl", event)
    write_json(out / "runinfo.json", {"protocol": str(proto_path.relative_to(ROOT)),
                                      "environment": environment(), **event})
    workers = int(proto["execution"]["workers"])
    low, stopped, pending, it = 0, None, set(), iter(jobs)
    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        while True:
            while stopped is None and len(pending) < workers:
                avail = psutil.virtual_memory().available / 2 ** 20
                low = low + 1 if avail < MEM_STOP_MB else 0
                if low >= MEM_STOP_CONSECUTIVE:
                    stopped = f"STOPPED_MEMORY (available {avail:.0f} MB)"
                    break
                j = next(it, None)
                if j is None:
                    break
                pending.add(ex.submit(run_dataset_job, j))
            if not pending:
                break
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in fin:
                append_csv(ev_path, f.result(), EVAL_FIELDS)    # Stop propagates
                n_done += 1
            if n_done % 10 == 0:
                print(f"{n_done}/{len(jobs)} jobs ({time.time() - t0:.0f}s)", flush=True)
    append_jsonl(out / "run_events.jsonl",
                 {"event": stopped or "finished", "time": now(), "jobs_run": n_done,
                  "wall_s": round(time.time() - t0, 1)})


def append_jsonl(p: Path, obj):
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


# --------------------------------------------------------------------------
# pre-specified analysis (pure functions; tested on fake data)
# --------------------------------------------------------------------------

def pick(bar: dict, tol=TIE_TOL):
    m = min(bar.values())
    return min(k for k, v in bar.items() if v - m <= tol)


def outcome(k, kt):
    return "exact" if k == kt else ("under" if k < kt else "over")


def clopper_pearson(x, n, a=0.05):
    from scipy.stats import beta
    lo = 0.0 if x == 0 else float(beta.ppf(a / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - a / 2, x + 1, n - x))
    return lo, hi


def cell_of(r):
    return f'{r["series"]}-{r["regime"]}-{r["k_true"]}'


def group_datasets(ev_rows):
    g = defaultdict(dict)
    for r in ev_rows:
        g[(r["source"], r["dataset_id"])][(int(r["k"]), int(r["t"]))] = r
    return g


def q_terms(r):
    cq, p, k = float(r["C_Q"]), int(r["num_params"]), int(r["k"])
    pz, pen = PZ_PER_DIM * k, p * LN_N
    return {"C_Q": cq, "Q_P_Z": pz, "Q_PEN": pen, "Q_D": cq - pz - pen}


def lap_terms(r):
    return {"C_Lap": float(r["C_Lap"]), "L_fit": float(r["lap_fit_mode"]),
            "L_zsq": float(r["lap_zhat_sq"]), "L_logdet": float(r["lap_logdet_H"]),
            "L_pen": float(r["lap_penalty"])}


def dataset_selection(src, ds, f):
    any_r = next(iter(f.values()))
    kt = int(any_r["k_true"])
    ks = sorted({k for k, _ in f})
    need(ks == list(range(1, 11)) and len(f) == 20, f"incomplete rows {src} {ds}")
    qbar = {k: (float(f[(k, 0)]["C_Q"]) + float(f[(k, 1)]["C_Q"])) / 2.0 for k in ks}
    kq = pick(qbar)
    bad = sorted(f"K{k}t{t}:{r['lap_status']}" for (k, t), r in f.items()
                 if r["lap_status"] != STATUS_OK)
    row = {"source": src, "checkpoint": any_r["checkpoint"], "dataset_id": ds,
           "cell": cell_of(any_r), "series": any_r["series"], "regime": any_r["regime"],
           "k_true": kt, "rep": any_r["rep"], "k_Q": kq, "outcome_Q": outcome(kq, kt),
           "k_Q_init0": pick({k: float(f[(k, 0)]["C_Q"]) for k in ks}),
           "k_Q_init1": pick({k: float(f[(k, 1)]["C_Q"]) for k in ks}),
           "lap_status": "INCOMPLETE" if bad else "COMPLETE", "lap_failed_fits": ";".join(bad)}
    for t in (0, 1):
        ok_t = all(f[(k, t)]["lap_status"] == STATUS_OK for k in ks)
        row[f"k_Lap_init{t}"] = pick({k: float(f[(k, t)]["C_Lap"]) for k in ks}) if ok_t else ""
    if bad:
        row.update(k_Lap="", outcome_Lap="INCOMPLETE", changed="", direction="",
                   margin_Lap="", gap_Lap_ktrue="")
    else:
        lbar = {k: (float(f[(k, 0)]["C_Lap"]) + float(f[(k, 1)]["C_Lap"])) / 2.0 for k in ks}
        kl = pick(lbar)
        srt = sorted(lbar.values())
        row.update(k_Lap=kl, outcome_Lap=outcome(kl, kt), changed=kl != kq,
                   direction="same" if kl == kq else ("Lap>Q" if kl > kq else "Lap<Q"),
                   margin_Lap=srt[1] - srt[0], gap_Lap_ktrue=lbar[kt] - lbar[kl])
    srt = sorted(qbar.values())
    row.update(margin_Q=srt[1] - srt[0], gap_Q_ktrue=qbar[kt] - qbar[kq])
    return row


def mean_terms(f, k, fn):
    a, b = fn(f[(k, 0)]), fn(f[(k, 1)])
    return {key: (a[key] + b[key]) / 2.0 for key in a}


def analyze_rows(ev_rows, ref_q=None):
    """Returns dict of tables. ref_q: {(source, dataset_id): selected K} or None."""
    g = group_datasets(ev_rows)
    sel = [dataset_selection(src, ds, f) for (src, ds), f in sorted(g.items())]
    checks = {}
    if ref_q is not None:
        diff = [(s["source"], s["dataset_id"]) for s in sel
                if ref_q.get((s["source"], s["dataset_id"])) != s["k_Q"]]
        need(not diff, f"C_Q selection differs from recorded analyses: {diff[:5]}")
        checks["C_Q_selection_equals_recorded"] = True
    # summary per (source, cell)
    by_cell = defaultdict(list)
    for s in sel:
        by_cell[(s["source"], s["cell"])].append(s)
    summary = []
    for (src, cell), v in sorted(by_cell.items()):
        n = len(v)
        cq = Counter(s["outcome_Q"] for s in v)
        cl = Counter(s["outcome_Lap"] for s in v)
        comp = [s for s in v if s["lap_status"] == "COMPLETE"]
        lo_q, hi_q = clopper_pearson(cq["exact"], n)
        lo_l, hi_l = clopper_pearson(cl["exact"], n)
        summary.append({
            "source": src, "cell": cell, "series": v[0]["series"], "regime": v[0]["regime"],
            "k_true": v[0]["k_true"], "n_datasets": n,
            "Q_exact": cq["exact"], "Q_under": cq["under"], "Q_over": cq["over"],
            "Q_exact_ci_lo": lo_q, "Q_exact_ci_hi": hi_q,
            "Lap_exact": cl["exact"], "Lap_under": cl["under"], "Lap_over": cl["over"],
            "Lap_incomplete": cl["INCOMPLETE"], "Lap_exact_ci_lo": lo_l, "Lap_exact_ci_hi": hi_l,
            "n_complete": len(comp), "n_changed": sum(s["changed"] is True for s in comp),
            "n_Lap_gt_Q": sum(s["direction"] == "Lap>Q" for s in comp),
            "n_Lap_lt_Q": sum(s["direction"] == "Lap<Q" for s in comp),
            "Lap_inits_disagree": sum(1 for s in comp
                                      if s["k_Lap_init0"] != s["k_Lap_init1"]),
            "Q_selected_dist": json.dumps(dict(sorted(Counter(s["k_Q"] for s in v).items()))),
            "Lap_selected_dist": json.dumps(dict(sorted(Counter(s["k_Lap"] for s in comp
                                                                ).items()))),
        })
    crosstab = Counter((s["source"], s["cell"], s["k_Q"], s["k_Lap"] if s["k_Lap"] != ""
                        else "INCOMPLETE") for s in sel)
    cross = [{"source": a, "cell": b, "k_Q": c, "k_Lap": d, "count": n}
             for (a, b, c, d), n in sorted(crosstab.items(), key=lambda x: tuple(map(str, x[0])))]
    # curves, steps, focus
    curves = defaultdict(lambda: defaultdict(list))
    steps = defaultdict(lambda: defaultdict(list))
    focus = []
    for s in sel:
        f = g[(s["source"], s["dataset_id"])]
        kt = s["k_true"]
        qt = {k: mean_terms(f, k, q_terms) for k in range(1, 11)}
        comp = s["lap_status"] == "COMPLETE"
        lt = {k: mean_terms(f, k, lap_terms) for k in range(1, 11)} if comp else None
        for k in range(1, 11):
            curves[(s["source"], s["cell"], k)]["Q"].append(qt[k]["C_Q"] - qt[kt]["C_Q"])
            if comp:
                curves[(s["source"], s["cell"], k)]["Lap"].append(lt[k]["C_Lap"] - lt[kt]["C_Lap"])
            if k < 10:
                key = (s["source"], s["cell"], k)
                for name in ("C_Q", "Q_D", "Q_P_Z", "Q_PEN"):
                    steps[key][name].append(qt[k + 1][name] - qt[k][name])
                if comp:
                    for name in ("C_Lap", "L_fit", "L_zsq", "L_logdet", "L_pen"):
                        steps[key][name].append(lt[k + 1][name] - lt[k][name])
        if s["cell"] in FOCUS_CELLS:
            kq = s["k_Q"]
            row = {"source": s["source"], "dataset_id": s["dataset_id"], "cell": s["cell"],
                   "k_true": kt, "k_Q": kq, "k_Lap": s["k_Lap"],
                   "lap_status": s["lap_status"]}
            for name in ("C_Q", "Q_D", "Q_P_Z", "Q_PEN"):
                row[f"d_{name}_ktrue_minus_kQ"] = qt[kt][name] - qt[kq][name]
            for name in ("C_Lap", "L_fit", "L_zsq", "L_logdet", "L_pen"):
                row[f"d_{name}_ktrue_minus_kQ"] = (lt[kt][name] - lt[kq][name]) if comp else ""
            focus.append(row)
    med = lambda v: stats.median(v) if v else ""          # noqa: E731
    curve_rows = [{"source": a, "cell": b, "k": k, "n_Q": len(d["Q"]), "n_Lap": len(d["Lap"]),
                   "median_CQ_minus_CQ_ktrue": med(d["Q"]),
                   "median_CLap_minus_CLap_ktrue": med(d["Lap"])}
                  for (a, b, k), d in sorted(curves.items())]
    step_rows = []
    for (a, b, k), d in sorted(steps.items()):
        row = {"source": a, "cell": b, "k_from": k, "k_to": k + 1,
               "n_Lap": len(d.get("C_Lap", []))}
        for name in ("C_Q", "Q_D", "Q_P_Z", "Q_PEN", "C_Lap", "L_fit", "L_zsq", "L_logdet",
                     "L_pen"):
            row[f"median_d_{name}"] = med(d.get(name, []))
        step_rows.append(row)
    # status counts and diagnostics
    st_count = Counter((r["source"], r["series"], r["regime"], int(r["k"]), r["lap_status"])
                       for r in ev_rows)
    status_rows = [{"source": a, "series": b, "regime": c, "k": k, "lap_status": s, "count": n}
                   for (a, b, c, k, s), n in sorted(st_count.items())]
    diag = defaultdict(list)
    for r in ev_rows:
        diag[(r["source"], cell_of(r), int(r["k"]))].append(r)
    diag_rows = []
    for (a, b, k), v in sorted(diag.items()):
        ok = [r for r in v if r["lap_status"] == STATUS_OK]
        fl = lambda c: [float(r[c]) for r in ok]          # noqa: E731
        diag_rows.append({
            "source": a, "cell": b, "k": k, "n": len(v), "n_ok": len(ok),
            "min_of_min_eig_H": min(fl("lap_min_eig_H")) if ok else "",
            "max_grad_inf": max(fl("lap_grad_inf")) if ok else "",
            "median_iterations": med([int(r["lap_iterations"]) for r in ok]),
            "median_z_shift_rms": med(fl("z_shift_rms")) if ok else "",
            "median_seconds": med([float(r["lap_seconds"]) for r in v]),
            "max_closure_absdiff": max(fl("lap_closure_absdiff")) if ok else ""})
    # K_TRUE = 3: regime a and b are design-identical -> identical C_Lap rows
    k3 = defaultdict(dict)
    for r in ev_rows:
        if int(r["k_true"]) == 3:
            k3[(r["source"], r["series"], r["rep"], r["k"], r["t"])][r["regime"]] = \
                (r["lap_status"], r["C_Lap"])
    pairs = [v for v in k3.values() if len(v) == 2]
    checks["K_TRUE3_a_b_identical_C_Lap"] = (all(v["a"] == v["b"] for v in pairs)
                                             if pairs else "n/a")
    checks["n_evaluations"] = len(ev_rows)
    checks["lap_status_total"] = dict(Counter(r["lap_status"] for r in ev_rows))
    return {"selection": sel, "summary": summary, "crosstab": cross, "curves": curve_rows,
            "steps": step_rows, "focus": focus, "status": status_rows, "diag": diag_rows,
            "checks": checks}


def recorded_q_selection() -> dict:
    out = {}
    for r in read_csv(INPUTS["main_selection_per_dataset"]):
        if r["series"] in SERIES_IN:
            out[("main10", r["dataset_id"])] = int(r["primary_k"])
    for r in read_csv(INPUTS["conv_selected_k_by_dataset"]):
        if r["checkpoint"] == "40":
            out[("conv40", r["dataset_id"])] = int(r["selected_k"])
    return out


CRIT_COLORS = {"C_Q": "#2a78d6", "C_Lap": "#eb6834"}   # fixed per criterion


def figures(res, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summ = [s for s in res["summary"] if s["regime"] == "b" and s["k_true"] >= 5]
    if summ:
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = [f'{s["cell"]}\n{s["source"]}' for s in summ]
        x = np.arange(len(summ))
        ax.bar(x - 0.2, [s["Q_exact"] for s in summ], 0.38, color=CRIT_COLORS["C_Q"],
               label="C_Q")
        ax.bar(x + 0.2, [s["Lap_exact"] for s in summ], 0.38, color=CRIT_COLORS["C_Lap"],
               label="C_Lap")
        for i, s in enumerate(summ):
            if s["Lap_incomplete"]:
                ax.text(i + 0.2, s["Lap_exact"] + 0.2, f'inc {s["Lap_incomplete"]}',
                        ha="center", fontsize=7)
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylim(0, 10.8)
        ax.set_ylabel("datasets selecting K_TRUE (of 10)")
        ax.set_title("Exact selections, regime b, K_TRUE >= 5 (same saved fits)")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "fig_exact_by_criterion.png", dpi=150)
        plt.close(fig)
    curves = [c for c in res["curves"] if c["cell"] in FOCUS_CELLS]
    keys = sorted({(c["source"], c["cell"]) for c in curves})
    if keys:
        fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.2), squeeze=False)
        for ax, (src, cell) in zip(axes[0], keys):
            v = sorted((c for c in curves if (c["source"], c["cell"]) == (src, cell)),
                       key=lambda c: c["k"])
            ks = [c["k"] for c in v]
            ax.plot(ks, [c["median_CQ_minus_CQ_ktrue"] for c in v], "-o", ms=4, lw=2,
                    color=CRIT_COLORS["C_Q"], label="C_Q")
            lv = [(c["k"], c["median_CLap_minus_CLap_ktrue"]) for c in v
                  if c["median_CLap_minus_CLap_ktrue"] != ""]
            if lv:
                ax.plot([a for a, _ in lv], [b for _, b in lv], "-s", ms=4, lw=2,
                        color=CRIT_COLORS["C_Lap"], label="C_Lap")
            ax.axhline(0, color="#888", lw=0.8)
            ax.set_title(f"{cell} ({src})", fontsize=9)
            ax.set_xlabel("K")
        axes[0][0].set_ylabel("median of crit(K) - crit(K_TRUE)")
        axes[0][0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_criterion_curves_focus.png", dpi=150)
        plt.close(fig)
    steps = [s for s in res["steps"] if s["cell"] in ("A-b-9", "O-b-9", "B-b-9")]
    keys = sorted({(s["source"], s["cell"]) for s in steps})
    if keys:
        fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.2), squeeze=False)
        for ax, key in zip(axes[0], keys):
            v = sorted((s for s in steps if (s["source"], s["cell"]) == key),
                       key=lambda s: s["k_from"])
            ks = [s["k_from"] + 0.5 for s in v]
            ax.plot(ks, [s["median_d_C_Q"] for s in v], "-o", ms=4, lw=2,
                    color=CRIT_COLORS["C_Q"], label="dC_Q")
            lv = [(a, s["median_d_C_Lap"]) for a, s in zip(ks, v) if s["median_d_C_Lap"] != ""]
            if lv:
                ax.plot([a for a, _ in lv], [b for _, b in lv], "-s", ms=4, lw=2,
                        color=CRIT_COLORS["C_Lap"], label="dC_Lap")
            ax.axhline(0, color="#888", lw=0.8)
            ax.set_title(f"{key[1]} ({key[0]})", fontsize=9)
            ax.set_xlabel("K -> K+1")
        axes[0][0].set_ylabel("median step of criterion (<0: K+1 better)")
        axes[0][0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_term_steps_focus.png", dpi=150)
        plt.close(fig)


def analyze(run: Path, out: Path, ref=True):
    need(not out.exists(), f"output exists: {out}")
    ev = read_csv(run / "evaluations.csv")
    res = analyze_rows(ev, recorded_q_selection() if ref else None)
    out.mkdir(parents=True)
    for name, key in (("selection_by_dataset.csv", "selection"),
                      ("summary_by_cell.csv", "summary"), ("paired_crosstab.csv", "crosstab"),
                      ("criterion_curves_median.csv", "curves"),
                      ("term_steps_median.csv", "steps"),
                      ("focus_terms_by_dataset.csv", "focus"),
                      ("lap_status_counts.csv", "status"),
                      ("lap_diagnostics_by_cell_k.csv", "diag")):
        if res[key]:
            write_csv(out / name, res[key])
    write_json(out / "checks.json", res["checks"])
    figures(res, out)
    print(json.dumps(res["checks"], indent=2))


# --------------------------------------------------------------------------
# technical check (SMOKE states only; different data seeds; no main-data score)
# --------------------------------------------------------------------------

def _independent_loglik(series, X, Y, Z, st):
    """ln p(X|Z) + ln p(Y|Z) written here from the model definition (upper pairs once)."""
    from scipy.special import gammaln
    fx, fy = kd.SERIES[series]
    eta_x = Z @ st["F"].T
    if fx == "gaussian":
        var = np.diag(st["sigma"])
        lx = float(np.sum(-0.5 * np.log(2 * np.pi * var) - (X - eta_x) ** 2 / (2 * var)))
    elif fx == "poisson":
        lx = float(np.sum(X * eta_x - np.exp(eta_x) - gammaln(X + 1)))
    else:
        lx = float(np.sum(X * eta_x - np.logaddexp(0, eta_x)))
    iu = np.triu_indices(Z.shape[0], 1)
    eta_y = (float(st["w0"]) + float(st["w"]) * (Z @ Z.T))[iu]
    y = Y[iu]
    if fy == "bernoulli":
        ly = float(np.sum(y * eta_y - np.logaddexp(0, eta_y)))
    else:
        ly = float(np.sum(y * eta_y - np.exp(eta_y) - gammaln(y + 1)))
    return lx + ly


def _fd_checks(model, X, Y, Z, rng, n_dir=3, h=1e-5):
    import laplace_k_criterion as lk
    g = lk.gradient(model, X, Y, Z)
    H = lk.negative_hessian(model, X, Y, Z)
    worst_g = worst_h = 0.0
    for _ in range(n_dir):
        v = rng.standard_normal(Z.shape)
        v /= np.linalg.norm(v)
        fp = lk.complete_log_density(model, X, Y, Z + h * v)
        fm = lk.complete_log_density(model, X, Y, Z - h * v)
        dd = (fp - fm) / (2 * h)
        worst_g = max(worst_g, abs(dd - float(np.sum(g * v))) / max(1.0, abs(dd)))
        gp = lk.gradient(model, X, Y, Z + h * v)
        gm = lk.gradient(model, X, Y, Z - h * v)
        hv_fd = -(gp - gm).ravel() / (2 * h)
        hv = H @ v.ravel()
        worst_h = max(worst_h, float(np.max(np.abs(hv_fd - hv)) / max(1.0, np.max(np.abs(hv)))))
    return worst_g, worst_h


def _peak_worker(args):
    """Evaluate one state in a fresh process; return (row-subset, peak working set MB)."""
    import psutil
    series, regime, k_true, rep, seed, path, k, trunc = args
    ds = regenerate(series, regime, k_true, rep, seed, bern_table())
    st = load_state(Path(path), k)
    if trunc:
        st = {**st, "F": st["F"][:, :trunc], "Z_last": st["Z_last"][:, :trunc]}
        k = trunc
    model = restore_model(series, k, st, st["Z_last"])
    r = evaluate_state(model, ds.X, ds.Y, st["Z_last"], k)
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None) or getattr(mi, "rss", 0)
    return {"k": k, "status": r["lap_status"], "C_Lap": r["C_Lap"],
            "seconds": r["lap_seconds"], "peak_MB": peak / 2 ** 20}


def tech_check(out: Path):
    import psutil
    from concurrent.futures import ProcessPoolExecutor
    need(not out.exists(), f"output exists: {out}")
    out.mkdir(parents=True)
    checks, notes, rows = {}, {}, []
    t_start = time.time()
    # (1) NPZ integrity of all 6,600 planned inputs (hash/keys/shape/finite only; no score)
    man = build_manifest()
    bad = []
    for r in man:
        p = STATE_DIRS[r["source"]] / r["state_file"]
        if not p.exists() or sha256_file(p) != r["state_sha256"]:
            bad.append(r["fit_id"] + "@" + r["source"])
            continue
        try:
            load_state(p, r["k"])
        except Stop as exc:
            bad.append(f"{r['fit_id']}@{r['source']}: {exc}")
    checks["npz_6600_sha256_keys_shapes_finite_var_z1"] = not bad
    notes["npz_bad"] = bad[:20]
    notes["npz_checked"] = len(man)
    # (2) regenerated X/Y equal recorded hashes for all 300 planned datasets (no score)
    table = bern_table()
    ds_jobs = {(r["dataset_id"]): r for r in man}
    mism = []
    sym_ok = True
    for dsid, r in sorted(ds_jobs.items()):
        d = regenerate(r["series"], r["regime"], r["k_true"], r["rep"], r["data_seed"], table)
        if kd.array_hash(d.X) != r["x_hash"] or kd.array_hash(d.Y) != r["y_hash"]:
            mism.append(dsid)
        sym_ok &= bool(np.array_equal(d.Y, d.Y.T) and np.all(np.diag(d.Y) == 0))
    checks["regenerated_xy_equal_recorded_hash_all_datasets"] = not mism
    checks["Y_symmetric_zero_diagonal"] = sym_ok
    notes["datasets_regenerated"] = len(ds_jobs)
    notes["dataset_mismatch"] = mism[:20]
    # SMOKE states (different data seeds) for everything that computes C_Lap
    smoke = [r for r in read_csv(SMOKE_DIR / "per_fit.csv")
             if r["series"] in SERIES_IN and r["status"] == "OK"]
    rng = np.random.default_rng(20261126)
    restore_ok = fd_ok = smoke_hash_ok = True
    worst = {"grad": 0.0, "hess": 0.0}
    det_ok = True
    start_dep = []
    for r in smoke:
        k = int(r["k"])
        path = SMOKE_DIR / "smoke_states" / r["state_file"]
        st = load_state(path, k)
        d = regenerate(r["series"], r["regime"], int(r["k_true"]), int(r["rep"]),
                       int(r["data_seed"]), table)
        if kd.array_hash(d.X) != r["x_hash"] or kd.array_hash(d.Y) != r["y_hash"]:
            smoke_hash_ok = False
        model = restore_model(r["series"], k, st, st["Z_last"])
        # (3) model restoration: model log-likelihood == independent formula at Z_last
        from eval_utils import calc_Q_dual_strict_exp
        phi = calc_Q_dual_strict_exp(d.X, d.Y, st["Z_last"][:, :, None], st["F"], st["sigma"],
                                     1.0, float(st["w0"]), float(st["w"]), model)
        lnp = -(kd.N * k / 2) * math.log(2 * math.pi) - 0.5 * float(np.sum(st["Z_last"] ** 2))
        indep = _independent_loglik(r["series"], d.X, d.Y, st["Z_last"], st)
        rel = abs((phi - lnp) - indep) / max(1.0, abs(indep))
        restore_ok &= rel < 1e-9
        # (4) applicability: evaluator runs; FD checks of gradient / Hessian
        wg, wh = _fd_checks(model, d.X, d.Y, st["Z_last"], rng)
        worst["grad"], worst["hess"] = max(worst["grad"], wg), max(worst["hess"], wh)
        fd_ok &= (wg < 1e-5 and wh < 1e-4)
        e1 = evaluate_state(model, d.X, d.Y, st["Z_last"], k)
        # (8) determinism: second evaluation in the same process
        model2 = restore_model(r["series"], k, st, st["Z_last"])
        e2 = evaluate_state(model2, d.X, d.Y, st["Z_last"], k)
        same = all(e1.get(c) == e2.get(c) for c in ("lap_status", "C_Lap", "lap_logdet_H",
                                                    "lap_zhat_sq", "lap_iterations"))
        det_ok &= same
        # start dependence diagnostic (Z_mean as start) -- tech only, not in the protocol
        model3 = restore_model(r["series"], k, st, st["Z_mean"])
        e3 = evaluate_state(model3, d.X, d.Y, st["Z_mean"], k)
        if e1["lap_status"] == STATUS_OK and e3["lap_status"] == STATUS_OK:
            start_dep.append(abs(float(e1["C_Lap"]) - float(e3["C_Lap"])))
        rows.append({"fit_id": r["fit_id"], "k": k, "restore_rel_err": rel, "fd_grad": wg,
                     "fd_hess": wh, **{f"z_last_{c}": e1.get(c) for c in (
                         "lap_status", "C_Lap", "lap_seconds", "lap_iterations", "lap_min_eig_H",
                         "lap_grad_inf", "lap_logdet_H", "lap_zhat_sq", "z_shift_rms")},
                     "z_mean_status": e3["lap_status"], "z_mean_C_Lap": e3["C_Lap"],
                     "deterministic_repeat": same})
    write_csv(out / "smoke_evaluations.csv", rows)
    checks["smoke_data_regenerated_equal_hash"] = smoke_hash_ok
    checks["model_restore_loglik_equals_independent_formula"] = restore_ok
    checks["fd_gradient_hessian_agree"] = fd_ok
    checks["evaluator_status_ok_on_all_smoke_states"] = all(
        r["z_last_lap_status"] == STATUS_OK for r in rows)
    checks["deterministic_repeat_bitwise"] = det_ok
    notes["fd_worst_rel"] = worst
    notes["start_dependence_absdiff_C_Lap_Zlast_vs_Zmean"] = {
        "n": len(start_dep), "max": max(start_dep) if start_dep else None,
        "median": stats.median(start_dep) if start_dep else None}
    # (9) failure handling: forced NOT_STATIONARY (max_iter=0) and forced ERROR (NaN theta)
    r0 = next(r for r in smoke if r["k"] == "10")
    st = load_state(SMOKE_DIR / "smoke_states" / r0["state_file"], 10)
    d = regenerate(r0["series"], r0["regime"], int(r0["k_true"]), int(r0["rep"]),
                   int(r0["data_seed"]), table)
    m = restore_model(r0["series"], 10, st, st["Z_last"])
    ns = evaluate_state(m, d.X, d.Y, st["Z_last"], 10, max_iter=0)
    bad_st = {**st, "F": st["F"].copy()}
    bad_st["F"][0, 0] = np.nan
    m_bad = restore_model(r0["series"], 10, bad_st, st["Z_last"])
    er = evaluate_state(m_bad, d.X, d.Y, st["Z_last"], 10)
    checks["failure_NOT_STATIONARY_recorded_no_value"] = (ns["lap_status"] == "NOT_STATIONARY"
                                                         and ns["C_Lap"] == "")
    checks["failure_exception_recorded_as_status_no_value"] = (
        er["lap_status"] in (STATUS_ERROR, "NOT_STATIONARY", "HESSIAN_NOT_PD")
        and er["C_Lap"] == "")
    notes["forced_failure_statuses"] = {"max_iter0": ns["lap_status"],
                                        "nan_theta": [er["lap_status"], er["lap_error"]]}
    # (5)(6) timing and peak memory, one state per fresh process; K = 1, 9 (proxy), 10
    timing_args = []
    for r in smoke:
        if r["k"] == "10":
            base = (r["series"], r["regime"], int(r["k_true"]), int(r["rep"]),
                    int(r["data_seed"]), str(SMOKE_DIR / "smoke_states" / r["state_file"]), 10)
            timing_args += [base + (None,), base + (9,)]
        elif r["k"] == "1":
            timing_args.append((r["series"], r["regime"], int(r["k_true"]), int(r["rep"]),
                                int(r["data_seed"]),
                                str(SMOKE_DIR / "smoke_states" / r["state_file"]), 1, None))
    timing = []
    for a in timing_args:                                  # serial, one process each
        with ProcessPoolExecutor(max_workers=1) as ex:
            timing.append({"series": a[0], "regime": a[1], "k_true": a[2],
                           "truncated_proxy": a[7] is not None,
                           **ex.submit(_peak_worker, a).result()})
    write_csv(out / "timing_memory.csv", timing)
    # (7) parallel safety: same K=10 states with 4 workers == serial values
    par_args = [a for a in timing_args if a[6] == 10 and a[7] is None]
    with ProcessPoolExecutor(max_workers=4) as ex:
        par = list(ex.map(_peak_worker, par_args))
    ser = [t for t in timing if t["k"] == 10 and not t["truncated_proxy"]]
    checks["parallel_4_workers_equal_serial_bitwise"] = (
        [p["C_Lap"] for p in par] == [s["C_Lap"] for s in ser])
    by_k = defaultdict(list)
    for t in timing:
        by_k[t["k"]].append(t)
    notes["timing_by_k"] = {k: {"n": len(v), "median_s": stats.median(x["seconds"] for x in v),
                                "max_s": max(x["seconds"] for x in v),
                                "max_peak_MB": max(x["peak_MB"] for x in v)}
                            for k, v in sorted(by_k.items())}
    vm = psutil.virtual_memory()
    notes["machine"] = {"total_MB": vm.total // 2 ** 20, "available_MB_now": vm.available // 2 ** 20,
                        "physical_cores": psutil.cpu_count(False),
                        "logical_cores": psutil.cpu_count()}
    notes["wall_s"] = round(time.time() - t_start, 1)
    res = {"created": now(), "git_head": git("rev-parse", "HEAD"), "environment": environment(),
           "checks": checks, "all_ok": all(v is True for v in checks.values()), "notes": notes,
           "scope": "NPZ/data integrity on the 6,600 planned inputs (no score computed); every "
                    "C_Lap value here is on SMOKE states (data seeds offset by 5,000,000), "
                    "never on main data"}
    write_json(out / "tech_check.json", res)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res["all_ok"] else 1


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tech-check", metavar="OUT")
    g.add_argument("--freeze", metavar="OUT")
    g.add_argument("--execute", metavar="OUT")
    g.add_argument("--analyze", metavar="RUN")
    ap.add_argument("--protocol")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--tech-check-dir")
    a = ap.parse_args()
    if a.tech_check:
        return tech_check(ROOT / a.tech_check)
    if a.freeze:
        return freeze(ROOT / a.freeze, a.workers, ROOT / a.tech_check_dir) or 0
    if a.execute:
        return execute(ROOT / a.protocol, ROOT / a.execute, a.resume) or 0
    return analyze(ROOT / a.analyze, ROOT / a.out) or 0


if __name__ == "__main__":
    raise SystemExit(main())
