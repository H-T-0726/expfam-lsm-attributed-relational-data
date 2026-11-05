"""R1 / R2: robustness of the d-sweep findings (lineage E, pre-registered).

R1  estimation-condition sensitivity at n=75, d in {15,30,60}:
    new fits with L=10, max_iter=16, evaluated at checkpoints 8 and 16
    (336 fits).  Baselines: L=5 / 8 it (d-sweep, all d) and L=5 / 16 it
    (EM-convergence checkpoint 16, d=15 only, auxiliary, read from the
    checkpoint re-evaluation CSV).
R2  node-count sensitivity at d=30, n in {50,100,150} (L=5, 8 it): new
    fits on the A1 datasets of each n extended with the d-sweep block rule.
    d=15 at the same n = the EM-convergence checkpoint-8 states re-evaluated
    (no MCEM).  n=75 is read from the d-sweep.  n=150 is staged: it runs only
    if the resource rule N150_RULE (fixed here, independent of any result)
    holds.

Everything that decides the experiment is in PREREG and is frozen by commit
before any production fit.  Fits use em_runner_instrumented (blob 5b3f7baa,
the code path of every reused state).

Failure rules (pre-registered): a NaN restart request (RetryDetected) is
recorded as FAILED_NAN for that fit -- no retry, no reseed -- and makes the
dataset INCOMPLETE for both criteria at that condition; the run continues.
Any other exception stops the run (fail-closed, partial rows written).
Laplace not OK -> C_Lap INCOMPLETE.  Nothing is imputed.

Stages:
  --r1-gates   data gates n=75, init-identity across L, replication of one
               d-sweep d=30 fit, one L=10/16-it pilot fit (not counted)
  --r1         336 production fits
  --r2-gates   data gates n=50,100,150, d=15 re-evaluation gate, replication
               of one n=50 d=15 EMC fit, one n=150 d=30 pilot fit (not counted)
  --r2 N       production fits for one n (50, 100 or 150)
  --r2-decide-n150   apply N150_RULE and record the decision
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
_EXPERIMENTAL = ROOT / "expfam" / "src" / "experimental"
for _p in (str(_EXPERIMENTAL), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_clap_clean_nsweep as a1                          # noqa: E402
import run_clap_checkpoint_reeval as cpr                    # noqa: E402
import run_dsweep_k_selection as ds2                        # noqa: E402
import k_criterion_panel as panel                           # noqa: E402
import laplace_k_criterion as lk                            # noqa: E402
import em_runner                                            # noqa: E402
from em_runner_instrumented import RetryDetected, run_em_instrumented  # noqa: E402
from data_generator_canonical import (build_full_rank_loadings,  # noqa: E402
                                      canonical_poisson_rate)

hist = a1.hist
need = ds2.need
write_rows = ds2.write_rows

EXPERIMENT_ID = "k_selection_robustness_20261105"
OUT_ROOT = ROOT / "expfam" / "results" / "k_selection" / EXPERIMENT_ID
STATE_ROOT = Path(r"D:\tento\kennkyu_artifacts") / EXPERIMENT_ID
DSWEEP_FULL = ROOT / "expfam" / "results" / "k_selection" / ds2.EXPERIMENT_ID / "full"
EXP1_FULL = ROOT / "expfam" / "results" / "k_selection" / cpr.EXPERIMENT_ID / "full"

K_TRUE = 5
D0 = hist.D_FEATURES
BLOCKS = ds2.BLOCKS
BLOCK_SEED_TAG = ds2.BLOCK_SEED_TAG
REPLICATES = ds2.REPLICATES
C2 = ds2.C2
QLAP_SEED_BASE = cpr.QLAP_SEED_BASE

R1 = {"n": 75, "d": (15, 30, 60), "L": 10, "max_iter": 16, "checkpoints": (8, 16)}
R2 = {"n": (50, 100, 150), "d": 30, "L": hist.L_SAMPLES, "max_iter": 8, "checkpoints": (8,)}
FIT_CAP = 700
N150_RULE = {"requires": ["R2 n=50 and n=100 production runs COMPLETED",
                          "projected n=150 wall time from the n=150 pilot fit <= 3 h",
                          "free space on D: >= 10 GB", "free RAM >= 2 GB"],
             "projection": "112 x (pilot EM seconds + pilot evaluation seconds) x 1.25"}

PREREG = {
    "experiment_id": EXPERIMENT_ID,
    "frozen_before": "any production fit (committed before --r1 / --r2)",
    "common": {"generator": "canonical-clean + d-sweep block rule", "family_x": "poisson",
               "family_y": "bernoulli", "K_GEN": K_TRUE, "candidate_K": list(hist.CANDIDATE_K),
               "replicates": list(REPLICATES), "starts": list(hist.STARTS),
               "numerics_mode": hist.NUMERICS_MODE, "model_seed": "A1 formula (paired across conditions)",
               "runner": "em_runner_instrumented (blob 5b3f7baa)", "selection": "mean over starts, argmin",
               "target": "K* = 5 (minimal representation; not K0 uniqueness); under and over are errors"},
    "R1": {**{k: list(v) if isinstance(v, tuple) else v for k, v in R1.items()},
           "comparisons": {"L_effect": "(L5, it8) vs (L10, it8)",
                           "iteration_effect": "(L10, it8) vs (L10, it16)",
                           "auxiliary": "(L5, it8) vs (L5, it16) at d=15 only (EMC checkpoint 16)"},
           "comparability": ("same data, mask, model seed and initialisation (gated); the MC "
                             "stream differs with L, so trajectories are NOT assumed identical"),
           "hypotheses": {
               "R1-H1": {"statement": "under (L10,it8) and (L10,it16) the median C_Q slack_45 is larger at d=60 than at d=15",
                         "refute": "median slack_45 at d=60 <= at d=15 at either checkpoint"},
               "R1-H2": {"statement": "under (L10,it8) and (L10,it16) the median 2dH_Lap(4->5) is smaller at d=60 than at d=15 and positive at d=60",
                         "refute": "median at d=60 >= at d=15, or median at d=60 <= 0, at either checkpoint"},
               "R1-H3": {"statement": "C_Q under-selects in the majority of d=15 datasets under (L10,it16)",
                         "refute": "fewer than 5 of the complete d=15 datasets are under-selected by C_Q"},
               "reported_not_predicted": ["exact counts per condition", "per-dataset agreement of selected K",
                                          "paired slack changes", "termA, R3, residual by K",
                                          "Gram / eta changes", "Laplace status"]}},
    "R2": {**{k: list(v) if isinstance(v, tuple) else v for k, v in R2.items()},
           "reuse": {"d15_all_n": "EMC checkpoint-8 states re-evaluated (no MCEM)",
                     "n75_d30": "d-sweep"},
           "datasets_across_n": "independent (A1 data seeds depend on n); same replicate number is NOT a repeated measurement",
           "staging": N150_RULE,
           "hypotheses": {
               "R2-H1": {"statement": "at each n, median C_Q slack_45 is larger at d=30 than at d=15 (paired, same data)",
                         "refute": "median at d=30 <= at d=15 at some completed n"},
               "R2-H2": {"statement": "at each n, median 2dH_Lap(4->5) is smaller at d=30 than at d=15",
                         "refute": "median at d=30 >= at d=15 at some completed n"},
               "R2-H3": {"statement": "at d=15, median 2dH_Lap(4->5)/n decreases from n=50 to n=150 (the per-node log-det increment grows with n)",
                         "refute": "median at the largest completed n >= at n=50",
                         "caution": "n changes more than Y information (estimation, penalty ln n, P_Z per node); not attributed to Y alone"},
               "reported_not_predicted": ["exact counts per n and d", "X / Y split of dln|H|", "fit gain vs penalties",
                                          "Gram / eta changes", "margins"]}},
    "failure_rules": ["data / replication / init gate failure -> stop before production",
                      "RetryDetected -> fit FAILED_NAN, no retry, dataset INCOMPLETE for both criteria",
                      "other exception -> stop run, write partial rows",
                      "Laplace not OK -> C_Lap INCOMPLETE; no imputation",
                      "no seed change, no tolerance change, no resume over existing output"],
    "fit_cap": FIT_CAP,
    "smoke_fits": {"R1": ["replication: d=30 rep1 K5 s1 at L5/it8 vs d-sweep state",
                          "pilot: d=60 rep1 K7 s1 at L10/it16"],
                   "R2": ["replication: n=50 d=15 rep1 K5 s1 vs EMC checkpoint 8",
                          "pilot: n=150 d=30 rep1 K7 s1"],
                   "role": "gates only; never counted as data"},
}


def prereg_hash() -> str:
    import hashlib
    return hashlib.sha256(json.dumps(PREREG, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ data (n-generic)
def build_data(n: int, rep: int, prov):
    ds, train, test = a1.make_cell(K_TRUE, n, rep, prov[(K_TRUE, n, rep)])
    scale = hist.f_scale_for_row_norm(hist.ROW_NORM_SQ_TARGET, d=D0, k=K_TRUE)
    need(scale == float(ds.metadata["f_scale"]), "f_scale differs from A1")
    blocks = []
    for b in range(1, max(BLOCKS.values()) + 1):
        rng = np.random.default_rng([hist.data_seed(K_TRUE, n, rep), BLOCK_SEED_TAG + b])
        Fb = build_full_rank_loadings(rng, d=D0, k=K_TRUE, f_scale=scale)
        rate = canonical_poisson_rate(ds.Z @ Fb.T, lambda_max=hist.POISSON_LAMBDA_MAX)
        Xb = rng.poisson(rate).astype(np.float64)
        need(bool(np.all(np.isfinite(Xb))) and bool(np.all(Xb >= 0)), f"block support n{n} r{rep}")
        need(bool(np.allclose(Fb.T @ Fb, scale ** 2 * np.eye(K_TRUE), atol=1e-10)), "block FtF")
        blocks.append((Fb, Xb))
    out = {}
    for d, nb in BLOCKS.items():
        out[d] = {"X": np.hstack([ds.X] + [b[1] for b in blocks[:nb]]),
                  "F": np.vstack([ds.F] + [b[0] for b in blocks[:nb]])}
    return ds, train, test, out, scale


def union_bound(n: int, F: np.ndarray) -> float:
    t = math.log(hist.POISSON_LAMBDA_MAX)
    s = np.sqrt(np.sum(F ** 2, axis=1))
    return float(n * np.sum(np.exp(norm.logsf(t / s))))


def oracle(n: int, ds, X, F, train) -> dict[str, float]:
    model = em_runner.build_model(n, X.shape[1], K_TRUE, hist.L_SAMPLES, hist.FAMILY_X,
                                  hist.FAMILY_Y, train_mask=train, numerics_mode=hist.NUMERICS_MODE)
    model.initialize_params(true_params=None, seed=0)
    model.params.update({"Z": ds.Z.copy(), "F": F, "w0": float(ds.metadata["w0"]),
                         "w": float(ds.metadata["w"]), "var_z": 1.0})
    s, ld = np.linalg.slogdet(lk.negative_hessian(model, X, ds.Y, ds.Z))
    cur = ds2.x_curvature(F, 1.0, ds.Z)
    return {"oracle_logdetH_per_n": float(ld / n) if s > 0 else float("nan"),
            "oracle_logdet_prior_X_per_n": cur["logdet_prior_X"] / n,
            "oracle_xfisher_median_eig": cur["xfisher_median_eig"]}


def data_gates(n: int, d_levels, prov, emc) -> tuple[dict, list[dict[str, Any]]]:
    cache, rows = {}, []
    for rep in REPLICATES:
        ds, train, test, dd, scale = build_data(n, rep, prov)
        ds_b, train_b, test_b, dd_b, _ = build_data(n, rep, prov)
        need(all(np.array_equal(getattr(ds, f), getattr(ds_b, f)) for f in "ZFXY")
             and np.array_equal(train, train_b) and np.array_equal(test, test_b)
             and all(np.array_equal(dd[d]["X"], dd_b[d]["X"]) for d in d_levels),
             f"regeneration not deterministic n{n} r{rep}")
        dseed = hist.data_seed(K_TRUE, n, rep)
        mseeds = set()
        for k in hist.CANDIDATE_K:
            for s in hist.STARTS:
                er = emc[(K_TRUE, n, rep, k, s)]
                ms = hist.model_seed(K_TRUE, n, rep, k, s)
                need(int(er["model_seed"]) == ms and int(er["data_seed"]) == dseed
                     and er["train_mask_hash"] == hist.stable_hash(train)
                     and er["test_mask_hash"] == hist.stable_hash(test), f"EMC identity n{n} r{rep} K{k} s{s}")
                need(ms not in mseeds, "duplicate model seed")
                mseeds.add(ms)
        for d in d_levels:
            X, F = dd[d]["X"], dd[d]["F"]
            need(X.shape == (n, d) and np.array_equal(X[:, :D0], ds.X)
                 and np.array_equal(F[:D0], ds.F), f"original columns changed n{n} d{d}")
            need(bool(np.allclose(F.T @ F, (d / D0) * scale ** 2 * np.eye(K_TRUE), atol=1e-10)), "FtF")
            need(bool(np.all(np.isfinite(X))), "non-finite X")
            need(float(np.max(np.exp(ds.Z @ F.T))) <= hist.POISSON_LAMBDA_MAX, "rate gate")
            rows.append({"n": n, "d": d, "replicate": rep, "data_seed": dseed,
                         "block_seeds": json.dumps([[dseed, BLOCK_SEED_TAG + b] for b in range(1, BLOCKS[d] + 1)]),
                         "FtF_diag": float(np.mean(np.diag(F.T @ F))),
                         "mean_row_norm_sq": float(np.mean(np.sum(F ** 2, 1))),
                         "max_rate_truth": float(np.max(np.exp(ds.Z @ F.T))),
                         "gate_fail_prob_union_bound": union_bound(n, F),
                         "hash_X15": ds2.sha(X[:, :D0]), "hash_X": ds2.sha(X), "hash_Y": ds2.sha(ds.Y),
                         "hash_Z": ds2.sha(ds.Z), "hash_train": ds2.sha(train), "hash_test": ds2.sha(test),
                         **oracle(n, ds, X, F, train)})
        cache[rep] = (ds, train, test, dd)
    return cache, rows


# ------------------------------------------------------------------ evaluation
def evaluate(n: int, model, X, Y, Zs, Z, *, fit_id: int, cp: int) -> dict[str, Any]:
    k = Zs.shape[1]
    d = X.shape[1]
    num_params = k * d - k * (k - 1) // 2
    cq = panel.cq_terms(model, X, Y, Zs, num_params=num_params, n=n)
    need(abs(cq["C_Q_recomposed"] - cq["C_Q"]) <= a1.RECOMPOSE_TOL, "C_Q closure")
    lap = panel.laplace_terms(model, X, Y, Z, num_params=num_params, N=n)
    z_hat = lap.pop("Z_hat")
    t = panel.theta_of(model)
    row = {"num_params": num_params, "C_Q": cq["C_Q"], "D_K": cq["D_K"], "P_Z": cq["P_Z"],
           "P_theta": cq["P_theta"], "q_strict": cq["q_strict"], "w0_hat": t["w0"],
           "w_hat": t["w"], "var_z_hat": t["var_z"], **lap}
    if lap["lap_status"] == "OK":
        need(abs(lap["C_Lap_recomposed"] - lap["C_Lap"]) <= a1.RECOMPOSE_TOL, "C_Lap closure")
        row.update(cpr.qlap_diag(model, X, Y, z_hat, QLAP_SEED_BASE + 1000 * fit_id + cp,
                                 cq["q_strict"], lap["lap_log_observed"]))
        cur = ds2.x_curvature(t["F"], t["var_z"], z_hat)
        row.update({"logdet_prior_X_hat": cur["logdet_prior_X"],
                    "logdet_Y_increment_hat": lap["lap_logdet_H"] - cur["logdet_prior_X"],
                    "xfisher_median_eig_hat": cur["xfisher_median_eig"],
                    "residual": cq["C_Q"] - lap["C_Lap"] - (n * k * C2 - lap["lap_logdet_H"])})
    return row


def fit_records(X, Y, train, k, seed, L, max_iter, checkpoints):
    recs = []
    res = run_em_instrumented(X, Y, family_x=hist.FAMILY_X, family_y=hist.FAMILY_Y, k=k, L=L,
                              max_iter=max_iter, checkpoints=checkpoints, seed=seed,
                              train_mask=train, numerics_mode=hist.NUMERICS_MODE,
                              stability_window=4, on_nan="fail",
                              on_checkpoint=lambda r: recs.append(
                                  {**{kk: vv for kk, vv in r.items() if kk != "model"},
                                   "Z_samples": np.array(r["Z_samples"]), "Z_est": np.array(r["Z_est"]),
                                   "F": np.array(r["F"]), "sigma": np.array(r["sigma"]),
                                   "_eval": None}))
    need(len(recs) == len(checkpoints), "checkpoint records missing")
    return res, recs


def model_from(n, d, k, L, train, rec):
    model = em_runner.build_model(n, d, k, L, hist.FAMILY_X, hist.FAMILY_Y, train_mask=train,
                                  numerics_mode=hist.NUMERICS_MODE)
    model.initialize_params(true_params=None, seed=0)
    model.params.update({"Z": rec["Z_est"].copy(), "F": rec["F"], "sigma": rec["sigma"],
                         "w0": float(rec["w0"]), "w": float(rec["w"]), "var_z": float(rec["var_z"])})
    return model


def arrays_of(rec):
    return {"Zsamples": rec["Z_samples"], "Z": rec["Z_est"], "F": rec["F"], "sigma": rec["sigma"],
            "scalars": np.array([rec["w0"], rec["w"], rec["var_z"]], dtype=np.float64)}


def select(rows, group_keys):
    """argmin of the mean over starts; INCOMPLETE if any fit failed (both criteria)
    or any Laplace evaluation is not OK (C_Lap)."""
    out = []
    groups = sorted({tuple(r[g] for g in group_keys) for r in rows})
    for gk in groups:
        by = {(r["k_est"], r["start"]): r for r in rows if tuple(r[g] for g in group_keys) == gk}
        base = dict(zip(group_keys, gk))
        failed = sorted(f"K{k}s{s}" for (k, s), r in by.items() if r["fit_status"] == "FAILED_NAN")
        for crit in ("C_Q", "C_Lap"):
            bad = list(failed)
            if crit == "C_Lap":
                bad += sorted(f"K{k}s{s}:{r['lap_status']}" for (k, s), r in by.items()
                              if r["fit_status"] != "FAILED_NAN" and r["lap_status"] != "OK")
            if bad:
                out.append({**base, "criterion": crit, "status": "INCOMPLETE", "selected_k": "",
                            "label": "", "margin": "", "runner_up": "", "incomplete_fits": ";".join(bad)})
                continue
            means = {k: float(np.mean([by[(k, s)][crit] for s in hist.STARTS])) for k in hist.CANDIDATE_K}
            chosen, _t, margin, _b = a1._select(means, True)
            runner = min((k for k in hist.CANDIDATE_K if k != chosen), key=lambda k: means[k])
            out.append({**base, "criterion": crit, "status": "COMPLETE", "selected_k": chosen,
                        "label": a1.label(chosen, K_TRUE), "margin": margin, "runner_up": runner,
                        "incomplete_fits": ""})
    return out


# ------------------------------------------------------------------ helpers
def common_inputs():
    need(a1.working_tree_clean(), "working tree must be clean")
    need(a1._git("rev-parse", "HEAD:expfam/src/experimental/em_runner_instrumented.py")
         == ds2.INSTRUMENTED_BLOB, "instrumented runner blob")
    need(a1.sha256_file(a1.EMC_NPZ) == a1.EMC_NPZ_SHA256, "EMC npz sha256")
    prov = a1.load_hist_provenance()
    emc, digest = a1.load_emc_identity()
    meta = {"experiment_id": EXPERIMENT_ID, "git_head": a1.git_head(),
            "branch": a1._git("rev-parse", "--abbrev-ref", "HEAD"), "working_tree_clean": True,
            "environment": a1.environment(), "prereg_sha256": prereg_hash(),
            "inputs": {"emc_npz_sha256": a1.EMC_NPZ_SHA256, "emc_fit_results_sha256": digest,
                       "dsweep_fit_criteria_sha256": a1.sha256_file(DSWEEP_FULL / "fit_criteria.csv"),
                       "exp1_fit_criteria_sha256": a1.sha256_file(EXP1_FULL / "fit_criteria.csv")}}
    return prov, emc, meta


def resources() -> dict[str, float]:
    free_d = shutil.disk_usage("D:\\").free / 1e9
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        ram = m.ullAvailPhys / 1e9
    except Exception:                                        # pragma: no cover
        ram = float("nan")
    return {"free_disk_D_GB": round(free_d, 2), "free_ram_GB": round(ram, 2)}


def fit_count_so_far() -> int:
    total = 0
    for p in OUT_ROOT.rglob("runinfo.json"):
        total += int(json.loads(p.read_text(encoding="utf-8")).get("new_fits", 0))
    for p in OUT_ROOT.rglob("gates.json"):
        total += int(json.loads(p.read_text(encoding="utf-8")).get("smoke_fits", 0))
    return total


def load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_npz(path: Path, arrays) -> str:
    need(not path.exists(), f"state file exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return a1.sha256_file(path)


# ------------------------------------------------------------------ production loop
def production(n: int, d_levels, L: int, max_iter: int, checkpoints, out: Path,
               state_path: Path, fit_id_base: int, cache) -> tuple[list, int, int, dict]:
    rows, arrays, local, failed = [], {}, 0, 0
    started = time.time()
    for d in d_levels:
        for rep in REPLICATES:
            ds, train, _t, dd = cache[rep]
            for k in hist.CANDIDATE_K:
                for s in hist.STARTS:
                    local += 1
                    fit_id = fit_id_base + d * 1000 + local
                    seed = hist.model_seed(K_TRUE, n, rep, k, s)
                    base = {"n": n, "d": d, "L": L, "replicate": rep, "k_est": k, "start": s,
                            "fit_id": fit_id, "model_seed": seed}
                    t0 = time.perf_counter()
                    try:
                        res, recs = fit_records(dd[d]["X"], ds.Y, train, k, seed, L, max_iter, checkpoints)
                    except RetryDetected as exc:
                        failed += 1
                        for cp in checkpoints:
                            rows.append({**base, "checkpoint": cp, "fit_status": "FAILED_NAN",
                                         "lap_status": "NOT_EVALUATED", "failure": str(exc)})
                        continue
                    em_s = time.perf_counter() - t0
                    for rec in recs:
                        cp = int(rec["iteration"])
                        model = model_from(n, d, k, L, train, rec)
                        t1 = time.perf_counter()
                        r = evaluate(n, model, dd[d]["X"], ds.Y, rec["Z_samples"], rec["Z_est"],
                                     fit_id=fit_id, cp=cp)
                        need(math.isclose(r["C_Q"], rec["bic"], rel_tol=1e-9, abs_tol=1e-6),
                             "panel C_Q differs from runner BIC")
                        for name, arr in arrays_of(rec).items():
                            arrays[f"n{n}_d{d}_L{L}_cp{cp}_r{rep}_k{k}_s{s}_{name}"] = arr
                        rows.append({**base, "checkpoint": cp, "fit_status": "FIT",
                                     "em_runtime_s": round(em_s, 3),
                                     "eval_runtime_s": round(time.perf_counter() - t1, 3),
                                     **{c: float(rec[c]) for c in ds2.DIAG_COLS}, **r})
            print(f"[n={n} L={L}] d={d} rep={rep} done ({time.time() - started:.0f}s, failed={failed})",
                  flush=True)
    sha = write_npz(state_path, arrays)
    return rows, local, failed, {"fit_states": str(state_path), "fit_states_sha256": sha,
                                 "wall_clock_seconds": round(time.time() - started, 1)}


def finish(out: Path, meta, rows, new_fits, failed, extra, group_keys):
    sel = select(rows, group_keys)
    write_rows(out / "fit_criteria.csv", rows)
    write_rows(out / "selection.csv", sel)
    statuses: dict[str, int] = {}
    for r in rows:
        key = f"d{r['d']}:cp{r['checkpoint']}:{r['fit_status']}:{r['lap_status']}"
        statuses[key] = statuses.get(key, 0) + 1
    a1.write_json(out / "runinfo.json", {**meta, "status": "COMPLETED", "rows": len(rows),
                                         "new_fits": new_fits, "failed_nan_fits": failed,
                                         "status_counts": statuses, **extra})


def guarded(out: Path, meta, fn):
    try:
        return fn()
    except Exception as exc:
        a1.write_json(out / "failure.json", {**meta, "status": "STOPPED", "error_type": type(exc).__name__,
                                             "error": str(exc)})
        print(f"STOPPED: {type(exc).__name__}: {exc}", flush=True)
        return 2


# ------------------------------------------------------------------ R1
def r1_gates() -> int:
    out = OUT_ROOT / "r1_gates"
    need(not out.exists(), f"exists {out}")
    prov, emc, meta = common_inputs()
    out.mkdir(parents=True)
    a1.write_json(out / "prereg.json", {**PREREG, "sha256": prereg_hash()})

    def body():
        n = R1["n"]
        cache, drows = data_gates(n, R1["d"], prov, emc)
        write_rows(out / "data_provenance.csv", drows)
        ref = {(int(r["d"]), int(r["replicate"])): r for r in load_csv(DSWEEP_FULL / "data_provenance.csv")}
        for r in drows:
            q = ref[(r["d"], r["replicate"])]
            need(all(r[c] == q[c] for c in ("hash_X", "hash_Y", "hash_Z", "hash_train", "hash_test")),
                 f"data differ from the d-sweep d{r['d']} r{r['replicate']}")
        # initialisation identical across L
        ds, train, _t, dd = cache[1]
        for d in R1["d"]:
            for k in hist.CANDIDATE_K:
                p = []
                for L in (hist.L_SAMPLES, R1["L"]):
                    m = em_runner.build_model(n, d, k, L, hist.FAMILY_X, hist.FAMILY_Y, train_mask=train,
                                              numerics_mode=hist.NUMERICS_MODE)
                    m.initialize_params(true_params=None, seed=hist.model_seed(K_TRUE, n, 1, k, 1))
                    p.append({kk: np.array(v, dtype=float) for kk, v in m.params.items()})
                need(all(np.array_equal(p[0][kk], p[1][kk]) for kk in p[0]), f"init differs with L d{d} K{k}")
        gates = {"G_data_equals_dsweep": "PASS", "G_init_identical_L5_L10": "PASS"}
        # replication of one d-sweep fit through this runner
        dz = np.load(ds2.STATE_ROOT / "fit_states.npz")
        res, recs = fit_records(dd[30]["X"], ds.Y, train, 5, hist.model_seed(K_TRUE, n, 1, 5, 1),
                                hist.L_SAMPLES, 8, (8,))
        mine = arrays_of(recs[0])
        diffs = {nm: float(np.max(np.abs(mine[nm] - dz[f"d30_rep1_k5_s1_{nm}"]))) for nm in mine}
        need(all(v == 0.0 for v in diffs.values()), f"replication differs {diffs}")
        dref = [r for r in load_csv(DSWEEP_FULL / "fit_criteria.csv")
                if (r["d"], r["replicate"], r["k_est"], r["start"]) == ("30", "1", "5", "1")][0]
        ev = evaluate(n, model_from(n, 30, 5, hist.L_SAMPLES, train, recs[0]), dd[30]["X"], ds.Y,
                      recs[0]["Z_samples"], recs[0]["Z_est"], fit_id=int(dref["fit_id"]), cp=8)
        for col in ("C_Q", "C_Lap", "termA", "R3", "logdet_prior_X_hat"):
            need(math.isclose(ev[col], float(dref[col]), rel_tol=1e-12, abs_tol=1e-9),
                 f"replication evaluation {col}: {ev[col]} vs {dref[col]}")
        gates["G_replication_dsweep_d30_r1_k5_s1"] = f"PASS bitwise state {diffs}; evaluation equal"
        # pilot: heaviest R1 fit
        t0 = time.perf_counter()
        res, recs = fit_records(dd[60]["X"], ds.Y, train, 7, hist.model_seed(K_TRUE, n, 1, 7, 1),
                                R1["L"], R1["max_iter"], R1["checkpoints"])
        em_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        evs = []
        for rec in recs:
            model = model_from(n, 60, 7, R1["L"], train, rec)
            evs.append(evaluate(n, model, dd[60]["X"], ds.Y, rec["Z_samples"], rec["Z_est"],
                                fit_id=999999, cp=int(rec["iteration"])))
        ev_s = time.perf_counter() - t1
        need(all(math.isfinite(e["C_Q"]) and math.isfinite(e["C_Lap"]) for e in evs), "pilot non-finite")
        gates["G_pilot_d60_k7_L10_it16"] = {"em_seconds": round(em_s, 1), "eval_seconds": round(ev_s, 1),
                                            "lap_status": [e["lap_status"] for e in evs],
                                            "projected_hours_upper": round(336 * (em_s + ev_s) / 3600, 2)}
        a1.write_json(out / "gates.json", {**meta, "status": "PASSED", "gates": gates, "smoke_fits": 2,
                                           "resources": resources()})
        print(json.dumps(gates, indent=1, default=str))
        return 0
    rc = guarded(out, meta, body)
    if rc and not (out / "gates.json").exists():
        a1.write_json(out / "gates.json", {**meta, "status": "FAILED", "smoke_fits": 2})
    return rc


def r1_full() -> int:
    out = OUT_ROOT / "r1"
    need(not out.exists(), f"exists {out}")
    g = json.loads((OUT_ROOT / "r1_gates" / "gates.json").read_text(encoding="utf-8"))
    need(g["status"] == "PASSED" and g["prereg_sha256"] == prereg_hash(), "R1 gates")
    need(fit_count_so_far() + 336 <= FIT_CAP, "fit cap")
    prov, emc, meta = common_inputs()
    out.mkdir(parents=True)

    def body():
        cache, drows = data_gates(R1["n"], R1["d"], prov, emc)
        write_rows(out / "data_provenance.csv", drows)
        rows, new, failed, extra = production(R1["n"], R1["d"], R1["L"], R1["max_iter"], R1["checkpoints"],
                                              out, STATE_ROOT / "r1_fit_states.npz", 100000, cache)
        need(new == 336, f"new fits {new}")
        finish(out, meta, rows, new, failed, extra, ("d", "checkpoint", "replicate"))
        return 0
    return guarded(out, meta, body)


# ------------------------------------------------------------------ R2
def eval_d15_reuse(n, cache, emc, npz, exp1_rows, cp_rows):
    cpd = {(int(r["fit_index"]), int(r["checkpoint_iteration"])): r for r in cp_rows}
    e1 = {(int(r["replicate"]), int(r["k_est"]), int(r["start"])): r for r in exp1_rows
          if int(r["n"]) == n and int(r["checkpoint"]) == 8}
    rows = []
    for rep in REPLICATES:
        ds, train, _t, dd = cache[rep]
        for k in hist.CANDIDATE_K:
            for s in hist.STARTS:
                fi = int(emc[(K_TRUE, n, rep, k, s)]["fit_index"])
                key, model, Zs, Z = cpr.state(npz, fi, 8, n, k, train)
                r = evaluate(n, model, dd[15]["X"], ds.Y, Zs, Z, fit_id=fi, cp=8)
                e = e1[(rep, k, s)]
                need(r["C_Q"] == float(e["C_Q"]) and r["lap_status"] == e["lap_status"], f"d15 vs exp1 {key}")
                if r["lap_status"] == "OK":
                    for col in ("C_Lap", "termA", "R3"):
                        need(math.isclose(r[col], float(e[col]), rel_tol=1e-12, abs_tol=1e-9), f"{col} {key}")
                c = cpd[(fi, 8)]
                rows.append({"n": n, "d": 15, "L": hist.L_SAMPLES, "replicate": rep, "k_est": k,
                             "start": s, "fit_id": fi, "model_seed": int(emc[(K_TRUE, n, rep, k, s)]["model_seed"]),
                             "checkpoint": 8, "fit_status": "REUSED", "source": f"emc_state:{key}",
                             **{col: float(c[col]) for col in ds2.DIAG_COLS}, **r})
    return rows


def r2_gates() -> int:
    out = OUT_ROOT / "r2_gates"
    need(not out.exists(), f"exists {out}")
    prov, emc, meta = common_inputs()
    out.mkdir(parents=True)

    def body():
        gates, drows_all = {}, []
        npz = np.load(a1.EMC_NPZ)
        exp1_rows = load_csv(EXP1_FULL / "fit_criteria.csv")
        cp_rows, _ = cpr.git_blob(cpr.EMC_CP_RESULTS)
        caches = {}
        for n in R2["n"]:
            cache, drows = data_gates(n, (15, 30), prov, emc)
            drows_all += drows
            caches[n] = cache
            eval_d15_reuse(n, cache, emc, npz, exp1_rows, cp_rows)
            gates[f"G_data_and_d15_reevaluation_n{n}"] = "PASS"
        write_rows(out / "data_provenance.csv", drows_all)
        ds, train, _t, dd = caches[50][1]
        fi = int(emc[(K_TRUE, 50, 1, 5, 1)]["fit_index"])
        res, recs = fit_records(dd[15]["X"], ds.Y, train, 5, hist.model_seed(K_TRUE, 50, 1, 5, 1),
                                hist.L_SAMPLES, 8, (8,))
        mine = arrays_of(recs[0])
        key = f"{fi:04d}_it008"
        diffs = {nm: float(np.max(np.abs(mine[nm] - np.asarray(npz[f"{key}_{nm}"], float))))
                 for nm in ("Zsamples", "F", "sigma", "scalars")}
        need(all(v == 0.0 for v in diffs.values()), f"replication n50 differs {diffs}")
        gates["G_replication_emc_n50_d15_r1_k5_s1"] = f"PASS bitwise {diffs}"
        ds, train, _t, dd = caches[150][1]
        t0 = time.perf_counter()
        res, recs = fit_records(dd[30]["X"], ds.Y, train, 7, hist.model_seed(K_TRUE, 150, 1, 7, 1),
                                hist.L_SAMPLES, 8, (8,))
        em_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        ev = evaluate(150, model_from(150, 30, 7, hist.L_SAMPLES, train, recs[0]), dd[30]["X"], ds.Y,
                      recs[0]["Z_samples"], recs[0]["Z_est"], fit_id=999998, cp=8)
        ev_s = time.perf_counter() - t1
        gates["G_pilot_n150_d30_k7"] = {"em_seconds": round(em_s, 1), "eval_seconds": round(ev_s, 1),
                                        "lap_status": ev["lap_status"],
                                        "projected_hours_n150": round(112 * (em_s + ev_s) * 1.25 / 3600, 2)}
        a1.write_json(out / "gates.json", {**meta, "status": "PASSED", "gates": gates, "smoke_fits": 2,
                                           "resources": resources()})
        print(json.dumps(gates, indent=1, default=str))
        return 0
    rc = guarded(out, meta, body)
    if rc and not (out / "gates.json").exists():
        a1.write_json(out / "gates.json", {**meta, "status": "FAILED", "smoke_fits": 2})
    return rc


def r2_decide_n150() -> int:
    out = OUT_ROOT / "r2_n150_decision.json"
    need(not out.exists(), "decision already recorded")
    g = json.loads((OUT_ROOT / "r2_gates" / "gates.json").read_text(encoding="utf-8"))
    done = all((OUT_ROOT / f"r2_n{n}" / "runinfo.json").exists() for n in (50, 100))
    proj = float(g["gates"]["G_pilot_n150_d30_k7"]["projected_hours_n150"])
    res = resources()
    ok = done and proj <= 3.0 and res["free_disk_D_GB"] >= 10 and res["free_ram_GB"] >= 2
    a1.write_json(out, {"rule": N150_RULE, "n50_n100_completed": done, "projected_hours": proj,
                        "resources": res, "decision": "RUN" if ok else "NOT_RUN_STAGED"})
    print("decision:", "RUN" if ok else "NOT_RUN_STAGED")
    return 0


def r2_full(n: int) -> int:
    need(n in R2["n"], f"n={n}")
    out = OUT_ROOT / f"r2_n{n}"
    need(not out.exists(), f"exists {out}")
    g = json.loads((OUT_ROOT / "r2_gates" / "gates.json").read_text(encoding="utf-8"))
    need(g["status"] == "PASSED" and g["prereg_sha256"] == prereg_hash(), "R2 gates")
    if n == 150:
        dec = json.loads((OUT_ROOT / "r2_n150_decision.json").read_text(encoding="utf-8"))
        need(dec["decision"] == "RUN", "n=150 not authorised by the staging rule")
    need(fit_count_so_far() + 112 <= FIT_CAP, "fit cap")
    prov, emc, meta = common_inputs()
    out.mkdir(parents=True)

    def body():
        cache, drows = data_gates(n, (15, 30), prov, emc)
        write_rows(out / "data_provenance.csv", drows)
        reuse = eval_d15_reuse(n, cache, emc, np.load(a1.EMC_NPZ), load_csv(EXP1_FULL / "fit_criteria.csv"),
                               cpr.git_blob(cpr.EMC_CP_RESULTS)[0])
        rows, new, failed, extra = production(n, (30,), hist.L_SAMPLES, 8, (8,), out,
                                              STATE_ROOT / f"r2_n{n}_fit_states.npz", 200000 + n * 10000, cache)
        need(new == 112, f"new fits {new}")
        finish(out, meta, reuse + rows, new, failed, extra, ("d", "checkpoint", "replicate"))
        return 0
    return guarded(out, meta, body)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--r1-gates", action="store_true")
    g.add_argument("--r1", action="store_true")
    g.add_argument("--r2-gates", action="store_true")
    g.add_argument("--r2", type=int)
    g.add_argument("--r2-decide-n150", action="store_true")
    a = ap.parse_args(argv)
    if a.r1_gates:
        return r1_gates()
    if a.r1:
        return r1_full()
    if a.r2_gates:
        return r2_gates()
    if a.r2_decide_n150:
        return r2_decide_n150()
    return r2_full(a.r2)


if __name__ == "__main__":
    sys.exit(main())
