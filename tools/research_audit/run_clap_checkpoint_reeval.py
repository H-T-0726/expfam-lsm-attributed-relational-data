"""Re-evaluate C_Lap (and C_Q) on the saved EM-convergence checkpoints.

Question: is the C_Lap selection observed at EM iteration 8 (Issue #128 A1,
K_TRUE = 5) stable when the SAME fits are read at later saved checkpoints
(12, 16, 24, 32, 40)?  No MCEM is run: every number is a deterministic function
of a saved state (theta, E-step samples, last sample) plus, for the q_Lap
diagnostic, a seeded Gaussian draw.

Inputs (read-only):
- states: ``D:\\tento\\kennkyu_artifacts\\em_convergence_sensitivity_20260905\\
  checkpoint_states.npz`` (sha256 pinned)
- identity / historical C_Q per checkpoint: git objects of the unmerged branch
  commit ``dad67b1`` (``fit_results.csv``, ``checkpoint_results.csv``)
- A1 checkpoint-8 C_Lap: ``expfam/results/k_selection/clap_clean_nsweep_
  20261006/full/fit_criteria.csv`` (reproduction gate)

Gates (fail closed, no retry / imputation / tolerance change):
- npz sha256; recomputed C_Q equals ``checkpoint_results.csv`` s2_q_based
  (rtol 1e-9, atol 1e-6, inherited from the audited EMC auditor)
- checkpoint 8: lap_status equal to A1 and C_Lap / logdet_H equal within
  rtol 1e-9 (same code, same input)
A Laplace evaluation that is NOT_STATIONARY / HESSIAN_NOT_PD is a recorded
result; that dataset's C_Lap selection at that checkpoint is INCOMPLETE.

Extra diagnostic (C_Q - C_Lap relation), OK fits only, at the checkpoints in
``QLAP_CHECKPOINTS``:  draws Z ~ q_Lap = N(Z_hat, H^-1), M = QLAP_DRAWS,
  Q_Lap   = E_qLap[ln p(X, Y, Z | theta)]          (MC, with standard error)
  H_Lap   = (nK/2)(1 + ln 2 pi) - 1/2 ln|H|
  R3      = l_Lap - Q_Lap - H_Lap  (= Phi(Z_hat) - nK/2 - E_qLap Phi)
  termA   = -2 (Q_MC - Q_Lap)      (E-step samples, L = 5, vs q_Lap)
so that exactly  C_Q - C_Lap - 2 H_Lap = termA + 2 R3 + (P_theta_Q - P_theta_Lap).

Lineage E (experimental prototype).  MANUSCRIPT_ADOPTION_STATUS:
NOT_DECIDED_BY_HUMAN.

Usage:
    python run_clap_checkpoint_reeval.py --pilot
    python run_clap_checkpoint_reeval.py --full
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
_EXPERIMENTAL = ROOT / "expfam" / "src" / "experimental"
for _p in (str(_EXPERIMENTAL), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_clap_clean_nsweep as a1                          # noqa: E402
import k_criterion_panel as panel                           # noqa: E402
import laplace_k_criterion as lk                            # noqa: E402

hist = a1.hist

EXPERIMENT_ID = "clap_checkpoint_reeval_20261105"
OUT_ROOT = ROOT / "expfam" / "results" / "k_selection" / EXPERIMENT_ID
EMC_CP_RESULTS = ("expfam/results/k_selection/em_convergence_sensitivity_"
                  "20260905/checkpoint_results.csv")
A1_FIT = (ROOT / "expfam" / "results" / "k_selection" /
          "clap_clean_nsweep_20261006" / "full" / "fit_criteria.csv")
CHECKPOINTS = (8, 12, 16, 24, 32, 40)
PILOT_CELLS = ((5, 50, 1), (5, 100, 1))
PILOT_CHECKPOINTS = (8, 40)
QLAP_CHECKPOINTS = (8, 40)
QLAP_DRAWS = 200
QLAP_CHUNK = 20
QLAP_SEED_BASE = 20261105
A1_RTOL = 1e-9
K_TRUE = 5
CRITERIA = ("C_Q", "C_Lap")


class Stop(RuntimeError):
    pass


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise Stop(msg)


def git_blob(path: str) -> tuple[list[dict[str, str]], str]:
    raw = subprocess.check_output(["git", "show", f"{a1.EMC_COMMIT}:{path}"],
                                  cwd=ROOT)
    return (list(csv.DictReader(io.StringIO(raw.decode("utf-8")))),
            hashlib.sha256(raw).hexdigest())


def load_cp_cq() -> tuple[dict[tuple[int, int], float], str]:
    rows, digest = git_blob(EMC_CP_RESULTS)
    out = {(int(r["fit_index"]), int(r["checkpoint_iteration"])):
           float(r["s2_q_based"]) for r in rows}
    need(len(out) == 448 * len(CHECKPOINTS), f"checkpoint rows {len(out)}")
    return out, digest


def load_a1() -> dict[tuple[int, int, int, int], dict[str, str]]:
    with open(A1_FIT, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["k_true"]) == K_TRUE]
    out = {(int(r["n"]), int(r["replicate"]), int(r["k_est"]),
            int(r["start"])): r for r in rows}
    need(len(out) == 448, f"A1 K5 rows {len(out)}")
    return out


def state(npz, fit_index: int, cp: int, n: int, k: int, train):
    import em_runner                                        # noqa: PLC0415
    key = f"{fit_index:04d}_it{cp:03d}"
    Zs = np.asarray(npz[f"{key}_Zsamples"], dtype=np.float64)
    Z = np.asarray(npz[f"{key}_Z"], dtype=np.float64)
    F = np.asarray(npz[f"{key}_F"], dtype=np.float64)
    sigma = np.asarray(npz[f"{key}_sigma"], dtype=np.float64)
    w0, w, var_z = (float(v) for v in npz[f"{key}_scalars"])
    need(Zs.shape == (n, k, hist.L_SAMPLES), f"Zsamples shape {key}")
    need(np.array_equal(Z, Zs[:, :, -1]), f"Z != last sample {key}")
    model = em_runner.build_model(n, hist.D_FEATURES, k, hist.L_SAMPLES,
                                  hist.FAMILY_X, hist.FAMILY_Y,
                                  train_mask=train,
                                  numerics_mode=hist.NUMERICS_MODE)
    model.initialize_params(true_params=None, seed=0)
    model.params.update({"Z": Z.copy(), "F": F, "sigma": sigma, "w0": w0,
                         "w": w, "var_z": var_z})
    return key, model, Zs, Z


def qlap_diag(model, X, Y, z_hat, seed: int, q_mc: float,
              log_obs: float) -> dict[str, float]:
    n, k = z_hat.shape
    H = lk.negative_hessian(model, X, Y, z_hat)
    Lc = np.linalg.cholesky(H)                 # H is PD for an OK evaluation
    logdet = 2.0 * float(np.sum(np.log(np.diag(Lc))))
    rng = np.random.default_rng(seed)
    t = panel.theta_of(model)
    means = []
    for _ in range(QLAP_DRAWS // QLAP_CHUNK):
        eps = rng.standard_normal((n * k, QLAP_CHUNK))
        # x = Lc^{-T} eps has covariance H^{-1}
        dz = np.linalg.solve(Lc.T, eps)        # (nK, chunk)
        Zd = z_hat.reshape(n * k, 1) + dz
        Zd = Zd.reshape(n, k, QLAP_CHUNK)
        means.append(float(panel.calc_Q_dual_strict_exp(
            X, Y, Zd, t["F"], t["sigma"], t["var_z"], t["w0"], t["w"],
            model)))
    q_lap = float(np.mean(means))
    se = float(np.std(means, ddof=1) / math.sqrt(len(means)))
    h_lap = 0.5 * n * k * (1.0 + math.log(2.0 * math.pi)) - 0.5 * logdet
    return {"Q_Lap": q_lap, "Q_Lap_se": se, "H_Lap": h_lap,
            "logdet_check": logdet,
            "R3": log_obs - q_lap - h_lap,
            "termA": -2.0 * (q_mc - q_lap)}


def evaluate(cells, cps, out_dir: Path, stage: str) -> int:
    need(not out_dir.exists(), f"output exists: {out_dir}")
    need(a1.working_tree_clean(), "working tree must be clean")
    need(a1.sha256_file(a1.EMC_NPZ) == a1.EMC_NPZ_SHA256, "npz sha256")
    emc, emc_digest = a1.load_emc_identity()
    cp_cq, cp_digest = load_cp_cq()
    a1rows = load_a1()
    prov = a1.load_hist_provenance()
    a1_sha = a1.sha256_file(A1_FIT)
    out_dir.mkdir(parents=True)
    started = time.time()
    meta = {"experiment_id": EXPERIMENT_ID, "stage": stage,
            "git_head": a1.git_head(),
            "branch": a1._git("rev-parse", "--abbrev-ref", "HEAD"),
            "environment": a1.environment(),
            "inputs": {"npz": str(a1.EMC_NPZ), "npz_sha256": a1.EMC_NPZ_SHA256,
                       "emc_fit_results": f"{a1.EMC_COMMIT}:{a1.EMC_FIT_RESULTS}",
                       "emc_fit_results_sha256": emc_digest,
                       "emc_checkpoint_results": f"{a1.EMC_COMMIT}:{EMC_CP_RESULTS}",
                       "emc_checkpoint_results_sha256": cp_digest,
                       "a1_fit_criteria": str(A1_FIT.relative_to(ROOT)),
                       "a1_fit_criteria_sha256": a1_sha},
            "cells": [list(c) for c in cells], "checkpoints": list(cps),
            "laplace": {"grad_tol": panel.LAPLACE_GRAD_TOL,
                        "max_iter": panel.LAPLACE_MAX_ITER,
                        "Z0": "last E-step sample of the checkpoint (as A1)"},
            "qlap": {"checkpoints": list(QLAP_CHECKPOINTS),
                     "draws": QLAP_DRAWS, "chunk": QLAP_CHUNK,
                     "seed": "QLAP_SEED_BASE + 1000*fit_index + checkpoint"},
            "gates": {"C_Q_vs_emc": {"rtol": a1.NPZ_RTOL, "atol": a1.NPZ_ATOL},
                      "cp8_vs_A1": {"status": "equal", "rtol": A1_RTOL}},
            "failure_policy": ["no_mcem", "no_retry", "no_imputation",
                               "no_tolerance_change", "no_resume"]}
    rows: list[dict[str, Any]] = []
    npz = np.load(a1.EMC_NPZ)
    try:
        for cell in cells:
            k_true, n, rep = cell
            ds, train, _test = a1.make_cell(k_true, n, rep, prov[cell])
            for cp in cps:
                for k in hist.CANDIDATE_K:
                    for s in hist.STARTS:
                        er = emc[(k_true, n, rep, k, s)]
                        fi = int(er["fit_index"])
                        need(er["train_mask_hash"] == hist.stable_hash(train),
                             f"mask identity {cell, k, s}")
                        key, model, Zs, Z = state(npz, fi, cp, n, k, train)
                        num_params = k * hist.D_FEATURES - k * (k - 1) // 2
                        cq = panel.cq_terms(model, ds.X, ds.Y, Zs,
                                            num_params=num_params, n=n)
                        ref = cp_cq[(fi, cp)]
                        need(math.isclose(cq["C_Q"], ref, rel_tol=a1.NPZ_RTOL,
                                          abs_tol=a1.NPZ_ATOL),
                             f"C_Q mismatch {key}: {cq['C_Q']!r} vs {ref!r}")
                        lap = panel.laplace_terms(model, ds.X, ds.Y, Z,
                                                  num_params=num_params, N=n)
                        z_hat = lap.pop("Z_hat")
                        row = {"n": n, "replicate": rep, "k_est": k,
                               "start": s, "checkpoint": cp, "fit_index": fi,
                               "state_key": key, "num_params": num_params,
                               "C_Q": cq["C_Q"], "C_Q_emc": ref,
                               "D_K": cq["D_K"], "P_Z": cq["P_Z"],
                               "P_theta": cq["P_theta"],
                               "q_strict": cq["q_strict"],
                               "w0_hat": float(model.params["w0"]),
                               "w_hat": float(model.params["w"]), **lap}
                        if cp == 8:
                            ar = a1rows[(n, rep, k, s)]
                            need(ar["lap_status"] == lap["lap_status"],
                                 f"cp8 status differs {key}")
                            if lap["lap_status"] == "OK":
                                for col, mine in (("C_Lap", lap["C_Lap"]),
                                                  ("lap_logdet_H",
                                                   lap["lap_logdet_H"])):
                                    need(math.isclose(mine, float(ar[col]),
                                                      rel_tol=A1_RTOL),
                                         f"cp8 {col} differs {key}")
                                row["absdiff_C_Lap_A1"] = abs(
                                    lap["C_Lap"] - float(ar["C_Lap"]))
                        if lap["lap_status"] == "OK":
                            need(abs(lap["C_Lap_recomposed"] - lap["C_Lap"])
                                 <= a1.RECOMPOSE_TOL, f"C_Lap closure {key}")
                            if cp in QLAP_CHECKPOINTS:
                                row.update(qlap_diag(
                                    model, ds.X, ds.Y, z_hat,
                                    QLAP_SEED_BASE + 1000 * fi + cp,
                                    cq["q_strict"], lap["lap_log_observed"]))
                        rows.append(row)
                print(f"[{stage}] cell={cell} cp={cp} done "
                      f"({time.time() - started:.0f}s)", flush=True)
    except Exception as exc:
        a1.write_csv(out_dir / "fit_criteria_partial.csv", rows)
        a1.write_json(out_dir / "failure.json",
                      {**meta, "status": "STOPPED",
                       "error_type": type(exc).__name__, "error": str(exc),
                       "completed_rows": len(rows)})
        print(f"STOPPED: {type(exc).__name__}: {exc}", flush=True)
        return 2

    sel = select(rows)
    a1.write_csv(out_dir / "fit_criteria.csv", rows)
    a1.write_csv(out_dir / "selection.csv", sel)
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[f"cp{r['checkpoint']}:{r['lap_status']}"] = \
            statuses.get(f"cp{r['checkpoint']}:{r['lap_status']}", 0) + 1
    done = time.time()
    a1.write_json(out_dir / "runinfo.json", {
        **meta, "status": "COMPLETED", "rows": len(rows),
        "wall_clock_seconds": round(done - started, 1),
        "laplace_status_counts": statuses,
        "max_absdiff_C_Q_vs_emc": max(abs(r["C_Q"] - r["C_Q_emc"])
                                      for r in rows),
        "max_absdiff_C_Lap_cp8_vs_A1": max(
            [r.get("absdiff_C_Lap_A1", 0.0) for r in rows], default=0.0),
        "mcem_fits_run": 0})
    print(f"[{stage}] completed {len(rows)} rows; {statuses}", flush=True)
    return 0


def select(rows) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(r["n"], r["replicate"], r["checkpoint"]) for r in rows})
    for n, rep, cp in keys:
        by = {(r["k_est"], r["start"]): r for r in rows
              if (r["n"], r["replicate"], r["checkpoint"]) == (n, rep, cp)}
        ks = sorted({k for k, _ in by})
        for crit in CRITERIA:
            if crit == "C_Lap":
                bad = sorted({f"K{k}s{s}:{by[(k, s)]['lap_status']}"
                              for k in ks for s in hist.STARTS
                              if by[(k, s)]["lap_status"] != "OK"})
                if bad:
                    out.append({"criterion": crit, "n": n, "replicate": rep,
                                "checkpoint": cp, "status": "INCOMPLETE",
                                "selected_k": "", "label": "", "margin": "",
                                "runner_up": "", "incomplete_fits": ";".join(bad)})
                    continue
            means = {k: float(np.mean([by[(k, s)][crit] for s in hist.STARTS]))
                     for k in ks}
            chosen, _ties, margin, _best = a1._select(means, True)
            runner = min((k for k in ks if k != chosen), key=lambda k: means[k])
            out.append({"criterion": crit, "n": n, "replicate": rep,
                        "checkpoint": cp, "status": "COMPLETE",
                        "selected_k": chosen,
                        "label": a1.label(chosen, K_TRUE), "margin": margin,
                        "runner_up": runner, "incomplete_fits": ""})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--full", action="store_true")
    a = ap.parse_args(argv)
    if a.pilot:
        return evaluate(list(PILOT_CELLS), PILOT_CHECKPOINTS,
                        OUT_ROOT / "pilot", "pilot")
    cells = [(K_TRUE, n, r) for n in hist.N_GRID
             for r in range(1, hist.REPLICATES_BY_K_TRUE[K_TRUE] + 1)]
    return evaluate(cells, CHECKPOINTS, OUT_ROOT / "full", "full")


if __name__ == "__main__":
    sys.exit(main())
