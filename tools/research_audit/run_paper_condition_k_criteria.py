"""Issue #128 experiment B: K-selection criteria on the paper-type generator.

NOT a reproduction of Mikawa et al. (2024) Experiment 2 and NOT a re-run of
``reproduction/src/experiment_paper_2.py``.  Purpose: on data drawn by the
paper-type generator ``reproduction/src/data_generator.py::set_true_params``
(unchanged), fit each candidate K ONCE and evaluate several criteria on that
same fit.

Differences from the historical reproduction (stated in every report):
estimator = objective-consistent ``DualExpFamLSMConsistent`` (lineage E), not
``LatentStructuralModel``; L = 5 and 8 EM iterations with 2 starts (mean of
starts), not L = 10 / 30 iterations / best of 3 trials; full Y, no held-out.

Criteria (all on the same fit):
- C_Q   primary   legacy Q-based complete-data / ICL-type (NOT Schwarz BIC)
- C_Lap primary   Laplace-evidence candidate (NOT exact evidence)
- C_cond_MC / C_cond_last / C_cond_mode   paper-inspired conditional
        diagnostics; none of them is the paper's BIC (the paper does not
        identify its Z evaluation point).
Penalty for every criterion: num_params * ln n with
num_params = K d - K(K-1)/2 + d  (= (K+1)d - K(K-1)/2, the paper's count).

Failure policy: ``failure_policy='fail_fast'``; an EM exception or non-finite
C_Q is recorded as a FAILED fit and the dataset becomes INCOMPLETE for every
criterion; no retry, replacement fit, seed change or resume.  C_Lap non-OK is
recorded and makes only the C_Lap selection INCOMPLETE.

Lineage E (experimental prototype).  MANUSCRIPT_ADOPTION_STATUS:
NOT_DECIDED_BY_HUMAN.

Usage:
    python run_paper_condition_k_criteria.py --preflight
    python run_paper_condition_k_criteria.py --pilot --allow-em
    python run_paper_condition_k_criteria.py --full --allow-em --confirm-b
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTAL = ROOT / "expfam" / "src" / "experimental"
for _path in (str(_EXPERIMENTAL), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import k_criterion_panel as panel                           # noqa: E402
import run_clap_clean_nsweep as a1                          # noqa: E402  (I/O helpers only)

_GEN_PATH = ROOT / "reproduction" / "src" / "data_generator.py"
_spec = importlib.util.spec_from_file_location("paper_data_generator",
                                               _GEN_PATH)
paper_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paper_gen)

# ===========================================================================
# FROZEN PROTOCOL (Issue #128, Human Gate 2026-10-06)
# ===========================================================================

EXPERIMENT_ID = "paper_condition_k_criteria_20261006"
PROTOCOL_VERSION = "paper-condition-k-criteria-protocol-v1"
OUT_ROOT = ROOT / "expfam" / "results" / "k_selection" / EXPERIMENT_ID
STATE_ROOT = Path(r"D:\tento\kennkyu_artifacts") / EXPERIMENT_ID

N = 150
D = 15
K_TRUE_GRID = (1, 3, 5, 7, 9)
CANDIDATE_K = tuple(range(1, 11))
STARTS = (1, 2)
REPLICATES = (1, 2, 3, 4, 5)            # this run (R = 5)
EXTENSION_REPLICATES = (6, 7, 8, 9, 10)  # fixed now, NOT run in this Issue
FAMILY_X = "gaussian"
FAMILY_Y = "bernoulli"
L_SAMPLES = 5
NUM_ITER = 8
NUMERICS_MODE = "consistent"
GENERATOR_KWARGS = {"var_f": 5.0, "uniq": 0.1, "w_scale": 3.0}  # defaults
TIE_TOLERANCE = np.float64(1e-12)
RECOMPOSE_TOL = 1e-6
PILOT_CELLS = ((3, 1),)

CRITERIA = ("C_Q", "C_Lap", "C_cond_MC", "C_cond_last", "C_cond_mode")
CRITERION_ROLE = {"C_Q": "primary", "C_Lap": "primary",
                  "C_cond_MC": "diagnostic", "C_cond_last": "diagnostic",
                  "C_cond_mode": "diagnostic"}


class BStop(RuntimeError):
    """Fail-closed stop.  Never caught to retry."""


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise BStop(message)


def data_seed(k_true: int, rep: int) -> int:
    """rep 1 is the historical reproduction seed 1980 + 100 K_TRUE."""
    return 1980 + 100 * k_true + 1000 * (rep - 1)


def model_seed(k_true: int, rep: int, k_est: int, start: int) -> int:
    return 2_000_000 + 100_000 * k_true + 1000 * rep + 10 * k_est + start


def num_params(k: int) -> int:
    return k * D - k * (k - 1) // 2 + D


def frozen_protocol() -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID, "protocol_version": PROTOCOL_VERSION,
        "issue": 128,
        "is_paper_experiment2_reproduction": False,
        "generator": "reproduction/src/data_generator.py::set_true_params",
        "generator_kwargs": GENERATOR_KWARGS,
        "n": N, "d": D, "k_true_grid": list(K_TRUE_GRID),
        "candidate_k": list(CANDIDATE_K), "starts": list(STARTS),
        "replicates": list(REPLICATES),
        "extension_replicates_fixed_not_run": list(EXTENSION_REPLICATES),
        "data_seed_rule": "1980 + 100*K_TRUE + 1000*(rep-1)",
        "model_seed_rule": "2000000 + 100000*K_TRUE + 1000*rep + 10*K + start",
        "family_x": FAMILY_X, "family_y": FAMILY_Y, "L": L_SAMPLES,
        "num_iter": NUM_ITER, "numerics_mode": NUMERICS_MODE,
        "failure_policy": "fail_fast", "heldout": False,
        "differences_from_historical_reproduction": [
            "estimator DualExpFamLSMConsistent (lineage E), not "
            "LatentStructuralModel",
            "L=5 / 8 iterations / 2 starts (mean), not L=10 / 30 iterations "
            "/ best of 3 trials",
            "criteria evaluated at the final state, not the best iteration",
            "5 datasets per K_TRUE, not 1"],
        "laplace": {"grad_tol": panel.LAPLACE_GRAD_TOL,
                    "max_iter": panel.LAPLACE_MAX_ITER,
                    "Z0": "fit Z_est (last E-step sample)",
                    "N": "n (working convention)", "Y": "all pairs",
                    "ridge_jitter_restart": False},
        "num_params": "K d - K(K-1)/2 + d, for every criterion",
        "criteria": CRITERION_ROLE,
        "c_cond_note": "paper-inspired conditional diagnostics; none is the "
                       "paper's BIC",
        "selector": "mean over the two starts per candidate K; minimum; "
                    "ties within 1e-12 -> smallest K",
        "incomplete_rule": "FAILED fit -> dataset INCOMPLETE for all "
                           "criteria; non-OK C_Lap -> C_Lap (and C_cond_mode) "
                           "INCOMPLETE; no imputation",
        "pilot_gate": "pilot cell K_TRUE=3 rep=1: no FAILED fit, all C_Q / "
                      "C_cond_MC / C_cond_last finite, C_Q closure, all 20 "
                      "C_Lap OK; otherwise stop for a human decision",
        "no_retry_no_replacement_no_seed_rescue_no_resume": True,
        "manuscript_adoption_status": "NOT_DECIDED_BY_HUMAN",
    }


def protocol_hash() -> str:
    return hashlib.sha256(json.dumps(frozen_protocol(), sort_keys=True)
                          .encode()).hexdigest()


def make_dataset(k_true: int, rep: int) -> dict[str, Any]:
    p = paper_gen.set_true_params(n=N, d=D, k=k_true,
                                  seed=data_seed(k_true, rep),
                                  **GENERATOR_KWARGS)
    X, Y = np.asarray(p["X"], float), np.asarray(p["Y"], float)
    _require(np.array_equal(Y, Y.T) and not Y.diagonal().any(), "Y shape")
    upper = np.triu(np.ones((N, N), dtype=bool), 1)
    return {"X": X, "Y": Y, "provenance": {
        "k_true": k_true, "replicate": rep, "data_seed": data_seed(k_true, rep),
        "w0_true": float(p["w0"]), "w_true": float(p["w"]),
        "y_density": float(Y[upper].mean()),
        "x_sha256": hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
        "y_sha256": hashlib.sha256(np.ascontiguousarray(Y).tobytes()).hexdigest()}}


def fit_and_evaluate(ds, k_true, rep, k_est, start):
    import em_runner                                        # noqa: PLC0415
    seed = model_seed(k_true, rep, k_est, start)
    base = {"k_true": k_true, "replicate": rep, "k_est": k_est, "start": start,
            "data_seed": data_seed(k_true, rep), "model_seed": seed,
            "num_params": num_params(k_est)}
    t0 = time.perf_counter()
    try:
        res = em_runner.run_em_experimental(
            X=ds["X"], Y=ds["Y"], family_x=FAMILY_X, family_y=FAMILY_Y,
            k=k_est, L=L_SAMPLES, num_iter=NUM_ITER, seed=seed,
            train_mask=None, numerics_mode=NUMERICS_MODE,
            compute_strict_Q=True, failure_policy="fail_fast")
    except Exception as exc:                                 # recorded, not retried
        return {**base, "fit_status": "FAILED",
                "failure": f"{type(exc).__name__}: {exc}",
                "em_seconds": round(time.perf_counter() - t0, 3)}, None
    em_s = round(time.perf_counter() - t0, 3)
    _require(int(res["num_params"]) == num_params(k_est), "num_params")
    _require(int(res["retry_count"]) == 0 and int(res["replacement_count"]) == 0
             and int(res["seed_rescue_count"]) == 0, "retry/replacement")
    model, Zs = res["model"], np.asarray(res["Z_samples"])
    Z_est = np.asarray(res["Z_est"], float)
    cq = panel.cq_terms(model, ds["X"], ds["Y"], Zs, num_params=num_params(k_est),
                        n=N)
    fit_status = "OK"
    failure = ""
    if not (math.isfinite(cq["C_Q"]) and math.isfinite(float(res["bic"]))):
        fit_status, failure = "FAILED", "non-finite C_Q"
    else:
        _require(cq["q_strict"] == float(res["Q_strict"]),
                 "panel q_strict != runner Q_strict")
        _require(abs(cq["C_Q_recomposed"] - cq["C_Q"]) <= RECOMPOSE_TOL,
                 "C_Q closure")
    lap = panel.laplace_terms(model, ds["X"], ds["Y"], Z_est,
                              num_params=num_params(k_est), N=N)
    z_hat = lap.pop("Z_hat")
    if lap["lap_status"] == "OK":
        _require(abs(lap["C_Lap_recomposed"] - lap["C_Lap"]) <= RECOMPOSE_TOL,
                 "C_Lap closure")
    cond = panel.cond_terms(model, ds["X"], ds["Y"], Z_samples=Zs,
                            Z_last=Z_est, Z_mode=z_hat,
                            mode_ok=lap["lap_status"] == "OK",
                            num_params=num_params(k_est), n=N)
    row = {**base, "fit_status": fit_status, "failure": failure,
           "C_Q": cq["C_Q"], "q_strict": cq["q_strict"],
           "Q_X": cq["Q_X"], "Q_Y": cq["Q_Y"], "Q_Z": cq["Q_Z"],
           "D_K": cq["D_K"], "P_Z": cq["P_Z"], "P_theta": cq["P_theta"],
           "C_Q_recomposed": cq["C_Q_recomposed"], **lap, **cond,
           "w0_hat": float(res["w0"]), "w_hat": float(res["w"]),
           "var_z": float(res["var_z"]),
           "sigma_x_diag_mean": float(np.mean(np.diag(res["sigma"]))),
           "retry_count": 0, "replacement_count": 0, "seed_rescue_count": 0,
           "em_seconds": em_s}
    state = {"Zsamples": Zs, "F": np.asarray(res["F"]),
             "sigma": np.diag(np.asarray(res["sigma"])), "Zhat": z_hat,
             "scalars": np.array([res["w0"], res["w"], res["var_z"]])}
    return row, state


def _select(values, lower=True):
    return a1._select(values, lower)


def analyse(rows: list[dict[str, Any]], cells) -> dict[str, list]:
    sel_rows, curve_rows, dec_rows = [], [], []
    for (k_true, rep) in cells:
        sub = {(r["k_est"], r["start"]): r for r in rows
               if r["k_true"] == k_true and r["replicate"] == rep}
        _require(len(sub) == len(CANDIDATE_K) * len(STARTS),
                 f"fit count {(k_true, rep)}")
        any_failed = any(r["fit_status"] != "OK" for r in sub.values())
        for crit in CRITERIA:
            ok = not any_failed and all(
                math.isfinite(float(sub[(k, s)].get(crit, float("nan"))))
                for k in CANDIDATE_K for s in STARTS)
            if not any_failed:
                for k in CANDIDATE_K:
                    vals = [float(sub[(k, s)][crit]) for s in STARTS]
                    curve_rows.append({
                        "criterion": crit, "role": CRITERION_ROLE[crit],
                        "k_true": k_true, "replicate": rep, "k_est": k,
                        "mean_value": float(np.mean(vals)),
                        "start1_value": vals[0], "start2_value": vals[1]})
            if not ok:
                sel_rows.append({"criterion": crit, "k_true": k_true,
                                 "replicate": rep, "status": "INCOMPLETE",
                                 "selected_k": "", "label": "",
                                 "tie_candidates": "", "margin": "",
                                 "best_value": "", "selected_start1": "",
                                 "selected_start2": ""})
                continue
            means = {k: float(np.mean([sub[(k, s)][crit] for s in STARTS]))
                     for k in CANDIDATE_K}
            chosen, ties, margin, best = _select(means)
            per = {s: _select({k: sub[(k, s)][crit] for k in CANDIDATE_K})[0]
                   for s in STARTS}
            sel_rows.append({"criterion": crit, "k_true": k_true,
                             "replicate": rep, "status": "COMPLETE",
                             "selected_k": chosen,
                             "label": a1.label(chosen, k_true),
                             "tie_candidates": json.dumps(ties),
                             "margin": margin, "best_value": best,
                             "selected_start1": per[1],
                             "selected_start2": per[2]})
        if any_failed:
            continue

        def m(k, col):
            return float(np.mean([sub[(k, s)][col] for s in STARTS]))
        for k in CANDIDATE_K[:-1]:
            lap_ok = all(sub[(kk, s)]["lap_status"] == "OK"
                         for kk in (k, k + 1) for s in STARTS)
            d_fit_q = m(k + 1, "D_K") - m(k, "D_K")
            row = {"k_true": k_true, "replicate": rep, "k_from": k,
                   "k_to": k + 1, "CQ_fit_gain": -d_fit_q,
                   "CQ_dPZ": m(k + 1, "P_Z") - m(k, "P_Z"),
                   "CQ_dPtheta": m(k + 1, "P_theta") - m(k, "P_theta"),
                   "CQ_delta": m(k + 1, "C_Q") - m(k, "C_Q"),
                   "Lap_complete": lap_ok}
            row["CQ_slack"] = row["CQ_fit_gain"] - row["CQ_dPZ"] - row["CQ_dPtheta"]
            if lap_ok:
                row.update({
                    "Lap_fit_gain": -(m(k + 1, "lap_fit_mode")
                                      - m(k, "lap_fit_mode")),
                    "Lap_dvolume": m(k + 1, "lap_volume") - m(k, "lap_volume"),
                    "Lap_dpenalty": m(k + 1, "lap_penalty")
                    - m(k, "lap_penalty"),
                    "Lap_delta": m(k + 1, "C_Lap") - m(k, "C_Lap")})
                row["Lap_slack"] = (row["Lap_fit_gain"] - row["Lap_dvolume"]
                                    - row["Lap_dpenalty"])
            else:
                row.update({c: float("nan") for c in (
                    "Lap_fit_gain", "Lap_dvolume", "Lap_dpenalty",
                    "Lap_delta", "Lap_slack")})
            dec_rows.append(row)
    return {"selection": sel_rows, "curves": curve_rows,
            "decomposition": dec_rows}


def summarise(sel_rows) -> dict[str, Any]:
    out = {}
    for crit in CRITERIA:
        cells = {}
        for k_true in K_TRUE_GRID:
            sub = sorted((r for r in sel_rows if r["criterion"] == crit
                          and r["k_true"] == k_true),
                         key=lambda r: r["replicate"])
            if not sub:
                continue
            comp = [r for r in sub if r["status"] == "COMPLETE"]
            cells[f"K{k_true}"] = {
                "datasets": len(sub), "complete": len(comp),
                "incomplete": len(sub) - len(comp),
                "selected_k": [r["selected_k"] if r["status"] == "COMPLETE"
                               else None for r in sub],
                "exact": sum(r["label"] == "exact" for r in comp),
                "under": sum(r["label"] == "under" for r in comp),
                "over": sum(r["label"] == "over" for r in comp)}
        out[crit] = {"role": CRITERION_ROLE[crit], "by_k_true": cells}
    return out


def run(cells, out_dir: Path, state_dir: Path, stage: str) -> int:
    _require(not out_dir.exists(), f"output directory exists: {out_dir}")
    _require(not state_dir.exists(), f"state directory exists: {state_dir}")
    _require(a1.working_tree_clean(), "working tree must be clean")
    out_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    started = time.time()
    meta = {"experiment_id": EXPERIMENT_ID, "stage": stage,
            "protocol_hash": protocol_hash(), "git_head": a1.git_head(),
            "branch": a1._git("rev-parse", "--abbrev-ref", "HEAD"),
            "working_tree_clean_before_execution": True,
            "environment": a1.environment(), "started_at": started,
            "generator_sha256": a1.sha256_file(_GEN_PATH),
            "cells": [list(c) for c in cells]}
    a1.write_json(out_dir / "protocol.json", {"protocol": frozen_protocol(),
                                              "protocol_hash": protocol_hash()})
    rows, prov, states = [], [], {}
    attempted = 0
    try:
        for (k_true, rep) in cells:
            ds = make_dataset(k_true, rep)
            prov.append(ds["provenance"])
            for k_est in CANDIDATE_K:
                for start in STARTS:
                    attempted += 1
                    row, st = fit_and_evaluate(ds, k_true, rep, k_est, start)
                    rows.append(row)
                    tag = f"K{k_true}_r{rep}_k{k_est}_s{start}"
                    if st is not None:
                        for name, arr in st.items():
                            states[f"{tag}_{name}"] = arr
                    print(f"[{stage}] {tag} fit={row['fit_status']} "
                          f"lap={row.get('lap_status')} em={row['em_seconds']}s "
                          f"lap_t={row.get('lap_seconds')}s", flush=True)
    except Exception as exc:
        a1.write_csv(out_dir / "fit_criteria_partial.csv", rows)
        a1.write_json(out_dir / "failure.json", {
            **meta, "status": "STOPPED", "error_type": type(exc).__name__,
            "error": str(exc), "attempted_fits": attempted,
            "completed_rows": len(rows), "retry_count": 0,
            "replacement_fits_executed": 0, "seed_rescue_count": 0})
        print(f"STOPPED: {type(exc).__name__}: {exc}", flush=True)
        return 2
    # pad failed rows so every CSV row shares one schema
    cols = list(next((r for r in rows if r["fit_status"] == "OK"),
                     rows[0]).keys())
    for c in [c for r in rows for c in r if c not in cols]:
        cols.append(c)
    rows = [{c: r.get(c, "") for c in cols} for r in rows]
    analysis = analyse(rows, cells)
    state_path = state_dir / "states.npz"
    np.savez_compressed(state_path, **states)
    a1.write_csv(out_dir / "fit_criteria.csv", rows)
    a1.write_csv(out_dir / "dataset_provenance.csv", prov)
    a1.write_csv(out_dir / "selection.csv", analysis["selection"])
    a1.write_csv(out_dir / "criterion_curves.csv", analysis["curves"])
    a1.write_csv(out_dir / "increment_decomposition.csv",
                 analysis["decomposition"])
    a1.write_json(out_dir / "summary.json", summarise(analysis["selection"]))
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[str(r.get("lap_status"))] = statuses.get(
            str(r.get("lap_status")), 0) + 1
    completed = time.time()
    a1.write_json(out_dir / "runinfo.json", {
        **meta, "status": "COMPLETED", "completed_at": completed,
        "wall_clock_seconds": round(completed - started, 1),
        "attempted_fits": attempted, "completed_rows": len(rows),
        "failed_fits": sum(r["fit_status"] != "OK" for r in rows),
        "laplace_status_counts": statuses,
        "retry_count": 0, "replacement_fits_executed": 0,
        "seed_rescue_count": 0, "tolerance_relaxations": 0, "resumed": False,
        "states_file": str(state_path),
        "states_sha256": a1.sha256_file(state_path)})
    print(f"[{stage}] completed {len(rows)} fits; laplace {statuses}",
          flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--allow-em", action="store_true")
    ap.add_argument("--confirm-b", action="store_true")
    a = ap.parse_args(argv)
    if a.preflight:
        seeds = [model_seed(t, r, k, s) for t in K_TRUE_GRID
                 for r in REPLICATES + EXTENSION_REPLICATES
                 for k in CANDIDATE_K for s in STARTS]
        dseeds = [data_seed(t, r) for t in K_TRUE_GRID
                  for r in REPLICATES + EXTENSION_REPLICATES]
        print(json.dumps({"protocol_hash": protocol_hash(),
                          "fits": len(K_TRUE_GRID) * len(REPLICATES)
                          * len(CANDIDATE_K) * len(STARTS),
                          "model_seeds_unique": len(seeds) == len(set(seeds)),
                          "data_seeds_unique": len(dseeds) == len(set(dseeds)),
                          "historical_rep1_seeds": {t: data_seed(t, 1)
                                                    for t in K_TRUE_GRID},
                          "working_tree_clean": a1.working_tree_clean(),
                          "git_head": a1.git_head(),
                          "environment": a1.environment()}, indent=2))
        return 0
    _require(a.allow_em, "--allow-em is required")
    if a.pilot:
        return run(list(PILOT_CELLS), OUT_ROOT / "pilot", STATE_ROOT / "pilot",
                   "pilot")
    _require(a.confirm_b, "--confirm-b is required for the full run")
    cells = [(t, r) for t in K_TRUE_GRID for r in REPLICATES]
    return run(cells, OUT_ROOT / "full", STATE_ROOT / "full", "full")


if __name__ == "__main__":
    sys.exit(main())
