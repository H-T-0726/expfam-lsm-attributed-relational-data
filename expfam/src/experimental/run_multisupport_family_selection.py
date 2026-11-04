"""Multi-support family-selection experiment runner (Experiments A / B / C).

Usage
-----
    python run_multisupport_family_selection.py init  --run-dir DIR
    python run_multisupport_family_selection.py run   --run-dir DIR --phase PHASE [--workers N]
    python run_multisupport_family_selection.py status --run-dir DIR

``init`` creates a FRESH run directory (refuses an existing one), runs the
selector-independent difficulty calibration, and freezes ``protocol.json``.
Every later phase verifies the protocol hash before running.

Phases (frozen in protocol.json)
--------------------------------
smoke        A, B, C at n=40, replicate 1
pilot        A, B, C at n=75, replicates 1-3
A_main       A at n in {75, 150}, replicates 1-20
B_main_r5    B at n in {75, 150}, replicates 1-5
C_main_r5    C at n in {75, 150}, replicates 1-5
B_main_r10   B at n in {75, 150}, replicates 6-10
C_main_r10   C at n in {75, 150}, replicates 6-10

Checkpointing: every case writes ``cases/<case>.json`` atomically the moment it
finishes, then its rows are appended to the live tables, then it is recorded
in ``completed_cases.csv`` / ``progress.csv`` / ``checkpoint_state.json``.
A rerun skips completed cases.  A FAILED case is a completed outcome and is
never retried.  No seed depends on any result.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_generator_canonical import GeneratorStop                       # noqa: E402
from data_generator_multisupport import (                                # noqa: E402
    MULTISUPPORT_GENERATOR_VERSION, generate_multisupport_data,
)
from difficulty_calibration import CALIBRATION_VERSION, TAU_GRID, calibrate  # noqa: E402
from em_runner import EMFailFast                                         # noqa: E402
from family_selection import SelectorStop                               # noqa: E402
from family_selection_multisupport import (                              # noqa: E402
    ROUTE_TRUST_EXACT, SELECTOR_VERSION_MS, TRUST_EXACT_GTOL,
    TRUST_EXACT_MAXITER, gate_columns, run_generic_hybrid, selector_only,
)
from family_specs import (                                               # noqa: E402
    FAMILY_SPECS_VERSION, GROUP_BOUNDED, GROUP_COUNT, GROUP_POSITIVE,
    GROUP_REAL, PRIMARY_PRIORITY, PRIORITY_RATIONALE, SupportViolation,
    build_primary_registry, compatibility_rows,
)
from model_dual_expfam_generic import GENERIC_MODEL_VERSION, run_em_generic  # noqa: E402

RUNNER_VERSION = "multisupport-runner-v1"
REGISTRY = build_primary_registry()

GROUP_SHORT = {GROUP_COUNT: "COUNT", GROUP_POSITIVE: "POSITIVE",
               GROUP_BOUNDED: "BOUNDED", "MIXED": "MIXED"}
GROUP_INDEX = {GROUP_COUNT: 0, GROUP_POSITIVE: 1, GROUP_BOUNDED: 2, "MIXED": 3}
DIFF_INDEX = {"EASY": 0, "HARD": 1}
N_INDEX = {40: 0, 75: 1, 150: 2}
STAGE_BLOCK = {"smoke": 1, "pilot": 2, "main": 3}
METHODS = ("ORACLE", "SUPPORT_ONLY", "AUTO_START_FIRST", "AUTO_START_LAST")
COLUMNS_PER_MODEL_SINGLE_GROUP = 3
COLUMNS_PER_MODEL_MIXED = 2
GAUSSIAN_COLUMNS_MIXED = 2
TAU_REAL_MIXED = 1.0

# Exceptions that are numerical / protocol outcomes (recorded as FAILURE and
# never retried).  Anything else is an UNEXPECTED_EXCEPTION, i.e. a technical
# defect that blocks progression.
OUTCOME_EXCEPTIONS = (EMFailFast, SelectorStop, FloatingPointError,
                      GeneratorStop, SupportViolation, np.linalg.LinAlgError,
                      OverflowError)

PHASES = {
    "smoke": {"stage": "smoke", "experiments": ("A", "B", "C"), "n": (40,),
              "replicates": (1,)},
    "pilot": {"stage": "pilot", "experiments": ("A", "B", "C"), "n": (75,),
              "replicates": (1, 2, 3)},
    "A_main": {"stage": "main", "experiments": ("A",), "n": (75, 150),
               "replicates": tuple(range(1, 21))},
    "B_main_r5": {"stage": "main", "experiments": ("B",), "n": (75, 150),
                  "replicates": tuple(range(1, 6))},
    "C_main_r5": {"stage": "main", "experiments": ("C",), "n": (75, 150),
                  "replicates": tuple(range(1, 6))},
    "B_main_r10": {"stage": "main", "experiments": ("B",), "n": (75, 150),
                   "replicates": tuple(range(6, 11))},
    "C_main_r10": {"stage": "main", "experiments": ("C",), "n": (75, 150),
                   "replicates": tuple(range(6, 11))},
}


# ==========================================================================
# small utilities
# ==========================================================================

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    raise TypeError(f"not JSON serialisable: {type(v)}")


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False,
                                       default=_json_default, allow_nan=True))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def append_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Append rows; the header is fixed by the first write and never changes."""
    if not rows:
        return
    new = not path.exists()
    if new:
        fields = list(rows[0].keys())
        for r in rows[1:]:
            for k in r:
                if k not in fields:
                    fields.append(k)
    else:
        with open(path, newline="", encoding="utf-8") as fh:
            fields = next(csv.reader(fh))
    extra = sorted({k for r in rows for k in r} - set(fields))
    if extra:
        raise RuntimeError(f"{path.name}: rows carry columns not in the fixed "
                           f"header: {extra}")
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
        fh.flush()
        os.fsync(fh.fileno())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=_HERE,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:                                     # pragma: no cover
        return ""


def procrustes_rmse_z(Z_est: np.ndarray, Z_true: np.ndarray) -> float:
    """Same alignment as run_family_selection_pilot.procrustes_rmse_z (Phase 9C)."""
    Z_est = np.asarray(Z_est, dtype=np.float64)
    Z_true = np.asarray(Z_true, dtype=np.float64)
    width = min(Z_est.shape[1], Z_true.shape[1])
    u, _, vt = np.linalg.svd(Z_est[:, :width].T @ Z_true[:, :width])
    aligned = Z_est[:, :width] @ (u @ vt)
    return float(np.sqrt(np.mean((aligned - Z_true[:, :width]) ** 2)))


# ==========================================================================
# protocol
# ==========================================================================

def column_design(group: str) -> list[str]:
    if group == "MIXED":
        cols = []
        for g in (GROUP_COUNT, GROUP_POSITIVE, GROUP_BOUNDED):
            for cm in PRIMARY_PRIORITY[g]:
                cols += [cm] * COLUMNS_PER_MODEL_MIXED
        cols += ["gaussian"] * GAUSSIAN_COLUMNS_MIXED
        return cols
    cols = []
    for cm in PRIMARY_PRIORITY[group]:
        cols += [cm] * COLUMNS_PER_MODEL_SINGLE_GROUP
    return cols


def column_tau(group: str, difficulty: str, tau_sel: dict) -> list[float]:
    out = []
    for cm in column_design(group):
        g = REGISTRY[cm].support_group
        out.append(TAU_REAL_MIXED if g == GROUP_REAL else float(tau_sel[g][difficulty]))
    return out


def build_protocol(calibration: dict) -> dict[str, Any]:
    tau_sel = calibration["selection"]
    designs = {}
    for g in (GROUP_COUNT, GROUP_POSITIVE, GROUP_BOUNDED, "MIXED"):
        designs[GROUP_SHORT[g]] = {
            "support_group": g, "columns": column_design(g),
            "d": len(column_design(g)),
            "tau_by_difficulty": {diff: column_tau(g, diff, tau_sel)
                                  for diff in ("EASY", "HARD")}}
    return {
        "protocol_name": "multisupport_family_selection_20261006",
        "task_type": "CHARACTERIZATION (controlled synthetic family-selection "
                     "experiment; lineage E prototype, not adoptable for the "
                     "manuscript)",
        "research_question": "Within one support / base measure, can the "
                             "continuous-Z candidate score recover the true "
                             "per-column family among several valid candidates?",
        "model": {"z": "N(0, I_K)", "eta_x": "f_l^T z_i (no intercept)",
                  "x": "canonical ExpFam_m(eta), dispersion 1 (Gaussian: "
                       "dispersion form, sigma^2 estimated)",
                  "y": "Bernoulli(sigmoid(w0 + w z_i^T z_j)), i<j"},
        "K_true": 3, "K_fit": 3, "K_search": False,
        "family_y": "bernoulli", "w0": -1.0, "w": 1.0,
        "L": 5, "exploration_num_iter": 8, "refit_num_iter": 8,
        "newton_alpha": 0.5,
        "scheme": "Phase 9C hybrid (Scheme C): A-type selection inside an "
                  "exploration MCEM, then a fresh fixed-assignment refit that "
                  "is reported",
        "candidate_route": ROUTE_TRUST_EXACT,
        "candidate_optimizer": f"scipy trust-exact (Newton trust region, exact "
                               f"analytic Hessian), maxiter {TRUST_EXACT_MAXITER}, "
                               f"gtol {TRUST_EXACT_GTOL}, for every non-Gaussian "
                               f"candidate; a NotEvaluable trial point is a "
                               f"rejected step; start = zero loading (A) or "
                               f"incumbent loading (MCEM exploration)",
        "candidate_convergence_rule": "grad_inf <= 1e-8 recorded as a "
                                      "DIAGNOSTIC (WARNING), not a blocker",
        "penalty": "none (equal free-parameter counts within every group; "
                   "raw complete score; margin recorded)",
        "registry": {cm: s.registry_row() for cm, s in REGISTRY.items()},
        "priority": {g: list(v) for g, v in PRIMARY_PRIORITY.items()},
        "priority_rationale": PRIORITY_RATIONALE,
        "support_only_rule": "first support-admissible candidate in the frozen "
                             "priority of the column's Level-0 support group",
        "starts": {"START_FIRST": "first admissible candidate in priority",
                   "START_LAST": "last admissible candidate in priority"},
        "methods": list(METHODS),
        "seed_rule": {
            "data_seed": "STAGE_BLOCK*10_000_000 + EXP_BLOCK*1_000_000 + "
                         "GROUP_INDEX*100_000 + DIFF_INDEX*10_000 + "
                         "N_INDEX*1_000 + replicate",
            "STAGE_BLOCK": STAGE_BLOCK, "EXP_BLOCK": {"AB": 0, "C": 1},
            "GROUP_INDEX": {GROUP_SHORT[k]: v for k, v in GROUP_INDEX.items()},
            "DIFF_INDEX": DIFF_INDEX, "N_INDEX": {str(k): v for k, v in N_INDEX.items()},
            "search_seed": "data_seed*10 + 1 (exploration, shared by both starts)",
            "refit_seed": "data_seed*10 + 2 (Auto refit, ORACLE and SUPPORT_ONLY)",
            "selector_only": "deterministic, no RNG",
            "A_and_B_share_datasets": True},
        "designs": designs,
        "information_matched_generator": "row l of F scaled by "
                                         "1/sqrt(A''_{m_l}(0)); E_l||f_l||^2 "
                                         "A''(0) = tau",
        "tau_real_mixed": TAU_REAL_MIXED,
        "difficulty": {"tau_grid": list(calibration["tau_grid"]),
                       "rule": calibration["rule"],
                       "selection": tau_sel,
                       "calibration_version": calibration["version"],
                       "gh_nodes": calibration["gh_nodes"],
                       "note": "computed from distributions only, before any "
                               "selector run; frozen"},
        "phases": {k: {**v, "experiments": list(v["experiments"]),
                       "n": list(v["n"]), "replicates": list(v["replicates"])}
                   for k, v in PHASES.items()},
        "extension_rule_r10": "run *_main_r10 only if the matching *_main_r5 "
                              "phase has 0 UNEXPECTED_EXCEPTION cases and a "
                              "numerical FAILURE fraction <= 10% of its fits",
        "mixed_inclusion_rule": "every candidate model of B enters C unless B "
                                "produced an UNEXPECTED_EXCEPTION attributable "
                                "to it",
        "metrics": {"primary": ["family recovery rate", "candidate-model "
                                "recovery rate", "confusion matrix",
                                "RMSE(Z) after Procrustes (Phase 9C method)",
                                "Auto - Oracle RMSE gap",
                                "Auto - SupportOnly RMSE gap"],
                    "secondary": ["score margin", "start agreement",
                                  "candidate optimiser convergence",
                                  "failure rate", "runtime"],
                    "recovery_scopes": ["all columns (gate + score)",
                                        "score-decided columns only"]},
        "failure_policy": "fail-fast; failures are completed outcomes; no "
                          "retry, no reseed, no replacement, no clipping",
        "claim_boundary": "controlled synthetic experiment with fixed nuisance "
                          "hyperparameters; not general automatic distribution "
                          "selection; not real data; K fixed",
    }


def data_seed(stage: str, exp_block: str, group: str, diff: str, n: int, rep: int) -> int:
    return (STAGE_BLOCK[stage] * 10_000_000 + {"AB": 0, "C": 1}[exp_block] * 1_000_000
            + GROUP_INDEX[group] * 100_000 + DIFF_INDEX[diff] * 10_000
            + N_INDEX[n] * 1_000 + rep)


def dataset_key(stage, exp_block, group, diff, n, rep) -> str:
    return f"{exp_block}_{stage}_{GROUP_SHORT[group]}_{diff}_n{n}_rep{rep:03d}"


# ==========================================================================
# datasets
# ==========================================================================

def get_dataset(run_dir: Path, protocol: dict, stage, exp_block, group, diff, n, rep):
    key = dataset_key(stage, exp_block, group, diff, n, rep)
    path = run_dir / "data" / f"dataset_{key}.npz"
    seed = data_seed(stage, exp_block, group, diff, n, rep)
    design = protocol["designs"][GROUP_SHORT[group]]
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata_json"]))
            if int(meta["seed"]) != seed:
                raise RuntimeError(f"{path.name}: seed mismatch")
            return key, path, {"X": z["X"].copy(), "Y": z["Y"].copy(),
                               "Z": z["Z_true"].copy(), "F": z["F_true"].copy(),
                               "meta": meta}, False
    specs = [REGISTRY[c] for c in design["columns"]]
    ds = generate_multisupport_data(n=n, k=protocol["K_true"], seed=seed,
                                    column_specs=specs,
                                    tau=design["tau_by_difficulty"][diff],
                                    w0=protocol["w0"], w=protocol["w"])
    meta = dict(ds.metadata)
    meta.update({"dataset_key": key, "stage": stage, "exp_block": exp_block,
                 "group": GROUP_SHORT[group], "difficulty": diff, "replicate": rep})
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, X=ds.X, Y=ds.Y, Z_true=ds.Z, F_true=ds.F,
                        true_family=np.array(meta["true_family_names"]),
                        true_candidate_model=np.array(meta["true_candidate_models"]),
                        metadata_json=np.array(json.dumps(meta, default=_json_default)))
    os.replace(tmp, path)
    atomic_write_json(path.with_suffix(".json"), meta)
    return key, path, {"X": ds.X, "Y": ds.Y, "Z": ds.Z, "F": ds.F, "meta": meta}, True


# ==========================================================================
# case execution (runs in a worker process)
# ==========================================================================

def _case_id(exp, stage, group, diff, n, rep, method=None) -> str:
    base = f"{exp}__{stage}__{GROUP_SHORT[group]}__{diff}__n{n}__rep{rep:03d}"
    return base + (f"__{method}" if method else "")


def enumerate_cases(phase: str) -> list[dict[str, Any]]:
    spec = PHASES[phase]
    stage = spec["stage"]
    cases = []
    for exp in spec["experiments"]:
        groups = ["MIXED"] if exp == "C" else [GROUP_COUNT, GROUP_POSITIVE, GROUP_BOUNDED]
        for group in groups:
            for diff in ("EASY", "HARD"):
                for n in spec["n"]:
                    for rep in spec["replicates"]:
                        common = dict(experiment=exp, stage=stage, group=group,
                                      difficulty=diff, n=n, replicate=rep,
                                      phase=phase,
                                      exp_block="C" if exp == "C" else "AB")
                        if exp == "A":
                            cases.append({**common, "method": "SELECTOR_ONLY",
                                          "case_id": _case_id(exp, stage, group, diff, n, rep)})
                        else:
                            for m in METHODS:
                                cases.append({**common, "method": m,
                                              "case_id": _case_id(exp, stage, group, diff, n, rep, m)})
    return cases


def _cond(case) -> str:
    return f"{GROUP_SHORT[case['group']]}|{case['difficulty']}|n{case['n']}"


def _base_row(case, meta) -> dict[str, Any]:
    return {"experiment": case["experiment"], "stage": case["stage"],
            "phase": case["phase"], "condition": _cond(case),
            "group_design": GROUP_SHORT[case["group"]],
            "difficulty": case["difficulty"], "n": case["n"],
            "replicate": case["replicate"], "method": case["method"],
            "case_id": case["case_id"], "data_seed": meta["seed"]}


def _col_truth(meta, col):
    return {"true_family": meta["true_family_names"][col],
            "true_candidate_model": meta["true_candidate_models"][col],
            "true_support_group": meta["true_support_groups"][col]}


def run_case(run_dir_str: str, protocol: dict, case: dict) -> dict[str, Any]:
    run_dir = Path(run_dir_str)
    t0 = time.perf_counter()
    out: dict[str, Any] = {"case": case, "status": "SUCCESS", "error_class": "",
                           "failure_reason": "", "rows": {}, "started_utc": _now()}
    rows = out["rows"]
    try:
        key, path, ds, created = get_dataset(run_dir, protocol, case["stage"],
                                             case["exp_block"], case["group"],
                                             case["difficulty"], case["n"],
                                             case["replicate"])
    except Exception as exc:                               # generator stop or bug
        out["status"] = "FAILURE"
        out["error_class"] = ("OUTCOME" if isinstance(exc, OUTCOME_EXCEPTIONS)
                              else "UNEXPECTED_EXCEPTION")
        out["failure_reason"] = f"dataset: {type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
        out["runtime_seconds"] = round(time.perf_counter() - t0, 3)
        return out
    meta = ds["meta"]
    out["dataset_key"] = key
    out["dataset_path"] = str(path.relative_to(run_dir))
    out["dataset_created"] = created
    out["dataset_sha256"] = sha256_file(path)
    seed = int(meta["seed"])
    search_seed, refit_seed = seed * 10 + 1, seed * 10 + 2
    out["seeds"] = {"data_seed": seed, "search_seed": search_seed,
                    "refit_seed": refit_seed}
    base = _base_row(case, meta)
    X, Y, Z_true = ds["X"], ds["Y"], ds["Z"]
    d = X.shape[1]
    try:
        if case["experiment"] == "A":
            gates, sels = selector_only(X, Z_true, REGISTRY, PRIMARY_PRIORITY,
                                        route=protocol["candidate_route"])
            _rows_selector(rows, base, meta, gates, sels, start_assignment="",
                           final_assignment="")
        else:
            gates = gate_columns(X, REGISTRY, PRIMARY_PRIORITY)
            out["gates"] = [g.as_row() for g in gates]
            method = case["method"]
            fit_info = {"fit_seed": refit_seed}
            if method == "ORACLE":
                assign = list(meta["true_candidate_models"])
                start = assign
                res = run_em_generic(X, Y, column_specs=[REGISTRY[c] for c in assign],
                                     family_y="bernoulli", k=protocol["K_fit"],
                                     L=protocol["L"], num_iter=protocol["refit_num_iter"],
                                     seed=refit_seed, newton_alpha=protocol["newton_alpha"])
            elif method == "SUPPORT_ONLY":
                assign = [g.admissible[0] for g in gates]
                start = assign
                res = run_em_generic(X, Y, column_specs=[REGISTRY[c] for c in assign],
                                     family_y="bernoulli", k=protocol["K_fit"],
                                     L=protocol["L"], num_iter=protocol["refit_num_iter"],
                                     seed=refit_seed, newton_alpha=protocol["newton_alpha"])
            else:
                start_label = method.replace("AUTO_", "")
                hyb = run_generic_hybrid(
                    X, Y, k=protocol["K_fit"], registry=REGISTRY,
                    priority=PRIMARY_PRIORITY, start=start_label,
                    L=protocol["L"], exploration_num_iter=protocol["exploration_num_iter"],
                    refit_num_iter=protocol["refit_num_iter"],
                    search_seed=search_seed, refit_seed=refit_seed,
                    newton_alpha=protocol["newton_alpha"],
                    route=protocol["candidate_route"])
                res = hyb["refit"]
                start = hyb["start_models"]
                assign = hyb["exploration"]["selected_models"]
                fit_info["search_seed"] = search_seed
                fit_info["exploration_estep_backtracks"] = hyb["exploration"]["estep_backtracks"]
                _rows_auto(rows, base, meta, gates, hyb)
            rmse = procrustes_rmse_z(res["Z_est"], Z_true)
            rows["fit_results"] = [{
                **base, "d": d, "K": protocol["K_fit"], "final_Q": res["Q_strict"],
                "criterion_field_bic_ICL_type": res["bic"],
                "num_params": res["num_params"], "rmse_Z": rmse,
                "converged": "fixed_iterations", "iterations": protocol["refit_num_iter"],
                "start_assignment": "|".join(start), "final_assignment": "|".join(assign),
                "true_assignment": "|".join(meta["true_candidate_models"]),
                "n_correct_candidate": sum(a == t for a, t in zip(assign, meta["true_candidate_models"])),
                "n_correct_family": sum(REGISTRY[a].family_name == t for a, t in
                                        zip(assign, meta["true_family_names"])),
                "fit_seed": refit_seed, "search_seed": fit_info.get("search_seed", ""),
                "refit_estep_backtracks": res.get("estep_backtracks", 0),
                "exploration_estep_backtracks": fit_info.get("exploration_estep_backtracks", ""),
                "w0_est": res["w0"], "w_est": res["w"],
                "runtime_seconds": None, "status": "SUCCESS", "failure_reason": "",
                "retry_count": 0, "replacement_count": 0, "seed_rescue_count": 0}]
            zpath = run_dir / "data" / "fits" / f"{case['case_id']}.npz"
            tmp = zpath.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, Z_est=res["Z_est"], F_est=res["F"],
                                sigma_est=np.diag(res["sigma"]))
            os.replace(tmp, zpath)
            out["fit_artifact"] = str(zpath.relative_to(run_dir))
    except Exception as exc:
        out["status"] = "FAILURE"
        out["error_class"] = ("OUTCOME" if isinstance(exc, OUTCOME_EXCEPTIONS)
                              else "UNEXPECTED_EXCEPTION")
        out["failure_reason"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
        if case["experiment"] == "A" or case["method"].startswith("AUTO_"):
            # every column still counts in the recovery denominator
            rows["selection_results"] = [{
                **base, "column": col, **_col_truth(meta, col),
                "selection_status": "FAILURE", "correct_family": False,
                "correct_candidate_model": False,
                "failure_reason": out["failure_reason"]} for col in range(d)]
            rows.pop("candidate_scores", None)
            rows.pop("selection_trace", None)
            rows.pop("optimization_diagnostics", None)
        if case["experiment"] != "A":
            rows["fit_results"] = [{**base, "d": d, "K": protocol["K_fit"],
                                    "final_Q": "", "rmse_Z": "", "status": "FAILURE",
                                    "failure_reason": out["failure_reason"],
                                    "fit_seed": refit_seed, "retry_count": 0,
                                    "replacement_count": 0, "seed_rescue_count": 0}]
    rt = round(time.perf_counter() - t0, 3)
    out["runtime_seconds"] = rt
    for r in rows.get("fit_results", []):
        r["runtime_seconds"] = rt
    out["finished_utc"] = _now()
    return out


def _rows_selector(rows, base, meta, gates, sels, start_assignment, final_assignment):
    by_col = {s.column: s for s in sels}
    cand_rows, sel_rows, diag_rows = [], [], []
    for g in gates:
        truth = _col_truth(meta, g.column)
        sel = by_col.get(g.column)
        for r in (sel.records if sel else []):
            cand_rows.append({**base, "column": g.column, "support_group": g.support_group,
                              **truth, **r.as_row(),
                              "optimized_loading": "|".join(f"{v:.17g}" for v in r.loading)
                              if r.loading is not None else "",
                              "iteration": "final"})
            diag_rows.append({**base, "column": g.column, "iteration": "final",
                              "candidate_model": r.candidate_model,
                              "status": r.status, "optimizer_route": r.optimizer_route,
                              "scipy_success": r.scipy_success, "scipy_status": r.scipy_status,
                              "n_iter": r.n_iter, "gradient_inf": r.gradient_inf,
                              "converged_1e-8": r.converged})
        selected = sel.selected if sel and sel.status == "SELECTED" else ""
        sel_rows.append({**base, "column": g.column, "support_group": g.support_group,
                         **truth,
                         "group_gate_correct": g.support_group == truth["true_support_group"],
                         "admissible": "|".join(g.admissible),
                         "true_admissible": truth["true_candidate_model"] in g.admissible,
                         "selected_family": REGISTRY[selected].family_name if selected else "",
                         "selected_candidate_model": selected,
                         "selection_status": sel.status if sel else "NO_CANDIDATES",
                         "correct_family": bool(selected) and REGISTRY[selected].family_name == truth["true_family"],
                         "correct_candidate_model": bool(selected) and selected == truth["true_candidate_model"],
                         "winner_score": sel.winner_score if sel else "",
                         "runnerup_score": sel.runnerup_score if sel else "",
                         "runner_up": sel.runner_up if sel else "",
                         "score_margin": sel.margin_neg2 if sel else "",
                         "decided_by": g.decided_by,
                         "start_assignment": start_assignment,
                         "final_assignment": final_assignment,
                         "failure_reason": sel.failure_reason if sel else ""})
    rows["candidate_scores"] = cand_rows
    rows["selection_results"] = sel_rows
    rows["optimization_diagnostics"] = diag_rows


def _rows_auto(rows, base, meta, gates, hyb):
    exp = hyb["exploration"]
    start, final = hyb["start_models"], exp["selected_models"]
    last = {s.column: s for s in exp["final_selections"]}
    by_gate = {g.column: g for g in gates}
    cand_rows, sel_rows, diag_rows, trace_rows = [], [], [], []
    for g in gates:
        truth = _col_truth(meta, g.column)
        sel = last.get(g.column)
        for r in (sel.records if sel else []):
            cand_rows.append({**base, "column": g.column, "support_group": g.support_group,
                              **truth, **r.as_row(),
                              "optimized_loading": "|".join(f"{v:.17g}" for v in r.loading)
                              if r.loading is not None else "",
                              "iteration": "last_exploration"})
        chosen = final[g.column]
        sel_rows.append({**base, "column": g.column, "support_group": g.support_group,
                         **truth,
                         "group_gate_correct": g.support_group == truth["true_support_group"],
                         "admissible": "|".join(g.admissible),
                         "true_admissible": truth["true_candidate_model"] in g.admissible,
                         "selected_family": REGISTRY[chosen].family_name,
                         "selected_candidate_model": chosen,
                         "selection_status": "SELECTED",
                         "correct_family": REGISTRY[chosen].family_name == truth["true_family"],
                         "correct_candidate_model": chosen == truth["true_candidate_model"],
                         "winner_score": sel.winner_score if sel else "",
                         "runnerup_score": sel.runnerup_score if sel else "",
                         "runner_up": sel.runner_up if sel else "",
                         "score_margin": sel.margin_neg2 if sel else "",
                         "decided_by": g.decided_by,
                         "start_candidate_model": start[g.column],
                         "start_assignment": "|".join(start),
                         "final_assignment": "|".join(final), "failure_reason": ""})
    for t in exp["selection_trace"]:
        tr = {**base, **{k: v for k, v in t.items()}}
        tr.update(_col_truth(meta, t["column"]))
        trace_rows.append(tr)
        for cm, gi in json.loads(t.get("candidate_grad_inf_json", "{}")).items():
            diag_rows.append({**base, "column": t["column"], "iteration": t["iteration"],
                              "candidate_model": cm, "status": "OK",
                              "optimizer_route": "", "scipy_success": "", "scipy_status": "",
                              "n_iter": "", "gradient_inf": gi,
                              "converged_1e-8": bool(gi <= 1e-8)})
    rows["candidate_scores"] = cand_rows
    rows["selection_results"] = sel_rows
    rows["selection_trace"] = trace_rows
    rows["optimization_diagnostics"] = diag_rows


# ==========================================================================
# bookkeeping in the main process
# ==========================================================================

TABLES = ("candidate_scores", "selection_results", "fit_results",
          "selection_trace", "optimization_diagnostics")

BASE_FIELDS = ["experiment", "stage", "phase", "condition", "group_design",
               "difficulty", "n", "replicate", "method", "case_id", "data_seed"]
TRUTH_FIELDS = ["true_family", "true_candidate_model", "true_support_group"]
SCHEMAS = {
    "candidate_scores": BASE_FIELDS + [
        "iteration", "column", "support_group"] + TRUTH_FIELDS + [
        "candidate_family", "candidate_model", "candidate_status",
        "candidate_score", "normalized_score_per_obs", "optimizer_success",
        "optimizer_status", "optimizer_iterations", "optimizer_route",
        "gradient_inf_norm", "gradient_converged_1e-8", "loading_norm",
        "sigma_sq", "optimized_loading", "failure_reason"],
    "selection_results": BASE_FIELDS + [
        "column", "support_group"] + TRUTH_FIELDS + [
        "group_gate_correct", "admissible", "true_admissible",
        "selected_family", "selected_candidate_model", "selection_status",
        "correct_family", "correct_candidate_model", "winner_score",
        "runnerup_score", "runner_up", "score_margin", "decided_by",
        "start_candidate_model", "start_assignment", "final_assignment",
        "failure_reason"],
    "selection_trace": BASE_FIELDS + [
        "iteration", "column", "support_group"] + TRUTH_FIELDS + [
        "admissible", "previous_candidate", "status", "selected_candidate",
        "selected_family", "runner_up", "changed", "margin_neg2",
        "winner_score", "runnerup_score", "all_candidates_converged",
        "max_candidate_grad_inf", "candidate_scores_json",
        "candidate_grad_inf_json", "selected_loading_sha256",
        "selected_loading_installed", "installed_loading_sha256",
        "failure_reason"],
    "optimization_diagnostics": BASE_FIELDS + [
        "column", "iteration", "candidate_model", "status", "optimizer_route",
        "scipy_success", "scipy_status", "n_iter", "gradient_inf",
        "converged_1e-8"],
}

FIT_FIELDS = ["experiment", "stage", "phase", "condition", "group_design",
              "difficulty", "n", "replicate", "method", "case_id", "data_seed",
              "d", "K", "final_Q", "criterion_field_bic_ICL_type", "num_params",
              "rmse_Z", "converged", "iterations", "start_assignment",
              "final_assignment", "true_assignment", "n_correct_candidate",
              "n_correct_family", "fit_seed", "search_seed",
              "refit_estep_backtracks", "exploration_estep_backtracks",
              "w0_est", "w_est",
              "runtime_seconds", "status", "failure_reason", "retry_count",
              "replacement_count", "seed_rescue_count"]


def case_file(run_dir: Path, case_id: str) -> Path:
    return run_dir / "cases" / f"{case_id}.json"


def completed_ids(run_dir: Path) -> set[str]:
    return {r["case_id"] for r in read_csv(run_dir / "completed_cases.csv")}


def _remove_case_rows(run_dir: Path, case_id: str) -> None:
    """Drop rows of one case from the live tables (partial-append repair)."""
    for name in TABLES + ("failures", "dataset_manifest"):
        path = run_dir / "tables" / f"{name}.csv"
        rows = read_csv(path)
        if not rows:
            continue
        keep = [r for r in rows if r.get("case_id") != case_id]
        if len(keep) != len(rows):
            fields = list(rows[0].keys())
            import io
            sio = io.StringIO()
            w = csv.DictWriter(sio, fieldnames=fields)
            w.writeheader()
            for r in keep:
                w.writerow(r)
            atomic_write_text(path, sio.getvalue())


def record_case(run_dir: Path, result: dict[str, Any]) -> None:
    case = result["case"]
    cid = case["case_id"]
    cpath = case_file(run_dir, cid)
    atomic_write_json(cpath, result)
    tables = run_dir / "tables"
    rows = result["rows"]
    schemas = dict(SCHEMAS, fit_results=FIT_FIELDS)
    for name in TABLES:
        projected = []
        for r in rows.get(name, []):
            extra = set(r) - set(schemas[name])
            if extra:
                raise RuntimeError(f"{cid}: {name} row has unknown fields {sorted(extra)}")
            projected.append({k: r.get(k, "") for k in schemas[name]})
        append_rows(tables / f"{name}.csv", projected)
    if result.get("dataset_created"):
        meta = json.loads(Path(run_dir / result["dataset_path"]).with_suffix(".json").read_text())
        append_rows(tables / "dataset_manifest.csv", [{
            "case_id": cid, "experiment": "AB" if case["exp_block"] == "AB" else "C",
            "stage": case["stage"], "condition": _cond(case),
            "replicate": case["replicate"], "n": meta["n"], "d": meta["d"],
            "K_true": meta["K_true"], "data_seed": meta["seed"],
            "true_family_assignment": "|".join(meta["true_family_names"]),
            "true_candidate_model_assignment": "|".join(meta["true_candidate_models"]),
            "tau": "|".join(str(t) for t in meta["tau"]),
            "y_density": meta["y_density"],
            "dataset_path": result["dataset_path"],
            "dataset_sha256": result["dataset_sha256"],
            "generator_version": meta["generator_version"]}])
    if result["status"] != "SUCCESS":
        append_rows(tables / "failures.csv", [{
            "case_id": cid, "experiment": case["experiment"], "stage": case["stage"],
            "phase": case["phase"], "condition": _cond(case),
            "replicate": case["replicate"], "method": case["method"],
            "error_class": result["error_class"],
            "failure_reason": result["failure_reason"],
            "retried": False, "replaced": False}])
        if result.get("traceback"):
            with open(run_dir / "logs" / "failure_tracebacks.log", "a", encoding="utf-8") as fh:
                fh.write(f"===== {cid} =====\n{result['traceback']}\n")
    # sub-selection failures inside a SUCCESS selector-only case
    for r in (rows.get("selection_results", []) if result["status"] == "SUCCESS" else []):
        if r.get("selection_status") == "FAILURE":
            append_rows(tables / "failures.csv", [{
                "case_id": cid, "experiment": case["experiment"], "stage": case["stage"],
                "phase": case["phase"], "condition": _cond(case),
                "replicate": case["replicate"], "method": case["method"],
                "error_class": "OUTCOME_COLUMN_SELECTION",
                "failure_reason": f"column {r['column']}: {r['failure_reason']}",
                "retried": False, "replaced": False}])
    sha = sha256_file(cpath)
    append_rows(run_dir / "progress.csv", [{
        "timestamp_utc": _now(), "case_id": cid, "phase": case["phase"],
        "status": result["status"], "error_class": result["error_class"],
        "runtime_seconds": result.get("runtime_seconds", "")}])
    append_rows(run_dir / "completed_cases.csv", [{
        "case_id": cid, "phase": case["phase"], "status": result["status"],
        "case_file": str(cpath.relative_to(run_dir)), "case_file_sha256": sha}])


def update_checkpoint(run_dir: Path, phase: str, total: int, note: str = "") -> dict:
    done = read_csv(run_dir / "completed_cases.csv")
    by_phase: dict[str, dict[str, int]] = {}
    for r in done:
        p = by_phase.setdefault(r["phase"], {"SUCCESS": 0, "FAILURE": 0})
        p[r["status"]] = p.get(r["status"], 0) + 1
    state = {"updated_utc": _now(), "current_phase": phase,
             "current_phase_expected_cases": total,
             "completed_by_phase": by_phase, "note": note,
             "protocol_sha256": sha256_file(run_dir / "protocol.json")}
    atomic_write_json(run_dir / "checkpoint_state.json", state)
    return state


# ==========================================================================
# commands
# ==========================================================================

def family_test_status(junit: Path | None) -> dict[str, str]:
    """PASS / FAIL / NOT_RUN per family from a pytest junit XML report."""
    if junit is None or not junit.exists():
        return {}
    import xml.etree.ElementTree as ET
    cases = []
    for tc in ET.parse(junit).getroot().iter("testcase"):
        failed = any(child.tag in ("failure", "error") for child in tc)
        skipped = any(child.tag == "skipped" for child in tc)
        cases.append((f"{tc.get('classname')}::{tc.get('name')}", failed, skipped))
    status = {}
    for fam in {s.family_name for s in REGISTRY.values()} | {"bernoulli"}:
        cms = [cm for cm, s in REGISTRY.items() if s.family_name == fam] or [fam]
        keys = cms + ([fam] if fam in ("gaussian", "bernoulli", "poisson") else [])
        rel = [c for c in cases
               if any(k in c[0] for k in keys)
               or (fam in ("gaussian", "bernoulli", "poisson")
                   and "test_generic_regression" in c[0])]
        if not rel:
            status[fam] = "NOT_RUN"
        elif any(f for _, f, _ in rel):
            status[fam] = "FAIL"
        else:
            status[fam] = f"PASS({sum(1 for _, f, sk in rel if not sk)} tests)"
    return status


def cmd_init(run_dir: Path, junit: Path | None = None) -> None:
    if run_dir.exists():
        raise SystemExit(f"{run_dir} already exists; refusing to overwrite a run")
    for sub in ("data", "data/fits", "tables", "figures", "logs", "audit", "cases"):
        (run_dir / sub).mkdir(parents=True, exist_ok=False)
    started = _now()
    calibration = calibrate(REGISTRY, PRIMARY_PRIORITY)
    protocol = build_protocol(calibration)
    tables = run_dir / "tables"
    cal_rows = []
    for r in calibration["pair_rows"]:
        cal_rows.append({"row_type": "pair", **r})
    for r in calibration["true_rows"]:
        cal_rows.append({"row_type": "true_candidate_min", **r})
    for r in calibration["group_rows"]:
        sel = calibration["selection"][r["support_group"]]
        label = ("EASY" if r["tau"] == sel["EASY"] else
                 "HARD" if r["tau"] == sel["HARD"] else "")
        cal_rows.append({"row_type": "group_difficulty", **r, "frozen_label": label})
    for r in calibration["moment_rows"]:
        cal_rows.append({"row_type": "moments_at_eta0", **r})
    append_rows(tables / "difficulty_calibration.csv", cal_rows)
    append_rows(tables / "candidate_registry.csv", [
        {**s.registry_row(),
         "candidate_priority": (list(PRIMARY_PRIORITY[s.support_group]).index(cm) + 1
                                if cm in PRIMARY_PRIORITY[s.support_group] else ""),
         "in_primary_candidate_set": cm in PRIMARY_PRIORITY[s.support_group]}
        for cm, s in REGISTRY.items()])
    append_rows(tables / "family_compatibility.csv",
                compatibility_rows(family_test_status(junit)))
    if junit is not None and junit.exists():
        import shutil
        shutil.copy2(junit, run_dir / "audit" / "pytest_junit_init.xml")
    atomic_write_json(run_dir / "protocol.json", protocol)
    psha = sha256_file(run_dir / "protocol.json")
    seeds = {}
    for phase in PHASES:
        for c in enumerate_cases(phase):
            s = data_seed(c["stage"], c["exp_block"], c["group"], c["difficulty"],
                          c["n"], c["replicate"])
            key = dataset_key(c["stage"], c["exp_block"], c["group"], c["difficulty"],
                              c["n"], c["replicate"])
            seeds[key] = {"data_seed": s, "search_seed": s * 10 + 1,
                          "refit_seed": s * 10 + 2}
    atomic_write_json(run_dir / "seed_manifest.json",
                      {"protocol_sha256": psha, "rule": protocol["seed_rule"],
                       "datasets": seeds,
                       "n_unique_data_seeds": len({v["data_seed"] for v in seeds.values()})})
    try:
        import scipy, matplotlib  # noqa: F401
        dep = {"numpy": np.__version__, "scipy": scipy.__version__,
               "matplotlib": matplotlib.__version__}
    except Exception:                                     # pragma: no cover
        dep = {"numpy": np.__version__}
    req = Path(_HERE).parents[2] / "requirements.txt"
    runinfo = {
        "runner_version": RUNNER_VERSION, "selector_version": SELECTOR_VERSION_MS,
        "generator_version": MULTISUPPORT_GENERATOR_VERSION,
        "family_specs_version": FAMILY_SPECS_VERSION,
        "generic_model_version": GENERIC_MODEL_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_tracked": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "created_utc": started, "python_version": platform.python_version(),
        "platform": platform.platform(), "hostname": socket.gethostname(),
        "dependency_versions": dep,
        "requirements_sha256": sha256_file(req) if req.exists() else "",
        "protocol_sha256": psha, "lineage": "E (experimental prototype; not "
                                            "adoptable for the manuscript)",
        "numerics_mode": "consistent", "failure_policy": "fail_fast",
        "phase_runs": [],
    }
    atomic_write_json(run_dir / "runinfo.json", runinfo)
    update_checkpoint(run_dir, "init", 0, note="protocol frozen")
    print(json.dumps(calibration["selection"], indent=1))
    print(f"protocol sha256 {psha}")


def load_protocol(run_dir: Path) -> dict:
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    info = json.loads((run_dir / "runinfo.json").read_text(encoding="utf-8"))
    if sha256_file(run_dir / "protocol.json") != info["protocol_sha256"]:
        raise SystemExit("protocol.json changed after freezing; refusing to run")
    return protocol


def cmd_run(run_dir: Path, phase: str, workers: int) -> int:
    protocol = load_protocol(run_dir)
    cases = enumerate_cases(phase)
    done = completed_ids(run_dir)
    # repair: a case file without a completion record (crash between steps)
    for c in cases:
        if c["case_id"] not in done and case_file(run_dir, c["case_id"]).exists():
            result = json.loads(case_file(run_dir, c["case_id"]).read_text(encoding="utf-8"))
            _remove_case_rows(run_dir, c["case_id"])
            record_case(run_dir, result)
            done.add(c["case_id"])
    todo = [c for c in cases if c["case_id"] not in done]
    info_path = run_dir / "runinfo.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    head = _git("rev-parse", "HEAD")
    info["phase_runs"].append({"phase": phase, "started_utc": _now(),
                               "git_sha": head,
                               "git_dirty_tracked": bool(_git("status", "--porcelain",
                                                              "--untracked-files=no")),
                               "n_cases": len(cases), "n_todo": len(todo),
                               "workers": workers})
    atomic_write_json(info_path, info)
    # Dataset creation must not race between workers: create them first.
    seen = set()
    for c in todo:
        k = (c["stage"], c["exp_block"], c["group"], c["difficulty"], c["n"], c["replicate"])
        if k in seen:
            continue
        seen.add(k)
        try:
            get_dataset(run_dir, protocol, *k)
        except OUTCOME_EXCEPTIONS:
            pass        # the case itself records the generator stop
    update_checkpoint(run_dir, phase, len(cases), note="running")
    log = open(run_dir / "logs" / f"run_{phase}.log", "a", encoding="utf-8")
    log.write(f"{_now()} start phase={phase} todo={len(todo)}/{len(cases)} "
              f"workers={workers} git={head}\n")
    log.flush()
    unexpected = 0
    t0 = time.time()
    # Longest jobs first so the pool drains evenly.
    weight = {"AUTO_START_FIRST": 3, "AUTO_START_LAST": 3, "ORACLE": 1,
              "SUPPORT_ONLY": 1, "SELECTOR_ONLY": 0}
    todo.sort(key=lambda c: (-c["n"], -weight[c["method"]]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_case, str(run_dir), protocol, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                result = fut.result()
            except Exception as exc:                       # worker crashed
                result = {"case": c, "status": "FAILURE",
                          "error_class": "UNEXPECTED_EXCEPTION",
                          "failure_reason": f"worker: {type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc(), "rows": {}}
            # the dataset was created before the pool; mark creation once
            result["dataset_created"] = (result.get("dataset_key") is not None and
                                         not any(r.get("dataset_path") == result.get("dataset_path")
                                                 for r in read_csv(run_dir / "tables" /
                                                                   "dataset_manifest.csv")))
            record_case(run_dir, result)
            if result["error_class"] == "UNEXPECTED_EXCEPTION":
                unexpected += 1
            log.write(f"{_now()} [{i}/{len(todo)}] {c['case_id']} {result['status']} "
                      f"{result['error_class']} {result.get('runtime_seconds', '')}s "
                      f"{result['failure_reason'][:200]}\n")
            log.flush()
            update_checkpoint(run_dir, phase, len(cases),
                              note=f"{i}/{len(todo)} done in this invocation; "
                                   f"elapsed {time.time() - t0:.0f}s")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["phase_runs"][-1].update({"finished_utc": _now(),
                                   "unexpected_exceptions": unexpected,
                                   "elapsed_seconds": round(time.time() - t0, 1)})
    atomic_write_json(info_path, info)
    update_checkpoint(run_dir, phase, len(cases), note="phase finished")
    log.write(f"{_now()} end phase={phase} unexpected={unexpected}\n")
    log.close()
    print(f"phase {phase}: {len(todo)} cases run, unexpected={unexpected}, "
          f"{time.time() - t0:.0f}s")
    return 1 if unexpected else 0


def cmd_status(run_dir: Path) -> None:
    print(json.dumps(json.loads((run_dir / "checkpoint_state.json").read_text()), indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("init", "run", "status"))
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--phase", choices=tuple(PHASES))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--junit", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.command == "init":
        cmd_init(a.run_dir, a.junit)
        return 0
    if a.command == "status":
        cmd_status(a.run_dir)
        return 0
    if not a.phase:
        ap.error("--phase is required for run")
    return cmd_run(a.run_dir, a.phase, a.workers)


if __name__ == "__main__":
    sys.exit(main())
