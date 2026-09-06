"""Historical Poisson-X / Bernoulli-Y, K_TRUE=5: sweep n, change nothing else.

THE ONE QUESTION
----------------
The historical experiment picked K=5 for all five trials at n=150
(``expfam/results/exp2_bic_A.csv``, scenario A, k_true=5).  Does that
recovery survive a smaller n?

This is a BRIDGE experiment, not a competition with the current clean sweep.
It answers whether sample size alone can account for the difference, and
nothing else.

WHAT IS HELD FIXED
------------------
Everything the historical run used, reached through the historical modules
themselves -- ``data_generator_expfam.generate_dual_data``,
``utils_expfam.run_em_dual`` and ``utils_expfam.calc_bic_dual``.  That means
the historical generator (column z-scored Z, row-normalised F with
``|f_j|^2 = 1 - uniq = 0.9``, Poisson eta clipped to [-20, 10]), the
historical informed initialisation, the internal NaN fallback that halves
``newton_alpha``, the Q-based criterion and its parameter count, all Y pairs
used for training, w0 = -1.0 and w = 1.5 (the ``_Y_DEFAULTS`` for Bernoulli-Y,
which the historical runner took by not passing them), and the seed
convention

    data seed  = 8000 + k_true*1000 + trial*100
    model seed = data seed + 50

which depends on neither ``n`` nor ``k_est``, exactly as in
``run_exp2_bic_v2.py``.

**The historical files are not modified.**  Nothing from the current clean
lineage is mixed in: no canonical-clean-v1 generator, no matched signal, no
w = sqrt(3/K), no held-out Y, no S1, no second start, no masked prototype.

CRITERION NAMING
----------------
The column is called ``BIC`` because ``calc_bic_dual`` and the historical
artifact call it that, and past results must stay quotable.  It is
``-2*Q_strict + p*log(n)`` -- a Q-based complete-data criterion, NOT a
Schwarz BIC (KI-010).

Usage
-----
    python tools/research_audit/run_historical_pb_k5_n_sweep.py --gate
    python tools/research_audit/run_historical_pb_k5_n_sweep.py --small-n
    python tools/research_audit/run_historical_pb_k5_n_sweep.py --summarise
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_SRC = ROOT / "expfam" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

EXPERIMENT_ID = "historical_pb_k5_n_sweep_20260906"
OUT_DIR = ROOT / "expfam" / "results" / "k_selection" / EXPERIMENT_ID
FITS_CSV = OUT_DIR / "fit_results.csv"
RUNINFO = OUT_DIR / "runinfo.json"

#: The historical artifact this bridge is anchored to.
HISTORICAL_CSV = ROOT / "expfam" / "results" / "exp2_bic_A.csv"
HISTORICAL_SCENARIO = "A"

# ---- the historical configuration, restated verbatim ----------------------
K_TRUE = 5
D = 15
L = 5
NUM_ITER = 8
N_TRIALS = 5
K_EST_LIST = list(range(1, 11))
FAMILY_X = "poisson"
FAMILY_Y = "bernoulli"
BASE_SEED = 8000                      # run_exp2_bic_v2.run_exp2_bic default

GATE_N = 150                          # the n the historical run used
SMALL_N = (50, 75, 100)

FIT_COLUMNS = [
    "n", "trial", "k_est", "data_seed", "model_seed",
    "BIC", "num_params", "Q_strict", "Q_final",
    "rmse_Z", "rmse_X", "rmse_F", "rmse_Y",
    "w0_est", "w_est", "w0_err", "w_err",
    "y_density", "x_count_max",
    "nan_occurred", "nan_count", "fit_ok", "error", "runtime_s",
]


class Stop(RuntimeError):
    """A refusal.  Raised before or during a run."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stop(message)


def data_seed(trial: int) -> int:
    """run_exp2_bic_v2: base_seed + k_true*1000 + trial*100.  No n, no k_est."""

    return BASE_SEED + K_TRUE * 1000 + trial * 100


def model_seed(trial: int) -> int:
    return data_seed(trial) + 50


def git_head() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def working_tree_clean() -> bool:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                         capture_output=True, text=True)
    return out.returncode == 0 and out.stdout.strip() == ""


def environment() -> dict[str, Any]:
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:                                       # noqa: BLE001
        scipy_version = "absent"
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy_version, "platform": platform.system(),
            "machine": platform.machine()}


# ---------------------------------------------------------------------------
# the fit loop -- the historical call sequence, with n as the only variable
# ---------------------------------------------------------------------------

def run_cells(n_values, *, progress_every: int = 10) -> list[dict[str, Any]]:
    """Run every (n, trial, k_est) cell through the historical code path.

    The historical modules are imported HERE so that merely importing this
    file pulls in no model or EM code.
    """

    from data_generator_expfam import generate_dual_data      # noqa: PLC0415
    from utils_expfam import calc_bic_dual, run_em_dual       # noqa: PLC0415

    warnings.filterwarnings("ignore")
    rows: list[dict[str, Any]] = []
    total = len(n_values) * N_TRIALS * len(K_EST_LIST)
    done = 0
    started = time.time()

    for n in n_values:
        for trial in range(N_TRIALS):
            dseed, mseed = data_seed(trial), model_seed(trial)
            for k_est in K_EST_LIST:
                done += 1
                t0 = time.perf_counter()
                row: dict[str, Any] = {
                    "n": n, "trial": trial, "k_est": k_est,
                    "data_seed": dseed, "model_seed": mseed,
                }
                try:
                    # Called exactly as run_exp2_bic_v2 calls it: w0_true and
                    # w_true are NOT passed, so the Bernoulli-Y defaults
                    # (-1.0, 1.5) apply, as they did historically.
                    data = generate_dual_data(n=n, d=D, k=K_TRUE, seed=dseed,
                                              family_x=FAMILY_X,
                                              family_y=FAMILY_Y)
                    res = run_em_dual(
                        X=data["X"], Y=data["Y"], true_params=data,
                        family_x=FAMILY_X, family_y=FAMILY_Y,
                        k=k_est, L=L, num_iter=NUM_ITER, seed=mseed,
                        compute_strict_Q=True,
                    )
                    bic, npar = calc_bic_dual(
                        Q_strict=res["Q_strict"], k=k_est, n=n, d=D,
                        family_x=FAMILY_X, family_y=FAMILY_Y,
                    )
                    row.update({
                        "BIC": bic, "num_params": npar,
                        "Q_strict": res["Q_strict"], "Q_final": res["Q_final"],
                        "rmse_Z": res["rmse_Z"], "rmse_X": res["rmse_X"],
                        "rmse_F": res["rmse_F"], "rmse_Y": res["rmse_Y"],
                        "w0_est": res["w0_est"], "w_est": res["w_est"],
                        "w0_err": res["w0_err"], "w_err": res["w_err"],
                        "y_density": data.get("y_density"),
                        "x_count_max": data.get("x_count_max"),
                        "nan_occurred": bool(res["nan_occurred"]),
                        "nan_count": int(res["nan_count"]),
                        "fit_ok": True, "error": "",
                    })
                except Exception as exc:                     # noqa: BLE001
                    # Recorded, never dropped: a cell that failed is part of
                    # the result, and removing it would change the question.
                    row.update({k: float("nan") for k in (
                        "BIC", "Q_strict", "Q_final", "rmse_Z", "rmse_X",
                        "rmse_F", "rmse_Y", "w0_est", "w_est", "w0_err",
                        "w_err", "y_density", "x_count_max")})
                    row.update({"num_params": "", "nan_occurred": "",
                                "nan_count": "", "fit_ok": False,
                                "error": f"{type(exc).__name__}: {exc}"})
                    print(f"  ERROR n={n} trial={trial} k_est={k_est}: {exc}",
                          flush=True)

                row["runtime_s"] = round(time.perf_counter() - t0, 3)
                rows.append(row)
                if done % progress_every == 0 or done == total:
                    elapsed = time.time() - started
                    eta = elapsed / done * (total - done)
                    print(f"  [{done:3d}/{total}] n={n} trial={trial} "
                          f"k_est={k_est:2d} {elapsed/60:.1f}m "
                          f"ETA {eta/60:.1f}m", flush=True)
    return rows


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def selected_k_per_trial(rows, n: int) -> dict[int, int | None]:
    """argmin of the criterion over candidate K, within one trial.

    Ties go to the smaller K, which is what ``idxmin`` does on an ascending
    index and what the historical reporting relied on.
    """

    out: dict[int, int | None] = {}
    for trial in range(N_TRIALS):
        cells = [r for r in rows if int(r["n"]) == n and int(r["trial"]) == trial]
        usable = [r for r in cells
                  if str(r["fit_ok"]).lower() == "true"
                  and np.isfinite(float(r["BIC"]))]
        out[trial] = (min(usable, key=lambda r: (float(r["BIC"]),
                                                 int(r["k_est"])))["k_est"]
                      if usable else None)
    return {t: (int(k) if k is not None else None) for t, k in out.items()}


def selected_k_from_trial_mean(rows, n: int) -> int | None:
    """The rule ``run_exp2_bic_v2.print_bic_accuracy`` itself used.

    It averages the criterion over trials first, then takes the argmin.  That
    is a different statistic from the per-trial argmin, so both are reported
    rather than one silently standing in for the other.
    """

    means: dict[int, float] = {}
    for k_est in K_EST_LIST:
        vals = [float(r["BIC"]) for r in rows
                if int(r["n"]) == n and int(r["k_est"]) == k_est
                and str(r["fit_ok"]).lower() == "true"
                and np.isfinite(float(r["BIC"]))]
        if len(vals) == N_TRIALS:
            means[k_est] = float(np.mean(vals))
    return min(means, key=lambda k: (means[k], k)) if means else None


# ---------------------------------------------------------------------------
# the n=150 replication gate against the historical artifact
# ---------------------------------------------------------------------------

def read_historical_k5() -> dict[tuple[int, int], dict[str, float]]:
    """The historical rows for scenario A, k_true=5, keyed by (trial, k_est)."""

    _require(HISTORICAL_CSV.exists(), f"no historical artifact {HISTORICAL_CSV}")
    with HISTORICAL_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        if row["scenario"] != HISTORICAL_SCENARIO or int(row["k_true"]) != K_TRUE:
            continue
        key = (int(row["trial"]), int(row["k_est"]))
        _require(key not in out,
                 f"the historical artifact has a duplicate row for {key}")
        out[key] = {"BIC": float(row["BIC"]), "rmse_Z": float(row["rmse_Z"])}
    return out


def compare_with_historical(rows) -> dict[str, Any]:
    """Compare every n=150 cell with the historical artifact.

    Bit-identity is not expected across NumPy and BLAS versions, so the
    comparison is reported at several tolerances rather than asserted at one:
    the number that decides the gate is the selected K, and the numeric
    agreement is reported beside it so a reader can see how close it was.
    """

    historical = read_historical_k5()
    per_cell: list[dict[str, Any]] = []
    for row in rows:
        if int(row["n"]) != GATE_N:
            continue
        key = (int(row["trial"]), int(row["k_est"]))
        reference = historical.get(key)
        if reference is None:
            per_cell.append({**{k: row[k] for k in ("trial", "k_est")},
                             "found": False})
            continue
        bic_new, bic_old = float(row["BIC"]), reference["BIC"]
        z_new, z_old = float(row["rmse_Z"]), reference["rmse_Z"]
        per_cell.append({
            "trial": int(row["trial"]), "k_est": int(row["k_est"]),
            "found": True,
            "bic_historical": bic_old, "bic_new": bic_new,
            "bic_abs_diff": abs(bic_new - bic_old),
            "bic_rel_diff": abs(bic_new - bic_old) / abs(bic_old),
            "rmse_z_historical": z_old, "rmse_z_new": z_new,
            "rmse_z_abs_diff": abs(z_new - z_old),
        })
    found = [c for c in per_cell if c["found"]]
    exact = [c for c in found if c["bic_new"] == c["bic_historical"]]
    return {
        "compared_cells": len(found),
        "missing_cells": [c for c in per_cell if not c["found"]],
        "bit_identical_bic": len(exact),
        "max_bic_rel_diff": max((c["bic_rel_diff"] for c in found), default=None),
        "max_bic_abs_diff": max((c["bic_abs_diff"] for c in found), default=None),
        "max_rmse_z_abs_diff": max((c["rmse_z_abs_diff"] for c in found),
                                   default=None),
        "per_cell": per_cell,
    }


def run_gate() -> dict[str, Any]:
    """n=150 only: 5 trials x candidate K 1..10 = 50 fits."""

    _require(not FITS_CSV.exists(),
             f"{FITS_CSV} already exists; results are never overwritten")
    print(f"replication gate: n={GATE_N}, {N_TRIALS} trials x "
          f"{len(K_EST_LIST)} candidate K = "
          f"{N_TRIALS * len(K_EST_LIST)} fits", flush=True)

    started = time.time()
    rows = run_cells([GATE_N])
    comparison = compare_with_historical(rows)

    historical_selected = {t: 5 for t in range(N_TRIALS)}   # the archived result
    selected = selected_k_per_trial(rows, GATE_N)
    failures = [r for r in rows if str(r["fit_ok"]).lower() != "true"]
    nans = [r for r in rows if str(r["nan_occurred"]).lower() == "true"]

    passed = (selected == historical_selected
              and not failures and not nans
              and comparison["compared_cells"] == N_TRIALS * len(K_EST_LIST)
              and not comparison["missing_cells"])

    result = {
        "verdict": "PASS" if passed else "FAIL",
        "experiment_id": EXPERIMENT_ID,
        "gate_n": GATE_N,
        "expected_fits": N_TRIALS * len(K_EST_LIST),
        "actual_fits": len(rows),
        "selected_k_per_trial": selected,
        "selected_k_historical_artifact": historical_selected,
        "selected_k_from_trial_mean": selected_k_from_trial_mean(rows, GATE_N),
        "failed_fits": len(failures),
        "nan_fits": len(nans),
        "historical_comparison": comparison,
        "wall_clock_seconds": round(time.time() - started, 1),
        "code_sha": git_head(),
        "working_tree_clean": working_tree_clean(),
        "environment": environment(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_rows(rows)
    write_json(OUT_DIR / "replication_gate.json", result)
    print(json.dumps({k: result[k] for k in (
        "verdict", "expected_fits", "actual_fits", "selected_k_per_trial",
        "failed_fits", "nan_fits", "wall_clock_seconds")},
        indent=2, default=str))
    print("historical agreement: "
          f"{comparison['bit_identical_bic']}/{comparison['compared_cells']} "
          f"bit-identical BIC, max relative difference "
          f"{comparison['max_bic_rel_diff']}")
    return result


def run_small_n() -> dict[str, Any]:
    """n in {50, 75, 100}: 3 x 5 x 10 = 150 new fits, appended to the gate's."""

    _require(FITS_CSV.exists(),
             f"no {FITS_CSV}; run --gate first")
    existing = read_rows()
    have = {int(r["n"]) for r in existing}
    _require(have == {GATE_N},
             f"{FITS_CSV} already holds n={sorted(have)}; the small-n run is "
             f"appended exactly once")
    gate = json.loads((OUT_DIR / "replication_gate.json").read_text(encoding="utf-8"))
    _require(gate["verdict"] == "PASS",
             f"the replication gate says {gate['verdict']}; the small-n run "
             f"does not start")

    started = time.time()
    rows = run_cells(list(SMALL_N))
    write_rows(existing + rows)
    result = {
        "new_fits": len(rows),
        "total_fits": len(existing) + len(rows),
        "wall_clock_seconds": round(time.time() - started, 1),
        "code_sha": git_head(),
        "environment": environment(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(RUNINFO, {
        "experiment_id": EXPERIMENT_ID,
        "question": ("does the historical Poisson-X / Bernoulli-Y K_TRUE=5 "
                     "recovery survive a smaller n?"),
        "held_fixed": (
            "historical generator (generate_dual_data), historical EM "
            "(run_em_dual), historical Q-based criterion (calc_bic_dual), "
            "K_TRUE=5, candidate K 1..10, d=15, L=5, num_iter=8, 5 trials, "
            "1 start, w0=-1.0, w=1.5 from _Y_DEFAULTS, seed convention "
            "8000 + k_true*1000 + trial*100 and +50 for the model"),
        "varied": "n only: 150 (gate) and 50, 75, 100",
        "historical_artifact": str(HISTORICAL_CSV.relative_to(ROOT)).replace("\\", "/"),
        "gate": gate,
        "small_n_run": result,
        "expected_fits": (1 + len(SMALL_N)) * N_TRIALS * len(K_EST_LIST),
        "actual_fits": len(existing) + len(rows),
        "criterion_note": (
            "the BIC column is -2*Q_strict + p*log(n) from calc_bic_dual: a "
            "Q-based complete-data criterion, NOT a Schwarz BIC (KI-010). The "
            "name is kept because the historical artifact uses it"),
    })
    print(json.dumps(result, indent=2, default=str))
    return result


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")


def write_rows(rows) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with FITS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_rows() -> list[dict[str, Any]]:
    with FITS_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def summarise() -> dict[str, Any]:
    rows = read_rows()
    all_n = sorted({int(r["n"]) for r in rows})
    per_n = {}
    for n in all_n:
        selected = selected_k_per_trial(rows, n)
        values = [selected[t] for t in range(N_TRIALS)]
        per_n[n] = {
            "selected_k_per_trial": values,
            "exact": sum(1 for v in values if v == K_TRUE),
            "under": sum(1 for v in values if v is not None and v < K_TRUE),
            "over": sum(1 for v in values if v is not None and v > K_TRUE),
            "unresolved": sum(1 for v in values if v is None),
            "selected_k_from_trial_mean": selected_k_from_trial_mean(rows, n),
            "mean_criterion_by_k": {
                k: float(np.mean([float(r["BIC"]) for r in rows
                                  if int(r["n"]) == n and int(r["k_est"]) == k
                                  and str(r["fit_ok"]).lower() == "true"]))
                for k in K_EST_LIST},
            "mean_rmse_z_at_k_true": float(np.mean(
                [float(r["rmse_Z"]) for r in rows
                 if int(r["n"]) == n and int(r["k_est"]) == K_TRUE
                 and str(r["fit_ok"]).lower() == "true"])),
            "failed_fits": sum(1 for r in rows if int(r["n"]) == n
                               and str(r["fit_ok"]).lower() != "true"),
            "nan_fits": sum(1 for r in rows if int(r["n"]) == n
                            and str(r["nan_occurred"]).lower() == "true"),
        }
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "k_true": K_TRUE,
        "candidate_k": K_EST_LIST,
        "trials": N_TRIALS,
        "total_fits": len(rows),
        "by_n": per_n,
        "rmse_note": ("rmse_Z is reported for description only. It is NOT a "
                      "K-selection criterion and is not used to select K"),
    }
    write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true",
                        help="n=150 only, compared with exp2_bic_A.csv")
    parser.add_argument("--small-n", action="store_true",
                        help="n in {50,75,100}; requires a PASSed gate")
    parser.add_argument("--summarise", action="store_true")
    args = parser.parse_args(argv)

    if args.gate:
        return 0 if run_gate()["verdict"] == "PASS" else 1
    if args.small_n:
        run_small_n()
        return 0
    if args.summarise:
        summarise()
        return 0
    parser.error("choose --gate, --small-n or --summarise")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
