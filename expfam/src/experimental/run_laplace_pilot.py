"""Phase 9I (Issue #90): frozen 3-dataset pilot of the Candidate-B Laplace criterion.

Reuses the Phase 9D/9E joint family + K runner unchanged and the merged
Phase 9H evaluator. Only the scientific differences are set here:

- three new frozen datasets rep01..rep03, seeds 981000+r / 982000+r /
  983000+r (data / family search / refit); start_B only; K = 1..5
- after each fresh refit, Candidate B is evaluated post hoc on THAT refit:
  theta = the refit's final parameters, joint-mode start Z0 = the refit's
  Z_est, grad_tol 1e-8, max_iter 200 (Phase 9H defaults), no restart,
  no alternate start, no jitter/ridge
- C_Lap(K) = −2 · laplace_log_observed + d_K · log(N), N = 75 (a working
  convention, not a proven sample size), d_K = K d − K(K−1)/2 +
  n_gaussian_selected(K); w0/w not counted (K-independent)
- a dataset is C_Lap-complete only if all five K evaluate with status OK;
  only then is K_hat_Lap = argmin C_Lap computed (never over a subset)

The current C_Q is read from the same refit, unchanged. Nothing here ranks
the two criteria. Lineage E (experimental prototype; not manuscript).

Usage (once, fresh directory)::

    python run_laplace_pilot.py --out <fresh directory>
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import family_selection as fs                                      # noqa: E402
import laplace_k_criterion as lk                                   # noqa: E402
import run_family_selection_pilot as pilot                         # noqa: E402
import run_joint_family_k_selection as joint                       # noqa: E402

WRAPPER_VERSION = "laplace-pilot-v1"

PROTOCOL = dataclasses.replace(
    joint.PROTOCOL,
    stage="laplace_pilot",
    replicates=tuple(joint.Replicate(f"rep{r:02d}", 981000 + r, 982000 + r,
                                     983000 + r) for r in (1, 2, 3)),
    starts=(("start_B", "bernoulli"),),
    failure_scope="dataset",
)

AUTHORIZATION: dict[str, Any] = {
    "gate": "Phase 9I bounded Laplace pilot",
    "authorized": True,
    "authorized_by": "Human",
    "authorized_in": "Issue #90 body (bounded execution authorization)",
}

LAPLACE_SETTINGS: dict[str, Any] = {
    "evaluator_version": lk.EVALUATOR_VERSION,
    "theta": "fresh refit final parameters",
    "joint_mode_start": "fresh refit Z_est",
    "grad_tol": 1e-8,
    "max_iter": 200,
    "restart": False, "alternate_start": False, "jitter_or_ridge": False,
    "N": 75,
    "N_status": "working convention (Phase 9G), not a proven sample size",
    "d_K": "K*d - K(K-1)/2 + n_gaussian_selected(K); w0/w not counted",
}

STATUS_EVALUATION_ERROR = "EVALUATION_ERROR"


def d_K(k: int, d: int, n_gaussian_selected: int) -> int:
    return lk.loading_parameter_count(k, d) + int(n_gaussian_selected)


def _replicate_for(search_seed: int) -> str:
    for rep in PROTOCOL.replicates:
        if rep.search_seed == search_seed:
            return rep.label
    raise joint.RunnerStop(f"search seed {search_seed} is not frozen")


def laplace_row(result: dict[str, Any], X: np.ndarray, Y: np.ndarray,
                k: int, replicate: str) -> dict[str, Any]:
    """Evaluate Candidate B on one refit. Never raises for a numerical
    failure of the evaluator; records it as a status instead."""

    refit = result["refit"]
    selected = result["selected_assignment"]
    n_gauss = sum(f == "gaussian" for f in selected)
    dk = d_K(k, X.shape[1], n_gauss)
    row: dict[str, Any] = {
        "replicate": replicate, "start_label": "start_B", "k": k,
        "selected_assignment": "|".join(selected),
        "n_gaussian_selected": n_gauss, "d_K": dk,
        "N": LAPLACE_SETTINGS["N"], "C_Q": refit.get("bic"),
        "Q_strict": refit.get("Q_strict"),
        "num_params_C_Q": refit.get("num_params"),
    }
    try:
        res = lk.laplace_log_observed(
            refit["model"], X, Y, np.asarray(refit["Z_est"], float),
            grad_tol=LAPLACE_SETTINGS["grad_tol"],
            max_iter=LAPLACE_SETTINGS["max_iter"])
        out = lk.calc_C_Lap(res, d_K=dk, N=LAPLACE_SETTINGS["N"])
        row.update({
            "laplace_status": res.status,
            "laplace_log_observed": res.log_observed,
            "C_Lap": out["C_Lap"],
            "integration_term": out["integration_term"],
            "parameter_term": out["parameter_term"],
            "phi_at_mode": res.phi_at_mode,
            "logdet_H": res.logdet_H, "logdet_sign": res.logdet_sign,
            "min_eigenvalue_H": res.min_eigenvalue_H,
            "grad_inf": res.grad_inf, "joint_mode_iterations": res.iterations,
            "Z_hat_sq_norm": float(np.sum(res.Z_hat ** 2)),
            "Z0_sq_norm": float(np.sum(np.asarray(refit["Z_est"]) ** 2)),
            "notes": "; ".join(res.notes),
        })
    except lk.UnsupportedLaplacePath:
        raise                                   # a systemic defect: stop
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
        row.update({"laplace_status": STATUS_EVALUATION_ERROR,
                    "C_Lap": float("nan"),
                    "notes": f"{type(exc).__name__}: {exc}"})
    return row


def make_driver(rows: list[dict[str, Any]],
                inner: Callable[..., dict[str, Any]] | None = None):
    """Wrap the #74 driver: run it unchanged, then evaluate Candidate B on
    the refit it returns. The EM calls and their ledger are untouched."""

    def driver(X, Y, **kwargs):
        run = inner if inner is not None else fs.run_hybrid_family_selection
        result = run(X, Y, **kwargs)
        rows.append(laplace_row(result, X, Y, kwargs["k"],
                                _replicate_for(kwargs["search_seed"])))
        return result

    return driver


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def dataset_summary(laplace_rows: Sequence[dict[str, Any]],
                    family_rows: Sequence[dict[str, Any]],
                    replicate: str, k_grid=(1, 2, 3, 4, 5)
                    ) -> dict[str, Any]:
    mine = {int(r["k"]): r for r in laplace_rows
            if r["replicate"] == replicate}
    have_all = set(mine) == set(k_grid)
    c_q = {k: float(r["C_Q"]) for k, r in mine.items()}
    k_hat_q = joint.select_k_hat(c_q) if have_all else None
    complete = have_all and all(r["laplace_status"] == lk.STATUS_OK
                                for r in mine.values())
    c_lap = {k: float(r["C_Lap"]) for k, r in mine.items()}
    k_hat_lap = joint.select_k_hat(c_lap) if complete else None

    def cols_3_8(k):
        if k is None:
            return None
        fam = {int(r["column"]): r["selected_family"] for r in family_rows
               if r["replicate"] == replicate and int(r["k"]) == k}
        return [fam.get(c) for c in range(3, 9)]

    return {
        "replicate": replicate,
        "em_complete": have_all,
        "c_lap_complete": complete,
        "statuses": {k: mine[k]["laplace_status"] for k in sorted(mine)},
        "C_Q": c_q, "C_Lap": c_lap,
        "K_hat_Q": k_hat_q, "K_hat_Lap": k_hat_lap,
        "delta_Q_23": (c_q[3] - c_q[2]) if {2, 3} <= set(c_q) else None,
        "delta_Lap_23": (c_lap[3] - c_lap[2]) if complete else None,
        "cols_3_8_at_K_hat_Q": cols_3_8(k_hat_q),
        "cols_3_8_at_K_hat_Lap": cols_3_8(k_hat_lap),
    }


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def analyse(run_dir: Path, laplace_rows: Sequence[dict[str, Any]]
            ) -> dict[str, Any]:
    family_rows = _read(run_dir / "family_by_k.csv")
    cq_rows = _read(run_dir / "cq_by_k.csv")
    datasets = [dataset_summary(laplace_rows, family_rows, r.label)
                for r in PROTOCOL.replicates]
    statuses = [r["laplace_status"] for r in laplace_rows]
    return {
        "wrapper_version": WRAPPER_VERSION,
        "laplace_settings": LAPLACE_SETTINGS,
        "P0_evaluability": {
            "refits_evaluated": len(laplace_rows),
            "OK": statuses.count(lk.STATUS_OK),
            "NOT_STATIONARY": statuses.count(lk.STATUS_NOT_STATIONARY),
            "HESSIAN_NOT_PD": statuses.count(lk.STATUS_HESSIAN_NOT_PD),
            "EVALUATION_ERROR": statuses.count(STATUS_EVALUATION_ERROR),
            "c_lap_complete_datasets": sum(d["c_lap_complete"]
                                           for d in datasets),
        },
        "datasets": datasets,
        "integrity": {c: sum(int(r[c]) for r in cq_rows)
                      for c in ("retry_count", "replacement_count",
                                "seed_rescue_count")},
        "claim_boundary": {
            "unit": "3 frozen datasets, start_B only; descriptive",
            "comparison": "C_Q and C_Lap on the same refits; no claim that "
                          "either is better or correct",
            "N": LAPLACE_SETTINGS["N_status"],
            "lineage": "E (experimental prototype; not adoptable for the "
                       "manuscript)",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Issue #90 Phase 9I Laplace pilot once.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    rows: list[dict[str, Any]] = []
    try:
        joint.execute(args.out, protocol=PROTOCOL, driver=make_driver(rows),
                      authorization=AUTHORIZATION)
    finally:
        if rows and args.out.is_dir():
            pilot._write_csv(args.out / "laplace_by_k.csv", rows)
    summary = analyse(args.out, rows)
    pilot._write_json(args.out / "laplace_pilot_summary.json", summary)
    print(json.dumps(summary["P0_evaluability"]))
    for d in summary["datasets"]:
        print(d["replicate"], "K_hat_Q", d["K_hat_Q"], "K_hat_Lap",
              d["K_hat_Lap"], "complete", d["c_lap_complete"])
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
