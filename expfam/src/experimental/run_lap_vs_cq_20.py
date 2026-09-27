"""Phase 9K (Issue #94): frozen 20-dataset paired characterization, C_Lap vs C_Q.

Reuses unchanged: the Phase 9D/9E joint family + K runner, the Phase 9H
evaluator, and the Phase 9I per-refit evaluation (``run_laplace_pilot.
laplace_row``: theta = refit final, Z0 = refit Z_est, grad_tol 1e-8,
max_iter 200, no restart/jitter/ridge; C_Lap = −2·laplace + d_K ln 75 with
d_K = K d − K(K−1)/2 + n_gaussian_selected) and its completeness rule
(``dataset_summary``: K_hat_Lap only when all five K are OK).

Set here only: the 20 frozen datasets (1001000+r / 1002000+r / 1003000+r),
start_B only, the paired summary, and a Git-trackable JSON of every
technically successful refit's fitted state. No importance-sampling
correction (Phase 9J: weights too degenerate). N = 75 is a working
convention. Nothing here ranks the two criteria.

Lineage E (experimental prototype; not adoptable for the manuscript).

Usage (once, fresh directory)::

    python run_lap_vs_cq_20.py --out <fresh directory>
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import statistics
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
import run_laplace_pilot as lp                                     # noqa: E402

WRAPPER_VERSION = "lap-vs-cq-20-v1"
K_TRUE = 3

PROTOCOL = dataclasses.replace(
    joint.PROTOCOL,
    stage="lap_vs_cq_20",
    replicates=tuple(joint.Replicate(f"rep{r:02d}", 1001000 + r,
                                     1002000 + r, 1003000 + r)
                     for r in range(1, 21)),
    starts=(("start_B", "bernoulli"),),
    failure_scope="dataset",
)

AUTHORIZATION: dict[str, Any] = {
    "gate": "Phase 9K bounded 20-dataset paired characterization",
    "authorized": True,
    "authorized_by": "Human",
    "authorized_in": "Issue #94 body (bounded authorization)",
}


def _replicate_for(search_seed: int) -> str:
    for rep in PROTOCOL.replicates:
        if rep.search_seed == search_seed:
            return rep.label
    raise joint.RunnerStop(f"search seed {search_seed} is not frozen")


def fitted_state(result: dict[str, Any], replicate: str, k: int
                 ) -> dict[str, Any]:
    refit = result["refit"]
    params = refit["model"].params
    rep = next(r for r in PROTOCOL.replicates if r.label == replicate)
    return {
        "replicate": replicate, "k": k, "data_seed": rep.data_seed,
        "search_seed": rep.search_seed, "refit_seed": rep.refit_seed,
        "selected_assignment": list(result["selected_assignment"]),
        "F": np.asarray(params["F"], float).tolist(),
        "sigma_diag": np.diag(np.asarray(params["sigma"], float)).tolist(),
        "w0": float(params["w0"]), "w": float(params["w"]),
        "var_z": float(params["var_z"]),
        "Z_est": np.asarray(refit["Z_est"], float).tolist(),
    }


def make_driver(rows: list, states: list,
                inner: Callable[..., dict[str, Any]] | None = None):
    """Run the #74 driver unchanged, then the Phase 9I evaluation on the
    refit it returns, and keep the fitted state."""

    def driver(X, Y, **kwargs):
        run = inner if inner is not None else fs.run_hybrid_family_selection
        result = run(X, Y, **kwargs)
        label = _replicate_for(kwargs["search_seed"])
        rows.append(lp.laplace_row(result, X, Y, kwargs["k"], label))
        states.append(fitted_state(result, label, kwargs["k"]))
        return result

    return driver


# --------------------------------------------------------------------------
# paired summary
# --------------------------------------------------------------------------

def best_second(curve: dict[int, float]) -> tuple[int, int, float]:
    order = joint.cq_ordering(curve)
    return order[0], order[1], curve[order[1]] - curve[order[0]]


def category(k_hat: int | None) -> str | None:
    if k_hat is None:
        return None
    return ("exact" if k_hat == K_TRUE else
            "under" if k_hat < K_TRUE else "over")


def _dist(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": min(values),
            "median": statistics.median(values), "max": max(values),
            "negative": sum(v < 0 for v in values),
            "positive": sum(v > 0 for v in values),
            "zero": sum(v == 0 for v in values)}


def paired_summary(rows: Sequence[dict[str, Any]],
                   family_rows: Sequence[dict[str, Any]]
                   ) -> dict[str, Any]:
    per = []
    for rep in PROTOCOL.replicates:
        d = lp.dataset_summary(rows, family_rows, rep.label)
        if d["c_lap_complete"]:
            best, second, gap = best_second(d["C_Lap"])
            d.update({"lap_best_k": best, "lap_second_k": second,
                      "lap_gap": gap})
        if d["K_hat_Q"] is not None:
            best, second, gap = best_second(d["C_Q"])
            d.update({"q_best_k": best, "q_second_k": second,
                      "q_gap": gap})
        per.append(d)

    lap = [d for d in per if d["c_lap_complete"]]
    q = [d for d in per if d["K_hat_Q"] is not None]
    paired = [d for d in per if d["c_lap_complete"]
              and d["K_hat_Q"] is not None]

    def counts(items, key):
        return {k: sum(d[key] == k for d in items) for k in range(1, 6)}

    def cats(items, key):
        c = [category(d[key]) for d in items]
        return {x: c.count(x) for x in ("exact", "under", "over")}

    statuses = [r["laplace_status"] for r in rows]
    gaps = sorted(lap, key=lambda d: d["lap_gap"])
    return {
        "wrapper_version": WRAPPER_VERSION,
        "laplace_settings": lp.LAPLACE_SETTINGS,
        "datasets_planned": len(PROTOCOL.replicates),
        "em_complete_datasets": len(q),
        "c_lap_complete_datasets": len(lap),
        "P1_K_hat_Lap": {"counts": counts(lap, "K_hat_Lap"),
                         **cats(lap, "K_hat_Lap"), "denominator": len(lap)},
        "P2_K_hat_Q": {"counts": counts(q, "K_hat_Q"),
                       **cats(q, "K_hat_Q"), "denominator": len(q)},
        "P3_paired": {
            "denominator": len(paired),
            "both_K3": sum(d["K_hat_Q"] == 3 and d["K_hat_Lap"] == 3
                           for d in paired),
            "lap_only_K3": sum(d["K_hat_Q"] != 3 and d["K_hat_Lap"] == 3
                               for d in paired),
            "q_only_K3": sum(d["K_hat_Q"] == 3 and d["K_hat_Lap"] != 3
                             for d in paired),
            "neither_K3": sum(d["K_hat_Q"] != 3 and d["K_hat_Lap"] != 3
                              for d in paired),
            "same_K": sum(d["K_hat_Q"] == d["K_hat_Lap"] for d in paired),
            "different_K": sum(d["K_hat_Q"] != d["K_hat_Lap"]
                               for d in paired),
            "pairs": {d["replicate"]: [d["K_hat_Q"], d["K_hat_Lap"]]
                      for d in per},
        },
        "P4_margins": {
            "delta_Lap_23": _dist([d["delta_Lap_23"] for d in lap]),
            "delta_Q_23": _dist([d["delta_Q_23"] for d in q]),
            "lap_gap": _dist([d["lap_gap"] for d in lap]),
            "q_gap": _dist([d["q_gap"] for d in q]),
            "five_smallest_lap_gaps": [
                {"replicate": d["replicate"], "gap": d["lap_gap"],
                 "best_k": d["lap_best_k"], "second_k": d["lap_second_k"]}
                for d in gaps[:5]],
        },
        "P5_technical": {
            "refits_evaluated": len(rows),
            "OK": statuses.count(lk.STATUS_OK),
            "NOT_STATIONARY": statuses.count(lk.STATUS_NOT_STATIONARY),
            "HESSIAN_NOT_PD": statuses.count(lk.STATUS_HESSIAN_NOT_PD),
            "EVALUATION_ERROR": statuses.count(lp.STATUS_EVALUATION_ERROR),
        },
        "datasets": per,
        "claim_boundary": {
            "unit": "20 prospectively frozen datasets under one condition, "
                    "start_B only; C_Q and C_Lap on the same refits",
            "do_not_claim": ["consistency", "general superiority",
                             "general recovery rate", "robustness",
                             "generalisation to other n/d/K_true/signal",
                             "theoretical correctness of N = 75",
                             "exact marginal likelihood",
                             "a Laplace error bound",
                             "real-data effectiveness"],
            "not_pooled_with": ["Phase 9E", "Phase 9I", "Phase 9J"],
            "lineage": "E (experimental prototype; not adoptable for the "
                       "manuscript)",
        },
    }


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_states(path: Path, states: Sequence[dict[str, Any]],
                 code_sha: str) -> None:
    payload = {
        "code_sha": code_sha, "evaluator_version": lk.EVALUATOR_VERSION,
        "wrapper_version": WRAPPER_VERSION,
        "x_y_provenance": "X and Y are not stored. They are fully "
                          "regenerable from data_seed with "
                          "run_family_selection_pilot.build_dataset("
                          "run_lap_vs_cq_20.PROTOCOL, replicate).",
        "float_format": "Python round-trip float repr (exact float64)",
        "entries": list(states),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Issue #94 Phase 9K paired study once.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    rows: list = []
    states: list = []
    code_sha = pilot._git_sha()
    try:
        joint.execute(args.out, protocol=PROTOCOL,
                      driver=make_driver(rows, states),
                      authorization=AUTHORIZATION)
    finally:
        if args.out.is_dir():
            if rows:
                pilot._write_csv(args.out / "laplace_by_k.csv", rows)
            if states:
                write_states(args.out / "fitted_states.json", states,
                             code_sha)
    summary = paired_summary(rows, _read(args.out / "family_by_k.csv"))
    pilot._write_json(args.out / "paired_summary.json", summary)
    print(json.dumps({k: summary[k] for k in
                      ("c_lap_complete_datasets", "P1_K_hat_Lap",
                       "P2_K_hat_Q", "P5_technical")}, default=str))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
