"""Phase 9O (Issue #102): classify the three Phase 9N DERIVATIVE_UNAVAILABLE
endpoints. Classification only: no theta update, no optimization step.

At each saved Phase 9N endpoint (rep04 K=2, rep09 K=3, rep09 K=4):
  - X, Y regenerated from the Phase 9K data seed with the unchanged generator
  - theta and the final Z mode restored from Phase 9N final_theta.json
  - the chart is rebuilt from the endpoint F with the same split_basis path
    and the coordinates are ordered as in Phase 9N:
    F_horizontal[j], log_sigma[column=l] (Gaussian columns), w0, w
  - the unperturbed endpoint is evaluated once, then every coordinate once
    at +h and once at -h with the Phase 9N step sizes (SavedState.step_sizes
    / 2), every call starting the joint-mode solver at the saved final Z mode
The structured Candidate-B result of every call is kept (not collapsed to
NaN); an exception is recorded with its class and message. Lineage E.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import laplace_k_criterion as lk                                   # noqa: E402
import run_family_selection_pilot as pilot                         # noqa: E402
import run_lap_vs_cq_20 as pc                                      # noqa: E402
import theta_stationarity_diagnostic as td                         # noqa: E402

CLASSIFIER_VERSION = "derivative-failure-classification-v1"
TARGETS = (("rep04", 2, 4), ("rep09", 3, 14), ("rep09", 4, 6))
SOURCE_DIR = Path("expfam/results/multistep_local/phase9n_20260928")
PHASE9K_DIR = td.SOURCE_DIR

MECH_OK = "OK"
MECH_NOT_STATIONARY = "A_NOT_STATIONARY"
MECH_HESSIAN = "B_HESSIAN_NOT_PD"
MECH_EXCEPTION = "C_EXCEPTION"
MECH_NONFINITE = "D_NONFINITE_OTHER"
MECH_OTHER = "E_OTHER_STRUCTURED_NON_OK"


def build_state(entry: dict[str, Any], X: np.ndarray,
                Y: np.ndarray) -> td.SavedState:
    """Endpoint state: theta and chart at the Phase 9N endpoint."""
    state = td.SavedState({**entry, "Z_est": entry["Z_mode"]}, X, Y)
    state.Z_hat0 = np.array(entry["Z_mode"], float)   # the fixed Z start
    return state


def coordinate_labels(state: td.SavedState) -> list[tuple[str, str]]:
    """(block, label) in the Phase 9N coordinate order."""
    return ([("F_horizontal", f"F_horizontal[{j}]")
             for j in range(state.n_horizontal)]
            + [("log_sigma", f"log_sigma[column={l}]") for l in state.gauss]
            + [("w0", "w0"), ("w", "w")])


def evaluate_structured(state: td.SavedState,
                        theta: np.ndarray) -> dict[str, Any]:
    """One Candidate-B call at theta from the fixed Z start; nothing dropped."""
    state.set_theta(theta)
    try:
        res = lk.laplace_log_observed(state.model, state.X, state.Y,
                                      state.Z_hat0)
    except Exception as exc:                        # recorded, never retried
        return {"status": "EXCEPTION", "log_observed": float("nan"),
                "log_observed_finite": False, "phi_at_mode": float("nan"),
                "grad_inf": float("nan"), "iterations": -1,
                "min_eigenvalue_H": float("nan"), "logdet_sign": float("nan"),
                "logdet_H": float("nan"), "notes": "",
                "exception_class": type(exc).__name__,
                "exception_message": str(exc)}
    return {"status": res.status, "log_observed": res.log_observed,
            "log_observed_finite": bool(math.isfinite(res.log_observed)),
            "phi_at_mode": res.phi_at_mode, "grad_inf": res.grad_inf,
            "iterations": res.iterations,
            "min_eigenvalue_H": res.min_eigenvalue_H,
            "logdet_sign": res.logdet_sign, "logdet_H": res.logdet_H,
            "notes": "; ".join(res.notes),
            "exception_class": "", "exception_message": ""}


def mechanism(record: dict[str, Any]) -> str:
    status = record["status"]
    if status == "EXCEPTION":
        return MECH_EXCEPTION
    if status == lk.STATUS_OK:
        return MECH_OK if record["log_observed_finite"] else MECH_NONFINITE
    if status == lk.STATUS_NOT_STATIONARY:
        return MECH_NOT_STATIONARY
    if status == lk.STATUS_HESSIAN_NOT_PD:
        return MECH_HESSIAN
    return MECH_OTHER


def classify_state(state: td.SavedState) -> list[dict[str, Any]]:
    """+h and -h once per coordinate; the state's theta is never updated."""
    steps = state.step_sizes() / 2                  # Phase 9N primary h
    rows = []
    for i, (block, label) in enumerate(coordinate_labels(state)):
        for side, sign in (("+", 1.0), ("-", -1.0)):
            e = np.zeros(state.n_coords)
            e[i] = sign * steps[i]
            record = evaluate_structured(state, e)
            rows.append({"replicate": state.replicate, "k": state.k,
                         "coordinate_index": i, "block": block,
                         "label": label, "side": side, "h": float(steps[i]),
                         **record, "mechanism": mechanism(record)})
    state.set_theta()                               # leave theta at endpoint
    return rows


def summarize_state(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    failing = [r for r in rows if r["mechanism"] != MECH_OK]
    coords = sorted({r["coordinate_index"] for r in failing})
    return {
        "coordinates": len({r["coordinate_index"] for r in rows}),
        "failing_coordinates": len(coords),
        "failing_sides": len(failing),
        "failures": [{"coordinate_index": r["coordinate_index"],
                      "label": r["label"], "block": r["block"],
                      "side": r["side"], "status": r["status"],
                      "mechanism": r["mechanism"], "grad_inf": r["grad_inf"],
                      "min_eigenvalue_H": r["min_eigenvalue_H"],
                      "logdet_sign": r["logdet_sign"],
                      "logdet_H": r["logdet_H"],
                      "iterations": r["iterations"], "notes": r["notes"],
                      "exception_class": r["exception_class"],
                      "exception_message": r["exception_message"]}
                     for r in failing],
        "pattern": ("none" if not coords else
                    "one coordinate, one side" if len(failing) == 1 else
                    "one coordinate, both sides" if len(coords) == 1 else
                    "multiple coordinates"),
    }


def decide(summaries: Sequence[dict[str, Any]]) -> str:
    """All endpoints reconstructed (checked before this is reached)."""
    if any(s["failing_coordinates"] == 0 for s in summaries):
        return "FAILURE_REPRODUCTION_MISMATCH"
    if any(s["failing_coordinates"] != 1 for s in summaries):
        return "FAILURE_REPRODUCTION_MISMATCH"
    return "FAILURE_MECHANISM_CLASSIFIED"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; a recorded run is never "
                         f"overwritten")
    final = json.loads((args.source / "final_theta.json").read_text("utf-8"))
    per_state = {(r["replicate"], int(r["k"])): r
                 for r in td._read_csv(args.source / "per_state_final.csv")}
    trajectory = td._read_csv(args.source / "trajectory.csv")
    k9 = {(e["replicate"], int(e["k"])): e for e in json.loads(
        (PHASE9K_DIR / "fitted_states.json").read_text("utf-8"))["entries"]}
    entries = {(e["replicate"], int(e["k"])): e for e in final["entries"]}

    def blocker(msg: str) -> None:
        raise SystemExit(f"systemic reconstruction blocker: {msg}")

    for rep, k, iters in TARGETS:                    # artifact cross-checks
        e, p, s9 = entries[(rep, k)], per_state[(rep, k)], k9[(rep, k)]
        traj = [t for t in trajectory
                if t["replicate"] == rep and int(t["k"]) == k]
        checks = {
            "status": e["status"] == p["status"] == "DERIVATIVE_UNAVAILABLE",
            "iterations": e["iterations"] == int(p["iterations"])
            == len(traj) == iters,
            "final_ell": e["final_ell"] == float(p["final_ell"])
            == float(traj[-1]["ell_Lap"]),
            "seeds/assignment": all(e[x] == s9[x] for x in
                                    ("data_seed", "search_seed", "refit_seed",
                                     "selected_assignment")),
            "reason": p["reason"] == "1 gradient component(s) unavailable",
        }
        if not all(checks.values()):
            blocker(f"{rep} K={k} artifact check failed: {checks}")

    args.out.mkdir(parents=True)
    protocol = {
        "classifier_version": CLASSIFIER_VERSION,
        "code_sha": pilot._git_sha(), "git_dirty": pilot._git_dirty(),
        "source_dir": str(args.source),
        "source_sha256": {n: td._sha256(args.source / n) for n in
                          ("final_theta.json", "per_state_final.csv",
                           "trajectory.csv", "protocol.json")},
        "phase9k_fitted_states_sha256":
            td._sha256(PHASE9K_DIR / "fitted_states.json"),
        "phase9n_code_sha": final["code_sha"],
        "targets": [list(t) for t in TARGETS],
        "fd_step": "Phase 9N primary: SavedState.step_sizes()/2 = 5e-5 "
                   "(F horizontal, log-variance), 5e-5*max(1,|w0|), "
                   "5e-5*max(1,|w|) at the endpoint; +h and -h once each",
        "z_start": "saved Phase 9N final Z mode for every call",
        "joint_mode": "Phase 9H defaults (grad_tol 1e-8, max_iter 200)",
        "reconstruction_rtol": td.RECONSTRUCTION_RTOL,
        "optimization_steps": 0, "em_executions": 0,
    }
    pilot._write_json(args.out / "protocol.json", protocol)

    datasets = {r.label: pilot.build_dataset(pc.PROTOCOL, r)
                for r in pc.PROTOCOL.replicates
                if r.label in {t[0] for t in TARGETS}}
    rows, summaries = [], []
    for rep, k, iters in TARGETS:
        e = entries[(rep, k)]
        ds = datasets[rep]
        state = build_state(e, ds.X, ds.Y)
        expected = k * state.F0.shape[0] - k * (k - 1) // 2
        base = evaluate_structured(state, np.zeros(state.n_coords))
        ok = (base["status"] == lk.STATUS_OK and base["log_observed_finite"]
              and math.isclose(base["log_observed"], e["final_ell"],
                               rel_tol=td.RECONSTRUCTION_RTOL)
              and state.n_horizontal == expected)
        if not ok:
            blocker(f"{rep} K={k} endpoint: {base}, n_horizontal "
                    f"{state.n_horizontal} vs {expected}")
        state_rows = classify_state(state)
        rows.extend(state_rows)
        summary = {"replicate": rep, "k": k, "phase9n_iterations": iters,
                   "phase9n_status": e["status"],
                   "endpoint_status": base["status"],
                   "endpoint_ell": base["log_observed"],
                   "phase9n_final_ell": e["final_ell"],
                   "endpoint_abs_diff": abs(base["log_observed"]
                                            - e["final_ell"]),
                   "horizontal_dimension": state.n_horizontal,
                   **summarize_state(state_rows)}
        summaries.append(summary)
        print(f"{rep} K={k} coords={summary['coordinates']} failing "
              f"coords={summary['failing_coordinates']} sides="
              f"{summary['failing_sides']} "
              f"{[(f['label'], f['side'], f['mechanism']) for f in summary['failures']]}",
              flush=True)

    pilot._write_csv(args.out / "perturbation_classification.csv", rows)
    flat = [{k: v for k, v in s.items() if k != "failures"}
            | {"failures": " | ".join(f"{f['label']} {f['side']} "
                                      f"{f['mechanism']}"
                                      for f in s["failures"])}
            for s in summaries]
    pilot._write_csv(args.out / "state_failure_summary.csv", flat)
    decision = decide(summaries)
    pilot._write_json(args.out / "state_failure_summary.json", {
        "classifier_version": CLASSIFIER_VERSION, "states": summaries,
        "mechanism_counts": {m: sum(r["mechanism"] == m for r in rows)
                             for m in (MECH_OK, MECH_NOT_STATIONARY,
                                       MECH_HESSIAN, MECH_EXCEPTION,
                                       MECH_NONFINITE, MECH_OTHER)},
        "phase9n_one_component_reproduced": all(
            s["failing_coordinates"] == 1 for s in summaries),
        "DECISION": decision, "optimization_steps": 0, "em_executions": 0})
    print(f"DECISION: {decision}")
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
