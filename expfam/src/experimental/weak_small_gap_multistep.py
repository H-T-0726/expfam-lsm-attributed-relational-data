"""Phase 9R (Issue #108): Phase-9N bounded multi-step local diagnostic on
the five smallest Phase 9P weak-w C_Lap gaps (the Phase 9Q pairs).

A thin wrapper around multistep_local_pilot (Phase 9N), unchanged: the
recentred quotient chart, primary h = 5e-5 central differences, eps_dir,
backtracking factors, score tolerance, statuses and <= 20 accepted steps.
It starts from the original Phase 9P weak_w saved states, regenerates X, Y
from the weak_w protocol, and adds only:

  - a Phase 9Q first-step gate: every state first runs the procedure to
    exactly one accepted step; only if all 10 first steps reproduce the
    committed Phase 9Q one-step values do the states continue (up to 19
    further accepted steps). The gradient at the step-1 point is then
    evaluated again deterministically at the start of the continuation;
    its equality with the first pass is recorded.
  - structured failure logging: each Candidate-B evaluation the procedure
    already makes is recorded (status, grad_inf, iterations, Hessian,
    logdet, notes or exception); nothing is re-evaluated or rescued.

C_Lap_multistep_diag(K) = C_Lap_Phase9P_weak(K) - 2 * cumulative_delta_ell
is a diagnostic only. No EM, refit or family search. Lineage E.
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
import multistep_local_pilot as mp                                 # noqa: E402
import run_family_selection_pilot as pilot                         # noqa: E402
import run_w_sensitivity as ws                                     # noqa: E402
import theta_stationarity_diagnostic as td                         # noqa: E402
import weak_small_gap_one_step as wg                               # noqa: E402

DIAGNOSTIC_VERSION = "weak-small-gap-multistep-v1"
SOURCE_DIR = wg.SOURCE_DIR
PHASE9Q_DIR = ws.REPO_ROOT / "expfam/results/weak_small_gap_one_step/phase9q_20260928"
PAIRS = tuple((rep, best, second) for rep, best, second, _ in wg.PAIRS)
STATES = wg.STATES
FIRST_STEP_RTOL = 1e-9


class LoggingState(mp.RecenteredState):
    """RecenteredState that records every evaluation it is asked to make."""

    def __init__(self, saved: td.SavedState, base: lk.LaplaceResult):
        super().__init__(saved, base)
        self.log: list[dict[str, Any]] = []

    def evaluate(self, theta: np.ndarray) -> tuple[float, Any]:
        self.s.set_theta(theta)
        entry: dict[str, Any] = {"theta": np.array(theta, float)}
        try:
            res = lk.laplace_log_observed(self.s.model, self.s.X, self.s.Y,
                                          self.s.Z_hat0)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            entry.update(status="EXCEPTION", value=float("nan"),
                         exception_class=type(exc).__name__,
                         exception_message=str(exc))
            self.log.append(entry)
            return float("nan"), None
        value = res.log_observed if res.ok else float("nan")
        entry.update(status=res.status, value=value, grad_inf=res.grad_inf,
                     iterations=res.iterations,
                     min_eigenvalue_H=res.min_eigenvalue_H,
                     logdet_sign=res.logdet_sign, logdet_H=res.logdet_H,
                     notes="; ".join(res.notes))
        self.log.append(entry)
        return value, res

    def accept(self, theta: np.ndarray, payload: lk.LaplaceResult) -> None:
        super().accept(theta, payload)
        self.log = []                      # keep the current iteration only

    def labels(self) -> list[tuple[str, str]]:
        s = self.s
        return ([("F_horizontal", f"F_horizontal[{j}]")
                 for j in range(s.n_horizontal)]
                + [("log_sigma", f"log_sigma[column={l}]") for l in s.gauss]
                + [("w0", "w0"), ("w", "w")])


def failure_rows(state: LoggingState, replicate: str, k: int,
                 iteration: int) -> list[dict[str, Any]]:
    """Non-finite evaluations already made in the failing iteration."""
    labels = state.labels()
    rows = []
    for e in state.log:
        if math.isfinite(e["value"]):
            continue
        nz = np.flatnonzero(e["theta"])
        if len(nz) == 1:
            i = int(nz[0])
            block, label = labels[i]
            side = "+" if e["theta"][i] > 0 else "-"
        else:
            i, block, label, side = -1, "direction", "curvature/step along u", ""
        rows.append({"replicate": replicate, "k": k,
                     "failing_iteration": iteration, "coordinate_index": i,
                     "block": block, "label": label, "side": side,
                     **{key: e.get(key, "") for key in
                        ("status", "grad_inf", "iterations",
                         "min_eigenvalue_H", "logdet_sign", "logdet_H",
                         "notes", "exception_class", "exception_message")}})
    return rows


def first_step_check(row: dict[str, Any], q: dict[str, str]) -> dict[str, Any]:
    """Compare a first accepted step with the committed Phase 9Q values."""
    close = lambda a, b: math.isclose(float(a), float(b),          # noqa: E731
                                      rel_tol=FIRST_STEP_RTOL)
    checks = {
        "status": q["status"] == "ACCEPTED" and row is not None,
        "accepted_factor": row is not None
        and float(q["accepted_factor"]) == row["accepted_factor"],
        "delta_ell": row is not None
        and close(q["delta_ell_one_step"], row["step_delta_ell"]),
        "curvature": row is not None and close(q["curvature"],
                                               row["curvature"]),
        "t_star": row is not None and close(q["t_star"], row["t_star"]),
    }
    return {**checks, "pass": all(checks.values())}


def combine(first: dict[str, Any], rest: dict[str, Any] | None,
            ell0: float) -> dict[str, Any]:
    """Join the first-step pass and the continuation into one result."""
    if rest is None:                        # stopped within the first pass
        return first
    traj = list(first["trajectory"])
    offset = len(first["trajectory"])
    for r in rest["trajectory"]:
        traj.append({**r, "iteration": r["iteration"] + offset,
                     "cumulative_delta_ell": r["ell_Lap"] - ell0})
    return {**rest, "iterations": len(traj), "trajectory": traj,
            "cumulative_delta_ell": rest["final_ell"] - ell0}


def decide(pairs: Sequence[dict[str, Any]]) -> str:
    orderings = [p["ordering"] for p in pairs]
    if "unavailable" in orderings:
        return "WEAK_SMALL_GAP_MULTISTEP_INCONCLUSIVE"
    if "flipped" in orderings or "tie" in orderings:
        return "WEAK_SMALL_GAP_MULTISTEP_CAN_CHANGE"
    return "WEAK_SMALL_GAP_MULTISTEP_STABLE"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; a recorded run is never "
                         f"overwritten")
    states_json = json.loads((SOURCE_DIR / "fitted_states.json")
                             .read_text("utf-8"))
    lap_rows = {(r["replicate"], int(r["k"])): r
                for r in td._read_csv(SOURCE_DIR / "laplace_by_k.csv")}
    paired = json.loads((SOURCE_DIR / "paired_summary.json")
                        .read_text("utf-8"))
    source_protocol = json.loads((SOURCE_DIR / "protocol.json")
                                 .read_text("utf-8"))
    protocol = ws.protocol_for(wg.CONDITION)
    if source_protocol["w"] != protocol.w \
            or states_json["condition"] != wg.CONDITION:
        raise SystemExit("source is not the weak_w condition: blocker")
    wg.check_pairs(paired)
    q_rows = {(r["replicate"], int(r["k"])): r for r in
              td._read_csv(PHASE9Q_DIR / "per_state_one_step.csv")}
    entries = {(e["replicate"], int(e["k"])): e
               for e in states_json["entries"]}

    args.out.mkdir(parents=True)
    protocol_json = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "reused": "multistep_local_pilot (Phase 9N) "
                  f"{mp.PILOT_VERSION}; theta_stationarity_diagnostic "
                  f"{td.DIAGNOSTIC_VERSION}",
        "code_sha": pilot._git_sha(), "git_dirty": pilot._git_dirty(),
        "source_dir": str(SOURCE_DIR.relative_to(ws.REPO_ROOT)),
        "source_sha256": {n: td._sha256(SOURCE_DIR / n) for n in
                          ("fitted_states.json", "laplace_by_k.csv",
                           "paired_summary.json", "protocol.json")},
        "phase9q_one_step_sha256":
            td._sha256(PHASE9Q_DIR / "per_state_one_step.csv"),
        "source_code_sha": states_json["code_sha"],
        "pairs": [list(p) for p in wg.PAIRS],
        "fd_step": "Phase 9N primary: 5e-5 (F horizontal, log-variance); "
                   "5e-5*max(1,|w0|), 5e-5*max(1,|w|) at the current centre",
        "eps_dir": mp.EPS_DIR, "step_factors": list(mp.STEP_FACTORS),
        "score_tol": mp.SCORE_TOL, "max_accepted": mp.MAX_ACCEPTED,
        "first_step_gate": f"all 10 first steps vs Phase 9Q, rel "
                           f"{FIRST_STEP_RTOL}, factor exact",
        "reconstruction_rtol": td.RECONSTRUCTION_RTOL,
        "em_executions": 0, "refits": 0,
    }
    pilot._write_json(args.out / "protocol.json", protocol_json)

    reps = {rep for rep, _, _ in PAIRS}
    datasets = {r.label: pilot.build_dataset(protocol, r)
                for r in protocol.replicates if r.label in reps}
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for rep, k in STATES:                              # reconstruction gate
        ds = datasets[rep]
        saved = td.SavedState(entries[(rep, k)], ds.X, ds.Y)
        row = lap_rows[(rep, k)]
        base = saved.base()
        ell0 = base.log_observed if base.ok else float("nan")
        c_rec = -2.0 * ell0 + int(row["d_K"]) * math.log(float(row["N"]))
        if not (base.ok and math.isclose(
                ell0, float(row["laplace_log_observed"]),
                rel_tol=td.RECONSTRUCTION_RTOL)
                and math.isclose(c_rec, float(row["C_Lap"]),
                                 rel_tol=td.RECONSTRUCTION_RTOL)):
            raise SystemExit(f"reconstruction mismatch at {rep} K={k}: "
                             f"systemic blocker, stopped")
        runs[(rep, k)] = {"state": LoggingState(saved, base), "ell0": ell0,
                          "C_Lap": float(row["C_Lap"]),
                          "ell_committed": float(row["laplace_log_observed"])}

    gate_rows, failures = [], []
    for (rep, k), run in runs.items():                 # pass A: first step
        run["state"].log = []
        first = mp.ascend(run["state"], run["ell0"], max_accepted=1)
        run["first"] = first
        row = first["trajectory"][0] if first["trajectory"] else None
        check = first_step_check(row, q_rows[(rep, k)])
        gate_rows.append({"replicate": rep, "k": k, **check,
                          "phase9r_step_delta_ell":
                          row["step_delta_ell"] if row else float("nan"),
                          "phase9q_delta_ell": float(
                              q_rows[(rep, k)]["delta_ell_one_step"]),
                          "first_pass_status": first["status"]})
        if first["status"] not in (mp.MAX_ITER, mp.CONVERGED):
            failures += failure_rows(run["state"], rep, k,
                                     first["iterations"] + 1)
        print(f"{rep} K={k} first step: {check['pass']} "
              f"dell={gate_rows[-1]['phase9r_step_delta_ell']:.6g}",
              flush=True)
    pilot._write_csv(args.out / "first_step_gate.csv", gate_rows)
    if not all(g["pass"] for g in gate_rows):
        pilot._write_json(args.out / "summary.json", {
            "DECISION": None, "first_step_gate": "BLOCKED",
            "note": "Phase 9Q first-step reproduction failed; iterations "
                    "beyond 1 were not run"})
        raise SystemExit("Phase 9Q first-step reproduction failed: blocked")

    results, finals, trajectory, final_theta, c_lap = {}, [], [], [], {}
    for (rep, k), run in runs.items():                 # pass B: continue
        first, state = run["first"], run["state"]
        rest = None
        repeat_check = ""
        if first["status"] == mp.MAX_ITER:             # 1 accepted, go on
            state.log = []
            rest = mp.ascend(state, first["final_ell"],
                             max_accepted=mp.MAX_ACCEPTED - 1)
            second = rest["trajectory"][0]["grad_inf"] \
                if rest["trajectory"] else rest["final_grad_inf"]
            repeat_check = second == first["final_grad_inf"]
            if rest["status"] not in (mp.MAX_ITER, mp.CONVERGED):
                failures += failure_rows(state, rep, k,
                                         1 + rest["iterations"] + 1)
        result = combine(first, rest, run["ell0"])
        results[(rep, k)] = result
        c_lap[(rep, k)] = run["C_Lap"]
        for t in result["trajectory"]:
            trajectory.append({"replicate": rep, "k": k, **t})
        e = entries[(rep, k)]
        finals.append({
            "replicate": rep, "k": k, "status": result["status"],
            "reason": result["reason"], "iterations": result["iterations"],
            "ell_committed": run["ell_committed"],
            "ell_reconstructed": run["ell0"],
            "final_ell": result["final_ell"],
            "cumulative_delta_ell": result["cumulative_delta_ell"],
            "initial_grad_L2": result["trajectory"][0]["grad_L2"],
            "initial_grad_inf": result["trajectory"][0]["grad_inf"],
            "final_grad_L2": result["final_grad_L2"],
            "final_grad_inf": result["final_grad_inf"],
            "step1_gradient_repeat_identical": repeat_check,
            "C_Lap_committed": run["C_Lap"],
            "C_Lap_multistep_diag": run["C_Lap"]
            - 2 * result["cumulative_delta_ell"]})
        final_theta.append({
            "replicate": rep, "k": k, "data_seed": e["data_seed"],
            "search_seed": e["search_seed"], "refit_seed": e["refit_seed"],
            "selected_assignment": e["selected_assignment"],
            **state.theta_record(),
            "final_ell": result["final_ell"],
            "cumulative_delta_ell": result["cumulative_delta_ell"],
            "final_grad_L2": result["final_grad_L2"],
            "final_grad_inf": result["final_grad_inf"],
            "status": result["status"], "iterations": result["iterations"]})
        print(f"{rep} K={k} {result['status']} it={result['iterations']} "
              f"dell={result['cumulative_delta_ell']:.4g} |g|inf "
              f"{finals[-1]['initial_grad_inf']:.3g}->"
              f"{result['final_grad_inf']:.3g}", flush=True)

    pairs, pair_traj = mp.pair_rows(c_lap, results, PAIRS)
    for p in pairs:
        gaps = [t["diag_gap"] for t in pair_traj
                if t["replicate"] == p["replicate"]]
        p["intermediate_tie_rounds"] = ";".join(
            str(r) for r, g in enumerate(gaps) if g == 0)
        p["k3_gained_more"] = (
            (p["best_cumulative_delta_ell"] if p["best_k"] == 3
             else p["second_cumulative_delta_ell"])
            > (p["best_cumulative_delta_ell"] if p["best_k"] == 2
               else p["second_cumulative_delta_ell"]))
    pilot._write_csv(args.out / "trajectory.csv", trajectory)
    pilot._write_csv(args.out / "per_state_final.csv", finals)
    pilot._write_csv(args.out / "pair_summary.csv", pairs)
    pilot._write_csv(args.out / "pair_trajectory.csv", pair_traj)
    if failures:
        pilot._write_csv(args.out / "failure_metadata.csv", failures)
        pilot._write_json(args.out / "failure_metadata.json", failures)
    pilot._write_json(args.out / "final_theta.json", {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "code_sha": protocol_json["code_sha"],
        "source_code_sha": states_json["code_sha"],
        "source_fitted_states_sha256":
            protocol_json["source_sha256"]["fitted_states.json"],
        "float_format": "json float (repr), lossless", "entries": final_theta})
    statuses = [f["status"] for f in finals]
    orderings = [p["ordering"] for p in pairs]
    summary = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "states_requested": len(STATES), "states_reconstructed": len(runs),
        "first_step_gate": "PASS",
        "status_counts": {s: statuses.count(s) for s in
                          (mp.CONVERGED, mp.MAX_ITER, mp.NO_ACCEPTED,
                           mp.CURVATURE_NONNEG, mp.DERIVATIVE_UNAVAILABLE,
                           mp.CHART_RANK_FAILURE, mp.OBJECTIVE_UNAVAILABLE)},
        "ordering_counts": {o: orderings.count(o) for o in
                            ("retained", "flipped", "tie", "unavailable")},
        "endpoint_flip_or_tie": [p["replicate"] for p in pairs
                                 if p["ordering"] in ("flipped", "tie")],
        "intermediate_flip": {p["replicate"]: p["intermediate_flip_rounds"]
                              for p in pairs
                              if p["intermediate_flip_rounds"]},
        "intermediate_tie": {p["replicate"]: p["intermediate_tie_rounds"]
                             for p in pairs if p["intermediate_tie_rounds"]},
        "failures_logged": len(failures),
        "pairs": pairs,
        "DECISION": decide(pairs),
        "decision_scope": "bounded (<=20 accepted steps) local diagnostic on "
                          "5 fixed weak_w pairs; MAX_ITER_20 is not "
                          "convergence; not a new criterion or MLE claim; "
                          "Phase 9P counts unchanged",
        "em_executions": 0, "refits": 0,
    }
    pilot._write_json(args.out / "pair_summary.json", summary)
    print(json.dumps({k: summary[k] for k in
                      ("status_counts", "ordering_counts",
                       "intermediate_flip", "DECISION")}))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
