"""Phase 9N (Issue #100): bounded multi-step local-optimization pilot of the
implemented Candidate-B objective on the five smallest Phase 9K margins.

Objective (implemented, not the exact marginal):
    ell_Lap(theta, K) = laplace_k_criterion.laplace_log_observed(...)
Nothing here runs EM, an E-step, family selection or a refit, and no new
data is drawn: X, Y are regenerated from the frozen data seeds with the
Phase 9K generator. Scope is fixed to 5 pairs / 10 saved states.

Per state, starting from the Phase 9M base (saved Z_est -> joint mode):
  1. primary Phase 9M central-difference gradient (h = 5e-5; w0, w scaled by
     max(1, |current|)) in the quotient chart centred at the current F
  2. stop with CONVERGED_SCORE if ||g||_inf <= 1e-3
  3. u = g/||g||_2, directional curvature with eps_dir = 1e-3 (must be < 0)
  4. t* = ||g||_2/|curvature|, factors 1 ... 1/64, first finite increase
  5. accept: F, Gaussian variances, w0, w and the accepted Z mode become the
     new centre; the horizontal chart is rebuilt from the new F
at most 20 accepted steps (MAX_ITER_20 is a bounded stop, not convergence).
Every evaluation in an iteration starts the joint-mode solver at that
iteration's current accepted Z mode, with the Phase 9H tolerances. Non-OK
evaluations are never rescued.

C_Lap_multistep_diag(K) = C_Lap_Phase9K(K) - 2 * cumulative_delta_ell(K)
is a diagnostic only, not a new criterion. Lineage E.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import laplace_k_criterion as lk                                   # noqa: E402
import run_family_selection_pilot as pilot                         # noqa: E402
import run_lap_vs_cq_20 as pc                                      # noqa: E402
import theta_stationarity_diagnostic as td                         # noqa: E402

PILOT_VERSION = "multistep-local-v1"
PAIRS = (("rep10", 3, 2), ("rep04", 3, 2), ("rep19", 3, 2),
         ("rep09", 3, 4), ("rep18", 3, 4))            # (replicate, best, second)
MAX_ACCEPTED = 20
SCORE_TOL = 1e-3
EPS_DIR = td.EPS_DIR
STEP_FACTORS = td.STEP_FACTORS
SOURCE_DIR = td.SOURCE_DIR

CONVERGED = "CONVERGED_SCORE"
MAX_ITER = "MAX_ITER_20"
NO_ACCEPTED = "NO_ACCEPTED_STEP"
CURVATURE_NONNEG = "CURVATURE_NONNEGATIVE"
DERIVATIVE_UNAVAILABLE = "DERIVATIVE_UNAVAILABLE"
CHART_RANK_FAILURE = "CHART_RANK_FAILURE"
OBJECTIVE_UNAVAILABLE = "OBJECTIVE_UNAVAILABLE"
NORMAL = (CONVERGED, MAX_ITER)


class AscentState(Protocol):
    n_coords: int

    def chart_ok(self) -> bool: ...
    def steps(self) -> np.ndarray: ...
    def evaluate(self, theta: np.ndarray) -> tuple[float, Any]: ...
    def accept(self, theta: np.ndarray, payload: Any) -> None: ...
    def blocks(self, g: np.ndarray) -> dict[str, float]: ...
    def mode_info(self, payload: Any) -> dict[str, float]: ...


# --------------------------------------------------------------------------
# the recentred-chart state
# --------------------------------------------------------------------------

class RecenteredState:
    """A td.SavedState whose chart centre moves with every accepted step."""

    def __init__(self, saved: td.SavedState, base: lk.LaplaceResult):
        self.s = saved
        d = saved.F0.shape[0]
        self.expected_horizontal = saved.k * d - saved.k * (saved.k - 1) // 2
        self.s.Z_hat0 = base.Z_hat

    @property
    def n_coords(self) -> int:
        return self.s.n_coords

    def chart_ok(self) -> bool:
        return self.s.n_horizontal == self.expected_horizontal

    def steps(self) -> np.ndarray:
        return self.s.step_sizes() / 2           # Phase 9M primary (h/2)

    def evaluate(self, theta: np.ndarray) -> tuple[float, Any]:
        self.s.set_theta(theta)
        try:
            res = lk.laplace_log_observed(self.s.model, self.s.X, self.s.Y,
                                          self.s.Z_hat0)
        except (FloatingPointError, np.linalg.LinAlgError):
            return float("nan"), None
        return (res.log_observed if res.ok else float("nan")), res

    def accept(self, theta: np.ndarray, payload: lk.LaplaceResult) -> None:
        self.s.set_theta(theta)
        p = self.s.model.params
        self.s.F0 = np.array(p["F"], float).copy()
        self.s.var0 = np.diag(p["sigma"]).astype(float).copy()
        self.s.w0_0, self.s.w_0 = float(p["w0"]), float(p["w"])
        self.s.B_V, self.s.B_H = td.split_basis(self.s.F0)
        self.s.Z_hat0 = payload.Z_hat

    def blocks(self, g: np.ndarray) -> dict[str, float]:
        nh, ng = self.s.n_horizontal, len(self.s.gauss)
        return {"gradF_L2": float(np.linalg.norm(g[:nh])),
                "grad_logvar_L2": float(np.linalg.norm(g[nh:nh + ng])),
                "abs_grad_w0": float(abs(g[nh + ng])),
                "abs_grad_w": float(abs(g[nh + ng + 1]))}

    def mode_info(self, payload: lk.LaplaceResult) -> dict[str, float]:
        return {"zmode_grad_inf": float(payload.grad_inf),
                "min_H_eigenvalue": float(payload.min_eigenvalue_H)}

    def theta_record(self) -> dict[str, Any]:
        s = self.s
        return {"F": s.F0.tolist(), "sigma_diag": s.var0.tolist(),
                "w0": s.w0_0, "w": s.w_0, "var_z": s.var_z,
                "Z_mode": np.asarray(s.Z_hat0).tolist()}


# --------------------------------------------------------------------------
# frozen multi-step ascent
# --------------------------------------------------------------------------

def ascend(state: AscentState, ell0: float,
           max_accepted: int = MAX_ACCEPTED) -> dict[str, Any]:
    """Repeated frozen score-direction steps; returns status and trajectory.

    The gradient is evaluated at every centre, including the endpoint, so a
    state whose score falls under the tolerance is CONVERGED_SCORE even on
    the last allowed step; otherwise 20 accepted steps give MAX_ITER_20.
    """

    ell, accepted, trajectory = ell0, 0, []
    g = np.full(state.n_coords, np.nan)

    def finish(status: str, reason: str = "") -> dict[str, Any]:
        finite = np.all(np.isfinite(g))
        return {"status": status, "reason": reason, "iterations": accepted,
                "final_ell": ell, "cumulative_delta_ell": ell - ell0,
                "final_grad_L2": float(np.linalg.norm(g)) if finite
                else float("nan"),
                "final_grad_inf": float(np.max(np.abs(g))) if finite
                else float("nan"),
                "trajectory": trajectory}

    while True:
        if not state.chart_ok():
            return finish(CHART_RANK_FAILURE, "horizontal dimension changed")
        f = lambda t: state.evaluate(t)[0]                     # noqa: E731
        g, unavailable = td.central_gradient(f, state.n_coords, state.steps())
        if unavailable:
            return finish(DERIVATIVE_UNAVAILABLE,
                          f"{unavailable} gradient component(s) unavailable")
        g_inf = float(np.max(np.abs(g)))
        if g_inf <= SCORE_TOL:
            return finish(CONVERGED)
        if accepted >= max_accepted:
            return finish(MAX_ITER)
        norm = float(np.linalg.norm(g))
        u = g / norm
        plus, minus = f(EPS_DIR * u), f(-EPS_DIR * u)
        if not (math.isfinite(plus) and math.isfinite(minus)):
            return finish(DERIVATIVE_UNAVAILABLE,
                          "curvature evaluation unavailable")
        curvature = (plus - 2 * ell + minus) / EPS_DIR ** 2
        if curvature >= 0:
            return finish(CURVATURE_NONNEG)
        t_star = -norm / curvature
        for factor in STEP_FACTORS:
            theta = factor * t_star * u
            value, payload = state.evaluate(theta)
            if math.isfinite(value) and value > ell:
                break
        else:
            return finish(NO_ACCEPTED, "no factor gave a finite increase")
        state.accept(theta, payload)
        accepted += 1
        row = {"iteration": accepted, "ell_Lap": value,
               "cumulative_delta_ell": value - ell0,
               "grad_L2": norm, "grad_inf": g_inf, **state.blocks(g),
               "curvature": curvature, "t_star": t_star,
               "accepted_factor": factor, "step_delta_ell": value - ell,
               **state.mode_info(payload)}
        trajectory.append(row)
        ell = value


# --------------------------------------------------------------------------
# pairwise diagnostic
# --------------------------------------------------------------------------

def _ordering(gap: float) -> str:
    return "retained" if gap > 0 else "flipped" if gap < 0 else "tie"


def cumulative_at(result: dict[str, Any], r: int) -> float:
    """Cumulative delta after synchronized round r (carry forward if normal)."""
    traj = result["trajectory"]
    if r <= len(traj):
        return 0.0 if r == 0 else traj[r - 1]["cumulative_delta_ell"]
    return (result["cumulative_delta_ell"] if result["status"] in NORMAL
            else float("nan"))


def pair_rows(c_lap: dict[tuple[str, int], float],
              results: dict[tuple[str, int], dict[str, Any]],
              pairs: Sequence[tuple[str, int, int]] = PAIRS
              ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary, trajectory = [], []
    for rep, best, second in pairs:
        rb, rs = results[(rep, best)], results[(rep, second)]
        cb, cs = c_lap[(rep, best)], c_lap[(rep, second)]
        available = rb["status"] in NORMAL and rs["status"] in NORMAL
        gaps = []
        for r in range(MAX_ACCEPTED + 1):
            gap = ((cs - 2 * cumulative_at(rs, r))
                   - (cb - 2 * cumulative_at(rb, r)))
            gaps.append(gap)
            trajectory.append({"replicate": rep, "round": r, "best_k": best,
                               "second_k": second, "diag_gap": gap})
        final = ((cs - 2 * rs["cumulative_delta_ell"])
                 - (cb - 2 * rb["cumulative_delta_ell"]))
        summary.append({
            "replicate": rep, "best_k": best, "second_k": second,
            "original_gap": cs - cb,
            "best_status": rb["status"], "second_status": rs["status"],
            "best_iterations": rb["iterations"],
            "second_iterations": rs["iterations"],
            "best_cumulative_delta_ell": rb["cumulative_delta_ell"],
            "second_cumulative_delta_ell": rs["cumulative_delta_ell"],
            "final_diag_gap": final if available else float("nan"),
            "ordering": _ordering(final) if available else "unavailable",
            "intermediate_flip_rounds": ";".join(
                str(r) for r, g in enumerate(gaps) if g < 0),
            "min_round_gap": float(np.nanmin(gaps)),
        })
    return summary, trajectory


def decide(summary: Sequence[dict[str, Any]]) -> str:
    orderings = [p["ordering"] for p in summary]
    if "unavailable" in orderings:
        return "MULTISTEP_PILOT_INCONCLUSIVE"
    if "flipped" in orderings or "tie" in orderings:
        return "MULTISTEP_PILOT_ORDERING_CAN_CHANGE"
    return "MULTISTEP_PILOT_ORDERING_STABLE"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; a recorded run is never "
                         f"overwritten")
    states_json = json.loads((args.source / "fitted_states.json")
                             .read_text("utf-8"))
    lap_rows = {(r["replicate"], int(r["k"])): r
                for r in td._read_csv(args.source / "laplace_by_k.csv")}
    paired = {d["replicate"]: d for d in json.loads(
        (args.source / "paired_summary.json").read_text("utf-8"))["datasets"]}
    for rep, best, second in PAIRS:              # frozen scope vs Phase 9K
        d = paired[rep]
        if (d["lap_best_k"], d["lap_second_k"]) != (best, second):
            raise SystemExit(f"{rep}: frozen pair differs from Phase 9K")
    wanted = {(rep, k) for rep, b, s in PAIRS for k in (b, s)}
    entries = [e for e in states_json["entries"]
               if (e["replicate"], int(e["k"])) in wanted]
    if len(entries) != len(wanted):
        raise SystemExit("saved states missing for the frozen scope")

    args.out.mkdir(parents=True)
    protocol = {
        "pilot_version": PILOT_VERSION,
        "code_sha": pilot._git_sha(), "git_dirty": pilot._git_dirty(),
        "source_dir": str(args.source),
        "source_sha256": {n: td._sha256(args.source / n) for n in
                          ("fitted_states.json", "laplace_by_k.csv",
                           "paired_summary.json")},
        "source_code_sha": states_json["code_sha"],
        "pairs": [list(p) for p in PAIRS],
        "fd_step": "Phase 9M primary: 5e-5 (F horizontal, log-variance); "
                   "5e-5*max(1,|w0|), 5e-5*max(1,|w|) at the current centre",
        "eps_dir": EPS_DIR, "step_factors": list(STEP_FACTORS),
        "score_tol": SCORE_TOL, "max_accepted": MAX_ACCEPTED,
        "chart": "horizontal complement of {F A : A^T=-A}, rebuilt from the "
                 "current F after every accepted step",
        "joint_mode": "Phase 9H defaults (grad_tol 1e-8, max_iter 200); "
                      "every evaluation starts at the current accepted mode",
        "reconstruction_rtol": td.RECONSTRUCTION_RTOL,
        "em_executions": 0,
    }
    pilot._write_json(args.out / "protocol.json", protocol)

    reps = {rep for rep, _, _ in PAIRS}
    datasets = {r.label: pilot.build_dataset(pc.PROTOCOL, r)
                for r in pc.PROTOCOL.replicates if r.label in reps}
    finals, trajectory, final_theta, results, c_lap = [], [], [], {}, {}
    for entry in entries:
        rep, k = entry["replicate"], int(entry["k"])
        ds = datasets[rep]
        saved = td.SavedState(entry, ds.X, ds.Y)
        row = lap_rows[(rep, k)]
        committed_ell, committed_c = (float(row["laplace_log_observed"]),
                                      float(row["C_Lap"]))
        base = saved.base()
        ell_rec = base.log_observed if base.ok else float("nan")
        c_rec = -2.0 * ell_rec + int(row["d_K"]) * math.log(float(row["N"]))
        if not (base.ok and math.isclose(ell_rec, committed_ell,
                                         rel_tol=td.RECONSTRUCTION_RTOL)
                and math.isclose(c_rec, committed_c,
                                 rel_tol=td.RECONSTRUCTION_RTOL)):
            raise SystemExit(f"reconstruction mismatch at {rep} K={k}: "
                             f"systemic blocker, stopped")
        c_lap[(rep, k)] = committed_c
        state = RecenteredState(saved, base)
        if not math.isfinite(ell_rec):
            result = {"status": OBJECTIVE_UNAVAILABLE, "reason": "base",
                      "iterations": 0, "final_ell": float("nan"),
                      "cumulative_delta_ell": float("nan"),
                      "final_grad_L2": float("nan"),
                      "final_grad_inf": float("nan"), "trajectory": []}
        else:
            result = ascend(state, ell_rec)
        results[(rep, k)] = result
        for t in result["trajectory"]:
            trajectory.append({"replicate": rep, "k": k, **t})
        first = result["trajectory"][0] if result["trajectory"] else {}
        finals.append({
            "replicate": rep, "k": k, "status": result["status"],
            "reason": result["reason"], "iterations": result["iterations"],
            "ell_committed": committed_ell, "ell_reconstructed": ell_rec,
            "final_ell": result["final_ell"],
            "cumulative_delta_ell": result["cumulative_delta_ell"],
            "initial_grad_L2": first.get("grad_L2", result["final_grad_L2"]),
            "initial_grad_inf": first.get("grad_inf",
                                          result["final_grad_inf"]),
            "final_grad_L2": result["final_grad_L2"],
            "final_grad_inf": result["final_grad_inf"],
            "C_Lap_committed": committed_c,
            "C_Lap_multistep_diag": committed_c
            - 2 * result["cumulative_delta_ell"]})
        final_theta.append({
            "replicate": rep, "k": k, "data_seed": entry["data_seed"],
            "search_seed": entry["search_seed"],
            "refit_seed": entry["refit_seed"],
            "selected_assignment": entry["selected_assignment"],
            **state.theta_record(),
            "final_ell": result["final_ell"],
            "cumulative_delta_ell": result["cumulative_delta_ell"],
            "final_grad_L2": result["final_grad_L2"],
            "final_grad_inf": result["final_grad_inf"],
            "status": result["status"], "iterations": result["iterations"]})
        print(f"{rep} K={k} {result['status']} it={result['iterations']} "
              f"dell={result['cumulative_delta_ell']:.4g} "
              f"|g|inf {finals[-1]['initial_grad_inf']:.3g}->"
              f"{result['final_grad_inf']:.3g}", flush=True)

    pairs, pair_traj = pair_rows(c_lap, results)
    pilot._write_csv(args.out / "trajectory.csv", trajectory)
    pilot._write_csv(args.out / "per_state_final.csv", finals)
    pilot._write_csv(args.out / "pair_summary.csv", pairs)
    pilot._write_csv(args.out / "pair_trajectory.csv", pair_traj)
    pilot._write_json(args.out / "final_theta.json", {
        "pilot_version": PILOT_VERSION, "code_sha": protocol["code_sha"],
        "source_code_sha": states_json["code_sha"],
        "source_fitted_states_sha256":
            protocol["source_sha256"]["fitted_states.json"],
        "float_format": "json float (repr), lossless", "entries": final_theta})
    statuses = [f["status"] for f in finals]
    orderings = [p["ordering"] for p in pairs]
    summary = {
        "pilot_version": PILOT_VERSION,
        "states_requested": len(wanted), "states_reconstructed": len(finals),
        "status_counts": {s: statuses.count(s) for s in
                          (CONVERGED, MAX_ITER, NO_ACCEPTED, CURVATURE_NONNEG,
                           DERIVATIVE_UNAVAILABLE, CHART_RANK_FAILURE,
                           OBJECTIVE_UNAVAILABLE)},
        "ordering_counts": {o: orderings.count(o) for o in
                            ("retained", "flipped", "tie", "unavailable")},
        "flipped_replicates": [p["replicate"] for p in pairs
                               if p["ordering"] == "flipped"],
        "intermediate_flip_replicates": [p["replicate"] for p in pairs
                                         if p["intermediate_flip_rounds"]],
        "DECISION": decide(pairs),
        "decision_scope": "bounded (<=20 accepted steps) local diagnostic on "
                          "5 fixed pairs; MAX_ITER_20 is not convergence; not "
                          "a new criterion or an MLE claim",
        "em_executions": 0,
    }
    pilot._write_json(args.out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in
                      ("status_counts", "ordering_counts", "DECISION")}))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
