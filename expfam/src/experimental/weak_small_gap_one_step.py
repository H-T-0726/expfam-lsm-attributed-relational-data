"""Phase 9Q (Issue #106): Phase-9M one-step local-stability diagnostic on the
five smallest Phase 9P weak-w C_Lap gaps.

A thin wrapper: the reconstruction, quotient chart, finite differences
(h = 1e-4 and primary h/2), rotation diagnostic, one frozen step and the
pairwise labels are theta_stationarity_diagnostic (Phase 9M) unchanged,
pointed at the committed Phase 9P weak_w artifacts and restricted to the
10 states of the five fixed pairs. X, Y are regenerated from the weak_w
protocol (run_w_sensitivity.protocol_for("weak_w")). No EM, refit, family
search or multi-step optimization. Lineage E.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import run_family_selection_pilot as pilot                         # noqa: E402
import run_w_sensitivity as ws                                     # noqa: E402
import theta_stationarity_diagnostic as td                         # noqa: E402

DIAGNOSTIC_VERSION = "weak-small-gap-one-step-v1"
CONDITION = "weak_w"
SOURCE_DIR = (ws.REPO_ROOT
              / "expfam/results/relational_w_sensitivity/phase9p_20260928"
              / CONDITION)
# (replicate, best K, second K, original gap), fixed in the Phase 9P report
PAIRS = (("rep10", 2, 3, 1.5360794014), ("rep11", 3, 2, 4.5371893571),
         ("rep07", 2, 3, 5.2639028342), ("rep04", 2, 3, 7.0229266121),
         ("rep09", 3, 2, 10.7023345301))
STATES = tuple((rep, k) for rep, b, s, _ in PAIRS for k in (b, s))


def check_pairs(paired: dict[str, Any]) -> None:
    by = {d["replicate"]: d for d in paired["datasets"]}
    for rep, best, second, gap in PAIRS:
        d = by[rep]
        if (d["lap_best_k"], d["lap_second_k"]) != (best, second) \
                or round(d["lap_gap"], 10) != gap:
            raise SystemExit(f"{rep}: frozen pair differs from Phase 9P "
                             f"weak_w: systemic blocker")


def decide(pairs: Sequence[dict[str, Any]]) -> str:
    orderings = [p["ordering"] for p in pairs]
    if "unavailable" in orderings:
        return "WEAK_SMALL_GAP_ONE_STEP_INCONCLUSIVE"
    if "flipped" in orderings or "tie" in orderings:
        return "WEAK_SMALL_GAP_ONE_STEP_CAN_CHANGE"
    return "WEAK_SMALL_GAP_ONE_STEP_STABLE"


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
    protocol = ws.protocol_for(CONDITION)
    if source_protocol["w"] != protocol.w or states_json["condition"] \
            != CONDITION:
        raise SystemExit("source is not the weak_w condition: blocker")
    check_pairs(paired)
    entries = {(e["replicate"], int(e["k"])): e
               for e in states_json["entries"]}
    if not all(s in entries for s in STATES):
        raise SystemExit("saved states missing for the frozen scope")

    args.out.mkdir(parents=True)
    pilot._write_json(args.out / "protocol.json", {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "reused": "theta_stationarity_diagnostic (Phase 9M) "
                  f"{td.DIAGNOSTIC_VERSION}",
        "code_sha": pilot._git_sha(), "git_dirty": pilot._git_dirty(),
        "source_dir": str(SOURCE_DIR.relative_to(ws.REPO_ROOT)),
        "source_sha256": {n: td._sha256(SOURCE_DIR / n) for n in
                          ("fitted_states.json", "laplace_by_k.csv",
                           "paired_summary.json", "protocol.json")},
        "source_code_sha": states_json["code_sha"],
        "condition": CONDITION, "w": protocol.w,
        "pairs": [list(p) for p in PAIRS],
        "h_base": td.H_BASE, "primary_step": "h/2", "eps_dir": td.EPS_DIR,
        "step_factors": list(td.STEP_FACTORS),
        "reconstruction_rtol": td.RECONSTRUCTION_RTOL,
        "joint_mode": "Phase 9H defaults (grad_tol 1e-8, max_iter 200); "
                      "every evaluation starts at the base Z_hat0",
        "x_y": "regenerated with run_family_selection_pilot.build_dataset("
               "run_w_sensitivity.protocol_for('weak_w'), replicate)",
        "em_executions": 0, "refits": 0, "max_accepted_steps_per_state": 1,
    })

    reps = {rep for rep, *_ in PAIRS}
    datasets = {r.label: pilot.build_dataset(protocol, r)
                for r in protocol.replicates if r.label in reps}
    grads, steps = [], []
    for rep, k in STATES:
        ds = datasets[rep]
        state = td.SavedState(entries[(rep, k)], ds.X, ds.Y)
        row = lap_rows[(rep, k)]
        committed = {"laplace": float(row["laplace_log_observed"]),
                     "C_Lap": float(row["C_Lap"]), "d_K": int(row["d_K"]),
                     "N": float(row["N"])}
        grad, step = td.diagnose_state(state, committed)
        expected = k * state.F0.shape[0] - k * (k - 1) // 2
        grad["horizontal_dimension_ok"] = grad["horizontal_F_coords"] \
            == expected
        if not (grad["reconstruction_ok"] and grad["horizontal_dimension_ok"]):
            pilot._write_csv(args.out / "per_state_gradient.csv",
                             grads + [grad])
            raise SystemExit(f"reconstruction mismatch at {rep} K={k}: "
                             f"systemic blocker, stopped")
        grads.append(grad)
        steps.append(step)
        print(f"{rep} K={k} |g|={grad['grad_L2']:.3e} step={step['status']} "
              f"dell={step['delta_ell_one_step']:.4g}", flush=True)

    scoped = {**paired, "datasets": [d for d in paired["datasets"]
                                     if d["replicate"] in reps]}
    order = [rep for rep, *_ in PAIRS]
    pairs = sorted(td.pairwise(scoped, steps),
                   key=lambda p: order.index(p["replicate"]))
    for p in pairs:
        p["one_step_gap"] = p.pop("adjusted_gap")
    pilot._write_csv(args.out / "per_state_gradient.csv", grads)
    pilot._write_csv(args.out / "per_state_one_step.csv", steps)
    pilot._write_csv(args.out / "pair_summary.csv", pairs)
    orderings = [p["ordering"] for p in pairs]
    summary = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "states_requested": len(STATES), "states_reconstructed": sum(
            g["reconstruction_ok"] for g in grads),
        **td.availability_counts(grads),
        "one_step_status": {s: sum(x["status"] == s for x in steps)
                            for s in ("ACCEPTED", "NO_ACCEPTED_STEP",
                                      "UNAVAILABLE")},
        "ordering_counts": {o: orderings.count(o) for o in
                            ("retained", "flipped", "tie", "unavailable")},
        "flip_or_tie_replicates": [p["replicate"] for p in pairs
                                   if p["ordering"] in ("flipped", "tie")],
        "pairs": pairs,
        "DECISION": decide(pairs),
        "decision_scope": "one-step local diagnostic on 5 fixed weak_w pairs "
                          "only; not a full-optimization or MLE claim; "
                          "Phase 9P counts unchanged",
        "em_executions": 0, "refits": 0,
    }
    pilot._write_json(args.out / "pair_summary.json", summary)
    print(json.dumps({k: summary[k] for k in
                      ("states_reconstructed", "one_step_status",
                       "ordering_counts", "DECISION")}))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
