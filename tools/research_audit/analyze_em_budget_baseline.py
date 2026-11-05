"""Read-only baseline analysis for the EM convergence-sensitivity design.

Everything here is computed from the ALREADY ARCHIVED clean true-K sweep
(``expfam/results/k_selection/clean_true_k_asymptotics_20260904/``).  It runs
no EM, produces no fit, and writes nothing into any artifact directory.  Its
only purpose is to supply the design document
``reports/k_selection/em_convergence_sensitivity_design_20260905.md`` with
numbers that are regenerable rather than typed in by hand.

Two things it deliberately does NOT do:

* It does not decide any protocol parameter.  The scope of the proposed
  experiment is fixed in the DRAFT protocol JSON, not here.
* It does not claim to measure convergence.  The 896-fit artifact carries no
  per-iteration quantity at all (see section "A" of the design document), so
  everything below is a PROXY -- runtime accounting and the shape of the
  score-versus-K curve at the single frozen budget num_iter=8.

Usage
-----
    python tools/research_audit/analyze_em_budget_baseline.py
    python tools/research_audit/analyze_em_budget_baseline.py --json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = (ROOT / "expfam" / "results" / "k_selection"
           / "clean_true_k_asymptotics_20260904")

CANDIDATE_K = (1, 2, 3, 4, 5, 6, 7)
N_GRID = (50, 75, 100, 150)
BASELINE_NUM_ITER = 8

# The proposed experiment's scope.  Kept here only so the runtime projection
# and the protocol JSON cannot silently disagree; the authority is the JSON.
PROPOSED_K_TRUE = 5
# Settled by human decision on 2026-09-05: the baseline's WHOLE n grid,
# so the scope is not a subset chosen after seeing which n under-selected.
PROPOSED_N = (50, 75, 100, 150)
PROPOSED_MAX_ITER = 40


def read_fits() -> list[dict[str, str]]:
    with (RUN_DIR / "fit_results.csv").open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def runtime_table(fits: list[dict[str, str]]) -> dict:
    """Mean seconds per fit at the frozen budget, by n and by candidate K."""

    cell: dict[tuple[int, int], list[float]] = {}
    by_n: dict[int, list[float]] = {}
    by_k: dict[int, list[float]] = {}
    by_start: dict[int, list[float]] = {}
    for row in fits:
        n, k = int(row["n"]), int(row["k_est"])
        t = float(row["runtime_s"])
        cell.setdefault((n, k), []).append(t)
        by_n.setdefault(n, []).append(t)
        by_k.setdefault(k, []).append(t)
        by_start.setdefault(int(row["start"]), []).append(t)
    return {
        "total_runtime_s": round(sum(float(r["runtime_s"]) for r in fits), 1),
        "mean_by_n": {n: round(float(np.mean(v)), 2) for n, v in sorted(by_n.items())},
        "mean_by_k_est": {k: round(float(np.mean(v)), 2)
                          for k, v in sorted(by_k.items())},
        "mean_by_start": {s: round(float(np.mean(v)), 2)
                          for s, v in sorted(by_start.items())},
        "mean_by_n_and_k": {f"{n}|{k}": round(float(np.mean(cell[(n, k)])), 2)
                            for n in N_GRID for k in CANDIDATE_K},
    }


def runtime_projection(fits: list[dict[str, str]]) -> dict:
    """Linear-in-iterations projection for the proposed scope.

    Linear in the iteration count.  It is an ESTIMATE, not a bound, because it
    ignores two effects that push in opposite directions: it counts the
    archive's fixed end-of-run Q/BIC/S3 cost as if it scaled with iterations
    (over-estimate), while the new run pays that cost once per checkpoint --
    six times per fit -- instead of once (under-estimate).  The operational
    guard is the 2x abort threshold, which is a SAFETY STOP and not a
    scientific inclusion criterion (finding M-02).
    """

    scoped = [r for r in fits
              if int(r["k_true"]) == PROPOSED_K_TRUE and int(r["n"]) in PROPOSED_N]
    base_s = sum(float(r["runtime_s"]) for r in scoped)
    out = {
        "scope_fits": len(scoped),
        "scope_runtime_s_at_baseline_budget": round(base_s, 1),
        "projection_hours": {},
    }
    for m in (16, 24, 32, 40, 50):
        out["projection_hours"][m] = round(base_s * m / BASELINE_NUM_ITER / 3600, 2)
    return out


def score_shape(fits: list[dict[str, str]]) -> dict:
    """Mean S1 held-out score and mean Q_strict per candidate K, K_TRUE=5.

    This is the shape the confounder argument is about: at the frozen budget
    both curves fall off for large K.  Whether that fall-off is the criterion
    penalising complexity or the optimiser failing to fit the larger model in
    8 iterations is exactly what the artifact cannot answer.
    """

    out: dict[str, dict] = {"s1_mean_by_k": {}, "q_strict_mean_by_k": {},
                            "argmax_k": {}}
    for n in N_GRID:
        sub = [r for r in fits if int(r["k_true"]) == PROPOSED_K_TRUE
               and int(r["n"]) == n]
        s1 = [float(np.mean([float(r["heldout_mean_log_score"]) for r in sub
                             if int(r["k_est"]) == k])) for k in CANDIDATE_K]
        qs = [float(np.mean([float(r["q_strict"]) for r in sub
                             if int(r["k_est"]) == k])) for k in CANDIDATE_K]
        out["s1_mean_by_k"][n] = [round(v, 5) for v in s1]
        out["q_strict_mean_by_k"][n] = [round(v, 1) for v in qs]
        out["argmax_k"][n] = {"S1": CANDIDATE_K[int(np.argmax(s1))],
                              "Q_strict": CANDIDATE_K[int(np.argmax(qs))]}
    return out


def free_parameters(fits: list[dict[str, str]]) -> dict[int, int]:
    """num_params per candidate K, straight from the artifact."""

    out: dict[int, int] = {}
    for row in fits:
        out[int(row["k_est"])] = int(row["num_params"])
    return dict(sorted(out.items()))


def convergence_evidence(fits: list[dict[str, str]]) -> dict:
    """What the artifact does and does not say about convergence."""

    columns = sorted(fits[0].keys())
    per_iteration = [c for c in columns
                     if "iter" in c or "trajectory" in c or "history" in c]
    return {
        "fit_columns": columns,
        "per_iteration_columns": per_iteration,
        "artifact_can_decide_convergence": bool(per_iteration),
        "nan_occurred_rows": sum(1 for r in fits if r["nan_occurred"] == "True"),
        "nan_count_positive_rows": sum(1 for r in fits if int(r["nan_count"]) > 0),
        "q_bic_failed_rows": sum(1 for r in fits if r["q_bic_failed"] == "True"),
    }


def build() -> dict:
    fits = read_fits()
    return {
        "source_run_dir": str(RUN_DIR.relative_to(ROOT)).replace("\\", "/"),
        "source_fits": len(fits),
        "baseline_num_iter": BASELINE_NUM_ITER,
        "free_parameters_by_k": free_parameters(fits),
        "convergence_evidence": convergence_evidence(fits),
        "runtime": runtime_table(fits),
        "runtime_projection": runtime_projection(fits),
        "score_shape_k_true_5": score_shape(fits),
    }


def render(report: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"source            : {report['source_run_dir']}")
    add(f"fits              : {report['source_fits']} at num_iter="
        f"{report['baseline_num_iter']}")
    add("")
    ce = report["convergence_evidence"]
    add("-- can the archived artifact decide convergence? --")
    add(f"  per-iteration columns present : {ce['per_iteration_columns'] or 'NONE'}")
    add(f"  verdict                       : "
        f"{'yes' if ce['artifact_can_decide_convergence'] else 'NO'}")
    add(f"  nan_occurred rows             : {ce['nan_occurred_rows']}")
    add(f"  q_bic_failed rows             : {ce['q_bic_failed_rows']}")
    add("")
    add("-- free parameters by candidate K --")
    add("  " + "  ".join(f"K={k}:{p}" for k, p in
                         report["free_parameters_by_k"].items()))
    add("")
    rt = report["runtime"]
    add("-- mean seconds per fit at the frozen budget --")
    add(f"  total {rt['total_runtime_s']} s")
    add("  by n      : " + "  ".join(f"{n}:{v}" for n, v in rt["mean_by_n"].items()))
    add("  by K_est  : " + "  ".join(f"{k}:{v}" for k, v in
                                     rt["mean_by_k_est"].items()))
    add("  by start  : " + "  ".join(f"{s}:{v}" for s, v in
                                     rt["mean_by_start"].items()))
    add("  n x K_est :")
    add("          " + "".join(f"   K={k}" for k in CANDIDATE_K))
    for n in N_GRID:
        add(f"    n={n:<4d}" + "".join(
            f"{rt['mean_by_n_and_k'][f'{n}|{k}']:>7.2f}" for k in CANDIDATE_K))
    add("")
    pr = report["runtime_projection"]
    add(f"-- projection for the proposed scope (K_TRUE={PROPOSED_K_TRUE}, "
        f"n in {PROPOSED_N}) --")
    add(f"  {pr['scope_fits']} fits, {pr['scope_runtime_s_at_baseline_budget']} s "
        f"at num_iter={BASELINE_NUM_ITER}")
    for m, h in pr["projection_hours"].items():
        mark = "  <- proposed" if m == PROPOSED_MAX_ITER else ""
        add(f"    num_iter={m:<3d} -> {h:5.2f} h (linear estimate){mark}")
    add("")
    ss = report["score_shape_k_true_5"]
    add(f"-- K_TRUE={PROPOSED_K_TRUE}: mean S1 held-out score by candidate K "
        "(higher is better) --")
    add("          " + "".join(f"     K={k}" for k in CANDIDATE_K))
    for n in N_GRID:
        add(f"    n={n:<4d}" + "".join(f"{v:>9.5f}" for v in ss["s1_mean_by_k"][n])
            + f"   argmax K={ss['argmax_k'][n]['S1']}")
    add("")
    add(f"-- K_TRUE={PROPOSED_K_TRUE}: mean Q_strict by candidate K "
        "(higher is better) --")
    add("          " + "".join(f"     K={k}" for k in CANDIDATE_K))
    for n in N_GRID:
        add(f"    n={n:<4d}" + "".join(
            f"{v:>9.0f}" for v in ss["q_strict_mean_by_k"][n])
            + f"   argmax K={ss['argmax_k'][n]['Q_strict']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="emit the machine-readable report instead of text")
    args = parser.parse_args(argv)
    report = build()
    print(json.dumps(report, indent=2, sort_keys=True) if args.json
          else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
