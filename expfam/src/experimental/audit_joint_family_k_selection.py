"""Independent artifact audit for the Issue #75 joint family + K run.

Reads a run directory and never imports the runner: the frozen protocol below
is transcribed from the Issue #75 body (``PHASE9D_PROTOCOL_FROZEN``), so a
drift in the runner's constants shows up as a difference instead of agreeing
with itself.

Severity (root CLAUDE.md, research-first policy):
- BLOCKER: the result is not valid evidence (protocol drift, missing or
  duplicated primary rows, non-finite required values, retry / replacement /
  seed rescue, cross-K state reuse, criterion inconsistency, provenance
  mismatch, summary disagreeing with the raw rows).
- HIGH: the run is not clean evidence (dirty worktree, status mismatch).
- WARNING: recorded, does not invalidate (candidate convergence misses with
  finite values; a missing diagnostic decomposition).

technical_validity = VALID iff BLOCKER = 0, HIGH = 0 and run_status SUCCESS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Shared audit helpers from the #74 auditor (row-level provenance checks). It
# does not import any runner either.
import audit_family_selection_pilot as audit74                    # noqa: E402

AUDITOR_VERSION = "joint-family-k-auditor-v1"

# ---- Issue #75 frozen protocol, transcribed ------------------------------
EXPECTED: dict[str, Any] = {
    "stage": "joint",
    "n": 75, "d": 12, "k_true": 3,
    "family_x_list": ["gaussian"] * 3 + ["bernoulli"] * 6 + ["poisson"] * 3,
    "family_y": "bernoulli",
    "sigma_x_var": 1.0, "w0": -1.0, "w": 1.0,
    "f_scale": math.sqrt(2.0),
    "L": 5,
    "exploration_num_iter": 8, "refit_num_iter": 8,
    "k_candidates": [1, 2, 3, 4, 5],
    "replicates": [
        {"label": "rep1", "data_seed": 961001, "search_seed": 962001,
         "refit_seed": 963001},
        {"label": "rep2", "data_seed": 961002, "search_seed": 962002,
         "refit_seed": 963002},
        {"label": "rep3", "data_seed": 961003, "search_seed": 962003,
         "refit_seed": 963003},
    ],
    "starts": [{"label": "start_B", "ambiguous_start": "bernoulli"},
               {"label": "start_P", "ambiguous_start": "poisson"}],
    "expected_em_executions": 60,          # 3 x 2 x 5 x 2
    "candidate_optimizer": "bfgs",
    "optimizer_settings": {"method": "BFGS",
                           "jac": "analytic_production_gradient",
                           "maxiter": 2000, "gtol": 1e-10},
    "convergence_grad_inf_tol": 1e-8,
    "ambiguous_loading_installation": "selected_candidate_loading",
    "refit_failure_policy": "fail_fast",
}
SCORE_DECIDED_TRUE_BERNOULLI = [3, 4, 5, 6, 7, 8]
COUNTERS = ("retry_count", "replacement_count", "seed_rescue_count")
BASE_ARTIFACTS = ("protocol.json", "runinfo.json", "execution_ledger.csv")
COMPLETE_ARTIFACTS = ("generator_provenance.csv", "support_gate.csv",
                      "family_scores.csv", "selection_trace.csv",
                      "family_by_k.csv", "cq_by_k.csv",
                      "cq_decomposition.csv", "joint_selection.csv",
                      "summary.json")


def _f(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Audit:
    def __init__(self) -> None:
        self.findings: list[dict[str, str]] = []

    def add(self, severity: str, check: str, message: str) -> None:
        self.findings.append({"severity": severity, "check": check,
                              "message": message})

    def count(self, severity: str) -> int:
        return sum(f["severity"] == severity for f in self.findings)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_protocol(protocol: dict[str, Any], a: Audit) -> None:
    for key in ("stage", "n", "d", "k_true", "family_x_list", "family_y",
                "L", "exploration_num_iter", "refit_num_iter",
                "k_candidates", "starts", "expected_em_executions",
                "candidate_optimizer", "ambiguous_loading_installation",
                "refit_failure_policy"):
        if protocol.get(key) != EXPECTED[key]:
            a.add("BLOCKER", "protocol", f"{key}: {protocol.get(key)!r} != "
                                         f"frozen {EXPECTED[key]!r}")
    for key in ("sigma_x_var", "w0", "w", "f_scale"):
        value = _f(protocol.get(key))
        if value is None or not math.isclose(value, EXPECTED[key],
                                             rel_tol=1e-12, abs_tol=1e-12):
            a.add("BLOCKER", "protocol", f"{key}: {protocol.get(key)!r} != "
                                         f"frozen {EXPECTED[key]!r}")
    reps = [{k: r.get(k) for k in ("label", "data_seed", "search_seed",
                                    "refit_seed")}
            for r in protocol.get("replicates", [])]
    if reps != EXPECTED["replicates"]:
        a.add("BLOCKER", "protocol", f"replicates/seeds: {reps!r}")
    settings = protocol.get("candidate_optimizer_settings") or {}
    for key, wanted in EXPECTED["optimizer_settings"].items():
        if settings.get(key) != wanted:
            a.add("BLOCKER", "protocol",
                  f"candidate_optimizer_settings.{key}: "
                  f"{settings.get(key)!r} != {wanted!r}")
    tol = (protocol.get("candidate_convergence_rule") or {}).get(
        "convergence_grad_inf_tol")
    if tol != EXPECTED["convergence_grad_inf_tol"]:
        a.add("BLOCKER", "protocol", f"convergence_grad_inf_tol: {tol!r}")


def expected_executions() -> list[tuple[str, str, str, str, int]]:
    rows = []
    for rep in EXPECTED["replicates"]:
        for start in EXPECTED["starts"]:
            for k in EXPECTED["k_candidates"]:
                rows.append((rep["label"], start["label"], str(k),
                             "exploration", rep["search_seed"]))
                rows.append((rep["label"], start["label"], str(k), "refit",
                             rep["refit_seed"]))
    return rows


def check_ledger(rows: Sequence[dict[str, str]], runinfo: dict[str, Any],
                 complete: bool, a: Audit) -> int:
    attempted = len(rows)
    if runinfo.get("em_executions") != attempted:
        a.add("BLOCKER", "ledger", f"runinfo em_executions "
              f"{runinfo.get('em_executions')!r} != ledger rows {attempted}")
    if attempted > EXPECTED["expected_em_executions"]:
        a.add("BLOCKER", "ledger", f"{attempted} attempts exceed the frozen "
              f"{EXPECTED['expected_em_executions']} (hidden retry?)")
    if [row.get("sequence") for row in rows] != \
            [str(i + 1) for i in range(attempted)]:
        a.add("BLOCKER", "ledger", "sequence is not 1..N")
    keys = [(r.get("replicate"), r.get("start_label"), r.get("k"),
             r.get("execution_kind")) for r in rows]
    if len(set(keys)) != len(keys):
        a.add("BLOCKER", "ledger", "an execution appears twice (retry?)")
    expected = expected_executions()
    wanted_seed = {e[:4]: e[4] for e in expected}
    for row, key in zip(rows, keys):
        if key not in wanted_seed:
            a.add("BLOCKER", "ledger", f"unexpected execution {key}")
        elif _f(row.get("seed")) != wanted_seed[key]:
            a.add("BLOCKER", "ledger", f"{key}: seed {row.get('seed')} != "
                                       f"frozen {wanted_seed[key]}")
    if complete:
        missing = [e[:4] for e in expected if e[:4] not in set(keys)]
        if missing:
            a.add("BLOCKER", "ledger", f"{len(missing)} expected executions "
                                       f"missing, e.g. {missing[:3]}")
        bad = [r for r in rows if r.get("status") != "SUCCESS"]
        if bad:
            a.add("BLOCKER", "ledger", f"{len(bad)} executions not SUCCESS")
    return attempted


def check_cq(cq_rows: Sequence[dict[str, str]],
             gate_rows: Sequence[dict[str, str]], a: Audit
             ) -> dict[tuple[str, str], dict[int, float]]:
    """Coverage, finiteness, counters, cross-K independence, p_K and C_Q."""

    expected_keys = {(r["label"], s["label"], str(k))
                     for r in EXPECTED["replicates"]
                     for s in EXPECTED["starts"]
                     for k in EXPECTED["k_candidates"]}
    keys = [(r.get("replicate"), r.get("start_label"), r.get("k"))
            for r in cq_rows]
    if len(keys) != len(set(keys)):
        a.add("BLOCKER", "cq_by_k", "duplicate (replicate, start, K) rows")
    if set(keys) != expected_keys:
        a.add("BLOCKER", "cq_by_k", f"coverage differs: missing "
              f"{sorted(expected_keys - set(keys))[:3]}, extra "
              f"{sorted(set(keys) - expected_keys)[:3]}")
    seeds = {r["label"]: r for r in EXPECTED["replicates"]}
    start_family = {s["label"]: s["ambiguous_start"]
                    for s in EXPECTED["starts"]}
    ambiguous_by = {}
    for g in gate_rows:
        key = (g.get("replicate"), g.get("start_label"), g.get("k"))
        if g.get("decided_by") != "gate":
            ambiguous_by.setdefault(key, set()).add(int(g["column"]))
    d, log_n = EXPECTED["d"], math.log(EXPECTED["n"])
    curves: dict[tuple[str, str], dict[int, float]] = {}
    for row in cq_rows:
        label = f"{row.get('replicate')}/{row.get('start_label')}/K={row.get('k')}"
        rep = seeds.get(row.get("replicate"), {})
        if (_f(row.get("data_seed")) != rep.get("data_seed")
                or _f(row.get("search_seed")) != rep.get("search_seed")
                or _f(row.get("refit_seed")) != rep.get("refit_seed")):
            a.add("BLOCKER", "cq_by_k", f"{label}: seeds differ from frozen")
        for counter in COUNTERS:
            if _f(row.get(counter)) != 0:
                a.add("BLOCKER", "cq_by_k", f"{label}: {counter}="
                                            f"{row.get(counter)!r}")
        if row.get("failure_policy") != "fail_fast":
            a.add("BLOCKER", "cq_by_k", f"{label}: failure_policy "
                                        f"{row.get('failure_policy')!r}")
        if row.get("warm_start_source") != "none":
            a.add("BLOCKER", "cross_k", f"{label}: warm_start_source "
                                        f"{row.get('warm_start_source')!r}")
        # Every K starts from the start family on its OWN ambiguous columns,
        # never from another K's selection.
        initial = row.get("initial_assignment", "").split("|")
        ambiguous = ambiguous_by.get((row.get("replicate"),
                                      row.get("start_label"), row.get("k")),
                                     set())
        if any(initial[c] != start_family.get(row.get("start_label"))
               for c in ambiguous if c < len(initial)):
            a.add("BLOCKER", "cross_k", f"{label}: an ambiguous column did "
                  f"not start from the start family (carried-over state?)")
        q, cq = _f(row.get("Q_strict")), _f(row.get("C_Q"))
        if q is None or cq is None or not (math.isfinite(q)
                                           and math.isfinite(cq)):
            a.add("BLOCKER", "cq_by_k", f"{label}: Q_strict/C_Q not finite")
            continue
        k = int(row["k"])
        selected = row.get("selected_assignment", "").split("|")
        n_gauss = sum(f == "gaussian" for f in selected)
        p_k = (k * d - k * (k - 1) // 2 + n_gauss
               + (1 if EXPECTED["family_y"] == "gaussian" else 0))
        if _f(row.get("n_gaussian_x_cols")) != n_gauss:
            a.add("BLOCKER", "criterion", f"{label}: n_gaussian_x_cols "
                  f"{row.get('n_gaussian_x_cols')} != {n_gauss} counted from "
                  f"the selected assignment")
        if _f(row.get("num_params")) != p_k:
            a.add("BLOCKER", "criterion", f"{label}: num_params "
                                          f"{row.get('num_params')} != p_K {p_k}")
        if not math.isclose(cq, -2.0 * q + p_k * log_n, rel_tol=1e-12,
                            abs_tol=1e-9):
            a.add("BLOCKER", "criterion", f"{label}: C_Q {cq!r} != "
                                          f"-2Q + p_K ln n")
        curves.setdefault((row["replicate"], row["start_label"]), {})[k] = cq
    return curves


def check_same_data_across_k(gate_rows: Sequence[dict[str, str]],
                             prov_rows: Sequence[dict[str, str]],
                             a: Audit) -> None:
    """One draw per replicate: gate results identical across K and starts."""

    per_rep: dict[str, set] = {}
    for g in gate_rows:
        per_rep.setdefault(g.get("replicate"), set()).add(
            (g.get("column"), g.get("decided_by"), g.get("candidates")))
    for rep, signature in per_rep.items():
        if len(signature) != EXPECTED["d"]:
            a.add("BLOCKER", "data", f"{rep}: support gate differs across "
                                     f"K/starts (data redrawn?)")
    keys = [(r.get("replicate"), r.get("column")) for r in prov_rows]
    if len(keys) != len(set(keys)) or len(keys) != \
            len(EXPECTED["replicates"]) * EXPECTED["d"]:
        a.add("BLOCKER", "data", "generator_provenance is not one draw per "
                                 "replicate")
    for r in prov_rows:
        rep = next((x for x in EXPECTED["replicates"]
                    if x["label"] == r.get("replicate")), None)
        if rep is None or _f(r.get("data_seed")) != rep["data_seed"]:
            a.add("BLOCKER", "data", f"{r.get('replicate')}: data seed "
                                     f"{r.get('data_seed')}")


def select_k_hat(curve: dict[int, float]) -> int:
    best = None
    for k in sorted(curve):                      # exact tie -> smaller K
        if best is None or curve[k] < curve[best]:
            best = k
    return best


def recompute_paths(curves, fam_rows) -> dict[tuple[str, str], dict]:
    out = {}
    for (rep, start), curve in curves.items():
        k_hat = select_k_hat(curve)
        at_k = {int(r["column"]): r["selected_family"] for r in fam_rows
                if r.get("replicate") == rep and r.get("start_label") == start
                and r.get("k") == str(k_hat)}
        selected = [at_k.get(c) for c in SCORE_DECIDED_TRUE_BERNOULLI]
        all_six = all(f == "bernoulli" for f in selected)
        ordering = "|".join(str(k) for k in sorted(curve,
                                                   key=lambda k: (curve[k], k)))
        out[(rep, start)] = {
            "k_hat": k_hat, "b_to_p": sum(f == "poisson" for f in selected),
            "all_six": all_six, "ordering": ordering,
            "joint_exact": k_hat == EXPECTED["k_true"] and all_six,
            "category": ("exact" if k_hat == EXPECTED["k_true"] else
                         "under" if k_hat < EXPECTED["k_true"] else "over"),
        }
    return out


def check_selection(paths, joint_rows, summary, a: Audit) -> None:
    by_key = {(r.get("replicate"), r.get("start_label")): r
              for r in joint_rows}
    if len(joint_rows) != len(EXPECTED["replicates"]) * len(EXPECTED["starts"]):
        a.add("BLOCKER", "joint_selection", f"{len(joint_rows)} path rows; "
              f"both starts must be kept for every replicate")
    for key, want in paths.items():
        row = by_key.get(key)
        if row is None:
            a.add("BLOCKER", "joint_selection", f"{key}: path missing")
            continue
        checks = (("k_hat", str(want["k_hat"])),
                  ("cq_ordering", want["ordering"]),
                  ("bernoulli_to_poisson_at_k_hat", str(want["b_to_p"])),
                  ("all_six_bernoulli_at_k_hat", str(want["all_six"])),
                  ("joint_exact", str(want["joint_exact"])),
                  ("k_category", want["category"]))
        for field, value in checks:
            if str(row.get(field)) != value:
                a.add("BLOCKER", "joint_selection", f"{key}: {field}="
                      f"{row.get(field)!r}, raw rows imply {value!r}")
    values = list(paths.values())
    wanted = {
        ("P1_k_recovery", "exact"): sum(p["category"] == "exact" for p in values),
        ("P1_k_recovery", "under"): sum(p["category"] == "under" for p in values),
        ("P1_k_recovery", "over"): sum(p["category"] == "over" for p in values),
        ("P2_family_recovery_at_k_hat", "bernoulli_to_poisson_total"):
            sum(p["b_to_p"] for p in values),
        ("P2_family_recovery_at_k_hat", "all_six_exact_paths"):
            sum(p["all_six"] for p in values),
    }
    for (section, field), value in wanted.items():
        if (summary.get(section) or {}).get(field) != value:
            a.add("BLOCKER", "summary", f"{section}.{field}="
                  f"{(summary.get(section) or {}).get(field)!r}, raw rows "
                  f"imply {value!r}")
    joint = sum(p["joint_exact"] for p in values)
    if summary.get("P3_joint_exact_paths") != joint:
        a.add("BLOCKER", "summary", f"P3_joint_exact_paths="
              f"{summary.get('P3_joint_exact_paths')!r}, raw rows imply {joint}")
    stability = summary.get("P4_start_stability") or {}
    for rep in {k[0] for k in paths}:
        b, p = paths.get((rep, "start_B")), paths.get((rep, "start_P"))
        if b is None or p is None:
            continue
        got = stability.get(rep) or {}
        if got.get("k_hat_agree") != (b["k_hat"] == p["k_hat"]) or \
                got.get("cq_ordering_agree") != (b["ordering"] == p["ordering"]):
            a.add("BLOCKER", "summary", f"P4 {rep} disagrees with raw rows")


def check_decomposition(rows, curves, a: Audit) -> None:
    if not rows:
        a.add("WARNING", "cq_decomposition", "no decomposition rows")
        return
    for row in rows:
        label = f"{row.get('replicate')}/{row.get('start_label')}/K={row.get('k')}"
        direct = _f(row.get("C_Q_direct"))
        recomposed = _f(row.get("C_Q_recomposed"))
        stored = curves.get((row.get("replicate"), row.get("start_label")),
                            {}).get(int(row.get("k", -1)))
        if direct is None or recomposed is None:
            a.add("WARNING", "cq_decomposition", f"{label}: not available")
            continue
        if stored is not None and direct != stored:
            a.add("BLOCKER", "cq_decomposition", f"{label}: C_Q_direct "
                  f"{direct!r} is not the selection value {stored!r}")
        if not math.isclose(recomposed, direct, rel_tol=1e-9, abs_tol=1e-6):
            a.add("BLOCKER", "criterion", f"{label}: D_K + P_Z + P_theta = "
                  f"{recomposed!r} != direct C_Q {direct!r}")


def check_families(fam_rows, cq_rows, gate_rows, a: Audit) -> None:
    by_path = {}
    for r in fam_rows:
        by_path.setdefault((r.get("replicate"), r.get("start_label"),
                            r.get("k")), {})[int(r["column"])] = r
    for row in cq_rows:
        key = (row.get("replicate"), row.get("start_label"), row.get("k"))
        cols = by_path.get(key, {})
        selected = row.get("selected_assignment", "").split("|")
        if len(cols) != EXPECTED["d"] or \
                [cols[c]["selected_family"] for c in sorted(cols)] != selected:
            a.add("BLOCKER", "family_by_k", f"{key}: family rows disagree "
                                            f"with the refit assignment")
    for g in gate_rows:
        key = (g.get("replicate"), g.get("start_label"), g.get("k"))
        fam = by_path.get(key, {}).get(int(g["column"]))
        if fam is None:
            continue
        decided = "gate" if g.get("decided_by") == "gate" else "score"
        if fam.get("decided_by") != decided:
            a.add("BLOCKER", "family_by_k", f"{key} column {g['column']}: "
                  f"decided_by differs from the support gate")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def audit(run_dir: Path) -> dict[str, Any]:
    a = Audit()
    missing = [n for n in BASE_ARTIFACTS if not (run_dir / n).is_file()]
    if missing:
        a.add("BLOCKER", "artifacts", f"missing {missing}")
        return _finish(run_dir, a, "UNKNOWN", None, 0)
    protocol = json.loads((run_dir / "protocol.json").read_text("utf-8"))
    runinfo = json.loads((run_dir / "runinfo.json").read_text("utf-8"))
    check_protocol(protocol, a)
    status = runinfo.get("run_status", "UNKNOWN")
    failure = run_dir / "failure.json"
    ledger = _read_csv(run_dir / "execution_ledger.csv")

    if status == "FAILED" or failure.is_file():
        attempted = check_ledger(ledger, runinfo, False, a)
        if not failure.is_file():
            a.add("BLOCKER", "failure", "FAILED without failure.json")
        else:
            info = json.loads(failure.read_text("utf-8"))
            if info.get("attempted_em_executions") != attempted:
                a.add("BLOCKER", "failure", "failure.json attempts differ "
                                            "from the ledger")
            for counter in COUNTERS:
                if info.get(counter) != 0:
                    a.add("BLOCKER", "failure", f"{counter}="
                                                f"{info.get(counter)!r}")
        return _finish(run_dir, a, "FAILED", None, attempted)

    attempted = check_ledger(ledger, runinfo, True, a)
    if status != "SUCCESS":
        a.add("HIGH", "runinfo", f"run_status {status!r}")
    if runinfo.get("git_dirty") is not False:
        a.add("HIGH", "runinfo", "worktree was not clean when the run began")
    missing = [n for n in COMPLETE_ARTIFACTS if not (run_dir / n).is_file()]
    if missing:
        a.add("BLOCKER", "artifacts", f"missing {missing}")
        return _finish(run_dir, a, status, None, attempted)

    rows = {n[:-4]: _read_csv(run_dir / n) for n in COMPLETE_ARTIFACTS
            if n.endswith(".csv")}
    summary = json.loads((run_dir / "summary.json").read_text("utf-8"))

    check_same_data_across_k(rows["support_gate"],
                             rows["generator_provenance"], a)
    curves = check_cq(rows["cq_by_k"], rows["support_gate"], a)
    check_families(rows["family_by_k"], rows["cq_by_k"], rows["support_gate"],
                   a)
    paths = recompute_paths(curves, rows["family_by_k"])
    check_selection(paths, rows["joint_selection"], summary, a)
    check_decomposition(rows["cq_decomposition"], curves, a)

    # Row-level #74 checks: selected/installed loading provenance, finite
    # candidate values; convergence misses are counted as a diagnostic.
    old = []
    audit74._check_installation_provenance(protocol, rows["selection_trace"],
                                           old)
    audit74._check_finite(rows["family_scores"], "family_scores",
                          ("score", "neg2_score", "loading_norm"), old)
    diagnostic = audit74._recompute_convergence_diagnostic(
        rows["family_scores"], rows["selection_trace"], old)
    for f in old:
        a.add(f["severity"], f["check"], f["message"])
    if diagnostic["candidate_warnings"]:
        a.add("WARNING", "candidate_convergence",
              f"{diagnostic['candidate_warnings']}/"
              f"{diagnostic['candidate_evaluations']} candidate evaluations "
              f"missed grad_inf <= 1e-8 with finite values (diagnostic)")
    return _finish(run_dir, a, status, diagnostic, attempted)


def _finish(run_dir: Path, a: Audit, status: str, diagnostic,
            attempted: int) -> dict[str, Any]:
    blockers, high = a.count("BLOCKER"), a.count("HIGH")
    report = {
        "auditor_version": AUDITOR_VERSION,
        "run_dir": str(run_dir),
        "run_status": status,
        "verdict": "FAIL" if blockers else "PASS",
        "blocker_count": blockers, "high_count": high,
        "warning_count": a.count("WARNING"),
        "technical_validity": ("VALID" if blockers == 0 and high == 0
                               and status == "SUCCESS" else "INVALID"),
        "technical_validity_rule": "BLOCKER == 0 and HIGH == 0 and "
                                   "run_status == SUCCESS",
        "attempted_em_executions": attempted,
        "expected_em_executions": EXPECTED["expected_em_executions"],
        "candidate_convergence_diagnostic": diagnostic,
        "findings": a.findings,
    }
    if run_dir.is_dir():
        (run_dir / "audit_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit an Issue #75 joint family + K run directory.")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = audit(args.run_dir)
    print(f"verdict={report['verdict']} technical_validity="
          f"{report['technical_validity']} blockers={report['blocker_count']} "
          f"high={report['high_count']} warnings={report['warning_count']}")
    for f in report["findings"]:
        print(f"  [{f['severity']}] {f['check']}: {f['message']}")
    return 0 if report["technical_validity"] == "VALID" else 1


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
