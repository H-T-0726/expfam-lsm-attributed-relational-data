"""Phase 9P (Issue #104): relational-w sensitivity characterization of
C_Lap vs C_Q under the Phase 9K condition.

Only w changes: weak w = 1/sqrt(2), strong w = sqrt(2). The w = 1 baseline
is the historical Phase 9K run and is read, never rerun. Everything else
(n, d, K_true, X families, w0 = -1, f_scale, L, iterations, K grid,
start_B, the 20 seed tuples, C_Q, Candidate B, N = 75) is the Phase 9K
protocol via dataclasses.replace, and each new condition runs the Phase 9K
driver (run_lap_vs_cq_20.make_driver) unchanged.

Subcommands (each once, into fresh directories)::

    python run_w_sensitivity.py preflight --root <root>
    python run_w_sensitivity.py run --condition weak_w --root <root>
    python run_w_sensitivity.py run --condition strong_w --root <root>
    python run_w_sensitivity.py analyze --root <root>

preflight regenerates the three datasets per replicate without inference,
checks Z/F/X bit-identity and records the true-eta_Y context. Nothing here
ranks the two criteria. Lineage E.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import laplace_k_criterion as lk                                   # noqa: E402
import run_family_selection_pilot as pilot                         # noqa: E402
import run_joint_family_k_selection as joint                       # noqa: E402
import run_lap_vs_cq_20 as pc                                      # noqa: E402

WRAPPER_VERSION = "w-sensitivity-v1"
REPO_ROOT = _HERE.parents[2]
BASELINE_DIR = REPO_ROOT / "expfam/results/lap_vs_cq_20/phase9k_20260928"
PHASE9K_RUN_SHA = "bbf45e461a2ef252909a02a64732faaa721a8311"
W_VALUES = {"weak_w": 1 / math.sqrt(2), "baseline": 1.0,
            "strong_w": math.sqrt(2)}
NEW_CONDITIONS = ("weak_w", "strong_w")
CONDITIONS = ("weak_w", "baseline", "strong_w")
K_TRUE = pc.K_TRUE
NEW_EM_CAP = 400
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def protocol_for(condition: str) -> joint.JointProtocol:
    if condition == "baseline":
        return pc.PROTOCOL
    return dataclasses.replace(pc.PROTOCOL, stage=f"w_sensitivity_{condition}",
                               w=W_VALUES[condition])


def authorization(condition: str) -> dict[str, Any]:
    return {"gate": f"Phase 9P relational-w sensitivity, {condition} "
                    f"(w = {W_VALUES[condition]!r}), one run, <= 200 EM",
            "authorized": True, "authorized_by": "Human",
            "authorized_in": "Issue #104 body (bounded authorization)"}


# --------------------------------------------------------------------------
# preflight: no inference
# --------------------------------------------------------------------------

def signal_context(condition: str, replicate: str, dataset,
                   w0: float, w: float) -> dict[str, Any]:
    upper = np.triu_indices(dataset.Z.shape[0], k=1)
    eta = w0 + w * (dataset.Z @ dataset.Z.T)[upper]
    prob = 1.0 / (1.0 + np.exp(-eta))
    row = {"condition": condition, "replicate": replicate, "w": w,
           "w2_K_true": w * w * K_TRUE,
           "edge_density": float(np.mean(dataset.Y[upper])),
           "eta_mean": float(np.mean(eta)), "eta_sd": float(np.std(eta)),
           "prob_mean": float(np.mean(prob))}
    for q in QUANTILES:
        row[f"eta_q{int(q * 100):02d}"] = float(np.quantile(eta, q))
        row[f"prob_q{int(q * 100):02d}"] = float(np.quantile(prob, q))
    return row


def preflight(out: Path) -> dict[str, Any]:
    if out.exists():
        raise SystemExit(f"{out} exists; a recorded run is never overwritten")
    identity, context = [], []
    for rep in pc.PROTOCOL.replicates:
        data = {c: pilot.build_dataset(protocol_for(c), rep)
                for c in CONDITIONS}
        base = data["baseline"]
        identity.append({
            "replicate": rep.label, "data_seed": rep.data_seed,
            **{f"{m}_identical_{c}": bool(np.array_equal(
                getattr(data[c], m), getattr(base, m)))
               for c in NEW_CONDITIONS for m in ("Z", "F", "X")},
            **{f"Y_identical_{c}": bool(np.array_equal(data[c].Y, base.Y))
               for c in NEW_CONDITIONS}})
        for c in CONDITIONS:
            p = protocol_for(c)
            context.append(signal_context(c, rep.label, data[c], p.w0, p.w))
    all_zfx = all(r[f"{m}_identical_{c}"] for r in identity
                  for c in NEW_CONDITIONS for m in ("Z", "F", "X"))
    out.mkdir(parents=True)
    pilot._write_csv(out / "zfx_identity.csv", identity)
    pilot._write_csv(out / "relational_signal_context.csv", context)
    result = {"wrapper_version": WRAPPER_VERSION,
              "code_sha": pilot._git_sha(), "git_dirty": pilot._git_dirty(),
              "replicates": len(identity), "zfx_identical_all": all_zfx,
              "zfx_identical_count": sum(
                  all(r[f"{m}_identical_{c}"] for c in NEW_CONDITIONS
                      for m in ("Z", "F", "X")) for r in identity),
              "inference_runs": 0,
              "note": "Y is the manipulated side; no claim of identical "
                      "Bernoulli random-number coupling across w."}
    pilot._write_json(out / "preflight.json", result)
    return result


# --------------------------------------------------------------------------
# run one new condition (the Phase 9K driver, unchanged)
# --------------------------------------------------------------------------

def write_states(path: Path, states: Sequence[dict[str, Any]],
                 code_sha: str, condition: str) -> None:
    payload = {
        "code_sha": code_sha, "evaluator_version": lk.EVALUATOR_VERSION,
        "wrapper_version": WRAPPER_VERSION, "condition": condition,
        "w": W_VALUES[condition],
        "x_y_provenance": "X and Y are not stored. They are fully "
                          "regenerable from data_seed with "
                          "run_family_selection_pilot.build_dataset("
                          "run_w_sensitivity.protocol_for("
                          f"{condition!r}), replicate).",
        "float_format": "Python round-trip float repr (exact float64)",
        "entries": list(states),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")


def run_condition(condition: str, root: Path) -> dict[str, Any]:
    if condition not in NEW_CONDITIONS:
        raise SystemExit(f"{condition!r} is not a new condition; the "
                         f"baseline is historical and is never rerun")
    pre = json.loads((root / "preflight" / "preflight.json")
                     .read_text("utf-8"))
    if not pre["zfx_identical_all"]:
        raise SystemExit("Z/F/X identity failed in preflight: blocked")
    out = root / condition
    protocol = protocol_for(condition)
    rows: list = []
    states: list = []
    code_sha = pilot._git_sha()
    try:
        joint.execute(out, protocol=protocol,
                      driver=pc.make_driver(rows, states),
                      authorization=authorization(condition))
    finally:
        if out.is_dir():
            if rows:
                pilot._write_csv(out / "laplace_by_k.csv", rows)
            if states:
                write_states(out / "fitted_states.json", states, code_sha,
                             condition)
    summary = pc.paired_summary(rows, pc._read(out / "family_by_k.csv"))
    summary["condition"], summary["w"] = condition, W_VALUES[condition]
    pilot._write_json(out / "paired_summary.json", summary)
    return summary


# --------------------------------------------------------------------------
# combined descriptive analysis (baseline read-only)
# --------------------------------------------------------------------------

def _dist(values: Sequence[float]) -> dict[str, Any]:
    v = [x for x in values if x is not None and math.isfinite(x)]
    return pc._dist(v)


def condition_dir(root: Path, condition: str) -> Path:
    return BASELINE_DIR if condition == "baseline" else root / condition


def load_condition(root: Path, condition: str) -> dict[str, Any]:
    d = condition_dir(root, condition)
    lap = pc._read(d / "laplace_by_k.csv")
    fam = pc._read(d / "family_by_k.csv")
    return {"laplace": lap, "family": fam,
            "cq": pc._read(d / "cq_decomposition.csv"),
            "ledger": pc._read(d / "execution_ledger.csv"),
            "runinfo": json.loads((d / "runinfo.json").read_text("utf-8")),
            "joint_summary": json.loads((d / "summary.json")
                                        .read_text("utf-8")),
            "paired": pc.paired_summary(lap, fam)}


def transition_label(a: int | None, b: int | None) -> str:
    if a is None or b is None:
        return "unavailable"
    if a == b:
        return "same"
    return f"{pc.category(a)}->{pc.category(b)}"


def transitions(paired: dict[str, dict[str, Any]], key: str
                ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by = {c: {d["replicate"]: d for d in paired[c]["datasets"]}
          for c in CONDITIONS}
    rows = []
    for rep in pc.PROTOCOL.replicates:
        path = [by[c][rep.label][key] for c in CONDITIONS]
        rows.append({"replicate": rep.label,
                     **{c: k for c, k in zip(CONDITIONS, path)},
                     "path": "->".join("NA" if k is None else str(k)
                                       for k in path),
                     "all_same": None not in path and len(set(path)) == 1,
                     "weak_to_baseline": transition_label(path[0], path[1]),
                     "baseline_to_strong": transition_label(path[1],
                                                            path[2])})
    summary: dict[str, Any] = {
        "all_three_same": sum(r["all_same"] for r in rows),
        "available": sum("NA" not in r["path"] for r in rows)}
    for leg in ("weak_to_baseline", "baseline_to_strong"):
        labels = [r[leg] for r in rows]
        summary[leg] = {x: labels.count(x) for x in sorted(set(labels))}
    summary["paths"] = {}
    for r in rows:
        summary["paths"][r["path"]] = summary["paths"].get(r["path"], 0) + 1
    return rows, summary


def matched_margins(paired: dict[str, dict[str, Any]]
                    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by = {c: {d["replicate"]: d for d in paired[c]["datasets"]}
          for c in CONDITIONS}
    keys = ("delta_Lap_23", "delta_Q_23", "lap_gap", "q_gap")
    rows = []
    for rep in pc.PROTOCOL.replicates:
        row: dict[str, Any] = {"replicate": rep.label}
        for key in keys:
            vals = {c: by[c][rep.label].get(key) for c in CONDITIONS}
            for c in CONDITIONS:
                row[f"{key}_{c}"] = vals[c]
            for c in NEW_CONDITIONS:
                ok = vals[c] is not None and vals["baseline"] is not None
                row[f"{key}_change_baseline_to_{c}"] = (
                    vals[c] - vals["baseline"] if ok else None)
        rows.append(row)
    summary = {f"{key}_change_baseline_to_{c}": _dist(
        [r[f"{key}_change_baseline_to_{c}"] for r in rows])
        for key in keys for c in NEW_CONDITIONS}
    summary.update({f"{key}_sign_change_baseline_to_{c}": sum(
        r[f"{key}_{c}"] is not None and r[f"{key}_baseline"] is not None
        and (r[f"{key}_{c}"] > 0) != (r[f"{key}_baseline"] > 0)
        for r in rows) for key in ("delta_Lap_23", "delta_Q_23")
        for c in NEW_CONDITIONS})
    return rows, summary


def decomposition(data: dict[str, Any], n: int) -> dict[str, Any]:
    """Phase 9K P6 quantities, K = 2 -> 3, per replicate.

    -2 ell_Lap = D_mode + V with D_mode = -2 phi - ||Z_hat||^2 - nK ln 2pi
    and the Laplace volume V = ||Z_hat||^2 + log|H|.
    """
    lap = {(r["replicate"], int(r["k"])): r for r in data["laplace"]}
    cq = {(r["replicate"], int(r["k"])): r for r in data["cq"]}
    per = []
    for rep in pc.PROTOCOL.replicates:
        a, b = lap.get((rep.label, 2)), lap.get((rep.label, 3))
        qa, qb = cq.get((rep.label, 2)), cq.get((rep.label, 3))
        row: dict[str, Any] = {"replicate": rep.label}
        if a and b and a["laplace_status"] == b["laplace_status"] == "OK":
            def d_mode(r, k):
                return (-2 * float(r["phi_at_mode"])
                        - float(r["Z_hat_sq_norm"])
                        - n * k * math.log(2 * math.pi))

            def vol(r):
                return float(r["Z_hat_sq_norm"]) + float(r["logdet_H"])
            row.update({
                "lap_fit_gain": d_mode(a, 2) - d_mode(b, 3),
                "lap_volume_increment": vol(b) - vol(a),
                "lap_parameter_increment": float(b["parameter_term"])
                - float(a["parameter_term"])})
        if qa and qb:
            row.update({
                "q_fit_gain": float(qa["D_K"]) - float(qb["D_K"]),
                "q_PZ_increment": float(qb["P_Z"]) - float(qa["P_Z"]),
                "q_parameter_increment": float(qb["P_theta"])
                - float(qa["P_theta"])})
        if "lap_fit_gain" in row and "q_fit_gain" in row:
            row["fit_gain_lap_minus_q"] = row["lap_fit_gain"] \
                - row["q_fit_gain"]
        per.append(row)
    keys = ("lap_fit_gain", "lap_volume_increment", "lap_parameter_increment",
            "q_fit_gain", "q_PZ_increment", "q_parameter_increment",
            "fit_gain_lap_minus_q")
    return {"per_replicate": per,
            "summary": {k: _dist([r.get(k) for r in per]) for k in keys}}


def technical(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    warnings = data["joint_summary"].get(
        "candidate_convergence_warnings_by_k", {})
    for k in range(1, 6):
        rows = [r for r in data["laplace"] if int(r["k"]) == k]
        statuses = [r["laplace_status"] for r in rows]
        ok = [r for r in rows if r["laplace_status"] == "OK"]
        out[f"K{k}"] = {
            "refits": len(rows),
            **{s: statuses.count(s) for s in
               ("OK", "NOT_STATIONARY", "HESSIAN_NOT_PD",
                "EVALUATION_ERROR")},
            "joint_mode_iterations": _dist(
                [float(r["joint_mode_iterations"]) for r in ok]),
            "grad_inf": _dist([float(r["grad_inf"]) for r in ok]),
            "min_eigenvalue_H": _dist([float(r["min_eigenvalue_H"])
                                       for r in ok]),
            "logdet_H": _dist([float(r["logdet_H"]) for r in ok]),
            "candidate_convergence_warnings": _dist(
                [float(v) for key, v in warnings.items()
                 if key.endswith(f"/K={k}")]),
        }
    return out


TRUE_ASSIGNMENT = "|".join(pc.PROTOCOL.family_x_list)


def family_behavior(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in range(1, 6):
        rows = [r for r in data["laplace"] if int(r["k"]) == k]
        fam = [r for r in data["family"] if int(r["k"]) == k]
        groups: dict[str, dict[str, int]] = {}
        for r in fam:
            g = groups.setdefault(r["family_x_true"], {})
            g[r["selected_family"]] = g.get(r["selected_family"], 0) + 1
        out[f"K{k}"] = {
            "exact_true_assignment": sum(
                r["selected_assignment"] == TRUE_ASSIGNMENT for r in rows),
            "refits": len(rows),
            "selected_by_true_family": groups}
    return out


def technical_counts(data: dict[str, Any]) -> dict[str, Any]:
    ledger = data["ledger"]
    p = data["paired"]
    return {"em_attempted": int(data["runinfo"]["em_executions"]),
            "em_success": sum(r["status"] == "SUCCESS" for r in ledger),
            "run_status": data["runinfo"]["run_status"],
            "em_complete_datasets": p["em_complete_datasets"],
            "incomplete_datasets": p["datasets_planned"]
            - p["em_complete_datasets"],
            "c_lap_complete_datasets": p["c_lap_complete_datasets"],
            **p["P5_technical"]}


def paired_counts(p: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in p["P3_paired"].items() if k != "pairs"}


def analyze(root: Path) -> dict[str, Any]:
    out = root / "combined"
    if out.exists():
        raise SystemExit(f"{out} exists; a recorded run is never overwritten")
    data = {c: load_condition(root, c) for c in CONDITIONS}
    committed = json.loads((BASELINE_DIR / "paired_summary.json")
                           .read_text("utf-8"))
    recomputed = json.loads(json.dumps(data["baseline"]["paired"],
                                       default=pilot._json_default))
    base_check = all(committed[key] == recomputed[key] for key in
                     ("P1_K_hat_Lap", "P2_K_hat_Q", "P3_paired", "P4_margins",
                      "P5_technical"))
    paired = {c: data[c]["paired"] for c in CONDITIONS}
    n = pc.PROTOCOL.n
    lap_rows, lap_tr = transitions(paired, "K_hat_Lap")
    q_rows, q_tr = transitions(paired, "K_hat_Q")
    margin_rows, margin_summary = matched_margins(paired)
    decomp = {c: decomposition(data[c], n) for c in CONDITIONS}
    context = pc._read(root / "preflight" / "relational_signal_context.csv")
    ctx_summary = {c: {key: _dist([float(r[key]) for r in context
                                   if r["condition"] == c])
                       for key in ("edge_density", "eta_mean", "eta_sd",
                                   "prob_mean", "prob_q05", "prob_q95")}
                   for c in CONDITIONS}
    per_condition = {c: {
        "w": W_VALUES[c], "w2_K_true": W_VALUES[c] ** 2 * K_TRUE,
        "source": str(condition_dir(root, c).resolve().relative_to(REPO_ROOT)),
        "technical": technical_counts(data[c]),
        "K_hat_Lap": paired[c]["P1_K_hat_Lap"],
        "K_hat_Q": paired[c]["P2_K_hat_Q"],
        "paired": paired_counts(paired[c]),
        "margins": {k: v for k, v in paired[c]["P4_margins"].items()},
        "decomposition_K2_to_K3": decomp[c]["summary"],
        "technical_by_K": technical(data[c]),
        "family_by_K": family_behavior(data[c]),
        "signal_context": ctx_summary[c]} for c in CONDITIONS}
    complete = [per_condition[c]["technical"]["c_lap_complete_datasets"] == 20
                and per_condition[c]["technical"]["em_complete_datasets"] == 20
                for c in NEW_CONDITIONS]
    pre = json.loads((root / "preflight" / "preflight.json")
                     .read_text("utf-8"))
    new_em = sum(per_condition[c]["technical"]["em_attempted"]
                 for c in NEW_CONDITIONS)
    if not pre["zfx_identical_all"] or not base_check or new_em > NEW_EM_CAP:
        decision = "RELATIONAL_W_SENSITIVITY_BLOCKED"
    elif all(complete):
        decision = "RELATIONAL_W_SENSITIVITY_CHARACTERIZED"
    else:
        decision = "RELATIONAL_W_SENSITIVITY_PARTIAL"
    summary = {
        "wrapper_version": WRAPPER_VERSION,
        "code_sha": pilot._git_sha(), "git_dirty": pilot._git_dirty(),
        "baseline_source": str(BASELINE_DIR.relative_to(REPO_ROOT)),
        "baseline_recomputed_summary_matches_committed": base_check,
        "baseline_em_added": 0,
        "new_em_planned": NEW_EM_CAP, "new_em_attempted": new_em,
        "new_em_successful": sum(per_condition[c]["technical"]["em_success"]
                                 for c in NEW_CONDITIONS),
        "zfx_identical_all": pre["zfx_identical_all"],
        "conditions": per_condition,
        "K_hat_Lap_transitions": lap_tr, "K_hat_Q_transitions": q_tr,
        "matched_margin_changes": margin_summary,
        "DECISION": decision,
        "unit": "the same 20 replicate lineages (same Z/F/X and seeds) "
                "characterised under three w values; not 60 independent "
                "datasets",
        "do_not_claim": ["general robustness", "consistency",
                         "general superiority of C_Lap or C_Q",
                         "general recovery rates",
                         "a pure signal-strength effect independent of "
                         "edge density / saturation",
                         "generalisation to other n/d/K_true/X signal/"
                         "family compositions", "real-data effectiveness"],
        "lineage": "E (experimental prototype; not adoptable for the "
                   "manuscript)",
    }
    out.mkdir(parents=True)
    pilot._write_json(out / "combined_summary.json", summary)
    pilot._write_csv(out / "k_hat_lap_transitions.csv", lap_rows)
    pilot._write_csv(out / "k_hat_q_transitions.csv", q_rows)
    pilot._write_csv(out / "matched_margins.csv", margin_rows)
    pilot._write_csv(out / "decomposition_by_replicate.csv", [
        {"condition": c, **r} for c in CONDITIONS
        for r in decomp[c]["per_replicate"]])
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("preflight", "run", "analyze"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--condition", choices=NEW_CONDITIONS)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.root / "preflight")
    elif args.command == "run":
        if args.condition is None:
            parser.error("--condition is required for run")
        s = run_condition(args.condition, args.root)
        result = {k: s[k] for k in ("c_lap_complete_datasets", "P1_K_hat_Lap",
                                    "P2_K_hat_Q", "P5_technical")}
    else:
        s = analyze(args.root)
        result = {k: s[k] for k in ("new_em_attempted", "zfx_identical_all",
                                    "baseline_recomputed_summary_matches_"
                                    "committed", "DECISION")}
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
