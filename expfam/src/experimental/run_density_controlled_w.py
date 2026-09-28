"""Phase 9T (Issue #113): mean-density-controlled relational-w sensitivity.

Weak / strong conditions are instantiated ONLY from the committed Phase 9S2
future_protocol.json (status FROZEN_NOT_EXECUTED, calibration READY): the
Phase 9P protocol with the frozen condition-level w0. The frozen protocol
JSON must equal the instantiated protocol field by field. The baseline
(w0, w) = (-1, 1) is the historical Phase 9K run, read only.

Each new condition runs the unchanged Phase 9K driver once
(run_lap_vs_cq_20.make_driver via joint.execute). The combined analysis
reuses the Phase 9P functions (run_w_sensitivity) and adds the
density-control context and the paired Phase 9P (w0 = -1) vs Phase 9T
(calibrated w0) comparison. No local theta optimization or solver change.

Subcommands (each once, into fresh directories)::

    python run_density_controlled_w.py preflight --root <root>
    python run_density_controlled_w.py run --condition weak_w --root <root>
    python run_density_controlled_w.py run --condition strong_w --root <root>
    python run_density_controlled_w.py analyze --root <root>

Lineage E.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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
import run_w_sensitivity as ws                                     # noqa: E402

WRAPPER_VERSION = "density-controlled-w-v1"
REPO_ROOT = ws.REPO_ROOT
CALIBRATION_DIR = (REPO_ROOT / "expfam/results/"
                   "density_controlled_w_recalibration/phase9s2_20260928")
PHASE9P_ROOT = REPO_ROOT / "expfam/results/relational_w_sensitivity/phase9p_20260928"
CONDITIONS = ws.CONDITIONS
NEW_CONDITIONS = ws.NEW_CONDITIONS
NEW_EM_CAP = 400
READY = "DENSITY_CONTROLLED_W_CALIBRATION_READY"
QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)


def load_frozen(calibration_dir: Path = CALIBRATION_DIR) -> dict[str, Any]:
    """The Phase 9S2 prerequisite; refuses anything but READY + FROZEN."""
    cal = json.loads((calibration_dir / "calibration.json").read_text("utf-8"))
    fp_path = calibration_dir / "future_protocol.json"
    if cal.get("DECISION") != READY or not fp_path.is_file():
        raise SystemExit("Phase 9S2 is not READY: Phase 9T not started")
    fp = json.loads(fp_path.read_text("utf-8"))
    if fp.get("status") != "FROZEN_NOT_EXECUTED" \
            or not (cal["freeze_rule"]["pass"]
                    and cal["crosscheck_rule"]["pass"]) \
            or not all(c in fp.get("frozen", {}) for c in CONDITIONS):
        raise SystemExit("Phase 9S2 prerequisite not satisfied: not started")
    return {"calibration": cal, "future_protocol": fp,
            "sha256": {n: pilot_sha(calibration_dir / n) for n in
                       ("calibration.json", "future_protocol.json")}}


def pilot_sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_for(condition: str, frozen: dict[str, Any] | None = None
                 ) -> joint.JointProtocol:
    """Phase 9P protocol with w0 taken from the frozen artifact."""
    if condition == "baseline":
        return pc.PROTOCOL
    frozen = load_frozen() if frozen is None else frozen
    spec = frozen["future_protocol"]["conditions"][condition]
    proto = dataclasses.replace(ws.protocol_for(condition),
                                stage=spec["stage"], w0=spec["w0"])
    if proto.as_json() != spec:
        raise SystemExit(f"{condition}: protocol differs from the frozen "
                         f"future_protocol.json: blocked")
    fz = frozen["future_protocol"]["frozen"][condition]
    if (proto.w0, proto.w) != (fz["w0"], fz["w"]):
        raise SystemExit(f"{condition}: frozen (w0, w) mismatch: blocked")
    return proto


# --------------------------------------------------------------------------
# preflight (no inference)
# --------------------------------------------------------------------------

def density_context(condition: str, replicate: str, dataset, w0: float,
                    w: float, p_target: float) -> dict[str, Any]:
    upper = np.triu_indices(dataset.Z.shape[0], k=1)
    eta = w0 + w * (dataset.Z @ dataset.Z.T)[upper]
    prob = 1.0 / (1.0 + np.exp(-eta))
    row = {"condition": condition, "replicate": replicate, "w0": w0, "w": w,
           "p_target": p_target, "true_mean_probability": float(prob.mean()),
           "realized_edge_density": float(np.mean(dataset.Y[upper])),
           "eta_mean": float(eta.mean()), "eta_sd": float(eta.std()),
           "prob_mean": float(prob.mean()), "prob_sd": float(prob.std())}
    for q in QUANTILES:
        row[f"prob_q{int(round(q * 100)):02d}"] = float(np.quantile(prob, q))
    row["share_p_lt_0.05"] = float(np.mean(prob < 0.05))
    row["share_p_gt_0.95"] = float(np.mean(prob > 0.95))
    return row


def preflight(root: Path) -> dict[str, Any]:
    out = root / "preflight"
    if out.exists():
        raise SystemExit(f"{out} exists; a recorded run is never overwritten")
    frozen = load_frozen()
    p_target = frozen["future_protocol"]["p_target"]
    protocols = {c: protocol_for(c, frozen) for c in CONDITIONS}
    identity, context = [], []
    for rep in pc.PROTOCOL.replicates:
        data = {c: pilot.build_dataset(protocols[c], rep) for c in CONDITIONS}
        base = data["baseline"]
        identity.append({
            "replicate": rep.label, "data_seed": rep.data_seed,
            **{f"{m}_identical_{c}": bool(np.array_equal(
                getattr(data[c], m), getattr(base, m)))
               for c in NEW_CONDITIONS for m in ("Z", "F", "X")}})
        for c in CONDITIONS:
            p = protocols[c]
            context.append(density_context(c, rep.label, data[c], p.w0, p.w,
                                           p_target))
    all_zfx = all(v for r in identity for k, v in r.items()
                  if k.endswith(NEW_CONDITIONS))
    out.mkdir(parents=True)
    pilot._write_csv(out / "zfx_identity.csv", identity)
    pilot._write_csv(out / "density_context.csv", context)
    result = {
        "wrapper_version": WRAPPER_VERSION,
        "code_sha": pilot._git_sha(),
        "phase9s2_sha256": frozen["sha256"],
        "phase9s2_calibration_code_sha": frozen["calibration"]["code_sha"],
        "phase9s2_decision": frozen["calibration"]["DECISION"],
        "future_protocol_status": frozen["future_protocol"]["status"],
        "p_target": p_target,
        "frozen": {c: {"w0": protocols[c].w0, "w": protocols[c].w}
                   for c in CONDITIONS},
        "replicates": len(identity), "zfx_identical_all": all_zfx,
        "zfx_identical_count": sum(all(v for k, v in r.items()
                                       if k.endswith(NEW_CONDITIONS))
                                   for r in identity),
        "inference_runs": 0,
        "note": "git_dirty is not recorded here because this directory is "
                "itself untracked while it is written",
    }
    pilot._write_json(out / "preflight.json", result)
    return result


# --------------------------------------------------------------------------
# run one new condition
# --------------------------------------------------------------------------

def write_states(path: Path, states: Sequence[dict[str, Any]], code_sha: str,
                 condition: str, protocol: joint.JointProtocol) -> None:
    payload = {
        "code_sha": code_sha, "evaluator_version": lk.EVALUATOR_VERSION,
        "wrapper_version": WRAPPER_VERSION, "condition": condition,
        "w0": protocol.w0, "w": protocol.w,
        "x_y_provenance": "X and Y are not stored. They are regenerable "
                          "from data_seed with run_family_selection_pilot."
                          "build_dataset(run_density_controlled_w."
                          f"protocol_for({condition!r}), replicate).",
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
    protocol = protocol_for(condition)
    out = root / condition
    rows: list = []
    states: list = []
    code_sha = pilot._git_sha()
    auth = {"gate": f"Phase 9T mean-density-controlled {condition} "
                    f"(w0 = {protocol.w0!r}, w = {protocol.w!r}), one run, "
                    f"<= 200 EM",
            "authorized": True, "authorized_by": "Human",
            "authorized_in": "Issue #113 body (prospective, conditional on "
                             "Phase 9S2 READY)"}
    try:
        joint.execute(out, protocol=protocol,
                      driver=pc.make_driver(rows, states),
                      authorization=auth)
    finally:
        if out.is_dir():
            if rows:
                pilot._write_csv(out / "laplace_by_k.csv", rows)
            if states:
                write_states(out / "fitted_states.json", states, code_sha,
                             condition, protocol)
    summary = pc.paired_summary(rows, pc._read(out / "family_by_k.csv"))
    summary.update(condition=condition, w=protocol.w, w0=protocol.w0)
    pilot._write_json(out / "paired_summary.json", summary)
    return summary


# --------------------------------------------------------------------------
# combined analysis
# --------------------------------------------------------------------------

def _dist(values) -> dict[str, Any]:
    return ws._dist(values)


def context_summary(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    keys = [k for k in rows[0] if k not in ("condition", "replicate")]
    return {c: {k: _dist([float(r[k]) for r in rows if r["condition"] == c])
                for k in keys} for c in CONDITIONS}


def phase9p_comparison(t_data: dict[str, Any], p_data: dict[str, Any],
                       t_ctx: Sequence[dict[str, str]],
                       p_ctx: Sequence[dict[str, str]],
                       t_dec: dict[str, Any], p_dec: dict[str, Any]
                       ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = [], {}
    for c in NEW_CONDITIONS:
        tb = {d["replicate"]: d for d in t_data[c]["paired"]["datasets"]}
        pb = {d["replicate"]: d for d in p_data[c]["paired"]["datasets"]}
        tc = {r["replicate"]: r for r in t_ctx if r["condition"] == c}
        pcx = {r["replicate"]: r for r in p_ctx if r["condition"] == c}
        td_ = {r["replicate"]: r for r in t_dec[c]["per_replicate"]}
        pd_ = {r["replicate"]: r for r in p_dec[c]["per_replicate"]}
        crow = []
        for rep in pc.PROTOCOL.replicates:
            r = rep.label
            row: dict[str, Any] = {
                "condition": c, "replicate": r,
                "true_mean_prob_9P": float(pcx[r]["prob_mean"]),
                "true_mean_prob_9T": float(tc[r]["true_mean_probability"]),
                "density_9P": float(pcx[r]["edge_density"]),
                "density_9T": float(tc[r]["realized_edge_density"])}
            for key in ("K_hat_Lap", "K_hat_Q"):
                row[f"{key}_9P"], row[f"{key}_9T"] = pb[r][key], tb[r][key]
                row[f"{key}_transition"] = ws.transition_label(pb[r][key],
                                                               tb[r][key])
            for key in ("delta_Lap_23", "delta_Q_23", "lap_gap", "q_gap"):
                a, b = pb[r].get(key), tb[r].get(key)
                row[f"{key}_9P"], row[f"{key}_9T"] = a, b
                row[f"{key}_change"] = (b - a if a is not None
                                        and b is not None else None)
            for key in ("lap_fit_gain", "lap_volume_increment",
                        "q_fit_gain", "fit_gain_lap_minus_q"):
                a, b = pd_[r].get(key), td_[r].get(key)
                row[f"{key}_change"] = (b - a if a is not None
                                        and b is not None else None)
            row["true_mean_prob_change"] = (row["true_mean_prob_9T"]
                                            - row["true_mean_prob_9P"])
            row["density_change"] = row["density_9T"] - row["density_9P"]
            crow.append(row)
        rows.extend(crow)
        summary[c] = {
            **{f"{key}_transitions": _counts([x[f"{key}_transition"]
                                              for x in crow])
               for key in ("K_hat_Lap", "K_hat_Q")},
            **{f"{key}_change": _dist([x[f"{key}_change"] for x in crow])
               for key in ("true_mean_prob", "density", "delta_Lap_23",
                           "delta_Q_23", "lap_gap", "q_gap", "lap_fit_gain",
                           "lap_volume_increment", "q_fit_gain",
                           "fit_gain_lap_minus_q")},
            "sign_change_delta_Lap_23": sum(
                (x["delta_Lap_23_9P"] > 0) != (x["delta_Lap_23_9T"] > 0)
                for x in crow if x["delta_Lap_23_9P"] is not None
                and x["delta_Lap_23_9T"] is not None),
            "sign_change_delta_Q_23": sum(
                (x["delta_Q_23_9P"] > 0) != (x["delta_Q_23_9T"] > 0)
                for x in crow if x["delta_Q_23_9P"] is not None
                and x["delta_Q_23_9T"] is not None),
        }
    return rows, summary


def _counts(labels: Sequence[str]) -> dict[str, int]:
    return {x: labels.count(x) for x in sorted(set(labels))}


def analyze(root: Path) -> dict[str, Any]:
    out = root / "combined"
    cmp_out = root / "phase9p_comparison"
    for d in (out, cmp_out):
        if d.exists():
            raise SystemExit(f"{d} exists; never overwritten")
    frozen = load_frozen()
    data = {c: ws.load_condition(root, c) for c in CONDITIONS}
    committed = json.loads((ws.BASELINE_DIR / "paired_summary.json")
                           .read_text("utf-8"))
    recomputed = json.loads(json.dumps(data["baseline"]["paired"],
                                       default=pilot._json_default))
    base_check = all(committed[k] == recomputed[k] for k in
                     ("P1_K_hat_Lap", "P2_K_hat_Q", "P3_paired",
                      "P4_margins", "P5_technical"))
    paired = {c: data[c]["paired"] for c in CONDITIONS}
    lap_rows, lap_tr = ws.transitions(paired, "K_hat_Lap")
    q_rows, q_tr = ws.transitions(paired, "K_hat_Q")
    margin_rows, margin_summary = ws.matched_margins(paired)
    n = pc.PROTOCOL.n
    decomp = {c: ws.decomposition(data[c], n) for c in CONDITIONS}
    ctx = pc._read(root / "preflight" / "density_context.csv")
    pre = json.loads((root / "preflight" / "preflight.json")
                     .read_text("utf-8"))
    per_condition = {c: {
        "w0": pre["frozen"][c]["w0"], "w": pre["frozen"][c]["w"],
        "technical": ws.technical_counts(data[c]),
        "K_hat_Lap": paired[c]["P1_K_hat_Lap"],
        "K_hat_Q": paired[c]["P2_K_hat_Q"],
        "paired": ws.paired_counts(paired[c]),
        "margins": paired[c]["P4_margins"],
        "decomposition_K2_to_K3": decomp[c]["summary"],
        "technical_by_K": ws.technical(data[c]),
        "family_by_K": ws.family_behavior(data[c]),
    } for c in CONDITIONS}
    # historical Phase 9P uncontrolled (w0 = -1) weak / strong, read only
    p_data = {c: ws.load_condition(PHASE9P_ROOT, c) for c in NEW_CONDITIONS}
    p_dec = {c: ws.decomposition(p_data[c], n) for c in NEW_CONDITIONS}
    p_ctx = pc._read(PHASE9P_ROOT / "preflight"
                     / "relational_signal_context.csv")
    cmp_rows, cmp_summary = phase9p_comparison(data, p_data, ctx, p_ctx,
                                               decomp, p_dec)
    new_em = sum(per_condition[c]["technical"]["em_attempted"]
                 for c in NEW_CONDITIONS)
    complete = all(per_condition[c]["technical"]["c_lap_complete_datasets"]
                   == 20 and per_condition[c]["technical"]
                   ["em_complete_datasets"] == 20 for c in NEW_CONDITIONS)
    if not pre["zfx_identical_all"] or not base_check or new_em > NEW_EM_CAP:
        decision = "DENSITY_CONTROLLED_W_SENSITIVITY_BLOCKED"
    elif complete:
        decision = "DENSITY_CONTROLLED_W_SENSITIVITY_CHARACTERIZED"
    else:
        decision = "DENSITY_CONTROLLED_W_SENSITIVITY_PARTIAL"
    summary = {
        "wrapper_version": WRAPPER_VERSION,
        "code_sha": pilot._git_sha(),
        "phase9s2_sha256": frozen["sha256"],
        "p_target": frozen["future_protocol"]["p_target"],
        "baseline_recomputed_summary_matches_committed": base_check,
        "baseline_em_added": 0, "new_em_planned": NEW_EM_CAP,
        "new_em_attempted": new_em,
        "new_em_successful": sum(per_condition[c]["technical"]["em_success"]
                                 for c in NEW_CONDITIONS),
        "retries": 0, "replacements": 0,
        "zfx_identical_all": pre["zfx_identical_all"],
        "density_context": context_summary(ctx),
        "conditions": per_condition,
        "K_hat_Lap_transitions": lap_tr, "K_hat_Q_transitions": q_tr,
        "matched_margin_changes": margin_summary,
        "phase9p_uncontrolled_vs_phase9t_controlled": cmp_summary,
        "DECISION": decision,
        "unit": "the same 20 replicate lineages (same Z/F/X and seeds) "
                "under three w values with the population mean edge "
                "probability matched; not 60 independent datasets",
        "controls_only": "population mean Bernoulli edge probability; not "
                         "the probability spread, quantiles, saturation, "
                         "full distribution or realized density",
        "lineage": "E (experimental prototype; not adoptable for the "
                   "manuscript)",
    }
    out.mkdir(parents=True)
    cmp_out.mkdir(parents=True)
    pilot._write_json(out / "combined_summary.json", summary)
    pilot._write_csv(out / "k_hat_lap_transitions.csv", lap_rows)
    pilot._write_csv(out / "k_hat_q_transitions.csv", q_rows)
    pilot._write_csv(out / "matched_margins.csv", margin_rows)
    pilot._write_csv(out / "decomposition_by_replicate.csv", [
        {"condition": c, **r} for c in CONDITIONS
        for r in decomp[c]["per_replicate"]])
    pilot._write_csv(cmp_out / "paired_by_replicate.csv", cmp_rows)
    pilot._write_json(cmp_out / "summary.json", cmp_summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("preflight", "run", "analyze"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--condition", choices=NEW_CONDITIONS)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        r = preflight(args.root)
        result = {k: r[k] for k in ("zfx_identical_all", "frozen",
                                    "p_target")}
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
