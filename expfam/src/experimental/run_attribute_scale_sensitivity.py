"""Phase 9V (Issue #118): paired attribute-loading-scale sensitivity.

The three conditions (weak f_scale = 1, base sqrt(2), strong 2) are
instantiated only from the committed Phase 9U future_protocol.json
(refuses unless DECISION READY, status FROZEN_NOT_EXECUTED,
future_base_rerun_required true, future_new_em_cap 600). All three run
anew on the Phase 9U paired generator; no historical baseline is reused.

Data adapter (the only new piece between generator and pipeline): during
joint.execute, run_family_selection_pilot.build_dataset is substituted --
for that call only, restored afterwards -- by `paired_dataset`, which
routes the frozen protocol (exact f_scale) to
paired_attribute_scale.generate(data_seed, condition) and wraps the arrays
unchanged (no redraw, normalization, column reordering or family change).
Everything else is the unchanged Phase 9K driver (family selection,
joint K, C_Q, Candidate B, fitted states). No local theta optimization.

Subcommands (each once, into fresh directories)::

    python run_attribute_scale_sensitivity.py preflight --root <root>
    python run_attribute_scale_sensitivity.py run --condition weak --root <root>
    (base, strong likewise)
    python run_attribute_scale_sensitivity.py analyze --root <root>

Lineage E.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import laplace_k_criterion as lk                                   # noqa: E402
import paired_attribute_scale as pa                                # noqa: E402
import run_family_selection_pilot as pilot                         # noqa: E402
import run_joint_family_k_selection as joint                       # noqa: E402
import run_lap_vs_cq_20 as pc                                      # noqa: E402
import run_w_sensitivity as ws                                     # noqa: E402
from data_generator_canonical_mixed import MixedCanonicalDataset   # noqa: E402

WRAPPER_VERSION = "attribute-scale-sensitivity-v1"
REPO_ROOT = ws.REPO_ROOT
DESIGN_DIR = REPO_ROOT / "expfam/results/attribute_scale_design/phase9u_20260928"
CONDITIONS = ("weak", "base", "strong")
NEW_EM_CAP = 600
CONTEXT_KEYS = ("eta_sd_overall", "gaussian_snr_mean", "bernoulli_prob_sd",
                "bernoulli_share_p_gt_0.95", "poisson_rate_q99")


# --------------------------------------------------------------------------
# prerequisite and protocols
# --------------------------------------------------------------------------

def load_design(design_dir: Path = DESIGN_DIR) -> dict[str, Any]:
    design = json.loads((design_dir / "design.json").read_text("utf-8"))
    fp_path = design_dir / "future_protocol.json"
    if design.get("DECISION") != "ATTRIBUTE_SCALE_DESIGN_READY" \
            or not fp_path.is_file():
        raise SystemExit("Phase 9U is not READY: Phase 9V not started")
    fp = json.loads(fp_path.read_text("utf-8"))
    if not (fp.get("status") == "FROZEN_NOT_EXECUTED"
            and fp.get("future_base_rerun_required") is True
            and fp.get("future_new_em_cap") == NEW_EM_CAP
            and set(fp.get("conditions", {})) == set(CONDITIONS)):
        raise SystemExit("Phase 9U prerequisite not satisfied: not started")
    return {"design": design, "future_protocol": fp,
            "sha256": {n: _sha(design_dir / n) for n in
                       ("design.json", "future_protocol.json",
                        "pairing_checks.json")}}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_for(condition: str, design: dict[str, Any] | None = None
                 ) -> joint.JointProtocol:
    design = load_design() if design is None else design
    spec = dict(design["future_protocol"]["conditions"][condition])
    spec.pop("data_generator")
    proto = dataclasses.replace(pc.PROTOCOL, stage=spec["stage"],
                                f_scale=spec["f_scale"])
    if proto.as_json() != spec:
        raise SystemExit(f"{condition}: protocol differs from the frozen "
                         f"future_protocol.json: blocked")
    if proto.f_scale != pa.F_SCALES[condition]:
        raise SystemExit(f"{condition}: f_scale differs from the paired "
                         f"generator: blocked")
    return proto


def condition_of(protocol) -> str:
    matches = [c for c, f in pa.F_SCALES.items() if protocol.f_scale == f]
    if len(matches) != 1:
        raise SystemExit(f"f_scale {protocol.f_scale!r} is not a frozen "
                         f"Phase 9U condition")
    return matches[0]


# --------------------------------------------------------------------------
# data adapter
# --------------------------------------------------------------------------

def _check_common(protocol) -> None:
    ok = (protocol.n == pa.N and protocol.d == pa.D
          and protocol.k_true == pa.K_TRUE
          and tuple(protocol.family_x_list) == pa.FAMILY_X
          and protocol.family_y == "bernoulli"
          and protocol.sigma_x_var == pa.SIGMA_X_VAR
          and protocol.w0 == pa.W0 and protocol.w == pa.W)
    if not ok:
        raise SystemExit("protocol and paired generator disagree: blocked")


def paired_dataset(protocol, replicate) -> MixedCanonicalDataset:
    """The paired generator's arrays, wrapped unchanged for the pipeline."""
    _check_common(protocol)
    condition = condition_of(protocol)
    ds = pa.generate(replicate.data_seed, condition)
    meta = {"generator_version": pa.PAIRED_GENERATOR_VERSION,
            "condition": condition, "f_scale": ds.f_scale,
            "f_row_norms_sq": [float(v) for v in np.sum(ds.F ** 2, axis=1)],
            "sigma_x_var": [pa.SIGMA_X_VAR if f == "gaussian" else None
                            for f in pa.FAMILY_X],
            "seed": replicate.data_seed}
    return MixedCanonicalDataset(Z=ds.Z, F=ds.F, X=ds.X, Y=ds.Y,
                                 metadata=meta)


@contextlib.contextmanager
def paired_data_source() -> Iterator[None]:
    original = pilot.build_dataset
    pilot.build_dataset = paired_dataset
    try:
        yield
    finally:
        pilot.build_dataset = original


def y_hash(y: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(y, dtype=np.float64)
                          .tobytes()).hexdigest()


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------

def preflight(root: Path) -> dict[str, Any]:
    out = root / "preflight"
    if out.exists():
        raise SystemExit(f"{out} exists; a recorded run is never overwritten")
    design = load_design()
    protocols = {c: protocol_for(c, design) for c in CONDITIONS}
    checks, context, hashes = [], [], []
    adapter_ok = True
    for rep in pc.PROTOCOL.replicates:
        sets = pa.generate_all(rep.data_seed)
        checks.append(pa.pairing_check(rep.label, rep.data_seed, sets))
        context += [pa.signal_context(rep.label, sets[c]) for c in CONDITIONS]
        with paired_data_source():
            adapted = {c: pilot.build_dataset(protocols[c], rep)
                       for c in CONDITIONS}
        for c in CONDITIONS:
            adapter_ok = adapter_ok and all(
                np.array_equal(getattr(adapted[c], m), getattr(sets[c], m))
                for m in ("Z", "F", "X", "Y"))
        hashes.append({"replicate": rep.label,
                       **{f"Y_sha256_{c}": y_hash(sets[c].Y)
                          for c in CONDITIONS},
                       "Y_hash_identical": len({y_hash(sets[c].Y)
                                                for c in CONDITIONS}) == 1,
                       "edge_density": float(sets["base"].Y[
                           np.triu_indices(pa.N, k=1)].mean())})
    required = pa.REQUIRED_CHECKS
    all_pass = (all(c[k] for c in checks for k in required)
                and all(h["Y_hash_identical"] for h in hashes) and adapter_ok)
    out.mkdir(parents=True)
    pilot._write_csv(out / "pairing_checks.csv", checks)
    pilot._write_csv(out / "y_hashes.csv", hashes)
    pilot._write_csv(out / "attribute_signal_context.csv", context)
    result = {
        "wrapper_version": WRAPPER_VERSION, "code_sha": pilot._git_sha(),
        "phase9u_sha256": design["sha256"],
        "phase9u_decision": design["design"]["DECISION"],
        "future_protocol_status": design["future_protocol"]["status"],
        "f_scales": {c: protocols[c].f_scale for c in CONDITIONS},
        "pass_counts": {k: sum(bool(c[k]) for c in checks)
                        for k in required},
        "y_hash_identical": sum(h["Y_hash_identical"] for h in hashes),
        "adapter_bit_identical": adapter_ok,
        "max_poisson_rate_strong": max(c["max_poisson_rate_strong"]
                                       for c in checks),
        "replicates": len(checks), "all_pass": all_pass,
        "inference_runs": 0,
    }
    pilot._write_json(out / "preflight.json", result)
    return result


# --------------------------------------------------------------------------
# run one condition
# --------------------------------------------------------------------------

def write_states(path: Path, states, code_sha: str, condition: str,
                 protocol) -> None:
    payload = {
        "code_sha": code_sha, "evaluator_version": lk.EVALUATOR_VERSION,
        "wrapper_version": WRAPPER_VERSION, "condition": condition,
        "f_scale": protocol.f_scale,
        "x_y_provenance": "X and Y are not stored. They are regenerable "
                          "from data_seed with paired_attribute_scale."
                          f"generate(data_seed, {condition!r}) "
                          f"({pa.PAIRED_GENERATOR_VERSION}).",
        "float_format": "Python round-trip float repr (exact float64)",
        "entries": list(states),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")


def run_condition(condition: str, root: Path) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise SystemExit(f"unknown condition {condition!r}")
    pre = json.loads((root / "preflight" / "preflight.json")
                     .read_text("utf-8"))
    if not pre["all_pass"]:
        raise SystemExit("preflight failed: blocked")
    protocol = protocol_for(condition)
    out = root / condition
    rows: list = []
    states: list = []
    code_sha = pilot._git_sha()
    auth = {"gate": f"Phase 9V paired attribute-scale {condition} "
                    f"(f_scale = {protocol.f_scale!r}), one run, <= 200 EM",
            "authorized": True, "authorized_by": "Human",
            "authorized_in": "Issue #118 body (bounded authorization)"}
    try:
        with paired_data_source():
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
            pilot._write_json(out / "data_source.json", {
                "data_generator": f"paired_attribute_scale.generate("
                                  f"data_seed, {condition!r})",
                "paired_generator_version": pa.PAIRED_GENERATOR_VERSION,
                "note": "runinfo.json's generator_version names the joint "
                        "runner's constant; the data actually came from the "
                        "paired generator (see generator_provenance.csv)"})
    summary = pc.paired_summary(rows, pc._read(out / "family_by_k.csv"))
    summary.update(condition=condition, f_scale=protocol.f_scale)
    pilot._write_json(out / "paired_summary.json", summary)
    return summary


# --------------------------------------------------------------------------
# combined analysis
# --------------------------------------------------------------------------

def _dist(values) -> dict[str, Any]:
    return ws._dist(values)


def _counts(labels: Sequence[str]) -> dict[str, int]:
    return {x: labels.count(x) for x in sorted(set(labels))}


def transitions(paired, key):
    by = {c: {d["replicate"]: d for d in paired[c]["datasets"]}
          for c in CONDITIONS}
    rows = []
    for rep in pc.PROTOCOL.replicates:
        path = [by[c][rep.label][key] for c in CONDITIONS]
        rows.append({"replicate": rep.label,
                     **dict(zip(CONDITIONS, path)),
                     "path": "->".join("NA" if k is None else str(k)
                                       for k in path),
                     "all_same": None not in path and len(set(path)) == 1,
                     "weak_to_base": ws.transition_label(path[0], path[1]),
                     "base_to_strong": ws.transition_label(path[1], path[2])})
    return rows, {
        "all_three_same": sum(r["all_same"] for r in rows),
        "weak_to_base": _counts([r["weak_to_base"] for r in rows]),
        "base_to_strong": _counts([r["base_to_strong"] for r in rows]),
        "paths": _counts([r["path"] for r in rows]),
        "over_selection": sum(k is not None and k > pc.K_TRUE
                              for r in rows for k in
                              (r[c] for c in CONDITIONS))}


def matched_margins(paired):
    by = {c: {d["replicate"]: d for d in paired[c]["datasets"]}
          for c in CONDITIONS}
    keys = ("delta_Lap_23", "delta_Q_23", "lap_gap", "q_gap")
    legs = (("weak", "base"), ("base", "strong"))
    rows = []
    for rep in pc.PROTOCOL.replicates:
        row = {"replicate": rep.label}
        for key in keys:
            vals = {c: by[c][rep.label].get(key) for c in CONDITIONS}
            row.update({f"{key}_{c}": vals[c] for c in CONDITIONS})
            for a, b in legs:
                row[f"{key}_change_{a}_to_{b}"] = (
                    vals[b] - vals[a] if vals[a] is not None
                    and vals[b] is not None else None)
        rows.append(row)
    summary = {f"{key}_change_{a}_to_{b}": _dist(
        [r[f"{key}_change_{a}_to_{b}"] for r in rows])
        for key in keys for a, b in legs}
    summary.update({f"{key}_sign_change_{a}_to_{b}": sum(
        r[f"{key}_{a}"] is not None and r[f"{key}_{b}"] is not None
        and (r[f"{key}_{a}"] > 0) != (r[f"{key}_{b}"] > 0) for r in rows)
        for key in ("delta_Lap_23", "delta_Q_23") for a, b in legs})
    return rows, summary


def decomposition_changes(decomp):
    keys = ("lap_fit_gain", "lap_volume_increment", "q_fit_gain",
            "fit_gain_lap_minus_q")
    by = {c: {r["replicate"]: r for r in decomp[c]["per_replicate"]}
          for c in CONDITIONS}
    out = {}
    for a, b in (("weak", "base"), ("base", "strong")):
        for key in keys:
            out[f"{key}_change_{a}_to_{b}"] = _dist([
                by[b][r][key] - by[a][r][key] for r in by[a]
                if key in by[a][r] and key in by[b].get(r, {})])
    return out


def family_changes(data):
    by = {c: {(r["replicate"], int(r["k"])): r["selected_assignment"]
              for r in data[c]["laplace"]} for c in CONDITIONS}
    out = {}
    for a, b in (("weak", "base"), ("base", "strong")):
        keys = set(by[a]) & set(by[b])
        out[f"assignment_changed_{a}_to_{b}"] = sum(by[a][k] != by[b][k]
                                                    for k in keys)
        out[f"compared_{a}_to_{b}"] = len(keys)
    return out


def context_alignment(context, lap_rows, q_rows, margin_rows):
    """Replicate-level table of the prespecified context changes next to
    the selected-K paths and delta_23 changes. Descriptive only."""
    ctx = {(r["replicate"], r["condition"]): r for r in context}
    lap = {r["replicate"]: r["path"] for r in lap_rows}
    q = {r["replicate"]: r["path"] for r in q_rows}
    mm = {r["replicate"]: r for r in margin_rows}
    rows = []
    for rep in pc.PROTOCOL.replicates:
        r = rep.label
        row = {"replicate": r, "K_hat_Lap_path": lap[r], "K_hat_Q_path": q[r]}
        for a, b in (("weak", "base"), ("base", "strong")):
            for key in CONTEXT_KEYS:
                row[f"{key}_change_{a}_to_{b}"] = (
                    float(ctx[(r, b)][key]) - float(ctx[(r, a)][key]))
            for key in ("delta_Lap_23", "delta_Q_23"):
                row[f"{key}_change_{a}_to_{b}"] = \
                    mm[r][f"{key}_change_{a}_to_{b}"]
        rows.append(row)
    return rows


def analyze(root: Path) -> dict[str, Any]:
    out = root / "combined"
    if out.exists():
        raise SystemExit(f"{out} exists; never overwritten")
    design = load_design()
    data = {c: ws.load_condition(root, c) for c in CONDITIONS}
    paired = {c: data[c]["paired"] for c in CONDITIONS}
    lap_rows, lap_tr = transitions(paired, "K_hat_Lap")
    q_rows, q_tr = transitions(paired, "K_hat_Q")
    margin_rows, margin_summary = matched_margins(paired)
    decomp = {c: ws.decomposition(data[c], pc.PROTOCOL.n)
              for c in CONDITIONS}
    context = pc._read(root / "preflight" / "attribute_signal_context.csv")
    pre = json.loads((root / "preflight" / "preflight.json")
                     .read_text("utf-8"))
    ctx_keys = [k for k in context[0] if k not in
                ("replicate", "condition", "singular_values",
                 "poisson_inversion_ok")]
    per_condition = {c: {
        "f_scale": pa.F_SCALES[c],
        "loading_energy": pa.F_SCALES[c] ** 2 * pa.K_TRUE / pa.D,
        "technical": ws.technical_counts(data[c]),
        "K_hat_Lap": paired[c]["P1_K_hat_Lap"],
        "K_hat_Q": paired[c]["P2_K_hat_Q"],
        "paired": ws.paired_counts(paired[c]),
        "margins": paired[c]["P4_margins"],
        "decomposition_K2_to_K3": decomp[c]["summary"],
        "technical_by_K": ws.technical(data[c]),
        "family_by_K": ws.family_behavior(data[c]),
        "signal_context": {k: _dist([float(r[k]) for r in context
                                     if r["condition"] == c])
                           for k in ctx_keys},
    } for c in CONDITIONS}
    new_em = sum(per_condition[c]["technical"]["em_attempted"]
                 for c in CONDITIONS)
    complete = all(per_condition[c]["technical"]["c_lap_complete_datasets"]
                   == 20 and per_condition[c]["technical"]
                   ["em_complete_datasets"] == 20 for c in CONDITIONS)
    if not pre["all_pass"] or new_em > NEW_EM_CAP:
        decision = "ATTRIBUTE_SCALE_SENSITIVITY_BLOCKED"
    elif complete:
        decision = "ATTRIBUTE_SCALE_SENSITIVITY_CHARACTERIZED"
    else:
        decision = "ATTRIBUTE_SCALE_SENSITIVITY_PARTIAL"
    summary = {
        "wrapper_version": WRAPPER_VERSION, "code_sha": pilot._git_sha(),
        "phase9u_sha256": design["sha256"],
        "new_em_planned": NEW_EM_CAP, "new_em_attempted": new_em,
        "new_em_successful": sum(per_condition[c]["technical"]["em_success"]
                                 for c in CONDITIONS),
        "retries": 0, "replacements": 0, "historical_baseline_reused": False,
        "preflight_all_pass": pre["all_pass"],
        "y_hash_identical": pre["y_hash_identical"],
        "conditions": per_condition,
        "K_hat_Lap_transitions": lap_tr, "K_hat_Q_transitions": q_tr,
        "matched_margin_changes": margin_summary,
        "decomposition_changes": decomposition_changes(decomp),
        "family_assignment_changes": family_changes(data),
        "DECISION": decision,
        "unit": "the same 20 paired replicate lineages (shared Z, Q, Y and "
                "search/refit seeds) under three attribute loading scales",
        "interpretation": "attribute-loading-scale sensitivity; not a pure "
                          "or family-independent attribute-information "
                          "effect",
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
    pilot._write_csv(out / "context_alignment.csv", context_alignment(
        context, lap_rows, q_rows, margin_rows))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("preflight", "run", "analyze"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--condition", choices=CONDITIONS)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        r = preflight(args.root)
        result = {k: r[k] for k in ("all_pass", "pass_counts",
                                    "y_hash_identical",
                                    "adapter_bit_identical")}
    elif args.command == "run":
        if args.condition is None:
            parser.error("--condition is required for run")
        s = run_condition(args.condition, args.root)
        result = {k: s[k] for k in ("c_lap_complete_datasets", "P1_K_hat_Lap",
                                    "P2_K_hat_Q", "P5_technical")}
    else:
        s = analyze(args.root)
        result = {k: s[k] for k in ("new_em_attempted", "preflight_all_pass",
                                    "DECISION")}
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
