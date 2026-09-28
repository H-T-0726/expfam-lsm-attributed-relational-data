"""Phase 9X (Issue #122): matched-signal K_true sensitivity.

Conditions are read only from the merged Phase 9W future_protocol.json
(frozen, not authorized there). Phase 9X's own execution authority is the
separate record `execution_authorization.json` (Issue #122). New runs:
K_true 1, 2, 4 on rep01..rep10 with the unchanged Phase 9K driver; K_true
3 is the historical Phase 9K rep01..rep10 anchor, read only (0 new EM).

All outcomes are computed relative to each condition's own K_true by the
generic functions below; the Phase 9K helpers that hard-code K_TRUE = 3
(category, paired_summary's *_K3 counts) are not used. No local theta
optimization, no solver change. Lineage E.

Subcommands (each once, into fresh directories)::

    python run_matched_k_true.py preflight --root <root>
    python run_matched_k_true.py run --k-true 1 --root <root>   (2, 4 likewise)
    python run_matched_k_true.py analyze --root <root>
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
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
import run_laplace_pilot as lp                                     # noqa: E402
import run_w_sensitivity as ws                                     # noqa: E402

RUNNER_VERSION = "matched-k-true-sensitivity-v1"
REPO_ROOT = ws.REPO_ROOT
DESIGN_DIR = REPO_ROOT / "expfam/results/matched_k_true_design/phase9w_20260928"
PHASE9K_DIR = ws.BASELINE_DIR
PHASE9W_MERGE = "ba399d5779375540e3b50aa397d81598fbec736a"
K_TRUES = (1, 2, 3, 4)
NEW_K_TRUES = (1, 2, 4)
CANDIDATE_K = (1, 2, 3, 4, 5)
REPLICATES = tuple(f"rep{r:02d}" for r in range(1, 11))
NEW_EM_CAP = 300
HASHED = ("design.json", "future_protocol.json", "calibration_by_k.json",
          "k3_anchor_compatibility.json")
NON_PROTOCOL_KEYS = ("role", "execution_authorization", "start_policy")
ADJACENT = {1: (2,), 2: (1, 3), 3: (2, 4), 4: (3, 5)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# prerequisite, protocols, authorization
# --------------------------------------------------------------------------

def load_frozen(design_dir: Path = DESIGN_DIR) -> dict[str, Any]:
    design = json.loads((design_dir / "design.json").read_text("utf-8"))
    fp = json.loads((design_dir / "future_protocol.json").read_text("utf-8"))
    ok = (design.get("DECISION") == "MATCHED_K_TRUE_DESIGN_READY"
          and fp.get("status") == "FROZEN_NOT_EXECUTED"
          and fp.get("execution_authorized") is False
          and fp.get("K3_anchor_reusable") is True
          and fp.get("future_new_em_cap") == NEW_EM_CAP
          and fp.get("candidate_K") == list(CANDIDATE_K)
          and fp.get("replicates") == list(REPLICATES)
          and fp.get("new_K_true") == list(NEW_K_TRUES)
          and fp.get("start") == "start_B only"
          and sorted(fp.get("conditions", {})) == [f"K{k}" for k in K_TRUES])
    if not ok:
        raise SystemExit("Phase 9W prerequisite not satisfied: Phase 9X "
                         "not started")
    return {"design": design, "future_protocol": fp,
            "sha256": {n: _sha(design_dir / n) for n in HASHED}}


def protocol_for(k: int, frozen: dict[str, Any] | None = None):
    frozen = load_frozen() if frozen is None else frozen
    spec = frozen["future_protocol"]["conditions"][f"K{k}"]
    reps = tuple(r for r in pc.PROTOCOL.replicates if r.label in REPLICATES)
    proto = dataclasses.replace(pc.PROTOCOL, stage=spec["stage"],
                                k_true=spec["k_true"],
                                f_scale=spec["f_scale"], w=spec["w"],
                                w0=spec["w0"], replicates=reps)
    mine = proto.as_json()
    if {k2: v for k2, v in mine.items() if k2 not in NON_PROTOCOL_KEYS} != \
            {k2: v for k2, v in spec.items() if k2 not in NON_PROTOCOL_KEYS}:
        raise SystemExit(f"K{k}: protocol differs from the frozen Phase 9W "
                         f"future_protocol.json: blocked")
    if spec["starts"] != [{"label": "start_B", "ambiguous_start": "bernoulli"}]:
        raise SystemExit(f"K{k}: not start_B only: blocked")
    return proto


def authorization_record(frozen: dict[str, Any]) -> dict[str, Any]:
    return {"issue": 122, "authorized": True, "authorized_by": "Human",
            "authorization_scope": "Phase 9X only",
            "gate": "Phase 9X matched-signal K_true sensitivity",
            "authorized_in": "Issue #122 body (bounded authorization)",
            "phase9w_merge_sha": PHASE9W_MERGE,
            "phase9w_artifact_sha256": frozen["sha256"],
            "new_K_true": list(NEW_K_TRUES), "K3_rerun": False,
            "new_em_cap": NEW_EM_CAP,
            "replicates": list(REPLICATES),
            "not_authorized": ["rerunning K_true=3", "K_true=5",
                               "rep11..rep20", "new signal matching",
                               "per-family ablation",
                               "local theta optimization",
                               "solver tuning", "extra starts"],
            "note": "Phase 9W future_protocol.json stays "
                    "execution_authorized=false; this record is the only "
                    "Phase 9X authority"}


# --------------------------------------------------------------------------
# generic K_true outcome logic
# --------------------------------------------------------------------------

def category(k_hat: int | None, k_true: int) -> str | None:
    if k_hat is None:
        return None
    return ("exact" if k_hat == k_true else
            "under" if k_hat < k_true else "over")


def errors(k_hat: int | None, k_true: int) -> tuple[int | None, int | None]:
    if k_hat is None:
        return None, None
    return k_hat - k_true, abs(k_hat - k_true)


def paired_labels(k_lap: int | None, k_q: int | None, k_true: int
                  ) -> dict[str, bool] | None:
    if k_lap is None or k_q is None:
        return None
    lap, q = k_lap == k_true, k_q == k_true
    return {"both_exact": lap and q, "Lap_exact_only": lap and not q,
            "Q_exact_only": q and not lap, "neither_exact": not lap and not q,
            "same_selected_K": k_lap == k_q,
            "different_selected_K": k_lap != k_q}


def contrasts(curve: dict[int, float], k_true: int) -> dict[str, float]:
    """Delta_C(true, k) = C(K_true) - C(k); negative favours K_true."""
    return {f"delta_true_minus_K{k}": curve[k_true] - curve[k]
            for k in ADJACENT[k_true]}


def dataset_rows(k_true: int, laplace_rows, family_rows
                 ) -> list[dict[str, Any]]:
    out = []
    for rep in REPLICATES:
        d = lp.dataset_summary(laplace_rows, family_rows, rep)
        row: dict[str, Any] = {"K_true": k_true, "replicate": rep,
                               "em_complete": d["em_complete"],
                               "c_lap_complete": d["c_lap_complete"]}
        for crit, key, curve_key, ok in (
                ("Lap", "K_hat_Lap", "C_Lap", d["c_lap_complete"]),
                ("Q", "K_hat_Q", "C_Q", d["K_hat_Q"] is not None)):
            k_hat = d[key]
            row[key] = k_hat
            row[f"category_{crit}"] = category(k_hat, k_true)
            row[f"signed_error_{crit}"], row[f"abs_error_{crit}"] = \
                errors(k_hat, k_true)
            if ok:
                curve = d[curve_key]
                best, second, gap = pc.best_second(curve)
                row.update({f"best_{crit}": best, f"second_{crit}": second,
                            f"gap_{crit}": gap})
                row.update({f"{crit}_{k}": v
                            for k, v in contrasts(curve, k_true).items()})
        row["paired"] = paired_labels(d["K_hat_Lap"], d["K_hat_Q"], k_true)
        out.append(row)
    return out


def _dist(values) -> dict[str, Any]:
    return ws._dist([v for v in values if v is not None])


def condition_summary(k_true: int, rows: Sequence[dict[str, Any]]
                      ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "K_true": k_true, "datasets": len(rows),
        "em_complete": sum(r["em_complete"] for r in rows),
        "c_lap_complete": sum(r["c_lap_complete"] for r in rows)}
    for crit, key in (("Lap", "K_hat_Lap"), ("Q", "K_hat_Q")):
        ks = [r[key] for r in rows if r[key] is not None]
        cats = [r[f"category_{crit}"] for r in rows
                if r[f"category_{crit}"] is not None]
        out[crit] = {
            "denominator": len(ks),
            "counts": {k: ks.count(k) for k in CANDIDATE_K},
            **{c: cats.count(c) for c in ("exact", "under", "over")},
            "signed_error": _dist([r[f"signed_error_{crit}"] for r in rows]),
            "abs_error": _dist([r[f"abs_error_{crit}"] for r in rows]),
            "gap": _dist([r.get(f"gap_{crit}") for r in rows]),
            **{f"delta_true_minus_K{k}": _dist(
                [r.get(f"{crit}_delta_true_minus_K{k}") for r in rows])
               for k in ADJACENT[k_true]}}
    paired = [r["paired"] for r in rows if r["paired"] is not None]
    out["paired"] = {"denominator": len(paired),
                     **{lab: sum(p[lab] for p in paired) for lab in
                        ("both_exact", "Lap_exact_only", "Q_exact_only",
                         "neither_exact", "same_selected_K",
                         "different_selected_K")}}
    gaps = sorted((r for r in rows if r.get("gap_Lap") is not None),
                  key=lambda r: r["gap_Lap"])
    out["five_smallest_lap_gaps"] = [
        {"replicate": r["replicate"], "gap": r["gap_Lap"],
         "best_k": r["best_Lap"], "second_k": r["second_Lap"]}
        for r in gaps[:5]]
    return out


def confusion(rows_by_k: dict[int, list[dict[str, Any]]], key: str):
    table = []
    for k_true in K_TRUES:
        rows = rows_by_k[k_true]
        ks = [r[key] for r in rows]
        table.append({"K_true": k_true,
                      **{f"K_hat_{k}": ks.count(k) for k in CANDIDATE_K},
                      "NA": ks.count(None), "total": len(rows)})
    return table


# --------------------------------------------------------------------------
# data loading (new runs and the read-only K3 anchor)
# --------------------------------------------------------------------------

def _filter(rows, reps=REPLICATES):
    return [r for r in rows if r["replicate"] in reps]


def load_condition(k_true: int, root: Path) -> dict[str, Any]:
    src = PHASE9K_DIR if k_true == 3 else root / f"K{k_true}"
    runinfo = json.loads((src / "runinfo.json").read_text("utf-8"))
    summary = json.loads((src / "summary.json").read_text("utf-8"))
    warn = {key: v for key, v in
            summary.get("candidate_convergence_warnings_by_k", {}).items()
            if key.split("/")[0] in REPLICATES}
    ledger = _filter(pc._read(src / "execution_ledger.csv"))
    return {"source": "historical_phase9k" if k_true == 3 else "phase9x_new",
            "laplace": _filter(pc._read(src / "laplace_by_k.csv")),
            "family": _filter(pc._read(src / "family_by_k.csv")),
            "cq": _filter(pc._read(src / "cq_decomposition.csv")),
            "ledger": ledger, "runinfo": runinfo,
            "joint_summary": {"candidate_convergence_warnings_by_k": warn},
            "em_attempted_in_scope": len(ledger),
            "em_success_in_scope": sum(r["status"] == "SUCCESS"
                                       for r in ledger)}


def family_summary(data) -> dict[str, Any]:
    out = {}
    true = "|".join(pc.PROTOCOL.family_x_list)
    for k in CANDIDATE_K:
        lap = [r for r in data["laplace"] if int(r["k"]) == k]
        fam = [r for r in data["family"] if int(r["k"]) == k]
        conf: dict[str, dict[str, int]] = {}
        for r in fam:
            g = conf.setdefault(r["family_x_true"], {})
            g[r["selected_family"]] = g.get(r["selected_family"], 0) + 1
        out[f"K{k}"] = {"refits": len(lap),
                        "exact_true_assignment": sum(
                            r["selected_assignment"] == true for r in lap),
                        "selected_by_true_family": conf}
    return out


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------

def preflight(root: Path) -> dict[str, Any]:
    out = root / "preflight"
    if out.exists():
        raise SystemExit(f"{out} exists; never overwritten")
    frozen = load_frozen()
    protocols = {k: protocol_for(k, frozen) for k in K_TRUES}
    auth = authorization_record(frozen)
    safety = []
    for k in NEW_K_TRUES:
        p = protocols[k]
        for rep in p.replicates:
            ds = pilot.build_dataset(p, rep)            # GeneratorStop -> stop
            fam = np.array(p.family_x_list)
            b, pois = np.flatnonzero(fam == "bernoulli"), \
                np.flatnonzero(fam == "poisson")
            rate = np.exp((ds.Z @ ds.F.T)[:, pois])
            safety.append({
                "K_true": k, "replicate": rep.label,
                "finite": bool(all(np.all(np.isfinite(a))
                                   for a in (ds.Z, ds.F, ds.X, ds.Y))),
                "support_ok": bool(
                    np.all((ds.X[:, b] == 0) | (ds.X[:, b] == 1))
                    and np.all(ds.X[:, pois] >= 0)
                    and np.all(ds.X[:, pois] == np.floor(ds.X[:, pois]))),
                "rank_F": int(np.linalg.matrix_rank(ds.F)),
                "max_poisson_rate": float(rate.max()),
                "poisson_safe": bool(rate.max()
                                     <= ds.metadata["poisson_lambda_max"]),
                "frozen_values_exact": bool(
                    ds.metadata["f_scale"] == p.f_scale
                    and ds.metadata["w"] == p.w
                    and ds.metadata["w0"] == p.w0)})
    anchor = []
    for rep in protocols[3].replicates:
        new = pilot.build_dataset(protocols[3], rep)
        hist = pilot.build_dataset(pc.PROTOCOL, rep)
        anchor.append({"replicate": rep.label,
                       **{m: bool(np.array_equal(getattr(new, m),
                                                 getattr(hist, m)))
                          for m in ("Z", "F", "X", "Y")}})
    needed = ("laplace_by_k.csv", "family_by_k.csv", "cq_decomposition.csv",
              "execution_ledger.csv", "summary.json", "runinfo.json",
              "fitted_states.json")
    anchor_files = {n: _sha(PHASE9K_DIR / n) for n in needed
                    if (PHASE9K_DIR / n).is_file()}
    k3 = load_condition(3, root)
    k3_rows = {(r["replicate"], int(r["k"])) for r in k3["laplace"]}
    anchor_complete = all((rep, k) in k3_rows for rep in REPLICATES
                          for k in CANDIDATE_K)
    # generic-K logic self-check (would fail if K_true were hard-coded to 3)
    generic_ok = (category(3, 1) == "over" and category(1, 1) == "exact"
                  and category(2, 4) == "under"
                  and paired_labels(3, 3, 2)["neither_exact"]
                  and paired_labels(2, 3, 2)["Lap_exact_only"])
    all_pass = (all(s["finite"] and s["support_ok"] and s["poisson_safe"]
                    and s["frozen_values_exact"]
                    and s["rank_F"] == s["K_true"] for s in safety)
                and len(safety) == 30
                and all(all(v for key, v in a.items() if key != "replicate")
                        for a in anchor)
                and len(anchor_files) == len(needed) and anchor_complete
                and generic_ok)
    out.mkdir(parents=True)
    pilot._write_json(out / "execution_authorization.json", auth)
    pilot._write_csv(out / "new_condition_safety.csv", safety)
    pilot._write_csv(out / "k3_anchor_identity.csv", anchor)
    result = {"runner_version": RUNNER_VERSION, "code_sha": pilot._git_sha(),
              "phase9w_merge_sha": PHASE9W_MERGE,
              "phase9w_sha256": frozen["sha256"],
              "phase9k_anchor_sha256": anchor_files,
              "frozen": {f"K{k}": {"f_scale": protocols[k].f_scale,
                                   "w": protocols[k].w, "w0": protocols[k].w0}
                         for k in K_TRUES},
              "new_condition_datasets": len(safety),
              "k3_anchor_identity": {m: sum(a[m] for a in anchor)
                                     for m in ("Z", "F", "X", "Y")},
              "k3_anchor_rows_complete": anchor_complete,
              "generic_k_logic_ok": generic_ok,
              "all_pass": all_pass, "inference_runs": 0}
    pilot._write_json(out / "preflight.json", result)
    return result


# --------------------------------------------------------------------------
# run one new condition
# --------------------------------------------------------------------------

def run_condition(k_true: int, root: Path) -> dict[str, Any]:
    if k_true not in NEW_K_TRUES:
        raise SystemExit(f"K_true={k_true} is not a new Phase 9X condition "
                         f"(K3 is the historical anchor; never rerun)")
    pre = json.loads((root / "preflight" / "preflight.json")
                     .read_text("utf-8"))
    if not pre["all_pass"]:
        raise SystemExit("preflight failed: blocked")
    auth = json.loads((root / "preflight" / "execution_authorization.json")
                      .read_text("utf-8"))
    protocol = protocol_for(k_true)
    out = root / f"K{k_true}"
    rows: list = []
    states: list = []
    code_sha = pilot._git_sha()
    try:
        joint.execute(out, protocol=protocol,
                      driver=pc.make_driver(rows, states),
                      authorization={**auth, "gate": f"{auth['gate']}, "
                                                     f"K_true={k_true}"})
    finally:
        if out.is_dir():
            if rows:
                pilot._write_csv(out / "laplace_by_k.csv", rows)
            if states:
                with (out / "fitted_states.json").open("w",
                                                      encoding="utf-8") as h:
                    json.dump({"code_sha": code_sha,
                               "evaluator_version": lk.EVALUATOR_VERSION,
                               "runner_version": RUNNER_VERSION,
                               "K_true": k_true, "f_scale": protocol.f_scale,
                               "w": protocol.w, "w0": protocol.w0,
                               "x_y_provenance": "regenerable with "
                               "run_family_selection_pilot.build_dataset("
                               f"run_matched_k_true.protocol_for({k_true}), "
                               "replicate)",
                               "entries": states}, h)
                    h.write("\n")
    fam = pc._read(out / "family_by_k.csv")
    return condition_summary(k_true, dataset_rows(k_true, rows, fam))


# --------------------------------------------------------------------------
# combined analysis
# --------------------------------------------------------------------------

def analyze(root: Path) -> dict[str, Any]:
    out = root / "combined"
    anchor_out = root / "K3_anchor"
    for d in (out, anchor_out):
        if d.exists():
            raise SystemExit(f"{d} exists; never overwritten")
    pre = json.loads((root / "preflight" / "preflight.json")
                     .read_text("utf-8"))
    frozen = load_frozen()
    data = {k: load_condition(k, root) for k in K_TRUES}
    rows_by_k = {k: dataset_rows(k, data[k]["laplace"], data[k]["family"])
                 for k in K_TRUES}
    per_k = {}
    for k in K_TRUES:
        s = condition_summary(k, rows_by_k[k])
        statuses = [r["laplace_status"] for r in data[k]["laplace"]]
        s.update({
            "source": data[k]["source"],
            "em_attempted_in_scope": data[k]["em_attempted_in_scope"],
            "em_success_in_scope": data[k]["em_success_in_scope"],
            "new_em": 0 if k == 3 else data[k]["em_attempted_in_scope"],
            "candidate_b_status": {st: statuses.count(st) for st in
                                   ("OK", "NOT_STATIONARY", "HESSIAN_NOT_PD",
                                    "EVALUATION_ERROR")},
            "technical_by_model_K": ws.technical(data[k]),
            # No K2->K3 (or K3->K4) decomposition: not a prespecified
            # Phase 9X output (Issue #122). cq_decomposition.csv stays a raw
            # runner artifact and is not used for interpretation.
            "family_by_model_K": family_summary(data[k])})
        per_k[f"K{k}"] = s
    context = pc._read(DESIGN_DIR / "zero_inference_context.csv")
    ctx_keys = [c for c in context[0] if c not in ("K_true", "replicate",
                                                    "finite", "supports_ok")]
    per_ctx = {f"K{k}": {c: _dist([float(r[c]) for r in context
                                   if int(r["K_true"]) == k])
                         for c in ctx_keys} for k in K_TRUES}
    new_em = sum(per_k[f"K{k}"]["new_em"] for k in NEW_K_TRUES)
    complete = all(per_k[f"K{k}"]["c_lap_complete"] == 10
                   and per_k[f"K{k}"]["em_complete"] == 10
                   for k in NEW_K_TRUES)
    anchor_ok = (pre["all_pass"] and per_k["K3"]["c_lap_complete"] == 10)
    if not pre["all_pass"] or not anchor_ok or new_em > NEW_EM_CAP:
        decision = "MATCHED_K_TRUE_SENSITIVITY_BLOCKED"
    elif complete:
        decision = "MATCHED_K_TRUE_SENSITIVITY_CHARACTERIZED"
    else:
        decision = "MATCHED_K_TRUE_SENSITIVITY_PARTIAL"
    summary = {
        "runner_version": RUNNER_VERSION, "code_sha": pilot._git_sha(),
        "phase9w_merge_sha": PHASE9W_MERGE,
        "phase9w_sha256": frozen["sha256"],
        "new_em_planned": NEW_EM_CAP, "new_em_attempted": new_em,
        "new_em_successful": sum(per_k[f"K{k}"]["em_success_in_scope"]
                                 for k in NEW_K_TRUES),
        "K3_new_em": 0, "retries": 0, "replacements": 0,
        "conditions": per_k, "matched_signal_context": per_ctx,
        "confusion_Lap": confusion(rows_by_k, "K_hat_Lap"),
        "confusion_Q": confusion(rows_by_k, "K_hat_Q"),
        "DECISION": decision,
        "interpretation": "matched-signal K_true sensitivity on rep01..rep10; "
                          "not a pure latent-dimension effect, consistency "
                          "study or general recovery probability",
        "lineage": "E (experimental prototype; not adoptable for the "
                   "manuscript)",
    }
    anchor_out.mkdir(parents=True)
    pilot._write_json(anchor_out / "anchor_provenance.json", {
        "source": "historical_phase9k", "phase9k_dir": str(
            PHASE9K_DIR.relative_to(REPO_ROOT)),
        "phase9k_sha256": pre["phase9k_anchor_sha256"],
        "replicates": list(REPLICATES), "new_em": 0,
        "note": "derived read-only from Phase 9K rep01..rep10; not mixed "
                "with Phase 9K's 20-replicate summaries"})
    pilot._write_csv(anchor_out / "anchor_dataset_rows.csv", [
        {k: v for k, v in r.items() if k != "paired"} | (r["paired"] or {})
        for r in rows_by_k[3]])
    out.mkdir(parents=True)
    pilot._write_json(out / "combined_summary.json", summary)
    pilot._write_csv(out / "dataset_rows.csv", [
        {k: v for k, v in r.items() if k != "paired"} | (r["paired"] or {})
        for k in K_TRUES for r in rows_by_k[k]])
    pilot._write_csv(out / "confusion_Lap.csv", summary["confusion_Lap"])
    pilot._write_csv(out / "confusion_Q.csv", summary["confusion_Q"])
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("preflight", "run", "analyze"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--k-true", type=int, choices=NEW_K_TRUES)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        r = preflight(args.root)
        result = {k: r[k] for k in ("all_pass", "k3_anchor_identity",
                                    "k3_anchor_rows_complete",
                                    "generic_k_logic_ok")}
    elif args.command == "run":
        if args.k_true is None:
            parser.error("--k-true is required for run")
        s = run_condition(args.k_true, args.root)
        result = {k: s[k] for k in ("K_true", "c_lap_complete", "Lap", "Q")}
        result["Lap"] = {k: result["Lap"][k] for k in
                         ("counts", "exact", "under", "over")}
        result["Q"] = {k: result["Q"][k] for k in
                       ("counts", "exact", "under", "over")}
    else:
        s = analyze(args.root)
        result = {k: s[k] for k in ("new_em_attempted", "DECISION")}
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
