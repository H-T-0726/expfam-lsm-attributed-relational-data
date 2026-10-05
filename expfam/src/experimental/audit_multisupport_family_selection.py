"""Artifact consistency audit for a multi-support family-selection run.

Reads the run directory only (does not import the runner's execution path
beyond small CSV helpers) and writes ``audit_report.json`` at the run root
plus a copy in ``audit/``.  Severity: BLOCKER makes the run's results
unusable as reported; WARNING must be disclosed; DIAGNOSTIC is information.

    python audit_multisupport_family_selection.py --run-dir DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from run_multisupport_family_selection import (  # noqa: E402
    PHASES, atomic_write_json, dataset_key, enumerate_cases, read_csv,
)

# Files whose change between the init commit and a phase commit would change
# what the experiment computes.
EXPERIMENT_PATH_FILES = [
    "expfam/src/experimental/family_specs.py",
    "expfam/src/experimental/model_dual_expfam_generic.py",
    "expfam/src/experimental/family_selection_multisupport.py",
    "expfam/src/experimental/data_generator_multisupport.py",
    "expfam/src/experimental/difficulty_calibration.py",
    "expfam/src/experimental/run_multisupport_family_selection.py",
    "expfam/src/experimental/family_selection.py",
    "expfam/src/experimental/model_dual_expfam_consistent.py",
    "expfam/src/experimental/model_dual_expfam_percolumn.py",
    "expfam/src/experimental/model_dual_expfam_masked.py",
    "expfam/src/experimental/objective_consistent_numerics.py",
    "expfam/src/experimental/em_runner.py",
    "expfam/src/experimental/eval_utils.py",
    "expfam/src/experimental/data_generator_canonical.py",
    "expfam/src/model_dual_expfam_fixed.py",
    "expfam/src/model_dual_expfam.py",
    "expfam/src/model_expfam.py",
    "reproduction/src/model.py",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    a = ap.parse_args(argv)
    run = a.run_dir
    t = run / "tables"
    findings = []

    def add(sev, check, detail):
        findings.append({"severity": sev, "check": check, "detail": detail})

    checks = {}
    info = json.loads((run / "runinfo.json").read_text(encoding="utf-8"))
    # protocol hash
    psha = sha(run / "protocol.json")
    checks["protocol_hash_unchanged"] = psha == info["protocol_sha256"]
    if not checks["protocol_hash_unchanged"]:
        add("BLOCKER", "protocol_hash", "protocol.json changed after freeze")

    completed = read_csv(run / "completed_cases.csv")
    progress = read_csv(run / "progress.csv")
    phases_run = sorted({r["phase"] for r in completed})
    expected = {}
    for ph in phases_run:
        for c in enumerate_cases(ph):
            expected[c["case_id"]] = c
    comp_ids = [r["case_id"] for r in completed]
    dup = [k for k, v in Counter(comp_ids).items() if v > 1]
    missing = sorted(set(expected) - set(comp_ids))
    unexpected_ids = sorted(set(comp_ids) - set(expected))
    checks.update({"phases_run": phases_run, "expected_cases": len(expected),
                   "completed_cases": len(set(comp_ids)),
                   "duplicate_completed": dup, "missing_cases": missing,
                   "unexpected_case_ids": unexpected_ids})
    if dup:
        add("BLOCKER", "duplicate_completed", dup[:20])
    if missing:
        add("WARNING", "missing_cases", f"{len(missing)} expected cases not completed "
                                        f"(phase not finished?): {missing[:10]}")
    # hidden retry: a case attempted more than once
    prog_dup = [k for k, v in Counter(r["case_id"] for r in progress).items() if v > 1]
    checks["hidden_retry_cases"] = prog_dup
    if prog_dup:
        add("BLOCKER", "hidden_retry", prog_dup[:20])
    # case files
    bad_hash = [r["case_id"] for r in completed
                if not (run / r["case_file"]).exists() or sha(run / r["case_file"]) != r["case_file_sha256"]]
    checks["case_file_hash_mismatch"] = bad_hash
    if bad_hash:
        add("BLOCKER", "case_file_hash", bad_hash[:20])

    # datasets / seeds
    manifest = read_csv(t / "dataset_manifest.csv")
    exp_keys = {dataset_key(c["stage"], c["exp_block"], c["group"], c["difficulty"],
                            c["n"], c["replicate"]) for c in expected.values()}
    man_paths = [r["dataset_path"] for r in manifest]
    checks["datasets_expected"] = len(exp_keys)
    checks["datasets_in_manifest"] = len(set(man_paths))
    checks["duplicate_manifest_rows"] = [k for k, v in Counter(man_paths).items() if v > 1]
    if checks["duplicate_manifest_rows"]:
        add("BLOCKER", "duplicate_manifest", checks["duplicate_manifest_rows"][:10])
    if len(set(man_paths)) != len(exp_keys):
        add("WARNING", "manifest_count", f"{len(set(man_paths))} manifest datasets vs "
                                         f"{len(exp_keys)} expected")
    bad_ds = [r["dataset_path"] for r in manifest
              if not (run / r["dataset_path"]).exists()
              or sha(run / r["dataset_path"]) != r["dataset_sha256"]]
    checks["dataset_hash_mismatch"] = bad_ds
    if bad_ds:
        add("BLOCKER", "dataset_hash", bad_ds[:10])
    seeds = [r["data_seed"] for r in manifest]
    checks["duplicate_data_seeds"] = [k for k, v in Counter(seeds).items() if v > 1]
    if checks["duplicate_data_seeds"]:
        add("BLOCKER", "seed_duplicate", checks["duplicate_data_seeds"][:10])
    sm = json.loads((run / "seed_manifest.json").read_text(encoding="utf-8"))
    checks["seed_manifest_unique"] = sm["n_unique_data_seeds"] == len(sm["datasets"])
    if not checks["seed_manifest_unique"]:
        add("BLOCKER", "seed_manifest", "non-unique data seeds in the frozen manifest")

    # fits
    fits = read_csv(t / "fit_results.csv")
    fit_ids = [r["case_id"] for r in fits]
    checks["duplicate_fit_rows"] = [k for k, v in Counter(fit_ids).items() if v > 1]
    exp_fits = {k for k, c in expected.items() if c["method"] != "SELECTOR_ONLY"}
    checks["missing_fit_rows"] = sorted(exp_fits & set(comp_ids) - set(fit_ids))
    for key in ("duplicate_fit_rows", "missing_fit_rows"):
        if checks[key]:
            add("BLOCKER", key, checks[key][:10])
    retry = [r["case_id"] for r in fits
             if any(str(r.get(k, "0")) not in ("0", "") for k in
                    ("retry_count", "replacement_count", "seed_rescue_count"))]
    checks["retry_or_rescue_fits"] = retry
    if retry:
        add("BLOCKER", "retry_or_rescue", retry[:10])
    nonfinite_fit = [r["case_id"] for r in fits if r["status"] == "SUCCESS"
                     and not (finite(r["final_Q"]) and finite(r["rmse_Z"]))]
    checks["nonfinite_successful_fit"] = nonfinite_fit
    if nonfinite_fit:
        add("BLOCKER", "nonfinite_fit", nonfinite_fit[:10])
    failed_fits = [r["case_id"] for r in fits if r["status"] != "SUCCESS"]
    checks["failed_fits"] = failed_fits
    if failed_fits:
        add("WARNING", "failed_fits", f"{len(failed_fits)} failed fits (kept as outcomes)")
    # same seed policy: Oracle / SupportOnly / Auto refit share refit seed per dataset
    by_ds = defaultdict(set)
    for r in fits:
        if r.get("fit_seed"):
            by_ds[r["case_id"].rsplit("__", 1)[0]].add(r["fit_seed"])
    checks["datasets_with_inconsistent_fit_seed"] = [k for k, v in by_ds.items() if len(v) > 1]
    if checks["datasets_with_inconsistent_fit_seed"]:
        add("BLOCKER", "fit_seed_policy", checks["datasets_with_inconsistent_fit_seed"][:10])
    bt = [(r["case_id"], r.get("refit_estep_backtracks"), r.get("exploration_estep_backtracks"))
          for r in fits if str(r.get("refit_estep_backtracks", "0")) not in ("0", "")
          or str(r.get("exploration_estep_backtracks", "")) not in ("0", "")]
    checks["fits_with_estep_guard_halvings"] = len(bt)
    if bt:
        add("DIAGNOSTIC", "estep_guard", f"{len(bt)} fits used the COM-Poisson E-step "
                                          f"domain guard (halvings counted): {bt[:5]}")

    # selection rows
    sel = read_csv(t / "selection_results.csv")
    keys = [(r["case_id"], r["column"]) for r in sel]
    checks["duplicate_selection_rows"] = len(keys) - len(set(keys))
    if checks["duplicate_selection_rows"]:
        add("BLOCKER", "duplicate_selection_rows", checks["duplicate_selection_rows"])
    sel_cases = {k for k, c in expected.items()
                 if c["method"] in ("SELECTOR_ONLY", "AUTO_START_FIRST", "AUTO_START_LAST")}
    d_by_case = {}
    for r in manifest:
        pass
    count_by_case = Counter(r["case_id"] for r in sel)
    miss_sel = [c for c in sel_cases & set(comp_ids) if count_by_case.get(c, 0) == 0]
    checks["selection_cases_without_rows"] = miss_sel
    if miss_sel:
        add("BLOCKER", "selection_rows_missing", miss_sel[:10])
    gate_wrong = sum(1 for r in sel if r.get("group_gate_correct") == "False")
    checks["support_gate_misassigned_columns"] = gate_wrong
    if gate_wrong:
        add("WARNING", "support_gate", f"{gate_wrong} columns gated to a different "
                                       f"support group than their true family")
    sel_fail = sum(1 for r in sel if r.get("selection_status") == "FAILURE")
    checks["selection_rows_failure"] = sel_fail
    # nonfinite candidate scores among OK candidates
    cand = read_csv(t / "candidate_scores.csv")
    nf = sum(1 for r in cand if r["candidate_status"] == "OK" and not finite(r["candidate_score"]))
    checks["nonfinite_ok_candidate_scores"] = nf
    if nf:
        add("BLOCKER", "nonfinite_candidate", nf)
    ckeys = [(r["case_id"], r["column"], r["candidate_model"], r["iteration"]) for r in cand]
    checks["duplicate_candidate_rows"] = len(ckeys) - len(set(ckeys))
    if checks["duplicate_candidate_rows"]:
        add("BLOCKER", "duplicate_candidate_rows", checks["duplicate_candidate_rows"])
    # a selected column whose admissible candidates include a FAILED one would
    # be a silent promotion
    failed_cand = {(r["case_id"], r["column"], r["iteration"]) for r in cand
                   if r["candidate_status"] != "OK"}
    promoted = [r for r in sel if r.get("selection_status") == "SELECTED"
                and ((r["case_id"], r["column"], "final") in failed_cand
                     or (r["case_id"], r["column"], "last_exploration") in failed_cand)]
    checks["selected_despite_failed_candidate"] = len(promoted)
    if promoted:
        add("BLOCKER", "silent_promotion", len(promoted))

    # derived tables
    conf = read_csv(t / "confusion_matrix_family.csv")
    fam = read_csv(t / "family_recovery_summary.csv")
    if conf and fam:
        tot_conf = defaultdict(int)
        for r in conf:
            if r["difficulty"] == "ALL" and r["n"] == "ALL":
                tot_conf[(r["experiment"], r["stage"], r["method"], r["group_design"])] += int(r["count"])
        tot_sel = Counter((r["experiment"], r["stage"], r["method"], r["group_design"]) for r in sel)
        checks["confusion_sum_matches_selection_rows"] = dict(tot_conf) == dict(tot_sel)
        if not checks["confusion_sum_matches_selection_rows"]:
            add("BLOCKER", "confusion_sum", "confusion totals differ from selection rows")
        den_bad = []
        for r in fam:
            if (r["scope"] == "all_columns" and r["grouping"] == "pooled_difficulty_n"
                    and r["support_group"] == "ALL" and r["true_family"] == "ALL"):
                k = (r["experiment"], r["stage"], r["method"], r["group_design"])
                if int(r["n_evaluated"]) != tot_sel.get(k, -1):
                    den_bad.append(k)
                nfail = sum(1 for s in sel if (s["experiment"], s["stage"], s["method"],
                                               s["group_design"]) == k
                            and s.get("selection_status") == "FAILURE")
                if int(r["n_failed"]) != nfail:
                    den_bad.append(("n_failed",) + k)
        checks["recovery_denominator_errors"] = den_bad
        if den_bad:
            add("BLOCKER", "recovery_denominator", den_bad[:10])
        # score-decided scope must contain no gate-decided column
        sc = Counter((s["experiment"], s["stage"], s["method"], s["group_design"])
                     for s in sel if s.get("decided_by") == "score")
        gate_mix = []
        for r in fam:
            if (r["scope"] == "score_decided_only" and r["grouping"] == "pooled_difficulty_n"
                    and r["support_group"] == "ALL" and r["true_family"] == "ALL"):
                k = (r["experiment"], r["stage"], r["method"], r["group_design"])
                if int(r["n_evaluated"]) != sc.get(k, 0):
                    gate_mix.append(k)
        checks["gate_only_columns_in_score_scope"] = gate_mix
        if gate_mix:
            add("BLOCKER", "gate_column_mixing", gate_mix[:10])
    else:
        add("WARNING", "derived_tables", "summary tables not yet generated")

    # figures and their sources
    caps = run / "figures" / "figure_captions.md"
    figs = sorted(p.name for p in (run / "figures").glob("*.png"))
    checks["figures"] = figs
    if caps.exists():
        import re
        srcs = set(re.findall(r"source: `tables/([^`]+)`", caps.read_text(encoding="utf-8")))
        missing_src = [s for s in srcs if not (t / s).exists()]
        checks["figure_sources_missing"] = missing_src
        if missing_src:
            add("BLOCKER", "figure_source", missing_src)
        if "figure skipped" in caps.read_text(encoding="utf-8"):
            add("WARNING", "figure_skipped", "at least one figure could not be drawn")

    # git provenance.  BLOCKER: experiment-path code differs between any two
    # MAIN-stage phase runs.  WARNING: it differs between the init commit /
    # technical stages (smoke, pilot) and the main stage -- allowed only for
    # disclosed numerical fixes found by those technical stages.
    repo = _HERE.parents[2]
    runs = info.get("phase_runs", [])

    def diff(a, b):
        out = subprocess.check_output(["git", "diff", "--name-only", a, b],
                                      cwd=repo, text=True)
        return [f for f in out.split() if f in EXPERIMENT_PATH_FILES]

    changed = {}
    main_shas = sorted({p["git_sha"] for p in runs if PHASES[p["phase"]]["stage"] == "main"})
    for i in range(len(main_shas)):
        for j in range(i + 1, len(main_shas)):
            hit = diff(main_shas[i], main_shas[j])
            changed[f"main:{main_shas[i][:8]}..{main_shas[j][:8]}"] = hit
            if hit:
                add("BLOCKER", "experiment_code_changed_within_main", hit)
    ref = main_shas[0] if main_shas else None
    if ref:
        others = {("init", info["git_sha"])} | {(p["phase"], p["git_sha"]) for p in runs
                                                if PHASES[p["phase"]]["stage"] != "main"}
        for label, s_ in sorted(others):
            hit = diff(s_, ref)
            changed[f"{label}:{s_[:8]}..main:{ref[:8]}"] = hit
            if hit:
                add("WARNING", "experiment_code_changed_before_main",
                    {"from": f"{label} {s_[:8]}", "to_main": ref[:8], "files": hit})
    checks["init_git_sha"] = info["git_sha"]
    checks["phase_git_shas"] = [(p["phase"], p["git_sha"], p.get("git_dirty_tracked"))
                                for p in info.get("phase_runs", [])]
    checks["git_changes_after_init"] = changed
    dirty = [p["phase"] for p in info.get("phase_runs", []) if p.get("git_dirty_tracked")]
    if dirty:
        add("WARNING", "dirty_tree_during_phase", dirty)

    blockers = [f for f in findings if f["severity"] == "BLOCKER"]
    report = {"run_dir": run.name, "verdict": "PASS" if not blockers else "FAIL",
              "n_blocker": len(blockers),
              "n_warning": sum(f["severity"] == "WARNING" for f in findings),
              "n_diagnostic": sum(f["severity"] == "DIAGNOSTIC" for f in findings),
              "findings": findings, "checks": checks}
    atomic_write_json(run / "audit_report.json", report)
    atomic_write_json(run / "audit" / "audit_report.json", report)
    print(json.dumps({k: report[k] for k in ("verdict", "n_blocker", "n_warning",
                                              "n_diagnostic")}))
    return 0 if not blockers else 1


if __name__ == "__main__":
    sys.exit(main())
