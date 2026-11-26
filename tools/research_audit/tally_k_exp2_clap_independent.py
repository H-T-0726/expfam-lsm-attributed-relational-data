"""Independent tally of the C_Q vs C_Lap comparison run (6,600 evaluations).

Written 2026-11-26 while the frozen run was executing, separately from
run_k_exp2_clap_comparison.py (which it does NOT import). Standard library + numpy only.
Reads: the frozen protocol/manifest, evaluations.csv, the pre-specified analysis
outputs, and the recorded main / convergence-sensitivity CSVs. Re-implements:
  * C_Lap algebra:  ln p_Lap = phi + (nK/2) ln 2pi - 1/2 ln|H|;  C_Lap = -2 ln p_Lap + d_K ln 150;
    fit_mode = -2 (phi - ln p(Z_hat)),  ln p(Z_hat) = -(nK/2) ln 2pi - ||Z_hat||^2 / 2;
    C_Lap = fit_mode + ||Z_hat||^2 + ln|H| + d_K ln 150
  * selection: per (source, dataset), argmin_K of the 2-init mean; ties <= 1e-9 -> smaller K;
    C_Lap INCOMPLETE if any of the 20 evaluations is not OK.

Usage: python tools/research_audit/tally_k_exp2_clap_independent.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "expfam/results/k_selection/k_exp2_generalization_20261125"
CC = BASE / "clap_comparison"
N, D = 150, 15
LN2PI = math.log(2 * math.pi)


def rd(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pick(bar):
    m = min(bar.values())
    return min(k for k, v in bar.items() if v - m <= 1e-9)


def oc(k, kt):
    return "exact" if k == kt else ("under" if k < kt else "over")


def main():
    out = CC / "independent_tally"
    if out.exists():
        raise SystemExit(f"STOP: {out} exists")
    checks, notes = {}, {}
    proto = json.loads((CC / "protocol/protocol.json").read_text(encoding="utf-8"))
    man = rd(CC / "protocol/manifest.csv")
    ev = rd(CC / "run/evaluations.csv")
    # completeness
    key = lambda r: (r["source"], r["fit_id"])                      # noqa: E731
    checks["6600_rows_each_planned_evaluation_once"] = (
        len(ev) == 6600 and sorted(map(key, ev)) == sorted(map(key, man))
        and len(set(map(key, ev))) == 6600)
    st = Counter(r["lap_status"] for r in ev)
    notes["lap_status_counts"] = dict(st)
    notes["lap_status_by_source"] = {s: dict(Counter(r["lap_status"] for r in ev
                                                    if r["source"] == s))
                                     for s in ("main10", "conv40")}
    # C_Q copied unchanged
    mrow = {key(r): r for r in man}
    checks["C_Q_equals_manifest_strings"] = all(r["C_Q"] == mrow[key(r)]["C_Q"] for r in ev)
    main_pf = {r["fit_id"]: r["C_Q"] for r in rd(BASE / "main/per_fit.csv")}
    conv40 = {r["fit_id"]: r["C_Q"] for r in rd(BASE / "convergence_sensitivity/run/"
                                                     "checkpoint_scores.csv")
              if r["checkpoint"] == "40"}
    checks["C_Q_equals_recorded_runs"] = all(
        r["C_Q"] == (main_pf if r["source"] == "main10" else conv40)[r["fit_id"]] for r in ev)
    checks["state_sha_and_data_flags_true"] = all(
        r["state_sha256_ok"] == "True" and r["data_equal"] == "True" for r in ev)
    # C_Lap algebra and numerical validity (OK rows)
    worst = {"C_Lap_vs_logobs": 0.0, "logobs_vs_phi": 0.0, "fit_mode": 0.0, "closure": 0.0}
    bad_valid = []
    for r in ev:
        if r["lap_status"] != "OK":
            if r["C_Lap"] != "":
                bad_valid.append(("value_on_failure", r["fit_id"]))
            continue
        k = int(r["k"])
        nk = N * k
        dk = k * D - k * (k - 1) // 2
        pen = dk * math.log(N)
        c = float(r["C_Lap"])
        lo = float(r["lap_log_observed"])
        phi = float(r["lap_phi_at_mode"])
        ld = float(r["lap_logdet_H"])
        zsq = float(r["lap_zhat_sq"])
        fm = float(r["lap_fit_mode"])
        rel = lambda a, b: abs(a - b) / max(1.0, abs(b))          # noqa: E731
        worst["C_Lap_vs_logobs"] = max(worst["C_Lap_vs_logobs"], rel(c, -2 * lo + pen))
        worst["logobs_vs_phi"] = max(worst["logobs_vs_phi"], rel(lo, phi + nk / 2 * LN2PI - ld / 2))
        worst["fit_mode"] = max(worst["fit_mode"],
                                rel(fm, -2 * (phi - (-nk / 2 * LN2PI - zsq / 2))))
        worst["closure"] = max(worst["closure"], rel(c, fm + zsq + ld + pen))
        if not (float(r["lap_min_eig_H"]) > 0 and float(r["lap_logdet_sign"]) == 1.0
                and float(r["lap_grad_inf"]) <= 1e-8 and int(r["d_K"]) == dk):
            bad_valid.append(("ok_but_invalid", r["fit_id"]))
    notes["worst_relative_algebra_error"] = worst
    checks["C_Lap_algebra_consistent_1e-10"] = all(v <= 1e-10 for v in worst.values())
    checks["OK_rows_stationary_PD_no_value_on_failures"] = not bad_valid
    notes["validity_violations"] = bad_valid[:20]
    # selection
    g = defaultdict(dict)
    for r in ev:
        g[(r["source"], r["dataset_id"])][(int(r["k"]), int(r["t"]))] = r
    sel = []
    for (src, ds), f in sorted(g.items()):
        r0 = f[(1, 0)]
        kt = int(r0["k_true"])
        q = {k: (float(f[(k, 0)]["C_Q"]) + float(f[(k, 1)]["C_Q"])) / 2 for k in range(1, 11)}
        kq = pick(q)
        if all(x["lap_status"] == "OK" for x in f.values()):
            lap = {k: (float(f[(k, 0)]["C_Lap"]) + float(f[(k, 1)]["C_Lap"])) / 2
                   for k in range(1, 11)}
            kl, ol = pick(lap), None
            ol = oc(kl, kt)
        else:
            kl, ol = "", "INCOMPLETE"
        sel.append(dict(source=src, dataset_id=ds, cell=f'{r0["series"]}-{r0["regime"]}-{kt}',
                        k_true=kt, k_Q=kq, outcome_Q=oc(kq, kt), k_Lap=kl, outcome_Lap=ol))
    rec = {("main10", r["dataset_id"]): int(r["primary_k"])
           for r in rd(BASE / "analysis/selection_per_dataset.csv")}
    rec.update({("conv40", r["dataset_id"]): int(r["selected_k"])
                for r in rd(BASE / "convergence_sensitivity/analysis/selected_k_by_dataset.csv")
                if r["checkpoint"] == "40"})
    checks["C_Q_selection_equals_recorded_analyses"] = all(
        rec[(s["source"], s["dataset_id"])] == s["k_Q"] for s in sel)
    ana = {(r["source"], r["dataset_id"]): r
           for r in rd(CC / "analysis/selection_by_dataset.csv")}
    checks["selection_equals_prespecified_analysis"] = all(
        str(s["k_Q"]) == ana[(s["source"], s["dataset_id"])]["k_Q"]
        and str(s["k_Lap"]) == ana[(s["source"], s["dataset_id"])]["k_Lap"]
        and s["outcome_Lap"] == ana[(s["source"], s["dataset_id"])]["outcome_Lap"] for s in sel)
    cnt = Counter((s["source"], s["cell"], "Q", s["outcome_Q"]) for s in sel)
    cnt.update((s["source"], s["cell"], "Lap", s["outcome_Lap"]) for s in sel)
    summ = rd(CC / "analysis/summary_by_cell.csv")
    checks["cell_counts_equal_prespecified_summary"] = all(
        int(r["Q_exact"]) == cnt[(r["source"], r["cell"], "Q", "exact")]
        and int(r["Q_under"]) == cnt[(r["source"], r["cell"], "Q", "under")]
        and int(r["Q_over"]) == cnt[(r["source"], r["cell"], "Q", "over")]
        and int(r["Lap_exact"]) == cnt[(r["source"], r["cell"], "Lap", "exact")]
        and int(r["Lap_under"]) == cnt[(r["source"], r["cell"], "Lap", "under")]
        and int(r["Lap_over"]) == cnt[(r["source"], r["cell"], "Lap", "over")]
        and int(r["Lap_incomplete"]) == cnt[(r["source"], r["cell"], "Lap", "INCOMPLETE")]
        for r in summ)
    notes["totals"] = {src: {"Q": dict(Counter(s["outcome_Q"] for s in sel if s["source"] == src)),
                             "Lap": dict(Counter(s["outcome_Lap"] for s in sel
                                                 if s["source"] == src)),
                             "changed_complete": sum(1 for s in sel if s["source"] == src
                                                     and s["k_Lap"] != ""
                                                     and s["k_Lap"] != s["k_Q"])}
                       for src in ("main10", "conv40")}
    # K_TRUE = 3 identity
    k3 = defaultdict(dict)
    for r in ev:
        if r["k_true"] == "3":
            k3[(r["source"], r["series"], r["rep"], r["k"], r["t"])][r["regime"]] = \
                (r["lap_status"], r["C_Lap"], r["lap_logdet_H"])
    checks["K_TRUE3_regime_a_b_identical"] = all(v["a"] == v["b"] for v in k3.values()
                                                 if len(v) == 2)
    # preservation of existing results
    checks["recorded_inputs_unchanged"] = all(
        sha(ROOT / p) == proto["input_sha256"][k] for k, p in {
            "main_per_fit": BASE / "main/per_fit.csv",
            "main_states_sha256_manifest": BASE / "main/states_sha256_manifest.csv",
            "main_selection_per_dataset": BASE / "analysis/selection_per_dataset.csv",
            "conv_per_fit": BASE / "convergence_sensitivity/run/per_fit.csv",
            "conv_checkpoint_scores": BASE / "convergence_sensitivity/run/checkpoint_scores.csv",
            "conv_states_sha256_manifest":
                BASE / "convergence_sensitivity/run/states_sha256_manifest.csv",
            "conv_selected_k_by_dataset":
                BASE / "convergence_sensitivity/analysis/selected_k_by_dataset.csv"}.items())
    dirs = {"main10": BASE / "main/states", "conv40": BASE / "convergence_sensitivity/run/states"}
    checks["all_6600_npz_unchanged"] = all(
        sha(dirs[r["source"]] / r["state_file"]) == r["state_sha256"] for r in man)
    out.mkdir(parents=True)
    with open(out / "independent_selection.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sel[0]))
        w.writeheader()
        w.writerows(sel)
    res = {"checks": checks, "all_ok": all(checks.values()), "notes": notes}
    (out / "tally_checks.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
