"""AF1 analysis: regenerate every summary table and figure from the primary CSVs.

Written before any AF1 result was read (pre-registered analysis; see
``run_auto_family_z_recovery.protocol()``).  Reads only
``<stage>/fit_runs.csv``, ``fit_estimators.csv``, ``selection_columns.csv``
(and, for the reproduction diagnostic, E1's ``fit_estimators.csv``).  Writes
into ``<stage>/analysis/`` and refuses to overwrite an existing directory
unless ``--overwrite`` is given (re-generation is deterministic).

Rules
  * statistical unit = replicate; seed series and Auto starts are not
    independent datasets;
  * replicate-level contrast = mean over the seed series where both members
    are OK (fit OK and, for the mode, mode status OK); exclusions are counted;
  * 95% percentile bootstrap over replicates, 10000 resamples, seed 20261109;
  * exact zeros are counted separately from positives / negatives.

Usage
    python analyze_auto_family_z_recovery.py --stage-dir <.../af1_20261109/full> [--overwrite]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_auto_family_z_recovery as af                                 # noqa: E402

E1_ESTIMATORS = ROOT / "expfam" / "results" / "z_point_estimator_e1" / "e1_20261104" / "full" / "fit_estimators.csv"
METHOD_LABEL = {"oracle": "Oracle", "wrong": "Wrong", "auto_pstart": "Auto (P-start, primary)",
                "auto_bstart": "Auto (B-start)"}
COLORS = {"oracle": "#2a78d6", "wrong": "#eb6834", "auto_pstart": "#1baf7a", "auto_bstart": "#1baf7a"}
CONTRASTS = {
    "delta_wrong": ("wrong", "oracle"),
    "delta_auto_p": ("auto_pstart", "oracle"),
    "benefit_p": ("wrong", "auto_pstart"),
    "delta_auto_b": ("auto_bstart", "oracle"),
    "benefit_b": ("wrong", "auto_bstart"),
    "auto_b_minus_auto_p": ("auto_bstart", "auto_pstart"),
}


def read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fnum(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def dist(values: list[float]) -> dict[str, Any]:
    v = np.array([x for x in values if np.isfinite(x)], float)
    if v.size == 0:
        return {"n": 0, "mean": "", "sd": "", "median": "", "q25": "", "q75": "", "min": "", "max": ""}
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else "",
            "median": float(np.median(v)), "q25": float(np.quantile(v, 0.25)),
            "q75": float(np.quantile(v, 0.75)), "min": float(v.min()), "max": float(v.max())}


def boot_ci(values: list[float]) -> tuple[float, float]:
    v = np.array([x for x in values if np.isfinite(x)], float)
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(af.BOOTSTRAP_SEED)
    idx = rng.integers(0, v.size, size=(af.BOOTSTRAP_RESAMPLES, v.size))
    means = v[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def analyze(stage: Path, out: Path) -> dict[str, Any]:
    runs = read(stage / "fit_runs.csv")
    ests = read(stage / "fit_estimators.csv")
    cols = read(stage / "selection_columns.csv")
    run_by_key = {r["fit_key"]: r for r in runs}
    plan = af.planned_fits()
    planned_keys = {af.fit_key(p) for p in plan}
    present = set(run_by_key)

    # ── integrity ──────────────────────────────────────────────────────
    cells = defaultdict(list)
    for r in runs:
        cells[(r["series"], r["replicate"], r["regime"])].append(r)
    hash_mismatch = []
    for cell, rs in cells.items():
        for h in ("x_sha", "y_sha", "z_true_sha", "train_mask_sha", "test_mask_sha"):
            if len({r[h] for r in rs}) != 1:
                hash_mismatch.append({"cell": "|".join(cell), "hash": h})
    # the same data / test mask across regimes and series for a replicate
    for rep in {r["replicate"] for r in runs}:
        rs = [r for r in runs if r["replicate"] == rep]
        for h in ("x_sha", "y_sha", "z_true_sha", "test_mask_sha"):
            if len({r[h] for r in rs}) != 1:
                hash_mismatch.append({"cell": rep, "hash": h})
    ok_runs = [r for r in runs if r["status"] == "OK"]
    integrity = {
        "planned_fits_in_stage": None, "fits_recorded": len(runs),
        "unknown_fit_keys": sorted(present - planned_keys),
        "hash_mismatches": hash_mismatch,
        "train_test_overlap_total": int(sum(int(r["train_test_overlap"]) for r in runs)),
        "refit_mask_mismatch": [r["fit_key"] for r in ok_runs if r.get("refit_mask_matches") != "True"],
        "nonzero_retry_replacement_rescue": [
            r["fit_key"] for r in ok_runs
            if any(int(float(r[c])) != 0 for c in ("retry_count", "replacement_count", "seed_rescue_count"))],
        "nan_occurred": [r["fit_key"] for r in ok_runs if r.get("nan_occurred") == "True"],
    }
    integrity["pass"] = not (integrity["unknown_fit_keys"] or hash_mismatch
                             or integrity["train_test_overlap_total"]
                             or integrity["refit_mask_mismatch"]
                             or integrity["nonzero_retry_replacement_rescue"])

    # ── RMSE lookup ────────────────────────────────────────────────────
    rmse: dict[tuple, float] = {}
    side: dict[tuple, dict[str, float]] = {}
    for e in ests:
        key = (e["series"], e["replicate"], e["regime"], e["method"], e["estimator"])
        ok = e["estimator_status"] == "OK" and run_by_key.get(e["fit_key"], {}).get("status") == "OK"
        rmse[key] = fnum(e["rmse_Z"]) if ok else float("nan")
        side[key] = {k: fnum(e.get(k)) if ok else float("nan")
                     for k in ("heldout_mean_log_score", "heldout_auc", "gram_corr")}

    series_list = sorted({p["series"] for p in plan})
    reps = sorted({p["replicate"] for p in plan})
    regimes = list(af.REGIMES)

    # method comparison (long, one row per series x replicate x regime x estimator)
    comp_rows, pair_rows = [], []
    for s in series_list:
        for rep in reps:
            for reg in regimes:
                for est in af.ESTIMATORS:
                    row = {"series": s, "replicate": rep, "regime": reg, "estimator": est}
                    for m in af.METHODS:
                        row[f"rmse_{m}"] = rmse.get((s, rep, reg, m, est), float("nan"))
                        row[f"status_{m}"] = run_by_key.get(f"{s}_{rep}_{reg}_{m}", {}).get("status", "NOT_RUN")
                        for k, v in side.get((s, rep, reg, m, est), {}).items():
                            row[f"{k}_{m}"] = v
                    comp_rows.append(row)
                    pr = {"series": s, "replicate": rep, "regime": reg, "estimator": est}
                    for c, (a, b) in CONTRASTS.items():
                        pr[c] = row[f"rmse_{a}"] - row[f"rmse_{b}"]
                    for m in ("auto_pstart", "auto_bstart"):
                        rr = run_by_key.get(f"{s}_{rep}_{reg}_{m}", {})
                        pr[f"{m}_all_correct"] = rr.get("all_correct", "")
                        pr[f"{m}_n_correct_columns"] = rr.get("n_correct_columns", "")
                    pair_rows.append(pr)
    write(out / "method_comparison.csv", comp_rows)
    write(out / "paired_contrasts_by_series.csv", pair_rows)

    # replicate-level contrasts (mean over series with both members OK)
    rep_rows = []
    for rep in reps:
        for reg in regimes:
            for est in af.ESTIMATORS:
                row = {"replicate": rep, "regime": reg, "estimator": est}
                for c in CONTRASTS:
                    vals = [p[c] for p in pair_rows if p["replicate"] == rep and p["regime"] == reg
                            and p["estimator"] == est and np.isfinite(p[c])]
                    row[c] = float(np.mean(vals)) if vals else float("nan")
                    row[f"{c}_n_series"] = len(vals)
                rep_rows.append(row)
    write(out / "paired_contrasts_replicate.csv", rep_rows)

    # contrast summaries
    summ = []
    def add_summary(rows, agg):
        for reg in regimes:
            for est in af.ESTIMATORS:
                sub = [r for r in rows if r["regime"] == reg and r["estimator"] == est]
                for c in CONTRASTS:
                    vals = [r[c] for r in sub]
                    fin = [v for v in vals if np.isfinite(v)]
                    lo, hi = boot_ci(fin)
                    summ.append({"aggregation": agg, "regime": reg, "estimator": est, "contrast": c,
                                 **dist(fin), "n_excluded": len(vals) - len(fin),
                                 "n_pos": sum(v > 0 for v in fin), "n_zero": sum(v == 0 for v in fin),
                                 "n_neg": sum(v < 0 for v in fin),
                                 "boot_ci95_lo": lo, "boot_ci95_hi": hi,
                                 "primary": agg == "replicate_mean_over_series" and est == af.PRIMARY_ESTIMATOR
                                 and c in ("delta_wrong", "delta_auto_p", "benefit_p")})
    add_summary(rep_rows, "replicate_mean_over_series")
    for s in series_list:
        add_summary([p for p in pair_rows if p["series"] == s], f"series_{s}")
    write(out / "summary_contrasts.csv", summ)

    # RMSE summaries per method
    rs_rows = []
    for s in series_list + ["all"]:
        for reg in regimes:
            for est in af.ESTIMATORS:
                for m in af.METHODS:
                    keys = [k for k in rmse if k[2] == reg and k[3] == m and k[4] == est
                            and (s == "all" or k[0] == s)]
                    fits = [r for r in runs if r["regime"] == reg and r["method"] == m
                            and (s == "all" or r["series"] == s)]
                    vals = [rmse[k] for k in keys]
                    rs_rows.append({"series": s, "regime": reg, "estimator": est, "method": m,
                                    "n_fits": len(fits),
                                    "n_fit_failed": sum(r["status"] != "OK" for r in fits),
                                    "n_estimator_not_ok": sum(not np.isfinite(v) for v in vals)
                                    - sum(r["status"] != "OK" for r in fits),
                                    **{f"rmse_{k}": v for k, v in dist(vals).items()}})
    write(out / "rmse_summary.csv", rs_rows)

    # selection summaries
    sel_rows = []
    for s in series_list + ["all"]:
        for reg in regimes:
            for m in ("auto_pstart", "auto_bstart"):
                fits = [r for r in runs if r["regime"] == reg and r["method"] == m
                        and (s == "all" or r["series"] == s)]
                okf = [r for r in fits if r["status"] == "OK"]
                cs = [c for c in cols if c["regime"] == reg and c["method"] == m
                      and (s == "all" or c["series"] == s)]
                score_cols = [c for c in cs if c["decided_by"] == "score"]
                bern = [c for c in cs if c["true_family"] == "bernoulli"]
                pois = [c for c in cs if c["true_family"] == "poisson"]
                margins = [fnum(c["final_margin_neg2"]) for c in score_cols]
                sel_rows.append({
                    "series": s, "regime": reg, "method": m, "n_fits": len(fits),
                    "n_failed": len(fits) - len(okf),
                    "n_all_columns_correct": sum(r["all_correct"] == "True" for r in okf),
                    "n_selected_equals_oracle": sum(r["selected_equals_oracle"] == "True" for r in okf),
                    "columns_total": len(cs),
                    "columns_correct": sum(c["correct"] == "True" for c in cs),
                    "gate_decided_columns": sum(c["decided_by"] == "gate" for c in cs),
                    "score_decided_columns": len(score_cols),
                    "score_decided_correct": sum(c["correct"] == "True" for c in score_cols),
                    "bernoulli_columns": len(bern),
                    "bernoulli_selected_bernoulli": sum(c["selected_family"] == "bernoulli" for c in bern),
                    "bernoulli_selected_poisson": sum(c["selected_family"] == "poisson" for c in bern),
                    "poisson_columns": len(pois),
                    "poisson_columns_score_decided": sum(c["decided_by"] == "score" for c in pois),
                    "poisson_selected_poisson": sum(c["selected_family"] == "poisson" for c in pois),
                    "final_margin_min": min(margins) if margins else "",
                    "final_margin_median": float(np.median(margins)) if margins else "",
                    "final_nonconverged_columns": sum(c["final_all_candidates_converged"] == "False"
                                                      for c in score_cols),
                    "fits_convergence_gate_blocked": sum(r.get("convergence_gate_status") == "BLOCKED_FOR_PILOT"
                                                         for r in okf),
                    "nonconverged_selection_rows_total": sum(int(fnum(r.get("n_nonconverged_selection_rows", 0)) or 0)
                                                             for r in okf),
                })
    write(out / "selection_summary.csv", sel_rows)

    # start agreement (column level, per series x replicate x regime)
    agree_rows = []
    by = defaultdict(dict)
    for c in cols:
        by[(c["series"], c["replicate"], c["regime"], c["column"])][c["method"]] = c["selected_family"]
    for (s, rep, reg, j), d in sorted(by.items()):
        if "auto_pstart" in d and "auto_bstart" in d:
            agree_rows.append({"series": s, "replicate": rep, "regime": reg, "column": j,
                               "auto_pstart": d["auto_pstart"], "auto_bstart": d["auto_bstart"],
                               "agree": d["auto_pstart"] == d["auto_bstart"]})
    write(out / "start_agreement.csv", agree_rows)

    # failures and warnings
    fail_rows = []
    for r in runs:
        if r["status"] != "OK":
            fail_rows.append({"fit_key": r["fit_key"], "kind": "FIT_FAILED",
                              "detail": r.get("error", ""), "stage": r.get("failed_stage", "")})
        elif r.get("mode_status") != "OK":
            fail_rows.append({"fit_key": r["fit_key"], "kind": "MODE_NOT_OK",
                              "detail": r.get("mode_status", ""), "stage": "joint_mode"})
        if r["status"] == "OK" and r.get("convergence_gate_status") == "BLOCKED_FOR_PILOT":
            fail_rows.append({"fit_key": r["fit_key"], "kind": "CANDIDATE_NONCONVERGENCE_WARNING",
                              "detail": f"rows={r.get('n_nonconverged_selection_rows')} "
                                        f"final_candidates={r.get('n_nonconverged_final_candidates')}",
                              "stage": "exploration"})
        if int(fnum(r.get("n_warnings", 0)) or 0) > 0:
            fail_rows.append({"fit_key": r["fit_key"], "kind": "PYTHON_WARNING",
                              "detail": r.get("warning_messages", ""), "stage": ""})
    write(out / "failures_and_warnings.csv", fail_rows or [{"fit_key": "", "kind": "NONE", "detail": "", "stage": ""}])

    # questions F and G: correct selection with worse RMSE / wrong selection with small effect
    case_rows = []
    for p in pair_rows:
        if p["estimator"] != af.PRIMARY_ESTIMATOR:
            continue
        for m, c_auto in (("auto_pstart", "delta_auto_p"), ("auto_bstart", "delta_auto_b")):
            allc = p[f"{m}_all_correct"]
            if allc == "":
                continue
            case_rows.append({"series": p["series"], "replicate": p["replicate"], "regime": p["regime"],
                              "method": m, "all_correct": allc,
                              "n_correct_columns": p[f"{m}_n_correct_columns"],
                              "delta_auto": p[c_auto], "delta_wrong": p["delta_wrong"]})
    write(out / "selection_vs_rmse_cases.csv", case_rows)

    # E1 reproduction diagnostic
    e1rows = read(E1_ESTIMATORS)
    rep_diag = []
    arm_of = {"oracle": "correct", "wrong": "wrong"}
    e1map = {(r["part"], r["replicate"], r["regime"], r["arm"], r["estimator"]): fnum(r["rmse_Z"]) for r in e1rows}
    for e in ests:
        if e["method"] in arm_of and e["estimator_status"] == "OK":
            ref = e1map.get((e["series"], e["replicate"], e["regime"], arm_of[e["method"]], e["estimator"]))
            if ref is not None:
                rep_diag.append({"fit_key": e["fit_key"], "estimator": e["estimator"],
                                 "rmse_af1": fnum(e["rmse_Z"]), "rmse_e1": ref,
                                 "abs_diff": abs(fnum(e["rmse_Z"]) - ref)})
    write(out / "e1_reproduction.csv", rep_diag or [{"fit_key": "", "estimator": "", "rmse_af1": "",
                                                     "rmse_e1": "", "abs_diff": ""}])
    e1_max = max((r["abs_diff"] for r in rep_diag), default=float("nan"))

    summary = {
        "integrity": integrity,
        "fits_recorded": len(runs), "fits_ok": len(ok_runs),
        "fits_failed": len(runs) - len(ok_runs),
        "modes_not_ok": sum(1 for r in ok_runs if r.get("mode_status") != "OK"),
        "e1_reproduction": {"n_compared": len(rep_diag), "max_abs_diff": e1_max},
        "primary": [r for r in summ if r["primary"]],
        "selection_all_series": [r for r in sel_rows if r["series"] == "all"],
        "start_agreement": {"columns": len(agree_rows),
                            "agree": sum(r["agree"] for r in agree_rows)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    figures(out, rep_rows, sel_rows)
    return summary


def figures(out: Path, rep_rows, sel_rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#33332f",
                         "xtick.color": "#55554f", "ytick.color": "#55554f"})
    # Fig 1: replicate-level contrasts (primary estimator), one panel per regime
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    show = [("delta_wrong", "Δ_wrong\nWrong − Oracle", COLORS["wrong"]),
            ("delta_auto_p", "Δ_auto\nAuto(P) − Oracle", COLORS["auto_pstart"]),
            ("benefit_p", "Benefit\nWrong − Auto(P)", "#5b5b57")]
    for ax, reg in zip(axes, af.REGIMES):
        sub = [r for r in rep_rows if r["regime"] == reg and r["estimator"] == af.PRIMARY_ESTIMATOR]
        ax.axhline(0, color="#b5b4ad", lw=1, zorder=0)
        for i, (c, lab, col) in enumerate(show):
            v = np.array([r[c] for r in sub], float)
            v = v[np.isfinite(v)]
            x = i + np.linspace(-0.18, 0.18, len(v)) if len(v) > 1 else np.full(len(v), i)
            ax.scatter(x, np.sort(v), s=22, color=col, edgecolor="white", linewidth=0.8, zorder=2)
            if len(v):
                ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color="#1f1f1c", lw=2, zorder=3)
        ax.set_xticks(range(len(show)), [s[1] for s in show])
        ax.set_title(f"{reg}-Y (n_rep={len(sub)}), joint-mode RMSE(Z)", fontsize=9, color="#1f1f1c")
    axes[0].set_ylabel("replicate-level difference in RMSE(Z)\n(mean over seed series A, B)")
    fig.text(0.01, 0.005, "black bar = mean over replicates; points = replicates (sorted)",
             fontsize=7.5, color="#6b6b65")
    fig.tight_layout()
    fig.savefig(out / "fig1_paired_contrasts_mode.png", dpi=160)
    plt.close(fig)

    # Fig 2: Bernoulli-column selection counts per regime and start
    fig, ax = plt.subplots(figsize=(6, 3))
    rows = [r for r in sel_rows if r["series"] == "all"]
    labels = [f"{r['regime']}\n{'P-start' if r['method'] == 'auto_pstart' else 'B-start'}" for r in rows]
    b = np.array([r["bernoulli_selected_bernoulli"] for r in rows], float)
    p = np.array([r["bernoulli_selected_poisson"] for r in rows], float)
    ax.bar(range(len(rows)), b, color=COLORS["oracle"], width=0.6, label="selected Bernoulli (correct)")
    ax.bar(range(len(rows)), p, bottom=b, color=COLORS["wrong"], width=0.6,
           label="selected Poisson (wrong)", edgecolor="white", linewidth=2)
    for i, (bb, pp) in enumerate(zip(b, p)):
        ax.text(i, bb + pp + 2, f"{int(bb)}/{int(bb + pp)}", ha="center", fontsize=8, color="#33332f")
    ax.set_xticks(range(len(rows)), labels)
    ax.set_ylabel("true-Bernoulli columns\n(6 × replicates × series)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig2_bernoulli_column_selection.png", dpi=160)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    stage = Path(args.stage_dir)
    out = stage / "analysis"
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"refusing to overwrite {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    s = analyze(stage, out)
    print(json.dumps({"integrity_pass": s["integrity"]["pass"], "fits_ok": s["fits_ok"],
                      "fits_failed": s["fits_failed"], "modes_not_ok": s["modes_not_ok"],
                      "e1_max_abs_diff": s["e1_reproduction"]["max_abs_diff"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
