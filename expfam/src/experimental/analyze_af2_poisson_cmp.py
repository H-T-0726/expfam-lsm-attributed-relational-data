"""AF2 analysis: every table and figure regenerated from the primary CSVs.

Written before any AF2 result was read (pre-registered; see
``run_af2_poisson_cmp.protocol()``).  Reads ``<stage>/fit_runs.csv``,
``fit_estimators.csv``, ``selection_columns.csv``, ``selection_trace.csv``.
Writes ``<stage>/analysis/`` (refuses to overwrite without --overwrite).

Rules: statistical unit = replicate (one seed series, so the replicate value
is the single paired difference); contrasts only where both fits are OK and,
for the mode, both modes are OK; exclusions counted; 95 % percentile bootstrap
(10000, seed 20261110); exact zeros counted separately.  Selection accuracy is
reported twice: over all Auto fits (a FAILED fit counts as wrong for every
column) and over successful fits only, always with the failure rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_af2_poisson_cmp as af2                                       # noqa: E402

CONTRASTS = {
    "delta_wrong": ("wrong", "oracle"),
    "delta_auto": ("auto_first", "oracle"),
    "benefit_wrong": ("wrong", "auto_first"),
    "delta_support_only": ("support_only", "oracle"),
    "benefit_support_only": ("support_only", "auto_first"),
    "delta_auto_last": ("auto_last", "oracle"),
    "auto_last_minus_auto_first": ("auto_last", "auto_first"),
}
PRIMARY = ("delta_wrong", "delta_auto", "benefit_wrong")
COLORS = {"oracle": "#2a78d6", "wrong": "#eb6834", "auto": "#1baf7a", "support": "#eda100"}


def read(p: Path):
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(p: Path, rows):
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def fnum(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def dist(v):
    v = np.array([x for x in v if np.isfinite(x)], float)
    if v.size == 0:
        return {"n": 0, "mean": "", "sd": "", "median": "", "q25": "", "q75": "", "min": "", "max": ""}
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else "",
            "median": float(np.median(v)), "q25": float(np.quantile(v, .25)),
            "q75": float(np.quantile(v, .75)), "min": float(v.min()), "max": float(v.max())}


def boot(v):
    v = np.array([x for x in v if np.isfinite(x)], float)
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(af2.BOOTSTRAP_SEED)
    m = v[rng.integers(0, v.size, size=(af2.BOOTSTRAP_RESAMPLES, v.size))].mean(axis=1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def _f(x, nd=4):
    return "" if x == "" or x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def analyze(stage: Path, out: Path):
    runs = read(stage / "fit_runs.csv")
    ests = read(stage / "fit_estimators.csv")
    cols = read(stage / "selection_columns.csv")
    trace = read(stage / "selection_trace.csv")
    by_key = {r["fit_key"]: r for r in runs}
    plan_keys = {af2.fit_key(p) for p in af2.planned_fits()}

    # integrity
    cells = defaultdict(list)
    for r in runs:
        cells[(r["replicate"], r["regime"])].append(r)
    mism = [{"cell": "|".join(c), "hash": h} for c, rs in cells.items()
            for h in ("x_sha", "y_sha", "z_true_sha", "train_mask_sha", "test_mask_sha")
            if len({r[h] for r in rs}) != 1]
    ok = [r for r in runs if r["status"] == "OK"]
    integ = {
        "fits_recorded": len(runs),
        "missing_planned_fits": sorted(plan_keys - set(by_key)) if stage.name == "full" else [],
        "duplicate_fit_keys": sorted({r["fit_key"] for r in runs if sum(x["fit_key"] == r["fit_key"] for x in runs) > 1}),
        "unknown_fit_keys": sorted(set(by_key) - plan_keys),
        "hash_mismatches": mism,
        "train_test_overlap_total": int(sum(int(r["train_test_overlap"]) for r in runs)),
        "refit_mask_mismatch": [r["fit_key"] for r in ok if r.get("refit_mask_matches") != "True"],
        "nonzero_retry_replacement_rescue": [r["fit_key"] for r in ok if any(
            int(fnum(r[c]) or 0) != 0 for c in ("retry_count", "replacement_count", "seed_rescue_count"))],
    }
    integ["pass"] = not any(integ[k] for k in ("missing_planned_fits", "duplicate_fit_keys",
                                               "unknown_fit_keys", "hash_mismatches",
                                               "train_test_overlap_total", "refit_mask_mismatch",
                                               "nonzero_retry_replacement_rescue"))

    val: dict[tuple, float] = {}
    side: dict[tuple, dict] = {}
    for e in ests:
        k = (e["replicate"], e["regime"], e["method"], e["estimator"])
        good = e["estimator_status"] == "OK" and by_key.get(e["fit_key"], {}).get("status") == "OK"
        val[k] = fnum(e["rmse_Z"]) if good else float("nan")
        side[k] = {m: fnum(e.get(m)) if good else float("nan")
                   for m in ("heldout_mean_log_score", "heldout_auc", "gram_corr")}

    reps = sorted({p["replicate"] for p in af2.planned_fits()})
    comp, pairs = [], []
    for rep in reps:
        for reg in af2.REGIMES:
            for est in ("last", "mean5", "mode"):
                row = {"replicate": rep, "regime": reg, "estimator": est}
                for m in af2.METHODS:
                    row[f"rmse_{m}"] = val.get((rep, reg, m, est), float("nan"))
                    row[f"status_{m}"] = by_key.get(f"{rep}_{reg}_{m}", {}).get("status", "NOT_RUN")
                    for s, v in side.get((rep, reg, m, est), {}).items():
                        row[f"{s}_{m}"] = v
                comp.append(row)
                pr = {"replicate": rep, "regime": reg, "estimator": est}
                for c, (a, b) in CONTRASTS.items():
                    pr[c] = row[f"rmse_{a}"] - row[f"rmse_{b}"]
                for m in ("auto_first", "auto_last"):
                    rr = by_key.get(f"{rep}_{reg}_{m}", {})
                    pr[f"{m}_n_correct_count_columns"] = rr.get("n_correct_count_columns", "")
                    pr[f"{m}_status"] = rr.get("status", "")
                pairs.append(pr)
    write(out / "method_comparison.csv", comp)
    write(out / "paired_contrasts.csv", pairs)

    summ = []
    for reg in af2.REGIMES:
        for est in ("last", "mean5", "mode"):
            sub = [p for p in pairs if p["regime"] == reg and p["estimator"] == est]
            for c in CONTRASTS:
                v = [p[c] for p in sub]
                fin = [x for x in v if np.isfinite(x)]
                lo, hi = boot(fin)
                summ.append({"regime": reg, "estimator": est, "contrast": c, **dist(fin),
                             "n_excluded": len(v) - len(fin), "n_pos": sum(x > 0 for x in fin),
                             "n_zero": sum(x == 0 for x in fin), "n_neg": sum(x < 0 for x in fin),
                             "boot_ci95_lo": lo, "boot_ci95_hi": hi,
                             "primary": est == af2.PRIMARY_ESTIMATOR and c in PRIMARY})
    write(out / "summary_contrasts.csv", summ)

    rs = []
    for reg in af2.REGIMES:
        for est in ("last", "mean5", "mode"):
            for m in af2.METHODS:
                fits = [r for r in runs if r["regime"] == reg and r["method"] == m]
                v = [val.get((rep, reg, m, est), float("nan")) for rep in reps]
                rs.append({"regime": reg, "estimator": est, "method": m, "n_fits": len(fits),
                           "n_fit_failed": sum(r["status"] != "OK" for r in fits),
                           "n_with_value": sum(np.isfinite(x) for x in v),
                           **{f"rmse_{k}": x for k, x in dist(v).items()}})
    write(out / "rmse_summary.csv", rs)

    # selection
    sel = []
    for reg in af2.REGIMES:
        for m in ("auto_first", "auto_last"):
            fits = [r for r in runs if r["regime"] == reg and r["method"] == m]
            okf = [r for r in fits if r["status"] == "OK"]
            cs = [c for c in cols if c["regime"] == reg and c["method"] == m and c["true_model"] != "gaussian"]
            cs_ok = [c for c in cs if by_key.get(f"{c['replicate']}_{reg}_{m}", {}).get("status") == "OK"]
            row = {"regime": reg, "method": m, "n_fits": len(fits), "n_failed": len(fits) - len(okf),
                   "count_columns_all_fits": len(cs),
                   "correct_all_fits": sum(c["correct"] == "True" for c in cs),
                   "count_columns_ok_fits": len(cs_ok),
                   "correct_ok_fits": sum(c["correct"] == "True" for c in cs_ok),
                   "fits_all_count_correct": sum(r.get("all_correct") == "True" for r in okf),
                   "score_decided_columns": sum(c.get("decided_by") == "score" for c in cs_ok)}
            for t in ("poisson", "cmp_nu2p0"):
                ct = [c for c in cs_ok if c["true_model"] == t]
                row[f"true_{t}_columns_ok"] = len(ct)
                row[f"true_{t}_sel_poisson"] = sum(c["selected_model"] == "poisson" for c in ct)
                row[f"true_{t}_sel_cmp_nu2p0"] = sum(c["selected_model"] == "cmp_nu2p0" for c in ct)
            mg = [fnum(c["final_margin_neg2"]) for c in cs_ok if np.isfinite(fnum(c.get("final_margin_neg2")))]
            row.update({"margin_min": min(mg) if mg else "", "margin_median": float(np.median(mg)) if mg else "",
                        "margin_lt_2": sum(x < 2 for x in mg),
                        "final_nonconverged_columns": sum(c.get("final_all_converged") == "False" for c in cs_ok)})
            sel.append(row)
    write(out / "selection_summary.csv", sel)

    margin_rows = []
    for c in cols:
        if c["true_model"] != "gaussian" and c.get("final_margin_neg2", "") != "":
            margin_rows.append({k: c[k] for k in ("replicate", "regime", "method", "column", "true_model",
                                                  "selected_model", "correct", "final_margin_neg2",
                                                  "final_score_poisson", "final_score_cmp_nu2p0")})
    write(out / "column_margins.csv", margin_rows or [{"replicate": ""}])

    agree = []
    byc = defaultdict(dict)
    for c in cols:
        if c["true_model"] != "gaussian":
            byc[(c["replicate"], c["regime"], c["column"])][c["method"]] = c["selected_model"]
    for (rep, reg, j), d in sorted(byc.items()):
        if d.get("auto_first") and d.get("auto_last"):
            agree.append({"replicate": rep, "regime": reg, "column": j, "auto_first": d["auto_first"],
                          "auto_last": d["auto_last"], "agree": d["auto_first"] == d["auto_last"]})
    write(out / "start_agreement.csv", agree or [{"replicate": ""}])

    # failures, numerical diagnostics
    fails = []
    for r in runs:
        if r["status"] != "OK":
            fails.append({"fit_key": r["fit_key"], "kind": "FIT_FAILED", "stage": r.get("failed_stage", ""),
                          "detail": r.get("error", "")})
        elif r.get("mode_status") != "OK":
            fails.append({"fit_key": r["fit_key"], "kind": "MODE_NOT_OK", "stage": "joint_mode",
                          "detail": r.get("mode_status", "")})
        for c in ("exploration_estep_backtracks", "refit_estep_backtracks"):
            if fnum(r.get(c)) and fnum(r.get(c)) > 0:
                fails.append({"fit_key": r["fit_key"], "kind": "ESTEP_BACKTRACKS", "stage": c,
                              "detail": r.get(c)})
        if fnum(r.get("n_nonconverged_trace_rows")) and fnum(r.get("n_nonconverged_trace_rows")) > 0:
            fails.append({"fit_key": r["fit_key"], "kind": "CANDIDATE_NONCONVERGENCE_WARNING",
                          "stage": "exploration",
                          "detail": f"rows={r.get('n_nonconverged_trace_rows')} max_grad_inf={r.get('max_trace_grad_inf')}"})
    write(out / "failures_and_warnings.csv", fails or [{"fit_key": "", "kind": "NONE"}])
    fail_by = []
    for reg in af2.REGIMES:
        for m in af2.METHODS:
            fits = [r for r in runs if r["regime"] == reg and r["method"] == m]
            fail_by.append({"regime": reg, "method": m, "n_fits": len(fits),
                            "n_failed": sum(r["status"] != "OK" for r in fits),
                            "n_mode_not_ok": sum(r["status"] == "OK" and r.get("mode_status") != "OK" for r in fits),
                            "failure_reasons": " || ".join(sorted({r.get("error", "")[:120] for r in fits
                                                                   if r["status"] != "OK"}))})
    write(out / "failure_summary.csv", fail_by)
    tr_rows = []
    for reg in af2.REGIMES:
        for m in ("auto_first", "auto_last"):
            t = [x for x in trace if x["regime"] == reg and x["method"] == m]
            g = [fnum(x.get("max_candidate_grad_inf")) for x in t]
            g = [x for x in g if np.isfinite(x)]
            tr_rows.append({"regime": reg, "method": m, "trace_rows": len(t),
                            "rows_not_selected": sum(x.get("status") != "SELECTED" for x in t),
                            "rows_nonconverged": sum(x.get("all_candidates_converged") == "False" for x in t),
                            "max_grad_inf_median": float(np.median(g)) if g else "",
                            "max_grad_inf_max": float(np.max(g)) if g else ""})
    write(out / "candidate_convergence.csv", tr_rows)

    summary = {"integrity": integ, "fits_recorded": len(runs), "fits_ok": len(ok),
               "fits_failed": len(runs) - len(ok),
               "modes_not_ok": sum(1 for r in ok if r.get("mode_status") != "OK"),
               "primary": [s for s in summ if s["primary"]], "selection": sel,
               "start_agreement": {"columns": len(agree), "agree": sum(a["agree"] for a in agree)},
               "failure_summary": fail_by}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "results_tables.md").write_text(tables_md(summ, rs, sel, fail_by, tr_rows, summary)
                                          + posthoc(out, comp, pairs, cols), encoding="utf-8")
    figures(out, pairs, sel)
    return summary


def posthoc(out, comp, pairs, cols):
    """POST-HOC descriptive tables added after the AF2 full results were read.

    They regroup columns the pre-registered analysis already saved and change
    no pre-registered number: (1) Auto - Oracle and Support-only - Auto by the
    number of misselected count columns (questions F/G), (2) held-out Y.
    """
    L = ["", "### 事後に追加した記述表（post-hoc、事前登録外）", ""]
    rows = []
    for reg in af2.REGIMES:
        sub = [p for p in pairs if p["regime"] == reg and p["estimator"] == "mode"
               and p["auto_first_status"] == "OK"]
        for nwrong in sorted({6 - int(p["auto_first_n_correct_count_columns"]) for p in sub}):
            g = [p for p in sub if 6 - int(p["auto_first_n_correct_count_columns"]) == nwrong]
            da = [p["delta_auto"] for p in g]
            bs = [p["benefit_support_only"] for p in g]
            rows.append({"regime": reg, "n_misselected_count_columns": nwrong, "replicates": len(g),
                         "delta_auto_mean": float(np.mean(da)), "delta_auto_max": float(np.max(da)),
                         "n_delta_auto_zero": sum(x == 0 for x in da),
                         "benefit_support_only_mean": float(np.mean(bs)),
                         "n_benefit_support_only_pos": sum(x > 0 for x in bs)})
    write(out / "posthoc_by_misselection.csv", rows)
    L += ["Auto（Poisson 開始）の誤選択列数ごとの paired 差（mode、replicate 単位）", "",
          "| regime | 誤選択列数 | replicate | Auto − Oracle 平均 | 最大 | =0 | Support-only − Auto 平均 | >0 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['regime']} | {r['n_misselected_count_columns']} | {r['replicates']} | "
                 f"{_f(r['delta_auto_mean'])} | {_f(r['delta_auto_max'])} | {r['n_delta_auto_zero']} | "
                 f"{_f(r['benefit_support_only_mean'])} | {r['n_benefit_support_only_pos']} |")
    hy = []
    for reg in af2.REGIMES:
        sub = [c for c in comp if c["regime"] == reg and c["estimator"] == "mode"]
        for m in af2.METHODS:
            nan = float("nan")
            ls = [c.get(f"heldout_mean_log_score_{m}", nan) for c in sub]
            au = [c.get(f"heldout_auc_{m}", nan) for c in sub]
            d = [c.get(f"heldout_mean_log_score_{m}", nan) - c.get("heldout_mean_log_score_oracle", nan)
                 for c in sub]
            hy.append({"regime": reg, "method": m, "n": len(sub), "log_score_mean": float(np.nanmean(ls)),
                       "auc_mean": float(np.nanmean(au)), "minus_oracle_mean": float(np.nanmean(d)),
                       "n_below_oracle": sum(x < 0 for x in d)})
    write(out / "posthoc_heldout_y_summary.csv", hy)
    L += ["", "held-out Y（test ペア、joint mode の plug-in η）", "",
          "| regime | method | n | mean log score | mean AUC | − Oracle 平均 | Oracle 未満 |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for r in hy:
        L.append(f"| {r['regime']} | {r['method']} | {r['n']} | {_f(r['log_score_mean'])} | {_f(r['auc_mean'])} | "
                 f"{_f(r['minus_oracle_mean'])} | {r['n_below_oracle']} |")
    L.append("")
    return "\n".join(L)


def tables_md(summ, rs, sel, fail_by, tr_rows, summary):
    L = ["<!-- generated by analyze_af2_poisson_cmp.py; do not edit by hand -->", "",
         "### RMSE(Z)（joint mode、fit 単位）", "",
         "| regime | method | n fits | failed | value あり | mean | median | IQR |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rs:
        if r["estimator"] == "mode":
            L.append(f"| {r['regime']} | {r['method']} | {r['n_fits']} | {r['n_fit_failed']} | {r['n_with_value']} | "
                     f"{_f(r['rmse_mean'])} | {_f(r['rmse_median'])} | {_f(r['rmse_q25'])}–{_f(r['rmse_q75'])} |")
    for est in ("mode", "last", "mean5"):
        L += ["", f"### replicate 単位の paired 差（{est}、20 replicate）", "",
              "| regime | contrast | n | excluded | mean | median | IQR | >0 | =0 | <0 | 95% bootstrap CI |",
              "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|"]
        for r in summ:
            if r["estimator"] == est:
                L.append(f"| {r['regime']} | {r['contrast']} | {r['n']} | {r['n_excluded']} | {_f(r['mean'])} | "
                         f"{_f(r['median'])} | {_f(r['q25'])}–{_f(r['q75'])} | {r['n_pos']} | {r['n_zero']} | "
                         f"{r['n_neg']} | [{_f(r['boot_ci95_lo'])}, {_f(r['boot_ci95_hi'])}] |")
    L += ["", "### 分布選択（count 6 列 × fit）", "",
          "| regime | start | fits | failed | 正解/全 fit の列 | 正解/成功 fit の列 | 6 列全正解の fit | "
          "真 Poisson→Poisson | 真 Poisson→CMP2 | 真 CMP2→CMP2 | 真 CMP2→Poisson | margin 最小 | 中央値 | margin<2 の列 |",
          "|---|---|---:|---:|---|---|---:|---|---|---|---|---:|---:|---:|"]
    for r in sel:
        st = "Poisson（primary）" if r["method"] == "auto_first" else "CMP2"
        tp, tc = r["true_poisson_columns_ok"], r["true_cmp_nu2p0_columns_ok"]
        L.append(f"| {r['regime']} | {st} | {r['n_fits']} | {r['n_failed']} | {r['correct_all_fits']}/{r['count_columns_all_fits']} | "
                 f"{r['correct_ok_fits']}/{r['count_columns_ok_fits']} | {r['fits_all_count_correct']} | "
                 f"{r['true_poisson_sel_poisson']}/{tp} | {r['true_poisson_sel_cmp_nu2p0']}/{tp} | "
                 f"{r['true_cmp_nu2p0_sel_cmp_nu2p0']}/{tc} | {r['true_cmp_nu2p0_sel_poisson']}/{tc} | "
                 f"{_f(r['margin_min'], 2)} | {_f(r['margin_median'], 2)} | {r['margin_lt_2']} |")
    L += ["", "### 失敗（fit 単位）", "", "| regime | method | fits | FAILED | mode 非 OK | 理由 |", "|---|---|---:|---:|---:|---|"]
    for r in fail_by:
        L.append(f"| {r['regime']} | {r['method']} | {r['n_fits']} | {r['n_failed']} | {r['n_mode_not_ok']} | {r['failure_reasons'][:200]} |")
    L += ["", "### 候補最適化（trust-exact）の診断（探索の全反復）", "",
          "| regime | start | trace 行 | 選択不能の行 | 未収束の行 | max grad_inf 中央値 | 最大 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in tr_rows:
        L.append(f"| {r['regime']} | {r['method']} | {r['trace_rows']} | {r['rows_not_selected']} | {r['rows_nonconverged']} | "
                 f"{r['max_grad_inf_median'] if r['max_grad_inf_median'] == '' else format(r['max_grad_inf_median'], '.2e')} | "
                 f"{r['max_grad_inf_max'] if r['max_grad_inf_max'] == '' else format(r['max_grad_inf_max'], '.2e')} |")
    L += ["", f"- 開始分布間の列単位一致（両方成功）: {summary['start_agreement']['agree']}/{summary['start_agreement']['columns']}",
          f"- fit: OK {summary['fits_ok']} / 記録 {summary['fits_recorded']}、FAILED {summary['fits_failed']}、joint mode 非 OK {summary['modes_not_ok']}",
          f"- integrity pass: {summary['integrity']['pass']}", ""]
    return "\n".join(L)


def figures(out, pairs, sel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8a85", "xtick.color": "#55554f", "ytick.color": "#55554f"})
    show = [("delta_wrong", "Wrong − Oracle", COLORS["wrong"]),
            ("delta_support_only", "Support-only − Oracle", COLORS["support"]),
            ("delta_auto", "Auto(P) − Oracle", COLORS["auto"]),
            ("benefit_wrong", "Wrong − Auto(P)", "#5b5b57")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, reg in zip(axes, af2.REGIMES):
        sub = [p for p in pairs if p["regime"] == reg and p["estimator"] == af2.PRIMARY_ESTIMATOR]
        ax.axhline(0, color="#b5b4ad", lw=1, zorder=0)
        for i, (c, lab, col) in enumerate(show):
            v = np.sort(np.array([p[c] for p in sub], float))
            v = v[np.isfinite(v)]
            x = i + (np.linspace(-.18, .18, len(v)) if len(v) > 1 else np.zeros(len(v)))
            ax.scatter(x, v, s=22, color=col, edgecolor="white", linewidth=.8, zorder=2)
            if len(v):
                ax.plot([i - .28, i + .28], [v.mean()] * 2, color="#1f1f1c", lw=2, zorder=3)
        ax.set_xticks(range(len(show)), [s[1] for s in show], fontsize=8)
        ax.set_title(f"{reg}-Y, joint-mode RMSE(Z)", fontsize=9)
    axes[0].set_ylabel("paired difference in RMSE(Z)")
    fig.text(.01, .005, "points = replicates with both fits OK (sorted); black bar = mean", fontsize=7.5, color="#6b6b65")
    fig.tight_layout()
    fig.savefig(out / "fig1_paired_contrasts_mode.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels, ok_, bad, fl = [], [], [], []
    for r in sel:
        for t in ("poisson", "cmp_nu2p0"):
            other = "cmp_nu2p0" if t == "poisson" else "poisson"
            labels.append(f"{r['regime']}\n{'P' if r['method'] == 'auto_first' else 'C'}-start\ntrue {'Pois' if t == 'poisson' else 'CMP2'}")
            ok_.append(r[f"true_{t}_sel_{t}"])
            bad.append(r[f"true_{t}_sel_{other}"])
            fl.append(r["n_failed"] * 3)
    x = np.arange(len(labels))
    ok_, bad, fl = map(np.array, (ok_, bad, fl))
    ax.bar(x, ok_, color=COLORS["oracle"], width=.6, label="correct")
    ax.bar(x, bad, bottom=ok_, color=COLORS["wrong"], width=.6, label="other candidate", edgecolor="white", linewidth=2)
    ax.bar(x, fl, bottom=ok_ + bad, color="#b5b4ad", width=.6, label="fit failed", edgecolor="white", linewidth=2)
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("count columns (3 × replicates)")
    ax.set_ylim(0, max(1, float((ok_ + bad + fl).max())) * 1.15)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.18))
    fig.tight_layout()
    fig.savefig(out / "fig2_count_column_selection.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    stage = Path(a.stage_dir)
    out = stage / "analysis"
    if out.exists():
        if not a.overwrite:
            raise SystemExit(f"refusing to overwrite {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    s = analyze(stage, out)
    print(json.dumps({"integrity_pass": s["integrity"]["pass"], "fits_ok": s["fits_ok"],
                      "fits_failed": s["fits_failed"], "modes_not_ok": s["modes_not_ok"]}))


if __name__ == "__main__":
    raise SystemExit(main())
