"""Analysis of the MCEM-stability (RQ1) and signal-matched K3/K5 (RQ2) experiment.  No fit.

Reads   expfam/results/k_selection/mcem_stability_signal_matched_20261106/{gates,rq1_AB,rq1_CD,rq2}
Writes  .../analysis/*.csv, analysis_manifest.json and .../figures/*.{png,pdf}
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, wilcoxon

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
BASE = ROOT / "expfam" / "results" / "k_selection" / "mcem_stability_signal_matched_20261106"
OUT, FIG = BASE / "analysis", BASE / "figures"
PREV = ROOT / "expfam" / "results" / "k_selection" / "historical_canonical_causal_20261106"
EXP = "mcem_stability_signal_matched_20261106"
STARTS = (1, 2)
RQ1_CPS = (8, 16, 24, 32, 48, 64)
RQ2_CPS = (8, 16, 24, 32)
TOL = 0.01
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "K3": "#2a78d6", "K5": "#eb6834"}
MK = {"A": "o", "B": "s", "C": "^", "D": "D", "K3": "o", "K5": "s"}
LAB = {"A": "A baseline", "B": "B F-strong", "C": "C w-strong", "D": "D F+w-strong"}
INK2 = "#52514e"


def need(c, m):
    if not c:
        raise RuntimeError(m)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(df: pd.DataFrame, name: str) -> None:
    p = OUT / name
    need(not p.exists(), f"refuse to overwrite {p}")
    df.to_csv(p, index=False, float_format="%.12g")


def load():
    rows, iters, ledg, runs = [], [], [], {}
    for st in ("rq1_AB", "rq1_CD", "rq2"):
        d = BASE / st
        if not (d / "runinfo.json").exists():
            continue
        runs[st] = json.loads((d / "runinfo.json").read_text(encoding="utf-8"))
        rows.append(pd.read_csv(d / "checkpoint_rows.csv"))
        iters.append(pd.read_csv(d / "iteration_rows.csv"))
        ledg.append(pd.read_csv(d / "ledger.csv").assign(stage=st))
    need(runs, "no stage completed")
    return pd.concat(rows, ignore_index=True), pd.concat(iters, ignore_index=True), pd.concat(ledg, ignore_index=True), runs


def stable_flag(r) -> bool:
    return (r.max_gram_rel_change_last8 <= TOL and r.max_eta_x_rel_change_last8 <= TOL
            and r.max_eta_y_rel_change_last8 <= TOL)


def ci_order(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if n < 6:
        return np.nan, np.nan, np.nan
    r = 2 if n >= 8 else 1
    return x[r - 1], x[n - r], 1 - 2 * binom.cdf(r - 1, n, 0.5)


def wil(x):
    x = np.asarray(x, float)
    if len(x) < 2 or not np.any(x != 0):
        return np.nan
    try:
        return float(wilcoxon(x, method="exact").pvalue)
    except ValueError:
        return np.nan


def local_slacks(f: pd.DataFrame, k_lo: int, k_hi: int, keys) -> pd.DataFrame:
    """S(k_lo -> k_hi) on start means; NaN-safe with explicit status."""
    out = []
    ok = f[f.status == "FIT"]
    for key, g in ok.groupby(keys):
        by = {(int(r.k_fit), int(r.start)): r for r in g.itertuples()}
        if not all((k, s) in by for k in (k_lo, k_hi) for s in STARTS):
            out.append({**dict(zip(keys, key)), "k_from": k_lo, "k_to": k_hi, "status": "INCOMPLETE_FIT"})
            continue
        m = lambda k, c: float(np.mean([getattr(by[(k, s)], c) for s in STARTS]))  # noqa: E731
        rep_ok = all(getattr(by[(k, s)], "replication_status") in ("MATCH", "NOT_APPLICABLE") for k in (k_lo, k_hi) for s in STARTS)
        lap_ok = all(by[(k, s)].lap_status == "OK" for k in (k_lo, k_hi) for s in STARTS)
        row = {**dict(zip(keys, key)), "k_from": k_lo, "k_to": k_hi, "status": "OK",
               "replication_ok": rep_ok, "lap_ok": lap_ok,
               "S_Q": m(k_lo, "C_Q") - m(k_hi, "C_Q"),
               "gain_x": 2 * (m(k_hi, "Q_X") - m(k_lo, "Q_X")), "gain_y": 2 * (m(k_hi, "Q_Y") - m(k_lo, "Q_Y")),
               "prior_cost": -2 * (m(k_hi, "Q_Z") - m(k_lo, "Q_Z")), "penalty": m(k_hi, "P_theta") - m(k_lo, "P_theta"),
               "S_Q_eval": m(k_lo, "C_Q_eval") - m(k_hi, "C_Q_eval"),
               "S_Q_eval_se": math.sqrt(sum(getattr(by[(k, s)], "C_Q_eval_mcse") ** 2 for k in (k_lo, k_hi) for s in STARTS)) / 2,
               "gain_x_eval": 2 * (m(k_hi, "Q_X_eval") - m(k_lo, "Q_X_eval")),
               "gain_y_eval": 2 * (m(k_hi, "Q_Y_eval") - m(k_lo, "Q_Y_eval")),
               "start_gap_C_Q_klo": abs(by[(k_lo, 1)].C_Q - by[(k_lo, 2)].C_Q),
               "start_gap_C_Q_khi": abs(by[(k_hi, 1)].C_Q - by[(k_hi, 2)].C_Q),
               "S_Q_start1": by[(k_lo, 1)].C_Q - by[(k_hi, 1)].C_Q, "S_Q_start2": by[(k_lo, 2)].C_Q - by[(k_hi, 2)].C_Q,
               "max_gram_last8_khi": max(getattr(by[(k_hi, s)], "max_gram_rel_change_last8") for s in STARTS),
               "all_fits_stable": all(stable_flag(by[(k, s)]) for k in (k_lo, k_hi) for s in STARTS)}
        row["closure_error"] = row["S_Q"] - (row["gain_x"] + row["gain_y"] - row["prior_cost"] - row["penalty"])
        if lap_ok:
            row.update({"S_Lap": m(k_lo, "C_Lap") - m(k_hi, "C_Lap"),
                        "two_dH_Lap": m(k_hi, "H_Lap_2") - m(k_lo, "H_Lap_2"),
                        "d_residual": m(k_hi, "residual") - m(k_lo, "residual"),
                        "d_volume": m(k_hi, "lap_volume") - m(k_lo, "lap_volume"),
                        "dlogdet_H": m(k_hi, "lap_logdet_H") - m(k_lo, "lap_logdet_H")})
            row["accounting_error"] = (row["S_Lap"] - row["S_Q"]) - (row["two_dH_Lap"] + row["d_residual"])
        out.append(row)
    return pd.DataFrame(out)


def verdict(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return "NO_DATA"
    if np.median(x) > 0 and (x > 0).sum() >= 6:
        return "MAINTAINED"
    if np.median(x) <= 0:
        return "NOT_MAINTAINED"
    return "INCONCLUSIVE"


def paired(loc: pd.DataFrame):
    per, summ = [], []
    for crit, col in (("C_Q", "S_Q"), ("C_Lap", "S_Lap"), ("C_Q_eval", "S_Q_eval")):
        for cp in RQ1_CPS:
            sub = loc[loc.checkpoint == cp].set_index(["condition", "replicate"])
            for c in "BCD":
                diffs = []
                for rep in range(1, 9):
                    ok = all((cc, rep) in sub.index for cc in (c, "A"))
                    excl = ""
                    if ok:
                        ra, rc = sub.loc[("A", rep)], sub.loc[(c, rep)]
                        if not (ra.replication_ok and rc.replication_ok):
                            excl = "REPLICATION_FAIL"
                        elif crit == "C_Lap" and not (ra.lap_ok and rc.lap_ok):
                            excl = "LAPLACE_INCOMPLETE"
                    else:
                        excl = "MISSING"
                    v = np.nan if excl else float(rc[col] - ra[col])
                    if not excl:
                        diffs.append(v)
                    per.append({"experiment_id": EXP, "criterion": crit, "checkpoint": cp, "contrast": f"{c}-A",
                                "replicate": rep, "slack_A": np.nan if excl else float(ra[col]),
                                "slack_cond": np.nan if excl else float(rc[col]), "difference": v,
                                "status": excl or "OK"})
                lo, hi, cov = ci_order(diffs)
                summ.append({"experiment_id": EXP, "criterion": crit, "checkpoint": cp, "contrast": f"{c}-A",
                             "pairs": len(diffs), "median": float(np.median(diffs)) if diffs else np.nan,
                             "n_positive": int(np.sum(np.array(diffs) > 0)), "median_ci_lo": lo, "median_ci_hi": hi,
                             "ci_coverage": cov, "wilcoxon_exact_p": wil(diffs), "verdict_rule": verdict(diffs),
                             "prereg_primary": crit == "C_Q" and cp == 64})
    return pd.DataFrame(per), pd.DataFrame(summ)


def main() -> int:
    need(not OUT.exists(), f"output exists: {OUT}")
    f, it, ledg, runs = load()
    gates = json.loads((BASE / "gates" / "gates.json").read_text(encoding="utf-8"))
    pre = json.loads((BASE / "gates" / "prereg.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True)
    key = ["rq", "condition", "replicate", "k_fit", "start", "checkpoint"]
    ok = f[f.status == "FIT"]
    need(not ok.duplicated(key).any(), "duplicate checkpoint keys")
    f1, f2 = f[f.rq == "RQ1"], f[f.rq == "RQ2"]
    # ---- protocol / provenance / ledger
    flat = []

    def walk(prefix, v):
        if isinstance(v, dict):
            for kk, vv in v.items():
                walk(f"{prefix}.{kk}" if prefix else kk, vv)
        else:
            flat.append({"experiment_id": EXP, "key": prefix, "value": json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v})
    walk("", {k: v for k, v in pre.items()})
    save(pd.DataFrame(flat), "experiment_protocol.csv")
    p1 = pd.read_csv(PREV / "analysis" / "dataset_provenance.csv")
    p1 = p1[p1.n == 75].assign(rq="RQ1", experiment_id=EXP, source="PR #155 provenance (regenerated and hash-checked in gate G1)")
    p2 = pd.read_csv(BASE / "gates" / "rq2_dataset_provenance.csv").assign(experiment_id=EXP)
    save(pd.concat([p1, p2], ignore_index=True), "dataset_provenance.csv")
    gate_led = pd.DataFrame([{"stage": "gates", "rq": "RQ1", "condition": "A", "replicate": 1, "k_fit": 5, "start": 1,
                              "status": "FIT", "fit_iterations": 64, "role": "pilot / replication gate (not data)"},
                             {"stage": "gates", "rq": "RQ1", "condition": "A", "replicate": 1, "k_fit": 5, "start": 1,
                              "status": "FIT", "fit_iterations": 16, "role": "evaluation-isolation gate (not data)"}])
    led = pd.concat([ledg.assign(role="production"), gate_led], ignore_index=True).assign(experiment_id=EXP)
    save(led, "fit_execution_ledger.csv")
    # ---- checkpoint tables
    metr_cols = [c for c in f.columns if c.startswith(("gram_", "f_raw", "w_change", "w0_change", "eta_", "max_"))]
    m = ok[key + ["experiment_id", "n", "d", "k_gen", "model_seed", "replication_status"] + metr_cols].copy()
    m["stable_at_checkpoint"] = ok.apply(stable_flag, axis=1).values
    save(m.assign(status="FIT"), "checkpoint_metrics.csv")
    long = []
    for r in ok.itertuples():
        for crit, val, stt in (("C_Q", r.C_Q, "OK"), ("C_Lap", r.C_Lap if r.lap_status == "OK" else np.nan,
                                                         "OK" if r.lap_status == "OK" else f"INCOMPLETE_{r.lap_status}"),
                               ("C_Q_eval", r.C_Q_eval, "OK")):
            long.append({"experiment_id": EXP, "rq": r.rq, "condition": r.condition, "n": r.n, "d": r.d, "k_gen": r.k_gen,
                         "k_fit": r.k_fit, "replicate": r.replicate, "start": r.start, "model_seed": r.model_seed,
                         "checkpoint": r.checkpoint, "criterion": crit, "value": val, "status": stt})
    save(pd.DataFrame(long), "checkpoint_criterion_scores.csv")
    qcols = ["Q_X", "Q_Y", "Q_Z", "P_Z", "P_theta", "D_K", "q_strict", "Q_X_eval", "Q_Y_eval", "Q_Z_eval", "C_Q_eval_mcse",
             "lap_status", "lap_fit_mode", "lap_volume", "lap_logdet_H", "lap_grad_inf", "lap_min_eig_H", "lap_ll_X_mode",
             "lap_ll_Y_mode", "logdet_prior_X_hat", "logdet_Y_increment_hat", "H_Lap_2", "residual", "termA", "R3", "identity_error"]
    save(ok[key + ["experiment_id", "n", "d", "k_gen", "model_seed"] + [c for c in qcols if c in ok]].assign(status="FIT"),
         "checkpoint_q_decomposition.csv")
    # ---- local comparisons
    loc = local_slacks(f1, 4, 5, ["condition", "replicate", "checkpoint"]) if len(f1) else pd.DataFrame()
    if len(loc):
        loc = loc.assign(experiment_id=EXP, n=75, d=15, k_gen=5,
                         local_choice_C_Q=np.where(loc.S_Q > 0, 5, 4),
                         local_choice_C_Lap=np.where(loc.get("S_Lap", np.nan) > 0, 5, np.where(loc.get("S_Lap").notna(), 4, -1)))
        save(loc, "local_k4_k5_comparison.csv")
        per, summ = paired(loc)
        save(pd.concat([per.assign(row_type="per_replicate"), summ.assign(row_type="summary")], ignore_index=True),
             "paired_slack_changes.csv")
    # ---- stability
    st = []
    for (c, rep, k, s), g in f1[f1.status == "FIT"].groupby(["condition", "replicate", "k_fit", "start"]):
        g = g.set_index("checkpoint")
        st.append({"experiment_id": EXP, "level": "fit", "rq": "RQ1", "condition": c, "replicate": rep, "k_fit": k, "start": s,
                   "label_at_64": "STABLE" if stable_flag(g.loc[64]) else "NOT_STABLE",
                   "first_stable_checkpoint": next((cp for cp in RQ1_CPS if stable_flag(g.loc[cp])), ""),
                   "max_gram_last8_at_64": g.loc[64].max_gram_rel_change_last8,
                   "max_eta_x_last8_at_64": g.loc[64].max_eta_x_rel_change_last8,
                   "max_eta_y_last8_at_64": g.loc[64].max_eta_y_rel_change_last8,
                   "replication_status": g.loc[8].replication_status})
    for (c, rep), g in (loc.groupby(["condition", "replicate"]) if len(loc) else []):
        g = g.set_index("checkpoint")
        sg = {cp: np.sign(g.loc[cp].S_Q) for cp in (8, 32, 48, 64)}
        st.append({"experiment_id": EXP, "level": "dataset", "rq": "RQ1", "condition": c, "replicate": rep,
                   "SIGN_STABLE_C_Q": sg[32] == sg[48] == sg[64], "sign_change_8_to_64_C_Q": bool(sg[8] != sg[64]),
                   "SLACK_STABLE_C_Q": abs(g.loc[64].S_Q - g.loc[48].S_Q) <= 2 * g.loc[64].S_Q_eval_se,
                   "S_Q_64_minus_48": g.loc[64].S_Q - g.loc[48].S_Q, "two_se_eval_64": 2 * g.loc[64].S_Q_eval_se,
                   "sign_change_8_to_64_C_Lap": (bool(np.sign(g.loc[8].S_Lap) != np.sign(g.loc[64].S_Lap))
                                                 if g.loc[8].lap_ok and g.loc[64].lap_ok else "INCOMPLETE"),
                   "sign_Q_ne_sign_Lap_at_8": (bool(np.sign(g.loc[8].S_Q) != np.sign(g.loc[8].S_Lap)) if g.loc[8].lap_ok else "INCOMPLETE"),
                   "sign_Q_ne_sign_Lap_at_64": (bool(np.sign(g.loc[64].S_Q) != np.sign(g.loc[64].S_Lap)) if g.loc[64].lap_ok else "INCOMPLETE")})
    stab = pd.DataFrame(st)
    save(stab, "stability_diagnostics.csv")
    # ---- RQ2
    sm = pd.DataFrame()
    if len(f2):
        parts = []
        for kg in (3, 5):
            g = f2[f2.k_gen == kg]
            a = local_slacks(g, kg - 1, kg, ["replicate", "checkpoint"]).assign(k_gen=kg, side="K_GEN-1 -> K_GEN")
            b = local_slacks(g, kg, kg + 1, ["replicate", "checkpoint"]).assign(k_gen=kg, side="K_GEN -> K_GEN+1")
            parts += [a, b]
        sm = pd.concat(parts, ignore_index=True).assign(experiment_id=EXP, n=75, d=15)
        sm["gain_x_per_node"] = sm.gain_x / 75; sm["gain_y_per_node"] = sm.gain_y / 75
        rows = []
        for cp in RQ2_CPS:
            for crit, col in (("C_Q", "S_Q"), ("C_Lap", "S_Lap"), ("C_Q_eval", "S_Q_eval")):
                up = sm[(sm.checkpoint == cp) & (sm.side == "K_GEN-1 -> K_GEN")].set_index(["k_gen", "replicate"])
                d_ = []
                for rep in range(1, 9):
                    if (3, rep) in up.index and (5, rep) in up.index and col in up and pd.notna(up.loc[(3, rep), col]) \
                            and pd.notna(up.loc[(5, rep), col]):
                        dv = float(up.loc[(5, rep), col] - up.loc[(3, rep), col]); d_.append(dv)
                        rows.append({"row_type": "per_replicate", "criterion": crit, "checkpoint": cp, "replicate": rep,
                                     "S_K3_2to3": float(up.loc[(3, rep), col]), "S_K5_4to5": float(up.loc[(5, rep), col]),
                                     "delta_K5_minus_K3": dv, "status": "OK"})
                    else:
                        rows.append({"row_type": "per_replicate", "criterion": crit, "checkpoint": cp, "replicate": rep,
                                     "status": "INCOMPLETE"})
                d_ = np.array(d_)
                lo, hi, cov = ci_order(d_)
                v = ("NO_DATA" if len(d_) == 0 else "K_HARDER" if np.median(d_) < 0 and (d_ < 0).sum() >= 6
                     else "NOT_HARDER" if np.median(d_) >= 0 else "INCONCLUSIVE")
                rows.append({"row_type": "summary", "criterion": crit, "checkpoint": cp, "pairs": len(d_),
                             "median_delta": float(np.median(d_)) if len(d_) else np.nan, "n_negative": int((d_ < 0).sum()),
                             "median_ci_lo": lo, "median_ci_hi": hi, "ci_coverage": cov, "wilcoxon_exact_p": wil(d_),
                             "verdict_rule": v, "prereg_primary": crit == "C_Q" and cp == 32})
        save(pd.concat([sm.assign(row_type="local_slack"), pd.DataFrame(rows)], ignore_index=True), "signal_matched_k3_k5.csv")
    # ---- failures / incomplete
    fi = []
    for r in f[f.status != "FIT"].itertuples():
        fi.append({"experiment_id": EXP, "rq": r.rq, "condition": r.condition, "replicate": r.replicate, "k_fit": r.k_fit,
                   "start": r.start, "checkpoint": "", "kind": "FIT_FAILED", "detail": r.exception})
    for r in ok[ok.lap_status != "OK"].itertuples():
        fi.append({"experiment_id": EXP, "rq": r.rq, "condition": r.condition, "replicate": r.replicate, "k_fit": r.k_fit,
                   "start": r.start, "checkpoint": r.checkpoint, "kind": f"LAPLACE_{r.lap_status}",
                   "detail": f"grad_inf={r.lap_grad_inf}"})
    for r in ok[(ok.rq == "RQ1") & (ok.checkpoint == 8) & (ok.replication_status != "MATCH")].itertuples():
        fi.append({"experiment_id": EXP, "rq": "RQ1", "condition": r.condition, "replicate": r.replicate, "k_fit": r.k_fit,
                   "start": r.start, "checkpoint": 8, "kind": "REPLICATION_FAIL", "detail": r.replication_detail})
    for r in led[led.status.astype(str).str.startswith("NOT_RUN")].itertuples():
        fi.append({"experiment_id": EXP, "rq": r.rq, "condition": r.condition, "replicate": r.replicate, "k_fit": r.k_fit,
                   "start": r.start, "checkpoint": "", "kind": r.status, "detail": ""})
    save(pd.DataFrame(fi, columns=["experiment_id", "rq", "condition", "replicate", "k_fit", "start", "checkpoint", "kind", "detail"]),
         "failure_and_incomplete.csv")
    save(pd.DataFrame([{"experiment_id": EXP, "status": "NOT_PRODUCED",
                        "reason": "full-candidate K (1..7) evaluation was excluded before results by the pre-registered budget rule; RQ1 is a LOCAL 4-vs-5 comparison"}]),
         "full_k_selection.csv")
    save(it.assign(experiment_id=EXP), "iteration_trajectories.csv")
    figures(f1, it, loc, sm, stab)
    man = {"script": "tools/research_audit/analyze_mcem_stability.py", "mcem_fits": 0,
           "prereg_sha256": pre["sha256"], "gates_status": gates["status"],
           "runs": {k: {kk: v.get(kk) for kk in ("status", "fits", "fit_status_counts", "wall_clock_seconds", "git_head", "states_sha256")}
                    for k, v in runs.items()},
           "integrity": {"duplicate_keys": 0, "fit_failed": int((f.status != "FIT").sum()),
                         "laplace_not_ok": int((ok.lap_status != "OK").sum()),
                         "replication_fail_fits": int(((ok.rq == "RQ1") & (ok.checkpoint == 8) & (ok.replication_status != "MATCH")).sum()),
                         "max_abs_closure": float(np.nanmax(np.abs(loc.closure_error))) if len(loc) else None,
                         "max_abs_accounting": float(np.nanmax(np.abs(loc.accounting_error))) if len(loc) and "accounting_error" in loc else None,
                         "max_abs_identity_error_fit": float(np.nanmax(np.abs(ok.identity_error)))},
           "outputs_sha256": {p.name: sha256(p) for p in sorted(OUT.glob("*.csv"))},
           "figures_sha256": {p.name: sha256(p) for p in sorted(FIG.glob("*"))}}
    (OUT / "analysis_manifest.json").write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")
    print(json.dumps(man["integrity"], indent=1))
    return 0


def figures(f1, it, loc, sm, stab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "legend.frameon": False, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "lines.linewidth": 2})
    FIG.mkdir(parents=True, exist_ok=True)

    def sv(fig, name):
        for ext in ("png", "pdf"):
            p = FIG / f"{name}.{ext}"
            need(not p.exists(), f"refuse to overwrite {p}")
            fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
    if len(loc):
        # 1 C_Q slack per iteration (from the per-iteration C_Q trajectory)
        r1 = it[it.rq == "RQ1"]
        piv = r1.pivot_table(index=["condition", "replicate", "iteration"], columns=["k_fit", "start"], values="C_Q")
        sq = (piv[(4, 1)] + piv[(4, 2)]) / 2 - (piv[(5, 1)] + piv[(5, 2)]) / 2
        sq = sq.reset_index(name="S")
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
        for ax, c in zip(axes, "ABCD"):
            g = sq[sq.condition == c]
            for rep, h in g.groupby("replicate"):
                ax.plot(h.iteration, h.S, color=COL[c], alpha=0.35, lw=1)
            med = g.groupby("iteration").S.median()
            ax.plot(med.index, med.values, color="#0b0b0b", lw=2, label="median")
            ax.axhline(0, color=INK2, lw=0.8); ax.axvline(8, color=INK2, lw=0.6, ls=":")
            ax.set_title(LAB[c], fontsize=9); ax.set_xlabel("MCEM iteration")
        axes[0].set_ylabel("C_Q(4) - C_Q(5)  (>0 favours K=5)"); axes[0].legend(fontsize=7)
        fig.suptitle("RQ1: local C_Q 4->5 slack along the same fits, iterations 1..64 (n=75, K_GEN=5, d=15; thin = replicate)", fontsize=9)
        fig.tight_layout(); sv(fig, "fig1_cq_slack_vs_iteration")
        # 2 C_Lap slack vs checkpoint
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
        for ax, c in zip(axes, "ABCD"):
            g = loc[(loc.condition == c) & loc.lap_ok]
            for rep, h in g.groupby("replicate"):
                ax.plot(h.checkpoint, h.S_Lap, color=COL[c], alpha=0.4, lw=1, marker=MK[c], ms=3)
            med = g.groupby("checkpoint").S_Lap.median()
            ax.plot(med.index, med.values, color="#0b0b0b", lw=2)
            ax.axhline(0, color=INK2, lw=0.8); ax.set_title(LAB[c], fontsize=9); ax.set_xlabel("checkpoint")
            ax.set_xticks(RQ1_CPS)
        axes[0].set_ylabel("C_Lap(4) - C_Lap(5)")
        fig.suptitle("RQ1: local C_Lap 4->5 slack at checkpoints 8..64 (Laplace-OK datasets; black = median)", fontsize=9)
        fig.tight_layout(); sv(fig, "fig2_clap_slack_vs_checkpoint")
        # 3 paired differences
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
        for ax, crit, col in zip(axes, ("C_Q", "C_Lap", "C_Q_eval"), ("S_Q", "S_Lap", "S_Q_eval")):
            for c in "BCD":
                vals = []
                for cp in RQ1_CPS:
                    a = loc[(loc.condition == "A") & (loc.checkpoint == cp)].set_index("replicate")[col]
                    b = loc[(loc.condition == c) & (loc.checkpoint == cp)].set_index("replicate")[col]
                    d_ = (b - a).dropna(); vals.append(d_)
                    ax.scatter([cp + {"B": -1.5, "C": 0, "D": 1.5}[c]] * len(d_), d_, color=COL[c], marker=MK[c], s=12, alpha=0.5)
                ax.plot(RQ1_CPS, [np.median(v) for v in vals], color=COL[c], marker=MK[c], label=f"{c}-A median")
            ax.axhline(0, color=INK2, lw=0.8); ax.set_title(f"{crit}: paired slack difference", fontsize=9)
            ax.set_xlabel("checkpoint"); ax.set_xticks(RQ1_CPS)
        axes[0].set_ylabel("slack(condition) - slack(A), same replicate"); axes[0].legend(fontsize=7)
        fig.suptitle("RQ1: is the signal-strength effect on the 4->5 slack maintained? (dots = replicates)", fontsize=9)
        fig.tight_layout(); sv(fig, "fig3_paired_differences")
        # 4 Gram / eta changes
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
        for ax, col in zip(axes, ("gram_rel_change", "eta_x_rel_change", "eta_y_rel_change")):
            for c in "ABCD":
                g = r1[(r1.condition == c) & (r1.k_fit == 5)].groupby("iteration")[col].median()
                ax.plot(g.index, g.values, color=COL[c], label=LAB[c])
            ax.axhline(TOL, color="#d03b3b", lw=1, ls="--", label="pre-registered 0.01")
            ax.set_yscale("log"); ax.set_title(f"median {col}, K_fit=5", fontsize=9); ax.set_xlabel("iteration")
        axes[0].legend(fontsize=7)
        fig.suptitle("RQ1: per-iteration relative changes (rotation-invariant quantities)", fontsize=9)
        fig.tight_layout(); sv(fig, "fig4_gram_eta_changes")
        # 5 Q decomposition
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
        for ax, col, t in zip(axes, ("gain_x", "gain_y", "S_Q"), ("4->5 fit gain X", "4->5 fit gain Y", "C_Q slack 4->5")):
            for c in "ABCD":
                g = loc[loc.condition == c].groupby("checkpoint")[col].median()
                ax.plot(g.index, g.values, color=COL[c], marker=MK[c], label=LAB[c])
            ax.set_title(f"median {t}", fontsize=9); ax.set_xlabel("checkpoint"); ax.set_xticks(RQ1_CPS)
        axes[2].axhline(0, color=INK2, lw=0.8); axes[0].legend(fontsize=7)
        fig.suptitle("RQ1: X / Y fit gains of the 5th dimension (Z prior cost 212.8 and penalty 47.5 are constant)", fontsize=9)
        fig.tight_layout(); sv(fig, "fig5_q_decomposition")
        # 6 start differences
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for c in "ABCD":
            g = loc[loc.condition == c]
            axes[0].plot(g.groupby("checkpoint").start_gap_C_Q_khi.median(), color=COL[c], marker=MK[c], label=LAB[c])
            g64 = g[g.checkpoint == 64]
            axes[1].scatter(g64.S_Q_start1, g64.S_Q_start2, color=COL[c], marker=MK[c], s=28, label=LAB[c])
        axes[0].set_title("median |C_Q(start1) - C_Q(start2)|, K_fit=5", fontsize=9); axes[0].set_xlabel("checkpoint")
        lim = [loc.S_Q_start1.min() - 10, loc.S_Q_start1.max() + 10]
        axes[1].plot(lim, lim, color=INK2, ls="--", lw=0.8); axes[1].axhline(0, color=INK2, lw=0.6); axes[1].axvline(0, color=INK2, lw=0.6)
        axes[1].set_xlabel("S_Q(4->5), start 1"); axes[1].set_ylabel("start 2"); axes[1].set_title("per-start slack at checkpoint 64", fontsize=9)
        axes[0].legend(fontsize=7)
        fig.tight_layout(); sv(fig, "fig6_start_differences")
        # 7 stability vs selection
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        g64 = loc[loc.checkpoint == 64]
        for c in "ABCD":
            g = g64[g64.condition == c]
            ax.scatter(g.max_gram_last8_khi, g.S_Q, color=COL[c], marker=MK[c], s=32, label=LAB[c])
        ax.axvline(TOL, color="#d03b3b", ls="--", lw=1); ax.axhline(0, color=INK2, lw=0.8); ax.set_xscale("log")
        ax.set_xlabel("max gram_rel_change over iterations 57-64 (K_fit=5, worse start)")
        ax.set_ylabel("C_Q slack 4->5 at 64"); ax.set_title("Stability vs local selection at checkpoint 64", fontsize=9)
        ax.legend(fontsize=7); fig.tight_layout(); sv(fig, "fig7_stability_vs_selection")
    if len(sm):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        up = sm[sm.side == "K_GEN-1 -> K_GEN"]
        for ax, col, t in zip(axes, ("S_Q", "S_Lap", "gain_y_per_node"), ("C_Q slack K_GEN-1 -> K_GEN", "C_Lap slack", "fit gain Y / n")):
            for cp in (32,):
                g = up[up.checkpoint == cp].pivot(index="replicate", columns="k_gen", values=col)
                for rep, row in g.iterrows():
                    ax.plot([0, 1], [row.get(3), row.get(5)], color="#c3c2b7", lw=0.8)
                ax.scatter([0] * len(g), g[3], color=COL["K3"], marker="o", s=30, label="K_GEN=3 (2->3)")
                ax.scatter([1] * len(g), g[5], color=COL["K5"], marker="s", s=30, label="K_GEN=5 (4->5)")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["K_GEN=3", "K_GEN=5"]); ax.set_title(f"{t}, checkpoint 32", fontsize=9)
            if col != "gain_y_per_node":
                ax.axhline(0, color=INK2, lw=0.8)
        axes[0].legend(fontsize=7)
        fig.suptitle("RQ2: per-direction signal matched (F^T F eigenvalues 2.5, w^2 = 1); grey lines join the same replicate (shared Z directions and mask)", fontsize=9)
        fig.tight_layout(); sv(fig, "fig8_signal_matched_k3_k5")


if __name__ == "__main__":
    sys.exit(main())
