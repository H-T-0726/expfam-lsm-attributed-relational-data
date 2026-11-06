"""Analysis of mcem_noise_separation_20261106 (rules R1-R8 of the pre-registration).

Usage:  python tools/research_audit/analyze_mcem_noise_separation.py
Reads   expfam/results/k_selection/mcem_noise_separation_20261106/{phase3_*,phase4_*}/ and PR #156 results (read only)
Writes  .../analysis/*.csv, .../figures/*.png|pdf, .../analysis/analysis_manifest.json  (refuses to overwrite)
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import run_mcem_noise_separation as ns  # noqa: E402

OUT = ns.OUT_ROOT
AN = OUT / "analysis"
FIG = OUT / "figures"
SRC = ns.SRC_RESULTS
CONDS = ns.CONDS
COLORS = {"A": "#2a6fdb", "B": "#e8603c", "C": "#1aab7a", "D": "#f0a30a"}
LABEL = {"A": "A baseline", "B": "B F-strong", "C": "C w-strong", "D": "D F+w-strong"}
EID = ns.EXPERIMENT_ID


EXEC = OUT / "exec_v2"


def _load_agg(sname: str, names) -> dict:
    d = EXEC / sname / "aggregate"
    info = json.loads((d / "runinfo.json").read_text(encoding="utf-8"))
    ns.need(info["prereg_sha256"] == ns.prereg_hash(), f"{sname}: prereg changed")
    out = {k: pd.read_csv(d / f"{k}.csv") for k in names}
    out["status"] = info["status"]
    return out


def load_phase3():
    return _load_agg("phase3", ("state_rows", "probe_rows", "replicate_rows", "ledger"))


def load_phase4():
    parts = {k: [] for k in ("checkpoint_rows", "probe_rows", "replicate_rows", "iteration_rows", "ledger")}
    status = {}
    for L in ns.P4_LS:
        if not (EXEC / f"phase4_L{L}" / "aggregate").exists():
            status[L] = "NOT_RUN"
            continue
        x = _load_agg(f"phase4_L{L}", parts.keys())
        status[L] = x["status"]
        for k in parts:
            if len(x[k]):
                parts[k].append(x[k])
    out = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in parts.items()}
    out["status"] = status
    return out


def observed_diag():
    it = pd.concat([pd.read_csv(SRC / s / "iteration_rows.csv") for s in ("rq1_AB", "rq1_CD")], ignore_index=True)
    out = []
    for cp, lo in ((32, 25), (64, 57)):
        w = it[(it.iteration >= lo) & (it.iteration <= cp)]
        g = w.groupby(["condition", "replicate", "k_fit", "start"])[["gram_rel_change", "eta_x_rel_change", "eta_y_rel_change"]].median()
        g = g.add_prefix("obs_median_").reset_index()
        g["checkpoint"] = cp
        out.append(g)
    return pd.concat(out, ignore_index=True), it


def r1_label(x):
    if not np.isfinite(x):
        return "NA"
    return "ONE_STEP_NOISE_DOMINATED" if x <= 0.2 else ("ONE_STEP_SYSTEMATIC_DOMINATED" if x >= 0.5 else "MIXED")


def r2_label(x):
    return "CONSISTENT_WITH_ONE_STEP_NOISE" if 0.67 <= x <= 1.5 else ("EXCESS_MOVEMENT" if x > 1.5 else "LESS_THAN_ONE_STEP")


def paired(diff: np.ndarray):
    d = np.sort(np.asarray(diff, float))
    n = len(d)
    med = float(np.median(d))
    pos = int((d > 0).sum())
    lo, hi = (float(d[1]), float(d[-2])) if n == 8 else (float(d[0]), float(d[-1]))
    verdict = "MAINTAINED" if (med > 0 and pos >= 6) else ("NOT_MAINTAINED" if med <= 0 else "INCONCLUSIVE")
    return med, pos, lo, hi, verdict


def savefig(fig, name, made):
    for ext in ("png", "pdf"):
        p = FIG / f"{name}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        made.append(p)
    plt.close(fig)


def main() -> int:
    for d in (AN, FIG):
        if d.exists():
            print(f"refusing to overwrite {d}")
            return 1
        d.mkdir(parents=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "figure.facecolor": "white"})
    p3 = load_phase3()
    p4 = load_phase4()
    obs, it = observed_diag()
    S, P, R, LG = p3["state_rows"], p3["probe_rows"], p3["replicate_rows"], p3["ledger"]
    made: list[Path] = []
    keys = ["condition", "replicate", "k_fit", "start"]

    # ---- definitions -------------------------------------------------------
    defs = pd.DataFrame([{"name": k, "definition": v} for k, v in ns.PREREG["definitions"].items()]
                        + [{"name": k, "definition": v} for k, v in ns.PREREG["rules"].items()])
    defs.to_csv(AN / "diagnostic_definitions.csv", index=False)

    # ---- failures ----------------------------------------------------------
    fails = []
    for _, x in LG[LG.status != "OK"].iterrows():
        fails.append({"phase": 3, **{k: x[k] for k in keys}, "kind": f"FIT_{x.status}", "detail": x.reason})
    for _, x in S[S.lap_status != "OK"].iterrows():
        fails.append({"phase": 3, **{k: x[k] for k in keys}, "checkpoint": x.checkpoint, "kind": "LAPLACE_" + str(x.lap_status),
                      "detail": f"grad_inf={x.lap_grad_inf}"})
    if len(p4["ledger"]):
        for _, x in p4["ledger"][p4["ledger"].status != "OK"].iterrows():
            fails.append({"phase": 4, **{k: x[k] for k in keys}, "L_fit": x.L_fit, "kind": f"FIT_{x.status}", "detail": x.reason})
        cr = p4["checkpoint_rows"]
        for _, x in cr[cr.lap_status != "OK"].iterrows():
            fails.append({"phase": 4, **{k: x[k] for k in keys}, "L_fit": x.L_fit, "checkpoint": x.checkpoint,
                          "kind": "LAPLACE_" + str(x.lap_status), "detail": f"grad_inf={x.lap_grad_inf}"})
    pd.DataFrame(fails, columns=["phase", *keys, "L_fit", "checkpoint", "kind", "detail"]).to_csv(AN / "failure_and_incomplete.csv", index=False)

    # ---- R1 / R2 / R3: fixed-theta one-step vs observed ----------------------
    ft = P.merge(obs, on=keys + ["checkpoint"], how="left")
    for q, col in (("gram", "one_step_gram"), ("eta_x", "one_step_eta_x"), ("eta_y", "one_step_eta_y")):
        ft[f"R2_ratio_{q}"] = ft[f"obs_median_{q}_rel_change"] / ft[f"median_{col}"]
        ft[f"R2_{q}"] = ft[f"R2_ratio_{q}"].map(r2_label)
    for q in ("x", "y"):
        ft[f"R3_share_eta_{q}"] = ft[f"median_single_draw_eta_{q}"] / ft[f"obs_median_eta_{q}_rel_change"]
        ft[f"R3_eta_{q}"] = np.where(ft[f"R3_share_eta_{q}"] >= 0.8, "ETA_DIAGNOSTIC_SAMPLING_DOMINATED", "NOT_DOMINATED")
    for nm in ("G", "w0", "w"):
        ft[f"R1_{nm}"] = ft[f"{nm}_drift_share"].map(r1_label)
    ft.insert(0, "experiment_id", EID)
    ft.to_csv(AN / "fixed_theta_noise.csv", index=False)

    rows = []
    for (cond, cp), g in ft.groupby(["condition", "checkpoint"]):
        r = {"condition": cond, "checkpoint": cp, "fits": len(g)}
        for nm in ("G", "w0", "w"):
            r[f"median_{nm}_drift_share"] = g[f"{nm}_drift_share"].median()
            r[f"median_{nm}_drift_rel"] = g[f"{nm}_drift_rel"].median()
            r[f"median_{nm}_noise_rel"] = g[f"{nm}_noise_rel"].median()
            for lab in ("ONE_STEP_NOISE_DOMINATED", "MIXED", "ONE_STEP_SYSTEMATIC_DOMINATED"):
                r[f"n_{nm}_{lab}"] = int((g[f"R1_{nm}"] == lab).sum())
        r["median_G_mean_reversion_cos"] = g.G_mean_reversion_cos.median()
        r["frac_G_cos_positive"] = float((g.G_mean_reversion_cos > 0).mean())
        r["frac_w0_toward_center"] = float(g.w0_step_toward_center.astype(bool).mean())
        r["frac_w_toward_center"] = float(g.w_step_toward_center.astype(bool).mean())
        for q in ("gram", "eta_x", "eta_y"):
            r[f"median_R2_ratio_{q}"] = g[f"R2_ratio_{q}"].median()
            r[f"n_R2_{q}_consistent"] = int((g[f"R2_{q}"] == "CONSISTENT_WITH_ONE_STEP_NOISE").sum())
            r[f"n_R2_{q}_excess"] = int((g[f"R2_{q}"] == "EXCESS_MOVEMENT").sum())
            r[f"n_R2_{q}_less"] = int((g[f"R2_{q}"] == "LESS_THAN_ONE_STEP").sum())
        rows.append(r)
    allr = {"condition": "ALL", "checkpoint": 64, "fits": int((ft.checkpoint == 64).sum())}
    g = ft[ft.checkpoint == 64]
    for nm in ("G", "w0", "w"):
        allr[f"median_{nm}_drift_share"] = g[f"{nm}_drift_share"].median()
        for lab in ("ONE_STEP_NOISE_DOMINATED", "MIXED", "ONE_STEP_SYSTEMATIC_DOMINATED"):
            allr[f"n_{nm}_{lab}"] = int((g[f"R1_{nm}"] == lab).sum())
    allr["median_G_mean_reversion_cos"] = g.G_mean_reversion_cos.median()
    allr["frac_G_cos_positive"] = float((g.G_mean_reversion_cos > 0).mean())
    for q in ("gram", "eta_x", "eta_y"):
        allr[f"median_R2_ratio_{q}"] = g[f"R2_ratio_{q}"].median()
        allr[f"n_R2_{q}_consistent"] = int((g[f"R2_{q}"] == "CONSISTENT_WITH_ONE_STEP_NOISE").sum())
        allr[f"n_R2_{q}_excess"] = int((g[f"R2_{q}"] == "EXCESS_MOVEMENT").sum())
        allr[f"n_R2_{q}_less"] = int((g[f"R2_{q}"] == "LESS_THAN_ONE_STEP").sum())
    rows.append(allr)
    pd.DataFrame(rows).to_csv(AN / "one_step_decomposition_summary.csv", index=False)

    # ---- eta diagnostics comparison (R3, R4) ---------------------------------
    p64 = ft[ft.checkpoint == 64].copy()
    p32 = ft[ft.checkpoint == 32][keys + ["eta_x_pm_mc_error", "eta_y_pm_mc_error"]].rename(
        columns={"eta_x_pm_mc_error": "e32_x", "eta_y_pm_mc_error": "e32_y"})
    p64 = p64.merge(p32, on=keys)
    for q in ("x", "y"):
        p64[f"R4_ratio_eta_{q}"] = p64[f"D_eta_{q}_pm_32_64"] / np.sqrt(p64[f"e32_{q}"] ** 2 + p64[f"eta_{q}_pm_mc_error"] ** 2)
        p64[f"R4_eta_{q}"] = np.where(p64[f"R4_ratio_eta_{q}"] > 2, "PARAMETER_CHANGE_DETECTED", "NOT_DETECTED")
    smode = S[S.checkpoint == 48][keys + ["D64_eta_x_mode", "D64_eta_y_mode", "D64_G"]]
    p64 = p64.merge(smode, on=keys, how="left")
    eta_rows = []
    for cond, g in p64.groupby("condition"):
        eta_rows.append({
            "condition": cond, "fits": len(g),
            "obs_eta_x_diag_median": g.obs_median_eta_x_rel_change.median(),
            "one_step_eta_x_median": g.median_one_step_eta_x.median(),
            "single_draw_eta_x_median": g.median_single_draw_eta_x.median(),
            "pm_eta_x_mc_error_median": g.eta_x_pm_mc_error.median(),
            "pm_eta_x_change_32_64_median": g.D_eta_x_pm_32_64.median(),
            "mode_eta_x_change_48_64_median": g.D64_eta_x_mode.median(),
            "n_R3_eta_x_sampling_dominated": int((g.R3_eta_x == "ETA_DIAGNOSTIC_SAMPLING_DOMINATED").sum()),
            "n_R4_eta_x_change_detected": int((g.R4_eta_x == "PARAMETER_CHANGE_DETECTED").sum()),
            "obs_eta_y_diag_median": g.obs_median_eta_y_rel_change.median(),
            "single_draw_eta_y_median": g.median_single_draw_eta_y.median(),
            "pm_eta_y_mc_error_median": g.eta_y_pm_mc_error.median(),
            "pm_eta_y_change_32_64_median": g.D_eta_y_pm_32_64.median(),
            "mode_eta_y_change_48_64_median": g.D64_eta_y_mode.median(),
            "n_R3_eta_y_sampling_dominated": int((g.R3_eta_y == "ETA_DIAGNOSTIC_SAMPLING_DOMINATED").sum()),
            "n_R4_eta_y_change_detected": int((g.R4_eta_y == "PARAMETER_CHANGE_DETECTED").sum()),
            "eta_y_plugin_minus_pm_rel_median": g.eta_y_plugin_minus_pm_rel.median(),
            "eta_y_plugin_minus_pm_mean_median": g.eta_y_plugin_minus_pm_mean.median(),
            "gram_obs_median": g.obs_median_gram_rel_change.median(),
            "gram_one_step_median": g.median_one_step_gram.median(),
            "G_change_48_64_median": g.D64_G.median()})
    pd.DataFrame(eta_rows).to_csv(AN / "eta_diagnostic_comparison.csv", index=False)
    p64.to_csv(AN / "eta_diagnostic_per_fit.csv", index=False)

    # ---- R5 long-run movement ------------------------------------------------
    lag = S.pivot_table(index=keys, columns="checkpoint",
                        values=[f"D64_{q}" for q in ("G", "w0", "w", "eta_x_mode", "eta_y_mode")])
    r5 = []
    for q in ("G", "w0", "w", "eta_x_mode", "eta_y_mode"):
        d16, d48 = lag[(f"D64_{q}", 16)], lag[(f"D64_{q}", 48)]
        ok = d16.notna() & d48.notna()
        for cond in ("ALL", *CONDS):
            sel = ok if cond == "ALL" else ok & (lag.index.get_level_values("condition") == cond)
            a, b = d16[sel], d48[sel]
            cnt = int((a > b).sum()); n = int(sel.sum())
            p = binomtest(cnt, n, 0.5).pvalue if n else float("nan")
            ratio = float(np.median(a / b)) if n else float("nan")
            verdict = ("SYSTEMATIC_MOVEMENT" if (p < 0.01 and cnt / n > 0.5 and ratio > 1.1) else "NO_DETECTABLE_SYSTEMATIC_MOVEMENT") \
                if cond == "ALL" else "DESCRIPTIVE"
            row = {"quantity": q, "condition": cond, "fits": n, "n_D16_gt_D48": cnt, "binom_p_two_sided": p,
                   "median_ratio_D16_over_D48": ratio, "verdict": verdict}
            for c in (8, 16, 24, 32, 48):
                row[f"median_D64_from_{c}"] = float(lag[(f"D64_{q}", c)][sel].median())
            r5.append(row)
    pd.DataFrame(r5).to_csv(AN / "long_run_movement.csv", index=False)

    # ---- slack: L=5 / L=20 / L=80 fixed theta, C_Lap, fit's own (R6, R7) -----
    rr = R[R.kind.str.contains("eval")]
    pv = rr.pivot_table(index=["condition", "replicate", "checkpoint", "r"], columns=["k_fit", "start"], values="C_Q_L5")
    sl5 = ((pv[(4, 1)] - pv[(5, 1)]) + (pv[(4, 2)] - pv[(5, 2)])) / 2
    sl5 = sl5.rename("S_L5").reset_index()
    sd5 = sl5.groupby(["condition", "replicate", "checkpoint"]).S_L5.agg(S_L5_sd="std", S_L5_mean="mean").reset_index()
    pp = P.pivot_table(index=["condition", "replicate", "checkpoint"], columns=["k_fit", "start"], values="C_Q_L80")
    sl80 = (((pp[(4, 1)] - pp[(5, 1)]) + (pp[(4, 2)] - pp[(5, 2)])) / 2).rename("S_L80").reset_index()
    l20 = []
    for g_ in range(4):
        q = P.pivot_table(index=["condition", "replicate", "checkpoint"], columns=["k_fit", "start"], values=f"C_Q_L20_g{g_}")
        l20.append((((q[(4, 1)] - q[(5, 1)]) + (q[(4, 2)] - q[(5, 2)])) / 2).rename(f"S_L20_g{g_}"))
    l20 = pd.concat(l20, axis=1)
    l20["S_L20_sd"] = l20.std(axis=1, ddof=1)
    ps = S.pivot_table(index=["condition", "replicate", "checkpoint"], columns=["k_fit", "start"], values="C_Lap")
    slap = (((ps[(4, 1)] - ps[(5, 1)]) + (ps[(4, 2)] - ps[(5, 2)])) / 2).rename("S_Lap_recomputed").reset_index()
    loc = pd.read_csv(SRC / "analysis" / "local_k4_k5_comparison.csv")[["condition", "replicate", "checkpoint", "S_Q", "S_Q_eval",
                                                                         "S_Q_eval_se", "S_Lap"]]
    loc = loc.rename(columns={"S_Q": "S_fit_pr156", "S_Q_eval": "S_eval20_pr156", "S_Q_eval_se": "S_eval20_se_pr156",
                              "S_Lap": "S_Lap_pr156"})
    sl = slap.merge(loc, on=["condition", "replicate", "checkpoint"], how="left")
    sl = sl.merge(sl80, on=["condition", "replicate", "checkpoint"], how="left").merge(sd5, on=["condition", "replicate", "checkpoint"], how="left")
    sl = sl.merge(l20[["S_L20_sd"]].reset_index(), on=["condition", "replicate", "checkpoint"], how="left")
    sl["S_L80_se"] = sl.S_L5_sd / math.sqrt(ns.R_CHAINS)
    sl["S_Lap_recompute_diff"] = sl.S_Lap_recomputed - sl.S_Lap_pr156
    sl.insert(0, "experiment_id", EID)
    sl.to_csv(AN / "slack_by_checkpoint.csv", index=False)
    w = sl.pivot_table(index=["condition", "replicate"], columns="checkpoint",
                       values=["S_fit_pr156", "S_L80", "S_L80_se", "S_Lap_recomputed", "S_L5_sd"])
    r7 = pd.DataFrame({
        "abs_dS_fit_48_64": (w[("S_fit_pr156", 64)] - w[("S_fit_pr156", 48)]).abs(),
        "S_L5_sd_64": w[("S_L5_sd", 64)],
        "S_L80_32": w[("S_L80", 32)], "S_L80_64": w[("S_L80", 64)],
        "abs_dS_L80_32_64": (w[("S_L80", 64)] - w[("S_L80", 32)]).abs(),
        "se_dS_L80_32_64": np.sqrt(w[("S_L80_se", 64)] ** 2 + w[("S_L80_se", 32)] ** 2),
        "abs_dS_Lap_32_64": (w[("S_Lap_recomputed", 64)] - w[("S_Lap_recomputed", 32)]).abs(),
        "sign_L80_32": np.sign(w[("S_L80", 32)]), "sign_L80_64": np.sign(w[("S_L80", 64)]),
    }).reset_index()
    r7["z_dS_L80_32_64"] = r7.abs_dS_L80_32_64 / r7.se_dS_L80_32_64
    r7.insert(0, "experiment_id", EID)
    r7.to_csv(AN / "slack_mc_noise.csv", index=False)

    eff = []
    for cp in (32, 64):
        x = sl[sl.checkpoint == cp].set_index(["condition", "replicate"])
        for crit, col in (("S_L80", "S_L80"), ("S_Lap", "S_Lap_recomputed"), ("S_fit_pr156", "S_fit_pr156")):
            for c in ("B", "C", "D"):
                dd = (x.loc[c, col] - x.loc["A", col]).dropna()
                med, pos, lo, hi, verdict = paired(dd.values)
                eff.append({"criterion": crit, "checkpoint": cp, "contrast": f"{c}-A", "pairs": len(dd), "median": med,
                            "n_positive": pos, "median_ci_lo": lo, "median_ci_hi": hi, "verdict": verdict,
                            "prereg_primary": crit == "S_L80"})
    pd.DataFrame(eff).to_csv(AN / "paired_effects_L80.csv", index=False)

    qn = P.groupby(["condition", "k_fit", "checkpoint"])[["C_Q_L5_sd", "Q_X_L5_sd", "Q_Y_L5_sd", "Q_Z_L5_sd"]].median().reset_index()
    qn.to_csv(AN / "q_component_noise.csv", index=False)

    # ---- Phase 4 L pilot ----------------------------------------------------
    p4rows, p4sum = [], []
    if len(p4["probe_rows"]):
        it4 = p4["iteration_rows"]
        base5 = P[(P.checkpoint == 32) & (P.start == 1) & P.condition.isin(ns.P4_CONDS) & P.replicate.isin(ns.P4_REPS)].copy()
        base5["L_fit"] = 5
        o5 = it[(it.iteration >= 25) & (it.iteration <= 32) & (it.start == 1)].groupby(["condition", "replicate", "k_fit"])[
            ["gram_rel_change", "eta_x_rel_change", "eta_y_rel_change"]].median().add_prefix("plateau_").reset_index()
        base5 = base5.merge(o5, on=["condition", "replicate", "k_fit"], how="left")
        lap5 = S[(S.checkpoint == 32) & (S.start == 1)][["condition", "replicate", "k_fit", "C_Lap"]]
        base5 = base5.merge(lap5, on=["condition", "replicate", "k_fit"], how="left")
        sd5k = rr[(rr.checkpoint == 32) & (rr.start == 1)].groupby(["condition", "replicate", "k_fit"]).C_Q_L5.std().rename("C_Q_L5_eval_sd").reset_index()
        base5 = base5.merge(sd5k, on=["condition", "replicate", "k_fit"], how="left")
        pr4 = p4["probe_rows"].copy()
        o4 = it4[(it4.iteration >= 25) & (it4.iteration <= 32)].groupby(["L_fit", "condition", "replicate", "k_fit"])[
            ["gram_rel_change", "eta_x_rel_change", "eta_y_rel_change"]].median().add_prefix("plateau_").reset_index()
        pr4 = pr4.merge(o4, on=["L_fit", "condition", "replicate", "k_fit"], how="left")
        lap4 = p4["checkpoint_rows"][p4["checkpoint_rows"].checkpoint == 32][["L_fit", "condition", "replicate", "k_fit", "C_Lap"]]
        pr4 = pr4.merge(lap4, on=["L_fit", "condition", "replicate", "k_fit"], how="left")
        r4 = p4["replicate_rows"]
        sd4 = r4[r4.kind == "eval"].groupby(["L_fit", "condition", "replicate", "k_fit"]).C_Q_L5.std().rename("C_Q_L5_eval_sd").reset_index()
        pr4 = pr4.merge(sd4, on=["L_fit", "condition", "replicate", "k_fit"], how="left")
        cols = ["L_fit", "condition", "replicate", "k_fit", "C_Q_L80", "C_Q_L5_eval_sd", "C_Lap", "G_drift_share", "G_noise_rel",
                "median_one_step_gram", "median_single_draw_eta_x", "median_single_draw_eta_y", "plateau_gram_rel_change",
                "plateau_eta_x_rel_change", "plateau_eta_y_rel_change", "eta_x_pm_mc_error"]
        allp = pd.concat([base5[cols], pr4[cols]], ignore_index=True)
        allp.insert(0, "experiment_id", EID)
        allp.to_csv(AN / "phase4_L_pilot_fits.csv", index=False)
        wide = allp.pivot_table(index=["L_fit", "condition", "replicate"], columns="k_fit", values=["C_Q_L80", "C_Q_L5_eval_sd", "C_Lap"])
        sl4 = pd.DataFrame({"S_L80_start1": wide[("C_Q_L80", 4)] - wide[("C_Q_L80", 5)],
                            "S_L80_se": np.sqrt(wide[("C_Q_L5_eval_sd", 4)] ** 2 + wide[("C_Q_L5_eval_sd", 5)] ** 2) / math.sqrt(ns.R_CHAINS),
                            "S_Lap_start1": wide[("C_Lap", 4)] - wide[("C_Lap", 5)]}).reset_index()
        sl4.to_csv(AN / "phase4_L_pilot_slack.csv", index=False)
        for cond in ns.P4_CONDS:
            ref = sl4[(sl4.L_fit == 5) & (sl4.condition == cond)].set_index("replicate")
            fr5 = allp[(allp.L_fit == 5) & (allp.condition == cond)]
            for L in ns.P4_LS:
                cur = sl4[(sl4.L_fit == L) & (sl4.condition == cond)].set_index("replicate")
                if cur.empty:
                    continue
                d = (cur.S_L80_start1 - ref.S_L80_start1)
                thr = 2 * np.sqrt(cur.S_L80_se ** 2 + ref.S_L80_se ** 2)
                same = (np.sign(d) == np.sign(d.iloc[0])).all()
                verdict = "L_CHANGES_SLACK" if (same and (d.abs() > thr).all()) else "NO_CLEAR_L_EFFECT"
                frL = allp[(allp.L_fit == L) & (allp.condition == cond)]
                p4sum.append({
                    "condition": cond, "L_fit": L, "reps": len(d),
                    "median_dS_L80_vs_L5": float(d.median()), "dS_L80_values": json.dumps([round(v, 1) for v in d.values]),
                    "thresholds_2se": json.dumps([round(v, 1) for v in thr.values]), "R8_slack_verdict": verdict,
                    "median_dS_Lap_vs_L5": float((cur.S_Lap_start1 - ref.S_Lap_start1).median()),
                    "median_S_L80_L5": float(ref.S_L80_start1.median()), "median_S_L80_L": float(cur.S_L80_start1.median()),
                    "median_S_Lap_L5": float(ref.S_Lap_start1.median()), "median_S_Lap_L": float(cur.S_Lap_start1.median()),
                    "plateau_gram_ratio_L_over_5": float(frL.plateau_gram_rel_change.median() / fr5.plateau_gram_rel_change.median()),
                    "predicted_ratio_1_over_sqrtL": math.sqrt(5 / L),
                    "plateau_eta_x_ratio_L_over_5": float(frL.plateau_eta_x_rel_change.median() / fr5.plateau_eta_x_rel_change.median()),
                    "one_step_G_noise_ratio_L_over_5": float(frL.G_noise_rel.median() / fr5.G_noise_rel.median()),
                    "single_draw_eta_x_ratio_L_over_5": float(frL.median_single_draw_eta_x.median() / fr5.median_single_draw_eta_x.median())})
        pd.DataFrame(p4sum).to_csv(AN / "phase4_L_pilot_summary.csv", index=False)

    # ---- figures -------------------------------------------------------------
    # fig1: observed diagnostics vs fixed-theta one-step distribution (checkpoint 64)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, q, col in zip(axes, ("gram", "eta_x", "eta_y"), ("one_step_gram", "one_step_eta_x", "one_step_eta_y")):
        g = ft[ft.checkpoint == 64]
        for c in CONDS:
            h = g[g.condition == c]
            ax.scatter(h[f"median_{col}"], h[f"obs_median_{q}_rel_change"], s=14, color=COLORS[c], label=LABEL[c], alpha=0.8)
        lo = min(g[f"median_{col}"].min(), g[f"obs_median_{q}_rel_change"].min()) * 0.9
        hi = max(g[f"median_{col}"].max(), g[f"obs_median_{q}_rel_change"].max()) * 1.1
        ax.plot([lo, hi], [lo, hi], color="#555", lw=1)
        ax.set_xlabel(f"fixed-theta one-step {q} change (median of 16)")
        ax.set_ylabel(f"observed {q}_rel_change (median, it 57-64)")
        ax.set_title(q)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Observed per-iteration changes vs one-step changes from a FIXED archived state (checkpoint 64; line = equality)")
    savefig(fig, "fig1_observed_vs_fixed_theta_one_step", made)

    # fig2: drift share and mean reversion
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    g = ft[ft.checkpoint == 64]
    for i, nm in enumerate(("G", "w0", "w")):
        for j, c in enumerate(CONDS):
            v = g[g.condition == c][f"{nm}_drift_share"]
            axes[0].scatter(np.full(len(v), i * 5 + j) + np.random.default_rng(0).uniform(-0.2, 0.2, len(v)), v, s=10, color=COLORS[c],
                            label=LABEL[c] if i == 0 else None)
    axes[0].axhline(0.2, color="#888", ls="--", lw=0.8); axes[0].axhline(0.5, color="#888", ls="--", lw=0.8)
    axes[0].set_xticks([1.5, 6.5, 11.5]); axes[0].set_xticklabels(["G = F F^T", "w0", "w"])
    axes[0].set_ylabel("systematic share of one-step change (R1)"); axes[0].legend(fontsize=7, frameon=False)
    for c in CONDS:
        v = g[g.condition == c].G_mean_reversion_cos
        axes[1].hist(v, bins=np.linspace(-1, 1, 21), histtype="step", color=COLORS[c], label=LABEL[c])
    axes[1].set_xlabel("cos(systematic one-step move, direction to the long-run centre)")
    axes[1].set_ylabel("fits")
    fig.suptitle("One-step update from a fixed state: systematic part and where it points (checkpoint 64)")
    savefig(fig, "fig2_one_step_systematic_share", made)

    # fig3: lag profile
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, q in zip(axes, ("G", "eta_x_mode", "w0")):
        for c in CONDS:
            vals = [lag[(f"D64_{q}", cp)][lag.index.get_level_values("condition") == c].median() for cp in (8, 16, 24, 32, 48)]
            ax.plot([56, 48, 40, 32, 16], vals, marker="o", color=COLORS[c], label=LABEL[c])
        ax.set_xlabel("lag to iteration 64 (iterations)"); ax.set_ylabel(f"median rel. distance of {q} to iteration 64")
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Distance to the iteration-64 state vs lag: flat = fluctuation in a region, rising = systematic movement")
    savefig(fig, "fig3_lag_profile", made)

    # fig4: eta diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, q in zip(axes, ("x", "y")):
        names = ["observed diag\n(last draw)", "single draw at\nfixed theta", "posterior mean\nMC error", "posterior mean\nchange 32->64",
                 "Laplace mode\nchange 48->64"]
        for j, c in enumerate(CONDS):
            h = p64[p64.condition == c]
            vals = [h[f"obs_median_eta_{q}_rel_change"].median(), h[f"median_single_draw_eta_{q}"].median(),
                    h[f"eta_{q}_pm_mc_error"].median(), h[f"D_eta_{q}_pm_32_64"].median(), h[f"D64_eta_{q}_mode"].median()]
            ax.bar(np.arange(5) + (j - 1.5) * 0.2, vals, width=0.18, color=COLORS[c], label=LABEL[c])
        ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=7); ax.set_ylabel(f"relative change of eta_{q}")
        ax.set_title(f"eta_{q}")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("What the eta diagnostic measures: one posterior draw vs posterior mean vs Laplace mode (medians)")
    savefig(fig, "fig4_eta_diagnostics", made)

    # fig5: slack noise
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, c in zip(axes, CONDS):
        h = sl[sl.condition == c]
        for rep, hh in h.groupby("replicate"):
            hh = hh.sort_values("checkpoint")
            ax.plot(hh.checkpoint, hh.S_fit_pr156, color=COLORS[c], alpha=0.35, lw=1)
            q = hh[hh.checkpoint.isin([32, 64])]
            ax.errorbar(q.checkpoint + (rep - 4.5) * 0.6, q.S_L80, yerr=2 * q.S_L80_se, fmt="o", ms=3, color="#222", lw=0.8)
        ax.axhline(0, color="#555", lw=0.8); ax.set_title(LABEL[c]); ax.set_xlabel("checkpoint")
    axes[0].set_ylabel("4->5 slack (thin: fit's own C_Q; dots: L=80 fixed theta, ±2 se)")
    fig.suptitle("Fit's own C_Q slack vs fixed-theta L=80 evaluation of the same states")
    savefig(fig, "fig5_slack_fit_vs_L80", made)

    # fig6: paired effects
    e = pd.DataFrame(eff)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, crit in enumerate(("S_fit_pr156", "S_L80", "S_Lap")):
        for j, con in enumerate(("B-A", "C-A", "D-A")):
            h = e[(e.criterion == crit) & (e.contrast == con) & (e.checkpoint == 64)].iloc[0]
            x = i * 4 + j
            ax.errorbar(x, h["median"], yerr=[[h["median"] - h.median_ci_lo], [h.median_ci_hi - h["median"]]], fmt="o",
                        color=COLORS[con[0]], label=con if i == 0 else None)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xticks([1, 5, 9]); ax.set_xticklabels(["fit's own C_Q (PR #156)", "C_Q, L=80 fixed theta", "C_Lap"])
    ax.set_ylabel("paired slack difference at 64 (median, order-stat CI)"); ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Signal-strength effect at checkpoint 64 under three evaluations")
    savefig(fig, "fig6_paired_effects_L80", made)

    # fig7: L pilot
    if p4sum:
        ap = pd.read_csv(AN / "phase4_L_pilot_fits.csv")
        s4 = pd.read_csv(AN / "phase4_L_pilot_slack.csv")
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for c in ns.P4_CONDS:
            h = ap[ap.condition == c].groupby("L_fit")[["plateau_gram_rel_change", "plateau_eta_x_rel_change", "G_noise_rel"]].median()
            axes[0].plot(h.index, h.plateau_gram_rel_change, marker="o", color=COLORS[c], label=f"{LABEL[c]}: gram (observed plateau)")
            axes[0].plot(h.index, h.plateau_eta_x_rel_change, marker="s", ls="--", color=COLORS[c], label=f"{LABEL[c]}: eta_x (last draw)")
            ref = h.plateau_gram_rel_change.loc[5]
            axes[0].plot([5, 20, 50], [ref, ref * math.sqrt(5 / 20), ref * math.sqrt(5 / 50)], color=COLORS[c], lw=0.6, ls=":")
            for rep, hh in s4[s4.condition == c].groupby("replicate"):
                axes[1].plot(hh.L_fit, hh.S_L80_start1, marker="o", color=COLORS[c], alpha=0.7, lw=1)
                axes[2].plot(hh.L_fit, hh.S_Lap_start1, marker="o", color=COLORS[c], alpha=0.7, lw=1)
        for ax in axes:
            ax.set_xscale("log"); ax.set_xticks([5, 20, 50]); ax.set_xticklabels(["5", "20", "50"]); ax.set_xlabel("E-step sample size L")
        axes[0].set_yscale("log"); axes[0].set_ylabel("median per-iteration change, it 25-32"); axes[0].legend(fontsize=6, frameon=False)
        axes[1].set_ylabel("C_Q slack 4->5 at 32 (L=80 eval, start 1)"); axes[2].set_ylabel("C_Lap slack 4->5 at 32 (start 1)")
        axes[1].axhline(0, color="#555", lw=0.8); axes[2].axhline(0, color="#555", lw=0.8)
        fig.suptitle("Phase 4 pilot: changing the E-step sample size L (dotted = 1/sqrt(L) prediction from L=5); lines join replicates")
        savefig(fig, "fig7_L_pilot", made)

    files = sorted(AN.glob("*.csv")) + made
    manifest = {"experiment_id": EID, "prereg_sha256": ns.prereg_hash(), "git_head": ns.git_head(),
                "files": {str(p.relative_to(OUT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (AN / "analysis_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({"phase3_status": p3["status"], "phase4_status": p4["status"], "phase3_fits": len(LG), "phase3_failed": int((LG.status != "OK").sum()),
                      "laplace_not_ok": int((S.lap_status != "OK").sum()), "phase4_fits": len(p4["ledger"]),
                      "max_abs_S_Lap_recompute_diff": float(sl.S_Lap_recompute_diff.abs().max())}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
