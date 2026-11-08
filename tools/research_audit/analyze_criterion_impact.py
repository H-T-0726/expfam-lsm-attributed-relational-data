"""Analysis of the criterion impact validation (all tables and figures from the score CSVs).

Inputs : expfam/results/k_selection/criterion_impact_validation_20261108/score_by_state_{existing,confirm}.csv
         confirm_fits/*.json
Outputs: CSVs and figures/ in the same directory (refuses to overwrite).
Rules  : criterion_impact_protocol.py (selector, MC uncertainty, comparison metrics M1-M8).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/research_audit"))
import criterion_impact_protocol as cp  # noqa: E402

RES = ROOT / cp.RESULTS_REL
FIG = RES / "figures"
CRIT = {"CQ": "A_CQ", "cond_MC": "B_CQ_minus_PZ", "cond_last": "C_cond_last", "cond_mode": "D_cond_mode",
        "CLap": "E_CLap", "CELBO": "F_CELBO"}
TERMS = ["Q_X", "Q_Y", "Q_Z", "P_Z", "P_theta", "Q_MC", "ll_cond_mode", "phi_hat", "prior_hat", "zhat_sq", "logdet_H",
         "lap_volume", "H_q", "two_H_q", "E_q_phi", "R3", "termA", "id1_resid", "id2_resid", "F_CELBO_mcse"]
DKEY = ["phase", "xfam", "signal", "n", "rep"]
TOL = 1e-12


def write(df, name):
    p = RES / name
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    df.to_csv(p, index=False, lineterminator="\n")


def load():
    parts = []
    for nm in ("score_by_state_existing.csv", "score_by_state_confirm.csv"):
        p = RES / nm
        if p.exists():
            parts.append(pd.read_csv(p))
    df = pd.concat(parts, ignore_index=True)
    df["n"] = df["n"].astype(int)
    df["rep"] = df["rep"].astype(int)
    df["k"] = df["k"].astype(int)
    return df


def by_dataset_k(df):
    rows = []
    for key, g in df.groupby(DKEY + ["k"]):
        r = dict(zip(DKEY + ["k"], key))
        for c, col in CRIT.items():
            v = g[col].to_numpy(float)
            ok = len(v) == 2 and np.all(np.isfinite(v))
            r[c] = float(v.mean()) if ok else np.nan
            r[f"{c}_starts_ok"] = int(np.isfinite(v).sum())
        se = g["F_CELBO_mcse"].to_numpy(float)
        r["CELBO_se_mean"] = 0.5 * math.sqrt(float(np.sum(se ** 2))) if len(se) == 2 and np.all(np.isfinite(se)) else np.nan
        for t in TERMS:
            if t in g:
                v = g[t].to_numpy(float)
                r[t] = float(v.mean()) if len(v) == 2 and np.all(np.isfinite(v)) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def select(bdk):
    sel, mar, mcu = [], [], []
    for key, g in bdk.groupby(DKEY):
        g = g.set_index("k").reindex(cp.K_CAND)
        base = dict(zip(DKEY, key))
        for c in CRIT:
            v = g[c].to_numpy(float)
            r = dict(base, criterion=c)
            if np.all(np.isfinite(v)):
                kh = int(cp.K_CAND[int(np.flatnonzero(v <= v.min() + TOL)[0])])
                r.update(K_HAT=kh, status="COMPLETE", outcome="exact" if kh == cp.K_GEN else ("under" if kh < cp.K_GEN else "over"),
                         k_hat_at_max_candidate=kh == max(cp.K_CAND))
                if c == "CELBO":
                    order = np.argsort(v, kind="stable")
                    b, s2 = order[0], order[1]
                    se = g["CELBO_se_mean"].to_numpy(float)
                    se_diff = math.sqrt(se[b] ** 2 + se[s2] ** 2)
                    gap = float(v[s2] - v[b])
                    flag = gap < 2 * se_diff
                    r["MC_UNCERTAIN"] = bool(flag)
                    mcu.append(dict(base, best_K=cp.K_CAND[b], runner_up_K=cp.K_CAND[s2], gap=gap, se_diff=se_diff,
                                    MC_UNCERTAIN=bool(flag)))
            else:
                r.update(K_HAT=np.nan, status="INCOMPLETE", outcome="INCOMPLETE",
                         missing_K=json.dumps([int(k) for k, x in zip(cp.K_CAND, v) if not np.isfinite(x)]))
            sel.append(r)
            m = dict(base, criterion=c)
            for k in (3, 4, 5, 6):
                a, b2 = g.loc[k, c], g.loc[k + 1, c]
                m[f"slack_{k}{k + 1}"] = float(a - b2) if np.isfinite(a) and np.isfinite(b2) else np.nan
            if c == "CELBO":
                se = g["CELBO_se_mean"]
                for k in (4, 5):
                    m[f"slack_{k}{k + 1}_se"] = float(math.hypot(se.loc[k], se.loc[k + 1]))
            mar.append(m)
    return pd.DataFrame(sel), pd.DataFrame(mar), pd.DataFrame(mcu)


def decomposition(bdk):
    rows = []
    for key, g in bdk.groupby(DKEY):
        g = g.set_index("k")
        for k in (4, 5):
            a, b = g.loc[k], g.loc[k + 1]
            r = dict(zip(DKEY, key), pair=f"{k}->{k + 1}")
            sl = {c: a[c] - b[c] for c in ("CQ", "CLap", "CELBO", "cond_MC", "cond_last", "cond_mode")}
            r.update({f"slack_{c}": v for c, v in sl.items()})
            d = {t: b[t] - a[t] for t in ("two_H_q", "termA", "R3", "Q_X", "Q_Y", "P_Z", "P_theta", "logdet_H", "phi_hat",
                                          "ll_cond_mode", "zhat_sq", "E_q_phi")}
            r.update(d_2H=d["two_H_q"], d_termA=d["termA"], d_2R3=2 * d["R3"],
                     d_fit_X=-2 * d["Q_X"], d_fit_Y=-2 * d["Q_Y"], d_PZ=d["P_Z"], d_Ptheta=d["P_theta"],
                     d_logdetH=d["logdet_H"], d_minus2_phi_hat=-2 * d["phi_hat"], d_minus2_ll_mode=-2 * d["ll_cond_mode"],
                     d_zhat_sq=d["zhat_sq"])
            dq = sl["CQ"] - sl["CLap"]
            r["slackQ_minus_slackLap"] = dq
            r["identity1_resid"] = dq + (r["d_2H"] + r["d_termA"] + r["d_2R3"])
            r["identity2_resid"] = (sl["CELBO"] - sl["CLap"]) + r["d_2R3"]
            r["sign_match_minus_d2H"] = bool(np.sign(dq) == np.sign(-r["d_2H"])) if np.isfinite(dq) and np.isfinite(r["d_2H"]) else np.nan
            r["ratio_termA_R3_over_2H"] = abs(r["d_termA"] + r["d_2R3"]) / abs(r["d_2H"]) if np.isfinite(dq) and r["d_2H"] != 0 else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


PAIRS = [("CQ", "CLap"), ("CLap", "CELBO"), ("CQ", "CELBO"), ("CQ", "cond_MC"), ("CQ", "cond_last"), ("CQ", "cond_mode")]


def pairwise(sel):
    w = sel.pivot_table(index=DKEY, columns="criterion", values="K_HAT").reset_index()
    rows = []
    for grp, g in [(("all",), w)] + [((p,), gg) for p, gg in w.groupby("phase")] + \
            [((p, x, n, s), gg) for (p, x, n, s), gg in w.groupby(["phase", "xfam", "n", "signal"])]:
        for a, b in PAIRS:
            if a not in g or b not in g:
                continue
            m = g[[a, b]].dropna()
            rows.append({"group": "|".join(map(str, grp)), "first": a, "second": b, "n_both_complete": len(m),
                         "agree": int((m[a] == m[b]).sum()), "first_smaller": int((m[a] < m[b]).sum()),
                         "first_larger": int((m[a] > m[b]).sum())})
    return pd.DataFrame(rows)


def cell_summary(sel, mar):
    rows = []
    mm = mar.set_index(DKEY + ["criterion"])
    for (p, x, n, s, c), g in sel.groupby(["phase", "xfam", "n", "signal", "criterion"]):
        sl = mm.loc[[tuple(r[k] for k in DKEY) + (c,) for _, r in g.iterrows()]]
        rows.append({"phase": p, "xfam": x, "n": n, "signal": s, "criterion": c, "datasets": len(g),
                     "complete": int((g.status == "COMPLETE").sum()), "exact": int((g.outcome == "exact").sum()),
                     "under": int((g.outcome == "under").sum()), "over": int((g.outcome == "over").sum()),
                     "incomplete": int((g.outcome == "INCOMPLETE").sum()),
                     "k_hat_values": " ".join("NA" if not np.isfinite(v) else str(int(v)) for v in g.K_HAT),
                     "median_slack_45": float(np.nanmedian(sl["slack_45"])) if sl["slack_45"].notna().any() else np.nan,
                     "median_slack_56": float(np.nanmedian(sl["slack_56"])) if sl["slack_56"].notna().any() else np.nan})
    return pd.DataFrame(rows)


def metrics(sel, dec):
    w = sel.pivot_table(index=DKEY, columns="criterion", values="K_HAT").reset_index()
    out = []
    for p in sorted(w.phase.unique()):
        for xf in ("all", "gaussian", "poisson"):
            g = w[w.phase == p] if xf == "all" else w[(w.phase == p) & (w.xfam == xf)]
            d = dec[dec.phase == p] if xf == "all" else dec[(dec.phase == p) & (dec.xfam == xf)]
            s = sel[sel.phase == p] if xf == "all" else sel[(sel.phase == p) & (sel.xfam == xf)]
            m = {}
            ql = g[["CQ", "CLap"]].dropna()
            m["M1_KQ_le_KLap"] = f"{int((ql.CQ <= ql.CLap).sum())}/{len(ql)}"
            le = g[["CLap", "CELBO"]].dropna()
            m["M2_KLap_eq_KELBO"] = f"{int((le.CLap == le.CELBO).sum())}/{len(le)}"
            for c in ("cond_MC", "cond_last", "cond_mode"):
                sc = s[s.criterion == c]
                m[f"M3_over_{c}"] = f"{int((sc.outcome == 'over').sum())}/{int((sc.status == 'COMPLETE').sum())}"
            d45 = d[d.pair == "4->5"]
            m["M4_d2H_45_positive"] = f"{int((d45.d_2H > 0).sum())}/{int(d45.d_2H.notna().sum())}"
            sm = d["sign_match_minus_d2H"].dropna()
            m["M5_sign_match"] = f"{int(sm.astype(bool).sum())}/{len(sm)}"
            m["M6_Lap_ELBO_differ"] = f"{int((le.CLap != le.CELBO).sum())}/{len(le)}"
            for c in CRIT:
                sc = s[s.criterion == c]
                comp = int((sc.status == "COMPLETE").sum())
                m[f"M7_exact_{c}"] = f"{int((sc.outcome == 'exact').sum())}/{comp}"
                m[f"M8_under_over_{c}"] = f"{int((sc.outcome == 'under').sum())}/{int((sc.outcome == 'over').sum())} of {comp}"
            for k, v in m.items():
                out.append({"phase": "EXPLORATORY_existing" if p == 1 else "CONFIRMATORY_new", "x_family": xf,
                            "metric": k, "value": v})
    t = pd.DataFrame(out)
    return t.pivot_table(index=["x_family", "metric"], columns="phase", values="value", aggfunc="first").reset_index()


# ---------------------------------------------------------------- figures
COL = {"under": "#2a78d6", "exact": "#1baf7a", "over": "#eb6834", "INCOMPLETE": "#b9b8b2"}
CRIT_ORDER = ["cond_MC", "cond_last", "cond_mode", "CQ", "CLap", "CELBO"]
CCOL = {"CQ": "#2a78d6", "CLap": "#eb6834", "CELBO": "#1baf7a"}


def figures(sel, bdk, dec, cell):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    FIG.mkdir(parents=True, exist_ok=True)
    phases = sorted(sel.phase.unique())
    pname = {1: "Phase C existing (EXPLORATORY)", 2: "Phase E new (CONFIRMATORY)"}

    def outcome_bars(ax, sub, title):
        x = np.arange(len(CRIT_ORDER))
        bottom = np.zeros(len(x))
        for o in ("under", "exact", "over", "INCOMPLETE"):
            h = np.array([int(((sub.criterion == c) & (sub.outcome == o)).sum()) for c in CRIT_ORDER])
            ax.bar(x, h, bottom=bottom, color=COL[o], width=0.7, edgecolor="white", linewidth=1, label=o)
            bottom += h
        ax.set_xticks(x, CRIT_ORDER, rotation=35, ha="right")
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("datasets")

    # fig1 overall by criterion
    fig, axs = plt.subplots(1, len(phases), figsize=(4 * len(phases), 3), squeeze=False)
    for ax, p in zip(axs[0], phases):
        outcome_bars(ax, sel[sel.phase == p], pname[p] + "\nK_HAT vs K_GEN=5")
    axs[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig1_selection_by_criterion.png", dpi=150); plt.close(fig)

    # fig2 score curves (median over datasets of C(K) - C(5)) per cell
    cells = sorted(bdk.groupby(["xfam", "n", "signal"]).groups)
    for p in phases:
        fig, axs = plt.subplots(2, 4, figsize=(12, 5.5), sharex=True)
        for ax, (x, n, s) in zip(axs.ravel(), cells):
            g = bdk[(bdk.phase == p) & (bdk.xfam == x) & (bdk.n == n) & (bdk.signal == s)]
            for c in ("CQ", "CLap", "CELBO"):
                rel = []
                for _, gg in g.groupby("rep"):
                    gg = gg.set_index("k")[c]
                    rel.append((gg - gg.loc[5]).reindex(cp.K_CAND).to_numpy())
                med = np.nanmedian(np.array(rel), axis=0)
                ax.plot(cp.K_CAND, med, color=CCOL[c], lw=2, marker="o", ms=4, label=c)
            ax.axhline(0, color="#b9b8b2", lw=0.8)
            ax.set_title(f"{x} n={n} {s}", fontsize=8)
        axs[1][0].set_xlabel("K"); axs[0][0].set_ylabel("median C(K) - C(5)")
        axs[0][0].legend(frameon=False, fontsize=7)
        fig.suptitle(pname[p] + ": score curves (lower = preferred)", fontsize=9)
        fig.tight_layout(); fig.savefig(FIG / f"fig2_score_curves_phase{p}.png", dpi=150); plt.close(fig)

    # fig3 / fig4 weak vs strong, n75 vs n150
    for fname, fac, lv in (("fig3_weak_vs_strong.png", "signal", ["weak", "strong"]),
                           ("fig4_n75_vs_n150.png", "n", [75, 150])):
        fig, axs = plt.subplots(len(phases), 2, figsize=(8, 3 * len(phases)), squeeze=False)
        for i, p in enumerate(phases):
            for j, v in enumerate(lv):
                outcome_bars(axs[i][j], sel[(sel.phase == p) & (sel[fac] == v)], f"{pname[p]}\n{fac}={v}")
        axs[0][0].legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / fname, dpi=150); plt.close(fig)

    # fig5 / fig6 selection agreement
    w = sel.pivot_table(index=DKEY, columns="criterion", values="K_HAT").reset_index()
    for fname, a, b in (("fig5_CQ_vs_CLap.png", "CQ", "CLap"), ("fig6_CLap_vs_CELBO.png", "CLap", "CELBO")):
        fig, axs = plt.subplots(1, len(phases), figsize=(3.6 * len(phases), 3.2), squeeze=False)
        for ax, p in zip(axs[0], phases):
            m = w[w.phase == p][[a, b]].dropna()
            tab = m.groupby([a, b]).size()
            for (xa, yb), cnt in tab.items():
                ax.scatter(xa, yb, s=40 + 40 * cnt, color="#2a78d6", edgecolor="white", linewidth=2, zorder=3)
                ax.annotate(str(cnt), (xa, yb), ha="center", va="center", fontsize=7, color="white", zorder=4)
            ax.plot([1, 7], [1, 7], color="#b9b8b2", lw=0.8)
            ax.set_xlim(0.5, 7.5); ax.set_ylim(0.5, 7.5)
            ax.set_xlabel(f"K_HAT {a}"); ax.set_ylabel(f"K_HAT {b}")
            ax.set_title(pname[p] + f"\n{len(m)} complete datasets", fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / fname, dpi=150); plt.close(fig)

    # fig7 decomposition of slack_Q - slack_Lap at 4->5 and 5->6 (medians per cell)
    for p in phases:
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
        for ax, pr in zip(axs, ("4->5", "5->6")):
            d = dec[(dec.phase == p) & (dec.pair == pr)]
            labs, comps = [], {"-d_2H": [], "-d_termA": [], "-d_2R3": []}
            for (x, n, s), g in d.groupby(["xfam", "n", "signal"]):
                labs.append(f"{x[:4]} {n} {s[:2]}")
                comps["-d_2H"].append(np.nanmedian(-g.d_2H)); comps["-d_termA"].append(np.nanmedian(-g.d_termA))
                comps["-d_2R3"].append(np.nanmedian(-g.d_2R3))
            x = np.arange(len(labs)); wdt = 0.26
            for i, (kk, col) in enumerate(zip(comps, ("#2a78d6", "#eb6834", "#1baf7a"))):
                ax.bar(x + (i - 1) * wdt, comps[kk], width=wdt, color=col, edgecolor="white", linewidth=1, label=kk)
            ax.axhline(0, color="#8a8984", lw=0.8)
            ax.set_xticks(x, labs, rotation=35, ha="right")
            ax.set_title(f"{pname[p]}: slack_Q - slack_Lap components, {pr} (medians)", fontsize=8)
        axs[0].legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / f"fig7_decomposition_phase{p}.png", dpi=150); plt.close(fig)

    # fig8 condition map
    fig, axs = plt.subplots(1, len(phases), figsize=(5.2 * len(phases), 3.6), squeeze=False)
    for ax, p in zip(axs[0], phases):
        c = cell[cell.phase == p]
        rows = sorted(c.groupby(["xfam", "n", "signal"]).groups)
        for i, rk in enumerate(rows):
            for j, cr in enumerate(CRIT_ORDER):
                r = c[(c.xfam == rk[0]) & (c.n == rk[1]) & (c.signal == rk[2]) & (c.criterion == cr)].iloc[0]
                frac = [r.under, r.exact, r.over, r.incomplete]
                dom = ["under", "exact", "over", "INCOMPLETE"][int(np.argmax(frac))]
                ax.add_patch(plt.Rectangle((j, i), 0.95, 0.95, color=COL[dom], alpha=0.35 + 0.15 * max(frac)))
                ax.text(j + 0.47, i + 0.47, f"{r.under}/{r.exact}/{r.over}" + (f"/I{r.incomplete}" if r.incomplete else ""),
                        ha="center", va="center", fontsize=6.5, color="#0b0b0b")
        ax.set_xlim(0, len(CRIT_ORDER)); ax.set_ylim(0, len(rows))
        ax.set_xticks(np.arange(len(CRIT_ORDER)) + 0.47, CRIT_ORDER, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(rows)) + 0.47, [f"{a} n={b} {s}" for a, b, s in rows])
        ax.set_title(pname[p] + "\nunder/exact/over per cell (colour = most frequent)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig8_condition_map.png", dpi=150); plt.close(fig)


def fit_results():
    rows = []
    for j in sorted((RES / "confirm_fits").glob("*.json")):
        r = json.loads(j.read_text(encoding="utf-8"))
        rows.append({k: r.get(k) for k in ("key", "xfam", "signal", "n", "rep", "k", "s", "seed", "data_seed", "status",
                                           "exception", "runtime_s", "npz_sha256", "git_head", "protocol_hash",
                                           "x_sha256", "y_sha256", "train_sha256")})
    return pd.DataFrame(rows)


def main():
    df = load()
    bdk = by_dataset_k(df)
    sel, mar, mcu = select(bdk)
    dec = decomposition(bdk)
    pw = pairwise(sel)
    cell = cell_summary(sel, mar)
    inc = []
    for _, r in df.iterrows():
        bad = []
        if str(r.get("fit_status", "OK")) not in ("OK", "OK (reused, gate PASS)", "nan"):
            bad.append(f"fit {r.get('fit_status')}")
        if str(r.get("lap_status")) != "OK":
            bad.append(f"laplace {r.get('lap_status')}")
        elif str(r.get("elbo_status")) != "OK":
            bad.append(f"elbo {r.get('elbo_status')}")
        if bad:
            inc.append({"level": "state", **{k: r[k] for k in ("phase", "key", "xfam", "signal", "n", "rep", "k", "s")},
                        "reason": "; ".join(bad), "affected_criteria": "CLap, CELBO, cond_mode" if "laplace" in bad[0] else "all"})
    for _, r in sel[sel.status == "INCOMPLETE"].iterrows():
        inc.append({"level": "dataset", **{k: r[k] for k in DKEY}, "criterion": r.criterion, "reason": f"missing K {r.missing_K}"})
    mcunc = df.groupby(["phase", "xfam", "n", "signal"]).agg(states=("F_CELBO_mcse", "size"),
                                                              median_CELBO_mcse=("F_CELBO_mcse", "median"),
                                                              max_CELBO_mcse=("F_CELBO_mcse", "max"),
                                                              median_batch_sd=("elbo_batch_sd", "median"),
                                                              median_elbo_seconds=("elbo_seconds", "median")).reset_index()
    mcu_all = mcu.merge(mcunc, on=["phase", "xfam", "n", "signal"], how="left")
    write(df, "score_by_state.csv")
    write(bdk, "score_by_dataset_k.csv")
    write(sel, "selection_by_dataset.csv")
    write(mar, "selection_margins.csv")
    write(pw, "criterion_pairwise_comparison.csv")
    write(dec, "entropy_and_r3_decomposition.csv")
    write(mcu_all, "mc_uncertainty.csv")
    write(cell, "family_condition_summary.csv")
    write(pd.DataFrame(inc), "incomplete_and_invalid_runs.csv")
    fr = fit_results()
    if len(fr):
        write(fr, "fit_results.csv")
    write(metrics(sel, dec), "confirmation_vs_exploration.csv")
    proto = cp.protocol()
    write(pd.DataFrame([{"section": k, "content": json.dumps(v, ensure_ascii=False)} for k, v in proto.items()]
                       + [{"section": "protocol_hash", "content": cp.protocol_hash()}]), "experiment_protocol.csv")
    figures(sel, bdk, dec, cell)
    print("identity residual max |id1|", float(np.nanmax(np.abs(df.id1_resid))), "|id2|", float(np.nanmax(np.abs(df.id2_resid))))


if __name__ == "__main__":
    main()
