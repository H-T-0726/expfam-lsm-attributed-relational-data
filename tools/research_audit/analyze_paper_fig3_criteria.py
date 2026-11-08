"""Analysis for paper_fig3_condition_criteria_20261109 (all tables/figures from the score CSVs and fit JSONs).

Rules: paper_fig3_protocol.SELECTION.  Refuses to overwrite.
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
import paper_fig3_protocol as pp  # noqa: E402

RES = ROOT / pp.RESULTS_REL
FIG = RES / "figures"
COL = {"CQ": "A_CQ", "cond_MC": "B_CQ_minus_PZ", "cond_last": "C_cond_last", "cond_mode": "D_cond_mode",
       "CLap": "E_CLap", "CELBO": "F_CELBO"}
TOL = 1e-12
PAPER_FIG3 = ROOT / "reports/k_selection_theory/historical_code_lineage_audit_20261106/paper_fig3_implied_lnL.csv"


def write(df, name):
    p = RES / name
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    df.to_csv(p, index=False, lineterminator="\n")


def argmin_k(v):
    v = np.asarray(v, float)
    return int(pp.K_CAND[int(np.flatnonzero(v <= v.min() + TOL)[0])])


def outcome(kh, kg):
    return "exact" if kh == kg else ("under" if kh < kg else "over")


def main():
    df = pd.concat([pd.read_csv(RES / f"score_by_state_K{k}.csv") for k in pp.K_GEN], ignore_index=True)
    df["k"] = df["k"].astype(int)
    # -------- per-trial selection
    sel, mcu = [], []
    for (kg, t), g in df.groupby(["kgen", "trial"]):
        g = g.set_index("k").reindex(pp.K_CAND)
        for c, col in COL.items():
            v = g[col].to_numpy(float) if col in g else np.full(len(pp.K_CAND), np.nan)
            r = {"K_GEN": kg, "trial": t, "criterion": c}
            if np.all(np.isfinite(v)):
                kh = argmin_k(v)
                r.update(K_HAT=kh, outcome=outcome(kh, kg), k_hat_at_max_candidate=kh == max(pp.K_CAND))
                if c == "CELBO":
                    o = np.argsort(v, kind="stable")
                    se = g["F_CELBO_mcse"].to_numpy(float)
                    sd = math.hypot(se[o[0]], se[o[1]])
                    gap = float(v[o[1]] - v[o[0]])
                    r["MC_UNCERTAIN"] = gap < 2 * sd
                    mcu.append({"K_GEN": kg, "trial": t, "best_K": pp.K_CAND[o[0]], "runner_up_K": pp.K_CAND[o[1]],
                                "gap": gap, "se_diff": sd, "MC_UNCERTAIN": gap < 2 * sd})
            else:
                r.update(K_HAT=np.nan, outcome="INCOMPLETE",
                         missing_K=json.dumps([k for k, x in zip(pp.K_CAND, v) if not np.isfinite(x)]))
            sel.append(r)
    sel = pd.DataFrame(sel)
    # -------- summary per K_GEN x criterion
    summ = []
    for (kg, c), g in sel.groupby(["K_GEN", "criterion"]):
        ks = g.sort_values("trial").K_HAT
        dist = pd.Series([int(x) for x in ks.dropna()]).value_counts().sort_index()
        summ.append({"K_GEN": kg, "criterion": c, "exact": int((g.outcome == "exact").sum()),
                     "under": int((g.outcome == "under").sum()), "over": int((g.outcome == "over").sum()),
                     "incomplete": int((g.outcome == "INCOMPLETE").sum()),
                     "k_hat_by_trial": " ".join("NA" if not np.isfinite(x) else str(int(x)) for x in ks),
                     "k_hat_distribution": "; ".join(f"K={k}:{n}" for k, n in dist.items()),
                     "mc_uncertain": int(g.get("MC_UNCERTAIN", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())})
    summ = pd.DataFrame(summ)
    # -------- aggregated selections
    agg = []
    for kg, g in df.groupby("kgen"):
        for c, col in COL.items():
            p = g.pivot_table(index="trial", columns="k", values=col, aggfunc="first").reindex(index=pp.TRIALS, columns=pp.K_CAND)
            full = p.notna().all().all()
            mean = p.mean(axis=0).to_numpy() if full else None
            mn = p.min(axis=0, skipna=True).to_numpy()
            agg.append({"K_GEN": kg, "criterion": c,
                        "K_HAT_mean_over_trials": argmin_k(mean) if full else "INCOMPLETE",
                        "K_HAT_min_over_trials": argmin_k(mn) if np.all(np.isfinite(mn)) else "INCOMPLETE",
                        "n_finite_states": int(p.notna().sum().sum())})
    agg = pd.DataFrame(agg)
    # -------- pairwise per trial
    w = sel.pivot_table(index=["K_GEN", "trial"], columns="criterion", values="K_HAT").reset_index()
    pw = []
    for kg in ["all"] + pp.K_GEN:
        g = w if kg == "all" else w[w.K_GEN == kg]
        for a, b in (("CQ", "CLap"), ("CLap", "CELBO"), ("CQ", "CELBO"), ("CQ", "cond_MC"), ("CQ", "cond_last"), ("CQ", "cond_mode")):
            m = g[[a, b]].dropna()
            pw.append({"K_GEN": kg, "first": a, "second": b, "n_both_complete": len(m), "agree": int((m[a] == m[b]).sum()),
                       "first_smaller": int((m[a] < m[b]).sum()), "first_larger": int((m[a] > m[b]).sum())})
    pw = pd.DataFrame(pw)
    # -------- decomposition at K_GEN-1 -> K_GEN and K_GEN -> K_GEN+1
    dec = []
    for (kg, t), g in df.groupby(["kgen", "trial"]):
        g = g.set_index("k")
        for k0 in (kg - 1, kg):
            if k0 < 1 or k0 + 1 > max(pp.K_CAND) or k0 not in g.index or k0 + 1 not in g.index:
                continue
            a, b = g.loc[k0], g.loc[k0 + 1]
            r = {"K_GEN": kg, "trial": t, "pair": f"{k0}->{k0 + 1}", "side": "under-side" if k0 < kg else "over-side"}
            for c, col in COL.items():
                r[f"slack_{c}"] = a[col] - b[col]
            for tname in ("two_H_q", "termA", "R3", "P_Z", "P_theta", "logdet_H", "Q_X", "Q_Y", "phi_hat", "ll_cond_mode"):
                r[f"d_{tname}"] = b.get(tname, np.nan) - a.get(tname, np.nan)
            r["d_2H"], r["d_2R3"] = r.pop("d_two_H_q"), 2 * r.pop("d_R3")
            dq = r["slack_CQ"] - r["slack_CLap"]
            r["slackQ_minus_slackLap"] = dq
            r["identity1_resid"] = dq + (r["d_2H"] + r["d_termA"] + r["d_2R3"])
            r["identity2_resid"] = (r["slack_CELBO"] - r["slack_CLap"]) + r["d_2R3"]
            r["sign_match_minus_d2H"] = bool(np.sign(dq) == np.sign(-r["d_2H"])) if np.isfinite(dq) else np.nan
            dec.append(r)
    dec = pd.DataFrame(dec)
    # -------- fits, convergence, laplace status, incomplete
    fits, conv = [], []
    for j in sorted((RES / "fits").glob("*.json")):
        r = json.loads(j.read_text(encoding="utf-8"))
        fits.append({k: r.get(k) for k in ("key", "kgen", "k", "trial", "seed", "data_seed", "status", "exception",
                                           "runtime_s", "npz_sha256", "git_head", "protocol_hash", "x_sha256", "y_sha256")})
        tr = r.get("q_strict_trajectory") or []
        if len(tr) >= 2:
            q = [x["q_strict"] for x in tr]
            conv.append({"kgen": r["kgen"], "k": r["k"], "trial": r["trial"], "q_last": q[-1],
                         "abs_change_last": abs(q[-1] - q[-2]), "rel_change_last": abs(q[-1] - q[-2]) / abs(q[-1]),
                         "abs_change_last3_max": max(abs(q[i] - q[i - 1]) for i in range(len(q) - 3, len(q))),
                         "q_increase_last3": sum(q[i] > q[i - 1] for i in range(len(q) - 3, len(q)))})
    fits, conv = pd.DataFrame(fits), pd.DataFrame(conv)
    convs = conv.groupby(["kgen", "k"]).agg(median_abs_change_last=("abs_change_last", "median"),
                                            max_abs_change_last=("abs_change_last", "max"),
                                            median_rel_change_last=("rel_change_last", "median")).reset_index()
    lap = df.groupby(["kgen", "k"]).agg(states=("key", "size"),
                                        laplace_ok=("lap_status", lambda s: int((s == "OK").sum())),
                                        elbo_ok=("elbo_status", lambda s: int((s == "OK").sum())),
                                        median_lap_grad_inf=("lap_grad_inf", "median"),
                                        median_CELBO_mcse=("F_CELBO_mcse", "median")).reset_index()
    inc = df[(df.fit_status != "OK") | (df.lap_status != "OK") | (df.elbo_status != "OK")][
        ["key", "kgen", "k", "trial", "fit_status", "lap_status", "elbo_status", "lap_grad_inf", "lap_min_eig_H"]]
    # -------- paper Fig.3 comparison (relative curves)
    pf = pd.read_csv(PAPER_FIG3)
    cmp_rows = []
    for kg in pp.K_GEN:
        g = df[df.kgen == kg]
        for k in pp.K_CAND:
            r = {"K_GEN": kg, "k": k}
            pr = pf[(pf.k_star == kg) & (pf.k == k)]
            r["paper_bic_read"] = float(pr.bic_read.iloc[0]) if len(pr) else np.nan
            for c in ("CQ", "CLap", "CELBO", "cond_MC"):
                v = g[g.k == k][COL[c]]
                r[f"{c}_median_over_trials"] = float(v.median()) if v.notna().any() else np.nan
            cmp_rows.append(r)
    cmp_ = pd.DataFrame(cmp_rows)
    for c in ("paper_bic_read", "CQ_median_over_trials", "CLap_median_over_trials", "CELBO_median_over_trials", "cond_MC_median_over_trials"):
        cmp_[c + "_minus_at_KGEN"] = cmp_.groupby("K_GEN")[c].transform(lambda s: s - s.iloc[cmp_.loc[s.index, "k"].tolist().index(cmp_.loc[s.index, "K_GEN"].iloc[0])])
    write(df, "score_by_state.csv")
    write(sel, "selection_per_trial.csv")
    write(summ, "selection_summary.csv")
    write(agg, "selection_aggregated.csv")
    write(pw, "pairwise_agreement.csv")
    write(dec, "decomposition.csv")
    write(pd.DataFrame(mcu), "mc_uncertainty.csv")
    write(fits, "fit_results.csv")
    write(conv, "convergence_by_fit.csv")
    write(convs, "convergence_diagnostic.csv")
    write(lap, "laplace_status_by_k.csv")
    write(inc, "incomplete_and_invalid_runs.csv")
    write(cmp_, "paper_fig3_comparison.csv")
    write(pd.DataFrame(pp.CONDITIONS, columns=["item", "paper", "this_study", "class"]), "condition_fidelity.csv")
    write(pd.DataFrame([{"section": k, "content": json.dumps(v, ensure_ascii=False)} for k, v in pp.protocol().items()]
                       + [{"section": "protocol_hash", "content": pp.protocol_hash()}]), "experiment_protocol.csv")
    figures(df, sel, dec, cmp_)
    print("identity max |id1|", float(np.nanmax(np.abs(df.id1_resid))), "|id2|", float(np.nanmax(np.abs(df.id2_resid))))


def figures(df, sel, dec, cmp_):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    FIG.mkdir(parents=True, exist_ok=True)
    crits = ["cond_MC", "cond_last", "cond_mode", "CQ", "CLap", "CELBO"]
    # fig1: K_HAT distribution heatmaps
    fig, axs = plt.subplots(1, len(crits), figsize=(16, 3.4), sharey=True)
    for ax, c in zip(axs, crits):
        M = np.zeros((len(pp.K_GEN), len(pp.K_CAND) + 1))
        for i, kg in enumerate(pp.K_GEN):
            g = sel[(sel.criterion == c) & (sel.K_GEN == kg)]
            for kh in g.K_HAT:
                M[i, -1 if not np.isfinite(kh) else int(kh) - 1] += 1
        ax.imshow(M, cmap="Blues", vmin=0, vmax=10, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j]:
                    ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] > 5 else "#0b0b0b")
            ax.add_patch(plt.Rectangle((pp.K_GEN[i] - 1.5, i - 0.5), 1, 1, fill=False, edgecolor="#eb6834", lw=1.5))
        ax.set_xticks(range(11), [str(k) for k in pp.K_CAND] + ["I"])
        ax.set_yticks(range(len(pp.K_GEN)), [f"K_GEN={k}" for k in pp.K_GEN])
        ax.set_title(c, fontsize=8)
        ax.set_xlabel("selected K (I = INCOMPLETE)")
    fig.suptitle("Per-trial K_HAT counts (10 trials per K_GEN; orange box = K_GEN)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig1_khat_distribution.png", dpi=150); plt.close(fig)
    # fig2: score curves relative to K_GEN, median over trials with min-max band
    cc = {"CQ": "#2a78d6", "CLap": "#eb6834", "CELBO": "#1baf7a", "cond_MC": "#6250d6"}
    fig, axs = plt.subplots(1, len(pp.K_GEN), figsize=(17, 3.4))
    for ax, kg in zip(axs, pp.K_GEN):
        g = df[df.kgen == kg]
        for c in cc:
            p = g.pivot_table(index="trial", columns="k", values=COL[c], aggfunc="first").reindex(columns=pp.K_CAND)
            rel = p.sub(p[kg], axis=0)
            ax.plot(pp.K_CAND, rel.median(), color=cc[c], lw=2, marker="o", ms=3.5, label=c)
            ax.fill_between(pp.K_CAND, rel.min(), rel.max(), color=cc[c], alpha=0.12, lw=0)
        ax.axhline(0, color="#b9b8b2", lw=0.8)
        ax.axvline(kg, color="#b9b8b2", lw=0.8, ls=":")
        ax.set_yscale("symlog", linthresh=50)
        ax.set_title(f"K_GEN={kg}", fontsize=8)
        ax.set_xlabel("K")
    axs[0].set_ylabel("C(K) - C(K_GEN) (median over trials, band = min-max; symlog)")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig2_score_curves.png", dpi=150); plt.close(fig)
    # fig3: decomposition of slack_Q - slack_Lap (medians over trials)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, side in zip(axs, ("under-side", "over-side")):
        d = dec[dec.side == side]
        labs = [f"K_GEN={k}" for k in sorted(d.K_GEN.unique())]
        x = np.arange(len(labs))
        for i, (kk, col) in enumerate((("d_2H", "#2a78d6"), ("d_termA", "#eb6834"), ("d_2R3", "#1baf7a"))):
            vals = [np.nanmedian(-d[d.K_GEN == k][kk]) for k in sorted(d.K_GEN.unique())]
            ax.bar(x + (i - 1) * 0.26, vals, width=0.26, color=col, edgecolor="white", linewidth=1, label="-" + kk)
        ax.axhline(0, color="#8a8984", lw=0.8)
        ax.set_xticks(x, labs)
        ax.set_title(f"slack_Q - slack_Lap components, {side} pair (medians)", fontsize=8)
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig3_decomposition.png", dpi=150); plt.close(fig)
    # fig4: paper Fig.3 (digitised) vs this study, relative to K_GEN
    fig, axs = plt.subplots(1, len(pp.K_GEN), figsize=(17, 3.2))
    for ax, kg in zip(axs, pp.K_GEN):
        g = cmp_[cmp_.K_GEN == kg]
        ax.plot(g.k, g["paper_bic_read_minus_at_KGEN"], color="#0b0b0b", lw=2, marker="s", ms=3.5, label="paper Fig.3 (digitised)")
        for c in ("CQ", "CLap", "cond_MC"):
            ax.plot(g.k, g[f"{c}_median_over_trials_minus_at_KGEN"], color=cc[c], lw=1.5, marker="o", ms=3, label=f"{c} (median)")
        ax.axhline(0, color="#b9b8b2", lw=0.8)
        ax.set_yscale("symlog", linthresh=50)
        ax.set_title(f"K_GEN={kg}", fontsize=8)
        ax.set_xlabel("K")
    axs[0].legend(frameon=False, fontsize=6.5)
    axs[0].set_ylabel("value(K) - value(K_GEN), symlog")
    fig.tight_layout(); fig.savefig(FIG / "fig4_paper_fig3_vs_this_study.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
