"""Analysis of the signal-strength experiment (no MCEM fit).

Reads   expfam/results/k_selection/historical_canonical_causal_20261106/signal_strength/{gates,full_n75,full_n150}
        expfam/results/k_selection/historical_canonical_causal_20261106/existing_reanalysis
Writes  expfam/results/k_selection/historical_canonical_causal_20261106/analysis/   (CSV + JSON)
        expfam/results/k_selection/historical_canonical_causal_20261106/figures/    (PDF + PNG)
Every number is a deterministic function of the stored CSVs; nothing is refit.
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

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "expfam" / "results" / "k_selection" / "historical_canonical_causal_20261106"
SS = BASE / "signal_strength"
EX = BASE / "existing_reanalysis"
OUT = BASE / "analysis"
FIG = BASE / "figures"
STARTS = (1, 2)
KS = tuple(range(1, 8))
K_GEN = 5
C2 = 1.0 + math.log(2.0 * math.pi)
COND_COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
COND_MARK = {"A": "o", "B": "s", "C": "^", "D": "D"}
COND_LABEL = {"A": "A baseline", "B": "B F-strong", "C": "C w-strong", "D": "D F+w-strong"}
INK, INK2 = "#0b0b0b", "#52514e"


def need(c, m):
    if not c:
        raise RuntimeError(m)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(df: pd.DataFrame, name: str) -> None:
    p = OUT / name
    need(not p.exists(), f"refuse to overwrite {p}")
    df.to_csv(p, index=False, float_format="%.12g")


# ----------------------------------------------------------------- load
def load():
    fits, sel, iters, ledg, prov, runs = [], [], [], [], [], {}
    for n in (75, 150):
        d = SS / f"full_n{n}"
        if not (d / "runinfo.json").exists():
            continue
        runs[n] = json.loads((d / "runinfo.json").read_text(encoding="utf-8"))
        fits.append(pd.read_csv(d / "fit_criteria.csv"))
        sel.append(pd.read_csv(d / "selection.csv"))
        iters.append(pd.read_csv(d / "iteration_diagnostics.csv"))
        ledg.append(pd.read_csv(d / "execution_ledger.csv"))
        prov.append(pd.read_csv(d / "dataset_provenance.csv"))
    need(runs, "no completed production run")
    f = pd.concat(fits, ignore_index=True)
    # A states carry no runner diagnostics; take them from the stored d=15 re-evaluation rows
    # of the same states (d-sweep for n=75, R2 for n=150), matched on the EMC state key.
    ks = ROOT / "expfam" / "results" / "k_selection"
    srcs = [pd.read_csv(ks / "dsweep_k_selection_20261105" / "full" / "fit_criteria.csv").query("d == 15"),
            pd.read_csv(ks / "k_selection_robustness_20261105" / "r2_n150" / "fit_criteria.csv").query("d == 15 and L == 5 and checkpoint == 8")]
    diag = pd.concat(srcs, ignore_index=True)[["source", "gram_rel_change", "eta_x_rel_change", "eta_y_rel_change",
                                              "max_gram_rel_change_last4"]].drop_duplicates("source").set_index("source")
    isA = f.source.str.startswith("emc_state:")
    for col in diag.columns:
        if col not in f:
            f[col] = np.nan
        f.loc[isA, col] = f.loc[isA, "source"].map(diag[col]).values
    f["diag_source"] = np.where(isA, "stored d=15 re-evaluation row (same EMC state)", "runner")
    return (f, pd.concat(sel, ignore_index=True),
            pd.concat(iters, ignore_index=True), pd.concat(ledg, ignore_index=True),
            pd.concat(prov, ignore_index=True), runs)


# ----------------------------------------------------------------- decomposition
def decomposition(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = f[f.fit_status != "FAILED"]
    for (c, n, rep), g in ok.groupby(["condition", "n", "replicate"]):
        by = {(int(r.k_est), int(r.start)): r for r in g.itertuples()}
        if len(by) != len(KS) * len(STARTS):
            continue
        m = lambda k, col: float(np.mean([getattr(by[(k, s)], col) for s in STARTS]))  # noqa: E731
        lapok = lambda k: all(by[(k, s)].lap_status == "OK" for s in STARTS)  # noqa: E731
        for k in KS[:-1]:
            k1 = k + 1
            base = {"condition": c, "n": n, "d": 15, "k_gen": K_GEN, "replicate": rep, "k_from": k, "k_to": k1,
                    "experiment_id": "historical_canonical_causal_20261106/signal_strength",
                    "generator_version": "canonical-clean-v1 A1 + exact coupling", "estimator_lineage": "E (consistent)",
                    "family_x": "poisson", "family_y": "bernoulli"}
            gx = 2 * (m(k1, "Q_X") - m(k, "Q_X")); gy = 2 * (m(k1, "Q_Y") - m(k, "Q_Y"))
            pc = -2 * (m(k1, "Q_Z") - m(k, "Q_Z")); pp = m(k1, "P_theta") - m(k, "P_theta")
            sd = m(k, "C_Q") - m(k1, "C_Q")
            se = math.sqrt(sum(getattr(by[(kk, s)], "mcse_Q_total") ** 2 for kk in (k, k1) for s in STARTS))
            rq = {**base, "criterion": "C_Q", "fit_gain_x": gx, "fit_gain_y": gy, "fit_gain_total": gx + gy,
                  "prior_cost": pc, "parameter_penalty": pp, "laplace_volume_term": np.nan,
                  "score_delta": sd, "selection_slack": sd, "closure_error": sd - (gx + gy - pc - pp),
                  "mc_se_slack": se, "status": "OK", "reason": ""}
            rows.append(rq)
            if lapok(k) and lapok(k1):
                lx = 2 * (m(k1, "lap_ll_X_mode") - m(k, "lap_ll_X_mode"))
                ly = 2 * (m(k1, "lap_ll_Y_mode") - m(k, "lap_ll_Y_mode"))
                vol = m(k1, "lap_volume") - m(k, "lap_volume")
                sl = m(k, "C_Lap") - m(k1, "C_Lap")
                h2 = m(k1, "H_Lap_2") - m(k, "H_Lap_2")
                dr = m(k1, "residual") - m(k, "residual")
                rows.append({**base, "criterion": "C_Lap", "fit_gain_x": lx, "fit_gain_y": ly, "fit_gain_total": lx + ly,
                             "prior_cost": np.nan, "parameter_penalty": pp, "laplace_volume_term": vol,
                             "score_delta": sl, "selection_slack": sl, "closure_error": sl - (lx + ly - vol - pp),
                             "two_dH_Lap": h2, "d_residual": dr, "slack_Q": sd,
                             "accounting_error": (sl - sd) - (h2 + dr),
                             "dlogdet_H": m(k1, "lap_logdet_H") - m(k, "lap_logdet_H"),
                             "dlogdet_prior_X": m(k1, "logdet_prior_X_hat") - m(k, "logdet_prior_X_hat"),
                             "dlogdet_Y_increment": m(k1, "logdet_Y_increment_hat") - m(k, "logdet_Y_increment_hat"),
                             "dtermA": m(k1, "termA") - m(k, "termA"), "dR3": m(k1, "R3") - m(k, "R3"),
                             "status": "OK", "reason": ""})
            else:
                rows.append({**base, "criterion": "C_Lap", "status": "INCOMPLETE",
                             "reason": "Laplace not OK at K or K+1 (not imputed)"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- paired
def ci_order(x: np.ndarray):
    x = np.sort(x); n = len(x)
    if n < 6:
        return np.nan, np.nan, np.nan
    r = 2 if n >= 8 else 1
    cov = 1 - 2 * binom.cdf(r - 1, n, 0.5)
    return x[r - 1], x[n - r], cov


METRICS_Q = ("selection_slack", "fit_gain_x", "fit_gain_y", "fit_gain_total", "prior_cost", "parameter_penalty")
METRICS_L = ("selection_slack", "fit_gain_x", "fit_gain_y", "laplace_volume_term", "two_dH_Lap", "d_residual",
             "dlogdet_prior_X", "dlogdet_Y_increment")


def paired(dec: pd.DataFrame, sel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    d45 = dec[(dec.k_from == 4) & (dec.status == "OK")]
    for n in sorted(d45.n.unique()):
        conds = sorted(d45[d45.n == n].condition.unique())
        contrasts = [(f"{c}-A", {c: 1, "A": -1}) for c in conds if c != "A"]
        if set("ABCD") <= set(conds):
            contrasts.append(("interaction D-B-C+A", {"D": 1, "B": -1, "C": -1, "A": 1}))
        for crit, metrics in (("C_Q", METRICS_Q), ("C_Lap", METRICS_L)):
            sub = d45[(d45.n == n) & (d45.criterion == crit)].set_index(["condition", "replicate"])
            for name, w in contrasts:
                for rep in range(1, 9):
                    if not all((c, rep) in sub.index for c in w):
                        rows.append({"n": n, "criterion": crit, "contrast": name, "replicate": rep,
                                     "metric": "ALL", "value": np.nan, "status": "INCOMPLETE"})
                        continue
                    for mt in metrics:
                        rows.append({"n": n, "criterion": crit, "contrast": name, "replicate": rep, "metric": mt,
                                     "value": sum(wt * float(sub.loc[(c, rep), mt]) for c, wt in w.items()),
                                     "status": "OK"})
    # selected-K changes
    sc = sel[sel.status == "COMPLETE"].set_index(["condition", "n", "replicate", "criterion"])
    for (c, n, rep, crit), r in sc.iterrows():
        if c == "A" or ("A", n, rep, crit) not in sc.index:
            continue
        rows.append({"n": n, "criterion": crit, "contrast": f"{c}-A", "replicate": rep,
                     "metric": "selected_k_change", "value": float(r.selected_k - sc.loc[("A", n, rep, crit)].selected_k),
                     "status": "OK"})
    per = pd.DataFrame(rows)
    summ = []
    for (n, crit, con, mt), g in per[per.status == "OK"].groupby(["n", "criterion", "contrast", "metric"]):
        x = g.value.to_numpy(float)
        lo, hi, cov = ci_order(x)
        try:
            p = float(wilcoxon(x, alternative="two-sided", method="exact").pvalue) if np.any(x != 0) else 1.0
        except ValueError:
            p = np.nan
        summ.append({"n": n, "criterion": crit, "contrast": con, "metric": mt, "replicates": len(x),
                     "median": float(np.median(x)), "mean": float(np.mean(x)), "min": float(x.min()),
                     "max": float(x.max()), "n_positive": int((x > 0).sum()), "n_negative": int((x < 0).sum()),
                     "n_zero": int((x == 0).sum()), "median_ci_lo": lo, "median_ci_hi": hi,
                     "median_ci_coverage": cov, "wilcoxon_exact_p_two_sided": p})
    return per, pd.DataFrame(summ)


def frequency(sel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (c, n, crit), g in sel.groupby(["condition", "n", "criterion"]):
        comp = g[g.status == "COMPLETE"]
        row = {"condition": c, "n": n, "criterion": crit, "datasets": len(g), "complete": len(comp),
               "incomplete": int((g.status != "COMPLETE").sum()),
               "exact": int((comp.label == "exact").sum()), "under": int((comp.label == "under").sum()),
               "over": int((comp.label == "over").sum()),
               "starts_agree": int(comp.starts_agree.astype(str).eq("True").sum()),
               "median_slack_45": float(comp.slack_45.median()) if len(comp) else np.nan}
        for k in KS:
            row[f"selected_K{k}"] = int((comp.selected_k == k).sum())
        out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------- hypotheses
def verdict_sr(s: pd.DataFrame, n, crit, con, mt, k_pos=6):
    r = s[(s.n == n) & (s.criterion == crit) & (s.contrast == con) & (s.metric == mt)]
    if r.empty:
        return None
    r = r.iloc[0]
    if r["median"] > 0 and r.n_positive >= k_pos:
        v = "SUPPORTED"
    elif r["median"] <= 0:
        v = "REFUTED"
    else:
        v = "INCONCLUSIVE"
    return v, r


def hypotheses(summ, dec, freq) -> pd.DataFrame:
    rows = []

    def add(h, statement, rule, v, evidence):
        rows.append({"hypothesis": h, "statement": statement, "prereg_rule": rule,
                     "verdict": v, "evidence": evidence})

    def ev(r):
        return (f"median={r['median']:.2f} [order-stat CI {r.median_ci_lo:.2f},{r.median_ci_hi:.2f}; cov "
                f"{r.median_ci_coverage:.3f}], positive {r.n_positive}/{r.replicates}, Wilcoxon exact p={r.wilcoxon_exact_p_two_sided:.4f}")
    x = verdict_sr(summ, 75, "C_Q", "B-A", "selection_slack")
    if x:
        add("P1_primary", "n=75: F-strong increases the C_Q 4->5 slack (paired B-A)",
            "support: median>0 and >=6/8 positive; refute: median<=0", x[0], ev(x[1]))
        for mt in ("fit_gain_x", "fit_gain_y", "prior_cost", "parameter_penalty"):
            y = verdict_sr(summ, 75, "C_Q", "B-A", mt)
            add(f"P1_mechanism:{mt}", f"B-A change of {mt} (4->5)", "reported, not tested", "DESCRIPTIVE", ev(y[1]))
    a = verdict_sr(summ, 75, "C_Q", "C-A", "fit_gain_y"); b = verdict_sr(summ, 75, "C_Q", "C-A", "selection_slack")
    if a and b:
        v = ("SUPPORTED" if a[0] == "SUPPORTED" and b[0] == "SUPPORTED" else
             "REFUTED" if b[1]["median"] <= 0 else "INCONCLUSIVE")
        add("S1_w", "n=75: w-strong increases fit_gain_y(4->5) and the C_Q slack (paired C-A)",
            "support: both medians>0 and >=6/8 positive; refute: slack median<=0", v,
            f"fit_gain_y: {ev(a[1])} | slack: {ev(b[1])}")
    i = verdict_sr(summ, 75, "C_Q", "interaction D-B-C+A", "selection_slack")
    if i:
        r = i[1]
        dA = summ[(summ.n == 75) & (summ.criterion == "C_Q") & (summ.metric == "selection_slack")].set_index("contrast")["median"]
        add("S2_interaction", "interaction I = slack_D - slack_B - slack_C + slack_A (C_Q, n=75)",
            "descriptive, two-sided", "DESCRIPTIVE",
            f"{ev(r)}; median D-A={dA.get('D-A', np.nan):.2f}, B-A={dA.get('B-A', np.nan):.2f}, C-A={dA.get('C-A', np.nan):.2f}")
    lap = dec[(dec.criterion == "C_Lap") & (dec.status == "OK") & (dec.k_from == 4)]
    if len(lap):
        npos = int((lap.two_dH_Lap > 0).sum())
        add("S3_positivity", "2 dH_Lap(4->5) > 0 in every start-mean pair", "refute: any pair <= 0",
            "SUPPORTED" if npos == len(lap) else "REFUTED", f"{npos}/{len(lap)} pairs positive; min={lap.two_dH_Lap.min():.2f}")
        for n in sorted(lap.n.unique()):
            med = lap[lap.n == n].groupby("condition").two_dH_Lap.median()
            for c in med.index:
                if c == "A":
                    continue
                add(f"S3_decrease:n{n}:{c}", f"median 2dH_Lap(4->5) lower in {c} than in A (n={n})",
                    "refute: median(strong) >= median(A)", "SUPPORTED" if med[c] < med["A"] else "REFUTED",
                    f"median A={med['A']:.2f}, {c}={med[c]:.2f}")
        add("S3_identity", "slack_Lap - slack_Q = 2dH_Lap + d resid (algebraic identity)", "max |error| (numerical)",
            "DERIVED+VERIFIED", f"max abs accounting error {np.nanmax(np.abs(lap.accounting_error)):.3g}")
    x = verdict_sr(summ, 150, "C_Q", "B-A", "selection_slack")
    if x:
        add("P2_n150", "n=150: paired B-A C_Q slack difference > 0", "support: median>0 and >=6/8; refute: median<=0",
            x[0], ev(x[1]))
    else:
        add("P2_n150", "n=150: paired B-A C_Q slack difference > 0", "not executed", "NOT_RUN", "")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- figures
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "lines.linewidth": 2})
    return plt


def savefig(fig, name):
    for ext in ("pdf", "png"):
        p = FIG / f"{name}.{ext}"
        need(not p.exists(), f"refuse to overwrite {p}")
        fig.savefig(p, dpi=200, bbox_inches="tight")


def figures(f, sel, dec, freq, iters, summ):
    plt = setup_mpl()
    import matplotlib.patches as mp
    FIG.mkdir(parents=True, exist_ok=True)
    hf = pd.read_csv(EX / "historical_k_selection_frequency.csv")
    cf = pd.read_csv(EX / "canonical_k_selection_frequency.csv")
    # 1 historical vs canonical
    items = []
    for r in hf.itertuples():
        nm = r.experiment_id.replace("initial_dual_expfam_scenario_", "initial 0.5 Sc.").replace("_exp1", "") \
            .replace("fixed_official_exp1_k1to9_scenario_", "fixed Sc.").replace("paper_condition_k_criteria_20261006 (#139 B)", "paper-type B")
        if "paper-type" in nm and r.k_gen != 5:
            continue
        items.append((f"{nm} {r.criterion.split(' ')[0]} K={r.k_gen}", r.exact, r.under, r.over, r.incomplete))
    for r in cf[(cf.k_gen == 5) & (cf.d == 15) & cf.experiment_id.str.contains("A1")].itertuples():
        items.append((f"canonical A1 {r.criterion} K=5 n={r.n}", r.exact, r.under, r.over, r.incomplete))
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(items) + 1.2))
    y = np.arange(len(items))[::-1]
    cols = [("exact", "#2a78d6"), ("under", "#eb6834"), ("over", "#6250d6"), ("incomplete", "#c3c2b7")]
    for i, (nm, *vals) in enumerate(items):
        tot = sum(vals); left = 0
        for (lab, col), v in zip(cols, vals):
            if v:
                ax.barh(y[i], v / tot, left=left, color=col, height=0.7, edgecolor="#fcfcfb", linewidth=1)
            left += v / tot
        ax.text(1.01, y[i], f"{vals[0]}/{tot}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([it[0] for it in items], fontsize=7.5)
    ax.set_xlim(0, 1.12); ax.set_xlabel("fraction of datasets")
    ax.set_title("K selection: historical series (K=3, n=150) vs canonical A1 (K_GEN=5, d=15)\nlabels: exact / total; series use different generators and lineages - do not pool", fontsize=9)
    ax.legend(handles=[mp.Patch(color=c, label=l) for l, c in cols], loc="lower center", bbox_to_anchor=(0.45, -0.25 - 0.004 * len(items)), ncol=4)
    savefig(fig, "fig1_historical_vs_canonical_selection"); plt.close(fig)

    ok = f[f.fit_status != "FAILED"]
    nlist = sorted(ok.n.unique())
    # 2 score curves (relative to K=5) per condition
    for n in nlist:
        conds = sorted(ok[ok.n == n].condition.unique())
        fig, axes = plt.subplots(2, len(conds), figsize=(2.6 * len(conds), 5), sharey="row", squeeze=False)
        for j, c in enumerate(conds):
            g = ok[(ok.n == n) & (ok.condition == c)]
            for i, crit in enumerate(("C_Q", "C_Lap")):
                ax = axes[i, j]
                m = g.groupby(["replicate", "k_est"])[crit].mean().unstack()
                if crit == "C_Lap":
                    bad = g[g.lap_status != "OK"].replicate.unique()
                    m = m.drop(index=[r for r in bad if r in m.index])
                rel = m.sub(m[5], axis=0)
                for rep, row in rel.iterrows():
                    ax.plot(row.index, row.values / n, color=COND_COLOR[c], alpha=0.45, lw=1.2, marker=COND_MARK[c], ms=3)
                ax.axhline(0, color=INK2, lw=0.8); ax.axvline(5, color=INK2, lw=0.6, ls=":")
                ax.set_title(f"{COND_LABEL[c]} - {crit}", fontsize=8.5)
                ax.set_ylim(-4, 12); ax.set_xticks(KS)
                if j == 0:
                    ax.set_ylabel(f"[{crit}(K) - {crit}(5)] / n")
                if i == 1:
                    ax.set_xlabel("candidate K")
        fig.suptitle(f"Criterion curves relative to K=5 (one line per replicate, mean of 2 starts), n={n}, d=15, K_GEN=5\nbelow 0 at K<5 = that K beats K=5 (under-selection)", fontsize=9)
        fig.tight_layout(); savefig(fig, f"fig2_criterion_curves_n{n}"); plt.close(fig)

    # 3 slack 4->5 paired
    fig, axes = plt.subplots(1, 2 * len(nlist), figsize=(3.3 * 2 * len(nlist), 3.6), squeeze=False)
    j = 0
    for n in nlist:
        for crit in ("C_Q", "C_Lap"):
            ax = axes[0, j]; j += 1
            d = dec[(dec.n == n) & (dec.k_from == 4) & (dec.criterion == crit) & (dec.status == "OK")]
            conds = sorted(d.condition.unique())
            piv = d.pivot(index="replicate", columns="condition", values="selection_slack")
            for rep, row in piv.iterrows():
                ax.plot(range(len(conds)), [row.get(c, np.nan) for c in conds], color="#c3c2b7", lw=0.8, zorder=1)
            for x, c in enumerate(conds):
                ax.scatter([x] * piv[c].notna().sum(), piv[c].dropna(), color=COND_COLOR[c], marker=COND_MARK[c], s=28,
                           zorder=3, edgecolor="#fcfcfb", linewidth=0.8)
                ax.plot([x - 0.25, x + 0.25], [piv[c].median()] * 2, color=INK, lw=2)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(range(len(conds))); ax.set_xticklabels([COND_LABEL[c].replace(" ", "\n", 1) for c in conds], fontsize=7.5)
            ax.set_title(f"{crit} slack 4->5, n={n}", fontsize=9); ax.set_ylabel("C(4) - C(5)  (>0 favours K=5)")
    fig.suptitle("Paired 4->5 selection slack by condition (grey lines join the same replicate; black bar = median)", fontsize=9)
    fig.tight_layout(); savefig(fig, "fig3_slack45_paired"); plt.close(fig)

    # 4 contributions
    d = dec[(dec.k_from == 4) & (dec.criterion == "C_Q") & (dec.status == "OK")]
    med = d.groupby(["n", "condition"])[["fit_gain_x", "fit_gain_y", "prior_cost", "parameter_penalty", "selection_slack"]].median()
    ex = pd.read_csv(EX / "k_increment_decomposition.csv")
    ex = ex[(ex.criterion == "C_Q") & (ex.k_from == 4)].groupby(["n", "d"])[["fit_gain_x", "fit_gain_y", "prior_cost", "parameter_penalty", "selection_slack"]].median()
    labels = [f"new {c} n={n}" for n, c in med.index] + [f"existing n={n} d={dd}" for n, dd in ex.index]
    allm = pd.concat([med.reset_index(drop=True), ex.reset_index(drop=True)], ignore_index=True)
    nn = [n for n, _ in med.index] + [n for n, _ in ex.index]
    fig, ax = plt.subplots(figsize=(8, 0.33 * len(labels) + 1.4))
    y = np.arange(len(labels))[::-1]
    for i in range(len(labels)):
        r = allm.iloc[i]; n = nn[i]
        ax.barh(y[i] + 0.18, r.fit_gain_x / n, height=0.34, color="#2a78d6", edgecolor="#fcfcfb")
        ax.barh(y[i] + 0.18, r.fit_gain_y / n, left=r.fit_gain_x / n, height=0.34, color="#1baf7a", edgecolor="#fcfcfb")
        ax.barh(y[i] - 0.18, r.prior_cost / n, height=0.34, color="#c3c2b7", edgecolor="#fcfcfb")
        ax.barh(y[i] - 0.18, r.parameter_penalty / n, left=r.prior_cost / n, height=0.34, color="#6250d6", edgecolor="#fcfcfb")
        ax.text(max((r.fit_gain_x + r.fit_gain_y), r.prior_cost + r.parameter_penalty) / n + 0.08, y[i],
                f"slack/n {r.selection_slack / n:+.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5); ax.set_xlabel("per node (median over replicates)")
    ax.set_title("C_Q 4->5: gain (X blue + Y aqua, upper bar) vs cost (Z prior grey + P_theta violet, lower bar)\nK=5 is favoured when the upper bar is longer", fontsize=9)
    ax.legend(handles=[mp.Patch(color=c, label=l) for l, c in (("fit gain X", "#2a78d6"), ("fit gain Y", "#1baf7a"),
                                                              ("Z prior cost (=2.84 n)", "#c3c2b7"), ("parameter penalty", "#6250d6"))],
              loc="lower center", bbox_to_anchor=(0.4, -0.12 - 0.3 / len(labels) * 4), ncol=4)
    ax.set_xlim(0, 7.5)
    savefig(fig, "fig4_xyz_contributions"); plt.close(fig)

    # 5/6 F and w comparison: selection distribution
    for name, pair in (("fig5_F_strength_selection", ("A", "B")), ("fig6_w_strength_selection", ("A", "C"))):
        fig, axes = plt.subplots(1, len(nlist) * 2, figsize=(3.0 * 2 * len(nlist), 3.1), squeeze=False)
        j = 0
        for n in nlist:
            for crit in ("C_Q", "C_Lap"):
                ax = axes[0, j]; j += 1
                for o, c in enumerate(pair):
                    r = freq[(freq.n == n) & (freq.condition == c) & (freq.criterion == crit)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    v = [r[f"selected_K{k}"] for k in KS]
                    ax.bar(np.array(KS) + (o - 0.5) * 0.38, v, width=0.36, color=COND_COLOR[c], label=f"{COND_LABEL[c]} (inc {r.incomplete})",
                           edgecolor="#fcfcfb")
                ax.axvline(5, color=INK2, lw=0.6, ls=":"); ax.set_xticks(KS)
                ax.set_title(f"{crit}, n={n}", fontsize=9); ax.set_xlabel("selected K"); ax.set_ylabel("datasets")
                ax.legend(fontsize=7)
        fig.suptitle(f"Selected K distribution: {COND_LABEL[pair[0]]} vs {COND_LABEL[pair[1]]} (K_GEN=5, d=15, 8 paired datasets)", fontsize=9)
        fig.tight_layout(); savefig(fig, name); plt.close(fig)

    # 7 C_Q vs C_Lap slack
    lap = dec[(dec.criterion == "C_Lap") & (dec.status == "OK")]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for c in sorted(lap.condition.unique()):
        g = lap[(lap.condition == c) & (lap.k_from == 4)]
        for n, mk in ((75, COND_MARK[c]), (150, COND_MARK[c])):
            h = g[g.n == n]
            axes[0].scatter(h.slack_Q, h.selection_slack, color=COND_COLOR[c], marker=mk, s=26 if n == 75 else 46,
                            edgecolor="#fcfcfb" if n == 75 else INK, linewidth=0.8, label=f"{COND_LABEL[c]} n={n}" if len(h) else None)
        axes[1].scatter(g.two_dH_Lap / g.n, (g.selection_slack - g.slack_Q) / g.n, color=COND_COLOR[c], marker=COND_MARK[c], s=26,
                        edgecolor="#fcfcfb", label=COND_LABEL[c])
    lim = [min(lap.slack_Q.min(), lap.selection_slack.min()), max(lap.slack_Q.max(), lap.selection_slack.max())]
    g4 = lap[lap.k_from == 4]
    lim = [min(g4.slack_Q.min(), g4.selection_slack.min()) - 10, max(g4.slack_Q.max(), g4.selection_slack.max()) + 10]
    axes[0].plot(lim, lim, color=INK2, lw=0.8, ls="--"); axes[0].axhline(0, color=INK2, lw=0.6); axes[0].axvline(0, color=INK2, lw=0.6)
    axes[0].set_xlabel("C_Q slack 4->5"); axes[0].set_ylabel("C_Lap slack 4->5"); axes[0].set_title("Same fits: C_Lap vs C_Q slack (dashed y=x)", fontsize=9)
    axes[0].legend(fontsize=6.5)
    xx = np.linspace(0, max(1e-9, (g4.two_dH_Lap / g4.n).max()) * 1.1, 2)
    axes[1].plot(xx, xx, color=INK2, lw=0.8, ls="--")
    axes[1].set_xlabel("2 dH_Lap(4->5) / n"); axes[1].set_ylabel("(slack_Lap - slack_Q) / n")
    axes[1].set_title("Difference = 2dH_Lap + d residual (dashed: residual 0)", fontsize=9)
    fig.tight_layout(); savefig(fig, "fig7_cq_vs_clap"); plt.close(fig)

    # 8 n, d, K trends: fit gain per node
    ex = pd.read_csv(EX / "k_increment_decomposition.csv")
    ex = ex[(ex.criterion == "C_Q") & (ex.k_from == 4)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for dd, col, mk in ((15, "#2a78d6", "o"), (30, "#eb6834", "s"), (60, "#1baf7a", "^")):
        g = ex[ex.d == dd].groupby("n")
        axes[0].plot(g.fit_gain_x.median().index, g.fit_gain_x.median() / g.fit_gain_x.median().index, color=col, marker=mk, label=f"d={dd}")
        axes[1].plot(g.fit_gain_y.median().index, g.fit_gain_y.median() / g.fit_gain_y.median().index, color=col, marker=mk, label=f"d={dd}")
        axes[2].plot(g.selection_slack.median().index, g.selection_slack.median() / g.selection_slack.median().index, color=col, marker=mk, label=f"d={dd}")
    d4 = dec[(dec.k_from == 4) & (dec.criterion == "C_Q") & (dec.status == "OK")]
    for c in sorted(d4.condition.unique()):
        if c == "A":
            continue
        g = d4[d4.condition == c].groupby("n")
        for ax, col in zip(axes, ("fit_gain_x", "fit_gain_y", "selection_slack")):
            s = g[col].median()
            ax.plot(s.index, s / s.index, color=COND_COLOR[c], marker=COND_MARK[c], ls=":", lw=1.5, label=f"{COND_LABEL[c]} (d=15)")
    for ax, t in zip(axes, ("4->5 fit gain X / n", "4->5 fit gain Y / n", "C_Q slack 4->5 / n")):
        ax.set_xlabel("n"); ax.set_title(t, fontsize=9); ax.set_xticks([50, 75, 100, 150])
    axes[2].axhline(0, color=INK2, lw=0.8); axes[0].legend(fontsize=7)
    fig.suptitle("Medians over 8 datasets; solid = existing canonical states (A1, d-sweep, R2), dotted = this experiment's strong conditions", fontsize=9)
    fig.tight_layout(); savefig(fig, "fig8_n_d_trends"); plt.close(fig)

    # 9 convergence diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    fo = f[(f.fit_status != "FAILED")]
    for c in sorted(fo.condition.unique()):
        for n in sorted(fo.n.unique()):
            g = fo[(fo.condition == c) & (fo.n == n)].groupby("k_est").gram_rel_change.median()
            if len(g):
                axes[0].plot(g.index, g.values, color=COND_COLOR[c], marker=COND_MARK[c], ls="-" if n == 75 else "--",
                             label=f"{COND_LABEL[c]} n={n}")
    axes[0].set_xlabel("candidate K"); axes[0].set_ylabel("median gram_rel_change at iteration 8")
    axes[0].set_title("Non-convergence check (lower = more stable)", fontsize=9); axes[0].legend(fontsize=6.5)
    it = iters[(iters.k_est.isin([4, 5]))]
    for c in sorted(it.condition.unique()):
        for k, ls in ((4, ":"), (5, "-")):
            g = it[(it.condition == c) & (it.k_est == k) & (it.n == 75)]
            if g.empty:
                continue
            last = g[g.iteration == 8].set_index(["replicate", "start"]).C_Q
            rel = g.set_index(["replicate", "start"]).C_Q - last.reindex(g.set_index(["replicate", "start"]).index).values
            m = rel.groupby(g.iteration.values).median()
            axes[1].plot(m.index, m.values, color=COND_COLOR[c], ls=ls, marker=COND_MARK[c], ms=3, label=f"{c} K={k}")
    axes[1].set_yscale("symlog", linthresh=10); axes[1].set_xlabel("EM iteration"); axes[1].set_ylabel("C_Q(it) - C_Q(8), median")
    axes[1].set_title("C_Q trajectory, n=75 new fits (A not refit)", fontsize=9); axes[1].legend(fontsize=6.5, ncol=2)
    fig.tight_layout(); savefig(fig, "fig9_convergence_diagnostics"); plt.close(fig)


# ----------------------------------------------------------------- main
def main() -> int:
    need(not OUT.exists(), f"output exists: {OUT}")
    f, sel, iters, ledg, prov, runs = load()
    gates = json.loads((SS / "gates" / "gates.json").read_text(encoding="utf-8"))
    prereg = json.loads((SS / "gates" / "prereg.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True)
    # integrity
    key = ["condition", "n", "replicate", "k_est", "start"]
    need(not f.duplicated(key).any(), "duplicate fit key")
    planned = {(75, "A"): 112, (75, "B"): 112, (75, "C"): 112, (75, "D"): 112, (150, "A"): 112, (150, "B"): 112}
    got = f.groupby(["n", "condition"]).size().to_dict()
    for (n, c), v in planned.items():
        if n in runs:
            need(got.get((n, c), 0) == v, f"row count {n} {c}: {got.get((n, c))}")
    num = f.select_dtypes("number")
    nonfinite = int((~np.isfinite(num.to_numpy(float)) & num.notna().to_numpy()).sum())
    dec = decomposition(f)
    per, summ = paired(dec, sel)
    freq = frequency(sel)
    hyp = hypotheses(summ, dec, freq)
    # recompute selection from fits (independent of runner's select)
    mism = 0
    for r in sel[sel.status == "COMPLETE"].itertuples():
        g = f[(f.condition == r.condition) & (f.n == r.n) & (f.replicate == r.replicate)]
        m = g.groupby("k_est")[r.criterion].mean()
        best = m.min(); ties = sorted(m[m <= best + 1e-12].index)
        mism += int(ties[0] != r.selected_k)
    # CSVs
    cond_rows = []
    for n in sorted(runs):
        for c in ("A", "B", "C", "D"):
            p = prov[(prov.n == n) & (prov.condition == c)]
            if p.empty:
                continue
            ex = got.get((n, c), 0)
            cond_rows.append({"condition": c, "label": COND_LABEL[c], "n": n, "d": 15, "k_gen": K_GEN,
                              "candidate_k": "1..7", "starts": 2, "replicates": 8, "L": 5, "em_iterations": 8,
                              "family_x": "poisson", "family_y": "bernoulli",
                              "mean_row_norm_sq": float(p.mean_row_norm_sq.mean()), "FtF_per_direction": float(p.FtF_per_direction.mean()),
                              "f_scale_factor": float(p.f_scale_factor.iloc[0]), "w": float(p.w.iloc[0]), "w0": float(p.w0.iloc[0]),
                              "w2_times_K": float(p.w2_times_K.iloc[0]), "population_density": float(p.population_density.iloc[0]),
                              "mean_y_density_upper": float(p.y_density_upper.mean()),
                              "mean_bernoulli_fisher_p1mp": float(p.bernoulli_fisher_mean_p1mp.mean()),
                              "mean_x": float(p.x_mean.mean()), "max_rate_truth": float(p.max_rate_truth.max()),
                              "fit_source": "reused A1 EMC checkpoint-8 states" if c == "A" else "new MCEM fits",
                              "planned_fits": planned.get((n, c), 0), "executed_rows": ex,
                              "new_fits": 0 if c == "A" else ex,
                              "failed_fits": int(((f.n == n) & (f.condition == c) & (f.fit_status == "FAILED")).sum())})
    for n, c in (("150", "C"), ("150", "D")):
        cond_rows.append({"condition": c, "label": COND_LABEL[c], "n": int(n), "planned_fits": 0, "executed_rows": 0,
                          "fit_source": "NOT RUN (fixed before results: 480-fit cap; Priority 2 = B only)"})
    save(pd.DataFrame(cond_rows), "experiment_conditions.csv")
    save(prov, "dataset_provenance.csv")
    save(f, "fit_results.csv")
    save(iters, "iteration_diagnostics.csv")
    long = []
    for r in f[f.fit_status != "FAILED"].itertuples():
        for crit, val in (("C_Q", r.C_Q), ("C_Lap", r.C_Lap if r.lap_status == "OK" else np.nan)):
            long.append({"condition": r.condition, "n": r.n, "replicate": r.replicate, "k_est": r.k_est, "start": r.start,
                         "criterion": crit, "value": val, "status": "OK" if np.isfinite(val) else f"LAPLACE_{r.lap_status}",
                         "Q_X": r.Q_X, "Q_Y": r.Q_Y, "Q_Z": r.Q_Z, "P_theta": r.P_theta,
                         "lap_volume": getattr(r, "lap_volume", np.nan), "lap_logdet_H": getattr(r, "lap_logdet_H", np.nan)})
    save(pd.DataFrame(long), "criterion_scores.csv")
    save(dec, "score_decomposition.csv")
    save(sel, "selection_by_replicate.csv")
    save(freq, "selection_frequency.csv")
    save(per, "paired_comparison.csv")
    save(summ, "paired_comparison_summary.csv")
    save(hyp, "hypothesis_evidence.csv")
    led = ledg.assign(role="production")
    reused = f[f.source != "new_fit"][key + ["fit_id", "model_seed", "source"]].assign(status="REUSED", role="baseline A (no MCEM)")
    gate = pd.read_csv(SS / "gates" / "pilot_fits.csv")[["condition", "n", "replicate", "k_est", "start", "model_seed", "runtime_s"]]
    gate = pd.concat([gate.assign(status="FIT", role="pilot (not counted as data)"),
                      pd.DataFrame([{"condition": "A", "n": 75, "replicate": 1, "k_est": 5, "start": 1, "status": "FIT",
                                     "role": "replication gate (not counted as data)"}])], ignore_index=True)
    save(pd.concat([led, reused, gate], ignore_index=True), "execution_ledger.csv")
    new_fits = int((led.status == "FIT").sum() + (led.status == "FAILED").sum())
    figures(f, sel, dec, freq, iters, summ)
    manifest = {
        "script": "tools/research_audit/analyze_k_signal_strength.py", "mcem_fits": 0,
        "prereg_sha256_in_file": prereg["sha256"], "gates_status": gates["status"],
        "runs": {str(n): {k: runs[n][k] for k in ("status", "new_fits", "failed_fits", "reused_A_states", "wall_clock_seconds",
                                                  "git_head", "fit_states_sha256")} for n in runs},
        "new_mcem_fits_total_incl_gates": new_fits + 5, "fit_cap": 480,
        "integrity": {"duplicate_keys": 0, "nonfinite_numeric_cells": nonfinite,
                      "selection_recompute_mismatches": mism,
                      "max_abs_cq_closure": float(np.nanmax(np.abs(dec[dec.criterion == "C_Q"].closure_error))),
                      "max_abs_clap_closure": float(np.nanmax(np.abs(dec[(dec.criterion == "C_Lap") & (dec.status == "OK")].closure_error))),
                      "max_abs_identity_error_fit": float(np.nanmax(np.abs(f.identity_error))),
                      "max_abs_P_Z_minus_formula": float(np.nanmax(np.abs(f.P_Z - f.n * f.k_est * C2))),
                      "laplace_not_ok_fits": int((f.lap_status.fillna("FAILED") != "OK").sum()),
                      "failed_fits": int((f.fit_status == "FAILED").sum()),
                      "retries": 0, "seed_rescue": False, "dataset_replacement": False},
        "outputs_sha256": {p.name: sha256(p) for p in sorted(OUT.glob("*.csv"))},
        "figures_sha256": {p.name: sha256(p) for p in sorted(FIG.glob("*"))},
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(json.dumps(manifest["integrity"], indent=1))
    print(hyp[["hypothesis", "verdict", "evidence"]].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
