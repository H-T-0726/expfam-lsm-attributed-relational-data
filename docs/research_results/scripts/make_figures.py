"""Figures for docs/research_results, regenerated from primary CSVs read with ``git show``.

Only experiments whose numbers were re-aggregated in verify_primary_numbers.py are drawn.
Each figure states lineage, conditions, replicate count and source in its title/footer.
Lineages are never merged into one curve (KI-002).

Usage (repository root):  python docs/research_results/scripts/make_figures.py
"""
from __future__ import annotations

import csv
import io
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from verify_primary_numbers import REFS, KX, num  # noqa: E402  (same directory)

OUT = Path(__file__).resolve().parents[1] / "figures"
C1, C2, C3, C4, C7 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
GRAY = "#9a9994"
INK, INK2 = "#0b0b0b", "#52514e"
OUTCOME_COLORS = {"exact": C1, "under": C2, "over": C7, "incomplete": GRAY}
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 9.5, "lines.linewidth": 2,
})


def rows(ref, path):
    out = subprocess.run(["git", "show", f"{REFS[ref]}:{path}"], check=True, capture_output=True).stdout
    return list(csv.DictReader(io.StringIO(out.decode("utf-8"))))


def footer(fig, text):
    import textwrap
    fig.text(0.01, 0.005, chr(10).join(textwrap.wrap(text, 175)), fontsize=7, color=INK2, ha="left", va="bottom")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ fixed lineage (C)
def fig_fixed_sweeps():
    for kind, col, fname, xlabel in [("exp2/fixed_exp2_n_sweep_agg.csv", "n", "fig02_fixed_n_sweep.png", "n (nodes)"),
                                     ("exp3/fixed_exp3_d_sweep_agg.csv", "d", "fig03_fixed_d_sweep.png", "d (attribute dimension)")]:
        p = f"expfam/results/fixed_official/{kind}"
        r = rows("MAIN", p)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
        for ax, s, c in zip(axes, "ABC", (C1, C2, C3)):
            rr = sorted((x for x in r if x["scenario"] == s), key=lambda x: int(x[col]))
            xs = [int(x[col]) for x in rr]
            ys = [num(x["rmse_z_mean"]) for x in rr]
            sd = [num(x["rmse_z_std"]) for x in rr]
            ax.errorbar(xs, ys, yerr=sd, color=c, marker="o", ms=5, capsize=3)
            ax.set_title(f"Scenario {s}")
            ax.set_xlabel(xlabel)
            ax.set_ylim(bottom=0)
        axes[0].set_ylabel("Procrustes RMSE(Z)  (mean ± SD, 10 trials)")
        fig.suptitle(f"Fixed lineage (C, DualExpFamLSMFixed): RMSE(Z) vs {col}  — K_TRUE=3, historical generator", fontsize=10)
        footer(fig, f"Source: {p} @ {REFS['MAIN']}.  Each panel has its own y-axis (scenarios differ in scale). Not comparable with the 0.5 lineage (B).")
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        save(fig, fname)


def fig_fixed_misspec():
    fams = ["gaussian", "bernoulli", "poisson"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, s in zip(axes, "abc"):
        p = f"expfam/results/fixed_official/exp4/fixed_exp4_scen_{s}_ratios.csv"
        r = [x for x in rows("MAIN", p) if x["fix_w"] == "False" and x["fix_x"] == "False"]
        m = np.full((3, 3), np.nan)
        for x in r:
            m[fams.index(x["est_x"]), fams.index(x["est_y"])] = num(x["ratio_vs_oracle"])
        im = ax.imshow(np.log10(m), cmap="Blues", vmin=0, vmax=np.log10(50))
        for i in range(3):
            for j in range(3):
                v = m[i, j]
                ax.text(j, i, f"{v:.2f}×", ha="center", va="center", fontsize=8,
                        color="white" if np.log10(v) > 0.9 else INK)
        ax.set_xticks(range(3), ["Gauss", "Bern", "Pois"])
        ax.set_yticks(range(3), ["Gauss", "Bern", "Pois"])
        ax.set_xlabel("estimated Y family")
        ax.set_title(f"Scenario {s.upper()}")
        ax.grid(False)
    axes[0].set_ylabel("estimated X family")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("RMSE(Z) / RMSE(Z | correct families), log scale")
    cb.set_ticks([0, 1, np.log10(50)], labels=["1×", "10×", "50×"])
    fig.suptitle("Fixed lineage (C): RMSE(Z) ratio vs. the correct-family fit (mean of 10 trials). True X/Y: A Pois/Bern, B Gauss/Pois, C Bern/Gauss", fontsize=10)
    footer(fig, f"Source: expfam/results/fixed_official/exp4/fixed_exp4_scen_{{a,b,c}}_ratios.csv @ {REFS['MAIN']} (rows fix_w=False, fix_x=False; ablation rows excluded).")
    fig.subplots_adjust(left=0.07, right=0.88, top=0.85, bottom=0.17, wspace=0.35)
    save(fig, "fig04_fixed_misspecification.png")


# ------------------------------------------------------------------ sparse vs dense Y (lineage E, per-column)
def fig_sparse_dense():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    p = "expfam/results/story_diagnostics/complementary_blocks_consistent_20260821_paired.csv"
    r = rows("MAIN", p)
    comps = ["y_only", "single_bernoulli", "single_poisson", "single_gaussian"]
    ax = axes[0]
    x = np.arange(len(comps))
    for off, rate, c, lab in [(-0.18, "1.0", C1, "dense Y (y_obs_rate=1.0)"), (0.18, "0.1", C2, "sparse Y (y_obs_rate=0.1)")]:
        mu = [num(next(z for z in r if z["comparator"] == cc and z["y_obs_rate"] == rate)["delta_rmse_mean"]) for cc in comps]
        sd = [num(next(z for z in r if z["comparator"] == cc and z["y_obs_rate"] == rate)["delta_rmse_std"]) for cc in comps]
        ax.bar(x + off, mu, width=0.34, color=c, yerr=sd, capsize=3, label=lab)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x, comps, rotation=12)
    ax.set_ylabel("ΔRMSE_Z = comparator − per_column_all\n(mean ± SD over 10 trials; >0 favours joint)")
    ax.set_title("Complementary blocks (#27): n=80, d=9, K_TRUE=3")
    ax.legend(frameon=False, fontsize=8)
    p2 = "expfam/results/misspecification_m1/phase_m1_20261013/paired_rmse_z.csv"
    r2 = rows("MAIN", p2)
    ax = axes[1]
    for i, (reg, c) in enumerate([("dense", C1), ("sparse", C2)]):
        d = [num(z[f"delta_RMSE_Z_{reg}"]) for z in r2]
        jit = np.linspace(-0.12, 0.12, len(d))
        ax.scatter(i + jit, d, s=22, color=c, edgecolor="#fcfcfb", linewidth=0.8, zorder=3)
        ax.hlines(mean(d), i - 0.25, i + 0.25, color=INK, lw=2, zorder=4)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks([0, 1], ["dense Y (100%)", "sparse Y (10%)"])
    ax.set_ylabel("ΔRMSE_Z = wrong(B→P) − correct\n(points = 20 replicates, bar = mean)")
    ax.set_title("M1 misspecification (#134): n=75, d=12, K=3")
    fig.suptitle("Per-column prototypes (left: lineage E+F, right: lineage E-3; not adopted): effect depends on how much Y is observed", fontsize=10)
    footer(fig, f"Sources: {p} and {p2} @ {REFS['MAIN']}. Two different generators — panels are not directly comparable.")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save(fig, "fig05_sparse_vs_dense_y.png")


# ------------------------------------------------------------------ K selection
def _sel(fits, score="C_Q"):
    agg = defaultdict(lambda: defaultdict(list))
    for x in fits:
        agg[x["dataset_id"]][int(x["k"])].append(num(x[score]))
    out = {}
    for ds, kk in agg.items():
        means = {k: mean(v) for k, v in kk.items()}
        m = min(means.values())
        out[ds] = min(k for k, v in means.items() if v <= m + 1e-9)
    return out


def _lap_sel(fits):
    by = defaultdict(list)
    for x in fits:
        by[x["dataset_id"]].append(x)
    return {ds: (None if any(z["lap_status"] != "OK" for z in xs) else _sel(xs, "C_Lap")[ds]) for ds, xs in by.items()}


def _outcome(k, kt):
    return "incomplete" if k is None else ("exact" if k == kt else ("under" if k < kt else "over"))


def _stack(ax, labels, counts, horizontal=False):
    left = np.zeros(len(labels))
    for o in ("exact", "under", "over", "incomplete"):
        v = np.array([c.get(o, 0) for c in counts])
        kw = dict(color=OUTCOME_COLORS[o], label=o, edgecolor="#fcfcfb", linewidth=1)
        if o == "incomplete":
            kw["hatch"] = "///"
        if horizontal:
            ax.barh(labels, v, left=left, **kw)
        else:
            ax.bar(labels, v, bottom=left, **kw)
        left += v


def fig_kexp2_main():
    p = f"{KX}/main/per_fit.csv"
    r = rows("KEXP2", p)
    sel = _sel(r)
    meta = {x["dataset_id"]: (x["series"], x["regime"], int(x["k_true"])) for x in r}
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.2), sharey=True)
    for ax, reg in zip(axes, "ab"):
        labels, counts = [], []
        for s in "OABC":
            for kt in (1, 3, 5, 7, 9):
                c = defaultdict(int)
                for ds, k in sel.items():
                    if meta[ds] == (s, reg, kt):
                        c[_outcome(k, kt)] += 1
                labels.append(f"{s}-{kt}")
                counts.append(c)
        _stack(ax, labels, counts)
        ax.set_title(f"regime {reg}: " + ("per-direction signal fixed (PRIMARY)" if reg == "a" else "total signal fixed (pre-registered sensitivity)"))
        ax.set_ylabel("datasets (of 10)")
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, ncol=4, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.95))
    axes[1].set_xlabel("series-K_TRUE   (O: Gauss-X/Bern-Y, A: Pois-X/Bern-Y, B: Gauss-X/Pois-Y, C: Bern-X/Gauss-Y)")
    fig.suptitle("K-exp2 main run (lineage E-scalar, prototype): C_Q selection, n=150, d=15, L=10, 10 MCEM iterations, 2 inits, 8,000 fits", fontsize=10)
    footer(fig, f"Source: {p} @ {REFS['KEXP2']} (unmerged branch). Selection = argmin of the 2-init mean C_Q, recomputed from per_fit.csv. C_Q is not Schwarz BIC.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    save(fig, "fig06_kexp2_cq_selection.png")


def fig_kexp2_cq_vs_clap():
    p = f"{KX}/clap_comparison/run/evaluations.csv"
    r = rows("KEXP2", p)
    cells = [("A", 7), ("A", 9), ("B", 9), ("O", 9)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True)
    for ax, src in zip(axes, ("main10", "conv40")):
        labels, counts = [], []
        for s, kt in cells:
            rr = [x for x in r if x["source"] == src and x["series"] == s and x["regime"] == "b" and int(x["k_true"]) == kt]
            if not rr:
                continue
            q = _sel(rr)
            l = _lap_sel(rr)
            for name, d in (("C_Q", q), ("C_Lap", l)):
                c = defaultdict(int)
                for k in d.values():
                    c[_outcome(k, kt)] += 1
                labels.append(f"{s}-b-{kt} {name}")
                counts.append(c)
        _stack(ax, labels[::-1], counts[::-1], horizontal=True)
        ax.set_title(f"{src}: θ̂ after {'10' if src == 'main10' else '40'} MCEM iterations")
        ax.set_xlabel("datasets (of 10)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=4, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Same fits, two criteria: C_Q (Q-based, ICL-type) vs C_Lap (Laplace approximation, not an exact marginal likelihood)", fontsize=10)
    footer(fig, f"Source: {p} @ {REFS['KEXP2']}. INCOMPLETE = any of the dataset's 20 C_Lap evaluations not OK (denominator stays 10). Exploratory.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    save(fig, "fig07_kexp2_cq_vs_clap.png")


def fig_mixedx_curves():
    p = f"{KX}/mixedx_highk_20261126/run/per_fit.csv"
    r = [x for x in rows("KEXP2", p) if x["phase"] == "main"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), sharey=False)
    for ax, kt in zip(axes, (5, 7, 9)):
        rr = [x for x in r if int(x["k_true"]) == kt]
        for score, c, lab in (("C_Q", C2, "C_Q"), ("C_Lap", C1, "C_Lap")):
            per = defaultdict(lambda: defaultdict(list))
            for x in rr:
                if score == "C_Lap" and x["lap_status"] != "OK":
                    continue
                per[x["dataset_id"]][int(x["k"])].append(num(x[score]))
            ks = list(range(1, 11))
            meds, used = [], []
            for k in ks:
                diffs = [mean(v[k]) - mean(v[kt]) for v in per.values() if len(v.get(k, [])) == 2 and len(v.get(kt, [])) == 2]
                meds.append(median(diffs) if diffs else np.nan)
                used.append(len(diffs))
            ax.plot(ks, meds, color=c, marker="o", ms=4, label=f"{lab} (datasets used per K: {min(used)}–{max(used)} of 10)")
        ax.axhline(0, color=INK2, lw=1)
        ax.axvline(kt, color=GRAY, lw=1, ls="--")
        ax.set_title(f"K_TRUE = {kt}")
        ax.set_xlabel("candidate K")
        ax.set_ylim(-400, 1500)
    axes[0].set_ylabel("median over 10 datasets of\nC(K) − C(K_TRUE)   (<0 means K beats K_TRUE)")
    for ax in axes:
        ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.suptitle("Mixed-family X (G5/B5/P5, Bernoulli-Y, regime-b signal; lineage E per-column, prototype): criterion curves", fontsize=10)
    footer(fig, f"Source: {p} @ {REFS['KEXP2']}. n=150, L=10, 10 MCEM iterations, 2 inits; C_Lap evaluations with non-OK status dropped per K. y-axis clipped at 1500.")
    fig.tight_layout(rect=(0.01, 0.06, 1, 0.93))
    save(fig, "fig08_mixedx_criterion_curves.png")


def fig_pf3():
    p = "expfam/results/k_selection/paper_fig3_condition_criteria_20261109/selection_per_trial.csv"
    r = rows("PF3", p)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, (crit, name) in zip(axes, (("CQ", "C_Q"), ("CLap", "C_Lap"), ("CELBO", "C_ELBO"))):
        labels, counts = [], []
        for kg in (1, 3, 5, 7, 9):
            c = defaultdict(int)
            for x in r:
                if x["criterion"] == crit and int(x["K_GEN"]) == kg:
                    o = x["outcome"] if x["outcome"] in ("exact", "under", "over") else "incomplete"
                    c[o] += 1
            labels.append(str(kg))
            counts.append(c)
        _stack(ax, labels, counts)
        ax.set_title(name)
        ax.set_xlabel("K_GEN")
    axes[0].set_ylabel("trials (of 10)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=4, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.92))
    fig.suptitle("Strong-signal condition near the original paper's Fig.3 (#165; lineage E, prototype; NOT a reproduction): per-trial selection", fontsize=10)
    footer(fig, f"Source: {p} @ {REFS['PF3']} (unmerged). One dataset per K_GEN × 10 trials (inits), n=150, d=15, L=10, 10 iterations. Trials share data: not independent replicates.")
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    save(fig, "fig09_paper_fig3_condition.png")


def fig_clean_nsweep():
    p = "expfam/results/k_selection/clean_true_k_asymptotics_20260904/selection_matrix.csv"
    r = [x for x in rows("MAIN", p) if x["K_TRUE"] == "5"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for crit, c, off in (("S1", C1, -3), ("S2", C2, 3)):
        ns = sorted({int(x["n"]) for x in r})
        for n in ns:
            ks = [int(x["selected_k"]) for x in r if x["criterion"] == crit and int(x["n"]) == n]
            ax.scatter([n + off] * len(ks) + np.linspace(-1.5, 1.5, len(ks)), ks, s=14, color=c, alpha=0.6, edgecolor="none")
        ax.plot([n + off for n in ns], [mean(int(x["selected_k"]) for x in r if x["criterion"] == crit and int(x["n"]) == n) for n in ns],
                color=c, marker="o", label={"S1": "S1 held-out log score", "S2": "S2 = C_Q (Q-based)"}[crit])
    ax.axhline(5, color=GRAY, ls="--", lw=1)
    ax.set_xlabel("n (nodes)")
    ax.set_ylabel("selected K (points = 8 replicates; line = mean)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Clean true-K n-sweep, K_TRUE=5 (lineage E-scalar): Poisson-X/Bern-Y, d=15, 8 iterations", fontsize=9)
    footer(fig, f"Source: {p} @ {REFS['MAIN']}. Fixed EM budget confounds n. Not consistency evidence.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig10_clean_n_sweep_k5.png")


# ------------------------------------------------------------------ real data
def fig_real():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    p = "expfam/results/real_data/cora_strict_heldout_family_main_20261124/heldout_scores.csv"
    r = [x for x in rows("THESIS", p) if x["estimator"] == "mc_predictive"]
    sc = {(x["arm"], x["K"], x["split_index"], x["init_seed"]): num(x["log_score"]) for x in r}
    ax = axes[0]
    for i, (K, a, b, c) in enumerate([("6", "BX", "GX", C1), ("3", "BX", "GX", C3), ("6", "BX", "GXc", C2), ("6", "GXc", "GX", C4)]):
        keys = [(s, j) for (arm, k, s, j) in sc if arm == a and k == K and (b, K, s, j) in sc]
        d = [1e4 * (sc[(a, K, s, j)] - sc[(b, K, s, j)]) for s, j in keys]
        ax.scatter(i + np.linspace(-0.15, 0.15, len(d)), d, s=20, color=c, edgecolor="#fcfcfb", linewidth=0.8, zorder=3)
        ax.hlines(mean(d), i - 0.25, i + 0.25, color=INK, lw=2, zorder=4)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(4), ["K=6\nBX−GX", "K=3\nBX−GX", "K=6\nBX−GXc", "K=6\nGXc−GX"])
    ax.set_ylabel("Δ mean held-out log score ×10⁻⁴ (nats/pair)\npoints = 15 split×seed cells, bar = mean")
    ax.set_title("Cora strict held-out (CORA-SH1, 1 network, 5 splits × 3 seeds)")
    p2 = "expfam/results/real_data/movielens_userdisjoint/movielens_userdisjoint_20260822_paired.csv"
    r2 = rows("MAIN", p2)
    ax = axes[1]
    names = [("delta_primary", "mixed(log) − genre_only", C1)]
    rp = sorted({x["contrast"] for x in r2 if "rawpoisson" in x["contrast"]})
    if rp:
        names.append((rp[0], "raw-Poisson − log-count", C2))
    for i, (cn, lab, c) in enumerate(names):
        d = [num(x["value"]) for x in r2 if x["contrast"] == cn]
        ax.scatter(i + np.linspace(-0.15, 0.15, len(d)), d, s=16, color=c, edgecolor="#fcfcfb", linewidth=0.6, zorder=3)
        ax.hlines(mean(d), i - 0.25, i + 0.25, color=INK, lw=2, zorder=4)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(names)), [n[1] for n in names])
    ax.set_ylabel("Δ test Y log score per pair\n(points = 30 user-disjoint splits)")
    ax.set_title("MovieLens user-disjoint (#33; per-column prototype)")
    fig.suptitle("Real data, one lineage per panel (descriptive; cells/splits are not independent samples)", fontsize=10)
    footer(fig, f"Sources: {p} @ {REFS['THESIS']} (unmerged; lineage E-scalar + BX-Newton) and {p2} @ {REFS['MAIN']} (lineage E+F).")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save(fig, "fig11_real_data_deltas.png")


if __name__ == "__main__":
    fig_fixed_sweeps()
    fig_fixed_misspec()
    fig_sparse_dense()
    fig_kexp2_main()
    fig_kexp2_cq_vs_clap()
    fig_mixedx_curves()
    fig_pf3()
    fig_clean_nsweep()
    fig_real()
