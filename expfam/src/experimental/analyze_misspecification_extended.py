"""Generate publication-size figures and a factual report from MX2 CSVs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import run_misspecification_extended as mx

OUT = mx.OUT
FIG = OUT / "figures"
REPORT = ROOT / "reports" / "misspecification_extended" / "results_20261110.md"
ARM_LABELS = {
    "correct": "Correct", "bp6": "B→P (6)", "bg6": "B→G (6)",
    "pg3": "P→G (3)", "allg": "All Gaussian",
    "bp2": "B→P (2)", "bp4": "B→P (4)", "bg2": "B→G (2)",
    "bg4": "B→G (4)", "pg1": "P→G (1)", "pg2": "P→G (2)",
}


def save(name):
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()


def maybe_plot():
    fits = pd.read_csv(OUT / "fit_results.csv")
    pairs = pd.read_csv(OUT / "paired_rmse_z.csv")
    pairs = pairs[pairs["valid"] == True].copy()
    if pairs.empty:
        return
    FIG.mkdir(exist_ok=True)
    arms = ["bp6", "bg6", "pg3", "allg"]
    main = pairs[pairs.arm.isin(arms)]
    rates = [1.0, .5, .25, .1]
    colors = ["#1670aa", "#d96827", "#53944d", "#9467bd"]

    # 1: paired absolute RMSE for principal contrasts.
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    for ax, arm in zip(axes, arms):
        sub = main[main.arm == arm]
        for rate, rate_color in zip(rates, colors):
            rr = sub[sub.y_rate == rate]
            if rr.empty:
                continue
            ax.scatter(rr.rmse_correct, rr.rmse_wrong, alpha=.32, s=14,
                       color=rate_color, label=f"{rate:g}" if ax is axes[0] else None)
        ax.plot([0, 1.5], [0, 1.5], "--", color="gray", linewidth=1)
        ax.set(title=ARM_LABELS[arm], xlabel="Correct mode RMSE", xlim=(0, 1.5),
               ylim=(0, 1.5))
    axes[0].set_ylabel("Wrong mode RMSE")
    axes[0].legend(title="Y fraction", fontsize=8)
    save("fig1_paired_absolute_rmse")

    # 2: distribution by wrong family and Y rate.
    fig, ax = plt.subplots(figsize=(11, 5))
    positions, arrays, labels = [], [], []
    for ai, arm in enumerate(arms):
        for ri, rate in enumerate(rates):
            values = main[(main.arm == arm) & (main.y_rate == rate)].delta_rmse.values
            if len(values):
                positions.append(ai * 5 + ri)
                arrays.append(values)
                labels.append(f"{ARM_LABELS[arm]}\n{rate:g}")
    if arrays:
        ax.boxplot(arrays, positions=positions, widths=.65, showfliers=False)
        for x, vals in zip(positions, arrays):
            ax.scatter(np.full(len(vals), x) + np.linspace(-.18, .18, len(vals)),
                       vals, s=7, alpha=.25)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(positions, labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Paired ΔRMSE (wrong − correct)")
    save("fig2_wrong_family_delta")

    # 3: number of wrong columns, seed-averaged per dataset.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, prefix, title in zip(axes, ("bp", "bg", "pg"),
                                 ("Bernoulli→Poisson", "Bernoulli→Gaussian",
                                  "Poisson→Gaussian")):
        sub = pairs[pairs.arm.str.startswith(prefix) & pairs.y_rate.isin((1., .1))]
        by = sub.groupby(["replicate", "y_rate", "n_wrong_columns"],
                         as_index=False).delta_rmse.mean()
        for rate, color in ((1., "#1670aa"), (.1, "#d96827")):
            ss = by[by.y_rate == rate]
            for _, group in ss.groupby("replicate"):
                if len(group):
                    ax.plot(np.r_[0, group.n_wrong_columns], np.r_[0, group.delta_rmse],
                            color=color, alpha=.12)
            avg = ss.groupby("n_wrong_columns").delta_rmse.mean()
            if len(avg):
                ax.plot(np.r_[0, avg.index], np.r_[0, avg.values], "o-",
                        color=color, label=f"Y={rate:g}")
        ax.set(title=title, xlabel="Wrong columns")
        ax.axhline(0, color="gray", linewidth=.8)
    axes[0].set_ylabel("Paired ΔRMSE")
    axes[0].legend()
    save("fig3_wrong_columns_delta")

    # 4: Y training fraction vs paired effect, dataset-level variation.
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm, color in zip(arms, colors):
        sub = main[main.arm == arm]
        by = sub.groupby(["replicate", "y_rate"], as_index=False).delta_rmse.mean()
        avg = by.groupby("y_rate").delta_rmse.agg(["mean", "std", "count"]).reindex(rates)
        ax.plot(avg.index, avg["mean"], "o-", label=ARM_LABELS[arm], color=color)
        ax.scatter(by.y_rate, by.delta_rmse, alpha=.13, s=8, color=color)
    ax.axhline(0, color="gray", linewidth=.8)
    ax.set(xlabel="Y training fraction of candidate pool", ylabel="Paired ΔRMSE")
    ax.set_xscale("log")
    ax.set_xticks(rates, [str(x) for x in rates])
    ax.legend(fontsize=8)
    save("fig4_y_fraction_delta")

    # 5: absolute RMSE by Y fraction, same data pairs.
    fig, ax = plt.subplots(figsize=(9, 5))
    for arm, color in zip(arms, colors):
        sub = main[main.arm == arm]
        avg = sub.groupby("y_rate")[["rmse_correct", "rmse_wrong"]].mean()
        ax.plot(avg.index, avg.rmse_wrong, "o-", label=ARM_LABELS[arm], color=color)
    base = main[main.arm == "bp6"].groupby("y_rate").rmse_correct.mean()
    ax.plot(base.index, base.values, "ko--", label="Correct")
    ax.set(xlabel="Y training fraction", ylabel="Absolute mode RMSE")
    ax.set_xscale("log")
    ax.set_xticks(rates, [str(x) for x in rates])
    ax.legend(fontsize=8)
    save("fig5_y_fraction_absolute")

    # 6: each dataset's paired effects, with two model seeds visible.
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for ax, arm in zip(axes.ravel(), arms):
        sub = main[main.arm == arm]
        for rate, color in ((1., "#1670aa"), (.1, "#d96827")):
            ss = sub[sub.y_rate == rate]
            for series, marker in (("A", "o"), ("B", "x")):
                sss = ss[ss.seed_series == series]
                ax.scatter(sss.replicate.str[3:].astype(int), sss.delta_rmse,
                           color=color, marker=marker, alpha=.65, s=20,
                           label=f"Y={rate:g}, seed={series}")
        ax.axhline(0, color="gray", linewidth=.8)
        ax.set(title=ARM_LABELS[arm], ylabel="Paired ΔRMSE")
    axes[0, 0].legend(fontsize=7, ncol=2)
    for ax in axes[1]:
        ax.set_xlabel("Synthetic dataset number")
    save("fig6_dataset_seed_variation")


def report():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    state = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(OUT / "summary_by_condition.csv")
    fit = pd.read_csv(OUT / "fit_results.csv")
    duplicate_incident = pd.read_csv(OUT / "duplicate_incident.csv")
    paired = pd.read_csv(OUT / "paired_rmse_z.csv")
    paired = paired[paired.valid == True]
    interactions = pd.read_csv(OUT / "interaction_by_dataset.csv")
    seed_summary = pd.read_csv(OUT / "summary_by_seed.csv")
    dataset_mean = pd.read_csv(OUT / "dataset_mean_delta.csv")
    main = summary[summary.condition_id.str.startswith(
        tuple(f"{a}@" for a in ("bp6", "bg6", "pg3", "allg")))]
    table = ["| Condition | Y fraction | Valid pairs | Correct RMSE | Wrong RMSE | Mean Δ | Median Δ | Positive / negative |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in main.sort_values("condition_id").iterrows():
        rate = str(r.condition_id).split("@")[1]
        def fmt(v): return f"{v:.3f}" if pd.notna(v) else "—"
        table.append(f"| {r.condition_id.split('@')[0]} | {rate} | {int(r.valid_pairs)} | "
                     f"{fmt(r.rmse_correct_mean)} | {fmt(r.rmse_wrong_mean)} | "
                     f"{fmt(r.delta_mean)} | {fmt(r.delta_median)} | "
                     f"{int(r.delta_positive)} / {int(r.delta_negative)} |")
    interaction_lines = ["| Arm | Complete seed-pairs | Mean (Δ10% − Δ100%) | Positive / negative |",
                         "|---|---:|---:|---:|"]
    for arm in ("bp6", "bg6", "pg3", "allg"):
        sub = interactions[interactions.arm == arm]
        if len(sub):
            d = sub.sparse_minus_dense
            interaction_lines.append(
                f"| {arm} | {len(d)} | {d.mean():+.3f} | {(d>0).sum()} / {(d<0).sum()} |")
        else:
            interaction_lines.append(f"| {arm} | 0 | — | 0 / 0 |")
    seed_lines = ["| Arm | Y fraction | Seed A mean Δ | Seed B mean Δ |",
                  "|---|---:|---:|---:|"]
    dataset_lines = ["| Arm | Y fraction | Complete datasets (2 seeds) | Mean Δ | Positive / negative datasets |",
                     "|---|---:|---:|---:|---:|"]
    for arm in ("bp6", "bg6", "pg3", "allg"):
        for rate in (1., .5, .25, .1):
            c = f"{arm}@{rate:g}"
            rows = seed_summary[seed_summary.condition_id == c]
            def seed_value(series):
                values = rows.loc[rows.seed_series == series, "mean_delta"]
                return f"{values.iloc[0]:+.3f}" if len(values) and pd.notna(values.iloc[0]) else "—"
            seed_lines.append(f"| {arm} | {rate:g} | {seed_value('A')} | {seed_value('B')} |")
            ds = dataset_mean[(dataset_mean.condition_id == c) &
                              (dataset_mean.valid_seed_count == 2)]
            if len(ds):
                v = ds.mean_delta_two_seeds
                dataset_lines.append(f"| {arm} | {rate:g} | {len(ds)} | {v.mean():+.3f} | "
                                     f"{(v>0).sum()} / {(v<0).sum()} |")
            else:
                dataset_lines.append(f"| {arm} | {rate:g} | 0 | — | 0 / 0 |")
    def mean_delta(arm, rate):
        subset = summary[summary.condition_id == f"{arm}@{rate:g}"]
        if len(subset) and pd.notna(subset.iloc[0].delta_mean):
            return float(subset.iloc[0].delta_mean)
        return None
    family_observations = []
    for rate in (1., .5, .25, .1):
        values = []
        for arm in ("bp6", "bg6", "pg3", "allg"):
            subset = summary[summary.condition_id == f"{arm}@{rate:g}"]
            value = mean_delta(arm, rate)
            if value is not None and len(subset) and pd.notna(subset.iloc[0].rmse_correct_mean):
                baseline = float(subset.iloc[0].rmse_correct_mean)
                values.append((arm, value, 100 * value / baseline))
        if values:
            family_observations.append(
                f"At Y={rate:g}, observed mean paired Δ by arm was " +
                ", ".join(f"{arm} {value:+.3f} ({percent:+.1f}% of paired correct RMSE)"
                          for arm, value, percent in values) + ".")
    count_observations = []
    for prefix, counts in (("bp", (2, 4, 6)), ("bg", (2, 4, 6)),
                           ("pg", (1, 2, 3))):
        for rate in (1., .1):
            values = [(count, mean_delta(f"{prefix}{count}", rate)) for count in counts]
            values = [(count, value) for count, value in values if value is not None]
            if values:
                count_observations.append(
                    f"{prefix} at Y={rate:g}: " +
                    ", ".join(f"{count} columns {value:+.3f}" for count, value in values) +
                    " (0 columns is Δ=0 by definition).")
    failures = fit[fit.status != "OK"]
    new_runtime = fit.loc[fit.source == "MX2", "runtime_s"].dropna()
    runtime_line = (f"New fit runtime: median {new_runtime.median():.2f} s, "
                    f"sum {new_runtime.sum():.1f} worker-seconds across "
                    f"{len(new_runtime)} recorded new fits."
                    if len(new_runtime) else "No new-fit runtime recorded.")
    failure_lines = (["Technical non-OK fit cells (retained, without retry):"] +
                     [f"- `{r.experiment_id}`: {r.status}, mode={r.mode_status}, "
                      f"gradient={r.mode_grad_inf}, min Hessian eigenvalue={r.mode_min_eig_H}; "
                      f"secondary-start status={r.mode_from_mean5_status}, "
                      f"two-start max difference={r.modes_max_abs_diff}; "
                      f"error={r.error if pd.notna(r.error) else 'none'}"
                      for _, r in failures.iterrows()])
    if failures.empty:
        failure_lines = ["No technical non-OK fit was observed in the recorded cells."]
    lines = [
        "# Extended X-family misspecification: result record (2026-11-10)",
        "",
        "## 1. Research objective",
        "Assess how fixed X-family misspecification changes recovery of latent Z across wrong-family type, wrong-column count, and available Y training pairs. A positive Δ is worse under the wrong family; no sign is a technical success criterion.",
        "",
        "## 2. Conditions and provenance",
        f"Lineage: {protocol['lineage']}. Data: n=75, d=12, K=3, G3/B6/P3 X, Bernoulli Y, 20 draws, two model seeds, L=5, eight MCEM iterations. The 10% and 100% Y masks reproduce M1, and the 25%/50% sets extend the 10% subset. The 278 held-out pairs are shared. See `protocol.json`, `mask_provenance.csv`, and the pre-experiment audit. Neither family nor K is selected.",
        "Arm IDs: correct=G3/B6/P3; bp6=B6→P6; bg6=B6→G6; pg3=P3→G3; allg=all 12 columns Gaussian. bp2/bp4, bg2/bg4, and pg1/pg2 are the intermediate column-count conditions. All fits use the same raw observations; the Gaussian arms treat integer observations as continuous values without converting them.",
        f"Attempted {state['attempted']}/{state['planned']} unique fit cells; successful {state['successful']}, failed/non-OK {state['failed']}; remaining {state['remaining']}. Of the attempted cells, 160 came from E1 primary CSVs and carry source hashes in protocol. Completion status: **{'complete' if state['completed'] else 'incomplete'}**.",
        "",
        "## 3. Correct versus wrong absolute RMSE and paired change",
        "Primary endpoint: joint conditional posterior mode at estimated θ, rotation-only Procrustes RMSE. Both arm fits must be technically valid. The table reports completed pairs only and is descriptive; cells with fewer than 40 pairs have incomplete coverage.",
        "",
        *table,
        "",
        "OBSERVED: In the correct arm, mean mode RMSE was 0.347 at Y=100% and 0.717 at Y=10%. For Bernoulli→Poisson (6 columns) it was 0.359 and 0.799, respectively; for all-Gaussian it was 0.669 and 0.893. Thus the absolute recovery problem became harder when Y pairs were removed in every displayed arm, even when the paired misspecification penalty became smaller.",
        "",
        "## 4. Wrong-family differences",
        "The same data, Y mask and model seed are paired within each comparison. Compare the signed Δ by arm and Y fraction in the table and Figure 2. The model seed controls both initialization and Monte Carlo draws, so differences cannot be assigned to one source of randomness.",
        *family_observations,
        "",
        *seed_lines,
        "",
        "The following rows average the two model seeds within each synthetic dataset and include only datasets with both valid seeds:",
        "",
        *dataset_lines,
        "",
        "OBSERVED: Bernoulli→Gaussian and all-Gaussian had much larger mean penalties than Bernoulli→Poisson at every Y fraction. Poisson→Gaussian was intermediate at Y=100% and close to Bernoulli→Gaussian at Y=10%. These are comparisons within the same E-consistent prototype series and fixed generator; they are not a ranking of families in general.",
        "",
        "## 5. Number of wrong columns",
        "Nested Bernoulli column prefixes define 2/4/6 wrong columns and Poisson prefixes define 1/2/3, separately for each data draw. Correct (0-column) results are shared. The count comparisons are recorded in `paired_rmse_z.csv` and Figure 3. A single deterministic column order per draw limits generality.",
        *count_observations,
        "OBSERVED: At Y=10%, the mean penalty increased across the registered column-count steps for all three error types. At Y=100%, Bernoulli→Poisson at two columns had a small negative mean Δ (−0.002), so an unconditional monotonicity claim fails even in these means. Individual dataset trajectories are not all monotone (Figure 3).",
        "",
        "## 6. Y information",
        "The manipulation changes the number of observed training pairs, not the edge density of generated Y. Figures 4 and 5 show paired Δ and both absolute RMSEs. Comparing the two is necessary because a larger Δ at sparse Y can coexist with a larger correct-arm error.",
        "",
        *interaction_lines,
        "OBSERVED: The 10%-minus-100% paired penalty was +0.069 for Bernoulli→Poisson, +0.041 for Poisson→Gaussian, −0.125 for Bernoulli→Gaussian, and −0.147 for all-Gaussian (means across 40 matched seed-pairs per arm). Poisson→Gaussian was nonmonotone across the four Y rates, with the largest mean Δ at 50% (+0.209). Therefore the direction of the Y-information interaction depends on the wrong-family type in this setup.",
        "The shrinking Gaussian-misspecification penalty at sparse Y does **not** mean that its wrong-arm Z estimates improved: all-Gaussian absolute RMSE increased from 0.669 to 0.893 as Y went from 100% to 10%. The correct-arm RMSE increased more, from 0.347 to 0.717. The difference of these changes accounts arithmetically for the smaller paired Δ; its underlying statistical cause remains untested.",
        "",
        "## 7. Interpretation",
        "DERIVED (RESEARCH_MASTER.md §6): for an attribute column, the conditional Z score contains `f_l (x_il − μ_l)/φ_l`, while its negative curvature contains `A_l''(η) f_l f_lᵀ/φ_l`. Bernoulli uses μ=sigmoid(η) and A''=μ(1−μ); Poisson uses μ=A''=exp(η); Gaussian uses μ=η and constant curvature scaled by its fitted dispersion. Changing a family's assignment therefore changes both the direction and strength of the X contribution. Y contributes a sum over observed training partners; reducing the Y mask removes terms from this sum, so X may contribute relatively more. This mechanism is plausible but is **not established as the causal explanation** by these descriptive fits. No raw X log likelihood across discrete and continuous families is compared.",
        "",
        "## 8. Limits and future work",
        "The point estimate is a local, fixed-θ joint mode, not an established universally optimal estimator. The final sample and mean-five RMSE remain in `fit_results.csv`. Two model seeds do not isolate initialization from Monte Carlo variability. The generator, K, Y law, and column-order rule are fixed. Any incomplete cells stay visible in manifest; no failed fit was replaced. A data-level paired uncertainty calculation would have to cluster both model seeds within each synthetic dataset.",
        "The inherited E-step uses a conditional Laplace Gaussian sample after a Newton update (`reproduction/src/model.py:360`); its sampling fallback can retain the mode if covariance sampling fails. That fallback is not counted as a fit failure by the runner, and this experiment did not instrument its frequency. The joint mode evaluation uses a separate full Hessian check and does not certify global optimality. E1's reused 160 rows have no held-out AP field, which remains blank rather than inferred.",
        "Priorities for a human decision after reviewing these results: (1) repeat selected column-count contrasts under additional preregistered column orders if count effects depend on which columns were changed; (2) split initialization and MC seeds to quantify their contributions to run variation; (3) test another generator or Y law before making a broader claim. Each is a new experiment requiring the Human Gate in AGENTS.md §6.",
        "",
        "## 9. Position in the master's research",
        "This experiment characterizes when using the correct X observation family may help Z recovery in one mixed-family synthetic setup. It is an E-consistent per-column **prototype** and cannot be promoted to the adopted manuscript method without the Human Gate in AGENTS.md §6. Automatic family selection is outside this experiment.",
        "",
        "## Technical record",
        *failure_lines,
        runtime_line,
        f"Execution incident: {len(duplicate_incident)} extra completed duplicate runs were excluded by the documented first-record rule; the untouched raw CSV and row differences are retained. A process interruption may also have stopped an unrecorded in-flight run. The planned/attempted counts above are unique fit cells, not a count of every physical process attempt.",
        f"Protocol source HEAD: `{protocol['source_head']}`. Current branch: `{mx.git('branch', '--show-current')}`. Analysis code: `expfam/src/experimental/analyze_misspecification_extended.py`. Environment: Python {sys.version.split()[0]}, NumPy {np.__version__}, SciPy {scipy.__version__}, scikit-learn {sklearn.__version__}, pandas {pd.__version__}, matplotlib {matplotlib.__version__}. Reused E1 file SHA-256 values are in protocol. Completion and technical counts are recomputed from the fit CSV.",
        "Reproduction from a fresh checkout with the same dependencies and historical E1 artifacts: run `python expfam/src/experimental/run_misspecification_extended.py --prepare`, then `--pilot`, then `--execute --workers 4`; run `python expfam/src/experimental/analyze_misspecification_extended.py` and `python expfam/src/experimental/verify_misspecification_extended.py`. The fixed output directory must not already exist at preparation time. The runner resumes from unique fit keys after interruption.",
        "",
    ]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report()
    else:
        mx.analyze()
        maybe_plot()
        report()
