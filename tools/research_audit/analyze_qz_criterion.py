"""Aggregate the Q_Z / conditional-criterion re-scoring and draw the figures.

Reads expfam/results/k_selection/qz_criterion_existing_states_20261106/
{criterion_scores, full_k_selection, local_selection_saved_states}.csv
(produced by qz_criterion_existing_states.py) and the PR #156 checkpoint
rows (K = 4, 5 only: local comparisons).  Writes the remaining CSVs, figures
and a manifest.  Refuses to overwrite.  No fit, no sampling.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "expfam/results/k_selection/qz_criterion_existing_states_20261106"
FIG = OUT / "figures"
KSEL = ROOT / "expfam/results/k_selection"
C2 = 1.0 + math.log(2.0 * math.pi)
CRITS = ["A_CQ", "B_CQ_minus_PZ", "C_cond_last", "D_cond_mode", "E_CLap"]
LABEL = {"A_CQ": "A  C_Q", "B_CQ_minus_PZ": "B  C_Q - P_Z (Q1, MC mean)", "C_cond_last": "C  Q1 plug-in, last sample",
         "D_cond_mode": "D  Q1 plug-in, joint mode", "E_CLap": "E  C_Lap"}
COLOR = {"A_CQ": "#2a6fdb", "B_CQ_minus_PZ": "#e8822a", "C_cond_last": "#3a9e5b", "D_cond_mode": "#c0392b",
         "E_CLap": "#7d5ba6"}


def need(c, m):
    if not c:
        raise SystemExit("STOP: " + m)


def write(name, df):
    p = OUT / name
    need(not p.exists(), f"refusing to overwrite {p}")
    df.to_csv(p, index=False, lineterminator="\n")


def group_label(r):
    return f"{r.series} | cond {r.condition} | K_GEN={r.k_gen} | n={r.n} | d={r.d}"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    sc = pd.read_csv(OUT / "criterion_scores.csv")
    full = pd.read_csv(OUT / "full_k_selection.csv")
    loc = pd.read_csv(OUT / "local_selection_saved_states.csv")

    # ---- criterion definitions -------------------------------------------------
    defs = pd.DataFrame([
        ("A_CQ", "-2(Q_X+Q_Y+Q_Z) + p_K ln n", "Q2 MC mean (L=5 final E-step samples)", "yes", "k_criterion_panel.cq_terms", "C_Q; Q-based complete-data / ICL-type; not Schwarz BIC"),
        ("B_CQ_minus_PZ", "-2(Q_X+Q_Y) + p_K ln n = C_cond_MC", "Q1 MC mean over the same samples", "no", "C_Q - P_Z (identity checked)", "Eq.(16)-type at the MC mean; not the paper's BIC"),
        ("C_cond_last", "-2 ln p(X,Y_train|Z_last,theta) + p_K ln n", "Q1 plug-in", "no", "k_criterion_panel.data_loglik(Z_samples[:,:,-1])", "Eq.(16)-type at the last E-step sample"),
        ("D_cond_mode", "-2 ln p(X,Y_train|Z_hat,theta) + p_K ln n", "Q1 plug-in", "no", "lap_fit_mode + P_theta (Z_hat = joint mode of Q2)", "Eq.(16)-type at the Laplace joint mode"),
        ("E_CLap", "-2[Q2(Z_hat) + nK/2 ln 2pi - 1/2 ln|H|] + p_K ln n", "Laplace approximation of Q3", "integrated", "laplace_k_criterion.calc_C_Lap", "not an exact marginal likelihood"),
        ("penalty Eq.(26)", "((k+1)d - k(k-1)/2) ln n", "-", "-", "paper p.343", "differs from kd - k(k-1)/2 by the constant d ln n: no effect on K_hat or slack (Poisson-X); identical for Gaussian-X"),
    ], columns=["criterion", "formula", "evaluates", "includes_ln_p_Z", "implementation", "note"])
    write("criterion_definitions.csv", defs)

    # ---- Q_Z identity ------------------------------------------------------------
    sc["PZ_minus_formula"] = sc.P_Z - sc.P_Z_formula
    qz = (sc.groupby(["series", "n", "k_est"]).agg(fits=("P_Z", "size"), P_Z_min=("P_Z", "min"), P_Z_max=("P_Z", "max"),
                                                    formula=("P_Z_formula", "first"),
                                                    max_abs_diff=("PZ_minus_formula", lambda v: float(np.abs(v).max())))
          .reset_index())
    qz["per_dimension_increment_2.8379n"] = qz.n * C2
    shift = loc[loc.criterion.isin(["A_CQ", "B_CQ_minus_PZ"])].pivot_table(
        index=["series", "condition", "k_gen", "n", "d", "replicate", "k_from", "k_to"], columns="criterion", values="slack").dropna()
    shift["shift"] = shift.B_CQ_minus_PZ - shift.A_CQ
    shift = shift.reset_index()
    shift["predicted"] = shift.n * C2 * (shift.k_to - shift.k_from)
    qz_shift = shift.groupby(["series", "n"]).apply(lambda g: pd.Series({
        "local_steps": len(g), "max_abs_shift_minus_2.8379n": float((g["shift"] - g.predicted).abs().max())})).reset_index()
    qz = qz.merge(qz_shift, on=["series", "n"], how="left")
    write("qz_penalty_identity.csv", qz)

    # ---- MC-closeness of the full-K choice (A/B: MC se of the L-sample mean; C: one-sample sd) --
    sc["se_A"] = sc.mcse_B
    sc["se_C"] = sc.sd_single_sample_C
    keys = ["series", "condition", "k_gen", "n", "d", "replicate"]
    def _se(v):
        v = np.asarray(v, float)
        return float("nan") if np.isnan(v).any() else 0.5 * math.sqrt(float((v ** 2).sum()))

    se_k = sc.groupby(keys + ["k_est"]).agg(seA=("se_A", _se), seC=("se_C", _se)).reset_index()
    rows = []
    for _, r in full.iterrows():
        if r.label == "INCOMPLETE" or r.criterion not in ("A_CQ", "B_CQ_minus_PZ", "C_cond_last"):
            rows.append(np.nan)
            continue
        g = sc[(sc.series == r.series) & (sc.condition == r.condition) & (sc.k_gen == r.k_gen) & (sc.n == r.n)
               & (sc.d == r.d) & (sc.replicate == r.replicate)]
        m = g.groupby("k_est")[r.criterion].mean().sort_values()
        kb, k2 = m.index[0], m.index[1]
        s = se_k[(se_k.series == r.series) & (se_k.condition == r.condition) & (se_k.k_gen == r.k_gen) & (se_k.n == r.n)
                 & (se_k.d == r.d) & (se_k.replicate == r.replicate)].set_index("k_est")
        col = "seC" if r.criterion == "C_cond_last" else "seA"
        a_, b_ = s.loc[kb, col], s.loc[k2, col]
        se = math.sqrt(a_ ** 2 + b_ ** 2) if pd.notna(a_) and pd.notna(b_) else np.nan
        rows.append(se)
    full["gap_mc_se_approx"] = rows
    full["mc_close"] = (full.best_minus_second < 2 * full.gap_mc_se_approx).astype(object)
    full.loc[full.gap_mc_se_approx.isna(), "mc_close"] = np.nan
    write("full_k_selection_with_mc.csv", full)

    # ---- condition effect --------------------------------------------------------
    eff = []
    for gk, g in full.groupby(keys[:-1] + ["criterion"]):
        lab = g.label.value_counts()
        kh = g.k_hat.dropna().astype(int)
        lg = loc[(loc.series == gk[0]) & (loc.condition == gk[1]) & (loc.k_gen == gk[2]) & (loc.n == gk[3])
                 & (loc.d == gk[4]) & (loc.criterion == gk[5])]
        under = lg[(lg.k_from == gk[2] - 1) & (lg.k_to == gk[2])].slack
        over = lg[(lg.k_from == gk[2]) & (lg.k_to == gk[2] + 1)].slack
        eff.append(dict(zip(keys[:-1] + ["criterion"], gk), datasets=len(g), exact=int(lab.get("exact", 0)),
                        under=int(lab.get("under", 0)), over=int(lab.get("over", 0)),
                        incomplete=int(lab.get("INCOMPLETE", 0)),
                        k_hat_values=" ".join(map(str, sorted(kh.tolist()))),
                        k_hat_at_max_candidate=int((kh == g.candidate_k.str.split("..", regex=False).str[1].astype(int).iloc[0]).sum()),
                        median_slack_under=float(under.median()) if len(under) else np.nan,
                        median_slack_over=float(over.median()) if len(over) else np.nan,
                        mc_close=(int(g.mc_close.dropna().astype(bool).sum()) if g.mc_close.notna().any() else np.nan)))
    eff = pd.DataFrame(eff)
    write("criterion_effect_by_condition.csv", eff)

    # ---- local comparisons: saved states + PR #156 checkpoints (K = 4, 5 only) -----
    pr156 = []
    for part in ("rq1_AB", "rq1_CD"):
        cr = pd.read_csv(KSEL / f"mcem_stability_signal_matched_20261106/{part}/checkpoint_rows.csv")
        cr = cr[cr.status == "FIT"]  # PR #156 marks every saved checkpoint row "FIT"
        for (c, rep, cp), g in cr.groupby(["condition", "replicate", "checkpoint"]):
            vals = {}
            for crit, f in (("A_CQ", lambda d: d.C_Q), ("B_CQ_minus_PZ", lambda d: d.C_Q - d.P_Z),
                            ("D_cond_mode", lambda d: d.lap_fit_mode + d.P_theta), ("E_CLap", lambda d: d.C_Lap)):
                okk = not ((crit in ("D_cond_mode", "E_CLap")) and (g.lap_status != "OK").any())
                m = g.assign(v=f(g)).groupby("k_fit").v.mean()
                vals[crit] = (float(m[4] - m[5]) if okk and 4 in m and 5 in m and len(g) == 4 else np.nan)
            for crit, v in vals.items():
                pr156.append(dict(series=f"PR #156 RQ1 (64 iterations, checkpoints)", condition=c, k_gen=5, n=int(g.n.iloc[0]),
                                  d=15, replicate=rep, checkpoint=cp, criterion=crit, k_from=4, k_to=5, slack=v,
                                  slack_per_node=v / int(g.n.iloc[0]) if pd.notna(v) else np.nan,
                                  status="OK" if pd.notna(v) else "INCOMPLETE (Laplace not OK or missing)",
                                  note="K = 4, 5 only: LOCAL comparison, no K_hat. C_cond_last not computed (states not re-scored)"))
    cp = loc.series.str.extract(r"it=(\d+)")[0].astype(float).fillna(8.0)
    cp[loc.series.str.startswith("paper-type")] = 8.0   # #139: 8 iterations
    loc2 = pd.concat([loc.assign(checkpoint=cp), pd.DataFrame(pr156)], ignore_index=True)
    write("local_selection_comparisons.csv", loc2)

    # ---- available saved states ----------------------------------------------------
    av = [
        ("clap_clean_nsweep_20261006/full/states.npz + EMC checkpoint-8 npz", "canonical A1 n-sweep", "1,3,5", "50,75,100,150", 15, "1..7", "Zsamples(L=5), Z_last, F, scalars, Zhat", "USED (A-E, full K)"),
        ("historical_canonical_causal_20261106/fit_states_n75.npz, _n150.npz", "PR #155 signal strength B/C/D (n=75), B (n=150)", "5", "75,150", 15, "1..7", "Zsamples, Z_last, F, sigma, scalars (no Zhat; D from stored lap_fit_mode)", "USED (A-E, full K)"),
        ("dsweep_k_selection_20261105/fit_states.npz", "d-sweep", "5", "75", "30,60", "1..7", "Zsamples, Z_last, F, sigma, scalars", "USED (A-E, full K)"),
        ("k_selection_robustness_20261105/r2_n{50,100,150}_fit_states.npz", "R2", "5", "50,100,150", 30, "1..7", "same", "USED (A-E, full K)"),
        ("k_selection_robustness_20261105/r1_fit_states.npz", "R1 (L=10, checkpoints 8/16)", "5", "75", "15,30,60", "1..7", "same", "USED (A-E, full K)"),
        ("paper_condition_k_criteria_20261006/full/states.npz", "#139 paper-type (Gaussian-X)", "1,3,5,7,9", "150", 15, "1..10", "Zsamples, F, sigma, Zhat; criteria stored by the same panel code", "USED via stored panel CSV (A-E, full K; not re-scored)"),
        ("mcem_stability_signal_matched_20261106/rq1_*_states.npz", "PR #156 RQ1 (64 iterations)", "5", "75", 15, "4,5", "Zsamples, Z, F, sigma, scalars at checkpoints", "LOCAL only (A, B, D, E from stored checkpoint rows)"),
        ("mcem_stability_signal_matched_20261106/rq2_states.npz", "PR #156 RQ2 (K_GEN 3 vs 5)", "3,5", "75", 15, "K_GEN +/- 1", "same", "NOT USED (only K_GEN +/- 1; different generator)"),
        ("mcem_noise_separation_20261106/exec_v2", "PR #157 fixed-theta probes", "5", "75", 15, "4,5", "probe arrays", "NOT USED (local, probe states)"),
        ("em_convergence_sensitivity_20260905/checkpoint_states.npz", "EM convergence sensitivity", "?", "?", "?", "?", "Zsamples, Z, F, sigma, scalars", "NOT USED (not mapped in this study)"),
        ("z_point_estimator_e1 / seed_separation_e2a / em_iterations_e2c", "Z point-estimator studies", "-", "-", "-", "not a K sweep", "Zsamples, Zmode, mean5", "NOT USED (no K sweep)"),
    ]
    av = pd.DataFrame(av, columns=["path_under_kennkyu_artifacts", "study", "k_gen", "n", "d", "candidate_k", "saved_latent_states", "use"])
    cnt = sc.groupby("series").size().rename("fits_scored").reset_index()
    write("available_saved_states.csv", av)
    write("fits_scored_by_series.csv", cnt)

    # ---- incomplete and limitations ------------------------------------------------
    inc = full[full.label == "INCOMPLETE"].groupby(["series", "criterion"]).size().rename("datasets").reset_index()
    inc["reason"] = "Laplace mode not OK at some (K, start); not imputed"
    lim = pd.DataFrame([
        ("all", "-", "B, C, D are Eq.(16)-type; none is the paper's BIC (evaluation point, aggregation, generator and normalisation of the paper are unknown)"),
        ("D_cond_mode", "-", "Z_hat is the joint (Q2) mode, not the maximiser of Q1; for PR #155, d-sweep, R1/R2 D comes from stored lap_fit_mode (Z_hat not saved)"),
        ("C_cond_last", "-", "one E-step sample: MC error is the single-sample sd (not reduced by L)"),
        ("all", "-", "states are from short MCEM (8 or 16 iterations; PR #156: up to 64) with an approximate Gibbs E-step"),
        ("PR #156", "-", "K = 4, 5 only: local comparison, no K_hat"),
        ("paper-type", "-", "values taken from the #139 panel CSV (same code), not re-scored from states"),
    ], columns=["criterion", "datasets", "reason"])
    write("incomplete_and_limitations.csv", pd.concat([inc.rename(columns={"series": "scope"}), lim.rename(columns={"criterion": "scope"}).assign(criterion=lim.criterion)], ignore_index=True))

    # ---- figures -------------------------------------------------------------------
    def curves(ax, sub, kgen, title):
        for c in CRITS:
            g = sub.groupby(["replicate", "k_est"])[c].mean().unstack()
            g = g.sub(g.min(axis=1), axis=0)
            med = g.median()
            ax.plot(med.index, med.values, "-o", ms=4, lw=2, color=COLOR[c], label=LABEL[c])
        ax.axvline(kgen, color="0.4", ls=":", lw=1)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("candidate K")
        ax.grid(alpha=0.3)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    a1 = sc[(sc.series == "canonical A1 n-sweep") & (sc.k_gen == 5) & (sc.n == 75)]
    a2 = sc[(sc.series == "canonical A1 n-sweep") & (sc.k_gen == 5) & (sc.n == 150)]
    pt = sc[(sc.series == "paper-type #139 (Gaussian-X)") & (sc.k_gen == 5)]
    curves(axs[0], a1, 5, "canonical A, K_GEN=5, n=75 (8 datasets)")
    curves(axs[1], a2, 5, "canonical A, K_GEN=5, n=150 (8 datasets)")
    curves(axs[2], pt, 5, "paper-type #139, K*=5, n=150 (5 datasets)")
    axs[0].set_ylabel("C(K) - min_K C(K), median over datasets (symlog)")
    axs[0].legend(fontsize=8, loc="upper center")
    fig.suptitle("Selection-score curves on the same saved states (lower is better; dotted = K_GEN). Fig.1", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_criterion_curves.png", dpi=150)
    fig.savefig(FIG / "fig1_criterion_curves.pdf")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    l5 = loc[(loc.series == "canonical A1 n-sweep") & (loc.k_gen == 5)]
    for ax, (a, b) in zip(axs, ((4, 5), (5, 6))):
        for i, c in enumerate(CRITS):
            for j, n in enumerate((50, 75, 100, 150)):
                v = l5[(l5.criterion == c) & (l5.n == n) & (l5.k_from == a) & (l5.k_to == b)].slack.dropna()
                x = j + (i - 2) * 0.16
                ax.scatter(np.full(len(v), x), v, s=14, color=COLOR[c], alpha=0.8, label=LABEL[c] if j == 0 else None)
        ax.axhline(0, color="0.3", lw=1)
        ax.set_xticks(range(4), [f"n={n}" for n in (50, 75, 100, 150)])
        ax.set_title(f"slack C({a}) - C({b}) per dataset (> 0 favours K={b})", fontsize=10)
        ax.set_yscale("symlog", linthresh=50)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.suptitle("Q_Z removal shifts every 4->5 slack by exactly 2.8379 n; canonical A1, K_GEN=5. Fig.2", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_qz_slack_shift.png", dpi=150)
    fig.savefig(FIG / "fig2_qz_slack_shift.pdf")
    plt.close(fig)

    sel_groups = [("canonical A1 n-sweep", "A", 5, 75, 15), ("canonical A1 n-sweep", "A", 5, 150, 15),
                  ("canonical signal-strength (PR #155)", "D", 5, 75, 15), ("canonical d-sweep / R2 (d=60)", "A", 5, 75, 60),
                  ("paper-type #139 (Gaussian-X)", "paper-type", 5, 150, 15), ("paper-type #139 (Gaussian-X)", "paper-type", 9, 150, 15)]
    fig, axs = plt.subplots(1, len(sel_groups), figsize=(18, 4.2), sharey=True)
    for ax, (s, c, kg, n, d) in zip(axs, sel_groups):
        e = eff[(eff.series == s) & (eff.condition == c) & (eff.k_gen == kg) & (eff.n == n) & (eff.d == d)].set_index("criterion").loc[CRITS]
        bottom = np.zeros(len(CRITS))
        for lab, col in (("under", "#8fb3e8"), ("exact", "#4a8a4a"), ("over", "#e39a6b"), ("incomplete", "#cccccc")):
            vals = e[lab].values / e.datasets.values
            ax.bar(range(len(CRITS)), vals, bottom=bottom, color=col, edgecolor="white", linewidth=1.5, label=lab)
            bottom += vals
        ax.set_xticks(range(len(CRITS)), ["A", "B", "C", "D", "E"])
        ax.set_title(f"{s.split(' (')[0].replace('canonical ', '')}\ncond {c}, K_GEN={kg}, n={n}, d={d}", fontsize=8)
    axs[0].set_ylabel("share of datasets")
    axs[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Under / exact / over selection by criterion (A C_Q, B C_Q-P_Z, C last-sample plug-in, D mode plug-in, E C_Lap). Fig.3", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_selection_outcome.png", dpi=150)
    fig.savefig(FIG / "fig3_selection_outcome.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in CRITS:
        e = eff[(eff.series == "canonical A1 n-sweep") & (eff.k_gen == 5) & (eff.criterion == c)].sort_values("n")
        ax.plot(e.n, e.exact / (e.datasets - e.incomplete), "-o", color=COLOR[c], lw=2, label=LABEL[c])
    ax.set_xlabel("n")
    ax.set_ylabel("share of complete datasets with K_hat = K_GEN")
    ax.set_title("canonical A1, K_GEN = 5, d = 15: exact selection vs n. Fig.4", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_condition_effect_n.png", dpi=150)
    fig.savefig(FIG / "fig4_condition_effect_n.pdf")
    plt.close(fig)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    man = {"git_head": head, "mcem_fits": 0,
           "files": {str(p.relative_to(OUT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "manifest.json"}}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(eff.to_string())


if __name__ == "__main__":
    main()
