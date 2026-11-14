"""Pre-registered analysis for AF4 / S1 (Laplace whole-model comparison).

Statistical unit: dataset (20 per regime).  Writes ``<stage>/analysis/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
BOOT_N, BOOT_SEED = 10000, 20261114
AF3_COLS = (ROOT / "expfam" / "results" / "auto_family_z_recovery" / "af3_selector_rescoring_20261113"
            / "full" / "column_selection.csv")
PARTS = ["phi_prior", "phi_x_gauss", "phi_x_true_pois_cols", "phi_x_true_cmp_cols", "phi_y_train"]
CONTRASTS = [("oracle", "support_only", True), ("oracle", "wrong", True),
             ("oracle", "auto_first", False), ("oracle", "auto_last", False)]


def boot_ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(BOOT_SEED)
    m = v[rng.integers(0, v.size, size=(BOOT_N, v.size))].mean(axis=1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def af3_z5_misselected() -> dict[str, set]:
    if not AF3_COLS.exists():
        return {}
    c = pd.read_csv(AF3_COLS, keep_default_na=False, na_values=[""])
    c = c[(c.arm == "Z5") & (c.start.astype(str) == "zero") & (c.true_model == "cmp_nu2p0")]
    bad = c[c.correct.astype(str).str.lower() != "true"]
    return {reg: set(bad[bad.regime == reg].replicate) for reg in ("dense", "sparse")}


def paired(mc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    idx = mc.set_index(["replicate", "regime", "method"])
    z5 = af3_z5_misselected()
    for a, b, primary in CONTRASTS:
        for (rep, reg), _ in mc.groupby(["replicate", "regime"]):
            A, B = idx.loc[(rep, reg, a)], idx.loc[(rep, reg, b)]
            ok = A["status"] == "OK" and B["status"] == "OK"
            row = {"contrast": f"{a} - {b}", "primary": primary, "replicate": rep, "regime": reg,
                   "status_a": A["status"], "status_b": B["status"], "both_ok": ok,
                   "same_assignment": A["assignment"] == B["assignment"],
                   "af3_z5_misselected_cmp": rep in z5.get(reg, set()),
                   "b_n_correct_count_columns": int(B["n_correct_count_columns"])}
            if ok:
                row["delta_laplace"] = A["laplace"] - B["laplace"]
                row["delta_phi"] = A["phi_phi"] - B["phi_phi"]
                for p in PARTS:
                    row[f"delta_{p}"] = A[p] - B[p]
                row["delta_neg_half_logdet"] = -0.5 * (A["logdet"] - B["logdet"])
                row["delta_best_of_starts"] = A["laplace_best_of_starts"] - B["laplace_best_of_starts"]
                row["delta_is_corrected"] = A["laplace_is_corrected"] - B["laplace_is_corrected"]
                row["is_readable"] = bool(A["is_relative_ess"] >= 0.01 and B["is_relative_ess"] >= 0.01)
                row["delta_heldout_y_plugin"] = A["heldout_y_plugin_mean_log"] - B["heldout_y_plugin_mean_log"]
                row["delta_heldout_y_pred"] = A["heldout_y_laplace_pred_mean_log"] - B["heldout_y_laplace_pred_mean_log"]
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(p: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (con, reg), g in p.groupby(["contrast", "regime"], sort=False):
        for subset_name, s in (("all", g), ("assignment_differs", g[~g.same_assignment]),
                               ("af3_z5_misselected", g[g.af3_z5_misselected_cmp])):
            if subset_name == "assignment_differs" and con.endswith(("support_only", "wrong")):
                continue
            if subset_name == "af3_z5_misselected" and not con.endswith("support_only"):
                continue
            ok = s[s.both_ok]
            d = ok.delta_laplace
            r = {"contrast": con, "regime": reg, "subset": subset_name, "datasets": len(s), "both_ok": len(ok),
                 "excluded": int((~s.both_ok).sum()), "n_oracle_higher": int((d > 0).sum()),
                 "n_equal": int((d == 0).sum()), "n_oracle_lower": int((d < 0).sum()),
                 "mean_delta": float(d.mean()) if len(d) else np.nan, "median_delta": float(d.median()) if len(d) else np.nan,
                 "ci": boot_ci(d.values), "min_delta": float(d.min()) if len(d) else np.nan,
                 "mean_delta_phi": float(ok.delta_phi.mean()) if len(d) else np.nan,
                 "mean_delta_neg_half_logdet": float(ok.delta_neg_half_logdet.mean()) if len(d) else np.nan}
            for q in ["phi_prior", "phi_x_gauss", "phi_x_true_pois_cols", "phi_x_true_cmp_cols", "phi_y_train"]:
                r[f"mean_delta_{q}"] = float(ok[f"delta_{q}"].mean()) if len(d) else np.nan
            nz = ok[ok.delta_laplace != 0]
            r["R3_sign_differs_from_delta_phi"] = int((np.sign(nz.delta_laplace) != np.sign(nz.delta_phi)).sum())
            r["n_sign_reversed_best_of_starts"] = int((np.sign(nz.delta_best_of_starts) != np.sign(nz.delta_laplace)).sum())
            rd = nz[nz.is_readable]
            r["is_readable_pairs"] = int(len(rd))
            r["n_sign_reversed_is_corrected_readable"] = int((np.sign(rd.delta_is_corrected) != np.sign(rd.delta_laplace)).sum())
            r["n_oracle_higher_heldout_y_pred"] = int((ok.delta_heldout_y_pred > 0).sum())
            r["mean_delta_heldout_y_pred"] = float(ok.delta_heldout_y_pred.mean()) if len(d) else np.nan
            out.append(r)
    return pd.DataFrame(out)


def reading_guide(s: pd.DataFrame) -> dict:
    a = s[(s.subset == "all")]
    get = lambda con, reg: a[(a.contrast == con) & (a.regime == reg)].iloc[0]
    res = {}
    for reg in ("dense", "sparse"):
        so, wr = get("oracle - support_only", reg), get("oracle - wrong", reg)
        res[reg] = {"oracle_gt_support_only": f"{so.n_oracle_higher}/{so.both_ok}",
                    "oracle_gt_wrong": f"{wr.n_oracle_higher}/{wr.both_ok}",
                    "R1_component": bool(so.n_oracle_higher >= 18 and wr.n_oracle_higher >= 18),
                    "R2_not_reliable_vs_support_only": bool(so.n_oracle_higher <= 14),
                    "R3_hessian_decisive": {"support_only": int(so.R3_sign_differs_from_delta_phi),
                                            "wrong": int(wr.R3_sign_differs_from_delta_phi)},
                    "R4_fragile": {"support_only": bool(so.n_sign_reversed_best_of_starts >= 2
                                                        or so.n_sign_reversed_is_corrected_readable >= 2),
                                   "wrong": bool(wr.n_sign_reversed_best_of_starts >= 2
                                                 or wr.n_sign_reversed_is_corrected_readable >= 2)}}
    res["R1"] = bool(res["dense"]["R1_component"] and res["sparse"]["R1_component"])
    res["note"] = "pre-declared judgement aids, not significance tests"
    return res


def diagnostics_summary(stage: Path, mc: pd.DataFrame) -> pd.DataFrame:
    dg = pd.read_csv(stage / "diagnostics.csv", keep_default_na=False, na_values=[""])
    rows = []
    for (reg, m, st), g in dg.groupby(["regime", "method", "start"]):
        rows.append({"regime": reg, "method": m, "start": st, "cases": len(g),
                     "ok": int((g.status == "OK").sum()),
                     "statuses": "|".join(sorted(set(g.status.astype(str)))),
                     "iterations_max": int(g.iterations.max()), "grad_inf_max": float(g.grad_inf.max()),
                     "min_eig_min": float(g.min_eig.min()), "cond_max": float(g.cond.max()),
                     "logdet_slogdet_vs_chol_max_abs": float((g.logdet - g.logdet_chol).abs().max())})
    d = pd.DataFrame(rows)
    m2 = mc.groupby(["regime", "method"]).agg(
        mode_shift_from_saved_max=("mode_shift_from_saved", "max"),
        max_mode_diff_across_starts=("max_mode_diff_across_starts", "max"),
        max_laplace_diff_across_starts=("max_laplace_diff_across_starts", "max"),
        is_relative_ess_median=("is_relative_ess", "median"), is_relative_ess_min=("is_relative_ess", "min"),
        is_correction_median=("is_correction", "median"), is_batch_sd_max=("is_batch_sd", "max"),
        is_non_finite_total=("is_non_finite", "sum")).reset_index()
    return d.merge(m2, on=["regime", "method"], how="left")


def figures(out: Path, p: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, reg in zip(axes, ("dense", "sparse")):
        for i, con in enumerate(["oracle - support_only", "oracle - wrong", "oracle - auto_first"]):
            g = p[(p.contrast == con) & (p.regime == reg) & p.both_ok]
            if con.endswith("auto_first"):
                g = g[~g.same_assignment]
            x = np.full(len(g), i) + np.linspace(-0.15, 0.15, max(len(g), 1))[:len(g)]
            ax.scatter(x, g.delta_laplace, s=16, label=f"Laplace", color="tab:blue" if i == 0 else None)
            ax.scatter(x + 0.3, g.delta_phi, s=10, marker="x", color="gray")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks([0.15, 1.15, 2.15], ["O − SupportOnly", "O − Wrong", "O − Auto\n(differs)"], fontsize=8)
        ax.set_title(f"{reg}: paired Δ Laplace (dots) and Δ Phi(Z_hat) (x)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig1_paired_laplace_differences.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    comps = ["delta_phi_prior", "delta_phi_x_gauss", "delta_phi_x_true_pois_cols", "delta_phi_x_true_cmp_cols",
             "delta_phi_y_train", "delta_neg_half_logdet"]
    labels = ["prior", "X gauss", "X true-Pois", "X true-CMP2", "Y train", "−½log|H|"]
    for ax, reg in zip(axes, ("dense", "sparse")):
        for i, con in enumerate(["oracle - support_only", "oracle - wrong"]):
            g = p[(p.contrast == con) & (p.regime == reg) & p.both_ok]
            ax.bar(np.arange(len(comps)) + 0.38 * i, [g[c].mean() for c in comps], width=0.38, label=con)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(np.arange(len(comps)) + 0.19, labels, fontsize=8, rotation=20)
        ax.set_title(f"{reg}: mean decomposition of Δ Laplace", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig2_decomposition.png", dpi=150)
    plt.close(fig)


def md_table(df: pd.DataFrame, cols: list[str], fmt: int = 3) -> str:
    d = df[cols].copy()
    for k in cols:
        if pd.api.types.is_float_dtype(d[k]):
            d[k] = d[k].map(lambda v: "" if pd.isna(v) else f"{v:.{fmt}f}")
        elif d[k].map(lambda v: isinstance(v, list)).any():
            d[k] = d[k].map(lambda v: f"[{v[0]:.2f}, {v[1]:.2f}]")
    return ("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
            + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.values) + "\n")


def posthoc(mc: pd.DataFrame, p: pd.DataFrame, out: Path) -> str:
    """POST-HOC (added after the full results were read; not pre-registered).
    Paired differences by the number of count columns whose family differs from
    Oracle, the list of datasets where Oracle is lower, and the spread of the
    nuisance terms (Gaussian X, Y) between separately fitted models."""
    orc = mc[mc.method == "oracle"].set_index(["replicate", "regime"]).assignment
    asg = mc.set_index(["replicate", "regime", "method"]).assignment

    def ndiff(r):
        a = orc[(r.replicate, r.regime)].split("|")
        b = asg[(r.replicate, r.regime, r.contrast.split(" - ")[1])].split("|")
        return sum(x != y for x, y in zip(a, b))
    q = p[p.both_ok].copy()
    q["n_diff_cols"] = q.apply(ndiff, axis=1)
    by = q.groupby(["contrast", "regime", "n_diff_cols"]).agg(
        datasets=("delta_laplace", "size"), n_oracle_higher=("delta_laplace", lambda x: int((x > 0).sum())),
        mean_delta=("delta_laplace", "mean"), mean_delta_phi_x_true_cmp_cols=("delta_phi_x_true_cmp_cols", "mean"),
        mean_delta_neg_half_logdet=("delta_neg_half_logdet", "mean"),
        mean_delta_phi_y_train=("delta_phi_y_train", "mean"), mean_delta_phi_x_gauss=("delta_phi_x_gauss", "mean")
    ).reset_index()
    by.to_csv(out / "posthoc_by_n_differing_columns.csv", index=False)
    neg = q[q.delta_laplace < 0][["contrast", "regime", "replicate", "n_diff_cols", "delta_laplace", "delta_phi",
                                  "delta_neg_half_logdet", "delta_phi_x_true_cmp_cols", "delta_phi_x_gauss",
                                  "delta_phi_y_train", "delta_is_corrected", "is_readable"]]
    neg.to_csv(out / "posthoc_oracle_lower_cases.csv", index=False)
    nuis = q[q.contrast.isin(["oracle - support_only", "oracle - wrong"])].groupby(["contrast", "regime"]).agg(
        sd_delta_phi_x_gauss=("delta_phi_x_gauss", "std"), sd_delta_phi_y_train=("delta_phi_y_train", "std"),
        sd_delta_neg_half_logdet=("delta_neg_half_logdet", "std"), sd_delta_laplace=("delta_laplace", "std")).reset_index()
    nuis.to_csv(out / "posthoc_nuisance_spread.csv", index=False)
    return ("\n### 事後に追加した記述表（post-hoc、事前登録外）\n"
            "Oracle と割当てが異なる count 列の数ごとの paired 差\n\n"
            + md_table(by, list(by.columns), 2)
            + "\nOracle の方が低かったケース\n\n" + md_table(neg, list(neg.columns), 2)
            + "\n別々に推定した fit の間の nuisance 項のばらつき（データセット間 SD）\n\n"
            + md_table(nuis, list(nuis.columns), 2))


def main(stage: Path) -> None:
    out = stage / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    mc = pd.read_csv(stage / "model_comparison.csv", keep_default_na=False, na_values=[""])
    p = paired(mc)
    s = summarise(p)
    guide = reading_guide(s)
    dsum = diagnostics_summary(stage, mc)
    p.to_csv(out / "paired_comparison.csv", index=False)
    s.to_csv(out / "paired_summary.csv", index=False)
    dsum.to_csv(out / "diagnostics_summary.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"reading_guide": guide,
                                                   "statuses": mc.status.value_counts().to_dict()},
                                                  indent=2, default=float), encoding="utf-8")
    figures(out, p)
    md = ["<!-- generated by analyze_af4_laplace_model_comparison.py; do not edit by hand -->\n",
          "### paired 比較（データセット単位、Δ = Laplace(Oracle) − Laplace(他)）\n",
          md_table(s, ["contrast", "regime", "subset", "datasets", "both_ok", "n_oracle_higher", "n_equal",
                       "n_oracle_lower", "mean_delta", "median_delta", "ci", "min_delta"], 2),
          "\n### 差の分解（平均、OK ペアのみ）\n",
          md_table(s[s.subset == "all"], ["contrast", "regime", "mean_delta", "mean_delta_phi",
                                          "mean_delta_neg_half_logdet", "mean_delta_phi_prior", "mean_delta_phi_x_gauss",
                                          "mean_delta_phi_x_true_pois_cols", "mean_delta_phi_x_true_cmp_cols",
                                          "mean_delta_phi_y_train", "R3_sign_differs_from_delta_phi"], 2),
          "\n### 頑健性（best-of-starts、IS 補正）と held-out Y（副次）\n",
          md_table(s[s.subset == "all"], ["contrast", "regime", "n_sign_reversed_best_of_starts", "is_readable_pairs",
                                          "n_sign_reversed_is_corrected_readable", "n_oracle_higher_heldout_y_pred",
                                          "mean_delta_heldout_y_pred"], 4),
          "\n### 数値診断\n",
          md_table(dsum, list(dsum.columns), 4),
          "\n### 事前宣言した読み方（R1–R4）\n```json\n" + json.dumps(guide, indent=2) + "\n```\n",
          posthoc(mc, p, out)]
    (out / "results_tables.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
