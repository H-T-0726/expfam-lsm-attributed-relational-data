"""Pre-registered analysis for AF5 / S2-B (exploratory).

Statistical unit: dataset (20 per regime).  Writes ``<stage>/analysis/``.
Columns decided UNDECIDED_NOT_CONVERGED / UNDECIDED_NUMERIC / FAILED are
never counted as Laplace decisions; in every "applied" comparison they keep
the AF2 Auto selection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
RES = ROOT / "expfam" / "results" / "auto_family_z_recovery"
AF2_COLS = RES / "af2_20261110" / "full" / "selection_columns.csv"
AF3_COLS = RES / "af3_selector_rescoring_20261113" / "full" / "column_selection.csv"
BOOT_N, BOOT_SEED = 10000, 20261115
SIGN_ELIGIBLE = ("DECIDED", "DECIDED_LAPLACE_UNASSESSED", "UNDECIDED_LAPLACE")
CONFIDENT = ("DECIDED",)
Q_ARM_OF_ANCHOR = {"auto_first": "Z7", "support_only": "Z5", "oracle": "Z3"}


def boot_ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(BOOT_SEED)
    m = v[rng.integers(0, v.size, size=(BOOT_N, v.size))].mean(axis=1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def load(stage: Path) -> pd.DataFrame:
    c = pd.read_csv(stage / "column_selection.csv", keep_default_na=False, na_values=[""])
    c["sign_eligible"] = c.status.isin(SIGN_ELIGIBLE)
    c["confident"] = c.status.isin(CONFIDENT)
    c["lap_correct"] = c.laplace_sign_correct.astype(str).str.lower() == "true"
    a2 = pd.read_csv(AF2_COLS)
    a2 = a2[(a2.method == "auto_first") & (a2.decided_by == "score")][["replicate", "regime", "column", "selected_model"]]
    c = c.merge(a2.rename(columns={"selected_model": "af2_auto_selected"}), on=["replicate", "regime", "column"], how="left")
    c["af2_auto_correct"] = c.af2_auto_selected == c.true_model
    if AF3_COLS.exists():
        q = pd.read_csv(AF3_COLS, keep_default_na=False, na_values=[""])
        q = q[(q.start.astype(str) == "zero") & q.arm.isin(Q_ARM_OF_ANCHOR.values())]
        q = q[["arm", "regime", "replicate", "column", "selected"]].rename(columns={"selected": "q_same_anchor_selected"})
        c["q_arm"] = c.anchor.map(Q_ARM_OF_ANCHOR)
        c = c.merge(q, left_on=["q_arm", "regime", "replicate", "column"],
                    right_on=["arm", "regime", "replicate", "column"], how="left").drop(columns=["arm"])
        c["q_same_anchor_correct"] = c.q_same_anchor_selected == c.true_model
    # applied selections: undecided / failed keep the AF2 Auto choice
    c["applied_sign"] = np.where(c.sign_eligible, c.laplace_selected, c.af2_auto_selected)
    c["applied_confident"] = np.where(c.confident, c.laplace_selected, c.af2_auto_selected)
    c["applied_sign_correct"] = c.applied_sign == c.true_model
    c["applied_confident_correct"] = c.applied_confident == c.true_model
    return c


def status_table(c):
    return (c.groupby(["anchor", "regime", "status"]).size().unstack(fill_value=0).reset_index())


def accuracy_table(c):
    rows = []
    for (a, reg), g in c.groupby(["anchor", "regime"], sort=False):
        e, cf = g[g.sign_eligible], g[g.confident]
        rep = g.groupby("replicate").apply(lambda d: d.loc[d.sign_eligible, "lap_correct"].mean(), include_groups=False)
        row = {"anchor": a, "role": g.anchor_role.iloc[0], "regime": reg, "columns": len(g),
               "sign_eligible": len(e), "confident": len(cf),
               "undecided_or_failed": int((~g.sign_eligible).sum()),
               "sign_acc_all": float(e.lap_correct.mean()), "sign_acc_replicate_mean": float(rep.mean()),
               "sign_acc_ci": boot_ci(rep.values),
               "sign_acc_true_cmp": float(e[e.true_model == "cmp_nu2p0"].lap_correct.mean()),
               "sign_acc_true_pois": float(e[e.true_model == "poisson"].lap_correct.mean()),
               "n_true_cmp_eligible": int((e.true_model == "cmp_nu2p0").sum()),
               "n_true_pois_eligible": int((e.true_model == "poisson").sum()),
               "confident_acc": float(cf.lap_correct.mean()) if len(cf) else np.nan,
               "confident_acc_true_cmp": float(cf[cf.true_model == "cmp_nu2p0"].lap_correct.mean()) if len(cf) else np.nan,
               "confident_acc_true_pois": float(cf[cf.true_model == "poisson"].lap_correct.mean()) if len(cf) else np.nan}
        if "q_same_anchor_correct" in g:
            row["q_same_anchor_acc"] = float(g.q_same_anchor_correct.mean())
        row["af2_auto_acc"] = float(g.af2_auto_correct.mean())
        row["jointmap_variant_acc"] = float(((g.delta_jointmap_variant > 0) == (g.true_model == "cmp_nu2p0"))[g.sign_eligible].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tables(c):
    rows, drows = [], []
    for (a, reg), g in c.groupby(["anchor", "regime"], sort=False):
        for ref, refcol in (("af2_auto", "af2_auto_correct"), ("q_same_anchor", "q_same_anchor_correct")):
            if refcol not in g:
                continue
            for mode, col in (("sign", "applied_sign_correct"), ("confident", "applied_confident_correct")):
                rows.append({"anchor": a, "regime": reg, "reference": ref, "laplace_mode": mode,
                             "ref_wrong_lap_right": int((~g[refcol] & g[col]).sum()),
                             "ref_right_lap_wrong": int((g[refcol] & ~g[col]).sum()),
                             "both_right": int((g[refcol] & g[col]).sum()),
                             "both_wrong": int((~g[refcol] & ~g[col]).sum()),
                             "ref_wrong_total": int((~g[refcol]).sum()),
                             "ref_wrong_true_cmp": int((~g[refcol] & (g.true_model == "cmp_nu2p0")).sum()),
                             "corrected_true_cmp": int((~g[refcol] & g[col] & (g.true_model == "cmp_nu2p0")).sum())})
                per = g.groupby("replicate").apply(lambda d: int(d[col].sum()) - int(d[refcol].sum()), include_groups=False)
                drows.append({"anchor": a, "regime": reg, "reference": ref, "laplace_mode": mode,
                              "datasets": len(per), "mean_change_correct_cols": float(per.mean()),
                              "ci": boot_ci(per.values), "n_improved": int((per > 0).sum()),
                              "n_same": int((per == 0).sum()), "n_worse": int((per < 0).sum())})
    return pd.DataFrame(rows), pd.DataFrame(drows)


def anchor_sensitivity(c):
    p = c.pivot_table(index=["replicate", "regime", "column", "true_model"], columns="anchor",
                      values="laplace_selected", aggfunc="first").reset_index()
    e = c.pivot_table(index=["replicate", "regime", "column"], columns="anchor", values="sign_eligible",
                      aggfunc="first").reset_index()
    rows = []
    for reg in ("dense", "sparse"):
        for a, b in (("auto_first", "support_only"), ("auto_first", "oracle"), ("support_only", "oracle")):
            m = p[p.regime == reg].merge(e[e.regime == reg][["replicate", "column", a, b]].rename(
                columns={a: "ea", b: "eb"}), on=["replicate", "column"])
            m = m[m.ea.astype(bool) & m.eb.astype(bool)]
            rows.append({"regime": reg, "anchor_a": a, "anchor_b": b, "both_eligible": len(m),
                         "agree": int((m[a] == m[b]).sum()),
                         "agree_true_cmp": int(((m[a] == m[b]) & (m.true_model == "cmp_nu2p0")).sum()),
                         "n_true_cmp": int((m.true_model == "cmp_nu2p0").sum())})
    return pd.DataFrame(rows)


def error_diag(c):
    rows = []
    for (a, reg), g in c.groupby(["anchor", "regime"], sort=False):
        ok = g[g.status != "FAILED"]
        rd = ok[ok.is_readable.astype(str).str.lower() == "true"]
        rows.append({"anchor": a, "regime": reg, "columns": len(g), "is_readable": len(rd),
                     "median_abs_delta": float(ok.delta.abs().median()),
                     "median_abs_is_delta_correction": float(rd.is_delta_correction.abs().median()) if len(rd) else np.nan,
                     "median_is_delta_correction_se": float(rd.is_delta_correction_se.median()) if len(rd) else np.nan,
                     "n_sign_flip_is_readable": int((np.sign(rd.delta_is_corrected) != np.sign(rd.delta)).sum()),
                     "n_undecided_laplace": int((g.status == "UNDECIDED_LAPLACE").sum()),
                     "n_undecided_numeric": int((g.status == "UNDECIDED_NUMERIC").sum()),
                     "n_not_converged": int((g.status == "UNDECIDED_NOT_CONVERGED").sum()),
                     "n_failed": int((g.status == "FAILED").sum()),
                     "median_numeric_uncertainty": float(ok.numeric_uncertainty.median())})
    return pd.DataFrame(rows)


def decomposition(c):
    g = c[c.status != "FAILED"].copy()
    g["correct_sign"] = g.lap_correct
    return g.groupby(["anchor", "regime", "true_model", "correct_sign"]).agg(
        n=("delta", "size"), mean_delta=("delta", "mean"), mean_delta_phi_target=("delta_phi_target_col", "mean"),
        mean_delta_phi_other=("delta_phi_other", "mean"), mean_delta_neg_half_logdet=("delta_neg_half_logdet", "mean"),
        median_true_norm=("true_loading_norm", "median")).reset_index()


def optimisation_diag(stage: Path):
    s = pd.read_csv(stage / "candidate_scores.csv", keep_default_na=False, na_values=[""])
    ok = s[s.status == "OK"]
    return ok.groupby(["anchor", "regime", "candidate"]).agg(
        n=("S", "size"), failed=("S", lambda x: 0), outer_fd_grad_inf_max=("outer_fd_grad_inf", "max"),
        fresh_restart_diff_max=("fresh_restart_diff", "max"), start_disagreement_max=("start_disagreement", "max"),
        start_disagreement_median=("start_disagreement", "median"), inner_grad_inf_max=("inner_grad_inf", "max"),
        hess_min_eig_min=("hess_min_eig", "min"), outer_iter_max=("outer_iter", "max"),
        evals_median=("n_laplace_evals", "median"), seconds_median=("seconds", "median"),
        seconds_total=("seconds", "sum")).reset_index().assign(
        failed_rows=lambda d: d.apply(lambda r: int(((s.anchor == r.anchor) & (s.regime == r.regime)
                                                     & (s.candidate == r.candidate) & (s.status != "OK")).sum()), axis=1))


def figures(out: Path, c: pd.DataFrame, pt: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, a in zip(axes, ("auto_first", "support_only", "oracle")):
        for i, (reg, tm) in enumerate([(r, t) for r in ("dense", "sparse") for t in ("cmp_nu2p0", "poisson")]):
            g = c[(c.anchor == a) & (c.regime == reg) & (c.true_model == tm) & (c.status != "FAILED")]
            x = np.full(len(g), i) + np.linspace(-0.25, 0.25, max(len(g), 1))[:len(g)]
            col = np.where(g.sign_eligible, np.where(g.af2_auto_correct, "tab:blue", "tab:red"), "lightgray")
            ax.scatter(x, g.delta, s=10, c=col)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks(range(4), ["dense\ntrue CMP2", "dense\ntrue Pois", "sparse\ntrue CMP2", "sparse\ntrue Pois"], fontsize=8)
        ax.set_title(f"anchor = {a}: Δ = S(CMP2) − S(Pois)\nred: AF2 Auto wrong, gray: undecided", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_delta_by_column.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    d = pt[(pt.reference == "af2_auto") & (pt.laplace_mode == "sign")]
    lab = [f"{r.anchor}\n{r.regime}" for r in d.itertuples()]
    ax.bar(np.arange(len(d)) - 0.2, d.ref_wrong_lap_right, width=0.4, label="AF2 wrong → Laplace right")
    ax.bar(np.arange(len(d)) + 0.2, d.ref_right_lap_wrong, width=0.4, label="AF2 right → Laplace wrong")
    ax.set_xticks(range(len(d)), lab, fontsize=7)
    ax.legend(fontsize=8)
    ax.set_title("corrections and harms vs AF2 Auto (sign rule; undecided keep AF2)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig2_corrections_harms.png", dpi=150)
    plt.close(fig)


def md_table(df, cols, fmt=3):
    d = df[cols].copy()
    for k in cols:
        if pd.api.types.is_float_dtype(d[k]):
            d[k] = d[k].map(lambda v: "" if pd.isna(v) else f"{v:.{fmt}f}")
        elif d[k].map(lambda v: isinstance(v, list)).any():
            d[k] = d[k].map(lambda v: f"[{v[0]:.2f}, {v[1]:.2f}]" if isinstance(v, list) else v)
    return ("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
            + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.values) + "\n")


def main(stage: Path) -> None:
    out = stage / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    c = load(stage)
    st, acc = status_table(c), accuracy_table(c)
    pt, dt = paired_tables(c)
    an, er, dec = anchor_sensitivity(c), error_diag(c), decomposition(c)
    od = optimisation_diag(stage)
    c.to_csv(out / "column_selection_annotated.csv", index=False)
    st.to_csv(out / "status_counts.csv", index=False)
    acc.to_csv(out / "accuracy.csv", index=False)
    pt.to_csv(out / "paired_comparison.csv", index=False)
    dt.to_csv(out / "dataset_change.csv", index=False)
    an.to_csv(out / "anchor_sensitivity.csv", index=False)
    er.to_csv(out / "error_diagnostics.csv", index=False)
    dec.to_csv(out / "decomposition.csv", index=False)
    od.to_csv(out / "optimization_diagnostics.csv", index=False)
    figures(out, c, pt)
    md = ["<!-- generated by analyze_af5_local_laplace_selection.py; do not edit by hand -->\n",
          "### 判定状態\n", md_table(st, list(st.columns)),
          "\n### 正解率（列単位、sign = 判定不能・失敗を除く Δ の符号、confident = DECIDED のみ）\n",
          md_table(acc, list(acc.columns)),
          "\n### 列ごとの訂正と副作用（判定不能・失敗は AF2 Auto の選択を維持）\n",
          md_table(pt, list(pt.columns)),
          "\n### データセット単位の正解列数の変化（Laplace 適用後 − 参照）\n", md_table(dt, list(dt.columns)),
          "\n### anchor 間の一致\n", md_table(an, list(an.columns)),
          "\n### 近似誤差の診断\n", md_table(er, list(er.columns), 4),
          "\n### Δ の分解\n", md_table(dec, list(dec.columns), 2),
          "\n### 最適化の診断\n", md_table(od, list(od.columns), 5)]
    (out / "results_tables.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
