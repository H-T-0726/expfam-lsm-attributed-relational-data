"""Pre-registered analysis for AF3 / S0 (selector re-scoring).

Reads only the stage directory written by ``run_af3_selector_rescoring.py``
(and AF2's selection_columns.csv for the column-level linkage table) and
writes ``<stage>/analysis/``.  Statistical unit: replicate.  Columns and noise
seeds are averaged inside a replicate before any between-replicate summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

BOOT_N, BOOT_SEED = 10000, 20261113
AF2_COLS = ROOT / "expfam" / "results" / "auto_family_z_recovery" / "af2_20261110" / "full" / "selection_columns.csv"
KEY = ["arm", "variant", "regime", "start"]


def boot_ci(values: np.ndarray, seed: int = BOOT_SEED) -> tuple[float, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(BOOT_N, v.size))
    m = v[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def load(stage: Path) -> pd.DataFrame:
    c = pd.read_csv(stage / "column_selection.csv", keep_default_na=False, na_values=[""])
    for k in ("regime", "start", "variant"):
        c[k] = c[k].fillna("").astype(str)
    c["correct"] = c["correct"].astype(str).str.lower() == "true"
    c["failed"] = c["status"] != "SELECTED"
    c["cmp_to_pois"] = (c.true_model == "cmp_nu2p0") & (c.selected == "poisson")
    c["pois_to_cmp"] = (c.true_model == "poisson") & (c.selected == "cmp_nu2p0")
    return c


def replicate_rates(c: pd.DataFrame) -> pd.DataFrame:
    """Per (arm instance without noise seed) x replicate; seeds averaged inside."""
    g = c.groupby(KEY + ["level", "noise_seed", "replicate"], dropna=False)
    per = g.apply(lambda d: pd.Series({
        "cmp_correct": d.loc[d.true_model == "cmp_nu2p0", "correct"].mean(),
        "pois_correct": d.loc[d.true_model == "poisson", "correct"].mean(),
        "cmp_to_pois": d.loc[d.true_model == "cmp_nu2p0", "cmp_to_pois"].mean(),
        "pois_to_cmp": d.loc[d.true_model == "poisson", "pois_to_cmp"].mean(),
        "n_failed": int(d["failed"].sum()),
        "cmp_margin_median": d.loc[d.true_model == "cmp_nu2p0", "signed_margin"].median(),
        "pois_margin_median": d.loc[d.true_model == "poisson", "signed_margin"].median(),
    }), include_groups=False).reset_index()
    rep = per.groupby(KEY + ["level", "replicate"], dropna=False).agg(
        cmp_correct=("cmp_correct", "mean"), pois_correct=("pois_correct", "mean"),
        cmp_to_pois=("cmp_to_pois", "mean"), pois_to_cmp=("pois_to_cmp", "mean"),
        n_failed=("n_failed", "sum"), cmp_margin_median=("cmp_margin_median", "mean"),
        pois_margin_median=("pois_margin_median", "mean"),
        n_seeds=("noise_seed", "nunique")).reset_index()
    return rep


def arm_summary(c: pd.DataFrame, rep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, d in rep.groupby(KEY + ["level"], dropna=False):
        cc = c
        for k, v in zip(KEY + ["level"], key):
            cc = cc[(cc[k] == v) | (cc[k].isna() & pd.isna(v))]
        cm = cc[cc.true_model == "cmp_nu2p0"]
        pm = cc[cc.true_model == "poisson"]
        err = cc[~cc.correct & ~cc.failed]
        lo, hi = boot_ci(d.cmp_correct.values)
        rows.append({**dict(zip(KEY + ["level"], key)),
                     "replicates": d.replicate.nunique(),
                     "cmp_columns_scored": len(cm), "pois_columns_scored": len(pm),
                     "cmp_correct_rate": d.cmp_correct.mean(), "cmp_correct_ci_low": lo, "cmp_correct_ci_high": hi,
                     "pois_correct_rate": d.pois_correct.mean(),
                     "cmp_to_pois_rate": d.cmp_to_pois.mean(), "pois_to_cmp_rate": d.pois_to_cmp.mean(),
                     "replicates_all_cmp_correct": int((d.cmp_correct == 1).sum()),
                     "n_failed_columns": int(cc.failed.sum()),
                     "n_errors": len(err), "share_errors_cmp_to_pois": (err.cmp_to_pois.mean() if len(err) else float("nan")),
                     "cmp_margin_median": cm.signed_margin.median(),
                     "cmp_margin_q25": cm.signed_margin.quantile(.25), "cmp_margin_q75": cm.signed_margin.quantile(.75),
                     "cmp_margin_per_obs_median": cm.signed_margin_per_obs.median(),
                     "pois_margin_median": pm.signed_margin.median(),
                     "z_rmse_per_sample_median": cc.z_rmse_per_sample.median(),
                     "z_linres_per_sample_median": cc.z_linres_per_sample.median()})
    return pd.DataFrame(rows)


def dose_response(rep: pd.DataFrame, arm: str = "Z2") -> tuple[pd.DataFrame, dict]:
    d = rep[rep.arm == arm].copy()
    d["level"] = d["level"].astype(float)
    slopes = []
    for r, g in d.groupby("replicate"):
        g = g.sort_values("level")
        slopes.append({"replicate": r, "slope_cmp_to_pois": float(np.polyfit(g.level, g.cmp_to_pois, 1)[0]),
                       "slope_pois_to_cmp": float(np.polyfit(g.level, g.pois_to_cmp, 1)[0]),
                       **{f"cmp_to_pois_s{lv}": v for lv, v in zip(g.level, g.cmp_to_pois)}})
    s = pd.DataFrame(slopes)
    lo, hi = boot_ci(s.slope_cmp_to_pois.values)
    summ = {"arm": arm, "replicates": len(s), "mean_slope_cmp_to_pois": float(s.slope_cmp_to_pois.mean()),
            "ci": [lo, hi], "n_positive": int((s.slope_cmp_to_pois > 0).sum()),
            "n_zero": int((s.slope_cmp_to_pois == 0).sum()), "n_negative": int((s.slope_cmp_to_pois < 0).sum()),
            "mean_slope_pois_to_cmp": float(s.slope_pois_to_cmp.mean()),
            "ci_pois_to_cmp": list(boot_ci(s.slope_pois_to_cmp.values))}
    return s, summ


def paired(rep: pd.DataFrame, a: str, b: str, metric: str = "cmp_to_pois") -> tuple[pd.DataFrame, list]:
    base = rep[rep.start == "zero"]
    out, summ = [], []
    for regime in ("dense", "sparse"):
        x = base[(base.arm == a) & (base.regime == regime)].set_index("replicate")[metric]
        y = base[(base.arm == b) & (base.regime == regime)].set_index("replicate")[metric]
        dlt = (x - y).dropna()
        for r, v in dlt.items():
            out.append({"regime": regime, "replicate": r, f"{a}_minus_{b}_{metric}": v})
        lo, hi = boot_ci(dlt.values)
        summ.append({"contrast": f"{a} - {b}", "metric": metric, "regime": regime, "n": len(dlt),
                     "mean": float(dlt.mean()), "ci": [lo, hi], "n_pos": int((dlt > 0).sum()),
                     "n_zero": int((dlt == 0).sum()), "n_neg": int((dlt < 0).sum())})
    return pd.DataFrame(out), summ


def af2_linkage(c: pd.DataFrame) -> pd.DataFrame:
    a = pd.read_csv(AF2_COLS)
    a = a[(a.method == "auto_first") & (a.decided_by == "score")][["replicate", "regime", "column", "correct"]]
    a = a.rename(columns={"correct": "af2_auto_correct"})
    rows = []
    for arm in ("Z3", "Z5", "Z7", "Z4"):
        z = c[(c.arm == arm) & (c.start == "zero")][["replicate", "regime", "column", "true_model", "correct"]]
        m = z.merge(a, on=["replicate", "regime", "column"])
        for (regime, tm), g in m.groupby(["regime", "true_model"]):
            ct = pd.crosstab(g.af2_auto_correct.astype(bool), g.correct.astype(bool))
            get = lambda i, j: int(ct.loc[i, j]) if (i in ct.index and j in ct.columns) else 0
            rows.append({"arm": arm, "regime": regime, "true_model": tm, "columns": len(g),
                         "af2_wrong_s0_wrong": get(False, False), "af2_wrong_s0_right": get(False, True),
                         "af2_right_s0_wrong": get(True, False), "af2_right_s0_right": get(True, True)})
    return pd.DataFrame(rows)


def loading_relation(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cm = c[(c.true_model == "cmp_nu2p0") & ~c.failed]
    for key, g in cm.groupby(KEY + ["level"], dropna=False):
        ok, bad = g[g.correct], g[~g.correct]
        rho = g[["true_loading_norm", "signed_margin"]].corr(method="spearman").iloc[0, 1] if len(g) > 2 else np.nan
        rows.append({**dict(zip(KEY + ["level"], key)), "n": len(g), "n_misselected": len(bad),
                     "median_norm_correct": ok.true_loading_norm.median(),
                     "median_norm_misselected": bad.true_loading_norm.median() if len(bad) else np.nan,
                     "spearman_norm_vs_signed_margin_pooled_descriptive": rho})
    return pd.DataFrame(rows)


def optimizer_diag(stage: Path, c: pd.DataFrame) -> pd.DataFrame:
    s = pd.read_csv(stage / "candidate_scores.csv", keep_default_na=False, na_values=[""])
    s["start"] = s["start"].fillna("").astype(str)
    s["regime"] = s["regime"].fillna("").astype(str)
    s["conv"] = s["converged_1e8"].astype(str).str.lower() == "true"
    rows = []
    for key, g in s.groupby(["arm", "start"]):
        ok = g[g.status == "OK"]
        cc = c[(c.arm == key[0]) & (c.start == key[1]) & ~c.failed]
        ratio = (cc.signed_margin.abs() / (2 * 2 * cc.max_score_gap_bound)).replace([np.inf], np.nan)
        rows.append({"arm": key[0], "start": key[1], "candidate_rows": len(g), "failed": int((g.status != "OK").sum()),
                     "converged_1e8_share": float(ok.conv.mean()), "grad_inf_max": float(ok.grad_inf.max()),
                     "grad_inf_median": float(ok.grad_inf.median()),
                     "hess_min_eig_min": float(ok.hess_min_eig.min()),
                     "score_gap_bound_max": float(ok.score_gap_bound.max()),
                     "per_obs_sum_check_max": float(ok.per_obs_sum_check_abs.max()),
                     "columns_where_bound_could_flip": int((cc.signed_margin.abs() <= 4 * cc.max_score_gap_bound).sum()),
                     "min_margin_over_bound": float(ratio.min()) if len(ratio.dropna()) else np.nan,
                     "failure_reasons": " || ".join(sorted(set(g.loc[g.status != "OK", "failure_reason"].astype(str))))[:500]})
    return pd.DataFrame(rows)


def reading_guide(summ: pd.DataFrame, dose: dict, pair: list, c: pd.DataFrame) -> dict:
    z = summ[summ.start == "zero"]
    get = lambda arm, regime="": z[(z.arm == arm) & (z.regime == regime)]
    z1 = float(get("Z1").cmp_correct_rate.iloc[0])
    z2 = z[z.arm == "Z2"]
    share = float((z2.share_errors_cmp_to_pois * z2.n_errors).sum() / max(z2.n_errors.sum(), 1))
    z3 = {r: float(1 - get("Z3", r).cmp_correct_rate.iloc[0]) for r in ("dense", "sparse")}
    z5 = {r: float(1 - get("Z5", r).cmp_correct_rate.iloc[0]) for r in ("dense", "sparse")}
    mis = c[c.arm.isin(["Z3", "Z5"]) & (c.start == "zero") & ~c.correct & ~c.failed]
    near = float((mis.signed_margin.abs() < 2).mean()) if len(mis) else float("nan")
    return {
        "G1_selector_problem": {"z1_cmp_correct": z1, "triggered": z1 < 0.90},
        "G2_z_error_drives_cmp_to_pois": {"z1_ok": z1 >= 0.90, "mean_slope": dose["mean_slope_cmp_to_pois"],
                                          "ci": dose["ci"], "share_errors_cmp_to_pois": share,
                                          "triggered": bool(z1 >= 0.90 and dose["ci"][0] > 0 and share >= 0.80)},
        "G3_oracle_z_misselects": {"z3_cmp_misselection": z3, "triggered": any(v >= 0.10 for v in z3.values())},
        "G4_incumbent_q": {"z3": z3, "z5": z5,
                           "triggered": any(z3[r] <= 0.05 and z5[r] - z3[r] >= 0.10 for r in z3)},
        "G5_near_ties": {"n_misselected_Z3_Z5": len(mis), "share_abs_margin_lt_2": near,
                         "triggered": bool(len(mis) and near >= 0.5)},
        "note": "pre-declared reading guides, not significance tests",
    }


def figures(out: Path, summ: pd.DataFrame, c: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = summ[summ.start == "zero"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for arm, mk, lab in (("Z2", "o-", "Z2: true Z + iid noise (L=5)"), ("Z2c", "s--", "Z2c: true Z + common noise (L=1)")):
        d = z[z.arm == arm].copy()
        d["level"] = d.level.astype(float)
        d = d.sort_values("level")
        x = np.r_[0.0, d.level.values]
        y = np.r_[z[z.arm == "Z1"].cmp_correct_rate.iloc[0], d.cmp_correct_rate.values]
        ax.plot(x, y, mk, label=lab)
        if arm == "Z2":
            ax.fill_between(d.level, d.cmp_correct_ci_low, d.cmp_correct_ci_high, alpha=0.15)
    colors = {"Z3": "tab:green", "Z5": "tab:red", "Z2m": "tab:purple"}
    for arm in ("Z3", "Z5", "Z2m"):
        for regime, mk in (("dense", "^"), ("sparse", "v")):
            d = z[(z.arm == arm) & (z.regime == regime)]
            if len(d):
                ax.scatter(d.z_rmse_per_sample_median, d.cmp_correct_rate, marker=mk, s=60,
                           color=colors[arm], label=f"{arm} {regime}")
    ax.set_xlabel("noise s  /  per-sample rotation-Procrustes RMSE(Z) (markers)")
    ax.set_ylabel("true CMP2 columns selected correctly")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("AF3/S0: selection accuracy vs Z error (20 AF2 datasets)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig1_cmp_accuracy_vs_z_error.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, tm in zip(axes, ("cmp_nu2p0", "poisson")):
        labels, data = [], []
        cz = c[(c.start == "zero") & (c.true_model == tm) & ~c.failed]
        for arm, regime in (("Z1", ""), ("Z3", "dense"), ("Z3", "sparse"), ("Z5", "dense"), ("Z5", "sparse"),
                            ("Z4", "dense"), ("Z4", "sparse")):
            d = cz[(cz.arm == arm) & (cz.regime == regime)].signed_margin.dropna()
            labels.append(f"{arm}\n{regime}" if regime else arm)
            data.append(d.values)
        for lv in (0.1, 0.3, 0.6):
            d = cz[(cz.arm == "Z2") & (cz.level.astype(str) == str(lv))].signed_margin.dropna()
            labels.append(f"Z2\ns={lv}")
            data.append(d.values)
        ax.boxplot(data, showfliers=True, flierprops={"markersize": 2})
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_title(f"true {tm}: signed margin 2(Q_true - Q_wrong) [symlog]", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig2_signed_margin_by_arm.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cz = c[(c.start == "zero") & (c.true_model == "cmp_nu2p0") & ~c.failed]
    for arm, regime, col in (("Z1", "", "k"), ("Z3", "sparse", "tab:green"), ("Z5", "sparse", "tab:red")):
        d = cz[(cz.arm == arm) & (cz.regime == regime)]
        ax.scatter(d.true_loading_norm, d.signed_margin, s=14, alpha=0.7, color=col,
                   label=f"{arm} {regime}".strip())
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("true loading norm ||f_j||")
    ax.set_ylabel("signed margin (true CMP2 columns)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_margin_vs_loading_norm.png", dpi=150)
    plt.close(fig)


def md_table(df: pd.DataFrame, cols: list[str], fmt: int = 3) -> str:
    d = df[cols].copy()
    for k in cols:
        if pd.api.types.is_float_dtype(d[k]):
            d[k] = d[k].map(lambda v: "" if pd.isna(v) else f"{v:.{fmt}f}")
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.values) + "\n"


def posthoc(stage: Path, c: pd.DataFrame, out: Path) -> dict:
    """POST-HOC (added after the full results were read; not pre-registered).
    Descriptive only: optimiser error scale, zero vs incumbent start, and the
    character of the misselections (margin size, descriptive Vuong-type z,
    per-sample sign disagreement)."""
    s = pd.read_csv(stage / "candidate_scores.csv", keep_default_na=False, na_values=[""])
    z = c[c.arm.isin(["Z3", "Z5", "Z6", "Z7"])]
    p = z.pivot_table(index=["arm", "regime", "replicate", "column"], columns="start", values="signed_margin")
    sel = z.pivot_table(index=["arm", "regime", "replicate", "column"], columns="start",
                        values="correct", aggfunc="first")
    m = c[(c.start == "zero") & ~c.correct & ~c.failed]
    props = m.groupby(["arm", "regime", "true_model"], dropna=False).agg(
        n=("correct", "size"), abs_margin_median=("signed_margin", lambda x: x.abs().median()),
        abs_margin_lt2=("signed_margin", lambda x: int((x.abs() < 2).sum())),
        vuong_z_median=("vuong_z_descriptive", "median"),
        samples_sign_disagree_mean=("n_samples_sign_disagree", "mean")).reset_index()
    props.to_csv(out / "posthoc_misselection_properties.csv", index=False)
    res = {"label": "POST-HOC descriptive (not pre-registered)",
           "grad_inf_max": float(s.grad_inf.max()), "score_gap_bound_max": float(s.score_gap_bound.max()),
           "hess_min_eig_min": float(s.hess_min_eig.min()),
           "per_obs_sum_check_max": float(s.per_obs_sum_check_abs.max()),
           "zero_vs_incumbent_max_abs_margin_diff": float((p["zero"] - p["incumbent"]).abs().max()),
           "zero_vs_incumbent_selection_disagreements": int((sel.nunique(axis=1) > 1).sum())}
    (out / "posthoc_summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main(stage: Path) -> None:
    out = stage / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    c = load(stage)
    rep = replicate_rates(c)
    summ = arm_summary(c, rep)
    dose_rows, dose = dose_response(rep, "Z2")
    dose_c_rows, dose_c = dose_response(rep, "Z2c")
    pr, pr_s = paired(rep, "Z5", "Z3")
    pr2, pr2_s = paired(rep, "Z4", "Z3")
    link = af2_linkage(c) if AF2_COLS.exists() else pd.DataFrame()
    lrel = loading_relation(c)
    opt = optimizer_diag(stage, c)
    guide = reading_guide(summ, dose, pr_s, c)
    rep.to_csv(out / "replicate_rates.csv", index=False)
    summ.to_csv(out / "arm_summary.csv", index=False)
    dose_rows.to_csv(out / "dose_response_Z2_by_replicate.csv", index=False)
    dose_c_rows.to_csv(out / "dose_response_Z2c_by_replicate.csv", index=False)
    pd.concat([pr, pr2.rename(columns=lambda k: k)], axis=0).to_csv(out / "paired_contrasts_by_replicate.csv", index=False)
    link.to_csv(out / "af2_column_linkage.csv", index=False)
    lrel.to_csv(out / "loading_norm_relation.csv", index=False)
    opt.to_csv(out / "optimizer_diagnostics.csv", index=False)
    c[["instance", "arm", "variant", "regime", "level", "noise_seed", "start", "replicate", "column",
       "true_model", "status", "selected", "correct", "signed_margin", "signed_margin_per_obs",
       "per_obs_diff_mean", "vuong_z_descriptive", "per_sample_margin_min", "per_sample_margin_max",
       "n_samples_sign_disagree", "true_loading_norm", "z_rmse_per_sample"]].to_csv(out / "signed_margins.csv", index=False)
    summary = {"dose_response_Z2": dose, "dose_response_Z2c": dose_c, "paired": pr_s + pr2_s,
               "reading_guide": guide, "n_columns_scored": int(len(c)), "n_failed_columns": int(c.failed.sum())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    figures(out, summ, c)
    cols = ["arm", "variant", "regime", "start", "replicates", "cmp_correct_rate", "cmp_correct_ci_low",
            "cmp_correct_ci_high", "pois_correct_rate", "cmp_to_pois_rate", "pois_to_cmp_rate",
            "n_failed_columns", "cmp_margin_median", "z_rmse_per_sample_median"]
    order = {"Z1": 0, "Z2": 1, "Z3": 2, "Z5": 3, "Z4": 4, "Z6": 5, "Z7": 6, "Z2c": 7, "Z2m": 8}
    s2 = summ.assign(o=summ.arm.map(order)).sort_values(["start", "o", "regime", "level"], ascending=[False, True, True, True])
    md = ["<!-- generated by analyze_af3_selector_rescoring.py; do not edit by hand -->\n",
          "### 腕ごとの選択率（replicate 単位の平均、Z2 系は replicate 内で 5 seed を平均）\n",
          md_table(s2, cols),
          "\n### Z2 の用量反応（replicate ごとの OLS 傾き、CMP2→Poisson 誤選択率 ~ s）\n",
          md_table(pd.DataFrame([dose, dose_c]).assign(ci=lambda d: d.ci.map(lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]")),
                   ["arm", "replicates", "mean_slope_cmp_to_pois", "ci", "n_positive", "n_zero", "n_negative",
                    "mean_slope_pois_to_cmp"]),
          "\n### paired 対比（replicate 単位、CMP2→Poisson 誤選択率）\n",
          md_table(pd.DataFrame(pr_s + pr2_s).assign(ci=lambda d: d.ci.map(lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]")),
                   ["contrast", "regime", "n", "mean", "ci", "n_pos", "n_zero", "n_neg"]),
          "\n### 候補最適化の診断\n",
          md_table(opt, ["arm", "start", "candidate_rows", "failed", "converged_1e8_share", "grad_inf_max",
                         "hess_min_eig_min", "score_gap_bound_max", "columns_where_bound_could_flip"], 4),
          "\n### 真の loading ノルムと選択（真 CMP2 列、記述）\n",
          md_table(lrel[lrel.start == "zero"], ["arm", "variant", "regime", "level", "n", "n_misselected",
                                                "median_norm_correct", "median_norm_misselected",
                                                "spearman_norm_vs_signed_margin_pooled_descriptive"]),
          "\n### AF2 Auto（Poisson 開始）との列単位の対応\n",
          md_table(link, list(link.columns)) if len(link) else "(none)\n",
          "\n### 事前宣言した読み方（G1–G5）\n```json\n" + json.dumps(guide, indent=2, default=float) + "\n```\n"]
    ph = posthoc(stage, c, out)
    props = pd.read_csv(out / "posthoc_misselection_properties.csv", keep_default_na=False, na_values=[""])
    md += ["\n### 事後に追加した記述表（post-hoc、事前登録外）\n",
           "最適化誤差の規模と開始値: ```" + json.dumps(ph) + "```\n\n",
           md_table(props, list(props.columns), 2)]
    (out / "results_tables.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
