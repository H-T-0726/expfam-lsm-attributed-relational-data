"""AF8-v3 final analysis (written before the re-pilot and before any accuracy of an AF8 variant is computed).

--pilot-audit   numerical stop rules on the re-pilot (rep01-03); no accuracy, no AUC; v3-pilot vs re-pilot differences
--full-audit    the same stop rules on the 40-dataset run (computed BEFORE any accuracy)
--manifest      data / anchor manifest of the 40 datasets (seeds, hashes asserted against AF6)
--analyze       exploratory accuracy analysis; refuses to run unless --full-audit passed

Strictness (audit W3): no statistic silently skips NaN.  `worst()` maps a NaN to the failing side of its rule and
counts it; expected cells are checked against the full grid; every column has an explicit state.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_af8_integrated_loading as a8                # noqa: E402
import run_af8v2_integrated_loading as v2              # noqa: E402
import run_af8v3_integrated_loading as v3              # noqa: E402
import run_af8v3_full_integrated_loading as vf         # noqa: E402

P, C = a8.P, a8.C
OUT = vf.OUT
KEYS = ["replicate", "regime", "column", "prior"]
MIS = ["SAME_POINT_VALUE_MISMATCH", "MULTIMODAL_NONNEGLIGIBLE", "MULTIMODAL_DOMINANT"]
V3_PILOT = v3.OUT / "pilot"
AF6_FULL = a8.AF6
AF6_ROOT = AF6_FULL.parent
BOOT_SEED, BOOT_B, BOOT_B_AUC = a8.BOOT_SEED, 10000, 2000
METHODS = ["A_Q", "B_L6", "C_JM", "D_MAP", "E2", "E", "IS"]
FLOOR = a8.NUMERIC_FLOOR


# --------------------------------------------------------------------------
# strict helpers
# --------------------------------------------------------------------------

def num(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def worst(x, kind: str, name: str, nan_log: dict) -> float:
    """max (kind='max') / min (kind='min') / median with NaN mapped to the FAILING side (+inf for an upper-bounded
    statistic, -inf for a lower-bounded one) and counted in nan_log[name].  An empty input is a failure."""
    a = np.asarray(num(pd.Series(x)), float)
    n_nan = int(np.isnan(a).sum())
    nan_log[name] = n_nan
    if a.size == 0:
        return float("inf") if kind in ("max", "median") else float("-inf")
    a = np.where(np.isnan(a), np.inf if kind in ("max", "median") else -np.inf, a)
    return float({"max": np.max, "min": np.min, "median": np.median}[kind](a))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_runinfo(path: Path, inputs: list[Path], extra: dict):
    info = {"script": "expfam/src/experimental/analyze_af8v3_full_integrated_loading.py", **vf.env_info(),
            "inputs": {str(p.relative_to(a8.ROOT)) if p.is_relative_to(a8.ROOT) else str(p): sha256_file(p)
                       for p in inputs}, **extra}
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def expected_cells(reps) -> pd.DataFrame:
    return pd.DataFrame([{"replicate": f"rep{r + 1:02d}", "regime": g, "column": j, "candidate": cand, "prior": pr}
                         for r, g, j, cand, pr in a8.cells(reps)])


def completeness(c: pd.DataFrame, reps) -> dict:
    exp = expected_cells(reps)
    k = ["replicate", "regime", "column", "candidate", "prior"]
    have = c[k].astype({"column": int})
    dup = int(have.duplicated().sum())
    m = exp.merge(have.drop_duplicates(), on=k, how="left", indicator=True)
    missing = m[m._merge == "left_only"]
    extra = have.merge(exp, on=k, how="left", indicator=True)
    return {"expected": len(exp), "present": len(c), "missing": len(missing), "duplicates": dup,
            "unexpected": int((extra._merge == "left_only").sum()),
            "missing_examples": missing.head(10)[k].to_dict("records"),
            "complete": len(missing) == 0 and dup == 0 and int((extra._merge == "left_only").sum()) == 0}


# --------------------------------------------------------------------------
# pairs (label-free) with explicit states
# --------------------------------------------------------------------------

def cell_valid(r: dict | None, m: str) -> bool:
    if r is None:
        return False
    if m in ("E2", "D"):
        return r.get("status") == "OK" and np.isfinite(num(pd.Series([r.get(m)])).iloc[0])
    if m == "E":
        return r.get("E_status") == "OK" and np.isfinite(num(pd.Series([r.get("E")])).iloc[0])
    if m == "IS":
        return r.get("IS_status") == "OK" and np.isfinite(num(pd.Series([r.get("IS")])).iloc[0])
    raise ValueError(m)


def state(vp: bool, vc: bool, missing: bool, d: float, u: float) -> str:
    if missing:
        return "MISSING"
    if not vp and not vc:
        return "BOTH_FAILED"
    if not (vp and vc):
        return "ONE_FAILED"
    return "NUMERIC_UNDECIDED" if abs(d) <= u else "DECIDED"


def pairs(c: pd.DataFrame, reps) -> pd.DataFrame:
    """one row per (replicate, regime, column, prior) of the EXPECTED grid; Delta and state per method."""
    grid = expected_cells(reps).drop(columns="candidate").drop_duplicates()
    c = c.copy()
    c["column"] = c.column.astype(int)
    idx = {(r["replicate"], r["regime"], r["column"], r["prior"], r["candidate"]): r for r in c.to_dict("records")}
    rows = []
    for g in grid.itertuples(index=False):
        k = (g.replicate, g.regime, g.column, g.prior)
        rp, rc = idx.get(k + (P,)), idx.get(k + (C,))
        missing = rp is None or rc is None
        row = dict(zip(KEYS, k))
        # E2 state with the v2 uncertainty rule (FD-Hessian-step sensitivity)
        st, _, d = v2.selection_state(rp, rc) if not missing else ("MISSING", None, np.nan)
        row["state_E2"] = {"DECIDED": "DECIDED", "UNCERTAIN_NUMERIC": "NUMERIC_UNDECIDED", "UNDECIDED_ONE_FAILED": "ONE_FAILED",
                           "UNDECIDED_BOTH_FAILED": "BOTH_FAILED", "MISSING": "MISSING"}[st]
        row["Delta_E2"] = d if row["state_E2"] in ("DECIDED", "NUMERIC_UNDECIDED") else np.nan
        for m in ("D", "E", "IS"):
            if m == "IS" and g.prior != a8.MAIN_PRIOR:
                continue
            vp, vc = (cell_valid(rp, m), cell_valid(rc, m)) if not missing else (False, False)
            dm = float(num(pd.Series([rc[m]])).iloc[0] - num(pd.Series([rp[m]])).iloc[0]) if (vp and vc) else np.nan
            row[f"state_{m}"] = state(vp, vc, missing, dm, FLOOR)
            row[f"Delta_{m}"] = dm
        if not missing:
            for h in vf.FD_H_HESS_ALT:
                key = f"E2_h{h:g}"
                row[f"Delta_{key}"] = float(num(pd.Series([rc.get(key)])).iloc[0] - num(pd.Series([rp.get(key)])).iloc[0])
            if g.prior == a8.MAIN_PRIOR:
                row["IS_se_delta"] = float(np.hypot(num(pd.Series([rp.get("IS_mc_se")])).iloc[0],
                                                    num(pd.Series([rc.get("IS_mc_se")])).iloc[0]))
            for key in ("E2_logdet_f", "E_logdet_schur_f", "E_logdet_ZZ", "E", "E2", "D", "fD_norm"):
                row[f"Delta_{key}"] = float(num(pd.Series([rc.get(key)])).iloc[0] - num(pd.Series([rp.get(key)])).iloc[0])
            row["D_multistart_P"], row["D_multistart_C"] = rp.get("D_multistart"), rc.get("D_multistart")
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# numerical audit (stop rules)
# --------------------------------------------------------------------------

def start_log_frame(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in c.itertuples(index=False):
        s = getattr(r, "D_start_log", None)
        if isinstance(s, str):
            for name, e in json.loads(s).items():
                rows.append({"replicate": r.replicate, "regime": r.regime, "column": r.column, "candidate": r.candidate,
                             "prior": r.prior, "start": name, **e})
    return pd.DataFrame(rows)


def numerical_audit(stage: str, reps, with_s7: bool) -> dict:
    out = OUT / stage
    c = pd.read_csv(out / "candidate_scores.csv", low_memory=False)
    nan_log: dict[str, int] = {}
    comp = completeness(c, reps)
    pr = pairs(c, reps)
    pr.to_csv(out / f"{stage}_pairs.csv", index=False)
    main = pr[pr.prior == a8.MAIN_PRIOR]
    cm = c[c.prior == a8.MAIN_PRIOR]
    n_main_exp = len(expected_cells(reps).query("prior == @a8.MAIN_PRIOR"))
    failed = (c.status != "OK") | (c.E2_status != "OK")
    n_fail_all = int(failed.sum()) + comp["missing"]                    # a missing cell is a failed cell
    sl = start_log_frame(c)
    sl.to_csv(out / f"{stage}_start_certificates.csv", index=False)
    # IS: a main cell without an IS result (not run / E2 failed) counts as an IS cell with failed draws
    isf, isr = num(cm.IS_failed), num(cm.IS_rescued)
    is_missing = isf.isna() | isr.isna()
    cells_failed_is = int(((isf > 0) | is_missing).sum()) + (n_main_exp - len(cm))
    cells_resc = int(((isr > 0) | is_missing).sum()) + (n_main_exp - len(cm))
    cells_v2def = int((((isf + isr) > 0) | is_missing).sum()) + (n_main_exp - len(cm))
    both_is = main[(main.state_IS.isin(["DECIDED", "NUMERIC_UNDECIDED"]))
                   & main.state_E2.isin(["DECIDED", "NUMERIC_UNDECIDED"])]
    d_e2_is = (both_is.Delta_E2 - both_is.Delta_IS).abs()
    alt = main[[f"Delta_E2_h{h:g}" for h in vf.FD_H_HESS_ALT]].sub(main.Delta_E2, axis=0).abs()
    fdsens = alt.max(axis=1, skipna=False)                              # NaN stays NaN -> failing side in worst()
    s3 = worst(d_e2_is, "median", "S3", nan_log)
    med_abs_e2 = worst(both_is.Delta_E2.abs(), "median", "S3b_scale", nan_log)   # same pair set as S3 (protocol units)
    res = {
        "stage": stage, "completeness": comp,
        "cells": len(c), "cells_by_status": c.status.value_counts().to_dict(),
        "cells_E2_status": c.E2_status.fillna("NA").value_counts().to_dict(),
        "cells_E_status": c.E_status.fillna("NA").value_counts().to_dict(),
        "S1_cell_numerical_failure_share": n_fail_all / comp["expected"],
        "S2_multistart_disagreement_share": float(c.D_multistart.isin(MIS).sum() / comp["expected"]),
        "S2b_cells_with_fewer_than_2_certified_starts_share": float(((num(c.D_n_valid_starts).fillna(0) < 2).sum() + comp["missing"]) / comp["expected"]),
        "D_start_certificate_status_counts": sl.status.value_counts().to_dict() if len(sl) else {},
        "D_start_extra_step_fraction_counts": {str(k): int(v) for k, v in sl.extra_step_fraction.value_counts(dropna=False).items()} if "extra_step_fraction" in sl else {},
        "D_valid_starts_per_cell": {int(k): int(v) for k, v in num(c.D_n_valid_starts).fillna(0).astype(int).value_counts().sort_index().items()},
        "column_states_main_prior": {m: main[f"state_{m}"].value_counts().to_dict() for m in ("E2", "D", "E", "IS")},
        "column_states_all_priors": {m: pr[f"state_{m}"].value_counts().to_dict() for m in ("E2", "D", "E")},
        "main_cells_expected": n_main_exp, "main_cells": len(cm),
        "main_pairs": len(main), "main_pairs_with_IS_and_E2_valid": len(both_is),
        "main_pairs_excluded_from_S3": int(len(main) - len(both_is)),
        "S3_median_abs_DeltaE2_minus_DeltaIS": s3,
        "S3_max_abs_DeltaE2_minus_DeltaIS": worst(d_e2_is, "max", "S3max", nan_log),
        "median_abs_DeltaE2_main": med_abs_e2,
        "S4_IS_cells_with_failed_draws_share": cells_failed_is / n_main_exp,
        "S4v2_IS_cells_with_any_cold_start_failure_share": cells_v2def / n_main_exp,
        "S4r_IS_cells_with_any_rescued_draw_share": cells_resc / n_main_exp,
        "IS_cells_without_result": int(is_missing.sum()) + (n_main_exp - len(cm)),
        "IS_draws_total": int(num(cm.IS_draws).fillna(0).sum()), "IS_failed_draws_total": int(isf.fillna(0).sum()),
        "IS_rescued_draws_total": int(isr.fillna(0).sum()),
        "S4b_IS_min_ess": worst(cm.IS_ess, "min", "S4b", nan_log),
        "S5_determinism_max_abs_diff": worst(cm.determinism_abs_diff, "max", "S5", nan_log),
        "S6_median_fd_step_sensitivity_DeltaE2": worst(fdsens, "median", "S6", nan_log),
        "S6_max_fd_step_sensitivity_DeltaE2": worst(fdsens, "max", "S6max", nan_log),
        "S8_nelder_mead_gain_over_fD_max": worst(cm.nm_gain_over_fD, "max", "S8", nan_log),
        "nm_invalid_evals_total": int(num(cm.nm_invalid_evals).fillna(0).sum()),
        "S9_fd_noise_implied_gap_max": worst(cm.fd_noise_implied_gap, "max", "S9", nan_log),
        "outer_lam2_half_max": worst(c.D_lam2_half_f[c.status == "OK"], "max", "outer_lam2", nan_log),
        "outer_newton_gain_max": worst(c.D_newton_gain_f[c.status == "OK"], "max", "outer_gain", nan_log),
        "outer_grad_inf_max": worst(c.D_grad_inf[c.status == "OK"], "max", "outer_grad", nan_log),
        "outer_min_eig_min": worst(c.D_min_eig_f[c.status == "OK"], "min", "outer_min_eig", nan_log),
        "E_lam2_half_max": worst(c.E_lam2_half[c.E_status == "OK"], "max", "E_lam2", nan_log),
        "E_lap_step_change_max": worst(c.E_lap_step_change[c.E_status == "OK"], "max", "E_lap", nan_log),
        "E_min_eig_min": worst(c.E_min_eig[c.E_status == "OK"], "min", "E_min_eig", nan_log),
        "n_S_evals_total": int(num(c.n_S_evals).sum()), "n_S_not_ok_total": int(num(c.n_S_not_ok).sum()),
        "diag_E_vs_E2_main_pairs_both_valid": int((main.state_E.isin(["DECIDED", "NUMERIC_UNDECIDED"]) & main.state_E2.isin(["DECIDED", "NUMERIC_UNDECIDED"])).sum()),
        "diag_median_abs_DeltaE_minus_DeltaE2_main": float((main.Delta_E - main.Delta_E2).abs().dropna().median()),
        "diag_sign_disagree_E_vs_E2_main": int(((main.Delta_E * main.Delta_E2) < 0).sum()),
        "diag_sign_disagree_E2_vs_IS_main": int(((both_is.Delta_E2 * both_is.Delta_IS) < 0).sum()),
        "seconds_total": float(num(c.seconds).sum()),
    }
    sr = vf.STOP_RULES
    stop = {
        "S0": not comp["complete"],
        "S1": not res["S1_cell_numerical_failure_share"] <= sr["S1_cell_numerical_failure_share_max"],
        "S2": not res["S2_multistart_disagreement_share"] <= sr["S2_multistart_disagreement_share_max"],
        "S2b": not res["S2b_cells_with_fewer_than_2_certified_starts_share"] <= sr["S2b_cells_with_fewer_than_2_certified_starts_share_max"],
        "S3": not s3 <= sr["S3_main_median_abs_DeltaE2_minus_DeltaIS_max"],
        "S3b": not s3 < 0.5 * med_abs_e2,
        "S4": not res["S4_IS_cells_with_failed_draws_share"] <= sr["S4_IS_cells_with_failed_draws_share_max"],
        "S4v2": not res["S4v2_IS_cells_with_any_cold_start_failure_share"] <= sr["S4v2_IS_cells_with_any_cold_start_failure_share_max"],
        "S4r": not res["S4r_IS_cells_with_any_rescued_draw_share"] <= sr["S4r_IS_cells_with_any_rescued_draw_share_max"],
        "S4b": not res["S4b_IS_min_ess"] >= sr["S4b_IS_min_ess"],
        "S5": not res["S5_determinism_max_abs_diff"] <= sr["S5_determinism_max_abs_diff"],
        "S6": not res["S6_median_fd_step_sensitivity_DeltaE2"] <= sr["S6_fd_step_sensitivity_median_abs_DeltaE2_max"],
        "S8": not res["S8_nelder_mead_gain_over_fD_max"] <= sr["S8_nelder_mead_gain_over_fD_max"],
        "S9": not res["S9_fd_noise_implied_gap_max"] <= sr["S9_fd_noise_implied_gap_max"],
    }
    if with_s7:
        import analyze_af8v3_integrated_loading as an3
        fdc = an3.fd_check3(c)
        fdc.to_csv(out / f"{stage}_fd_check.csv", index=False)
        s7 = max(worst(fdc.get("grad_rel_err_max", pd.Series([np.nan])), "max", "S7g", nan_log),
                 worst(fdc.get("hess_rel_err_max", pd.Series([np.nan])), "max", "S7h", nan_log))
        res["S7_fd_derivative_rel_err_max"] = s7
        stop["S7"] = not s7 <= sr["S7_fd_derivative_rel_err_max"]
    res["nan_counts_mapped_to_failing_side"] = nan_log
    res["stop"] = stop
    res["all_stop_rules_pass"] = not any(stop.values())
    log = []
    for r in cm.itertuples(index=False):
        s = getattr(r, "IS_draw_log", None)
        if isinstance(s, str):
            for e in json.loads(s):
                log.append({"replicate": r.replicate, "regime": r.regime, "column": r.column, "candidate": r.candidate, **e})
    pd.DataFrame(log, columns=["replicate", "regime", "column", "candidate", "draw", "f", "cold_status", "cold_grad_inf",
                               "cold_lam2_half", "cold_phi_last", "warm_status", "warm_phi", "phi_diff", "outcome"]
                 ).to_csv(out / f"{stage}_is_draw_log.csv", index=False)
    name = "pilot_numerical_audit.json" if stage == "pilot" else "full_numerical_audit.json"
    (out / name).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    write_runinfo(out / f"runinfo_{stage}_audit.json", [out / "candidate_scores.csv", OUT / "protocol.json"],
                  {"stage": f"{stage}-audit"})
    print(json.dumps({k: v for k, v in res.items() if k not in ("column_states_all_priors",)}, indent=2, default=str))
    return res


def compare_v3_pilot() -> pd.DataFrame:
    """re-pilot vs AF8-v3 pilot (c1f37a9): per cell differences of D, E2, E, IS, f_D and the start statuses."""
    a = pd.read_csv(V3_PILOT / "candidate_scores.csv", low_memory=False)
    b = pd.read_csv(OUT / "pilot" / "candidate_scores.csv", low_memory=False)
    k = ["replicate", "regime", "column", "candidate", "prior"]
    m = a.merge(b, on=k, how="outer", suffixes=("_v3", "_new"), indicator=True)
    rows = []
    for d in m.to_dict("records"):
        row = {kk: d[kk] for kk in k}
        row["present"] = d["_merge"]
        for q in ("D", "E2", "E", "IS"):
            x, y = num(pd.Series([d.get(f"{q}_v3")])).iloc[0], num(pd.Series([d.get(f"{q}_new")])).iloc[0]
            row[f"abs_diff_{q}"] = abs(y - x) if np.isfinite(x) and np.isfinite(y) else (0.0 if np.isnan(x) and np.isnan(y) else np.inf)
        fa, fb = d.get("f_D_v3"), d.get("f_D_new")
        if isinstance(fa, str) and isinstance(fb, str):
            row["max_abs_diff_f_D"] = float(np.max(np.abs(np.array(fa.split("|"), float) - np.array(fb.split("|"), float))))
        else:
            row["max_abs_diff_f_D"] = 0.0 if not isinstance(fa, str) and not isinstance(fb, str) else np.inf
        row["status_v3"], row["status_new"] = d.get("status_v3"), d.get("status_new")
        row["D_start_status_v3"], row["D_start_status_new"] = d.get("D_start_status_v3"), d.get("D_start_status_new")
        row["changed"] = bool(row["status_v3"] != row["status_new"] or row["D_start_status_v3"] != row["D_start_status_new"]
                              or max(row[f"abs_diff_{q}"] for q in ("D", "E2", "E", "IS")) > 0 or row["max_abs_diff_f_D"] > 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pilot" / "v3_pilot_vs_repilot_cells.csv", index=False)
    summ = {"cells": len(df), "changed_cells": int(df.changed.sum()),
            **{f"max_abs_diff_{q}": float(df[f"abs_diff_{q}"].max()) for q in ("D", "E2", "E", "IS")},
            "max_abs_diff_f_D": float(df.max_abs_diff_f_D.max()),
            "status_changes": int((df.status_v3 != df.status_new).sum()),
            "start_status_changes": int((df.D_start_status_v3 != df.D_start_status_new).sum())}
    (OUT / "pilot" / "v3_pilot_vs_repilot_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))
    return df


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def manifest() -> pd.DataFrame:
    import run_af6_laplace_confirmation as a6
    sm = pd.read_csv(AF6_ROOT / "seed_manifest.csv")
    fm = pd.read_csv(AF6_FULL / "fit_manifest.csv")
    rows = []
    for r in vf.FULL_REPS:
        for g in a8.REGIMES:
            case = a8.load_case(r, g)                    # asserts X, Y, train-mask hashes against AF6 fit_manifest
            m = fm[(fm.replicate == a6.label(r)) & (fm.regime == g)].iloc[0]
            s = sm[sm.replicate == a6.label(r)].iloc[0]
            st_sha = sha256_file(Path(m.state_file))
            rows.append({"replicate": a6.label(r), "regime": g, "data_seed": int(s.data_seed),
                         "thin_seed": int(s[f"thin_seed_{g}"]), "anchor": m.method, "fit_status": m.status,
                         "x_sha16": a8.sha16(case["X"]), "y_sha16": a8.sha16(case["Y"]),
                         "train_mask_sha16": a8.sha16(case["train"]),
                         "hashes_match_af6": bool(a8.sha16(case["X"]) == m.x_sha and a8.sha16(case["Y"]) == m.y_sha
                                                  and a8.sha16(case["train"]) == m.train_mask_sha),
                         "seed_matches_af6": bool(int(s.data_seed) == int(m.data_seed)),
                         "state_file": Path(m.state_file).name, "state_sha256": st_sha,
                         "state_sha256_matches_af6": st_sha == m.state_sha256, "assignment": m.assignment})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "manifest.csv", index=False)
    print(df[["hashes_match_af6", "seed_matches_af6", "state_sha256_matches_af6"]].all().to_dict(), len(df))
    return df


# --------------------------------------------------------------------------
# exploratory accuracy analysis (only after a passing full audit)
# --------------------------------------------------------------------------

def af6_columns() -> pd.DataFrame:
    import analyze_af6_laplace_confirmation as an6
    s = pd.read_csv(AF6_FULL / "column_selection.csv", keep_default_na=False, na_values=[""])
    grid = pd.DataFrame([{"replicate": f"rep{r + 1:02d}", "regime": g, "column": j, "true_model": an6.TRUE_MODELS[j]}
                         for r in vf.FULL_REPS for g in a8.REGIMES for j in a8.COUNT_COLS])
    s = grid.merge(s, on=["replicate", "regime", "column"], how="left", validate="1:1", suffixes=("", "_af6"),
                   indicator=True)
    assert (s.true_model_af6.isna() | (s.true_model_af6 == s.true_model)).all()
    miss = s._merge == "left_only"
    out = s[["replicate", "regime", "column", "true_model"]].copy()
    q = num(s.q_margin_cmp_minus_pois)
    vq = (s.q_status == "SELECTED") & np.isfinite(q)
    out["Delta_A_Q"] = q.where(vq)
    out["state_A_Q"] = np.where(miss, "MISSING", np.where(vq, np.where(q.abs() <= FLOOR, "NUMERIC_UNDECIDED", "DECIDED"), "BOTH_FAILED"))
    dl = num(s.delta)
    vb = s.status.isin(an6.FINITE_DELTA) & np.isfinite(dl)
    out["Delta_B_L6"] = dl.where(vb)
    out["state_B_L6"] = np.where(miss, "MISSING", np.where(vb, np.where(dl.abs() <= FLOOR, "NUMERIC_UNDECIDED", "DECIDED"), "FAILED_AF6"))
    jm = num(s.delta_jointmap_variant)
    vj = np.isfinite(jm)
    out["Delta_C_JM"] = jm.where(vj)
    out["state_C_JM"] = np.where(miss, "MISSING", np.where(vj, np.where(jm.abs() <= FLOOR, "NUMERIC_UNDECIDED", "DECIDED"), "FAILED_AF6"))
    return out


def column_frame(pr: pd.DataFrame, prior: str) -> pd.DataFrame:
    s = af6_columns().merge(pr[pr.prior == prior].drop(columns="prior"), on=["replicate", "regime", "column"],
                            how="left", validate="1:1")
    for m, src in (("D_MAP", "D"), ("E2", "E2"), ("E", "E"), ("IS", "IS")):
        if f"state_{src}" in s:
            s[f"state_{m}"] = s[f"state_{src}"].fillna("MISSING")
            s[f"Delta_{m}"] = s[f"Delta_{src}"]
    for m in METHODS:
        if f"state_{m}" not in s:
            continue
        dec = s[f"state_{m}"] == "DECIDED"
        und = s[f"state_{m}"] == "NUMERIC_UNDECIDED"
        x = s[f"Delta_{m}"]
        s[f"sel_{m}"] = np.where(dec, np.where(x > 0, C, P), "")
        s[f"sel_sign_{m}"] = np.where(dec | und, np.where(x > 0, C, P), "")
        s[f"ok_{m}"] = s[f"sel_{m}"] == s.true_model
        s[f"ok_sign_{m}"] = s[f"sel_sign_{m}"] == s.true_model
    # exploratory secondary hybrid: E2, Q where E2 is not decided (named separately; never the main method)
    hyb = s.state_E2 == "DECIDED"
    s["sel_E2orQ"] = np.where(hyb, s.sel_E2, s.sel_A_Q)
    s["ok_E2orQ"] = s.sel_E2orQ == s.true_model
    s["state_E2orQ"] = np.where(hyb, "DECIDED", np.where(s.state_A_Q == "DECIDED", "DECIDED_BY_Q", s.state_A_Q))
    s["Delta_E2orQ"] = np.nan                     # no common score scale: AUC not defined for the hybrid
    return s


def _auc(score, y) -> tuple[float, int]:
    from scipy.stats import rankdata
    score, y = np.asarray(score, float), np.asarray(y, bool)
    ok = np.isfinite(score)
    score, y = score[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan"), int(ok.sum())
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), int(ok.sum())


def counts(s: pd.DataFrame, m: str) -> dict:
    tc = (s.true_model == C).to_numpy()
    st = s[f"state_{m}"].to_numpy() if f"state_{m}" in s else np.array([""] * len(s))
    sel = s[f"sel_{m}"].to_numpy()
    out = {"nP": int((~tc).sum()), "nC": int(tc.sum())}
    for t, mask in (("P", ~tc), ("C", tc)):
        out[f"correct_{t}"] = int((sel[mask] == (C if t == "C" else P)).sum())
        out[f"wrong_{t}"] = int((sel[mask] == (P if t == "C" else C)).sum())
        out[f"decided_{t}"] = int(((st[mask] == "DECIDED") | (st[mask] == "DECIDED_BY_Q")).sum())
        out[f"undecided_{t}"] = int((st[mask] == "NUMERIC_UNDECIDED").sum())
        out[f"missing_{t}"] = int((st[mask] == "MISSING").sum())
        out[f"failed_{t}"] = int(mask.sum()) - out[f"decided_{t}"] - out[f"undecided_{t}"] - out[f"missing_{t}"]
        if f"ok_sign_{m}" in s:
            out[f"correct_sign_{t}"] = int(s[f"ok_sign_{m}"].to_numpy()[mask].sum())
    return out


def rates(k: dict) -> dict:
    nP, nC = k["nP"], k["nC"]
    r = {"acc_poisson": k["correct_P"] / nP, "acc_cmp2": k["correct_C"] / nC,
         "balanced": (k["correct_P"] / nP + k["correct_C"] / nC) / 2, "acc_all": (k["correct_P"] + k["correct_C"]) / (nP + nC),
         "mis_P_to_C": k["wrong_P"] / nP, "mis_C_to_P": k["wrong_C"] / nC,
         "decided_rate": (k["decided_P"] + k["decided_C"]) / (nP + nC),
         "undecided_rate": (k["undecided_P"] + k["undecided_C"]) / (nP + nC),
         "failed_rate": (k["failed_P"] + k["failed_C"]) / (nP + nC),
         "missing_rate": (k["missing_P"] + k["missing_C"]) / (nP + nC)}
    dP, dC = k["decided_P"], k["decided_C"]
    r["acc_poisson_evaluable"] = k["correct_P"] / dP if dP else float("nan")
    r["acc_cmp2_evaluable"] = k["correct_C"] / dC if dC else float("nan")
    r["balanced_evaluable"] = (r["acc_poisson_evaluable"] + r["acc_cmp2_evaluable"]) / 2
    if "correct_sign_P" in k:
        r["balanced_sign"] = (k["correct_sign_P"] / nP + k["correct_sign_C"] / nC) / 2
    return r


RATE_KEYS = ["acc_poisson", "acc_cmp2", "balanced", "acc_all", "mis_P_to_C", "mis_C_to_P"]


def boot_metrics(s: pd.DataFrame, methods, pairs_, rng_seed=BOOT_SEED):
    """dataset-cluster bootstrap: per-dataset count vectors, resampled datasets (both regimes, all columns)."""
    reps = sorted(s.replicate.unique())
    idx = np.random.default_rng(rng_seed).integers(0, len(reps), (BOOT_B, len(reps)))
    per = {}
    for m in methods:
        cnt = pd.DataFrame([counts(s[s.replicate == r], m) for r in reps])
        per[m] = cnt
    def rates_vec(cnt: pd.DataFrame):
        A = {k: cnt[k].to_numpy()[idx].sum(1).astype(float) for k in cnt.columns}
        out = {"acc_poisson": A["correct_P"] / A["nP"], "acc_cmp2": A["correct_C"] / A["nC"]}
        out["balanced"] = (out["acc_poisson"] + out["acc_cmp2"]) / 2
        out["acc_all"] = (A["correct_P"] + A["correct_C"]) / (A["nP"] + A["nC"])
        out["mis_P_to_C"], out["mis_C_to_P"] = A["wrong_P"] / A["nP"], A["wrong_C"] / A["nC"]
        return out
    rv = {m: rates_vec(per[m]) for m in methods}
    ci = {}
    for m in methods:
        for k in RATE_KEYS:
            ci[(m, k)] = (float(np.quantile(rv[m][k], .025)), float(np.quantile(rv[m][k], .975)))
    for a, b in pairs_:
        for k in RATE_KEYS:
            v = rv[a][k] - rv[b][k]
            ci[(f"{a}-{b}", k)] = (float(np.quantile(v, .025)), float(np.quantile(v, .975)))
    # AUC (fewer resamples; rank AUC on finite scores)
    rng = np.random.default_rng(rng_seed + 1)
    rows = {r: np.flatnonzero(s.replicate.to_numpy() == r) for r in reps}
    au = {m: [] for m in methods}
    for _ in range(BOOT_B_AUC):
        t = s.iloc[np.concatenate([rows[reps[i]] for i in rng.integers(0, len(reps), len(reps))])]
        for m in methods:
            au[m].append(_auc(t[f"Delta_{m}"], t.true_model == C)[0])
    for m in methods:
        v = np.array(au[m])
        ci[(m, "auc")] = (float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975)), int(np.isnan(v).sum()))
    for a, b in pairs_:
        v = np.array(au[a]) - np.array(au[b])
        ci[(f"{a}-{b}", "auc")] = (float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975)), int(np.isnan(v).sum()))
    return ci


def per_dataset_diff(s: pd.DataFrame, a: str, b: str, truth: str | None = None) -> dict:
    t = s if truth is None else s[s.true_model == truth]
    reps = sorted(s.replicate.unique())
    v = (t[f"ok_{a}"].astype(int) - t[f"ok_{b}"].astype(int)).groupby(t.replicate).sum().reindex(reps, fill_value=0)
    mb = v.to_numpy()[np.random.default_rng(BOOT_SEED).integers(0, len(reps), (BOOT_B, len(reps)))].mean(1)
    return {"mean_per_dataset": float(v.mean()), "ci_low": float(np.quantile(mb, .025)), "ci_high": float(np.quantile(mb, .975)),
            "total": int(v.sum()), "n_datasets_pos": int((v > 0).sum()), "n_datasets_zero": int((v == 0).sum()),
            "n_datasets_neg": int((v < 0).sum())}


PAIRS = [("E2", "A_Q"), ("E2", "B_L6"), ("E2", "C_JM"), ("E2", "D_MAP"), ("D_MAP", "B_L6"), ("E2", "E"), ("E", "B_L6"),
         ("IS", "E2"), ("E2orQ", "A_Q")]


def subsets(s):
    return [("dense", s[s.regime == "dense"]), ("sparse", s[s.regime == "sparse"]), ("pooled", s)]


def analyze():
    full_audit = OUT / "full" / "full_numerical_audit.json"
    if not full_audit.exists() or json.loads(full_audit.read_text(encoding="utf-8")).get("all_stop_rules_pass") is not True:
        raise SystemExit("full numerical audit missing or not passed: accuracy is NOT computed (protocol gate)")
    c = pd.read_csv(OUT / "full" / "candidate_scores.csv", low_memory=False)
    pr = pd.read_csv(OUT / "full" / "full_pairs.csv")
    main = column_frame(pr, a8.MAIN_PRIOR)
    assert len(main) == 480 and not main.duplicated(["replicate", "regime", "column"]).any()
    main.to_csv(OUT / "selection_results.csv", index=False)
    methods = METHODS + ["E2orQ"]
    acc, paired, stat = [], [], []
    for name, s in subsets(main):
        ci = boot_metrics(s, methods, PAIRS)
        for m in methods:
            k = counts(s, m)
            r = {"subset": name, "method": m, "prior": a8.MAIN_PRIOR if m in ("D_MAP", "E2", "E", "IS", "E2orQ") else "",
                 "n_columns": len(s), **k, **rates(k)}
            r["auc"], r["auc_n_finite"] = _auc(s[f"Delta_{m}"] if f"Delta_{m}" in s else np.full(len(s), np.nan), s.true_model == C)
            for q in RATE_KEYS:
                r[f"{q}_ci_low"], r[f"{q}_ci_high"] = ci[(m, q)]
            r["auc_ci_low"], r["auc_ci_high"], r["auc_boot_nan"] = ci[(m, "auc")]
            acc.append(r)
        for a, b in PAIRS:
            ka, kb = rates(counts(s, a)), rates(counts(s, b))
            r = {"subset": name, "comparison": f"{a} - {b}"}
            for q in RATE_KEYS:
                r[f"diff_{q}"] = ka[q] - kb[q]
                r[f"diff_{q}_ci_low"], r[f"diff_{q}_ci_high"] = ci[(f"{a}-{b}", q)]
            aa = _auc(s[f"Delta_{a}"], s.true_model == C)[0] if f"Delta_{a}" in s else np.nan
            ab = _auc(s[f"Delta_{b}"], s.true_model == C)[0] if f"Delta_{b}" in s else np.nan
            r["diff_auc"] = aa - ab
            r["diff_auc_ci_low"], r["diff_auc_ci_high"], r["diff_auc_boot_nan"] = ci[(f"{a}-{b}", "auc")]
            oa, ob = s[f"ok_{a}"], s[f"ok_{b}"]
            for t, tn in ((P, "poisson"), (C, "cmp2")):
                tt = s.true_model == t
                r[f"true_{tn}_right_to_wrong"] = int((ob & ~oa & tt).sum())      # b right, a wrong
                r[f"true_{tn}_wrong_to_right"] = int((~ob & oa & tt).sum())      # b wrong, a right
            paired.append(r)
            for truth in (None, P, C):
                stat.append({"subset": name, "comparison": f"{a} - {b}", "truth": truth or "all",
                             "quantity": "per-dataset #correct difference", **per_dataset_diff(s, a, b, truth)})
    pd.DataFrame(acc).to_csv(OUT / "accuracy.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "paired_comparison.csv", index=False)
    pd.DataFrame(stat).to_csv(OUT / "statistical_analysis.csv", index=False)
    # prior sensitivity (exploratory): D, E2, E for every prior; no CIs
    ps = []
    for prior in a8.PRIORS:
        s_all = column_frame(pr, prior)
        for name, s in subsets(s_all):
            ref = main if name == "pooled" else main[main.regime == name]
            for m in ("D_MAP", "E2", "E"):
                k = counts(s, m)
                tc = s.true_model == C
                ps.append({"prior": prior, "kind": a8.PRIORS[prior][0], "scale": a8.PRIORS[prior][1], "subset": name,
                           "method": m, **rates(k), "auc": _auc(s[f"Delta_{m}"], tc)[0],
                           "median_Delta_true_poisson": float(s.loc[~tc, f"Delta_{m}"].median()),
                           "median_Delta_true_cmp2": float(s.loc[tc, f"Delta_{m}"].median()),
                           "selection_changes_vs_main_prior": int((s[f"sel_{m}"].to_numpy() != ref[f"sel_{m}"].to_numpy()).sum())})
    pd.DataFrame(ps).to_csv(OUT / "prior_sensitivity.csv", index=False)
    # approximation diagnostics (main prior, per column)
    ad = main[["replicate", "regime", "column", "true_model", "state_E2", "state_E", "state_IS", "Delta_E2", "Delta_E",
               "Delta_IS", "IS_se_delta", "Delta_D_MAP", "Delta_B_L6", "Delta_E2_logdet_f", "Delta_E_logdet_schur_f",
               "Delta_E_logdet_ZZ"] + [f"Delta_E2_h{h:g}" for h in vf.FD_H_HESS_ALT]].copy()
    ad["E_minus_E2"] = ad.Delta_E - ad.Delta_E2
    ad["term_f_curvature"] = 0.5 * (ad.Delta_E2_logdet_f - ad.Delta_E_logdet_schur_f)
    ad["term_mode_and_Z"] = ad.E_minus_E2 - ad.term_f_curvature
    ad["E2_minus_IS"] = ad.Delta_E2 - ad.Delta_IS
    ad["E2_minus_IS_over_se"] = ad.E2_minus_IS / ad.IS_se_delta
    ad["fd_step_sensitivity"] = ad[[f"Delta_E2_h{h:g}" for h in vf.FD_H_HESS_ALT]].sub(ad.Delta_E2, axis=0).abs().max(axis=1)
    ad["sign_E_ne_E2"] = (ad.Delta_E * ad.Delta_E2) < 0
    ad["sign_IS_ne_E2"] = (ad.Delta_IS * ad.Delta_E2) < 0
    ad["ambiguous_label_free"] = ad.state_E2.isin(["DECIDED", "NUMERIC_UNDECIDED"]) & (
        (ad.Delta_E2.abs() <= 1.0) | ad.sign_E_ne_E2 | ad.sign_IS_ne_E2 | (ad.state_E2 == "NUMERIC_UNDECIDED"))
    ad["IS_sign_resolved_2se"] = (ad.Delta_IS.abs() > 2 * ad.IS_se_delta)
    ad.to_csv(OUT / "approximation_diagnostics.csv", index=False)
    figures(main, pd.DataFrame(ps), ad)
    write_runinfo(OUT / "runinfo.json",
                  [OUT / "full" / "candidate_scores.csv", OUT / "full" / "full_pairs.csv", OUT / "protocol.json",
                   AF6_FULL / "column_selection.csv", OUT / "manifest.csv", full_audit],
                  {"stage": "analysis", "bootstrap": {"seed": BOOT_SEED, "B": BOOT_B, "B_auc": BOOT_B_AUC}})
    a = pd.DataFrame(acc)
    print(a[a.subset == "pooled"][["method", "acc_poisson", "acc_cmp2", "balanced", "mis_P_to_C", "mis_C_to_P", "auc",
                                   "failed_rate", "undecided_rate"]].round(3).to_string())


def figures(main, ps, ad):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharey="row")
    for i, g in enumerate(a8.REGIMES):
        s = main[main.regime == g]
        for k, (m, lab) in enumerate((("A_Q", "A: Q margin"), ("B_L6", "B: AF6 Laplace"), ("D_MAP", "D: prior MAP"),
                                      ("E2", "E2: nested Laplace (main)"))):
            ax = axes[i, k]
            for t, col in ((P, "#1f77b4"), (C, "#d62728")):
                x = s.loc[s.true_model == t, f"Delta_{m}"].dropna().clip(-20, 20)
                ax.hist(x, bins=40, range=(-20, 20), alpha=.55, color=col, label=f"true {t}")
            ax.axvline(0, color="k", lw=.8)
            ax.set_title(f"{g}: {lab}", fontsize=9)
            if i == 1:
                ax.set_xlabel("Delta = CMP2 - Poisson (clipped +-20)")
        axes[i, 0].set_ylabel("columns")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_delta_by_truth.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, g in zip(axes, ("dense", "sparse", "pooled")):
        t = ps[(ps.subset == g) & (ps.method == "E2")]
        x = np.arange(len(t))
        ax.plot(x, t.acc_poisson, "o-", label="true Poisson")
        ax.plot(x, t.acc_cmp2, "s-", label="true CMP2")
        ax.plot(x, t.balanced, "k^-", label="balanced")
        ax.set_xticks(x, t.prior, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{g}: E2 by prior (main {a8.MAIN_PRIOR})", fontsize=9)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_prior_sensitivity.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for t, col in ((P, "#1f77b4"), (C, "#d62728")):
        q = ad[ad.true_model == t]
        ax.scatter(q.Delta_E2.clip(-15, 15), q.Delta_E.clip(-15, 15), s=8, alpha=.6, color=col, label=f"true {t}")
    ax.axhline(0, color="k", lw=.6)
    ax.axvline(0, color="k", lw=.6)
    ax.plot([-15, 15], [-15, 15], "k:", lw=.6)
    ax.set_xlabel("Delta E2 (nested Laplace)")
    ax.set_ylabel("Delta E (joint Laplace)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_E_vs_E2.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--pilot-audit" in argv:
        numerical_audit("pilot", vf.PILOT_REPS, with_s7=True)
        compare_v3_pilot()
    if "--manifest" in argv:
        manifest()
    if "--full-audit" in argv:
        numerical_audit("full", vf.FULL_REPS, with_s7=False)
    if "--analyze" in argv:
        analyze()
    if "--compare-full-repilot" in argv:
        compare_full_vs_repilot()
    return 0


def compare_full_vs_repilot() -> dict:
    """full-run rep01-03 vs the re-pilot (same code; OMP_NUM_THREADS 1 vs 2): per-cell differences and sign changes
    of Delta for D, E2, E, IS. Added after the independent audit (W2) so that this comparison is script output."""
    a = pd.read_csv(OUT / "pilot" / "candidate_scores.csv", low_memory=False)
    b = pd.read_csv(OUT / "full" / "candidate_scores.csv", low_memory=False)
    b = b[b.replicate.isin(a.replicate.unique())]
    k = ["replicate", "regime", "column", "candidate", "prior"]
    m = a.merge(b, on=k, how="outer", suffixes=("_pilot", "_full"), indicator=True)
    summ: dict = {"cells_pilot": len(a), "cells_full_same_reps": len(b), "unmatched": int((m["_merge"] != "both").sum())}
    m = m[m["_merge"] == "both"]
    for q in ("D", "E2", "E", "IS"):
        x, y = num(m[f"{q}_pilot"]), num(m[f"{q}_full"])
        both = np.isfinite(x) & np.isfinite(y)
        summ[f"max_abs_diff_{q}"] = float(np.max(np.abs(x[both] - y[both]))) if both.any() else None
        summ[f"finite_mismatch_{q}"] = int((np.isfinite(x) != np.isfinite(y)).sum())
    summ["status_changes"] = int((m.status_pilot != m.status_full).sum())
    kk = ["replicate", "regime", "column", "prior"]
    sign_changes = {}
    for q in ("D", "E2", "E", "IS"):
        d = {}
        for tag in ("pilot", "full"):
            s = m[kk + ["candidate", f"{q}_{tag}"]].copy()
            s["v"] = num(s[f"{q}_{tag}"])
            w = s.pivot_table(index=kk, columns="candidate", values="v", aggfunc="first")
            d[tag] = np.sign(w[C] - w[P])
        j = pd.concat(d, axis=1).dropna()
        sign_changes[q] = {"pairs_compared": len(j), "sign_changes": int((j["pilot"] != j["full"]).sum())}
    summ["Delta_sign_changes"] = sign_changes
    (OUT / "full" / "full_vs_repilot_rep01_03_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    write_runinfo(OUT / "full" / "runinfo_full_vs_repilot.json",
                  [OUT / "pilot" / "candidate_scores.csv", OUT / "full" / "candidate_scores.csv"],
                  {"stage": "full_vs_repilot", "command": "analyze_af8v3_full_integrated_loading.py --compare-full-repilot"})
    print(json.dumps(summ, indent=2))
    return summ


if __name__ == "__main__":
    raise SystemExit(main())
