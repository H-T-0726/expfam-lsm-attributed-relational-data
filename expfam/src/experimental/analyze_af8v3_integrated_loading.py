"""AF8-v3 analysis (numerical only: no selection accuracy, no AUC, no labels).

--v2-summary     : re-aggregate the AF8-v2 failures from its saved CSVs (by candidate / start / status)
--make-fixtures  : regenerate the EXACT 9 failed IS draws of the v2 pilot (v2 code, v2 seeds) and record the v2
                   inner state there (status, ||g||, Newton decrement, min eigenvalue, phi, S change under one Newton step)
--regression     : evaluate v3 on the known failures (9 draws, 2 cells without a valid outer start)
--pilot-audit    : stop rules S1-S9 (+ S4 with the v2 definition), diagnostics, v2-vs-v3 comparison
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_af8_integrated_loading as a8        # noqa: E402
import run_af8v2_integrated_loading as v2      # noqa: E402
import run_af8v3_integrated_loading as v3      # noqa: E402

V2P = v2.OUT / "pilot"
REG = v3.OUT / "regression"
P, C = a8.P, a8.C
KEYS = ["replicate", "regime", "column", "prior"]


def _case(rep_label, g, cache):
    ri = int(rep_label[3:]) - 1
    if (ri, g) not in cache:
        cs = a8.load_case(ri, g)
        cs["regime"] = g
        cache[(ri, g)] = cs
    return ri, cache[(ri, g)]


# --------------------------------------------------------------------------
# v2 re-aggregation (saved values only)
# --------------------------------------------------------------------------

def v2_summary():
    REG.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(V2P / "candidate_scores.csv")
    d = pd.read_csv(V2P / "pilot_failure_diagnosis_v2.csv")
    m = c[c.prior == a8.MAIN_PRIOR]
    isf = pd.to_numeric(m.IS_failed, errors="coerce").fillna(0)
    st = []
    for col, meth in (("D_start_status", "D"), ("E_start_status", "E")):
        for _, r in c.iterrows():
            if isinstance(r[col], str):
                for part in r[col].split("|"):
                    nm, s = part.split(":", 1)
                    st.append({"method": meth, "candidate": r.candidate, "start": nm, "status": s})
    st = pd.DataFrame(st)
    a = d[d.kind == "IS_failed_draw"]
    o = d[d.kind == "no_valid_outer_start"]
    res = {
        "source": "AF8-v2 saved CSVs (a5cffa1): pilot/candidate_scores.csv, pilot/pilot_failure_diagnosis_v2.csv",
        "IS_failed_draws": int(isf.sum()), "IS_draws": int(pd.to_numeric(m.IS_draws, errors="coerce").sum()),
        "IS_cells_with_failed_draws": int((isf > 0).sum()), "IS_main_cells": int(len(m)),
        "IS_main_cells_with_IS": int(m.IS_draws.notna().sum()),
        "IS_failed_cells_by_candidate": m[isf > 0].candidate.value_counts().to_dict(),
        "IS_failed_cells_by_regime": m[isf > 0].regime.value_counts().to_dict(),
        "cells_E2_not_ok": int((c.E2_status != "OK").sum()),
        "cells_E2_not_ok_by_candidate": c[c.E2_status != "OK"].candidate.value_counts().to_dict(),
        "start_status_by_method_candidate": {f"{k[0]}:{k[1]}:{k[2]}": int(v) for k, v in
                                             st.groupby(["method", "candidate", "status"]).size().items()},
        "valid_D_starts_per_cell": c.D_n_valid_starts.value_counts().sort_index().to_dict(),
        "failed_draw_cold_grad_inf_last": a.cold100_grad_inf_last.describe().to_dict() if "cold100_grad_inf_last" in a else None,
        "failed_draw_warm_minus_cold200_phi_absmax": float(a.diag_cold200_minus_warm_phi.abs().max()),
        "failed_draw_warm_min_eig_HZZ": a.diag_warmstart_min_eig.describe().to_dict(),
        "failed_draw_mahalanobis_sq": a.mahalanobis_sq_wrt_hess.describe().to_dict(),
        "outer_fail_grad_inf_last": o.outer_grad_inf.describe().to_dict(),
        "outer_fail_fd_grad_by_h": o[[x for x in o.columns if x.startswith("fd_grad_inf_h")]].to_dict("records"),
        "outer_fail_T_spread_per_cell": {f"{k[0]}-{k[1]}": float(g.T_last.max() - g.T_last.min()) for k, g in o.groupby(["replicate", "column"])},
        "note": "labels (true family) are not used; candidate = fitted family of the cell",
    }
    (REG / "v2_failure_summary.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=2, default=str))


# --------------------------------------------------------------------------
# fixtures: exact v2 failed draws
# --------------------------------------------------------------------------

def make_fixtures():
    """re-runs the v2 D/E2 stage (v2 code, deterministic) for the 9 cells to recover f_D, the FD Hessian and thus the
    exact v2 IS draws; records the failed draws at full precision and the v2 inner state there."""
    from scipy.stats import multivariate_t
    REG.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(V2P / "candidate_scores.csv")
    sel = c[(c.prior == a8.MAIN_PRIOR) & (pd.to_numeric(c.IS_failed, errors="coerce").fillna(0) > 0)]
    cache, rows = {}, []
    for _, r in sel.iterrows():
        ri, case = _case(r.replicate, r.regime, cache)
        j, cand = int(r.column), r.candidate
        J, _ = a8.joint_for(case, j, cand, a8.MAIN_PRIOR)
        Za, starts = v3.make_starts(case, j, cand)
        L = v2.Local(J, Za)
        de = v2.method_D_E2(L, starts)
        assert abs(de["D"] - r.D) <= 1e-9, ("v2 D not reproduced", de["D"], r.D)
        ci = v3.case_index(ri, r.regime, j, cand)
        q = multivariate_t(loc=de["f_D"], shape=np.linalg.inv(de["hess_f"]), df=v2.IS_DF,
                           seed=np.random.default_rng(v2.IS_SEED_BASE + ci))
        fs = np.atleast_2d(q.rvs(size=v2.IS_DRAWS))
        bad = [s for s in range(v2.IS_DRAWS) if not L.S(fs[s]).ok]
        for s in bad:
            f = fs[s]
            vgh = L._vgh_Z(f)
            res = v2.newton_max(vgh, Za.ravel(), v2.INNER_GTOL, v2.INNER_MAXITER, v2.INNER_STEP_CAP)
            v, g, H = vgh(res.x)
            Hs = 0.5 * (H + H.T)
            ok_pd, ld, _ = v2.chol_logdet(Hs)
            p = np.linalg.solve(Hs, g)
            vt, _, Ht = vgh(res.x + p)
            _, ldt, _ = v2.chol_logdet(0.5 * (Ht + Ht.T))
            rows.append({"replicate": r.replicate, "regime": r.regime, "column": j, "candidate": cand,
                         "case_index": ci, "draw": s, "n_failed_in_cell_rerun": len(bad), "n_failed_in_cell_v2": int(r.IS_failed),
                         "f": "|".join(repr(float(a)) for a in f), "fD": "|".join(repr(float(a)) for a in de["f_D"]),
                         "v2_D_reproduced": True, "v2_status": res.status, "v2_iters": res.iters,
                         "v2_rejected_trials": res.n_rejected_trials, "v2_grad_inf": float(np.max(np.abs(g))),
                         "v2_lam2_half": 0.5 * float(g @ p), "v2_min_eig_HZZ": float(np.linalg.eigvalsh(Hs)[0]),
                         "v2_phi": v, "v2_phi_ulp": float(np.spacing(abs(v))),
                         "v2_phi_change_full_newton_step": vt - v,
                         "v2_S_change_full_newton_step": (vt - 0.5 * ldt) - (v - 0.5 * ld), "v2_HZZ_pd": ok_pd})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(REG / "known_failed_draws_v2.csv", index=False)


# --------------------------------------------------------------------------
# regression: v3 on the known failures
# --------------------------------------------------------------------------

def regression():
    fx = pd.read_csv(REG / "known_failed_draws_v2.csv")
    cache, rows = {}, []
    for _, r in fx.iterrows():
        ri, case = _case(r.replicate, r.regime, cache)
        j, cand = int(r.column), r.candidate
        J, _ = a8.joint_for(case, j, cand, a8.MAIN_PRIOR)
        Za = np.asarray(case["st"]["Zmode"], float)
        f = np.array([float(a) for a in r.f.split("|")])
        L = v3.Local3(J, Za)
        t0 = time.perf_counter()
        s = L.S(f)
        # independent reference: v2 warm start from a converged nearby mode is NOT used; reference = v3 from a
        # second deterministic start (anchor + 0.05) -> same certified mode?
        s2 = L.solve_from(f, Za + 0.05)
        rows.append({"kind": "IS_draw", "replicate": r.replicate, "regime": r.regime, "column": j, "candidate": cand,
                     "draw": int(r.draw), "v2_status": r.v2_status, "v3_status": s.status, "v3_iters": s.iters,
                     "v3_grad_inf": s.grad_inf, "v3_lam2_half": s.lam2_half, "v3_lap_step_change": s.lap_step_change,
                     "v3_min_eig": s.min_eig, "v3_S": s.S, "alt_start_status": s2.status,
                     "alt_start_S_diff": (s2.S - s.S) if (s.ok and s2.ok) else float("nan"),
                     "seconds": round(time.perf_counter() - t0, 2)})
        print(rows[-1], flush=True)
    for rep, g, j, cand, pr in (("rep01", "sparse", 4, C, "A_sf2_0.667"), ("rep02", "sparse", 5, C, a8.MAIN_PRIOR)):
        ri, case = _case(rep, g, cache)
        J, _ = a8.joint_for(case, j, cand, pr)
        Za, starts = v3.make_starts(case, j, cand)
        L = v3.Local3(J, Za)
        t0 = time.perf_counter()
        de = v3.method_D_E2(L, starts)
        rows.append({"kind": "outer_cell", "replicate": rep, "regime": g, "column": j, "candidate": cand, "prior": pr,
                     "v2_status": "NO_VALID_START", "v3_status": de["E2_status"], "D_start_status": de["D_start_status"],
                     "D_bfgs_status": de["D_bfgs_status"], "D_multistart": de.get("D_multistart"),
                     "D_n_valid_starts": de.get("D_n_valid_starts"), "D_lam2_half_f": de.get("D_lam2_half_f"),
                     "D_newton_gain_f": de.get("D_newton_gain_f"), "D_grad_inf": de.get("D_grad_inf"),
                     "D_min_eig_f": de.get("D_min_eig_f"), "D_same_point_value_spread": de.get("D_same_point_value_spread"),
                     "n_S_not_ok": int(sum(not s.ok for s in L.cache.values())), "n_S_evals": L.n_eval,
                     "seconds": round(time.perf_counter() - t0, 2)})
        if de.get("f_D") is not None:
            oc = v3.outer_checks(L, de["f_D"], de["D"], de["hess_f"])
            rows[-1].update(oc)
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(REG / "known_failures_v3_regression.csv", index=False)


# --------------------------------------------------------------------------
# pilot audit
# --------------------------------------------------------------------------

import analyze_af8v2_integrated_loading as an2      # noqa: E402  (pair(): Delta per method, label-free states)

MIS = ["SAME_POINT_VALUE_MISMATCH", "MULTIMODAL_NONNEGLIGIBLE", "MULTIMODAL_DOMINANT"]


def fd_check3(c: pd.DataFrame) -> pd.DataFrame:
    """S7 as v2 (column 3, main prior, every pilot case), Z from the v3 certified S."""
    rows = []
    rng = np.random.default_rng(7)
    for r in v3.PILOT_REPS:
        for g in a8.REGIMES:
            case = a8.load_case(r, g)
            case["regime"] = g
            for cand in a8.CANDS:
                row = c[(c.replicate == case["label"]) & (c.regime == g) & (c.column == 3) & (c.candidate == cand)
                        & (c.prior == a8.MAIN_PRIOR)]
                if row.empty or not isinstance(row.iloc[0].get("f_D"), str):
                    rows.append({"replicate": case["label"], "regime": g, "candidate": cand, "status": "NO_F_D"})
                    continue
                f = np.array([float(v) for v in row.iloc[0].f_D.split("|")])
                J, _ = a8.joint_for(case, 3, cand, a8.MAIN_PRIOR)
                s = v3.Local3(J, np.asarray(case["st"]["Zmode"], float)).S(f)
                Z0, f0 = s.Z + 0.01, f + 0.01
                n, k = Z0.shape
                u0 = np.concatenate([Z0.ravel(), f0])

                def split(u):
                    return u[:n * k].reshape(n, k), u[n * k:]
                gr = np.concatenate([a.ravel() for a in J.grad_u(Z0, f0)])
                H = J.negH_u(Z0, f0)
                h = 1e-5
                eg, eh = 0.0, 0.0
                for q in range(len(u0)):
                    e = np.zeros(len(u0))
                    e[q] = h
                    fd = (J.psi(*split(u0 + e)) - J.psi(*split(u0 - e))) / (2 * h)
                    eg = max(eg, abs(fd - gr[q]) / max(1.0, abs(gr[q])))
                for q in list(rng.choice(n * k, 9, replace=False)) + [n * k, n * k + 1, n * k + 2]:
                    e = np.zeros(len(u0))
                    e[q] = h
                    gp = np.concatenate([a.ravel() for a in J.grad_u(*split(u0 + e))])
                    gm = np.concatenate([a.ravel() for a in J.grad_u(*split(u0 - e))])
                    eh = max(eh, float(np.max(np.abs(-(gp - gm) / (2 * h) - H[:, q])) / max(1.0, np.max(np.abs(H[:, q])))))
                rows.append({"replicate": case["label"], "regime": g, "candidate": cand, "status": "OK",
                             "grad_rel_err_max": eg, "hess_rel_err_max": eh})
    return pd.DataFrame(rows)


def _num(x):
    return pd.to_numeric(x, errors="coerce")


def pilot_audit() -> dict:
    out = v3.OUT / "pilot"
    c = pd.read_csv(out / "candidate_scores.csv")
    pr = an2.pair(c)
    pr.to_csv(out / "pilot_pairs_v3.csv", index=False)
    main = pr[pr.prior == a8.MAIN_PRIOR]
    cm = c[c.prior == a8.MAIN_PRIOR]
    cmi = cm[cm.IS_draws.notna()]
    failed = c.status.isin(["FAILED", "EXCEPTION"]) | (c.E2_status != "OK")
    dis = c.D_multistart.isin(MIS)
    ok_is = main.dropna(subset=["Delta_IS", "Delta_E2"])
    d_e2_is = (ok_is.Delta_E2 - ok_is.Delta_IS).abs()
    fdsens = main[[f"Delta_E2_h{h:g}" for h in v3.FD_H_HESS_ALT]].sub(main.Delta_E2, axis=0).abs().max(axis=1)
    fdc = fd_check3(c)
    fdc.to_csv(out / "pilot_fd_check_v3.csv", index=False)
    inv = []
    for col in ("D_start_status", "E_start_status"):
        for _, r in c.iterrows():
            if isinstance(r[col], str):
                for part in r[col].split("|"):
                    nm, st = part.split(":", 1)
                    inv.append({"method": col[0], "candidate": r.candidate, "start": nm, "status": st})
    inv = pd.DataFrame(inv)
    log = []
    for _, r in cmi.iterrows():
        if isinstance(r.get("IS_draw_log"), str):
            for e in json.loads(r.IS_draw_log):
                log.append({"replicate": r.replicate, "regime": r.regime, "column": r.column, "candidate": r.candidate, **e})
    pd.DataFrame(log).to_csv(out / "pilot_is_draw_log_v3.csv", index=False)
    isf, isr = _num(cmi.IS_failed).fillna(0), _num(cmi.IS_rescued).fillna(0)
    s3 = float(d_e2_is.median()) if len(ok_is) else float("nan")
    res = {
        "cells": len(c), "cells_by_status": c.status.value_counts().to_dict(),
        "cells_E2_not_ok_by_candidate": c[failed].candidate.value_counts().to_dict(),
        "S1_cell_numerical_failure_share": float(failed.mean()),
        "S2_multistart_disagreement_share": float(dis.mean()),
        "S2_detail": c.D_multistart.value_counts().to_dict(),
        "S2b_cells_with_fewer_than_2_certified_starts_share": float((_num(c.D_n_valid_starts).fillna(0) < 2).mean()),
        "D_valid_starts_per_cell": {int(k): int(v) for k, v in _num(c.D_n_valid_starts).fillna(0).astype(int).value_counts().sort_index().items()},
        "E_multistart_detail_diagnostic": c.E_multistart.value_counts().to_dict(),
        "start_status_counts": {f"{m}:{cd}:{s}": int(n) for (m, cd, s), n in inv.groupby(["method", "candidate", "status"]).size().items()},
        "D_bfgs_phase_status_counts": pd.Series([p.split(":", 1)[1] for s in c.D_bfgs_status.dropna() for p in s.split("|")]).value_counts().to_dict(),
        "invalid_start_share_D": float((inv[inv.method == "D"].status != "CERT_OK").mean()),
        "invalid_start_share_E": float((inv[inv.method == "E"].status != "OK").mean()),
        "main_cells": len(cm), "main_cells_with_IS": len(cmi),
        "main_pairs": len(main), "main_pairs_with_IS_both_valid": len(ok_is),
        "main_pairs_excluded_from_S3": int(len(main) - len(ok_is)),
        "S3_median_abs_DeltaE2_minus_DeltaIS": s3,
        "S3_max_abs_DeltaE2_minus_DeltaIS": float(d_e2_is.max()) if len(ok_is) else float("nan"),
        "median_abs_DeltaE2": float(main.Delta_E2.abs().median()),
        "S4_IS_cells_with_failed_draws_share": float((isf > 0).sum() / len(cm)),
        "S4_v2_definition_cells_with_any_cold_start_failure_share": float(((isf + isr) > 0).sum() / len(cm)),
        "S4r_IS_cells_with_any_rescued_draw_share": float((isr > 0).sum() / len(cm)),
        "IS_draws_total": int(_num(cmi.IS_draws).sum()), "IS_failed_draws_total": int(isf.sum()),
        "IS_rescued_draws_total": int(isr.sum()),
        "IS_draw_failure_share": float(isf.sum() / _num(cmi.IS_draws).sum()),
        "S4b_IS_min_ess": float(_num(cmi.IS_ess).min()),
        "IS_median_se_delta": float(main.IS_se_delta.median()),
        "S5_determinism_max_abs_diff": float(_num(cm.determinism_abs_diff).max()),
        "S6_median_fd_step_sensitivity_DeltaE2": float(fdsens.median()),
        "S6_max_fd_step_sensitivity_DeltaE2": float(fdsens.max()),
        "S7_fd_grad_rel_err_max": float(fdc.grad_rel_err_max.max()) if "grad_rel_err_max" in fdc else float("nan"),
        "S7_fd_hess_rel_err_max": float(fdc.hess_rel_err_max.max()) if "hess_rel_err_max" in fdc else float("nan"),
        "S8_nelder_mead_gain_over_fD_max": float(_num(cm.nm_gain_over_fD).max()),
        "S8_nelder_mead_gain_over_fD_min": float(_num(cm.nm_gain_over_fD).min()),
        "nm_invalid_evals_total": int(_num(cm.nm_invalid_evals).fillna(0).sum()),
        "S9_fd_noise_implied_gap_max": float(_num(cm.fd_noise_implied_gap).max()),
        "fd_grad_spread_median": float(_num(cm.fd_grad_spread).median()),
        "fd_grad_spread_max": float(_num(cm.fd_grad_spread).max()),
        "outer_lam2_half_max": float(_num(c.D_lam2_half_f).max()),
        "outer_newton_gain_max": float(_num(c.D_newton_gain_f).max()),
        "outer_grad_inf_median": float(_num(c.D_grad_inf).median()),
        "outer_grad_inf_max": float(_num(c.D_grad_inf).max()),
        "outer_min_eig_min": float(_num(c.D_min_eig_f).min()),
        "outer_cond_max": float(_num(c.D_cond_f).max()),
        "E_lam2_half_max": float(_num(c.E_lam2_half).max()),
        "E_lap_step_change_max": float(_num(c.E_lap_step_change).max()),
        "E_min_eig_min": float(_num(c.E_min_eig).min()),
        "n_S_evals_total": int(_num(c.n_S_evals).sum()), "n_S_not_ok_total": int(_num(c.n_S_not_ok).sum()),
        "diag_median_abs_DeltaE_minus_DeltaE2": float((main.Delta_E - main.Delta_E2).abs().median()),
        "diag_max_abs_DeltaE_minus_DeltaE2": float((main.Delta_E - main.Delta_E2).abs().max()),
        "diag_median_abs_DeltaE_minus_DeltaIS": float((ok_is.Delta_E - ok_is.Delta_IS).abs().median()) if len(ok_is) else float("nan"),
        "diag_sign_disagree_E_vs_E2": int(((main.Delta_E * main.Delta_E2) < 0).sum()),
        "diag_pairs_E_and_E2_finite": int((main.Delta_E.notna() & main.Delta_E2.notna()).sum()),
        "diag_sign_disagree_E2_vs_IS": int(((ok_is.Delta_E2 * ok_is.Delta_IS) < 0).sum()),
        "selection_state_counts_label_free": {f"{k[0]}:{k[1]}": int(v) for k, v in pr.groupby(["prior", "selection_state"]).size().items()},
        "seconds_total": float(c.seconds.sum()), "seconds_median_main": float(cm.seconds.median()),
        "seconds_median_other": float(c[c.prior != a8.MAIN_PRIOR].seconds.median()),
    }
    sr = v3.STOP_RULES_V3
    s7 = max(res["S7_fd_grad_rel_err_max"], res["S7_fd_hess_rel_err_max"])
    res["stop"] = {   # a NaN statistic never passes
        "S1": not res["S1_cell_numerical_failure_share"] <= sr["S1_cell_numerical_failure_share_max"],
        "S2": not res["S2_multistart_disagreement_share"] <= sr["S2_multistart_disagreement_share_max"],
        "S2b": not res["S2b_cells_with_fewer_than_2_certified_starts_share"] <= sr["S2b_cells_with_fewer_than_2_certified_starts_share_max"],
        "S3": not s3 <= sr["S3_main_median_abs_DeltaE2_minus_DeltaIS_max"],
        "S3b": not s3 < 0.5 * res["median_abs_DeltaE2"],
        "S4": not res["S4_IS_cells_with_failed_draws_share"] <= sr["S4_IS_cells_with_failed_draws_share_max"],
        "S4r": not res["S4r_IS_cells_with_any_rescued_draw_share"] <= sr["S4r_IS_cells_with_any_rescued_draw_share_max"],
        "S4b": not res["S4b_IS_min_ess"] >= sr["S4b_IS_min_ess"],
        "S5": not res["S5_determinism_max_abs_diff"] <= sr["S5_determinism_max_abs_diff"],
        "S6": not res["S6_median_fd_step_sensitivity_DeltaE2"] <= sr["S6_fd_step_sensitivity_median_abs_DeltaE2_max"],
        "S7": not s7 <= sr["S7_fd_derivative_rel_err_max"],
        "S8": not res["S8_nelder_mead_gain_over_fD_max"] <= sr["S8_nelder_mead_gain_over_fD_max"],
        "S9": not res["S9_fd_noise_implied_gap_max"] <= sr["S9_fd_noise_implied_gap_max"],
    }
    res["all_stop_rules_pass"] = not any(res["stop"].values())
    v2a = json.loads((V2P / "pilot_numerical_audit_v2.json").read_text(encoding="utf-8"))
    comp = pd.DataFrame([
        {"quantity": "cells", "v2": v2a["cells"], "v3": len(c)},
        {"quantity": "S1 share of cells with E2 not OK", "v2": v2a["S1_cell_numerical_failure_share"], "v3": res["S1_cell_numerical_failure_share"]},
        {"quantity": "S2 disagreement among valid starts (v2 valid = GRAD_OK + PD FD Hessian; v3 valid = CERT_OK)", "v2": v2a["S2_multistart_disagreement_share"], "v3": res["S2_multistart_disagreement_share"]},
        {"quantity": "invalid D start share", "v2": v2a["invalid_start_share_D"], "v3": res["invalid_start_share_D"]},
        {"quantity": "invalid E start share (v2 valid = OK/OK_FLOOR; v3 valid = certified OK)", "v2": v2a["invalid_start_share_E"], "v3": res["invalid_start_share_E"]},
        {"quantity": "S4 with the v2 definition: main cells with >=1 cold-start failed draw / 72", "v2": v2a["S4_IS_cells_with_failed_draws_share"], "v3": res["S4_v2_definition_cells_with_any_cold_start_failure_share"]},
        {"quantity": "S4 v3: main cells with >=1 draw failed after the same-mode rescue / 72", "v2": None, "v3": res["S4_IS_cells_with_failed_draws_share"]},
        {"quantity": "IS failed draws / total", "v2": f"{v2a['IS_failed_draws_total']}/{v2a['IS_draws_total']}", "v3": f"{res['IS_failed_draws_total']}/{res['IS_draws_total']}"},
        {"quantity": "S3 median |dE2-dIS| (number of pairs)", "v2": f"{v2a['S3_median_abs_DeltaE2_minus_DeltaIS']:.4g} ({v2a['main_pairs_with_IS_both_OK']})", "v3": f"{s3:.4g} ({len(ok_is)})"},
        {"quantity": "median |dE-dE2|", "v2": v2a["diag_median_abs_DeltaE_minus_DeltaE2"], "v3": res["diag_median_abs_DeltaE_minus_DeltaE2"]},
        {"quantity": "S4b min ESS", "v2": v2a["S4_IS_min_ess"], "v3": res["S4b_IS_min_ess"]},
        {"quantity": "S6 median FD-step sensitivity of dE2", "v2": v2a["S6_median_fd_step_sensitivity_DeltaE2"], "v3": res["S6_median_fd_step_sensitivity_DeltaE2"]},
        {"quantity": "CPU seconds total", "v2": v2a["seconds_total"], "v3": res["seconds_total"]},
    ])
    comp.to_csv(out / "v2_vs_v3_comparison.csv", index=False)
    (out / "pilot_numerical_audit_v3.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("selection_state_counts_label_free", "start_status_counts")}, indent=2, default=str))
    print(comp.to_string())
    return res


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--v2-summary" in argv:
        v2_summary()
    if "--make-fixtures" in argv:
        make_fixtures()
    if "--regression" in argv:
        regression()
    if "--pilot-audit" in argv:
        pilot_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
