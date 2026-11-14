"""AF8-v2 pilot audit (numerical only: no selection accuracy, no AUC, no labels).

--pilot-audit : stop rules S1-S7, diagnostics, AF8-vs-v2 comparison
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_af8_integrated_loading as a8        # noqa: E402
import run_af8v2_integrated_loading as v2      # noqa: E402

P, C = a8.P, a8.C
KEYS = ["replicate", "regime", "column", "prior"]


def pair(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in c.groupby(KEYS):
        rp = g[g.candidate == P].iloc[0].to_dict() if (g.candidate == P).any() else None
        rc = g[g.candidate == C].iloc[0].to_dict() if (g.candidate == C).any() else None
        st, _, d = v2.selection_state(rp, rc)
        r = dict(zip(KEYS, k), selection_state=st, Delta_E2=d)
        both = rp is not None and rc is not None
        for m in ("E", "IS", "D"):
            if both and m in rp and m in rc:
                r[f"Delta_{m}"] = pd.to_numeric(rc.get(m), errors="coerce") - pd.to_numeric(rp.get(m), errors="coerce")
        for h in v2.FD_H_HESS_ALT:
            key = f"E2_h{h:g}"
            if both and key in rp and key in rc:
                r[f"Delta_{key}"] = pd.to_numeric(rc.get(key), errors="coerce") - pd.to_numeric(rp.get(key), errors="coerce")
        if both and "IS_mc_se" in rp:
            r["IS_se_delta"] = float(np.hypot(pd.to_numeric(rp.get("IS_mc_se"), errors="coerce"),
                                              pd.to_numeric(rc.get("IS_mc_se"), errors="coerce")))
        rows.append(r)
    return pd.DataFrame(rows)


def fd_check(c: pd.DataFrame) -> pd.DataFrame:
    """S7: analytic vs FD gradient/Hessian of Psi near the reported optimum (column 3, main prior, every pilot case)."""
    rows = []
    rng = np.random.default_rng(7)
    for r in v2.PILOT_REPS:
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
                L = v2.Local(J, np.asarray(case["st"]["Zmode"], float))
                s = L.S(f)
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


def pilot_audit() -> dict:
    out = v2.OUT / "pilot"
    c = pd.read_csv(out / "candidate_scores.csv")
    pr = pair(c)
    pr.to_csv(out / "pilot_pairs_v2.csv", index=False)
    main = pr[pr.prior == a8.MAIN_PRIOR]
    cm = c[c.prior == a8.MAIN_PRIOR]
    failed = c.status.isin(["FAILED", "EXCEPTION"])
    dis = c.D_multistart.isin(["SAME_POINT_VALUE_MISMATCH", "MULTIMODAL_NONNEGLIGIBLE", "MULTIMODAL_DOMINANT"])
    e_dis = c.E_multistart.isin(["SAME_POINT_VALUE_MISMATCH", "MULTIMODAL_NONNEGLIGIBLE", "MULTIMODAL_DOMINANT"])
    ok_is = main.dropna(subset=["Delta_IS"])
    d_e2_is = (ok_is.Delta_E2 - ok_is.Delta_IS).abs()
    fdsens = main[[f"Delta_E2_h{h:g}" for h in v2.FD_H_HESS_ALT]].sub(main.Delta_E2, axis=0).abs().max(axis=1)
    fdc = fd_check(c)
    fdc.to_csv(out / "pilot_fd_check_v2.csv", index=False)
    inv = []
    for col in ("D_start_status", "E_start_status"):
        for s in c[col].dropna():
            for part in s.split("|"):
                nm, st = part.split(":", 1)
                inv.append({"method": col[0], "start": nm, "status": st})
    inv = pd.DataFrame(inv)
    res = {
        "cells": len(c), "cells_by_status": c.status.value_counts().to_dict(),
        "S1_cell_numerical_failure_share": float(failed.mean()),
        "S2_multistart_disagreement_share": float(dis.mean()),
        "S2_detail": c.D_multistart.value_counts().to_dict(),
        "E_multistart_detail_diagnostic": c.E_multistart.value_counts().to_dict(),
        "E_disagreement_share_diagnostic": float(e_dis.mean()),
        "start_status_counts": inv.groupby(["method", "status"]).size().to_dict() if len(inv) else {},
        "invalid_start_share_D": float((inv[inv.method == "D"].status != "GRAD_OK").mean()) if len(inv) else None,
        "invalid_start_share_E": float((~inv[inv.method == "E"].status.isin(["OK", "OK_FLOOR"])).mean()) if len(inv) else None,
        "main_pairs": len(main), "main_pairs_with_IS_both_OK": len(ok_is),
        "S3_median_abs_DeltaE2_minus_DeltaIS": float(d_e2_is.median()),
        "S3_max_abs_DeltaE2_minus_DeltaIS": float(d_e2_is.max()),
        "median_abs_DeltaE2": float(main.Delta_E2.abs().median()),
        "S4_IS_cells_with_failed_draws_share": float((pd.to_numeric(cm.IS_failed, errors="coerce").fillna(0) > 0).mean()),
        "S4_IS_min_ess": float(pd.to_numeric(cm.IS_ess, errors="coerce").min()),
        "IS_draws_total": int(pd.to_numeric(cm.IS_draws, errors="coerce").sum()),
        "IS_failed_draws_total": int(pd.to_numeric(cm.IS_failed, errors="coerce").fillna(0).sum()),
        "IS_median_se_delta": float(main.IS_se_delta.median()),
        "S5_determinism_max_abs_diff": float(pd.to_numeric(cm.determinism_abs_diff, errors="coerce").max()),
        "S6_median_fd_step_sensitivity_DeltaE2": float(fdsens.median()),
        "S6_max_fd_step_sensitivity_DeltaE2": float(fdsens.max()),
        "S7_fd_grad_rel_err_max": float(fdc.grad_rel_err_max.max()) if "grad_rel_err_max" in fdc else None,
        "S7_fd_hess_rel_err_max": float(fdc.hess_rel_err_max.max()) if "hess_rel_err_max" in fdc else None,
        "diag_median_abs_DeltaE_minus_DeltaE2": float((main.Delta_E - main.Delta_E2).abs().median()),
        "diag_max_abs_DeltaE_minus_DeltaE2": float((main.Delta_E - main.Delta_E2).abs().max()),
        "diag_median_abs_DeltaE_minus_DeltaIS": float((ok_is.Delta_E - ok_is.Delta_IS).abs().median()),
        "diag_sign_disagree_E_vs_E2": int((np.sign(main.Delta_E) != np.sign(main.Delta_E2)).sum()),
        "diag_sign_disagree_E2_vs_IS": int((np.sign(ok_is.Delta_E2) != np.sign(ok_is.Delta_IS)).sum()),
        "diag_D_min_eig_f_min": float(pd.to_numeric(c.D_min_eig_f, errors="coerce").min()),
        "diag_E_min_eig_min": float(pd.to_numeric(c.E_min_eig, errors="coerce").min()),
        "selection_state_counts_label_free": pr.groupby(["prior", "selection_state"]).size().unstack(fill_value=0).to_dict("index"),
        "seconds_total": float(c.seconds.sum()), "seconds_median_main": float(cm.seconds.median()),
        "seconds_median_other": float(c[c.prior != a8.MAIN_PRIOR].seconds.median()),
    }
    sr = v2.STOP_RULES_V2
    res["stop"] = {
        "S1": res["S1_cell_numerical_failure_share"] > sr["S1_cell_numerical_failure_share_max"],
        "S2": res["S2_multistart_disagreement_share"] > sr["S2_multistart_disagreement_share_max"],
        "S3": res["S3_median_abs_DeltaE2_minus_DeltaIS"] > sr["S3_main_median_abs_DeltaE2_minus_DeltaIS_max"],
        "S3b": res["S3_median_abs_DeltaE2_minus_DeltaIS"] > 0.5 * res["median_abs_DeltaE2"],
        "S4": res["S4_IS_cells_with_failed_draws_share"] > sr["S4_IS_cells_with_failed_draws_share_max"],
        "S4b": res["S4_IS_min_ess"] < sr["S4b_IS_min_ess"],
        "S5": res["S5_determinism_max_abs_diff"] > sr["S5_determinism_max_abs_diff"],
        "S6": res["S6_median_fd_step_sensitivity_DeltaE2"] > sr["S6_fd_step_sensitivity_median_abs_DeltaE2_max"],
        "S7": (res["S7_fd_grad_rel_err_max"] is None or res["S7_fd_hess_rel_err_max"] is None
               or max(res["S7_fd_grad_rel_err_max"], res["S7_fd_hess_rel_err_max"]) > sr["S7_fd_derivative_rel_err_max"]),
    }
    res["all_stop_rules_pass"] = not any(res["stop"].values())
    # AF8 vs v2 (numerical)
    af8 = json.loads((a8.OUT / "pilot" / "pilot_numerical_audit.json").read_text(encoding="utf-8"))
    comp = pd.DataFrame([
        {"quantity": "cells", "AF8": af8["cells"], "AF8v2": len(c)},
        {"quantity": "failed cells (no valid score / exception)", "AF8": af8["failed_cells"], "AF8v2": int(failed.sum())},
        {"quantity": "AF8 start-disagreement share (all starts incl. sentinel/saddle) / v2 disagreement among valid starts",
         "AF8": af8["start_disagreement_gt_1e-6_share"], "AF8v2": res["S2_multistart_disagreement_share"]},
        {"quantity": "v2 invalid-start share D (reported, not hidden)", "AF8": None, "AF8v2": res["invalid_start_share_D"]},
        {"quantity": "v2 invalid-start share E (reported, not hidden)", "AF8": None, "AF8v2": res["invalid_start_share_E"]},
        {"quantity": "main: median |DeltaE - DeltaIS|", "AF8": af8["median_abs_DeltaE_minus_DeltaIS"], "AF8v2": res["diag_median_abs_DeltaE_minus_DeltaIS"]},
        {"quantity": "main: median |DeltaE2 - DeltaIS|", "AF8": af8["median_abs_DeltaE2_minus_DeltaIS"], "AF8v2": res["S3_median_abs_DeltaE2_minus_DeltaIS"]},
        {"quantity": "main: pairs compared", "AF8": af8["main_both_ok"], "AF8v2": len(ok_is)},
        {"quantity": "min E negH eigenvalue (accepted)", "AF8": af8["E_min_eig_min"], "AF8v2": res["diag_E_min_eig_min"]},
        {"quantity": "IS min ESS (main)", "AF8": af8["min_IS_ess"], "AF8v2": res["S4_IS_min_ess"]},
        {"quantity": "CPU seconds total", "AF8": af8["seconds_total"], "AF8v2": res["seconds_total"]},
    ])
    comp.to_csv(out / "af8_vs_v2_comparison.csv", index=False)
    (out / "pilot_numerical_audit_v2.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "selection_state_counts_label_free"}, indent=2, default=str))
    print(comp.to_string())
    return res


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--pilot-audit" in argv:
        pilot_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
