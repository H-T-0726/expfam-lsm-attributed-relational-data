"""
Independent integrity audit for the canonical Gaussian-Y K-selection experiment.

This auditor does NOT import the runner. Every constant it checks against is
held as an independent literal here, and selection is recomputed from raw
stored scores. Gates are INTEGRITY ONLY -- no gate refers to which K was
selected, to a recovery rate, or to the shape of the score curve. Those are
research results and must never gate the experiment.

Usage:
    python tools/research_audit/audit_canonical_gaussian_y_k_selection.py --mode pilot
    python tools/research_audit/audit_canonical_gaussian_y_k_selection.py --mode final
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
RES = _ROOT / "expfam" / "results" / "canonical_gaussian_y_k_selection_20260907"

# ---- independent literals (deliberately NOT imported from the runner) -------
EXP_CANDIDATE_K = [1, 2, 3, 4, 5, 6, 7, 8, 9]
EXP_D = 15
EXP_L = 5
EXP_NUM_ITER = 8
EXP_FAMILY_X = "bernoulli"
EXP_FAMILY_Y = "gaussian"
EXP_W0 = 0.0
EXP_SIGMA_Y = 0.1
RECON_TOL = 1e-8
GENERATOR_SOURCE = _ROOT / "expfam" / "src" / "experimental" / "data_generator_canonical_gaussian_y_k.py"
FORBIDDEN = ("normalize_zscore", "whiten(", "standardize(")


def expected_param_count(k: int, d: int) -> int:
    """Independent re-implementation of calc_bic_dual's parameter count.
    family_x=bernoulli -> no Sigma_X params; family_y=gaussian -> +1 for sigma_y."""
    return k * d - k * (k - 1) // 2 + 0 + 1


def audit(mode: str) -> int:
    rows = []

    def gate(gid, name, ok, detail):
        rows.append({"gate_id": gid, "gate_name": name,
                     "status": "PASS" if ok else "FAIL", "detail": detail})

    src = RES / (f"fit_results_pilot.csv" if mode == "pilot" else "fit_results.csv")
    if not src.exists():
        gate("P0", "fit_results_present", False, f"missing {src}")
        pd.DataFrame(rows).to_csv(RES / f"audit_results_{mode}.csv", index=False)
        return 1
    df = pd.read_csv(src)
    gate("P0", "fit_results_present", True, f"{src.name}: {len(df)} rows")

    # ---- P1 process completion --------------------------------------------
    n_fail = int((~df["success"].astype(bool)).sum())
    gate("P1", "all_fits_terminated_or_recorded", True,
         f"{len(df)} rows present; {n_fail} recorded as failed "
         f"(failures are recorded, not deleted)")

    # ---- P2 no NaN / Inf in the criterion ---------------------------------
    ok_rows = df[df["success"].astype(bool)]
    bad = ok_rows[~np.isfinite(ok_rows["criterion_score"])]
    gate("P2", "no_nan_inf_criterion", len(bad) == 0,
         f"{len(bad)} non-finite criterion among {len(ok_rows)} successful fits")

    # ---- P3 dataset hash constant across candidate K within a trial -------
    viol = []
    for (cfg, tr), sub in df.groupby(["config_id", "trial"]):
        if sub["dataset_hash_X"].nunique() != 1 or sub["dataset_hash_Y"].nunique() != 1:
            viol.append((cfg, tr))
    gate("P3", "same_dataset_across_candidate_k", len(viol) == 0,
         f"{len(viol)} (config,trial) groups with inconsistent dataset hash")

    # ---- P4 theory gate ----------------------------------------------------
    tg = RES / "theory_gate.csv"
    if tg.exists():
        g = pd.read_csv(tg)
        nf = int((g["status"] != "PASS").sum())
        gate("P4", "theory_gate_all_pass", nf == 0,
             f"{len(g) - nf}/{len(g)} PASS")
    else:
        gate("P4", "theory_gate_all_pass", False, "theory_gate.csv missing")

    # ---- P5 criterion algebra reconstruction ------------------------------
    # independent recomputation: score = -2*(lpZ+lpX+lpY) + p_hat*log(n)
    r = ok_rows.copy()
    indep = (-2.0 * (r["mean_log_pZ"] + r["mean_log_pX_given_Z"] + r["mean_log_pY_given_Z"])
             + r["parameter_count"] * np.log(r["n"]))
    resid = np.abs(indep - r["criterion_score"])
    worst = float(resid.max()) if len(resid) else 0.0
    gate("P5a", "criterion_algebra_reconstruction", worst <= RECON_TOL,
         f"max |independent - stored| = {worst:.3e} (tol {RECON_TOL:.0e})")
    worst_run = float(r["reconstruction_residual"].max()) if len(r) else 0.0
    gate("P5b", "runner_self_reconstruction", worst_run <= RECON_TOL,
         f"max stored residual = {worst_run:.3e} (tol {RECON_TOL:.0e})")

    # ---- P5c parameter count independently recomputed ----------------------
    exp_p = r.apply(lambda x: expected_param_count(int(x["k_est"]), int(x["d"])), axis=1)
    nbad = int((exp_p != r["parameter_count"]).sum())
    gate("P5c", "parameter_count_independent", nbad == 0,
         f"{nbad} rows disagree with independently computed p_hat")

    # ---- P5d decomposition identity ---------------------------------------
    dres = np.abs((r["fit_only_score"] + r["direct_prior_score"] + r["penalty_term"])
                  - r["criterion_score"])
    gate("P5d", "decomposition_sums_to_criterion", float(dres.max()) <= RECON_TOL,
         f"max |D_K + P_Z + P_theta - C_Q| = {float(dres.max()):.3e}")

    # ---- P6 row counts ------------------------------------------------------
    n_ds = df.groupby(["config_id", "trial"]).ngroups
    expected = n_ds * len(EXP_CANDIDATE_K)
    gate("P6a", "row_count_matches_grid", len(df) == expected,
         f"{len(df)} rows vs expected {n_ds} datasets x {len(EXP_CANDIDATE_K)} K = {expected}")
    missing = []
    for (cfg, tr), sub in df.groupby(["config_id", "trial"]):
        got = sorted(sub["k_est"].tolist())
        if got != EXP_CANDIDATE_K:
            missing.append((cfg, tr, got))
    gate("P6b", "every_candidate_k_present", len(missing) == 0,
         f"{len(missing)} groups with a missing/extra candidate K")
    dup = df.duplicated(subset=["config_id", "trial", "k_est"]).sum()
    gate("P6c", "no_duplicate_keys", dup == 0, f"{dup} duplicate (config,trial,k_est)")

    # ---- P7 generator has no Z standardization ----------------------------
    gsrc = GENERATOR_SOURCE.read_text(encoding="utf-8")
    body = gsrc.split("def generate_canonical_gaussian_y", 1)[-1].split("\ndef ", 1)[0]
    hits = [t for t in FORBIDDEN if t in body]
    gate("P7", "no_z_standardization_in_generator", len(hits) == 0,
         f"forbidden tokens in generator body: {hits}")

    # ---- P8 fixed settings constant ---------------------------------------
    const_ok = (df["L"].nunique() == 1 and int(df["L"].iloc[0]) == EXP_L
                and df["num_iter"].nunique() == 1 and int(df["num_iter"].iloc[0]) == EXP_NUM_ITER
                and df["d"].nunique() == 1 and int(df["d"].iloc[0]) == EXP_D
                and set(df["family_x"]) == {EXP_FAMILY_X}
                and set(df["family_y"]) == {EXP_FAMILY_Y}
                and set(df["w0_true"]) == {EXP_W0}
                and set(df["sigma_y_true"]) == {EXP_SIGMA_Y})
    gate("P8", "frozen_settings_constant", const_ok,
         f"L={sorted(set(df['L']))} num_iter={sorted(set(df['num_iter']))} "
         f"d={sorted(set(df['d']))} w0={sorted(set(df['w0_true']))} "
         f"sigma_y={sorted(set(df['sigma_y_true']))}")

    # ---- P9 seed policy: model_seed = data_seed + 50, independent of k_est --
    seed_ok = bool((df["model_seed"] == df["data_seed"] + 50).all())
    kdep = df.groupby(["config_id", "trial"])["model_seed"].nunique().max()
    gate("P9", "seed_policy_respected", seed_ok and kdep == 1,
         f"model_seed==data_seed+50: {seed_ok}; max distinct model_seed per dataset: {kdep}")

    # ---- P10 seed collisions ------------------------------------------------
    m = df.groupby("data_seed")[["config_id", "trial"]].nunique()
    coll = int(((m["config_id"] > 1) | (m["trial"] > 1)).sum())
    gate("P10", "no_seed_collision", coll == 0, f"{coll} data_seed values shared by >1 cell")

    # ---- P11 selection recomputed independently ---------------------------
    if "selected_k_flag" in df.columns:
        mism = 0
        for (cfg, tr), sub in df.groupby(["config_id", "trial"]):
            o = sub[sub["success"].astype(bool) & np.isfinite(sub["criterion_score"])]
            if len(o) == 0:
                continue
            sel = int(o.loc[o["criterion_score"].idxmin(), "k_est"])
            mn = o["criterion_score"].min()
            sel = int(o[o["criterion_score"] == mn]["k_est"].min())
            flagged = sub[sub["selected_k_flag"].astype(bool)]["k_est"].tolist()
            if flagged != [sel]:
                mism += 1
        gate("P11", "selection_flag_reproducible", mism == 0,
             f"{mism} groups where the stored selected_k_flag disagrees with recomputation")
    else:
        gate("P11", "selection_flag_reproducible", True, "phase file: flag not yet assigned (set at merge)")

    # ---- P12 trial completeness --------------------------------------------
    incomplete = 0
    for (cfg, tr), sub in df.groupby(["config_id", "trial"]):
        if not sub["success"].astype(bool).all():
            incomplete += 1
    gate("P12", "trial_completeness_recorded", True,
         f"{incomplete} of {n_ds} datasets have >=1 failed candidate fit (recorded, not removed)")

    out = pd.DataFrame(rows)
    out.to_csv(RES / f"audit_results_{mode}.csv", index=False)
    nf = int((out["status"] == "FAIL").sum())
    print(out.to_string(index=False))
    print(f"\nAUDIT [{mode}]: {len(out) - nf}/{len(out)} PASS, {nf} FAIL")
    print(f"wrote {RES / f'audit_results_{mode}.csv'}")
    return 1 if nf else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "final"], required=True)
    raise SystemExit(audit(ap.parse_args().mode))
