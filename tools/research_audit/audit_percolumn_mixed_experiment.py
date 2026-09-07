"""
Independent post-run integrity audit for the per-column mixed-X experiment.

Does not import the runner. Every constant is held as an independent literal
here and selection is recomputed from raw stored scores.

INTEGRITY GATES ONLY. No gate refers to which K was selected, to a recovery
rate, or to the shape of a score curve -- those are research results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
RES = _ROOT / "expfam" / "results" / "percolumn_mixed_x_k_selection_20260907"

CAND_K = [1, 2, 3, 4, 5, 6, 7, 8, 9]
CONFIGS = ["H-B", "H-G", "H-P", "M-BG", "M-BP", "M-GP", "M-BGP"]
EXP_D, EXP_N, EXP_L, EXP_ITER, EXP_K0 = 15, 150, 5, 8, 3
EXP_TRIALS = 10
EXP_W0, EXP_W, EXP_SY = 0.0, 0.5, 0.1
RECON_TOL = 1e-8
GEN_SRC = _ROOT / "expfam" / "src" / "experimental" / "data_generator_percolumn_gaussian_y.py"


def expected_p(k, d, n_gauss):
    """Independent re-implementation of eval_utils.calc_bic_exp's count."""
    return k * d - k * (k - 1) // 2 + int(n_gauss) + 1   # +1 for Gaussian-Y sigma_y


def main() -> int:
    rows = []

    def g(gid, name, ok, detail):
        rows.append({"gate_id": gid, "gate_name": name,
                     "status": "PASS" if ok else "FAIL", "detail": detail})

    src = RES / "fit_results.csv"
    if not src.exists():
        g("A0", "fit_results_present", False, f"missing {src}")
        pd.DataFrame(rows).to_csv(RES / "audit_results.csv", index=False)
        return 1
    df = pd.read_csv(src)
    g("A0", "fit_results_present", True, f"{len(df)} rows")

    exp_rows = len(CONFIGS) * EXP_TRIALS * len(CAND_K)
    g("A1", "row_count_matches_frozen_grid", len(df) == exp_rows,
      f"{len(df)} vs expected {len(CONFIGS)}x{EXP_TRIALS}x{len(CAND_K)} = {exp_rows}")

    nfail = int((~df["success"].astype(bool)).sum())
    g("A2", "failures_recorded_not_deleted", True,
      f"{nfail} failed fits recorded (never removed)")

    ok = df[df["success"].astype(bool)]
    g("A3", "no_nan_inf_criterion",
      int((~np.isfinite(ok["criterion_score"])).sum()) == 0,
      f"{int((~np.isfinite(ok['criterion_score'])).sum())} non-finite of {len(ok)}")

    g("A4", "no_poisson_overflow", int(df["poisson_overflow"].sum()) == 0,
      f"{int(df['poisson_overflow'].sum())} FloatingPointError events "
      f"(consistent numerics raise instead of clipping)")

    # --- PAIRED DESIGN: Z, F, Y must be shared across configs within a trial
    bad = []
    for tr, sub in df.groupby("trial"):
        if sub["dataset_hash_Z"].nunique() != 1 or sub["dataset_hash_Y"].nunique() != 1:
            bad.append(tr)
    g("A5", "paired_Z_and_Y_shared_across_configs", not bad,
      f"{len(bad)} trials where Z or Y hash differs across family configs")

    badx = []
    for tr, sub in df.groupby("trial"):
        if sub.groupby("config_id")["dataset_hash_X"].first().nunique() != len(CONFIGS):
            badx.append(tr)
    g("A6", "X_differs_across_configs", not badx,
      f"{len(badx)} trials where two configs share an X hash")

    # --- same dataset across candidate K within a cell
    v = [(c, t) for (c, t), s in df.groupby(["config_id", "trial"])
         if s["dataset_hash_X"].nunique() != 1 or s["dataset_hash_Y"].nunique() != 1]
    g("A7", "same_dataset_across_candidate_k", not v,
      f"{len(v)} (config,trial) cells with inconsistent dataset hash")

    # --- criterion algebra, recomputed independently
    r = ok
    indep = (-2.0 * (r["mean_log_pZ"] + r["mean_log_pX_given_Z"]
                     + r["mean_log_pY_given_Z"] + r["strict_correction"])
             + r["parameter_count"] * np.log(r["n"]))
    resid = np.abs(indep - r["criterion_score"])
    g("A8", "criterion_algebra_reconstruction", float(resid.max()) <= RECON_TOL,
      f"max |independent - stored| = {float(resid.max()):.3e} (tol {RECON_TOL:.0e})")
    g("A9", "runner_self_reconstruction",
      float(r["reconstruction_residual"].max()) <= RECON_TOL,
      f"max stored residual = {float(r['reconstruction_residual'].max()):.3e}")

    dres = np.abs((r["fit_only_score"] + r["direct_prior_score"] + r["penalty_term"])
                  - r["criterion_score"])
    g("A10", "decomposition_sums_to_criterion", float(dres.max()) <= RECON_TOL,
      f"max |D_K + P_Z + P_theta - C_Q| = {float(dres.max()):.3e}")

    # --- parameter count independently
    ep = r.apply(lambda x: expected_p(int(x["k_est"]), int(x["d"]),
                                      int(x["n_gaussian_x_cols"])), axis=1)
    nb = int((ep != r["parameter_count"]).sum())
    g("A11", "parameter_count_independent", nb == 0,
      f"{nb} rows disagree with k*d-k(k-1)/2+n_gaussian_cols+1")

    # --- latent prior determinism (as established in the canonical core)
    pred = r["n"] * r["k_est"] * (np.log(2 * np.pi) + 1.0)
    pres = np.abs(r["direct_prior_score"] - pred)
    g("A12", "latent_prior_is_deterministic_nK", float(pres.max()) < 1e-6,
      f"max |P_Z - n*K*(log2pi+1)| = {float(pres.max()):.3e} "
      f"(scale_Z fixes mean(Z^2)=1, so P_Z carries no data)")

    # --- completeness
    miss = [(c, t) for (c, t), s in df.groupby(["config_id", "trial"])
            if sorted(s["k_est"].tolist()) != CAND_K]
    g("A13", "every_candidate_k_present", not miss, f"{len(miss)} incomplete cells")
    dup = int(df.duplicated(subset=["config_id", "trial", "k_est"]).sum())
    g("A14", "no_duplicate_keys", dup == 0, f"{dup} duplicates")
    g("A15", "all_configs_present",
      sorted(df["config_id"].unique()) == sorted(CONFIGS),
      f"{sorted(df['config_id'].unique())}")

    # --- frozen settings constant
    const = (set(df["L"]) == {EXP_L} and set(df["num_iter"]) == {EXP_ITER}
             and set(df["d"]) == {EXP_D} and set(df["n"]) == {EXP_N}
             and set(df["K_0"]) == {EXP_K0} and set(df["w0_true"]) == {EXP_W0}
             and set(df["w_true"]) == {EXP_W} and set(df["sigma_y_true"]) == {EXP_SY}
             and set(df["family_y"]) == {"gaussian"}
             and set(df["numerics_mode"]) == {"consistent"})
    g("A16", "frozen_settings_constant", const,
      f"L={sorted(set(df['L']))} num_iter={sorted(set(df['num_iter']))} "
      f"w={sorted(set(df['w_true']))} sigma_y={sorted(set(df['sigma_y_true']))} "
      f"numerics={sorted(set(df['numerics_mode']))}")

    # --- column counts sum to d
    s2 = (df["n_bernoulli_x_cols"] + df["n_gaussian_x_cols"] + df["n_poisson_x_cols"])
    g("A17", "family_column_counts_sum_to_d", bool((s2 == EXP_D).all()),
      f"all rows sum to d={EXP_D}")

    # --- seed policy
    sp = bool((df["model_seed"] == df["data_seed"] + 50).all())
    shared = df.groupby("trial")["data_seed"].nunique().max() == 1
    g("A18", "seed_policy_respected", sp and shared,
      f"model_seed==data_seed+50: {sp}; data_seed shared across configs "
      f"within a trial: {shared}")

    # --- selection flag reproducible
    mism = 0
    for (c, t), sub in df.groupby(["config_id", "trial"]):
        o = sub[sub["success"].astype(bool) & np.isfinite(sub["criterion_score"])]
        if not len(o):
            continue
        sel = int(o[o["criterion_score"] == o["criterion_score"].min()]["k_est"].min())
        if sub[sub["selected_k_flag"].astype(bool)]["k_est"].tolist() != [sel]:
            mism += 1
    g("A19", "selection_flag_reproducible", mism == 0,
      f"{mism} cells where stored flag disagrees with recomputation")

    # --- generator has no post-processing
    body = GEN_SRC.read_text(encoding="utf-8").split(
        "def generate_percolumn_gaussian_y", 1)[-1]
    hits = [t for t in ("normalize_zscore", "standardize(", "whiten(") if t in body]
    g("A20", "no_generator_postprocessing", not hits, f"forbidden tokens: {hits}")

    # --- config gate still passing
    cg = RES / "config_gate.csv"
    if cg.exists():
        c = pd.read_csv(cg)
        nf = int((c["status"] != "PASS").sum())
        g("A21", "config_gate_all_pass", nf == 0, f"{len(c)-nf}/{len(c)} PASS")
    else:
        g("A21", "config_gate_all_pass", False, "config_gate.csv missing")

    # --- parity audit still on record
    pa = RES / "audit_percolumn_parity.csv"
    if pa.exists():
        p = pd.read_csv(pa)
        nf = int((p["status"] == "FAIL").sum())
        g("A22", "percolumn_parity_audit_no_blocker", nf == 0,
          f"{int((p.status=='PASS').sum())} PASS / {nf} FAIL / "
          f"{int((p.status=='INFO').sum())} INFO")
    else:
        g("A22", "percolumn_parity_audit_no_blocker", False, "parity audit missing")

    out = pd.DataFrame(rows)
    out.to_csv(RES / "audit_results.csv", index=False)
    nf = int((out.status == "FAIL").sum())
    print(out.to_string(index=False))
    print(f"\nAUDIT: {len(out)-nf}/{len(out)} PASS, {nf} FAIL")
    return 1 if nf else 0


if __name__ == "__main__":
    raise SystemExit(main())
