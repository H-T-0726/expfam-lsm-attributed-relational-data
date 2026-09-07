"""
Per-column mixed-X K-selection experiment (2026-09-07).

QUESTION
--------
Y is held at canonical Gaussian with w != 0, so K_0 = K_GEN is a theorem and
does NOT depend on how X's columns are composed. Under that fixed ground truth,
how does the CURRENT Q-based K-selection criterion behave when the columns of X
carry DIFFERENT exponential families?

This is NOT an experiment about per-column identifiability. It is an experiment
about the behaviour of the existing K-selection rule under mixed-family X.

FROZEN BEFORE EXECUTION. Nothing here is tuned to any observed result.

  model     : experimental/model_dual_expfam_consistent.py
              DualExpFamLSMPerColumnConsistent
              (per-column prototype + objective-consistent numerics; KI-015
               requires numerics_mode="consistent" for new per-column work)
  criterion : experimental/eval_utils.py
              calc_Q_dual_strict_exp + calc_bic_exp
              Q-based complete-data criterion. NOT a Schwarz BIC (KI-010).
  generator : experimental/data_generator_percolumn_gaussian_y.py

Parity with the single-family path was audited first
(tools/research_audit/audit_percolumn_parity.py): with all columns Bernoulli
the two paths agree bit-for-bit on Z, F, w0, w, sigma_y and num_params, and the
criterion differs only by a K-independent Gaussian-Y 2*pi constant.

Usage:
    python tools/research_audit/run_percolumn_mixed_x_k_selection.py --gate
    python tools/research_audit/run_percolumn_mixed_x_k_selection.py --pilot
    python tools/research_audit/run_percolumn_mixed_x_k_selection.py --full
    python tools/research_audit/run_percolumn_mixed_x_k_selection.py --merge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "expfam" / "src"))
sys.path.insert(0, str(_ROOT / "expfam" / "src" / "experimental"))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from data_generator_percolumn_gaussian_y import (  # noqa: E402
    GENERATOR_ID, generate_percolumn_gaussian_y, make_family_list,
)
from model_dual_expfam_consistent import DualExpFamLSMPerColumnConsistent  # noqa: E402
import eval_utils  # noqa: E402

# =====================================================================
# FROZEN CONSTANTS (fixed before any result was observed)
# =====================================================================

EXPERIMENT_ID = "percolumn_mixed_x_k_selection_20260907"
OUT_DIR = _ROOT / "expfam" / "results" / EXPERIMENT_ID
JST = ZoneInfo("Asia/Tokyo")

N, D, K_GEN = 150, 15, 3
L_FIXED, NUM_ITER_FIXED = 5, 8
CANDIDATE_K = [1, 2, 3, 4, 5, 6, 7, 8, 9]
TRIALS = list(range(10))
PILOT_TRIALS = [0]

W0_TRUE, W_TRUE, SIGMA_Y_TRUE = 0.0, 0.5, 0.1
VAR_F, UNIQ = 5.0, 0.1
FAMILY_Y = "gaussian"

# estimator settings, verbatim from the canonical Gaussian-Y core runner
MAX_RETRIES = 2
NEWTON_ALPHA_BASE, NEWTON_MAX_ITER = 0.5, 10
ADAM_MAX_ITER = 50
INIT_W_GAUSSIAN_Y = 0.5
# Applied to EVERY config, including all-Gaussian. The core runner applied it
# only for discrete X; here initialisation must not covary with the family
# composition being compared, so it is made uniform. Frozen decision.
INIT_F_SCALE = 0.2

FAMILY_CONFIGS: dict[str, list[tuple[str, int]]] = {
    "H-B":   [("bernoulli", 15)],
    "H-G":   [("gaussian", 15)],
    "H-P":   [("poisson", 15)],
    "M-BG":  [("bernoulli", 8), ("gaussian", 7)],
    "M-BP":  [("bernoulli", 8), ("poisson", 7)],
    "M-GP":  [("gaussian", 8), ("poisson", 7)],
    "M-BGP": [("bernoulli", 5), ("gaussian", 5), ("poisson", 5)],
}
CONFIG_KIND = {"H-B": "homogeneous", "H-G": "homogeneous", "H-P": "homogeneous",
               "M-BG": "mixed", "M-BP": "mixed", "M-GP": "mixed", "M-BGP": "mixed"}

TIE_RULE = "smallest_k_among_ties"
RECON_TOLERANCE = 1e-8


def data_seed_of(trial: int) -> int:
    """SHARED across family configs -> paired design (same Z, F, Y)."""
    return 800000 + trial


def model_seed_of(data_seed: int) -> int:
    return data_seed + 50


def enumerate_cells(phase: str):
    trials = PILOT_TRIALS if phase == "pilot" else [
        t for t in TRIALS if t not in PILOT_TRIALS]
    return [(cfg, t, data_seed_of(t)) for cfg in FAMILY_CONFIGS for t in trials]


# =====================================================================
# config gate
# =====================================================================

def build_config_gate() -> pd.DataFrame:
    rows = []

    def g(gid, name, ok, ev):
        rows.append({"gate_id": gid, "gate_name": name,
                     "status": "PASS" if ok else "FAIL", "evidence": ev})

    # paired design: Z/F/Y identical across configs for a fixed trial
    hz, hf, hy, hx = set(), set(), set(), set()
    for cfg, spec in FAMILY_CONFIGS.items():
        d = generate_percolumn_gaussian_y(
            n=N, d=D, k=K_GEN, seed=data_seed_of(0),
            family_x_list=make_family_list(spec, D),
            w0_true=W0_TRUE, w_true=W_TRUE, sigma_y_true=SIGMA_Y_TRUE,
            var_f=VAR_F, uniq=UNIQ)
        hz.add(d["dataset_hash_Z"]); hf.add(d["dataset_hash_F"])
        hy.add(d["dataset_hash_Y"]); hx.add(d["dataset_hash_X"])
    g("C1", "paired_Z_F_Y_across_configs",
      len(hz) == 1 and len(hf) == 1 and len(hy) == 1,
      f"distinct hashes Z={len(hz)} F={len(hf)} Y={len(hy)} over "
      f"{len(FAMILY_CONFIGS)} configs (must all be 1)")
    g("C2", "X_actually_varies_across_configs", len(hx) == len(FAMILY_CONFIGS),
      f"{len(hx)} distinct X hashes over {len(FAMILY_CONFIGS)} configs")

    # column counts
    ok = all(len(make_family_list(s, D)) == D for s in FAMILY_CONFIGS.values())
    g("C3", "family_lists_have_length_d", ok, f"all configs sum to d={D}")

    # Y side is canonical Gaussian with w != 0 -> K_0 = K_GEN
    g("C4", "gaussian_y_w_nonzero", FAMILY_Y == "gaussian" and W_TRUE != 0.0,
      f"family_y={FAMILY_Y}, w={W_TRUE} != 0 -> single-dyad Y marginal "
      f"identifies K, so K_0 = K_GEN = {K_GEN} independent of X composition")

    # no standardization anywhere in the generator
    import data_generator_percolumn_gaussian_y as gm
    import inspect
    src = inspect.getsource(gm.generate_percolumn_gaussian_y)
    bad = [t for t in ("normalize_zscore", "standardize", "whiten", "scale_Z") if t in src]
    g("C5", "no_generator_postprocessing", not bad, f"forbidden tokens: {bad}")

    # Poisson stays far from the overflow boundary of the consistent numerics
    worst, outside = -np.inf, 0
    for t in TRIALS:
        d = generate_percolumn_gaussian_y(
            n=N, d=D, k=K_GEN, seed=data_seed_of(t),
            family_x_list=make_family_list(FAMILY_CONFIGS["H-P"], D),
            w0_true=W0_TRUE, w_true=W_TRUE, sigma_y_true=SIGMA_Y_TRUE,
            var_f=VAR_F, uniq=UNIQ)
        worst = max(worst, d["poisson_eta_max"])
        outside += d["poisson_eta_outside_legacy_clip"]
    g("C6", "poisson_eta_safe", worst < 100.0 and outside == 0,
      f"max generator eta over 10 trials = {worst:.3f} "
      f"(consistent numerics raise above log(finfo.max)=709.78); "
      f"entries outside the legacy clip window [-20,10] = {outside}")

    # candidate set contains the truth
    g("C7", "candidate_contains_truth", K_GEN in CANDIDATE_K,
      f"candidate K = {CANDIDATE_K} contains K_GEN = {K_GEN}")

    # objective-consistent class (KI-015 operational rule)
    g("C8", "objective_consistent_numerics_used",
      getattr(DualExpFamLSMPerColumnConsistent, "numerics_mode", None) == "consistent",
      "DualExpFamLSMPerColumnConsistent.numerics_mode == 'consistent' "
      "(KI-015 requires this for new per-column experiments)")

    # seed policy: shared data seed, no collision within a trial
    seeds = [data_seed_of(t) for t in TRIALS]
    g("C9", "seed_policy_no_collision", len(set(seeds)) == len(seeds),
      f"{len(set(seeds))} unique data seeds for {len(TRIALS)} trials; "
      f"data_seed is SHARED across configs by design (paired)")

    # criterion identity
    g("C10", "criterion_is_percolumn_aware", True,
      "eval_utils.calc_Q_dual_strict_exp + calc_bic_exp(family_x='mixed', "
      "n_gaussian_x_cols=<count>); utils_expfam.calc_bic_dual is NOT used "
      "because it cannot count per-column Gaussian dispersions")

    return pd.DataFrame(rows)


def build_family_config_table() -> pd.DataFrame:
    rows = []
    for cfg, spec in FAMILY_CONFIGS.items():
        fl = make_family_list(spec, D)
        rows.append({
            "config_id": cfg, "kind": CONFIG_KIND[cfg],
            "n_bernoulli": sum(1 for f in fl if f == "bernoulli"),
            "n_gaussian": sum(1 for f in fl if f == "gaussian"),
            "n_poisson": sum(1 for f in fl if f == "poisson"),
            "n_families": len({f for f in fl}),
            "family_x_list": "|".join(fl),
        })
    return pd.DataFrame(rows)


# =====================================================================
# one fit
# =====================================================================

def run_one_fit(X, Y, k_est, n_gauss_cols, family_x_list, seed_model):
    t0 = time.perf_counter()
    out = {"success": False, "failure_reason": "", "nan_occurred": True,
           "internal_retry": 0, "poisson_overflow": False}
    n = X.shape[0]
    try:
        for retry in range(MAX_RETRIES + 1):
            alpha = NEWTON_ALPHA_BASE / (2 ** retry)
            rng = np.random.default_rng(seed_model + retry * 1000)
            m = DualExpFamLSMPerColumnConsistent(
                n=n, d=D, k=k_est, L=L_FIXED,
                family_x_list=list(family_x_list), family_y=FAMILY_Y)
            m.initialize_params(true_params={"var_f": VAR_F},
                                seed=seed_model + retry * 1000)
            um = np.triu(np.ones((n, n), dtype=bool), k=1)
            uv = Y[um]
            m.params["w0"] = float(uv.mean())
            m.params["w"] = INIT_W_GAUSSIAN_Y
            m.sigma_y = float(max(uv.std(), 0.01))
            m.params["F"] *= INIT_F_SCALE

            Z = m.params["Z"].copy(); F = m.params["F"].copy()
            sigma = m.params["sigma"].copy()
            w0 = m.params["w0"]; w = m.params["w"]; var_z = m.params["var_z"]
            Z_prev = Z.copy(); nan_count = 0

            for _ in range(NUM_ITER_FIXED):
                Zs = np.zeros((n, k_est, L_FIXED))
                for l in range(L_FIXED):
                    m.params.update(dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w))
                    Zn = m.calc_eta_newton(X, Y, rng=rng,
                                           max_iter=NEWTON_MAX_ITER, alpha=alpha)
                    Zs[:, :, l] = Zn
                    Z = Zn.copy()
                if np.any(~np.isfinite(Zs)):
                    nan_count += 1
                    Zs = np.stack([Z_prev] * L_FIXED, axis=2)
                    Z = Z_prev.copy()
                Zs = m.scale_Z(Zs)
                Z_prev = Z.copy(); Z = Zs[:, :, -1].copy()
                F = m.calc_F(X, Zs)
                sigma = m.calc_sigma(X, Zs, F)
                w0 = m.calc_w0(Y, Zs, w0, w, max_iter=ADAM_MAX_ITER)
                w = m.calc_w(Y, Zs, w0, w, max_iter=ADAM_MAX_ITER)
                m.calc_sigma_y(Y, Zs, w0, w)

            out["internal_retry"] = retry
            if nan_count == 0:
                break

        m.params.update({"F": F, "sigma": sigma, "w0": w0, "w": w})

        q = eval_utils.calc_Q_dual_strict_exp(X, Y, Zs, F, sigma, var_z, w0, w, m)
        score, p_hat = eval_utils.calc_bic_exp(
            Q_strict=q, k=k_est, n=n, d=D, family_x="mixed",
            family_y_label=FAMILY_Y, n_gaussian_x_cols=int(n_gauss_cols))

        # ---- decomposition (no RNG consumed) ------------------------------
        s_pz = s_px = s_py = 0.0
        for l in range(L_FIXED):
            Zl = Zs[:, :, l]
            s_pz += float(-(n * k_est / 2.0) * np.log(2.0 * np.pi * var_z)
                          - (1.0 / (2.0 * var_z)) * np.sum(Zl ** 2))
            s_px += m.calc_log_likelihood_X(X, Zs[:, :, l:l + 1], F)
            s_py += m.calc_log_likelihood_Y(Y, Zs[:, :, l:l + 1], w0, w)
        mz, mx, my = s_pz / L_FIXED, s_px / L_FIXED, s_py / L_FIXED

        # K-independent strict corrections carried by calc_Q_dual_strict_exp
        from scipy.special import gammaln
        pois_cols = m.columns_of("poisson")
        corr = 0.0
        if len(pois_cols):
            corr -= float(np.sum(gammaln(X[:, pois_cols] + 1)))
        corr -= 0.5 * np.log(2.0 * np.pi) * float(um.sum())

        penalty = float(p_hat) * float(np.log(n))
        fit_only = -2.0 * (mx + my + corr)      # D_K  (includes K-independent corr)
        direct_prior = -2.0 * mz                # P_Z(K)
        recon = fit_only + direct_prior + penalty

        eta_x_final = Zs[:, :, -1] @ F.T
        out.update({
            "success": True, "nan_occurred": nan_count > 0,
            "q_strict": float(q), "criterion_score": float(score),
            "parameter_count": int(p_hat), "penalty_term": penalty,
            "mean_log_pZ": mz, "mean_log_pX_given_Z": mx,
            "mean_log_pY_given_Z": my, "strict_correction": corr,
            "fit_only_score": float(fit_only),
            "direct_prior_score": float(direct_prior),
            "score_without_direct_prior": float(fit_only + penalty),
            "reconstructed_score": float(recon),
            "reconstruction_residual": float(abs(recon - score)),
            "w0_hat": float(w0), "w_hat": float(w),
            "sigma_y_hat": float(m.sigma_y),
            "eta_x_abs_max_fit": float(np.abs(eta_x_final).max()),
            "sigma_x_hat_mean_gaussian_cols": (
                float(np.mean(np.diag(sigma)[m.columns_of("gaussian")]))
                if n_gauss_cols else np.nan),
        })
    except FloatingPointError as exc:
        out["poisson_overflow"] = True
        out["failure_reason"] = f"FloatingPointError: {exc}"[:300]
    except Exception as exc:  # noqa: BLE001
        out["failure_reason"] = f"{type(exc).__name__}: {exc}"[:300]
        traceback.print_exc()
    if not out["success"]:
        for key in ("q_strict", "criterion_score", "penalty_term", "mean_log_pZ",
                    "mean_log_pX_given_Z", "mean_log_pY_given_Z", "strict_correction",
                    "fit_only_score", "direct_prior_score",
                    "score_without_direct_prior", "reconstructed_score",
                    "reconstruction_residual", "w0_hat", "w_hat", "sigma_y_hat",
                    "eta_x_abs_max_fit", "sigma_x_hat_mean_gaussian_cols"):
            out[key] = np.nan
        out["parameter_count"] = -1
    out["runtime_seconds"] = time.perf_counter() - t0
    return out


# =====================================================================
# phase driver
# =====================================================================

def run_phase(phase: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = enumerate_cells(phase)
    total = len(cells) * len(CANDIDATE_K)
    print(f"[{phase}] {len(cells)} datasets x {len(CANDIDATE_K)} K = {total} fits")

    fit_rows, gen_rows, seed_rows = [], [], []
    t0 = time.perf_counter(); done = 0

    for cfg, trial, ds in cells:
        fl = make_family_list(FAMILY_CONFIGS[cfg], D)
        ms = model_seed_of(ds)
        data = generate_percolumn_gaussian_y(
            n=N, d=D, k=K_GEN, seed=ds, family_x_list=fl,
            w0_true=W0_TRUE, w_true=W_TRUE, sigma_y_true=SIGMA_Y_TRUE,
            var_f=VAR_F, uniq=UNIQ)
        X, Y = data["X"], data["Y"]
        ngauss = data["n_gaussian_x_cols"]

        seed_rows.append({"experiment_id": EXPERIMENT_ID, "phase": phase,
                          "config_id": cfg, "trial": trial, "data_seed": ds,
                          "model_seed": ms, "seed_rule_data": "800000 + trial",
                          "seed_rule_model": "data_seed + 50",
                          "x_stream_offset": 500000,
                          "shared_across_configs": "Z, F, Y",
                          "hash_Z": data["dataset_hash_Z"],
                          "hash_F": data["dataset_hash_F"],
                          "hash_Y": data["dataset_hash_Y"],
                          "hash_X": data["dataset_hash_X"]})
        gen_rows.append({"experiment_id": EXPERIMENT_ID, "phase": phase,
                         "config_id": cfg, "kind": CONFIG_KIND[cfg],
                         "trial": trial, "data_seed": ds,
                         **{k2: data[k2] for k2 in (
                             "n", "d", "K_GEN", "n_gaussian_x_cols",
                             "n_bernoulli_x_cols", "n_poisson_x_cols",
                             "rank_Z", "sigma_min_Z", "sigma_max_Z",
                             "rank_F", "sigma_min_F", "sigma_max_F",
                             "eta_x_min", "eta_x_max", "eta_x_abs_max",
                             "poisson_eta_max", "poisson_eta_min",
                             "poisson_eta_outside_legacy_clip",
                             "poisson_count_max", "poisson_count_mean",
                             "x_mean", "y_mean", "y_variance",
                             "dataset_hash_X", "dataset_hash_Y",
                             "dataset_hash_Z", "dataset_hash_F")}})

        for k_est in CANDIDATE_K:
            done += 1
            res = run_one_fit(X, Y, k_est, ngauss, fl, ms)
            row = {"experiment_id": EXPERIMENT_ID, "phase": phase,
                   "config_id": cfg, "kind": CONFIG_KIND[cfg],
                   "trial": trial, "data_seed": ds, "model_seed": ms,
                   "n": N, "d": D, "K_GEN": K_GEN, "K_0": K_GEN,
                   "family_y": FAMILY_Y, "w0_true": W0_TRUE, "w_true": W_TRUE,
                   "sigma_y_true": SIGMA_Y_TRUE,
                   "n_gaussian_x_cols": ngauss,
                   "n_bernoulli_x_cols": data["n_bernoulli_x_cols"],
                   "n_poisson_x_cols": data["n_poisson_x_cols"],
                   "k_est": k_est, "L": L_FIXED, "num_iter": NUM_ITER_FIXED,
                   "dataset_hash_X": data["dataset_hash_X"],
                   "dataset_hash_Y": data["dataset_hash_Y"],
                   "dataset_hash_Z": data["dataset_hash_Z"],
                   "generator_id": GENERATOR_ID,
                   "estimator_class": "DualExpFamLSMPerColumnConsistent",
                   "numerics_mode": "consistent",
                   "criterion_fn": "eval_utils.calc_bic_exp"}
            row.update(res)
            fit_rows.append(row)
            el = (time.perf_counter() - t0) / 60
            print(f"  [{done:4d}/{total}] {cfg:6s} t={trial} k={k_est} "
                  f"score={res.get('criterion_score', float('nan')):.1f} "
                  f"ok={res['success']} [{el:.1f} min]", flush=True)

    pd.DataFrame(fit_rows).to_csv(OUT_DIR / f"fit_results_{phase}.csv", index=False)
    pd.DataFrame(gen_rows).to_csv(OUT_DIR / f"diagnostics_{phase}.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUT_DIR / f"seed_provenance_{phase}.csv", index=False)
    df = pd.DataFrame(fit_rows)
    print(f"[{phase}] done: {len(df)} fits in {(time.perf_counter()-t0)/60:.1f} min; "
          f"median {df['runtime_seconds'].median():.2f} s/fit; "
          f"failures {int((~df['success']).sum())}")


def merge_phases() -> None:
    f, d, s = [], [], []
    for p in ("pilot", "full"):
        for lst, name in ((f, "fit_results"), (d, "diagnostics"), (s, "seed_provenance")):
            q = OUT_DIR / f"{name}_{p}.csv"
            if q.exists():
                lst.append(pd.read_csv(q))
    if not f:
        raise SystemExit("nothing to merge")
    df = pd.concat(f, ignore_index=True).sort_values(
        ["config_id", "trial", "k_est"]).reset_index(drop=True)
    df["selected_k_flag"] = False
    for (c, t), sub in df.groupby(["config_id", "trial"], sort=False):
        ok = sub[sub["success"] & sub["criterion_score"].notna()]
        if not len(ok):
            continue
        mn = ok["criterion_score"].min()
        sel = int(ok[ok["criterion_score"] == mn]["k_est"].min())
        df.loc[(df.config_id == c) & (df.trial == t) & (df.k_est == sel),
               "selected_k_flag"] = True
    df.to_csv(OUT_DIR / "fit_results.csv", index=False)
    pd.concat(d, ignore_index=True).to_csv(OUT_DIR / "diagnostics.csv", index=False)
    pd.concat(s, ignore_index=True).to_csv(OUT_DIR / "seed_provenance.csv", index=False)
    print(f"merged {len(df)} fits -> {OUT_DIR/'fit_results.csv'}")


def _git(*a) -> str:
    try:
        return subprocess.run(["git", *a], cwd=_ROOT, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return "UNAVAILABLE"


def write_manifest(extra: dict | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib, scipy  # noqa: E402
    man = {
        "experiment_id": EXPERIMENT_ID, "generator_id": GENERATOR_ID,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "head_sha": _git("rev-parse", "HEAD"),
        "base_commit": "bf07887 (canonical Gaussian-Y experiment tip)",
        "git_status_porcelain": _git("status", "--porcelain"),
        "timestamp_jst": datetime.now(JST).isoformat(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "frozen_settings": {
            "K_GEN": K_GEN, "n": N, "d": D, "candidate_K": CANDIDATE_K,
            "trials": TRIALS, "L": L_FIXED, "num_iter": NUM_ITER_FIXED,
            "family_y": FAMILY_Y, "w0_true": W0_TRUE, "w_true": W_TRUE,
            "sigma_y_true": SIGMA_Y_TRUE, "var_f": VAR_F, "uniq": UNIQ,
            "family_configs": {k: v for k, v in FAMILY_CONFIGS.items()},
            "init_F_scale": INIT_F_SCALE,
            "init_F_scale_note": (
                "applied to EVERY config including all-Gaussian, so that "
                "initialisation does not covary with the family composition "
                "under comparison; the core runner applied it only for discrete X"),
            "init_w_gaussian_y": INIT_W_GAUSSIAN_Y,
            "max_retries": MAX_RETRIES, "newton_alpha_base": NEWTON_ALPHA_BASE,
            "newton_max_iter": NEWTON_MAX_ITER, "adam_max_iter": ADAM_MAX_ITER,
            "tie_rule": TIE_RULE, "recon_tolerance": RECON_TOLERANCE,
        },
        "seed_policy": {
            "data": "data_seed = 800000 + trial  (SHARED across family configs)",
            "model": "model_seed = data_seed + 50 (independent of k_est and config)",
            "x_stream": "X drawn from default_rng(data_seed + 500000), a separate "
                        "stream so that changing the family composition cannot "
                        "perturb Z, F or Y",
        },
        "estimator": {
            "class": "DualExpFamLSMPerColumnConsistent",
            "module": "expfam/src/experimental/model_dual_expfam_consistent.py",
            "numerics_mode": "consistent",
            "lineage_note": (
                "per-column prototype (fixed -> masked -> percolumn) with "
                "objective-consistent numerics. PROTOTYPE, not a manuscript "
                "method (root CLAUDE.md section 3). KI-015 requires "
                "numerics_mode='consistent' for new per-column experiments."),
            "criterion_fn": (
                "eval_utils.calc_Q_dual_strict_exp + calc_bic_exp; Q-based "
                "complete-data criterion, NOT a Schwarz BIC (KI-010)"),
        },
        "comparability": {
            "vs_canonical_core": (
                "NOT numerically comparable. The core used calc_bic_dual, which "
                "omits the Gaussian-Y -0.5*ln(2pi) per-pair constant that "
                "calc_Q_dual_strict_exp includes. The parity audit measured the "
                "offset at +20538.2762 on the criterion, constant in K, so "
                "argmin is unaffected but absolute scores must never be merged."),
            "parity_audit": "tools/research_audit/audit_percolumn_parity.py",
        },
    }
    if extra:
        man.update(extra)
    (OUT_DIR / "run_manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_DIR/'run_manifest.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    for flag in ("gate", "pilot", "full", "merge", "manifest"):
        ap.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args()
    if a.gate:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        g = build_config_gate()
        g.to_csv(OUT_DIR / "config_gate.csv", index=False)
        build_family_config_table().to_csv(OUT_DIR / "family_config.csv", index=False)
        print(g.to_string(index=False))
        nf = int((g.status == "FAIL").sum())
        print(f"\nCONFIG GATE: {len(g)-nf}/{len(g)} PASS")
        if nf:
            return 1
    if a.manifest:
        write_manifest()
    if a.pilot:
        run_phase("pilot")
    if a.full:
        run_phase("full")
    if a.merge:
        merge_phases()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
