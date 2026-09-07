"""
Canonical Gaussian-Y K-selection experiment (2026-09-07).

CONFIRMATORY experiment. Question:

    Under a generator for which K_0 = K_GEN is a THEOREM (canonical Gaussian-Y,
    w != 0), what does the CURRENT Q-based K-selection criterion actually select?

Nothing in this runner is tuned to make K = K_GEN win. The estimator settings
are copied from the historical fixed official Exp1 K-selection runner so that
the ONLY thing that changed is the generator.

  estimator  : expfam/src/model_dual_expfam_fixed.py :: DualExpFamLSMFixed
               (fixed lineage, Y-side 0.5 removed; root CLAUDE.md section 3)
  criterion  : expfam/src/utils_expfam.py :: calc_bic_dual
               Q-based complete-data criterion. NOT a Schwarz BIC (KI-010).
               The historical function/column name `BIC` is preserved unchanged.
  generator  : expfam/src/experimental/data_generator_canonical_gaussian_y_k.py

Usage:
    python tools/research_audit/run_canonical_gaussian_y_k_selection.py --gate
    python tools/research_audit/run_canonical_gaussian_y_k_selection.py --pilot
    python tools/research_audit/run_canonical_gaussian_y_k_selection.py --full [--fallback]
    python tools/research_audit/run_canonical_gaussian_y_k_selection.py --sensitivity
    python tools/research_audit/run_canonical_gaussian_y_k_selection.py --merge
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
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

from data_generator_canonical_gaussian_y_k import (  # noqa: E402
    GENERATOR_ID,
    first_latent_draw,
    generate_canonical_gaussian_y,
)
from model_dual_expfam_fixed import DualExpFamLSMFixed  # noqa: E402
from utils_expfam import _lnpZ, calc_Q_dual_strict, calc_bic_dual  # noqa: E402

# =====================================================================
# FROZEN EXPERIMENT CONSTANTS  (set before any result was observed)
# =====================================================================

EXPERIMENT_ID = "canonical_gaussian_y_k_selection_20260907"
OUT_DIR = _ROOT / "expfam" / "results" / EXPERIMENT_ID
JST = ZoneInfo("Asia/Tokyo")

FAMILY_X = "bernoulli"
FAMILY_Y = "gaussian"

D_FIXED = 15
L_FIXED = 5
NUM_ITER_FIXED = 8
CANDIDATE_K = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# estimator settings copied from run_fixed_official_exp1_bic_full.py
MAX_RETRIES = 2
NEWTON_ALPHA_BASE = 0.5          # newton_alpha = 0.5 / 2**retry
NEWTON_MAX_ITER = 10
ADAM_MAX_ITER = 50               # calc_w0 / calc_w
INIT_W_GAUSSIAN_Y = 0.5          # historical initialiser constant
INIT_F_SCALE_DISCRETE_X = 0.2    # historical: F *= 0.2 when family_x is discrete

# true parameters -- FROZEN, never changed in response to results
W0_TRUE = 0.0
W_TRUE_PRIMARY = 0.5
SIGMA_Y_TRUE = 0.1
VAR_F = 5.0
UNIQ = 0.1

# grids
PILOT_K0, PILOT_N, PILOT_TRIALS = 3, 150, [0, 1, 2]
FULL_K0_LIST = [1, 3, 5]
FULL_N_LIST = [50, 150, 300]
FULL_TRIALS = list(range(10))
FALLBACK_CELLS = (
    [(k0, 150) for k0 in FULL_K0_LIST]
    + [(3, 50), (3, 300)]
)
SENS_W_LIST = [0.25, 1.0]
SENS_K0, SENS_N, SENS_TRIALS = 3, 150, list(range(10))

TIE_RULE = "smallest_k_among_ties"   # declared in advance; ties recorded separately
RECON_TOLERANCE = 1e-8


# =====================================================================
# seed policy  (deterministic, declared in advance)
# =====================================================================

def core_data_seed(k0: int, n: int, trial: int) -> int:
    return 900000 + k0 * 10000 + n * 10 + trial


def sens_data_seed(wcode: int, trial: int) -> int:
    return 960000 + wcode * 1000 + trial


def model_seed_of(data_seed: int) -> int:
    return data_seed + 50


WCODE = {0.25: 1, 1.0: 2}


def config_id_of(k0: int, n: int, w: float) -> str:
    return f"K{k0}_n{n}_w{w:g}"


def enumerate_cells(phase: str, fallback: bool = False):
    """Return list of (config_id, k0, n, w, trial, data_seed). Pure function of the frozen grid."""
    cells = []
    if phase == "pilot":
        for t in PILOT_TRIALS:
            cells.append((config_id_of(PILOT_K0, PILOT_N, W_TRUE_PRIMARY), PILOT_K0,
                          PILOT_N, W_TRUE_PRIMARY, t, core_data_seed(PILOT_K0, PILOT_N, t)))
        return cells

    if phase == "full":
        grid = FALLBACK_CELLS if fallback else [(k0, n) for k0 in FULL_K0_LIST for n in FULL_N_LIST]
        seen = set()
        for k0, n in grid:
            if (k0, n) in seen:
                continue
            seen.add((k0, n))
            for t in FULL_TRIALS:
                # pilot cells are reused, not re-run
                if k0 == PILOT_K0 and n == PILOT_N and t in PILOT_TRIALS:
                    continue
                cells.append((config_id_of(k0, n, W_TRUE_PRIMARY), k0, n, W_TRUE_PRIMARY,
                              t, core_data_seed(k0, n, t)))
        return cells

    if phase == "sensitivity":
        for w in SENS_W_LIST:
            for t in SENS_TRIALS:
                cells.append((config_id_of(SENS_K0, SENS_N, w), SENS_K0, SENS_N, w,
                              t, sens_data_seed(WCODE[w], t)))
        return cells

    raise ValueError(phase)


def check_seed_collisions() -> dict:
    """All (config_id, trial) must map to distinct data seeds; and no seed reused across configs."""
    allc = (enumerate_cells("pilot") + enumerate_cells("full")
            + enumerate_cells("sensitivity"))
    seeds = {}
    collisions = []
    for cfg, k0, n, w, t, ds in allc:
        if ds in seeds and seeds[ds] != (cfg, t):
            collisions.append({"seed": ds, "a": seeds[ds], "b": (cfg, t)})
        seeds[ds] = (cfg, t)
    return {"n_cells": len(allc), "n_unique_seeds": len(seeds), "collisions": collisions}


# =====================================================================
# theory gate
# =====================================================================

def build_theory_gate() -> pd.DataFrame:
    import data_generator_canonical_gaussian_y_k as gmod

    src = inspect.getsource(gmod.generate_canonical_gaussian_y)
    probe_seed = 20260907
    probe = generate_canonical_gaussian_y(n=30, d=D_FIXED, k=3, seed=probe_seed,
                                          w0_true=W0_TRUE, w_true=W_TRUE_PRIMARY,
                                          sigma_y_true=SIGMA_Y_TRUE)
    expected_Z = first_latent_draw(30, 3, probe_seed)

    # G8: estimator-side var_z fixed at 1.0
    m = DualExpFamLSMFixed(n=10, d=D_FIXED, k=3, L=L_FIXED,
                           family_x=FAMILY_X, family_y=FAMILY_Y)
    m.initialize_params(true_params={"var_f": VAR_F}, seed=1)
    var_z_val = float(m.params["var_z"])

    # G10: the historical module must not be reachable from this generator
    hist_used = ("data_generator_expfam" in src) or ("generate_dual_data" in src)

    # G6/G7 probe: two different candidate K must not change the dataset
    # (the runner generates once per trial; here we assert the generator has no k_est arg)
    gen_params = list(inspect.signature(generate_canonical_gaussian_y).parameters)

    # G3: sigma_y=0 reproduces the deterministic mean exactly -> additive noise model
    p0 = generate_canonical_gaussian_y(n=30, d=D_FIXED, k=3, seed=probe_seed, sigma_y_true=0.0)
    um = np.triu(np.ones((30, 30), dtype=bool), k=1)
    eta0 = p0["w0"] + p0["w"] * (p0["Z"] @ p0["Z"].T)
    additive_ok = bool(np.allclose(p0["Y"][um], eta0[um], atol=1e-12))

    # G5: Y not renormalised -> Y mean/var free to vary (structural: no token in source)
    y_post_ok = not any(tok in src for tok in ("normalize_zscore(Y", "Y = (Y -", "Y /= "))

    gates = [
        ("G1", "canonical_z_iid_gaussian",
         bool(np.array_equal(probe["Z"], expected_Z)),
         "generator Z is bit-identical to default_rng(seed).normal(0,1,(n,K))"),
        ("G2", "no_generator_z_postprocessing",
         not any(t in src for t in ("normalize_zscore", "standardize", "whiten", "scale_Z")),
         "no standardization/whitening/scaling token appears in the generator body"),
        ("G3", "gaussian_y_additive_noise", additive_ok,
         "sigma_y=0 yields Y == w0 + w z_i^T z_j exactly (additive Gaussian noise)"),
        ("G4", "scalar_w_nonzero",
         isinstance(W_TRUE_PRIMARY, float) and W_TRUE_PRIMARY != 0.0,
         f"w is a python scalar and w={W_TRUE_PRIMARY} != 0"),
        ("G5", "no_y_postprocessing", y_post_ok,
         "no sample normalization of Y after generation"),
        ("G6", "same_dataset_across_candidate_k", "k_est" not in gen_params,
         "generator takes no candidate-K argument; runner generates once per trial "
         "and asserts hash equality across k_est at run time"),
        ("G7", "K_GEN_not_changed_by_candidate", True,
         "candidate K is passed only to DualExpFamLSMFixed(k=k_est); generator k is fixed per trial"),
        ("G8", "model_prior_var_fixed", var_z_val == 1.0,
         f"estimator params['var_z'] == {var_z_val} (no M-step exists for var_z)"),
        ("G9", "candidate_contains_truth",
         all(k0 in CANDIDATE_K for k0 in set(FULL_K0_LIST + [PILOT_K0, SENS_K0])),
         f"candidate K = {CANDIDATE_K} contains every K_GEN used"),
        ("G10", "historical_generator_not_used", not hist_used,
         "generator source references neither data_generator_expfam nor generate_dual_data"),
    ]
    rows = [{"gate_id": g, "gate_name": nm, "status": "PASS" if ok else "FAIL",
             "evidence": ev} for g, nm, ok, ev in gates]
    return pd.DataFrame(rows)


# =====================================================================
# one EM fit  (mirrors run_fixed_official_exp1_bic_full.run_em_fixed_bic)
# =====================================================================

def run_one_fit(X, Y, true_params, k_est, n, d, seed_model):
    """Returns dict of results. Never raises; failures are recorded."""
    t0 = time.perf_counter()
    out = {
        "success": False, "failure_reason": "", "nan_occurred": True,
        "internal_retry": 0,
    }
    try:
        for retry in range(MAX_RETRIES + 1):
            newton_alpha = NEWTON_ALPHA_BASE / (2 ** retry)
            rng = np.random.default_rng(seed_model + retry * 1000)

            model = DualExpFamLSMFixed(n=n, d=d, k=k_est, L=L_FIXED,
                                       family_x=FAMILY_X, family_y=FAMILY_Y)
            model.initialize_params(true_params=true_params, seed=seed_model + retry * 1000)

            upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            upper_vals = Y[upper_mask]
            # Gaussian-Y initialiser, verbatim from the historical runner
            model.params["w0"] = float(upper_vals.mean())
            model.params["w"] = INIT_W_GAUSSIAN_Y
            model.sigma_y = float(max(upper_vals.std(), 0.01))
            # discrete X initialiser, verbatim from the historical runner
            model.params["F"] *= INIT_F_SCALE_DISCRETE_X

            Z = model.params["Z"].copy()
            F = model.params["F"].copy()
            sigma = model.params["sigma"].copy()
            w0 = model.params["w0"]
            w = model.params["w"]
            var_z = model.params["var_z"]

            Z_prev = Z.copy()
            nan_count = 0

            for _ in range(NUM_ITER_FIXED):
                Z_samples = np.zeros((n, k_est, L_FIXED))
                for l in range(L_FIXED):
                    model.params.update(dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w))
                    Z_new = model.calc_eta_newton(X, Y, rng=rng,
                                                  max_iter=NEWTON_MAX_ITER,
                                                  alpha=newton_alpha)
                    Z_samples[:, :, l] = Z_new
                    Z = Z_new.copy()

                if np.any(np.isnan(Z_samples)) or np.any(np.isinf(Z_samples)):
                    nan_count += 1
                    Z_samples = np.stack([Z_prev] * L_FIXED, axis=2)
                    Z = Z_prev.copy()

                Z_samples = model.scale_Z(Z_samples)
                Z_prev = Z.copy()
                Z = Z_samples[:, :, -1].copy()

                F = model.calc_F(X, Z_samples)
                sigma = model.calc_sigma(X, Z_samples, F)
                w0 = model.calc_w0(Y, Z_samples, w0, w, max_iter=ADAM_MAX_ITER)
                w = model.calc_w(Y, Z_samples, w0, w, max_iter=ADAM_MAX_ITER)
                model.calc_sigma_y(Y, Z_samples, w0, w)

            out["internal_retry"] = retry
            if nan_count == 0:
                break

        model.params.update({"F": F, "sigma": sigma, "w0": w0, "w": w})

        # ---- criterion (library call, unmodified) --------------------------
        q_strict = calc_Q_dual_strict(X, Y, Z_samples, F, sigma, var_z, w0, w, model)
        score, p_hat = calc_bic_dual(Q_strict=q_strict, k=k_est, n=n, d=d,
                                     family_x=FAMILY_X, family_y=FAMILY_Y)

        # ---- score decomposition (no RNG consumed; post-hoc instrumentation)
        s_pz = s_px = s_py = 0.0
        for l in range(L_FIXED):
            Zl = Z_samples[:, :, l]
            s_pz += _lnpZ(Zl, var_z)
            s_px += model.calc_log_likelihood_X(X, Z_samples[:, :, l:l + 1], F)
            s_py += model.calc_log_likelihood_Y(Y, Z_samples[:, :, l:l + 1], w0, w)
        mean_lpz = s_pz / L_FIXED
        mean_lpx = s_px / L_FIXED
        mean_lpy = s_py / L_FIXED

        penalty = float(p_hat) * float(np.log(n))
        fit_only = -2.0 * (mean_lpx + mean_lpy)          # D_K
        direct_prior = -2.0 * mean_lpz                   # P_Z(K)
        reconstructed = fit_only + direct_prior + penalty
        residual = abs(reconstructed - score)

        # ---- descriptive recovery diagnostics ------------------------------
        Z_est = Z_samples[:, :, -1]

        out.update({
            "success": True,
            "nan_occurred": nan_count > 0,
            "q_strict": float(q_strict),
            "criterion_score": float(score),
            "parameter_count": int(p_hat),
            "penalty_term": penalty,
            "mean_log_pZ": float(mean_lpz),
            "mean_log_pX_given_Z": float(mean_lpx),
            "mean_log_pY_given_Z": float(mean_lpy),
            "fit_only_score": float(fit_only),
            "direct_prior_score": float(direct_prior),
            "score_without_direct_prior": float(fit_only + penalty),
            "reconstructed_score": float(reconstructed),
            "reconstruction_residual": float(residual),
            "w0_hat": float(w0),
            "w_hat": float(w),
            "sigma_y_hat": float(model.sigma_y),
            "z_est_frob_sq": float(np.sum(Z_est ** 2)),
        })
    except Exception as exc:  # noqa: BLE001
        out["failure_reason"] = f"{type(exc).__name__}: {exc}"[:300]
        traceback.print_exc()
        for key in ("q_strict", "criterion_score", "penalty_term", "mean_log_pZ",
                    "mean_log_pX_given_Z", "mean_log_pY_given_Z", "fit_only_score",
                    "direct_prior_score", "score_without_direct_prior",
                    "reconstructed_score", "reconstruction_residual",
                    "w0_hat", "w_hat", "sigma_y_hat", "z_est_frob_sq"):
            out[key] = np.nan
        out["parameter_count"] = -1

    out["runtime_seconds"] = time.perf_counter() - t0
    return out


# =====================================================================
# phase driver
# =====================================================================

def run_phase(phase: str, fallback: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = enumerate_cells(phase, fallback=fallback)
    total = len(cells) * len(CANDIDATE_K)
    print(f"[{phase}] {len(cells)} datasets x {len(CANDIDATE_K)} candidate K = {total} fits")

    fit_rows, gen_rows = [], []
    t_start = time.perf_counter()
    done = 0

    for cfg, k0, n, w, trial, ds in cells:
        ms = model_seed_of(ds)
        data = generate_canonical_gaussian_y(
            n=n, d=D_FIXED, k=k0, seed=ds,
            w0_true=W0_TRUE, w_true=w, sigma_y_true=SIGMA_Y_TRUE,
            var_f=VAR_F, uniq=UNIQ,
        )
        X, Y = data["X"], data["Y"]
        hx, hy = data["dataset_hash_X"], data["dataset_hash_Y"]

        gen_rows.append({
            "experiment_id": EXPERIMENT_ID, "phase": phase, "config_id": cfg,
            "trial": trial, "data_seed": ds, "n": n, "d": D_FIXED, "K_GEN": k0,
            "w_true": w, "w0_true": W0_TRUE, "sigma_y_true": SIGMA_Y_TRUE,
            "generator_id": GENERATOR_ID,
            **{key: data[key] for key in (
                "rank_Z", "sigma_min_Z", "sigma_max_Z",
                "rank_F", "sigma_min_F", "sigma_max_F", "condition_number_F",
                "z_col_mean_max_abs", "z_col_var_min", "z_col_var_max",
                "z_frobenius_sq", "z_frobenius_sq_over_nK",
                "x_mean", "y_mean", "y_variance", "n_pairs",
                "dataset_hash_X", "dataset_hash_Y", "dataset_hash_Z")},
        })

        for k_est in CANDIDATE_K:
            done += 1
            # G6 run-time assertion: the arrays handed to every candidate K are identical
            assert hashlib.sha256(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest() == hx
            assert hashlib.sha256(np.ascontiguousarray(Y, dtype=np.float64).tobytes()).hexdigest() == hy

            res = run_one_fit(X, Y, data, k_est, n, D_FIXED, ms)
            row = {
                "experiment_id": EXPERIMENT_ID, "phase": phase, "config_id": cfg,
                "trial": trial, "data_seed": ds, "model_seed": ms,
                "n": n, "d": D_FIXED, "K_GEN": k0, "K_0": k0,
                "family_x": FAMILY_X, "family_y": FAMILY_Y,
                "w0_true": W0_TRUE, "w_true": w, "sigma_y_true": SIGMA_Y_TRUE,
                "k_est": k_est, "L": L_FIXED, "num_iter": NUM_ITER_FIXED,
                "dataset_hash_X": hx, "dataset_hash_Y": hy,
                "generator_id": GENERATOR_ID,
                "estimator_class": "DualExpFamLSMFixed",
                "criterion_fn": "calc_bic_dual",
            }
            row.update(res)
            fit_rows.append(row)

            el = (time.perf_counter() - t_start) / 60.0
            print(f"  [{done:4d}/{total}] {cfg} t={trial} k={k_est} "
                  f"score={res.get('criterion_score', float('nan')):.1f} "
                  f"ok={res['success']} [{el:.1f} min]", flush=True)

    df = pd.DataFrame(fit_rows)
    df.to_csv(OUT_DIR / f"fit_results_{phase}.csv", index=False)
    pd.DataFrame(gen_rows).to_csv(OUT_DIR / f"generator_diagnostics_{phase}.csv", index=False)

    el = (time.perf_counter() - t_start) / 60.0
    med = float(df["runtime_seconds"].median())
    print(f"[{phase}] done: {len(df)} fits in {el:.1f} min; median {med:.2f} s/fit")
    print(f"[{phase}] wrote {OUT_DIR / f'fit_results_{phase}.csv'}")


def merge_phases() -> None:
    parts, gparts = [], []
    for p in ("pilot", "full", "sensitivity"):
        f = OUT_DIR / f"fit_results_{p}.csv"
        g = OUT_DIR / f"generator_diagnostics_{p}.csv"
        if f.exists():
            parts.append(pd.read_csv(f))
        if g.exists():
            gparts.append(pd.read_csv(g))
    if not parts:
        raise SystemExit("no phase files to merge")
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["config_id", "trial", "k_est"]).reset_index(drop=True)
    df["selected_k_flag"] = False
    for (cfg, tr), sub in df.groupby(["config_id", "trial"], sort=False):
        ok = sub[sub["success"] & sub["criterion_score"].notna()]
        if len(ok) == 0:
            continue
        mn = ok["criterion_score"].min()
        ties = ok[ok["criterion_score"] == mn]
        sel = int(ties["k_est"].min())          # TIE_RULE
        df.loc[(df["config_id"] == cfg) & (df["trial"] == tr)
               & (df["k_est"] == sel), "selected_k_flag"] = True
    df.to_csv(OUT_DIR / "fit_results.csv", index=False)
    pd.concat(gparts, ignore_index=True).to_csv(OUT_DIR / "generator_diagnostics.csv", index=False)
    print(f"merged {len(df)} fits -> {OUT_DIR / 'fit_results.csv'}")


# =====================================================================
# manifest
# =====================================================================

def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return "UNAVAILABLE"


def write_manifest(extra: dict | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib, scipy  # noqa: E402
    man = {
        "experiment_id": EXPERIMENT_ID,
        "generator_id": GENERATOR_ID,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "head_sha": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain"),
        "timestamp_jst": datetime.now(JST).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "frozen_settings": {
            "family_x": FAMILY_X, "family_y": FAMILY_Y, "d": D_FIXED,
            "L": L_FIXED, "num_iter": NUM_ITER_FIXED, "candidate_K": CANDIDATE_K,
            "w0_true": W0_TRUE, "w_true_primary": W_TRUE_PRIMARY,
            "sigma_y_true": SIGMA_Y_TRUE, "var_f": VAR_F, "uniq": UNIQ,
            "max_retries": MAX_RETRIES, "newton_alpha_base": NEWTON_ALPHA_BASE,
            "newton_max_iter": NEWTON_MAX_ITER, "adam_max_iter": ADAM_MAX_ITER,
            "init_w_gaussian_y": INIT_W_GAUSSIAN_Y,
            "init_F_scale_discrete_x": INIT_F_SCALE_DISCRETE_X,
            "tie_rule": TIE_RULE, "recon_tolerance": RECON_TOLERANCE,
        },
        "seed_policy": {
            "core": "data_seed = 900000 + K_GEN*10000 + n*10 + trial",
            "sensitivity": "data_seed = 960000 + wcode*1000 + trial (wcode: 0.25->1, 1.0->2)",
            "model": "model_seed = data_seed + 50 (independent of k_est)",
        },
        "seed_collision_check": check_seed_collisions(),
        "estimator": {
            "class": "DualExpFamLSMFixed",
            "module": "expfam/src/model_dual_expfam_fixed.py",
            "lineage": "fixed (Y-side 0.5 removed)",
            "criterion_fn": "calc_bic_dual (Q-based complete-data criterion; NOT Schwarz BIC, KI-010)",
        },
    }
    if extra:
        man.update(extra)
    p = OUT_DIR / "run_manifest.json"
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        old.setdefault("history", []).append({k: v for k, v in old.items() if k != "history"})
        man["history"] = old.get("history", [])
    p.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {p}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    a = ap.parse_args()

    if a.gate:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        g = build_theory_gate()
        g.to_csv(OUT_DIR / "theory_gate.csv", index=False)
        print(g.to_string(index=False))
        n_fail = int((g["status"] == "FAIL").sum())
        print(f"\nTHEORY GATE: {len(g) - n_fail}/{len(g)} PASS")
        col = check_seed_collisions()
        print(f"seed check: {col['n_cells']} cells, {col['n_unique_seeds']} unique seeds, "
              f"{len(col['collisions'])} collisions")
        if n_fail or col["collisions"]:
            return 1
    if a.manifest:
        write_manifest()
    if a.pilot:
        run_phase("pilot")
    if a.full:
        run_phase("full", fallback=a.fallback)
    if a.sensitivity:
        run_phase("sensitivity")
    if a.merge:
        merge_phases()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
