"""
Baseline (Mikawa 2024) wrapper for fair comparison with Dual-ExpFam LSM.

Provides:
  run_baseline_on_dual_data : run LatentStructuralModel on data from
      generate_dual_data, return metrics in the same format as run_em_dual.
  run_comparison_exp1_control : Exp1 (k variation) for Control scenario
      (Gauss-X × Bern-Y) — main fair comparison.
  run_comparison_exp1_scenario_a : Exp1 for Scenario A (Pois-X × Bern-Y)
      — auxiliary mismatch comparison.

Policy:
  - Scenarios B (Pois-Y) and C (Gauss-Y) are NOT applicable to Baseline
    because Baseline requires binary Y ∈ {0,1}.
  - RMSE(Y) is computed as mu_est vs mu_true (true mean, not raw observation),
    consistent with run_em_dual.
  - Procrustes rotation applied to both RMSE(Z) and RMSE(F).
  - Data seeds and model seeds are matched to Dual-ExpFam conventions:
    data_seed = base_seed + trial * 100, model_seed = data_seed + 50.
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

_REPRO_SRC = Path(__file__).parent
_EXPFAM_SRC = _REPRO_SRC.parent.parent / "expfam" / "src"
sys.path.insert(0, str(_REPRO_SRC))
sys.path.insert(0, str(_EXPFAM_SRC))

from model import LatentStructuralModel                      # baseline
from data_generator_expfam import generate_dual_data         # shared
from utils_expfam import procrustes_rotation, calc_rmse      # shared helpers

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _calc_log_likelihood_baseline(X, Y, Z, F, sigma, var_z, w0, w):
    """
    Complete-data log-likelihood for Baseline (Gauss-X × Bern-Y).
    Used for BIC computation.
    """
    n, k = Z.shape
    d = X.shape[1]

    # ln p(Z)
    ln_p_Z = (-(n * k / 2) * np.log(2 * np.pi)
              - (n * k / 2) * np.log(var_z)
              - (1 / (2 * var_z)) * np.sum(Z ** 2))

    # ln p(X|Z) — Gaussian
    resid = X - Z @ F.T
    sd = np.diag(sigma)
    ln_p_X = (-(n * d / 2) * np.log(2 * np.pi)
              - (n / 2) * np.sum(np.log(sd))
              - 0.5 * np.sum(resid ** 2 / sd))

    # ln p(Y|Z) — Bernoulli (upper triangle only, ×0.5 for symmetry)
    ZZT = Z @ Z.T
    logits = w0 + w * ZZT
    S = _sigmoid(logits)
    S = np.clip(S, 1e-10, 1 - 1e-10)
    ln_p_Y = Y * np.log(S) + (1 - Y) * np.log(1 - S)
    np.fill_diagonal(ln_p_Y, 0.0)
    ln_p_Y = 0.5 * np.sum(ln_p_Y)

    return ln_p_Z + ln_p_X + ln_p_Y


def _calc_bic_baseline(log_likelihood, k, n, d):
    """
    BIC = -2 logL + num_params * ln(n).
    num_params = (k+1)*d - k*(k-1)//2   (same as Dual-ExpFam Gauss-X Bern-Y).
    """
    num_params = (k + 1) * d - k * (k - 1) // 2
    return -2.0 * log_likelihood + num_params * np.log(n)


# ─────────────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────────────

def run_baseline_on_dual_data(
    X: np.ndarray,
    Y: np.ndarray,
    true_params: dict,
    k: int,
    L: int = 5,
    num_iter: int = 8,
    seed: int = 42,
    x_is_misspecified: bool = False,
) -> dict:
    """
    Run LatentStructuralModel on data from generate_dual_data.

    Returns metrics dict compatible with run_em_dual output.

    Parameters
    ----------
    X, Y : np.ndarray
        Data matrices from generate_dual_data.
    true_params : dict
        Ground-truth dict (keys: Z, F, sigma, w0, w, ...).
    k : int
        Estimated latent dimensionality.
    L : int
        Number of Monte Carlo samples per E-step.
    num_iter : int
        Number of EM iterations.
    seed : int
        Random seed for initialization and E-step sampling.
    x_is_misspecified : bool
        Flag: True if X comes from a non-Gaussian distribution
        (Baseline still runs but its X likelihood is misspecified).

    Raises
    ------
    ValueError
        If Y contains non-binary values (Baseline requires binary Y).
    """
    n, d = X.shape

    # Validate Y is binary — hard requirement for Baseline
    unique_y = np.unique(Y[np.triu(np.ones((n, n), dtype=bool), k=1)])
    if not np.all(np.isin(unique_y, [0.0, 1.0])):
        raise ValueError(
            f"Baseline requires binary Y ∈ {{0,1}}. "
            f"Found: {sorted(set(unique_y[:10].tolist()))}... "
            "Use Dual-ExpFam instead, or log this case as 'Not applicable'."
        )

    model = LatentStructuralModel(n=n, d=d, k=k, L=L)
    model.initialize_params(true_params=true_params, seed=seed)

    # Informed initialisation (same strategy as baseline experiments)
    density = float(np.clip(np.mean(Y), 1e-6, 1 - 1e-6))
    model.params["w0"] = np.log(density / (1 - density))
    model.params["w"] = 0.5

    rng = np.random.default_rng(seed)

    Z = model.params["Z"].copy()
    F = model.params["F"].copy()
    sigma = model.params["sigma"].copy()
    w0 = model.params["w0"]
    w = model.params["w"]
    var_z = model.params["var_z"]

    best_logL = -np.inf
    best_state = None

    for iteration in range(1, num_iter + 1):
        # E-step
        Z_samples = np.zeros((n, k, L))
        for l in range(L):
            model.params.update(dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w))
            Z_new = model.calc_eta_newton(
                X, Y, rng=rng, max_iter=10, alpha=0.5
            )
            Z_samples[:, :, l] = Z_new
            Z = Z_new.copy()

        Z_samples = model.scale_Z(Z_samples)
        Z = Z_samples[:, :, -1].copy()

        # M-step
        F = model.calc_F(X, Z_samples)
        sigma = model.calc_sigma(X, Z_samples, F)
        w0 = model.calc_w0(Y, Z_samples, w0, w, max_iter=50)
        w = model.calc_w(Y, Z_samples, w0, w, max_iter=50)

        # Track best log-likelihood
        logL = _calc_log_likelihood_baseline(X, Y, Z, F, sigma, var_z, w0, w)
        if logL > best_logL:
            best_logL = logL
            best_state = dict(
                Z=Z.copy(), F=F.copy(), sigma=sigma.copy(), w0=w0, w=w
            )

    # Use best state
    Z_est = best_state["Z"]
    F_est = best_state["F"]
    sigma_est = best_state["sigma"]
    w0_est = best_state["w0"]
    w_est = best_state["w"]

    # BIC
    bic = _calc_bic_baseline(best_logL, k, n, d)

    # RMSE(X): reconstruction using Gaussian assumption
    X_recon = Z_est @ F_est.T
    rmse_X = calc_rmse(X, X_recon)

    # Ground-truth metrics
    has_truth = (
        true_params is not None
        and "Z" in true_params
        and true_params["Z"] is not None
    )

    if has_truth:
        R, k_min = procrustes_rotation(Z_est, true_params["Z"])
        Z_rot = Z_est[:, :k_min] @ R
        rmse_Z = calc_rmse(true_params["Z"][:, :k_min], Z_rot)

        F_aligned = F_est[:, :k_min] @ R
        rmse_F = calc_rmse(true_params["F"][:, :k_min], F_aligned)

        rmse_sigma = calc_rmse(np.diag(sigma_est), np.diag(true_params["sigma"]))

        # RMSE(Y): mu_est vs mu_true (Bernoulli mean, consistent with run_em_dual)
        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        eta_est = float(w0_est) + float(w_est) * (Z_est @ Z_est.T)
        mu_est = _sigmoid(np.clip(eta_est, -500, 500))
        eta_true = float(true_params["w0"]) + float(true_params["w"]) * (
            true_params["Z"] @ true_params["Z"].T)
        mu_true = _sigmoid(np.clip(eta_true, -500, 500))
        rmse_Y = calc_rmse(mu_est[upper_mask], mu_true[upper_mask])

        w0_err = abs(float(w0_est) - float(true_params.get("w0", 0.0)))
        w_err = abs(float(w_est) - float(true_params.get("w", 0.5)))
    else:
        rmse_Z = rmse_F = rmse_sigma = rmse_Y = float("nan")
        w0_err = w_err = float("nan")

    return {
        "model": "baseline",
        "x_misspecified": x_is_misspecified,
        "k": k,
        "n": n,
        "rmse_Z": rmse_Z,
        "rmse_X": rmse_X,
        "rmse_F": rmse_F,
        "rmse_sigma": rmse_sigma,
        "rmse_Y": rmse_Y,
        "w0_err": w0_err,
        "w_err": w_err,
        "w0_est": float(w0_est),
        "w_est": float(w_est),
        "log_likelihood": best_logL,
        "BIC": bic,
    }


# ─────────────────────────────────────────────────────────────────────
# Experiment runners
# ─────────────────────────────────────────────────────────────────────

def run_comparison_exp1_control(
    N_TRIALS: int = 5,
    L: int = 5,
    num_iter: int = 8,
    base_seed: int = 4000,
    k_list: list = None,
    n: int = 150,
    d: int = 15,
    k_true: int = 3,
) -> pd.DataFrame:
    """
    Exp 1 (k variation) for Control scenario: True X=Gaussian, True Y=Bernoulli.

    Main fair comparison between Baseline and Dual-ExpFam.
    Both use the same data (from generate_dual_data), same L, num_iter, seeds.
    Returns DataFrame with one row per (k_est, trial, model).
    """
    from utils_expfam import run_em_dual, calc_bic_dual

    if k_list is None:
        k_list = [1, 2, 3, 4, 5, 6]

    records = []
    total = len(k_list) * N_TRIALS
    done = 0

    for k_est in k_list:
        for trial in range(N_TRIALS):
            done += 1
            dseed = base_seed + trial * 100
            mseed = dseed + 50

            # Generate data — True X=Gaussian, True Y=Bernoulli
            data = generate_dual_data(
                n=n, d=d, k=k_true, seed=dseed,
                family_x="gaussian", family_y="bernoulli",
            )

            print(f"  [{done:3d}/{total}] Control k={k_est} t={trial}", end="  ")
            t0 = time.perf_counter()

            # ── Dual-ExpFam (Gauss-X, Bern-Y) ────────────────────────
            res_dual = run_em_dual(
                X=data["X"], Y=data["Y"], true_params=data,
                family_x="gaussian", family_y="bernoulli",
                k=k_est, L=L, num_iter=num_iter, seed=mseed,
                compute_strict_Q=True,
            )
            bic_dual, _ = calc_bic_dual(
                Q_strict=res_dual["Q_strict"], k=k_est, n=n, d=d,
                family_x="gaussian", family_y="bernoulli",
            )

            # ── Baseline ──────────────────────────────────────────────
            res_bl = run_baseline_on_dual_data(
                X=data["X"], Y=data["Y"], true_params=data,
                k=k_est, L=L, num_iter=num_iter, seed=mseed,
                x_is_misspecified=False,
            )

            elapsed = time.perf_counter() - t0
            print(
                f"Dual RMSE(Z)={res_dual['rmse_Z']:.4f} BIC={bic_dual:.0f}  "
                f"Base RMSE(Z)={res_bl['rmse_Z']:.4f} BIC={res_bl['BIC']:.0f}  "
                f"({elapsed:.1f}s)"
            )

            # Record Dual-ExpFam
            records.append({
                "scenario": "Control",
                "model": "dual_expfam",
                "family_x": "gaussian", "family_y": "bernoulli",
                "k_est": k_est, "k_true": k_true, "trial": trial,
                "n": n, "d": d,
                "rmse_Z":     res_dual["rmse_Z"],
                "rmse_F":     res_dual["rmse_F"],
                "rmse_Y":     res_dual["rmse_Y"],
                "rmse_X":     res_dual["rmse_X"],
                "rmse_sigma": res_dual["rmse_sigma"],
                "w0_err":     res_dual["w0_err"],
                "w_err":      res_dual["w_err"],
                "BIC":        bic_dual,
                "Q_strict":   res_dual["Q_strict"],
                "x_misspecified": False,
                "comparison_type": "main",
            })

            # Record Baseline
            records.append({
                "scenario": "Control",
                "model": "baseline",
                "family_x": "gaussian", "family_y": "bernoulli",
                "k_est": k_est, "k_true": k_true, "trial": trial,
                "n": n, "d": d,
                "rmse_Z":     res_bl["rmse_Z"],
                "rmse_F":     res_bl["rmse_F"],
                "rmse_Y":     res_bl["rmse_Y"],
                "rmse_X":     res_bl["rmse_X"],
                "rmse_sigma": res_bl["rmse_sigma"],
                "w0_err":     res_bl["w0_err"],
                "w_err":      res_bl["w_err"],
                "BIC":        res_bl["BIC"],
                "Q_strict":   res_bl["log_likelihood"],
                "x_misspecified": False,
                "comparison_type": "main",
            })

    return pd.DataFrame(records)


def run_comparison_exp1_scenario_a(
    N_TRIALS: int = 5,
    L: int = 5,
    num_iter: int = 8,
    base_seed: int = 4000,
    k_list: list = None,
    n: int = 150,
    d: int = 15,
    k_true: int = 3,
) -> pd.DataFrame:
    """
    Exp 1 (k variation) for Scenario A: True X=Poisson, True Y=Bernoulli.

    Auxiliary comparison:
    - Dual-ExpFam(poisson, bernoulli): correct model
    - Baseline: X is misspecified (Gaussian assumption on Poisson data)

    This is labeled as 'auxiliary' and should NOT be used to claim
    Dual-ExpFam is fundamentally superior; it shows X-mismatch robustness.
    """
    from utils_expfam import run_em_dual, calc_bic_dual

    if k_list is None:
        k_list = [1, 2, 3, 4, 5, 6]

    records = []
    total = len(k_list) * N_TRIALS
    done = 0

    for k_est in k_list:
        for trial in range(N_TRIALS):
            done += 1
            dseed = base_seed + trial * 100
            mseed = dseed + 50

            # Generate data — True X=Poisson, True Y=Bernoulli (Scenario A)
            data = generate_dual_data(
                n=n, d=d, k=k_true, seed=dseed,
                family_x="poisson", family_y="bernoulli",
            )

            print(f"  [{done:3d}/{total}] ScenA k={k_est} t={trial}", end="  ")
            t0 = time.perf_counter()

            # ── Dual-ExpFam correct (Pois-X, Bern-Y) ─────────────────
            res_dual = run_em_dual(
                X=data["X"], Y=data["Y"], true_params=data,
                family_x="poisson", family_y="bernoulli",
                k=k_est, L=L, num_iter=num_iter, seed=mseed,
                compute_strict_Q=True,
            )
            bic_dual, _ = calc_bic_dual(
                Q_strict=res_dual["Q_strict"], k=k_est, n=n, d=d,
                family_x="poisson", family_y="bernoulli",
            )

            # ── Baseline (X misspecified: Gaussian assumption on Poisson data) ──
            try:
                res_bl = run_baseline_on_dual_data(
                    X=data["X"], Y=data["Y"], true_params=data,
                    k=k_est, L=L, num_iter=num_iter, seed=mseed,
                    x_is_misspecified=True,  # explicitly flag
                )
                bl_status = "ok"
            except ValueError as e:
                print(f"\n  [SKIP] Baseline not applicable: {e}")
                bl_status = "not_applicable"
                res_bl = {
                    "rmse_Z": float("nan"), "rmse_F": float("nan"),
                    "rmse_Y": float("nan"), "rmse_X": float("nan"),
                    "rmse_sigma": float("nan"),
                    "w0_err": float("nan"), "w_err": float("nan"),
                    "BIC": float("nan"), "log_likelihood": float("nan"),
                }

            elapsed = time.perf_counter() - t0
            print(
                f"Dual RMSE(Z)={res_dual['rmse_Z']:.4f}  "
                f"Base RMSE(Z)={res_bl['rmse_Z']:.4f}  "
                f"({elapsed:.1f}s)"
            )

            # Record Dual-ExpFam (correct)
            records.append({
                "scenario": "A_Pois_Bern",
                "model": "dual_expfam",
                "family_x": "poisson", "family_y": "bernoulli",
                "k_est": k_est, "k_true": k_true, "trial": trial,
                "n": n, "d": d,
                "rmse_Z":     res_dual["rmse_Z"],
                "rmse_F":     res_dual["rmse_F"],
                "rmse_Y":     res_dual["rmse_Y"],
                "rmse_X":     res_dual["rmse_X"],
                "rmse_sigma": res_dual["rmse_sigma"],
                "w0_err":     res_dual["w0_err"],
                "w_err":      res_dual["w_err"],
                "BIC":        bic_dual,
                "Q_strict":   res_dual["Q_strict"],
                "x_misspecified": False,
                "comparison_type": "auxiliary",
            })

            # Record Baseline (X misspecified)
            records.append({
                "scenario": "A_Pois_Bern",
                "model": "baseline_X_misspecified",
                "family_x": "gaussian(assumed)", "family_y": "bernoulli",
                "k_est": k_est, "k_true": k_true, "trial": trial,
                "n": n, "d": d,
                "rmse_Z":     res_bl["rmse_Z"],
                "rmse_F":     res_bl["rmse_F"],
                "rmse_Y":     res_bl["rmse_Y"],
                "rmse_X":     res_bl["rmse_X"],
                "rmse_sigma": res_bl.get("rmse_sigma", float("nan")),
                "w0_err":     res_bl["w0_err"],
                "w_err":      res_bl["w_err"],
                "BIC":        res_bl["BIC"],
                "Q_strict":   res_bl.get("log_likelihood", float("nan")),
                "x_misspecified": True,
                "comparison_type": "auxiliary",
                "status": bl_status,
            })

    return pd.DataFrame(records)
