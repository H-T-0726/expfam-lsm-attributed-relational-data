"""Issue #128: evaluate several K-selection criteria on ONE fitted state.

Fit once -> evaluate many.  Every function here is a deterministic function of
(fitted theta, latent samples / starting Z, X, Y, mask).  Nothing fits, samples
or changes the estimator.

Criteria (names are fixed; see CLAUDE.md §5 and the Issue #128 scope):

- ``C_Q``   legacy Q-based complete-data / ICL-type criterion
            ``-2 Q_strict + p_K ln n`` = ``D_K + P_Z + P_theta``.
            NOT a Schwarz BIC.
- ``C_Lap`` Phase 9 Candidate B (Laplace-evidence candidate)
            ``-2 l_Lap + d_K ln N`` = ``fit_mode + volume + P_theta``,
            ``fit_mode = -2[l_X + l_Y](Z_hat)``,
            ``volume  = ||Z_hat||^2 / var_z + nK ln var_z + ln|H(Z_hat)|``.
            NOT an exact marginal likelihood.
- ``C_cond_MC`` / ``C_cond_last`` / ``C_cond_mode``
            paper-inspired conditional diagnostics
            ``-2 l_data(Z*) + P_theta`` at three latent evaluation points
            (MC average over the E-step samples / last E-step sample /
            joint mode).  NONE of them is the source paper's BIC: the paper
            text does not identify its evaluation point for Z.

``l_data(Z)`` always means ``log p(X, Y | Z, theta)`` with every normalising
constant that ``eval_utils.calc_Q_dual_strict_exp`` includes, so all three
criteria share one likelihood convention.  Only Bernoulli Y is supported here
(the Y-side strict corrections are then zero).

Lineage E (experimental prototype).  MANUSCRIPT_ADOPTION_STATUS:
NOT_DECIDED_BY_HUMAN.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import laplace_k_criterion as lk                                 # noqa: E402
from eval_utils import calc_Q_dual_strict_exp                    # noqa: E402

PANEL_VERSION = "k-criterion-panel-v1"
LAPLACE_GRAD_TOL = 1e-8          # Phase 9H defaults, unchanged
LAPLACE_MAX_ITER = 200


def _require_bernoulli_y(model) -> None:
    label = getattr(model, "family_y_label", model.family)
    if label != "bernoulli":
        raise ValueError(f"k_criterion_panel supports Bernoulli Y only, "
                         f"got {label!r}")


def log_prior(Z: np.ndarray, var_z: float) -> float:
    """ln p(Z) for Z ~ N(0, var_z I), all constants included."""

    n, k = Z.shape
    return float(-(n * k / 2.0) * math.log(2.0 * math.pi * var_z)
                 - float(np.sum(Z ** 2)) / (2.0 * var_z))


def data_loglik(model, X, Y, Z: np.ndarray, theta: dict[str, Any]) -> float:
    """ln p(X, Y | Z, theta) at ONE Z, strict constants included."""

    complete = float(calc_Q_dual_strict_exp(
        X, Y, Z[:, :, None], theta["F"], theta["sigma"], theta["var_z"],
        theta["w0"], theta["w"], model))
    return complete - log_prior(Z, theta["var_z"])


def theta_of(model) -> dict[str, Any]:
    p = model.params
    return {"F": np.asarray(p["F"], float), "sigma": np.asarray(p["sigma"], float),
            "w0": float(p["w0"]), "w": float(p["w"]), "var_z": float(p["var_z"])}


def cq_terms(model, X, Y, Z_samples: np.ndarray, *, num_params: int,
             n: int) -> dict[str, float]:
    """C_Q = D_K + P_Z + P_theta from the fit's own E-step samples."""

    _require_bernoulli_y(model)
    t = theta_of(model)
    q_strict = float(calc_Q_dual_strict_exp(
        X, Y, Z_samples, t["F"], t["sigma"], t["var_z"], t["w0"], t["w"],
        model))
    L = Z_samples.shape[2]
    q_z = sum(log_prior(Z_samples[:, :, l], t["var_z"]) for l in range(L)) / L
    q_y = sum(float(model.calc_log_likelihood_Y(
        Y, Z_samples[:, :, l:l + 1], t["w0"], t["w"])) for l in range(L)) / L
    q_x = q_strict - q_z - q_y
    p_theta = float(num_params) * math.log(n)
    d_k = -2.0 * (q_x + q_y)
    p_z = -2.0 * q_z
    return {"q_strict": q_strict, "Q_X": q_x, "Q_Y": q_y, "Q_Z": q_z,
            "D_K": d_k, "P_Z": p_z, "P_theta": p_theta,
            "C_Q": -2.0 * q_strict + p_theta,
            "C_Q_recomposed": d_k + p_z + p_theta}


def laplace_terms(model, X, Y, Z0: np.ndarray, *, num_params: int,
                  N: float) -> dict[str, Any]:
    """C_Lap and its fit / volume / penalty split at the joint mode."""

    _require_bernoulli_y(model)
    t0 = time.perf_counter()
    res = lk.laplace_log_observed(model, X, Y, Z0, grad_tol=LAPLACE_GRAD_TOL,
                                  max_iter=LAPLACE_MAX_ITER)
    out = lk.calc_C_Lap(res, d_K=int(num_params), N=float(N))
    seconds = time.perf_counter() - t0
    t = theta_of(model)
    Z_hat = res.Z_hat
    n, k = Z_hat.shape
    prior_hat = log_prior(Z_hat, t["var_z"])
    fit_mode = -2.0 * (res.phi_at_mode - prior_hat)
    volume = (float(np.sum(Z_hat ** 2)) / t["var_z"]
              + n * k * math.log(t["var_z"]) + res.logdet_H)
    penalty = out["parameter_term"]
    finite = bool(np.isfinite([res.phi_at_mode, res.logdet_H]).all())
    return {"lap_status": res.status, "C_Lap": out["C_Lap"],
            "lap_fit_mode": fit_mode, "lap_volume": volume,
            "lap_penalty": penalty,
            "C_Lap_recomposed": (fit_mode + volume + penalty
                                 if res.ok else float("nan")),
            "lap_log_observed": res.log_observed,
            "lap_phi_at_mode": res.phi_at_mode, "lap_logdet_H": res.logdet_H,
            "lap_logdet_sign": res.logdet_sign,
            "lap_min_eig_H": res.min_eigenvalue_H,
            "lap_grad_inf": res.grad_inf, "lap_iterations": res.iterations,
            "lap_zhat_sq": float(np.sum(Z_hat ** 2)),
            "lap_components_finite": finite,
            "lap_seconds": round(seconds, 3),
            "lap_notes": "; ".join(res.notes),
            "Z_hat": Z_hat}


def cond_terms(model, X, Y, *, Z_samples: np.ndarray, Z_last: np.ndarray,
               Z_mode: np.ndarray | None, mode_ok: bool, num_params: int,
               n: int) -> dict[str, float]:
    """Paper-inspired conditional diagnostics at three evaluation points."""

    _require_bernoulli_y(model)
    t = theta_of(model)
    p_theta = float(num_params) * math.log(n)
    L = Z_samples.shape[2]
    ll_mc = sum(data_loglik(model, X, Y, Z_samples[:, :, l], t)
                for l in range(L)) / L
    ll_last = data_loglik(model, X, Y, Z_last, t)
    if Z_mode is not None and mode_ok:
        ll_mode = data_loglik(model, X, Y, Z_mode, t)
        c_mode = -2.0 * ll_mode + p_theta
    else:
        ll_mode, c_mode = float("nan"), float("nan")
    return {"C_cond_MC": -2.0 * ll_mc + p_theta,
            "C_cond_last": -2.0 * ll_last + p_theta,
            "C_cond_mode": c_mode,
            "cond_ll_MC": ll_mc, "cond_ll_last": ll_last,
            "cond_ll_mode": ll_mode}
