"""
data_generator_overdispersed — NB2-Y 過分散カウント関係データの人工生成器。

data_generator_expfam.generate_dual_data と同じ潜在構造・X 生成規約
（Z 列正規化、F 行正規化、Bernoulli X は非正規化）を踏襲し、
Y のみ NB2（gamma-Poisson 混合）で生成する。

    z_i ~ N(0, I_k)（列 z-score 正規化）
    η_ij = w0 + w z_i^T z_j,  μ_ij = exp(η_ij)
    λ_ij ~ Gamma(shape=r, scale=μ_ij/r)      ← E[λ]=μ, Var[λ]=μ²/r
    y_ij ~ Poisson(λ_ij)                      ← 周辺は NB2(μ, r)

    E[y] = μ,  Var[y] = μ + μ²/r,  var/mean = 1 + μ/r

nb_r=None（または inf）で純 Poisson 生成（過分散なし）に退化する。
既存 data_generator_expfam.py は変更しない。
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, Optional

_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent.parent / "reproduction" / "src"))


def generate_dual_data_nb_y(
    n: int,
    d: int,
    k: int,
    seed: int = 1980,
    family_x: str = "bernoulli",
    var_f: float = 5.0,
    uniq: float = 0.1,
    sigma_x_true: float = 0.1,
    w0_true: float = 1.5,
    w_true: float = 0.3,
    nb_r: Optional[float] = 5.0,
) -> Dict[str, Any]:
    """
    NB2-Y の Dual-ExpFam 人工データを生成する。

    nb_r : float or None
        NB2 dispersion。None または inf で Poisson（過分散なし）。
    """
    from data_generator import normalize_zscore  # noqa (reproduction/src)

    valid = ("gaussian", "bernoulli", "poisson")
    if family_x not in valid:
        raise ValueError(f"family_x must be one of {valid}, got '{family_x}'")

    rng = np.random.default_rng(seed)
    var_z = 1.0

    # ── 共有潜在構造（generate_dual_data と同一規約） ────────────────
    Z = rng.normal(0.0, np.sqrt(var_z), size=(n, k))
    Z = normalize_zscore(Z, axis=0)

    sigma_x = np.diag(np.full(d, uniq))
    F = rng.normal(0.0, np.sqrt(var_f), size=(d, k))
    for j in range(d):
        nrm = np.linalg.norm(F[j, :])
        if nrm > 0:
            F[j, :] = (F[j, :] / nrm) * np.sqrt(1.0 - sigma_x[j, j])

    # ── X 生成 ───────────────────────────────────────────────────────
    eta_x_full = Z @ F.T
    if family_x == "gaussian":
        noise = rng.multivariate_normal(np.zeros(d), sigma_x, size=n)
        X = eta_x_full + noise
        X = normalize_zscore(X, axis=0)
    elif family_x == "bernoulli":
        prob_x = 1.0 / (1.0 + np.exp(-np.clip(eta_x_full, -500, 500)))
        X = rng.binomial(1, prob_x).astype(np.float64)
    else:  # poisson
        lam_x = np.exp(np.clip(eta_x_full, -20, 10))
        X = rng.poisson(lam_x).astype(np.float64)

    # ── Y 生成（NB2 = gamma-Poisson 混合、上三角→対称化） ────────────
    ZZT = Z @ Z.T
    eta_y = w0_true + w_true * ZZT
    mu_y = np.exp(np.clip(eta_y, -20, 10))
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    Y = np.zeros((n, n), dtype=np.float64)
    mu_upper = mu_y[upper_mask]
    if nb_r is None or not np.isfinite(nb_r):
        lam_upper = mu_upper                      # Poisson（過分散なし）
        y_family = "poisson"
    else:
        if nb_r <= 0:
            raise ValueError(f"nb_r must be > 0, got {nb_r}")
        lam_upper = rng.gamma(shape=nb_r, scale=mu_upper / nb_r)
        y_family = "nb"
    Y[upper_mask] = rng.poisson(lam_upper).astype(np.float64)
    Y = Y + Y.T
    np.fill_diagonal(Y, 0.0)

    upper_Y = Y[upper_mask]
    return {
        "X": X, "Y": Y, "Z": Z, "F": F,
        "sigma": sigma_x,
        "w0": w0_true, "w": w_true,
        "var_z": var_z, "var_f": var_f,
        "family_x": family_x,
        "family_y_true": y_family,
        "nb_r_true": (float(nb_r) if (nb_r is not None and np.isfinite(nb_r))
                      else float("inf")),
        "y_mean": float(upper_Y.mean()),
        "y_var": float(upper_Y.var()),
        "y_var_mean_ratio": float(upper_Y.var() / max(upper_Y.mean(), 1e-10)),
        "y_zero_ratio": float((upper_Y == 0).mean()),
        "y_max": float(upper_Y.max()),
        "theoretical_var_mean": (
            float(1.0 + upper_Y.mean() / nb_r)
            if (nb_r is not None and np.isfinite(nb_r)) else 1.0),
    }
