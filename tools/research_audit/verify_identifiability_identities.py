"""Independent numerical checks of the true-K identifiability derivations.

This script performs NO EM and touches NO historical artifact.  It exists so
that every closed-form identity claimed in
``reports/identifiability/true_k_identifiability_hardened_20260904.md`` is
checked against an independent Monte-Carlo or numerical evaluation, rather than
being asserted from algebra alone.

Every check prints the analytic value, the numerical value and the relative
error.  Tolerances are declared UP FRONT in ``TOLERANCES`` and are never
adjusted after seeing a result.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np

# Tolerances are fixed before running anything.  Monte-Carlo checks of high
# cumulants are noisy, so the tolerance grows with cumulant order; these are
# sanity checks of an algebraic derivation, not estimation accuracy claims.
TOLERANCES = {
    "mgf": 0.02,
    "cumulant_2": 0.02,
    "cumulant_4": 0.10,
    "cumulant_6": 0.35,
    "poisson_x_gram": 0.05,
    "bernoulli_x_mean": 0.01,
    "gaussian_y_recovery": 0.15,
    "triangle": 0.05,
}

SEED = 20260904


def _rel_err(numeric: float, analytic: float) -> float:
    denom = abs(analytic) if abs(analytic) > 1e-12 else 1.0
    return abs(numeric - analytic) / denom


def _cumulants_from_sample(sample: np.ndarray, order: int) -> float:
    """Plug-in cumulant of the given order from raw central moments.

    These are plug-in (biased) estimates built from raw central moments, not
    unbiased k-statistics; that is adequate for checking an algebraic identity
    at large sample sizes but should not be read as an estimator claim.  The
    sixth-order formula keeps the -10 m3^2 term: the population value is zero
    for the symmetric variables checked here, but the REALISED sample third
    moment is not, and a general helper must not bake in a population fact.
    """

    centred = sample - sample.mean()
    m2 = float(np.mean(centred ** 2))
    m4 = float(np.mean(centred ** 4))
    m6 = float(np.mean(centred ** 6))
    if order == 2:
        return m2
    if order == 4:
        return m4 - 3.0 * m2 ** 2
    if order == 6:
        m3 = float(np.mean(centred ** 3))
        return m6 - 15.0 * m4 * m2 - 10.0 * m3 ** 2 + 30.0 * m2 ** 3
    raise ValueError(order)


def check_s_mgf_and_cumulants(k_values=(1, 3, 5), n_draws=4_000_000) -> list[dict[str, Any]]:
    """S = z_i^T z_j with z_i, z_j iid N(0, I_K).

    Claim: M_S(t) = (1 - t^2)^(-K/2) for |t| < 1, hence
    kappa_2 = K, kappa_4 = 6K, kappa_6 = 120K.
    """

    out = []
    rng = np.random.default_rng(SEED)
    for k in k_values:
        zi = rng.standard_normal((n_draws, k))
        zj = rng.standard_normal((n_draws, k))
        s = np.einsum("ij,ij->i", zi, zj)

        for t in (0.3, 0.5):
            numeric = float(np.mean(np.exp(t * s)))
            analytic = (1.0 - t ** 2) ** (-k / 2.0)
            out.append({
                "check": "S_mgf", "K": k, "t": t,
                "analytic": analytic, "numeric": numeric,
                "rel_err": _rel_err(numeric, analytic),
                "tol": TOLERANCES["mgf"],
                "pass": _rel_err(numeric, analytic) <= TOLERANCES["mgf"],
            })

        for order, analytic in ((2, float(k)), (4, 6.0 * k), (6, 120.0 * k)):
            numeric = _cumulants_from_sample(s, order)
            tol = TOLERANCES[f"cumulant_{order}"]
            out.append({
                "check": f"S_kappa_{order}", "K": k,
                "analytic": analytic, "numeric": numeric,
                "rel_err": _rel_err(numeric, analytic), "tol": tol,
                "pass": _rel_err(numeric, analytic) <= tol,
            })
    return out


def check_gaussian_y_recovery(k=3, w=0.8, w0=0.4, sigma_y=0.5,
                              n_draws=8_000_000) -> list[dict[str, Any]]:
    """Y = w0 + w S + eps.  Claim: w^2 = kappa_6 / (20 kappa_4), K = kappa_4 / (6 w^4)."""

    rng = np.random.default_rng(SEED + 1)
    zi = rng.standard_normal((n_draws, k))
    zj = rng.standard_normal((n_draws, k))
    s = np.einsum("ij,ij->i", zi, zj)
    y = w0 + w * s + rng.normal(0.0, sigma_y, size=n_draws)

    kappa4 = _cumulants_from_sample(y, 4)
    kappa6 = _cumulants_from_sample(y, 6)
    kappa2 = _cumulants_from_sample(y, 2)

    w2_hat = kappa6 / (20.0 * kappa4)
    k_hat = kappa4 / (6.0 * w2_hat ** 2)
    sigma_y2_hat = kappa2 - k_hat * w2_hat

    tol = TOLERANCES["gaussian_y_recovery"]
    return [
        {"check": "gaussian_y_w2", "analytic": w ** 2, "numeric": float(w2_hat),
         "rel_err": _rel_err(w2_hat, w ** 2), "tol": tol,
         "pass": _rel_err(w2_hat, w ** 2) <= tol},
        {"check": "gaussian_y_K", "analytic": float(k), "numeric": float(k_hat),
         "rel_err": _rel_err(k_hat, k), "tol": tol,
         "pass": _rel_err(k_hat, k) <= tol},
        {"check": "gaussian_y_sigma_y2", "analytic": sigma_y ** 2,
         "numeric": float(sigma_y2_hat),
         "rel_err": _rel_err(sigma_y2_hat, sigma_y ** 2), "tol": tol,
         "pass": _rel_err(sigma_y2_hat, sigma_y ** 2) <= tol},
    ]


def check_poisson_x_gram(k=3, d=6, n_draws=4_000_000) -> list[dict[str, Any]]:
    """X_l | z ~ Poisson(exp(f_l^T z)).

    Claim: ||f_l||^2 = 2 log E[X_l] and, for l != m,
    f_l^T f_m = log( E[X_l X_m] / (E[X_l] E[X_m]) ).
    Hence FF^T -- and therefore K = rank(FF^T) -- is population identifiable.
    """

    rng = np.random.default_rng(SEED + 2)
    f = rng.normal(0.0, 0.45, size=(d, k))
    z = rng.standard_normal((n_draws, k))
    eta = z @ f.T
    x = rng.poisson(np.exp(eta))

    mean_x = x.mean(axis=0)
    gram_true = f @ f.T
    gram_hat = np.empty((d, d))
    for l in range(d):
        gram_hat[l, l] = 2.0 * math.log(mean_x[l])
        for m in range(l + 1, d):
            cross = float(np.mean(x[:, l].astype(np.float64) * x[:, m]))
            value = math.log(cross / (mean_x[l] * mean_x[m]))
            gram_hat[l, m] = gram_hat[m, l] = value

    max_abs = float(np.max(np.abs(gram_hat - gram_true)))
    scale = float(np.max(np.abs(gram_true)))
    rel = max_abs / scale
    tol = TOLERANCES["poisson_x_gram"]

    eig_true = np.linalg.eigvalsh(gram_true)[::-1]
    eig_hat = np.linalg.eigvalsh(gram_hat)[::-1]
    return [
        {"check": "poisson_x_gram_recovery", "K": k, "d": d,
         "analytic": 0.0, "numeric": rel, "rel_err": rel, "tol": tol,
         "pass": rel <= tol},
        # The TRUE Gram has rank K by construction, so checking it proves
        # nothing.  What matters is the ESTIMATED Gram: report its unthresholded
        # rank, its eigen-gap, and whether it even stayed inside the PSD cone.
        {"check": "poisson_x_rank_true_by_construction", "K": k, "d": d,
         "analytic": float(k),
         "numeric": float(int(np.linalg.matrix_rank(gram_true, tol=1e-9))),
         "rel_err": 0.0, "tol": 0.0,
         "note": "construction check only; carries no evidence about estimation",
         "pass": int(np.linalg.matrix_rank(gram_true, tol=1e-9)) == k},
        {"check": "poisson_x_estimated_gram_is_not_psd", "K": k, "d": d,
         "estimated_rank_unthresholded":
             int(np.linalg.matrix_rank(gram_hat)),
         "min_eigenvalue_estimated": float(np.min(eig_hat)),
         "eigen_gap_ratio_k_over_k_plus_1":
             (float(eig_hat[k - 1] / eig_hat[k]) if k < d and eig_hat[k] != 0.0
              else None),
         "note": ("the moment estimator is unconstrained, so the estimated Gram "
                  "can leave the PSD cone and its rank is not well defined "
                  "without a threshold (UNRESOLVED U7)"),
         "pass": True},
        {"check": "poisson_x_eigen_true", "eigenvalues": [float(v) for v in eig_true],
         "pass": True},
        {"check": "poisson_x_eigen_estimated", "eigenvalues": [float(v) for v in eig_hat],
         "pass": True},
    ]


def check_bernoulli_x_first_moment(k_values=(1, 3, 5), d=4,
                                   n_draws=2_000_000) -> list[dict[str, Any]]:
    """X_l | z ~ Bernoulli(sigmoid(f_l^T z)) with z symmetric => E[X_l] = 1/2."""

    out = []
    rng = np.random.default_rng(SEED + 3)
    for k in k_values:
        f = rng.normal(0.0, 1.0, size=(d, k))
        z = rng.standard_normal((n_draws, k))
        p = 1.0 / (1.0 + np.exp(-(z @ f.T)))
        x = rng.binomial(1, p)
        for l in range(d):
            numeric = float(x[:, l].mean())
            out.append({
                "check": "bernoulli_x_mean", "K": k, "column": l,
                "analytic": 0.5, "numeric": numeric,
                "rel_err": _rel_err(numeric, 0.5),
                "tol": TOLERANCES["bernoulli_x_mean"],
                "pass": _rel_err(numeric, 0.5) <= TOLERANCES["bernoulli_x_mean"],
            })
    return out


def check_poisson_y_moment_existence(k_values=(1, 3, 5)) -> list[dict[str, Any]]:
    """lambda = exp(w0 + w S).  E[lambda^r] = exp(r w0) (1 - r^2 w^2)^(-K/2).

    Finite iff |r w| < 1, so the canonical Poisson-Y has a finite mean iff
    |w| < 1 and a finite variance iff |w| < 1/2.  This is an analytic check of
    the divergence structure, not a Monte-Carlo one: at and beyond the boundary
    the sample moments do not converge, so a simulation cannot certify it.
    """

    out = []
    for k in k_values:
        for w in (0.2, 0.4, 0.49, 0.5, 0.6, 0.9, 1.0):
            mean_finite = abs(w) < 1.0
            var_finite = abs(2.0 * w) < 1.0
            e_lam = ((1.0 - w ** 2) ** (-k / 2.0)) if mean_finite else math.inf
            e_lam2 = ((1.0 - 4.0 * w ** 2) ** (-k / 2.0)) if var_finite else math.inf
            out.append({
                "check": "poisson_y_moment_existence", "K": k, "w": w,
                "mean_finite": mean_finite, "variance_finite": var_finite,
                "E_lambda": e_lam if math.isfinite(e_lam) else "inf",
                "E_lambda2": e_lam2 if math.isfinite(e_lam2) else "inf",
                "pass": True,
            })
    return out


def check_poisson_y_factorial_moment_identifiability(
        n_grid=4000) -> list[dict[str, Any]]:
    """Is (w0, w^2, K) recoverable from the first three factorial moments?

    a_r = E[lambda^r] = exp(r w0) (1 - r^2 w^2)^(-K/2), r = 1,2,3, all finite
    iff |w| < 1/3.  Eliminating w0 gives

        R(w^2) = (2 b_1 - b_2) / (3 b_1 - b_3),   b_r = log a_r,

    which depends on w^2 alone.  If R is strictly monotone on (0, 1/9) then w^2
    is determined by R, and then K and w0 follow.  This checks monotonicity
    numerically; it does not prove it.
    """

    v = np.linspace(1e-6, 1.0 / 9.0 - 1e-6, n_grid)      # v = w^2
    l1 = np.log1p(-1.0 * v)
    l4 = np.log1p(-4.0 * v)
    l9 = np.log1p(-9.0 * v)
    num = 2.0 * l1 - l4
    den = 3.0 * l1 - l9
    ratio = num / den
    diffs = np.diff(ratio)
    strictly_monotone = bool(np.all(diffs > 0) or np.all(diffs < 0))
    return [{
        "check": "poisson_y_factorial_moment_monotonicity",
        "domain": "w^2 in (0, 1/9)",
        "strictly_monotone_on_grid": strictly_monotone,
        "ratio_at_left": float(ratio[0]),
        "ratio_at_right": float(ratio[-1]),
        "note": "numerical evidence only; not a proof",
        "pass": strictly_monotone,
    }]


def check_triangle_identifies_sign_of_w(k_values=(1, 3, 5), w=0.8,
                                        n_draws=3_000_000) -> list[dict[str, Any]]:
    """The sign of w IS identified once three nodes are observed.

    Conditioning on z_j, z_k and using E[z_i z_i^T] = I,
        E[S_ij S_ik S_jk] = E[(z_j^T z_k)^2] = kappa_2(S) = K,
    so for Gaussian-Y the third joint central moment of a triangle is
        E[(Y_ij-w0)(Y_ik-w0)(Y_jk-w0)] = w^3 K,
    whose sign is the sign of w.  The single-dyad marginal only gives w^2,
    because S is symmetric; that symmetry does not survive to the triangle.
    """

    out = []
    rng = np.random.default_rng(SEED + 4)
    for k in k_values:
        zi = rng.standard_normal((n_draws, k))
        zj = rng.standard_normal((n_draws, k))
        zl = rng.standard_normal((n_draws, k))
        s_ij = np.einsum("ab,ab->a", zi, zj)
        s_il = np.einsum("ab,ab->a", zi, zl)
        s_jl = np.einsum("ab,ab->a", zj, zl)
        triple = s_ij * s_il * s_jl
        numeric = float(np.mean(triple))
        out.append({
            "check": "triangle_third_moment_of_S", "K": k,
            "analytic": float(k), "numeric": numeric,
            "rel_err": _rel_err(numeric, float(k)),
            "tol": TOLERANCES["triangle"],
            "pass": _rel_err(numeric, float(k)) <= TOLERANCES["triangle"],
        })
        for signed in (w, -w):
            observed = float(np.mean((signed ** 3) * triple))
            analytic = (signed ** 3) * k
            out.append({
                "check": "triangle_identifies_sign_of_w", "K": k, "w": signed,
                "analytic": analytic, "numeric": observed,
                "rel_err": _rel_err(observed, analytic),
                "tol": TOLERANCES["triangle"],
                "sign_matches": bool(np.sign(observed) == np.sign(signed)),
                "pass": bool(
                    _rel_err(observed, analytic) <= TOLERANCES["triangle"]
                    and np.sign(observed) == np.sign(signed)),
            })
    return out


def check_km1_nesting_gaussian_y() -> list[dict[str, Any]]:
    """Gaussian-Y: can a (K+1)-model reproduce a K-model's dyad marginal?

    kappa_4 = 6 K w^4 and kappa_6 = 120 K w^6 give w'^2 = kappa_6/(20 kappa_4)
    = w^2 and then K' = kappa_4/(6 w'^4) = K.  A (K+1)-parameterisation would
    need K' = K+1, a contradiction whenever w != 0.  Checked here arithmetically
    for a grid of (K, w).
    """

    out = []
    for k in (1, 2, 3, 5, 7):
        for w in (0.3, 0.8, 1.5):
            kappa4 = 6.0 * k * w ** 4
            kappa6 = 120.0 * k * w ** 6
            w2_hat = kappa6 / (20.0 * kappa4)
            k_hat = kappa4 / (6.0 * w2_hat ** 2)
            out.append({
                "check": "km1_nesting_gaussian_y", "K": k, "w": w,
                "recovered_w2": float(w2_hat), "recovered_K": float(k_hat),
                "recovered_K_equals_K": abs(k_hat - k) < 1e-9,
                "recovered_K_equals_K_plus_1": abs(k_hat - (k + 1)) < 1e-9,
                "pass": abs(k_hat - k) < 1e-9 and abs(k_hat - (k + 1)) >= 1e-9,
            })
    return out


CHECKS = {
    "s_mgf_cumulants": check_s_mgf_and_cumulants,
    "gaussian_y_recovery": check_gaussian_y_recovery,
    "poisson_x_gram": check_poisson_x_gram,
    "bernoulli_x_first_moment": check_bernoulli_x_first_moment,
    "poisson_y_moment_existence": check_poisson_y_moment_existence,
    "poisson_y_factorial_identifiability": check_poisson_y_factorial_moment_identifiability,
    "triangle_sign_of_w": check_triangle_identifies_sign_of_w,
    "km1_nesting_gaussian_y": check_km1_nesting_gaussian_y,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(CHECKS))
    parser.add_argument("--out", default=None)
    parser.add_argument("--fast", action="store_true",
                        help="smaller Monte-Carlo sizes for a quick smoke run")
    args = parser.parse_args(argv)

    selected = args.only or sorted(CHECKS)
    results: list[dict[str, Any]] = []
    for name in selected:
        fn = CHECKS[name]
        if args.fast and name in ("s_mgf_cumulants", "gaussian_y_recovery",
                                  "poisson_x_gram", "bernoulli_x_first_moment",
                                  "triangle_sign_of_w"):
            results.extend(fn(n_draws=200_000))          # type: ignore[call-arg]
        else:
            results.extend(fn())

    failures = [r for r in results if not r.get("pass", True)]
    payload = {
        "seed": SEED,
        "tolerances": TOLERANCES,
        "checks_run": selected,
        "result_count": len(results),
        "failure_count": len(failures),
        "failures": failures,
        "results": results,
        "verdict": "PASS" if not failures else "FAIL",
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
