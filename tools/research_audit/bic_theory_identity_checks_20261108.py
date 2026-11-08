"""Deterministic identity checks for the K-selection theory assessment (2026-11-08).

THEORY / AUDIT only.  No MCEM fit, no K-selection, no success-rate evaluation.
Every check is a fixed, small, deterministic computation of a mathematical
identity derived in
``reports/k_selection_theory/bic_theory_candidate_assessment_20261108/``.

Checks
------
T1  ELBO identity  log p(X,Y) = E_q log p(X,Y,Z) + H(q) + KL(q || post)
    for a Gaussian q, by exact grid quadrature in a tiny NON-Gaussian model
    (n=3 nodes, K=1, Poisson-X d=2, Bernoulli-Y, 3 pairs).
T2  Exact-posterior decompositions used for C_Q / C_Q - P_Z:
      E_post log p(X,Y|Z) = l_marg + KL(post || prior)
      E_post log p(Z)     = -KL(post || prior) - H(post)
T3  Scale-direction identity (any theta, sigma_z^2 = 1):
      d/dc l_marg(cF, w0, c^2 w) |_{c=1} = E_post ||Z||_F^2 - nK
T4  Laplace-q ELBO: ELBO(q_Lap) = l_Lap - R3,  R3 = Phi(Z_hat) - nK/2 - E_qLap Phi
T5  Linear-Gaussian factor model (Gaussian-X): Laplace is exact and R3 = 0;
    at the exact MLE (EM run to convergence) E_post ||Z||^2 = nK.
T6  Parameter count: rank of the Jacobian of (F, Psi) -> F F^T + Psi
    equals min(dK + d - K(K-1)/2, d(d+1)/2) at generic F (d=15, K=1..12),
    and drops when F has a zero column (over-specified K).
T7  The repository's ``scale_Z`` makes -2 Q_Z = nK(1 + ln 2 pi) exactly.

Run from the repository root:
    python tools/research_audit/bic_theory_identity_checks_20261108.py
Outputs (new directory, never overwrites other artifacts):
    reports/k_selection_theory/bic_theory_candidate_assessment_20261108/
        identity_checks.csv, identity_checks_runinfo.json
"""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "k_selection_theory" / \
    "bic_theory_candidate_assessment_20261108"
sys.path.insert(0, str(ROOT / "reproduction" / "src"))

LN2PI = math.log(2.0 * math.pi)
rows: list[dict] = []


def record(check_id, description, lhs, rhs, tol):
    diff = abs(float(lhs) - float(rhs))
    rows.append({"check_id": check_id, "description": description,
                 "lhs": f"{float(lhs):.12g}", "rhs": f"{float(rhs):.12g}",
                 "abs_diff": f"{diff:.3e}", "tol": f"{tol:.1e}",
                 "pass": bool(diff <= tol)})


# ---------------------------------------------------------------------------
# Tiny non-Gaussian model: n=3, K=1, Poisson-X (d=2), Bernoulli-Y (3 pairs)
# ---------------------------------------------------------------------------
X_T = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
Y_T = {(0, 1): 1.0, (0, 2): 0.0, (1, 2): 1.0}
F_T = np.array([0.6, -0.4])
W0_T, W_T = -0.3, 0.9
N_T, K_T = 3, 1
GRID = np.linspace(-7.0, 7.0, 161)
H_GRID = GRID[1] - GRID[0]


def phi_tiny(z, F, w0, w):
    """log p(X, Y, Z | theta) for z of shape (..., 3)."""
    out = np.sum(-0.5 * z ** 2 - 0.5 * LN2PI, axis=-1)
    for i in range(N_T):
        for l in range(2):
            eta = F[l] * z[..., i]
            out = out + X_T[i, l] * eta - np.exp(eta) - gammaln(X_T[i, l] + 1)
    for (i, j), y in Y_T.items():
        eta = w0 + w * z[..., i] * z[..., j]
        out = out + y * eta - np.logaddexp(0.0, eta)
    return out


def grid_points():
    a, b, c = np.meshgrid(GRID, GRID, GRID, indexing="ij")
    return np.stack([a, b, c], axis=-1)


def log_marginal(phi_vals):
    m = phi_vals.max()
    return float(m + math.log(np.sum(np.exp(phi_vals - m)) * H_GRID ** 3))


def tiny_checks():
    Zg = grid_points()
    phi = phi_tiny(Zg, F_T, W0_T, W_T)
    l_marg = log_marginal(phi)
    log_post = phi - l_marg
    post = np.exp(log_post)
    dv = H_GRID ** 3
    sq = np.sum(Zg ** 2, axis=-1)
    log_prior = np.sum(-0.5 * Zg ** 2 - 0.5 * LN2PI, axis=-1)
    loglik = phi - log_prior
    record("T0", "posterior integrates to 1 on the grid",
           np.sum(post) * dv, 1.0, 1e-10)

    # T2 exact-posterior decompositions
    kl_post_prior = float(np.sum(post * (log_post - log_prior)) * dv)
    h_post = float(-np.sum(post * log_post) * dv)
    e_loglik = float(np.sum(post * loglik) * dv)
    e_logprior = float(np.sum(post * log_prior) * dv)
    record("T2a", "E_post log p(X,Y|Z) = l_marg + KL(post||prior)",
           e_loglik, l_marg + kl_post_prior, 1e-9)
    record("T2b", "E_post log p(Z) = -KL(post||prior) - H(post)",
           e_logprior, -kl_post_prior - h_post, 1e-9)
    record("T2c", "-2 E_post log p(X,Y,Z) = -2 l_marg + 2 H(post)  (C_Q core)",
           -2 * (e_loglik + e_logprior), -2 * l_marg + 2 * h_post, 1e-8)
    record("T2d", "-2 E_post log p(X,Y|Z) = -2 l_marg - 2 KL(post||prior)  "
           "(C_Q - P_Z core)", -2 * e_loglik,
           -2 * l_marg - 2 * kl_post_prior, 1e-8)

    # T3 scale-direction identity at a generic (non-MLE) theta
    c_h = 1e-4
    lp = log_marginal(phi_tiny(Zg, (1 + c_h) * F_T, W0_T, (1 + c_h) ** 2 * W_T))
    lm = log_marginal(phi_tiny(Zg, (1 - c_h) * F_T, W0_T, (1 - c_h) ** 2 * W_T))
    record("T3", "d/dc l_marg(cF, w0, c^2 w)|c=1 = E_post||Z||^2 - nK "
           "(generic theta)", (lp - lm) / (2 * c_h),
           float(np.sum(post * sq) * dv) - N_T * K_T, 1e-6)

    # Laplace at the joint mode (Newton on Phi, analytic derivatives)
    z = np.zeros(3)
    for _ in range(100):
        g, Hm = grad_hess_tiny(z)
        step = np.linalg.solve(Hm, g)
        z = z + step
        if np.max(np.abs(g)) < 1e-13:
            break
    g, Hm = grad_hess_tiny(z)
    phi_hat = float(phi_tiny(z[None, :], F_T, W0_T, W_T)[0])
    sign, logdet = np.linalg.slogdet(Hm)
    l_lap = phi_hat + 0.5 * N_T * K_T * LN2PI - 0.5 * logdet
    record("T4pre", "joint mode is stationary (max|grad Phi|)",
           float(np.max(np.abs(g))), 0.0, 1e-10)

    # q_Lap = N(z_hat, H^{-1})
    Hinv = np.linalg.inv(Hm)
    diff = Zg - z
    quad = np.einsum("...a,ab,...b->...", diff, Hm, diff)
    log_q = -0.5 * quad - 1.5 * LN2PI + 0.5 * logdet
    q = np.exp(log_q)
    e_q_phi = float(np.sum(q * phi) * dv)
    h_q = 0.5 * N_T * K_T * (1 + LN2PI) - 0.5 * logdet
    elbo = e_q_phi + h_q
    kl_q_post = float(np.sum(q * (log_q - log_post)) * dv)
    record("T1", "l_marg = E_q log p(X,Y,Z) + H(q) + KL(q||post), q = N(Zhat,H^-1)",
           l_marg, elbo + kl_q_post, 1e-8)
    record("T1b", "KL(q_Lap || post) >= 0 (value reported; lhs=KL, rhs=0)",
           kl_q_post, 0.0, max(kl_q_post, 0.0) + 1e-12)
    # Gaussian q away from the mode (arbitrary normalized q)
    m2 = z + np.array([0.3, -0.2, 0.1])
    S2 = np.diag([0.5, 0.8, 0.3])
    S2inv = np.linalg.inv(S2)
    d2 = Zg - m2
    log_q2 = (-0.5 * np.einsum("...a,ab,...b->...", d2, S2inv, d2)
              - 1.5 * LN2PI - 0.5 * math.log(np.linalg.det(S2)))
    q2 = np.exp(log_q2)
    elbo2 = float(np.sum(q2 * phi) * dv) + 0.5 * 3 * (1 + LN2PI) \
        + 0.5 * math.log(np.linalg.det(S2))
    kl2 = float(np.sum(q2 * (log_q2 - log_post)) * dv)
    record("T1c", "same identity for an arbitrary Gaussian q (not the mode)",
           l_marg, elbo2 + kl2, 1e-8)
    r3 = phi_hat - 0.5 * N_T * K_T - e_q_phi
    record("T4", "ELBO(q_Lap) = l_Lap - R3", elbo, l_lap - r3, 1e-9)
    record("T4b", "Laplace error l_marg - l_Lap (value; rhs=0, informative)",
           l_marg - l_lap, 0.0, abs(l_marg - l_lap) + 1e-12)
    record("T4c", "R3 (value; rhs=0, informative)", r3, 0.0, abs(r3) + 1e-12)
    return {"l_marg": l_marg, "l_lap": l_lap, "elbo_lap": elbo,
            "kl_q_post": kl_q_post, "R3": r3}


def grad_hess_tiny(z):
    g = -z.copy()
    H = np.eye(3)
    for i in range(N_T):
        for l in range(2):
            eta = F_T[l] * z[i]
            g[i] += F_T[l] * (X_T[i, l] - math.exp(eta))
            H[i, i] += F_T[l] ** 2 * math.exp(eta)
    for (i, j), y in Y_T.items():
        eta = W0_T + W_T * z[i] * z[j]
        s = 1.0 / (1.0 + math.exp(-eta))
        r = y - s
        v = s * (1 - s)
        g[i] += W_T * r * z[j]
        g[j] += W_T * r * z[i]
        H[i, i] += W_T ** 2 * v * z[j] ** 2
        H[j, j] += W_T ** 2 * v * z[i] ** 2
        off = W_T ** 2 * v * z[i] * z[j] - W_T * r
        H[i, j] += off
        H[j, i] += off
    return g, H


# ---------------------------------------------------------------------------
# T5 linear-Gaussian factor model
# ---------------------------------------------------------------------------
def fa_checks():
    rng = np.random.default_rng(20261108)
    n, d, K = 40, 5, 2
    F0 = rng.normal(size=(d, K))
    Psi0 = np.full(d, 0.3)
    Z0 = rng.normal(size=(n, K))
    X = Z0 @ F0.T + rng.normal(size=(n, d)) * np.sqrt(Psi0)

    def post_moments(F, Psi):
        A = np.eye(K) + F.T @ (F / Psi[:, None])
        Ainv = np.linalg.inv(A)
        M = X @ (F / Psi[:, None]) @ Ainv
        return A, Ainv, M

    def l_marg(F, Psi):
        C = F @ F.T + np.diag(Psi)
        sign, ld = np.linalg.slogdet(C)
        Ci = np.linalg.inv(C)
        return float(-0.5 * n * (d * LN2PI + ld)
                     - 0.5 * np.einsum("ia,ab,ib->", X, Ci, X))

    def phi(Z, F, Psi):
        R = X - Z @ F.T
        return float(-0.5 * np.sum(Z ** 2) - 0.5 * n * K * LN2PI
                     - 0.5 * n * np.sum(np.log(2 * np.pi * Psi))
                     - 0.5 * np.sum(R ** 2 / Psi))

    # exact EM (Rubin-Thayer) to convergence
    F, Psi = F0 * 0.5 + 0.1, np.ones(d)
    for _ in range(200000):
        A, Ainv, M = post_moments(F, Psi)
        Ezz = n * Ainv + M.T @ M
        F_new = (X.T @ M) @ np.linalg.inv(Ezz)
        Psi_new = np.mean(X ** 2, axis=0) - np.sum(F_new * (X.T @ M), axis=1) / n
        if max(np.max(np.abs(F_new - F)), np.max(np.abs(Psi_new - Psi))) < 1e-14:
            F, Psi = F_new, Psi_new
            break
        F, Psi = F_new, Psi_new
    A, Ainv, M = post_moments(F, Psi)
    e_sq = float(np.sum(M ** 2) + n * np.trace(Ainv))
    record("T5a", "FA exact MLE: E_post||Z||^2 = nK", e_sq, n * K, 1e-7)
    A2, Ainv2, M2 = post_moments(1.3 * F, Psi)
    e_sq2 = float(np.sum(M2 ** 2) + n * np.trace(Ainv2))
    record("T5b", "FA non-stationary theta (1.3 F): E_post||Z||^2 - nK "
           "(value; rhs=0, informative, expected != 0)", e_sq2 - n * K, 0.0,
           abs(e_sq2 - n * K) + 1e-12)
    # Laplace exact for quadratic Phi: joint Hessian is block-diagonal A
    Zhat = M
    logdetH = n * np.linalg.slogdet(A)[1]
    l_lap = phi(Zhat, F, Psi) + 0.5 * n * K * LN2PI - 0.5 * logdetH
    record("T5c", "FA: Laplace l_Lap = exact l_marg", l_lap, l_marg(F, Psi), 1e-8)
    # E_q Phi computed from first and second moments of q = N(M_i, A^{-1})
    # (independent of the quadratic-Taylor argument that gives R3 = 0)
    Rm = X - M @ F.T
    e_q_phi = float(-0.5 * (np.sum(M ** 2) + n * np.trace(Ainv))
                    - 0.5 * n * K * LN2PI
                    - 0.5 * n * np.sum(np.log(2 * np.pi * Psi))
                    - 0.5 * np.sum(Rm ** 2 / Psi)
                    - 0.5 * n * np.trace((F.T @ (F / Psi[:, None])) @ Ainv))
    r3 = phi(Zhat, F, Psi) - 0.5 * n * K - e_q_phi
    record("T5d", "FA: R3 = 0 for quadratic Phi", r3, 0.0, 1e-9)
    # C_Q at exact posterior: -2Q_Z = nK(1+ln2pi) at the MLE
    e_logprior = -0.5 * n * K * LN2PI - 0.5 * e_sq
    record("T5e", "FA exact MLE: -2 E_post log p(Z) = nK(1+ln 2pi)",
           -2 * e_logprior, n * K * (1 + LN2PI), 1e-7)


# ---------------------------------------------------------------------------
# T6 parameter count by Jacobian rank
# ---------------------------------------------------------------------------
def jac_rank(F, with_psi=True):
    d, K = F.shape
    iu = np.triu_indices(d)
    cols = []
    for a in range(d):
        for b in range(K):
            dF = np.zeros_like(F)
            dF[a, b] = 1.0
            dC = dF @ F.T + F @ dF.T
            cols.append(dC[iu])
    if with_psi:
        for a in range(d):
            dC = np.zeros((d, d))
            dC[a, a] = 1.0
            cols.append(dC[iu])
    J = np.array(cols).T
    s = np.linalg.svd(J, compute_uv=False)
    return int(np.sum(s > s[0] * 1e-10))


def param_count_checks():
    rng = np.random.default_rng(15)
    d = 15
    for K in range(1, 13):
        F = rng.normal(size=(d, K))
        formula = min(d * K + d - K * (K - 1) // 2, d * (d + 1) // 2)
        record(f"T6_K{K}", f"rank J(F,Psi) generic, d=15, K={K} vs "
               "min(dK+d-K(K-1)/2, d(d+1)/2)", jac_rank(F), formula, 0)
        formula_f = d * K - K * (K - 1) // 2
        record(f"T6F_K{K}", f"rank J(F -> FF^T) generic, d=15, K={K} vs "
               "dK-K(K-1)/2", jac_rank(F, with_psi=False), formula_f, 0)
    for K in (3, 5):
        F = rng.normal(size=(d, K))
        F[:, -1] = 0.0
        record(f"T6sing_K{K}", f"rank J(F,Psi) with zero last column, K={K} "
               "(value vs generic count; informative, expected lower)",
               jac_rank(F), d * K + d - K * (K - 1) // 2,
               abs(jac_rank(F) - (d * K + d - K * (K - 1) // 2)))


# ---------------------------------------------------------------------------
# T7 repository scale_Z
# ---------------------------------------------------------------------------
def scale_z_check():
    from model import LatentStructuralModel  # reproduction/src/model.py
    rng = np.random.default_rng(7)
    n, k, L = 75, 5, 5
    Zs = rng.normal(scale=0.7, size=(n, k, L))
    m = LatentStructuralModel(n=n, d=15, k=k, L=L)
    Zs = m.scale_Z(Zs)
    qz = np.mean([-0.5 * n * k * LN2PI - 0.5 * np.sum(Zs[:, :, l] ** 2)
                  for l in range(L)])
    record("T7a", "after scale_Z: -2 Q_Z (L-sample mean) = nK(1+ln 2pi)",
           -2 * qz, n * k * (1 + LN2PI), 1e-8)
    last = Zs[:, :, -1]
    record("T7b", "after scale_Z: ||Z_last||^2 - nK (value; informative, "
           "identity is for the L-sample mean only)",
           float(np.sum(last ** 2)) - n * k, 0.0,
           abs(float(np.sum(last ** 2)) - n * k) + 1e-12)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tiny = tiny_checks()
    fa_checks()
    param_count_checks()
    scale_z_check()
    with open(OUT / "identity_checks.csv", "w", newline="",
              encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", "tools", "reproduction"],
            cwd=ROOT, text=True).strip())
    except Exception:
        head, dirty = "unknown", None
    info = {"script": "tools/research_audit/bic_theory_identity_checks_20261108.py",
            "git_head": head, "tools_or_reproduction_dirty": dirty,
            "python": sys.version, "numpy": np.__version__,
            "platform": platform.platform(),
            "n_checks": len(rows), "n_pass": sum(r["pass"] for r in rows),
            "tiny_model_values": tiny,
            "note": "THEORY/AUDIT identity checks; no MCEM fit, no K selection"}
    (OUT / "identity_checks_runinfo.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in rows:
        print(r["check_id"], r["pass"], r["lhs"], r["rhs"], r["abs_diff"])
    print("pass", info["n_pass"], "/", info["n_checks"])


if __name__ == "__main__":
    main()
