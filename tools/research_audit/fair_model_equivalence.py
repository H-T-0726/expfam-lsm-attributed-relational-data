"""Phase A/B numerical equivalence checks on a small deterministic instance (no fit).

Writes, under the experiment results directory:
  equivalence_numeric_checks.csv  (every check with its value and tolerance)
Run by the tests as well (test_fair_common_evaluator.py).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "reproduction/src", ROOT / "expfam/src/experimental", ROOT / "tools/research_audit"):
    sys.path.insert(0, str(p))

import fair_common_evaluator as ev  # noqa: E402
import fair_protocol as fp  # noqa: E402


def instance(family_x="gaussian", seed=20261106, n=9, d=4, k=2):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, k))
    F = rng.standard_normal((d, k)) * (0.8 if family_x == "gaussian" else 0.4)
    s = rng.uniform(0.2, 0.6, d)
    if family_x == "gaussian":
        X = Z @ F.T + rng.standard_normal((n, d)) * np.sqrt(s)
    else:
        X = rng.poisson(np.exp(Z @ F.T)).astype(float)
    w0, w = -0.3, 1.3
    P = 1 / (1 + np.exp(-(w0 + w * Z @ Z.T)))
    Y = np.triu((rng.random((n, n)) < P).astype(float), 1)
    Y = Y + Y.T
    return X, Y, Z, {"F": F, "sigma_diag": s, "w0": w0, "w": w, "var_z": 1.0}


def fd_grad(fun, z, h=1e-6):
    return np.array([(fun(z + h * e) - fun(z - h * e)) / (2 * h) for e in np.eye(len(z))])


def fd_hess(fun, z, h=1e-4):
    k = len(z)
    E = np.eye(k)
    return np.array([[(fun(z + h * E[a] + h * E[b]) - fun(z + h * E[a] - h * E[b]) - fun(z - h * E[a] + h * E[b])
                       + fun(z - h * E[a] - h * E[b])) / (4 * h * h) for b in range(k)] for a in range(k)])


def checks():
    rows = []

    def add(name, value, tol, layer, note=""):
        rows.append({"check": name, "value": float(value), "tolerance": tol, "pass": bool(abs(value) <= tol),
                     "layer": layer, "note": note})

    X, Y, Z, th = instance("gaussian")
    n, d = X.shape
    # model / objective equivalence: paper form vs ExpFam form
    add("Gaussian-X: paper N(Fz,Sigma) log-density minus ExpFam form", ev.loglik_x_paper(X, Z, th["F"], th["sigma_diag"])
        - ev.loglik_x_expfam_gauss(X, Z, th["F"], th["sigma_diag"]), 1e-9, "model")
    add("Gaussian-X: ExpFam form minus evaluator", ev.loglik_x_expfam_gauss(X, Z, th["F"], th["sigma_diag"])
        - ev.loglik_x(X, Z, th["F"], "gaussian", th["sigma_diag"]), 1e-9, "model")
    add("Bernoulli-Y: paper Eq.(16) 1/2 * ordered pairs minus unordered pairs",
        ev.loglik_y_paper_eq16(Y, Z, th["w0"], th["w"]) - ev.loglik_y(Y, Z, th["w0"], th["w"]), 1e-9, "model")
    # rotation invariance
    R, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((2, 2)))
    th_r = dict(th, F=th["F"] @ R)
    add("rotation invariance (Z R, F R) of log p(Z,X,Y)",
        (ev.log_prior(Z @ R) + ev.data_loglik(X, Y, Z @ R, th_r, "gaussian"))
        - (ev.log_prior(Z) + ev.data_loglik(X, Y, Z, th, "gaussian")), 1e-9, "model", "DERIVED: R orthogonal")
    # implementation: lineage-E strict Q and the historical Python plug-in
    import em_runner
    from eval_utils import calc_Q_dual_strict_exp
    m = em_runner.build_model(n, d, 2, 1, "gaussian", "bernoulli", train_mask=None, numerics_mode="consistent")
    m.initialize_params(true_params=None, seed=0)
    ref = ev.log_prior(Z) + ev.data_loglik(X, Y, Z, th, "gaussian")
    q_e0 = calc_Q_dual_strict_exp(X, Y, Z[:, :, None], th["F"], np.diag(th["sigma_diag"]), 1.0, th["w0"], th["w"], m)
    add("lineage-E calc_Q_dual_strict_exp with model.params['sigma'] NOT set to the argument (hazard)",
        q_e0 - ref, float("inf"), "implementation",
        "FINDING: the Gaussian-X term reads model.params['sigma'] and ignores the sigma argument; harmless in the "
        "runners (params hold the fitted sigma) but a latent hazard")
    m.params["sigma"] = np.diag(th["sigma_diag"])
    q_e = calc_Q_dual_strict_exp(X, Y, Z[:, :, None], th["F"], np.diag(th["sigma_diag"]), 1.0, th["w0"], th["w"], m)
    add("lineage-E calc_Q_dual_strict_exp minus evaluator complete log density (params set)", q_e - ref, 1e-8,
        "implementation")
    from experiment_paper_2 import Experiment2Runner
    q_py = Experiment2Runner.calc_log_likelihood(None, X, Y, Z, th["F"], np.diag(th["sigma_diag"]), 1.0, th["w0"], th["w"])
    add("Python reproduction calc_log_likelihood minus evaluator complete log density",
        q_py - (ev.log_prior(Z) + ev.data_loglik(X, Y, Z, th, "gaussian")), 1e-6, "implementation",
        "reproduction clips S to [1e-10, 1-1e-10]")
    # Poisson-X strict Q
    Xp, Yp, Zp, thp = instance("poisson")
    mp = em_runner.build_model(n, d, 2, 1, "poisson", "bernoulli", train_mask=None, numerics_mode="consistent")
    mp.initialize_params(true_params=None, seed=0)
    q_ep = calc_Q_dual_strict_exp(Xp, Yp, Zp[:, :, None], thp["F"], np.eye(d), 1.0, thp["w0"], thp["w"], mp)
    add("Poisson-X: lineage-E strict Q minus evaluator", q_ep - (ev.log_prior(Zp) + ev.data_loglik(Xp, Yp, Zp, thp, "poisson")),
        1e-8, "implementation")
    # derivatives: evaluator analytic vs finite differences, then each implementation vs FD
    i = 3
    for fam, (XX, YY, ZZ, tt) in (("gaussian", (X, Y, Z, th)), ("poisson", (Xp, Yp, Zp, thp))):
        def f(zi, XX=XX, YY=YY, ZZ=ZZ, tt=tt, fam=fam):
            Zc = ZZ.copy()
            Zc[i] = zi
            return ev.log_prior(Zc) + ev.data_loglik(XX, YY, Zc, tt, fam)
        g_fd = fd_grad(f, ZZ[i].copy())
        H_fd = -fd_hess(f, ZZ[i].copy())
        g = ev.grad_zi(XX, YY, ZZ, tt, fam, i)
        H = ev.neg_hess_zi(XX, YY, ZZ, tt, fam, i)
        add(f"{fam}: evaluator gradient vs FD (rel)", np.linalg.norm(g - g_fd) / np.linalg.norm(g_fd), 1e-6, "derivation")
        add(f"{fam}: evaluator -Hessian vs FD (rel)", np.linalg.norm(H - H_fd) / np.linalg.norm(H_fd), 1e-5, "derivation")
        sig = np.diag(tt["sigma_diag"]) if fam == "gaussian" else np.eye(d)
        me = em_runner.build_model(n, d, 2, 1, fam, "bernoulli", train_mask=None, numerics_mode="consistent")
        me.initialize_params(true_params=None, seed=0)
        ge = -me._calc_gradient(XX, YY, ZZ, tt["F"], sig, 1.0, tt["w0"], tt["w"], i)
        Ae = me._calc_precision_matrix(ZZ, tt["F"], sig, 1.0, tt["w0"], tt["w"], i)
        add(f"{fam}: lineage-E E-step gradient vs FD (rel)", np.linalg.norm(ge - g_fd) / np.linalg.norm(g_fd), 1e-6, "implementation")
        add(f"{fam}: lineage-E precision vs FD (rel)", np.linalg.norm(Ae - H_fd) / np.linalg.norm(H_fd), 1e-5, "implementation")
        if fam == "gaussian":
            from model import LatentStructuralModel
            mo = LatentStructuralModel(n=n, d=d, k=2, L=1)
            mo.initialize_params(true_params=None, seed=0)
            go = -mo._calc_gradient(XX, YY, ZZ, tt["F"], sig, 1.0, tt["w0"], tt["w"], i)
            Ao = mo._calc_precision_matrix(ZZ, tt["F"], sig, 1.0, tt["w0"], tt["w"], i)
            add("gaussian: Python reproduction E-step gradient vs FD (rel)", np.linalg.norm(go - g_fd) / np.linalg.norm(g_fd),
                1e-6, "implementation", "expected to FAIL: 1/2 on the Y score (paper Eq.23)")
            add("gaussian: Python reproduction precision vs FD (rel)", np.linalg.norm(Ao - 1e-6 * np.eye(2) - H_fd) / np.linalg.norm(H_fd),
                1e-5, "implementation", "expected to FAIL: 1/2 on the Y curvature (paper Eq.22); ridge 1e-6 removed before comparing")
            yg = tt["w"] * (ZZ.T @ np.where(np.arange(n) == i, 0.0, YY[i] - 1 / (1 + np.exp(-(tt['w0'] + tt['w'] * ZZ @ ZZ[i])))))
            add("gaussian: Python reproduction Y-score / true Y-score (ratio - 0.5)",
                float(np.dot(go - (g_fd - yg), yg) / np.dot(yg, yg)) - 0.5, 1e-6, "implementation", "ratio 0.5 = the extra 1/2")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = ROOT / fp.RESULTS_REL / "equivalence_numeric_checks.csv"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    df = checks()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n")
    print(df.to_string())
