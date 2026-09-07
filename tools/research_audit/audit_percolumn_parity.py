"""
Per-column implementation parity audit (2026-09-07).

QUESTION
--------
Before running the mixed-family K-selection experiment we must know whether the
per-column implementation computes the SAME objective as the single-family
model that the canonical Gaussian-Y core experiment used, when every column is
set to the same family.

Compared, on identical data / K / seed / initialisation:

  A. DualExpFamLSMFixed(family_x="bernoulli", family_y="gaussian")
     + utils_expfam.calc_Q_dual_strict / calc_bic_dual      <- core experiment path

  B. DualExpFamLSMPerColumnConsistent(family_x_list=["bernoulli"]*15,
                                      family_y="gaussian")
     + eval_utils.calc_Q_dual_strict_exp / calc_bic_exp      <- per-column path

Nothing is refit or tuned here. The audit separates
  (i)  quantities that MUST agree if both implement the same model, from
  (ii) quantities that are ALLOWED to differ by a known, K-independent constant.

Run:
    python tools/research_audit/audit_percolumn_parity.py
Exit 0 = no blocker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "expfam" / "src"))
sys.path.insert(0, str(_ROOT / "expfam" / "src" / "experimental"))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from data_generator_canonical_gaussian_y_k import generate_canonical_gaussian_y  # noqa: E402
from model_dual_expfam_fixed import DualExpFamLSMFixed  # noqa: E402
from model_dual_expfam_consistent import DualExpFamLSMPerColumnConsistent  # noqa: E402
from model_dual_expfam_percolumn import DualExpFamLSMPerColumn  # noqa: E402
from utils_expfam import _lnpZ, calc_Q_dual_strict, calc_bic_dual  # noqa: E402
import eval_utils  # noqa: E402

N, D, K_TRUE, L, NUM_ITER = 150, 15, 3, 5, 8
SEED_DATA, SEED_MODEL = 700001, 700051
CAND_K = [1, 2, 3, 4, 5, 6, 7, 8, 9]
ROWS: list[dict] = []


def note(check, status, detail, category):
    ROWS.append({"check": check, "status": status, "category": category,
                 "detail": detail})
    print(f"  [{status:7s}] {check}: {detail}")


def build(kind, k, seed):
    """Construct a model with identical initialisation for either path."""
    if kind == "single":
        m = DualExpFamLSMFixed(n=N, d=D, k=k, L=L,
                               family_x="bernoulli", family_y="gaussian")
    elif kind == "percolumn_consistent":
        m = DualExpFamLSMPerColumnConsistent(
            n=N, d=D, k=k, L=L,
            family_x_list=["bernoulli"] * D, family_y="gaussian")
    else:
        m = DualExpFamLSMPerColumn(
            n=N, d=D, k=k, L=L,
            family_x_list=["bernoulli"] * D, family_y="gaussian")
    m.initialize_params(true_params={"var_f": 5.0}, seed=seed)
    return m


def run_em(kind, X, Y, k, seed):
    """EM loop copied verbatim from the canonical core runner."""
    rng = np.random.default_rng(seed)
    m = build(kind, k, seed)
    um = np.triu(np.ones((N, N), dtype=bool), k=1)
    uv = Y[um]
    m.params["w0"] = float(uv.mean())
    m.params["w"] = 0.5
    m.sigma_y = float(max(uv.std(), 0.01))
    m.params["F"] *= 0.2

    Z = m.params["Z"].copy(); F = m.params["F"].copy()
    sigma = m.params["sigma"].copy()
    w0 = m.params["w0"]; w = m.params["w"]; var_z = m.params["var_z"]
    Z_prev = Z.copy()

    for _ in range(NUM_ITER):
        Zs = np.zeros((N, k, L))
        for l in range(L):
            m.params.update(dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w))
            Zn = m.calc_eta_newton(X, Y, rng=rng, max_iter=10, alpha=0.5)
            Zs[:, :, l] = Zn
            Z = Zn.copy()
        if np.any(~np.isfinite(Zs)):
            Zs = np.stack([Z_prev] * L, axis=2); Z = Z_prev.copy()
        Zs = m.scale_Z(Zs)
        Z_prev = Z.copy(); Z = Zs[:, :, -1].copy()
        F = m.calc_F(X, Zs)
        sigma = m.calc_sigma(X, Zs, F)
        w0 = m.calc_w0(Y, Zs, w0, w, max_iter=50)
        w = m.calc_w(Y, Zs, w0, w, max_iter=50)
        m.calc_sigma_y(Y, Zs, w0, w)

    m.params.update({"F": F, "sigma": sigma, "w0": w0, "w": w})
    return m, Zs, F, sigma, w0, w, var_z


def main() -> int:
    print("per-column parity audit (all-Bernoulli X, Gaussian Y)")
    print("=" * 74)
    data = generate_canonical_gaussian_y(n=N, d=D, k=K_TRUE, seed=SEED_DATA,
                                         w0_true=0.0, w_true=0.5,
                                         sigma_y_true=0.1)
    X, Y = data["X"], data["Y"]

    # ---------------------------------------------------------------
    # 1. static structural checks (no fitting)
    # ---------------------------------------------------------------
    print("\n[1] static structural checks")
    ms = build("single", 3, SEED_MODEL)
    mp = build("percolumn_consistent", 3, SEED_MODEL)

    note("initial Z identical", "PASS" if np.array_equal(ms.params["Z"], mp.params["Z"]) else "FAIL",
         f"max|diff|={np.max(np.abs(ms.params['Z']-mp.params['Z'])):.3e}", "must-agree")
    note("initial F identical", "PASS" if np.array_equal(ms.params["F"], mp.params["F"]) else "FAIL",
         f"max|diff|={np.max(np.abs(ms.params['F']-mp.params['F'])):.3e}", "must-agree")
    note("var_z identical", "PASS" if ms.params["var_z"] == mp.params["var_z"] else "FAIL",
         f"{ms.params['var_z']} vs {mp.params['var_z']}", "must-agree")

    eta = np.linspace(-8, 8, 401)[None, :].repeat(D, axis=0).T[:, :D]
    a = ms._mean_function_x(eta); b = mp._mean_function_x(eta)
    note("A'_X (Bernoulli mean)", "PASS" if np.allclose(a, b, atol=1e-15, rtol=0) else "FAIL",
         f"max|diff|={np.max(np.abs(a-b)):.3e} over eta in [-8,8]", "must-agree")
    a = ms._variance_function_x(eta); b = mp._variance_function_x(eta)
    note("A''_X (Bernoulli curvature)", "PASS" if np.allclose(a, b, atol=1e-15, rtol=0) else "FAIL",
         f"max|diff|={np.max(np.abs(a-b)):.3e}", "must-agree")

    # tail behaviour (KI-015 region)
    eta_tail = np.full((2, D), 30.0)
    at = ms._variance_function_x(eta_tail); bt = mp._variance_function_x(eta_tail)
    note("A''_X far tail (eta=30)", "INFO",
         f"legacy-single={at[0,0]:.3e} vs consistent-percolumn={bt[0,0]:.3e} "
         f"(floors differ; KI-015 region)", "may-differ")

    ml = build("percolumn_legacy", 3, SEED_MODEL)
    lt = ml._variance_function_x(eta_tail)
    note("KI-015 legacy per-column tail", "INFO",
         f"legacy per-column A''(30)={lt[0,0]:.3e} (floored, objective-inconsistent); "
         f"consistent={bt[0,0]:.3e}. Experiment will use the CONSISTENT class.",
         "known-issue")

    # ---------------------------------------------------------------
    # 2. parameter count
    # ---------------------------------------------------------------
    print("\n[2] parameter count")
    ok = True
    for k in CAND_K:
        _, p_dual = calc_bic_dual(Q_strict=0.0, k=k, n=N, d=D,
                                  family_x="bernoulli", family_y="gaussian")
        _, p_exp = eval_utils.calc_bic_exp(Q_strict=0.0, k=k, n=N, d=D,
                                           family_x="mixed",
                                           family_y_label="gaussian",
                                           n_gaussian_x_cols=0)
        ok &= (p_dual == p_exp)
    note("num_params agree for all K", "PASS" if ok else "FAIL",
         f"k*d-k(k-1)/2+0+1 in both paths, K=1..9", "must-agree")

    # ---------------------------------------------------------------
    # 3. full EM parity across K
    # ---------------------------------------------------------------
    print("\n[3] end-to-end EM parity, K=1..9")
    recs = []
    for k in CAND_K:
        ms, Zs_s, F_s, sg_s, w0_s, w_s, vz = run_em("single", X, Y, k, SEED_MODEL)
        mp, Zs_p, F_p, sg_p, w0_p, w_p, _ = run_em("percolumn_consistent", X, Y, k, SEED_MODEL)

        q_s = calc_Q_dual_strict(X, Y, Zs_s, F_s, sg_s, vz, w0_s, w_s, ms)
        c_s, p_s = calc_bic_dual(Q_strict=q_s, k=k, n=N, d=D,
                                 family_x="bernoulli", family_y="gaussian")
        q_p = eval_utils.calc_Q_dual_strict_exp(X, Y, Zs_p, F_p, sg_p, vz, w0_p, w_p, mp)
        c_p, p_p = eval_utils.calc_bic_exp(Q_strict=q_p, k=k, n=N, d=D,
                                           family_x="mixed",
                                           family_y_label="gaussian",
                                           n_gaussian_x_cols=0)
        recs.append(dict(k=k, Z_maxdiff=float(np.max(np.abs(Zs_s - Zs_p))),
                         F_maxdiff=float(np.max(np.abs(F_s - F_p))),
                         w0_s=w0_s, w0_p=w0_p, w_s=w_s, w_p=w_p,
                         sy_s=ms.sigma_y, sy_p=mp.sigma_y,
                         q_s=q_s, q_p=q_p, c_s=c_s, c_p=c_p,
                         p_s=p_s, p_p=p_p))
    df = pd.DataFrame(recs)

    note("latent samples Z identical", "PASS" if df.Z_maxdiff.max() < 1e-9 else "FAIL",
         f"max over K of max|Z_single - Z_percolumn| = {df.Z_maxdiff.max():.3e}", "must-agree")
    note("estimated F identical", "PASS" if df.F_maxdiff.max() < 1e-9 else "FAIL",
         f"max|F_single - F_percolumn| = {df.F_maxdiff.max():.3e}", "must-agree")
    note("w0 / w / sigma_y identical", "PASS" if max(
        (df.w0_s - df.w0_p).abs().max(), (df.w_s - df.w_p).abs().max(),
        (df.sy_s - df.sy_p).abs().max()) < 1e-9 else "FAIL",
        f"max|dw0|={np.abs(df.w0_s-df.w0_p).max():.3e} "
        f"max|dw|={np.abs(df.w_s-df.w_p).max():.3e} "
        f"max|dsy|={np.abs(df.sy_s-df.sy_p).max():.3e}", "must-agree")

    # Q differs by the Gaussian-Y -(1/2)ln(2pi) per observed pair (K-independent)
    n_pairs = N * (N - 1) // 2
    expected_offset = -0.5 * np.log(2.0 * np.pi) * n_pairs
    dq = df.q_p - df.q_s
    note("Q offset is the known Gaussian-Y 2pi constant",
         "PASS" if np.max(np.abs(dq - expected_offset)) < 1e-6 else "FAIL",
         f"Q_exp - Q_dual = {dq.mean():.4f} (expected {expected_offset:.4f}); "
         f"spread over K = {float(dq.max()-dq.min()):.3e}", "may-differ")

    dc = df.c_p - df.c_s
    note("criterion offset is constant in K",
         "PASS" if float(dc.max() - dc.min()) < 1e-6 else "FAIL",
         f"C_exp - C_dual = {dc.mean():.4f}, spread over K = {float(dc.max()-dc.min()):.3e} "
         f"-> argmin over K is unaffected", "may-differ")

    ks = int(df.loc[df.c_s.idxmin(), "k"]); kp = int(df.loc[df.c_p.idxmin(), "k"])
    note("selected K identical", "PASS" if ks == kp else "FAIL",
         f"single-family selects K={ks}; per-column selects K={kp}", "must-agree")

    out = _ROOT / "expfam" / "results" / "percolumn_mixed_x_k_selection_20260907"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "parity_by_k.csv", index=False)
    res = pd.DataFrame(ROWS)
    res.to_csv(out / "audit_percolumn_parity.csv", index=False)

    print("\n" + "=" * 74)
    print(df[["k", "Z_maxdiff", "F_maxdiff", "q_s", "q_p", "c_s", "c_p"]].round(4).to_string(index=False))
    nfail = int((res.status == "FAIL").sum())
    print(f"\nPARITY AUDIT: {int((res.status=='PASS').sum())} PASS / {nfail} FAIL / "
          f"{int((res.status=='INFO').sum())} INFO")
    print(f"wrote {out/'audit_percolumn_parity.csv'} and parity_by_k.csv")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
