"""AF4 / S1: can a whole-model evaluation that integrates out Z (Laplace, at
the saved theta-hat of each candidate assignment) tell the true family
assignment from wrong ones, where the column-wise Q-score on the incumbent q
could not?  NO MCEM fit is run (budget 0); only fixed-theta joint-mode
polishing, full Hessians, and importance-sampling diagnostics.

Quantity (per dataset r, Y regime, candidate assignment c)

    Phi_c(Z)   = log p(Z) + log p(X | Z, theta_c, c) + log p(Y_train | Z, theta_c)
                 (complete densities: N(0, var_z I) prior with its constant;
                  X: Gaussian with sigma^2 and 2*pi, Poisson with -log x!,
                  CMP2 with -2 log x! and its log-normaliser; Y: Bernoulli on
                  the unordered TRAIN pairs i<j only)
    L_c        = log p(X, Y_train | theta_c, c) = log ∫ exp Phi_c(Z) dZ
               ≈ Phi_c(Z_hat_c) + (nK/2) log 2π − ½ log|H_c|      (Laplace)
    Z_hat_c    = joint mode of Phi_c at FIXED theta_c (not an E-step sample)
    H_c        = −∇²Phi_c(Z_hat_c), full nK × nK (Y couples nodes)

L_c is a latent-variable marginal likelihood at a plug-in theta_c.  It is NOT
the Bayesian evidence (theta is not integrated), NOT Schwarz BIC, and NOT a
Q-function value.  Every candidate assignment here has the same number of
free parameters (Poisson and CMP(nu=2 fixed) both use K loadings per column),
so no parameter penalty enters a paired difference.

Phi is computed by an implementation written for this experiment
(``log_joint_parts``) from FamilySpec.log_prob; gradient / Hessian come from
the existing ``laplace_k_criterion`` and are validated against finite
differences of this Phi.  The existing ``mode_phi`` of AF2 is NOT used as a
density: ``calc_Q_dual_strict_exp`` adds an extra −Σ log x! to every column
labelled 'poisson' (CMP columns included).

Lineage: E-consistent per-column + PR #129 generic model (experimental
prototype; not adoptable for the manuscript).

Usage
    python run_af4_laplace_model_comparison.py --phase-a
    python run_af4_laplace_model_comparison.py --write-protocol
    python run_af4_laplace_model_comparison.py --smoke
    python run_af4_laplace_model_comparison.py --execute
    python run_af4_laplace_model_comparison.py --analyze --stage full
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit, gammaln

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
for p in (HERE, HERE.parent, ROOT / "tools" / "research_audit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_af2_poisson_cmp as af2                                      # noqa: E402
from run_auto_family_z_recovery import sha16, sha256_file, write_csv  # noqa: E402

EXPERIMENT = "AF4-S1"
N, K, D = af2.N, af2.K, af2.D
TRUE_MODELS = list(af2.TRUE_MODELS)
GAUSS_COLS, POIS_COLS, CMP_COLS = (0, 1, 2), (3, 4, 5), (6, 7, 8)
REGIMES = ("dense", "sparse")
PRIMARY_METHODS = ("oracle", "support_only", "wrong")
SECONDARY_METHODS = ("auto_first", "auto_last")
METHODS = PRIMARY_METHODS + SECONDARY_METHODS
STARTS = ("saved_mode", "last_sample", "mean5")       # primary = saved_mode
MODE_GRAD_TOL, MODE_MAX_ITER = 1e-8, 200              # = E1 / AF2 joint-mode settings
IS_BATCHES, IS_PER_BATCH, IS_SEED_BASE = 5, 400, 1665000
PRED_DRAWS, PRED_SEED_BASE = 2000, 1666000
FD_STEP_GRAD, FD_STEP_HESS = 1e-5, 1e-5
FD_TOL_GRAD_REL, FD_TOL_HESS_REL = 1e-5, 1e-5
CONST_TOL = 1e-8
FD_SUBSET = [(r, reg, m) for r in (0, 1, 2) for reg in REGIMES for m in PRIMARY_METHODS]
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 10000, 20261114

AF2_FULL = ROOT / "expfam" / "results" / "auto_family_z_recovery" / "af2_20261110" / "full"
AF2_STATE = af2.STATE_ROOT / "full" / "fits"
OUT_ROOT = ROOT / "expfam" / "results" / "auto_family_z_recovery" / "af4_laplace_model_comparison_20261113"
LOG2PI = math.log(2.0 * math.pi)


def git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# model reconstruction from an AF2 state
# --------------------------------------------------------------------------

def load_state(r: int, regime: str, method: str) -> dict[str, np.ndarray]:
    key = f"{af2.m1.replicate_label(r)}_{regime}_{method}"
    with np.load(AF2_STATE / f"{key}.npz", allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def build_model(assignment: list[str], train_mask: np.ndarray, st: dict[str, np.ndarray]):
    from model_dual_expfam_generic import DualExpFamLSMGeneric
    reg, _ = af2.registry_and_priority()
    model = DualExpFamLSMGeneric(n=N, d=D, k=K, L=af2.L, column_specs=[reg[c] for c in assignment],
                                 family_y="bernoulli", train_mask=train_mask)
    model.initialize_params(true_params=None, seed=0)
    w0, w, var_z = (float(v) for v in st["scalars"])
    model.params.update({"F": np.asarray(st["F"], float), "sigma": np.asarray(st["sigma"], float),
                         "w0": w0, "w": w, "var_z": var_z})
    return model


# --------------------------------------------------------------------------
# independent complete log joint (decomposed)
# --------------------------------------------------------------------------

def log_joint_parts(model, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> dict[str, float]:
    p = model.params
    F, sigma, w0, w, var_z = (np.asarray(p["F"], float), np.asarray(p["sigma"], float),
                              float(p["w0"]), float(p["w"]), float(p["var_z"]))
    n, k = Z.shape
    prior = -0.5 * n * k * math.log(2.0 * math.pi * var_z) - float(np.sum(Z ** 2)) / (2.0 * var_z)
    eta_x = Z @ F.T
    col = np.zeros(D)
    for j, spec in enumerate(model.column_specs):
        if spec.family_name == "gaussian":
            s2 = float(np.diag(sigma)[j])
            col[j] = float(np.sum(-0.5 * (X[:, j] - eta_x[:, j]) ** 2 / s2 - 0.5 * math.log(s2) - 0.5 * LOG2PI))
        else:
            col[j] = float(np.sum(spec.log_prob(X[:, j], eta_x[:, j])))
    up = np.triu(np.asarray(model.train_mask, bool), k=1)
    eta_y = (w0 + w * (Z @ Z.T))[up]
    y = Y[up]
    ll_y = float(np.sum(y * eta_y - np.logaddexp(0.0, eta_y)))
    out = {"prior": prior, "x_gauss": float(col[list(GAUSS_COLS)].sum()),
           "x_true_pois_cols": float(col[list(POIS_COLS)].sum()),
           "x_true_cmp_cols": float(col[list(CMP_COLS)].sum()), "y_train": ll_y}
    out["x_total"] = out["x_gauss"] + out["x_true_pois_cols"] + out["x_true_cmp_cols"]
    out["phi"] = prior + out["x_total"] + ll_y
    return out


def phi_true(model, X, Y, Z) -> float:
    return log_joint_parts(model, X, Y, Z)["phi"]


# --------------------------------------------------------------------------
# numerical validation pieces
# --------------------------------------------------------------------------

def fd_gradient(model, X, Y, Z, h=FD_STEP_GRAD) -> np.ndarray:
    g = np.zeros(Z.size)
    z = Z.ravel()
    for a in range(z.size):
        zp, zm = z.copy(), z.copy()
        zp[a] += h
        zm[a] -= h
        g[a] = (phi_true(model, X, Y, zp.reshape(Z.shape)) - phi_true(model, X, Y, zm.reshape(Z.shape))) / (2 * h)
    return g.reshape(Z.shape)


def fd_neg_hessian(model, X, Y, Z, h=FD_STEP_HESS) -> np.ndarray:
    import laplace_k_criterion as lk
    z = Z.ravel()
    Hn = np.zeros((z.size, z.size))
    for a in range(z.size):
        zp, zm = z.copy(), z.copy()
        zp[a] += h
        zm[a] -= h
        gp = lk.gradient(model, X, Y, zp.reshape(Z.shape)).ravel()
        gm = lk.gradient(model, X, Y, zm.reshape(Z.shape)).ravel()
        Hn[:, a] = -(gp - gm) / (2 * h)
    return 0.5 * (Hn + Hn.T)


def offdiag_block_norm(H: np.ndarray) -> float:
    Hb = H.reshape(N, K, N, K).transpose(0, 2, 1, 3).copy()
    Hb[np.arange(N), np.arange(N)] = 0.0
    return float(np.sqrt(np.sum(Hb ** 2)))


# --------------------------------------------------------------------------
# Phase A: feasibility audit (no logdet, no Laplace value, no comparison)
# --------------------------------------------------------------------------

def phase_a(out: Path) -> dict[str, Any]:
    import laplace_k_criterion as lk
    out.mkdir(parents=True, exist_ok=False)
    fr = {r["fit_key"]: r for r in read_csv(AF2_FULL / "fit_runs.csv")}
    rows, fd_rows, leak_rows = [], [], []
    t0 = time.perf_counter()
    for r in range(20):
        ds = af2.generate(r)
        X, Y = np.asarray(ds.X, float), np.asarray(ds.Y, float)
        masks = af2.m1.build_masks(r)
        count_logfact = float(np.sum(gammaln(X[:, 3:] + 1.0)))
        for regime in REGIMES:
            train = masks[regime]
            for method in METHODS:
                key = f"{af2.m1.replicate_label(r)}_{regime}_{method}"
                f = fr[key]
                st = load_state(r, regime, method)
                assign = f["assignment"].split("|")
                model = build_model(assign, train, st)
                Zm = np.asarray(st["Zmode"], float)
                g = lk.gradient(model, X, Y, Zm)
                phi_existing = lk.complete_log_density(model, X, Y, Zm)
                parts = log_joint_parts(model, X, Y, Zm)
                offs = []
                for Zt in (Zm, st["Zsamples"][:, :, -1], st["Zsamples"].mean(axis=2)):
                    offs.append(phi_true(model, X, Y, Zt) - lk.complete_log_density(model, X, Y, Zt))
                sig2 = np.diag(st["sigma"])[list(GAUSS_COLS)]
                rows.append({
                    "fit_key": key, "replicate": af2.m1.replicate_label(r), "regime": regime,
                    "method": method, "assignment": f["assignment"],
                    "npz_sha_match": sha256_file(AF2_STATE / f"{key}.npz") == f["state_sha256"],
                    "data_sha_match": sha16(ds.X) == f["x_sha"] and sha16(ds.Y) == f["y_sha"],
                    "train_mask_sha_match": sha16(train) == f["train_mask_sha"],
                    "train_test_overlap": int(np.triu(train & masks["test"], 1).sum()),
                    "mask_symmetric_zero_diag": bool(np.array_equal(train, train.T) and not np.any(np.diag(train))),
                    "theta_match": abs(float(st["scalars"][0]) - float(f["w0_hat"])) < 1e-12
                                   and abs(float(st["scalars"][1]) - float(f["w_hat"])) < 1e-12,
                    "var_z": float(st["scalars"][2]),
                    "gauss_sigma2_min": float(sig2.min()),
                    "nongauss_sigma_is_one": bool(np.allclose(np.diag(st["sigma"])[3:], 1.0)),
                    "saved_mode_grad_inf_recomputed": float(np.max(np.abs(g))),
                    "fit_runs_mode_grad_inf": float(f["mode_grad_inf"]),
                    "phi_existing_minus_mode_phi": phi_existing - float(f["mode_phi"]),
                    "offset_true_minus_existing_at_mode": offs[0],
                    "offset_spread_over_3_Z": float(max(offs) - min(offs)),
                    "expected_offset_sum_logfact_count_cols": count_logfact,
                    "offset_minus_expected": offs[0] - count_logfact,
                    "phi_true_finite": bool(np.isfinite(parts["phi"])),
                })
        print(f"phase A {af2.m1.replicate_label(r)} {round(time.perf_counter() - t0, 1)} s", flush=True)
    # finite-difference validation on the pre-declared subset
    for r, regime, method in FD_SUBSET:
        ds = af2.generate(r)
        X, Y = np.asarray(ds.X, float), np.asarray(ds.Y, float)
        masks = af2.m1.build_masks(r)
        st = load_state(r, regime, method)
        f = fr[f"{af2.m1.replicate_label(r)}_{regime}_{method}"]
        model = build_model(f["assignment"].split("|"), masks[regime], st)
        for zname, Zt in (("saved_mode", np.asarray(st["Zmode"], float)),
                          ("last_sample", np.asarray(st["Zsamples"][:, :, -1], float))):
            ga = lk.gradient(model, X, Y, Zt)
            gf = fd_gradient(model, X, Y, Zt)
            Ha = 0.5 * (lk.negative_hessian(model, X, Y, Zt) + lk.negative_hessian(model, X, Y, Zt).T)
            Hf = fd_neg_hessian(model, X, Y, Zt)
            ev = np.linalg.eigvalsh(Ha)
            try:
                np.linalg.cholesky(Ha)
                chol_ok = True
            except np.linalg.LinAlgError:
                chol_ok = False
            fd_rows.append({
                "replicate": af2.m1.replicate_label(r), "regime": regime, "method": method, "point": zname,
                "grad_scale": float(np.max(np.abs(gf))) if zname != "saved_mode" else float("nan"),
                "grad_max_abs_err": float(np.max(np.abs(ga - gf))),
                "grad_rel_err": float(np.max(np.abs(ga - gf)) / max(1.0, float(np.max(np.abs(gf))))),
                "hess_max_abs_err": float(np.max(np.abs(Ha - Hf))),
                "hess_rel_err": float(np.max(np.abs(Ha - Hf)) / float(np.max(np.abs(Hf)))),
                "offdiag_norm_analytic": offdiag_block_norm(Ha),
                "offdiag_norm_fd": offdiag_block_norm(Hf),
                "offdiag_rel_err": float(abs(offdiag_block_norm(Ha) - offdiag_block_norm(Hf)) / max(offdiag_block_norm(Hf), 1e-300)),
                "hess_min_eig": float(ev[0]), "hess_max_eig": float(ev[-1]),
                "cholesky_ok": chol_ok, "symmetric_err": float(np.max(np.abs(
                    lk.negative_hessian(model, X, Y, Zt) - lk.negative_hessian(model, X, Y, Zt).T)))})
            # leakage: Phi and gradient must not depend on test-pair Y
            test_up = np.triu(masks["test"], 1)
            Yf = Y.copy()
            Yf[test_up] = 1.0 - Yf[test_up]
            Yf = np.triu(Yf, 1) + np.triu(Yf, 1).T
            tr_up = np.argwhere(np.triu(masks[regime], 1))[0]
            Yt = Y.copy()
            Yt[tr_up[0], tr_up[1]] = Yt[tr_up[1], tr_up[0]] = 1.0 - Y[tr_up[0], tr_up[1]]
            leak_rows.append({
                "replicate": af2.m1.replicate_label(r), "regime": regime, "method": method, "point": zname,
                "phi_change_test_flip": abs(phi_true(model, X, Yf, Zt) - phi_true(model, X, Y, Zt)),
                "grad_change_test_flip": float(np.max(np.abs(lk.gradient(model, X, Yf, Zt) - ga))),
                "phi_change_train_flip": abs(phi_true(model, X, Yt, Zt) - phi_true(model, X, Y, Zt))})
        print(f"phase A FD {af2.m1.replicate_label(r)} {regime} {method} {round(time.perf_counter() - t0, 1)} s", flush=True)
    write_csv(out / "reconstruction_audit.csv", rows)
    write_csv(out / "numerical_validation.csv", fd_rows)
    write_csv(out / "leakage_check.csv", leak_rows)
    summ = {
        "cases": len(rows),
        "all_npz_sha_match": all(x["npz_sha_match"] for x in rows),
        "all_data_sha_match": all(x["data_sha_match"] for x in rows),
        "all_train_mask_sha_match": all(x["train_mask_sha_match"] for x in rows),
        "all_no_train_test_overlap": all(x["train_test_overlap"] == 0 for x in rows),
        "all_mask_symmetric_zero_diag": all(x["mask_symmetric_zero_diag"] for x in rows),
        "all_theta_match": all(x["theta_match"] for x in rows),
        "var_z_values": sorted({x["var_z"] for x in rows}),
        "gauss_sigma2_min": min(x["gauss_sigma2_min"] for x in rows),
        "all_nongauss_sigma_one": all(x["nongauss_sigma_is_one"] for x in rows),
        "max_saved_mode_grad_inf_recomputed": max(x["saved_mode_grad_inf_recomputed"] for x in rows),
        "max_abs_phi_existing_minus_mode_phi": max(abs(x["phi_existing_minus_mode_phi"]) for x in rows),
        "max_offset_spread_over_Z": max(x["offset_spread_over_3_Z"] for x in rows),
        "max_abs_offset_minus_expected": max(abs(x["offset_minus_expected"]) for x in rows),
        "fd_cases": len(fd_rows),
        "max_grad_rel_err": max(x["grad_rel_err"] for x in fd_rows),
        "max_hess_rel_err": max(x["hess_rel_err"] for x in fd_rows),
        "max_offdiag_rel_err": max(x["offdiag_rel_err"] for x in fd_rows),
        "min_offdiag_norm": min(x["offdiag_norm_analytic"] for x in fd_rows),
        "all_cholesky_ok_at_saved_mode": all(x["cholesky_ok"] for x in fd_rows if x["point"] == "saved_mode"),
        "max_phi_change_test_flip": max(x["phi_change_test_flip"] for x in leak_rows),
        "max_grad_change_test_flip": max(x["grad_change_test_flip"] for x in leak_rows),
        "min_phi_change_train_flip": min(x["phi_change_train_flip"] for x in leak_rows),
        "runtime_s": round(time.perf_counter() - t0, 1),
        "git_head": git("rev-parse", "HEAD"),
    }
    summ["gate_pass"] = bool(
        summ["all_npz_sha_match"] and summ["all_data_sha_match"] and summ["all_train_mask_sha_match"]
        and summ["all_no_train_test_overlap"] and summ["all_mask_symmetric_zero_diag"] and summ["all_theta_match"]
        and summ["max_saved_mode_grad_inf_recomputed"] <= MODE_GRAD_TOL
        and summ["max_abs_phi_existing_minus_mode_phi"] <= 1e-6
        and summ["max_offset_spread_over_Z"] <= CONST_TOL * 1e3
        and summ["max_grad_rel_err"] <= FD_TOL_GRAD_REL and summ["max_hess_rel_err"] <= FD_TOL_HESS_REL
        and summ["max_phi_change_test_flip"] == 0.0 and summ["max_grad_change_test_flip"] == 0.0
        and summ["min_phi_change_train_flip"] > 0.0)
    (out / "phase_a_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    return summ


# --------------------------------------------------------------------------
# Phase B: one case
# --------------------------------------------------------------------------

def laplace_at(model, X, Y, Z0) -> dict[str, Any]:
    import laplace_k_criterion as lk
    t0 = time.perf_counter()
    Zh, iters, ginf = lk.find_joint_mode(model, X, Y, Z0, grad_tol=MODE_GRAD_TOL, max_iter=MODE_MAX_ITER)
    parts = log_joint_parts(model, X, Y, Zh)
    H = lk.negative_hessian(model, X, Y, Zh)
    H = 0.5 * (H + H.T)
    sign, logdet = np.linalg.slogdet(H)
    ev = np.linalg.eigvalsh(H)
    try:
        Lc = np.linalg.cholesky(H)
        logdet_chol = float(2.0 * np.sum(np.log(np.diag(Lc))))
    except np.linalg.LinAlgError:
        logdet_chol = float("nan")
    status = "OK"
    if ginf > MODE_GRAD_TOL:
        status = "NOT_STATIONARY"
    elif sign <= 0 or ev[0] <= 0:
        status = "HESSIAN_NOT_PD"
    lap = parts["phi"] + 0.5 * N * K * LOG2PI - 0.5 * logdet if status == "OK" else float("nan")
    return {"Z_hat": Zh, "H": H, "status": status, "iterations": iters, "grad_inf": ginf,
            "logdet": float(logdet), "logdet_chol": logdet_chol, "logdet_sign": float(sign),
            "min_eig": float(ev[0]), "max_eig": float(ev[-1]), "cond": float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf"),
            "laplace": float(lap), "seconds": round(time.perf_counter() - t0, 3),
            **{f"phi_{k}": v for k, v in parts.items()}}


def heldout_y(model, Y, test: np.ndarray, Zh: np.ndarray, H: np.ndarray, seed: int) -> dict[str, float]:
    w0, w = float(model.params["w0"]), float(model.params["w"])
    rows, cols = np.nonzero(np.triu(test, 1))
    y = Y[rows, cols]
    eta = w0 + w * np.sum(Zh[rows] * Zh[cols], axis=1)
    plug = float(np.mean(y * eta - np.logaddexp(0.0, eta)))
    try:
        Lc = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        return {"heldout_y_plugin_mean_log": plug, "heldout_y_laplace_pred_mean_log": float("nan")}
    from scipy.linalg import solve_triangular
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((PRED_DRAWS, N * K))
    dz = solve_triangular(Lc.T, eps.T, lower=False).T.reshape(PRED_DRAWS, N, K)
    Zd = Zh[None] + dz
    eta_d = w0 + w * np.sum(Zd[:, rows, :] * Zd[:, cols, :], axis=2)
    pbar = expit(eta_d).mean(axis=0)
    pred = float(np.mean(y * np.log(pbar) + (1 - y) * np.log1p(-pbar)))
    return {"heldout_y_plugin_mean_log": plug, "heldout_y_laplace_pred_mean_log": pred}


def run_case(r: int, regime: str, method: str, ds, masks, fr, case_index: int) -> tuple[dict, list]:
    import laplace_importance as li
    key = f"{af2.m1.replicate_label(r)}_{regime}_{method}"
    f = fr[key]
    st = load_state(r, regime, method)
    X, Y = np.asarray(ds.X, float), np.asarray(ds.Y, float)
    model = build_model(f["assignment"].split("|"), masks[regime], st)
    base = {"replicate": af2.m1.replicate_label(r), "data_seed": af2.DATA_SEEDS[r], "regime": regime,
            "method": method, "primary_method": method in PRIMARY_METHODS, "assignment": f["assignment"],
            "n_correct_count_columns": int(f["n_correct_count_columns"]),
            "w0_hat": float(st["scalars"][0]), "w_hat": float(st["scalars"][1]), "var_z": float(st["scalars"][2])}
    starts = {"saved_mode": np.asarray(st["Zmode"], float),
              "last_sample": np.asarray(st["Zsamples"][:, :, -1], float),
              "mean5": np.asarray(st["Zsamples"].mean(axis=2), float)}
    res, diag_rows = {}, []
    for name, Z0 in starts.items():
        try:
            res[name] = laplace_at(model, X, Y, Z0)
        except Exception as exc:                         # recorded, never retried
            res[name] = {"status": f"ERROR: {type(exc).__name__}: {exc}"[:200], "laplace": float("nan")}
        d = {k: v for k, v in res[name].items() if k not in ("Z_hat", "H")}
        diag_rows.append(dict(base, start=name, **d))
    prim = res["saved_mode"]
    row = dict(base, **{k: v for k, v in prim.items() if k not in ("Z_hat", "H")})
    row["mode_shift_from_saved"] = (float(np.max(np.abs(prim["Z_hat"] - starts["saved_mode"])))
                                    if "Z_hat" in prim else float("nan"))
    ok_starts = {k: v for k, v in res.items() if v.get("status") == "OK"}
    row["n_starts_ok"] = len(ok_starts)
    if ok_starts:
        best = max(ok_starts, key=lambda k: ok_starts[k]["laplace"])
        row["best_start"] = best
        row["laplace_best_of_starts"] = ok_starts[best]["laplace"]
        row["phi_best_of_starts"] = ok_starts[best]["phi_phi"]
        if "Z_hat" in prim and prim["status"] == "OK":
            row["max_mode_diff_across_starts"] = max(float(np.max(np.abs(v["Z_hat"] - prim["Z_hat"])))
                                                     for v in ok_starts.values())
            row["max_laplace_diff_across_starts"] = max(abs(v["laplace"] - prim["laplace"]) for v in ok_starts.values())
    if prim.get("status") == "OK":
        isr = li.importance_correction(model, X, Y, prim["Z_hat"], prim["H"],
                                       seeds=[IS_SEED_BASE + 100 * case_index + b for b in range(IS_BATCHES)],
                                       samples_per_batch=IS_PER_BATCH)
        pooled, bs = isr["pooled"], isr["batch_correction_stats"]
        row.update({"is_correction": pooled["correction"], "is_ess": pooled["ess"],
                    "is_relative_ess": pooled["relative_ess"], "is_max_weight_share": pooled["max_weight_share"],
                    "is_non_finite": pooled["non_finite"], "is_batch_sd": bs["sd"],
                    "is_batch_min": bs["min"], "is_batch_max": bs["max"],
                    "laplace_is_corrected": prim["laplace"] + pooled["correction"]})
        row.update(heldout_y(model, Y, masks["test"], prim["Z_hat"], prim["H"], PRED_SEED_BASE + case_index))
    return row, diag_rows


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------

def protocol() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "mode": "CONFIRMATORY for the primary comparisons (written after the Phase A feasibility audit, "
                "before any Laplace value, log-determinant or model comparison was computed)",
        "research_question": "Does a whole-model Laplace approximation of log p(X, Y_train | theta_c, c), with Z "
                             "integrated out at each candidate's saved theta-hat, rank the true assignment above "
                             "Support-only and Wrong on the same data?",
        "not_claimed": "a new automatic selection method; Oracle is a diagnostic reference that knows the truth",
        "lineage": "E-consistent per-column + PR #129 generic model (experimental prototype; not adoptable). "
                   "PR #129 not merged; AF1/AF2 (#166/#167) Draft; AF3 unpushed",
        "mcem_fits": 0,
        "quantity": {
            "target": "log p(X, Y_train | theta_hat_c, c) = log ∫ p(X, Y_train, Z | theta_hat_c, c) dZ "
                      "(latent marginal likelihood at plug-in theta; NOT Bayesian evidence, NOT Schwarz BIC, "
                      "NOT a Q value)",
            "laplace": "Phi(Z_hat) + (nK/2) log 2π − ½ log|H|, Z_hat = joint mode at fixed theta_hat_c, "
                       "H = full nK×nK negative Hessian including the Y cross blocks",
            "phi": "independent implementation log_joint_parts: N(0, var_z I) prior with constant; Gaussian X with "
                   "sigma^2 and 2π; Poisson −log x!; CMP2 −2 log x! and log-normaliser; Bernoulli Y over unordered "
                   "train pairs i<j only; test pairs never enter",
            "gradient_hessian": "laplace_k_criterion.gradient / negative_hessian (existing), validated by finite "
                                "differences of the independent Phi in Phase A",
            "penalty": "none: all candidate assignments have identical free-parameter counts"},
        "data": {"datasets": "AF2, 20 (seed 1661001+r)", "regimes": list(REGIMES),
                 "masks": "AF2 / M1 build_masks(r); Y_train = regime mask; test pairs only for secondary held-out Y"},
        "assignments": {"primary": list(PRIMARY_METHODS), "secondary": list(SECONDARY_METHODS),
                        "theta": "each assignment's own saved AF2 refit theta-hat (F, sigma, w0, w, var_z)"},
        "mode": {"starts": list(STARTS), "primary_start": "saved_mode (AF2 joint mode, re-polished)",
                 "optimizer": "laplace_k_criterion.find_joint_mode (damped full Newton, backtracking on Phi)",
                 "grad_tol": MODE_GRAD_TOL, "max_iter": MODE_MAX_ITER,
                 "sensitivity": "best-of-starts Laplace; mode differences across starts recorded"},
        "hessian": "full analytic, symmetrised; logdet by slogdet and by Cholesky (agreement recorded); no ridge, "
                   "no jitter; OK only if max|grad| <= grad_tol and H positive definite",
        "importance_sampling": {"purpose": "diagnostic of the Laplace error only (secondary)",
                                "proposal": "N(Z_hat, H^-1)", "batches": IS_BATCHES, "per_batch": IS_PER_BATCH,
                                "seed": f"{IS_SEED_BASE} + 100*case_index + batch"},
        "heldout_y": {"secondary": True, "plugin": "mean Bernoulli log-lik of test pairs at Z_hat",
                      "laplace_predictive": f"mean log of E_q[p] with q = N(Z_hat, H^-1), {PRED_DRAWS} draws, "
                                            f"seed {PRED_SEED_BASE} + case_index",
                      "caution": "Y prediction alone does not identify the X family"},
        "x_caution": "all X entries were used to fit; any X log-likelihood here is in-sample, never 'held-out X'",
        "primary_metrics": {
            "per regime": "number of datasets (of 20 with both OK) with Laplace(oracle) > Laplace(support_only), "
                          "and > Laplace(wrong); mean paired difference with bootstrap CI",
            "note": "a post-truth diagnostic of discriminability, not a selection accuracy"},
        "secondary_metrics": [
            "decomposition of each paired difference into Δ Phi(Z_hat) [prior, X gauss, X true-Poisson cols, X true-CMP "
            "cols, Y] and Δ(−½ log|H|)",
            "oracle vs auto_first / auto_last on datasets where the Auto assignment differs from Oracle",
            "oracle vs support_only on datasets where the AF3 Z5 Q-score misselected at least one CMP2 column",
            "best-of-starts sensitivity", "IS-corrected Laplace and IS diagnostics (ESS, batch spread)",
            "held-out Y plug-in and Laplace-predictive log score", "Hessian eigenvalues, condition number, iterations"],
        "statistical_unit": "dataset (20 per regime); columns and models within a dataset are never independent units",
        "interval": f"percentile bootstrap over datasets ({BOOTSTRAP_RESAMPLES}, seed {BOOTSTRAP_SEED})",
        "failure_rule": "a non-OK primary-start Laplace (NOT_STATIONARY, HESSIAN_NOT_PD, ERROR) is kept and reported; "
                        "a paired comparison uses only datasets where both members are OK; exclusions counted; "
                        "nothing is retried, re-seeded or ridged",
        "reading_guide_predeclared": {
            "note": "judgement aids, not significance tests",
            "R1": "Oracle > Support-only AND Oracle > Wrong in >= 18/20 datasets in both regimes -> the Laplace "
                  "whole-model comparison discriminates in this setting",
            "R2": "Oracle > Support-only in <= 14/20 in a regime -> not reliably discriminative there",
            "R3": "if the sign of a paired difference differs from the sign of its Δ Phi in >= 3 datasets, the "
                  "Hessian term is decisive",
            "R4": "if best-of-starts or IS-corrected values reverse the sign of a primary paired difference in >= 2 "
                  "datasets of a regime, the Laplace result is fragile there (IS read only where relative ESS >= 0.01)"},
        "budget": {"cases": 20 * 2 * len(METHODS), "mode_optimisations": 20 * 2 * len(METHODS) * len(STARTS),
                   "is_samples": 20 * 2 * len(METHODS) * IS_BATCHES * IS_PER_BATCH},
        "artifacts": {"results": str(OUT_ROOT.relative_to(ROOT)).replace("\\", "/")},
    }


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True).encode()).hexdigest()


def runinfo(extra: dict[str, Any]) -> dict[str, Any]:
    import scipy
    req = ROOT / "requirements.txt"
    return {"experiment": EXPERIMENT, "script": "expfam/src/experimental/run_af4_laplace_model_comparison.py",
            "git_head": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_tracked": bool(git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "requirements_sha256": sha256_file(req) if req.exists() else "",
            "protocol_sha256": protocol_sha256(), "command": " ".join(sys.argv), **extra}


def execute(out: Path, replicates: list[int]) -> None:
    out.mkdir(parents=True, exist_ok=False)
    fr = {r["fit_key"]: r for r in read_csv(AF2_FULL / "fit_runs.csv")}
    rows, diag = [], []
    t0 = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    case_index = 0
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        for r in replicates:
            ds = af2.generate(r)
            masks = af2.m1.build_masks(r)
            for regime in REGIMES:
                for method in METHODS:
                    row, d = run_case(r, regime, method, ds, masks, fr, case_index)
                    rows.append(row)
                    diag.extend(d)
                    case_index += 1
            print(f"{af2.m1.replicate_label(r)} done {round(time.perf_counter() - t0, 1)} s", flush=True)
        warn = sorted({f"{x.category.__name__}: {str(x.message)[:160]}" for x in wl})[:20]
        n_warn = len(wl)
    write_csv(out / "model_comparison.csv", rows)
    write_csv(out / "diagnostics.csv", diag)
    (out / "runinfo.json").write_text(json.dumps(runinfo({
        "stage": out.name, "replicates": [af2.m1.replicate_label(r) for r in replicates],
        "cases": len(rows), "mode_optimisations": len(diag),
        "is_samples": sum(1 for x in rows if "is_correction" in x) * IS_BATCHES * IS_PER_BATCH,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_s": round(time.perf_counter() - t0, 1), "n_warnings": n_warn, "warning_messages": warn,
        "af2_state_dir": str(AF2_STATE), "mcem_fits": 0}), indent=2), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "AF4").splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--phase-a", action="store_true")
    g.add_argument("--write-protocol", action="store_true")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--execute", action="store_true")
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--stage", default="full")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args(argv)
    out_root = Path(args.out_root)
    if args.phase_a:
        s = phase_a(out_root / "phase_a")
        print(json.dumps(s, indent=2))
        return 0
    if args.write_protocol:
        p = out_root / "protocol.json"
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
        p.write_text(json.dumps(protocol(), indent=2), encoding="utf-8")
        write_csv(out_root / "manifest.csv", [
            {"replicate": af2.m1.replicate_label(r), "data_seed": af2.DATA_SEEDS[r], "regime": reg,
             "method": m, "primary": m in PRIMARY_METHODS, "starts": "|".join(STARTS),
             "af2_state": f"{af2.m1.replicate_label(r)}_{reg}_{m}.npz"}
            for r in range(20) for reg in REGIMES for m in METHODS])
        print(f"wrote {p} sha256(canonical)={protocol_sha256()}")
        return 0
    if args.analyze:
        import analyze_af4_laplace_model_comparison as an
        an.main(out_root / args.stage)
        return 0
    frozen = json.loads((out_root / "protocol.json").read_text(encoding="utf-8"))
    if hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest() != protocol_sha256():
        raise SystemExit("protocol.json on disk differs from protocol(); refusing to run")
    pa = json.loads((out_root / "phase_a" / "phase_a_summary.json").read_text(encoding="utf-8"))
    if not pa["gate_pass"]:
        raise SystemExit("Phase A gate did not pass; refusing to run Phase B")
    execute(out_root / ("smoke" if args.smoke else "full"), [0] if args.smoke else list(range(20)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
