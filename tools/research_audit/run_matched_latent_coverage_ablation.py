"""
Matched latent-coverage ablation (Issue #31).

GitHub Issue #31 is the sole authoritative scientific specification for this experiment.
Nothing here may be tuned after seeing output.

Research question
-----------------
Does the joint per-column advantage become substantially smaller when each individual family
already covers all latent dimensions, while family composition, column count, loading
magnitudes, Gaussian sigma, K, Y information and the fitting procedure are held fixed?

This experiment **more tightly targets latent-coverage / block-rank geometry while holding
the major previously identified factors fixed**. It does NOT isolate that factor alone -
Bernoulli/Poisson curvature stays eta-dependent, their true block traces are only
approximately matched, finite-sample latent correlation remains, and the joint model's own
precision geometry changes across regimes. The words "alone" and "fully isolated" must not
be used about this experiment.

Two F-structure regimes
-----------------------
A `complementary`   : bernoulli rows -> z1, gaussian rows -> z2, poisson rows -> z3
B `full_coverage`   : within each family, row1 -> z1, row2 -> z2, row3 -> z3

F_full is NOT independently resampled. It is a per-row latent-coordinate permutation of
F_complementary:  shift = (target_dim - source_dim) % K_TRUE ;  np.roll(F_comp[l], shift).

Fixed design (pre-registered)
-----------------------------
N=80, D=9, K_TRUE=3, 3 bernoulli + 3 gaussian + 3 poisson
L=5, num_iter=8, trials=10, test_ratio=0.2, y_obs_rate in {1.0, 0.1}
sigma_G=0.3, dominant_weight=0.9, minor_weight=0.15, w0_true=1.2, w_true=0.3
Z_true = column-wise standardized N(0,I) draw ; generator eta/mu clipping = NONE
all fits: numerics_mode="consistent"

Primary (exactly one estimand)
------------------------------
domain y_obs_rate=0.1 ; endpoint whole-space Procrustes RMSE_Z ; comparator single_gaussian
delta_G(r,t) = RMSE_Z(single_gaussian,r,t) - RMSE_Z(per_column_all,r,t)
I_t          = delta_G(complementary,t) - delta_G(full_coverage,t)
Mandatory decomposition components (NOT co-primary):
D_G,t = RMSE_Z(single_gaussian,comp,t) - RMSE_Z(single_gaussian,full,t)
D_J,t = RMSE_Z(per_column_all,comp,t)  - RMSE_Z(per_column_all,full,t)
identity asserted:  I_t == D_G,t - D_J,t

Poisson sampling provenance
---------------------------
common-random-number inverse-CDF coupling, with q=0 replaced only by nextafter(0,1) for
floating-point endpoint safety; no eta/mu clipping. The sampling implementation path differs
from Issue #27, so bitwise reproduction of Issue #27 is NOT claimed; the marginal generator
distribution is the same statistical model.

Integrity
---------
Any of: generator-only criterion failure / internal retry ("[NaN iter=") / nan_occurred /
q_bic_failed / support violation / non-finite eta, mu or scientific metric /
numerics_mode != "consistent" / wrong fit count / duplicate key / hash-pairing mismatch /
precision identity failure / I=D_G-D_J identity failure  -> STOP the whole experiment.
No seed change, no drop, no retry rescue, no parameter change.

Run
---
    python tools/research_audit/run_matched_latent_coverage_ablation.py --smoke --out-dir DIR
    python tools/research_audit/run_matched_latent_coverage_ablation.py
"""

import argparse
import contextlib
import hashlib
import io
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson as _spoisson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "expfam" / "src"))
sys.path.insert(0, str(_ROOT / "expfam" / "src" / "experimental"))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from utils_expfam import procrustes_rotation, calc_rmse            # noqa: E402
from em_runner import run_em_experimental, predict_mu_y            # noqa: E402
from eval_utils import (make_pair_split, heldout_count_metrics,    # noqa: E402
                        upper_pairs_of)

# ── Fixed design (Issue #31) ──────────────────────────────────────────────
N, D, K_TRUE = 80, 9, 3
W0_TRUE, W_TRUE = 1.2, 0.3
L, NITER = 5, 8
TEST_RATIO = 0.2
TRIALS_FULL, TRIALS_SMOKE = 10, 1
Y_OBS_RATES = [1.0, 0.1]
SIGMA_G = 0.3
DOMINANT_WEIGHT, MINOR_WEIGHT = 0.9, 0.15

DATA_SEED_BASE = 120000
MODEL_SEED_BASE = 121000
SPLIT_SEED_BASE = 122000
THIN_SEED_BASE = 123000

SPAWN_ORDER = ["Z", "F", "X_bern", "X_gauss", "X_pois", "Y"]

# true block -> (columns, source/complementary dominant latent dimension)
BLOCKS = {"bern": (np.arange(0, 3), 0),
          "gauss": (np.arange(3, 6), 1),
          "pois": (np.arange(6, 9), 2)}
BLOCK_FAMILY = {"bern": "bernoulli", "gauss": "gaussian", "pois": "poisson"}
FAM_LIST_TRUE = (["bernoulli"] * 3) + (["gaussian"] * 3) + (["poisson"] * 3)

REGIMES = ["complementary", "full_coverage"]
X_CONDITIONS = ["single_bernoulli", "single_gaussian", "single_poisson", "per_column_all"]
PRIMARY_RATE = 0.1
PRIMARY_COMPARATOR = "single_gaussian"
SECONDARY_COMPARATORS = ["single_bernoulli", "single_poisson"]

POISSON_SAMPLING_PROVENANCE = (
    "common-random-number inverse-CDF coupling, with q=0 replaced only by "
    "nextafter(0,1) for floating-point endpoint safety; no eta/mu clipping")

STEM = "matched_latent_coverage_ablation_20260821"
OUT_DIR = _ROOT / "expfam" / "results" / "story_diagnostics"
FIG_DIR = _ROOT / "figures" / "story_diagnostics"

IDENTITY_TOL = 1e-10


class ExperimentStop(RuntimeError):
    """Integrity violation. The experiment is abandoned, never rescued."""


def _sha16(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=_ROOT).stdout.strip()
    except Exception:
        return "unknown"


def stable_sigmoid(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# ── Generator ─────────────────────────────────────────────────────────────

def _make_F_complementary(rng_F):
    F = np.zeros((D, K_TRUE))
    for bname, (cols, src) in BLOCKS.items():
        e = np.zeros(K_TRUE)
        e[src] = 1.0
        for l in cols:
            v = DOMINANT_WEIGHT * e + MINOR_WEIGHT * rng_F.standard_normal(K_TRUE)
            F[l] = v / np.linalg.norm(v)
    return F


def _roll_to_full_coverage(F_comp):
    """Per-row latent-coordinate permutation. No new random draws, no rescaling."""
    F_full = np.zeros_like(F_comp)
    for bname, (cols, src) in BLOCKS.items():
        for target, l in enumerate(cols):       # target = row index within block
            shift = (target - src) % K_TRUE
            F_full[l] = np.roll(F_comp[l], shift)
    return F_full


def _build_X(eta, U_bern, E_gauss, U_pois):
    """Common-random-number inverse-CDF coupling. Shared variates across both regimes."""
    bc, gc, pc = BLOCKS["bern"][0], BLOCKS["gauss"][0], BLOCKS["pois"][0]
    X = np.zeros((N, D))
    X[:, bc] = (U_bern < stable_sigmoid(eta[:, bc])).astype(float)
    X[:, gc] = eta[:, gc] + SIGMA_G * E_gauss
    mu_p = np.exp(eta[:, pc])                                   # canonical, no clip
    if not np.all(np.isfinite(mu_p)):
        raise ExperimentStop("generator: non-finite Poisson-X mu")
    U_pois_safe = np.where(U_pois == 0.0, np.nextafter(0.0, 1.0), U_pois)
    xp = _spoisson.ppf(U_pois_safe, mu_p)
    if not np.all(np.isfinite(xp)):
        raise ExperimentStop("generator: non-finite Poisson-X sample")
    if not np.all(xp >= 0):
        raise ExperimentStop("generator: negative Poisson-X sample")
    if not np.array_equal(xp, np.floor(xp)):
        raise ExperimentStop("generator: non-integer Poisson-X sample")
    X[:, pc] = xp
    if not np.all(np.isfinite(X)):
        raise ExperimentStop("generator: non-finite X")
    return X, mu_p


def generate(trial):
    """One trial: shared Z/Y/variates, both F regimes. Generator only, no model."""
    master = np.random.SeedSequence(DATA_SEED_BASE + trial)
    children = master.spawn(6)                    # FIXED ORDER, see SPAWN_ORDER
    rng_Z, rng_F, rng_Xb, rng_Xg, rng_Xp, rng_Y = (
        np.random.default_rng(c) for c in children)

    Z_raw = rng_Z.standard_normal((N, K_TRUE))
    Z_true = (Z_raw - Z_raw.mean(0)) / Z_raw.std(0)

    F_comp = _make_F_complementary(rng_F)
    F_full = _roll_to_full_coverage(F_comp)

    U_bern = rng_Xb.random((N, 3))
    E_gauss = rng_Xg.standard_normal((N, 3))
    U_pois = rng_Xp.random((N, 3))

    # The pre-registered Gaussian requirement is that the SAME parameter-free
    # pre-drawn noise source is used for both regimes. That property is checked
    # directly on the source arrays (exact identity + hash), never by subtracting
    # eta back out of a floating-point X.
    E_gauss_regime = {"complementary": E_gauss, "full_coverage": E_gauss.copy()}

    eta = {"complementary": Z_true @ F_comp.T, "full_coverage": Z_true @ F_full.T}
    for r, e in eta.items():
        if not np.all(np.isfinite(e)):
            raise ExperimentStop(f"generator: non-finite eta_x in regime {r}")
    Xs, mus = {}, {}
    for r in REGIMES:
        Xs[r], mus[r] = _build_X(eta[r], U_bern, E_gauss_regime[r], U_pois)

    # ── Y: exact Issue #31 generator, once per trial, shared across regimes ──
    eta_y_full = W0_TRUE + W_TRUE * (Z_true @ Z_true.T)
    iu = np.triu_indices(N, 1)
    mu_y = np.exp(eta_y_full[iu])
    if not np.all(np.isfinite(mu_y)):
        raise ExperimentStop("generator: non-finite Poisson-Y mu")
    Y = np.zeros((N, N))
    Y[iu] = rng_Y.poisson(mu_y)
    Y = Y + Y.T
    if not np.all(np.isfinite(Y)):
        raise ExperimentStop("generator: non-finite Y")
    if not np.all(np.diag(Y) == 0.0):
        raise ExperimentStop("generator: Y diagonal is not zero")
    if not np.array_equal(Y, Y.T):
        raise ExperimentStop("generator: Y is not symmetric")

    C = np.corrcoef(Z_true.T)
    return dict(Z=Z_true, F={"complementary": F_comp, "full_coverage": F_full},
                eta_x=eta, X=Xs, mu_pois_x=mus, Y=Y,
                eta_y_upper=eta_y_full[iu], mu_y=mu_y,
                U_bern=U_bern, E_gauss=E_gauss, E_gauss_regime=E_gauss_regime,
                U_pois=U_pois,
                z_corr={"z_corr_12": float(C[0, 1]), "z_corr_13": float(C[0, 2]),
                        "z_corr_23": float(C[1, 2])})


# ── Generator-side precision spectrum ─────────────────────────────────────

def _true_c(eta, bname):
    cols, _ = BLOCKS[bname]
    e = eta[:, cols]
    if bname == "gauss":
        return np.full(len(cols), 1.0 / SIGMA_G ** 2)
    if bname == "bern":
        p = stable_sigmoid(e)
        return (p * (1.0 - p)).mean(0)
    return np.exp(e).mean(0)


def _spectrum(P):
    lam = np.sort(np.linalg.eigvalsh(P))[::-1]
    lam = np.clip(lam, 0.0, None)
    tr = float(lam.sum())
    tr2 = float(np.sum(lam ** 2))
    return {
        "eig1": float(lam[0]), "eig2": float(lam[1]), "eig3": float(lam[2]),
        "trace": tr, "lambda_min": float(lam[-1]), "lambda_max": float(lam[0]),
        "coverage_index": float(lam[-1] / tr) if tr > 0 else np.nan,
        "effective_rank": float(tr ** 2 / tr2) if tr2 > 0 else np.nan,
    }


def _block_precision(F, eta, bname):
    cols, _ = BLOCKS[bname]
    c = _true_c(eta, bname)
    return sum(c[i] * np.outer(F[l], F[l]) for i, l in enumerate(cols)), c


# ── Phase 4: generator-only pre-fit gate ──────────────────────────────────

def generator_prefit_gate(trials, verbose=True):
    """All 15 Issue #31 criteria over every trial. Raises ExperimentStop on any failure."""
    rows, checks = [], {}

    def rec(name, ok):
        checks[name] = checks.get(name, True) and bool(ok)

    for t in range(trials):
        d = generate(t)
        Fc, Ff = d["F"]["complementary"], d["F"]["full_coverage"]

        # 1 / 2 / 3
        for bname, (cols, src) in BLOCKS.items():
            for target, l in enumerate(cols):
                rec("1_row_norm", np.isclose(np.linalg.norm(Fc[l]), np.linalg.norm(Ff[l]),
                                             rtol=0.0, atol=1e-15))
                rec("2_magnitude_multiset",
                    np.array_equal(np.sort(np.abs(Fc[l])), np.sort(np.abs(Ff[l]))))
                rec("3_target_dim", int(np.argmax(np.abs(Ff[l]))) == target)
                rec("3_source_dim", int(np.argmax(np.abs(Fc[l]))) == src)

        # 10: shared Z / Y / eta_y ; symmetry ; zero diagonal ; upper-triangle only
        h_Z, h_Y, h_etay = _sha16(d["Z"]), _sha16(d["Y"]), _sha16(d["eta_y_upper"])
        rec("10_Y_symmetric", np.array_equal(d["Y"], d["Y"].T))
        rec("10_Y_diag_zero", np.all(np.diag(d["Y"]) == 0.0))

        # 11
        rec("11_finite", all(np.all(np.isfinite(x)) for x in
                             [d["Z"], Fc, Ff, d["Y"], d["mu_y"],
                              d["eta_x"]["complementary"], d["eta_x"]["full_coverage"],
                              d["X"]["complementary"], d["X"]["full_coverage"],
                              d["mu_pois_x"]["complementary"], d["mu_pois_x"]["full_coverage"]]))

        # 14 (Issue #31 as written): inverse-CDF / common-U coupling monotonicity.
        # Applies to Bernoulli and Poisson only. Gaussian is additive-noise coupled and
        # is covered by the separate source-identity provenance check below.
        bc, gc, pc = BLOCKS["bern"][0], BLOCKS["gauss"][0], BLOCKS["pois"][0]
        pc_c = stable_sigmoid(d["eta_x"]["complementary"][:, bc])
        pc_f = stable_sigmoid(d["eta_x"]["full_coverage"][:, bc])
        xb_c = d["X"]["complementary"][:, bc]
        xb_f = d["X"]["full_coverage"][:, bc]
        rec("14_bern_monotone", not np.any((pc_f > pc_c) & (xb_f < xb_c))
            and not np.any((pc_f < pc_c) & (xb_f > xb_c)))
        mp_c, mp_f = d["mu_pois_x"]["complementary"], d["mu_pois_x"]["full_coverage"]
        xp_c = d["X"]["complementary"][:, pc]
        xp_f = d["X"]["full_coverage"][:, pc]
        rec("14_pois_monotone", not np.any((mp_f > mp_c) & (xp_f < xp_c))
            and not np.any((mp_f < mp_c) & (xp_f > xp_c)))

        # Gaussian provenance/integrity: the SAME parameter-free pre-drawn noise source
        # is used for both regimes. Checked directly on the source arrays, never by
        # subtracting eta back out of a floating-point X.
        Ec = d["E_gauss_regime"]["complementary"]
        Ef = d["E_gauss_regime"]["full_coverage"]
        rec("prov_gaussian_common_noise_source_equal", np.array_equal(Ec, Ef))
        rec("prov_gaussian_common_noise_hash_equal", _sha16(Ec) == _sha16(Ef))

        # 15
        for r in REGIMES:
            xp = d["X"][r][:, pc]
            rec("15_pois_support", np.all(np.isfinite(xp)) and np.all(xp >= 0)
                and np.array_equal(xp, np.floor(xp)))

        # 4-9 + 12 + 13: per block, per regime
        gtrace = {}
        for bname in BLOCKS:
            for r in REGIMES:
                P, c = _block_precision(d["F"][r], d["eta_x"][r], bname)
                sp = _spectrum(P)
                if bname == "gauss":
                    gtrace[r] = sp["trace"]
                rows.append({
                    "trial": t, "regime": r, "block": bname,
                    "family": BLOCK_FAMILY[bname],
                    "source_dim": BLOCKS[bname][1] + 1,
                    "true_c_mean": float(np.mean(c)),
                    **sp,
                    "f_norm_mean": float(np.mean([np.linalg.norm(d["F"][r][l])
                                                  for l in BLOCKS[bname][0]])),
                    "eta_min": float(d["eta_x"][r][:, BLOCKS[bname][0]].min()),
                    "eta_max": float(d["eta_x"][r][:, BLOCKS[bname][0]].max()),
                    "x_mean": float(d["X"][r][:, BLOCKS[bname][0]].mean()),
                    "x_var": float(d["X"][r][:, BLOCKS[bname][0]].var()),
                    "hash_Z": h_Z, "hash_Y": h_Y, "hash_eta_y": h_etay,
                    "hash_F": _sha16(d["F"][r]),
                    "hash_U_bern": _sha16(d["U_bern"]),
                    "hash_E_gauss": _sha16(d["E_gauss"]),
                    "hash_E_gauss_regime": _sha16(d["E_gauss_regime"][r]),
                    "gaussian_common_noise_source_equal": bool(np.array_equal(
                        d["E_gauss_regime"]["complementary"],
                        d["E_gauss_regime"]["full_coverage"])),
                    "gaussian_common_noise_hash_equal": bool(
                        _sha16(d["E_gauss_regime"]["complementary"])
                        == _sha16(d["E_gauss_regime"]["full_coverage"])),
                    "hash_U_pois": _sha16(d["U_pois"]),
                    "spawn_order": "|".join(SPAWN_ORDER),
                    "spawn_index": SPAWN_ORDER.index(
                        {"bern": "X_bern", "gauss": "X_gauss",
                         "pois": "X_pois"}[bname]),
                    "poisson_sampling": POISSON_SAMPLING_PROVENANCE,
                    **d["z_corr"],
                })
        # 8
        rec("8_gauss_trace_equal",
            abs(gtrace["complementary"] - gtrace["full_coverage"]) < 1e-10)

    gen = pd.DataFrame(rows)
    # 9: record drift (never rescale)
    drift = {}
    for bname in BLOCKS:
        s = gen[gen.block == bname]
        tc = s[s.regime == "complementary"].trace.mean()
        tf = s[s.regime == "full_coverage"].trace.mean()
        drift[bname] = float((tc - tf) / tf * 100.0)
    rec("9_drift_recorded", all(np.isfinite(v) for v in drift.values()))
    for key in ("4_eigen", "5_trace", "6_coverage_index", "7_effective_rank",
                "12_latent_corr", "13_rng_provenance"):
        rec(key, True)
    rec("4_eigen", gen[["eig1", "eig2", "eig3"]].notna().all().all())
    rec("5_trace", gen["trace"].notna().all())
    rec("6_coverage_index", gen["coverage_index"].notna().all())
    rec("7_effective_rank", gen["effective_rank"].notna().all())
    rec("12_latent_corr", gen[["z_corr_12", "z_corr_13", "z_corr_23"]].notna().all().all())
    rec("13_rng_provenance", gen["spawn_order"].eq("|".join(SPAWN_ORDER)).all())
    # 10 hash equality across regimes within trial
    for t, g in gen.groupby("trial"):
        rec("10_shared_Z_Y_etay",
            g.hash_Z.nunique() == 1 and g.hash_Y.nunique() == 1
            and g.hash_eta_y.nunique() == 1)

    failed = [k for k, v in checks.items() if not v]
    if verbose:
        print("=== PHASE 4 generator-only pre-fit gate ===")
        for k in sorted(checks):
            print(f"  [{'PASS' if checks[k] else 'FAIL'}] {k}")
        print(f"  block trace drift comp vs full (%): "
              + ", ".join(f"{b}={v:+.3f}" for b, v in drift.items()))
    if failed:
        raise ExperimentStop(f"generator-only pre-fit gate FAILED: {failed}")
    return gen, drift


# ── Conditions ────────────────────────────────────────────────────────────

def build_conditions(X):
    conds = {}
    for bname, (cols, _s) in BLOCKS.items():
        fam = BLOCK_FAMILY[bname]
        conds[f"single_{fam}"] = (X[:, cols], cols, dict(family_x=fam, family_x_list=None))
    conds["per_column_all"] = (X, np.arange(D),
                               dict(family_x=None, family_x_list=list(FAM_LIST_TRUE)))
    return conds


def thin_pool_mask(pool_mask, keep_rate, seed):
    if keep_rate >= 1.0:
        return pool_mask.copy()
    n = pool_mask.shape[0]
    rows, cols = upper_pairs_of(pool_mask)
    n_pool = len(rows)
    n_keep = max(1, int(round(n_pool * keep_rate)))
    keep = np.random.default_rng(seed).permutation(n_pool)[:n_keep]
    m = np.zeros((n, n), dtype=bool)
    m[rows[keep], cols[keep]] = True
    m |= m.T
    np.fill_diagonal(m, False)
    return m


# ── Fitted-side precision diagnostics ─────────────────────────────────────

def curvature_c(model, eta_x, sigma):
    """c_il exactly as the model's own X precision term uses it."""
    if getattr(model, "family_x_list", None) is not None:
        return model._variance_function_x(eta_x) * model._x_weight_vector(sigma)[None, :]
    if model.family_x == "gaussian":
        inv = 1.0 / np.maximum(np.diag(sigma), 1e-8)      # variance, NOT squared
        return np.broadcast_to(inv[None, :], eta_x.shape).copy()
    return model._variance_function_x(eta_x)


def precision_diagnostics(res, R, used_cols, is_y_only):
    if is_y_only:
        out = {"t_total": 0.0, "identity_max_abs_err": 0.0}
        for b in BLOCKS:
            out[f"t_{b}"] = 0.0
            out[f"t_{b}_share"] = np.nan
            out[f"fnorm_{b}_mean"] = 0.0
            out[f"eig1_{b}"] = np.nan
            out[f"eig2_{b}"] = np.nan
            out[f"eig3_{b}"] = np.nan
            out[f"trace_{b}"] = 0.0
            out[f"lambda_min_{b}"] = np.nan
            out[f"lambda_max_{b}"] = np.nan
            out[f"coverage_index_{b}"] = np.nan
            out[f"effective_rank_{b}"] = np.nan
            for j in range(K_TRUE):
                out[f"t_{b}_to_dim{j+1}"] = 0.0
        return out

    model, F_est, sigma = res["model"], res["F"], res["sigma"]
    eta_x = res["Z_est"] @ F_est.T
    c = curvature_c(model, eta_x, sigma)
    fn2 = np.sum(F_est ** 2, axis=1)
    t_col = np.mean(c, axis=0) * fn2
    F_al = F_est @ R
    t_cd = np.mean(c, axis=0)[:, None] * (F_al ** 2)
    ident = float(np.max(np.abs(t_cd.sum(axis=1) - t_col))) if len(t_col) else 0.0
    t_total = float(t_col.sum())
    out = {"t_total": t_total, "identity_max_abs_err": ident}
    cbar = np.mean(c, axis=0)
    for bname, (cols, _s) in BLOCKS.items():
        pos = [int(np.where(used_cols == cc)[0][0]) for cc in cols if cc in used_cols]
        if not pos:
            out[f"t_{bname}"] = np.nan
            out[f"t_{bname}_share"] = np.nan
            out[f"fnorm_{bname}_mean"] = np.nan
            for k in ("eig1", "eig2", "eig3", "trace", "lambda_min", "lambda_max",
                      "coverage_index", "effective_rank"):
                out[f"{k}_{bname}"] = np.nan
            for j in range(K_TRUE):
                out[f"t_{bname}_to_dim{j+1}"] = np.nan
            continue
        tb = float(t_col[pos].sum())
        out[f"t_{bname}"] = tb
        out[f"t_{bname}_share"] = tb / t_total if t_total > 0 else np.nan
        out[f"fnorm_{bname}_mean"] = float(np.mean(np.sqrt(fn2[pos])))
        P = sum(cbar[p] * np.outer(F_est[p], F_est[p]) for p in pos)
        sp = _spectrum(P)
        for k, v in sp.items():
            out[f"{k}_{bname}"] = v
        for j in range(K_TRUE):
            out[f"t_{bname}_to_dim{j+1}"] = float(t_cd[pos, j].sum())
    return out


# ── One instrumented fit ──────────────────────────────────────────────────

def run_one_fit(X_used, Y, train_mask, seed, kw, key):
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            with contextlib.redirect_stdout(buf):
                res = run_em_experimental(
                    X_used, Y, family_y="poisson", k=K_TRUE, L=L, num_iter=NITER,
                    seed=seed, train_mask=train_mask,
                    numerics_mode="consistent", verbose=True,
                    validate_support=True, allow_support_mismatch=False,
                    compute_clip_diagnostic=True, **kw)
        except Exception as exc:
            raise ExperimentStop(
                f"{key}: fit raised {type(exc).__name__}: {exc}\n"
                f"stdout tail:\n{buf.getvalue()[-800:]}") from exc
        wrecs = [{"category": w.category.__name__, "message": str(w.message),
                  "filename": str(w.filename), "lineno": int(w.lineno)} for w in wlist]
    captured = buf.getvalue()
    return res, captured, ("[NaN iter=" in captured), wrecs


def check_fit(key, res, retry_flag, captured, wrecs):
    if retry_flag:
        raise ExperimentStop(
            f"{key}: internal NaN reset / retry detected "
            f"(nan_occurred={res['nan_occurred']}, nan_count={res['nan_count']}). "
            "Result NOT used.\n"
            + "\n".join(ln for ln in captured.splitlines() if "[NaN iter=" in ln))
    if res["nan_occurred"]:
        raise ExperimentStop(f"{key}: nan_occurred=True (nan_count={res['nan_count']})")
    if res.get("q_bic_failed"):
        raise ExperimentStop(
            f"{key}: q_bic_failed=True, failure_reason={res.get('failure_reason')!r}")
    if res.get("numerics_mode") != "consistent":
        raise ExperimentStop(f"{key}: numerics_mode={res.get('numerics_mode')!r}")
    cd = res.get("clip_diag") or {}
    if cd.get("status") != "not_applicable":
        raise ExperimentStop(f"{key}: unexpected clip_diag={cd!r}")
    if wrecs:
        print(f"    [warning x{len(wrecs)}] {key}: "
              f"{wrecs[0]['category']}: {wrecs[0]['message'][:120]}")


# ── Main ──────────────────────────────────────────────────────────────────

def main(smoke, out_dir, fig_dir):
    t0 = time.perf_counter()
    trials = TRIALS_SMOKE if smoke else TRIALS_FULL
    tag = f"{STEM}_smoke" if smoke else STEM
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_df, trace_drift = generator_prefit_gate(trials)

    rows, block_rows, warn_rows = [], [], []
    seen = set()

    for trial in range(trials):
        data = generate(trial)
        pool_mask, test_mask = make_pair_split(
            N, TEST_RATIO, seed=SPLIT_SEED_BASE + trial * 100)
        h_test = _sha16(test_mask)
        h_Z, h_Y = _sha16(data["Z"]), _sha16(data["Y"])
        model_seed = MODEL_SEED_BASE + trial * 10

        for rate_idx, rate in enumerate(Y_OBS_RATES):
            train_mask = thin_pool_mask(
                pool_mask, rate, seed=THIN_SEED_BASE + trial * 100 + rate_idx)
            h_train = _sha16(train_mask)
            n_train_pairs = int(np.triu(train_mask, k=1).sum())

            fits = [(r, c) for r in REGIMES for c in X_CONDITIONS]
            fits.append(("shared", "y_only"))
            for regime, cname in fits:
                key = (trial, rate, regime, cname)
                if key in seen:
                    raise ExperimentStop(f"duplicate key {key}")
                seen.add(key)

                if cname == "y_only":
                    # regime-independent: F=0 blocks X entirely.
                    # deterministic single X array = complementary-regime X (recorded).
                    X_used = data["X"]["complementary"]
                    used_cols = np.array([], dtype=int)
                    kw = dict(family_x="gaussian", family_x_list=None, fix_x=True)
                else:
                    X_used, used_cols, kw = build_conditions(data["X"][regime])[cname]

                res, cap, retry_flag, wrecs = run_one_fit(
                    X_used, data["Y"], train_mask, model_seed, kw, str(key))
                check_fit(str(key), res, retry_flag, cap, wrecs)
                for w in wrecs:
                    warn_rows.append({"trial": trial, "y_obs_rate": rate,
                                      "regime": regime, "condition": cname, **w})

                R, k_min = procrustes_rotation(res["Z_est"], data["Z"])
                if k_min != K_TRUE:
                    raise ExperimentStop(f"{key}: k_min={k_min} != {K_TRUE}")
                Z_al = res["Z_est"][:, :k_min] @ R
                rmse_Z = calc_rmse(data["Z"][:, :k_min], Z_al)
                dim_rmse = [float(np.sqrt(np.mean((data["Z"][:, j] - Z_al[:, j]) ** 2)))
                            for j in range(k_min)]

                mu_y = predict_mu_y(res)
                if not np.all(np.isfinite(mu_y)):
                    raise ExperimentStop(f"{key}: non-finite mu_y")
                m_tr = heldout_count_metrics(data["Y"], mu_y, train_mask, "poisson")
                m_te = heldout_count_metrics(data["Y"], mu_y, test_mask, "poisson")

                pd_ = precision_diagnostics(res, R, used_cols, is_y_only=(cname == "y_only"))
                if pd_["identity_max_abs_err"] > 1e-8:
                    raise ExperimentStop(
                        f"{key}: precision identity failure "
                        f"({pd_['identity_max_abs_err']:.3e})")

                row = {
                    "trial": trial, "y_obs_rate": rate, "regime": regime,
                    "condition": cname, "n_cols_used": int(len(used_cols)),
                    "n_train_pairs": n_train_pairs, "rmse_Z": rmse_Z,
                    "rmse_z_dim1": dim_rmse[0], "rmse_z_dim2": dim_rmse[1],
                    "rmse_z_dim3": dim_rmse[2],
                    "test_y_ll": m_te.get("mean_ll", np.nan),
                    "test_y_rmse": m_te.get("rmse", np.nan),
                    "train_y_ll": m_tr.get("mean_ll", np.nan),
                    "w0_err": abs(res["w0"] - W0_TRUE),
                    "w_err": abs(res["w"] - W_TRUE),
                    "w0_est": res["w0"], "w_est": res["w"],
                    "bic_diagnostic_only": res["bic"],
                    "q_strict_diagnostic_only": res["Q_strict"],
                    "numerics_mode": res["numerics_mode"],
                    "internal_retry_detected": retry_flag,
                    "nan_occurred": res["nan_occurred"], "nan_count": res["nan_count"],
                    "q_bic_failed": res["q_bic_failed"],
                    "failure_reason": res["failure_reason"],
                    "n_warnings": len(wrecs),
                    "clip_diag_status": (res.get("clip_diag") or {}).get("status"),
                    "hash_Z": h_Z, "hash_Y": h_Y, "hash_test_mask": h_test,
                    "hash_train_mask": h_train, "model_seed": model_seed,
                    "runtime_s": res["runtime_s"],
                }
                for k, v in pd_.items():
                    row[f"prec_{k}"] = v
                for mname in ("rmse_Z", "rmse_z_dim1", "rmse_z_dim2", "rmse_z_dim3",
                              "test_y_ll", "test_y_rmse", "train_y_ll"):
                    if not np.isfinite(row[mname]):
                        raise ExperimentStop(f"{key}: non-finite metric {mname}")
                rows.append(row)

                for bname in BLOCKS:
                    br = {"trial": trial, "y_obs_rate": rate, "regime": regime,
                          "condition": cname, "block": bname,
                          "t_block": pd_[f"t_{bname}"],
                          "t_block_share": pd_[f"t_{bname}_share"],
                          "fnorm_mean": pd_[f"fnorm_{bname}_mean"],
                          "trace": pd_[f"trace_{bname}"],
                          "lambda_min": pd_[f"lambda_min_{bname}"],
                          "lambda_max": pd_[f"lambda_max_{bname}"],
                          "eig1": pd_[f"eig1_{bname}"], "eig2": pd_[f"eig2_{bname}"],
                          "eig3": pd_[f"eig3_{bname}"],
                          "coverage_index": pd_[f"coverage_index_{bname}"],
                          "effective_rank": pd_[f"effective_rank_{bname}"],
                          "t_total": pd_["t_total"]}
                    for j in range(K_TRUE):
                        br[f"to_dim{j+1}"] = pd_[f"t_{bname}_to_dim{j+1}"]
                    block_rows.append(br)

                print(f"t={trial} rate={rate:.1f} {regime:14s} {cname:17s} "
                      f"rmse_Z={rmse_Z:.4f} te_ll={row['test_y_ll']:.4f}")

    df = pd.DataFrame(rows)
    expected = trials * len(Y_OBS_RATES) * (len(REGIMES) * len(X_CONDITIONS) + 1)
    if len(df) != expected:
        raise ExperimentStop(f"fit count {len(df)} != expected {expected}")
    if df["internal_retry_detected"].any() or df["nan_occurred"].any() \
            or df["q_bic_failed"].any():
        raise ExperimentStop("integrity flags set after the loop")
    if not (df["numerics_mode"] == "consistent").all():
        raise ExperimentStop("not all fits ran in consistent mode")
    for tr, g in df.groupby("trial"):
        if g.hash_Z.nunique() != 1 or g.hash_Y.nunique() != 1 \
                or g.hash_test_mask.nunique() != 1:
            raise ExperimentStop(f"trial {tr}: Z/Y/test hash mismatch")
        for rate in Y_OBS_RATES:
            if g[g.y_obs_rate == rate].hash_train_mask.nunique() != 1:
                raise ExperimentStop(f"trial {tr} rate {rate}: train hash mismatch")
    cnt = df[df.condition != "y_only"].groupby(["regime", "y_obs_rate", "condition"]).size()
    if not (cnt == trials).all():
        raise ExperimentStop(f"per-cell counts not all {trials}:\n{cnt}")
    cnt_y = df[df.condition == "y_only"].groupby(["y_obs_rate"]).size()
    if not (cnt_y == trials).all():
        raise ExperimentStop(f"y_only counts not all {trials}:\n{cnt_y}")

    df.to_csv(out_dir / f"{tag}_summary.csv", index=False)
    gen_df.to_csv(out_dir / f"{tag}_generator.csv", index=False)
    pd.DataFrame(block_rows).to_csv(out_dir / f"{tag}_blockdiag.csv", index=False)

    agg = df.groupby(["regime", "condition", "y_obs_rate"]).agg(
        n_trials=("rmse_Z", "count"), n_train_pairs_mean=("n_train_pairs", "mean"),
        rmse_Z_mean=("rmse_Z", "mean"), rmse_Z_std=("rmse_Z", "std"),
        rmse_z_dim1_mean=("rmse_z_dim1", "mean"),
        rmse_z_dim2_mean=("rmse_z_dim2", "mean"),
        rmse_z_dim3_mean=("rmse_z_dim3", "mean"),
        test_y_ll_mean=("test_y_ll", "mean"), test_y_ll_std=("test_y_ll", "std"),
        test_y_rmse_mean=("test_y_rmse", "mean"), train_y_ll_mean=("train_y_ll", "mean"),
        w0_err_mean=("w0_err", "mean"), w_err_mean=("w_err", "mean"),
        runtime_s_mean=("runtime_s", "mean"), n_nan=("nan_occurred", "sum"),
        n_retry=("internal_retry_detected", "sum"),
        n_qbic_failed=("q_bic_failed", "sum")).reset_index()
    agg.to_csv(out_dir / f"{tag}_agg.csv", index=False)

    # ── paired + interaction ────────────────────────────────────────────
    piv = df.pivot_table(index=["y_obs_rate", "trial"], columns=["regime", "condition"],
                         values=["rmse_Z", "test_y_ll", "rmse_z_dim1", "rmse_z_dim2",
                                 "rmse_z_dim3"])
    paired_rows, inter_rows = [], []
    identity_errs = []
    for rate in Y_OBS_RATES:
        sub = piv.loc[rate]
        J = {r: sub[("rmse_Z", r, "per_column_all")] for r in REGIMES}
        pairs = [(r, c) for r in REGIMES for c in X_CONDITIONS if c != "per_column_all"]
        pairs += [(r, "y_only") for r in REGIMES]   # shared baseline vs each regime's joint
        for r, comp in pairs:
            if comp == "y_only":
                G = sub[("rmse_Z", "shared", "y_only")]
                Gll = sub[("test_y_ll", "shared", "y_only")]
            else:
                G = sub[("rmse_Z", r, comp)]
                Gll = sub[("test_y_ll", r, comp)]
            dz = G - J[r]
            dl = sub[("test_y_ll", r, "per_column_all")] - Gll
            paired_rows.append({
                    "y_obs_rate": rate, "regime": r,
                    "comparator": comp + ("_shared_baseline" if comp == "y_only" else ""),
                    "n_trials": int(dz.notna().sum()),
                    "delta_rmse_mean": float(dz.mean()), "delta_rmse_std": float(dz.std()),
                    "delta_rmse_median": float(dz.median()),
                    "n_favoring_per_column_rmse": int((dz > 0).sum()),
                    "delta_test_ll_mean": float(dl.mean()),
                    "n_favoring_per_column_ll": int((dl > 0).sum()),
                })
        for comp in [PRIMARY_COMPARATOR] + SECONDARY_COMPARATORS:
            Gc = sub[("rmse_Z", "complementary", comp)]
            Gf = sub[("rmse_Z", "full_coverage", comp)]
            Jc, Jf = J["complementary"], J["full_coverage"]
            dG_c, dG_f = Gc - Jc, Gf - Jf
            I = dG_c - dG_f
            D_G, D_J = Gc - Gf, Jc - Jf
            err = float(np.max(np.abs(I - (D_G - D_J))))
            identity_errs.append(err)
            if err > IDENTITY_TOL:
                raise ExperimentStop(
                    f"identity I != D_G - D_J for {comp} at rate {rate}: max err {err:.3e}")
            for t in sub.index:
                inter_rows.append({
                    "y_obs_rate": rate, "comparator": comp, "trial": int(t),
                    "is_primary": bool(rate == PRIMARY_RATE and comp == PRIMARY_COMPARATOR),
                    "rmse_comparator_complementary": float(Gc[t]),
                    "rmse_comparator_fullcoverage": float(Gf[t]),
                    "rmse_joint_complementary": float(Jc[t]),
                    "rmse_joint_fullcoverage": float(Jf[t]),
                    "delta_G_complementary": float(dG_c[t]),
                    "delta_G_fullcoverage": float(dG_f[t]),
                    "I": float(I[t]), "D_G": float(D_G[t]), "D_J": float(D_J[t]),
                    "identity_abs_err": float(abs(I[t] - (D_G[t] - D_J[t]))),
                })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out_dir / f"{tag}_paired.csv", index=False)
    inter = pd.DataFrame(inter_rows)
    inter.to_csv(out_dir / f"{tag}_interaction.csv", index=False)
    identity_max_err = float(max(identity_errs)) if identity_errs else 0.0

    if warn_rows:
        pd.DataFrame(warn_rows).to_csv(out_dir / f"{tag}_warnings.csv", index=False)

    gtr = gen_df[gen_df.block == "gauss"].groupby(["trial", "regime"]).trace.first().unstack()
    gauss_trace_err = float((gtr["complementary"] - gtr["full_coverage"]).abs().max())

    runinfo = [{
        "script": "tools/research_audit/run_matched_latent_coverage_ablation.py",
        "issue": 31, "datetime": datetime.now().isoformat(timespec="seconds"),
        "git_head": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
        "mode": "smoke" if smoke else "full",
        "n": N, "d": D, "k_true": K_TRUE, "trials": trials, "L": L, "num_iter": NITER,
        "test_ratio": TEST_RATIO, "y_obs_rates": str(Y_OBS_RATES),
        "regimes": str(REGIMES), "x_conditions": str(X_CONDITIONS),
        "y_only_policy": ("regime-independent shared baseline, fit once per (trial, rate); "
                          "deterministic X array passed = complementary-regime X "
                          "(irrelevant because fix_x=True holds F=0)"),
        "all_gaussian": "excluded (optimizer-path confound established in Issue #27)",
        "w0_true": W0_TRUE, "w_true": W_TRUE, "sigma_g": SIGMA_G,
        "dominant_weight": DOMINANT_WEIGHT, "minor_weight": MINOR_WEIGHT,
        "block_source_dim": "bernoulli->z1, gaussian->z2, poisson->z3 (complementary)",
        "full_coverage_construction": ("per-row np.roll of F_complementary with "
                                       "shift=(target_dim-source_dim)%K_TRUE; no resampling, "
                                       "no rescaling"),
        "fam_list_true": str(FAM_LIST_TRUE),
        "data_seed_base": DATA_SEED_BASE, "model_seed_base": MODEL_SEED_BASE,
        "split_seed_base": SPLIT_SEED_BASE, "thin_seed_base": THIN_SEED_BASE,
        "spawn_policy": ("np.random.SeedSequence(DATA_SEED_BASE+trial).spawn(6), fixed order "
                         + "|".join(SPAWN_ORDER)),
        "poisson_sampling": POISSON_SAMPLING_PROVENANCE,
        "gaussian_sampling": ("X_gauss = eta_gauss + SIGMA_G * E_gauss with the SAME "
                              "parameter-free pre-drawn E_gauss shared across both regimes; "
                              "verified by exact source-array identity and hash equality, "
                              "not by subtracting eta back out of a floating-point X"),
        "validator_correction_note": (
            "During the pre-fit generator gate an implementation-only validator incorrectly "
            "required bitwise equality of Gaussian residuals reconstructed as X - eta across "
            "regimes. No model fit had been run. Human review determined that this predicate "
            "was not part of the pre-registered Issue #31 specification and is invalid under "
            "floating-point rounding (fl(fl(eta+noise)-eta) need not equal noise; observed "
            "discrepancy 2.220446e-16 = 1 ULP). It was replaced by a direct exact "
            "identity/hash check of the shared pre-drawn Gaussian noise array E_gauss. "
            "No seed, parameter, generator distribution, endpoint, model fit, or scientific "
            "result was changed. This was not a scientific failure."),
        "bitwise_reproduction_of_issue27": "NOT claimed (different sampling path)",
        "numerics_mode": "consistent", "generator_clipping": "none",
        "z_true_definition": "column-wise standardized N(0,I) draw",
        "n_fits": len(df), "n_internal_retry": int(df["internal_retry_detected"].sum()),
        "n_nan": int(df["nan_occurred"].sum()),
        "n_q_bic_failed": int(df["q_bic_failed"].sum()),
        "n_warnings": int(df["n_warnings"].sum()),
        "primary_domain": f"y_obs_rate={PRIMARY_RATE}",
        "primary_endpoint": "whole-space Procrustes RMSE_Z",
        "primary_comparator": PRIMARY_COMPARATOR,
        "primary_estimand": ("I_t = delta_G(complementary,t) - delta_G(full_coverage,t), "
                             "delta_G(r,t) = RMSE_Z(single_gaussian) - RMSE_Z(per_column_all)"),
        "decomposition": "D_G = G_comp - G_full ; D_J = J_comp - J_full ; I = D_G - D_J",
        "identity_max_abs_err": identity_max_err,
        "effective_rank_definition": "(trace(P))^2 / trace(P^2) on Pbar_b (PSD block)",
        "coverage_index_definition": "lambda_min(Pbar_b) / trace(Pbar_b)",
        "diagnostic_bounds": "for a 3-row block coverage_index <= 1/3 and effective_rank <= 3",
        "gauss_true_trace_max_abs_err_comp_vs_full": gauss_trace_err,
        "block_trace_drift_pct_comp_vs_full": str({k: round(v, 4)
                                                   for k, v in trace_drift.items()}),
        "limitation_not_isolated": ("more tightly targets latent-coverage / block-rank "
                                    "geometry while holding the major previously identified "
                                    "factors fixed; NOT isolated alone"),
        "total_runtime_s": round(time.perf_counter() - t0, 1),
    }]
    pd.DataFrame(runinfo).to_csv(out_dir / f"{tag}_runinfo.csv", index=False)

    print("\n=== PRIMARY (y_obs_rate=0.1, comparator=single_gaussian) ===")
    p = inter[(inter.y_obs_rate == PRIMARY_RATE)
              & (inter.comparator == PRIMARY_COMPARATOR)]
    for col in ("delta_G_complementary", "delta_G_fullcoverage", "I", "D_G", "D_J"):
        v = p[col]
        print(f"  {col:24s} mean={v.mean():+.4f} std={v.std():.4f} "
              f"median={v.median():+.4f} pos={int((v > 0).sum())}/{len(v)}")
    print(f"  identity max abs err = {identity_max_err:.3e}")
    print(f"  gaussian true trace comp-vs-full max abs err = {gauss_trace_err:.3e}")

    if not smoke:
        make_figures(agg, inter, gen_df, fig_dir)
    print(f"\nfits={len(df)}  total {(time.perf_counter() - t0)/60:.1f} min")
    return df, agg, paired, inter, gen_df


# ── Figures ───────────────────────────────────────────────────────────────

COLOR = {"y_only": "#888888", "single_bernoulli": "#8172B2",
         "single_gaussian": "#4C72B0", "single_poisson": "#CCB974",
         "per_column_all": "#C44E52"}


def make_figures(agg, inter, gen_df, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, regime in zip(axes, REGIMES):
        for c in X_CONDITIONS + ["y_only"]:
            r = "shared" if c == "y_only" else regime
            s = agg[(agg.regime == r) & (agg.condition == c)].sort_values("y_obs_rate")
            if s.empty:
                continue
            ax.plot(s["y_obs_rate"], s["rmse_Z_mean"], marker="o", label=c, color=COLOR[c])
        ax.set_xlabel("y_obs_rate")
        ax.set_title(regime, fontsize=10)
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("RMSE_Z after Procrustes (lower is better)")
    axes[1].legend(fontsize=7)
    fig.suptitle("Matched latent-coverage ablation (10 trials, consistent numerics)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{STEM}_rmse_z.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, rate in zip(axes, Y_OBS_RATES):
        p = inter[(inter.y_obs_rate == rate) & (inter.comparator == PRIMARY_COMPARATOR)]
        x = np.arange(len(p))
        ax.bar(x - 0.27, p["D_G"], 0.27, label="D_G (comparator shift)", color="#4C72B0")
        ax.bar(x, p["D_J"], 0.27, label="D_J (joint shift)", color="#C44E52")
        ax.bar(x + 0.27, p["I"], 0.27, label="I = D_G - D_J", color="#55A868")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(p["trial"].astype(int), fontsize=7)
        ax.set_xlabel("trial")
        ax.set_title(f"y_obs_rate = {rate}"
                     + (" (PRIMARY)" if rate == PRIMARY_RATE else " (dense control)"),
                     fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("RMSE_Z difference")
    axes[1].legend(fontsize=7)
    fig.suptitle("Primary interaction and its mandatory decomposition "
                 "(comparator = single_gaussian)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{STEM}_interaction.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    width, xs = 0.16, np.arange(K_TRUE)
    dl = ["dim1 (bern source)", "dim2 (gauss source)", "dim3 (pois source)"]
    for ax, rate in zip(axes, Y_OBS_RATES):
        i = 0
        for regime in REGIMES:
            for c in ("single_gaussian", "per_column_all"):
                s = agg[(agg.regime == regime) & (agg.condition == c)
                        & (agg.y_obs_rate == rate)]
                if s.empty:
                    continue
                vals = [float(s[f"rmse_z_dim{j+1}_mean"].iloc[0]) for j in range(K_TRUE)]
                ax.bar(xs + (i - 1.5) * width, vals, width,
                       label=f"{regime[:4]}/{c}", alpha=0.9)
                i += 1
        ax.set_xticks(xs)
        ax.set_xticklabels(dl, fontsize=7)
        ax.set_title(f"y_obs_rate = {rate}", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("dimension-wise RMSE (shared rotation)")
    axes[1].legend(fontsize=7)
    fig.suptitle("Dimension-wise Z recovery (mechanism diagnostic, one shared rotation)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{STEM}_dimwise_rmse.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    g = gen_df.groupby(["block", "regime"])[["coverage_index", "effective_rank"]].mean()
    blocks = ["bern", "gauss", "pois"]
    xs = np.arange(len(blocks))
    for ax, metric, bound, lab in (
            (axes[0], "coverage_index", 1 / 3, "coverage_index (bound 1/3)"),
            (axes[1], "effective_rank", 3.0, "effective_rank (bound 3)")):
        for i, regime in enumerate(REGIMES):
            vals = [g.loc[(b, regime), metric] for b in blocks]
            ax.bar(xs + (i - 0.5) * 0.35, vals, 0.35, label=regime)
        ax.axhline(bound, color="k", ls="--", lw=0.9, label="upper bound")
        ax.set_xticks(xs)
        ax.set_xticklabels(blocks)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=7)
    fig.suptitle("TRUE generator X-side local-precision orientation by block "
                 "(not posterior information)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{STEM}_coverage_spectrum.png", dpi=150)
    plt.close(fig)
    for f in ("rmse_z", "interaction", "dimwise_rmse", "coverage_spectrum"):
        print(f"saved {fig_dir / f'{STEM}_{f}.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    od = Path(a.out_dir) if a.out_dir else OUT_DIR
    try:
        main(a.smoke, od, FIG_DIR)
    except ExperimentStop as exc:
        print("\n" + "=" * 70)
        print("EXPERIMENT STOPPED - integrity violation, result NOT used")
        print("=" * 70)
        print(exc)
        sys.exit(2)
