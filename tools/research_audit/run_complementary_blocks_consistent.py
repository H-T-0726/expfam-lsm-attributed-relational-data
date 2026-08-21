"""
Complementary mixed-X blocks — objective-consistent validation experiment (Issue #27).

Research question
-----------------
Gaussian / Bernoulli / Poisson の属性 block が潜在構造 Z の *異なる方向* の情報を持つとき、
family ごとに別々に fit するより per-column family で全属性を 1 つの shared Z へ joint
integration することで Z recovery / held-out Y prediction が改善する条件はあるか。
特に Y が疎なときに価値が現れ、Y が dense なら差が縮むか。

Outcome A-E（joint 強く優位 / sparse-Y のみ優位 / 差なし / single 優位 / 不安定）はすべて
valid である。結果を見た後に generator・seed・weight・rate・metric・条件を変更しない。

Fixed design (pre-registered, DO NOT CHANGE AFTER SEEING RESULTS)
----------------------------------------------------------------
N=80, D=9, K_TRUE=3, L=5, num_iter=8, trials=10, test_ratio=0.2
y_obs_rate in {1.0 (dense negative control), 0.1 (sparse primary)}
conditions: y_only / single_bernoulli / single_gaussian / single_poisson
            / per_column_all / all_gaussian
sigma_G=0.3, dominant_weight=0.9, minor_weight=0.15, w0_true=1.2, w_true=0.3
Z_true = column-wise standardized N(0,I) draw; generator clipping: NONE
all fits: numerics_mode="consistent"

primary domain    : y_obs_rate = 0.1
primary endpoint  : whole-space Procrustes RMSE_Z
primary contrasts : per_column_all vs {single_bernoulli, single_gaussian,
                    single_poisson, y_only}
delta_rmse        : comparator_rmse - per_column_rmse  (positive = per_column better)

Mandatory limitations (must be repeated in the report and PR)
-------------------------------------------------------------
1. K_TRUE=3 here vs k*=2 in the existing sparse-Y evidence. This experiment differs from
   that evidence in TWO respects (complementary F structure AND latent dimension), so an
   outcome difference must not be attributed to complementarity alone.
2. The per-observation local curvature A''(eta)/phi differs across blocks by up to ~54x.
   This is NOT a pure family effect: the Gaussian value depends on the pre-registered
   sigma_G = 0.3 (A''/phi = 1/sigma^2) and the Bernoulli/Poisson values depend on the eta
   distribution. Describe it as a
   "family / dispersion / link-induced local-curvature imbalance under this pre-registered
   generator". It is measured and retained, not removed.
3. all_gaussian vs per_column_all is a
   "same-column misspecification contrast (family specification + M-step optimizer path
   confounded)": all-Gaussian uses the analytical closed-form F update while the mixed
   per-column model uses weighted Adam. It is NOT a pure family-assignment effect.
4. single_* vs per_column_all differs in the number of observed X columns as well as in
   family integration. That is part of the research question, but it is not a pure
   family-assignment contrast.
5. The Poisson X marginal variance exceeds its mean purely because of latent
   heterogeneity (Var(X) = E[mu(Z)] + Var(mu(Z)) for a conditionally Poisson generator).
   Do NOT call this "Poisson overdispersion".
6. The generator is deliberately constructed so that blocks are complementary; external
   validity is limited. 10 trials -> report effect sizes only, no significance claims.

Integrity policy
----------------
Any of the following STOPS the whole experiment; the affected fit is never used and no
seed is changed, dropped, or retried:
  internal_retry_detected / nan_occurred / q_bic_failed / support violation /
  non-finite metric / FloatingPointError / unexpected exception /
  numerics_mode != "consistent" / wrong fit count / duplicate key / hash mismatch.

`run_em_experimental` retries internally with a different seed and a halved Newton alpha
and resets `nan_count` per retry, so `res["nan_occurred"]` can be False even after a NaN
reset actually happened. We therefore capture the runner's stdout with verbose=True and
detect the literal "[NaN iter=" it prints on every reset.

Run
---
    python tools/research_audit/run_complementary_blocks_consistent.py --smoke --out-dir DIR
    python tools/research_audit/run_complementary_blocks_consistent.py
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

# ── Fixed design ──────────────────────────────────────────────────────────
N, D, K_TRUE = 80, 9, 3
W0_TRUE, W_TRUE = 1.2, 0.3
L, NITER = 5, 8
TEST_RATIO = 0.2
TRIALS_FULL, TRIALS_SMOKE = 10, 1
Y_OBS_RATES = [1.0, 0.1]
SIGMA_G = 0.3
DOMINANT_WEIGHT, MINOR_WEIGHT = 0.9, 0.15

DATA_SEED_BASE = 110000
MODEL_SEED_BASE = 111000
SPLIT_SEED_BASE = 112000
THIN_SEED_BASE = 113000

# true block -> (columns, intended dominant latent dimension index)
BLOCKS = {"bern": (np.arange(0, 3), 0),
          "gauss": (np.arange(3, 6), 1),
          "pois": (np.arange(6, 9), 2)}
BLOCK_FAMILY = {"bern": "bernoulli", "gauss": "gaussian", "pois": "poisson"}
FAM_LIST_TRUE = (["bernoulli"] * 3) + (["gaussian"] * 3) + (["poisson"] * 3)

CONDITIONS = ["y_only", "single_bernoulli", "single_gaussian", "single_poisson",
              "per_column_all", "all_gaussian"]
PRIMARY_RATE = 0.1
PRIMARY_COMPARATORS = ["single_bernoulli", "single_gaussian", "single_poisson", "y_only"]

STEM = "complementary_blocks_consistent_20260821"
OUT_DIR = _ROOT / "expfam" / "results" / "story_diagnostics"
FIG_DIR = _ROOT / "figures" / "story_diagnostics"

CONTRAST_LABEL = {
    "single_bernoulli": "joint_vs_single_block (bernoulli; column-count differs)",
    "single_gaussian": "joint_vs_single_block (gaussian; column-count differs)",
    "single_poisson": "joint_vs_single_block (poisson; column-count differs)",
    "y_only": "joint_vs_y_only",
    "all_gaussian": ("same_column_misspecification_contrast"
                     "_family_spec_and_mstep_optimizer_path_confounded"),
}


class ExperimentStop(RuntimeError):
    """Raised on any integrity violation. The experiment is abandoned, never rescued."""


def _sha16(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=_ROOT).stdout.strip()
    except Exception:
        return "unknown"


def _git_branch():
    try:
        return subprocess.run(["git", "branch", "--show-current"], capture_output=True,
                              text=True, cwd=_ROOT).stdout.strip()
    except Exception:
        return "unknown"


def _stable_sigmoid(x):
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# ── Generator ─────────────────────────────────────────────────────────────

def generate(seed):
    """Complementary-block generator. NO clipping anywhere."""
    rng = np.random.default_rng(seed)

    Z = rng.standard_normal((N, K_TRUE))
    Z = (Z - Z.mean(0)) / Z.std(0)          # truth = standardized Z

    F = np.zeros((D, K_TRUE))
    for bname, (cols, dim) in BLOCKS.items():
        e = np.zeros(K_TRUE)
        e[dim] = 1.0
        for l in cols:
            v = DOMINANT_WEIGHT * e + MINOR_WEIGHT * rng.standard_normal(K_TRUE)
            F[l] = v / np.linalg.norm(v)

    eta = Z @ F.T
    if not np.all(np.isfinite(eta)):
        raise ExperimentStop("generator: non-finite eta_x")

    X = np.zeros((N, D))
    bc, gc, pc = BLOCKS["bern"][0], BLOCKS["gauss"][0], BLOCKS["pois"][0]
    X[:, bc] = rng.binomial(1, _stable_sigmoid(eta[:, bc]))
    X[:, gc] = eta[:, gc] + rng.normal(0.0, SIGMA_G, (N, len(gc)))
    mu_pois_x = np.exp(eta[:, pc])                       # canonical, no clip
    if not np.all(np.isfinite(mu_pois_x)):
        raise ExperimentStop("generator: non-finite Poisson-X mu")
    X[:, pc] = rng.poisson(mu_pois_x)

    eta_y_full = W0_TRUE + W_TRUE * (Z @ Z.T)
    iu = np.triu_indices(N, 1)
    mu_y = np.exp(eta_y_full[iu])                        # canonical, no clip
    if not np.all(np.isfinite(mu_y)):
        raise ExperimentStop("generator: non-finite Poisson-Y mu")
    Y = np.zeros((N, N))
    Y[iu] = rng.poisson(mu_y)
    Y = Y + Y.T

    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        raise ExperimentStop("generator: non-finite X or Y")

    C = np.corrcoef(Z.T)
    z_corr = {"z_corr_12": float(C[0, 1]), "z_corr_13": float(C[0, 2]),
              "z_corr_23": float(C[1, 2])}
    return dict(X=X, Y=Y, Z=Z, F=F, eta_x=eta,
                eta_y_upper=eta_y_full[iu], z_corr=z_corr)


def generator_diagnostics(trial, data):
    """Per (trial, column) generator diagnostics, including TRUE precision traces."""
    Z, F, X, eta = data["Z"], data["F"], data["X"], data["eta_x"]
    rows = []
    for bname, (cols, dim) in BLOCKS.items():
        fam = BLOCK_FAMILY[bname]
        for l in cols:
            f = F[l]
            fn2 = float(np.sum(f ** 2))
            e = eta[:, l]
            x = X[:, l]
            if fam == "gaussian":
                c_true = np.full(N, 1.0 / SIGMA_G ** 2)
            elif fam == "bernoulli":
                p = _stable_sigmoid(e)
                c_true = p * (1.0 - p)
            else:
                c_true = np.exp(e)
            c_mean = float(np.mean(c_true))
            row = {
                "trial": trial, "col": int(l), "block": bname, "family": fam,
                "intended_dominant_dim": dim + 1,
                "f_1": float(f[0]), "f_2": float(f[1]), "f_3": float(f[2]),
                "f_norm": float(np.sqrt(fn2)),
                "dominant_loading_abs": float(abs(f[dim])),
                "off_dim_norm": float(np.sqrt(max(fn2 - f[dim] ** 2, 0.0))),
                "dominant_share_of_sq_norm": float(f[dim] ** 2 / fn2),
                "eta_min": float(e.min()), "eta_max": float(e.max()),
                "eta_mean": float(e.mean()), "eta_var": float(e.var()),
                "x_mean": float(x.mean()), "x_var": float(x.var()),
                "true_c_mean": c_mean,
                "true_t_col": c_mean * fn2,
                **{f"true_t_dim{j+1}": c_mean * float(F[l, j] ** 2)
                   for j in range(K_TRUE)},
                **data["z_corr"],
            }
            # family-scale diagnostics only (NOT evidence of overdispersion)
            row["bernoulli_event_rate"] = float(x.mean()) if fam == "bernoulli" else np.nan
            row["poisson_mean"] = float(x.mean()) if fam == "poisson" else np.nan
            row["poisson_var"] = float(x.var()) if fam == "poisson" else np.nan
            row["gaussian_var"] = float(x.var()) if fam == "gaussian" else np.nan
            rows.append(row)
    return rows


# ── Conditions ────────────────────────────────────────────────────────────

def build_conditions(X):
    """condition -> (X_used, used_cols, family_list_used, run_kwargs)."""
    conds = {}
    for bname, (cols, _dim) in BLOCKS.items():
        fam = BLOCK_FAMILY[bname]
        conds[f"single_{fam}"] = (X[:, cols], cols, [fam] * len(cols),
                                  dict(family_x=fam, family_x_list=None))
    conds["per_column_all"] = (X, np.arange(D), list(FAM_LIST_TRUE),
                               dict(family_x=None, family_x_list=list(FAM_LIST_TRUE)))
    conds["all_gaussian"] = (X, np.arange(D), ["gaussian"] * D,
                             dict(family_x="gaussian", family_x_list=None))
    # y_only keeps the full 9-column X (so d=9) but blocks it with fix_x=True
    conds["y_only"] = (X, np.array([], dtype=int), [],
                       dict(family_x="gaussian", family_x_list=None, fix_x=True))
    return conds


def thin_pool_mask(pool_mask, keep_rate, seed):
    if keep_rate >= 1.0:
        return pool_mask.copy()
    n = pool_mask.shape[0]
    rows, cols = upper_pairs_of(pool_mask)
    n_pool = len(rows)
    n_keep = max(1, int(round(n_pool * keep_rate)))
    rng = np.random.default_rng(seed)
    keep_idx = rng.permutation(n_pool)[:n_keep]
    m = np.zeros((n, n), dtype=bool)
    m[rows[keep_idx], cols[keep_idx]] = True
    m |= m.T
    np.fill_diagonal(m, False)
    return m


# ── Precision diagnostics ─────────────────────────────────────────────────

def curvature_c(model, eta_x, sigma):
    """
    Plug-in per-column curvature weight c_il used by the model's own X precision term:

        P_X,i = sum_l c_il f_l f_l^T

        Gaussian : c = 1 / diag(sigma)[l]     (diag(sigma) stores VARIANCE)
        Bernoulli: c = p(1-p)
        Poisson  : c = exp(eta)

    Computed through the model's own hooks so it cannot drift from the implementation.
    """
    if getattr(model, "family_x_list", None) is not None:
        return (model._variance_function_x(eta_x)
                * model._x_weight_vector(sigma)[None, :])
    if model.family_x == "gaussian":
        inv = 1.0 / np.maximum(np.diag(sigma), 1e-8)      # NOT squared
        return np.broadcast_to(inv[None, :], eta_x.shape).copy()
    return model._variance_function_x(eta_x)


def precision_diagnostics(res, R, used_cols, is_y_only):
    """
    Plug-in mean trace contribution to the X-side local precision.

    t_il  = c_il * ||f_l||^2                       (column contribution)
    t_ilj = c_il * F_aligned[l, j]^2               (block x latent-dimension)
    with F_aligned = F_est @ R, the SAME R as the whole-space Procrustes rotation.
    This is the exact diagonal of the aligned X-side precision, not an approximation.
    NOT a posterior information quantity, and never used for weighting.
    """
    if is_y_only:
        out = {"t_total": 0.0, "identity_max_abs_err": 0.0}
        for b in BLOCKS:
            out[f"t_{b}"] = 0.0
            out[f"t_{b}_share"] = np.nan
            out[f"fnorm_{b}_mean"] = 0.0
            for j in range(K_TRUE):
                out[f"t_{b}_to_dim{j+1}"] = 0.0
                out[f"t_{b}_to_dim{j+1}_share"] = np.nan
            out[f"eta_{b}_min"] = np.nan
            out[f"eta_{b}_max"] = np.nan
            out[f"eta_{b}_var"] = np.nan
        return out

    model, F_est, sigma = res["model"], res["F"], res["sigma"]
    eta_x = res["Z_est"] @ F_est.T
    c = curvature_c(model, eta_x, sigma)                      # (n, d_used)
    fn2 = np.sum(F_est ** 2, axis=1)                          # (d_used,)
    t_col = np.mean(c, axis=0) * fn2                          # mean_i t_il
    F_al = F_est @ R
    t_cd = np.mean(c, axis=0)[:, None] * (F_al ** 2)          # mean_i t_ilj

    ident_err = float(np.max(np.abs(t_cd.sum(axis=1) - t_col))) if len(t_col) else 0.0
    t_total = float(t_col.sum())
    out = {"t_total": t_total, "identity_max_abs_err": ident_err}

    for bname, (cols, _dim) in BLOCKS.items():
        pos = [int(np.where(used_cols == cc)[0][0]) for cc in cols if cc in used_cols]
        if not pos:
            out[f"t_{bname}"] = np.nan
            out[f"t_{bname}_share"] = np.nan
            out[f"fnorm_{bname}_mean"] = np.nan
            out[f"eta_{bname}_min"] = np.nan
            out[f"eta_{bname}_max"] = np.nan
            out[f"eta_{bname}_var"] = np.nan
            for j in range(K_TRUE):
                out[f"t_{bname}_to_dim{j+1}"] = np.nan
                out[f"t_{bname}_to_dim{j+1}_share"] = np.nan
            continue
        tb = float(t_col[pos].sum())
        out[f"t_{bname}"] = tb
        out[f"t_{bname}_share"] = tb / t_total if t_total > 0 else np.nan
        out[f"fnorm_{bname}_mean"] = float(np.mean(np.sqrt(fn2[pos])))
        eb = eta_x[:, pos]
        out[f"eta_{bname}_min"] = float(eb.min())
        out[f"eta_{bname}_max"] = float(eb.max())
        out[f"eta_{bname}_var"] = float(eb.var())
        for j in range(K_TRUE):
            v = float(t_cd[pos, j].sum())
            out[f"t_{bname}_to_dim{j+1}"] = v
            out[f"t_{bname}_to_dim{j+1}_share"] = v / t_total if t_total > 0 else np.nan
    return out


# ── One fit, fully instrumented ───────────────────────────────────────────

def run_one_fit(X_used, Y, train_mask, seed, kw, key):
    """Run one fit with stdout capture (retry detection) and warning capture."""
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
                f"captured stdout tail:\n{buf.getvalue()[-800:]}") from exc
        wrecs = [{"category": w.category.__name__, "message": str(w.message),
                  "filename": str(w.filename), "lineno": int(w.lineno)} for w in wlist]
    captured = buf.getvalue()
    internal_retry_detected = "[NaN iter=" in captured
    return res, captured, internal_retry_detected, wrecs


def check_fit(key, res, internal_retry_detected, captured, wrecs):
    if internal_retry_detected:
        raise ExperimentStop(
            f"{key}: internal NaN reset / retry detected in the runner "
            f"(res['nan_occurred']={res['nan_occurred']}, "
            f"nan_count={res['nan_count']}). Result is NOT used.\n"
            f"captured retry log:\n"
            + "\n".join(ln for ln in captured.splitlines() if "[NaN iter=" in ln))
    if res["nan_occurred"]:
        raise ExperimentStop(f"{key}: nan_occurred=True (nan_count={res['nan_count']})")
    if res.get("q_bic_failed"):
        raise ExperimentStop(
            f"{key}: q_bic_failed=True, failure_reason={res.get('failure_reason')!r}. "
            "strict Q/BIC evaluation must complete cleanly (integrity requirement).")
    if res.get("numerics_mode") != "consistent":
        raise ExperimentStop(f"{key}: numerics_mode={res.get('numerics_mode')!r}")
    cd = res.get("clip_diag") or {}
    if cd.get("status") != "not_applicable":
        raise ExperimentStop(f"{key}: unexpected clip_diag={cd!r}")
    if wrecs:
        print(f"    [warning x{len(wrecs)}] {key}: "
              f"{wrecs[0]['category']}: {wrecs[0]['message'][:120]}")


# ── Main ──────────────────────────────────────────────────────────────────

def main(smoke: bool, out_dir: Path, fig_dir: Path):
    t0 = time.perf_counter()
    trials = TRIALS_SMOKE if smoke else TRIALS_FULL
    tag = f"{STEM}_smoke" if smoke else STEM
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, gen_rows, block_rows, warn_rows = [], [], [], []
    seen_keys = set()

    for trial in range(trials):
        data = generate(DATA_SEED_BASE + trial)
        gen_rows.extend(generator_diagnostics(trial, data))
        pool_mask, test_mask = make_pair_split(
            N, TEST_RATIO, seed=SPLIT_SEED_BASE + trial * 100)
        conds = build_conditions(data["X"])
        h_data = _sha16(data["X"], data["Y"], data["Z"], data["F"])
        h_test = _sha16(test_mask)

        for rate_idx, rate in enumerate(Y_OBS_RATES):
            train_mask = thin_pool_mask(
                pool_mask, rate, seed=THIN_SEED_BASE + trial * 100 + rate_idx)
            h_train = _sha16(train_mask)
            n_train_pairs = int(np.triu(train_mask, k=1).sum())

            for cname in CONDITIONS:
                X_used, used_cols, fam_used, kw = conds[cname]
                key = (trial, rate, cname)
                if key in seen_keys:
                    raise ExperimentStop(f"duplicate key {key}")
                seen_keys.add(key)

                res, captured, retry_flag, wrecs = run_one_fit(
                    X_used, data["Y"], train_mask,
                    MODEL_SEED_BASE + trial * 10, kw, str(key))
                check_fit(str(key), res, retry_flag, captured, wrecs)
                for w in wrecs:
                    warn_rows.append({"trial": trial, "y_obs_rate": rate,
                                      "condition": cname, **w})

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

                pdiag = precision_diagnostics(res, R, used_cols,
                                              is_y_only=(cname == "y_only"))
                if pdiag["identity_max_abs_err"] > 1e-8:
                    raise ExperimentStop(
                        f"{key}: precision identity sum_j t_ilj != t_il "
                        f"(max abs err {pdiag['identity_max_abs_err']:.3e})")

                row = {
                    "trial": trial, "y_obs_rate": rate, "condition": cname,
                    "n_cols_used": int(len(used_cols)), "n_train_pairs": n_train_pairs,
                    "rmse_Z": rmse_Z,
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
                    "hash_dataset": h_data, "hash_test_mask": h_test,
                    "hash_train_mask": h_train,
                    "runtime_s": res["runtime_s"],
                }
                for kk, vv in pdiag.items():
                    row[f"prec_{kk}"] = vv
                for mname in ("rmse_Z", "rmse_z_dim1", "rmse_z_dim2", "rmse_z_dim3",
                              "test_y_ll", "test_y_rmse", "train_y_ll"):
                    if not np.isfinite(row[mname]):
                        raise ExperimentStop(f"{key}: non-finite metric {mname}")
                rows.append(row)

                for bname in BLOCKS:
                    br = {"trial": trial, "y_obs_rate": rate, "condition": cname,
                          "block": bname}
                    br["t_block"] = pdiag[f"t_{bname}"]
                    br["t_block_share"] = pdiag[f"t_{bname}_share"]
                    br["fnorm_mean"] = pdiag[f"fnorm_{bname}_mean"]
                    br["eta_min"] = pdiag[f"eta_{bname}_min"]
                    br["eta_max"] = pdiag[f"eta_{bname}_max"]
                    br["eta_var"] = pdiag[f"eta_{bname}_var"]
                    for j in range(K_TRUE):
                        br[f"to_dim{j+1}"] = pdiag[f"t_{bname}_to_dim{j+1}"]
                        br[f"to_dim{j+1}_share"] = pdiag[f"t_{bname}_to_dim{j+1}_share"]
                    br["t_total"] = pdiag["t_total"]
                    block_rows.append(br)

                print(f"t={trial} rate={rate:.1f} {cname:17s} "
                      f"rmse_Z={rmse_Z:.4f} dim=({dim_rmse[0]:.3f},{dim_rmse[1]:.3f},"
                      f"{dim_rmse[2]:.3f}) te_ll={row['test_y_ll']:.4f}")

    df = pd.DataFrame(rows)
    expected = trials * len(Y_OBS_RATES) * len(CONDITIONS)
    if len(df) != expected:
        raise ExperimentStop(f"fit count {len(df)} != expected {expected}")
    counts = df.groupby(["y_obs_rate", "condition"]).size()
    if not (counts == trials).all():
        raise ExperimentStop(f"per-cell counts not all {trials}:\n{counts}")
    if df["internal_retry_detected"].any() or df["nan_occurred"].any() \
            or df["q_bic_failed"].any():
        raise ExperimentStop("integrity flags set after the loop")
    if not (df["numerics_mode"] == "consistent").all():
        raise ExperimentStop("not all fits ran in consistent mode")
    for trial in df["trial"].unique():
        sub = df[df["trial"] == trial]
        if sub["hash_dataset"].nunique() != 1 or sub["hash_test_mask"].nunique() != 1:
            raise ExperimentStop(f"trial {trial}: dataset/test hash mismatch")
        for rate in Y_OBS_RATES:
            s2 = sub[sub["y_obs_rate"] == rate]
            if s2["hash_train_mask"].nunique() != 1:
                raise ExperimentStop(f"trial {trial} rate {rate}: train hash mismatch")

    df.to_csv(out_dir / f"{tag}_summary.csv", index=False)
    pd.DataFrame(gen_rows).to_csv(out_dir / f"{tag}_generator.csv", index=False)
    pd.DataFrame(block_rows).to_csv(out_dir / f"{tag}_blockdiag.csv", index=False)

    agg = df.groupby(["condition", "y_obs_rate"]).agg(
        n_trials=("rmse_Z", "count"), n_train_pairs_mean=("n_train_pairs", "mean"),
        rmse_Z_mean=("rmse_Z", "mean"), rmse_Z_std=("rmse_Z", "std"),
        rmse_z_dim1_mean=("rmse_z_dim1", "mean"),
        rmse_z_dim2_mean=("rmse_z_dim2", "mean"),
        rmse_z_dim3_mean=("rmse_z_dim3", "mean"),
        test_y_ll_mean=("test_y_ll", "mean"), test_y_ll_std=("test_y_ll", "std"),
        test_y_rmse_mean=("test_y_rmse", "mean"),
        train_y_ll_mean=("train_y_ll", "mean"),
        w0_err_mean=("w0_err", "mean"), w_err_mean=("w_err", "mean"),
        runtime_s_mean=("runtime_s", "mean"),
        n_nan=("nan_occurred", "sum"), n_retry=("internal_retry_detected", "sum"),
        n_qbic_failed=("q_bic_failed", "sum")).reset_index()
    agg.to_csv(out_dir / f"{tag}_agg.csv", index=False)

    # paired analysis (trial-matched)
    piv = df.pivot_table(index=["y_obs_rate", "trial"], columns="condition",
                         values=["rmse_Z", "test_y_ll", "rmse_z_dim1",
                                 "rmse_z_dim2", "rmse_z_dim3"])
    prows = []
    for rate in Y_OBS_RATES:
        for comp in [c for c in CONDITIONS if c != "per_column_all"]:
            dz = (piv.loc[rate][("rmse_Z", comp)]
                  - piv.loc[rate][("rmse_Z", "per_column_all")])
            dl = (piv.loc[rate][("test_y_ll", "per_column_all")]
                  - piv.loc[rate][("test_y_ll", comp)])
            rec = {
                "y_obs_rate": rate, "comparator": comp,
                "contrast_label": CONTRAST_LABEL[comp],
                "is_primary_contrast": bool(rate == PRIMARY_RATE
                                            and comp in PRIMARY_COMPARATORS),
                "n_trials": int(dz.notna().sum()),
                "delta_rmse_mean": float(dz.mean()), "delta_rmse_std": float(dz.std()),
                "delta_rmse_median": float(dz.median()),
                "n_favoring_per_column_rmse": int((dz > 0).sum()),
                "delta_test_ll_mean": float(dl.mean()),
                "delta_test_ll_std": float(dl.std()),
                "delta_test_ll_median": float(dl.median()),
                "n_favoring_per_column_ll": int((dl > 0).sum()),
            }
            for j in range(1, K_TRUE + 1):
                d = (piv.loc[rate][(f"rmse_z_dim{j}", comp)]
                     - piv.loc[rate][(f"rmse_z_dim{j}", "per_column_all")])
                rec[f"delta_rmse_dim{j}_mean"] = float(d.mean())
                rec[f"n_favoring_per_column_dim{j}"] = int((d > 0).sum())
            prows.append(rec)
    paired = pd.DataFrame(prows)
    paired.to_csv(out_dir / f"{tag}_paired.csv", index=False)

    if warn_rows:
        pd.DataFrame(warn_rows).to_csv(out_dir / f"{tag}_warnings.csv", index=False)

    runinfo = [{
        "script": "tools/research_audit/run_complementary_blocks_consistent.py",
        "issue": 27, "datetime": datetime.now().isoformat(timespec="seconds"),
        "git_head": _git_head(), "branch": _git_branch(),
        "mode": "smoke" if smoke else "full",
        "n": N, "d": D, "k_true": K_TRUE, "trials": trials, "L": L, "num_iter": NITER,
        "test_ratio": TEST_RATIO, "y_obs_rates": str(Y_OBS_RATES),
        "conditions": str(CONDITIONS),
        "w0_true": W0_TRUE, "w_true": W_TRUE, "sigma_g": SIGMA_G,
        "dominant_weight": DOMINANT_WEIGHT, "minor_weight": MINOR_WEIGHT,
        "block_dominant_dim": "bernoulli->z1, gaussian->z2, poisson->z3",
        "fam_list_true": str(FAM_LIST_TRUE),
        "data_seed_base": DATA_SEED_BASE, "model_seed_base": MODEL_SEED_BASE,
        "split_seed_base": SPLIT_SEED_BASE, "thin_seed_base": THIN_SEED_BASE,
        "numerics_mode": "consistent", "generator_clipping": "none",
        "z_true_definition": "column-wise standardized N(0,I) draw",
        "n_fits": len(df), "n_internal_retry": int(df["internal_retry_detected"].sum()),
        "n_nan": int(df["nan_occurred"].sum()),
        "n_q_bic_failed": int(df["q_bic_failed"].sum()),
        "n_warnings": int(df["n_warnings"].sum()),
        "primary_domain": f"y_obs_rate={PRIMARY_RATE}",
        "primary_endpoint": "whole-space Procrustes RMSE_Z",
        "primary_contrasts": str([f"per_column_all vs {c}"
                                  for c in PRIMARY_COMPARATORS]),
        "delta_convention": "comparator_rmse - per_column_rmse (positive = per_column better)",
        "limitation_k": ("K_TRUE=3 here vs k*=2 in the existing sparse-Y evidence; "
                         "complementary F structure AND latent dimension both differ"),
        "limitation_curvature": ("family / dispersion / link-induced local-curvature "
                                 "imbalance under this pre-registered generator "
                                 "(Gaussian value depends on sigma_G=0.3; Bernoulli and "
                                 "Poisson depend on the eta distribution)"),
        "limitation_all_gaussian": ("same-column misspecification contrast; family "
                                    "specification and M-step optimizer path confounded "
                                    "(closed form vs weighted Adam)"),
        "limitation_single": ("single_* vs per_column_all also differs in the number of "
                              "observed X columns; not a pure family-assignment contrast"),
        "limitation_poisson_var": ("Poisson X marginal var > mean follows from latent "
                                   "heterogeneity in a conditionally Poisson generator; "
                                   "not overdispersion"),
        "total_runtime_s": round(time.perf_counter() - t0, 1),
    }]
    pd.DataFrame(runinfo).to_csv(out_dir / f"{tag}_runinfo.csv", index=False)

    print("\n=== AGG ===")
    print(agg.to_string(index=False))
    print("\n=== PAIRED (primary rows marked) ===")
    print(paired[["y_obs_rate", "comparator", "is_primary_contrast", "n_trials",
                  "delta_rmse_mean", "n_favoring_per_column_rmse",
                  "delta_test_ll_mean", "n_favoring_per_column_ll"]].to_string(index=False))

    if not smoke:
        make_figures(agg, paired, fig_dir)
    print(f"\nfits={len(df)}  total {(time.perf_counter() - t0)/60:.1f} min")
    return df, agg, paired


# ── Figures ───────────────────────────────────────────────────────────────

COLOR = {"y_only": "#888888", "single_bernoulli": "#8172B2",
         "single_gaussian": "#4C72B0", "single_poisson": "#CCB974",
         "per_column_all": "#C44E52", "all_gaussian": "#55A868"}


def make_figures(agg, paired, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylab, fname in (
            ("rmse_Z_mean", "RMSE_Z after Procrustes (lower is better)",
             f"{STEM}_rmse_z.png"),
            ("test_y_ll_mean", "held-out Y mean log-likelihood / pair (higher is better)",
             f"{STEM}_test_y_ll.png")):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for c in CONDITIONS:
            s = agg[agg["condition"] == c].sort_values("y_obs_rate")
            ax.plot(s["y_obs_rate"], s[metric], marker="o", label=c, color=COLOR[c])
        ax.set_xlabel("y_obs_rate (Y training observation rate)")
        ax.set_ylabel(ylab)
        ax.set_title("Complementary mixed-X blocks, consistent numerics\n"
                     "(10 trials; dense 1.0 = negative control, sparse 0.1 = primary)",
                     fontsize=9)
        ax.invert_xaxis()
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)
        print(f"saved {fig_dir / fname}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    width = 0.13
    xs = np.arange(K_TRUE)
    dimlab = ["dim1 (bernoulli block)", "dim2 (gaussian block)", "dim3 (poisson block)"]
    for ax, rate in zip(axes, Y_OBS_RATES):
        for i, c in enumerate(CONDITIONS):
            s = agg[(agg["condition"] == c) & (agg["y_obs_rate"] == rate)]
            vals = [float(s[f"rmse_z_dim{j+1}_mean"].iloc[0]) for j in range(K_TRUE)]
            ax.bar(xs + (i - 2.5) * width, vals, width, label=c, color=COLOR[c])
        ax.set_xticks(xs)
        ax.set_xticklabels(dimlab, fontsize=7)
        ax.set_title(f"y_obs_rate = {rate}", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("dimension-wise RMSE (shared Procrustes rotation)")
    axes[1].legend(fontsize=7)
    fig.suptitle("Dimension-wise Z recovery under one shared rotation "
                 "(mechanism diagnostic, not a general metric)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{STEM}_dimwise_rmse.png", dpi=150)
    plt.close(fig)
    print(f"saved {fig_dir / f'{STEM}_dimwise_rmse.png'}")


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
        print("EXPERIMENT STOPPED — integrity violation, result NOT used")
        print("=" * 70)
        print(exc)
        sys.exit(2)
