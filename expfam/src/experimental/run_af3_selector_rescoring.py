"""AF3 / S0: does the quality of the Z used by the Q-score decide whether the
true family (Poisson vs COM-Poisson nu=2) is selected?  Selector re-scoring
only -- NO MCEM fit is run (budget 0).

Background (AF2, PR #167, read-only audit 2026-11-13)
    AF2 misselected true CMP2 count columns as Poisson (dense 11/60, sparse
    27/60, Poisson start).  Most errors were not near ties: the signed margin
    stayed negative in every late exploration iteration.  Hypothesis (NOT
    assumed): an error in Z makes x | z_used over-dispersed relative to the
    true CMP2 conditional, so the equidispersed Poisson candidate wins.

What is scored (the existing selector, unchanged)
    For a fixed set of latent samples Z^(1..L) and each count column j, every
    candidate m in {poisson, cmp_nu2p0} is optimised and scored by the
    existing ``family_selection_multisupport.score_candidate`` (route
    trust_exact, the AF2 route):
        Q_j(m) = max_f (1/L) sum_l sum_i log p_m(x_ij | f^T z_i^(l))
    with the COMPLETE log-pmf (base measures included).  The winner is chosen
    by the existing ``select_column`` (tie rule: poisson first).
        signed margin  dQ_j = 2 (Q_j(true) - Q_j(wrong))   (> 0: true chosen)

Z sources ("arms", all on the SAME 20 AF2 datasets, regenerated)
    Z1   true Z (L = 1)                                         PRIMARY
    Z2   true Z + iid N(0, s^2) noise per sample (L = 5)         PRIMARY
    Z3   Oracle refit Z samples (AF2 npz, L = 5)                 PRIMARY
    Z5   Support-only refit Z samples (AF2 npz)                  PRIMARY
    Z4   Oracle joint mode (plug-in, L = 1)                      secondary
    Z6   Wrong refit Z samples                                   secondary
    Z7   Auto (Poisson start) refit Z samples                    secondary
    Z2c  true Z + one common noise draw (L = 1)                  secondary
    Z2m  true Z + common + per-sample noise matched to Oracle    secondary
    start sensitivity: Z3/Z5/Z6/Z7 re-scored from the fit's own loading
    (the incumbent) instead of the zero loading                  secondary

Lineage: E-consistent per-column + PR #129 generic model (experimental
prototype; not adoptable for the manuscript).  PR #129 is not merged.

Usage
    python run_af3_selector_rescoring.py --audit           # pre-run audit (no selection result)
    python run_af3_selector_rescoring.py --write-protocol
    python run_af3_selector_rescoring.py --smoke
    python run_af3_selector_rescoring.py --execute
    python run_af3_selector_rescoring.py --analyze --stage full
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
for p in (HERE, HERE.parent, ROOT / "tools" / "research_audit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_af2_poisson_cmp as af2                                      # noqa: E402
from run_auto_family_z_recovery import sha16, sha256_file, write_csv  # noqa: E402
from utils_expfam import calc_rmse, procrustes_rotation               # noqa: E402

EXPERIMENT = "AF3-S0"
N, K, D = af2.N, af2.K, af2.D
TRUE_MODELS = list(af2.TRUE_MODELS)
COUNT_COLS = tuple(range(3, 9))
CANDS = ("poisson", "cmp_nu2p0")
WRONG_OF = {"poisson": "cmp_nu2p0", "cmp_nu2p0": "poisson"}
ROUTE = "trust_exact"
REGIMES = ("dense", "sparse")
L_SAMPLES = 5

AF2_DIR = ROOT / "expfam" / "results" / "auto_family_z_recovery" / "af2_20261110"
AF2_FULL = AF2_DIR / "full"
AF2_STATE = af2.STATE_ROOT / "full" / "fits"
OUT_ROOT = ROOT / "expfam" / "results" / "auto_family_z_recovery" / "af3_selector_rescoring_20261113"

# noise grid: the proposal {0.1, 0.2, 0.3, 0.45} plus 0.6, added BEFORE any
# selection result because the pre-run calibration (audit/z_error_calibration.csv)
# put the per-sample rotation-Procrustes RMSE of the Oracle samples at about
# 0.37 (dense) and 0.56 (sparse); 0.45 would not reach the sparse range.
NOISE_LEVELS = (0.1, 0.2, 0.3, 0.45, 0.6)
NOISE_SEEDS = (0, 1, 2, 3, 4)
NOISE_SEED_BASE = 1664000
# matched noise z + a*e0 + b*e_l, from the Oracle calibration medians (audit):
#   b = within_sd * sqrt(5/4)  (within_sd is the ddof-0 spread of 5 samples)
#   a = sqrt(rmse_mean5^2 - b^2/5)
MATCHED = {"dense": {"a": 0.27, "b": 0.25}, "sparse": {"a": 0.41, "b": 0.38}}
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 10000, 20261113
NPZ_METHODS = {"Z3": "oracle", "Z5": "support_only", "Z6": "wrong", "Z7": "auto_first"}
PRIMARY_ARMS = ("Z1", "Z2", "Z3", "Z5")

PR129_BRANCH = "origin/exp/multisupport-family-selection"
PR129_RESULTS = "expfam/results/family_selection/expanded_multisupport_20261006_034826"
PR129_REPRO_CASES = ("AB_main_COUNT_EASY_n75_rep001", "AB_main_COUNT_EASY_n75_rep002")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=ROOT).stdout.strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def registry_and_priority():
    return af2.registry_and_priority()


def prmse(Ze: np.ndarray, Zt: np.ndarray) -> float:
    R, k = procrustes_rotation(Ze, Zt)
    return float(calc_rmse(Zt[:, :k], Ze[:, :k] @ R))


def linres(Ze: np.ndarray, Zt: np.ndarray) -> float:
    B, *_ = np.linalg.lstsq(Ze, Zt, rcond=None)
    return float(calc_rmse(Zt, Ze @ B))


def noise_rng(*key: int) -> np.random.Generator:
    return np.random.default_rng([NOISE_SEED_BASE, *key])


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------

def arm_instances() -> list[dict[str, Any]]:
    """Every (arm, variant) scored on every replicate."""
    inst = [{"arm": "Z1", "variant": "true", "regime": "", "level": "", "noise_seed": "",
             "start": "zero", "primary": True}]
    for li, s in enumerate(NOISE_LEVELS):
        for j in NOISE_SEEDS:
            inst.append({"arm": "Z2", "variant": f"iid_s{s}", "regime": "", "level": s,
                         "level_index": li, "noise_seed": j, "start": "zero", "primary": True})
    for arm, method in NPZ_METHODS.items():
        for reg in REGIMES:
            inst.append({"arm": arm, "variant": f"{method}_samples", "regime": reg,
                         "level": "", "noise_seed": "", "start": "zero",
                         "primary": arm in PRIMARY_ARMS})
    for reg in REGIMES:
        inst.append({"arm": "Z4", "variant": "oracle_mode", "regime": reg, "level": "",
                     "noise_seed": "", "start": "zero", "primary": False})
    for li, s in enumerate(NOISE_LEVELS):
        for j in NOISE_SEEDS:
            inst.append({"arm": "Z2c", "variant": f"common_s{s}", "regime": "", "level": s,
                         "level_index": li, "noise_seed": j, "start": "zero", "primary": False})
    for ri, reg in enumerate(REGIMES):
        for j in NOISE_SEEDS:
            inst.append({"arm": "Z2m", "variant": f"matched_{reg}", "regime": reg,
                         "level": "", "regime_index": ri, "noise_seed": j, "start": "zero",
                         "primary": False})
    for arm, method in NPZ_METHODS.items():
        for reg in REGIMES:
            inst.append({"arm": arm, "variant": f"{method}_samples", "regime": reg,
                         "level": "", "noise_seed": "", "start": "incumbent", "primary": False})
    for x in inst:
        x["instance"] = "|".join(str(x[k]) for k in ("arm", "variant", "regime", "noise_seed", "start"))
    return inst


def load_npz(r: int, regime: str, method: str) -> dict[str, np.ndarray]:
    key = f"{af2.m1.replicate_label(r)}_{regime}_{method}"
    with np.load(AF2_STATE / f"{key}.npz", allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def latent_samples(inst, r: int, Z_true: np.ndarray):
    """Return (Z_samples (n,K,L), loading_init (D,K) or None)."""
    arm = inst["arm"]
    if arm == "Z1":
        return Z_true[:, :, None].copy(), None
    if arm == "Z2":
        rng = noise_rng(r, 1, inst["level_index"], inst["noise_seed"])
        eps = rng.standard_normal((N, K, L_SAMPLES))
        return Z_true[:, :, None] + inst["level"] * eps, None
    if arm == "Z2c":
        rng = noise_rng(r, 2, inst["level_index"], inst["noise_seed"])
        eps = rng.standard_normal((N, K))
        return (Z_true + inst["level"] * eps)[:, :, None], None
    if arm == "Z2m":
        rng = noise_rng(r, 3, inst["regime_index"], inst["noise_seed"])
        a, b = MATCHED[inst["regime"]]["a"], MATCHED[inst["regime"]]["b"]
        e0 = rng.standard_normal((N, K))
        el = rng.standard_normal((N, K, L_SAMPLES))
        return Z_true[:, :, None] + a * e0[:, :, None] + b * el, None
    if arm == "Z4":
        st = load_npz(r, inst["regime"], "oracle")
        return st["Zmode"][:, :, None].copy(), None
    st = load_npz(r, inst["regime"], NPZ_METHODS[arm])
    return st["Zsamples"].copy(), (st["F"] if inst["start"] == "incumbent" else None)


# --------------------------------------------------------------------------
# scoring one column (existing selector + extra read-only diagnostics)
# --------------------------------------------------------------------------

def candidate_diagnostics(spec, x, Zs, rec) -> dict[str, Any]:
    """Per-observation and per-sample contributions at the returned loading,
    the Hessian's smallest eigenvalue, and the score-gap bound g^2/(2 lam)."""
    out = {"per_obs": None, "per_sample": None, "hess_min_eig": float("nan"),
           "score_gap_bound": float("nan"), "sum_check_abs": float("nan")}
    if rec.status != "OK" or rec.loading is None:
        return out
    eta = np.einsum("nkl,k->nl", Zs, rec.loading)
    lp = spec.log_prob(x[:, None], eta)
    out["per_obs"] = lp.mean(axis=1)
    out["per_sample"] = lp.sum(axis=0)
    out["sum_check_abs"] = abs(float(out["per_obs"].sum()) - rec.score)
    var = spec.variance_statistic(eta)
    H = np.einsum("nl,nkl,njl->kj", var, Zs, Zs) / Zs.shape[2]
    lam = float(np.linalg.eigvalsh((H + H.T) / 2.0)[0])
    out["hess_min_eig"] = lam
    if lam > 0 and math.isfinite(rec.gradient_inf):
        g2 = K * rec.gradient_inf ** 2                  # ||g||_2^2 <= K ||g||_inf^2
        out["score_gap_bound"] = g2 / (2.0 * lam)
    return out


def score_instance(inst, r: int, ds, reg, prio) -> tuple[list, list, dict]:
    from family_selection_multisupport import score_candidate, select_column
    from family_specs import GROUP_COUNT
    Zs, Finit = latent_samples(inst, r, np.asarray(ds.Z, float))
    Zt = np.asarray(ds.Z, float)
    Lz = Zs.shape[2]
    zdiag = {"rmse_per_sample": float(np.mean([prmse(Zs[:, :, l], Zt) for l in range(Lz)])),
             "rmse_mean_of_samples": prmse(Zs.mean(axis=2), Zt),
             "linres_per_sample": float(np.mean([linres(Zs[:, :, l], Zt) for l in range(Lz)])),
             "within_sample_sd": float(np.sqrt(np.mean((Zs - Zs.mean(axis=2, keepdims=True)) ** 2))),
             "z_mean_square": float(np.mean(Zs ** 2)), "L": Lz,
             "z_samples_sha": sha16(Zs)}
    cand_rows, col_rows = [], []
    base = {k: inst[k] for k in ("instance", "arm", "variant", "regime", "level",
                                  "noise_seed", "start", "primary")}
    base.update(replicate=af2.m1.replicate_label(r), data_seed=af2.DATA_SEEDS[r])
    F_true = np.asarray(ds.F, float)
    for col in COUNT_COLS:
        x = np.asarray(ds.X[:, col], float)
        init = np.zeros(K) if Finit is None else np.asarray(Finit[col], float)
        recs, diags = [], {}
        for c in CANDS:
            spec = reg[c]
            t0 = time.perf_counter()
            rec = score_candidate(x, Zs, spec, init, col, route=ROUTE)
            dt = time.perf_counter() - t0
            dg = candidate_diagnostics(spec, x, Zs, rec)
            diags[c] = dg
            recs.append(rec)
            cand_rows.append(dict(base, column=col, true_model=TRUE_MODELS[col],
                                  candidate=c, status=rec.status, score=rec.score,
                                  score_per_obs=rec.score / N if rec.status == "OK" else "",
                                  start_loading_norm=float(np.linalg.norm(init)),
                                  loading_norm=(float(np.linalg.norm(rec.loading))
                                                if rec.loading is not None else ""),
                                  grad_inf=rec.gradient_inf, converged_1e8=rec.converged,
                                  n_iter=rec.n_iter, scipy_success=rec.scipy_success,
                                  scipy_status=rec.scipy_status,
                                  optimizer_route=rec.optimizer_route,
                                  hess_min_eig=dg["hess_min_eig"],
                                  score_gap_bound=dg["score_gap_bound"],
                                  per_obs_sum_check_abs=dg["sum_check_abs"],
                                  failure_reason=rec.failure_reason,
                                  seconds=round(dt, 4)))
        sel = select_column(recs, prio[GROUP_COUNT])
        tm = TRUE_MODELS[col]
        wm = WRONG_OF[tm]
        row = dict(base, column=col, true_model=tm,
                   true_loading_norm=float(np.linalg.norm(F_true[col])),
                   x_mean=float(x.mean()), x_var_over_mean=float(x.var() / max(x.mean(), 1e-12)),
                   x_max=float(x.max()), status=sel.status, selected=sel.selected,
                   correct=bool(sel.status == "SELECTED" and sel.selected == tm),
                   misselected_as=(sel.selected if sel.status == "SELECTED" and sel.selected != tm else ""),
                   failure_reason=sel.failure_reason, **{f"z_{k}": v for k, v in zdiag.items()})
        rt = next(r_ for r_ in recs if r_.candidate_model == tm)
        rw = next(r_ for r_ in recs if r_.candidate_model == wm)
        if sel.status == "SELECTED":
            dq = 2.0 * (rt.score - rw.score)
            d_i = diags[tm]["per_obs"] - diags[wm]["per_obs"]
            sd = float(np.std(d_i, ddof=1))
            ps = 2.0 * (diags[tm]["per_sample"] - diags[wm]["per_sample"])
            row.update(score_true=rt.score, score_wrong=rw.score, signed_margin=dq,
                       signed_margin_per_obs=dq / N,
                       margin_neg2_selector=sel.margin_neg2,
                       per_obs_diff_mean=float(d_i.mean()), per_obs_diff_sd=sd,
                       vuong_z_descriptive=(float(d_i.sum() / (math.sqrt(N) * sd)) if sd > 0 else float("nan")),
                       n_obs_favouring_true=int(np.sum(d_i > 0)),
                       per_sample_margin_min=float(ps.min()), per_sample_margin_max=float(ps.max()),
                       per_sample_margin_sd=float(np.std(ps, ddof=1)) if Lz > 1 else float("nan"),
                       n_samples_sign_disagree=int(np.sum(np.sign(ps) != np.sign(dq))),
                       max_score_gap_bound=max(diags[c]["score_gap_bound"] for c in CANDS),
                       min_hess_eig=min(diags[c]["hess_min_eig"] for c in CANDS),
                       both_converged_1e8=bool(rt.converged and rw.converged))
        col_rows.append(row)
    return cand_rows, col_rows, zdiag


# --------------------------------------------------------------------------
# pre-run audit (no selection result is computed or read here, except the
# PR #129 reproduction, whose expected values are PR #129's published ones)
# --------------------------------------------------------------------------

def pr129_reproduction() -> list[dict[str, Any]]:
    from data_generator_multisupport import generate_multisupport_data
    from family_selection_multisupport import selector_only
    from family_specs import PRIMARY_PRIORITY, build_primary_registry
    reg = build_primary_registry()
    prio = {g: tuple(v) for g, v in PRIMARY_PRIORITY.items()}
    raw = subprocess.run(["git", "show", f"{PR129_BRANCH}:{PR129_RESULTS}/tables/candidate_scores.csv"],
                         capture_output=True, text=True, cwd=ROOT, encoding="utf-8").stdout
    published = [r for r in csv.DictReader(io.StringIO(raw))
                 if r["experiment"] == "A" and r["stage"] == "main"]
    rows = []
    for key in PR129_REPRO_CASES:
        meta = json.loads(subprocess.run(
            ["git", "show", f"{PR129_BRANCH}:{PR129_RESULTS}/data/dataset_{key}.json"],
            capture_output=True, text=True, cwd=ROOT, encoding="utf-8").stdout)
        ds = generate_multisupport_data(n=meta["n"], k=meta["K_true"], seed=meta["seed"],
                                        column_specs=[reg[m] for m in meta["true_candidate_models"]],
                                        tau=meta["tau"], w0=meta["w0"], w=meta["w"])
        _, sels = selector_only(ds.X, ds.Z, reg, prio, route=ROUTE)
        case_id = f"A__main__{meta['group']}__{meta['difficulty']}__n{meta['n']}__rep{meta['replicate']:03d}"
        pub = {(int(r["column"]), r["candidate_model"]): r for r in published if r["case_id"] == case_id}
        for s in sels:
            for rec in s.records:
                p = pub.get((s.column, rec.candidate_model))
                ref = float(p["candidate_score"]) if p else float("nan")
                rows.append({"case": key, "column": s.column, "candidate": rec.candidate_model,
                             "reproduced_score": rec.score, "published_score": ref,
                             "abs_diff": abs(rec.score - ref) if p else float("nan"),
                             "found_in_published": p is not None})
    return rows


def run_audit(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    fr = {r["fit_key"]: r for r in read_csv(AF2_FULL / "fit_runs.csv")}
    reg, prio = registry_and_priority()
    from family_selection_multisupport import gate_columns
    data_rows, npz_rows, cal_rows = [], [], []
    for r in range(len(af2.DATA_SEEDS)):
        ds = af2.generate(r)
        rep = af2.m1.replicate_label(r)
        gates = gate_columns(ds.X, reg, prio)
        for regime in REGIMES:
            for method in af2.METHODS:
                key = f"{rep}_{regime}_{method}"
                f = fr[key]
                p = AF2_STATE / f"{key}.npz"
                st = load_npz(r, regime, method)
                sha = sha256_file(p)
                npz_rows.append({
                    "fit_key": key, "replicate": rep, "regime": regime, "method": method,
                    "npz_sha256": sha, "fit_runs_state_sha256": f["state_sha256"],
                    "sha_match": sha == f["state_sha256"],
                    "x_sha_match": sha16(ds.X) == f["x_sha"], "y_sha_match": sha16(ds.Y) == f["y_sha"],
                    "z_true_sha_match": sha16(ds.Z) == f["z_true_sha"],
                    "Zsamples_shape": "x".join(map(str, st["Zsamples"].shape)),
                    "Zmode_shape": "x".join(map(str, st["Zmode"].shape)) if "Zmode" in st else "",
                    "F_shape": "x".join(map(str, st["F"].shape)),
                    "w0_match": abs(float(st["scalars"][0]) - float(f["w0_hat"])) < 1e-12,
                    "w_match": abs(float(st["scalars"][1]) - float(f["w_hat"])) < 1e-12,
                    "assignment": f["assignment"], "status": f["status"], "mode_status": f["mode_status"],
                    "zsamples_finite": bool(np.all(np.isfinite(st["Zsamples"]))),
                })
                if method in ("oracle", "support_only", "wrong", "auto_first"):
                    Zs, Zt = st["Zsamples"], np.asarray(ds.Z, float)
                    within = float(np.sqrt(np.mean((Zs - Zs.mean(axis=2, keepdims=True)) ** 2)))
                    cal_rows.append({"replicate": rep, "regime": regime, "method": method,
                                     "rmse_per_sample": float(np.mean([prmse(Zs[:, :, l], Zt) for l in range(Zs.shape[2])])),
                                     "rmse_mean5": prmse(Zs.mean(axis=2), Zt),
                                     "rmse_mode": prmse(st["Zmode"], Zt),
                                     "linres_per_sample": float(np.mean([linres(Zs[:, :, l], Zt) for l in range(Zs.shape[2])])),
                                     "within_sd": within, "z_mean_square": float(np.mean(Zs ** 2))})
        data_rows.append({"replicate": rep, "data_seed": af2.DATA_SEEDS[r], "x_sha": sha16(ds.X),
                          "y_sha": sha16(ds.Y), "z_true_sha": sha16(ds.Z),
                          "gate_count_cols_admissible": "|".join(
                              "+".join(gates[c].admissible) for c in COUNT_COLS),
                          "count_cols_all_score_decided": all(gates[c].decided_by == "score" for c in COUNT_COLS),
                          "true_loading_norms_count": "|".join(f"{np.linalg.norm(ds.F[c]):.4f}" for c in COUNT_COLS)})
    # noise-level interpretation: rotation-Procrustes RMSE of true Z + s*eps
    noise_rows = []
    for r in range(len(af2.DATA_SEEDS)):
        Zt = np.asarray(af2.generate(r).Z, float)
        for li, s in enumerate(NOISE_LEVELS):
            rng = np.random.default_rng([NOISE_SEED_BASE, r, 99, li])
            Zn = Zt + s * rng.standard_normal(Zt.shape)
            noise_rows.append({"replicate": af2.m1.replicate_label(r), "s": s,
                               "rmse_procrustes": prmse(Zn, Zt), "linres": linres(Zn, Zt)})
    repro = pr129_reproduction()
    write_csv(out / "data_audit.csv", data_rows)
    write_csv(out / "npz_audit.csv", npz_rows)
    write_csv(out / "z_error_calibration.csv", cal_rows)
    write_csv(out / "noise_level_interpretation.csv", noise_rows)
    write_csv(out / "pr129_selector_reproduction.csv", repro)
    calib = {}
    for regime in REGIMES:
        o = [c for c in cal_rows if c["regime"] == regime and c["method"] == "oracle"]
        med = {k: float(np.median([c[k] for c in o])) for k in ("rmse_per_sample", "rmse_mean5", "within_sd", "rmse_mode")}
        b = med["within_sd"] * math.sqrt(5 / 4)
        a = math.sqrt(max(med["rmse_mean5"] ** 2 - b ** 2 / 5, 0.0))
        calib[regime] = dict(med, matched_a=a, matched_b=b)
    summary = {
        "n_npz": len(npz_rows),
        "all_npz_sha_match": all(x["sha_match"] for x in npz_rows),
        "all_data_sha_match": all(x["x_sha_match"] and x["y_sha_match"] and x["z_true_sha_match"] for x in npz_rows),
        "all_shapes_ok": all(x["Zsamples_shape"] == f"{N}x{K}x5" and x["Zmode_shape"] == f"{N}x{K}"
                             and x["F_shape"] == f"{D}x{K}" for x in npz_rows),
        "all_theta_match": all(x["w0_match"] and x["w_match"] for x in npz_rows),
        "all_fits_ok": all(x["status"] == "OK" and x["mode_status"] == "OK" for x in npz_rows),
        "all_count_cols_score_decided": all(x["count_cols_all_score_decided"] for x in data_rows),
        "pr129_repro_rows": len(repro),
        "pr129_repro_all_found": all(x["found_in_published"] for x in repro),
        "pr129_repro_max_abs_diff": max(x["abs_diff"] for x in repro),
        "oracle_calibration": calib,
        "noise_level_rmse_median": {str(s): float(np.median([x["rmse_procrustes"] for x in noise_rows if x["s"] == s]))
                                    for s in NOISE_LEVELS},
        "git_head": git("rev-parse", "HEAD"),
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------

def protocol() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "mode": "CONFIRMATORY for the primary questions (written before any AF3 selection result); "
                "Z2c / Z2m / Z4 / Z6 / Z7 / incumbent start are secondary",
        "research_question": "If the Z used by the Q-score is less accurate, does the rate of "
                             "selecting the true family (Poisson vs COM-Poisson nu=2) fall, and "
                             "do the errors go CMP2 -> Poisson?",
        "hypothesis_not_assumed": "true Z selects correctly; adding error to Z increases CMP2 -> Poisson",
        "lineage": "E-consistent per-column + PR #129 generic model (experimental prototype; "
                   "not adoptable for the manuscript). PR #129 not merged; AF1/AF2 PRs (#166/#167) Draft",
        "mcem_fits": 0,
        "data": {"source": "AF2 datasets regenerated with af2.generate(r) (PR #129 generator, unchanged)",
                 "n": N, "K": K, "d": D, "true_models": TRUE_MODELS, "count_columns": list(COUNT_COLS),
                 "data_seeds": list(af2.DATA_SEEDS),
                 "check": "x/y/z_true sha16 equal to AF2 fit_runs.csv for every fit"},
        "af2_states": {"dir_outside_repo": str(AF2_STATE),
                       "check": "sha256 equal to AF2 fit_runs.csv state_sha256 (200 files)",
                       "used": {"Z3": "oracle Zsamples", "Z5": "support_only Zsamples",
                                "Z6": "wrong Zsamples", "Z7": "auto_first (refit) Zsamples",
                                "Z4": "oracle Zmode", "incumbent start": "the same fit's F row"}},
        "candidates": list(CANDS), "nu": 2.0, "route": ROUTE,
        "selector": "family_selection_multisupport.score_candidate + select_column (unchanged); "
                    "tie rule poisson first; complete log-pmf",
        "start": {"primary": "zero loading for both candidates (as PR #129 selector_only)",
                  "sensitivity": "incumbent = the AF2 fit's own refit loading for that column"},
        "arms": {
            "Z1": "true Z, L=1 [PRIMARY]",
            "Z2": f"true Z + s*eps, eps iid N(0,I) per sample, L={L_SAMPLES}, s in {list(NOISE_LEVELS)}, "
                  f"noise seeds {list(NOISE_SEEDS)} [PRIMARY]",
            "Z3": "Oracle refit Zsamples, dense and sparse [PRIMARY]",
            "Z5": "Support-only refit Zsamples, dense and sparse [PRIMARY]",
            "Z4": "Oracle Zmode (plug-in, L=1) [secondary]",
            "Z6": "Wrong refit Zsamples [secondary]",
            "Z7": "Auto (Poisson start) refit Zsamples [secondary; for column-level linkage to AF2]",
            "Z2c": "true Z + one common draw s*eps, L=1 [secondary]",
            "Z2m": f"true Z + a*e0 + b*e_l, L=5, (a,b) {MATCHED} [secondary]",
            "incumbent_start": "Z3/Z5/Z6/Z7 re-scored from the incumbent loading [secondary]"},
        "noise_interpretation": (
            "Q_j is invariant to any invertible linear map of Z applied to all samples (no "
            "intercept: f^T z = (A^-T f)^T (A z)), so rotation and global scale of Z do not "
            "matter for selection; translation would. s is the per-coordinate SD of an isotropic "
            "error on N(0,I) latent positions; for one draw the rotation-Procrustes RMSE is about "
            "s (audit/noise_level_interpretation.csv). The grid adds 0.6 to the proposed "
            "{0.1,0.2,0.3,0.45} because the Oracle per-sample RMSE is ~0.37 dense / ~0.56 sparse "
            "(audit/z_error_calibration.csv; Z geometry only, no selection result)."),
        "noise_seed_rule": f"np.random.default_rng([{NOISE_SEED_BASE}, r, arm_code, level_or_regime_index, noise_seed]); "
                           "arm_code 1=Z2, 2=Z2c, 3=Z2m",
        "signed_margin": "dQ = 2 (Q_true - Q_wrong); also dQ / n",
        "primary_metrics": ["true-CMP2 column correct rate", "true-CMP2 signed margin",
                            "Z2 noise level vs CMP2 -> Poisson misselection rate"],
        "secondary_metrics": ["true-Poisson correct rate", "Z5 vs Z3 (paired)",
                              "true loading norm vs selection", "per-observation score differences "
                              "(descriptive Vuong-type z, conditional on the given Z; NOT a test)",
                              "per-sample margins (L=5 chain draws; NOT an MC standard error)",
                              "optimizer gradient, Hessian min eigenvalue, score-gap bound g^2/(2 lam)"],
        "statistical_unit": "replicate (20 independent datasets). Columns within a replicate and noise "
                            "seeds are averaged inside the replicate first; never counted as independent",
        "aggregation": {
            "rate": "per replicate: share of its 3 true-CMP2 (or true-Poisson) columns correct; Z2/Z2c/Z2m "
                    "averaged over the 5 noise seeds inside the replicate; then mean over replicates",
            "interval": f"percentile bootstrap over replicates ({BOOTSTRAP_RESAMPLES}, seed {BOOTSTRAP_SEED})",
            "dose_response": "per replicate OLS slope of the seed-averaged CMP2->Poisson misselection rate "
                             "on s (5 levels); mean slope with bootstrap CI and the count of positive slopes; "
                             "Z1 is reported as s=0 but not used in the slope",
            "paired": "Z5 - Z3 per replicate and regime (CMP2 misselection rate), bootstrap CI, sign counts"},
        "failure_rule": "a FAILED candidate makes the column FAILURE (existing rule); it counts as not "
                        "correct in every rate and is reported separately; nothing is retried, reseeded or "
                        "excluded silently",
        "technical_stop_conditions": ["data hash mismatch with AF2", "npz sha256 mismatch",
                                      "npz shape mismatch", "PR #129 selector reproduction abs diff > 1e-9",
                                      "protocol.json differs from protocol()"],
        "reading_guide_predeclared": {
            "note": "pre-declared guides for the next step, not significance tests; the thresholds "
                    "are judgement aids without a formal statistical basis",
            "G1": "Z1 true-CMP2 correct rate < 0.90 -> selector/candidate problem first",
            "G2": "Z1 >= 0.90, Z2 mean slope > 0 with CI excluding 0, and >= 80% of Z2 errors are "
                  "CMP2->Poisson -> Z error drives CMP2->Poisson (for this noise model)",
            "G3": "Z3 CMP2 misselection >= 0.10 in a regime -> even Oracle-fit Z misselects -> S1 priority: "
                  "selection robust to Z uncertainty (candidate-wise refit + predictive comparison)",
            "G4": "Z3 CMP2 misselection <= 0.05 and Z5 - Z3 >= 0.10 -> incumbent-dependent q -> S1 priority: "
                  "burn-in / outer cycle",
            "G5": ">= 50% of Z3+Z5 misselections with |dQ| < 2 -> near ties -> stabilised selection (B) justified"},
        "budget": {"instances": len(arm_instances()), "replicates": 20,
                   "candidate_optimisations": len(arm_instances()) * 20 * len(COUNT_COLS) * len(CANDS)},
        "artifacts": {"results": str(OUT_ROOT.relative_to(ROOT)).replace("\\", "/")},
    }


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True).encode()).hexdigest()


def runinfo(extra: dict[str, Any]) -> dict[str, Any]:
    import scipy
    req = ROOT / "requirements.txt"
    return {"experiment": EXPERIMENT, "script": "expfam/src/experimental/run_af3_selector_rescoring.py",
            "git_head": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_tracked": bool(git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(),
            "requirements_sha256": sha256_file(req) if req.exists() else "",
            "protocol_sha256": protocol_sha256(), "command": " ".join(sys.argv), **extra}


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def execute(out: Path, replicates: list[int]) -> None:
    out.mkdir(parents=True, exist_ok=False)
    reg, prio = registry_and_priority()
    insts = arm_instances()
    cand_rows, col_rows, z_rows = [], [], []
    t_all = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    n_warn = 0
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        for r in replicates:
            ds = af2.generate(r)
            for inst in insts:
                c, cols, zd = score_instance(inst, r, ds, reg, prio)
                cand_rows.extend(c)
                col_rows.extend(cols)
                z_rows.append({"instance": inst["instance"], "arm": inst["arm"], "variant": inst["variant"],
                               "regime": inst["regime"], "level": inst["level"],
                               "noise_seed": inst["noise_seed"], "start": inst["start"],
                               "replicate": af2.m1.replicate_label(r), **zd})
            print(f"{af2.m1.replicate_label(r)} done, {round(time.perf_counter() - t_all, 1)} s", flush=True)
        n_warn = len(wl)
        warn_msgs = sorted({f"{x.category.__name__}: {str(x.message)[:160]}" for x in wl})[:20]
    write_csv(out / "candidate_scores.csv", cand_rows)
    write_csv(out / "column_selection.csv", col_rows)
    write_csv(out / "z_error_by_instance.csv", z_rows)
    (out / "runinfo.json").write_text(json.dumps(runinfo({
        "stage": out.name, "replicates": [af2.m1.replicate_label(r) for r in replicates],
        "instances": len(insts), "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_s": round(time.perf_counter() - t_all, 1),
        "n_warnings": n_warn, "warning_messages": warn_msgs,
        "seeds": {"data": "1661001 + r", "noise": f"[{NOISE_SEED_BASE}, r, arm_code, index, seed]",
                  "bootstrap": BOOTSTRAP_SEED},
        "af2_state_dir": str(AF2_STATE), "mcem_fits": 0}), indent=2), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit", action="store_true")
    g.add_argument("--write-protocol", action="store_true")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--execute", action="store_true")
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--stage", default="full")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args(argv)
    out_root = Path(args.out_root)

    if args.audit:
        if (out_root / "audit").exists():
            raise SystemExit("refusing to overwrite audit/")
        s = run_audit(out_root / "audit")
        print(json.dumps(s, indent=2))
        return 0
    if args.write_protocol:
        p = out_root / "protocol.json"
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
        out_root.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(protocol(), indent=2), encoding="utf-8")
        write_csv(out_root / "manifest.csv",
                  [dict(i, replicate=af2.m1.replicate_label(r)) for r in range(20) for i in arm_instances()])
        print(f"wrote {p} sha256(canonical)={protocol_sha256()}")
        return 0
    if args.analyze:
        import analyze_af3_selector_rescoring as an
        an.main(out_root / args.stage)
        return 0
    frozen = json.loads((out_root / "protocol.json").read_text(encoding="utf-8"))
    if hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest() != protocol_sha256():
        raise SystemExit("protocol.json on disk differs from protocol(); refusing to run")
    audit = json.loads((out_root / "audit" / "audit_summary.json").read_text(encoding="utf-8"))
    ok = (audit["all_npz_sha_match"] and audit["all_data_sha_match"] and audit["all_shapes_ok"]
          and audit["pr129_repro_all_found"] and audit["pr129_repro_max_abs_diff"] <= 1e-9)
    if not ok:
        raise SystemExit("pre-run audit did not pass; refusing to run")
    if args.smoke:
        execute(out_root / "smoke", [0])
    else:
        execute(out_root / "full", list(range(20)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
