"""Experiment M1 (Issue #134): Bernoulli -> Poisson X misspecification
under dense vs sparse Y training observation.

Research question
-----------------
On the same synthetic dataset, fit the six truly-Bernoulli attribute columns
either as Bernoulli (correct) or as Poisson (wrong, support-compatible: 0/1 is
inside the Poisson support).  How does Procrustes RMSE_Z change, and does the
change differ between dense and sparse Y training observation?

A positive delta (wrong worse) is NOT a success criterion.  No difference,
reversed sign, and replicate instability are all valid results.

Frozen design (Issue #134; do not change after seeing results)
--------------------------------------------------------------
data      : Phase 9K generator and condition (canonical-clean-mixed-v1),
            n=75, d=12, K_true=3, X = G3/B6/P3, Y = Bernoulli,
            sigma_x_var=1, f_scale=sqrt(2), w0=-1, w=1,
            data seeds 1001001..1001020 (one draw per replicate, shared by
            all four fits of that replicate)
arms      : correct G3/B6/P3, wrong G3/P6/P3 (only columns 3..8 change);
            raw X is never transformed
K         : K_fit = K_true = 3; no K sweep, no K selection
family    : fixed per arm; no family selection
Y masks   : per replicate, a fixed 10% test set of upper-triangle pairs;
            the remaining 90% is the training candidate pool;
            dense train = 100% of the pool, sparse train = 10% of the pool
            (sparse subset of dense); identical across arms
fit       : lineage E-consistent per-column model through
            em_runner.run_em_experimental(numerics_mode="consistent",
            failure_policy="fail_fast"), L=5, num_iter=8, one model seed per
            replicate shared by its four fits
size      : 20 replicates x 2 regimes x 2 arms = exactly 80 fits

Primary   : Procrustes RMSE_Z (rotation only), delta = wrong - correct
Secondary : held-out Bernoulli plug-in mean log score (existing scorer),
            AUC / AP on the same test pairs, |w_hat - 1|, |w0_hat + 1|,
            Gram correlation, technical status / warnings / runtime.

Failure policy (research-first): a fit that raises is recorded as a failed
fit and the run continues; it is never retried, replaced or re-seeded.  A
failed preflight stops the run before any EM.

Run
---
    python expfam/src/experimental/run_misspecification_m1.py --preflight-only
    python expfam/src/experimental/run_misspecification_m1.py --execute
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
for p in (HERE, HERE.parent, ROOT / "tools" / "research_audit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_generator_canonical_mixed import generate_canonical_mixed_data  # noqa: E402
from eval_utils import make_pair_split, upper_pairs_of                   # noqa: E402
from utils_expfam import procrustes_rotation, calc_rmse                  # noqa: E402

# ── Frozen design ─────────────────────────────────────────────────────────
ISSUE = 134
N, D, K_TRUE = 75, 12, 3
K_FIT = 3
FAMILY_X_TRUE = (["gaussian"] * 3) + (["bernoulli"] * 6) + (["poisson"] * 3)
FAMILY_Y = "bernoulli"
SIGMA_X_VAR = 1.0
F_SCALE = math.sqrt(2.0)
W0_TRUE, W_TRUE = -1.0, 1.0
L, NUM_ITER = 5, 8
DATA_SEEDS = tuple(1001001 + r for r in range(20))
TEST_RATIO = 0.10
REGIMES = {"dense": 1.0, "sparse": 0.1}       # share of the training candidate pool
ARMS = {
    "correct": list(FAMILY_X_TRUE),
    "wrong": (["gaussian"] * 3) + (["poisson"] * 6) + (["poisson"] * 3),
}
BERNOULLI_COLUMNS = tuple(range(3, 9))
SPLIT_SEED_BASE = 1341000     # test mask
THIN_SEED_BASE = 1342000      # sparse subset of the pool
MODEL_SEED_BASE = 1343000     # shared by the four fits of one replicate
EXPECTED_FITS = len(DATA_SEEDS) * len(REGIMES) * len(ARMS)   # 80

PHASE9K_DIR = ROOT / "expfam" / "results" / "lap_vs_cq_20" / "phase9k_20260928"
OUT_DIR = ROOT / "expfam" / "results" / "misspecification_m1" / "phase_m1_20261013"


class PreflightStop(RuntimeError):
    """A systemic design defect.  No EM is started."""


# ── helpers ───────────────────────────────────────────────────────────────

def sha16(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=ROOT).stdout.strip()
    except Exception:
        return "unknown"


def replicate_label(r: int) -> str:
    return f"rep{r + 1:02d}"


def generate(data_seed: int):
    """One full synthetic draw with the Phase 9K generator and condition."""
    return generate_canonical_mixed_data(
        n=N, d=D, k=K_TRUE, seed=data_seed, family_x_list=list(FAMILY_X_TRUE),
        family_y=FAMILY_Y, f_scale=F_SCALE, sigma_x_var=SIGMA_X_VAR,
        w0=W0_TRUE, w=W_TRUE)


def thin_pool_mask(pool_mask: np.ndarray, keep_rate: float, seed: int) -> np.ndarray:
    """Keep a random share of the pool's upper-triangle pairs (same rule as Issue #27)."""
    if keep_rate >= 1.0:
        return pool_mask.copy()
    n = pool_mask.shape[0]
    rows, cols = upper_pairs_of(pool_mask)
    n_keep = max(1, int(round(len(rows) * keep_rate)))
    keep = np.random.default_rng(seed).permutation(len(rows))[:n_keep]
    m = np.zeros((n, n), dtype=bool)
    m[rows[keep], cols[keep]] = True
    m |= m.T
    np.fill_diagonal(m, False)
    return m


def build_masks(r: int) -> dict[str, np.ndarray]:
    pool, test = make_pair_split(N, TEST_RATIO, seed=SPLIT_SEED_BASE + r)
    masks = {"test": test, "pool": pool}
    for i, (regime, rate) in enumerate(REGIMES.items()):
        masks[regime] = thin_pool_mask(pool, rate, seed=THIN_SEED_BASE + r * 10 + i)
    return masks


def n_upper(mask: np.ndarray) -> int:
    return int(np.triu(mask, k=1).sum())


# ── preflight ─────────────────────────────────────────────────────────────

def preflight() -> dict[str, Any]:
    """The 11 preflight checks of Issue #134.  Raises PreflightStop on failure."""

    checks: dict[str, Any] = {}
    fails: list[str] = []

    def check(name: str, ok: bool, detail: Any = None):
        checks[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            fails.append(name)

    # design-level checks
    diff_cols = [l for l in range(D) if ARMS["correct"][l] != ARMS["wrong"][l]]
    check("05_family_assignment_only_B6_to_P6",
          diff_cols == list(BERNOULLI_COLUMNS)
          and all(ARMS["correct"][l] == "bernoulli" and ARMS["wrong"][l] == "poisson"
                  for l in diff_cols)
          and ARMS["correct"] == list(FAMILY_X_TRUE),
          {"changed_columns": diff_cols, "correct": ARMS["correct"], "wrong": ARMS["wrong"]})
    check("06_K_is_3_only", K_FIT == K_TRUE == 3, {"K_fit": K_FIT, "K_true": K_TRUE})
    plan = planned_fits()
    check("11_expected_fit_count_80", len(plan) == EXPECTED_FITS == 80
          and len({(p["replicate"], p["regime"], p["arm"]) for p in plan}) == 80,
          {"planned": len(plan)})

    # Phase 9K protocol identity (data condition)
    p9k = json.loads((PHASE9K_DIR / "protocol.json").read_text(encoding="utf-8"))
    same_condition = (p9k["n"] == N and p9k["d"] == D and p9k["k_true"] == K_TRUE
                      and p9k["family_x_list"] == list(FAMILY_X_TRUE)
                      and p9k["family_y"] == FAMILY_Y
                      and p9k["sigma_x_var"] == SIGMA_X_VAR
                      and p9k["f_scale"] == F_SCALE
                      and p9k["w0"] == W0_TRUE and p9k["w"] == W_TRUE
                      and [r["data_seed"] for r in p9k["replicates"]] == list(DATA_SEEDS))
    check("00_phase9k_condition_identity", same_condition)

    # Phase 9K generator provenance: column means must match bit for bit
    prov = {}
    with (PHASE9K_DIR / "generator_provenance.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            prov[(int(row["data_seed"]), int(row["column"]))] = float(row["column_mean"])

    per_rep = []
    data_ok = masks_ok = True
    prov_ok = True
    for r, seed in enumerate(DATA_SEEDS):
        # Two independent draws per replicate stand in for "what the correct
        # and the wrong arm receive": the runner hands the SAME object to all
        # four fits, and this shows the draw itself is deterministic.
        a, b = generate(seed), generate(seed)
        same = {k: bool(np.array_equal(getattr(a, k), getattr(b, k)))
                for k in ("Z", "F", "X", "Y")}
        data_ok &= all(same.values())
        for l in range(D):
            prov_ok &= float(np.mean(a.X[:, l])) == prov[(seed, l)]
        bern_binary = bool(np.all(np.isin(a.X[:, list(BERNOULLI_COLUMNS)], (0.0, 1.0))))
        data_ok &= bern_binary

        m1, m2 = build_masks(r), build_masks(r)
        test, dense, sparse = m1["test"], m1["dense"], m1["sparse"]
        rep_checks = {
            "test_identical": bool(np.array_equal(m1["test"], m2["test"])),
            "train_identical_across_arms": all(
                bool(np.array_equal(m1[g], m2[g])) for g in REGIMES),
            "sparse_subset_of_dense": bool(np.all(dense[sparse])),
            "train_test_disjoint": not bool(np.any(dense & test)) and not bool(np.any(sparse & test)),
            "pool_equals_dense": bool(np.array_equal(m1["pool"], dense)),
            "symmetric_no_diagonal": all(
                bool(np.array_equal(m, m.T)) and not bool(np.any(np.diag(m)))
                for m in (test, dense, sparse)),
        }
        masks_ok &= all(rep_checks.values())
        per_rep.append({
            "replicate": replicate_label(r), "data_seed": seed,
            "hash_dataset": sha16(a.Z, a.F, a.X, a.Y),
            "hash_test": sha16(test), "hash_dense": sha16(dense), "hash_sparse": sha16(sparse),
            "n_test_pairs": n_upper(test), "n_pool_pairs": n_upper(m1["pool"]),
            "n_dense_train_pairs": n_upper(dense), "n_sparse_train_pairs": n_upper(sparse),
            "dense_share_of_pool": n_upper(dense) / n_upper(m1["pool"]),
            "sparse_share_of_pool": n_upper(sparse) / n_upper(m1["pool"]),
            "full_y_density": float(a.Y[np.triu_indices(N, 1)].mean()),
            "test_y_density": float(a.Y[upper_pairs_of(test)].mean()),
            "bernoulli_cols_binary": bern_binary,
            **{f"same_{k}": v for k, v in same.items()},
            **rep_checks,
        })

    check("01_04_Z_F_X_Y_identical_for_both_arms", data_ok,
          "one deterministic draw per replicate shared by all four fits")
    check("00b_phase9k_generator_provenance_column_means_bitwise", prov_ok)
    check("07_10_masks", masks_ok,
          "test identical; train identical across arms; sparse subset of dense; "
          "train/test disjoint")

    # support: wrong arm must be support-compatible (validate_xy, no mismatch allowance)
    from diagnostics import validate_xy  # noqa: E402
    support_ok = True
    for r, seed in enumerate(DATA_SEEDS):
        ds = generate(seed)
        masks = build_masks(r)
        for arm, fam in ARMS.items():
            for regime in REGIMES:
                try:
                    validate_xy(ds.X, ds.Y, family_x_list=fam, family_y=FAMILY_Y,
                                train_mask=masks[regime], allow_support_mismatch=False)
                except Exception:
                    support_ok = False
    check("12_support_compatible_both_arms", support_ok)

    out = {"issue": ISSUE, "pass": not fails, "failed_checks": fails,
           "checks": checks, "replicates": per_rep}
    return out


def planned_fits() -> list[dict[str, Any]]:
    plan = []
    for r, seed in enumerate(DATA_SEEDS):
        for regime in REGIMES:
            for arm in ARMS:
                plan.append({"replicate": replicate_label(r), "r": r, "data_seed": seed,
                             "regime": regime, "arm": arm,
                             "model_seed": MODEL_SEED_BASE + r})
    return plan


# ── one fit ───────────────────────────────────────────────────────────────

def score_fit(res, ds, test_mask) -> dict[str, Any]:
    from run_heldout_k_selection_pilot import heldout_bernoulli_mean_log_score  # noqa: E402
    from sklearn.metrics import roc_auc_score, average_precision_score

    Z_est, Z_true = res["Z_est"], ds.Z
    R, k_min = procrustes_rotation(Z_est, Z_true)
    if k_min != K_TRUE:
        raise RuntimeError(f"k_min={k_min} != {K_TRUE}")
    rmse_z = float(calc_rmse(Z_true[:, :k_min], Z_est[:, :k_min] @ R))
    iu = np.triu_indices(N, 1)
    gram_corr = float(np.corrcoef((Z_est @ Z_est.T)[iu], (Z_true @ Z_true.T)[iu])[0, 1])

    rows, cols = upper_pairs_of(test_mask)
    eta = res["w0"] + res["w"] * np.sum(Z_est[rows] * Z_est[cols], axis=1)
    y = ds.Y[rows, cols]
    p = 1.0 / (1.0 + np.exp(-eta))
    return {
        "rmse_Z": rmse_z, "gram_corr": gram_corr,
        "heldout_mean_log_score": heldout_bernoulli_mean_log_score(y, eta),
        "heldout_auc": float(roc_auc_score(y, p)),
        "heldout_ap": float(average_precision_score(y, p)),
        "w0_hat": float(res["w0"]), "w_hat": float(res["w"]),
        "abs_w0_err": abs(float(res["w0"]) - W0_TRUE),
        "abs_w_err": abs(float(res["w"]) - W_TRUE),
        "z_est_norm_ratio": float(np.linalg.norm(Z_est) / np.linalg.norm(Z_true)),
    }


def run_fit(fit: dict[str, Any], ds, masks) -> dict[str, Any]:
    from em_runner import run_em_experimental  # noqa: E402

    row = {k: fit[k] for k in ("replicate", "data_seed", "regime", "arm", "model_seed")}
    row.update({"K_fit": K_FIT, "family_x_list": "|".join(ARMS[fit["arm"]]),
                "n_train_pairs": n_upper(masks[fit["regime"]]),
                "n_test_pairs": n_upper(masks["test"])})
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        try:
            res = run_em_experimental(
                ds.X, ds.Y, family_x=None, family_y=FAMILY_Y, k=K_FIT, L=L,
                num_iter=NUM_ITER, seed=fit["model_seed"], train_mask=masks[fit["regime"]],
                family_x_list=list(ARMS[fit["arm"]]), compute_strict_Q=False,
                validate_support=True, allow_support_mismatch=False,
                numerics_mode="consistent", failure_policy="fail_fast")
            metrics = score_fit(res, ds, masks["test"])
            finite = all(np.isfinite(v) for v in metrics.values())
            row.update(metrics)
            row.update({"status": "OK" if finite else "NONFINITE_METRIC",
                        "error": "", "retry_count": res["retry_count"],
                        "replacement_count": res["replacement_count"],
                        "seed_rescue_count": res["seed_rescue_count"],
                        "nan_occurred": bool(res["nan_occurred"]),
                        "numerics_mode": res["numerics_mode"],
                        "em_runtime_s": res["runtime_s"]})
        except Exception as exc:          # recorded, never retried or replaced
            row.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"[:500],
                        "retry_count": 0, "replacement_count": 0, "seed_rescue_count": 0,
                        "nan_occurred": "", "numerics_mode": "consistent",
                        "em_runtime_s": ""})
        row["n_warnings"] = len(wl)
        row["warning_messages"] = " || ".join(sorted({f"{w.category.__name__}: {str(w.message)[:160]}"
                                                       for w in wl}))[:1000]
    row["wall_runtime_s"] = round(time.perf_counter() - t0, 2)
    return row


# ── analysis ──────────────────────────────────────────────────────────────

def dist(values) -> dict[str, Any]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0}
    q1, q3 = np.quantile(v, [0.25, 0.75])
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
            "min": float(v.min()), "max": float(v.max())}


def analyze(rows: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    ok = {(r["replicate"], r["regime"], r["arm"]): r for r in rows if r["status"] == "OK"}
    reps = [replicate_label(r) for r in range(len(DATA_SEEDS))]

    paired, heldout, params = [], [], []
    for rep in reps:
        rec = {"replicate": rep}
        for regime in REGIMES:
            c, w = ok.get((rep, regime, "correct")), ok.get((rep, regime, "wrong"))
            rec[f"rmse_Z_correct_{regime}"] = c["rmse_Z"] if c else None
            rec[f"rmse_Z_wrong_{regime}"] = w["rmse_Z"] if w else None
            rec[f"delta_RMSE_Z_{regime}"] = (w["rmse_Z"] - c["rmse_Z"]) if (c and w) else None
            for arm, f in (("correct", c), ("wrong", w)):
                if f:
                    heldout.append({"replicate": rep, "regime": regime, "arm": arm,
                                    "n_test_pairs": f["n_test_pairs"],
                                    "heldout_mean_log_score": f["heldout_mean_log_score"],
                                    "heldout_auc": f["heldout_auc"], "heldout_ap": f["heldout_ap"]})
                    params.append({"replicate": rep, "regime": regime, "arm": arm,
                                   "w0_hat": f["w0_hat"], "w_hat": f["w_hat"],
                                   "abs_w0_err": f["abs_w0_err"], "abs_w_err": f["abs_w_err"],
                                   "gram_corr": f["gram_corr"]})
        dd, ds_ = rec["delta_RMSE_Z_dense"], rec["delta_RMSE_Z_sparse"]
        rec["interaction_delta"] = (ds_ - dd) if (dd is not None and ds_ is not None) else None
        paired.append(rec)

    write_csv(out / "paired_rmse_z.csv", paired)
    write_csv(out / "heldout_y.csv", heldout)
    write_csv(out / "parameter_errors.csv", params)

    def paired_metric(metric, regime, sign=1.0):
        d = []
        for rep in reps:
            c, w = ok.get((rep, regime, "correct")), ok.get((rep, regime, "wrong"))
            if c and w:
                d.append(sign * (w[metric] - c[metric]))
        return d

    summary = {"issue": ISSUE, "planned_fits": EXPECTED_FITS, "attempted_fits": len(rows),
               "successful_fits": sum(r["status"] == "OK" for r in rows),
               "failed_fits": [{k: r[k] for k in ("replicate", "regime", "arm", "status", "error")}
                               for r in rows if r["status"] != "OK"],
               "retries": sum(int(r["retry_count"] or 0) for r in rows),
               "replacements": sum(int(r["replacement_count"] or 0) for r in rows),
               "seed_rescues": sum(int(r["seed_rescue_count"] or 0) for r in rows),
               "total_warnings": sum(int(r["n_warnings"]) for r in rows),
               "K_values": sorted({int(r["K_fit"]) for r in rows}),
               "family_auto_selection": False, "regimes": {}}
    for regime in REGIMES:
        deltas = [p[f"delta_RMSE_Z_{regime}"] for p in paired if p[f"delta_RMSE_Z_{regime}"] is not None]
        summary["regimes"][regime] = {
            "rmse_Z_correct": dist([p[f"rmse_Z_correct_{regime}"] for p in paired]),
            "rmse_Z_wrong": dist([p[f"rmse_Z_wrong_{regime}"] for p in paired]),
            "delta_RMSE_Z": dist(deltas),
            "wrong_gt_correct": int(sum(d > 0 for d in deltas)),
            "wrong_lt_correct": int(sum(d < 0 for d in deltas)),
            "n_pairs": len(deltas),
            # secondary: positive = wrong worse
            "delta_heldout_log_score_correct_minus_wrong": dist(paired_metric("heldout_mean_log_score", regime, -1.0)),
            "delta_heldout_auc_correct_minus_wrong": dist(paired_metric("heldout_auc", regime, -1.0)),
            "delta_heldout_ap_correct_minus_wrong": dist(paired_metric("heldout_ap", regime, -1.0)),
            "delta_abs_w_err_wrong_minus_correct": dist(paired_metric("abs_w_err", regime)),
            "delta_abs_w0_err_wrong_minus_correct": dist(paired_metric("abs_w0_err", regime)),
            "delta_gram_corr_correct_minus_wrong": dist(paired_metric("gram_corr", regime, -1.0)),
            "heldout_log_score_correct": dist([h["heldout_mean_log_score"] for h in heldout
                                               if h["regime"] == regime and h["arm"] == "correct"]),
            "heldout_log_score_wrong": dist([h["heldout_mean_log_score"] for h in heldout
                                             if h["regime"] == regime and h["arm"] == "wrong"]),
            "heldout_auc_correct": dist([h["heldout_auc"] for h in heldout if h["regime"] == regime and h["arm"] == "correct"]),
            "heldout_auc_wrong": dist([h["heldout_auc"] for h in heldout if h["regime"] == regime and h["arm"] == "wrong"]),
            "abs_w_err_correct": dist([p["abs_w_err"] for p in params if p["regime"] == regime and p["arm"] == "correct"]),
            "abs_w_err_wrong": dist([p["abs_w_err"] for p in params if p["regime"] == regime and p["arm"] == "wrong"]),
            "abs_w0_err_correct": dist([p["abs_w0_err"] for p in params if p["regime"] == regime and p["arm"] == "correct"]),
            "abs_w0_err_wrong": dist([p["abs_w0_err"] for p in params if p["regime"] == regime and p["arm"] == "wrong"]),
        }
    inter = [p["interaction_delta"] for p in paired if p["interaction_delta"] is not None]
    summary["interaction_delta"] = {**dist(inter), "positive": int(sum(x > 0 for x in inter)),
                                    "negative": int(sum(x < 0 for x in inter))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def protocol() -> dict[str, Any]:
    return {
        "issue": ISSUE, "experiment": "M1 Bernoulli->Poisson X misspecification",
        "lineage": "E-consistent per-column (experimental prototype; not adoptable for the manuscript)",
        "generator": "canonical-clean-mixed-v1 (data_generator_canonical_mixed.generate_canonical_mixed_data)",
        "n": N, "d": D, "K_true": K_TRUE, "K_fit": K_FIT,
        "family_x_true": list(FAMILY_X_TRUE), "family_y": FAMILY_Y,
        "sigma_x_var": SIGMA_X_VAR, "f_scale": F_SCALE, "w0_true": W0_TRUE, "w_true": W_TRUE,
        "data_seeds": list(DATA_SEEDS), "arms": ARMS,
        "regimes_share_of_training_candidate_pool": REGIMES,
        "test_ratio_of_all_upper_pairs": TEST_RATIO,
        "seeds": {"split_seed_base": SPLIT_SEED_BASE, "thin_seed_base": THIN_SEED_BASE,
                  "model_seed_base": MODEL_SEED_BASE,
                  "rule": "split = base + r; thin = base + 10 r + regime_index; "
                          "model = base + r (shared by all four fits of replicate r)"},
        "em": {"runner": "em_runner.run_em_experimental", "numerics_mode": "consistent",
               "failure_policy": "fail_fast", "L": L, "num_iter": NUM_ITER,
               "compute_strict_Q": False, "validate_support": True,
               "allow_support_mismatch": False},
        "expected_fits": EXPECTED_FITS,
        "primary": "Procrustes RMSE_Z (rotation only); delta_RMSE_Z = wrong - correct",
        "interaction": "interaction_delta = delta_RMSE_Z_sparse - delta_RMSE_Z_dense",
        "secondary": ["held-out Bernoulli plug-in mean log score (run_heldout_k_selection_pilot."
                      "heldout_bernoulli_mean_log_score)", "held-out AUC / AP (sklearn)",
                      "|w_hat - 1|", "|w0_hat + 1|", "Gram correlation", "status / warnings / runtime"],
        "x_metric": "not computed (no cross-family raw likelihood ranking)",
        "forbidden": ["K sweep", "family auto-selection", "extra Y rates", "extra arms",
                      "extra seeds / replacement", "retry", "result-dependent metrics"],
        "success_criterion": "none on the sign of delta; any direction is a valid result",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight-only", action="store_true")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pre = preflight()
    (out / "preflight.json").write_text(json.dumps(pre, indent=2), encoding="utf-8")
    write_csv(out / "mask_provenance.csv", pre["replicates"])
    print(f"preflight: {'PASS' if pre['pass'] else 'FAIL'} {pre['failed_checks']}")
    if not pre["pass"]:
        raise PreflightStop(f"preflight failed: {pre['failed_checks']}")
    if a.preflight_only:
        return 0

    proto = protocol()
    proto["git_head"] = git("rev-parse", "HEAD")
    proto["git_dirty_tracked"] = bool(git("status", "--porcelain", "--untracked-files=no"))
    (out / "protocol.json").write_text(json.dumps(proto, indent=2), encoding="utf-8")

    t0 = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, cache = [], {}
    plan = planned_fits()
    assert len(plan) == EXPECTED_FITS
    for i, fit in enumerate(plan, 1):
        if fit["r"] not in cache:
            cache.clear()
            cache[fit["r"]] = (generate(fit["data_seed"]), build_masks(fit["r"]))
        ds, masks = cache[fit["r"]]
        row = run_fit(fit, ds, masks)
        row["hash_dataset"] = sha16(ds.Z, ds.F, ds.X, ds.Y)
        row["hash_test"] = sha16(masks["test"])
        row["hash_train"] = sha16(masks[fit["regime"]])
        rows.append(row)
        print(f"[{i:2d}/{EXPECTED_FITS}] {fit['replicate']} {fit['regime']:6s} {fit['arm']:7s} "
              f"{row['status']} rmse_Z={row.get('rmse_Z', float('nan')):.4f} "
              f"t={row['wall_runtime_s']}s", flush=True)
        write_csv(out / "fit_results.csv", rows)       # incremental, never rewritten selectively

    tech = [{k: r.get(k, "") for k in ("replicate", "regime", "arm", "status", "error",
                                        "retry_count", "replacement_count", "seed_rescue_count",
                                        "nan_occurred", "numerics_mode", "n_warnings",
                                        "warning_messages", "em_runtime_s", "wall_runtime_s",
                                        "hash_dataset", "hash_test", "hash_train")} for r in rows]
    write_csv(out / "technical_diagnostics.csv", tech)
    summary = analyze(rows, out)
    runinfo = {"issue": ISSUE, "script": "expfam/src/experimental/run_misspecification_m1.py",
               "git_head": proto["git_head"], "git_branch": git("branch", "--show-current"),
               "git_dirty_tracked": proto["git_dirty_tracked"], "started_utc": started,
               "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "total_runtime_s": round(time.perf_counter() - t0, 1),
               "python": platform.python_version(), "numpy": np.__version__,
               "attempted_fits": len(rows), "successful_fits": summary["successful_fits"]}
    (out / "runinfo.json").write_text(json.dumps(runinfo, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("attempted_fits", "successful_fits", "retries",
                                              "replacements", "K_values")}, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PreflightStop as exc:
        print(f"STOPPED BEFORE EM: {exc}")
        sys.exit(2)
