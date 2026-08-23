"""
K-selection score lineage / diagnostic pilot (Issue #37, Phase 7b).

Purpose
-------
Compute three candidate dimension-selection scores ON THE SAME FITS and record the
previously unmeasured quantity `(1/L) sum_l sum_i log det A_i^{post,(l)}`.
This is a DIAGNOSTIC pilot:
no hypothesis test, no primary estimand, no consistency claim, no "correct score" verdict.

Quantities that must never be conflated (Phase 7a / Issue #37)
--------------------------------------------------------------
Q1  log p(X,Y | Z,theta)      paper Eq.(16). Conditional / plug-in; Z is NOT integrated out.
Q2  log p(Z,X,Y | theta)      what the current Q_strict averages under the sampling procedure.
Q3  log p(X,Y | theta)        = log int p(Z,X,Y|theta) dZ. Observed-data marginal likelihood;
                              the quantity Schwarz BIC concerns in the usual sense.

NONE of the three scores below is Q3.

Candidate scores
----------------
C1  bic_impl       = -2*Q_strict + p_hat*log(n)
                     Existing implementation, value unchanged. Positioning: DESCRIPTIVE BASELINE.

C2  scf            = -2*(Q_strict - lnpZ_det) + p_hat*log(n)
      lnpZ_det     = -(n*k/2)*(1 + log(2*pi))
                     Positioning: COUNTERFACTUAL DIAGNOSTIC SCORE.
                     Its fit-term components align with paper Eq.(16), but it is NOT a
                     reproduction of paper Eq.(26) and it is NOT Q3.

C3  s_laplace_post = scf + (1/L) sum_l sum_i log det A_i^{post,(l)}
                     Positioning: POST-HOC LAPLACE-CURVATURE DIAGNOSTIC.

Naming prohibitions (enforced in the report, restated here so the code is self-documenting)
-------------------------------------------------------------------------------------------
  * `scf` is NEVER "corrected BIC", "modified BIC", "true BIC", "Schwarz BIC",
    or "a reproduction of the paper's BIC".
  * `s_laplace_post` is NEVER "ELBO", "variational BIC", "corrected BIC",
    or "marginal likelihood" / "a lower bound on the marginal likelihood".
    `A_i^{post}` is recomputed post hoc from the FINAL, scale_Z-applied samples; it is not
    the precision matrix used at sampling time, and the Gaussian entropy surrogate built
    from it is not the entropy of the actual algorithmic law (Phase 7a audit 6.5.1).

Implementation lineage
----------------------
`DualExpFamLSMConsistent` lives under `expfam/src/experimental/` and is, per root
CLAUDE.md section 3, PROTOTYPE / NOT MANUSCRIPT-APPROVED. Phase 7b output is diagnostic
evidence only. Numeric values from the old-0.5 / fixed / consistent lineages must never be
placed in the same table or figure (KI-002).

No model code is modified. Every quantity beyond the runner's own return keys is computed
post hoc in this file.

Integrity policy
----------------
Any of the following STOPS the run; the affected fit is never used, and no seed is changed,
dropped, or retried:
  internal retry detected / nan_occurred / q_bic_failed / numerics_mode mismatch /
  wrong fit count / duplicate fit key / X,Y hash mismatch within an (n, trial) /
  lnpZ_abs_err >= 1e-6 / non-finite required metric.
`slogdet` sign violations are BLOCKING (gate G6). C3 = S_cf + (1/L) sum_l sum_i
logdet(A_i^post,(l)), so a
single A_i with slogdet sign <= 0 means logdet is not a real number there and C3 is not a
real-valued diagnostic for that fit: the fit FAILS. It is never rescued with logabsdet, with
jitter, with a symmetrised-matrix determinant, or by dropping the offending node.

An internal NaN reset / retry is likewise a HARD FAILURE (gate G8): such a fit is not used
even when it returns finite values.

`run_em_experimental` retries internally with a halved Newton alpha and resets `nan_count`
per retry, so `res["nan_occurred"]` can be False even after a NaN reset actually happened,
and the runner does NOT return a retry counter. We therefore capture the runner's stdout
with verbose=True and detect the literal "[NaN iter=" it prints on every reset. This is the
existing repository convention (see run_complementary_blocks_consistent.py).

Run
---
    python tools/research_audit/run_k_selection_score_pilot.py --validate-only
    python tools/research_audit/run_k_selection_score_pilot.py --postprocess-only
    python tools/research_audit/run_k_selection_score_pilot.py --smoke
    python tools/research_audit/run_k_selection_score_pilot.py --full

Exactly one mode flag is required. With no flag the script prints usage and exits without
fitting anything.
"""

import argparse
import contextlib
import hashlib
import io
import json
import re
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

from utils_expfam import procrustes_rotation, calc_rmse, _lnpZ      # noqa: E402
from data_generator_expfam import generate_dual_data                # noqa: E402
from em_runner import run_em_experimental                           # noqa: E402

# -- Fixed design (pre-registered; DO NOT CHANGE AFTER SEEING RESULTS) -----
SCENARIO = "A"
FAMILY_X = "poisson"
FAMILY_Y = "bernoulli"
K_TRUE = 3
K_EST_LIST = [1, 2, 3, 4, 5, 6, 7]
N_LIST = [75, 150]
D = 15
TRIALS = [1, 2, 3]              # 1-BASED. Recorded in runinfo as trial_convention.
L = 5
NITER = 8
NUMERICS_MODE = "consistent"
IMPLEMENTATION_LINEAGE = (
    "objective-consistent (expfam/src/experimental/model_dual_expfam_consistent.py, "
    "DualExpFamLSMConsistent) - PROTOTYPE, NOT manuscript-approved (root CLAUDE.md 3)"
)

# Smoke subset
SMOKE_N = 75
SMOKE_TRIAL = TRIALS[0]         # = 1
SMOKE_K_EST = [2, 3, 4]

SMOKE_FIT_COUNT = 3
FULL_FIT_COUNT = 42

# Seeds (Issue #37 convention; trial is 1-based here)
SEED_DATA_BASE = 150000
SEED_MODEL_BASE = 151000
N_INDEX = {75: 0, 150: 1}

# Pre-fixed tolerances
RANK_F_TOL = 1e-8
LNPZ_TOL = 1e-6
NAN_FIT_FRACTION_MAX = 0.05

STEM_PREFIX = "k_selection_score_pilot"
OUT_DIR = _ROOT / "expfam" / "results" / "k_selection"
FIG_DIR = _ROOT / "figures" / "k_selection"
RUNNER_REL_PATH = Path("tools/research_audit/run_k_selection_score_pilot.py")
EXPECTED_FULL_BRANCH = "experiment/37-k-selection-score-pilot"
HASH_VERSION = "sha256-full-shape-dtype-contiguous-bytes-v2"

TWO_PI = 2.0 * np.pi


class ExperimentStop(Exception):
    """Integrity violation. The run stops and the affected result is never used."""

    def __init__(self, message, *, failure_stage=None, failure_type=None,
                 warning_records=None, retry_info=None):
        super().__init__(message)
        self.failure_stage = failure_stage
        self.failure_type = failure_type
        self.warning_records = warning_records
        self.retry_info = retry_info


# -- Helpers ---------------------------------------------------------------

def _sha256_arrays(*arrays):
    """Full SHA-256 over shape + dtype + bytes. Never Python's built-in hash()."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _sha256_file(path):
    """Full-content SHA-256 for provenance."""
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_git(args):
    cp = subprocess.run(["git", *args], capture_output=True, text=True,
                        cwd=_ROOT)
    if cp.returncode != 0:
        raise ExperimentStop(
            f"git {' '.join(args)} failed: {cp.stderr.strip() or cp.stdout.strip()}",
            failure_stage="repository_preflight", failure_type="GitCommandError")
    return cp.stdout.strip()


def _git_head():
    try:
        return _run_git(["rev-parse", "HEAD"])
    except Exception:
        return "unknown"


def _git_branch():
    try:
        return _run_git(["branch", "--show-current"])
    except Exception:
        return "unknown"


def _git_provenance():
    """Snapshot repository and runner provenance before any output is written."""
    status = _run_git(["status", "--porcelain", "--untracked-files=all"])
    runner_abs = (_ROOT / RUNNER_REL_PATH).resolve()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", RUNNER_REL_PATH.as_posix()],
        capture_output=True, text=True, cwd=_ROOT).returncode == 0
    return {
        "git_head": _run_git(["rev-parse", "HEAD"]),
        "git_branch": _run_git(["branch", "--show-current"]),
        "git_is_dirty": bool(status),
        "git_status_porcelain": json.dumps(status.splitlines(), ensure_ascii=False),
        "runner_path": RUNNER_REL_PATH.as_posix(),
        "runner_sha256": _sha256_file(runner_abs),
        "runner_is_tracked": tracked,
    }


def validate_full_repository_state(provenance=None):
    """Full fits are permitted only from the committed, clean review branch."""
    p = provenance or _git_provenance()
    problems = []
    if not p["runner_is_tracked"]:
        problems.append(f"runner is not git tracked: {p['runner_path']}")
    if p["git_is_dirty"]:
        problems.append("working tree is dirty: " + p["git_status_porcelain"])
    if p["git_branch"] != EXPECTED_FULL_BRANCH:
        problems.append(
            f"current branch {p['git_branch']!r} != {EXPECTED_FULL_BRANCH!r}")
    if problems:
        raise ExperimentStop(
            "full repository preflight failed before data generation / EM:\n  "
            + "\n  ".join(problems), failure_stage="repository_preflight",
            failure_type="FullRepositoryPreflightError")
    return p


def seed_data_of(n, trial):
    return SEED_DATA_BASE + 100 * N_INDEX[n] + trial


def seed_model_of(n, trial, k_est):
    return SEED_MODEL_BASE + 100 * N_INDEX[n] + 10 * trial + k_est


def lnpZ_det_expected(n, k):
    """Analytic value of the deterministic Z-prior term under scale_Z with var_z = 1.

    lnpZ_det = -(n*k/2) * (1 + log(2*pi))

    This is the (Q2)-only term that C2 removes. It is data- and theta-independent.
    """
    return -(n * k / 2.0) * (1.0 + np.log(TWO_PI))


def lnpZ_observed_of(Z_samples, var_z):
    """(1/L) sum_l log p(Z^(l)) on the FINAL, scale_Z-applied samples.

    Reuses utils_expfam._lnpZ so the formula is not re-implemented here.
    """
    _, _, n_samples = Z_samples.shape
    return float(np.mean([_lnpZ(Z_samples[:, :, l], var_z) for l in range(n_samples)]))


def scf_of(q_strict, lnpz_det, num_params, n):
    """C2. Counterfactual diagnostic score. NOT a corrected / Schwarz / paper BIC."""
    return -2.0 * (q_strict - lnpz_det) + num_params * np.log(n)


def s_laplace_post_of(scf, sum_log_det_a_post):
    """C3. Post-hoc Laplace-curvature diagnostic. NOT an ELBO, NOT a marginal likelihood."""
    return scf + sum_log_det_a_post


def build_manifest(mode):
    """Deterministic fit manifest. No data generation, no fitting."""
    rows = []
    if mode == "smoke":
        combos = [(SMOKE_N, SMOKE_TRIAL, k) for k in SMOKE_K_EST]
    else:
        combos = [(n, t, k) for n in N_LIST for t in TRIALS for k in K_EST_LIST]
    for n, trial, k_est in combos:
        rows.append({
            "scenario": SCENARIO, "n": n, "d": D, "K_TRUE": K_TRUE,
            "trial": trial, "k_est": k_est,
            "family_x": FAMILY_X, "family_y": FAMILY_Y,
            "L": L, "num_iter": NITER,
            "numerics_mode": NUMERICS_MODE,
            "seed_data": seed_data_of(n, trial),
            "seed_model": seed_model_of(n, trial, k_est),
        })
    return pd.DataFrame(rows)


def validate_manifest(man, mode):
    """Gates G3 / G4 and the fit-count assertion. Pure, no fitting."""
    problems = []

    expected = SMOKE_FIT_COUNT if mode == "smoke" else FULL_FIT_COUNT
    if len(man) != expected:
        problems.append(f"fit count {len(man)} != expected {expected} for mode={mode}")

    # G4: no duplicate fit key (n, trial, k_est)
    keys = list(zip(man["n"], man["trial"], man["k_est"]))
    if len(set(keys)) != len(keys):
        dup = [k for k in set(keys) if keys.count(k) > 1]
        problems.append(f"G4 duplicate fit key (n, trial, k_est): {dup}")

    # G3: seed collisions. Model seeds must be unique per fit; data seeds must be
    # constant within an (n, trial) and distinct across (n, trial); the two seed
    # families must not overlap.
    sm = list(man["seed_model"])
    if len(set(sm)) != len(sm):
        problems.append("G3 seed_model collision")

    dmap = {}
    for _, r in man.iterrows():
        dmap.setdefault((r["n"], r["trial"]), set()).add(int(r["seed_data"]))
    for key, s in dmap.items():
        if len(s) != 1:
            problems.append(f"G3 seed_data varies within (n,trial)={key}: {sorted(s)}")
    flat = [next(iter(s)) for s in dmap.values()]
    if len(set(flat)) != len(flat):
        problems.append("G3 seed_data collision across (n,trial)")

    overlap = set(man["seed_data"]) & set(man["seed_model"])
    if overlap:
        problems.append(f"G3 seed_data / seed_model overlap: {sorted(overlap)}")

    # seed_data must not depend on k_est
    for (n, trial), grp in man.groupby(["n", "trial"]):
        if grp["seed_data"].nunique() != 1:
            problems.append(f"seed_data depends on k_est at (n={n}, trial={trial})")

    return problems


def seed_manifests(man):
    """Concrete data/model seed manifests recorded in runinfo."""
    data_rows = (man[["n", "trial", "seed_data"]]
                 .drop_duplicates().sort_values(["n", "trial"]))
    model_rows = man[["n", "trial", "k_est", "seed_model"]].sort_values(
        ["n", "trial", "k_est"])
    return (
        json.dumps(data_rows.astype(int).to_dict("records"), separators=(",", ":")),
        json.dumps(model_rows.astype(int).to_dict("records"), separators=(",", ":")),
    )


def a_post_diagnostics(model, Z_samples, F, sigma, var_z, w0, w):
    """Post-hoc curvature diagnostics on the FINAL, scale_Z-applied samples.

    `A_i^{post,(l)} := model._calc_precision_matrix(Z_samples[:,:,l], F, sigma,
                                                    var_z, w0, w, i)`

    This is the EXISTING method (inherited by DualExpFamLSMConsistent from
    DualExpFamLSMMasked, with the consistent variance functions supplied by
    _ObjectiveConsistentYMixin / the consistent _variance_function_x). No formula is
    reimplemented here, and no jitter or symmetrisation is applied before slogdet -
    the E-step's own `A_i + 1e-6*I` regularisation is NOT reproduced, because this is a
    post-hoc re-evaluation, not a reconstruction of the sampling-time matrix.

    Averaging convention (Phase 7a audit 6.5.2, Issue #37):
        sum over nodes i WITHIN each sample l, then average over the L samples.

        sum_log_det_A_post = (1/L) * sum_l ( sum_i logdet(A_i^{post,(l)}) )

    Eigenvalues are taken with eigvalsh on the symmetrised copy (A + A.T)/2, which is
    exact up to floating point since A is symmetric by construction; `max_abs_asym` is
    recorded so any departure is visible rather than hidden. slogdet uses the RAW matrix.
    """
    n, _, n_samples = Z_samples.shape
    per_sample_sums, all_logdets, signs = [], [], []
    min_eig, max_eig, max_asym = np.inf, -np.inf, 0.0

    for l in range(n_samples):
        Z_l = Z_samples[:, :, l]
        s_l = 0.0
        for i in range(n):
            A = model._calc_precision_matrix(Z_l, F, sigma, var_z, w0, w, i)
            sign, logdet = np.linalg.slogdet(A)
            signs.append(float(sign))
            all_logdets.append(float(logdet))
            s_l += float(logdet)
            max_asym = max(max_asym, float(np.max(np.abs(A - A.T))))
            ev = np.linalg.eigvalsh((A + A.T) / 2.0)
            min_eig = min(min_eig, float(ev.min()))
            max_eig = max(max_eig, float(ev.max()))
        per_sample_sums.append(s_l)

    sum_log_det = float(np.mean(per_sample_sums))
    return {
        "sum_log_det_A_post": sum_log_det,
        "slogdet_sign_all_pos": bool(np.all(np.asarray(signs) > 0)),
        "n_slogdet_sign_violations": int(np.sum(np.asarray(signs) <= 0)),
        "min_eig_A": min_eig,
        "max_eig_A": max_eig,
        "mean_log_det_A_per_node": float(np.mean(all_logdets)),
        "max_abs_asym_A": max_asym,
        "n_A_evaluated": int(n * n_samples),
    }


# -- One fit ---------------------------------------------------------------

def run_one_fit(X, Y, k_est, seed_model, key):
    """Run one fit with stdout capture (internal-retry detection) and warning capture."""
    buf = io.StringIO()
    fit_exc = None
    res = None
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            with contextlib.redirect_stdout(buf):
                res = run_em_experimental(
                    X, Y,
                    family_x=FAMILY_X, family_y=FAMILY_Y, k=k_est,
                    L=L, num_iter=NITER, seed=seed_model,
                    compute_strict_Q=True,
                    mstep_q_diagnostic=True,
                    numerics_mode=NUMERICS_MODE,
                    verbose=True,
                )
        except Exception as exc:
            fit_exc = exc
        wrecs = [{"category": w.category.__name__, "message": str(w.message)}
                 for w in wlist]
    captured = buf.getvalue()
    retry = parse_internal_retry(captured)
    if fit_exc is not None:
        raise ExperimentStop(
            f"{key}: fit raised {type(fit_exc).__name__}: {fit_exc}\n"
            f"captured stdout tail:\n{captured[-800:]}",
            failure_stage="fit", failure_type=type(fit_exc).__name__,
            warning_records=wrecs, retry_info=retry) from fit_exc
    return res, captured, retry, wrecs


def parse_internal_retry(captured):
    """Detect and describe the runner's internal NaN reset / retry.

    `run_em_experimental` prints exactly

        f"  [NaN iter={iteration} retry={retry}] Resetting."   (em_runner.py L.222)

    on every reset, resets `nan_count` per retry, and returns no retry counter, so the
    return dict alone CANNOT reveal that a retry happened: a run that succeeded on the
    final retry reports nan_occurred=False. Parsing this line is the only available
    signal, and em_runner.py is NOT modified.

    Returns a dict; `internal_retry` is the authoritative boolean. If the marker is
    present but the indices cannot be parsed, `internal_retry` stays True and the
    indices are left empty - a parse failure never downgrades the flag.
    """
    lines = [ln for ln in captured.splitlines() if "[NaN iter=" in ln]
    idx = []
    for ln in lines:
        m = re.search(r"\[NaN iter=(\d+) retry=(\d+)\]", ln)
        if m:
            idx.append(int(m.group(2)))
    return {
        "internal_retry": bool(lines),
        "internal_retry_detected": bool(lines),
        "internal_retry_indices": sorted(set(idx)),
        "internal_retry_max_index": (max(idx) if idx else -1),
        "internal_retry_n_resets": len(lines),
        "internal_retry_lines": lines,
    }


def check_fit(key, res, retry_info, captured):
    # Hard failure. A fit whose internal state was reset mid-run is NOT used in
    # Phase 7b even though it returned finite values, and the seed is NOT changed.
    if retry_info["internal_retry"]:
        raise ExperimentStop(
            f"{key}: internal NaN reset / retry detected in the runner "
            f"(n_resets={retry_info['internal_retry_n_resets']}, "
            f"retry_indices={retry_info['internal_retry_indices']}, "
            f"nan_occurred={res['nan_occurred']}, nan_count={res['nan_count']}). "
            "Result is NOT used and the seed is NOT changed.\n"
            + "\n".join(retry_info["internal_retry_lines"]),
            failure_stage="internal_retry_gate",
            failure_type="InternalRetryDetected")
    if res["nan_occurred"]:
        raise ExperimentStop(
            f"{key}: nan_occurred=True (nan_count={res['nan_count']})",
            failure_stage="nan_gate", failure_type="NaNAffectedFit")
    if res.get("q_bic_failed"):
        raise ExperimentStop(
            f"{key}: q_bic_failed=True, failure_reason={res.get('failure_reason')!r}",
            failure_stage="q_bic_gate", failure_type="QBICFailure")
    # G2
    if res.get("numerics_mode") != NUMERICS_MODE:
        raise ExperimentStop(
            f"{key}: G2 numerics_mode={res.get('numerics_mode')!r} != {NUMERICS_MODE!r}",
            failure_stage="numerics_mode_gate",
            failure_type="NumericsModeMismatch")


def check_curvature(key, row):
    """G6, blocking.

    C3 = S_cf + (1/L) sum_l sum_i logdet(A_i^post,(l)). A single A_i with
    slogdet sign <= 0 means
    logdet is not a real number there, so C3 is not a usable real-valued diagnostic
    for this fit. The fit FAILS. It is not rescued by substituting logabsdet, by
    adding jitter, by using the symmetrised matrix's determinant, or by dropping the
    offending node - all of those would silently change the quantity being reported.
    """
    if not row["slogdet_sign_all_pos"]:
        raise ExperimentStop(
            f"{key}: G6 slogdet sign <= 0 for "
            f"{row['n_slogdet_sign_violations']} of {row['n_A_evaluated']} "
            f"A_i^post evaluations (min_eig_A={row['min_eig_A']:.6e}). "
            "S_laplace_post is not a real-valued diagnostic for this fit. "
            "No logabsdet substitution, no jitter, no symmetrised-determinant "
            "replacement, no silent drop, no seed change.",
            failure_stage="curvature_gate",
            failure_type="NonPositiveSlogdetSign")


REQUIRED_FINITE_METRICS = (
    "q_strict", "bic_impl", "num_params",
    "lnpZ_det_expected", "lnpZ_observed", "lnpZ_abs_err",
    "scf", "sum_log_det_A_post", "s_laplace_post",
    "min_eig_A", "max_eig_A", "mean_log_det_A_per_node",
    "max_abs_asym_A", "rank_F", "rmse_Z", "rmse_X", "runtime_s",
    "mstep_q_n_iters", "mstep_q_n_decreased", "mstep_q_min_diff",
    "nan_count", "n_slogdet_sign_violations", "n_A_evaluated",
)


def check_required_finite(key, row):
    """Reject every non-finite required scalar before fit acceptance."""
    missing = [name for name in REQUIRED_FINITE_METRICS if name not in row]
    bad = []
    for name in REQUIRED_FINITE_METRICS:
        if name in row:
            try:
                finite = bool(np.isfinite(row[name]))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                bad.append(f"{name}={row[name]!r}")
    if missing or bad:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if bad:
            parts.append("non-finite: " + ", ".join(bad))
        raise ExperimentStop(
            f"{key}: required numeric metric gate failed ({'; '.join(parts)})",
            failure_stage="finite_metric_gate",
            failure_type="NonFiniteRequiredMetric")


def check_lnpz_gate(key, row):
    err = row["lnpZ_abs_err"]
    if not (np.isfinite(err) and err < LNPZ_TOL):
        raise ExperimentStop(
            f"{key}: G1 requires finite lnpZ_abs_err < {LNPZ_TOL}; got {err!r}",
            failure_stage="lnpz_gate", failure_type="LnpZIntegrityError")


def score_one_fit(res, X, Z_true, n, k_est):
    """All post-hoc quantities. The model/runner is not modified."""
    model = res["model"]
    Z_samples = res["Z_samples"]      # already scale_Z-applied by em_runner (L.226)
    F, sigma = res["F"], res["sigma"]
    var_z, w0, w = res["var_z"], res["w0"], res["w"]

    q_strict = float(res["Q_strict"])
    bic_impl = float(res["bic"])              # C1, unchanged existing value
    num_params = int(res["num_params"])

    lnpz_exp = lnpZ_det_expected(n, k_est)
    lnpz_obs = lnpZ_observed_of(Z_samples, var_z)
    lnpz_err = abs(lnpz_obs - lnpz_exp)

    scf = scf_of(q_strict, lnpz_exp, num_params, n)       # C2
    ad = a_post_diagnostics(model, Z_samples, F, sigma, var_z, w0, w)
    s_lap = s_laplace_post_of(scf, ad["sum_log_det_A_post"])   # C3

    Z_est = res["Z_est"]
    R, k_min = procrustes_rotation(Z_est, Z_true)
    rmse_Z = calc_rmse(Z_true[:, :k_min], Z_est[:, :k_min] @ R)
    rmse_X = calc_rmse(X, model._mean_function_x(Z_est @ F.T))

    hist = res.get("mstep_q_history") or []

    row = {
        "q_strict": q_strict,
        "bic_impl": bic_impl,
        "num_params": num_params,
        "lnpZ_det_expected": lnpz_exp,
        "lnpZ_observed": lnpz_obs,
        "lnpZ_abs_err": lnpz_err,
        "scf": scf,
        "s_laplace_post": s_lap,
        "rank_F": int(np.linalg.matrix_rank(F, tol=RANK_F_TOL)),
        "rmse_Z": rmse_Z,
        "rmse_X": rmse_X,
        "mstep_q_n_iters": len(hist),
        "mstep_q_n_decreased": int(sum(bool(h["decreased"]) for h in hist)),
        "mstep_q_min_diff": (float(min(h["q_diff"] for h in hist)) if hist
                             else float("nan")),
        "mstep_q_history": json.dumps(hist),
        "nan_occurred": bool(res["nan_occurred"]),
        "nan_count": int(res["nan_count"]),
        "q_bic_failed": bool(res.get("q_bic_failed")),
        "numerics_mode": res.get("numerics_mode"),
        "runtime_s": float(res["runtime_s"]),
    }
    row.update(ad)
    return row


# -- Main ------------------------------------------------------------------

def _resolve_paths(mode, out_dir, fig_dir, date_tag):
    """Smoke and full write DISJOINT file names.

    smoke -> k_selection_score_pilot_smoke_<date>_*
    full  -> k_selection_score_pilot_<date>_*

    The `smoke` marker precedes the date so that a smoke run can never consume the
    file name the full pilot will later use. `paths_disjoint()` asserts this.
    """
    tag = (f"{STEM_PREFIX}_smoke_{date_tag}" if mode == "smoke"
           else f"{STEM_PREFIX}_{date_tag}")
    csvs = {name: out_dir / f"{tag}_{name}.csv"
            for name in ("summary", "agg", "selection", "runinfo")}
    figs = {name: fig_dir / f"{tag}_{name}.png"
            for name in ("score_curves", "log_det_A")}
    return tag, csvs, figs


def _failure_path(mode, out_dir, date_tag):
    return out_dir / f"{STEM_PREFIX}_{mode}_{date_tag}_failures.csv"


def paths_disjoint(out_dir, fig_dir, date_tag):
    """No smoke output path may collide with any full output path."""
    _, sc, sf = _resolve_paths("smoke", out_dir, fig_dir, date_tag)
    _, fc, ff = _resolve_paths("full", out_dir, fig_dir, date_tag)
    smoke = {p.resolve() for p in list(sc.values()) + list(sf.values())}
    full = {p.resolve() for p in list(fc.values()) + list(ff.values())}
    smoke.add(_failure_path("smoke", out_dir, date_tag).resolve())
    full.add(_failure_path("full", out_dir, date_tag).resolve())
    return smoke.isdisjoint(full), sorted(str(p) for p in smoke & full)


def _guard_overwrite(paths, allow_overwrite):
    existing = [str(p) for p in paths if p.exists()]
    if existing and not allow_overwrite:
        raise ExperimentStop(
            "refusing to overwrite existing output (pass --allow-overwrite to override):\n  "
            + "\n  ".join(existing))


def write_failure_artifact(path, *, key, seed_data, seed_model, x_hash, y_hash,
                           failure_stage, failure_type, failure_reason,
                           retry_info=None, res=None, warning_records=None,
                           provenance=None):
    """Persist one blocking fit failure; never place it in summary/aggregation."""
    retry_info = retry_info or {}
    res = res or {}
    warning_records = warning_records or []
    provenance = provenance or _git_provenance()
    n, trial, k_est = key
    record = {
        "n": n, "trial": trial, "k_est": k_est,
        "seed_data": seed_data, "seed_model": seed_model,
        "x_hash": x_hash, "y_hash": y_hash,
        "failure_stage": failure_stage,
        "failure_type": failure_type,
        "failure_reason": str(failure_reason),
        "internal_retry": bool(retry_info.get("internal_retry", False)),
        "nan_occurred": bool(res.get("nan_occurred", False)),
        "q_bic_failed": bool(res.get("q_bic_failed", False)),
        "n_warnings": len(warning_records),
        "warning_details": json.dumps(warning_records, ensure_ascii=False),
        "git_head": provenance["git_head"],
        "git_branch": provenance["git_branch"],
        "git_is_dirty": provenance["git_is_dirty"],
        "git_status_porcelain": provenance["git_status_porcelain"],
        "runner_path": provenance["runner_path"],
        "runner_sha256": provenance["runner_sha256"],
        "runner_is_tracked": provenance["runner_is_tracked"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record]).to_csv(path, index=False)
    print(f"saved failure evidence {path}")
    return record


def _validate_smoke_gate_records(gates):
    """Strictly validate the required G1-G8 PASS records from smoke runinfo."""
    expected_gates = {f"G{i}" for i in range(1, 9)}
    problems = []
    gate_names = []

    if not isinstance(gates, list):
        return [f"gates must be a list, got {type(gates).__name__}"]

    for index, record in enumerate(gates):
        if not isinstance(record, dict):
            problems.append(
                f"gate record {index} must be a dict, got {type(record).__name__}")
            continue

        missing = [key for key in ("gate", "passed") if key not in record]
        if missing:
            problems.append(
                f"gate record {index} missing required key(s): {', '.join(missing)}")

        if "gate" in record:
            gate = record["gate"]
            if not isinstance(gate, str):
                problems.append(
                    f"gate record {index} gate must be a string, "
                    f"got {type(gate).__name__}")
            else:
                gate_names.append(gate)

        if "passed" in record:
            passed = record["passed"]
            if type(passed) is not bool:
                problems.append(
                    f"gate record {index} passed must be bool, "
                    f"got {type(passed).__name__}")
            elif passed is not True:
                problems.append(f"gate record {index} passed is not True")

    duplicate_gates = sorted(
        gate for gate in set(gate_names) if gate_names.count(gate) > 1)
    if duplicate_gates:
        problems.append(f"duplicate gate name(s): {duplicate_gates}")

    actual_gates = set(gate_names)
    missing_gates = sorted(expected_gates - actual_gates)
    unknown_gates = sorted(actual_gates - expected_gates)
    if missing_gates:
        problems.append(f"missing required gate(s): {missing_gates}")
    if unknown_gates:
        problems.append(f"unknown gate name(s): {unknown_gates}")

    return problems


def validate_smoke_prerequisite(out_dir, fig_dir, date_tag):
    """Read and fully validate existing smoke primary/derived artifacts before full."""
    _, csvs, _ = _resolve_paths("smoke", out_dir, fig_dir, date_tag)
    required_paths = [csvs["summary"], csvs["selection"], csvs["runinfo"]]
    missing_paths = [str(p) for p in required_paths if not p.exists()]
    if missing_paths:
        raise ExperimentStop(
            "full smoke prerequisite missing before data generation / EM: "
            + ", ".join(missing_paths), failure_stage="smoke_preflight",
            failure_type="SmokePrerequisiteMissing")

    try:
        summary = pd.read_csv(csvs["summary"])
        selection = pd.read_csv(csvs["selection"])
        runinfo = pd.read_csv(csvs["runinfo"])
    except Exception as exc:
        raise ExperimentStop(
            f"full smoke prerequisite unreadable: {type(exc).__name__}: {exc}",
            failure_stage="smoke_preflight",
            failure_type="SmokePrerequisiteMalformed") from exc

    problems = []
    expected_keys = {(75, 1, 2), (75, 1, 3), (75, 1, 4)}
    required_key_cols = {"n", "trial", "k_est"}
    if not required_key_cols.issubset(summary.columns):
        actual_keys = set()
        problems.append("summary missing n/trial/k_est")
    else:
        actual_keys = set(zip(summary["n"], summary["trial"], summary["k_est"]))
    if len(summary) != SMOKE_FIT_COUNT or actual_keys != expected_keys:
        problems.append(f"summary keys/count invalid: count={len(summary)}, keys={actual_keys}")

    for col in REQUIRED_FINITE_METRICS:
        if col not in summary:
            problems.append(f"summary missing required numeric metric {col}")
        else:
            vals = pd.to_numeric(summary[col], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                problems.append(f"summary metric {col} contains non-finite values")

    if "lnpZ_abs_err" in summary:
        errs = pd.to_numeric(summary["lnpZ_abs_err"], errors="coerce").to_numpy(float)
        if not (np.isfinite(errs).all() and np.all(errs < LNPZ_TOL)):
            problems.append("G1 lnpZ_abs_err is non-finite or outside tolerance")
    for col in ("internal_retry", "nan_occurred", "q_bic_failed"):
        if col not in summary or summary[col].astype(str).str.lower().ne("false").any():
            problems.append(f"summary requires all {col}=False")
    if ("slogdet_sign_all_pos" not in summary
            or summary["slogdet_sign_all_pos"].astype(str).str.lower().ne("true").any()):
        problems.append("summary requires all slogdet signs positive")
    for hash_col in ("x_hash", "y_hash"):
        if hash_col not in summary or summary[hash_col].nunique(dropna=False) != 1:
            problems.append(f"summary requires identical {hash_col}")

    expected_selection = {
        "bic_impl": (3, False, True),
        "scf": (4, True, False),
        "s_laplace_post": (3, False, True),
    }
    if ("score" not in selection or len(selection) != 3
            or set(selection["score"]) != set(expected_selection)):
        problems.append("selection rows/scores invalid")
    else:
        for score, (k, boundary, interior) in expected_selection.items():
            r = selection[selection["score"] == score].iloc[0]
            if (int(r["n"]), int(r["trial"]), int(r["argmin_k"])) != (75, 1, k):
                problems.append(f"selection {score} argmin/key invalid")
            if (str(r["at_range_boundary"]).lower() != str(boundary).lower()
                    or str(r["interior_minimum"]).lower() != str(interior).lower()):
                problems.append(f"selection {score} boundary flags invalid")

    if len(runinfo) != 1:
        problems.append(f"runinfo row count {len(runinfo)} != 1")
    else:
        try:
            gates = json.loads(runinfo.iloc[0]["gates"])
            gate_problems = _validate_smoke_gate_records(gates)
            problems.extend(
                f"runinfo gates malformed: {problem}"
                for problem in gate_problems)
        except Exception as exc:
            problems.append(f"runinfo gates malformed: {type(exc).__name__}: {exc}")

    if problems:
        raise ExperimentStop(
            "full smoke prerequisite failed before data generation / EM:\n  "
            + "\n  ".join(problems), failure_stage="smoke_preflight",
            failure_type="SmokePrerequisiteFailed")
    return {"summary": summary, "selection": selection, "runinfo": runinfo}


def run_pilot(mode, out_dir, fig_dir, allow_overwrite, date_tag):
    provenance = _git_provenance()
    if mode == "full":
        validate_full_repository_state(provenance)
        validate_smoke_prerequisite(out_dir, fig_dir, date_tag)

    man = build_manifest(mode)
    problems = validate_manifest(man, mode)
    if problems:
        raise ExperimentStop("manifest validation failed:\n  " + "\n  ".join(problems))

    tag, csvs, figs = _resolve_paths(mode, out_dir, fig_dir, date_tag)
    failure_path = _failure_path(mode, out_dir, date_tag)
    _guard_overwrite(list(csvs.values()) + list(figs.values()) + [failure_path],
                     allow_overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    rows, seen = [], set()
    combos = sorted({(int(r["n"]), int(r["trial"])) for _, r in man.iterrows()})

    for n, trial in combos:
        sd = seed_data_of(n, trial)
        data = generate_dual_data(n=n, d=D, k=K_TRUE, seed=sd,
                                  family_x=FAMILY_X, family_y=FAMILY_Y)
        X, Y, Z_true = data["X"], data["Y"], data["Z"]
        x_hash, y_hash = _sha256_arrays(X), _sha256_arrays(Y)

        ks = sorted(man[(man["n"] == n) & (man["trial"] == trial)]["k_est"])
        for k_est in ks:
            key = (n, trial, int(k_est))
            if key in seen:
                raise ExperimentStop(f"G4 duplicate fit key {key}")
            seen.add(key)

            sm = seed_model_of(n, trial, int(k_est))
            print(f"[fit] n={n} trial={trial} k_est={k_est} "
                  f"seed_data={sd} seed_model={sm}")
            res, retry, wrecs = {}, {}, []
            failure_stage = "fit"
            try:
                res, captured, retry, wrecs = run_one_fit(
                    X, Y, int(k_est), sm, key)
                failure_stage = "fit_result_gate"
                check_fit(key, res, retry, captured)
                failure_stage = "score_calculation"
                row = score_one_fit(res, X, Z_true, n, int(k_est))
                failure_stage = "finite_metric_gate"
                check_required_finite(key, row)
                failure_stage = "lnpz_gate"
                check_lnpz_gate(key, row)
                failure_stage = "curvature_gate"
                check_curvature(key, row)                             # G6, blocking
            except Exception as exc:
                if getattr(exc, "warning_records", None) is not None:
                    wrecs = exc.warning_records
                if getattr(exc, "retry_info", None) is not None:
                    retry = exc.retry_info
                write_failure_artifact(
                    failure_path, key=key, seed_data=sd, seed_model=sm,
                    x_hash=x_hash, y_hash=y_hash,
                    failure_stage=(getattr(exc, "failure_stage", None)
                                   or failure_stage),
                    failure_type=(getattr(exc, "failure_type", None)
                                  or type(exc).__name__),
                    failure_reason=str(exc), retry_info=retry, res=res,
                    warning_records=wrecs, provenance=provenance)
                if isinstance(exc, ExperimentStop):
                    raise
                raise ExperimentStop(
                    f"{key}: {failure_stage} raised {type(exc).__name__}: {exc}",
                    failure_stage=failure_stage,
                    failure_type=type(exc).__name__) from exc

            row.update({
                "scenario": SCENARIO, "n": n, "d": D, "K_TRUE": K_TRUE,
                "trial": trial, "k_est": int(k_est),
                "family_x": FAMILY_X, "family_y": FAMILY_Y,
                "L": L, "num_iter": NITER,
                "seed_data": sd, "seed_model": sm,
                "x_hash": x_hash, "y_hash": y_hash,
                "hash_version": HASH_VERSION,
                "n_warnings": len(wrecs),
                "warning_details": json.dumps(wrecs, ensure_ascii=False),
                "internal_retry": retry["internal_retry"],
                "internal_retry_detected": retry["internal_retry_detected"],
                "internal_retry_indices": str(retry["internal_retry_indices"]),
                "internal_retry_max_index": retry["internal_retry_max_index"],
                "internal_retry_n_resets": retry["internal_retry_n_resets"],
                "implementation_lineage": IMPLEMENTATION_LINEAGE,
            })
            rows.append(row)

    df = pd.DataFrame(rows)

    expected = SMOKE_FIT_COUNT if mode == "smoke" else FULL_FIT_COUNT
    if len(df) != expected:
        raise ExperimentStop(f"produced {len(df)} fits, expected {expected}")

    gates = evaluate_gates(df, mode)
    for g in gates:
        print(f"  [{g['gate']}] {'PASS' if g['passed'] else 'FAIL'}: {g['detail']}")
    blocking = [g for g in gates if not g["passed"] and g["blocking"]]
    if blocking:
        raise ExperimentStop(
            "integrity gate(s) failed before aggregation:\n  "
            + "\n  ".join(f"{g['gate']}: {g['detail']}" for g in blocking))

    agg = aggregate(df)
    sel = selection_table(df)

    df.to_csv(csvs["summary"], index=False)
    agg.to_csv(csvs["agg"], index=False)
    sel.to_csv(csvs["selection"], index=False)
    pd.DataFrame(runinfo_rows(mode, df, gates, t0, provenance, man)).to_csv(
        csvs["runinfo"], index=False)
    for p in csvs.values():
        print(f"saved {p}")

    make_figures(agg, figs)
    print(f"\ntotal runtime {time.perf_counter() - t0:.1f}s")


def evaluate_gates(df, mode):
    """G1-G7. Reported as a table; blocking gates stop the run before aggregation."""
    gates = []

    errs = pd.to_numeric(df["lnpZ_abs_err"], errors="coerce").to_numpy(float)
    g1_pass = bool(np.isfinite(errs).all() and np.all(errs < LNPZ_TOL))
    worst = float(np.max(errs)) if np.isfinite(errs).all() else float("nan")
    gates.append({"gate": "G1", "blocking": True, "passed": g1_pass,
                  "detail": f"max lnpZ_abs_err={worst:.3e} (tol {LNPZ_TOL})"})

    bad = df[df["numerics_mode"] != NUMERICS_MODE]
    gates.append({"gate": "G2", "blocking": True, "passed": bad.empty,
                  "detail": f"{len(bad)} fits with numerics_mode != {NUMERICS_MODE!r}"})

    man = build_manifest(mode)
    probs = validate_manifest(man, mode)
    gates.append({"gate": "G3", "blocking": True, "passed": not probs,
                  "detail": "seed uniqueness/independence: "
                            + ("OK" if not probs else "; ".join(probs))})

    keys = list(zip(df["n"], df["trial"], df["k_est"]))
    gates.append({"gate": "G4", "blocking": True, "passed": len(set(keys)) == len(keys),
                  "detail": f"{len(keys)} fits, {len(set(keys))} distinct (n,trial,k_est)"})

    off = [f"(n={n},trial={t})"
           for (n, t), g in df.groupby(["n", "trial"])
           if g["x_hash"].nunique() != 1 or g["y_hash"].nunique() != 1]
    gates.append({"gate": "G5", "blocking": True, "passed": not off,
                  "detail": "identical X/Y hashes within each (n,trial): "
                            + ("OK" if not off else "MISMATCH " + ",".join(off))})

    viol = int(df["n_slogdet_sign_violations"].sum())
    gates.append({"gate": "G6", "blocking": True,
                  "passed": bool(df["slogdet_sign_all_pos"].all()),
                  "detail": f"{viol} non-positive slogdet signs over "
                            f"{int(df['n_A_evaluated'].sum())} A_i^post evaluations "
                            "(recorded AND blocking: S_laplace_post would not be a "
                            "real-valued diagnostic)"})

    nr = int(df["internal_retry"].sum())
    gates.append({"gate": "G8", "blocking": True, "passed": nr == 0,
                  "detail": f"{nr} fits with an internal NaN reset / retry "
                            "(hard failure; such a fit is never used even if it "
                            "returned finite values)"})

    frac = float(df["nan_occurred"].mean())
    gates.append({"gate": "G7", "blocking": True, "passed": frac <= NAN_FIT_FRACTION_MAX,
                  "detail": f"NaN-affected fit fraction={frac:.3f} "
                            f"(max {NAN_FIT_FRACTION_MAX})"})
    return gates


def aggregate(df):
    cols = ["bic_impl", "scf", "s_laplace_post", "sum_log_det_A_post",
            "mean_log_det_A_per_node", "min_eig_A", "max_eig_A",
            "rmse_Z", "rmse_X", "rank_F", "num_params", "q_strict",
            "lnpZ_abs_err", "runtime_s"]
    g = df.groupby(["n", "k_est"])[cols].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


def selection_table(df):
    """argmin per (n, trial) per score, flagged interior vs range boundary."""
    rows = []
    for (n, trial), g in df.groupby(["n", "trial"]):
        g = g.sort_values("k_est")
        available_k = sorted(g["k_est"].unique())
        lo, hi = min(available_k), max(available_k)
        for score in ("bic_impl", "scf", "s_laplace_post"):
            kbest = int(g.loc[g[score].idxmin(), "k_est"])
            at_range_boundary = kbest == lo or kbest == hi
            rows.append({
                "n": n, "trial": trial, "score": score, "argmin_k": kbest,
                "k_range_lo": lo, "k_range_hi": hi,
                "at_range_boundary": bool(at_range_boundary),
                "interior_minimum": bool(not at_range_boundary),
                "matches_K_TRUE": bool(kbest == K_TRUE),
                "monotone_decreasing": bool(np.all(np.diff(g[score].values) < 0)),
                "monotone_increasing": bool(np.all(np.diff(g[score].values) > 0)),
            })
    return pd.DataFrame(rows)


def runinfo_rows(mode, df, gates, t0, provenance, manifest):
    data_seed_manifest, model_seed_manifest = seed_manifests(manifest)
    return [{
        "issue": 37,
        "script": "tools/research_audit/run_k_selection_score_pilot.py",
        "git_head": provenance["git_head"],
        "git_branch": provenance["git_branch"],
        "git_is_dirty": provenance["git_is_dirty"],
        "git_status_porcelain": provenance["git_status_porcelain"],
        "runner_path": provenance["runner_path"],
        "runner_sha256": provenance["runner_sha256"],
        "runner_is_tracked": provenance["runner_is_tracked"],
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "scenario": SCENARIO, "family_x": FAMILY_X, "family_y": FAMILY_Y,
        "K_TRUE": K_TRUE, "k_est_list": str(K_EST_LIST), "n_list": str(N_LIST),
        "d": D, "L": L, "num_iter": NITER, "trials": str(TRIALS),
        "trial_convention": "1-based: trials = [1, 2, 3]",
        "seed_convention": ("n_index: 75->0, 150->1; "
                            "seed_data = 150000 + 100*n_index + trial (k-independent); "
                            "seed_model = 151000 + 100*n_index + 10*trial + k_est"),
        "data_seed_manifest": data_seed_manifest,
        "model_seed_manifest": model_seed_manifest,
        "hash_version": HASH_VERSION,
        "numerics_mode": NUMERICS_MODE,
        "implementation_lineage": IMPLEMENTATION_LINEAGE,
        "mstep_q_diagnostic": True,
        "rank_F_tol": RANK_F_TOL, "lnpZ_tol": LNPZ_TOL,
        "g6_policy": ("BLOCKING: any A_i^post with slogdet sign <= 0 fails the fit; "
                      "no logabsdet substitution, no jitter, no symmetrised-matrix "
                      "determinant, no silent drop, no seed change"),
        "g8_policy": ("BLOCKING: an internal NaN reset / retry (em_runner.py L.222 "
                      "'[NaN iter={i} retry={r}]') fails the fit even if it returned "
                      "finite values; detected by stdout capture because the runner "
                      "returns no retry counter and resets nan_count per retry"),
        "nan_fit_fraction_max": NAN_FIT_FRACTION_MAX,
        "smoke_fit_count": SMOKE_FIT_COUNT, "full_fit_count": FULL_FIT_COUNT,
        "n_fits": len(df),
        "a_post_definition": ("A_i^post = model._calc_precision_matrix("
                              "Z_samples[:,:,l], F, sigma, var_z, w0, w, i) on the FINAL "
                              "scale_Z-applied samples; post-hoc, NOT the sampling-time A_i"),
        "a_post_averaging": ("sum over nodes i within each sample l, then mean over L: "
                             "(1/L) sum_l sum_i logdet(A_i^post,(l))"),
        "score_c1": "bic_impl = -2*Q_strict + p_hat*log(n) (descriptive baseline)",
        "score_c2": ("scf = -2*(Q_strict - lnpZ_det) + p_hat*log(n); "
                     "counterfactual diagnostic score, NOT corrected/true/Schwarz BIC, "
                     "NOT a paper Eq.(26) reproduction, NOT Q3"),
        "score_c3": ("s_laplace_post = scf + (1/L) sum_l sum_i "
                     "logdet(A_i^post,(l)); "
                     "post-hoc Laplace-curvature diagnostic, NOT an ELBO, "
                     "NOT a marginal likelihood or a bound on one"),
        "quantities": ("Q1 = log p(X,Y|Z,theta) (paper Eq.16); "
                       "Q2 = log p(Z,X,Y|theta) (Q_strict target); "
                       "Q3 = log p(X,Y|theta) (observed-data marginal). "
                       "None of C1/C2/C3 is Q3."),
        "gates": json.dumps(gates),
        "total_runtime_s": round(time.perf_counter() - t0, 2),
    }]


def make_figures(agg, figs):
    available_n = sorted(agg["n"].unique())
    fig, axes = plt.subplots(1, len(available_n),
                             figsize=(5.2 * len(available_n), 4.2))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, available_n):
        s = agg[agg["n"] == n].sort_values("k_est")
        for col, lab in (("bic_impl_mean", "C1 bic_impl (baseline)"),
                         ("scf_mean", "C2 scf (counterfactual diagnostic)"),
                         ("s_laplace_post_mean", "C3 s_laplace_post (curvature diagnostic)")):
            ax.plot(s["k_est"], s[col], marker="o", label=lab)
        ax.axvline(K_TRUE, ls="--", c="grey", lw=1)
        ax.set_title(f"n = {n}", fontsize=10)
        ax.set_xlabel("estimated dimension k")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("score (lower = preferred)")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Candidate dimension-selection scores on the SAME fits "
                 "(diagnostic; none is the observed-data marginal likelihood)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(figs["score_curves"], dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {figs['score_curves']}")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for n in sorted(agg["n"].unique()):
        s = agg[agg["n"] == n].sort_values("k_est")
        ax.errorbar(s["k_est"], s["sum_log_det_A_post_mean"],
                    yerr=s["sum_log_det_A_post_std"], marker="o", capsize=3,
                    label=f"n = {n}")
    ax.axvline(K_TRUE, ls="--", c="grey", lw=1)
    ax.set_xlabel("estimated dimension k")
    ax.set_ylabel(r"$(1/L)\sum_l \sum_i \log\det A_i^{post}$")
    ax.set_title("Post-hoc Laplace curvature (first measurement; not an ELBO term)",
                 fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs["log_det_A"], dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {figs['log_det_A']}")


def do_postprocess_only(summary_path, out_dir, fig_dir, date_tag):
    """Regenerate derived smoke artifacts from an existing summary; never run EM."""
    _, csvs, figs = _resolve_paths("smoke", out_dir, fig_dir, date_tag)
    summary_path = summary_path.resolve()
    if not summary_path.exists():
        raise ExperimentStop(f"summary input does not exist: {summary_path}")
    if summary_path != csvs["summary"].resolve():
        raise ExperimentStop(
            f"summary input/path mismatch: {summary_path} != {csvs['summary'].resolve()}")

    df = pd.read_csv(summary_path)
    if len(df) != SMOKE_FIT_COUNT:
        raise ExperimentStop(
            f"postprocess input has {len(df)} fits, expected smoke count {SMOKE_FIT_COUNT}")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate(df)
    sel = selection_table(df)
    agg.to_csv(csvs["agg"], index=False)
    sel.to_csv(csvs["selection"], index=False)
    print(f"saved {csvs['agg']}")
    print(f"saved {csvs['selection']}")
    make_figures(agg, figs)
    print("POSTPROCESS-ONLY COMPLETE. No data generation. No EM fit executed.")


def do_validate_only(out_dir, fig_dir, date_tag):
    """Static validation. Generates no data, runs no EM, writes no output file."""
    print("=" * 72)
    print("VALIDATE-ONLY - no data generation, no EM fit, no file written")
    print("=" * 72)

    ok = True
    for mode, expected in (("smoke", SMOKE_FIT_COUNT), ("full", FULL_FIT_COUNT)):
        man = build_manifest(mode)
        probs = validate_manifest(man, mode)
        print(f"\n--- {mode}: {len(man)} fits (expected {expected}) ---")
        print(man.to_string(index=False))
        if probs:
            ok = False
            print("  MANIFEST PROBLEMS: " + "; ".join(probs))
        else:
            print("  manifest OK: fit count, no duplicate key, no seed collision, "
                  "seed_data independent of k_est")
        _, csvs, figs = _resolve_paths(mode, out_dir, fig_dir, date_tag)
        print("  planned outputs:")
        planned = (list(csvs.values()) + list(figs.values())
                   + [_failure_path(mode, out_dir, date_tag)])
        for p in planned:
            print(f"    {'EXISTS ' if p.exists() else 'new    '} {p}")

    print("\n--- smoke / full artifact separation ---")
    disj, clash = paths_disjoint(out_dir, fig_dir, date_tag)
    ok &= disj
    print("  smoke and full output paths disjoint: "
          + ("OK" if disj else f"FAIL, colliding: {clash}"))
    _, fc, ff = _resolve_paths("full", out_dir, fig_dir, date_tag)
    full_paths = (list(fc.values()) + list(ff.values())
                  + [_failure_path("full", out_dir, date_tag)])
    taken = [str(p) for p in full_paths if p.exists()]
    ok &= not taken
    print("  full-pilot output paths currently free (a smoke run must not consume "
          "them): " + ("OK" if not taken else f"FAIL, already present: {taken}"))

    print("\n--- formula unit checks ---")
    for n, k in ((75, 3), (150, 3), (150, 7)):
        got = lnpZ_det_expected(n, k)
        ref = -(n * k / 2.0) * (1.0 + np.log(2.0 * np.pi))
        same = abs(got - ref) < 1e-12
        ok &= same
        print(f"  lnpZ_det_expected(n={n},k={k}) = {got:.6f}  "
              f"[= -1.4189385332*n*k -> {-1.4189385332 * n * k:.6f}]  {'OK' if same else 'FAIL'}")

    # lnpZ_observed must equal the analytic value for any array with mean square 1
    rng = np.random.default_rng(0)
    for n, k in ((75, 3), (150, 7)):
        Zs = rng.normal(size=(n, k, L))
        Zs = Zs / np.sqrt(np.mean(Zs ** 2))          # replicates scale_Z's post-condition
        obs = lnpZ_observed_of(Zs, 1.0)
        err = abs(obs - lnpZ_det_expected(n, k))
        good = err < LNPZ_TOL
        ok &= good
        print(f"  lnpZ_observed vs expected (n={n},k={k}): abs_err={err:.3e} "
              f"(tol {LNPZ_TOL})  {'OK' if good else 'FAIL'}")

    q, npar, n, k = -5000.0, 39, 150, 3
    c1 = -2.0 * q + npar * np.log(n)
    c2 = scf_of(q, lnpZ_det_expected(n, k), npar, n)
    c3 = s_laplace_post_of(c2, 12.5)
    id1 = abs((c2 - c1) - (2.0 * lnpZ_det_expected(n, k))) < 1e-9
    id2 = abs((c3 - c2) - 12.5) < 1e-12
    ok &= id1 and id2
    print(f"  C1={c1:.4f} C2={c2:.4f} C3={c3:.4f}")
    print(f"  identity C2-C1 == 2*lnpZ_det : {'OK' if id1 else 'FAIL'}")
    print(f"  identity C3-C2 == sum_logdet : {'OK' if id2 else 'FAIL'}")

    print("\n--- hash helper check ---")
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    h1, h2 = _sha256_arrays(a), _sha256_arrays(a.copy())
    h3 = _sha256_arrays(a.astype(np.float32))
    h4 = _sha256_arrays(a.reshape(4, 3))
    checks = [("stable across copies", h1 == h2),
              ("dtype-sensitive", h1 != h3),
              ("shape-sensitive", h1 != h4),
              ("full SHA-256 64-hex digest",
               len(h1) == 64 and all(c in "0123456789abcdef" for c in h1))]
    for name, good in checks:
        ok &= good
        print(f"  {name}: {'OK' if good else 'FAIL'}")

    print("\n--- selection-table helper check (synthetic, no fit) ---")
    fake = pd.DataFrame({
        "n": [75] * 7, "trial": [1] * 7, "k_est": K_EST_LIST,
        "bic_impl": [9, 7, 5, 6, 7, 8, 9],       # interior min at k=3
        "scf": [9, 8, 7, 6, 5, 4, 3],            # monotone decreasing -> boundary at k=7
        "s_laplace_post": [1, 2, 3, 4, 5, 6, 7],  # boundary at k=1
    })
    sel = selection_table(fake)
    exp = {("bic_impl", 3, True), ("scf", 7, False), ("s_laplace_post", 1, False)}
    got = {(r["score"], r["argmin_k"], r["interior_minimum"]) for _, r in sel.iterrows()}
    good = got == exp
    ok &= good
    print(sel.to_string(index=False))
    print(f"  argmin / boundary flags: {'OK' if good else 'FAIL got=' + str(got)}")

    print("\n--- lineage / naming assertions ---")
    # A bare occurrence count is not a test: these phrases legitimately appear when
    # DEFINING Q3 and when stating the prohibitions. What must hold is that no line
    # LABELS one of our scores with a prohibited name. So every occurrence line is
    # required to carry a negating or defining marker. Lines tagged
    # naming-check-exempt are the checker's own literals.
    # A sentence may wrap, so the preceding line is inspected too.
    markers = ("NEVER", "NOT ", "not ", "none", "concerns", "is not", "prohibit",
               "Q3", "Observed-data", "lower bound")
    src_lines = Path(__file__).read_text(encoding="utf-8").split("\n")
    for bad in ("corrected BIC", "Schwarz BIC", "true BIC",          # naming-check-exempt
                "ELBO", "marginal likelihood"):                      # naming-check-exempt
        hits = [(i + 1, ln) for i, ln in enumerate(src_lines)
                if bad in ln and "naming-check-exempt" not in ln]
        unguarded = [i for i, ln in hits
                     if not any(m in ln or m in src_lines[i - 2]
                                for m in markers)]
        good = not unguarded
        ok &= good
        print(f"  {bad!r}: {len(hits)} mention(s), {len(unguarded)} unguarded  "
              + ("OK" if good else f"FAIL at lines {unguarded}"))
    print(f"  implementation_lineage records prototype status: "
          f"{'OK' if 'PROTOTYPE' in IMPLEMENTATION_LINEAGE else 'FAIL'}")

    print("\n" + "=" * 72)
    print("VALIDATE-ONLY RESULT:", "ALL CHECKS PASSED" if ok else "FAILURES PRESENT")
    print("SMOKE NOT RUN. FULL NOT RUN. No EM fit was executed.")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Issue #37 Phase 7b K-selection score diagnostic pilot.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--validate-only", action="store_true",
                   help="static checks only: manifest, seeds, formulas, paths. No EM fit.")
    g.add_argument("--postprocess-only", action="store_true",
                   help="regenerate smoke derived artifacts from an existing summary; no EM fit")
    g.add_argument("--smoke", action="store_true",
                   help=f"run the {SMOKE_FIT_COUNT}-fit smoke subset")
    g.add_argument("--full", action="store_true",
                   help=f"run the full {FULL_FIT_COUNT}-fit pilot")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fig-dir", default=None)
    ap.add_argument("--date-tag", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--summary", default=None,
                    help="existing smoke summary CSV for --postprocess-only")
    ap.add_argument("--allow-overwrite", action="store_true",
                    help="permit overwriting existing output files")
    a = ap.parse_args()

    if not (a.validate_only or a.postprocess_only or a.smoke or a.full):
        ap.print_help()
        print("\nERROR: one of --validate-only / --postprocess-only / --smoke / "
              "--full is required. Nothing was run.")
        sys.exit(2)

    od = Path(a.out_dir) if a.out_dir else OUT_DIR
    fd = Path(a.fig_dir) if a.fig_dir else FIG_DIR

    if a.validate_only:
        sys.exit(do_validate_only(od, fd, a.date_tag))

    try:
        if a.postprocess_only:
            summary = (Path(a.summary) if a.summary else
                       _resolve_paths("smoke", od, fd, a.date_tag)[1]["summary"])
            do_postprocess_only(summary, od, fd, a.date_tag)
        else:
            run_pilot("smoke" if a.smoke else "full", od, fd,
                      a.allow_overwrite, a.date_tag)
    except ExperimentStop as exc:
        print("\n" + "=" * 70)
        print("EXPERIMENT STOPPED - integrity violation, result NOT used")
        print("=" * 70)
        print(exc)
        sys.exit(2)
