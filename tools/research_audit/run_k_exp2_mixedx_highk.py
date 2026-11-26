"""Experiment C: K selection with per-column mixed-family X at K_TRUE = 5, 7, 9.

EXPLORATORY (planned 2026-11-26 after the single-family-X 8,000-fit, 600-fit and
C_Lap results were seen). New MCEM fits; nothing of the earlier runs is changed.

Design (frozen in protocol.json before any main-data fit):
  * n = 150, d = 15, X columns: Gaussian x5 (var 0.1), Bernoulli x5, Poisson x5
    (blocks in this column order), Y Bernoulli.
  * Generator: data_generator_canonical_mixed.generate_canonical_mixed_data
    (canonical-clean-mixed-v1; literal draw, shared Z and F, no clipping).
  * Signal = regime b of the single-X run: F = s Q with s^2 = 0.5 d / K (mean squared row
    norm 0.5), w = sqrt(3/K), w0 from the FROZEN regime-b calibration table (population
    edge probability 0.20). Same per-direction dilution as regime b; Fisher information
    is NOT assumed equal to any single-family series.
  * Estimator: em_runner.run_em_experimental with family_x_list (->
    DualExpFamLSMPerColumnConsistent), numerics_mode='consistent',
    failure_policy='fail_fast', L = 10, 10 iterations, init/mc seeds as the main run.
  * Criteria on the SAME saved state: C_Q (calc_bic_exp, p = Kd - K(K-1)/2 + 5 Gaussian
    dispersions) and C_Lap (laplace_k_criterion v1, Z0 = Z_last, grad_tol 1e-8, max_iter
    200, d_K ln n). Pre-specified secondary: NOT_STATIONARY evaluations once more with
    max_iter 400 (as in follow-up B).
  * K_TRUE in {5, 7, 9}, candidates 1..10, 10 datasets each, 2 inits -> 600 fits.
  * Seeds: data_seed = 30,261,126,000 + 1000 K_TRUE + rep (pilot: + 5,000,000, reps
    100.., K_TRUE 7 and 9 only); init_seed = 10 data_seed + t, mc_seed = 10 data_seed + 5 + t.

Modes:
  --tech-check OUT       math/FD/generator checks (no fit)
  --pilot OUT            80 technical-pilot fits on pilot seeds (technical criteria only)
  --freeze OUT --pilot-dir P
  --execute OUT --protocol P [--resume] [--deadline ISO]
  --analyze RUN --out OUT
Lineage E prototype; C_Q is not Schwarz BIC; C_Lap is not an exact marginal likelihood.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics as stats  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_k_exp2_clap_comparison as rc  # noqa: E402  (frozen; functions reused unchanged)

ROOT = rc.ROOT
kd = rc.kd
N, D = kd.N, kd.D
FAMILY_X_LIST = ["gaussian"] * 5 + ["bernoulli"] * 5 + ["poisson"] * 5
FAM_IDX = {f: [i for i, x in enumerate(FAMILY_X_LIST) if x == f]
           for f in ("gaussian", "bernoulli", "poisson")}
K_TRUES = (5, 7, 9)
K_CANDS = tuple(range(1, 11))
N_REP = 10
L_MC, NUM_ITER = 10, 10
SIGMA_X_VAR = 0.1
SEED_BASE = 30_261_126_000
PILOT_OFFSET = 5_000_000
PILOT_K_TRUES = (7, 9)
PILOT_REPS = (100, 101)
WORKERS = 4
MEM_STOP_MB = 1500
CLAP_SECONDARY_MAX_ITER = 400
OUT_BASE = rc.BASE / "mixedx_highk_20261126"
CODE_FILES = rc.CODE_FILES + (
    "tools/research_audit/run_k_exp2_mixedx_highk.py",
    "expfam/src/experimental/data_generator_canonical_mixed.py",
    "expfam/src/experimental/model_dual_expfam_percolumn.py",
)
need, Stop = rc.need, rc.Stop

FIT_FIELDS = [
    "phase", "fit_id", "dataset_id", "k_true", "rep", "k", "t", "data_seed", "init_seed",
    "mc_seed", "status", "failure_reason", "x_hash", "y_hash", "y_density",
    "Q_strict", "C_Q", "num_params", "C_Q_recomputed_absdiff",
    "llZ", "llX_gauss", "llX_bern", "llX_pois", "llY", "terms_closure_absdiff",
    "q_after_last", "q_rel_change_last", "mstep_q_decreases", "w0_hat", "w_hat",
    "retry_count", "replacement_count", "nan_count", "runtime_s", "state_file", "state_sha256",
    "lap_status", "lap_error", "C_Lap", "lap_log_observed", "lap_phi_at_mode", "lap_fit_mode",
    "lap_zhat_sq", "lap_logdet_H", "lap_logdet_sign", "lap_min_eig_H", "lap_grad_inf",
    "lap_iterations", "lap_penalty", "d_K", "lap_closure_absdiff", "z_shift_rms",
    "lap_seconds", "mode_llX_gauss", "mode_llX_bern", "mode_llX_pois", "mode_llY",
    "lap400_status", "lap400_C_Lap", "lap400_grad_inf", "lap400_iterations",
    "lap400_min_eig_H", "worker_pid",
]


def now():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def bern_w0(k):
    return float(rc.bern_table()[("b", k)]["w0"])


def design(k_true):
    return {"f_scale": math.sqrt(0.5 * D / k_true), "w": math.sqrt(3.0 / k_true),
            "w0": bern_w0(k_true), "sigma_x_var": SIGMA_X_VAR}


def data_seed(k_true, rep, pilot=False):
    return SEED_BASE + (PILOT_OFFSET if pilot else 0) + 1000 * k_true + rep


def generate(k_true, seed):
    from data_generator_canonical_mixed import generate_canonical_mixed_data
    dz = design(k_true)
    return generate_canonical_mixed_data(
        n=N, d=D, k=k_true, seed=seed, family_x_list=FAMILY_X_LIST, family_y="bernoulli",
        f_scale=dz["f_scale"], sigma_x_var=dz["sigma_x_var"], w0=dz["w0"], w=dz["w"])


def jobs_hash(jobs):
    import hashlib
    return hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest()


def fit_id(phase, k_true, rep, k, t):
    return f"{'P-' if phase == 'pilot' else ''}MX-KT{k_true}-r{rep}-k{k:02d}-t{t}"


def jobs_for(phase):
    kts, reps = (PILOT_K_TRUES, PILOT_REPS) if phase == "pilot" else (K_TRUES, range(N_REP))
    out = []
    for kt in kts:
        for rep in reps:
            ds = data_seed(kt, rep, phase == "pilot")
            for k in K_CANDS:
                for t in (0, 1):
                    out.append({"phase": phase, "fit_id": fit_id(phase, kt, rep, k, t),
                                "dataset_id": f"{'P-' if phase == 'pilot' else ''}MX-KT{kt}-r{rep}",
                                "k_true": kt, "rep": rep, "k": k, "t": t, "data_seed": ds,
                                "init_seed": 10 * ds + t, "mc_seed": 10 * ds + 5 + t})
    return out


# --------------------------------------------------------------------------
# family-wise terms (written here from the model definition)
# --------------------------------------------------------------------------

def family_terms(X, Y, Z, F, sigma, w0, w):
    """ln p(X_f | Z) per family and ln p(Y | Z) (strict constants), one Z."""
    from scipy.special import gammaln
    eta = Z @ F.T
    g, b, p = FAM_IDX["gaussian"], FAM_IDX["bernoulli"], FAM_IDX["poisson"]
    var = np.maximum(np.diag(sigma)[g], 1e-8)
    lg = float(np.sum(-0.5 * (X[:, g] - eta[:, g]) ** 2 / var - 0.5 * np.log(var)
                      - 0.5 * math.log(2 * math.pi)))
    lb = float(np.sum(X[:, b] * eta[:, b] - np.logaddexp(0.0, eta[:, b])))
    lp = float(np.sum(X[:, p] * eta[:, p] - np.exp(eta[:, p]) - gammaln(X[:, p] + 1)))
    iu = np.triu_indices(Z.shape[0], 1)
    ey = (w0 + w * (Z @ Z.T))[iu]
    ly = float(np.sum(Y[iu] * ey - np.logaddexp(0.0, ey)))
    return lg, lb, lp, ly


def restore(k, params, Z0):
    from em_runner import build_model
    m = build_model(N, D, k, L_MC, "gaussian", "bernoulli", family_x_list=FAMILY_X_LIST,
                    numerics_mode="consistent")
    m.initialize_params(true_params=None, seed=0)
    m.params.update({"F": np.array(params["F"], float), "sigma": np.array(params["sigma"], float),
                     "w0": float(params["w0"]), "w": float(params["w"]),
                     "var_z": float(params["var_z"]), "Z": Z0.copy()})
    return m


def eval_clap(params, X, Y, k, max_iter=rc.MAX_ITER):
    """Frozen evaluator wrapper (rc.evaluate_state) + family terms at the joint mode."""
    import laplace_k_criterion as lk
    Z0 = np.array(params["Z_last"], float)
    out = rc.evaluate_state(restore(k, params, Z0), X, Y, Z0, k, max_iter=max_iter)
    if out["lap_status"] == "OK":
        res = lk.laplace_log_observed(restore(k, params, Z0), X, Y, Z0, grad_tol=rc.GRAD_TOL,
                                      max_iter=max_iter)
        need(repr(res.phi_at_mode) == out["lap_phi_at_mode"], "mode recomputation differs")
        lg, lb, lp, ly = family_terms(X, Y, res.Z_hat, params["F"], params["sigma"],
                                      params["w0"], params["w"])
        out.update(mode_llX_gauss=repr(lg), mode_llX_bern=repr(lb), mode_llX_pois=repr(lp),
                   mode_llY=repr(ly))
    return out


def fit_one(job, state_dir):
    from em_runner import run_em_experimental
    from eval_utils import calc_bic_exp
    t0 = time.perf_counter()
    row = {f: "" for f in FIT_FIELDS}
    row.update({k: job[k] for k in ("phase", "fit_id", "dataset_id", "k_true", "rep", "k", "t",
                                    "data_seed", "init_seed", "mc_seed")})
    row["worker_pid"] = os.getpid()
    ds = generate(job["k_true"], job["data_seed"])
    X, Y = ds.X, ds.Y
    iu = np.triu_indices(N, 1)
    row.update(x_hash=kd.array_hash(X), y_hash=kd.array_hash(Y), y_density=float(Y[iu].mean()))
    try:
        r = run_em_experimental(
            X, Y, "gaussian", "bernoulli", job["k"], L=L_MC, num_iter=NUM_ITER,
            family_x_list=FAMILY_X_LIST, numerics_mode="consistent", failure_policy="fail_fast",
            init_seed=job["init_seed"], mc_seed=job["mc_seed"], compute_strict_Q=True,
            mstep_q_diagnostic=True)
        q, cq, npar = float(r["Q_strict"]), float(r["bic"]), int(r["num_params"])
        hist = r.get("mstep_q_history") or []
        qa = [h["q_after"] for h in hist]
        finite = all(np.all(np.isfinite(np.asarray(r[k], float)))
                     for k in ("F", "sigma", "w0", "w"))
        ok = (np.isfinite(q) and np.isfinite(cq) and not r["q_bic_failed"] and finite
              and r["retry_count"] == 0 and r["replacement_count"] == 0)
        row.update(status="OK" if ok else "FAILED",
                   failure_reason="" if ok else (r.get("failure_reason") or "non-finite/integrity"),
                   Q_strict=repr(q), C_Q=repr(cq), num_params=npar,
                   C_Q_recomputed_absdiff=repr(abs(-2 * q + npar * math.log(N) - cq)),
                   q_after_last=repr(qa[-1]) if qa else "",
                   q_rel_change_last=(repr(abs(qa[-1] - qa[-2]) / abs(qa[-2]))
                                      if len(qa) > 1 else ""),
                   mstep_q_decreases=sum(bool(h["decreased"]) for h in hist),
                   w0_hat=repr(float(r["w0"])), w_hat=repr(float(r["w"])),
                   retry_count=r["retry_count"], replacement_count=r["replacement_count"],
                   nan_count=r["nan_count"])
        if ok:
            Zs = r["Z_samples"]
            k = job["k"]
            parts = np.zeros(4)
            for l in range(Zs.shape[2]):
                parts += family_terms(X, Y, Zs[:, :, l], r["F"], r["sigma"], r["w0"], r["w"])
            parts /= Zs.shape[2]
            llz = float(np.mean([-(N * k / 2) * math.log(2 * math.pi * r["var_z"])
                                 - np.sum(Zs[:, :, l] ** 2) / (2 * r["var_z"])
                                 for l in range(Zs.shape[2])]))
            tot = llz + float(parts.sum())
            row.update(llZ=repr(llz), llX_gauss=repr(parts[0]), llX_bern=repr(parts[1]),
                       llX_pois=repr(parts[2]), llY=repr(parts[3]),
                       terms_closure_absdiff=repr(abs(tot - q)))
            need(abs(tot - q) <= 1e-6 + 1e-10 * abs(q), f"family-term closure {job['fit_id']}")
            params = {"F": r["F"], "sigma": r["sigma"], "w0": r["w0"], "w": r["w"],
                      "var_z": r["var_z"], "Z_last": r["Z_est"], "Z_mean": Zs.mean(axis=2)}
            sp = Path(state_dir) / f"{job['fit_id']}.npz"
            np.savez_compressed(sp, F=r["F"], sigma=r["sigma"], w0=r["w0"], w=r["w"],
                                var_z=r["var_z"], Z_last=r["Z_est"], Z_mean=Zs.mean(axis=2),
                                Z_samples=Zs)
            row["state_file"] = sp.name
            row["state_sha256"] = rc.sha256_file(sp)
            row.update(eval_clap(params, X, Y, k))
            if row["lap_status"] == "NOT_STATIONARY":      # pre-specified secondary, once
                e4 = rc.evaluate_state(restore(k, params, params["Z_last"]), X, Y,
                                       params["Z_last"], k, max_iter=CLAP_SECONDARY_MAX_ITER)
                row.update(lap400_status=e4["lap_status"], lap400_C_Lap=e4["C_Lap"],
                           lap400_grad_inf=e4.get("lap_grad_inf", ""),
                           lap400_iterations=e4.get("lap_iterations", ""),
                           lap400_min_eig_H=e4.get("lap_min_eig_H", ""))
    except rc.Stop:
        raise
    except Exception as exc:                 # recorded once; never retried / re-seeded
        row.update(status="FAILED", failure_reason=f"{type(exc).__name__}: {str(exc)[:300]}")
    row["runtime_s"] = round(time.perf_counter() - t0, 2)
    return row


# --------------------------------------------------------------------------
# runner (shared by pilot and main)
# --------------------------------------------------------------------------

def run_jobs(jobs, out: Path, deadline=None):
    import psutil
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    state_dir = out / "states"
    state_dir.mkdir(parents=True, exist_ok=True)
    pf = out / "per_fit.csv"
    done = {r["fit_id"] for r in rc.read_csv(pf)} if pf.exists() else set()
    todo = [j for j in jobs if j["fit_id"] not in done]
    rc.append_jsonl(out / "run_events.jsonl", {"event": "start", "time": now(),
                                               "git_head": rc.git("rev-parse", "HEAD"),
                                               "done": len(done), "todo": len(todo),
                                               "workers": WORKERS, "deadline": deadline})
    t0, n, low, stopped = time.time(), 0, 0, None
    it = iter(todo)
    pending = set()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        while True:
            while stopped is None and len(pending) < WORKERS:
                if deadline and _dt.datetime.now().astimezone() >= _dt.datetime.fromisoformat(deadline):
                    stopped = "STOPPED_DEADLINE"
                    break
                avail = psutil.virtual_memory().available / 2 ** 20
                low = low + 1 if avail < MEM_STOP_MB else 0
                if low >= 2:
                    stopped = f"STOPPED_MEMORY ({avail:.0f} MB)"
                    break
                j = next(it, None)
                if j is None:
                    break
                pending.add(ex.submit(fit_one, j, str(state_dir)))
            if not pending:
                break
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in fin:
                rc.append_csv(pf, [f.result()], FIT_FIELDS)
                n += 1
                if n % 20 == 0:
                    print(f"{n}/{len(todo)} fits ({time.time() - t0:.0f}s)", flush=True)
    rc.append_jsonl(out / "run_events.jsonl", {"event": stopped or "finished", "time": now(),
                                               "fits_run": n, "wall_s": round(time.time() - t0, 1)})
    return stopped


# --------------------------------------------------------------------------
# technical check (no fit)
# --------------------------------------------------------------------------

def tech_check(out: Path):
    import laplace_k_criterion as lk
    from scipy import stats as sst
    from eval_utils import calc_Q_dual_strict_exp, calc_bic_exp
    need(not out.exists(), f"output exists: {out}")
    checks, notes = {}, {}
    rng = np.random.default_rng(20261126)
    # generator: all main + pilot datasets (generation only)
    gen = []
    for phase, kts, reps in (("main", K_TRUES, range(N_REP)), ("pilot", PILOT_K_TRUES, PILOT_REPS)):
        for kt in kts:
            for rep in reps:
                s = data_seed(kt, rep, phase == "pilot")
                a, b = generate(kt, s), generate(kt, s)
                eta = a.Z @ a.F.T
                p = FAM_IDX["poisson"]
                gres = a.X[:, FAM_IDX["gaussian"]] - eta[:, FAM_IDX["gaussian"]]
                iu = np.triu_indices(N, 1)
                gen.append({"phase": phase, "k_true": kt, "rep": rep, "seed": s,
                            "rank_F": int(np.linalg.matrix_rank(a.F)),
                            "mean_row_norm_sq": float(np.mean(np.sum(a.F ** 2, 1))),
                            "max_eta_poisson": float(eta[:, p].max()),
                            "max_lambda_poisson": float(np.exp(eta[:, p]).max()),
                            "poisson_x_mean": float(a.X[:, p].mean()),
                            "poisson_expected_mean": float(np.mean(np.exp(0.5 * np.sum(
                                a.F[p] ** 2, 1)))),
                            "bern_x_mean": float(a.X[:, FAM_IDX["bernoulli"]].mean()),
                            "gauss_resid_var": float(gres.var()),
                            "z_col_var_mean": float(a.Z.var(axis=0).mean()),
                            "y_density": float(a.Y[iu].mean()),
                            "y_symmetric_zero_diag": bool(np.array_equal(a.Y, a.Y.T)
                                                          and np.all(np.diag(a.Y) == 0)),
                            "reproducible": bool(np.array_equal(a.X, b.X) and np.array_equal(a.Y, b.Y))})
    checks["rank_F_equals_K_TRUE_all"] = all(g["rank_F"] == g["k_true"] for g in gen)
    checks["mean_row_norm_sq_0.5"] = all(abs(g["mean_row_norm_sq"] - 0.5) < 1e-12 for g in gen)
    checks["generation_reproducible_bitwise"] = all(g["reproducible"] for g in gen)
    checks["Y_symmetric_zero_diag"] = all(g["y_symmetric_zero_diag"] for g in gen)
    checks["poisson_lambda_below_1e3"] = max(g["max_lambda_poisson"] for g in gen) < 1e3
    notes["generator"] = {"n_datasets": len(gen),
                          "max_lambda_poisson": max(g["max_lambda_poisson"] for g in gen),
                          "y_density_range": [min(g["y_density"] for g in gen),
                                              max(g["y_density"] for g in gen)],
                          "gauss_resid_var_range": [min(g["gauss_resid_var"] for g in gen),
                                                    max(g["gauss_resid_var"] for g in gen)],
                          "poisson_mean_obs_vs_expected_max_absdiff": max(
                              abs(g["poisson_x_mean"] - g["poisson_expected_mean"]) for g in gen)}
    # model math on a mixed dataset with random theta (K = 4)
    ds = generate(7, data_seed(7, 100, True))
    k = 4
    Z = rng.standard_normal((N, k)) * 0.8
    F = rng.standard_normal((D, k)) * 0.3
    sigma = np.diag(rng.uniform(0.05, 0.3, D))
    params = {"F": F, "sigma": sigma, "w0": -1.5, "w": 0.4, "var_z": 1.0, "Z_last": Z}
    m = restore(k, params, Z)
    eta = Z @ F.T
    g, b, p = FAM_IDX["gaussian"], FAM_IDX["bernoulli"], FAM_IDX["poisson"]
    sc = (sst.norm.logpdf(ds.X[:, g], eta[:, g], np.sqrt(np.diag(sigma)[g])).sum()
          + sst.bernoulli.logpmf(ds.X[:, b], 1 / (1 + np.exp(-eta[:, b]))).sum()
          + sst.poisson.logpmf(ds.X[:, p], np.exp(eta[:, p])).sum())
    iu = np.triu_indices(N, 1)
    ey = -1.5 + 0.4 * (Z @ Z.T)[iu]
    sc += sst.bernoulli.logpmf(ds.Y[iu], 1 / (1 + np.exp(-ey))).sum()
    lnpz = -(N * k / 2) * math.log(2 * math.pi) - 0.5 * float(np.sum(Z ** 2))
    q = calc_Q_dual_strict_exp(ds.X, ds.Y, Z[:, :, None], F, sigma, 1.0, -1.5, 0.4, m)
    mine = sum(family_terms(ds.X, ds.Y, Z, F, sigma, -1.5, 0.4)) + lnpz
    checks["strict_Q_equals_scipy_loglik"] = bool(abs((q - lnpz) - sc) / abs(sc) < 1e-10)
    checks["family_terms_equal_strict_Q"] = abs(mine - q) / abs(q) < 1e-12
    _, npar = calc_bic_exp(0.0, k, N, D, "mixed", "bernoulli", n_gaussian_x_cols=5)
    checks["num_params_Kd_minus_rot_plus_5"] = npar == k * D - k * (k - 1) // 2 + 5
    # E-step gradient / precision of the model vs FD of the joint log density per node
    worst_g = worst_p = 0.0
    h = 1e-5
    for i in (0, 17, 99):
        gi = m._calc_gradient(ds.X, ds.Y, Z, F, sigma, 1.0, -1.5, 0.4, i)
        Pi = m._calc_precision_matrix(Z, F, sigma, 1.0, -1.5, 0.4, i)
        fd = np.zeros(k)
        jac = np.zeros((k, k))
        for a in range(k):
            Zp, Zm = Z.copy(), Z.copy()
            Zp[i, a] += h
            Zm[i, a] -= h
            fd[a] = (lk.complete_log_density(m, ds.X, ds.Y, Zp)
                     - lk.complete_log_density(m, ds.X, ds.Y, Zm)) / (2 * h)
            # _calc_gradient returns the gradient of the NEGATIVE log posterior (-grad Phi),
            # so the precision (Hessian of -Phi) is its forward Jacobian.
            jac[:, a] = (m._calc_gradient(ds.X, ds.Y, Zp, F, sigma, 1.0, -1.5, 0.4, i)
                         - m._calc_gradient(ds.X, ds.Y, Zm, F, sigma, 1.0, -1.5, 0.4, i)) / (2 * h)
        worst_g = max(worst_g, float(np.max(np.abs(gi + fd)) / max(1.0, np.max(np.abs(fd)))))
        worst_p = max(worst_p, float(np.max(np.abs(Pi - jac)) / max(1.0, np.max(np.abs(Pi)))))
    notes["estep_fd_rel"] = {"gradient": worst_g, "precision": worst_p}
    checks["estep_gradient_matches_fd"] = worst_g < 1e-5
    checks["estep_precision_matches_fd_jacobian"] = worst_p < 1e-4
    # Laplace evaluator (mixed) gradient / Hessian vs FD
    wg, wh = rc._fd_checks(m, ds.X, ds.Y, Z, rng)
    notes["laplace_fd_rel"] = {"gradient": wg, "hessian": wh}
    checks["laplace_gradient_hessian_match_fd"] = wg < 1e-5 and wh < 1e-4
    lap = rc.evaluate_state(m, ds.X, ds.Y, Z, k)
    checks["laplace_evaluator_runs_on_mixed_model"] = lap["lap_status"] in (
        "OK", "NOT_STATIONARY", "HESSIAN_NOT_PD")
    notes["laplace_status_random_theta"] = lap["lap_status"]
    out.mkdir(parents=True)
    rc.write_csv(out / "generator_checks.csv", gen)
    res = {"created": now(), "git_head": rc.git("rev-parse", "HEAD"), "checks": checks,
           "all_ok": all(v is True for v in checks.values()), "notes": notes}
    rc.write_json(out / "tech_check.json", res)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res["all_ok"] else 1


def pilot(out: Path):
    need(not out.exists(), f"output exists: {out}")
    out.mkdir(parents=True)
    t0 = time.time()
    stopped = run_jobs(jobs_for("pilot"), out)
    rows = rc.read_csv(out / "per_fit.csv")
    ok = [r for r in rows if r["status"] == "OK"]
    # NPZ reload -> C_Lap bitwise (4 states)
    reload_ok = True
    for r in ok[:: max(1, len(ok) // 4)][:4]:
        with np.load(out / "states" / r["state_file"]) as z:
            params = {k: np.array(z[k]) for k in z.files}
        ds = generate(int(r["k_true"]), int(r["data_seed"]))
        e = eval_clap(params, ds.X, ds.Y, int(r["k"]))
        reload_ok &= (e["lap_status"] == r["lap_status"] and str(e["C_Lap"]) == r["C_Lap"])
        reload_ok &= rc.sha256_file(out / "states" / r["state_file"]) == r["state_sha256"]
    rt = [float(r["runtime_s"]) for r in rows]
    by_k = defaultdict(list)
    for r in rows:
        by_k[int(r["k"])].append(float(r["runtime_s"]))
    mean_rt = stats.mean(rt) if rt else float("nan")
    checks = {
        "all_80_recorded": len(rows) == 80 and stopped is None,
        "no_failed_fit": all(r["status"] == "OK" for r in rows),
        "C_Q_recomputed_exact": all(float(r["C_Q_recomputed_absdiff"] or "nan") == 0.0 for r in ok),
        "family_terms_closure": all(float(r["terms_closure_absdiff"]) <= 1e-6 for r in ok),
        "clap_evaluated_every_ok_fit": all(r["lap_status"] in ("OK", "NOT_STATIONARY",
                                                                "HESSIAN_NOT_PD", "ERROR")
                                           for r in ok),
        "npz_reload_reproduces_C_Lap_bitwise": reload_ok,
    }
    proj_h = mean_rt * 600 / WORKERS / 3600 * 1.15
    notes = {"wall_s": round(time.time() - t0, 1), "mean_runtime_s": mean_rt,
             "runtime_by_k": {k: round(stats.mean(v), 1) for k, v in sorted(by_k.items())},
             "projected_main_hours_4_workers": proj_h,
             "lap_status": dict(Counter(r["lap_status"] for r in ok)),
             "mstep_q_decreases_total": sum(int(r["mstep_q_decreases"] or 0) for r in ok),
             "failures": [r["failure_reason"] for r in rows if r["status"] != "OK"][:10]}
    res = {"created": now(), "checks": checks, "all_ok": all(checks.values()), "notes": notes,
           "scope": "technical pilot on pilot seeds; selections are NOT computed or used"}
    rc.write_json(out / "pilot_check.json", res)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res["all_ok"] else 1


def freeze(out: Path, pilot_dir: Path, tech_dir: Path):
    need(not out.exists(), f"output exists: {out}")
    pc = json.loads((pilot_dir / "pilot_check.json").read_text(encoding="utf-8"))
    tc = json.loads((tech_dir / "tech_check.json").read_text(encoding="utf-8"))
    need(pc["all_ok"] and tc["all_ok"], "pilot / tech check not OK")
    jobs = jobs_for("main")
    out.mkdir(parents=True)
    rc.write_csv(out / "manifest.csv", jobs)
    proto = {
        "experiment_id": "k_exp2_mixedx_highk_20261126", "status": "FROZEN_NOT_EXECUTED",
        "frozen_at": now(), "git_head_at_freeze": rc.git("rev-parse", "HEAD"),
        "nature": "EXPLORATORY; planned after single-X 8,000/600-fit and C_Lap results",
        "lineage": "E (DualExpFamLSMPerColumnConsistent via em_runner family_x_list); prototype",
        "design": {"n": N, "d": D, "family_x_list": FAMILY_X_LIST, "family_y": "bernoulli",
                   "generator": "canonical-clean-mixed-v1", "regime": "b (total signal fixed)",
                   "per_k_true": {kt: design(kt) for kt in K_TRUES},
                   "k_true": list(K_TRUES), "candidates": list(K_CANDS), "datasets": N_REP,
                   "inits": [0, 1], "L": L_MC, "num_iter": NUM_ITER,
                   "seeds": "data = 30,261,126,000 + 1000 K_TRUE + rep; init = 10 data + t; "
                            "mc = 10 data + 5 + t"},
        "n_fits": len(jobs), "manifest_sha256": jobs_hash(jobs),
        "code_sha256": {f: rc.sha256_file(ROOT / f) for f in CODE_FILES},
        "calibration_source": {"protocol_v2_sha256": rc.MAIN_PROTOCOL_SHA256},
        "criteria": {"C_Q": "-2 Q_strict + p ln n, p = Kd - K(K-1)/2 + 5",
                     "C_Lap": "laplace_k_criterion v1 via rc.evaluate_state; Z0 = Z_last; "
                              "grad_tol 1e-8; max_iter 200; d_K ln n",
                     "C_Lap_secondary": "NOT_STATIONARY evaluations once with max_iter 400 "
                                        "(pre-specified, reported separately)"},
        "selection_rule": "argmin_K of 2-init unweighted mean; ties <= 1e-9 -> smaller K",
        "failure_rule": {"fit": "fail_fast; exception / non-finite / retry>0 -> FAILED once, "
                                "never re-run or re-seeded; any FAILED fit -> dataset "
                                "INCOMPLETE for both criteria",
                         "clap": "any non-OK of the 20 evaluations -> C_Lap INCOMPLETE",
                         "generator": "GeneratorStop -> dataset INCOMPLETE (no reseed)",
                         "stop": ["family-term closure failure", "mode recomputation "
                                  "mismatch", "code sha256 mismatch", "dirty working tree"]},
        "execution": {"workers": WORKERS, "blas_threads": 1,
                      "resume_rule": "a fit is done iff its fit_id is in per_fit.csv",
                      "memory_rule": f"stop dispatching if available < {MEM_STOP_MB} MB twice",
                      "deadline_rule": "stop dispatching new fits at the deadline; running "
                                       "fits finish; resumable",
                      "states": "states/*.npz (gitignored; sha256 in per_fit.csv)"},
        "metrics": ["exact", "|K_hat - K_TRUE| <= 1", "MAE over complete datasets (n stated)",
                    "under", "over", "INCOMPLETE", "criterion curves K=1..10",
                    "adjacent gaps K_TRUE-1 / K_TRUE+1", "C_Q vs C_Lap crosstab",
                    "family-wise likelihood steps (MC samples and joint mode)",
                    "convergence (relative Q change, M-step decreases)",
                    "init dependence"],
        "interpretation_rules": [
            "match to the generating K_TRUE, not to a population minimal dimension; "
            "identifiability of K for mixed X is unproven (UNRESOLVED)",
            "comparisons with the single-X runs and Phase 9X are across different "
            "generators/families/inits: descriptive only, never a direct causal comparison",
            "an improvement is not attributed to family mixing itself",
            "pilot results were not used to change any setting",
            "INCOMPLETE never imputed; denominator 10"],
        "pilot": {"dir": str(pilot_dir.relative_to(ROOT)), "checks": pc["checks"],
                  "projected_main_hours": pc["notes"]["projected_main_hours_4_workers"]},
        "tech_check": str(tech_dir.relative_to(ROOT)),
    }
    rc.write_json(out / "protocol.json", proto)
    print("protocol sha256", rc.sha256_file(out / "protocol.json"))


def execute(proto_path: Path, out: Path, resume: bool, deadline):
    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    need(proto["code_sha256"] == {f: rc.sha256_file(ROOT / f) for f in CODE_FILES}, "code sha256")
    need(not rc.git("status", "--porcelain"), "working tree must be clean")
    jobs = jobs_for("main")
    need(jobs_hash(jobs) == proto["manifest_sha256"], "manifest")
    if out.exists():
        need(resume, f"output exists (use --resume): {out}")
    else:
        out.mkdir(parents=True)
    rc.write_json(out / "runinfo.json", {"protocol": str(proto_path.relative_to(ROOT)),
                                         "protocol_sha256": rc.sha256_file(proto_path),
                                         "environment": rc.environment(), "started": now()})
    stopped = run_jobs(jobs, out, deadline)
    print("stopped:", stopped)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def num(s):
    """Parse a recorded number; family-term columns were written as 'np.float64(x)'
    (repr of a numpy scalar) by fit_one - same value, different spelling."""
    s = str(s)
    if s.startswith("np.float64(") and s.endswith(")"):
        s = s[len("np.float64("):-1]
    return float(s)


def _sel(f, col, statuscol=None):
    if any(f.get((k, t)) is None for k in K_CANDS for t in (0, 1)):
        return None, {}
    if any(f[(k, t)]["status"] != "OK" for k in K_CANDS for t in (0, 1)):
        return None, {}
    if statuscol and any(f[(k, t)][statuscol] != "OK" for k in K_CANDS for t in (0, 1)):
        return None, {}
    bar = {k: (float(f[(k, 0)][col]) + float(f[(k, 1)][col])) / 2 for k in K_CANDS}
    return rc.pick(bar), bar


def analyze_rows(rows):
    g = defaultdict(dict)
    for r in rows:
        g[r["dataset_id"]][(int(r["k"]), int(r["t"]))] = r
    sel, curves, steps, gaps = [], defaultdict(lambda: defaultdict(list)), \
        defaultdict(lambda: defaultdict(list)), []
    for ds, f in sorted(g.items()):
        r0 = next(iter(f.values()))
        kt = int(r0["k_true"])
        kq, bq = _sel(f, "C_Q")
        kl, bl = _sel(f, "C_Lap", "lap_status")
        # secondary: NOT_STATIONARY replaced by its max_iter-400 result
        f2 = {key: {**v, "lap2_status": v["lap400_status"] if v["lap_status"] == "NOT_STATIONARY"
                    else v["lap_status"], "lap2_C": v["lap400_C_Lap"]
                    if v["lap_status"] == "NOT_STATIONARY" else v["C_Lap"]}
              for key, v in f.items()}
        kl2, _ = _sel(f2, "lap2_C", "lap2_status")
        lab = lambda k: "INCOMPLETE" if k is None else ("exact" if k == kt else  # noqa: E731
                                                        ("under" if k < kt else "over"))
        per_init = {}
        for crit, col, sc in (("Q", "C_Q", None), ("Lap", "C_Lap", "lap_status")):
            for t in (0, 1):
                okt = all(f.get((k, t)) and f[(k, t)]["status"] == "OK"
                          and (sc is None or f[(k, t)][sc] == "OK") for k in K_CANDS)
                per_init[f"k_{crit}_init{t}"] = rc.pick({k: float(f[(k, t)][col])
                                                         for k in K_CANDS}) if okt else ""
        sel.append({"dataset_id": ds, "k_true": kt, "rep": r0["rep"],
                    "k_Q": kq if kq else "INCOMPLETE", "outcome_Q": lab(kq),
                    "k_Lap": kl if kl else "INCOMPLETE", "outcome_Lap": lab(kl),
                    "k_Lap_secondary400": kl2 if kl2 else "INCOMPLETE",
                    "outcome_Lap_secondary400": lab(kl2), **per_init,
                    "lap_failed": ";".join(sorted(f"K{k}t{t}:{v['lap_status']}"
                                                  for (k, t), v in f.items()
                                                  if v["status"] == "OK" and v["lap_status"] != "OK")),
                    "fit_failed": ";".join(sorted(f"K{k}t{t}" for (k, t), v in f.items()
                                                  if v["status"] != "OK"))})
        for crit, bar in (("Q", bq), ("Lap", bl)):
            if bar:
                for k in K_CANDS:
                    curves[(kt, k)][crit].append(bar[k] - bar[kt])
                gaps.append({"dataset_id": ds, "k_true": kt, "criterion": crit,
                             "gap_minus1": bar[kt - 1] - bar[kt],
                             "gap_plus1": bar[kt + 1] - bar[kt] if kt < 10 else ""})
        if bq:
            mean = lambda k, c: (num(f[(k, 0)][c]) + num(f[(k, 1)][c])) / 2   # noqa: E731
            for k in K_CANDS[:-1]:
                for c in ("llX_gauss", "llX_bern", "llX_pois", "llY"):
                    steps[(kt, k)][f"MC_{c}"].append(-2 * (mean(k + 1, c) - mean(k, c)))
                steps[(kt, k)]["C_Q"].append(bq[k + 1] - bq[k])
                if bl:
                    for c in ("mode_llX_gauss", "mode_llX_bern", "mode_llX_pois", "mode_llY"):
                        steps[(kt, k)][c].append(-2 * (mean(k + 1, c) - mean(k, c)))
                    for c in ("lap_zhat_sq", "lap_logdet_H"):
                        steps[(kt, k)][c].append(mean(k + 1, c) - mean(k, c))
                    steps[(kt, k)]["C_Lap"].append(bl[k + 1] - bl[k])
    summ = []
    for kt in sorted({s["k_true"] for s in sel}):
        v = [s for s in sel if s["k_true"] == kt]
        row = {"k_true": kt, "n_datasets": len(v)}
        for crit, col in (("Q", "k_Q"), ("Lap", "k_Lap"), ("Lap400", "k_Lap_secondary400")):
            comp = [s[col] for s in v if s[col] != "INCOMPLETE"]
            ex = sum(k == kt for k in comp)
            lo, hi = rc.clopper_pearson(ex, len(v))
            row.update({f"{crit}_exact": ex, f"{crit}_exact_ci_lo": lo, f"{crit}_exact_ci_hi": hi,
                        f"{crit}_within1": sum(abs(k - kt) <= 1 for k in comp),
                        f"{crit}_under": sum(k < kt for k in comp),
                        f"{crit}_over": sum(k > kt for k in comp),
                        f"{crit}_incomplete": len(v) - len(comp),
                        f"{crit}_MAE_complete": (stats.mean(abs(k - kt) for k in comp)
                                                 if comp else ""),
                        f"{crit}_n_complete": len(comp),
                        f"{crit}_dist": json.dumps(dict(sorted(Counter(comp).items())))})
        both = [s for s in v if s["k_Q"] != "INCOMPLETE" and s["k_Lap"] != "INCOMPLETE"]
        row["n_changed_Q_to_Lap"] = sum(s["k_Q"] != s["k_Lap"] for s in both)
        row["Lap_gt_Q"] = sum(s["k_Lap"] > s["k_Q"] for s in both)
        row["Lap_lt_Q"] = sum(s["k_Lap"] < s["k_Q"] for s in both)
        row["Q_init_disagree"] = sum(1 for s in v if s["k_Q_init0"] != "" and s["k_Q_init1"] != ""
                                     and s["k_Q_init0"] != s["k_Q_init1"])
        row["Lap_init_disagree"] = sum(1 for s in v if s["k_Lap_init0"] != ""
                                       and s["k_Lap_init1"] != ""
                                       and s["k_Lap_init0"] != s["k_Lap_init1"])
        summ.append(row)
    med = lambda x: stats.median(x) if x else ""                     # noqa: E731
    curve_rows = [{"k_true": kt, "k": k, "n_Q": len(d["Q"]), "n_Lap": len(d["Lap"]),
                   "median_CQ_minus_ktrue": med(d["Q"]), "median_CLap_minus_ktrue": med(d["Lap"])}
                  for (kt, k), d in sorted(curves.items())]
    step_rows = []
    for (kt, k), d in sorted(steps.items()):
        step_rows.append({"k_true": kt, "k_from": k, "k_to": k + 1,
                          **{f"median_d_{c}": med(v) for c, v in sorted(d.items())},
                          "P_Z_step": rc.PZ_PER_DIM, "PEN_step": (D - k) * rc.LN_N})
    conv = []
    byk = defaultdict(list)
    for r in rows:
        if r["status"] == "OK":
            byk[(int(r["k_true"]), int(r["k"]))].append(r)
    for (kt, k), v in sorted(byk.items()):
        conv.append({"k_true": kt, "k": k, "n_ok": len(v),
                     "median_q_rel_change_last": med([float(r["q_rel_change_last"]) for r in v]),
                     "max_q_rel_change_last": max(float(r["q_rel_change_last"]) for r in v),
                     "mstep_q_decreases": sum(int(r["mstep_q_decreases"]) for r in v),
                     "lap_status": json.dumps(dict(Counter(r["lap_status"] for r in v))),
                     "median_runtime_s": med([float(r["runtime_s"]) for r in v])})
    checks = {"n_fits": len(rows), "fit_status": dict(Counter(r["status"] for r in rows)),
              "lap_status": dict(Counter(r["lap_status"] for r in rows if r["status"] == "OK")),
              "lap400_status": dict(Counter(r["lap400_status"] for r in rows
                                            if r["lap_status"] == "NOT_STATIONARY")),
              "max_C_Q_recomputed_absdiff": max([float(r["C_Q_recomputed_absdiff"])
                                                 for r in rows if r["status"] == "OK"], default=""),
              "max_terms_closure_absdiff": max([float(r["terms_closure_absdiff"])
                                                for r in rows if r["status"] == "OK"], default="")}
    return {"selection": sel, "summary": summ, "curves": curve_rows, "steps": step_rows,
            "gaps": gaps, "convergence": conv, "checks": checks}


def figures(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kts = sorted({c["k_true"] for c in res["curves"]})
    if not kts:
        return
    fig, axes = plt.subplots(1, len(kts), figsize=(3.6 * len(kts), 3.2), squeeze=False)
    for ax, kt in zip(axes[0], kts):
        v = [c for c in res["curves"] if c["k_true"] == kt]
        for crit, col, color, mk in (("C_Q", "median_CQ_minus_ktrue", "#2a78d6", "o"),
                                     ("C_Lap", "median_CLap_minus_ktrue", "#eb6834", "s")):
            pts = [(c["k"], c[col]) for c in v if c[col] != ""]
            if pts:
                ax.plot([a for a, _ in pts], [b for _, b in pts], f"-{mk}", ms=4, lw=2,
                        color=color, label=crit)
        ax.axhline(0, color="#888", lw=0.8)
        ax.axvline(kt, color="#bbb", lw=0.8, ls="--")
        ax.set_title(f"mixed X, K_TRUE = {kt}", fontsize=9)
        ax.set_xlabel("K")
    axes[0][0].set_ylabel("median of crit(K) - crit(K_TRUE)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_criterion_curves.png", dpi=150)
    plt.close(fig)
    s = res["summary"]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(s))
    ax.bar(x - 0.2, [r["Q_exact"] for r in s], 0.38, color="#2a78d6", label="C_Q")
    ax.bar(x + 0.2, [r["Lap_exact"] for r in s], 0.38, color="#eb6834", label="C_Lap")
    ax.set_xticks(x, [f"K_TRUE={r['k_true']}" for r in s])
    ax.set_ylim(0, 10.8)
    ax.set_ylabel("datasets selecting K_TRUE (of 10)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_exact.png", dpi=150)
    plt.close(fig)


def analyze(run: Path, out: Path):
    need(not out.exists(), f"output exists: {out}")
    rows = rc.read_csv(run / "per_fit.csv")
    res = analyze_rows(rows)
    out.mkdir(parents=True)
    rc.write_csv(out / "selection_by_dataset.csv", res["selection"])
    rc.write_csv(out / "summary_by_cell.csv", res["summary"])
    rc.write_csv(out / "criterion_curves.csv", res["curves"])
    rc.write_csv(out / "term_steps_median.csv", res["steps"])
    rc.write_csv(out / "adjacent_gaps.csv", res["gaps"])
    rc.write_csv(out / "convergence_by_k.csv", res["convergence"])
    fails = [r for r in rows if r["status"] != "OK" or r["lap_status"] != "OK"]
    rc.write_csv(out / "failures.csv", fails or [{"none": ""}])
    rc.write_json(out / "checks.json", res["checks"])
    figures(res, out)
    print(json.dumps(res["checks"], indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech-check")
    ap.add_argument("--pilot")
    ap.add_argument("--freeze")
    ap.add_argument("--pilot-dir")
    ap.add_argument("--tech-dir")
    ap.add_argument("--execute")
    ap.add_argument("--protocol")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--deadline")
    ap.add_argument("--analyze")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.tech_check:
        return tech_check(ROOT / a.tech_check)
    if a.pilot:
        return pilot(ROOT / a.pilot)
    if a.freeze:
        return freeze(ROOT / a.freeze, ROOT / a.pilot_dir, ROOT / a.tech_dir) or 0
    if a.execute:
        return execute(ROOT / a.protocol, ROOT / a.execute, a.resume, a.deadline) or 0
    if a.analyze:
        return analyze(ROOT / a.analyze, ROOT / a.out) or 0
    ap.error("no mode")


if __name__ == "__main__":
    raise SystemExit(main())
