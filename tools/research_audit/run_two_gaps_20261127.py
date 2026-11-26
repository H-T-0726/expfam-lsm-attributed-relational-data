"""Two exploratory experiments planned 2026-11-27 after the research-results compendium.

Experiment A  Scalar-family X misspecification x amount of observed Y.
  n=150, d=15, K_TRUE=3, X Bernoulli in all 15 columns, Y Bernoulli, 20 datasets.
  Signal = K-exp2 rule at K=3 (regime a == regime b at K=3): F = sqrt(2.5) Q (Q orthonormal
  columns), w = 1.0, w0 from the frozen K-exp2 calibration (population edge prob. 0.20).
  Y split: 10% of upper pairs = test (never used for training); remaining pool = dense
  training mask; sparse = random 10% subset of the pool (sparse subset of dense).
  Fit family_x in {bernoulli (correct), poisson (support-compatible misspecification)},
  family_y bernoulli, K = 3, 2 inits, L = 10, 10 MCEM iterations.  160 main fits.
  Primary: Procrustes RMSE_Z of Z_last; per dataset average over the 2 inits;
  Delta = RMSE(wrong) - RMSE(correct); I = Delta_sparse - Delta_dense.

Experiment B  Information-matched Gaussian-X vs Poisson-X at K_TRUE = 9 (regime b).
  n=150, d=15, Y Bernoulli, 10 datasets; within a dataset Z, F, Y (and w, w0) are SHARED;
  only X differs: G01 Gaussian var 0.1 | GM Gaussian var sigma2_matched | P Poisson.
  sigma2_matched = tr(F'F) / tr(E_z[I_P]),  E_z[I_P] = F' diag(exp(||f_l||^2/2)) F
  (trace matching only; eigen-structure recorded).  Candidates K = 1..10, 2 inits,
  L = 10, 10 iterations -> 600 main fits.  Criteria on the same saved state: C_Q and C_Lap
  (frozen K-exp2 evaluator, Z0 = Z_last, grad_tol 1e-8, max_iter 200).

Both: lineage E-scalar (DualExpFamLSMConsistent, numerics_mode='consistent',
failure_policy='fail_fast'), no retry / replacement / seed rescue.  Not the fixed lineage.
C_Q is not Schwarz BIC; C_Lap is not an exact marginal likelihood.  EXPLORATORY.

Modes:
  tech  A|B OUT                       no-fit technical checks
  pilot A|B OUT                       technical pilot on pilot seeds (+1 same-seed repro fit)
  freeze A|B OUT --tech T --pilot P [--doc MD]
  execute A|B OUT --protocol P [--resume] [--deadline ISO]
  analyze A|B RUN --out OUT
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
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_k_exp2_clap_comparison as rc  # noqa: E402  (frozen helpers, used unchanged)

ROOT = rc.ROOT
kd = rc.kd
N, D = kd.N, kd.D
L_MC, NUM_ITER = 10, 10
WORKERS = 4
MEM_STOP_MB = 1500
OUT_BASE = ROOT / "expfam/results/k_selection/two_gaps_20261127"
need, Stop = rc.need, rc.Stop
TIE_TOL = 1e-9

# ---------------- Experiment A constants
A_K = 3
A_N_REP = 20
A_SEED = 31_261_127_000
A_PILOT_REPS = (100, 101)
A_PILOT_OFFSET = 5_000_000
A_TEST_RATIO = 0.10
A_SPARSE_RATE = 0.10
A_RATES = ("dense", "sparse")
A_FAMS = ("bernoulli", "poisson")      # correct, wrong
A_BOOT = 10_000
A_BOOT_SEED = 20261127

# ---------------- Experiment B constants
B_K = 9
B_N_REP = 10
B_SEED = 32_261_127_000
B_PILOT_REP = 100
B_PILOT_OFFSET = 5_000_000
B_K_CANDS = tuple(range(1, 11))
B_CONDS = ("G01", "GM", "P")
B_SIGMA_FIXED = 0.1
B_FX = {"G01": "gaussian", "GM": "gaussian", "P": "poisson"}
B_SERIES = {"G01": "O", "GM": "O", "P": "A"}          # rc.restore_model family lookup
B_STREAM_Y, B_STREAM_XN, B_STREAM_XP = 11, 22, 33      # SeedSequence streams

CODE_FILES = (
    "tools/research_audit/run_two_gaps_20261127.py",
    "tools/research_audit/run_k_exp2_clap_comparison.py",
    "expfam/src/experimental/k_exp2_design.py",
    "expfam/src/experimental/data_generator_canonical.py",
    "expfam/src/experimental/em_runner.py",
    "expfam/src/experimental/model_dual_expfam_consistent.py",
    "expfam/src/experimental/model_dual_expfam_masked.py",
    "expfam/src/experimental/eval_utils.py",
    "expfam/src/experimental/laplace_k_criterion.py",
    "expfam/src/utils_expfam.py",
)
PROTECTED = (   # earlier results: sha256 recorded at freeze, re-checked by the tally
    "expfam/results/k_selection/k_exp2_generalization_20261125/main/per_fit.csv",
    "expfam/results/k_selection/k_exp2_generalization_20261125/clap_comparison/run/evaluations.csv",
    "expfam/results/k_selection/k_exp2_generalization_20261125/mixedx_highk_20261126/run/per_fit.csv",
    "expfam/results/k_selection/k_exp2_generalization_20261125/protocol_v2/protocol.json",
    "expfam/results/misspecification_m1/phase_m1_20261013/paired_rmse_z.csv",
)


def now():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha16(a):
    return kd.array_hash(np.asarray(a, dtype=np.float64))


# =========================================================================
# Experiment A: data and masks
# =========================================================================

def a_design():
    tab = rc.bern_table()
    return {"f_scale": kd.f_scale_of("a", A_K), "w": kd.w_of("O", "a", A_K),
            "w0": float(tab[("a", A_K)]["w0"])}


def a_data_seed(rep, pilot=False):
    return A_SEED + (A_PILOT_OFFSET if pilot else 0) + rep


def a_generate(seed):
    from data_generator_canonical import generate_canonical_data
    dz = a_design()
    return generate_canonical_data(n=N, d=D, k=A_K, seed=seed, family_x="bernoulli",
                                   family_y="bernoulli", f_scale=dz["f_scale"],
                                   sigma_x_var=1.0, w0=dz["w0"], w=dz["w"])


def a_masks(seed):
    from eval_utils import make_pair_split
    pool, test = make_pair_split(N, A_TEST_RATIO, seed=seed + 700_000)
    rows, cols = np.where(np.triu(pool, 1))
    n_keep = max(1, int(round(len(rows) * A_SPARSE_RATE)))
    keep = np.random.default_rng(seed + 800_000).permutation(len(rows))[:n_keep]
    sp = np.zeros((N, N), dtype=bool)
    sp[rows[keep], cols[keep]] = True
    sp |= sp.T
    np.fill_diagonal(sp, False)
    return {"test": test, "dense": pool, "sparse": sp}


def a_fit_id(rep, rate, fam, t, pilot=False):
    return f"{'P-' if pilot else ''}A-r{rep:03d}-{rate}-{fam[:4]}-t{t}"


def a_jobs(phase):
    out = []
    if phase == "pilot":
        for rep in A_PILOT_REPS:
            ds = a_data_seed(rep, True)
            for rate in A_RATES:
                for fam in A_FAMS:
                    out.append(_a_job(phase, rep, ds, rate, fam, 0))
        rp = dict(out[0])
        rp["fit_id"] = rp["fit_id"] + "-repro"
        rp["repro_of"] = out[0]["fit_id"]
        out.append(rp)
        return out
    for rep in range(A_N_REP):
        ds = a_data_seed(rep)
        for rate in A_RATES:
            for fam in A_FAMS:
                for t in (0, 1):
                    out.append(_a_job(phase, rep, ds, rate, fam, t))
    return out


def _a_job(phase, rep, ds, rate, fam, t):
    return {"exp": "A", "phase": phase, "fit_id": a_fit_id(rep, rate, fam, t, phase == "pilot"),
            "dataset_id": f"{'P-' if phase == 'pilot' else ''}A-r{rep:03d}", "rep": rep,
            "rate": rate, "family_x_fit": fam, "t": t, "k": A_K, "data_seed": ds,
            "init_seed": 10 * ds + t, "mc_seed": 10 * ds + 5 + t, "repro_of": ""}


A_FIELDS = ["exp", "phase", "fit_id", "dataset_id", "rep", "rate", "family_x_fit", "t", "k",
            "data_seed", "init_seed", "mc_seed", "repro_of", "status", "failure_reason",
            "x_hash", "y_hash", "test_hash", "train_hash", "n_train_pairs", "n_test_pairs",
            "train_density", "Q_strict", "C_Q", "num_params", "C_Q_recomputed_absdiff",
            "q_rel_change_last", "mstep_q_decreases", "w0_hat", "w_hat",
            "rmse_Z_last", "rmse_Z_mean", "gram_corr_last", "heldout_log_score",
            "retry_count", "replacement_count", "nan_count", "runtime_s", "state_file",
            "state_sha256", "worker_pid"]


def heldout_log_score(Y, Z, w0, w, mask):
    r, c = np.where(np.triu(mask, 1))
    eta = w0 + w * np.sum(Z[r] * Z[c], axis=1)
    y = Y[r, c]
    return float(np.mean(y * eta - np.logaddexp(0.0, eta)))


def procrustes_rmse(Z_est, Z_true):
    from utils_expfam import calc_rmse, procrustes_rotation
    R, kmin = procrustes_rotation(Z_est, Z_true)
    need(kmin == Z_true.shape[1], "procrustes k_min")
    return float(calc_rmse(Z_true[:, :kmin], Z_est[:, :kmin] @ R))


def a_fit_one(job, state_dir):
    from em_runner import run_em_experimental
    t0 = time.perf_counter()
    row = {f: "" for f in A_FIELDS}
    row.update({k: job[k] for k in job if k in A_FIELDS})
    row["worker_pid"] = os.getpid()
    ds = a_generate(job["data_seed"])
    masks = a_masks(job["data_seed"])
    tm = masks[job["rate"]]
    iu = np.triu_indices(N, 1)
    row.update(x_hash=sha16(ds.X), y_hash=sha16(ds.Y), test_hash=sha16(masks["test"]),
               train_hash=sha16(tm), n_train_pairs=int(np.triu(tm, 1).sum()),
               n_test_pairs=int(np.triu(masks["test"], 1).sum()),
               train_density=float(ds.Y[np.triu(tm, 1)].mean()))
    try:
        r = run_em_experimental(ds.X, ds.Y, job["family_x_fit"], "bernoulli", A_K, L=L_MC,
                                num_iter=NUM_ITER, train_mask=tm, numerics_mode="consistent",
                                failure_policy="fail_fast", init_seed=job["init_seed"],
                                mc_seed=job["mc_seed"], compute_strict_Q=True,
                                mstep_q_diagnostic=True)
        q, cq, npar = float(r["Q_strict"]), float(r["bic"]), int(r["num_params"])
        hist = r.get("mstep_q_history") or []
        qa = [h["q_after"] for h in hist]
        finite = all(np.all(np.isfinite(np.asarray(r[k], float))) for k in ("F", "w0", "w", "Z_est"))
        ok = (np.isfinite(q) and np.isfinite(cq) and not r["q_bic_failed"] and finite
              and r["retry_count"] == 0 and r["replacement_count"] == 0)
        row.update(status="OK" if ok else "FAILED",
                   failure_reason="" if ok else (r.get("failure_reason") or "non-finite/integrity"),
                   Q_strict=repr(q), C_Q=repr(cq), num_params=npar,
                   C_Q_recomputed_absdiff=repr(abs(-2 * q + npar * math.log(N) - cq)),
                   q_rel_change_last=(repr(abs(qa[-1] - qa[-2]) / abs(qa[-2])) if len(qa) > 1 else ""),
                   mstep_q_decreases=sum(bool(h["decreased"]) for h in hist),
                   w0_hat=repr(float(r["w0"])), w_hat=repr(float(r["w"])),
                   retry_count=r["retry_count"], replacement_count=r["replacement_count"],
                   nan_count=r["nan_count"])
        if ok:
            Zl, Zm = r["Z_est"], r["Z_samples"].mean(axis=2)
            row.update(rmse_Z_last=repr(procrustes_rmse(Zl, ds.Z)),
                       rmse_Z_mean=repr(procrustes_rmse(Zm, ds.Z)),
                       gram_corr_last=repr(float(np.corrcoef((Zl @ Zl.T)[iu], (ds.Z @ ds.Z.T)[iu])[0, 1])),
                       heldout_log_score=repr(heldout_log_score(ds.Y, Zl, float(r["w0"]), float(r["w"]),
                                                                masks["test"])))
            sp = Path(state_dir) / f"{job['fit_id']}.npz"
            np.savez_compressed(sp, F=r["F"], w0=r["w0"], w=r["w"], Z_last=Zl, Z_mean=Zm)
            row["state_file"] = sp.name
            row["state_sha256"] = rc.sha256_file(sp)
    except rc.Stop:
        raise
    except Exception as exc:                      # recorded once; never retried / re-seeded
        row.update(status="FAILED", failure_reason=f"{type(exc).__name__}: {str(exc)[:300]}")
    row["runtime_s"] = round(time.perf_counter() - t0, 2)
    return row


# =========================================================================
# Experiment B: data (shared Z, F, Y; X per condition)
# =========================================================================

def b_design():
    tab = rc.bern_table()
    return {"f_scale": kd.f_scale_of("b", B_K), "w": kd.w_of("O", "b", B_K),
            "w0": float(tab[("b", B_K)]["w0"])}


def b_data_seed(rep, pilot=False):
    return B_SEED + (B_PILOT_OFFSET if pilot else 0) + rep


def fisher_poisson_expected(F):
    """E_z[F' diag(exp(f_l'z)) F] for z ~ N(0, I):  E[exp(f'z)] = exp(||f||^2 / 2)."""
    return F.T @ (np.exp(0.5 * np.sum(F ** 2, axis=1))[:, None] * F)


def sigma2_matched(F):
    return float(np.trace(F.T @ F) / np.trace(fisher_poisson_expected(F)))


def b_core(seed):
    """Shared latent part: Z, F, Y (independent of the X condition)."""
    from data_generator_canonical import build_full_rank_loadings, canonical_sigmoid
    dz = b_design()
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, B_K))
    F = build_full_rank_loadings(rng, d=D, k=B_K, f_scale=dz["f_scale"])
    ry = np.random.default_rng([seed, B_STREAM_Y])
    iu = np.triu_indices(N, 1)
    eta_y = dz["w0"] + dz["w"] * (Z @ Z.T)[iu]
    Y = np.zeros((N, N))
    Y[iu] = ry.binomial(1, canonical_sigmoid(eta_y)).astype(np.float64)
    Y = Y + Y.T
    return Z, F, Y


def b_generate(seed, cond):
    from data_generator_canonical import DEFAULT_POISSON_LAMBDA_MAX, canonical_poisson_rate
    Z, F, Y = b_core(seed)
    eta = Z @ F.T
    if cond == "P":
        rate = canonical_poisson_rate(eta, lambda_max=DEFAULT_POISSON_LAMBDA_MAX)
        X = np.random.default_rng([seed, B_STREAM_XP]).poisson(rate).astype(np.float64)
        var = float("nan")
    else:
        var = B_SIGMA_FIXED if cond == "G01" else sigma2_matched(F)
        E = np.random.default_rng([seed, B_STREAM_XN]).standard_normal((N, D))
        X = eta + math.sqrt(var) * E
    need(np.all(np.isfinite(X)) and np.all(np.isfinite(Y)), "non-finite data")
    return X, Y, Z, F, var


def b_info(F):
    """Information diagnostics (no data needed)."""
    ip = fisher_poisson_expected(F)
    gram = F.T @ F
    s2 = sigma2_matched(F)
    out = {"tr_FtF": float(np.trace(gram)), "tr_EIP": float(np.trace(ip)), "sigma2_matched": s2,
           "row_norm2_min": float(np.min(np.sum(F ** 2, 1))),
           "row_norm2_max": float(np.max(np.sum(F ** 2, 1)))}
    for name, M in (("G01", gram / B_SIGMA_FIXED), ("GM", gram / s2), ("P", ip)):
        ev = np.linalg.eigvalsh(M)
        out.update({f"{name}_trace": float(np.trace(M)), f"{name}_eig_min": float(ev[0]),
                    f"{name}_eig_max": float(ev[-1]), f"{name}_cond": float(ev[-1] / ev[0]),
                    f"{name}_eigs": json.dumps([float(v) for v in ev])})
    return out


def b_fit_id(cond, rep, k, t, pilot=False):
    return f"{'P-' if pilot else ''}B-{cond}-r{rep:03d}-k{k:02d}-t{t}"


def b_jobs(phase):
    out = []
    if phase == "pilot":
        ds = b_data_seed(B_PILOT_REP, True)
        for cond in B_CONDS:
            for k in B_K_CANDS:
                out.append(_b_job(phase, cond, B_PILOT_REP, ds, k, 0))
        rp = dict(out[-1])           # P, K=10: the slowest/most fragile fit
        rp["fit_id"] += "-repro"
        rp["repro_of"] = out[-1]["fit_id"]
        out.append(rp)
        return out
    for rep in range(B_N_REP):
        ds = b_data_seed(rep)
        for cond in B_CONDS:
            for k in B_K_CANDS:
                for t in (0, 1):
                    out.append(_b_job(phase, cond, rep, ds, k, t))
    return out


def _b_job(phase, cond, rep, ds, k, t):
    return {"exp": "B", "phase": phase, "fit_id": b_fit_id(cond, rep, k, t, phase == "pilot"),
            "dataset_id": f"{'P-' if phase == 'pilot' else ''}B-{cond}-r{rep:03d}",
            "replicate_id": f"{'P-' if phase == 'pilot' else ''}B-r{rep:03d}", "cond": cond,
            "rep": rep, "k": k, "t": t, "data_seed": ds, "init_seed": 10 * ds + t,
            "mc_seed": 10 * ds + 5 + t, "repro_of": ""}


B_FIELDS = ["exp", "phase", "fit_id", "dataset_id", "replicate_id", "cond", "rep", "k", "t",
            "data_seed", "init_seed", "mc_seed", "repro_of", "status", "failure_reason",
            "x_hash", "y_hash", "z_hash", "f_hash", "x_var_true", "y_density", "Q_strict", "C_Q",
            "num_params", "C_Q_recomputed_absdiff", "q_rel_change_last", "mstep_q_decreases",
            "w0_hat", "w_hat", "retry_count", "replacement_count", "nan_count",
            "lap_status", "lap_error", "C_Lap", "lap_log_observed", "lap_phi_at_mode",
            "lap_fit_mode", "lap_zhat_sq", "lap_logdet_H", "lap_logdet_sign", "lap_min_eig_H",
            "lap_grad_inf", "lap_iterations", "lap_penalty", "d_K", "lap_closure_absdiff",
            "z_shift_rms", "lap_seconds", "runtime_s", "state_file", "state_sha256", "worker_pid"]


def b_fit_one(job, state_dir):
    from em_runner import run_em_experimental
    t0 = time.perf_counter()
    row = {f: "" for f in B_FIELDS}
    row.update({k: job[k] for k in job if k in B_FIELDS})
    row["worker_pid"] = os.getpid()
    X, Y, Z, F, var = b_generate(job["data_seed"], job["cond"])
    iu = np.triu_indices(N, 1)
    row.update(x_hash=sha16(X), y_hash=sha16(Y), z_hash=sha16(Z), f_hash=sha16(F),
               x_var_true=repr(var), y_density=float(Y[iu].mean()))
    k = job["k"]
    try:
        r = run_em_experimental(X, Y, B_FX[job["cond"]], "bernoulli", k, L=L_MC, num_iter=NUM_ITER,
                                numerics_mode="consistent", failure_policy="fail_fast",
                                init_seed=job["init_seed"], mc_seed=job["mc_seed"],
                                compute_strict_Q=True, mstep_q_diagnostic=True)
        q, cq, npar = float(r["Q_strict"]), float(r["bic"]), int(r["num_params"])
        hist = r.get("mstep_q_history") or []
        qa = [h["q_after"] for h in hist]
        finite = all(np.all(np.isfinite(np.asarray(r[kk], float))) for kk in ("F", "sigma", "w0", "w"))
        ok = (np.isfinite(q) and np.isfinite(cq) and not r["q_bic_failed"] and finite
              and r["retry_count"] == 0 and r["replacement_count"] == 0)
        row.update(status="OK" if ok else "FAILED",
                   failure_reason="" if ok else (r.get("failure_reason") or "non-finite/integrity"),
                   Q_strict=repr(q), C_Q=repr(cq), num_params=npar,
                   C_Q_recomputed_absdiff=repr(abs(-2 * q + npar * math.log(N) - cq)),
                   q_rel_change_last=(repr(abs(qa[-1] - qa[-2]) / abs(qa[-2])) if len(qa) > 1 else ""),
                   mstep_q_decreases=sum(bool(h["decreased"]) for h in hist),
                   w0_hat=repr(float(r["w0"])), w_hat=repr(float(r["w"])),
                   retry_count=r["retry_count"], replacement_count=r["replacement_count"],
                   nan_count=r["nan_count"])
        if ok:
            st = {"F": np.asarray(r["F"], float), "sigma": np.asarray(r["sigma"], float),
                  "w0": float(r["w0"]), "w": float(r["w"]), "var_z": float(r["var_z"])}
            Z0 = np.asarray(r["Z_est"], float)
            sp = Path(state_dir) / f"{job['fit_id']}.npz"
            np.savez_compressed(sp, Z_last=Z0, Z_mean=r["Z_samples"].mean(axis=2), **st)
            row["state_file"] = sp.name
            row["state_sha256"] = rc.sha256_file(sp)
            model = rc.restore_model(B_SERIES[job["cond"]], k, st, Z0)
            row.update(rc.evaluate_state(model, X, Y, Z0, k))
    except rc.Stop:
        raise
    except Exception as exc:
        row.update(status="FAILED", failure_reason=f"{type(exc).__name__}: {str(exc)[:300]}")
    row["runtime_s"] = round(time.perf_counter() - t0, 2)
    return row


# =========================================================================
# runner
# =========================================================================

def _dispatch(job, state_dir):
    return a_fit_one(job, state_dir) if job["exp"] == "A" else b_fit_one(job, state_dir)


def run_jobs(jobs, fields, out: Path, deadline=None):
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
                pending.add(ex.submit(_dispatch, j, str(state_dir)))
            if not pending:
                break
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in fin:
                rc.append_csv(pf, [f.result()], fields)
                n += 1
                if n % 10 == 0:
                    print(f"{n}/{len(todo)} fits ({time.time() - t0:.0f}s)", flush=True)
    rc.append_jsonl(out / "run_events.jsonl", {"event": stopped or "finished", "time": now(),
                                               "fits_run": n, "wall_s": round(time.time() - t0, 1)})
    return stopped


# =========================================================================
# technical checks (no fit)
# =========================================================================

def _fd_grad(f, z, h=1e-6):
    g = np.zeros_like(z)
    for a in range(z.size):
        e = np.zeros_like(z)
        e[a] = h
        g[a] = (f(z + e) - f(z - e)) / (2 * h)
    return g


def tech_A(out: Path):
    from em_runner import build_model
    from eval_utils import calc_Q_dual_strict_exp
    checks, notes = {}, {}
    gen = []
    for pilot, reps in ((False, range(A_N_REP)), (True, A_PILOT_REPS)):
        for rep in reps:
            s = a_data_seed(rep, pilot)
            ds, ds2 = a_generate(s), a_generate(s)
            m, m2 = a_masks(s), a_masks(s)
            iu = np.triu_indices(N, 1)
            te, de, sp = (np.triu(m[x], 1) for x in ("test", "dense", "sparse"))
            gen.append({"pilot": pilot, "rep": rep, "data_seed": s,
                        "x01": bool(np.isin(ds.X, (0, 1)).all()), "y01": bool(np.isin(ds.Y, (0, 1)).all()),
                        "rank_F": int(np.linalg.matrix_rank(ds.F)), "x_mean": float(ds.X.mean()),
                        "y_density": float(ds.Y[iu].mean()),
                        "test_pairs": int(te.sum()), "dense_pairs": int(de.sum()), "sparse_pairs": int(sp.sum()),
                        "test_dense_disjoint": bool(not (te & de).any()),
                        "cover_all": bool((te | de).sum() == len(iu[0])),
                        "sparse_subset_dense": bool(not (sp & ~de).any()),
                        "sym_diag": all(bool((m[x] == m[x].T).all() and not m[x].diagonal().any())
                                        for x in m),
                        "deterministic": bool(np.array_equal(ds.X, ds2.X) and np.array_equal(ds.Y, ds2.Y)
                                              and all(np.array_equal(m[x], m2[x]) for x in m)),
                        "y_symmetric": bool((ds.Y == ds.Y.T).all())})
    rc.write_csv(out / "generator_checks.csv", gen)
    tot = N * (N - 1) // 2
    checks["generator_support_rank_masks"] = all(
        g["x01"] and g["y01"] and g["rank_F"] == A_K and g["test_dense_disjoint"] and g["cover_all"]
        and g["sparse_subset_dense"] and g["sym_diag"] and g["deterministic"] and g["y_symmetric"]
        and g["test_pairs"] == round(tot * A_TEST_RATIO)
        and g["sparse_pairs"] == round(g["dense_pairs"] * A_SPARSE_RATE) for g in gen)
    # ---- mask handling in E-step / M-step / strict Q (both fitted families)
    s = a_data_seed(A_PILOT_REPS[0], True)
    ds = a_generate(s)
    m = a_masks(s)
    rng = np.random.default_rng(1)
    mask_rows = []
    for fam in A_FAMS:
        model = build_model(N, D, A_K, L_MC, fam, "bernoulli", train_mask=m["sparse"],
                            numerics_mode="consistent")
        model.initialize_params(true_params=None, seed=0)
        F = ds.F + 0.05 * rng.standard_normal(ds.F.shape)
        sigma = model.params["sigma"]
        w0, w = a_design()["w0"], 0.8
        Z = ds.Z + 0.1 * rng.standard_normal(ds.Z.shape)
        Yflip = ds.Y.copy()
        out_pairs = np.triu(~m["sparse"], 1) & np.triu(np.ones((N, N), bool), 1)
        Yflip[out_pairs] = 1 - Yflip[out_pairs]
        Yflip = np.triu(Yflip, 1) + np.triu(Yflip, 1).T
        for i in (0, 7, 77):
            def logpost(zi, Yuse=ds.Y):
                Zt = Z.copy()
                Zt[i] = zi
                eta_x = Zt[i] @ F.T
                if fam == "bernoulli":
                    lx = float(np.sum(ds.X[i] * eta_x - np.logaddexp(0, eta_x)))
                else:
                    lx = float(np.sum(ds.X[i] * eta_x - np.exp(eta_x)))
                js = np.where(m["sparse"][i])[0]
                ey = w0 + w * (Zt[js] @ Zt[i])
                ly = float(np.sum(Yuse[i, js] * ey - np.logaddexp(0, ey)))
                return -0.5 * float(zi @ zi) + lx + ly
            g_fd = _fd_grad(logpost, Z[i].copy())
            g_m = np.asarray(model._calc_gradient(ds.X, ds.Y, Z, F, sigma, 1.0, w0, w, i), float)
            g_flip = np.asarray(model._calc_gradient(ds.X, Yflip, Z, F, sigma, 1.0, w0, w, i), float)
            sign = 1.0 if np.linalg.norm(g_m - g_fd) <= np.linalg.norm(g_m + g_fd) else -1.0
            rel_g = float(np.linalg.norm(g_m - sign * g_fd) / max(np.linalg.norm(g_fd), 1e-12))
            H_fd = np.zeros((A_K, A_K))
            for a in range(A_K):
                e = np.zeros(A_K)
                e[a] = 1e-5
                H_fd[:, a] = (_fd_grad(logpost, Z[i] + e) - _fd_grad(logpost, Z[i] - e)) / 2e-5
            P_m = np.asarray(model._calc_precision_matrix(Z, F, sigma, 1.0, w0, w, i), float)
            rel_h = float(np.linalg.norm(P_m + H_fd) / max(np.linalg.norm(H_fd), 1e-12))
            mask_rows.append({"family": fam, "i": i, "grad_sign_vs_logpost": sign, "grad_rel_err": rel_g,
                              "precision_rel_err_vs_neg_hessian": rel_h,
                              "grad_invariant_to_unobserved_Y": bool(np.array_equal(g_m, g_flip))})
        Zs = np.repeat(Z[:, :, None], 3, axis=2) + 0.05 * rng.standard_normal((N, A_K, 3))
        q1 = calc_Q_dual_strict_exp(ds.X, ds.Y, Zs, F, sigma, 1.0, w0, w, model)
        q2 = calc_Q_dual_strict_exp(ds.X, Yflip, Zs, F, sigma, 1.0, w0, w, model)
        w0a = model.calc_w0(ds.Y, Zs, w0, w)
        w0b = model.calc_w0(Yflip, Zs, w0, w)
        mask_rows.append({"family": fam, "i": "Q/M-step", "strictQ_invariant_to_unobserved_Y": bool(q1 == q2),
                          "calc_w0_invariant_to_unobserved_Y": bool(np.allclose(w0a, w0b, rtol=0, atol=0)),
                          "Q_value": q1})
    rc.write_csv(out / "mask_checks.csv", mask_rows,
                 fields=sorted({k for r in mask_rows for k in r}))
    gr = [r for r in mask_rows if r["i"] != "Q/M-step"]
    qr = [r for r in mask_rows if r["i"] == "Q/M-step"]
    checks["estep_gradient_fd"] = all(r["grad_rel_err"] < 1e-5 for r in gr)
    checks["estep_precision_fd"] = all(r["precision_rel_err_vs_neg_hessian"] < 1e-4 for r in gr)
    checks["estep_ignores_unobserved_Y"] = all(r["grad_invariant_to_unobserved_Y"] for r in gr)
    checks["strictQ_ignores_unobserved_Y"] = all(r["strictQ_invariant_to_unobserved_Y"] for r in qr)
    checks["mstep_w0_ignores_unobserved_Y"] = all(r["calc_w0_invariant_to_unobserved_Y"] for r in qr)
    notes["design"] = a_design()
    notes["grad_sign"] = sorted({r["grad_sign_vs_logpost"] for r in gr})
    rc.write_json(out / "tech_checks.json", {"checks": checks, "notes": notes, "time": now(),
                                             "all_pass": all(checks.values())})
    print(json.dumps(checks, indent=1))
    return all(checks.values())


def tech_B(out: Path):
    checks, notes = {}, {}
    rng = np.random.default_rng(7)
    # 1. E_z[exp(f'z)] = exp(||f||^2/2) and the expected Poisson Fisher matrix (Monte Carlo)
    _, F, _ = b_core(b_data_seed(B_PILOT_REP, True))
    zs = rng.standard_normal((400_000, B_K))
    ex = np.exp(zs @ F.T)                                   # (M, d)
    mc = F.T @ (ex.mean(0)[:, None] * F)
    rel_mc = float(np.linalg.norm(mc - fisher_poisson_expected(F)) / np.linalg.norm(fisher_poisson_expected(F)))
    checks["expected_poisson_fisher_mc"] = rel_mc < 5e-3
    notes["expected_poisson_fisher_mc_rel_err"] = rel_mc
    # 2. Hessians of the per-node log-likelihood (finite differences): Poisson and Gaussian
    x = rng.poisson(1.0, size=D).astype(float)
    z0 = rng.standard_normal(B_K) * 0.5
    # v2 (2026-11-27): v1 differentiated the log-likelihood twice numerically (nested FD,
    # rel. error ~2.6e-4 > its 1e-5 gate; record kept in tech_check/).  v2 checks in two
    # single-level steps: FD(log-lik) vs analytic gradient, then FD(analytic gradient) vs -Fisher.
    def _jac(g, z, h=1e-6):
        return np.column_stack([(g(z + h * np.eye(B_K)[a]) - g(z - h * np.eye(B_K)[a])) / (2 * h)
                                for a in range(B_K)])
    lp = lambda z: float(np.sum(x * (F @ z) - np.exp(F @ z)))            # noqa: E731
    gp = lambda z: F.T @ (x - np.exp(F @ z))                              # noqa: E731
    Ip = F.T @ (np.exp(F @ z0)[:, None] * F)
    e_gp = float(np.linalg.norm(_fd_grad(lp, z0) - gp(z0)) / np.linalg.norm(gp(z0)))
    e_hp = float(np.linalg.norm(_jac(gp, z0) + Ip) / np.linalg.norm(Ip))
    s2 = 0.37
    xg = rng.standard_normal(D)
    lg = lambda z: float(-0.5 * np.sum((xg - F @ z) ** 2) / s2)          # noqa: E731
    gg = lambda z: F.T @ (xg - F @ z) / s2                                # noqa: E731
    e_gg = float(np.linalg.norm(_fd_grad(lg, z0) - gg(z0)) / np.linalg.norm(gg(z0)))
    e_hg = float(np.linalg.norm(_jac(gg, z0) + F.T @ F / s2) / np.linalg.norm(F.T @ F / s2))
    checks["poisson_gradient_fd"] = e_gp < 1e-6
    checks["poisson_fisher_equals_neg_hessian"] = e_hp < 1e-6
    checks["gaussian_gradient_fd"] = e_gg < 1e-6
    checks["gaussian_fisher_equals_neg_hessian"] = e_hg < 1e-6
    notes["fd_rel_errors"] = {"poisson_grad": e_gp, "poisson_hess": e_hp, "gauss_grad": e_gg, "gauss_hess": e_hg}
    # 3. generator: shared Z/F/Y, trace matching, support, determinism, density
    rows = []
    for pilot, reps in ((False, range(B_N_REP)), (True, (B_PILOT_REP,))):
        for rep in reps:
            s = b_data_seed(rep, pilot)
            gen = {c: b_generate(s, c) for c in B_CONDS}
            again = b_generate(s, "P")
            Z, F, Y = gen["G01"][2], gen["G01"][3], gen["G01"][1]
            info = b_info(F)
            iu = np.triu_indices(N, 1)
            row = {"pilot": pilot, "rep": rep, "data_seed": s, **info,
                   "shared_ZFY": all(np.array_equal(gen[c][2], Z) and np.array_equal(gen[c][3], F)
                                     and np.array_equal(gen[c][1], Y) for c in B_CONDS),
                   "x_differs": bool(not np.array_equal(gen["G01"][0], gen["GM"][0])),
                   "deterministic": bool(np.array_equal(again[0], gen["P"][0])),
                   "rank_F": int(np.linalg.matrix_rank(F)), "y_density": float(Y[iu].mean()),
                   "y01_sym": bool(np.isin(Y, (0, 1)).all() and (Y == Y.T).all() and not Y.diagonal().any()),
                   "p_x_max": float(gen["P"][0].max()), "p_x_mean": float(gen["P"][0].mean()),
                   "p_x_int_nonneg": bool(np.all(gen["P"][0] >= 0) and np.all(gen["P"][0] == np.round(gen["P"][0]))),
                   "GM_var": gen["GM"][4],
                   "trace_match_relerr": abs(info["GM_trace"] - info["P_trace"]) / info["P_trace"]}
            rows.append(row)
    rc.write_csv(out / "information_diagnostics.csv", rows)
    checks["shared_Z_F_Y_across_conditions"] = all(r["shared_ZFY"] for r in rows)
    checks["trace_matching_exact"] = all(r["trace_match_relerr"] < 1e-12 for r in rows)
    checks["generator_support_rank_determinism"] = all(
        r["rank_F"] == B_K and r["y01_sym"] and r["p_x_int_nonneg"] and r["deterministic"]
        and r["x_differs"] for r in rows)
    notes["design"] = b_design()
    notes["sigma2_matched_range"] = [min(r["GM_var"] for r in rows), max(r["GM_var"] for r in rows)]
    notes["y_density_range"] = [min(r["y_density"] for r in rows), max(r["y_density"] for r in rows)]
    rc.write_json(out / "tech_checks.json", {"checks": checks, "notes": notes, "time": now(),
                                             "all_pass": all(checks.values())})
    print(json.dumps(checks, indent=1), json.dumps(notes, indent=1))
    return all(checks.values())


# =========================================================================
# freeze / execute
# =========================================================================

def freeze(exp, out: Path, tech: Path, pilot: Path, doc: Path | None):
    need(not (out / "protocol.json").exists(), "protocol exists")
    tj = json.loads((tech / "tech_checks.json").read_text(encoding="utf-8"))
    need(tj["all_pass"], "technical checks did not all pass")
    prow = rc.read_csv(pilot / "per_fit.csv")
    jobs = a_jobs("main") if exp == "A" else b_jobs("main")
    out.mkdir(parents=True, exist_ok=True)
    proto = {"experiment": exp, "frozen_at": now(), "git_head": rc.git("rev-parse", "HEAD"),
             "git_dirty": rc.git("status", "--porcelain"), "code_sha256": {f: rc.sha256_file(ROOT / f) for f in CODE_FILES},
             "protected_sha256": {f: rc.sha256_file(ROOT / f) for f in PROTECTED},
             "tech_dir": str(tech.relative_to(ROOT)), "pilot_dir": str(pilot.relative_to(ROOT)),
             "pilot_status": {r["fit_id"]: r["status"] for r in prow},
             "n_jobs": len(jobs), "jobs_sha256": __import__("hashlib").sha256(
                 json.dumps(jobs, sort_keys=True).encode()).hexdigest(),
             "frozen_doc": str(doc.relative_to(ROOT)) if doc else "",
             "frozen_doc_sha256": rc.sha256_file(doc) if doc else "",
             "design": a_design() if exp == "A" else b_design()}
    rc.write_json(out / "protocol.json", proto)
    rc.write_csv(out / "manifest.csv", jobs)
    print("frozen", exp, len(jobs), "jobs")


def execute(exp, out: Path, proto_path: Path, resume: bool, deadline):
    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    need(proto["experiment"] == exp, "protocol/experiment mismatch")
    for f, h in proto["code_sha256"].items():
        need(rc.sha256_file(ROOT / f) == h, f"code changed since freeze: {f}")
    jobs = a_jobs("main") if exp == "A" else b_jobs("main")
    need(__import__("hashlib").sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest()
         == proto["jobs_sha256"], "jobs differ from frozen manifest")
    if (out / "per_fit.csv").exists():
        need(resume, "per_fit.csv exists: use --resume")
    out.mkdir(parents=True, exist_ok=True)
    return run_jobs(jobs, A_FIELDS if exp == "A" else B_FIELDS, out, deadline)


def pilot_run(exp, out: Path):
    need(not (out / "per_fit.csv").exists(), "pilot output exists")
    out.mkdir(parents=True, exist_ok=True)
    jobs = a_jobs("pilot") if exp == "A" else b_jobs("pilot")
    run_jobs(jobs, A_FIELDS if exp == "A" else B_FIELDS, out)
    rows = rc.read_csv(out / "per_fit.csv")
    by = {r["fit_id"]: r for r in rows}
    rep = [r for r in rows if r["repro_of"]]
    keys = ["Q_strict", "C_Q"] + (["rmse_Z_last"] if exp == "A" else ["C_Lap", "lap_status"])
    repro = all(by[r["repro_of"]][k] == r[k] for r in rep for k in keys)
    summ = {"fits": len(rows), "ok": sum(r["status"] == "OK" for r in rows), "same_seed_reproduction": repro,
            "runtime_s": [float(r["runtime_s"]) for r in rows]}
    if exp == "B":
        summ["lap_status"] = {s: sum(r["lap_status"] == s for r in rows) for s in {r["lap_status"] for r in rows}}
    rc.write_json(out / "pilot_summary.json", summ)
    print(json.dumps({k: v for k, v in summ.items() if k != "runtime_s"}), "mean runtime",
          np.mean(summ["runtime_s"]))


# =========================================================================
# analysis
# =========================================================================

def fnum(s):
    s = str(s).strip()
    return float(s) if s not in ("", "nan") else float("nan")


def boot_ci(v, seed=A_BOOT_SEED, B=A_BOOT):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def t_ci(v):
    from scipy import stats as sst
    v = np.asarray(v, float)
    se = v.std(ddof=1) / math.sqrt(len(v))
    q = sst.t.ppf(0.975, len(v) - 1)
    return float(v.mean() - q * se), float(v.mean() + q * se)


def summ(v):
    from scipy import stats as sst
    v = np.asarray(v, float)
    npos = int((v > 0).sum())
    return {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
            "sd": float(v.std(ddof=1)), "n_pos": npos, "n_neg": int((v < 0).sum()),
            "boot95_lo": boot_ci(v)[0], "boot95_hi": boot_ci(v)[1],
            "t95_lo": t_ci(v)[0], "t95_hi": t_ci(v)[1],
            "sign_test_p_two_sided": float(sst.binomtest(npos, int((v != 0).sum()), 0.5).pvalue)}


def analyze_A(run: Path, out: Path):
    rows = [r for r in rc.read_csv(run / "per_fit.csv") if r["phase"] == "main"]
    out.mkdir(parents=True, exist_ok=True)
    fails = [r for r in rows if r["status"] != "OK"]
    rc.write_csv(out / "failure_log.csv", fails or [{"fit_id": "", "status": "none"}],
                 fields=["fit_id", "status", "failure_reason"])
    cell = defaultdict(list)
    for r in rows:
        if r["status"] == "OK":
            cell[(r["dataset_id"], r["rate"], r["family_x_fit"])].append(r)
    per_ds = []
    for rep in range(A_N_REP):
        dsid = f"A-r{rep:03d}"
        rec = {"dataset_id": dsid}
        complete = True
        for rate in A_RATES:
            for fam in A_FAMS:
                rr = cell[(dsid, rate, fam)]
                if len(rr) != 2:
                    complete = False
                    continue
                for met in ("rmse_Z_last", "rmse_Z_mean", "heldout_log_score"):
                    rec[f"{met}_{rate}_{fam}"] = float(np.mean([fnum(x[met]) for x in rr]))
                rec[f"rmse_Z_last_{rate}_{fam}_init_absdiff"] = abs(fnum(rr[0]["rmse_Z_last"]) - fnum(rr[1]["rmse_Z_last"]))
        rec["complete"] = complete
        if complete:
            for met in ("rmse_Z_last", "rmse_Z_mean"):
                for rate in A_RATES:
                    rec[f"delta_{met}_{rate}"] = rec[f"{met}_{rate}_poisson"] - rec[f"{met}_{rate}_bernoulli"]
                rec[f"I_{met}"] = rec[f"delta_{met}_sparse"] - rec[f"delta_{met}_dense"]
            for rate in A_RATES:   # held-out score: correct - wrong (positive = correct better)
                rec[f"delta_heldout_{rate}"] = rec[f"heldout_log_score_{rate}_bernoulli"] - rec[f"heldout_log_score_{rate}_poisson"]
        per_ds.append(rec)
    rc.write_csv(out / "per_dataset.csv", per_ds, fields=sorted({k for r in per_ds for k in r}))
    comp = [r for r in per_ds if r["complete"]]
    prim = []
    for name, key, role in (("I (RMSE_Z last)", "I_rmse_Z_last", "PRIMARY"),
                            ("Delta dense (RMSE_Z last)", "delta_rmse_Z_last_dense", "component"),
                            ("Delta sparse (RMSE_Z last)", "delta_rmse_Z_last_sparse", "component"),
                            ("I (RMSE_Z mean of L samples)", "I_rmse_Z_mean", "secondary"),
                            ("Delta dense (RMSE_Z mean)", "delta_rmse_Z_mean_dense", "secondary"),
                            ("Delta sparse (RMSE_Z mean)", "delta_rmse_Z_mean_sparse", "secondary"),
                            ("held-out score correct-wrong dense", "delta_heldout_dense", "secondary"),
                            ("held-out score correct-wrong sparse", "delta_heldout_sparse", "secondary")):
        v = [r[key] for r in comp]
        prim.append({"quantity": name, "column": key, "role": role, "datasets_complete": len(v),
                     "datasets_total": A_N_REP, **(summ(v) if len(v) >= 2 else {})})
    for rate in A_RATES:
        for fam in A_FAMS:
            v = [r[f"rmse_Z_last_{rate}_{fam}"] for r in comp]
            prim.append({"quantity": f"RMSE_Z last {rate} {fam}", "column": f"rmse_Z_last_{rate}_{fam}",
                         "role": "level", "datasets_complete": len(v), "datasets_total": A_N_REP,
                         "n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)),
                         "sd": float(np.std(v, ddof=1))})
    rc.write_csv(out / "primary_results.csv", prim, fields=sorted({k for r in prim for k in r},
                                                                 key=lambda k: (k not in ("quantity", "role"), k)))
    _fig_A(comp, out)
    rc.write_json(out / "analysis_summary.json", {"fits": len(rows), "ok": len(rows) - len(fails),
                                                  "failed": len(fails), "datasets_complete": len(comp),
                                                  "primary": prim[0]})
    print(json.dumps(prim[:3], indent=1))


def _fig_A(comp, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for i, rate in enumerate(A_RATES):
        v = [r[f"delta_rmse_Z_last_{rate}"] for r in comp]
        ax.scatter(i + np.linspace(-0.15, 0.15, len(v)), v, s=18, color=("#2a78d6", "#eb6834")[i], zorder=3)
        ax.hlines(np.mean(v), i - 0.25, i + 0.25, color="k", lw=2)
    ax.axhline(0, color="#52514e", lw=1)
    ax.set_xticks([0, 1], ["dense Y (100% of pool)", "sparse Y (10% of pool)"])
    ax.set_ylabel("Delta RMSE_Z (Poisson - Bernoulli)\n(points = datasets, bar = mean)")
    ax.set_title("Per-dataset misspecification effect")
    ax = axes[1]
    v = [r["I_rmse_Z_last"] for r in comp]
    ax.scatter(np.linspace(-0.15, 0.15, len(v)), v, s=18, color="#6250d6", zorder=3)
    ax.hlines(np.mean(v), -0.25, 0.25, color="k", lw=2)
    ax.axhline(0, color="#52514e", lw=1)
    ax.set_xticks([0], ["I = Delta_sparse - Delta_dense"])
    ax.set_title("Primary contrast I (per dataset)")
    fig.suptitle(f"Exp A (lineage E-scalar, all-Bernoulli X, n=150, d=15, K=3; {len(comp)} datasets, mean of 2 inits)", fontsize=10)
    fig.text(0.01, 0.01, "Source: per_dataset.csv (from per_fit.csv). Exploratory.", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(fd / "figA_delta_and_interaction.png", dpi=130)
    plt.close(fig)


def _sel(d):
    m = min(d.values())
    return min(k for k, v in d.items() if v <= m + TIE_TOL)


def analyze_B(run: Path, out: Path):
    rows = [r for r in rc.read_csv(run / "per_fit.csv") if r["phase"] == "main"]
    out.mkdir(parents=True, exist_ok=True)
    fails = [r for r in rows if r["status"] != "OK" or (r["lap_status"] not in ("OK",))]
    rc.write_csv(out / "failure_log.csv", fails or [{"fit_id": "", "status": "none"}],
                 fields=["fit_id", "status", "failure_reason", "lap_status", "lap_error", "lap_grad_inf",
                         "lap_min_eig_H"])
    by = defaultdict(list)
    for r in rows:
        by[r["dataset_id"]].append(r)
    sel_rows = []
    for rep in range(B_N_REP):
        for cond in B_CONDS:
            dsid = f"B-{cond}-r{rep:03d}"
            rr = by[dsid]
            rec = {"dataset_id": dsid, "replicate_id": f"B-r{rep:03d}", "cond": cond, "rep": rep,
                   "n_fits": len(rr), "n_fit_ok": sum(r["status"] == "OK" for r in rr)}
            for crit, col, stcol in (("Q", "C_Q", None), ("Lap", "C_Lap", "lap_status")):
                d = defaultdict(list)
                bad = False
                for r in rr:
                    if r["status"] != "OK" or (stcol and r[stcol] != "OK"):
                        bad = True
                        continue
                    d[int(r["k"])].append(fnum(r[col]))
                complete = (not bad) and len(rr) == 20 and all(len(d[k]) == 2 for k in B_K_CANDS)
                if complete:
                    means = {k: float(np.mean(v)) for k, v in d.items()}
                    k_hat = _sel(means)
                    rec[f"k_{crit}"] = k_hat
                    rec[f"outcome_{crit}"] = "exact" if k_hat == B_K else ("under" if k_hat < B_K else "over")
                    rec[f"margin_{crit}"] = sorted(means.values())[1] - min(means.values())
                    rec[f"gap_ktrue_{crit}"] = means[B_K] - min(means.values())
                    for t in (0, 1):
                        dt = {int(r["k"]): fnum(r[col]) for r in rr if int(r["t"]) == t}
                        rec[f"k_{crit}_t{t}"] = _sel(dt)
                    for k in B_K_CANDS:
                        rec[f"{crit}_mean_k{k:02d}"] = means[k]
                else:
                    rec[f"k_{crit}"] = ""
                    rec[f"outcome_{crit}"] = "INCOMPLETE"
            sel_rows.append(rec)
    rc.write_csv(out / "per_dataset_selection.csv", sel_rows, fields=sorted({k for r in sel_rows for k in r}))
    prim = []
    for cond in B_CONDS:
        for crit in ("Q", "Lap"):
            rr = [r for r in sel_rows if r["cond"] == cond]
            c = {o: sum(r[f"outcome_{crit}"] == o for r in rr) for o in ("exact", "under", "over", "INCOMPLETE")}
            ks = [r[f"k_{crit}"] for r in rr if r[f"k_{crit}"] != ""]
            prim.append({"cond": cond, "criterion": "C_Q" if crit == "Q" else "C_Lap", "datasets": len(rr), **c,
                         "selected_k": json.dumps(sorted(ks)),
                         "mean_k_minus_ktrue": float(np.mean([k - B_K for k in ks])) if ks else "",
                         "inits_disagree": sum(1 for r in rr if r.get(f"k_{crit}_t0", "") != ""
                                               and r[f"k_{crit}_t0"] != r[f"k_{crit}_t1"])})
    rc.write_csv(out / "primary_results.csv", prim)
    # paired by replicate
    pair = []
    for rep in range(B_N_REP):
        g = {r["cond"]: r for r in sel_rows if r["rep"] == rep}
        rec = {"replicate_id": f"B-r{rep:03d}"}
        for crit in ("Q", "Lap"):
            for c in B_CONDS:
                rec[f"k_{crit}_{c}"] = g[c][f"k_{crit}"]
        pair.append(rec)
    rc.write_csv(out / "paired_by_replicate.csv", pair)
    # criterion curves (median of C(K) - C(K_TRUE) over complete datasets)
    curves = []
    for cond in B_CONDS:
        for crit in ("Q", "Lap"):
            rr = [r for r in sel_rows if r["cond"] == cond and r[f"outcome_{crit}"] != "INCOMPLETE"]
            for k in B_K_CANDS:
                v = [r[f"{crit}_mean_k{k:02d}"] - r[f"{crit}_mean_k{B_K:02d}"] for r in rr]
                curves.append({"cond": cond, "criterion": crit, "k": k, "n": len(v),
                               "median_C_minus_C_ktrue": float(np.median(v)) if v else ""})
    rc.write_csv(out / "criterion_curves.csv", curves)
    _fig_B(sel_rows, curves, out)
    rc.write_json(out / "analysis_summary.json", {"fits": len(rows), "fit_ok": sum(r["status"] == "OK" for r in rows),
                                                  "lap_status": {s: sum(r["lap_status"] == s for r in rows)
                                                                 for s in {r["lap_status"] for r in rows}},
                                                  "primary": prim})
    print(json.dumps(prim, indent=1))


def _fig_B(sel_rows, curves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    col = {"exact": "#2a78d6", "under": "#eb6834", "over": "#6250d6", "INCOMPLETE": "#9a9994"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, crit in zip(axes, ("Q", "Lap")):
        left = np.zeros(3)
        for o in ("exact", "under", "over", "INCOMPLETE"):
            v = np.array([sum(r[f"outcome_{crit}"] == o for r in sel_rows if r["cond"] == c) for c in B_CONDS])
            ax.barh(list(B_CONDS), v, left=left, color=col[o], label=o, edgecolor="white",
                    hatch="///" if o == "INCOMPLETE" else None)
            left += v
        ax.set_title("C_Q" if crit == "Q" else "C_Lap")
        ax.set_xlabel("datasets (of 10)")
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(1.05, -0.2))
    fig.suptitle("Exp B (K_TRUE=9, regime b; G01: Gaussian var 0.1, GM: trace-matched var, P: Poisson; shared Z,F,Y)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(fd / "figB_selection_outcomes.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, cond in zip(axes, B_CONDS):
        for crit, c in (("Q", "#eb6834"), ("Lap", "#2a78d6")):
            cc = [x for x in curves if x["cond"] == cond and x["criterion"] == crit and x["median_C_minus_C_ktrue"] != ""]
            if cc:
                ax.plot([x["k"] for x in cc], [x["median_C_minus_C_ktrue"] for x in cc], marker="o", ms=4, color=c,
                        label=f"{'C_Q' if crit == 'Q' else 'C_Lap'} (n={cc[0]['n']})")
        ax.axhline(0, color="#52514e", lw=1)
        ax.axvline(B_K, color="#9a9994", ls="--", lw=1)
        ax.set_title(cond)
        ax.set_xlabel("candidate K")
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("median over complete datasets\nC(K) - C(K_TRUE)")
    fig.suptitle("Exp B criterion curves (absolute values are not compared across families)", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(fd / "figB_criterion_curves.png", dpi=130)
    plt.close(fig)


# =========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tech", "pilot", "freeze", "execute", "analyze"])
    ap.add_argument("exp", choices=["A", "B"])
    ap.add_argument("out")
    ap.add_argument("--tech")
    ap.add_argument("--pilot")
    ap.add_argument("--doc")
    ap.add_argument("--protocol")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--deadline")
    ap.add_argument("--run")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if a.mode == "tech":
        out.mkdir(parents=True, exist_ok=False)
        ok = tech_A(out) if a.exp == "A" else tech_B(out)
        sys.exit(0 if ok else 2)
    if a.mode == "pilot":
        pilot_run(a.exp, out)
    elif a.mode == "freeze":
        freeze(a.exp, out, Path(a.tech).resolve(), Path(a.pilot).resolve(),
               Path(a.doc).resolve() if a.doc else None)
    elif a.mode == "execute":
        st = execute(a.exp, out, Path(a.protocol).resolve(), a.resume, a.deadline)
        print("stopped:", st)
    elif a.mode == "analyze":
        (analyze_A if a.exp == "A" else analyze_B)(Path(a.run).resolve(), out)


if __name__ == "__main__":
    main()
