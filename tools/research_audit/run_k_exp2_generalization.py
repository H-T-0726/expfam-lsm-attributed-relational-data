"""K-selection Exp2-generalisation runner (lineage E scalar, canonical-clean data).

Modes
  --preflight OUT   calibration + generation of every dataset (no fit); checks
  --smoke OUT       technical smoke on DISJOINT seeds (never the main-run data);
                    timing / memory / code path only
  --freeze OUT      write protocol.json (design, calibration, manifest hash,
                    code hashes, analysis plan).  Commit it before --execute.
  --execute OUT     run the frozen manifest; refuses on code-hash mismatch or a
                    dirty git tree; --resume continues an interrupted run

Failure policy: fail_fast.  No retry, no seed change, no replacement fit, no
extra iterations.  A failed fit is recorded once and never re-run.  --resume
only runs fit_ids that have NO recorded row; each fit is a deterministic
function of its frozen seeds, so resuming adds no new random trial.

Lineage E (DualExpFamLSMConsistent, numerics_mode='consistent').  Not adopted
for the manuscript (Human Gate).  C_Q is the Q-based complete-data / ICL-type
criterion, NOT Schwarz BIC.  C_Lap is NOT evaluated here (deferred to a
separate pre-registered analysis on the saved states).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in ("expfam/src", "expfam/src/experimental", "reproduction/src"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

import k_exp2_design as kd  # noqa: E402

N_WORKERS = 8
SMOKE_SEED_OFFSET = 5_000_000
CODE_FILES = (
    "expfam/src/experimental/k_exp2_design.py",
    "tools/research_audit/run_k_exp2_generalization.py",
    "tools/research_audit/analyze_k_exp2_generalization.py",
    "expfam/src/experimental/data_generator_canonical.py",
    "expfam/src/experimental/matched_k_true_design.py",
    "expfam/src/experimental/em_runner.py",
    "expfam/src/experimental/model_dual_expfam_consistent.py",
    "expfam/src/experimental/model_dual_expfam_masked.py",
    "expfam/src/experimental/eval_utils.py",
    "expfam/src/model_dual_expfam_fixed.py",
    "expfam/src/model_dual_expfam.py",
    "expfam/src/model_expfam.py",
    "reproduction/src/model.py",
)

FIT_FIELDS = [
    "fit_id", "dataset_id", "series", "regime", "k_true", "rep", "k", "t",
    "family_x", "family_y", "data_seed", "init_seed", "mc_seed",
    "status", "failure_reason", "Q_strict", "C_Q", "num_params", "C_Q_recomputed_absdiff",
    "w0_hat", "w_hat", "sigma_y_hat", "retry_count", "replacement_count",
    "seed_rescue_count", "nan_count", "q_after_last", "q_after_prev",
    "q_rel_change_last", "mstep_q_decreases", "x_hash", "y_hash",
    "y_mean", "x_mean", "true_w0", "true_w", "true_f_scale",
    "runtime_s", "worker_pid", "state_file", "state_sha256",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_hashes() -> dict:
    return {f: sha256_file(ROOT / f) for f in CODE_FILES}


def git_head() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def git_dirty() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                          capture_output=True, text=True).stdout.strip()


def manifest_hash(rows) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def table_to_json(table: dict) -> dict:
    return {f"{r}-K{k}": v for (r, k), v in table.items()}


def table_from_json(obj: dict) -> dict:
    out = {}
    for key, v in obj.items():
        r, k = key.split("-K")
        out[(r, int(k))] = v
    return out


# --------------------------------------------------------------------------
# one fit
# --------------------------------------------------------------------------

def fit_one(job: dict, bern_table: dict, state_dir: str | None) -> dict:
    from em_runner import run_em_experimental  # imported in the worker

    t0 = time.perf_counter()
    row = {f: "" for f in FIT_FIELDS}
    row.update({k: job[k] for k in ("fit_id", "dataset_id", "series", "regime", "k_true",
                                    "rep", "k", "t", "family_x", "family_y",
                                    "data_seed", "init_seed", "mc_seed")})
    row["worker_pid"] = os.getpid()
    spec = kd.DatasetSpec(job["series"], job["regime"], job["k_true"], job["rep"],
                          kd.w_of(job["series"], job["regime"], job["k_true"]),
                          kd.w0_of(job["series"], job["regime"], job["k_true"], bern_table),
                          kd.f_scale_of(job["regime"], job["k_true"]), job["data_seed"])
    ds = kd.generate(spec)              # GeneratorStop / DesignStop propagate -> STOP
    X, Y = ds.X, ds.Y
    row.update(x_hash=kd.array_hash(X), y_hash=kd.array_hash(Y),
               y_mean=float(Y[np.triu_indices(kd.N, 1)].mean()), x_mean=float(X.mean()),
               true_w0=spec.w0, true_w=spec.w, true_f_scale=spec.f_scale)
    try:
        # Only the observed (X, Y) reach the estimator; no truth is passed.
        r = run_em_experimental(
            X, Y, job["family_x"], job["family_y"], job["k"], L=kd.L_MC,
            num_iter=kd.NUM_ITER, numerics_mode="consistent", failure_policy="fail_fast",
            init_seed=job["init_seed"], mc_seed=job["mc_seed"], compute_strict_Q=True,
            mstep_q_diagnostic=True)
        q = float(r["Q_strict"])
        cq = float(r["bic"])
        npar = int(r["num_params"])
        recomputed = -2.0 * q + npar * math.log(kd.N)
        hist = r.get("mstep_q_history") or []
        qa = [h["q_after"] for h in hist]
        finite_theta = all(np.all(np.isfinite(np.asarray(r[key], float)))
                           for key in ("F", "sigma", "w0", "w"))
        ok = (np.isfinite(q) and np.isfinite(cq) and not r["q_bic_failed"] and finite_theta
              and r["retry_count"] == 0 and r["replacement_count"] == 0)
        row.update(
            status="OK" if ok else "FAILED",
            failure_reason="" if ok else (r.get("failure_reason") or "non-finite or integrity"),
            Q_strict=repr(q), C_Q=repr(cq), num_params=npar,
            C_Q_recomputed_absdiff=repr(abs(recomputed - cq)),
            w0_hat=repr(float(r["w0"])), w_hat=repr(float(r["w"])),
            sigma_y_hat=repr(float(r["sigma_y_est"])),
            retry_count=r["retry_count"], replacement_count=r["replacement_count"],
            seed_rescue_count=r["seed_rescue_count"], nan_count=r["nan_count"],
            q_after_last=repr(qa[-1]) if qa else "", q_after_prev=repr(qa[-2]) if len(qa) > 1 else "",
            q_rel_change_last=(repr(abs(qa[-1] - qa[-2]) / abs(qa[-2]))
                               if len(qa) > 1 and qa[-2] != 0 else ""),
            mstep_q_decreases=sum(bool(h["decreased"]) for h in hist))
        row["_hist"] = [{"fit_id": job["fit_id"], **h} for h in hist]
        if state_dir and ok:
            sp = Path(state_dir) / f"{job['fit_id']}.npz"
            np.savez_compressed(sp, F=r["F"], sigma=r["sigma"], w0=r["w0"], w=r["w"],
                                var_z=r["var_z"], sigma_y=r["sigma_y_est"],
                                Z_last=r["Z_est"], Z_mean=r["Z_samples"].mean(axis=2))
            row["state_file"] = sp.name
            row["state_sha256"] = sha256_file(sp)[:16]
    except Exception as exc:  # recorded once; never retried
        row.update(status="FAILED", failure_reason=f"{type(exc).__name__}: {str(exc)[:300]}")
        row["_hist"] = []
    row["runtime_s"] = round(time.perf_counter() - t0, 2)
    return row


def _worker(args):
    job, table_json, state_dir = args
    return fit_one(job, table_from_json(table_json), state_dir)


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------

class Writers:
    def __init__(self, out: Path):
        self.out = out
        self.fit_path = out / "per_fit.csv"
        self.hist_path = out / "mstep_q_history.csv"
        self.fail_path = out / "failures.csv"
        for p, fields in ((self.fit_path, FIT_FIELDS),
                          (self.hist_path, ["fit_id", "iteration", "q_before", "q_after",
                                            "q_diff", "decreased"]),
                          (self.fail_path, ["fit_id", "dataset_id", "failure_reason"])):
            if not p.exists():
                with open(p, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=fields).writeheader()

    def write(self, row: dict):
        hist = row.pop("_hist", [])
        with open(self.fit_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIT_FIELDS).writerow(row)
        if hist:
            with open(self.hist_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["fit_id", "iteration", "q_before",
                                                  "q_after", "q_diff", "decreased"])
                for h in hist:
                    w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in h.items()})
        if row["status"] != "OK":
            with open(self.fail_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=["fit_id", "dataset_id", "failure_reason"]).writerow(
                    {k: row[k] for k in ("fit_id", "dataset_id", "failure_reason")})


def recorded_fit_ids(out: Path) -> set:
    p = out / "per_fit.csv"
    if not p.exists():
        return set()
    with open(p, encoding="utf-8") as f:
        return {r["fit_id"] for r in csv.DictReader(f)}


def run_jobs(jobs, table, out: Path, state_dir: Path | None, log_every=8):
    W = Writers(out)
    payload = [(j, table_to_json(table), str(state_dir) if state_dir else None) for j in jobs]
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    done = 0
    with ctx.Pool(N_WORKERS, maxtasksperchild=50) as pool:
        for row in pool.imap_unordered(_worker, payload, chunksize=1):
            W.write(row)
            done += 1
            if done % log_every == 0 or done == len(jobs):
                el = time.perf_counter() - t0
                print(f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S}] {done}/{len(jobs)} "
                      f"elapsed {el/3600:.2f} h  eta {el/done*(len(jobs)-done)/3600:.2f} h",
                      flush=True)
    return time.perf_counter() - t0


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

def preflight(out: Path):
    out.mkdir(parents=True, exist_ok=False)
    table = kd.bernoulli_w0_table()
    rows = kd.build_manifest()
    ids = [r["fit_id"] for r in rows]
    dseeds = {(r["series"], r["k_true"], r["rep"]): r["data_seed"] for r in rows}
    fit_seeds = [(r["init_seed"], r["mc_seed"]) for r in rows if r["regime"] == "a" and r["k"] == 1]
    checks = {
        "n_fits": len(rows), "expected": 4 * 2 * 5 * 10 * 10 * 2,
        "fit_ids_unique": len(set(ids)) == len(ids),
        "data_seeds_unique_per_dataset": len(set(dseeds.values())) == len(dseeds),
        "init_mc_seeds_unique": len({s for p in fit_seeds for s in p}) == 2 * len(fit_seeds),
        "data_seeds_disjoint_from_init_mc": not (set(dseeds.values()) & {s for p in fit_seeds for s in p}),
    }
    ds_rows = []
    for s in kd.SERIES:
        for regime in kd.REGIMES:
            for kt in kd.K_TRUES:
                for rep in range(kd.N_REPLICATES):
                    spec = kd.dataset_spec(s, regime, kt, rep, table)
                    d = kd.generate(spec)
                    iu = np.triu_indices(kd.N, 1)
                    ds_rows.append(dict(
                        dataset_id=kd.dataset_id(s, regime, kt, rep), series=s, regime=regime,
                        k_true=kt, rep=rep, data_seed=spec.seed, w=spec.w, w0=spec.w0,
                        f_scale=spec.f_scale, rank_F=int(np.linalg.matrix_rank(d.F)),
                        mean_row_norm_sq=float(np.mean(np.sum(d.F ** 2, 1))),
                        rho_target=kd.rho_x_of(regime, kt),
                        y_mean=float(d.Y[iu].mean()), x_mean=float(d.X.mean()),
                        z_hash=kd.array_hash(d.Z), q_hash=kd.array_hash(np.round(d.F / spec.f_scale, 12)),
                        x_hash=kd.array_hash(d.X), y_hash=kd.array_hash(d.Y)))
    import pandas as pd
    df = pd.DataFrame(ds_rows)
    checks["rank_F_equals_K_TRUE"] = bool((df.rank_F == df.k_true).all())
    checks["mean_row_norm_sq_matches_target_max_absdiff"] = float(
        (df.mean_row_norm_sq - df.rho_target).abs().max())
    pa = df.pivot_table(index=["series", "k_true", "rep"], columns="regime",
                        values=["z_hash", "q_hash"], aggfunc="first")
    checks["Z_shared_across_regimes"] = bool((pa[("z_hash", "a")] == pa[("z_hash", "b")]).all())
    checks["Q_shared_across_regimes"] = bool((pa[("q_hash", "a")] == pa[("q_hash", "b")]).all())
    df.to_csv(out / "datasets.csv", index=False)
    (out / "calibration.json").write_text(json.dumps(table_to_json(table), indent=2))
    (out / "preflight_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    return checks


def smoke(out: Path):
    out.mkdir(parents=True, exist_ok=False)
    table = kd.bernoulli_w0_table()
    jobs = []
    for s in kd.SERIES:
        for kt in (1, 9):
            for k in (1, 10):
                regimes = ("a", "b") if (kt == 9 and k == 10) else ("a",)
                for regime in regimes:
                    ds = kd.data_seed(s, kt, 0) + SMOKE_SEED_OFFSET
                    jobs.append(dict(fit_id="SMOKE-" + kd.fit_id(s, regime, kt, 0, k, 0),
                                     dataset_id="SMOKE-" + kd.dataset_id(s, regime, kt, 0),
                                     series=s, regime=regime, k_true=kt, rep=0, k=k, t=0,
                                     family_x=kd.SERIES[s][0], family_y=kd.SERIES[s][1],
                                     data_seed=ds, init_seed=kd.init_seed(ds, 0),
                                     mc_seed=kd.mc_seed(ds, 0)))
    (out / "smoke_states").mkdir()
    wall = run_jobs(jobs, table, out, out / "smoke_states", log_every=4)
    import pandas as pd
    df = pd.read_csv(out / "per_fit.csv")
    rt = df.groupby("k")["runtime_s"].mean().to_dict()
    # budget: mean runtime per candidate K (linear interpolation between K=1 and K=10)
    r1, r10 = rt.get(1, float("nan")), rt.get(10, float("nan"))
    per_k = {k: r1 + (r10 - r1) * (k - 1) / 9 for k in kd.K_CANDIDATES}
    cpu_h = sum(per_k.values()) * (8000 / 10) / 3600
    summary = {
        "purpose": "technical smoke on seeds disjoint from the main run; values not inspected",
        "n_fits": len(df), "status_counts": df.status.value_counts().to_dict(),
        "mean_runtime_s_by_k": rt, "max_runtime_s": float(df.runtime_s.max()),
        "wall_s": round(wall, 1), "workers": N_WORKERS,
        "budget_estimate_cpu_h": round(cpu_h, 1),
        "budget_estimate_wall_h_8_workers": round(cpu_h / N_WORKERS, 1),
        "C_Q_recomputed_absdiff_max": float(pd.to_numeric(df.C_Q_recomputed_absdiff, errors="coerce").max()),
        "retry_replacement_rescue_all_zero": bool(
            (df[["retry_count", "replacement_count", "seed_rescue_count"]].fillna(0) == 0).all().all()),
    }
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


ANALYSIS_PLAN = {
    "unit": "independent generated dataset (series x regime x K_TRUE x rep); inits and regimes never counted as independent replicates",
    "primary_selection": "per dataset: C_Q_bar(K) = unweighted mean of C_Q over the 2 inits; selected K = argmin_K C_Q_bar; ties within 1e-9 -> smallest K",
    "incomplete": "a dataset is INCOMPLETE for a rule if any fit it needs is FAILED or missing; no imputation; denominators always 10 with incomplete counts shown",
    "secondary": ["per-init selected K (t=0, t=1) and their agreement", "C_Q margin best vs second-best on C_Q_bar", "exact / under / over counts", "confusion matrices", "mean C_Q_bar - min curves", "regime a vs b paired comparison on the same (series, K_TRUE, rep)", "MCEM non-convergence diagnostic: relative change of the M-step Q between iterations 9 and 10 (descriptive, includes MC noise)", "runtime", "failures"],
    "ci": "exact Clopper-Pearson 95% for exact-count / 10 per cell",
    "deferred": "C_Lap on saved states (no MCEM) is a separate pre-registered analysis; not part of this execution",
    "target": "agreement with the generating K (K_TRUE); not K*",
}


def freeze(out: Path):
    out.mkdir(parents=True, exist_ok=False)
    table = kd.bernoulli_w0_table()
    rows = kd.build_manifest()
    proto = {
        "experiment_id": "k_exp2_generalization_20261125",
        "design_version": kd.DESIGN_VERSION, "generator_version": kd.GENERATOR_VERSION,
        "lineage": "E scalar (DualExpFamLSMConsistent, numerics_mode=consistent); not adopted for manuscript",
        "criterion_primary": "C_Q = -2 Q_strict + p ln n (Q-based complete-data / ICL-type; NOT Schwarz BIC)",
        "series": kd.SERIES, "regimes": {"a": "per-direction signal fixed (PRIMARY)", "b": "total signal fixed (SENSITIVITY)"},
        "k_trues": kd.K_TRUES, "k_candidates": kd.K_CANDIDATES, "n": kd.N, "d": kd.D,
        "replicates": kd.N_REPLICATES, "inits": kd.N_INITS, "L": kd.L_MC, "num_iter": kd.NUM_ITER,
        "w_rule": {"O/A/C": "a: 1.0, b: sqrt(3/K)", "B": "a: 0.12, b: 0.12 sqrt(3/K)"},
        "f_rule": {"a": "F = sqrt(d/6) Q", "b": "F = sqrt(0.5 d / K) Q", "Q": "canonical-clean-v1 build_full_rank_loadings (QR of Gaussian, sign of diag(R) fixed)"},
        "w0_rule": {"B": "(K/2) log(1-w^2) -> E[lambda]=1", "C": 0.0, "O/A": "population mean edge prob 0.20 (calibration table)"},
        "noise": {"gaussian_x_variance": kd.SIGMA_X_VAR, "gaussian_y_variance": kd.SIGMA_Y_SD ** 2},
        "calibration": table_to_json(table),
        "calibration_tolerances": {"w0_agree": kd.CAL_TOL_W0_AGREE, "p_resid": kd.CAL_TOL_P_RESID,
                                   "p_indep": kd.CAL_TOL_P_INDEP, "root_xtol": kd.CAL_ROOT_XTOL,
                                   "trapezoid_h": kd.CAL_TRAPEZOID_H, "hermite_nodes": kd.HERMITE_NODES},
        "seeds": {"data_seed": "20261125000 + 100000*series_idx + 1000*K_TRUE + rep (regime/K/init independent)",
                  "init_seed": "10*data_seed + t", "mc_seed": "10*data_seed + 5 + t"},
        "failure_policy": "fail_fast; no retry / seed change / replacement / extra iterations; failed fit recorded once",
        "stop_rules": ["GeneratorStop or DesignStop", "code-hash mismatch", "dirty git tree at execute", "output dir exists without --resume"],
        "resume_rule": "run only fit_ids without a recorded row; same frozen seeds; failures never re-run; every resume logged",
        "workers": N_WORKERS, "blas_threads": 1,
        "analysis_plan": ANALYSIS_PLAN,
        "n_fits": len(rows), "manifest_sha256": manifest_hash(rows),
        "code_sha256": code_hashes(),
        "frozen_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    p = out / "protocol.json"
    p.write_text(json.dumps(proto, indent=2, sort_keys=True))
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("protocol sha256", sha256_file(p))


def execute(proto_path: Path, out: Path, resume: bool):
    proto = json.loads(proto_path.read_text())
    if proto["code_sha256"] != code_hashes():
        bad = [f for f in CODE_FILES if proto["code_sha256"].get(f) != sha256_file(ROOT / f)]
        raise SystemExit(f"STOP: code hash mismatch: {bad}")
    dirty = git_dirty()
    if dirty:
        raise SystemExit(f"STOP: git tree not clean:\n{dirty}")
    rows = kd.build_manifest()
    if manifest_hash(rows) != proto["manifest_sha256"]:
        raise SystemExit("STOP: manifest hash mismatch")
    table = table_from_json(proto["calibration"])
    fresh = kd.bernoulli_w0_table()
    if any(abs(fresh[key]["w0"] - table[key]["w0"]) > 0 for key in table):
        raise SystemExit("STOP: calibration differs from the frozen table")
    if out.exists() and not resume:
        raise SystemExit("STOP: output exists; use --resume")
    out.mkdir(parents=True, exist_ok=True)
    (out / "states").mkdir(exist_ok=True)
    done = recorded_fit_ids(out)
    jobs = [r for r in rows if r["fit_id"] not in done]
    event = {"time": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
             "event": "resume" if done else "start", "already_recorded": len(done),
             "to_run": len(jobs), "git_head": git_head(),
             "protocol_sha256": sha256_file(proto_path)}
    with open(out / "run_events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    if not (out / "runinfo.json").exists():
        (out / "runinfo.json").write_text(json.dumps({
            "protocol": proto_path.as_posix(), "protocol_sha256": event["protocol_sha256"],
            "git_head_at_start": event["git_head"], "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "workers": N_WORKERS, "blas_threads": 1,
            "n_fits_planned": len(rows)}, indent=2))
    wall = run_jobs(jobs, table, out, out / "states", log_every=40)
    with open(out / "run_events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                            "event": "finished_batch", "ran": len(jobs), "wall_s": round(wall, 1),
                            "recorded_total": len(recorded_fit_ids(out))}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", type=Path)
    g.add_argument("--smoke", type=Path)
    g.add_argument("--freeze", type=Path)
    g.add_argument("--execute", type=Path)
    ap.add_argument("--protocol", type=Path)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.preflight:
        preflight(a.preflight)
    elif a.smoke:
        smoke(a.smoke)
    elif a.freeze:
        freeze(a.freeze)
    else:
        if not a.protocol:
            raise SystemExit("--protocol required")
        execute(a.protocol, a.execute, a.resume)


if __name__ == "__main__":
    main()
