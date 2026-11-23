"""CORA-SH1 runner: Cora strict held-out X-family comparison.

Modes
-----
prepare  : build the 5 main splits (no fit) and write split_summary.csv,
           manifest.csv and runinfo.json.
smoke    : TECHNICAL pre-check on the reserved smoke split (index 90, never a
           main split): runtime, memory, masks, convergence diagnostics, saving,
           fast-vs-original equivalence and a real-data two-canary check.
           Held-out scores are written ONLY to a sealed file outside the
           technical output (``--sealed-dir``); they are not summarised.
execute  : the main 90-fit experiment.  Refused unless an approved, frozen
           protocol file and an explicit confirmation flag are given.

Every fit runs in a fresh worker process (max_tasks_per_child=1) so that the
peak working set per fit can be measured.  No retry, replacement or seed
change exists anywhere (fail_fast); failures are rows, not reruns.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import cora_strict_heldout as csh   # noqa: E402

SMOKE_GROUPS = {
    # group -> (workers, blas_threads, cells)
    # cell = (tag, selection, arm, K, model_kind, compute_mode, num_iter, canary)
    "equivalence": (1, None, [
        ("EQ-td700-BX-K6-original", "train_degree", "BX", 6, "original", False, None, None),
        ("EQ-td700-BX-K6-fast", "train_degree", "BX", 6, "fast", False, None, None),
    ]),
    "full_K6_parallel": (3, 2, [
        ("FULL-BX-K6", "full", "BX", 6, "fast", False, None, None),
        ("FULL-GX-K6", "full", "GX", 6, "fast", False, None, None),
        ("FULL-YO-K6", "full", "YO", 6, "fast", False, None, None),
    ]),
    "full_K3_parallel": (3, 2, [
        ("FULL-BX-K3", "full", "BX", 3, "fast", False, None, None),
        ("FULL-GX-K3", "full", "GX", 3, "fast", False, None, None),
        ("FULL-YO-K3", "full", "YO", 3, "fast", False, None, None),
    ]),
    "canary": (2, 2, [
        ("CANARY0-full-BX-K3", "full", "BX", 3, "fast", False, 1, 0),
        ("CANARY1-full-BX-K3", "full", "BX", 3, "fast", False, 1, 1),
    ]),
}


def _peak_mb() -> float:
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) or getattr(info, "rss", 0)
        return round(peak / 2 ** 20, 1)
    except Exception:
        return float("nan")


def _worker(job: dict) -> dict:
    """Runs in a fresh process.  Rebuilds the split deterministically."""
    data = csh.load_cora()
    split = csh.build_split(data, job["split_index"], selection=job.get("selection"))
    Y_override = None
    if job.get("canary") == 1:
        Y_override = np.where(split.test_mask, 1.0, split.Y_fit)
    t0 = time.perf_counter()
    tech, scores, states = csh.run_cell(
        split, job["arm"], job["K"], job["init_seed"], job["mc_seed"],
        model_kind=job["model_kind"], num_iter=job.get("num_iter"),
        Y_fit_override=Y_override, compute_mode=job["compute_mode"],
        mode_second_start=job["mode_second_start"])
    tech["wall_seconds_in_worker"] = round(time.perf_counter() - t0, 3)
    tech["peak_working_set_mb"] = _peak_mb()
    tech["tag"] = job["tag"]
    tech["blas_threads"] = job.get("blas_threads") or "default"
    if job.get("state_path"):
        Path(job["state_path"]).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(job["state_path"], **states)
        tech["state_sha256"] = hashlib.sha256(
            Path(job["state_path"]).read_bytes()).hexdigest()[:16]
    for r in scores:
        r["tag"] = job["tag"]
    return {"tech": tech, "scores": scores}


_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    out = []
    threads = {j.get("blas_threads") for j in jobs}
    saved = {v: os.environ.get(v) for v in _BLAS_VARS}
    if threads != {None}:                 # spawned children read this at import
        for v in _BLAS_VARS:
            os.environ[v] = str(next(iter(threads)))
    try:
        out = _run_pool(jobs, workers)
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
    return out


def _run_pool(jobs: list[dict], workers: int) -> list[dict]:
    out = []
    with cf.ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
        futures = {ex.submit(_worker, j): j for j in jobs}
        for fut in cf.as_completed(futures):
            j = futures[fut]
            try:
                out.append(fut.result())
            except Exception as exc:          # worker crash: recorded, not rerun
                out.append({"tech": {"tag": j["tag"],
                                     "fit_status": f"WORKER_CRASH:{type(exc).__name__}",
                                     "fit_error": str(exc)[:300]}, "scores": []})
            print(f"[done] {j['tag']}", flush=True)
    return out


def _new_dir(path: Path) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"output directory exists and is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def cmd_prepare(args):
    out = _new_dir(args.out)
    data = csh.load_cora()
    rows = [csh.build_split(data, s, selection=sel).summary
            for sel in args.selection for s in range(csh.CONFIG["n_splits"])]
    pd.DataFrame(rows).to_csv(out / "split_summary.csv", index=False)
    pd.DataFrame(csh.build_manifest()).to_csv(out / "manifest.csv", index=False)
    (out / "runinfo.json").write_text(json.dumps(
        {"mode": "prepare", **csh.environment_info()}, indent=2, default=str))
    print(pd.DataFrame(rows)[["selection", "split_index", "n", "n_train_pairs",
                              "n_test_pairs", "n_train_pos", "n_test_pos",
                              "train_pair_density", "test_pair_density",
                              "n_isolated_in_train_subgraph"]].to_string(index=False))


def cmd_smoke(args):
    out = _new_dir(args.out)
    sealed = _new_dir(args.sealed_dir)
    s = csh.CONFIG["smoke_split_index"]
    t0 = time.perf_counter()
    results = []
    group_seconds = {}
    for group in args.groups:
        workers, threads, cells = SMOKE_GROUPS[group]
        jobs = []
        for tag, selection, arm, K, kind, mode, num_iter, canary in cells:
            seed_index = 1 if canary is not None else 0
            init_seed, mc_seed = csh.seeds_for(s, K, seed_index)
            jobs.append(dict(tag=tag, group=group, split_index=s, selection=selection,
                             arm=arm, K=K, init_seed=init_seed, mc_seed=mc_seed,
                             model_kind=kind, compute_mode=mode, mode_second_start=False,
                             num_iter=num_iter, canary=canary, blas_threads=threads,
                             state_path=str(sealed / "states" / f"{tag}.npz")))
        tg = time.perf_counter()
        res = _run_jobs(jobs, workers=workers)
        group_seconds[group] = round(time.perf_counter() - tg, 1)
        for r in res:
            r["tech"]["group"] = group
            r["tech"]["selection"] = next(j["selection"] for j in jobs
                                          if j["tag"] == r["tech"]["tag"])
        results.extend(res)
    tech = pd.DataFrame([r["tech"] for r in results]).sort_values("tag")
    tech.to_csv(out / "smoke_fit_technical.csv", index=False)
    scores = [row for r in results for row in r["scores"]]
    pd.DataFrame(scores).to_csv(sealed / "smoke_scores_SEALED.csv", index=False)

    # equivalence and canary checks (technical only; no score is printed)
    def load(name):
        return np.load(sealed / "states" / f"{name}.npz")
    checks: dict = {"group_wall_seconds": group_seconds}
    if "equivalence" in args.groups:
        a, b = load("EQ-td700-BX-K6-original"), load("EQ-td700-BX-K6-fast")
        checks["fast_vs_original_Z_samples_max_abs_diff"] = float(
            np.max(np.abs(a["Z_samples"] - b["Z_samples"])))
        checks["fast_vs_original_F_max_abs_diff"] = float(np.max(np.abs(a["F"] - b["F"])))
        checks["fast_vs_original_w0_abs_diff"] = float(abs(a["w0"] - b["w0"]))
        checks["fast_vs_original_w_abs_diff"] = float(abs(a["w"] - b["w"]))
    sc = pd.DataFrame(scores)
    if "canary" in args.groups:
        c0, c1 = load("CANARY0-full-BX-K3"), load("CANARY1-full-BX-K3")
        checks["canary_bitwise_identical_states"] = bool(
            c0.files == c1.files and all(np.array_equal(c0[k], c1[k]) for k in c0.files))
        can = sc[sc["tag"].str.startswith("CANARY")]
        checks["canary_scores_identical"] = bool(
            can.groupby("estimator")["log_score"].nunique().eq(1).all())
        ct = tech[tech["tag"].str.startswith("CANARY")]
        checks["canary_y_fit_hashes_differ"] = bool(ct["y_fit_hash"].nunique() == 2)
    if len(sc):
        checks["sealed_scores_rows"] = int(len(sc))
        checks["sealed_scores_all_finite"] = bool(np.isfinite(
            sc[["log_score", "ap", "auc", "brier"]].to_numpy(float)).all())
        checks["sealed_scores_sha256"] = hashlib.sha256(
            (sealed / "smoke_scores_SEALED.csv").read_bytes()).hexdigest()[:16]
    checks["smoke_total_seconds"] = round(time.perf_counter() - t0, 1)
    (out / "smoke_checks.json").write_text(json.dumps(checks, indent=2))
    data = csh.load_cora()
    (out / "smoke_split_summary.json").write_text(json.dumps(
        {sel: csh.build_split(data, s, selection=sel).summary
         for sel in ("full", "train_degree")}, indent=2))
    (out / "runinfo.json").write_text(json.dumps(
        {"mode": "smoke", "sealed_dir": "outside repository (not committed)",
         **csh.environment_info()}, indent=2, default=str))
    cols = [c for c in ("tag", "fit_status", "em_seconds", "total_seconds",
                        "mode_seconds", "mode_from_mean5_seconds", "mode_status",
                        "mode_iterations", "peak_working_set_mb", "mstep_q_decreases",
                        "sampling_fallbacks", "n_warnings_em", "F_grad_max")
            if c in tech.columns]
    print(tech[cols].to_string(index=False))
    print(json.dumps(checks, indent=2))


def cmd_execute(args):
    if not (args.approved_protocol and args.i_confirm_human_approval):
        raise SystemExit("REFUSED: the main experiment needs a frozen, human-approved "
                         "protocol file (--approved-protocol) and "
                         "--i-confirm-human-approval.  Not run.")
    raise SystemExit("REFUSED: execute is intentionally not wired before the protocol "
                     "is frozen; bind it to the approved protocol first.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--selection", nargs="+", default=["full"], choices=csh.SELECTIONS)
    b = sub.add_parser("smoke")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--sealed-dir", type=Path, required=True)
    b.add_argument("--groups", nargs="+", default=list(SMOKE_GROUPS),
                   choices=list(SMOKE_GROUPS))
    c = sub.add_parser("execute")
    c.add_argument("--approved-protocol", type=Path)
    c.add_argument("--i-confirm-human-approval", action="store_true")
    args = p.parse_args()
    {"prepare": cmd_prepare, "smoke": cmd_smoke, "execute": cmd_execute}[args.mode](args)


if __name__ == "__main__":
    main()
