"""Fault-tolerant execution of mcem_noise_separation_20261106 (operational addendum v2).

Same science as the pre-registration (run_mcem_noise_separation.PREREG, unchanged): the per-fit computations
are run_mcem_noise_separation.phase3_fit / phase4_fit.  What changes is only HOW they are executed and saved:

* each fit is saved as soon as it finishes: <results>/<stage>/fits/<fit_key>/{*.csv, manifest.json} and
  <artifacts>/<stage>/<fit_key>.npz; every file is written to a temporary name and atomically renamed;
  manifest.json is written last and lists the sha256 of every file, so a fit without a valid manifest is never
  treated as finished;
* on (re)start every expected fit is validated (manifest parses, config hash matches, all checksums match):
  OK and FAILED fits are skipped (a FAILED fit is never retried or reseeded), anything else is INCOMPLETE and
  its leftovers are renamed aside (never deleted) before the fit is run again from iteration 1;
* BLAS / OpenMP threads are pinned to 1 (measured: identical numbers, ~8x less private memory per process);
* a memory guard stops launching new fits when available physical memory is low; a monitor thread logs memory.

There is no mid-fit resume: the MCEM RNG state is not checkpointed, so an interrupted fit is rerun from the start.

Usage
  python tools/research_audit/run_mcem_noise_separation_v2.py run --stage phase3 --conds A
  python tools/research_audit/run_mcem_noise_separation_v2.py run --stage phase4 --L 20 --conds A D
  python tools/research_audit/run_mcem_noise_separation_v2.py status --stage phase3
  python tools/research_audit/run_mcem_noise_separation_v2.py aggregate --stage phase3
"""
from __future__ import annotations

import os

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for _v in THREAD_VARS:
    os.environ[_v] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import run_mcem_noise_separation as ns  # noqa: E402

EXEC = "exec_v2"
RESULTS_ROOT = ns.OUT_ROOT / EXEC
ARTIFACT_ROOT = ns.STATE_ROOT / EXEC
ADDENDUM_VERSION = "operational-addendum-v2"
COMPUTE_FILES = (
    "tools/research_audit/run_mcem_noise_separation.py",
    "tools/research_audit/run_mcem_stability.py",
    "tools/research_audit/run_k_signal_strength.py",
    "tools/research_audit/run_clap_clean_nsweep.py",
    "expfam/src/experimental/em_runner_instrumented.py",
    "expfam/src/experimental/em_runner.py",
    "expfam/src/experimental/k_criterion_panel.py",
    "expfam/src/experimental/laplace_k_criterion.py",
    "expfam/src/experimental/model_dual_expfam_consistent.py",
    "expfam/src/experimental/model_dual_expfam_masked.py",
    "expfam/src/model_dual_expfam_fixed.py",
    "expfam/src/model_dual_expfam.py",
    "expfam/src/model_expfam.py",
    "reproduction/src/model.py",
)
CSV_NAMES = {"phase3": ("state_rows", "probe_rows", "replicate_rows"),
             "phase4": ("checkpoint_rows", "probe_rows", "replicate_rows", "iteration_rows")}


# --------------------------------------------------------------------------- hashing / atomic io
def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return ns.sha_file(p)


def sha_array(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    return sha_bytes(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def code_hash() -> str:
    h = hashlib.sha256()
    for rel in COMPUTE_FILES:
        h.update(rel.encode())
        h.update((ns.rm.ROOT / rel).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def csv_bytes(rows: list[dict]) -> bytes:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def npz_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


# --------------------------------------------------------------------------- fit specs
def stage_name(stage: str, L: int | None) -> str:
    return "phase3" if stage == "phase3" else f"phase4_L{L}"


def expected_fits(stage: str, L: int | None, conds) -> list[dict]:
    out = []
    if stage == "phase3":
        for c in conds:
            for rep in ns.REPS:
                for k in ns.KS:
                    for s in ns.STARTS:
                        out.append({"stage": "phase3", "fit_key": f"{c}_r{rep}_k{k}_s{s}", "condition": c, "replicate": rep,
                                    "k_fit": k, "start": s, "L_fit": 5, "fit_id": ns.fit_id_of(c, rep, k, s),
                                    "model_seed": ns.hist.model_seed(5, ns.N, rep, k, s), "max_iter": 64,
                                    "probe_seed": f"[{ns.TAG_PROBE}, fit_id, checkpoint, r]"})
    else:
        for c in conds:
            for rep in ns.P4_REPS:
                for k in ns.KS:
                    out.append({"stage": f"phase4_L{L}", "fit_key": f"L{L}_{c}_r{rep}_k{k}_s{ns.P4_START}", "condition": c,
                                "replicate": rep, "k_fit": k, "start": ns.P4_START, "L_fit": L,
                                "fit_id": ns.phase4_fit_id(L, c, rep, k), "model_seed": ns.hist.model_seed(5, ns.N, rep, k, ns.P4_START),
                                "max_iter": ns.P4_MAX_ITER,
                                "probe_seed": f"[{ns.TAG_PROBE + 1000 * L}, fit_id, 32, r]; eval [{ns.TAG_P4_EVAL}, fit_id, 32, r]"})
    return out


class Ctx:
    def __init__(self, results_root: Path, artifact_root: Path):
        self.results_root = results_root
        self.artifact_root = artifact_root
        self._data = None
        self._src_sha: dict[str, str] = {}
        self.code_sha = code_hash()

    @property
    def data(self):
        if self._data is None:
            self._data = ns.rq1_data()
        return self._data

    def src_sha(self, cond: str) -> str:
        name = "rq1_AB_states.npz" if cond in "AB" else "rq1_CD_states.npz"
        if name not in self._src_sha:
            self._src_sha[name] = sha_file(ns.SRC_STATE_ROOT / name)
        return self._src_sha[name]

    def fit_dir(self, spec) -> Path:
        return self.results_root / spec["stage"] / "fits" / spec["fit_key"]

    def npz_path(self, spec) -> Path:
        return self.artifact_root / spec["stage"] / f"{spec['fit_key']}.npz"

    def input_hashes(self, spec) -> dict:
        b = self.data[spec["replicate"]]
        v = b["cond"][spec["condition"]]
        h = {"X": sha_array(v["X"]), "Y": sha_array(v["Y"]), "train": sha_array(b["train"])}
        if spec["stage"] == "phase3":
            h["pr156_states_npz"] = self.src_sha(spec["condition"])
        return h

    def config_hash(self, spec) -> tuple[str, dict]:
        cfg = {"prereg_sha256": ns.prereg_hash(), "code_sha256": self.code_sha, "spec": spec,
               "inputs": self.input_hashes(spec), "thread_vars": {v: os.environ.get(v) for v in THREAD_VARS}}
        return sha_bytes(json.dumps(cfg, sort_keys=True).encode()), cfg


# --------------------------------------------------------------------------- validation
def validate(ctx: Ctx, spec) -> tuple[str, str]:
    """Return (state, reason): OK | FAILED (valid records) | INCOMPLETE | NOT_RUN."""
    d = ctx.fit_dir(spec)
    npz = ctx.npz_path(spec)
    if not d.exists() and not npz.exists():
        return "NOT_RUN", ""
    man = d / "manifest.json"
    if not man.exists():
        return "INCOMPLETE", "no manifest"
    try:
        m = json.loads(man.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return "INCOMPLETE", f"manifest unreadable: {e!r}"
    ch, _ = ctx.config_hash(spec)
    if m.get("config_hash") != ch:
        return "INCOMPLETE", "config hash mismatch (stale)"
    if m.get("status") not in ("OK", "FAILED"):
        return "INCOMPLETE", f"status {m.get('status')}"
    for name, h in m.get("files", {}).items():
        p = npz if name == "__npz__" else d / name
        if not p.exists():
            return "INCOMPLETE", f"missing {name}"
        if sha_file(p) != h:
            return "INCOMPLETE", f"checksum mismatch {name}"
    if m["status"] == "OK":
        need = set(f"{x}.csv" for x in CSV_NAMES["phase3" if spec["stage"] == "phase3" else "phase4"]) | {"__npz__"}
        if not need <= set(m.get("files", {})):
            return "INCOMPLETE", "file set incomplete"
    return m["status"], ""


def set_aside(ctx: Ctx, spec, reason: str, log) -> None:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    d, npz = ctx.fit_dir(spec), ctx.npz_path(spec)
    if d.exists():
        os.replace(d, d.with_name(f"{d.name}.incomplete-{stamp}"))
    if npz.exists():
        os.replace(npz, npz.with_name(f"{npz.name}.incomplete-{stamp}"))
    log(f"SET_ASIDE {spec['fit_key']} ({reason})")


# --------------------------------------------------------------------------- one fit
def run_one(ctx: Ctx, spec) -> str:
    ch, cfg = ctx.config_hash(spec)
    d = ctx.fit_dir(spec)
    d.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()
    peak = [proc.memory_info().rss]
    stop = threading.Event()

    def watch():
        while not stop.wait(1.0):
            peak[0] = max(peak[0], proc.memory_info().rss)
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    t0 = time.time()
    files: dict[str, str] = {}
    status, exc = "OK", ""
    try:
        if spec["stage"] == "phase3":
            S, P, R, arrays = ns.phase3_fit(ctx.data, ns.src_npz(spec["condition"]), spec["condition"], spec["replicate"],
                                            spec["k_fit"], spec["start"])
            tables = {"state_rows": S, "probe_rows": P, "replicate_rows": R}
        else:
            _, C, P, R, IT, arrays = ns.phase4_fit(ctx.data, spec["condition"], spec["replicate"], spec["k_fit"], spec["L_fit"],
                                                   max_iter=spec["max_iter"])
            tables = {"checkpoint_rows": C, "probe_rows": P, "replicate_rows": R, "iteration_rows": IT}
        b = npz_bytes(arrays)
        atomic_write_bytes(ctx.npz_path(spec), b)
        files["__npz__"] = sha_bytes(b)
        for name, rows in tables.items():
            ns.need(len(rows) > 0, f"empty {name}")
            cb = csv_bytes(rows)
            atomic_write_bytes(d / f"{name}.csv", cb)
            files[f"{name}.csv"] = sha_bytes(cb)
    except Exception as e:  # noqa: BLE001  -- recorded, never retried / reseeded
        status, exc = "FAILED", repr(e)
        files = {}
    stop.set()
    th.join()
    vm = psutil.virtual_memory()
    manifest = {"addendum": ADDENDUM_VERSION, "experiment_id": ns.EXPERIMENT_ID, **spec, "status": status, "exception": exc,
                "config_hash": ch, "config": cfg, "files": files, "npz_path": str(ctx.npz_path(spec)),
                "seconds": round(time.time() - t0, 1), "peak_rss_mb": round(peak[0] / 2**20, 1),
                "available_mb_after": round(vm.available / 2**20), "pid": os.getpid(), "git_head": ns.git_head(),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    atomic_write_bytes(d / "manifest.json", json.dumps(manifest, indent=1).encode("utf-8"))
    return status


# --------------------------------------------------------------------------- stage loop
def run_stage(ctx: Ctx, stage: str, L, conds, *, mem_min_gb: float, mem_wait_min: float, deadline_h: float,
              max_fits: int | None) -> int:
    sname = stage_name(stage, L)
    sdir = ctx.results_root / sname
    sdir.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    logf = sdir / f"run_log_{run_id}.txt"
    memf = sdir / f"memory_log_{run_id}.csv"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(logf, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    stop_mon = threading.Event()
    proc = psutil.Process()

    def monitor():
        with open(memf, "a", encoding="utf-8") as f:
            f.write("time,pid,rss_mb,private_mb,available_mb,percent_used\n")
            while True:
                vm = psutil.virtual_memory()
                mi = proc.memory_info()
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{os.getpid()},{mi.rss >> 20},{getattr(mi, 'private', 0) >> 20},"
                        f"{vm.available >> 20},{vm.percent}\n")
                f.flush()
                if stop_mon.wait(30):
                    break
    mt = threading.Thread(target=monitor, daemon=True)
    mt.start()
    log(f"start {sname} conds={conds} prereg={ns.prereg_hash()[:12]} code={ctx.code_sha[:12]} threads=1 mem_min_gb={mem_min_gb}")
    t_start = time.time()
    specs = expected_fits(stage, L, conds)
    ran = 0
    stop_reason = ""
    for spec in specs:
        state, why = validate(ctx, spec)
        if state in ("OK", "FAILED"):
            log(f"SKIP {spec['fit_key']} (validated {state})")
            continue
        if state == "INCOMPLETE":
            set_aside(ctx, spec, why, log)
        if max_fits is not None and ran >= max_fits:
            stop_reason = "MAX_FITS"
            break
        if (time.time() - t_start) / 3600 > deadline_h:
            stop_reason = "DEADLINE"
            break
        waited = 0.0
        while psutil.virtual_memory().available / 2**30 < mem_min_gb:
            if waited >= mem_wait_min * 60:
                stop_reason = "LOW_MEMORY"
                break
            log(f"low memory ({psutil.virtual_memory().available / 2**30:.2f} GB); waiting")
            time.sleep(30)
            waited += 30
        if stop_reason:
            break
        st = run_one(ctx, spec)
        ran += 1
        state, why = validate(ctx, spec)
        log(f"{st} {spec['fit_key']} -> validated {state} {why} "
            f"(avail {psutil.virtual_memory().available / 2**30:.2f} GB, {time.time() - t_start:.0f}s)")
    stop_mon.set()
    mt.join()
    summary = stage_status(ctx, stage, L, conds)
    summary.update({"run_id": run_id, "stop_reason": stop_reason or "ALL_ATTEMPTED", "fits_run_this_session": ran})
    atomic_write_bytes(sdir / f"status_{run_id}.json", json.dumps(summary, indent=1).encode("utf-8"))
    log(f"end {sname}: {summary['counts']} stop_reason={summary['stop_reason']}")
    return 0 if not stop_reason or stop_reason == "MAX_FITS" else 3


def stage_status(ctx: Ctx, stage: str, L, conds) -> dict:
    counts = {"OK": 0, "FAILED": 0, "INCOMPLETE": 0, "NOT_RUN": 0}
    detail = []
    for spec in expected_fits(stage, L, conds):
        st, why = validate(ctx, spec)
        counts[st] += 1
        detail.append({"fit_key": spec["fit_key"], "fit_id": spec["fit_id"], "state": st, "reason": why})
    return {"stage": stage_name(stage, L), "conds": list(conds), "counts": counts,
            "complete": counts["INCOMPLETE"] == 0 and counts["NOT_RUN"] == 0, "fits": detail}


# --------------------------------------------------------------------------- aggregation
def aggregate(ctx: Ctx, stage: str, L, conds, allow_partial: bool) -> int:
    sname = stage_name(stage, L)
    out = ctx.results_root / sname / "aggregate"
    ns.need(not out.exists(), f"refusing to overwrite {out}")
    st = stage_status(ctx, stage, L, conds)
    ns.need(st["complete"] or allow_partial, f"stage incomplete: {st['counts']}")
    tables: dict[str, list] = {n: [] for n in CSV_NAMES["phase3" if stage == "phase3" else "phase4"]}
    import pandas as pd
    ledger, index = [], []
    for spec in expected_fits(stage, L, conds):
        state, why = validate(ctx, spec)
        d = ctx.fit_dir(spec)
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8")) if state in ("OK", "FAILED") else {}
        ledger.append({**{k: spec[k] for k in ("fit_key", "fit_id", "condition", "replicate", "k_fit", "start", "L_fit",
                                                "model_seed", "max_iter")},
                       "status": state if state in ("OK", "FAILED") else state, "reason": why or m.get("exception", ""),
                       "fit_iterations": spec["max_iter"] if (stage == "phase4" and state == "OK") else 0,
                       "seconds": m.get("seconds"), "peak_rss_mb": m.get("peak_rss_mb"), "config_hash": m.get("config_hash")})
        if state == "OK":
            for n in tables:
                tables[n].append(pd.read_csv(d / f"{n}.csv"))
            index.append({"fit_key": spec["fit_key"], "fit_id": spec["fit_id"], "manifest_sha256": sha_file(d / "manifest.json"),
                          "npz_path": m["npz_path"], "npz_sha256": m["files"]["__npz__"], **{f: h for f, h in m["files"].items()
                                                                                             if f != "__npz__"}})
    out.mkdir(parents=True)
    keys = {"state_rows": ["fit_id", "checkpoint"], "probe_rows": ["fit_id", "checkpoint"],
            "replicate_rows": ["fit_id", "checkpoint", "r", "kind"], "checkpoint_rows": ["fit_id", "checkpoint"],
            "iteration_rows": ["fit_id", "iteration"]}
    for n, parts in tables.items():
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if len(df):
            ns.need(not df.duplicated(keys[n]).any(), f"duplicate rows in {n}")
        df.to_csv(out / f"{n}.csv", index=False)
    pd.DataFrame(ledger).to_csv(out / "ledger.csv", index=False)
    pd.DataFrame(index).to_csv(out / "fit_index.csv", index=False)
    info = {**ns.common_meta(sname), "addendum": ADDENDUM_VERSION, "status": "COMPLETED" if st["complete"] else "PARTIAL",
            "counts": st["counts"], "code_sha256": ctx.code_sha, "artifact_root": str(ctx.artifact_root / sname),
            "thread_vars": {v: os.environ.get(v) for v in THREAD_VARS}}
    (out / "runinfo.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps({"stage": sname, **st["counts"], "status": info["status"]}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "status", "aggregate"))
    ap.add_argument("--stage", choices=("phase3", "phase4"), required=True)
    ap.add_argument("--L", type=int)
    ap.add_argument("--conds", nargs="+", choices=list(ns.CONDS))
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    ap.add_argument("--mem-min-gb", type=float, default=3.0)
    ap.add_argument("--mem-wait-min", type=float, default=10.0)
    ap.add_argument("--deadline-h", type=float, default=6.0)
    ap.add_argument("--max-fits", type=int)
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args(argv)
    if a.stage == "phase4":
        ns.need(a.L in ns.P4_LS, "phase4 needs a pre-registered --L")
    conds = a.conds or (list(ns.CONDS) if a.stage == "phase3" else list(ns.P4_CONDS))
    if a.stage == "phase4":
        ns.need(set(conds) <= set(ns.P4_CONDS), "phase 4 conditions are A and D")
    ctx = Ctx(a.results_root, a.artifact_root)
    if a.cmd == "status":
        st = stage_status(ctx, a.stage, a.L, conds)
        print(json.dumps({k: v for k, v in st.items() if k != "fits"}, indent=1))
        bad = [f for f in st["fits"] if f["state"] not in ("OK",)]
        for f in bad[:20]:
            print(f)
        return 0
    if a.cmd == "aggregate":
        return aggregate(ctx, a.stage, a.L, conds, a.allow_partial)
    if a.results_root == RESULTS_ROOT:
        ns.need(ns.tree_clean(), "working tree not clean (outside this experiment's output dir)")
        g = json.loads((ns.OUT_ROOT / "gates" / "gates.json").read_text(encoding="utf-8"))
        ns.need(g.get("status") == "PASSED" and g.get("prereg_sha256") == ns.prereg_hash(), "gates not passed / prereg changed")
        ns.need((ns.OUT_ROOT / "operational_addendum_v2.json").exists(), "operational addendum not recorded")
    return run_stage(ctx, a.stage, a.L, conds, mem_min_gb=a.mem_min_gb, mem_wait_min=a.mem_wait_min, deadline_h=a.deadline_h,
                     max_fits=a.max_fits)


if __name__ == "__main__":
    sys.exit(main())
