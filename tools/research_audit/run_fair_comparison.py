"""Runner for the pre-registered fair comparison (fair_protocol.py).

Stages: gate | e1 | e2 | e3.  One fit = one (stage, generator/condition, K_GEN,
rep, K, start, estimator).  Every fit is saved atomically (npz state, then a
JSON row with the npz sha256, written last).  On resume a fit whose JSON and
npz hash validate is skipped; anything else is recomputed from scratch (a fit
is never resumed mid-way).  No retry, no seed change: a failure is recorded
as FAILED.  BLAS threads pinned to 1, one process, memory guard 3 GB.

Lineage E prototype.  The historical estimator (reproduction/src/model.py) is
called through an adapter; its code is not modified.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "reproduction/src", ROOT / "expfam/src/experimental", ROOT / "tools/research_audit"):
    sys.path.insert(0, str(p))

import fair_protocol as fp  # noqa: E402
from data_generator import set_true_params  # noqa: E402
from model import LatentStructuralModel  # noqa: E402
from experiment_paper_2 import Experiment2Runner  # noqa: E402
import run_clap_clean_nsweep as a1  # noqa: E402
import run_k_signal_strength as rss  # noqa: E402
from data_generator_canonical import generate_canonical_data, f_scale_for_row_norm  # noqa: E402
from em_runner_instrumented import run_em_instrumented  # noqa: E402

hist = a1.hist
RESULTS = ROOT / fp.RESULTS_REL
STATES = Path(fp.STATE_ROOT)


def need(c, m):
    if not c:
        raise SystemExit("STOP: " + m)


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha_arr(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def atomic_write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_npz_atomic(path: Path, arrays: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return sha_file(path)


def avail_gb() -> float:
    return psutil.virtual_memory().available / 1e9


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def upper_mask(n):
    return np.triu(np.ones((n, n), dtype=bool), 1)


def all_pairs(n):
    m = np.ones((n, n), dtype=bool)
    np.fill_diagonal(m, False)
    return m


def e1_data(g: int, kgen: int, rep: int):
    n, d = 150, 15
    if g == 0:
        p = set_true_params(n=n, d=d, k=kgen, seed=1980 + 100 * kgen + 1000 * (rep - 1))
        return dict(X=p["X"], Y=p["Y"], Z=p["Z"], F=p["F"], w0=p["w0"], w=p["w"], train=all_pairs(n),
                    test=None, data_seed=1980 + 100 * kgen + 1000 * (rep - 1), x_family="gaussian",
                    note="G_old: Z and X z-scored; F is pre-standardisation")
    seed = 7610000 + 100 * kgen + rep
    ds = generate_canonical_data(n=n, d=d, k=kgen, seed=seed, family_x="gaussian", family_y="bernoulli",
                                 f_scale=f_scale_for_row_norm(0.9, d=d, k=kgen), sigma_x_var=0.1,
                                 w0=-1.0, w=fp.e1_canon_w(kgen))
    return dict(X=ds.X, Y=ds.Y, Z=ds.Z, F=ds.F, w0=-1.0, w=fp.e1_canon_w(kgen), train=all_pairs(n), test=None,
                data_seed=seed, x_family="gaussian", note="G_canon")


_PROV = None
_W0C = None


def e2_bundle(n, rep):
    global _PROV, _W0C
    if _PROV is None:
        _PROV = a1.load_hist_provenance()
        _W0C = rss.w0_c()
    return rss.build_all(n, rep, _PROV, _W0C)


def gaussian_x(Z, F, dseed):
    s2 = np.exp(-0.5 * np.sum(F ** 2, axis=1))
    rng = np.random.default_rng([dseed, fp.TAG_GX])
    return Z @ F.T + rng.standard_normal((Z.shape[0], F.shape[0])) * np.sqrt(s2), s2


def e2_data(n, rep, signal, xfam):
    b = e2_bundle(n, rep)
    c = "A" if signal == "weak" else "D"
    cond = b["cond"][c]
    if xfam == "poisson":
        X = cond["X"]
        s2 = None
    else:
        X, s2 = gaussian_x(b["ds"].Z, cond["F"], b["dseed"])
    return dict(X=X, Y=cond["Y"], Z=b["ds"].Z, F=cond["F"], w0=cond["w0"], w=cond["w"], train=b["train"],
                test=b["test"], data_seed=b["dseed"], x_family=xfam, gaussian_noise_var=s2,
                note=f"PR #155 condition {c}")


def thin_mask(train, dseed, frac=0.8):
    n = train.shape[0]
    up = np.argwhere(np.triu(train, 1))
    rng = np.random.default_rng([dseed, fp.TAG_MASK])
    keep = rng.choice(len(up), size=int(round(frac * len(up))), replace=False)
    m = np.zeros((n, n), dtype=bool)
    sel = up[np.sort(keep)]
    m[sel[:, 0], sel[:, 1]] = True
    return m | m.T


def e3_data(rep, frac):
    d = e2_data(75, rep, "weak", "poisson")
    if frac < 1.0:
        d["train"] = thin_mask(d["train"], d["data_seed"], frac)
        need(not np.any(d["train"] & d["test"]), "thinned mask overlaps holdout")
    d["note"] += f"; train fraction {frac}"
    return d


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

def fit_old(X, Y, k, seed, L=5, iters=8):
    n, d = X.shape
    m = LatentStructuralModel(n=n, d=d, k=k, L=L)
    m.initialize_params(true_params=None, seed=seed)
    up = upper_mask(n)
    dens = float(np.clip(Y[up].mean(), 1e-6, 1 - 1e-6))
    m.params["w0"] = math.log(dens / (1 - dens))
    m.params["w"] = 0.5
    rng = np.random.default_rng(seed)
    Z = m.params["Z"].copy()
    F = m.params["F"].copy()
    sigma = m.params["sigma"].copy()
    w0, w, var_z = float(m.params["w0"]), float(m.params["w"]), float(m.params["var_z"])
    plug = []
    for _ in range(iters):
        Zs = np.zeros((n, k, L))
        for l in range(L):
            m.params.update(dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w))
            Zn = m.calc_eta_newton(X, Y, rng=rng, max_iter=fp.COMMON["newton_max_iter"], alpha=fp.COMMON["newton_alpha"])
            Zs[:, :, l] = Zn
            Z = Zn.copy()
        if not np.all(np.isfinite(Zs)):
            raise FloatingPointError("non-finite E-step (OLD)")
        Zs = m.scale_Z(Zs)
        Z = Zs[:, :, -1].copy()
        F = m.calc_F(X, Zs)
        sigma = m.calc_sigma(X, Zs, F)
        w0 = float(m.calc_w0(Y, Zs, w0, w, max_iter=fp.COMMON["adam_max_iter"]))
        w = float(m.calc_w(Y, Zs, w0, w, max_iter=fp.COMMON["adam_max_iter"]))
        plug.append(float(Experiment2Runner.calc_log_likelihood(None, X, Y, Z, F, sigma, var_z, w0, w)))
    need(all(np.isfinite(plug)), "non-finite OLD plug-in")
    return {"Zsamples": Zs, "Z": Z, "F": F, "sigma": np.asarray(sigma, float),
            "scalars": np.array([w0, w, var_z]), "plugin_by_iter": np.array(plug)}


def fit_E(X, Y, train, k, seed, x_family):
    holder = {}

    def cb(rec):
        if int(rec["iteration"]) == fp.COMMON["em_iterations"]:
            holder["rec"] = rec
    res = run_em_instrumented(X, Y, family_x=x_family, family_y="bernoulli", k=k, L=fp.COMMON["L"],
                              max_iter=fp.COMMON["em_iterations"],
                              checkpoints=tuple(range(1, fp.COMMON["em_iterations"] + 1)), seed=seed,
                              train_mask=train, numerics_mode="consistent", stability_window=4, on_nan="fail",
                              on_checkpoint=cb)
    need(res["retry_count"] == 0 and not res["nan_detected"], "unexpected retry / NaN in E")
    rec = holder["rec"]
    Zs = np.array(rec["Z_samples"])
    sig = np.asarray(rec["sigma"], float)
    return {"Zsamples": Zs, "Z": np.array(rec["Z_est"]), "F": np.array(rec["F"]), "sigma": sig,
            "scalars": np.array([rec["w0"], rec["w"], rec["var_z"]], dtype=float)}


# ---------------------------------------------------------------------------
# fit list per stage
# ---------------------------------------------------------------------------

def stage_fits(stage):
    out = []
    if stage == "gate":
        for src, n, sig in (("A_n75", 75, "weak"), ("D_n75", 75, "strong"), ("A_n150", 150, "weak")):
            out.append(dict(stage="gate", key=f"gate_{src}_r1_k5_s1", kind="E2", n=n, rep=1, k=5, s=1,
                            signal=sig, xfam="poisson", estimator="E", seed=hist.model_seed(5, n, 1, 5, 1)))
    elif stage == "e1":
        for g in (0, 1):
            for kg in fp.E1["K_GEN"]:
                for rep in fp.E1["replicates"]:
                    for k in fp.E1["K_CAND"]:
                        for s in fp.COMMON["starts"]:
                            for est in ("OLD", "E"):
                                gname = "Gold" if g == 0 else "Gcanon"
                                out.append(dict(stage="e1", key=f"e1_{gname}_K{kg}_r{rep}_k{k}_s{s}_{est}", kind="E1",
                                                g=g, kgen=kg, rep=rep, k=k, s=s, estimator=est,
                                                seed=fp.e1_seed(g, kg, rep, k, s)))
    elif stage == "e2":
        for xfam in ("gaussian", "poisson"):
            for n in fp.E2["n"]:
                for sig in ("weak", "strong"):
                    if xfam == "poisson" and not (n == 150 and sig == "strong"):
                        continue          # reused cells
                    for rep in fp.E2["replicates"]:
                        for k in fp.E2["K_CAND"]:
                            for s in fp.COMMON["starts"]:
                                out.append(dict(stage="e2", key=f"e2_{xfam}_{sig}_n{n}_r{rep}_k{k}_s{s}", kind="E2",
                                                n=n, rep=rep, k=k, s=s, signal=sig, xfam=xfam, estimator="E",
                                                seed=hist.model_seed(5, n, rep, k, s)))
    elif stage == "e3":
        for rep in fp.E3["replicates"]:
            for k in fp.E3["K_CAND"]:
                for s in fp.COMMON["starts"]:
                    out.append(dict(stage="e3", key=f"e3_frac080_r{rep}_k{k}_s{s}", kind="E3", n=75, rep=rep, k=k,
                                    s=s, frac=0.8, estimator="E", seed=hist.model_seed(5, 75, rep, k, s)))
    return out


def data_for(f):
    if f["kind"] == "E1":
        return e1_data(f["g"], f["kgen"], f["rep"])
    if f["kind"] == "E2":
        return e2_data(f["n"], f["rep"], f["signal"], f["xfam"])
    return e3_data(f["rep"], f["frac"])


def valid(f) -> bool:
    j = RESULTS / f["stage"] / "fits" / f"{f['key']}.json"
    if not j.exists():
        return False
    row = json.loads(j.read_text(encoding="utf-8"))
    if row.get("status") != "OK":
        return row.get("status") == "FAILED"          # recorded failure: never retried
    npz = STATES / f["stage"] / f"{f['key']}.npz"
    return npz.exists() and sha_file(npz) == row.get("npz_sha256")


def run_stage(stage, max_fits=None):
    fits = stage_fits(stage)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "tools", "expfam/src", "reproduction/src"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    need(stage == "gate" or not dirty, "code tree dirty: commit before a production stage")
    log = RESULTS / stage / "run_log.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    done = 0
    with log.open("a", encoding="utf-8") as lg:
        lg.write(f"{time.strftime('%H:%M:%S')} start {stage} fits={len(fits)} head={head} protocol={fp.protocol_hash()[:12]}\n")
        for f in fits:
            if valid(f):
                continue
            if max_fits is not None and done >= max_fits:
                break
            waited = 0
            while avail_gb() < fp.COMMON["mem_min_gb"]:
                if waited > 600:
                    lg.write(f"{time.strftime('%H:%M:%S')} STOP low memory\n")
                    return
                time.sleep(30)
                waited += 30
            dkey = (f["kind"], f.get("g"), f.get("kgen"), f.get("n"), f["rep"], f.get("signal"), f.get("xfam"), f.get("frac"))
            if dkey not in cache:
                cache.clear()
                cache[dkey] = data_for(f)
            D = cache[dkey]
            t0 = time.perf_counter()
            row = dict(f, protocol_hash=fp.protocol_hash(), git_head=head, data_seed=D["data_seed"],
                       x_sha256=sha_arr(D["X"]), y_sha256=sha_arr(D["Y"]), train_sha256=sha_arr(D["train"]),
                       x_family=D["x_family"])
            try:
                if f["estimator"] == "OLD":
                    st = fit_old(D["X"], D["Y"], f["k"], f["seed"])
                else:
                    st = fit_E(D["X"], D["Y"], D["train"], f["k"], f["seed"], D["x_family"])
                npz = STATES / stage / f"{f['key']}.npz"
                row["npz_sha256"] = save_npz_atomic(npz, st)
                row["status"] = "OK"
                row["exception"] = ""
            except (FloatingPointError, SystemExit, Exception) as e:          # recorded, never retried
                row["status"] = "FAILED"
                row["exception"] = f"{type(e).__name__}: {e}"[:300]
                row["npz_sha256"] = ""
            row["runtime_s"] = round(time.perf_counter() - t0, 2)
            row["avail_gb"] = round(avail_gb(), 2)
            row["rss_mb"] = round(psutil.Process().memory_info().rss / 1e6, 1)
            atomic_write_bytes(RESULTS / stage / "fits" / f"{f['key']}.json",
                               json.dumps(row, indent=1, default=str).encode("utf-8"))
            lg.write(f"{time.strftime('%H:%M:%S')} {row['status']} {f['key']} {row['runtime_s']}s avail={row['avail_gb']}GB\n")
            lg.flush()
            done += 1
        lg.write(f"{time.strftime('%H:%M:%S')} end {stage} new={done}\n")


def status(stage):
    fits = stage_fits(stage)
    c = {"OK": 0, "FAILED": 0, "NOT_RUN": 0}
    for f in fits:
        j = RESULTS / stage / "fits" / f"{f['key']}.json"
        if not j.exists():
            c["NOT_RUN"] += 1
            continue
        st = json.loads(j.read_text(encoding="utf-8")).get("status")
        c[st if st in c else "NOT_RUN"] += 1 if valid(f) else 0
    print(stage, len(fits), c)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status"])
    ap.add_argument("--stage", required=True, choices=["gate", "e1", "e2", "e3"])
    ap.add_argument("--max-fits", type=int, default=None)
    a = ap.parse_args()
    run_stage(a.stage, a.max_fits) if a.cmd == "run" else status(a.stage)
