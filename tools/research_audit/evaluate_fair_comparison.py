"""Score every fitted state of the fair comparison with the common evaluator.

For each fit (new E1/E2/E3 fits and the reused PR #155 / A1 states, gate PASS)
computes, on the SAME state:
  A C_Q, B C_Q - P_Z, C conditional plug-in at Z_last (fair_common_evaluator),
  Laplace mode Z_hat and C_Lap (laplace_k_criterion; estimator-independent: the
  theta of either estimator is loaded into a lineage-E consistent model object
  used only as a container of formulas), D conditional plug-in at Z_hat,
  PY_hist (E1 OLD only),
  predictive metrics (held-out Y log score; X fit) and parameter recovery.
Writes criterion_scores.csv (one row per fit).  Refuses to overwrite.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "reproduction/src", ROOT / "expfam/src/experimental", ROOT / "tools/research_audit"):
    sys.path.insert(0, str(p))

import fair_protocol as fp  # noqa: E402
import fair_common_evaluator as ev  # noqa: E402
import run_fair_comparison as rf  # noqa: E402
import laplace_k_criterion as lk  # noqa: E402
import em_runner  # noqa: E402
import run_clap_clean_nsweep as a1  # noqa: E402

hist = a1.hist
RES = ROOT / fp.RESULTS_REL
ART = Path(r"D:\tento\kennkyu_artifacts")


def reused_fits():
    """E2 / E3 cells covered by existing states (gate PASS)."""
    out = []
    emc, _ = a1.load_emc_identity()
    for n in (75, 150):
        for rep in fp.E2["replicates"]:
            for k in fp.E2["K_CAND"]:
                for s in fp.COMMON["starts"]:
                    fi = int(emc[(5, n, rep, k, s)]["fit_index"])
                    out.append(dict(stage="reuse", key=f"reuse_A1emc_n{n}_r{rep}_k{k}_s{s}", kind="E2", n=n, rep=rep,
                                    k=k, s=s, signal="weak", xfam="poisson", estimator="E",
                                    seed=hist.model_seed(5, n, rep, k, s), src="emc", src_key=f"{fi:04d}_it008"))
    for rep in fp.E2["replicates"]:
        for k in fp.E2["K_CAND"]:
            for s in fp.COMMON["starts"]:
                out.append(dict(stage="reuse", key=f"reuse_pr155D_n75_r{rep}_k{k}_s{s}", kind="E2", n=75, rep=rep, k=k,
                                s=s, signal="strong", xfam="poisson", estimator="E",
                                seed=hist.model_seed(5, 75, rep, k, s), src="pr155_n75",
                                src_key=f"n75_D_r{rep}_k{k}_s{s}"))
    return out


_NPZ = {}


def load_state(f):
    if f["stage"] != "reuse":
        z = np.load(rf.STATES / f["stage"] / f"{f['key']}.npz")
        return {k: z[k] for k in z.files}
    if f["src"] == "emc":
        z = _NPZ.setdefault("emc", np.load(a1.EMC_NPZ))
    else:
        z = _NPZ.setdefault("pr155", np.load(ART / "historical_canonical_causal_20261106/fit_states_n75.npz"))
    return {k: z[f"{f['src_key']}_{k}"] for k in ("Zsamples", "Z", "F", "sigma", "scalars")}


def theta_of(st, xfam):
    w0, w, vz = (float(v) for v in st["scalars"])
    sig = np.asarray(st["sigma"], float)
    sd = np.diag(sig) if sig.ndim == 2 else sig
    return {"F": np.asarray(st["F"], float), "sigma_diag": sd, "w0": w0, "w": w, "var_z": vz}


def laplace(X, Y, train, st, th, xfam, k):
    n, d = X.shape
    m = em_runner.build_model(n, d, k, fp.COMMON["L"], xfam, "bernoulli", train_mask=train, numerics_mode="consistent")
    m.initialize_params(true_params=None, seed=0)
    sig = np.diag(th["sigma_diag"])
    m.params.update({"Z": np.asarray(st["Z"], float).copy(), "F": th["F"], "sigma": sig, "w0": th["w0"], "w": th["w"],
                     "var_z": th["var_z"]})
    res = lk.laplace_log_observed(m, X, Y, np.asarray(st["Z"], float).copy(), grad_tol=1e-8, max_iter=200)
    out = lk.calc_C_Lap(res, d_K=ev.num_params(k, d, xfam), N=float(n))
    return res, out


def metrics(D, st, th, xfam, train, test):
    Zs = np.asarray(st["Zsamples"], float)
    n, k, L = Zs.shape
    out = {}
    # X fit (in-sample): Gaussian R^2 of the posterior-mean prediction; Poisson mean deviance
    mu = np.mean([Zs[:, :, l] @ th["F"].T if xfam == "gaussian" else np.exp(Zs[:, :, l] @ th["F"].T) for l in range(L)], axis=0)
    X = D["X"]
    if xfam == "gaussian":
        out["x_r2_in_sample"] = float(1 - np.sum((X - mu) ** 2) / np.sum((X - X.mean(0)) ** 2))
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(X > 0, X * np.log(X / mu), 0.0) - (X - mu)
        out["x_poisson_mean_deviance"] = float(2 * np.mean(t))
    # held-out Y
    P = np.mean([1 / (1 + np.exp(-(th["w0"] + th["w"] * Zs[:, :, l] @ Zs[:, :, l].T))) for l in range(L)], axis=0)
    if test is not None:
        up = np.triu(test, 1)
        y, p = D["Y"][up], np.clip(P[up], 1e-12, 1 - 1e-12)
        out["y_heldout_mean_logscore"] = float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        out["y_heldout_pairs"] = int(up.sum())
    # recovery (truth available; rotation-invariant quantities)
    Zt, Ft = D["Z"], D["F"]
    iu = np.triu_indices(n, 1)
    G_hat = np.mean([Zs[:, :, l] @ Zs[:, :, l].T for l in range(L)], axis=0)
    out["zzT_corr"] = float(np.corrcoef(G_hat[iu], (Zt @ Zt.T)[iu])[0, 1])
    eta_hat = th["w0"] + th["w"] * G_hat[iu]
    eta_true = D["w0"] + D["w"] * (Zt @ Zt.T)[iu]
    out["etaY_rel_rmse"] = float(np.sqrt(np.mean((eta_hat - eta_true) ** 2)) / np.std(eta_true))
    if D.get("note", "").startswith("G_old"):
        out["gram_FFT_rel_err"] = float("nan")       # X z-scored: F_true is not on the fitted scale
    else:
        A = Ft @ Ft.T
        out["gram_FFT_rel_err"] = float(np.linalg.norm(th["F"] @ th["F"].T - A) / np.linalg.norm(A))
    return out


def all_fits():
    fl = []
    for st in ("e1", "e2", "e3"):
        for f in rf.stage_fits(st):
            j = RES / st / "fits" / f"{f['key']}.json"
            row = json.loads(j.read_text(encoding="utf-8")) if j.exists() else {"status": "NOT_RUN"}
            fl.append(dict(f, status=row.get("status"), runtime_s=row.get("runtime_s"), exception=row.get("exception", "")))
    for f in reused_fits():
        fl.append(dict(f, status="OK (reused, gate PASS)", runtime_s=None, exception=""))
    return fl


def main():
    out = RES / "criterion_scores.csv"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    rows = []
    cache = {}
    fits = all_fits()
    t0 = time.time()
    for i, f in enumerate(fits):
        base = {k: f.get(k) for k in ("stage", "key", "kind", "estimator", "g", "kgen", "n", "rep", "k", "s", "signal",
                                      "xfam", "frac", "seed", "status", "runtime_s", "exception")}
        if not str(f["status"]).startswith("OK"):
            rows.append(base)
            continue
        dkey = (f["kind"], f.get("g"), f.get("kgen"), f.get("n"), f["rep"], f.get("signal"), f.get("xfam"), f.get("frac"))
        if dkey not in cache:
            cache.clear()
            cache[dkey] = rf.data_for(f) if f["stage"] != "reuse" else rf.e2_data(f["n"], f["rep"], f["signal"], "poisson")
        D = cache[dkey]
        st = load_state(f)
        xfam = D["x_family"]
        th = theta_of(st, xfam)
        X, Y, train, test = D["X"], D["Y"], D["train"], D["test"]
        Zs = np.asarray(st["Zsamples"], float)
        Zl = np.asarray(st["Z"], float)
        c = ev.criteria_abc(X, Y, Zs, Zl, th, xfam, train)
        r = dict(base, **{k: v for k, v in c.items()}, var_z=th["var_z"], w0_hat=th["w0"], w_hat=th["w"],
                 last_is_sample_L=bool(np.array_equal(Zl, Zs[:, :, -1])))
        tl = time.perf_counter()
        try:
            res, lap = laplace(X, Y, train, st, th, xfam, f["k"])
            r.update(lap_status=res.status, E_CLap=lap["C_Lap"] if res.ok else np.nan,
                     D_cond_mode=(-2 * ev.data_loglik(X, Y, res.Z_hat, th, xfam, train) + c["P_theta"]) if res.ok else np.nan,
                     lap_grad_inf=res.grad_inf, lap_min_eig_H=res.min_eigenvalue_H, lap_iterations=res.iterations)
        except Exception as e:  # recorded, not imputed
            r.update(lap_status=f"ERROR {type(e).__name__}", E_CLap=np.nan, D_cond_mode=np.nan)
        r["lap_seconds"] = round(time.perf_counter() - tl, 2)
        if f.get("estimator") == "OLD" and "plugin_by_iter" in st:
            d = X.shape[1]
            pen26 = ((f["k"] + 1) * d - f["k"] * (f["k"] - 1) / 2) * math.log(X.shape[0])
            r["PY_hist"] = -2 * float(np.max(st["plugin_by_iter"])) + pen26
            r["PY_final_iter"] = -2 * float(st["plugin_by_iter"][-1]) + pen26
        r.update(metrics(D, st, th, xfam, train, test))
        rows.append(r)
        if i % 25 == 0:
            print(f"{i}/{len(fits)} {f['key']} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, lineterminator="\n")
    print("rows", len(df))


if __name__ == "__main__":
    main()
