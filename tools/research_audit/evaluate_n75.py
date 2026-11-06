"""Score every state of the n=75 study (and the reused PR #160 n=150 E1 K_GEN=5 states) with the common evaluator.

Per state: criteria A-E (fair_common_evaluator + laplace_k_criterion, estimator-independent),
PY_type for OLD (reference only), recovery and predictive metrics (n75_protocol.SECONDARY).
Reused PR #160 states are re-scored with the same code and compared with PR #160
criterion_scores.csv (abs diff < 1e-6 required, recorded in reuse_crosscheck.csv).
Writes criterion_scores.csv.  Refuses to overwrite.  No fit, no sampling.
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

import n75_protocol as npc  # noqa: E402
import run_n75 as rn  # noqa: E402
import run_fair_comparison as rf  # noqa: E402
import fair_common_evaluator as ev  # noqa: E402
import evaluate_fair_comparison as efc  # noqa: E402

RES = ROOT / npc.RESULTS_REL
P160 = ROOT / npc.PR160_REL


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def all_fits():
    fl = []
    for st in npc.STAGES:
        for f in rn.stage_fits(st):
            j = RES / st / "fits" / f"{f['key']}.json"
            row = json.loads(j.read_text(encoding="utf-8")) if j.exists() else {"status": "NOT_RUN"}
            fl.append(dict(f, n=75, status=row.get("status"), runtime_s=row.get("runtime_s"), exception=row.get("exception", ""),
                           src="new"))
    for g, gname in ((1, "Gcanon"), (0, "Gold")):
        for rep in (1, 2):
            for k in npc.COMMON["K_CAND"]:
                for s in npc.COMMON["starts"]:
                    for est in ("OLD", "E"):
                        key = f"e1_{gname}_K5_r{rep}_k{k}_s{s}_{est}"
                        j = P160 / "e1" / "fits" / f"{key}.json"
                        row = json.loads(j.read_text(encoding="utf-8"))
                        fl.append(dict(stage="P160", key=key, g=g, rep=rep, k=k, s=s, estimator=est, mask="all_pairs",
                                       seed=row["seed"], n=150, status=row["status"], runtime_s=row.get("runtime_s"),
                                       exception=row.get("exception", ""), src="PR #160 E1 (reused, read-only)"))
    return fl


def load_state(f):
    root = Path(npc.STATE_ROOT) if f["stage"] != "P160" else Path(npc.PR160_STATES)
    sub = f["stage"] if f["stage"] != "P160" else "e1"
    z = np.load(root / sub / f"{f['key']}.npz")
    return {k: z[k] for k in z.files}


def data_for(f):
    if f["stage"] == "P160":
        d = rf.e1_data(f["g"], 5, f["rep"])
        d["g"] = f["g"]
        return d
    return rn.stage_data(f["stage"], f["rep"])


def subspace_recovery(Zhat, Zt):
    q1, _ = np.linalg.qr(Zhat)
    q2, _ = np.linalg.qr(Zt)
    s = np.linalg.svd(q1.T @ q2, compute_uv=False)
    return float(np.sum(np.clip(s, 0, 1) ** 2) / Zt.shape[1])


def metrics(D, st, th):
    Zs = np.asarray(st["Zsamples"], float)
    n, k, L = Zs.shape
    Zt = D["Z"]
    iu = np.triu_indices(n, 1)
    out = {}
    G_hat = np.mean([Zs[:, :, l] @ Zs[:, :, l].T for l in range(L)], axis=0)
    G_true = Zt @ Zt.T
    out["zzT_corr"] = float(np.corrcoef(G_hat[iu], G_true[iu])[0, 1])
    out["subspace_recovery"] = subspace_recovery(Zs.mean(axis=2), Zt)
    eta_hat = th["w0"] + th["w"] * G_hat[iu]
    eta_true = D["w0"] + D["w"] * G_true[iu]
    out["etaY_rel_rmse"] = float(np.sqrt(np.mean((eta_hat - eta_true) ** 2)) / np.std(eta_true))
    p_true = sig(eta_true)
    p_hat = np.clip(np.mean([sig(th["w0"] + th["w"] * (Zs[:, :, l] @ Zs[:, :, l].T)[iu]) for l in range(L)], axis=0), 1e-12, 1 - 1e-12)
    out["y_population_logscore"] = float(np.mean(p_true * np.log(p_hat) + (1 - p_true) * np.log(1 - p_hat)))
    out["y_population_logscore_oracle"] = float(np.mean(p_true * np.log(np.clip(p_true, 1e-12, 1)) + (1 - p_true) * np.log(np.clip(1 - p_true, 1e-12, 1))))
    if D.get("test") is not None:
        P = np.mean([sig(th["w0"] + th["w"] * Zs[:, :, l] @ Zs[:, :, l].T) for l in range(L)], axis=0)
        up = np.triu(D["test"], 1)
        y, p = D["Y"][up], np.clip(P[up], 1e-12, 1 - 1e-12)
        out["y_heldout_logscore"] = float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        out["y_heldout_pairs"] = int(up.sum())
    if D["g"] == 1:      # canonical truth on the fitted scale
        A = D["F"] @ D["F"].T
        out["gram_FFT_rel_err"] = float(np.linalg.norm(th["F"] @ th["F"].T - A) / np.linalg.norm(A))
        mu_true = Zt @ D["F"].T
        mu_hat = np.mean([Zs[:, :, l] @ th["F"].T for l in range(L)], axis=0)
        out["x_signal_r2"] = float(1 - np.sum((mu_hat - mu_true) ** 2) / np.sum((mu_true - mu_true.mean(0)) ** 2))
    X = D["X"]
    mu = np.mean([Zs[:, :, l] @ th["F"].T for l in range(L)], axis=0)
    out["x_r2_in_sample"] = float(1 - np.sum((X - mu) ** 2) / np.sum((X - X.mean(0)) ** 2))
    out["mc_sample_var_trace"] = float(np.mean([np.trace(np.cov(Zs[i], bias=False)) if k > 1 else float(np.var(Zs[i], ddof=1))
                                                for i in range(n)]))
    return out


def main():
    out = RES / "criterion_scores.csv"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    rows = []
    cache = {}
    t0 = time.time()
    fits = all_fits()
    for i, f in enumerate(fits):
        base = {k: f.get(k) for k in ("stage", "key", "g", "rep", "k", "s", "estimator", "mask", "seed", "n", "status",
                                      "runtime_s", "exception", "src")}
        base["generator"] = "G_canon" if f["g"] == 1 else "G_old"
        if f["status"] != "OK":
            rows.append(base)
            continue
        dk = (f["stage"], f["g"], f["rep"])
        if dk not in cache:
            cache = {dk: data_for(f)}
        D = cache[dk]
        st = load_state(f)
        th = efc.theta_of(st, "gaussian")
        X, Y, train = D["X"], D["Y"], D["train"]
        Zs, Zl = np.asarray(st["Zsamples"], float), np.asarray(st["Z"], float)
        c = ev.criteria_abc(X, Y, Zs, Zl, th, "gaussian", train)
        r = dict(base, **c, w0_hat=th["w0"], w_hat=th["w"], var_z=th["var_z"], w_true=float(D["w"]), w0_true=float(D["w0"]))
        tl = time.perf_counter()
        try:
            res, lap = efc.laplace(X, Y, train, st, th, "gaussian", f["k"])
            r.update(lap_status=res.status, E_CLap=lap["C_Lap"] if res.ok else np.nan,
                     D_cond_mode=(-2 * ev.data_loglik(X, Y, res.Z_hat, th, "gaussian", train) + c["P_theta"]) if res.ok else np.nan,
                     lap_grad_inf=res.grad_inf, lap_min_eig_H=res.min_eigenvalue_H)
        except Exception as e:  # recorded, not imputed
            r.update(lap_status=f"ERROR {type(e).__name__}", E_CLap=np.nan, D_cond_mode=np.nan)
        r["lap_seconds"] = round(time.perf_counter() - tl, 2)
        if f["estimator"] == "OLD" and "plugin_by_iter" in st:
            n, d = X.shape
            pen26 = ((f["k"] + 1) * d - f["k"] * (f["k"] - 1) / 2) * math.log(n)
            r["PY_type"] = -2 * float(np.max(st["plugin_by_iter"])) + pen26
        r.update(metrics(D, st, th))
        rows.append(r)
        if i % 50 == 0:
            print(f"{i}/{len(fits)} {f['key']} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    # reuse cross-check against PR #160
    old = pd.read_csv(P160 / "criterion_scores.csv").set_index("key")
    rp = df[df.stage == "P160"].set_index("key")
    cc = []
    for c in ("A_CQ", "B_CQ_minus_PZ", "C_cond_last", "D_cond_mode", "E_CLap", "PY_type"):
        oc = "PY_hist" if c == "PY_type" else c
        dlt = (rp[c] - old.loc[rp.index, oc]).abs()
        cc.append(dict(criterion=c, n_states=int(dlt.notna().sum()), max_abs_diff=float(dlt.max()), pass_1e6=bool(dlt.max() < 1e-6)))
    pd.DataFrame(cc).to_csv(RES / "reuse_crosscheck.csv", index=False, lineterminator="\n")
    print(pd.DataFrame(cc).to_string())
    df.to_csv(out, index=False, lineterminator="\n")
    print("rows", len(df))


if __name__ == "__main__":
    main()
