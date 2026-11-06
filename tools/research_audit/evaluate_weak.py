"""Score every state of weak_protocol (new all-pairs OLD / E) and the reused PR #160 E2 weak n=75 80% E states.

Per state: criteria A-E (fair_common_evaluator + laplace, estimator-independent, on the condition's
training pairs), PY_type for OLD (reference only), recovery / predictive metrics (evaluate_n75.metrics,
canonical truth).  Reused PR #160 states are re-scored with the same code and compared with PR #160
criterion_scores.csv (abs diff < 1e-6 required, recorded in reuse_crosscheck.csv).
Writes criterion_scores.csv.  Refuses to overwrite.  No fit, no sampling.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "reproduction/src", ROOT / "expfam/src/experimental", ROOT / "tools/research_audit"):
    sys.path.insert(0, str(p))

import weak_protocol as wp  # noqa: E402
import run_weak as rw  # noqa: E402
import fair_common_evaluator as ev  # noqa: E402
import evaluate_fair_comparison as efc  # noqa: E402
from evaluate_n75 import metrics  # noqa: E402  (unchanged PR #161 metric code)

RES = ROOT / wp.RESULTS_REL
P160 = ROOT / wp.PR160_REL


def all_fits():
    fl = []
    for f in rw.stage_fits("W"):
        j = RES / "W" / "fits" / f"{f['key']}.json"
        row = json.loads(j.read_text(encoding="utf-8")) if j.exists() else {"status": "NOT_RUN"}
        fl.append(dict(f, status=row.get("status"), runtime_s=row.get("runtime_s"), exception=row.get("exception", ""),
                       src="new", root=str(Path(wp.STATE_ROOT) / "W")))
    for rep in wp.COMMON["replicates"]:
        for k in wp.COMMON["K_CAND"]:
            for s in wp.COMMON["starts"]:
                key = rw.p160_key(rep, k, s)
                row = json.loads((P160 / "e2" / "fits" / f"{key}.json").read_text(encoding="utf-8"))
                fl.append(dict(stage="P160_80", key=key, rep=rep, k=k, s=s, estimator="E", mask="mask_80", seed=row["seed"],
                               status=row["status"], runtime_s=row.get("runtime_s"), exception=row.get("exception", ""),
                               src="PR #160 E2 (reused, read-only)", root=str(Path(wp.PR160_STATES) / "e2")))
    return fl


def main():
    out = RES / "criterion_scores.csv"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rows, cache = [], {}
    t0 = time.time()
    fits = all_fits()
    for i, f in enumerate(fits):
        base = {k: f.get(k) for k in ("stage", "key", "rep", "k", "s", "estimator", "mask", "seed", "status", "runtime_s",
                                      "exception", "src")}
        base.update(n=75, eval_git_head=head)
        if f["status"] != "OK":
            rows.append(base)
            continue
        dk = (f["rep"], f["mask"])
        if dk not in cache:
            cache = {dk: rw.weak_data(f["rep"], f["mask"])}
        D = dict(cache[dk], g=1)            # canonical truth scale (no z-scoring)
        z = np.load(Path(f["root"]) / f"{f['key']}.npz")
        st = {k: z[k] for k in z.files}
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
        if i % 20 == 0:
            print(f"{i}/{len(fits)} {f['key']} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    old = pd.read_csv(P160 / "criterion_scores.csv").set_index("key")
    rp = df[df.stage == "P160_80"].set_index("key")
    cc = []
    for c in ("A_CQ", "B_CQ_minus_PZ", "C_cond_last", "D_cond_mode", "E_CLap", "w_hat", "zzT_corr"):
        dlt = (rp[c] - old.loc[rp.index, c]).abs()
        both_nan = int((rp[c].isna() & old.loc[rp.index, c].isna()).sum())
        cc.append(dict(quantity=c, n_states=int(dlt.notna().sum()), n_both_nan=both_nan,
                       n_nan_mismatch=int((rp[c].isna() ^ old.loc[rp.index, c].isna()).sum()),
                       max_abs_diff=float(dlt.max()), pass_1e6=bool(dlt.max() < 1e-6)))
    cc = pd.DataFrame(cc)
    cc.to_csv(RES / "reuse_crosscheck.csv", index=False, lineterminator="\n")
    print(cc.to_string())
    df.to_csv(out, index=False, lineterminator="\n")
    print("rows", len(df))


if __name__ == "__main__":
    main()
