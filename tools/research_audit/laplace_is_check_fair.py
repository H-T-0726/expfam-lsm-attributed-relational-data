"""Section 9: importance-sampling check of the Laplace approximation on EXISTING states (no fit).

Pre-registered in fair_protocol.LAPLACE_IS: canonical A1 n-sweep K_GEN=5,
n in {75, 150}, reps 1-8, K in {4, 5, 6}, start 1, theta and Z_hat as saved;
q = N(Z_hat, H^-1); 8 batches x 512 samples; batch seed
9950000 + 1000*n + 100*rep + 10*K + b.  log p(X,Y|theta) = laplace_log_observed + c_K.
The correction is NOT treated as the true marginal likelihood when weights
degenerate (relative ESS < 0.01).  Writes laplace_is_check.csv.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "expfam/src/experimental", ROOT / "tools/research_audit"):
    sys.path.insert(0, str(p))

import fair_protocol as fp  # noqa: E402
import run_clap_clean_nsweep as a1  # noqa: E402
import laplace_k_criterion as lk  # noqa: E402
import laplace_importance as li  # noqa: E402
import em_runner  # noqa: E402

hist = a1.hist
ART = Path(r"D:\tento\kennkyu_artifacts")


def main():
    out = ROOT / fp.RESULTS_REL / "laplace_is_check.csv"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    prov = a1.load_hist_provenance()
    emc, _ = a1.load_emc_identity()
    E = np.load(a1.EMC_NPZ)
    S = np.load(ART / "clap_clean_nsweep_20261006/full/states.npz")
    rows = []
    t0 = time.time()
    for n in (75, 150):
        for rep in range(1, 9):
            ds, train, _ = a1.make_cell(5, n, rep, prov[(5, n, rep)])
            for k in (4, 5, 6):
                key = f"{int(emc[(5, n, rep, k, 1)]['fit_index']):04d}_it008"
                m = em_runner.build_model(n, 15, k, 5, "poisson", "bernoulli", train_mask=train, numerics_mode="consistent")
                m.initialize_params(true_params=None, seed=0)
                w0, w, vz = (float(v) for v in E[f"{key}_scalars"])
                m.params.update({"Z": np.asarray(E[f"{key}_Z"], float).copy(), "F": np.asarray(E[f"{key}_F"], float),
                                 "sigma": np.asarray(E[f"{key}_sigma"], float), "w0": w0, "w": w, "var_z": vz})
                Zhat_saved = np.asarray(S[f"K5_n{n}_r{rep}_k{k}_s1_Zhat"], float)
                res = lk.laplace_log_observed(m, ds.X, ds.Y, np.asarray(E[f"{key}_Z"], float).copy(), grad_tol=1e-8, max_iter=200)
                row = {"n": n, "replicate": rep, "k": k, "lap_status": res.status,
                       "zhat_matches_saved": bool(res.ok and np.allclose(res.Z_hat, Zhat_saved, rtol=1e-8, atol=1e-8))}
                if res.ok:
                    H = lk.negative_hessian(m, ds.X, ds.Y, res.Z_hat)
                    H = 0.5 * (H + H.T)
                    seeds = [9950000 + 1000 * n + 100 * rep + 10 * k + b for b in range(1, 9)]
                    imp = li.importance_correction(m, ds.X, ds.Y, res.Z_hat, H, seeds, 512)
                    bs = imp["batch_correction_stats"]
                    pooled = imp["pooled"]
                    row.update(laplace_log_observed=res.log_observed, c_pooled=pooled["correction"],
                               c_batch_mean=bs["mean"], c_batch_sd=bs["sd"], c_batch_min=bs["min"], c_batch_max=bs["max"],
                               relative_ess=pooled["relative_ess"], max_weight_share=pooled["max_weight_share"],
                               non_finite=pooled["non_finite"], degenerate=bool(pooled["relative_ess"] < 0.01))
                rows.append(row)
                print(f"n{n} r{rep} k{k} {row.get('c_pooled')} ess={row.get('relative_ess')} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
