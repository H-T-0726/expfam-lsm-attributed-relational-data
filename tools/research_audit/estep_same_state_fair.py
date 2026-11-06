"""Same parameter state, old vs new E-step formulas (deterministic, no fit).

For E1 states (estimator E, K = K_GEN, rep 1, start 1; both generators, K_GEN 5 and 9)
take theta and Z from the saved state and, for every node i with the other
nodes fixed, compute the conditional mode of each implementation's own
objective by Newton iterations (alpha = 1, 50 steps) and its Laplace
covariance A_i^-1:
  OLD = reproduction LatentStructuralModel (_calc_gradient / _calc_precision_matrix: 1/2 on Y)
  E   = lineage-E consistent model (no extra 1/2).
Reports the relative mode difference and the ratio of covariance traces.
Writes estep_same_state.csv.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "reproduction/src", ROOT / "expfam/src/experimental", ROOT / "tools/research_audit"):
    sys.path.insert(0, str(p))

import fair_protocol as fp  # noqa: E402
import run_fair_comparison as rf  # noqa: E402
from model import LatentStructuralModel  # noqa: E402
import em_runner  # noqa: E402


def node_mode(model, X, Y, Z, F, S, w0, w, i, steps=50):
    Zc = Z.copy()
    for _ in range(steps):
        g = model._calc_gradient(X, Y, Zc, F, S, 1.0, w0, w, i)        # = - d log f / d z_i
        A = model._calc_precision_matrix(Zc, F, S, 1.0, w0, w, i)
        A = 0.5 * (A + A.T)
        step = np.linalg.solve(A, g)
        Zc[i] = Zc[i] - step
        if np.linalg.norm(step) < 1e-10:
            break
    A = model._calc_precision_matrix(Zc, F, S, 1.0, w0, w, i)
    return Zc[i].copy(), np.linalg.inv(0.5 * (A + A.T))


def main():
    out = ROOT / fp.RESULTS_REL / "estep_same_state.csv"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    rows = []
    for g, gname in ((0, "Gold"), (1, "Gcanon")):
        for kg in (5, 9):
            D = rf.e1_data(g, kg, 1)
            key = f"e1_{gname}_K{kg}_r1_k{kg}_s1_E"
            z = np.load(rf.STATES / "e1" / f"{key}.npz")
            Z, F, S = z["Z"], z["F"], z["sigma"]
            w0, w, _ = (float(v) for v in z["scalars"])
            n, d = D["X"].shape
            mo = LatentStructuralModel(n=n, d=d, k=kg, L=1)
            mo.initialize_params(true_params=None, seed=0)
            me = em_runner.build_model(n, d, kg, 1, "gaussian", "bernoulli", train_mask=None, numerics_mode="consistent")
            me.initialize_params(true_params=None, seed=0)
            rel, tr = [], []
            for i in range(n):
                mo_i, Co = node_mode(mo, D["X"], D["Y"], Z, F, S, w0, w, i)
                me_i, Ce = node_mode(me, D["X"], D["Y"], Z, F, S, w0, w, i)
                rel.append(np.linalg.norm(mo_i - me_i) / max(np.linalg.norm(me_i), 1e-12))
                tr.append(np.trace(Co) / np.trace(Ce))
            rows.append({"state": key, "generator": gname, "K_GEN": kg, "w_hat": w,
                         "median_rel_mode_diff": float(np.median(rel)), "max_rel_mode_diff": float(np.max(rel)),
                         "median_cov_trace_ratio_old_over_E": float(np.median(tr)),
                         "min_cov_trace_ratio": float(np.min(tr)), "max_cov_trace_ratio": float(np.max(tr))})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
