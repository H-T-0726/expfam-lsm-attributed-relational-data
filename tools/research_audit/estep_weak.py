"""weak_protocol.ESTEP: same theta, Z, X, Y; OLD vs lineage-E E-step on the weak n=75 data, all pairs.  No fit.

States: K = 5, start 1, all pairs, rep 1-4, the lineage-E state AND the OLD state (each implementation's
E-step is evaluated at both).  Per node i (others fixed), with the functions of estep_n75 (unchanged):
  target_OLD(z) = log N(x_i; F z, Sigma) + 1/2 sum_{j!=i} log Bern(y_ij; sigmoid(w0 + w z'z_j)) - ||z||^2/(2 var_z)
  target_E(z)   = same without the 1/2
  * finite-difference check of each implementation's gradient / precision against ITS OWN target (first 15 nodes)
  * conditional mode and Laplace covariance of each implementation; precision trace split (prior, X, Y)
Writes estep_same_state.csv and estep_nodes.csv.  Refuses to overwrite.
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

import weak_protocol as wp  # noqa: E402
import run_weak as rw  # noqa: E402
from estep_n75 import target, fd_grad_hess, rel  # noqa: E402
from estep_same_state_fair import node_mode  # noqa: E402
from model import LatentStructuralModel  # noqa: E402
import em_runner  # noqa: E402

RES = ROOT / wp.RESULTS_REL


def main():
    out = RES / "estep_same_state.csv"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    rows, nodes = [], []
    for rep in wp.COMMON["replicates"]:
        D = rw.weak_data(rep)
        X, Y = D["X"], D["Y"]
        n = X.shape[0]
        for fitted_by in ("E", "OLD"):
            key = f"W_r{rep}_k5_s1_{fitted_by}"
            z = np.load(Path(wp.STATE_ROOT) / "W" / f"{key}.npz")
            Z, F, S = z["Z"], z["F"], np.asarray(z["sigma"], float)
            sd = np.diag(S) if S.ndim == 2 else S
            Smat = np.diag(sd)
            w0, w, vz = (float(v) for v in z["scalars"])
            k = Z.shape[1]
            mo = LatentStructuralModel(n=n, d=X.shape[1], k=k, L=1)
            mo.initialize_params(true_params=None, seed=0)
            me = em_runner.build_model(n, X.shape[1], k, 1, "gaussian", "bernoulli", train_mask=None, numerics_mode="consistent")
            me.initialize_params(true_params=None, seed=0)
            cur = []
            for i in range(n):
                r = dict(state=key, rep=rep, state_fitted_by=fitted_by, n=n, node=i)
                if i < 15:
                    for name, mdl, fac in (("OLD", mo, 0.5), ("E", me, 1.0)):
                        fun = lambda zz: target(zz, i, X, Y, Z, F, sd, w0, w, vz, fac)  # noqa: E731
                        g_fd, H_fd = fd_grad_hess(fun, Z[i].copy())
                        r[f"{name}_grad_rel_err_own_target"] = rel(-mdl._calc_gradient(X, Y, Z, F, Smat, vz, w0, w, i), g_fd)
                        r[f"{name}_prec_rel_err_own_target"] = rel(mdl._calc_precision_matrix(Z, F, Smat, vz, w0, w, i), -H_fd)
                    fun = lambda zz: target(zz, i, X, Y, Z, F, sd, w0, w, vz, 1.0)  # noqa: E731
                    g_fd, H_fd = fd_grad_hess(fun, Z[i].copy())
                    r["OLD_grad_rel_err_full_target"] = rel(-mo._calc_gradient(X, Y, Z, F, Smat, vz, w0, w, i), g_fd)
                    r["OLD_prec_rel_err_full_target"] = rel(mo._calc_precision_matrix(Z, F, Smat, vz, w0, w, i), -H_fd)
                mo_i, Co = node_mode(mo, X, Y, Z, F, Smat, w0, w, i)
                me_i, Ce = node_mode(me, X, Y, Z, F, Smat, w0, w, i)
                r["rel_mode_diff"] = rel(mo_i, me_i)
                r["mode_norm_OLD"] = float(np.linalg.norm(mo_i))
                r["mode_norm_E"] = float(np.linalg.norm(me_i))
                r["cov_trace_OLD"] = float(np.trace(Co))
                r["cov_trace_E"] = float(np.trace(Ce))
                r["cov_trace_ratio_old_over_E"] = float(np.trace(Co) / np.trace(Ce))
                eta = w0 + w * (Z @ Z[i])
                s = 1 / (1 + np.exp(-eta))
                q = s * (1 - s)
                q[i] = 0.0
                r["prec_trace_prior"] = k / vz
                r["prec_trace_X"] = float(np.trace(F.T @ np.diag(1 / sd) @ F))
                r["prec_trace_Y_full"] = float(w ** 2 * np.sum(q * np.sum(Z ** 2, axis=1)))
                nodes.append(r)
                cur.append(r)
            nd = pd.DataFrame(cur)
            rows.append(dict(state=key, rep=rep, state_fitted_by=fitted_by, n=n, w_hat=w, w0_hat=w0, w_true=float(D["w"]),
                             median_rel_mode_diff=float(nd.rel_mode_diff.median()), max_rel_mode_diff=float(nd.rel_mode_diff.max()),
                             median_mode_norm_ratio_old_over_E=float((nd.mode_norm_OLD / nd.mode_norm_E).median()),
                             median_cov_trace_ratio=float(nd.cov_trace_ratio_old_over_E.median()),
                             min_cov_trace_ratio=float(nd.cov_trace_ratio_old_over_E.min()),
                             max_cov_trace_ratio=float(nd.cov_trace_ratio_old_over_E.max()),
                             median_prec_trace_X=float(nd.prec_trace_X.median()), median_prec_trace_Y_full=float(nd.prec_trace_Y_full.median()),
                             median_Y_share_of_precision_E=float((nd.prec_trace_Y_full / (nd.prec_trace_Y_full + nd.prec_trace_X + nd.prec_trace_prior)).median()),
                             max_OLD_grad_rel_err_own=float(nd.OLD_grad_rel_err_own_target.max()),
                             max_OLD_prec_rel_err_own=float(nd.OLD_prec_rel_err_own_target.max()),
                             max_E_grad_rel_err_own=float(nd.E_grad_rel_err_own_target.max()),
                             max_E_prec_rel_err_own=float(nd.E_prec_rel_err_own_target.max()),
                             median_OLD_grad_rel_err_full=float(nd.OLD_grad_rel_err_full_target.median()),
                             median_OLD_prec_rel_err_full=float(nd.OLD_prec_rel_err_full_target.median())))
            print({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    pd.DataFrame(nodes).to_csv(RES / "estep_nodes.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
