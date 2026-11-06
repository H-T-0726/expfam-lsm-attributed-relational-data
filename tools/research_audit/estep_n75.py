"""Phase F (n75_protocol.ESTEP): same theta, Z, X, Y; OLD vs lineage-E E-step, n=75 and n=150.  No fit.

Per state (lineage-E fitted state at K = 5, start 1) and per node i (others fixed):
  target_OLD(z) = log N(x_i; F z, Sigma) + 1/2 sum_{j!=i} log Bern(y_ij; sigmoid(w0 + w z'z_j)) - ||z||^2/(2 var_z)
  target_E(z)   = same without the 1/2
  * finite-difference check of each implementation's gradient / precision against ITS OWN target
    (first 15 nodes; central differences, h = 1e-5)
  * conditional mode (Newton alpha 1, <= 50 steps) and Laplace covariance A_i^{-1} of each implementation
  * precision trace split: prior, X part tr(F' Sigma^-1 F), Y part tr(w^2 sum_j s(1-s) z_j z_j') (OLD uses 1/2 of it)
Writes estep_same_state.csv (one row per state, medians over nodes) and estep_nodes.csv.  Refuses to overwrite.
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

import n75_protocol as npc  # noqa: E402
import run_n75 as rn  # noqa: E402
import run_fair_comparison as rf  # noqa: E402
from model import LatentStructuralModel  # noqa: E402
import em_runner  # noqa: E402
from estep_same_state_fair import node_mode  # noqa: E402

RES = ROOT / npc.RESULTS_REL


def target(z, i, X, Y, Z, F, sd, w0, w, var_z, yfac):
    r = X[i] - F @ z
    lx = -0.5 * np.sum(r ** 2 / sd) - 0.5 * np.sum(np.log(2 * np.pi * sd))
    eta = w0 + w * (Z @ z)
    ll = Y[i] * eta - np.logaddexp(0, eta)
    ll[i] = 0.0
    return lx + yfac * ll.sum() - 0.5 * z @ z / var_z


def fd_grad_hess(fun, z, h=1e-5):
    k = len(z)
    g = np.zeros(k)
    H = np.zeros((k, k))
    for a in range(k):
        e = np.zeros(k)
        e[a] = h
        g[a] = (fun(z + e) - fun(z - e)) / (2 * h)
        for b in range(k):
            e2 = np.zeros(k)
            e2[b] = h
            H[a, b] = (fun(z + e + e2) - fun(z + e - e2) - fun(z - e + e2) + fun(z - e - e2)) / (4 * h * h)
    return g, H


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def states():
    out = []
    for rep in (1, 2, 3, 4):
        out.append(("B", f"B_r{rep}_k5_s1_E", 75, "G_canon", lambda r=rep: rn.stage_data("B", r), Path(npc.STATE_ROOT) / "B"))
    for rep in (1, 2):
        out.append(("C", f"C_r{rep}_k5_s1_E", 75, "G_old", lambda r=rep: rn.stage_data("C", r), Path(npc.STATE_ROOT) / "C"))
    for g, gname in ((1, "Gcanon"), (0, "Gold")):
        for rep in (1, 2):
            out.append(("P160", f"e1_{gname}_K5_r{rep}_k5_s1_E", 150, "G_canon" if g else "G_old",
                        lambda g=g, r=rep: rf.e1_data(g, 5, r), Path(npc.PR160_STATES) / "e1"))
    return out


def main():
    out = RES / "estep_same_state.csv"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    rows, nodes = [], []
    for stage, key, n, gen, dfun, root in states():
        D = dfun()
        z = np.load(root / f"{key}.npz")
        Z, F, S = z["Z"], z["F"], np.asarray(z["sigma"], float)
        sd = np.diag(S) if S.ndim == 2 else S
        Smat = np.diag(sd)
        w0, w, vz = (float(v) for v in z["scalars"])
        X, Y = D["X"], D["Y"]
        k = Z.shape[1]
        mo = LatentStructuralModel(n=n, d=X.shape[1], k=k, L=1)
        mo.initialize_params(true_params=None, seed=0)
        me = em_runner.build_model(n, X.shape[1], k, 1, "gaussian", "bernoulli", train_mask=None, numerics_mode="consistent")
        me.initialize_params(true_params=None, seed=0)
        for i in range(n):
            r = dict(stage=stage, state=key, n=n, generator=gen, node=i)
            if i < 15:
                for name, mdl, fac in (("OLD", mo, 0.5), ("E", me, 1.0)):
                    fun = lambda zz: target(zz, i, X, Y, Z, F, sd, w0, w, vz, fac)  # noqa: E731
                    g_fd, H_fd = fd_grad_hess(fun, Z[i].copy())
                    g_impl = -mdl._calc_gradient(X, Y, Z, F, Smat, vz, w0, w, i)
                    A_impl = mdl._calc_precision_matrix(Z, F, Smat, vz, w0, w, i)
                    r[f"{name}_grad_rel_err_own_target"] = rel(g_impl, g_fd)
                    r[f"{name}_prec_rel_err_own_target"] = rel(A_impl, -H_fd)
                # OLD implementation against the FULL (untempered) target
                fun = lambda zz: target(zz, i, X, Y, Z, F, sd, w0, w, vz, 1.0)  # noqa: E731
                g_fd, H_fd = fd_grad_hess(fun, Z[i].copy())
                r["OLD_grad_rel_err_full_target"] = rel(-mo._calc_gradient(X, Y, Z, F, Smat, vz, w0, w, i), g_fd)
                r["OLD_prec_rel_err_full_target"] = rel(mo._calc_precision_matrix(Z, F, Smat, vz, w0, w, i), -H_fd)
            mo_i, Co = node_mode(mo, X, Y, Z, F, Smat, w0, w, i)
            me_i, Ce = node_mode(me, X, Y, Z, F, Smat, w0, w, i)
            r["rel_mode_diff"] = rel(mo_i, me_i)
            r["cov_trace_ratio_old_over_E"] = float(np.trace(Co) / np.trace(Ce))
            eta = w0 + w * (Z @ Z[i])
            s = 1 / (1 + np.exp(-eta))
            q = s * (1 - s)
            q[i] = 0.0
            r["prec_trace_prior"] = k / vz
            r["prec_trace_X"] = float(np.trace(F.T @ np.diag(1 / sd) @ F))
            r["prec_trace_Y_full"] = float(w ** 2 * np.sum(q * np.sum(Z ** 2, axis=1)))
            r["target_logdens_gap_full_minus_old_at_mode_E"] = float(
                target(me_i, i, X, Y, Z, F, sd, w0, w, vz, 1.0) - target(me_i, i, X, Y, Z, F, sd, w0, w, vz, 0.5))
            nodes.append(r)
        nd = pd.DataFrame([x for x in nodes if x["state"] == key])
        rows.append(dict(stage=stage, state=key, n=n, generator=gen, w_hat=w, w0_hat=w0,
                         median_rel_mode_diff=float(nd.rel_mode_diff.median()), max_rel_mode_diff=float(nd.rel_mode_diff.max()),
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
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    pd.DataFrame(nodes).to_csv(RES / "estep_nodes.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
