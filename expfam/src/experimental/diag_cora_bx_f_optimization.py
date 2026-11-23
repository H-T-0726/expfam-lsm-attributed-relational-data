"""Training-side audit of the Bernoulli-X F M-step (Cora strict held-out runner).

Technical diagnostic only (lineage E prototype).  It never computes, reads or
writes held-out scores: the EM is called directly (not through ``run_cell``)
and only X / Z_samples / train-masked Y enter.

Per M-step of the Bernoulli-X F update it records, on the *same* Z_samples:
  * Q_X before Adam (previous F), after Adam (F used by the runner), and at the
    exact per-column maximiser F* (damped Newton; Q_X is concave in each f_l),
  * Adam steps taken and stop reason (tol vs max_iter),
  * |grad| raw, per-row normalised, Newton decrement g'H^{-1}g/2 at F_adam,
  * standardised residual max |delta_jk| / sqrt((H_j^{-1})_kk) with delta = H^{-1}g,
  * separation checks (column has 0s and 1s, |eta| max, Newton convergence).

Paths:
  adam   : the runner's path, unchanged (Adam result is returned).  Must
           reproduce the smoke technical row (w0, Q_strict, F_grad_max).
  newton : F* replaces the Adam result (a candidate fix; not adopted).

Usage:
  python diag_cora_bx_f_optimization.py --out DIR [--jobs adam:6 adam:3 newton:6 newton:3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Q_X pieces (independent of the model code; canonical Bernoulli, no bias)
# ─────────────────────────────────────────────────────────────────────────────

def qx_cols(X, Zs, F):
    """Per-column (1/L) sum_l sum_i [x eta - log(1+e^eta)], eta = z_i^(l) f_j."""
    L = Zs.shape[2]
    q = np.zeros(F.shape[0])
    for l in range(L):
        eta = Zs[:, :, l] @ F.T
        q += (X * eta - np.logaddexp(0.0, eta)).sum(axis=0)
    return q / L


def grad_hess(X, Zs, F):
    """grad (d,k) and positive-definite negative Hessian H (d,k,k) of Q_X."""
    L = Zs.shape[2]
    d, k = F.shape
    g = np.zeros((d, k))
    H = np.zeros((d, k, k))
    for l in range(L):
        Z = Zs[:, :, l]
        eta = Z @ F.T
        p = 0.5 * (1.0 + np.tanh(0.5 * eta))
        g += (X - p).T @ Z
        H += np.einsum("ik,id,im->dkm", Z, p * (1.0 - p), Z, optimize=True)
    return g / L, H / L


def newton_F(X, Zs, F0, max_iter=100, tol_dec=1e-12):
    F = F0.copy()
    q = qx_cols(X, Zs, F)
    status = "max_iter"
    it = 0
    for it in range(1, max_iter + 1):
        g, H = grad_hess(X, Zs, F)
        delta = np.linalg.solve(H, g[:, :, None])[:, :, 0]
        dec = np.einsum("dk,dk->d", g, delta)          # g'H^{-1}g per column
        if np.max(dec) / 2.0 < tol_dec:
            status = "converged"
            break
        t = np.ones(F.shape[0])
        for _ in range(40):
            Fc = F + t[:, None] * delta
            qc = qx_cols(X, Zs, Fc)
            bad = qc < q + 1e-4 * t * dec
            if not bad.any():
                break
            t = np.where(bad, t / 2.0, t)
        F, q = Fc, qc
    g, H = grad_hess(X, Zs, F)
    return F, q, {"newton_iters": it, "newton_status": status,
                  "newton_grad_max": float(np.max(np.abs(g))),
                  "newton_minEigH": float(np.min(np.linalg.eigvalsh(H)))}


def adam_replica(model, X, Zs, F0, max_iter=50, lr=0.01, beta1=0.9,
                 beta2=0.999, eps=1e-8, tol=1e-6):
    """Line-by-line replica of DualExpFamLSM._calc_F_adam with stop reason."""
    L = Zs.shape[2]
    F = F0.copy()
    m = np.zeros_like(F)
    v = np.zeros_like(F)
    reason, last_step, t = "max_iter", float("nan"), 0
    for t in range(1, max_iter + 1):
        grad = np.zeros_like(F)
        for l in range(L):
            Z_l = Zs[:, :, l]
            grad += (X - model._mean_function_x(Z_l @ F.T)).T @ Z_l
        grad /= L
        g = -grad
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g ** 2
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        F_new = F - lr * m_hat / (np.sqrt(v_hat) + eps)
        last_step = float(np.max(np.abs(F_new - F)))
        if last_step < tol:
            F = F_new
            reason = "tol"
            break
        F = F_new
    return F, {"adam_steps": t, "adam_stop": reason, "adam_last_step": last_step}


def audit_mstep(model, X, Zs, F_prev, F_adam):
    n = X.shape[0]
    q0 = qx_cols(X, Zs, F_prev).sum()
    qa_c = qx_cols(X, Zs, F_adam)
    g, H = grad_hess(X, Zs, F_adam)
    Hinv = np.linalg.inv(H)
    delta = np.einsum("dkm,dm->dk", Hinv, g)
    dec = np.einsum("dk,dk->d", g, delta)
    se = np.sqrt(np.einsum("dkk->dk", Hinv))
    t0 = time.perf_counter()
    F_star, qs_c, ninfo = newton_F(X, Zs, F_adam)
    newton_sec = time.perf_counter() - t0
    eta_star = np.max(np.abs(np.einsum("ikl,dk->ild", Zs, F_star)))
    colsum = X.sum(axis=0)
    p_star_mean = []
    curv_mean = []
    for l in range(Zs.shape[2]):
        p = 0.5 * (1.0 + np.tanh(0.5 * (Zs[:, :, l] @ F_star.T)))
        p_star_mean.append(p.mean(axis=0))
        curv_mean.append((p * (1 - p)).mean(axis=0))
    qa = qa_c.sum()
    qs = qs_c.sum()
    return {
        "Q_X_before": q0, "Q_X_adam": qa, "Q_X_star": qs,
        "adam_gain": qa - q0, "remaining_gap": qs - qa,
        "gap_fraction_of_available": (qs - qa) / (qs - q0) if qs > q0 else 0.0,
        "gap_per_obs": (qs - qa) / X.size,
        "newton_decrement_half": float(dec.sum() / 2.0),
        "grad_max": float(np.max(np.abs(g))),
        "grad_max_per_row": float(np.max(np.abs(g)) / n),
        "grad_max_rel_colsum": float(np.max(np.abs(g) / np.maximum(colsum, 1)[:, None])),
        "std_residual_max": float(np.max(np.abs(delta) / se)),
        "std_residual_median": float(np.median(np.abs(delta) / se)),
        "F_adam_absmax": float(np.max(np.abs(F_adam))),
        "F_star_absmax": float(np.max(np.abs(F_star))),
        "F_diff_max": float(np.max(np.abs(F_star - F_adam))),
        "eta_star_absmax": float(eta_star),
        "cols_without_both_values": int(np.sum((colsum == 0) | (colsum == n))),
        "mean_curvature_min": float(np.min(np.mean(curv_mean, axis=0))),
        "mean_curvature_max": float(np.max(np.mean(curv_mean, axis=0))),
        "newton_seconds": newton_sec,
        **ninfo,
    }, F_star


def make_class(base, path):
    class FAudit(base):
        def _calc_F_adam(self, X, Z_samples, **kw):
            F_prev = self.params["F"].copy()
            F_adam = super()._calc_F_adam(X, Z_samples, **kw)
            F_rep, ainfo = adam_replica(self, X, Z_samples, F_prev)
            rec, F_star = audit_mstep(self, X, Z_samples, F_prev, F_adam)
            rec.update(ainfo)
            rec["adam_replica_maxdiff"] = float(np.max(np.abs(F_rep - F_adam)))
            if not hasattr(self, "f_audit"):
                self.f_audit = []
            rec["mstep"] = len(self.f_audit) + 1
            rec["path"] = path
            self.f_audit.append(rec)
            return F_star if path == "newton" else F_adam
    FAudit.__name__ = f"FAudit_{path}"
    return FAudit


def run_job(job):
    import cora_strict_heldout as csh
    import em_runner
    path, K = job["path"], job["K"]
    cfg = csh.CONFIG
    data = csh.load_cora()
    split = csh.build_split(data, cfg["smoke_split_index"], cfg, selection="full")
    init_seed, mc_seed = csh.seeds_for(split.split_index, K, 0, cfg)
    cls = make_class(csh.CoraConsistentFast, path)
    t0 = time.perf_counter()
    with csh.use_model_class(cls):
        res = em_runner.run_em_experimental(
            split.X_raw, split.Y_fit, "bernoulli", cfg["family_y"], k=K,
            L=cfg["L"], num_iter=cfg["num_iter"], seed=init_seed,
            train_mask=split.train_mask, fix_x=False,
            compute_strict_Q=True, mstep_q_diagnostic=True,
            numerics_mode=cfg["numerics_mode"], failure_policy=cfg["failure_policy"],
            init_seed=init_seed, mc_seed=mc_seed)
    sec = time.perf_counter() - t0
    model = res["model"]
    Zs = res["Z_samples"]
    summary = {
        "path": path, "K": K, "split_index": split.split_index,
        "init_seed": init_seed, "mc_seed": mc_seed, "em_seconds": round(sec, 1),
        "w0": res["w0"], "w": res["w"], "Q_strict": res["Q_strict"],
        "mstep_q_decreases": int(sum(h["decreased"] for h in res["mstep_q_history"])),
        "mstep_q_diffs": [h["q_diff"] for h in res["mstep_q_history"]],
        "Z_mean_norm": float(np.linalg.norm(Zs[:, :, -1].mean(axis=0))),
        "F_grad_max_final": float(np.max(np.abs(grad_hess(split.X_raw, Zs, res["F"])[0]))),
        "sampling_fallbacks": model.sampling_stats["fallbacks"],
        "retry_count": res["retry_count"],
    }
    gram = Zs.mean(axis=2) @ Zs.mean(axis=2).T
    iu = np.triu_indices(gram.shape[0], 1)
    return summary, model.f_audit, gram[iu].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", nargs="+", default=["adam:6", "adam:3", "newton:6", "newton:3"])
    ap.add_argument("--blas-threads", default="2")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = a.blas_threads
    jobs = [{"path": j.split(":")[0], "K": int(j.split(":")[1])} for j in a.jobs]
    import pandas as pd
    import cora_strict_heldout as csh
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=len(jobs), max_tasks_per_child=1) as ex:
        results = list(ex.map(run_job, jobs))
    rows, summaries, grams = [], [], {}
    for (s, recs, g) in results:
        summaries.append(s)
        for r in recs:
            rows.append({"K": s["K"], **r})
        grams[(s["path"], s["K"])] = g
    pd.DataFrame(rows).to_csv(out / "f_mstep_audit.csv", index=False)
    cmp = {}
    for K in sorted({k for _, k in grams}):
        if ("adam", K) in grams and ("newton", K) in grams:
            a_, b_ = grams[("adam", K)].astype(float), grams[("newton", K)].astype(float)
            cmp[f"K{K}"] = {"gram_mean5_corr_adam_vs_newton": float(np.corrcoef(a_, b_)[0, 1]),
                            "gram_mean5_maxabs_diff": float(np.max(np.abs(a_ - b_)))}
    info = {"summaries": summaries, "adam_vs_newton_training_gram": cmp,
            "wall_seconds": round(time.perf_counter() - t0, 1),
            "blas_threads": a.blas_threads, "environment": csh.environment_info(),
            "note": "training-side only; no held-out score computed or read"}
    (out / "f_audit_runinfo.json").write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps({"summaries": summaries, "cmp": cmp}, indent=1, default=str))


if __name__ == "__main__":
    main()
