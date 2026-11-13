"""AF5 / S2-B: column-wise Laplace family selection with shared (anchor)
parameters.  EXPLORATORY (the 20 AF2 datasets were already analysed in
AF2-AF4).  NO MCEM fit is run (budget 0).

For a count column j, an anchor fit a (AF2 refit state) and a candidate
m in {poisson, cmp_nu2p0}:

    fixed   : theta_{-j}^(a) (other loadings, Gaussian variances, w0, w, var_z)
              and the anchor's families c_{-j}^(a) for every other column
    free    : the family m of column j, its loading f_j, the joint mode Z_hat
    Phi_m(Z, f_j) = log p(Z) + sum_l log p_{c_l}(x_l | Z f_l) + log p(Y_train | Z)
                    (AF4's independent complete log joint; base measures included)
    S_j(m; a) = max_{f_j} [ Phi_m(Z_hat_m(f_j), f_j) + (nK/2) log 2π
                            − ½ log|H_m(f_j)| ]              (PRIMARY objective)
    Z_hat_m(f) = argmax_Z Phi_m(Z, f),  H_m(f) = −∇²_Z Phi_m at Z_hat (full nK×nK)
    Δ_j = S_j(cmp2; a) − S_j(poisson; a)    (> 0 supports CMP2)

S_j is a Laplace approximation to the profile log marginal likelihood
max_f log p(X, Y_train | f, theta_{-j}^(a), c) with Z integrated out.  It is
NOT the Bayesian evidence (theta is not integrated) and NOT BIC.  Both
candidates have the same number of free parameters (K loadings).

SECONDARY "joint-MAP" variant (recorded, never used for the primary
decision): f_j maximises Phi_m(Z_hat(f), f) (envelope gradient
sum_i (x_ij − A'_m(η_ij)) z_i), and the Laplace term is added afterwards.

Usage
    python run_af5_local_laplace_selection.py --phase0-audit
    python run_af5_local_laplace_selection.py --write-protocol
    python run_af5_local_laplace_selection.py --pilot
    python run_af5_local_laplace_selection.py --execute
    python run_af5_local_laplace_selection.py --analyze --stage full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
for p in (HERE, HERE.parent, ROOT / "tools" / "research_audit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_af4_laplace_model_comparison as af4                         # noqa: E402
from run_auto_family_z_recovery import sha16, sha256_file, write_csv  # noqa: E402

af2 = af4.af2
EXPERIMENT = "AF5-S2B"
N, K, D = af4.N, af4.K, af4.D
TRUE_MODELS = list(af2.TRUE_MODELS)
COUNT_COLS = (3, 4, 5, 6, 7, 8)
CANDS = ("poisson", "cmp_nu2p0")
REGIMES = ("dense", "sparse")
ANCHORS = {"auto_first": "primary", "support_only": "secondary", "oracle": "diagnostic"}
LOG2PI = af4.LOG2PI

# numerical settings (fixed in the protocol before the pilot)
# Phase 0 attempt 1 used grad_tol 1e-10 as the OK criterion; it is below the
# floating-point floor of the damped Newton line search (one case stalled at
# 4.4e-9 after 200 iterations, and finite-difference points returned -inf).
# Change (before any Δ was computed; see numerical_audit.md): aim at 1e-11
# with the damped Newton, then polish with up to 5 pure Newton steps, and
# accept a mode when max|grad| <= 1e-8 (the AF4-validated criterion).
INNER_AIM_TOL, INNER_MAX_ITER, INNER_POLISH = 1e-11, 50, 5
INNER_GRAD_TOL = 1e-8
OUTER_FD_STEP = 1e-4
OUTER_GTOL, OUTER_MAXITER = 1e-5, 200
OUTER_CONV_TOL = 1e-3          # |central-FD grad of S|_inf at the optimum
START_AGREE_TOL = 1e-3          # |S(start A) − S(start B)|
FRESH_RESTART_TOL = 1e-6        # S at f* recomputed from the anchor Zmode
DELTA_NUMERIC_FLOOR = 1e-2
IS_BATCHES, IS_PER_BATCH, IS_SEED_BASE = 5, 400, 1667000
IS_READABLE_RELESS = 0.01
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 10000, 20261115

OUT_ROOT = ROOT / "expfam" / "results" / "auto_family_z_recovery" / "af5_local_laplace_selection_20261113"
PILOT_REPLICATES = (0, 1, 2)


# --------------------------------------------------------------------------
# the local problem
# --------------------------------------------------------------------------

class LocalProblem:
    """Anchor a, column j, candidate m: everything except (m, f_j, Z) fixed."""

    def __init__(self, ds, masks, regime: str, anchor: str, r: int, j: int, cand: str, fr):
        self.X = np.asarray(ds.X, float)
        self.Y = np.asarray(ds.Y, float)
        self.st = af4.load_state(r, regime, anchor)
        assign = fr[f"{af2.m1.replicate_label(r)}_{regime}_{anchor}"]["assignment"].split("|")
        self.anchor_assignment = list(assign)
        self.anchor_family_j = assign[j]
        assign = list(assign)
        assign[j] = cand
        self.model = af4.build_model(assign, masks[regime], self.st)
        self.F0 = np.asarray(self.st["F"], float).copy()
        self.j, self.cand = j, cand
        self.spec = self.model.column_specs[j]
        self.Z_anchor = np.asarray(self.st["Zmode"], float)
        self.Z_warm = self.Z_anchor.copy()
        self.n_evals = 0
        self.n_inner_newton = 0

    def set_f(self, f):
        F = self.F0.copy()
        F[self.j] = np.asarray(f, float)
        self.model.params["F"] = F

    def mode(self, f, Z0=None):
        import laplace_k_criterion as lk
        self.set_f(f)
        Z0 = self.Z_warm if Z0 is None else Z0
        Zh, it, g = lk.find_joint_mode(self.model, self.X, self.Y, Z0,
                                       grad_tol=INNER_AIM_TOL, max_iter=INNER_MAX_ITER)
        for _ in range(INNER_POLISH):              # pure Newton polish near the mode
            gr = lk.gradient(self.model, self.X, self.Y, Zh)
            H = lk.negative_hessian(self.model, self.X, self.Y, Zh)
            try:
                Zt = Zh + np.linalg.solve(0.5 * (H + H.T), gr.ravel()).reshape(Zh.shape)
                gt = float(np.max(np.abs(lk.gradient(self.model, self.X, self.Y, Zt))))
            except (np.linalg.LinAlgError, FloatingPointError):
                break
            if not gt < g:
                break
            Zh, g = Zt, gt
            it += 1
        self.n_inner_newton += it
        return Zh, it, g

    def laplace(self, f, Z0=None, keep=False) -> dict[str, Any]:
        import laplace_k_criterion as lk
        Zh, it, g = self.mode(f, Z0)
        self.n_evals += 1
        parts = af4.log_joint_parts(self.model, self.X, self.Y, Zh)
        H = lk.negative_hessian(self.model, self.X, self.Y, Zh)
        H = 0.5 * (H + H.T)
        sign, logdet = np.linalg.slogdet(H)
        ok = bool(g <= INNER_GRAD_TOL and sign > 0)
        if ok:
            self.Z_warm = Zh
        S = parts["phi"] + 0.5 * N * K * LOG2PI - 0.5 * logdet if ok else -math.inf
        out = {"S": float(S), "ok": ok, "inner_iter": it, "inner_grad": float(g),
               "logdet": float(logdet), "logdet_sign": float(sign), "parts": parts}
        if keep:
            out["Z_hat"], out["H"] = Zh, H
        return out

    def S(self, f) -> float:
        return self.laplace(f)["S"]

    def fd_grad_S(self, f, h=OUTER_FD_STEP) -> np.ndarray:
        g = np.zeros(K)
        for a in range(K):
            e = np.zeros(K)
            e[a] = h
            g[a] = (self.S(f + e) - self.S(f - e)) / (2 * h)
        return g

    def joint_phi_and_grad(self, f):
        """Profile log joint Phi(Z_hat(f), f) and its envelope gradient in f."""
        Zh, it, g = self.mode(f)
        if g > INNER_GRAD_TOL:
            return -math.inf, np.zeros(K)
        self.Z_warm = Zh
        phi = af4.log_joint_parts(self.model, self.X, self.Y, Zh)["phi"]
        eta = Zh @ np.asarray(f, float)
        resid = self.X[:, self.j] - self.spec.mean_statistic(eta)
        return float(phi), resid @ Zh


def plugin_start(prob: LocalProblem) -> np.ndarray:
    """Candidate-specific start: the Q-type optimum on the anchor Zmode (L=1)."""
    from family_selection_multisupport import score_candidate
    rec = score_candidate(prob.X[:, prob.j], prob.Z_anchor[:, :, None], prob.spec,
                          np.zeros(K), prob.j, route="trust_exact")
    if rec.status != "OK":
        raise FloatingPointError(f"plug-in start failed: {rec.failure_reason}")
    return np.asarray(rec.loading, float)


def optimise_S(prob: LocalProblem, f_start: np.ndarray) -> dict[str, Any]:
    prob.Z_warm = prob.Z_anchor.copy()
    t0 = time.perf_counter()

    def fun(f):
        s = prob.S(f)
        return -s if math.isfinite(s) else 1e300

    def jac(f):
        return -prob.fd_grad_S(f)

    res = minimize(fun, np.asarray(f_start, float), jac=jac, method="BFGS",
                   options={"gtol": OUTER_GTOL, "maxiter": OUTER_MAXITER})
    f = np.asarray(res.x, float)
    fin = prob.laplace(f, keep=True)
    gfd = prob.fd_grad_S(f)
    fresh = prob.laplace(f, Z0=prob.Z_anchor)
    return {"f": f, "S": fin["S"], "ok": fin["ok"], "final": fin,
            "outer_iter": int(res.nit), "scipy_success": bool(res.success), "scipy_status": int(res.status),
            "fd_grad_inf": float(np.max(np.abs(gfd))), "fresh_restart_diff": abs(fresh["S"] - fin["S"]),
            "seconds": round(time.perf_counter() - t0, 3)}


def optimise_joint_map(prob: LocalProblem, f_start: np.ndarray) -> dict[str, Any]:
    prob.Z_warm = prob.Z_anchor.copy()

    def fun(f):
        v, g = prob.joint_phi_and_grad(f)
        return (-v, -g) if math.isfinite(v) else (1e300, np.zeros(K))

    res = minimize(fun, np.asarray(f_start, float), jac=True, method="BFGS",
                   options={"gtol": 1e-6, "maxiter": OUTER_MAXITER})
    f = np.asarray(res.x, float)
    lap = prob.laplace(f)
    v, g = prob.joint_phi_and_grad(f)
    return {"f": f, "S_jointmap": lap["S"], "phi_jointmap": v, "grad_inf": float(np.max(np.abs(g))),
            "ok": lap["ok"]}


def is_crn(probs: dict[str, LocalProblem], results: dict[str, dict], seed: int) -> dict[str, Any]:
    """Importance-sampling correction for both candidates with COMMON random
    numbers eps (Δ_m = L_m^{-T} eps); per-batch paired differences."""
    from scipy.linalg import solve_triangular
    from scipy.special import logsumexp
    out = {}
    per_batch = {c: [] for c in probs}
    pooled = {c: [] for c in probs}
    rng = np.random.default_rng(seed)
    for b in range(IS_BATCHES):
        eps = rng.standard_normal((IS_PER_BATCH, N * K))
        for c, prob in probs.items():
            fin = results[c]["final"]
            prob.set_f(results[c]["f"])
            Zh, H = fin["Z_hat"], fin["H"]
            Lc = np.linalg.cholesky(H)
            dz = solve_triangular(Lc.T, eps.T, lower=False).T
            phi_hat = af4.log_joint_parts(prob.model, prob.X, prob.Y, Zh)["phi"]
            r = np.empty(IS_PER_BATCH)
            for s in range(IS_PER_BATCH):
                d = dz[s]
                try:
                    ph = af4.log_joint_parts(prob.model, prob.X, prob.Y, Zh + d.reshape(N, K))["phi"]
                except FloatingPointError:
                    ph = float("nan")
                r[s] = ph - phi_hat + 0.5 * float(d @ H @ d) if math.isfinite(ph) else -math.inf
            per_batch[c].append(float(logsumexp(r) - math.log(IS_PER_BATCH)))
            pooled[c].append(r)
    for c in probs:
        r = np.concatenate(pooled[c])
        lse = float(logsumexp(r))
        w = np.exp(r - lse)
        out[c] = {"correction": lse - math.log(r.size), "relative_ess": float(1.0 / np.sum(w ** 2)) / r.size,
                  "non_finite": int(np.sum(~np.isfinite(r))), "batch_sd": float(np.std(per_batch[c], ddof=1))}
    dc = np.array(per_batch["cmp_nu2p0"]) - np.array(per_batch["poisson"])
    out["delta_correction"] = out["cmp_nu2p0"]["correction"] - out["poisson"]["correction"]
    out["delta_correction_batch_se"] = float(np.std(dc, ddof=1) / math.sqrt(IS_BATCHES))
    out["readable"] = bool(min(out[c]["relative_ess"] for c in probs) >= IS_READABLE_RELESS)
    return out


# --------------------------------------------------------------------------
# one column x anchor
# --------------------------------------------------------------------------

def run_column(r, regime, anchor, j, ds, masks, fr, case_index) -> tuple[list, dict]:
    cand_rows, res = [], {}
    probs = {}
    base = {"replicate": af2.m1.replicate_label(r), "regime": regime, "anchor": anchor,
            "anchor_role": ANCHORS[anchor], "column": j, "true_model": TRUE_MODELS[j]}
    for c in CANDS:
        prob = LocalProblem(ds, masks, regime, anchor, r, j, c, fr)
        probs[c] = prob
        row = dict(base, candidate=c, anchor_family_j=prob.anchor_family_j)
        try:
            starts = {"anchor_loading": prob.F0[j].copy(), "plugin": plugin_start(prob)}
            runs = {k: optimise_S(prob, v) for k, v in starts.items()}
            okruns = {k: v for k, v in runs.items() if v["ok"]}
            if not okruns:
                raise FloatingPointError("no start reached an OK Laplace value")
            best = max(okruns, key=lambda k: okruns[k]["S"])
            b = okruns[best]
            res[c] = b
            jm = optimise_joint_map(prob, b["f"])
            p = b["final"]["parts"]
            row.update({
                "status": "OK", "S": b["S"], "best_start": best,
                "S_start_anchor": runs["anchor_loading"]["S"], "S_start_plugin": runs["plugin"]["S"],
                "start_disagreement": abs(runs["anchor_loading"]["S"] - runs["plugin"]["S"]),
                "f_star_norm": float(np.linalg.norm(b["f"])),
                "f_star": "|".join(f"{v:.10g}" for v in b["f"]),
                "f_start_diff": float(np.max(np.abs(runs["anchor_loading"]["f"] - runs["plugin"]["f"]))),
                "outer_iter": b["outer_iter"], "scipy_success": b["scipy_success"], "scipy_status": b["scipy_status"],
                "outer_fd_grad_inf": b["fd_grad_inf"], "fresh_restart_diff": b["fresh_restart_diff"],
                "inner_grad_inf": b["final"]["inner_grad"], "logdet": b["final"]["logdet"],
                "hess_min_eig": float(np.linalg.eigvalsh(b["final"]["H"])[0]),
                "phi": p["phi"], "phi_prior": p["prior"], "phi_x_gauss": p["x_gauss"],
                "phi_x_count_total": p["x_true_pois_cols"] + p["x_true_cmp_cols"],
                "phi_x_target_col": float(np.sum(prob.spec.log_prob(prob.X[:, j], b["final"]["Z_hat"] @ b["f"]))),
                "phi_y_train": p["y_train"], "neg_half_logdet": -0.5 * b["final"]["logdet"],
                "S_jointmap_variant": jm["S_jointmap"], "phi_jointmap_variant": jm["phi_jointmap"],
                "jointmap_grad_inf": jm["grad_inf"], "jointmap_f_norm": float(np.linalg.norm(jm["f"])),
                "n_laplace_evals": prob.n_evals, "n_inner_newton": prob.n_inner_newton,
                "seconds": round(sum(v["seconds"] for v in runs.values()), 2)})
        except Exception as exc:                      # recorded, never retried
            row.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"[:300]})
        cand_rows.append(row)
    col = dict(base, anchor_family_j=probs["poisson"].anchor_family_j)
    if all(c in res for c in CANDS):
        d = res["cmp_nu2p0"]["S"] - res["poisson"]["S"]
        num_unc = max(DELTA_NUMERIC_FLOOR,
                      max(r_["start_disagreement"] for r_ in cand_rows))
        conv_ok = all(r_["outer_fd_grad_inf"] <= OUTER_CONV_TOL and r_["fresh_restart_diff"] <= FRESH_RESTART_TOL
                      for r_ in cand_rows)
        try:
            isr = is_crn(probs, res, IS_SEED_BASE + case_index)
        except Exception as exc:
            isr = {"readable": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
        if not conv_ok:
            status = "UNDECIDED_NOT_CONVERGED"
        elif abs(d) <= num_unc:
            status = "UNDECIDED_NUMERIC"
        elif isr.get("readable") and ((np.sign(d + isr["delta_correction"]) != np.sign(d))
                                      or abs(d) < 2 * isr["delta_correction_batch_se"]):
            status = "UNDECIDED_LAPLACE"
        elif isr.get("readable"):
            status = "DECIDED"
        else:
            status = "DECIDED_LAPLACE_UNASSESSED"
        sel = "cmp_nu2p0" if d > 0 else "poisson"
        cp = {c: next(x for x in cand_rows if x["candidate"] == c) for c in CANDS}
        col.update({
            "status": status, "delta": d, "delta_per_obs": d / N, "laplace_selected": sel,
            "laplace_sign_correct": sel == TRUE_MODELS[j], "numeric_uncertainty": num_unc,
            "delta_phi": cp["cmp_nu2p0"]["phi"] - cp["poisson"]["phi"],
            "delta_phi_target_col": cp["cmp_nu2p0"]["phi_x_target_col"] - cp["poisson"]["phi_x_target_col"],
            "delta_phi_other": (cp["cmp_nu2p0"]["phi"] - cp["cmp_nu2p0"]["phi_x_target_col"])
                               - (cp["poisson"]["phi"] - cp["poisson"]["phi_x_target_col"]),
            "delta_neg_half_logdet": cp["cmp_nu2p0"]["neg_half_logdet"] - cp["poisson"]["neg_half_logdet"],
            "delta_jointmap_variant": cp["cmp_nu2p0"]["S_jointmap_variant"] - cp["poisson"]["S_jointmap_variant"],
            "is_readable": isr.get("readable", False),
            "is_delta_correction": isr.get("delta_correction", float("nan")),
            "is_delta_correction_se": isr.get("delta_correction_batch_se", float("nan")),
            "is_rel_ess_poisson": isr.get("poisson", {}).get("relative_ess", float("nan")),
            "is_rel_ess_cmp": isr.get("cmp_nu2p0", {}).get("relative_ess", float("nan")),
            "is_non_finite": sum(isr.get(c, {}).get("non_finite", 0) for c in CANDS),
            "delta_is_corrected": d + isr.get("delta_correction", float("nan")),
            "true_loading_norm": float(np.linalg.norm(np.asarray(ds.F, float)[j]))})
    else:
        col.update({"status": "FAILED"})
    return cand_rows, col


# --------------------------------------------------------------------------
# Phase 0 audit: inner gradient/Hessian FD at new configurations, base
# measures per candidate, reproducibility (no Δ is computed here)
# --------------------------------------------------------------------------

def phase0_audit(out: Path) -> dict[str, Any]:
    import laplace_k_criterion as lk
    from scipy.special import gammaln
    out.mkdir(parents=True, exist_ok=False)
    fr = {r["fit_key"]: r for r in af4.read_csv(af4.AF2_FULL / "fit_runs.csv")}
    rows = []
    for r in (0,):
        ds, masks = af2.generate(r), af2.m1.build_masks(r)
        for regime in REGIMES:
            for anchor in ANCHORS:
                for j in (4, 7):                     # one true-Poisson and one true-CMP2 column
                    for c in CANDS:
                        prob = LocalProblem(ds, masks, regime, anchor, r, j, c, fr)
                        f = plugin_start(prob)
                        Zh, it, g = prob.mode(f, prob.Z_anchor)
                        X, Y = prob.X, prob.Y
                        ga = lk.gradient(prob.model, X, Y, Zh)
                        gf = af4.fd_gradient(prob.model, X, Y, Zh)
                        Ha = lk.negative_hessian(prob.model, X, Y, Zh)
                        Ha = 0.5 * (Ha + Ha.T)
                        Hf = af4.fd_neg_hessian(prob.model, X, Y, Zh)
                        # base measure of the target column exactly as the candidate's
                        lp0 = prob.spec.log_prob(X[:, j], np.zeros(N))
                        nu = 2.0 if c == "cmp_nu2p0" else 1.0
                        base_ok = bool(np.allclose(lp0 + prob.spec.log_partition(np.zeros(N)),
                                                   -nu * gammaln(X[:, j] + 1)))
                        # the legacy offset is the same constant for both candidates
                        off = af4.phi_true(prob.model, X, Y, Zh) - lk.complete_log_density(prob.model, X, Y, Zh)
                        # envelope gradient of the joint-MAP profile vs FD
                        phi0, genv = prob.joint_phi_and_grad(f)
                        gfd = np.zeros(K)
                        for a in range(K):
                            e = np.zeros(K)
                            e[a] = 1e-5
                            gfd[a] = (prob.joint_phi_and_grad(f + e)[0] - prob.joint_phi_and_grad(f - e)[0]) / 2e-5
                        # S smoothness: FD gradient with two step sizes
                        g1 = prob.fd_grad_S(f, 1e-4)
                        g2 = prob.fd_grad_S(f, 1e-3)
                        # reproducibility
                        s1 = prob.laplace(f, Z0=prob.Z_anchor)["S"]
                        s2 = prob.laplace(f, Z0=prob.Z_anchor)["S"]
                        rows.append({
                            "replicate": af2.m1.replicate_label(r), "regime": regime, "anchor": anchor,
                            "column": j, "candidate": c, "inner_iter": it, "inner_grad_inf": g,
                            "grad_rel_err": float(np.max(np.abs(ga - gf)) / max(1.0, float(np.max(np.abs(gf))))),
                            "hess_rel_err": float(np.max(np.abs(Ha - Hf)) / float(np.max(np.abs(Hf)))),
                            "offdiag_norm": af4.offdiag_block_norm(Ha),
                            "min_eig": float(np.linalg.eigvalsh(Ha)[0]),
                            "target_base_measure_ok": base_ok,
                            "legacy_offset_minus_count_logfact": off - float(np.sum(gammaln(X[:, 3:] + 1))),
                            "envelope_grad_rel_err": float(np.max(np.abs(genv - gfd)) / max(1.0, float(np.max(np.abs(gfd))))),
                            "S_fd_grad_step_ratio_diff": float(np.max(np.abs(g1 - g2))),
                            "S_repeat_identical": s1 == s2})
                print(f"phase0 {regime} {anchor}", flush=True)
    write_csv(out / "phase0_numerical_audit.csv", rows)
    summ = {"cases": len(rows),
            "max_grad_rel_err": max(x["grad_rel_err"] for x in rows),
            "max_hess_rel_err": max(x["hess_rel_err"] for x in rows),
            "min_offdiag_norm": min(x["offdiag_norm"] for x in rows),
            "min_eig": min(x["min_eig"] for x in rows),
            "all_target_base_measure_ok": all(x["target_base_measure_ok"] for x in rows),
            "max_abs_legacy_offset_minus_count_logfact": max(abs(x["legacy_offset_minus_count_logfact"]) for x in rows),
            "max_envelope_grad_rel_err": max(x["envelope_grad_rel_err"] for x in rows),
            "max_S_fd_grad_step_diff": max(x["S_fd_grad_step_ratio_diff"] for x in rows),
            "all_repeat_identical": all(x["S_repeat_identical"] for x in rows),
            "git_head": af4.git("rev-parse", "HEAD")}
    summ["gate_pass"] = bool(summ["max_grad_rel_err"] <= 1e-5 and summ["max_hess_rel_err"] <= 1e-5
                             and summ["all_target_base_measure_ok"] and summ["min_eig"] > 0
                             and summ["max_abs_legacy_offset_minus_count_logfact"] <= 1e-8
                             and summ["max_envelope_grad_rel_err"] <= 1e-4 and summ["all_repeat_identical"])
    (out / "phase0_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    return summ


# --------------------------------------------------------------------------
# protocol, execution
# --------------------------------------------------------------------------

def protocol() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "study_type": "EXPLORATORY (algorithm development; the 20 AF2 datasets were already analysed in AF2-AF4, "
                      "so freezing this protocol does not make AF5 an independent confirmatory test)",
        "lineage": "E-consistent per-column + PR #129 generic model (experimental prototype; not adoptable). "
                   "PR #129 not merged; AF1/AF2 Draft PRs; AF3/AF4 unpushed",
        "mcem_fits": 0,
        "objective": {
            "primary": "S_j(m;a) = max_f [Phi_m(Z_hat_m(f), f) + (nK/2) log 2π − ½ log|H_m(f)|]; the Laplace-"
                       "corrected objective is what is maximised over f_j",
            "secondary_variant": "joint-MAP: f_j maximises Phi_m(Z_hat(f), f) (envelope gradient), Laplace term "
                                 "added afterwards; recorded only",
            "fixed": "theta_{-j} (other loadings, Gaussian variances, w0, w, var_z) and families of other columns, "
                     "from the anchor's saved AF2 refit state; identical for both candidates",
            "free": "family m of column j, loading f_j, Z (joint mode re-solved per candidate and per f_j)",
            "phi": "AF4 log_joint_parts (independent; base measures included: Poisson −log x!, CMP2 −2 log x!)",
            "hessian": "full nK×nK with Y cross blocks (laplace_k_criterion.negative_hessian)",
            "not": "not Bayesian evidence (theta not integrated), not BIC, not a Q value; equal parameter counts"},
        "anchors": {"auto_first": "PRIMARY (AF2 Auto refit, Poisson start; usable without the truth)",
                    "support_only": "SECONDARY", "oracle": "DIAGNOSTIC only, never in the primary improvement"},
        "unit_of_decision": "each count column independently; other columns keep the anchor's families; no sequential update",
        "delta": "Δ_j = S_j(cmp2) − S_j(poisson); > 0 supports CMP2",
        "optimisation": {
            "inner": f"find_joint_mode (damped full Newton) aiming at {INNER_AIM_TOL} (max_iter {INNER_MAX_ITER}), then up "
                     f"to {INNER_POLISH} pure Newton polish steps; accepted when max|grad| <= {INNER_GRAD_TOL}; warm "
                     "start from the previous accepted mode, first start = anchor Zmode",
            "change_log": "Phase 0 attempt 1 (kept in phase0_attempt1/) required 1e-10 and failed the gate on "
                          "floating-point stalls; changed before any Δ was computed",
            "outer": f"BFGS on −S with central finite-difference gradient (step {OUTER_FD_STEP}), gtol {OUTER_GTOL}, "
                     f"maxiter {OUTER_MAXITER}",
            "starts": "anchor's own loading for column j; candidate-specific plug-in (Q-type optimum on anchor Zmode, L=1)",
            "selection_of_start": "maximum S over OK starts"},
        "decision_status": {
            "FAILED": "computation failed for a candidate (recorded, not retried)",
            "UNDECIDED_NOT_CONVERGED": f"either candidate: |FD grad S|_inf > {OUTER_CONV_TOL} at the optimum, or S "
                                       f"recomputed from the anchor Zmode differs by > {FRESH_RESTART_TOL}",
            "UNDECIDED_NUMERIC": f"|Δ| <= max({DELTA_NUMERIC_FLOOR}, start disagreement of either candidate)",
            "UNDECIDED_LAPLACE": "IS readable (relative ESS >= 0.01 for both) and (sign(Δ + Δc) != sign(Δ) or "
                                 "|Δ| < 2 SE_batch(Δc)); Δc from common-random-number IS",
            "DECIDED_LAPLACE_UNASSESSED": "IS not readable; decided by the sign of Δ, flagged",
            "DECIDED": "otherwise"},
        "importance_sampling": {"proposal": "N(Z_hat_m, H_m^-1) per candidate with COMMON eps", "batches": IS_BATCHES,
                                "per_batch": IS_PER_BATCH, "seed": f"{IS_SEED_BASE} + case_index",
                                "caution": "with low ESS the correction is not a high-precision truth"},
        "data": {"datasets": "AF2, 20 (seed 1661001+r)", "regimes": list(REGIMES), "count_columns": list(COUNT_COLS),
                 "masks": "AF2 / M1 build_masks(r): Y train mask of the regime; test pairs never used"},
        "primary_metrics": [
            "Auto anchor: column accuracy by sign of Δ (all non-FAILED, non-UNDECIDED_NOT_CONVERGED/NUMERIC columns), "
            "by true family and regime; and 'confident' accuracy among DECIDED",
            "corrections: AF2 Auto (Poisson start) misselected columns that the Auto-anchor Laplace gets right",
            "harms: AF2 Auto correct columns that the Auto-anchor Laplace gets wrong",
            "dataset-level change in the number of correct count columns (Laplace − AF2 Auto), bootstrap CI",
            "Δ distribution and undecided rates"],
        "secondary_metrics": ["Support-only anchor (same metrics)", "agreement Auto vs Support-only anchors",
                              "dense vs sparse", "true loading norm", "decomposition (target column, other terms, "
                              "−½ log|H|)", "start / joint-MAP sensitivity",
                              "paired with the Q-score on the SAME anchor (AF3: Z7 Auto refit, Z5 Support-only, "
                              "Z3 Oracle samples, zero start)"],
        "diagnostic": "Oracle anchor accuracy and Δ",
        "statistical_unit": "dataset (20 per regime); columns within a dataset are not independent",
        "interval": f"percentile bootstrap over datasets ({BOOTSTRAP_RESAMPLES}, seed {BOOTSTRAP_SEED})",
        "pilot": {"replicates": [af2.m1.replicate_label(r) for r in PILOT_REPLICATES],
                  "purpose": "feasibility, time, memory, reproducibility, numerical error; settings are NOT tuned on pilot "
                             "accuracy; the pilot replicates are recomputed in the full run with the same settings",
                  "stop_rule": "stop before Phase 2 if > 10% of pilot columns are FAILED or UNDECIDED_NOT_CONVERGED, "
                               "or the Phase 0 gate fails"},
        "not_claimed": "Z recovery after re-fitting (no MCEM refit of a corrected assignment); a confirmed algorithm "
                       "(a new-seed confirmation study is required)",
        "budget": {"columns": 20 * 2 * len(ANCHORS) * len(COUNT_COLS),
                   "outer_optimisations": 20 * 2 * len(ANCHORS) * len(COUNT_COLS) * len(CANDS) * 2,
                   "is_draws": 20 * 2 * len(ANCHORS) * len(COUNT_COLS) * len(CANDS) * IS_BATCHES * IS_PER_BATCH},
        "artifacts": {"results": str(OUT_ROOT.relative_to(ROOT)).replace("\\", "/")},
    }


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True).encode()).hexdigest()


def runinfo(extra):
    import scipy
    req = ROOT / "requirements.txt"
    return {"experiment": EXPERIMENT, "script": "expfam/src/experimental/run_af5_local_laplace_selection.py",
            "git_head": af4.git("rev-parse", "HEAD"), "git_branch": af4.git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_tracked": bool(af4.git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "requirements_sha256": sha256_file(req) if req.exists() else "",
            "protocol_sha256": protocol_sha256(), "command": " ".join(sys.argv), **extra}


def execute(out: Path, replicates, resume: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=resume)
    fr = {r["fit_key"]: r for r in af4.read_csv(af4.AF2_FULL / "fit_runs.csv")}
    cand_path, col_path, prog = out / "candidate_scores.csv", out / "column_selection.csv", out / "progress.jsonl"
    done = set()
    cand_rows, col_rows = [], []
    if resume and col_path.exists():
        col_rows = af4.read_csv(col_path)
        cand_rows = af4.read_csv(cand_path)
        done = {(x["replicate"], x["regime"], x["anchor"], str(x["column"])) for x in col_rows}
    t0 = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        for r in replicates:
            ds, masks = af2.generate(r), af2.m1.build_masks(r)
            for ri, regime in enumerate(REGIMES):
                for ai, anchor in enumerate(ANCHORS):
                    for j in COUNT_COLS:
                        key = (af2.m1.replicate_label(r), regime, anchor, str(j))
                        if key in done:
                            continue
                        case_index = ((r * 2 + ri) * 3 + ai) * 10 + j
                        c, col = run_column(r, regime, anchor, j, ds, masks, fr, case_index)
                        cand_rows.extend(c)
                        col_rows.append(col)
                        write_csv(cand_path, cand_rows)
                        write_csv(col_path, col_rows)
                        with open(prog, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps({"key": list(key), "status": col.get("status"),
                                                 "t": round(time.perf_counter() - t0, 1)}) + "\n")
                print(f"{af2.m1.replicate_label(r)} {regime} done {round(time.perf_counter() - t0, 1)} s", flush=True)
        warn = sorted({f"{x.category.__name__}: {str(x.message)[:160]}" for x in wl})[:20]
        n_warn = len(wl)
    prev = json.loads((out / "runinfo.json").read_text(encoding="utf-8")) if (out / "runinfo.json").exists() else {}
    sessions = prev.get("sessions", []) + [{"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                                            "runtime_s": round(time.perf_counter() - t0, 1),
                                            "git_head": af4.git("rev-parse", "HEAD"), "n_warnings": n_warn,
                                            "warning_messages": warn}]
    (out / "runinfo.json").write_text(json.dumps(runinfo({
        "stage": out.name, "replicates": [af2.m1.replicate_label(r) for r in replicates], "columns": len(col_rows),
        "sessions": sessions, "af2_state_dir": str(af4.AF2_STATE), "mcem_fits": 0}), indent=2), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AF5 local Laplace family selection")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--phase0-audit", action="store_true")
    g.add_argument("--write-protocol", action="store_true")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--execute", action="store_true")
    g.add_argument("--analyze", action="store_true")
    g.add_argument("--posthoc", action="store_true", help="POST-HOC analyses (not pre-registered)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stage", default="full")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args(argv)
    out_root = Path(args.out_root)
    if args.phase0_audit:
        print(json.dumps(phase0_audit(out_root / "phase0"), indent=2))
        return 0
    if args.write_protocol:
        p = out_root / "protocol.json"
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
        p.write_text(json.dumps(protocol(), indent=2), encoding="utf-8")
        write_csv(out_root / "manifest.csv", [
            {"replicate": af2.m1.replicate_label(r), "regime": reg, "anchor": a, "anchor_role": ANCHORS[a],
             "column": j, "true_model": TRUE_MODELS[j], "candidates": "|".join(CANDS),
             "af2_state": f"{af2.m1.replicate_label(r)}_{reg}_{a}.npz", "pilot": r in PILOT_REPLICATES}
            for r in range(20) for reg in REGIMES for a in ANCHORS for j in COUNT_COLS])
        print(f"wrote {p} sha256(canonical)={protocol_sha256()}")
        return 0
    if args.analyze:
        import analyze_af5_local_laplace_selection as an
        an.main(out_root / args.stage)
        return 0
    if args.posthoc:
        import analyze_af5_local_laplace_selection as an
        an.posthoc(out_root / args.stage, out_root / "pilot")
        return 0
    frozen = json.loads((out_root / "protocol.json").read_text(encoding="utf-8"))
    if hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest() != protocol_sha256():
        raise SystemExit("protocol.json on disk differs from protocol(); refusing to run")
    p0 = json.loads((out_root / "phase0" / "phase0_summary.json").read_text(encoding="utf-8"))
    if not p0["gate_pass"]:
        raise SystemExit("Phase 0 gate did not pass")
    if args.pilot:
        execute(out_root / "pilot", list(PILOT_REPLICATES), resume=args.resume)
    else:
        execute(out_root / "full", list(range(20)), resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
