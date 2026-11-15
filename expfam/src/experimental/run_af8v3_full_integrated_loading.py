"""AF8-v3 final (exploratory): audited fixes to the v3 numerics, re-pilot, and the 40-dataset exploratory run.

Parent: AF8-v3 (code/protocol e183c45, pilot results c1f37a9; all v3 stop rules passed).  The independent audit of the
v3 pilot raised (W2) a code path in which the outer certificate returned CERT_OK although T at the extra Newton step
was invalid (gain NaN), and (W3) NaN-skipping aggregation in the analysis.  This module

  * replaces the outer certificate (certify_outer) by one that never certifies on a non-finite / invalid evaluation,
    re-validates the current point independently in every round (T finite, FD gradient and Hessian finite, Cholesky
    PD), records a decrement failure explicitly, and records the certificate of EVERY start (not only the best);
  * keeps all other v3 numerics unchanged (inner/joint certificate, IS with logged same-mode rescue, outer checks);
  * writes to a NEW directory (af8v3_full_exploratory/); AF8, AF8-v2 and AF8-v3 artifacts are not touched.

Everything else (model, theta_{-j}, data, priors, starts, FD steps, IS draws/seeds, E2 as main score) is AF8-v3.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_af8_integrated_loading as a8          # noqa: E402  (math, data loading; unchanged)
import run_af8v2_integrated_loading as v2        # noqa: E402  (shared helpers; unchanged)
import run_af8v3_integrated_loading as v3        # noqa: E402  (certified inner/joint Newton, IS; unchanged)

OUT = a8.RES / "af8v3_full_exploratory"
V3_CODE_HEAD = "e183c4505532c8607456dc2bbe1e300ba6a13ac2"
V3_PILOT_HEAD = "c1f37a9"
K = a8.K
LOG2PI = a8.LOG2PI
P, C = a8.P, a8.C
CANDS, REGIMES, COUNT_COLS = a8.CANDS, a8.REGIMES, a8.COUNT_COLS
PRIORS, MAIN_PRIOR = a8.PRIORS, a8.MAIN_PRIOR
EvalFail, chol_logdet = v2.EvalFail, v2.chol_logdet
OUTER_CERT_DEC, OUTER_CERT_MAXITER = v3.OUTER_CERT_DEC, v3.OUTER_CERT_MAXITER
FD_H_HESS_ALT = v3.FD_H_HESS_ALT
IS_SEED_BASE = v3.IS_SEED_BASE
PILOT_REPS = v3.PILOT_REPS
FULL_REPS = tuple(range(40))
N_DATASETS = 40

# extra-step check of the outer certificate: the full Newton step first; if T is not defined there (inner Z problem
# not certified), the same direction is checked at 1/2 and 1/4 (the local quadratic model predicts a gain <= the
# decrement for every fraction in (0, 1]).  If no fraction gives a finite T the start is NOT certified.
EXTRA_STEP_FRACTIONS = (1.0, 0.5, 0.25)

# stop rules: the v3 rules unchanged + S4v2 (the v2-definition S4 as a rule, audit W1; stricter, fixed BEFORE the
# re-pilot) + S0 (completeness: every expected cell present, no statistic computed on a silently reduced set).
STOP_RULES = dict(v3.STOP_RULES_V3)
STOP_RULES["S4v2_IS_cells_with_any_cold_start_failure_share_max"] = 0.05
STOP_RULES["S0_all_expected_cells_present"] = True


# --------------------------------------------------------------------------
# outer certificate (fix of audit W2)
# --------------------------------------------------------------------------

def _finite(*xs) -> bool:
    return all(np.all(np.isfinite(np.asarray(x, float))) for x in xs)


def certify_outer(L: "v3.Local3", f0) -> dict:
    """outer Newton certificate.  CERT_OK only if, at the returned f:
         T(f) valid and finite; FD gradient and FD Hessian finite; -d2T/df2 PD (Cholesky);
         lambda_f^2/2 finite and <= OUTER_CERT_DEC; and T at f + a H^-1 g is finite for some a in
         EXTRA_STEP_FRACTIONS with gain T(f + a p) - T(f) finite and <= OUTER_CERT_DEC.
       Every other outcome is a named failure; a NaN never passes a comparison."""
    f = np.asarray(f0, float).copy()
    ok, t = L.T(f)
    if not (ok and np.isfinite(t)):
        return {"status": "START_INVALID", "f": f, "T": float("nan"), "moves": 0}
    moves = 0
    last: dict[str, Any] = {}
    for _ in range(OUTER_CERT_MAXITER):
        ok, t = L.T(f)                                   # independent re-validation of the current point
        if not (ok and np.isfinite(t)):
            return {"status": "POINT_INVALID", "f": f, "T": float("nan"), "moves": moves}
        try:
            g = v2.fd_grad(L, f)
            Hm = v2.fd_hess(L, f)
        except EvalFail as exc:
            return {"status": "FD_FAIL", "f": f, "T": t, "message": str(exc), "moves": moves}
        if not _finite(g, Hm):
            return {"status": "FD_NONFINITE", "f": f, "T": t, "moves": moves}
        Hs = 0.5 * (Hm + Hm.T)
        pd_ok, _, _ = chol_logdet(Hs)
        ev = np.linalg.eigvalsh(Hs)
        if not (pd_ok and ev[0] > 0):
            return {"status": "T_STATIONARY_NOT_MAX", "f": f, "T": t, "min_eig_f": float(ev[0]), "moves": moves}
        p = np.linalg.solve(Hs, g)
        dec = 0.5 * float(g @ p)
        if not (np.isfinite(dec) and _finite(p)):
            return {"status": "DECREMENT_FAIL", "f": f, "T": t, "min_eig_f": float(ev[0]), "moves": moves}
        last = {"lam2_half_f": dec, "min_eig_f": float(ev[0])}
        if dec <= OUTER_CERT_DEC:
            frac, gain, n_inv = None, float("nan"), 0
            for a in EXTRA_STEP_FRACTIONS:
                okp, tp = L.T(f + a * p)
                if okp and np.isfinite(tp):
                    frac, gain = a, float(tp - t)
                    break
                n_inv += 1
            if frac is None:
                return {"status": "EXTRA_STEP_INVALID", "f": f, "T": t, "moves": moves, "extra_step_invalid": n_inv, **last}
            if not np.isfinite(gain):
                return {"status": "EXTRA_STEP_NONFINITE", "f": f, "T": t, "moves": moves, **last}
            if gain > OUTER_CERT_DEC:                 # the quadratic model under-states the gap: move, re-certify
                f, t, moves = f + frac * p, t + gain, moves + 1
                continue
            return {"status": "CERT_OK", "f": f, "T": t, "hess_f": Hm, "min_eig_f": float(ev[0]),
                    "cond_f": float(ev[-1] / ev[0]), "lam2_half_f": dec, "newton_gain_f": gain,
                    "extra_step_fraction": frac, "extra_step_invalid": n_inv,
                    "grad_inf": float(np.max(np.abs(g))), "moves": moves}
        nrm = float(np.linalg.norm(p))
        if nrm > v3.OUTER_STEP_CAP:
            p = p * (v3.OUTER_STEP_CAP / nrm)
        slope = float(g @ p)
        a, acc = 1.0, False
        while a >= 1e-6:
            okt, tt = L.T(f + a * p)
            if okt and np.isfinite(tt) and tt >= t + v3.ARMIJO_C * a * slope:
                f, t, acc, moves = f + a * p, tt, True, moves + 1
                break
            a *= 0.5
        if not acc:
            return {"status": "OUTER_NOT_CERTIFIED", "f": f, "T": t, "moves": moves, **last}
    return {"status": "OUTER_NOT_CERTIFIED", "f": f, "T": t, "moves": moves, **last}


def start_valid(c: dict) -> bool:
    """a start counts as a valid local maximum only with a complete, finite certificate."""
    return (c.get("status") == "CERT_OK" and np.isfinite(c.get("T", np.nan))
            and np.isfinite(c.get("lam2_half_f", np.nan)) and np.isfinite(c.get("newton_gain_f", np.nan))
            and c.get("hess_f") is not None and _finite(c["hess_f"]))


def method_D_E2(L: "v3.Local3", starts: dict) -> dict:
    runs, log = {}, {}
    for name, f0 in starts.items():
        b = v2.maximise_T(L, f0)                       # unchanged BFGS phase (its status is recorded only)
        if b["status"] in ("START_INVALID", "FD_FAIL"):
            runs[name] = {"status": "BFGS_" + b["status"], "valid": False, "bfgs_status": b["status"]}
        else:
            c = certify_outer(L, b["f"])
            c["bfgs_status"] = b["status"]
            c["valid"] = start_valid(c)
            if c["status"] == "CERT_OK" and not c["valid"]:
                c["status"] = "CERT_INCOMPLETE"
            runs[name] = c
        r = runs[name]
        log[name] = {k: (float(r[k]) if isinstance(r.get(k), (int, float, np.floating)) else r.get(k))
                     for k in ("status", "bfgs_status", "T", "lam2_half_f", "newton_gain_f", "extra_step_fraction",
                               "extra_step_invalid", "min_eig_f", "grad_inf", "moves") if k in r}
        if r.get("f") is not None:
            log[name]["f"] = "|".join(repr(float(a)) for a in r["f"])
    cl = v2.classify_starts(runs, "T", "f")
    out = {"D_" + k: v for k, v in cl.items()}
    out["D_start_status"] = "|".join(f"{k}:{r['status']}" for k, r in runs.items())
    out["D_bfgs_status"] = "|".join(f"{k}:{r.get('bfgs_status')}" for k, r in runs.items())
    out["D_start_log"] = json.dumps(log)
    if cl["best"] is None:
        out["E2_status"] = "NO_VALID_LOCAL_MAX"
        return out
    b = runs[cl["best"]]
    f, Hm = b["f"], b["hess_f"]
    s = L.S(f)
    ld = float(np.linalg.slogdet(Hm)[1])
    out.update({"D": b["T"], "f_D": f, "Z_D": s.Z, "D_S_at_fD": s.S, "D_logprior_at_fD": a8.log_prior(f, L.J.pv),
                "D_grad_inf": b["grad_inf"], "D_min_eig_f": b["min_eig_f"], "D_cond_f": b["cond_f"],
                "D_lam2_half_f": b["lam2_half_f"], "D_newton_gain_f": b["newton_gain_f"],
                "D_extra_step_fraction": b["extra_step_fraction"], "D_cert_moves": b["moves"],
                "E2": b["T"] + 0.5 * K * LOG2PI - 0.5 * ld, "E2_logdet_f": ld, "hess_f": Hm})
    for h in FD_H_HESS_ALT:
        try:
            ev = np.linalg.eigvalsh(v2.fd_hess(L, f, h))
            out[f"E2_h{h:g}"] = b["T"] + 0.5 * K * LOG2PI - 0.5 * float(np.sum(np.log(ev))) if ev[0] > 0 else float("nan")
        except EvalFail:
            out[f"E2_h{h:g}"] = float("nan")
    if not np.isfinite(out["E2"]):
        out["E2_status"] = "E2_NONFINITE"
    else:
        out["E2_status"] = "MULTIMODAL_NONNEGLIGIBLE" if cl["multistart"] == "MULTIMODAL_NONNEGLIGIBLE" else "OK"
    return out


# --------------------------------------------------------------------------
# one cell (as v3.run_cell, with the fixed outer certificate and explicit E status)
# --------------------------------------------------------------------------

def run_cell(case, j, cand, prior_name, case_index, main: bool) -> dict:
    t0 = time.perf_counter()
    J, v = a8.joint_for(case, j, cand, prior_name)
    Za, starts = v3.make_starts(case, j, cand)
    L = v3.Local3(J, Za)
    row: dict[str, Any] = {"replicate": case["label"], "regime": case["regime"], "column": j, "candidate": cand,
                           "prior": prior_name, "prior_var_f": v}
    try:
        de = method_D_E2(L, starts)
        hess_f, Z_D, f_D = de.pop("hess_f", None), de.pop("Z_D", None), de.pop("f_D", None)
        row.update(de)
        if f_D is not None:
            row["f_D"] = "|".join(repr(float(a)) for a in f_D)
            row["fD_norm"] = float(np.linalg.norm(f_D))
        e_starts, z_starts = dict(starts), {}
        if f_D is not None:
            e_starts["f_D"], z_starts["f_D"] = f_D, Z_D
        J2, _ = a8.joint_for(case, j, cand, prior_name)
        e = v3.method_E(J2, Za, e_starts, z_starts)
        f_E = e.pop("f_E", None)
        row.update(e)
        if f_E is not None:
            row["f_E"] = "|".join(f"{a:.10g}" for a in f_E)
        if "E" in row and not np.isfinite(row["E"]):
            row["E_status"] = "E_NONFINITE"
        if main and f_D is not None and row.get("E2_status") in ("OK", "MULTIMODAL_NONNEGLIGIBLE"):
            L2 = v3.Local3(a8.joint_for(case, j, cand, prior_name)[0], Za)
            L2.T(f_D + 0.3)
            L2.T(np.zeros(K))
            ok2, t2 = L2.T(f_D)
            row["determinism_abs_diff"] = abs(t2 - row["D"]) if ok2 else float("nan")
            row.update(v3.outer_checks(L, f_D, row["D"], hess_f))
            row.update(v3.method_IS(L, f_D, hess_f, Z_D, IS_SEED_BASE + case_index))
        row["status"] = "OK" if row.get("E2_status") == "OK" else ("FLAGGED" if row.get("E2_status") else "FAILED")
    except Exception as exc:                         # unexpected: recorded, never retried
        row.update({"status": "EXCEPTION", "error": f"{type(exc).__name__}: {exc}"[:300]})
    row["n_S_evals"] = L.n_eval
    row["n_S_not_ok"] = int(sum(not s.ok for s in L.cache.values()))
    row["seconds"] = round(time.perf_counter() - t0, 2)
    return row


# --------------------------------------------------------------------------
# protocol (fixed before the re-pilot and before the 40-dataset run)
# --------------------------------------------------------------------------

def protocol() -> dict[str, Any]:
    return {
        "experiment": "AF8-v3 final: fixes + re-pilot + 40-dataset exploratory run",
        "study_type": "EXPLORATORY. The 40 datasets are the AF6 datasets, already used by AF6-AF8 analyses; results are not independent confirmatory evidence. Written after the AF8-v3 pilot numerics (no accuracy of any AF8 variant has been computed).",
        "lineage": "lineage E + PR #129 generic model (prototype, not a formal method, not manuscript evidence); AF6 auto_first anchors",
        "parent": {"AF8_v3_code_head": V3_CODE_HEAD, "AF8_v3_pilot_results": V3_PILOT_HEAD,
                   "AF8_v3_protocol_sha256": "c853c0092167a089a19ff285bcd42d262204be499efbc80151478607a71897cb"},
        "research_question": "Does integrating the loading f_j (prior + nested Laplace E2) improve the BALANCE of the two misselection directions between Poisson and CMP2 relative to Q and to the AF6 Laplace (f optimised) score? Higher overall accuracy alone is not counted as success; Poisson->CMP2 and CMP2->Poisson errors are both evaluated.",
        "changes_vs_v3": {
            "outer_certificate": "CERT_OK only with T(f) finite and re-validated each round, finite FD gradient/Hessian, Cholesky-PD -d2T/df2, finite lambda_f^2/2 <= 1e-6, and a FINITE extra-step gain <= 1e-6 at the full Newton step or (if T is undefined there) at 1/2 or 1/4 of it; otherwise EXTRA_STEP_INVALID / EXTRA_STEP_NONFINITE / FD_NONFINITE / DECREMENT_FAIL / POINT_INVALID (failures). Every start's certificate is logged (D_start_log). A start is valid only if CERT_OK with all certificate fields finite.",
            "E_status": "E with a non-finite value -> E_NONFINITE (failure)",
            "stop_rules": "v3 rules unchanged; added before the re-pilot: S4v2 (v2-definition S4 <= 0.05, audit W1; stricter), S0 completeness (all expected cells present). No threshold relaxed.",
            "analysis": "no NaN-skipping aggregation: every statistic is computed by functions that fail on NaN unless the NaN is an explicit state; completeness of the expected cell grid is checked; analysis runinfo with git head, protocol hash, environment and sha256 of every input file",
            "unchanged": "model, theta_{-j}, data, priors, starts, inner/joint certificates, FD steps, IS proposal/draws/seeds/rescue, outer_checks, E2 as main score",
        },
        "data": "AF6 saved datasets rep01-rep40 (regenerated from the AF6 seeds; X, Y, train-mask hashes asserted against AF6 fit_manifest; anchor state files checked by sha256); n=75, K=3, columns Gaussian x3, Poisson x3, CMP2 x3; dense-Y and sparse-Y; same Auto (auto_first) anchor. No MCEM fit, no new data.",
        "units": {"cell": "dataset x regime x count column x candidate x prior (40 x 2 x 6 x 2 x 9 = 8640)",
                  "column": "dataset x regime x count column (480); selection unit",
                  "statistical_unit": "dataset (40). Columns and regimes are not independent; dense and sparse are reported separately and pooled with dataset-cluster resampling"},
        "methods": {
            "A_Q": "AF6 q_selected / q_margin_cmp_minus_pois (same anchor), comparator",
            "B_L6": "AF6 Laplace delta (f maximises the Z-Laplace objective, no prior), comparator; AF6 status FAILED -> failed (no Q fallback)",
            "C_JM": "AF6 delta_jointmap_variant (joint MAP then Laplace), comparator",
            "D_MAP": "max_f T(f) = S(f) + log p(f) (prior, f optimised, not integrated), comparator",
            "E2": "MAIN: nested Laplace, T(f_D) + K/2 log 2pi - 1/2 log|FD Hessian of -T|",
            "E": "joint Laplace over (vec Z, f_j), comparator / approximation diagnostic",
            "IS": "f integrated by importance sampling given the Z-Laplace S(f) (main prior only): numerical check of the outer Gaussian approximation of E2, NOT the exact marginal",
        },
        "prior_main": MAIN_PRIOR, "priors_sensitivity": [p for p in PRIORS if p != MAIN_PRIOR],
        "fixed_choices": "main method E2 and main prior B_sg2_0.667 are fixed here and not changed after seeing accuracy; E vs E2 is not chosen by performance",
        "selection": {
            "Delta": "M(CMP2) - M(Poisson) for every method; CMP2 if Delta > 0",
            "column_states": {"BOTH_VALID_DECIDED": "both candidates valid and |Delta| > u", "NUMERIC_UNDECIDED": "both valid, |Delta| <= u",
                              "ONE_FAILED": "exactly one candidate failed", "BOTH_FAILED": "both failed", "MISSING": "expected input/row absent"},
            "u": "E2: v2.selection_state (u = max(0.01, 2 x FD-Hessian-step sensitivity)); other methods: 0.01",
            "validity": "E2: cell status OK (certified, multistart AGREE or dominant); E: E_status OK; D: E2_status OK (same optimisation); IS: IS_status OK; B: AF6 status in FINITE_DELTA with finite delta; C: finite delta_jointmap_variant; A: q_status SELECTED",
            "no_fallback": "a failed / undecided / missing column is NEVER replaced by Q; a Q-fallback hybrid is reported only as a separately named exploratory secondary method (E2|Q)",
        },
        "metrics": {
            "denominators": "acc_all = correct decided / all columns of that truth (failed, undecided and missing count as not correct); acc_evaluable = correct / decided columns; rates of decided, undecided, failed, missing reported separately",
            "primary": ["accuracy on true Poisson (acc_all)", "accuracy on true CMP2 (acc_all)", "balanced accuracy", "Poisson->CMP2 misselection rate", "CMP2->Poisson misselection rate"],
            "secondary": ["overall accuracy", "AUC of Delta (finite scores; n reported)", "per-dataset #correct difference E2-Q, E2-B, E2-D, E2-E, D-B", "right->wrong and wrong->right transitions by truth", "sign-based accuracy (undecided resolved by sign) as sensitivity", "prior sensitivity (D, E2, E for every prior)", "E vs E2 sign disagreements", "numerical approximation errors", "undecided and failure rates"],
            "bootstrap": "dataset-cluster bootstrap (resample the 40 datasets with both regimes and all columns), 10000 resamples, seed 20261118 (AF8 BOOT_SEED), percentile 95% CI",
            "multiplicity": "one primary comparison family (E2 vs Q and E2 vs B on the per-truth accuracies and balanced accuracy, main prior); CIs are unadjusted; all other comparisons and the prior sensitivity are exploratory",
        },
        "ambiguous_columns_label_free": "main prior, E2 valid: |Delta_E2| <= 1, or sign(Delta_E) != sign(Delta_E2), or sign(Delta_IS) != sign(Delta_E2), or E2 state NUMERIC_UNDECIDED. Diagnostics reported for them (IS +- 2 se, decomposition of Delta_E - Delta_E2, FD-step sensitivity); never used to change method, prior or thresholds",
        "stop_rules": {k: v for k, v in STOP_RULES.items()},
        "stop_rule_units": "S1, S2, S2b over all cells of the stage; S3/S3b over main-prior pairs with both IS valid; S4, S4v2, S4r, S4b, S5, S8, S9 over main-prior cells; S6 over main-prior pairs; S7 (FD derivative check) in the re-pilot only (column 3, main prior, every pilot case)",
        "gates": {"re_pilot": "rep01-rep03, 648 cells, numerical only (no accuracy, no AUC). ANY stop rule -> no 40-dataset run",
                  "full": "rep01-rep40, 8640 cells. The same stop rules are evaluated on the full run BEFORE any accuracy is computed; if any fires the analysis refuses to compute accuracy and the run is reported as stopped",
                  "publication": "after a passing re-pilot: push the v3 branch and open a Draft PR on the AF8-v2 branch"},
        "not_claimed": ["Bayesian evidence of the whole model (theta_{-j} fixed)", "BIC", "exact marginal likelihood",
                        "E2 ~ IS implies closeness to the exact (Z, f) marginal", "independent confirmation", "real-data validity",
                        "posterior probability of a family"],
    }


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def env_info() -> dict:
    import scipy
    return {"git_head": a8.git("rev-parse", "HEAD"), "git_branch": a8.git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_tracked": bool(a8.git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(), "python_executable": sys.executable,
            "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
            "platform": platform.platform(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "protocol_sha256": protocol_sha256(), "command": " ".join(sys.argv)}


def runinfo(extra):
    return {"experiment": "AF8-v3 final", "script": "expfam/src/experimental/run_af8v3_full_integrated_loading.py",
            **env_info(), **extra}


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------

def run(stage, reps, shard=None):
    out = OUT / stage
    out.mkdir(parents=True, exist_ok=True)
    items = list(a8.cells(reps))
    keys = sorted({(r, g) for r, g, *_ in items})
    tag = ""
    if shard:
        i, n = (int(v) for v in shard.split("/"))
        keep = set(keys[i::n])
        items = [c for c in items if (c[0], c[1]) in keep]
        tag = f"_{i}of{n}"
    ri = out / f"runinfo{tag}.json"
    prev = json.loads(ri.read_text(encoding="utf-8")) if ri.exists() else None
    info = runinfo({"stage": stage, "shard": shard, "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    if prev is not None:                                  # a restart keeps the earlier start record
        info["previous_starts"] = prev.get("previous_starts", []) + [{k: prev.get(k) for k in ("git_head", "started", "protocol_sha256")}]
        if prev.get("git_head") != info["git_head"] or prev.get("protocol_sha256") != info["protocol_sha256"]:
            raise SystemExit("restart with a different git head / protocol; refusing to mix")
    ri.write_text(json.dumps(info, indent=2), encoding="utf-8")
    path = out / f"progress{tag}.jsonl"
    done = set()
    if path.exists():
        done = {(d["replicate"], d["regime"], d["column"], d["candidate"], d["prior"])
                for d in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    case, t0 = None, time.time()
    for r, g, j, cand, pr in items:
        label = f"rep{r + 1:02d}"
        if (label, g, j, cand, pr) in done:
            continue
        if case is None or case["label"] != label or case["regime"] != g:
            case = a8.load_case(r, g)
            case["regime"] = g
        row = run_cell(case, j, cand, pr, v3.case_index(r, g, j, cand), main=(pr == MAIN_PRIOR))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        print(f"  {label} {g} j={j} {cand[:4]} {pr} {row['status']} {row['seconds']}s total {time.time() - t0:.0f}s", flush=True)


def collect(stage):
    rows = []
    for p in sorted((OUT / stage).glob("progress*.jsonl")):
        rows += [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()]
    c = pd.DataFrame(rows)
    c.to_csv(OUT / stage / "candidate_scores.csv", index=False)
    return c


def check_protocol():
    p = OUT / "protocol.json"
    if not p.exists() or json.loads(p.read_text(encoding="utf-8")) != json.loads(json.dumps(protocol(), ensure_ascii=False)):
        raise SystemExit("frozen protocol.json missing or differs from code; refusing to run")
    if a8.git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked files have uncommitted changes; commit code and protocol before running")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    sh = argv[argv.index("--shard") + 1] if "--shard" in argv else None
    if "--write-protocol" in argv:
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / "protocol.json"
        if p.exists():
            raise SystemExit("protocol.json exists; refusing to overwrite")
        p.write_text(json.dumps(protocol(), indent=2, ensure_ascii=False), encoding="utf-8")
        print("protocol sha256", protocol_sha256())
    if "--pilot" in argv:
        check_protocol()
        run("pilot", PILOT_REPS, sh)
    if "--full" in argv:
        check_protocol()
        audit = OUT / "pilot" / "pilot_numerical_audit.json"
        if not audit.exists() or not json.loads(audit.read_text(encoding="utf-8")).get("all_stop_rules_pass") is True:
            raise SystemExit("re-pilot audit missing or not passed; refusing to start the 40-dataset run")
        run("full", FULL_REPS, sh)
    if "--collect-pilot" in argv:
        collect("pilot")
    if "--collect-full" in argv:
        collect("full")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
