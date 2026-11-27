"""Post-hoc diagnostics for the two-gaps experiments (A, B), 2026-11-27.  NO new fits.

Uses only: saved NPZ states (sha256-checked against per_fit.csv), per_fit.csv, and the
training data re-built deterministically from the recorded seeds with the runner's own
generators (hashes checked against per_fit.csv).  Protocol: diagnostics_20261127/analysis_protocol.md
POST HOC / EXPLORATORY.  C_Q is not Schwarz BIC; C_Lap is not an exact marginal likelihood.

Usage: python tools/research_audit/diagnose_two_gaps_20261127.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_two_gaps_20261127 as tg  # noqa: E402  (data generators / constants, unchanged)

rc = tg.rc
ROOT = tg.ROOT
BASE = ROOT / "expfam/results/k_selection/two_gaps_20261127"
OUT = BASE / "diagnostics_20261127"
N, D = tg.N, tg.D
LN_N = math.log(N)
BOOT, BOOT_SEED = 10_000, 20261128
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"check": name, "pass": bool(ok), "detail": str(detail)[:400]})


def rd(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def wr(p, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def load_state(run, row):
    p = run / "states" / row["state_file"]
    ok = rc.sha256_file(p) == row["state_sha256"]
    return dict(np.load(p)), ok


def eig_stats(M, prefix):
    ev = np.linalg.eigvalsh(0.5 * (M + M.T))
    k = len(ev)
    return {f"{prefix}_trace": float(ev.sum()), f"{prefix}_trace_per_dim": float(ev.sum() / k),
            f"{prefix}_eig_min": float(ev[0]), f"{prefix}_eig_max": float(ev[-1]),
            f"{prefix}_cond": float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf")}


def y_info(Z, w0, w, mask=None):
    """(w^2/n) sum_i sum_{j != i, observed} p(1-p) z_j z_j^T  (per-node average Y-side information)."""
    P = sig(w0 + w * (Z @ Z.T))
    V = P * (1 - P)
    np.fill_diagonal(V, 0.0)
    if mask is not None:
        V = V * mask
    c = V.sum(axis=0)
    return (w ** 2 / Z.shape[0]) * (Z.T * c) @ Z


def x_info(family, F, Z=None, var=None):
    if family == "gaussian":
        return F.T @ (F / var[:, None])
    if family == "bernoulli":
        P = sig(Z @ F.T)
        a = (P * (1 - P)).mean(axis=0)
    else:  # poisson, empirical over the rows of Z
        a = np.exp(Z @ F.T).mean(axis=0)
    return F.T @ (F * a[:, None])


# ---------------------------------------------------------------------------- metrics
def auc(y, p):
    from scipy.stats import rankdata
    r = rankdata(p)
    npos = y.sum()
    nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def avg_precision(y, p):
    o = np.argsort(-p, kind="mergesort")
    ys = y[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def calib_slope(y, p):
    x = np.log(p) - np.log1p(-p)
    if np.ptp(x) < 1e-12:           # constant predictor: slope undefined
        return float("nan"), float("nan")
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(50):
        m = sig(X @ b)
        g = X.T @ (y - m)
        H = (X * (m * (1 - m))[:, None]).T @ X
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-12:
            break
    return float(b[1]), float(b[0])


def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def pred_metrics(y, eta, prefix):
    p = np.clip(sig(eta), 1e-15, 1 - 1e-15)
    ls = y * eta - np.logaddexp(0.0, eta)
    pos, neg = y == 1, y == 0
    slope, icpt = calib_slope(y, p)
    return {f"{prefix}_log_score": float(ls.mean()), f"{prefix}_log_score_pos": float(ls[pos].mean()),
            f"{prefix}_log_score_neg": float(ls[neg].mean()), f"{prefix}_p_mean_pos": float(p[pos].mean()),
            f"{prefix}_p_mean_neg": float(p[neg].mean()), f"{prefix}_p_mean": float(p.mean()),
            f"{prefix}_p_sd": float(p.std()), f"{prefix}_p_q10": float(np.quantile(p, 0.1)),
            f"{prefix}_p_q50": float(np.quantile(p, 0.5)), f"{prefix}_p_q90": float(np.quantile(p, 0.9)),
            f"{prefix}_brier": float(np.mean((p - y) ** 2)), f"{prefix}_auc": auc(y, p),
            f"{prefix}_ap": avg_precision(y, p), f"{prefix}_calib_in_large": float(p.mean() - y.mean()),
            f"{prefix}_calib_slope": slope, f"{prefix}_calib_intercept": icpt, f"{prefix}_ece": ece(y, p)}


def boot_ci(v, seed=BOOT_SEED):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (BOOT, len(v)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


# ============================================================================ Experiment A
def diag_A():
    run = BASE / "expA/run"
    rows = [r for r in rd(run / "per_fit.csv") if r["phase"] == "main" and r["status"] == "OK"]
    check("A: 160 OK main fits available", len(rows) == 160, len(rows))
    dz = tg.a_design()
    iu = np.triu_indices(N, 1)
    cache, fits, par = {}, [], []
    sha_bad, hash_bad, reprod = [], [], []
    for r in rows:
        ds_seed = int(r["data_seed"])
        if ds_seed not in cache:
            ds = tg.a_generate(ds_seed)
            cache[ds_seed] = (ds, tg.a_masks(ds_seed))
        ds, masks = cache[ds_seed]
        if (tg.sha16(ds.X), tg.sha16(ds.Y), tg.sha16(masks["test"]), tg.sha16(masks[r["rate"]])) != (
                r["x_hash"], r["y_hash"], r["test_hash"], r["train_hash"]):
            hash_bad.append(r["fit_id"])
        st, ok = load_state(run, r)
        if not ok:
            sha_bad.append(r["fit_id"])
        Zl, Zm, F = st["Z_last"], st["Z_mean"], st["F"]
        w0, w = float(st["w0"]), float(st["w"])
        tr, tc = np.where(np.triu(masks["test"], 1))
        y = ds.Y[tr, tc]
        s_l = np.sum(Zl[tr] * Zl[tc], 1)
        s_m = np.sum(Zm[tr] * Zm[tc], 1)
        s_t = np.sum(ds.Z[tr] * ds.Z[tc], 1)
        eta_true = dz["w0"] + dz["w"] * s_t
        trm = np.triu(masks[r["rate"]], 1)
        base_p = float(ds.Y[trm].mean())
        rec = {"fit_id": r["fit_id"], "dataset_id": r["dataset_id"], "rate": r["rate"],
               "family_x_fit": r["family_x_fit"], "t": r["t"], "n_test": len(y), "test_pos_rate": float(y.mean()),
               "train_pos_rate": base_p}
        rec.update(pred_metrics(y, w0 + w * s_l, "plugin_last"))
        rec.update(pred_metrics(y, w0 + w * s_m, "plugin_mean"))
        rec.update(pred_metrics(y, np.full_like(y, math.log(base_p / (1 - base_p))), "const"))
        rec.update(pred_metrics(y, eta_true, "truth"))
        rec["corr_eta_hat_true"] = float(np.corrcoef(w0 + w * s_l, eta_true)[0, 1])
        d_ls = abs(rec["plugin_last_log_score"] - float(r["heldout_log_score"]))
        reprod.append(d_ls)
        fits.append(rec)
        G = (Zl @ Zl.T)[iu]
        p_rec = {"exp": "A", "fit_id": r["fit_id"], "dataset_id": r["dataset_id"], "cond": r["rate"],
                 "family_x_fit": r["family_x_fit"], "k": 3, "t": r["t"], "w_hat": w, "w0_hat": w0,
                 "w_true": dz["w"], "w0_true": dz["w0"], "F_fro": float(np.linalg.norm(F)),
                 "F_fro_true": float(np.linalg.norm(ds.F)), "Z_meansq": float(np.mean(Zl ** 2)),
                 "zz_sd_all_pairs": float(G.std()), "zz_sd_true": float((ds.Z @ ds.Z.T)[iu].std()),
                 "eta_sd_test": float((w * s_l).std()), "eta_sd_true_test": float((dz["w"] * s_t).std()),
                 "gram_corr_last": float(np.corrcoef(G, (ds.Z @ ds.Z.T)[iu])[0, 1]),
                 "rmse_Z_last": tg.procrustes_rmse(Zl, ds.Z)}
        p_rec.update(eig_stats(x_info(r["family_x_fit"], F, Zl), "xinfo_est"))
        p_rec.update(eig_stats(y_info(Zl, w0, w, masks[r["rate"]]), "yinfo_est"))
        p_rec.update(eig_stats(x_info("bernoulli", ds.F, ds.Z), "xinfo_true"))
        p_rec.update(eig_stats(y_info(ds.Z, dz["w0"], dz["w"], masks[r["rate"]]), "yinfo_true"))
        for c in ("gram_corr_last", "rmse_Z_last"):
            reprod.append(abs(p_rec[c] - float(r[c])))
        par.append(p_rec)
    check("A: rebuilt X/Y/test/train hashes = per_fit", not hash_bad, hash_bad[:5])
    check("A: NPZ sha256 = per_fit", not sha_bad, sha_bad[:5])
    check("A: recomputed heldout_log_score / gram_corr / rmse_Z = per_fit (max abs diff < 1e-9)",
          max(reprod) < 1e-9, max(reprod))

    # per (dataset, rate, family): mean of 2 inits (protocol), then paired differences
    keyed = defaultdict(list)
    pidx = {p["fit_id"]: p for p in par}
    for f in fits:
        keyed[(f["dataset_id"], f["rate"], f["family_x_fit"])].append({**f, **pidx[f["fit_id"]]})
    cells = []
    for (ds_id, rate, fam), v in sorted(keyed.items()):
        assert len(v) == 2
        c = {"dataset_id": ds_id, "rate": rate, "family_x_fit": fam}
        for k in v[0]:
            if isinstance(v[0][k], float) and k not in ("t",):
                c[k] = float(np.mean([x[k] for x in v]))
        cells.append(c)
    # cross-family gram agreement (same dataset, rate, t)
    zl = {}
    for r in rows:
        zl[r["fit_id"]] = np.load(run / "states" / r["state_file"])["Z_last"]
    cross = {}
    for ds_id in sorted({r["dataset_id"] for r in rows}):
        rep = int(ds_id[-3:])
        for rate in tg.A_RATES:
            v = []
            for t in (0, 1):
                a = zl[tg.a_fit_id(rep, rate, "bernoulli", t)]
                b = zl[tg.a_fit_id(rep, rate, "poisson", t)]
                v.append(np.corrcoef((a @ a.T)[iu], (b @ b.T)[iu])[0, 1])
            cross[(ds_id, rate)] = float(np.mean(v))
    for c in cells:
        c["gram_corr_bern_vs_pois_fit"] = cross[(c["dataset_id"], c["rate"])]

    metrics = [("plugin_last_log_score", 1), ("plugin_last_log_score_pos", 1), ("plugin_last_log_score_neg", 1),
               ("plugin_last_brier", -1), ("plugin_last_auc", 1), ("plugin_last_ap", 1),
               ("plugin_last_ece", -1), ("plugin_last_calib_slope", 0), ("plugin_last_calib_in_large", 0),
               ("plugin_last_p_mean_pos", 0), ("plugin_last_p_mean_neg", 0), ("plugin_last_p_sd", 0),
               ("plugin_mean_log_score", 1), ("plugin_mean_auc", 1), ("plugin_mean_brier", -1),
               ("w_hat", 0), ("w0_hat", 0), ("eta_sd_test", 0), ("corr_eta_hat_true", 1),
               ("gram_corr_last", 1), ("rmse_Z_last", -1), ("F_fro", 0),
               ("xinfo_est_trace", 0), ("yinfo_est_trace", 0)]
    cm = {(c["dataset_id"], c["rate"], c["family_x_fit"]): c for c in cells}
    dsets = sorted({c["dataset_id"] for c in cells})
    summ = []
    for rate in tg.A_RATES:
        for m, better in metrics:
            b = np.array([cm[(d, rate, "bernoulli")][m] for d in dsets])
            p = np.array([cm[(d, rate, "poisson")][m] for d in dsets])
            diff = b - p
            lo, hi = boot_ci(diff)
            rec = {"rate": rate, "metric": m, "higher_is_better": {1: "yes", -1: "no", 0: "n/a"}[better],
                   "n_datasets": len(dsets), "mean_bernoulli": float(b.mean()), "mean_poisson": float(p.mean()),
                   "mean_diff_bern_minus_pois": float(diff.mean()), "median_diff": float(np.median(diff)),
                   "boot95_lo": lo, "boot95_hi": hi,
                   "n_bern_better": int((diff * better > 0).sum()) if better else "",
                   "n_pois_better": int((diff * better < 0).sum()) if better else ""}
            summ.append(rec)
        for ref in ("const", "truth"):
            for m in ("log_score", "brier", "auc"):
                v = np.array([cm[(d, rate, "bernoulli")][f"{ref}_{m}"] for d in dsets])
                summ.append({"rate": rate, "metric": f"{ref}_{m}", "higher_is_better": "",
                             "n_datasets": len(dsets), "mean_bernoulli": float(v.mean()), "mean_poisson": "",
                             "mean_diff_bern_minus_pois": "", "median_diff": "", "boot95_lo": "", "boot95_hi": "",
                             "n_bern_better": "", "n_pois_better": ""})
        # each fit family vs constant baseline (log score)
        for fam in tg.A_FAMS:
            diff = np.array([cm[(d, rate, fam)]["plugin_last_log_score"] - cm[(d, rate, fam)]["const_log_score"]
                             for d in dsets])
            lo, hi = boot_ci(diff)
            summ.append({"rate": rate, "metric": f"{fam}_log_score_minus_const", "higher_is_better": "yes",
                         "n_datasets": len(dsets), "mean_bernoulli": "", "mean_poisson": "",
                         "mean_diff_bern_minus_pois": float(diff.mean()), "median_diff": float(np.median(diff)),
                         "boot95_lo": lo, "boot95_hi": hi, "n_bern_better": int((diff > 0).sum()),
                         "n_pois_better": ""})
        # decomposition of the overall log-score difference into positive / negative contributions
        pi = np.array([cm[(d, rate, "bernoulli")]["test_pos_rate"] for d in dsets])
        dpos = np.array([cm[(d, rate, "bernoulli")]["plugin_last_log_score_pos"]
                         - cm[(d, rate, "poisson")]["plugin_last_log_score_pos"] for d in dsets])
        dneg = np.array([cm[(d, rate, "bernoulli")]["plugin_last_log_score_neg"]
                         - cm[(d, rate, "poisson")]["plugin_last_log_score_neg"] for d in dsets])
        for nm, v in (("contrib_pos = pi*dpos", pi * dpos), ("contrib_neg = (1-pi)*dneg", (1 - pi) * dneg)):
            lo, hi = boot_ci(v)
            summ.append({"rate": rate, "metric": nm, "higher_is_better": "yes", "n_datasets": len(dsets),
                         "mean_bernoulli": "", "mean_poisson": "", "mean_diff_bern_minus_pois": float(v.mean()),
                         "median_diff": float(np.median(v)), "boot95_lo": lo, "boot95_hi": hi,
                         "n_bern_better": int((v > 0).sum()), "n_pois_better": int((v < 0).sum())})
        # association (descriptive, across datasets): w_hat gap vs log-score gap
        dw = np.array([cm[(d, rate, "bernoulli")]["w_hat"] - cm[(d, rate, "poisson")]["w_hat"] for d in dsets])
        dl = np.array([cm[(d, rate, "bernoulli")]["plugin_last_log_score"]
                       - cm[(d, rate, "poisson")]["plugin_last_log_score"] for d in dsets])
        summ.append({"rate": rate, "metric": "spearman(diff w_hat, diff log_score) across datasets",
                     "higher_is_better": "", "n_datasets": len(dsets), "mean_bernoulli": "", "mean_poisson": "",
                     "mean_diff_bern_minus_pois": _spearman(dw, dl), "median_diff": "", "boot95_lo": "",
                     "boot95_hi": "", "n_bern_better": "", "n_pois_better": ""})
    return fits, par, cells, summ


def _spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


# ============================================================================ Experiment B
def diag_B():
    from eval_utils import calc_Q_dual_strict_exp
    from scipy.special import gammaln
    run = BASE / "expB/run"
    rows = [r for r in rd(run / "per_fit.csv") if r["phase"] == "main" and r["status"] == "OK"]
    check("B: 600 OK main fits available", len(rows) == 600, len(rows))
    cache, par, terms = {}, [], []
    sha_bad, hash_bad, qres, zsq_dev = [], [], [], []
    for r in rows:
        key = (int(r["data_seed"]), r["cond"])
        if key not in cache:
            cache[key] = tg.b_generate(*key)
        X, Y, Z, F, var = cache[key]
        if (tg.sha16(X), tg.sha16(Y), tg.sha16(Z), tg.sha16(F)) != (r["x_hash"], r["y_hash"], r["z_hash"], r["f_hash"]):
            hash_bad.append(r["fit_id"])
        st, ok = load_state(run, r)
        if not ok:
            sha_bad.append(r["fit_id"])
        k = int(r["k"])
        fam = tg.B_FX[r["cond"]]
        Fh, Zl = st["F"], st["Z_last"]
        w0, w, vz = float(st["w0"]), float(st["w"]), float(st["var_z"])
        p = {"exp": "B", "fit_id": r["fit_id"], "dataset_id": r["dataset_id"], "replicate_id": r["replicate_id"],
             "cond": r["cond"], "family_x_fit": fam, "k": k, "t": r["t"], "w_hat": w, "w0_hat": w0, "var_z": vz,
             "F_fro": float(np.linalg.norm(Fh)), "Z_meansq": float(np.mean(Zl ** 2)),
             "x_var_true": var if fam == "gaussian" else ""}
        if fam == "gaussian":
            sv = np.diag(st["sigma"]).astype(float)
            p.update(sigma2_mean=float(sv.mean()), sigma2_min=float(sv.min()), sigma2_max=float(sv.max()),
                     sigma2_mean_over_true=float(sv.mean() / var),
                     sigma2_derived_if_missing_dims_absorbed=float(var + max(0, tg.B_K - k) * 0.5 / tg.B_K))
            p.update(eig_stats(x_info("gaussian", Fh, var=sv), "xinfo_est"))
        else:
            p.update(eig_stats(x_info("poisson", Fh, Zl), "xinfo_est"))
            p.update(eig_stats(tg.fisher_poisson_expected(Fh), "xinfo_est_prior_expect"))
        p.update(eig_stats(y_info(Zl, w0, w), "yinfo_est"))
        par.append(p)
        # ---- C_Q terms on the single saved sample Z_last (existing model functions)
        model = rc.restore_model(tg.B_SERIES[r["cond"]], k, st, Zl)
        Z3 = Zl[:, :, None]
        lnpZ = float(-(N * k / 2.0) * math.log(2.0 * math.pi * vz) - np.sum(Zl ** 2) / (2.0 * vz))
        lnpX = float(model.calc_log_likelihood_X(X, Z3, Fh))
        if fam == "poisson":
            lnpX -= float(np.sum(gammaln(X + 1)))
        lnpY = float(model.calc_log_likelihood_Y(Y, Z3, w0, w))
        q1 = calc_Q_dual_strict_exp(X, Y, Z3, Fh, st["sigma"], vz, w0, w, model)
        npar = int(r["num_params"])
        q_saved = float(r["Q_strict"])
        rec = {"fit_id": r["fit_id"], "dataset_id": r["dataset_id"], "replicate_id": r["replicate_id"],
               "cond": r["cond"], "k": k, "t": r["t"], "m2_lnpZ": -2 * lnpZ, "m2_lnpX": -2 * lnpX,
               "m2_lnpY": -2 * lnpY, "penalty": npar * LN_N, "num_params": npar,
               "C_Q_1sample": -2 * (lnpZ + lnpX + lnpY) + npar * LN_N, "C_Q_saved": float(r["C_Q"]),
               "mc_residual_C_Q_saved_minus_1sample": float(r["C_Q"]) - (-2 * (lnpZ + lnpX + lnpY) + npar * LN_N),
               "sum_terms_minus_calcQ_1sample": (lnpZ + lnpX + lnpY) - q1,
               "derived_m2_lnpZ_if_sumsq_eq_nK": N * k * (1 + math.log(2 * math.pi)),
               "lap_status": r["lap_status"]}
        for c in ("lap_fit_mode", "lap_zhat_sq", "lap_logdet_H", "lap_penalty", "C_Lap"):
            rec[c] = float(r[c]) if r[c] not in ("",) else ""
        rec["q_rel_change_last"] = float(r["q_rel_change_last"]) if r["q_rel_change_last"] else ""
        rec["mstep_q_decreases"] = int(r["mstep_q_decreases"])
        terms.append(rec)
        qres.append(abs(rec["sum_terms_minus_calcQ_1sample"]))
        zsq_dev.append(abs(rec["m2_lnpZ"] - rec["derived_m2_lnpZ_if_sumsq_eq_nK"]))
        _ = q_saved
    check("B: rebuilt X/Y/Z/F hashes = per_fit", not hash_bad, hash_bad[:5])
    check("B: NPZ sha256 = per_fit", not sha_bad, sha_bad[:5])
    check("B: lnpZ + lnpX + lnpY = calc_Q_dual_strict_exp on Z_last (max abs < 1e-6)", max(qres) < 1e-6, max(qres))
    check("B: -2 lnpZ(Z_last) vs nK(1 + ln 2pi): max abs deviation recorded (scale_Z normalises over all L samples jointly, not per sample)", True, max(zsq_dev))
    return par, terms


def summarize_B(par, terms):
    # per dataset x K: mean over 2 inits; then medians over datasets
    def cell_means(rows, cols):
        g = defaultdict(list)
        for r in rows:
            g[(r["cond"], r["dataset_id"], r["k"])].append(r)
        out = {}
        for key, v in g.items():
            out[key] = {c: float(np.mean([x[c] for x in v])) for c in cols if all(x.get(c, "") != "" for x in v)}
            out[key]["init_absdiff"] = {c: abs(v[0][c] - v[1][c]) for c in cols
                                        if len(v) == 2 and all(x.get(c, "") != "" for x in v)}
        return out

    pcols = ["w_hat", "w0_hat", "F_fro", "sigma2_mean", "sigma2_min", "sigma2_max", "sigma2_mean_over_true",
             "sigma2_derived_if_missing_dims_absorbed", "xinfo_est_trace", "xinfo_est_trace_per_dim",
             "xinfo_est_eig_min", "xinfo_est_eig_max", "xinfo_est_cond", "xinfo_est_prior_expect_trace",
             "xinfo_est_prior_expect_trace_per_dim", "xinfo_est_prior_expect_cond", "yinfo_est_trace",
             "yinfo_est_trace_per_dim", "yinfo_est_eig_min", "yinfo_est_eig_max", "yinfo_est_cond"]
    pc = cell_means(par, pcols)
    vsum = []
    for cond in tg.B_CONDS:
        for k in tg.B_K_CANDS:
            keys = [key for key in pc if key[0] == cond and key[2] == k]
            rec = {"cond": cond, "k": k, "n_datasets": len(keys)}
            for c in pcols:
                v = [pc[key][c] for key in keys if c in pc[key]]
                if v:
                    rec[f"{c}_median"] = float(np.median(v))
                    rec[f"{c}_min"] = float(np.min(v))
                    rec[f"{c}_max"] = float(np.max(v))
            for c in ("sigma2_mean", "xinfo_est_trace", "w_hat"):
                v = [pc[key]["init_absdiff"][c] for key in keys if c in pc[key]["init_absdiff"]]
                if v:
                    rec[f"{c}_init_absdiff_median"] = float(np.median(v))
            vsum.append(rec)

    tcols = ["m2_lnpZ", "m2_lnpX", "m2_lnpY", "penalty", "C_Q_1sample", "C_Q_saved",
             "mc_residual_C_Q_saved_minus_1sample", "lap_fit_mode", "lap_zhat_sq", "lap_logdet_H", "lap_penalty", "C_Lap"]
    tc = cell_means(terms, tcols)
    dsum, sel = [], []
    for cond in tg.B_CONDS:
        dsets = sorted({key[1] for key in tc if key[0] == cond})
        # selection on saved C_Q vs on the 1-sample C_Q
        for ds in dsets:
            sq = {k: tc[(cond, ds, k)]["C_Q_saved"] for k in tg.B_K_CANDS}
            s1 = {k: tc[(cond, ds, k)]["C_Q_1sample"] for k in tg.B_K_CANDS}
            ks, k1 = _sel(sq), _sel(s1)
            # init spread vs gap between selected K and K=9 (saved C_Q)
            spread9 = tc[(cond, ds, 9)]["init_absdiff"].get("C_Q_saved", float("nan"))
            spreadk = tc[(cond, ds, ks)]["init_absdiff"].get("C_Q_saved", float("nan"))
            sel.append({"cond": cond, "dataset_id": ds, "k_sel_saved_CQ": ks, "k_sel_1sample_CQ": k1,
                        "gap_CQ_saved_9_minus_ksel": sq[9] - sq[ks], "init_absdiff_CQ_at_9": spread9,
                        "init_absdiff_CQ_at_ksel": spreadk})
        for k in tg.B_K_CANDS:
            rec = {"cond": cond, "k": k, "n_datasets": len(dsets)}
            for c in tcols:
                dv = [tc[(cond, ds, k)][c] - tc[(cond, ds, 9)][c] for ds in dsets
                      if c in tc[(cond, ds, k)] and c in tc[(cond, ds, 9)]]
                if dv:
                    rec[f"d_{c}_vs9_median"] = float(np.median(dv))
                    rec[f"d_{c}_vs9_n"] = len(dv)
            # increments K-1 -> K (median over datasets), the 'per added dimension' balance
            if k > 1:
                for c in ("m2_lnpZ", "m2_lnpX", "m2_lnpY", "penalty", "C_Q_1sample", "C_Q_saved"):
                    dv = [tc[(cond, ds, k)][c] - tc[(cond, ds, k - 1)][c] for ds in dsets]
                    rec[f"inc_{c}_from_k-1_median"] = float(np.median(dv))
            qv = [x["q_rel_change_last"] for x in terms if x["cond"] == cond and x["k"] == k and x["q_rel_change_last"] != ""]
            rec["q_rel_change_last_median"] = float(np.median(qv))
            rec["q_rel_change_last_max"] = float(np.max(qv))
            rec["mstep_q_decreases_total"] = int(sum(x["mstep_q_decreases"] for x in terms if x["cond"] == cond and x["k"] == k))
            rec["mc_residual_abs_median"] = float(np.median([abs(x["mc_residual_C_Q_saved_minus_1sample"])
                                                             for x in terms if x["cond"] == cond and x["k"] == k]))
            dsum.append(rec)
    return vsum, dsum, sel


def _sel(d):
    m = min(d.values())
    return min(k for k, v in d.items() if v <= m + 1e-9)


# ============================================================================ figures
def figures(cells, vsum, dsum):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = OUT / "figures"
    fd.mkdir(exist_ok=True)
    col = {"bernoulli": "#2a78d6", "poisson": "#eb6834"}
    # A: per-dataset w_hat vs log score, and class-wise log scores
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, rate in enumerate(tg.A_RATES):
        ax = axes[i]
        for fam in tg.A_FAMS:
            v = [c for c in cells if c["rate"] == rate and c["family_x_fit"] == fam]
            ax.scatter([c["w_hat"] for c in v], [c["plugin_last_log_score"] for c in v], s=18, color=col[fam],
                       label=f"{fam} fit")
        v = [c for c in cells if c["rate"] == rate and c["family_x_fit"] == "bernoulli"]
        ax.axhline(np.mean([c["const_log_score"] for c in v]), color="#52514e", lw=1, ls="--",
                   label="constant (train rate)")
        ax.axhline(np.mean([c["truth_log_score"] for c in v]), color="#52514e", lw=1, ls=":",
                   label="true parameters")
        ax.set_xlabel("estimated w (true 1.0)")
        ax.set_title(f"{rate} Y")
        if i == 0:
            ax.set_ylabel("held-out mean log score (plug-in, Z_last)")
        ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    xs = np.arange(4)
    for j, fam in enumerate(tg.A_FAMS):
        v = [c for c in cells if c["rate"] == "sparse" and c["family_x_fit"] == fam]
        ax.bar(xs[:2] + (j - 0.5) * 0.38, [np.mean([c["plugin_last_p_mean_pos"] for c in v]),
                                           np.mean([c["plugin_last_p_mean_neg"] for c in v])],
               width=0.36, color=col[fam], label=f"{fam} fit")
        ax.bar(xs[2:] + (j - 0.5) * 0.38, [np.mean([c["plugin_last_auc"] for c in v]),
                                           np.mean([c["plugin_last_calib_slope"] for c in v]) / 2],
               width=0.36, color=col[fam])
    ax.set_xticks(xs, ["p | y=1", "p | y=0", "AUC", "calib slope / 2"])
    ax.set_title("sparse Y: means over 20 datasets")
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Exp A post-hoc diagnostics (lineage E-scalar; 2-init mean per dataset; exploratory)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fd / "figA_prediction_diagnostics.png", dpi=130)
    plt.close(fig)

    ccol = {"G01": "#2a78d6", "GM": "#eb6834", "P": "#6250d6"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ks = list(tg.B_K_CANDS)
    ax = axes[0]
    for cond in ("G01", "GM"):
        v = {r["k"]: r for r in vsum if r["cond"] == cond}
        ax.plot(ks, [v[k]["sigma2_mean_over_true_median"] for k in ks], marker="o", color=ccol[cond], label=cond)
    ax.axhline(1, color="#52514e", lw=1)
    ax.axvline(9, color="#8c8b86", lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("candidate K")
    ax.set_ylabel("estimated var / true var (median)")
    ax.set_title("Gaussian-X estimated variance")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    for cond in tg.B_CONDS:
        v = {r["k"]: r for r in vsum if r["cond"] == cond}
        ax.plot(ks, [v[k]["xinfo_est_trace_per_dim_median"] for k in ks], marker="o", color=ccol[cond],
                label=f"{cond}: X info")
    v = {r["k"]: r for r in vsum if r["cond"] == "GM"}
    ax.plot(ks, [v[k]["yinfo_est_trace_per_dim_median"] for k in ks], color="#52514e", ls="--",
            label="GM: Y info")
    ax.set_yscale("log")
    ax.axvline(9, color="#8c8b86", lw=1, ls="--")
    ax.set_xlabel("candidate K")
    ax.set_ylabel("estimated information trace / K (median)")
    ax.set_title("Estimated information per latent dimension")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    for cond in tg.B_CONDS:
        v = {r["k"]: r for r in dsum if r["cond"] == cond}
        ax.plot(ks[1:], [v[k]["inc_m2_lnpX_from_k-1_median"] + v[k]["inc_m2_lnpY_from_k-1_median"] for k in ks[1:]],
                marker="o", color=ccol[cond], label=f"{cond}: -2(dlnpX + dlnpY)")
    cost = [v[k]["inc_m2_lnpZ_from_k-1_median"] + v[k]["inc_penalty_from_k-1_median"] for k in ks[1:]]
    ax.plot(ks[1:], [-c for c in cost], color="#52514e", ls="--", label="-(prior + penalty increment)")
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xlabel("K (increment from K-1)")
    ax.set_ylabel("change in C_Q terms (median)")
    ax.set_title("C_Q: fit gain vs. per-dimension cost")
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Exp B post-hoc diagnostics (1-sample term split at Z_last; exploratory)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fd / "figB_variance_information_cq_terms.png", dpi=130)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    fits, parA, cells, summA = diag_A()
    parB, terms = diag_B()
    vsum, dsum, sel = summarize_B(parB, terms)
    wr(OUT / "prediction_diagnostics.csv", fits)
    wr(OUT / "a_dataset_condition_means.csv", cells)
    wr(OUT / "a_paired_summary.csv", summA)
    wr(OUT / "parameter_diagnostics.csv", parA + parB)
    wr(OUT / "b_variance_information_summary.csv", vsum)
    wr(OUT / "cq_term_decomposition.csv", terms)
    wr(OUT / "cq_term_delta_summary.csv", dsum)
    wr(OUT / "b_selection_check.csv", sel)
    n_match = sum(s["k_sel_saved_CQ"] == s["k_sel_1sample_CQ"] for s in sel)
    check("B: selection by saved C_Q vs 1-sample C_Q (recorded, not a gate)", True, f"{n_match}/{len(sel)} equal")
    # saved-C_Q selection equals the primary analysis
    prim = {r["dataset_id"]: r for r in rd(BASE / "expB/analysis/per_dataset_selection.csv")}
    check("B: saved-C_Q selection = primary analysis k_Q", all(str(s["k_sel_saved_CQ"]) == prim[s["dataset_id"]]["k_Q"]
                                                             for s in sel))
    figures(cells, vsum, dsum)
    wr(OUT / "consistency_checks.csv", CHECKS, ["check", "pass", "detail"])
    print(f"{sum(c['pass'] for c in CHECKS)}/{len(CHECKS)} checks pass")
    for c in CHECKS:
        print(("ok   " if c["pass"] else "FAIL ") + c["check"] + " | " + c["detail"])


if __name__ == "__main__":
    main()
