"""AF7 (exploratory, post-hoc): why does the column-wise Laplace selector of AF5/AF6 move true-Poisson
columns to CMP(nu=2)?  No MCEM, no new data, no new seeds; nothing in AF1-AF6 is modified.

Part 1 (--part1): saved CSVs only (AF6 full + AF5 full, all anchors).
  Per candidate m in {P = Poisson, C = CMP2}, with c = (nK/2) log 2 pi:
      L-point  f_L = argmax_f S(f),            S(f) = Phi(Zhat(f), f) + c - 1/2 log|H(f)|
      J-point  f_J = argmax_f Phi(Zhat(f), f)  (joint MAP in (Z, f_j))
      S_L = S(f_L), S_J = S(f_J), phi_L = Phi at f_L, phi_J = Phi at f_J,
      h_L = -1/2 log|H(f_L)|, h_J = S_J - phi_J - c.
  Column differences (C - P):
      Delta_L = S_L(C) - S_L(P)            (AF5/AF6 Laplace selector, method A)
      Delta_J = S_J(C) - S_J(P)            (AF5 "joint-MAP variant", method E)
      Delta_profile = Delta_L - Delta_J = gain(C) - gain(P),  gain(m) = S_L(m) - S_J(m) >= 0
      Delta_phiJ = phi_J(C) - phi_J(P)     (pure joint-MAP log joint, no volume term)
      Delta_hJ = Delta_J - Delta_phiJ ; Delta_phiL, Delta_hL (recorded at f_L), Delta_phiL = target + other.
  Labels are used only to describe (accuracy, transitions); no threshold is fitted.

Part 2 (--part2): diagnostic recomputation on the stored AF6 fit states (no MCEM), independent code path
  (audit_af6_af5_independent.Joint): for every AF6 column, for both candidates at f_L:
      Zhat, column-j Pearson dispersion, Z displacement Zhat(C) - Zhat(P), the per-block split of Delta_phiL;
  fixed-Z plug-in column log-likelihood at the anchor mode (Z not re-optimised) for both candidates;
  data features of x_j; oracle information: sum_i KL between the true conditional law and the moment-matched
  other family at the TRUE eta (Z, F of the generator), both directions.

Writes to expfam/results/auto_family_z_recovery/af7_cmp_bias_analysis_20261114/.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

ROOT = HERE.parent.parent
RES = ROOT / "results" / "auto_family_z_recovery"
AF5 = RES / "af5_local_laplace_selection_20261113" / "full"
AF6 = RES / "af6_laplace_confirmation_20261114" / "full"
OUT = RES / "af7_cmp_bias_analysis_20261114"
N, K = 75, 3
C0 = 0.5 * N * K * math.log(2 * math.pi)
P, C = "poisson", "cmp_nu2p0"


# --------------------------------------------------------------------------
# Part 1
# --------------------------------------------------------------------------

def columns_frame(cs: pd.DataFrame, sel: pd.DataFrame, study: str) -> pd.DataFrame:
    cs = cs[cs.status == "OK"].copy()
    cs["h_L"] = cs.neg_half_logdet
    cs["phi_J"] = cs.phi_jointmap_variant
    cs["S_J"] = cs.S_jointmap_variant
    cs["h_J"] = cs.S_J - cs.phi_J - C0
    cs["gain"] = cs.S - cs.S_J
    cs["gain_phi"] = cs.phi - cs.phi_J
    cs["gain_h"] = cs.h_L - cs.h_J
    cs["phi_other"] = cs.phi - cs.phi_x_target_col
    keys = ["replicate", "regime", "anchor", "column"]
    v = ["S", "S_J", "phi", "phi_J", "h_L", "h_J", "gain", "gain_phi", "gain_h", "phi_x_target_col", "phi_other",
         "phi_prior", "phi_x_gauss", "phi_x_count_total", "phi_y_train", "f_star_norm", "jointmap_f_norm"]
    w = cs.pivot_table(index=keys, columns="candidate", values=v)
    w = w[[c for c in w.columns if c[1] in (P, C)]].dropna()
    d = pd.DataFrame(index=w.index)
    d["Delta_L"] = w["S", C] - w["S", P]
    d["Delta_J"] = w["S_J", C] - w["S_J", P]
    d["Delta_profile"] = d.Delta_L - d.Delta_J
    d["gain_P"], d["gain_C"] = w["gain", P], w["gain", C]
    d["gain_phi_P"], d["gain_phi_C"] = w["gain_phi", P], w["gain_phi", C]
    d["gain_h_P"], d["gain_h_C"] = w["gain_h", P], w["gain_h", C]
    d["Delta_phiJ"] = w["phi_J", C] - w["phi_J", P]
    d["Delta_hJ"] = w["h_J", C] - w["h_J", P]
    d["Delta_phiL"] = w["phi", C] - w["phi", P]
    d["Delta_hL"] = w["h_L", C] - w["h_L", P]
    d["Delta_phiL_target"] = w["phi_x_target_col", C] - w["phi_x_target_col", P]
    d["Delta_phiL_other"] = w["phi_other", C] - w["phi_other", P]
    for b in ("phi_prior", "phi_x_gauss", "phi_y_train"):
        d[f"Delta_{b}"] = w[b, C] - w[b, P]
    d["Delta_phi_other_count_cols"] = (w["phi_x_count_total", C] - w["phi_x_target_col", C]) \
        - (w["phi_x_count_total", P] - w["phi_x_target_col", P])
    for m, t in ((P, "P"), (C, "C")):
        d[f"fL_norm_{t}"], d[f"fJ_norm_{t}"] = w["f_star_norm", m], w["jointmap_f_norm", m]
    d = d.reset_index()
    s = sel.copy()
    s["column"] = s.column.astype(int)
    keep = [c for c in ("true_model", "anchor_family_j", "status", "q_margin_cmp_minus_pois", "q_selected", "q_status",
                        "delta_ti", "delta_ti_mc_se", "true_loading_norm", "is_readable") if c in s.columns]
    d = d.merge(s[keys + keep], on=keys, how="left", validate="1:1")
    d.insert(0, "study", study)
    d["truth_C"] = d.true_model == C
    for crit, col in (("L", "Delta_L"), ("J", "Delta_J"), ("phiJ", "Delta_phiJ"), ("phiL", "Delta_phiL"),
                      ("Q", "q_margin_cmp_minus_pois"), ("TI", "delta_ti")):
        if col in d:
            x = pd.to_numeric(d[col], errors="coerce")
            d[f"ok_{crit}"] = np.where(x.notna(), (x > 0) == d.truth_C, np.nan)
    return d


def describe(d: pd.DataFrame, by) -> pd.DataFrame:
    rows = []
    for k, g in d.groupby(by):
        r = dict(zip(by, k if isinstance(k, tuple) else (k,)))
        r["columns"] = len(g)
        for c in ("ok_L", "ok_J", "ok_phiJ", "ok_phiL", "ok_Q", "ok_TI"):
            if c in g:
                x = pd.to_numeric(g[c], errors="coerce")
                r[f"acc_{c[3:]}"] = float(x.mean()) if x.notna().any() else np.nan
        for c in ("Delta_L", "Delta_J", "Delta_profile", "gain_P", "gain_C", "Delta_phiJ", "Delta_hJ", "Delta_phiL",
                  "Delta_hL", "Delta_phiL_target", "Delta_phiL_other", "Delta_phi_prior", "Delta_phi_x_gauss",
                  "Delta_phi_other_count_cols", "Delta_phi_y_train", "q_margin_cmp_minus_pois"):
            if c in g:
                r[f"med_{c}"] = float(pd.to_numeric(g[c], errors="coerce").median())
        r["share_profile_toward_C"] = float((g.Delta_profile > 0).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def transitions(d: pd.DataFrame, a: str, b: str, by) -> pd.DataFrame:
    rows = []
    for k, g in d.groupby(by):
        r = dict(zip(by, k))
        oa, ob = pd.to_numeric(g[f"ok_{a}"]), pd.to_numeric(g[f"ok_{b}"])
        r.update({"from": a, "to": b, "columns": len(g),
                  "right_to_wrong": int(((oa == 1) & (ob == 0)).sum()),
                  "wrong_to_right": int(((oa == 0) & (ob == 1)).sum()),
                  "both_right": int(((oa == 1) & (ob == 1)).sum()), "both_wrong": int(((oa == 0) & (ob == 0)).sum())})
        rows.append(r)
    return pd.DataFrame(rows)


def part1() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    d6 = columns_frame(pd.read_csv(AF6 / "candidate_scores.csv"),
                       pd.read_csv(AF6 / "column_selection.csv", keep_default_na=False, na_values=[""]), "AF6")
    d5 = columns_frame(pd.read_csv(AF5 / "candidate_scores.csv"),
                       pd.read_csv(AF5 / "column_selection.csv", keep_default_na=False, na_values=[""]), "AF5")
    d = pd.concat([d6, d5], ignore_index=True)
    d.to_csv(OUT / "p1_column_decomposition.csv", index=False)
    by = ["study", "anchor", "regime", "true_model"]
    describe(d, by).to_csv(OUT / "p1_summary_by_truth.csv", index=False)
    tr = pd.concat([transitions(d, a, b, by) for a, b in (("J", "L"), ("phiJ", "J"), ("Q", "L"), ("Q", "J"),
                                                          ("L", "TI"))], ignore_index=True)
    tr.to_csv(OUT / "p1_transitions.csv", index=False)
    return d


# --------------------------------------------------------------------------
# Part 2 (diagnostic recomputation on stored AF6 states, no MCEM)
# --------------------------------------------------------------------------

def cmp2_logpmf(x, eta):
    import audit_af6_af5_independent as aud
    from scipy.special import gammaln
    return x * eta - 2 * gammaln(x + 1) - aud.cmp2_A(eta)


def pois_logpmf(x, eta):
    from scipy.special import gammaln
    return x * eta - gammaln(x + 1) - np.exp(eta)


def moments(fam, eta):
    import audit_af6_af5_independent as aud
    if fam == P:
        return np.exp(eta), np.exp(eta)
    return aud.cmp2_dA(eta), aud.cmp2_d2A(eta)


def kl_matched(true_fam, eta_true, kmax=400):
    """sum_i KL(p_true(.|eta_i) || q_i), q_i = other family with the same mean (per-observation best case)."""
    from scipy.optimize import brentq
    x = np.arange(kmax)[:, None]
    tot = 0.0
    for e in np.atleast_1d(eta_true):
        lp = (pois_logpmf if true_fam == P else cmp2_logpmf)(x[:, 0], e)
        mu = float(np.sum(np.exp(lp) * x[:, 0]))
        other = C if true_fam == P else P
        if other == P:
            eo = math.log(max(mu, 1e-300))
        else:
            import audit_af6_af5_independent as aud
            eo = brentq(lambda t: aud.cmp2_dA(np.array([t]))[0] - mu, -60, 60)
        lq = (pois_logpmf if other == P else cmp2_logpmf)(x[:, 0], eo)
        p = np.exp(lp)
        tot += float(np.sum(p * (lp - lq)))
    return tot


def fixed_z_loglik(fam, x, Z):
    """max_f sum_i log p_fam(x_i | f'z_i) with Z fixed (Newton on the concave column log-likelihood)."""
    f = np.zeros(Z.shape[1])
    for _ in range(200):
        eta = Z @ f
        mu, var = moments(fam, eta)
        g = Z.T @ (x - mu)
        H = (Z * var[:, None]).T @ Z + 1e-12 * np.eye(len(f))
        step = np.linalg.solve(H, g)
        t = 1.0
        ll0 = np.sum((pois_logpmf if fam == P else cmp2_logpmf)(x, eta))
        while t > 1e-8:
            fn = f + t * step
            if np.sum((pois_logpmf if fam == P else cmp2_logpmf)(x, Z @ fn)) >= ll0 - 1e-12:
                break
            t *= 0.5
        f = fn
        if np.max(np.abs(g)) < 1e-9:
            break
    eta = Z @ f
    return float(np.sum((pois_logpmf if fam == P else cmp2_logpmf)(x, eta))), f


def part2() -> None:
    import audit_af6_af5_independent as aud
    import run_af6_laplace_confirmation as a6
    OUT.mkdir(parents=True, exist_ok=True)
    fm = pd.read_csv(AF6 / "fit_manifest.csv")
    cs = pd.read_csv(AF6 / "candidate_scores.csv")
    cs = cs[cs.status == "OK"]
    rows, t0 = [], time.time()
    for r in range(a6.N_DATASETS):
        ds, masks = a6.dataset(r)
        X, Y = np.asarray(ds.X, float), np.asarray(ds.Y, float)
        Zt, Ft = np.asarray(ds.Z, float), np.asarray(ds.F, float)
        for g in a6.REGIMES:
            m = fm[(fm.replicate == a6.label(r)) & (fm.regime == g)].iloc[0]
            with np.load(m.state_file) as z:
                st = {k: np.asarray(z[k]) for k in z.files}
            assign = m.assignment.split("|")
            w0, w, var_z = (float(v) for v in st["scalars"])
            Za = np.asarray(st["Zmode"], float)
            for j in a6.COUNT_COLS:
                x = X[:, j]
                tm = a6.TRUE_MODELS[j]
                eta_true = Zt @ Ft[j]
                mu_t, var_t = moments(tm, eta_true)
                row = {"replicate": a6.label(r), "regime": g, "column": j, "true_model": tm,
                       "anchor_family_j": assign[j], "x_mean": x.mean(), "x_var_over_mean": x.var() / max(x.mean(), 1e-12),
                       "x_zero_share": float((x == 0).mean()), "x_max": x.max(), "x_share_le1": float((x <= 1).mean()),
                       "true_loading_norm": float(np.linalg.norm(Ft[j])), "true_eta_sd": float(eta_true.std()),
                       "pearson_disp_true": float(np.mean((x - mu_t) ** 2 / var_t)),
                       "oracle_kl_truth_vs_matched_other": kl_matched(tm, eta_true)}
                for fam, t in ((P, "P"), (C, "C")):
                    ll, f = fixed_z_loglik(fam, x, Za)
                    row[f"fixedZ_anchor_ll_{t}"] = ll
                    llt, _ = fixed_z_loglik(fam, x, Zt)
                    row[f"fixedZ_true_ll_{t}"] = llt
                    # under the (P) model fitted with Z fixed at the anchor mode: Pearson dispersion w.r.t. Poisson var
                    if fam == P:
                        mu = np.exp(Za @ f)
                        row["pearson_disp_anchorZ_P"] = float(np.mean((x - mu) ** 2 / mu))
                Zh = {}
                for fam, t in ((P, "P"), (C, "C")):
                    c = cs[(cs.replicate == a6.label(r)) & (cs.regime == g) & (cs.column == j) & (cs.candidate == fam)]
                    if c.empty:
                        continue
                    f = aud.parse_f(c.iloc[0].f_star)
                    fams = list(assign)
                    fams[j] = fam
                    F = np.array(st["F"], float)
                    F[j] = f
                    J = aud.Joint(X, Y, masks[g], fams, F, np.diag(st["sigma"]), w0, w, var_z)
                    L = aud.laplace_S(J, Za)
                    Zh[t] = L["Z"]
                    eta = L["Z"] @ f
                    mu, var = moments(fam, eta)
                    row[f"S_indep_{t}"] = L["S"]
                    row[f"pearson_disp_ZL_{t}"] = float(np.mean((x - mu) ** 2 / var))
                    muP = np.exp(eta) if fam == P else mu
                    row[f"pearson_vs_poisson_ZL_{t}"] = float(np.mean((x - muP) ** 2 / np.maximum(muP, 1e-12)))
                    row[f"zshift_from_anchor_{t}"] = float(np.linalg.norm(L["Z"] - Za) / np.linalg.norm(Za))
                    row[f"rmse_eta_true_{t}"] = float(np.sqrt(np.mean((moments(fam, eta)[0] - mu_t) ** 2)))
                if len(Zh) == 2:
                    row["zshift_C_minus_P"] = float(np.linalg.norm(Zh["C"] - Zh["P"]) / np.linalg.norm(Zh["P"]))
                rows.append(row)
            print(f"  {a6.label(r)} {time.time() - t0:.0f}s", flush=True)
    p2 = pd.DataFrame(rows)
    p2["Delta_fixedZ_anchor"] = p2.fixedZ_anchor_ll_C - p2.fixedZ_anchor_ll_P
    p2["Delta_fixedZ_true"] = p2.fixedZ_true_ll_C - p2.fixedZ_true_ll_P
    p2.to_csv(OUT / "p2_column_diagnostics.csv", index=False)


# --------------------------------------------------------------------------
# Part 3 (diagnostic): the same local criteria with the TRUE theta as the anchor
#   theta_{-j} = generator values (F, sigma^2 = 1, w0 = -1, w = 1, var_z = 1, true families of the other
#   columns); only (m, f_j, Z) vary, exactly as in AF5/AF6.  Separates "anchor theta estimated by MCEM"
#   from "the criterion itself".  No MCEM, no new data.
# --------------------------------------------------------------------------

def _local_true_theta(ds, masks, regime, j, fam):
    import audit_af6_af5_independent as aud
    import run_af6_laplace_confirmation as a6
    fams = list(a6.TRUE_MODELS)
    fams[j] = fam
    s2 = np.ones(len(fams))
    return aud.Joint(np.asarray(ds.X, float), np.asarray(ds.Y, float), masks[regime], fams,
                     np.asarray(ds.F, float).copy(), s2, -1.0, 1.0, 1.0)


def _opt_local(J, j, x, Zstart, starts):
    """L-point (max S over f_j, FD gradient) and J-point (max Phi(Zhat(f), f), envelope gradient)."""
    import audit_af6_af5_independent as aud
    from scipy.optimize import minimize
    warm = {"Z": Zstart.copy()}

    def S_of(f):
        J.F[j] = f
        L = aud.laplace_S(J, warm["Z"])
        if L["grad_inf"] > 1e-6 or L["min_eig"] <= 0:
            return -1e300, L
        warm["Z"] = L["Z"]
        return L["S"], L

    def negS(f):
        return -S_of(np.asarray(f, float))[0]

    def jac(f, h=1e-4):
        g = np.zeros(len(f))
        for a in range(len(f)):
            e = np.zeros(len(f))
            e[a] = h
            g[a] = (negS(f + e) - negS(f - e)) / (2 * h)
        return g

    def negphi(f):
        J.F[j] = np.asarray(f, float)
        Zh, _, gi = J.mode(warm["Z"])
        warm["Z"] = Zh
        fam = J.fams[j]
        eta = Zh @ J.F[j]
        mu = np.exp(eta) if fam == P else __import__("audit_af6_af5_independent").cmp2_dA(eta)
        return -J.phi(Zh), -((x - mu) @ Zh)

    best = None
    for name, f0 in starts.items():
        warm["Z"] = Zstart.copy()
        res = minimize(negS, np.asarray(f0, float), jac=jac, method="BFGS", options={"gtol": 1e-4, "maxiter": 200})
        if best is None or -res.fun > best[0]:
            best = (-res.fun, np.asarray(res.x, float), name)
    S_L, fL, start = best
    warm["Z"] = Zstart.copy()
    _, Lf = S_of(fL)
    phi_L = J.phi(Lf["Z"])
    tgt_L = float(np.sum(aud.col_terms(J.fams[j], x, Lf["Z"] @ fL)[0]))
    gfd = float(np.max(np.abs(jac(fL))))
    warm["Z"] = Lf["Z"].copy()
    rj = minimize(negphi, fL, jac=True, method="BFGS", options={"gtol": 1e-7, "maxiter": 500})
    fJ = np.asarray(rj.x, float)
    warm["Z"] = Lf["Z"].copy()
    S_J, LJ = S_of(fJ)
    phi_J = J.phi(LJ["Z"])
    return {"S_L": S_L, "phi_L": phi_L, "h_L": S_L - phi_L - C0, "target_L": tgt_L, "fL_norm": float(np.linalg.norm(fL)),
            "fL": "|".join(f"{v:.8g}" for v in fL), "best_start": start, "L_fd_grad_inf": gfd,
            "S_J": S_J, "phi_J": phi_J, "h_J": S_J - phi_J - C0, "fJ_norm": float(np.linalg.norm(fJ)),
            "J_grad_inf": float(np.max(np.abs(rj.jac))) if rj.jac is not None else np.nan}


def part3(shard: str | None) -> None:
    import run_af6_laplace_confirmation as a6
    OUT.mkdir(parents=True, exist_ok=True)
    reps = list(range(a6.N_DATASETS))
    tag = ""
    if shard:
        i, n = (int(v) for v in shard.split("/"))
        reps = reps[i::n]
        tag = f"_{i}of{n}"
    path = OUT / f"p3_true_theta_progress{tag}.jsonl"
    done = set()
    if path.exists():
        done = {(d["replicate"], d["regime"], d["column"], d["candidate"])
                for d in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    t0 = time.time()
    for r in reps:
        ds, masks = a6.dataset(r)
        Zt, Ft = np.asarray(ds.Z, float), np.asarray(ds.F, float)
        for g in a6.REGIMES:
            for j in a6.COUNT_COLS:
                x = np.asarray(ds.X, float)[:, j]
                for fam in (P, C):
                    key = (a6.label(r), g, j, fam)
                    if key in done:
                        continue
                    J = _local_true_theta(ds, masks, g, j, fam)
                    _, fplug = fixed_z_loglik(fam, x, Zt)
                    try:
                        rec = _opt_local(J, j, x, Zt, {"true_f": Ft[j], "plugin_trueZ": fplug})
                        rec["status"] = "OK"
                    except Exception as exc:                       # recorded, never retried
                        rec = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"[:200]}
                    rec.update(dict(zip(("replicate", "regime", "column", "candidate"), key)),
                               true_model=a6.TRUE_MODELS[j])
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
            print(f"  {a6.label(r)} {g} {time.time() - t0:.0f}s", flush=True)


def part3_collect() -> pd.DataFrame:
    rows = []
    for p in sorted(OUT.glob("p3_true_theta_progress*.jsonl")):
        rows += [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()]
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "p3_true_theta_candidates.csv", index=False)
    ok = c[c.status == "OK"]
    keys = ["replicate", "regime", "column", "true_model"]
    w = ok.pivot_table(index=keys, columns="candidate", values=["S_L", "S_J", "phi_L", "phi_J", "h_L", "h_J", "target_L"])
    d = pd.DataFrame(index=w.index)
    d["Delta_L"] = w["S_L", C] - w["S_L", P]
    d["Delta_J"] = w["S_J", C] - w["S_J", P]
    d["Delta_profile"] = d.Delta_L - d.Delta_J
    d["Delta_phiJ"] = w["phi_J", C] - w["phi_J", P]
    d["Delta_hJ"] = w["h_J", C] - w["h_J", P]
    d["Delta_phiL"] = w["phi_L", C] - w["phi_L", P]
    d["Delta_hL"] = w["h_L", C] - w["h_L", P]
    d["Delta_phiL_target"] = w["target_L", C] - w["target_L", P]
    d = d.reset_index()
    d["truth_C"] = d.true_model == C
    for crit in ("L", "J", "phiJ"):
        d[f"ok_{crit}"] = (d[f"Delta_{crit}"] > 0) == d.truth_C
    d.to_csv(OUT / "p3_true_theta_columns.csv", index=False)
    return d


# --------------------------------------------------------------------------
# Summary (exploratory): accuracy by truth, AUC (rank, label-descriptive only; no threshold is fitted),
# dataset-cluster bootstrap CIs for AUC differences, feature tables.
# --------------------------------------------------------------------------

def _auc(x, y):
    from scipy.stats import mannwhitneyu
    pos, neg = x[y], x[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def summary() -> None:
    d1 = pd.read_csv(OUT / "p1_column_decomposition.csv")
    d1 = d1[d1.study == "AF6"]
    p2 = pd.read_csv(OUT / "p2_column_diagnostics.csv")
    keys = ["replicate", "regime", "column"]
    m = p2.merge(d1[keys + ["Delta_L", "Delta_J", "Delta_profile", "Delta_phiJ", "q_margin_cmp_minus_pois", "delta_ti"]],
                 on=keys, how="left")
    crit = {"fixedZ_trueZ": "Delta_fixedZ_true", "fixedZ_anchorZmode": "Delta_fixedZ_anchor",
            "Q_sameanchor": "q_margin_cmp_minus_pois", "phiJ_jointMAP_logjoint": "Delta_phiJ",
            "J_S_at_fJ": "Delta_J", "L_laplace": "Delta_L", "TI_at_fL": "delta_ti"}
    p3 = OUT / "p3_true_theta_columns.csv"
    if p3.exists():
        t = pd.read_csv(p3)[keys + ["Delta_L", "Delta_J", "Delta_phiJ"]] \
            .rename(columns={"Delta_L": "tt_L", "Delta_J": "tt_J", "Delta_phiJ": "tt_phiJ"})
        m = m.merge(t, on=keys, how="left")
        crit.update({"trueTheta_L": "tt_L", "trueTheta_J": "tt_J", "trueTheta_phiJ": "tt_phiJ"})
    m.to_csv(OUT / "summary_column_table.csv", index=False)
    rng = np.random.default_rng(20261117)
    reps = sorted(m.replicate.unique())
    rows = []
    for g in ("dense", "sparse"):
        s = m[m.regime == g]
        tc = (s.true_model == C).to_numpy()
        boots = {k: [] for k in crit}
        idx = {r: np.flatnonzero(s.replicate.to_numpy() == r) for r in reps}
        for _ in range(2000):
            take = np.concatenate([idx[r] for r in rng.choice(reps, len(reps))])
            for k, c in crit.items():
                x = s[c].to_numpy()[take]
                y = tc[take]
                ok = np.isfinite(x)
                boots[k].append(_auc(x[ok], y[ok]))
        for k, c in crit.items():
            x = s[c].to_numpy()
            ok = np.isfinite(x)
            rows.append({"regime": g, "criterion": k, "columns": int(ok.sum()),
                         "acc_true_poisson": float(np.mean(x[ok & ~tc] <= 0)), "acc_true_cmp2": float(np.mean(x[ok & tc] > 0)),
                         "acc_all": float(np.mean((x[ok] > 0) == tc[ok])),
                         "median_true_poisson": float(np.median(x[ok & ~tc])), "median_true_cmp2": float(np.median(x[ok & tc])),
                         "auc": _auc(x[ok], tc[ok]),
                         "auc_ci_low": float(np.nanquantile(boots[k], .025)), "auc_ci_high": float(np.nanquantile(boots[k], .975)),
                         "auc_minus_L": float(np.nanmean(np.array(boots[k]) - np.array(boots["L_laplace"]))),
                         "auc_minus_L_ci_low": float(np.nanquantile(np.array(boots[k]) - np.array(boots["L_laplace"]), .025)),
                         "auc_minus_L_ci_high": float(np.nanquantile(np.array(boots[k]) - np.array(boots["L_laplace"]), .975))})
    pd.DataFrame(rows).to_csv(OUT / "summary_criteria_accuracy_auc.csv", index=False)
    feats = ["x_mean", "x_var_over_mean", "x_zero_share", "x_max", "true_loading_norm", "pearson_disp_true",
             "pearson_disp_anchorZ_P", "pearson_disp_ZL_P", "oracle_kl_truth_vs_matched_other", "zshift_C_minus_P",
             "Delta_fixedZ_true", "Delta_fixedZ_anchor", "Delta_L", "Delta_profile", "q_margin_cmp_minus_pois"]
    m["L_correct"] = (m.Delta_L > 0) == (m.true_model == C)
    f = m.groupby(["regime", "true_model", "L_correct"])[feats].median().reset_index()
    f.insert(3, "columns", m.groupby(["regime", "true_model", "L_correct"]).size().to_numpy())
    f.to_csv(OUT / "summary_features_by_L_correct.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--part1" in argv:
        part1()
    if "--part2" in argv:
        part2()
    if "--part3" in argv:
        sh = argv[argv.index("--shard") + 1] if "--shard" in argv else None
        part3(sh)
    if "--part3-collect" in argv:
        part3_collect()
    if "--summary" in argv:
        summary()
        import run_af6_laplace_confirmation as a6
        info = a6.runinfo({"experiment": "AF7", "script": "expfam/src/experimental/analyze_af7_cmp_bias.py",
                           "note": "written at --summary time; parts 1-3 ran earlier in this branch at the same "
                                   "git head (scripts uncommitted at run time); exploratory, 0 MCEM, no new data"})
        info.pop("protocol_sha256", None)
        (OUT / "runinfo_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
