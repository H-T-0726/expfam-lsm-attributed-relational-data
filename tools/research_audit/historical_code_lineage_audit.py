"""Historical code-lineage audit: why did K = 1..9 succeed before but K_GEN = 5 fail now?

Re-analysis only (no MCEM fit).  Reads primary CSVs, the committed Fig. 3
vector-vertex CSV, and the generator code; recomputes selection outcomes,
selection slack, signal-strength summaries and finite-difference checks of the
E-step formulas.  Curated comparison tables (with file:line references) are
written from the dictionaries in this file.

Outputs: reports/k_selection_theory/historical_code_lineage_audit_20261106/*.csv
Lineage tags: every row names its estimator lineage; nothing here is
manuscript evidence (Human Gate).  C_Q is not called Schwarz BIC; C_Lap is not
an exact marginal likelihood.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/k_selection_theory/historical_code_lineage_audit_20261106"
sys.path.insert(0, str(ROOT / "reproduction/src"))
sys.path.insert(0, str(ROOT / "expfam/src"))
sys.path.insert(0, str(ROOT / "expfam/src/experimental"))

from data_generator import set_true_params  # noqa: E402  (reproduction / MATLAB port)
from data_generator_expfam import generate_dual_data  # noqa: E402  (early ExpFam)
from data_generator_canonical import (  # noqa: E402
    f_scale_for_row_norm, generate_canonical_data)

LN150 = math.log(150.0)
RES = ROOT / "expfam/results"


def t_paper(k: int, d: int = 15) -> float:
    """Eq.(26) parameter count (k+1)d - k(k-1)/2."""
    return (k + 1) * d - k * (k - 1) / 2


def write(name: str, df: pd.DataFrame) -> None:
    path = OUT / name
    if path.exists():
        sys.exit(f"refusing to overwrite {path}")
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. selection outcomes and slack, per series
# ---------------------------------------------------------------------------

def slack_rows(series, lineage, criterion, fam, n, k_true, unit, curve: dict, source, note=""):
    """curve: {k: score}; lower is better.  slack(a->b) = C(a) - C(b) (>0 favours b)."""
    ks = sorted(k for k, v in curve.items() if v is not None and np.isfinite(v))
    khat = min(ks, key=lambda k: (curve[k], k))
    def sl(a, b):
        return curve[a] - curve[b] if a in curve and b in curve and curve[a] is not None and curve[b] is not None else np.nan
    row = dict(series=series, lineage=lineage, criterion=criterion, family_x=fam[0], family_y=fam[1], n=n,
               k_true=k_true, unit=unit, candidate_k=f"{min(ks)}..{max(ks)}", k_hat=khat, correct=int(khat == k_true),
               slack_under=sl(k_true - 1, k_true), slack_over=sl(k_true + 1, k_true),
               slack_4_5=sl(4, 5), slack_6_7=sl(6, 7), slack_8_9=sl(8, 9), source=source, note=note)
    row["slack_under_per_node"] = row["slack_under"] / n
    row["slack_over_per_node"] = row["slack_over"] / n
    return row


def series_paper_fig3():
    path = OUT / "paper_fig3_vector_vertices.csv"
    df = pd.read_csv(path, comment="#")
    rows, implied = [], []
    for ks, g in df.groupby("k_star"):
        curve = dict(zip(g.k, g.bic_read))
        rows.append(slack_rows("paper Fig.3 (digitised vector paths)", "original paper (code not in repo)",
                               "paper BIC Eq.(26)", ("gaussian", "bernoulli"), 150, ks, "1 curve per k*",
                               curve, "paper p.345 Fig.3 -> paper_fig3_vector_vertices.csv",
                               "aggregation over 10 trials not stated; k*=1,k=9 vertex absent"))
        for k, b in curve.items():
            pen = t_paper(k) * LN150
            implied.append(dict(k_star=ks, k=k, bic_read=b, eq26_penalty=pen, implied_minus2_lnL=b - pen))
    return rows, pd.DataFrame(implied)


def series_python_repro():
    path = ROOT / "reproduction/results/results_exp2_bic.csv"
    df = pd.read_csv(path)
    rows, dec = [], []
    n = 150
    for kt, g in df.groupby("k_true"):
        curve = dict(zip(g.k_est, g.BIC))
        rows.append(slack_rows("Python reproduction exp2", "reproduction (0.5 in E-step, alpha=0.5)",
                               "-2*max_iter lnp(Z,X,Y|plug-in last Z) + Eq.26 penalty", ("gaussian", "bernoulli"),
                               n, kt, "1 dataset, best of 3 inits", curve, "reproduction/results/results_exp2_bic.csv",
                               "num_iter=30, L=10, n_trials=3, best iteration"))
        for _, r in g.iterrows():
            k = int(r.k_est)
            lpz = -(n * k / 2) * math.log(2 * math.pi) - n * k / 2   # E||z||^2 = k after scale_Z (approx.)
            dec.append(dict(k_true=kt, k_est=k, BIC=r.BIC, log_likelihood=r.log_likelihood,
                            penalty=t_paper(k) * LN150, approx_ln_p_Z=lpz,
                            approx_ln_p_XY=r.log_likelihood - lpz,
                            recompute_diff=r.BIC - (-2 * r.log_likelihood + t_paper(k) * LN150)))
    return rows, pd.DataFrame(dec)


def series_early_expfam():
    rows = []
    fams = {"A": ("poisson", "bernoulli"), "B": ("gaussian", "poisson"), "C": ("bernoulli", "gaussian")}
    for s, fam in fams.items():
        path = RES / f"exp2_bic_{s}.csv"
        df = pd.read_csv(path)
        for (kt, tr), g in df.groupby(["k_true", "trial"]):
            curve = dict(zip(g.k_est, g.BIC))
            rows.append(slack_rows(f"early ExpFam exp2 Scenario {s}", "0.5 (model_dual_expfam.py)",
                                   "calc_bic_dual(Q_strict)", fam, 150, kt, f"dataset trial {tr}", curve,
                                   f"expfam/results/exp2_bic_{s}.csv", "1 start per (dataset,k); L=5, 8 iterations"))
    return rows


def series_paper_type_E():
    base = RES / "k_selection/paper_condition_k_criteria_20261006/full"
    df = pd.read_csv(base / "criterion_curves.csv")
    rows = []
    for (crit, kt, rep), g in df[df.criterion.isin(["C_Q", "C_Lap"])].groupby(["criterion", "k_true", "replicate"]):
        curve = {int(k): v for k, v in zip(g.k_est, g.mean_value)}
        if any(pd.isna(v) for v in curve.values()):
            rows.append(dict(series="paper-type #139 (lineage E)", lineage="E (consistent)", criterion=crit,
                             family_x="gaussian", family_y="bernoulli", n=150, k_true=kt, unit=f"dataset rep {rep}",
                             k_hat=np.nan, correct=np.nan, source=str(base.relative_to(ROOT) / "criterion_curves.csv"),
                             note="INCOMPLETE (non-OK C_Lap), not imputed"))
            continue
        rows.append(slack_rows("paper-type #139 (lineage E)", "E (consistent)", crit, ("gaussian", "bernoulli"),
                               150, kt, f"dataset rep {rep}", curve,
                               str(base.relative_to(ROOT) / "criterion_curves.csv"),
                               "mean of 2 starts; rep 1 data = Python reproduction data (seed 1980+100K)"))
    return rows


def series_canonical():
    base = RES / "k_selection/historical_canonical_causal_20261106/existing_reanalysis"
    df = pd.read_csv(base / "k_increment_decomposition.csv")
    rows = []
    for (crit, n, d, rep), g in df.groupby(["criterion", "n", "d", "replicate"]):
        # rebuild relative curve from slacks: C(1)=0, C(k+1)=C(k)-slack(k->k+1)
        g = g.sort_values("k_from")
        curve, c = {1: 0.0}, 0.0
        ok = True
        for _, r in g.iterrows():
            if r.status != "OK":
                ok = False
                break
            c = c - r.selection_slack
            curve[int(r.k_to)] = c
        if not ok:
            rows.append(dict(series=f"canonical-clean-v1 K_GEN=5 d={d}", lineage="E (consistent)", criterion=crit,
                             family_x="poisson", family_y="bernoulli", n=n, k_true=5, unit=f"dataset rep {rep}",
                             k_hat=np.nan, correct=np.nan, source=str(base.relative_to(ROOT)),
                             note="LAPLACE_NOT_OK step, not imputed"))
            continue
        rows.append(slack_rows(f"canonical-clean-v1 K_GEN=5 d={d}", "E (consistent)", crit, ("poisson", "bernoulli"),
                               int(n), 5, f"dataset rep {rep}", curve,
                               str(base.relative_to(ROOT) / "k_increment_decomposition.csv"),
                               "curve rebuilt from consecutive slacks; mean of 2 starts; 80% train pairs"))
    return rows


# ---------------------------------------------------------------------------
# 2. signal strength
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def y_info(Z, w0, w, mask_frac=1.0):
    """Mean over nodes of eig(w^2 sum_j s(1-s) z_j z_j^T): (mean eig, min eig)."""
    n, k = Z.shape
    S = sigmoid(w0 + w * Z @ Z.T)
    V = S * (1 - S)
    np.fill_diagonal(V, 0.0)
    eigs = []
    for i in range(n):
        H = (w ** 2) * (Z.T * V[i]) @ Z * mask_frac
        eigs.append(np.linalg.eigvalsh(H))
    e = np.array(eigs)
    return float(e.mean()), float(e[:, 0].mean()), float(np.triu(S, 1).sum() / (n * (n - 1) / 2)), float(V.sum() / (n * (n - 1)))


def x_info_gauss(F, noise_var):
    e = np.linalg.eigvalsh(F.T @ F / noise_var)
    return float(e.mean()), float(e.min())


def x_info_pois(F, Z, clip=None):
    eta = Z @ F.T
    if clip is not None:
        eta = np.clip(eta, *clip)
    lam = np.exp(eta)
    eigs = [np.linalg.eigvalsh((F.T * lam[i]) @ F) for i in range(Z.shape[0])]
    e = np.array(eigs)
    return float(e.mean()), float(e[:, 0].mean()), float(lam.mean())


def signal_rows():
    rows = []
    d = 15

    def add(series, k, n, unit, F, xfam, xi, yi, w0, w, mask, src, note=""):
        ev = np.linalg.eigvalsh(F.T @ F)
        rows.append(dict(series=series, k_true=k, n=n, unit=unit, x_family=xfam,
                         FtF_eig_mean=float(ev.mean()), FtF_eig_min=float(ev.min()), FtF_trace=float(ev.sum()),
                         x_fisher_per_dir_mean=xi[0], x_fisher_per_dir_min=xi[1],
                         poisson_mean_rate=(xi[2] if len(xi) > 2 else np.nan),
                         w0=w0, w=w, w2=w * w, w2_times_k=w * w * k,
                         y_edge_density=yi[2], y_mean_s1ms=yi[3],
                         y_fisher_per_dir_mean=yi[0], y_fisher_per_dir_min=yi[1], y_pair_fraction=mask,
                         total_fisher_per_dir_mean=xi[0] + yi[0], total_fisher_per_dir_min_lb=xi[1] + yi[1],
                         source=src, note=note))

    # (a) MATLAB/Python reproduction generator (seed 1980 + 100 K; = #139 rep 1)
    for k in (1, 3, 5, 7, 9):
        for rep in (1, 2, 3, 4, 5):
            seed = 1980 + 100 * k + 1000 * (rep - 1)
            p = set_true_params(n=150, d=d, k=k, seed=seed)
            add("reproduction/MATLAB-port generator (Gaussian-X)", k, 150, f"seed {seed}", p["F"], "gaussian",
                x_info_gauss(p["F"], 0.1), y_info(p["Z"], p["w0"], p["w"]), p["w0"], p["w"], 1.0,
                "reproduction/src/data_generator.py::set_true_params",
                "rep1 = Python exp2 data; reps 1-5 = #139 data. X z-scored afterwards (Fisher is invariant to column scaling)")
    # (b) paper text generator (F ~ N(0,10), Sigma=0.1, no normalisation); w unspecified in paper
    for k in (1, 3, 5, 7, 9):
        rng = np.random.default_rng(424242 + k)
        for r in range(5):
            F = rng.normal(0, math.sqrt(10.0), size=(d, k))
            ev = x_info_gauss(F, 0.1)
            Z = rng.standard_normal((150, k))
            add("paper text generator (F_ij ~ N(0,10), no row normalisation)", k, 150, f"sim {r}", F, "gaussian",
                ev, (np.nan, np.nan, np.nan, np.nan), np.nan, np.nan, 1.0, "paper Sect.4.1 Step 1-3",
                "w0, w 'random values' (distribution unstated): Y side UNRESOLVED")
    # (c) early ExpFam Scenario A (Poisson-X / Bernoulli-Y), seeds of run_exp2_bic_v2.py
    for k in (1, 3, 5, 7, 9):
        for tr in range(5):
            seed = 8000 + k * 1000 + tr * 100
            p = generate_dual_data(n=150, d=d, k=k, seed=seed, family_x="poisson", family_y="bernoulli")
            add("early ExpFam Scenario A generator (Poisson-X)", k, 150, f"seed {seed}", p["F"], "poisson",
                x_info_pois(p["F"], p["Z"], clip=(-20, 10)), y_info(p["Z"], p["w0"], p["w"]), p["w0"], p["w"], 1.0,
                "expfam/src/data_generator_expfam.py::generate_dual_data; seeds of run_exp2_bic_v2.py")
    # (d) canonical-clean-v1 (fresh seeds, same definition as PR #155 condition A / A1)
    for k, n in ((5, 75), (5, 100), (5, 150), (3, 75), (1, 75)):
        for r in range(8):
            seed = 990000 + 100 * k + n + r
            fs = f_scale_for_row_norm(0.5, d=d, k=k)
            ds = generate_canonical_data(n=n, d=d, k=k, seed=seed, family_x="poisson", family_y="bernoulli",
                                         f_scale=fs, w0=-1.0, w=math.sqrt(3.0 / k))
            add("canonical-clean-v1 (A / A1 definition, fresh seeds)", k, n, f"sim seed {seed}", ds.F, "poisson",
                x_info_pois(ds.F, ds.Z), y_info(ds.Z, -1.0, math.sqrt(3.0 / k), mask_frac=0.8), -1.0,
                math.sqrt(3.0 / k), 0.8, "expfam/src/experimental/data_generator_canonical.py",
                "not the experiment's datasets; Y information x0.8 for the train-pair fraction")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. finite-difference check of E-step gradient / precision
# ---------------------------------------------------------------------------

def fd_check():
    rng = np.random.default_rng(20261106)
    n, k, d = 12, 3, 5
    Z = rng.standard_normal((n, k))
    F = rng.standard_normal((d, k))
    sig = np.full(d, 0.3)
    X = Z @ F.T + rng.standard_normal((n, d)) * np.sqrt(sig)
    w0, w = -0.4, 1.7
    P = sigmoid(w0 + w * Z @ Z.T)
    Y = np.triu((rng.random((n, n)) < P).astype(float), 1)
    Y = Y + Y.T
    i = 4

    def logf(zi):
        Zc = Z.copy()
        Zc[i] = zi
        S = sigmoid(w0 + w * Zc @ Zc.T)
        ly = Y * np.log(S) + (1 - Y) * np.log(1 - S)
        np.fill_diagonal(ly, 0.0)
        # Eq.(20): 1/2 * sum_i sum_{j != i}  == sum over unordered pairs
        return (-0.5 * np.sum(Zc ** 2) - 0.5 * np.sum((X - Zc @ F.T) ** 2 / sig) + 0.5 * ly.sum())

    z = Z[i].copy()
    h = 1e-5
    g_fd = np.array([(logf(z + h * e) - logf(z - h * e)) / (2 * h) for e in np.eye(k)])
    H_fd = np.array([[(logf(z + h * a + h * b) - logf(z + h * a - h * b) - logf(z - h * a + h * b)
                       + logf(z - h * a - h * b)) / (4 * h * h) for b in np.eye(k)] for a in np.eye(k)])
    s = sigmoid(w0 + w * Z @ z)
    r = Y[i] - s
    r[i] = 0.0
    v = s * (1 - s)
    v_noi = v.copy()
    v_noi[i] = 0.0
    base_g = -z + F.T @ ((X[i] - F @ z) / sig)
    base_A = np.eye(k) + F.T @ (F / sig[:, None])
    cands = {
        "adopted (no extra 1/2), = model_dual_expfam_fixed / lineage E": (
            base_g + w * Z.T @ r, base_A + w ** 2 * (Z.T * v_noi) @ Z),
        "paper Eq.(23)/(22) and Python reproduction / 0.5 lineage (1/2)": (
            base_g + 0.5 * w * Z.T @ r, base_A + 0.5 * w ** 2 * (Z.T * v_noi) @ Z),
    }
    # MATLAB calcEtaNewton.m: calcGrad returns -(gradient) WITHOUT w; calcAi diagonal correction
    S_row = sigmoid(w0 + w * Z[i] @ Z.T)
    YSWZ = (Y[i] - S_row) @ Z - (Y[i, i] - S_row[i]) * Z[i]
    partE = z - F.T @ ((X[i] - F @ z) / sig) - YSWZ          # calcEtaNewton.m L44-48 (varZ=1)
    sigd = w ** 2 * S_row * (1 - S_row)                       # L58-59
    tmp = (Z.T * sigd) @ Z
    tmp = tmp - sigd[i] * (1 - sigd[i]) * w ** 2 * np.outer(Z[i], Z[i])   # L61
    cands["MATLAB calcEtaNewton.m (calcGrad / calcAi as written)"] = (-partE, base_A + tmp)
    rows = []
    for name, (g, A) in cands.items():
        rows.append(dict(implementation=name,
                         grad_rel_err=float(np.linalg.norm(g - g_fd) / np.linalg.norm(g_fd)),
                         precision_rel_err=float(np.linalg.norm(A - (-H_fd)) / np.linalg.norm(H_fd)),
                         y_grad_component_ratio=float(np.dot(g - base_g, w * Z.T @ r) / np.dot(w * Z.T @ r, w * Z.T @ r)),
                         note="target: d/dz_i and -d2/dz_i2 of Eq.(20) (prior+X+1/2*ordered-pair Y = unordered pairs), "
                              f"finite differences h={h}; n={n},k={k},d={d},w={w}"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. git provenance of the inventory files
# ---------------------------------------------------------------------------

def git_info(path: str):
    out = subprocess.run(["git", "log", "--follow", "--format=%h|%ad|%an|%s", "--date=short", "--", path],
                         cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.strip().splitlines()
    if not out:
        return dict(first_commit="", first_date="", first_author="", last_commit="", last_date="", n_commits=0)
    f, l = out[-1].split("|", 3), out[0].split("|", 3)
    return dict(first_commit=f[0], first_date=f[1], first_author=f[2], last_commit=l[0], last_date=l[1],
                n_commits=len(out))


INVENTORY = [
    # path, era/lineage, role, used for which study, provenance note
    ("paper/A_study_on_latent_structural_models_for_binary_rel.pdf", "original paper", "primary source (Exp.1-3, Eq.16-26, Fig.3)", "Mikawa et al. 2024", "published paper"),
    ("Mato Lab Program/setTrueParams.m", "MATLAB", "data generator (row-normalised F, z-scored Z and X, rng(1980) inside)", "lab MATLAB experiments (Exp.1/3 live scripts)", "origin outside git; imported in bulk"),
    ("Mato Lab Program/setParamsDesc.m", "MATLAB", "initial values (Z, F random; numIter=10)", "same", "same"),
    ("Mato Lab Program/calcdescmetric_ver4.m", "MATLAB", "MCEM loop, M-step, Q (with ln p(Z)) per iteration", "same", "same"),
    ("Mato Lab Program/calcEtaNewton.m", "MATLAB", "E-step (alpha=0.01 Newton, Laplace sample)", "same", "same"),
    ("Mato Lab Program/EffcalcEtaNewton (1).m", "MATLAB", "unfinished efficiency variant (calcInitA has no output)", "not called", "same; file name '(1)' suggests a downloaded copy"),
    ("Mato Lab Program/calcw.m", "MATLAB", "Adam for w", "same", "same"),
    ("Mato Lab Program/calcw0.m", "MATLAB", "Adam for w0", "same", "same"),
    ("Mato Lab Program/calcRmseDesc.m", "MATLAB", "RMSE after SVD normalisation", "same", "same"),
    ("Mato Lab Program/DecideNumFactor.m", "MATLAB", "X-only factor-analysis BIC (factoran)", "not called by any repository script", "same"),
    ("Mato Lab Program/NOLTA_exp_ver3_revise_batch.m", "MATLAB", "batch: 'k fixed, vary L'; loops k=1:10 but fits kT=3", "NOT Exp.2", "same"),
    ("Mato Lab Program/NOLTA_exp_ver5 (1).pdf", "MATLAB", "live-script export: Exp.3 (vary d), printed Q per iteration", "Exp.3-type run", "PDF created 2025-08-04 (metadata)"),
    ("Mato Lab Program/main_save_func_parfor (1).m", "MATLAB", "save helper", "same", "same"),
    ("reproduction/src/data_generator.py", "Python reproduction", "port of setTrueParams.m", "Python reproduction; #139 paper-type data", "states 'implements setTrueParams.m'"),
    ("reproduction/src/model.py", "Python reproduction", "LatentStructuralModel (1/2 in Y gradient and precision)", "Python reproduction", ""),
    ("reproduction/src/experiment_paper_2.py", "Python reproduction", "Exp.2 runner: plug-in lnp(Z,X,Y), best iteration, best of 3", "results_exp2_bic.csv", ""),
    ("reproduction/results/results_exp2_bic.csv", "Python reproduction", "Exp.2 result (1 dataset per k*)", "5/5 claim", "no runinfo / seed log beyond the script"),
    ("reproduction/reports/FINAL_REPRODUCTION_REPORT.md", "Python reproduction", "narrative report ('100% / 120% reproduction')", "secondary", "AI-assisted style; claims not primary evidence"),
    ("reproduction/reports/baseline_audit.md", "Python reproduction", "baseline audit narrative", "secondary", ""),
    ("expfam/src/data_generator_expfam.py", "ExpFam 0.5 lineage", "generate_dual_data (historical generator for 3 families)", "early ExpFam experiments", ""),
    ("expfam/src/model_dual_expfam.py", "ExpFam 0.5 lineage", "DualExpFamLSM (1/2 in Y term)", "conference-draft experiments", ""),
    ("expfam/src/utils_expfam.py", "ExpFam 0.5 lineage", "run_em_dual (retries), calc_Q_strict, calc_bic_dual", "early ExpFam / fixed", ""),
    ("expfam/src/run_exp2_bic_v2.py", "ExpFam 0.5 lineage", "Exp.2-type K*=1..9 runner, 3 scenarios", "exp2_bic_{A,B,C}.csv", ""),
    ("expfam/src/model_dual_expfam_fixed.py", "ExpFam fixed lineage", "removes extra 1/2", "real-data phase", ""),
    ("expfam/src/experimental/model_dual_expfam_consistent.py", "lineage E", "objective-consistent numerics", "current K-selection studies", ""),
    ("expfam/src/experimental/data_generator_canonical.py", "lineage E", "canonical-clean-v1 generator", "PR #155-#157", ""),
    ("expfam/src/experimental/k_criterion_panel.py", "lineage E", "same-fit C_Q / C_Lap panel", "#139 and later", ""),
    ("expfam/src/experimental/laplace_k_criterion.py", "lineage E", "C_Lap evaluator", "#139 and later", ""),
]


def inventory():
    rows = []
    for path, era, role, used, prov in INVENTORY:
        p = ROOT / path
        gi = git_info(path)
        rows.append(dict(path=path, era_lineage=era, role=role, used_for=used, exists=p.exists(),
                         sha256=sha(p) if p.exists() else "", **gi,
                         provenance_note=prov + ("; repository history starts with bulk import 7c3040f (2026-04-27); "
                                                 "git author does not establish who wrote the code"
                                                 if gi["first_commit"] == "7c3040f" else "")))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. curated comparison tables (FACT rows cite file:line)
# ---------------------------------------------------------------------------

PAPER_VS_MATLAB = [
    ("Z generation", "z_i ~ N(0, sigma_z^2 I), sigma_z^2=1 (Sect.4.1 Step 1)", "normalize(normrnd(0,1,[n,k])) column z-score (setTrueParams.m:22)", "differs (sample normalisation)", "FACT"),
    ("F generation", "each entry of F* ~ N(0,10) (Sect.4.1 Step 3)", "normrnd(0, sqrt(5)); each row rescaled to norm sqrt(1-0.1) (setTrueParams.m:16,23,26-28)", "differs: var 5 not 10; row norm fixed -> F^T F trace = 13.5 for every k", "FACT"),
    ("Sigma", "diag(0.1,...,0.1)", "diag(0.1) (setTrueParams.m:15,20)", "same", "FACT"),
    ("mu_x / X post-processing", "mu_x = 0; X ~ N(F z, Sigma)", "X = normalize(mvnrnd(Z F', Sigma)) column z-score (setTrueParams.m:38-41)", "differs (X z-scored)", "FACT"),
    ("w0, w", "'random values' (distribution not stated)", "w0 = randn(); w = 3*randn() after rng(1980) (setTrueParams.m:12,30-31)", "paper silent", "FACT"),
    ("seed", "not stated", "rng(1980) inside setTrueParams: same stream every call (setTrueParams.m:12)", "paper silent", "FACT"),
    ("Y", "y_ij ~ Bern(s_ij); y_ii = 1 by convention (Sect.2.2)", "triu(binornd(...),1) + transpose; y_ii = 0 (setTrueParams.m:49-51)", "diagonal convention differs (diagonal excluded in fit anyway)", "FACT"),
    ("L", "10 (Table I)", "batch: L = 11 (NOLTA_exp_ver3_revise_batch.m:19); live script Exp.3: L = 10", "batch differs", "FACT"),
    ("iterations", "10", "numIter = 10 (setParamsDesc.m:4)", "same", "FACT"),
    ("trials", "10 trials with different initial values", "batch loop 'for k=1:10' = 10 restarts of the same K (NOLTA_exp_ver3_revise_batch.m:30-33)", "the loop variable k is never used for the fitted dimension", "FACT"),
    ("fitted dimension", "k = 1..10 for each k* (Exp.2)", "setParamsDesc(trueParams, kT, i) and calcdescmetric_ver4(X, Y, kT, ...) -> always kT = 3 (batch L32-33)", "no repository MATLAB script varies the fitted K", "FACT"),
    ("BIC Eq.(26)", "-2 ln L + ((k+1)d - k(k-1)/2) ln n, L from Eq.(16)", "no Exp.2 BIC code; DecideNumFactor.m = X-only factoran BIC with -2*n*loglike (L11-14), never called", "Exp.2 code path absent", "FACT"),
    ("Q function", "Eq.(18): (1/L) sum_l ln p(Z^(l), X, Y | theta) (includes ln p(Z))", "Q = (1/L) sum_l [p_X + p_Y + p_Z] (calcdescmetric_ver4.m:37-47)", "same quantity (Q2 MC mean)", "FACT"),
    ("E-step gradient", "Eq.(23): ... + (1/2) sum_j (y_ij - s_ij) w z_j", "calcGrad: (Y - S) Z without w and without 1/2 (calcEtaNewton.m:43-49)", "MATLAB omits w; paper has an extra 1/2 (KI-001)", "FACT"),
    ("precision A_i", "Eq.(22): (1/2) sum_j s(1-s) w^2 z_j z_j^T", "calcAi: w^2 sum_j s(1-s) z_j z_j^T, self term removed with an extra factor (calcEtaNewton.m:56-63)", "no 1/2 in MATLAB; diagonal-removal defect", "FACT"),
    ("Newton step", "Newton method (step size not stated)", "alpha = 0.01, at most 9 steps per node; 'diff' not reset between nodes (calcEtaNewton.m:3-22)", "MATLAB effectively takes ~1e-2 Newton steps", "FACT"),
    ("w0/w update", "Adam (Eq.24-25)", "Adam-like with m <- b1 m + (1 - b1^t) g (non-standard), lr 0.01, <=49 steps (calcw.m, calcw0.m)", "close but non-standard moment update", "FACT"),
    ("sigma_z update", "Eq.(14)/(B-13)", "varZ fixed at 1 and Z rescaled (scaleZ) so mean ||z||^2 = k (calcdescmetric_ver4.m:29,65-79)", "differs (fixed + rescale)", "FACT"),
]

PAPER_VS_PYTHON = [
    ("generator", "Sect.4.1: F ~ N(0,10), no normalisation", "port of setTrueParams.m: var_f=5, row norm sqrt(0.9), Z and X z-scored (data_generator.py:100-153)", "follows MATLAB, not the paper text", "FACT"),
    ("seed per k*", "not stated", "1980 + 100 k* (experiment_paper_2.py:269-272); 1 dataset per k*", "", "FACT"),
    ("L / iterations / trials", "10 / 10 / 10", "10 / 30 / 3 (experiment_paper_2.py:437-439)", "iterations x3, trials /3", "FACT"),
    ("trial aggregation", "not stated for Exp.2 (Sect.4.1 defines min over trials and iterations for Exp.1,3,'4')", "best BIC over 3 inits; within a run the best iteration (experiment_paper_2.py:234-245, 289-297)", "Python uses max-over-iterations and min-over-trials", "FACT"),
    ("ln L in BIC", "Eq.(16): ln p(X, Y | Z, theta) (no p(Z))", "ln p(Z) + ln p(X|Z) + ln p(Y|Z) at one plug-in Z (experiment_paper_2.py:124-170)", "Python adds ln p(Z) (= about -213 k at n=150) and uses one Z", "FACT"),
    ("penalty", "((k+1)d - k(k-1)/2) ln n", "same (experiment_paper_2.py:119-121)", "same", "FACT"),
    ("E-step 1/2", "Eq.(22)/(23) with 1/2", "0.5 in gradient and precision (model.py:276, 337)", "same as paper print (not the adopted formula)", "FACT"),
    ("Newton step", "not stated", "alpha = 0.5 in Exp.2 (experiment_paper_2.py:215); model default 0.01", "chosen after the alpha=0.01 run selected k=1 (FINAL_REPRODUCTION_REPORT.md:166-176; no primary CSV of that run)", "FACT (code) / UNRESOLVED (report claim)"),
    ("w0, w initialisation", "not stated", "w0 = logit(density), w = 0.5 (experiment_paper_2.py:186-190)", "Python-specific", "FACT"),
    ("true values in fit", "-", "initialize_params uses true var_f only for scale; Z, F random (model.py:63-130)", "no true-value leakage", "FACT"),
    ("Fig.3 scale", "BIC 340-5700; -2lnL implied 2-190 at k >= k*", "BIC 7,174-20,709", "absolute values not reproduced (factor ~10-30)", "OBSERVED"),
]

HIST_VS_EXPFAM = [
    # item, paper, MATLAB, Python repro, early ExpFam A (0.5), fixed, canonical E (PR #155 A)
    ("X / Y family", "Gauss / Bern", "Gauss / Bern", "Gauss / Bern", "Pois / Bern (A); Gauss/Pois (B); Bern/Gauss (C)", "same as early", "Pois / Bern"),
    ("K_GEN and candidate K", "1,3,5,7,9; 1..10", "kT=3 fixed; no K sweep", "1,3,5,7,9; 1..10", "1,3,5,7,9; 1..10 (exp2_bic_v2)", "3; 1..9 (Exp1)", "5; 1..7 (A1: 1,3,5)"),
    ("n, d", "150, 15", "150, 15", "150, 15", "150, 15", "150, 15", "75 (50..150), 15 (30, 60)"),
    ("L, EM iterations", "10, 10", "11 (batch) / 10, 10", "10, 30", "5, 8", "5, 8", "5, 8 (PR #156: 64)"),
    ("datasets x starts", "10 trials (aggregation unstated)", "10 restarts of K=3", "1 dataset, best of 3", "5 datasets, 1 start", "10 datasets, 1 start", "8 datasets, 2 starts (mean)"),
    ("Z generation", "N(0,I)", "N(0,I) then column z-score", "same as MATLAB", "same as MATLAB", "same", "N(0,I), no normalisation"),
    ("F generation", "N(0,10) entries", "N(0,5), rows to norm sqrt(0.9)", "same as MATLAB", "same as MATLAB", "same", "QR, all singular values equal, mean row norm^2 0.5"),
    ("F^T F eigenvalues", "about 10 d = 150 per direction (expectation)", "trace 13.5, random split (mean 13.5/K)", "same as MATLAB", "same as MATLAB", "same", "all equal 0.5 d / K (K=5: 1.5)"),
    ("w0, w", "random, unstated", "randn, 3 randn", "same as MATLAB (seeded)", "-1, 1.5 (Bern-Y)", "same", "-1, sqrt(3/K) (K=5: 0.775)"),
    ("Gaussian X noise variance", "0.1", "0.1 then z-score", "0.1 then z-score", "0.1 then z-score (Gauss-X only)", "same", "- (Poisson-X)"),
    ("Poisson X rate", "-", "-", "-", "exp(f'z), ||f||^2 = 0.9, eta clipped to [-20,10]", "same", "exp(f'z), mean ||f||^2 0.5, no clipping"),
    ("observation mask", "none", "none", "none", "none", "none", "20% of pairs held out (train 80%)"),
    ("E-step formula", "Eq.(22)/(23) with 1/2", "no 1/2, gradient without w, alpha 0.01", "with 1/2, alpha 0.5", "with 1/2 (Y term), alpha 0.5 with halving retries", "no 1/2", "no 1/2, objective-consistent numerics"),
    ("score", "Eq.(26) with ln L of Eq.(16) (evaluation Z unstated)", "Q (Q2 MC mean) computed; no BIC", "-2 max_iter ln p(Z,X,Y | last Z) + penalty", "calc_bic_dual(Q_strict): Q2 MC mean + penalty", "same", "C_Q (Q2 MC mean) and C_Lap (Laplace)"),
    ("parameter penalty", "(k+1)d - k(k-1)/2", "-", "(k+1)d - k(k-1)/2", "kd - k(k-1)/2 (+d for Gauss-X)", "same", "kd - k(k-1)/2 (+d Gauss-X)"),
    ("selection", "argmin of plotted curve", "-", "argmin over k of best-of-3", "argmin per dataset", "argmin per dataset", "argmin of 2-start mean"),
]

CRITERION_MATRIX = [
    ("paper Eq.(26)", "ln p(X, Y | Z, theta) (Eq.16; Y with power 1/2 over ordered pairs)", "no", "not stated", "not stated (Fig.3 values imply -2lnL in [2, 192] at k >= k*)", "(k+1)d - k(k-1)/2", "paper p.338,343", "called BIC in the paper; not the observed-data marginal likelihood"),
    ("MATLAB calcdescmetric_ver4 Q", "(1/L) sum_l ln p(Z^(l), X, Y | theta)", "yes", "MC mean over L draws of the last iteration", "printed Q/L about -6,500 at d=15 (NOLTA_exp_ver5 PDF)", "none (no BIC)", "calcdescmetric_ver4.m:37-47", "Q2; not used for K selection in repository code"),
    ("MATLAB DecideNumFactor", "factoran loglike on X only", "no (X marginal of FA)", "n/a", "-", "(i+1)d - i(i-1)/2 with -2*n*loglike", "DecideNumFactor.m:11-14", "X-only factor analysis; never called"),
    ("Python reproduction", "ln p(Z) + ln p(X|Z) + ln p(Y|Z) at one Z (last draw of an iteration)", "yes", "plug-in, max over 30 iterations, min over 3 inits", "-2lnL 6,000-20,000", "(k+1)d - k(k-1)/2", "experiment_paper_2.py:124-170, 234-245", "Q2 plug-in with selection over iterations"),
    ("early ExpFam calc_bic_dual", "Q_strict = Q2 MC mean (with Poisson -ln y! correction)", "yes", "MC mean over L of final iteration", "-", "kd - k(k-1)/2 (+d Gauss-X, +1 Gauss-Y)", "utils_expfam.py:80-105, 386-404", "same family as C_Q"),
    ("lineage E C_Q", "Q2 MC mean at the final state", "yes", "MC mean, mean over 2 starts", "-", "kd - k(k-1)/2 (+d Gauss-X)", "k_criterion_panel.py", "Q-based complete-data / ICL-type; not Schwarz BIC"),
    ("lineage E C_Lap", "Laplace approximation of ln p(X,Y|theta) at the fit's Z mode", "integrated (approx.)", "deterministic given theta", "-", "same count", "laplace_k_criterion.py", "Laplace approximation; not the exact marginal likelihood"),
]

DISCREPANCIES = [
    ("D1", "MATLAB", "NOLTA_exp_ver3_revise_batch.m:30-33", "loop 'for k=1:10' passes kT (=3) to setParamsDesc and calcdescmetric_ver4; every run fits K=3", "the only batch script cannot have produced Exp.2", "FACT", "none on paper results (path not the Exp.2 path)"),
    ("D2", "MATLAB", "calcdescmetric_ver4.m:10", "argument i assigned to m but m never used; dimension comes from params.Z", "harmless given D1", "FACT", "-"),
    ("D3", "MATLAB", "calcEtaNewton.m:43-49", "Y part of the gradient lacks the factor w", "Newton target is not the posterior mode unless w=1", "FACT (finite-difference check)", "UNRESOLVED (no MATLAB Exp.2 run available)"),
    ("D4", "MATLAB", "calcEtaNewton.m:61", "self-term removal uses sig(ind,ind)*(1-sig(ind,ind))*w^2 where sig already holds w^2 s(1-s)", "precision slightly wrong", "FACT", "UNRESOLVED"),
    ("D5", "MATLAB", "calcEtaNewton.m:8,15", "Newton step size alpha = 0.01, <= 9 steps", "E-step moves about 9% toward the mode; sampler is then N(near-current z, A^-1)", "FACT", "Python report says alpha=0.01 made BIC select k=1 (no primary CSV)"),
    ("D6", "MATLAB", "calcEtaNewton.m:5-22", "'diff' is not reset per node; if one node converges, later nodes skip Newton and reuse the previous node's invAi", "latent defect", "FACT", "UNRESOLVED"),
    ("D7", "MATLAB", "setTrueParams.m:12", "rng(1980) inside the generator: all calls share one stream", "datasets for different calls are not independent draws", "FACT", "-"),
    ("D8", "MATLAB", "DecideNumFactor.m:14", "BIC uses -2*n*loglike (n times factoran's loglike)", "scale depends on factoran's loglike definition", "FACT / UNRESOLVED", "never called"),
    ("D9", "paper", "Eq.(23), (A-5)", "1/2 in the gradient although Eq.(20) already counts each pair once (KI-001)", "-", "FACT + DERIVED", "-"),
    ("D10", "paper", "Eq.(A-6)", "contains 1/(2 sigma_y^2), a symbol not in the binary model", "typo-level inconsistency", "FACT", "-"),
    ("D11", "paper", "Eq.(16) vs Eq.(26) vs Eq.(18)", "BIC's ln L (no p(Z), Z fixed) differs from Q (with p(Z), integrated over posterior)", "the paper's 'BIC' is neither Q nor the marginal likelihood", "FACT + DERIVED", "-"),
    ("D12", "paper", "Sect.4.1", "generator text (F ~ N(0,10)) differs from MATLAB/Python generator (row norm sqrt(0.9))", "signal per direction differs by about 50-100x", "FACT + DERIVED", "which generator made Fig.3 is UNRESOLVED"),
    ("D13", "Python", "experiment_paper_2.py:124-170", "ln L includes ln p(Z) and uses one Z; Eq.(16) has no p(Z)", "-2 ln p(Z) adds about +426 per extra dimension at n=150", "FACT + DERIVED", "acts as an extra penalty; selection still correct because under-fit loss >> 426"),
    ("D14", "Python", "experiment_paper_2.py:234-245, 289-297", "best iteration and best of 3 inits", "optimistic selection of the fit term", "FACT", "-"),
    ("D15", "Python", "experiment_paper_2.py:215 and report L166-176", "alpha changed from 0.01 to 0.5 after the selection result was seen", "post-hoc tuning on the outcome", "FACT (code) / report claim UNRESOLVED", "-"),
    ("D16", "Python", "model.py:276,337", "1/2 in E-step (paper print), not the adopted formula", "posterior targeted with half the Y information", "FACT", "KI-001"),
    ("D17", "early ExpFam", "data_generator_expfam.py:304", "Poisson eta clipped to [-20, 10]", "minor for ||f||^2=0.9", "FACT", "-"),
]

HYPOTHESES = [
    ("H1", "attribute signal per direction is larger in historical generators",
     "Gaussian-X Fisher per direction = eig(F'F)/sigma^2; Poisson-X: eig(sum_l lambda_il f_l f_l')",
     "historical rows fixed at ||f_l||^2 = 0.9 -> trace(F'F) = 13.5 for every K; canonical mean ||f_l||^2 = 0.5, equal singular values",
     "K=5 median X Fisher per direction: canonical 2.07 (n=75); early ExpFam Poisson 4.14; MATLAB/Python Gaussian 27.0; paper-text generator ~1500",
     "SUPPORTED as a difference (2x Poisson-to-Poisson, 13x Gaussian)", "co-varies with Y signal, n, mask, Z normalisation and estimator", "E3/E5"),
    ("H2", "relational signal is stronger historically",
     "Y Fisher per direction ~ w^2 sum_j s(1-s) z_j z_j'",
     "MATLAB/Python w = 3 randn (K*=5 data: w = 5.03); early ExpFam w = 1.5; canonical w = sqrt(3/5) = 0.775 with 80% of pairs",
     "K=5 median Y Fisher per direction: canonical 5.1 (n=75); early ExpFam 32.1 (n=150); MATLAB/Python 105",
     "SUPPORTED (largest single difference, 6-20x)", "includes the n difference (Y Fisher ~ n)", "E5 (w alone); PR #155 C varied w x sqrt(1.8) at n=75"),
    ("H3", "Gaussian-X vs Poisson-X information structure",
     "Gaussian Fisher does not depend on z; Poisson Fisher scales with the rate exp(f'z)",
     "early ExpFam Scenario A (Poisson-X / Bernoulli-Y) selected K* in 25/25 datasets for K*=1..9",
     "historical_k_selection_summary.csv: Scenario A under-slack K*=5 median 1095, K*=9 236 (min 90)",
     "REJECTED as a sufficient cause (Poisson-X succeeded under the historical generator)", "Scenario A also had historical F, w, n", "E4"),
    ("H4", "Z / X normalisation changes the generating law",
     "column z-score of Z fixes each column's sample variance; X z-score changes units only (Fisher invariant)",
     "MATLAB, Python, early ExpFam z-score Z; canonical does not",
     "-", "INCONCLUSIVE (not isolated; X z-score is information-neutral, Z z-score is not tested)", "co-varies with everything else", "E3 variant: canonical with and without Z standardisation"),
    ("H5", "criteria evaluate different quantities",
     "Eq.(16) plug-in (no p(Z)) vs Q2 plug-in best iteration vs Q2 MC mean (C_Q) vs Laplace (C_Lap)",
     "paper Fig.3 implies -2lnL between 1.6 and 193 at k >= k* (scale incompatible with any repository Q2 score); Python and C_Q include ln p(Z): +425.7 per added dimension at n=150",
     "same 5 datasets (X sha256 equal): Python plug-in, E C_Q and E C_Lap all select K*; under-slack 1,600-3,500",
     "SUPPORTED that they differ; NOT the cause of historical success (all criteria agree on historical data); C_Lap also fails at canonical n=75 (3/8)", "paper's own lnL path UNRESOLVED", "E1"),
    ("H6", "E-step implementation differences (1/2, missing w, alpha)",
     "finite-difference check (estep_finite_difference_check.csv)",
     "0.5 and fixed lineages both succeed at K_GEN=3 historical (PR #155); 0.5 Python, E on the same data both succeed; Python alpha 0.01 -> k=1 is a report claim without CSV",
     "estep_finite_difference_check.csv", "NOT THE CAUSE of the historical success; effect near the canonical boundary UNTESTED", "only historical data tested", "E2"),
    ("H7", "trials, initial values, best-iteration selection",
     "max over iterations / min over inits biases the fit term upward",
     "Python: best of 3 and best iteration; paper: aggregation unstated; canonical: mean of 2 starts, final state",
     "on historical data, E (mean of 2 starts, final state) also succeeds 25/25",
     "NOT NEEDED for historical success; effect on canonical UNTESTED", "-", "E1 (rescoring) / E2"),
    ("H8", "observation mask", "Y information x0.8 for 80% train pairs", "canonical holds out 20% of pairs; historical none",
     "included in canonical Y Fisher (x0.8)", "MINOR (20% of Y information)", "-", "E5 (mask factor)"),
    ("H9", "F eigenvalues as K grows",
     "historical: trace fixed -> mean per-direction energy 13.5/K, random split (min eigenvalue small); canonical: equal 0.5d/K",
     "historical K=9: mean per-direction F energy 1.5 (= canonical K=5) and min 0.18, yet K=9 selected in Scenario A with slack 90-450",
     "signal_strength_summary.csv", "F energy alone does NOT explain; Y signal and n differ", "-", "E4"),
    ("H10", "selection slack hidden behind accuracy",
     "slack = C(K*-1) - C(K*) and C(K*+1) - C(K*)",
     "paper Fig.3: K*=9 under-slack 57, K*=3 over-slack 4.1; historical C_Q under-slack per node 1.6-36; canonical n=75 C_Q -1.7 per node",
     "historical_k_selection_summary.csv", "SUPPORTED: 'K=9 succeeded' had very different margins across lineages; paper Fig.3 margins are small on both sides", "Fig.3 is 1 curve per k*", "report slack, not accuracy"),
    ("H11", "sample size n (added)",
     "Y information grows ~ n; per-node prior cost of C_Q is fixed (2.84 per node per dimension)",
     "historical n=150 everywhere; canonical failure is at n=75",
     "canonical K_GEN=5 d=15 C_Q: n=75 0/8, n=100 1/8, n=150 7/8 (same generator, same estimator)",
     "SUPPORTED (largest controlled effect available)", "-", "already observed (PR #155 / #139 A1)"),
]

EXPERIMENTS = [
    ("E1", "same data, criterion only", "rescore existing canonical and #139 states with: Eq.(16)-type plug-in (no p(Z)); Q2 plug-in best-iteration (Python rule); C_Q; C_Lap", "uses saved states only; no new fit", "0 fits", "isolates H5/H7 exactly (same states)"),
    ("E2", "same data, estimator only", "canonical K_GEN=5, n=75 data (PR #155 A) fitted with 0.5 lineage, fixed lineage and lineage E; same seeds, L, iterations, starts; score with C_Q", "estimator lineage differs in more than the 1/2 (numerics); call it 'estimator family', not one factor", "8 datasets x K 1..7 x 2 starts x 3 lineages = 336 fits (~1-2 h at 1 process)", "H6"),
    ("E3", "same estimator, generator only", "lineage E on (i) historical Poisson-X generator (Scenario A defaults), (ii) canonical, at the same n=75 and n=150, K_GEN=5", "several generator features change together (F law, w, Z normalisation); not one factor", "2 generators x 2 n x 8 datasets x K 3..7 x 2 starts = 320 fits", "H1/H2/H4 jointly"),
    ("E4", "Gaussian-X / Bernoulli-Y, K=5 vs 9", "lineage E, canonical-style generator but Gaussian-X with sigma^2 chosen to match per-direction Fisher information of the historical generator at K=5 and K=9", "matching Fisher information is approximate (Poisson rate varies with z)", "2 K_GEN x 8 datasets x 5 candidate K x 2 starts = 160 fits", "H3/H9"),
    ("E5", "factorial on the canonical generator", "2x2x2: w (0.775 vs 1.5), F energy (0.5 vs 0.9 mean row norm^2), mask (0.8 vs 1.0), K_GEN=5, n=75", "pre-register effect sizes on slack, not accuracy", "8 cells x 8 datasets x K 4..6 x 2 starts = 384 fits", "ranks H1/H2/H8"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows_fig3, implied = series_paper_fig3()
    rows_py, dec_py = series_python_repro()
    rows = rows_fig3 + rows_py + series_early_expfam() + series_paper_type_E() + series_canonical()
    sel = pd.DataFrame(rows)
    write("historical_k_selection_recomputed.csv", sel)
    summ = (sel.dropna(subset=["k_hat"]).groupby(["series", "criterion", "n", "k_true"])
            .agg(units=("correct", "size"), correct=("correct", "sum"),
                 median_slack_under=("slack_under", "median"), min_slack_under=("slack_under", "min"),
                 median_slack_over=("slack_over", "median"), min_slack_over=("slack_over", "min"),
                 median_slack_under_per_node=("slack_under_per_node", "median")).reset_index())
    inc = sel[sel.k_hat.isna()].groupby(["series", "criterion", "n", "k_true"]).size().rename("incomplete")
    summ = summ.merge(inc.reset_index(), how="left", on=["series", "criterion", "n", "k_true"]).fillna({"incomplete": 0})
    write("historical_k_selection_summary.csv", summ)
    write("paper_fig3_implied_lnL.csv", implied)
    write("python_reproduction_bic_decomposition.csv", dec_py)
    sig = signal_rows()
    write("signal_strength_comparison.csv", sig)
    ssum = (sig.groupby(["series", "k_true", "n"])
            [["FtF_eig_mean", "FtF_eig_min", "x_fisher_per_dir_mean", "x_fisher_per_dir_min", "w2", "y_edge_density",
              "y_fisher_per_dir_mean", "total_fisher_per_dir_mean", "total_fisher_per_dir_min_lb"]]
            .median().reset_index())
    write("signal_strength_summary.csv", ssum)
    fd = fd_check()
    write("estep_finite_difference_check.csv", fd)
    write("code_lineage_inventory.csv", inventory())
    write("paper_vs_matlab.csv", pd.DataFrame(PAPER_VS_MATLAB, columns=["item", "paper", "matlab", "difference", "label"]))
    write("paper_vs_python.csv", pd.DataFrame(PAPER_VS_PYTHON, columns=["item", "paper", "python_reproduction", "difference", "label"]))
    write("historical_vs_expfam.csv", pd.DataFrame(HIST_VS_EXPFAM, columns=[
        "item", "paper", "matlab_repo", "python_reproduction", "early_expfam_0p5", "fixed", "canonical_E"]))
    write("criterion_definition_matrix.csv", pd.DataFrame(CRITERION_MATRIX, columns=[
        "score", "fit_term", "includes_ln_p_Z", "z_handling", "observed_scale", "penalty", "source", "note"]))
    write("implementation_discrepancies.csv", pd.DataFrame(DISCREPANCIES, columns=[
        "id", "where", "file_lines", "finding", "consequence", "label", "impact_on_results"]))
    write("causal_hypotheses.csv", pd.DataFrame(HYPOTHESES, columns=[
        "id", "hypothesis", "mathematical_basis", "implementation_fact", "csv_evidence", "verdict", "remaining_confound", "falsifying_experiment"]))
    write("proposed_controlled_experiments.csv", pd.DataFrame(EXPERIMENTS, columns=[
        "id", "contrast", "design", "caveat", "budget", "targets"]))
    inputs = ["reproduction/results/results_exp2_bic.csv", "expfam/results/exp2_bic_A.csv", "expfam/results/exp2_bic_B.csv",
              "expfam/results/exp2_bic_C.csv",
              "expfam/results/k_selection/paper_condition_k_criteria_20261006/full/criterion_curves.csv",
              "expfam/results/k_selection/historical_canonical_causal_20261106/existing_reanalysis/k_increment_decomposition.csv",
              "reports/k_selection_theory/historical_code_lineage_audit_20261106/paper_fig3_vector_vertices.csv",
              "paper/A_study_on_latent_structural_models_for_binary_rel.pdf"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    runinfo = dict(script="tools/research_audit/historical_code_lineage_audit.py", git_head=head,
                   python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                   inputs={p: sha(ROOT / p) for p in inputs}, mcem_fits=0,
                   outputs={p.name: sha(p) for p in sorted(OUT.glob("*.csv"))})
    (OUT / "runinfo.json").write_text(json.dumps(runinfo, indent=1), encoding="utf-8")
    print(summ.to_string())
    print(fd.to_string())


if __name__ == "__main__":
    main()
