"""
MovieLens Y の Poisson 仮定診断（Phase 1）。

診断の設計:
  (a) 周辺（marginal）統計: mean / var / var-mean比 / zero率 / max / 分位点。
      注意: 周辺の過分散は潜在構造による μ_ij の変動でも生じるため、
      それだけでは「条件付き Poisson の破れ」を証明しない。
  (b) 条件付き（conditional）診断: Poisson モデル（fixed 版系列、full mask）を
      フィットし、Pearson 残差過分散 (1/N)Σ(y−μ̂)²/μ̂ を計算。
      これが 1 を大きく超えれば、潜在構造で説明した後にも過分散が残る。
  (c) NB dispersion r のモーメント推定（Pearson 残差ベース）。
  (d) Posterior-predictive check (PPC): μ̂ から Y_rep を Poisson / NB2 で
      生成し、var/mean・max・q99・zero率 の PPC p 値を計算。
      注: μ̂ は plug-in（事後分布積分なし）の近似 PPC。

入力（read-only）:
  expfam/data/movielens_pilot/movielens_Y_count.npy      共評価カウント
  expfam/data/movielens_pilot/movielens_X_genre.npy
  expfam/data/ml-100k.zip                                co-like Y 再構築用

出力（新規ディレクトリのみ、既存結果に触れない）:
  expfam/results/overdispersion/movielens_overdispersion_diagnostics.csv
  expfam/results/overdispersion/movielens_ppc_summary.csv
  expfam/results/overdispersion/movielens_overdispersion_runinfo.csv
  figures/overdispersion/movielens_y_distribution.png/pdf
  figures/overdispersion/movielens_mean_variance.png/pdf

実行: python tools/overdispersion/diagnose_movielens_overdispersion.py
"""

import sys
import time
import zipfile
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "expfam" / "src"))
sys.path.insert(0, str(_ROOT / "expfam" / "src" / "experimental"))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from em_runner import run_em_experimental, predict_mu_y            # noqa
from eval_utils import (                                           # noqa
    pearson_dispersion, moment_estimate_nb_r, upper_pairs_of)

DATA_DIR = _ROOT / "expfam" / "data" / "movielens_pilot"
ZIP_PATH = _ROOT / "expfam" / "data" / "ml-100k.zip"
OUT_DIR = _ROOT / "expfam" / "results" / "overdispersion"
FIG_DIR = _ROOT / "figures" / "overdispersion"

K_LIST = [3, 5]              # 条件付き診断のフィット k
MODEL_SEED = 21000
L, NITER = 5, 8
N_PPC = 300
PPC_SEED = 31000
LIKE_THRESHOLD = 4           # co-like: 両ユーザが rating >= 4

# dataviz reference palette (validated, light mode)
C_OBS = "#52514e"       # observed data: neutral ink
C_POIS = "#2a78d6"      # Poisson model: slot 1 blue
C_NB = "#1baf7a"        # NB model: slot 2 aqua

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.dpi": 150,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def upper_vals(Y):
    n = Y.shape[0]
    iu = np.triu_indices(n, k=1)
    return Y[iu]


def marginal_stats(y, label):
    y = np.asarray(y, float)
    return {
        "y_definition": label,
        "n_pairs": int(len(y)),
        "mean": float(y.mean()),
        "var": float(y.var()),
        "var_mean_ratio": float(y.var() / max(y.mean(), 1e-10)),
        "zero_ratio": float((y == 0).mean()),
        "max": float(y.max()),
        "q50": float(np.quantile(y, 0.50)),
        "q90": float(np.quantile(y, 0.90)),
        "q99": float(np.quantile(y, 0.99)),
        "skewness": float(
            ((y - y.mean()) ** 3).mean() / max(y.std() ** 3, 1e-10)),
    }


def rebuild_colike_Y(movie_ids, threshold=LIKE_THRESHOLD):
    """ml-100k.zip から co-like カウント（両ユーザ rating>=threshold）を再構築。"""
    with zipfile.ZipFile(ZIP_PATH) as zf:
        with zf.open("ml-100k/u.data") as f:
            df = pd.read_csv(f, sep="\t", header=None,
                             names=["uid", "mid", "rating", "ts"])
    id_to_idx = {int(m): i for i, m in enumerate(movie_ids)}
    liked = df[(df["rating"] >= threshold) & (df["mid"].isin(id_to_idx))]
    n = len(movie_ids)
    user_movie = np.zeros((liked["uid"].max() + 1, n), dtype=np.int8)
    for uid, mid in zip(liked["uid"].values, liked["mid"].values):
        user_movie[uid, id_to_idx[int(mid)]] = 1
    Y = (user_movie.T.astype(np.int64) @ user_movie.astype(np.int64)).astype(float)
    np.fill_diagonal(Y, 0.0)
    return Y


def ppc_stats(y):
    return {
        "var_mean_ratio": float(y.var() / max(y.mean(), 1e-10)),
        "max": float(y.max()),
        "q99": float(np.quantile(y, 0.99)),
        "zero_ratio": float((y == 0).mean()),
    }


def run_ppc(y_obs, mu_hat, nb_r, n_rep=N_PPC, seed=PPC_SEED):
    """plug-in PPC: Poisson(μ̂) と NB2(μ̂, r̂) から複製を生成し p 値を計算。"""
    rng = np.random.default_rng(seed)
    obs = ppc_stats(y_obs)
    rows = []
    for fam in ("poisson", "nb"):
        reps = {k: [] for k in obs}
        for _ in range(n_rep):
            if fam == "poisson":
                y_rep = rng.poisson(mu_hat).astype(float)
            else:
                lam = rng.gamma(shape=nb_r, scale=mu_hat / nb_r)
                y_rep = rng.poisson(lam).astype(float)
            st = ppc_stats(y_rep)
            for k2 in obs:
                reps[k2].append(st[k2])
        for stat, obs_val in obs.items():
            rep_vals = np.array(reps[stat])
            p_val = float((rep_vals >= obs_val).mean())
            rows.append({
                "ppc_family": fam, "statistic": stat,
                "observed": obs_val,
                "rep_mean": float(rep_vals.mean()),
                "rep_q025": float(np.quantile(rep_vals, 0.025)),
                "rep_q975": float(np.quantile(rep_vals, 0.975)),
                "p_value_geq": p_val,
                "n_rep": n_rep,
            })
    return rows, obs


def fig_distribution(y_obs, mu_hat, nb_r, seed=PPC_SEED + 1):
    """観測 Y ヒストグラム + Poisson 複製のステップ重ね描き。

    注: k=5 の条件付き診断で r̂ が実質 ∞（残差過分散なし）となったため、
    NB 複製は Poisson と同一になる。r̂ < 1e4 のときのみ NB も描く。
    """
    rng = np.random.default_rng(seed)
    y_pois = rng.poisson(mu_hat).astype(float)

    bins = np.arange(0, y_obs.max() + 4, 3)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(y_obs, bins=bins, color=C_OBS, alpha=0.45,
            label="Observed", edgecolor="none")
    ax.hist(y_pois, bins=bins, histtype="step", linewidth=2.0,
            color=C_POIS, label="Poisson replicate (from fitted μ̂)")
    if nb_r < 1e4:
        lam = rng.gamma(shape=nb_r, scale=mu_hat / nb_r)
        y_nb = rng.poisson(lam).astype(float)
        ax.hist(y_nb, bins=bins, histtype="step", linewidth=2.0,
                color=C_NB, label=f"NB replicate (r̂={nb_r:.1f})")
    ax.set_xlabel("Y_count (co-rating count, upper-triangle pairs)")
    ax.set_ylabel("Number of pairs")
    ax.set_title(
        "MovieLens Y_count vs plug-in Poisson replicate (k=5 fit)\n"
        "Marginal var/mean≈9.9 is reproduced by latent heterogeneity in μ̂")
    ax.annotate(
        f"observed marginal var/mean = {y_obs.var()/y_obs.mean():.2f}\n"
        f"Poisson replicate var/mean = {y_pois.var()/y_pois.mean():.2f}",
        xy=(0.62, 0.55), xycoords="axes fraction", fontsize=8,
        color="#52514e")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"movielens_y_distribution.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


def fig_mean_variance(y_obs, mu_hat, nb_r, n_bins=12):
    """μ̂ ビン内の観測分散 vs Poisson / NB2 理論分散。"""
    order = np.argsort(mu_hat)
    y_s, mu_s = y_obs[order], mu_hat[order]
    edges = np.array_split(np.arange(len(mu_s)), n_bins)
    bin_mu, bin_var = [], []
    for idx in edges:
        if len(idx) < 10:
            continue
        bin_mu.append(mu_s[idx].mean())
        bin_var.append(((y_s[idx] - mu_s[idx]) ** 2).mean())
    bin_mu, bin_var = np.array(bin_mu), np.array(bin_var)

    grid = np.linspace(bin_mu.min() * 0.9, bin_mu.max() * 1.05, 100)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(grid, grid, color=C_POIS, linewidth=2.0,
            label="Poisson: Var = μ")
    if nb_r < 1e4:
        ax.plot(grid, grid + grid ** 2 / nb_r, color=C_NB, linewidth=2.0,
                label=f"NB2: Var = μ + μ²/r̂  (r̂={nb_r:.1f})")
    ax.scatter(bin_mu, bin_var, s=42, color=C_OBS, zorder=3,
               label="Observed (binned by μ̂)")
    ax.set_xlabel("Fitted mean μ̂ (bin average)")
    ax.set_ylabel("Conditional variance of Y (bin average)")
    ax.set_title(
        "Mean–variance relationship, MovieLens Y_count (k=5, in-sample)\n"
        "Conditional variance ≤ μ: no residual overdispersion in-sample\n"
        "(below-line points suggest in-sample optimism → see held-out check)")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"movielens_mean_variance.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=_ROOT).stdout.strip()
    except Exception:
        return "unknown"


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    Y_count = np.load(DATA_DIR / "movielens_Y_count.npy").astype(float)
    X = np.load(DATA_DIR / "movielens_X_genre.npy").astype(float)
    movie_ids = np.load(DATA_DIR / "movielens_movie_ids.npy")
    Y_b10 = np.load(DATA_DIR / "movielens_Y_binary_t10.npy").astype(float)
    Y_b20 = np.load(DATA_DIR / "movielens_Y_binary_t20.npy").astype(float)
    n, d = X.shape

    # ── (a) 周辺統計 ─────────────────────────────────────────────────
    rows = []
    y_up = upper_vals(Y_count)
    rows.append({**marginal_stats(y_up, "co_rating_count (movielens_Y_count.npy)"),
                 "diag_type": "marginal"})
    Y_colike = rebuild_colike_Y(movie_ids)
    y_cl = upper_vals(Y_colike)
    rows.append({**marginal_stats(
        y_cl, f"co_like_count (rebuilt from ml-100k.zip, rating>={LIKE_THRESHOLD})"),
        "diag_type": "marginal"})
    for Yb, lbl in ((Y_b10, "binary_t10"), (Y_b20, "binary_t20")):
        rows.append({**marginal_stats(upper_vals(Yb), lbl),
                     "diag_type": "marginal_binary"})

    print("=== Marginal stats ===")
    for r in rows:
        print(f"  {r['y_definition'][:50]:52s} mean={r['mean']:.2f} "
              f"var/mean={r['var_mean_ratio']:.2f} zero={r['zero_ratio']:.3f}")

    # ── (b)(c) 条件付き診断: Poisson フィット + Pearson 過分散 + r̂ ────
    cond_rows = []
    mu_best, r_hat_k5 = None, None
    for k in K_LIST:
        print(f"\n=== Conditional diagnosis: Poisson fit (k={k}) ===")
        res = run_em_experimental(
            X, Y_count, family_x="bernoulli", family_y="poisson",
            k=k, L=L, num_iter=NITER, seed=MODEL_SEED + k)
        mu = predict_mu_y(res)
        iu = np.triu_indices(n, k=1)
        y_pairs, mu_pairs = Y_count[iu], mu[iu]
        disp = pearson_dispersion(y_pairs, mu_pairs)
        r_hat = moment_estimate_nb_r(y_pairs, mu_pairs)
        cond_rows.append({
            "diag_type": "conditional", "k": k,
            "model_seed": MODEL_SEED + k, "L": L, "num_iter": NITER,
            "pearson_dispersion": disp,
            "nb_r_moment_estimate": r_hat,
            "w0_est": res["w0"], "w_est": res["w"],
            "bic": res["bic"], "q_strict": res["Q_strict"],
            "nan_occurred": res["nan_occurred"],
            "runtime_s": res["runtime_s"],
        })
        print(f"  pearson_dispersion={disp:.3f}  r_hat={r_hat:.2f}  "
              f"w0={res['w0']:.3f} w={res['w']:.3f}  [{res['runtime_s']}s]")
        if k == 5:
            mu_best, r_hat_k5 = mu_pairs, r_hat
            y_best = y_pairs

    diag_df = pd.concat([pd.DataFrame(rows), pd.DataFrame(cond_rows)],
                        ignore_index=True)
    diag_path = OUT_DIR / "movielens_overdispersion_diagnostics.csv"
    diag_df.to_csv(diag_path, index=False)
    print(f"\nSaved: {diag_path}")

    # ── (d) PPC ──────────────────────────────────────────────────────
    print("\n=== PPC (plug-in, k=5 fit) ===")
    ppc_rows, obs_stats = run_ppc(y_best, mu_best, r_hat_k5)
    ppc_df = pd.DataFrame(ppc_rows)
    ppc_path = OUT_DIR / "movielens_ppc_summary.csv"
    ppc_df.to_csv(ppc_path, index=False)
    print(ppc_df[["ppc_family", "statistic", "observed", "rep_mean",
                  "p_value_geq"]].to_string(index=False))
    print(f"Saved: {ppc_path}")

    # ── 図 ───────────────────────────────────────────────────────────
    fig_distribution(y_best, mu_best, r_hat_k5)
    fig_mean_variance(y_best, mu_best, r_hat_k5)
    print(f"Figures: {FIG_DIR}")

    # ── run info ─────────────────────────────────────────────────────
    runinfo = pd.DataFrame([{
        "script": "tools/overdispersion/diagnose_movielens_overdispersion.py",
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "branch": "research/overdispersion-z-ablation",
        "n": n, "d": d, "k_list": str(K_LIST),
        "model_seed_base": MODEL_SEED, "ppc_seed": PPC_SEED,
        "L": L, "num_iter": NITER, "n_ppc": N_PPC,
        "model_class": "DualExpFamLSMMasked (fixed-lineage, full mask)",
        "total_runtime_s": round(time.perf_counter() - t0, 1),
    }])
    runinfo.to_csv(OUT_DIR / "movielens_overdispersion_runinfo.csv", index=False)
    print(f"\nTotal: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
