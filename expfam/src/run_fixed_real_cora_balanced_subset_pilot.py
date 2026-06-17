"""
Cora balanced subset pilot — subset 選択 strategy 比較スクリプト。

Strategy A (balanced_random):  各クラスから最大 PER_CLASS 件をランダムに選ぶ。
Strategy B (balanced_degree):  各クラス内で全グラフ次数が高いノードを PER_CLASS 件選ぶ。

前回の BFS subset 結果と比較する。

使用データ:
  expfam/data/cora/cora.content  (既存、再ダウンロード不要)
  expfam/data/cora/cora.cites

出力:
  expfam/results/real_data/cora_balanced_subset_pilot/
  expfam/figures/real_data/cora_balanced_subset_pilot/
"""

import sys
import os
import time
import traceback
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, silhouette_score,
    normalized_mutual_info_score, adjusted_rand_score,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

_SRC  = Path(__file__).parent
_ROOT = _SRC.parent.parent

sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from model_dual_expfam_fixed import DualExpFamLSMFixed
from utils_expfam import calc_Q_dual_strict, calc_bic_dual

# ─────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────

DATA_DIR    = _ROOT / "expfam" / "data" / "cora"
OUT_DIR     = _ROOT / "expfam" / "results" / "real_data" / "cora_balanced_subset_pilot"
FIG_DIR     = _ROOT / "expfam" / "figures" / "real_data" / "cora_balanced_subset_pilot"
BFS_OUT_DIR = _ROOT / "expfam" / "results" / "real_data" / "cora_subset_pilot"

PER_CLASS   = 40
D_SUBSET    = 50
FAMILY_X    = "bernoulli"
FAMILY_Y    = "bernoulli"
K_LIST      = [2, 3, 5]
SEED_BASE   = 4000    # balanced_random: +k*10,  balanced_degree: +1000+k*10
L, NITER    = 5, 8

STRATEGIES  = ["balanced_random", "balanced_degree"]

CORA_COLORS = [
    "#2196F3", "#FF5722", "#4CAF50", "#9C27B0",
    "#FF9800", "#795548", "#607D8B",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 6,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────────────
# Data parsing (no download — data must exist)
# ─────────────────────────────────────────────────────────────────────

def parse_cora():
    content_path = DATA_DIR / "cora.content"
    cites_path   = DATA_DIR / "cora.cites"
    assert content_path.exists(), f"Not found: {content_path}"
    assert cites_path.exists(),   f"Not found: {cites_path}"

    node_ids, features, labels = [], [], []
    label_set = []
    with open(content_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            node_ids.append(parts[0])
            features.append([int(x) for x in parts[1:-1]])
            lbl = parts[-1]
            if lbl not in label_set:
                label_set.append(lbl)
            labels.append(lbl)

    node_ids  = np.array(node_ids)
    X_full    = np.array(features, dtype=np.float32)
    label_idx = np.array([label_set.index(l) for l in labels], dtype=int)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    edges = set()
    with open(cites_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            if u in id_to_idx and v in id_to_idx:
                i, j = id_to_idx[u], id_to_idx[v]
                if i != j:
                    edges.add((min(i, j), max(i, j)))

    n_full = len(node_ids)
    print(f"Full Cora: n={n_full}, d={X_full.shape[1]}, edges={len(edges)}, labels={len(label_set)}")

    # full-graph degree (undirected)
    degree = np.zeros(n_full, dtype=int)
    for (i, j) in edges:
        degree[i] += 1
        degree[j] += 1

    return node_ids, X_full, label_idx, label_set, edges, degree


# ─────────────────────────────────────────────────────────────────────
# Subset builders
# ─────────────────────────────────────────────────────────────────────

def build_balanced_random(label_idx, n_classes, per_class, rng):
    selected = []
    counts   = {}
    for c in range(n_classes):
        idx_c = np.where(label_idx == c)[0]
        k_c   = min(per_class, len(idx_c))
        chosen = rng.choice(idx_c, size=k_c, replace=False)
        selected.extend(chosen.tolist())
        counts[c] = k_c
    return np.array(selected), counts


def build_balanced_degree(label_idx, n_classes, per_class, degree):
    selected = []
    counts   = {}
    for c in range(n_classes):
        idx_c  = np.where(label_idx == c)[0]
        k_c    = min(per_class, len(idx_c))
        # 次数降順ソート → 上位 k_c
        order  = np.argsort(-degree[idx_c])
        chosen = idx_c[order[:k_c]]
        selected.extend(chosen.tolist())
        counts[c] = k_c
    return np.array(selected), counts


def subset_stats(subset_idx, X_full, label_idx, edges, label_set, d_sub):
    n_sub  = len(subset_idx)
    subset_set = {v: i for i, v in enumerate(subset_idx)}

    # select top-d_sub columns by frequency in THIS subset
    X_sub_full = X_full[subset_idx, :]
    col_sums   = X_sub_full.sum(axis=0)
    top_cols   = np.argsort(-col_sums)[:d_sub]
    X_sub      = X_sub_full[:, top_cols]

    # Y matrix
    Y_sub      = np.zeros((n_sub, n_sub), dtype=float)
    adj_list   = [[] for _ in range(n_sub)]
    n_edges    = 0
    for (u, v) in edges:
        if u in subset_set and v in subset_set:
            i, j = subset_set[u], subset_set[v]
            Y_sub[i, j] = 1.0
            Y_sub[j, i] = 1.0
            adj_list[i].append(j)
            adj_list[j].append(i)
            n_edges += 1
    np.fill_diagonal(Y_sub, 0.0)

    upper_mask = np.triu(np.ones((n_sub, n_sub), dtype=bool), k=1)
    density    = float(Y_sub[upper_mask].mean())

    # isolated nodes
    node_degree_sub = np.array([len(adj_list[i]) for i in range(n_sub)])
    n_isolated       = int((node_degree_sub == 0).sum())

    # largest connected component (BFS)
    visited = [False] * n_sub
    comp_sizes = []
    for start in range(n_sub):
        if not visited[start]:
            comp = []
            q    = deque([start])
            visited[start] = True
            while q:
                nd = q.popleft()
                comp.append(nd)
                for nb in adj_list[nd]:
                    if not visited[nb]:
                        visited[nb] = True
                        q.append(nb)
            comp_sizes.append(len(comp))
    lcc_size = max(comp_sizes) if comp_sizes else 0

    label_sub   = label_idx[subset_idx]
    class_counts = {label_set[c]: int(np.sum(label_sub == c)) for c in range(len(label_set))}

    return {
        "X_sub":       X_sub,
        "Y_sub":       Y_sub,
        "label_sub":   label_sub,
        "top_cols":    top_cols,
        "n_sub":       n_sub,
        "n_edges":     n_edges,
        "density":     density,
        "n_isolated":  n_isolated,
        "lcc_size":    lcc_size,
        "class_counts": class_counts,
    }


# ─────────────────────────────────────────────────────────────────────
# EM runner
# ─────────────────────────────────────────────────────────────────────

def run_em_fixed_cora(X, Y, family_x, family_y, k, L=5, num_iter=8, seed=42):
    n, d = X.shape
    max_retries = 2

    for retry in range(max_retries + 1):
        newton_alpha = 0.5 / (2 ** retry)
        rng   = np.random.default_rng(seed + retry * 1000)
        model = DualExpFamLSMFixed(n=n, d=d, k=k, L=L, family_x=family_x, family_y=family_y)
        model.initialize_params(true_params=None, seed=seed + retry * 1000)

        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if family_y == "bernoulli":
            density = float(np.clip(Y[upper_mask].mean(), 1e-6, 1 - 1e-6))
            model.params["w0"] = np.log(density / (1 - density))
            model.params["w"]  = 0.5
        elif family_y == "poisson":
            upper = Y[upper_mask]
            mean_cnt = float(upper[upper > 0].mean()) if np.any(upper > 0) else 1.0
            model.params["w0"] = np.log(mean_cnt + 1e-10)
            model.params["w"]  = 0.1 / (2 ** retry)
        else:
            upper_vals = Y[upper_mask]
            model.params["w0"] = float(upper_vals.mean())
            model.params["w"]  = 0.5
            model.sigma_y      = float(max(upper_vals.std(), 0.01))

        if family_x in ("bernoulli", "poisson"):
            model.params["F"] *= 0.2

        Z     = model.params["Z"].copy()
        F     = model.params["F"].copy()
        sigma = model.params["sigma"].copy()
        w0    = model.params["w0"]
        w     = model.params["w"]
        var_z = model.params["var_z"]
        Z_prev    = Z.copy()
        nan_count = 0

        for _ in range(1, num_iter + 1):
            Z_samples = np.zeros((n, k, L))
            for l in range(L):
                model.params.update(dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w))
                Z_new = model.calc_eta_newton(X, Y, rng=rng, max_iter=10, alpha=newton_alpha)
                Z_samples[:, :, l] = Z_new
                Z = Z_new.copy()

            if np.any(np.isnan(Z_samples)) or np.any(np.isinf(Z_samples)):
                nan_count += 1
                Z_samples = np.stack([Z_prev] * L, axis=2)
                Z = Z_prev.copy()

            Z_samples = model.scale_Z(Z_samples)
            Z_prev    = Z.copy()
            Z         = Z_samples[:, :, -1].copy()
            F         = model.calc_F(X, Z_samples)
            sigma     = model.calc_sigma(X, Z_samples, F)
            w0        = model.calc_w0(Y, Z_samples, w0, w, max_iter=50)
            w         = model.calc_w(Y, Z_samples, w0, w, max_iter=50)
            if family_y == "gaussian":
                model.calc_sigma_y(Y, Z_samples, w0, w)

        nan_occurred = nan_count > 0
        if not nan_occurred:
            break

    model.params.update({"F": F, "sigma": sigma, "w0": w0, "w": w})
    Q_strict   = calc_Q_dual_strict(X, Y, Z_samples, F, sigma, var_z, w0, w, model)
    bic, npar  = calc_bic_dual(Q_strict, k, n, d, family_x, family_y)

    Z_est  = Z_samples[:, :, -1]
    eta_y  = float(w0) + float(w) * (Z_est @ Z_est.T)
    mu_y   = model._mean_function(eta_y)
    y_true = Y[upper_mask].astype(int)
    y_scr  = mu_y[upper_mask]

    try:
        auc_y = float(roc_auc_score(y_true, y_scr))
        ap_y  = float(average_precision_score(y_true, y_scr))
    except Exception:
        auc_y = float("nan")
        ap_y  = float("nan")

    mu_x   = model._mean_function_x(Z_est @ F.T)
    rmse_x = float(np.sqrt(np.mean((X - mu_x) ** 2)))

    return {
        "Q_strict": Q_strict, "bic": bic, "num_params": npar,
        "auc_y": auc_y, "ap_y": ap_y, "rmse_x": rmse_x,
        "w0": float(w0), "w": float(w),
        "Z_est": Z_est, "nan_occurred": nan_occurred, "nan_count": nan_count,
    }


# ─────────────────────────────────────────────────────────────────────
# Z metrics
# ─────────────────────────────────────────────────────────────────────

def compute_z_metrics(Z_est, labels, n_clusters=None):
    k     = Z_est.shape[1]
    n_cl  = n_clusters or len(np.unique(labels))
    if k == 1:
        Z_2d = np.hstack([Z_est, np.zeros((len(Z_est), 1))])
    elif k == 2:
        Z_2d = Z_est.copy()
    else:
        Z_2d = PCA(n_components=2).fit_transform(Z_est)

    try:
        sil = float(silhouette_score(Z_2d, labels))
    except Exception:
        sil = float("nan")
    try:
        km  = KMeans(n_clusters=n_cl, random_state=42, n_init=10).fit_predict(Z_2d)
        nmi = float(normalized_mutual_info_score(labels, km))
        ari = float(adjusted_rand_score(labels, km))
    except Exception:
        nmi = ari = float("nan")
    return sil, nmi, ari, Z_2d


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────

def save_fig(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = FIG_DIR / f"{stem}.{ext}"
        if p.exists():
            print(f"  SKIP (exists): {p}")
        else:
            fig.savefig(p, dpi=200, bbox_inches="tight")
            print(f"  Saved: {p}")
    plt.close(fig)


def make_z_by_label_figure(all_z_results, label_name_list):
    """
    2 rows (strategy) × 3 cols (k) の Z 可視化
    all_z_results: dict {strategy: {k: (Z_est, label_sub)}}
    """
    n_strats = len(STRATEGIES)
    n_ks     = len(K_LIST)
    n_labels = len(label_name_list)
    colors   = CORA_COLORS[:n_labels]

    fig, axes = plt.subplots(n_strats, n_ks, figsize=(5.0 * n_ks, 4.8 * n_strats))

    for r, strat in enumerate(STRATEGIES):
        for c, k_val in enumerate(K_LIST):
            ax = axes[r][c]
            if strat not in all_z_results or k_val not in all_z_results[strat]:
                ax.set_title(f"{strat}\nk={k_val}\n(no data)")
                continue
            Z_est, label_sub = all_z_results[strat][k_val]
            kk = Z_est.shape[1]
            pca_used = kk > 2
            if kk == 1:
                Z_2d = np.hstack([Z_est, np.zeros((len(Z_est), 1))])
            elif kk == 2:
                Z_2d = Z_est
            else:
                Z_2d = PCA(n_components=2).fit_transform(Z_est)

            for ci in range(n_labels):
                mask = label_sub == ci
                if mask.sum() == 0:
                    continue
                short = label_name_list[ci].replace("_", "\n")
                ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                           c=colors[ci], label=short,
                           s=20, alpha=0.75, edgecolors="none")

            ax.set_xlabel("PC1" if pca_used else "Z₁", fontsize=8)
            ax.set_ylabel("PC2" if pca_used else "Z₂", fontsize=8)
            title_strat = strat.replace("_", "\n")
            ax.set_title(f"{title_strat}  k={k_val}", fontsize=9)
            ax.legend(fontsize=5, loc="best", framealpha=0.6)
            ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle(
        "Cora Balanced Subset Pilot — Z Visualization by Topic Label\n"
        f"(d={D_SUBSET}, X=Bernoulli, Y=Bernoulli, 1 trial/k)",
        fontsize=11,
    )
    fig.tight_layout()
    save_fig(fig, "cora_balanced_z_by_label")


def make_strategy_comparison_figure(cmp_df):
    """
    3 strategies (BFS, balanced_random, balanced_degree) × 4 指標 のバー比較
    """
    strategies = cmp_df["strategy"].tolist()
    metrics    = [
        ("y_density",  "Y Density"),
        ("best_ap",    "Best AP (max over k)"),
        ("best_auc",   "Best AUC (max over k)"),
        ("best_nmi",   "Best NMI (max over k)"),
    ]
    n_metrics  = len(metrics)
    x          = np.arange(len(strategies))
    bar_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 4.2))

    for ax, (col, label), color in zip(axes, metrics, bar_colors):
        vals = [float(cmp_df.loc[cmp_df["strategy"] == s, col].iloc[0])
                if col in cmp_df.columns else float("nan")
                for s in strategies]
        bars = ax.bar(x, vals, color=color, alpha=0.8, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in strategies], fontsize=7)
        ax.set_title(label, fontsize=9)
        ax.set_ylim(0, max(v for v in vals if not np.isnan(v)) * 1.2 + 1e-8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + ax.get_ylim()[1] * 0.01,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        "Cora Subset Strategy Comparison (BFS vs balanced_random vs balanced_degree)",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, "cora_subset_strategy_comparison")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Parse ──────────────────────────────────────────────────────
    node_ids, X_full, label_idx, label_set, edges, degree = parse_cora()
    n_full    = len(node_ids)
    n_classes = len(label_set)

    print(f"\nClass distribution (full Cora):")
    for c, lbl in enumerate(label_set):
        cnt = int(np.sum(label_idx == c))
        print(f"  {lbl}: {cnt}")

    # ── Build subsets ──────────────────────────────────────────────
    rng_random = np.random.default_rng(4000)

    idx_A, counts_A = build_balanced_random(label_idx, n_classes, PER_CLASS, rng_random)
    idx_B, counts_B = build_balanced_degree(label_idx, n_classes, PER_CLASS, degree)

    stats_map = {}
    for strat, idx, counts in [
        ("balanced_random", idx_A, counts_A),
        ("balanced_degree", idx_B, counts_B),
    ]:
        print(f"\n--- {strat} ---")
        st = subset_stats(idx, X_full, label_idx, edges, label_set, D_SUBSET)
        stats_map[strat] = {"idx": idx, **st}
        print(f"  n_subset={st['n_sub']}, edges={st['n_edges']}, "
              f"density={st['density']:.5f}, isolated={st['n_isolated']}, "
              f"LCC={st['lcc_size']}")
        print(f"  class_counts: " +
              " | ".join(f"{l}:{c}" for l, c in st["class_counts"].items() if c > 0))

    # ── Data summary CSV ───────────────────────────────────────────
    dsummary_rows = []
    for strat in STRATEGIES:
        st = stats_map[strat]
        dsummary_rows.append({
            "strategy":    strat,
            "n_subset":    st["n_sub"],
            "d_subset":    D_SUBSET,
            "n_edges":     st["n_edges"],
            "y_density":   st["density"],
            "n_isolated":  st["n_isolated"],
            "lcc_size":    st["lcc_size"],
            "n_classes":   n_classes,
            "per_class_target": PER_CLASS,
            "class_counts": str(st["class_counts"]),
            "family_x":    FAMILY_X,
            "family_y":    FAMILY_Y,
        })
    ds_df = pd.DataFrame(dsummary_rows)
    out_ds = OUT_DIR / "cora_balanced_data_summary.csv"
    if not out_ds.exists():
        ds_df.to_csv(out_ds, index=False)
        print(f"\nSaved: {out_ds}")

    # ── EM experiment ──────────────────────────────────────────────
    t0 = time.perf_counter()
    total_fits = len(STRATEGIES) * len(K_LIST)
    print(f"\n=== Pilot: {total_fits} fits "
          f"({len(STRATEGIES)} strategies × {len(K_LIST)} k values) ===")

    metric_rows   = []
    z_emb_rows    = []
    all_z_results = {}   # {strat: {k: (Z_est, label_sub)}}
    fit_count     = 0

    for strat in STRATEGIES:
        st          = stats_map[strat]
        X_sub       = st["X_sub"]
        Y_sub       = st["Y_sub"]
        label_sub   = st["label_sub"]
        n_sub       = st["n_sub"]
        n_cl        = len(np.unique(label_sub))
        strat_idx   = STRATEGIES.index(strat)
        all_z_results[strat] = {}

        for k_val in K_LIST:
            fit_count += 1
            seed = SEED_BASE + strat_idx * 1000 + k_val * 10
            print(f"\n[{fit_count}/{total_fits}] {strat}  k={k_val}  seed={seed}")

            try:
                t_k = time.perf_counter()
                res = run_em_fixed_cora(
                    X_sub, Y_sub, FAMILY_X, FAMILY_Y, k=k_val,
                    L=L, num_iter=NITER, seed=seed,
                )
                elapsed_k = time.perf_counter() - t_k
                sil, nmi, ari, Z_2d = compute_z_metrics(
                    res["Z_est"], label_sub, n_clusters=n_cl
                )
                success = True
                err_msg = ""
                print(
                    f"  BIC={res['bic']:.1f}  AUC={res['auc_y']:.4f}  "
                    f"AP={res['ap_y']:.4f}  sil={sil:.4f}  "
                    f"NMI={nmi:.4f}  ARI={ari:.4f}  [{elapsed_k:.1f}s]  "
                    f"w0={res['w0']:.3f}  w={res['w']:.3f}"
                )
                all_z_results[strat][k_val] = (res["Z_est"], label_sub)

            except Exception as e:
                traceback.print_exc()
                res = {
                    "Q_strict": float("nan"), "bic": float("nan"), "num_params": -1,
                    "auc_y": float("nan"), "ap_y": float("nan"), "rmse_x": float("nan"),
                    "w0": float("nan"), "w": float("nan"),
                    "nan_occurred": True, "nan_count": -1,
                }
                sil = nmi = ari = float("nan")
                elapsed_k = float("nan")
                success   = False
                err_msg   = str(e)
                print(f"  ERROR: {err_msg[:80]}")

            metric_rows.append({
                "strategy":    strat,
                "k":           k_val,
                "seed":        seed,
                "n":           n_sub,
                "d":           D_SUBSET,
                "family_x":    FAMILY_X,
                "family_y":    FAMILY_Y,
                "y_density":   st["density"],
                "q_strict":    res["Q_strict"],
                "bic":         res["bic"],
                "num_params":  res["num_params"],
                "auc_y":       res["auc_y"],
                "ap_y":        res["ap_y"],
                "rmse_x":      res["rmse_x"],
                "silhouette":  sil,
                "nmi":         nmi,
                "ari":         ari,
                "w0":          res["w0"],
                "w":           res["w"],
                "nan_occurred": res["nan_occurred"],
                "runtime_s":   elapsed_k,
                "success":     success,
                "error_message": err_msg,
            })

            # Z embeddings rows (best k per strategy = where AP is max)
            if success and "Z_est" in res:
                Z_est = res["Z_est"]
                kk    = Z_est.shape[1]
                pca_used = kk > 2
                if kk == 1:
                    Z_2d_emb = np.hstack([Z_est, np.zeros((n_sub, 1))])
                elif kk == 2:
                    Z_2d_emb = Z_est
                else:
                    Z_2d_emb = PCA(n_components=2).fit_transform(Z_est)

                subset_idx = stats_map[strat]["idx"]
                for i in range(n_sub):
                    row = {
                        "strategy":     strat,
                        "k":            k_val,
                        "sample_id":    i,
                        "orig_node_id": str(node_ids[subset_idx[i]]),
                        "label":        int(label_sub[i]),
                        "label_name":   label_set[int(label_sub[i])],
                        "z_pc1":        float(Z_2d_emb[i, 0]),
                        "z_pc2":        float(Z_2d_emb[i, 1]),
                        "pca_used":     pca_used,
                    }
                    for j in range(kk):
                        row[f"z_{j+1}"] = float(Z_est[i, j])
                    z_emb_rows.append(row)

    # ── Metrics CSV ────────────────────────────────────────────────
    km_df = pd.DataFrame(metric_rows)
    out_km = OUT_DIR / "cora_balanced_k_metrics.csv"
    if not out_km.exists():
        km_df.to_csv(out_km, index=False)
        print(f"\nSaved: {out_km}  ({len(km_df)} rows)")

    # ── Z embeddings CSV ───────────────────────────────────────────
    ze_df  = pd.DataFrame(z_emb_rows)
    out_ze = OUT_DIR / "cora_balanced_z_embeddings.csv"
    if not out_ze.exists() and len(z_emb_rows) > 0:
        ze_df.to_csv(out_ze, index=False)
        print(f"Saved: {out_ze}  ({len(ze_df)} rows)")

    # ── Strategy comparison CSV (BFS + balanced) ───────────────────
    cmp_rows = []

    # BFS results (from previous pilot)
    bfs_path = BFS_OUT_DIR / "cora_subset_k_metrics.csv"
    if bfs_path.exists():
        bfs_df = pd.read_csv(bfs_path)
        bfs_suc = bfs_df[bfs_df["success"] == True]
        bfs_ds  = pd.read_csv(BFS_OUT_DIR / "cora_subset_data_summary.csv")
        bfs_density = float(bfs_ds["y_density"].iloc[0])
        bfs_n       = int(bfs_ds["n_subset"].iloc[0])
        bfs_edges   = int(bfs_ds["n_edges_subset"].iloc[0])
        bfs_classes = int(bfs_ds["n_classes_subset"].iloc[0])
        cmp_rows.append({
            "strategy":    "BFS",
            "n_subset":    bfs_n,
            "d_subset":    D_SUBSET,
            "n_classes_present": bfs_classes,
            "y_density":   bfs_density,
            "n_edges":     bfs_edges,
            "n_isolated":  "N/A",
            "lcc_size":    "N/A",
            "best_auc":    float(bfs_suc["auc_y"].max()) if len(bfs_suc) > 0 else float("nan"),
            "best_ap":     float(bfs_suc["ap_y"].max())  if len(bfs_suc) > 0 else float("nan"),
            "best_nmi":    float(bfs_suc["nmi"].max())   if len(bfs_suc) > 0 else float("nan"),
            "best_ari":    float(bfs_suc["ari"].max())   if len(bfs_suc) > 0 else float("nan"),
            "best_sil":    float(bfs_suc["silhouette"].max()) if len(bfs_suc) > 0 else float("nan"),
            "class_balance": "skewed (Genetic_Alg=78%)",
        })

    for strat in STRATEGIES:
        st  = stats_map[strat]
        suc = km_df[(km_df["strategy"] == strat) & (km_df["success"] == True)]
        cmp_rows.append({
            "strategy":    strat,
            "n_subset":    st["n_sub"],
            "d_subset":    D_SUBSET,
            "n_classes_present": len([v for v in st["class_counts"].values() if v > 0]),
            "y_density":   st["density"],
            "n_edges":     st["n_edges"],
            "n_isolated":  st["n_isolated"],
            "lcc_size":    st["lcc_size"],
            "best_auc":    float(suc["auc_y"].max()) if len(suc) > 0 else float("nan"),
            "best_ap":     float(suc["ap_y"].max())  if len(suc) > 0 else float("nan"),
            "best_nmi":    float(suc["nmi"].max())   if len(suc) > 0 else float("nan"),
            "best_ari":    float(suc["ari"].max())   if len(suc) > 0 else float("nan"),
            "best_sil":    float(suc["silhouette"].max()) if len(suc) > 0 else float("nan"),
            "class_balance": "balanced (7 classes, ~" + str(PER_CLASS) + " each)",
        })

    cmp_df   = pd.DataFrame(cmp_rows)
    out_cmp  = OUT_DIR / "cora_subset_strategy_comparison.csv"
    if not out_cmp.exists():
        cmp_df.to_csv(out_cmp, index=False)
        print(f"Saved: {out_cmp}  ({len(cmp_df)} rows)")

    # ── Figures ────────────────────────────────────────────────────
    print("\n=== Figures ===")
    make_z_by_label_figure(all_z_results, label_set)
    make_strategy_comparison_figure(cmp_df)

    # ── Final summary ──────────────────────────────────────────────
    elapsed_total = (time.perf_counter() - t0) / 60
    suc_count     = int(km_df["success"].sum())
    print(f"\nDone in {elapsed_total:.1f} min  ({suc_count}/{total_fits} fits)")

    print("\n=== k別指標 ===")
    for _, row in km_df.iterrows():
        if row["success"]:
            print(f"  {row['strategy']:20s}  k={int(row['k'])}: "
                  f"BIC={row['bic']:.1f}  AUC={row['auc_y']:.4f}  AP={row['ap_y']:.4f}  "
                  f"sil={row['silhouette']:.4f}  NMI={row['nmi']:.4f}  ARI={row['ari']:.4f}"
                  f"  [{row['runtime_s']:.0f}s]")
        else:
            print(f"  {row['strategy']:20s}  k={int(row['k'])}: FAILED")

    print("\n=== Strategy comparison ===")
    print(cmp_df[["strategy", "n_subset", "y_density", "n_edges",
                  "n_isolated", "lcc_size", "best_ap", "best_auc", "best_nmi"]].to_string(index=False))

    nan_fits = int(km_df["nan_occurred"].sum())
    if nan_fits > 0:
        print(f"\nWARNING: {nan_fits} fits had NaN in E-step (fallback used)")


if __name__ == "__main__":
    main()
