"""
Cora balanced-degree subset — k=1~6 mini BIC sweep スクリプト。

strategy = balanced_degree（各クラス内で全グラフ次数上位 PER_CLASS 件）
n_subset = 280 (7 classes × 40)
d_subset = 50
family_x = bernoulli / family_y = bernoulli

k_values = 1, 2, 3, 4, 5, 6
trials   = 0, 1, 2      → 18 fits

目的:
  - BIC最小 k の特定
  - Y AUC / AP が最良の k の特定
  - NMI / ARI が最良の k の特定
  - BIC と再構成性能のトレードオフ確認

出力:
  expfam/results/real_data/cora_balanced_k_sweep/
  expfam/figures/real_data/cora_balanced_k_sweep/
"""

import sys
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

DATA_DIR = _ROOT / "expfam" / "data" / "cora"
OUT_DIR  = _ROOT / "expfam" / "results" / "real_data" / "cora_balanced_k_sweep"
FIG_DIR  = _ROOT / "expfam" / "figures" / "real_data" / "cora_balanced_k_sweep"

PER_CLASS = 40
D_SUBSET  = 50
FAMILY_X  = "bernoulli"
FAMILY_Y  = "bernoulli"
K_LIST    = [1, 2, 3, 4, 5, 6]
N_TRIALS  = 3
SEED_BASE = 6000    # seed = SEED_BASE + k*100 + trial*10
L, NITER  = 5, 8

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
    "legend.fontsize": 7,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────────────
# Data parsing
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

    edges = set()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
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

    degree = np.zeros(len(node_ids), dtype=int)
    for (i, j) in edges:
        degree[i] += 1
        degree[j] += 1

    print(f"Full Cora: n={len(node_ids)}, d={X_full.shape[1]}, "
          f"edges={len(edges)}, labels={len(label_set)}")
    return node_ids, X_full, label_idx, label_set, edges, degree


# ─────────────────────────────────────────────────────────────────────
# Subset builder (balanced_degree — same as previous pilot)
# ─────────────────────────────────────────────────────────────────────

def build_balanced_degree(label_idx, n_classes, per_class, degree):
    selected, counts = [], {}
    for c in range(n_classes):
        idx_c  = np.where(label_idx == c)[0]
        k_c    = min(per_class, len(idx_c))
        order  = np.argsort(-degree[idx_c])
        chosen = idx_c[order[:k_c]]
        selected.extend(chosen.tolist())
        counts[c] = k_c
    return np.array(selected), counts


def build_subset(subset_idx, X_full, label_idx, edges, label_set, d_sub):
    n_sub      = len(subset_idx)
    subset_set = {v: i for i, v in enumerate(subset_idx)}

    X_sub_full = X_full[subset_idx, :]
    col_sums   = X_sub_full.sum(axis=0)
    top_cols   = np.argsort(-col_sums)[:d_sub]
    X_sub      = X_sub_full[:, top_cols]

    Y_sub    = np.zeros((n_sub, n_sub), dtype=float)
    adj_list = [[] for _ in range(n_sub)]
    n_edges  = 0
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

    node_deg   = np.array([len(adj_list[i]) for i in range(n_sub)])
    n_isolated = int((node_deg == 0).sum())

    visited, comp_sizes = [False] * n_sub, []
    for start in range(n_sub):
        if not visited[start]:
            comp, q = [], deque([start])
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

    label_sub    = label_idx[subset_idx]
    class_counts = {label_set[c]: int(np.sum(label_sub == c))
                    for c in range(len(label_set))}

    return {
        "X_sub": X_sub, "Y_sub": Y_sub, "label_sub": label_sub,
        "top_cols": top_cols, "n_sub": n_sub, "n_edges": n_edges,
        "density": density, "n_isolated": n_isolated, "lcc_size": lcc_size,
        "class_counts": class_counts, "upper_mask": upper_mask,
    }


# ─────────────────────────────────────────────────────────────────────
# EM runner
# ─────────────────────────────────────────────────────────────────────

def run_em_fixed(X, Y, family_x, family_y, k, L=5, num_iter=8, seed=42):
    n, d = X.shape
    for retry in range(3):
        newton_alpha = 0.5 / (2 ** retry)
        rng   = np.random.default_rng(seed + retry * 1000)
        model = DualExpFamLSMFixed(n=n, d=d, k=k, L=L,
                                   family_x=family_x, family_y=family_y)
        model.initialize_params(true_params=None, seed=seed + retry * 1000)

        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if family_y == "bernoulli":
            density = float(np.clip(Y[upper_mask].mean(), 1e-6, 1 - 1e-6))
            model.params["w0"] = np.log(density / (1 - density))
            model.params["w"]  = 0.5
        if family_x in ("bernoulli", "poisson"):
            model.params["F"] *= 0.2

        Z     = model.params["Z"].copy()
        F     = model.params["F"].copy()
        sigma = model.params["sigma"].copy()
        w0    = model.params["w0"]
        w     = model.params["w"]
        var_z = model.params["var_z"]
        Z_prev, nan_count = Z.copy(), 0

        for _ in range(1, num_iter + 1):
            Z_samples = np.zeros((n, k, L))
            for l in range(L):
                model.params.update(dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w))
                Z_new = model.calc_eta_newton(
                    X, Y, rng=rng, max_iter=10, alpha=newton_alpha)
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

        if nan_count == 0:
            break

    model.params.update({"F": F, "sigma": sigma, "w0": w0, "w": w})
    Q_strict  = calc_Q_dual_strict(X, Y, Z_samples, F, sigma, var_z, w0, w, model)
    bic, npar = calc_bic_dual(Q_strict, k, n, d, family_x, family_y)

    Z_est  = Z_samples[:, :, -1]
    eta_y  = float(w0) + float(w) * (Z_est @ Z_est.T)
    mu_y   = model._mean_function(eta_y)
    y_true = Y[upper_mask].astype(int)
    y_scr  = mu_y[upper_mask]

    try:
        auc_y = float(roc_auc_score(y_true, y_scr))
        ap_y  = float(average_precision_score(y_true, y_scr))
    except Exception:
        auc_y = ap_y = float("nan")

    mu_x   = model._mean_function_x(Z_est @ F.T)
    rmse_x = float(np.sqrt(np.mean((X - mu_x) ** 2)))

    return {
        "Q_strict": Q_strict, "bic": bic, "num_params": npar,
        "auc_y": auc_y, "ap_y": ap_y, "rmse_x": rmse_x,
        "w0": float(w0), "w": float(w), "Z_est": Z_est,
        "nan_occurred": nan_count > 0, "nan_count": nan_count,
    }


# ─────────────────────────────────────────────────────────────────────
# Z metrics
# ─────────────────────────────────────────────────────────────────────

def compute_z_metrics(Z_est, labels, n_clusters=None):
    k_z   = Z_est.shape[1]
    n_cl  = n_clusters or len(np.unique(labels))
    if k_z == 1:
        Z_2d = np.hstack([Z_est, np.zeros((len(Z_est), 1))])
    elif k_z == 2:
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


def make_bic_curve(agg_df, best_k_bic):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ks  = agg_df["k"].values
    mu  = agg_df["bic_mean"].values
    sd  = agg_df["bic_std"].values
    ax.errorbar(ks, mu, yerr=sd, fmt="o-", color="#2196F3",
                linewidth=2, markersize=7, capsize=4, label="BIC (mean ± std)")
    ax.axvline(best_k_bic, color="#FF5722", linestyle="--", linewidth=1.5,
               label=f"Best k={best_k_bic}")
    ax.set_xlabel("k (latent dimension)")
    ax.set_ylabel("BIC")
    ax.set_xticks(ks.tolist())
    ax.set_title(
        f"Cora balanced_degree subset — BIC vs k\n"
        f"(n=280, d=50, X=Bern, Y=Bern, {N_TRIALS} trials/k)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "cora_balanced_k_bic_curve")


def make_metrics_figure(agg_df, best_k_bic, best_k_ap, best_k_nmi):
    metrics = [
        ("auc_mean",       "auc_std",       "Y AUC",       "#2196F3"),
        ("ap_mean",        "ap_std",         "Y AP",        "#FF5722"),
        ("nmi_mean",       "nmi_std",        "NMI",         "#4CAF50"),
        ("ari_mean",       "ari_std",        "ARI",         "#9C27B0"),
        ("silhouette_mean","silhouette_std",  "Silhouette",  "#FF9800"),
    ]
    n_m  = len(metrics)
    fig, axes = plt.subplots(1, n_m, figsize=(4.5 * n_m, 4.0))
    ks = agg_df["k"].values

    for ax, (mu_col, sd_col, label, color) in zip(axes, metrics):
        mu = agg_df[mu_col].values
        sd = agg_df[sd_col].values if sd_col in agg_df.columns else np.zeros_like(mu)
        ax.errorbar(ks, mu, yerr=sd, fmt="o-", color=color,
                    linewidth=2, markersize=6, capsize=3)
        if label == "Y AP":
            ax.axvline(best_k_ap, color="gray", linestyle=":", linewidth=1.4,
                       label=f"best AP k={best_k_ap}")
        if label == "NMI":
            ax.axvline(best_k_nmi, color="gray", linestyle=":", linewidth=1.4,
                       label=f"best NMI k={best_k_nmi}")
        ax.axvline(best_k_bic, color="#FF5722", linestyle="--", linewidth=1.2,
                   label=f"best BIC k={best_k_bic}")
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=9)
        ax.set_xticks(ks.tolist())
        ax.legend(fontsize=6)
        ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle(
        "Cora balanced_degree — Metrics vs k (mean ± std)\n"
        f"(n=280, d=50, X=Bern, Y=Bern, {N_TRIALS} trials/k)",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, "cora_balanced_k_metrics")


def make_z_figure(Z_est, label_sub, label_set, title, stem):
    k_z      = Z_est.shape[1]
    pca_used = k_z > 2
    if k_z == 1:
        Z_2d = np.hstack([Z_est, np.zeros((len(Z_est), 1))])
    elif k_z == 2:
        Z_2d = Z_est
    else:
        Z_2d = PCA(n_components=2).fit_transform(Z_est)

    n_labels = len(label_set)
    colors   = CORA_COLORS[:n_labels]

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for c in range(n_labels):
        mask = label_sub == c
        if mask.sum() == 0:
            continue
        short = label_set[c].replace("_", "\n")
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                   c=colors[c], label=short, s=28, alpha=0.78, edgecolors="none")
    ax.set_xlabel("PC1" if pca_used else "Z₁", fontsize=9)
    ax.set_ylabel("PC2" if pca_used else "Z₂", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6, loc="best", framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, stem)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Parse ──────────────────────────────────────────────────────
    node_ids, X_full, label_idx, label_set, edges, degree = parse_cora()
    n_classes = len(label_set)

    # ── Build balanced_degree subset ───────────────────────────────
    subset_idx, counts = build_balanced_degree(label_idx, n_classes, PER_CLASS, degree)
    st = build_subset(subset_idx, X_full, label_idx, edges, label_set, D_SUBSET)

    X_sub     = st["X_sub"]
    Y_sub     = st["Y_sub"]
    label_sub = st["label_sub"]
    n_sub     = st["n_sub"]
    n_cl      = len(np.unique(label_sub))

    print(f"\nbalanced_degree subset: n={n_sub}, edges={st['n_edges']}, "
          f"density={st['density']:.5f}, isolated={st['n_isolated']}, "
          f"LCC={st['lcc_size']}")
    print(f"  class_counts: " +
          " | ".join(f"{l}:{c}" for l, c in st["class_counts"].items() if c > 0))

    # ── Data summary CSV ───────────────────────────────────────────
    ds_row = {
        "strategy":    "balanced_degree",
        "per_class":   PER_CLASS,
        "n_subset":    n_sub,
        "d_subset":    D_SUBSET,
        "n_edges":     st["n_edges"],
        "y_density":   st["density"],
        "n_isolated":  st["n_isolated"],
        "lcc_size":    st["lcc_size"],
        "n_classes":   n_classes,
        "family_x":    FAMILY_X,
        "family_y":    FAMILY_Y,
        "class_counts": str(st["class_counts"]),
    }
    out_ds = OUT_DIR / "cora_balanced_k_sweep_data_summary.csv"
    if not out_ds.exists():
        pd.DataFrame([ds_row]).to_csv(out_ds, index=False)
        print(f"Saved: {out_ds}")

    # ── k-sweep experiment ─────────────────────────────────────────
    t0 = time.perf_counter()
    total_fits = len(K_LIST) * N_TRIALS
    print(f"\n=== k sweep: k={K_LIST}, {N_TRIALS} trials = {total_fits} fits ===")

    summary_rows = []
    z_store      = {}    # {(k, trial): Z_est}
    fit_count    = 0

    for k_val in K_LIST:
        for trial in range(N_TRIALS):
            fit_count += 1
            seed = SEED_BASE + k_val * 100 + trial * 10
            print(f"\n[{fit_count:2d}/{total_fits}] k={k_val}  trial={trial}  seed={seed}")

            try:
                t_k = time.perf_counter()
                res = run_em_fixed(
                    X_sub, Y_sub, FAMILY_X, FAMILY_Y, k=k_val,
                    L=L, num_iter=NITER, seed=seed,
                )
                elapsed_k = time.perf_counter() - t_k
                sil, nmi, ari, _ = compute_z_metrics(
                    res["Z_est"], label_sub, n_clusters=n_cl)
                success = True
                err_msg = ""
                z_store[(k_val, trial)] = res["Z_est"]
                print(
                    f"  BIC={res['bic']:.1f}  Q={res['Q_strict']:.1f}"
                    f"  AUC={res['auc_y']:.4f}  AP={res['ap_y']:.4f}"
                    f"  sil={sil:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}"
                    f"  [{elapsed_k:.1f}s]  w0={res['w0']:.3f}  w={res['w']:.3f}"
                )
            except Exception as e:
                traceback.print_exc()
                res = {
                    "Q_strict": float("nan"), "bic": float("nan"),
                    "num_params": -1, "auc_y": float("nan"),
                    "ap_y": float("nan"), "rmse_x": float("nan"),
                    "w0": float("nan"), "w": float("nan"),
                    "nan_occurred": True, "nan_count": -1,
                }
                sil = nmi = ari = float("nan")
                elapsed_k = float("nan")
                success   = False
                err_msg   = str(e)
                print(f"  ERROR: {err_msg[:80]}")

            summary_rows.append({
                "k":           k_val,
                "trial":       trial,
                "seed":        seed,
                "n":           n_sub,
                "d":           D_SUBSET,
                "family_x":    FAMILY_X,
                "family_y":    FAMILY_Y,
                "strategy":    "balanced_degree",
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
                "nan_count":   res.get("nan_count", 0),
                "runtime_s":   elapsed_k,
                "success":     success,
                "error_message": err_msg,
            })

    # ── Summary CSV ────────────────────────────────────────────────
    sum_df  = pd.DataFrame(summary_rows)
    out_sum = OUT_DIR / "cora_balanced_k_sweep_summary.csv"
    if not out_sum.exists():
        sum_df.to_csv(out_sum, index=False)
        print(f"\nSaved: {out_sum}  ({len(sum_df)} rows)")

    # ── Aggregation CSV ────────────────────────────────────────────
    suc_df  = sum_df[sum_df["success"] == True].copy()
    agg_rows = []
    for k_val in K_LIST:
        kdf = suc_df[suc_df["k"] == k_val]
        if len(kdf) == 0:
            continue
        agg_rows.append({
            "k":              k_val,
            "n_trials":       len(kdf),
            "bic_mean":       kdf["bic"].mean(),
            "bic_std":        kdf["bic"].std(ddof=0),
            "bic_min":        kdf["bic"].min(),
            "q_mean":         kdf["q_strict"].mean(),
            "auc_mean":       kdf["auc_y"].mean(),
            "auc_std":        kdf["auc_y"].std(ddof=0),
            "ap_mean":        kdf["ap_y"].mean(),
            "ap_std":         kdf["ap_y"].std(ddof=0),
            "silhouette_mean": kdf["silhouette"].mean(),
            "silhouette_std":  kdf["silhouette"].std(ddof=0),
            "nmi_mean":       kdf["nmi"].mean(),
            "nmi_std":        kdf["nmi"].std(ddof=0),
            "ari_mean":       kdf["ari"].mean(),
            "ari_std":        kdf["ari"].std(ddof=0),
            "rmse_x_mean":    kdf["rmse_x"].mean(),
            "w0_mean":        kdf["w0"].mean(),
            "w_mean":         kdf["w"].mean(),
            "runtime_mean":   kdf["runtime_s"].mean(),
            "success_rate":   len(kdf) / N_TRIALS,
            "n_nan":          int(sum_df[sum_df["k"] == k_val]["nan_occurred"].sum()),
        })
    agg_df  = pd.DataFrame(agg_rows)
    out_agg = OUT_DIR / "cora_balanced_k_sweep_agg.csv"
    if not out_agg.exists():
        agg_df.to_csv(out_agg, index=False)
        print(f"Saved: {out_agg}  ({len(agg_df)} rows)")

    # ── Best-k CSV ─────────────────────────────────────────────────
    best_k_bic = int(agg_df.loc[agg_df["bic_mean"].idxmin(), "k"])
    best_k_ap  = int(agg_df.loc[agg_df["ap_mean"].idxmax(), "k"])
    best_k_auc = int(agg_df.loc[agg_df["auc_mean"].idxmax(), "k"])
    best_k_nmi = int(agg_df.loc[agg_df["nmi_mean"].idxmax(), "k"])
    best_k_ari = int(agg_df.loc[agg_df["ari_mean"].idxmax(), "k"])

    bestk_df = pd.DataFrame([{
        "best_k_by_BIC":  best_k_bic,
        "best_k_by_AP":   best_k_ap,
        "best_k_by_AUC":  best_k_auc,
        "best_k_by_NMI":  best_k_nmi,
        "best_k_by_ARI":  best_k_ari,
        "bic_at_best_k":  float(agg_df.loc[agg_df["k"] == best_k_bic, "bic_mean"].iloc[0]),
        "ap_at_best_k":   float(agg_df.loc[agg_df["k"] == best_k_ap,  "ap_mean"].iloc[0]),
        "auc_at_best_k":  float(agg_df.loc[agg_df["k"] == best_k_auc, "auc_mean"].iloc[0]),
        "nmi_at_best_k":  float(agg_df.loc[agg_df["k"] == best_k_nmi, "nmi_mean"].iloc[0]),
        "ari_at_best_k":  float(agg_df.loc[agg_df["k"] == best_k_ari, "ari_mean"].iloc[0]),
    }])
    out_bk = OUT_DIR / "cora_balanced_k_sweep_bestk.csv"
    if not out_bk.exists():
        bestk_df.to_csv(out_bk, index=False)
        print(f"Saved: {out_bk}")

    # ── Figures ────────────────────────────────────────────────────
    print(f"\n=== Figures ===")
    print(f"  best_k_BIC={best_k_bic}, best_k_AP={best_k_ap}, best_k_NMI={best_k_nmi}")

    make_bic_curve(agg_df, best_k_bic)
    make_metrics_figure(agg_df, best_k_bic, best_k_ap, best_k_nmi)

    # Z: best BIC k, best trial (lowest BIC)
    bic_k_rows = sum_df[(sum_df["k"] == best_k_bic) & (sum_df["success"] == True)]
    if len(bic_k_rows) > 0:
        best_trial_bic = int(bic_k_rows.loc[bic_k_rows["bic"].idxmin(), "trial"])
        Z_bic = z_store.get((best_k_bic, best_trial_bic))
        if Z_bic is not None:
            make_z_figure(
                Z_bic, label_sub, label_set,
                title=(f"Cora balanced_degree — Z (k={best_k_bic}, best BIC trial={best_trial_bic})\n"
                       f"n=280, d=50, X=Bern, Y=Bern"),
                stem="cora_balanced_z_best_bic",
            )

    # Z: best AP k (if different from best BIC k)
    ap_k_rows = sum_df[(sum_df["k"] == best_k_ap) & (sum_df["success"] == True)]
    if len(ap_k_rows) > 0:
        best_trial_ap = int(ap_k_rows.loc[ap_k_rows["ap_y"].idxmax(), "trial"])
        Z_ap = z_store.get((best_k_ap, best_trial_ap))
        if Z_ap is not None:
            make_z_figure(
                Z_ap, label_sub, label_set,
                title=(f"Cora balanced_degree — Z (k={best_k_ap}, best AP trial={best_trial_ap})\n"
                       f"n=280, d=50, X=Bern, Y=Bern"),
                stem="cora_balanced_z_best_ap",
            )

    # ── Final report ───────────────────────────────────────────────
    elapsed_total = (time.perf_counter() - t0) / 60
    suc_count     = int(sum_df["success"].sum())
    print(f"\nDone in {elapsed_total:.1f} min  ({suc_count}/{total_fits} fits)")

    print("\n=== Aggregated results (mean over trials) ===")
    for _, row in agg_df.iterrows():
        print(f"  k={int(row['k'])}: "
              f"BIC={row['bic_mean']:.1f}±{row['bic_std']:.1f}  "
              f"AUC={row['auc_mean']:.4f}  AP={row['ap_mean']:.4f}  "
              f"sil={row['silhouette_mean']:.4f}  "
              f"NMI={row['nmi_mean']:.4f}  ARI={row['ari_mean']:.4f}  "
              f"[{row['runtime_mean']:.0f}s]")

    print(f"\n=== Best k ===")
    print(f"  best_k_BIC={best_k_bic}  best_k_AP={best_k_ap}  "
          f"best_k_AUC={best_k_auc}  best_k_NMI={best_k_nmi}  best_k_ARI={best_k_ari}")

    nan_total = int(sum_df["nan_occurred"].sum())
    if nan_total > 0:
        print(f"\nWARNING: {nan_total} fits had NaN in E-step (fallback used)")


if __name__ == "__main__":
    main()
