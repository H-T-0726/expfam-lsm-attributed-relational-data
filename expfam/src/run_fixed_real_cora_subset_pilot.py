"""
Cora 引用ネットワーク — fixed版 real-network subset pilot 実験スクリプト。

データ: Cora citation network (Yang et al., 2016 / LINQS)
  論文間の引用関係（自然なネットワーク Y）+ BoW 単語特徴量（X）

  X = BoW 特徴量（0/1）  → family_x=bernoulli
  Y = 引用有無（無向化）  → family_y=bernoulli

Subset 設計:
  n_subset = 300  (BFS from highest-degree node → 連結部分グラフ)
  d_subset = 50   (出現頻度上位 50 単語)

Pilot 内容:
  k = 2, 3, 5 各 1 試行（動作確認が目的）
  BIC / Y-AUC / Y-AP / silhouette / Z 2D 可視化

データ取得:
  expfam/data/cora/ に cora.content + cora.cites が存在しない場合は
  LINQS サーバーから自動ダウンロード (urllib.request, ~168KB)。
  URL: https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

出力:
  expfam/results/real_data/cora_subset_pilot/
  expfam/figures/real_data/cora_subset_pilot/

実行例:
  cd expfam/src
  python run_fixed_real_cora_subset_pilot.py
"""

import sys
import os
import io
import gzip
import tarfile
import time
import traceback
import urllib.request
import warnings
from pathlib import Path
from collections import deque

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

from model_dual_expfam_fixed import DualExpFamLSMFixed     # noqa: E402
from utils_expfam import calc_Q_dual_strict, calc_bic_dual  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────

DATA_DIR = _ROOT / "expfam" / "data" / "cora"
OUT_DIR  = _ROOT / "expfam" / "results" / "real_data" / "cora_subset_pilot"
FIG_DIR  = _ROOT / "expfam" / "figures" / "real_data" / "cora_subset_pilot"

CORA_URL    = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
N_SUBSET    = 300
D_SUBSET    = 50
FAMILY_X    = "bernoulli"
FAMILY_Y    = "bernoulli"
K_LIST      = [2, 3, 5]
SEED_BASE   = 3000
L, NITER    = 5, 8

# 7 Cora topic categories
CORA_LABELS = [
    "Case_Based", "Genetic_Algorithms", "Neural_Networks",
    "Probabilistic_Methods", "Reinforcement_Learning",
    "Rule_Learning", "Theory",
]
LABEL_COLORS = [
    "#2196F3", "#FF5722", "#4CAF50", "#9C27B0",
    "#FF9800", "#795548", "#607D8B",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 7,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────────────
# Data download + parsing
# ─────────────────────────────────────────────────────────────────────

def download_cora():
    """LINQS から cora.tgz をダウンロードして DATA_DIR に展開する。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    content_path = DATA_DIR / "cora.content"
    cites_path   = DATA_DIR / "cora.cites"

    if content_path.exists() and cites_path.exists():
        print(f"Cora data already present: {DATA_DIR}")
        return

    print(f"Downloading Cora from {CORA_URL} ...")
    try:
        with urllib.request.urlopen(CORA_URL, timeout=60) as resp:
            tgz_bytes = resp.read()
        print(f"  Downloaded {len(tgz_bytes) // 1024} KB")
    except Exception as e:
        raise RuntimeError(f"Failed to download Cora: {e}") from e

    # tgz 展開
    with tarfile.open(fileobj=io.BytesIO(tgz_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            fname = Path(member.name).name
            if fname in ("cora.content", "cora.cites"):
                member.name = fname  # フラット展開
                tar.extract(member, path=DATA_DIR)
                print(f"  Extracted: {DATA_DIR / fname}")

    assert content_path.exists(), f"cora.content not found after extraction"
    assert cites_path.exists(),   f"cora.cites not found after extraction"
    print("Cora download complete.")


def parse_cora():
    """cora.content + cora.cites を読み込み numpy 配列に変換する。"""
    content_path = DATA_DIR / "cora.content"
    cites_path   = DATA_DIR / "cora.cites"

    # node features + labels
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

    node_ids = np.array(node_ids)
    X_full   = np.array(features, dtype=np.float32)   # (2708, 1433)
    label_idx = np.array([label_set.index(l) for l in labels], dtype=int)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # edges (undirected)
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
    print(f"Full Cora: n={n_full}, d={X_full.shape[1]}, "
          f"edges={len(edges)}, labels={len(label_set)}")
    print(f"  Label categories: {label_set}")

    return node_ids, X_full, label_idx, edges, label_set


def make_subset_bfs(n_full, edges, target_n):
    """
    BFS サブセット: 最大次数ノードを起点に BFS で target_n ノードを選ぶ。
    連結性を保ちながら Y 密度を高めるため最大次数ノード起点を選択。
    """
    # adjacency list
    adj = [[] for _ in range(n_full)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    # 最大次数ノードを起点に
    degrees = [len(adj[i]) for i in range(n_full)]
    start = int(np.argmax(degrees))
    print(f"  BFS start node: {start} (degree={degrees[start]})")

    visited = [False] * n_full
    queue   = deque([start])
    visited[start] = True
    order   = [start]

    while queue and len(order) < target_n:
        node = queue.popleft()
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)
                order.append(nb)
                if len(order) >= target_n:
                    break

    subset_idx = np.array(order[:target_n])
    print(f"  BFS subset: {len(subset_idx)} nodes selected")
    return subset_idx


def build_data_subset(X_full, label_idx, edges, subset_node_idx, d_sub):
    """
    subset_node_idx に基づいて X (n_sub × d_sub) と Y (n_sub × n_sub) を構築する。
    """
    n_sub = len(subset_node_idx)
    X_sub_full = X_full[subset_node_idx, :]   # (n_sub, 1433)

    # d_subset: 列和上位 d_sub
    col_sums = X_sub_full.sum(axis=0)
    top_cols = np.argsort(-col_sums)[:d_sub]
    X_sub    = X_sub_full[:, top_cols]         # (n_sub, d_sub)

    # Y: n_sub × n_sub 隣接行列
    subset_set = {v: i for i, v in enumerate(subset_node_idx)}
    Y_sub = np.zeros((n_sub, n_sub), dtype=float)
    n_edges_sub = 0
    for (u, v) in edges:
        if u in subset_set and v in subset_set:
            i, j = subset_set[u], subset_set[v]
            Y_sub[i, j] = 1.0
            Y_sub[j, i] = 1.0
            n_edges_sub += 1
    np.fill_diagonal(Y_sub, 0.0)

    label_sub = label_idx[subset_node_idx]
    upper_mask = np.triu(np.ones((n_sub, n_sub), dtype=bool), k=1)
    density = float(Y_sub[upper_mask].mean())

    print(f"  Subset X: {X_sub.shape}, Y: {Y_sub.shape}")
    print(f"  Y edges in subset: {n_edges_sub}  density: {density:.5f}")

    return X_sub, Y_sub, label_sub, n_edges_sub, density, top_cols


# ─────────────────────────────────────────────────────────────────────
# Core EM runner (real data)
# ─────────────────────────────────────────────────────────────────────

def run_em_fixed_cora(X, Y, family_x, family_y, k,
                      L=5, num_iter=8, seed=42):
    """
    MC-EM using DualExpFamLSMFixed for real Cora subset data.
    真のZなし → Procrustes/rmse_Z 不使用。
    """
    n, d = X.shape
    max_retries = 2

    for retry in range(max_retries + 1):
        newton_alpha = 0.5 / (2 ** retry)
        rng = np.random.default_rng(seed + retry * 1000)

        model = DualExpFamLSMFixed(
            n=n, d=d, k=k, L=L,
            family_x=family_x, family_y=family_y,
        )
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
            model.sigma_y = float(max(upper_vals.std(), 0.01))

        # Bernoulli-X の場合 F スケールを小さくする
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
                model.params.update(
                    dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w)
                )
                Z_new = model.calc_eta_newton(
                    X, Y, rng=rng, max_iter=10, alpha=newton_alpha
                )
                Z_samples[:, :, l] = Z_new
                Z = Z_new.copy()

            if np.any(np.isnan(Z_samples)) or np.any(np.isinf(Z_samples)):
                nan_count += 1
                Z_samples = np.stack([Z_prev] * L, axis=2)
                Z = Z_prev.copy()

            Z_samples = model.scale_Z(Z_samples)
            Z_prev = Z.copy()
            Z = Z_samples[:, :, -1].copy()

            F     = model.calc_F(X, Z_samples)
            sigma = model.calc_sigma(X, Z_samples, F)
            w0    = model.calc_w0(Y, Z_samples, w0, w, max_iter=50)
            w     = model.calc_w(Y, Z_samples, w0, w, max_iter=50)
            if family_y == "gaussian":
                model.calc_sigma_y(Y, Z_samples, w0, w)

        nan_occurred = nan_count > 0
        if not nan_occurred:
            break

    model.params.update({"F": F, "sigma": sigma, "w0": w0, "w": w})
    Q_strict = calc_Q_dual_strict(X, Y, Z_samples, F, sigma, var_z, w0, w, model)
    bic, num_params = calc_bic_dual(
        Q_strict=Q_strict, k=k, n=n, d=d,
        family_x=family_x, family_y=family_y,
    )

    # Y 再構成指標
    Z_est  = Z_samples[:, :, -1]
    eta_y  = float(w0) + float(w) * (Z_est @ Z_est.T)
    mu_y   = model._mean_function(eta_y)
    y_true = Y[upper_mask].astype(int)
    y_score = mu_y[upper_mask]

    try:
        auc_y = float(roc_auc_score(y_true, y_score))
        ap_y  = float(average_precision_score(y_true, y_score))
    except Exception:
        auc_y = float("nan")
        ap_y  = float("nan")

    # X 再構成
    mu_x   = model._mean_function_x(Z_est @ F.T)
    rmse_x = float(np.sqrt(np.mean((X - mu_x) ** 2)))

    return {
        "Q_strict":    Q_strict,
        "bic":         bic,
        "num_params":  num_params,
        "auc_y":       auc_y,
        "ap_y":        ap_y,
        "rmse_x":      rmse_x,
        "w0":          float(w0),
        "w":           float(w),
        "Z_est":       Z_est,
        "nan_occurred": nan_occurred,
        "nan_count":   nan_count,
    }


# ─────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────

def compute_z_metrics(Z_est, labels, n_clusters=None):
    k = Z_est.shape[1]
    if k == 1:
        Z_2d = np.hstack([Z_est, np.zeros((len(Z_est), 1))])
    elif k == 2:
        Z_2d = Z_est.copy()
    else:
        Z_2d = PCA(n_components=2).fit_transform(Z_est)

    n_cl = n_clusters or len(np.unique(labels))
    try:
        sil = float(silhouette_score(Z_2d, labels))
    except Exception:
        sil = float("nan")

    try:
        km_labels = KMeans(n_clusters=n_cl, random_state=42, n_init=10).fit_predict(Z_2d)
        nmi = float(normalized_mutual_info_score(labels, km_labels))
        ari = float(adjusted_rand_score(labels, km_labels))
    except Exception:
        nmi = float("nan")
        ari = float("nan")

    return sil, nmi, ari, Z_2d


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────

def save_fig(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{stem}.{ext}"
        if path.exists():
            print(f"  SKIP (exists): {path}")
        else:
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"  Saved: {path}")
    plt.close(fig)


def make_z_by_label_figure(z_results, label_sub, label_names, n_sub, d_sub):
    """k別Z可視化 (3 panel: k=2,3,5)"""
    k_list = sorted(z_results.keys())
    ncols  = len(k_list)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.0))
    if ncols == 1:
        axes = [axes]

    n_labels = len(label_names)
    colors = LABEL_COLORS[:n_labels]

    for ax, k_val in zip(axes, k_list):
        Z_est = z_results[k_val]
        kk = Z_est.shape[1]
        pca_used = kk > 2
        if kk == 1:
            Z_2d = np.hstack([Z_est, np.zeros((len(Z_est), 1))])
        elif kk == 2:
            Z_2d = Z_est
        else:
            Z_2d = PCA(n_components=2).fit_transform(Z_est)

        for c in range(n_labels):
            mask = label_sub == c
            if mask.sum() == 0:
                continue
            short = label_names[c].replace("_", "\n")
            ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                       c=colors[c], label=short,
                       s=25, alpha=0.75, edgecolors="none")

        xlabel = "PC1" if pca_used else "Z₁"
        ylabel = "PC2" if pca_used else "Z₂"
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"k={k_val}" + (" (PCA 2D)" if pca_used else ""), fontsize=10)
        ax.legend(fontsize=6, loc="best", framealpha=0.7)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(
        f"Cora Subset Pilot: Z Visualization by Topic Label\n"
        f"(n={n_sub}, d={d_sub}, X=Bernoulli, Y=Bernoulli, 1 trial/k)",
        fontsize=11,
    )
    fig.tight_layout()
    save_fig(fig, "cora_subset_z_by_label")


def make_bic_auc_figure(k_df, n_sub, d_sub):
    """k vs BIC and AUC の 2 軸図"""
    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(k_df["k"], k_df["bic"], "o-", color="#2196F3", linewidth=1.8,
             markersize=6, label="BIC")
    ax2.plot(k_df["k"], k_df["auc_y"], "s--", color="#FF5722", linewidth=1.8,
             markersize=6, label="Y AUC")
    ax2.plot(k_df["k"], k_df["ap_y"], "^:", color="#4CAF50", linewidth=1.8,
             markersize=6, label="Y AP")

    ax1.set_xlabel("k (latent dimension)")
    ax1.set_ylabel("BIC", color="#2196F3")
    ax2.set_ylabel("AUC / AP", color="#FF5722")
    ax1.set_xticks(k_df["k"].tolist())
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax2.tick_params(axis="y", labelcolor="#FF5722")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="center right", framealpha=0.9, fontsize=8)
    ax1.set_title(
        f"Cora Subset Pilot: BIC and Y-Reconstruction vs. k\n"
        f"(n={n_sub}, d={d_sub}, X=Bernoulli, Y=Bernoulli)",
        fontsize=10,
    )
    ax1.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "cora_subset_k_bic_auc")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: データ取得 ───────────────────────────────────────────
    download_cora()
    node_ids, X_full, label_idx, edges, label_set = parse_cora()
    n_full = len(node_ids)

    # ── Step 2: BFS サブセット ───────────────────────────────────────
    print(f"\n--- BFS subset: n_subset={N_SUBSET}, d_subset={D_SUBSET} ---")
    subset_node_idx = make_subset_bfs(n_full, edges, N_SUBSET)
    X_sub, Y_sub, label_sub, n_edges_sub, density, top_cols = \
        build_data_subset(X_full, label_idx, edges, subset_node_idx, D_SUBSET)

    n_sub, d_sub = X_sub.shape
    n_classes_sub = len(np.unique(label_sub))
    label_names_sub = label_set

    print(f"  Classes in subset: {sorted(np.unique(label_sub).tolist())} "
          f"({n_classes_sub} classes)")
    print(f"  Class distribution: "
          + " | ".join(f"{label_set[c]}:{np.sum(label_sub==c)}"
                       for c in np.unique(label_sub)))

    # ── データ概要 CSV ───────────────────────────────────────────────
    data_summary = pd.DataFrame([{
        "dataset":          "Cora_subset",
        "n_full":           n_full,
        "n_subset":         n_sub,
        "d_full":           X_full.shape[1],
        "d_subset":         d_sub,
        "n_classes_full":   len(label_set),
        "n_classes_subset": n_classes_sub,
        "n_edges_subset":   n_edges_sub,
        "y_density":        density,
        "family_x":         FAMILY_X,
        "family_y":         FAMILY_Y,
        "subset_method":    "BFS from highest-degree node",
        "feature_select":   "top frequency words",
        "note": (
            "Cora citation network (LINQS). Y=citation Y_ij=1 if undirected edge. "
            "NOT a label-derived matrix (unlike Wine pilot)."
        ),
    }])
    out_dsummary = OUT_DIR / "cora_subset_data_summary.csv"
    if not out_dsummary.exists():
        data_summary.to_csv(out_dsummary, index=False)
        print(f"\nSaved: {out_dsummary}")

    # ── Step 3: Pilot 1 — k = 2, 3, 5 ─────────────────────────────
    t0 = time.perf_counter()
    print(f"\n=== Pilot 1: k={K_LIST}, 1 trial each ({len(K_LIST)} fits) ===")

    metric_rows = []
    z_results   = {}

    for k_val in K_LIST:
        seed = SEED_BASE + k_val * 10
        print(f"\n--- k={k_val} (seed={seed}) ---")
        try:
            t_k = time.perf_counter()
            res = run_em_fixed_cora(
                X_sub, Y_sub, FAMILY_X, FAMILY_Y, k=k_val,
                L=L, num_iter=NITER, seed=seed,
            )
            elapsed_k = time.perf_counter() - t_k
            sil, nmi, ari, Z_2d = compute_z_metrics(
                res["Z_est"], label_sub, n_clusters=n_classes_sub
            )
            success = True
            err_msg = ""
            print(
                f"  BIC={res['bic']:.1f}  Q={res['Q_strict']:.1f}"
                f"  AUC={res['auc_y']:.4f}  AP={res['ap_y']:.4f}"
                f"  sil={sil:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}"
                f"  [{elapsed_k:.1f}s]"
                f"  w0={res['w0']:.3f}  w={res['w']:.3f}"
            )
            z_results[k_val] = res["Z_est"]

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
            elapsed_k = time.perf_counter() - t0
            success = False
            err_msg = str(e)
            print(f"  ERROR: {err_msg[:80]}")

        metric_rows.append({
            "k":            k_val,
            "trial":        0,
            "seed":         seed,
            "n":            n_sub,
            "d":            d_sub,
            "family_x":     FAMILY_X,
            "family_y":     FAMILY_Y,
            "q_strict":     res["Q_strict"],
            "bic":          res["bic"],
            "num_params":   res["num_params"],
            "auc_y":        res["auc_y"],
            "ap_y":         res["ap_y"],
            "rmse_x":       res["rmse_x"],
            "silhouette":   sil,
            "nmi":          nmi,
            "ari":          ari,
            "w0":           res["w0"],
            "w":            res["w"],
            "nan_occurred": res["nan_occurred"],
            "runtime_s":    elapsed_k if success else float("nan"),
            "success":      success,
            "error_message": err_msg,
            "note": (
                f"cora_subset_pilot k={k_val} "
                f"(n={n_sub}, d={d_sub}, BFS subset, top-freq features)"
            ),
        })

    # ── Metrics CSV ─────────────────────────────────────────────────
    k_df = pd.DataFrame(metric_rows)
    out_kmets = OUT_DIR / "cora_subset_k_metrics.csv"
    if not out_kmets.exists():
        k_df.to_csv(out_kmets, index=False)
        print(f"\nSaved: {out_kmets}  ({len(k_df)} rows)")
    else:
        print(f"\nSKIP (exists): {out_kmets}")

    # ── Z embeddings CSV (k=3 の場合、なければ最大 AUC の k) ─────────
    best_k_for_z = 3 if 3 in z_results else (
        int(k_df.loc[k_df["auc_y"].idxmax(), "k"]) if len(k_df) > 0 else K_LIST[0]
    )
    Z_for_emb = z_results.get(best_k_for_z, list(z_results.values())[0] if z_results else None)

    z_emb_rows = []
    if Z_for_emb is not None:
        k_act = Z_for_emb.shape[1]
        pca_used = k_act > 2
        if k_act == 1:
            Z_2d_emb = np.hstack([Z_for_emb, np.zeros((n_sub, 1))])
        elif k_act == 2:
            Z_2d_emb = Z_for_emb
        else:
            Z_2d_emb = PCA(n_components=2).fit_transform(Z_for_emb)

        for i in range(n_sub):
            row = {
                "sample_id":     i,
                "orig_node_id":  str(node_ids[subset_node_idx[i]]),
                "label":         int(label_sub[i]),
                "label_name":    label_set[int(label_sub[i])],
                "z_pc1":         float(Z_2d_emb[i, 0]),
                "z_pc2":         float(Z_2d_emb[i, 1]),
                "k":             best_k_for_z,
                "pca_used":      pca_used,
                "note": f"cora_subset_pilot k={best_k_for_z} best",
            }
            for j in range(k_act):
                row[f"z_{j+1}"] = float(Z_for_emb[i, j])
            z_emb_rows.append(row)

    z_emb_df = pd.DataFrame(z_emb_rows)
    out_zemb = OUT_DIR / "cora_subset_z_embeddings.csv"
    if not out_zemb.exists() and len(z_emb_rows) > 0:
        z_emb_df.to_csv(out_zemb, index=False)
        print(f"Saved: {out_zemb}  ({len(z_emb_df)} rows)")

    # ── Step 4: 図 ─────────────────────────────────────────────────
    print("\n=== Figures ===")
    if z_results:
        make_z_by_label_figure(z_results, label_sub, label_set, n_sub, d_sub)
    make_bic_auc_figure(k_df[k_df["success"]], n_sub, d_sub)

    # ── Final summary ────────────────────────────────────────────────
    elapsed_total = (time.perf_counter() - t0) / 60
    total_fits    = len(K_LIST)
    success_fits  = int(k_df["success"].sum())
    print(f"\nDone in {elapsed_total:.1f} min  ({success_fits}/{total_fits} fits)")

    print("\n=== k別結果 ===")
    for _, row in k_df.iterrows():
        if row["success"]:
            print(f"  k={int(row['k'])}: "
                  f"BIC={row['bic']:.1f}  "
                  f"AUC={row['auc_y']:.4f}  "
                  f"AP={row['ap_y']:.4f}  "
                  f"sil={row['silhouette']:.4f}  "
                  f"NMI={row['nmi']:.4f}  "
                  f"time={row['runtime_s']:.1f}s")
        else:
            print(f"  k={int(row['k'])}: FAILED — {row['error_message'][:60]}")

    print(f"\nY density in subset: {density:.5f} "
          f"({n_edges_sub} edges / {n_sub*(n_sub-1)//2} pairs)")

    nan_fits = int(k_df["nan_occurred"].sum())
    if nan_fits > 0:
        print(f"\nWARNING: {nan_fits} fits had NaN in E-step (fallback used)")


if __name__ == "__main__":
    main()
