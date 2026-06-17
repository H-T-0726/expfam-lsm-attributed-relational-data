"""
Cora balanced_degree subset — held-out link prediction pilot スクリプト。

Y の正例エッジの 20% を test set として隠し、
train Y で学習後に held-out エッジを予測する。

Strategy: balanced_degree (各クラス内次数上位 40 件)
n_subset = 280 (7 classes × 40), d_subset = 50
family_x = bernoulli, family_y = bernoulli

k_values       = [3, 6]
split_trials   = [0, 1, 2]   (異なるエッジ分割)
model_trials   = [0, 1]      (異なるモデル seed)
Total          = 2 k × 3 split × 2 model = 12 fits

test_edge_ratio = 0.2
neg_ratio       = 5  (test/train neg = 5 × positive 数)

出力:
  expfam/results/real_data/cora_heldout_link_prediction/
  expfam/figures/real_data/cora_heldout_link_prediction/
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
OUT_DIR  = _ROOT / "expfam" / "results" / "real_data" / "cora_heldout_link_prediction"
FIG_DIR  = _ROOT / "expfam" / "figures" / "real_data" / "cora_heldout_link_prediction"

PER_CLASS        = 40
D_SUBSET         = 50
FAMILY_X         = "bernoulli"
FAMILY_Y         = "bernoulli"
K_LIST           = [3, 6]
SPLIT_TRIALS     = [0, 1, 2]
MODEL_TRIALS     = [0, 1]
TEST_EDGE_RATIO  = 0.2
NEG_RATIO        = 5     # negatives per positive for train/test eval
L, NITER         = 5, 8

# Seeds
SPLIT_SEED_BASE  = 7000   # + split_trial*100
TRAIN_NEG_BASE   = 7500   # + split_trial*100
TEST_NEG_BASE    = 7600   # + split_trial*100
MODEL_SEED_BASE  = 8000   # + k*100 + split_trial*10 + model_trial

RANDOM_AP_BASELINE = 1.0 / (1.0 + NEG_RATIO)   # = 1/6 ≈ 0.167

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
    "legend.fontsize": 8,
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

    degree = np.zeros(len(node_ids), dtype=int)
    for (i, j) in edges:
        degree[i] += 1
        degree[j] += 1

    print(f"Full Cora: n={len(node_ids)}, d={X_full.shape[1]}, edges={len(edges)}")
    return node_ids, X_full, label_idx, label_set, edges, degree


# ─────────────────────────────────────────────────────────────────────
# Subset builder (balanced_degree — identical to previous scripts)
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
    top_cols   = np.argsort(-X_sub_full.sum(axis=0))[:d_sub]
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

    label_sub    = label_idx[subset_idx]
    class_counts = {label_set[c]: int(np.sum(label_sub == c))
                    for c in range(len(label_set))}

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

    return {
        "X_sub": X_sub, "Y_sub": Y_sub, "label_sub": label_sub,
        "n_sub": n_sub, "n_edges": n_edges, "density": density,
        "n_isolated": n_isolated, "lcc_size": lcc_size,
        "class_counts": class_counts,
    }


# ─────────────────────────────────────────────────────────────────────
# Train/test split
# ─────────────────────────────────────────────────────────────────────

def split_positive_edges(Y_sub, test_ratio, seed):
    """
    Y_sub の正例エッジ (upper triangle) を train/test に分割する。
    Y_train にはtest エッジが 0 として隠される。
    """
    rng = np.random.default_rng(seed)
    n   = len(Y_sub)
    pos_i, pos_j = np.where(np.triu(Y_sub > 0.5, k=1))
    pos_pairs = list(zip(pos_i.tolist(), pos_j.tolist()))

    n_test  = max(1, int(len(pos_pairs) * test_ratio))
    perm    = rng.permutation(len(pos_pairs))
    test_pairs  = [pos_pairs[perm[k]] for k in range(n_test)]
    train_pairs = [pos_pairs[perm[k]] for k in range(n_test, len(pos_pairs))]

    Y_train = Y_sub.copy()
    for i, j in test_pairs:
        Y_train[i, j] = 0.0
        Y_train[j, i] = 0.0

    return Y_train, train_pairs, test_pairs


def sample_negatives(Y_sub, n_neg, seed):
    """
    Y_sub == 0 (upper triangle) からランダムに n_neg ペアをサンプリングする。
    正例が含まれないことは Y_sub == 0 の定義から保証される。
    """
    rng = np.random.default_rng(seed)
    neg_i, neg_j = np.where(np.triu(Y_sub < 0.5, k=1))
    all_neg = list(zip(neg_i.tolist(), neg_j.tolist()))
    n_neg   = min(n_neg, len(all_neg))
    perm    = rng.permutation(len(all_neg))
    return [all_neg[perm[k]] for k in range(n_neg)]


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

        Z = model.params["Z"].copy()
        F = model.params["F"].copy()
        sigma = model.params["sigma"].copy()
        w0 = model.params["w0"]
        w  = model.params["w"]
        var_z = model.params["var_z"]
        Z_prev, nan_count = Z.copy(), 0

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
            Z = Z_samples[:, :, -1].copy()
            F     = model.calc_F(X, Z_samples)
            sigma = model.calc_sigma(X, Z_samples, F)
            w0    = model.calc_w0(Y, Z_samples, w0, w, max_iter=50)
            w     = model.calc_w(Y, Z_samples, w0, w, max_iter=50)
        if nan_count == 0:
            break

    model.params.update({"F": F, "sigma": sigma, "w0": w0, "w": w})
    Q_strict  = calc_Q_dual_strict(X, Y, Z_samples, F, sigma, var_z, w0, w, model)
    bic, npar = calc_bic_dual(Q_strict, k, n, d, family_x, family_y)

    Z_est = Z_samples[:, :, -1]
    return {
        "Q_strict": Q_strict, "bic": bic, "num_params": npar,
        "w0": float(w0), "w": float(w), "Z_est": Z_est,
        "nan_occurred": nan_count > 0, "nan_count": nan_count,
        "model": model,
    }


# ─────────────────────────────────────────────────────────────────────
# Link prediction evaluation
# ─────────────────────────────────────────────────────────────────────

def eval_link_pred(Z_est, w0, w, model, pos_pairs, neg_pairs):
    """AUC, AP を計算する。"""
    eta_y = float(w0) + float(w) * (Z_est @ Z_est.T)
    mu_y  = model._mean_function(eta_y)

    scores, labels = [], []
    for i, j in pos_pairs:
        scores.append(float(mu_y[i, j]))
        labels.append(1)
    for i, j in neg_pairs:
        scores.append(float(mu_y[i, j]))
        labels.append(0)

    scores = np.array(scores)
    labels = np.array(labels)

    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    try:
        auc = float(roc_auc_score(labels, scores))
        ap  = float(average_precision_score(labels, scores))
    except Exception:
        auc = ap = float("nan")
    return auc, ap


# ─────────────────────────────────────────────────────────────────────
# Z metrics
# ─────────────────────────────────────────────────────────────────────

def compute_z_metrics(Z_est, labels, n_clusters=None):
    k_z  = Z_est.shape[1]
    n_cl = n_clusters or len(np.unique(labels))
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


def make_metrics_figure(agg_df, random_ap_baseline):
    """
    k=3 vs k=6 の train/test AUC・AP を 1×4 パネルで比較。
    """
    metrics = [
        ("train_auc_mean", "train_auc_std", "Train AUC", None),
        ("test_auc_mean",  "test_auc_std",  "Test AUC (held-out)", None),
        ("train_ap_mean",  "train_ap_std",  "Train AP", None),
        ("test_ap_mean",   "test_ap_std",   "Test AP (held-out)", random_ap_baseline),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    colors = ["#2196F3", "#FF5722"]
    k_list = agg_df["k"].tolist()

    for ax, (mu_col, sd_col, label, baseline) in zip(axes, metrics):
        mus = agg_df[mu_col].values
        sds = agg_df[sd_col].values if sd_col in agg_df.columns else np.zeros_like(mus)
        x   = np.arange(len(k_list))
        bars = ax.bar(x, mus, yerr=sds, color=colors[:len(k_list)],
                      alpha=0.8, capsize=5, edgecolor="white")
        if baseline is not None:
            ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.4,
                       label=f"random ({baseline:.3f})")
            ax.legend(fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in k_list])
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, min(1.0, max(mus) * 1.25 + 0.05))
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        for bar, mu, sd in zip(bars, mus, sds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ax.get_ylim()[1] * 0.01,
                    f"{mu:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Cora balanced_degree — held-out link prediction\n"
        f"(n=280, d=50, test_ratio=0.2, neg_ratio={NEG_RATIO}, "
        f"{len(SPLIT_TRIALS)} splits × {len(MODEL_TRIALS)} models)",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, "cora_heldout_metrics")


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
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                   c=colors[c], label=label_set[c].replace("_", "\n"),
                   s=28, alpha=0.78, edgecolors="none")
    ax.set_xlabel("PC1" if pca_used else "Z₁", fontsize=9)
    ax.set_ylabel("PC2" if pca_used else "Z₂", fontsize=9)
    ax.set_title(title, fontsize=9)
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

    # ── Build balanced_degree subset (固定) ─────────────────────────
    subset_idx, counts = build_balanced_degree(label_idx, n_classes, PER_CLASS, degree)
    st = build_subset(subset_idx, X_full, label_idx, edges, label_set, D_SUBSET)
    X_sub     = st["X_sub"]
    Y_sub     = st["Y_sub"]
    label_sub = st["label_sub"]
    n_sub     = st["n_sub"]
    n_cl      = len(np.unique(label_sub))

    orig_edges = st["n_edges"]
    n_test_pos = max(1, int(orig_edges * TEST_EDGE_RATIO))
    n_train_pos = orig_edges - n_test_pos

    print(f"\nbalanced_degree subset: n={n_sub}, edges={orig_edges}, density={st['density']:.5f}")
    print(f"  test_ratio={TEST_EDGE_RATIO}: ~{n_test_pos} test pos, ~{n_train_pos} train pos")
    print(f"  neg_ratio={NEG_RATIO}: ~{n_test_pos*NEG_RATIO} test neg per split")
    print(f"  random_AP_baseline = {RANDOM_AP_BASELINE:.4f}")

    # ── Data summary CSV ───────────────────────────────────────────
    ds_row = {
        "strategy": "balanced_degree", "per_class": PER_CLASS,
        "n_subset": n_sub, "d_subset": D_SUBSET,
        "original_edges": orig_edges, "test_edge_ratio": TEST_EDGE_RATIO,
        "approx_train_pos": n_train_pos, "approx_test_pos": n_test_pos,
        "neg_ratio": NEG_RATIO, "random_ap_baseline": RANDOM_AP_BASELINE,
        "y_density_original": st["density"], "n_isolated": st["n_isolated"],
        "lcc_size": st["lcc_size"], "n_classes": n_classes,
        "family_x": FAMILY_X, "family_y": FAMILY_Y,
        "class_counts": str(st["class_counts"]),
    }
    out_ds = OUT_DIR / "cora_heldout_data_summary.csv"
    if not out_ds.exists():
        pd.DataFrame([ds_row]).to_csv(out_ds, index=False)
        print(f"\nSaved: {out_ds}")

    # ── Experiments ─────────────────────────────────────────────────
    t0         = time.perf_counter()
    total_fits = len(K_LIST) * len(SPLIT_TRIALS) * len(MODEL_TRIALS)
    print(f"\n=== held-out LP: {total_fits} fits "
          f"(k={K_LIST}, splits={SPLIT_TRIALS}, models={MODEL_TRIALS}) ===")

    summary_rows = []
    z_store      = {}   # {(k_val, split_trial, model_trial): (Z_est, label_sub)}
    fit_count    = 0

    for split_trial in SPLIT_TRIALS:
        split_seed     = SPLIT_SEED_BASE + split_trial * 100
        train_neg_seed = TRAIN_NEG_BASE  + split_trial * 100
        test_neg_seed  = TEST_NEG_BASE   + split_trial * 100

        Y_train, train_pairs, test_pairs = split_positive_edges(
            Y_sub, TEST_EDGE_RATIO, split_seed)

        # sample negatives from Y_sub=0 (true negatives)
        n_train_neg  = len(train_pairs) * NEG_RATIO
        n_test_neg   = len(test_pairs)  * NEG_RATIO
        train_neg    = sample_negatives(Y_sub, n_train_neg, train_neg_seed)
        test_neg     = sample_negatives(Y_sub, n_test_neg,  test_neg_seed)
        random_ap    = len(test_pairs) / (len(test_pairs) + len(test_neg)) if test_neg else RANDOM_AP_BASELINE

        print(f"\n--- split_trial={split_trial}  "
              f"train_pos={len(train_pairs)}, test_pos={len(test_pairs)}, "
              f"test_neg={len(test_neg)}, random_AP={random_ap:.4f} ---")

        for k_val in K_LIST:
            for model_trial in MODEL_TRIALS:
                fit_count += 1
                model_seed = MODEL_SEED_BASE + k_val * 100 + split_trial * 10 + model_trial
                print(f"[{fit_count:2d}/{total_fits}] k={k_val}  "
                      f"split={split_trial}  model={model_trial}  seed={model_seed}")

                try:
                    t_k = time.perf_counter()
                    res = run_em_fixed(
                        X_sub, Y_train, FAMILY_X, FAMILY_Y, k=k_val,
                        L=L, num_iter=NITER, seed=model_seed,
                    )
                    elapsed_k = time.perf_counter() - t_k

                    Z_est  = res["Z_est"]
                    w0, w  = res["w0"], res["w"]
                    model  = res["model"]

                    train_auc, train_ap = eval_link_pred(Z_est, w0, w, model, train_pairs, train_neg)
                    test_auc,  test_ap  = eval_link_pred(Z_est, w0, w, model, test_pairs,  test_neg)

                    sil, nmi, ari, _ = compute_z_metrics(Z_est, label_sub, n_clusters=n_cl)
                    success  = True
                    err_msg  = ""
                    z_store[(k_val, split_trial, model_trial)] = (Z_est, label_sub)

                    print(f"  BIC={res['bic']:.1f}  "
                          f"train(AUC={train_auc:.4f}, AP={train_ap:.4f})  "
                          f"test(AUC={test_auc:.4f}, AP={test_ap:.4f})  "
                          f"NMI={nmi:.4f}  [{elapsed_k:.1f}s]")

                except Exception as e:
                    traceback.print_exc()
                    res = {
                        "Q_strict": float("nan"), "bic": float("nan"),
                        "num_params": -1, "w0": float("nan"), "w": float("nan"),
                        "nan_occurred": True, "nan_count": -1,
                    }
                    train_auc = train_ap = test_auc = test_ap = float("nan")
                    sil = nmi = ari = float("nan")
                    elapsed_k = float("nan")
                    success   = False
                    err_msg   = str(e)
                    print(f"  ERROR: {err_msg[:80]}")

                summary_rows.append({
                    "k":            k_val,
                    "split_trial":  split_trial,
                    "model_trial":  model_trial,
                    "model_seed":   model_seed,
                    "split_seed":   split_seed,
                    "n":            n_sub,
                    "d":            D_SUBSET,
                    "family_x":     FAMILY_X,
                    "family_y":     FAMILY_Y,
                    "strategy":     "balanced_degree",
                    "train_edges":  len(train_pairs),
                    "test_edges":   len(test_pairs),
                    "train_neg":    len(train_neg),
                    "test_neg":     len(test_neg),
                    "random_ap_baseline": random_ap,
                    "q_strict":     res["Q_strict"],
                    "bic":          res["bic"],
                    "num_params":   res["num_params"],
                    "train_auc":    train_auc,
                    "train_ap":     train_ap,
                    "test_auc":     test_auc,
                    "test_ap":      test_ap,
                    "silhouette":   sil,
                    "nmi":          nmi,
                    "ari":          ari,
                    "w0":           res["w0"],
                    "w":            res["w"],
                    "nan_occurred": res["nan_occurred"],
                    "nan_count":    res.get("nan_count", 0),
                    "runtime_s":    elapsed_k,
                    "success":      success,
                    "error_message": err_msg,
                })

    # ── Summary CSV ────────────────────────────────────────────────
    sum_df  = pd.DataFrame(summary_rows)
    out_sum = OUT_DIR / "cora_heldout_summary.csv"
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
        n_trials = len(kdf)
        agg_rows.append({
            "k":              k_val,
            "n_trials":       n_trials,
            "train_auc_mean": kdf["train_auc"].mean(),
            "train_auc_std":  kdf["train_auc"].std(ddof=0),
            "train_ap_mean":  kdf["train_ap"].mean(),
            "train_ap_std":   kdf["train_ap"].std(ddof=0),
            "test_auc_mean":  kdf["test_auc"].mean(),
            "test_auc_std":   kdf["test_auc"].std(ddof=0),
            "test_ap_mean":   kdf["test_ap"].mean(),
            "test_ap_std":    kdf["test_ap"].std(ddof=0),
            "delta_auc_mean": (kdf["train_auc"] - kdf["test_auc"]).mean(),
            "delta_ap_mean":  (kdf["train_ap"]  - kdf["test_ap"]).mean(),
            "random_ap_baseline": kdf["random_ap_baseline"].mean(),
            "silhouette_mean": kdf["silhouette"].mean(),
            "silhouette_std":  kdf["silhouette"].std(ddof=0),
            "nmi_mean":       kdf["nmi"].mean(),
            "nmi_std":        kdf["nmi"].std(ddof=0),
            "ari_mean":       kdf["ari"].mean(),
            "ari_std":        kdf["ari"].std(ddof=0),
            "bic_mean":       kdf["bic"].mean(),
            "w0_mean":        kdf["w0"].mean(),
            "w_mean":         kdf["w"].mean(),
            "runtime_mean":   kdf["runtime_s"].mean(),
            "success_rate":   len(kdf) / total_fits * len(K_LIST),
            "n_nan":          int(sum_df[sum_df["k"] == k_val]["nan_occurred"].sum()),
        })
    agg_df  = pd.DataFrame(agg_rows)
    out_agg = OUT_DIR / "cora_heldout_agg.csv"
    if not out_agg.exists():
        agg_df.to_csv(out_agg, index=False)
        print(f"Saved: {out_agg}  ({len(agg_df)} rows)")

    # ── Best-k CSV ─────────────────────────────────────────────────
    best_k_test_ap  = int(agg_df.loc[agg_df["test_ap_mean"].idxmax(),  "k"])
    best_k_test_auc = int(agg_df.loc[agg_df["test_auc_mean"].idxmax(), "k"])
    best_k_nmi      = int(agg_df.loc[agg_df["nmi_mean"].idxmax(),      "k"])
    best_k_ari      = int(agg_df.loc[agg_df["ari_mean"].idxmax(),      "k"])

    bestk_df = pd.DataFrame([{
        "best_k_by_test_AP":   best_k_test_ap,
        "best_k_by_test_AUC":  best_k_test_auc,
        "best_k_by_NMI":       best_k_nmi,
        "best_k_by_ARI":       best_k_ari,
        "test_ap_at_best_k":   float(agg_df.loc[agg_df["k"] == best_k_test_ap,  "test_ap_mean"].iloc[0]),
        "test_auc_at_best_k":  float(agg_df.loc[agg_df["k"] == best_k_test_auc, "test_auc_mean"].iloc[0]),
        "nmi_at_best_k":       float(agg_df.loc[agg_df["k"] == best_k_nmi,      "nmi_mean"].iloc[0]),
        "ari_at_best_k":       float(agg_df.loc[agg_df["k"] == best_k_ari,      "ari_mean"].iloc[0]),
        "random_ap_baseline":  RANDOM_AP_BASELINE,
    }])
    out_bk = OUT_DIR / "cora_heldout_bestk.csv"
    if not out_bk.exists():
        bestk_df.to_csv(out_bk, index=False)
        print(f"Saved: {out_bk}")

    # ── Figures ────────────────────────────────────────────────────
    print(f"\n=== Figures ===")
    make_metrics_figure(agg_df, RANDOM_AP_BASELINE)

    # Z for k=3: pick trial with best test_AP
    for k_target, stem in [(3, "cora_heldout_z_k3"), (6, "cora_heldout_z_k6")]:
        k_rows = suc_df[suc_df["k"] == k_target]
        if len(k_rows) == 0:
            continue
        best_row  = k_rows.loc[k_rows["test_ap"].idxmax()]
        best_key  = (k_target, int(best_row["split_trial"]), int(best_row["model_trial"]))
        Z_est, lb = z_store.get(best_key, (None, None))
        if Z_est is not None:
            make_z_figure(
                Z_est, lb, label_set,
                title=(f"Cora balanced_degree — Z (k={k_target}, "
                       f"split={int(best_row['split_trial'])}, "
                       f"model={int(best_row['model_trial'])})\n"
                       f"test_AP={best_row['test_ap']:.4f}  NMI={best_row['nmi']:.4f}"),
                stem=stem,
            )

    # ── Final summary ──────────────────────────────────────────────
    elapsed_total = (time.perf_counter() - t0) / 60
    suc_count     = int(sum_df["success"].sum())
    print(f"\nDone in {elapsed_total:.1f} min  ({suc_count}/{total_fits} fits)")

    print("\n=== Aggregated results ===")
    cols = ["k", "n_trials", "train_auc_mean", "train_ap_mean",
            "test_auc_mean", "test_ap_mean", "random_ap_baseline",
            "delta_ap_mean", "silhouette_mean", "nmi_mean", "ari_mean"]
    print(agg_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n=== Best k ===")
    print(f"  test_AP={best_k_test_ap}  test_AUC={best_k_test_auc}  "
          f"NMI={best_k_nmi}  ARI={best_k_ari}")
    print(f"  random_AP_baseline (neg_ratio={NEG_RATIO}) = {RANDOM_AP_BASELINE:.4f}")

    nan_total = int(sum_df["nan_occurred"].sum())
    if nan_total > 0:
        print(f"\nWARNING: {nan_total} fits had NaN in E-step (fallback used)")


if __name__ == "__main__":
    main()
