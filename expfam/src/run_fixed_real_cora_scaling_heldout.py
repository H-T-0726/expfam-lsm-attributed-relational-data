"""
Cora scaling held-out link prediction experiment.

既存の cora_balanced_k_sweep / cora_heldout_link_prediction (n=280, per_class=40)
を、nested balanced-degree subset で n=490, 700 (per_class=70, 100) まで拡張し、
n=980 (per_class=140) を stress test として実行する。

設計方針 (既存実装からの継承点):
  - parse_cora / build_balanced_degree / build_subset のロジックは
    run_fixed_real_cora_heldout_link_prediction.py と同一にする
  - citation edge は (min(i,j), max(i,j)) の set で無向化・重複除去
  - feature 列選択は n=280 baseline で1回だけ決め、全 n で固定 (canonical feature set)
  - held-out split は既存と同じ "zero-filled edge hiding" 方式:
      test 正例ペアを Y_train で 0 に置き換える (pair mask ではない)
      evaluation_mode = "zero_filled_edge_hiding_no_pair_mask"
  - sampled-negative 評価 (neg_ratio=5) は既存 cora_heldout と同じ方式
  - all-candidate 評価 (test 正例 vs 全 non-edge) を新たに追加

nested 性:
  各クラス内で「全グラフ次数降順」の順序を1回だけ計算し (np.argsort(-degree[idx_c])),
  per_class=40/70/100/140 はこの同一順序の先頭からの prefix を取る。
  これにより subset_280 ⊂ subset_490 ⊂ subset_700 ⊂ subset_980 が構成的に保証される
  (本スクリプトでも nestedness を実際に検証する)。

出力:
  expfam/results/real_data/cora_scaling_heldout/
  expfam/figures/real_data/cora_scaling_heldout/

実行:
  cd expfam/src
  python run_fixed_real_cora_scaling_heldout.py
"""

import sys
import time
import hashlib
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

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

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

DATA_DIR   = _ROOT / "expfam" / "data" / "cora"
OUT_DIR    = _ROOT / "expfam" / "results" / "real_data" / "cora_scaling_heldout"
FIG_DIR    = _ROOT / "expfam" / "figures" / "real_data" / "cora_scaling_heldout"
SUBSET_DIR = OUT_DIR / "subsets"

D_SUBSET = 50
FAMILY_X = "bernoulli"
FAMILY_Y = "bernoulli"
K_LIST   = [3, 6]

MAIN_PER_CLASS    = [40, 70, 100]     # -> n = 280, 490, 700
STRESS_PER_CLASS  = 140               # -> n = 980
ALL_PER_CLASS     = [40, 70, 100, 140]

SPLIT_TRIALS        = [0, 1, 2]
MODEL_TRIALS        = [0]
STRESS_SPLIT_TRIALS = [0]
STRESS_MODEL_TRIALS = [0]

TEST_EDGE_RATIO = 0.2
NEG_RATIO       = 5
L, NITER        = 5, 8

BASE_SEED = 20260618

EVALUATION_MODE = "zero_filled_edge_hiding_no_pair_mask"
SAMPLED_RANDOM_AP_BASELINE_NOMINAL = 1.0 / (1.0 + NEG_RATIO)   # 1/6 ~= 0.1667

FULL_CORA_N = 2708

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
# Data parsing (identical logic to existing cora_heldout script)
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

    print(f"Full Cora: n={len(node_ids)}, d={X_full.shape[1]}, "
          f"edges={len(edges)}, labels={len(label_set)}")
    return node_ids, X_full, label_idx, label_set, edges, degree


# ─────────────────────────────────────────────────────────────────────
# Nested balanced-degree subset construction
# ─────────────────────────────────────────────────────────────────────

def compute_class_orders(label_idx, n_classes, degree):
    """
    各クラスについて、全グラフ次数降順の順序を1回だけ計算する。
    既存スクリプトと同一の np.argsort(-degree[idx_c]) を使用 (決定的)。
    per_class=40/70/100/140 はこの順序の prefix を取ることで nested を保証する。
    """
    orders = {}
    for c in range(n_classes):
        idx_c = np.where(label_idx == c)[0]
        order = np.argsort(-degree[idx_c])
        orders[c] = idx_c[order]
    return orders


def build_balanced_degree_subset(orders, n_classes, per_class):
    selected = []
    rank_within_class = {}
    class_counts = {}
    for c in range(n_classes):
        ordered_idx = orders[c]
        k_c = min(per_class, len(ordered_idx))
        chosen = ordered_idx[:k_c]
        for rank, i in enumerate(chosen):
            rank_within_class[int(i)] = rank
        selected.extend(chosen.tolist())
        class_counts[c] = k_c
    return np.array(selected), rank_within_class, class_counts


def build_subset(subset_idx, X_full, label_idx, edges, label_set, d_sub,
                  fixed_feature_cols=None):
    n_sub      = len(subset_idx)
    subset_set = {v: i for i, v in enumerate(subset_idx)}

    X_sub_full = X_full[subset_idx, :]
    if fixed_feature_cols is None:
        col_sums = X_sub_full.sum(axis=0)
        top_cols = np.argsort(-col_sums)[:d_sub]
    else:
        top_cols = np.asarray(fixed_feature_cols)
    X_sub = X_sub_full[:, top_cols]

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
    lcc_size   = _lcc_size(adj_list, n_sub)

    label_sub    = label_idx[subset_idx]
    class_counts = {label_set[c]: int(np.sum(label_sub == c))
                    for c in range(len(label_set))}

    return {
        "X_sub": X_sub, "Y_sub": Y_sub, "label_sub": label_sub,
        "top_cols": top_cols, "n_sub": n_sub, "n_edges": n_edges,
        "density": density, "n_isolated": n_isolated, "lcc_size": lcc_size,
        "class_counts": class_counts, "upper_mask": upper_mask,
        "node_deg": node_deg, "average_degree": float(node_deg.mean()),
    }


def _lcc_size(adj_list, n_sub):
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
    return max(comp_sizes) if comp_sizes else 0


def graph_stats_from_Y(Y):
    n = Y.shape[0]
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    n_edges = int(Y[upper_mask].sum())
    density = float(Y[upper_mask].mean())
    adj_list = [[] for _ in range(n)]
    ii, jj = np.where(np.triu(Y > 0.5, k=1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        adj_list[i].append(j)
        adj_list[j].append(i)
    node_deg = np.array([len(a) for a in adj_list])
    n_isolated = int((node_deg == 0).sum())
    lcc_size = _lcc_size(adj_list, n)
    return {
        "n_edges": n_edges, "density": density,
        "n_isolated": n_isolated, "lcc_size": lcc_size,
    }


def hash_ids(values):
    joined = ",".join(str(v) for v in values)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────
# Train/test split & negative sampling (identical to existing cora_heldout)
# ─────────────────────────────────────────────────────────────────────

def split_positive_edges(Y_sub, test_ratio, seed):
    rng = np.random.default_rng(seed)
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
    rng = np.random.default_rng(seed)
    neg_i, neg_j = np.where(np.triu(Y_sub < 0.5, k=1))
    all_neg = list(zip(neg_i.tolist(), neg_j.tolist()))
    n_neg   = min(n_neg, len(all_neg))
    perm    = rng.permutation(len(all_neg))
    return [all_neg[perm[k]] for k in range(n_neg)]


# ─────────────────────────────────────────────────────────────────────
# EM runner (identical to existing cora_heldout_link_prediction.py)
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
# Evaluation: sampled-negative (existing protocol) + all-candidate (new)
# ─────────────────────────────────────────────────────────────────────

def eval_sampled(Z_est, w0, w, model, pos_pairs, neg_pairs):
    eta_y = float(w0) + float(w) * (Z_est @ Z_est.T)
    mu_y  = model._mean_function(eta_y)

    scores, labels = [], []
    for i, j in pos_pairs:
        scores.append(float(mu_y[i, j])); labels.append(1)
    for i, j in neg_pairs:
        scores.append(float(mu_y[i, j])); labels.append(0)

    scores = np.array(scores); labels = np.array(labels)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    try:
        auc = float(roc_auc_score(labels, scores))
        ap  = float(average_precision_score(labels, scores))
    except Exception:
        auc = ap = float("nan")
    return auc, ap


def eval_full_candidates(Z_est, w0, w, model, Y_sub, test_pairs):
    """
    候補集合 = test 正例ペア + Y_sub==0 の全 non-edge ペア (train 正例は Y_sub==1 のため自動的に除外)。
    """
    n = Y_sub.shape[0]
    eta_y = float(w0) + float(w) * (Z_est @ Z_est.T)
    mu_y  = model._mean_function(eta_y)

    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    neg_mask   = upper_mask & (Y_sub < 0.5)
    neg_i, neg_j = np.where(neg_mask)

    pos_scores = np.array([mu_y[i, j] for i, j in test_pairs]) if test_pairs else np.array([])
    neg_scores = mu_y[neg_i, neg_j]

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    try:
        auc = float(roc_auc_score(labels, scores))
        ap  = float(average_precision_score(labels, scores))
    except Exception:
        auc = ap = float("nan")

    n_pos, n_neg = len(pos_scores), len(neg_scores)
    random_baseline = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else float("nan")

    K = n_pos
    if K > 0:
        order = np.argsort(-scores)
        topk_labels = labels[order[:K]]
        precision_at_k = float(topk_labels.sum() / K)
        recall_at_k    = float(topk_labels.sum() / n_pos)
    else:
        precision_at_k = recall_at_k = float("nan")

    return {
        "test_auc_all_candidates": auc,
        "test_ap_all_candidates": ap,
        "all_candidate_random_ap_baseline": random_baseline,
        "test_precision_at_K": precision_at_k,
        "test_recall_at_K": recall_at_k,
        "n_candidates": int(n_pos + n_neg),
        "n_candidate_neg": int(n_neg),
    }


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


def get_peak_memory_mb():
    if not _HAS_PSUTIL:
        return float("nan")
    try:
        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────
# Resumability helpers
# ─────────────────────────────────────────────────────────────────────

TRIAL_SUMMARY_PATH = OUT_DIR / "cora_scaling_trial_summary.csv"
KEY_COLS = ["phase", "n", "k", "split_trial", "model_trial"]


def load_completed_keys():
    if not TRIAL_SUMMARY_PATH.exists():
        return set()
    df = pd.read_csv(TRIAL_SUMMARY_PATH)
    completed = set()
    for _, row in df.iterrows():
        if bool(row.get("success", False)):
            key = tuple(row[c] for c in KEY_COLS)
            completed.add(key)
    return completed


def append_trial_row(row):
    df_row = pd.DataFrame([row])
    if TRIAL_SUMMARY_PATH.exists():
        df_row.to_csv(TRIAL_SUMMARY_PATH, mode="a", header=False, index=False)
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df_row.to_csv(TRIAL_SUMMARY_PATH, mode="w", header=True, index=False)


# ─────────────────────────────────────────────────────────────────────
# Single fit (one phase/n/k/split/model combination)
# ─────────────────────────────────────────────────────────────────────

def run_one_fit(phase, n_val, per_class, k_val, split_trial, model_trial,
                 X_sub, Y_sub, label_sub, n_cl, train_pairs, test_pairs,
                 train_neg, test_neg, model_seed, split_seed,
                 train_neg_seed, test_neg_seed, sampled_random_ap):
    n_pairs_total = n_val * (n_val - 1) // 2

    Y_train = Y_sub.copy()
    for i, j in test_pairs:
        Y_train[i, j] = 0.0
        Y_train[j, i] = 0.0

    mem_before = get_peak_memory_mb()
    t0 = time.perf_counter()
    try:
        res = run_em_fixed(X_sub, Y_train, FAMILY_X, FAMILY_Y, k=k_val,
                            L=L, num_iter=NITER, seed=model_seed)
        runtime_s = time.perf_counter() - t0
        mem_after = get_peak_memory_mb()

        Z_est, w0, w, model = res["Z_est"], res["w0"], res["w"], res["model"]

        train_auc_s, train_ap_s = eval_sampled(Z_est, w0, w, model, train_pairs, train_neg)
        test_auc_s,  test_ap_s  = eval_sampled(Z_est, w0, w, model, test_pairs,  test_neg)
        full_eval = eval_full_candidates(Z_est, w0, w, model, Y_sub, test_pairs)

        sil, nmi, ari, _ = compute_z_metrics(Z_est, label_sub, n_clusters=n_cl)

        bic, q_strict = res["bic"], res["Q_strict"]
        row = {
            "phase": phase, "n": n_val, "per_class": per_class, "k": k_val,
            "split_trial": split_trial, "model_trial": model_trial,
            "model_seed": model_seed, "split_seed": split_seed,
            "train_neg_seed": train_neg_seed, "test_neg_seed": test_neg_seed,
            "d": D_SUBSET, "family_x": FAMILY_X, "family_y": FAMILY_Y,
            "evaluation_mode": EVALUATION_MODE,
            "train_edges": len(train_pairs), "test_edges": len(test_pairs),
            "train_neg": len(train_neg), "test_neg": len(test_neg),
            "sampled_random_ap_baseline": sampled_random_ap,
            "train_auc_sampled": train_auc_s, "train_ap_sampled": train_ap_s,
            "test_auc_sampled": test_auc_s, "test_ap_sampled": test_ap_s,
            "test_ap_over_sampled_random": (
                test_ap_s / sampled_random_ap if sampled_random_ap not in (0, None)
                and not np.isnan(test_ap_s) else float("nan")
            ),
            "test_auc_all_candidates": full_eval["test_auc_all_candidates"],
            "test_ap_all_candidates": full_eval["test_ap_all_candidates"],
            "all_candidate_random_ap_baseline": full_eval["all_candidate_random_ap_baseline"],
            "test_precision_at_K": full_eval["test_precision_at_K"],
            "test_recall_at_K": full_eval["test_recall_at_K"],
            "n_candidates": full_eval["n_candidates"],
            "n_candidate_neg": full_eval["n_candidate_neg"],
            "bic": bic, "bic_per_train_pair": bic / n_pairs_total,
            "q_strict": q_strict, "q_per_train_pair": q_strict / n_pairs_total,
            "num_params": res["num_params"],
            "nmi": nmi, "ari": ari, "silhouette": sil,
            "w0": w0, "w": w,
            "nan_occurred": res["nan_occurred"], "nan_count": res["nan_count"],
            "success": True, "error_message": "",
            "runtime_seconds": runtime_s,
            "peak_memory_mb": mem_after,
            "memory_measurement_note": "rss_after_fit" if _HAS_PSUTIL else "unavailable",
        }
        Z_for_fig = Z_est
    except Exception as e:
        traceback.print_exc()
        runtime_s = time.perf_counter() - t0
        row = {
            "phase": phase, "n": n_val, "per_class": per_class, "k": k_val,
            "split_trial": split_trial, "model_trial": model_trial,
            "model_seed": model_seed, "split_seed": split_seed,
            "train_neg_seed": train_neg_seed, "test_neg_seed": test_neg_seed,
            "d": D_SUBSET, "family_x": FAMILY_X, "family_y": FAMILY_Y,
            "evaluation_mode": EVALUATION_MODE,
            "train_edges": len(train_pairs), "test_edges": len(test_pairs),
            "train_neg": len(train_neg), "test_neg": len(test_neg),
            "sampled_random_ap_baseline": sampled_random_ap,
            "train_auc_sampled": float("nan"), "train_ap_sampled": float("nan"),
            "test_auc_sampled": float("nan"), "test_ap_sampled": float("nan"),
            "test_ap_over_sampled_random": float("nan"),
            "test_auc_all_candidates": float("nan"), "test_ap_all_candidates": float("nan"),
            "all_candidate_random_ap_baseline": float("nan"),
            "test_precision_at_K": float("nan"), "test_recall_at_K": float("nan"),
            "n_candidates": -1, "n_candidate_neg": -1,
            "bic": float("nan"), "bic_per_train_pair": float("nan"),
            "q_strict": float("nan"), "q_per_train_pair": float("nan"),
            "num_params": -1,
            "nmi": float("nan"), "ari": float("nan"), "silhouette": float("nan"),
            "w0": float("nan"), "w": float("nan"),
            "nan_occurred": True, "nan_count": -1,
            "success": False, "error_message": str(e),
            "runtime_seconds": runtime_s,
            "peak_memory_mb": float("nan"),
            "memory_measurement_note": "fit_failed",
        }
        Z_for_fig = None

    return row, Z_for_fig


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


def make_network_stats_figure(subset_summary_df):
    df = subset_summary_df.sort_values("n")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    panels = [
        ("y_density", "Y density"),
        ("isolated_ratio", "Isolated node ratio"),
        ("lcc_ratio", "LCC ratio"),
        ("edge_count", "Edge count"),
    ]
    for ax, (col, label) in zip(axes, panels):
        ax.plot(df["n"], df[col], "o-", color="#2196F3", linewidth=2, markersize=7)
        for _, r in df.iterrows():
            ax.annotate(f"n={int(r['n'])}", (r["n"], r[col]),
                        textcoords="offset points", xytext=(0, 6), fontsize=7)
        ax.set_xlabel("n (subset size)")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle("Cora nested balanced-degree subsets — network statistics vs n", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "cora_scaling_network_stats_vs_n")


def make_prediction_figure(agg_df, stress_df):
    metrics = [
        ("test_ap_sampled_mean", "test_ap_sampled_std", "Test AP (sampled neg)"),
        ("test_ap_all_candidates_mean", "test_ap_all_candidates_std", "Test AP (all candidates)"),
        ("test_auc_all_candidates_mean", "test_auc_all_candidates_std", "Test AUC (all candidates)"),
        ("test_ap_over_sampled_random_mean", "test_ap_over_sampled_random_std", "Test AP / sampled random"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    colors = {3: "#2196F3", 6: "#FF5722"}

    for ax, (mu_col, sd_col, label) in zip(axes, metrics):
        for k_val in K_LIST:
            sub = agg_df[agg_df["k"] == k_val].sort_values("n")
            if len(sub) == 0:
                continue
            ax.errorbar(sub["n"], sub[mu_col], yerr=sub[sd_col],
                        fmt="o-", color=colors.get(k_val, "gray"),
                        linewidth=2, markersize=6, capsize=4, label=f"k={k_val} (main)")
            if stress_df is not None:
                s_row = stress_df[stress_df["k"] == k_val]
                if len(s_row) > 0 and mu_col in s_row.columns:
                    ax.scatter(s_row["n"], s_row[mu_col], marker="^", s=90,
                               color=colors.get(k_val, "gray"), edgecolors="black",
                               zorder=5, label=f"k={k_val} (stress n=980, single split)")
        ax.set_xlabel("n (subset size)")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=6)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(
        "Cora scaling held-out link prediction — prediction metrics vs n\n"
        "(main: n=280/490/700, mean±std over 3 splits; stress: n=980, single split, no std)",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, "cora_scaling_prediction_vs_n")


def make_runtime_figure(trial_df):
    df = trial_df[trial_df["success"] == True].copy()
    df["pair_count"] = df["n"] * (df["n"] - 1) // 2

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {3: "#2196F3", 6: "#FF5722"}

    for k_val in K_LIST:
        sub = df[df["k"] == k_val].sort_values("pair_count")
        axes[0].scatter(sub["pair_count"], sub["runtime_seconds"],
                        color=colors.get(k_val, "gray"), label=f"k={k_val}", s=40, alpha=0.8)
        if sub["peak_memory_mb"].notna().any():
            axes[1].scatter(sub["pair_count"], sub["peak_memory_mb"],
                            color=colors.get(k_val, "gray"), label=f"k={k_val}", s=40, alpha=0.8)

    axes[0].set_xlabel("pair count (n(n-1)/2)")
    axes[0].set_ylabel("runtime (seconds)")
    axes[0].set_title("Runtime vs pair count", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    if df["peak_memory_mb"].notna().any():
        axes[1].set_xlabel("pair count (n(n-1)/2)")
        axes[1].set_ylabel("peak memory (MB, RSS after fit)")
        axes[1].set_title("Memory vs pair count", fontsize=9)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, linestyle="--", alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "psutil unavailable\n(peak_memory_mb = NaN)",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=10)
        axes[1].set_title("Memory vs pair count (unavailable)", fontsize=9)

    fig.suptitle("Cora scaling — runtime / memory vs pair count (all phases)", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "cora_scaling_runtime_vs_pairs")


def make_z_n700_figure(z_dict, label_sub_700, label_set):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    n_labels = len(label_set)
    colors = CORA_COLORS[:n_labels]

    for ax, k_val in zip(axes, K_LIST):
        Z_est = z_dict.get(k_val)
        if Z_est is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"k={k_val}: N/A")
            continue
        k_z = Z_est.shape[1]
        pca_used = k_z > 2
        if k_z == 1:
            Z_2d = np.hstack([Z_est, np.zeros((len(Z_est), 1))])
        elif k_z == 2:
            Z_2d = Z_est
        else:
            Z_2d = PCA(n_components=2).fit_transform(Z_est)

        for c in range(n_labels):
            mask = label_sub_700 == c
            if mask.sum() == 0:
                continue
            ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=colors[c],
                       label=label_set[c].replace("_", "\n"), s=26, alpha=0.78,
                       edgecolors="none")
        ax.set_xlabel("PC1" if pca_used else "Z₁", fontsize=9)
        ax.set_ylabel("PC2" if pca_used else "Z₂", fontsize=9)
        ax.set_title(f"k={k_val} (best test_AP_all_candidates trial)", fontsize=9)
        ax.legend(fontsize=6, loc="best", framealpha=0.7)
        ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle("Cora scaling — Z visualization at n=700 (k=3 vs k=6)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "cora_scaling_z_n700")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    print("=" * 70)
    print("Phase 0: nested subset construction & canonical feature fixing")
    print("=" * 70)

    node_ids, X_full, label_idx, label_set, edges, degree = parse_cora()
    n_classes = len(label_set)
    orders = compute_class_orders(label_idx, n_classes, degree)

    per_class_to_n = {}
    subset_data = {}   # per_class -> dict(subset_idx, rank, counts, st)

    # n=280 baseline first -> determines canonical feature columns
    subset_idx_40, rank_40, counts_40 = build_balanced_degree_subset(orders, n_classes, 40)
    st_40 = build_subset(subset_idx_40, X_full, label_idx, edges, label_set, D_SUBSET,
                          fixed_feature_cols=None)
    canonical_feature_cols = st_40["top_cols"]
    subset_data[40] = dict(subset_idx=subset_idx_40, rank=rank_40, counts=counts_40, st=st_40)
    per_class_to_n[40] = st_40["n_sub"]

    print(f"\nn=280 baseline (per_class=40): n={st_40['n_sub']}, edges={st_40['n_edges']}, "
          f"density={st_40['density']:.6f}, isolated={st_40['n_isolated']}, lcc={st_40['lcc_size']}")

    for pc in [70, 100, 140]:
        subset_idx, rank, counts = build_balanced_degree_subset(orders, n_classes, pc)
        st = build_subset(subset_idx, X_full, label_idx, edges, label_set, D_SUBSET,
                           fixed_feature_cols=canonical_feature_cols)
        subset_data[pc] = dict(subset_idx=subset_idx, rank=rank, counts=counts, st=st)
        per_class_to_n[pc] = st["n_sub"]
        print(f"n={st['n_sub']} (per_class={pc}): edges={st['n_edges']}, "
              f"density={st['density']:.6f}, isolated={st['n_isolated']}, lcc={st['lcc_size']}")

    # ── n=280 baseline reproduction check vs existing artifacts ────────
    existing_k_sweep_path = (_ROOT / "expfam" / "results" / "real_data" /
                              "cora_balanced_k_sweep" / "cora_balanced_k_sweep_data_summary.csv")
    existing_heldout_path = (_ROOT / "expfam" / "results" / "real_data" /
                              "cora_heldout_link_prediction" / "cora_heldout_data_summary.csv")
    baseline_check_notes = []
    if existing_k_sweep_path.exists():
        ex = pd.read_csv(existing_k_sweep_path).iloc[0]
        checks = [
            ("n_subset", ex["n_subset"], st_40["n_sub"]),
            ("d_subset", ex["d_subset"], D_SUBSET),
            ("n_edges", ex["n_edges"], st_40["n_edges"]),
            ("y_density", round(float(ex["y_density"]), 6), round(st_40["density"], 6)),
            ("n_isolated", ex["n_isolated"], st_40["n_isolated"]),
            ("lcc_size", ex["lcc_size"], st_40["lcc_size"]),
        ]
        for name, old_v, new_v in checks:
            match = (old_v == new_v)
            baseline_check_notes.append(f"{name}: existing={old_v} new={new_v} match={match}")
            if not match:
                print(f"  WARNING: baseline mismatch on {name}: existing={old_v} new={new_v}")
        print("\nn=280 baseline reproduction check (vs cora_balanced_k_sweep):")
        for line in baseline_check_notes:
            print("  " + line)
    else:
        baseline_check_notes.append("existing cora_balanced_k_sweep_data_summary.csv not found")

    # ── Nestedness check ────────────────────────────────────────────
    nestedness_rows = []
    prev_set = None
    for pc in ALL_PER_CLASS:
        sd = subset_data[pc]
        cur_set = set(sd["subset_idx"].tolist())
        if prev_set is None:
            nested_flag = "n/a_baseline"
        else:
            nested_flag = bool(prev_set.issubset(cur_set))
        nestedness_rows.append({
            "n": sd["st"]["n_sub"], "per_class": pc,
            "node_id_hash": hash_ids(sorted(node_ids[sd["subset_idx"]].tolist())),
            "feature_index_hash": hash_ids(sorted(sd["st"]["top_cols"].tolist())),
            "previous_subset_included": nested_flag,
            "edge_count": sd["st"]["n_edges"],
            "density": sd["st"]["density"],
            "isolated_ratio": sd["st"]["n_isolated"] / sd["st"]["n_sub"],
            "lcc_ratio": sd["st"]["lcc_size"] / sd["st"]["n_sub"],
        })
        prev_set = cur_set
    nestedness_df = pd.DataFrame(nestedness_rows)
    out_nest = OUT_DIR / "cora_scaling_nestedness_check.csv"
    if not out_nest.exists():
        nestedness_df.to_csv(out_nest, index=False)
        print(f"\nSaved: {out_nest}")

    nestedness_ok = all(
        r["previous_subset_included"] in ("n/a_baseline", True)
        for r in nestedness_rows
    )
    print(f"\nNestedness check: {'PASS' if nestedness_ok else 'FAIL'}")
    if not nestedness_ok:
        print("ABORTING: nested subset condition not satisfied. "
              "Main scaling experiment will NOT run.")
        return

    # ── Subset summary CSV ──────────────────────────────────────────
    subset_summary_rows = []
    for pc in ALL_PER_CLASS:
        sd = subset_data[pc]
        st = sd["st"]
        n_pairs = st["n_sub"] * (st["n_sub"] - 1) // 2
        subset_summary_rows.append({
            "n": st["n_sub"], "per_class": pc, "d": D_SUBSET,
            "class_counts": str(st["class_counts"]),
            "edge_count": st["n_edges"], "pair_count": n_pairs,
            "y_density": st["density"], "average_degree": st["average_degree"],
            "isolated_nodes": st["n_isolated"],
            "isolated_ratio": st["n_isolated"] / st["n_sub"],
            "lcc_size": st["lcc_size"], "lcc_ratio": st["lcc_size"] / st["n_sub"],
            "node_id_hash": hash_ids(sorted(node_ids[sd["subset_idx"]].tolist())),
            "feature_index_hash": hash_ids(sorted(st["top_cols"].tolist())),
            "subset_is_nested_with_previous": (
                nestedness_rows[ALL_PER_CLASS.index(pc)]["previous_subset_included"]
            ),
            "preprocessing_notes": (
                "feature columns fixed from n=280 baseline (top-50 by column sum within "
                "n=280 subset); per-class degree order computed once over full graph "
                "(np.argsort(-degree), not explicitly tie-broken, matches existing n=280 script); "
                "citation edges symmetrized via (min(i,j),max(i,j)) dedup set"
            ),
        })
    subset_summary_df = pd.DataFrame(subset_summary_rows)
    out_subsum = OUT_DIR / "cora_scaling_subset_summary.csv"
    if not out_subsum.exists():
        subset_summary_df.to_csv(out_subsum, index=False)
        print(f"Saved: {out_subsum}")

    # ── Feature indices CSV ─────────────────────────────────────────
    feat_rows = []
    for rank, idx in enumerate(canonical_feature_cols.tolist()):
        feat_rows.append({
            "feature_rank": rank, "feature_index": idx,
            "feature_name_if_available": "not_available_in_cora_content",
            "selection_source": "n280_baseline_top50_by_column_sum_within_subset",
        })
    feat_df = pd.DataFrame(feat_rows)
    out_feat = OUT_DIR / "cora_scaling_feature_indices.csv"
    if not out_feat.exists():
        feat_df.to_csv(out_feat, index=False)
        print(f"Saved: {out_feat}")

    # ── Subset node ID CSVs ──────────────────────────────────────────
    for pc, n_val in per_class_to_n.items():
        sd = subset_data[pc]
        st = sd["st"]
        rows = []
        for local_i, full_i in enumerate(sd["subset_idx"].tolist()):
            rows.append({
                "paper_id": node_ids[full_i],
                "class_label": label_set[label_idx[full_i]],
                "degree_full_graph": int(degree[full_i]),
                "degree_subset_if_available": int(st["node_deg"][local_i]),
                "selection_rank_within_class": sd["rank"][full_i],
            })
        sub_df = pd.DataFrame(rows)
        out_path = SUBSET_DIR / f"cora_subset_n{n_val}.csv"
        if not out_path.exists():
            sub_df.to_csv(out_path, index=False)
            print(f"Saved: {out_path}  ({len(sub_df)} rows)")

    # ── Full Cora feasibility CSV (no fitting) ──────────────────────
    full_pair_count = FULL_CORA_N * (FULL_CORA_N - 1) // 2
    n280_pairs = 280 * 279 // 2
    n700_pairs = 700 * 699 // 2
    feas_row = {
        "full_n": FULL_CORA_N,
        "full_pair_count": full_pair_count,
        "pair_ratio_vs_n280": full_pair_count / n280_pairs,
        "pair_ratio_vs_n700": full_pair_count / n700_pairs,
        "dense_matrix_lower_bound_memory_mb_float64": FULL_CORA_N * FULL_CORA_N * 8 / (1024 * 1024),
        "notes": (
            "Lower bound is for a single dense n x n float64 matrix (e.g. Y). "
            "Actual peak usage during MC-EM is a multiple of this (Y, mu_y, eta_y, "
            "gradient/precision buffers per Newton step, L=5 MC samples kept simultaneously). "
            "Current implementation has no pair mask and no sparse representation, so "
            "all O(n^2) pair terms are computed densely every Newton iteration; "
            "full Cora was NOT executed in this script."
        ),
    }
    feas_df = pd.DataFrame([feas_row])
    out_feas = OUT_DIR / "cora_full_cora_feasibility.csv"
    if not out_feas.exists():
        feas_df.to_csv(out_feas, index=False)
        print(f"Saved: {out_feas}")

    # ── Network stats figure (Phase 0) ──────────────────────────────
    make_network_stats_figure(subset_summary_df)

    # =================================================================
    # Phase 1: main scaling experiment (n=280,490,700; k=3,6; 3 splits)
    # =================================================================
    print("\n" + "=" * 70)
    print("Phase 1: main scaling experiment")
    print("=" * 70)

    completed_keys = load_completed_keys()
    total_fits_main = len(MAIN_PER_CLASS) * len(K_LIST) * len(SPLIT_TRIALS) * len(MODEL_TRIALS)
    fit_count = 0
    z_store_700 = {}   # k -> (Z_est, best_test_ap_all_candidates)
    phase1_all_success = True

    for pc in MAIN_PER_CLASS:
        sd = subset_data[pc]
        st = sd["st"]
        X_sub, Y_sub, label_sub = st["X_sub"], st["Y_sub"], st["label_sub"]
        n_val = st["n_sub"]
        n_cl = len(np.unique(label_sub))

        for split_trial in SPLIT_TRIALS:
            split_seed     = BASE_SEED + 100000 * pc + split_trial
            train_neg_seed = split_seed + 700000
            test_neg_seed  = split_seed + 800000

            Y_train_unused, train_pairs, test_pairs = split_positive_edges(
                Y_sub, TEST_EDGE_RATIO, split_seed)
            n_train_neg = len(train_pairs) * NEG_RATIO
            n_test_neg  = len(test_pairs) * NEG_RATIO
            train_neg = sample_negatives(Y_sub, n_train_neg, train_neg_seed)
            test_neg  = sample_negatives(Y_sub, n_test_neg, test_neg_seed)
            sampled_random_ap = (
                len(test_pairs) / (len(test_pairs) + len(test_neg)) if test_neg
                else SAMPLED_RANDOM_AP_BASELINE_NOMINAL
            )

            for k_val in K_LIST:
                fit_count += 1
                model_trial = 0
                model_seed = BASE_SEED + 100000 * pc + 1000 * k_val + split_trial
                key = ("main", n_val, k_val, split_trial, model_trial)

                if key in completed_keys:
                    print(f"[{fit_count:2d}/{total_fits_main}] SKIP (already completed): "
                          f"n={n_val} k={k_val} split={split_trial}")
                    continue

                print(f"[{fit_count:2d}/{total_fits_main}] n={n_val} per_class={pc} "
                      f"k={k_val} split={split_trial} model_seed={model_seed}")

                row, Z_est = run_one_fit(
                    "main", n_val, pc, k_val, split_trial, model_trial,
                    X_sub, Y_sub, label_sub, n_cl, train_pairs, test_pairs,
                    train_neg, test_neg, model_seed, split_seed,
                    train_neg_seed, test_neg_seed, sampled_random_ap,
                )
                append_trial_row(row)

                if not row["success"]:
                    phase1_all_success = False
                    print(f"  ERROR: {row['error_message'][:100]}")
                else:
                    print(f"  BIC={row['bic']:.1f}  test_AP_sampled={row['test_ap_sampled']:.4f}  "
                          f"test_AP_all_cand={row['test_ap_all_candidates']:.4f}  "
                          f"NMI={row['nmi']:.4f}  [{row['runtime_seconds']:.1f}s]")

                if n_val == 700 and Z_est is not None:
                    prev = z_store_700.get(k_val)
                    if prev is None or row["test_ap_all_candidates"] > prev[1]:
                        z_store_700[k_val] = (Z_est, row["test_ap_all_candidates"], label_sub)

    if not phase1_all_success:
        print("\nWARNING: at least one Phase 1 fit failed. Stress test (Phase 2) will be SKIPPED.")

    # =================================================================
    # Phase 2: stress test (n=980)
    # =================================================================
    print("\n" + "=" * 70)
    print("Phase 2: stress test (n=980)")
    print("=" * 70)

    if phase1_all_success:
        pc = STRESS_PER_CLASS
        sd = subset_data[pc]
        st = sd["st"]
        X_sub, Y_sub, label_sub = st["X_sub"], st["Y_sub"], st["label_sub"]
        n_val = st["n_sub"]
        n_cl = len(np.unique(label_sub))

        for split_trial in STRESS_SPLIT_TRIALS:
            split_seed     = BASE_SEED + 100000 * pc + split_trial
            train_neg_seed = split_seed + 700000
            test_neg_seed  = split_seed + 800000

            Y_train_unused, train_pairs, test_pairs = split_positive_edges(
                Y_sub, TEST_EDGE_RATIO, split_seed)
            n_train_neg = len(train_pairs) * NEG_RATIO
            n_test_neg  = len(test_pairs) * NEG_RATIO
            train_neg = sample_negatives(Y_sub, n_train_neg, train_neg_seed)
            test_neg  = sample_negatives(Y_sub, n_test_neg, test_neg_seed)
            sampled_random_ap = (
                len(test_pairs) / (len(test_pairs) + len(test_neg)) if test_neg
                else SAMPLED_RANDOM_AP_BASELINE_NOMINAL
            )

            for k_val in K_LIST:
                model_trial = 0
                model_seed = BASE_SEED + 100000 * pc + 1000 * k_val + split_trial
                key = ("stress", n_val, k_val, split_trial, model_trial)

                if key in completed_keys:
                    print(f"SKIP (already completed): n={n_val} k={k_val} split={split_trial}")
                    continue

                print(f"n={n_val} per_class={pc} k={k_val} split={split_trial} "
                      f"model_seed={model_seed}  (STRESS TEST)")

                row, _ = run_one_fit(
                    "stress", n_val, pc, k_val, split_trial, model_trial,
                    X_sub, Y_sub, label_sub, n_cl, train_pairs, test_pairs,
                    train_neg, test_neg, model_seed, split_seed,
                    train_neg_seed, test_neg_seed, sampled_random_ap,
                )
                append_trial_row(row)

                if not row["success"]:
                    print(f"  STRESS TEST FAILURE (resource-limited): {row['error_message'][:150]}")
                else:
                    print(f"  BIC={row['bic']:.1f}  test_AP_sampled={row['test_ap_sampled']:.4f}  "
                          f"test_AP_all_cand={row['test_ap_all_candidates']:.4f}  "
                          f"[{row['runtime_seconds']:.1f}s]  mem={row['peak_memory_mb']}")
    else:
        print("Skipped (Phase 1 did not complete successfully for all fits).")

    # =================================================================
    # Aggregation
    # =================================================================
    print("\n" + "=" * 70)
    print("Aggregation")
    print("=" * 70)

    if not TRIAL_SUMMARY_PATH.exists():
        print("No trial summary found; nothing to aggregate.")
        return

    trial_df = pd.read_csv(TRIAL_SUMMARY_PATH)
    main_df  = trial_df[(trial_df["phase"] == "main") & (trial_df["success"] == True)].copy()
    stress_df_raw = trial_df[trial_df["phase"] == "stress"].copy()

    agg_rows = []
    for n_val in sorted(main_df["n"].unique()):
        for k_val in K_LIST:
            sub = main_df[(main_df["n"] == n_val) & (main_df["k"] == k_val)]
            if len(sub) == 0:
                continue
            agg_rows.append({
                "n": n_val, "k": k_val, "n_success": len(sub),
                "n_total": len(SPLIT_TRIALS),
                "test_ap_sampled_mean": sub["test_ap_sampled"].mean(),
                "test_ap_sampled_std": sub["test_ap_sampled"].std(ddof=0),
                "test_auc_sampled_mean": sub["test_auc_sampled"].mean(),
                "test_auc_sampled_std": sub["test_auc_sampled"].std(ddof=0),
                "test_ap_all_candidates_mean": sub["test_ap_all_candidates"].mean(),
                "test_ap_all_candidates_std": sub["test_ap_all_candidates"].std(ddof=0),
                "test_auc_all_candidates_mean": sub["test_auc_all_candidates"].mean(),
                "test_auc_all_candidates_std": sub["test_auc_all_candidates"].std(ddof=0),
                "test_ap_over_sampled_random_mean": sub["test_ap_over_sampled_random"].mean(),
                "test_ap_over_sampled_random_std": sub["test_ap_over_sampled_random"].std(ddof=0),
                "all_candidate_random_ap_baseline_mean": sub["all_candidate_random_ap_baseline"].mean(),
                "test_precision_at_K_mean": sub["test_precision_at_K"].mean(),
                "test_recall_at_K_mean": sub["test_recall_at_K"].mean(),
                "nmi_mean": sub["nmi"].mean(), "nmi_std": sub["nmi"].std(ddof=0),
                "ari_mean": sub["ari"].mean(), "ari_std": sub["ari"].std(ddof=0),
                "bic_per_train_pair_mean": sub["bic_per_train_pair"].mean(),
                "q_per_train_pair_mean": sub["q_per_train_pair"].mean(),
                "runtime_seconds_mean": sub["runtime_seconds"].mean(),
                "peak_memory_mb_mean": sub["peak_memory_mb"].mean(),
            })
    agg_df = pd.DataFrame(agg_rows)
    out_agg = OUT_DIR / "cora_scaling_agg.csv"
    if not out_agg.exists():
        agg_df.to_csv(out_agg, index=False)
        print(f"Saved: {out_agg}")

    best_rows = []
    for n_val in sorted(main_df["n"].unique()):
        sub = agg_df[agg_df["n"] == n_val]
        if len(sub) == 0:
            continue
        best_rows.append({
            "n": n_val,
            "best_k_by_test_AP_sampled": int(sub.loc[sub["test_ap_sampled_mean"].idxmax(), "k"]),
            "best_k_by_test_AP_all_candidates": int(sub.loc[sub["test_ap_all_candidates_mean"].idxmax(), "k"]),
            "best_k_by_test_AUC_all_candidates": int(sub.loc[sub["test_auc_all_candidates_mean"].idxmax(), "k"]),
            "best_k_by_NMI": int(sub.loc[sub["nmi_mean"].idxmax(), "k"]),
            "best_k_by_BIC": int(sub.loc[sub["bic_per_train_pair_mean"].idxmin(), "k"]),
        })
    best_df = pd.DataFrame(best_rows)
    out_best = OUT_DIR / "cora_scaling_best_by_n.csv"
    if not out_best.exists():
        best_df.to_csv(out_best, index=False)
        print(f"Saved: {out_best}")

    out_stress = OUT_DIR / "cora_scaling_stress_summary.csv"
    if not out_stress.exists() and len(stress_df_raw) > 0:
        stress_df_raw.to_csv(out_stress, index=False)
        print(f"Saved: {out_stress}")

    # =================================================================
    # Figures (prediction + runtime + Z@700)
    # =================================================================
    print("\n=== Figures (Phase 1/2 results) ===")
    stress_success = stress_df_raw[stress_df_raw["success"] == True] if len(stress_df_raw) > 0 else None
    make_prediction_figure(agg_df, stress_success if stress_success is not None and len(stress_success) > 0 else None)
    make_runtime_figure(trial_df)

    if len(z_store_700) > 0:
        z_dict_700 = {k_val: z_store_700[k_val][0] for k_val in z_store_700}
        any_k = next(iter(z_store_700))
        label_sub_700 = z_store_700[any_k][2]
        make_z_n700_figure(z_dict_700, label_sub_700, label_set)

    # =================================================================
    # Final console summary
    # =================================================================
    elapsed_total = (time.perf_counter() - t_start) / 60
    print(f"\nDone in {elapsed_total:.1f} min total.")
    print("\n=== Phase 1 aggregated (main) ===")
    if len(agg_df) > 0:
        cols = ["n", "k", "n_success", "test_ap_sampled_mean", "test_ap_all_candidates_mean",
                "test_auc_all_candidates_mean", "nmi_mean", "runtime_seconds_mean"]
        print(agg_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if len(stress_df_raw) > 0:
        print("\n=== Phase 2 stress test (n=980) ===")
        scols = ["n", "k", "success", "test_ap_sampled", "test_ap_all_candidates",
                 "runtime_seconds", "peak_memory_mb", "error_message"]
        print(stress_df_raw[scols].to_string(index=False))


if __name__ == "__main__":
    main()
