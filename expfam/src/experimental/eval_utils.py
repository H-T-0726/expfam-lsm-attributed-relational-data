"""
eval_utils — 過分散・strict held-out 実験用の評価ユーティリティ。

含まれるもの:
    make_pair_split          : 上三角ペアの train/test 分割（対称 bool マスク）
    poisson_ll_pairs / nb_ll_pairs / gaussian_ll_pairs : 全定数込み対数尤度
    heldout_count_metrics    : held-out カウント予測の総合評価
    pearson_dispersion       : Pearson 残差過分散統計量
    moment_estimate_nb_r     : NB dispersion r のモーメント推定
    calc_Q_dual_strict_exp   : NB / mask 対応の厳密 Q
    calc_bic_exp             : NB dispersion を数える BIC

注: held-out 対数尤度は最終推定値 (Z_est, w0, w) による plug-in 近似であり、
    事後分布で積分した厳密な予測分布ではない（全 family 同一条件の比較なので
    相対比較としては公平）。
"""

import numpy as np
from scipy.special import gammaln
from scipy.stats import pearsonr, spearmanr


# ──────────────────────────────────────────────────────────────────────
# Pair split
# ──────────────────────────────────────────────────────────────────────

def make_pair_split(n: int, test_ratio: float, seed: int):
    """
    上三角ペア (i<j) をランダムに train/test 分割する。

    Returns
    -------
    train_mask, test_mask : (n, n) bool（対称、対角 False）
        train_mask ∨ test_mask = 全非対角、train ∧ test = ∅
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.triu_indices(n, k=1)
    n_total = len(rows)
    n_test = max(1, int(round(n_total * test_ratio)))
    perm = rng.permutation(n_total)
    test_idx = perm[:n_test]

    test_mask = np.zeros((n, n), dtype=bool)
    test_mask[rows[test_idx], cols[test_idx]] = True
    test_mask |= test_mask.T

    train_mask = ~test_mask
    np.fill_diagonal(train_mask, False)
    np.fill_diagonal(test_mask, False)
    return train_mask, test_mask


def upper_pairs_of(mask: np.ndarray):
    """マスクの上三角 True 位置 (rows, cols) を返す。"""
    upper = np.triu(mask, k=1)
    return np.where(upper)


# ──────────────────────────────────────────────────────────────────────
# Full-constant per-pair log-likelihoods（family 間比較用）
# ──────────────────────────────────────────────────────────────────────

def poisson_ll_pairs(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    mu = np.clip(mu, 1e-10, None)
    return y * np.log(mu) - mu - gammaln(y + 1)


def nb_ll_pairs(y: np.ndarray, mu: np.ndarray, r: float) -> np.ndarray:
    mu = np.clip(mu, 1e-10, None)
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
            + y * (np.log(mu) - np.log(mu + r))
            + r * (np.log(r) - np.log(mu + r)))


def gaussian_ll_pairs(y: np.ndarray, mu: np.ndarray, sigma_y: float) -> np.ndarray:
    sig2 = max(float(sigma_y) ** 2, 1e-8)
    return -0.5 * (y - mu) ** 2 / sig2 - 0.5 * np.log(2.0 * np.pi * sig2)


def ll_pairs(y, mu, family_y_label, nb_r=None, sigma_y=None):
    """family ラベルで分岐する full-constant 対数尤度。"""
    if family_y_label == "poisson":
        return poisson_ll_pairs(y, mu)
    if family_y_label == "nb":
        if nb_r is None:
            raise ValueError("nb_r required for family 'nb'")
        return nb_ll_pairs(y, mu, nb_r)
    if family_y_label == "gaussian":
        if sigma_y is None:
            raise ValueError("sigma_y required for family 'gaussian'")
        return gaussian_ll_pairs(y, mu, sigma_y)
    raise ValueError(f"unsupported family_y_label '{family_y_label}'")


# ──────────────────────────────────────────────────────────────────────
# Overdispersion diagnostics
# ──────────────────────────────────────────────────────────────────────

def pearson_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """
    Pearson 残差過分散統計量 (1/N) Σ (y−μ)²/μ。
    Poisson が正しければ ≈ 1、過分散なら > 1。
    """
    mu = np.clip(mu, 1e-10, None)
    return float(np.mean((y - mu) ** 2 / mu))


def moment_estimate_nb_r(y: np.ndarray, mu: np.ndarray,
                         r_min: float = 0.1, r_max: float = 1e6) -> float:
    """
    NB2 dispersion r のモーメント推定。

    NB2: E[(y−μ)²] = μ + μ²/r  →  E[((y−μ)² − μ)/μ²] = 1/r
    r̂ = 1 / mean(((y−μ)² − μ)/μ²)

    過分散が検出されない（分母 ≤ 0）場合は r_max（実質 Poisson）を返す。
    """
    mu = np.clip(mu, 1e-10, None)
    inv_r = float(np.mean(((y - mu) ** 2 - mu) / mu ** 2))
    if inv_r <= 1.0 / r_max:
        return r_max
    return float(np.clip(1.0 / inv_r, r_min, r_max))


# ──────────────────────────────────────────────────────────────────────
# Held-out evaluation
# ──────────────────────────────────────────────────────────────────────

def heldout_count_metrics(Y, mu_y, eval_mask, family_y_label,
                          nb_r=None, sigma_y=None,
                          high_count_threshold=None):
    """
    eval_mask（対称 bool）の上三角ペアでカウント予測を評価する。

    Returns dict: n_pairs, rmse, mae, pearson, spearman,
                  mean_ll（1ペアあたり plug-in 対数尤度）,
                  pearson_dispersion,
                  hc_auc / hc_ap / hc_density（threshold 指定時）
    """
    rows, cols = upper_pairs_of(eval_mask)
    y_true = Y[rows, cols].astype(float)
    y_hat = mu_y[rows, cols].astype(float)
    out = {"n_pairs": int(len(y_true))}
    if len(y_true) == 0:
        return out

    resid = y_true - y_hat
    out["rmse"] = float(np.sqrt(np.mean(resid ** 2)))
    out["mae"] = float(np.mean(np.abs(resid)))
    try:
        out["pearson"] = float(pearsonr(y_true, y_hat)[0])
    except Exception:
        out["pearson"] = float("nan")
    try:
        out["spearman"] = float(spearmanr(y_true, y_hat)[0])
    except Exception:
        out["spearman"] = float("nan")

    try:
        out["mean_ll"] = float(np.mean(
            ll_pairs(y_true, y_hat, family_y_label, nb_r=nb_r, sigma_y=sigma_y)))
    except Exception:
        out["mean_ll"] = float("nan")

    out["pearson_dispersion"] = pearson_dispersion(y_true, y_hat)

    if high_count_threshold is not None:
        from sklearn.metrics import roc_auc_score, average_precision_score
        y_bin = (y_true >= high_count_threshold).astype(int)
        out["hc_density"] = float(y_bin.mean())
        try:
            out["hc_auc"] = float(roc_auc_score(y_bin, y_hat))
        except Exception:
            out["hc_auc"] = float("nan")
        try:
            out["hc_ap"] = float(average_precision_score(y_bin, y_hat))
        except Exception:
            out["hc_ap"] = float("nan")
    return out


# ──────────────────────────────────────────────────────────────────────
# Strict Q and BIC（NB / mask 対応版）
# ──────────────────────────────────────────────────────────────────────

def calc_Q_dual_strict_exp(X, Y, Z_samples, F, sigma, var_z, w0, w, model):
    """
    NB / mask 対応の厳密 Q（全正規化定数込み）。

    utils_expfam.calc_Q_dual_strict と異なり:
      - model が DualExpFamLSMNB のとき Poisson 階乗補正を適用しない
        （NB の calc_log_likelihood_Y が全定数込みのため）
      - Poisson-Y の −Σln(y!)、Gaussian-Y の −(1/2)ln(2π) 補正は
        観測（train_mask）上三角ペアのみに適用
    """
    n, k, L = Z_samples.shape
    label = getattr(model, "family_y_label", model.family)
    mask = getattr(model, "train_mask", None)
    if mask is None:
        obs_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    else:
        obs_upper = np.triu(mask, k=1)

    Q = 0.0
    for l in range(L):
        Z_l = Z_samples[:, :, l]
        lnpZ = float(
            -(n * k / 2.0) * np.log(2.0 * np.pi * var_z)
            - (1.0 / (2.0 * var_z)) * np.sum(Z_l ** 2))
        lnpX = model.calc_log_likelihood_X(X, Z_samples[:, :, l:l + 1], F)
        lnpY = model.calc_log_likelihood_Y(Y, Z_samples[:, :, l:l + 1], w0, w)
        Q += lnpZ + lnpX + lnpY
    Q /= L

    corr = 0.0
    # X-side strict corrections
    if model.family_x == "poisson":
        corr -= float(np.sum(gammaln(X + 1)))
    elif model.family_x == "mixed":
        # per-column モデル: Poisson 列のみ階乗補正
        pois_cols = model.columns_of("poisson")
        if len(pois_cols):
            corr -= float(np.sum(gammaln(X[:, pois_cols] + 1)))
    # Y-side strict corrections（NB は補正不要）
    if label == "poisson":
        corr -= float(np.sum(gammaln(Y[obs_upper] + 1)))
    elif label == "gaussian":
        corr -= 0.5 * np.log(2.0 * np.pi) * float(obs_upper.sum())
    return Q + corr


def calc_bic_exp(Q_strict, k, n, d, family_x, family_y_label,
                 nb_r_estimated=True, n_gaussian_x_cols=None):
    """
    BIC = −2 Q_strict + num_params ln(n)

    num_params:
        F: k*d − k(k−1)/2（回転制約）
        Sigma: d（Gaussian X）/ Gaussian 列数（family_x='mixed' のとき
               n_gaussian_x_cols を指定）
        sigma_y: 1（Gaussian Y のみ）
        nb_r: 1（NB Y かつ r をデータから推定した場合）
        w0, w: NOLTA 2024 規約に合わせ暗黙扱い（数えない; KI-010 未検証のまま踏襲）
    """
    f_params = k * d - k * (k - 1) // 2
    num_params = f_params
    if family_x == "gaussian":
        num_params += d
    elif family_x == "mixed":
        num_params += int(n_gaussian_x_cols or 0)
    if family_y_label == "gaussian":
        num_params += 1
    if family_y_label == "nb" and nb_r_estimated:
        num_params += 1
    bic = -2.0 * Q_strict + num_params * np.log(n)
    return bic, num_params
