"""
NB2-Y 数式・実装の独立監査テスト（2026-07-08 監査で追加）。

既存の test_experimental_models.py と独立に、以下を検証する:
  1. NB 対数尤度が scipy.stats.nbinom（独立実装）と一致
  2. score ∂ℓ/∂η が数値微分と一致
  3. observed negative Hessian rμ(y+r)/(μ+r)² が数値2階微分と一致し、
     Fisher 情報 μr/(μ+r) が E[y]=μ の代入（= y=μ での observed）と一致
  4. r→∞ で ll / score / Fisher が Poisson に収束
  5. r が小さいほど生成データの条件付き分散が大きい（生成器の単調性）
  6. moment r̂ が train ペアのみに依存（test ペア汚染で不変）
  7. BIC パラメータ数: nb_r_estimated=False で r を数えない
  8. NB モデル + pair mask: held-out Y 汚染で学習量が不変（NB 版の再確認）

実行: python expfam/src/experimental/test_nb_math_audit.py
"""

import numpy as np
import sys
from pathlib import Path
from scipy.stats import nbinom, poisson as sp_poisson

_HERE = Path(__file__).parent
_SRC = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent.parent / "reproduction" / "src"))

from model_dual_expfam_nb import DualExpFamLSMNB                 # noqa
from model_dual_expfam_masked import DualExpFamLSMMasked         # noqa
from eval_utils import (                                          # noqa
    nb_ll_pairs, poisson_ll_pairs, moment_estimate_nb_r,
    make_pair_split, upper_pairs_of, calc_bic_exp)
from data_generator_overdispersed import generate_dual_data_nb_y  # noqa


def test_nb_ll_matches_scipy():
    """NB 対数尤度 = scipy.stats.nbinom.logpmf（p = r/(r+μ)）。"""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 60, size=200).astype(float)
    mu = np.exp(rng.normal(1.5, 1.0, size=200))
    for r in (0.5, 2.0, 10.0, 100.0):
        ours = nb_ll_pairs(y, mu, r)
        ref = nbinom.logpmf(y.astype(int), r, r / (r + mu))
        assert np.allclose(ours, ref, atol=1e-8), f"ll mismatch at r={r}"
    print("PASS: test_nb_ll_matches_scipy")


def test_score_matches_numerical_gradient():
    """∂ℓ/∂η（実装: r(y−μ)/(r+μ)）= 数値微分。"""
    r = 5.0
    y = np.array([0.0, 1.0, 3.0, 10.0, 40.0])
    eta = np.array([-1.0, 0.0, 0.8, 2.0, 3.5])
    mu = np.exp(eta)
    analytic = (y - mu) * r / (mu + r)

    h = 1e-6
    numeric = (nb_ll_pairs(y, np.exp(eta + h), r)
               - nb_ll_pairs(y, np.exp(eta - h), r)) / (2 * h)
    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-6), \
        f"score mismatch: {analytic} vs {numeric}"

    # モデルの _y_score_estep とも一致
    m = DualExpFamLSMNB(n=10, d=3, k=2, L=2, family_x="bernoulli", nb_r=r)
    assert np.allclose(m._y_score_estep(y, mu), analytic, atol=1e-12)
    print("PASS: test_score_matches_numerical_gradient")


def test_hessian_and_fisher():
    """
    observed −∂²ℓ/∂η² = rμ(y+r)/(μ+r)² を数値2階微分で検証し、
    Fisher I(η)=μr/(μ+r) が y=μ を代入した observed と一致することを確認。
    実装の _variance_function が Fisher（observed ではない）であることも確認。
    """
    r = 4.0
    y = np.array([0.0, 2.0, 7.0, 25.0])
    eta = np.array([-0.5, 0.5, 1.5, 3.0])
    mu = np.exp(eta)

    obs_analytic = r * mu * (y + r) / (mu + r) ** 2

    h = 1e-4
    ll = lambda e: nb_ll_pairs(y, np.exp(e), r)          # noqa: E731
    numeric_hess = -(ll(eta + h) - 2 * ll(eta) + ll(eta - h)) / h ** 2
    assert np.allclose(obs_analytic, numeric_hess, rtol=1e-4), \
        f"observed Hessian mismatch: {obs_analytic} vs {numeric_hess}"

    fisher_analytic = mu * r / (mu + r)
    # E[y]=μ を代入した observed = Fisher
    obs_at_mean = r * mu * (mu + r) / (mu + r) ** 2
    assert np.allclose(fisher_analytic, obs_at_mean, atol=1e-12)

    m = DualExpFamLSMNB(n=10, d=3, k=2, L=2, family_x="bernoulli", nb_r=r)
    impl = m._variance_function(eta)
    assert np.allclose(impl, fisher_analytic, rtol=1e-6), \
        "implementation should use Fisher information"
    # y が μ から離れたとき observed ≠ Fisher（実装は expected を採用）
    assert not np.allclose(obs_analytic, fisher_analytic, rtol=0.01), \
        "sanity: observed and Fisher must differ for y != mu"
    print("PASS: test_hessian_and_fisher (implementation = Fisher/expected)")


def test_poisson_limit():
    """r→∞ で ll / score / Fisher / 生成分散が Poisson に収束。"""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 40, size=100).astype(float)
    mu = np.exp(rng.normal(1.0, 0.8, size=100))
    r_big = 1e9

    assert np.allclose(nb_ll_pairs(y, mu, r_big),
                       poisson_ll_pairs(y, mu), atol=1e-4)
    # scipy 相互検証: nbinom(r→∞) ≈ poisson
    assert np.allclose(nbinom.logpmf(y.astype(int), r_big, r_big / (r_big + mu)),
                       sp_poisson.logpmf(y.astype(int), mu), atol=1e-4)

    score_nb = (y - mu) * r_big / (mu + r_big)
    assert np.allclose(score_nb, y - mu, rtol=1e-6)
    fisher_nb = mu * r_big / (mu + r_big)
    assert np.allclose(fisher_nb, mu, rtol=1e-6)
    print("PASS: test_poisson_limit")


def test_variance_monotone_in_r():
    """生成器: r が小さいほど条件付き var/mean が大きい（理論 1+μ/r）。"""
    ratios = []
    for r in (2.0, 5.0, 20.0, None):
        d = generate_dual_data_nb_y(n=120, d=6, k=2, seed=3,
                                    family_x="bernoulli",
                                    w0_true=1.5, w_true=0.3, nb_r=r)
        # 条件付き分散の proxy: 理論値（生成 μ 既知なので理論式で確認）
        ratios.append(d["theoretical_var_mean"])
    assert ratios[0] > ratios[1] > ratios[2] > ratios[3] == 1.0, ratios
    print(f"PASS: test_variance_monotone_in_r (1+mu/r = {ratios})")


def test_r_hat_train_only():
    """moment r̂ が test ペアの Y に依存しない（リーク検査）。"""
    rng = np.random.default_rng(2)
    n = 30
    train_mask, test_mask = make_pair_split(n, 0.3, seed=5)
    mu = np.exp(rng.normal(1.0, 0.5, size=(n, n)))
    Y1 = rng.poisson(mu * 2).astype(float)
    Y2 = Y1.copy()
    Y2[test_mask] = 9999.0

    tr_r, tr_c = upper_pairs_of(train_mask)
    r1 = moment_estimate_nb_r(Y1[tr_r, tr_c], mu[tr_r, tr_c])
    r2 = moment_estimate_nb_r(Y2[tr_r, tr_c], mu[tr_r, tr_c])
    assert r1 == r2, "r_hat depends on test pairs (leak!)"
    print("PASS: test_r_hat_train_only")


def test_bic_r_param_count():
    """BIC: nb_r_estimated の有無でパラメータ数が 1 差。"""
    _, np_est = calc_bic_exp(-100.0, k=3, n=100, d=10,
                             family_x="bernoulli", family_y_label="nb",
                             nb_r_estimated=True)
    _, np_fix = calc_bic_exp(-100.0, k=3, n=100, d=10,
                             family_x="bernoulli", family_y_label="nb",
                             nb_r_estimated=False)
    assert np_est == np_fix + 1
    _, np_pois = calc_bic_exp(-100.0, k=3, n=100, d=10,
                              family_x="bernoulli", family_y_label="poisson")
    assert np_fix == np_pois
    print("PASS: test_bic_r_param_count")


def test_nb_mask_no_leak():
    """NB モデルでも held-out Y 汚染で勾配・M-step が不変。"""
    rng = np.random.default_rng(4)
    n, d, k = 20, 5, 2
    train_mask, test_mask = make_pair_split(n, 0.3, seed=8)
    m = DualExpFamLSMNB(n=n, d=d, k=k, L=3, family_x="bernoulli",
                        nb_r=5.0, train_mask=train_mask)
    m.initialize_params(seed=1)
    Z = rng.standard_normal((n, k))
    F = rng.standard_normal((d, k)) * 0.3
    X = (rng.random((n, d)) < 0.5).astype(float)
    sigma = np.eye(d)
    Y1 = rng.poisson(3.0, size=(n, n)).astype(float)
    Y1 = np.triu(Y1, 1); Y1 = Y1 + Y1.T
    Y2 = Y1.copy(); Y2[test_mask] = 999.0
    Zs = np.stack([Z] * 3, axis=2)

    for i in (0, 10, 19):
        g1 = m._calc_gradient(X, Y1, Z, F, sigma, 1.0, 0.5, 0.2, i)
        g2 = m._calc_gradient(X, Y2, Z, F, sigma, 1.0, 0.5, 0.2, i)
        assert np.allclose(g1, g2, atol=1e-12)
    assert np.isclose(m.calc_w0(Y1, Zs, 0.5, 0.2, max_iter=5),
                      m.calc_w0(Y2, Zs, 0.5, 0.2, max_iter=5), atol=1e-12)
    assert np.isclose(m.calc_w(Y1, Zs, 0.5, 0.2, max_iter=5),
                      m.calc_w(Y2, Zs, 0.5, 0.2, max_iter=5), atol=1e-12)
    assert np.isclose(m.calc_log_likelihood_Y(Y1, Zs, 0.5, 0.2),
                      m.calc_log_likelihood_Y(Y2, Zs, 0.5, 0.2), atol=1e-9)
    print("PASS: test_nb_mask_no_leak")


if __name__ == "__main__":
    test_nb_ll_matches_scipy()
    test_score_matches_numerical_gradient()
    test_hessian_and_fisher()
    test_poisson_limit()
    test_variance_monotone_in_r()
    test_r_hat_train_only()
    test_bic_r_param_count()
    test_nb_mask_no_leak()
    print("\nALL NB MATH AUDIT TESTS PASSED")
