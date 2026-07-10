"""
experimental モデル群のスモークテスト。

実行: python expfam/src/experimental/test_experimental_models.py
全テスト PASS を確認してから実験スクリプトを走らせること。
"""

import numpy as np
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_SRC = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent.parent / "reproduction" / "src"))

from model_dual_expfam_fixed import DualExpFamLSMFixed          # noqa
from model_dual_expfam_masked import DualExpFamLSMMasked        # noqa
from model_dual_expfam_nb import DualExpFamLSMNB                # noqa
from eval_utils import (                                        # noqa
    make_pair_split, moment_estimate_nb_r, pearson_dispersion,
    poisson_ll_pairs, nb_ll_pairs,
)
from data_generator_overdispersed import generate_dual_data_nb_y  # noqa
from em_runner import run_em_experimental, predict_mu_y           # noqa


def _toy(seed=0, n=20, d=6, k=2, family_x="bernoulli", family_y="poisson"):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, k))
    F = rng.standard_normal((d, k)) * 0.3
    X = (rng.random((n, d)) < 0.5).astype(float)
    Y = rng.poisson(3.0, size=(n, n)).astype(float)
    Y = np.triu(Y, 1)
    Y = Y + Y.T
    sigma = np.eye(d)
    return dict(n=n, d=d, k=k, Z=Z, F=F, X=X, Y=Y, sigma=sigma,
                var_z=1.0, w0=0.5, w=0.2,
                family_x=family_x, family_y=family_y)


def test_masked_full_equals_fixed():
    """train_mask=None の masked モデル == DualExpFamLSMFixed（数値一致）。"""
    t = _toy()
    for fy in ("poisson", "bernoulli", "gaussian"):
        Yv = t["Y"] if fy != "bernoulli" else (t["Y"] > 3).astype(float)
        fixed = DualExpFamLSMFixed(n=t["n"], d=t["d"], k=t["k"], L=3,
                                   family_x=t["family_x"], family_y=fy)
        masked = DualExpFamLSMMasked(n=t["n"], d=t["d"], k=t["k"], L=3,
                                     family_x=t["family_x"], family_y=fy,
                                     train_mask=None)
        for m in (fixed, masked):
            m.initialize_params(seed=1)
        for i in (0, 5, 19):
            g1 = fixed._calc_gradient(t["X"], Yv, t["Z"], t["F"], t["sigma"],
                                      t["var_z"], t["w0"], t["w"], i)
            g2 = masked._calc_gradient(t["X"], Yv, t["Z"], t["F"], t["sigma"],
                                       t["var_z"], t["w0"], t["w"], i)
            assert np.allclose(g1, g2, atol=1e-12), f"gradient mismatch fy={fy}"
            p1 = fixed._calc_precision_matrix(t["Z"], t["F"], t["sigma"],
                                              t["var_z"], t["w0"], t["w"], i)
            p2 = masked._calc_precision_matrix(t["Z"], t["F"], t["sigma"],
                                               t["var_z"], t["w0"], t["w"], i)
            assert np.allclose(p1, p2, atol=1e-12), f"precision mismatch fy={fy}"
        Zs = np.stack([t["Z"]] * 3, axis=2)
        w0_1 = fixed.calc_w0(Yv, Zs, t["w0"], t["w"], max_iter=5)
        w0_2 = masked.calc_w0(Yv, Zs, t["w0"], t["w"], max_iter=5)
        assert np.isclose(w0_1, w0_2, atol=1e-12), f"calc_w0 mismatch fy={fy}"
        ll1 = fixed.calc_log_likelihood_Y(Yv, Zs, t["w0"], t["w"])
        ll2 = masked.calc_log_likelihood_Y(Yv, Zs, t["w0"], t["w"])
        assert np.isclose(ll1, ll2, atol=1e-9), f"llY mismatch fy={fy}"
    print("PASS: test_masked_full_equals_fixed")


def test_masked_ignores_heldout_pairs():
    """held-out ペアの Y 値を書き換えても masked の勾配・M-step が不変。"""
    t = _toy()
    train_mask, test_mask = make_pair_split(t["n"], 0.3, seed=7)
    masked = DualExpFamLSMMasked(n=t["n"], d=t["d"], k=t["k"], L=3,
                                 family_x=t["family_x"], family_y="poisson",
                                 train_mask=train_mask)
    masked.initialize_params(seed=1)

    Y1 = t["Y"].copy()
    Y2 = t["Y"].copy()
    Y2[test_mask] = 999.0    # held-out ペアを汚染

    Zs = np.stack([t["Z"]] * 3, axis=2)
    for i in (0, 10):
        g1 = masked._calc_gradient(t["X"], Y1, t["Z"], t["F"], t["sigma"],
                                   t["var_z"], t["w0"], t["w"], i)
        g2 = masked._calc_gradient(t["X"], Y2, t["Z"], t["F"], t["sigma"],
                                   t["var_z"], t["w0"], t["w"], i)
        assert np.allclose(g1, g2, atol=1e-12), "gradient leaked held-out info"
    w0_1 = masked.calc_w0(Y1, Zs, t["w0"], t["w"], max_iter=5)
    w0_2 = masked.calc_w0(Y2, Zs, t["w0"], t["w"], max_iter=5)
    assert np.isclose(w0_1, w0_2, atol=1e-12), "calc_w0 leaked held-out info"
    w_1 = masked.calc_w(Y1, Zs, t["w0"], t["w"], max_iter=5)
    w_2 = masked.calc_w(Y2, Zs, t["w0"], t["w"], max_iter=5)
    assert np.isclose(w_1, w_2, atol=1e-12), "calc_w leaked held-out info"
    ll1 = masked.calc_log_likelihood_Y(Y1, Zs, t["w0"], t["w"])
    ll2 = masked.calc_log_likelihood_Y(Y2, Zs, t["w0"], t["w"])
    assert np.isclose(ll1, ll2, atol=1e-9), "llY leaked held-out info"
    print("PASS: test_masked_ignores_heldout_pairs")


def test_split_masks_partition():
    """train/test マスクが非対角の分割になっている。"""
    n = 30
    train_mask, test_mask = make_pair_split(n, 0.2, seed=3)
    off_diag = ~np.eye(n, dtype=bool)
    assert not (train_mask & test_mask).any(), "masks overlap"
    assert ((train_mask | test_mask) == off_diag).all(), "masks not a partition"
    assert np.array_equal(train_mask, train_mask.T)
    assert np.array_equal(test_mask, test_mask.T)
    n_test = int(np.triu(test_mask, 1).sum())
    n_total = n * (n - 1) // 2
    assert abs(n_test - 0.2 * n_total) <= 1
    print("PASS: test_split_masks_partition")


def test_nb_large_r_matches_poisson():
    """r→∞ で NB の score / curvature / ll が Poisson に一致。"""
    t = _toy()
    nb = DualExpFamLSMNB(n=t["n"], d=t["d"], k=t["k"], L=3,
                         family_x=t["family_x"], nb_r=1e9)
    pois = DualExpFamLSMMasked(n=t["n"], d=t["d"], k=t["k"], L=3,
                               family_x=t["family_x"], family_y="poisson")
    for m in (nb, pois):
        m.initialize_params(seed=1)
    eta = np.linspace(-2, 3, 50)
    mu = np.exp(eta)
    y = np.arange(50, dtype=float)
    assert np.allclose(nb._variance_function(eta),
                       pois._variance_function(eta), rtol=1e-6)
    assert np.allclose(nb._y_score_estep(y, mu),
                       pois._y_score_estep(y, mu), rtol=1e-6)
    # full-constant per-pair ll の一致（eval_utils 側）
    assert np.allclose(nb_ll_pairs(y, mu, 1e9),
                       poisson_ll_pairs(y, mu), atol=1e-4)
    print("PASS: test_nb_large_r_matches_poisson")


def test_moment_estimator_recovers_r():
    """真の μ を使ったモーメント推定が生成 r をおおむね回復する。"""
    rng = np.random.default_rng(0)
    r_true = 5.0
    mu = np.exp(rng.normal(2.0, 0.5, size=20000))
    lam = rng.gamma(shape=r_true, scale=mu / r_true)
    y = rng.poisson(lam).astype(float)
    r_hat = moment_estimate_nb_r(y, mu)
    assert 3.0 < r_hat < 8.0, f"r_hat={r_hat} far from r_true={r_true}"
    disp = pearson_dispersion(y, mu)
    assert disp > 2.0, f"dispersion={disp} should exceed 2 for r=5"
    print(f"PASS: test_moment_estimator_recovers_r (r_hat={r_hat:.2f}, "
          f"pearson_disp={disp:.2f})")


def test_nb_em_smoke():
    """NB-Y データ + NB モデルで EM が NaN なく回り、有限の推定値を返す。"""
    data = generate_dual_data_nb_y(n=40, d=8, k=2, seed=11,
                                   family_x="bernoulli",
                                   w0_true=1.2, w_true=0.3, nb_r=5.0)
    train_mask, test_mask = make_pair_split(40, 0.2, seed=1)
    res = run_em_experimental(
        data["X"], data["Y"], family_x="bernoulli", family_y="nb",
        k=2, L=3, num_iter=3, seed=5, train_mask=train_mask, nb_r=5.0)
    assert not res["nan_occurred"], "NaN occurred in NB EM"
    assert np.isfinite(res["w0"]) and np.isfinite(res["w"])
    assert np.isfinite(res["Q_strict"]), "Q_strict not finite"
    mu = predict_mu_y(res)
    assert np.all(np.isfinite(mu))
    print(f"PASS: test_nb_em_smoke (w0={res['w0']:.3f}, w={res['w']:.3f}, "
          f"BIC={res['bic']:.1f}, {res['runtime_s']}s)")


def test_masked_em_smoke_poisson():
    """Poisson + mask で EM が回る。"""
    data = generate_dual_data_nb_y(n=40, d=8, k=2, seed=12,
                                   family_x="bernoulli",
                                   w0_true=1.2, w_true=0.3, nb_r=None)
    train_mask, _ = make_pair_split(40, 0.2, seed=2)
    res = run_em_experimental(
        data["X"], data["Y"], family_x="bernoulli", family_y="poisson",
        k=2, L=3, num_iter=3, seed=6, train_mask=train_mask)
    assert not res["nan_occurred"]
    assert np.isfinite(res["Q_strict"])
    print(f"PASS: test_masked_em_smoke_poisson ({res['runtime_s']}s)")


if __name__ == "__main__":
    test_masked_full_equals_fixed()
    test_masked_ignores_heldout_pairs()
    test_split_masks_partition()
    test_nb_large_r_matches_poisson()
    test_moment_estimator_recovers_r()
    test_nb_em_smoke()
    test_masked_em_smoke_poisson()
    print("\nALL TESTS PASSED")
