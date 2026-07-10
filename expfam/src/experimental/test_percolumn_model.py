"""
DualExpFamLSMPerColumn のスモークテスト。

実行: python expfam/src/experimental/test_percolumn_model.py
"""

import numpy as np
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_SRC = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent.parent / "reproduction" / "src"))

from model_dual_expfam_masked import DualExpFamLSMMasked        # noqa
from model_dual_expfam_percolumn import DualExpFamLSMPerColumn  # noqa
from em_runner import run_em_experimental                       # noqa
from utils_expfam import procrustes_rotation, calc_rmse         # noqa


def _toy(seed=0, n=20, d=6, k=2):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, k))
    F = rng.standard_normal((d, k)) * 0.3
    X = (rng.random((n, d)) < 0.5).astype(float)
    Y = rng.poisson(3.0, size=(n, n)).astype(float)
    Y = np.triu(Y, 1)
    Y = Y + Y.T
    sigma = np.eye(d)
    return dict(n=n, d=d, k=k, Z=Z, F=F, X=X, Y=Y, sigma=sigma,
                var_z=1.0, w0=0.5, w=0.2)


def test_uniform_percolumn_equals_scalar():
    """全列同一 family の per-column == スカラー family モデル。"""
    t = _toy()
    for fam in ("bernoulli", "poisson", "gaussian"):
        Xv = t["X"] if fam == "bernoulli" else (
            np.abs(t["X"] * 2 + 1) if fam == "poisson" else
            np.random.default_rng(1).standard_normal(t["X"].shape))
        scalar = DualExpFamLSMMasked(n=t["n"], d=t["d"], k=t["k"], L=3,
                                     family_x=fam, family_y="poisson")
        percol = DualExpFamLSMPerColumn(n=t["n"], d=t["d"], k=t["k"], L=3,
                                        family_x_list=[fam] * t["d"],
                                        family_y="poisson")
        for m in (scalar, percol):
            m.initialize_params(seed=1)
        for i in (0, 7, 19):
            g1 = scalar._calc_gradient(Xv, t["Y"], t["Z"], t["F"], t["sigma"],
                                       t["var_z"], t["w0"], t["w"], i)
            g2 = percol._calc_gradient(Xv, t["Y"], t["Z"], t["F"], t["sigma"],
                                       t["var_z"], t["w0"], t["w"], i)
            assert np.allclose(g1, g2, atol=1e-12), f"grad mismatch fam={fam}"
            p1 = scalar._calc_precision_matrix(t["Z"], t["F"], t["sigma"],
                                               t["var_z"], t["w0"], t["w"], i)
            p2 = percol._calc_precision_matrix(t["Z"], t["F"], t["sigma"],
                                               t["var_z"], t["w0"], t["w"], i)
            assert np.allclose(p1, p2, atol=1e-12), f"prec mismatch fam={fam}"
        Zs = np.stack([t["Z"]] * 3, axis=2)
        ll1 = scalar.calc_log_likelihood_X(Xv, Zs, t["F"])
        ll2 = percol.calc_log_likelihood_X(Xv, Zs, t["F"])
        assert np.isclose(ll1, ll2, atol=1e-6), f"llX mismatch fam={fam}"
    print("PASS: test_uniform_percolumn_equals_scalar")


def generate_mixed_x_data(n=60, d=9, k=2, seed=7,
                          w0_true=1.2, w_true=0.3):
    """混在 X（gaussian 3列 + bernoulli 3列 + poisson 3列）+ Poisson Y。"""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, k))
    Z = (Z - Z.mean(0)) / Z.std(0)
    F = rng.standard_normal((d, k))
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    fam_list = (["gaussian"] * 3 + ["bernoulli"] * 3 + ["poisson"] * 3)
    eta = Z @ F.T
    X = np.zeros((n, d))
    X[:, 0:3] = eta[:, 0:3] + rng.normal(0, 0.3, (n, 3))
    X[:, 3:6] = rng.binomial(1, 1 / (1 + np.exp(-eta[:, 3:6])))
    X[:, 6:9] = rng.poisson(np.exp(np.clip(eta[:, 6:9], -20, 5)))
    eta_y = w0_true + w_true * (Z @ Z.T)
    Y = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    Y[iu] = rng.poisson(np.exp(np.clip(eta_y[iu], -20, 10)))
    Y = Y + Y.T
    return dict(X=X, Y=Y, Z=Z, F=F, family_x_list=fam_list,
                w0=w0_true, w=w_true)


def test_mixed_em_smoke():
    """混在 X で EM が回り、Z を X-only ノイズ水準より良く回復する。"""
    data = generate_mixed_x_data()
    res = run_em_experimental(
        data["X"], data["Y"], family_x=None, family_y="poisson",
        k=2, L=3, num_iter=4, seed=9,
        family_x_list=data["family_x_list"])
    assert not res["nan_occurred"], "NaN in mixed EM"
    assert np.isfinite(res["Q_strict"])
    R, k_min = procrustes_rotation(res["Z_est"], data["Z"])
    rmse_Z = calc_rmse(data["Z"][:, :k_min], res["Z_est"][:, :k_min] @ R)
    assert rmse_Z < 0.8, f"rmse_Z={rmse_Z} too large"
    print(f"PASS: test_mixed_em_smoke (rmse_Z={rmse_Z:.3f}, "
          f"w0={res['w0']:.3f}, w={res['w']:.3f}, BIC={res['bic']:.1f})")


def test_mixed_sigma_only_gaussian_cols():
    """calc_sigma が Gaussian 列のみ推定し、他列は 1 のまま。"""
    data = generate_mixed_x_data()
    model = DualExpFamLSMPerColumn(n=60, d=9, k=2, L=3,
                                   family_x_list=data["family_x_list"],
                                   family_y="poisson")
    model.initialize_params(seed=1)
    Zs = np.stack([data["Z"]] * 3, axis=2)
    sigma = model.calc_sigma(data["X"], Zs, data["F"])
    diag = np.diag(sigma)
    assert np.allclose(diag[3:], 1.0), "non-gaussian sigma should stay 1"
    assert np.all(diag[:3] > 0) and not np.allclose(diag[:3], 1.0)
    print("PASS: test_mixed_sigma_only_gaussian_cols")


if __name__ == "__main__":
    test_uniform_percolumn_equals_scalar()
    test_mixed_sigma_only_gaussian_cols()
    test_mixed_em_smoke()
    print("\nALL PER-COLUMN TESTS PASSED")
