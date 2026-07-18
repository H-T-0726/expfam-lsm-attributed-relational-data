"""diagnostics.py と em_runner の診断・validation 追加分のテスト。

実行: python expfam/src/experimental/test_diagnostics_validation.py
      または pytest -p no:cacheprovider

確認事項（承認条件 2026-07-19）:
    - diagnostics の import に副作用がない（乱数状態を変えない）
    - clip 診断・validation の正常系 / 違反系 / 許可フラグ系
    - 既定引数の run_em_experimental が診断追加後も決定的で、
      診断フラグ on/off で推定値が変わらない
    - 通常実行では runner 由来の warning が出ない
    - Q/BIC 失敗時のみ failure_reason / q_bic_failed / warning が出る
    - validate_support の opt-in 動作
"""

import warnings

import numpy as np
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_SRC = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent.parent / "reproduction" / "src"))

import diagnostics  # noqa: E402
from diagnostics import (  # noqa: E402
    clip_activation_rate, poisson_clip_diagnostics,
    validate_family_support, validate_family_x_list, validate_xy,
    POISSON_CLIP_LO, POISSON_CLIP_HI,
)
import em_runner  # noqa: E402
from em_runner import run_em_experimental  # noqa: E402
from eval_utils import make_pair_split  # noqa: E402
from model_dual_expfam_percolumn import DualExpFamLSMPerColumn  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────

COMPARABLE_NUMERIC_KEYS = (
    "Z_est", "Z_samples", "F", "sigma",
    "w0", "w", "var_z", "sigma_y_est",
    "Q_strict", "bic", "num_params", "nan_occurred", "nan_count",
)


def _toy_data(seed=5, n=16, d=6, k=2):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, k))
    F = rng.standard_normal((d, k)) * 0.4
    eta = Z @ F.T
    X = np.zeros((n, d))
    X[:, :2] = eta[:, :2] + rng.normal(0, 0.3, (n, 2))
    X[:, 2:4] = rng.binomial(1, 1 / (1 + np.exp(-eta[:, 2:4])))
    X[:, 4:6] = rng.poisson(np.exp(np.clip(eta[:, 4:6], -20, 5)))
    eta_y = 0.8 + 0.3 * (Z @ Z.T)
    Y = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    Y[iu] = rng.poisson(np.exp(np.clip(eta_y[iu], -20, 10)))
    Y = Y + Y.T
    fam_list = ["gaussian"] * 2 + ["bernoulli"] * 2 + ["poisson"] * 2
    return X, Y, fam_list


def _run(X, Y, fam_list, **kw):
    return run_em_experimental(
        X, Y, family_x=None, family_y="poisson", k=2, L=2, num_iter=2,
        seed=21, family_x_list=fam_list, **kw)


def _run_masked(X, Y, **kw):
    return run_em_experimental(
        X, Y, family_x="gaussian", family_y="poisson", k=2, L=2,
        num_iter=2, seed=21, **kw)


def _assert_same_comparable_numeric_keys(r1, r2):
    """Compare 13 deterministic numeric results.

    ``model`` is an object with mutable internal state and is not a direct bitwise
    comparison target. ``runtime_s`` is elapsed wall-clock time and is likewise
    excluded. This is an off/on comparison within the modified code, not a
    reconstruction of the deleted before-change snapshot.
    """
    assert len(COMPARABLE_NUMERIC_KEYS) == 13
    for kk in COMPARABLE_NUMERIC_KEYS:
        v1, v2 = r1[kk], r2[kk]
        if isinstance(v1, np.ndarray):
            assert np.array_equal(v1, v2), f"array key '{kk}' differs"
        elif isinstance(v1, float) and np.isnan(v1):
            assert np.isnan(v2), f"scalar key '{kk}' differs (nan vs value)"
        else:
            assert v1 == v2, f"scalar key '{kk}' differs: {v1} vs {v2}"


# ──────────────────────────────────────────────────────────────────────
# diagnostics module
# ──────────────────────────────────────────────────────────────────────

def test_import_has_no_side_effects():
    """import が乱数状態を変えない（副作用なしの代表チェック）。"""
    state_before = np.random.get_state()[1].copy()
    import importlib
    importlib.reload(diagnostics)
    state_after = np.random.get_state()[1].copy()
    assert np.array_equal(state_before, state_after), \
        "importing diagnostics consumed global RNG state"
    print("PASS: test_import_has_no_side_effects")


def test_clip_activation_rate():
    eta = np.array([-25.0, -20.0, 0.0, 10.0, 12.0, 30.0])
    r = clip_activation_rate(eta)
    assert r["n_total"] == 6
    assert r["n_below"] == 1          # -25 のみ（-20 は境界内）
    assert r["n_above"] == 2          # 12, 30（10 は境界内）
    assert np.isclose(r["rate"], 3 / 6)
    r0 = clip_activation_rate(np.array([]))
    assert r0["n_total"] == 0 and r0["rate"] == 0.0
    assert POISSON_CLIP_LO == -20.0 and POISSON_CLIP_HI == 10.0
    print("PASS: test_clip_activation_rate")


def test_poisson_clip_diagnostics_readonly():
    X, Y, fam_list = _toy_data()
    n, d, k = X.shape[0], X.shape[1], 2
    model = DualExpFamLSMPerColumn(n=n, d=d, k=k, L=2,
                                   family_x_list=fam_list,
                                   family_y="poisson")
    model.initialize_params(seed=3)
    rng_state = np.random.get_state()[1].copy()
    Z_pt = np.random.default_rng(9).standard_normal((n, k))
    F = model.params["F"].copy()
    F_before = F.copy()
    Z_before = Z_pt.copy()

    diag = poisson_clip_diagnostics(model, Z_pt, F, w0=0.5, w=0.2)

    assert diag["x_side"] is not None, "poisson X cols should be diagnosed"
    assert diag["y_side"] is not None, "poisson Y should be diagnosed"
    assert diag["x_side"]["n_total"] == n * 2      # poisson 2 列
    assert diag["y_side"]["n_total"] == n * (n - 1) // 2
    # 読み取り専用: 入力・乱数状態を変更していない
    assert np.array_equal(F, F_before) and np.array_equal(Z_pt, Z_before)
    assert np.array_equal(np.random.get_state()[1], rng_state)
    # 極端な w0 で Y 側 clip を強制発動させ、率が 1 になることを確認
    diag_hi = poisson_clip_diagnostics(model, Z_pt, F, w0=100.0, w=0.0)
    assert diag_hi["y_side"]["rate"] == 1.0
    print("PASS: test_poisson_clip_diagnostics_readonly")


# ──────────────────────────────────────────────────────────────────────
# validation
# ──────────────────────────────────────────────────────────────────────

def test_validation_normal_cases():
    ok_bern = np.array([[0.0, 1.0], [1.0, 0.0]])     # float の 0/1 は許可
    ok_pois = np.array([[0.0, 3.0], [2.0, 7.0]])     # float 表現の整数は許可
    ok_gauss = np.array([[-1.2, 0.5], [3.4, -0.1]])
    for data, fam in [(ok_bern, "bernoulli"), (ok_pois, "poisson"),
                      (ok_gauss, "gaussian")]:
        rep = validate_family_support(data, fam)
        assert rep["ok"] and rep["n_violations"] == 0, fam
    # Poisson: 数値誤差の範囲は許可（丸めはしない）
    rep = validate_family_support(np.array([3.0 + 5e-9]), "poisson")
    assert rep["ok"]
    print("PASS: test_validation_normal_cases")


def test_validation_violations_and_flag():
    bad_bern = np.array([[0.0, 0.5], [1.0, 1.0]])
    bad_pois = np.array([[-1.0, 2.0], [2.5, 3.0]])
    bad_gauss = np.array([[np.inf, 0.0], [np.nan, 1.0]])
    for data, fam, n_bad in [(bad_bern, "bernoulli", 1),
                             (bad_pois, "poisson", 2),
                             (bad_gauss, "gaussian", 2)]:
        try:
            validate_family_support(data, fam)
            raise AssertionError(f"should raise for {fam}")
        except ValueError:
            pass
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            rep = validate_family_support(
                data, fam, allow_support_mismatch=True)
        assert rep["n_violations"] == n_bad, (fam, rep)
        assert any("quasi-likelihood" in str(x.message) for x in wlist), \
            "allow flag must still warn"
    # unknown family は許可フラグでも常に ValueError
    try:
        validate_family_support(np.zeros(3), "categorical",
                                allow_support_mismatch=True)
        raise AssertionError("unknown family should raise")
    except ValueError:
        pass
    print("PASS: test_validation_violations_and_flag")


def test_validation_mask_and_xy():
    # mask=False（未観測）の NaN は検査対象外
    Y = np.array([[0.0, 1.0, np.nan],
                  [1.0, 0.0, 0.0],
                  [np.nan, 0.0, 0.0]])
    mask = ~np.isnan(Y)
    rep = validate_family_support(Y, "bernoulli", mask=mask)
    assert rep["ok"]
    # validate_xy: 列ごと family + Y 観測ペアのみ
    X, Yc, fam_list = _toy_data()
    reports = validate_xy(X, Yc, family_x_list=fam_list, family_y="poisson")
    assert all(r["ok"] for r in reports), reports
    # family_x_list 長さ違反
    try:
        validate_family_x_list(["gaussian"], d=3)
        raise AssertionError("length mismatch should raise")
    except ValueError:
        pass
    print("PASS: test_validation_mask_and_xy")


# ──────────────────────────────────────────────────────────────────────
# em_runner 互換性
# ──────────────────────────────────────────────────────────────────────

def test_runner_default_no_warning_and_new_keys():
    X, Y, fam_list = _toy_data()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        res = _run(X, Y, fam_list)
    runner_warnings = [x for x in wlist
                       if "run_em_experimental" in str(x.message)]
    assert not runner_warnings, f"unexpected warnings: {runner_warnings}"
    assert res["failure_reason"] is None
    assert res["q_bic_failed"] is False
    assert res["mstep_q_history"] == []
    assert res["clip_diag"] is None
    assert np.isfinite(res["Q_strict"]) and np.isfinite(res["bic"])
    print("PASS: test_runner_default_no_warning_and_new_keys")


def test_runner_deterministic_and_rng_isolated():
    X, Y, fam_list = _toy_data()
    g_state = np.random.get_state()[1].copy()
    r1 = _run(X, Y, fam_list)
    r2 = _run(X, Y, fam_list)
    _assert_same_comparable_numeric_keys(r1, r2)
    assert np.array_equal(np.random.get_state()[1], g_state), \
        "runner must not consume the global RNG"
    print("PASS: test_runner_deterministic_and_rng_isolated")


def test_diagnostics_flags_do_not_change_comparable_numeric_results():
    X, Y, fam_list = _toy_data()
    configurations = (
        ("per-column", lambda **kw: _run(X, Y, fam_list, **kw)),
        ("masked", lambda **kw: _run_masked(X, Y, **kw)),
    )
    for label, runner in configurations:
        r_off = runner()
        r_on = runner(mstep_q_diagnostic=True, validate_support=True,
                      compute_clip_diagnostic=True)
        _assert_same_comparable_numeric_keys(r_off, r_on)
        assert r_off["clip_diag"] is None
        assert r_on["clip_diag"] is not None
        hist = r_on["mstep_q_history"]
        assert len(hist) == 2      # num_iter=2
        for h in hist:
            assert set(h) == {"iteration", "q_before", "q_after",
                              "q_diff", "decreased"}
            assert np.isfinite(h["q_before"]) and np.isfinite(h["q_after"])
            assert np.isclose(h["q_diff"], h["q_after"] - h["q_before"])
        print(f"PASS: comparable numeric keys unchanged ({label}, 13/13)")


def test_clip_diagnostic_is_optin_and_failure_is_nonfatal():
    X, Y, fam_list = _toy_data()
    original = em_runner.poisson_clip_diagnostics
    calls = []

    def _record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    em_runner.poisson_clip_diagnostics = _record
    try:
        r_off = _run(X, Y, fam_list)
        assert calls == [], "clip helper must not be called when opt-in is False"
        assert r_off["clip_diag"] is None
        r_on = _run(X, Y, fam_list, compute_clip_diagnostic=True)
        assert len(calls) == 1
        assert r_on["clip_diag"] is not None
        _assert_same_comparable_numeric_keys(r_off, r_on)
    finally:
        em_runner.poisson_clip_diagnostics = original

    def _boom(*args, **kwargs):
        raise RuntimeError("forced clip diagnostic failure")

    baseline = _run(X, Y, fam_list)
    em_runner.poisson_clip_diagnostics = _boom
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            failed = _run(X, Y, fam_list, compute_clip_diagnostic=True)
    finally:
        em_runner.poisson_clip_diagnostics = original
    assert failed["clip_diag"] is None
    assert any("clip diagnostics failed" in str(x.message) for x in wlist)
    _assert_same_comparable_numeric_keys(baseline, failed)
    print("PASS: clip diagnostic is opt-in and nonfatal")


def test_clip_diagnostic_without_poisson_family():
    X, Y, fam_list = _toy_data()
    model = DualExpFamLSMPerColumn(
        n=X.shape[0], d=X.shape[1], k=2, L=2,
        family_x_list=["gaussian"] * X.shape[1], family_y="bernoulli")
    model.initialize_params(seed=3)
    diag = poisson_clip_diagnostics(
        model, model.params["Z"], model.params["F"],
        model.params["w0"], model.params["w"])
    assert diag["x_side"] is None and diag["y_side"] is None
    print("PASS: no-Poisson model returns empty clip diagnostics")


def test_runner_failure_reason_on_q_exception():
    X, Y, fam_list = _toy_data()
    orig = em_runner.calc_Q_dual_strict_exp

    def _boom(*args, **kwargs):
        raise RuntimeError("forced Q failure for test")

    em_runner.calc_Q_dual_strict_exp = _boom
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            res = _run(X, Y, fam_list)
    finally:
        em_runner.calc_Q_dual_strict_exp = orig

    assert res["q_bic_failed"] is True
    assert res["failure_reason"] is not None
    assert "RuntimeError" in res["failure_reason"]
    assert np.isnan(res["Q_strict"]) and np.isnan(res["bic"])
    assert any("Q/BIC computation failed" in str(x.message) for x in wlist)
    # 失敗しても他の推定結果は返る（処理継続）
    assert np.all(np.isfinite(res["Z_est"]))
    print("PASS: test_runner_failure_reason_on_q_exception")


def test_runner_validate_support_optin():
    X, Y, fam_list = _toy_data()
    X_bad = X.copy()
    X_bad[0, 2] = 0.5            # bernoulli 列に台違反
    # 既定（validate_support=False）: 従来どおり通る
    res = _run(X_bad, Y, fam_list)
    assert np.all(np.isfinite(res["Z_est"]))
    # opt-in: 違反で ValueError
    try:
        _run(X_bad, Y, fam_list, validate_support=True)
        raise AssertionError("validate_support=True should raise")
    except ValueError:
        pass
    # 明示的許可フラグ: 警告付きで通る
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        res2 = _run(X_bad, Y, fam_list, validate_support=True,
                    allow_support_mismatch=True)
    assert any("quasi-likelihood" in str(x.message) for x in wlist)
    assert np.all(np.isfinite(res2["Z_est"]))
    print("PASS: test_runner_validate_support_optin")


if __name__ == "__main__":
    test_import_has_no_side_effects()
    test_clip_activation_rate()
    test_poisson_clip_diagnostics_readonly()
    test_validation_normal_cases()
    test_validation_violations_and_flag()
    test_validation_mask_and_xy()
    test_runner_default_no_warning_and_new_keys()
    test_runner_deterministic_and_rng_isolated()
    test_diagnostics_flags_do_not_change_comparable_numeric_results()
    test_clip_diagnostic_is_optin_and_failure_is_nonfatal()
    test_clip_diagnostic_without_poisson_family()
    test_runner_failure_reason_on_q_exception()
    test_runner_validate_support_optin()
    print("\nALL diagnostics/validation TESTS PASSED")
