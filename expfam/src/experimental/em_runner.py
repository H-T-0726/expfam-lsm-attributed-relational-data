"""
em_runner — masked / NB / ablation 対応の汎用 MCEM ランナー。

utils_expfam.run_em_dual と同じ EM ループ構造（NaN ガード + リトライ、
informed init、scale_Z）を、experimental モデル群
（DualExpFamLSMMasked / DualExpFamLSMNB）に対して提供する。
既存の run_em_dual は変更しない。

対応オプション:
    train_mask : strict held-out 学習（観測ペアのみで学習）
    family_y='nb' + nb_r : NB2-Y
    fix_x : F=0 固定（Y-only; X 信号遮断）
    fix_w : w=0 固定（X-only; Y 信号遮断）
"""

import time
import warnings
import numpy as np
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from model_dual_expfam_masked import DualExpFamLSMMasked      # noqa: E402
from model_dual_expfam_nb import DualExpFamLSMNB              # noqa: E402
from model_dual_expfam_percolumn import DualExpFamLSMPerColumn  # noqa: E402
from eval_utils import calc_Q_dual_strict_exp, calc_bic_exp   # noqa: E402
from diagnostics import poisson_clip_diagnostics, validate_xy  # noqa: E402


def build_model(n, d, k, L, family_x, family_y, nb_r=None,
                sigma_y=1.0, train_mask=None, family_x_list=None):
    """family_y / family_x_list に応じて experimental モデルを構築する。"""
    if family_x_list is not None:
        if family_y == "nb":
            raise NotImplementedError(
                "per-column X と NB-Y の併用は未実装（設計書参照）")
        return DualExpFamLSMPerColumn(n=n, d=d, k=k, L=L,
                                      family_x_list=family_x_list,
                                      family_y=family_y, sigma_y=sigma_y,
                                      train_mask=train_mask)
    if family_y == "nb":
        if nb_r is None:
            raise ValueError("family_y='nb' requires nb_r")
        return DualExpFamLSMNB(n=n, d=d, k=k, L=L, family_x=family_x,
                               nb_r=nb_r, sigma_y=sigma_y,
                               train_mask=train_mask)
    return DualExpFamLSMMasked(n=n, d=d, k=k, L=L, family_x=family_x,
                               family_y=family_y, sigma_y=sigma_y,
                               train_mask=train_mask)


def run_em_experimental(
    X, Y,
    family_x: str,
    family_y: str,          # 'gaussian' | 'bernoulli' | 'poisson' | 'nb'
    k: int,
    L: int = 5,
    num_iter: int = 8,
    seed: int = 42,
    train_mask=None,
    nb_r: float = None,
    fix_x: bool = False,
    fix_w: bool = False,
    family_x_list=None,        # per-column X（指定時 family_x は無視）
    compute_strict_Q: bool = True,
    verbose: bool = False,
    validate_support: bool = False,
    allow_support_mismatch: bool = False,
    mstep_q_diagnostic: bool = False,
    compute_clip_diagnostic: bool = False,
) -> dict:
    """
    MCEM 実行（NaN ガード + 最大 2 回リトライ、リトライ毎に newton_alpha 半減）。

    Optional diagnostics（いずれも既定 False。推定アルゴリズム・
    乱数生成順序・比較可能な数値結果は変更しない）:

    validate_support : True のとき学習前に family とデータの台を検査する
        （diagnostics.validate_xy）。違反時は ValueError。誤指定実験は
        allow_support_mismatch=True で警告付きで通せる。既定 False では
        従来どおり検査しない。
    mstep_q_diagnostic : True のとき各 EM 反復で、E-step 直後の
        **同一 Z_samples・同一 train_mask** を固定して M-step 前後の
        strict Q を計算し `mstep_q_history` に記録する（M-step 単調性診断）。
        E-step を跨いだ Q 比較は行わない（MC サンプルが異なるため
        単調性診断とは呼べない）。診断計算は決定的で乱数を消費しない。
    compute_clip_diagnostic : True のときのみ、最終推定値に対する
        Poisson clip 発動率を post-hoc に計算する。EM 反復中の発動率ではない。
        False のとき診断関数・診断用行列積を実行しない。

    Returns dict:
        Z_est, Z_samples, F, sigma, w0, w, var_z, sigma_y_est, model,
        Q_strict, bic, num_params, nan_occurred, nan_count, runtime_s,
        failure_reason  : Q/BIC 計算が失敗した場合の "例外型: メッセージ"、
                          成功時・未計算時は None（新キー）
        q_bic_failed    : Q/BIC 計算が例外で失敗したか（新キー、bool）
        clip_diag       : compute_clip_diagnostic=True の場合のみ、最終推定値での
                          Poisson clip 発動診断
                          （diagnostics.poisson_clip_diagnostics の dict。
                          post-hoc・純計算）。無効時・計算不能時は None（新キー）
        mstep_q_history : mstep_q_diagnostic=True のときのみ各反復の
                          {iteration, q_before, q_after, q_diff, decreased}
                          のリスト。無効時は空リスト（新キー）
    """
    n, d = X.shape
    max_retries = 2
    t0 = time.perf_counter()

    if validate_support:
        # 'nb' の台は Poisson と同じ非負整数
        _fy_support = "poisson" if family_y == "nb" else family_y
        validate_xy(X, Y, family_x=family_x, family_x_list=family_x_list,
                    family_y=_fy_support, train_mask=train_mask,
                    allow_support_mismatch=allow_support_mismatch)

    mstep_q_history = []

    for retry in range(max_retries + 1):
        newton_alpha = 0.5 / (2 ** retry)

        rng = np.random.default_rng(seed + retry * 1000)
        model = build_model(n, d, k, L, family_x, family_y,
                            nb_r=nb_r, train_mask=train_mask,
                            family_x_list=family_x_list)
        model.initialize_params(true_params=None, seed=seed + retry * 1000)

        # ── Informed init: Y 側（観測ペアのみで統計をとる） ──────────
        obs_upper = np.triu(model.train_mask, k=1)
        y_obs = Y[obs_upper]
        if family_y == "bernoulli":
            density = float(np.clip(y_obs.mean(), 1e-6, 1 - 1e-6))
            model.params["w0"] = np.log(density / (1 - density))
            model.params["w"] = 0.5
        elif family_y in ("poisson", "nb"):
            mean_cnt = float(y_obs[y_obs > 0].mean()) if np.any(y_obs > 0) else 1.0
            model.params["w0"] = np.log(mean_cnt + 1e-10)
            model.params["w"] = 0.1 / (2 ** retry)
        else:  # gaussian
            model.params["w0"] = float(y_obs.mean())
            model.params["w"] = 0.5
            model.sigma_y = float(max(y_obs.std(), 0.01))

        # ── Informed init: X 側 ──────────────────────────────────────
        if family_x_list is not None:
            # 非 Gaussian 列の F 行のみ縮小（η を数値安全域に）
            for j, fam in enumerate(family_x_list):
                if fam in ("bernoulli", "poisson"):
                    model.params["F"][j, :] *= 0.2
        elif family_x in ("bernoulli", "poisson"):
            model.params["F"] *= 0.2

        Z = model.params["Z"].copy()
        F = model.params["F"].copy()
        sigma = model.params["sigma"].copy()
        w0 = float(model.params["w0"])
        w = float(model.params["w"])
        var_z = float(model.params["var_z"])

        if fix_w:
            w = 0.0
            model.params["w"] = 0.0
        if fix_x:
            F = np.zeros((d, k))
            model.params["F"] = np.zeros((d, k))

        Z_prev = Z.copy()
        nan_count = 0
        mstep_q_history = []   # retry ごとにリセット（最終 retry の履歴のみ返す）

        for iteration in range(1, num_iter + 1):
            # ── E-step ───────────────────────────────────────────────
            Z_samples = np.zeros((n, k, L))
            for l in range(L):
                model.params.update(dict(Z=Z.copy(), F=F, sigma=sigma,
                                         w0=w0, w=w))
                Z_new = model.calc_eta_newton(
                    X, Y, rng=rng, max_iter=10, alpha=newton_alpha)
                Z_samples[:, :, l] = Z_new
                Z = Z_new.copy()

            if np.any(np.isnan(Z_samples)) or np.any(np.isinf(Z_samples)):
                nan_count += 1
                if verbose:
                    print(f"  [NaN iter={iteration} retry={retry}] Resetting.")
                Z_samples = np.stack([Z_prev] * L, axis=2)
                Z = Z_prev.copy()

            Z_samples = model.scale_Z(Z_samples)
            Z_prev = Z.copy()
            Z = Z_samples[:, :, -1].copy()

            # ── M-step Q 診断（前）: 同一 Z_samples・同一 mask で旧 θ の Q ──
            # model.params / model.sigma_y はこの時点で旧 θ のまま（E-step が
            # 旧 θ で更新済み）。診断は決定的で乱数を消費しない。
            if mstep_q_diagnostic:
                try:
                    _q_before = calc_Q_dual_strict_exp(
                        X, Y, Z_samples, F, sigma, var_z, w0, w, model)
                except Exception:
                    _q_before = float("nan")

            # ── M-step ───────────────────────────────────────────────
            if not fix_x:
                F = model.calc_F(X, Z_samples)
                sigma = model.calc_sigma(X, Z_samples, F)
            w0 = float(model.calc_w0(Y, Z_samples, w0, w, max_iter=50))
            if not fix_w:
                w = float(model.calc_w(Y, Z_samples, w0, w, max_iter=50))
            if family_y == "gaussian":
                model.calc_sigma_y(Y, Z_samples, w0, w)

            # ── M-step Q 診断（後）: 同一 Z_samples で新 θ の Q ──────────
            if mstep_q_diagnostic:
                # calc_log_likelihood_X(Gaussian) は model.params["sigma"] を
                # 参照するため新 θ を反映（次の E-step が同じ値で上書きするので
                # 推定経路は変わらない）
                model.params.update(dict(F=F, sigma=sigma, w0=w0, w=w))
                try:
                    _q_after = calc_Q_dual_strict_exp(
                        X, Y, Z_samples, F, sigma, var_z, w0, w, model)
                except Exception:
                    _q_after = float("nan")
                _diff = _q_after - _q_before
                mstep_q_history.append({
                    "iteration": iteration,
                    "q_before": float(_q_before),
                    "q_after": float(_q_after),
                    "q_diff": float(_diff),
                    "decreased": bool(np.isfinite(_diff) and _diff < 0.0),
                })

            if verbose:
                print(f"    iter={iteration:2d} w0={w0:.4f} w={w:.4f}")

        nan_occurred = nan_count > 0
        if not nan_occurred:
            break

    model.params.update({"F": F, "sigma": sigma, "w0": w0, "w": w})

    Q_strict = float("nan")
    bic = float("nan")
    npar = 0
    failure_reason = None
    q_bic_failed = False
    if compute_strict_Q:
        try:
            Q_strict = calc_Q_dual_strict_exp(
                X, Y, Z_samples, F, sigma, var_z, w0, w, model)
            label = getattr(model, "family_y_label", model.family)
            n_gauss_cols = (len(model.columns_of("gaussian"))
                            if model.family_x == "mixed" else None)
            bic, npar = calc_bic_exp(Q_strict, k, n, d,
                                     model.family_x, label,
                                     n_gaussian_x_cols=n_gauss_cols)
        except Exception as exc:
            # 例外の握りつぶしをやめ、失敗を可視化する（処理は継続、
            # 従来どおり Q_strict / bic は NaN のまま返す）
            failure_reason = f"{type(exc).__name__}: {exc}"
            q_bic_failed = True
            warnings.warn(
                "run_em_experimental: Q/BIC computation failed; "
                f"Q_strict/bic set to NaN ({failure_reason})",
                RuntimeWarning, stacklevel=2)

    # ── opt-in post-hoc 診断: Poisson clip 発動率（純計算、乱数不使用） ──
    clip_diag = None
    if compute_clip_diagnostic:
        try:
            clip_diag = poisson_clip_diagnostics(
                model, Z_samples[:, :, -1], F, w0, w)
        except Exception as exc:
            warnings.warn(
                "run_em_experimental: clip diagnostics failed "
                f"({type(exc).__name__}: {exc})", RuntimeWarning, stacklevel=2)

    return {
        "Z_est": Z_samples[:, :, -1].copy(),
        "Z_samples": Z_samples,
        "F": F, "sigma": sigma,
        "w0": w0, "w": w, "var_z": var_z,
        "sigma_y_est": float(model.sigma_y),
        "model": model,
        "Q_strict": Q_strict, "bic": bic, "num_params": npar,
        "nan_occurred": nan_occurred, "nan_count": nan_count,
        "runtime_s": round(time.perf_counter() - t0, 2),
        # ── 新キー（診断・可視化のみ） ──────────────────────────────
        "failure_reason": failure_reason,
        "q_bic_failed": q_bic_failed,
        "clip_diag": clip_diag,
        "mstep_q_history": mstep_q_history,
    }


def predict_mu_y(result: dict, clip_max: float = 1e5) -> np.ndarray:
    """最終推定値から E[Y] 行列（plug-in）を計算する。"""
    model = result["model"]
    Z_est = result["Z_est"]
    eta_y = result["w0"] + result["w"] * (Z_est @ Z_est.T)
    return np.clip(model._mean_function(eta_y), 0.0, clip_max)
