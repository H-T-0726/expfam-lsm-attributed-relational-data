"""diagnostics — 読み取り専用の診断・validation ユーティリティ（experimental）。

位置づけ（理論監査 2026-07-18/19、承認項目 P0-2 / P1-6）:
    - import 時の副作用なし（ファイル生成・ログ出力・乱数消費・
      モデル状態の変更・外部実験の実行を行わない）。
    - 全関数は入力を変更しない読み取り専用（診断）関数である。
    - 本モジュールの診断量は、推定アルゴリズム・モデル選択基準には接続しない。

含まれるもの:
    POISSON_CLIP_LO / POISSON_CLIP_HI : モデル実装が用いる clip 範囲の記録値
    clip_activation_rate     : 自然パラメータ配列の clip 域該当率（純関数）
    poisson_clip_diagnostics : 最終推定値からの Poisson clip 発動診断（post-hoc）
    validate_family_support  : family とデータの台の整合 validation
    validate_family_x_list   : per-column family リストの検査
    validate_xy              : X / Y 一括 validation（em_runner の opt-in 用）

Poisson clip について（P0-2 文書化の要点）:
    実装は自然パラメータを eta_c = clip(eta, -20, 10) として A'(eta_c), A''(eta_c),
    尤度 x*eta_c - exp(eta_c) を計算する。clip 域では d(eta_c)/d(eta) = 0 である
    一方、実装の E-step 勾配は clip 後の残差 x - A'(eta_c) をそのまま返すため、
    clip 域では「実装尤度の勾配」と「実装勾配」は一致しない。clip が発動しない
    領域では両者は厳密に一致する（per-column 数式監査 31/31 PASS は非発動域）。
    発動率はこれまで未計測だったため、本モジュールで post-hoc に計測できるようにする。
    ここでは clip 範囲・モデル挙動は一切変更しない。
"""

import warnings

import numpy as np

# モデル実装（model_expfam._mean_function ほか）が用いる clip 範囲の記録値。
# ここを変えてもモデル挙動は変わらない（診断の判定基準としてのみ使用）。
POISSON_CLIP_LO = -20.0
POISSON_CLIP_HI = 10.0

_VALID_FAMILIES = ("gaussian", "bernoulli", "poisson")


# ──────────────────────────────────────────────────────────────────────
# P0-2: Poisson clip 診断（post-hoc、純関数）
# ──────────────────────────────────────────────────────────────────────

def clip_activation_rate(eta, lo=POISSON_CLIP_LO, hi=POISSON_CLIP_HI):
    """自然パラメータ配列 eta のうち clip 域 [lo, hi] の外にある割合を返す。

    純関数: 入力を変更せず、乱数も消費しない。

    Returns
    -------
    dict with keys: n_total, n_below, n_above, rate
    """
    eta = np.asarray(eta, dtype=float)
    n_total = int(eta.size)
    if n_total == 0:
        return {"n_total": 0, "n_below": 0, "n_above": 0, "rate": 0.0}
    n_below = int(np.sum(eta < lo))
    n_above = int(np.sum(eta > hi))
    return {
        "n_total": n_total,
        "n_below": n_below,
        "n_above": n_above,
        "rate": float((n_below + n_above) / n_total),
    }


def poisson_clip_diagnostics(model, Z_point, F, w0, w,
                             lo=POISSON_CLIP_LO, hi=POISSON_CLIP_HI):
    """最終推定値（点推定 Z_point と θ）から Poisson clip 発動率を計測する。

    post-hoc 診断であり、モデル・パラメータ・乱数状態を一切変更しない。
    EM 反復中の発動率ではなく「最終推定値における clip 域該当率」である点に注意
    （反復中の発動はこれより多い/少ない可能性がある）。

    Parameters
    ----------
    model : DualExpFamLSM 系列のインスタンス（family_x / family / columns_of /
            train_mask を読み取り専用で参照する）
    Z_point : (n, k) 点推定に用いる Z（例: 最終 MC サンプル）
    F, w0, w : 最終推定パラメータ

    Returns
    -------
    dict with keys:
        clip_lo, clip_hi : 判定に用いた範囲
        x_side : Poisson 列がある場合 clip_activation_rate の dict、なければ None
        y_side : family_y='poisson' の場合 観測上三角ペアに対する dict、なければ None
    """
    Z_point = np.asarray(Z_point, dtype=float)
    F = np.asarray(F, dtype=float)
    out = {"clip_lo": float(lo), "clip_hi": float(hi),
           "x_side": None, "y_side": None}

    # ── X 側: Poisson 列の eta ────────────────────────────────────────
    family_x = getattr(model, "family_x", None)
    pois_cols = None
    if family_x == "poisson":
        pois_cols = np.arange(F.shape[0])
    elif family_x == "mixed" and hasattr(model, "columns_of"):
        cols = model.columns_of("poisson")
        if len(cols):
            pois_cols = np.asarray(cols)
    if pois_cols is not None and len(pois_cols):
        eta_x = Z_point @ F[pois_cols, :].T      # (n, n_pois_cols)
        out["x_side"] = clip_activation_rate(eta_x, lo, hi)

    # ── Y 側: family_y='poisson' の観測上三角ペアの eta ───────────────
    if getattr(model, "family", None) == "poisson":
        n = Z_point.shape[0]
        eta_y = float(w0) + float(w) * (Z_point @ Z_point.T)
        train_mask = getattr(model, "train_mask", None)
        if train_mask is not None:
            obs_upper = np.triu(np.asarray(train_mask, dtype=bool), k=1)
        else:
            obs_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
        out["y_side"] = clip_activation_rate(eta_y[obs_upper], lo, hi)

    return out


# ──────────────────────────────────────────────────────────────────────
# P1-6: family とデータの台の validation
# ──────────────────────────────────────────────────────────────────────

def validate_family_support(data, family, *, mask=None,
                            allow_support_mismatch=False,
                            name="data", integer_tol=1e-8):
    """family の台とデータ値の整合を検査する（非破壊・読み取り専用）。

    検査規則:
        - 全 family: 検査対象の値は有限（NaN / inf は違反）。
        - bernoulli: 値は 0.0 または 1.0（float 表現の 0.0 / 1.0 を許可）。
        - poisson  : 有限・非負・整数値（|v - round(v)| <= integer_tol を許可。
                     丸めは行わない）。
        - gaussian : 有限実数のみ。

    Parameters
    ----------
    mask : bool array（data と同形）or None
        True = 観測（検査対象）、False = 未観測（検査しない）。
        未観測エントリの NaN 等はここで除外して渡すこと。
    allow_support_mismatch : bool
        True のとき、違反があっても例外を投げず UserWarning を出して
        report を返す（誤指定実験の明示的許可フラグ）。違反は
        quasi-likelihood 的な使用であり正しい確率モデルではないことを警告する。

    Returns
    -------
    report : dict with keys
        name, family, n_checked, n_violations, violation_examples
        （最初の最大 5 件の (flat_index, value)）, ok

    Raises
    ------
    ValueError : family が不正、mask 形状不一致、または
        違反があり allow_support_mismatch=False の場合。
    """
    if family not in _VALID_FAMILIES:
        raise ValueError(
            f"unknown family '{family}' for {name}; "
            f"choose from {_VALID_FAMILIES}")

    arr = np.asarray(data, dtype=float)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != arr.shape:
            raise ValueError(
                f"mask shape {m.shape} != {name} shape {arr.shape}")
        flat_idx = np.flatnonzero(m.ravel())
        vals = arr.ravel()[flat_idx]
    else:
        flat_idx = np.arange(arr.size)
        vals = arr.ravel()

    bad = ~np.isfinite(vals)
    if family == "bernoulli":
        bad |= ~((vals == 0.0) | (vals == 1.0))
    elif family == "poisson":
        with np.errstate(invalid="ignore"):
            bad |= (vals < 0.0)
            bad |= np.abs(vals - np.round(vals)) > integer_tol
    # gaussian: 有限性のみ

    viol_pos = np.flatnonzero(bad)
    examples = [(int(flat_idx[p]), float(vals[p])) for p in viol_pos[:5]]
    report = {
        "name": name,
        "family": family,
        "n_checked": int(vals.size),
        "n_violations": int(viol_pos.size),
        "violation_examples": examples,
        "ok": viol_pos.size == 0,
    }

    if viol_pos.size:
        msg = (f"{name}: {viol_pos.size}/{vals.size} values violate the "
               f"support of family '{family}' (examples: {examples}). ")
        if allow_support_mismatch:
            warnings.warn(
                msg + "Proceeding because allow_support_mismatch=True: this "
                "is a quasi-likelihood use, NOT a valid probability model "
                "for the data; likelihood/BIC values are not interpretable "
                "as such.", UserWarning, stacklevel=2)
        else:
            raise ValueError(
                msg + "Pass allow_support_mismatch=True to proceed "
                "explicitly (misspecification experiments).")
    return report


def validate_family_x_list(family_x_list, d):
    """per-column family リストの長さと値を検査する（読み取り専用）。"""
    if len(family_x_list) != d:
        raise ValueError(
            f"family_x_list length {len(family_x_list)} != d={d}")
    for j, fam in enumerate(family_x_list):
        if fam not in _VALID_FAMILIES:
            raise ValueError(
                f"family_x_list[{j}]='{fam}' is not one of {_VALID_FAMILIES}")


def validate_xy(X, Y, *, family_x=None, family_x_list=None, family_y=None,
                train_mask=None, allow_support_mismatch=False):
    """X（列ごと/全列）と Y（観測ペアのみ）の台を一括検査する。

    - X: family_x_list があれば列ごとに、なければ family_x で全列を検査。
    - Y: train_mask（True=観測）があれば観測エントリのみ、なければ非対角のみ検査。
      train_mask=False は「未観測ペア」であり「観測された 0」ではない（検査しない）。

    Returns
    -------
    list of report dicts（validate_family_support の戻り値）。
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, d = X.shape
    reports = []

    if family_x_list is not None:
        validate_family_x_list(family_x_list, d)
        for fam in _VALID_FAMILIES:
            cols = [j for j, f in enumerate(family_x_list) if f == fam]
            if cols:
                reports.append(validate_family_support(
                    X[:, cols], fam,
                    allow_support_mismatch=allow_support_mismatch,
                    name=f"X[:, {fam} cols]"))
    elif family_x is not None:
        reports.append(validate_family_support(
            X, family_x,
            allow_support_mismatch=allow_support_mismatch, name="X"))

    if family_y is not None:
        if train_mask is not None:
            y_mask = np.asarray(train_mask, dtype=bool).copy()
        else:
            y_mask = np.ones_like(Y, dtype=bool)
        np.fill_diagonal(y_mask, False)
        reports.append(validate_family_support(
            Y, family_y, mask=y_mask,
            allow_support_mismatch=allow_support_mismatch, name="Y(observed)"))

    return reports
