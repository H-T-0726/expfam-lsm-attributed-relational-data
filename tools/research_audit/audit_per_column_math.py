"""
per-column family（DualExpFamLSMPerColumn）の数式監査スクリプト。

目的:
    per-column X の尤度・勾配・precision contribution が数学的に正しいかを、
    実装から独立に書き下した式と数値微分で検証する（prototype 検証フェーズ）。

監査項目:
  A. family 基本形     : A(η), A'(η), A''(η) の有限差分整合（bern/pois/gauss）
  B. 勾配              : _calc_gradient vs 独立実装の負対数事後の中心差分
                         （family_y ∈ {poisson, bernoulli, gaussian}、mask あり/なし）
  C. precision         : _calc_precision_matrix vs _calc_gradient の数値ヤコビアン
                         （canonical link なので Hessian は厳密に一致するはず）
  D. 列和構造          : Term2 が gradient_X = Σ_l g_l / precision_X = Σ_l P_l と
                         列ごとの独立計算の和に一致するか
  E. 既存モデル整合    : 全列同一 family の per-column ≡ DualExpFamLSMMasked
                         （勾配・precision・llX）+ _calc_F_adam_weighted ≡ _calc_F_adam
  F. 尤度 vs scipy     : calc_log_likelihood_X を scipy.stats と照合
                         （Poisson は −ln(x!) 省略規約を補正して比較）
  G. ブロック重み診断  : ブロック別列数・勾配ノルム・llX 寄与の記録（情報のみ、
                         weighting は実装しない）

出力: expfam/results/per_column_family/per_column_math_audit_summary.csv

実行: python tools/research_audit/audit_per_column_math.py
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm, poisson as sp_poisson

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "expfam" / "src"))
sys.path.insert(0, str(_ROOT / "expfam" / "src" / "experimental"))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from model_dual_expfam_masked import DualExpFamLSMMasked        # noqa: E402
from model_dual_expfam_percolumn import DualExpFamLSMPerColumn  # noqa: E402

OUT_DIR = _ROOT / "expfam" / "results" / "per_column_family"

ROWS = []


def record(check, detail, value, tol, extra=""):
    ok = bool(value < tol)
    ROWS.append({"check": check, "detail": detail,
                 "max_abs_diff": float(value), "tol": tol,
                 "result": "PASS" if ok else "FAIL", "extra": extra})
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {check} / {detail}: {value:.3e} (tol {tol:g}) {extra}")
    return ok


def record_info(check, detail, value, extra=""):
    ROWS.append({"check": check, "detail": detail,
                 "max_abs_diff": float(value), "tol": np.nan,
                 "result": "INFO", "extra": extra})
    print(f"[INFO] {check} / {detail}: {value:.4g} {extra}")


# ──────────────────────────────────────────────────────────────────────
# 実装から独立な family 定義（監査基準）
# ──────────────────────────────────────────────────────────────────────

def A_bern(eta):
    return np.logaddexp(0.0, eta)          # log(1 + e^eta)


def A_pois(eta):
    return np.exp(eta)


def A_gauss(eta):
    return 0.5 * eta ** 2                  # canonical: A(η)=η²/2（分散はφで扱う）


FAMILY_DEFS = {
    "bernoulli": (A_bern, lambda e: 1.0 / (1.0 + np.exp(-e)),
                  lambda e: (1.0 / (1.0 + np.exp(-e))) * (1.0 - 1.0 / (1.0 + np.exp(-e)))),
    "poisson":   (A_pois, np.exp, np.exp),
    "gaussian":  (A_gauss, lambda e: e, lambda e: np.ones_like(np.asarray(e, float))),
}


def col_loglik(x, eta, fam, sigma_sq=1.0):
    """1 列分の対数尤度（定数込みでなくてよい: 勾配検証用に x·η − A(η) 形）。

    Gaussian は dispersion σ² で割る（(xη − η²/2)/σ²、x²項は z に依存しない定数）。
    """
    A, _, _ = FAMILY_DEFS[fam]
    ll = x * eta - A(eta)
    if fam == "gaussian":
        ll = ll / sigma_sq
    return ll


def neg_log_posterior_i(z, i, X, Y, Z, F, sigma_diag, var_z, w0, w,
                        fam_list, family_y, sigma_y, mask_row):
    """独立実装: −ln f(z_i | X, Y)（z_i 以外は固定、z_i に依存しない定数は省略）。"""
    val = 0.5 * np.dot(z, z) / var_z                     # Z prior
    eta_x = F @ z
    for l, fam in enumerate(fam_list):                    # X 側: 列ごとの和
        val -= float(col_loglik(X[i, l], eta_x[l], fam, sigma_sq=sigma_diag[l]))
    Zmod = Z.copy()
    Zmod[i, :] = z
    eta_y = w0 + w * (Zmod @ z)                           # (n,)
    A_y, _, _ = FAMILY_DEFS[family_y]
    lly = Y[i, :] * eta_y - A_y(eta_y)                    # fixed 系列: Σ_{j≠i}, 1/2 なし
    if family_y == "gaussian":
        lly = lly / sigma_y ** 2
    val -= float(np.sum(lly * mask_row))
    return val


def numerical_gradient(fun, z, h=1e-6):
    g = np.zeros_like(z)
    for a in range(len(z)):
        zp, zm = z.copy(), z.copy()
        zp[a] += h
        zm[a] -= h
        g[a] = (fun(zp) - fun(zm)) / (2.0 * h)
    return g


def numerical_jacobian(gradfun, z, h=1e-6):
    k = len(z)
    J = np.zeros((k, k))
    for a in range(k):
        zp, zm = z.copy(), z.copy()
        zp[a] += h
        zm[a] -= h
        J[:, a] = (gradfun(zp) - gradfun(zm)) / (2.0 * h)
    return 0.5 * (J + J.T)


# ──────────────────────────────────────────────────────────────────────
# テストデータ（η が clip 域 [-20,10] に入らない小スケール）
# ──────────────────────────────────────────────────────────────────────

FAM_LIST = ["gaussian"] * 3 + ["bernoulli"] * 3 + ["poisson"] * 3
N, D, K = 15, 9, 2


def make_test_case(seed, family_y):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, K)) * 0.5
    F = rng.standard_normal((D, K)) * 0.5
    eta_x = Z @ F.T
    X = np.zeros((N, D))
    X[:, 0:3] = eta_x[:, 0:3] + rng.normal(0, 0.4, (N, 3))
    X[:, 3:6] = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta_x[:, 3:6])))
    X[:, 6:9] = rng.poisson(np.exp(eta_x[:, 6:9]))
    w0, w = 0.3, 0.2
    eta_y = w0 + w * (Z @ Z.T)
    if family_y == "poisson":
        Yu = rng.poisson(np.exp(eta_y))
    elif family_y == "bernoulli":
        Yu = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta_y)))
    else:
        Yu = eta_y + rng.normal(0, 0.5, (N, N))
    Y = np.triu(Yu, 1).astype(float)
    Y = Y + Y.T
    sigma_diag = np.ones(D)
    sigma_diag[0:3] = np.array([0.16, 0.25, 0.36])       # Gaussian 列のみ σ² ≠ 1
    return dict(Z=Z, F=F, X=X, Y=Y, sigma=np.diag(sigma_diag),
                sigma_diag=sigma_diag, var_z=1.0, w0=w0, w=w)


def build_model(family_y, train_mask=None, sigma_y=0.5):
    m = DualExpFamLSMPerColumn(n=N, d=D, k=K, L=3,
                               family_x_list=FAM_LIST, family_y=family_y,
                               sigma_y=sigma_y, train_mask=train_mask)
    m.initialize_params(seed=0)
    return m


# ──────────────────────────────────────────────────────────────────────
# A. family 基本形
# ──────────────────────────────────────────────────────────────────────

def audit_family_basics():
    eta = np.linspace(-3.0, 3.0, 25)
    h = 1e-6
    for fam, (A, Ap, App) in FAMILY_DEFS.items():
        d1 = np.max(np.abs((A(eta + h) - A(eta - h)) / (2 * h) - Ap(eta)))
        d2 = np.max(np.abs((Ap(eta + h) - Ap(eta - h)) / (2 * h) - App(eta)))
        record("A_family_basics", f"{fam}: A'(η) vs FD[A]", d1, 1e-6)
        record("A_family_basics", f"{fam}: A''(η) vs FD[A']", d2, 1e-6)


# ──────────────────────────────────────────────────────────────────────
# B. 勾配 / C. precision
# ──────────────────────────────────────────────────────────────────────

def audit_gradient_precision():
    rng = np.random.default_rng(123)
    for family_y in ("poisson", "bernoulli", "gaussian"):
        for use_mask in (False, True):
            if use_mask:
                M = rng.random((N, N)) < 0.7
                M = np.triu(M, 1)
                M = M | M.T
                train_mask = M
            else:
                train_mask = None
            t = make_test_case(seed=11, family_y=family_y)
            model = build_model(family_y, train_mask=train_mask, sigma_y=0.5)
            mask_row_all = model._mask_f
            tag = f"Y={family_y}, mask={'70%' if use_mask else 'none'}"

            gmax, pmax = 0.0, 0.0
            for i in (0, 4, 14):
                def f_obj(z, i=i):
                    return neg_log_posterior_i(
                        z, i, t["X"], t["Y"], t["Z"], t["F"], t["sigma_diag"],
                        t["var_z"], t["w0"], t["w"], FAM_LIST, family_y,
                        model.sigma_y, mask_row_all[i, :])

                def g_model(z, i=i):
                    Zm = t["Z"].copy()
                    Zm[i, :] = z
                    return model._calc_gradient(
                        t["X"], t["Y"], Zm, t["F"], t["sigma"],
                        t["var_z"], t["w0"], t["w"], i)

                z_i = t["Z"][i, :]
                g_num = numerical_gradient(f_obj, z_i)
                gmax = max(gmax, float(np.max(np.abs(g_model(z_i) - g_num))))

                P = model._calc_precision_matrix(
                    t["Z"], t["F"], t["sigma"], t["var_z"], t["w0"], t["w"], i)
                H_num = numerical_jacobian(g_model, z_i)
                pmax = max(pmax, float(np.max(np.abs(P - H_num))))

            record("B_gradient", tag + " (_calc_gradient vs FD[独立実装])",
                   gmax, 1e-5)
            record("C_precision", tag + " (_calc_precision vs FD[_calc_gradient])",
                   pmax, 1e-4)


# ──────────────────────────────────────────────────────────────────────
# D. 列和構造（Term2 = Σ_l 列ごと寄与）
# ──────────────────────────────────────────────────────────────────────

def audit_column_sum_structure():
    t = make_test_case(seed=11, family_y="poisson")
    model = build_model("poisson")
    gmax, pmax = 0.0, 0.0
    for i in (0, 4, 14):
        z_i = t["Z"][i, :]
        # モデル側 Term2 抽出: w=0 で Y 項を消し、prior 項を解析的に除去
        g_full = model._calc_gradient(t["X"], t["Y"], t["Z"], t["F"], t["sigma"],
                                      t["var_z"], t["w0"], 0.0, i)
        # -g_full = term1 + term2、term1 = -(1/var_z) z_i → term2 = -g_full + z_i/var_z
        term2_model = -g_full + (1.0 / t["var_z"]) * z_i      # ∇_z Σ_l ll_l
        P_full = model._calc_precision_matrix(t["Z"], t["F"], t["sigma"],
                                              t["var_z"], t["w0"], 0.0, i)
        prec2_model = P_full - (1.0 / t["var_z"]) * np.eye(K)

        # 独立側: 列ごとに g_l = w_l (x_l − A'(η_l)) f_l を素朴に加算
        g_sum = np.zeros(K)
        P_sum = np.zeros((K, K))
        eta_x = t["F"] @ z_i
        for l, fam in enumerate(FAM_LIST):
            _, Ap, App = FAMILY_DEFS[fam]
            w_l = 1.0 / t["sigma_diag"][l] if fam == "gaussian" else 1.0
            g_sum += w_l * (t["X"][i, l] - Ap(eta_x[l])) * t["F"][l, :]
            P_sum += w_l * App(eta_x[l]) * np.outer(t["F"][l, :], t["F"][l, :])
        gmax = max(gmax, float(np.max(np.abs(term2_model - g_sum))))
        pmax = max(pmax, float(np.max(np.abs(prec2_model - P_sum))))
    record("D_column_sum", "gradient_X(z_i) = Σ_l gradient_l(z_i)", gmax, 1e-10)
    record("D_column_sum", "precision_X(z_i) = Σ_l precision_l(z_i)", pmax, 1e-10)


# ──────────────────────────────────────────────────────────────────────
# E. 既存モデル整合（全列同一 family）
# ──────────────────────────────────────────────────────────────────────

def audit_uniform_equivalence():
    rng = np.random.default_rng(7)
    Z = rng.standard_normal((N, K)) * 0.5
    F = rng.standard_normal((D, K)) * 0.5
    Y = rng.poisson(2.0, (N, N)).astype(float)
    Y = np.triu(Y, 1)
    Y = Y + Y.T
    sigma = np.diag(np.linspace(0.2, 1.5, D))
    Zs = np.stack([Z] * 3, axis=2)

    for fam in ("gaussian", "bernoulli", "poisson"):
        if fam == "gaussian":
            Xv = Z @ F.T + rng.normal(0, 0.5, (N, D))
        elif fam == "bernoulli":
            Xv = rng.binomial(1, 0.5, (N, D)).astype(float)
        else:
            Xv = rng.poisson(2.0, (N, D)).astype(float)
        scalar = DualExpFamLSMMasked(n=N, d=D, k=K, L=3,
                                     family_x=fam, family_y="poisson")
        percol = DualExpFamLSMPerColumn(n=N, d=D, k=K, L=3,
                                        family_x_list=[fam] * D,
                                        family_y="poisson")
        for m in (scalar, percol):
            m.initialize_params(seed=1)
        gmax = pmax = 0.0
        for i in (0, 7, 14):
            g1 = scalar._calc_gradient(Xv, Y, Z, F, sigma, 1.0, 0.3, 0.2, i)
            g2 = percol._calc_gradient(Xv, Y, Z, F, sigma, 1.0, 0.3, 0.2, i)
            gmax = max(gmax, float(np.max(np.abs(g1 - g2))))
            p1 = scalar._calc_precision_matrix(Z, F, sigma, 1.0, 0.3, 0.2, i)
            p2 = percol._calc_precision_matrix(Z, F, sigma, 1.0, 0.3, 0.2, i)
            pmax = max(pmax, float(np.max(np.abs(p1 - p2))))
        record("E_uniform_equiv", f"all-{fam}: gradient ≡ scalar model", gmax, 1e-12)
        record("E_uniform_equiv", f"all-{fam}: precision ≡ scalar model", pmax, 1e-12)
        ll1 = scalar.calc_log_likelihood_X(Xv, Zs, F)
        ll2 = percol.calc_log_likelihood_X(Xv, Zs, F)
        record("E_uniform_equiv", f"all-{fam}: llX ≡ scalar model",
               abs(ll1 - ll2), 1e-6,
               extra=f"llX={ll1:.4f}")

    # _calc_F_adam_weighted（全列 bernoulli → 重み全て 1）≡ 親 _calc_F_adam
    Xb = rng.binomial(1, 0.5, (N, D)).astype(float)
    scalar = DualExpFamLSMMasked(n=N, d=D, k=K, L=3,
                                 family_x="bernoulli", family_y="poisson")
    percol = DualExpFamLSMPerColumn(n=N, d=D, k=K, L=3,
                                    family_x_list=["bernoulli"] * D,
                                    family_y="poisson")
    for m in (scalar, percol):
        m.initialize_params(seed=2)
    F1 = scalar._calc_F_adam(Xb, Zs)
    F2 = percol._calc_F_adam_weighted(Xb, Zs)
    record("E_uniform_equiv", "all-bernoulli: _calc_F_adam_weighted ≡ _calc_F_adam",
           float(np.max(np.abs(F1 - F2))), 1e-12)


# ──────────────────────────────────────────────────────────────────────
# F. 尤度 vs scipy
# ──────────────────────────────────────────────────────────────────────

def audit_loglik_vs_scipy():
    t = make_test_case(seed=11, family_y="poisson")
    model = build_model("poisson")
    model.params["sigma"] = t["sigma"]
    Zs = t["Z"][:, :, None]                     # L=1
    ll_model = model.calc_log_likelihood_X(t["X"], Zs, t["F"])

    eta_x = t["Z"] @ t["F"].T
    ll_ref = 0.0
    g_idx, b_idx, p_idx = slice(0, 3), slice(3, 6), slice(6, 9)
    ll_ref += float(np.sum(norm.logpdf(
        t["X"][:, g_idx], loc=eta_x[:, g_idx],
        scale=np.sqrt(t["sigma_diag"][g_idx])[None, :])))
    S = 1.0 / (1.0 + np.exp(-eta_x[:, b_idx]))
    ll_ref += float(np.sum(t["X"][:, b_idx] * np.log(S)
                           + (1 - t["X"][:, b_idx]) * np.log(1 - S)))
    ll_ref += float(np.sum(sp_poisson.logpmf(
        t["X"][:, p_idx].astype(int), np.exp(eta_x[:, p_idx]))))

    # モデルは Poisson 列の −ln(x!) を省略する規約 → 補正して比較
    fact = float(np.sum(gammaln(t["X"][:, p_idx] + 1)))
    record("F_loglik_scipy", "calc_log_likelihood_X（−ln(x!)補正後）vs scipy",
           abs((ll_model - fact) - ll_ref), 1e-6,
           extra=f"model={ll_model:.4f}, scipy={ll_ref:.4f}, corr={-fact:.4f}")


# ──────────────────────────────────────────────────────────────────────
# G. ブロック重み診断（情報のみ）
# ──────────────────────────────────────────────────────────────────────

def audit_block_weights():
    t = make_test_case(seed=11, family_y="poisson")
    model = build_model("poisson")
    model.params["sigma"] = t["sigma"]
    blocks = {"gaussian": slice(0, 3), "bernoulli": slice(3, 6),
              "poisson": slice(6, 9)}
    Zs = t["Z"][:, :, None]

    for bname, sl in blocks.items():
        # ブロック別 llX（該当列だけの per-column サブモデルで評価）
        sub = DualExpFamLSMPerColumn(n=N, d=3, k=K, L=1,
                                     family_x_list=[bname] * 3,
                                     family_y="poisson")
        sub.initialize_params(seed=0)
        sub.params["sigma"] = np.diag(t["sigma_diag"][sl])
        ll_b = sub.calc_log_likelihood_X(t["X"][:, sl], Zs, t["F"][sl, :])

        # ブロック別 E-step 勾配ノルム（全ノード平均）
        gn = []
        for i in range(N):
            eta_x = t["F"] @ t["Z"][i, :]
            g = np.zeros(K)
            for l in range(sl.start, sl.stop):
                fam = FAM_LIST[l]
                _, Ap, _ = FAMILY_DEFS[fam]
                w_l = 1.0 / t["sigma_diag"][l] if fam == "gaussian" else 1.0
                g += w_l * (t["X"][i, l] - Ap(eta_x[l])) * t["F"][l, :]
            gn.append(np.linalg.norm(g))
        record_info("G_block_weights", f"{bname} block (3 cols): llX", ll_b,
                    extra=f"mean|grad|={np.mean(gn):.4f}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== per-column family math audit ===")
    audit_family_basics()
    audit_gradient_precision()
    audit_column_sum_structure()
    audit_uniform_equivalence()
    audit_loglik_vs_scipy()
    audit_block_weights()

    df = pd.DataFrame(ROWS)
    df.insert(0, "datetime", datetime.now().isoformat(timespec="seconds"))
    out = OUT_DIR / "per_column_math_audit_summary.csv"
    df.to_csv(out, index=False)

    n_fail = int((df["result"] == "FAIL").sum())
    n_pass = int((df["result"] == "PASS").sum())
    print(f"\n=== {n_pass} PASS / {n_fail} FAIL "
          f"(+{int((df['result'] == 'INFO').sum())} INFO) → {out} ===")
    if n_fail:
        print(df[df["result"] == "FAIL"].to_string(index=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
