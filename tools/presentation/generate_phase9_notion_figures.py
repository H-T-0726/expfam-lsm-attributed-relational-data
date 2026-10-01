"""Generate the figures for the Phase 9 Notion presentation page.

Specification: reports/presentation/phase9_progress_notion_figure_spec_20261001.md
Output (fixed): reports/presentation/figures/phase9_notion_20261001/

Figures 1-3 are conceptual diagrams (no data). Figures 5-6 are drawn from the
committed primary artifacts, which are only read: no fit, no estimation, no new
statistic. The expected values asserted below are the counts already audited in
reports/presentation/phase9_progress_source_audit_20261001.md; if an artifact
disagrees with them the script stops instead of drawing.

Usage:
    python tools/presentation/generate_phase9_notion_figures.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib import font_manager                                 # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "presentation" / "figures" / "phase9_notion_20261001"

CONFUSION_LAP = (REPO / "expfam" / "results" / "matched_k_true_sensitivity"
                 / "phase9x_20260928" / "combined" / "confusion_Lap.csv")
CONFUSION_Q = (REPO / "expfam" / "results" / "matched_k_true_sensitivity"
               / "phase9x_20260928" / "combined" / "confusion_Q.csv")
K34_ROWS = (REPO / "expfam" / "results" / "k34_boundary_decomposition"
            / "phase9y_20260929" / "ktrue4_rows.csv")
JOINT_SELECTION = (REPO / "expfam" / "results" / "joint_family_k_selection"
                   / "gate75b_20260927" / "joint_selection.csv")

# Audited values (source audit / experiment inventory Table F, G, B).
EXPECTED_EXACT_LAP = {1: 10, 2: 10, 3: 10, 4: 8}
EXPECTED_EXACT_Q = {1: 10, 2: 9, 3: 7, 4: 1}
EXPECTED_TOTAL = 10
EXPECTED_K34_ROWS = 10
EXPECTED_CQ_THRESHOLD = 75 * (1 + math.log(2 * math.pi)) + 9 * math.log(75)  # 251.698...
EXPECTED_MARGIN_POSITIVE_Q = 1
EXPECTED_MARGIN_POSITIVE_LAP = 8
EXPECTED_JOINT_K_HAT = {"rep1": 2, "rep2": 3, "rep3": 3}

# Neutral two-series palette (validated: dataviz validate_palette.js, light, all PASS).
COLOR_LAP = "#2a78d6"
COLOR_Q = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d3"
BOX_FILL = "#f3f3f1"
BOX_EDGE = "#8a8984"
SCORE_FILL = "#e6eef9"   # light tint for the score-decided path only (not a series colour)


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------

def _setup_fonts() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Yu Gothic", "Meiryo", "Noto Sans JP", "BIZ UDGothic",
                 "MS Gothic", "IPAexGothic", "Hiragino Sans"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({
        "font.size": 12,
        "axes.unicode_minus": False,
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.hashsalt": "phase9-notion-20261001",
    })


def _save(fig, stem: str) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kwargs in (("png", {"dpi": 200}), ("svg", {})):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25,
                    metadata={"Date": None} if ext == "svg" else {"Software": None},
                    **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def _box(ax, x, y, w, h, text, *, fill=BOX_FILL, edge=BOX_EDGE, size=12,
         weight="normal", ls="-", color=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fill, ec=edge, lw=1.2, ls=ls))
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            fontweight=weight, color=color, linespacing=1.45)


def _arrow(ax, p, q, *, color=INK_2, lw=1.4, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


def _canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("auto")
    ax.axis("off")
    return fig, ax


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------
# Figure 1 — latent model (conceptual)
# --------------------------------------------------------------------------

def figure1() -> list[Path]:
    fig, ax = _canvas(9.5, 5.6, (0, 10), (0, 6))
    ax.text(5, 5.7, "共通潜在構造から属性と関係を生成", ha="center",
            fontsize=15, fontweight="bold", color=INK)

    for (cx, cy), name in (((2.0, 3.9), "z_i"), ((2.0, 1.5), "z_j")):
        ax.add_patch(Circle((cx, cy), 0.55, fc="white", ec=INK_2, lw=1.6))
        ax.text(cx, cy, f"${name}$", ha="center", va="center", fontsize=17,
                color=INK)
    ax.text(2.0, 0.45, r"$z_i \in \mathbb{R}^K$：連続潜在ベクトル" + "\n"
            r"$z_i \sim \mathcal{N}(0, I_K)$",
            ha="center", va="center", fontsize=11, color=INK_2,
            linespacing=1.5)

    _box(ax, 7.6, 4.1, 3.6, 1.6,
         "属性\n" + r"$x_{i1},\ \ldots,\ x_{id}$" + "\n"
         + r"$x_{il} \sim \mathrm{ExpFam}_X(\eta^X_{il})$", size=12)
    _box(ax, 7.6, 1.5, 3.6, 1.3,
         "関係\n" + r"$y_{ij} \sim \mathrm{ExpFam}_Y(\eta^Y_{ij})$", size=12)

    _arrow(ax, (2.6, 4.0), (5.75, 4.1))
    ax.text(4.15, 4.35, r"$\eta^X_{il} = f_l^{\top} z_i$", ha="center",
            fontsize=13, color=INK)
    _arrow(ax, (2.5, 3.55), (5.75, 1.75), rad=-0.05)
    _arrow(ax, (2.6, 1.5), (5.75, 1.45))
    ax.text(4.15, 1.0, r"$\eta^Y_{ij} = w_0 + w\, z_i^{\top} z_j$",
            ha="center", va="center", fontsize=13, color=INK)

    ax.text(7.6, 0.35, "X 側に切片はない／$w_0,\\ w$ はスカラー",
            ha="center", fontsize=10.5, color=INK_2)
    return _save(fig, "figure1_latent_model")


# --------------------------------------------------------------------------
# Figure 2 — family-selection flow (conceptual)
# --------------------------------------------------------------------------

def figure2() -> list[Path]:
    fig, ax = _canvas(10.5, 12.6, (0, 11), (-1.0, 13))
    ax.text(5, 12.75, "属性分布族の選択の流れ（1 つの属性列 l）", ha="center",
            fontsize=15, fontweight="bold", color=INK)

    _box(ax, 5, 12.0, 3.4, 0.7, r"属性列 $x_{\cdot l}$")
    _arrow(ax, (5, 11.63), (5, 11.17))
    _box(ax, 5, 10.8, 5.2, 0.75, "Support gate（尤度は使わない）",
         weight="bold")

    branch_y = 9.35
    specs = [
        (1.9, "A：0/1 のみ\n候補 Bernoulli / Poisson\n→ score で比較", SCORE_FILL),
        (6.4, "B：2 以上を含む\n非負整数\n→ Poisson（gate で決定）", BOX_FILL),
        (9.4, "C：その他\n（非整数・負値を含む）\n→ Gaussian（gate で決定）", BOX_FILL),
    ]
    for x, text, fill in specs:
        _arrow(ax, (5, 10.42), (x, branch_y + 0.62), rad=0.0)
        _box(ax, x, branch_y, 2.9, 1.15, text, fill=fill, size=11)

    # exploration MCEM frame (branch A only)
    ax.add_patch(FancyBboxPatch((0.25, 3.95), 5.3, 4.55,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="none", ec=BOX_EDGE, lw=1.2, ls="--"))
    ax.text(2.2, 8.25, "探索 MCEM（8 反復）", fontsize=11.5,
            fontweight="bold", color=INK_2, va="center")
    _arrow(ax, (1.9, branch_y - 0.6), (1.9, 7.85))
    _box(ax, 2.7, 7.45, 4.4, 0.75,
         r"E-step：$Z^{(1)}, \ldots, Z^{(L)}$（$L=5$）を固定", size=11)
    _arrow(ax, (1.6, 7.06), (1.45, 6.62))
    _arrow(ax, (3.8, 7.06), (3.95, 6.62))
    _box(ax, 1.45, 6.05, 2.2, 1.05,
         "Bernoulli\n" + r"$f_l^{(B)}$ を最適化" + "\n" + r"$Q_l(B)$",
         fill=SCORE_FILL, size=10.5)
    _box(ax, 3.95, 6.05, 2.2, 1.05,
         "Poisson\n" + r"$f_l^{(P)}$ を最適化" + "\n" + r"$Q_l(P)$",
         fill=SCORE_FILL, size=10.5)
    _arrow(ax, (1.45, 5.51), (2.3, 5.07))
    _arrow(ax, (3.95, 5.51), (3.1, 5.07))
    _box(ax, 2.7, 4.65, 4.4, 0.75,
         r"score の大きい分布族：$\hat c_l = \arg\max_m Q_l(m)$", size=10.5)
    _arrow(ax, (5.25, 4.65), (5.25, 7.45), rad=0.0, ls=(0, (3, 2)))
    ax.text(5.65, 6.05, "次の\n反復", fontsize=9.5, color=INK_2,
            va="center", ha="left")

    # merge
    _box(ax, 5, 2.85, 6.0, 0.75, "全列の分布族の割り当てを固定", weight="bold")
    _arrow(ax, (2.7, 4.26), (3.6, 3.24))
    _arrow(ax, (6.4, branch_y - 0.6), (6.0, 3.24), rad=0.0)
    _arrow(ax, (9.4, branch_y - 0.6), (7.4, 3.24), rad=0.0)
    _arrow(ax, (5, 2.47), (5, 2.03))
    _box(ax, 5, 1.65, 6.0, 0.75, "fresh refit（固定した割り当てで MCEM を 8 反復）")
    _arrow(ax, (5, 1.27), (5, 0.83))
    _box(ax, 5, 0.45, 6.0, 0.72, "最終結果（報告するのはこの値）", weight="bold")

    ax.text(5.0, -0.45,
            "score で選ぶのは 0/1 列だけ ／ gate では尤度を使わない\n"
            "loading は候補ごとに別々に最適化 ／ 報告値は fresh refit の結果",
            fontsize=10.5, color=INK_2, va="center", ha="center",
            linespacing=1.7,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=GRID))
    return _save(fig, "figure2_family_selection_flow")


# --------------------------------------------------------------------------
# Figure 3 — joint family + K pipeline (conceptual)
# --------------------------------------------------------------------------

def _joint_k_hat() -> dict[str, int]:
    rows = _read_csv(JOINT_SELECTION)
    k_hat = {r["replicate"]: int(r["k_hat"]) for r in rows
             if r["start_label"] == "start_B"}
    assert k_hat == EXPECTED_JOINT_K_HAT, f"joint K_hat {k_hat} != audited"
    return k_hat


def figure3() -> list[Path]:
    k_hat = _joint_k_hat()
    fig, ax = _canvas(13.0, 6.6, (0, 13.6), (0, 7.4))
    ax.text(6.8, 7.15, "分布族 + K の同時選択", ha="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(6.8, 6.65,
            "K ごとに分布族の選択を最初から実行（K 間の warm start・割り当ての持ち越しなし）",
            ha="center", fontsize=11, color=INK_2)

    _box(ax, 0.95, 3.4, 1.5, 1.2, "観測\nX, Y", weight="bold")
    steps = ["support\ngate", "分布族の\n選択", "割り当て\nを固定", "fresh\nrefit"]
    xs = [2.95, 4.65, 6.35, 8.05]
    ys = [5.7, 4.55, 3.4, 2.25, 1.1]
    for k, y in zip(range(1, 6), ys):
        _arrow(ax, (1.72, 3.4), (2.2, y), lw=1.0)
        ax.text(2.3, y + 0.42, f"K = {k}", fontsize=10.5, color=INK,
                fontweight="bold", va="bottom")
        for i, (x, s) in enumerate(zip(xs, steps)):
            _box(ax, x, y, 1.35, 0.78, s, size=9.5)
            if i:
                _arrow(ax, (xs[i - 1] + 0.7, y), (x - 0.7, y), lw=1.0)
        _arrow(ax, (xs[-1] + 0.7, y), (9.3, y), lw=1.0)
        _box(ax, 9.75, y, 0.9, 0.62, f"$C_Q({k})$", fill="white", size=11)
        _arrow(ax, (10.22, y), (10.95, 3.4), lw=1.0)
    _box(ax, 12.2, 3.4, 2.4, 1.15,
         r"$\hat K = \arg\min_K C_Q(K)$", fill="white", size=11.5,
         weight="bold")

    ax.text(12.2, 1.25,
            "実行例（3 dataset、真の K = 3）\n"
            + "選ばれた K：" + ", ".join(str(k_hat[r]) for r in ("rep1", "rep2", "rep3")),
            ha="center", va="center", fontsize=9.5, color=INK_2,
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))
    return _save(fig, "figure3_joint_family_k_pipeline")


# --------------------------------------------------------------------------
# Figure 5 — K_true vs exact count (data)
# --------------------------------------------------------------------------

def _exact_counts(path: Path) -> dict[int, int]:
    counts = {}
    for row in _read_csv(path):
        k_true = int(row["K_true"])
        total = int(row["total"])
        assert total == EXPECTED_TOTAL, f"{path.name}: total {total}"
        counts[k_true] = int(row[f"K_hat_{k_true}"])
    return counts


def figure5() -> list[Path]:
    lap = _exact_counts(CONFUSION_LAP)
    q = _exact_counts(CONFUSION_Q)
    assert lap == EXPECTED_EXACT_LAP, f"C_Lap counts {lap} != audited"
    assert q == EXPECTED_EXACT_Q, f"C_Q counts {q} != audited"

    k = sorted(lap)
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.plot(k, [lap[i] for i in k], color=COLOR_LAP, lw=2, marker="o",
            ms=9, mec="white", mew=2, label="C_Lap", zorder=3)
    ax.plot(k, [q[i] for i in k], color=COLOR_Q, lw=2, ls="--", marker="s",
            ms=9, mec="white", mew=2, label="C_Q", zorder=3)
    for i in k:
        ax.annotate(f"{lap[i]}/{EXPECTED_TOTAL}", (i, lap[i]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=10.5, color=INK)
        ax.annotate(f"{q[i]}/{EXPECTED_TOTAL}", (i, q[i]),
                    textcoords="offset points", xytext=(-16, -18), ha="center",
                    fontsize=10.5, color=INK_2)

    ax.set_xticks(k)
    ax.set_xticklabels([f"{i}" for i in k])
    ax.set_xlim(0.6, 4.4)
    ax.set_ylim(-0.3, 11.2)
    ax.set_yticks(range(0, 11, 2))
    ax.set_xlabel("真の潜在次元 K_true", color=INK)
    ax.set_ylabel("真の K を選んだ dataset 数（各 10 dataset 中）", color=INK)
    ax.set_title("真の潜在次元を変えたときの正解数（信号の平均的な強さを揃えた設計）",
                 fontsize=12.5, color=INK, pad=12)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.legend(loc="lower left", frameon=False, fontsize=11)
    fig.text(0.02, -0.06,
             "各条件 10 dataset の固定した人工データ設計。K_true = 3 は baseline 20 dataset のうち rep01..rep10。\n"
             "揃えた量：平均 X loading エネルギー 0.5、$w^2K = 3$、母集団の平均 edge 確率 0.3314。"
             "一般の回復確率を示すものではない。",
             fontsize=9.5, color=INK_2, ha="left", va="top", linespacing=1.6)
    return _save(fig, "figure5_ktrue_exact_counts")


# --------------------------------------------------------------------------
# Figure 6 — K3/K4 threshold balance (data)
# --------------------------------------------------------------------------

def _k34_rows() -> list[dict[str, float | str]]:
    rows = _read_csv(K34_ROWS)
    assert len(rows) == EXPECTED_K34_ROWS, f"{len(rows)} rows != audited"
    out = []
    for r in rows:
        out.append({
            "rep": r["replicate"],
            "fit_q": float(r["fit_gain_Q_34"]),
            "fit_lap": float(r["fit_gain_Lap_34"]),
            "thr_q": float(r["P_Z_increment_34"]) + float(r["param_increment_Q"]),
            "pz": float(r["P_Z_increment_34"]),
            "param_q": float(r["param_increment_Q"]),
            "thr_lap": float(r["volume_increment_34"]) + float(r["param_increment_Lap"]),
            "margin_q": float(r["margin_Q"]),
            "margin_lap": float(r["margin_Lap"]),
        })
    return out


def figure6() -> list[Path]:
    rows = _k34_rows()
    for r in rows:
        assert abs(r["thr_q"] - EXPECTED_CQ_THRESHOLD) < 1e-9, r["thr_q"]
        # the drawn comparison (point vs threshold) must agree with the committed margin
        assert (r["fit_q"] > r["thr_q"]) == (r["margin_q"] > 0), r["rep"]
        assert (r["fit_lap"] > r["thr_lap"]) == (r["margin_lap"] > 0), r["rep"]
    pos_q = sum(r["margin_q"] > 0 for r in rows)
    pos_lap = sum(r["margin_lap"] > 0 for r in rows)
    assert pos_q == EXPECTED_MARGIN_POSITIVE_Q, pos_q
    assert pos_lap == EXPECTED_MARGIN_POSITIVE_LAP, pos_lap

    labels = [r["rep"] for r in rows]
    x = list(range(1, len(rows) + 1))
    thr_q = rows[0]["thr_q"]
    pz, param = rows[0]["pz"], rows[0]["param_q"]
    n = len(rows)

    fig, (ax_q, ax_l) = plt.subplots(1, 2, figsize=(12.0, 5.8), sharey=True)
    for ax in (ax_q, ax_l):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9.5)
        ax.set_xlim(0.4, n + 0.6)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
        ax.tick_params(colors=INK_2)

    ax_q.axhline(thr_q, color=INK_2, lw=1.5, ls="--", zorder=1)
    ax_q.text(n + 0.5, thr_q + 4,
              f"閾値 {thr_q:.2f}\n（P_Z の増分 {pz:.2f} + パラメータ罰則 {param:.2f}、一定）",
              ha="right", va="bottom", fontsize=9.5, color=INK_2, linespacing=1.4)
    ax_q.scatter(x, [r["fit_q"] for r in rows], s=70, marker="s",
                 color=COLOR_Q, edgecolor="white", linewidth=1.5, zorder=3,
                 label="当てはまりの改善（事後サンプル平均）")
    ax_q.set_title(f"C_Q：閾値を上回った dataset {pos_q}/{n}", fontsize=12,
                   color=INK)
    ax_q.set_ylabel("−2 × 対数尤度の単位", color=INK)

    for xi, r in zip(x, rows):
        ax_l.plot([xi - 0.32, xi + 0.32], [r["thr_lap"]] * 2, color=INK_2,
                  lw=2.2, solid_capstyle="round", zorder=2)
    ax_l.plot([], [], color=INK_2, lw=2.2,
              label="dataset ごとの閾値（体積の増分 + パラメータ罰則）")
    ax_l.scatter(x, [r["fit_lap"] for r in rows], s=75, marker="o",
                 color=COLOR_LAP, edgecolor="white", linewidth=1.5, zorder=3,
                 label="当てはまりの改善（同時最頻値）")
    ax_l.set_title(f"C_Lap：閾値を上回った dataset {pos_lap}/{n}",
                   fontsize=12, color=INK)

    for ax in (ax_q, ax_l):
        ax.legend(loc="upper left", frameon=False, fontsize=9.5)

    fig.suptitle("K = 3 と K = 4 の境界（真の K = 4 の 10 dataset、保存済みの値の分解）",
                 fontsize=13, color=INK, y=1.01)
    fig.text(0.02, -0.04,
             "点が閾値より上なら K = 4 が K = 3 に勝つ。左右で当てはまりの改善の定義が違う"
             "（左：事後サンプル平均、右：同時最頻値）ので、左右の点を直接比べない。\n"
             "算術的な分解であり、過小選択の原因を示すものではない。固定した人工データ条件。",
             fontsize=9.5, color=INK_2, ha="left", va="top", linespacing=1.6)
    fig.tight_layout()
    return _save(fig, "figure6_k34_threshold_balance")


def main() -> int:
    _setup_fonts()
    written = []
    for build in (figure1, figure2, figure3, figure5, figure6):
        written.extend(build())
    for path in written:
        print(path.relative_to(REPO).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
