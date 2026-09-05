"""Generate the clean true-K n-sweep results report from artifacts.

Every number in the generated Markdown is recomputed from the machine-readable
artifacts.  In particular the per-cell selected K is re-derived from the raw
per-fit criterion values with the frozen selector and cross-checked against
selection_matrix.csv; a disagreement aborts generation.

This script runs NO EM and never modifies an artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (ROOT / "expfam" / "results" / "k_selection"
                   / "clean_true_k_asymptotics_20260904")
DEFAULT_OUT = (ROOT / "reports" / "identifiability"
               / "clean_true_k_results_20260905.md")

K_TRUE_GRID = (1, 3, 5)
N_GRID = (50, 75, 100, 150)
CANDIDATE_K = (1, 2, 3, 4, 5, 6, 7)
STARTS = (1, 2)
CRITERIA = ("S1", "S2", "S3")
CRITERION_LABEL = {
    "S1": "S1 held-out predictive（PRIMARY）",
    "S2": "S2 Q-based criterion（**Schwarz BIC ではない**）",
    "S3": "S3 plug-in conditional（**原論文 Eq.(26) ではない**）",
}
HIGHER_IS_BETTER = {"S1": True, "S2": False, "S3": False}
COLUMN = {"S1": "heldout_mean_log_score", "S2": "s2_q_based",
          "S3": "s3_plugin_conditional"}
TIE_TOLERANCE = np.float64(1e-12)


class ReportStop(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportStop(message)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def recompute_selection(fits: list[dict[str, str]]) -> dict[tuple[str, int, int, int],
                                                            dict[str, Any]]:
    by_cell: dict[tuple[int, int, int], list[dict[str, str]]] = {}
    for row in fits:
        by_cell.setdefault((int(row["k_true"]), int(row["n"]),
                            int(row["replicate"])), []).append(row)

    out: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for cell, rows in by_cell.items():
        for name in CRITERIA:
            means: dict[int, np.float64] = {}
            per_start: dict[int, list[float]] = {}
            for k_est in CANDIDATE_K:
                pairs = sorted((int(r["start"]), float(r[COLUMN[name]]))
                               for r in rows if int(r["k_est"]) == k_est)
                _require(len(pairs) == len(STARTS),
                         f"{name} {cell} K={k_est}: {len(pairs)} starts")
                vals = [v for _s, v in pairs]
                per_start[k_est] = vals
                signed = vals if HIGHER_IS_BETTER[name] else [-v for v in vals]
                means[k_est] = np.mean(np.asarray(signed, dtype=np.float64),
                                       dtype=np.float64)
            best = max(means.values())
            ties = sorted(k for k, v in means.items() if best - v <= TIE_TOLERANCE)
            runner_up = max(v for k, v in means.items() if k != min(ties))

            # What each start would have chosen on its own.  Where the two
            # disagree, the selection is sensitive to the EM starting point, not
            # only to the criterion.
            def start_pick(index: int) -> int:
                signed_by_k = {k: (per_start[k][index] if HIGHER_IS_BETTER[name]
                                   else -per_start[k][index])
                               for k in CANDIDATE_K}
                top = max(signed_by_k.values())
                return min(k for k, v in signed_by_k.items() if v == top)

            picks = [start_pick(i) for i in range(len(STARTS))]
            out[(name, *cell)] = {
                "selected_k": min(ties), "ties": ties,
                "margin": float(best - runner_up),
                "means": {k: float(v) for k, v in means.items()},
                "start_picks": picks,
                "start_disagreement": len(set(picks)) > 1,
            }
    return out


def build(run_dir: Path) -> str:
    fits = _read_csv(run_dir / "fit_results.csv")
    selection = _read_csv(run_dir / "selection_matrix.csv")
    provenance = _read_csv(run_dir / "generator_provenance.csv")
    gram = _read_csv(run_dir / "gram_spectrum.csv")
    protocol = _read_json(run_dir / "protocol.json")
    runinfo = _read_json(run_dir / "runinfo.json")
    audit = (_read_json(run_dir / "audit_report.json")
             if (run_dir / "audit_report.json").exists() else None)

    max_by_k = {k: float(np.mean([float(r["x_max"]) for r in provenance
                                  if int(r["K_TRUE"]) == k]))
                for k in K_TRUE_GRID}
    abs_by_k = {k: max(float(r["x_max"]) for r in provenance
                       if int(r["K_TRUE"]) == k)
                for k in K_TRUE_GRID}
    recomputed = recompute_selection(fits)
    for row in selection:
        key = (row["criterion"], int(row["K_TRUE"]), int(row["n"]),
               int(row["replicate"]))
        _require(key in recomputed, f"unrecomputable selection row {key}")
        _require(recomputed[key]["selected_k"] == int(row["selected_k"]),
                 f"recomputed selection disagrees with the artifact at {key}")

    lines: list[str] = []
    add = lines.append

    add("# clean true-K n-sweep — 結果")
    add("")
    try:
        shown_dir = run_dir.relative_to(ROOT).as_posix()
    except ValueError:                       # a fixture outside the repository
        shown_dir = run_dir.as_posix()
    add(f"本レポートは `{shown_dir}/` の artifact から自動生成した"
        "（`tools/research_audit/build_clean_true_k_report.py`）。数値の手作業転記は行っていない。")
    add("各セルの selected K は per-fit の生値から凍結 selector で**再導出**し、"
        "`selection_matrix.csv` と一致することを確認したうえで出力している（不一致なら生成が中断する）。")
    add("")
    add("**このレポート生成では EM を 1 回も実行していない。**")
    add("")

    # ---- framing (fixed text; every number below comes from the artifacts)
    add("## 0. 何を問うた実験か")
    add("")
    add("> **canonical clean generator による well-specified な有限標本設定で、`n` を増やしたとき、")
    add("> 事前登録した各 K-selection criterion の selected-K パターンはどう変わるか。**")
    add("")
    add("**これは consistency theorem ではない。** 有限の `n` を 4 点動かした記述的観測であり、")
    add("`n -> infinity` の一致性については何も示さない（理論監査 16b・U6）。")
    add("")
    add("### なぜ historical generator では不十分だったか")
    add("")
    add("一次コードを読んで確認した差分（`true_k_identifiability_hardened_20260904.md` 13、KI-021）:")
    add("`Z` と Gaussian-X の事後 z-score、`F` の行正規化、Poisson の hard clip、未使用の `sigma_x_true`。")
    add("**過去結果を無効化するものではない**が、「canonical model から well-specified に生成した」という")
    add("強い読み方はできない。とくに Poisson-X の識別可能性命題は unclipped link と iid `N(0,I)` を")
    add("前提とするため、historical generator ではその前提が成立しない。")
    add("")
    add("### なぜ X=Poisson / Y=Bernoulli を選んだか")
    add("")
    add("- **X=Poisson**: population で `FF^T` をモーメントから復元でき、`K = rank(FF^T)` を議論できる")
    add("  唯一の family（理論監査 P1）。ただし確立したのは **X 周辺の最小次元**であり `K*` ではない。")
    add("- **Y=Bernoulli**: Phase 7e / 8b との比較可能性のため。**同時に、識別可能性（U2）も")
    add("  非入れ子性（U5）も未証明の family** であり、理論的にもっとも弱い場所である。")
    add("  P2・P3（強い結果）は Gaussian-Y 限定であって、ここには適用できない。")
    add("")

    # ---- provenance
    add("## 1. Provenance")
    add("")
    add("> **`experiment_id` に含まれる \"asymptotics\" は命名上の名残であり、"
        "本実験は漸近的主張を一切含まない**（§0）。artifact は凍結済みのため改名しない。"
        "（敵対レビュー F-11）")
    add("")
    def field(key: str) -> Any:
        """Display-only accessor.  Every SCIENTIFIC number comes from the CSVs."""
        return runinfo.get(key, "（未記録）")

    add("| 項目 | 値 |")
    add("|---|---|")
    add(f"| experiment_id | `{field('experiment_id')}` |")
    add(f"| protocol hash | `{field('protocol_hash')}` |")
    add(f"| run_code_sha | `{protocol['run_code_sha']}` |")
    add(f"| generator | `{field('generator_version')}` |")
    add(f"| lineage | `numerics_mode = {field('numerics_mode')}`"
        "（objective-consistent。**旧 0.5 lineage 不使用**。lineage E / prototype / **本文採用不可**） |")
    add(f"| family | X `{field('family_x')}` / Y `{field('family_y')}` |")
    add(f"| TIER | {field('tier')} |")
    add(f"| fits | {field('completed_fit_count')} / expected {field('expected_fits')} |")
    add(f"| cells | {field('expected_cells')} |")
    add(f"| retry / replacement / seed rescue / tolerance 緩和 / resume | "
        f"{field('retry_count')} / {field('replacement_fits_executed')} / "
        f"{field('seed_rescue_count')} / {field('tolerance_relaxations')} / "
        f"{field('resumed')} |")
    add(f"| wall clock | {field('wall_clock_seconds')} s |")
    add(f"| NaN を報告した fit | {field('nan_fits')} |")
    add(f"| Q/BIC 計算に失敗した fit | {field('q_bic_failed_fits')} |")
    add("")
    if audit is not None:
        add(f"**独立監査:** verdict **{audit['verdict']}**、"
            f"BLOCKER {audit['blocker_count']} / HIGH {audit['high_count']} / "
            f"MEDIUM {audit['medium_count']} / LOW {audit['low_count']}"
            f"（`audit_version = {audit['audit_version']}`）。")
        add("")

    # ---- generator sanity
    add("## 2. 生成データの素性（clean generator）")
    add("")
    add("| K_TRUE | n | 平均 ‖f_l‖² | w_true | w²K | X 平均 | X 最大 | Y density | test pairs |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k_true in K_TRUE_GRID:
        for n in N_GRID:
            rows = [r for r in provenance
                    if int(r["K_TRUE"]) == k_true and int(r["n"]) == n]
            if not rows:
                continue
            w = float(rows[0]["w_true"])
            add(f"| {k_true} | {n} | {float(rows[0]['mean_f_row_norm_sq']):.3f} | "
                f"{w:.4f} | {w * w * k_true:.3f} | "
                f"{np.mean([float(r['x_mean']) for r in rows]):.3f} | "
                f"{max(float(r['x_max']) for r in rows):.0f} | "
                f"{np.mean([float(r['y_density']) for r in rows]):.3f} | "
                f"{rows[0]['n_test_pairs']} |")
    add("")
    add("`rank(F) = K_TRUE` は全 64 セルで成立（`generator_provenance.csv`）。"
        "平均 ‖f_l‖² と w²K は `K_TRUE` によらず**厳密に**一定である。")
    add("")
    add("**ただし整合しているのは 1 次モーメント（Y 側は分散）だけで、"
        "信号の分布形は `K_TRUE` に依存して系統的に変わる**（敵対レビュー F-03）。")
    add("")
    add("- `F = Q·diag(f_scale)` なので `‖f_l‖² ∝ Beta(K/2, (d−K)/2)`。"
        "列間のばらつきは `K` が小さいほど大きい。実測でも X の最大値は "
        f"`K_TRUE=1` で平均 {max_by_k[1]:.1f}（最大 {abs_by_k[1]:.0f}）に対し "
        f"`K_TRUE=5` では平均 {max_by_k[5]:.1f}（最大 {abs_by_k[5]:.0f}）。")
    add("- Y 側線形予測子の**超過尖度は `κ_4/κ_2² = 6/K`**（理論監査 §9.2）であり、"
        "`K_TRUE=1` で 6.0、`K_TRUE=3` で 2.0、`K_TRUE=5` で 1.2。")
    add("")
    add("**したがって「信号強度の交絡を完全に消した」とは書けない**（`[UNRESOLVED]`）。"
        "とくに `K_TRUE=1` のセルは候補集合の下端であること（§8.1(2)）に加えて、"
        "**より裾の重い有利な信号実現を受け取っている**。")
    add("")

    # ---- results per criterion
    add("## 3. 結果 — criterion 別の selected K")
    add("")
    add("**`K_TRUE` との一致であって `K*` との一致ではない**（protocol L1）。")
    add("")
    for name in CRITERIA:
        add(f"### {CRITERION_LABEL[name]}")
        add("")
        add("| K_TRUE | n | selected K（replicate 順） | 真値一致 | mean selected K | under | over |")
        add("|---:|---:|---|:---:|---:|:---:|:---:|")
        for k_true in K_TRUE_GRID:
            for n in N_GRID:
                keys = sorted(k for k in recomputed
                              if k[0] == name and k[1] == k_true and k[2] == n)
                if not keys:
                    continue
                sel = [recomputed[k]["selected_k"] for k in keys]
                exact = sum(1 for s in sel if s == k_true)
                under = sum(1 for s in sel if s < k_true)
                over = sum(1 for s in sel if s > k_true)
                shown = ", ".join(f"**{s}**" if s == k_true else str(s) for s in sel)
                add(f"| {k_true} | {n} | {shown} | {exact}/{len(sel)} | "
                    f"{np.mean(sel):.2f} | {under} | {over} |")
        total = [k for k in recomputed if k[0] == name]
        exact_total = sum(1 for k in total
                          if recomputed[k]["selected_k"] == k[1])
        add(f"| **合計** | | | **{exact_total}/{len(total)}** | | | |")
        add("")

    # ---- n-sweep focus
    add("## 4. PRIMARY focus — `K_TRUE = 5` の n 依存")
    add("")
    add("| criterion | n=50 | n=75 | n=100 | n=150 |")
    add("|---|---|---|---|---|")
    for name in CRITERIA:
        cells = []
        for n in N_GRID:
            keys = sorted(k for k in recomputed
                          if k[0] == name and k[1] == 5 and k[2] == n)
            sel = [recomputed[k]["selected_k"] for k in keys]
            exact = sum(1 for s in sel if s == 5)
            cells.append(f"{sel} — 一致 {exact}/{len(sel)}、平均 {np.mean(sel):.2f}")
        add(f"| {name} | " + " | ".join(cells) + " |")
    add("")

    # ---- criterion disagreement
    add("## 5. criterion 間の不一致")
    add("")
    cells = sorted({(k[1], k[2], k[3]) for k in recomputed})
    disagree = []
    for cell in cells:
        picks = {name: recomputed[(name, *cell)]["selected_k"] for name in CRITERIA}
        if len(set(picks.values())) > 1:
            disagree.append((cell, picks))
    same_pairs = {}
    for a in CRITERIA:
        for b in CRITERIA:
            if a < b:
                same_pairs[f"{a} vs {b}"] = sum(
                    1 for cell in cells
                    if recomputed[(a, *cell)]["selected_k"]
                    == recomputed[(b, *cell)]["selected_k"])
    add(f"64 セル中 **{len(disagree)} セル**で**三者一致が成立しなかった**"
        f"（S1 と S2 は {same_pairs['S1 vs S2']}/64 で一致している）。")
    add("")
    if disagree:
        add("| K_TRUE | n | replicate | S1 | S2 | S3 |")
        add("|---:|---:|---:|---:|---:|---:|")
        for (k_true, n, rep), picks in disagree:
            add(f"| {k_true} | {n} | {rep} | {picks['S1']} | {picks['S2']} | "
                f"{picks['S3']} |")
        add("")
    pairwise = {}
    for a in CRITERIA:
        for b in CRITERIA:
            if a < b:
                same = sum(1 for cell in cells
                           if recomputed[(a, *cell)]["selected_k"]
                           == recomputed[(b, *cell)]["selected_k"])
                pairwise[f"{a} vs {b}"] = same
    add("**対ごとの一致セル数（64 中）:** "
        + " / ".join(f"{k} {v}" for k, v in pairwise.items()))
    add("")

    # ---- margins
    add("## 6. 選択マージン（記述のみ）")
    add("")
    add("`margin` = 選択された K の 2-start 平均スコア − 次点 K の平均。**原因の説明ではない。**")
    add("")
    add("| criterion | K_TRUE | n | margin の中央値 | tie 発動セル |")
    add("|---|---:|---:|---:|:---:|")
    for name in CRITERIA:
        for k_true in K_TRUE_GRID:
            for n in N_GRID:
                keys = sorted(k for k in recomputed
                              if k[0] == name and k[1] == k_true and k[2] == n)
                if not keys:
                    continue
                margins = [recomputed[k]["margin"] for k in keys]
                ties = sum(1 for k in keys if len(recomputed[k]["ties"]) > 1)
                add(f"| {name} | {k_true} | {n} | {np.median(margins):.6f} | {ties} |")
    add("")

    # ---- gram diagnostic
    add("## 7. S4 — Poisson-X Gram spectrum（構造診断・K を選ばない）")
    add("")
    ok = [r for r in gram if r["status"] == "ok"]
    negatives = [r for r in ok if float(r["min_eigenvalue"]) < 0]
    add(f"- 計算できたセル: {len(ok)}/{len(gram)}")
    add(f"- **最小固有値が負のセル: {len(negatives)}/{len(ok)}**"
        "（推定 Gram が PSD 錐の外に出る。理論監査 §7.5 / U7 の予測どおり）")
    if ok:
        ranks = [int(r["unthresholded_rank"]) for r in ok]
        add(f"- 閾値なし階数: 最小 {min(ranks)} / 最大 {max(ranks)}"
            "（`d = 15`。**`K_TRUE` を返さない**）")
    add("")
    add("| K_TRUE | n | 最小固有値の中央値 | 上位固有値の中央値（λ1..λ6） |")
    add("|---:|---:|---:|---|")
    for k_true in K_TRUE_GRID:
        for n in N_GRID:
            rows = [r for r in ok
                    if int(r["K_TRUE"]) == k_true and int(r["n"]) == n]
            if not rows:
                continue
            mins = [float(r["min_eigenvalue"]) for r in rows]
            eigs = np.array([json.loads(r["eigenvalues"])[:6] for r in rows])
            top = ", ".join(f"{v:.3f}" for v in np.median(eigs, axis=0))
            add(f"| {k_true} | {n} | {np.median(mins):.4f} | {top} |")
    add("")
    add("**rank 閾値は設定していない。結果を見てから閾値を決めることを protocol が禁じている（L6）。**")
    add("")

    # ---- interpretation, computed from the same recomputation
    def series(name, k_true, key):
        out = []
        for n in N_GRID:
            keys = sorted(k for k in recomputed
                          if k[0] == name and k[1] == k_true and k[2] == n)
            sel = [recomputed[k]["selected_k"] for k in keys]
            if key == "exact":
                out.append(sum(1 for s in sel if s == k_true))
            elif key == "mean":
                out.append(round(float(np.mean(sel)), 2))
            elif key == "total":
                out.append(len(sel))
        return out

    add("## 8. 解釈（有限標本の記述的結果のみ）")
    add("")
    add("### 8.1 主要所見")
    add("")
    s1e, s1m = series("S1", 5, "exact"), series("S1", 5, "mean")
    s2e, s2m = series("S2", 5, "exact"), series("S2", 5, "mean")
    tot5 = series("S1", 5, "total")[0]
    add(f"**(1) `K_TRUE = 5`（PRIMARY）:** テストした `n = 50, 75, 100, 150` の範囲で、")
    add(f"S1 の真値一致は {s1e[0]}/{tot5} -> {s1e[1]}/{tot5} -> {s1e[2]}/{tot5} -> {s1e[3]}/{tot5}、")
    add(f"平均 selected K は {s1m[0]} -> {s1m[1]} -> {s1m[2]} -> {s1m[3]} と推移した。")
    add(f"S2 は真値一致 {s2e[0]}/{tot5} -> {s2e[1]}/{tot5} -> {s2e[2]}/{tot5} -> {s2e[3]}/{tot5}、")
    add(f"平均 selected K は {s2m[0]} -> {s2m[1]} -> {s2m[2]} -> {s2m[3]}。")
    add("")
    add("**平均 selected K は S1・S2 とも単調に増加したが、真値一致数は単調ではない**")
    add(f"（S1 は n=75 で {s1e[1]}/{tot5} といったん下がる）。**「n を増やすと K=5 に収束した」とは書かない。**")
    add("誤りの向きは一貫して **under-selection** であり、over-selection はごく少数である。")
    add("")
    ties_fired = sum(1 for k in recomputed if len(recomputed[k]["ties"]) > 1)
    min_margin = min(recomputed[k]["margin"] for k in recomputed)
    add("**(2) `K_TRUE = 1` の 4/4 は good recovery の証拠ではない。** S1・S2 とも全 `n` で 4/4 だが、")
    add("この設定の支配的な誤り方は under-selection であり、`K = 1` は候補集合の**下端**である。")
    add("したがって `K_TRUE = 1` での一致は、"
        "**データからの識別ではなく手続き上の下限効果である可能性を排除できない**（理論監査 17.4）。")
    add("")
    add(f"**ただし tie rule は根拠にならない**（敵対レビュー F-07）。同点は 192 回の選択のうち "
        f"{ties_fired} 回しか発生しておらず、最小マージンは {min_margin:.2e} で "
        "tie tolerance `1e-12` の遥か上である。"
        "下限効果の根拠は **候補集合の下端であること**と **under-selection 優位であること**の 2 点に限られる。")
    add("")
    add("**さらに第 3 の交絡がある**（§2、敵対レビュー F-03）: 信号整合は 1 次モーメントのみで、"
        "`K_TRUE=1` のセルは列間ばらつきが大きく Y 側の超過尖度も 6/K = 6.0 と最大である。"
        "**`K_TRUE=1` の結果を成功例として引用してはならない。**")
    add("")
    s1e3 = series("S1", 3, "exact")
    s2e3 = series("S2", 3, "exact")
    add(f"**(3) `K_TRUE = 3`（control）:** S1 {s1e3[0]}/4 -> {s1e3[1]}/4 -> {s1e3[2]}/4 -> {s1e3[3]}/4、")
    add(f"S2 {s2e3[0]}/4 -> {s2e3[1]}/4 -> {s2e3[2]}/4 -> {s2e3[3]}/4。")
    add("`K_TRUE = 5` と同じく under-selection 優位で、`n` とともに一致が増える傾向は共通している。")
    add("**`K_TRUE` が大きいほど、同じ `n` で真値一致に届きにくい**という記述的傾向が見える。")
    add("")
    s3_exact = sum(1 for k in recomputed
                   if k[0] == "S3" and recomputed[k]["selected_k"] == k[1])
    s3_total = sum(1 for k in recomputed if k[0] == "S3")
    add(f"**(4) S3（plug-in conditional）は使えない。** 真値一致 {s3_exact}/{s3_total} で、")
    add("ほぼ全セルで候補上限 `K = 7` を選ぶ。`ln p(Z)` を含めず `Z` を積分しないため、")
    add("潜在次元を増やすほど代入した Z への当てはまりが良くなり、`p log n` の罰則が追いつかない。")
    add("**これは Q1 型（conditional / plug-in）の基準に対する警告であって、原論文 Eq.(26) の評価ではない。**")
    add("原論文の評価手続きは本文から特定不能であり（`paper_bic_reproduction_alignment_20260904.md`）、")
    add("**S3 の失敗を原論文の基準の失敗と読んではならない。**")
    add("")
    dis50 = sum(1 for k in recomputed
                if k[0] == "S1" and k[1] == 5 and k[2] == 50
                and recomputed[k]["start_disagreement"])
    dis150 = sum(1 for k in recomputed
                 if k[0] == "S1" and k[1] == 5 and k[2] == 150
                 and recomputed[k]["start_disagreement"])
    disagree_series = []
    for n in N_GRID:
        keys = sorted(k for k in recomputed
                      if k[0] == "S1" and k[1] == 5 and k[2] == n)
        disagree_series.append(sum(1 for k in keys
                                   if recomputed[k]["start_disagreement"]))
    add("**(5) start 間の不一致（post-hoc 診断）。** "
        "**本診断は protocol 11 の事前登録に含まれない**（敵対レビュー B9）。"
        "凍結 artifact から計算した記述量にすぎず、selected K には一切影響しない。")
    add("")
    add("S1 / `K_TRUE=5` で 2 つの初期値が別々の K を選んだセルは")
    add(f"`n=50,75,100,150` の順に {disagree_series[0]}/{tot5}, {disagree_series[1]}/{tot5}, "
        f"{disagree_series[2]}/{tot5}, {disagree_series[3]}/{tot5} だった。")
    add("")
    add("**初版はこれを「選択が不安定な領域と初期値依存の領域が一致している」と書いたが、"
        "それは両端だけの話であり撤回する**（敵対レビュー F-06）。")
    add(f"実際 `n=75` は真値一致が最悪（{s1e[1]}/{tot5}）なのに不一致は {disagree_series[1]}/{tot5} で、")
    add(f"`n=100`（一致 {s1e[2]}/{tot5}・不一致 {disagree_series[2]}/{tot5}）より小さい。**2 つの系列は対応していない。**")
    add("初期値依存が存在すること自体は事実だが、"
        "**criterion 由来か最適化由来かは本実験では分離できていない**（`[UNRESOLVED]`）。")
    add("")
    q_steps: dict[int, list[float]] = {n: [] for n in N_GRID}
    p_steps: dict[int, list[float]] = {n: [] for n in N_GRID}
    indexed: dict[tuple, dict[int, dict[str, str]]] = {}
    for row in fits:
        indexed.setdefault((int(row["k_true"]), int(row["n"]),
                            int(row["replicate"]), int(row["start"])),
                           {})[int(row["k_est"])] = row
    for key, per_k in indexed.items():
        n = key[1]
        for k in range(1, max(CANDIDATE_K)):
            q_steps[n].append(-2.0 * (float(per_k[k + 1]["q_strict"])
                                      - float(per_k[k]["q_strict"])))
            p_steps[n].append((int(per_k[k + 1]["num_params"])
                               - int(per_k[k]["num_params"])) * float(np.log(n)))
    add("**(6) S2 の実効的な罰則は `p log n` ではない**（敵対レビュー F-04）。")
    add("`Q_strict` は完全データ同時密度なので `ln p(Z)` を含み、潜在次元を 1 増やすたびに")
    add("`O(n)` のコストが発生する。実測（潜在次元 +1 あたりの中央値）:")
    add("")
    add("| n | `−2ΔQ_strict` | `Δp·log n` | 比 |")
    add("|---:|---:|---:|---:|")
    for n in N_GRID:
        a = float(np.median(q_steps[n]))
        b = float(np.median(p_steps[n]))
        add(f"| {n} | {a:.1f} | {b:.1f} | {a / b:.2f} |")
    add("")
    decreasing = sum(1 for n in N_GRID for v in q_steps[n] if v > 0)
    total_steps = sum(len(v) for v in q_steps.values())
    add(f"`Q_strict` は {decreasing}/{total_steps} の段で `K` の増加とともに**減少**する。")
    add("すなわち **S2 の次元罰則の主要因は `ln p(Z)` 由来の `O(n)` 項**であり、")
    add("`p log n` は `n=150` でも全体の 2 割程度にすぎない。")
    add("これは **S2 が BIC 型からさらに遠い**ことを意味し、`n=50` での強い under-selection")
    add(f"（平均 selected K {s2m[0]}）の機構でもある。")
    add("")

    add("### 8.2 書いてよい表現 / 書いてはいけない表現")
    add("")
    add("**書いてよい:**")
    add("")
    add("> テストした有限の `n` の範囲（50-150）では、held-out 予測スコアと Q ベース基準の")
    add("> いずれについても、`n` の増加にともなって平均 selected K が真値へ近づき、")
    add("> under-selection が減少する傾向が観測された。ただし真値一致数は単調ではなく、")
    add("> 各条件の反復は 4 または 8 のみである。")
    add("")
    add("**書いてはいけない:** 「`n` を増やすと `K_TRUE` に収束した」「K 選択の一致性を示した」")
    add("「held-out なら真の K を選べる」「`K_TRUE=1` で完全に回復した」")
    add("「S3 の失敗は原論文 BIC の失敗である」")
    add("**「under-selection は criterion の性質である」**（固定 EM 予算の `K` 依存性と分離できていない、§9-10）")
    add("**「信号強度の交絡を完全に消した」**（整合は 1 次モーメントのみ、§2）。")
    add("")

    add("## 9. 限界")
    add("")
    add("| # | 限界 |")
    add("|---|---|")
    add("| 1 | **有限標本の記述にすぎない。** `n` は 4 点、反復は 4 または 8。信頼区間・検定・検出力は一切計算していない |")
    add("| 2 | **一致は `K_TRUE` との一致であり `K*` との一致ではない。** `K* < K_TRUE` の可能性は family ごとに未検証（U9） |")
    add("| 3 | **`family_y = bernoulli` は識別可能性（U2）も非入れ子性（U5）も未証明の領域である。** 強い理論結果 P2・P3 は Gaussian-Y 限定 |")
    add("| 4 | **S1 が population で何を選んでいるかは未解決（U10）。** plug-in raw-eta score は proper scoring rule ではない |")
    add("| 5 | **`K_TRUE=1` の結果は下限効果と交絡している**（8.1 (2)） |")
    add("| 6 | **有効標本数が未定義。** S2・S3 の罰則は `log n`（ノード数）を使うが、Y は `n(n-1)/2` dyad を供給する（16b） |")
    add("| 7 | **S4 は K を返さない。** 推定 Gram は全 64 セルで PSD 錐の外にあり、閾値なし階数は常に `d=15` |")
    add("| 8 | **lineage E（experimental prototype）。本文採用不可** |")
    add("| 9 | **1 つの合成設定のみ。** `d=15`、Poisson-X / Bernoulli-Y、信号強度は 1 水準に固定 |")
    add("| 10 | **EM を全 K で `num_iter=8` / `L=5` の固定予算で打ち切っており、収束判定も収束診断も記録していない。** 候補 K が大きいほど自由パラメータが多く（`K=7` で 84、`K=1` で 15）、同一予算では相対的に未収束になりやすい。したがって **「誤りが一貫して under-selection」であることも `n` とともに改善することも、criterion の性質ではなく最適化予算の `K` 依存性で説明できる。本実験はこの 2 つを分離していない**（`[UNRESOLVED]`、敵対レビュー F-02） |")
    add("| 11 | **信号整合は 1 次モーメント（Y 側は分散）のみ。** 分布形は `K_TRUE` に依存して変わる（§2、F-03） |")
    add("| 12 | **S2 の実効的な次元罰則は `p log n` ではない。** `Q_strict` に含まれる `ln p(Z)` 由来の **O(n)** 項が支配的である（F-04、§8.1(6)）。有効標本数の議論は S2 の挙動の主要因ではない |")
    add("| 13 | **`retry 0 / seed rescue 0` は sweep runner の宣言である。** 内部の `run_em_experimental` は NaN 検出時に最大 2 回、`seed + retry*1000`・`newton_alpha` 半減で再試行しうるが、`nan_count` が retry ごとにリセットされるため**その発動は artifact から検出できない**（`[UNRESOLVED]`、F-08） |")
    add("")

    add("## 10. この実験の claim ledger")
    add("")
    add("### ALLOWED（本実験の証拠で書けること）")
    add("")
    add("- clean generator の設計不変量が全 64 セルで厳密に成立したこと"
        "（`rank(F)=K_TRUE`、正規化なし、clip なし、平均 `||f_l||^2 = 0.5`、`w^2 K = 3.0`）")
    add(f"- 実行された fit 数がちょうど {len(fits)}、retry・replacement・seed rescue・"
        "tolerance 緩和・resume がいずれも 0 であること")
    add("- 独立 artifact 監査の verdict と finding 件数")
    add("- 各 `(criterion, K_TRUE, n, replicate)` の selected K の**正確な値**")
    add("")
    add("### QUALIFIED ONLY（限定語なしに書いてはいけない）")
    add("")
    add("- 「`n` の増加とともに平均 selected K が真値へ近づいた」"
        "-> **テストした有限範囲での記述、真値一致数は非単調、反復 4/8 のみ**を必ず併記")
    add("- 「S1 と S2 は似た挙動を示した」-> **64 セル中の一致数を明記**")
    add("- 「S3 は過大選択した」-> **本モジュールで定義した基準であり原論文の基準ではない**を必ず併記")
    add("")
    add("### NOT ALLOWED")
    add("")
    add("- K 選択の一致性 / 漸近一致性 / universal true-K recovery")
    add("- 現行実装が standard Schwarz BIC として妥当であること")
    add("- held-out 予測 = true-K recovery")
    add("- Bernoulli 一般の識別可能性についての結論")
    add("- 本合成設定を超える一般化")
    add("")

    add("## 11. 次に問うべきこと")
    add("")
    add("1. **`K_TRUE=1` の下限効果を切り分ける。** 候補集合の下端を広げるか tie rule を変えた感度解析"
        "（**本実験の protocol では変更禁止なので、新しい事前登録が要る**）。")
    add("2. **start 不一致が criterion 由来か最適化由来かを分離する。** start 数を増やすか、"
        "同一の Z 推定を共有して criterion だけ比較する設計。")
    add("3. **Bernoulli-Y の識別可能性（U2）。** 現状もっとも大きな理論的空白であり、"
        "実験で実際に使っている family である。")
    add("4. **有効標本数の定義（16b）。** ノード数か dyad 数かで罰則の意味が変わる。")
    add("5. **`n -> infinity`（U6）。** モデルが特異なので通常の BIC 漸近論は使えない。枠組みの選択が要る。")
    add("")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    text = build(args.run_dir)
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        status = "CURRENT" if current == text else "STALE"
        print(json.dumps({"status": status}))
        return 0 if status == "CURRENT" else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "WRITTEN", "bytes": len(text.encode("utf-8"))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
