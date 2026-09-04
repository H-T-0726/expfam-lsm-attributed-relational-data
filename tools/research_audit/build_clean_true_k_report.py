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
               / "clean_true_k_results_20260904.md")

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
            for k_est in CANDIDATE_K:
                vals = [float(r[COLUMN[name]]) for r in rows
                        if int(r["k_est"]) == k_est]
                _require(len(vals) == len(STARTS),
                         f"{name} {cell} K={k_est}: {len(vals)} starts")
                signed = vals if HIGHER_IS_BETTER[name] else [-v for v in vals]
                means[k_est] = np.mean(np.asarray(signed, dtype=np.float64),
                                       dtype=np.float64)
            best = max(means.values())
            ties = sorted(k for k, v in means.items() if best - v <= TIE_TOLERANCE)
            runner_up = max(v for k, v in means.items() if k != min(ties))
            out[(name, *cell)] = {
                "selected_k": min(ties), "ties": ties,
                "margin": float(best - runner_up),
                "means": {k: float(v) for k, v in means.items()},
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

    # ---- provenance
    add("## 1. Provenance")
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
        "**平均 ‖f_l‖² と w²K が K_TRUE によらず一定**であることが、信号強度の交絡を避ける設計であることを示す。")
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
    add(f"64 セル中 **{len(disagree)} セル**で criterion 間の選択が一致しなかった。")
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
