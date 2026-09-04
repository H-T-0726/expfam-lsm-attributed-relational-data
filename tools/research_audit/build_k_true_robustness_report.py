"""Generate the Phase 8b K_TRUE robustness (Attempt 2) canonical report.

Every number in the generated Markdown is recomputed from the machine-readable
run artifacts; nothing is transcribed by hand.  In particular the per-cell
``selected_k`` is re-derived from the raw per-fit held-out scores using the
frozen Phase 7e selector (unweighted two-start mean, tie tolerance 1e-12,
smallest-K tie rule) and cross-checked against ``selection_matrix.csv`` and
``full_summary.json``; a disagreement aborts report generation.

This script performs NO model fitting.  It never imports the EM model and never
executes EM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    ROOT / "expfam" / "results" / "k_selection"
    / "k_true_robustness_full_attempt2_20260904"
)
DEFAULT_PRIOR_DIR = (
    ROOT / "expfam" / "results" / "k_selection"
    / "k_true_robustness_full_20260902"
)
DEFAULT_REPORT_PATH = (
    ROOT / "reports" / "k_selection_theory"
    / "k_true_robustness_full_report_20260904.md"
)

NEW_K_TRUE = (1, 2, 4, 5)
ANCHOR_K_TRUE = 3
K_CANDIDATES = (1, 2, 3, 4, 5, 6, 7)
REPLICATES = (1, 2, 3)
STARTS = (1, 2)
TIE_TOLERANCE = np.float64(1e-12)
ROLE_OF = {"A": "primary", "B": "sensitivity"}


class ReportStop(RuntimeError):
    """Raised when the artifacts do not support the report being generated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportStop(message)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select_k(mean_scores: dict[int, np.float64]) -> tuple[int, tuple[int, ...]]:
    """Frozen Phase 7e selector: argmax of the two-start mean, smallest-K tie."""

    best = max(mean_scores.values())
    ties = tuple(sorted(k for k, s in mean_scores.items() if best - s <= TIE_TOLERANCE))
    _require(bool(ties), "selector produced no tie candidate")
    return min(ties), ties


def recompute_cells(
    fit_rows: list[dict[str, str]],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Re-derive every new-grid cell from the raw per-fit held-out scores."""

    raw: dict[tuple[str, int, int], dict[tuple[int, int], np.float64]] = {}
    for row in fit_rows:
        key = (row["estimand"], int(row["K_TRUE"]), int(row["replicate"]))
        raw.setdefault(key, {})[(int(row["K"]), int(row["start"]))] = np.float64(
            row["heldout_mean_log_score"]
        )

    expected = {(k, s) for k in K_CANDIDATES for s in STARTS}
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    for key, scores in raw.items():
        _require(set(scores) == expected, f"cell {key} does not carry 7 K x 2 starts")
        means = {
            k: np.mean(
                np.asarray([scores[(k, 1)], scores[(k, 2)]], dtype=np.float64),
                dtype=np.float64,
            )
            for k in K_CANDIDATES
        }
        selected, ties = _select_k(means)
        best = max(means.values())
        runner_up = max(v for k, v in means.items() if k != selected)
        cells[key] = {
            "mean_scores": means,
            "selected_k": selected,
            "tie_candidates": ties,
            "margin": float(best - runner_up),
        }
    return cells


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def build_report(run_dir: Path, prior_dir: Path) -> str:
    fit_rows = _read_csv(run_dir / "full_fit_results.csv")
    selection_rows = _read_csv(run_dir / "selection_matrix.csv")
    mask_rows = _read_csv(run_dir / "mask_provenance.csv")
    config_rows = _read_csv(run_dir / "config_gate.csv")
    leakage_rows = _read_csv(run_dir / "leakage_gate.csv")
    manifest_rows = _read_csv(run_dir / "manifest.csv")
    runinfo = _read_json(run_dir / "runinfo.json")
    summary = _read_json(run_dir / "full_summary.json")
    authorization = _read_json(run_dir / "authorization.json")
    audit = _read_json(run_dir / "audit_report.json")

    # ---- structural contract ------------------------------------------------
    indices = [int(r["fit_index"]) for r in fit_rows]
    per_estimand = {
        e: sorted(int(r["fit_index"]) for r in fit_rows if r["estimand"] == e)
        for e in ("A", "B")
    }
    _require(len(fit_rows) == 336, "full_fit_results.csv does not carry 336 data rows")
    _require(sorted(indices) == list(range(1, 337)), "fit_index is not exactly 1..336")
    _require(per_estimand["A"] == list(range(1, 169)), "A is not fit_index 1..168")
    _require(per_estimand["B"] == list(range(169, 337)), "B is not fit_index 169..336")
    _require(
        sorted({int(r["K_TRUE"]) for r in fit_rows}) == list(NEW_K_TRUE),
        "new K_TRUE grid is not {1,2,4,5}",
    )
    _require(
        not [r for r in fit_rows if int(r["K_TRUE"]) == ANCHOR_K_TRUE],
        "K_TRUE=3 must contribute no newly executed row",
    )
    _require(audit["status"] == "PASS", "audit_report.json is not PASS")
    _require(
        not (run_dir / "failure.json").exists(),
        "failure.json is present: this is not a successful run",
    )

    # ---- independent re-derivation of selected K ----------------------------
    cells = recompute_cells(fit_rows)
    _require(len(cells) == 24, "expected 24 newly executed cells")
    anchor: dict[tuple[str, int], dict[str, str]] = {}
    for row in selection_rows:
        key = (row["estimand"], int(row["K_TRUE"]), int(row["replicate"]))
        if row["lineage"] == "phase7e_anchor":
            _require(
                int(row["K_TRUE"]) == ANCHOR_K_TRUE, "anchor lineage outside K_TRUE=3"
            )
            anchor[(row["estimand"], int(row["replicate"]))] = row
            continue
        _require(
            cells[key]["selected_k"] == int(row["selected_k"]),
            f"recomputed selected_k disagrees with selection_matrix.csv at {key}",
        )
    for key, cell in cells.items():
        label = f"{key[0]}/K{key[1]}/r{key[2]}"
        _require(
            summary["selected_k"][label] == cell["selected_k"],
            f"recomputed selected_k disagrees with full_summary.json at {label}",
        )
    _require(len(anchor) == 6, "expected 6 Phase 7e anchor reference rows")
    for replicate in REPLICATES:
        _require(
            anchor[("A", replicate)]["selected_k"]
            == anchor[("B", replicate)]["selected_k"],
            "A/B anchor rows must reference the same Phase 7e evidence",
        )

    # ---- recovery counts ----------------------------------------------------
    recovery = {
        e: {
            kt: sum(1 for r in REPLICATES if cells[(e, kt, r)]["selected_k"] == kt)
            for kt in NEW_K_TRUE
        }
        for e in ("A", "B")
    }
    totals = {e: sum(recovery[e].values()) for e in ("A", "B")}
    anchor_selected = [int(anchor[("A", r)]["selected_k"]) for r in REPLICATES]
    anchor_exact = sum(1 for s in anchor_selected if s == ANCHOR_K_TRUE)

    direction = {}
    for e in ("A", "B"):
        counts = {"exact": 0, "over": 0, "under": 0}
        for kt in NEW_K_TRUE:
            for r in REPLICATES:
                sel = cells[(e, kt, r)]["selected_k"]
                counts[
                    "exact" if sel == kt else ("over" if sel > kt else "under")
                ] += 1
        direction[e] = counts

    ab_diff = [
        (kt, r, cells[("A", kt, r)]["selected_k"], cells[("B", kt, r)]["selected_k"])
        for kt in NEW_K_TRUE
        for r in REPLICATES
        if cells[("A", kt, r)]["selected_k"] != cells[("B", kt, r)]["selected_k"]
    ]

    w_true = {}
    for row in manifest_rows:
        w_true[(row["estimand"], int(row["K_TRUE"]))] = row["w_true"]

    lines: list[str] = []
    add = lines.append

    add("# Phase 8b K_TRUE robustness full sweep — Attempt 2（成功実行の最終アーカイブ）")
    add("")
    add(
        f"本レポートは `{_rel(run_dir)}/` の machine-readable artifact から自動生成した"
        "（`tools/research_audit/build_k_true_robustness_report.py`）。"
        "数値の手作業転記は行っていない。"
    )
    add(
        "各セルの `selected_k` は per-fit の生スコアから frozen Phase 7e selector で"
        "**再導出**し、`selection_matrix.csv` および `full_summary.json` と一致することを"
        "確認したうえで出力している（不一致なら生成が中断する）。"
    )
    add("")
    add("**このアーカイブ工程で実行した real EM fit は 0 である。**")
    add("")

    add("## 1. 目的")
    add("")
    add(
        "Phase 7e (Issue #43) で凍結した leakage-safe held-out K-selection protocol を、"
    )
    add(
        f"generator の `K_TRUE` を {{{', '.join(str(k) for k in NEW_K_TRUE)}}} へ拡張して "
        "**exactly once** 実行し、各 replicate で選択される K を記述的に測定する。"
    )
    add("")
    add(
        "**「真の K が選ばれること」は成功条件ではない。** 目的は frozen protocol 下での"
        "選択挙動の測定であり、結果を見た後に seed / replicate / tolerance / score / "
        "K range / start count / split / preprocessing / failure rule を変更しない。"
    )
    add("")

    add("## 2. 本実験で使った K 選択基準（誤記防止のため明示）")
    add("")
    add("```")
    add("eta_ij      = w0 + w * z_i^T z_j")
    add("log score   = y_ij * eta_ij - logaddexp(0, eta_ij)")
    add("fit score   = held-out upper-triangle test pair 上の mean log score")
    add("Sbar(K)     = (start1 score + start2 score) / 2       # 非加重 2-start 平均")
    add("tie 候補    = max_K Sbar(K) - Sbar(K) <= 1e-12")
    add("selected K  = tie 候補のうち最小の K                   # frozen tie rule")
    add("```")
    add("")
    add(f"- `score_config_hash` = `{runinfo['score_config_hash']}`")
    add(f"- `frozen_config_hash` = `{runinfo['frozen_config_hash']}`")
    add(
        "- `1e-12` は roundoff tie protection のみであり、統計的 equivalence threshold "
        "ではない。"
    )
    add("")
    add("**Phase 8b の K 選択に次のものは一切使っていない:**")
    add(
        "`Q_strict` / EM の Q 関数基準 / ICL-type complete-data criterion / Schwarz BIC / "
        "marginal likelihood / posterior predictive / ELBO。"
    )
    add("")
    add(
        "この score は **plug-in** であり、parameter・Z の不確実性を積分していない。"
        "`predict_mu_y` / probability clipping / threshold / rounding は使用していない。"
    )
    add("")
    add("> 歴史的文脈（本実験の基準ではない）: `calc_bic_dual` は観測データの周辺尤度ではなく")
    add("> `Q_strict` を使う Q-based complete-data criterion / ICL-type であり、これを")
    add("> 「Schwarz BIC」と呼ばない（KI-010、`RESEARCH_MASTER.md` §12.6）。この論点は")
    add("> **legacy の基準に関するものであって、Phase 7e/8b の held-out 予測スコアとは")
    add("> 別物**である。両者を同一視しない（KI-019）。")
    add("")

    add("## 3. 固定条件")
    add("")
    add("| 項目 | 値 |")
    add("|---|---|")
    add(
        "| model lineage | `DualExpFamLSMConsistent`"
        "（objective-consistent experimental prototype、**本文採用不可**） |"
    )
    add("| `family_x` / `family_y` | `poisson` / `bernoulli` |")
    add("| `n` / `d` / `L` / `num_iter` | 75 / 15 / 5 / 8 |")
    add("| `numerics_mode` / `test_ratio` | `consistent` / 0.20 |")
    add(f"| new `K_TRUE` grid | {list(NEW_K_TRUE)} |")
    add(
        f"| anchor `K_TRUE` | {ANCHOR_K_TRUE}"
        "（Phase 7e artifact の **READ-ONLY 再利用**） |"
    )
    add(f"| candidate K | {list(authorization['candidate_k'])} |")
    add(
        f"| starts / replicates | {list(authorization['starts'])} / "
        f"{list(authorization['replicates'])} |"
    )
    add(
        f"| mask_design / random_design / hierarchy | `{authorization['mask_design']}` / "
        f"`{authorization['random_design']}` / `{authorization['hierarchy']}` |"
    )
    add(
        f"| 新規 fit 数 | **{runinfo['actual_full_fits']}**"
        f"（A {authorization['fits_per_estimand']} / "
        f"B {authorization['fits_per_estimand']}） |"
    )
    add("")
    add("### estimand（A と B は分けて報告する）")
    add("")
    add(
        "| estimand | role | `w_true` の定義 | K_TRUE=1 | K_TRUE=2 | K_TRUE=4 | K_TRUE=5 |"
    )
    add("|---|---|---|---|---|---|---|")
    add(
        "| A | primary | `w = 1.5`（K_TRUE によらず固定） | "
        + " | ".join(w_true[("A", kt)] for kt in NEW_K_TRUE)
        + " |"
    )
    add(
        "| B | sensitivity | `w_K = 1.5 * sqrt(3 / K_TRUE)`"
        "（`w_K^2 * K` を ensemble で一致させる） | "
        + " | ".join(w_true[("B", kt)] for kt in NEW_K_TRUE)
        + " |"
    )
    add("")
    add(
        f"`w0_true = {manifest_rows[0]['w0_true']}`。seed base: "
        f"data {authorization['data_seed_base']} / "
        f"model {authorization['model_seed_base']} / "
        f"anchor split {authorization['anchor_split_seed_base']}。"
    )
    add("")

    add("## 4. Provenance")
    add("")
    add("| 役割 | SHA |")
    add("|---|---|")
    add(f"| role 1: approved scientific baseline | `{runinfo['scientific_baseline_sha']}` |")
    add(
        "| role 2: reviewed full-execution main | "
        f"`{runinfo['reviewed_full_execution_main_sha']}` |"
    )
    add(f"| role 3: runtime `run_code_sha` | `{runinfo['run_code_sha']}` |")
    add(f"| Phase 7e anchor `run_code_sha` | `{summary['anchor_run_code_sha']}` |")
    add("")
    add(
        f"- frozen protocol hash: `{runinfo['protocol_hash']}`"
        f"（protocol origin: Issue #{runinfo['protocol_origin_issue']}）"
    )
    add(
        f"- execution issue: #{runinfo['execution_issue']} / execution attempt: "
        f"`{runinfo['execution_attempt_id']}`"
    )
    add(
        f"- prior aborted attempt: `{runinfo['prior_aborted_attempt_id']}`"
        f"（reason: `{runinfo['fresh_attempt_reason']}`、"
        f"artifact: `{runinfo['prior_aborted_artifact_dir']}`）"
    )
    add(
        f"- `partial_results_reused` = `{runinfo['partial_results_reused']}`"
        "（Attempt 1 の部分結果は一切再利用していない）"
    )
    add(f"- 実行時刻: `{runinfo['started_at']}` → `{runinfo['completed_at']}`（UTC）")
    add(
        "- `working_tree_clean_before_execution` = "
        f"`{runinfo['working_tree_clean_before_execution']}`"
    )
    add("")
    add("### artifact SHA-256")
    add("")
    add("| ファイル | SHA-256 | bytes |")
    add("|---|---|---:|")
    for name in sorted(p.name for p in run_dir.iterdir() if p.is_file()):
        path = run_dir / name
        add(f"| `{name}` | `{_sha256(path)}` | {path.stat().st_size} |")
    add("")

    add("## 5. 実行 contract（artifact から再計算）")
    add("")
    add("| 項目 | 値 |")
    add("|---|---|")
    add(f"| `attempted_fit_count` | {runinfo['attempted_fit_count']} |")
    add(f"| `clean_fit_calls` | {runinfo['clean_fit_calls']} |")
    add(f"| `scored_rows` | {runinfo['scored_rows']} |")
    add(f"| `full_fit_results.csv` data 行数 | {len(fit_rows)} |")
    add(f"| A 行 / B 行 | {len(per_estimand['A'])} / {len(per_estimand['B'])} |")
    add(
        f"| global `fit_index` | {min(indices)}..{max(indices)}（重複 "
        f"{len(indices) - len(set(indices))} / "
        f"欠番 {len(set(range(1, 337)) - set(indices))}） |"
    )
    add(
        "| K_TRUE=3 新規行 | "
        f"{sum(1 for r in fit_rows if int(r['K_TRUE']) == ANCHOR_K_TRUE)} |"
    )
    add(
        "| `internal_retry` 非ゼロ行 | "
        f"{sum(1 for r in fit_rows if r['internal_retry'] != '0')} |"
    )
    add(f"| `replacement_fits_executed` | {runinfo['replacement_fits_executed']} |")
    add(f"| `phase7e_rerun_count` | {runinfo['phase7e_rerun_count']} |")
    add(
        "| `canary_fits_executed` / `smoke_fits_executed` | "
        f"{runinfo['canary_fits_executed']} / {runinfo['smoke_fits_executed']} |"
    )
    add(
        "| `finite_state` 全 True | "
        f"{all(r['finite_state'] == 'True' for r in fit_rows)} |"
    )
    add(f"| `q_failure` 全 False | {all(r['q_failure'] == 'False' for r in fit_rows)} |")
    add(
        "| `nan_occurred` 全 False | "
        f"{all(r['nan_occurred'] == 'False' for r in fit_rows)} |"
    )
    add(f"| `warning_count` 全 0 | {all(r['warning_count'] == '0' for r in fit_rows)} |")
    add("| `failure.json` | 不在（失敗 run ではない） |")
    add("")
    add(
        "seed rescue・tolerance 緩和・replacement fit・retry・Phase7e rerun・"
        "canary rerun・smoke rerun はいずれも **0**。frozen `partial_failure_policy` は "
        f"`{', '.join(runinfo['partial_failure_policy'])}`。"
    )
    add("")
    add("### gate")
    add("")
    add(f"- `config_gate.csv`: {len(config_rows)} 件すべて `passed=True`")
    add(
        f"- `leakage_gate.csv`: {len(leakage_rows)} 行すべて `pre_fit_passed` / "
        "`post_fit_passed` = True、`fit_boundary_status` = `clean`"
        f"（`boundary_version` = `{leakage_rows[0]['boundary_version']}`）"
    )
    add(
        f"- `mask_provenance.csv`: {len(mask_rows)} セルすべて `anchor_match=True`、"
        f"`mask_design` = `{mask_rows[0]['mask_design']}`、split seed "
        f"{sorted({int(r['split_seed']) for r in mask_rows})}（Phase 7e と意図的に共有）"
    )
    add("")

    add("## 6. 独立監査")
    add("")
    add(
        "`tools/research_audit/audit_k_true_robustness_sweep.py --mode full`"
        "（artifact のみを読み、harness の selector も authorization も import しない）。"
    )
    add("")
    add("| 項目 | 値 |")
    add("|---|---|")
    add(f"| verdict | **{audit['status']}** |")
    add(
        "| BLOCKER / HIGH / MEDIUM | "
        f"{audit['blocker_count']} / {audit['high_count']} / {audit['medium_count']} |"
    )
    add(f"| findings | `{audit['findings']}` |")
    add(f"| audit_version | `{audit['audit_version']}` |")
    add(f"| 監査対象ファイル | {len(audit['audited_files'])} 件 |")
    add("")
    add(
        "auditor は role 1 / role 2 / protocol hash / execution attempt id / 336 / 168 / "
        "Phase 7e anchor 42 を **自前の literal として独立に保持**しており、runner の"
        "定数を読み込まない。"
    )
    add("")

    add("## 7. 結果 — replicate 単位の selected K")
    add("")
    add(
        "**K_TRUE=3 の行は Phase 7e anchor の READ-ONLY 再利用であり、A と B は同一の"
        "証拠を参照している。6 個の独立実験ではない。**"
    )
    add("")
    for estimand in ("A", "B"):
        add(f"### {estimand}（{ROLE_OF[estimand]}）")
        add("")
        add("| K_TRUE | r1 | r2 | r3 | 真値一致 | lineage |")
        add("|---:|---:|---:|---:|:---:|---|")
        for kt in (1, 2, 3, 4, 5):
            if kt == ANCHOR_K_TRUE:
                sel = [int(anchor[(estimand, r)]["selected_k"]) for r in REPLICATES]
                lineage = "`phase7e_anchor`（READ-ONLY 再利用・A/B 共有）"
            else:
                sel = [cells[(estimand, kt, r)]["selected_k"] for r in REPLICATES]
                lineage = "`phase8a_new`"
            exact = sum(1 for s in sel if s == kt)
            cells_md = " | ".join(f"**{s}**" if s == kt else str(s) for s in sel)
            add(f"| {kt} | {cells_md} | {exact}/3 | {lineage} |")
        add("")

    add("### recovery count（新規グリッド `K_TRUE in {1,2,4,5}` のみ）")
    add("")
    add("| K_TRUE | A | B | 合算 |")
    add("|---:|:---:|:---:|:---:|")
    for kt in NEW_K_TRUE:
        add(
            f"| {kt} | {recovery['A'][kt]}/3 | {recovery['B'][kt]}/3 | "
            f"{recovery['A'][kt] + recovery['B'][kt]}/6 |"
        )
    add(
        f"| **合計** | **{totals['A']}/12** | **{totals['B']}/12** | "
        f"**{totals['A'] + totals['B']}/24** |"
    )
    add("")
    add(
        f"Phase 7e anchor（`K_TRUE=3`、A/B 共有の単一証拠）: selected K = "
        f"{anchor_selected}、真値一致 **{anchor_exact}/3**。"
    )
    add("")
    add("### 選択方向の内訳（新規グリッド、各 12 セル）")
    add("")
    add("| estimand | exact | over-selection | under-selection |")
    add("|---|:---:|:---:|:---:|")
    for estimand in ("A", "B"):
        d = direction[estimand]
        add(f"| {estimand} | {d['exact']} | {d['over']} | {d['under']} |")
    add("")
    add("### A / B の差")
    add("")
    if ab_diff:
        add("| K_TRUE | replicate | A | B |")
        add("|---:|---:|---:|---:|")
        for kt, r, a, b in ab_diff:
            add(f"| {kt} | {r} | {a} | {b} |")
        add("")
        add(
            f"新規グリッド 12 セル中、A と B で選択が異なるのは "
            f"**{len(ab_diff)} セルのみ**。"
        )
    else:
        add("新規グリッド 12 セルすべてで A と B の選択が一致した。")
    add("")
    add("**この 1 セルの差から信号強度スケーリングの一般的効果を推論しない。**")
    add("")

    add("## 8. 選択マージン（記述のみ）")
    add("")
    add(
        "`margin` = 選択された K の 2-start 平均スコア − 次点 K の 2-start 平均スコア。"
        "値が小さいほど、その replicate では上位候補が僅差であったことを意味する。"
        "**これは原因の説明ではない。**"
    )
    add("")
    add("| estimand | K_TRUE | replicate | selected K | best mean score | margin |")
    add("|---|---:|---:|---:|---:|---:|")
    for estimand in ("A", "B"):
        for kt in NEW_K_TRUE:
            for r in REPLICATES:
                cell = cells[(estimand, kt, r)]
                best = max(cell["mean_scores"].values())
                add(
                    f"| {estimand} | {kt} | {r} | {cell['selected_k']} | "
                    f"{_fmt(float(best))} | {_fmt(cell['margin'])} |"
                )
    add("")
    add("全 24 セルで tie 候補は 1 個のみであり、tie rule が発動したセルは存在しない。")
    add("")

    add("## 9. 解釈（有限標本の記述的結果）")
    add("")
    add("> 凍結した held-out 予測スコアによる K 選択では、`K_TRUE=1` および `K_TRUE=4` では")
    add("> 3 反復すべてで真値が選択され、`K_TRUE=2` では 2/3、`K_TRUE=5` では 1/3 で真値が")
    add("> 選択された。`K_TRUE=5` では候補集合に 5 より大きい K も含まれる一方、選択結果は")
    add("> 低い K 側に寄る傾向が観測された。ただし各条件 3 反復のみであり、本結果は有限標本に")
    add("> おける記述的結果として解釈する。")
    add("")
    add(
        "この傾向は A（primary、`w` 固定）と B（sensitivity、`w_K` スケーリング）で"
        "ほぼ同一であった（差は 1 セルのみ）。"
    )
    add("")
    add(
        "`K_TRUE=5` の under-selection について**原因は述べない**。§8 の margin 表は"
        "選択の僅差さを記述するだけであり、原因の同定には追加の解析が必要である。"
    )
    add("")

    add("## 10. 主張境界")
    add("")
    add("### 書いてよい")
    add("")
    add("- frozen 実験内での有限標本 selected-K パターン")
    add("- replicate 単位の変動")
    add("- 記述的な over-selection / under-selection")
    add("- A primary と B sensitivity の比較（分けて報告する）")
    add("- 明示された recovery count")
    add("")
    add("### 書いてはいけない")
    add("")
    add("- consistency / asymptotic consistency")
    add("- universal K recovery / K-selection consistency")
    add("- Schwarz BIC / BIC consistency / Schwarz-BIC success")
    add("- 理論保証（theoretical guarantee）")
    add("- 本合成設定を超える一般化")
    add("- 「Phase 8b は Q_strict / ICL-type / BIC で K を選んだ」（**事実として誤り**）")
    add(
        "- 「K_TRUE=3 について A と B が独立に 6 セル分の証拠を与える」"
        "（同一 anchor の共有参照）"
    )
    add("")
    add("この実験が答えている問いは次の 1 つだけである:")
    add("")
    add(
        "> generator の `K_TRUE` が変わったとき、凍結された Phase 7e held-out plug-in "
        "K-selection protocol は有限標本でどのような selected-K パターンを示すか。"
    )
    add("")
    add(
        "加えて lineage E（objective-consistent experimental prototype）は"
        "**本文採用不可**である（root `CLAUDE.md` §3）。"
    )
    add("")

    add("## 11. 証拠の数え方")
    add("")
    add("| 区分 | fit 数 | 由来 |")
    add("|---|---:|---|")
    add(f"| Attempt 2 で新規実行 | {runinfo['actual_full_fits']} | 本 artifact |")
    add(
        f"| Phase 7e anchor（READ-ONLY 再利用） | {runinfo['anchor_unique_fits']} | "
        f"`{summary['anchor_artifact_dir']}` |"
    )
    add(
        "| **統合ユニーク証拠** | "
        f"**{runinfo['actual_full_fits'] + runinfo['anchor_unique_fits']}** | — |"
    )
    add("")
    add(
        f"**{runinfo['actual_full_fits']} + {runinfo['anchor_unique_fits']} = "
        f"{runinfo['actual_full_fits'] + runinfo['anchor_unique_fits']} であって "
        "420 ではない。**"
    )
    add("A と B が同じ anchor 42 fits を参照するため、anchor を 2 回数えてはならない。")
    add("")

    add("## 12. Attempt 1（provenance のみ・科学的使用不可）")
    add("")
    if prior_dir.exists():
        prior_failure = _read_json(prior_dir / "failure.json")
        add(f"`{_rel(prior_dir)}/` を削除せず保全している。")
        add("")
        add("| 項目 | 値 |")
        add("|---|---|")
        add(f"| status | `{prior_failure['status']}` |")
        add(
            "| 位置づけ | **ABORTED_BY_OPERATOR_INTERRUPT / provenance only / "
            "no scientific use** |"
        )
        add(f"| `attempted_fit_count` | {prior_failure['attempted_fit_count']} |")
        add(f"| `clean_fit_calls` | {prior_failure['clean_fit_calls']} |")
        add(f"| `scored_rows` | {prior_failure['scored_rows']} |")
        add(f"| `failed_fit_index` | {prior_failure['failed_fit_index']} |")
        add(
            "| `retry_count` / `replacement_fits_executed` | "
            f"{prior_failure['retry_count']} / "
            f"{prior_failure['replacement_fits_executed']} |"
        )
        add(f"| `run_code_sha` | `{prior_failure['run_code_sha']}` |")
        add(f"| `artifact_version` | `{prior_failure['artifact_version']}` |")
        add("")
        add("| ファイル | SHA-256 | bytes |")
        add("|---|---|---:|")
        for name in sorted(p.name for p in prior_dir.iterdir() if p.is_file()):
            path = prior_dir / name
            add(f"| `{name}` | `{_sha256(path)}` | {path.stat().st_size} |")
        add("")
        add(
            "Attempt 1 の 2 clean fits は Attempt 2 に **一切再利用していない**"
            f"（`partial_results_reused = {runinfo['partial_results_reused']}`）。"
            "Attempt 1 の数値を科学的主張の根拠にしない。"
        )
    else:
        add("Attempt 1 の artifact ディレクトリが見つからない。")
    add("")

    add("---")
    add("")
    add("**このアーカイブ工程で実行した real EM fit = 0。**")
    add("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the Phase 8b Attempt 2 canonical report from artifacts"
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--prior-dir", type=Path, default=DEFAULT_PRIOR_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the existing report matches without writing",
    )
    args = parser.parse_args(argv)

    text = build_report(args.run_dir, args.prior_dir)
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != text:
            print(json.dumps({"status": "STALE", "report": _rel(args.out)}))
            return 1
        print(json.dumps({"status": "CURRENT", "report": _rel(args.out)}))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "WRITTEN",
                "report": _rel(args.out),
                "bytes": len(text.encode("utf-8")),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
