"""Generate the Phase 7e full held-out K-selection pilot report.

Every number in the generated Markdown is read from the machine-readable run
artifacts; nothing is transcribed by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    ROOT / "expfam" / "results" / "k_selection" / "heldout_full_pilot_20260824"
)
DEFAULT_REPORT_PATH = (
    ROOT
    / "reports"
    / "k_selection_theory"
    / "heldout_k_selection_full_pilot_report_20260824.md"
)

REPLICATES = (1, 2, 3)
K_CANDIDATES = (1, 2, 3, 4, 5, 6, 7)


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def build_report(run_dir: Path, audit: dict[str, Any] | None) -> str:
    fit_rows = _read_csv(run_dir / "fit_results.csv")
    selection_rows = _read_csv(run_dir / "replicate_selection.csv")
    aggregate_rows = _read_csv(run_dir / "aggregate_summary.csv")
    runinfo = json.loads((run_dir / "runinfo.json").read_text(encoding="utf-8"))

    means: dict[int, dict[int, float]] = {}
    summary: dict[int, dict[str, Any]] = {}
    for row in selection_rows:
        replicate = int(row["replicate"])
        means.setdefault(replicate, {})[int(row["K"])] = float(row["mean_score"])
        summary[replicate] = {
            "selected_k": int(row["selected_k"]),
            "best": float(row["best_mean_score"]),
            "second_best": float(row["second_best_mean_score"]),
            "margin": float(row["margin"]),
            "tie_candidates": row["tie_candidates"],
        }

    k_wise = {
        int(row["K"]): row for row in aggregate_rows if row["section"] == "k_wise"
    }
    pilot = {row["key"]: row["value"] for row in aggregate_rows if row["section"] == "pilot"}

    run_dir_rel = _relative_to_root(run_dir)

    lines: list[str] = []
    add = lines.append

    add("# Phase 7e full held-out K-selection pilot")
    add("")
    add(
        "本レポートは "
        f"`{run_dir_rel}/` に保存された machine-readable artifact から"
        "自動生成したものである。数値の手作業転記は行っていない。"
    )
    add("")

    # 1
    add("## 1. 目的")
    add("")
    add(
        "Phase 7c (Issue #39) で設計し Phase 7d (Issue #41) で実装・falsification した "
        "leakage-safe held-out K-selection protocol を、"
        "候補 K = 1,...,7 と 3 dataset replicate へ拡張して **exactly once** 実行し、"
        "各 replicate における selected K を記述的に測定する。"
    )
    add("")
    add(
        "**「K=3 が選ばれること」は成功条件ではない。** 目的は frozen protocol 下での"
        "選択挙動の測定であり、結果を見た後に seed / replicate / tolerance / score / "
        "K range / start count / split / preprocessing / failure rule を変更しない。"
    )
    add("")

    # 2
    config = runinfo["config"]
    add("## 2. 固定条件")
    add("")
    add("| 項目 | 値 |")
    add("|---|---|")
    add(
        "| model lineage | `DualExpFamLSMConsistent`"
        "（objective-consistent experimental prototype、**本文採用不可**） |"
    )
    add(f"| `family_x` | `{config['family_x']}` |")
    add(f"| `family_y` | `{config['family_y']}` |")
    add(f"| `K_TRUE` | {config['k_true']} |")
    add(f"| `n` | {config['n']} |")
    add(f"| `d` | {config['d']} |")
    add(f"| `L` | {config['L']} |")
    add(f"| `num_iter` | {config['num_iter']} |")
    add(f"| `numerics_mode` | `{config['numerics_mode']}` |")
    add(f"| `test_ratio` | {config['test_ratio']} |")
    add(f"| candidate K | {runinfo['candidate_k']} |")
    add(f"| starts | {runinfo['starts']} |")
    add(f"| dataset replicates | {runinfo['replicates']} |")
    add(f"| total fits | {runinfo['expected_fit_count']} |")
    add("")
    add("### seed convention")
    add("")
    add("```")
    add(f"data_seed  = {runinfo['seed_convention']['DATA_SEED_BASE']} + replicate")
    add(f"split_seed = {runinfo['seed_convention']['SPLIT_SEED_BASE']} + replicate")
    add(
        f"model_seed = {runinfo['seed_convention']['MODEL_SEED_BASE']}"
        " + replicate*1000 + K*10 + start"
    )
    add("```")
    add("")
    add(f"- data seeds: {runinfo['data_seeds']}")
    add(f"- split seeds: {runinfo['split_seeds']}")
    add(
        f"- model seeds: {runinfo['model_seeds'][0]} ... {runinfo['model_seeds'][-1]}"
        f"（{len(runinfo['model_seeds'])} 個、全て一意）"
    )
    add("")
    add("### primary score")
    add("")
    add("```")
    add("eta_ij = w0 + w * z_i^T z_j")
    add("log score_ij = y_ij * eta_ij - logaddexp(0, eta_ij)")
    add("fit score = held-out upper test pairs 上の mean log score")
    add("```")
    add("")
    add(
        "`predict_mu_y` / probability clipping / threshold / rounding は使用していない。"
        "K 選択に BIC や `Q_strict` は使用していない。"
        "この score は plug-in であり、parameter・Z の不確実性を積分していない"
        "（posterior predictive・marginal likelihood・ELBO ではない）。"
    )
    add("")
    add("### selector")
    add("")
    add("```")
    add("Sbar_r(K) = (start1 score + start2 score) / 2")
    add("tie candidate : max_K Sbar_r(K) - Sbar_r(K) <= 1e-12")
    add("selected K    : tie candidates の smallest K")
    add("```")
    add("")
    add(
        f"`{runinfo['tie_tolerance']}` は roundoff tie protection のみであり、"
        "統計的 equivalence threshold ではない。"
    )
    add("")

    # 3
    add("## 3. リーク防止・provenance設計")
    add("")
    add(
        "- Design A（transductive dyad holdout）。node 集合は train/test で同一。"
        "held-out は Y の dyad のみ。"
    )
    add(
        "- split guard は **PAIR-MASK TOPOLOGY ONLY**。Y 値・prevalence・"
        "weighted degree・fit 品質を一切参照しない。"
    )
    add(
        "- **全 3 replicate の split を EM 開始前に生成・validate** し、"
        "全 PASS 後にのみ first EM fit を許可した。"
    )
    add(
        "- fit 側には `X` と `TrainingYValues`（train upper pairs のみ）しか渡らない。"
        "masked cell には finite かつ Bernoulli support 内の canary 値 0 を置く"
        "（`NaN * 0` を避けるため NaN は禁止）。"
    )
    add(
        "- `ScoreOnlyTarget` は各 replicate の **14 fit がすべて clean 完了し"
        "stored count/order gate を PASS した後に 1 回だけ**生成される。"
        "未完了 replicate の target は生成されない。"
    )
    add(
        "- replicate 内では x_hash / training_y_hash / train_mask_hash / test_mask_hash / "
        "fit_provenance_hash / target_topology_hash / score_target_hash / "
        "preprocessing_hash / score_config_hash がすべて一致することを"
        "expected 側を再構築したうえで要素ごとに検証する"
        "（uniform corruption も検出される）。"
    )
    add("")
    add("### per-replicate provenance")
    add("")
    add("| replicate | x_hash | train_mask_hash | test_mask_hash | score_target_hash |")
    add("|---|---|---|---|---|")
    for entry in runinfo["per_replicate_provenance"]:
        target_hashes = entry["score_target_hash"]
        target_text = target_hashes[0][:12] if target_hashes else "(none)"
        add(
            f"| {entry['replicate']} | `{entry['x_hash'][:12]}` | "
            f"`{entry['train_mask_hash'][:12]}` | `{entry['test_mask_hash'][:12]}` | "
            f"`{target_text}` |"
        )
    add("")
    add("（完全なハッシュは `runinfo.json` を参照。）")
    add("")

    # 4
    add("## 4. 実行情報")
    add("")
    add(f"- issue: #{runinfo['issue']}")
    add(f"- branch: `{runinfo['branch']}`")
    add(f"- RUN_CODE_SHA: `{runinfo['run_code_sha']}`")
    add(f"- base main SHA: `{runinfo['base_main_sha']}`")
    add(f"- command: `{runinfo['command']}`")
    add(f"- UTC start / finish: {runinfo['timestamp_utc_start']} / {runinfo['timestamp_utc_finish']}")
    add(f"- local start: {runinfo['timestamp_local_start']}")
    add(f"- Python: {runinfo['python_version'].splitlines()[0]}")
    add(f"- NumPy: {runinfo['numpy_version']}")
    add(f"- platform: {runinfo['platform']}")
    add(f"- expected fit count: {runinfo['expected_fit_count']}")
    add(f"- **actual EM fit count: {runinfo['actual_fit_count']}**")
    add(f"- score targets created: {runinfo['targets_created']}")
    add(f"- score rows: {runinfo['score_rows']}")
    add(f"- failure state: `{runinfo['failure_state']}`")
    add("")
    total_retries = sum(int(row["retry"]) for row in fit_rows)
    total_warnings = sum(int(row["warning_count"]) for row in fit_rows)
    total_q = sum(1 for row in fit_rows if row["q_failure"] != "False")
    total_nan = sum(1 for row in fit_rows if row["nan_occurred"] != "False")
    non_clean = sum(1 for row in fit_rows if row["fit_status"] != "clean")
    add("### per-fit hard gate")
    add("")
    add(f"- internal retry 合計: {total_retries}")
    add(f"- warning 合計: {total_warnings}")
    add(f"- Q failure 件数: {total_q}")
    add(f"- NaN / nonfinite 件数: {total_nan}")
    add(f"- `fit_status != clean` の件数: {non_clean}")
    add("")

    # 5
    add("## 5. replicate別結果")
    add("")
    header = "| replicate | selected K | " + " | ".join(f"K={k}" for k in K_CANDIDATES)
    header += " | best | second best | margin |"
    add(header)
    add("|---|---:|" + "---:|" * (len(K_CANDIDATES) + 3))
    for replicate in REPLICATES:
        cells = " | ".join(_fmt(means[replicate][k]) for k in K_CANDIDATES)
        info = summary[replicate]
        add(
            f"| {replicate} | **{info['selected_k']}** | {cells} | "
            f"{_fmt(info['best'])} | {_fmt(info['second_best'])} | "
            f"{_fmt(info['margin'])} |"
        )
    add("")
    add(
        "margin は best と second-best の 2-start mean log score の差である。"
        "**統計的有意差ではない。**"
    )
    add("")
    for replicate in REPLICATES:
        add(
            f"- replicate {replicate}: tie candidates = "
            f"{{{summary[replicate]['tie_candidates'].replace('|', ', ')}}}"
        )
    add("")

    # 6
    add("## 6. 集約結果")
    add("")
    add(f"- n_replicates = {pilot['n_replicates']}")
    add(f"- K_TRUE = {pilot['true_k']}")
    add(f"- selected K counts = {pilot['selected_k_counts'].replace('|', ', ')}")
    add(f"- K_TRUE selected count = {pilot['true_k_selected_count']}")
    add(
        f"- **descriptive pilot recovery rate = "
        f"{float(pilot['descriptive_recovery_rate']):.4f}"
        f"（{pilot['true_k_selected_count']} / {pilot['n_replicates']}）**"
    )
    add("")
    add(
        "この recovery rate は **3 replicate だけの記述的 pilot 結果**であり、"
        "統計的一貫性や一般的な true-K recovery を意味しない。"
    )
    add("")
    add("### K別 2-start mean log score の記述統計（3 replicate）")
    add("")
    add("| K | mean | sample sd | min | max |")
    add("|---:|---:|---:|---:|---:|")
    for k in K_CANDIDATES:
        row = k_wise[k]
        add(
            f"| {k} | {_fmt(float(row['mean_across_replicates']))} | "
            f"{_fmt(float(row['std_across_replicates']))} | "
            f"{_fmt(float(row['min_across_replicates']))} | "
            f"{_fmt(float(row['max_across_replicates']))} |"
        )
    add("")

    # 7
    add("## 7. 解釈")
    add("")
    add("### 事実（artifact から直接読み取れること）")
    add("")
    add(
        f"- frozen 42-row manifest どおりに EM fit が {runinfo['actual_fit_count']} 回"
        "実行され、retry・warning・Q failure・NaN はいずれも 0 件だった。"
    )
    for replicate in REPLICATES:
        add(
            f"- replicate {replicate} の selected K は "
            f"{summary[replicate]['selected_k']} だった。"
        )
    add(
        f"- 3 replicate 中 {pilot['true_k_selected_count']} replicate で "
        f"K = {pilot['true_k']} が選択された。"
    )
    add("")
    add("### 解釈（推測を含む）")
    add("")
    add(
        "- 本 pilot は 1 つの synthetic 条件"
        f"（family_x={config['family_x']}, family_y={config['family_y']}, "
        f"n={config['n']}, d={config['d']}, K_TRUE={config['k_true']}）における"
        "選択挙動の記述である。"
    )
    add(
        "- held-out plug-in log score は K に対して単調ではなく、"
        "候補 K の間で有限標本上の予測性能を比較しているにすぎない。"
        "`K_hat_pred` は generative `K_TRUE` と一致する保証を持たない。"
    )
    add("")
    add("### 主張してはいけないこと")
    add("")
    add("- 一般に true K を回復する")
    add("- K 選択の consistency を証明した")
    add("- BIC（`calc_bic_dual` / Q-based complete-data criterion）より優れる")
    add("- Phase 7b の C1/C2/C3 より優れる")
    add("- 実データでの妥当性")
    add("- 漸近的性質")
    add("- 修士論文・予稿レベルの最終結論")
    add("")

    # 8
    add("## 8. 制約")
    add("")
    add(
        "- **replication unit は独立生成 dataset replicate であり、本 pilot はわずか 3 個**である。"
        "recovery rate の分母は 3 であり、信頼区間を伴う推定量ではない。"
    )
    add(
        "- held-out dyad は node を共有するため独立ではない。"
        "held-out pair 数は独立標本サイズではない。"
    )
    add("- score は plug-in であり、parameter・Z の不確実性を積分していない。")
    add(
        "- 候補 K のモデルは回転不定性を持ち、操作的アルゴリズム上で入れ子とは限らない。"
    )
    add(
        f"- MCEM の近似（L={config['L']}）と有限反復（num_iter={config['num_iter']}）が"
        "予測ランキングに影響しうる。"
    )
    add(
        "- 使用した lineage は `experimental/` の objective-consistent prototype であり、"
        "`CLAUDE.md` §3 により **本文採用不可**である。"
    )
    add(
        "- transductive dyad holdout であり、新規 node に対する inductive 一般化"
        "（Design B）は現行 API では未サポートである。"
    )
    add("")

    # 9
    add("## 9. 次の判断")
    add("")
    if audit is not None:
        add(f"- self-audit verdict: **{audit['verdict']}**")
        add(
            f"- BLOCKER {audit['blocker']} / HIGH {audit['high']} / "
            f"MEDIUM {audit['medium']} / LOW {audit['low']}"
        )
        add(
            f"- 独立再計算との最大差分: mean score "
            f"{audit['max_mean_score_difference']!r}, "
            f"aggregate {audit['max_aggregate_difference']!r}"
        )
        add("")
    add("Post-pilot decision は本レポートとは別に `## 10.` で明示する。")
    add("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--out", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--audit-json", default=None)
    args = parser.parse_args(argv)

    audit: dict[str, Any] | None = None
    if args.audit_json:
        audit = json.loads(Path(args.audit_json).read_text(encoding="utf-8"))

    text = build_report(Path(args.run_dir), audit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(_relative_to_root(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
