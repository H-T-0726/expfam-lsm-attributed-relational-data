"""実験条件・再現性・記録完全性の監査台帳を、固定 SHA の一次資料から生成する（読み取り専用）。

DOCUMENTATION / PROVENANCE AUDIT（2026-11-28）。新しい fit・データ生成・最適化は行わない。
入力:
  - docs/research_results/data/experiment_inventory.csv（88 件、変更しない）
  - docs/research_results/data/addendum_20261127_inventory.csv（追補 4 記録、変更しない）
  - docs/research_results/data/primary_results_index.csv / addendum_20261127_primary_results.csv
  - 各記録の固定 SHA の git オブジェクト（git ls-tree / git show、作業ツリーは読まない）
  - 各固定 SHA の EXPERIMENT_REGISTRY.md（canonical doc。記載は DOCUMENTED 扱い）
出力（docs/research_results/data/）:
  - experiment_conditions_audit.csv   項目ごとの値と状態（long 形式、条件別の子行あり）
  - experiment_provenance_map.csv     実験 ID → 生成器・runner・protocol・runinfo・CSV・図・報告書・commit
  - experiment_record_gaps.csv        記録不足（MISSING）・食い違い（CONFLICT）・既知の問題
状態の定義:
  VERIFIED       固定 SHA の一次資料（runinfo / protocol / manifest / 結果 CSV / コード）で値またはファイルの存在を確認
  DOCUMENTED     台帳・registry・報告書に記載があるが、一次資料では照合していない
  MISSING        必要な情報が見つからない
  CONFLICT       資料間で記載が食い違う（値・SHA・出典を並べて記録）
  NOT_APPLICABLE その種類の記録には該当しない（MISSING に数えない）
注意:
  - 現行 runner のデフォルト値は使わない。値は「その実験の出力に保存された runinfo / protocol / CSV」からだけ取る。
  - 自動抽出は JSON / CSV のキー名に基づく。キー名の意味が近い別物（例: Laplace の max_iter）を拾うことがあるため、
    extraction 列に JSON のキーパスを残す。値の解釈は読み手が source で確認する。

使い方:
  python docs/research_results/scripts/build_experiment_records_audit.py          # 生成
  python docs/research_results/scripts/build_experiment_records_audit.py --check  # 生成物の品質検査のみ
"""
from __future__ import annotations

import csv
import fnmatch
import io
import json
import re
import subprocess
import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs" / "research_results" / "data"
REPO_URL = "https://github.com/H-T-0726/expfam-lsm-attributed-relational-data"
MAIN_SHA = "1b5030b"

OUT_COND = DATA / "experiment_conditions_audit.csv"
OUT_PROV = DATA / "experiment_provenance_map.csv"
OUT_GAPS = DATA / "experiment_record_gaps.csv"

STATUSES = ("VERIFIED", "DOCUMENTED", "MISSING", "CONFLICT", "NOT_APPLICABLE")

# --------------------------------------------------------------------------------------
# git helpers（読み取り専用）
# --------------------------------------------------------------------------------------
_ls_cache: dict[str, list[str]] = {}
_show_cache: dict[tuple[str, str], bytes | None] = {}
_exists_cache: dict[str, bool] = {}
_remote_cache: dict[str, list[str]] = {}


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True)


def sha_exists(sha: str) -> bool:
    if sha not in _exists_cache:
        _exists_cache[sha] = git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0
    return _exists_cache[sha]


def remote_branches(sha: str) -> list[str]:
    if sha not in _remote_cache:
        out = git("branch", "-r", "--contains", sha).stdout.decode("utf-8", "replace")
        _remote_cache[sha] = sorted(
            b.strip() for b in out.splitlines() if b.strip() and "->" not in b
        )
    return _remote_cache[sha]


def ls(sha: str) -> list[str]:
    if sha not in _ls_cache:
        out = git("ls-tree", "-r", "--name-only", sha).stdout.decode("utf-8", "replace")
        _ls_cache[sha] = out.splitlines()
    return _ls_cache[sha]


def show(sha: str, path: str, max_bytes: int = 8_000_000) -> bytes | None:
    key = (sha, path)
    if key not in _show_cache:
        size = git("cat-file", "-s", f"{sha}:{path}")
        if size.returncode != 0 or int(size.stdout.strip() or 0) > max_bytes:
            _show_cache[key] = None
        else:
            r = git("show", f"{sha}:{path}")
            _show_cache[key] = r.stdout if r.returncode == 0 else None
    return _show_cache[key]


def short(sha: str) -> str:
    return sha[:7]


# --------------------------------------------------------------------------------------
# 記録ごとの探索範囲（root）。台帳の evidence 列から作り、曖昧なものだけ手で指定する。
# 手指定の理由は ROOT_NOTES に残し、gaps に「evidence 列が曖昧」として記録する。
# --------------------------------------------------------------------------------------
KX = "expfam/results/k_selection/k_exp2_generalization_20261125/"
TG = "expfam/results/k_selection/two_gaps_20261127/"
ROOT_OVERRIDES: dict[str, list[str]] = {
    "A-REPRO": ["reproduction/results/comparison/"],
    "B-OLD05-EXP1-4": ["expfam/results/exp_scenario_*_exp*_*.csv"],
    "B-OLD05-AUX": ["expfam/results/exp1_full_*.csv", "expfam/results/exp2_bic_*"],
    "B-FIXED-EXP1": ["expfam/results/fixed_official/exp1_k9/"],
    "B-FIXED-EXP3": ["expfam/results/fixed_official/exp3/"],
    "B-FIXED-EXP4": ["expfam/results/fixed_official/exp4/"],
    "C-OD-DIAG": ["expfam/results/overdispersion/movielens_overdispersion_diagnostics*"],
    "C-ML-MIXED": [
        "expfam/results/per_column_family/movielens_mixed_x_*",
        "expfam/results/story_diagnostics/movielens_attribute_diagnosis_*",
    ],
    "D-PC-DEMO": ["expfam/results/per_column_family/per_column_demo_*", "expfam/results/per_column_family/single_vs_joint_*",
                  "expfam/results/per_column_family/attribute_ablation_*", "expfam/results/per_column_family/noise_check_*"],
    "D-COMPL": ["expfam/results/story_diagnostics/complementary_blocks_consistent_20260821*"],
    "D-COVER": ["expfam/results/story_diagnostics/matched_latent_coverage_ablation_20260821*"],
    "D-E2A": ["expfam/results/seed_separation_e2a/e2a_20261104/"],
    "D-E2C": ["expfam/src/experimental/run_em_iterations_e2c.py", "expfam/src/experimental/test_em_iterations_e2c.py"],
    "E-AF6": ["expfam/results/auto_family_z_recovery/af6_laplace_confirmation_20261114/"],
    "E-AF8": ["expfam/results/auto_family_z_recovery/af8_integrated_loading_20261115/"],
    "F-7B": ["expfam/results/k_selection/k_selection_score_pilot_20260823_*"],
    "F-8B-SMOKE": ["expfam/results/k_selection/k_true_robustness_smoke_20260901/"],
    "F-EMCONV": ["expfam/results/k_selection/em_convergence_sensitivity_20260905/"],
    "F-9H-9J": ["expfam/results/laplace_pilot/", "expfam/results/laplace_is_diagnostic/"],
    "F-128": ["expfam/results/k_selection/k_criterion_comparison_analysis_20261104/",
              "expfam/results/k_selection/clap_clean_nsweep_20261006/",
              "expfam/results/k_selection/paper_condition_k_criteria_20261006/"],
    "F-9M-9O": ["expfam/results/theta_stationarity/", "expfam/results/multistep_local/",
                "expfam/results/derivative_failure_classification/"],
    "F-9P-9V": ["expfam/results/relational_w_sensitivity/", "expfam/results/density_controlled_w_*",
                "expfam/results/attribute_scale_*"],
    "F-9W-9Y": ["expfam/results/matched_k_true_*", "expfam/results/k34_boundary_decomposition/"],
    "F-KEXP2-MAIN": [KX + "main/", KX + "protocol_v2/", KX + "analysis/", KX + "independent_audit/",
                     KX + "generator_diagnostics/"],
    "TG-A": [TG + "expA/", TG + "frozen_protocol_A.md", TG + "runinfo.csv#experiment=A"],
    "TG-B": [TG + "expB/", TG + "frozen_protocol_B.md", TG + "runinfo.csv#experiment=B",
             TG + "fairness_audit.md"],
    "TG-DIAG": [TG + "diagnostics_20261127/*.csv", TG + "diagnostics_20261127/*.md",
                TG + "diagnostics_20261127/figures/"],
    "TG-CORR": [TG + "diagnostics_20261127/forward_correction_20261127/"],
}
ROOT_NOTES = {
    "D-E2A": "台帳の evidence 列が「expfam/results/（e2a）」で場所を特定できない。registry（E2a 節）の結果パスで補った",
    "F-8B-SMOKE": "台帳の evidence 列が「expfam/results/k_selection/（8b smoke）」で場所を特定できない。同名ディレクトリで補った",
    "F-EMCONV": "台帳の evidence 列が「expfam/results/k_selection/（em convergence）」で場所を特定できない。registry の結果パスで補った",
    "D-E2C": "設計のみ（fit 0）。evidence 列は「tools（runner）」で、runner の実ファイルは 11f42bf の commit 内容で特定した",
    "F-128": "evidence 列は派生表だけを指し、fit の一次出力の場所を特定できない。registry の #128 A1・B 行の結果パスで補った",
    "B-OLD05-EXP1-4": "evidence 列の brace 表記を glob に展開した",
    "B-OLD05-AUX": "evidence 列の brace 表記を glob に展開した",
    "B-FIXED-EXP4": "evidence 列の brace 表記を glob に展開した",
}
# 記録の sha（ファイルを読む SHA）を手で指定するもの
SHA_OVERRIDES = {"D-E2C": "11f42bf"}

MANUAL_RUNNERS = {
    # (exp_id): [(path, sha, 根拠)]  — registry 行や commit 内容で runner を特定したもの
    "D-E2C": [("expfam/src/experimental/run_em_iterations_e2c.py", "11f42bf", "commit 11f42bf の変更ファイル")],
    "TG-DIAG": [("tools/research_audit/diagnose_two_gaps_20261127.py", "2f71e61", "commit 2f71e61 の変更ファイル"),
                ("tools/research_audit/check_diagnostics_two_gaps_20261127.py", "2f71e61", "commit 2f71e61 の変更ファイル")],
    "TG-CORR": [("tools/research_audit/correct_two_gaps_diagnostics_20261127.py", "c035df7", "commit c035df7 の変更ファイル"),
                ("tools/research_audit/check_correction_two_gaps_20261127.py", "c035df7", "commit c035df7 の変更ファイル")],
}


def parse_evidence(ev: str) -> list[str]:
    ev = ev.strip()
    if not ev or ev == "—":
        return []
    m = re.search(r"/tree/[0-9a-f]+/(\S+)", ev)
    if m:
        return [m.group(1)]
    # brace の中のカンマで分割しないよう、brace を一時退避
    parts, buf, depth = [], "", 0
    for ch in ev:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch in ",、" and depth == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    parts.append(buf)
    out, base = [], ""
    for p in parts:
        p = p.strip().replace(" ほか", "")
        if not p:
            continue
        p = re.sub(r"（.*?）", "*", p)
        p = re.sub(r"\{[^}]*\}", "*", p)
        if not p.startswith(("expfam/", "reports/", "reproduction/", "paper/", "docs/", "tools/", "GEMINI")) and base:
            p = base + p
        if "/" in p:
            base = p.rsplit("/", 2)[0] + "/" if p.endswith("/") else p.rsplit("/", 1)[0] + "/"
        out.append(p)
    return out


def match_roots(files: list[str], roots: list[str]) -> list[str]:
    hits = []
    for f in files:
        for r in roots:
            r = r.split("#", 1)[0]
            if r.endswith("/") and "*" in r:
                if fnmatch.fnmatch(f, r + "*"):
                    hits.append(f)
                    break
            elif r.endswith("/"):
                if f.startswith(r):
                    hits.append(f)
                    break
            elif "*" in r:
                if fnmatch.fnmatch(f, r) or fnmatch.fnmatch(f, r + "*") or fnmatch.fnmatch(f, r.rstrip("*") + "*/*"):
                    hits.append(f)
                    break
            elif f == r or f.startswith(r.rstrip("/") + "/"):
                hits.append(f)
                break
    return sorted(set(hits))


def row_filter_for(roots: list[str], path: str) -> tuple[str, str] | None:
    for r in roots:
        if "#" in r and r.split("#", 1)[0] == path:
            k, v = r.split("#", 1)[1].split("=", 1)
            return k, v
    return None


def classify(f: str) -> str:
    name = f.rsplit("/", 1)[-1].lower()
    if "/fits/" in f or "/states/" in f:
        return "per_fit"
    if re.search(r"protocol|preregist|preflight|go_decision", name):
        return "protocol"
    if re.search(r"runinfo|run_info|run_manifest|seed_manifest|^manifest|analysis_manifest|sha256_manifest|^run_log|"
                 r"^run_events", name):
        return "runinfo"
    if name.endswith((".png", ".pdf", ".svg")):
        return "figure"
    if name.endswith(".py"):
        return "code"
    if name.endswith(".md"):
        return "report"
    if name.endswith((".csv", ".json", ".jsonl", ".txt", ".npy")):
        return "result"
    return "other"


# --------------------------------------------------------------------------------------
# 項目定義
# --------------------------------------------------------------------------------------
FIELDS: list[tuple[str, str, str]] = [
    # (group, field, 日本語ラベル)
    ("A", "experiment_name", "実験名"),
    ("A", "research_question", "Research Question"),
    ("A", "hypothesis", "検証した仮説"),
    ("A", "experiment_type", "実験の種類"),
    ("A", "comparison_targets", "比較対象"),
    ("A", "changed_conditions", "変更した条件（とそれ以外の固定条件）"),
    ("B", "data_kind", "人工データ / 実データ"),
    ("B", "data_generator", "データ生成器（または実データの前処理）"),
    ("B", "n", "ノード数 n"),
    ("B", "d", "属性次元 d"),
    ("B", "k_true", "生成に使った潜在次元 K_TRUE"),
    ("B", "family_x", "X の分布族"),
    ("B", "family_y", "Y の分布族"),
    ("B", "dist_params", "各分布のパラメータ"),
    ("B", "f_generation", "F の生成方法・信号強度"),
    ("B", "w_w0", "w・w0"),
    ("B", "gaussian_variance", "Gaussian の分散"),
    ("B", "normalization", "正規化・中心化・clip"),
    ("B", "mask_missing", "欠損・観測 mask"),
    ("B", "split", "train / validation / test 分割"),
    ("B", "data_seed", "データ生成 seed"),
    ("C", "lineage", "実装系列"),
    ("C", "model_class", "モデル class"),
    ("C", "impl_commit", "実装 commit（run 時の git head）"),
    ("C", "k_candidates", "推定した K・候補 K"),
    ("C", "mcem_iters", "MCEM の反復数"),
    ("C", "mc_samples_L", "E-step の MC サンプル数 L"),
    ("C", "newton", "Newton 法の設定"),
    ("C", "init", "初期値の設定"),
    ("C", "init_seed", "初期値 seed"),
    ("C", "mc_seed", "MC seed"),
    ("C", "retry_failfast", "retry・fail-fast"),
    ("C", "convergence", "収束判定"),
    ("C", "numerics_mode", "正規化（numerics_mode）の扱い"),
    ("C", "dispersion", "分散パラメータの扱い"),
    ("C", "failure_handling", "数値的失敗時の処理"),
    ("D", "primary_metric", "Primary Metric"),
    ("D", "secondary_metrics", "Secondary Metrics"),
    ("D", "comparator", "比較対象"),
    ("D", "paired", "対応比較 / 独立比較"),
    ("D", "unit", "独立な実験単位"),
    ("D", "replicates", "trial・replicate・初期値の数"),
    ("D", "aggregation", "集計方法"),
    ("D", "uncertainty", "不確実性の評価"),
    ("D", "incomplete_handling", "欠測・INCOMPLETE の扱い"),
    ("D", "k_selection_rule", "K 選択の基準と選択規則"),
    ("E", "n_fits", "実行した fit 数"),
    ("E", "success_fail", "成功・失敗・中断"),
    ("E", "main_result", "主要な結果"),
    ("E", "hypothesis_consistency", "仮説との整合"),
    ("E", "worse_or_null", "悪化・差なしの結果"),
    ("E", "fairness", "比較の公平性"),
    ("E", "learned", "分かったこと"),
    ("E", "not_claimable", "主張できないこと・限界"),
    ("E", "result_csv", "結果 CSV"),
    ("E", "runinfo", "runinfo"),
    ("E", "protocol", "protocol"),
    ("E", "figure", "図"),
    ("E", "code", "実行コード"),
    ("E", "commit", "結果の commit SHA"),
]
FIELD_LABEL = {f: lab for _, f, lab in FIELDS}
FIELD_GROUP = {f: g for g, f, _ in FIELDS}

# JSON / CSV のキー名 → 項目（葉のキー名で照合、大文字小文字を区別しない）
KEYMAP: dict[str, list[str]] = {
    "n": ["n", "n_nodes", "n_values", "n_grid", "n_list", "ns"],
    "d": ["d", "d_values", "d_grid", "n_attr", "d_list", "ds"],
    "k_true": ["k_true", "k_gen", "kgen", "k_true_grid", "new_k_true", "k_true_values", "k_true_list", "true_k"],
    "family_x": ["family_x", "family_x_list", "family_x_true", "families_x", "x_family", "x_families", "fam_x", "xfam"],
    "family_y": ["family_y", "y_family", "family_y_true", "fam_y", "yfam"],
    "w_w0": ["w0", "w0_true", "w_true", "w_y", "weak_w", "strong_w", "w0_y", "true_w", "true_w0", "w_k"],
    "f_generation": ["f_scale", "signal", "signal_strength", "f_sd", "f_std", "signal_rule", "f_generation", "loading_scale"],
    "gaussian_variance": ["sigma_x_var", "sigma_x", "sigma_y", "sigma_y_true", "var_x", "gaussian_var", "x_variance",
                          "x_var", "sigma2", "variance"],
    "data_generator": ["data_generator", "generator", "generator_version", "dataset", "data_source", "x_y"],
    "split": ["test_ratio", "split_rule", "test_ratio_of_all_upper_pairs", "split", "heldout_rule", "split_seed_base",
              "split_seed_formula"],
    "mask_missing": ["y_obs_rate", "sparse", "regimes_share_of_training_candidate_pool", "mask", "mask_rule",
                     "y_mask", "observed_rate"],
    "data_seed": ["data_seed", "data_seed_base", "data_seeds", "data_seed_formula", "seed_base", "seeds", "replicates",
                  "seed_rule", "dataset_seed"],
    "init_seed": ["init_seed", "model_seed", "model_seed_base", "model_seed_formula", "start_seed", "search_seed",
                  "refit_seed", "init_seeds"],
    "mc_seed": ["mc_seed", "mc_seed_base", "rng_seed", "mc_seeds", "qlap_seed_base"],
    "impl_commit": ["git_head", "git_sha", "git_commit", "run_code_sha", "code_sha", "run_git_head", "protocol_git_head",
                    "git_head_at_start", "git_head_at_end", "code_git_head", "head_sha", "commit_sha", "runinfo_written_at_git_head"],
    "code": ["script", "command", "runner", "requested_command", "argv", "entrypoint", "runner_script"],
    "numerics_mode": ["numerics_mode"],
    "failure_handling": ["on_nan", "failure_policy", "on_failure", "nan_policy", "nan_count"],
    "retry_failfast": ["retry_count", "fail_fast", "retries", "max_retries", "seed_rescue_count",
                       "tolerance_relaxations", "retry"],
    "convergence": ["convergence", "convergence_rule", "stopping", "adaptive_stopping", "stopping_rule", "grad_tol",
                    "convergence_grad_inf_tol", "gtol"],
    "mcem_iters": ["num_iter", "em_iter", "em_iters", "em_iterations", "n_em_iter", "mcem_iter",
                   "mcem_iters", "max_em_iter"],
    "mc_samples_L": ["l", "n_mc", "mc_samples", "n_samples_e", "num_samples", "l_samples"],
    "k_candidates": ["k_candidates", "candidate_k", "k_grid", "k_range", "k_values", "k_est_grid", "candidate_ks",
                     "k_list", "k_est"],
    "lineage": ["lineage"],
    "model_class": ["model_class", "model", "model_cls", "runner_version", "em"],
    "newton": ["newton", "newton_steps", "newton_iter", "newton_max_iter", "e_step"],
    "init": ["init", "initialisations", "initialization", "init_method", "starts", "n_starts", "start"],
    "dispersion": ["dispersion", "sigma_y_mode", "variance_estimation"],
    "primary_metric": ["primary", "primary_metric", "primary_endpoint", "k_criterion", "criteria", "primary_outcome"],
    "secondary_metrics": ["secondary", "secondary_metrics", "metrics"],
    "hypothesis": ["hypothesis", "hypotheses", "h1", "research_question", "estimand", "question"],
    "aggregation": ["selector", "aggregation", "inference", "statistical_test", "analysis"],
    "incomplete_handling": ["incomplete", "incomplete_rule", "missing_handling", "failure_rule", "stop_rules"],
    "k_selection_rule": ["selection_rule", "k_selection", "selector", "tie_tolerance", "target"],
    "replicates": ["replicates_by_k_true", "n_replicates", "replicate", "n_rep", "n_datasets", "reps", "trials",
                   "n_trials", "replicates"],
    "n_fits": ["n_jobs_frozen", "main_fits_rows", "n_fits", "expected_fits", "fits", "n_fit", "expected", "actual"],
    "success_fail": ["main_ok", "main_failed", "n_ok", "n_failed", "lap_status_counts", "failed", "ok", "status_counts"],
}
KEY2FIELD: dict[str, str] = {}
for fld, keys in KEYMAP.items():
    for k in keys:
        KEY2FIELD.setdefault(k.lower(), fld)

SEED_COL = re.compile(r"seed", re.I)


def flatten(o, prefix="", depth=0, out=None):
    if out is None:
        out = []
    if depth > 5:
        return out
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.append((str(k), p, v))
            flatten(v, p, depth + 1, out)
    elif isinstance(o, list) and o and isinstance(o[0], dict) and depth < 3:
        for i, v in enumerate(o[:3]):
            flatten(v, f"{prefix}[{i}]", depth + 1, out)
    return out


def fmt(v, limit=160) -> str:
    if isinstance(v, (dict, list)):
        s = json.dumps(v, ensure_ascii=False, sort_keys=False)
    else:
        s = str(v)
    s = s.replace("\n", " ").replace("\r", " ")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def extract_json(blob: bytes) -> list[tuple[str, str, str]]:
    """(field, keypath, value) を返す。"""
    try:
        d = json.loads(blob.decode("utf-8"))
    except Exception:
        return []
    res = []
    pyfiles = []
    for leaf, path, v in flatten(d):
        for cand in [leaf] + ([v] if isinstance(v, str) else []):
            for m in re.findall(r"((?:expfam|tools|reproduction)[/\\][\w/\\.\-]+\.py)", cand):
                m = m.replace("\\", "/")
                if m not in pyfiles:
                    pyfiles.append(m)
        fld = KEY2FIELD.get(leaf.lower())
        if not fld:
            continue
        if isinstance(v, dict) and fld not in ("data_seed", "init_seed", "w_w0", "family_x", "primary_metric",
                                               "hypothesis", "k_selection_rule", "incomplete_handling",
                                               "replicates", "success_fail", "convergence", "model_class",
                                               "newton", "init", "mask_missing", "aggregation"):
            continue
        if v in (None, "", [], {}):
            continue
        res.append((fld, path, fmt(v)))
    for pf in pyfiles:
        res.append(("code", "（JSON 内のコードパス）", pf))
        if re.search(r"data_generator|generator|generate", pf):
            res.append(("data_generator", "（JSON 内のコードパス）", pf))
    return res


def kv_csv_to_json(blob: bytes) -> bytes | None:
    """「項目名, 値」の 2 列 CSV（値に JSON を含む protocol 形式）を JSON に変換する。該当しなければ None。"""
    try:
        text = blob.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or len(rows[0]) != 2 or rows[0][0].strip().lower() not in ("section", "key", "parameter", "field",
                                                                            "name", "item"):
        return None
    d = OrderedDict()
    for r in rows[1:]:
        if len(r) != 2:
            continue
        try:
            d[r[0]] = json.loads(r[1])
        except Exception:
            d[r[0]] = r[1]
    return json.dumps(d, ensure_ascii=False).encode("utf-8")


PER_FIT_ONLY = {"n": "n", "d": "d", "k": "k_candidates", "k_est": "k_candidates", "k_fit": "k_candidates",
                "status": "success_fail", "true_w0": "w_w0", "true_w": "w_w0", "true_f_scale": "f_generation",
                "num_iter": "mcem_iters", "l": "mc_samples_L"}


def extract_csv(blob: bytes, rowfilter: tuple[str, str] | None, max_rows=200000):
    """runinfo / 結果 CSV から (field, column, distinct values) を返す。seed 列も拾う。
    n・d・k・status のような曖昧な列名は、seed 列を持つ fit 単位の CSV でだけ採用する
    （集計 CSV の n はデータ数のことがあるため）。"""
    try:
        text = blob.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = blob.decode("cp932", "replace")
    rdr = csv.DictReader(io.StringIO(text))
    cols = rdr.fieldnames or []
    want = {}
    per_fit = any(SEED_COL.search(c) for c in cols)
    for c in cols:
        low0 = c.strip().lower()
        if low0 in PER_FIT_ONLY:
            if per_fit:
                want[c] = PER_FIT_ONLY[low0]
            continue
        fld = KEY2FIELD.get(low0)
        if fld:
            want[c] = fld
        elif SEED_COL.search(c):
            low = c.lower()
            want[c] = "init_seed" if ("model" in low or "init" in low or "start" in low) else (
                "mc_seed" if "mc" in low else "data_seed")
    if not want:
        return cols, []
    vals: dict[str, OrderedDict] = {c: OrderedDict() for c in want}
    for i, row in enumerate(rdr):
        if i >= max_rows:
            break
        if rowfilter and row.get(rowfilter[0]) != rowfilter[1]:
            continue
        for c in want:
            v = (row.get(c) or "").strip()
            if v and len(vals[c]) < 400:
                vals[c][v] = None
    res = []
    for c, fld in want.items():
        vs = list(vals[c])
        if not vs:
            continue
        if len(vs) > 6:
            desc = f"{len(vs)} 種類（{', '.join(vs[:3])} …）"
        else:
            desc = ", ".join(vs)
        res.append((fld, c, fmt(desc)))
    return cols, res


# --------------------------------------------------------------------------------------
# 台帳テキストからの抽出（DOCUMENTED）
# --------------------------------------------------------------------------------------
TXT_PATTERNS = {
    "n": r"(?<![A-Za-z_])n\s*[=∈]\s*([\{\[]?[0-9][0-9,．.\s、…]*[\}\]]?)",
    "d": r"(?<![A-Za-z_])d\s*[=∈]\s*([\{\[]?[0-9][0-9,\s、…]*[\}\]]?)",
    "k_true": r"(?:K_TRUE|K_GEN|k_true|K_true|真の K)\s*[=∈]?\s*([\{\[]?[0-9][0-9,\s、…]*[\}\]]?)",
    "mc_samples_L": r"(?<![A-Za-z_])L\s*=\s*([0-9]+)",
    "mcem_iters": r"([0-9]+)\s*反復|num_iter\s*=\s*([0-9]+)|max_iter\s*=\s*([0-9]+)",
    "k_candidates": r"(?:候補 ?K|candidate K|K\s*∈|K 1\.\.|K=1\.\.)\s*[=:]?\s*([\{\[]?[0-9][0-9,\s\.…]*[\}\]]?)",
    "family_x": r"((?:Gaussian|Bernoulli|Poisson|G3/B6/P3|per-column)[^、,]*?-X|[Xx]\s*=\s*(?:Gaussian|Bernoulli|Poisson)|family_x\s*=\s*\w+)",
    "family_y": r"((?:Gaussian|Bernoulli|Poisson)-Y|[Yy]\s*=\s*(?:Gaussian|Bernoulli|Poisson)|family_y\s*=\s*\w+)",
    "data_seed": r"(seed[^、。|]{0,60})",
    "replicates": r"([0-9]+\s*(?:trial|rep(?:licate)?|データ|split|seed)[^、。|]{0,30})",
}


def text_extract(field: str, text: str) -> str | None:
    pat = TXT_PATTERNS.get(field)
    if not pat or not text:
        return None
    hits = []
    for m in re.finditer(pat, text):
        g = next((x for x in m.groups() if x), m.group(0))
        g = g.strip().rstrip("、,")
        if g and g not in hits:
            hits.append(g)
        if len(hits) >= 4:
            break
    return "; ".join(hits) if hits else None


def blank(v: str | None) -> bool:
    return v is None or v.strip() in ("", "—", "-", "未確認", "n/a")


# --------------------------------------------------------------------------------------
# registry の該当行（DOCUMENTED）
# --------------------------------------------------------------------------------------
_reg_cache: dict[str, list[str]] = {}


def registry_rows(sha: str) -> list[str]:
    if sha not in _reg_cache:
        b = show(sha, "EXPERIMENT_REGISTRY.md")
        txt = b.decode("utf-8", "replace") if b else ""
        _reg_cache[sha] = [ln for ln in txt.splitlines() if ln.startswith("|") and ln.count("|") >= 6]
    return _reg_cache[sha]


def registry_match(sha: str, roots: list[str]) -> list[str]:
    keys = []
    for r in roots:
        r = r.split("#", 1)[0].rstrip("/*")
        r = re.sub(r"\*.*$", "", r).rstrip("/_")
        if r.count("/") >= 2 and len(r) > 20:
            keys.append(r)
    rows = []
    for ln in registry_rows(sha):
        if any(k in ln for k in keys):
            rows.append(ln)
    return rows


REG_STATUS = re.compile(r"\|\s*(current_main|current_support|fixed_support|old|archive|ai_generated|unverified)\s*\|")


# --------------------------------------------------------------------------------------
# 種類（record_kind）と NOT_APPLICABLE 規則
# --------------------------------------------------------------------------------------

def record_kind(row: dict) -> str:
    t, eid = row["type"], row["experiment_id"]
    if eid == "TG-DIAG":
        return "analysis"
    if eid == "TG-CORR":
        return "correction"
    if t == "文献" or eid == "C-NIPS12":
        return "literature"
    if eid == "B-ARCHIVE":
        return "archive"
    if t in ("理論監査", "理論", "理論（恒等式 39 項目）"):
        return "theory"
    if t in ("コード監査", "文書監査"):
        return "audit"
    if t == "設計・実装":
        return "design"
    if t.startswith("実データ"):
        return "experiment_real" if "診断" not in t else "analysis_real"
    if t in ("読取専用",):
        return "analysis_real"
    if any(s in t for s in ("再評価", "再採点", "事後解析", "既存資料の再解析", "診断（EM 0）")):
        return "analysis"
    if eid == "F-8B-A1":
        return "experiment_stopped"
    if re.search(r"EM 0", row["run_count"]):
        return "analysis"
    return "experiment"


# 新しいデータ・fit を作らない分析の親実験（データ条件はこの行で確認する）
PARENTS = {
    "TG-DIAG": "TG-A;TG-B", "TG-CORR": "TG-A;TG-B", "F-KEXP2-CLAP": "F-KEXP2-MAIN", "F-KEXP2-A": "F-KEXP2-MAIN",
    "F-KEXP2-B": "F-KEXP2-MAIN", "F-CLAP-REEVAL": "F-EMCONV", "E-AF7": "E-AF6", "E-AF8V3": "E-AF6",
    "C-WINE-AUDIT": "C-WINE",
}
INHERITED = {"n", "d", "k_true", "family_x", "family_y"}


def involves_new_fits(row: dict, kind: str) -> str:
    rc = row["run_count"]
    if kind in ("literature", "theory", "audit", "design", "correction", "analysis", "analysis_real"):
        if re.search(r"(?<![0-9,])[1-9][0-9,]*\s*fit", rc) and not re.search(r"EM 0|MCEM 0", rc):
            return "yes"
        if kind == "analysis" and "pilot" in rc.lower():
            return "yes"
        return "no"
    if kind == "archive":
        return "not_counted"
    if re.search(r"EM 0|MCEM 0|^0", rc):
        return "no"
    if blank(rc) or rc.startswith("未確認"):
        return "unknown"
    return "yes"


DATA_GEN_FIELDS = {"data_generator", "n", "d", "k_true", "family_x", "family_y", "dist_params", "f_generation",
                   "w_w0", "gaussian_variance", "normalization", "mask_missing", "split", "data_seed"}
SYN_ONLY = {"k_true", "f_generation", "w_w0", "dist_params", "data_seed"}
EST_FIELDS = {f for g, f, _ in FIELDS if g == "C"} - {"lineage"}
EVAL_FIELDS = {f for g, f, _ in FIELDS if g == "D"}
RUN_E_FIELDS = {"n_fits", "success_fail", "runinfo", "figure", "hypothesis_consistency", "worse_or_null"}


def na_reason(kind: str, field: str, fam_text: str) -> str | None:
    if kind in ("literature", "theory", "audit", "archive"):
        if field in DATA_GEN_FIELDS | EST_FIELDS | EVAL_FIELDS | RUN_E_FIELDS | {"result_csv", "protocol", "code",
                                                                                 "hypothesis", "comparison_targets",
                                                                                 "changed_conditions"}:
            if kind == "archive":
                return "初期実装・AI 生成の archive（OLD_NOT_ADOPTED）。条件の確定は対象外"
            return f"{kind} の記録（fit・データ生成なし）"
        if field == "lineage" and kind in ("literature", "theory"):
            return f"{kind} の記録"
    if kind == "design":
        if field in DATA_GEN_FIELDS | EVAL_FIELDS | RUN_E_FIELDS | {"result_csv", "main_result", "fairness",
                                                                    "learned"} or field in EST_FIELDS - {"impl_commit"}:
            return "設計・実装のみ（fit 0）"
    if kind in ("analysis", "correction", "analysis_real"):
        if field in DATA_GEN_FIELDS - {"n", "d", "k_true", "family_x", "family_y"}:
            return "新しいデータ生成なし（親実験の保存状態・データを再利用）"
        if field in EST_FIELDS - {"impl_commit"}:
            return "新しい fit なし（推定条件は親実験の行を参照）"
        if field in ("figure",):
            return None
    if kind in ("experiment_real", "analysis_real"):
        if field in SYN_ONLY:
            return "実データ（生成パラメータなし）"
    if field == "gaussian_variance" and fam_text and "gauss" not in fam_text.lower() and "G3" not in fam_text \
            and "per-column" not in fam_text.lower():
        return f"Gaussian を含まない（分布族: {fam_text[:60]}）"
    if field == "dispersion" and fam_text and "gauss" not in fam_text.lower() and "G3" not in fam_text:
        return f"分散パラメータを持つ分布族を含まない（{fam_text[:60]}）"
    return None


# --------------------------------------------------------------------------------------
# メイン処理
# --------------------------------------------------------------------------------------

def load_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def where_sha(row: dict) -> str:
    eid = row["experiment_id"]
    if eid in SHA_OVERRIDES:
        return SHA_OVERRIDES[eid]
    m = re.search(r"@([0-9a-f]{7,40})", row["where"])
    if m:
        return m.group(1)
    if re.fullmatch(r"[0-9a-f]{7,40}", row["commit"]):
        return row["commit"]
    return MAIN_SHA


def build():
    inv = load_csv(DATA / "experiment_inventory.csv")
    add = load_csv(DATA / "addendum_20261127_inventory.csv")
    pri = {r["result_id"]: r for r in load_csv(DATA / "primary_results_index.csv")}
    pri.update({r["result_id"]: r for r in load_csv(DATA / "addendum_20261127_primary_results.csv")})
    fair = load_csv(DATA / "comparison_fairness_matrix.csv") + load_csv(DATA / "addendum_20261127_fairness.csv")

    records = []
    for r in inv:
        r = dict(r)
        r["_ledger"] = "experiment_inventory.csv（88 件）"
        r["_counts"] = "yes（既存 88 件の 1 件）"
        records.append(r)
    for r in add:
        r = dict(r)
        r["_ledger"] = "addendum_20261127_inventory.csv（追補）"
        r["_counts"] = r["counts_as_new_experiment"]
        records.append(r)

    cond_rows, prov_rows, gap_rows = [], [], []

    def gap(eid, field, current, kind, severity, source, resolvable, action, result):
        gap_rows.append(OrderedDict([
            ("gap_id", ""), ("experiment_id", eid), ("field", field), ("current_record", current),
            ("issue_type", kind), ("severity", severity), ("primary_source", source),
            ("resolvable", resolvable), ("action_needed", action), ("check_result", result)]))

    for r in records:
        eid = r["experiment_id"]
        kind = record_kind(r)
        newfits = involves_new_fits(r, kind)
        sha = where_sha(r)
        sha_ok = sha_exists(sha)
        commit = r["commit"] if re.fullmatch(r"[0-9a-f]{7,40}", r["commit"] or "") else ""
        commit_ok = sha_exists(commit) if commit else False
        roots = ROOT_OVERRIDES.get(eid) or parse_evidence(r["evidence"])
        files = match_roots(ls(sha), roots) if sha_ok else []
        by_cls: dict[str, list[str]] = {}
        for f in files:
            by_cls.setdefault(classify(f), []).append(f)

        # ---- 一次資料からの抽出 ----
        verified: dict[str, list[tuple[str, str, str]]] = {}  # field -> [(value, source, keypath)]
        child: dict[str, dict[str, tuple[str, str]]] = OrderedDict()  # source file -> field -> (value,keypath)
        scan = [f for f in by_cls.get("runinfo", []) + by_cls.get("protocol", []) if f.endswith((".json", ".csv"))]
        scan = scan[:40]
        # 実験単位の runinfo が無いとき、fit 単位の記録（fits/*.json）を 3 件だけ読む
        per_fit_json = [f for f in by_cls.get("per_fit", []) if f.endswith(".json")]
        if not [f for f in by_cls.get("runinfo", []) if f.endswith((".json", ".csv"))] and per_fit_json:
            scan += per_fit_json[:3]
        for f in scan:
            b = show(sha, f)
            if b is None:
                continue
            if f.endswith(".json"):
                items = extract_json(b)
            else:
                kv = kv_csv_to_json(b)
                items = extract_json(kv) if kv is not None else extract_csv(b, row_filter_for(roots, f))[1]
            for fld, kp, val in items:
                verified.setdefault(fld, []).append((val, f, kp))
                child.setdefault(f, OrderedDict()).setdefault(fld, (val, kp))
        # 結果 CSV の seed 列・条件列（ヘッダと値）
        seed_cols_found = []
        for f in by_cls.get("result", [])[:60]:
            if not f.endswith(".csv"):
                continue
            b = show(sha, f, max_bytes=3_000_000)
            if b is None:
                continue
            cols, items = extract_csv(b, row_filter_for(roots, f))
            for fld, kp, val in items:
                if fld in ("data_seed", "init_seed", "mc_seed", "n", "d", "k_true", "family_x", "family_y",
                           "mc_samples_L", "k_candidates", "mcem_iters", "success_fail", "retry_failfast",
                           "w_w0", "f_generation", "failure_handling"):
                    verified.setdefault(fld, []).append((val, f, kp))
                    if fld.endswith("seed"):
                        seed_cols_found.append(f"{f.rsplit('/', 1)[-1]}:{kp}")

        # registry
        reg = registry_match(sha, roots) if sha_ok else []
        reg_txt = " ".join(reg)
        reg_scripts = sorted(set(re.findall(r"`((?:expfam|tools|reproduction|reports)/[^`\s]+?\.py)`", reg_txt)))
        reg_scripts += sorted(set("expfam/" + s if not s.startswith(("expfam/", "tools/", "reproduction/")) else s
                                  for s in re.findall(r"`([A-Za-z0-9_]+\.py)`", reg_txt)))
        reg_reports = sorted(set(re.findall(r"(reports/[^\s`|）)]+?\.md)", reg_txt)))
        reg_status = sorted(set(m.group(1) for ln in reg for m in REG_STATUS.finditer(ln)))

        # runner の特定
        runner_candidates = []
        for val, src, kp in verified.get("code", []):
            for p in re.findall(r"((?:expfam|tools|reproduction)/[\w/.\-]+\.py)", val):
                runner_candidates.append((p, src))
        for p in reg_scripts:
            runner_candidates.append((p, "EXPERIMENT_REGISTRY.md@" + short(sha)))
        for p, s, why in MANUAL_RUNNERS.get(eid, []):
            runner_candidates.append((p, why))
        runner_checked = []
        allfiles = set(ls(sha)) if sha_ok else set()
        runner_candidates = [(p, s) for p, s in runner_candidates if not re.search(r"[*{]", p)]
        for p, src in runner_candidates:
            exists = p in allfiles
            if not exists:  # 短縮名（例 exp_run_scenario_A.py）を tree 内で探す
                base = p.rsplit("/", 1)[-1]
                alt = [f for f in allfiles if f.endswith("/" + base)]
                if len(alt) == 1:
                    p, exists = alt[0], True
            runner_checked.append((p, exists, src))
        runner_checked = list(OrderedDict(((p, e), s) for p, e, s in runner_checked).items())

        # generator
        gen_vals = verified.get("data_generator", [])

        # git head（run 時）と存在確認
        heads = []
        for val, src, kp in verified.get("impl_commit", []):
            for h in re.findall(r"\b[0-9a-f]{7,40}\b", val):
                heads.append((h, src, kp))
        head_status = [(h, sha_exists(h), src, kp) for h, src, kp in heads]

        fam_text = " ".join(v for v, _, _ in verified.get("family_x", []) + verified.get("family_y", []))
        fam_text = fam_text or (r["conditions"] + " " + r["lineage"])

        result_ids = [x for x in re.split(r"[;；]\s*", r.get("result_ids", "")) if x]
        res_status = Counter(pri[x]["status"] for x in result_ids if x in pri)
        fair_rows = [f for f in fair if eid in re.split(r"[;、,\s]+", f["experiments"])]

        # ---- 項目ごとの値と状態 ----
        def emit(field, value, status, source="", extraction="", note="", cond="ALL", parent=""):
            if cond == "ALL" and (eid, field) in MANUAL_VALUES:
                mv = MANUAL_VALUES[(eid, field)]
                auto = f"自動抽出: {status} {fmt(value, 80)}" if status != "MISSING" else "自動抽出: MISSING"
                value, status, source, extraction = mv[0], mv[1], mv[2], "人手で一次資料を確認"
                note = (mv[3] + "；" if mv[3] else "") + auto
            if cond == "ALL" and status == "MISSING" and field in NO_STRUCTURED_FIELD:
                note = note or "構造化された記録（runinfo・protocol JSON・台帳）に専用の欄が無い。報告書本文は自動照合の対象外"
            if cond == "ALL" and status == "MISSING" and field in INHERITED and eid in PARENTS \
                    and kind in ("analysis", "correction", "analysis_real"):
                status, note = "NOT_APPLICABLE", f"新しいデータを作らない分析。条件は親実験 {PARENTS[eid]} の行で確認する"
            cond_rows.append(OrderedDict([
                ("record_id", eid if cond == "ALL" else f"{eid}::{cond}"), ("experiment_id", eid),
                ("parent_id", parent if cond != "ALL" else PARENTS.get(eid, "")), ("condition_label", cond), ("record_kind", kind),
                ("involves_new_fits", newfits), ("group", FIELD_GROUP[field]), ("field", field),
                ("field_label", FIELD_LABEL[field]), ("value", value), ("status", status),
                ("source", source), ("source_sha", short(sha) if source and "@" not in source else ""),
                ("extraction", extraction), ("note", note)]))

        def from_verified(field):
            vs = verified.get(field)
            if not vs:
                return None
            distinct = list(OrderedDict((v, None) for v, _, _ in vs))
            srcs = list(OrderedDict((s, None) for _, s, _ in vs))
            kps = list(OrderedDict((k, None) for _, _, k in vs))
            val = " | ".join(distinct[:4]) + (f" | …（ほか {len(distinct) - 4} 値）" if len(distinct) > 4 else "")
            src = "; ".join(srcs[:3]) + (f"; …（ほか {len(srcs) - 3} ファイル）" if len(srcs) > 3 else "")
            return val, src, "; ".join(kps[:4])

        inv_map = {
            "experiment_name": r["name"], "research_question": r["question"], "experiment_type": r["type"],
            "comparison_targets": r["comparators"], "changed_conditions": r["conditions"],
            "lineage": r["lineage"], "primary_metric": r["metrics"], "comparator": r["comparators"],
            "n_fits": r["run_count"], "success_fail": r["outcome_status"], "main_result": r["primary_result"],
            "uncertainty": r["uncertainty"], "fairness": r["comparability"], "not_claimable": r["limitations"],
            "learned": r["primary_result"], "commit": commit,
        }
        inv_src = r["_ledger"].split("（")[0]

        for g, field, lab in FIELDS:
            v = from_verified(field)
            # --- 一次資料の値がある ---
            if field in ("result_csv", "runinfo", "protocol", "figure", "code", "commit", "experiment_name",
                         "research_question", "experiment_type", "main_result", "fairness", "learned",
                         "not_claimable", "data_kind", "lineage"):
                v = None if field not in ("lineage",) else v
            if field == "data_kind":
                dk = {"experiment_real": "実データ", "analysis_real": "実データ（診断・読取）"}.get(kind)
                if kind in ("literature", "theory", "audit", "archive", "design"):
                    emit(field, "—", "NOT_APPLICABLE", note=f"{kind} の記録")
                else:
                    emit(field, dk or ("人工データ" if kind.startswith("experiment") else
                                       "親実験のデータ（人工）" if "C-" not in eid else "親実験のデータ（実データ）"),
                         "DOCUMENTED", inv_src, "type 列", f"type={r['type']}")
                continue
            if field == "result_csv":
                csvs = [f for f in by_cls.get("result", []) if f.endswith(".csv")]
                if na_reason(kind, field, fam_text):
                    emit(field, r["evidence"] or "—", "NOT_APPLICABLE", note=na_reason(kind, field, fam_text) or "")
                elif csvs:
                    emit(field, f"{len(csvs)} ファイル（例: {csvs[0]}）", "VERIFIED", f"{csvs[0]}",
                         "git ls-tree で存在確認", f"evidence 列: {r['evidence'][:120]}")
                elif files:
                    emit(field, r["evidence"], "DOCUMENTED", inv_src, "evidence 列",
                         f"CSV 以外のファイルのみ {len(files)} 件（{files[0]}）")
                else:
                    emit(field, r["evidence"] or "—", "MISSING", inv_src, "evidence 列",
                         f"固定 SHA {short(sha)} に evidence のパスが見つからない")
                continue
            if field in ("runinfo", "protocol", "figure"):
                cls = {"runinfo": "runinfo", "protocol": "protocol", "figure": "figure"}[field]
                fl = by_cls.get(cls, [])
                if field == "runinfo" and not fl and per_fit_json:
                    emit(field, f"実験単位の runinfo は無い。fit 単位の記録 {len(per_fit_json)} 件（例: {per_fit_json[0]}）",
                         "VERIFIED", per_fit_json[0], "git ls-tree・git show",
                         "fit ごとの JSON に seed・git head などが保存されている")
                    continue
                if field == "protocol":
                    fl = fl + [f for f in by_cls.get("report", []) if "protocol" in f.lower()]
                if fl:
                    emit(field, f"{len(fl)} ファイル（例: {fl[0]}）", "VERIFIED", fl[0], "git ls-tree で存在確認")
                elif na_reason(kind, field, fam_text):
                    emit(field, "—", "NOT_APPLICABLE", note=na_reason(kind, field, fam_text) or "")
                elif field == "protocol" and reg_reports:
                    emit(field, "; ".join(reg_reports[:2]), "DOCUMENTED", "EXPERIMENT_REGISTRY.md@" + short(sha),
                         "registry 行の reports/ リンク", "設計・報告書が registry にある（protocol ファイル単体ではない）")
                else:
                    emit(field, "—", "MISSING", note=f"探索範囲: {'; '.join(roots)[:150]}")
                continue
            if field == "code":
                if runner_checked:
                    ok = [p for (p, e), s in runner_checked if e]
                    ng = [p for (p, e), s in runner_checked if not e]
                    if ok:
                        emit(field, "; ".join(ok[:4]), "VERIFIED", ok[0], "runinfo の script/command または registry の"
                             "スクリプト名 → git ls-tree で存在確認",
                             ("未発見: " + "; ".join(ng[:3])) if ng else "")
                    else:
                        emit(field, "; ".join(ng[:4]), "DOCUMENTED", runner_checked[0][1],
                             "名前の記載のみ", f"固定 SHA {short(sha)} にファイルが見つからない")
                elif by_cls.get("code"):
                    emit(field, "; ".join(by_cls["code"][:3]), "VERIFIED", by_cls["code"][0], "git ls-tree")
                elif na_reason(kind, field, fam_text):
                    emit(field, "—", "NOT_APPLICABLE", note=na_reason(kind, field, fam_text) or "")
                else:
                    emit(field, "—", "MISSING", note="runinfo・registry・evidence のいずれからも runner を特定できない")
                continue
            if field == "commit":
                if commit and commit_ok:
                    br = remote_branches(commit)
                    emit(field, commit, "VERIFIED", f"git cat-file（{len(br)} remote branch に含まれる）", "",
                         f"ファイルは {r['where']} で読んだ" if short(sha) != commit[:7] else "")
                elif commit:
                    emit(field, commit, "CONFLICT", inv_src, "", "commit が repository に存在しない")
                elif kind in ("literature",):
                    emit(field, "—", "NOT_APPLICABLE", note="文献")
                else:
                    emit(field, r["commit"] or "—", "MISSING", inv_src)
                continue
            if field == "impl_commit":
                if head_status:
                    good = [h for h, ok, _, _ in head_status if ok]
                    bad = [h for h, ok, _, _ in head_status if not ok]
                    hs = list(OrderedDict((h[:12], None) for h, _, _, _ in head_status))
                    srcs = list(OrderedDict((s, None) for _, _, s, _ in head_status))
                    if good and not bad:
                        emit(field, " | ".join(hs[:4]), "VERIFIED", "; ".join(srcs[:2]),
                             "; ".join(OrderedDict((k, None) for _, _, _, k in head_status).keys())[:120],
                             "run 時の git head は repository に存在")
                    else:
                        emit(field, " | ".join(hs[:4]), "DOCUMENTED", "; ".join(srcs[:2]), "runinfo の git head",
                             f"repository に存在しない head: {', '.join(b[:12] for b in bad[:3])}（未 push・rebase の可能性。"
                             "記録はあるが一次照合できない）")
                elif na_reason(kind, field, fam_text):
                    emit(field, "—", "NOT_APPLICABLE", note=na_reason(kind, field, fam_text) or "")
                elif commit:
                    emit(field, commit, "DOCUMENTED", inv_src, "commit 列",
                         "run 時の git head は runinfo に無い。結果 commit で代用（実装 commit と同一とは限らない）")
                else:
                    emit(field, "—", "MISSING")
                continue
            if field == "data_generator":
                if gen_vals:
                    val, src, kp = from_verified("data_generator") or ("", "", "")
                    emit(field, val, "VERIFIED", src, kp)
                    continue
                gens = [p for (p, e), s in runner_checked if re.search(r"generat|simulat|dataset|scenario_lib", p)]
                if gens:
                    emit(field, "; ".join(gens[:3]), "DOCUMENTED", "EXPERIMENT_REGISTRY.md@" + short(sha),
                         "registry のスクリプト名", "生成器と runner の対応は名前から推定。runner 内で呼ばれる生成関数は未照合")
                    continue
            if field == "lineage":
                emit(field, r["lineage"], "DOCUMENTED" if not blank(r["lineage"]) else
                     ("NOT_APPLICABLE" if na_reason(kind, field, fam_text) else "MISSING"), inv_src, "lineage 列",
                     ("runinfo の lineage: " + v[0]) if v else "")
                continue
            if field == "main_result":
                if result_ids and res_status.get("VERIFIED"):
                    emit(field, r["primary_result"], "VERIFIED", "primary_results_index.csv / addendum_20261127_primary_results.csv",
                         "result_ids: " + ";".join(result_ids[:8]),
                         "状態内訳: " + ", ".join(f"{k} {n}" for k, n in sorted(res_status.items())))
                elif not blank(r["primary_result"]):
                    emit(field, r["primary_result"], "DOCUMENTED", inv_src, "primary_result 列",
                         "主要数値は 96 項目監査・追補 38 項目の対象外" if not result_ids else
                         "result_ids はあるが VERIFIED ではない: " + ", ".join(res_status))
                else:
                    emit(field, "—", "NOT_APPLICABLE" if kind in ("design", "archive") else "MISSING")
                continue
            if field == "fairness":
                if fair_rows:
                    emit(field, "; ".join(f"{f['comparison_id']}={f['grade']}" for f in fair_rows[:6]), "DOCUMENTED",
                         "comparison_fairness_matrix.csv / addendum_20261127_fairness.csv", "experiments 列",
                         r["comparability"][:120])
                elif not blank(r["comparability"]):
                    emit(field, r["comparability"], "DOCUMENTED", inv_src, "comparability 列")
                elif na_reason(kind, field, fam_text):
                    emit(field, "—", "NOT_APPLICABLE", note=na_reason(kind, field, fam_text) or "")
                else:
                    emit(field, "—", "MISSING")
                continue
            if field == "paired":
                if fair_rows:
                    emit(field, "; ".join(f"{f['comparison_id']}: same_data={f['same_data']}, same_seed={f['same_seed']}"
                                          for f in fair_rows[:3]), "DOCUMENTED", "comparison_fairness_matrix.csv", "")
                    continue
            # --- 一般の項目 ---
            if v:
                val, src, kp = v
                emit(field, val, "VERIFIED", src, kp)
                continue
            txtv = None
            if field in inv_map and not blank(inv_map[field]):
                txtv = (inv_map[field], inv_src, f"{field} ← 台帳の列")
            elif field in TXT_PATTERNS:
                t = text_extract(field, r["conditions"])
                if t:
                    txtv = (t, inv_src, "conditions 列（正規表現抽出）")
                else:
                    t = text_extract(field, reg_txt)
                    if t:
                        txtv = (t, "EXPERIMENT_REGISTRY.md@" + short(sha), "registry 行（正規表現抽出）")
            if txtv:
                emit(field, txtv[0], "DOCUMENTED", txtv[1], txtv[2])
                continue
            nr = na_reason(kind, field, fam_text)
            if nr:
                emit(field, "—", "NOT_APPLICABLE", note=nr)
            else:
                emit(field, "—", "MISSING")

        # 子行（条件別：一次資料ファイルごと）
        if len(child) > 1:
            for f, flds in list(child.items())[:16]:
                label = f.split("/", 3)[-1] if f.count("/") >= 3 else f
                for fld, (val, kp) in flds.items():
                    if fld not in FIELD_GROUP:
                        continue
                    emit(fld, val, "VERIFIED", f, kp, cond=label, parent=eid)

        # ---- provenance ----
        def j(lst, n=4):
            return "; ".join(lst[:n]) + (f"; …（計 {len(lst)}）" if len(lst) > n else "")
        tree_root = ""
        if roots:
            r0 = roots[0].split("#")[0]
            if "*" in r0:  # glob はリンクにできないので、その親ディレクトリを指す
                r0 = r0.split("*")[0].rsplit("/", 1)[0]
            tree_root = r0.rstrip("/")
            if sha_ok and git("cat-file", "-e", f"{sha}:{tree_root}").returncode != 0:
                tree_root = tree_root.rsplit("/", 1)[0]
        prov_rows.append(OrderedDict([
            ("experiment_id", eid), ("ledger", r["_ledger"]), ("counts_as_experiment_record", r["_counts"]),
            ("record_kind", kind), ("involves_new_fits", newfits), ("domain", r["domain"]),
            ("lineage", r["lineage"]), ("where", r["where"]), ("read_sha", short(sha)),
            ("read_sha_exists", "yes" if sha_ok else "no"),
            ("read_sha_remote_branches", j(remote_branches(sha), 3) if sha_ok else ""),
            ("result_commit", commit), ("result_commit_exists", "yes" if commit_ok else ("n/a" if not commit else "no")),
            ("run_git_heads", j(list(OrderedDict((h[:12] + ("" if ok else "（repo に無い）"), None)
                                                  for h, ok, _, _ in head_status)), 4)),
            ("generator", j([v for v, _, _ in gen_vals], 2) if gen_vals else ""),
            ("runner", j([p + ("" if e else "（未発見）") for (p, e), s in runner_checked], 4)),
            ("protocol", j(by_cls.get("protocol", []), 3)),
            ("runinfo", j(by_cls.get("runinfo", []), 3)),
            ("result_files", f"{len(by_cls.get('result', []))}"),
            ("primary_evidence", r["evidence"]),
            ("primary_evidence_found", "yes" if files else ("n/a" if kind == "literature" and not roots else "no")),
            ("figures", j(by_cls.get("figure", []), 2)),
            ("reports", j(reg_reports + [f for f in by_cls.get("report", []) if "protocol" not in f.lower()], 3)),
            ("registry_rows", str(len(reg))), ("registry_status", ",".join(reg_status)),
            ("seed_columns_in_results", j(sorted(set(seed_cols_found)), 3)),
            ("github_tree", f"{REPO_URL}/tree/{short(sha)}/{tree_root}" if tree_root else ""),
            ("root_note", ROOT_NOTES.get(eid, "")),
        ]))

        # ---- gaps（自動） ----
        if ROOT_NOTES.get(eid) and "曖昧" not in ROOT_NOTES[eid] and "特定できない" in ROOT_NOTES[eid]:
            gap(eid, "evidence", r["evidence"], "LINK_AMBIGUOUS", "NOTATION", f"experiment_inventory.csv",
                "yes", "evidence 列に具体的なパスを書く（台帳の改訂は Human Gate）", ROOT_NOTES[eid])
        if not files and kind not in ("literature",) and roots:
            gap(eid, "evidence", r["evidence"], "LINK_BROKEN", "REPRODUCIBILITY", f"git ls-tree {short(sha)}",
                "unknown", "evidence の場所を特定する", f"固定 SHA {short(sha)} に該当ファイルが無い")
        for (p, e), s in runner_checked:
            if not e:
                gap(eid, "code", p, "LINK_BROKEN", "REPRODUCIBILITY", s, "unknown",
                    "runner の実パスを特定する", f"固定 SHA {short(sha)} に {p} が無い")
        for h, ok, src, kp in head_status:
            if not ok:
                gap(eid, "impl_commit", f"{h} ({kp})", "SHA_NOT_FOUND", "REPRODUCIBILITY", f"{src}@{short(sha)}",
                    "unknown", "その head を含む branch が残っているか確認する",
                    "runinfo の git head が local repository にも remote にも無い")

    # 後処理: 各記録の重要項目 MISSING を gaps に（REPRODUCIBILITY / INTERPRETATION）
    CRIT = {
        "data_seed": "REPRODUCIBILITY", "data_generator": "REPRODUCIBILITY", "code": "REPRODUCIBILITY",
        "impl_commit": "REPRODUCIBILITY", "mc_samples_L": "REPRODUCIBILITY", "mcem_iters": "REPRODUCIBILITY",
        "init_seed": "REPRODUCIBILITY", "n": "INTERPRETATION", "d": "INTERPRETATION", "k_true": "INTERPRETATION",
        "family_x": "INTERPRETATION", "family_y": "INTERPRETATION", "primary_metric": "INTERPRETATION",
        "n_fits": "INTERPRETATION", "result_csv": "REPRODUCIBILITY", "runinfo": "REPRODUCIBILITY",
        "protocol": "INTERPRETATION", "k_candidates": "INTERPRETATION",
    }
    for c in cond_rows:
        if c["condition_label"] != "ALL" or c["status"] != "MISSING" or c["field"] not in CRIT:
            continue
        gap(c["experiment_id"], c["field"], "—", "MISSING", CRIT[c["field"]],
            "固定 SHA の runinfo / protocol / 結果 CSV / registry / 台帳", "unknown",
            "当時の実行記録（ログ・runinfo・引数）を探す。見つからなければ「不明」のまま残す",
            f"{c['field_label']} を一次資料・台帳の両方で特定できなかった")

    # 既知の問題（手で確認した記録。値・SHA・出典を並べる）
    for g_ in MANUAL_GAPS:
        gap(*g_)

    for i, g_ in enumerate(gap_rows, 1):
        g_["gap_id"] = f"GAP-{i:03d}"

    # 記録品質の分類（研究結果の良し悪しとは独立。比較公平性 A〜D とも別物）
    inv_all = {r["experiment_id"]: r for r in records}
    st = {(c["experiment_id"], c["field"]): c["status"] for c in cond_rows if c["condition_label"] == "ALL"}
    blocking = {}
    for g_ in MANUAL_GAPS:
        if g_[3] == "CONFLICT" and g_[4] == "INTERPRETATION" and not str(g_[6]).startswith("no"):
            for e in g_[0].split(";"):
                blocking.setdefault(e, []).append(g_[1])
    for pr in prov_rows:
        eid = pr["experiment_id"]
        miss = [f for f in GRADE_FIELDS if st.get((eid, f)) == "MISSING"]
        doc = [f for f in GRADE_FIELDS if st.get((eid, f)) == "DOCUMENTED"]
        main_v = st.get((eid, "main_result")) == "VERIFIED"
        comp = inv_all[eid].get("completeness", "")
        if pr["record_kind"] == "archive" or comp in ("STOPPED", "OLD_NOT_ADOPTED") or eid == "G-LOCAL-ONLY" \
                or pr["primary_evidence_found"] != "yes":
            grade, why = "C", f"completeness={comp or '—'}・record_kind={pr['record_kind']}（主要な根拠に使わない記録）"
            if eid == "G-LOCAL-ONLY":
                grade, why = "C", "main に無い保存ブランチ上の報告書（統合は Human Gate）"
        elif len(miss) >= 4 and not main_v:
            grade, why = "C", f"主要数値が未照合で、主要条件の欠落 {len(miss)} 項目（{', '.join(miss[:5])}）"
        elif main_v and not miss and not blocking.get(eid) and comp != "NEEDS_LOCAL_NPZ":
            grade, why = "A", "主要数値が一次 CSV で照合済み、主要条件・runner・結果の所在が一次資料で確認できる" + (
                f"（DOCUMENTED {len(doc)} 項目: {', '.join(doc[:4])}）" if doc else "")
        else:
            parts = []
            if not main_v:
                parts.append("主要数値は未照合（台帳の記載のみ）")
            if miss:
                parts.append(f"主要条件の欠落 {len(miss)} 項目（{', '.join(miss[:5])}）")
            if blocking.get(eid):
                parts.append("訂正・食い違いあり: " + "; ".join(blocking[eid]))
            if comp == "NEEDS_LOCAL_NPZ":
                parts.append("再評価に repository 外の NPZ が必要")
            grade, why = "B", "、".join(parts) or "一部が DOCUMENTED"
        pr["record_quality_grade"] = grade
        pr["grade_reason"] = why
        pr["missing_key_fields"] = ";".join(miss)
    return cond_rows, prov_rows, gap_rows


GRADE_FIELDS = ["n", "d", "k_true", "family_x", "family_y", "data_seed", "init_seed", "mcem_iters", "mc_samples_L",
                "code", "runinfo", "result_csv", "impl_commit", "data_generator"]


NO_STRUCTURED_FIELD = {"hypothesis_consistency", "worse_or_null", "unit", "aggregation", "secondary_metrics",
                       "incomplete_handling", "k_selection_rule", "hypothesis", "normalization", "dist_params",
                       "newton", "dispersion", "convergence", "paired", "uncertainty"}

_FA = TG + "frozen_protocol_A.md@1ef9096"
_FB = TG + "frozen_protocol_B.md@1ef9096"
_OLD = "expfam/src/exp_scenario_lib.py@1b5030b"
_OLDN = ("runner の定数。結果 CSV と同じ commit 7c3040f（2026-04-27）で追加。run 時の git head・引数・seed は結果に"
         "保存されていない（CSV は trial 列のみ）。現行コードを根拠に実行条件を断定しない")
# 人手で一次資料を読んで確認した値（自動抽出より優先）。(exp, field) -> (value, status, source, note)
MANUAL_VALUES: dict[tuple[str, str], tuple[str, str, str, str]] = {
    ("TG-A", "n"): ("150", "VERIFIED", _FA, ""),
    ("TG-A", "d"): ("15", "VERIFIED", _FA, ""),
    ("TG-A", "k_true"): ("3", "VERIFIED", _FA, ""),
    ("TG-A", "family_x"): ("真: 全 15 列 Bernoulli ／ 推定: Bernoulli（正指定）と Poisson（誤指定）", "VERIFIED", _FA, ""),
    ("TG-A", "family_y"): ("Bernoulli", "VERIFIED", _FA, ""),
    ("TG-A", "normalization"): ("正規化・clip なし（canonical-clean-v1）", "VERIFIED", _FA, ""),
    ("TG-A", "split"): ("上三角ペアの 10%（1,118）を test として先に確保。split seed = data_seed + 700,000", "VERIFIED", _FA, ""),
    ("TG-A", "mask_missing"): ("dense = 学習 10,057 ペア全部 ／ sparse = その 10%（1,006）。抽出 seed = data_seed + 800,000",
                               "VERIFIED", _FA, ""),
    ("TG-A", "k_candidates"): ("K=3 のみ（K 選択なし）", "VERIFIED", _FA, ""),
    ("TG-A", "mcem_iters"): ("10", "VERIFIED", _FA, ""),
    ("TG-A", "mc_samples_L"): ("10", "VERIFIED", _FA, ""),
    ("TG-A", "init"): ("informed init（既存の規則）、2 初期値（t=0,1）", "VERIFIED", _FA, ""),
    ("TG-A", "model_class"): ("DualExpFamLSMConsistent（run_em_experimental）", "VERIFIED", _FA, ""),
    ("TG-A", "numerics_mode"): ("consistent", "VERIFIED", _FA, ""),
    ("TG-A", "retry_failfast"): ("fail_fast。retry・seed 置換・hidden repair なし", "VERIFIED", _FA, ""),
    ("TG-A", "hypothesis"): ("H_A: I > 0（sparse のほうが誤指定の悪影響が大きい）", "VERIFIED", _FA, ""),
    ("TG-A", "unit"): ("生成データ 20 個（初期値は単位ではない）", "VERIFIED", _FA, ""),
    ("TG-A", "aggregation"): ("2 初期値の RMSE_Z の算術平均 → Δ = Poisson − Bernoulli、I = Δ_sparse − Δ_dense", "VERIFIED", _FA, ""),
    ("TG-A", "uncertainty"): ("bootstrap 95%（10,000 回、データ単位）、t 区間、sign test（記述用）", "VERIFIED", _FA, ""),
    ("TG-A", "incomplete_handling"): ("FAILED は INCOMPLETE、分母は 20 のまま", "VERIFIED", _FA, ""),
    ("TG-A", "paired"): ("対応比較（4 条件で同じデータ・init/mc seed）", "VERIFIED", _FA, ""),
    ("TG-A", "replicates"): ("20 データ × 2 観測率 × 2 family × 2 初期値 = 160 fit", "VERIFIED", _FA, ""),
    ("TG-A", "secondary_metrics"): ("Z の L サンプル平均の RMSE_Z、gram 相関、test ペアの対数スコア", "VERIFIED", _FA, ""),
    ("TG-B", "n"): ("150", "VERIFIED", _FB, ""),
    ("TG-B", "d"): ("15", "VERIFIED", _FB, ""),
    ("TG-B", "k_true"): ("9", "VERIFIED", _FB, ""),
    ("TG-B", "family_x"): ("G01: Gaussian（分散 0.1）／GM: Gaussian（分散 = trace 一致の値、replicate ごと）／P: Poisson",
                           "VERIFIED", _FB, "条件ごとの値は 14 章に記載"),
    ("TG-B", "family_y"): ("Bernoulli", "VERIFIED", _FB, ""),
    ("TG-B", "gaussian_variance"): ("G01: 0.1 ／ GM: σ²_matched = tr(FᵀF)/tr(E_z[I_P])（replicate ごと）", "VERIFIED", _FB, ""),
    ("TG-B", "data_generator"): ("本実験用の新しい生成関数（Z ~ N(0,I)、build_full_rank_loadings）。K-exp2 のデータの再現ではない",
                                 "VERIFIED", _FB, ""),
    ("TG-B", "k_candidates"): ("1..10", "VERIFIED", _FB, ""),
    ("TG-B", "mcem_iters"): ("10", "VERIFIED", _FB, ""),
    ("TG-B", "mc_samples_L"): ("10", "VERIFIED", _FB, ""),
    ("TG-B", "model_class"): ("DualExpFamLSMConsistent（run_em_experimental）", "VERIFIED", _FB, ""),
    ("TG-B", "numerics_mode"): ("consistent", "VERIFIED", _FB, ""),
    ("TG-B", "retry_failfast"): ("fail_fast。再実行・seed 変更・hidden repair なし", "VERIFIED", _FB, ""),
    ("TG-B", "k_selection_rule"): ("データごとに 2 初期値の基準値の算術平均 → argmin。1e-9 以内の同点は小さい K", "VERIFIED", _FB, ""),
    ("TG-B", "incomplete_handling"): ("C_Lap は 20 評価のうち 1 つでも非 OK なら INCOMPLETE、分母 10", "VERIFIED", _FB, ""),
    ("TG-B", "primary_metric"): ("条件 × 基準ごとの正解・過小・過大・INCOMPLETE の数", "VERIFIED", _FB, ""),
    ("TG-B", "paired"): ("対応比較（3 条件で Z・F・Y と init/mc seed を共有）", "VERIFIED", _FB, ""),
    ("TG-B", "replicates"): ("3 条件 × 10 データ × 10 候補 K × 2 初期値 = 600 fit", "VERIFIED", _FB, ""),
    ("TG-B", "unit"): ("生成データ 10 個", "VERIFIED", _FB, ""),
    ("B-OLD05-EXP1-4", "n"): ("150（Exp2 は 50..300）", "DOCUMENTED", _OLD, _OLDN),
    ("B-OLD05-EXP1-4", "d"): ("15（Exp3 は 5..30）", "DOCUMENTED", _OLD, _OLDN),
    ("B-OLD05-EXP1-4", "k_true"): ("3", "DOCUMENTED", _OLD, _OLDN),
    ("B-OLD05-EXP1-4", "k_candidates"): ("Exp1: 1..6、Exp2–4: K=3 固定", "DOCUMENTED", _OLD, _OLDN),
    ("B-OLD05-EXP1-4", "mcem_iters"): ("8", "DOCUMENTED", _OLD, _OLDN),
    ("B-OLD05-EXP1-4", "mc_samples_L"): ("5", "DOCUMENTED", _OLD, _OLDN),
    ("B-OLD05-EXP1-4", "replicates"): ("10 trial / 条件", "DOCUMENTED", _OLD, _OLDN),
    ("B-OLD05-EXP1-4", "data_seed"): ("Exp1: 4000+off + 100·trial、Exp2/3: 5000+off + 1000·trial + n + d、Exp4: 6000+off + 100·trial"
                                      "（off = シナリオ A 0 / B 1000 / C 2000）", "DOCUMENTED",
                                      _OLD + "; expfam/src/exp_run_scenario_{A,B,C}.py@1b5030b", _OLDN),
    ("B-OLD05-EXP1-4", "init_seed"): ("data_seed + 50（Exp2/3 は + 500）", "DOCUMENTED", _OLD, _OLDN),
    ("B-OLD05-EXP1-4", "family_x"): ("A: Poisson ／ B: Gaussian ／ C: Bernoulli（Exp4 は推定側を 3×3 で変える）", "DOCUMENTED",
                                     "expfam/src/exp_run_scenario_{A,B,C}.py@1b5030b", _OLDN),
    ("B-OLD05-EXP1-4", "family_y"): ("A: Bernoulli ／ B: Poisson ／ C: Gaussian", "DOCUMENTED",
                                     "expfam/src/exp_run_scenario_{A,B,C}.py@1b5030b", _OLDN),
    ("B-OLD05-EXP1-4", "data_generator"): ("data_generator_expfam.generate_dual_data", "DOCUMENTED", _OLD, _OLDN),
}


# 手で確認した既知の食い違い・記録上の問題。
# (experiment_id, field, current_record, issue_type, severity, primary_source, resolvable, action_needed, check_result)
MANUAL_GAPS = [
    ("TG-A", "main_result（gram 相関・sparse・正指定）",
     "results_report.md・台帳・13 章（初出）: 0.241 ／ 一次 CSV primary_results.csv: 0.24046（四捨五入 0.240）",
     "CONFLICT", "NOTATION", TG + "expA/analysis/primary_results.csv@1ef9096", "yes",
     "既存記録は書き換えない。引用時は 0.240 と書く（13 章に記載済み）",
     "verify_addendum_20261127.py の TG-A-06 で MISMATCH として記録済み。主張は変わらない"),
    ("TG-B", "dist_params（Poisson の Fisher 情報行列の条件数）",
     "fairness_audit.md・frozen_protocol_B.md: 1.14〜1.17 ／ 一次 CSV tech_check_v2/information_diagnostics.csv: 1.11〜1.22",
     "CONFLICT", "NOTATION", TG + "expB/tech_check_v2/information_diagnostics.csv@1ef9096", "yes",
     "既存記録は書き換えない。引用時は一次 CSV の範囲を使う（13 章に記載済み）",
     "trace をそろえる設計判断（GM の分散）には影響しない。範囲の記載だけの差"),
    ("TG-DIAG", "C_Q の K 増加条件（符号・定数）",
     "diagnostic_report.md（2f71e61）: 誤った不等式 ／ forward_correction.md（c035df7）: 2Δℓ > n(1+ln2π)+(d−K+1)ln n",
     "CONFLICT", "INTERPRETATION", TG + "diagnostics_20261127/forward_correction_20261127/forward_correction.md@c035df7",
     "resolved", "forward correction を正とする（2f71e61 の報告書は書き換えない）",
     "13 章 §4 と 07 章 §5 は訂正後の式を使用。270/270 一致・別プロセス確認 17/17"),
    ("TG-B", "k_selection_rule（C_Q と C_Lap の罰則）",
     "C_Q: p_K = Kd − K(K−1)/2 + d（Gaussian-X の分散 15 個を含む） ／ C_Lap: d_K は分散を含まない（K に依存しない定数のため）",
     "CONFLICT", "INTERPRETATION", TG + "diagnostics_20261127/forward_correction_20261127/clap_formula_check.csv@c035df7",
     "no（定義の違い）", "C_Q と C_Lap の絶対値・条件間の基準値を直接比べない（公平性 D）",
     "定義上の違いで誤りではない。K の選択（argmin）には影響しないが、値の比較は不可"),
    ("ALL", "lineage（旧 0.5 系列と fixed 系列）",
     "A-HALF（B と C）・B-FIXED-GRID（C と B）・C-WINE-AUDIT（B/C）・F-160/F-161/F-162（E-scalar + OLD）など、1 記録に複数系列",
     "CONFLICT", "INTERPRETATION", "experiment_inventory.csv の lineage 列 / KI-002", "no（設計上の比較）",
     "表・図では系列ごとに列を分け、系列名を併記する。0.5 系列には KI-001 の注記を付ける",
     "結果集 03・04・07 章は系列を分けて記載していることを確認（混在表は見つからない）。比較自体が目的の記録である"),
    ("ALL", "件数の計上（88 件と追補）",
     "experiment_inventory.csv 88 件は実験・監査・理論・文献を含む記録数。追補は TG-A・TG-B（新規 fit）と TG-DIAG・TG-CORR（fit なし）",
     "CONFLICT", "INTERPRETATION", "experiment_inventory.csv / addendum_20261127_inventory.csv", "yes",
     "「88 件の実験」と書かない。「88 記録（うち新しい fit を伴う記録は involves_new_fits=yes の行）」と書く",
     "本台帳の involves_new_fits 列で区別した。TG-DIAG・TG-CORR は実験件数に数えない"),
    ("TG-A;TG-B", "registry 登録",
     "EXPERIMENT_REGISTRY.md（main・結果集ブランチ）に A・B の行が無い。K-exp2 の行は experiment/k-selection-exp2-generalization-20261125 にだけある",
     "MISSING", "REPRODUCIBILITY", "EXPERIMENT_REGISTRY.md@8da446a / @9d12dc4", "yes（Human Gate）",
     "14 章 §7 の統合案に従い、実験ブランチを main へ統合した後に registry へ追記する（追記のみ）",
     "今回は registry を変更しない"),
    ("F-CLAP-REEVAL;F-162", "runinfo（状態ファイル）",
     "保存状態 NPZ が repository 外（D:\\tento\\kennkyu_artifacts\\…）。sha256 は runinfo / STATE_ARCHIVE に記録",
     "MISSING", "REPRODUCIBILITY", "runinfo.json（states_file / npz キー）", "partial",
     "NPZ の保管先を研究者が確認する。GitHub だけでは再評価を再現できない",
     "completeness=NEEDS_LOCAL_NPZ（既存台帳と一致）"),
    ("G-LOCAL-ONLY", "evidence", "reports/research_scope/ ほか。main の作業ツリーでは untracked",
     "MISSING", "INTERPRETATION", "docs/untracked-reports-preservation-20261124@d946910", "yes（Human Gate）",
     "保存ブランチの統合可否を研究者が判断する", "d946910 には存在（GitHub で閲覧可能）。main には無い"),
    ("A-PAPER", "aggregation（原論文 Exp.2 の集計手順）", "原論文本文から Exp.2 の集計手順・Z の評価点・ln L の正規化を特定できない",
     "MISSING", "INTERPRETATION", "paper/A_study_on_latent_structural_models_for_binary_rel.pdf@1b5030b", "no",
     "著者への確認が必要。再現の比較では「手順不明」と併記する", "既存台帳 limitations 列と一致"),
    ("E-AF6", "n_fits", "台帳 run_count=未確認 ／ completeness=CONFIRMED", "MISSING", "INTERPRETATION",
     "expfam/results/auto_family_z_recovery/af6_laplace_confirmation_20261114/@8a66894", "yes",
     "AF6 の runinfo を探して fit 数を記入する（台帳の改訂は Human Gate）",
     "主要数値（statistical_analysis.csv）は検証済みだが、fit 数は台帳に無い"),
    ("D-MX2", "n_fits / replicates", "台帳 run_count=未確認（2 seed 系列）。duplicate_incident.csv あり（重複実行事故：最初の記録のみ採用）",
     "MISSING", "INTERPRETATION", "expfam/results/misspecification_extended/phase_20261110/@856d475", "yes",
     "manifest.csv から fit 数を数え、重複の扱いを台帳に記す（Human Gate）", "重複は事故として記録済み。分母の明記が必要"),
    ("F-KEXP2-MAIN;F-155;F-156;F-157;F-158;F-159;F-160;F-161;F-162;F-163;F-164;C-CORA-SH1;E-AF1;E-AF2;E-AF3-5;E-AF6;E-AF7;E-AF8;E-AF8V3",
     "commit", "台帳の commit 列（結果を作った commit）と where 列の SHA（ファイルを読んだ commit）が異なる",
     "CONFLICT", "NOTATION", "experiment_inventory.csv（where・commit 列）", "yes",
     "provenance_map の read_sha と result_commit を分けて読む", "どちらの SHA も repository・remote に存在（本台帳で確認）"),
]


# --------------------------------------------------------------------------------------
# 出力・検査
# --------------------------------------------------------------------------------------

def write(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


REQUIRED = {
    OUT_COND: ["record_id", "experiment_id", "parent_id", "condition_label", "record_kind", "involves_new_fits",
               "group", "field", "value", "status", "source"],
    OUT_PROV: ["experiment_id", "ledger", "record_kind", "read_sha", "read_sha_exists", "result_commit", "runner",
               "protocol", "runinfo", "primary_evidence", "primary_evidence_found", "github_tree"],
    OUT_GAPS: ["gap_id", "experiment_id", "field", "current_record", "issue_type", "severity", "primary_source",
               "resolvable", "action_needed", "check_result"],
}


def check() -> int:
    problems = []
    inv_ids = [r["experiment_id"] for r in load_csv(DATA / "experiment_inventory.csv")]
    add_ids = [r["experiment_id"] for r in load_csv(DATA / "addendum_20261127_inventory.csv")]
    all_ids = set(inv_ids) | set(add_ids)
    print(f"[台帳] experiment_inventory {len(inv_ids)} 件（一意 {len(set(inv_ids))}）、追補 {len(add_ids)} 記録")
    if len(inv_ids) != 88:
        problems.append("experiment_inventory が 88 件でない")
    for p, req in REQUIRED.items():
        rows = load_csv(p)
        cols = list(rows[0].keys()) if rows else []
        miss = [c for c in req if c not in cols]
        print(f"[CSV] {p.name}: {len(rows)} 行、列 {len(cols)}、必須列の欠落 {miss or 'なし'}")
        if miss:
            problems.append(f"{p.name} 必須列欠落 {miss}")
        for i, r in enumerate(rows):
            if None in r or any(v is None for v in r.values()):
                problems.append(f"{p.name} 行 {i + 2} の列数が不正")
                break
    cond = load_csv(OUT_COND)
    prov = load_csv(OUT_PROV)
    gaps = load_csv(OUT_GAPS)
    # ID 対応
    prov_ids = [r["experiment_id"] for r in prov]
    if len(prov_ids) != len(set(prov_ids)):
        problems.append("provenance_map の experiment_id が一意でない")
    if set(prov_ids) != all_ids:
        problems.append(f"provenance_map と台帳の ID 不一致: {sorted(set(prov_ids) ^ all_ids)}")
    cond_ids = {r["experiment_id"] for r in cond}
    if cond_ids != all_ids:
        problems.append(f"conditions と台帳の ID 不一致: {sorted(cond_ids ^ all_ids)}")
    keys = [(r["record_id"], r["field"]) for r in cond]
    dup = [k for k, n in Counter(keys).items() if n > 1]
    if dup:
        problems.append(f"conditions の (record_id, field) 重複 {len(dup)}: {dup[:3]}")
    per = Counter((r["experiment_id"]) for r in cond if r["condition_label"] == "ALL")
    bad = [k for k, n in per.items() if n != len(FIELDS)]
    if bad:
        problems.append(f"ALL 行の項目数が {len(FIELDS)} でない: {bad[:5]}")
    st = Counter(r["status"] for r in cond)
    if set(st) - set(STATUSES):
        problems.append(f"未定義の状態 {set(st) - set(STATUSES)}")
    for r in cond:
        if r["parent_id"] and any(x not in all_ids for x in r["parent_id"].split(";")):
            problems.append(f"parent_id 不明 {r['parent_id']}")
            break
        if r["status"] == "VERIFIED" and not r["source"]:
            problems.append(f"VERIFIED なのに source 空: {r['record_id']} {r['field']}")
            break
    for (e, f) in MANUAL_VALUES:
        if e not in all_ids or f not in FIELD_GROUP:
            problems.append(f"MANUAL_VALUES の不正なキー {(e, f)}")
    gid = [r["gap_id"] for r in gaps]
    if len(gid) != len(set(gid)):
        problems.append("gap_id が一意でない")
    for r in gaps:
        for e in r["experiment_id"].split(";"):
            if e not in all_ids and e != "ALL":
                problems.append(f"gaps の experiment_id 不明 {e}")
    # SHA と参照先
    nosha = [r["experiment_id"] for r in prov if r["read_sha_exists"] != "yes"]
    if nosha:
        problems.append(f"read_sha が存在しない: {nosha}")
    nocommit = [r["experiment_id"] for r in prov if r["result_commit_exists"] == "no"]
    if nocommit:
        problems.append(f"result_commit が存在しない: {nocommit}")
    noremote = [r["experiment_id"] for r in prov if not r["read_sha_remote_branches"]]
    if noremote:
        problems.append(f"read_sha を含む remote branch が無い: {noremote}")
    # 系列の区別：lineage が空の実験記録
    nolin = [r["experiment_id"] for r in prov if r["involves_new_fits"] == "yes" and r["lineage"] in ("", "—")]
    if nolin:
        problems.append(f"新規 fit を伴うのに lineage 不明: {nolin}")
    # 既存台帳を変更していないこと
    head = git("diff", "--stat", "8da446a", "--", "docs/research_results/data/experiment_inventory.csv",
               "docs/research_results/data/primary_results_index.csv",
               "docs/research_results/data/claims_evidence_matrix.csv",
               "docs/research_results/data/comparison_fairness_matrix.csv",
               "docs/research_results/data/source_coverage.csv",
               "docs/research_results/data/addendum_20261127_primary_results.csv",
               "docs/research_results/data/addendum_20261127_inventory.csv").stdout.decode().strip()
    if head:
        problems.append("既存台帳 CSV が 8da446a から変更されている: " + head)
    print("[状態] ALL 行:", dict(Counter(r["status"] for r in cond if r["condition_label"] == "ALL")))
    print("[状態] 子行:", dict(Counter(r["status"] for r in cond if r["condition_label"] != "ALL")),
          f"（子行 {sum(1 for r in cond if r['condition_label'] != 'ALL')}）")
    print("[gaps]", dict(Counter((r["issue_type"], r["severity"]) for r in gaps)))
    print("[provenance] evidence 発見:", dict(Counter(r["primary_evidence_found"] for r in prov)),
          " new fits:", dict(Counter(r["involves_new_fits"] for r in prov)))
    if problems:
        print("QC: PROBLEMS")
        for p in problems:
            print("  -", p)
        return 1
    print("QC: PASS（構文・必須列・ID 対応・一意性・SHA 存在・既存台帳の非変更）")
    return 0


def main():
    if "--check" not in sys.argv:
        cond, prov, gaps = build()
        write(OUT_COND, cond)
        write(OUT_PROV, prov)
        write(OUT_GAPS, gaps)
        print(f"wrote {OUT_COND.name} {len(cond)} 行 / {OUT_PROV.name} {len(prov)} 行 / {OUT_GAPS.name} {len(gaps)} 行")
    sys.exit(check())


if __name__ == "__main__":
    main()
