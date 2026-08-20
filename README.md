# Dual-ExpFam LSM — 属性情報付き関係データの潜在構造モデルの指数型分布族拡張

属性データ **X** と関係データ **Y** の両方を指数型分布族へ一般化した潜在構造モデル
**Dual-ExpFam LSM** の実装・実験・検証・原稿を管理する研究リポジトリ。
先行研究が Bernoulli-Y + Gaussian-X に固定していた分布仮定を、
Gaussian / Bernoulli / Poisson を X 側・Y 側で独立に選べる形へ一般化し、
分布族の誤指定が潜在変数の推定精度に与える影響を人工データと実データで評価している。

ベースライン: Mikawa et al., "A study on latent structural models for binary relational data
with attribute information," NOLTA, IEICE, vol. 15, no. 2, 2024.

---

## 現在の位置づけ

| フェーズ | 状態 |
|---|---|
| 人工データ実験フェーズ（シナリオ A/B/C、Exp1-4） | 完了。学会予稿 `conference_submission_final_draft.md` に収録済み |
| 実データ実験フェーズ（Wine / Cora / MovieLens、fixed 系列） | 完了（2026-06-17〜2026-07-07）。**学会予稿には未収録**、修論フェーズ向けの追加検証 |
| 理論監査フェーズ | 完了（2026-07-18〜19、read-only）。`reports/theory_audit/` |
| 実行環境の固定 | 完了（**2026-08-20 実測**）。`reports/environment/baseline_20260818.md`（ファイル名の `20260818` は Issue #9 由来の識別子であり、実測日ではない） |

現在進行中の作業は GitHub Issue で管理している。

---

## 環境構築

Python **3.13.14** が baseline（`.python-version`）。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    /  POSIX: source .venv/bin/activate
pip install -r requirements.txt        # 実験の再現に必要な最小構成
pip install -r requirements-dev.txt    # テストも動かす場合はこちら
```

依存は `requirements.in` / `requirements-dev.in` を入力として `uv pip compile --generate-hashes` で
機械生成している（直接依存は numpy / pandas / scipy / matplotlib / scikit-learn / psutil、dev に pytest）。

> **注意:** これは**今後の再現基準**であり、**過去の実験結果 CSV を生成した環境を再現するものではない。**
> 過去の実行環境は復元できない（`KNOWN_ISSUES.md` KI-014、`reports/environment/baseline_20260818.md`）。

動作確認:

```bash
python expfam/src/test_dual_expfam.py
python -m pytest expfam/src/experimental -q
```

---

## ディレクトリ規約

| パス | 内容 |
|---|---|
| `expfam/src/` | 人工データ・実データ実験の実装とスクリプト（研究の本体） |
| `expfam/src/experimental/` | masked / NB / per-column の prototype（本文採用不可） |
| `tools/` | 2026-07-08 以降のフェーズの実験スクリプト |
| `expfam/results/` | 結果 artifact の root。人工データ Scenario A/B/C の旧・主要実験（Exp1-4）の CSV と生成図は**この直下**に出力される（`expfam/src/exp_scenario_lib.py` の `_RES = expfam/results`）。新しいフェーズには subdirectory を使うものもある（`fixed_official/`, `real_data/` 等）。**正確な対応は `EXPERIMENT_REGISTRY.md` を参照** |
| `expfam/figures/` | fixed 系列・実データ等の図ディレクトリ（`fixed_official/`, `real_data/`, `distribution_mismatch_fixed/` 等） |
| `figures/` | 学会予稿用の最終図（`fig1a_*`, `fig1b_*`）および後続フェーズの一部の図 |
| `reports/<phase>/` | 各フェーズの設計・結論（日付入りで凍結） |
| `reproduction/` | 先行研究の Python 再現実装 |
| `Mato Lab Program/` | 先行研究の MATLAB 原実装 |
| `paper/` | 先行研究 PDF |
| `docs/` | 補助資料（`docs/README.md` 参照） |
| `docs_for_notebooklm/` | 外部 AI ツール投入用の派生資料（**一次根拠にしない**・`docs_for_notebooklm/README.md` 参照） |
| `archive/` | 研究本体外の歴史資料 |

> `figures/` が提出図とフェーズ図を兼ねているのは 2026-07-08 の規約変更の名残。
> パスは実験 provenance の一部なので**移動しない**。

クラス継承（詳細と混合禁止ルールは `CLAUDE.md` §3）:

```
reproduction/src/model.py               先行研究 Python 再現（基底）
└ expfam/src/model_expfam.py            Y 側 ExpFam 拡張
  └ expfam/src/model_dual_expfam.py     提案手法本体（DualExpFamLSM）
    └ expfam/src/model_dual_expfam_fixed.py
```

---

## 実験の回し方

venv を有効化し、**リポジトリルートから**実行する。

```bash
python expfam/src/exp_run_scenario_A.py   # 真の X=Poisson,  Y=Bernoulli
python expfam/src/exp_run_scenario_B.py   # 真の X=Gaussian, Y=Poisson
python expfam/src/exp_run_scenario_C.py   # 真の X=Bernoulli,Y=Gaussian
python tools/...                          # 2026-07-08 以降のフェーズ
```

出力先はスクリプトごとに異なる（Scenario A/B/C は `expfam/results/` 直下、
後続フェーズは subdirectory を使うものもある）。**どの実験がどこに出力するかは
`EXPERIMENT_REGISTRY.md` を正とする。**
**結果 CSV と図はスクリプト経由でのみ生成し、手で編集しない。**
実験を追加したら `EXPERIMENT_REGISTRY.md` に行を追記する（既存行は書き換えない）。

---

## 目的別に次に読む文書

| 目的 | 文書 |
|---|---|
| 研究全体を理解したい | `RESEARCH_MASTER.md` |
| 数値の根拠を辿りたい | `EXPERIMENT_REGISTRY.md` |
| 何を主張してよいか知りたい | `KNOWN_ISSUES.md` |
| 学会予稿を読みたい | `conference_submission_final_draft.md` |
| 1/2 係数の経緯を知りたい | `docs/math_notes/half_factor_primary_source_confirmation_20260818.md` |
| 実データ実験の総括を読みたい | `reports/real_data_experiment_summary.md` |
| 実行環境を知りたい | `reports/environment/baseline_20260818.md` |
| 補助資料の地図がほしい | `docs/README.md` |
| **Claude Code で作業する** | **`CLAUDE.md`** |

---

## source of truth

- 数値主張は必ず**一次データ**（結果 CSV・runinfo・実行ログ・実コード）に紐づける。
- `docs_for_notebooklm/*` と `GEMINI_REPORT_*` は AI 生成・派生資料であり、
  **一次根拠として単独で使わない**（`KNOWN_ISSUES.md` KI-007）。
- `reports/<phase>/` は日付入りで凍結された当時の記録。現在の状態とは異なることがある。
- 今後やること（TODO）は GitHub Issue で管理する。canonical docs には書かない。
