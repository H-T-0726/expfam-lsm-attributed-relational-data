# expfam — Dual-ExpFam LSM 実験フォルダ

提案手法 **Dual-ExpFam LSM** の実装・実験・結果を格納するフォルダ。
このフォルダがリポジトリのメイン。

> **初めて触る場合:** まずルートの `CLAUDE.md` を読んでから実験を始めること。

---

## 1. 提案手法の概要

**Dual-ExpFam LSM** は、属性データ X と関係データ Y の両方を指数型分布族で記述できる
属性情報付き潜在構造モデル。ベースラインは Mikawa et al. (NOLTA 2024)。

| 項目 | ベースライン（NOLTA 2024） | 本研究（Dual-ExpFam LSM） |
|-----|------------------------|------------------------|
| X の分布族 | Gaussian 固定 | Gaussian / Bernoulli / Poisson から任意指定 |
| Y の分布族 | Bernoulli 固定 | Gaussian / Bernoulli / Poisson から任意指定 |
| 推定方法 | MCEM + Laplace 近似 | 同上（一般化） |
| BIC による次元選択 | あり | あり（一般化） |

---

## 2. 実験シナリオ

| シナリオ | 真の ExpFam_X | 真の ExpFam_Y | 略称 |
|---------|-------------|-------------|------|
| **A** | Poisson | Bernoulli | P-B |
| **B** | Gaussian | Poisson | G-P |
| **C** | Bernoulli | Gaussian | B-G |

### 共通実験設定

```
n = 150   （オブジェクト数）
d = 15    （属性次元数）
k* = 3    （真の潜在次元数）
試行数 = 10
L = 5     （MC サンプル数）
EM 反復数 = 8
k 探索範囲 = [1, 2, 3, 4, 5, 6]
```

設定の定義場所: `src/exp_scenario_lib.py` L.40-45

---

## 3. 実験の実行方法

```bash
# 前提: kennkyu/ を作業ルートとして Python 環境が整っていること
cd expfam/src

# シナリオ A（Poisson-X, Bernoulli-Y）
python exp_run_scenario_A.py

# シナリオ B（Gaussian-X, Poisson-Y）
python exp_run_scenario_B.py

# シナリオ C（Bernoulli-X, Gaussian-Y）
python exp_run_scenario_C.py

# 実行前にユニットテストで動作確認
python test_dual_expfam.py   # 5 つ全 PASS を確認

# 補足実験
python run_exp1_full_metrics.py    # 全メトリクス詳細
python run_exp2_bic_v2.py          # BIC 詳細
python run_wine_dual.py            # Wine 実データ実験
```

**実験結果の保存先:** `expfam/results/`

---

## 4. src/ 主要ファイル

| ファイル | 役割 | 優先度 |
|---------|------|:------:|
| `model_dual_expfam.py` | **提案手法核心**（DualExpFamLSM クラス） | ★★★ |
| `model_expfam.py` | Y 側 ExpFam 拡張基底クラス | ★★★ |
| `utils_expfam.py` | Q 関数・BIC・RMSE・Procrustes・EM 実行 | ★★★ |
| `data_generator_expfam.py` | 人工データ生成 | ★★ |
| `exp_scenario_lib.py` | 実験共通設定・実験関数 | ★★★ |
| `exp_run_scenario_A.py` | シナリオ A 実行スクリプト | ★★ |
| `exp_run_scenario_B.py` | シナリオ B 実行スクリプト | ★★ |
| `exp_run_scenario_C.py` | シナリオ C 実行スクリプト | ★★ |
| `test_dual_expfam.py` | ユニットテスト（5 つ全 PASS 済み） | ★★ |
| `run_exp1_full_metrics.py` | BIC 付き全メトリクス詳細 | ★ |
| `run_exp2_bic_v2.py` | BIC 次元選択の詳細実験 | ★ |
| `run_wine_dual.py` | Wine 実データ実験 | ★ |
| `src/archive/` | 旧実験スクリプト（参照のみ） | — |

---

## 5. クラス継承構造

```
../reproduction/src/model.py
   └── LatentStructuralModel        先行研究 Python 再現（ベースクラス）

src/model_expfam.py
   └── ExpFamLatentStructuralModel  Y 側を ExpFam に拡張

src/model_dual_expfam.py
   └── DualExpFamLSM               提案手法本体（X・Y 両方を ExpFam に拡張）
```

`DualExpFamLSM` は `_calc_gradient` と `_calc_precision_matrix` を完全にオーバーライド。

---

## 6. results/ ファイル一覧

### 現行実験結果（参照・引用すべきもの）

| ファイル | 内容 |
|---------|------|
| `exp_scenario_{A,B,C}_exp1_k.csv` | k 変化実験・試行別 raw データ |
| `exp_scenario_{A,B,C}_exp2_n.csv` | n 変化実験・試行別 raw データ |
| `exp_scenario_{A,B,C}_exp3_d.csv` | d 変化実験・試行別 raw データ |
| `exp_scenario_{A,B,C}_exp4_mismatch.csv` | 誤指定実験・試行別 raw データ |
| `exp1_full_{A,B,C}.csv` | BIC 付き全メトリクス |
| `exp2_bic_{A,B,C}.csv` | BIC 次元選択実験 |
| `log_scenario_{A,B,C}.txt` | 実行ログ（収束状況） |
| `fig_scenario_{A,B,C}_*.pdf/png` | シナリオ別図（2026-03-24 版） |
| `wine_dual_results.csv`, `wine_F.npy` | Wine 実データ実験結果 |

**注意:** 提出用最終図は `../figures/` 配下（fig1a_n_sweep_color.*, fig1b_misspecification_color.*）を使うこと。
`results/` 内の `fig1_rmse_vs_n.*`・`fig2_*.* ` は提出直前の旧版。

### AI 生成レポート（参考のみ、未検証）

| ファイル | 注意 |
|---------|------|
| `GEMINI_REPORT_MULTI_SCENARIO.md` | Gemini 生成・研究者検証未完了 |
| `GEMINI_REPORT_PHASE2_FINAL.md` | 同上 |

### archive 済み

| フォルダ | 内容 |
|---------|------|
| `results/archive/` | 旧実験結果（初期シングルシナリオ実装） |
| `src/archive/` | 旧実験スクリプト |

---

## 7. 実験前に確認すべき注意事項

### 1. Python path の設定

`DualExpFamLSM` は `reproduction/src/model.py` の `LatentStructuralModel` を継承している。
実行前に `reproduction/src/` を Python path に追加すること。

```python
import sys
sys.path.insert(0, '../../reproduction/src')
```

各実験スクリプトには自動的に設定コードが含まれているが、
Jupyter 等から直接実行する場合は手動設定が必要。

### 2. E-step の 1/2 係数問題（重要）

現行 Python 実装には以下の不整合が残っている：

| 場所 | 内容 | 状態 |
|-----|------|------|
| `model_dual_expfam.py` L.159 | `0.5 * w * (Z.T @ residual_y)` | ⚠ 余分な 0.5 |
| `model_dual_expfam.py` L.200 | `0.5 * (w**2) * (...)` | ⚠ 余分な 0.5 |

**原稿採用式（正しい式）:** Y 側 Term3 は `w^Y Σ_{j≠i}(...)` で 1/2 なし。
**Python 実装:** `0.5 * w * Σ_{j≠i}(...)` で 0.5 あり。

0.5 が乗っているのは Y 側 Term3 のみ（Z 事前分布 Term1 と X 側 Term2 は正しい）。
Newton 方向への影響: Y 側が支配的な場合には近似的に打ち消される可能性があるが、
全体として原稿式と一致するとは断定できない。
**修正は修論フェーズで対応予定。現時点では触らない。**

詳細: `../CLAUDE.md` の「精度行列（確定式）」節、`../half_factor_math_explanation.md`

### 3. 旧 CLAUDE.md は使わない

`expfam/CLAUDE.md` は旧 Gemini セッション向けのファイル。
正しい確定事項はルートの `../CLAUDE.md` を参照すること。

### 4. 作業ディレクトリの注意

`expfam/handoff.md` には旧パス `C:/研究2/` が記載されているが、
現在の作業ディレクトリは `D:/tento/kennkyu/`。

---

## 8. 数値確認済みの主要実験結果

以下は `docs_for_notebooklm/02_experiment_result_verification.md` で CSV 照合済み：

| 指標 | Scen. A (P-B) | Scen. B (G-P) | Scen. C (B-G) |
|-----|:-----------:|:-----------:|:-----------:|
| RMSE(Z) at k=3 | 0.278 | 0.182 | 0.028 |
| n=50→300 削減率 | 48.8% | 31.0% | 62.0% |
| mismatch 最大倍率 | 3.41× | 7.35× | 41.5× |
| BIC 選択 k | 3 ✓ | 3 ✓ | 3 ✓ |

先行研究との比較（Control 条件, k=3, 5試行）: RMSE(Z) 差 = 0.0006 < 0.001

---

## 9. references/

| ファイル | 内容 |
|---------|------|
| `baseline_metrics.md` | 先行研究 (NOLTA 2024) の論文数値 + 再現実装値の対照表 |

NotebookLM 用詳細: `../docs_for_notebooklm/02_experiment_result_verification.md`
