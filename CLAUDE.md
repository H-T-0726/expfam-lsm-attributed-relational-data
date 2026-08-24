# CLAUDE.md — Dual-ExpFam LSM（Claude Code / Codex 共通作業規約）

人間向けの入口・環境構築・ディレクトリ規約は `README.md` を参照。
このファイルは **Claude Code と Codex が毎セッション共有する研究指示の正本**であり、
両者が常に守る研究上の制約だけを書く。ツール固有の workflow は各ツールの設定・拡張に分離する。
実験数値・ファイル一覧・先生対応履歴・TODO はここに置かない（正本は §8）。

---

## 1. 確定した生成モデルと確定式

これに反する式を書かない。

```
z_i  ~ N(0, I_k)
y_ij ~ ExpFam_Y( η_ij^Y = w_0^Y + w^Y z_i^T z_j )   i < j
x_il ~ ExpFam_X( η_il^X = f_l^T z_i )               バイアスなし
```

- `w_0^Y, w^Y ∈ R` は **スカラー**（行列 W_Y ではない）。
- **X は per-component**：X の尤度は列 `l` ごとに因子分解する。
  ただし標準の `DualExpFamLSM` では **`family_x` は全 X 列で共通の1種類**である。
  列ごとに異なる family を指定できるのは `experimental/model_dual_expfam_percolumn.py`
  （`family_x_list`）だけであり、**prototype・本文採用不可**。
- **θ = { F, w_0^Y, w^Y }**
  **＋ Gaussian-X のとき Σ_X（対角）／＋ Gaussian-Y のとき σ_Y²**。
  σ_Y² は M-step で MLE 推定される（`calc_sigma_y`）。
  **数理上は `σ_Y²` を dispersion として扱うが、実装は標準偏差 `σ_Y` を `self.sigma_y` に保持し、
  使用時に二乗する**（`self.sigma_y ** 2`）。

E-step（分散パラメータ φ を落とさない）:

```
V_Y(η) = A_Y''(η) / φ_Y        φ_Y = 1（Bernoulli/Poisson）, φ_Y = σ_Y²（Gaussian）
V_X    = Σ_X^{-1}              （Gaussian-X）
V_X    = diag(A_X''(F m_i))    （Bernoulli/Poisson-X）

gradient : ... + w^Y Σ_{j≠i} [ T_Y(y_ij) − A_Y'(η_ij^Y) ] / φ_Y · z_j
A_i      = I_k + F^T V_X(m_i) F + (w^Y)^2 Σ_{j≠i} V_Y(η_ij^Y) z_j z_j^T
```

---

## 2. 1/2 係数（5系統を絶対に混同しない）

| 系統 | 1/2 |
|---|:---:|
| **Mikawa et al. 2024 の印刷された原論文式**（Eq.19/20/22/23、Appendix A-1/A-3/A-5） | **あり** |
| old 0.5 Python 系列（`model_expfam.py` / `model_dual_expfam.py`） | あり |
| **本研究の独立再導出・採用式**（unique undirected-pair conditional） | **extra 1/2 なし** |
| fixed Python 系列（`model_dual_expfam_fixed.py`） | なし |
| MATLAB `calcAi` | なし |

- **原論文の印刷式には 1/2 がある**（2026-08-18 に原論文を直接確認）。
  **「原論文にも 1/2 がない」と書かない。**
- 本研究が採用式で 1/2 を外しているのは、**原論文の印刷式と本研究の採用式の意図的な差**である。
  根拠は独立な再導出であり、MATLAB は補助的な実装比較としてのみ参照する（単独の正解として扱わない）。
- 一次確認: `docs/math_notes/half_factor_primary_source_confirmation_20260818.md`
- 導出: `docs/math_notes/half_factor_math_explanation.md`
- 詳細: `RESEARCH_MASTER.md` §6 / `KNOWN_ISSUES.md` KI-001

---

## 3. 実装系列（結果を絶対に混ぜない）

```
reproduction/src/model.py                       LatentStructuralModel（先行研究 Python 再現）
└ expfam/src/model_expfam.py                    0.5 あり
  └ expfam/src/model_dual_expfam.py             0.5 あり ← 学会予稿の本文採用実験
    └ expfam/src/model_dual_expfam_fixed.py     0.5 なし ← 実データ実験フェーズ
      └ expfam/src/experimental/model_dual_expfam_masked.py
        ├ _nb.py    └ _percolumn.py             prototype・本文採用不可
```

- 数値を引用するときは**必ずどの系列か**を明記する（KI-002）。
- 異なる系列の結果を同じ表・図に混在させない。

---

## 4. source priority（どれを正とするか）

1. 一次データ — 結果 CSV・runinfo・実行ログ・実コード・**先行研究の原論文 PDF**（`paper/A_study_on_latent_structural_models_for_binary_rel.pdf`）
2. canonical docs — `RESEARCH_MASTER.md` / `KNOWN_ISSUES.md` / `EXPERIMENT_REGISTRY.md` / このファイル
3. 日付入りで凍結された `reports/<phase>/`（当時の記録として読む）
4. 参考のみ — `docs_for_notebooklm/*`、`GEMINI_REPORT_*`、`expfam/CLAUDE.md`、`expfam/handoff.md`

**AI 生成レポート・派生資料を一次根拠にしない（KI-007）。** 数値主張は必ず 1 に遡る。
歴史的文書（`reports/theory_audit/*`、`docs/math_notes/half_factor_literature_code_check.md` 等）は
その時点の記録であり、現在の状態と異なることがある。書き換えずに現行 canonical docs を正とする。

---

## 5. 表現・主張の限定条件（断定しない）

- **0.5 係数（KI-001）:** 採用式（1/2 なし）を正とする。本文採用実験は 0.5 あり実装で実行されている。
  0.5 が掛かるのは Y 側項のみで Z 事前分布項・X 側項には掛かっていないため、
  **「Newton 方向が全体として正しいとは断定できない」を必ず付記する。**
- **Scen.C の「Y=Gaussian が支配」** は Exp4 ablation からの推測であり、理論的証明はない。
- **誤指定倍率 23.6× / 41.45× / 38.97×** は系列も条件も異なる別々の値。並べるときは出所を明記する（KI-003）。
- **モデル選択基準を「Schwarz BIC」と呼ばない。** 現行 `calc_bic_dual` は観測データの周辺尤度ではなく
  `Q_strict`（EM の Q 関数の MC 近似）を使う。**Q-based complete-data criterion / ICL-type** として扱う
  （`reports/theory_audit/theory_audit_report_20260718.md` §6-7、KI-010）。
  関数名 `calc_bic_dual`・CSV 列名 `BIC`・過去結果の呼称は**変更しない**。
- `KNOWN_ISSUES.md`「まだ主張してはいけないこと」に該当する内容を報告書・原稿案に書かない。

---

## 6. Human Gate / Approved Task（正式な権限モデル）

ユーザーが目的と scope を明示して依頼した時点で、その依頼を Approved Task の人間承認とする。
Approved Task 内の個々のコマンド・編集・validation・commit・normal push・Draft PR について、
個別の再確認は不要。Human Gate に該当する判断が新たに必要になった場合だけ停止して確認する。

### Human Gate（人間の判断または人間自身による操作が必要）

- 研究目的・モデル・数式の変更
- family / K / 評価指標 / 実験条件の変更
- 結果に基づく次の実験の決定
- frozen spec の変更
- prototype の正式手法・manuscript evidence への昇格
- Issue close
- PR merge
- force push / published history rewrite
- main への直接変更

研究上の変更は人間が判断し、必要なら別の明示的 scope として承認する。
**Issue close / PR merge / force push / published history rewrite は、通常の agent workflow では
agent に委任せず、人間自身が実行する。**

### Approved Task（承認済み scope 内では Claude / Codex が自動実行してよい）

- repo 調査と承認済み scope 内の実装
- test / debug / validation
- 承認済み pilot / full experiment
- 承認済み script による artifact 生成（既存成果物の再生成・上書きは scope 明示時のみ）
- provenance 記録
- working branch での commit / normal push
- Draft PR の作成・更新
- CI 確認と承認済み scope 内の修正

実装中に別の研究課題や改善を発見しても scope を拡張しない。
承認済み作業が終わったら人間へ結果を返し、次の phase や次の実験を自動開始しない。

---

## 7. 作業時の安全ルール

- **main を直接編集しない。** `git switch -c <type>/<issue#>-<slug>` でブランチを切る
  （`experiment/` `audit/` `maintenance/` `docs/`）。
- 結果 CSV・図はスクリプト経由でのみ生成する。**手で編集しない。**
- **過去の CSV / runinfo を書き換えない**（事後に推測した情報の追記も禁止）。
- `EXPERIMENT_REGISTRY.md` は追記して育てる。**既存行のパス文字列を書き換えない・削除しない。**
- 実行環境は `.python-version`（3.13.14）と `requirements*.txt` を基準とする。
  ただしこれは今後の baseline であり、**過去実験の環境を再現するものではない**（KI-014）。
- 既存の記録済み成果物を手編集・無断で上書きしない。artifact は承認済み script から生成する。
  既存成果物の再生成・上書きが本当に必要な場合は、Approved Task の scope に明示されていること。
- 依頼に目的と scope が明示されていないコード修正・実験再実行・ファイル移動/削除は、
  実行前にユーザーへ確認する。

---

## 8. 参照先（必要になったときだけ読む）

| 文書 | 内容 |
|---|---|
| `README.md` | 人間向け入口・環境構築・ディレクトリ規約 |
| `RESEARCH_MASTER.md` | 研究内容の正本（目的・手法・数式の説明・フェーズ史・先生対応 Q1-Q4） |
| `KNOWN_ISSUES.md` | 事故台帳・「まだ主張してはいけないこと」 |
| `EXPERIMENT_REGISTRY.md` | 実験 → スクリプト → CSV → 図 → 主張の provenance |
| `conference_submission_final_draft.md` | 学会予稿（完成・変更しない） |
| `reports/environment/baseline_20260818.md` | 実行環境ベースライン |

**今後やること（TODO）は GitHub Issue で管理する。このファイルにも他の canonical docs にも書かない。**
