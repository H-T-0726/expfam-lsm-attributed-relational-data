# 先行研究の印刷式における 1/2 係数 — 一次確認ノート（2026-08-18）

**種別:** primary-source evidence note（一次確認記録）
**確認日:** 2026-08-18
**位置づけ:** 本ノートは、先行研究の**印刷された原論文の式に 1/2 が存在する**という一次確認の記録である。
現在の canonical docs（root `CLAUDE.md`、`RESEARCH_MASTER.md` §6、`KNOWN_ISSUES.md` KI-001）は本ノートを出典として参照する。

---

## 1. 確認日

**2026-08-18**

---

## 2. 一次資料（primary source）

| 項目 | 内容 |
|---|---|
| 文献 | Mikawa et al., "A study on latent structural models for binary relational data with attribute information," NOLTA, IEICE, vol. 15, no. 2, 2024 |
| リポジトリ内のファイル | **`paper/A_study_on_latent_structural_models_for_binary_rel.pdf`** |

**同ディレクトリの `paper/2.pdf` は本ノートの対象ではない。** このファイルの内容・書誌情報は
リポジトリ内で特定されていない（`docs_for_notebooklm/00_repository_inventory.md` L.326・L.374 でも
「内容未確認」と記録されている）。本ノートは `paper/2.pdf` について何も主張しない。

---

## 3. 確認方法

**研究者本人が原論文を直接閲覧して確認した。**

補足（誤読防止のため明記する）:

- Claude Code 側の PDF 読み込みは**この時点でも実施していない**。本ノートは、
  研究者本人による一次確認の結果を canonical docs から参照できる形で記録したものである。
- 従来リポジトリ内の複数の文書が「PDF は読み込みツール非対応のため直接確認不可」と
  記録していた（`docs/math_notes/half_factor_literature_code_check.md` L.28-29 ほか）。
  その制約は Claude 側のツールに関するものであり、**研究者本人による確認を妨げるものではない**。

---

## 4. 印刷された式の確認結果

原論文の以下の箇所に 1/2 が現れる。

| 箇所 | 1/2 |
|---|:---:|
| Eq.(19) | **あり** |
| Eq.(20) | **あり** |
| Eq.(22) | **あり** |
| Eq.(23) | **あり** |
| Appendix A-1 | **あり** |
| Appendix A-3 | **あり** |
| Appendix A-5 | **あり** |

すなわち、原論文において 1/2 は Eq.(19)/(20) の**尤度レベル**から現れ、
Eq.(22)/(23)・Appendix の対応箇所まで系統的に保持されている。

**したがって「先行研究の印刷式にも 1/2 がない」という記述は誤りである。**

---

## 5. 1/2 の所在 — 5 系統の整理

1/2 の有無は、どの系統を指しているかで異なる。**5 系統を混同してはならない。**

| # | 系統 | 1/2 | 根拠 |
|---|---|:---:|---|
| 1 | **Mikawa et al. 2024 の印刷された原論文式**（Eq.19 / 20 / 22 / 23、Appendix A-1 / A-3 / A-5） | **あり** | 本ノート（2026-08-18 の一次確認） |
| 2 | **old 0.5 Python 系列**（`expfam/src/model_expfam.py` L.109, L.135 / `expfam/src/model_dual_expfam.py` L.159, L.200 / `reproduction/src/model.py`） | **あり** | 実コード |
| 3 | **本研究の独立再導出・採用式**（unique undirected-pair conditional として整理したもの） | **extra 1/2 なし** | `docs/math_notes/half_factor_math_explanation.md`、`reports/theory_audit/theory_audit_report_20260718.md` §4.1 |
| 4 | **fixed Python 系列**（`expfam/src/model_dual_expfam_fixed.py` L.77, L.113） | **なし** | 実コード |
| 5 | **MATLAB `calcAi`**（`Mato Lab Program/calcEtaNewton.m` L.56-63） | **なし** | 実コード |

系統 1 と系統 3 の差が本研究の**意図的な設計判断**であり、
系統 2 と系統 4 の差が**実装系列の差**である。両者は別の話であって、混ぜて論じてはならない。

---

## 6. 採用式との意図的な差

本研究が採用している E-step の式は次のとおりで、Σ_{j≠i} に 1/2 を付けない。

```
V_Y(η)  = A_Y''(η) / φ_Y     φ_Y = 1（Bernoulli/Poisson）, φ_Y = σ_Y²（Gaussian-Y）

∇_{z_i} : w^Y Σ_{j≠i} [ T_Y(y_ij) − A_Y'(η_ij^Y) ] / φ_Y · z_j
A_i     : I_k + F^T V_X(m_i) F + (w^Y)^2 Σ_{j≠i} V_Y(η_ij^Y) z_j z_j^T
```

（分散パラメータ φ_Y は 1/2 の議論とは独立である。Gaussian-Y では σ_Y² が M-step で推定され、
勾配・精度行列の双方に残る。本ノートが扱うのは Σ_{j≠i} の前に付く **extra 1/2** の有無のみである。）

**これは原論文の印刷式と異なる。差は意図的である。**

根拠は独立な再導出である。Y 側の対数尤度を対称和で書いても、
z_i は j>i 側と j<i 側の両方のペアに現れるため、z_i について微分すると
両側からの寄与が合算され、結果として z_i の条件付き事後分布には
Σ_{j≠i} のみが残り extra 1/2 が消える（導出は
`docs/math_notes/half_factor_math_explanation.md`）。

**争点の所在（解釈・[DERIVED]）:**
争点は「原論文の尤度に 1/2 があるかどうか」ではない。**それが z_i の gradient / precision まで
残るかどうか**である。原論文の 1/2 が Eq.(19)/(20) の尤度レベルから系統的に現れていることは、
対称関係を順序対として二重に数えることへの補正として整合的であり、
本研究の再導出（z_i について微分すると両側の寄与が合算される）と**矛盾しない**。

**MATLAB の位置づけ:** MATLAB `calcAi`（系統 5）に 1/2 がないことは、
本研究の採用式と同じ結論を与えるが、**単独のゴールドスタンダードとしては扱わない**。
理由は `reports/theory_audit/theory_audit_report_20260718.md` の結論のとおりで、
手元 MATLAB コードには Y 側勾配の w 欠落など追加確認が必要な箇所があるためである。
主根拠はあくまで独立な再導出であり、MATLAB は補助的な実装比較として参照する。

**維持される限定条件:** 系統 2（old 0.5 Python）で実行された本文採用実験について、
「Newton 方向が全体として正しいとは断定できない」という限定条件は本ノートによって解除されない
（0.5 が掛かるのは Y 側項のみで、Z 事前分布項・X 側項には掛かっていないため。詳細は KI-001）。

---

## 7. 歴史的文書との関係（supersede の扱い）

以下 2 件は、**それぞれの作成時点では正確な記述である。書き換えない。削除しない。**

| 文書 | 当時の記述 | 当時の状況 |
|---|---|---|
| `reports/theory_audit/theory_audit_report_20260718.md`（2026-07-18） | 原論文 PDF の直接確認は未実施のため `[UNRESOLVED]` | 2026-07-18 時点では PDF の直接確認が完了していなかった |
| `docs/math_notes/half_factor_literature_code_check.md`（2026-05-08） | 「PDF を直接確認していないため断定できない」 | 2026-05-08 時点では PDF の直接確認が完了していなかった |

**時間順の整理:**

1. 2026-05-08 — 原論文 PDF の直接確認が未完了。判断を保留した（`half_factor_literature_code_check.md`）。
2. 2026-07-18 — 同じく直接確認が未完了。理論監査で `[UNRESOLVED]` として明示的に未解決扱いとした（`theory_audit_report_20260718.md`）。
3. **2026-08-18 — 研究者本人が原論文を直接確認し、Eq.19/20/22/23 および Appendix A-1/A-3/A-5 に 1/2 があることを確認した（本ノート）。**

**この経緯を「当時の記述が誤りだった」と書いてはならない。** 当時は一次確認が未完了であり、
そのことを正しく記録していた。本ノートはその未解決点を後から解消したものであって、
過去の監査記録の正確さを否定するものではない。

上記 2 件の本文は historical record として保持し、**1/2 の現在の状態については本ノートを正とする。**

---

## 8. 参照関係

- 現在の canonical な要約: root `CLAUDE.md` §1、`RESEARCH_MASTER.md` §6
- 事故台帳: `KNOWN_ISSUES.md` KI-001
- 数学的導出: `docs/math_notes/half_factor_math_explanation.md`
- 実装系列ごとの照合: `docs/math_notes/half_factor_literature_code_check.md`（2026-05-08 時点の記録）
- 理論監査: `reports/theory_audit/theory_audit_report_20260718.md`（2026-07-18 時点の記録）
- provenance: `EXPERIMENT_REGISTRY.md`「理論監査フェーズ」
