# CLAUDE.md — Dual-ExpFam LSM プロジェクト

## 概要

先行研究[1]（Bernoulli-Y + Gaussian-X 固定）を**指数型分布族に一般化**した潜在構造モデルの学会予稿。

- **メイン原稿：** `conference_submission_final_draft.md`（完成済み）
- **提出用図：** `figures/fig1a_n_sweep_color.pdf/png`（n-RMSE）, `figures/fig1b_misspecification_color.pdf/png`（誤指定）
- **先行研究：** Mikawa et al. (2024), NOLTA IEICE vol.15 no.2 — PDF は `paper/` にあるが読み込み不可

---

## 生成モデル（確定式）

```
z_i  ~ N(0, I_k)
y_ij ~ ExpFam_Y( η_ij^Y = w_0^Y + w^Y z_i^T z_j )   i < j
x_il ~ ExpFam_X( η_il^X = f_l^T z_i )                バイアスなし
```

- `w_0^Y, w^Y ∈ R`：**スカラー**（行列 W_Y ではない）
- **θ = { F, w^0_Y, w^Y }**、Gaussian-X のときのみ対角 Σ を追加で推定

---

## 精度行列（確定式）

```
A_i = I_k  +  F^T V_X(m_i) F  +  (w^Y)^2 Σ_{j≠i} A_Y''(η_ij^Y) z_j z_j^T
```

**1/2 は不要**（先生の指摘通り。MATLAB calcAi も 1/2 なし）。  
証明：`(1/2)Σ_{i≠j}` で Q 関数を書いても z_i 微分で両側から寄与が合算され 1/2 が消える。

```
V_X = Σ^{-1}              （Gaussian-X）
V_X = diag(A_X''(F m_i))  （Bernoulli/Poisson-X）
```

**⚠ Python 実装（model_expfam.py L.135、model_dual_expfam.py L.159, 200）には spurious な 0.5 が残っている。**  
現行 Python 実装では，E-step の Y 側 Term3 の gradient と precision の両方に 0.5 が残っている。  
ただし，0.5 が掛かっているのは Y 側項のみであり，Z 事前分布項（Term1）および X 側項（Term2）には掛かっていない。  
そのため，Y 側が支配的な場合には近似的に打ち消される可能性があるが，  
**posterior 全体の Newton 方向が正しいとは断定できない**（Term1・Term2 と Term3 のスケールが異なるため）。  
また，Laplace 近似のサンプリング分散は Y 側が支配的な方向で大きくなりやすい（2 倍と断定はできない）。  
本研究の原稿では，再導出および MATLAB 実装との照合に基づき，1/2 を含まない式を採用する。  
論文の式は正しい。実装修正は今後の課題（今は触らない）。

---

## 図1(b) と誤指定倍率の扱い（重要）

`figures/fig1b_misspecification_color.png` は以下の3条件のバーのみを表示する。

- X-side misspec.：Y を正解に固定し X のみ誤指定した場合の最大値
- Y-side misspec.：X を正解に固定し Y のみ誤指定した場合の最大値
- Fixed Gauss-X/Bern-Y：先行研究固定条件（X=Gaussian, Y=Bernoulli）

| 値 | 条件 | 図に表示 | 原稿本文 |
|---:|---|:---:|:---:|
| **23.6×** | Scen.C の Fixed Gauss-X/Bern-Y（X=Gaussian, Y=Bernoulli） | ✓ 灰色バー（図の視覚最大値） | 記載なし |
| **41.5×** | Scen.C の X=Gaussian, Y=Poisson（**X・Y 両方誤指定**） | ✗ バーなし | L.83「最大41.5倍」|

**方針：**
- 原稿本文の「41.5倍」は CSV 全条件中の最大値として事実上正確だが，図中に対応するバーがない。
- NotebookLM 用資料では 23.6× と 41.5× を必ず分けて説明し，混同しない。
- Scen.A の最大値 3.41× は X-only 誤指定（図に表示あり）。  
  Scen.B の 7.35× と Scen.C の 41.5× は両方誤指定（図に表示なし）。  
  本文の「最大悪化倍率」はシナリオごとに異なる種別の条件から来ている点に注意。

---

## 過去に直した誤り（再発防止）

| 式 | 誤（旧） | 正（現在） |
|----|---------|-----------|
| eq(1) | `σ(z_i^T W_Y z_j)` 行列 | `σ(w_0 + w z_i^T z_j)` スカラー |
| eq(2) | `N(w_{0l} + z_i^T w_l, σ_l²)` バイアスあり | `N(f_l^T z_i, σ_l²)` バイアスなし |
| eq(6) | `(w^Y)^2/2 Σ_{j≠i}` | `(w^Y)^2 Σ_{j≠i}` |
| θ | `{Z, W_Y, w_0, W_X}` | `{F, w_0^Y, w^Y}`（+Gaussian-X ときのみ Σ）|

---

## 先生の指摘と対応状況

| # | 指摘 | 結論 | 原稿反映 |
|---|------|------|---------|
| Q1 | 指数型分布族の式はスカラーか | OK。「各次元独立適用」を追記 | ✓ |
| Q2 | X は per-component か | Yes。先行研究の対角 Σ と同構造 | ✓ |
| Q3 | `Σ_{j≠i}` に 1/2 は不要では | **不要**（正しい）→ 除去済み | ✓ |
| Q4 | Σ はパラメータか | 条件付き（Gaussian-X のみ）→ 修正済み | ✓ |

**先生への返答ファイル：**
- Q1/Q2/Q4：`docs/teacher/teacher_reply_draft.md`
- Q3：**`docs/teacher/half_factor_teacher_reply.md`**（こちらが正しい回答）

---

## ファイル構成（重要なもののみ）

```
conference_submission_final_draft.md   メイン原稿（完成）
figures/fig1a_n_sweep_color.*          提出用図1
figures/fig1b_misspecification_color.* 提出用図2

expfam/src/model_dual_expfam.py        提案手法本体
expfam/src/model_expfam.py             基底クラス
reproduction/src/model.py             先行研究 Python 再現
Mato Lab Program/calcEtaNewton.m      先行研究 MATLAB（calcAi: 1/2 なし の根拠）

docs/teacher/teacher_reply_draft.md              Q1/Q2/Q4 返答案
docs/teacher/half_factor_teacher_reply.md        Q3 返答案（正しい版）
docs/math_notes/half_factor_math_explanation.md  1/2 不要の数学的証明
docs/math_notes/half_factor_literature_code_check.md  MATLAB vs Python 照合表
```

---

## 残タスク

- [ ] **先生への返答を送る**（`docs/teacher/teacher_reply_draft.md` + `docs/teacher/half_factor_teacher_reply.md` を参照）
- [ ] **Word 文書に原稿内容を反映する**（.docx はリポジトリ外にある）
- [ ] Python 実装の 1/2 修正（修論フェーズで対応、今は不要）
