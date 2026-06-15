# RESEARCH_MASTER.md

研究内容の正本（マスタードキュメント）。
確定事項はroot `CLAUDE.md`に基づく。未確定事項は断定せず、`KNOWN_ISSUES.md`への参照を付す。

---

## 1. 研究目的

関係データ Y（ネットワーク）と属性データ X の両方が指数型分布族（Exponential Family）に従う
潜在構造モデル（Latent Structural Model）を構築し、従来手法（Y=Bernoulli固定、X=Gaussian固定）からの一般化が
有効であることを実験的に示す。

成果物：`conference_submission_final_draft.md`（学会予稿）

---

## 2. 従来手法

Mikawa et al. (2024), NOLTA IEICE vol.15 no.2 の潜在構造モデル。

- Y（関係データ）：Bernoulli分布で固定
- X（属性データ）：Gaussian分布で固定
- 推定：MCEM + Laplace近似、BICによる潜在次元k選択
- Python再現実装：`reproduction/src/model.py`（`LatentStructuralModel`）
- MATLAB原実装：`Mato Lab Program/calcEtaNewton.m` 等

---

## 3. 提案手法 Dual-ExpFam LSM

X・Yの両方を指数型分布族（Gaussian / Bernoulli / Poisson）から任意に指定できるよう一般化したモデル。
実装：`expfam/src/model_dual_expfam.py`（`DualExpFamLSM`）

| 項目 | 従来手法 | 提案手法 |
|----|------|------|
| Xの分布族 | Gaussian固定 | Gaussian / Bernoulli / Poisson |
| Yの分布族 | Bernoulli固定 | Gaussian / Bernoulli / Poisson |
| 推定方法 | MCEM + Laplace近似 | 同上（一般化） |
| BICによる次元選択 | あり | あり（一般化） |

Categorical分布は未実装（KI-005）。

---

## 4. 生成モデル

root `CLAUDE.md`記載の確定式：

```
z_i  ~ N(0, I_k)
y_ij ~ ExpFam_Y( η_ij^Y = w_0^Y + w^Y z_i^T z_j )   i < j
x_il ~ ExpFam_X( η_il^X = f_l^T z_i )                バイアスなし
```

- `w_0^Y, w^Y ∈ R`：スカラー（行列 W_Y ではない）
- θ = { F, w_0^Y, w^Y }、Gaussian-Xのときのみ対角Σを追加で推定

過去に修正された誤り（root CLAUDE.mdより）：

| 式 | 誤（旧） | 正（現在） |
|----|---------|-----------|
| eq(1) | `σ(z_i^T W_Y z_j)`（行列） | `σ(w_0 + w z_i^T z_j)`（スカラー） |
| eq(2) | `N(w_{0l} + z_i^T w_l, σ_l²)`（バイアスあり） | `N(f_l^T z_i, σ_l²)`（バイアスなし） |
| eq(6) | `(w^Y)^2/2 Σ_{j≠i}` | `(w^Y)^2 Σ_{j≠i}` |
| θ | `{Z, W_Y, w_0, W_X}` | `{F, w_0^Y, w^Y}`（+Gaussian-Xのときのみ Σ） |

---

## 5. 推定アルゴリズム

MCEM（Monte Carlo EM）+ Laplace近似によるE-step、M-stepは解析解（Gaussian-X）またはAdamによる勾配法（非Gaussian）。

- 共通実験設定：n=150, d=15, k*=3, 10試行, L=5（MCサンプル数）, EM反復数=8（`expfam/src/exp_scenario_lib.py` L.40-45）
- 潜在変数Zの評価にはProcrustes回転によるアライメントが適用される（回転不変性への対処）

---

## 6. 指数型分布族化の中心

精度行列（確定式、root CLAUDE.md）：

```
A_i = I_k + F^T V_X(m_i) F + (w^Y)^2 Σ_{j≠i} A_Y''(η_ij^Y) z_j z_j^T
```

- `A_Y''(η)`：Y側の対数正規化関数の2階微分（分散関数）。分布族によって形が変わる（例：Bernoulliならσ(η)(1-σ(η))、Poissonならexp(η)）
- `V_X = Σ^{-1}`（Gaussian-X）、`V_X = diag(A_X''(F m_i))`（Bernoulli/Poisson-X）
- A'(η)（1階微分）はgradient（E-stepの勾配）に現れ、A''(η)（2階微分）は精度行列（Newton法のヘッセ行列近似）に現れる

**1/2係数について（KI-001）：** root CLAUDE.mdの数学的証明・MATLAB実装との照合に基づき、原稿はΣ_{j≠i}に1/2を付けない式を採用している。
ただし現行Python実装（`model_dual_expfam.py`, `model_expfam.py`）のY側Term3には0.5が残存しており、本文採用実験はこの0.5あり実装で実行されている。
「Newton方向が全体として正しいとは断定できない」という限定があることに注意（詳細はKI-001、`docs_for_notebooklm/01_formula_code_audit.md`）。

---

## 7. 実装対応表

| 数式・概念 | 実装ファイル | 備考 |
|--------|----------|----|
| 生成モデル（z_i, y_ij, x_il） | `expfam/src/data_generator_expfam.py` | 人工データ生成 |
| Y側 ExpFam拡張（Bernoulli/Poisson/Gaussian） | `expfam/src/model_expfam.py` | `ExpFamLatentStructuralModel`、Y側のみ拡張した中間クラス |
| X・Y両側 ExpFam拡張（提案手法本体） | `expfam/src/model_dual_expfam.py` | `DualExpFamLSM`、`_calc_gradient`・`_calc_precision_matrix`を完全オーバーライド |
| 精度行列のY側Term3（0.5除去版、補助） | `expfam/src/model_dual_expfam_fixed.py` | `DualExpFamLSMFixed`、補助実験のみで使用（KI-001, KI-002） |
| 先行研究の基底クラス | `reproduction/src/model.py` | `LatentStructuralModel`、`DualExpFamLSM`はこれを継承 |
| Q関数・BIC・RMSE・Procrustes・EM実行 | `expfam/src/utils_expfam.py` | `run_em_dual`は旧版（`DualExpFamLSM`）を使用 |
| シナリオA/B/C共通設定・実験関数 | `expfam/src/exp_scenario_lib.py` | n/d/k*/試行数/L/EM反復数の定義 |
| 各シナリオ実行 | `exp_run_scenario_{A,B,C}.py` | Exp1-4を実行しCSV・図を出力 |
| MATLAB原実装（先行研究、1/2なしの根拠） | `Mato Lab Program/calcEtaNewton.m`（calcAi） | root CLAUDE.mdが1/2不要の根拠として引用 |

---

## 8. 実験設計

3シナリオ：

| シナリオ | 真のExpFam_X | 真のExpFam_Y | 略称 |
|------|-----------|-----------|----|
| A | Poisson | Bernoulli | P-B |
| B | Gaussian | Poisson | G-P |
| C | Bernoulli | Gaussian | B-G |

各シナリオでExp1（k変化・BIC選択）、Exp2（n変化）、Exp3（d変化）、Exp4（誤指定）を実行（詳細は`EXPERIMENT_REGISTRY.md`）。

---

## 9. 現時点で確認できた結果

（出所はすべて`reports/claims_and_evidence.md`および`EXPERIMENT_REGISTRY.md`の該当行を参照）

- 3シナリオすべてでBICがk*=3を正確に選択（各10試行）
- nの増加に伴い3シナリオすべてでRMSE(Z)が改善（Scen.A: 49%減、Scen.B: 31%減、Scen.C: 62%減、n=50→300）
- 先行研究との同条件比較でRMSE(Z)差 < 0.001（5試行）
- 誤指定によりRMSE(Z)が悪化することを3シナリオで確認（Scen.A最大3.41倍、Scen.B最大7.35倍、Scen.C関連はセクション10参照）
- Categorical以外の3分布族（Gaussian/Bernoulli/Poisson）の全組み合わせで実装が動作する（`test_dual_expfam.py`、5テスト全PASS）

---

## 10. 注意が必要な結果

- **41.5倍（Scen.C, 本文記載の最大誤指定倍率）は、旧版実装（0.5係数が残存するmodel_dual_expfam.py）に基づく結果である。** 図1(b)には対応するバーがなく、本文の記述のみ（KI-003）。
- 図1(b)の灰色バー（視覚上の最大値）は23.6倍で、41.5倍とは異なる条件（X=Gaussian/Y=Bernoulli、先行研究固定条件）の値である。
- fixed版（0.5除去）の補助実験では38.97倍という別条件の値が得られているが、本文採用実験とは異なる実装・条件であり、直接比較はできない。
- Scen.Bの7.35倍についても、対応CSV内での条件特定は完了していない（`reports/claims_and_evidence.md` L.13）。
- Scen.Cの「Y=Gaussianが推定を支配している」という解釈は、Exp4 ablation（No X ≈ 提案手法）からの推測であり、理論的証明はない（`reports/claims_and_evidence.md` L.18）。

---

## 11. 研究主張の安全レベル

### 強く言えること

- Dual-ExpFam LSMはGaussian/Bernoulli/Poissonの3分布族について、X・Y両側を任意に指定できる実装が完成している。
- 3シナリオでBICによるk*=3の正確な選択を確認した。
- nの増加に伴うRMSE(Z)の改善を3シナリオで確認した。
- 先行研究（Y=Bernoulli固定、X=Gaussian固定）と同条件での結果が先行研究の再現実装と一致する（差 < 0.001）。
- 分布族の誤指定がRMSE(Z)を悪化させることを複数シナリオで確認した。

### 注意付きで言えること

- 「誤指定により最大41.5倍悪化する」（Scen.C）→ 旧版実装（0.5係数あり）に基づく結果であることを明記する（KI-001, KI-003）。
- 「Xの誤指定はXを使わないより悪い」（Scen.A）→ Scen.Aのみで確認、他シナリオへの一般化は未確認。
- 「dの増加でRMSE(Z)が改善する」（Scen.A/B）→ Scen.Cでは平坦であり、シナリオ依存。

### まだ言ってはいけないこと

- 「0.5係数を除去した実装の方が優れている」（comparison_quick.csvのratio_fix_oldは条件依存で0.27〜1.23倍、一貫しない）。
- 「Categorical分布にも対応している」（未実装）。
- 「Wine実データで有効性が確認された」（未評価）。
- 「精度行列のNewton方向は0.5係数があっても全体として正しい」（限定条件付きでのみ成立しうる、KI-001）。
