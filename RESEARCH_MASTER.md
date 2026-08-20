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
| 情報量規準による次元選択 | あり | あり（一般化） |

- Xの分布族は**全X列で共通の1種類**（`family_x` はスカラー引数）。X の尤度自体は列ごとに因子分解するが、
  列ごとに異なる family を指定できるのは experimental prototype
  `expfam/src/experimental/model_dual_expfam_percolumn.py`（`family_x_list`）のみで、本文採用不可。
- Categorical分布は未実装（KI-005）。

**次元選択基準の呼称について（重要）：**
現行実装 `calc_bic_dual`（`expfam/src/utils_expfam.py`）は、観測データの周辺尤度ではなく
`Q_strict`（EM の Q 関数の MC 近似）を用いて `-2·Q_strict + num_params·ln n` を計算する。
したがって**これは Schwarz BIC ではなく、「Q ベース完全データ型基準（ICL-type）」として扱う**
（`reports/theory_audit/theory_audit_report_20260718.md` §6-7、KI-010）。
ただし**既存の関数名 `calc_bic_dual`・CSV 列名 `BIC`・過去結果の呼称は変更しない**
（過去の provenance を壊さないため）。本文書内で「BIC」と書かれている箇所は、
この Q ベース基準を指す歴史的な呼称である。

---

## 4. 生成モデル

root `CLAUDE.md`記載の確定式：

```
z_i  ~ N(0, I_k)
y_ij ~ ExpFam_Y( η_ij^Y = w_0^Y + w^Y z_i^T z_j )   i < j
x_il ~ ExpFam_X( η_il^X = f_l^T z_i )                バイアスなし
```

- `w_0^Y, w^Y ∈ R`：スカラー（行列 W_Y ではない）
- **θ = { F, w_0^Y, w^Y } ＋ Gaussian-X のとき Σ_X（対角）＋ Gaussian-Y のとき σ_Y²**
  - Σ_X：`calc_sigma` で閉形式 MLE 更新（Gaussian-X のみ。非 Gaussian では単位行列固定）
  - σ_Y²：Gaussian-Y の dispersion。**数理上の推定対象は `σ_Y²`。実装では標準偏差 `σ_Y` を
    `self.sigma_y` として保持し `calc_sigma_y()` で M-step ごとに MLE 更新し、
    尤度・勾配・精度行列では `self.sigma_y ** 2` として使用する**（`model_expfam.py` L.234-235 で
    `sigma_sq` の平方根を格納、L.75 / L.108 / L.146 で二乗して参照）。
    `calc_bic_dual` の `num_params` でも Gaussian-Y のとき 1 パラメータとして数えられる

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
V_Y(η) = A_Y''(η) / φ_Y        φ_Y = 1（Bernoulli/Poisson）, φ_Y = σ_Y²（Gaussian）

A_i = I_k + F^T V_X(m_i) F + (w^Y)^2 Σ_{j≠i} V_Y(η_ij^Y) z_j z_j^T
```

対応する E-step 勾配の Y 側項（分散パラメータを落とさない）：

```
w^Y Σ_{j≠i} [ T_Y(y_ij) − A_Y'(η_ij^Y) ] / φ_Y · z_j
```

- `A_Y''(η)`：Y側の対数正規化関数の2階微分（分散関数）。分布族によって形が変わる（Bernoulliならσ(η)(1-σ(η))、Poissonならexp(η)、Gaussianなら1）
- **Gaussian-Y では φ_Y = σ_Y² が残る**ため、精度行列の Y 側の実効曲率は `1/σ_Y²`、勾配の残差は `/σ_Y²` で割られる
  （`model_expfam.py` の `_variance_function` / `_dispersion` を参照）。σ_Y² は M-step で推定される
- `V_X = Σ_X^{-1}`（Gaussian-X）、`V_X = diag(A_X''(F m_i))`（Bernoulli/Poisson-X）
- A'(η)（1階微分）はgradient（E-stepの勾配）に現れ、A''(η)（2階微分）は精度行列（Newton法のヘッセ行列近似）に現れる

**1/2係数について（KI-001）：** 本研究は、対称関係尤度からの独立導出に基づき、Σ_{j≠i}に1/2を付けない式を採用している。
（2026-07-19更新：1/2不要の主根拠は対称関係尤度からの独立導出であり、MATLABは
補助比較としてのみ扱う。手元コードにはY側勾配のw欠落等、追加確認が必要な箇所が
あるため、MATLABを単独の正解とはせず、先行研究全体の誤りとも断定しない。）
ただし現行Python実装（`model_dual_expfam.py`, `model_expfam.py`）のY側Term3には0.5が残存しており、本文採用実験はこの0.5あり実装で実行されている。
「Newton方向が全体として正しいとは断定できない」という限定があることに注意（詳細は`KNOWN_ISSUES.md` KI-001、
`docs/math_notes/half_factor_primary_source_confirmation_20260818.md`、
`docs/math_notes/half_factor_math_explanation.md`）。

### 6.1 1/2 の所在 — 5系統の整理（絶対に混同しない）

1/2 の有無は、どの系統を指しているかで異なる。

| # | 系統 | 1/2 | 根拠 |
|---|---|:---:|---|
| 1 | **Mikawa et al. 2024 の印刷された原論文式**（Eq.19 / 20 / 22 / 23、Appendix A-1 / A-3 / A-5） | **あり** | `docs/math_notes/half_factor_primary_source_confirmation_20260818.md`（2026-08-18 に原論文を直接確認） |
| 2 | old 0.5 Python 系列（`model_expfam.py` L.109/L.135、`model_dual_expfam.py` L.159/L.200、`reproduction/src/model.py`） | **あり** | 実コード |
| 3 | **本研究の独立再導出・採用式**（unique undirected-pair conditional） | **extra 1/2 なし** | `docs/math_notes/half_factor_math_explanation.md`、`reports/theory_audit/theory_audit_report_20260718.md` §4.1 |
| 4 | fixed Python 系列（`model_dual_expfam_fixed.py` L.77/L.113） | **なし** | 実コード |
| 5 | MATLAB `calcAi`（`Mato Lab Program/calcEtaNewton.m` L.56-63） | **なし** | 実コード |

**原論文の印刷式には 1/2 がある。**「先行研究側にも 1/2 がない」と書いてはならない。
系統1と系統3の差は**原論文の印刷式と本研究の採用式の意図的な差**であり、
系統2と系統4の差は**実装系列の差**である。両者は別の問題として扱う。

原論文で 1/2 が現れる箇所（2026-08-18 の一次確認）：

| 箇所 | 1/2 |
|---|:---:|
| Eq.(19) / Eq.(20) | あり（尤度レベル） |
| Eq.(22) / Eq.(23) | あり（precision / gradient） |
| Appendix A-1 / A-3 / A-5 | あり |

争点は「原論文の尤度に 1/2 があるか」ではなく、**それが z_i の gradient / precision まで残るか**である。
本研究の再導出（z_i について微分すると対称和の両側からの寄与が合算される）は、
原論文の尤度レベルの 1/2 と矛盾しない。詳細と歴史的経緯は
`docs/math_notes/half_factor_primary_source_confirmation_20260818.md` を参照。

---

## 6b. 先生からの指摘と対応（Q1–Q4）

指導教員からの技術的指摘と、それに対する本研究の結論。返答案の全文は `docs/teacher/` にある。

| # | 指摘 | 結論 | 反映先 |
|---|---|---|---|
| Q1 | 指数型分布族の式はスカラーか | **スカラーで正しい。** 「各次元に独立適用」であることを明示した | 原稿・§4 |
| Q2 | X は per-component か | **Yes（因子分解の意味で）。** X の尤度は列 `l` ごとに因子分解し、先行研究の対角 Σ と同構造。**ただし `family_x` は全 X 列で共通の1種類**であり、列ごとに異なる family を選べるわけではない（それは experimental prototype `model_dual_expfam_percolumn.py` の `family_x_list` のみ） | 原稿・§3・§4 |
| Q3 | `Σ_{j≠i}` に 1/2 は不要では | **不要（指摘のとおり）。** 本研究の採用式から 1/2 を除去済み。ただし下記の限定に注意 | 原稿 eq(6)・§6.1 |
| Q4 | Σ はパラメータか | **条件付き。** Gaussian-X のときのみ推定対象。それ以外の分布族では単位行列固定 | 原稿・§4（θ の定義） |

**Q3 の限定条件（欠落させてはならない）:**

- 「1/2 不要」が指すのは **本研究の採用式（系統3）** である。
  **先行研究の印刷式（系統1）には 1/2 がある**（2026-08-18 に一次確認）。
  「先生の指摘どおり先行研究にも 1/2 がない」という読み方は誤りである。
- 主根拠は**対称関係尤度からの独立導出**であり、MATLAB `calcAi` に 1/2 がないことは
  **補助的な実装比較**として扱う。MATLAB を単独のゴールドスタンダードとはしない
  （手元コードには Y 側勾配の w 欠落など追加確認が必要な箇所がある）。
- 現行 Python 実装（系統2）には 0.5 が残存しており、**本文採用実験はこの実装で実行されている**。
  したがって「Newton 方向が全体として正しいとは断定できない」という限定条件を必ず付記する（KI-001）。

返答案: `docs/teacher/teacher_reply_draft.md`（Q1/Q2/Q4）、
`docs/teacher/half_factor_teacher_reply.md`（Q3）。
いずれも作成時点の記録であり、Q3 については本節と §6.1 の整理を現在の正とする。

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
| MATLAB原実装（先行研究グループの実装コード。**コードには1/2なし。ただし原論文の印刷式には1/2がある**） | `Mato Lab Program/calcEtaNewton.m`（calcAi） | 系統5（§6.1）。**補助的な実装比較**として参照する。採用式の主根拠は独立導出であり、MATLABを単独の正解とはしない |

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

## 8b. 実データ実験フェーズ（Wine / Cora / MovieLens、2026-06-17〜2026-07-07）

人工データ実験（シナリオA/B/C）と学会予稿完成後、`DualExpFamLSMFixed`（0.5除去版）を用いて
3つの実データセットに提案手法を適用した追加フェーズ。修論フェーズに向けた検証であり、
学会予稿（`conference_submission_final_draft.md`）には含まれない。

計画・総括は `reports/real_data_experiment_plan.md`、`reports/movielens_pilot_design.md`、
`reports/real_data_experiment_summary.md` を参照。実験行の詳細は `EXPERIMENT_REGISTRY.md`
「実データ実験フェーズ」節を参照。

### 対象データセット

| データセット | X | Y | family_x | family_y | Yの起源 | 評価方法 |
|---|---|---|---|---|---|---|
| Wine (sklearn) | 化学成分13種 | 同クラス関係 | gaussian | bernoulli | ラベル由来（人工） | in-sample再構成 |
| Cora | BoW上位50語 | 引用関係 | bernoulli | bernoulli | 自然ネットワーク | held-out link prediction |
| MovieLens 100K（movie-node投影） | ジャンルmulti-hot(d=19) | 共評価数 | bernoulli | **poisson** | 共評価カウント（投影） | in-sample count regression |

### 系譜（pilot → audit → clean/final）

各データセットで「pilot（試行・設計探索）→ audit（Wineのみ、既存CSVとの読取専用突合）→
clean/final_clean（本文・スライド用の最終整形）」という段階を踏んでいる。
Coraは pilotの中でもサブセット設計自体を試行錯誤しており（BFSサブセット→偏りが判明し不採用→
balanced_degreeサブセットを採用→n=280→700までスケーリング検証）、その経緯は
`reports/real_data_experiment_summary.md` に記録されている。整形（clean/final_clean）系スクリプトは
いずれも既存CSVを読み込むのみでモデルの再学習は行わない設計になっている。

### 主な結果（すべて実データ・探索的検証であり、学会予稿の主張とは独立）

- Wine: BIC最小 k=3 が真のクラス数（3）と一致。X_only/Y_onlyのablationからYが分離の主要因であることを確認
  （ただしWineのYはラベル由来の人工的な関係であり、自然ネットワークではない）。
- Cora: held-out link predictionでtest_AP ≈ 0.43〜0.46（random基準の約2.6〜2.8倍）。
  自然な引用ネットワークでの汎化性能を確認。ただしBICは疎な密度（0.011）でk=1を選択する限界がある。
- MovieLens: Bernoulli-X / Poisson-Y という新規の分布族組み合わせが数値的に安定に収束し、
  in-sample再構成でPearson ≈ 0.96 を達成。ただしPoisson overdispersion（var/mean ≈ 10）があり、
  strict held-out評価（未知ペア予測）は現在のAPIでは未対応（pair mask非対応）。

### まだ言えないこと（実データ実験フェーズ固有）

- 「MovieLensで未知ペアの共評価数を予測できた」（strict held-outは未実装、in-sample再構成のみ）
- 「Wineで自然ネットワークの実験を行った」（Wineの Y はラベル由来）
- 「Cora（n=280 balanced subset）の結果がfull Cora（n=2708）に一般化する」（未確認）
- 「BICが実データで常に適切なkを選ぶ」（Coraでは疎密度によりk=1を選択）

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

- **41.45倍（Scen.C, 本文記載の最大誤指定倍率。est X=Gaussian/Y=Poisson の両側誤指定条件。本文 L.83 の「41.5倍」はこの値の丸め）は、旧版実装（0.5係数が残存するmodel_dual_expfam.py）に基づく結果である。** 図1(b)には対応するバーがなく、本文の記述のみ（KI-003、根拠は`reports/mismatch_audit/mismatch_audit_report_20260708.md` §1）。
- 図1(b)の灰色バー（視覚上の最大値）は23.6倍で、41.45倍とは異なる条件（X=Gaussian/Y=Bernoulli、先行研究固定条件）の値である。
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

- 「誤指定により最大41.45倍悪化する」（Scen.C、est X=Gaussian/Y=Poisson。本文表記は丸めた「41.5倍」）→ 旧版実装（0.5係数あり）に基づく結果であることを明記する（KI-001, KI-003）。
- 「Xの誤指定はXを使わないより悪い」（Scen.A）→ Scen.Aのみで確認、他シナリオへの一般化は未確認。
- 「dの増加でRMSE(Z)が改善する」（Scen.A/B）→ Scen.Cでは平坦であり、シナリオ依存。

### まだ言ってはいけないこと

- 「0.5係数を除去した実装の方が優れている」（comparison_quick.csvのratio_fix_oldは条件依存で0.27〜1.23倍、一貫しない）。
- 「Categorical分布にも対応している」（未実装）。
- 「Wine実データで有効性が確認された」（未評価）。
  （2026-07-19更新：この行は2026-05時点の記述。fixed版でのWine評価（BIC k選択・ablation・
  旧版突合）は§8bおよびKI-006のとおり実施済み。ただしWineのYはラベル由来のため
  「自然ネットワークでの有効性確認」とは引き続き言えない。§8bの限定付き記述を正とする。）
- 「精度行列のNewton方向は0.5係数があっても全体として正しい」（限定条件付きでのみ成立しうる、KI-001）。
