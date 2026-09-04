# RESEARCH_MASTER.md

研究内容の正本（マスタードキュメント）。
確定事項はroot `CLAUDE.md`に基づく。未確定事項は断定せず、`KNOWN_ISSUES.md`への参照を付す。

---

## 1. 研究目的

Gaussian / Bernoulli / Poisson を対象として、属性情報付き関係データの潜在構造モデル
（Latent Structural Model）を指数型分布族へ一般化する。
先行研究が Y=Bernoulli・X=Gaussian に固定していた分布仮定を、X 側・Y 側で独立に指定できる形へ拡張し、
**人工データおよび実データを用いて、その挙動・有効となる条件・限界を評価する。**

一般化が有効であることは前提としない。評価対象には、有効性が確認できなかった条件・
悪化が観測された条件も含む（§12・§14）。

成果物：`conference_submission_final_draft.md`（学会予稿）

### Historical note（2026-05、学会予稿フェーズの目的文）

> 関係データ Y（ネットワーク）と属性データ X の両方が指数型分布族（Exponential Family）に従う
> 潜在構造モデル（Latent Structural Model）を構築し、従来手法（Y=Bernoulli固定、X=Gaussian固定）からの一般化が
> 有効であることを実験的に示す。

上の historical 目的文は**現在の目的記述ではない**。2026-08 までに蓄積した証拠では
一般化の有効性は条件依存であり（人工データの sparse-Y 条件では改善、
実データ MovieLens の leakage-safe protocol では split 間変動に比べて小さい）、
「有効であることを示す」を目的として掲げない。

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
  本フェーズ（2026-06〜07）時点では strict held-out 評価（未知ペア予測）は
  当時の API では未対応だった（pair mask 非対応）。
  **その後の状態は §12.5 を参照** — pair mask は experimental 系列で実装済みであり、
  Issue #33 の user-disjoint validation も実施済みである（fixed 本体 API への統合は未実施）。

### まだ言えないこと（実データ実験フェーズ固有）

- 「MovieLensで未知ペアの共評価数を予測できた」
  （**2026-08-31 現在もこの禁止は有効**。ただし理由は更新されている:
  strict held-out は experimental 系列で実装済みであり「未実装だから」ではない。
  現在の理由は Issue #33 report §9 の禁止句リスト、Y の overdispersion により
  Poisson log-likelihood が score にすぎないこと、および lineage が
  experimental prototype であることによる。§12.5）
- 「Wineで自然ネットワークの実験を行った」（Wineの Y はラベル由来）
- 「Cora（n=280 balanced subset）の結果がfull Cora（n=2708）に一般化する」（未確認）
- 「BICが実データで常に適切なkを選ぶ」（Coraでは疎密度によりk=1を選択）

---

## 9. 現時点で確認できた結果

本節は **current interpretation** である。数値の lineage と限定条件は §12・§14 を参照する
（出所は `EXPERIMENT_REGISTRY.md` の該当行と一次 CSV）。

- **潜在次元の選択（明示設定内の観測）:** fixed 系列（`DualExpFamLSMFixed`）の人工データ実験では、
  歴史的に `BIC` と呼ばれてきた Q ベース基準が、候補 `K = 1,…,9` から `K = 3` を
  シナリオ A / B / C 各 10 trial、**計 30/30 trial** で選択した
  （`expfam/results/fixed_official/exp1_k9/fixed_exp1_bic_k1to9_bestk_by_trial.csv`）。
  **この観測結果は Schwarz BIC の妥当性、一般的な true-K recovery、
  model-selection consistency のいずれも意味しない**（§12.6、KI-010）。
- nの増加に伴い3シナリオすべてでRMSE(Z)が改善（旧 0.5 系列で Scen.A: 49%減、Scen.B: 31%減、Scen.C: 62%減、n=50→300）。
  fixed 系列では A −49.3% / B −41.2% / C −58.6%（`fixed_official/exp2/fixed_exp2_n_sweep_improvement.csv`）。
  **両系列の数値を同じ表・図に並べない**（KI-002）。
- 先行研究との同条件比較でRMSE(Z)差 < 0.001（5試行）。
- 分布族の誤指定によりRMSE(Z)が悪化することを3シナリオで確認
  （旧 0.5 系列 Scen.A最大3.41倍、Scen.B最大7.35倍、Scen.C関連は §10 参照。
  fixed 系列は A 4.3414× / B 9.0405× / C 40.3706×、§15）。
- Gaussian / Bernoulli / Poisson の3分布族の全組み合わせで実装が動作する（`test_dual_expfam.py`、5テスト全PASS）。
  **Categorical は未実装**（KI-005）。

### Historical wording（2026-05、現在は使用しない）

> - 3シナリオすべてでBICがk*=3を正確に選択（各10試行）

この表現は **current claim として使用しない**。現行基準は観測データ周辺尤度に基づく Schwarz BIC ではなく
`−2·Q_strict + p̂·ln n` の完全データ型（ICL 型）指標であり、
「正確に選択」は基準の妥当性・一般的な K 回復能力を含意してしまうためである（§12.6、KI-010）。
関数名 `calc_bic_dual`・CSV 列名 `BIC` は provenance のため変更しない。

---

## 10. 注意が必要な結果

- **41.45倍（Scen.C, 本文記載の最大誤指定倍率。est X=Gaussian/Y=Poisson の両側誤指定条件。本文 L.83 の「41.5倍」はこの値の丸め）は、旧版実装（0.5係数が残存するmodel_dual_expfam.py）に基づく結果である。** 図1(b)には対応するバーがなく、本文の記述のみ（KI-003、根拠は`reports/mismatch_audit/mismatch_audit_report_20260708.md` §1）。
- 図1(b)の灰色バー（視覚上の最大値）は23.6倍で、41.45倍とは異なる条件（X=Gaussian/Y=Bernoulli、先行研究固定条件）の値である。
- fixed版（0.5除去）の補助実験では38.97倍という別条件の値が得られているが、本文採用実験とは異なる実装・条件であり、直接比較はできない。
- Scen.Bの7.35倍についても、対応CSV内での条件特定は完了していない（`reports/claims_and_evidence.md` L.13）。
- Scen.Cの「Y=Gaussianが推定を支配している」という解釈は、Exp4 ablation（No X ≈ 提案手法）からの推測であり、理論的証明はない（`reports/claims_and_evidence.md` L.18）。

---

## 11. 研究主張の安全レベル

本節は claim gate である。**完全な分類は §14「2026-08-31 Current Claim Ledger」を正本とする。**
本節は §14 の要約であり、両者が食い違う場合は §14 を優先する。

### 強く言えること

- Dual-ExpFam LSMはGaussian/Bernoulli/Poissonの3分布族について、X・Y両側を任意に指定できる実装が完成している
  （**実装レベルの一般化**であり性能主張ではない。Categorical は未実装）。
- nの増加に伴うRMSE(Z)の改善を3シナリオで確認した（lineage を明記すること）。
- 先行研究（Y=Bernoulli固定、X=Gaussian固定）と同条件での結果が先行研究の再現実装と一致する（差 < 0.001）。
- 分布族の誤指定がRMSE(Z)を悪化させることを複数シナリオで確認した。

### 注意付きで言えること（限定語なしに書かない）

- **潜在次元の選択:** 「明示した人工設定では、歴史的に `BIC` と呼ばれてきた Q ベース基準が
  `K = 3` を候補 `K = 1,…,9` から **30/30 trial** で選択した」（fixed 系列）。
  **limitation:** これは Schwarz BIC の妥当性、一般的な K 回復、
  model-selection consistency のいずれも示すものではない（§12.6、KI-010）。
  **「BIC による k\*=3 の正確な選択」という表現は使わない。**
- 「誤指定により最大41.45倍悪化する」（Scen.C、est X=Gaussian/Y=Poisson。本文表記は丸めた「41.5倍」）→ 旧版実装（0.5係数あり）に基づく結果であることを明記する（KI-001, KI-003）。
- 「Xの誤指定はXを使わないより悪い」（Scen.A）→ Scen.Aのみで確認、他シナリオへの一般化は未確認。
- 「dの増加でRMSE(Z)が改善する」（Scen.A/B）→ Scen.Cでは平坦であり、シナリオ依存。
- per-column heterogeneous-X の利得・MovieLens の primary 正方向・raw-count Poisson の悪化・
  latent-coverage の解釈は、いずれも **QUALIFIED ONLY**。必須の限定語は §14 を参照。

### Historical wording（2026-05〜2026-07、現在は使用しない）

> 「強く言えること」に次の 1 行があった:
> - 3シナリオでBICによるk\*=3の正確な選択を確認した。

この行は **current claim list から除外した**。現在は上の「注意付きで言えること」の
限定付き表現のみが使用可能である。

### まだ言ってはいけないこと

- 「0.5係数を除去した実装の方が優れている」（comparison_quick.csvのratio_fix_oldは条件依存で0.27〜1.23倍、一貫しない）。
- 「Categorical分布にも対応している」（未実装）。
- 「Wine実データで有効性が確認された」（未評価）。
  （2026-07-19更新：この行は2026-05時点の記述。fixed版でのWine評価（BIC k選択・ablation・
  旧版突合）は§8bおよびKI-006のとおり実施済み。ただしWineのYはラベル由来のため
  「自然ネットワークでの有効性確認」とは引き続き言えない。§8bの限定付き記述を正とする。）
- 「精度行列のNewton方向は0.5係数があっても全体として正しい」（限定条件付きでのみ成立しうる、KI-001）。

---

## 12. Phase 6〜7e フェーズ史（2026-08-31 forward update）

`RESEARCH_MASTER.md` は historical frozen report ではなく **current canonical source** である。
そのため §1 / §9 / §11 の現在形の記述は 2026-08-31 に current wording へ直接更新し、
置き換えた旧文面は各節の `Historical note` / `Historical wording` へ隔離した。
本節および §14 が **current interpretation の正本**である。
なお `reports/` 配下の日付入り report・raw CSV・runinfo は frozen record として一切変更していない
（`.claude/rules/historical-records.md`）。

### 12.1 実装 lineage — 6 系統を絶対に混ぜない

| 記号 | lineage | 実体 | 位置づけ |
|---|---|---|---|
| A | 先行研究の印刷された式 | Mikawa et al. 2024 PDF | **1/2 あり**（§6.1 系統1） |
| B | old 0.5 Python | `expfam/src/model_dual_expfam.py`（`DualExpFamLSM`） | **学会予稿の本文採用実験** |
| C | fixed Python | `model_dual_expfam_fixed.py`（`DualExpFamLSMFixed`） | 実データ実験フェーズ（§8b） |
| D | experimental prototype | `experimental/model_dual_expfam_masked.py` / `_nb.py` | pair mask・NB。**本文採用不可** |
| E | objective-consistent experimental | `experimental/model_dual_expfam_consistent.py`（`DualExpFamLSMConsistent` / `DualExpFamLSMPerColumnConsistent`） | Issue #25 で新設。Phase 6 後半〜7e で使用。**本文採用不可** |
| F | per-column prototype | `experimental/model_dual_expfam_percolumn.py`（`DualExpFamLSMPerColumn`） | 列ごとに family を変える prototype。**本文採用不可** |

**標準の提案手法は今も「`family_x` が全 X 列で共通の 1 種類」である。**
F を修論の正式提案手法へ昇格させていない（Issue #23・#27・#31・#33・#43 のいずれの report もこれを明記）。

### 12.2 Issue 別の確定事項（すべて一次 artifact で確認済み）

| Issue | フェーズ | lineage | 確定した内容 | 一次根拠 |
|---|---|---|---|---|
| #23 | per-column 数理/コード監査 | F | interior は独立導出・有限差分と一致（勾配 2.26e-10 / 曲率 7.51e-09）。**PC-001 HIGH**: Poisson clip `[-20,10]` 外で報告目的関数の勾配が 0 なのに実装 score は非零（`eta=11.5, x=3` で score `-22023.47` / precision `22026.47`）。**PC-002 MEDIUM**: Bernoulli floor tail も同様 | `reports/per_column_family/per_column_math_code_audit_20260821.md` |
| #25 | objective-consistent numerics | E | PC-001/PC-002 を **forward 修正**した新 lineage を追加。歴史結果は不変。`numerics_mode` の既定は `legacy` のまま | `reports/per_column_family/objective_consistency_fix_20260821.md` |
| #28 | evidence-driven refinement audit | 監査（read-only） | 一次 CSV から全数値を再計算。failure map F1〜F9。**model modification は 1 つも JUSTIFIED_NOW に到達しない**。**F9 を新規に発見**（MovieLens では genre-only X ですら strict held-out Y を確実には改善しない） | `reports/model_refinement/evidence_driven_model_refinement_audit_20260821.md` |
| #27 | complementary blocks（人工データ、120 fits、10 trials） | E+F | §12.3 参照 | `expfam/results/story_diagnostics/complementary_blocks_consistent_20260821_paired.csv` |
| #31 | matched latent-coverage ablation（人工データ、180 fits） | E+F | §12.4 参照 | `..._matched_latent_coverage_ablation_20260821_interaction.csv` |
| #33 | MovieLens user-disjoint validation（実データ、30 splits・360 fits） | E+F | §12.5 参照 | `expfam/results/real_data/movielens_userdisjoint/movielens_userdisjoint_20260822_paired.csv` |
| #35 | K 選択の理論監査 | 監査（read-only） | §12.6 参照 | `reports/k_selection_theory/k_selection_theory_audit_20260822.md` |
| #37 | K-selection score pilot（42 fits） | E | 同一 fits 上で C1 `bic_impl` と C3 `S_laplace_post` は k=3 を 6/6、C2 `S_cf` は候補範囲上端 k=7 を 6/6 で選択。**同じフィットでも score 定義により選ばれる K が違う** | `expfam/results/k_selection/k_selection_score_pilot_20260823_selection.csv` |
| #39 | held-out K 選択の設計 | 設計のみ | transductive dyad holdout を primary design に採択。leakage matrix と two-canary gate を凍結 | `reports/k_selection_theory/heldout_k_selection_{design,leakage_matrix,implementation_plan}_20260823.md` |
| #41 | harness 実装・two-canary falsification・K={2,3,4} smoke | E | 実装と反証テストのみ。full pilot は未実行のまま次フェーズへ | `tools/research_audit/run_heldout_k_selection_pilot.py` |
| #43 / PR #44 | **Phase 7e full held-out K-selection pilot（42 fits）** | E | §12.7 参照。**実行済み・merge 済み** | `expfam/results/k_selection/heldout_full_pilot_20260824/` |

### 12.3 Issue #27 — sparse-Y での複数属性同時利用（人工データ）

lineage **E+F（objective-consistent per-column prototype）**。generator は
「family block ごとに主に別の潜在次元へ載る」complementary-F 構成（n=80, d=9, K_TRUE=3）。
primary domain は `y_obs_rate = 0.1`、primary endpoint は whole-space Procrustes RMSE_Z。
delta = comparator − per_column（正なら per_column が良い）。

| comparator | sparse (`y_obs_rate=0.1`) | 勝率 | dense (`y_obs_rate=1.0`) | 勝率 |
|---|---:|---:|---:|---:|
| `y_only` | **+0.5122** | 10/10 | +0.0510 | 10/10 |
| `single_bernoulli` | **+0.4218** | 10/10 | +0.0530 | 10/10 |
| `single_poisson` | **+0.3889** | 10/10 | +0.0490 | 10/10 |
| `single_gaussian` | **+0.2030** | 9/10 | **+0.0087** | 9/10 |

（値は `complementary_blocks_consistent_20260821_paired.csv` の `delta_rmse_mean` を本統合時に再計算し一致を確認。
`all_gaussian` は同一列の誤指定 contrast であり M-step optimizer 経路が交絡するため primary ではない：sparse +0.1180 / 8-10。）

**書いてよい表現:**
> 検証した人工データ設定では、Y が疎で属性 block が補完的な潜在情報を持つ場合に、
> 複数属性を同時に用いる条件が比較条件より潜在変数の推定誤差（RMSE_Z）を改善した。
> dense-Y では最良の単一 block 比で +0.009 まで縮小した。

**書いてはいけない:** 「一般に per-column が優れる」「実データでも優れる」「dense-Y では無意味」
「正式提案手法として確立した」。

### 12.4 Issue #31 — latent coverage 操作（人工データ）

同一 F 行を潜在座標 permutation した 2 regime（`complementary` / `full_coverage`）の比較。
primary comparator は `single_gaussian`、唯一の primary estimand は
`I = delta_G(comp) − delta_G(full)`。恒等式 `I = D_G − D_J` は必須分解であり、**I 単独解釈は禁止**。

| 量 | sparse (`0.1`) | dense control (`1.0`) |
|---|---:|---:|
| `delta_G` complementary | +0.2280 (10/10) | +0.0055 (7/10) |
| `delta_G` full coverage | +0.1141 (10/10) | −0.0004 (6/10) |
| **`I`（primary）** | **+0.1139**（std 0.0915、median +0.1364、**9/10**） | +0.0058（6/10） |
| `D_G` | +0.3151 (10/10) | +0.0241 (9/10) |
| `D_J`（必須併記） | **+0.2012 (10/10)** | +0.0183 (10/10) |

**secondary comparator は追随しない**（同 CSV より）:
`single_bernoulli` の `I = −0.2041`（**0/10**）、`single_poisson` の `I = −0.0046`（6/10）。
恒等式誤差は最大 5.55e-17。

**書いてよい表現:**
> Gaussian comparator では coverage / block-rank 操作と整合する差が観測されたが、
> secondary comparator では一貫せず、単一機構として確立していない。

**書いてはいけない:** 「latent coverage が改善原因である」「機構を分離した」「alone / fully isolated」。
理由は ① secondary の符号不一致 ② `D_J = +0.201` が大きく joint 側も同時に動いている
③ 操作は comparator の block rank も同時に下げる（構成上の事実）。

### 12.5 Issue #33 — MovieLens user-disjoint validation（実データ）

旧 MovieLens pilot は 100 映画 subset が **full-data popularity で選ばれていた**ため
selection leakage があった（`prepare_movielens_data.py:428` を一次確認）。
#33 はこれを除く方向で設計された protocol である。

- 30 回の **user-disjoint split**（471 train users / 471 test users / 1 unused）
- **movie selection は各 split の train users のみ**から genre-stratified に 100 本
- 属性は **train-only 由来**（`mean_rating_train` / `log_count_train` Gaussian、`count_train_raw` Poisson）＋ external metadata（genre19 Bernoulli、year Gaussian）
- 学習は `Y_train`、評価は `Y_test` 上の Poisson mean log score per pair。K=3 固定、360 fits

primary estimand `Delta_s = LL_test(mixed_train_log) − LL_test(genre_only)`、trial unit は user split（**n=30**）:

| 量 | 値 |
|---|---|
| mean | **+0.004248** |
| median | +0.006875 |
| positive | **23/30** |
| std | 0.012276 |
| **std / \|mean\|** | **2.89** |
| 経験的 p10–p90 | **[−0.009931, +0.016536]（0 をまたぐ）** |
| 必須分解 `A` / `B` | +0.002009 (22/30) / +0.002239 (19/30)、恒等式誤差 0.000e+00 |
| descriptive positive control `P` | +0.012437 (25/30) |

**書いてよい表現:**
> 平均方向は正だったが、効果量は split 間変動に比べて小さい。

**書いてはいけない:** 「MovieLens で有効性を確認」「statistically significant」「robust superiority」
「30 independent experiments」「causal contribution」。
30 splits は**同じ 943 users を再利用**しており independent replicate ではない。
**p 値・信頼区間・検出力・bootstrap は一切計算していない。**
残存する依存（category E）: `corr(count_full, count_train) = 0.943`、`corr(mr_full, mr_train) = 0.976`。

なお #28 の **F9**（MovieLens では genre-only X ですら strict held-out Y を確実には改善しない）は
未解消であり、#33 の小さい `Delta` は F9 と整合的である。

### 12.5b raw-count Poisson diagnostic（原因は未解明）

同じ #33 の secondary contrast `mixed_train_raw_poisson − mixed_train_log`:

| 量 | 値 |
|---|---|
| mean | **−0.100274** |
| 悪化した split | **29/30** |

**書いてよい表現:**
> 今回の MovieLens 設定・モデル・raw-count 表現では、raw count を Poisson-X として扱った条件が
> log-count 表現より一貫して悪化した。

**書いてはいけない:** 「Poisson は悪い」「Poisson は実データに不適」「Poisson-X は一般に有害」
「intercept 欠如が原因」「curvature が原因」。
**原因は未解明**である（#28 §9.3：intercept / raw scale / Poisson 曲率 / X 側過分散 var/mean=6.17
が互いに交絡し、F4 の remedy はいずれもこの 4 つを同時に除去してしまうため識別できない）。

### 12.6 Issue #35 — K 選択基準の理論的位置づけ

**現行基準を「Schwarz BIC」と呼ばない**（KI-010・root `CLAUDE.md` §5）。
関数名 `calc_bic_dual`・CSV 列名 `BIC`・過去結果の呼称は provenance のため**変更しない**。

一次確認で新たに確定したこと:

1. 現行実装の基準は `BIC_impl = −2·Q_strict + p̂·ln n`、`p̂ = kd − k(k−1)/2 + d·1{GX} + 1·1{GY}`。
   当てはまり項は **EM の Q 関数（完全データ対数密度の近似事後期待値）**であり、観測データ周辺尤度ではない。
2. `scale_Z` により潜在変数の事前分布項が定数 `−(nk/2)(1+ln2π)` に退化し、
   **潜在次元 1 あたり `n(1+ln2π) ≈ 2.84n` の固定的な次元罰則**が実効的に働いている。
   同じ構造は先行研究の MATLAB 実装にも存在する。
3. **先行研究の印刷された Eq.(26) の当てはまり項 `ln L`（Eq.16）は `p(Z)` を含まず、
   かつ `z_i` に条件づけた量であり `Z` を積分していない**（2026-08-23 に原論文 PDF を一次確認）。
   → **本研究の基準も先行研究の印刷された基準も、Schwarz BIC が対象とする観測データ周辺尤度には対応しない。**

区別すべき 3 つの量: (i) `ln p(X,Y|Z,θ)`（先行研究の印刷式）、(ii) `ln p(Z,X,Y|θ)`（本研究の当てはまり項）、
(iii) `ln p(X,Y|θ)`（Schwarz BIC の対象）。

**書いてよい:** 「その設定において、歴史的に `BIC` と呼ばれてきた基準が k=3 を選択した」。
**書いてはいけない:** 「BIC で正しい K を選べる」「theoretically valid K recovery」「consistency」
「true K を一般に回復する」「先行研究の Eq.(26) は standard Schwarz BIC である」。

### 12.7 Issue #43 / PR #44 — Phase 7e full held-out K-selection pilot（**実行済み**）

**Phase 7e は未実行ではない。** Issue #43 は close 済み、PR #44 は main へ merge 済み。

| 項目 | 値 |
|---|---|
| main merge commit | `ec6e646c2596527338a7e28e6076549a6aa50e6a` |
| RUN_CODE_SHA | `b9311e64a7b36c0a8a9704fff0ee7b38efe36a8a` |
| scientific result commit | `b816836f95945024f56ed7a4ac619e809bc16ded` |
| post-run audit hardening commit | `18176e545c7732aeb91c68032b18f3a3e8a6db0f` |
| lineage | E（`DualExpFamLSMConsistent`、experimental prototype、**本文採用不可**） |
| 構成 | `family_x=poisson` / `family_y=bernoulli` / `K_TRUE=3` / n=75 / d=15 / L=5 / num_iter=8 / `test_ratio=0.20` |
| manifest | candidate K = 1..7 × replicate {1,2,3} × start {1,2} = **42 fits** |
| primary score | held-out Bernoulli raw-eta plug-in mean log score（`y·eta − logaddexp(0,eta)`、`eta = w0 + w z_i^T z_j`） |

**結果**（`replicate_selection.csv` / `aggregate_summary.csv` を本統合時に再確認）:

| replicate | selected K |
|---:|---:|
| 1 | **3** |
| 2 | **3** |
| 3 | **5** |

selected-K counts `{3:2, 5:1}`、`true_k_selected_count = 2`、**descriptive recovery rate 2/3**。
integrity: 42/42 clean、internal retry 0 / warning 0 / q_failure 0 / NaN 0 / 非有限 0、score target ちょうど 3、score rows 42。
独立 self-audit（artifact のみ、harness selector を import しない）verdict PASS、BLOCKER 0 / HIGH 0。

**書いてよい表現:**
> frozen held-out K-selection protocol を 3 dataset replicate で記述的に評価したところ、
> K=3 が 2 replicate、K=5 が 1 replicate で選択された。

**書いてはいけない:** 「true K recovery 66.7% という一般性能」「consistency」「held-out criterion の優越」
「BIC より優れる」「manuscript-level proof」「実データ妥当性」「漸近的結果」。
これは **1 synthetic setting × 3 independently generated dataset replicate だけの descriptive pilot** である。

### 12.8 Phase 7e の provenance limitation

`reports/k_selection_theory/heldout_k_selection_full_pilot_provenance_addendum_20260831.md` を参照。

repository evidence から確認できるのは:

- frozen RUN_CODE_SHA の後に、**42 clean fits からなる 1 successful recorded execution** が保存 artifact として存在すること
- researcher の procedural record では rerun なしと記録されていること

確認**できない**のは:

- `stdout.log` を生成した outer capture command。**`NOT RECOVERABLE FROM REPOSITORY EVIDENCE`**

したがって、**「削除された先行実行が存在しないことまで含めて externally proven exactly once」とは書かない。**
この限定は 42 saved fit rows・selected K・arithmetic・seed・hash・leakage isolation の
いずれも無効化しない。

---

### 12.9 Issue #59 — Phase 8b K_TRUE robustness full sweep（Attempt 2、**実行済み**）

Phase 7e で凍結した held-out K-selection protocol を、**generator の `K_TRUE` を {1,2,4,5} へ
拡張**して exactly once 実行した本番 sweep。

| 項目 | 値 |
|---|---|
| role 1: approved scientific baseline | `68c78e1191889609dead05ea5a9fb11525ce92e2` |
| role 2: reviewed full-execution main | `ddc9b0b4c38da995fedf43ceef12f17dfb4db353` |
| role 3: runtime RUN_CODE_SHA | `ef85b4c921546129b8d4f7440f8a09a41aa652e5` |
| frozen protocol hash | `2d19c5fe6edadd0823925ed7dd051cb27837bccf51d5102e0bcee53271654eb9` |
| human approval | Issue #59 comment `5529711820` |
| execution attempt | `phase8b-full-attempt-2` |
| lineage | E（`DualExpFamLSMConsistent`、experimental prototype、**本文採用不可**） |
| 構成 | `family_x=poisson` / `family_y=bernoulli` / n=75 / d=15 / L=5 / num_iter=8 / `test_ratio=0.20` / `mask_design=S_C` / `random_design=CRN` / `hierarchy=H3_A` |
| manifest | estimand {A,B} × `K_TRUE` {1,2,4,5} × replicate {1,2,3} × candidate K 1..7 × start {1,2} = **336 fits**（A 168 / B 168） |
| estimand A（primary） | `w_true = 1.5`（`K_TRUE` によらず固定） |
| estimand B（sensitivity） | `w_K = 1.5 · sqrt(3 / K_TRUE)`（`w_K^2 · K` を ensemble で一致） |
| 選択基準 | **frozen Phase 7e held-out Bernoulli raw-eta plug-in mean log score**、2 start の非加重平均、tie tolerance 1e-12、smallest-K tie rule |
| report | `reports/k_selection_theory/k_true_robustness_full_report_20260904.md`（artifact から自動生成） |

**選択基準についての明示（KI-019）:** 本実験の K 選択に **`Q_strict` / EM の Q 関数基準 /
ICL-type complete-data criterion / Schwarz BIC / marginal likelihood は一切使っていない。**
§12.6 / KI-010 の「Q ベース基準を Schwarz BIC と呼ばない」という論点は `calc_bic_dual`
（legacy 基準）に関するものであり、Phase 7e/8b の held-out 予測スコアとは**別物**である。
両者を同一視した記述を書かない。

**結果**（`selection_matrix.csv` の selected K。本統合時に per-fit 生スコアから frozen
selector で再導出して一致を確認した）:

| estimand | K_TRUE=1 | K_TRUE=2 | K_TRUE=3（anchor） | K_TRUE=4 | K_TRUE=5 | 新規グリッド真値一致 |
|---|---|---|---|---|---|:---:|
| A（primary） | 1, 1, 1 | 2, **3**, 2 | 3, 3, **5** | 4, 4, 4 | **4**, **4**, 5 | **9/12** |
| B（sensitivity） | 1, 1, 1 | 2, **3**, 2 | 3, 3, **5** | 4, 4, 4 | **3**, **4**, 5 | **9/12** |

- 新規グリッド `{1,2,4,5}` の合算は **18/24**。K_TRUE 別は 1: 6/6、2: 4/6、4: 6/6、5: 2/6。
- **`K_TRUE=3` の行は Phase 7e artifact（`heldout_full_pilot_20260824/`）の READ-ONLY 参照**
  であり、Attempt 2 では 1 fit も再実行していない（`phase7e_rerun_count = 0`）。
  **A 行と B 行は同一の anchor 証拠を共有しており、6 個の独立実験ではない。**
- A と B で選択が異なるのは **`K_TRUE=5` replicate 1 のみ**（A=4 / B=3）。
- 全 24 セルで tie 候補は 1 個であり、tie rule は発動していない。

**integrity:** 336/336 clean、`internal_retry` 0 / `warning_count` 0 / `q_failure` false /
`nan_occurred` false / `finite_state` 全 True、`failure.json` 不在、retry 0 / replacement 0 /
seed rescue 0 / tolerance 緩和 0 / canary rerun 0 / smoke rerun 0、`config_gate` 103 件 passed、
`leakage_gate` 336 行すべて pre/post passed かつ `fit_boundary_status = clean`、
`mask_provenance` 24 セル `anchor_match = True`。独立 artifact 監査（runner を import しない）
verdict **PASS**、BLOCKER 0 / HIGH 0 / MEDIUM 0。

**証拠の数え方:** 新規 **336** + Phase 7e anchor **42**（READ-ONLY 再利用）= **統合ユニーク 378**。
**420 ではない。** A と B が同じ anchor 42 fits を参照するため、anchor を二重計上しない。

**書いてよい表現:**
> 凍結した held-out 予測スコアによる K 選択では、`K_TRUE=1` および `K_TRUE=4` では 3 反復
> すべてで真値が選択され、`K_TRUE=2` では 2/3、`K_TRUE=5` では 1/3 で真値が選択された。
> `K_TRUE=5` では候補集合に 5 より大きい K も含まれる一方、選択結果は低い K 側に寄る傾向が
> 観測された。ただし各条件 3 反復のみであり、本結果は有限標本における記述的結果として解釈する。

**書いてはいけない:** consistency / asymptotic consistency / universal K recovery /
K-selection consistency / Schwarz BIC / BIC consistency / 理論保証 / 本合成設定を超える一般化 /
「Phase 8b は Q_strict・ICL-type・BIC で K を選んだ」（事実として誤り）/
「`K_TRUE=3` について A と B が独立に 6 セル分の証拠を与える」。

`K_TRUE=5` の under-selection の**原因は未同定**である。report §8 の margin 表は選択の
僅差さを記述するだけであり、原因を説明していない。A/B の 1 セル差から信号強度スケーリングの
一般的効果を推論しない。

### 12.9b Attempt 1（中断・provenance のみ）

`expfam/results/k_selection/k_true_robustness_full_20260902/` は同一 protocol hash で開始した
先行試行が operator interrupt により `fit_index = 3` で停止したものである。

- `status = FAILED` / `attempted_fit_count = 3` / `clean_fit_calls = 2` / `scored_rows = 0`
- `retry_count = 0` / `replacement_fits_executed = 0` / RUN_CODE_SHA `1946953ffc7e7db586dda2933c9a25a6f0235d07`
- 位置づけ: **ABORTED_BY_OPERATOR_INTERRUPT / provenance only / no scientific use**

**この 2 clean fits を科学的主張の根拠にしない。** Attempt 2 は本結果を一切再利用していない
（`partial_results_reused = False`）。artifact は削除・改変しない。

---

## 13. 修論 backbone（2026-08-31 時点）

既存証拠から再確認したものであり、新しい主張を発明していない。

### A. methodological backbone

Gaussian / Bernoulli / Poisson について、X 側・Y 側を独立に指定できる指数型分布族一般化を実装した。
人工データ 3 シナリオでの回復・n 依存・誤指定の影響を評価した。
**Categorical は正式対応していない**ため「指数型分布族すべてに対応」とは書かない。

### B. conditional synthetic evidence

Y が疎で、属性 block が補完的な潜在情報を持つ人工条件では、複数属性を同時に用いることが
潜在構造推定を改善した（§12.3）。

観測された範囲は次のとおりである。Issue #27 の complementary synthetic 実験と、
Issue #31 の complementary regime における **primary comparator `single_gaussian`** では、
sparse-Y 条件で同方向の改善が観測された。
**ただし Issue #31 の Bernoulli / Poisson secondary comparator は同方向に追随しておらず
（`I = −0.2041`（0/10）、`I = −0.0046`（6/10））、一般的な per-column 優位や
単一機構の再現を意味しない**（§12.4）。
**「2 つの生成設定で per-column benefit を再現した」とは書かない。**
regime 依存であり無条件の主張ではない。

### C. real-data characterization

MovieLens の leakage-safe な user-disjoint protocol では、平均方向は小さな正方向だったが、
split variability に対して小さく、**明確な優位は確認されなかった**（§12.5）。

### D. negative / limitation evidence

raw-count Poisson 表現は今回の MovieLens 条件で大きく悪化した（§12.5b）。
family の選択だけでなく、**attribute representation / model structure / scaling / intercept・dispersion 設計**
などが結果を左右しうることを示唆するが、**どれが原因かは未検証であり、確定原因として書かない。**

### E. K-selection limitation

歴史的な Q-based criterion の理論的位置づけを監査し（§12.6）、
代替として held-out plug-in criterion を pilot 評価した（§12.7）。
**K-selection 問題を理論的に解決したとは主張しない。**

### backbone の要点

修論の骨格は **A を backbone、D+E を honest boundary、B を条件付き拡張、C を実データ characterization**
として構成する。**B と C を同じ主張の証拠として無条件に混ぜない**（人工データと実データを分けて記述する）。

---

## 14. 2026-08-31 Current Claim Ledger

**この表が「2026-08-31 時点で修論に何を書いてよいか」の正本である。**
lineage 記号は §12.1 に対応する。
synthetic evidence と real-data evidence を**同一 claim の証拠として無条件に混ぜない**。

### 分類の定義

| 分類 | 定義 |
|---|---|
| **ALLOWED** | 記載された scope 内で、**追加の本質的な但し書きを付けなくても** current thesis statement として使用可能な事実 |
| **QUALIFIED ONLY** | 重要な scope / limitation の**併記が不可欠**な主張。限定語を落とすと別の（許されない）主張になる |
| **NOT ALLOWED** | 現在の証拠では書いてはいけない主張 |
| **UNRESOLVED** | 科学的に未解決である事実。「未解明」と書く |

**同一の claim を ALLOWED と QUALIFIED ONLY に二重登録しない。**

### ALLOWED

| claim | evidence | lineage | evidence type | limitation | 修論での用途 |
|---|---|---|---|---|---|
| Gaussian / Bernoulli / Poisson について X 側・Y 側を独立指定できる実装が存在する | `expfam/src/model_dual_expfam.py` の `VALID_FAMILIES`、`test_dual_expfam.py` 5 テスト PASS | B（C/D/E/F も継承） | code | **実装レベルの一般化**であり性能主張ではない。Categorical は未実装 | §A backbone |
| fixed 系列の人工データで、n の増加に伴い RMSE(Z) が改善した | `fixed_official/exp2/fixed_exp2_n_sweep_improvement.csv`（A −49.3% / B −41.2% / C −58.6%、n=50→300） | **C（fixed）** | synthetic | 明示した生成設定内の有限標本観測。旧 0.5 系列（B）の数値と同じ表に置かない | §A |
| fixed 系列の人工データで、分布族の誤指定が RMSE(Z) を悪化させた | `fixed_official/exp4/fixed_exp4_scen_{a,b,c}_ratios.csv`（誤指定最悪 A 4.3414× / B 9.0405× / C 40.3706×、いずれも `fix_w=False` / `fix_x=False`） | **C（fixed）** | synthetic | ablation 行（`fix_w=True`）は誤指定倍率ではない（§15）。旧 0.5 系列の倍率と並べない | §A |
| 先行研究と同条件での結果が先行研究の再現実装と一致する（RMSE(Z) 差 < 0.001、5 試行） | `reproduction/results/comparison/comparison_main_table.csv` | B | synthetic | 5 試行のみ | §A |

### QUALIFIED ONLY（必須の限定語なしに書いてはいけない）

| claim | 必須の限定 | evidence | lineage | evidence type |
|---|---|---|---|---|
| **有限の人工設定における潜在次元選択**：「明示した fixed 系列の人工設定において、歴史的に `BIC` と呼ばれてきた Q ベース基準が `K = 3` を候補 `K = 1,…,9` から **30/30 trial** で選択した」 | ①明示した有限の実験設定内の観測に限る（n=150 の 1 点）②**Schwarz BIC ではない**（KI-010）③**一般的な true-K recovery ではない** ④**model-selection consistency ではない** | `fixed_official/exp1_k9/fixed_exp1_bic_k1to9_bestk_by_trial.csv` | **C（fixed）** | synthetic |
| **per-column（複数属性同時利用）の利得**：「検証した sparse-Y complementary な人工設定では、per-column joint model が指定した comparator より RMSE_Z を改善した」 | ①**experimental prototype**（本文採用不可）②**人工データのみ** ③**sparse-Y regime** に限る ④**complementary な属性構造**に限る ⑤**一般的優位ではない** ⑥**実データでの有効性ではない**。dense-Y では最良単一 block 比 +0.0087 まで縮小することを併記 | `complementary_blocks_consistent_20260821_paired.csv`（+0.5122 / +0.4218 / +0.3889 / +0.2030、10・10・10・9 of 10）、§12.3 | **E+F** | synthetic |
| **Phase 7e の held-out K 選択**：「frozen held-out protocol の 3 dataset replicate で K=3 が 2、K=5 が 1 選択された」 | ①**3 replicate の記述値**であり一般性能ではない ②**consistency ではない** ③**BIC・C1/C2/C3 に対する優越ではない** ④**実データ妥当性ではない** ⑤prototype（lineage E、本文採用不可） | `heldout_full_pilot_20260824/replicate_selection.csv`, `aggregate_summary.csv`、§12.7 | **E** | synthetic |
| MovieLens primary の正方向 | 「**平均方向は正だが split 間変動に比べて小さい**」＋「有意性検定・CI・検出力は未計算」＋「30 splits は独立 replicate ではない」 | §12.5 | E+F | real-data |
| raw-count Poisson の悪化 | 「**今回の MovieLens 設定・モデル・raw-count 表現では**」＋「**原因は未解明**」 | §12.5b | E+F | real-data / diagnostic |
| latent-coverage 機構の解釈 | 「Gaussian comparator では整合する差が観測されたが **secondary comparator では一貫せず、単一機構として確立していない**」＋「`D_J = +0.201` を必ず併記」 | §12.4 | E+F | synthetic |
| 誤指定倍率（23.6× / 41.45× / 38.97× など） | lineage・true/est 条件・根拠 CSV 行を必ず明記（KI-003）。**3 つを同一条件の値として並べない** | §10、`reports/mismatch_audit/mismatch_audit_report_20260708.md` | B / C | synthetic |
| 0.5 係数まわりの主張 | 「**Newton 方向が全体として正しいとは断定できない**」を必ず付記（KI-001）。原論文の印刷式には 1/2 が**ある** | §6.1 | A / B / C | theory/code audit |

### NOT ALLOWED（書いてはいけない）

| claim | 理由 |
|---|---|
| per-column が一般に優れる | dense-Y では最良単一 block 比 +0.009、`single_vs_joint` では符号すら逆（−0.0004）。regime 依存 |
| per-column が実データでも有効 | #33 の primary は spread 支配（std/\|mean\| 2.89、p10–p90 が 0 をまたぐ）。#28 F9 も未解消 |
| per-column を正式提案手法として採用した | prototype。root `CLAUDE.md` §3、#23/#27/#31/#33/#43 の全 report が明記 |
| Poisson は実データに不適 / Poisson-X は一般に有害 | #33 は 1 データセット・1 表現・1 モデル構成の diagnostic。原因未解明 |
| raw-count 悪化の原因は intercept 欠如（または curvature） | #28 §9.3 で 4 要因が交絡、識別されていない |
| 「Schwarz BIC で潜在次元を選択した」 | `Q_strict` は観測データ周辺尤度ではない（KI-010、#35） |
| 「先行研究の Eq.(26) は standard Schwarz BIC である」 | Eq.(16) の `ln L` は `z_i` に条件づけた量。2026-08-23 一次確認（#35 E25） |
| K-selection の consistency を確認した | 未証明。#37 は同一 fits 上で score 定義により選ばれる K が異なることを示した |
| 3 replicate の 2/3 を「true K recovery 66.7%」という一般性能として提示 | 3 replicate の記述値。CI を伴う推定量ではない |
| Categorical 分布に対応している | 未実装（KI-005） |
| MovieLens で提案手法の有効性を確認した / statistically significant / 30 independent experiments / causal contribution | #33 report §9 の禁止句リスト |
| 「削除された先行実行が存在しないことまで含めて externally proven exactly once」 | stdout capture の outer command が復元不能（§12.8） |
| 旧 0.5 系列（B）と fixed 系列（C）と consistent 系列（E）の数値を同じ表・図に並べる | KI-002、root `CLAUDE.md` §3 |

### UNRESOLVED（証拠が無い。「未解明」と書く）

| 項目 | 何が足りないか |
|---|---|
| raw-count Poisson 悪化の原因 | intercept / raw scale / Poisson 曲率 / X 側過分散（var/mean 6.17）が交絡しており、既存の条件では識別できない（#28 §9.3） |
| K-selection の formal theory | selection target が未確定（#35 U16）。K 選択の n 依存性は一度も測定されていない（U2）。本モデルの RLCT 未知（U5） |
| per-column の正式昇格可否 | #28 §20-8 は NOT_JUSTIFIED。#27 は 2 設定の人工データ、#31 は機構同定に失敗、#33 は実データで小さい |
| X intercept / dispersion-aware count family の設計 | #28 §19 でいずれも JUSTIFIED_NOW に到達していない |
| Phase 7e の stdout capture 方式 | `NOT RECOVERABLE FROM REPOSITORY EVIDENCE`（§12.8） |
| Cora で基準ごとに最適 k が割れる理由 | `Σ_i ln det A_i` 未測定、試行 3 のみ（KI-011、#35 PL1） |
| K 選択の n 依存性 | 一度も測定されていない（#35 U2） |
| Categorical family での挙動 | 未実装（KI-005） |
| Cora subset（n=280）の結果が full Cora（n=2708）に一般化するか | 未検証（KI-011） |
| 旧 0.5 系列（B）と fixed 系列（C）の条件対応 | 完全な condition-by-condition correspondence table は canonical artifact として存在しない（§15） |

---

## 15. KI-001 — 旧 0.5 系列と fixed 系列の対応（現状の監査結果）

修論で両系列を併記する場合に必要な対応関係の**現状**を記録する（新実験なし、既存 CSV のみ）。

| 条件 | 旧 0.5 系列（B）の値 | 根拠 CSV | fixed 系列（C）の値 | 根拠 CSV |
|---|---|---|---|---|
| Scen.A 誤指定 最悪比 | 3.41×（Bern-Bern） | `expfam/results/exp_scenario_A_exp4_mismatch.csv` | **4.3414×**（XBern_YBern） | `expfam/results/fixed_official/exp4/fixed_exp4_scen_a_ratios.csv` |
| Scen.B 誤指定 最悪比 | 7.35×（条件特定は未完了） | `expfam/results/exp_scenario_B_exp4_mismatch.csv` | **9.0405×**（XPois_YBern） | `..._scen_b_ratios.csv` |
| Scen.C 誤指定 最悪比 | 41.45×（est X=Gaussian / Y=Poisson、両側誤指定） | `expfam/results/exp_scenario_C_exp4_mismatch.csv` | **40.3706×**（XPois_YBern） | `..._scen_c_ratios.csv` |
| Scen.C 図1(b) 灰色バー | 23.6×（est X=Gauss / Y=Bern、X 側のみ誤指定） | 同上 | — | — |
| fixed 単独 mismatch grid | — | — | 38.97×（true=bern/gauss, est=poisson/bernoulli） | `expfam/results/distribution_mismatch_fixed/mismatch_fixed_summary.csv` |

**fixed 系列の値は ablation 行を除いた誤指定条件の最大値である**（2026-08-31 に一次 CSV で確認）。
`fixed_exp4_scen_c_ratios.csv` には `fix_w=True` の ablation 条件 `X_only` が **46.8637×** で存在するが、
これは誤指定ではなく Y を落とす ablation であり、誤指定倍率として引用しない。

**この表は「同じ条件の前後比較」ではない。** 旧 0.5 系列の最悪条件と fixed 系列の最悪条件は
**est 条件が異なる**（Scen.C: 旧 XGauss/YPois vs fixed XPois/YBern）。
したがって **41.45× → 40.37× を「0.5 除去の効果」と読んではいけない。**

**current state:** 旧 0.5 系列の各 est 条件に対して fixed 系列の同一 est 条件の値を並べた
**完全な condition-by-condition correspondence table は canonical artifact として存在しない。**

したがって lineage を跨いで数値を比較する場合は、
**一次 CSV・条件（true / est family、`fix_w` / `fix_x`）・seed・model lineage を個別に確認する必要がある。**
上の表は最悪比のみを並べたものであり、条件が一致していないため
**差分を「0.5 除去の効果」として読んではならない。**

---

## 16. 2026-09-04 Claim Ledger 追記（Phase 8b Attempt 2）

**§14 の 2026-08-31 ledger は当時の記録としてそのまま残す。** 本節はその後に確定した
Phase 8b Attempt 2（§12.9）に対応する差分だけを追記する。分類の定義は §14 と同じ。

### QUALIFIED ONLY（追加）

| claim | 必須の限定 | evidence | lineage | evidence type |
|---|---|---|---|---|
| **Phase 8b の K_TRUE robustness**：「frozen held-out K-selection protocol を `K_TRUE ∈ {1,2,4,5}` へ拡張したところ、真値一致は A 9/12・B 9/12（K_TRUE 別に 1: 3/3、2: 2/3、4: 3/3、5: 1/3）であった」 | ①**各条件 3 replicate のみの記述値**であり一般性能ではない ②**consistency / asymptotic consistency ではない** ③**Schwarz BIC ではなく、`Q_strict`・ICL-type でもない**。held-out plug-in 予測スコアである（KI-019）④**本合成設定を超えて一般化しない** ⑤prototype（lineage E、本文採用不可）⑥A（primary）と B（sensitivity）を分けて報告する | `k_true_robustness_full_attempt2_20260904/selection_matrix.csv`, `full_summary.json`、§12.9、`reports/k_selection_theory/k_true_robustness_full_report_20260904.md` | **E** | synthetic |
| **`K_TRUE=5` での低い K への偏り**：「`K_TRUE=5` では候補集合に 5 より大きい K も含まれる一方、選択結果は低い K 側に寄る傾向が観測された」 | ①**原因は未同定**（margin を記述しただけで説明していない）②A 1/3・B 1/3 の記述値 ③候補上限 7 の制約ではないことのみが言える | §12.9、同 report §8 | **E** | synthetic |

### NOT ALLOWED（追加）

| claim | 理由 |
|---|---|
| 「Phase 8b の K 選択は `Q_strict` / EM の Q 関数基準 / ICL-type complete-data criterion / Schwarz BIC / marginal likelihood で行った」 | **事実として誤り。** Phase 7e/8b は held-out Bernoulli raw-eta plug-in mean log score を使う（KI-019、§12.9） |
| 「`K_TRUE=3` について A と B が独立に 6 セル分の証拠を与える」 | 両者は同一の Phase 7e anchor 42 fits を READ-ONLY 参照している（§12.9） |
| 「Phase 8b の統合証拠は 420 fits である」 | 336 新規 + 42 anchor = **378**。anchor を A/B で二重計上しない（§12.9） |
| 「Phase 8b は K-selection consistency を示した」 | 各条件 3 replicate の有限標本記述値。CI を伴う推定量ではない |
| Attempt 1（`k_true_robustness_full_20260902/`）の 2 clean fits を科学的根拠にする | ABORTED_BY_OPERATOR_INTERRUPT / provenance only（§12.9b） |

### UNRESOLVED（追加）

| 項目 | 何が足りないか |
|---|---|
| `K_TRUE=5` での under-selection の原因 | held-out score の margin を記述しただけで、識別性・有効次元・推定分散のいずれが効いているかを分離していない（§12.9） |
| A（`w` 固定）と B（`w_K` スケーリング）の差が 1 セルに留まった理由 | 12 セル中 1 セルの差から機構を推論できない。信号強度スケーリングの効果は未測定（§12.9） |
| K 選択の n 依存性（Phase 8b でも未測定） | Phase 8b は n=75 の 1 点のみ。§14 UNRESOLVED の同項目は解消していない |

---

## 17. 2026-09-04〜05 true-K identifiability / clean K-selection フェーズ

先生のゼミ指摘 1〜5（生成モデルの成立性・「真の K」の定義・K_TRUE と識別可能性・
複数分布での model selection 理論・`n→∞` の一致性）に対する到達点。

### 17.1 成果物

| 種別 | ファイル |
|---|---|
| 理論監査（敵対レビュー済み） | `reports/identifiability/true_k_identifiability_hardened_20260904.md` |
| 敵対レビュー記録 | `reports/identifiability/true_k_identifiability_review_20260904.md` |
| clean generator 仕様 | `reports/identifiability/canonical_clean_generator_spec_20260904.md` |
| clean generator 実装 | `expfam/src/experimental/data_generator_canonical.py`（`canonical-clean-v1`） |
| 凍結プロトコル | `reports/identifiability/clean_true_k_experiment_protocol_20260904.md` |
| 実験結果 | `reports/identifiability/clean_true_k_results_20260905.md` |
| 先生向け説明 | `reports/identifiability/teacher_discussion_summary_20260905.md` |
| 数値検証 | `tools/research_audit/verify_identifiability_identities.py`（81 rows / failure 0） |

### 17.2 「真の K」の定義（指摘 2）

```
M_K = { p_{theta,K}(X,Y) : theta ∈ Theta_K },   p_{theta,K}(X,Y) = ∫ p(Z) p(X|Z) p(Y|Z) dZ
K*  = min { K : P0 ∈ M_K }
```

**必ず区別する:**

| 記号 | 意味 |
|---|---|
| `K_TRUE` | generator が `Z` を何列作ったかという**手続き上の数** |
| `K*` | その観測分布の**最小潜在次元** |
| `K^rank` | population Gram `FF^T` の階数（**X 周辺しか見ない**） |

`K* ≤ K_TRUE` で等号は自明でない。`K^rank ≤ K*` で等号も自明でない。

**仮定 M-closed:** `K*` は `P0 ∈ ⋃_K M_K` の下でのみ定義される。
**実データでは一般に成立せず、その場合 `K*` は存在しない。**

### 17.3 証明された命題

| ID | 命題 | 条件 |
|---|---|---|
| **P1** | canonical Poisson-X で `‖f_l‖² = 2 log E[X_l]`、`f_l·f_m = log(E[X_lX_m]/(E[X_l]E[X_m]))` により `FF^T` が population moment から復元でき、**X 周辺の最小潜在次元**が `rank(FF^T)` に一致する | unclipped exp link、`rank(F)=K`（**generic 条件**）。`K*` ではなく `K^rank` の主張 |
| **P2** | canonical Gaussian-Y で `M_S(t)=(1−t²)^{−K/2}` より `κ_2=K, κ_4=6K, κ_6=120K`、ゆえに `w²=κ_6/(20κ_4)`、`K=κ_4/(6w⁴)`、`σ_y²=κ_2−Kw²` が**単一 dyad 周辺**から決まる | `w ≠ 0`。単一 dyad からは `w` の符号は決まらない |
| **P3** | canonical Gaussian-Y で `{P ∈ M_K : w≠0}` は `M_{K+1}` と交わらない（族は入れ子でない） | `w ≠ 0`。**`M_K` と `M_{K+1}` 自体は `w=0` 切片で交わる** |
| **P6** | canonical Poisson-Y で `E[Y^r] < ∞ ⟺ \|w\| < 1/r`（平均 `\|w\|<1`、分散 `\|w\|<1/2`） | unclipped link。**KI-020** |
| **P8** | canonical Gaussian-Y で `E[(Y_ij−w0)(Y_ik−w0)(Y_jk−w0)] = w³K` により **`w` の符号が三角形（`n≥3`）から識別される** | `w ≠ 0`, `n ≥ 3` |

**反例:** Bernoulli-X `d=1` では `E[X_l]=1/2` が `K, f_1` によらず成り立ち、
**X 周辺からは** `K` を識別できない。**joint の反例ではない**（`w≠0` の Gaussian-Y なら P2 で識別される）。

**一般則:** 周辺分布で示した**肯定的**識別性は joint へ移送できる。**否定的**な非識別性は移送できない。

### 17.4 BIC が正当化されない理由（指摘 4）— §12.6 / KI-010 の forward update

非入れ子性（P3）が無効にするのは**尤度比検定の χ² 近似と Wilks の定理**であって、
**Schwarz BIC の導出ではない**（BIC はモデルごとの Laplace 近似で、非入れ子比較に使うのは標準的）。

**BIC がこのモデルで正当化されない実際の理由は 3 つ:**
①潜在変数モデルの**特異性**（`O(K)` 不変性で Fisher 情報が退化。`rank(F)<K`・`w=0` 上でも退化。RLCT 未知＝§14 U5）、
②**境界パラメータ**、③**有効標本数が未定義**（ノード数 `n` / dyad 数 `n(n−1)/2` / X 要素数 `nd`。
`calc_bic_dual` は `log n` にノード数を使う。潜在変数 `Z` は `n` とともに増える incidental parameter）。

### 17.5 clean true-K n-sweep（実行済み、独立監査 PASS）

| 項目 | 値 |
|---|---|
| artifact | `expfam/results/k_selection/clean_true_k_asymptotics_20260904/` |
| protocol hash | `547880a16aef6530cfdf7903c4e32f16062397e0bacc0c109d5c77fb9892ccc0` |
| run_code_sha | `63e1202258a71256a55732fc1832db13d7f7b2bd` |
| generator | `canonical-clean-v1`（**historical generator ではない**） |
| lineage | E（`DualExpFamLSMConsistent`、`numerics_mode="consistent"`）。**旧 0.5 lineage 不使用**。**本文採用不可** |
| 構成 | X=Poisson / Y=Bernoulli / d=15 / L=5 / num_iter=8 / test_ratio=0.20 |
| grid | `K_TRUE ∈ {1,3,5}`（反復 4/4/8）× `n ∈ {50,75,100,150}` × candidate K 1..7 × start {1,2} |
| **fits** | **896**（expected=actual=unique、重複 0・欠番 0） |
| integrity | retry 0 / replacement 0 / seed rescue 0 / tolerance 緩和 0 / resume False / NaN 0 / 非有限 0 |
| 独立監査 | **PASS**（BLOCKER 0 / HIGH 0 / MEDIUM 0 / LOW 2） |

**結果（真値一致数、`K_TRUE` との一致であって `K*` との一致ではない）:**

| criterion | K_TRUE=1（n=50→150） | K_TRUE=3 | K_TRUE=5 | 合計 |
|---|---|---|---|---|
| **S1** held-out predictive | 4/4, 4/4, 4/4, 4/4 | 1/4, 1/4, 3/4, 4/4 | 2/8, 0/8, 4/8, **8/8** | **39/64** |
| **S2** Q-based | 4/4, 4/4, 4/4, 4/4 | 2/4, 3/4, 4/4, 4/4 | 0/8, 0/8, 1/8, **7/8** | **37/64** |
| **S3** plug-in conditional | 0/4 ×4 | 0/4 ×4 | 3/8, 0/8, 0/8, 0/8 | **3/64** |

`K_TRUE=5` の平均 selected K: S1 `2.62 → 3.00 → 4.50 → 5.00`、S2 `1.75 → 3.25 → 3.62 → 4.88`。

**重要な限定:**

- **平均 selected K は単調増加したが、真値一致数は単調ではない**（S1 は n=75 でいったん 0/8 に下がる）。
- 誤りの向きは一貫して **under-selection**。
- **`K_TRUE=1` の 4/4 は good recovery の証拠ではない。** 支配的な誤り方が under-selection であり、
  `K=1` は候補集合の下端で、凍結 selector は同点時に最小 K を選ぶ。**下限効果と交絡している。**
- **S3 は 3/64 でほぼ全セルが候補上限 `K=7`。** `ln p(Z)` を含めず `Z` を積分しない基準は、
  潜在次元を増やすほど代入した `Ẑ` への当てはまりが良くなり罰則が追いつかない。
  **これは Q1 型基準への警告であって、原論文 Eq.(26) の評価ではない**（評価手続きは特定不能）。
- criterion 一致: S1 vs S2 **44/64**、S1 vs S3 2/64、S2 vs S3 0/64、三者一致 0/64。
- **start 不一致**（2 つの初期値が別の K を選ぶ）は S1/`K_TRUE=5` で `n=50` 8/8 → `n=150` 1/8。
  選択が不安定な領域と最適化が初期値依存の領域が一致するが、**主因は分離できていない**。
- **S4（Poisson-X Gram spectrum）は K を返さない。** 推定 Gram は全 64 セルで PSD 錐の外
  （最小固有値の中央値 −1.80〜−0.52）、閾値なし階数は常に `d=15`。
  真の `K` での固有値ギャップ比の中央値は `n` とともに増加（K=5: 1.79→2.30）するが、
  **事前に固定できる閾値がないため selected K を作らない**（U7）。

### 17.6 未解決（指摘 5 を含む）

| ID | 内容 |
|---|---|
| U2 | **Bernoulli-Y の一般識別可能性**（＝実験で実際に使っている family） |
| U5 | Bernoulli-Y / Poisson-Y の非入れ子性 |
| **U6** | **どの基準についても `n→∞` の一致性（指摘 5）。本フェーズでは証明していない** |
| U7 | 有限標本の rank 閾値。推定 Gram は PSD ですらない |
| U9 | clean construction で `K* = K_TRUE` が成り立つか（family ごと） |
| U10 | held-out plug-in score の population target |
| U11 | `M_K` の閉性、誤指定下の pseudo-true `K` |
| U12 | 候補集合 `{K : P0 ∈ M_K}` の連結性 |
