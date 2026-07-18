# Dual-ExpFam LSM 完全生成モデル仕様・用語正本（2026-07-19）

作成: 理論監査フェーズ（P0-5、承認済み）。
根拠: root `CLAUDE.md` の確定式、`reports/theory_audit/theory_audit_report_20260718.md`
§2-3・§12、および実装（`expfam/src/model_dual_expfam_fixed.py`、
`expfam/src/experimental/model_dual_expfam_masked.py`、`model_dual_expfam_percolumn.py`、
`expfam/src/data_generator_expfam.py`）との照合。
本文書は**モデルの定義（何を仮定しているか）**の正本であり、
**推定アルゴリズム（Laplace-MCEM）の正しさとは別問題**である（§6）。

---

## 1. 観測集合 O を固定した augmented observed-data model

観測: 属性行列 X = (x_il) ∈ 各列の台の直積（i=1..n, l=1..d）、
関係行列 Y = (y_ij)（対称、対角なし）。
観測ペア集合 O ⊆ {(i,j) : i<j}（完全観測では O = 全 i<j）。
潜在: Z = (z_1,…,z_n)^T ∈ R^{n×k}。列 family 割当 c(l) ∈ {gaussian, bernoulli, poisson}
（全列共通のスカラー family はその特殊形）。

パラメータ:
θ = { F ∈ R^{d×k}, w_0^Y ∈ R, w^Y ∈ R }
    ∪ { σ_l² : c(l)=gaussian } ∪ { σ_y² : family_y=gaussian }

```
p(X, Y_O, Z | θ, M, O)
  = ∏_{i=1}^{n} N(z_i ; 0, I_k)
  · ∏_{i=1}^{n} ∏_{l=1}^{d} p_{c(l)}( x_il | η_il^X, φ_l ),   η_il^X = f_l^T z_i
  · ∏_{(i,j)∈O} p_Y( y_ij | η_ij^Y, φ_Y ),                    η_ij^Y = w_0^Y + w^Y z_i^T z_j
```

- **X 側切片なし**: η_il^X = f_l^T z_i のみ（先行研究 eq(2) と同じ規約）。
  切片導入（η = μ_l + f_l^T z_i）は承認待ちの将来課題（P0-3）。
- 条件付き独立: Z を与えたとき全 x_il・全 y_ij は互いに独立、X ⊥ Y | Z。
- 対角 y_ii はモデルに含まれない。
- (i,j) ∉ O のペアは尤度に寄与しない（ignorability は §4 の条件を前提）。
- M は family 割当・k・O の指定を含むモデル指標。

これは、観測集合 O を所与とした「観測部分 Y_O と潜在変数 Z の同時分布」であり、
全 Y と観測指標 R を含む完全な missing-data 生成モデルではない。欠測機構まで
生成的に定義するには、全関係行列 Y、観測指標 R、および例えば
`p(R | X, Y, Z, psi)` を追加する必要がある。現行 masked 実装は欠測機構を
新たにモデル化せず、O を所与として観測ペアだけを尤度へ含める。

## 2. family 別仕様表

1 変量指数型分布族 p(x | η, φ) = h(x, φ) exp{ (η T(x) − A(η)) / φ }。

| 項目 | Bernoulli | Poisson | Gaussian（分散付き） |
|---|---|---|---|
| 標本空間 | {0, 1} | {0, 1, 2, …} | R |
| 支配測度 | 計数測度 | 計数測度 | Lebesgue 測度 |
| h(x, φ) | 1 | 1/x! | (2πφ)^{-1/2} exp(−x²/2φ) |
| T(x) | x | x | x |
| 自然パラメータ | η = logit(p) | η = log λ | η = μ |
| 自然パラメータ空間 | R | R | R |
| A(η) | log(1+e^η) | e^η | η²/2 |
| A′(η) = E[T] | σ(η) | e^η | η |
| A″(η) | σ(η)(1−σ(η)) | e^η | 1 |
| 分散・尺度 | A″(η), φ=1 | A″(η), φ=1 | φ = σ²（X: 列ごと σ_l² / Y: σ_y²、M-step で MLE） |
| 現在のリンク | canonical (logit) | canonical (log) | canonical = identity（mean link と同一） |
| 現在の切片 | なし | なし | なし |
| 数値ガード | 尤度で η clip ±500、A″ 下限 1e-8 | **η clip [−20, 10]**（§5）、A″ 下限 1e-8 | σ² 下限 1e-6〜1e-8 |
| 欠測時の尤度寄与 | 0 | 0 | 0 |

**family 間の尤度比較に関する注意** [DERIVED]: モデル比較は、同じ観測データ・
観測空間・整合する支配測度・データ変換の下で定義する必要がある。family 名が
異なることだけで常に比較禁止になるわけではない。例えば Poisson と NB は同じ
非負整数データ上で全正規化定数を含む尤度を定義すれば比較できる。固定した family
割当の下で同じデータについて k を比較する問題も、family 自体の比較とは別である。
一方、現在の誤指定実験のように条件ごとにデータ変換、台、観測値の解釈が変わる場合、
生の尤度・Qベース基準を直接比較してはならない。台外データを Bernoulli/Poissonへ
入力する条件は正しい確率尤度ではなく quasi-loss として扱う。Gaussian・Poisson・
Bernoulli の値を無条件に同列比較できない理由を、代表的支配測度の違いだけに
帰着させない。

## 3. 用語の定義（混同禁止）

| 用語 | 定義 | 実装上の対応 |
|---|---|---|
| 観測された 0 | ペア (i,j) ∈ O で y_ij = 0 が観測された | 尤度に y=0 として寄与 |
| 未観測ペア | (i,j) ∉ O。値は不明 | 尤度に寄与しない。`train_mask=False` は**これ**であり「観測された 0」ではない |
| 欠測 | 観測が意図されたが得られなかった値 | 現行では O から除外して扱う（機構は §4） |
| 完全観測の疎ネットワーク | O = 全ペアで 0 が多い | fixed 系列・Cora 実験がこれ |
| 部分観測ネットワーク | O ⊊ 全ペア、O は既知 | masked 系列（`DualExpFamLSMMasked`） |
| negative sampling | 負例（0）を部分抽出して学習 | 未実装（評価側の neg_ratio は評価専用） |
| positive-unlabeled (PU) | 1 のみ観測され、0 と未観測が区別不能 | 未対応。masked 尤度では正当化できない |
| MCAR | 欠測確率がデータと無関係 | ランダム pair split は構成的に MCAR |
| MAR | 欠測確率が観測値のみに依存 | MAR + distinctness なら O 上の尤度が正当（ignorability） |
| MNAR | 欠測確率が未観測値に依存 | 欠測機構のモデル化が必要。未対応 |
| transductive link prediction | 学習に現れたノード集合内の未観測ペアを予測 | 現行の held-out 評価はすべてこれ |
| cold-start prediction | 学習に現れないノードのペアを予測 | 未評価・未対応 |

## 4. 欠測の ignorability 条件

観測ペア尤度 ∏_{(i,j)∈O} p_Y(y_ij | ·) による推定が正当なのは、
欠測機構が MCAR または MAR で、かつ欠測機構のパラメータが θ と分離
（distinctness）している場合である。実験でのランダム pair split は
構成的に MCAR を満たす [DERIVED]。実データで O が「観測努力」等に依存する場合は
MAR/MNAR の検討が別途必要であり、現行実装は欠測機構をモデル化していない。

## 5. Poisson clip の明文化（P0-2）

実装（`model_expfam.py` `_mean_function`/`_variance_function`、
`model_dual_expfam.py` X 側、per-column 版、尤度関数群）は Poisson の
自然パラメータを η_c = clip(η, −20, 10) として
A′(η_c)、A″(η_c)、尤度 x·η_c − e^{η_c} を計算する。

- clip 域の外（−20 ≤ η ≤ 10）では clip は恒等写像であり、実装の勾配・曲率・尤度は
  指数型分布族の厳密な微分と一致する（per-column 数式監査 31/31 PASS はこの領域）
  [CONFIRMED_IN_REPOSITORY]。
- clip 域では ∂η_c/∂η = 0 なので、「clip 後尤度」の η 微分は 0 になる。
  一方、実装の E-step 勾配は clip 後の残差 x − A′(η_c) をそのまま返すため、
  **clip 域では実装尤度と実装勾配は互いに整合しない** [DERIVED]。
  これは発散防止のための意図的な数値ガードであり、clip が発動しない条件では
  問題は表面化しない。
- **発動率はこれまで未計測**であった。2026-07-19 より
  `expfam/src/experimental/diagnostics.py` の `poisson_clip_diagnostics` により
  最終推定値での clip 域該当率を post-hoc 計測でき、
  `run_em_experimental(compute_clip_diagnostic=True)` の結果 dict
  （`clip_diag` キー）に記録される。既定 False では診断計算を行わず None を返す。
  これは診断であり、clip 範囲・勾配・モデル挙動は一切変更していない。
  「clip があるから既存実験結果が誤っている」とは言えない（発動していなければ
  厳密一致するため）。既存実験の発動率は未計測である点のみ明示する。

## 6. 生成モデルと推定アルゴリズムの区別

本文書が定義するのは確率モデル（同時分布）のみである。
同時分布が正規化されていることは、Laplace-MCEM 実装
（per-node 逐次 Newton + Laplace サンプリング + scale_Z + Adam M-step）の
統計的正しさを保証しない。次は**別々に**評価する:
モデルの well-definedness（本文書）／ E-step 局所導出の正しさ
（fixed 系列で数値監査済み）／ 近似事後・サンプル列の分布的性質（未確立、
`reports/theory_audit/diagnostic_designs_20260719.md` §2）／ M-step の Q 増加
（`mstep_q_diagnostic` で診断可能、保証は未証明）／ EM 全体の収束（未証明）／
scale_Z 等のヒューリスティックによる目的関数の変更（同 §3）。
なお、現行のモデル選択基準 −2Q̂ + p̂ ln n は **Schwarz BIC ではなく**
Qベース完全データ型基準（ICL-type）である（同 §1 訂正4）。

## 7. 人工データ生成器の注意（`data_generator_expfam.generate_dual_data`）

- **Z の z-score**（L.281-283）: z_i ~ N(0,I) 生成後に列ごとに z-score する。
  返却される「真の Z」は prior の厳密なサンプルではなく、標本平均 0・
  標本分散 1 に条件付けられたもの（モデルとの差は O(n^{-1/2})）。
- **F の行正規化**（L.285-290）: 行ノルムを √(1−uniq)（既定 ≈0.949）に固定。
- **Gaussian-X の生成後 z-score**（L.292-298）: X = ZF^T + ε の後に列 z-score。
  設計上 Var(x_l) ≈ ‖f_l‖² + uniq = 1 なので z-score はほぼ恒等変換だが、
  厳密には**返却される F・Σ は z-score 後の X の生成パラメータと
  O(n^{-1/2}) のオーダーでずれる**。RMSE(F) 等の解釈時に留意する。
  Bernoulli/Poisson-X と Y 側は正規化なしで生成過程とモデルが一致する。
- **quasi-loss になる条件** [DERIVED]: 誤指定実験で family の台の外のデータを
  与える条件（例: 生カウントへの Bernoulli スコア x − σ(η)、z-score 連続値への
  Poisson リンク、{0,1} データへの Gaussian 密度は台としては valid だが
  支配測度が異なる）では、目的関数は数式として well-defined でも
  **正しい確率モデルの対数尤度ではない**（quasi-likelihood 的な作業損失）。
  これらの条件の「尤度」「Qベース基準」の値を確率モデルとして解釈・比較しない。
  台の整合は `diagnostics.validate_family_support`（opt-in、
  `allow_support_mismatch=True` で明示許可）で検査できる。
