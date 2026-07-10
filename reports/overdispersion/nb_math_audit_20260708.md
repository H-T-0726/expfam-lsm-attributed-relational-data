# NB2-Y 数式・実装監査レポート

作成日: 2026-07-08（独立監査）
監査者: 研究コード監査エージェント（実装フェーズとは別タスクとして実施）

## 0. 実行条件

- branch: `research/overdispersion-z-ablation`
- git HEAD: `02311e7`（分岐元 main と同一。本ブランチはまだ未コミット）
- git status before: 変更 1（`EXPERIMENT_REGISTRY.md` 追記）+ 未追跡 63（実装フェーズの成果物）
- git status after: 上記 + 本監査の新規 2 ファイル
  （`expfam/src/experimental/test_nb_math_audit.py`, 本レポート）
- 実行コマンド: `python expfam/src/experimental/test_nb_math_audit.py`（8 テスト全 PASS）、
  `git status --short -uall`、Grep による呼び出し経路確認
- 読んだファイル: `model_dual_expfam_nb.py`, `model_dual_expfam_masked.py`,
  `em_runner.py`, `eval_utils.py`, `data_generator_overdispersed.py`,
  `tools/overdispersion/*.py`, `reports/overdispersion/*.md`,
  `expfam/results/overdispersion/*.csv`
- 変更したファイル: **既存ファイルの変更なし**。新規追加のみ
  （監査テスト 1 + 本レポート 1）
- missing 報告:
  - `expfam/src/experimental/negative_binomial_family.py` — **存在しない**。
    NB は独立 family モジュールではなく `DualExpFamLSMNB`（サブクラス）として
    実装されている。設計として問題なし（ファイル名の期待違いのみ）。
  - `expfam/results/overdispersion/nb_vs_poisson_summary.csv` — **存在しない**。
    同等の結果は `movielens_strict_heldout_summary.csv`（実データ）と
    `poisson_misspecification_summary.csv`（人工）に分かれて保存されている。

## 1. NB2 モデル定義（判定: 正しい）

実装（`eval_utils.nb_ll_pairs`, `DualExpFamLSMNB.calc_log_likelihood_Y`）:

```
ln p = lnΓ(y+r) − lnΓ(r) − lnΓ(y+1) + y(ln μ − ln(μ+r)) + r(ln r − ln(μ+r))
```

指定された監査基準式
`lnΓ(y+r) − lnΓ(r) − lnΓ(y+1) + r ln r + y ln μ − (y+r) ln(r+μ)`
と**代数的に同一**（y と r の ln(μ+r) 項をまとめただけ）。

- μ>0, r>0、E[y]=μ、Var[y]=μ+μ²/r（NB2）— docstring と一致
- **独立実装との数値一致**: `scipy.stats.nbinom.logpmf(y, r, p=r/(r+μ))` と
  r∈{0.5, 2, 10, 100} で atol=1e-8 の一致を確認
  （`test_nb_ll_matches_scipy` PASS）。パラメータ対応 p=r/(r+μ) は
  mean=r(1−p)/p=μ, var=μ(r+μ)/r=μ+μ²/r で整合。

## 2. 微分の確認（判定: 正しい）

μ = exp(η)（log link）のもとで再導出:

- score: ∂ℓ/∂η = y − μ(y+r)/(μ+r) = **r(y−μ)/(r+μ)** — 監査基準式と一致。
  実装 `_y_score_estep = (y−μ)·r/(μ+r)` と一致。
  **数値微分（中心差分）と rtol=1e-5 で一致**（`test_score_matches_numerical_gradient` PASS）。
  「y−μ だけ」になっていない ✓、μr/(μ+r) の位置逆転なし ✓。
- observed negative Hessian: −∂²ℓ/∂η² = **rμ(y+r)/(μ+r)²** — 数値 2 階微分と
  rtol=1e-4 で一致（`test_hessian_and_fisher` PASS）。
- Fisher 情報: E[y]=μ を代入して **I(η) = μr/(μ+r)** — 実装
  `_variance_function` はこの **Fisher（期待情報量）** を返す。
  observed と Fisher が y≠μ で異なることもテストで確認済み
  （混同していないことの sanity check）。

**設計上の注記（問題ではないが明記すべき点）**:
実装は E-step の precision に observed Hessian ではなく Fisher 情報を使う
（= Fisher scoring / expected-information Laplace）。設計レポート
`negative_binomial_design_20260708.md` の記述と**一致**しており、
半正定値性の保証・大 μ での安定化という妥当な根拠がある。ただし
Laplace 近似の厳密な定義（モードでの observed Hessian）とは異なるため、
事後分散は y が μ から離れたペアで近似が粗くなる。修論では
「expected-information Laplace（Fisher scoring 型）」と明記すること。
なお正準リンクの既存 family（Bernoulli/Poisson/Gaussian）では
observed = expected なので、既存実装との**整合的な一般化**になっている。

## 3. Poisson 退化（判定: 正しい）

- 理論: r→∞ で ll → Poisson ll、score → y−μ、Fisher → μ、Var → μ。
- 実装テスト: r=1e9 で `nb_ll_pairs` ≈ `poisson_ll_pairs`（atol=1e-4）、
  scipy 相互検証（nbinom ≈ poisson.logpmf）、score・Fisher の収束、
  生成器の理論 var/mean = 1+μ/r の単調性（r=2: 3.44 > r=5: 1.98 >
  r=20: 1.24 > ∞: 1.0）— すべて PASS
  （`test_poisson_limit`, `test_variance_monotone_in_r`）。
- 既存テスト `test_nb_large_r_matches_poisson` も本質的
  （variance/score/ll の 3 点を検査）。監査テストは scipy と数値微分という
  **実装から独立した参照**を加えた。

## 4. E-step / M-step との整合性（判定: 整合。軽微な注記 2 件）

- η_Y = w0 + w z_i^T z_j、μ = exp(clip(η, −20, 10)) — 既存 Poisson と同一経路
  （`_mean_function` を親から継承）✓
- E-step gradient Term3: `w · Z^T [score × mask]`、score は NB 版に
  オーバーライド済み ✓（1/2 なし = fixed 系列準拠 ✓）
- E-step precision Term3: `w² · Z^T diag[Fisher × mask] Z` ✓
- M-step w0/w: Adam の勾配和に `_y_residual_mstep`（= NB score）を使用 ✓。
  φ の扱い: NB は `self.family='poisson'` のため `_phi()=1` — NB2 に
  dispersion φ は η 経由で入らない（r は score/Fisher 内で処理）ので正しい ✓
- Gaussian 専用処理（sigma_y、residual/σ² 除算）の混入: なし。
  `calc_sigma_y` は `family_y=='gaussian'` のときのみ呼ばれ、
  score の Gaussian 分岐は NB のオーバーライドで到達不能 ✓
- Poisson 用コード流用での NB 補正漏れ: **なし**。
  懸念箇所だった `utils_expfam.calc_Q_dual_strict`（`model.family=='poisson'`
  を見て階乗補正するため NB に誤適用されうる）と `calc_bic_dual` は、
  **tools/ のどのスクリプトからも呼ばれていない**ことを grep で確認。
  すべて NB-aware な `calc_Q_dual_strict_exp` / `calc_bic_exp`
  （`family_y_label` で分岐）経由 ✓
- pair mask 併用: NB モデルで held-out ペアの Y を汚染しても
  勾配・calc_w0・calc_w・尤度が不変（`test_nb_mask_no_leak` PASS）✓

**注記 A（M-step の呼称）**: w0/w 更新は「Fisher scoring」ではなく
「score を用いた Adam（1 次法）」である。設計レポートは E-step を
Fisher scoring と呼んでおり矛盾はないが、修論で M-step まで
Fisher scoring と書かないこと。

**注記 B（初期化）**: `em_runner` の NB 初期化 w0 = ln(正カウント平均) は
Poisson と同一で、NB でも E[y]=μ なので妥当 ✓。

## 5. held-out 尤度の公平性（判定: 公平）

- NB 側: lnΓ(y+r) − lnΓ(r) − lnΓ(y+1) + r ln r + y ln μ − (y+r)ln(r+μ) —
  **全項あり**（§1 で scipy 一致確認済み）
- Poisson 側: y ln μ − μ − lnΓ(y+1) — **全項あり**（`poisson_ll_pairs`）
- 実験スクリプトは `heldout_count_metrics(..., family_label, nb_r=)` で
  各モデルの family に対応する full-constant 尤度を評価 ✓
- したがって te_ll の比較（人工: r=2 で NB +0.76 nats/pair 等）は**公平**。
- 留意: plug-in（点推定 μ̂）であり事後予測分布ではない — 両 family 同条件
  なので相対比較は公平だが、絶対値は楽観的（既レポートに明記済み ✓）。
- NB の te_ll は train 推定の r̂ を使う（1 個の追加自由度を train で決めて
  いる）— 正当な two-stage 手続きであり、test 情報は不使用（§6）。

## 6. r の扱い（判定: リークなし。軽微な注記 1 件）

- **学習中の r 更新: なし**（コンストラクタで固定、EM ループ内に更新なし）✓
- moment r̂ の計算タイミング: Poisson strict フィット（train のみで学習）の
  **train ペア残差**から計算
  （`run_movielens_strict_heldout.py` L.134: `upper_pairs_of(train_mask)`、
  `run_poisson_misspecification_check.py` L.158 同様）。
  μ̂ 自体も masked 学習由来なので **test 情報は二重に不使用** ✓
- リークの単体テスト: test ペアの Y を 9999 に書き換えても r̂ が不変
  （`test_r_hat_train_only` PASS）✓
- oracle r: 人工データ実験（`nb_oracle` 条件）のみで使用、実データでは不使用 ✓
  （レポートにも「人工限定」と明記済み）
- 診断スクリプト（`diagnose_movielens_overdispersion.py`）の r̂ は
  full-mask フィットの全ペアから計算しているが、これは診断目的であり
  held-out 評価に使われていない — リークではない ✓
- MovieLens で r̂ が大（k=3 で ≈180）または ∞ 相当（k=5）になることの解釈:
  既レポートは「条件付き過分散が小さい」+「r̂ は k に依存」+「in-sample の
  k=5 は過適合の兆候」と保守的に記述しており**安全** ✓
- `min(r_hat, 1e6)` のキャップは `moment_estimate_nb_r` 内部の r_max=1e6 と
  重複（無害な冗長）。

**注記 C（BIC パラメータ数）**: `calc_bic_exp` はデフォルト
`nb_r_estimated=True` で r を +1 パラメータと数える。moment 推定の
`nb_moment`/実データ NB では正しい。ただし人工実験の **`nb_oracle`（r 既知）
でも +1 で数えられており、厳密には 1 個の過大カウント**。影響: BIC は
本フェーズの主張根拠に使われておらず（held-out 尤度が主指標）、数値上も
ln(100)≈4.6 の差で結論に影響しないが、oracle 条件の BIC を引用する場合は
注記が必要。`nb_r_estimated=False` の経路が正しく −1 になることは
`test_bic_r_param_count` で確認済み。

## 7. テスト結果

- 既存テスト（`test_experimental_models.py` 7 件）: 全 PASS（本質的:
  等価性・リーク・退化・r 回復・EM smoke をカバー）
- **追加した監査テスト（`test_nb_math_audit.py` 8 件、全 PASS）**:
  1. scipy.stats.nbinom との対数尤度一致（独立実装照合）
  2. score = 数値微分（中心差分）
  3. observed Hessian = 数値 2 階微分、Fisher = E[y]=μ 代入、
     実装が Fisher を採用していることの確認
  4. r→∞ の Poisson 退化（scipy 相互検証込み）
  5. 生成分散の r 単調性
  6. r̂ の train 限定性（test 汚染で不変）
  7. BIC の r パラメータ数（estimated/fixed の切替）
  8. NB + pair mask のリーク無し

## 8. 修論での扱い

### 強く言ってよいこと
- NB2 の対数尤度・score・Fisher 情報の実装は、独立実装（scipy）・
  数値微分・解析式の三重照合で正しいことを確認した
- r→∞ で Poisson に厳密に退化する（実装・理論とも）
- Poisson との held-out 尤度比較は全正規化定数込みで公平
- r̂ の推定に test 情報は使われていない（two-stage、テストで保証）
- pair mask との併用でリークがない（テストで保証）

### 弱めに言うべきこと
- E-step の precision は observed Hessian ではなく **Fisher（期待）情報量**
  を用いた expected-information Laplace である（安定性優先の設計選択。
  y が μ から遠いペアでは事後分散の近似が粗い）
- r は固定（moment two-stage）であり、プロファイル尤度推定・r の
  不確実性は未対応
- BIC 上の r の数え方は moment 推定前提（oracle 条件では 1 過大カウント）。
  BIC 自体を NB vs Poisson の主要根拠にしない

### 言ってはいけないこと
- 「NB は本モデルの正準指数型分布族拡張である」— **不可**。
  r 固定の NB2 は η'=ln(μ/(μ+r)) についてなら正準 ExpFam だが、
  本実装は η=ln μ（**非正準リンク**）であり、原稿の枠組み
  （score = T(y)−A'(η)、precision の A''(η)）そのままの instance ではない。
  正しい呼称: 「**experimental な NB2 拡張（log link + Fisher scoring）**。
  正準 ExpFam 枠組みの『一般リンクへの拡張可能性』を示す実例」
- 「NB が MovieLens で予測を改善した」（差は小さい。改善は人工過分散
  データでの話）
- 「observed Hessian を用いた厳密な Laplace 近似」（Fisher 版である）

## 9. 結論

1. **NB 実装は数式的に OK**（三重照合で確認、要設計修正なし）
2. **experimental として扱うべき**（非正準リンク + Fisher scoring という
   枠組み拡張を含むため、原稿の正準 ExpFam の系とは区別する）
3. **追加修正: 不要**（挙動を変えるバグなし。注記 A〜C は記述・引用上の
   注意であり、コード修正は不要と判断。calc_bic_exp の oracle 呼び分けは
   将来 BIC を主張に使う場合のみ対応）
4. **再実験: 不要**（既存結果の数値は本監査の範囲で妥当。
   将来課題として observed-Hessian 版 Laplace との比較、r プロファイル推定、
   posterior coverage 検証を推奨）
