# 実装監査：真値の漏洩、初期化、基準、系列

- 作成日：2026-11-25。基点 `origin/main` = `1b5030b`。コードは変更していない。
- 数値の確認は `tools/research_audit/check_true_params_leak_fixed.py` で行った（小さな設定でのコード経路テスト。実験ではない）。結果は `expfam/results/k_selection/k_exp2_generalization_20261125/true_params_leak_check/` にある。

## 1. fixed official Exp1 の `true_params`（最重要監査）

### 1.1 コードの経路（FACT）

- runner：`expfam/src/run_fixed_official_exp1_bic_full.py` L.87、`run_fixed_official_exp1_bic_k9_extension.py` L.101 は `model.initialize_params(true_params=true_params, seed=seed + retry*1000)` を呼ぶ。
- `DualExpFamLSMFixed` は `initialize_params` を上書きせず、`reproduction/src/model.py` L.63–130 を継承する。そこで `true_params` から読むのは **`var_f` だけ**（`params['var_f'] = 2·var_f`、L.101–104）。
- `params['var_f']` は E-step・M-step・Q のどこでも使われない（`grep` で確認。使われるのは初期化と `summary()` の表示だけ）。

| 確認項目 | 結果 |
|---|---|
| 真の Z を初期化に使うか | **使わない**。Z は `rng.standard_normal((n, k))` |
| 真の F を使うか | **使わない**。F は `rng.standard_normal((d, k))`（非 Gaussian-X では ×0.2） |
| 真の w0・w を使うか | **使わない**。初期化の w0・w は、その後 runner が**観測データから**上書きする（Bernoulli-Y：w0 = logit(Y の密度)、w = 0.5。Poisson-Y：w0 = log(正の count の平均)、w = 0.1/2^retry。Gaussian-Y：w0 = Y の平均、w = 0.5、σ_y = Y の SD） |
| 真の分散を推定で使うか | **使わない**。`var_f` は読むが未使用。Σ の初期値は I |
| 真値を使うのは評価だけか | **はい**。fit の後に Procrustes 回転した RMSE(Z) と η_Y の RMSE を、真の Z・w0・w で計算する（L.168–177） |
| 候補 K の間でデータが同じか | **同じ**。`seed_data = base + 100·trial` は K によらない。初期化の seed（`seed_data + 50`）も K によらない（次元だけが違う） |

### 1.2 数値による確認（4 つの family の組み合わせ）

次の 4 通りの `true_params` で fit した。

| 渡したもの | 内容 |
|---|---|
| (a) | 生成した辞書そのもの |
| (b) | (a) から `var_f` を除いたもの |
| (c) | `var_f` = 10⁶ |
| (d) | Z・F・w0・w を壊したもの（Z を −7 倍、F = 0、w0 = 99、w = −99） |

**4 通りすべてで、fit 後の Q_strict がビット単位で一致した**（P/B、G/P、B/G、G/B の 4 組）。

### 1.3 過去の K 選択への影響（JUDGMENT）

- **真値は推定に漏れていない。** fixed official Exp1（30/30）を「真値の情報で K=3 を選んだ」と解釈する根拠はない。
- 過去の結果の解釈に影響する別の点は次の 3 つである（いずれも既知か、今回の確認で分かったこと）。
  1. **NaN が出ると seed と Newton の歩幅を変えて最大 2 回やり直す**（seed rescue。L.93–95）。何回やり直したかは CSV に保存されていない（`success` は 270/270 で True。`nan_occurred` は返り値にあるが CSV の列にない）。**やり直しが選択に影響したかは確認できない。**
  2. 反復の途中で Z に NaN が出ると、前の Z で置き換えて続ける（L.141–144、黙った修復）。
  3. **「10 試行」は、データと初期値を同時に変えている**（試行ごとに別データ）。したがって 30/30 は「3 シナリオ × 10 データ × 1 初期値」であり、元論文の「同じデータで 10 個の初期値」とは試行の意味が違う。

## 2. lineage E の runner（`expfam/src/experimental/em_runner.py`）

- `true_params` を受け取らない（引数にない）。
- `failure_policy='fail_fast'` で、やり直しなしで失敗として記録できる。`init_seed` と `mc_seed` を分けられる。
- 初期化は fixed と同じ informed init（w0 = データの logit 密度など）。

## 3. 基準の実装と元論文 Eq.(26) の対応

| | 元論文 Eq.(26) | C_Q（`calc_bic_dual`、`utils_expfam.py` L.386–404） |
|---|---|---|
| 尤度の項 | −2 ln L、L は Z を条件とした尤度（Eq.(16)。ln p(Z) を含まない。Z の評価点は不明） | −2 Q_strict。Q_strict は E-step の MC サンプルについての ln p(Z) + ln p(X\|Z) + ln p(Y\|Z) の平均（`calc_Q_dual_strict`、L.355–378。Poisson の階乗項を含む） |
| Z の扱い | 不明（代入） | MC 平均（ln p(Z) を含む完全データ型） |
| ln p(Z) | 含まない | 含む。scale_Z と var_z = 1 のもとで **−nK(1+ln 2π)/2 という K だけで決まる値**になり、実質的に K あたり n(1+ln 2π) の罰則として働く（n=150 で 425.68/K。`q_based_criterion_decomposition_audit_20260907.md`、§19 の P_Z） |
| パラメータ数 | (K+1)d − K(K−1)/2 | Kd − K(K−1)/2 + d·[Gaussian-X] + 1·[Gaussian-Y] |
| Gaussian-X・Bernoulli-Y のとき | — | **Eq.(26) と同じ数**（Series O） |
| μ_x | 推定するが数えない | モデルにない |
| ln の中 | n | n（ノード数） |
| 呼称 | 論文は BIC と呼ぶ | **Schwarz BIC ではない**（ICL 型） |

**C_Q は Eq.(26) の実装ではない。** 罰則の数は G/B で一致するが、尤度の項が違う（Z を MC で平均し、ln p(Z) を含む）。

## 4. 副次基準の適用範囲

| 基準 | 実装 | 系列 | 使える系列 |
|---|---|---|---|
| C_Lap | `expfam/src/experimental/laplace_k_criterion.py` | **lineage E だけ**（`DualExpFamLSMConsistent` 以外は `UnsupportedLaplacePath`） | Y = Bernoulli・Poisson だけ。**Series C（Gaussian-Y）には使えない** |
| held-out 予測スコア | Phase 7e/8b の凍結 selector | pair mask が要る（masked・E 系列） | 学習データが 80% になるので、**全ペアで学習した C_Q の fit とは別の fit が要る** |

## 5. 系列の選択（Human Gate）

| | fixed（C） | lineage E-scalar |
|---|---|---|
| 採用状況 | 正式モデルの主系列として推奨（`docs/thesis-adoption-20261124` の資料。決定は Human） | prototype。MANUSCRIPT_ADOPTION_STATUS: NOT_DECIDED_BY_HUMAN |
| 数値 | η の clip と曲率の下限（legacy numerics） | clip なし（canonical） |
| 失敗処理 | 既存 runner は seed rescue あり。新しい runner が要る | fail_fast がある |
| C_Lap | 使えない | 使える（Gaussian-Y を除く） |
| 既存の K 選択研究との比較 | K_TRUE=3 の Exp1 だけ | Phase 7e〜9、clean n-sweep、Fig.3 条件、d-sweep などすべて |

**どちらで主実験を行うかは、本文で使う実装系列の決定そのものであり、Human Gate である。** 両系列の結果を同じ表に混ぜない（KI-002）。
