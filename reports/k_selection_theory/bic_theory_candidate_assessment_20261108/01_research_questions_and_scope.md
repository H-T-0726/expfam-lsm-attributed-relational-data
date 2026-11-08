# 01 研究の問い・作業範囲・一次資料・記法（2026-11-08）

- **モード**：THEORY / AUDIT。新しい MCEM fit、新しい K 選択の評価、成功率の実験はしていない。
- **系列**：理論の対象は採用式（Y 側に余分な 1/2 がない）のモデルである。過去の数値を引用するときは、系列を明記する（系列 E は prototype、OLD は 0.5 系列）。
- **呼称**：
  - C_Q は Q 関数に基づく完全データ型の基準（ICL 型）で、Schwarz BIC ではない（KI-010）。
  - C_Lap は Laplace 近似による基準で、厳密な周辺尤度ではない。
  - 原論文の Eq.(26) は、論文中で「BIC」と呼ばれている基準として扱う。
- **ブランチ**：`research/bic-theory-candidate-assessment-20261108`。PR #162 の先端（`8efe98e`）から分岐した。

## 1. 研究の問い

| # | 問い |
|---|---|
| RQ1 | 原論文 Eq.(26) の BIC 型基準は、数学的に何を計算しているか。Schwarz BIC の導出条件を満たしているか |
| RQ2 | 現在の C_Q は何を計算しているか。周辺尤度とどれだけ違うか |
| RQ3 | 弱信号・小さい n で C_Q が過小選択する現象は、式からどこまで説明できるか |
| RQ4 | Q_Z を除いた基準が過大選択する現象は、式からどこまで説明できるか |
| RQ5 | 潜在変数の不確実性（事後分布の広がり）を、選択基準にどう取り入れるべきか |
| RQ6 | 指数型分布族への拡張に固有の論点は何か |
| RQ7 | 既存の手法（ELBO、Laplace、BIC、ICL、sBIC、WBIC、JIC など）と比べて、研究上の貢献の余地はあるか |
| RQ8 | 数学的根拠と実装可能性の両面から、次に検証する候補は何か |

**前提にしないこと**
- 新しい BIC が必要であること。
- 提案すれば精度が上がること。
- 原論文の基準や C_Q が誤りであること。

## 2. 一次資料（証拠の優先順位は AGENTS.md §4）

| 種別 | 資料 | 本研究での使い方 |
|---|---|---|
| 原論文 | `paper/A_study_on_latent_structural_models_for_binary_rel.pdf`（Mikawa et al., NOLTA 15(2), 2024） | Eq.(1)(2)(3)(4)(16)〜(26)、§4.1〜4.4 を本文から直接確認した。テキストは pypdf で抽出した |
| 実コード | `reproduction/src/model.py`（`calc_eta_newton` L.360-466、`scale_Z` L.468-505） | E-step の標本の作り方と `scale_Z` |
| 実コード | `expfam/src/experimental/em_runner.py`（E-step L.247-275、M-step L.293-301、Q_strict L.330-341） | MCEM の手順と、Q を評価する時点 |
| 実コード | `expfam/src/experimental/eval_utils.py`（`calc_Q_dual_strict_exp` L.186-229、`calc_bic_exp` L.232-） | Q_strict と罰則 |
| 実コード | `expfam/src/experimental/k_criterion_panel.py`（`cq_terms` L.85-105、`laplace_terms` L.108-141、`cond_terms` L.144-） | C_Q・C_Lap・条件付き基準 |
| 実コード | `expfam/src/experimental/laplace_k_criterion.py`（`gradient` L.143、`negative_hessian` L.154、`find_joint_mode` L.199、`laplace_log_observed` L.243） | C_Lap |
| 実コード | `reproduction/src/experiment_paper_2.py`（`calc_log_likelihood` L.123-170） | Python 再現の BIC |
| canonical | `RESEARCH_MASTER.md` §12.6・§17・§19〜§22、`KNOWN_ISSUES.md` KI-001・KI-010 | 既存の確定事項と、主張してよい範囲 |
| 過去の研究（PR #155〜#162） | `reports/k_selection_theory/qz_criterion_existing_states_20261106/`、`fair_model_estimator_criterion_comparison_20261106/`、`n75_old_vs_expfam_controlled_20261106/`、`weak_n75_old_vs_expfam_allpairs_20261107.md`、`historical_code_lineage_audit_20261106/` | 観測結果。いずれも Draft PR 上にあり、main には未採用 |
| 本研究の確認計算 | `tools/research_audit/bic_theory_identity_checks_20261108.py` → `identity_checks.csv`（恒等式 39/39 PASS と、参考値 7 行） | 恒等式の数値確認（Phase J） |
| 文献 | 05 に一覧。本文または abstract を一次で確認したものと、書誌だけのものを分けて示す | 既存手法との比較 |

**注意**：PR #155〜#162 はすべて Draft で、main に採用されていない（AGENTS.md §3）。本研究では、それらの数値を「過去に観測されたこと」として引用する。採用済みの結果としては扱わない。

## 3. 記法

| 記号 | 意味 |
|---|---|
| $n$ | ノード数（対象の数） |
| $d$ | 属性の次元（X の列数） |
| $K$ | 候補の潜在次元 |
| $Z=(z_1,\dots,z_n)^\top\in\mathbb R^{n\times K}$ | 潜在変数 |
| $\theta_K$ | パラメータ。$\{F, w_0, w\}$。Gaussian-X なら対角の $\Sigma_X$、Gaussian-Y なら $\sigma_Y^2$ を加える |
| $\Phi(Z)=\log p(X,Y,Z\mid\theta)$ | 完全データ対数密度。コードの `complete_log_density` |
| $\ell_{\rm cond}(Z,\theta)=\log p(X,Y\mid Z,\theta)$ | 条件付き対数尤度（PR #159 の Q1） |
| $\ell_{\rm comp}(Z,\theta)=\Phi(Z)$ | 完全データ対数尤度（Q2） |
| $\ell_{\rm marg}(\theta)=\log\int e^{\Phi(Z)}dZ$ | 観測データの周辺対数尤度（Q3） |
| $\pi(Z)=p(Z\mid X,Y,\theta)$ | 厳密な事後分布 |
| $q(Z)$ | 正規化された近似分布 |
| $H(q)=-\mathbb E_q\log q$ | 微分エントロピー |
| $\mathrm{KL}(q\|p)=\mathbb E_q\log(q/p)$ | KL ダイバージェンス |
| $p_K$ | K に依存するパラメータ数。$p_K = Kd-K(K-1)/2$ に K によらない定数 c を足したもの |
| $\Lambda$ | 事後分布をガウスで近似したときの、Z 全体（$nK$ 次元）の精度行列 |
| $a_i$ | ノード i の、ある潜在方向の事後精度（1 次元に射影したもの） |

### 次元の 4 つの記号（Phase A2）

| 記号 | 定義 | 注意 |
|---|---|---|
| K_GEN | 人工データを生成したときの Z の列数 | 手続き上の数 |
| K_CAND | 推定で試す候補の次元 | |
| K_HAT | 基準が選んだ次元 | |
| K0 | 母集団の分布を表せる最小の潜在次元。$K_0=\min\{K: P_0\in\mathcal M_K\}$（`RESEARCH_MASTER.md` §17.2 の K\*） | M-closed（$P_0\in\bigcup_K\mathcal M_K$）のときだけ定義される。**K_GEN と同一視しない** |

## 4. 作業範囲

**行ったこと**
- 原論文・実コード・過去の一次 CSV/レポートの読み取り
- 数式の独立導出
- 文献の確認
- 小規模・決定論的な確認（恒等式 39 項目と参考値 7 項目）
- Markdown 文書の作成
- `EXPERIMENT_REGISTRY.md` への 1 行の追記（確認スクリプトの provenance）
- 独立レビュー

**行っていないこと**
- 新しい MCEM fit、乱数による成功率の評価、K 選択の新しいベンチマーク
- 既存の推定状態に新しい基準を適用して K_HAT を計算すること（Human Gate として残す。06・10）
- モデル・E-step・M-step・`scale_Z`・既存の選択関数の変更
- 既存 CSV・凍結レポートの変更
- 候補基準の正式な採用、prototype の昇格、修士論文本文の変更

## 5. 目標の 3 つの読み方（Phase A2 の後半）

| 目標 | 望ましい性質 | 適した基準の型 | 注意 |
|---|---|---|---|
| (G1) 人工データの K_GEN を回復する | 生成器が K_GEN を K0 にする条件の下で、一致性と有限標本での正解率 | K0 を推定する基準 | K_GEN = K0 は自明ではない（§17.2）。dsweep・R1/R2 の Poisson-X では作業上の前提として K0=5 を置いた |
| (G2) 母集団の最小次元 K0 を選ぶ | M-closed での選択の一致性 | 周辺尤度（モデル証拠）型。Schwarz BIC・sBIC・WBIC | 実データでは M-closed が一般に成り立たず、K0 は存在しないことがある |
| (G3) 予測に有効な次元を選ぶ | 期待予測損失の最小化（M-open でも意味がある） | held-out 予測、AIC・WAIC 型 | 一般に一致性はない。K0 より大きい K を選びうる。G2 とは別の目標 |

- 以下の理論は主に **G2**（そして、K_GEN = K0 が成り立つ設定での G1）を対象にする。
- 原論文 §4.3 の目的も「true dimension の回復」であり、G1/G2 の側である。
- G3 は比較対象として 05・06 で扱う。
