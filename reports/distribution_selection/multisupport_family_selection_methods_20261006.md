# Multi-support family selection — 方法・判断記録（2026-10-06）

- 位置づけ: **CHARACTERIZATION**（controlled synthetic family-selection experiment）。系列 **E（experimental prototype; 本文採用不可）**。
- branch: `exp/multisupport-family-selection`（base `origin/main = ee4be23`）。作業開始時の HEAD は別 branch `docs/notion-k-selection-story-20260906 = dd7ca14`、working tree は untracked 2 件（`reports/identifiability/gaussian_y_k_identifiability_adversarial_audit_20260907.md`, `reports/research_scope/`）のみで、これらには触れていない。
- run: `expfam/results/family_selection/expanded_multisupport_20261006_034826/`（init `37a17ac`、protocol sha256 `405fc22f…`）
- 数理監査: `reports/distribution_selection/family_candidate_compatibility_audit_20261006.md`
- 自動生成の表（FACT のみ）: `reports/distribution_selection/seminar_family_selection_summary_20261006.md`
- 結果と解釈: `reports/distribution_selection/multisupport_family_selection_report_20261006.md`

本文書は「何をどう決めたか」を記録する。結果の数値は含めない。

---

## 1. 研究質問と固定したもの

同一 support・同一 base measure 上に複数の妥当な候補分布があるとき、連続潜在 Z を考慮した candidate score で
真の属性分布族を回収できるか。

| 項目 | 固定値 |
|---|---|
| モデル | `z~N(0,I_3)`, `η=f_l^T z_i`（intercept なし）, X は canonical ExpFam（Gaussian のみ既存 dispersion 形）, Y Bernoulli(`w0+w z_i^T z_j`) |
| K | K_true = K_fit = 3（K 探索なし） |
| Y 生成 | w0 = −1, w = 1 |
| MCEM | L = 5, exploration 8 iter, refit 8 iter, Newton α = 0.5（Phase 9C と同じ） |
| 選択方式 | Phase 9C hybrid（exploration 内 A-type 更新 → 選ばれた assignment で fresh refit、refit を報告） |
| penalty | なし（同群の候補は自由パラメータ数が等しい。raw complete score と margin を記録） |
| nuisance | Binomial N=2, CMP ν∈{0.5, 2.0}, LN σ=1, PTN σ=1, logit-normal σ=1, TN01 σ=0.25・中心 0.5（すべて推定しない） |
| priority | COUNT: poisson → binomial_n2 → cmp_nu2p0 → cmp_nu0p5 → ztp / POSITIVE: lognormal_s1 → ptn_s1 / BOUNDED: logitnormal_s1 → cb → tn01_s0p25_c0p5 |
| baseline | ORACLE（真の candidate model 固定）/ AUTO（START_FIRST, START_LAST）/ SUPPORT_ONLY（各列の先頭 admissible 候補） |
| seed | dataset ごとに式で決定（protocol.json `seed_rule`）。ORACLE・SUPPORT_ONLY・AUTO refit は同じ refit seed、AUTO の 2 start は同じ search seed |

## 2. 生成器の設計判断（selector 結果を見る前）

**information-matched loadings.** family ごとに natural parameter のスケールが大きく違う
（A''(0) は CB で 1/12、CMP ν=0.5 で約 2.17）。全列同じ f_scale にすると一部 family に潜在信号がほぼ入らないので、
F の行 l を `1/sqrt(A''_{m_l}(0))` 倍し、η=0 での Fisher 情報が列平均で τ になるようにした。

**難易度 τ.** 事前宣言した格子 τ ∈ {0.25, 0.5, 1, 2} について、selector と独立な
期待 pointwise-projection KL（真の候補から最も近い同群 competitor への KL の下界）を計算し、
**EASY = argmax_τ D_g(τ), HARD = argmin_τ D_g(τ)** を support group ごとに機械的に決めた（`tables/difficulty_calibration.csv`）。
結果は COUNT・POSITIVE が EASY τ=2 / HARD τ=0.25、BOUNDED は逆に EASY τ=0.25 / HARD τ=2。
**τ は潜在信号の強さでもある**ので、EASY/HARD の違いは「分布の近さ」と「Z 情報量」の両方を動かす（BOUNDED では向きが逆になる）。
mixed-X の Gaussian 列は τ=1 固定（競合なし）。

**列構成.** 単一群（A/B）: 各 candidate model × 3 列（COUNT d=15, POSITIVE d=6, BOUNDED d=9）。
mixed（C）: 各 candidate model × 2 列 + Gaussian 2 列（d=22）。A と B は同じ dataset を共有する（B は rep 1–10、A は rep 1–20）。

## 3. 開発中の dry-run と、それによる実装変更（開示）

公式 run の前に、**smoke と同じ seed block（stage block 1, n=40）だけ**を使った開発用 dry-run を scratch directory で 3 回行った
（成果物ではない。main stage の dataset はこの時点で一度も生成していない）。そこで判明した**数値的失敗**に対して次を変更した。
いずれも失敗（例外）への対処であり、recovery 等の精度指標を理由に変えたものはない。また protocol の科学的条件
（候補集合・nuisance・priority・τ 規則・seed 規則・MCEM 条件）は変えていない。

| # | dry-run で起きたこと | 変更 | なぜ結論を歪めないか |
|---|---|---|---|
| 1 | COM-Poisson ν=0.5 の級数を全項和で計算していたため η≳7 で評価不能（項数上限超過） | モード周りの窓で和をとり、両側の裾を幾何級数で上から押さえる（log-concavity による厳密な上界、相対 tail < e^{-40}）方式に変更 | 値は同じ量の厳密な評価。正規化・モーメント・有限差分テストで照合 |
| 2 | BFGS の line search が試行点として極端な loading を評価し CMP 級数が評価不能 → 候補が失敗 | 全非 Gaussian 候補の loading 最適化を解析 Hessian 付き trust-region Newton（`trust-exact`）に統一。評価不能な**試行点**は棄却（objective = +∞）、受理点と報告 score は常に厳密評価 | 候補目的関数は loading について凹なので最適点は同じ。Poisson/Bernoulli で Phase 9C BFGS と score が 1e-8 相対で一致することを test |
| 3 | START_LAST で CMP ν=0.5 を割り当てた列（観測最大 1026）で、既存 Laplace-Newton E-step の初回更新が η=13.4 へ跳躍し評価不能 | 評価域が有限な spec（CMP）を含むときだけ、評価不能な Newton 試行点で**同じ Newton step を半減**する guard を追加し、回数を `estep_backtracks` として記録 | 目標（同じ対数事後のモードと Laplace 共分散）は不変。G/B/P のみの構成では guard は存在せず既存 sampler そのもの（bitwise test） |
| 4 | driver の構文エラー（case 実行前に停止、0 case） | 修正して再起動 | 実行 0 件 |

Phase 9C の既存関数・既存成果物は変更していない。Phase 9C 構成は新しい generic 経路（`route="phase9c"`）で
assignment・margin・install した loading・refit まで bitwise 再現する（`test_generic_regression.py`）。

## 4. 実装の要点

- `family_specs.py`: FamilySpec（T, log h, A, A', A'', log_prob, sample）。COUNT 4 family（5 candidate model）、POSITIVE 2、BOUNDED 3、REAL 1。
- `model_dual_expfam_generic.py`: per-column consistent クラスの `T(x)=x` 仮定の箇所だけを `T_m(x)`, `A'_m`, `A''_m` に置換。完全な X 対数尤度（base measure 込み）。
- `family_selection_multisupport.py`: Level 0 support gate → Level 1 admissible 候補の score 比較。失敗候補を含む列は選択しない（他候補を勝者にしない）。
- `run_multisupport_family_selection.py` / `drive_…py`: case 単位の atomic 保存、resume、fail-fast、retry なし。
- `summarize_…` / `plot_…` / `audit_…` / `report_…`: CSV からのみ生成。

## 5. 決めていないこと（Human Gate のまま）

- この prototype を正式手法・manuscript evidence に昇格するか（lineage E）。
- nuisance（N, ν, σ, c）の推定、K と family の同時選択、実データ適用。
- BLOCKED family（geometric, NB, gamma, exponential, beta, weibull, student-t, categorical, inverse Gaussian）を扱うための link 変更等のモデル変更。
