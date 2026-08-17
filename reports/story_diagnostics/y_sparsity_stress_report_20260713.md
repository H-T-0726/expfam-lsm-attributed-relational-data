# Y sparsity stress test（軽量版）— 報告

作成日: 2026-07-13
ブランチ: `research/story-diagnostics`
スクリプト: `tools/research_audit/run_y_sparsity_stress.py`
結果: `expfam/results/story_diagnostics/y_sparsity_stress_20260713{.,_agg.,_runinfo.}csv`
図: `figures/story_diagnostics/y_sparsity_{rmse_z,test_y_ll}.png`

**位置づけ: これは軽量版（smoke run）の診断結果である。** trials=2、条件は4つ（`y_only`, `single_gaussian`, `per_column_all`, `all_gaussian`）に限定した予備実験であり、正式な結論を出すものではない。傾向確認が目的。

---

## 1. 目的

既存の `single_vs_joint` 実験（`research/per-column-validation` フェーズ）では、per_column_all と all_gaussian・単独属性モデルの差が小さかった（RMSE_Z 0.235 vs 0.234 等）。既存レポートはこれを「n=80 の全ペアが観測されており Y 側の情報がすでに濃いため、X の追加寄与が相対的に小さい」という**仮説**として述べていたが、この仮説自体を検証する実験は行われていなかった。

本実験は、Y の学習観測率（train pair の割合、`y_obs_rate`）を 1.0→0.5→0.2→0.1 と下げていき、X を使うモデル（`per_column_all` 等）が `y_only` との差をどう広げるかを確認する。

## 2. 手法

- データ生成は `run_per_column_single_vs_joint.py` の `generate()` をそのまま複製（N=80, D=9, K_TRUE=2, Gaussian3+Bernoulli3+Poisson3, Poisson-Y, w0=1.2, w=0.3）。
- 固定 test set（`test_ratio=0.2`、全 `y_obs_rate` 条件で共通・trial内で同一）で評価。残り80%の「学習可能プール」から `y_obs_rate` 割合をランダム間引きして `train_mask` を作る（`y_obs_rate=1.0` はプール全部＝既存 single_vs_joint と同じ密度）。
- 条件（軽量版のため4条件に限定）: `y_only`（X不使用）, `single_gaussian`（Gaussian3列のみ）, `per_column_all`（9列 family_x_list 正指定）, `all_gaussian`（9列を生値のまま全列Gaussian強制、誤指定比較用）。
- trials=2（`DATA_SEED_BASE=94000`, `MODEL_SEED_BASE=95000`, `SPLIT_SEED_BASE=96000`, `THIN_SEED_BASE=97000`）、L=5, num_iter=8。
- 指標: RMSE_Z（Procrustes整合後）、test Y log-likelihood/pair、test Y RMSE、w0/w誤差、n_train_pairs。

## 3. 結果（trials=2平均、`y_sparsity_stress_20260713_agg.csv`）

### RMSE_Z（低いほど良い）

| y_obs_rate | n_train_pairs | y_only | single_gaussian | per_column_all | all_gaussian |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 2528 | 0.308 | 0.237 | **0.226** | 0.235 |
| 0.5 | 1264 | 0.457 | 0.276 | **0.262** | 0.267 |
| 0.2 | 506 | 0.802 | 0.358 | **0.319** | 0.647 |
| 0.1 | 253 | 1.187 | 0.418 | **0.343** | 0.835 |

### test Y log-likelihood / pair（高いほど良い）

| y_obs_rate | y_only | single_gaussian | per_column_all | all_gaussian |
|---:|---:|---:|---:|---:|
| 1.0 | −1.988 | −1.965 | **−1.957** | −1.959 |
| 0.5 | −2.072 | −1.985 | **−1.971** | −1.974 |
| 0.2 | −2.307 | −2.024 | **−2.002** | −2.138 |
| 0.1 | −2.528 | −2.064 | **−1.994** | −2.146 |

- NaN・発散は32 fit中0件。

## 4. 暫定的な解釈

- **y_obs_rate=1.0（既存実験と同じ密度）では、4条件の差はごく小さい**（RMSE_Z 0.226〜0.237）。これは既存 `single_vs_joint`（per_column_all 0.235 vs all_gaussian 0.234）の傾向と整合する。
- **y_obs_rateを下げるほど、y_only は急速に悪化する**（RMSE_Z 0.308→1.187）。Y情報が疎になるほどXなしでは推定が難しくなるのは自然な結果。
- **per_column_all と single_gaussian は y_obs_rate=0.1でも比較的安定**（RMSE_Z 0.343, 0.418）。y_only との差は y_obs_rate=1.0時点の約1.3〜1.4倍から、0.1時点で約3〜3.5倍に拡大しており、**Yが疎なときにXの寄与がより大きくなる、という既存仮説を支持する結果**が得られた。
- 一方で **all_gaussian（誤指定）は y_obs_rate が下がるにつれ per_column_all より大きく悪化する**（0.2で0.647、0.1で0.835、per_column_allの約2〜2.4倍）。Y情報が豊富なときは誤指定の悪影響が隠れていたが、**Y情報が乏しくなるほどfamily誤指定のコストが顕在化する**という、既存 single_vs_joint 実験だけでは見えなかった新しい傾向が確認できた。

## 5. 発表で使える主張（軽量版・prototype前提）

**言ってよい主張:**
- 「Yの学習観測率を下げていくと、Xを使うモデル（per_column_all等）とy_onlyの差が広がる傾向が、軽量な予備実験（trials=2、4条件）で確認できた」
- 「同時に、family誤指定（all_gaussian）のコストもY情報が乏しいほど大きくなる傾向が見られた」
- 「これは、既存の single_vs_joint 実験で差が小さかった理由についての仮説（Y情報が濃い設定だったため）と整合する結果である」

**言いすぎな主張（避ける）:**
- 「Y疎データでper_columnが有効であることを証明した」（trials=2、4条件のみの軽量予備実験であり、フルスケール検証ではない）
- 「per_column_all が single_gaussian より明確に優れている」（差は同程度〜やや小さい範囲で、trial数2では分散の評価が不十分）
- 「all_gaussianが常に悪化する」（本設定固有の傾向であり、他のデータ生成設定での一般化は未確認）

## 6. 限界・次のステップ

- **trials=2は傾向把握のための最小規模**。分散（std）の信頼性は低く、seedを増やした確認が必要。
- **条件は4つに限定**（`single_bernoulli`, `single_poisson`, `all_bernoulli` は未実施）。`ACTIVE_CONDITIONS` 定数を変更するだけでフル条件に拡張可能な構造にしてある。
- 本結果を確認した上で、フル条件・trial数拡張、または実験2（complementary blocks）・実験3（MovieLens attribute diagnosis）への着手を判断する（`story_diagnostics_next_plan_20260713.md` 参照）。
- `EXPERIMENT_REGISTRY.md` への追記は、結果を確認し採用を決定してから行う（今回はまだ実施していない）。
