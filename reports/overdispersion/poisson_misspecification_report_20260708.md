# Poisson 誤指定の影響分析レポート（Phase 3）

作成日: 2026-07-08
スクリプト: `tools/overdispersion/run_poisson_misspecification_check.py`
結果: `expfam/results/overdispersion/poisson_misspecification_{summary,agg}.csv`
図: `figures/overdispersion/poisson_misspec_{heldout_ll,rmse_z,w_err}.png/pdf`

## 1. 実験設定（confirmed）

- 生成: 共有 Z 構造（n=100, d=15, k*=3, X=Bernoulli, w0=1.5, w=0.3）、
  Y ~ NB2(μ_ij, r_true)（gamma-Poisson 混合で正確に生成）。
  r_true ∈ {2, 5, 20, ∞}。∞ = 純 Poisson。
- 推定: strict held-out（train 80% ペア、fixed 系列 masked モデル）で
  Poisson（誤指定）/ NB oracle（r=r_true）/ NB moment（r̂ two-stage）。
- 5 trials × 各条件、全 55 fits、NaN 発生 0。

## 2. 主要結果（confirmed、agg CSV より）

| r_true | 条件付き var/mean | 周辺 var/mean（生成実測） | Poisson te_ll | NB oracle te_ll | NB moment te_ll | Poisson w_err | NB oracle w_err |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | ≈3.5 | 6.36 | **−3.377** | −2.617 | −2.626 | **0.0256** | 0.0043 |
| 5 | ≈2.0 | 4.33 | −2.713 | −2.480 | −2.475 | 0.0093 | 0.0041 |
| 20 | ≈1.25 | 3.40 | −2.359 | −2.323 | −2.322 | 0.0037 | 0.0038 |
| ∞ | 1.0 | 3.10 | −2.230 | —（省略） | −2.229 | 0.0012 | — |

| r_true | Poisson RMSE(Z) | NB oracle RMSE(Z) | held-out Pearson 過分散（Poisson） |
|---:|---:|---:|---:|
| 2 | 0.380 | 0.441 | 4.17 |
| 5 | 0.298 | 0.337 | 2.37 |
| 20 | 0.252 | 0.268 | 1.51 |
| ∞ | 0.236 | 0.237（moment） | 1.20 |

## 3. 何が壊れ、何が壊れないか（Q1 への回答）

1. **held-out 予測尤度は用量反応的に壊れる**（confirmed）:
   r=2 で Poisson は NB 比 −0.76 nats/pair。過分散が弱まるにつれ差は消える
   （r=20 で −0.04、∞ で 0）。**held-out 尤度は誤指定検出器として機能する**。
2. **パラメータ推定（w）が壊れる**（confirmed）: r=2 で w_err が NB 比 6 倍。
   関係の強さ w の解釈（「潜在距離が関係に効く度合い」）が歪む。
3. **held-out Pearson 過分散が残る**（confirmed）: Poisson フィット後も
   r=2 で 4.2。診断量としても機能。
4. **RMSE(Z)（事後サンプル）はむしろ Poisson が小さい**（confirmed な観測、
   解釈は inference）: Poisson は Y 側曲率を μ（真の Fisher 情報
   μr/(μ+r) より過大）と誤認し、事後精度を過大評価する。その結果
   Laplace サンプルがモード周辺に集中し、サンプル vs 真値の RMSE は
   小さく出る。つまり**「過信した誤指定モデル」は点推定指標では
   良く見えることがある**。不確実性の較正（posterior variance の妥当性）
   まで含めると NB が正しい — この検証（coverage 実験）は今後の課題。
5. **moment 推定 r̂ の two-stage は oracle と同等**（confirmed）:
   全 r で te_ll 差 ≤ 0.01/pair。実務手続きとして成立する。
6. **周辺 var/mean は r=∞（正しい Poisson）でも 3.1**（confirmed）:
   潜在構造による周辺過分散の実例。周辺診断で family を選ぶと
   「Poisson は不適」と誤判定する。MovieLens 診断（周辺 9.89 vs
   条件付き ≈1）と同じ構図を人工データで統制的に再現した。

## 4. MovieLens 実データとの対応

- MovieLens の train 残差 r̂ は k=3 で ≈170–200、k=5 で実質 ∞。
  本実験のスケールでは r=20〜∞ の領域（Poisson との差が小さい領域）に相当。
  → 「MovieLens で NB の改善が小さい」ことは人工実験の用量反応と整合する。
- 逆に r̂ が 1 桁のデータ（例: RNA-seq 系、ゼロ過剰な共起カウント）では
  r=2〜5 領域の大きな劣化が予想される（inference; 実データでの確認は今後）。

## 5. 失敗した点・注意

- 初回実行時、図生成コードの列名バグ（`str.replace` の全置換）でスクリプトが
  クラッシュ。**全 55 fits と CSV 保存は完了しており数値に影響なし**。
  バグ修正後、保存済み agg CSV から図を再生成（runinfo の note に記録）。
- RMSE(Z) の解釈（§3-4）は事後サンプルベースの指標特性に依存する。
  モード（事後平均）ベースの RMSE(Z) や posterior coverage での再評価が
  修論では必要（未実施）。

## 6. 修論で使える主張

- 「過分散カウント関係に Poisson を使うと、held-out 予測尤度と
  関係強度パラメータ w の推定が用量反応的に劣化し、NB2（moment 推定 r̂ の
  two-stage で十分）が回復する」— confirmed
- 「点推定の RMSE(Z) は誤指定に鈍感どころか誤指定側に有利に出ることがあり、
  評価指標の選択自体が誤指定研究の論点になる」— confirmed（機構は inference）
- 「周辺 var/mean は潜在構造モデルの family 診断として不適切」— confirmed
