# 共有 Z 仮定の ablation 検証レポート（Phase 5）

作成日: 2026-07-08
スクリプト: `tools/shared_z_ablation/audit_existing_ablation_results.py`（既存結果の棚卸し）,
`tools/shared_z_ablation/run_movielens_shared_z_ablation.py`（新規実験）
結果: `expfam/results/shared_z_ablation/`
図: `figures/shared_z_ablation/movielens_shared_z_ablation.png/pdf`

## 1. 問い

`docs/presentation/seminar_notion_full.md` §14.3 で提起され未解決だった
「1 種類の z_i で属性 X と関係 Y の両方を説明してよいか」を、
Proposed(X+Y) / Y-only(fix_x: F=0) / X-only(fix_w: w=0) の ablation で検証する。
Z は因果的原因ではなく **shared latent factor** として扱う（表現ガイドライン遵守）。

## 2. 既存結果の棚卸し（confirmed、`existing_ablation_audit.csv`）

RMSE(Z)（人工データ、fixed_official/exp4）:

| データ | Proposed | Y-only | X-only | 読み取り |
|---|---:|---:|---:|---|
| Scen.A (Pois-X/Bern-Y) | **0.232** | 0.277 | 0.598 | X も Y も寄与。統合が最良 |
| Scen.B (Gauss-X/Pois-Y) | **0.140** | 0.190 | 0.252 | 同上 |
| Scen.C (Bern-X/Gauss-Y) | 0.0231 | **0.0232** | 1.085 | X の寄与ほぼゼロ（Y-only と同値） |

Wine 実データ（wine_fixed_pilot、AUC_Y）: X+Y ≈ 1.000、Y-only ≈ 0.9997、
X-only = 0.500。Y がラベル由来のため Y-only でほぼ完結し、X は Y 予測に
寄与しない。

## 3. 新規実験: MovieLens strict held-out ablation（confirmed）

k=5、Y=Poisson、test 20% ペア完全除外、3 splits × 2 seeds（18 fits、NaN 0）:

| 指標 | Proposed (X+Y) | Y-only (fix_x) | X-only (fix_w) |
|---|---:|---:|---:|
| held-out Y log-lik / pair | −3.340 | **−3.301** | −7.765（定数 exp(w0) 予測） |
| held-out Y Pearson | 0.955 | 0.955 | —（定数のため未定義） |
| X reconstruction RMSE | 0.311 | 0.500（F=0 の自明値） | **0.272** |
| NMI vs genre（フルZ KMeans） | 0.309 | 0.311 | **0.376** |

**読み取り:**
1. **X（ジャンル）は Y（共評価カウント）の held-out 予測を改善しない**
   （proposed ≈ y_only、差は splits 間標準偏差の範囲内）。
2. **Y はジャンル構造の回復を助けない**。むしろ X-only の NMI（0.376）が
   最高で、Y を入れると Z が共評価（人気・視聴層）方向に引かれ、
   ジャンル軸から離れる（inference）。
3. 共評価カウントの主因子はジャンルではなく人気・視聴者層の重なりで
   あることを示唆（inference; 既存 `movielens_colike_interpretation` の
   因子解釈とも整合的）。

## 4. 総合: 共有 Z 仮定が有効な条件・弱い条件

| 条件 | 証拠 | 判定 |
|---|---|---|
| X と Y が同一の潜在構造から生成される（人工 Scen.A/B） | 統合が単独に最大 2 倍以上優る | **有効** |
| 片側の尤度情報が支配的（Scen.C: 連続 Gaussian-Y） | Y-only と統合が同値 | 中立（統合は害もない） |
| Y が X 由来のラベルから人工構成（Wine） | Y-only で完結 | 中立 |
| X と Y の潜在因子が部分的にしか重ならない実データ（MovieLens: ジャンル vs 共評価） | 統合は Y 予測を改善せず、ジャンル NMI は X-only に劣る | **弱い** |

**修論で使える主張**: 「共有 Z の統合効果はデータの X–Y 潜在構造の重なりに
依存する。人工データでは統合が最良だが、検証した実データ 2 種
（Wine・MovieLens）では単独モデルと同等以下であり、共有仮定は自動的に
正当化されない。fix_x/fix_w ablation は追加実装なしで実行できる
**共有仮定の事前検査**として実務ワークフローに組み込むべきである」。

**言ってはいけないこと**: 「Z が X と Y の原因」（因果ではない）、
「実データ一般で共有 Z は無効」（2 データセット、いずれも Y の構成に
恣意性がある: Wine=ラベル由来、MovieLens=投影カウント）。

## 5. 限界・今後

- NMI はフル Z の KMeans によるもので、既存 pilot（PCA→2D→KMeans）と
  絶対値の互換性がない（実験内比較のみ有効）。
- 分離 Z（Z = [Z_shared, Z_X, Z_Y]）モデルの実装・比較は未着手
  （本 ablation はその必要性の予備証拠を与える）。
- Cora（引用ネットワーク: X=語彙と Y=引用の重なりが強いと期待される）での
  同 ablation は未実施 — 共有 Z が実データで「効く」例を確保する候補。
