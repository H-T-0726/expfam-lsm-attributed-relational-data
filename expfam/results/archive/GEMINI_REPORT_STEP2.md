# Geminiへの報告書 (Report from Claude — Step 2)

## 1. 現在のステータスと結論
Step 2 完走。Poisson（提案手法）vs Bernoulli（ベースライン）のPoC実験が正常終了。
n=150, d=15, k=3, L=10, num_iter=15 の合成Poissonデータで両モデルを比較。

## 2. 定量評価 (Metrics)

| 指標 | Bernoulli（ベースライン） | Poisson（提案手法） | 改善 |
|------|--------------------------|---------------------|------|
| RMSE(Z) [Procrustes] | 0.207040 | 0.167536 | Poissonが19.1%改善 |
| RMSE(X) | 0.313563 | 0.309783 | — |
| Q_final | -7292.08 | -3624.27 | 異族間比較は参考値 |
| w0誤差 | 0.6672 | 0.0228 | — |
| w 誤差 | 0.2341 | 0.0139 | — |

**注**: BernoulliはBernoulli尤度(logistic loss)、PoissonはPoisson尤度(y*eta-exp(eta)、ln(y!)定数除く)を使用。
異なる尤度関数のため絶対値の直接比較は不可。RMSE(Z)を主要比較指標とする。

## 3. 数理的・実装的な懸念点 (Roadblocks)

- Poisson の precision matrix term3 において A''(eta)=exp(eta) がクリップ上限 exp(10)≈22026 に
  達する場合、step size が縮小して収束が遅くなる可能性あり（特に w が大きいとき）。
  → w 初期値 0.1 で対処済み。
- Q関数の異族間比較：Poisson の Q には -ln(Y!) 定数が含まれないため絶対値比較は不可。
- NaN 自律デバッグ機能を実装済み（検知→Z_prev リセット→継続）。今回は NaN 発生なし。

## 4. Claudeからの次の一手 (Next Step) の提案

**Option A【推奨】: 実データ検証に進む**
- Amazon Co-Purchase または DBLP Co-authorship の小規模サブセット（n~500）で
  Poisson モデルを適用し、Bernoulli との比較を実証する。
- "合成データだけでなく実データでも有効" という KDD 必須の実証が得られる。

**Option B: Negative Binomial の実装に進む**
- 過分散パラメータ r の M-step 更新式を実装し、スパースカウントデータでの優位性を示す。
- 理論的完成度が高まるが、実データ実証より先行すると査読者に「実用性」を問われるリスクあり。

**推奨理由**: KDD の審査基準は「実世界のインパクト」を最重視するため、
Option A で実データ実証を固め、その後 Option B で理論を補強する順序が最適。