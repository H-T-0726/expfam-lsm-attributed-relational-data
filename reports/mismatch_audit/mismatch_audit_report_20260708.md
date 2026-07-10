# 既存 3×3 分布ミスマッチ実験 監査報告（Phase 7）

作成日: 2026-07-08
スクリプト: `tools/research_audit/audit_mismatch_experiments.py`（read-only 入力）
結果: `expfam/results/mismatch_audit/mismatch_audit_summary.csv`,
`mismatch_audit_old05_conditions.csv`

## 1. 主要成果: KI-003（41.5× の根拠 CSV 未特定）の解決（confirmed）

本文採用実験の raw CSV `expfam/results/exp_scenario_C_exp4_mismatch.csv` を
全条件走査した結果:

| 文書値 | 監査再計算 | 条件 | 根拠 CSV | 判定 |
|---:|---:|---|---|---|
| 41.5×（本文 L.83） | **41.45×** | Scen.C, est X=Gaussian, Y=Poisson（両側誤指定） | exp_scenario_C_exp4_mismatch.csv | **一致・特定完了** |
| 23.6×（図1b 灰バー） | **23.55×** | Scen.C, est X=Gaussian, Y=Bernoulli（先行研究固定） | 同上 | 一致 |
| 3.41×（Scen.A 最大） | **3.41×** | est X=Bernoulli, Y=Bernoulli（X-only 誤指定; Y=Bern は正解） | exp_scenario_A_exp4_mismatch.csv | 一致 |
| 7.35×（Scen.B 最大） | **7.35×** | est X=Poisson, Y=Bernoulli（両側誤指定） | exp_scenario_B_exp4_mismatch.csv | 一致 |

計算定義: 条件別 rmse_Z 平均 / oracle（正指定）条件の rmse_Z 平均。
`reports/claims_and_evidence.md` L.14 の「根拠CSV未特定・弱い」評価と
L.13 の「Scen.B 条件特定が必要」は、**上記のとおり解消可能**
（claims_and_evidence.md 本体は本フェーズでは変更しない。修論フェーズの
ドキュメント更新時に反映すべき事項として記録）。

## 2. fixed 版との対応（confirmed）

| ソース | Scen.A 最大 | Scen.B 最大 | Scen.C 最大 | C の最大条件 |
|---|---:|---:|---:|---|
| 旧 0.5 実装（本文） | 3.41×（XBern/YBern） | 7.35×（XPois/YBern） | 41.45×（XGauss/YPois） | X=Gauss, Y=Pois |
| fixed_official/exp4 | 4.34×（XBern/YBern） | 9.04×（XPois/YBern） | 40.37×（XPois/YBern） | **X=Pois, Y=Bern** |
| distribution_mismatch_fixed | 4.34× | 8.80× | 38.97×（XPois/YBern） | X=Pois, Y=Bern |

**読み取り:**
- 「誤指定で最大 3〜40 倍悪化」という**オーダーの主張は旧版・fixed 版で頑健**。
- ただし **Scen.C の最悪条件は実装間で異なる**（旧: XGauss/YPois、
  fixed: XPois/YBern）。最悪「条件」の同定は 10〜30 試行では不安定であり、
  修論では「最悪条件のラベル」ではなく「最悪倍率のオーダー」と
  「劣化の機構」を主張の単位にすべき（inference）。
- Scen.A の最大 3.41× は X-only 誤指定（Y は正解）、B/C の最大は両側誤指定 —
  root CLAUDE.md の注意書きと一致（confirmed）。

## 3. 0.5 係数問題との関係

- 本文採用 Exp4 は旧 0.5 実装、fixed_official は 0.5 除去版。
  oracle RMSE 自体が改善する（例 Scen.A 0.279→0.232、C 0.0287→0.0231）ため
  倍率の分母が変わるが、悪化倍率のオーダーは保存された（上表）。
- 本フェーズの新実験（overdispersion / shared_z / per_column）は
  **すべて fixed 系列**（`DualExpFamLSMFixed` 継承）で統一しており、
  旧版数値との混在はない（KI-002 遵守）。

## 4. 新研究方向への接続

既存 3×3 実験は「族の取り違え」という**離散的な誤指定**のみを扱う。
本フェーズの実験はこれを 3 方向に拡張する土台になる:

1. **連続的な誤指定（過分散）**: NB2 の r は「Poisson からの距離」を連続に
   制御するパラメータであり、3×3 グリッドの「Poisson 行」を
   r 軸方向に拡張したものと位置づけられる
   （`poisson_misspecification_*` 実験）。
2. **評価軸の拡張**: 既存 Exp4 は RMSE(Z)（真値必要）のみ。strict held-out
   予測尤度は実データでも使える誤指定検出器になる（周辺 var/mean が
   誤誘導する実例は MovieLens 診断で確認済み）。
3. **共有 Z との交絡**: Exp4 の fix_x/fix_w ablation を「共有 Z 仮定の検証」
   として再解釈すると、Scen.C（X 寄与ゼロ）と Scen.A/B（X 寄与あり）の差が
   誤指定被害の非対称性（どちら側の誤指定が致命的か）を説明する候補になる
   （inference; Term2/Term3 の定量比較は今後の課題 — 既存の表現ガイドライン
   どおり「Y 支配」の断定はしない）。

## 5. 修論で安全に使えるもの / 使えないもの

| 主張 | 判定 |
|---|---|
| 誤指定で RMSE(Z) が最大 3〜40 倍悪化（シナリオ依存） | **使える**（旧・fixed 両実装で再現） |
| 41.5× という具体値（旧実装・条件 XGauss/YPois） | 使えるが「旧 0.5 実装での値、fixed 版では 40.37×（別条件）」の脚注必須 |
| 最悪条件がどの family 組合せか | **使えない**（実装・試行間で不安定） |
| 図1b の 23.6× と本文 41.5× の関係 | 使える（同一 CSV 内の別条件と特定済み） |
