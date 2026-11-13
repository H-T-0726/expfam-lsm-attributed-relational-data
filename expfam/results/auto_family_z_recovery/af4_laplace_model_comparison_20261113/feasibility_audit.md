# AF4 / S1 Phase A：実現可能性と数理の監査（Laplace 値・logdet・モデル比較は未計算の時点で作成）

- 作成：2026-11-13
- branch：`experiment/af4-model-comparison`（基点は AF3 `86fba01`。その下は AF2 `b0a2398`）
- 実装系列：lineage E ＋ PR #129 の generic model（prototype、本文採用不可）
- 一次出力：`phase_a/phase_a_summary.json`、`reconstruction_audit.csv`（200 状態）、`numerical_validation.csv`（36 点）、`leakage_check.csv`
- 新しい MCEM fit：0 回

## 0. 2 つの量の区別

**現行の Q スコア**（AF2・AF3）は、incumbent の割当て c_t で作った近似事後 q_t を固定し、列ごとに候補 m を比べる量です。

```
Q_j(m | q_t) = max_f E_{q_t}[ Σ_i log p_m(x_ij | fᵀ z_i) ]
```

q_t は c_t に依存します。AF3 では、q_t が c_t の側に選択を引き寄せることを観測しました。

**モデル全体の評価**（AF4）は、割当て c ごとに、その割当ての fit で得た θ̂_c を固定し、Z を積分します。

```
L_c = log p(X, Y_train | θ̂_c, c) = log ∫ p(Z) p(X | Z, θ̂_c, c) p(Y_train | Z, θ̂_c) dZ
    ≈ Φ_c(Ẑ_c) + (nK/2) log 2π − ½ log|H_c|                          （Laplace）
Φ_c(Z) = log p(Z, X, Y_train | θ̂_c, c)
Ẑ_c    = argmax_Z Φ_c(Z)                  （θ̂_c 固定の joint mode。E-step のサンプルではない）
H_c    = −∇²_Z Φ_c(Ẑ_c)                   （nK × nK = 225 × 225 の完全な行列）
```

- L_c は q を含まないので、incumbent の q に依存しません。
- **完全な Bayesian evidence ではありません**（θ は積分せず、MCEM で得た推定値を差し込んでいます）。
- **Schwarz BIC でもありません**（罰則項はなく、観測データの最大尤度でもありません）。
- **Q 関数の値でもありません。**
- 比べる割当ては、どれも自由パラメータの数が同じです（Poisson と CMP(ν=2 固定) は、どちらも列あたり K 個の loading）。そのため、仮に罰則を付けても対比では打ち消し合います。
- 比較の質は θ̂_c の質に依存します（§11）。

## 1. 完全な log joint の定義

```
Φ(Z) = −(nK/2) log(2π var_z) − ‖Z‖²/(2 var_z)
     + Σ_{j∈Gauss} Σ_i [ −(x_ij − η_ij)²/(2σ_j²) − ½ log σ_j² − ½ log 2π ]
     + Σ_{j∈count} Σ_i log p_{c_j}(x_ij | η_ij)
     + Σ_{i<j, (i,j)∈train} [ y_ij η^Y_ij − log(1 + e^{η^Y_ij}) ]
η_ij = f_jᵀ z_i,   η^Y_ij = w0 + w z_iᵀ z_j
```

この Φ は AF4 用に新しく書きました（`log_joint_parts`）。既存実装との照合：
- 既存のモデル項（`calc_log_likelihood_X`、`calc_log_likelihood_Y`）に事前分布を足したものと、差 < 1e-8 で一致しました（テスト）。
- 既存の `laplace_k_criterion.complete_log_density`（＝ AF2 の `mode_phi`）との差は、どの Z（mode、最後のサンプル、サンプル平均）でも同じ定数 Σ_{count 列} log x! でした（ずれ ≤ 9.1e-13、Z によるばらつき ≤ 1.4e-12、200 状態すべて）。
- このずれの原因：`calc_Q_dual_strict_exp` は "poisson" ラベルの列に −Σ log x! を加えますが、generic model は CMP 列にも "poisson" ラベルを付けており、さらにその `log_prob` は自分の基底測度を既に含んでいます。
- この定数は X だけで決まるので、割当て間の差では打ち消し合います。それでも AF4 では独立実装の Φ を正としました。

## 2. X 側の基底測度

- Poisson：h(x) = 1/x!。
- CMP(ν=2)：h(x) = (x!)^(−2)、正規化定数は級数で計算した log Z(η)。
- 実装 `log_prob = log h(x) + η x − A(η)` と照合しました（テスト `test_cmp_column_uses_its_own_base_measure`）。
- 両者とも counting measure に関する pmf なので、比較できます。

## 3. Z の事前分布

- 200 状態すべてで var_z = 1.0（保存値）。
- 正規化定数 −(nK/2) log(2π var_z) を含めています。

## 4. Y の mask と対称関係

- 学習 mask は対称で対角は 0（200/200）。
- Φ では上三角（i<j）の学習ペアだけを 1 回ずつ数えます。
- 既存の勾配・Hessian は、順序付きペアの ½ と、z_i が 2 か所に現れることとが打ち消し合う形になっています。有限差分で一致を確認しました（§8）。

## 5. test ペアの混入

- 学習 mask と test の重なりは 0（200/200）。
- 18 件 × 2 点（mode と最後のサンプル）で、test ペアの Y を全部反転しても、Φ も勾配も**完全に**不変でした（差 0.0）。
- 逆に、学習ペアを 1 つ反転すると Φ は変化しました（最小 0.12）。
- held-out Y は副次指標としてだけ使います。

## 6. θ̂ と Ẑ の整合

保存状態（F、sigma、w0、w、var_z）からモデルを復元し、次を確認しました。
- 保存 Zmode での勾配（最大 8.87e-9）は、AF2 の `mode_grad_inf` と一致しました。
- 既存の Φ は、AF2 の `mode_phi` と**差 0.0** で一致しました（200/200）。
- npz の sha256、データの hash、学習 mask の hash、w0・w もすべて一致しました。
- 非 Gaussian 列の sigma は 1、Gaussian 列の σ² の最小値は 0.379 で、下限値（1e-8）には触れていません。

## 7. Hessian の交差ブロック

```
H_ii = I/var_z + Σ_j f_j f_jᵀ A''_j(η_ij)/φ_j + w² Σ_k C_ik z_k z_kᵀ
H_ik = w² C_ik z_k z_iᵀ − w R_ik I           （i ≠ k、(i,k) が学習ペアのとき）
C_ik = μ_ik(1 − μ_ik),   R_ik = y_ik − μ_ik
```

- 非対角ブロックのノルムは最小でも 11.3 で、H はブロック対角ではありません。
- 解析値と有限差分は、非対角ブロックのノルムで相対 7e-12 の差で一致しました。

## 8. 勾配と Hessian の有限差分検証（`numerical_validation.csv`）

- 対象：rep01–03 × dense/sparse × Oracle/Support-only/Wrong の 18 状態 × 2 点（保存 mode と、最後の E-step サンプル）。
- 勾配：独立実装 Φ の中心差分（h = 1e-5）と比べ、相対誤差の最大は 4.6e-8（許容 1e-5）。
- Hessian：解析勾配の中心差分（h = 1e-5）と比べ、相対誤差の最大は 1.5e-10（許容 1e-5）。

## 9. 正定値性と log determinant

- 保存 mode での Cholesky 分解は 18/18 で成功しました。
- 本番（Phase B）では、slogdet と Cholesky の両方で logdet を計算して一致を記録します。
- ridge や jitter は加えません。勾配が許容値を超える場合や H が正定値でない場合は、値を報告せず状態だけ記録します。

## 10. 多峰性・局所解・近似誤差の限界

- **多峰性**：θ̂ を固定すると F が回転を固定するので、Z の回転による自明な対称性はありません。それでも多峰性は否定できないので、3 つの開始点（保存 mode、最後のサンプル、5 サンプル平均）から mode を求め、差を記録します（best-of-starts は感度分析）。
- **Laplace 誤差**：N(Ẑ, H⁻¹) を提案分布とする重点サンプリング（2000 本）で補正量と ESS を記録します。225 次元では重みが偏りやすいため、相対 ESS < 0.01 の場合は読みません。
- Laplace の誤差を厳密に上から抑えることはできません。

## 11. 推定パラメータの質の違い

- θ̂_c は MCEM（L=5、8 反復）の最終値で、周辺尤度の最大点ではありません。
- 割当てごとに θ̂ の推定誤差の大きさが違う可能性があり、これは比較に混入します。分離するには、θ を最適化し直す計算（新しい推定）が必要で、今回の範囲外です。
- **したがって、AF4 は「保存された fit どうしの比較」です。「割当てそのものの比較」ではありません。**

## 12. 他の基準との違い

| 量 | Z の扱い | θ の扱い | 割当て間の比較 |
|---|---|---|---|
| Q スコア（AF2・AF3） | incumbent q での期待値 | 候補ごとの loading の最大化 | incumbent q を共有するため、q に依存する |
| C_Q（`calc_bic_dual`、KI-010） | q の MC 期待値 ＋ 罰則 | θ̂ | Q 系の量なので比較しない |
| Schwarz BIC | 積分済み（観測尤度の最大値） | 最大化 ＋ (p/2) log n | 今回は計算していない |
| 完全な evidence | 積分 | 積分 | 今回は計算していない |
| **AF4 の L_c** | Laplace で積分 | θ̂_c に固定 | 同じ観測データ・mask・定義の上で比較できる。θ̂ の質の差は残る |

## 判定

**Phase A gate：PASS**（`phase_a_summary.json` の `gate_pass = true`）。

保存状態だけで、各割当ての Laplace 近似を、共通の Φ の定義・共通のデータ・共通の学習 mask の上で評価できます。必要な追加計算は次のとおりで、MCEM は含みません。
- 固定 θ̂ での joint mode の再研磨（開始点 3 つ）
- 完全な Hessian と logdet
- 重点サンプリング

予算：200 状態 × 3 開始点 ＝ 600 回の mode 最適化、重点サンプリング 200 × 2000 本。
