# AF1〜AF8-v3 の経緯：問い・数式・実験・主要結果

- **位置づけ**：分布自動選択（AF）の探索的研究を、時系列で整理したもの。
  - すべて lineage E の prototype で、本文には採用していない（registry の「原稿採用」は ✗）。
  - AF2 以降は、未 merge の PR #129 の generic model（COM-Poisson を含む）を使う。
- **ラベル**：FACT / DERIVED / OBSERVED / INTERPRETATION / HYPOTHESIS。
- **出典**：数値は各 README と CSV（commit `f71c04d`）、[`EXPERIMENT_REGISTRY.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/EXPERIMENT_REGISTRY.md) の AF の行から取った。
- **注意**：AF は修士研究の中心（指数型分布族への拡張）とは別の、付随的な探索である。

結果のディレクトリ：[`expfam/results/auto_family_z_recovery/`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/auto_family_z_recovery)

---

## 0. AF に至るまでの経緯（FACT）

| 時期 | 出来事 | 出典 |
|---|---|---|
| 2026-09-23〜27 | 列ごとの family 選択の設計（Phase 9C）。0/1 列で Bernoulli と Poisson を選ぶ pilot は 36/36 正解。この段階では「確証的な証拠を得ることは目的ではない」としていた | [`reports/distribution_selection/`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/reports/distribution_selection) |
| 2026-10-06 | PR #129：複数の support の候補（count / positive / bounded）からの選択。真の Z では COUNT の 0.984 で family を回復した。Auto の RMSE(Z) は Oracle とほぼ同じだった。CMP の候補で 16 fit が失敗した | [PR #129](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/pull/129)（head [`4fa6159`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/4fa6159)） |
| 2026-10-13 | 先生の質問：「正しい分布で生成して正しい分布を仮定するだけでは、拡張の意味の説明として弱い。間違った分布を仮定すると何が起きるか」。既存の結果を統合し（#130）、M1 を設計した | [`distribution_misspecification_synthesis_20261013.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reports/advisor_feedback/distribution_misspecification_synthesis_20261013.md) |
| 2026-10-13 | M1：真の Bernoulli 属性 6 列を Poisson で fit した（support-compatible な誤指定）。RMSE_Z の差（wrong − correct）は dense +0.020（15/20）、sparse +0.066（17/20） | [`misspecification_m1_bernoulli_to_poisson_20261013.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reports/advisor_feedback/misspecification_m1_bernoulli_to_poisson_20261013.md) |
| 2026-11-04 | E1・E2a：Z の点推定（joint mode）で M1 を再評価した。sparse の悪化は別の seed 系列でも再現した（+0.076 / +0.087、18/20・20/20）。dense の last 推定は再現しなかった | [`e1_z_point_estimator_m1_20261104.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reports/research_improvement/e1_z_point_estimator_m1_20261104.md) |
| 2026-11-10 | PR #168：誤指定の種類 × 列数 × Y の量を拡張した。誤指定の種類によって、Y の情報量との交互作用の向きが変わった | [PR #168](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/pull/168)、[`results_20261110.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/856d475/reports/misspecification_extended/results_20261110.md) |

**AF を始めた理由（INTERPRETATION）**：誤指定が Z の推定を悪くすることが分かったので、「分布をデータから自動で選べば、その悪化を防げるか」が次の問いになった。

### 依存関係

```
M1 → E1 → E2a                                    （誤指定の影響。本来の研究に近い）
M1 のデータ → AF1（Bernoulli vs Poisson）
PR #129 の generic model → AF2（Poisson vs CMP2、新しい生成器）
AF2 の fit・データ → AF3（Q selector の再計算）→ AF4（割当て全体の Laplace）→ AF5（列ごとの Laplace）
AF5 の手法 → AF6（新しい 40 データで事前登録した確認）
AF6 の保存結果 → AF7（事後の原因分解）
AF6 の anchor・データ → AF8 → AF8-v2 → AF8-v3 → AF8-v3 final（40 データの探索実験）
```

---

## AF1（2026-11-09）：Bernoulli vs Poisson の自動選択は、誤指定による Z の悪化を防ぐか

出典：[`af1_20261109/README.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af1_20261109/README.md)、PR #166

- **問い**：誤指定で Z の推定が悪くなる条件で、自動分布選択はその悪化をどれだけ防げるか。
- **仮説**：Auto が正しく選べば、悪化は Oracle の水準まで消える。sparse-Y でも選択が崩れず、開始分布にも依存しない。
- **モデル**：M1 と同じ生成器（n=75、d=12、K=3、X = G3/B6/P3、Y Bernoulli、w0=−1、w=1）。Oracle / Wrong（B6→P6）/ Auto（Phase 9C の hybrid）を比べる。
- **数式（DERIVED）**：x ∈ {0,1} で同じ η について

  ```
  log Bern(x|η) − log Pois(x|η) = e^η − log(1+e^η) > 0        （log(1+u) < u による）
  ```

  したがって、最適化が届いていれば必ず Bernoulli が選ばれる。
- **条件と指標**：20 データ × dense/sparse × 2 seed 系列、L=5、8 反復。主な指標は joint mode の Procrustes RMSE(Z)。
- **事実（OBSERVED）**：
  - 160/160 fit で、12 列すべてを正しく選んだ。
  - Auto は Oracle と bit 単位で一致した（refit の seed を共有しているため）。
  - Benefit（= Wrong の悪化）は sparse で 0.0816 [0.0642, 0.0991]（20/20）、dense で 0.0122（13/20）。
- **解釈**：最も易しいケースでは、選択は確実に機能する。ただし、それは解析的な不等式から来ている。
- **否定された仮説**：なし。ただし「選択器が誤りうる状況での有効性」は、この実験では何も検証されていない。
- **未解決**：family の尤度差が小さい組（例：Poisson と過小分散の count）で、選択がどれだけ正しいか。
- **次へ進んだ理由**：選択が自明でない組（Poisson vs CMP2）で調べる必要があった。

## AF2（2026-11-10）：Poisson vs COM-Poisson(ν=2)

出典：[`af2_20261110/README.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af2_20261110/README.md)、[`af2_part_a_20261110/`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/auto_family_z_recovery/af2_part_a_20261110)、PR #167

- **問い**：選択がデータ依存の判断になる組で、自動選択はどれだけ正しく選び、Z の悪化をどれだけ防ぐか。
- **数式（DERIVED）**：
  - CMP(ν=2) の確率関数は p(x|η) = e^{ηx}/(x!)² / Z(η)、Z(η) = Σ_x e^{ηx}/(x!)² = I₀(2√λ)（λ = e^η）で、過小分散である。
  - {0,1} 列では log CMP2 − log Pois = e^η − log Z(η) > 0 が成り立つので、Poisson は必ず負ける。
  - x ≥ 2 を含む列では点ごとの優越がなく、選択はデータに依存する。
  - 1 観測あたりの期待 KL は 0.027〜0.061 と小さい（PR #129 の較正）。
  - 切片なしの本モデル（n=75）では、{0,1} だけの真 Poisson 列が起きる確率は ≤ 1.01e-10。
  - 負の切片を仮定すると起きるようになり、切片の欠如が選択と交絡する（Part A）。
- **条件**：PR #129 の生成器（n=75、K=3、G3/P3/CMP2×3）、M1 と同じ mask、20 データ。Oracle / Wrong / Support-only（count 列をすべて Poisson）/ Auto。
- **事実（OBSERVED）**：
  - count 列の正解は dense 108/120、sparse 92/120 で、誤りはほぼ CMP2→Poisson の向きだった。
  - RMSE(Z) の差：
    - Wrong − Auto：dense 0.0238、sparse 0.0542。
    - Support-only − Auto：dense 0.0051 [0.0023, 0.0079]、sparse 0.0059 [0.0013, 0.0109]。
    - Auto − Oracle：dense 0.0029、sparse 0.0116。
- **解釈**：完全な誤指定に比べると損失を大きく減らすが、もっともらしい既定（すべて Poisson）からの上積みは小さい。Y が疎になると、X の分布選択まで悪化した。
- **否定された仮説**：「Auto で Oracle の水準に戻る」は、sparse では成り立たなかった。
- **次へ進んだ理由**：CMP2→Poisson の誤りの原因が、Z の不正確さにあるのかを調べる必要があった。

## AF3（2026-11-13）：選択の誤りと、選択に使う Z の関係（MCEM 0 回）

出典：[`af3_selector_rescoring_20261113/README.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af3_selector_rescoring_20261113/README.md)

- **問い**：Z の推定精度が悪くなると、Q スコアで正しい分布を選ぶ率も下がるか。
- **数式**：

  ```
  Q_j(m) = max_f (1/L) Σ_l Σ_i log p_m(x_ij | fᵀ z_i^(l))
  ```

  - 母集団レベルでは、厳密な事後分布から Z を取れば E[log p_true − log p_m] = E_z KL ≥ 0 が成り立つ（DERIVED）。
  - この保証が崩れる場合として、次の 3 つを説明仮説とした（HYPOTHESIS）。
    - (a) 独立な誤差を含む Z では、混合によって過分散になり Poisson が有利になる。
    - (b) 別の分布の事後分布から作った Z では、その分布が有利になる（自己確認的な偏り）。
    - (c) mode の点推定では過小分散に見え、CMP2 が有利になる。
- **事実（OBSERVED、20 データ）**：
  - 真の Z では、真 CMP2 列 60/60 を正解した。
  - 独立なノイズを加えると正解率が単調に下がった（s=0.6 で 0.253、20/20 で傾きが正）。
  - Oracle の Z サンプルでは、RMSE が大きくても 60/60 を正解した。
  - Support-only の Z では、0.783（dense）と 0.533（sparse）に下がった。
  - Oracle の plug-in mode では、逆向きの誤り（Poisson→CMP2）が出た。
- **解釈**：決め手は Z の誤差の大きさではなく、「Z をどの分布の下で作ったか」だった。
- **否定された仮説**：「Z の誤差（RMSE）の大きさで選択の成否が決まる」は否定された。
- **次へ進んだ理由**：q を共有しない評価（Z を積分する評価）を試す必要があった。

## AF4（2026-11-13）：割当て全体を Laplace で評価する

出典：[`af4_laplace_model_comparison_20261113/README.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af4_laplace_model_comparison_20261113/README.md)

- **数式**：

  ```
  L_c = log p(X, Y_train | θ̂_c, c) ≈ Φ_c(Ẑ_c) + (nK/2) log 2π − ½ log|H_c|
  ```

  - H_c は、Y の交差ブロックを含む完全な nK×nK の負の Hessian。
  - θ は積分していないので、evidence でも BIC でも Q でもない。
- **事実（OBSERVED）**：
  - Oracle > Support-only は dense 20/20、sparse 19/20。Oracle > Wrong は 20/20 と 19/20。
  - Q で選んだ Auto（1 列だけ違う場合）との比較では、dense 5/8、sparse 4/7 で、区別できなかった。
  - Hessian 項は Oracle に不利に働いた。
- **解釈**：大きく誤った割当ては退けられるが、実際に問題になる 1 列の違いは n=75 では情報が小さい。そこに θ̂ の差と Laplace の誤差が重なる。
- **否定された仮説**：「モデル全体の Laplace で、1 列の誤りを区別できる」は支持されなかった。
- **次へ進んだ理由**：θ を共有し、1 列だけを入れ替える比較（AF5）で nuisance を消すため。

## AF5（2026-11-13）：列ごとの Laplace 分布選択（探索的）

出典：[`af5_local_laplace_selection_20261113/README.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af5_local_laplace_selection_20261113/README.md)

- **数式**：

  ```
  S_j(m; a) = max_{f_j} [ Φ_m(Ẑ_m(f_j), f_j) + (nK/2) log 2π − ½ log|H_m(f_j)| ]
  Δ_j = S_j(CMP2; a) − S_j(Poisson; a)        Δ > 0 なら CMP2
  ```

  - anchor a の θ_{−j} とほかの列の分布は固定する。
  - これは Laplace 近似による type-II 最尤のプロファイル比である。
- **事実（OBSERVED、AF2 と同じ 20 データ）**：
  - 正解率は dense 0.917、sparse 0.857（同じ anchor の Q は 0.925 と 0.783）。
  - Q との paired 比較は、sparse で +0.40 [0.05, 0.70]、dense で −0.05。
  - 誤りの向きの偏り（CMP2→Poisson）は消えた。
- **解釈**：sparse での改善は、IS で確かめられない列に偏っていた（判定できる列では +0.10 [−0.25, 0.45]）。同じデータで手法を作ったので、確認が必要だった。
- **次へ進んだ理由**：新しい seed のデータで、事前登録して確かめる必要があった。

## AF6（2026-11-14）：Laplace 分布選択の確認実験（事前登録、新しい 40 データ）

出典：[`af6_laplace_confirmation_20261114/README.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af6_laplace_confirmation_20261114/README.md)、PR #169

- **設計**：
  - Phase A で AF5 を独立に監査した。
  - protocol を凍結した後、AF1–AF5 と重ならない seed で 40 データを作り、MCEM を 80 fit 行った。
  - 主要評価項目は、sparse でのデータごとの正解列数の差 D_r（A Laplace − B Q）。
- **事実（OBSERVED）**：
  - **主要評価項目は未達成**：sparse の A−B は +0.225 [−0.075, 0.525]。
  - 正解率（sparse）：A 0.850、B 0.813。
    - 真 Poisson では A 0.792、B 0.975。
    - 真 CMP2 では A 0.908、B 0.650。
  - 副作用は sparse 22 列、dense 9 列で、すべて Poisson→CMP2 の向きだった。
  - TI（Z の数値積分）の符号で選んでも同じ傾向だったので、Laplace 近似の誤差によるものではない。
  - 独立な再計算で、957 個のスコアと主要評価項目を再現した。
- **解釈**：Laplace 型の基準は誤りの向きを入れ替えただけで、正味の改善は小さかった。観測された SD での検出力は約 0.30 で、効果がないとも、小さい正の効果があるとも言えない。
- **否定された仮説**：「AF5 の sparse での改善が、新しいデータで再現する」は、事前に決めた基準では支持されなかった。
- **次へ進んだ理由**：Poisson→CMP2 の副作用の原因を知る必要があった。

## AF7（2026-11-14）：Poisson→CMP2 の誤選択の原因分解（探索的・事後）

出典：[`af7_cmp_bias_analysis_20261114/README.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af7_cmp_bias_analysis_20261114/README.md)、PR #170

- **事実（OBSERVED、AF6 sparse の AUC）**：

  | 基準 | AUC |
  |---|---|
  | 真の Z で固定 | 0.993 |
  | Laplace L | 0.947 |
  | Q | 0.930 |
  | joint MAP の J | 0.919 |
  | 真の θ を anchor にした L | 0.968 |
  | φ_J（Hessian なし） | 0.611 |

  - profile 型の基準は、真の分布によらず CMP2 側へ一様にずれた（中央値 +1.0）。
- **解釈**：Z が潜在であることで、識別できる情報と判定境界の位置が変わる。anchor の θ の推定誤差は主な原因ではない。
- **訂正（AF8 の監査、`mathematical_derivation.md` §0）**：
  - 「profile 化の効果」という呼び方は不正確だった。L と J はどちらも f について profile である。
  - 「真の θ と厳密な積分の両方を使っても 18% 誤った」は、2 つの別々の計算を混同していた。
  - 「KL ≥ 0 なので Bayes 因子は真の分布を支持する」は母集団レベルの性質であり、f を固定値として生成した本研究のデータには、そのままは当てはまらない。
- **次へ進んだ理由**：f を最適化する代わりに積分すれば境界のずれが直るか、を調べるため（AF8）。

## AF8（2026-11-15）：loading を積分した基準（pilot で停止）

出典：[`af8_integrated_loading_20261115/`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/auto_family_z_recovery/af8_integrated_loading_20261115)（`mathematical_derivation.md`、`prior_design.md`）、PR #171

- **数式（DERIVED）**：

  ```
  S_m(f)  = Φ_m(Ẑ(f), f) + (nK/2) log 2π − ½ log|H_ZZ(f)|            （Z の Laplace）
  T_m(f)  = S_m(f) + log p(f|m)
  M_m     = log ∫ exp(S_m(f)) p(f|m) df
  E2      = T_m(f_D) + (K/2) log 2π − ½ log|−∇²_f T_m(f_D)|,  f_D = argmax T   （nested Laplace）
  E       = Ψ(û) + ((nK+K)/2) log 2π − ½ log|−∇²_u Ψ(û)|,  Ψ = Φ + log p(f), u = (vec Z, f)   （joint Laplace）
  ```

  - M − S* は、縮小による当てはまりの損失、log p(f_D)、f の体積項に分かれる。
  - **積分した値は、prior（測度）の選び方に依存する。**
  - prior B（g = √A''_m(0) f ~ N(0, sI)）は平坦極限でも候補間に (K/2) log(a_C/a_P) = −1.00 の差を残す（a_C = A''_CMP2(0) = 0.513）。
- **事実**：開始点の不一致 5.6%（> 5%）で停止規則が作動した。原因は、joint Newton が鞍点に収束したことと、BFGS の LinAlgError である。

## AF8-v2・AF8-v3（2026-11-15）：数値修正

- **v2**（[`af8v2_numerical_repair_20261115/`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/auto_family_z_recovery/af8v2_numerical_repair_20261115)、PR #172）
  - modified Newton を導入した。
  - S4（IS の draw が失敗したセル）が 9/72 で、停止規則が作動した。
- **v3**（[`af8v3_numerical_stability_20261115/`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/auto_family_z_recovery/af8v3_numerical_stability_20261115)）
  - 原因は、φ の丸めで Armijo の判定が止まったことだった（FACT、失敗した draw を再生成して確認）。
  - 証明書型の受理に変えた（H が PD、λ²/2 ≤ 1e-12、|ΔLap| ≤ 1e-10、外側は FD-Newton の λ_f²/2 ≤ 1e-6）。
  - pilot は停止規則 S1–S9 をすべて通過した。

## AF8-v3 final（2026-11-16）：40 データの探索実験（NO-GO）

出典：[`af8v3_full_exploratory/README.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af8v3_full_exploratory/README.md)、[`independent_audit.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af8v3_full_exploratory/independent_audit.md)、PR #173（NO-GO のコメントを投稿済み）

- **条件**：
  - AF6 の rep01–40 を再利用した（新しい MCEM・データなし）。8,640 セル。
  - 主手法 E2、主 prior B_sg2_0.667 を事前に固定した。
- **数値面（FACT）**：
  - 停止規則 S0–S9 をすべて通過した。IS の draw は 384,000 回中、失敗 0。
  - |ΔE2−ΔIS| は median 0.017。
- **主な結果（OBSERVED、pooled 480 列）**：

  | 手法 | 真 Poisson の正解 | 真 CMP2 の正解 | balanced | AUC |
  |---|---|---|---|---|
  | Q | 234/240 | 188/240 | 0.879 | 0.961 |
  | AF6 Laplace | 200/240 | 228/240 | 0.892 | 0.973 |
  | E2 | 223/240 | 216/240 | 0.915 | 0.980 |

  - E2 − AF6 Laplace：balanced +0.023 [−0.004, +0.050]。P→C の誤りは 37→17 に減り、CMP2 の正解は 228→216 に減った。
  - 移行は一方向だけで、判定境界の移動として振る舞った。
  - prior の尺度によって、balanced は 0.68〜0.915 と変わった。
  - E と E2 は sparse で 16 列の符号が逆になった。
- **解釈**：
  - 積分は判定境界を動かすが、どこに動くかは prior で決まる。
  - prior と積分への効果の配分は、座標に依存する（監査 W3）。
- **否定された仮説**：「f を積分すれば、CMP2 の検出を保ったまま P→C の誤りが減る」は支持されなかった（CMP2 も 12 列減った）。
- **研究判断**：NO-GO。正式手法として採用せず、成果物は保存する。

---

## まとめ：AF で否定された仮説・支持された観察

| # | 仮説・観察 | 状態 | 根拠 |
|---|---|---|---|
| 1 | {0,1} 列の Bernoulli と Poisson は自動選択で確実に区別できる | **支持**（ただし解析的に自明） | AF1 |
| 2 | 自動選択で Oracle の Z の精度に戻る | **否定**（sparse の Poisson/CMP2 で） | AF2 |
| 3 | Z の誤差の大きさが選択の成否を決める | **否定**（Z を作った分布が決め手） | AF3 |
| 4 | Z を積分した全体評価で、1 列の誤りを区別できる | **支持されず** | AF4 |
| 5 | 列ごとの Laplace 選択は Q より良い（新しいデータで） | **未達成**（主要評価項目の CI が 0 を含む） | AF6 |
| 6 | Laplace 型基準の Poisson→CMP2 の偏りは、Laplace 近似の誤差による | **否定**（TI でも同じ） | AF6 |
| 7 | f を積分すれば偏りが直る | **支持されず**（境界が移動するだけで、prior に依存） | AF8-v3 |
| 8 | Q スコアは incumbent の分布に引き寄せられる | **観察**（20 データ、統制された比較） | AF3 |
