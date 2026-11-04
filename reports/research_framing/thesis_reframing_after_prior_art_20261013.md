# 修論の主題・貢献・次の実験の再定義（Issue #138）

- 作業日: **2026-11-04**（ファイル名の日付 `20261013` は Issue #138 の指定に従った。内容は 2026-11-04 時点の証拠に基づく）
- 位置づけ: **THEORY / AUDIT（research framing）**。新しい EM・実験・モデル変更は 0。canonical docs（`RESEARCH_MASTER.md` など）は変更していない。
- 本書は候補を比較して **Human と指導教員が選べる状態にする**ためのもので、修論テーマを決定するものではない。
- 記号: **FACT** / **OBSERVED** / **INTERPRETATION** / **HYPOTHESIS** / **DECISION**（`docs/research_operating_policy.md`）。
- 数値は各出典の一次 artifact と報告書から引用し、本書では再計算していない。系列（lineage）をまたいだ数値の差や倍率は計算しない（KI-002）。

### 参照した成果物と証拠の状態（2026-11-04 時点）

PR が main に入っているかどうかと、結果が技術的に有効かどうかは別の軸である。

| 出典 | PR | main | 技術的な状態 | 種類 |
|---|---|---|---|---|
| Phase 9（K 選択、9K〜9Y） | #127 ほか | 済 | 各 phase の監査済み。`KNOWN_ISSUES.md` §R に主張の制限 | 実験（lineage E） |
| #128 K 選択基準の同一 fit 比較 | #139 | **未** | A1 / B とも独立 auditor PASS（24161 / 11305 checks）。集計表は一次 CSV から再作成して一致 | 実験（lineage E） |
| #129 multi-support family selection | #129 | **未** | audit PASS（BLOCKER 0、WARNING 4）。main stage の失敗 18 fit を記録 | 実験（lineage E） |
| #134 M1 Bernoulli→Poisson | #135 | **未** | 80/80 fit 成功、preflight 全 PASS、focused tests 10/10。**独立 auditor はない** | 実験（lineage E-consistent per-column） |
| #130 誤指定の統合 | #133 | **未** | commit 済み一次 CSV の再集計のみ（推論なし） | synthesis |
| #131 人工データと w/w0 | #132 | **未** | 同上 | synthesis |
| #136 先行研究レビュー | #137 | **未** | 26 本の比較表。本文を読んだのは 6 本、ほかは要旨・抜粋 | literature review |

---

## 1. Why reframing is necessary

- **FACT**: `RESEARCH_MASTER.md` §1 の研究目的は「Y=Bernoulli・X=Gaussian の固定を、X 側・Y 側で独立に指定できる指数型分布族へ拡張し、挙動・有効な条件・限界を評価する」である。
- **FACT（#136 §6, §11）**: 列ごとの指数型分布族の属性と、内積型の関係 Y を共有 Z で結ぶモデルは既存である。
  - Lv, Tang, Zhang 2025 と Fontaine, Kang, Zhu 2025 は K の選択まで含む。
  - CMF（Singh & Gordon 2008、Klami et al. 2014）は、行列ごとに異なる ExpFam の尤度で属性と関係を結ぶ。
- したがって「指数型分布族へ拡張した」こと自体は、主貢献として防御できない。
- 一方で、#136 の検索範囲では次の 2 つが確認できなかった。
  - 共有 Z の X–Y モデルの中で、属性の family を候補から選ぶ研究。
  - **属性の family の誤指定が Z の回復に与える影響を、関係 Y の観測量を変えて同じ dataset の対比較で調べた研究。**
- 修論の主題を、この差分と手元の証拠の強さに合わせて置き直す必要がある。

## 2. Current research assets

| 資産 | 内容（要点のみ。詳細は出典） | 強さ |
|---|---|---|
| 確定した生成モデルと実装 | X: 列ごとの ExpFam `η = f_l^T z_i`、Y: `w0 + w z_i^T z_j`、MCEM。Gaussian / Bernoulli / Poisson | 実装・数式監査済み（`CLAUDE.md` §1–3） |
| 誤指定の証拠（#130） | X 側の誤指定の Z への影響は「差なし」から「数倍の悪化」まで条件に依存する。Y の観測率を下げると all-Gaussian と正しい family の差が広がる（2 系列で同じ向き） | 系列をまたいだ複数の実験。向きは一貫し、大きさは系列・generator で異なる |
| M1（#134） | 同一データ・同一 mask で Bernoulli 列を Poisson として fit。ΔRMSE_Z は dense +0.020（15/20）、sparse +0.066（17/20）、interaction +0.046（15/20 が正） | 事前に凍結した設計、1 条件、20 replicate |
| family selection（#129） | support で絞った候補から score で family を選ぶ。score-decided の recovery は 0.94〜1.00。Auto と Oracle の RMSE_Z の差はほぼ 0 | controlled synthetic。K=3 固定、nuisance 固定、Y の観測量を変える条件なし（protocol に mask の設定がない）、失敗 18 fit |
| K 選択（Phase 9、#128） | C_Q と C_Lap は条件によって成績が入れ替わる。A1: C_Q 37/64、C_Lap 47/64（INCOMPLETE 4）。B: C_Q 25/25、C_Lap 19/19（完了分、INCOMPLETE 6） | prototype の基準。理論的な正当化（一致性など）はない |
| 人工データの条件（#131） | baseline は平均 edge 確率約 0.33 の密なネットワーク。w = 0.71〜1.41 では w が小さいほど難しい | 再集計。w0 の水準（まばらな Y）は未検討 |
| 実データ | Wine（Y はラベル由来）、Cora balanced subset n=280（held-out AP は random の約 2.6〜2.8 倍、BIC 型の基準は k=1）、MovieLens（strict held-out、mixed-X は悪化したネガティブ結果） | characterization。真の family / Z / K は不明 |

## 3. Prior-art constraint

#136 の比較表と本文確認を基礎に、主な制約は次の 4 点である。

1. **mixed / per-column ExpFam の潜在変数モデルは既存**: Moustaki & Knott 2000、GLLVM、GLRM（Udell et al. 2016）。
2. **X と Y を共有 Z で結ぶモデルと、その K 選択は既存**: Lv 2025、Fontaine 2025、CMF。
3. **同じ列に複数の候補 family を当てて選ぶ手法は X 単独では既存**: Valera & Ghahramani 2017、ABDA 2019。行列全体の family の適合度検定は Wang & Carvalho 2023（DMF）。
4. **誤指定の影響が条件に依存すること自体は既知**: Rhemtulla et al. 2012。結合モデルで片方の情報が他方を補うことは Fontaine 2025 と Zhang 2022 が報告している（ただし誤指定ではなく情報の追加の文脈）。

| 先行研究 | 似ている点 | 異なる点 | こちらで未確立な点 |
|---|---|---|---|
| Valera & Ghahramani 2017 | 列ごとに候補の尤度から型を推論、support で候補を絞る | X 単独（関係 Y なし）、K 固定、Bayesian mixture の重み | 本研究の選択は情報量規準型の score。Bernoulli と Poisson の区別は #129 の候補に入っていない |
| Udell et al. GLRM | 列ごとの loss、低ランク構造の共有 | loss は分析者が選ぶ（§8）、関係 Y なし | — |
| mixed-response GLLVM | 列ごとの ExpFam、Laplace / VA による推定 | family はモデル全体で指定し、比較は分析者が行う | — |
| Fontaine et al. 2025 | 属性 GLM + 内積 logistic の関係、Z 共有、ELBO で K | family は指定、誤指定は扱わない、count / categorical なし | — |
| Lv et al. 2025 | 正準 ExpFam の属性 + logistic の関係、Z 共有、K の事後集中を理論で示す | family は指定、Y は Bernoulli のみ | 本研究の K 選択には理論的保証がない |
| Singh & Gordon 2008 / Klami 2014（CMF） | 属性行列と関係行列が因子を共有、行列ごとに異なる尤度、ARD で次元 | 行列単位で指定、選択しない | — |
| Wang & Carvalho 2023（DMF） | family の妥当性と rank の決定の両方を扱う | 行列全体で 1 family、適合度検定、逐次的、X–Y 結合ではない | — |

**検索範囲の限界（#136 §13）**: 英語文献だけで、Web of Science / Scopus は使っていない。cited-by の全件確認もしていない。「確認できなかった」は「存在しない」ではない。

---

## 4. Candidate A — Misspecification-first

**主題例**: 属性情報付き関係データの共有潜在構造モデルにおいて、属性の観測分布の誤指定が潜在構造の推定に与える影響と、それが関係データの情報量によってどう変わるか。

- **問い**: 正しい family と誤った family で同じデータを fit したとき、Z の回復はどう変わるか。その差は Y の観測量で変わるか。誤指定の種類（support の中の別分布 vs 連続としての扱い）で違うか。
- **先行研究との差分**: 共有 Z の X–Y モデルで、Y の観測量と誤指定の影響の関係を同じ dataset の対比較で特徴づける研究は、#136 の範囲では見つからなかった。結論の方向は既存知見（Rhemtulla 2012、Fontaine 2025）と整合するので、**新しい理論ではなく、本研究のモデルにおける定量的な characterization** として位置づける。
- **現在の証拠**:
  - **OBSERVED（M1）**: support の中の誤指定（Bernoulli→Poisson）でも、dense で小さく、sparse で約 3 倍の悪化が出た。ただし replicate ごとの向きは揃っていない（interaction は 5/20 で逆向き）。
  - **OBSERVED（#130）**: 連続として扱う誤指定（6/9 列→Gaussian）は、dense ではほぼ差がなく、Y の観測率 0.1 で明確に悪化した（E-consistent +0.118、8/10）。
- **弱点**:
  - 方法の提案ではなく characterization 寄りである。
  - M1 は 1 条件だけで、sparse は Z の回復そのものが弱い領域である（correct の Gram 相関 0.34）。
  - 誤指定の種類の比較は、現在は系列と generator をまたいでおり、同じ設計の中で比べていない。
  - 機構（曲率 A''/φ の違い、推定した分散による重みの自動的な低下）は HYPOTHESIS のままである。
- **実現可能性**: 高い。M1 の runner・データ・mask をそのまま使う小さな追加（M2）で、主要な空白が埋まる。

## 5. Candidate B — Family-selection-first

**主題例**: 属性情報付き関係データの共有潜在構造モデルにおける、support に基づく候補からの観測分布族の選択。

- **既存の per-column モデルとの違い**: per-column モデルは分析者が列ごとに family を指定する（B 型）。本案は同じ support の中の候補を score で選ぶ（D 型）。
- **分布を事前に指定する場合との違い**: 指定が正しければ Oracle と同じであり、選択の価値は「指定を誤りうるとき」にしか生じない。
  - したがって B の価値の大きさは、**誤指定のコスト（Candidate A の問い）に依存する**。
- **関係 Y を共有 Z で使う意義**: HYPOTHESIS の段階である。Y が Z を通じて候補の score を安定させるかどうかは、#129 では調べていない。#129 には Y の観測量を変える条件も、X だけで選ぶ比較 arm もない。
- **support-aware な候補の選択の役割**: support と合わない候補（例: 実数値に Poisson）を事前に除き、崩壊や比較不能な尤度を防ぐ。
  - 連続の列と離散の列の区別は support の段階で決まる。つまり **「0/1 の列を Gaussian として扱う」誤り（M2 の誤指定）は、support の段階で防ぐ対象**である。
- **prototype で未確立な部分（#129 §5, §7）**:
  - nuisance（N, ν, σ）の推定、K との同時選択、実データ、Y がまばらな条件。
  - **Bernoulli と Poisson の区別は候補に入っていない**（`candidate_registry.csv` の COUNT 群は Poisson / binomial / COM-Poisson / ZTP）。
  - 数値失敗 18 fit（すべて EASY）。原因は、CMP 候補の開始 loading が評価できない領域にあることだった。
  - 離散の探索のコスト（penalty）をどう扱うか。
- **先行研究との差分**: Valera 2017 の考え方を、関係 Y と共有 Z を持つモデルに置くこと。#136 の範囲では、この組み合わせは見つからなかった。
  - ただし手法としての新しさは「組み合わせ」であり、Valera 2017 と同じ問題への比較実験（Bayesian mixture の重み、Gaussian copula による family 指定の回避、X 単独での選択）がないと、審査で優位を主張できない。
- **OBSERVED（#129）**: 条件が揃っていれば選択の精度は高い（score-decided 0.94〜1.00）。しかし Y の観測量を減らさない条件では、**Auto と Oracle の Z の差はほぼ 0 で、Support-only との差も小さかった**（平均 −0.009〜−0.071。主に Support-only が少数の dataset で大きく崩れたことによる）。
  - これは「dense-Y では family の正しさが Z にあまり効かない」という #130 / M1 と整合する。
  - したがって、**選択が Z に効くかを示すには、Y が少ない条件での検証が必要**である。
- **実現可能性**: 中程度。sparse-Y の family selection は、#129 の generic pipeline に Y の mask を通す実装と、比較手法の追加が必要である。

## 6. Candidate C — Family + K model-selection

**主題例**: 属性情報付き関係データの共有潜在構造モデルにおける観測分布族と潜在次元の選択。

- **#128 から判明したこと（OBSERVED）**:
  - A1（Poisson-X、n 50..150、K_TRUE 1,3,5）では、C_Lap が C_Q より多く真の K を選んだ。誤選択はすべて過小選択だった。
  - B（原論文型の generator、Gaussian-X、n=150、K_TRUE 1..9）では、両方完了した 19 データセットで C_Q と C_Lap の選択が全件一致した。
  - Z で条件づけた基準（C_cond_*）は、計算が完了したデータセットのすべてで候補の上端 K=9〜10 を選んだ。
- **C_Q と C_Lap の違い**:
  - C_Q は Q 関数に基づく complete-data / ICL 型の基準で、Schwarz BIC ではない。n に比例する一定の罰則 `P_Z` を含む。
  - C_Lap は θ̂ を固定した観測データ尤度の Laplace 近似で、厳密な周辺尤度ではない。
- **信号が強い条件で C_Q が成功した理由（HYPOTHESIS）**:
  - B の generator は X の各列の分散の 90% を Z で説明する。そのため K_TRUE までは当てはまりの改善（約 800〜5800）が罰則を大きく上回る。
  - K_TRUE を超えると、改善（160〜237）は `ΔP_θ` を超えるが `ΔP_θ + ΔP_Z` に届かない。この算術は確認したが、原因の証明ではない。
  - **A1 と B の差は、信号の強さだけでなく、generator・X の family（Poisson と Gaussian）・n・d・K_TRUE の範囲・候補 K の範囲が同時に違う**ので、どの要因の効果かは分離できない。
- **主貢献にする場合の課題**:
  - 理論: C_Q / C_Lap のどちらにも一致性などの保証がない。Lv 2025 は事後集中の理論を持つ。
  - 数値: 高次元側で Laplace の H が不定または非定常になる（INCOMPLETE）。
  - family と K の同時選択は実装も実験もしていない。
- **修論として完成可能か**: 現時点の証拠では最も遠い。少なくとも、信号強度の橋渡し sweep、Laplace の数値処理の事前登録、family と K の同時選択の実験が必要で、どれも結果次第で主張が成立しない可能性がある。

## 7. Optional Candidate D — Real-data characterization-first

**主題例**: 属性情報付き関係データへの指数型分布族の潜在構造モデルの適用と、その予測性能・安定性の特徴づけ。

- 採らない。理由は次の 3 点である。
  - 既存の実データの証拠は、Wine（Y がラベル由来）、Cora の subset（n=280）、MovieLens（mixed-X で悪化）である。
  - 実データでは真の family / Z / K が分からないため、「適切な分布を選んだ」ことを検証できない。
  - #136 で既存とされた結合モデル（Fontaine 2025、Lv 2025）がすでに実データで予測・補完を示しており、差分が最も小さい。
- ただし、どの候補でも実データは補助の章として必要である（§13）。

---

## 8. Side-by-side comparison

| 観点 | A: Misspecification-first | B: Family-selection-first | C: Family + K selection | D: Real-data-first |
|---|---|---|---|---|
| 先行研究との差分 | 関係 Y の観測量 × 誤指定 × Z の回復の対比較（検索範囲で未確認）。方向は既知と整合 | 共有 Z の X–Y モデルでの D 型の選択（検索範囲で未確認）。Valera 2017 に近い | 列ごとの family と K を同じ結合モデルで選ぶ（検索範囲で未確認） | 小さい |
| 現在の証拠の強さ | **最も強い**。事前凍結の M1 と、系列をまたいで向きが一貫する #130 | 中。選択の精度は高いが、Z への効果は dense-Y でほぼ 0 | 弱い。条件で成績が入れ替わり、理論がない | 弱い。真値なし、ネガティブ結果あり |
| 修論期限までの完成可能性 | 高い（M2 は M1 の runner の再利用） | 中（sparse-Y の family selection の実装と比較手法が必要） | 低い | 中（実験はあるが主張が弱い） |
| 追加実験の規模 | 小（新しい arm 1 つ、40〜80 fit） | 中 | 大 | 中 |
| 一貫したストーリー | 「いつ属性の分布が重要か」→「だから support で候補を絞り選ぶ」へ自然につながる | 「なぜ選ぶ必要があるか」を A の結果に頼る | 2 つの未解決の選択問題を重ねる | 方法の主張がない |
| 主な弱点 | characterization 寄り。1 条件。機構が未測定 | 選択の価値を示す条件（sparse-Y）が未検証 | 理論・数値とも未確立 | 評価できる量が限られる |

## 9. Evidence map

分類: **PRIMARY**（主張を直接支える）/ **SUPPORTING**（補強・文脈）/ **CONTEXT ONLY**（背景）/ **DO NOT USE FOR CLAIM**（その候補の主張には使わない）。

| 証拠 | A | B | C |
|---|---|---|---|
| 原論文の再現（`reproduction/`、Mikawa et al. 2024） | CONTEXT ONLY | CONTEXT ONLY | CONTEXT ONLY |
| 学会予稿の Exp1–4（old 0.5 系列） | DO NOT USE FOR CLAIM（倍率は系列・条件が別、KI-003） | DO NOT USE FOR CLAIM | DO NOT USE FOR CLAIM |
| fixed 系列の Exp4 誤指定（Scen A/B/C） | SUPPORTING（X 全列の誤指定。Y 側の誤指定を含む倍率は分けて書く） | CONTEXT ONLY | CONTEXT ONLY |
| per-column の y-sparsity（legacy F）と #27（E-consistent） | SUPPORTING（Gaussian 化 × Y 観測率） | SUPPORTING（選択の必要性の動機） | CONTEXT ONLY |
| #130 synthesis | SUPPORTING（統合の記述。数値の出典は元の CSV） | SUPPORTING | CONTEXT ONLY |
| **#134 M1** | **PRIMARY** | SUPPORTING（候補の間違いのコスト） | CONTEXT ONLY |
| #129 multi-support family selection | SUPPORTING（「選ぶ」部分の実現可能性） | **PRIMARY** | SUPPORTING |
| Phase 9 K 選択（9K〜9Y） | CONTEXT ONLY（K=3 を固定する根拠の文脈） | CONTEXT ONLY | **PRIMARY** |
| **#128 K 選択基準の比較** | CONTEXT ONLY（limitation の章） | CONTEXT ONLY | **PRIMARY** |
| #131 w/w0 synthesis | SUPPORTING（人工データの条件の説明、まばらな Y が未検討という限界） | SUPPORTING | SUPPORTING |
| Wine（fixed） | CONTEXT ONLY（Y はラベル由来、KI-006） | CONTEXT ONLY | DO NOT USE FOR CLAIM（k=3 の一致は 1 データセット） |
| Cora balanced subset | CONTEXT ONLY | CONTEXT ONLY | DO NOT USE FOR CLAIM（k=1 の選択は未解決、KI-011） |
| MovieLens（strict held-out、mixed-X） | SUPPORTING（limitation: 実データでは切片・スケールが交絡） | SUPPORTING（limitation: mixed_percolumn が悪化） | CONTEXT ONLY |
| NIPS12 | 本書では一次 artifact を確認していない（未分類） | 同左 | 同左 |

## 10. Claim map

| 主張（Candidate A を採る場合） | 状態 | 根拠 |
|---|---|---|
| この固定条件では、support の中の誤指定（Bernoulli→Poisson）でも Z の回復は平均で悪化した（dense +0.020、sparse +0.066） | 言える（OBSERVED、条件つき） | M1 |
| Y の観測率を下げると、連続として扱う誤指定の影響が大きくなった | 言える（OBSERVED、2 系列で同じ向き） | #130 §7 |
| dense-Y では all-Gaussian でも Z の回復はほぼ同じ | 言える（OBSERVED、条件つき） | #130 §6 |
| 誤指定の種類によって影響の大きさが違う | **まだ言えない**（系列・generator をまたいだ比較だけ） | M2 で同じ設計の中で確認できる |
| Y が少ないほど誤指定は必ず悪化する | **言えない**（M1 は 1 条件で、interaction は 5/20 で逆向き） | — |
| 機構は曲率 / 分散の推定による重みの違い | **言えない**（HYPOTHESIS） | — |
| 指数型分布族への拡張だけで新規性が確立した | **禁止** | #136 |
| 自動の family 選択が一般に優れる | **禁止** | #129 §7 |
| 真の K を一般に回復できる / C_Lap が常に C_Q より優れる | **禁止** | `KNOWN_ISSUES.md` §R、#128 |
| 原論文の BIC が誤りだと証明された | **禁止**（式(16) が Z に条件づけた尤度であることは確認したが、誤りの証明ではない） | #128 report |
| Categorical まで実装済み | **禁止**（未実装） | `KNOWN_ISSUES.md` |

## 11. Required next experiments by candidate

| 候補 | 最小限の追加実験（優先順） | 規模 |
|---|---|---|
| A | (1) **M2**: M1 と同じデータ・mask・seed で、6 つの Bernoulli 列を Gaussian として fit する arm。 (2) まばらな Y の w0 の水準の感度（#131 の提案）を、誤指定の対比較と組み合わせるかどうかを判断する | (1) 新しい fit 40（correct の arm を M1 から再利用する場合）〜80 |
| B | (1) Y の観測率を dense / sparse にした family selection（Auto / Oracle / Support-only、できれば X 単独の選択も）。 (2) Bernoulli と Poisson の区別を候補に入れるか決める。 (3) 失敗 18 fit の原因（開始 loading）を、事前に登録した新しい protocol で扱う | 実装 + 数百 fit |
| C | (1) 信号強度の橋渡し sweep（#128 の改善案 1）。 (2) EM 反復による推定不足の切り分け。 (3) family と K を同時に選ぶ実験 | 実装 + 数百〜千 fit |

## 12. M2 decision by candidate

M2 は**実行しない**。各候補の情報需要から、M2 が加える情報を評価した（結果を予想して選んだのではない）。

| 候補 | M2 が加える情報 | 優先度 |
|---|---|---|
| A | 誤指定の**種類**（support の中の別分布 vs 連続としての扱い）の違いを、M1 と同じデータ・mask・seed で比べられる。現在は系列と generator をまたいだ比較しかない。審査や指導教員から最も出やすい問い「単に Gaussian で扱ったら？」に、同じ設計の中で答えられる | **高い** |
| B | 連続か離散かは support の段階で決まるので、M2 は「**support による候補の制限が防いでいる損失の大きさ**」を示す。選択の score 自体の検証ではない | 中（B の主実験は sparse-Y の family selection） |
| C | K 選択には直接の情報を加えない | 低い |

**どちらの結果でも研究判断が変わる**:
- Gaussian 化の悪化が Poisson 化より明らかに大きい → 「誤指定の種類で重みが違う」と書け、support による候補の制限の意味が強まる。
- 同程度か小さい → 「Y が少ないときは、どちらの誤指定も同程度に効く」と書き、family の区別よりも Y の情報量が主な要因だという限界を正直に書く。

## 13. Real-data role by candidate

実データでは真の family / Z / K が分からないので、**truth recovery とは別の目的**になる。今回は実行しない。

| 候補 | 実データで評価できること | 評価できないこと |
|---|---|---|
| A | 属性の family を変えたときの held-out の関係予測（strict held-out）・安定性・Z の変化の大きさ。関係の観測を間引いたときに family の違いが大きくなるかの characterization | 「正しい family」での Z の回復 |
| B | 選ばれた family の記述（どの列で何が選ばれ、どれだけ安定か）、選択の前後での held-out 予測 | 選択が正しかったか |
| C | 選ばれた K の安定性（split・seed）、held-out 予測と K の関係 | 真の K の回復 |

既存の注意: MovieLens の mixed-X では、X に切片がないため評価件数の Poisson 列が Z を支配した可能性がある（registry）。実データの章では、切片・スケールの問題を誤指定の効果と分けて書く必要がある。

## 14. Recommended thesis title candidates

1. 属性情報付き関係データの共有潜在構造モデルにおける観測分布の誤指定の影響 ― 関係データの情報量との関係 ―
2. 関係データと属性を共有する潜在構造モデルにおいて、属性の観測分布はいつ重要か
3. 属性情報付き関係データの潜在構造モデルにおける観測分布の誤指定と support に基づく分布族の選択

（1 が最も控えめで、3 は family selection を副次的貢献として題目に入れる案）

## 15. Recommended research objective candidates

- **目的案 1（主）**: 属性 X と関係 Y を共有潜在変数 Z で結ぶ指数型分布族の潜在構造モデルについて、属性の観測分布を誤指定したときの Z の回復の変化を、同じデータの対比較で定量化する。その変化が関係 Y の観測量と誤指定の種類によってどう変わるかを明らかにする。
- **目的案 2（副）**: その知見に基づき、support で候補を絞った観測分布族の選択を、統制した人工データの条件で評価し、適用できる条件と限界を示す。
- **目的案 3（補助）**: 潜在次元 K の選択基準（Q 関数型の C_Q と Laplace 近似型の C_Lap）が条件によってどう振る舞うかを記述し、本モデルで K を選ぶときの注意点を限界として整理する。

**主貢献と副次的貢献の置き方（提案）**

| 区分 | 内容 |
|---|---|
| 主貢献 | 共有 Z の X–Y モデルにおける、属性の誤指定 × 関係 Y の観測量 × 誤指定の種類の対比較による characterization（M1 + M2 + #130） |
| 副次的貢献 | support に基づく候補の制限と family selection の prototype と、その統制条件での評価（#129） |
| 補助・限界の章 | K 選択基準の比較（Phase 9、#128）、w/w0 の条件（#131）、実データでの characterization |

## 16. Advisor-facing 2–3 minute explanation

1. **先行研究の確認**: 属性を列ごとに異なる分布で扱うこと、属性とネットワークを共通の潜在位置で同時にモデル化すること、潜在次元を選ぶことは、どれもすでにあります（GLLVM、Fontaine 2025、Lv 2025 など）。「指数型分布族へ拡張した」だけでは修論の差分として弱いです。
2. **見つからなかったこと**: 属性の分布を間違えたときに潜在構造の推定がどれだけ崩れるかを、ネットワークの観測量を変えて同じデータで比べた研究は、今回の検索範囲では見つかりませんでした。
3. **すでにある結果**:
   - 0/1 の属性を Poisson として扱うと、ネットワークを十分に観測しているときは悪化が小さく（RMSE で +0.020、20 データ中 15）、観測を 1 割にすると大きくなりました（+0.066、20 中 17）。
   - 連続として扱った場合も、別の実験で同じ向きでした。
   - ただし 1 つの条件で、逆向きのデータもあります。
4. **提案**: 修論の主題を「属性の分布はいつ重要か（関係データが少ないときに効く）」に置き、分布の選択はその応用（副次的な貢献）、潜在次元の選択は限界の章にしたいと考えています。
5. **次の一手**: 同じデータで、0/1 の属性を正規分布として扱った場合（M2）を 1 本追加すると、「誤り方によって影響が違うか」に答えられます。実行してよいか、ご判断をお願いします。

## 17. Preferred framing and why

**推奨: Candidate A（Misspecification-first）。family selection を副次的貢献、K 選択を補助・限界の章とする。**

最重要の根拠: **現在の証拠のうち、事前に凍結した設計で、先行研究の範囲で見つからなかった問いに直接答えているのは M1（と #130 の同じ向きの証拠）だけである。** また、その問いへの答えが B（なぜ選ぶのか）の動機にもなる。

その他の根拠:
- **先行研究との差**: A は Fontaine 2025 / Lv 2025 と競合しない（どちらも誤指定を扱わない）。B は Valera 2017 に近く、比較実験なしでは差分を防御しにくい。
- **完成可能性**: A の最小の追加は M2 だけで、M1 の runner を再利用できる。B は sparse-Y の family selection の実装、C は理論と複数の実験が必要である。
- **ストーリー**: 「dense-Y では分布の違いが隠れ、Y が少ないと効く」→「だから support で候補を絞り、データから選ぶ」→「選択は統制条件で高い精度で働いたが、効く条件はまだ限られる」と、B の結果を副次的な証拠として無理なくつなげられる。
- **#129 の結果との整合**: #129 では dense-Y で Auto と Oracle の Z の差がほぼ 0 だった。B を主にするとこれが弱点になるが、A の主張の中では同じ現象の一例として説明できる。

## 18. What remains uncertain

1. **修論の期限と指導教員の好み**（方法の提案を重視するか、characterization でよいか）。本書では確認できない。Human Gate。
2. **M1 は 1 条件**（n=75、d=12、K=3、G3/B6/P3、Bernoulli-Y、w0=−1、w=1）。sparse は Z の回復そのものが弱い領域である。
3. **Y の情報量の操作は観測ペアの mask だけ**。edge 密度（w0）を下げた場合は未検討（#131 §11）。
4. **誤指定の種類による違いは、同じ設計の中では未比較**（M2 の対象）。
5. **機構は未測定**（曲率、推定した分散による重みの自動的な低下）。
6. **すべて lineage E の prototype の合成データ**。本文への採用は未決定（NOT_DECIDED_BY_HUMAN）。
7. M1 には独立 auditor がない（preflight と focused tests のみ）。主貢献の PRIMARY にするなら、既存の artifact だけを読む小さな照合をするかは Human が判断する。新しい gate は提案しない。
8. 先行研究の検索の限界（英語のみ、cited-by の未確認）。主題を確定する前に Valera 2017・Lv 2025・Fontaine 2025 の cited-by を確認する価値がある。

## 19. Next action

| 優先 | 内容 | 種類 | 誰が |
|---|---|---|---|
| 1 | 本書 §16 で指導教員と framing を確認する | Human Gate | Human |
| 2 | A が採用されたら、M2 を別の experiment Issue として起票し、設計を凍結する（付録の本文案） | 実験の決定 | Human が承認 |
| 3 | M2 の実行（承認後） | CHARACTERIZATION | agent |
| 4 | M2 の結果を見たうえで、B の sparse-Y family selection、または w0 の水準の感度を次に行うか決める | 実験の決定 | Human |
| — | #128 の追加実験（信号強度の橋渡しなど） | — | A を採るなら後回し |

各優先度の判断:
- **M2**: 高い（A の主張の空白を、最小の規模で埋める）。
- **#128 の追加**: 低い（A では限界の章として既存の結果で足りる）。
- **実データ**: 中（どの候補でも補助の章として必要。ただし M2 の後）。
- **sparse-w0**: 中（Y の情報量の第 2 の操作として価値があるが、M2 より 1 段後）。

---

## 付録: 次の experiment Issue の本文案（起票・実行はしない）

> **タイトル案**: Experiment M2 — Bernoulli で生成した属性を Gaussian として誤指定したときの Z 回復（M1 と同一データ・同一 mask）
>
> **背景**: #134 M1 で、support の中の誤指定（Bernoulli→Poisson）の影響を dense / sparse の Y で測った。連続として扱う誤指定（Gaussian 化）の証拠は #130 にあるが、系列・generator が M1 と違い、誤指定の種類の比較ができない。
>
> **Research question**: M1 と同じデータで 6 つの Bernoulli 列を Gaussian として fit したとき、Z の回復は correct とどれだけ違うか。その差は Y の観測量（dense / sparse）で変わるか。Poisson 化（M1）と比べてどうか。
>
> **mode**: CHARACTERIZATION（correct が勝つことは成功条件ではない）
>
> **固定するもの（M1 と同じ）**: generator `canonical-clean-mixed-v1`、n=75、d=12、K_true=K_fit=3、X=G3/B6/P3、Y=Bernoulli、σ²_x=1、f_scale=√2、w0=−1、w=1、data_seed 1001001..1001020、test mask `1341000+r`、sparse の間引き `1342000+10r+regime_index`、モデル seed `1343000+r`、L=5、8 反復、`numerics_mode="consistent"`、`failure_policy="fail_fast"`。
>
> **変えるもの**: wrong-Gaussian arm = G3 / **G6** / P3（列 3〜8 だけ。生の 0/1 をそのまま使う）。
>
> **主要な指標**: `delta_RMSE_Z = RMSE_gaussian − RMSE_correct`（回転だけの Procrustes）、dense / sparse ごとの平均・中央値・IQR・正の数、`interaction = delta_sparse − delta_dense`。
>
> **副次的な指標（事前に決める）**: M1 の Poisson arm との replicate ごとの差 `RMSE_gaussian − RMSE_poisson`、held-out Y の log score / AUC / AP、`|ŵ0+1|`、`|ŵ−1|`、Gram 相関、Gaussian 列の推定分散 σ̂²_x（重みの自動的な低下の記述。機構の証明ではない）。
>
> **規模**: 20 replicate × 2 regime × 1 arm = 40 fit（correct の arm は M1 から再利用する）。
> - 再利用の条件: 同じ commit の fit 経路で、correct の 1 fit を再計算してビット単位で一致すること。一致しなければ correct も再計算して 80 fit とする。
> - M1 の runner に arm の定義を足す最小の wrapper だけを追加し、新しい gate や auditor は追加しない（`CLAUDE.md` §6）。
>
> **言わないこと**: 一般的な誤指定の法則、Gaussian 化が常に悪い、機構の証明。
>
> **Human Gate**: 実行の承認、correct の arm を再利用するかどうか。

---

## DECISION

## **A: MISSPECIFICATION_FIRST_RECOMMENDED**

- family selection（#129）は副次的貢献とする。
- K 選択（Phase 9、#128）は補助・限界の章とする。
- 判断の基準は、新規性の大きさではなく次の 4 点である。
  - 先行研究との差。
  - 現在の証拠の強さ（事前凍結の M1 と、系列をまたいで同じ向きの #130）。
  - 最小の追加（M2）で完成に近づくこと。
  - 指導教員へ一貫したストーリーで説明できること。
- 最終的な修論テーマの採用は、Human と指導教員が判断する。本書では canonical docs を変更していない。

---

## 追記（2026-11-05）: E1・E2a 以降の知見との違い

上の本文は M1（`last` 推定、seed 1 組）の時点の記述で、変更していない。その後の #141 E1（#145）と #142 E2a（#146）で、M1 の読み方が次のように更新された。数値は各報告書からの引用で、本書では再計算していない。

- **dense-Y の効果は、M1 の +0.020（15/20）ほど安定ではない。**
  - E1 で seed を替えると、`last` では +0.002（9/20）になった。seed 間の replicate ごとの相関は 0.03 で、M1 の dense の効果は大部分が実行ごとの揺れだった。
  - `mode`（E1 の主要推定）では、平均 +0.015 / +0.009、中央値 +0.006 / +0.004、正の数 13/20。
  - したがって §4・§10・§16 の「dense +0.020（15/20）」は、M1 時点の観測として読む。「dense-Y でも安定して悪化する」とは言えない。
- **sparse-Y の悪化は頑健だった。**
  - E1 の 6 通り（点推定 3 種 × seed 2 つ）すべてで +0.055〜+0.087、正の数は 16〜20/20。interaction は `mode` で 19/20 が正。
  - E2a（sparse）では、初期値の seed だけ、MC の seed だけを変えたときの差は同程度（|対比| の中央値 0.040 / 0.042）だった。4 通りの seed の組み合わせすべてで Δ > 0 だった replicate は 14/20。
- 多峰性か収束不足かは、E2a では区別できていない。
- 出典: `reports/research_improvement/e1_z_point_estimator_m1_20261104.md`、`reports/research_improvement/e2a_seed_separation_20261104.md`。
- 本書の DECISION（候補 A の推奨）は変更していない。採用は引き続き Human と指導教員が判断する。
