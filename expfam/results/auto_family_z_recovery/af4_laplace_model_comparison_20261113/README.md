# AF4 / S1：Z を積分したモデル全体の評価（Laplace）で、正しい分布割当てを誤った割当てから区別できるか

- 実験 ID：AF4-S1（2026-11-13）
- branch：`experiment/af4-model-comparison`。AF3（`86fba01`、未 push）の上で、その下は AF2 = PR #167、AF1 = PR #166。PR #129 は未マージ
- 実装系列：**lineage E ＋ PR #129 の generic model（experimental prototype、本文採用不可）**。extra 1/2 なし
- 新しい MCEM fit：**0 回**
- 追加の計算：
  - 固定 θ̂ での joint mode の最適化 600 回（200 状態 × 開始点 3）
  - 完全な Hessian（225×225）と logdet 600 回
  - 重点サンプリング 40 万本
- protocol：`protocol.json`（canonical sha256 `2d063ac2…87d6`）。Laplace 値をまだ 1 つも計算していない時点で、commit `27452ad` で固定した
- 実行：`full/runinfo.json`（git `27452ad`、tracked-dirty = False、351 秒、警告 0）

## 0. 結論（先に）

1. **[OBSERVED] Laplace による全体評価は、少数の割当て全体の比較では正しい割当てを上位にした。**

   | 比較 | dense | sparse |
   |---|---|---|
   | Oracle > Support-only（データ数） | 20/20、平均差 15.3 [13.0, 17.7] | 19/20、平均差 10.0 [7.1, 12.6] |
   | Oracle > Wrong（データ数） | 20/20、37.9 [33.6, 41.9] | 19/20、22.7 [18.4, 26.7] |

   - AF3 で、Support-only の Z に対する Q スコアが CMP2 列を誤選択したデータに限っても、Oracle が上位だった（dense 10/10、sparse 18/19）。
   - 事前宣言した R1 は該当した。
2. **[OBSERVED] ただし、Q スコアが実際に選んだ割当て（Auto）との比較では、区別は弱かった。**
   - Auto（Poisson 開始）の割当てが Oracle と異なるデータで、Oracle が上位だったのは dense 7/10（平均 +0.65、CI [−2.45, 3.31]）、sparse 12/16（+1.57、CI [−0.79, 3.58]）。
   - **異なる列が 1 列だけのとき**は dense 5/8（平均 +0.34）、sparse 4/7（+0.07）で、事実上区別できていない（事後表）。
3. **[OBSERVED] 差を生んだのは、真 CMP2 列の log joint だった。Hessian 項は逆向きに働いた。**
   - Oracle − Support-only の平均：ΔΦ(真 CMP2 列) は dense +34.5、sparse +43.8。−½Δlog|H| は dense −10.9、sparse −19.9。
   - 真 CMP2 を使う Oracle では Z の事後分布が鋭くなる（|H| が大きい）ので、Laplace の体積項が Oracle に不利に働く。
   - 一方、Oracle − Wrong では、真 Poisson 列の log joint が Wrong の方を支持した（dense −21.5、sparse −47.0）。mode での当てはまりだけを見ると、過小分散の CMP2 が有利に見える（AF3 の plug-in mode の偏りと同じ向き）。**Φ(Ẑ) だけを列ごとに比べる方法は、健全な基準ではない。**
4. **[OBSERVED] 数値面は安定していた。**
   - 200/200 で、mode は停留し（|∇| ≤ 1e-8）、Hessian は正定値だった。
   - 3 つの開始点からの mode の差は最大 2.9e-9（多峰性は見られなかった）。
   - slogdet と Cholesky の logdet は一致した。
   - 重点サンプリングの補正を加えても、読める組（相対 ESS ≥ 0.01）で符号が反転した例は 0 だった。
   - ただし相対 ESS は低く（中央値 0.015〜0.057、32/200 は < 0.01）、Laplace の誤差は 1 モデルあたり O(1)（補正量の中央値 −0.5〜−1.9）と見積もられる。この大きさは、1 列だけ違う場合の差と同じ程度である。
5. **[INTERPRETATION]**
   - q に依存しないモデル全体の評価は、incumbent の q による Q 比較の自己確認的な偏りの影響を受けずに、大きく誤った割当てを退けた。
   - しかし、実際に問題になる「1 列だけの誤り」については、n=75 では情報量が小さい。そこに、割当てごとに別々に推定した θ̂ の差（Y 項・Gaussian 列の項の SD が 4〜12）と、Laplace の誤差が重なり、区別できなかった。
   - **AF4 は、新しい自動選択法が有効であることを示していない。**

## 1. 研究問いと位置づけ

**RQ**：固定した近似事後サンプル上での列ごとの Q 比較の代わりに、割当てごとに Z を積分したモデル全体の評価を行うと、正しい割当てを誤った割当てから区別できるか。

**AF3 から AF4 へ進んだ理由**：AF3 では、Q スコアは「Z を作るのに使った分布」の側に引き寄せられた（Oracle の Z では正解、Support-only の Z では誤選択）。そこで、q を共有しない評価を試した。

**位置づけ**：
- Oracle は真の割当てを知っている診断用の参照であり、実運用できる選択法ではない。
- 主指標は「真実を知った後での判別力」であって、選択の正解率ではない。

## 2. 数式（詳細は `feasibility_audit.md`）

### 2.1 Q スコア（AF2・AF3）

```
Q_j(m | q_t) = max_f E_{q_t}[ Σ_i log p_m(x_ij | fᵀ z_i) ]
```

q_t は incumbent の割当て c_t の下で作られる。同じ q_t の上で比べると、その q_t が支持する側に偏る（AF3）。

### 2.2 モデル全体の評価（AF4）

```
L_c = log p(X, Y_train | θ̂_c, c) = log ∫ exp Φ_c(Z) dZ ≈ Φ_c(Ẑ_c) + (nK/2) log 2π − ½ log|H_c|
Φ_c(Z) = log N(Z; 0, var_z I) + Σ_j Σ_i log p_{c_j}(x_ij | f_jᵀ z_i) + Σ_{i<j∈train} log Bern(y_ij | w0 + w z_iᵀ z_j)
```

- Ẑ_c は θ̂_c を固定したときの joint mode、H_c は Y の交差ブロックを含む完全な nK × nK の負の Hessian。
- θ は積分しないので**完全な Bayesian evidence ではない**。最大化した観測尤度に罰則を加えたものでもないので **Schwarz BIC でもない**。**Q でもない**。
- 割当ての間で自由パラメータの数が同じなので、罰則は対比に影響しない。
- 異なる q の Q 値どうしを比べることはしていない。

### 2.3 差の分解

```
ΔL = Δ[prior] + Δ[X Gauss 列] + Δ[X 真 Poisson 列] + Δ[X 真 CMP2 列] + Δ[Y_train] + Δ[−½ log|H|]
```

θ̂ は fit ごとに違うので、Gaussian 列や Y の項も差に入る（nuisance）。

## 3. 実装の正当性（Phase A、`phase_a/` と `feasibility_audit.md`）

| 確認 | 結果 |
|---|---|
| 保存状態からのモデル復元 | 既存の Φ が AF2 の `mode_phi` と差 0.0 で一致（200/200）。保存 mode での勾配は最大 8.87e-9 |
| 独立に実装した Φ と既存の Φ | 差は Z によらない定数 Σ_{count 列} log x!（ずれ ≤ 9e-13）。原因は、既存の strict Q が CMP 列にも Poisson の階乗補正を足していること |
| 勾配（有限差分） | 相対誤差の最大 4.6e-8（36 点） |
| Hessian（有限差分） | 相対誤差の最大 1.5e-10。Y の交差ブロックは非零（ノルム ≥ 11.3）で一致 |
| test ペア | 反転しても Φ・勾配の変化は 0.0。学習ペアの反転では変化（≥ 0.12） |
| 正定値性 | 200/200 で正定値（最小固有値は dense ≥ 2.59、sparse ≥ 0.75、条件数 ≤ 706） |

## 4. 結果（`full/analysis/results_tables.md`）

### 4.1 primary：paired 比較（データセット単位）

| 比較 | regime | n | Oracle 上位 | 下位 | 平均 Δ [95% CI] | 最小 Δ |
|---|---|---:|---:|---:|---|---:|
| Oracle − Support-only | dense | 20 | 20 | 0 | 15.34 [13.00, 17.71] | 4.12 |
| Oracle − Support-only | sparse | 20 | 19 | 1 | 10.02 [7.10, 12.56] | −7.94 |
| Oracle − Wrong | dense | 20 | 20 | 0 | 37.90 [33.63, 41.86] | 17.00 |
| Oracle − Wrong | sparse | 20 | 19 | 1 | 22.73 [18.44, 26.69] | −1.42 |
| Oracle − Support-only（AF3 の Z5 で誤選択したデータ） | dense | 10 | 10 | 0 | 14.49 | 10.55 |
| Oracle − Support-only（同上） | sparse | 19 | 18 | 1 | 9.84 | −7.94 |

### 4.2 secondary：Q で選んだ割当て（Auto）との比較（割当てが異なるデータのみ）

| 比較 | regime | n | Oracle 上位 | 平均 Δ [95% CI] |
|---|---|---:|---:|---|
| Oracle − Auto（Poisson 開始） | dense | 10 | 7 | 0.65 [−2.45, 3.31] |
| Oracle − Auto（Poisson 開始） | sparse | 16 | 12 | 1.57 [−0.79, 3.58] |
| Oracle − Auto（CMP2 開始） | dense | 9 | 6 | 0.63 [−2.86, 3.74] |
| Oracle − Auto（CMP2 開始） | sparse | 17 | 13 | 2.13 [−0.03, 4.07] |

事後表（`posthoc_by_n_differing_columns.csv`、Oracle − Auto（Poisson 開始））：

| 異なる列の数 | dense：Oracle 上位 / n、平均 Δ | sparse：Oracle 上位 / n、平均 Δ |
|---|---|---|
| 1 列 | 5/8、+0.34 | 4/7、+0.07 |
| 2 列 | 2/2、+1.88 | 5/6、+1.42 |
| 3 列 | — | 3/3、+5.35 |

### 4.3 分解（平均）

| 比較 | regime | ΔL | ΔΦ | −½Δlog\|H\| | X 真 CMP2 列 | X 真 Poisson 列 | X Gauss 列 | Y |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| O − Support-only | dense | 15.34 | 26.26 | −10.92 | +34.53 | +0.08 | −1.22 | −7.39 |
| O − Support-only | sparse | 10.02 | 29.87 | −19.85 | +43.76 | −1.57 | −7.01 | −4.56 |
| O − Wrong | dense | 37.90 | 34.36 | +3.54 | +34.36 | −21.48 | +1.18 | +18.90 |
| O − Wrong | sparse | 22.73 | 14.15 | +8.58 | +44.55 | −47.00 | +5.49 | +10.43 |

- ΔL の符号が ΔΦ の符号と違ったのは、Support-only で 0（dense）と 1（sparse）、Wrong で 0 と 3（sparse）、Auto で 3 と 5。
- **Oracle が下位になったケース**（`posthoc_oracle_lower_cases.csv`）の多くは、ΔΦ > 0 なのに Hessian 項が上回ったもの。例：sparse rep06 の Support-only は ΔΦ +20.1、−½Δlog|H| −28.0。
- Gaussian 列の項（dense rep19 の Auto で −14.3）や Y 項が効いたケースもあった。

### 4.4 頑健性と副次指標

- best-of-starts による符号の反転：0。
- 重点サンプリングの補正による符号の反転（読める組）：0。読める組の数は、O − Support-only で 19・13、O − Wrong で 20・10。
- held-out Y（Laplace 予測、test ペア）：Oracle が高かったのは、Support-only との比較で 12/20・13/20（平均差 +0.001・+0.004）、Wrong との比較で 19/20・17/20。**Y の予測では、count 列の分布の違いはほとんど区別できない**（AF2 と同じ）。
- 図：`full/analysis/fig1_paired_laplace_differences.png`、`fig2_decomposition.png`。

## 5. 研究上の問いへの答え

1. **Q で誤った側が支持されていたデータで、Laplace は正しい割当てを支持するか**：
   - 割当て全体の比較（Oracle と Support-only）では、支持した（10/10、18/19）。
   - Q が実際に出した割当て（Auto）との比較では、1 列だけ違う場合はほぼ半々で、**支持したとは言えない**。
2. **dense と sparse の違い**：sparse の方が差が小さく、Hessian 項が大きい（Z の事後の不確実性が大きい）。逆転は sparse でだけ起きた。
3. **改善の源**：真 CMP2 列の log joint によるもの。Hessian 補正は Oracle に**不利**に働いた（O − Support-only）。O − Wrong では、Hessian 項と Y 項が Oracle に有利で、真 Poisson 列の log joint は Wrong に有利だった。
4. **正しい割当てが選ばれない理由**：
   - (a) 1 列あたりの情報が小さい（n=75）。
   - (b) Oracle（鋭い事後）に対する Laplace の体積項の不利。
   - (c) 別々の MCEM fit で推定した θ̂ の差が、Gaussian 列・Y の項に数単位の揺れを入れる。
   - (d) Laplace 自体の誤差が O(1)。
   - これらの寄与の大きさは分離していない。
5. **mode の最適化不足や Laplace の不安定さ**：mode は一意で十分に収束しており、問題はない。Laplace の誤差は IS でも反転を生まなかったが、IS の ESS が低く、1 列だけの差の大きさに比べて誤差を無視できるとは言えない。
6. **次のアルゴリズムの根拠になるか**：
   - 「q を共有しない評価は、大きく誤った割当てを退ける」ことの根拠にはなる。
   - 「1 列ごとの判定で Q 選択より良い」ことの根拠にはならない。
   - nuisance（θ̂ の差）を制御した比較が必要である。

## 6. 限界・主張できないこと

- 条件は 1 つ（n=75、K=3、τ=2、ν=2 固定、切片なし）、20 データ、各割当ての fit は MCEM 1 seed 系列。
- θ̂ は MCEM（L=5、8 反復）の値で、周辺尤度の最大点ではない。割当てごとの推定の質の差は分離していない。
- X はすべての観測を推定に使っているので、X の対数尤度は in-sample であり、**held-out X ではない**。
- Laplace の誤差は IS で診断しただけで、225 次元では ESS が低い。
- 「新しい自動選択法を得た」「Laplace で分布を正しく選べる」「BIC で選べる」「実データでも有効」とは書かない。Oracle との比較は選択の正解率ではない。

## 7. 次の検証の候補（実装しない。Human Gate）

| 候補 | 内容 | AF4 との関係 |
|---|---|---|
| **S2-B（推奨）** | Laplace を使ったモデル全体の選択を、nuisance を制御して検証する。曖昧な列 j の候補ごとに、ほかの列と Y のパラメータを共有したまま、f_j と Z を Laplace 目的で最適化し直し、L_c を比べる。IS で誤差を診断する。Q 選択（AF2）、Oracle、Support-only と同じデータで比べる | AF4 は、(c) の nuisance が 1 列の差を覆い隠すことを示した。S2-B はそれを取り除く最小の変更で、MCEM なしでも始められる |
| S2-A | 候補ごとに再推定し、held-out X（列の一部を隠す）の予測で比べる | double use を避けられる点は優れる。ただし E-step に X の mask を入れる prototype の変更が必要で、候補ごとに MCEM の fit が要る。n=75 では隠せる観測が少なく、1 列の判別力はさらに小さい見込み |

## 8. 成果物

| ファイル | 内容 |
|---|---|
| `protocol.json`、`manifest.csv` | 凍結した protocol と 200 状態の計画 |
| `feasibility_audit.md` | 数理監査 12 項目と Phase A の判定 |
| `phase_a/` | `phase_a_summary.json`、`reconstruction_audit.csv`、`numerical_validation.csv`（有限差分）、`leakage_check.csv` |
| `full/model_comparison.csv` | 200 状態：Laplace、Φ とその分解、logdet（2 通り）、固有値、mode の診断、best-of-starts、IS、held-out Y |
| `full/diagnostics.csv` | 600 回の mode 最適化（開始点ごと） |
| `full/analysis/paired_comparison.csv`、`paired_summary.csv` | paired 差と集計（事前登録） |
| `full/analysis/diagnostics_summary.csv`、`summary.json`、`results_tables.md`、図 2 枚 | 集計 |
| `full/analysis/posthoc_*` | 事後の記述表（異なる列の数ごと、Oracle が下位のケース、nuisance のばらつき） |
| `full/runinfo.json`、`smoke/` | provenance と、rep01 だけの確認実行 |

コード（`expfam/src/experimental/`）：
- `run_af4_laplace_model_comparison.py`
- `analyze_af4_laplace_model_comparison.py`
- `test_af4_laplace_model_comparison.py`（5 件）

AF2 の状態は `D:\tento\kennkyu_artifacts\auto_family_z_recovery_af2_20261110\full\fits\`（sha256 は AF2 の `fit_runs.csv` と照合済み）。

**開示**：`posthoc_*` の表は、本実行の結果を読んだ後に追加した。事前登録した数値・規則は変えていない。

## 8b. 独立監査（read-only の `research-auditor`、本実行の後）と対応

- **BLOCKER：0 件。** 次の点が、コードと一次 CSV で確かめられた。
  - Φ（Gaussian 列の sigma は分散行列の対角、count 列は基底測度込み、Y は学習の上三角だけ）
  - 勾配と Hessian（交差ブロックを含む）
  - Laplace の式（rep01 / dense / oracle を手計算 −2360.56585 で再現）
  - log x! のずれの原因
  - リークがないこと
  - paired 差と分解の和
  - AF3 での絞り込み
  - EM を呼んでいないこと
- **WARNING 1（protocol の欠陥）**：`protocol()` が `"mode"` キーを 2 回定義しており、後の定義（mode 最適化の設定）が前の定義（「CONFIRMATORY、Laplace 値を計算する前に作成」）を上書きした。そのため凍結した `protocol.json` には、確認的研究であることの宣言文が残っていない。
  - **凍結済みの protocol は遡って直さない。**
  - 事前登録の時点は git で確認した：commit `27452ad` には protocol・Phase A・コードだけが含まれ、`full/` は含まれていない。それ以降に変わったコードは、分析スクリプトへの post-hoc 部分の追加（+39/−1 行）だけである。
- **WARNING 2**：比較は「割当てどうし」ではなく「fit どうし」である。割当てが同じ列（Gaussian 列）や Y の項にも、θ̂ の差による平均 −1〜−7 程度の差が入っている（§4.3、§6）。
- **WARNING 3**：sparse の O − Wrong で **R3 に該当した**（Hessian 項が符号を決めたデータが 3/20）。R4 の「頑健」は、読める IS の組が少ないうえでの判定である（sparse の O − Wrong は 10/20、Auto 比較は 8/16）。IS の batch SD（最大 1.24）は、Auto 比較の差と同じ大きさである。
- **NOTE**：
  - 3 つの開始点はどれも同じ事後領域から始まっているので、多峰性の調べ方としては弱い。
  - `runinfo.mcem_fits = 0` は手で書いた値である。ただし runner が EM を呼ばないことは、コードの監査で確認した。
  - Phase A の `git_head`（86fba01）は、AF4 のコードを commit する前のものである（Phase A を実行したコードは、その直後の `27452ad` で commit した）。
  - AF3 の CSV が無い場合、`af3_z5_misselected()` は空集合を返す（今回は存在する）。
- **AF1〜AF3 の成果物**：`git diff 86fba01` で変更なし。

## 9. 再現（repository root で実行）

```bash
python -m pytest -q expfam/src/experimental/test_af4_laplace_model_comparison.py
python expfam/src/experimental/run_af4_laplace_model_comparison.py --phase-a        --out-root <new_dir>
python expfam/src/experimental/run_af4_laplace_model_comparison.py --write-protocol --out-root <new_dir>
python expfam/src/experimental/run_af4_laplace_model_comparison.py --smoke          --out-root <new_dir>
python expfam/src/experimental/run_af4_laplace_model_comparison.py --execute        --out-root <new_dir>
python expfam/src/experimental/run_af4_laplace_model_comparison.py --analyze --stage full --out-root <new_dir>
```
