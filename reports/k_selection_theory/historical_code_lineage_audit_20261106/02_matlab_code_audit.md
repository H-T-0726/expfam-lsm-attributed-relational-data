# 02 旧 MATLAB コード（`Mato Lab Program/`）の監査（2026-11-06）

- MATLAB は実行していない。静的な追跡と、式を Python に写した有限差分の確認（`estep_finite_difference_check.csv`）だけである。
- 由来：12 ファイルとも、リポジトリの最初の一括 commit `7c3040f`（2026-04-27）で追加されたもので、それ以前の履歴はない。**commit author からは作成者を特定できない。** ファイル名の「(1)」はダウンロード複製の可能性を示すだけである（`code_lineage_inventory.csv`）。`NOLTA_exp_ver5 (1).pdf` は MATLAB live script の出力で、PDF の作成日は 2025-08-04（メタデータ）。

## 1. どの K を推定していたか（Phase 2-A）

`NOLTA_exp_ver3_revise_batch.m`

```matlab
12 kT = 3;
14 [X, Y, trueParams] = setTrueParams(150, 15, kT);
19 L = 11;
30 for k = 1 : 10
32     params = setParamsDesc(trueParams, kT, i);
33     [estParams(m), history] = calcdescmetric_ver4(X, Y, kT, params, trueParams);
```

- FACT：ループ変数 `k` は、推定次元にもファイル名以外にも使われない。`setParamsDesc(tparams, k, L)` は Z・F の初期値を **kT=3 列**で作り（setParamsDesc.m:9-10）、`calcdescmetric_ver4` の第 3 引数 `i` は `m = i`（L10）に代入されるだけで使われない。次元は `params.Z` の列数で決まる。
- **したがってこのスクリプトは、K=3 のモデルを 10 回（初期値だけを変えて）推定する。** 冒頭のコメントも「k は固定したもとで，L を変化させてみる」で、L=11 は論文の L=10 と違う。
- FACT：BIC はどこでも計算されていない。`DecideNumFactor.m` は X だけの因子分析（`factoran`）による BIC で、リポジトリのどのスクリプトからも呼ばれていない。そのスケールは `-2 * n * loglike`（L14）で、factoran の `loglike` の定義によっては n 倍が重複する（UNRESOLVED）。
- **結論：リポジトリの MATLAB コードには、論文 Experiment 2（k = 1..10 の BIC 比較）を実行する経路がない。** 原論文 Fig. 3 の生成元は特定できない。

## 2. 生成データ（Phase 2-C）

`setTrueParams.m` と論文 §4.1 の違い（`paper_vs_matlab.csv`）：

| 項目 | 論文 | MATLAB |
|---|---|---|
| F | 各成分 N(0, 10) | N(0, 5) を引き、**各行のノルムを √0.9 に固定**（L23, 26-28） |
| Z | N(0, I) | N(0, I) の後、列ごとに z-score（L22） |
| X | N(F z, 0.1 I) | 生成後、列ごとに z-score（L40） |
| w0, w | random | randn、3·randn（L30-31） |
| 乱数 | 記載なし | 関数内で `rng(1980)`（L12）。呼ぶたびに同じ乱数列 |

- DERIVED：行ノルムを固定すると tr(FᵀF) = 0.9 d = 13.5 で、**K によらず一定**になる。1 方向あたりの平均エネルギーは 13.5/K（K=9 で 1.5）。論文の記載どおり N(0, 10) なら E[FᵀF] = 10d I = 150 I で、1 方向あたり約 100 倍大きい（`signal_strength_summary.csv`：論文記載の生成器の 1 方向あたり X の Fisher 情報は中央値 1,300〜1,800、MATLAB 型は 15〜135）。
- X の z-score は列ごとの尺度変換なので、Gaussian-X の Fisher 情報 FᵀΣ⁻¹F は変わらない（DERIVED）。Z の z-score は各列の標本分散を 1 に固定する。

## 3. E-step の数学的整合性（Phase 2-D）

`calcEtaNewton.m`（`implementation_discrepancies.csv` D3〜D6）

| 項目 | コード | 独立導出（採用式） | 判定 |
|---|---|---|---|
| Y 項の勾配 | `(Y(ind,:) - S) * Z - (自己項)`（L47）：**w がない**、1/2 なし | w Σ_{j≠i}(y_ij − s_ij) z_j | w が欠けている（既知：`docs/math_notes/half_factor_literature_code_check.md` §2-B） |
| 精度行列の Y 項 | w² Σ_j s(1−s) z_j z_jᵀ（L58-60）、1/2 なし | 同じ | 一致 |
| 自己項の除外 | `sig(ind,ind)*(1-sig(ind,ind))*w^2`（L61）。`sig` はすでに w²s(1−s) | w² s_ii(1−s_ii) z_i z_iᵀ を引く | 余分な因子（既知 §2-A） |
| Newton の刻み | α = 0.01、最大 9 ステップ（L3, 8, 15） | 論文は刻みの記載なし | mode まで約 9% しか動かない |
| 収束判定 | `diff` を節点ごとにリセットしない（L5, 9, 20） | — | ある節点が収束すると、以降の節点は Newton を飛ばし、前の節点の `invAi` を使う（潜在的な欠陥） |
| Laplace 共分散 | `mvnrnd(Z(i,:), invAi)`（L23） | N(η_i, A_i⁻¹) | 平均は mode ではなく「わずかに動いた現在値」 |

- OBSERVED（有限差分、n=12, k=3, d=5, w=1.7）：Eq.(20) の勾配に対する相対誤差は、採用式 2e−11、論文式（1/2）0.135、MATLAB 0.111。精度行列の相対誤差は、それぞれ 2e−6、0.066、0.002。MATLAB の Y 勾配成分の大きさは正しい値の 1/w 倍（この例で 0.588 = 1/1.7）。
- **評価**：α=0.01 の 9 ステップでは mode に届かないので、MATLAB の E-step は「現在の z_i を中心に、精度 A_i の正規分布から引き直す」ランダムウォークに近い（受理・棄却なし）。これは事後分布からの標本ではない（DERIVED）。ただし、これらの欠陥が論文の結果に影響したかどうかは、Experiment 2 の実行経路が不明なので **判断できない（UNRESOLVED）**。

## 4. その他（FACT）

- `calcw.m`・`calcw0.m`：Adam の 1 次モーメントの更新が `m = β1 m + (1 − β1^t) g`（標準は 1 − β1）。学習率 0.01、最大 49 ステップ。
- `calcdescmetric_ver4.m`：σ_z² は 1 に固定し、`scaleZ` で平均 ‖z_i‖² = k に再スケールする（L29, 65-79）。論文 Eq.(14) の σ_z² 更新とは違う。Q は p_X + p_Y + p_Z の L 平均（L37-47）で、Eq.(18) と同じ量（Q2 の MC 平均）。
- live script（`NOLTA_exp_ver5 (1).pdf`）は Experiment 3（d を変える、k=3 固定）の実行記録で、d=15 の最終 Q/L は約 −6,456。d=25 の実行では 10 反復の間 Q がほとんど変わっていない（−12,950 前後）。

## 5. この節の答え（Q2）

- リポジトリの MATLAB は、どの実行でも **K=3（kT）のモデルだけを推定している**（FACT）。K を変えて BIC を比べるコードは存在しない。
- MATLAB の生成器は、論文本文の生成器と F の分布・正規化が違う（FACT）。
- MATLAB の E-step には、w の欠落、自己項の除外の誤り、極端に小さい Newton 刻みがある（FACT）。結果への影響は UNRESOLVED。
