# 01 研究問いと事前登録（2026-11-06、新しい fit の前に commit）

- 位置づけ：系列 E は prototype（本文採用不可）。既存の Draft PR #155〜#159 の結果は main に採用された結果ではない。
- 機械可読の正本：`tools/research_audit/fair_protocol.py`（`protocol()` とそのハッシュ）。この文書はその要約である。
- C_Q は Q ベースの complete-data / ICL 型の基準で、Schwarz BIC ではない。C_Lap はラプラス近似で、厳密な周辺尤度ではない。
- 記号：K_GEN（生成に使った次元）、K0（canonical モデルで定義が成り立つ場合の母集団の最小次元。`reports/identifiability/true_k_identifiability_hardened_20260904.md` §2 の K\*）、K_CAND（候補）、K_HAT（選ばれた次元）。**K_GEN = K0 は仮定しない。**

## 1. 研究問い

依頼文の Q1〜Q10 をそのまま扱う。証拠が足りない問いは UNRESOLVED とする。

## 2. 実験（承認済みの範囲。新しい MCEM fit は最大 900）

| 実験 | 目的 | 新しい fit | 再利用 |
|---|---|---:|---|
| E1 | 同じデータ・同じ初期状態・同じ乱数 seed で、旧 Python 推定器と系列 E を比べる（Gaussian-X/Bernoulli-Y、全ペア観測） | 320 | なし（#139 は初期値が共有されていないので参照扱い） |
| E2 | 推定器と基準を固定し、X の分布族・n・信号を変える | 280 | Poisson の弱 n=75・150、強 n=75 は PR #155 の状態（照合ゲートつき） |
| E3 | 同じデータで、関係ペアの観測割合だけを変える | 56 | 100% 条件は PR #155 A の状態 |
| 照合ゲート | 再利用する状態の再現確認 | 3 | 予備枠 20 から |
| 合計 | | **659** | |

### 依頼からの変更（結果を見る前に決定）

- **E1 のデータ数**：依頼どおりにすると 2 推定器 × 2 生成器 × 2 K_GEN × 4 データ × 10 K × 2 初期値 = 640 fit で、上限 320 を超える。そこで**データ数を各条件 2 にし（rep 1, 2）、それ以外の要因はすべて残した。** 推定器・生成器・K_GEN の比較を同じ初期状態で行うことを、データ数より優先した。
- **E2 の再利用**：PR #155 の Poisson の状態は、同じ生成器・seed 規則・推定器・L・反復・start・マスクで作られている。再利用する前に 3 fit を再推定し、状態が一致すること（rtol 1e-9、atol 1e-6）を確かめる。一致しなければ再利用せず、停止して記録する。

## 3. 各実験の固定条件

共通：L=5、EM 8 反復、Newton 10 ステップ・刻み 0.5、Adam 50 ステップ、start 2 つ、1 プロセス・BLAS スレッド 1、空きメモリが 3 GB 未満なら新しい fit を始めない、fit ごとに原子的に保存して再開可能にする、失敗した fit は再試行しない。

**E1**
- n=150、d=15、K_GEN ∈ {5, 9}、K_CAND = 1..10、データ rep 1, 2。
- 生成器
  - G_old：MATLAB 移植。Python 再現・#139 の rep 1, 2 と同じデータ。
  - G_canon：Z を正規化しない、F の平均行ノルム² 0.9、Σ_X = 0.1 I、w0 = −1、w = √(6.75/K_GEN)。
- 推定器
  - OLD：`reproduction/src/model.py` を adapter から呼ぶ。コードは変更しない。
  - E：`em_runner_instrumented`、consistent。
- 両推定器に同じ model seed を与える。初期の Z・F・Σ は同じ `initialize_params(seed)` で作るので一致する。w0 = logit（上三角の辺密度）、w = 0.5。
- 旧 Python 再現は w0 の初期値に n×n 全体（対角を含む）の平均を使っていた。adapter はこれを系列 E の定義にそろえる（記録済みの変更）。

**E2**
- K_GEN=5、d=15、n ∈ {75, 150}、K_CAND = 1..7、データ rep 1..4、学習用ペアは 80%（PR #155 と同じ）。
- 信号
  - 弱：PR #155 A（F の平均行ノルム² 0.5、w² 0.6）。
  - 強：PR #155 D（0.9、1.08。w0 は母集団の辺密度が同じになるよう較正）。
  - **F と w を同時に変えるので、F と w の単独の効果ではない。** 単独の効果は PR #155 の B・C を参照する。
- Gaussian-X
  - 同じ Z・F・Y・マスクのもとで、X_G = Z Fᵀ + ε、ε_il ~ N(0, s_l²)。
  - s_l² = exp(−‖f_l‖²/2) とすることで、条件付き Fisher 情報の期待値を Poisson-X と列ごとに一致させる。
  - support・分散関数・曲率が違うので、**分布族の比較は条件群の比較**とする。

**E3**
- PR #155 A（Poisson-X、弱、n=75）の rep 1..4。
- PR #155 の test マスク（全ペアの 20%）を両条件から除外し、予測の評価に使う。
- 学習に使えるペアの 100%（PR #155 A の状態を再利用）と 80%（乱数で間引く。新しい fit）を比べる。マスク以外は変えない。

## 4. 指標

- **主要指標**
  - 全候補 K からの K_HAT
  - 正解・過小・過大の件数
  - C(K_GEN−1) − C(K_GEN) の余裕
  - C(K_GEN) − C(K_GEN+1) の余裕
- **基準**（同じ状態に適用）：A C_Q、B C_Q − P_Z、C 最後の標本での plug-in、D 結合モードでの plug-in、E C_Lap。E1 の OLD には、Python 再現の規則（最良反復の plug-in Q2 ＋ Eq.(26) の罰則）を参考として追加する。
- **副次指標**
  - X の再構成
  - held-out の Y の対数スコア（E2・E3）
  - FFᵀ の回復（canonical の真値）
  - η_Y の回復
  - 数値的な失敗
  - 計算時間
- **Laplace の近似精度**：既存の状態（canonical A1、n ∈ {75, 150}、rep 1..8、K ∈ {4, 5, 6}、start 1）で、importance sampling による補正 c_K を計算する。8 バッチ × 512 標本。重みが退化した場合（ESS 割合 < 0.01）は、真の周辺尤度として扱わない。

## 5. 停止条件

停止するのは次の場合だけ：
- 比較を成り立たなくする数式の不一致
- データの漏洩
- seed の規則違反
- CSV の破損
- 生成器の暗黙のクリッピング
- 条件を固定できない
- 状態を安全に扱えない
- 主目的を解釈できなくする技術的な欠陥

K 選択の失敗、精度の悪化、solver の警告、MC の変動では停止しない。

## 6. 公平性の分類

比較ごとに次のいずれかのラベルを付ける（`comparison_fairness_matrix.csv`）：`MATCHED_MODEL`、`MATCHED_ESTIMATOR`、`MATCHED_CRITERION`、`ONE_FACTOR_COMPARISON`、`MULTI_FACTOR_COMPARISON`、`HISTORICAL_REFERENCE`、`NOT_DIRECTLY_COMPARABLE`。
