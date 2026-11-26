# 事前登録：同じ fit の上での C_Q と C_Lap の比較（本実行は未実施）

- 作成日：2026-11-26。branch `experiment/k-selection-exp2-generalization-20261125`。
- **探索的な追加解析である。** 8,000 fit と 600 fit の C_Q の結果を見た後に計画した。結果は confirmatory evidence として扱わない。
- 新しい MCEM・fit はしない。8,000 fit と 600 fit の結果は変更しない。
- **今回は技術検証と事前登録まで行った。本実行（6,600 評価）は開始していない。**
- 機械可読な正本は `expfam/results/k_selection/k_exp2_generalization_20261125/clap_comparison/protocol/protocol.json` である。
  - protocol.json の SHA-256：`ae62f0413ec0b2ad5196bdd945847808fe3defe2f687fcbcd0dc103050f53432`
  - manifest.csv の SHA-256：`e21bfad7…50fd`
  - 凍結時の git head：`5478d60`
- 実装系列は lineage E（prototype、Human Gate 未通過）である。
- C_Q は Q ベースの完全データ型（ICL 型）の基準で、**Schwarz BIC ではない**。
- C_Lap はラプラス近似による基準で、**厳密な周辺尤度ではない**。

## 1. C_Q と C_Lap の厳密な違い（実装と照合済み）

記号：
- ℓ(Z) = ln p(X | Z, θ̂) + ln p(Y | Z, θ̂)。正規化定数と階乗項を含む。
- 実装は `calc_Q_dual_strict_exp`。C_Lap の Φ も、同じ関数を 1 点の Z で評価している（`laplace_k_criterion.complete_log_density`）。

### 1.1 C_Q（`em_runner` + `calc_bic_exp`）

```
C_Q(K) = −2·(1/L) Σ_l ℓ(Z⁽ˡ⁾)  +  nK(1 + ln 2π)  +  p_K ln n
         └ E-step の MC サンプル    └ 事前密度の項 P_Z    └ p_K = Kd − K(K−1)/2 + d·[G-X] + [G-Y]
```

P_Z = nK(1 + ln 2π) になるのは、`scale_Z` で (1/L) Σ‖Z⁽ˡ⁾‖² = nK になり、かつ var_z = 1 だからである。

### 1.2 C_Lap（`laplace_k_criterion.py`）

```
Φ(Z)  = ln p(Z) + ℓ(Z),    Ẑ = argmax Φ（同時 mode、Newton 法）,    H = −∇²_Z Φ(Ẑ)（nK × nK）
ln p_Lap = Φ(Ẑ) + (nK/2) ln 2π − ½ ln|H|      （停留条件 max|∇Φ| ≤ 10⁻⁸ と H の正定値性を満たすときだけ）
C_Lap(K) = −2 ln p_Lap + d_K ln N,   d_K = Kd − K(K−1)/2,   N = n = 150
         = −2 ℓ(Ẑ)  +  ‖Ẑ‖²  +  ln|H|  +  d_K ln N        （var_z = 1 のとき厳密）
```

最後の行は、−2 ln p(Ẑ) = nK ln 2π + ‖Ẑ‖² と、ラプラス近似の体積項 −nK ln 2π が打ち消し合うことから出る。

### 1.3 差（恒等式）

```
C_Q − C_Lap = −2[ ℓ̄_MC − ℓ(Ẑ) ]  +  [ nK(1 + ln 2π) − ‖Ẑ‖² − ln|H| ]  +  (p_K − d_K) ln n
               (i) 評価点の違い          (ii)(iii) 事前密度の項と体積項       (iv) K によらない定数（Gaussian-X では d ln n）
```

ℓ̄_MC = (1/L) Σ_l ℓ(Z⁽ˡ⁾) である。

**「C_Q の事前密度の罰則 P_Z を、C_Lap の ln|H| に置き換えたもの」という説明は、厳密には成り立たない。** 違いは 4 つある。
1. **評価点**：C_Q は E-step の MC サンプル（L = 10、scale_Z 後）の平均で ℓ を評価する。C_Lap は同時 mode Ẑ（決定論的で、scale_Z しない）で評価する。ℓ̄_MC ≠ ℓ(Ẑ) で、この差は K によって変わる。
2. **事前密度の項**：C_Q は、データによらない nK(1 + ln 2π) である。C_Lap は、データに依存する ‖Ẑ‖² である（nK ln 2π は体積項と打ち消し合う）。
3. **体積項 ln|H|**：C_Lap にだけある。データと K に依存する。
4. **パラメータ罰則**：p_K と d_K の差は d·[G-X]（ここでは + [G-Y] は該当しない）。K によらないので、選択には影響しない。

- 既存の分解（`RESEARCH_MASTER.md` §20）とも整合する。H_Lap = (nK/2)(1 + ln 2π) − ½ ln|H| とおくと、次の式が厳密に成り立つ。

  C_Q − C_Lap = 2H_Lap + termA + 2R3（+ 定数の罰則差）

  - termA は、E-step のサンプルとラプラス近似の正規分布との差である。
  - R3 は、Φ が Z について 2 次式でない部分である。
  - したがって、「P_Z を ln|H| に置き換えたもの」と言えるのは termA + 2R3 を無視した近似の場合だけである。既存のデータでは、termA と R3 はどちらも K とともに大きくなり、打ち消し合っていた（K = 7 で約 −90 と約 +80）。
- `advisor_brief_after_convergence_20261126.md` §8.2 の「C_Q は ln|H| の代わりに P_Z を持つ」は、この近似の意味で読む（旧資料は変更しない）。
- **2 つの基準は同じ θ̂ を使うが、θ̂ はどちらの最大化点でもない。** θ̂ は MCEM（Q ベース）の推定値で、ln p_Lap を最大化した値ではない（Z9-U2）。

### 1.4 近似の限界（変わらない）

- ラプラス近似の誤差は評価できていない（既存の重点サンプリングによる補正は、重みが退化した。KI）。
- N = n は作業上の約束で、有効標本数を示したものではない。
- 同時 mode が一意であることは保証されない（§4 の技術検証を参照）。
- C_Lap の一致性は検討していない。

## 2. 研究上の意義（A-b-9 を中心に）

- C_Q は A-b-9 で、10・20・40 反復のどれでも 0/10 だった。10 反復の時点で C_Q の改善はほぼ止まっていたので、反復不足では説明できない。
- 項の分解（`cq_term_decomposition_20261126/`）では、K を 1 増やしたときの当てはまりの改善が、5→6 あたりで「K だけで決まるコスト」（大部分は P_Z の 425.68）を下回った。
- 残る問いは、**同じ θ̂ のまま基準だけを入れ替えたときに選択が変わるか**である。これは、過小選択が基準の作りによるものか、それともこの θ̂ のもとで情報が本当に足りないのかを区別する、最も直接的な比較である。
- **ただし、C_Lap が K_TRUE を多く選んでも、「C_Q の P_Z が過小選択の唯一の原因」とは言えない。**
  - §1.3 のとおり、2 つの基準は 4 点で違う。
  - C_Lap 自体も近似である。
- 対照として次の 2 つを見る。
  - O-b-9：C_Q は 10/10。C_Lap が正しい選択を保つかを確かめる。
  - regime a：C_Lap が過大選択しないかを確かめる。

## 3. 設計

| 項目 | 内容 |
|---|---|
| 対象 | (1) 8,000 fit のうち系列 O・A・B の 6,000 fit（10 反復目の状態）。(2) 収束感度の 600 fit の 40 反復目の状態。**計 6,600 評価、330 の（source, データ）ジョブ** |
| 除外 | 系列 C（Gaussian-Y は、検証済みの evaluator の対象外） |
| 状態 | 保存済み NPZ の θ̂（F、Σ、w0、w、var_z）と Z_last。SHA-256 で照合する |
| データ | data seed から再生成し、記録済みの x_hash・y_hash と照合する。不一致なら STOP |
| 同時 mode の開始点 | **Z_last のみ**（その状態の、最後の E-step サンプル）。Z_mean やその他の開始点は使わない |
| evaluator | `laplace_k_criterion.py`（v1、変更なし）。grad_tol 10⁻⁸、max_iter 200。H にリッジや jitter は足さない |
| C_Lap の罰則 | d_K ln N（d_K = Kd − K(K−1)/2、N = n = 150） |
| C_Q | 記録済みの値をそのまま使う（main は per_fit.csv の C_Q、conv は checkpoint_scores.csv の 40 反復目）。再計算・置換はしない |
| 選択規則（両基準で同じ） | データごとに、2 初期値の基準の非加重平均が最小の K を選ぶ。差が 10⁻⁹ 以内の同点は小さい K |
| 初期値ごとの選択 | その初期値の 10 評価がすべて OK のときだけ選ぶ |
| 並列 | 4 worker、BLAS は 1 スレッド。ジョブ（20 評価）が終わってから行を追記する |
| 再開 | 20 行がそろっているジョブは完了とみなし、再実行しない。worker 数は数値に影響しない |
| メモリ | 空きメモリが 1,500 MB を下回る状態が 2 回続いたら、新しいジョブの投入を止める。実行中のジョブは完了させ、STOPPED_MEMORY として記録する（再開可能） |

## 4. 評価指標（事前に固定）

出力は `analysis/` に置く。

| 指標 | ファイル |
|---|---|
| C_Q と C_Lap が選んだ K、生成時 K との一致・過小・過大、変化の有無と向き | `selection_by_dataset.csv` |
| セルごとの一致数（Clopper–Pearson 95% CI）、過小・過大・INCOMPLETE の数、選択が変化した件数（Lap > Q / Lap < Q）、C_Lap の初期値の不一致 | `summary_by_cell.csv` |
| K̂_Q × K̂_Lap のクロス表 | `paired_crosstab.csv` |
| K_TRUE ごとの基準曲線（データの中央値で crit(K) − crit(K_TRUE)） | `criterion_curves_median.csv` |
| K → K+1 の各項の増分（中央値）：C_Q（D、P_Z、PEN）と C_Lap（ℓ(Ẑ) の項、‖Ẑ‖²、ln\|H\|、罰則） | `term_steps_median.csv` |
| 注目セル（A-b-9、A-b-7、B-b-9、O-b-9）のデータごとの内訳：K_TRUE と K̂_Q の差を項別に | `focus_terms_by_dataset.csv` |
| C_Lap の数値失敗の件数（source × 系列 × regime × K × status） | `lap_status_counts.csv` |
| H の最小固有値、max\|∇Φ\|、Newton の反復数、Z_last からの移動量、計算時間、分解の閉包誤差 | `lap_diagnostics_by_cell_k.csv` |
| 図 | `fig_exact_by_criterion.png`、`fig_criterion_curves_focus.png`、`fig_term_steps_focus.png` |
| 検査 | `checks.json`：C_Q の選択が記録済みの解析と一致すること（不一致なら STOP）。K_TRUE = 3 の regime a・b（設計上同一）で C_Lap が同一であること |

**主要な問い**：A-b-9（10 反復と 40 反復の状態）で、同じ θ̂ のまま C_Q を C_Lap に変えると、10 データのうち何データで選択が変わるか。また、何データで K_TRUE を選ぶか（CI 付き）。

### 解釈の規則（事前に固定）

- C_Q と C_Lap の差は、§1.3 の 4 つの違いの**組合せ**に帰する。どれか 1 つの項を「原因」とは呼ばない。項別の増分は算術的な内訳として示す。
- C_Lap が K_TRUE を多く選んでも、次のようには書かない。
  - 「C_Q の P_Z が過小選択の唯一の原因」
  - 「C_Lap が正しい」
  - 「C_Lap は一致性を持つ」
- A-b-9 で C_Lap も K ≈ 5 を選んだ場合は、「この θ̂ のもとで情報が足りないことと整合する」とまでしか書かない。証明とはしない。
- INCOMPLETE は補完しない。分母は 10 のままとする。
- 結果を見た後に、閾値・開始点・N・罰則を変えない。

## 5. 失敗の扱い

| 状態 | 意味 | 扱い |
|---|---|---|
| OK | 停留条件を満たし、H が正定値 | C_Lap を記録する |
| NOT_STATIONARY | 200 反復で max\|∇Φ\| ≤ 10⁻⁸ に届かない | 値なしで 1 回だけ記録する |
| HESSIAN_NOT_PD | mode で H が正定値でない | 同上 |
| ERROR | evaluator 内の例外、または OK なのに値が有限でない | 同上（例外の要約を記録する） |

- **再試行・別の開始点・tolerance や max_iter の変更・リッジ・候補 K の除外は、いずれも行わない。**
- 1 つのデータの 20 評価のうち 1 つでも OK でなければ、そのデータの C_Lap の選択は INCOMPLETE とする。C_Q の選択は全データで報告する。
- STOP（実行全体を止める）条件：
  - state の SHA-256 の不一致
  - 再生成した X・Y の hash の不一致
  - NPZ のキー・形・非有限値、var_z ≠ 1
  - C_Lap の分解の閉包の失敗
  - コード・入力の SHA-256 の不一致
  - working tree が clean でない
  - C_Q の選択が記録と違う

## 6. 技術検証（本実験に使わないデータで）

- 出力：`clap_comparison/tech_check/tech_check.json`、`smoke_evaluations.csv`、`timing_memory.csv`。**11/11 項目に合格した。**
- **C_Lap を計算したのは、smoke 状態 15 個（O・A・B。data seed を 5,000,000 ずらした別データ）だけである。本実験のデータでは計算していない。**

| 確認 | 結果 |
|---|---|
| 保存 NPZ の完全性（本実験の 6,600 件。hash・キー・形・有限性・var_z = 1 だけを確認し、スコアは計算していない） | 合格（不一致 0） |
| X・Y の再生成（本実験の 300 データ）と記録済み hash の一致、Y の対称性・対角 0 | 合格 |
| モデルの復元：復元したモデルの対数尤度と、独立に書いた式の値の一致 | 合格（相対誤差 < 10⁻⁹） |
| evaluator の適用：勾配と Hessian の有限差分チェック | 合格（相対誤差の最大：勾配 8.6×10⁻⁸、Hessian 5.0×10⁻¹⁰） |
| smoke 状態 15 個の評価（Z_last から） | 15/15 OK |
| 決定論性（同じ評価を 2 回） | ビット単位で一致 |
| 並列の安全性（4 worker と逐次） | ビット単位で一致 |
| 失敗処理：max_iter = 0 を強制 | NOT_STATIONARY として値なしで記録された |
| 失敗処理：θ に NaN を入れる | ERROR（LinAlgError）として値なしで記録された |

### 実測した時間とメモリ（n = 150、1 評価、1 プロセス）

| K | 件数 | 中央値 | 最大 | ピークメモリ（working set） |
|---|---|---|---|---|
| 1 | 6 | 0.02 s | 0.05 s | 98 MB |
| 9 | 9 | 1.07 s | 1.74 s | 143 MB |
| 10 | 9 | 1.46 s | 2.96 s | 153 MB |

- K = 9 は、K = 10 の smoke 状態の先頭 9 列を使った**時間計測用の代理**である（推定された状態ではない）。
- 見積もり：
  - K³ に比例するとして補間すると、1 データ・1 初期値の K = 1〜10 で約 4.5 s、1 ジョブで約 9 s。
  - 330 ジョブを逐次で約 50 分、最大値で見ると約 100 分。
  - **4 worker で約 15〜30 分**。
  - メモリは worker 4 つで約 0.6 GB、親プロセスを含めて 1 GB 未満。
- 計測時の空きメモリは約 6.7〜6.9 GB（全体 16 GB、物理 6 コア）。
- **8 並列は使わない。** 4 worker でも 30 分程度で終わる見込みで、前回メモリが逼迫したときのように他のプロセスと競合する余地を残すためである。

### 技術検証で見つかった注意点（科学的な BLOCKER ではない）

- **過剰な次元の状態では、H が特異に近い。**
  - K_TRUE = 1 のデータで K = 10 を当てはめた状態では、H の最小固有値が 0.007〜0.014 だった。
  - そのような状態の 1 つ（O-a-KT1、K = 10）では、Z_mean から始めると別の mode に収束し、C_Lap が 1.45 違った。
  - 別の状態（B-a-KT1、K = 10）では、Z_mean から始めると HESSIAN_NOT_PD になった（Z_last からは OK）。
- K_TRUE = 9 の状態（K = 10）では、2 つの開始点の差は 10⁻⁶ 以下だった。
- 予想：
  - 本実行では、K_TRUE の小さいセルの大きい候補 K で、NOT_STATIONARY・HESSIAN_NOT_PD が出る可能性がある（既存の実験での率は 0.1〜0.5%）。
  - その場合、規則によりそのデータの C_Lap は INCOMPLETE になる。開始点の依存は、Z_last だけを使う規則で固定する。
  - 特異に近い H では、ラプラス近似そのものの精度が落ちる。過大選択の側の解釈に注意が必要である。

## 7. 実行してよいか

- **技術的には実行できる。** 入力の完全性、再現性、決定論性、並列の安全性、失敗処理はすべて確認した。計算資源も十分である（4 worker、1 GB 未満、30 分程度）。
- **科学的な BLOCKER はない。** ただし次の限界は、結果の解釈で必ず付ける。
  - 結果を見た後の探索的な解析である
  - C_Lap は近似である
  - θ̂ はどちらの基準の最大化点でもない
  - 系列 C を除外している
  - 過剰な次元では H が特異に近い
  - lineage E は prototype である
- **本実行は新しい研究実験の開始にあたる（Human Gate）。研究者の承認を受けてから行う。**
- 承認後のコマンド：

  ```
  python tools/research_audit/run_k_exp2_clap_comparison.py --execute expfam/results/k_selection/k_exp2_generalization_20261125/clap_comparison/run --protocol expfam/results/k_selection/k_exp2_generalization_20261125/clap_comparison/protocol/protocol.json
  python tools/research_audit/run_k_exp2_clap_comparison.py --analyze expfam/results/k_selection/k_exp2_generalization_20261125/clap_comparison/run --out expfam/results/k_selection/k_exp2_generalization_20261125/clap_comparison/analysis
  ```

- 改行コードについての注意：code の SHA-256 は、作業ツリーのバイト列（LF）で計算した。別の autocrlf 設定で checkout し直すと一致しない場合がある（`independent_audit.md` §2 と同じ）。
