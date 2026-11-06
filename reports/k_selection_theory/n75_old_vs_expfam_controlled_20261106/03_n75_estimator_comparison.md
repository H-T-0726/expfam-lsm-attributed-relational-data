# 03 n=75 での旧 Python 推定器と系列 E の比較（Phase B、2026-11-06）

**ラベル**：OLD は 0.5 系列（KI-001：Newton 方向が全体として正しいとは断定できない）、E は系列 E（prototype）。2 つの系列を同じ表に並べるのは、推定器の比較という設計による例外（KI-002）で、行ごとに系列を明記する。

**出所**：`expfam/results/k_selection/n75_old_vs_expfam_controlled_20261106/` の `selection_summary.csv`、`full_k_selection.csv`、`selection_margins.csv`、`estimator_comparison.csv`、`parameter_recovery.csv`、`figures/fig1`〜`fig3`。

**公平性**：`ONE_FACTOR_COMPARISON`（推定器の全体）、`MATCHED_CRITERION`（`fairness_matrix.csv`）。次の項目が同じ：データ、n、d、K_GEN、候補 K、初期状態（ハッシュ一致）、予算、マスク（全ペア）、基準、集計。

## 1. 実行の健全性

- 新しい fit：B 160 件、C 80 件、D 80 件。すべて OK、FAILED 0、再試行 0。予備の 20 件は使っていない。
- `run_log.txt` に記録したハッシュは、`run_n75.py`・`run_fair_comparison.py`・`n75_protocol.py` の 3 ファイルで、3 ステージとも同じ。B と C・D の間で git head が違う（ecf2e2a・729a04f）。
  - 差分は解析スクリプト 3 本の新規追加だけ（`git diff --stat ecf2e2a 729a04f`）。
  - `git diff --quiet ecf2e2a 729a04f -- expfam/src reproduction/src tools/research_audit/{fair_protocol,run_fair_comparison,run_n75,n75_protocol}.py` は差分なし（監査の後に確認）。
- 共通評価器での採点は 480 状態（新規 320 ＋ PR #160 の n=150 の 160）。
  - Laplace の NOT_STATIONARY は 3 状態。補完せず INCOMPLETE とした。
  - PR #160 の状態の再採点は、PR #160 の値と最大 3.6e−12 で一致（`reuse_crosscheck.csv`）。
- 解析スクリプトは 1 回目に pandas の名前の衝突（`rec.at`）で途中停止した。修正し、途中まで書かれた出力（commit していない）を削除して再実行した。修正は 1 行（`rec["at"]`）だけで、選択の規則・数値の計算は変えていない（commit 42b5e50）。

## 2. K の選択（G_canon、n=75、全ペア、データ 4 つ）

| 基準 | OLD の K_HAT | E の K_HAT | OLD と E の一致 |
|---|---|---|---|
| A C_Q | 5 5 5 5（**4/4 正解**） | 5 5 5 5（**4/4 正解**） | 4/4 |
| E C_Lap | 5 5 5 INC | 5 5 5 5 | 3/3（完全なもの） |
| B C_Q − P_Z | 7 7 5 10 | 10 10 10 10 | 1/4 |
| C 最後の標本での plug-in | 7 5 8 9 | 10 10 10 10 | 0/4 |
| D 結合モードでの plug-in | 10 10 10 INC | 10 10 10 10 | 3/3 |
| Python 型（OLD のみ、参考） | 5 5 5 5 | — | — |

**余裕**（`selection_summary.csv`）

| 推定器 | C_Q 過小側 C(4)−C(5) | C_Q 過大側 C(5)−C(6) |
|---|---|---|
| OLD | 771、675、590、485 | 中央値 −194 |
| E | 769、666、607、536 | 中央値 −158 |

C_Lap の過小側は中央値で OLD 496・E 501、過大側は −50・−58。

- **OBSERVED**：n=75 のこの条件では、**OLD も E も C_Q で K_GEN=5 を全データで選んだ**。過小選択は起きなかった。K ごとのスコア曲線もほぼ重なる（`fig1`）。
- **OBSERVED**：Q_Z を含まない B・C と、D は過大選択する。E は常に上限の K=10 を選ぶ。OLD は 5〜10 にばらつく（推定器と基準の交互作用。PR #160 と同じ傾向）。
- **注**：`selection_summary.csv` と `full_k_selection.csv` には、E の行に PY_type（OLD だけの参考規則）が INCOMPLETE として現れる。これは定義上の空欄で、欠損した fit ではない。

## 3. 推定量（K=5、2 start の平均）

| データ | 真の w | ŵ OLD | ŵ E | MC 標本の分散（tr、OLD / E） |
|---|---:|---:|---:|---|
| rep 1 | 1.162 | 1.054 | 1.164 | 0.130 / 0.114 |
| rep 2 | 1.162 | 1.069 | 1.192 | 0.142 / 0.121 |
| rep 3 | 1.162 | 1.014 | 1.106 | 0.151 / 0.125 |
| rep 4 | 1.162 | 0.999 | 1.107 | 0.148 / 0.129 |

- OLD の ŵ は 4/4 で E より小さく、真値より小さい。E の ŵ は真値に近い（OBSERVED）。
- 保存された MC 標本の分散は、4/4 で OLD のほうが大きい（OLD の事後分布が広いことと整合。06）。
- 回復（K=5 の中央値）

| 指標 | OLD | E |
|---|---:|---:|
| ZZᵀ の相関 | 0.959 | 0.966 |
| 部分空間の回復 | 0.964 | 0.968 |
| η_Y の相対 RMSE | 0.309 | 0.265 |
| FFᵀ の相対誤差 | 0.196 | 0.190 |
| 母集団の Y 対数スコア | −0.4229 | −0.4176（真値では −0.3957） |

  E のほうがわずかに良いが、差は小さい。

## 4. 答え（Q1・Q2）

- **Q1**：同じ n=75・Gaussian-X/Bernoulli-Y のデータで、C_Q による K_HAT は **OLD と E で 4/4 一致**した（G_old を含めると 6/6。04）。
- **Q2**：OLD は n=75 で過小選択しなかった。**E も過小選択しなかった**。この条件は、PR #160 で過小選択が起きた n=75 の条件（E2）と同じではない（07・08）。したがって、「n=75 の失敗条件で OLD は成功するか」という問いには、**この実験では答えられていない**。答えられたのは、「n=75 でも、この信号の条件では両推定器とも成功し、結果は一致する」ということである。
