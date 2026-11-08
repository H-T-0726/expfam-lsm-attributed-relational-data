# 02 選択の結果（2026-11-09）

## 前提

- **系列**：E（prototype）。
- **データ**：K_GEN ごとに 1 データ（`G_paper_text`）。各データに 10 試行（初期値と MC 系列）× K 1〜10。合計 500 fit。
- **一次表**：`expfam/results/k_selection/paper_fig3_condition_criteria_20261109/` の次のファイル。
  - `selection_per_trial.csv`
  - `selection_summary.csv`
  - `selection_aggregated.csv`
  - `pairwise_agreement.csv`
  - `figures/fig1_khat_distribution.png`
  - `figures/fig2_score_curves.png`
- **正解の扱い**：「正解」は K_HAT = K_GEN を指す。K_GEN = K0 は仮定しない。
- **試行の独立性**：同じ K_GEN の 10 試行は**同じデータ**を使い、初期値と MC 系列だけが違う。データについての一般化はできない（各 K_GEN に 1 データ）。

## 1. technical validity

| 確認 | 結果 |
|---|---|
| fit | **500/500 OK、FAILED 0**。再試行・seed の変更・反復の追加はない。5 プロセスが並列に 1 回ずつ起動した（`run_log_K*.txt`。head `16e4deb`、protocol `2c61b259…`） |
| 恒等式の残差 | $C_Q-C_{\rm Lap}=2H+\text{termA}+2R_3$、$C_{\rm ELBO}-C_{\rm Lap}=2R_3$ とも、≤ 2.3e−11 |
| Laplace の失敗 | **8/500 状態**（`NOT_STATIONARY` 7、`HESSIAN_NOT_PD` 1）。内訳：K_GEN=1 で K=8 が 1、K_GEN=3 で K=2・7・8 が各 1、K=10 が 2（うち 1 は HESSIAN_NOT_PD）、K_GEN=5 で K=10 が 1、K_GEN=9 で K=10 が 1（`incomplete_and_invalid_runs.csv`）。該当する試行は、C_Lap・C_ELBO・cond_mode で INCOMPLETE とした |
| C_ELBO の `MC_UNCERTAIN` | 2 試行（K_GEN=3 の t7、K_GEN=9 の t4） |
| 収束（記述） | 最終反復での \|ΔQ_strict\| の中央値は、大きい K で 90〜240（K_GEN=7・9 の K≥7）。**10 反復では未収束**の fit が多い（`convergence_diagnostic.csv`） |

## 2. 試行ごとの K_HAT（10 試行。正解/過小/過大/INCOMPLETE）

| K_GEN | C_Q | C_Lap | C_ELBO | cond_MC | cond_last | cond_mode |
|---|---|---|---|---|---|---|
| 1 | **10/0/0/0** | 9/0/0/1 | 9/0/0/1 | 0/0/10/0（8〜10） | 0/0/10/0 | 0/0/9/1 |
| 3 | 7/0/3/0 | 5/0/2/3 | 6/0/1/3 | 0/0/10/0 | 0/0/10/0 | 0/0/7/3 |
| 5 | 6/0/4/0 | 3/0/6/1 | 4/0/5/1 | 0/0/10/0 | 0/0/10/0 | 0/0/9/1 |
| 7 | 5/0/5/0 | 5/0/5/0 | 5/0/5/0 | 0/0/10/0 | 0/0/10/0 | 0/0/10/0 |
| 9 | 1/0/9/0 | 2/0/7/1 | 2/0/7/1 | 0/0/10/0 | 0/0/10/0 | 0/0/9/1 |

**試行ごとの値**（`selection_summary.csv` の `k_hat_by_trial`）
- C_Q：K_GEN=3 で 3 3 3 3 4 4 3 3 3 4、K_GEN=5 で 5 6 6 5 6 5 5 5 6 5、K_GEN=7 で 8 7 7 7 8 7 8 8 7 8、K_GEN=9 で 10 10 10 10 10 9 10 10 10 10。
- **過小選択は、どの基準でも 0 件だった。C_Q・C_Lap・C_ELBO の誤りは、すべて K_GEN+1 への過大選択だった**（K_GEN=9 では 10、つまり候補の上限）。条件付き尤度型は K=8〜10 を選んだ。
- **主規則（試行ごと）の合計**：K_GEN を選んだ試行は、C_Q 29/50、C_Lap 24/50（完全な 44 試行中）、C_ELBO 26/50（同 44 試行中）。K_GEN=9 では、C_Q の正解は 1/10 だった。
- 条件付き尤度型は、全試行で過大選択した（ほとんどが K=9〜10）。

## 3. 試行をまとめたときの選択（`selection_aggregated.csv`）

| K_GEN | C_Q：平均 / 最小 | C_Lap：平均 / 最小 | C_ELBO：平均 / 最小 | cond_MC：平均 / 最小 |
|---|---|---|---|---|
| 1 | 1 / 1 | INCOMPLETE / 1 | INCOMPLETE / 1 | 10 / 10 |
| 3 | 3 / 3 | INCOMPLETE / 3 | INCOMPLETE / 3 | 10 / 10 |
| 5 | 5 / 5 | INCOMPLETE / 5 | INCOMPLETE / 5 | 10 / 10 |
| 7 | **8** / 7 | 8 / 7 | 8 / 7 | 10 / 10 |
| 9 | **10** / 9 | INCOMPLETE / 9 | INCOMPLETE / 9 | 10 / 10 |

**読み方**
- 「最小」は、各 K で 10 試行のうち最も良い（最小の）値を使う、事前登録した**参考の副次規則**である。
  - Python 再現は、対数尤度が最良の試行を選んでから BIC を計算した。本規則は、基準ごとに最良の試行を選ぶので、C_Q・C_Lap・C_ELBO が別の fit を使うことがある。C_Q 以外では、両者は同じではない。
- **副次（参考）の観測**：「最小」の規則では、C_Q・C_Lap・C_ELBO のいずれも、K_GEN = 1・3・5・7・9 を選んだ。**原論文 Fig. 3 と一致するのは argmin だけ**である。
  - 各 K_GEN について、1 データ・10 試行からの 1 回の結果である。
  - 余裕（min C(K_GEN) − min C(K_GEN+1)）は −100〜−344 だった。これは、K_GEN ≥ 3 での C(K_GEN) の試行間の SD（240〜480）より小さい。
  - 原論文の過大側の余裕は 4〜21 で、本研究の数十〜数百とは大きさが違う。
  - **主規則の結果（上の表）が、本研究の主な結論である。**
- 「平均」の規則では、C_Q は K_GEN=7・9 で過大選択した。C_Lap・C_ELBO は、Laplace の失敗による INCOMPLETE が多かった。
- **事前登録で主とした「試行ごと」の規則では、K_GEN ≥ 3 で過大選択が多かった。**

## 4. 基準の間の一致（試行単位、全 K_GEN、`pairwise_agreement.csv`）

| 組 | 両方完全 | 一致 | 前者が小さい | 前者が大きい |
|---|---:|---:|---:|---:|
| C_Q vs C_Lap | 44 | 40 | 3 | 1 |
| C_Lap vs C_ELBO | 44 | 42 | 0 | 2 |
| C_Q vs cond_MC | 50 | 9 | 41 | 0 |
| C_Q vs cond_mode | 44 | 8 | 36 | 0 |

- **この強信号の条件では、C_Q・C_Lap・C_ELBO の選択はほとんど同じだった**（44 試行中 40〜42 で一致）。
- PR #164 で観測された「C_Q ≤ C_Lap」も、ここでは 1 件の例外があった（C_Q のほうが大きい）。
- 基準の間の差より、試行の間の差（初期値と MCEM の経路）のほうが大きかった。これは**事後的・探索的な診断**（03 §2）による観測で、原因は分離していない。
