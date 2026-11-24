# 既存の K 選択研究の一次確認

- 作成日：2026-11-25。基点 `origin/main` = `1b5030b`。
- 「main」は `origin/main` を指す。main 以外の branch にある結果はそのことを明記する。
- 数値は canonical docs（`RESEARCH_MASTER.md` の Claim Ledger）と、そこから参照される一次 artifact に基づく。
- **基準の呼称**：C_Q（`calc_bic_dual`）は Schwarz BIC ではなく、Q ベースの完全データ型（ICL 型）の基準である（KI-010、§12.6・§17.4）。C_Lap はラプラス近似による基準で、厳密な周辺尤度ではない。

## 1. 一覧

| 研究 | 系列 | 生成器 | 条件 | 基準 | 主な結果 | 場所 |
|---|---|---|---|---|---|---|
| fixed official Exp1（K≤6）＋K9 拡張 | **C（fixed）** | historical（`generate_dual_data`） | K_TRUE=3 のみ、候補 1〜9、n=150, d=15、L=5、8 反復、3 シナリオ（P/B、G/P、B/G）× 10 試行（**試行ごとに別データ**、seed_data = base + 100·trial） | C_Q | 30/30 で K=3 | main（`expfam/results/fixed_official/exp1/`、`exp1_k9/`） |
| Phase 7e held-out pilot | E | historical 系 | P/B、K_TRUE=3 | held-out 予測スコア | K=3, 3, 5 | main |
| Phase 8b K_TRUE robustness | E | canonical 前の系 | P/B、n=75、K_TRUE {1,2,4,5}、候補 1〜7、2 start | held-out 予測スコア | 新規グリッド 18/24（K_TRUE=5 は 2/6） | main（§12.9） |
| clean true-K n-sweep | E | **canonical-clean-v1**（matched：平均行ノルム² 0.5、w = √(3/K)、w0 = −1） | P/B、K_TRUE {1,3,5}、n {50,75,100,150}、候補 1〜7 | S1 held-out、S2 C_Q、S3 条件付き plug-in | K_TRUE=5：S2 は 0/8→0/8→1/8→7/8。誤りはほぼ過小選択。S3 は 61/64 で過大選択 | main（§17.5） |
| Phase 9（9E〜9Y） | E | canonical 系、列ごとの family を含む | n=75、d=12 など | C_Q、C_Lap | matched K_true（1〜4）：C_Lap 10/10/10/8、C_Q 10/9/7/1 | main（§19）。**9K・9D は列ごとの family（G3/B6/P3）で正式モデルの範囲外** |
| EM 収束感度・checkpoint 再評価 | E | canonical-clean-v1 | P/B、K_TRUE=5、n {50..150}、反復 8〜40 | C_Q、C_Lap | C_Q の過小選択は反復 40 まで持続 | main（§20、PR #150） |
| d-sweep | E | canonical-clean-v1 | P/B、n=75、K_TRUE=5、d {15,30,60} | C_Q、C_Lap | C_Q 正解 0→3→5 /8 | main（§21、PR #154） |
| R1/R2 | E | canonical-clean-v1 | n×d と推定条件 | C_Q、C_Lap | 事前登録の仮説 6 つを支持 | main（§22、PR #153） |
| **原論文 Fig.3 条件の基準比較**（2026-11-09） | E | **元論文本文の生成器**（F ~ N(0,10) を分散と解釈、Σ=0.1I、正規化なし、w0 ~ N(0,1)、w = 3N(0,1)） | **G/B、K_GEN {1,3,5,7,9}、候補 1〜10、n=150、d=15、L=10、10 反復、各 K_GEN に 1 データ × 10 初期値 = 500 fit** | C_Q、C_Lap、C_ELBO、条件付き 3 種 | 試行ごと：C_Q 29/50（10, 7, 6, 5, 1）。**誤りはすべて K_GEN+1 への過大選択**。最良の試行の値を使う参考規則では argmin が 1, 3, 5, 7, 9（Fig.3 と一致） | **main 未統合**（`origin/research/paper-fig3-condition-criteria-20261109`、`bada1d2`） |
| Issue #128 B（元論文型生成器での基準比較） | E | `reproduction/src/data_generator.set_true_params`（MATLAB 移植） | L=5、8 反復、2 start | C_Q、C_Lap、条件付き | — | main（`tools/research_audit/run_paper_condition_k_criteria.py`） |

## 2. 今回の研究問いに対して既に分かっていること

1. **Series O に最も近い既存実験は、原論文 Fig.3 条件の基準比較（main 未統合）である。** 条件は Table I と一致し（n, d, K_GEN, 候補, L, 反復, 試行数）、生成器は元論文の本文に従う。ただし次の限定がある。
   - データは各 K_GEN に 1 つだけで、独立な生成データの反復がない。
   - w0・w の生成規則は仮定（既存コードから借用）。
   - 系列は E。
2. **既存の正式系列（fixed）の K 選択は K_TRUE=3 だけである。** K_TRUE を変えた fixed の実験はない（KS-4〜KS-6 の系列はすべて E）。
3. **誤りの向きは条件で逆になる。**
   - 信号をそろえた canonical 条件（P/B、n ≤ 150）では、C_Q は過小選択した。
   - 元論文型の強い信号では、C_Q は K_GEN+1 への過大選択だった。
   - どちらも MCEM は 8〜10 反復で未収束。
4. **試行（初期値と MC 系列）のばらつきが大きい。** 同じデータ・同じ K で、基準の値の SD は 240〜480 だった（Fig.3 条件、事後的な診断）。これは基準の間の差より大きい。
5. **G/P（Series B）と B/G（Series C）で K_TRUE を変えた実験は、どの系列にもない。**

## 3. 既存の主張の境界（今回も守る）

- 「人工データの K_TRUE が真の潜在次元 K* である」と書かない（`KNOWN_ISSUES.md` L.513）。
- 一致性、「n を増やせば必ず K_TRUE に収束する」と書かない（L.515）。
- C_Q を Schwarz BIC と呼ばない。C_Lap を厳密な周辺尤度と呼ばない。
- lineage E の結果を本文に自動採用しない（MANUSCRIPT_ADOPTION_STATUS: NOT_DECIDED_BY_HUMAN）。
