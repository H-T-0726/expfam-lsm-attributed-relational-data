# 自律研究セッション 引き継ぎ — true-K identifiability / clean K-selection

**作成日:** 2026-09-05
**種別:** handoff（人間のレビュー待ち）

---

## 1. Resume baseline

| 項目 | 値 |
|---|---|
| resume 時の HEAD | `53f347fa177aeee81947be581efd97a961acccf1` |
| resume 時の working tree | production artifact ディレクトリのみ untracked |
| production の状態 | **実行済み**（2026-09-04、exit 0、wall clock 8823.1 s） |

## 2. Branch

`research/true-k-clean-asymptotics-20260904`（**push していない。PR も作っていない**）

## 3. Baseline SHA

`origin/main` = `7e335602999977060208ce37ac8cdff8fedfa66e`

## 4. Final HEAD

`4f359e3f76a711bc5fe9a813f83b76d3cc5d91fa`（origin/main から **22 commits**）

---

## 5. resume 前に完了していたこと

- Phase 0–7: 理論監査、独立敵対レビュー 2 名（BLOCKER 3・HIGH 12 採択）、
  clean generator の仕様・実装・テスト 46 件、runtime benchmark、protocol 凍結
- Phase 9: production run（896 fits）が完了、exit 0
- Phase 11 の先行作業: 独立 auditor を**結果が存在する前に**作成

**中断箇所:** canonical docs への理論成果登録の途中。

## 6. resume 中に完了したこと

| Phase | 内容 |
|---|---|
| A | production evidence の凍結と一次 artifact からの検証 |
| B | 独立 artifact 監査（**PASS**）、`audit_report.json` 生成 |
| C–E | 生の fit 値からの選択の独立再計算（192 エントリ、**不一致 0**） |
| F | S2・S3・S4 を既存 fit のみから評価（**追加 EM なし**） |
| I | canonical docs 登録（KI-020・KI-021・KI-010/019 forward update、RESEARCH_MASTER §17、EXPERIMENT_REGISTRY） |
| K | `clean_true_k_results_20260905.md`（artifact から自動生成、`--check` CURRENT） |
| L | `teacher_discussion_summary_20260905.md`（20 節＋30 秒/2 分/5 分説明） |
| M | `teacher_expected_questions_20260905.md`（26 問） |
| N | `thesis_storyline_20260905.md`（W1–W7 の弱点監査つき） |
| P | 図 F8-1〜F8-5 を script 経由で生成し registry へ登録 |
| Q | `k_selection_theory_map_20260905.md` |
| R | `real_application_interpretation_20260905.md` |
| T | テスト・静的検証 |

---

## 7. Production evidence

| 項目 | 値 |
|---|---|
| artifact | `expfam/results/k_selection/clean_true_k_asymptotics_20260904/` |
| status | **SUCCESS_ARTIFACTS_PRESENT** |
| protocol hash | `547880a16aef6530cfdf7903c4e32f16062397e0bacc0c109d5c77fb9892ccc0` |
| run_code_sha | `63e1202258a71256a55732fc1832db13d7f7b2bd` |
| generator | `canonical-clean-v1`（**historical generator ではない**） |
| lineage | E（`numerics_mode="consistent"`）。**旧 0.5 lineage 不使用**。**本文採用不可** |
| 環境 | python 3.13.14 / numpy 2.3.5 / Windows-11 |
| `failure.json` | **不在** |

## 8. Production audit

`tools/research_audit/audit_clean_true_k_sweep.py`（**runner を import しない**、
全構造定数を独立 literal として保持、selection を生値から再計算）

**verdict PASS** / BLOCKER 0 / HIGH 0 / MEDIUM 0 / LOW 2
（LOW は「NaN fit 0 件」と「推定 Gram が全 64 セルで非 PSD（U7 の予測どおり）」）

監査器自体を **15 種の単一欠陥変異**でテストし、すべて検出することを確認済み。

## 9. fit 数

| 項目 | 値 |
|---|---|
| expected | **896** |
| actual | **896** |
| unique | **896** |
| duplicate | **0** |
| missing | **0** |
| failed | **0** |
| non-finite | **0** |
| NaN fit | **0** |
| retry | **0** |
| replacement | **0** |
| seed rescue | **0** |
| tolerance 緩和 | **0** |
| resumed | **False** |

## 10. true-K 理論

`K* = min{K : P0 ∈ M_K}`（M-closed 前提）。
**`K^rank ≤ K* ≤ K_TRUE` で、どの等号も自明でない。**

## 11. 識別可能性の結果

| ID | 命題 | 条件 |
|---|---|---|
| P1 | Poisson-X: モーメントから `FF^T` 復元、**X 周辺**の最小次元 = `rank(FF^T)` | unclipped link、`rank(F)=K`（generic） |
| P2 | Gaussian-Y: **単一 dyad** から `(w0, w², K, σ_y²)` | `w ≠ 0` |
| P3 | Gaussian-Y: `{P ∈ M_K : w≠0} ∩ M_{K+1} = ∅` | `w ≠ 0`。`M_K` 同士は `w=0` で交わる |
| P5 | Gaussian-X（Σ 既知）: 階数から `K` | `rank(F)=K` |
| P6 | Poisson-Y: `E[Y^r]<∞ ⟺ \|w\|<1/r` | unclipped link |
| P8 | Gaussian-Y: **三角形**から `w` の符号 | `w ≠ 0`, `n ≥ 3` |

**反例:** Bernoulli-X `d=1`（**X 周辺のみ**）、`w=0`、Gaussian-X `d=2,K=1`（counting）、
Bernoulli-Y `w0=0` の edge density。

## 12. clean generator

`expfam/src/experimental/data_generator_canonical.py`（`canonical-clean-v1`）。
`Z` 正規化なし・`F` 行正規化なし・X 事後正規化なし・Poisson clip なし・
分散パラメータを実際に使用・`rank(F)=K` を QR で構成保証（**seed rescue 禁止**）・
Poisson-Y は `|w|<1/2` を既定で強制。テスト 46 件。
**historical generator は byte-unchanged。**

## 13. 主要結果（`K_TRUE=5`）

| 基準 | n=50 | n=75 | n=100 | n=150 |
|---|:---:|:---:|:---:|:---:|
| **S1** held-out | 2/8 | **0/8** | 4/8 | **8/8** |
| **S2** Q ベース | 0/8 | 0/8 | 1/8 | **7/8** |
| S3 plug-in | 3/8 | 0/8 | 0/8 | 0/8 |

平均 selected K: S1 `2.62 → 3.00 → 4.50 → 5.00`、S2 `1.75 → 3.25 → 3.62 → 4.88`。

**平均は単調増加したが真値一致数は単調でない**（S1 は n=75 で 0/8）。
**誤りの向きは一貫して under-selection。**

## 14. control（`K_TRUE=1, 3`）

| 基準 | K_TRUE=1 | K_TRUE=3 |
|---|---|---|
| S1 | 4/4, 4/4, 4/4, 4/4 | 1/4, 1/4, 3/4, 4/4 |
| S2 | 4/4, 4/4, 4/4, 4/4 | 2/4, 3/4, 4/4, 4/4 |

**`K_TRUE=1` の 4/4 は成功例に使えない。** 支配的な誤り方が under-selection で、
`K=1` は候補下端、tie rule は最小 K を選ぶ。**下限効果と交絡している。**

## 15. criterion 比較

合計真値一致: **S1 39/64、S2 37/64、S3 3/64**。
一致セル数: S1 vs S2 **44/64**、S1 vs S3 2/64、S2 vs S3 0/64、**三者一致 0/64**。

**S3（`Z` を積分しない Q1 型）はほぼ全セルで候補上限 `K=7` を選んだ。**
**ただし S3 は本研究が定義した基準であり原論文 Eq.(26) ではない。**

## 16. 構造診断（S4）

推定 Poisson-X Gram は**全 64 セルで非 PSD**（最小固有値の中央値 −1.80 〜 −0.52）、
閾値なし階数は常に `d = 15` で `K` を返さない。
真の `K` での固有値ギャップ比は `n` とともに増加（`K_TRUE=5` で 1.79 → 2.30）するが、
**事前に固定できる閾値がないため selected K を作らない**（U7）。

## 17. 更新した canonical docs

| ファイル | 内容 | 削除行 |
|---|---|---|
| `KNOWN_ISSUES.md` | KI-020（Poisson-Y モーメント）・KI-021（generator 乖離）・KI-010/019 forward update | **0** |
| `RESEARCH_MASTER.md` | §17（true-K / clean K-selection フェーズ） | **0** |
| `EXPERIMENT_REGISTRY.md` | 実験行・EM なし成果物・図の行 | **0** |

**3 ファイルとも純粋な追記。** ブランチ全体で削除行 **0**。

## 18. 先生向けパッケージ

- `teacher_discussion_summary_20260905.md`（20 節＋30 秒/2 分/5 分説明）
- `teacher_expected_questions_20260905.md`（26 問、証拠と主張境界つき）
- 20260904 の草稿には**日付入り forward pointer のみ追加**（本文不変）

## 19. 修論統合

- `reports/thesis/thesis_storyline_20260905.md`（backbone A–J、**弱点 W1–W7**）
- `reports/thesis/thesis_figure_table_inventory_20260905.md`
- `reports/thesis/real_application_interpretation_20260905.md`
- `reports/identifiability/k_selection_theory_map_20260905.md`

**未作成:** 詳細章立て（`thesis_detailed_outline_20260905.md`）。

## 20. Tests

| 対象 | 結果 |
|---|---|
| `test_clean_true_k_sweep.py` | 32 passed |
| `test_data_generator_canonical.py` | 46 passed |
| `test_heldout_k_selection_pilot.py` | 216 passed |
| `verify_identifiability_identities.py` | 81 rows / **failure 0**（独立 41 / 構成上 40） |
| 監査再実行 | PASS |
| `build_clean_true_k_report.py --check` | CURRENT |
| `py_compile` | OK |

**production コマンドを呼ぶテストはない。** skip / xfail / assert 削除も行っていない。

## 21. 最終独立レビュー

**セッション終了時点で 2 名のレビューが実行中。結果は未取得。**
（Reviewer A: 科学・数理 / Reviewer B: 再現性・リポジトリ）

**次の人間はこれを再実行するか、レビューなしで進むかを判断すること。**
なお resume 前に別の敵対レビュー 2 名を通しており、そこでの BLOCKER 3・HIGH 12 は
すべて採択・訂正済み（`true_k_identifiability_review_20260904.md`）。

## 22. Claim ledger

### ALLOWED

- canonical model は明示条件下で proper（**Poisson-Y はモーメント有限性が別条件**）
- `K*` の定義と `K_TRUE` / `K^rank` からの分離
- 命題 P1・P2・P3・P5・P6・P8（**すべて前提を明記して**）
- historical generator が literal generator でないこと（**過去結果は無効化しない**）
- 実行された fit 数と integrity カウント
- 各 `(criterion, K_TRUE, n, replicate)` の selected K の正確な値
- 現行 `calc_bic_dual` が Q3 を使っておらず Schwarz BIC ではないこと

### QUALIFIED ONLY

- 「`n` の増加とともに平均 selected K が真値へ近づいた」
  → **有限範囲の記述・一致数は非単調・反復 4/8 のみ**を必ず併記
- 「S1 と S2 は似た挙動」→ **64 中 44 セル一致**を明記
- 「S3 は過大選択」→ **本研究の定義であり原論文の基準ではない**を併記

### NOT ALLOWED

K 選択の一致性 / 漸近一致性 / universal true-K recovery /
現行実装が Schwarz BIC として妥当 / held-out 予測 = true-K recovery /
Bernoulli 一般の識別可能性 / 「BIC は非入れ子だから使えない」（理由が誤り） /
「`n` を増やせば `K_TRUE` に収束する」/「`K_TRUE=1` で完全に回復した」/
「S3 の失敗は原論文 BIC の失敗」/「実データで `K*` を推定した」/
「historical generator のせいで過去結果は無効」

## 23. 未解決

| ID | 内容 |
|---|---|
| U1 | Bernoulli-X（`d>1`）の識別可能性 |
| **U2** | **Bernoulli-Y の識別可能性 — 実験で使っている family** |
| U3 | Gaussian-X（Σ 未知）の十分条件 |
| U4 | Poisson-Y の識別可能性 |
| U5 | Bernoulli-Y / Poisson-Y の非入れ子性 |
| **U6** | **`n→∞` の一致性（先生のご指摘 5）** |
| U7 | 有限標本の rank 閾値（推定 Gram は非 PSD） |
| U9 | clean construction で `K* = K_TRUE` か |
| U10 | held-out plug-in score の population target |
| U11 | `M_K` の閉性、誤指定下の pseudo-true `K` |
| U12 | `{K : P0 ∈ M_K}` の連結性 |
| — | 本モデルの RLCT、有効標本数の定義 |
| — | `K_TRUE=1` の下限効果の切り分け（**新しい事前登録が必要**） |
| — | 初期値不一致が criterion 由来か最適化由来か |
| — | X の寄与を分離した測定（今回は X 信号を 1 水準に固定） |
| — | inductive（新規ノード）評価、観測ゼロと欠測の区別 |

## 24. 変更ファイル

新規 29 / 追記 3（canonical docs）。**削除 0 行。**
`expfam/src/data_generator_expfam.py` は **byte-unchanged**。

## 25. Commit list（22）

```
2b6faa0 docs: start autonomous true-K identifiability research session
a9b7f38 docs: harden true-K identifiability theory
bab1837 docs: specify canonical clean generator
c673034 feat: add canonical clean synthetic generator
a9ff0eb test: validate canonical clean generator
10f94de docs: harden identifiability claims after adversarial review
63e1202 docs: freeze clean true-K experiment protocol   <- production ran at this SHA
3bd59d9 audit: add independent auditor for the clean true-K sweep
7f905b4 docs: update autonomous research state through protocol freeze
e6ef0cd test: validate the clean true-K runner and its auditor
885bce1 docs: add artifact-driven results report generator for the clean sweep
53f347f docs: draft the teacher discussion summary       <- resume baseline
123b4b7 audit: archive clean true-K production evidence and its independent audit
d4152c9 docs: register the true-K theory and the clean K-selection experiment
20962b7 docs: prepare the teacher discussion package with the experiment results
71b3218 docs: add expected advisor questions with evidence and claim boundaries
5c4967a docs: update autonomous research state after the resume session
885fa00 docs: audit the thesis storyline against the actual evidence
90b1d2d docs: map the K-selection theory, proven versus unresolved
8af0ae8 docs: interpret the results for real application, where K* does not exist
4147648 docs: inventory the thesis figures and tables with their claim boundaries
4f359e3 docs: generate the clean true-K figures from the artifacts
```

## 26. 人間が次にやること

1. **`git log` と `git diff origin/main...HEAD` を読む。** push・PR は**していない**。
2. **`teacher_discussion_summary_20260905.md` を読む**（先生への説明はこれを使う）。
   §14 の「n を増やして何が起きたか」と §15 の `K_TRUE=1` の交絡が最重要。
3. **最終独立レビュー（§21）を再実行するか判断する。** セッション終了時点で結果未取得。
4. **PR を作るかどうかを判断する**（Human Gate。agent は作っていない）。
5. **次の実験を決める場合**は新しい事前登録が必要。候補は
   ①`K_TRUE=1` の下限効果の切り分け ②start 数を増やして不安定性の原因を分離
   ③Bernoulli-Y の識別可能性（理論作業）。

---

## 最終ステータス

```
PRODUCTION RERUNS DURING RESUME      = 0
NEW PRODUCTION FITS DURING RESUME    = 0
ORIGINAL PRODUCTION EXPECTED FITS    = 896
ORIGINAL PRODUCTION ACTUAL FITS      = 896
ORIGINAL PRODUCTION VALID FITS       = 896
MISSING FITS                         = 0
DUPLICATE FITS                       = 0
RETRIES                              = 0
REPLACEMENTS                         = 0
PRODUCTION AUDIT VERDICT             = PASS (BLOCKER 0 / HIGH 0 / MEDIUM 0 / LOW 2)

PRIMARY K_TRUE=5 RESULT
  held-out (S1): exact 2/8, 0/8, 4/8, 8/8 at n=50,75,100,150;
                 mean selected K 2.62 -> 3.00 -> 4.50 -> 5.00
  Q-based  (S2): exact 0/8, 0/8, 1/8, 7/8; mean 1.75 -> 3.25 -> 3.62 -> 4.88
  the MEAN rose monotonically; the EXACT COUNT did not (S1 dips to 0/8 at n=75)
  every error was under-selection

CONTROL K_TRUE=1/3 RESULT
  K_TRUE=3: S1 1/4,1/4,3/4,4/4 ; S2 2/4,3/4,4/4,4/4 -- same direction as K_TRUE=5
  K_TRUE=1: S1 and S2 both 4/4 at every n, but CONFOUNDED with the candidate
            floor and the smallest-K tie rule; NOT usable as a success case

THEORETICAL CONSISTENCY PROVEN       = NO
CANONICAL DOCS STATUS                = COMPLETE
TEACHER PACKAGE STATUS               = COMPLETE
THESIS INTEGRATION STATUS            = PARTIAL (detailed outline not written)

FINAL REVIEW: two reviewers were still running at session end; results NOT
              obtained.  A previous adversarial round accepted 3 BLOCKER and
              12 HIGH findings and the corrections are committed.

WORKING TREE CLEAN                   = YES
READY_FOR_HUMAN_REVIEW               = YES
```
