# Autonomous research session — 2026-09-04

長時間（最大約9時間）の自律作業ログ。**context compaction 後はこのファイルと
`autonomous_research_state_20260904.json` と `git log` を読めば継続できる。**

## Baseline

| 項目 | 値 |
|---|---|
| baseline (`origin/main`) | `7e335602999977060208ce37ac8cdff8fedfa66e` |
| branch | `research/true-k-clean-asymptotics-20260904` |
| started | 2026-09-04T14:57:14+09:00 |
| working tree at start | clean |

`origin/main` は本セッション開始時点で PR #68（Phase 8b Attempt 2 archive）と
PR #69（paper BIC / reproduction alignment 監査）が merge 済みの状態だった。

## 権限（このセッション）

- ALLOWED: repo 調査 / 数理導出 / documentation / forward-only 新規コード / test /
  synthetic data 生成 / **本 protocol で定義する NEW clean experiment の EM 実行** /
  artifact 生成 / local branch / local commit / subagent / static check
- NOT ALLOWED: main への直接 commit / push / force push / PR 作成 / merge /
  Issue 作成・コメント / historical artifact の削除・書換え / seed rescue /
  結果を見た後の threshold・criterion 変更 / 失敗 replicate の削除 / theorem の捏造

## 進捗ログ

### Phase 0 — startup（完了）

- baseline 確定、branch 作成、state/progress ファイル作成。

### Phase 1 — true-K / identifiability theory audit（完了）

`reports/identifiability/true_k_identifiability_hardened_20260904.md`。
新命題 P1（Poisson-X の population 識別性）・P2（Gaussian-Y の単一 dyad からの復元）・
P3（Gaussian-Y の非入れ子性）を証明し、`verify_identifiability_identities.py` で独立に数値確認した。

### Phase 2 — 敵対レビュー（完了）

独立レビュアー 2 名（数理 / 統計）に反証を依頼。**BLOCKER 3・HIGH 12 を採択**し本文を改訂した。
記録は `true_k_identifiability_review_20260904.md`、改訂履歴は理論レポート §20。
とくに次は初版の**誤り**であり撤回した:

- 反例 C1 を joint model の非識別性として書いていた（自身の P2 と矛盾）→ X 周辺限定に rescope
- 「`w` の符号は識別されない」→ **三角形から識別される（新命題 P8）**
- 「Schwarz BIC は入れ子性を前提とする」→ 前提としない。理由は特異性・境界・有効標本数
- 「held-out は `K*` を推定していない」→ 論法が無効。`[UNRESOLVED]` に降格
- 「71 checks PASS」→ 独立 41 / 構成上 40 に内訳を明示

### Phase 3–5 — clean generator（完了）

仕様 `canonical_clean_generator_spec_20260904.md`、実装
`expfam/src/experimental/data_generator_canonical.py`、テスト 46 件 PASS。
historical generator は 1 文字も変更していない。

### Phase 6 — runtime benchmark（完了）

n=50/75/100/150 で 4.68/7.24/9.98/16.63 秒/fit。**TIER A（896 fits, 推定 2.40h）を wall-clock のみで選択。**

### Phase 7 — protocol 凍結（完了）

`clean_true_k_experiment_protocol_20260904.md`、protocol hash `547880a1...`。
結果を見る前に解釈上の限定 L1–L8 を固定した。

### Phase 9 — production（実行中）

896 fits、FAIL CLOSED。開始 2026-09-04 19:12。

### Phase 11 — independent auditor（先行作成済み）

`tools/research_audit/audit_clean_true_k_sweep.py`。**結果が存在する前に**書いた。
runner を import せず、全定数を独立 literal として保持し、selection を生値から再計算する。


---

## 2026-09-05 continuation（resume session）

前回 session は canonical docs への理論成果登録の途中で終了した。**production は再実行していない。**

### PHASE A/B — production evidence freeze + 独立監査（完了）

- status **SUCCESS_ARTIFACTS_PRESENT**。`failure.json` 不在、8 artifact 揃い、exit 0、wall clock 8823.1 s。
- 一次 artifact から確定: expected 896 = actual 896 = unique 896、重複 0・欠番 0・失敗 0・
  非有限 0・NaN 0・retry 0・replacement 0・seed rescue 0・tolerance 緩和 0・resume False。
- 独立監査（runner を import しない）**verdict PASS**、BLOCKER 0 / HIGH 0 / MEDIUM 0 / LOW 2。

### PHASE C-E — 生の fit からの独立再計算（完了）

192 の cell-criterion エントリすべてを per-fit の生値から再導出し、**artifact との不一致 0**。
tie rule は 192 中 0 回しか発動していない。

### PHASE F — 副次基準（追加 EM なし、完了）

S2（Q ベース）・S3（plug-in conditional）・S4（Gram spectrum）を既存 artifact のみから評価。

### PHASE I — canonical docs 登録（完了。前回の中断箇所）

`KNOWN_ISSUES.md` に **KI-020**（Poisson-Y のモーメント存在。既定 `w=0.5` が分散発散の境界）と
**KI-021**（historical generator が canonical model の literal generator でない）、
および KI-010/KI-019 の forward update を追記。
`RESEARCH_MASTER.md` §17、`EXPERIMENT_REGISTRY.md` に実験行を追記。
**3 ファイルとも純粋な追記で、既存行の削除・変更は 0 行。**

### PHASE K/L/M — 結果レポートと先生向けパッケージ（完了）

- `clean_true_k_results_20260905.md`（artifact から自動生成、`--check` CURRENT）
- `teacher_discussion_summary_20260905.md`（20 節＋30 秒/2 分/5 分説明）
- `teacher_expected_questions_20260905.md`（26 問、証拠と主張境界つき）
- 20260904 の草稿には**日付入りの forward pointer のみ追加**し、本文は不変。

### PHASE T — テスト（完了）

clean-sweep + generator 78 passed / Phase 7e 216 passed / sweep・auditor 32 passed /
identity checker 81 rows 0 failures / 監査再実行 PASS / py_compile OK。
**テストは production コマンドを呼ばない**（呼ぶのは「フラグなしで拒否する」ことの検証のみ）。

### 主要結果（`K_TRUE` との一致であって `K*` との一致ではない）

| criterion | K_TRUE=1 | K_TRUE=3 | K_TRUE=5 (n=50→150) | 合計 |
|---|---|---|---|---|
| S1 held-out | 4/4 ×4 | 1/4, 1/4, 3/4, 4/4 | 2/8, 0/8, 4/8, **8/8** | 39/64 |
| S2 Q ベース | 4/4 ×4 | 2/4, 3/4, 4/4, 4/4 | 0/8, 0/8, 1/8, **7/8** | 37/64 |
| S3 plug-in | 0/4 ×4 | 0/4 ×4 | 3/8, 0/8, 0/8, 0/8 | 3/64 |

**平均 selected K は単調増加（S1: 2.62→5.00）だが真値一致数は単調でない（n=75 で 0/8）。**
**`K_TRUE=1` の 4/4 は下限効果と交絡しており成功例に使えない。**
