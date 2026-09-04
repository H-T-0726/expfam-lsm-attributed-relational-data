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

