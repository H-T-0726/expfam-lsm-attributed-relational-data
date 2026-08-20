# START_HERE.md — DEPRECATED（入口は README.md に移動しました）

> ⚠ **このファイルは入口としては非推奨です。**
> かつてはリポジトリ全体の入口でしたが、案内層の刷新（2026-08-20）により
> その役割は `README.md`（人間向け）と `CLAUDE.md`（Claude Code 向け）に分割されました。
> **ここに書かれていた高信頼／低信頼ファイルの一覧・混同注意事項は、下表の各正本へ移設済みです。**
>
> このファイルは、凍結済み文書（`CLEANUP_MANIFEST.md`、
> `docs/theory_audit/CLAUDE_FABLE_5_THEORY_AUDIT_MASTER_PROMPT.md`）からの参照を
> 壊さないために stub として残しています。**削除しないでください。**

---

## 現在の入口

| 知りたいこと | 現在の正本 |
|---|---|
| リポジトリの入口・環境構築・ディレクトリ規約・実験の回し方 | **`README.md`** |
| 研究内容（目的・従来手法・提案手法・数式・フェーズ史・先生対応） | **`RESEARCH_MASTER.md`** |
| Claude Code での作業規約（確定式・実装系列・表現の限定条件・安全ルール） | **`CLAUDE.md`** |
| 実験の provenance（実験 → スクリプト → CSV → 図 → 主張） | **`EXPERIMENT_REGISTRY.md`** |
| 既知のリスク・混同しやすい数値・まだ主張してはいけないこと | **`KNOWN_ISSUES.md`** |

補助:

- 学会予稿本体 — `conference_submission_final_draft.md`
- 実データ実験フェーズの総括 — `reports/real_data_experiment_summary.md`
- 実行環境ベースライン — `reports/environment/baseline_20260818.md`
- 補助資料の地図 — `docs/README.md`
- 外部 AI ツール向け派生資料の注意 — `docs_for_notebooklm/README.md`

---

## かつてここにあった注意事項の移設先

| 旧内容 | 現在の正本 |
|---|---|
| 高信頼／低信頼ファイルの一覧 | `README.md`（ディレクトリ規約・source of truth）＋ `KNOWN_ISSUES.md` |
| 旧版と fixed 版を混同しない | `CLAUDE.md` §3（実装系列）、`KNOWN_ISSUES.md` KI-002 |
| 23.6× / 41.45× / 38.97× の混同注意 | `KNOWN_ISSUES.md` KI-003 |
| 原稿式と Python 実装の 0.5 差 | `CLAUDE.md` §2・§5、`KNOWN_ISSUES.md` KI-001 |
| 実データ実験フェーズは学会予稿に未収録 | `README.md`（現在の位置づけ）、`RESEARCH_MASTER.md` §8b |
| Claude Code に作業させる前のルール | `CLAUDE.md` §4〜§6 |
