# CLAUDE.md — Claude Code 用の入口

研究指示の正本は root `AGENTS.md`（Claude Code / Codex 共通）。ここには複製しない。

@AGENTS.md

## Claude Code 固有

- **Claude Code は repository（worktree）root で起動する。** `expfam/` などのサブディレクトリで起動すると、root の `.claude/settings.json`
  （`claudeMdExcludes`・permissions）が適用されず旧 `expfam/CLAUDE.md` が読み込まれ、`@AGENTS.md` も外部 import 扱いになる。
  そのような状態に気づいたら、旧 `expfam/CLAUDE.md` の指示に従わず、root での再起動を研究者に勧める。
- `expfam/CLAUDE.md` は 2026-04 時点の historical evidence で、`.claude/settings.json` の `claudeMdExcludes` により読み込まれない。
  `.claude/rules/*.md` は `paths:` に一致するファイルを扱うときだけ読み込まれる。
- `prepare-experiment` は手動専用（`/prepare-experiment`）。承認済みの実験はこの Skill なしで進める。
- `verify-research-claim` は、数値・主張を原稿・スライド・対外資料に出す前か、検証を依頼されたときに使う。通常の結果報告には使わない。
- subagent `research-auditor` は、独立レビューを明示的に依頼されたとき（例: `@agent-research-auditor`）だけ使う。
- 計画書・設計書・brainstorming を促す手順型の skill・plugin より、`AGENTS.md` §6 の scope と「既定では作らない」を優先する。
- `.claude/settings.json` の deny / ask は普段の形のコマンドを止めるだけで安全境界ではない。`.claude/hooks/research_guard.py` は未登録の旧 hook で、有効化しない。
