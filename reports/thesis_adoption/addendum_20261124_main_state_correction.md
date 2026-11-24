# 追補（forward addendum）：`main` の状態に関する誤記の訂正

- 作成日：2026-11-24（同日作成の 4 文書への追補）。branch `docs/thesis-adoption-20261124`。
- 対象：同じディレクトリの 4 文書（commit `cb86642`）。**4 文書の本文は書き換えない。** 本追補が優先する。
- **研究上の採用判断（区分 A〜F、推奨、Human Gate の項目）は変更しない。** 訂正するのは `main` の状態の記述だけである。

## 1. 何が誤っていたか

4 文書は、作業 PC の **local `main`（`8f6d41c`、PR #149 の merge）を最新の `main` として**記載した。
しかし 4 文書の作成時点（2026-11-24）で、GitHub の `origin/main` は既に `1b5030b` だった。

| 確認した事実（2026-11-24、`git ls-remote` と GitHub API） | 値 |
|---|---|
| `origin/main` の HEAD | `1b5030b`（PR #154 の merge、2026-11-06 00:12 JST） |
| local `main` | `8f6d41c`（`origin/main` より 41 commit 遅れ） |
| `8f6d41c` は `origin/main` の祖先か | はい |
| 間に merge された PR | #150（EM 収束感度、`dad67b1`、C_Lap checkpoint 再評価 `a1bd7a7` を含む）、#153（R1/R2、`9a702c4`）、#154（d-sweep、`cbd3bcd`）。いずれも 2026-11-05〜06 |

したがって、これは「作成時点では正しかった記述が後で古くなった」のではなく、**作成時点で既に誤っていた**（古い local ref を参照した）。

## 2. 該当箇所と訂正

| 文書・箇所 | 元の記述 | 訂正 |
|---|---|---|
| `implementation_lineage_adoption.md` §0 表 1 行目 | `8f6d41c` = `main`（PR #149 まで） | `main` は `1b5030b`（PR #154 まで）。`8f6d41c` はその祖先 |
| 同 §0 表 2 行目 | `f71c04d`（AF1〜AF8-v3、d-sweep・R1/R2・checkpoint 再評価を含む）「main には未 merge」 | `f71c04d` という commit と AF1〜AF8-v3 は `main` に未 merge（変更なし）。**ただし d-sweep・R1/R2・checkpoint 再評価・EM 収束感度は PR #150/#153/#154 で `main` に統合済み** |
| `claim_adoption_matrix.md` KS-4 行の commit 列 | 「`main` = `8f6d41c` には未収録。dsweep などは §20–22 の統合 PR 待ち」 | 3 つの結果ディレクトリ（`dsweep_k_selection_20261105`、`k_selection_robustness_20261105`、`clap_checkpoint_reeval_20261105`）と `RESEARCH_MASTER.md` §20–22 の Claim Ledger は **`origin/main`（`1b5030b`）に存在する** |
| `claim_adoption_matrix.md` の commit 列 `8f6d41c` 全般 | `8f6d41c` | 参照先として有効（`origin/main` の祖先なので同じ内容が `main` にある）。「最新 main」の意味では読まない |

`f71c04d`・`c535910`（総括 6 文書）・`856d475`（PR #168）・`4fa6159`（PR #129）が `main` に未 merge という記述は、2026-11-24 時点でも正しい。

## 3. 採用判断への影響

KS-4 の区分 **D（付録）は変えない。** D の理由は MCEM 未収束・原因未分離・lineage E-scalar であり、`main` への統合状況ではない。
`main` に統合されたことで KS-4 は「未 merge の作業 branch 上の結果」（`AGENTS.md` §3）ではなくなったが、本文採用にするかどうかは研究者と指導教員が判断する。

## 4. 今後の K 選択研究の基点

新しい作業は local `main` ではなく `origin/main`（2026-11-24 時点で `1b5030b`）から branch を切る。
