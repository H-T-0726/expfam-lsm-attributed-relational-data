# 09 独立レビューの記録（2026-11-06）

**レビュー者**：subagent `research-auditor`（読み取り専用。依頼で明示的に求められた独立レビュー）。
**対象**：bb46aa3 以降のすべて。報告 01〜08・10、`tools/research_audit/n75_*.py`・`run_n75.py`・`evaluate_n75.py`・`estep_n75.py`・`analyze_n75.py`、結果ディレクトリ、レジストリの行。
**確認できなかったこと**：レビュー者は git を実行できなかった。履歴に関わる項目は、作成者が §3 で確認した。

## 1. 結論

- 実行・評価器・seed・初期状態のハッシュ・再利用の照合・漏洩の扱いは、すべて問題なかった。
- 報告の数値は約 70 か所が CSV と一致した。不一致は 3 か所（M1）。
- **BLOCKER 1 件**：08・10・レジストリの因果の説明が、本研究自身の CSV と矛盾していた。修正した。

## 2. 指摘と対応

| # | 重大度 | 指摘 | 対応 |
|---|---|---|---|
| F1 | BLOCKER | 「n=75 そのものが原因ではないことを実証」「失敗は X の SNR が弱い条件でだけ起きる」は支持されない。PR #160 E2 の弱い信号では、X の SNR は n=75 と n=150 で同じ（0.68）だが、C_Q は 0/4 と 3/4。N3 の予測は反証されている。08 の N3 は n=150 の行を引用していなかった | **修正**。N2 を「n=75 は十分条件ではない。n は原因から除外されない」に、N3 を「述べた形では反証された（FALSIFIED）。事後的・未検証の仮説として総情報量（X＋Y）」に変えた。n=150 の行と PR #160 の結果も載せた（`causal_hypotheses.csv` を再生成、08 §1・§2、10 の 4・8）。レジストリには訂正行を追記した |
| F2 | MAJOR | 「本来の問いには答えられていない」が 10 で目立たない | **修正**。10 の冒頭に置き、答え 1 にも明記した |
| F3 | MAJOR | 失敗したセルがないのに、推定器を失敗の原因から外していた | **修正**。08 §4・10 の答え 10 を「失敗条件で OLD を実行していないので判定できない」に変えた |
| F4 | MAJOR | 以前の成功を「X の信号が強く全ペア」と言い切っていた | **修正**。候補（n=150・強い信号・全ペア）を並べ、どれが必要かは切り分けられていないとした（08 §3、10 の答え 9） |
| M1 | MINOR | 07 の ŵ 1.13→1.12、K=6 のスコア −0.428→−0.427、05 の余裕の変化「−30〜−100」の誤り | **修正**（05 はデータごとの変化 −58・+2・−65・−103 と中央値 −32） |
| M2 | MINOR | 10 の「K の選択では違いはなかった」は C_Q・C_Lap に限る。「1.14〜1.15」は G_canon に限る | **修正** |
| M3 | MINOR | 04・07 に KI-001 の注記、05 に prototype の明記、10 に C_Q/C_Lap の呼称と K_GEN≠K0 の注記がなかった | **修正** |
| M4 | MINOR | run_log のハッシュは 3 ファイルだけで、推定コードの同一性は git diff に依存している（パスの一覧が省略されていた） | **作成者が確認**（§3）。03 §1 にパスの一覧を明記した |
| M5 | MINOR | Y の対称性の precheck は G_old を除外していた | 02 に明記した（G_old は生成器の構成上対称） |
| M6 | MINOR | 有限差分の目標を「Eq.(20)」と呼ぶと、印刷式の正誤を判定しているように読める | **修正**。02・06・10 では「無向ペアを 1 回ずつ数える完全な条件付き分布」と呼び、印刷式は判定しないと明記した |
| M7 | MINOR | 部分空間の回復は、列を加えると減らない指標で、過大な K を罰しない | 07 に明記し、「過大な次元が部分空間を壊さない」という文を撤回した |
| N1〜N6 | NOTE | 数値の照合、初期状態、fit の実行、再利用、INCOMPLETE の扱い、漏洩：問題なし | — |
| N5 | NOTE | E の PY_type の INCOMPLETE 行は、定義上の空欄（`analyze_fair_comparison.select` は "PY_hist" だけを除外する） | 03 に記載済み。結果は変えない |
| N7 | NOTE | `analysis_manifest.json` は、後から作った `condition_information.csv`・`metrics_by_k.csv`・fig8・`causal_hypotheses.csv`・`fairness_matrix.csv`・`experiment_protocol.csv` を含まない。evaluate・estep・condition_information は git head を記録しない | ここに記録する。これらのファイルは、commit 履歴（c2f025e 以降）で作成時点を追える |
| N8 | NOTE | MC 標本の分散は、各推定器が自分の θ で得た標本で、同じ状態での比較ではない | 06 に「整合するまで」と明記した |
| N9 | NOTE | 「ZZᵀ 0.76〜0.84」は、PR #160 の報告から引用していた | PR #160 の `parameter_recovery.csv`（E2 n=75 弱の 8 行）を出所として明記した |

## 3. 作成者による git の確認

- `git log --oneline bb46aa3..HEAD`：ecf2e2a（事前登録・precheck、fit の前）→ c143d6f・729a04f（解析スクリプトの追加）→ 4cf633a（fit の記録）→ 42b5e50（評価・E-step・`rec.at` の修正）→ c2f025e・26afa33（解析・報告）。
- `git diff --stat ecf2e2a 729a04f`：`analyze_n75.py`・`estep_n75.py`・`evaluate_n75.py` の新規追加だけ。
- `git diff --quiet ecf2e2a 729a04f -- expfam/src reproduction/src tools/research_audit/{fair_protocol,run_fair_comparison,run_n75,n75_protocol}.py`：差分なし。
- 解析が途中で止まったときの出力は、一度も commit していない（`selection_summary.csv` が初めて現れる commit は c2f025e）。修正は 42b5e50 の 1 行。
- `git diff bb46aa3 HEAD -- EXPERIMENT_REGISTRY.md`：削除 0 行（追記だけ）。

## 4. 監査の後に再生成した成果物

- `causal_hypotheses.csv`：N2・N3 の文と結果の列を直し、push の前に再生成した。修正前の版は c2f025e に残っている。
- `fairness_matrix.csv`・`experiment_protocol.csv`：同じスクリプトで再生成した。内容はバイト単位で同一（cmp で確認）。
