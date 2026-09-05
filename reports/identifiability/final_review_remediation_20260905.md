# 最終敵対レビューの finding と対応記録

**作成日:** 2026-09-05
**レビュー実施:** 独立レビュアー 2 名（A: 科学・数理 / B: 再現性・リポジトリ）
**方式:** 完成物を「壊しに行く」ことを依頼。結論を正しい前提として与えていない。

**重要:** 本レビューは **production experiment を再実行していない。**
finding はすべて documentation / test / auditor の問題であり、
**896 fits・artifact・独立監査・独立再計算のいずれにも影響しない。**

**実装系列の明記（root `CLAUDE.md` §3 / KI-002）:** 本レポートが引用する実験数値は
**lineage E**（objective-consistent **experimental prototype**、**本文採用不可**）の
clean true-K n-sweep によるものである。

本レポートは AI が生成したレビューの記録であり、`KNOWN_ISSUES.md` KI-007 の対象である。
**採択した指摘はすべて本監査が独立に再検証してから反映した。**

---

## 0. 総括

| レビュアー | BLOCKER | HIGH | MEDIUM | LOW | ACCEPT（破れなかった） |
|---|---:|---:|---:|---:|---|
| A（科学・数理） | 0 | 3 | 9 | 2 | 全 headline 数値の再現、prose の漸近語彙の規律、S3/Eq.(26) の分離、Gram 診断 |
| B（再現性） | 0 | 3 | 7 | 3 | 再実行なし・provenance・seed/mask・生成器不変量・再現性・数値一致 |

**両者とも「production evidence の完全性は保たれている」と結論した。**
問題は **その完全性について、および結果について語った内容**にあった。

**対応後: BLOCKER 0 / HIGH 0。** MEDIUM・LOW も全件対応済み（未対応 0）。

---

## 1. Reviewer A（科学・数理）

| ID | severity | 判定 | 対象ファイル | 必要な修正 | 状態 | 検証 | 科学的影響 |
|---|---|---|---|---|---|---|---|
| F-01 | — | **ACCEPT** | — | なし | — | 全 headline 数値と約 120 の二次数値を独立再計算し完全一致 | なし（肯定的確認） |
| **F-02** | **HIGH** | **REJECT** | results report §9, storyline, registry | 固定 EM 予算（`num_iter=8`）の `K` 依存性が under-selection の代替説明になりうることを限界に追加 | **完了** | 限定 10 として追加、「書いてはいけない」にも追加 | **大。** 主要結果の因果解釈が criterion 由来と断定できなくなった |
| **F-03** | **HIGH** | **REJECT** | results report §2・§8.1(2) | 信号整合は 1 次モーメントのみ。`‖f_l‖² ∝ Beta(K/2,(d−K)/2)` で分散が K 依存、Y の超過尖度は `6/K` | **完了** | 実測 `x_max` 平均 67.3(K=1) 対 18.0(K=5)、最大 386 対 42 を独立確認 | **大。** `K_TRUE=1` に第 3 の交絡が加わった |
| **F-04** | **HIGH** | **REJECT** | results report §8.1(6)・§9 | S2 の実効罰則は `p log n` でなく `Q_strict` 内の `ln p(Z)` 由来の `O(n)` 項 | **完了** | `−2ΔQ` 中央値 79→257 対 `Δp·log n` 45→58（比 1.76→4.46）、`Q_strict` は 506/768 で減少 | 中。S2 が BIC 型からさらに遠いことが分かり、既存主張は**強化**される |
| F-05 | MEDIUM | **REJECT** | RESEARCH_MASTER §17.5, Q11, handoff | 固有値ギャップ比は非単調（n=75 で下がる） | **完了** | 1.794 / 1.635 / 2.125 / 2.299 を再計算 | 小。診断の記述のみ |
| F-06 | MEDIUM | **REJECT** | results report §8.1(5), RESEARCH_MASTER | 「不安定領域と初期値依存領域が一致」は両端のみ。**撤回** | **完了** | 全系列 8/8, 4/8, 5/8, 1/8 対 一致 2/8, 0/8, 4/8, 8/8 で逆転を確認 | 中。診断の解釈を撤回 |
| F-07 | MEDIUM | **REJECT** | results report §8.1(2) ほか 5 文書 | tie rule は下限効果の根拠にならない（同点 0/192、最小マージン 9.7e−05） | **完了** | 再計算で確認 | 小。caveat は他の 2 本の脚で成立 |
| F-08 | MEDIUM | **UNRESOLVED（記録）** | results report §9 | `em_runner` の内部 NaN リトライ（`seed+retry*1000`・`newton_alpha` 半減）は artifact から検出不能 | **完了（限界として記録）** | `em_runner.py` L.155-163 を一次確認 | 中。「retry 0」は sweep 宣言であり内部再試行を排除しない |
| F-09 | MEDIUM | **REJECT** | RESEARCH_MASTER §17.1 | 「81 rows」を内訳なしで再掲していた | **完了** | 独立 41 / 構成上 40 の内訳を復元 | 小 |
| F-10 | MEDIUM | **REJECT** | 理論監査 §9.4 | P2 の証明に競合表現への量化がない（P1 では前回追加済み） | **完了** | 一段を追加 | 小。結論は真、厳密性の欠陥 |
| F-11 | MEDIUM | **REJECT** | results report §1 | `experiment_id` の "asymptotics" が主張と矛盾 | **完了（注記）** | artifact は凍結のため改名せず注記 | 小 |
| F-12 | MEDIUM | **REJECT** | teacher summary §13・2分説明, Q12 | 口頭説明で `K*`/`K_TRUE` の区別が落ちる | **完了** | §13・2分・5分・Q12 に復元 | 中。ゼミで最も誤解されうる箇所 |
| F-13 | MEDIUM-LOW | **REJECT** | RESEARCH_MASTER §17.5 | 限定リストに Bernoulli-Y の gap がない | **完了** | bullet 追加 | 中。canonical 文書で最大の caveat が欠落 |
| F-14 | LOW | **REJECT** | Q7, registry | Y density の範囲がセル平均であることを明示 | **未対応（理由: 下記 §3）** | — | 極小 |
| F-15 | LOW | **ACCEPT（修正）** | results report §5 | 「64/64 で一致しなかった」→「三者一致が成立しなかった」 | **完了** | S1 vs S2 44/64 を併記 | 小 |

## 2. Reviewer B（再現性・リポジトリ）

| ID | severity | 判定 | 対象ファイル | 必要な修正 | 状態 | 検証 | 科学的影響 |
|---|---|---|---|---|---|---|---|
| A1–A8 | — | **ACCEPT** | — | なし | — | 再実行なし・provenance・seed/mask・生成器不変量・再現性・数値一致をすべて独立確認。**per-fit runtime 総和 8821.7 s が wall clock 8823.1 s に収まる**という決定的証拠を新たに提示 | なし（肯定的確認） |
| **B1** | **HIGH** | **REJECT** | auditor + handoff + registry | 「mutated runner が自己認証できない」は過大主張。6 改竄が通過 | **完了** | 2 改竄を自分で再現→修正→**6/6 すべて検出**を確認 | 中。保証の**強度**の主張が誤りだった |
| **B2** | **HIGH** | **REJECT** | doc-consistency test + 2 文書 | lineage テストが違反文書を除外する形で書かれていた（**私の失敗**） | **完了** | スコープを**ディスク走査で派生**に変更。ラベル除去で実際に FAIL することを確認。さらに 2 文書の違反を新たに検出・修正 | 中。先生に見せる文書が prototype 数値をラベルなしで引用していた |
| **B3** | **HIGH** | **REJECT** | storyline / outline / inventory | 修論 8 章が prototype を Human Gate 越しに昇格させていた | **完了** | 3 文書すべてに Human Gate 注記。**昇格判断は行っていない** | **大。** `CLAUDE.md` §6 の Human Gate 事項 |
| B4 | MEDIUM | **REJECT** | 8 文書 | 「一貫して under-selection」は S3 で偽（61 over / 0 under） | **完了** | S1 23under/2over, S2 27/0, S3 0/61 を再計算 | 中。実務ガイダンス W-a にも波及していた |
| B5 | MEDIUM | **REJECT** | RESEARCH_MASTER §17.5 | ギャップ比の単調性（F-05 と同一） | **完了** | 同上 | 小 |
| B6 | MEDIUM | **REJECT** | runner + 約 10 文書 | integrity カウンタはリテラルで、証拠として循環 | **完了** | runner に semantics 注記、文書は実際の 3 根拠を指すよう変更 | 小。主張自体は真、根拠の提示が誤り |
| B7 | MEDIUM | **REJECT** | RESEARCH_MASTER | claim ledger 節がない。P4/P5/P7 未登録 | **完了** | §18 を §14/§16 形式で追加、P4/P5/P7 を §17.3 に登録 | 中。canonical な claim 参照先がなかった |
| B8 | MEDIUM | **REJECT** | 20260904 草稿 | 実在しないファイルへの参照 2 件 | **完了（banner のみ）** | 本文は凍結のまま、banner で読み替えを明示 | 小 |
| B9 | MEDIUM | **REJECT** | results report・RESEARCH_MASTER・teacher summary | start 不一致診断が post-hoc であることが未表示 | **完了** | 各所に post-hoc ラベル | 中。事前登録の規律に関わる |
| B10 | MEDIUM | **REJECT** | handoff §4 | Final HEAD が古く、自身が言及する成果物を含まない commit を指していた | **完了** | 最終 commit 時に埋める指示に置換 | 中。人間が誤った tree をレビューしうる |
| B11 | LOW | **REJECT** | auditor | NaN fits で絶対に落ちない | **完了** | HIGH 化＋runinfo との突合。896/896 NaN 改竄を検出 | 小 |
| B12 | LOW | **ACCEPT（注記）** | — | run 中に commit あり。clean-tree は t=0 のみ | **記録のみ** | 該当 4 commit が runner/generator/model を触っていないことを確認済み | 極小 |
| B13 | LOW | **REJECT** | inventory, state JSON | G1–G7 と「5 つ」の不一致ほか | **完了（一部）** | inventory を修正 | 極小 |

---

## 3. 対応しなかったもの、およびその理由

| ID | 内容 | 理由 |
|---|---|---|
| **F-14** | Y density の範囲がセル平均であること | **未対応。** 記述は「Y density 0.318–0.340」で、`generator_provenance.csv` の per-cell 値（各セルの density）の範囲として正しい。個別 replicate との差は 0.295–0.359 で、**主張の向き（K_TRUE 間で揃っている）を変えない**。文言追加の価値が低いと判断した。**指摘自体は正しい。** |
| **B12** | run 中の commit / clean-tree が t=0 のみ | **記録のみ。** 該当 4 commit は auditor・test・report builder・teacher 草稿であり、runner・generator・model・`em_runner` のいずれにも触れていないことを確認済み。runner を run 完了時にも clean-tree 再確認させる改修は、**production を再実行しない以上この artifact には適用できない**ため、将来の実験への申し送りとする |
| **F-08** | 内部 NaN リトライが検出不能 | **UNRESOLVED として記録。** `em_runner` は forward-only の既存 lineage で、**本セッションの scope 外**。改修は新しい実験の事前登録を要する。現 artifact については「検出できない」と明記した |

---

## 4. Reviewer A と Reviewer B の修正が矛盾しないことの確認

| 論点 | A の要求 | B の要求 | 矛盾 | 対応 |
|---|---|---|---|---|
| 「一貫して under-selection」 | F-02 で「これは EM 予算でも説明できる」 | B4 で「S3 では偽」 | **なし。** 別々の欠陥 | 両方適用（スコープ＋代替説明） |
| ギャップ比の単調性 | F-05 | B5 | **同一指摘** | 一度の修正で両方充足 |
| `K_TRUE=1` の caveat | F-07（tie rule は根拠にならない）・F-03（第 3 の交絡） | — | **なし。** caveat を弱めず**強める**方向 | 両方適用 |
| auditor の強度 | — | B1 | **なし** | 適用 |
| lineage 表示 | — | B2 | **なし** | 適用 |
| post-hoc 診断 | F-06（解釈を撤回） | B9（post-hoc と明示） | **なし。** 補完的 | 両方適用 |

**矛盾は 1 件もなかった。** F-03 と F-07 はいずれも `K_TRUE=1` の caveat を**強める**方向で、
B4 の scope 修正とも整合する。

---

## 5. 再チェック結果（同じ観点）

| 観点 | 結果 |
|---|---|
| 文書の数値が artifact と一致するか | **50 tests PASS**（`test_clean_true_k_doc_consistency.py`） |
| lineage ラベルが全該当文書にあるか | **違反 0**（スコープをディスク走査で派生） |
| 「一貫して under-selection」の未スコープ残存 | **0 件** |
| ギャップ比の単調性主張の残存 | **0 件** |
| auditor が改竄を検出するか | **6/6 検出**（テストとして固定） |
| production 監査 | **PASS**（BLOCKER 0 / HIGH 0 / MEDIUM 0） |
| 結果レポートの再現性 | `--check` **CURRENT** |
| 全テスト | **135 passed** |

---

## 6. このレビューの限界

- レビュアーは **AI である**。本レポートは KI-007 の対象であり、それ自体を一次証拠としない。
- **Reviewer B は動いているブランチを監査した**（監査中に HEAD が 6 回進んだ）。
  B10 と B12 の一部はその過程で部分的に解消されている。
- 両レビュアーとも **`n→∞` の一致性（U6）には解答を持っていない。** 未解決のままである。
- **Reviewer B が指摘した「forgery が内部整合を保てば検出できない」という限界は解消していない。**
  auditor の保証は構造的であって証拠的ではない、と docstring に明記した。
