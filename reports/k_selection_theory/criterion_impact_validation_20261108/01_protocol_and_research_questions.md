# 01 プロトコルと研究の問い（2026-11-08）

## 位置づけ

- **ブランチ**：`research/criterion-impact-validation-20261108`。PR #163 の先端 `ebbe4ff` から分岐した。
- **系列**：E（prototype）。本文採用は未決定（NOT_DECIDED_BY_HUMAN）。
- **呼称**
  - C_Q は Q 関数に基づく完全データ型の基準（ICL 型）で、Schwarz BIC ではない。
  - C_Lap は Laplace 近似による基準で、厳密な周辺尤度ではない。
  - C_ELBO は、Laplace ガウスを q とした既存の変分下界による基準で、新しい BIC ではない。
  - B・C・D は、原論文 Eq.(16)/(26) に着想を得た条件付き尤度型の基準である。Z の評価点が原論文で指定されていないので、**原論文の再現ではない**。Poisson-X では「指数型分布族に拡張した条件付き尤度型」である。

## 研究の問い

| 問い | 内容 |
|---|---|
| RQ1 | 条件付き尤度型の基準・C_Q・C_Lap・C_ELBO で、同じデータの K_HAT はどの程度変わるか |
| RQ2 | 選択の違いは、当てはまり・潜在変数の事前分布の項・エントロピー・Laplace の非 2 次項のどれに対応するか |
| RQ3 | 弱信号で C_Q が過小選択する条件で、C_Lap や C_ELBO は K_GEN を選ぶか |
| RQ4 | Q_Z を除いた条件付き尤度型の基準の過大選択は、どの条件で起きるか |
| RQ5 | Gaussian-X と Poisson-X で、各基準の選択傾向に違いはあるか |
| RQ6 | Laplace と ELBO を比べたことで、周辺尤度の近似の限界について何が分かるか |
| RQ7 | 修士研究で、どの基準をどの目的で検証・採用するのが妥当か |

**正解の扱い**：「正解」は K_HAT = K_GEN（=5）を指す。K_GEN = K0 は仮定しない。

## 事前登録

正本は `tools/research_audit/criterion_impact_protocol.py`（protocol hash `3742ffe808c2…`）である。

| commit | 内容 | 時点 |
|---|---|---|
| `b630288` | プロトコル、C_ELBO の評価器とテスト、runner、no-fit の preflight | 既存状態の採点と新しい fit の**前** |
| `676eb9f` | MC の予算（事前の規則で M=2048 に固定） | 採点と fit の**前** |
| `cd16151` | 解析スクリプト | どの採点値も読む**前** |

## 2 つの段階

| 段階 | 対象 | 位置づけ |
|---|---|---|
| **Phase C（EXPLORATORY）** | PR #160 E2 の既存の 448 状態を再採点する（新しい fit なし） | 探索的 |
| **Phase E（CONFIRMATORY）** | 新しい独立な人工データ（rep 11〜14）で 448 fit を行う | 小規模な確認実験。4 データ/セルなので、一般的な優劣や一致性は示せない |

## Phase C で使う既存の状態

PR #160 E2 の 448 状態をそのまま使う。

| X の family | 信号・n | 状態の出所 |
|---|---|---|
| Gaussian-X | 全セル | E2 で新しく行った fit（224 件） |
| Poisson-X | 弱・n=75/150 | A1 EMC の checkpoint 8 の状態 |
| Poisson-X | 強・n=75 | PR #155 D の状態 |
| Poisson-X | 強・n=150 | E2 で新しく行った fit |

**照合**：基準 A〜E を同じコードで再計算し、PR #160 の `criterion_scores.csv` と照合した（許容誤差 1e−6。不一致なら停止）。

## Phase E の固定条件

| 項目 | 内容 |
|---|---|
| データの生成規則 | PR #160 E2 と同じ（`run_k_signal_strength.build_all` → `a1.make_cell`、Gaussian-X は `gaussian_x`（TAG_GX）） |
| 新しい rep | {11, 12, 13, 14}。この生成器で過去に使われた rep は 1〜8 |
| セル | x_family {gaussian, poisson} × n {75, 150} × 信号 {weak, strong} = 8 セル、各 4 データ |
| 候補と初期値 | K 1〜7、各 2 初期値 |
| 推定器 | `run_fair_comparison.fit_E`（系列 E、consistent、L=5、EM 8 反復）。変更なし |
| 観測マスク | Y の 80% |
| fit の数 | **448（上限）**。再試行・seed の追加・データの作り直しはしない |
| seed | data・split・model の seed は既存の規則（`hist.data_seed` / `split_seed` / `model_seed`）。model seed は E2 の規則どおり、x_family と信号の間で共有する（共通乱数）ので、異なる値は 112 個 |
| preflight（fit なし） | 新しいデータの組み立て処理は、rep 1〜4 で `run_fair_comparison.e2_data` と 32/32 がビット単位で一致した。新しい seed は、過去の CSV の全 seed 列（3,224 個の値）と衝突しなかった（`preflight.json`） |

## 選択の規則と評価

**選択の規則（両 Phase 共通）**
- 各 K について、2 つの初期値のスコアを平均し、最小値を与える K を選ぶ。
- 1e−12 以内の同点は、小さい K を選ぶ。
- 全候補の K で両方の初期値が有限でなければ、その基準はそのデータについて INCOMPLETE とする。補完や、別の初期値での代替はしない。

**主要評価（事前に固定）**
- **PE1**：C_Q・C_Lap・C_ELBO の K_HAT と、余裕（slack_45、slack_56）。
- **PE2**：$\text{slack}_Q-\text{slack}_{\rm Lap}=-[\Delta2H+\Delta\text{termA}+\Delta2R_3]$ と、$\text{slack}_{\rm ELBO}-\text{slack}_{\rm Lap}=-\Delta2R_3$ の厳密な分解。

**副次評価**
- 正解・過小・過大の件数
- B/C/D の結果
- family・n・信号による違い
- MC 誤差
- INCOMPLETE

**探索と確認の比較**
- 指標 M1〜M8 を、同じコードで両 Phase について計算し、並べる（`confirmation_vs_exploration.csv`）。
- 閾値や検定は使わない。

## データの結合（独立性の範囲。監査 M2 を受けて追記）

- 各 Phase の 32 データは、**8 個の独立な基本データ（2 n × 4 rep）**から作られている。
- 同じ (n, rep) の 4 セル（Gaussian/Poisson × 弱/強）は、Z と学習/評価の分割を共有する。
- 弱（条件 A）と強（条件 D）は、同じ基本データから結合した乱数で作る（`run_k_signal_strength.build_all`）。
- 同じ信号の Gaussian-X と Poisson-X は、Y をまったく同じに共有する。model seed も、x_family と信号の間で共有する。
- したがって、「30/30」「58/62」のような合算の件数は、**独立な反復の数を過大に見せる**。独立な基本データは、Phase ごとに n あたり 4 個である。
- Gaussian-X と Poisson-X の比較（RQ5）は、同じ Y の上での**対応のある比較**で、2 つの独立な標本ではない。
- Phase C（rep 1〜4）と Phase E（rep 11〜14）の基本データは、互いに独立である。

## 停止条件

- **停止する場合**：生成規則の不一致、seed の衝突、評価器の数式の誤り、上書き、主要な量が非有限で解釈できないこと。
- **停止しない場合**：選択の失敗、基準の成績の悪化、仮説が支持されないこと、一部の INCOMPLETE。

## 事前登録の時系列（監査 m1 を受けて、記録で確認した事実）

| 時刻（2026-11-08） | 出来事 | 根拠 |
|---|---|---|
| 18:35:08 | preflight.json の書き込み | ファイル時刻。runinfo の git_head は `ebbe4ff`（PR #163 の先端）。**preflight は、commit 前のコードで実行された**（下記の seed チェックの修正を含む）。protocol hash は `3742ffe…` で、commit されたものと一致する |
| 18:35:21 | `b630288`（プロトコル・評価器・runner）の commit | git |
| 18:35:54 | パイロット（elbo_mc_budget.json）。head `b630288` | ファイルと runinfo |
| 18:36:03 | `676eb9f`（予算）の commit | git |
| 18:36:13 | Phase E の fit を開始。head `676eb9f`（`confirm_run_log.txt`。ログに日付はない） | ログ |
| 18:37:53 | `cd16151`（解析スクリプト）の commit | git |
| 18:43:39 | Phase C の採点の出力（score_by_state_existing.csv） | ファイル時刻 |
| 19:59:38 | Phase E の fit が終了 | ログ |
| 20:07:14 | Phase E の採点の出力 | ファイル時刻 |

- 採点の CSV には git_head を記録していない（runner の不備）。Phase C の採点は、`676eb9f` の後に起動した（commit の後のコマンドとして実行）。
- 作成者が採点の値を初めて読んだのは、18:43 の出力の後である。

## seed の一意性（監査 m6）

- data_seed（base + 50000 + 10n + rep）と model_seed（rep·100 + n·1000 + …）は、rep ≥ 10 で n の桁と重なりうる（例：n=75・rep 11 と n=76・rep 1）。
- n=76・151 は使われていないので、実際の衝突はない。preflight の走査（過去の CSV の整数の seed 列、3,224 値）でも衝突はなかった。
- ただし、JSON やレポートにだけ記録された seed は走査していない。

## 計画からの技術的な修正（記録）

- preflight の最初の実行で、model seed の一意性チェックが「448 個が一意」を要求して停止した。
- E2 の規則では、seed は (n, rep, K, start) だけで決まり、X の family と信号の間で共有する。そのため、正しい期待値は 112 個である。
- チェックのコードだけを修正した。データ・seed・プロトコルの規則は変えていない。
- 最初の実行は preflight.json を書く前に止まっていたので、記録の上書きはない。
