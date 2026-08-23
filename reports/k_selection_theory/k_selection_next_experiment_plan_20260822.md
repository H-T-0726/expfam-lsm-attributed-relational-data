# 次段階の提案 — Phase 7b: K 選択スコアの系列整理と診断 pilot

**作成日:** 2026-08-22（第 1 巡監査の反映: 2026-08-23／PR #36 レビューによる全面改稿: 2026-08-23）
**対象 Issue:** #35（Phase 7a）の decision gate
`D: INVESTIGATE_ALTERNATIVE_CRITERION_BEFORE_K_SWEEP` に対応
**状態:** **DRAFT — 実行していない。実行しない。**
Issue #35 のスコープでは
**モデル変更・スクリプト作成・smoke・フィット・CSV 生成・図生成を一切行わない。**

前提となる監査: `reports/k_selection_theory/k_selection_theory_audit_20260822.md`（以下「監査」）

> **本改稿の要点**: 初版は「630 fits の `n × K_TRUE` full sweep を次に実行する」案だった。
> PR #36 のレビューを受けて、**まず量の定義と解釈を確定する軽量段階（Phase 7b）**を置き、
> full sweep は **Phase 7c の候補案**に降格した。**`RUN NEXT` とはしない。**

---

## 0. 用語（監査 §0 と同一）

```
lnpZ_det       := −(n·k/2)·(1 + ln 2π)                      監査 §5.3
S_cf(k)        := −2·( Q_strict − lnpZ_det ) + p̂·ln n       counterfactual diagnostic score
S_laplace_post := S_cf + Σ_i ln det A_i^{post}              post-hoc Laplace-curvature diagnostic score
```

**禁止される呼び方**（監査 §0・§19）:

- `S_cf` を「corrected BIC」「modified BIC」「true BIC」「Schwarz BIC」「原論文 BIC の再現」と呼ばない。
- `S_laplace_post` を「ELBO」「ELBO 補正 BIC」「variational BIC」と呼ばない。
  これは**指定した Gaussian entropy surrogate を代入したときの代数的恒等式**であり、
  実アルゴリズムが用いる `q` の ELBO ではない（監査 §6.5.1）。
- `S_cf` について使ってよい限定表現は **paper-Eq16-aligned diagnostic** まで。
  かつそれは「**当てはまり項の構成要素**が Eq.(16) と整合する」という意味に限る。
  罰則・Gaussian-Y の欠落定数・`μ_x` の 3 点で Eq.(26) とは異なる（監査 §7.5.3、E24）。
- **`S_cf` も `Q_strict` も「観測データ周辺尤度に基づく基準」と呼ばない**（監査 §0.1）。

### 0.1 3 つの「尤度」を分離する（監査 §0.1 と同一）

```
(Q1) ln p(X, Y | Z, θ)              条件付き / plug-in 尤度 = 原論文 Eq.(16) の ln L
(Q2) ln p(Z, X, Y | θ)              完全データ対数密度      = Eq.(18) の Q・現行 Q_strict の対象
(Q3) ln p(X, Y | θ) = ln ∫ …dZ      観測データ周辺尤度      = Schwarz BIC が本来対象とする量
```

| スコア | 対象とする量 |
|---|---|
| C1 現行 Q ベース | (Q2) の近似事後期待値 |
| C2 `S_cf` | (Q1) に構成要素レベルで整合 |
| C3 `S_laplace_post` | どれでもない（代数的診断量） |
| C4 held-out 予測 | どれでもない（予測リスク） |

> **4 候補のいずれも (Q3) ではない。**
> 「原論文 Eq.(26) は standard Schwarz BIC である」とは書かない。
> Eq.(16) は `z_i` に条件づけた量であり `Z` を積分していないからである（監査 E25、U6-c）。

---

## 1. なぜ full sweep を先にやらないのか

監査 §7.5.1 で、**一次資料により**次が確認された:

| | BIC の当てはまり項 | 対象の量（§0.1） | `p(Z)` を含むか | `Z` を積分するか |
|---|---|---|:---:|:---:|
| 原論文 Eq.(26) | `ln L` = Eq.(16) | (Q1) | **含まない** | **しない** |
| 現行 Python `calc_bic_dual` | `Q_strict` | (Q2) | **含む** | しない |
| 正しい意味の Schwarz BIC | — | (Q3) | — | **する** |

さらに `scale_Z` + `var_z = 1` により、その余分な項は `−(nk/2)(1+ln 2π)` に決定論化する（監査 §5.3）。
そして監査 §13.2a により、その項を除くと fixed 系列 5 ケース中 4 ケースで
スコアは k について単調減少し、内点最小を持たなくなる
（**唯一の例外は Wine で、内点 k=5 を持つ**）。

> **この状態で `P(K̂ = K_TRUE)` を大規模に測っても、
> 得られた選択率が「どの量についての性質なのか」が定まらない。**
>
> なお、ここで「原論文の側に合わせればよい」とは結論しない。
> 原論文 Eq.(26) も (Q3) ではなく、どの型の規準として意図されたかは未解決である（監査 U6-c）。
> **確定したのは「選択肢が 3 つ以上あり、それぞれ別の量である」ことである。**

したがって Phase 7b の目的は**測定量そのものではなく、測定量の定義と解釈を固定すること**である。

---

## 2. Phase 7b の目的（3 つ）

| # | 目的 | 成果物 |
|---|---|---|
| **P-1** | 候補スコア 4 種の定義・計算手続き・解釈限界を文書として固定する | 定義表（実験なしでも書ける部分を含む） |
| **P-2** | `Σ_i ln det A_i^{post}` を**初めて測定する**（監査 U1。既存 CSV には 1 件も存在しない） | 記述統計と k 依存性のプロット |
| **P-3** | 4 種のスコアが小規模条件で**同じ k を選ぶか**を観察する | argmin の一致表 |

**Phase 7b は仮説検定を行わない。primary estimand を置かない。**
これは意図的である。検定すべき仮説は「どの量を selection target とするか」が決まってからでなければ
定式化できない。**`Δ_sel` のような推定量を Phase 7b で事前登録しない。**

---

## 3. 候補スコア 4 種（Phase 7b で定義を固定する対象）

| ID | 名称 | 定義 | 現状 | 位置づけ |
|---|---|---|---|---|
| **C1** | 現行 Q ベーススコア | `−2·Q_strict + p̂ ln n`（`calc_bic_exp`） | 実装済み | **変更しない。** 記述的ベースライン |
| **C2** | paper-Eq16-aligned diagnostic | `S_cf` | 事後計算のみ | 当てはまり項が原論文 Eq.(16) の `ln L` と**構成要素レベルで**整合（監査 §7.5.3）。**原論文 BIC の再現ではない** |
| **C3** | post-hoc 曲率診断 | `S_laplace_post` | **未測定**（U1） | **ELBO ではない**（監査 §6.5.1）。**primary alternative criterion として事前確定しない** |
| **C4** | held-out 予測選択 | 未定義 | **未実装** | 候補として残すが、`train_mask` の設計・分割の独立性・X 側 held-out の扱いが別途必要。**Phase 7b では実装しない** |

> **C3 を「正しい基準」として扱わない。** 監査 §6.5.1 の 5 つの理由により、
> `Ĥ({A_i^{post}})` は実アルゴリズムの `q` のエントロピーではない。
> Phase 7b では C3 を**初めて数値として観測すること**だけを目的とする。

---

## 4. Design（軽量 pilot）

| 項目 | 値 | 理由 |
|---|---|---|
| シナリオ | **A: `family_x = poisson`, `family_y = bernoulli`** のみ | Gaussian dispersion を持たないので stale-Σ（監査 E16）の交絡が構造的にない。生成器が Poisson-X を z-score 化しないので X 側正規化交絡もない |
| `K_TRUE` | **3** のみ | 既存の合成実験と**同じ設計定数**を使う（数値の直接比較のためではない。系列が異なるので比較しない — 下記注記）。`K_TRUE` 依存性は Phase 7c |
| `k_est` | `1, …, 7` | 両側の候補を確保 |
| `n` | `{75, 150}` | 150 は既存の合成実験と**同じ設計定数**。75 で n 方向の傾きの符号だけを見る |
| `d` | `15`（固定） | 既存の合成実験と同一 |
| 試行数 | **3** | pilot。記憶されている作業ルール「実験規模は最初は小さく、seed はまず 3」に従う |
| `L` / `num_iter` | `5` / `8`（固定） | 既存と同一。変更すると比較不能になる |
| 実装系列 | **objective-consistent**（`DualExpFamLSMConsistent`、`numerics_mode="consistent"`） | 前方向で数値整合が確認された唯一の系列。旧 0.5 / fixed とは混ぜない（KI-002） |
| **フィット数** | `1 × 1 × 7 × 2 × 3 = ` **42** | |

> **設計定数の一致は「既存結果との比較可能性」を意味しない** [重要]。
> `K_TRUE = 3` と `n = 150` を既存の合成実験に合わせるのは、
> **本 pilot 内部で 4 種のスコアを比較するときの条件を素直にするため**であって、
> 旧 0.5 系列・fixed 系列の数値と本 pilot の consistent 系列の数値を
> 並べてよいという意味ではない。並べない（root `CLAUDE.md` §3、KI-002、本ファイル §12）。
>
> **実装系列の制約** [重要]: `experimental/model_dual_expfam_consistent.py` は
> root `CLAUDE.md` §3 の `experimental/` 系列に属し、**prototype・本文採用不可**である。
> `EXPERIMENT_REGISTRY.md` 上の既存 consistent 系列の行はいずれも原稿採用 **✗** である。
> 本 pilot も同じ扱いとし、**その出力を修論本文の主張の根拠にしない**。
> 本 pilot の目的は「候補スコアの定義を確定し、`Σ_i ln det A_i^{post}` を初めて数値として見る」ことに限る。

**Smoke**: `k_est ∈ {2,3,4}`, `n = 75`, 試行 1 → **3 フィット**。

### 想定計算時間

監査の外挿（`t(n) ≈ 0.0178·n^1.387`、**d と系列の交絡あり**、§Appendix）を用いると
`n=75` の 21 fits ≈ 2.5 分、`n=150` の 21 fits ≈ 6.5 分、**合計 10 分未満**。
上振れしても 30 分以内。**overnight runner を必要としない規模である。**

---

## 5. Seeds

```
n_index    : 75 -> 0, 150 -> 1
seed_data  = 150000 + 100*n_index + trial                      # k_est に依存しない
seed_model = 151000 + 100*n_index + 10*trial + k_est
```

同一 trial 内では全 `k_est` が**同一のデータ**にフィットされる。
実行前に seed の一意性を assert し `runinfo` に記録する。

---

## 6. 記録するもの（per fit）

`run_em_experimental(..., numerics_mode="consistent", compute_strict_Q=True)` の返り値
（`model` / `Z_samples` / `F` / `sigma` / `w0` / `w` / `var_z` を含む）から **post hoc に**計算する。
**モデルコードは変更しない。**

| 列 | 定義 | 備考 |
|---|---|---|
| `q_strict`, `bic_impl`, `num_params` | 返り値そのまま | **`calc_bic_exp` を変更しない** |
| `lnpZ_det_expected` | `−(n·k/2)(1 + ln 2π)` | 解析値 |
| `lnpZ_observed` | `(1/L) Σ_l [ −(nk/2)ln(2π var_z) − (1/(2 var_z))Σ Z^{(l)2} ]` | **`L` 平均でのみ一定**（監査 §5.3.1） |
| `lnpZ_abs_err` | `\|observed − expected\|` | **gate G1** |
| `scf` | `−2(q_strict − lnpZ_det_expected) + num_params·ln n` | C2 |
| **`sum_log_det_A_post`** | `(1/L) Σ_l Σ_i ln det A_i^{post,(l)}` | **P-2 の中心量。初測定** |
| `s_laplace_post` | `scf + sum_log_det_A_post` | C3。**ELBO ではない** |
| `slogdet_sign_all_pos` | 全 `A_i` で sign が `+1` か | gate G5 |
| `min_eig_A`, `max_eig_A`, `mean_log_det_A_per_node` | `A_i^{post}` の固有値統計 | record only |
| `rank_F` | `np.linalg.matrix_rank(F, tol=1e-8)`（tol 事前固定） | record only（監査 U13） |
| `rmse_Z`, `rmse_X` | 既存 | record only |
| `mstep_q_history` | 返り値の同名キー。**`run_em_experimental(..., mstep_q_diagnostic=True)` を明示的に渡すこと**（`em_runner.py` L.95 の既定は `False` で、渡さないと空リストが返る） | 収束の程度（C-4） |
| `nan_occurred`, `nan_count`, `internal_retry`, `q_bic_failed`, `n_warnings` | 既存 | integrity |
| `numerics_mode`, `runtime_s`, `seed_data`, `seed_model` | 既存 | provenance |

### `A_i^{post}` の定義と限定（監査 §6.5.2 と同一）

```
A_i^{post,(l)} := model._calc_precision_matrix(Z_samples[:, :, l], F, sigma, var_z, w0, w, i)
```
対称化して `np.linalg.slogdet`。乱数を使わない純関数評価。

> **`A_i^{post}` は E-step 内部で実際に使われた `A_i` と同一ではない** [CONFIRMED_IN_REPOSITORY]:
> (1) `calc_eta_newton`（`reproduction/src/model.py` L.410-460）は `Z` をノードごとに
> in-place 更新する Gauss–Seidel であり、`A_i` はそのノードのその時点の位置で評価される。
> (2) `Z_samples` は E-step 終了後に `scale_Z` で大域スケール変更を受ける（`em_runner.py` L.226）。
> `scale_Z` のスケール係数 `c` はランナーがスケール前の `Z` を返さないため**復元できない**。
>
> したがって `s_laplace_post` は**代数的診断量**であり、
> 周辺尤度の下界でも、実 `q` の ELBO でもない。報告書に必ず明記する。

---

## 7. Phase 7b の成果物（記述のみ）

1. **定義表**: C1–C4 それぞれについて「何を評価するか／前提／計算手続き／解釈限界／
   呼んではいけない名前」を 1 枚にまとめる。
2. **`Σ_i ln det A_i^{post}` の記述統計**: `n`・`k` ごとの平均・分散、
   および `n(1 + ln 2π)`（決定論項の k あたり寄与）との比。
3. **argmin 一致表**: 各 trial について C1・C2・C3 の argmin `k` を並べ、
   **候補範囲の端に出た回数を別カウントする**（監査 §13.2a で旧版が犯した誤りの再発防止）。
4. **判断しないことの明記**: Phase 7b は「どのスコアが良いか」を判定しない。

---

## 8. Failure conditions / integrity gates

| ゲート | 条件 | 違反時 |
|---|---|---|
| G1 | `lnpZ_abs_err < 1e-6` が全フィット | **STOP**。`scale_Z` の挙動が監査 §5.3 の導出と異なる |
| G2 | `numerics_mode == "consistent"` が全フィット | STOP |
| G3 | `nan_occurred` の割合 | 記録。**5% 超で集計前に STOP して報告** |
| G4 | `q_bic_failed == False` が全フィット | 違反フィットを除外せず記録して報告 |
| G5 | `slogdet` の sign が全て `+1` | 違反があれば `sum_log_det_A_post` を NaN として記録し、**除外せずに**報告 |
| G6 | seed の一意性 | 実行前 assert |
| G7 | `(K_TRUE, n, trial, k_est)` の重複なし・総数一致 | 実行後 assert |
| G8 | 同一 `(K_TRUE, n, trial)` の全 `k_est` で `X`/`Y` のハッシュが一致 | 実行後 assert |

**NaN / retry**: 内部リトライは既存動作のまま、回数を記録。
**失敗フィットを救済しない。再実行しない。seed を変えない。**

---

## 9. Output paths（案）

```
tools/research_audit/run_k_selection_score_pilot.py

expfam/results/k_selection/
  k_selection_score_pilot_20260XXX_{summary,agg,selection,runinfo}.csv
figures/k_selection/
  k_selection_score_pilot_20260XXX_{log_det_A,score_curves}.png
```

**既存の CSV・figure・registry 行は一切変更しない。`EXPERIMENT_REGISTRY.md` は追記のみ。**

---

## 10. Confounders（明示）

| # | 交絡 | 扱い |
|---|---|---|
| C-1 | 生成器が `Z` を列ごとに z-score 化する（監査 §4.1）。**原論文の生成手順はこれを行わない**（監査 §5.5） | 除去できない。全条件に共通。`K_TRUE` の意味の限定として報告 |
| C-2 | 生成器が `F` の行を正規化し、Gaussian-X の X を z-score 化する。**原論文はいずれも行わない** | pilot はシナリオ A（Poisson-X）なので X の z-score は該当しない。F の行正規化は該当する |
| C-3 | `L = 5` の逐次依存連鎖（監査 §6.3） | 固定。`s_laplace_post` は実 `q` の量ではないと明記 |
| C-4 | `num_iter = 8` で収束していない（監査 P5） | 固定。`mstep_q_diagnostic=True` を渡して `mstep_q_history` を記録 |
| C-5 | **`A_i^{post}` は E-step の `A_i` と同一でない**（Gauss-Seidel ＋ `scale_Z` 後の評価）。`c` は復元不能 | §6 に明記。**モデルコード不変の制約下では解消不能な限定** |
| C-6 | 本研究の実装は `μ_x` を持たない。**原論文は `μ_x` を推定する**（Eq.11） | 記録のみ。pilot では扱わない |
| C-7 | 本研究は `var_z = 1` + `scale_Z`。**原論文は `σ_z²` を推定する**（Eq.14） | 記録のみ。等価性は未証明（監査 U8） |
| C-8 | stale-Σ（監査 E16 / U3） | pilot では構造的に発生しない（Poisson-X、かつ `em_runner` は同期する）。**Phase 7b は U3 を解決しない** |
| C-9 | 試行 3・`K_TRUE` 1 点・`n` 2 点 | **pilot であり一般化しない**。§7(4) に明記 |
| C-10 | argmin が候補範囲の端に出る可能性 | §7(3) の規約で別カウント |

---

## 11. Phase 7c の候補案（**事前登録しない。ここでは列挙のみ**）

Phase 7b で selection target が確定した場合に限り、次のいずれかを別 Issue で pre-register する。

| 候補 | 内容 | 概算 |
|---|---|---|
| **7c-1** | `n × K_TRUE` full sweep（`K_TRUE ∈ {2,3,5}` × `k_est 1..7` × `n ∈ {75,150,300}` × 10 試行、シナリオ A） | 630 fits、約 4–5 時間（外挿は d/系列交絡あり） |
| **7c-2** | 原論文 Experiment 2 設定の再現（`σ_z²` 推定・`L=10`・反復 10・正規化なし生成器・`μ_x` 推定・`k* ∈ {1,3,5,7,9}`・`k̂ = 1..10`） | **モデル/生成器の変更を伴うため別 Issue が必須**。監査 U15 の解消に必要 |
| **7c-3** | held-out 予測選択（C4）の設計と実装 | `train_mask` 設計・分割の独立性・X 側 held-out の別途検討が必要 |
| **7c-4** | stale-Σ（U3）の影響量測定 | 同期版・非同期版の対比 |

> **7c-1 を「次に実行する」とは書かない。** Phase 7b の結果が selection target を確定させてからでなければ、
> full sweep の primary estimand を定義できない。
> **7c-2 は原論文の再現であり、現行モデルの評価ではない。混同しない。**

---

## 12. 何を主張しないか

- 「ELBO 補正基準を提案する」とは書かない。C3 は ELBO ではない（監査 §6.5.1）。
- 「`s_laplace_post` は正しい基準である」とは書かない。
- 「`S_cf` は原論文 BIC の再現である」とは書かない。限定表現は `paper-Eq16-aligned diagnostic` まで。
- 「本研究の結果は原論文 Fig.3 と矛盾する」とは書かない。原論文 Experiment 2 を再現していない（監査 §7.5.4）。
- 「原論文 Eq.(26) は standard Schwarz BIC である」とは書かない（監査 §0.1、E25）。
- 「本 pilot のいずれかのスコアが観測データ周辺尤度を近似する」とは書かない。4 候補はいずれも (Q3) ではない。
- 「WBIC / sBIC が正しい」とは書かない。本 pilot は扱わない。
- 「K 選択一致性を示した」とは書かない。pilot は `K_TRUE` 1 点・`n` 2 点・3 試行である。
- 実データについては何も主張しない。合成データのみである。
- 旧 0.5 系列・fixed 系列の既存 K 選択結果と、本 pilot の consistent 系列の数値を
  **同じ表・同じ図に混在させない**（root `CLAUDE.md` §3、KI-002）。

---

## 13. 本 Issue（#35）での状態

**実行しない。** 本ファイルは次段階の提案であり、
実際の実行は別 Issue の作成と human review を経てから行う。
Issue #35 の acceptance criteria「次の実験は pre-registration draft のみで、実行していない」を満たす。

**Issue #35 の時点では、新しい Issue も作成しない。**
