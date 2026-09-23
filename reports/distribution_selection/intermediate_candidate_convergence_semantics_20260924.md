# 中間 candidate convergence の意味論 — Gate 74-B4

- 日付: 2026-09-24
- Issue: #74 Gate 74-B4（parent #71 / Phase 9C）
- branch: `impl/74-family-selection-feasibility`
- authorization 時 PR #78 head: `3621341fedd6b29da76507a7f8e7dc20b8f72fb0`
- scope: **zero EM**。理論・コード意味論の監査と、観測専用の provenance 追加のみ。
  production optimizer 設定・既存 gate・既存 artifact は変更しない。

**この文書は Gate 74-B3 を通すためのものではない。** B3 は
`BLOCKED_FOR_PILOT` のまま保存される。ここで答えるのは次の 1 問である。

> 現行 Scheme-C exploration において「中間 candidate convergence」は何を意味すべきか、
> それぞれの gate はどの主張を支えるか、そして将来の Human Gate が
> **次の実行より前に**規則を選べるようにするには何を記録しておく必要があるか。

---

## Part A — 実装されている処理順序

すべて現行 Phase 9C コードパスの実コードから追跡した。

### A-1. 1 反復の処理順序

| # | 処理 | コード |
|---|---|---|
| 1 | E-step が固定 posterior samples を作る。`L` 回 `calc_eta_newton` を呼び、`scale_Z` で正規化 | `family_selection.py` `run_family_exploration`（E-step ループ → `model.scale_Z`） |
| 2 | M-step の入口で `calc_F` が呼ばれる | 同上（`F = model.calc_F(X, Z_samples)`） |
| 3 | `calc_F` が**まず** family を再選択する | `FamilySelectingPerColumnLSM.calc_F` → `self.reassign_families(self.select_families(X, Z_samples))` |
| 4 | 選択は固定 `Z_samples` 上で、候補ごとに `f_l` を最適化して score を比較する | `select_families` → `score_column_candidates(..., optimizer=self.candidate_optimizer)` |
| 5 | **family 名のみ**が install される | `reassign_families`（`_col_idx` / `family_x_list` / `family_x` を更新） |
| 6 | その後 parent の `calc_F` が **F 全体を作り直す** | `super().calc_F` → `DualExpFamLSMPerColumn.calc_F` → `_calc_F_adam_weighted`（混在列のため閉形式分岐に入らない） |
| 7 | `calc_sigma` / `calc_w0` / `calc_w`（Gaussian-Y なら `calc_sigma_y`）が続く | `run_family_exploration` の M-step |
| 8 | 更新後パラメータが次反復の E-step に入る | 次反復冒頭の `model.params.update(dict(Z=..., F=F, sigma=sigma, w0=w0, w=w))` |
| 9 | **報告される結果は exploration fit ではない。** 選択された assignment を固定した fresh refit である | `run_hybrid_family_selection` → `run_em_experimental(..., family_x_list=selected, failure_policy="fail_fast")` |

### A-2. 本監査で最も重要な事実

**候補 score を生んだ loading は、モデルに install されない。**

`select_families` は候補 family ごとに `f_l` を BFGS で最適化するが、
その最適 loading は score を計算するためだけに使われ、**破棄される**。
install されるのは family 名だけである（手順 5）。
続く `super().calc_F`（手順 6）は `self.params["F"]`（＝**前反復**の F）から出発して
`_calc_F_adam_weighted` で **F 全体**を 50 step Adam 更新する。

したがって:

- ambiguous 列 `l` についてモデルが次反復へ持ち越す `f_l` は、
  **その score を生んだ最適点ではない**。
- 候補最適化は **選択のための評価**であって、**パラメータ更新ではない**。

この事実は S3 の判定を直接決める（Part B-3）。
なお、この注意書きは `calc_F` の実装にもコメントとして残した。

### A-3. B2 migration が変えた範囲

Gate 74-B2 で BFGS に移行したのは **candidate scoring の optimizer のみ**である。
手順 6 の F 更新は引き続き **50-step Adam**（`_calc_F_adam_weighted`）であり、
B2 の承認範囲（「Phase 9C ambiguous-column candidate optimizer」）に照らして正しいが、
**exploration の実パラメータ更新は依然 Adam である**ことを明示しておく。

### A-4. 呼んではいけない名前

現行実装を **GEM / ECM / 厳密な block-coordinate ascent と呼んではならない。**

- 手順 4 の最適化対象（候補 `f_l`）と手順 6 の更新対象（F 全体）が一致しない。
- `run_family_exploration` は **Q を一度も評価しない**（ループ内に Q 計算が存在しない）。
  したがって「Q が単調に増える」という性質は**測定されていない**。
- E-step は MC sample であり、反復間で `Z_samples` が変わる。

言えるのは「固定 `q_t` の下で、列ごとの score 比較が well-posed である」までである
（`reports/distribution_selection/automatic_family_selection_design_20260923.md` §2.4 と同じ限定）。

---

## Part B — 3 つの gate 意味論（**選択肢**であり、採用しない）

### S1 — strict all-iterations convergence（現行 B3 の意味論）

```
すべての exploration 反復の、すべての ambiguous candidate が
承認済み収束規則（finite かつ final analytic grad_inf ≤ 1e-8）を満たすこと
```

**保証すること**: 各反復の family 選択が、**両候補とも停留点に達した score** の比較として
行われたこと。margin が「fitted 2 つの比較」であることが反復ごとに担保される。

**支える主張**: 「exploration path 上のどの選択も、両候補を最適化しきった score に基づく」。

**なぜ保守的か**: 選択に使う score の質だけを問い、install されるパラメータ（A-2）には触れない。
にもかかわらず全反復を要求するため、1 反復でも届かなければ全体が block される。
実際 B3 smoke-v2 では最終反復 8/8 が収束しながら中間 7/32 が未収束で block された。

**cost / failure mode**: 反復数 × ambiguous 列数 × 候補数だけ厳密最適化が要る。
posterior sample の変動で一時的に難しい問題が現れるだけで block される。
B3 smoke-v2 がまさにこの形。

**現行コード・artifact で certify できるか**: **できる。**
`selection_trace.csv` の `all_candidates_converged` が全反復について記録されており、
`pilot_convergence_gate` がそれを読む。B3 はこの規則で判定され、独立 auditor も
両テーブルから再計算して一致している。

**変更しない。**

### S2 — final-iteration-only convergence

```
中間選択は探索的とみなし、assignment を凍結して fresh refit に渡す直前の
最終 exploration 反復についてのみ、候補 score の完全収束を要求する
```

**保証すること**: 凍結される assignment が、最終反復において両候補を最適化しきった
score の比較で決まったこと。

**保証しないこと**（重要）:

1. **中間の family 選択は、その後の E-step に実際に影響する。**
   手順 5 で assignment が install され、手順 6–7 の M-step と手順 8 の次 E-step が
   その assignment の下で進む（E-step の precision `A_i` は全列 family に依存）。
   したがって最終反復に至る**経路そのもの**が中間選択の産物である。
2. **最終反復の収束は、経路を遡って正当化しない。**
   最終反復が停留点に達していても、そこへ至る軌道が未収束 score による選択を
   含んでいた事実は消えない。「最終が収束したから exploration は厳密最適化だった」
   とは言えない。
3. A-2 より、最終反復ですら **install される `f_l` は score を生んだ最適点ではない**。

**必要になる限定表現**: S2 を採る場合、報告は
「凍結された assignment は最終反復において収束した候補 score の比較で選ばれた。
ただし探索経路上の中間選択には収束が保証されていない反復が含まれる」
という形にしなければならない。「exploration が最適化として正しい」とは書けない。

**B4 では採用しない。**

### S3 — GEM 風 / 単調条件

**問い**: 完全な停留性より弱い中間条件を、**実装されている目的関数と実際の更新後
パラメータ**から正当化できるか。

「候補 score が改善した」で十分とは仮定しない。実コードを追うと:

- 候補最適化が改善するのは、その候補の `f_l` に関する列 score である。
  しかしその `f_l` は install されない（A-2）。
- install 後に実際に動くのは、手順 6 の `_calc_F_adam_weighted` による **F 全体**の
  50-step Adam 更新であり、これは固定 lr の第一次法で単調性を保証しない。
- 加えて assignment 自体が変わっているため、比較すべきは
  「同一 `Z_samples` 上での、(旧 assignment, 旧 θ) と (新 assignment, 新 θ) の Q」である。
- **`run_family_exploration` はその Q を一度も計算していない。**

したがって真に GEM 型の単調条件を書くなら、比較すべき量は次で、これは
**M-step ブロック全体の前後**でなければならない:

```
Q_before = Q_strict( X, Y, Z_samples ; assignment_t,   F_t,   sigma_t,   w0_t,   w_t   )
Q_after  = Q_strict( X, Y, Z_samples ; assignment_t+1, F_t+1, sigma_t+1, w0_t+1, w_t+1 )
条件: Q_after >= Q_before   （同一 Z_samples・同一 mask 上で評価すること）
```

`Z_samples` を跨いだ Q 比較は MC sample が異なるため単調性診断にならない
（`em_runner.run_em_experimental` の `mstep_q_diagnostic` が同じ理由で
同一 `Z_samples` に固定している）。

**判定: `NOT_CURRENTLY_CERTIFIABLE`**

現行コードにも既存 artifact にもこの量が存在しない。既存 run から遡って推定することも
できない（保存された B3 の 7 行は集約フラグのみを持つ）。

**必要になる追加 instrumentation**:

1. `run_family_exploration` の M-step ブロックを `em_runner` の `mstep_q_diagnostic` と
   同じ形で挟み、同一 `Z_samples` 上で `calc_Q_dual_strict_exp` を前後 2 回評価して
   `mstep_q_history` 相当を記録する。
2. その評価は assignment 変更を跨ぐため、`Q_before` は**旧 assignment のモデル**で
   評価する必要がある（現行 `calc_F` は評価前に `reassign_families` を済ませてしまう）。
   評価順序の設計が必要。
3. 記録した `Q_before` / `Q_after` / `Q_diff` / `decreased` を trace 行に持たせる。

**これらはいずれも新規実行を伴うため B4 では行わない。** B4 は zero EM である。

---

## Part C — claim matrix

各 gate を採用した場合に何が言えるか。
**B3 の family recovery や start 一致を、S1/S2/S3 の選択根拠に使ってはならない。**

| 主張 | S1 strict | S2 final-only | S3 monotonic |
|---|---|---|---|
| すべての exploration step で全候補 score が停留最適点にある | **YES**（規則がそれを要求し、artifact が certify する） | **NO**（中間は要求されない） | **NOT CERTIFIABLE**（S3 は停留性を要求しない設計であり、代替条件も現状測れない） |
| 最終的に選ばれた family の score が停留最適点にある | **YES** | **YES**（規則の内容そのもの） | **NOT CERTIFIABLE**（S3 は最終停留性を含意しない） |
| exploration path が厳密な coordinate ascent である | **NO** — A-2 より、score を生んだ loading は install されず、install されるのは別途 Adam 更新された F 全体 | **NO** — 同上 | **NO** — 同上。単調性が測れても厳密性は別問題 |
| exploration に検証済みの Q 単調性がある | **NO** — Q が一度も評価されていない | **NO** — 同上 | **NOT CERTIFIABLE** — まさにこれを測るのが S3 だが instrumentation が無い |
| 最終の fresh refit が通常の固定 family MCEM である | **YES** | **YES** | **YES**（gate 選択と独立。`run_em_experimental` をそのまま使う） |
| 記述的に報告してよいこと | 反復ごとの候補収束状況、選択 family、margin、start 一致、fresh refit の指標 | 同左。ただし中間非収束の存在を併記すること | 現状は S3 に基づく記述を行わない |
| 主張してはいけないこと | 「exploration が厳密最適化である」「Q 単調性がある」 | 左記に加えて「最終収束ゆえに経路も正当」「exploration が最適化として正しい」 | 「GEM である」「ECM である」「単調性が保証されている」 |

補足: どの gate でも共通して主張できないこと（#73 §8.2 の claim boundary は引き続き有効）:

- 「属性確率分布の自動選択一般を実証した」
- 「自動選択が人手指定より優れる」
- gate 決定列を分母に含めた accuracy

---

## Part D — provenance hardening（観測専用・実施済み）

`selection_trace` の各行に、その行の候補ごとの診断を追加した。

| 追加列 | 内容 |
|---|---|
| `<family>_optimizer` | その候補を最適化した optimizer 名 |
| `<family>_n_iter` | optimizer の反復数 |
| `<family>_converged` | 承認済み収束規則の判定 |
| `<family>_grad_inf` | 最終 analytic gradient infinity norm |
| `<family>_scipy_success` | SciPy の success（Adam route では空欄） |
| `<family>_scipy_status` | SciPy の status（同上） |

現行 B/P ambiguous ケースでは `bernoulli_*` / `poisson_*` の明示列になる。

**性質**:

- すべて既に計算済みの `CandidateRecord` からの射影であり、**optimizer を再実行しない**
  （test で `minimize` 呼び出し回数が候補数ちょうどであることを固定）。
- 数値経路を変えない（score / loading / selected family / margin が不変であることを test で固定）。
- 集約 `all_candidates_converged` は**残し、意味も変えない**。gate はこれだけを読む
  （`pilot_convergence_gate` が新列を参照しないことを test で固定）。
- solver message は gate に使わない。

**早速分かったこと**: 固定 B1 列で Poisson 候補は
`scipy_success = False` / `status = 2`（precision loss）を返しながら
`grad_inf = 2.2e-9 ≤ 1e-8` に到達している。
**収束規則を SciPy の success ではなく勾配に置いた判断が、実データで裏付けられた形**である。

**変更していないもの**: BFGS `maxiter=2000` / `gtol=1e-10` / analytic-jac 経路 /
収束閾値 `1e-8` / 選択規則 / margin 定義 / 現行 convergence gate / pilot progression rule。

---

## Part E — 保存された証拠

以下は一切変更・再生成していない。B3 artifact commit `3621341` 時点と同一である。

- `expfam/results/family_selection/smoke_20260923/`
- `expfam/results/family_selection/optimizer_validation_20260923/`
- `expfam/results/family_selection/optimizer_validation_analytic_jac_20260923/`
- `expfam/results/family_selection/optimizer_migration_bfgs_20260923/`
- `expfam/results/family_selection/smoke_v2_20260924/`

**B3 の中間 7 行は集約フラグのみを持つ歴史的証拠のままである。**
その勾配の大きさを推定・後埋めしていない。B4 で追加した列は**将来の run にのみ**現れる。

---

## Decision

## `B4_READY_FOR_HUMAN_SEMANTICS_DECISION`

- Part A の処理順序を実コードから確定し、**候補 loading が install されない**という
  中心的事実を特定した（A-2）。
- S1 / S2 / S3 を分離し、どれも採用していない。S1 は変更せず、S2 は採用せず、
  S3 は **`NOT_CURRENTLY_CERTIFIABLE`** として必要な instrumentation を明示した。
- GEM / ECM / 厳密 coordinate ascent とは呼んでいない（A-4）。
- provenance は観測専用で、数値経路・gate 意味論を変えていない。
- EM 実行 0。

**次に必要な Human 判断**:

1. **どの gate 意味論を採るか**（S1 維持 / S2 へ変更 / S3 のための instrumentation を先に作る）。
   B3 の recovery や start 一致を根拠にしないこと。
2. S3 を検討するなら、Part B-3 の 3 項目の instrumentation を承認するか。
   これは新規実行を伴う。
3. A-2 と A-3 を踏まえ、**exploration の F 更新が依然 50-step Adam である**ことを
   どう扱うか（B2 の承認範囲外であり、別の Human Gate）。
