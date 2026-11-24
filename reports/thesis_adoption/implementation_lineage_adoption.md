# 実装系列（lineage）の採用判定

- 作成日：2026-11-24。branch `docs/thesis-adoption-20261124`（base `d383cb5`）。
- **新しい実験・MCEM・データ生成は行っていない。** 既存のコード・CSV・canonical docs を直接読んで判定した。
- 本資料の A〜F は**この監査専用の推奨区分**である（`claim_adoption_matrix.md` の定義）。`RESEARCH_MASTER.md` の ALLOWED / QUALIFIED / NOT ALLOWED（Claim Ledger）を置き換えない。
- **採用の最終判断は研究者と指導教員が行う（Human Gate）。**

## 0. 参照の前提（GitHub で辿れるかどうか）

| commit | 内容 | remote にあるか |
|---|---|---|
| `8f6d41c` | `main`（PR #149 まで。M1・E1 を含む） | あり（origin/main） |
| `f71c04d` | AF1〜AF8-v3、d-sweep・R1/R2・checkpoint 再評価を含む | あり（origin の AF 系 branch）。main には未 merge |
| `c535910` | 総括 6 文書（`reports/research_synthesis_20261116/`）の最終監査版 | あり（origin/docs/af-research-synthesis-20261116）。main には未 merge |
| `856d475` | PR #168（MX2：誤指定の拡張） | あり（origin/experiment/misspecification-extended）。未 merge |
| `4fa6159` | PR #129（COM-Poisson を含む generic model） | あり（origin/exp/multisupport-family-selection）。未 merge |
| `59eac14`〜`d383cb5` | 追加実験の設計監査、新 Cora（CORA-SH1）の runner・監査・本実験・報告 | **local のみ（push していない）**。GitHub の permalink はまだ解決しない |

本資料のパスはすべて repository root からの相対パスである。

## 1. 判定の 3 軸

| 軸 | 問い |
|---|---|
| **A. Model scope** | その実装が表す観測モデルは、正式モデル（`AGENTS.md` §1：全 X 列で共通の 1 family、X 切片なし、スカラー w0・w）と同じか |
| **B. Evidence eligibility** | コード・推定法・数値計算が、修士論文の証拠として適切か（式の正しさ、数値の整合、provenance） |
| **C. Claim strength** | その系列の結果から、どの範囲まで言えるか |

**「正式モデルと同じ数式を使う」ことと「正式な実装系列として採用された」ことは別である。** 前者は A 軸、後者は Human Gate の決定事項。

## 2. 系列ごとの判定

| 系列 | 実体（コード） | A. Model scope | B. Evidence eligibility | C. 言える範囲 | 推奨 |
|---|---|---|---|---|---|
| **A** 原論文の印刷式 | Mikawa et al. 2024（`paper/A_study_on_latent_structural_models_for_binary_rel.pdf`） | 先行モデル（Y=Bernoulli、X=Gaussian 固定） | 一次資料。Eq.19/20/22/23 と Appendix A-1/A-3/A-5 に 1/2 が**ある**（2026-08-18 確認） | 先行研究の定式化として引用 | 関連研究・比較対象として使用 |
| **B** 旧 0.5 Python | `expfam/src/model_dual_expfam.py`（`DualExpFamLSM`）、`reproduction/src/model.py` | 正式モデルの観測モデルと同じ（全列共通 family） | **E-step の Y 側 Term3 に 0.5 が残る**（KI-001）。採用式と一致しない。η の clip あり。学会予稿の実験（Exp1–4、registry `current_main ✓`） | 「Newton 方向が全体として正しいとは断定できない」を必ず付記。先行研究再現は実装の同等性の確認にとどまる | **D**（学会予稿の歴史的記録として付録で言及。修論本文の主結果は C に置き換える案を推奨） |
| **C** fixed Python | `expfam/src/model_dual_expfam_fixed.py`（`DualExpFamLSMFixed`） | 正式モデルと同じ | **採用式（1/2 なし）と一致**。ただし η の clip と曲率の下限が残る（legacy numerics。極端な η で目的関数・score・曲率が不一致になりうる。PC-001/002 は F で測定し、C は同じコードからの推論）。pair mask なし | fixed_official Exp1–4 と実データ（Wine・旧 Cora・MovieLens）の根拠。人工データの結果は明示した設定の中で言える | **B**（正式モデルの本文採用系列として推奨。clip の限定を付記） |
| **D** masked / NB | `expfam/src/experimental/model_dual_expfam_masked.py`、`_nb.py` | masked は正式モデルと同じ（pair mask による学習対象の制限だけ）。NB は**モデル範囲外**（新 family） | masked は C を継承し、mask を勾配・精度・M-step・Q に通す（test で確認済み）。NB は prototype | strict held-out 評価の基盤。NB の結果は本文の根拠にしない | masked：E の基盤として扱う。NB：**F**（本文根拠に不可） |
| **E** objective-consistent | `expfam/src/experimental/model_dual_expfam_consistent.py` | **クラスにより異なる**（§3） | Bernoulli/Poisson の数値評価を canonical（clip なし、logaddexp）に修正した forward lineage（Issue #25）。fail_fast・seed 分離の runner（`em_runner.py`）を使う実験が多い | §3 | §3 |
| **F** per-column prototype | `expfam/src/experimental/model_dual_expfam_percolumn.py` | **モデル範囲外**（列ごとに異なる family） | legacy numerics の tail で目的関数と score が不一致（KI-015、PC-001/002） | 拡張モデルの探索的検討 | **D**（付録・今後の課題） |
| **PR #129** generic model | `4fa6159`（未 merge） | **モデル範囲外**（COM-Poisson を含む。指数型分布族の正準形でない候補を含む） | 未 merge。22 fit 失敗（CMP 候補の認証不可が中心） | AF2 以降の CMP 比較の基盤 | **D**（付録）。本文の family 一覧に CMP を入れない |
| **CORA-SH1 の BX-Newton** | `expfam/src/experimental/cora_strict_heldout.py`（`CoraConsistentFastNewton`、`036d0b5`、local） | **正式モデルと同じ観測モデル**（全列共通 family、切片なし、スカラー w0・w） | E-scalar に 2 つの推定上の変更：①精度行列の数学的に等価な高速化（`(Z*v).T@Z`、test で一致を確認）、②Bernoulli-X の F の M-step を Adam 50 step から列ごとの厳密な Newton 法に変更（Q_X は凹で、Newton は L-BFGS・有限差分と一致）。8 反復の EM は未収束 | 「現行仕様・固定予算のもとでのモデル仕様比較」（Cora 1 ネットワーク） | **C**（条件付きで本文採用を検討。Human Gate） |

## 3. lineage E の 3 層分解（一括で昇格させない）

| 層 | 中身 | 該当する実験 | 判定 |
|---|---|---|---|
| **E-1 観測モデル** | `DualExpFamLSMConsistent`（scalar）：全列共通 family、切片なし、スカラー w0・w、採用式（1/2 なし）。**数式は正式モデルと同一**。数値は canonical（clip なし）で、C より採用式の目的関数に忠実 | Phase 7e（Poisson-X）、Phase 8b、clean true-K n-sweep、EM 収束感度、d-sweep、R1/R2、CORA-SH1 | A 軸は「正式モデル」。ただし **B 軸の昇格（本文で使う実装として認めるか）は Human Gate** |
| **E-2 推定・実装上の変更** | pair mask（strict held-out）、objective-consistent numerics、fail_fast と seed 分離の runner、informed init（w0 = logit 密度、w=0.5、非 Gaussian X の F×0.2）、精度行列の高速化、BX-Newton（CORA-SH1 のみ） | 上記すべて | 数学的に等価な変更（高速化、mask の実装）は test で確認済み。**推定結果を変えうる変更**（numerics、初期化、Newton-F）は C 系列の結果と数値を混ぜない（KI-002） |
| **E-3 モデル範囲の拡張** | `DualExpFamLSMPerColumnConsistent`（列ごとの family）、自動分布選択（AF）、PR #129 の CMP | M1、E1、E2a、PR #168（MX2）、complementary blocks、latent coverage、MovieLens user-disjoint、Phase 9D〜9Y（G3/B6/P3）、#128、AF1〜AF8-v3 | **正式モデルの範囲外**。本文に入れる場合は「拡張モデルの探索的検討」として独立に扱う |

**重要（今回の一次確認）**：M1 の `protocol.json` では、正指定 arm も誤指定 arm も `family_x` が列ごとに異なる（G3/B6/P3 と G3/P6/P3）。したがって M1・E1・PR #168 の誤指定実験は、**全列共通 family の正式モデルの結果ではない**。総括資料（`reports/research_synthesis_20261116/research_overview.md` §4.1）はこれらを「本来の研究」の節に置いているが、A 軸ではモデル範囲外である（本資料では新たな主張ではなく、軸を分けた分類の結果として記す）。

## 4. 推奨（JUDGMENT。決定ではない）

1. **本文の主系列は C（fixed）**。人工データの主結果（n 依存、誤指定、K 選択の記述）は C の fixed_official で書く。B（0.5）の学会予稿の数値は、同じ表・図に入れずに「予稿時点の実装」として付録または脚注で言及する。
2. **E-scalar（E-1+E-2）を「strict held-out 評価用の実装」として限定的に昇格するか**を Human Gate で決める。昇格する場合も、C の数値と同じ表に置かない。CORA-SH1 はこの判断に依存する。
3. **E-3（per-column、AF、CMP）は正式モデルに含めない。** 誤指定の M1/E1/#168 と AF は、使う場合「拡張モデル（列ごとの family）での探索的検討」として章または付録で独立に扱う。
4. F・D-NB・PR #129 は本文の根拠に使わない（D/F 区分）。
