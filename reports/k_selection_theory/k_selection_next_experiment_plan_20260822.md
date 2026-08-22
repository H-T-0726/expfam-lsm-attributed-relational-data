# 次実験の pre-registration draft — K 選択基準の識別実験

**作成日:** 2026-08-22（独立監査の反映: 2026-08-23）
**対象 Issue:** #35（Phase 7a）の decision gate `D: INVESTIGATE_ALTERNATIVE_CRITERION_BEFORE_K_SWEEP` に対応
**状態:** **DRAFT — 実行していない。実行しない。**
本ファイルは次 Issue の事前登録案であり、Issue #35 のスコープでは
**モデル変更・スクリプト作成・smoke・フィット・CSV 生成・図生成を一切行わない。**

前提となる監査: `reports/k_selection_theory/k_selection_theory_audit_20260822.md`（以下「監査」）

---

## 0. 用語（監査 §0 と同一）

```
lnpZ_det := −(n·k/2)·(1 + ln 2π)                     監査 §5.3
S_cf(k)  := −2·( Q_strict − lnpZ_det ) + p̂·ln n      counterfactual diagnostic score
```

> **`S_cf` を「corrected BIC」「modified BIC」「true BIC」「Schwarz BIC」と呼ばない。**
> `S_cf` は当該項の寄与を測る診断量であり、補正された基準ではない。

---

## 1. Research question

監査 §6.5 で次の**厳密な恒等式**が得られている:

```
BIC_ELBO := −2(Q_strict + Ĥ) + p̂ ln n ,   Ĥ := n(k/2)ln(2πe) − (1/2)·Σ_i ln det A_i

  ⟹   BIC_ELBO = S_cf + Σ_i ln det A_i          （厳密）
```

また監査 §13.2a により、`S_cf` は **fixed 系列 5 ケース中 4 ケースで
候補範囲内に内点最小を持たない**（k について単調減少）。したがって研究上の問いは 1 点に縮約される。

> **`Σ_i ln det A_i` は、k の増加に対して `S_cf` の減少を打ち消すほど急に増えるか。
> 言い換えれば、ELBO 補正基準は内点最小を持ち、それは `K_TRUE` と一致するか。
> またその挙動は標本サイズ n とともにどう変化するか。**

`Σ_i ln det A_i` は E-step で構成される `A_i` から得られるが、
**どの実験にも記録されていない**（監査 §13.6、U1）。本実験の役割は**この 1 つの欠落量を測ること**である。

**この実験は新しいモデルを提案しない。** モデル・E-step・M-step・`calc_bic_exp` は変更しない。

---

## 2. Competing hypotheses（3 つとも解釈可能でなければならない）

`A_i ⪰ I` より `Σ_i ln det A_i ≥ 0` なので `BIC_ELBO ≥ S_cf` は常に成り立つ（監査 §6.4-6.5）。上界はない。

| ID | 仮説 | 予測 | もし支持されたら |
|---|---|---|---|
| **H0** | `Σ_i ln det A_i` の k 依存が十分急で、ELBO 補正基準が `K_TRUE` を回復する | `P(K̂_ELBO = K_TRUE)` が高く `Δ_sel ≈ 0` または `> 0` | 現行基準の成功は決定論項に依存するが、原理的により根拠のある基準でも同じ結論が得られる。修論は ELBO 補正を併記できる |
| **H1** | `Σ_i ln det A_i` は増えるが足りず、ELBO 補正基準が過大選択する | `P(K̂_impl = K_TRUE)` は高いまま、`P(K̂_ELBO = K_TRUE)` が低下（`Δ_sel < 0`） | ELBO 補正は改善ではない。現行基準を ICL 型基準として正直に報告し、その次元罰則が完全データ事前項に由来することを明示する |
| **H2** | 現行基準が n の増加とともに壊れる | `P(K̂_impl = K_TRUE)` が n について低下し、`P(K̂_ELBO = K_TRUE)` は維持または改善（`Δ_sel > 0` が n とともに拡大） | 現行基準の n=150 での成功は漸近的な保証を与えない。修論の主張を検証した n の範囲に限定する |

**この設計は「現行 criterion が失敗することを見せる実験」ではない。**
H0・H1・H2 のいずれが出ても解釈が確定し、修論への反映内容が決まる。

補助的な問い（**primary ではない**）:

- **Q-a**: `Σ_i ln det A_i` は k・n・データの疎密でどう振る舞うか（監査 PL1 の機構仮説の直接観測）。
- **Q-b**: `rank F` は過大指定 k で欠損するか（監査 U13）。
- **Q-c**: stale-Σ（監査 E16 / U3）の影響 — **本実験では扱わない**（§13 C-7）。
- **Q-d**: `{M_K}` の非入れ子性（監査 §9.4 / U12）— **本実験では扱わない**（理論課題）。

---

## 3. Null / alternative の解釈規約（事前登録）

- `Δ_sel` の符号がどちらであっても、また 0 に近くても、**それが結果である**。
  「差が出なかったので追試する」ことはしない。
- 選択率が 1.00 で天井に張り付いた場合、**天井効果として報告し**、より大きな n の結果で議論する。
  事後に n を追加しない。
- **`K̂_ELBO` が候補範囲の端に出た場合は「範囲境界であって内点最小ではない」と明記する**
  （監査 §13.2a で旧版が犯した誤りの再発防止）。範囲の端が選ばれた回数を必ず別カウントする。
- `K_TRUE` ごとの内訳は必ず併記する。プールした値だけを報告しない。

---

## 4. Design

### 4.1 Primary（1 シナリオのみ）

| 項目 | 値 | 理由 |
|---|---|---|
| シナリオ | **A: `family_x = poisson`, `family_y = bernoulli`** | (i) Gaussian dispersion を持たないので監査 E16（stale-Σ）の交絡が構造的に存在しない、(ii) 生成器が Poisson-X を z-score 正規化しないので監査 §4.1 の X 側正規化交絡がない、(iii) Bernoulli-Y は先行研究の設定 |
| `K_TRUE` | `{2, 3, 5}` | 既存証拠は `K_TRUE = 3` に偏っている（監査 §13.1a）。2 と 5 を加えて `K_TRUE` 依存性を分離する |
| `k_est` | `1, …, 7` | すべての `K_TRUE` に対して両側（過小・過大）の候補を確保する。`K_TRUE=5` では過大側が 2 段しかない（§13 C-8） |
| `n` | `{75, 150, 300}` | 150 は既存実験と直接比較可能な点。75 と 300 で 4 倍の幅を取る |
| `d` | `15`（固定） | 既存の合成実験（`exp_scenario_lib.py` L.40）と同一 |
| 試行数 | **10**（`K_TRUE` × `n` ごと） | 既存 `exp2_bic` の 5 より増やす |
| `L` | `5`（固定） | 既存と同一。変更すると比較不能になる |
| `num_iter` | `8`（固定） | 同上 |
| 実装系列 | **objective-consistent**（`DualExpFamLSMConsistent`、`numerics_mode="consistent"`） | 前方向で数値整合が確認された唯一の系列（Issue #25/#26）。旧 0.5 / fixed とは混ぜない（KI-002） |
| フィット数 | `3 × 7 × 3 × 10 = ` **630** | |

### 4.2 Secondary（**任意**。人間が削ってよい）

| ID | 内容 | フィット数 |
|---|---|---:|
| **S1** | `n = 600` を primary と同条件（試行数 5） | `3 × 7 × 1 × 5 = 105` |
| **S2** | family 頑健性: シナリオ B（`gaussian`/`poisson`）と C（`bernoulli`/`gaussian`）を `n = 150` のみ、試行数 5 | `2 × 3 × 7 × 5 = 210` |

S2 は **Gaussian-X（B）と Gaussian-Y（C）を含むため、監査 §4.1（X の z-score）と E17（正規化定数）の
交絡を持ち込む。** したがって S2 は記述的な頑健性チェックであり、primary の判断には使わない。

### 4.3 Smoke

`K_TRUE = 3`, `k_est ∈ {2,3,4}`, `n = 75`, 試行 2 → **6 フィット**（約 1 分）。
結果を主張に使わない。

---

## 5. Seeds（衝突しない規約）

```
n_index      : 75 -> 0, 150 -> 1, 300 -> 2, 600 -> 3
scen_index   : A -> 0, B -> 1, C -> 2

seed_data  = 140000 + 10000*scen_index + 1000*K_TRUE + 100*n_index + trial
seed_model = 141000 + 10000*scen_index + 1000*K_TRUE + 100*n_index + 10*trial + k_est
```

- **データは `(scenario, K_TRUE, n, trial)` のみに依存する。** 同一 trial 内では
  すべての `k_est` が**同一のデータ**にフィットされる（対応比較の単位を作るため）。
- モデル初期化は `k_est` ごとに変える。
- 実行前に seed の一意性を assert し、`runinfo` に記録する。

---

## 6. 記録するもの（per fit）

`run_em_experimental(..., numerics_mode="consistent", compute_strict_Q=True)` の返り値
（`model` / `Z_samples` / `F` / `sigma` / `w0` / `w` / `var_z` を含む）を用いて **post hoc に**計算する。
**モデルコードは変更しない。**

### 6.1 記録量

| 列 | 定義 | 備考 |
|---|---|---|
| `q_strict` | 返り値の `Q_strict` | 既存 |
| `bic_impl` | 返り値の `bic` | 既存 `calc_bic_exp`。**変更しない** |
| `num_params` | 返り値の `num_params` | 既存 |
| `lnpZ_det_expected` | `−(n·k/2)(1 + ln 2π)` | 解析値 |
| `lnpZ_observed` | `(1/L) Σ_l [ −(nk/2)ln(2π var_z) − (1/(2 var_z))Σ Z^{(l)2} ]` | **`L` 平均でのみ一定**（監査 §5.3.1）。スライスごとには一定でない |
| `lnpZ_abs_err` | `|observed − expected|` | **integrity gate G1**（§10） |
| `scf` | `−2(q_strict − lnpZ_det_expected) + num_params·ln n` | 診断スコア（§0） |
| **`sum_log_det_A_post`** | `(1/L) Σ_l Σ_i ln det A_i^{post,(l)}` | **本実験の中心量**。定義は §6.2 |
| `H_hat_post` | `n·(k/2)·ln(2πe) − 0.5·sum_log_det_A_post` | per-node Laplace を仮定した entropy の**推定値** |
| `bic_elbo_post` | `scf + sum_log_det_A_post` | 監査 §6.5 の恒等式。**事後計算のみ。新しい criterion を実装しない** |
| `slogdet_sign_all_pos` | 全 `A_i` で `slogdet` の sign が `+1` か | integrity gate G5 |
| `min_eig_A`, `max_eig_A`, `mean_log_det_A_per_node` | `A_i^{post}` の固有値統計 | record only（Q-a） |
| `rank_F` | `np.linalg.matrix_rank(F, tol=1e-8)`（tol を事前固定） | record only（Q-b, U13） |
| `rmse_Z` | 既存の Procrustes RMSE（`k_min = min(k_est, K_TRUE)`） | record only |
| `rmse_X` | 既存 | record only |
| `mstep_q_history` | 返り値の同名キー | 収束の程度の可視化（C-4） |
| `nan_occurred`, `nan_count`, `internal_retry`, `q_bic_failed`, `n_warnings` | 既存 | integrity |
| `numerics_mode`, `runtime_s`, `seed_data`, `seed_model` | 既存 | provenance |

### 6.2 `A_i^{post}` の定義と、E-step の `A_i` との差（監査指摘の反映）

```
A_i^{post,(l)} := model._calc_precision_matrix(Z_samples[:, :, l], F, sigma, var_z, w0, w, i)
```
を対称化し `np.linalg.slogdet` を取る。乱数は使わない純粋な関数評価である。

> **重要な限定**: `A_i^{post}` は **E-step 内部で実際に使われた `A_i` と同一ではない。**
> 理由は 2 つある [CONFIRMED_IN_REPOSITORY]:
> 1. `calc_eta_newton`（`reproduction/src/model.py` L.410-460）は `Z` をノードごとに
>    **in-place で更新する Gauss-Seidel** であり、`A_i` はそのノードのその時点の位置で評価される。
> 2. `Z_samples` は E-step 終了後に `scale_Z` で**大域スケール変更**を受ける
>    （`em_runner.py` L.226）。`A_i` は `Z` に非線形に依存するため、スケール後の値は別の行列になる。
>
> したがって `H_hat_post` は「実際に使われた `q` の entropy」ではなく、
> **最終サンプルにおける per-node Laplace 曲率から構成した推定量**である。
> `bic_elbo_post` も同様に推定量であり、監査 §6.5 の恒等式は
> `A_i` の取り方を固定したうえで成り立つ代数的関係である。
> **モデルコードを変更せずに得られるのはこの版だけであり、その限定を報告書に明記する。**
>
> なお `scale_Z` のスケール係数 `c` は、`run_em_experimental` が
> スケール前の `Z` を返さないため**復元できない**。C-5 参照。

---

## 7. 比較する score（3 本、同一フィットの上で）

| ID | 定義 | 位置づけ |
|---|---|---|
| `C1` | `K̂_impl = argmin_k bic_impl` | **現行基準。実装・値ともに変更しない** |
| `C2` | `K̂_ELBO = argmin_k bic_elbo_post` | 事後計算による**推定量**（§6.2 の限定つき） |
| `C3` | `K̂_cf = argmin_k scf` | **診断スコア**（§0）。基準ではない。record only |

**表題は「selection criteria」ではなく「比較する score」とする。**
`C3` は criterion ではなく diagnostic であり、`C2` は推定量である。

**held-out 予測基準は本実験に含めない**（`train_mask` の設計が別の事前登録を要するため）。
意図的なスコープ制限であり、監査 §14.1 に将来課題として記録済み。

---

## 8. Primary estimand（**ちょうど 1 つ**）

```
Δ_sel(300) = P(K̂_ELBO = K_TRUE | n = 300)  −  P(K̂_impl = K_TRUE | n = 300)
```

- **trial unit = `(K_TRUE, trial)`**。`n = 300` の primary シナリオで `3 × 10 = 30` 個。
- 各 trial について、その trial のデータに対する `k_est = 1..7` の 7 フィットから argmin を取る。
- **`n = 300` を選ぶ理由**: primary の中で最大の n であり、監査 §8.4 のオーダー競合が最も表面化する点。
- `n = 75` と `n = 150` の `Δ_sel` は **secondary**（同じ式、同じ trial unit）。

### 対応比較の単位

`C1` と `C2` は**同一のフィット**から計算されるので trial ごとに対応がある。
trial 単位の 2×2 分割表（両方正解 / C1 のみ / C2 のみ / 両方不正解）を必ず報告する。

---

## 9. Decision rule（事前登録）

1. **主報告は記述的**: 各 `n`・各 `K_TRUE`・各 score について `P(K̂ = K_TRUE)` を
   **カウント（x/10）と、プールしたカウント（x/30）で報告する。**
   **加えて、`K̂` が候補範囲の端（1 または 7）になった回数を別途カウントする**（§3）。
2. `Δ_sel(300)` を、対応する 2×2 分割表とともに報告する。
3. **統計的検定・信頼区間は secondary かつ任意**。
   本実験の反復は **`#33` の repeated split とは異なり、独立な生成器ドローによる独立反復である**
   （§5 の seed 規約により各 trial のデータは独立に生成される）。
   exact binomial / McNemar は原理的に適用可能だが、
   **事前登録では descriptive を primary とし、推測統計は人間の判断で追加する**。
   自動で p 値を計算して主張に使わない。
4. 判定:
   - `Δ_sel(300)` の絶対値が **1/30 以下**（≤ 0.033）で、かつ n=75/150 でも符号が一貫しない
     → **H0 を採用**（差なし）。
   - `Δ_sel(300) < 0` が n=150 と n=300 の両方で観測される → **H1**。
   - `P(K̂_impl = K_TRUE)` が n について単調減少し、かつ `Δ_sel(300) > 0` → **H2**。
   - いずれにも当てはまらない → **「判定不能」と明記して報告する。** 追加実験を自動で計画しない。
5. `sum_log_det_A_post` の k 依存性（Q-a）は、判定とは独立に**記述的に**報告する。
   特に `Σ_i ln det A_i` の k あたり増分を `n(1 + ln 2π)` と並べて示す。

---

## 10. Failure conditions / integrity gates

実行前に事前登録し、事後に緩めない。

| ゲート | 条件 | 違反時の動作 |
|---|---|---|
| G1 | `lnpZ_abs_err < 1e-6` が **全フィット**で成立 | **STOP**。`scale_Z` の挙動が監査 §5.3 の導出と異なることを意味するので、集計せず報告する |
| G2 | `numerics_mode == "consistent"` が全フィット | STOP |
| G3 | `nan_occurred` の割合 | 記録する。**5% を超えたら集計前に STOP して報告** |
| G4 | `q_bic_failed == False` が全フィット | 違反フィットを除外せず、そのまま記録して報告する |
| G5 | `slogdet` の sign が全て `+1`（`A_i` が正定値） | 違反があれば `sum_log_det_A_post` を NaN として記録し、その fit を**除外せずに**報告する |
| G6 | seed の一意性 | 実行前 assert。違反なら STOP |
| G7 | `(scenario, K_TRUE, n, trial, k_est)` の重複なし・総数一致 | 実行後 assert |
| G8 | 同一 `(scenario, K_TRUE, n, trial)` の全 `k_est` で `X`/`Y` のハッシュが一致 | 実行後 assert（データ共有の検証） |

### NaN / retry ポリシー

- `run_em_experimental` の内部リトライは **既存動作のまま**。回数を `internal_retry` に記録する。
- **失敗したフィットを救済しない。再実行しない。seed を変えない。**
- 中断（quota・電源等）からの再開時は、**既に完了した部分を再実行しない**。
  部分結果が壊れている場合は、**その旨を報告して人間の判断を仰ぐ**。

---

## 11. Compute budget（既存実行からの外挿のみ）

### 実測アンカー（一次 runinfo / summary）

| 出典 | n | d | 系列 | k | フィットあたり秒 |
|---|---:|---:|---|---:|---:|
| `story_diagnostics/complementary_blocks_consistent_20260821_runinfo.csv`（941.2 s / 120 fits） | 80 | 9 | objective-consistent per-column | 3 | 7.84 |
| `story_diagnostics/matched_latent_coverage_ablation_20260821_runinfo.csv`（1384.3 s / 180 fits） | 80 | 9 | objective-consistent per-column | 3 | 7.69 |
| `real_data/movielens_userdisjoint/..._summary.csv`（`runtime_s` 平均） | 100 | 可変 | objective-consistent per-column | 3 | 10.92 |
| `real_data/cora_balanced_k_sweep/..._agg.csv`（`runtime_mean`, k=3） | 280 | 50 | **fixed** | 3 | 44.15 |

### 外挿モデルと、その**明示的な弱点**

2 点（n=80, 7.77 s）と（n=280, 44.15 s）から `t(n) ≈ 0.0178 · n^1.387` 秒。
`n = 100` の予測 10.60 s に対し実測 10.92 s（誤差 3%）。

> **この外挿は 3 つの交絡を含む**（監査指摘）:
> 1. **`d` が違う**（9 vs 50）。本実験の目標は `d = 15` である。
> 2. **実装系列が違う**（objective-consistent per-column vs **fixed**）。
> 3. **train/test mask の有無が違う**。
>
> `n` のべき乗則にこれらが吸収されているため、**`n = 300` の 48.6 s/fit は過大見積もりの可能性が高い**。
> これは**計画上の見積もりであり、科学的主張には一切影響しない**。

`k` 依存性は Cora の `runtime_mean`（k=1: 32.3、k=3: 44.2、k=6: 42.0）から
`k ≥ 2` でほぼ一定とみなし係数 1.0 とする。**これも近似である。**

| ブロック | フィット数 | 予測秒/フィット | 予測合計 |
|---|---:|---:|---:|
| smoke | 6 | 7.1 | 約 1 分 |
| primary `n=75` | 210 | 7.1 | 約 25 分 |
| primary `n=150` | 210 | 18.6 | 約 65 分 |
| primary `n=300` | 210 | 48.6 | 約 170 分 |
| **primary 計** | **630** | — | **約 4.3 時間（上振れ見積もり）** |
| S1 `n=600`（任意） | 105 | 127.0 | 約 3.7 時間 |
| S2 family 頑健性（任意） | 210 | 18.6 | 約 1.1 時間 |
| **全部込み** | **951** | — | **約 9.1 時間** |

`sum_log_det_A_post` の post hoc 計算コストは 1 フィットあたり `n·L` 回の `k×k` の `slogdet`。
`n=300, k=7, L=5` でも 1 秒未満と見積もる（**実測アンカーなし。見積もりである**）。

**人間が削る場合の推奨順序**: S1 → S2 → `n=300` の試行数を 10 → 5。
`n` の 3 点と `K_TRUE` の 3 点は削らない（本実験の識別力の源だから）。

---

## 12. Output paths（案）

```
tools/research_audit/run_k_selection_criterion_comparison.py     （新規スクリプト）

expfam/results/k_selection/
  k_selection_criterion_20260XXX_summary.csv     （fit 単位、全列）
  k_selection_criterion_20260XXX_agg.csv         （(scenario,K_TRUE,n,k_est) 集計）
  k_selection_criterion_20260XXX_selection.csv   （trial 単位の K̂ と正誤、3 score、範囲端フラグ）
  k_selection_criterion_20260XXX_paired.csv      （C1 vs C2 の 2×2 分割表）
  k_selection_criterion_20260XXX_runinfo.csv
figures/k_selection/
  k_selection_criterion_20260XXX_{selection_rate,log_det_A,score_decomposition}.png
```

`runinfo` に記録する項目:
`script, datetime, git_head, branch, issue, scenario_list, K_TRUE_list, k_est_list, n_list,
d, L, num_iter, n_trials, seed_data_base, seed_model_base, numerics_mode, model_class,
lnpZ_tol, rank_F_tol, total_fits, total_runtime_s, n_nan, n_internal_retry, gate_results`

**既存の CSV・figure・registry 行は一切変更しない。`EXPERIMENT_REGISTRY.md` は追記のみ。**

---

## 13. Confounders（明示）

| # | 交絡 | 扱い |
|---|---|---|
| C-1 | 生成器が `Z` を列ごとに z-score 化する（監査 §4.1） | **除去できない**。全条件に共通なので条件間比較には影響しないが、`K_TRUE` の意味の限定として報告する |
| C-2 | Gaussian-X の X も z-score 化される | **primary はシナリオ A（Poisson-X）なので該当しない**。S2 のシナリオ B にのみ該当 |
| C-3 | `L = 5` の逐次依存連鎖（監査 §6.3） | 固定。`H_hat_post` は per-node Laplace を仮定した推定量であり、実際の `q` の entropy ではない。**補正しない**。報告書に明記する |
| C-4 | `num_iter = 8` で収束していない（監査 P5） | 固定。`mstep_q_history` を記録して収束の程度を可視化する |
| C-5 | **`A_i^{post}` は E-step の `A_i` と同一でない**（Gauss-Seidel の in-place 更新 ＋ `scale_Z` 後の評価） | §6.2 に明記。`scale_Z` のスケール係数 `c` は返り値から**復元できない**ため記録できない。**モデルコードを変更しない制約下では解消不能な限定**として報告する |
| C-6 | `k_est` の増加は runtime も変える | 記録のみ。判断には使わない |
| C-7 | stale-Σ（監査 E16 / U3） | **primary では構造的に発生しない**（Poisson-X）。`em_runner` は L.285 で params を同期しているので S2 でも該当しない。**したがって本実験は U3 を解決しない** |
| C-8 | `K_TRUE = 5` では `k_est` の上限 7 が過大側で 2 段しかない | 設計上の非対称。`K_TRUE` 別に報告することで可視化する |
| C-9 | 選択率の天井効果 | §3 の規約に従い天井として報告する |
| C-10 | `K̂` が候補範囲の端に出る可能性 | §3・§9(1) の規約に従い、**内点最小でないことを明記し別カウントする** |

---

## 14. 何を主張しないか

- 「ELBO 補正基準を提案する」とは書かない。本実験は**既存量の記録と事後計算**である。
- 「`bic_elbo_post` は正しい基準である」とは書かない。§6.2 の限定つきの推定量である。
- 「WBIC / sBIC が正しい」とは書かない。本実験はそれらを扱わない。
- 「K 選択一致性を示した」とは書かない。有限の n 3 点（+任意で 4 点）の観測である。
- 実データについては何も主張しない。本実験は合成データのみである。
- `S_cf` を「補正された BIC」と呼ばない（§0）。
- 旧 0.5 系列・fixed 系列の既存 K 選択結果と、本実験の consistent 系列の数値を
  **同じ表・同じ図に混在させない**（root `CLAUDE.md` §3、KI-002）。

---

## 15. 本 Issue（#35）での状態

**実行しない。** 本ファイルは pre-registration draft であり、
実際の実行は別 Issue の作成と human review を経てから行う。
Issue #35 の acceptance criteria「次の実験は pre-registration draft のみで、実行していない」を満たす。
