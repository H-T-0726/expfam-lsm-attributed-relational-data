# 潜在次元 K 選択・識別可能性・Q-based criterion の理論監査

**作成日:** 2026-08-22（独立監査の反映: 2026-08-23）
**対象 Issue:** #35（Phase 7a）
**ブランチ:** `audit/35-k-selection-theory-audit`（base: `main` = `be1a74cdfa122bd992493f8e7dfb17b4bf3aa72e`、working tree clean）
**種別:** READ-ONLY 理論監査。**コード変更なし・実験実行なし・既存 CSV / figure / historical report の変更なし。**
新規作成は本ファイルと `k_selection_next_experiment_plan_20260822.md` の 2 件のみ。

本監査における「再計算」とは、**既にコミット済みの一次 CSV の数値を読み込んで恒等式的に分解した**ことのみを指す。
モデルの再フィットは一切行っていない。

**独立監査:** 本レポートは `.claude/agents/research-auditor.md` による独立監査を受け、
17 件の finding をすべて反映済みである。反映内容は §21 に記録する。

---

## 0. 用語の固定（最重要）

本レポートで繰り返し使う次の量に、**専用の名前を与え、他の名前で呼ばない**。

```
lnpZ_det      := −(n·k/2)·(1 + ln 2π)
                 scale_Z 適用後の Z サンプルで評価した Z 事前分布項の値（§5.3）

counterfactual diagnostic score
   S_cf(k)   := −2·( Q_strict − lnpZ_det ) + p̂·ln n
                 Q_strict から lnpZ_det だけを取り除いた診断量
```

> **`S_cf` を「corrected BIC」「modified BIC」「true BIC」「Schwarz BIC」と呼んではならない。**
> `S_cf` は**この項が選択結果にどれだけ効いたかを測るためだけの診断スコア**であり、
> 何らかの意味で「正しい」基準ではない。周辺尤度にも ELBO にも対応しない
> （ELBO との正確な関係は §6.5）。

---

## 1. Executive Summary

1. **`scale_Z` により、`Q` に入る Z 事前分布項が決定論的な値になる**
   [DERIVED]+[CONFIRMED_IN_REPOSITORY]。
   `scale_Z`（`reproduction/src/model.py` L.468-504）は Monte Carlo サンプル配列**全体**の平均二乗を 1 へ正規化する。
   `var_z = 1` の下で `(1/L)Σ_l ln p(Z^{(l)}) = −(nk/2)(1 + ln 2π) = lnpZ_det` となり、
   **データにも θ にも依存しない**。BIC 換算では潜在次元 1 あたり
   `n(1 + ln 2π) ≈ 2.8379 n` の固定加算である。
   同じ構造は先行研究の MATLAB (`Mato Lab Program/calcdescmetric_ver4.m`) にも存在する（§5.5）。
2. **この項だけを取り除いた診断スコア `S_cf` は、fixed 系列の 5 ケース中 4 ケースで
   検討範囲内に内点最小を持たない**（k について単調減少）[DERIVED（一次 CSV からの恒等分解）]。
   合成 A/B/C（k=1..9）と Cora（k=1..6）は範囲の上端が argmin になり、Wine（k=1..9）のみ
   内点 k=5 に最小を持つ。すなわち `S_cf` の argmin として報告できる「6」「9」は**範囲境界**であって最適点ではない。
   **限定**: これは **fixed 系列**の結果である。**学会予稿が依拠する旧 0.5 系列**では
   シナリオ A と C が `S_cf` でも内点 k=3 を保つ（§13.2b）。
   したがって「既存の k\*=3 という記述は Schwarz 型ペナルティの結果ではない」と**無条件には言えない**。
3. **Cora の k=1 選択（KI-011）と合成データの k=3 選択には、共通して同じ決定論項が関与している**
   [EMPIRICAL]。Cora（n=280, d=50, **3 試行**）では、
   当てはまり改善 `2·Δfit(1→2) = 900.68` に対して
   決定論項 `794.61` とパラメータ数罰則 `276.11` の和 `1070.71` が**上回るため k=1 が選ばれる**。
   **「両者は同一の機構である」とは主張しない**。n 以外に d・family・密度・試行数・
   当てはまり曲線の形状が同時に異なり、機構同定は行っていない。
   到達しているのは「**KI-011 と整合する重要な機構を特定した**」までである（§13.3）。
4. **`Q` と周辺尤度の差は entropy だけではない。KL 項が残る** [DERIVED]。
   正確な恒等式は `ln p(D|θ) = Q(θ;q) + H(q) + KL(q ‖ p(Z|D,θ))`。
   本実装では (i) Laplace 近似、(ii) `q` が 1 M-step 前の θ に基づく、(iii) `scale_Z` による押し出し、
   の 3 つの理由で KL は消えない。
   `reports/theory_audit/theory_audit_report_20260718.md` は **§7 本文 L.278 では KL を正しく書いているが、
   §1 エグゼクティブサマリー第 4 項（L.37-42）は「エントロピー項 H(q) の分だけ」と書いており、
   同一文書内で不整合である** [CONFIRMED_IN_REPOSITORY]。歴史的記録として当該ファイルは変更しない。
   **なお同文書 L.282-287 は既に「k を 1 増やすごとに約 2.84n の追加ペナルティ」という同じ大きさの数値に到達している。**
   本監査の新規性は数値の大きさではなく、**その項が `scale_Z` により厳密に決定論的である**という点にある（§6.2）。
5. **新発見（証明はスケッチ）: 本モデルの族 `{M_K}` は K について入れ子（nested）ではないと考えられる** [DERIVED・部分]。
   `w` がスカラーで全潜在次元に共有されているため、K+1 次元モデルで F の余分な列を 0 にしても
   `η_Y = w0 + w(z_i·z_j + u_i u_j)` の余分項が残る。
   **ただし現時点で示したのはこの特定の埋め込みが失敗することであり、
   `M_{K+1}` の全パラメータにわたる非包含は示していない**（§9.4）。
6. **新発見: `calc_Q_dual` は引数 `sigma` を本文中で一切使っていない** [CONFIRMED_IN_REPOSITORY]。
   X 側対数尤度は `model.params["sigma"]` を読むため、`utils_expfam.run_em_dual` 経由の実験
   （**旧 0.5 系列**）では `Q_strict` の Gaussian-X 部分が **1 M-step 古い Σ** で評価されている。
   fixed 系列の実験スクリプトと `em_runner.py` は明示同期しているため該当しない。
   **数値への影響量は未測定** [UNRESOLVED]。
7. **新発見: 先行研究の MATLAB で `BIC` という名前を持つ唯一の量は、joint モデルの `Q` ではない**
   [CONFIRMED_IN_REPOSITORY]。`Mato Lab Program/DecideNumFactor.m` L.13-14 は
   `factoran`（**X のみの古典的探索的因子分析**）の周辺対数尤度に
   `t = (i+1)*d − 0.5*i*(i−1)` を掛けた罰則を付す。
   これは現行 Python の `−2·Q_strict + p̂ ln n` とは**量のレベルで別物**である（§7.5）。
8. **「異種 family だから BIC が使えない」は誤りである** [DERIVED]。
   同一 family 割当のもとでの k 比較には支障がない。致命的なのは
   **異なる支配測度をまたぐ family 間の尤度値比較**であり、一次 CSV で直接観測できる
   （`per_column_demo_summary.csv`: 台違反の `all_bernoulli` が最小 BIC 9672.73 かつ最悪の rmse_Z 0.5919）
   [EMPIRICALLY_OBSERVED]。
9. **K_TRUE を振った選択率は、旧 0.5 系列に既に存在する** [CONFIRMED_IN_REPOSITORY]。
   `exp2_bic_{A,B,C}.csv`（n=150, d=15, K_TRUE ∈ {1,3,5,7,9}, k_est=1..10, 5 試行）で
   15 セル中 13 セルが `P(K̂=K_TRUE)=1.00`。低下するのは**シナリオ B のみ**。
10. **n を振った K 選択は一度も行われていない** [CONFIRMED_IN_REPOSITORY]（§13.5）。
11. **決定ゲート: `D: INVESTIGATE_ALTERNATIVE_CRITERION_BEFORE_K_SWEEP`**（次点 `A`）。§20。

---

## 2. Scope / non-goals

### 対象
- `calc_bic_dual` / `calc_Q_dual` / `calc_Q_dual_strict`（`expfam/src/utils_expfam.py`）
- `calc_bic_exp` / `calc_Q_dual_strict_exp`（`expfam/src/experimental/eval_utils.py`）
- E-step の勾配・precision（`model_expfam.py` / `model_dual_expfam.py` / `_fixed.py` /
  `experimental/model_dual_expfam_percolumn.py` / `_masked.py` / `_consistent.py`）
- `scale_Z` / `calc_F` / `calc_sigma` / `calc_eta_newton`（`reproduction/src/model.py`）
- 生成器（`expfam/src/data_generator_expfam.py`）
- **先行研究の MATLAB 実装**（`Mato Lab Program/*.m`）
- 既存 k 選択結果 CSV（合成 fixed / 合成旧 0.5 / Wine / Cora / MovieLens / per-column prototype）

### 非対象（本 Issue では行わない）
- モデル実装・`calc_bic_dual` の変更
- 新しい criterion の実装・実行
- 実験の新規実行・既存実験の再実行
- 過去 CSV / runinfo / figure / historical report の変更
- canonical docs（`RESEARCH_MASTER.md` / `KNOWN_ISSUES.md` / `CLAUDE.md` / `EXPERIMENT_REGISTRY.md`）の更新
- per-column prototype の本文採用可否の判断
- 一致性定理の完全証明 [OUT_OF_SCOPE]

### 本監査が読めなかった一次資料
- `paper/A_study_on_latent_structural_models_for_binary_rel.pdf` は本環境で**機械読み取り不可**
  （PDF 抽出ライブラリ未導入。環境変更は本 Issue のスコープ外）。
  原論文に関する記述はすべて **研究者本人による一次確認の伝聞**であり、
  `[UNVERIFIED_IN_REPOSITORY]` として明示する。
  この扱いは `docs/math_notes/half_factor_primary_source_confirmation_20260818.md` §3 の先例に従う。

---

## 3. Evidence labels

| ラベル | 意味 |
|---|---|
| `[CONFIRMED_IN_REPOSITORY]` | リポジトリ内のコード行・CSV 値を直接確認した |
| `[SUPPORTED_BY_PRIMARY_SOURCE]` | 外部一次文献（原論文・出版社ページ）で確認した |
| `[UNVERIFIED_IN_REPOSITORY]` | 研究者本人の一次確認に基づくが、リポジトリ内では検証できない |
| `[DERIVED]` | 本監査で導出した（導出過程を本文に明記） |
| `[EMPIRICALLY_OBSERVED]` | 既存実験の一次 CSV に現れている現象 |
| `[PLAUSIBLE]` | もっともらしいが検証していない |
| `[UNRESOLVED]` | 未解決 |
| `[CONTRADICTED]` | 既存記述と矛盾を検出した |
| `[OUT_OF_SCOPE]` | 本監査の範囲外 |

主張の強度は次の 4 段階で区別する。**混ぜない。**

| 段階 | 意味 |
|---|---|
| **FACT** | 一次コード / CSV が直接示すこと |
| **EMPIRICAL** | 調べた既存ケースで観測されたこと（ケース数と試行数を必ず併記） |
| **INFERENCE** | そこから合理的に言えること |
| **NOT YET PROVED** | まだ示していないこと |

---

## 4. RQ1 — 「真の潜在次元 K」とは何か

### 4.1 (A) 人工データ / realizable case

生成器 `generate_dual_data`（`data_generator_expfam.py` L.223-345）[CONFIRMED_IN_REPOSITORY]:

```
L.282  Z = rng.normal(0, 1, (n, k))
L.283  Z = normalize_zscore(Z, axis=0)          # 列ごとに標本平均 0・標本 sd 1（ddof=0）へ強制
L.286-289  F ~ N(0, var_f);  各行を ||f_j|| = sqrt(1 - sigma_x[j,j]) へ再スケール
L.298      Gaussian-X: X = Z F^T + noise;  X = normalize_zscore(X, axis=0)
L.300-307  Bernoulli-X / Poisson-X: 正規化なし
```

したがって:

- **返される `true_params["Z"]` は宣言モデル `z_i ~ iid N(0, I_k)` からの標本ではない**。
  列ごとの標本モーメントが厳密に固定され、行間に依存が入る（ずれは各モーメントで `O(1/√n)`）。
- **Gaussian-X の場合、`true_params["F"]` と `true_params["sigma"]` は、
  返された X を実際に生成した分布のパラメータと厳密には一致しない** [DERIVED]。
  列ごとの z-score `X_norm = (X − m) D^{-1}` により、実効負荷は `F_j / sd_j`、
  実効ノイズ分散は `σ_j² / sd_j²` になる。設計上 `Var(X_j) ≈ 1` なので歪みは小さいが、ゼロではない。
- `sigma_x_true` 引数は宣言されているが本文で使われていない（実効ノイズ共分散は `uniq`）
  [CONFIRMED_IN_REPOSITORY: L.233 のみに出現]。

**それでも `K_TRUE` は well-defined である** [DERIVED]。
列ごとの z-score は各列に対するアフィン変換であり、`Z ↦ E[η_X]` の**階数**も
`η_Y = w0 + w z_i^T z_j` の内積の次元も変えない。すなわち

> **`K_TRUE` = 「潜在座標から自然パラメータへの線形写像の階数」かつ「Y の内積が取られる空間の次元」**

という**構造的な量**として定義でき、正規化はこれを保存する。
一方、`true_params` を「真のパラメータ値」として RMSE の基準に使うことは、上記のずれの範囲でのみ正当化される。

なお `normalize_zscore` は `ddof=0`（`reproduction/src/data_generator.py` L.39）なので
`mean(Z²) = 1` が厳密に成り立ち、**真の Z も `scale_Z` の制約を厳密に満たす** [DERIVED]。

### 4.2 (B) 実データ / realizable case

`K0 = min { K : p0(X,Y) ∈ M_K }` を使うには [DERIVED]:

1. `p0` がいずれかの `M_K` に属すること（realizable）。
2. `{M_K}` が K について入れ子であること。**本モデルではこれが成り立たないと考えられる**（§9.4）。
3. `min` の一意性。

**実データでは 1 が成り立つ保証がない** [UNRESOLVED]。
Wine で「BIC 最小 k=3 が真のクラス数 3 と一致した」（`KNOWN_ISSUES.md` KI-006）は、
クラス数と潜在次元が一致すべき理論的根拠を持たない。Wine の Y はクラスラベル由来であり、
**k=2 の時点で 5 試行すべて `auc_y = 1.0`** である（`wine_bic_k1to9.csv`）[CONFIRMED_IN_REPOSITORY]。
これは「一致した」という**観測**であって `K0` の推定ではない。

### 4.3 (C) Misspecified / unrealizable case

| 概念 | 定義 | 本モデルで使えるか |
|---|---|---|
| `K0`（realizable） | `min{K : p0 ∈ M_K}` | 実データでは前提未検証 [UNRESOLVED] |
| pseudo-true K | `argmin_K min_{θ∈Θ_K} KL(p0 ‖ p_{K,θ})` に対応する最小の K | 定義可能だが、入れ子でないため「最小の」が自明でない [DERIVED] |
| predictive-optimal K | held-out 予測リスクを最小化する K | 定義可能・測定可能。masked lineage で実装済み |
| model-selection target | criterion が argmin として返す K（`K̂`） | 定義は明確だが、上の 3 つのどれとも一致が保証されない [DERIVED] |

Watanabe (2013) の WBIC は unrealizable でも Bayes free energy と同じ漸近展開を持つ
[SUPPORTED_BY_PRIMARY_SOURCE]。ただし逆温度 `1/ln n` の事後サンプリングを要し、現行枠組みの外にある。

### 4.4 (D) Z と K — 回転不定性は K の非識別性を意味しない

| 対象 | 識別されるか |
|---|---|
| 「真の Z そのもの」 | **されない**。`Z ↦ ZR`, `F ↦ FR` (`R ∈ O(k)`) は観測分布を変えない |
| 「直交同値類としての潜在空間」（`ZZ^T`） | **される**（一般の位置で） |
| 「潜在次元 K」 | **回転不定性からは何も帰結しない** |

`O(k)` は `k` 次元空間**の内部**の変換であり、次元そのものを動かさない [DERIVED]。
したがって **「回転で識別できないから K も識別できない」は誤った推論である。**
Procrustes 評価（`utils_expfam.procrustes_rotation` L.38-43）はこの不定性を処理しており、K の識別性とは別問題。

K の識別性を脅かすのは回転ではなく、
(i) `w = 0` で Y が Z から独立になる点、(ii) `F` が階数欠損になる点、(iii) 入れ子性の破れ（§10）。

---

## 5. RQ2 — 現行 Q-based criterion をコードから再導出する

### 5.1 式とコード行の 1 対 1 対応

```
BIC_impl = −2 · Q_strict + p̂ · ln(n)                       utils_expfam.py L.403
p̂        = kd − k(k−1)//2                                   L.399  f_params
           + d · 1{family_x = gaussian}                     L.400  sigma_x_p
           + 1 · 1{family_y = gaussian}                     L.401  sigma_y_p
```

```
Q_strict = Q_dual + corr                                    L.373-378
Q_dual   = (1/L) Σ_{l=1}^{L} [ lnpZ(Z^{(l)}) + lnpX(X|Z^{(l)},F) + lnpY(Y|Z^{(l)},w0,w) ]
                                                            L.344-351
lnpZ     = −(nk/2)·ln(2π·var_z) − (1/(2 var_z))·Σ Z^{(l)2}  L.315-321  (_lnpZ)
lnpX     = model.calc_log_likelihood_X(X, Z^{(l)}, F)       L.348 → model_dual_expfam.py L.291-334
lnpY     = model.calc_log_likelihood_Y(Y, Z^{(l)}, w0, w)   L.349 → model_expfam.py L.241-269
corr     = −Σ_{i<j} ln(y_ij!)  if family_y = poisson        L.375-376
           −Σ_{il}  ln(x_il!)  if family_x = poisson        L.377-378
```

`var_z` は `initialize_params` で `1.0` に固定され（`reproduction/src/model.py` L.99、注釈 L.98）、
M-step で更新されない [CONFIRMED_IN_REPOSITORY: 全ソースを網羅 grep して更新箇所なし。
`model.py` L.818 は自己テストの assert]。

### 5.2 [CONFIRMED_IN_REPOSITORY] `calc_Q_dual` は引数 `sigma` を使っていない

`utils_expfam.py` L.324-352 の本体を機械的に走査した結果、識別子 `sigma` は**シグネチャの L.329 にのみ**現れ、
本文中で一度も参照されていない。X 側対数尤度は `model.calc_log_likelihood_X` が
`self.params["sigma"]`（`model_dual_expfam.py` L.315）を読む。

`run_em_dual` では `model.params.update(dict(..., sigma=sigma, ...))` が
**E-step ループ内部の L.498** でのみ行われ、M-step は L.521 でローカル `sigma` を更新する。
`Q_strict` は最終ループ後 L.581 で計算されるため、
**X 側は `Σ^(T−1)`、F は `F^(T)` という混在した θ で評価されている** [DERIVED]。

| 系列 / スクリプト | 同期 | 該当 |
|---|---|---|
| `utils_expfam.run_em_dual`（**旧 0.5 系列**: `exp_scenario_lib.py`, `run_exp2_bic_v2.py`） | **なし** | Gaussian-X のときのみ実害 |
| `run_fixed_official_exp1_bic_full.py` L.156 | あり | 影響なし |
| `run_fixed_real_wine_pilot.py` L.218 | あり | 影響なし |
| `run_fixed_real_cora_balanced_k_sweep.py` L.261 | あり | 影響なし |
| `experimental/em_runner.py` L.285 | あり | 影響なし |

**数値への影響量は未測定** [UNRESOLVED]。関連する経験的観測は §13.3b。

### 5.3 [DERIVED] `scale_Z` により `lnpZ` は決定論的な値になる

`scale_Z`（`reproduction/src/model.py` **L.468-504**）は `Z_samples` 配列**全体**の平均二乗
`avg_sq = np.mean(Z_samples**2)`（軸指定なし＝全要素）を計算し、
`scale = np.sqrt(k / (avg_sq * k))`（= `1/√avg_sq`）を全要素に掛ける（L.501-503）。
実数演算では `mean((cZ)²) = avg_sq/avg_sq = 1` が厳密に成り立つ。したがって

```
(1/L) Σ_l Σ_{i,a} (Z^{(l)}_{ia})² = n·k
```

`var_z = 1` を代入すると

```
lnpZ_det := (1/L) Σ_l lnpZ(Z^{(l)}) = −(nk/2)·ln(2π) − (nk)/2
                                    = −(nk/2)·(ln 2π + 1)
                                    = −1.4189385332 · n k
```

BIC 換算では潜在次元 1 あたり **`+2.8378770664 n`** の固定加算となる（符号は罰則側、係数 2 は `−2·Q` から）。

| n | 決定論項の BIC 寄与 / 次元 | 参考: `(d−k)·ln n`（k=3 での Schwarz 側の増分） |
|---:|---:|---:|
| 150（合成, d=15） | **425.68** | 12·5.011 = 60.13 |
| 178（Wine, d=13） | **505.14** | 10·5.182 = 51.82 |
| 280（Cora, d=50） | **794.61** | 49·5.635 = 276.11（k=1→2） |

#### 5.3.1 精度と例外経路（監査で補強）

- **厳密性**: 実数演算では厳密。float64 では `sqrt(k/(avg_sq*k))` が `1/sqrt(avg_sq)` に対し
  丸めを 2 回余分に含むため、事後条件は `1 ± O(10⁻¹⁶)` である。「厳密に」は理想化された表現。
- **正規化されない経路は 2 つだけ** [CONFIRMED_IN_REPOSITORY]:
  1. `avg_sq == 0`（Z が全ゼロ）: `avg_sq > 0` が偽になり未変更で返る（L.504）。理論上のみ。
  2. `NaN`: `np.mean` が NaN を返し `NaN > 0` は偽なので未変更で返る。
     **ただし全ランナーで NaN/Inf ガードが `scale_Z` の前に発火し**
     （`utils_expfam.py` L.507-512 → L.514、`run_fixed_official_exp1_bic_full.py` L.134-137 → L.139、
     `em_runner.py` L.219-224 → L.226）、置換値 `Z_prev` は常に有限なので**この経路は閉じている**。
- **制約は配列全体に効く**（スライスごとではない）。したがって
  `lnpZ(Z^{(l)})` は `l` ごとには一定でない。**`L` 平均だけが一定**であり、
  それは `_lnpZ` が `Σ Z²` についてアフィンだからである。
- **実行順序**（6 ランナーすべてで確認）: `scale_Z` は E-step の最後・M-step の前に走り、
  `Q_strict` は**同じ scaled `Z_samples`** に対して計算される。

| ランナー | `scale_Z` | M-step | `Q_strict` |
|---|---|---|---|
| `utils_expfam.run_em` | L.209 | L.214-220 | ループ後、同一 `Z_samples` |
| `utils_expfam.run_em_dual` | L.514 | L.519-527 | L.581 |
| `run_fixed_official_exp1_bic_full.py` | L.139 | L.144-149 | L.159（同期 L.156 の後） |
| `run_fixed_real_wine_pilot.py` | L.200 | — | L.219（同期 L.218 の後） |
| `run_fixed_real_cora_balanced_k_sweep.py` | L.250 | — | L.262（同期 L.261 の後） |
| `experimental/em_runner.py` | L.226 | L.245-252 | L.294（同期 L.285 の後） |

**この項は「複雑度ペナルティ」として設計されたものではない。**
完全データ尤度基準における `E_q[ln p(Z)]` 項として形式上は正当だが、`scale_Z` によって
データからの情報を持たない定数に退化しており、実効的には `2.838 n` という
**n に比例する固定次元罰則**として働いている [DERIVED]。

### 5.4 [DERIVED] Gaussian-Y の正規化定数が系列間で非対称

`model_expfam.calc_log_likelihood_Y` L.263-264 は Gaussian-Y に対して
`−0.5(y−η)²/σ² − 0.5 ln σ²` を返し、**`−0.5 ln(2π)` を落としている**。
一方 `model_dual_expfam.calc_log_likelihood_X` L.323 は Gaussian-X で**含めている**。
`utils_expfam.calc_Q_dual_strict` はこの欠落を補正しない。

`experimental/eval_utils.calc_Q_dual_strict_exp` **L.227-228** は
`corr −= 0.5·ln(2π)·(観測ペア数)` として補正する [CONFIRMED_IN_REPOSITORY]。

**ただし補正されるのは `Q_strict` のレベルだけである。**
`experimental/model_dual_expfam_consistent.py` L.61-63 の
`calc_log_likelihood_Y` オーバーライド自体は依然として `−0.5 ln(2π)` を落としており、
下流の `calc_Q_dual_strict_exp` が補う構造になっている [CONFIRMED_IN_REPOSITORY]。

結果として **Gaussian-Y の `Q_strict` の絶対値は utils 系列と experimental 系列で
`0.5·ln(2π)·n(n−1)/2` だけ異なる**。n に依存し k には依存しないため、
**同一系列内の k 選択には影響しないが、系列をまたいだ絶対値比較は不可**である [DERIVED]。
（該当は合成シナリオ C のみ。n=150 で `0.918938533 × 11175 = 10269.14`。）

### 5.5 [CONFIRMED_IN_REPOSITORY] 3 系列の分離 — `σ_z²` の扱い

**LINEAGE 1: 印刷された原論文** [UNVERIFIED_IN_REPOSITORY]
研究者本人の一次確認によれば、Eq.(14) は
`σ_z² = 1/(L k n) Σ ||z_i^(l)||²` として `σ_z²` を**更新する**と記載されている。
本監査は PDF を読めないため、この記述を検証していない。

**LINEAGE 2: MATLAB**（`Mato Lab Program/calcdescmetric_ver4.m`）[CONFIRMED_IN_REPOSITORY]

```matlab
L.29   Z_new2 = scaleZ(Z_new);
L.30   % Z_new2 = Z_new;
L.35   % varZ = calcVarZ(Z_new2, params.L);        ← コメントアウト
L.40-42  Q = Q + calcp_X(...) + calcp_Y(...) + calcp_Z(Z_new2(:,:,l), 1);   ← σ_z を 1 に直書き
L.65-77  function Z_new2 = scaleZ(Z)
           a = Σ_{l,i,k} Z²;  a = a / (L·n);       ← すなわち a = k · mean(Z²)
           Z_new2 = Z .* sqrt(size(Z,2) / a);      ← = Z / sqrt(mean(Z²))
L.98-100 function varZ = calcVarZ(Z, L)            ← 定義はあるが呼ばれていない
```

- `scaleZ` は Python の `scale_Z` と**代数的に同一**（`sqrt(k/a)` で `a = k·mean(Z²)` ⇒ `1/√mean(Z²)`）。
  Python 側の docstring が「Based on scaleZ function in calcdescmetric_ver4.m」と述べる通り、忠実な移植である。
- `calcVarZ` の呼び出しは**コメントアウトされ**、`calcp_Z` には `σ_z = 1` が直書きされている。
- `scaleZ` 内にも中心化版 `%Z_new2 = (Z - mean(Z,1)).*...` がコメントアウトで残る。

**LINEAGE 3: 現行 Python** — `var_z = 1.0` 固定 ＋ `scale_Z`。**MATLAB と同じ構造**である。

**評価** [DERIVED + UNRESOLVED]:
- 目的は 3 系列とも**尺度の識別性制約**とみてよい（`model.py` L.98 の注釈と整合）。
- **代数的な接点**: `scale_Z` の下では `calcVarZ` は厳密に 1 を返す。すなわち両者はこの点で互いの不動点である。
- しかし **EM の軌道が同一になるか、`Q` / BIC の評価が同一になるかは別問題であり、証明していない** [UNRESOLVED]。
  **`Q` / BIC への影響は明確に異なる**: `scale_Z` ＋ `var_z = 1` では `lnpZ` 寄与が定数に退化するが、
  `σ_z²` を真に再推定する方式ではデータ依存の量として残る。
- **どちらが「正しい」とは判断しない。**

---

## 6. 正確な variational / EM 恒等式

### 6.1 導出

任意の（`p(Z|D,θ)` に絶対連続な）密度 `q(Z)` に対して:

```
ln p(D|θ) = E_q[ ln p(D|θ) ]
          = E_q[ ln p(D,Z|θ) − ln p(Z|D,θ) ]
          = E_q[ ln p(D,Z|θ) ] − E_q[ ln q(Z) ] + E_q[ ln q(Z) − ln p(Z|D,θ) ]
```

`Q(θ; q) := E_q[ln p(D,Z|θ)]`、`H(q) := −E_q[ln q]`、`KL := E_q[ln q − ln p(·|D,θ)]` と置くと

> **`ln p(D|θ) = Q(θ; q) + H(q) + KL(q ‖ p(Z|D,θ))`**  … (★)

`KL ≥ 0` なので `ln p(D|θ) ≥ Q + H`（ELBO）。

### 6.2 [CONTRADICTED] 「差は entropy だけ」という記述、および先行監査との関係

`reports/theory_audit/theory_audit_report_20260718.md`:

- **§7 本文 L.278**: 「標準 EM 分解 `ln p(X,Y|θ) = Q(θ) + H(q) + KL(q ‖ p(Z|X,Y,θ))`」——**正しい**。
  L.280 も `BIC_impl = [標準BIC相当] + 2·H(q) + 2·KL(…)` と正しく書いている。
- **§1 エグゼクティブサマリー第 4 項 L.37-42**: 「`Q_strict` は …… 周辺尤度 `ln p(X,Y|θ)` とは
  **エントロピー項 H(q) の分だけ**体系的にずれる」——**KL 項が落ちている**。

同一文書内の不整合である [CONFIRMED_IN_REPOSITORY]。
**当該歴史文書は本 Issue では変更しない**（`git log` で単一コミット `95484bb` のみ、
`git diff be1a74cd` は空であることを確認済み）。本監査の (★) を現行の正しい記述として用いる。

#### 先行監査への帰属（重要）

同文書は **L.274 で既に「`Z^{(l)}` は …（`scale_Z` 適用後）」と記録**し、
**L.282-287 で既に「k を 1 増やすごとに約 2.84n の追加ペナルティ」「n=280（Cora）では約 795/k で、
パラメータ側ペナルティ `d·ln n ≈ 282/k` を大きく上回る」という数値に到達している**
[CONFIRMED_IN_REPOSITORY]。

ただしそこでの `2.84n` は **欠落している `H(q)`**（`A_i ≈ I` の仮定の下）に帰されており、
本監査の `lnpZ_det` とは**別の量**である（両者は `A_i = I` の極限でのみ一致する。§6.4）。
また同文書は当該機構を `PLAUSIBLE（因果断定は不可）` と正しく限定している。

> **本監査の新規性は「2.84n という大きさ」ではない。**
> それは 2026-07-18 の記録に既にある。新規なのは
> **`scale_Z` によって `lnpZ` 項が厳密に決定論的（データ非依存・θ 非依存）になる**という点と、
> **一次 CSV からの恒等分解によってその項の選択への寄与を定量化した**点である。

### 6.3 [DERIVED] 本実装で KL が消えない 3 つの理由

1. **Laplace 近似**: `q` は各ノードごとの `N(η_i, A_i^{-1})` の積であり、厳密事後ではない
   （`calc_eta_newton` L.443-458）。ノード間の事後相関も無視されている。
2. **θ のずれ**: `q` は M-step 前の θ で作られ、`Q_strict` は M-step 後の θ で評価される。
   有限 `num_iter = 8` で停止するため EM 停留点でもない。
3. **`scale_Z` の押し出し**: 実際に使われるのは Laplace 密度ではなく、その大域スカラー倍による押し出し。
   大域スケール `c` 倍は entropy を `+ n k ln c` 動かす（符号は両方向あり得る）。

さらに `Q` の Monte Carlo 平均は **iid 標本ではない**:
E-step ループ（`utils_expfam.py` L.496-504）は各 `l` で直前のサンプルから Newton を開始し次の `l` へ渡すため、
`L = 5` 個は**逐次依存の連鎖**である（burn-in・thinning なし）[CONFIRMED_IN_REPOSITORY]。

### 6.4 [DERIVED・条件付き] `A_i ⪰ I` から従う上界

全 lineage で precision 行列は

```
A_i = (1/var_z)·I  +  F^T diag(非負) F  +  c·w²·Z^T diag(非負) Z ,   c ∈ {0.5, 1}
```

の形をしている [CONFIRMED_IN_REPOSITORY: `model_dual_expfam.py` L.181/194/200（c=0.5）、
`model_dual_expfam_fixed.py` L.97/107/113（c=1）、`_percolumn.py` L.141/145/149（c=1）。
`model_dual_expfam_consistent.py` は `_calc_precision_matrix` を持たず、
`DualExpFamLSMMasked`（L.118）→ `DualExpFamLSMFixed` を通じて c=1 を継承する]。
第 2・第 3 項は半正定値なので `var_z = 1` の下で **`A_i ⪰ I`、`det A_i ≥ 1`**。

**`q` を per-node Laplace ガウスの積と仮定すれば**
`H(q) = Σ_i [ (k/2)·ln(2πe) − (1/2)·ln det A_i ] ≤ (nk/2)(1 + ln 2π) = −lnpZ_det`。

> **条件付きの結論**: per-node Laplace の `q` に対して `H(q) ≤ −lnpZ_det`、
> したがって `Q + H(q) ≤ Q − lnpZ_det`。[DERIVED、ただし §6.3 の (3) により
> **実際に使われている `q`（`scale_Z` 押し出し後、かつ逐次依存連鎖の経験分布）に対しては成立を保証しない**]

### 6.5 [DERIVED] ELBO 補正基準と診断スコアの正確な関係（新規）

`BIC_ELBO := −2(Q_strict + Ĥ) + p̂ ln n`、`Ĥ := (nk/2)ln(2πe) − (1/2)Σ_i ln det A_i` と置くと

```
−2Ĥ = −n k (1 + ln 2π) + Σ_i ln det A_i
BIC_ELBO = BIC_impl − n k (1 + ln 2π) + Σ_i ln det A_i
```

一方 `S_cf = BIC_impl − n k (1 + ln 2π)`（§0）なので

> **`BIC_ELBO = S_cf + Σ_i ln det A_i`（厳密）** [DERIVED]

すなわち **診断スコア `S_cf` と ELBO 補正基準の差は、`Σ_i ln det A_i` ちょうど 1 項である。**
`A_i ⪰ I` より `Σ_i ln det A_i ≥ 0` なので `BIC_ELBO ≥ S_cf` であり、上界はない。

`S_cf` は fixed 系列 4/5 ケースで k について単調減少する（§13.2a）。したがって:

> **「ELBO 補正基準が内点最小を持つかどうか」は、
> 「`Σ_i ln det A_i` が k について `S_cf` の減少を打ち消すほど急に増えるか」と完全に同値である。**

**`Σ_i ln det A_i` はどの実験でも記録されていない** [CONFIRMED_IN_REPOSITORY:
`expfam/results` 配下の全 CSV を `log_det|logdet|eigen|det_A|posterior_var` で grep して 0 件]。
したがって ELBO 補正後の argmin は**現時点では計算できない** [UNRESOLVED]。
これが §20 の決定ゲートと次実験計画の中心にある。

---

## 7. Parameter count audit

### 7.1 実際に推定されている自由パラメータ

| 記号 | 個数 | M-step で推定されるか | `calc_bic_dual` が数えるか |
|---|---:|---|---|
| `F` (d×k) | `dk` | ✓（Gaussian-X は閉形式 `calc_F` / それ以外は Adam `_calc_F_adam`） | ✓（`O(k)` 商を引いて `dk − k(k−1)/2`） |
| `Σ_X`（対角） | `d` | ✓ Gaussian-X のみ（`calc_sigma` L.548-586）。他 family では単位行列を返すので**推定していない** | ✓ Gaussian-X のとき `d` |
| `σ_Y` | `1` | ✓ Gaussian-Y のみ（`calc_sigma_y` L.212-235） | ✓ Gaussian-Y のとき `1` |
| `w0` | `1` | ✓ 常に（Adam `calc_w0`, L.149-178） | **✗** |
| `w` | `1` | ✓ 常に（Adam `calc_w`, L.180-210。`fix_w` 指定時を除く） | **✗** |
| `var_z` | — | ✗ 恒久的に `1.0` | ✗（正しい） |
| `Z` (n×k) | `nk` | サンプリング（推定ではない） | ✗ |

**手計算検証**（`fixed_exp1_bic_full_summary.csv` の `num_params` 列と照合、全 18 セル一致）:

| シナリオ | family_x / family_y | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | poisson / bernoulli | 15 | 29 | 42 | 54 | 65 | 75 |
| B | gaussian / poisson | 30 | 44 | 57 | 69 | 80 | 90 |
| C | bernoulli / gaussian | 16 | 30 | 43 | 55 | 66 | 76 |

いずれも `k·15 − k(k−1)/2 + (15 or 0) + (1 or 0)` と一致 [CONFIRMED_IN_REPOSITORY]。

### 7.2 `w0, w` を数えないことの評価

- **k 選択の argmin には影響しない** [DERIVED]。k に依存しない定数 2 個であり、
  すべての候補 k に共通の `2·ln n` を加えるだけである。
- **絶対値としては誤り** [DERIVED]。`w0, w` は M-step で実際に推定されている自由パラメータである。
  `BIC_impl` は両者を数える基準より `2 ln n` だけ小さい（n=150 で 10.02、n=280 で 11.27）。
- 「数えなくてよい」ことと「正しい Schwarz の parameter count である」ことは別問題である [DERIVED]。

### 7.3 `k(k−1)/2` を引く根拠

観測分布を不変にする群は `O(k)`（§9.1）、次元 `k(k−1)/2`。
`O(k)` の作用を受けるのは `F` のみ（`z_i^T z_j` と事前分布 `N(0,I)` は回転不変）なので、
`F` の実質自由度は `dk − k(k−1)/2` [DERIVED]。因子分析の慣行と一致する。

**ただしこれは `O(k)` の作用が自由（`rank F = k`）な点でのみ正しい局所次元である。**
`rank F < k` の点では stabiliser が非自明になり、商の局所次元は下がる（§10-8）[DERIVED]。

### 7.4 `calc_bic`（Y-only 系列）との整合

`utils_expfam.calc_bic` L.104 は `num_params = (k+1)d − k(k−1)/2` を使う。
これは `kd + d − k(k−1)/2` であり、**`calc_bic_dual` の Gaussian-X かつ非 Gaussian-Y の場合と厳密に一致する**
[DERIVED]。**Poisson-X / Bernoulli-X の場合は一致しない**（`calc_bic_dual` は `+d` を付けないため）。

### 7.5 [CONFIRMED_IN_REPOSITORY] 先行研究の parameter count と「量」の違い（新規）

3 系列を混同しない。

**LINEAGE 1: 印刷された原論文** [UNVERIFIED_IN_REPOSITORY]
研究者本人の一次確認によれば、Eq.(26) は
`BIC = −2 ln L + ((k+1)d − k(k−1)/2) ln n` であり、本文は
「the true dimension k\* affects the likelihood function L from Eq. (16)」と述べている。
すなわち印刷された定義は **`Q` ではなく尤度 `L`** を用いているように読める。
**本監査は PDF を読めないため、この記述を検証していない。**

**LINEAGE 2: MATLAB** `Mato Lab Program/DecideNumFactor.m` [CONFIRMED_IN_REPOSITORY]

```matlab
L.11-12  [~,~,~,stats] = factoran(X, i, 'Rotate', 'none');
         loglikely(1, i) = stats.loglike;
L.13     t = (i + 1) * d - .5 * i * (i - 1);
L.14     BIC_X(1, i) = -2 * n * loglikely(1, i) + t * log(n);
```

読み取れること:

1. **parameter count `(i+1)d − 0.5·i(i−1)` は、研究者が報告する Eq.(26) の形と一致する。**
   これは `utils_expfam.calc_bic` L.104 とも厳密に一致し、`calc_bic_dual` とは Gaussian-X の場合のみ一致する。
   → **リポジトリ内に、Python のコード註釈以外の、より一次に近い痕跡が存在する。**
   本レポートの旧版が書いていた「リポジトリ内の唯一の痕跡はコード註釈である」は**誤りであった。撤回する。**
2. **ここで使われている尤度は `factoran` の X のみの周辺対数尤度であり、
   `calcdescmetric_ver4.m` L.40-42 の joint な `Q = calcp_X + calcp_Y + calcp_Z` ではない。**
   すなわち MATLAB で `BIC` という名前を持つ唯一の量は、**関係データ Y を含まない**。
3. `DecideNumFactor` は `Mato Lab Program/` 内のどの `.m` からも呼ばれていない
   （grep で自身の定義以外のヒットなし）。
   **したがって原論文 Experiment 2 の joint モデル BIC がこの関数を使ったかどうかは、
   ファイルからは確定できない** [UNRESOLVED]。**推測で接続しない。**
4. 記録のみ: `−2 * n * loglikely` の `n` 倍が何を意味するかは
   `factoran` の `stats.loglike` の定義に依存する。本監査は MATLAB を実行できず、
   オフラインで公式ドキュメントを確認できないため [UNRESOLVED] とする。研究者の確認事項。

**LINEAGE 3: 現行 Python** — `calc_bic_dual` L.403 は `−2·Q_strict + p̂·ln n` を明示的に使う。

**帰結** [DERIVED]:
研究者報告の Eq.(26) が `ln L`（尤度）であるとすれば、現行 Python は
**`Q_strict`（完全データ対数尤度の近似事後期待値）を同じ位置に置いており、量のレベルで別物である。**
これは既存の「現行基準は Schwarz BIC ではない」（KI-010）より**具体的な、量レベルの不一致の指摘**である。
ただし LINEAGE 1 は本監査では未検証であり、**この帰結は研究者の一次確認に条件付きである。**

---

## 8. Sample-size audit — `ln(n)` の `n` は何か

### 8.1 観測構造

- X: `n × d` 個。列 `l` の負荷 `f_l` と `σ_l` はその列の `n` 観測から識別される。
- Y: 最大 `n(n−1)/2` 個の dyad。`Z` を条件づければ独立だが、周辺では同一ノードを共有する dyad 間に依存が生じる。

### 8.2 各パラメータの実効標本数

| パラメータ | 識別に使われる観測 | `ln n` は妥当か |
|---|---|---|
| `F` の各行 `f_l` | X の列 `l` の `n` 観測 | **オーダーとして妥当** [DERIVED] |
| `Σ_X` の各対角要素 | X の列 `l` の `n` 観測 | **オーダーとして妥当** [DERIVED] |
| `σ_Y` | `n(n−1)/2` dyad | `ln n` は過小。ただし 1 個・k 非依存 [DERIVED] |
| `w0, w` | `n(n−1)/2` dyad（依存あり） | 数えていないので現状は実害なし [DERIVED] |

### 8.3 [SUPPORTED_BY_PRIMARY_SOURCE] ネットワークの実効標本数

Krivitsky & Kolaczyk (2015), *Statistical Science* 30(2):184-198, DOI: 10.1214/14-STS502 は
ERGM を例に、実効標本数 `n_eff`（最尤推定量の分散のスケーリングとして定義）が
**疎なら `O(n_V)`、密なら `O(n_V²)` と 1 桁変わる**ことを示している。

> **「Y が `n²` 個だから sample size = `n²`」とも「ノード数だから必ず `n`」とも決められない**
> [SUPPORTED_BY_PRIMARY_SOURCE + DERIVED]。

ERGM に対する結果であり、**潜在変数モデルへは自動的には移らない** [UNRESOLVED]。

### 8.4 [DERIVED] 設計上の非対称性 — k ペナルティは X からしか入らない

`p̂` の中で **k に依存するのは `F` の項 `kd − k(k−1)/2` だけ**である。増分は

```
Δp̂(K → K+1) = [d(K+1) − (K+1)K/2] − [dK − K(K−1)/2] = d − K
```

（`d − K + 1/2` ではない。§13.2 の `Δpenalty = 12·ln150 = 60.128 = (15−3)·ln150` と整合する。）

つまり:

> **`n(n−1)/2` 個の Y 観測は、当てはまり項（`lnpY`）には寄与するが、
> k に対するペナルティには一切寄与しない。**

n が増えたときのオーダー比較 [DERIVED]:

| 項 | k=K→K+1 の変化量のオーダー |
|---|---|
| `lnpY` の改善（完全データ、Z を代入） | 最大 `O(n²)` 項の和（実際のオーダーは未確定） |
| `lnpX` の改善 | 最大 `O(nd)` 項の和 |
| 決定論的 `lnpZ` 項 | 厳密に `−1.41894 n`（BIC 換算 `+2.83788 n`） |
| Schwarz ペナルティ | `(d − K)·ln n` |

したがって **`n → ∞` で Schwarz ペナルティは他の 3 項に対して漸近的に無視できる**。
競合するのは「当てはまり改善」対「決定論的 `lnpZ` 項」であり、**どちらが勝つかは未解決** [UNRESOLVED]。

### 8.5 incidental parameters との関係

潜在変数 `z_i` の総数は `nk` で n に比例して増える。
Neyman & Scott (1948), *Econometrica* 16(1):1-32 の incidental parameters 問題と**構造的に類似**する
[SUPPORTED_BY_PRIMARY_SOURCE + PLAUSIBLE]。

ただし `z_i` は**周辺化されるべき潜在確率変数**であって固定パラメータではない。
**周辺化すれば問題は生じない。** 問題は `Q_strict` が周辺化せずにサンプルを代入している点にあり、
これは §6 の entropy / KL 欠落と同じ事象の別表現である [DERIVED]。

Shun & McCullagh (1995), *JRSS-B* 57(4):749-760, DOI: 10.1111/j.2517-6161.1995.tb02060.x は
「積分の次元が極限パラメータ n と同程度のとき、通常の Laplace 近似は妥当な漸近近似ではない」ことを示す
[SUPPORTED_BY_PRIMARY_SOURCE]。本モデルの積分次元は `nk` であり、
**per-node に分解しているとはいえこの警告は直接関係する** [DERIVED]。
per-node 分解後の誤差評価は本監査では行っていない [UNRESOLVED]。

---

## 9. Identifiability

### 9.1 [DERIVED] 観測分布を不変にする変換群

`R ∈ O(k)` に対し `z ↦ Rz` は `N(0, I_k)` を不変にし、`(Rz_i)^T(Rz_j) = z_i^T z_j` も不変。
`F ↦ F R^T` とすれば `η_X` も不変。よって
`(F, w0, w, Σ_X, σ_Y) ↦ (F R^T, w0, w, Σ_X, σ_Y)` は `p(X,Y|θ)` を不変にする。群次元 `k(k−1)/2`。

### 9.2 [DERIVED] スケール不定性は存在しない

`z ↦ cz`, `F ↦ F/c`, `w ↦ w/c²` は `η_X`・`η_Y` を保つが、**事前分布 `N(0, I_k)` は固定測度であり
パラメータではない**。したがって `p(X,Y|θ)` は変わる。尺度は識別されている。
`var_z = 1.0` の固定は「識別性のため」というコード注釈（`model.py` L.98、docstring L.84）と整合する
[CONFIRMED_IN_REPOSITORY]。

**アルゴリズム側の `scale_Z` は、モデルには存在しない経験的尺度制約をサンプルに課している**
[DERIVED]。これはモデルの識別性とは別種の操作であり、§5.3 の決定論性を生む。

### 9.3 [DERIVED] `w` の符号は識別される

`z_i^T z_j` を符号反転させる実直交変換は存在しないので、`w` の符号は識別される。

### 9.4 [DERIVED・部分。証明はスケッチ] `{M_K}` の非入れ子性

**示したこと**: `M_{K+1}` で `F' = [F, 0]`（最終列 0）とすると `η_X` は不変だが
```
η_Y = w0 + w (z_i^T z_j + u_i u_j),  u_i ~ N(0,1) iid
```
となり、余分な確率項 `w·u_i u_j` が残る。`w` はスカラーで全次元に共有されているため
この項だけを消す方法はない。したがって**この特定の埋め込みは失敗する**。

**まだ示していないこと（ギャップ）**: 非入れ子性の主張には
`M_{K+1}` の**すべての** `(F', w0', w', Σ', σ_Y')` について `p_{K,θ} ≠ p_{K+1,θ'}` を示す必要がある。
本監査はこれを示していない。

**示唆される理由（[PLAUSIBLE]、証明ではない）**: `w' ≠ 0` のとき
`w'·z'^T z'`（`z' ~ N(0, I_{K+1})`）の法は `w·z^T z`（`z ~ N(0, I_K)`）の法と異なると考えられ、
`w' = 0` では Y が Z から独立になる。

**設計上の由来** [DERIVED]:
この現象は **`w^Y` がスカラーであり行列 `W_Y` ではない**という設計（root `CLAUDE.md` §1、先行研究由来）に起因する。
`W_Y` が `k×k` 行列であれば、余分な行・列を 0 にすることで入れ子性は回復する。

**もし非入れ子性が成立すれば**（条件付きの帰結）:
- `K0 = min{K : p0 ∈ M_K}` の「min」が持つ通常の直観が成立しない（§4.2）。
- 過大指定 K は「境界・rank 欠損の特異点」ではなく**誤指定**として扱う必要がある（§10-7）。
- 誤指定モデルは KL 距離で離れうるので、K 選択が `ln n` ペナルティ差ではなく KL ギャップで決まりうる。
  ただし (i) 推定量が KL 射影に到達すること、(ii) 基準が対数尤度の妥当な近似であること、
  のいずれも本実装では確立していない。KL ギャップの大きさも計算していない [UNRESOLVED]。

---

## 10. Singularity / regularity checklist

| # | 条件 | 判定 | 根拠 |
|---:|---|---|---|
| 1 | 局所／大域的識別可能性 | **VIOLATED（大域）／SUPPORTED（`O(k)` 商の上で局所）** | §9.1 [DERIVED] |
| 2 | パラメータ→分布の写像が 1 対 1 | **VIOLATED** | `O(k)` 軌道全体が同じ分布を与える [DERIVED] |
| 3 | Fisher 情報の非退化 | **VIOLATED（生パラメータ）／UNRESOLVED（商の上）** | `O(k)` 方向で厳密にゼロ。商の上での非退化は未検証 |
| 4 | 真値が内点 | **UNRESOLVED** | 実データでは `p0 ∈ M_K` 自体が未検証（§4.2）。合成データでも生成器の正規化により真値が部分多様体上にある（§4.1） |
| 5 | 尤度の滑らかさ | **POSSIBLY_VIOLATED** | 実装は `np.clip(eta, -20, 10)`（`model_expfam.py` L.57・L.73）や下限 `1e-8` を持ち、clip 作動域で微分不整合。consistent 系列は前方向にこれを解消済み（Issue #25/#26） |
| 6 | パラメータ次元が固定 | **SUPPORTED（θ）／VIOLATED（潜在変数込み）** | `Z` は `nk` 次元で n とともに増える（§8.5） |
| 7 | 入れ子 K における特異点 | **条件付きで該当しない** | §9.4 の非入れ子性が成立すれば、古典的な「過大指定＝特異点」の図式が joint モデルには移らない。**非入れ子性自体がスケッチ段階** |
| 8 | `rank F < k` | **POSSIBLY_VIOLATED** | `O(k)` の作用が自由でなくなり商の局所次元が下がる。実行時の `rank F` は記録されていない [UNRESOLVED]（`F` の `.npy` は gitignore） |
| 9 | `w = 0` で Z が Y から識別されなくなる点 | **VIOLATED（その点で）** | Y 側の Fisher 情報が消える [DERIVED] |
| 10 | 過大指定 K における未使用潜在次元 | **条件付きで該当しない** | `w` がスカラーのため「未使用次元」を作れない（§9.4） |
| 11 | 潜在変数次元が n とともに増加 | **VIOLATED** | `nk` 次元。Shun & McCullagh (1995) の警告が該当 |
| 12 | ネットワーク依存 | **POSSIBLY_VIOLATED** | 実効標本数は `O(n)`〜`O(n²)`（§8.3）[UNRESOLVED] |

### 10.1 [SUPPORTED_BY_PRIMARY_SOURCE] 「潜在変数モデルは特異である」

Drton & Plummer (2017), *JRSS-B* 79(2):323-380, DOI: 10.1111/rssb.12187 abstract:

> "We consider approximate Bayesian model choice for model selection problems that involve models
> whose Fisher information matrices may fail to be invertible along other competing submodels.
> Such singular models do not obey the regularity conditions underlying the derivation of
> Schwarz's Bayesian information criterion BIC and the penalty structure in BIC generally does not
> reflect the frequentist large sample behaviour of the marginal likelihood."

同論文は特異モデルの例として
"determining the number of components in mixture models, the number of factors in latent factor models
or the rank in reduced rank regression" を挙げ、
**"all the classical hidden or latent variable models are singular"** と述べている。

### 10.2 [DERIVED] `O(k)` 商だけでは特異性は消えない

上表の 8（`rank F` 欠損）・9（`w = 0`）・11（潜在次元の増加）・12（依存）は `O(k)` 商とは独立に残る。

### 10.3 [SUPPORTED_BY_PRIMARY_SOURCE, preprint] 反対側の証拠

Nguyen & Hirose, "Consistency of the Bayesian Information Criterion for Model Selection in
Exploratory Factor Analysis", arXiv:2604.07998（2026、**査読前 preprint**）は、
探索的因子分析において BIC が因子数を一致選択することを、正則条件・識別条件・情報行列の非退化の下で示している
（**本監査は abstract レベルの確認のみ。定理の仮定を逐条検証していない** [UNRESOLVED]。
また X 側のみのモデルであり、Y 側を持つ本モデルには自動的には移らない）。

すなわち **「特異だから BIC は必ず一致しない」とは言えない**。
特異性は Schwarz の**導出**を無効にするが、**結論（一致性）を自動的に否定するものではない** [DERIVED]。

---

## 11. RQ4 — heterogeneous family の問題の分解

| # | 論点 | 判定 | 根拠 |
|---|---|---|---|
| A | Gaussian/Bernoulli/Poisson を同一 joint model 内で使うこと**それ自体** | **問題ではない** | 各列・各ペアが正しい確率モデルなら対数尤度の和は well-defined [DERIVED] |
| B | family 割当を固定したまま **K だけ**比較する | **原理的に可能** | 支配測度が全候補で同一。差は `Q ≠ 周辺尤度`（§6）と特異性（§10）に帰着 [DERIVED] |
| C | **family 割当が違うモデル同士**を比較する | **BIC 値の直接比較は無意味** | 下記 D [DERIVED] |
| D | 台・支配測度・正規化定数の違い | **致命的**（family 間比較のみ） | 離散（計数測度）と連続（Lebesgue 測度）の尤度は支配測度が異なる。台違反の quasi-likelihood は確率モデルですらない [DERIVED] |
| E | dispersion parameter | **数え方が family 依存** | Gaussian-X は `d` 個、Gaussian-Y は 1 個、Bernoulli/Poisson は 0 個（§7.1）[CONFIRMED_IN_REPOSITORY] |
| F | model misspecification | **中程度の影響** | 誤指定下では BIC の二次展開は KL 射影点の周りでしか成立しない [DERIVED] |
| G | `Q` を周辺尤度の代わりに使うこと | **最大の要因** | §5.3・§6・§13.2 [DERIVED] |
| H | 特異性 | **中程度** | §10 |

### 11.1 [EMPIRICALLY_OBSERVED] D の直接証拠（一次 CSV、**per-column prototype 系列**）

`expfam/results/per_column_family/per_column_demo_summary.csv`（20 行、`condition` 別平均）:

| condition | BIC | rmse_Z |
|---|---:|---:|
| **all_bernoulli** | **9672.73**（最小） | **0.5919**（最悪） |
| percolumn_correct | 14040.93 | 0.2192（最良） |
| all_poisson | 14697.44 | 0.2752 |
| all_gaussian | 14734.58 | 0.2211 |

`single_vs_joint_summary.csv`（27 行）でも同様（`all_bernoulli` 7216.36 が最小、
`all_bernoulli_binarized` 11238.42、`per_column_all` 11647.79）。

`all_bernoulli` は非二値 X に Bernoulli 尤度を当てる**台違反**の条件であり、確率モデルではない。
その「BIC」が最小になるのは、異なる支配測度の尤度値を比較したことの直接の帰結である
[DERIVED + EMPIRICALLY_OBSERVED]。

### 11.2 判定

> **「異種 family を混ぜるから BIC が使えない」は誤りである。**
> 使えなくなるのは **family 選択に BIC を用いる場合**であり、
> 同一台上の分布同士（例: Poisson vs 負の二項）以外は BIC で family を選んではならない [DERIVED]。
> 同一 family 割当のもとでの **k 選択**については、family 混在は障害にならない。
> そこでの障害は G（`Q ≠ 周辺尤度`）と H（特異性）である。

---

## 12. Asymptotic target map — RQ5

| # | 命題 | 形式的表現 | 本研究での状態 |
|---:|---|---|---|
| P1 | パラメータ一致性 | `θ̂_n → θ0`（`O(k)` 商の上で） | **UNRESOLVED**。証明なし |
| P2 | 潜在空間の回復 | `Ẑ_n → Z_true`（`O(k)` を法として） | **EMPIRICALLY_OBSERVED（有限標本）**。`fixed_official/exp2` で n=50→300 の rmse_Z 改善。**漸近的一致性は未証明** |
| P3 | K 選択一致性 | `P(K̂_n = K0) → 1` | **UNRESOLVED**。n=150 の 1 点でしか測っていない。n を振った測定は皆無（§13.5） |
| P4 | 予測リスク一致性 | held-out 予測リスクが最適値に収束 | **UNRESOLVED** |
| P5 | EM の収束 | `θ^(t)` が停留点へ収束 | **VIOLATED（現行設定では未達）**。`num_iter = 8` 固定、収束判定なし（`run_fixed_official_exp1_bic_full.py` L.70、`run_fixed_real_cora_balanced_k_sweep.py` L.211）[CONFIRMED_IN_REPOSITORY] |
| P6 | Monte Carlo 近似の一致性 | `Q̂_L → Q` as `L → ∞` | **UNRESOLVED**。`L = 5` 固定かつ逐次依存連鎖（§6.3）。Fort & Moulines (2003), *Ann. Statist.* 31(4):1220-1259 は MCEM のほぼ確実収束を弱い条件下で示すが、**シミュレーション核の ergodicity と MC サンプル数の増加を仮定しており `L = 5` 固定の設定は対象外** |
| P7 | Laplace 近似の精度 | per-node Laplace 誤差 → 0 | **UNRESOLVED**。Shun & McCullagh (1995) の警告が該当 |
| P8 | 有限反復のアルゴリズム誤差 | `θ^(8)` と停留点の差 | **UNRESOLVED**。測定されていない |

**P1・P2・P3 は互いに独立の命題である。**
`Ẑ → Z_true` が成り立っても `P(K̂ = K0) → 1` は従わないし、その逆も従わない [DERIVED]。

**P3 は現行の criterion に対しては「何に一致するのか」自体が不明である。**
`BIC_impl` は周辺尤度に基づかないので Schwarz の一致性理論の適用対象ではない。
§8.4 のオーダー比較はむしろ **n が大きいとき決定論項と当てはまり項の競合で決まる**ことを示唆しており、
n=150 での成功が漸近挙動を保証しない [DERIVED]。

> **修論スコープ**: 完全な一致性定理の証明は **[OUT_OF_SCOPE]**。
> 「有限標本での経験的検証」と「漸近的一致性の理論的証明」を明確に区別し、前者のみを主張する。

---

## 13. Existing empirical evidence

**すべての数値は一次 CSV から本監査で読み直した。派生レポートからは引用していない。**
記法は §0 に従う（`lnpZ_det`、`S_cf`）。

### 13.1a 実験一覧 — **fixed 系列**

| データセット | K 候補 | 試行 | 選択基準 | 選ばれた K | ground truth | 一次ソース |
|---|---|---:|---|---|---|---|
| 合成 A/B/C（n=150, d=15, K_TRUE=3） | 1–9 | 10 | `BIC_impl` | **3**（3/3 シナリオ、trial 単位 **30/30**） | K_TRUE=3（生成器） | `fixed_official/exp1_k9/fixed_exp1_bic_k1to9_{agg,bestk_by_trial}.csv` |
| Wine（n=178, d=13, gaussX/bernY） | 1–9 | 5 | `BIC_impl` | **3** | なし | `real_data/wine_fixed_pilot/wine_bic_k1to9.csv` |
| Cora balanced（n=280, d=50, bernX/bernY, density 0.011086, 7 クラス） | 1–6 | **3** | `BIC_impl` / AP / AUC / NMI / ARI | **BIC:1, AP:6, AUC:6, NMI:3, ARI:3** | なし | `real_data/cora_balanced_k_sweep/*.csv` |

解釈上の限界: 合成は n の 1 点のみ、生成器の正規化（§4.1）あり。
Wine は Y がラベル由来で k=2 で既に `auc_y = 1.0`。Cora は KI-011（基準ごとに最適 k が異なる）、**試行 3**。

### 13.1b 実験一覧 — **旧 0.5 系列**

| データセット | K 候補 | 試行 | 選ばれた K | 一次ソース |
|---|---|---:|---|---|
| 合成 A/B/C（n=150, d=15, K_TRUE=3） | 1–6 | 10 | **3**（3/3） | `exp_scenario_{A,B,C}_exp1_k.csv` |
| 合成 A/B/C（n=150, K_TRUE ∈ {1,3,5,7,9}） | 1–10 | 5 | §13.4 | `exp2_bic_{A,B,C}.csv` |

**学会予稿（`conference_submission_final_draft.md` L.79・表1 L.88-101）が依拠するのはこの系列である**
（`EXPERIMENT_REGISTRY.md` L.20/24/28: `current_main / ✓`）[CONFIRMED_IN_REPOSITORY]。

### 13.1c 実験一覧 — その他の系列

| データセット | 系列 | 選択基準 | 備考 |
|---|---|---|---|
| MovieLens user-disjoint（30 splits） | objective-consistent per-column | **使っていない** | K=3 は**固定設計定数**。`k` 列は全 360 行で `3`、BIC 列は存在しない（`bic` の唯一のヒットはヘッダの `q_bic_failed`）[CONFIRMED_IN_REPOSITORY] |
| per_column_demo / single_vs_joint | per-column prototype | family 比較に BIC | §11.1。family 間 BIC 比較の反例 |

> **系列をまたいだ数値の同一表への混在を避けるため、13.1 と 13.2 を系列ごとに分割している**
> （root `CLAUDE.md` §3、KI-002）。

### 13.2a [DERIVED] 診断スコア `S_cf` — **fixed 系列のみ**

各行は公開 CSV の trial 平均 `Q_strict` から `BIC_impl` を再構成し、公開値と一致することを確認したうえで
（**全 8 ブロックで最大絶対差 `7.276e-12`**、`python` で実行して確認）、
`lnpZ_det` を除いた `S_cf` を計算したものである。
**`S_cf` は診断スコアであり、補正版・正しい版の BIC ではない（§0）。**
すべて **trial 平均レベル**の計算であり、trial ごとの `S_cf` は計算していない。

| ケース | n | d | 試行 | k 範囲 | `argmin BIC_impl` | `argmin S_cf` | `S_cf` は単調減少か |
|---|---:|---:|---:|---|---:|---:|---|
| 合成 A (poisX/bernY) | 150 | 15 | 10 | 1–9 | **3** | 9 = **範囲上端** | **はい（内点最小なし）** |
| 合成 B (gaussX/poisY) | 150 | 15 | 10 | 1–9 | **3** | 9 = **範囲上端** | **はい（内点最小なし）** |
| 合成 C (bernX/gaussY) | 150 | 15 | 10 | 1–9 | **3** | 9 = **範囲上端** | **はい（内点最小なし）** |
| Wine (gaussX/bernY) | 178 | 13 | 5 | 1–9 | **3** | **5** | いいえ（**内点最小**） |
| Cora (bernX/bernY) | 280 | 50 | **3** | 1–6 | **1** | 6 = **範囲上端** | **はい（内点最小なし）** |

`S_cf` の値（参考）:
- 合成 A: 20513.47, 17692.79, 15182.57, 15093.73, 15007.86, 14919.05, 14848.89, 14790.92, 14753.07
- Wine: 12905.61, 5412.93, 4809.35, 4670.50, **4435.39**, 4471.03, 4567.31, 4710.99, 4889.64
- Cora: 14017.34, 13392.76, 12908.86, 12629.04, 12433.26, 12254.37

> **重要な読み替え**: 旧版は「argmin が 3→6 に変わる」と書いていたが、
> **その「6」は k=1..6 の CSV を使ったことによる範囲境界であった。** 撤回する。
> 正しくは **fixed 系列 5 ケース中 4 ケースで、`S_cf` は検討範囲内に内点最小を持たない**。
> Wine のみ内点 k=5 に最小を持つ（手計算による spot check 済み）。

k\*→k\*+1 の分解（`Δ BIC_impl = −2·Δfit + Δpenalty + (−2·Δ lnpZ_det)`）:

| ケース | `2·Δfit` | `Δ penalty` | `−2·Δ lnpZ_det` | `Δ BIC_impl` |
|---|---:|---:|---:|---:|
| 合成 A (3→4) | 148.96 | 60.13 | 425.68 | +336.85 |
| 合成 B (3→4) | 181.23 | 60.13 | 425.68 | +304.58 |
| 合成 C (3→4) | 160.09 | 60.13 | 425.68 | +325.72 |
| Wine (3→4) | 190.66 | 51.82 | 505.14 | +366.30 |
| Cora (1→2) | 900.68 | 276.11 | 794.61 | +170.03 |

### 13.2b [DERIVED] 診断スコア `S_cf` — **旧 0.5 系列のみ**

| ケース | n | d | 試行 | k 範囲 | `argmin BIC_impl` | `argmin S_cf` | 内点最小か |
|---|---:|---:|---:|---|---:|---:|---|
| 合成 A (poisX/bernY) | 150 | 15 | 10 | 1–6 | **3** | **3** | **はい（内点）** |
| 合成 B (gaussX/poisY) | 150 | 15 | 10 | 1–6 | **3** | 6 = 範囲上端 | いいえ |
| 合成 C (bernX/gaussY) | 150 | 15 | 10 | 1–6 | **3** | **3** | **はい（内点）** |

`2·Δfit`(3→4) は A: 25.14、B: 370.69、C: 18.17。A と C は `Δpenalty = 60.13` 単独で k=3 を保つ。

> **これは重要な限定である。**
> 学会予稿が依拠するのは旧 0.5 系列であり、そのうち **2/3 のシナリオでは
> パラメータ数罰則だけで k=3 が選ばれる。**
> したがって「既存の k\*=3 という記述は Schwarz 型ペナルティが効いた結果ではない」とは
> **無条件には言えない。系列とシナリオを特定して述べる必要がある。**

### 13.3 [EMPIRICAL / INFERENCE の区別] Cora と合成データの関係

| 段階 | 内容 |
|---|---|
| **FACT** | 決定論的な `O(nk)` 事前分布寄与が存在する（§5.3）。Cora では `2·Δfit(1→2) = 900.68`、`Δpenalty = 276.11`、`−2·Δ lnpZ_det = 794.61`、和は `+170.03` で公開値 `14981.9706 − 14811.9432 = +170.027` と一致する |
| **EMPIRICAL** | 決定論項を除いた `S_cf` は、fixed 系列 5 ケース中 4 ケースで内点最小を失う（Cora を含む）。試行数は合成 10・Wine 5・**Cora 3** |
| **INFERENCE** | この項は、調べた n = 150〜280 のデータセットにおいて選択結果に**実質的（selection-dominant）な影響**を与えている |
| **NOT YET PROVED** | この項が各結果の**唯一の原因**であること。Cora と合成データが**同一の機構**であること |

Cora と合成データの間では、n（150→280）以外に **d（15→50、したがって Schwarz 増分 60.13→276.11）**、
family_x/family_y、エッジ密度、**試行数（10→3）**、そして**当てはまり改善曲線の形状**が同時に異なる。
本監査はこれらを分離していない。

> **したがって Cora について言えるのは
> 「KI-011 と整合する重要な機構を特定した」までであり、
> 「KI-011 を完全に説明した」でも「合成データと同一の機構である」でもない。**

なお `reports/theory_audit/theory_audit_report_20260718.md` L.282-293 は
同じ Cora の説明を既に提示し、`PLAUSIBLE（因果断定は不可）` と限定している。
本監査はその限定を維持する（新しい測定を行っていないため、強度を上げる根拠がない）。
未測定の鍵は `Σ_i ln det A_i` である（§6.5、§17-U1）。

### 13.3b [PLAUSIBLE] stale-Σ とシナリオ B の関係

`exp2_bic` で `P(K̂=K_TRUE) < 1` になるのは **シナリオ B のみ**（§13.4）。
B は 3 シナリオ中**唯一 Gaussian-X** であり、§5.2 の stale-Σ が実害を持つ唯一のケースでもある。
**ただしこれは相関であって因果ではない** [PLAUSIBLE]。B は Poisson-Y でもあり、BIC の絶対値も最大である。
同じ stale-Σ を持つ旧 0.5 系列の B も `K_TRUE = 3` では正しく選べている。
**因果の切り分けには実験が必要** [UNRESOLVED]。

### 13.4 [CONFIRMED_IN_REPOSITORY] K_TRUE 掃引（**旧 0.5 系列**、n=150, d=15, L=5, num_iter=8, 5 試行）

trial ごとに `argmin_k BIC_impl` を取り、`K_TRUE` と一致した割合:

| K_TRUE | A | B | C |
|---:|---|---|---|
| 1 | 1.00 | 1.00 | 1.00 |
| 3 | 1.00 | 1.00 | 1.00 |
| 5 | 1.00 | **0.80**（`K̂` の多重集合 `{5,5,5,5,6}`） | 1.00 |
| 7 | 1.00 | **0.60**（`{7,7,7,8,8}`） | 1.00 |
| 9 | 1.00 | **0.60**（`{9,9,9,10,10}`） | 1.00 |

（`K̂` は**多重集合**として示す。trial 順ではない。）

- 誤りはすべて **+1〜+2 側への過大選択**であり、過小選択は 1 件もない [CONFIRMED_IN_REPOSITORY]。
- trial 平均 `rmse_Z` の argmin は 15 セル中 15 セルで `K_TRUE` に一致。
  trial 平均 `BIC_impl` の argmin は B の `K_TRUE = 9` のみ 10 になる。
- **registry 上の位置づけは `current_support / △`**（`EXPERIMENT_REGISTRY.md` L.33「KI-010 の検証対象」）であり、
  `conference_submission_final_draft.md` には引用されていない [CONFIRMED_IN_REPOSITORY]。

### 13.5 [CONFIRMED_IN_REPOSITORY] n を振った K 選択は存在しない

`fixed_official/exp2`（n sweep）の列は
`scenario, trial, seed_data, seed_model, n, d, k, family_x, family_y, rmse_z, rmse_x, rmse_y, q_final, w_est, ...`
であり、**`k` は k_true 固定、BIC 列を持たない**。
旧 0.5 の `exp_scenario_*_exp2_n.csv` も `rmse_*` と `n, trial` のみ。
したがって **K 選択の n 依存性は一度も測定されていない**。

### 13.6 [CONFIRMED_IN_REPOSITORY] `Σ_i ln det A_i` はどこにも記録されていない

`expfam/results` 配下の全 CSV を `log_det|logdet|eigen|det_A|posterior_var` で grep して 0 件。
したがって §6.5 の `BIC_ELBO` は**既存データからは計算できない**。

---

## 14. Literature evidence

| # | 文献 | 何が主張されているか | 前提 | 本モデルへの関連 | **自動的には移らない点** |
|---:|---|---|---|---|---|
| L1 | Schwarz, G. (1978). "Estimating the Dimension of a Model." *Ann. Statist.* 6(2):461-464. DOI: 10.1214/aos/1176344136 | Bayes 解の漸近展開の主要項が `ln L(θ̂) − (k/2) ln n` になること | **観測が Koopman-Darmois（指数型）族から独立に得られること**（PDF からのテキスト抽出でこの記述を確認したが、圧縮 PDF のため**逐字一致は保証しない**） | 本研究の基準名の起源 | 本モデルは潜在変数モデルで観測周辺分布は指数型族でない。dyad は独立でない。**`BIC_impl` は周辺尤度を使っていない**（§5） |
| L2 | Drton, M. & Plummer, M. (2017). *JRSS-B* 79(2):323-380. DOI: 10.1111/rssb.12187 | 特異モデルで Schwarz の正則条件が破れること、sBIC の定義 | RLCT の知識または上界 | 本モデルは潜在変数モデルなので該当 | **本モデルの RLCT は未知** [UNRESOLVED]。sBIC をそのまま適用できない |
| L3 | Watanabe, S. (2013). *JMLR* 14:867-897. arXiv:1208.6338 | WBIC が特異でも unrealizable でも Bayes free energy と同じ漸近展開を持つ | **逆温度 `1/ln n` の事後サンプリング** | 特異性・誤指定に理論的に対応する唯一の候補 | 現行は Laplace 近似ベースで事後 MCMC を持たない |
| L4 | Biernacki, C., Celeux, G. & Govaert, G. (2000). *IEEE TPAMI* 22(7):719-725. DOI: 10.1109/34.865189 | ICL の定義と、混合モデルの仮定違反に対する BIC より高い頑健性 | 混合モデル、MAP 割当 | 現行 `BIC_impl` を ICL 型として解釈する根拠 | ICL は**離散**クラスタ割当向け。本モデルの潜在変数は連続 |
| L5 | Krivitsky, P. N. & Kolaczyk, E. D. (2015). *Statist. Sci.* 30(2):184-198. DOI: 10.1214/14-STS502 | ネットワークの実効標本数が疎密で `O(n_V)`〜`O(n_V²)` と変わる | ERGM | `ln n` の `n` の判断根拠（§8.3） | **ERGM に対する結果**。潜在変数モデルへは移らない |
| L6 | Shun, Z. & McCullagh, P. (1995). *JRSS-B* 57(4):749-760. DOI: 10.1111/j.2517-6161.1995.tb02060.x | 積分次元が n と同程度のとき通常の Laplace 近似は妥当な漸近近似でない | 積分次元が n と同程度 | 本モデルの積分次元は `nk`（§8.5） | 本実装は **per-node に分解**した Laplace。分解後の誤差評価は未実施 |
| L7 | Neyman, J. & Scott, E. L. (1948). *Econometrica* 16(1):1-32 | incidental parameters の下で構造パラメータの一致性が壊れうること | 各 incidental parameter が有限個の観測にしか関わらない | `z_i` の総数が `nk` で増える | `z_i` は**周辺化されるべき確率変数**。周辺化すれば問題は生じない |
| L8 | Fort, G. & Moulines, E. (2003). *Ann. Statist.* 31(4):1220-1259. DOI: 10.1214/aos/1059655912 | MCEM のほぼ確実収束（弱い条件下） | ergodic なシミュレーション核、**MC サンプル数の増加** | 本実装は MCEM | **`L = 5` 固定・非増加**、`num_iter = 8` 固定、逐次依存連鎖 |
| L9 | Nguyen, H. D. & Hirose, K. (2026). arXiv:2604.07998（**査読前 preprint**） | 探索的因子分析における BIC の因子数一致選択 | 正則条件・識別条件・情報行列の非退化 | 「特異でも BIC が一致することはある」ことの反対証拠 | **abstract レベルの確認のみ**。X 側のみのモデル |

**すべて `[SUPPORTED_BY_PRIMARY_SOURCE]`。ただし独立監査は Web にアクセスできないため
`UNVERIFIED_EXTERNAL` のままである。本監査本体の Web 確認をもって auditor の確認とは扱わない。**

### 14.1 候補手法の適用条件比較（ランキングではない）

| 手法 | 何を評価するか | 前提 | 計算要件 | 直接適用可能性 | 未知の要件 | 修論スコープ |
|---|---|---|---|---|---|---|
| Schwarz BIC（正しい意味の） | 観測データ周辺尤度 + 正則 Laplace 展開 | 正則性・内点真値・iid 指数型族 | 周辺尤度の最大値 | **不可** | 周辺尤度の近似法 | ✗ |
| 現行 `BIC_impl` | 完全データ対数尤度の近似事後期待値（ICL 型） | — | **実装済み・追加コスト 0** | 可（§5.3 の性質を明示すること） | — | ○（記述的ベースライン） |
| ELBO 補正（`−2(Q̂ + Ĥ) + p̂ ln n`） | 周辺尤度の**下界** | per-node Laplace `q` の妥当性 | **`Σ_i ln det A_i` の記録のみ追加** | 可 | 下界の緩みの k 依存性 | ○（低コスト） |
| held-out 予測（pair split / CV） | 予測リスク | 分割の独立性・MCAR | 分割数 × フィット | 可（masked lineage 実装済み） | X 側 held-out は未実装 | ○ |
| WBIC | Bayes free energy | 逆温度 `1/ln n` の事後サンプリング | MCMC 必須 | **不可** | 事後サンプラ設計 | ✗（将来課題） |
| sBIC | 特異 BIC | 学習係数（RLCT）の知識 | 方程式系の求解 | **不可** | **本モデルの RLCT は未発見** | ✗（将来課題） |
| 周辺尤度の直接近似（bridge/IS） | `ln p(X,Y)` | サンプラ品質 | 高コスト・高分散 | 原理的には可 | サンプラ設計 | △（将来課題） |

---

## 15. What is established（確立している）

| # | 主張 | ラベル | 一次根拠 |
|---:|---|---|---|
| E1 | `BIC_impl = −2 Q_strict + p̂ ln n`、`p̂ = kd − k(k−1)/2 + d·1{GX} + 1·1{GY}` | `[CONFIRMED_IN_REPOSITORY]` | `utils_expfam.py` L.399-403、18 セルの `num_params` 一致 |
| E2 | `w0, w` は M-step で推定されるが `p̂` に数えられていない。k 選択の argmin には影響しない | `[CONFIRMED_IN_REPOSITORY]`+`[DERIVED]` | 同上 + `model_expfam.py` L.149-210 |
| E3 | `var_z` は 1.0 固定で推定されない | `[CONFIRMED_IN_REPOSITORY]` | `model.py` L.99、網羅 grep |
| E4 | `scale_Z` により `(1/L)Σ_l ln p(Z^{(l)}) = −(nk/2)(1+ln 2π)`（実数演算で厳密、float64 で `1 ± O(10⁻¹⁶)`） | `[DERIVED]` | `model.py` L.468-504 + `_lnpZ` L.315-321 |
| E5 | 公開 BIC 値は `−2 Q_strict + p̂ ln n` から**最大絶対差 `7.276e-12`** で再構成できる（8 ブロック） | `[DERIVED]` | 合成 fixed(k1-9) / 合成旧 0.5 / Wine / Cora |
| E6 | 診断スコア `S_cf` は fixed 系列 5 ケース中 4 ケースで検討範囲内に内点最小を持たない（k について単調減少）。Wine のみ内点 k=5。**旧 0.5 系列では A・C が内点 k=3 を保つ** | `[DERIVED]` | §13.2a・§13.2b |
| E7 | 正確な恒等式は `ln p(D|θ) = Q + H(q) + KL`。KL は本実装で消えない | `[DERIVED]` | §6.1・§6.3 |
| E8 | **per-node Laplace の `q` に対して**、`A_i ⪰ I` ゆえ `H(q) ≤ −lnpZ_det`、したがって `Q + H ≤ Q − lnpZ_det`。**実際に使われている押し出し後の `q` に対しては保証しない** | `[DERIVED・条件付き]` | §6.4 |
| E9 | `BIC_ELBO = S_cf + Σ_i ln det A_i`（厳密） | `[DERIVED]` | §6.5 |
| E10 | 観測分布を不変にする群は `O(k)`（一般の位置で）。尺度と `w` の符号は識別される | `[DERIVED]` | §9.1-9.3 |
| E11 | 潜在変数モデルは Schwarz の正則条件を満たさない | `[SUPPORTED_BY_PRIMARY_SOURCE]` | Drton & Plummer (2017) abstract |
| E12 | 異種 family 混在それ自体は k 選択の障害ではない。障害は family **間**比較（支配測度の違い） | `[DERIVED]`+`[EMPIRICALLY_OBSERVED]` | §11・§11.1 |
| E13 | fixed 系列の合成 3 シナリオで trial 単位 30/30 が `K̂ = 3 = K_TRUE`（k=1..9、10 試行） | `[CONFIRMED_IN_REPOSITORY]` | `fixed_exp1_bic_k1to9_bestk_by_trial.csv` |
| E14 | Cora（n=280, d=50, density 0.011086, 7 クラス, **3 試行**）で `BIC_impl` は k=1、AP/AUC は k=6、NMI/ARI は k=3 | `[CONFIRMED_IN_REPOSITORY]` | `cora_balanced_k_sweep_{agg,bestk,data_summary}.csv`、`run_fixed_real_cora_balanced_k_sweep.py` L.67 |
| E15 | MovieLens #33 の K=3 は固定設計定数であり、BIC 選択の結果ではない | `[CONFIRMED_IN_REPOSITORY]` | `movielens_userdisjoint_20260822_summary.csv`（360 行すべて `k=3`、BIC 列なし） |
| E16 | `calc_Q_dual` は引数 `sigma` を使っていない。旧 0.5 系列の `Q_strict` は Gaussian-X で 1 M-step 古い Σ を使う | `[CONFIRMED_IN_REPOSITORY]` | `utils_expfam.py` L.329・L.498・L.521・L.581 |
| E17 | Gaussian-Y の `−0.5 ln 2π` は utils 系列で欠落し、experimental 系列では `calc_Q_dual_strict_exp` L.228 が補う（**モデル側のオーバーライドは consistent 系列でも落としたまま**） | `[CONFIRMED_IN_REPOSITORY]` | `model_expfam.py` L.263-264 / `model_dual_expfam_consistent.py` L.61-63 / `eval_utils.py` L.228 |
| E18 | 生成器は Z を列ごとに z-score 化（`ddof=0`）し、Gaussian-X の X も z-score 化する。`K_TRUE` は階数として保存される | `[CONFIRMED_IN_REPOSITORY]`+`[DERIVED]` | `data_generator_expfam.py` L.282-298 |
| E19 | 旧 0.5 系列で `K_TRUE ∈ {1,3,5,7,9}` の掃引が存在し、15 セル中 13 セルで `P(K̂=K_TRUE)=1.00`。低下は B のみ | `[CONFIRMED_IN_REPOSITORY]` | `exp2_bic_{A,B,C}.csv` |
| E20 | MATLAB `DecideNumFactor.m` L.13-14 は `factoran` の **X のみ**の周辺対数尤度に `(i+1)d − 0.5i(i−1)` の罰則を付す。`calcdescmetric_ver4.m` の joint `Q` ではない | `[CONFIRMED_IN_REPOSITORY]` | `Mato Lab Program/DecideNumFactor.m`, `calcdescmetric_ver4.m` L.40-42 |
| E21 | MATLAB でも `varZ` 更新はコメントアウトされ、`calcp_Z` に `σ_z = 1` が直書きされ、`scaleZ` が Python の `scale_Z` と代数的に同一である | `[CONFIRMED_IN_REPOSITORY]` | `calcdescmetric_ver4.m` L.29/35/42/65-77/98-100 |

---

## 16. What is plausible only（もっともらしいだけ）

| # | 主張 | なぜ確立していないか |
|---:|---|---|
| PL1 | Cora の k=1 選択は「疎 → 事後拡散 → H(q) が大 → 完全データ型基準の過大ペナルティ」で説明できる | 機構は整合的だが `Σ_i ln det A_i` を実測していない。2026-07-18 の記録も `PLAUSIBLE` と限定している |
| PL2 | Cora と合成データが同一の機構である | n 以外に d・family・密度・試行数・当てはまり曲線が同時に異なる（§13.3） |
| PL3 | シナリオ B の過大選択（K_TRUE≥5）は stale-Σ（E16）に起因する | 相関のみ。B は Poisson-Y でもあり交絡 |
| PL4 | ELBO 補正を入れると Cora でより大きい k が選ばれる | `BIC_ELBO = S_cf + Σ_i ln det A_i`（E9）の第 2 項が未測定 |
| PL5 | `n → ∞` で決定論項（`O(nk)`）が当てはまり項に負けて過大選択に転じる | §8.4 のオーダー比較は当てはまり項の実オーダーを確定していない |
| PL6 | `{M_K}` が K について非入れ子である | §9.4。示したのは 1 つの埋め込みの失敗のみ。全パラメータにわたる非包含は未証明 |
| PL7 | 非入れ子性による KL ギャップが K 選択を有利にする | ギャップの大きさを計算していない。推定量が KL 射影に到達する保証もない |
| PL8 | Wine の `BIC_impl` argmin k=3 がクラス数 3 と一致したのは潜在次元の回復を意味する | Y がラベル由来で k=2 で既に `auc_y = 1.0`。`S_cf` では k=5 |
| PL9 | 現行 Python の `Q_strict` が原論文 Eq.(26) の `ln L` に対応する | §7.5。原論文は未検証、MATLAB の `DecideNumFactor` は呼ばれておらず、Experiment 2 との接続は確認できない |

---

## 17. What remains unresolved（未解決）

| # | 未解決事項 | 解消に必要なもの |
|---:|---|---|
| U1 | `Σ_i ln det A_i` の値、したがって `BIC_ELBO` の argmin | 新規フィットでの記録。既存データからは計算不可（§13.6） |
| U2 | K 選択の n 依存性 | n を振った K 掃引。一度も行われていない（§13.5） |
| U3 | E16（stale-Σ）の数値的影響量 | 同期版・非同期版の対比フィット |
| U4 | `O(k)` 商の上での Fisher 情報の非退化 | 解析または数値評価 |
| U5 | 本モデルクラスの実対数閾値（RLCT）／学習係数 | 文献調査で発見できず。sBIC 適用の前提 |
| U6 | 原論文 Eq.(26) の「量」が joint `Q` か X のみの `ln L` か、および Experiment 2 が `DecideNumFactor` を用いたか | **研究者本人による原論文の該当節の確認**。parameter count `(k+1)d − k(k−1)/2` 自体は MATLAB `DecideNumFactor.m` L.13 で確認済み（§7.5） |
| U7 | MATLAB `BIC_X = -2 * n * loglikely + t*log(n)` の `n` 倍の意味 | `factoran` の `stats.loglike` の定義確認（本監査は MATLAB を実行できない） |
| U8 | `σ_z²` 更新方式と `scale_Z` + `var_z=1` の等価性 | 証明または反例（§5.5）。**等価性を証明できないため UNRESOLVED** |
| U9 | 本モデルの実効標本数 `n_eff` | Krivitsky-Kolaczyk 型の解析の潜在変数モデルへの拡張 |
| U10 | per-node 分解後の Laplace 近似誤差の n 依存性 | Shun-McCullagh 型の解析 |
| U11 | 実データで `p0 ∈ M_K` が成り立つか（realizability） | 適合度検定 |
| U12 | `{M_K}` の非入れ子性の完全証明、および KL ギャップの大きさ | §9.4 のギャップを埋める |
| U13 | `rank F` が実行時に欠損しているか | `F` は `.npy` が gitignore されており保存されていない |
| U14 | Nguyen & Hirose (2026) の仮定が本モデルの X 側に当てはまるか | 定理の逐条検証 |

---

## 18. Thesis-safe claims（修論に書いてよい形）

**(1) 基準の呼称と位置づけ**
> 本研究で潜在次元の選択に用いる指標は、観測データの周辺尤度に基づく Schwarz の BIC ではなく、
> MC-EM の Q 関数（完全データ対数尤度の近似事後期待値）に `−2·Q + p̂ ln n` の形の罰則を加えた
> **完全データ型（ICL 型）の指標**である。関数名・CSV 列名は provenance のため `BIC` のまま維持する。

**(2) 基準の実効的な次元罰則（新しい記述）**
> この指標では、EM の各反復で潜在変数サンプルを平均二乗が 1 になるよう大域的に正規化しているため、
> Q 関数に含まれる潜在変数の事前分布項がデータに依存しない定数
> `−(nk/2)(1 + ln 2π)` に退化する。その結果、**潜在次元 1 あたり `n(1 + ln 2π) ≈ 2.84n` の
> 固定的な次元罰則が実効的に働いている**。同じ構造は先行研究の MATLAB 実装にも存在する。
> 本研究の設定（n = 150〜280）では、この項はパラメータ数罰則 `p̂ ln n` の k 依存部分より大きい。
> 実際、この項だけを取り除いた診断スコアは、検討した 5 ケース中 4 ケースで
> 候補範囲内に内点最小を持たなくなる。
> **ただしこの診断スコアは補正された基準ではなく、当該項の寄与を測るための量である。**

**(3) 合成データの結果**
> `K_TRUE = 3` の合成データ 3 シナリオ（n=150, d=15、10 試行）において、
> `k = 1,…,9` の候補から本指標が `k = 3` を選ぶことを 30 試行すべてで確認した。
> ただしこれは n = 150 の 1 点における有限標本の観測であり、
> **大標本での選択一致性を示すものではない。**

**(4) 実データの結果**
> Wine（n=178、5 試行）では本指標の最小が `k = 3` であり、クラス数 3 と一致した。
> ただし Wine の関係データはクラスラベルから構成されており、`k = 2` の時点で
> 関係データの再現 AUC が 1.0 に達している。したがってこの一致は観測であり、
> 潜在次元の回復を意味するとは主張しない。
> 疎な Cora balanced subset（n=280, 密度 0.011、3 試行）では本指標は `k = 1` を選ぶ一方、
> held-out link prediction の AP/AUC は `k = 6`、クラスタリング指標 NMI/ARI は `k = 3` を最大にした。
> **単一の指標で潜在次元を決められるとは主張しない。**

**(5) 識別可能性**
> 本モデルでは、潜在座標の直交変換 `Z ↦ ZR`, `F ↦ FR` (`R ∈ O(k)`) が観測分布を不変にする。
> 尺度と `w` の符号は識別される。
> `F` の実質自由度を `kd − k(k−1)/2` と数えるのはこの群次元に対応する。
> **回転の不定性は潜在空間の内部の不定性であり、潜在次元 K の識別可能性とは別問題である。**

**(6) 入れ子性（限定つき）**
> 本モデルでは関係データの結合強度 `w` がスカラーとして全潜在次元に共有されているため、
> `K + 1` 次元モデルで負荷行列の余分な列を 0 にしても関係データ側に余分な項が残る。
> **すなわち、属性側の自然な埋め込みでは `M_K` を `M_{K+1}` の部分モデルとして実現できない。**
> `M_{K+1}` のすべてのパラメータにわたる非包含は本研究では示していない。今後の課題である。

**(7) 異種 family と model selection**
> 分布族を列ごと・データ種別ごとに変えること自体は、
> 同一の分布族割当のもとでの潜在次元比較を妨げない。
> 一方、支配測度の異なる分布族どうしで対数尤度の値を直接比較することはできない。
> 実際、非二値の属性に Bernoulli 尤度を当てた条件が最小の指標値を示しつつ潜在変数の推定誤差は最悪であった。
> **本研究では分布族の選択に本指標を用いない。**

**(8) 大標本での主張の限定**
> 本研究では、パラメータ一致性・潜在空間回復・次元選択一致性・予測リスク一致性・
> MC-EM の収束・Laplace 近似の精度を別個の命題として区別する。
> 修論で示すのは**有限標本における経験的な観測**であり、
> いずれについても漸近的一致性の理論的証明は与えない。

**(9) 実装上の限定事項**
> Monte Carlo サンプル数 `L = 5`・EM 反復数 8 はいずれも固定であり、収束判定を行っていない。
> また `L` 個のサンプルは逐次依存の連鎖であり、独立標本ではない。

---

## 19. Claims that must NOT be made（書いてはいけない）

| 禁止表現 | 理由 |
|---|---|
| 「Schwarz BIC で潜在次元を選択した」 | `Q_strict` は周辺尤度ではない（§5・§6）。KI-010 |
| 診断スコア `S_cf` を「corrected BIC」「modified BIC」「true BIC」「Schwarz BIC」と呼ぶ | §0。`S_cf` は補正版ではない |
| 「BIC により真の潜在次元が選ばれることを確認した」（無条件） | n=150 の 1 点。`K_TRUE` を振った掃引は旧 0.5 系列の 5 試行のみ |
| 「Schwarz 型の罰則が過大次元を防いでいる」 | fixed 系列 4/5 で `S_cf` に内点最小がない（§13.2a） |
| 「既存の k\*=3 は Schwarz 型ペナルティの結果ではない」（無条件） | **旧 0.5 系列の A・C では罰則単独で k=3 が残る**（§13.2b）。系列とシナリオを特定すること |
| 「Cora の BIC 失敗と合成データの BIC 成功は同一の機構である」 | 機構同定を行っていない（§13.3）。「KI-011 と整合する機構を特定した」までにとどめる |
| 「KI-011 を完全に説明した」 | 同上。`Σ_i ln det A_i` 未測定 |
| 「潜在変数モデルだから標準 BIC は絶対に使えない」 | 特異性は Schwarz の**導出**を無効にするが一致性を自動的に否定しない（§10.3） |
| 「異種 family を混ぜるから BIC が使えない」 | 誤り（§11.2）。使えないのは family **間**比較 |
| 「WBIC / sBIC に変更すれば解決する」 | WBIC は事後 MCMC を要し枠組み外。sBIC は本モデルの RLCT が未知（U5） |
| 「n → ∞ なら真の K が必ず回復する」 | 未証明。§8.4 はむしろ逆の可能性を示唆する（PL5） |
| 「回転不定性があるので K も識別できない」 | 誤った推論（§4.4） |
| 「`{M_K}` は入れ子でないことを証明した」 | §9.4 はスケッチであり、全パラメータにわたる非包含は未証明（PL6） |
| 「Wine で潜在次元がクラス数と一致したので手法が妥当である」 | Y がラベル由来。k=2 で AUC 1.0 |
| 「MovieLens 実験で K=3 が選ばれた」 | K=3 は固定設計定数。BIC 列すら存在しない（E15） |
| 「原論文の BIC は joint モデルの Q を使っている」 | 未検証。MATLAB で `BIC` と名の付く唯一の量は X のみの `factoran` 尤度（§7.5、PL9） |
| 「per-column heterogeneous-X を提案手法として採用した」 | prototype。root `CLAUDE.md` §3、Issue #26 の「Prototype promoted: NO」 |
| 「`Q` と周辺尤度の差はエントロピーである」 | KL 項が残る（§6.2） |
| 旧 0.5 系列と fixed 系列と consistent 系列の数値を同じ表・図に並べる | root `CLAUDE.md` §3、KI-002。§13 では表を系列ごとに分割している |

---

## 20. Decision gate

### 20.1 反証パス（adversarial pass）

| 自問 | 結果 |
|---|---|
| standard BIC が実は K 選択に一致的である可能性は残っていないか | **残っている**。Nguyen & Hirose (2026, preprint) は因子分析で BIC の一致性を示す（§10.3） |
| 「特異だから BIC は必ず inconsistent」という誤解をしていないか | していない。§10.3 で明示的に区別した |
| 現行 Q-based criterion が意図せず良い selector になる条件はないか | **ある**。決定論項 `2.838 n·k` は n 比例の次元罰則として働き、n=150 では `K_TRUE=3` を 30/30 で当てている（E13）。旧 0.5 系列では `K_TRUE ∈ {1,3,5,7,9}` の 15 セル中 13 セルで的中（§13.4）。**「artefact だから悪い selector だ」とは言えない** |
| Cora の失敗は criterion ではなく optimization / finite sample / sparsity 由来ではないか | **切り分けられていない**。`rank F`・`Σ_i ln det A_i`・収束状態のいずれも記録がなく、**試行数は 3** である（U1・U13）。断定しない |
| heterogeneous family 問題と misspecification を混同していないか | §11 で A–H に分解した |
| `K_TRUE` は生成器の設定と本当に一致しているか | **厳密には一致しない**（§4.1）。`K_TRUE` は「階数」としてのみ well-defined |
| `O(k)` 非識別性と K 非識別性を混同していないか | §4.4 で分離した |
| sample-size 定義を根拠なく選んでいないか | §8 で `O(n)`〜`O(n²)` の幅を明示し、決めていない |
| 診断スコア `S_cf` を作る反実仮想は正当か | **限定つきで正当**。`S_cf` は ELBO とは `Σ_i ln det A_i` だけ違う（E9）。**`S_cf` の argmin が ELBO 補正基準の argmin を意味することはない** |
| `S_cf` の argmin として報告した値は本当に最小点か | **いいえ**。4/5 は範囲の上端であり内点最小ではない（§13.2a）。旧版の「6」は k=1..6 の CSV を使った副産物であった。撤回済み |
| 旧 0.5 系列でも同じ結論か | **いいえ**。A と C は `S_cf` でも内点 k=3 を保つ（§13.2b）。系列依存であり一律には言えない |
| 先行監査（2026-07-18）の貢献を自分のものにしていないか | §6.2 で帰属を明記した。`2.84n` の数値は既に同文書にある |

### 20.2 主決定

> ## `D: INVESTIGATE_ALTERNATIVE_CRITERION_BEFORE_K_SWEEP`

**次点（SECONDARY）: `A: RUN_K_SELECTION_SIMULATION_NEXT`**

#### D を主決定とする根拠

1. **§6.5 の恒等式 `BIC_ELBO = S_cf + Σ_i ln det A_i` により、
   未解決の問いが `Σ_i ln det A_i` ただ 1 つに縮約された。**
   `S_cf` は fixed 系列 4/5 で k について単調減少するので（§13.2a）、
   「ELBO 補正基準が内点最小を持つか」は「`Σ_i ln det A_i` が k について十分急に増えるか」と完全に同値である。
2. その測定量は E-step で既に計算されている `A_i` から得られ、**追加コストはほぼゼロ**である。
   しかも**どの実験にも記録されていない**（§13.6）。
3. 現状で `P(K̂ = K_TRUE)` だけを測る K 掃引を先に行っても、
   得られる数値は「決定論項がその n で偶然うまく効くか」に帰着する。
   **同一のフィットの上で 2 つの基準を比較できなければ、K 掃引の解釈が定まらない。**
4. `K_TRUE` を振った選択率は旧 0.5 系列に既に存在し（§13.4）、n=150 では 15 セル中 13 セルで 1.00 である。
   **単純な K 掃引の追加情報量は小さい。** 真に未測定なのは **n 依存性**（U2）であり、D の実験に同居させられる。
5. Issue #35 の non-goals（新 criterion の実装）に抵触しない。
   ELBO 補正は**新しいモデルではなく、既存量の記録と事後計算**である。

#### D を反証しうる証拠

- `Σ_i ln det A_i` を記録した結果、ELBO 補正基準が `BIC_impl` と**同じ k** を選ぶ場合
  → 決定論項の議論は「絶対値の解釈」の問題に縮小し、A（K 掃引）を先にすべきだった、となる。
- 現行基準が n を変えても安定して `K_TRUE` を当てる場合
  → 「artefact だが良い selector」として記述的に採用すればよく、代替基準の検討は不要（→ C に近づく）。

#### A（次点）を主決定にすべき条件

- `Σ_i ln det A_i` の測定が技術的に不可能であることが判明した場合。
- あるいは修論の残り時間が、選択率の記述的報告以上を許さない場合。

#### C・B・E を主決定にしなかった理由

- **C（現行基準を descriptive baseline としてのみ使う）** は root `CLAUDE.md` §5 と KI-010 により
  **既に発効している運用ルール**である。次の研究決定として選ぶことは実質的に E と同じであり、新しい情報を生まない。
  ただし本監査の §18(2) と §7.5 は C の記述内容を**大幅に具体化する**ものであり、
  D の結果がどうであれ修論本文には反映されるべきである。
- **B（さらに理論が必要）** は採用しない。本監査で導出可能な理論
  （正確な恒等式・`BIC_ELBO = S_cf + Σ ln det A_i`・parameter count・sample size・識別性・`A_i ⪰ I` の上界）は
  完了した。残る理論ギャップ（§9.4 の非入れ子性の完全証明、U5 の RLCT）は
  **修論の中心主張に影響せず、今後の課題として書ける**。次に必要なのは**測定**である。
- **E（変更不要）** は E6・E16・E20 が未報告のまま残ることを意味するため採用しない。

#### 次 Issue の提案

`k_selection_next_experiment_plan_20260822.md` に pre-registration draft を置いた。
**本 Issue では実行しない。**

---

## 21. 独立監査（research-auditor）の findings と判定

`.claude/agents/research-auditor.md` による READ-ONLY 独立監査を実施。17 件の finding すべてを判定した。

| # | 重要度 | 内容 | 判定 | 反映箇所 |
|---:|---|---|---|---|
| 1 | HIGH | `S_cf` の argmin「6」は範囲境界であり内点最小ではない。k=1..9 では 9（単調減少） | **ACCEPT**（自分でも k=1..9 の CSV から再計算し確認） | §1(2), §13.2a, E6, §19, §20.1 |
| 2 | HIGH | 「リポジトリ内の唯一の痕跡はコード註釈」は誤り。`DecideNumFactor.m` L.13 に parameter count が存在。かつ `factoran` の X のみ尤度である | **ACCEPT**（MATLAB を直接確認） | §1(7), **§7.5 新設**, E20, U6, U7, PL9 |
| 3 | HIGH | Cora の文が論理的に反転。「同一の機構」は overclaim | **ACCEPT** | §1(3), **§13.3 新設**, PL2, §19 |
| 4 | HIGH | §13 の表が系列を混在。`CLAUDE.md` §3 違反。§1 の断定が旧 0.5 系列では成立しない | **ACCEPT** | §13.1a/b/c, §13.2a/b に分割, §1(2), §19 |
| 5 | MEDIUM | E8 の `H(q) ≤ −lnpZ_det` は per-node Laplace の `q` に条件付き | **ACCEPT** | §6.4, E8 |
| 6 | MEDIUM | 非入れ子性の証明は 1 つの埋め込みのみ。全パラメータ非包含は未証明 | **ACCEPT** | §9.4, E（削除）, PL6, §18(6), §19, U12 |
| 7 | MEDIUM | 試行数の記載なし（Cora は 3 試行）。`S_cf` は trial 平均レベル | **ACCEPT** | §13.1a, §13.2a, E14, §18(3)(4) |
| 8 | MEDIUM | `Δp̂ = d − K`。`+1/2` は誤り | **ACCEPT**（代数で確認） | §8.4 |
| 9 | MEDIUM | post hoc の `A_i` は E-step の `A_i` と同一ではない（Gauss-Seidel + `scale_Z` 後） | **ACCEPT** | 実験計画 §6 |
| 10 | MEDIUM | runtime 外挿が d と系列を交絡 | **ACCEPT** | 実験計画 §11 |
| 11 | LOW | 行番号 12 箇所の誤り | **ACCEPT**（全件自分で確認） | 全体 |
| 12 | LOW | `10269.8` → `10269.14` | **ACCEPT** | §5.4 |
| 13 | LOW | §13.4 のベクトルは trial 順ではなくソート済み | **ACCEPT** | §13.4（多重集合として表記） |
| 14 | LOW | 命名規則（`corrected BIC` 等）は遵守済み。§7 の表題のみ要調整 | **ACCEPT** | **§0 新設**, 実験計画 §7 |
| 15 | LOW | consistent 系列のモデル側オーバーライドも `−0.5 ln 2π` を落としている | **ACCEPT** | §5.4, E17 |
| 16 | MEDIUM | 2026-07-18 記録が既に `2.84n` に到達している。帰属が必要 | **ACCEPT** | §6.2 に帰属節を新設 |
| 17 | LOW | `7.28e-12` は未実行のため未検証 | **RESOLVED**（実行して `7.276e-12` を確認、8 ブロック） | E5 |

**REJECT した finding はない。** 独立監査が確認できなかった 2 点は本監査本体で解消した:

- **歴史文書の非改変**: `git log --follow` で `reports/theory_audit/theory_audit_report_20260718.md` は
  単一コミット `95484bb` のみ。`git diff be1a74cd -- <file>` は空。**VERIFIED**。
- **再構成誤差**: 実行して 8 ブロック全体で `7.276e-12`。**VERIFIED**。

**独立監査は Web にアクセスできないため、§14 の外部文献はすべて `UNVERIFIED_EXTERNAL` のままである。**
本監査本体の Web 確認をもって auditor の確認とは扱わない。

---

## 付録 A. 成果物を 2 件に留めた理由

Issue #35 は「必要なら 3 つ目として数式・コード対応表を独立ファイルにしてよい」としているが、
§5.1 の対応表は 20 行程度で本文に収まり、
§5.2〜§5.5 の指摘（`sigma` 未使用・決定論項・正規化定数の非対称・3 系列の `σ_z²` 比較）と
不可分に読む必要があるため、独立ファイルにせず本文に含めた。

## 付録 B. 検証コマンド

すべて読み取り専用であり、リポジトリを変更しない。

```bash
# (1) 決定論項の恒等式と公開 BIC の再構成（8 ブロック、最大誤差 7.276e-12）
python - <<'PY'
import pandas as pd, numpy as np
C = 1 + np.log(2*np.pi)
def blk(n,d,fx,fy,ks,Q,pub):
    logn=np.log(n); ks=np.array(ks); Q=np.array(Q,float)
    npar=np.array([k*d-k*(k-1)//2+(d if fx=="gaussian" else 0)+(1 if fy=="gaussian" else 0) for k in ks])
    bic=-2*Q+npar*logn
    scf=-2*(Q+(n*ks/2)*C)+npar*logn
    return np.abs(bic-np.asarray(pub)).max(), ks[bic.argmin()], ks[scf.argmin()], bool(np.all(np.diff(scf)<0))
a=pd.read_csv('expfam/results/fixed_official/exp1_k9/fixed_exp1_bic_k1to9_agg.csv')
for s,fx,fy in [("A","poisson","bernoulli"),("B","gaussian","poisson"),("C","bernoulli","gaussian")]:
    g=a[a.scenario==s].sort_values('k_est')
    print(s, blk(150,15,fx,fy,g.k_est,g.q_strict_mean,g.bic_mean))
w=pd.read_csv('expfam/results/real_data/wine_fixed_pilot/wine_bic_k1to9.csv').groupby('k').agg(q=('q_strict','mean'),b=('bic','mean'))
print("Wine", blk(178,13,"gaussian","bernoulli",w.index,w.q,w.b))
c=pd.read_csv('expfam/results/real_data/cora_balanced_k_sweep/cora_balanced_k_sweep_agg.csv')
print("Cora", blk(280,50,"bernoulli","bernoulli",c.k,c.q_mean,c.bic_mean))
PY

# (2) calc_Q_dual が引数 sigma を使っていないことの確認
python - <<'PY'
s = open('expfam/src/utils_expfam.py', encoding='utf-8').read()
b = s[s.index('def calc_Q_dual('):s.index('def calc_Q_dual_strict(')]
print([l for l in b.splitlines() if 'sigma' in l])   # -> シグネチャ 1 行のみ
PY

# (3) MovieLens #33 の k が固定定数であることの確認
python -c "import pandas as pd; d=pd.read_csv('expfam/results/real_data/movielens_userdisjoint/movielens_userdisjoint_20260822_summary.csv'); print(d.k.unique(), [c for c in d.columns if 'bic' in c.lower()])"

# (4) K_TRUE 掃引（旧 0.5 系列）の選択率
python - <<'PY'
import pandas as pd
for s in "ABC":
    d = pd.read_csv(f'expfam/results/exp2_bic_{s}.csv')
    sel = d.loc[d.groupby(['k_true','trial'])['BIC'].idxmin()]
    print(s, {int(kt): float((g.k_est==kt).mean()) for kt, g in sel.groupby('k_true')})
PY

# (5) MATLAB 一次資料
sed -n '1,17p'  "Mato Lab Program/DecideNumFactor.m"
sed -n '28,45p' "Mato Lab Program/calcdescmetric_ver4.m"
sed -n '64,100p' "Mato Lab Program/calcdescmetric_ver4.m"

# (6) 歴史文書の非改変
git log --follow --oneline -- reports/theory_audit/theory_audit_report_20260718.md
git diff be1a74cd -- reports/theory_audit/theory_audit_report_20260718.md    # 空であること
```
