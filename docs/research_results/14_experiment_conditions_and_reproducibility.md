# 14. 実験条件・再現性・記録完全性の総点検（2026-11-28）

> **この章の位置づけ**：DOCUMENTATION / PROVENANCE AUDIT です。新しい fit・データ生成・MCEM・Laplace 最適化はしていません。
> 既存の結果 CSV・NPZ・protocol・runinfo、01〜13 章、既存の 5 つの台帳と追補の台帳、`EXPERIMENT_REGISTRY.md` は変更していません。
> 対象は、研究結果集の 88 記録（`data/experiment_inventory.csv`）と追補の 4 記録（`data/addendum_20261127_inventory.csv`）、合わせて 92 記録です。
> 基点は `8da446a`（Draft PR #175 の head）です。

この章は、**各実験が実際にどんな条件で実行され、何を比較し、どの一次資料に結果が残っていて、その記録をどこまで信用できるか**を、GitHub だけで追えるようにするためのものです。
研究結果の良し悪しは扱いません。記録がそろっていることと、実験が正しかったことは別の問題です。記録が見つからないことは、実験が不正だったことを意味しません。

## 0. 先に結論

- **記録の所在**：92 記録すべてについて、台帳が指す一次資料のファイルを固定 SHA の git tree で確認できました。結果の commit 91 件はすべて repository と remote branch に存在します（残り 1 件は文献で、commit はありません）。
- **実行時の git head**：runinfo などに実行時の git head が保存されている記録は 59 件で、その head は**すべて repository に存在**しました。
- **新しい fit を伴う記録**：51 件。台帳に fit 数が書かれていない記録が 16 件（`involves_new_fits=unknown`）。新しい fit を伴わない分析・監査・理論・文献・設計が 24 件、archive が 1 件です。**92 件を「92 の実験」と数えてはいけません**（§5.6）。
- **記録品質の分類**（今回の定義、§1.3）：**A 11 件・B 64 件・C 17 件**。研究結果の強弱や、既存の比較公平性 A〜D とは別の尺度です。
- **再現性に最も影響する不足**：
  - 学会予稿の本文に採用した旧 0.5 系列の実験（`B-OLD05-EXP1-4`）には runinfo と実行時の git head が無く、seed は runner のコードにだけ書かれています。
  - 2026-08 より前の fixed 系列・実データの記録の多くに、L・MCEM 反復数・生成器が出力として残っていません。
  - fit を伴う 67 記録のうち、データ生成器を一次資料で特定できたのは 26 件、protocol ファイルが結果と一緒に残るのは 31 件です。
- **資料間の食い違い**：
  - 自動照合で新しく見つかったものはありません。同じディレクトリの protocol と runinfo・manifest の間、および台帳と一次資料の間で、n・d・K_TRUE・L・反復数・分布族を照合しました。
  - 既知の 6 件（gram 相関の丸め、Poisson の条件数、C_Q の不等式の訂正、C_Q と C_Lap の罰則の違い、系列の混在、件数の計上）は、値・SHA・出典を並べて `data/experiment_record_gaps.csv` に記録しました。
  - ほかに、台帳の evidence 列が曖昧で場所を特定できない記録が 4 件ありました。

## 1. 読み方

### 1.1 新しく作ったファイル

| ファイル | 内容 | 行数 |
|---|---|---|
| [`data/experiment_conditions_audit.csv`](data/experiment_conditions_audit.csv) | 記録 × 60 項目（研究目的・データ生成・推定条件・評価方法・結果）の値と状態。long 形式。条件が複数ある記録は、一次資料ファイルごとの子行（`parent_id`・`condition_label`）を持つ | 6,861（記録単位 5,520 ＋ 子行 1,341） |
| [`data/experiment_provenance_map.csv`](data/experiment_provenance_map.csv) | 記録 ID → 読んだ SHA・結果 commit・実行時 git head・生成器・runner・protocol・runinfo・図・報告書・registry 行・GitHub リンク・記録品質 | 92 |
| [`data/experiment_record_gaps.csv`](data/experiment_record_gaps.csv) | 記録不足（MISSING）・食い違い（CONFLICT）・リンクの曖昧さ。重要度・一次資料・解消可能性・必要な作業・確認結果 | 307 |
| [`scripts/build_experiment_records_audit.py`](scripts/build_experiment_records_audit.py) | 上の 3 つを固定 SHA の git オブジェクトから生成し、品質検査（`--check`）を行う。読み取り専用 | — |

### 1.2 状態（各項目）

| 状態 | 意味 |
|---|---|
| **VERIFIED** | 固定 SHA の一次資料（runinfo・protocol・manifest・fit 単位の記録・結果 CSV・コード）で、値またはファイルの存在を確認した。主要数値は、既存の 96 項目監査または追補の 38 項目監査で照合済みのもの |
| **DOCUMENTED** | 台帳・`EXPERIMENT_REGISTRY.md`・報告書に記載はあるが、一次資料では照合していない |
| **MISSING** | 構造化された一次記録（runinfo・protocol・manifest・fit 単位の JSON・結果 CSV の列）、registry の該当行、台帳のいずれにも見つからない。**報告書本文（Markdown）は自動照合の対象外**なので、報告書には書かれている可能性がある |
| **CONFLICT** | 資料間で記載が食い違う。値・SHA・出典を並べて `experiment_record_gaps.csv` に記録した |
| **NOT_APPLICABLE** | その種類の記録には該当しない。理論・文献・監査、新しいデータを作らない分析（条件は `parent_id` の親実験の行で確認）、実データの生成パラメータ、Gaussian を含まない条件の分散など。**MISSING には数えない** |

### 1.3 記録品質の分類 A・B・C（今回の定義）

研究結果の良し悪しとは独立です。既存の比較公平性 A〜D（09 章）とも別物です。分類は `scripts/build_experiment_records_audit.py` の規則で機械的に決めています。

| 分類 | 規則 | 指導教員に示すときの扱い |
|---|---|---|
| **A** | 主要数値が一次 CSV で照合済み（96 項目監査・追補 38 項目監査）であり、主要 14 項目（n・d・K_TRUE・X/Y の分布族・データ seed・初期値 seed・反復数・L・runner・runinfo・結果 CSV・実装 commit・生成器）に MISSING が無い。未解決の解釈上の食い違いが無い | 実験条件・主要数値・一次根拠を確認できる |
| **B** | A でも C でもないもの。資料として説明できるが、主要数値が未照合、または主要条件の一部が DOCUMENTED・MISSING | 一部の確認が必要 |
| **C** | archive・中断（STOPPED）・不採用（OLD_NOT_ADOPTED）・main に無い保存ブランチの記録、または主要数値が未照合で主要 14 項目の MISSING が 4 項目以上 | 記録不足または矛盾があり、現時点で主要な根拠に使わない方がよい |

理論・文献・監査の記録は条件項目がすべて NOT_APPLICABLE になり、主要数値の照合も無いため B になります。これは「数式の正しさを確認した」ことを意味しません。

## 2. 全体の集計

### 2.1 記録の種類と新しい fit

| record_kind | 件数 | 新しい fit |
|---|---:|---|
| experiment（人工データ） | 56 | あり 45・未確認 11（`E-AF8V3` などの EM 0 の記録は analysis に分類） |
| experiment_real（実データ） | 10 | あり 5・未確認 5 |
| experiment_stopped | 1 | あり（`F-8B-A1`、3 試行で中断） |
| analysis（保存状態の再評価・事後解析） | 10 | なし |
| analysis_real（実データの診断・読み取り） | 2 | なし |
| correction（forward correction） | 1 | なし（`TG-CORR`） |
| theory・audit・literature・design | 5・3・2・1 | なし |
| archive | 1 | 対象外（`B-ARCHIVE`） |

新しい fit を伴う記録は 51 件、未確認は 16 件です。未確認は、台帳の run_count が「未確認」のものです。

### 2.2 fit を伴う 67 記録（あり 51 ＋ 未確認 16）の主要項目

| 項目 | VERIFIED | DOCUMENTED | MISSING | NOT_APPLICABLE |
|---|---:|---:|---:|---:|
| ノード数 n | 52 | 5 | 10 | 0 |
| 属性次元 d | 47 | 5 | 15 | 0 |
| K_TRUE | 45 | 5 | 7 | 10（実データ） |
| X の分布族 | 35 | 9 | 23 | 0 |
| データ seed | 56 | 3 | 2 | 6（実データ） |
| 初期値 seed | 55 | 1 | 11 | 0 |
| MCEM 反復数 | 29 | 10 | 28 | 0 |
| E-step の L | 43 | 6 | 18 | 0 |
| データ生成器 | 26 | 1 | 40 | 0 |
| runner（実行コード） | 64 | 0 | 3 | 0 |
| 実装 commit（実行時の git head） | 51 | 16 | 0 | 0 |
| runinfo（または fit 単位の記録） | 53 | 0 | 14 | 0 |
| protocol | 31 | 4 | 32 | 0 |

- **seed は良く残っています**。データ seed は 67 件中 56 件が一次資料（runinfo・protocol・結果 CSV の seed 列）にあります。MISSING の 2 件は `A-REPRO` と `B-OLD05-AUX` です。
- **実装 commit**：16 件は実行時の git head が出力に無く、台帳の結果 commit で代用しています（DOCUMENTED）。結果 commit と実行時のコードが同じとは限りません。
- **生成器**：40 件は、結果と一緒に残る記録から生成関数を特定できません。生成関数は runner 内の import を読めば分かることが多いですが、自動照合の対象外です。
- **protocol**：32 件は結果ディレクトリに protocol ファイルが無く、registry の該当行にも設計書へのリンクがありません。設計書が `reports/` にある記録も含まれます（例：`F-DSWEEP`、`F-155` は事前登録と台帳に書かれています）。

### 2.3 記録品質の分布

| 分野 | A | B | C |
|---|---:|---:|---:|
| 元論文・再現 | 0 | 3 | 1 |
| 指数型分布族への拡張 | 0 | 4 | 2 |
| 分布の誤指定 | 2 | 5 | 1 |
| 実データ | 0 | 8 | 3 |
| per-column | 1 | 4 | 1 |
| 分布の自動選択 | 1 | 8 | 2 |
| 潜在次元 K の選択 | 5 | 24 | 6 |
| 追加実験 A・B と原因分析・訂正 | 2 | 2 | 0 |
| 理論・監査 | 0 | 6 | 1 |
| **計** | **11** | **64** | **17** |

## 3. 分野別の一覧

各行の記号は `experiment_conditions_audit.csv` の状態です：**V** = VERIFIED、**D** = DOCUMENTED、**M** = MISSING、**C** = CONFLICT、**–** = NOT_APPLICABLE。
「seed」はデータ seed・初期値 seed、「L・反復」は E-step の L・MCEM 反復数です。値そのものは CSV の `value` 列、出典は `source` 列にあります。
リンクは、記録を読んだ固定 SHA の protocol と一次資料のディレクトリ（またはファイル）です。「fit」は新しい fit を伴うかどうかです。

### 元論文・再現

| ID | 何を調べたか（変えた条件） | 系列 | fit | n | d | K_TRUE | X / Y | seed | L・反復 | runner | 主要数値 | 品質 | リンク |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-PAPER | 原論文 Mikawa et al. 2024：原論文の条件（n=150, d=15 など） | A（原論文の印刷式、1/2 あり） | なし | D | D | – | –/– | –・– | –・– | – | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/paper/A_study_on_latent_structural_models_for_binary_rel.pdf) |
| A-MATLAB | 研究室 MATLAB コードの監査（#158）：— | MATLAB | なし | – | – | – | –/– | –・– | –・– | – | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/bada1d2/reports/k_selection_theory/historical_code_lineage_audit_20261106/02_matlab_code_audit.md) |
| A-REPRO | 先行研究再現との同条件比較（Control）：Control_Gauss_Bern, K_TRUE=3, k_est=1.. | B（0.5）と reproduction（… | あり | **M** | **M** | V | V/V | **M**・**M** | **M**・**M** | V | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/reproduction/results/comparison) |
| A-HALF | 1/2 係数の追加検証（half_factor_check）：dry_run / full / scenario_c_extra | B と C | 未確認 | **M** | **M** | **M** | **M**/**M** | V・V | **M**・**M** | V | D | **C** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/half_factor_check) |

### 指数型分布族への拡張（旧 0.5 系列・fixed 系列）

| ID | 何を調べたか（変えた条件） | 系列 | fit | n | d | K_TRUE | X / Y | seed | L・反復 | runner | 主要数値 | 品質 | リンク |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B-OLD05-EXP1-4 | Scenario A/B/C Exp1–4（旧 0.5 系列、学会予稿）：Scen.A/B/C、K_TRUE=3、n=50–300、d、3×3 family 組合せ | B（0.5 あり） | あり | D | D | D | D/D | D・D | D・D | V | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results) |
| B-OLD05-AUX | exp1_full / exp2_bic（旧 0.5 補助）：— | B | 未確認 | **M** | **M** | V | **M**/**M** | **M**・**M** | **M**・**M** | V | D | **C** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results) |
| B-FIXED-EXP1 | fixed official Exp1（K=1..9）：Scen.A/B/C、n=150、K_TRUE=3、候補 K=1..9、historical generator | C（fixed） | あり | V | V | V | V/V | V・V | **M**・**M** | **M** | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/fixed_official/exp1_k9) |
| B-FIXED-EXP2 | fixed official Exp2（n-sweep）：n=50..300、K_TRUE=3、d=15 | C | あり | V | V | D | V/V | V・V | **M**・**M** | V | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/fixed_official/exp2) |
| B-FIXED-EXP3 | fixed official Exp3（d-sweep）：d=5..30 | C | あり | V | V | **M** | V/V | V・V | **M**・**M** | V | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/fixed_official/exp3) |
| B-ARCHIVE | 初期実装 archive と GEMINI レポート：— | 初期実装・AI 生成 | 対象外 | V | V | V | –/– | V・– | –・– | – | – | **C** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/archive) |

### 分布の誤指定

| ID | 何を調べたか（変えた条件） | 系列 | fit | n | d | K_TRUE | X / Y | seed | L・反復 | runner | 主要数値 | 品質 | リンク |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B-FIXED-EXP4 | fixed official Exp4（誤指定 3×3）：Scen.A/B/C × 推定 family 3×3 | C | あり | V | V | **M** | **M**/**M** | V・V | **M**・**M** | V | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/fixed_official/exp4) |
| B-FIXED-GRID | fixed 版 mismatch grid と old-vs-fixed …：3×3×3 | C（と B） | 未確認 | V | V | V | **M**/**M** | V・V | V・**M** | V | D | **C** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/distribution_mismatch_fixed) |
| D-M1 | 分布誤指定 M1（#134）：Phase 9K と同じ 20 データ、K=3 固定 | E-3（per-column、G3/B6/… | あり | V | V | V | V/V | V・V | V・V | V | V | **A** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/expfam/results/misspecification_m1/phase_m1_20261013/preflight.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/misspecification_m1/phase_m1_20261013) |
| D-E1 | E1：Z の点推定による M1 の再評価（#141）：Part A = M1 replay、Part B = 別 seed | E-3 | あり | V | V | V | V/V | V・V | V・V | V | V | **A** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/expfam/results/z_point_estimator_e1/e1_20261104/full/preflight.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/z_point_estimator_e1/e1_20261104/full) |
| D-E2A | E2a：初期値と MC 乱数の切り分け（#142）：(初期値 seed, MC seed) ∈ {A,B}² | E-3 | あり | V | V | V | V/V | V・V | V・V | V | D | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/expfam/results/seed_separation_e2a/e2a_20261104/full/preflight.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/seed_separation_e2a/e2a_20261104) |
| D-E2C | E2c：EM 反復数（#142）：— | E-3 | なし | – | – | – | –/– | –・– | –・– | V | – | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/11f42bf/expfam/src/experimental/run_em_iterations_e2c.py) |
| D-MX2 | MX2：誤指定の拡張（PR #168）：M1 と同じデータ、bp6 / bg6 / all-G など × Y 観測率 | E-3 | 未確認 | V | V | V | V/V | V・V | V・**M** | V | V | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/856d475/expfam/results/misspecification_extended/phase_20261110/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/856d475/expfam/results/misspecification_extended/phase_20261110) |
| C-OD-SYN | Poisson 誤指定（人工 NB-Y）：r_true∈{2,5,20,∞} | D | 未確認 | V | V | V | **M**/**M** | V・V | V・V | V | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/overdispersion) |

### 実データ

| ID | 何を調べたか（変えた条件） | 系列 | fit | n | d | K_TRUE | X / Y | seed | L・反復 | runner | 主要数値 | 品質 | リンク |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C-WINE | Wine（fixed pilot / clean）：Gaussian-X / Bernoulli-Y（Y はラベル由来） | C | あり | **M** | **M** | – | D/D | –・**M** | **M**・**M** | V | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/real_data/wine_clean) |
| C-WINE-AUDIT | Wine old05 audit：— | B/C | なし | – | – | – | –/– | –・– | –・– | V | **M** | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/real_data/wine_old05_audit) |
| C-CORA-BFS | Cora BFS subset（不採用）：max-degree BFS | C | 未確認 | V | V | – | V/V | V・**M** | **M**・**M** | V | D | **C** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/real_data/cora_subset_pilot) |
| C-CORA-OLD | Cora balanced subset（k-sweep / held-o…：balanced_degree subset n=280→700、Bernoulli-X / Bernoulli-Y | C | あり | D | **M** | – | D/D | –・**M** | **M**・**M** | V | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/real_data/cora_clean) |
| C-ML-INSAMPLE | MovieLens（Poisson-Y / Bernoulli-X、in-…：100 映画 subset（full-data popularity で選択） | C | 未確認 | **M** | **M** | – | **M**/**M** | –・**M** | **M**・**M** | **M** | D | **C** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/real_data/movielens_final_clean) |
| C-RECON | 3 データセット横断の再構成比較：— | C | 未確認 | **M** | **M** | – | V/V | –・**M** | **M**・**M** | V | **M** | **C** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/real_data/common_reconstruction_eval) |
| C-OD-DIAG | MovieLens 過分散診断：co-rating count | C/D | なし | **M** | **M** | – | **M**/**M** | –・V | V・V | V | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/overdispersion) |
| C-OD-STRICT | MovieLens strict held-out（Poisson / N…：pair mask | D（masked / NB） | 未確認 | V | V | – | **M**/V | V・V | V・V | V | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/overdispersion) |
| C-ML-SHAREDZ | MovieLens 共有 Z ablation：Proposed / fix_x / fix_w | D | 未確認 | V | V | – | **M**/V | –・V | V・V | V | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/shared_z_ablation) |
| C-ML-USERDISJ | MovieLens user-disjoint validation（#3…：30 user-disjoint split、train users だけで映画選択 | E+F | あり | D | **M** | – | D/V | D・V | V・V | V | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/real_data/movielens_userdisjoint) |
| C-CORA-SH1 | CORA-SH1：Cora strict held-out での X fa…：Cora 全体、全ペアの 20% を mask、K=6（primary）・3 | E-scalar + BX-Newton | あり | V | V | – | V/V | V・V | V・V | V | V | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/8a66894/expfam/results/real_data/cora_strict_heldout_family_main_20261124/protocol_frozen.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/8a66894/expfam/results/real_data/cora_strict_heldout_family_main_20261124) |

### per-column（列ごとの分布族）

| ID | 何を調べたか（変えた条件） | 系列 | fit | n | d | K_TRUE | X / Y | seed | L・反復 | runner | 主要数値 | 品質 | リンク |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D-PC-DEMO | per-column family デモ・単独 vs 同時・属性 abla…：gauss/bern/pois 各 3 列 + Poisson-Y | F（per-column、legacy n… | あり | V | V | V | **M**/D | V・V | V・V | V | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/per_column_family) |
| D-YSPARSE | Y sparsity stress（trials=10）：y_obs_rate ∈ {1.0,0.5,0.2,0.1} × 4 条件 | F | あり | V | V | V | **M**/D | V・V | V・V | V | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/story_diagnostics) |
| D-COMPL | complementary blocks（#27）：n=80, d=9, K_TRUE=3, bern→z1 / gauss→z2 / pois→z3 | E+F（per-column consis… | あり | V | V | V | **M**/**M** | V・V | V・V | V | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/story_diagnostics) |
| D-COVER | matched latent-coverage ablation（#31）：complementary vs full_coverage | E+F | あり | V | V | V | **M**/**M** | V・V | V・V | V | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/story_diagnostics) |
| D-PCMIX-K | per-column mixed-X での K 選択（2026-09-07）：homogeneous 3 + mixed 4 config、K_0=3、Gaussian-Y | E-3 | あり | V | V | V | V/V | V・V | V・V | V | V | **A** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/d610a8d/expfam/results/percolumn_mixed_x_k_selection_20260907) |
| C-ML-MIXED | MovieLens mixed-X pilot と属性診断（2026-07）：genre19 + 平均評価・公開年・件数 | F（per-column） | あり | V | **M** | – | **M**/**M** | –・V | V・V | V | D | **C** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/per_column_family) |

### 分布の自動選択

| ID | 何を調べたか（変えた条件） | 系列 | fit | n | d | K_TRUE | X / Y | seed | L・反復 | runner | 主要数値 | 品質 | リンク |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E-9C | Phase 9C 固定 K の family 選択（#74）：— | E | 未確認 | V | V | V | V/V | V・V | V・**M** | **M** | D | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/expfam/results/family_selection/optimizer_migration_bfgs_20260923/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/family_selection) |
| E-9D | Phase 9D family + K の同時選択（#75）：3 データ × 2 start | E | 未確認 | V | V | V | V/V | V・V | V・**M** | V | D | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/expfam/results/joint_family_k_selection/gate75b_20260927/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/joint_family_k_selection/gate75b_20260927) |
| E-129 | multi-support family selection（PR #12…：A/B/C | E + generic model（CMP… | 未確認 | V | V | V | **M**/V | V・V | V・**M** | V | D | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/4fa6159/expfam/results/family_selection/expanded_multisupport_20261006_034826/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/4fa6159/expfam/results/family_selection/expanded_multisupport_20261006_034826) |
| E-AF1 | AF1：{0,1} 列の Bernoulli vs Poisson 自動選択：M1 データ、Oracle / Auto / Wrong | E-3 | あり | V | V | V | V/V | V・V | V・V | V | V | **A** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/8a66894/expfam/results/auto_family_z_recovery/af1_20261109/full/preflight_m1.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/8a66894/expfam/results/auto_family_z_recovery/af1_20261109) |
| E-AF2 | AF2：Poisson vs CMP2 の自動選択：20 データ、1 条件、dense/sparse | E-3 + PR #129 | 未確認 | V | V | V | **M**/V | V・V | V・D | V | D | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/8a66894/expfam/results/auto_family_z_recovery/af2_20261110/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/8a66894/expfam/results/auto_family_z_recovery/af2_20261110) |
| E-AF3-5 | AF3–AF5：Q スコアの偏りと Laplace 型選択：AF2 と同じ 20 データ | E-3 + PR #129 | あり | V | V | **M** | **M**/**M** | V・**M** | V・**M** | V | D | **C** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/8a66894/expfam/results/auto_family_z_recovery/af3_selector_rescoring_20261113/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/8a66894/expfam/results/auto_family_z_recovery) |
| E-AF6 | AF6：Laplace 選択の新しい seed による確認：新 40 データ | E-3 + PR #129 | 未確認 | V | **M** | **M** | **M**/**M** | V・V | V・**M** | V | V | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/8a66894/expfam/results/auto_family_z_recovery/af6_laplace_confirmation_20261114/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/8a66894/expfam/results/auto_family_z_recovery/af6_laplace_confirmation_20261114) |
| E-AF7 | AF7：Poisson→CMP2 誤選択の原因分解：AF6 データ | E-3 | なし | – | – | – | –/– | –・– | –・– | V | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/8a66894/expfam/results/auto_family_z_recovery/af7_cmp_bias_analysis_20261114) |
| E-AF8 | AF8 / AF8-v2：loading を積分した選択基準の pilot：— | E-3 + PR #129 | あり | **M** | **M** | **M** | **M**/**M** | V・**M** | **M**・**M** | V | D | **C** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/8a66894/expfam/results/auto_family_z_recovery/af8_integrated_loading_20261115/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/8a66894/expfam/results/auto_family_z_recovery/af8_integrated_loading_20261115) |
| E-AF8V3 | AF8-v3：数値を認証した積分型選択（探索）：8,640 セル | E-3 + PR #129 | なし | D | – | – | –/– | V・– | –・– | V | D | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/8a66894/expfam/results/auto_family_z_recovery/af8v3_full_exploratory/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/8a66894/expfam/results/auto_family_z_recovery/af8v3_full_exploratory) |
| C-NIPS12 | NIPS12 など他の実データ：— | — | なし | – | – | – | –/– | –・– | –・– | – | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/reports/prior_art/prior_art_family_selection_joint_latent_models_20261013.md) |

### 潜在次元 K の選択

| ID | 何を調べたか（変えた条件） | 系列 | fit | n | d | K_TRUE | X / Y | seed | L・反復 | runner | 主要数値 | 品質 | リンク |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F-7B | Phase 7b：K 選択スコア診断 pilot（#37）：n∈{75,150}、K_TRUE=3、k=1..7 | E | あり | V | V | V | V/V | V・V | V・V | V | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/k_selection) |
| F-7E | Phase 7e：held-out K 選択 full pilot（#43）：K_TRUE=3、n=75、d=15、L=5、8 反復、K=1..7、2 start | E-scalar（Poisson-X） | あり | V | V | V | V/V | V・V | V・V | V | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/k_selection/heldout_full_pilot_20260824) |
| F-8B-SMOKE | Phase 8b S2c real smoke（#55）：K_TRUE=1、候補 K={2,3,4} | E-scalar | あり | D | D | V | D/D | V・V | D・D | V | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/k_selection/k_true_robustness_smoke_20260901) |
| F-8B-A1 | Phase 8b Attempt 1（中断）：— | E-scalar | あり | **M** | **M** | V | **M**/**M** | V・V | **M**・**M** | V | D | **C** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/k_selection/k_true_robustness_full_20260902) |
| F-8B-A2 | Phase 8b K_TRUE robustness（Attempt 2、…：K_TRUE∈{1,2,4,5} × 3 rep × estimand A/B、K=1..7 | E-scalar | あり | D | D | V | D/D | V・V | D・D | V | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/k_selection/k_true_robustness_full_attempt2_20260904) |
| F-CLEAN-N | clean true-K n-sweep（canonical clean …：K_TRUE∈{1,3,5}、n∈{50,75,100,150}、K=1..7、2 start、8 反復 | E-scalar | あり | V | V | V | V/V | V・V | V・V | V | V | **A** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/expfam/results/k_selection/clean_true_k_asymptotics_20260904/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/k_selection/clean_true_k_asymptotics_20260904) |
| F-EMCONV | EM 収束感度実験：K_TRUE=5、n×8 rep×K=1..7×2 start、iter 8→40 | E-scalar | あり | V | V | V | V/V | V・V | V・D | V | D | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/expfam/results/k_selection/em_convergence_sensitivity_20260905/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/k_selection/em_convergence_sensitivity_20260905) |
| F-HPB | historical P/B K_TRUE=5 n-sweep：K_TRUE=5、n∈{50,75,100,150}、5 trial、K=1..10 | historical（0.5 系統の EM） | あり | V | D | D | D/D | V・V | D・D | V | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/600108d/expfam/results/k_selection/historical_pb_k5_n_sweep_20260906) |
| F-GAUSSY | canonical Gaussian-Y の K 回復（2026-09-0…：K_0∈{1,3,5}、n∈{50,150,300}、w∈{0.25,0.5,1.0} | E-scalar | あり | V | V | V | V/V | V・V | V・V | V | V | **A** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/d610a8d/expfam/results/canonical_gaussian_y_k_selection_20260907) |
| F-9E | Phase 9E：K 反復性（20 データ）：n=75, d=12, K_TRUE=3, K=1..5 | E-3（G3/B6/P3） | あり | V | V | V | V/V | V・V | V・**M** | V | V | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/expfam/results/k_repeatability/phase9e_20260927/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/k_repeatability/phase9e_20260927) |
| F-9H-9J | Phase 9H–9J：Laplace 評価関数・pilot・IS 診断：— | E-3 | あり | V | V | V | V/V | V・V | V・**M** | V | D | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/expfam/results/laplace_is_diagnostic/phase9j_20260928/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/laplace_pilot) |
| F-9K | Phase 9K：C_Lap vs C_Q（20 データ）：n=75, d=12, K_TRUE=3 | E-3 | あり | V | V | V | V/V | V・V | V・**M** | V | V | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/expfam/results/lap_vs_cq_20/phase9k_20260928/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/lap_vs_cq_20/phase9k_20260928) |
| F-9M-9O | Phase 9M–9O：θ̂ の停留性と局所最適化：60 保存状態 | E-3 | なし | **M** | **M** | **M** | **M**/**M** | –・– | –・– | V | D | **C** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/expfam/results/derivative_failure_classification/phase9o_20260928/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/theta_stationarity) |
| F-9P-9V | Phase 9P–9V：w・密度・属性スケールの感度：w、w0 較正、f_scale | E-3 | あり | V | V | V | V/V | V・V | V・**M** | V | D | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/expfam/results/attribute_scale_design/phase9u_20260928/future_protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/relational_w_sensitivity) |
| F-9W-9Y | Phase 9W–9Y：信号をそろえた K_true 比較と K3→K4 …：K_true=1..4、rep01..10 | E-3 | 未確認 | V | V | V | V/V | V・V | V・**M** | V | D | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/expfam/results/matched_k_true_design/phase9w_20260928/future_protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results) |
| F-128 | #128：K 選択基準の同一 fit 比較（A1 / B）：A1：clean n-sweep の fit。B：paper-type generator、K_TRUE∈{1,3,5… | E-scalar | あり | V | V | V | V/V | V・V | V・V | V | D | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1b5030b/expfam/results/k_selection/clap_clean_nsweep_20261006/full/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/k_selection/k_criterion_comparison_analysis_20261104) |
| F-CLAP-REEVAL | C_Lap の checkpoint 再評価：448 fit × 6 checkpoint = 2,688 状態 | E-scalar | なし | D | – | D | –/– | –・– | –・– | V | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/k_selection/clap_checkpoint_reeval_20261105) |
| F-DSWEEP | d-sweep（n=75, K_GEN=5）：d∈{15,30,60}、8 rep、K=1..7、2 start | E-scalar | あり | V | V | D | D/D | V・V | D・D | V | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/k_selection/dsweep_k_selection_20261105) |
| F-R1R2 | R1/R2：MCEM 推定条件の頑健性と n×d：R1：L=10, 16 反復。R2：n×d | E-scalar | あり | V | V | D | **M**/**M** | V・V | V・D | V | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/expfam/results/k_selection/k_selection_robustness_20261105) |
| F-155 | #155：過小選択の因果分析と信号強度実験：n=75, K_GEN=5, 条件 A/B/C/D（F・w を強化） | E-scalar | あり | V | V | V | V/V | V・V | **M**・V | V | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/bada1d2/expfam/results/k_selection/historical_canonical_causal_20261106) |
| F-156 | #156：MCEM 安定性と信号をそろえた K3/K5：#155 の 128 fit を 64 反復まで | E-scalar | あり | V | V | V | **M**/**M** | V・V | **M**・D | V | D | **C** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/bada1d2/expfam/results/k_selection/mcem_stability_signal_matched_20261106/analysis/experiment_protocol.csv) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/bada1d2/expfam/results/k_selection/mcem_stability_signal_matched_20261106) |
| F-157 | #157：MCEM の推定変動と MC 誤差の分離：128 状態 + L=20/50 各 16 fit | E-scalar | あり | **M** | **M** | **M** | **M**/**M** | D・V | D・**M** | V | D | **C** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/bada1d2/expfam/results/k_selection/mcem_noise_separation_20261106) |
| F-158 | #158：原論文・MATLAB・Python・ExpFam の歴史的比較：— | — | なし | **M** | **M** | V | V/V | –・– | –・– | V | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/bada1d2/reports/k_selection_theory/historical_code_lineage_audit_20261106) |
| F-159 | #159：Q_Z を除いた原論文型基準での再採点：3,076 保存状態 × 5 基準 | E-scalar | なし | **M** | **M** | V | **M**/**M** | –・– | –・– | V | D | **C** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/bada1d2/expfam/results/k_selection/qz_criterion_existing_states_20261106) |
| F-160 | #160：モデル・推定器・基準の公平な比較：n∈{75,150}、信号、観測率、生成器 | E-scalar + OLD（0.5） | あり | V | V | V | V/D | V・V | V・V | V | D | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/bada1d2/expfam/results/k_selection/fair_model_estimator_criterion_comparison_20261106/experiment_protocol.csv) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/bada1d2/expfam/results/k_selection/fair_model_estimator_criterion_comparison_20261106) |
| F-161 | #161：n=75 での OLD 推定器 vs 系列 E：n=75、G_canon / G_old | E-scalar + OLD | あり | V | V | V | **M**/**M** | V・V | V・V | V | D | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/bada1d2/expfam/results/k_selection/n75_old_vs_expfam_controlled_20261106/experiment_protocol.csv) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/bada1d2/expfam/results/k_selection/n75_old_vs_expfam_controlled_20261106) |
| F-162 | #162：失敗した弱信号データでの OLD vs E（全ペア）：#160 E2 の 4 データ、全ペア観測 | E-scalar + OLD | あり | V | **M** | V | D/D | V・**M** | **M**・D | V | D | **C** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/bada1d2/expfam/results/k_selection/weak_n75_old_vs_expfam_allpairs_20261107) |
| F-164 | #164：C_Q・C_Lap・C_ELBO・条件付き尤度の統制比較：K_GEN=5、d=15、n∈{75,150}、Gauss/Pois-X、弱/強信号 | E-scalar | あり | V | D | V | V/**M** | V・V | **M**・**M** | V | V | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/bada1d2/expfam/results/k_selection/criterion_impact_validation_20261108/experiment_protocol.csv) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/bada1d2/expfam/results/k_selection/criterion_impact_validation_20261108) |
| F-165 | #165：原論文 Fig.3 に近い条件での基準比較：K_GEN=1,3,5,7,9 各 1 データ × 10 試行、n=150、d=15、F~N(0,10)、全ペア | E-scalar | あり | V | V | V | V/D | V・**M** | V・V | V | V | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/bada1d2/expfam/results/k_selection/paper_fig3_condition_criteria_20261109/experiment_protocol.csv) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/bada1d2/expfam/results/k_selection/paper_fig3_condition_criteria_20261109) |
| F-KEXP2-MAIN | K-exp2 本実験：原論文 Exp.2 条件の一般化（8,000 fit）：series O/A/B/C × regime a/b × K_TRUE 1,3,5,7,9 × 10 データ、K=1… | E-scalar | あり | V | V | V | V/V | V・V | V・V | V | V | **A** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/9d12dc4/expfam/results/k_selection/k_exp2_generalization_20261125/protocol_v2/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/9d12dc4/expfam/results/k_selection/k_exp2_generalization_20261125/main) |
| F-KEXP2-CONV | K-exp2 収束感度（600 fit）：A-b-9、B-b-9、O-b-9 を 40 反復まで（checkpoint 10/20/40） | E-scalar | あり | **M** | **M** | V | V/V | V・V | V・V | V | V | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/9d12dc4/expfam/results/k_selection/k_exp2_generalization_20261125/convergence_sensitivity/protocol/preflight.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/9d12dc4/expfam/results/k_selection/k_exp2_generalization_20261125/convergence_sensitivity) |
| F-KEXP2-CLAP | K-exp2 C_Q と C_Lap の比較（6,600 評価）：main10（series O/A/B の 300 データ。series C は対象外）+ conv40（30 データ） | E-scalar | なし | V | – | V | –/– | V・– | –・D | V | V | **A** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/9d12dc4/expfam/results/k_selection/k_exp2_generalization_20261125/clap_comparison/protocol/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/9d12dc4/expfam/results/k_selection/k_exp2_generalization_20261125/clap_comparison) |
| F-KEXP2-A | 追加検証 A：C_Lap の開始点感度（1,400 評価）：main10 4 セル + conv40 3 セル、Z_last vs Z_mean | E-scalar | なし | – | – | V | –/– | V・– | –・– | V | D | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/9d12dc4/expfam/results/k_selection/k_exp2_generalization_20261125/clap_followup_20261126/expA/protocol/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/9d12dc4/expfam/results/k_selection/k_exp2_generalization_20261125/clap_followup_20261126/expA) |
| F-KEXP2-B | 追加検証 B：NOT_STATIONARY の max_iter=400 …：NOT_STATIONARY 52 + HESSIAN_NOT_PD 5 | E-scalar | なし | – | – | V | –/– | V・– | –・– | V | D | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/9d12dc4/expfam/results/k_selection/k_exp2_generalization_20261125/clap_followup_20261126/expB/protocol/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/9d12dc4/expfam/results/k_selection/k_exp2_generalization_20261125/clap_followup_20261126/expB) |
| F-KEXP2-C | 追加検証 C：混合分布 X の高次元 K 選択（600 fit）：K_TRUE 5/7/9 × 10 データ、K=1..10、2 init、L=10、10 反復、regime b の信号 | E-3（per-column consis… | あり | V | V | V | V/V | V・V | V・V | V | V | **A** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/9d12dc4/expfam/results/k_selection/k_exp2_generalization_20261125/mixedx_highk_20261126/protocol/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/9d12dc4/expfam/results/k_selection/k_exp2_generalization_20261125/mixedx_highk_20261126) |

### 追加実験 A・B と原因分析・訂正

| ID | 何を調べたか（変えた条件） | 系列 | fit | n | d | K_TRUE | X / Y | seed | L・反復 | runner | 主要数値 | 品質 | リンク |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TG-A | Experiment A：Y の観測量 × X の分布誤指定（全列共通 f…：n=150、d=15、K_TRUE=3、全列 Bernoulli-X・Bernoulli-Y、dense（学習 Y 1… | E-scalar（DualExpFamLS… | あり | V | V | V | V/V | V・V | V・V | V | V | **A** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1ef9096/expfam/results/k_selection/two_gaps_20261127/expA/protocol/protocol.json) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1ef9096/expfam/results/k_selection/two_gaps_20261127/expA) |
| TG-B | Experiment B：Fisher 情報の trace をそろえた G…：n=150、d=15、K_TRUE=9、候補 K=1..10、regime b の信号、10 データ × 2 初期値、… | E-scalar（prototype） | あり | V | V | V | V/V | V・V | V・V | V | V | **A** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1ef9096/expfam/results/k_selection/two_gaps_20261127/expB/protocol/go_decision.md) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1ef9096/expfam/results/k_selection/two_gaps_20261127/expB) |
| TG-DIAG | A・B の原因分析（保存済み状態のみ、新しい fit なし）：NPZ 760 個（SHA-256 一致）、seed から再構成したデータ（hash 一致） | E-scalar（prototype） | なし | – | – | – | –/– | D・– | –・– | V | V | **B** | [protocol](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/2f71e61/expfam/results/k_selection/two_gaps_20261127/diagnostics_20261127/analysis_protocol.md) [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/2f71e61/expfam/results/k_selection/two_gaps_20261127/diagnostics_20261127) |
| TG-CORR | 原因分析の数理的な最終訂正（forward correction）：per_fit.csv・NPZ（var_z・Z_mean）・既存の診断 CSV | E-scalar（prototype） | なし | – | – | – | –/– | –・– | –・– | V | V | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/c035df7/expfam/results/k_selection/two_gaps_20261127/diagnostics_20261127/forward_correction_20261127) |

### 理論・監査

| ID | 何を調べたか（変えた条件） | 系列 | fit | n | d | K_TRUE | X / Y | seed | L・反復 | runner | 主要数値 | 品質 | リンク |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-THEORY0718 | 理論監査 2026-07-18（同時分布・1/2・BIC の位置づけ）：— | — | なし | – | – | – | –/– | –・– | –・– | – | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/reports/theory_audit/theory_audit_report_20260718.md) |
| D-PC-MATH | per-column 数式監査（31/31 PASS）と objectiv…：— | F → E | なし | – | – | – | –/– | –・– | –・– | – | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/reports/per_column_family) |
| E-HARDCASE | family 選択の hard case 監査：— | — | なし | – | – | – | –/– | –・– | –・– | – | **M** | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/9beadce/reports/family_selection/family_selection_hard_case_audit_20261002.md) |
| F-7A | Phase 7a：K 選択の理論監査（#35）：— | — | なし | – | – | – | –/– | –・– | –・– | – | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/reports/k_selection_theory/k_selection_theory_audit_20260822.md) |
| F-163 | #163：K 選択基準の数理監査：— | — | なし | D | – | – | –/– | –・– | –・– | V | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/bada1d2/reports/k_selection_theory/bic_theory_candidate_assessment_20261108) |
| G-IDENT | true-K 識別可能性の理論監査（P1–P8）：— | — | なし | – | – | – | –/– | –・– | –・– | – | D | **B** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/1b5030b/reports/identifiability/true_k_identifiability_hardened_20260904.md) |
| G-LOCAL-ONLY | 未追跡の理論監査 4 件（2026-09-07）：— | — | なし | – | – | – | –/– | –・– | –・– | – | D | **C** | — [一次資料](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/d946910/reports/research_scope) |

## 4. 条件の記録が特に不足している実験

`experiment_record_gaps.csv` の重要度は次の 3 段階です。
- **REPRODUCIBILITY**：再現（再生成・再実行）に必要な情報が無い。
- **INTERPRETATION**：科学的な解釈・比較に影響する。
- **NOTATION**：表記だけの問題。

### 4.1 再現性に影響する不足（REPRODUCIBILITY、123 行・58 記録）

| 記録 | 不足している情報 | 影響 |
|---|---|---|
| `B-OLD05-EXP1-4`（学会予稿の本文採用実験、旧 0.5 系列） | runinfo・実行時の git head。結果 CSV には `trial` 列だけがあり、seed 列が無い | seed・L=5・8 反復・n=150・d=15・K_TRUE=3 は runner（`expfam/src/exp_scenario_lib.py`、結果 CSV と同じ commit `7c3040f` で追加）の定数として **DOCUMENTED**。run 時にこのコードで実行されたことは、出力からは確認できない |
| `A-REPRO`（Python 再現 vs 一般化実装） | n・d・seed・L・反復数・runinfo・生成器 | 主要数値（RMSE(Z) 差 < 0.001）は照合済み。条件は registry と報告書の記載に頼る |
| `B-FIXED-EXP1`〜`EXP4`（fixed 系列） | L・MCEM 反復数・runinfo・生成器（EXP1 は runner も） | n・d・K_TRUE・分布族・seed は結果 CSV の列で **VERIFIED**（例：`seed_data=4000+100·trial` は旧 0.5 系列の runner の規則と一致）。L・反復数は出力に無い |
| `C-WINE`・`C-CORA-OLD`・`C-ML-INSAMPLE`・`C-RECON`（初期の実データ） | runinfo・実行時の git head・L・反復数・前処理 | 2026-08 以前の実データ記録。`C-ML-INSAMPLE` は runner も特定できない |
| `A-HALF`・`B-OLD05-AUX`・`B-FIXED-GRID` | 条件の大半と runinfo。主要数値も未照合 | 記録品質 C |
| `F-CLAP-REEVAL`・`F-162` | 保存状態の NPZ が repository 外（sha256 は runinfo に記録） | GitHub だけでは再評価を再現できない |
| `F-164`・`F-165` | 実験単位の runinfo（`F-165` は fit 単位の JSON がある） | `F-164` は 448 fit の確認実験だが、runinfo を特定できない |

### 4.2 解釈に影響する不足（INTERPRETATION、主なもの）

| 記録 | 不足 | 備考 |
|---|---|---|
| `E-AF6` | fit 数（台帳は「未確認」、completeness は CONFIRMED） | 主要数値は照合済み。分母の明記が必要 |
| `D-MX2` | fit 数（台帳は「未確認（2 seed 系列）」）と重複実行事故の扱い | `duplicate_incident.csv` に「最初の記録のみ採用」と記録あり |
| `C-OD-*`・`C-ML-SHAREDZ`・`E-9C`・`E-9D`・`E-AF2`・`F-9W-9Y` など 14 件 | fit 数が台帳に無い | 一次資料の行数から数えられる可能性がある（未実施） |
| `A-PAPER` | 原論文の Exp.2 の集計手順・Z の評価点・ln L の正規化 | 本文から特定できない（既存台帳と同じ） |

### 4.3 evidence 列が曖昧で、場所を特定できなかった記録（LINK_AMBIGUOUS、NOTATION）

`D-E2A`（「expfam/results/（e2a）」）、`F-8B-SMOKE`（「（8b smoke）」）、`F-EMCONV`（「（em convergence）」）、`F-128`（派生表だけを指す）の 4 件です。
registry の結果パスで補って照合しました。補ったパスは `experiment_provenance_map.csv` の `root_note` 列にあります。台帳の改訂は Human Gate です。

## 5. 数式・実験条件の食い違い

### 5.1 自動照合の結果（新しい食い違いは見つからなかった）

- **protocol と runinfo・manifest（同じディレクトリ）**：
  - n・d・K_TRUE・L・反復数・X/Y の分布族・w/w0・信号強度・seed を比べました。値の違いは書式だけでした（`[1, 3, 5]` と `1, 3, 5`、`75.0` と `75` など）。
  - seed は、protocol に式があるものについて manifest の値を手で再計算し、一致を確認しました。
    - `F-CLEAN-N`：`810000 + 10000·K_TRUE + 10·n + rep` → manifest `820501`。
    - `F-KEXP2-MAIN`：`20261125000 + 100000·series + 1000·K_TRUE + rep` → `20261126000`・`20261134000`。`F-KEXP2-CONV` の seed は本実験の manifest から複製されていることも確認。
    - `F-KEXP2-C`：`30261126000 + 1000·K_TRUE + rep` → `30261131000`。
    - `TG-A`：`31,261,127,000 + rep` → `per_fit.csv` の `31261127000`。
- **台帳の conditions 列と一次資料**：n・d・K_TRUE・L・反復数を照合しました。食い違いは無く、書式の差が 2 件（`F-160`、`F-164`）だけでした。
- **実行時の git head**：記録のある 59 件すべてが repository に存在しました（未 push・rebase で消えた head は無い）。

照合できたのは、両方の資料に値がある項目だけです。片方が MISSING の項目は照合していません。

### 5.2 Experiment A の gram 相関（CONFLICT・NOTATION、`GAP-295`）

| 資料 | 値 | SHA |
|---|---|---|
| `results_report.md`・追補台帳・13 章の初出 | 0.241 | `1ef9096` / `8da446a` |
| 一次 CSV `expA/analysis/primary_results.csv` | 0.24046（四捨五入で 0.240） | `1ef9096` |

0.24046 を切り上げた表記の差です。主張は変わりません。既存の記録は書き換えず、引用時は 0.240 と書きます（13 章・追補の `TG-A-06` に記載済み）。

### 5.3 Experiment B の Poisson の情報行列の条件数（CONFLICT・NOTATION、`GAP-296`）

| 資料 | 値 | SHA |
|---|---|---|
| `fairness_audit.md`・`frozen_protocol_B.md` | 1.14〜1.17 | `1ef9096` |
| 一次 CSV `expB/tech_check_v2/information_diagnostics.csv` | 1.11〜1.22 | `1ef9096` |

trace をそろえる設計（GM の分散）には影響しません。引用時は一次 CSV の範囲を使います。

### 5.4 C_Q の K 増加条件（CONFLICT・INTERPRETATION、解消済み、`GAP-297`）

| 資料 | 内容 | SHA |
|---|---|---|
| `diagnostic_report.md`（TG-DIAG） | 誤った不等式 | `2f71e61` |
| `forward_correction.md`（TG-CORR） | $2\Delta\ell > n(1+\ln 2\pi) + (d-K+1)\ln n$ | `c035df7` |

forward correction を正とします。2f71e61 の報告書は書き換えていません。13 章 §4 と 07 章 §5 は訂正後の式です（270/270 一致、別プロセスでの確認 17/17）。
TG-DIAG の記録品質は、この訂正があるため B です。

### 5.5 C_Q と C_Lap の罰則の違い（定義の違い、`GAP-298`）

- **C_Q**：$p_K = Kd - K(K-1)/2$ に、Gaussian-X のとき分散 $d$ 個を加える。
- **C_Lap**：$d_K$ に分散を含めない（K によらない定数のため）。

定義上の違いで誤りではありません。K の選択（argmin）には影響しませんが、C_Q と C_Lap の値や、分布族の違う条件間の基準値は直接比べられません（公平性 D）。

### 5.6 件数の計上（`GAP-300`）

- `experiment_inventory.csv` の 88 件は、実験・監査・理論・文献・設計を含む**記録の数**です。
- 追補は TG-A・TG-B が新しい fit を伴う実験、TG-DIAG・TG-CORR は同じ fit の事後分析と訂正です。
- 本台帳の `involves_new_fits` 列で区別しました。「88 件の実験」「92 件の実験」とは書かず、たとえば「92 記録のうち新しい fit を伴うもの 51 件（ほか未確認 16 件）」と書きます。

`F-KEXP2-CLAP`・`F-KEXP2-A`・`F-KEXP2-B` も、`F-KEXP2-MAIN` の保存状態の再評価で、新しい fit はありません（`parent_id` で親を示した）。

### 5.7 where 列と commit 列の SHA の違い（NOTATION、`GAP-307`）

19 記録では、台帳の commit 列（結果を作った commit）と、where 列の SHA（ファイルを読んだ commit）が異なります。
`experiment_provenance_map.csv` では `result_commit` と `read_sha` を分けました。どちらの SHA も repository と remote に存在します。

## 6. 実装系列が 1 つの記録に混在している箇所

次の記録は、比較そのものが目的で複数の系列を含みます（`GAP-299`）。
結果集の 03・04・07 章は系列を分けて書いており、異なる系列の数値を同じ表の同じ列に並べている箇所は見つかりませんでした。
引用するときは系列名を併記し、旧 0.5 系列には KI-001 の注記（「Newton 方向が全体として正しいとは断定できない」）を付けてください。

| 記録 | 系列 | 内容 |
|---|---|---|
| `A-REPRO` | B（0.5）と reproduction（0.5） | 先行研究の Python 再現と一般化実装の同条件比較 |
| `A-HALF` | B と C | 1/2 係数の有無の比較 |
| `B-FIXED-GRID` | C と B | fixed 系列の誤指定グリッド（旧版との比較を含む） |
| `C-WINE-AUDIT` | B/C | Wine の旧 0.5・fixed・論文再現の突合 |
| `C-OD-DIAG` | C/D | 過分散の診断 |
| `D-PC-MATH` | F → E | per-column の数式監査（legacy numerics から consistent へ） |
| `D-COMPL`・`D-COVER`・`C-ML-USERDISJ` | E+F | per-column consistent |
| `F-HPB` | historical（0.5 系統の EM） | 過去の K=5 の n-sweep |
| `F-160`・`F-161`・`F-162` | E-scalar + OLD（0.5） | 旧推定器（`reproduction/src/model.py`）と E 系列の公平比較 |

## 7. registry の状態と統合案（実施していない）

### 7.1 現状

| 場所 | `EXPERIMENT_REGISTRY.md` の状態 |
|---|---|
| main `1b5030b` | 549 行。K-exp2・A・B の行は無い |
| `experiment/k-selection-exp2-generalization-20261125` `9d12dc4` | main ＋ 12 行（K-exp2 の 5 行） |
| `experiment/k-exp2-two-gaps-20261127` `1ef9096` → `audit/two-gaps-diagnostics-20261127` `2f71e61` → `audit/two-gaps-diagnostics-correction-20261127` `c035df7` | `9d12dc4` と同じ（A・B・原因分析・訂正の行は**どこにも無い**） |
| 研究結果集 `f8c84d8` → `8da446a` → 本ブランチ | main と同じ（変更なし） |

`9d12dc4` → `1ef9096` → `2f71e61` → `c035df7` は一直線の祖先関係です（`git merge-base --is-ancestor` で確認）。
ほかにも registry の末尾に追記している未統合ブランチがあります：`bada1d2`（+26 行）、`8a66894`（+35）、`856d475`（+9）、`d610a8d`（+20／−1）、`600108d`（+19）、`4fa6159`（+16）。
後の 3 つは、それぞれの分岐元（当時の main）からの差分です。

### 7.2 統合案（研究者が判断・実施する）

1. **A・B・原因分析・訂正の行**は、`c035df7` から分岐した新しいブランチで registry の末尾に追記します（追記のみ、既存行は変更しない）。K-exp2 の行と同じ系統に置くためです。
2. **main への統合順**：K-exp2 → two-gaps → 原因分析 → 訂正は一直線なので、`c035df7` を含むブランチ（またはその先の registry 追記ブランチ）を 1 回統合すれば K-exp2 の行も入ります。
3. **研究結果集のブランチ**（`f8c84d8`・`8da446a`・本ブランチ）は registry を触らないので、統合の順番によらず registry では衝突しません。
4. **ほかの未統合ブランチ**も末尾に追記しているため、main に順に統合すると registry の末尾でテキストの衝突が起きます。解消方法は「両方のブロックを残し、日付順に並べる」です。既存行の書き換え・削除はしません。
5. **追記する行の案**（列は既存表と同じ。状態は K-exp2 の行に合わせて `current_support`、原稿採用 ✗）：

| 実験ID | 内容 | 実装/スクリプト | 結果 | 図 | 状態 | 原稿採用 | 注意 |
|---|---|---|---|---|---|---|---|
| two_gaps_expA_20261127 | Y の観測量（dense / sparse 10%）× X の誤指定（Bernoulli→Poisson）。全 15 列 Bernoulli-X・Bernoulli-Y、n=150、d=15、K=3、20 データ × 2 初期値、L=10、10 反復 | `tools/research_audit/run_two_gaps_20261127.py` | `expfam/results/k_selection/two_gaps_20261127/expA/` | `expA/analysis/figures/` | current_support | ✗ | 160/160 OK。I=+0.211（20/20 正）。gram 相関は 0.240（報告の 0.241 は丸め）。prototype・探索的 |
| two_gaps_expB_20261127 | Fisher 情報の trace をそろえた Gaussian-X（GM）と Poisson-X（P）、分散 0.1 の Gaussian-X（G01）。n=150、d=15、K_TRUE=9、候補 K 1..10、10 データ × 2 初期値、L=10、10 反復 | 同上 | `expfam/results/k_selection/two_gaps_20261127/expB/` | `expB/analysis/figures/` | current_support | ✗ | 600/600 OK。C_Q：G01 10/10・GM 0/10・P 0/10。条件数は一次 CSV で 1.11〜1.22 |
| two_gaps_diagnostics_20261127 | A・B の保存状態の事後分析（新しい fit なし） | `tools/research_audit/diagnose_two_gaps_20261127.py` | `two_gaps_20261127/diagnostics_20261127/` | 同ディレクトリ `figures/` | current_support | ✗ | C_Q の不等式は forward correction（`c035df7`）で訂正 |
| two_gaps_forward_correction_20261127 | C_Q の厳密な K 増加条件・2Δℓ・C_Lap の式の確認（新しい fit なし） | `tools/research_audit/correct_two_gaps_diagnostics_20261127.py` | `diagnostics_20261127/forward_correction_20261127/` | なし | current_support | ✗ | 2Δℓ > n(1+ln2π)+(d−K+1)ln n（270/270）。別プロセスでの確認 17/17 |

## 8. あえて変更しなかったもの

- 既存の結果 CSV・NPZ・protocol・runinfo・報告書（`2f71e61` の誤った不等式、0.241、1.14〜1.17 の表記を含む）。食い違いは本章と gaps CSV に記録しただけです。
- 01〜13 章の本文と、既存の 5 つの台帳・追補の 4 台帳（`build_experiment_records_audit.py --check` が `8da446a` からの非変更を確認）。
- 台帳の evidence 列の曖昧な記載（4 件）と、run_count が「未確認」の記録（16 件）。台帳の改訂は Human Gate です。
- `EXPERIMENT_REGISTRY.md`（§7 は案のみ）。
- 実験条件の不明な項目。現行 runner のデフォルト値で埋めていません。旧 0.5 系列の値は、結果と同じ commit の runner の定数として DOCUMENTED に留めました。
- 記録品質 C の記録の扱い（研究結果として誤りという意味ではない）。

## 9. 自動照合の範囲と限界

- **読んだもの**：各記録の固定 SHA の git tree、runinfo・protocol・manifest（JSON・CSV、「項目名, 値」形式を含む）、fit 単位の JSON（実験単位の runinfo が無いときに 3 件）、結果 CSV のヘッダと条件列・seed 列、その SHA の `EXPERIMENT_REGISTRY.md` の該当行、台帳 2 種。
- **読んでいないもの**：
  - 報告書・protocol の Markdown 本文（A・B の凍結 protocol と旧 0.5 系列の runner 定数だけは人手で読み、`MANUAL_VALUES` として記録）。
  - runner の中の生成関数の呼び出し。
  - GitHub の Issue・PR の本文とレビューコメント。
  - ローカルにだけある NPZ と作業ツリー。
- **キー名による抽出**：
  - JSON・CSV のキー名で項目を判定しています。意味の近い別物を拾わないよう、MCEM 反復数は `num_iter`・`em_iterations` などに限りました（BFGS の `n_iter` などは除外）。
  - 集計 CSV の `n` 列（データ数のことがある）は、seed 列を持つ fit 単位の CSV でだけ採用しています。
  - それでも誤った対応が残る可能性があるので、`extraction` 列に JSON のキーパスを、`source` 列にファイルを残しました。
- **MISSING の意味**：上の範囲で見つからなかったことです。報告書本文や Issue に書かれている可能性はあります。「不明」を「無い」と読み替えないでください。
- **人の第三者査読ではありません**。同じ AI が書いたスクリプトによる照合です。

## 10. GitHub でどこを見れば各実験を確認できるか

1. **記録 ID を決める**：本章 §3 の表、または [`data/experiment_provenance_map.csv`](data/experiment_provenance_map.csv)。
2. **一次資料**：provenance map の `github_tree`（固定 SHA のディレクトリ）、`protocol`・`runinfo`・`runner`・`figures`・`reports`。
3. **条件の値と出典**：[`data/experiment_conditions_audit.csv`](data/experiment_conditions_audit.csv) を `experiment_id` で絞る。`condition_label=ALL` が記録単位、それ以外は条件別（一次資料ファイルごと）の子行。
4. **主要数値の照合**：[`data/primary_results_index.csv`](data/primary_results_index.csv)（96 項目）、[`data/addendum_20261127_primary_results.csv`](data/addendum_20261127_primary_results.csv)（38 項目）。
5. **不足と食い違い**：[`data/experiment_record_gaps.csv`](data/experiment_record_gaps.csv)。
6. **研究上の解釈・公平性**：02〜13 章、09 章（比較公平性 A〜D）。

## 11. 再現方法と品質検査

```bash
git fetch origin
python docs/research_results/scripts/build_experiment_records_audit.py          # 3 つの CSV を生成して検査
python docs/research_results/scripts/build_experiment_records_audit.py --check  # 検査のみ
python docs/research_results/scripts/check_links.py                             # 章のリンク検査
```

スクリプトはモデルを import せず、`git ls-tree` と `git show <sha>:<path>` で固定 SHA の内容を読むだけです。既存のファイルには書き込みません。同じ入力からは同じ出力になります（2 回の生成で md5 一致）。

`--check` の検査項目：
- CSV の構文と列数、必須列。
- 記録 ID の一意性と、既存の台帳 2 種との ID 対応（92/92）。
- (record_id, 項目) の重複と、記録単位の行が 60 項目そろっていること。
- 状態が 5 種類だけであること、VERIFIED に出典があること、`parent_id` が実在すること、`MANUAL_VALUES` のキーが正しいこと、gaps の記録 ID。
- 読んだ SHA・結果 commit の存在と remote branch への包含。
- 新しい fit を伴う記録に系列が書かれていること。
- 既存台帳 7 ファイルが `8da446a` から変わっていないこと。

結果：**PASS**。
