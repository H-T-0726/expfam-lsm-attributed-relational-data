# K 選択基準の同一 fit 比較 — 実験 A1 確定結果と実験 B（原論文型 generator）の結果（Issue #128, 2026-11-04）

- 位置づけ: **CHARACTERIZATION**（承認済み Issue #128 scope の実行と分析）。新しい実験・条件変更は行っていない。
- 系列: **lineage E**（objective-consistent prototype, `DualExpFamLSMConsistent`, `numerics_mode="consistent"`）。
  **MANUSCRIPT_ADOPTION_STATUS = NOT_DECIDED_BY_HUMAN**（本文採用は人間が未決定）。
- ラベル: FACT（一次成果物・コード・原論文で確認）／DERIVED（式から導出）／OBSERVED（この固定条件の結果）／
  INTERPRETATION（解釈）／HYPOTHESIS（未検証の仮説）／DECISION（人間の判断が必要）。
- 数はすべて「この固定条件の R データセット中 x 個」であり、accuracy・recovery rate・一般的な優越性・一致性ではない。
- **実験 B は原論文（Mikawa et al. 2024）Experiment 2 の再現ではない**（相違は §5.2）。

---

## 0. 先生に説明するための要約（2〜3 分）

1. **何をしたか。** 真の潜在次元 K_TRUE が既知の人工データで、各候補 K のモデルを 1 回だけ推定し、
   **同じ推定結果の上で**複数の K 選択基準を計算して比較した。比較したのは、現行の Q 関数ベースの基準 `C_Q`
   （Schwarz BIC ではない）と、観測データの周辺尤度をラプラス近似した基準の候補 `C_Lap`（厳密な周辺尤度ではない）である。
   実験 B では、原論文の式(26)に倣って Z に条件づけた尤度を使う診断用の基準 `C_cond_*` も同じ fit 上で計算した。
2. **弱い信号・小さい n（実験 A1: Poisson-X, n=50〜150, K_TRUE=1,3,5）** では、`C_Q` は 64 データセット中 37 で、
   `C_Lap` は 47 で K_TRUE を選んだ（`C_Lap` は 4 データセットで数値的に計算が完了しなかった）。
   誤選択は**両基準とも過小選択だけ**で、過大選択は 0 件だった。2 つの基準が異なる K を選んだ 24 データセットでは、
   **すべて `C_Lap` の方が大きい K（K_TRUE 寄り）**を選んでいた。
3. **原論文型の強い信号（実験 B: Gaussian-X, n=150, K_TRUE=1,3,5,7,9）** では、
   `C_Q` は 25/25 で K_TRUE を選んだ。`C_Lap` は計算が完了した 19/19 で K_TRUE を選び、残り 6 は数値的に未完了だった。
   したがって、**この条件では `C_Lap` が `C_Q` より K_TRUE を選びやすいという証拠は得られなかった**（両者が異なる K を選んだデータセットは 0）。
   一方、Z に条件づけた尤度＋同じ罰則 `p_K ln n` の診断基準 `C_cond_*` は **25/25（完了分すべて）で候補の上端 K=9〜10 を選んだ**。
4. **言えること。** `C_Q` と `C_Lap` の違いが選択に効くのは、K_TRUE 直下の次元を増やしたときの当てはまりの改善が
   「罰則」と同程度の大きさになる弱い信号の条件である。強い信号では両者は一致する。
   Z を積分しない（`p(Z)` を含まない）条件付き尤度に `p_K ln n` の罰則だけを加えた基準は、本研究の実装・評価点では過大選択した。
5. **言えないこと。** `C_Lap` の一般的な優位、一致性、K 選択の解決、原論文の BIC が誤っていること
   （原論文が Z をどこで評価したかは本文から特定できない）、過小選択の原因の断定。
6. **提案。** K 選択を修論の主貢献にするより、「信号の強さによって基準の違いが現れる・現れない」ことを示す
   補助的な章として位置づけるのが、現在の証拠の強さに見合うと考える（判断は Human Gate）。

---

## 1. 調査した GitHub / ローカルの状態（2026-11-03〜04）

| 項目 | 状態 | 根拠 |
|---|---|---|
| main（origin） | `ee4be23`（PR #127 merge） | `git log origin/main` |
| Issue #128 | OPEN。コメントなし。scope は本文（Human Gate 2026-10-06） | `gh issue view 128` |
| Issue #138 | OPEN。research reframing（zero-experiment）。対応ブランチ・PR は未作成 | `gh issue view 138`, `gh pr list` |
| #128 の PR | 作業前は**未作成** | `gh pr list --state all` |
| `experiment/128-k-criterion-comparison` | worktree `D:\tento\kennkyu-128-kcrit`、作業前 HEAD `0310230` = origin、working tree clean | `git status`, `git rev-parse` |
| A1 成果物 | pilot / full / 図 / 監査 / STATE_ARCHIVE が commit 済み | commit `f6ace60`, `97d36ca`, `0310230` |
| B 成果物（作業前） | **未実行**。`expfam/results/k_selection/paper_condition_k_criteria_20261006/` も状態ディレクトリも存在せず、他 worktree にもなし | `ls`, `find` |
| ローカル未反映 | 主 worktree（`exp/multisupport-family-selection`）に untracked の `.local/`、`reports/identifiability/gaussian_y_k_identifiability_adversarial_audit_20260907.md`、`reports/research_scope/`（2026-09-07 作成）。**#128 とは無関係で、触れていない** | `git status` |
| 実行環境 | Python 3.13.14、NumPy 2.3.5、SciPy 1.16.3（`.python-version`・requirements と一致）、Windows 11 | preflight 出力 |

---

## 2. 基準の数式と意味

### 2.1 原論文（一次確認: `paper/A_study_on_latent_structural_models_for_binary_rel.pdf`）

- **FACT** 式(16): `L = Π_i p(x_i | F, z_i, Σ_x, …) · Π_{i≠j} p(y_ij | z_i, z_j, w0, w)^{1/2}`。
  **z に条件づけた尤度で、`p(Z)` を含まず Z も積分していない**。
  （`Π_{i≠j}(·)^{1/2}` は y_ij = y_ji なので無向ペア `i<j` の積と同じ量。KI-001 の Newton 係数 1/2 とは別の話である。）
- **FACT** 式(26): `BIC = −2 ln L + ((k+1)d − k(k−1)/2) ln n`。論文は採用理由として一致性を**述べている**が、このモデルでの証明は本文にない。
- **FACT** Table I の Experiment 2: d=15、n=150、L=10、反復 10、試行 10、候補 k=1..10、真の k ∈ {1,3,5,7,9}。
  推定に使った Z の評価点（どのサンプルか・平均か・mode か）と試行の集約方法は本文から特定できない。

### 2.2 本研究で比較した基準（`expfam/src/experimental/k_criterion_panel.py`）

共通の罰則: `P_θ = p_K ln n`。
A1 は `p_K = Kd − K(K−1)/2`（Poisson-X）、B は `p_K = Kd − K(K−1)/2 + d`（Gaussian-X の Σ_X を含む。原論文の数え方と同じ。w0, w は数えない）。

| 基準 | 定義 | 意味 | 呼んではいけない名前 |
|---|---|---|---|
| `C_Q`（主） | `−2 Q_strict + P_θ = D_K + P_Z + P_θ`。`Q_strict = (1/L)Σ_l ln p(X, Y, Z^(l) | θ̂)`、`D_K = −2(Q_X + Q_Y)`、`P_Z = −2(1/L)Σ_l ln p(Z^(l))` | E-step サンプルで Z を「完全データ」として扱う complete-data / ICL 型の基準 | Schwarz BIC |
| `C_Lap`（主） | `−2 ℓ_Lap + P_θ`、`ℓ_Lap = Φ(Ẑ) + (nK/2) ln 2π − ½ ln|H(Ẑ)|`、`Φ(Z) = ln p(X, Y | Z, θ̂) + ln p(Z)`、`Ẑ = argmax Φ`、`H = −∇²Φ(Ẑ)`。分解 `= fit_mode + volume + P_θ`（`fit_mode = −2 ln p(X,Y|Ẑ,θ̂)`、`volume = ‖Ẑ‖²/σ_z² + nK ln σ_z² + ln|H|`） | θ̂ を固定して Z を積分した観測データ尤度 `ln ∫ p(X,Y,Z|θ̂) dZ` の Laplace 近似 | 厳密な周辺尤度、true BIC、一致性が保証された基準 |
| `C_cond_MC`（診断、B のみ） | `−2 (1/L)Σ_l ln p(X,Y|Z^(l),θ̂) + P_θ` | 原論文式(26)に**着想を得た** z 条件付き基準（評価点 = E-step サンプル平均） | 原論文 BIC |
| `C_cond_last`（診断） | 同上、Z = 最後の E-step サンプル | 同上 | 原論文 BIC |
| `C_cond_mode`（診断） | 同上、Z = joint mode Ẑ（Laplace が OK のときのみ） | 同上 | 原論文 BIC |

- **DERIVED**（監査で全 fit について再計算・確認）: `C_cond_MC = D_K + P_θ = C_Q − P_Z`、`C_cond_mode = fit_mode + P_θ`。
  つまり **`C_Q` と `C_cond_MC` の違いは `P_Z` だけ**、`C_Lap` と `C_cond_mode` の違いは `volume` だけである。
- **DERIVED / FACT**: σ_z² = 1 かつ Z の列を標準化しているので、`P_Z = nK(1 + ln 2π)` で**データによらず一定**
  （1 次元あたり 2.838·n。n=150 で 425.7。A1 の監査と B の分解表で確認）。
  したがって `C_Q` は、`ln n` オーダーの `P_θ` に加えて **n に比例する 1 次元あたりの罰則**を持つ。
  `C_Lap` の `volume` はデータに依存する（B の K_TRUE 直上の増分は 404〜487、直下の増分は 802〜947）。

---

## 3. 実験 A1（canonical clean n-sweep への C_Lap 追加）の確定結果

一次成果物: `expfam/results/k_selection/clap_clean_nsweep_20261006/full/`（`selection.csv`, `fit_criteria.csv`,
`increment_decomposition.csv`, `summary.json`, `runinfo.json`, `audit_report.json`）。
条件: Poisson-X / Bernoulli-Y、d=15、K_TRUE ∈ {1,3,5}、n ∈ {50,75,100,150}、候補 K 1..7、replicate 4/4/8、L=5、8 反復、2 start（平均）。
K_TRUE=1,3 は既存 seed の再現（448/448 fit が履歴とビット一致）、K_TRUE=5 は監査済み npz の checkpoint 8（448/448 が最大絶対差 0.0）。

**再照合（本作業）**: 一次 CSV から再集計し、`C_Q` 37/64、`C_Lap` 47/64（INCOMPLETE 4）、
独立監査 PASS（24161 checks, 0 findings）が報告値と一致した。派生表: `expfam/results/k_selection/k_criterion_comparison_analysis_20261104/A_*.csv`。

### 3.1 K_TRUE × n 別（exact / データセット数。括弧は選ばれた K）

| K_TRUE | n | `C_Q` | `C_Lap` | S1 held-out（参考） | S3（診断） |
|---|---|---|---|---|---|
| 1 | 50 | 4/4 | 2/4（1,1,inc,inc） | 4/4 | 0/4 |
| 1 | 75 | 4/4 | 3/4（1,inc,1,1） | 4/4 | 0/4 |
| 1 | 100 | 4/4 | 4/4 | 4/4 | 0/4 |
| 1 | 150 | 4/4 | 4/4 | 4/4 | 0/4 |
| 3 | 50 | 2/4（3,2,3,2） | 4/4 | 1/4 | 0/4 |
| 3 | 75 | 3/4（3,2,3,3） | 4/4 | 1/4 | 0/4 |
| 3 | 100 | 4/4 | 4/4 | 3/4 | 0/4 |
| 3 | 150 | 4/4 | 3/4（3,3,inc,3） | 4/4 | 0/4 |
| 5 | 50 | 0/8（2,1,1,2,2,2,2,2） | 0/8（2,2,4,4,4,3,3,4） | 2/8 | 3/8 |
| 5 | 75 | 0/8（2,4,4,3,4,4,2,3） | 3/8（4,4,5,4,4,5,4,5） | 0/8 | 0/8 |
| 5 | 100 | 1/8（4,3,4,4,4,3,2,5） | **8/8** | 4/8 | 0/8 |
| 5 | 150 | 7/8（5,5,5,4,5,5,5,5） | **8/8** | 8/8 | 0/8 |
| **計** | | **37/64**（under 27, over 0） | **47/64**（under 13, over 0, inc 4） | 39/64（under 23, over 2） | 3/64（over 61） |

### 3.2 n の増加による変化（OBSERVED）

- K_TRUE=5: `C_Q` は 0→0→1→7（/8）、`C_Lap` は 0→3→8→8（/8）。どちらも n とともに K_TRUE に近づき、
  `C_Lap` の方が**小さい n で** K_TRUE に到達した。n=150 では両者の差は 1 データセットだけである。
- K_TRUE=3: `C_Q` は n≤75 で 3 件過小選択し、`C_Lap` は過小選択 0（INCOMPLETE 1）。
- **単調な収束や一致性を示すものではない**（各セル 4〜8 データセット、n は 4 点）。

### 3.3 C_Lap の 4 件の INCOMPLETE の理由（FACT, `A_lap_incomplete.csv`）

| K_TRUE | n | rep | 失敗した fit (K, start) | grad∞ | H の最小固有値 | 解釈 |
|---|---|---|---|---|---|---|
| 1 | 50 | 3 | (7, 2) | 7.5e-3 | −0.26 | 過剰次元で mode が見つからない（H が不定） |
| 1 | 50 | 4 | (5, 2) | 9.0e-2 | −0.14 | 同上 |
| 1 | 75 | 2 | (6, 2) | 7.4e-2 | −0.07 | 同上 |
| 3 | 150 | 3 | (2, 2) | 1.10e-8 | +4.77 | 許容値 1e-8 をわずかに超えて 200 反復で停止（H は正定値） |

すべて joint-mode の Newton 法が 200 反復以内に `max|∇Φ| ≤ 1e-8` に達しなかった `NOT_STATIONARY` である。
規則どおり、そのデータセットの `C_Lap` 選択は INCOMPLETE とし、**補完・retry・許容値の緩和はしていない**。
3 件は K_TRUE=1 のデータで K ≥ 5 の過剰な次元を当てはめたときに起き、H が正定値でない（停止点は極大ではない）。

### 3.4 C_Q と C_Lap が異なる K を選んだケース（`A_disagreements.csv`）

- 28 データセットで異なる（うち 4 件は `C_Lap` が INCOMPLETE）。両方とも完了している 24 件では、**24/24 で `C_Lap` > `C_Q`**。
- 24 件の内訳: K_TRUE=3 で 3 件（`C_Q`=2 → `C_Lap`=3）、K_TRUE=5 で 21 件（`C_Lap` が K_TRUE に一致 11 件、なお過小 10 件）。
- `C_Q` が K_TRUE を選び `C_Lap` が外したケースは 0。

### 3.5 K_TRUE 境界での算術的分解（OBSERVED / DERIVED; 原因の証明ではない）

`increment_decomposition.csv`（2 start 平均）。K → K+1 への「当てはまりの改善」と罰則の増分の差を slack とし、slack > 0 なら K+1 が有利になる。

| K_TRUE, n, 段 | `C_Q`: fit gain / ΔP_Z / ΔP_θ / slack | `C_Lap`: fit gain / Δvolume / ΔP_θ / slack |
|---|---|---|
| 5, 75, 4→5 | 147.9 / 212.8 / 47.5 / **−112.4**（slack>0 は 0/8） | 211.7 / 172.0 / 47.5 / −7.7（3/8） |
| 5, 100, 4→5 | 284.8 / 283.8 / 50.7 / **−49.7**（1/8） | 378.0 / 273.1 / 50.7 / **+54.2**（8/8） |
| 5, 150, 4→5 | 628.5 / 425.7 / 55.1 / +147.7（7/8） | 768.5 / 474.5 / 55.1 / +238.9（8/8） |
| 5, 150, 5→6 | 124.0 / 425.7 / 50.1 / −351.8（0/8） | 288.9 / 315.0 / 50.1 / −76.2（0/8） |

- **OBSERVED**: K_TRUE=5、n=100 の境界では、`C_Q` の当てはまりの改善（284.8）は一定の `ΔP_Z`（283.8）とほぼ同じ大きさで、
  `ΔP_θ` を加えると負になった。`C_Lap` は当てはまりの改善が大きく（mode での評価）、`Δvolume` が `ΔP_Z` より小さかった。
- これは各項の算術的な寄与であり、「`P_Z` が過小選択の原因」とは書かない（KNOWN_ISSUES R）。

### 3.6 A1 から言えること・言えないこと

- 言える（OBSERVED）: この固定条件では、`C_Lap` の選択は `C_Q` と同じか、より大きい K だった。両者とも過大選択はなかった。
  `C_Q` が K_TRUE を外したケースの多くで `C_Lap` は K_TRUE を選んだ。`C_Lap` は 4/64 で数値的に未完了だった。
- 言えない: `C_Lap` の一般的な優位、K_TRUE=5 で n をさらに増やしたときの挙動（n=300 は seed 衝突のため実行していない）、
  過小選択の原因、Poisson-X 以外への一般化。

---

## 4. 実験 B の実施状況

| 段階 | 結果 | 根拠 |
|---|---|---|
| 1. 実装とプロトコルの整合 | runner の frozen protocol が Issue #128 本文・計画書 §4 と一致（n, d, K_TRUE, 候補 K, family, L, 反復, start, R, seed 規則, 失敗規則, pilot gate） | `run_paper_condition_k_criteria.py` |
| 2. 既存テスト | `test_paper_condition_k_criteria.py` + `test_clap_clean_nsweep.py`: **24 passed** | pytest |
| 3. preflight | protocol hash `e095ea9b…917f`、500 fits、model / data seed 重複なし、tree clean、git head `0310230` | `--preflight` |
| 4. pilot（K_TRUE=3, rep 1, 20 fits） | FAILED 0、Laplace OK 20/20、所要 347.8 秒 | `pilot/runinfo.json` |
| 5. pilot gate | **PASS**（FAILED なし、`C_Q`/`C_cond_MC`/`C_cond_last` 有限、`C_Q` 閉包、`C_Lap` 20/20 OK）。独立監査 PASS（495 checks, 0 findings） | `pilot/audit_report.json` |
| 6. full（500 fits） | FAILED 0、Laplace OK 492 / NOT_STATIONARY 8、所要 8391 秒（約 2.3 時間）、git head `1ae0d6c` | `full/runinfo.json` |
| 7. 独立監査 | **PASS**（11305 checks, 0 findings, BLOCKER 0, HIGH 0） | `full/audit_report.json` |
| 8. 分析・レポート | 本文書 | — |

- retry / replacement / seed rescue / 許容値の緩和 / resume はすべて 0（runinfo と監査で確認）。
- 新規作成した独立 auditor（`tools/research_audit/audit_paper_condition_k_criteria.py`）は計画書 §6「B 用も同様」に従うもので、
  runner・model・generator のコードを import せず、protocol hash、seed、fit 数、`C_Q`/`C_Lap` 閉包、Laplace 生成分からの `C_Lap` 再計算、
  `C_cond` の恒等式、選択・label・margin・tie・start 別選択、増分分解、summary の数、generator ファイルの sha256 不変、状態ファイルの sha256 を再計算する。

### 4.1 条件と原論文との相違（必ず併記）

| 項目 | 原論文 Experiment 2（Table I） | historical Python 再現 `experiment_paper_2.py` | **本実験 B** |
|---|---|---|---|
| generator | 論文の設定 | `set_true_params` | `set_true_params`（変更なし、var_f=5, uniq=0.1, w_scale=3） |
| 推定器 | 論文の EM | `LatentStructuralModel` | `DualExpFamLSMConsistent`（lineage E） |
| L / 反復 / 試行 | 10 / 10 / 10 | 10 / 30 / 3（最良 BIC） | **5 / 8 / 2 start（平均）** |
| 評価する状態 | 不明 | 反復中の最良 | **最終状態** |
| データセット数 / K_TRUE | 1（と読める） | 1（seed 1980+100K_TRUE） | **5**（rep1 は historical と同じ data seed） |
| 基準 | 式(26)（Z の評価点は不明） | 独自実装の BIC | `C_Q`, `C_Lap`, `C_cond_*`（いずれも原論文 BIC ではない） |
| Y の扱い | 全ペア | 全ペア | 全ペア（held-out なし） |

**FACT（generator の性質）**: `set_true_params` は F の各行を `‖F_i‖ = √(1−0.1)` に正規化するので、
X の各列の分散のうち 90% が Z で説明される（独自分散 0.1）。その後 X を列ごとに標準化する。
**A1 の Poisson-X clean generator よりかなり強い属性信号の条件である**（var_f は正規化後は効かない）。

---

## 5. 実験 B の結果

一次成果物: `expfam/results/k_selection/paper_condition_k_criteria_20261006/full/`。派生表: `expfam/results/k_selection/k_criterion_comparison_analysis_20261104/B_*.csv`。

### 5.1 K_TRUE 別の選択（exact / 5。括弧は選ばれた K）

| K_TRUE | `C_Q` | `C_Lap` | `C_cond_MC` | `C_cond_last` | `C_cond_mode` |
|---|---|---|---|---|---|
| 1 | **5/5**（1,1,1,1,1） | 4/5（1,inc,1,1,1） | 0/5（10,9,9,10,10） | 0/5（10,10,10,9,9） | 0/5（10,inc,10,10,10） |
| 3 | **5/5** | 2/5（3,3,inc,inc,inc） | 0/5（9,10,10,10,10） | 0/5（10×5） | 0/5（10,10,inc,inc,inc） |
| 5 | **5/5** | **5/5** | 0/5（10×5） | 0/5（10×5） | 0/5（10×5） |
| 7 | **5/5** | 4/5（7,7,7,inc,7） | 0/5（10×5） | 0/5（10,9,10,10,10） | 0/5（10,10,10,inc,10） |
| 9 | **5/5** | 4/5（9,9,9,9,inc） | 0/5（10×5） | 0/5（10×5） | 0/5（10,10,10,10,inc） |
| **計** | **25/25**（under 0, over 0） | **19/25**（under 0, over 0, **inc 6**） | 0/25（over 25） | 0/25（over 25） | 0/25（over 19, inc 6） |

- **OBSERVED**: `C_Q` と `C_Lap` の両方が完了した 19 データセットで、両者の選択は 19/19 一致した（すべて K_TRUE）。
- start ごとの選択: `C_Q` は 50/50 の start で K_TRUE、`C_Lap` は完了分 38/38 の start で K_TRUE。2 start の選択は全データセットで一致した。
- 選択の margin（最良と次点の差）: `C_Q` は最小 90.6（K_TRUE=7）・中央値 279〜385、`C_Lap` は最小 49.2・中央値 82〜264。
  **`C_Q` の方が margin が大きかった**。
- `C_cond_*` は 75 の完了選択すべてで候補の上端（K=9 か 10）を選んだ。K_TRUE=9 でも K=10 を選んでいる。

### 5.2 C_Lap の 6 件の INCOMPLETE（`B_lap_nonok.csv`）

| K_TRUE | rep | (K, start) | grad∞ | H の最小固有値 | 種類 |
|---|---|---|---|---|---|
| 1 | 2 | (1, 1) | 2.0e-8 | +141.7 | 許容値のわずか上で停止（H 正定値） |
| 1 | 2 | (9, 1) | 1.6e-8 | +0.03 | 同上 |
| 1 | 2 | (10, 1) | 5.3e-2 | −0.026 | 過剰次元で H が不定 |
| 3 | 3 | (10, 1) | 2.2e-1 | −0.051 | 過剰次元で H が不定 |
| 3 | 4 | (3, 1) | 5.3e-8 | +35.5 | 許容値のわずか上で停止（**K = K_TRUE の fit**） |
| 3 | 5 | (10, 2) | 2.2e-1 | −0.076 | 過剰次元で H が不定 |
| 7 | 4 | (8, 1) | 3.1e-2 | −0.107 | 過剰次元で H が不定 |
| 9 | 5 | (10, 1) | 1.04 | −0.042 | 過剰次元で H が不定 |

（8 fit / 6 データセット。K_TRUE=1 rep 2 は 3 fit が該当。）

- **OBSERVED**: 非 OK の 8 fit は 2 種類に分かれた。(a) H が不定で、grad が大きいまま停止したもの（5 fit、すべて K ≥ 8 の過剰次元）。
  (b) H は正定値で、grad が許容値 1e-8 の 1.6〜5.3 倍で 200 反復に達したもの（3 fit。うち 1 つは K=1、1 つは K=K_TRUE）。
- (b) は Phase 9N/9O/9R で記録されている「許容値のわずか上で 200 反復止まる」失敗と同じ型である（KNOWN_ISSUES Q）。
- OK だった fit でも、H の最小固有値は K とともに小さくなった（K=1 で 2.75、K=9 で 0.0037）。過剰な次元では事後分布の山が平たい方向がある。
- **INCOMPLETE は INCOMPLETE として扱い、他の start の値や近い値で補完していない。**

### 5.3 K_TRUE 境界での算術的分解（`B_boundary_terms.csv`, 図 `B_boundary_slack.png`）

| 段 | `C_Q` の fit gain（平均） | ΔP_Z | ΔP_θ | `C_Q` slack（平均 / 最も K_TRUE に不利な値） | `C_Lap` の fit gain / Δvolume / slack（平均） |
|---|---|---|---|---|---|
| 2→3（K_TRUE=3） | 4567.6 | 425.7 | 65.1 | +4076.8 / +1850.7 | 5185.1 / 917.7 / +4202.3 |
| 4→5（K_TRUE=5） | 3456.3 | 425.7 | 55.1 | +2975.5 / +1084.9 | 3643.0 / 919.0 / +2668.9 |
| 6→7（K_TRUE=7） | 2113.4 | 425.7 | 45.1 | +1642.6 / +804.4 | 2271.3 / 802.3 / +1423.8 |
| 8→9（K_TRUE=9） | 2259.3 | 425.7 | 35.1 | +1798.6 / +1617.7 | 2478.8 / 947.1 / +1496.7 |
| 1→2（K_TRUE=1） | 103.1 | 425.7 | 70.1 | −392.8 / −455.0 | 242.1 / 422.7 / −250.7 |
| 3→4（K_TRUE=3） | 236.6 | 425.7 | 60.1 | −249.2 / −337.9 | 387.7 / 487.4 / −159.8 |
| 5→6（K_TRUE=5） | 198.1 | 425.7 | 50.1 | −277.7 / −320.7 | 333.7 / 437.0 / −153.5 |
| 7→8（K_TRUE=7） | 203.0 | 425.7 | 40.1 | −262.7 / −374.5 | 355.7 / 404.3 / −88.7 |
| 9→10（K_TRUE=9） | 160.7 | 425.7 | 30.1 | −295.0 / −325.1 | 305.3 / 407.2 / −132.0 |

- **OBSERVED**: K_TRUE 直下の段では当てはまりの改善が 800〜5800 で、どの罰則よりはるかに大きい。
  境界が非常に鋭いので、`C_Q` の一定の `P_Z` も `C_Lap` の volume も選択を変えなかった。
- **OBSERVED / DERIVED**: K_TRUE 直上の段では、`C_Q` の当てはまりの改善（160〜237）は `ΔP_θ`（30〜70）より大きく、`ΔP_θ + ΔP_Z` より小さい。
  `C_cond_MC = C_Q − P_Z` なので、**`P_Z` を除いた `C_cond_MC` は K_TRUE より上でも減り続け、上端を選ぶ**。
  これが `C_cond_*` の 25/25 過大選択の算術的な内訳である。
- K_TRUE 直上の slack は `C_Lap` の方が 0 に近い（平均 −89〜−251 に対し `C_Q` は −249〜−393）。
  **この条件で `C_Lap` は `C_Q` より過大選択側の余裕が小さい**（ただし過大選択は 0 件）。

---

## 6. 図

| 図 | パス | 内容 |
|---|---|---|
| B 選択 K vs K_TRUE | `expfam/figures/k_criterion_comparison_20261104/B_selected_vs_true.png` | 基準ごとの選択 K（円の面積 = データセット数、下の数字 = INCOMPLETE） |
| B 選択行列 | `expfam/figures/k_criterion_comparison_20261104/B_selection_matrix.png` | データセット × 基準の選択 K |
| B 基準曲線 | `expfam/figures/k_criterion_comparison_20261104/B_curves_by_criterion.png` | 候補 K に対する各基準（最小値からの差、symlog）。**INCOMPLETE データセットの曲線は有限の点だけで描いているので、その最小点は選択結果ではない**（K_TRUE=1, 3 の `C_Lap` 行に見える K_TRUE 以外の最小はこれ） |
| B 境界の slack | `expfam/figures/k_criterion_comparison_20261104/B_boundary_slack.png` | §5.3 の slack（左: K_TRUE−1→K_TRUE、右: K_TRUE→K_TRUE+1） |
| A1 選択 K vs n | `expfam/figures/k_criterion_comparison_20261006/A1_selected_k_vs_n.png` | 既存（2026-10-06 生成、再生成していない） |
| A1 exact 数 vs n | `expfam/figures/k_criterion_comparison_20261006/A1_exact_counts_vs_n.png` | 既存 |
| A1 境界の slack | `expfam/figures/k_criterion_comparison_20261006/A1_boundary_slack.png` | 既存 |
| A1 不一致ヒートマップ | `expfam/figures/k_criterion_comparison_20261006/A1_disagreement_heatmap.png` | 既存 |

---

## 7. 研究上の問いへの回答

### RQ1: C_Lap は原論文型のデータでも C_Q より真の K を選びやすいか？

**この固定条件では、そうとは言えない。** `C_Q` 25/25、`C_Lap` 19/19（完了分）で、両者が完了した全データセットで選択は一致した。
`C_Lap` には 6/25 の数値的未完了があり、margin も `C_Q` より小さかった。
`C_Lap` が `C_Q` より K_TRUE に近い K を選んだのは A1（弱い Poisson-X 信号、小さい n）だけである（24/24 の不一致で `C_Lap` の方が大きい K）。
**INTERPRETATION**: 基準の違いは、K_TRUE 直下の当てはまりの改善が罰則と同程度になる弱い信号の条件で現れ、強い信号の条件では消える。
この解釈は 2 つの条件（generator・X family・n がすべて異なる）の対比にもとづくもので、**どの要因が効いたかは切り分けていない**。

### RQ2: 真の K が大きくなったとき、どのような誤選択が起きるか？

- B（強い信号）: K_TRUE=9 まで `C_Q`/`C_Lap` の誤選択は 0。数値的未完了は K_TRUE によらず起きた（K_TRUE=1, 3, 7, 9）。
- A1（弱い信号）: 誤選択は**過小選択だけ**で、K_TRUE=5、n ≤ 100 に集中した。`C_Q` の過小は 1〜4 次元（n=50 で K=1〜2 を選ぶこともあった）。
  `C_Lap` の過小は 1〜3 次元。K_TRUE が大きいほど過小選択が増えた（K_TRUE=1: 0、3: `C_Q` 3、5: `C_Q` 24 / `C_Lap` 13）。
- z 条件付き基準（B）: K_TRUE にかかわらず候補の上端への**過大選択**。
- **INTERPRETATION**: K_TRUE が大きいほど 1 次元あたりの信号が小さくなり、`C_Q` の一定の 1 次元あたり罰則 `2.838n` を超えにくくなる、という読み方は分解と整合する。
  ただし「信号の希釈が原因」とは主張しない（KNOWN_ISSUES R、Phase 9Y は算術の分解だけ）。

### RQ3: 誤選択は推定不足、ラプラス近似、罰則のどこに関係していそうか？

| 要因 | 証拠から支持される範囲 | 未検証（HYPOTHESIS） |
|---|---|---|
| **罰則（`P_Z`）** | **DERIVED**: `C_Q` は `P_θ` に加えて、n に比例し一定の `P_Z = 2.838nK` を持つ。**OBSERVED**: A1 の K_TRUE=5、n=75〜100 では、境界の当てはまりの改善が `ΔP_Z` と同程度で、`C_Q` の slack が負になった。B では `P_Z` を除くと（`C_cond_MC`）25/25 過大選択した。つまり `P_Z` は**過大選択を防ぐ方向と、弱い信号での過小選択の方向の両方に働いている**ことが算術上確認できる | `P_Z` が過小選択の「原因」であること。`P_Z` を別の量に置き換えれば K_TRUE を回復できること |
| **ラプラス近似** | **OBSERVED**: 数値的な未完了（A1 4/896 fit、B 8/500 fit）。H が不定になる fit は過剰な次元（K ≥ 5）に偏り、OK の fit でも H の最小固有値が K とともに 0 に近づく。近似の精度そのものは評価できていない（Phase 9J の IS は ESS が退化） | 過剰次元で mode の周りの Gaussian 近似が悪く、`C_Lap` の volume が過大／過小になっていること。これが B の K_TRUE 直上の slack が 0 に近いことに関係していること |
| **推定不足（8 反復 MCEM）** | **OBSERVED**: B では 2 start の選択が全データセットで一致し、margin も大きいので、選択は start に頑健だった。MCEM 8 反復の θ̂ が `ℓ_Lap` の停留点でないことは既知（Phase 9M） | A1 の過小選択が、反復を増やすと減ること（本 Issue では検証していない）。θ̂ を `ℓ_Lap` で再最適化したときに `C_Lap` の選択が変わること |

**原因の断定はしない。** 現時点で最も強く言えるのは、「`C_Q` と `C_cond_*` と `C_Lap` の違いは、それぞれ `P_Z` と volume の扱いの違いに算術的に還元でき、
その違いが選択を変えるのは境界の当てはまりの改善が数百程度の条件である」ということまでである。

### RQ4: 現在の結果から、K 選択手法として何を提案できそうか？

**提案候補（未検証。完成した手法ではない）:**
1. **`C_Q` と `C_Lap` を併記し、両者が一致するかを不確かさの指標とする運用。** A1 では、両者が一致した 36 データセットのうち 33 で K_TRUE、
   不一致は全て `C_Q` < `C_Lap` だった。B では両者は完了分すべてで一致した。「一致すれば比較的安心、不一致なら `C_Lap` 側が上限の目安」という
   読み方は、この 2 条件では矛盾しない。ただし一般の判断規則としては未検証である。
2. **`C_Lap` を使う場合は Laplace の数値診断（停留性・H の正定値性・最小固有値）を必ず報告し、未完了は未完了として扱う。**
3. **z 条件付き尤度に `p_K ln n` だけを加えた基準は、本研究の実装・評価点では使わない**（B で 25/25 過大選択）。
   ただしこれは原論文の評価手続きの否定ではない（評価点が特定できないため）。

### RQ5: これ以上の K 選択研究を修論の主貢献にする価値があるか？

**推奨（DECISION は Human Gate）: 主貢献にはしない。補助的な章（「どの条件で選べ、どの条件で選べないか」の characterization）として使う。**

理由:
- 原論文型の強い信号では、既存の `C_Q` でも 25/25 で K_TRUE を選んだ。新しい基準が必要になるのは弱い信号の条件だけで、その範囲はまだ一部しか調べていない。
- `C_Lap` には、理論的な正当化（特異モデル・RLCT は未知、KNOWN_ISSUES O）、近似精度の評価、数値的な完全性のどれもまだない。
  主貢献として防御するには、これらを修論期限までに揃える必要があり、規模が大きい。
- Issue #138 の比較で Candidate C（family + K selection）は「evidence が最も弱い」とされている。今回の結果は、その評価を変えるほどの強さはない。
  - Candidate A（misspecification-first）: K 選択は「K を固定して比較するための前提」として、本結果の「強い信号では `C_Q` で十分」が背景に使える。
  - Candidate B（family-selection-first）: K 選択は family 選択と同じ fit 上で行う補助手順として位置づけられる。今回の「同一 fit で複数基準」の枠組みはそのまま使える。
  - Candidate C: 主張を「弱い信号での K 選択の characterization」に限定するなら成り立つが、主題としては最も弱い。

---

## 8. 近似・最適化・識別可能性の限界

1. **近似**: `C_Lap` は θ̂ を固定した Z の積分の Laplace 近似で、θ の積分は `p_K ln N` という BIC 型の項で代用している。
   N = n（ノード数）は作業上の約束で、有効標本数は定義されていない（KNOWN_ISSUES O-3）。近似誤差は評価できていない。
2. **最適化**: MCEM は 8 反復・L=5 で、θ̂ は収束していない可能性がある（Phase 9M）。B では最終状態を評価した。
   joint mode の Newton 法は 200 反復・許容値 1e-8 で、未達の fit は記録どおり未完了とした。
3. **識別可能性**: 人工データの K_TRUE が母集団レベルで識別される潜在次元 K* と一致することは自明ではない（KNOWN_ISSUES P）。
   B の K=9, 10 は d=15 で因子分析の Ledermann 限界（(d−K)² ≥ d+K）に近い（K=10 で等号）。
4. **標本**: B は K_TRUE ごとに 5 データセット、A1 はセルごとに 4〜8。数は率ではない。
5. **条件の交絡**: A1 と B は generator・X の分布族・n・候補 K の範囲がすべて異なる。RQ1 の解釈（信号の強さ）は、この対比からの推測である。
6. **原論文との関係**: B は原論文の再現ではない（§4.1）。`C_cond_*` の過大選択は、原論文の BIC が過大選択することを意味しない。

---

## 9. 次に何をすれば真の K をより適切に選べそうか（優先順位付きの改善候補）

いずれも**新しい実験 / 新しい scope**であり、実行には Human Gate での承認が必要。結果 A/B のどちらでも研究判断が変わるかを併記する。

| 優先 | 候補 | 何が分かるか | 結果で判断が変わるか | 規模 |
|---|---|---|---|---|
| **1** | **信号強度の橋渡し sweep**: 原論文型 generator のまま、独自分散 `uniq`（例 0.1→0.5→0.8）と n を動かし、同じ同一 fit パネルで `C_Q`/`C_Lap` を比較 | A1 と B の間のどこで `C_Q` が過小選択を始め、`C_Lap` との差が現れるか。RQ1 の解釈（信号の強さ）を、generator を固定したまま検証できる | **変わる**: 差が出る範囲が広ければ `C_Lap` を補助章の中心に、狭ければ「`C_Q` で十分」を結論にする | 中（既存 runner を条件だけ変えて使える） |
| 2 | **推定不足の切り分け**: A1 の過小選択セル（K_TRUE=5, n=75/100）だけで EM 反復数を増やして同じ基準を再計算（事前に固定した反復数・seed） | 過小選択が推定不足によるものか | 変わる: 反復で解消するなら基準でなく推定の問題 | 小〜中 |
| 3 | **Laplace の数値安定化を事前登録した別プロトコル**（例: 反復上限の拡大、相対許容値。既存結果は救済しない） | INCOMPLETE が数値の問題か、mode が存在しないためか | 一部変わる: INCOMPLETE が消えれば `C_Lap` の実用性の評価が上がる | 小 |
| 4 | **Laplace 近似の精度評価**（小さい n・小さい K で、より正確な周辺尤度の推定と比較） | `C_Lap` の値そのものの信頼性 | 変わる: 近似が悪ければ `C_Lap` は「経験的な基準」にとどめる | 中〜大 |
| 5 | **実データでの基準の挙動の characterization**（held-out 予測・start 間安定性。真の K はない） | 実データで `C_Q`/`C_Lap` がどれだけ食い違うか | 修論の応用章の内容が変わる | 中 |
| 6 | 理論（特異モデルの罰則、RLCT） | 罰則の正当化 | 長期 | 大（修論期限内は困難） |

**推奨順**: 1 → 2 →（必要なら）3。1 と 2 は「基準の違い」「推定不足」を分けるもので、RQ3 の未検証部分に直接答える。
R=10 への拡張（seed 固定済みの replicate 6〜10）は、B の結果がすでに天井（25/25）にあるので、**優先度は低い**。

---

## 10. 実行したコマンド・commit・成果物・監査

### コマンド（worktree `D:\tento\kennkyu-128-kcrit`）

```
python -m pytest tools/research_audit/test_paper_condition_k_criteria.py tools/research_audit/test_clap_clean_nsweep.py -q   # 24 passed
python tools/research_audit/run_paper_condition_k_criteria.py --preflight
python tools/research_audit/run_paper_condition_k_criteria.py --pilot --allow-em          # 20 fits, 347.8 s
python tools/research_audit/audit_paper_condition_k_criteria.py --stage pilot --write-report   # PASS
python tools/research_audit/run_paper_condition_k_criteria.py --full --allow-em --confirm-b   # 500 fits, 8391.2 s
python tools/research_audit/audit_paper_condition_k_criteria.py --stage full --write-report    # PASS
python tools/research_audit/analyze_k_criterion_comparison.py --a-dir expfam/results/k_selection/clap_clean_nsweep_20261006/full --b-dir expfam/results/k_selection/paper_condition_k_criteria_20261006/full --out expfam/results/k_selection/k_criterion_comparison_analysis_20261104 --fig-dir expfam/figures/k_criterion_comparison_20261104
python tools/research_audit/plot_k_criterion_comparison.py --b-dir expfam/results/k_selection/paper_condition_k_criteria_20261006/full --out expfam/figures/k_criterion_comparison_20261104
```

### commit（branch `experiment/128-k-criterion-comparison`）

| commit | 内容 |
|---|---|
| `1ae0d6c` | B pilot 結果、B 独立 auditor、派生表スクリプト |
| `3209b6c` | B full 結果（500 fits）、監査レポート、STATE_ARCHIVE |
| `9704a06` | 派生表・B 図 |
| （本レポートの commit） | 本レポート、EXPERIMENT_REGISTRY 追記 |

### 成果物

| 種類 | パス |
|---|---|
| B pilot | `expfam/results/k_selection/paper_condition_k_criteria_20261006/pilot/` |
| B full | `expfam/results/k_selection/paper_condition_k_criteria_20261006/full/`（`fit_criteria.csv`, `selection.csv`, `criterion_curves.csv`, `increment_decomposition.csv`, `dataset_provenance.csv`, `summary.json`, `protocol.json`, `runinfo.json`, `audit_report.json`） |
| B 状態 | `D:\tento\kennkyu_artifacts\paper_condition_k_criteria_20261006\{pilot,full}\states.npz`（sha256 は `STATE_ARCHIVE.md`） |
| 派生表 | `expfam/results/k_selection/k_criterion_comparison_analysis_20261104/` |
| 図 | `expfam/figures/k_criterion_comparison_20261104/` |
| A1（既存、変更なし） | `expfam/results/k_selection/clap_clean_nsweep_20261006/`, `expfam/figures/k_criterion_comparison_20261006/` |

### 監査

| 対象 | verdict | checks | findings |
|---|---|---|---|
| A1 full（既存） | PASS | 24161 | 0 |
| B pilot | PASS | 495 | 0 |
| B full | PASS | 11305 | 0 |

既存の CSV・JSON・NPZ は上書きしていない（A1 ディレクトリへの書き込みなし。B と派生表は新規ディレクトリで、スクリプトは既存ファイルがあれば停止する）。
