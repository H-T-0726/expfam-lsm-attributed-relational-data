# Phase 8a: K_TRUE robustness harness — implementation plan

**日付:** 2026-09-01
**Issue:** #47（design）→ 次の implementation Issue（未作成）
**対になる design:** `reports/k_selection_theory/k_true_robustness_design_20260901.md`
**種別:** PLAN ONLY — 本 Issue では**実装しない**（コード 1 行も書かない）
**改訂:** 2026-09-01 — Codex review findings HIGH-03 / LOW-01 / H4 HIGH / MEDIUM-01 / F-01 を反映、
および **Human Gate decision（H1–H4）を frozen config として反映**

**Human Gate frozen:** 2026-09-01
**Decision source:** GitHub Issue #47 — Human Gate Decision comment（2026-09-01, by H-T-0726）
**Decision type:** **HUMAN GATE DECISION**（AI recommendation ではない）

> **前提条件は充足済み。** design report §15 の decision gate は
> `A: IMPLEMENT_K_TRUE_ROBUSTNESS_HARNESS_NEXT` へ更新され、
> human gate H1–H4 は 2026-09-01 に確定した（design §16）。
>
> **ただし `A` は full experiment の実行許可ではない。**
> 次に許可されるのは **implementation と zero-EM validation のみ**であり、
> **336 fits の full run はまだ許可されていない**（§7・design §15.1）。
> 本 plan の時点では**まだ実装しない**。

---

## 0. Stage 分離（Phase 7e と同じ 4 段階）

| Stage | Issue | 内容 | EM fits | 出力 |
|---|---|---|---:|---|
| **S0** | #47（本 Issue） | design freeze・comparability 監査 | **0** | 本 plan + design report |
| **S0.5** | #47 | **Human Gate H1–H4** — **完了（2026-09-01）** | 0 | Issue #47 Human Gate Decision comment |
| **S1** | 次 Issue | harness 実装 + static / adversarial tests | **0** | 新規 module + tests |
| **S2** | 同 Issue（別 gate） | configuration gate + canary + smoke | 少数（smoke のみ） | smoke artifact |
| **S3** | 別 Issue | full run + 独立 audit | **336**（A: 168 / B: 168） | full artifact + report |

各 Stage の完了時に人間へ返す。**S1 完了後に自動で S2 へ進まない。S2 完了後に自動で S3 へ進まない。**

---

## 1. 変更を許可する exact files（次の implementation Issue の scope）

### 1.1 新規作成（既存 Phase 7e コードは**触らない**）

| # | path | 種別 | 必要性 | 役割 |
|---|---|---|---|---|
| 1 | `tools/research_audit/run_k_true_robustness_sweep.py` | 新規 | **必須** | Phase 8a harness 本体 |
| 2 | `tools/research_audit/audit_k_true_robustness_sweep.py` | 新規 | **必須** | artifact のみを読む独立 audit |
| 3 | `tools/research_audit/test_k_true_robustness_sweep.py` | 新規 | **必須** | static / adversarial tests |
| 4 | `tools/research_audit/run_k_true_robustness_full.ps1` | 新規 | **optional — §9 で正当化された場合のみ** | full run 起動 wrapper（stdout capture を committed 化する場合） |
| 5 | `expfam/results/k_selection/k_true_robustness_<estimand>_<YYYYMMDD>/` | 新規（S3 のみ） | 必須 | full artifact 出力先 |
| 6 | `reports/k_selection_theory/k_true_robustness_report_<YYYYMMDD>.md` | 新規（S3 のみ） | 必須 | 自動生成 report |

**#4 について（LOW-01 の解消）:** wrapper は **required ではなく optional** である。
§9 のとおり、採用する場合のみ上記の exact path を allowed list に含める。
採用しない場合は作成しない。**「required wrapper」という表現は本 plan では使わない。**
path を変更する必要が生じた場合は、implementation Issue の scope に明示して承認を取る。

### 1.2 変更を**禁止**する files

| path | 理由 |
|---|---|
| `tools/research_audit/run_heldout_k_selection_pilot.py` | **RUN_CODE_SHA `b9311e64…` の scientific execution code。** 変更すると anchor の code provenance（design §3.3 の diff 空）が失われる |
| `tools/research_audit/audit_heldout_full_pilot.py` | Phase 7e artifact 専用 audit（42 行固定） |
| `tools/research_audit/test_heldout_k_selection_pilot.py` | 120 tests が 42 行 / 3 replicate を固定 assertion している |
| `expfam/src/data_generator_expfam.py` | **generator semantics。** Option B は既存の `w_true` 引数を呼び出し側から渡すだけで実現でき、generator の改変を要しない（design §8.3） |
| `expfam/src/experimental/em_runner.py` / `model_dual_expfam_consistent.py` | model semantics。Human Gate |
| `expfam/src/experimental/eval_utils.py` | `make_pair_split` / `calc_bic_exp` |
| `expfam/results/k_selection/heldout_full_pilot_20260824/**` | 凍結成果物。**読み取り専用** |
| `reports/k_selection_theory/heldout_k_selection_full_pilot_report_20260824.md` | historical frozen record |
| canonical docs（`RESEARCH_MASTER.md` / `KNOWN_ISSUES.md` / `EXPERIMENT_REGISTRY.md` / `CLAUDE.md`） | Human Gate（`EXPERIMENT_REGISTRY.md` への追記は S3 完了後に別途承認） |

### 1.3 なぜ既存 harness を改造せず新規 module にするか

1. `run_heldout_k_selection_pilot.py` は `run_code_sha = b9311e64…` の実行コードそのものであり、
   現 HEAD で diff 空であることが anchor 再利用の一次根拠（design §3.3）。改造するとこの根拠が失われる。
2. `EXPECTED_FULL_FITS = 42` / `FULL_REPLICATES = (1,2,3)` / `validate_full_manifest` の
   「42 行厳密」制約と 120 tests は、`K_TRUE` 次元の追加と構造的に非互換。
3. Phase 8a の artifact schema は `K_TRUE` 列を持ち、Phase 7e schema と非互換（§4）。

新 module は Phase 7e module から**再利用可能な純粋関数を import する**（改変せず利用する）:

```
from run_heldout_k_selection_pilot import (
    stable_array_hash, stable_config_hash,
    validate_pair_masks, heldout_bernoulli_mean_log_score,
    heldout_raw_eta_pairs, score_heldout_bernoulli,
    make_training_y_values, make_score_only_target,
    prepare_training_data, authorize_canary_preflight,
    build_fit_payload, FitCallBoundary, AuthorizedEMFitAdapter,
    FrozenFitConfig, FrozenScoreConfig, score_config_hash,
    select_k_from_two_starts, require_no_blocking_failures,
    run_em_with_initialization_capture, HarnessStop,
)
```

**import は改変ではない。** leakage 境界・score・selector・fit boundary を
Phase 7e と**同一実装**のまま使うことが、protocol 同一性の最強の保証になる。

---

## 2. 新規 harness の関数設計

### 2.1 module 定数

```python
PHASE = "8a"
FAMILY_X, FAMILY_Y = "poisson", "bernoulli"
N_NODES, N_FEATURES = 75, 15
TEST_RATIO, L_SAMPLES, NUM_ITER = 0.20, 5, 8
NUMERICS_MODE = "consistent"
K_CANDIDATES = tuple(range(1, 8))
START_LABELS = (1, 2)
REPLICATES = (1, 2, 3)
NEW_K_TRUE = (1, 2, 4, 5)          # anchor K_TRUE=3 は含めない
ANCHOR_K_TRUE = 3
TIE_TOLERANCE = np.float64(1e-12)

W0_TRUE = -1.0
W_REF, K_REF = 1.5, 3

# --- HUMAN GATE FROZEN (2026-09-01, GitHub Issue #47) --------------------
# placeholder ではない。これが current executable config である。
ESTIMANDS       = "AB"        # H1: Option A と Option B を両方 pre-register
PRIMARY_ESTIMAND     = "A"    # H3-a
SENSITIVITY_ESTIMAND = "B"    # H3-a
HIERARCHY       = "H3_A"      # H3: A primary + B pre-registered sensitivity
RANDOM_DESIGN   = "CRN"       # H2: data_seed / model_seed のみを支配
MASK_DESIGN     = "S_C"       # H4: Phase 7e anchor-aligned pair-index mask
SPLIT_VARIANT   = MASK_DESIGN # 後方互換の別名（manifest 上の field 名は mask_design）

DATA_SEED_BASE, MODEL_SEED_BASE = 51000, 530000
SPLIT_SEED_BASE = 52000        # S_A / S_B 用（NOT SELECTED、履歴のため保持）
ANCHOR_SPLIT_SEED_BASE = 42000 # S_C 用（Phase 7e anchor の split seed を意図的に再利用）

FITS_PER_ESTIMAND = len(NEW_K_TRUE) * len(REPLICATES) * len(K_CANDIDATES) * len(START_LABELS)  # 168
EXPECTED_NEW_FITS = FITS_PER_ESTIMAND * 2   # 336 (A + B)
```

### 2.2 `w_true` 解決関数（Option A / B の唯一の分岐点）

```python
def resolve_w_true(estimand: str, k_true: int) -> float:
    if estimand == "A":
        return W_REF                              # 1.5 固定
    if estimand == "B":
        return W_REF * math.sqrt(K_REF / k_true)  # w_K^2 * K = W_REF^2 * K_REF
    raise HarnessStop("estimand is not frozen")
```

**不変条件（static test で強制）:** `resolve_w_true("A", 3) == resolve_w_true("B", 3) == 1.5`。
これが anchor 互換性と、A/B が anchor を共有できること（design §11.2）のコード上の保証になる。

### 2.2b estimand role 解決関数（H3 の実装、design §16）

A/B の hierarchy を **実行前に一意に固定する**ための関数。
role は manifest / runinfo / report template に保存され、
**full run 後に変更できない**（§7 の freeze・A17 の回帰テスト）。

```python
def resolve_role(estimand: str) -> str:
    if ESTIMANDS != "AB":
        return "single"                         # H1 = A only / B only
    if HIERARCHY == "H3_A":                     # A primary + B pre-registered sensitivity
        return {"A": "primary", "B": "sensitivity"}[estimand]
    if HIERARCHY == "H3_B":                     # A/B co-equal separate estimands
        return {"A": "coequal_A", "B": "coequal_B"}[estimand]
    raise HarnessStop("A/B hierarchy is not frozen")
```

| H3 | `role` (A) | `role` (B) | 意味 |
|---|---|---|---|
| **H3-a** ← **SELECTED（Human Gate 2026-09-01）** | `primary` | `sensitivity` | A が primary estimand、B が **pre-registered** sensitivity estimand |
| **H3-b** *(NOT SELECTED)* | `coequal_A` | `coequal_B` | hierarchy なし（履歴のため分岐を残す） |

**current config は `HIERARCHY = "H3_A"` であるため、実行時の role は
`A -> primary` / `B -> sensitivity` に一意に確定する。**

**exact field name は実装時に調整してよいが、上記の scientific semantics は保持する。**
とくに H3-b の値は「どちらかが上位である」と読めてはならない。

### 2.3 seed 関数

```python
def expected_data_seed(k_true, replicate)  -> DATA_SEED_BASE  + 100 * k_true + replicate

def expected_split_seed(k_true, replicate):
    # S_A: K_TRUE ごとに独立な mask
    if SPLIT_VARIANT == "S_A":
        return SPLIT_SEED_BASE + 100 * k_true + replicate
    # S_B: 新規 K_TRUE {1,2,4,5} の間だけ mask を共有。
    #      Phase 7e K_TRUE=3 anchor (42000+replicate) とは共有しない。
    if SPLIT_VARIANT == "S_B":
        return SPLIT_SEED_BASE + replicate
    # S_C: Phase 7e anchor の split seed を意図的に再利用し、
    #      K_TRUE=1..5 の全水準で同一 pair-index mask を得る。
    #      これは accidental collision ではなく pre-registered common-mask design。
    if SPLIT_VARIANT == "S_C":
        return ANCHOR_SPLIT_SEED_BASE + replicate
    raise HarnessStop("split variant is not frozen")

def expected_model_seed(k_true, replicate, k, start) \
    -> MODEL_SEED_BASE + 10000 * k_true + 1000 * replicate + 10 * k + start
```

#### estimand offset の適用範囲（H2 と H4 の責任分離、design §10.7.1）

**`RANDOM_DESIGN`（H2）が estimand-specific offset を与えるのは
`data_seed` と `model_seed` のみである。`split_seed` は H4 が単独で支配し、
estimand を理由に自動 offset しない。**

| seed helper | 依存する gate | `INDEPENDENT` 時の estimand offset |
|---|---|:--:|
| `expected_data_seed(k_true, replicate, estimand)` | **H2** | **YES** |
| `expected_model_seed(k_true, replicate, k, start, estimand)` | **H2** | **YES** |
| `expected_split_seed(k_true, replicate)` | **H4** | **NO — H4 governs** |

```python
# H2 依存: estimand を引数に取る（CRN では offset 0、INDEPENDENT では estimand 別 block）
def expected_data_seed(k_true, replicate, estimand)
def expected_model_seed(k_true, replicate, k, start, estimand)

# H4 依存: estimand を引数に取らない（上の定義のまま）
def expected_split_seed(k_true, replicate)
```

- **`CRN`（← current config, Human Gate 2026-09-01）:** `data_seed` / `model_seed` は
  A と B で対応する（offset 0）。
  **split mask は H4 の定義に従う（current config では `S_C`）。
  「CRN だから split seed も必ず同一」とはしない。**
- **`INDEPENDENT`（*NOT SELECTED*、履歴のため分岐を残す）:** `data_seed` / `model_seed` に
  estimand 別 block を用い、その block も §5 T09 と同じ全数衝突検査の対象にする。
  **`split_seed` には estimand offset を適用しない。**

current config は `CRN` × `S_C` である。この責任分離により、
（*NOT SELECTED* の）**`RANDOM_DESIGN == "INDEPENDENT"` かつ `MASK_DESIGN == "S_C"` も合法**であり、
§3.4 の MC1（anchor mask 一致）を A・B どちらについても満たす。
**H2 の independence が H4=S_C の mask alignment を上書きすることは、設計上あり得ない。**
この場合の independent は data / model RNG に限られ、
`fully independent experiments` / `all random numbers independent` とは呼ばない（design §10.7.3）。

### 2.4 CLI gates

| コマンド | EM | 用途 |
|---|:--:|---|
| `--validate-only` | **0** | 定数・manifest・seed 衝突・score config・frozen config hash を検査して JSON 出力 |
| `--config-gate` | **0** | §3 の **deterministic / algebraic** generator configuration gate |
| `--record-diagnostics` | **0** | §3.2 の RECORD ONLY 診断を計測・保存（**停止条件を持たない**） |
| `--canary` | 2 | leakage falsification（Phase 7e と同じ 2-canary） |
| `--smoke` | 少数 | 1 K_TRUE × 1 replicate × candidate K {2,3,4} × 2 starts = 6 fits |
| `--full --allow-em --confirm-k-true-sweep --estimand <A\|B>` | 168 / estimand | full run（quadruple gate） |

- `--full` は `--allow-em` と `--confirm-k-true-sweep` の**両方**が無いと拒否。
- `--estimand` は module 定数 `ESTIMANDS` と整合しなければ拒否
  （CLI と code の二重確認。片方だけ書き換えた事故を止める）。
- `--full` 実行前に `--validate-only` と `--config-gate` が **PASS していることを内部で再実行して確認**する。
- A+B の場合、full run は **estimand ごとに 1 回ずつ**（別 artifact ディレクトリ）実行する。

---

## 3. Generator configuration gate（**HIGH-03 対応**）

design §12.3 の方針をそのまま実装する。
**blocking gate は deterministic / algebraic な量のみに課す。標本統計量には課さない。**

### 3.1 BLOCKING gate（`--config-gate`、EM fits = 0）

Option B（`ESTIMANDS` に B を含む場合）:

```
G1  configured w_K == W_REF * sqrt(K_REF / K_TRUE)          for all K_TRUE  (exact formula match)
G2  w_K^2 * K_TRUE == W_REF^2 * K_REF                       for all K_TRUE  (numerical tolerance)
G3  resolve_w_true("B", 3) == 1.5                                           (anchor 互換)
G4  generator formula / parameter mapping に unexpected branch が無い
      - generate_dual_data の呼び出し引数が manifest の値と完全一致
      - w0_true / var_f / uniq / family / n / d が全 cell で同一
G5  manifest に expected w_K が列として保存されている
```

Option A（`ESTIMANDS` に A を含む場合）:

```
G1'  configured w_K == 1.5                                   for all K_TRUE
G3'  resolve_w_true("A", 3) == 1.5
G4'  G4 と同じ
G5'  G5 と同じ
```

**Option A には variance 一定条件（G2 相当）を課さない。**
分散が `K_TRUE` で動くことは Option A の estimand 定義そのものだからである（design §8.1）。

### 3.2 RECORD ONLY（**停止条件を持たない**）

`--record-diagnostics` は次を計測して `diagnostics.csv` / `runinfo.json` に保存する。
**いかなる閾値でも full run を停止させない。**

| 項目 | 扱い |
|---|---|
| 標本 `sd(η^Y)` | RECORD ONLY |
| Y density | RECORD ONLY |
| 条件付きエントロピー | RECORD ONLY |
| oracle mean log score | RECORD ONLY |
| 高次モーメント診断（尖度など） | RECORD ONLY |
| `Σ_i‖z_i‖²/n`、`‖F‖_F²`、`tr(F^TF)/K` | RECORD ONLY（解析式との整合確認用） |

**理由（コメントとしてコードに残す）:** Option B では `w_K² K` は代数的に一定だが、
有限データセットで観測される標本 `sd(η^Y)` は確率変動する。
正しい generator でも標本レベルの閾値を超えうるため、
標本統計量を pass/fail gate にすると **false failure** が生じる。

### 3.3 seed rescue 等は禁止のまま維持

diagnostics の値を見て seed / tolerance / K range / replicate を変更してはならない
（design §12.2）。

### 3.4 Split-mask provenance gate（**H4 依存・zero-EM・BLOCKING**）

`--config-gate` の一部として、**full fit より前に**（EM fits = 0 の段階で）
mask の整合を検査する。design §10.4 の 3 案それぞれで要求が異なる。

#### 3.4.0 canonical hash contract（**FROZEN — 実装任せにしない**）

Phase 7e の一次証拠（`heldout_full_pilot_20260824/fit_results.csv`）が保持している mask hash 列は
**`train_mask_hash` と `test_mask_hash` の 2 つであり、`split_mask_hash` という列は存在しない。**
したがって新設する単一 hash が「何を hash した値か」を、blocking gate に使う前に一意に固定する。

```
split_mask_hash   := stable_array_hash(test_mask)
anchor_mask_hash  := Phase 7e の保存済み test_mask_hash
train_mask_hash   := stable_array_hash(train_mask)          （artifact に併せて保持）
anchor_train_mask_hash := Phase 7e の保存済み train_mask_hash
```

**理由:** H4 で揃えたい科学的対象は **held-out pair-index mask**、すなわち `test_mask` である。
よって canonical object を `test_mask` に固定する。

**blocking verification は test / train の両方で行う。**
canonical provenance field は `split_mask_hash`（= `test_mask_hash`）1 本に固定しつつ、
gate では `test_mask` と `train_mask` の両方が anchor と一致することを要求する。
加えて `validate_pair_masks` により train/test の complement semantics も確認する。

```
canonical provenance field : split_mask_hash = test_mask_hash
blocking verification      : test AND train both match anchor
```

```
M0  (全案共通) 生成した mask が validate_pair_masks を満たす
                （対称・対角 False・train ⊻ test = 全非対角・expected_test_pairs = 555）
M1  (全案共通) manifest に split_seed / split_mask_hash / train_mask_hash / mask_design /
                mask_group_id / anchor_mask_hash / anchor_train_mask_hash /
                intentional_seed_reuse が保存される
M2  (全案共通) mask_design == MASK_DESIGN
M3  (全案共通) split_mask_hash == stable_array_hash(test_mask) であること
                （canonical contract 3.4.0 の遵守）
```

**S_A:**

```
MA1  split_mask_hash (= test_mask_hash) が (K_TRUE, replicate) ごとに相異なる
MA2  intentional_seed_reuse == False
```

**S_B:**

```
MB1  同一 replicate の新規 K_TRUE {1,2,4,5} で split_mask_hash (= test_mask_hash) が一致
MB2  その値が Phase 7e K3 anchor の test_mask_hash と一致「しない」ことを記録する
       （不一致は失敗ではない。S_B の定義どおりであることの確認）
MB3  intentional_seed_reuse == True（新規 K_TRUE 間のみ）
```

**S_C（← current config）:**

```
for each replicate r in {1,2,3}:
    # Phase 7e artifact を read-only で読むだけ。再実行しない。
    # 出典: expfam/results/k_selection/heldout_full_pilot_20260824/fit_results.csv
    anchor_test_hash  = Phase 7e stored test_mask_hash(r)
    anchor_train_hash = Phase 7e stored train_mask_hash(r)

    for K_TRUE in {1,2,4,5}:
        for estimand in {A, B}:                        # current config は ESTIMANDS="AB"
            assert stable_array_hash(test_mask)  == anchor_test_hash
            assert stable_array_hash(train_mask) == anchor_train_hash

MC1  上記 assert が test / train ともにすべて成立
MC2  intentional_seed_reuse == True
MC3  anchor_mask_hash       列に anchor_test_hash  が保存される
     anchor_train_mask_hash 列に anchor_train_hash が保存される
MC4  RANDOM_DESIGN == "INDEPENDENT" であっても MC1 が成立する
       （estimand offset は split_seed に適用されないため。§2.3）
MC5  各 new row の anchor_match == True
```

**`RANDOM_DESIGN`（H2）の値は MC1 の要求を緩めない。**
`INDEPENDENT` でも A・B の両方が anchor mask に一致しなければならない。

**不一致なら STOP。** 以下は**禁止**する（design §10.6）:

- seed rescue / seed 差し替えによる整合化
- **Phase 7e を再実行して合わせること**
- tolerance の緩和
- 不一致 cell の drop

**注:** S_C において一致するのは **pair-index held-out mask のみ**である。
`Z` / `F` / `X` / `Y` は `K_TRUE` ごとに別の generator realization であり、
`same dataset` / `paired statistical replicate` / `identical Y` / `identical Z` とは呼ばない
（design §10.4）。

---

## 4. Manifest schema / artifact schema

### 4.1 manifest（estimand あたり 168 行、`k_true → replicate → K → start` 昇順で凍結）

```csv
fit_index,estimand,role,K_TRUE,replicate,K,start,data_seed,split_seed,split_mask_hash,
mask_design,mask_group_id,anchor_mask_hash,intentional_seed_reuse,
model_seed,w0_true,w_true
1,B,sensitivity,1,1,1,1,51101,<split>,<hash>,S_C,r1,<anchor_hash>,True,541011,-1.0,2.598076211353316
...
168,B,sensitivity,5,3,7,2,51503,<split>,<hash>,S_C,r3,<anchor_hash>,True,583072,-1.0,1.161895003862225
```

（上例は `HIERARCHY = H3_A` の場合。`H3_B` なら `role = coequal_B`、
`ESTIMANDS != "AB"` なら `role = single`。§2.2b）

Phase 7e schema との差分:
**`estimand` / `role` / `K_TRUE` / `w0_true` / `w_true` に加え、
§3.4 の mask provenance 列（`split_mask_hash` / `mask_design` / `mask_group_id` /
`anchor_mask_hash` / `intentional_seed_reuse`）を追加**（design §10.5）。
exact field name は既存 schema との整合を見て決めてよいが、上記の意味は必ず保持する。

`mask_group_id` は「同一 mask を共有する cell 群」の識別子である。
S_A では `(K_TRUE, replicate)` ごとに別 ID、
S_B / S_C では `replicate` ごとの ID になる（S_B は新規 4 水準のみ、S_C は 5 水準すべて）。

#### `random_design` と `mask_design` は独立した field として保持する

H2 と H4 の責任分離（design §10.7.1）を provenance 上でも保つため、
両者を **1 つの field に畳まない**。

```
random_design = CRN | INDEPENDENT      # H2: data_seed / model_seed のみを支配
mask_design   = S_A | S_B | S_C        # H4: split_seed / pair-index mask を支配
```

**`random_design = INDEPENDENT` と `mask_design = S_C` の同時指定は合法である。
config validator はこの組み合わせを reject してはならない。**
（reject すると design §10.7.3 で human に提示した選択肢が実行不能になる。）

### 4.2 出力 artifact（行数は **current frozen config へ確定**）

current config は `NEW_K_TRUE = {1,2,4,5}`・`REPLICATES = {1,2,3}` なので、
**estimand あたりの new dataset cell 数は `4 × 3 = 12` である。**
以下の行数はこの前提で一意に固定する（範囲表記を残さない）。

| file | 行数（estimand あたり） | Phase 7e からの差分 |
|---|---:|---|
| `manifest.csv` | 168 | `estimand`, `role`, `K_TRUE`, `w0_true`, `w_true` および mask provenance 列を追加 |
| `fit_results.csv` | 168 | 同上。他列は Phase 7e と同一 |
| `replicate_selection.csv` | 4×3×7 = 84 | `estimand`, `role`, `K_TRUE` 列追加 |
| `cell_selection.csv` | **12** | `estimand, role, K_TRUE, replicate, selected_k, tie_candidates, best_score, second_best_score, margin, signed_error, abs_error, label` |
| `aggregate_summary.csv` | K_TRUE 別 k_wise 28 行 + pilot 行 | `estimand`, `role`, `K_TRUE` 列追加 |
| `config_gate.csv` | G1–G5 + M0–M3 + MA/MB/MC 各判定 1 行 | §3.1 と §3.4 の判定結果 |
| `mask_provenance.csv` | **12** | §4.2.1 |
| `diagnostics.csv` | **12** | §4.2.2 |
| `runinfo.json` / `runinfo.md` | — | `estimand` / `role` / `hierarchy` / `w_true_rule` / **`random_design`（H2）/ `mask_design`（H4）を別 field で** / `anchor_reference` / `config_gate` / `diagnostics` を追加 |
| `stdout.log` | — | §9 |

#### 4.2.1 `mask_provenance.csv` — **exactly 12 rows / estimand**

key: `(estimand, K_TRUE, replicate)`、`K_TRUE ∈ {1,2,4,5}` × `replicate ∈ {1,2,3}`。
artifact directory が estimand 別であっても `estimand` 列は保持してよい。

各 new row が保持する provenance:

```
estimand, role, K_TRUE, replicate,
split_seed, split_mask_hash (= test_mask_hash), train_mask_hash,
mask_design, mask_group_id,
anchor_mask_hash (= Phase 7e stored test_mask_hash),
anchor_train_mask_hash (= Phase 7e stored train_mask_hash),
intentional_seed_reuse, anchor_match
```

**Phase 7e anchor の 3 行そのものを `mask_provenance.csv` へコピーしない。**
anchor evidence は各 new row の `anchor_mask_hash` / `anchor_train_mask_hash` から参照する。
これにより §4.3 の
「`k_true_selection_matrix.csv` が anchor と new result を同居させる唯一の統合成果物」
という規約と整合する。

#### 4.2.2 `diagnostics.csv` — **exactly 12 rows / estimand**

RECORD ONLY diagnostics は dataset realization ごとに異なるため、
key は `(K_TRUE, replicate)`、`K_TRUE ∈ {1,2,4,5}` × `replicate ∈ {1,2,3}` の 12 行。

各 row が保持する項目:

```
K_TRUE, replicate, estimand, role,
sample sd(eta_Y), Y density, conditional entropy, oracle mean log score,
higher-moment diagnostics, latent / F invariants
  （Sigma_i||z_i||^2/n, ||F||_F^2, tr(F^T F)/K）
```

**`K_TRUE=3` anchor について新しい diagnostic row を生成しない。**
Phase 7e は frozen read-only anchor であり、
Phase 8a の new diagnostics lineage へ新しい K3 measurement を混ぜない。
report では **`K3 diagnostics were not newly generated`** と明記する。

**これらの値はいかなる場合も full run を停止させない**（RECORD ONLY、§3.2）。

### 4.3 anchor 統合成果物（**別ファイル・provenance 列必須**）

`k_true_selection_matrix.csv`（estimand あたり 15 行）
— **唯一 anchor と新規を同居させるファイル**:

列は 11 個（`estimand, role, K_TRUE, replicate, selected_k, signed_error, abs_error,
label, lineage, run_code_sha, artifact_dir`）。**全行が header と同じ列数でなければならない。**

Option B（`role = sensitivity`、H3-a）の matrix 例:

```csv
estimand,role,K_TRUE,replicate,selected_k,signed_error,abs_error,label,lineage,run_code_sha,artifact_dir
B,sensitivity,3,1,3,0,0,exact,phase7e_anchor,b9311e64...,expfam/results/k_selection/heldout_full_pilot_20260824
B,sensitivity,3,2,3,0,0,exact,phase7e_anchor,b9311e64...,expfam/results/k_selection/heldout_full_pilot_20260824
B,sensitivity,3,3,5,2,2,over,phase7e_anchor,b9311e64...,expfam/results/k_selection/heldout_full_pilot_20260824
B,sensitivity,1,1,<k>,<signed>,<abs>,<label>,phase8a_new,<sha>,expfam/results/k_selection/k_true_robustness_B_<date>
B,sensitivity,1,2,<k>,<signed>,<abs>,<label>,phase8a_new,<sha>,expfam/results/k_selection/k_true_robustness_B_<date>
```

Option A（`role = primary`、H3-a）の matrix では `estimand,role` が `A,primary` になる:

```csv
estimand,role,K_TRUE,replicate,selected_k,signed_error,abs_error,label,lineage,run_code_sha,artifact_dir
A,primary,3,1,3,0,0,exact,phase7e_anchor,b9311e64...,expfam/results/k_selection/heldout_full_pilot_20260824
A,primary,1,1,<k>,<signed>,<abs>,<label>,phase8a_new,<sha>,expfam/results/k_selection/k_true_robustness_A_<date>
```

`K_TRUE=3` の 3 行は A・B いずれの matrix でも同一の anchor 値になる
（`w_3 = 1.5` が共通のため。design §11.1）。**anchor fits は再実行しない。**

**`lineage` / `run_code_sha` / `artifact_dir` 列は必須**（KI-002、design §14 C7）。
K_TRUE=3 の 3 行は Phase 7e の `replicate_selection.csv` から**読み取るだけ**で、
再計算も再実行もしない。A+B の場合、この 3 行は A・B 双方の matrix に同一値で現れる
（`w_3 = 1.5` が共通であるため。design §11.2）。

**`best_score` / `margin` はこの統合ファイルに含めない**
（design §13: score 水準の `K_TRUE` 間比較を禁止しているため）。

---

## 5. Static tests（EM を一切呼ばない）

`test_k_true_robustness_sweep.py`。Phase 7e の 120 tests と同じ方針
（fake adapter / temp dir / artifact 破壊による negative test）。

| # | test | 内容 |
|---|---|---|
| T01 | `resolve_w_true("A",3) == resolve_w_true("B",3) == 1.5` | anchor 互換・A/B anchor 共有の code 保証 |
| T02 | Option B で `w_K² · K_TRUE` が全 `K_TRUE` で一定（数値許容内） | **代数的**不変量の確認（標本統計量ではない） |
| T03 | Option B で `w_K == 1.5*sqrt(3/K_TRUE)`（exact formula） | G1 |
| T04 | Option A で `w_K == 1.5`（全 K_TRUE） | G1' |
| T05 | manifest 行数 == 168（estimand あたり） | fit budget |
| T06 | manifest の順序・key 集合が凍結どおり | `{1,2,4,5}×{1,2,3}×{1..7}×{1,2}` |
| T07 | manifest に `K_TRUE=3` が**含まれない** | anchor 再実行防止 |
| T08 | model seed 168 個すべて distinct | seed 一意性 |
| T09 | **data seed / model seed** が Phase 7e 全 seed と交差しない（S_A / S_B / S_C いずれでも、A+B の場合は両 block） | **意図しない seed collision** |
| T09b | `SPLIT_VARIANT == "S_C"` のとき split seed の Phase 7e 再利用が**意図的**として記録される（`intentional_seed_reuse == True`）。この再利用が T09 の collision 検査で失敗として扱われないこと | pre-registered common-mask design（**split seed に一律の一意性を課さない**） |
| T10 | **data / model** seed が役割間で重複しない | 役割衝突（split seed は H4 依存のため対象外） |
| T11 | seed が `(K_TRUE, replicate, K, start)` の関数として決定的 | 再現性 |
| T12a | `S_A` で split_mask_hash が `(K_TRUE, replicate)` ごとに相異なる | MA1 |
| T12b | `S_B` で**新規 `K_TRUE` {1,2,4,5} の間だけ** mask が一致し、**K3 anchor とは一致しない**ことを検査 | MB1 / MB2（partial alignment。「全 K_TRUE 共通」ではない） |
| T12c | `S_C` で `K_TRUE ∈ {1,2,4,5}` の mask hash が Phase 7e K3 anchor の保存値と一致 | MC1（**anchor は読み取りのみ・再実行しない**） |
| T12d | mask 不一致時に `HarnessStop`（seed 差し替え・Phase 7e 再実行の経路が存在しない） | §3.4 fail-closed |
| T12e | manifest に `split_mask_hash` / `mask_design` / `mask_group_id` / `anchor_mask_hash` / `intentional_seed_reuse` が存在 | M1 / M2 |
| T12f | **`expected_split_seed` が `estimand` を引数に取らず、`RANDOM_DESIGN` を参照しない** | **H2/H4 責任分離（F-01）** |
| T12g | `RANDOM_DESIGN` を `CRN` / `INDEPENDENT` に振っても `expected_split_seed` の戻り値が不変 | 同上 |
| T12h | `RANDOM_DESIGN == "INDEPENDENT"` かつ `SPLIT_VARIANT == "S_C"` で MC1 が成立し、config validator が reject しない | **MC4・合法な組み合わせの保証** |
| T12i | `RANDOM_DESIGN == "INDEPENDENT"` で `expected_data_seed` / `expected_model_seed` は estimand ごとに異なる値を返す | H2 の offset が data / model にのみ効く |
| T12j | `runinfo` / manifest に `random_design` と `mask_design` が**別 field** として保存される | provenance 上の責任分離 |
| T12k | **`split_mask_hash == stable_array_hash(test_mask)`**（canonical contract §3.4.0） | **HIGH-01: hash 定義の一意性** |
| T12l | `anchor_mask_hash` が Phase 7e stored `test_mask_hash`、`anchor_train_mask_hash` が stored `train_mask_hash` と一致 | 同上 |
| T12m | `S_C` gate が **test と train の両方**を検査する（片方だけ一致では PASS しない） | MC1 |
| T12n | `mask_provenance.csv` が **exactly 12 rows / estimand**、key `(estimand, K_TRUE, replicate)` が重複なし | **MEDIUM-03** |
| T12o | `diagnostics.csv` が **exactly 12 rows / estimand**、key `(K_TRUE, replicate)` が重複なし、`K_TRUE=3` の行を含まない | **MEDIUM-03** |
| T12p | `mask_provenance.csv` に Phase 7e anchor 行そのものが含まれない（anchor は hash 参照のみ） | §4.2.1 |
| T12q | `k_true_selection_matrix.csv` の全行が header と同じ列数（11 列） | **MEDIUM-02: 列ずれ防止** |
| T13 | `make_pair_split` が `K_TRUE` に依存しない | split の K 不変性 |
| T14 | `score_config_hash` が Phase 7e の値と一致 | **score protocol の同一性** |
| T15 | `select_k_from_two_starts` を Phase 7e から import している | selector 同一性 |
| T16 | `Σ_i‖z_i‖²/n == k_true`（全 k、tol 1e-12） | 生成規約の恒等式（design (5.1)） |
| T17 | `‖F‖_F² == d(1−uniq)`（全 k、tol 1e-9） | design (5.5) |
| T18 | k=1 で `F` の rank == 1、全行ノルムが `√(1−uniq)` | K_TRUE=1 boundary |
| T19 | k=1 で `Z → −Z` に対し score が不変 | O(1) 不変性 |
| T20 | **config gate（G1–G5 および §3.4 の M0–M2 / MA / MB / MC）が違反時に `HarnessStop`** | fail-closed |
| T21 | **`diagnostics` の値がいかなる場合も full run を停止させない** | **HIGH-03: false failure の防止** |
| T22 | `--full` が `--allow-em` 無しで拒否 | CLI gate |
| T23 | `--full` が `--confirm-k-true-sweep` 無しで拒否 | CLI gate |
| T24 | `--estimand` と module 定数の不整合で拒否 | 二重確認 |
| T24b | `ESTIMANDS == "AB"` かつ `HIERARCHY` 未確定のとき `resolve_role` が `HarnessStop` | **H3 未 freeze のまま実行できない** |
| T24c | `HIERARCHY == "H3_A"` で `role(A)=="primary"` かつ `role(B)=="sensitivity"`、`H3_B` で `coequal_A` / `coequal_B` | **H3 の役割固定（MEDIUM-01）** |
| T24d | manifest / runinfo / `cell_selection.csv` の `role` が `resolve_role` の値と全行一致 | role の provenance |
| T25 | `--validate-only` / `--config-gate` / `--record-diagnostics` の `em_fits_executed == 0` | **no-EM 検証** |
| T26 | module import で `em_runner` が `sys.modules` に入らない | **no-EM 境界** |
| T27 | 既存 artifact がある出力先で full が拒否 | 上書き防止 |
| T28 | 未知 artifact が出力先にあると拒否 | Phase 7e `_require_only_expected_artifacts` 相当 |
| T29 | `k_true_selection_matrix.csv` の anchor 3 行が Phase 7e CSV と一致 | anchor 読み取りの正しさ |
| T30 | anchor 行の `lineage == "phase7e_anchor"` かつ `run_code_sha` / `artifact_dir` が非空 | provenance 列必須 |
| T31 | Phase 7e artifact ディレクトリへの書き込み試行で `HarnessStop` | **凍結成果物の保護** |
| T32 | `k_true_selection_matrix.csv` に `best_score` / `margin` 列が存在しない | 水準間 score 比較の構造的禁止 |

## 6. Adversarial tests

| # | 攻撃 | 期待 |
|---|---|---|
| A01 | fit boundary に raw test Y を混入させる fake payload | `HarnessStop`（leakage） |
| A02 | `ScoreOnlyTarget` を fit 引数に渡す | 拒否 |
| A03 | test mask を fit 後にすり替える | hash mismatch で拒否 |
| A04 | 1 セルの `w_true` だけ manifest と違う値で fit | `fit_config_hash` 不一致で拒否 |
| A05 | 1 fit の model seed を manifest 外の値にする | seed 規約違反で拒否 |
| A06 | `K_TRUE=3` を新規 manifest に混入させる | T07 と同じく拒否 |
| A07 | config gate 通過後に `ESTIMANDS` / `w_true` を書き換えて full 起動 | `runinfo` の config hash 不一致で拒否 |
| A08 | 168 行のうち 1 行を欠落させて集計 | 行数厳密検査で BLOCKER |
| A09 | `replicate_selection.csv` が header-only | 84 行厳密で BLOCKER（Phase 7e addendum §8 の fail-open 教訓） |
| A10 | `config_gate.csv` / `diagnostics.csv` 欠損 | required artifact 検査で BLOCKER |
| A11 | selected K を artifact 側で改竄 | 独立 audit の再計算で BLOCKER |
| A12 | 2 セルで同一 `(estimand, K_TRUE, replicate, K, start)` | 重複 key で BLOCKER |
| A13 | **`diagnostics` の値に閾値を課して停止させる実装を混入** | T21 で検出（HIGH-03 の回帰防止） |
| A14 | `S_C` で 1 replicate だけ mask hash が anchor と不一致 | §3.4 MC1 で BLOCKER（seed 差し替え・Phase 7e 再実行の経路なし） |
| A15 | `S_B` の結果を「全 `K_TRUE` 共通 mask」として集計・記述 | `mask_design` / `mask_group_id` 列と audit の整合検査で BLOCKER（**partial alignment のみ**） |
| A16 | `S_C` の anchor mask を得るために Phase 7e を再実行しようとする | T31（Phase 7e ディレクトリ保護）＋ audit の `lineage` 検査で BLOCKER |
| A17 | **full run 後に `HIERARCHY` を書き換えて role を入れ替える**（A primary → B primary 等） | `runinfo` の config hash 不一致＋`role` 列との突合で BLOCKER（**H3 role switching の防止**） |
| A18 | `H3_B`（co-equal）なのに片方だけを main result として集計・記述 | `role` 列が `coequal_A` / `coequal_B` であることと report template の整合検査で BLOCKER |
| A19 | A/B を同一 estimand の replication として集計（両者を平均する等） | `estimand` 列の集約禁止検査で BLOCKER（**別 generator family である**） |
| A20 | **`RANDOM_DESIGN == "INDEPENDENT"` を理由に `split_seed` へ estimand offset を加える実装を混入** | T12f / T12g で検出。S_C では MC1 / MC4 でも BLOCKER（**F-01 の回帰防止**） |
| A21 | config validator が `INDEPENDENT` × `S_C` を reject する実装を混入 | T12h で検出（合法な組み合わせを塞がない） |
| A22 | `split_mask_hash` を `train_mask` や `(train,test)` 連結から計算する実装を混入 | T12k で検出（**canonical contract 違反**） |
| A23 | `S_C` gate で test mask のみ一致・train mask 不一致の状態を PASS させる | T12m / MC1 で BLOCKER |
| A24 | `diagnostics.csv` に `K_TRUE=3` の新規 measurement 行を追加 | T12o で BLOCKER（**anchor lineage の混入防止**） |
| A25 | A/B の結果差を「Y variance の isolated causal contribution」として report に記述 | report template の解釈境界検査で BLOCKER（design §13） |

**audit script は harness の selector を import せず、artifact のみから
seed 規約・selector 算術・行数・key 集合・hash 一貫性を再計算する**
（Phase 7e `audit_heldout_full_pilot.py` と同方針、fail-closed）。

---

## 7. Smoke / full gating plan

### S2: smoke（少数 fit）

```
python tools/research_audit/run_k_true_robustness_sweep.py --validate-only
python tools/research_audit/run_k_true_robustness_sweep.py --config-gate
python tools/research_audit/run_k_true_robustness_sweep.py --record-diagnostics
python tools/research_audit/run_k_true_robustness_sweep.py --canary
python tools/research_audit/run_k_true_robustness_sweep.py --smoke
```

smoke 対象: `K_TRUE=1`（最も boundary に近い水準）× `replicate=1` ×
candidate K `{2,3,4}` × starts `{1,2}` = **6 fits**。
smoke 専用 seed block を使い、full の seed とは衝突させない。

**smoke 完了後に人間へ返す。自動で full へ進まない。**

### S3: full run

```
python tools/research_audit/run_k_true_robustness_sweep.py \
    --full --allow-em --confirm-k-true-sweep --estimand <A|B>
```

> **S3 はまだ許可されていない。** design §15.1 のとおり、`A: IMPLEMENT_..._NEXT` が許可するのは
> implementation と zero-EM validation のみである。336 fits の実行には
> **smoke 後の independent review と明示的な人間の承認**が別途必要である。

前提（すべて満たさなければ起動しない）:

- design §16 の H1–H4 が module 定数に反映されている
  （`ESTIMANDS="AB"` / `HIERARCHY="H3_A"` / `RANDOM_DESIGN="CRN"` / `MASK_DESIGN="S_C"`、
  Human Gate 2026-09-01 で確定済み）
- **`role` が manifest / runinfo / report template に `A→primary` / `B→sensitivity` で固定されている**
  （design §16 Reporting freeze）
- **`--config-gate` の MC1–MC4（S_C anchor mask 一致）が PASS**
- `--validate-only` / `--config-gate` / `--canary` / `--smoke` がすべて PASS
- **smoke に対する independent review が完了し、人間が full run を明示的に承認している**
- working tree が clean で、出力先ディレクトリが存在しない
- branch が `experiment/<issue#>-k-true-robustness`（main ではない）

full 完了後:

1. `audit_k_true_robustness_sweep.py` を実行（artifact のみ、BLOCKER 0 / HIGH 0 を要求）
2. report を **script で自動生成**（手作業転記しない — CLAUDE.md §7）
3. `EXPERIMENT_REGISTRY.md` への追記案を作成し、**人間の承認を得てから**追記
4. Draft PR 作成 → 独立 review → **merge は人間**

**full 完了後に禁止される行為（design §16 Reporting freeze）:**

- **primary / sensitivity の入れ替え**（A primary → B primary、およびその逆）
- **sensitivity result の昇格または隠蔽**
- **favorable な option だけを本文採用することを result を見て決めること**
- **A/B role の post-hoc reinterpretation**
- `H3_B` のもとで片方だけを main result として扱うこと

---

## 8. No-EM validation path

`--validate-only` / `--config-gate` / `--record-diagnostics` は
**EM を実行しないことを構造的に保証する**:

1. これらのコードパスから `em_runner` を import しない
   （Phase 7e と同じく、EM import は `AuthorizedEMFitAdapter.fit` の内部に閉じる）。
2. 出力 JSON に `"em_fits_executed": 0` を必ず含める。
3. T25 / T26 が `sys.modules` を検査して、
   これらのモード実行後に `em_runner` が読み込まれていないことを assert する。
4. これらのモードは generator と純関数のみを呼ぶ
   （`generate_dual_data` / `make_pair_split` / numpy）。

**design Issue #47 で行った検証はすべてこの経路上にある（EM fits = 0）。**

---

## 9. stdout capture の provenance（optional wrapper）

Phase 7e では `stdout.log` の **outer capture command が repository から復元できない**という
limitation が残った（provenance addendum §5）。Phase 8a では次の 2 案がある。

| 案 | 内容 | 追加ファイル |
|---|---|---|
| **W1（推奨・optional）** | committed wrapper（`tools/research_audit/run_k_true_robustness_full.ps1`）から full run を起動し、capture 方法をコード上に固定。`runinfo.json` に `outer_command` を記録 | §1.1 #4 を allowed list に追加 |
| **W2** | wrapper を作らず、`runinfo.json` に `outer_command` を**引数として渡して**記録する | 追加ファイルなし |

**どちらでもよい。W1 を採る場合のみ §1.1 #4 を allowed list に含める。**
本 plan は wrapper を **required とはしない**（LOW-01 の解消）。

いずれの案でも **Phase 7e 側の runner helper は変更しない**（addendum §9 の判断を尊重）。
`_require_no_existing_full_artifacts` への `stdout.log` 追加は
capture 方法を仮定することになるため、Phase 8a 側でも行わない。

---

## 10. Fit budget（**FROZEN — H1 = A+B**）

design §11 のとおり。

```
4 new K_TRUE x 3 rep x 7 cand K x 2 starts x 2 estimands = 336 new fits
existing Phase 7e K_TRUE=3 anchor                        =  42 fits (A/B 共有)
unique total                                             = 378 fits
per-estimand matrix : 168 new + 42 anchor                = 210-fit equivalent
```

**anchor 42 は A・B で共有され、二重に数えない**（`K_TRUE=3` で `w = 1.5` が共通）。
**`210 × 2 = 420` unique fits ではない。**
full run は estimand ごとに 1 回ずつ、計 **2 回**。

| Strategy | 状態 | full run 回数 | unique new | unique total |
|---|---|---:|---:|---:|
| 1（A only） | *NOT SELECTED* | 1 | 168 | 210 |
| 2（B only） | *NOT SELECTED* | 1 | 168 | 210 |
| **3（A+B）** | **SELECTED** | **2** | **336** | **378** |
| 4（redesign） | *NOT SELECTED* | — | 未定 | 未定 |

**H3 / H4 の選択は fit 数を変えない。** H3 は role ラベルの固定のみ、
H4 は mask の作り方のみを変え、cell 数・candidate K・start 数・replicate 数・estimand 数は不変である。

**budget が確定していることは full run の実行許可を意味しない（§7・design §15.1）。**

---

## 11. 本 plan で明示的に行わないこと

- 実装（S1 以降は次の Issue）
- EM fit / smoke / full run
- estimand（A / B / A+B / redesign）の決定 — design §16 H1

- H1–H4 の決定 — **Human Gate 2026-09-01 で確定済み**（design §16）。本 plan では変更しない

- `n` sweep / asymptotic theory（別 Issue）
- canonical docs（`RESEARCH_MASTER.md` / `KNOWN_ISSUES.md` / `EXPERIMENT_REGISTRY.md`）の更新
- per-column の昇格 / 新 criterion の実装 / real-data K selection
- Phase 7e artifact・report の書き換え
