# 修士研究｜現在地・理論整理・潜在次元選択

> **研究題目:** 属性情報付き関係データの潜在構造モデルの指数型分布族への拡張
> **最終更新:** 2026-09-06

属性データ `X` と関係データ `Y` が同じ潜在変数 `Z` を共有するモデルを、指数型分布族（Gaussian / Bernoulli / Poisson）へ拡張している。
いま取り組んでいるのは **「潜在次元 `K` をどう決めるか」** で、理論的な意味づけと有限標本での挙動の両方を整理している段階。

---

## 🔎 30秒で分かる現在地

- ✅ **モデル自体は生成モデルとして成立している。** X 側 3 × Y 側 3 = **9 通り**すべてで同時分布は proper
- ✅ **Poisson-X では条件付きで真の潜在次元を識別できる。** 母集団モーメントから `FF^T` が復元できる
- ✅ **昔の K 選択成功は実在した。** fixed 系列で 30/30、historical P/B `K=5` で 5/5
- ⚠️ **small-n では昔の条件でも K 選択が悪化する。** `n=150` の 5/5 が `n=50` で 1/5 まで落ちる
- ⚠️ **8→40 反復にしても解決しなかった。** 反復数不足だけが原因ではない
- 🔍 **historical の「BIC」は standard Schwarz BIC ではなかった。** 別の量（Q2）を使っていた
- ❓ **Bernoulli-Y の識別可能性と BIC consistency は未解決。** Bernoulli-Y は主要実験で使っている分布

> 🧭 **現在は「さらに実験を増やす段階」ではない。**
> どの未解決問題を修論の中心に置くかを、先生と相談して決める段階。

---

## 🧩 研究の全体像

### モデル

```
潜在変数    z_i  ~ N(0, I_K)                     i = 1..n
属性        η^X_il = f_l^T z_i                   バイアスなし
関係        η^Y_ij = w0 + w z_i^T z_j            i < j、w0・w はスカラー
```

**`X` と `Y` は同じ `Z` を共有する。** これがこのモデルの核で、属性情報が関係の推定を助け、関係情報が属性の潜在構造推定を助ける、という構造になっている。

同時分布:

```
p(Z, X, Y | θ, K) = Π_i p(z_i) · Π_{i,l} p_X(x_il | z_i) · Π_{i<j} p_Y(y_ij | z_i, z_j)
```

### 対象 family

| | Gaussian | Bernoulli | Poisson |
|---|:---:|:---:|:---:|
| **X 側** | ✓ | ✓ | ✓ |
| **Y 側** | ✓ | ✓ | ✓ |

**X 側 3 × Y 側 3 = 9 通りの組合せすべてで、有限パラメータの下で同時分布は proper（全確率が 1）。**

> ⚠️ **「モデルが proper」と「必要な周辺モーメントが有限」は別問題。**
> 例: Poisson-Y は分布としては常に成立するが、分散が有限なのは `|w| < 1/2` のときだけ。

---

## 🎯 「真の潜在次元 K」の整理

**ここが今回いちばん大きな整理。** 「真の K」には少なくとも 4 つの意味がある。

| 記号 | 意味 | 今の研究での位置づけ |
|---|---|---|
| **`K_GEN`** | 人工データ生成時に指定した `K` | 手続き上の生成次元。一般には統計的 estimand とは限らない |
| **`K*`** | 観測分布を表現できる最小の `K` | 理論上もっとも自然な「真の K」候補 |
| **`K^rank`** | X 側から rank として識別される `K` | Poisson-X・Gaussian-X で利用 |
| **`K_PRED`** | 有限標本で予測性能が最良の `K` | held-out 予測で選ばれる `K` |

> 🔴 **`K_GEN`・`K*`・`K_PRED` は一般には一致しない。**
> 関係は `K^rank ≤ K* ≤ K_GEN` で、**どの不等号も等号とは限らない。**

### Poisson-X では条件付きで `K* = K_GEN`

母集団モーメントから

```
‖f_l‖² = 2 log E[X_l]          f_l·f_m = log{ E[X_l X_m] / (E[X_l] E[X_m]) }
```

により `FF^T` が復元でき、そこから挟み込みで結論が出る。

```
rank(F) = K_GEN  ⟹  K^rank = K_GEN
K^rank ≤ K* ≤ K_GEN
─────────────────────────────
⟹  K* = K_GEN
```

**⚠️ 条件付きで成立。以下がすべて必要:**

- [x] canonical unclipped model（リンク関数を打ち切らない）
- [x] M-closed（真の分布が候補族に含まれる）
- [x] `rank(F) = K_GEN`
- [x] `d ≥ K_GEN`
- [x] population moments（有限標本ではない）
- [x] candidate space が必要な次元を含む（どの `K` でも同じ canonical Poisson-X）

現在の人工データ実験（`canonical-clean-v1`）はこれらを満たすよう設計してある。

> ❓ **一方、Bernoulli-Y 一般については K identifiability が未解決。**
> 辺 1 本の周辺分布はスカラー 1 個の情報しかなく、`K` の情報は共有ノードを持つ部分構造に入るが、それが `K` を決めるかは証明できていない。

---

## 📐 理論面で分かったこと

| 項目 | 状態 | 内容 |
|---|---|---|
| 生成モデル properness | ✅ 解決 | 9 family combinations すべてで成立 |
| Poisson-X K identifiability | ✅ 条件付き | `FF^T` を population moments から復元 |
| Gaussian-X | △ 条件付き | `Σ` 既知なら rank から識別可能。`Σ` 未知は未解決 |
| Bernoulli-X | ❓ 未解決 | `d>1` の一般的 K identifiability 未証明 |
| Gaussian-Y | ✅ 条件付き | `z_i^T z_j` のキュムラントから単一 dyad で識別可能（`w ≠ 0`） |
| Bernoulli-Y | ❓ 未解決 | **現在の主要実験で使う family** |
| Poisson-Y | △ | moment existence（`\|w\| < 1/r`）は解決。identifiability は未解決 |
| BIC consistency | ❓ 未解決 | current model で standard BIC consistency 未証明 |

### 証拠ラベルの意味

| ラベル | 意味 |
|---|---|
| **PROVED** | 数式として証明した。条件は明示する |
| **DERIVED** | 既存の定義・コード・一次証拠から直接導ける |
| **EMPIRICAL** | 有限の実験で観測しただけ。**証明ではない** |
| **UNRESOLVED** | まだ言えない |

> 📌 **実験結果は EMPIRICAL であり、理論の証明には使わない。** この区別を崩さないことが今回の整理の主眼。

---

## 📊 BIC について分かったこと

潜在変数 `Z` の扱いで、3 つの量を区別する必要があった。

| 名前 | 数式 | 意味 | どこで使われているか |
|---|---|---|---|
| **Q1** | `log p(X,Y \| Z, θ)` | conditional / plug-in | 原論文 Eq.(16)/(26) 側 |
| **Q2** | `log p(Z,X,Y \| θ)` | complete-data | **historical expfam criterion** |
| **Q3** | `log p(X,Y \| θ)` | observed marginal | **standard Schwarz BIC が通常対象** |

`Q3 = log ∫ p(Z,X,Y|θ) dZ` で、`Z` を**積分する**のが Q3 だけ。

> 🔴 **過去の「BIC で真の K を選べた」という実験結果は事実。**
> **ただし、それは standard Schwarz BIC の一致性を確認したことを意味しない。**
> historical criterion は Q2 を、原論文 Eq.(26) は Q1 を使っており、どちらも Q3 ではない。

### standard BIC consistency が未証明である理由

「モデル族が入れ子でないから使えない」という説明は**理由として誤り**（非入れ子モデルの比較に BIC を使うのは標準的）。実際に未整理なのは:

- 回転不変性による**特異性**（Fisher 情報が退化。RLCT が未知）
- **境界・冗長次元**でのパラメータ
- `Y_ij` と `Y_ik` が `z_i` を共有する**依存構造**
- **有効標本数の定義**（ノード数 `n` か辺数 `n(n−1)/2` か）
- **漸近レジーム**の選択
- `K` をまたぐ**モデル間の関係**
- **分布族ごとの正則性**

### current S1 の位置づけ

- held-out Bernoulli log score
- **scoring rule 自体は proper**（ここは問題ではない）
- ただし予測分布を積分したものではなく、**推定値を代入した plug-in predictor** を評価している
- ❓ **`K_PRED` が `K*` と一致する保証は未解決**

---

## 🧪 これまでの K 選択実験

### ① 過去実験の再確認

**目的:** 昔、本当に K 選択まで成功していたか。

| 系列 | 条件 | 結果 |
|---|---|---|
| **fixed** | P/B・G/P・B/G、`K_TRUE=3`、`n=150`、候補 K 1..9、10 trials × 3 | **30/30** |
| **historical** | P/B、`K_TRUE=5`、`n=150`、候補 K 1..10、5 trials | **5/5** |

> ✅ **昔の成功は本物。**
> 「K を固定して RMSE を見ていただけ」ではなく、候補集合から K を選んで当てていた。

---

### ② Current clean K-selection

**目的:** canonical-clean generator でも K を選べるか。
**条件:** X=Poisson / Y=Bernoulli、`K_TRUE=5`、held-out S1、8 replicates、候補 K 1..7

| n | exact K=5 |
|---:|---:|
| 50 | 2/8 |
| 75 | 0/8 |
| 100 | 4/8 |
| 150 | **8/8** |

> ⚠️ **`n=150` では選択できるが、small-n では under-selection が増える。**

---

### ③ EM continuation

**目的:** 8 反復で止めていることが原因か。
**条件:** ②と同じ data / seed / mask / 初期値のまま、反復だけ 8 → 40 へ延長

| n | 8 iter | 40 iter |
|---:|---:|---:|
| 50 | 2/8 | 2/8 |
| 75 | 0/8 | 1/8 |
| 100 | 4/8 | 6/8 |
| 150 | 8/8 | 7/8 |

> ⚠️ **40 反復にしても一貫して K=5 へ改善しない。**
> したがって「8 反復だけが原因」とは言えない。
> （追加反復では新しい Monte Carlo サンプルも増えるので、純粋な最適化効果の測定ではない点にも注意）

---

### ④ Historical bridge experiment

**目的:** 昔の成功条件でも small-n になると崩れるか。
**条件:** `n` 以外は historical 条件を完全に固定（generator・EM・criterion・seed 規約すべて）

| n | exact K=5 |
|---:|---:|
| 50 | **1/5** |
| 75 | 4/5 |
| 100 | 5/5 |
| 150 | 5/5 |

`n=150` の 50 セルは過去の記録と数値が完全一致することを確認したうえで実施。

> ⚠️ **historical 条件でも small-n で K 選択は悪化する。**
> **sample size は実際に影響する要因の 1 つ。**

**ただし `n=75` を見ると:**

| | OLD historical | CURRENT clean |
|---|---:|---:|
| n=75 | **4/5** | **0/8** |

> 🔴 **sample size だけでは OLD/CURRENT 差を説明できない。**

---

## ❌ 単純には説明できなかったこと

- [x] **「Poisson だから K が選べない」** → 過去 P/B `K=5` で 5/5、現在も `n=150` で 8/8。**否定される**
- [x] **「K=5 だから難しい」** → 過去 `K=5` でも成功している
- [x] **「8 iterations だから失敗」** → 40 iterations でも一貫改善せず
- [x] **「current 実装が完全に壊れている」** → `n=150` では 8/8 で K=5 を選択
- [ ] **sample size** → 実際に影響するが、**全差を説明するわけではない**（`n=75` の 4/5 対 0/8）

---

## 🔀 過去と現在で残っている違い

| 項目 | OLD historical | CURRENT clean |
|---|---|---|
| X / Y | Poisson / Bernoulli | Poisson / Bernoulli |
| `K_TRUE` | 5 | 5 |
| signal | 強い | matched / 弱め |
| `w` | 1.5（固定） | `sqrt(3/K)` ≈ 0.77（K=5） |
| `‖f‖²` | 0.9（行正規化） | 0.5（平均） |
| generator | historical (H-D) | canonical-clean-v1 |
| Z 標準化 | あり | なし |
| Poisson clip | あり（`[−20,10]`） | なし |
| Y 学習 | 全ペア | 約 80%、20% holdout |
| criterion | Q ベース（Q2） | held-out S1 |
| starts | 1 | 2 |
| trials / replicates | 5 | 8 |
| 候補 K | 1..10 | 1..7 |
| inference lineage | historical 系列 | objective-consistent 系列 |

### まだ切り分けていないもの

- ⬜ **signal** 強度
- ⬜ **generator** の差（Z 標準化・clip・行正規化）
- ⬜ **criterion** の差（Q ベース vs held-out）
- ⬜ **Y holdout** の有無
- ⬜ **inference lineage** の差

---

## 📉 RMSE は何を見ていたか

| 指標 | 見ているもの |
|---|---|
| `RMSE_Z` | 潜在構造の回復 |
| `RMSE_F` | loading の回復 |
| `RMSE_X` | 属性の再構成 |
| `RMSE_Y` | 関係の平均構造の回復 |

> 🔴 **RMSE が良いことと、正しい K を選択できることは別。**

昔の n-sweep（fixed 系列 Exp2）は **`K_TRUE=3` を固定**して `n = 50 → 300` を振り、**parameter recovery（`RMSE_Z`）が改善すること**を見た実験。3 シナリオで `n=50 → 300` の相対改善は約 41〜59%。

**これは K selection experiment とは別物なので、混同しない。**

---

## 🧑‍🏫 先生からの指摘に対してどこまで答えられたか

| 先生の指摘 | 今回分かったこと | まだ未解決 |
|---|---|---|
| **真の K をどう出すか** | K の意味を 4 種類に整理。Poisson-X では条件付き識別可能 | Bernoulli-Y 一般、criterion consistency |
| **BIC で本当にいいか** | historical BIC の実体（Q2）を確認 | standard BIC consistency |
| **生成モデルとして成立するか** | canonical model は 9 通りすべて proper | moment existence は別問題 |
| **真のモデルをどう当てるか** | まず identifiability が必要と整理 | 全 family 一般では未解決 |
| **sparse / missing Y** | held-out 予測として利用可能 | `K_PRED` と `K*` の関係 |
| **n → ∞** | 漸近設定を先に決める必要があると整理 | consistency 未証明 |

---

## ♾️ データ数が増えたとき

`P(K̂ = K*) → 1` のような consistency を議論するには、**まず何を無限大にするのかを決める必要がある。**

| レジーム | 内容 |
|---|---|
| A | `n → ∞`、`d`・`K` fixed |
| B | `n → ∞`、Y が dense |
| C | `n → ∞`、Y が sparse |
| D | `d` も増える |

> ⚠️ **`Y` は `O(n²)` ペアあるが、`z_i` を共有するため `n²` 個の iid observation ではない。**
> `Y_ij` と `Y_ik` は独立ではないので、標準的な漸近論をそのまま当てはめられない。

**現在: UNRESOLVED。**

---

## 🔗 sparse / missing relation への応用

| | 意味 |
|---|---|
| **sparse** | 関係は観測済みだが 0 が多い |
| **missing** | 関係自体が未観測 |

**missing なら、こういう応用が考えられる:**

```
observed X, Y
      ↓
    Z を推定
      ↓
 未観測 Y_ij を予測
```

current の held-out S1 はこの方向に近い形になっている。

> ⚠️ ただし **`K_PRED = K*` は保証されていない。**

---

## ✅ 分かったこと

- ✅ canonical 生成モデルは **9 通りの family 組合せすべてで proper**
- ✅ 「真の K」には **4 つの意味**があり、`K^rank ≤ K* ≤ K_GEN`
- ✅ **Poisson-X では条件付きで `K* = K_GEN`** を導ける
- ✅ historical の「BIC」の実体は **Q2** で、standard Schwarz BIC（Q3）ではない
- ✅ **昔の K 選択成功は本物**（fixed 30/30、historical 5/5）
- ✅ **有限標本では sample size が K 選択に影響する**（historical 条件でも 5/5 → 1/5）
- ✅ **8 → 40 反復だけでは under-selection は解決しない**
- ✅ **「Poisson だから選べない」は否定される**

## ❓ まだ分からないこと

- ❓ **Bernoulli-Y を含む joint model の一般的な K identifiability**（主要実験で使う family）
- ❓ **standard BIC / その他 criterion の consistency**
- ❓ **`K*` と `K_PRED` の関係**（現行 S1 の population limit）
- ❓ **`n → ∞` の asymptotic regime**（何を無限大にするかが未確定）
- ❓ **OLD/CURRENT 差の主要因**（signal / generator / criterion / holdout のどれか）

---

## 💬 次回先生に相談したいこと

1. **修論では、`K_GEN` / `K*` / `K_PRED` のどれを中心に議論すべきか。**
   理論上は `K*` が自然だが、`K*` は M-closed 前提で実データでは存在しないこともある。実応用重視なら `K_PRED` も要検討。

2. **Bernoulli-Y の identifiability を証明まで追うべきか。**
   それとも、条件付きで理論が成立する範囲（Poisson-X・Gaussian-Y）を明示し、Bernoulli-Y は未解決として有限標本実験で扱う形で十分か。

3. **K 選択基準は、standard BIC の理論を追うか、predictive criterion を中心にするか、有限標本比較に留めるか。**

4. **次の実験で切り分けるなら、signal / generator / criterion / Y holdout のどれを優先すべきか。**
   `n=75` での食い違い（4/5 対 0/8）が最大の手掛かり。

---

## 🛣️ 先生との相談後に考える候補

- [ ] Bernoulli-Y identifiability の理論検討
- [ ] signal sensitivity experiment
- [ ] generator bridge
- [ ] Y holdout sensitivity
- [ ] criterion comparison
- [ ] sparse / missing Y prediction experiment
- [ ] asymptotic regime の定式化

> 🛑 **どれをやるかは先生との相談後に決める。** いま勝手に着手しない。

---

## 📍 現在地

指数型分布族への拡張自体は、生成モデルとして成立していることを確認できた。
K 選択については、過去の成功が実在したことも確認できた一方で、有限標本では条件依存性が強いことが分かった。
Poisson-X では条件付きで真の K を理論的に意味づけられるが、**Bernoulli-Y の identifiability や BIC consistency など、重要な未解決問題が残っている。**
次は「修論でどこまで理論を扱うか」を先生と決める段階であり、実験を増やす段階ではない。

---

## 📎 詳細資料

| 内容 | ファイル |
|---|---|
| 理論詳細（証明・条件・claim ledger） | `reports/k_selection_theory/true_k_identifiability_and_selection_theory_20260906.md` |
| 先生向け相談メモ | `reports/advisor/advisor_k_selection_discussion_20260906.md` |
| 実験記録（provenance） | `EXPERIMENT_REGISTRY.md` |
| 研究内容の正本 | `RESEARCH_MASTER.md` |
| 事故台帳・禁止表現 | `KNOWN_ISSUES.md` |
