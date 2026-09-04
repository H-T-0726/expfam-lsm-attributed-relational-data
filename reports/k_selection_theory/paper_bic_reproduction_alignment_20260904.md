# 原論文 BIC・MATLAB・Python 再現・現行 ExpFam の対応監査

**作成日:** 2026-09-04  
**種別:** READ-ONLY / documentation-only audit  
**baseline main:** `818d17f8a05d8c8cf389681e61401ae634e66e36`  
**実験実行:** なし  
**コード変更:** なし

---

## 0. 目的

潜在次元 `K` 選択について、次の 4 系統で「BIC の当てはまり項に何を入れているか」を混同しないために記録する。

1. Mikawa et al. (2024) の印刷された原論文 Eq.(26)
2. 手元 MATLAB 実装
3. `reproduction/src/experiment_paper_2.py`
4. 現行 ExpFam の `calc_bic_dual`

本記録は、**どれか 1 つを「正しい BIC」と認定するものではない**。特に、原論文が `BIC` と呼んでいることと、それが standard Schwarz BIC の観測データ周辺尤度に対応することは別問題として扱う。

---

## 1. 用語固定

潜在変数 `Z` の扱いが異なる 3 つの量を区別する。

```text
Q1: conditional / plug-in likelihood
    ln p(X, Y | Z, θ)

Q2: complete joint log density
    ln p(Z, X, Y | θ)
    = ln p(Z|θ) + ln p(X|Z,θ) + ln p(Y|Z,θ)

Q3: observed-data marginal likelihood
    ln p(X, Y | θ)
    = ln ∫ p(Z, X, Y | θ) dZ
```

既存の `reports/k_selection_theory/k_selection_theory_audit_20260822.md` の用語を踏襲する。

---

## 2. 結論表

| 系統 | BIC / score の当てはまり項 | `p(Z)` | MC 平均 | 現時点の位置づけ |
|---|---|---:|---:|---|
| 原論文 Eq.(26) | Eq.(16) の `ln L` = Q1 | なし | 本文から評価手続きは特定不能 | 論文上は `BIC` と定義。ただし Q3 ではない |
| MATLAB `calcdescmetric_ver4.m` | `ln p(X|Z)+ln p(Y|Z)+ln p(Z)` の平均 = Q2 の MC 平均 | **あり** | **あり** | Q 関数を計算していることは確認済み。これが paper Exp.2 の Eq.(26) に実際に代入されたかは未確定 |
| Python reproduction `experiment_paper_2.py` | 1 個の `Z` に対する `ln p(Z)+ln p(X|Z)+ln p(Y|Z)` = Q2 plug-in | **あり** | **なし** | コメント上は Eq.(26) 再現だが、当てはまり項は Eq.(16) と一致しない |
| 現行 `calc_bic_dual` | `Q_strict` = Q2 の MC 近似（Poisson の定数補正を含む） | **あり** | **あり** | legacy Q-based complete-data criterion。Schwarz BIC と呼ばない |
| standard Schwarz BIC | Q3 | `Z` を積分 | — | 上記 4 系統とは別 |

---

## 3. 原論文 Eq.(26)

### 3.1 [SUPPORTED_BY_PRIMARY_SOURCE] 原論文は `BIC` と記載している

Mikawa et al. (2024) の Eq.(26) は

```text
BIC = -2 ln L + [(k+1)d - k(k-1)/2] ln n
```

という形で記載され、潜在次元選択に用いられている。

したがって、

> 「先行研究では BIC を用いて潜在次元を選択した」

という記述自体は原論文の記載と整合する。

### 3.2 [SUPPORTED_BY_PRIMARY_SOURCE] Eq.(26) の `ln L` は Eq.(16)

原論文の Eq.(16) の尤度は `Z` を積分せず、`Z` を与えた条件付きの

```text
ln p(X, Y | Z, θ)
```

に相当する。Eq.(16) には `p(Z|θ)` は入っていない。

一方、原論文 Eq.(18) の Q 関数には `ln p(Z^(l)|θ)` が入る。

よって、印刷された原論文では

```text
Eq.(26) の ln L ≠ Eq.(18) の Q
```

である。

### 3.3 [DERIVED] standard Schwarz BIC と同一とは断定しない

standard Schwarz BIC が通常対象とするのは観測データ周辺尤度

```text
ln p(X,Y|θ) = ln ∫ p(Z,X,Y|θ) dZ
```

である。

原論文 Eq.(16) は `Z` を積分していないため、

> 「原論文 Eq.(26) は standard Schwarz BIC そのものである」

とは記載しない。

原論文著者が Eq.(16) の `ln L` をどの `Z` で評価したか（最終 `Z` の plug-in、サンプル平均、その他）は、本文からは特定できないため **UNRESOLVED** とする。

---

## 4. MATLAB 実装

対象:

- `Mato Lab Program/calcdescmetric_ver4.m`
- `Mato Lab Program/DecideNumFactor.m`
- `Mato Lab Program/NOLTA_exp_ver3_revise_batch.m`

### 4.1 [CONFIRMED_IN_REPOSITORY] `calcdescmetric_ver4.m` の Q は `p(Z)` を含む

`calcdescmetric_ver4.m` は各 Monte Carlo sample `l` について

```matlab
Q = Q + calcp_X(...)
      + calcp_Y(...)
      + calcp_Z(...);
```

とし、最後に `Q / params.L` を記録する。

したがってここで計算されている量は

```text
(1/L) Σ_l [ln p(X|Z^(l)) + ln p(Y|Z^(l)) + ln p(Z^(l))]
```

であり、Q2 の MC 平均に対応する。

### 4.2 [UNRESOLVED] この Q が paper Experiment 2 の Eq.(26) に実際に代入されたかは未確定

現在の `Mato Lab Program/` に存在するファイルからは、paper Experiment 2 の最終 BIC 計算経路を一意に復元できない。

したがって、

> 「元 MATLAB は paper Eq.(26) に Q を代入していた」

とは断定しない。

### 4.3 `DecideNumFactor.m` は別系統

`DecideNumFactor.m` は MATLAB `factoran` の `stats.loglike` を用いて X の因子数を選択する処理であり、X+Y joint LSM の paper Experiment 2 の BIC 実装だとみなさない。

---

## 5. Python reproduction `experiment_paper_2.py`

対象:

- `reproduction/src/experiment_paper_2.py`

### 5.1 [CONFIRMED_IN_REPOSITORY] `calc_BIC()` 自体は Eq.(26) の形

```python
num_params = (k + 1) * d - k * (k - 1) / 2
BIC = -2 * log_likelihood + num_params * np.log(n)
```

と実装されている。

ペナルティ項の形は原論文 Eq.(26) と一致する。

### 5.2 [CONFIRMED_IN_REPOSITORY] ただし `calc_log_likelihood()` は Eq.(16) ではない

同ファイルの `calc_log_likelihood()` は

```text
ln p(Z) + ln p(X|Z) + ln p(Y|Z)
```

を返す。

つまり、関数名は `log_likelihood` だが実体は Q2 の complete joint plug-in であり、原論文 Eq.(16) の Q1 とは一致しない。

### 5.3 [CONFIRMED_IN_REPOSITORY] MC 平均でもない

`experiment_paper_2.py` は `Z_samples` 全体に対して Q の MC 平均を計算するのではなく、各 EM iteration の最後の `Z` に対して `calc_log_likelihood()` を評価し、反復中の最大値を `best_log_likelihood` として BIC に用いる。

したがって、

```text
Python reproduction の paper Exp.2 score
≠ 原論文 Eq.(16) を文字通り用いた Eq.(26)
≠ 現行 calc_bic_dual の Q_strict MC average
```

である。

### 5.4 current claim

`reproduction/src/experiment_paper_2.py` は、コメント上は paper Experiment 2 / Eq.(26) の再現を意図しているが、**当てはまり項の実装は原論文 Eq.(16) の literal reproduction ではない**。

これは historical reproduction artifact の provenance に関する指摘であり、既存結果を削除・上書きしない。

---

## 6. 現行 ExpFam `calc_bic_dual`

対象:

- `expfam/src/utils_expfam.py`

### 6.1 [CONFIRMED_IN_REPOSITORY] `Q_strict` は `p(Z)` を含む

`calc_Q_dual()` は

```text
Q = (1/L) Σ_l [ln p(Z_l)
               + ln p(X|Z_l)
               + ln p(Y|Z_l)]
```

を計算する。

`calc_Q_dual_strict()` はそこへ Poisson の factorial correction を加える。

### 6.2 [CONFIRMED_IN_REPOSITORY] `calc_bic_dual()`

```text
score = -2 * Q_strict + num_params * ln(n)
```

である。

Gaussian-X / Bernoulli-Y では

```text
num_params
= k*d - k(k-1)/2 + d
= (k+1)d - k(k-1)/2
```

となるため、**K に依存するペナルティ部分は原論文 Eq.(26) と一致する**。

一方、当てはまり項は原論文 Eq.(16) ではなく Q2 の MC 近似である。

したがって current guidance は従来どおり、

> `calc_bic_dual` の値を standard Schwarz BIC と呼ばず、legacy Q-based complete-data criterion として扱う。

とする。

---

## 7. 研究上の説明ルール

### 7.1 使用可

> 先行研究では BIC と呼ぶ基準を用いて潜在次元を選択している。

> ただし、原論文 Eq.(26)、Python 再現、現行 ExpFam では当てはまり項の定義が一致していない。

> 現行 ExpFam の historical `BIC` は `Q_strict` を用いる Q-based criterion であり、standard Schwarz BIC と同一視しない。

> Phase 7e / 8b の K 選択では、この legacy Q-based criterion ではなく frozen held-out predictive score を用いている。

### 7.2 使用禁止 / 要限定

次は無限定に書かない。

- 「原論文 Eq.(26) は standard Schwarz BIC である」
- 「原論文 Eq.(26) は Q 関数を用いている」
- 「Python reproduction は原論文 Eq.(26) を完全に忠実再現している」
- 「MATLAB では paper Experiment 2 に Q を代入したことが確認済み」
- 「現行 `calc_bic_dual` と原論文 Eq.(26) は同じ BIC である」
- 「Phase 7e / 8b は Q-based BIC で K を選択した」

---

## 8. 修論での推奨説明

短く書く場合:

> 先行研究では BIC と呼ばれる基準により潜在次元を選択している。一方、手元の再現実装および本研究の旧来実装では、原論文 Eq.(26) の条件付き尤度とは異なり、潜在変数の事前分布項を含む Q 関数型の量が用いられていた。このため、本研究ではこれを standard Schwarz BIC と同一視せず、K 選択の追加検証では held-out 関係データに対する予測スコアを用いた。

より厳密に書く場合:

> Mikawa et al. (2024) の Eq.(26) は `BIC = -2 ln L + [(k+1)d-k(k-1)/2] ln n` と定義されているが、当てはまり項 `ln L` は Eq.(16) の `Z` に条件づけた尤度であり、`Z` を積分した観測データ周辺尤度ではない。また、Python 再現実装と本研究の legacy `calc_bic_dual` は `ln p(Z)` を含む complete-joint / Q-based quantity を用いており、原論文 Eq.(26) とも一致しない。したがって、本研究ではこれらの historical score を standard Schwarz BIC と呼ばず、Phase 7e / 8b では frozen held-out predictive score により K を評価した。

---

## 9. 既存文書との関係

本記録は次の既存監査を**置き換えず、具体的な reproduction alignment を追記する補助記録**である。

- `reports/k_selection_theory/k_selection_theory_audit_20260822.md`
- `KNOWN_ISSUES.md` KI-010 / KI-019
- `RESEARCH_MASTER.md` §12.6 / §12.7 / §12.9

特に KI-010 の結論

> 現行 legacy criterion を standard Schwarz BIC と呼ばない

は変更しない。

KI-019 の結論

> Phase 7e / 8b の held-out predictive score と legacy Q-based criterion を混同しない

も変更しない。

---

## 10. 今回新たに明示記録した点

今回の追加価値は次の 2 点である。

1. **`reproduction/src/experiment_paper_2.py` の `calc_log_likelihood()` が、原論文 Eq.(16) ではなく `ln p(Z)+ln p(X|Z)+ln p(Y|Z)` を使っていることを、paper Experiment 2 の再現性という観点から明示した。**
2. **MATLAB `calcdescmetric_ver4.m` が Q2 の MC 平均を計算する事実と、「それが paper Experiment 2 の Eq.(26) に実際に使われたかは現存コードだけでは確定できない」という未解決点を分離した。**

この 2 点は、今後「元論文 BIC」「MATLAB」「Python reproduction」「ExpFam legacy BIC」を同じものとして扱わないための provenance 記録とする。
