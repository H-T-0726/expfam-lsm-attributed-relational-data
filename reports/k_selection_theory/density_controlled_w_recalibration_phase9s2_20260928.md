# Phase 9S2 — 厳密な密度による 1 次元の再較正（2026-09-28）

- 位置づけ: **ZERO-EM NUMERICAL RECALIBRATION ONLY**（Issue #112）。EM・refit・family 選択・K 選択なし。
- 結果: `expfam/results/density_controlled_w_recalibration/phase9s2_20260928/`
  （`density_checks.json`, `primary_adaptive_convergence.csv`, `fixed_node_crosscheck.csv/json`, `calibration.json`, `independent_mc_crosscheck.json`, `future_protocol.json`）
- 実行コード: `248b11a`（`expfam/src/experimental/density_w_recalibration.py`、実行時 clean）。数値の設定はすべて実行前にコードで固定し、1 回だけ実行した。
  Phase 9S の artifact は変更していない。
- 系列 E（experimental prototype; 本文採用不可）

---

## 1. 問い

Phase 9S の推定対象 `p̄(b, w) = E[σ(b + wS)]`（S = z_iᵀz_j、K = 3）の較正を、主の積分経路だけを S の厳密な密度による 1 次元積分に置き換えて、
**Phase 9S と同じ 1e-12 の基準**で固定できるか。推定対象・target・w・根の探し方は Phase 9S から変えていない。

## 2. 厳密な密度（DERIVED）

S = Σ_{k=1}^{K} X_k Y_k（X_k, Y_k は独立な標準正規）とする。

1. 1 つの積について、Y を条件づけると tXY | X ~ N(0, t²X²) なので
   `E[e^{itXY}] = E_X[e^{−t²X²/2}] = (1 + t²)^{−1/2}`（X² ~ χ²_1 の積率母関数 `E[e^{−uX²}] = (1+2u)^{−1/2}` で u = t²/2）。
2. K 個の項は独立なので `φ_S(t) = (1 + t²)^{−K/2}`。
3. 逆 Fourier 変換は、Basset の積分 `K_ν(x) = Γ(ν+½)(2x)^ν / √π · ∫_0^∞ cos(t) / (t² + x²)^{ν+½} dt` を ν = (K−1)/2 で使うと
   `f_K(s) = 1/(√π Γ(K/2)) · (|s|/2)^{(K−1)/2} · K_{(K−1)/2}(|s|)`。
4. K = 3 では Γ(3/2) = √π/2 より `f_S(s) = 1/(√π · √π/2) · (|s|/2) · K_1(|s|) = |s| K_1(|s|) / π`。
5. 対称性から
   `p̄(b, w) = (1/π) ∫_0^∞ [σ(b + ws) + σ(b − ws)] · s K_1(s) ds`。
   s → 0+ で `s K_1(s) → 1`（K_1(s) ~ 1/s）なので、s = 0 では 1 と明示的に置き、それ以外では `s · k1e(s) · e^{−s}` で計算した。

一意性・存在（p̄ は b について連続で狭義単調増加、極限 0 と 1）は Phase 9S の監査のとおり。

## 3. 密度の数値確認（較正とは独立）

| 量 | 数値 | 厳密値 | 誤差 |
|---|---|---|---|
| 正規化 ∫ f_S | 1 | 1 | 2.2e-16 |
| 平均 | 0（対称性から厳密） | 0 | — |
| 2 次の積率 E[S²] | 3 | K = 3 | 1.3e-15 |
| 4 次の積率 E[S⁴] | 45 | 3K(K+2) = 45 | 0.0 |
| 特性関数 t = 0.5 / 1 / 2 | — | (1+t²)^{−3/2} | 8.3e-14 / 9.9e-14 / 5.7e-14 |

事前に決めた許容値（1e-11）を満たした。

## 4. 主の較正（adaptive QUADPACK、実行前に固定）

設定: A（epsabs = epsrel = 1e-10）、B（1e-12）、C（2e-13）、subdivision の上限 200。根は Phase 9S と同じ brentq と bracket。

| 設定 | p_target | w0 weak | w0 strong | 最大の abserr 推定 | IntegrationWarning |
|---|---|---|---|---|---|
| A | 0.33140621213148430 | −0.87809944053935110 | −1.1890648128922830 | 6.8e-11 | 0 |
| B | 0.33140621213146787 | −0.87809944053934290 | −1.1890648128923007 | 6.2e-13 | 0 |
| C | 0.33140621213146765 | −0.87809944053934260 | −1.1890648128923011 | 2.0e-13 | 0 |

- baseline の w0 はすべての設定で厳密に −1（復元誤差 0）。
- 隣り合う設定の差: A→B は p_target 1.6e-14・w0 weak 8.2e-15・w0 strong 1.8e-14、**B→C は p_target 2.2e-16・w0 weak 2.2e-16・w0 strong 4.4e-16**。
- C での根の残差: weak 5.6e-17、baseline 0、strong 0。
- **固定の規則（B と C の差がすべて ≤ 1e-12、C での残差 ≤ 1e-12、密度の確認に合格）: PASS**。本番の値は設定 C。

## 5. 独立の決定的な cross-check（固定点の Gauss–Legendre）

写像 s = x/(1−x) の後、[0,1) 上の固定点 Gauss–Legendre（256, 512, 1024, 2048 点）。

| 点数 | p_target | w0 weak | w0 strong |
|---|---|---|---|
| 256 | 0.33140621213146360 | −0.8780994405393432 | −1.1890648128922991 |
| 512 | 0.33140621213147525 | −0.8780994405393409 | −1.1890648128923025 |
| 1024 | 0.33140621213147214 | −0.8780994405393419 | −1.1890648128923020 |
| 2048 | 0.33140621213153970 | −0.8780994405393294 | −1.1890648128923251 |

- 最も細かい 2 つ（1024 と 2048）の差: p_target 6.8e-14、w0 weak 1.2e-14、w0 strong 2.3e-14。
- 2048 点と主の本番値の差: p_target 7.2e-14、w0 weak 1.3e-14、w0 strong 2.4e-14。
- **cross-check: PASS**（いずれも ≤ 1e-12）。2 つの方法を平均したり、都合の良い方を選んだりはしていない。

## 6. 較正値（固定）

| 条件 | w | w0 | p̄(w0, w) − p_target |
|---|---|---|---|
| weak | 1/√2 | **−0.8780994405393426** | 5.6e-17 |
| baseline | 1 | **−1**（定義） | 0 |
| strong | √2 | **−1.1890648128923011** | 0 |

**p_target = 0.33140621213146765**。

### Phase 9S の暫定値との比較（記述のみ、較正には使っていない）

差（Phase 9S2 − Phase 9S の最も細かい tensor 求積）: p_target −6.8e-14、w0 weak −3.8e-13、w0 strong **−3.0e-11**。
Phase 9S で strong の収束が最も遅かったことと整合する（tensor 求積の値は strong で約 3e-11 ずれていた）。

## 7. Monte Carlo の監査（両方の決定的な経路が PASS した後に実行）

固定 seed 20260928、1e7 サンプル、潜在事前分布から直接生成。

| 条件 | MC の推定値 | 標準誤差 | 決定的な値 − MC（SE 単位） |
|---|---|---|---|
| weak | 0.331424 | 6.6e-5 | −0.28 |
| baseline | 0.331437 | 8.1e-5 | −0.38 |
| strong | 0.331454 | 9.5e-5 | −0.50 |

いずれも 1 SE 以内。MC は本番の値を決めても変えてもいない。

## 8. 将来の Phase 9T protocol

`future_protocol.json`: **FROZEN_NOT_EXECUTED**（`execution_authorized: false`）。
Phase 9P の protocol から変わるのは各条件の `w0` と stage だけ（w は Phase 9P の値のまま）。baseline（−1, 1）は Phase 9K の記録を再利用し、再実行しない。
新しい EM の上限は weak + strong で 400。Z/F/X の同一性の preflight、Y の文脈（真の平均確率、実現 edge 密度、η の平均・SD、確率の分位、飽和の割合）を必須とした。

**制約**: 揃えるのは**母集団の平均 Bernoulli edge 確率だけ**である。有限標本の実現 edge 密度、確率の分散・分位・飽和、確率の分布全体は w によって違う。
将来の実験は「mean-density-controlled relational-w sensitivity」と呼び、純粋な信号の効果とは呼ばない。

## 9. DECISION

## **DECISION: DENSITY_CONTROLLED_W_CALIBRATION_READY**

主の 1 次元積分が変更していない 1e-12 の規則を満たし、独立の決定的な cross-check も同じ基準で一致し、baseline と密度の確認も合格した。

## 10. CAN SAY / CANNOT SAY

**CAN SAY**
- K = 3 の潜在事前分布のもとで、母集団の平均 edge 確率を baseline（0.33140621213146765）に合わせる固定の w0 は、weak で −0.8780994405393426、strong で −1.1890648128923011 であり、
  2 つの独立な決定的な積分経路で 1e-12 より細かく一致した。

**CANNOT SAY**
- 純粋な信号の強さの効果の分離、確率の分布・飽和・実現密度の一致。
- K 選択・改善・優越性に関すること（本 Phase では K 選択をしていない）。

## 11. 最大の UNRESOLVED

較正そのものは固定できた。残るのは、平均だけを揃えても、確率の分布の形（分散・飽和）が w によって変わることであり、Phase 9T の結果の解釈ではこれを区別する必要がある。
