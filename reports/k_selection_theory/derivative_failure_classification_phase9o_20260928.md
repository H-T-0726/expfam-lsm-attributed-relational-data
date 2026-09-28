# Phase 9O — Phase 9N の DERIVATIVE_UNAVAILABLE 3 endpoint の失敗分類（2026-09-28）

- 位置づけ: **FAILURE-CLASSIFICATION DIAGNOSTIC ONLY**（Issue #102）。θ 更新・最適化ステップ・EM・refit・family 選択・新しいデータなし。
- 対象: Phase 9N の終点（`expfam/results/multistep_local/phase9n_20260928/final_theta.json`）のうち rep04 K=2（4 ステップ）、rep09 K=3（14）、rep09 K=4（6）の 3 状態だけ。
- 結果: `expfam/results/derivative_failure_classification/phase9o_20260928/`
  （`protocol.json`＝source の SHA-256 と設定、`perturbation_classification.csv`＝状態 × 座標 × 側 の 1 行ずつ、`state_failure_summary.csv/json`）
- 実行コード: `5f3bc24`（`expfam/src/experimental/derivative_failure_classification.py`、実行時 clean）。1 回だけ実行。Phase 9K・9M・9N の artifact は変更していない。
- 系列 E（experimental prototype; 本文採用不可）

---

## 1. 研究上の問い

Phase 9N で DERIVATIVE_UNAVAILABLE になった 3 終点で、どの θ 座標の、+h と −h のどちら側で、Candidate B の評価がどの理由で OK にならなかったか。原因の分類だけを行う。

## 2. 何をしたか

- **artifact の照合**: 3 状態それぞれで、`final_theta.json`・`per_state_final.csv`・`trajectory.csv` の status（DERIVATIVE_UNAVAILABLE）、
  採用ステップ数（4 / 14 / 6、trajectory の行数とも一致）、`final_ell`（trajectory の最後の `ell_Lap` とも一致）、reason（`1 gradient component(s) unavailable`）、
  seeds と family 割当（Phase 9K の `fitted_states.json` と一致）を確認した。
- **終点の再構成**: data seed から X, Y を再生成し、`final_theta.json` の F・分散・w0・w・var_z と最後の Z mode を復元。
  終点を 1 回評価し、**3/3 で status OK、`ell_Lap` は Phase 9N の `final_ell` と差 0.0 で一致**。水平次元は `Kd − K(K−1)/2`（23 / 33 / 42）と一致。
- **stencil**: 終点の F から同じ `split_basis` で chart を作り、Phase 9N と同じ座標順（F_horizontal[j] → log_sigma[column=l] → w0 → w）で、
  Phase 9N と同じ step（`SavedState.step_sizes()/2`: F と対数分散 5e-5、w0・w は `5e-5·max(1,|終点値|)`）を使った。
  各座標で +h と −h を 1 回ずつ評価し、**すべて保存された最後の Z mode から開始**した（restart・ridge・jitter・許容値変更・再試行なし）。
- **結果の保存**: 各評価の構造化結果（status, log_observed が有限か, phi_at_mode, grad_inf, mode の反復数, H の最小固有値, logdet の符号と値, notes）をそのまま保存した。例外は発生しなかった。

## 3. 結果

| 状態 | 座標数 | 失敗した座標数 | 失敗した側の数 | 座標 | 側 | 分類 | grad_inf | mode 反復 | H の最小固有値 | logdet 符号 |
|---|---|---|---|---|---|---|---|---|---|---|
| rep04 K=2 | 28 | **1** | 1 | `w`（index 27） | **−** | A: NOT_STATIONARY | 1.031e-8 | 200（上限） | 1.493 | +1 |
| rep09 K=3 | 38 | **1** | 1 | `w`（index 37） | **+** | A: NOT_STATIONARY | 1.482e-8 | 200（上限） | 0.731 | +1 |
| rep09 K=4 | 47 | **1** | 1 | `w`（index 46） | **+** | A: NOT_STATIONARY | 1.389e-8 | 200（上限） | 0.504 | +1 |

- notes: `max|grad Phi|=1.031e-08 > 1.0e-08`（rep04 K=2）、`1.482e-08`（rep09 K=3）、`1.389e-08`（rep09 K=4）。
- 分類の内訳（全 226 評価）: OK 223、A: NOT_STATIONARY 3、B: HESSIAN_NOT_PD 0、C: EXCEPTION 0、D: NONFINITE_OTHER 0、E: OTHER_STRUCTURED_NON_OK 0。
- **Phase 9N の「1 gradient component unavailable」は 3 状態すべてで再現した**（失敗した座標 1、失敗した側 1）。
- 同じ `w` 座標の反対側は OK だった（mode 反復 2、grad_inf ≈ 4e-15、H の最小固有値は失敗側とほぼ同じ）。
- OK だった 223 評価の mode 反復は最大 3、grad_inf は最大 8.4e-9、H の最小固有値は最小 0.504。

## 4. DECISION

## **DECISION: FAILURE_MECHANISM_CLASSIFIED**

3 終点すべてで再構成に成功し、各状態で non-OK の摂動が 1 つ再現し、その座標・側・分類を記録できた。

## 5. 研究上の解釈（この固定評価の範囲で）

- 3 つの失敗はいずれも **片側だけ・1 座標だけ**で、座標はすべて **`w`**（Y 側の係数）だった。F・対数分散・w0 では失敗はなかった。
- 失敗の分類はすべて **NOT_STATIONARY** で、**H は正定値のまま**だった（最小固有値 0.50–1.49、logdet 符号 +1）。Hessian の正定値性の喪失や例外・非有限値ではない。
- 失敗した評価では、joint mode の solver が反復上限 200 に達し、そのときの `max|∇Φ|` は 1.03e-8 〜 1.48e-8 で、Phase 9H の許容値 1e-8 をわずかに上回っていた。
  同じ座標の反対側の評価は 2 反復で grad_inf ≈ 4e-15 に達していた。
- したがって、この固定点で観察されたのは「solver が 200 反復で許容値 1e-8 をわずかに満たせなかった」ことであり、H の退化を伴う失敗ではない。
  これ以上の原因（なぜその側だけか、solver の反復が止まった理由）は、この固定評価からは判断できない。

## 6. CAN SAY / CANNOT SAY

**CAN SAY**
- Phase 9N の保存された失敗終点で、固定した同一の有限差分 stencil を使うと、各状態で `w` 座標の片側（rep04 K=2 は −h、rep09 K=3・K=4 は +h）で、
  Candidate B が NOT_STATIONARY（反復上限 200、`max|∇Φ|` 1.03–1.48e-8 > 1e-8、H は正定値）を返す失敗が再現した。
- Phase 9N の「1 成分が利用不可」は、3 状態とも 1 座標・1 側の失敗として再現した。

**CANNOT SAY**
- この失敗がどの最適化手法でも避けられないこと。
- h を変える、許容値を変える、反復上限を変えると直ること。
- 終点が数学的な特異点にあること。
- 最適化を続けた場合の順序、rep04・rep09 の最終的な順序。
- MLE に関すること。

## 7. 最大の UNRESOLVED と次の Human 判断

**最大の UNRESOLVED**: 失敗した評価で joint-mode solver が許容値 1e-8 のわずか上で 200 反復止まっていた理由（例えば、damped Newton の更新が数値精度の限界で進まなくなったのか）は、
この固定評価では確認していない。また、この失敗によって unavailable になった rep04・rep09 の順序は判定できないままである。

**次の Human 判断**: これを実装の joint-mode solver の数値上の停止条件として扱うか（その場合の変更は評価の定義を変えるため frozen spec の変更になる）、
それとも Phase 9N の結果（3/5 retained、2/5 unavailable、flip 0）で局所最適化の問いを閉じるか。いずれも本 Issue の範囲外で、新しい Human Gate が必要。
