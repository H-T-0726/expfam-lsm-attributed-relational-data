# Phase 9J — Candidate B の Laplace 近似誤差の prospective 診断（K=2/3, 2026-09-28）

- 位置づけ: **PROSPECTIVELY FROZEN APPROXIMATION-ERROR CHARACTERIZATION**（Issue #92）。K 選択性能の研究ではない。
- 系列: **E（experimental prototype; 本文採用不可）**
- 一次データ: `expfam/results/laplace_is_diagnostic/phase9j_20260928/`
  （`is_records.json`＝fit ごとの Laplace と全 batch の診断、`is_summary.json`、`fitted_states.npz`、Phase 9E 形式の CSV）
- 実行コード: `368487a`（git_dirty = False）。1 回だけ実行。rerun / 追加サンプル / proposal 変更なし。

---

## 1. 研究上の問い

同じ条件の新しい 3 dataset で、fitted K=2 と K=3 について
(1) Laplace-Gaussian proposal による importance-sampling 補正 `c_K = log E_q exp(r)` はどの程度か、
(2) 補正は batch 間で安定しているか、(3) K=3 と K=2 の補正差は `delta_Lap_23` と比べてどの程度か。

恒等式: `log p(X,Y|θ,K) = laplace_log_observed + c_K`、`q = N(Ẑ, H⁻¹)`、`r = Φ(Z) − Φ(Ẑ) + ½ΔᵀHΔ`。

## 2. 何をしたか

- 条件は Phase 9I と同一（n=75, d=12, K_true=3, G×3/B×6/P×3, Bernoulli-Y, σ²_x=1, w0=−1, w=1, f_scale=√2, L=5, 8/8 反復, start_B のみ）。**K は 2 と 3 だけ**。
- seeds: rep01–rep03 = 991000+r / 992000+r / 993000+r。dataset は 1 回だけ生成し、K=2, 3 で共有。
- 各 refit: Phase 9H の評価関数（θ = refit の最終値、Z0 = refit の Z_est、grad_tol 1e-8、max_iter 200）→ status OK のときだけ IS 補正。
- IS: H の Cholesky（`Δ = solve(Lᵀ, ε)`、H⁻¹ は作らない）、**8 batch × 512 = 4096 samples / fit**、batch seed `995000 + 100·rep + 10·K + b`。log 空間で計算。
- `C_Lap` は Phase 9I と同じ（N = 75 は working convention、`d_K = Kd − K(K−1)/2 + n_gaussian_selected`）。補正はパラメータ項に触れない。
- zero-EM の単体テスト: Gaussian-X・w=0 の固定ケースでは Φ が 2 次なので r ≡ 0 となり、batch 補正は 1e-9 以内で 0（真値 0）。

## 3. 結果（OBSERVED）

### 実行と Laplace

| 項目 | 値 |
|---|---|
| 実 EM 実行 | **12 / 12** SUCCESS |
| retry / replacement / seed rescue | 0 / 0 / 0 |
| Laplace status | **6 / 6 OK**（停留 ≤ 3.6e-9、H 最小固有値 0.99–2.58） |
| IS の non-finite 評価 | 0 / 24,576 |
| selected assignment | 全 6 fit で G×3 / B×6 / P×3 |

### C_Lap と補正

| rep | C_Lap(2) | C_Lap(3) | delta_Lap_23 | c_2（pooled） | c_3（pooled） | delta_c_23 | delta_IS_23 = delta_Lap_23 − 2·delta_c_23 |
|---|---|---|---|---|---|---|---|
| rep01 | 5120.66 | 5060.78 | −59.88 | −0.369 | −0.894 | −0.525 | −58.83 |
| rep02 | 5257.07 | 5202.02 | −55.04 | −0.267 | −0.577 | −0.311 | −54.42 |
| rep03 | 5056.52 | 4864.56 | −191.96 | −0.456 | −0.185 | +0.271 | −192.50 |

### batch 間のばらつきと重みの診断

| rep | K | batch 補正 mean / SD | min / median / max | pooled ESS（相対） | pooled 最大重み比 | batch 相対 ESS の範囲 | batch 最大重み比の範囲 | r の min / median / max |
|---|---|---|---|---|---|---|---|---|
| rep01 | 2 | −0.382 / 0.170 | −0.559 / −0.437 / −0.136 | 178.4 (4.4%) | 0.038 | 2.6–10.0% | 0.08–0.25 | −9.7 / −2.1 / 4.7 |
| rep01 | 3 | −0.936 / 0.302 | −1.294 / −0.975 / −0.367 | 62.7 (1.5%) | 0.062 | 1.1–5.6% | 0.09–0.40 | −16.8 / −4.0 / 4.6 |
| rep02 | 2 | −0.289 / 0.228 | −0.689 / −0.257 / −0.055 | 215.1 (5.3%) | 0.034 | 2.8–18.8% | 0.04–0.25 | −10.4 / −1.8 / 4.7 |
| rep02 | 3 | −0.601 / 0.233 | −1.004 / −0.611 / −0.281 | 92.8 (2.3%) | 0.046 | 1.1–4.9% | 0.13–0.40 | −17.1 / −3.6 / 4.7 |
| rep03 | 2 | −0.463 / 0.132 | −0.593 / −0.501 / −0.285 | 490.7 (12.0%) | 0.012 | 9.2–20.7% | 0.04–0.09 | −9.4 / −1.6 / 3.4 |
| rep03 | 3 | −0.325 / 0.533 | −0.802 / −0.520 / **+0.696** | **17.0 (0.4%)** | **0.227** | **0.3**–9.4% | 0.07–**0.75** | −14.8 / −2.8 / 6.7 |

### batch ごとの delta_IS_23

| rep | min | median | max | < 0 | > 0 | = 0 |
|---|---|---|---|---|---|---|
| rep01 | −60.19 | −58.71 | −57.74 | 8 | 0 | 0 |
| rep02 | −55.02 | −54.52 | −53.59 | 8 | 0 | 0 |
| rep03 | −194.51 | −191.88 | −190.93 | 8 | 0 | 0 |

## 4. 観測されたこと

- **補正の推定値の大きさ**: pooled で −0.89 〜 −0.18（log 尤度単位）。criterion 単位（−2 倍）での K 間差 `2·|delta_c_23|` は 0.54 〜 1.05。
  これらの 3 dataset では `|delta_Lap_23|` が 55 〜 192 と大きく、batch ごとの delta_IS_23 の符号は 24 batch すべてで delta_Lap_23 と同じだった。
- **重みの退化（それ自体が結果）**: 相対 ESS は pooled で 0.4% 〜 12%、K=3 の方が K=2 より一貫して低い（1.5% vs 4.4%、2.3% vs 5.3%、0.4% vs 12.0%）。
  rep03・K=3 では pooled ESS 17 / 4096、1 batch で最大重み比 0.75、batch 補正の範囲が −0.80 〜 +0.70 と、**batch 間で符号まで変わった**。
- **r の分布**: median が −1.6 〜 −4.0 と負で、右裾（最大 3.4 〜 6.7）が一部のサンプルに重みを集中させている。
  Laplace-Gaussian proposal は、事後の形（非 2 次の落ち方と右裾）とかなりずれている（INTERPRETATION）。
- 事前の予想と違い、補正の推定値はすべてのケースで負だった（1 batch を除く）。ただし重みが退化しているため、この符号を信頼できる推定とは扱わない。

## 5. 言えること / 言えないこと

**言える**
- 同じ条件の新しい 3 frozen dataset で、fitted K=2/3 の事後に対する Laplace-Gaussian proposal の IS 補正の推定値は −0.89 〜 −0.18 で、
  K 間の補正差（criterion 単位 0.54–1.05）はこれらの dataset の `|delta_Lap_23|`（55–192）より 2 桁小さかった。
- 同時に、IS の重みは強く退化しており（相対 ESS 0.4–12%）、特に K=3 で退化が強かった。

**言えない**
- **厳密な周辺尤度が分かった**とは言えない。ESS が低いので、IS 推定そのものに大きな誤差がありうる。特に重い右裾を取り逃すと補正を過小評価しうる。
- **Laplace 誤差の一般的な上界**は言えない。
- **Phase 9I rep01 の差 1.53 の誤差が分かった**とは言えない（別の dataset・別の θ̂）。
  ただし本 pilot の補正差の推定値（最大 1.05）は、1.53 と**同じ桁**であることは記録しておく。
- Candidate B が優れている、一致性がある、K 回復率、のいずれも言えない。
- Phase 9E/9I の頻度と混ぜない（別の dataset・別の K grid）。

## 6. fitted state の保存

`fitted_states.npz`: 6 refit それぞれの F、Gaussian 分散（sigma 対角）、w0、w、var_z、最終 Z_est、family assignment、seeds、
X と Y の配列（`X`, `Y` を保存。data seed から `run_family_selection_pilot.build_dataset` でも再生成できる）、code SHA `368487a`、evaluator / IS version を含む。
今後の事後解析はこの state を使えば、EM を再実行せずにできる。

## 7. 次の Human 判断

1. 重みの退化が強い（特に K=3）ので、Laplace-Gaussian proposal での IS は補正の信頼できる推定になっていない。
   補正を信頼できる精度で知る必要があるか（研究判断が変わる場合に限る）。
2. 必要なら、保存した fitted state 上で、より事後に合う proposal（例: 重い裾を持つ分布、事後の局所的な形の調整）や別の周辺尤度推定法を、事前固定の protocol で比較する。
   本タスクではいずれも行っていない（proposal 変更・追加サンプルは禁止）。
3. これらの 3 dataset では `|delta_Lap_23|` が補正差より十分大きかったが、Phase 9I rep01 のような小さな差の解釈にはこの結果を直接使えない。
