# Phase 9 K 選択研究の統合と claim ledger（Phase 9Z, 2026-09-29）

- 位置づけ: **SYNTHESIS / CLAIM-LEDGER INTEGRATION / ZERO-EXPERIMENT**（Issue #126）。mode は THEORY / AUDIT。
- **EM 0、refit 0、Candidate B の評価 0、θ の最適化 0、新しい dataset・感度実験・回帰・検定・信頼区間・新しい指標 0**。
  既存の merge 済み artifact と report を読んで照合し、転記しただけである。
- 数値はすべて merge 済みの一次 artifact（`expfam/results/...` の JSON / CSV）で確認した。Issue 本文だけから写した数値はない。
- 系列: Phase 9 の Candidate B・C_Q の比較・各感度実験はすべて **lineage E（experimental prototype）**。
- **MANUSCRIPT_ADOPTION_STATUS: NOT_DECIDED_BY_HUMAN**（本 Issue は採用を決めない。§14 は採用する場合の候補の表現にすぎない）。
- 本文中のラベル: FACT / DERIVED / OBSERVED / INTERPRETATION / HYPOTHESIS / DECISION / UNRESOLVED（`docs/research_operating_policy.md`）。

---

## 1. Phase 9 で何を問題にしたか

Dual-ExpFam LSM の潜在次元 K を選ぶ現在の基準（`calc_bic_dual` の `C_Q`）が、何を測っているのか、そして固定した人工データ条件でどの K を選ぶのか。
その結果として、観測データの周辺尤度を動機とする別の基準（Candidate B、`C_Lap`）を設計・実装し、同じ fit の上で C_Q と並べて、
ベースライン条件、Y 側の信号（relational w）、X 側の信号（attribute loading scale）、真の次元 K_true を変えたときの選択の振る舞いを記述した。
一致性や一般の recovery を示すことは目的にしていない（有限標本の特徴づけ）。

## 2. C_Q の正体

- **FACT**（`cq_decomposition_minimal_check_20260923.md`, #72 / PR #76; `run_joint_family_k_selection.cq_decomposition`）:
  現行 `C_Q = −2 Q_strict + p_K ln n = D_K + P_Z + P_θ`、`D_K = −2(Q_X + Q_Y)`、`P_Z = −2 Q_Z`、`P_θ = num_params · ln n`（MC サンプル L = 5 の平均）。
- **DERIVED**（`cq_theoretical_clarification_20260927.md`, #84 / PR #85）: var_z = 1 と `scale_Z`（潜在座標の平均二乗を 1 に揃える）のもとで `P_Z = nK(1 + ln 2π)` が厳密に成り立つ。
  n = 75 で 1 次元あたり 212.84（Phase 9Y で commit 済みの値からも 10/10 で確認）。
- **FACT / DERIVED**: C_Q は観測データの周辺尤度を使わないので **Schwarz BIC ではない**。Z の事前密度を含む完全データの Q 関数に基づく基準であり、
  構造的には「観測データ型 BIC + 2 × 潜在変数の事後エントロピー」と同型なので **Q-based complete-data / ICL-type criterion** と呼ぶ。
  ただし連続な Z の微分エントロピーは尺度に依存し、ICL の理論的性質はそのまま移らない（同 report）。
- 関数名 `calc_bic_dual`、CSV の列名 `BIC`、過去の結果の呼称は変更しない（CLAUDE.md §5、KI-010）。

## 3. Candidate B（C_Lap）の定義

- **DERIVED**（`k_true_criterion_design_comparison_20260928.md` §3, #86 / PR #87）: Z の同時最頻値 Ẑ と nK × nK の同時 Hessian H（relational の非対角ブロックを含む）で Z を Laplace 積分し、θ の積分に BIC 型の罰則を使う:

  ```
  C_Lap(K) = −2[ℓ_X(Ẑ) + ℓ_Y(Ẑ)] + ‖Ẑ‖² + ln|H(Ẑ)| + d_K ln N
  ```

- **FACT**（実装, `laplace_k_criterion.py`, #88 / PR #89）: `laplace_log_observed` が `ℓ_Lap = Φ(Ẑ) + (nK/2) ln 2π − ½ ln|H|` を計算し、`calc_C_Lap` が `C_Lap = −2 ℓ_Lap + d_K ln N` を返す（上の式と同値）。
  joint mode は減衰つき full Newton で求め、`max|∇Φ| ≤ 1e-8` と H の正定値を満たさなければ値を返さない（ridge / jitter なし）。
- **パラメータ数の規則**（`run_laplace_pilot.d_K`）: `d_K = K d − K(K−1)/2 + (選ばれた Gaussian-X の列数)`（w0・w は K によらないので数えない）。N = 75 は working convention。
- **DECISION**（Phase 9L, #96 / PR #97, `laplace_theta_penalty_theory_20260928.md`）: `RETAIN_N_EQ_n_QUALIFIED` — N = n を、正則性・非退化の仮定のもとで K に依存する F ブロックの主要な尺度として保持する。
  w の率、実装の Laplace と厳密な周辺尤度の橋渡しは UNRESOLVED。

## 4. 理論上分かったこと

- **DERIVED**: C_Q は Schwarz BIC ではなく ICL-type（§2）。
- **DERIVED**: Candidate B は観測データの evidence（Z を積分した量）の Laplace 近似を動機とし、K_true の回復という目標に合う（Phase 9G の比較。原論文型の条件付き基準 Candidate A は合わないと判断）。
- **DERIVED**（Phase 9L）: F の情報は O(n)（上界; Louis の恒等式）、θ の罰則の N = n は仮定つきで保持。K と K+1 の間には、w ≠ 0 では単純な入れ子の埋め込みがない（識別性の仮定のもと）。
- **UNRESOLVED**: 実装の ℓ_Lap と厳密な周辺尤度の差（Phase 9J で IS 補正の重みは強く退化し、厳密な値は得られなかった）、w の率、一致性。

## 5. 実装上確認できたこと

| phase | 何を確認したか | 一次の結果（OBSERVED / VERIFIED） |
|---|---|---|
| 9H（#88/#89） | Candidate B の評価関数の zero-EM 検証 | 勾配と同時 Hessian が有限差分と ~1e-9 で一致、Gaussian の厳密な場合と一致、非定常・非正定値は値を返さない |
| 9I（#90/#91） | 3 dataset の frozen pilot | 15/15 の refit で評価可能（OK）。C_Lap は 3/3 で K=3、C_Q は 1/3 で K=3 |
| 9J（#92/#93） | Laplace 近似の IS 診断（K=2, 3） | IS の相対 ESS は 0.4〜12%、K=3 で退化が強い。**厳密な周辺尤度は得られていない** |
| 9K（#94/#95） | 20 dataset の特徴づけ | 100/100 の refit で Candidate B が OK |
| 9M（#98） | 保存済み θ̂ の停留性と 1 ステップ診断（60 状態） | θ̂ は ℓ_Lap の停留点ではない。1 ステップで C_Lap は最大 2.81 動き、20/20 の最良 vs 次点の順序は保たれた（`ONE_STEP_LOCAL_ORDERING_STABLE`） |
| 9N（#100） | 最小 5 マージンでの 20 ステップ | 7 が上限到達（未収束）、3 が DERIVATIVE_UNAVAILABLE。反転 0、利用可能 3 ペア（`MULTISTEP_PILOT_INCONCLUSIVE`） |
| 9O（#102） | 9N の失敗の分類 | 3/3 が片側 1 座標の NOT_STATIONARY（solver が 200 反復で許容値 1e-8 をわずかに超えた、H は正定値） |
| 9Q / 9R（#106 / #108） | weak-w の小さな差の 1 ステップ / 20 ステップ | 1 ステップ: 5/5 保持。20 ステップ: 2 保持、3 unavailable、途中の反転 0（`WEAK_SMALL_GAP_MULTISTEP_INCONCLUSIVE`） |

- **INTERPRETATION**: 評価関数そのものは数式と一致し、本研究の規模（nK ≤ 375）で安定に評価できた。一方、MCEM 8 反復の θ̂ は ℓ_Lap の最適点ではなく、
  局所最適化を続けると joint-mode solver の許容値のわずか上で止まる失敗が一定の割合で起きる。これは C_Lap の値に最適化の誤差が含まれうることを意味する。

## 6. Phase 9K ベースライン（20 dataset、#94 / PR #95）

条件（FACT）: n = 75, d = 12, K_true = 3, X = Gaussian×3 / Bernoulli×6 / Poisson×3, Y = Bernoulli, w0 = −1, w = 1, f_scale = √2, L = 5, 探索・refit 8/8, 候補 K = 1..5, start_B のみ, seed 1001000+r ほか（r = 1..20）。
EM 200/200 成功、Candidate B 100/100 OK。

| 基準 | K1 | K2 | K3 | K4 | K5 | exact / under / over |
|---|---|---|---|---|---|---|
| C_Lap | 0 | 1 | 19 | 0 | 0 | 19 / 1 / 0 |
| C_Q | 1 | 5 | 14 | 0 | 0 | 14 / 6 / 0 |

- 同じ条件内の対応: 両方 K3 14、C_Lap だけ K3 5、C_Q だけ K3 0、どちらも K≠3 1（`paired_summary.json` P3）。最良 vs 次点の C_Lap の差は 22.0〜58.4。
- **注**: これは 20 dataset 版である。後の Phase 9X の K_true = 3 anchor は、この rep01..rep10 の部分集合（C_Lap 10/10、C_Q 7/10）であり、20 dataset 版と混ぜない。
- **OBSERVED（参考）**: 同じ protocol・別の seed の組（971000+r）の Phase 9E（#81 / PR #83、C_Q のみ）では C_Q の K=3 は 17/20 だった。
  同じ条件でも seed の組によって C_Q の数は 14/20 と 17/20 のように違う。これは矛盾ではなく、条件ごとの数を一般の確率として読んではいけない理由の一つである。

## 7. Y の信号（relational w）の感度

**Phase 9P（#104、w0 = −1 固定の w-sensitivity）**: w = 1/√2, 1, √2（w²K = 1.5 / 3 / 6）、同じ 20 の seed と同じ Z/F/X。

| 条件 | C_Lap の K1..K5 | C_Q の K1..K5 |
|---|---|---|
| weak | 0/4/16/0/0 | 9/10/1/0/0 |
| baseline（9K） | 0/1/19/0/0 | 1/5/14/0/0 |
| strong | 0/0/20/0/0 | 0/1/19/0/0 |

w を変えると平均 edge 密度も変わった（median 0.306 / 0.333 / 0.352）。

**Phase 9S / 9S2 / 9T（#110 / #112 / #113、平均 edge 確率を揃えた w-sensitivity）**: 母集団の平均 edge 確率を baseline（0.33140621213146765）に揃える w0 を、厳密な密度による 1 次元積分で 1e-12 の基準で較正（weak −0.8780994405393426、strong −1.1890648128923011）。

| 条件 | C_Lap の K1..K5 | C_Q の K1..K5 |
|---|---|---|
| controlled weak | 0/3/17/0/0 | 8/11/1/0/0 |
| baseline（9K） | 0/1/19/0/0 | 1/5/14/0/0 |
| controlled strong | 0/0/20/0/0 | 0/1/19/0/0 |

- **OBSERVED**: 両方の設計で、w を弱めると K ≤ 2 が増え（C_Q で顕著）、強めると両基準とも K=3 に集まった。過大選択は 0。
- **INTERPRETATION（条件つき）**: この固定条件では、母集団の平均 edge 確率を揃えても w による選択のパターンはほぼ残った。平均密度の違いだけでは説明されなかった。
  ただし確率の分布の形（飽和）は揃えていないので、純粋な関係信号の効果とは呼ばない。controlled weak の rep04 では C_Lap の差が 0.20 と非常に小さかった（追加の診断はしていない）。

## 8. X の信号（attribute loading scale）の感度（#116 / #118）

Phase 9U で Z と実現した Y を条件間で共有する成分分離の対応生成器を作り（20/20 で Z・Q・Y が同一）、Phase 9V で f_scale = 1 / √2 / 2（平均 loading energy 0.25 / 0.50 / 1.00）を実行した。
historical な Phase 9K とは別のデータ系列（Z/F/X/Y の一致 0/20）なので base も新しく実行した。

| 条件 | C_Lap の K1..K5 | C_Q の K1..K5 | 不完全 |
|---|---|---|---|
| weak | 0/0/20/0/0 | 0/9/11/0/0 | 0 |
| base | 0/0/20/0/0 | 0/5/15/0/0 | 0 |
| strong（19） | 0/0/18/**1**/0 | 0/1/18/0/0 | **1**（rep04） |

- EM 600 計画、593 試行、592 成功。strong の rep04 は K=2 の探索 EM で推定中の Poisson の exp(η) が overflow し、不完全として記録（retry なし）。strong の rep06 で C_Lap が K=4 を選んだ（過大選択 1 件）。
- **OBSERVED**: loading energy を上げると C_Q の K=3 は 11 → 15 → 18（/19）、C_Lap はどの条件でもほぼ K=3。`Phase 9V DECISION: ATTRIBUTE_SCALE_SENSITIVITY_PARTIAL`。
- 純粋な属性の情報の効果とは呼ばない（f_scale は Gaussian の SNR、Bernoulli の飽和、Poisson の裾を同時に変える）。

## 9. K_true の感度（#120 / #122）

Phase 9W で、K_true = 1..4 について平均の X loading energy（0.5）、Y の自然パラメータの分散（3）、母集団の平均 edge 確率（Phase 9K の K3 baseline）を揃えた設計を凍結した（w0 は 1e-12 の基準で較正、K3 は Phase 9K と bit 単位で同じ）。
Phase 9X で rep01..rep10 について K_true = 1, 2, 4 を実行し（EM 300/300）、K_true = 3 は Phase 9K の rep01..rep10 を読み取りのみで使った。

| K_true | C_Lap の K1..K5 | C_Lap exact / under / over | C_Q の K1..K5 | C_Q exact / under / over |
|---|---|---|---|---|
| 1 | 10/0/0/0/0 | 10 / 0 / 0 | 10/0/0/0/0 | 10 / 0 / 0 |
| 2 | 0/10/0/0/0 | 10 / 0 / 0 | 1/9/0/0/0 | 9 / 1 / 0 |
| 3（9K rep01..10） | 0/0/10/0/0 | 10 / 0 / 0 | 1/2/7/0/0 | 7 / 3 / 0 |
| 4 | 0/0/2/8/0 | 8 / 2 / 0 | 1/5/3/1/0 | 1 / 9 / 0 |

- **OBSERVED**: 過大選択は両基準とも 0。過小選択は K_true が大きいほど増え、C_Q で顕著だった。K_true = 4 では C_Lap の最良 vs 次点の差も小さく（median 13.4、最小 1.13）、K3 との境界に近かった。
- 一般の recovery 確率とは呼ばない（各セル 10 dataset、1 つの固定条件）。

## 10. K3 → K4 の境界の分解（#124）

Phase 9Y で、Phase 9X の K_true = 4 の既存の fit について、commit 済みの成分だけから Δ34 = C(K=4) − C(K=3) を分解した（新しい fit なし）。10/10 で許容値 1e-10 以内（最大 1.6e-12）で再構成できた。

- **FACT / DERIVED（基準の算術）**:
  - C_Lap で K4 が K3 に勝つ条件: `fit_gain_Lap_34 > volume_increment_34 + 9 ln 75`（volume = Δ(‖Ẑ‖² + log|H|)、fit gain は joint mode でのデータの当てはまりの差）。
  - C_Q で K4 が K3 に勝つ条件: `fit_gain_Q_34 > 212.84 + 38.86 = 251.70`（P_Z の 1 次元の増分 + 9 ln 75、どちらも一定）。
- **OBSERVED**: 余裕（正なら K4 が良い）は C_Lap で 8/10 が正（median 13.4）、C_Q で 1/10 が正（median −87.1）。
  同じ fit で C_Q の fit gain（MC 平均、median 164.6）は C_Lap の fit gain（joint mode、median 254.8）より小さく、C_Lap の体積の増分（median 203.9）は C_Q の P_Z の増分より小さかった。
- **言えないこと**: 過小選択の原因。P_Z が原因であること、1 次元あたりの信号の希釈が原因であることは証明されていない。fit gain がその大きさである理由は分解からは分からない。

## 11. 全体として何が分かったか

1. **（FACT / DERIVED）** 現行の C_Q は Schwarz BIC ではなく Q-based complete-data / ICL-type の基準である。Z の事前の項 P_Z は、scale_Z のもとで 1 次元あたり一定（n(1 + ln 2π)）である。
2. **（FACT）** 観測データの evidence の Laplace 近似を動機とする Candidate B（C_Lap）を定義・実装し、評価関数が数式と一致すること、本研究の規模で評価できることを確認した。
3. **（OBSERVED、条件つき）** 同じ fit の上で、C_Lap と C_Q は多くの固定人工データ条件で違う K を選んだ。検討したどの条件でも C_Q の誤りは主に過小選択で、C_Lap の方が true K を選んだ dataset が多い条件が多かった。
   ただし C_Lap も weak な信号や K_true = 4 で境界に近づき、strong な属性の条件で過大選択が 1 件あった。
4. **（OBSERVED、条件つき）** Y 側・X 側の信号を強めると、検討した条件では両基準とも K = 3 に集まった。Y 側の感度は平均 edge 確率を揃えてもほぼ残った。
5. **（OBSERVED、条件つき）** 信号を揃えた K_true の設計では、K_true が大きいほど過小選択が増えた。K3 → K4 の境界は、fit gain と「体積 or P_Z + パラメータの罰則」の釣り合いとして記述できた。
6. **（UNRESOLVED）** 過小選択の原因、θ̂ の最適化の誤差の影響、一致性、他の条件への一般化、実データでの妥当性。

## 12. Claim Ledger

分類は RESEARCH_MASTER §14 と同じ。すべて lineage E（experimental prototype）であり、本文採用は決めていない（§14 参照）。

### ALLOWED（直接主張してよい）

| claim | evidence | label |
|---|---|---|
| 現行 C_Q（`calc_bic_dual`）は Schwarz BIC ではなく Q-based complete-data / ICL-type の基準である | #72 / #84 の report、KI-010 | FACT / DERIVED |
| scale_Z と var_z = 1 のもとで、C_Q の P_Z は nK(1 + ln 2π) であり、n = 75 で 1 次元あたり 212.84 増える | #84 report、Phase 9Y の commit 済みの値（10/10） | DERIVED |
| 観測データの evidence の Laplace 近似（nK 次元の同時 mode と同時 Hessian）を動機とする Candidate B（C_Lap）を定義・実装した | #86 / #88 の report、`laplace_k_criterion.py` | FACT |
| Candidate B の評価関数は、勾配・同時 Hessian が有限差分と ~1e-9 で一致し、本研究の規模（n = 75、K ≤ 5、Gaussian / Bernoulli / Poisson の X、Bernoulli の Y）で評価できた | Phase 9H、9I（15/15）、9K（100/100）、9P / 9T / 9V / 9X の全 refit | FACT / OBSERVED |
| Phase 9K の固定条件（20 dataset）で、同じ fit の上で C_Lap は K=3 を 19/20、C_Q は 14/20 選び、選択のパターンが違った | `lap_vs_cq_20/phase9k_20260928/paired_summary.json` | OBSERVED |
| 信号を揃えた K_true の設計（rep01..rep10）で、過小選択は K_true が大きいほど増え、特に C_Q で顕著だった（C_Q の exact は 10/9/7/1、C_Lap は 10/10/10/8） | `matched_k_true_sensitivity/phase9x_20260928/combined/` | OBSERVED |
| Phase 9Y は K_true = 4 の K3 → K4 の差を commit 済みの成分から 1e-10 以内で再構成し、C_Lap の余裕は 8/10、C_Q の余裕は 1/10 が正だった | `k34_boundary_decomposition/phase9y_20260929/` | OBSERVED |
| Phase 9 の実験は retry・置き換え・seed の救済なしで実行され、不完全な dataset（Phase 9V strong rep04）は不完全のまま記録された | 各 runinfo / ledger | FACT |

### QUALIFIED ONLY（条件を明記すれば使える）

| claim | 必須の条件 |
|---|---|
| C_Lap は一部の固定人工データ条件で C_Q より true K を選んだ dataset が多かった | n = 75、d = 12、G3/B6/P3、Y Bernoulli、8 反復の MCEM、start_B のみ、条件ごとに 10〜20 dataset。一般の優越性ではない。例外（K_true = 4 の 2 件、strong 属性での過大選択 1 件）を併記 |
| relational w を強めると、検討した条件では両基準とも選択が K = 3 に移った。平均 edge 確率を揃えてもこのパターンはほぼ残った | Phase 9P / 9T の固定条件。揃えたのは母集団の平均 edge 確率だけで、確率の分布・飽和は揃えていない。純粋な関係信号の効果ではない |
| attribute loading を強めると、検討した条件では C_Q の選択が K = 3 に移り、C_Lap はほぼ K = 3 のままだった | Phase 9U / 9V の対応設計（Z と Y を共有）。strong の 1 dataset は不完全。family ごとの寄与は分けていない |
| K_true = 4 では C_Lap も過小選択の境界に近づいた（8/10、差の median 13.4、最小 1.13） | 信号を揃えた設計の rep01..rep10。θ̂ の最適化の誤差の範囲かどうかは未検討 |
| 同じ条件でも seed の組によって C_Q の数は違う（Phase 9E 17/20 と Phase 9K 14/20） | 同じ protocol、別の seed の組。変動の記述であり確率の推定ではない |

### NOT ALLOWED（書いてはいけない）

- C_Lap は一般に C_Q より優れている / 基準の勝者を決めた
- Candidate B は一致性をもつ / K 選択の一致性を示した
- K 選択の問題を解決した
- Phase 9 の数は一般の recovery 確率を推定している
- 一般的な頑健性を示した
- P_Z が過小選択の原因である / 1 次元あたりの信号の希釈が原因である
- 実データでも正しい K を選べる / 実データの潜在次元の妥当性を示した
- lineage E（Candidate B とその比較）は自動的に本文に採用できる
- Poisson の列は一般に悪い（Phase 9V の 1 件の数値的な失敗からの一般化）
- 平均 edge 確率を揃えた実験は純粋な関係信号の効果を分離した / loading scale の実験は純粋な属性の情報の効果を分離した
- C_Q を「Schwarz BIC」、C_Lap を「厳密な周辺尤度」と呼ぶ

### UNRESOLVED

| 番号 | 未解決の問い |
|---|---|
| Z9-U1 | K3 → K4 の fit gain がなぜその大きさになるか（MC 平均と joint mode の違い、1 次元あたりの信号、fit の誤差） |
| Z9-U2 | 有限回（8 反復）の MCEM と、θ̂ が ℓ_Lap の最大化点から離れていることの影響（Phase 9M〜9R は 1〜20 ステップの範囲しか見ていない） |
| Z9-U3 | 漸近的な振る舞い・一致性（有効標本数、特異性を含む） |
| Z9-U4 | n / d / K_true / family の構成を変えたときの一般化 |
| Z9-U5 | 別の信号の揃え方（例: 次元ごとの energy を固定）で K_true のパターンが変わるか |
| Z9-U6 | 実データでの K 選択の妥当性（真の K が分からない状況での評価方法を含む） |
| Z9-U7 | 実装の ℓ_Lap と厳密な周辺尤度の差（Phase 9J で IS の重みが退化） |
| Z9-U8 | joint-mode solver の NOT_STATIONARY（許容値のわずか上で 200 反復）を評価の定義としてどう扱うか |

## 13. UNRESOLVED の位置づけ

上の Z9-U1〜U8 はすべて **将来の研究課題**であり、完了した有限標本の実験の解釈を妨げる blocker ではない。
各実験は事前に固定した protocol で実行され、technical validity（provenance、retry なし、再構成の確認）が保たれている。
Phase 9 を閉じるために一致性・実データでの妥当性・fit gain の原因を解く必要はない（Issue #126 の closure test）。

## 14. 修論で使う場合の候補の表現（採用は決めていない）

**MANUSCRIPT_ADOPTION_STATUS: NOT_DECIDED_BY_HUMAN**。以下は、後の Human Gate が lineage E を使うと決めた場合に科学的に安全な表現の候補である。

1. 既存の Q 型基準は Schwarz BIC ではなく、Z の事前密度を含む完全データの Q 関数に基づく ICL 型の基準であることを整理した。
2. 観測データの周辺尤度（Z を積分した evidence）の Laplace 近似を動機とする基準（Candidate B）を構成・実装し、評価関数が数式と一致することを確認した。
3. 固定した人工データ条件（n = 75, d = 12, 混合型属性, Bernoulli 関係）で、同じ推定結果に対して Candidate B と既存の Q 型基準は異なる K の選択のパターンを示し、既存の基準の誤りは主に過小選択だった。
4. 信号の平均的な強さを揃えた K_true の実験では、K_true が大きい条件で過小選択が増え、既存の Q 型基準で顕著だった。
5. K_true = 4 の K = 3 と 4 の境界は、当てはまりの改善と、体積（または Z の事前の項）およびパラメータの罰則との釣り合いとして記述できた。

## 15. Phase 9 の closure の判断

| closure test | 判定 | 根拠 |
|---|---|---|
| 1. 基準の正体が明確 | YES | §2・§3 |
| 2. Candidate B の実装の技術的な確認が述べた範囲で十分 | YES | §5（9H・9I・9K ほか、全 refit で評価可能） |
| 3. ベースラインの特徴づけ | YES | §6（Phase 9K） |
| 4. Y 側の感度 | YES | §7（9P・9T） |
| 5. X 側の感度 | YES | §8（9U・9V、1 dataset 不完全を明記） |
| 6. K_true の感度 | YES | §9（9W・9X） |
| 7. K3 → K4 の境界の整理 | YES | §10（9Y） |
| 8. 残りは将来の課題で、既存の実験の解釈を妨げない | YES | §13 |
| 9. claim の境界が明示されている | YES | §12 |

**source の矛盾: なし**。確認した注記は 2 つ。(a) Phase 9E と 9K の C_Q の数の違いは seed の組が違うためで矛盾ではない（§6）。
(b) Phase 9Y の Issue 本文は volume を「logdet の差」と書いていたが、既存の定義（‖Ẑ‖² を含む）に従い内訳も記録しており、数値の矛盾ではない。

## **DECISION: PHASE9_K_SELECTION_SYNTHESIS_COMPLETE**

merge 済みの証拠は一つの claim ledger に矛盾なく統合でき、残る問いは完了した有限標本の研究の解釈を妨げない将来の課題である。
**Phase 9 の K 選択の research thread: CLOSEABLE**（close するのは Human）。この判断は証拠の統合についてであり、本文採用の判断ではない。

**最大の将来の課題**: K_true が大きいときの過小選択が、基準の構造（C_Q の P_Z のような固定の罰則）によるのか、fit の当てはまりの大きさ（MC 平均と joint mode の違い、1 次元あたりの信号、有限回の MCEM）によるのかを、事前に設計した実験で分けること。

## 付録: 確認した source

| phase | Issue / PR | report | 一次 artifact |
|---|---|---|---|
| C_Q の確認 | #72 / #76 | `reports/k_selection_theory/cq_decomposition_minimal_check_20260923.md` | （session validation の記録） |
| 9D family + K | #75 / #80 | `reports/distribution_selection/phase9d_gate75b_joint_family_k_summary_20260927.md` | `expfam/results/joint_family_k_selection/gate75b_20260927/`（6 path、exact 4 / under 2） |
| 9E 再現性 | #81 / #83 | `reports/distribution_selection/phase9e_k_repeatability_summary_20260927.md` | `expfam/results/k_repeatability/phase9e_20260927/`（C_Q K3 17/20） |
| 9F 理論 | #84 / #85 | `reports/k_selection_theory/cq_theoretical_clarification_20260927.md` | — |
| 9G 設計 | #86 / #87 | `reports/k_selection_theory/k_true_criterion_design_comparison_20260928.md` | — |
| 9H 実装 | #88 / #89 | `reports/k_selection_theory/laplace_evaluator_zero_em_verification_20260928.md` | `expfam/src/experimental/laplace_k_criterion.py` とテスト |
| 9I pilot | #90 / #91 | `reports/k_selection_theory/laplace_pilot_phase9i_20260928.md` | `expfam/results/laplace_pilot/phase9i_20260928/` |
| 9J IS 診断 | #92 / #93 | `reports/k_selection_theory/laplace_is_diagnostic_phase9j_20260928.md` | `expfam/results/laplace_is_diagnostic/phase9j_20260928/` |
| 9K 20 dataset | #94 / #95 | `reports/k_selection_theory/lap_vs_cq_paired_phase9k_20260928.md` | `expfam/results/lap_vs_cq_20/phase9k_20260928/` |
| 9L θ の罰則 | #96 / #97 | `reports/k_selection_theory/laplace_theta_penalty_theory_20260928.md` | — |
| 9M / 9N / 9O | #98 / #100 / #102 | 各 report | `theta_stationarity/`, `multistep_local/`, `derivative_failure_classification/` |
| 9P | #104 | `relational_w_sensitivity_phase9p_20260928.md` | `expfam/results/relational_w_sensitivity/phase9p_20260928/` |
| 9Q / 9R | #106 / #108 | 各 report | `weak_small_gap_one_step/`, `weak_small_gap_multistep/` |
| 9S / 9S2 / 9T | #110 / #112 / #113 | 各 report | `density_controlled_w_design/`, `density_controlled_w_recalibration/`, `density_controlled_w_sensitivity/` |
| 9U / 9V | #116 / #118 | 各 report | `attribute_scale_design/`, `attribute_scale_sensitivity/` |
| 9W / 9X | #120 / #122 | 各 report | `matched_k_true_design/`, `matched_k_true_sensitivity/` |
| 9Y | #124 | `k34_boundary_decomposition_phase9y_20260929.md` | `expfam/results/k34_boundary_decomposition/phase9y_20260929/` |

Phase 9 より前の K 選択の研究（Phase 7e / 8b の held-out score、clean true-K n-sweep の S1〜S4）は別の基準・別の条件であり、本統合の数とは混ぜていない（RESEARCH_MASTER §12.7〜§18）。
