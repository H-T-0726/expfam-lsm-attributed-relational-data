# Family-selection 実験の coverage map：既存 artifact で何を評価できたか（2026-10-02）

- 位置づけ: **AUDIT / ZERO-EXPERIMENT**。既存 artifact の read-only・決定論的な集計のみ（新しい EM・dataset・seed 0）。
- 系列: すべて **lineage E（experimental prototype、本文採用不可）**。
- 理論的背景: `family_selection_hard_case_audit_20261002.md`（以下「監査」）。
- ラベル: FACT（artifact で確認）／ DERIVED ／ INTERPRETATION。

---

## 1. 対象 artifact（FACT）

| 区分 | artifact（`expfam/results/` 以下） | n / d / K_true / f_scale | X family（列） | dataset 数 | start |
|---|---|---|---|---|---|
| smoke | `family_selection/smoke_20260923`, `smoke_v2_20260924`, `smoke_v3_20260924` | 40 / 6 / 3 / 1 | G2 B2 P2 | 各 1（同じ data_seed 941001） | B, P |
| pilot C2（#74） | `family_selection/pilot_c2_20260925` | 75 / 12 / 3 / √2 | G3 B6 P3 | 3 | B, P |
| 9D joint family+K（#75） | `joint_family_k_selection/gate75b_20260927` | 75 / 12 / 3 / √2 | G3 B6 P3 | 3 | B, P |
| 9E K 再現性 | `k_repeatability/phase9e_20260927` | 同上 | 同上 | 20 | B |
| 9I / 9J | `laplace_pilot/phase9i_20260928`, `laplace_is_diagnostic/phase9j_20260928` | 同上 | 同上 | 各 3 | B |
| 9K | `lap_vs_cq_20/phase9k_20260928` | 同上 | 同上 | 20 | B |
| 9P / 9T | `relational_w_sensitivity/phase9p_20260928/{weak,strong}_w`, `density_controlled_w_sensitivity/phase9t_20260928/{weak,strong}_w` | 同上（X は 9K と共有） | 同上 | 各 20 | B |
| 9V | `attribute_scale_sensitivity/phase9v_20260928/{weak,base,strong}` | 75 / 12 / 3 / 1, √2, 2 | 同上 | 各 20 | B |
| 9X | `matched_k_true_sensitivity/phase9x_20260928/{K1,K2,K4}` | 75 / 12 / 1,2,4 / √6, √3, √1.5 | 同上 | 各 10 | B |

`generator_provenance.csv` を data_seed・列・`f_row_norm_sq`・列平均で重複除去すると、
**真 Bernoulli 列 854、真 Poisson 列 428、真 Gaussian 列 428**（9K / 9P / 9T は同じ X を共有）。
`family_by_k.csv` の score 判定行は合計 **6582 行**（K ごと・start ごとに数えた延べ数）。

---

## 2. Coverage map

| true family | observed support | gate or score | score で決まった件数 | 結果 | difficulty | limitation |
|---|---|---|---|---|---|---|
| Bernoulli | {0,1}（全列、確率 1） | **score** | pilot C2: **36**（3 rep × 2 start × 6 列、unique 列 18）。9D: 36（K_hat、全 K で 180）。Phase 9 全体: 延べ 6582 行、unique 列 854 | **全て Bernoulli**（pilot 36/36、Phase 9 全行）。最終 margin（−2 score）: pilot 45.69–65.50、全 artifact の最小 43.6 | **構造的に決定**（監査 §5：0/1 列では真の family によらず Bernoulli が厳密に勝つ）。「易しい」ではなく「score に判別の余地がない」 | 判別能力の証拠にならない。正解は不等式・最適化の収束・実装の整合性の確認。pilot C2 では最終候補 8/72 が収束 warning（grad_inf ≤ 1.30e-6）、影響は未測定 |
| Poisson | 非負整数・最大 ≥ 2 | **gate** | 0（gate） | 全て Poisson（gate） | support で自明 | **selector の評価ではない**。真 Poisson 列の最大値の最小は 3（全 artifact）、4（pilot C2） |
| Poisson | {0,1} のみ | score（になるはず） | **0** | **UNOBSERVED** | **STRUCTURALLY RARE**：1 列あたり ≤ 1.0e-10（n = 75、任意の f）、既存条件の F 周辺化で 4.5e-12、全 artifact の期待発生数合計 2.3e-7（監査 §4） | 起きた場合も score は**必ず Bernoulli**を選ぶ（監査 §5）。no-intercept では評価不能かつ結果は事前に決まる |
| Gaussian | 非整数 / 負を含む（確率 1） | **gate** | 0 | 全て Gaussian（gate） | support で自明 | selector の評価ではない |
| Gaussian | 整数のみ | — | 0 | 起こりえない（連続分布、確率 0） | — | — |
| Bernoulli | 2 以上 / 非整数を含む | — | 0 | 起こりえない（support 外） | — | — |

---

## 3. 何が評価でき、何が評価できていないか

### 評価できたこと（FACT / INTERPRETATION）

1. **機構の technical feasibility**（FACT）: 3 family・support gate・per-column 候補最適化・fresh refit が、
   retry / replacement / seed rescue 0 で最後まで動いた（pilot C2: 12/12、Phase 9 の各 runinfo）。
2. **開始点非依存**（FACT）: start_P から始めても iteration 1 で Bernoulli に入れ替わり、以後変化しない（pilot C2: 18 件、9D: 90 件）。
   **INTERPRETATION**: 監査 §5 より、これは「どの Z サンプルでも Bernoulli が勝つ」ことから当然に従う。
3. **margin の大きさが理論の下界と整合**（INTERPRETATION）: `2n g(0) = 46.0`（n = 75）、`24.55`（n = 40）に対し、
   観測 margin は 43.6 以上（n = 75）、23.9–24.8（n = 40）。
4. **Phase 9 の K 選択は実質 oracle family のもとで行われた**（FACT + DERIVED）: 全 score 判定行・gate 判定行で真の family と一致し、
   監査 §5.4 によりそれは構造的に保証される。K 選択の結果に family 誤選択の交絡はない。

### 評価できていないこと

| 項目 | 状態 | 理由 |
|---|---|---|
| 0/1 に見える真 Poisson 列での挙動 | UNOBSERVED | no-intercept + n = 75 で構造的に起きない |
| selector の判別能力（Bernoulli vs Poisson） | **評価不能（現行 score では定義上存在しない）** | 0/1 列では尤度 score は常に Bernoulli |
| 低 rate count 列・偏った binary 列 | 対象外 | no-intercept model の族に含まれない（Poisson 平均 ≥ 1、Bernoulli 率 = 1/2） |
| Gaussian vs 離散の比較 | 対象外（設計上 gate） | 測度が違い、尤度比較ができない（design §3.3） |
| 過分散 count（NB）、実データ | 対象外 | HG-4、実データ未実施 |
| 収束 warning の score への影響 | 未測定 | pilot C2 report §5 と同じ |

---

## 4. 36/36 が支持する claim（KEY FINDING 2 の根拠）

- **支持する**: 「凍結 C2 条件（n = 75, d = 12, K = 3, G3/B6/P3, f_scale = √2, 3 dataset × 2 start）で、prototype は
  真 Bernoulli の 0/1 列 36 件（unique 18 列）すべてで Bernoulli を選び、開始点に依らず同じ割り当てに収束した」。
  これは**実装が理論上の順序（0/1 列で Bernoulli > Poisson）どおりに動いた**ことの確認である。
- **支持しない**: 「selector が Bernoulli と Poisson を判別できる」「自動 family 選択の精度 100%」
  「難しい条件でも正しく選べる」「人手指定と同等以上」。
  真 Poisson の 0/1 列が来ても同じく Bernoulli を選ぶので、36/36 は判別能力について何も言わない。
- 既存 report（`reports/distribution_selection/phase9c_c2_research_first_pilot_summary_20260925.md` §4–§5）の記述は当時の記録として変更しない。
  そこでの「margin は大きいが易しい側にある可能性」という留保は、本監査で「score の構造で決まる」と精密化された。
