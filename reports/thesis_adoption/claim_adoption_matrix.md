# 研究成果の採用判定表（claim adoption matrix）

- 作成日：2026-11-24。branch `docs/thesis-adoption-20261124`（base `d383cb5`）。新しい実験・MCEM・データ生成はしていない。
- **区分 A〜F はこの監査専用の推奨区分**であり、`RESEARCH_MASTER.md` の Claim Ledger（ALLOWED / QUALIFIED ONLY / NOT ALLOWED / UNRESOLVED）を置き換えない。表の「許される主張」「禁止される過剰主張」は Ledger と整合させ、Ledger にない項目は本資料の提案として示す。
- **最終的な採用は研究者と指導教員が決める（Human Gate）。**
- 実装系列の詳細は `implementation_lineage_adoption.md`。章の対応は `thesis_chapter_evidence_map.md`。

## 区分の定義

| 区分 | 意味 |
|---|---|
| **A** | 正式モデルの定式化・理論 |
| **B** | 正式モデルの本文に採用する検証結果 |
| **C** | 条件付きで本文採用を検討する結果（Human Gate が要る） |
| **D** | 探索的検証・付録 |
| **E** | 今後の課題・未解決 |
| **F** | 本文の根拠として使用不可 |

## 3 軸の略号

- **A 軸（Model scope）**：「同一」＝正式モデル（全列共通 family、X 切片なし、スカラー w0・w）と同じ観測モデル。「拡張」＝列ごとの family・自動選択・追加 family。「先行」＝Y=Bernoulli・X=Gaussian の先行モデル。
- **B 軸（Evidence eligibility）**と **C 軸（Claim strength）**は、表の「既知の問題」「許される主張」の列に反映した。

## 今回の一次確認（2026-11-24、数値を一次 CSV から再計算して一致を確認したもの）

| 確認項目 | 一次資料 | 再計算した値 |
|---|---|---|
| fixed K 選択 | `expfam/results/fixed_official/exp1_k9/fixed_exp1_bic_k1to9_bestk_by_trial.csv` | `best_k_by_trial_bic` は 30 行すべて 3 |
| fixed n 依存 | `expfam/results/fixed_official/exp2/fixed_exp2_n_sweep_improvement.csv` | A 0.4928、B 0.4124、C 0.5865（n=50→300） |
| fixed 誤指定 | `expfam/results/fixed_official/exp4/fixed_exp4_scen_{a,b,c}_ratios.csv`（`fix_w=False, fix_x=False`） | 最大 A 4.3414（XBern_YBern）、B 9.0405（XPois_YBern）、C 40.3706（XPois_YBern）。C の全行最大 46.8637 は ablation 行 |
| fixed d 依存 | `expfam/results/fixed_official/exp3/fixed_exp3_d_sweep_agg.csv` | A は d=5→30 で 0.2639→0.2046 と単調減少、B は d=15 で最小（0.1401）の後に増加、C は 0.0231〜0.0238 で平坦 |
| 先行研究再現 | `reproduction/results/comparison/comparison_main_table.csv` | k_est=3：0.17934 と 0.17988（差 0.00054）、k_est=4：差 0.00026、k_est=6：差 0.0236 |
| M1 | `expfam/results/misspecification_m1/phase_m1_20261013/summary.json` | dense +0.0200（15/20）、sparse +0.0662（17/20）。`protocol.json` は列ごとの family（G3/B6/P3） |
| E1 | `expfam/results/z_point_estimator_e1/e1_20261104/full/summary.json` | B_mode sparse +0.0871（20/20）、B_last dense +0.0021（9/20） |
| complementary blocks | `expfam/results/story_diagnostics/complementary_blocks_consistent_20260821_paired.csv` | sparse：y_only +0.5122（10）、single_bernoulli +0.4218（10）、single_poisson +0.3889（10）、single_gaussian +0.2030（9） |
| MovieLens user-disjoint | `expfam/results/real_data/movielens_userdisjoint/movielens_userdisjoint_20260822_paired.csv` | delta_primary +0.004248（23/30）、raw-Poisson −0.100274（1/30 で正） |
| 旧 Cora | `expfam/results/real_data/cora_clean/cora_heldout_summary_clean.csv`、`expfam/src/run_fixed_real_cora_heldout_link_prediction.py` | test_AP 0.4319（k=3）、0.4631（k=6）、random 0.1667。**test 辺を Y_train=0 として学習**（L.223–238）、次数による subset 選択（L.150–155）、NaN 時に seed を変えて retry（L.262–267） |
| Wine | `expfam/results/real_data/wine_clean/wine_bic_summary_clean.csv` | BIC_mean 最小は k=3（6324.77） |
| AF1 / AF2 / AF6 / AF8-v3 | 各 README と `af6_laplace_confirmation_20261114/statistical_analysis.csv` | AF6 A−B sparse +0.225 [−0.075, 0.525]（PRIMARY、success False）。AF8-v3 E2 balanced 0.915、E2−AF6 +0.023 [−0.004, +0.050] |
| CORA-SH1 | `expfam/results/real_data/cora_strict_heldout_family_main_20261124/heldout_scores.csv` と `manifest.csv`（集計スクリプトを通さずに再計算） | K=6 BX−GX +1.059e-4（13/15）、K=3 +1.136e-4（13/15）、BX−GXc −4.4e-5（3/15） |

旧 0.5 系列（B）の数値（n 依存 49/31/62%、誤指定 3.41×/7.35×/41.45×）は今回は再計算せず、`RESEARCH_MASTER.md` §9・§15 と `reports/mismatch_audit/mismatch_audit_report_20260708.md` の記載に依拠した。

---

## 採用判定表

| Claim ID | 研究上の問い | 成果・主張候補 | 一次資料のパス | Commit | 数値の根拠（CSV・行・条件） | 数式の根拠 | 実装系列 | 正式モデルと同一の数理構造か | 実験の種類 | データの独立性 | 評価指標 | 既知の問題 | 許される主張 | 禁止される過剰主張 | 推奨区分 | 修論の章 | Human Gate | 判断に必要な残りの情報 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TH-1 | 先行モデルをどう一般化するか | X・Y を G/B/P から独立に指定するモデルの定式化（θ、生成モデル） | `AGENTS.md` §1、`RESEARCH_MASTER.md` §3–§4 | `8f6d41c` | — | 生成モデル（§4）。w0・w はスカラー、X 切片なし | — | 同一（定義そのもの） | 理論 | — | — | Categorical 未対応（KI-005）。family_x は全列共通 | G/B/P について X・Y を独立に指定できるモデルを定式化した | 「指数型分布族すべてに対応」 | **A** | 3 | 不要 | なし |
| TH-2 | E-step を family に依存しない形で書けるか | 精度行列 A_i と勾配の一般式（V=A''/φ、Σ_{j≠i} に extra 1/2 なし） | `RESEARCH_MASTER.md` §6、`docs/math_notes/half_factor_math_explanation.md`、`reports/theory_audit/theory_audit_report_20260718.md` §4.1 | `8f6d41c` | CORA-SH1 の test（有限差分で一致、旧 0.5 クラスは同じ検査で失敗）：`expfam/src/experimental/test_cora_strict_heldout.py` | 対称関係尤度からの独立導出 | 採用式（C・D・E が実装） | 同一 | 理論＋数値検証 | — | 有限差分誤差 | 原論文の印刷式には 1/2 が**ある**（KI-001）。B 系列は Y 側に 0.5 が残る | 採用式は独立導出に基づき、有限差分と一致する | 「原論文にも 1/2 がない」「0.5 があっても Newton 方向は全体として正しい」 | **A** | 3 | 不要 | なし |
| TH-3 | 1/2 の所在の整理 | 5 系統の整理（原論文あり／B あり／採用式なし／C なし／MATLAB なし） | `RESEARCH_MASTER.md` §6.1、`docs/math_notes/half_factor_primary_source_confirmation_20260818.md` | `8f6d41c` | 原論文 Eq.19/20/22/23、Appendix A-1/A-3/A-5 | 尤度レベルの 1/2 と z_i 微分での合算 | A・B・C・MATLAB | — | 理論・文献確認 | — | — | MATLAB は補助比較（Y 側勾配に w 欠落の疑い） | 採用式と原論文印刷式の差は独立導出に基づく意図的な差 | 「MATLAB が正解」「先行研究全体の誤り」 | **A** | 3（または付録） | 不要 | なし |
| TH-4 | 分散関数の役割 | V(η)=A''(η)/φ（Bernoulli σ(1−σ)、Poisson e^η、Gaussian 1/σ²）が勾配・精度に入る | `RESEARCH_MASTER.md` §6 | `8f6d41c` | — | 指数型分布族の標準的性質 | 全系列 | 同一 | 理論 | — | — | Gaussian-Y の φ=σ_Y²（実装は σ_Y を保持し二乗） | 分布族の違いが尤度・勾配・曲率に入る経路 | — | **A** | 3 | 不要 | なし |
| TH-5 | K は観測分布から識別できるか | P1（Poisson-X の K^rank）、P2・P3・P8（Gaussian-Y、w≠0）、P4、P5、P6、P7 | `RESEARCH_MASTER.md` §17.3、`reports/identifiability/true_k_identifiability_hardened_20260904.md` | `8f6d41c` | `tools/research_audit/verify_identifiability_identities.py`（81 行、独立に反証可能 41 行。P3・P6 は独立数値なし） | 母集団モーメント・キュムラント | — | 同一（canonical モデル） | 理論（population） | — | — | すべて population の主張。有限標本・一致性なし（U6）。**実験で使う Bernoulli-Y は未解決（U2）** | 明示した条件の下で population の識別性が成り立つ（Ledger §18 ALLOWED） | 「Bernoulli-Y でも K が識別できる」「有限標本で K を推定できる」「一致性」 | **A**（前提を明記）。Bernoulli-Y 部分は **E** | 3（または付録） | 不要（本文に入れる範囲は相談） | P3・P6 の独立数値確認 |
| TH-6 | 現行の K 選択基準は何か | `calc_bic_dual`（C_Q）は Schwarz BIC ではなく Q ベース完全データ型（ICL 型）。P_Z=nK(1+ln2π) の固定罰則 | `RESEARCH_MASTER.md` §12.6・§17.4・§19、KI-010/KI-019 | `8f6d41c` | `utils_expfam.py` の実装 | Q 関数と周辺尤度の差 | B/C/E | 同一 | 理論・コード監査 | — | — | 関数名・CSV 列名 `BIC` は変更しない | C_Q は ICL 型の基準で Schwarz BIC ではない。正当化されない理由は特異性・境界・有効標本数 | 「Schwarz BIC で K を選んだ」「非入れ子だから BIC が使えない」 | **A** | 3（または 4） | 不要 | なし |
| TH-7 | Bernoulli-Y で K を識別できるか | 未解決（U2）。w0=0 の Bernoulli-Y では一次モーメントが K の情報を持たない（P7） | `RESEARCH_MASTER.md` §17.6 | `8f6d41c` | — | — | — | 同一 | 理論（未解決） | — | — | 実験の大部分が Bernoulli-Y | 「未解決」と明記 | 識別可能／不可能のどちらの結論も | **E** | 7 | 不要 | 証明 |
| TH-8 | K 選択基準の一致性 | どの基準についても未証明（U6、Z9-U3） | `RESEARCH_MASTER.md` §17.6・§19 | `8f6d41c` | — | 特異モデル、RLCT 未知 | — | 同一 | 理論（未解決） | — | — | — | 「未解決」 | 「一致性を確認」 | **E** | 7 | 不要 | — |
| TH-9 | 分布選択の KL 議論 | KL≥0 は (z,x) の同時分布での母集団の性質。Q スコアに適用できるのは Z を真のモデルの厳密な事後分布から取る場合だけ | `reports/research_synthesis_20261116/mathematical_open_questions.md` §2.1（`c535910`）、AF3 README §2.3（`f71c04d`） | `c535910` / `f71c04d` | — | 標準的な KL 不等式 | — | **拡張**（列ごとの family 選択） | 理論 | — | — | 有限標本の保証なし | 母集団の性質と適用条件 | 「Q スコアは真の分布を選ぶ」 | **D** | 6（または付録） | 要（AF の扱い） | — |
| TH-10 | 積分型の選択基準は何に依存するか | 積分した基準は prior（測度）に依存し、非入れ子の 2 候補で O(1) の差が残る（AF8） | `expfam/results/auto_family_z_recovery/af8_integrated_loading_20261115/mathematical_derivation.md` §1 | `f71c04d` | A''_CMP2(0)=0.5131 | 再パラメータ化と Jacobian | E＋PR #129 | **拡張**（CMP を含む） | 理論 | — | — | 「正しい」prior は規範的な問題で未解決 | 積分型基準は prior に依存する | 「積分すれば正しく選べる」 | **D** | 6（または付録） | 要 | — |
| SY-1 | 一般化は実装として成立するか | X・Y を G/B/P から独立に指定できる実装がある | `expfam/src/model_dual_expfam.py`（`VALID_FAMILIES`）、`test_dual_expfam.py` | `8f6d41c` | 5 テスト PASS | — | B（C・D・E が継承） | 同一 | コード | — | test | B は Y 側 0.5（KI-001） | 実装レベルの一般化（性能主張ではない） | 「性能が向上した」 | **B** | 3 | 不要 | C 系列で同じ test を引用するか |
| SY-2 | 先行研究と同条件で同じ結果になるか | RMSE(Z) 差 < 0.001（k_est=3・4、5 試行） | `reproduction/results/comparison/comparison_main_table.csv` | `8f6d41c` | k_est=3：0.17934 と 0.17988。k_est=6 では差 0.0236 | — | B と `reproduction/src/model.py`（どちらも 0.5 系統） | 先行（Gaussian-X/Bernoulli-Y） | 人工データ | 5 試行 | RMSE(Z) | 比べた両方が 0.5 系統。**採用式の検証ではない** | 「先行研究条件で再現実装と同じ挙動をする（k_est=3・4）」 | 「採用式が正しいことを示した」「全 K で一致」 | **B**（限定付き） | 4 | 要（B 系列を本文に残すか） | C 系列での同条件比較の有無 |
| SY-3 | n を増やすと Z の推定は良くなるか | n=50→300 で RMSE(Z) が A −49.3%、B −41.2%、C −58.6% | `expfam/results/fixed_official/exp2/fixed_exp2_n_sweep_improvement.csv` | `8f6d41c` | 3 行（scenario A/B/C）、`relative_improvement_rate` | — | **C** | 同一 | 人工データ（historical generator） | 各 n 10 試行 | Procrustes RMSE(Z) | historical generator は canonical の literal generator ではない（KI-021）。clip あり | 明示した生成設定の中での有限標本観測（Ledger ALLOWED） | 「一致性」「一般に n で改善」。B 系列の 49/31/62% と同じ表に置く | **B** | 4 | 不要 | なし |
| SY-4 | d を増やすと Z の推定は良くなるか | A は単調改善、B は d=15 で最小の後に悪化、C は平坦 | `expfam/results/fixed_official/exp3/fixed_exp3_d_sweep_agg.csv` | `8f6d41c` | A 0.2639→0.2046、B 最小 0.1401（d=15）、C 0.0231〜0.0238 | — | **C** | 同一 | 人工データ | 各 d 10 試行 | RMSE(Z) | シナリオ依存 | 「d の効果はシナリオに依存する」 | 「d を増やせば改善する」 | **C**（registry △） | 4 | 要（本文か付録か） | なし |
| SY-5 | 分布を誤指定すると何が起きるか（正式モデル） | fixed 系列で誤指定により RMSE(Z) が悪化（最悪 A 4.34×、B 9.04×、C 40.37×） | `expfam/results/fixed_official/exp4/fixed_exp4_scen_{a,b,c}_ratios.csv` | `8f6d41c` | `fix_w=False, fix_x=False` の行の `ratio_vs_oracle` 最大。C の 46.86 は ablation | — | **C** | **同一**（全列共通 family の誤指定） | 人工データ | 10 試行 | RMSE(Z) 比 | 最大値は X・Y 両側の誤指定セル。Y 側だけで最大 7.9×（B）・19.6×（C） | 誤指定が悪化させた（Ledger ALLOWED）。両側／片側を区別 | 「0.5 除去の効果」（B 系列 41.45× との差）。23.6/41.45/38.97 を同条件として並べる | **B** | 4 | 不要 | 片側誤指定の内訳表を本文に入れるか |
| SY-6 | 正式モデルで K を選べるか（記述） | C_Q が K=3 を候補 1–9 から 30/30 trial で選択 | `expfam/results/fixed_official/exp1_k9/fixed_exp1_bic_k1to9_bestk_by_trial.csv` | `8f6d41c` | `best_k_by_trial_bic` 30 行すべて 3 | — | **C** | 同一 | 人工データ | 各シナリオ 10 | 選択 K | n=150 の 1 点、historical generator | Ledger QUALIFIED の文言どおり（Schwarz BIC ではない、一般の K 回復ではない） | 「BIC で正しい K を選べる」 | **C** | 4 | 要 | なし |
| SY-7 | 学会予稿（B 系列）の数値をどう扱うか | Exp1–4（B 系列）：n 依存 49/31/62%、誤指定 3.41×/7.35×/41.45× | `expfam/results/exp_scenario_{A,B,C}_exp{1,2,3,4}_*.csv`、`conference_submission_final_draft.md` | `8f6d41c` | `RESEARCH_MASTER.md` §9・§15（今回は再計算していない） | — | **B（0.5 あり）** | 同一 | 人工データ | 10 試行 | RMSE(Z) | KI-001（Newton 方向の正しさを断定できない）。7.35× の条件特定は未完了 | 予稿時点の記録として「0.5 が残る実装による」と明記 | C 系列と同じ表・図に置く、差を 0.5 除去の効果と読む | **D** | 付録（予稿との対応） | **要** | B・C の条件対応表（存在しない、§15） |
| SY-8 | per-column の同時利用は効くか | sparse-Y・補完的 block の人工設定で RMSE_Z 改善（y_only 比 +0.51 ほか） | `expfam/results/story_diagnostics/complementary_blocks_consistent_20260821_paired.csv` | `8f6d41c` | sparse の 4 行（10/10/10/9） | — | **E+F**（per-column） | **拡張** | 人工データ | 10 trials | RMSE_Z | dense では +0.009 まで縮小。prototype | Ledger QUALIFIED の 6 限定付き | 「一般に優れる」「実データでも有効」「正式手法」 | **D** | 6 または付録 | 要 | なし |
| SY-9 | 改善の機構は latent coverage か | Gaussian comparator で I=+0.114（9/10）、secondary は追随しない | `expfam/results/story_diagnostics/matched_latent_coverage_ablation_20260821_interaction.csv` | `8f6d41c` | Ledger §12.4 | 恒等式 I=D_G−D_J | **E+F** | **拡張** | 人工データ | 10 | RMSE_Z 差 | D_J=+0.201 が大きい | 「単一機構として確立していない」 | 「latent coverage が原因」 | **D** | 付録 | 要 | — |
| MS-1 | 誤指定の悪影響は Y の情報量で変わるか（B→P） | dense +0.020（15/20）、sparse +0.066（17/20）。E1 で sparse を別 seed 系列で再現（mode +0.087、20/20） | `expfam/results/misspecification_m1/phase_m1_20261013/`、`expfam/results/z_point_estimator_e1/e1_20261104/full/` | `8f6d41c` | `summary.json`（M1 regimes、E1 deltas） | — | **E（per-column）** | **拡張**（protocol の `family_x_true` は G3/B6/P3） | 人工データ（canonical-clean-mixed-v1） | 20 データ。E1 は同じデータで別 seed | Procrustes RMSE_Z | dense の last 推定は E1 で再現しない（+0.002、9/20）。1 条件 | 「この拡張設定で B→P の悪影響が sparse で大きかった」 | 「誤指定の悪影響は一般に Y が少ないほど大きい」「正式モデルでの結果」 | **C**（拡張モデルの結果として） | 4（拡張節）または 6 | **要** | 全列共通 family で同種の結果を得るか（SY-5 で代替できるか） |
| MS-2 | 誤指定の種類で交互作用は変わるか | B6→P6 は Y10% で悪化大（+0.082 vs +0.012）、B6→G6・全 G は Y100% で大 | `856d475:expfam/results/misspecification_extended/phase_20261110/`（registry 行 MX2） | `856d475`（未 merge） | registry MX2 行の mean Δ | — | **E（per-column）** | **拡張** | 人工データ | 20 データ、2 seed | RMSE_Z | 重複実行事故（最初の記録のみ採用、`duplicate_run_incident_20261110.md`）。未 merge | 「交互作用の向きは誤指定の種類に依存した」 | 「一般法則」 | **C**（MS-1 と同じ扱い） | 4 拡張節または 6 | **要**（未 merge PR の扱い） | PR #168 を merge するか |
| MS-3 | 正式モデルの誤指定で Y の情報量の効果は | — | — | — | — | — | — | — | — | — | — | **正式モデル（全列共通）での Y 情報量 × 誤指定の実験は存在しない**（SY-5 は Y の観測率を変えていない） | — | 「正式モデルで Y が少ないほど誤指定の影響が大きい」 | **E** | 7 | 要（必要か） | — |
| KS-1 | held-out 予測で K を選べるか | Phase 7e：3 replicate で K=3,3,5。Phase 8b：新規グリッド 18/24 | `expfam/results/k_selection/heldout_full_pilot_20260824/`、`k_true_robustness_full_attempt2_20260904/` | `8f6d41c` | `replicate_selection.csv`、`selection_matrix.csv` | — | **E-scalar**（Poisson-X） | 同一（scalar） | 人工データ（historical generator、KI-021） | 各条件 3 replicate | held-out plug-in log score | prototype、generator が canonical でない | Ledger §12.7・§16 の文言 | 「66.7% の回復率」「consistency」「Q_strict で選んだ」 | **D** | 付録（または 4 の K 節） | 要 | E-scalar の昇格可否 |
| KS-2 | n を増やすと K 選択は真値に近づくか | clean n-sweep：K_TRUE=5 で S1 2.62→5.00、S2 1.75→4.88（896 fits） | `expfam/results/k_selection/clean_true_k_asymptotics_20260904/` | `8f6d41c` | `selection_matrix.csv` | — | **E-scalar** | 同一（scalar、canonical clean generator） | 人工データ | 4/8 反復 | 選択 K | 真値一致は単調でない。固定 EM 予算の交絡。Bernoulli-Y は U2 | Ledger §18 QUALIFIED | 「一致性」「K_TRUE に収束」 | **C** | 4（K 選択節） | **要** | 同上 |
| KS-3 | Laplace 型基準は C_Q より K を選ぶか | Phase 9K：C_Lap 19/20、C_Q 14/20。matched K_true 設計など | `expfam/results/lap_vs_cq_20/phase9k_20260928/` ほか | `8f6d41c` | `paired_summary.json` | C_Lap の定義（同時 mode・Hessian） | **E（per-column、G3/B6/P3）** | **拡張** | 人工データ | 20 データ | 選択 K | 同条件・別 seed で C_Q は 14/20 と 17/20。厳密な周辺尤度ではない | Ledger §19 QUALIFIED | 「C_Lap は一般に優れる」 | **D**（MANUSCRIPT_ADOPTION_STATUS: NOT_DECIDED） | 付録 | **要** | — |
| KS-4 | C_Q の過小選択は何で決まるか | d-sweep（d=15→60 で正解 0→3→5/8）、R1/R2、checkpoint 再評価、恒等式 C_Q−C_Lap−2H_Lap=termA+2R3 | `expfam/results/k_selection/{dsweep_k_selection_20261105,k_selection_robustness_20261105,clap_checkpoint_reeval_20261105}/` | `f71c04d`（`main` = `8f6d41c` には未収録。dsweep などは §20–22 の統合 PR 待ち） | `by_d.csv` ほか | 代数的分解（DERIVED） | **E-scalar**（Poisson-X） | 同一（scalar） | 人工データ | 8 データ | 選択 K、slack | MCEM 未収束。原因は未分離 | Ledger §20–22 QUALIFIED | 「過小選択の原因を解明」「収束させても変わらない」 | **D** | 付録 | 要 | — |
| RD-1 | Wine に適用できるか | fixed で BIC 最小 k=3（真のクラス数と一致）、ablation で Y が主要因 | `expfam/results/real_data/wine_clean/` | `8f6d41c` | `wine_bic_summary_clean.csv`（k=3：6324.77） | — | **C** | 同一（Gaussian-X / Bernoulli-Y） | 実データ（in-sample） | 1 データ | C_Q、NMI など | **Y はラベル由来**（自然ネットワークではない） | 「ラベル由来の関係での in-sample 検証」 | 「自然ネットワークでの有効性」「K を正しく選んだ」 | **D** | 5（補助）または付録 | 要 | — |
| RD-2 | 旧 Cora で予測できるか | balanced_degree subset（n=280）で test_AP 0.43–0.46（random 0.167 の 2.6–2.8 倍） | `expfam/results/real_data/cora_clean/cora_heldout_summary_clean.csv` | `8f6d41c` | k=3：0.4319、k=6：0.4631 | — | **C** | 同一（Bernoulli-X / Bernoulli-Y） | 実データ（transductive） | 3 split × 2 trial | AP（負例 1:5 サンプリング）、AUC | **今回の一次確認**：①test 辺を Y_train=0 として学習（スクリプト L.223–238）②全グラフの次数で subset を選ぶ（L.150–155）③NaN 時に seed を変えて retry（L.262–267）④負例サンプリングで AP の基準が人工的。**KNOWN_ISSUES「今すぐ主張してよいこと」に test_AP≈2.6–2.8×random が載っているが、strict held-out ではない** | 「test 辺を学習時に 0 とする transductive 設計で、負例 1:5 のサンプリング評価では random を上回った」 | 「strict held-out で予測できた」「full Cora に一般化」 | **D** | 5（経緯）または付録 | **要**（canonical docs との不一致の扱い） | KNOWN_ISSUES の該当行の forward update（研究者の判断） |
| RD-3 | 旧 Cora で K を選べるか | C_Q 最小 k=1、AUC/AP 最大 k=6、NMI 最大 k=3 | `expfam/results/real_data/cora_clean/cora_bestk_clean.csv` | `8f6d41c` | 同 CSV | — | **C** | 同一 | 実データ | 3 試行 | 複数 | 原因未解明（KI-011） | 「基準ごとに最適 k が割れた」 | 「C_Q は実データで適切な k を選ぶ」 | **D**（限界の例として 5 で言及可） | 5 | 要 | — |
| RD-4 | MovieLens（in-sample） | Bernoulli-X / Poisson-Y で Pearson ≈ 0.96 の再構成 | `expfam/results/real_data/movielens_final_clean/`、`movielens_colike_clean/` | `8f6d41c` | registry の実データ節 | — | **C** | 同一 | 実データ（in-sample） | 1 データ | 再構成 Pearson | 旧 pilot は full-data popularity で movie 選択（selection leakage）。過分散 var/mean≈10 | 「数値的に安定に収束し、in-sample で再構成できた」 | 「未知ペアの共評価数を予測できた」 | **D** | 付録 | 要 | — |
| RD-5 | MovieLens（strict、user-disjoint） | delta_primary +0.0042（23/30）、std/|mean| 2.89 | `expfam/results/real_data/movielens_userdisjoint/movielens_userdisjoint_20260822_paired.csv` | `8f6d41c` | contrast `delta_primary` | — | **E+F** | **拡張**（mixed X） | 実データ | 30 split は同じ 943 users の再利用 | Poisson mean log score（score として） | 過分散、prototype | Ledger QUALIFIED（平均は正だが変動に比べ小さい） | 「有効性を確認」「有意」「30 independent」 | **D** | 6 または付録 | 要 | — |
| RD-6 | raw-count Poisson は悪化するか | raw-Poisson − log-count = −0.100（29/30 で悪化） | 同上 `sec_rawpoisson_vs_mixedlog` | `8f6d41c` | 同上 | — | **E+F** | **拡張** | 実データ | 同上 | 同上 | 原因未解明（4 要因が交絡、KI-018） | Ledger QUALIFIED | 「Poisson は不適」「切片欠如が原因」 | **D** | 7（限界）または付録 | 要 | — |
| CO-1 | 現行仕様の X family で Cora の strict held-out 予測は変わるか | primary Δ(BX−GX raw, K=6, mc_predictive) = +1.06e-4 nats/pair（13/15 対、4/5 split） | `expfam/results/real_data/cora_strict_heldout_family_main_20261124/`、`reports/real_data/cora_strict_heldout_family_main_report_20261124.md` | `d383cb5`（local） | `heldout_scores.csv`（mc_predictive、K=6）、protocol `74f874b` | — | **E-scalar＋BX-Newton** | **同一** | 実データ（strict dyad held-out、全ペアの 20%、mask で除外） | 5 split × 3 seed。**統計単位は Cora 1 ネットワーク**（独立標本ではない） | 平均 Bernoulli log score（自然陽性率） | 8 反復で EM 未収束（BX の Q 増分が最大）。最終サンプル plug-in では −1.7e-5 で向きがそろわない。fit_technical.csv の列ずれ（解析スクリプトで決定的に復元、生ファイル不変）。npz は git 管理外（sha256 記録あり）。K=6 は過去の Cora 実験に基づく | 「固定予算・現行仕様で、BX raw が GX raw をわずかに上回った（記述的）」 | 「Bernoulli が正しい分布だから良い」「family の優越」「有意」「一般化」 | **C** | 5 | **要**（E-scalar と Newton-F の昇格） | 研究者による結果の確認、E-scalar 昇格の判断 |
| CO-2 | その差は分布族の違いによるか | GXc（中心化 Gaussian、K=6 secondary）が BX を上回る（BX−GXc −4.4e-5、3/15）。GXc−GX +1.50e-4（15/15） | 同上 | `d383cb5` | 同上（arm GXc） | — | 同上 | **同一の観測モデルに前処理（列中心化）を加えた arm** | 同上 | 同上 | 同上 | GXc は事前登録の secondary。前処理の違いは切片なし仕様と交絡 | 「family の差は小さく、中心化の有無で逆転した」 | 「分布族そのものの優越」「切片欠如が原因と確定」 | **C**（CO-1 と一体で） | 5・7 | **要** | 同上 |
| CO-3 | X は Cora の予測に寄与するか | K=6 では BX・GX とも YO を 15/15 で上回る。K=3 では BX≈YO、GX は YO より悪い | 同上 | `d383cb5` | 同上 | — | 同上 | 同一 | 同上 | 同上 | 同上 | K 依存。統計単位は 1 ネットワーク | 「X の寄与は K に依存した」 | 「属性を入れれば予測が良くなる」 | **C** | 5 | **要** | 同上 |
| AF-1 | {0,1} 列で分布をデータから選べるか | Bernoulli vs Poisson は 160/160 fit で全列正解、Wrong の悪化（sparse 0.0816）を回避 | `expfam/results/auto_family_z_recovery/af1_20261109/` | `f71c04d`（未 merge） | README L.14–16、L.101 | log Bern − log Pois = e^η − log(1+e^η) > 0 | E（per-column） | **拡張** | 人工データ（事前登録） | 20 replicate | 選択の正解、RMSE(Z) | 結果は解析的にほぼ決まる | 「{0,1} 列では解析的にほぼ決まる」 | 「自動選択は一般に機能する」 | **D** | 6 | **要** | — |
| AF-2 | 近い分布（Poisson vs CMP2）を選べるか | Q で dense 108/120、sparse 92/120。Z は Support-only より小さく改善（joint mode のみ CI が 0 を含まない） | `expfam/results/auto_family_z_recovery/af2_20261110/` | `f71c04d` | README L.14、L.211–212 | — | E＋PR #129 | **拡張**（CMP） | 人工データ | 20 データ、1 条件 | 正解率、RMSE(Z) | CMP→Poisson へ偏る。新データでの Z の再評価なし | 総括 B4・B5 の限定付き文言 | 「近い分布でも一般に選べる」 | **D** | 6 | **要** | — |
| AF-3 | Q を上回る基準を作れたか | AF5 の改善は AF6（新 40 データ、事前登録）で未達成（+0.225 [−0.075, 0.525]）。AF8-v3 は境界の移動で prior 依存（NO-GO） | `af6_laplace_confirmation_20261114/statistical_analysis.csv`、`af8v3_full_exploratory/README.md` | `f71c04d` | AF6 PRIMARY 行（success False）、AF8-v3 E2 行 | AF8 の測度依存の導出 | E＋PR #129 | **拡張** | 人工データ（AF6 confirmatory、AF8-v3 exploratory） | AF8-v3 は AF6 データの再利用 | 正解数、balanced | forking paths（AF8-v3） | 「事前登録した確認で主要評価項目は未達成」 | 「Laplace / 積分基準は Q より優れる」 | **D**（否定的結果として） | 6 | **要** | — |
| AF-4 | Q スコアが偏る機構 | Q は Z を作った分布に引き寄せられる（AF3）。機構 (a)〜(c) は仮説 | `af3_selector_rescoring_20261113/README.md` | `f71c04d` | 同 README | — | E | **拡張** | 人工データ（統制比較） | 同じ 20 データ | 選択の割合 | 機構は HYPOTHESIS | 「同じ 20 データの統制比較で観察された」 | 「機構を証明した」 | **D** | 6 | 要 | — |

## 判定の要約

| 区分 | Claim ID |
|---|---|
| **A** | TH-1、TH-2、TH-3、TH-4、TH-5（Bernoulli-Y 部分を除く）、TH-6 |
| **B** | SY-1、SY-2（限定付き）、SY-3、SY-5 |
| **C** | SY-4、SY-6、MS-1、MS-2、KS-2、CO-1、CO-2、CO-3 |
| **D** | TH-9、TH-10、SY-7、SY-8、SY-9、KS-1、KS-3、KS-4、RD-1〜RD-6、AF-1〜AF-4 |
| **E** | TH-7、TH-8、MS-3、TH-5 の Bernoulli-Y 部分 |
| **F**（本文の根拠に不可） | AI 生成レポート（KI-007）、Phase 8b Attempt 1、`comparison_old_vs_fixed.png`（KI-004）、D-NB・F の legacy tail に依存する結果、旧 Cora を「strict held-out」として引用すること |

## canonical docs との不一致（報告のみ。canonical docs は書き換えていない）

1. **旧 Cora（RD-2）**：`KNOWN_ISSUES.md`「今すぐ主張してよいこと」に「Cora の held-out link prediction で random 基準を上回る予測性能を確認した（test_AP≈2.6〜2.8×random）」とある。一次コードでは test 辺を Y_train=0 として学習し、全グラフの次数で subset を選び、NaN 時に seed を変えて retry する設計だった。数値そのものは正しいが、「strict held-out」と読める書き方は避けるべきである。forward update するかは研究者が判断する。
2. **M1・E1・PR #168 のモデル範囲（MS-1、MS-2）**：総括資料は「本来の研究（指数型分布族への拡張）」の節に置いているが、`protocol.json` では列ごとの family（G3/B6/P3）であり、A 軸では拡張モデルである。総括資料は「本文採用は未決定（prototype）」と正しく書いているので矛盾ではないが、章の配置を決めるときに区別が要る。
