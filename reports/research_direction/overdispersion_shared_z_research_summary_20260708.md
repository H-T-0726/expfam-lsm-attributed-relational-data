# 過分散・共有Z・per-column family 研究フェーズ 統合レポート

作成日: 2026-07-08
ブランチ: `research/overdispersion-z-ablation`（`main` 02311e7 から分岐）
実施者: Claude Code（研究開発エージェント）+ ユーザ承認のもとで実行

関連詳細レポート:
- Phase 0 監査: `reports/research_direction/phase0_current_state_20260708.md`
- 過分散診断: `reports/overdispersion/movielens_overdispersion_diagnostics_20260708.md`
- pair mask 設計・実装: `reports/overdispersion/pair_mask_design_20260708.md`
- NB 設計・実装: `reports/overdispersion/negative_binomial_design_20260708.md`
- Poisson 誤指定分析: `reports/overdispersion/poisson_misspecification_report_20260708.md`
- 共有 Z ablation: `reports/shared_z_ablation/shared_z_ablation_report_20260708.md`
- per-column 設計: `reports/research_direction/per_column_family_design_20260708.md`
- ミスマッチ監査: `reports/mismatch_audit/mismatch_audit_report_20260708.md`

---

## 1. 何を実装したか（すべて新規、既存コード変更なし）

`expfam/src/experimental/`（fixed 系列 `DualExpFamLSMFixed` 継承、旧 0.5 版不使用）:

| ファイル | 内容 |
|---|---|
| `model_dual_expfam_masked.py` | pair mask 対応（strict held-out）。E-step/M-step/尤度の 6 箇所をマスク |
| `model_dual_expfam_nb.py` | experimental な NB2-Y 拡張（固定 r、log link + E-step は Fisher 情報量重み。M-step の w0/w 更新は既存同様 Adam であり Fisher scoring ではない — `nb_math_audit_20260708.md` 参照）。r→∞ で Poisson に退化 |
| `model_dual_expfam_percolumn.py` | 列ごとの X 分布族（gaussian/bernoulli/poisson 混在） |
| `em_runner.py` | 上記対応の汎用 MCEM ランナー（fix_x/fix_w ablation 対応） |
| `eval_utils.py` | held-out 予測尤度（全定数込み）・Pearson 過分散・moment r̂・mixed/NB 対応 BIC/Q_strict |
| `data_generator_overdispersed.py` | NB2-Y 人工データ生成（gamma-Poisson 混合） |
| `test_experimental_models.py` / `test_percolumn_model.py` | 計 10 テスト、全 PASS |

正しさの保証（テストで confirmed）:
- masked(mask なし) ≡ DualExpFamLSMFixed（数値一致）
- held-out ペアの Y を書き換えても学習が不変（**リーク無しの直接検証**）
- NB(r=1e9) ≡ Poisson、moment 推定が r=5 を r̂=4.95 で回復
- per-column（全列同一）≡ スカラー family モデル

既存ファイルの変更: `EXPERIMENT_REGISTRY.md` への本フェーズ実験表の**追記のみ**
（既存行の変更なし）。既存 CSV・図・原稿・モデルコードは一切変更していない。

## 2. 何を実験したか

| 実験 | 規模 | 結果先 |
|---|---|---|
| MovieLens 過分散診断（周辺 vs 条件付き + PPC） | 2 fits + PPC 600 複製 | `expfam/results/overdispersion/movielens_overdispersion_*` |
| MovieLens strict held-out（Poisson/NB/full 参照） | 36 fits, 5.3 分 | `movielens_strict_heldout_*` |
| 人工 NB-Y 誤指定（r∈{2,5,20,∞}×3 条件×5 trials） | 55 fits, ~9 分 | `poisson_misspecification_*` |
| MovieLens 共有 Z ablation（strict held-out） | 18 fits, ~4 分 | `expfam/results/shared_z_ablation/` |
| per-column デモ（混在 X、正指定 vs 全列強制） | 20 fits, 2.3 分 | `expfam/results/per_column_family/` |
| 既存ミスマッチ監査 / 既存 ablation 棚卸し | read-only | `expfam/results/mismatch_audit/`, `shared_z_ablation/existing_ablation_audit.csv` |

seed 設計: data/split/model を分離（51000/52000/53000, 41000/42000, 61000, 71000/72000, 21000, 31000）。
すべての実験に runinfo CSV（コマンド・日時・git HEAD・seed・設定・実行時間）を保存。

## 3. 主要結果（5 つの発見）

### 発見 1: 周辺過分散 ≠ 条件付き過分散（KI-012 の再解釈）
MovieLens Y_count の周辺 var/mean=9.89（KI-012 どおり）だが、潜在構造
フィット後の条件付き Pearson 過分散は k=3 で 1.14、k=5 で 0.76（in-sample）、
strict held-out でも k=3 で ≈1.34 / k=5 で ≈0.9。plug-in PPC でも Poisson は
棄却されない。**周辺過分散の大半は μ_ij の潜在的異質性で説明される**。
純 Poisson 生成の人工データですら周辺 var/mean=3.1 になることを統制実験で確認。
→ 「周辺 var/mean での family 診断は誤る」という方法論的主張が立つ。

### 発見 2: 過分散があるとき Poisson 誤指定は held-out 尤度と w 推定を壊すが、RMSE(Z) では見えない
人工 NB-Y（r=2）で Poisson は NB 比 held-out −0.76 nats/pair、w_err 6 倍。
用量反応が明瞭（r=20 でほぼ消失）。一方 **RMSE(Z)（事後サンプル）は
Poisson の方が小さい** — Poisson は Y 側曲率を過大評価（μ vs 真の μr/(μ+r)）
し事後分散を過小評価するため、サンプルがモードに寄る（機構は inference）。
→ 「評価指標の選択自体が誤指定研究の論点」という新しい論点。

### 発見 3: strict held-out が可能になり、従来評価の楽観を定量化
pair mask 実装により MovieLens で初の strict held-out 評価を実施。
full 学習（従来 masked evaluation 相当）は strict 比で te_ll +0.09〜0.18/pair、
te_rmse −0.5〜−0.9 の楽観。strict でも Pearson ≈ 0.93–0.95 で結論の方向は
維持（従来数値は下方修正が必要）。NB は本データでは te_ll 改善が小さい
（r̂≈170〜∞ の領域なので人工実験の用量反応と整合）が、Poisson が発散した
1 fit で NB は安定（Fisher 重みの飽和による頑健化）。

### 発見 4: 共有 Z のシナジーはデータ依存で、検証した実データでは弱い
人工 Scen.A/B では統合が単独比最大 2 倍以上良いが、Scen.C は Y-only と同値、
Wine は Y-only で完結、MovieLens（strict held-out）では X は Y 予測を改善せず
（−3.34 vs −3.30）、ジャンル NMI は X-only が最高（0.376 vs 0.31）。
→ fix_x/fix_w ablation を「共有仮定の事前検査」として定式化できる。

### 発見 5: 全列共通 family の強制は族の選び方次第で致命的
混在 X（gauss/bern/pois 各 3 列）で per-column 正指定 rmse_Z=0.219 に対し、
全列 bernoulli 強制は 5 trial 平均 rmse_Z=0.592（平均 w_err=0.52）、
うち最悪 1 trial は rmse_Z=1.37・w_err=2.53 と崩壊。
一方全列 gaussian 強制は 0.221 とほぼ無害（quasi-likelihood 的頑健性、
inference）。→ per-column 化の必要性と「Gaussian は比較的安全な
デフォルト」という実務指針の両方が得られた。

### 付録的成果: KI-003 の解決
41.5×（本文）の根拠 = `exp_scenario_C_exp4_mismatch.csv` の
X=Gaussian/Y=Poisson 条件（再計算 41.45×）と特定。23.6×/3.41×/7.35× も
すべて再現一致。fixed 版（4.34/9.04/40.37×）との対応表も作成。
ただし Scen.C の「最悪条件のラベル」は実装間で不安定（旧: XGauss/YPois、
fixed: XPois/YBern）— 主張の単位は倍率のオーダーにすべき。

## 4. confirmed / inferred / unknown

**confirmed（CSV・テストに基づく）:**
- 上記発見 1–5 の数値、KI-003 の条件特定、リーク無し・退化・等価性テスト
- 既知問題の状態: KI-001（0.5 は旧版のみ、新実験は fixed 系列）、
  KI-012（周辺過分散の数値は正しいが条件付きでは小さい）、
  pair mask 未対応（→ experimental で解消）、family_x 全列共通（→ 同）、
  NB/Categorical/有向 Y/欠測 MNAR 未対応（Categorical・有向は本フェーズも未着手）

**inferred（機構の解釈、要追加検証）:**
- RMSE(Z) の逆転機構（事後分散の過小評価）→ coverage 実験で検証可能
- NB の頑健化機構（Fisher 重みの飽和）
- MovieLens で Y を入れるとジャンル NMI が下がる理由（共評価の主因子が人気・視聴層）
- 全列 Gaussian 強制の頑健性（quasi-likelihood 解釈）

**unknown:**
- r̂ が 1 桁の実データ（zero 過剰カウント等）での NB の実利
- Cora での共有 Z ablation の結果（X–Y 重なりが強い候補、未実施）
- 事後分散の較正（Laplace 近似の coverage）
- BIC の観測ペア数依存性（mask 下での n の取り方、KI-010/011 系）
- MNAR 欠測下での挙動

## 5. 修論で使える主張 / まだ弱い主張

**使える（本フェーズの証拠で支持）:**
1. 周辺 var/mean による family 診断は潜在構造モデルでは誤誘導する
   （実データ + 統制人工実験の両方で実証）
2. 過分散下の Poisson 誤指定は held-out 尤度・w 推定を用量反応的に劣化させ、
   two-stage NB2 が回復する
3. strict held-out（pair mask）は小さな実装で可能であり、従来評価の楽観を
   定量化できる（リーク無しをテストで保証）
4. 共有 Z のシナジーはデータ依存 — ablation を事前検査として使うべき
5. 全列共通 family は混在属性で崩壊しうる（bernoulli 強制）/
   per-column 化は列重み付けとして自然に実装できる
6. 点推定指標（RMSE(Z)）は誤指定検出に不適切な場合がある

**まだ弱い（追加作業が必要）:**
- 「NB が実データで有効」（MovieLens では差が小さい。r̂ が小さい実データが必要）
- 「共有 Z は実データ一般で弱い」（2 データセットのみ、Y 構成に恣意性）
- RMSE(Z) 逆転の機構（サンプル vs モードの切り分け未実施）
- k と family の交絡の一般論（k∈{3,5} の 2 点のみ）

## 6. 図表候補（修論）

| 図 | ファイル | 主張 |
|---|---|---|
| 周辺 vs 条件付き過分散（ヒストグラム+PPC） | `figures/overdispersion/movielens_y_distribution.*` | 発見 1 |
| 平均–分散関係（条件付き） | `movielens_mean_variance.*` | 発見 1 |
| 誤指定の用量反応（held-out ll） | `poisson_misspec_heldout_ll.*` | 発見 2 |
| w 推定誤差 | `poisson_misspec_w_err.*` | 発見 2 |
| RMSE(Z) の逆転 | `poisson_misspec_rmse_z.*` | 発見 2（指標論） |
| strict vs full の楽観 + NB 比較 | `movielens_strict_heldout_comparison_k{3,5}.*` | 発見 3 |
| 共有 Z ablation 4 指標 | `figures/shared_z_ablation/movielens_shared_z_ablation.*` | 発見 4 |
| per-column デモ表 | `per_column_demo_agg.csv`（図化は今後） | 発見 5 |

## 7. advisor 想定問答（18 問）

1. **Q: 周辺過分散が潜在構造で説明されるなら、そもそも NB は不要では？**
   A: MovieLens ではその通り差が小さい。しかし人工実験で条件付き過分散
   （r=2〜5）があるとき Poisson は held-out で −0.76 nats/pair 劣化する。
   「不要かどうかをデータから診断する手続き（条件付き Pearson 過分散・
   held-out 過分散・r̂）」を与えたことが貢献であり、NB はその処方箋。
2. **Q: RMSE(Z) で Poisson の方が良いなら誤指定でも問題ないのでは？**
   A: RMSE(Z) は事後サンプルベースで、事後分散を過小評価する過信モデルに
   有利に出る指標特性がある（機構は inference、coverage 実験を計画）。
   予測尤度・w 推定・不確実性較正では NB が優る。「どの損失で評価するか」
   自体が主張の一部。
3. **Q: pair mask の実装は正しいと言えるか？**
   A: (a) mask なしで既存 fixed 版と数値一致、(b) held-out ペアの Y を
   書き換えても学習結果が bit 単位で不変、の 2 テストで保証（全 PASS）。
4. **Q: なぜ r を推定せず固定にした？**
   A: r 固定なら NB2 は指数型分布族に留まり枠組みが崩れない。moment 推定の
   two-stage が oracle と te_ll 差 ≤0.01/pair で十分と実証済み。プロファイル
   尤度推定は M-step への 1 次元最適化追加で可能、今後の課題として明記。
5. **Q: NB は正準リンクではないが ExpFam の枠組みと矛盾しないか？**
   A: log link は非正準なので score に重み r/(μ+r) が付く。これは
   「正準 ExpFam → 一般リンク + Fisher scoring」への自然な拡張であり、
   Newton の重みを Fisher 情報にすることで半正定値性も保たれる。
   むしろ枠組みの拡張可能性を示す実例（新規性は literature check required）。
6. **Q: 共有 Z が実データで効かないなら提案モデルの意義は？**
   A: (a) 人工 A/B では明確に効く、(b) 効かない場合でも害はほぼない、
   (c) 「効くかを ablation で事前検査できる」ことが実務的価値。また
   MovieLens はジャンル d=19 の弱い X であり、Cora（語彙 d=50+）など
   X が豊かなデータでの検証が残っている。
7. **Q: MovieLens の Y は「関係データ」と呼べるのか（投影カウント）？**
   A: 限界として明記。user-node（案B）・二部グラフ（案C)への拡張が
   per-column 実装（属性混在対応）とセットで次フェーズの候補。
8. **Q: k=5 で Poisson が 1 fit 発散したのは実装バグでは？**
   A: 旧来からの Newton の既知の脆弱性（NaN ガード・リトライで対処してきた
   もの）で、大 μ ペアの曲率 μ が非有界なことに起因。NB の Fisher 重みは
   r で飽和するため同条件で安定 — バグではなく family 選択の頑健性差として
   報告する（当該 fit は summary CSV で識別可能にしてある）。
9. **Q: 従来の masked evaluation の結果（Pearson 0.96）は撤回するのか？**
   A: 撤回ではなく下方修正。strict でも 0.93–0.95 であり結論の方向は同じ。
   楽観量（+0.09〜0.18 nats/pair）を定量化したことが前進。
10. **Q: 全列 Gaussian 強制がほぼ無害なら per-column は不要では？**
    A: 本デモの範囲ではそう見えるが、(a) bernoulli 強制は崩壊する
    （どの族が「安全」かは事前に自明でない）、(b) Gaussian 強制は尤度・BIC の
    比較可能性を失い family/k 選択が壊れる、(c) 予測分布（カウントの区間予測
    等）が正しくない。点推定の頑健性と統計モデルとしての妥当性は別。
11. **Q: 条件付き過分散が k に依存する（k=3 で 1.14、k=5 で 0.76）のは問題では？**
    A: それ自体が発見。family 診断と k 選択は交絡しており、「k を増やして
    過分散を吸収する」ことと「NB で吸収する」ことのトレードオフがある。
    修論では k×family の同時選択問題として整理する。
12. **Q: PPC の p 値 0.15 で「Poisson で良い」と言い切れるか？**
    A: 言い切らない。plug-in（in-sample μ̂）なのでモデルに有利な保守的検定。
    strict held-out の過分散（k=3 で 1.34）は軽度の破れを示しており、
    「大部分は潜在構造で説明され、残差は軽度」が正確な表現。
13. **Q: 0.5 係数問題は今回の結果に影響しないか？**
    A: 新実験はすべて fixed 系列（E-step 0.5 なし）で統一（KI-002 遵守）。
    なお fixed 版にも `calc_w0`/`calc_w`/`calc_log_likelihood_Y` に 0.5/2L
    系係数が残るが、これは全 (i,j) 両方向和の対称性補正（上三角和と等価）で
    あり E-step の spurious 0.5 とは別物（Phase 0 で整理、正式な数式照合は
    残タスク）。
14. **Q: 41.5× は結局使ってよいのか？**
    A: 使える。根拠 CSV・条件を特定し再計算 41.45× で一致（KI-003 解決）。
    ただし「旧 0.5 実装の値、fixed 版では 40.37×（別条件）」の脚注が必須で、
    最悪「条件」の同定は実装・試行間で不安定なので条件ラベルは主張しない。
15. **Q: held-out 対数尤度は事後予測分布ではなく plug-in だが公平か？**
    A: 全 family 同一条件の plug-in なので相対比較は公平。絶対値は
    事後不確実性を無視する分楽観的。Laplace 事後からのサンプル平均化で
    改善可能（今後の課題）。
16. **Q: なぜ Cora でなく MovieLens を主軸にした？**
    A: 過分散カウント Y（本フェーズの主題）は MovieLens にしかない。
    Cora は Bernoulli-Y で疎性・BIC 問題（KI-011）という別の主題。
    共有 Z ablation の Cora 版は次フェーズ最優先候補。
17. **Q: この結果は「3×3 で足りるか」にどう答える？**
    A: 3 つの不足を実証: (a) 過分散（NB 軸 = Poisson 行の連続拡張）、
    (b) 混在属性（per-column）、(c) 欠測/評価（pair mask は分布族ではなく
    観測モデルの拡張）。一方で「単純に族を増やす」より「診断→選択の手続き」
    が重要という方向性も示した（周辺診断の誤誘導）。
18. **Q: 修論の章としてどう組む？**
    A: §10 の章構成案参照（診断方法論 → 誤指定の定量化 → NB 処方 →
    共有 Z 検証 → 拡張設計）。

## 8. 次にやるべき実験（優先順）

1. **Cora 共有 Z ablation（strict held-out 化含む）** — 共有 Z が「効く」
   実データ例の確保。X=BoW と Y=引用の重なりは強いはず（inference）
2. **posterior coverage 実験** — RMSE(Z) 逆転の機構検証
   （Poisson の事後分散過小評価を直接測る）
3. **zero 過剰・低 r̂ の実データ**（共起カウント系）での Poisson vs NB
4. **k×family 同時選択実験** — 条件付き過分散の k 依存性（発見 1/11）の体系化
5. r のプロファイル尤度推定、per-column × NB の結合、user-node MovieLens（案B）
6. fixed 版残存 0.5/2L 係数の数式照合（Phase 0 の unknown 解消）

## 9. git 状態

- 作業前: branch `main`、未追跡 3 件（cleanup_audit 系）のみ
- 作業後: branch `research/overdispersion-z-ablation`、
  変更 1 件（`EXPERIMENT_REGISTRY.md` 追記のみ）+ 新規 60 件超
  （experimental 9、tools 7、reports 9、results CSV 19、figures 16）
- 既存の結果 CSV・図・原稿・モデルコードへの変更・削除・移動: **なし**
- コミットは未実施（ユーザ判断待ち。推奨: 本ブランチでコミット後、
  main への PR でレビュー）
