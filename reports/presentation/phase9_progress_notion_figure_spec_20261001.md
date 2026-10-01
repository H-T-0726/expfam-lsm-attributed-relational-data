# Notion 進捗資料 — Figure 1〜6 の仕様（2026-10-01）

- 対象ページ: `reports/presentation/phase9_progress_notion_draft_20261001.md`（各 `[FIGURE n HERE]` の位置）
- 根拠: Phase 1 の storyboard（図案）、experiment inventory（Table A〜G）、equation inventory（E1〜E13）。数値は primary artifact で確認済みのものだけ使う。
- 画像はまだ作っていない。作るときは承認済みの script から生成し、数値を手で入力しない（データ図は下記 source の CSV を読む）。
- 共通の約束:
  - 系列は lineage E（experimental prototype）。図の脚注に「固定した人工データ条件」を入れる。
  - 縦軸に件数を描くときは「件数（分母）」を軸ラベルか注記に書く。割合・確率・% にしない。
  - 誤差棒・信頼区間は描かない（計算していない）。
  - C_Lap を「正解」と読める色（緑など）、C_Q を「誤り」と読める色（赤など）にしない。2 基準は中立な 2 色（例: 青系・橙系）。
  - Phase 番号を図に入れない。

---

## Figure 1 — 共通潜在構造 Z から X / Y が生成される図

| 項目 | 内容 |
|---|---|
| 配置 | §1.1 |
| purpose | 1 つの連続潜在ベクトル $z_i$ が、属性 $x_i$ と関係 $y_{ij}$ の両方を生むことを示す |
| 種類 | 概念図（グラフィカルモデル風）。データは使わない |
| 構成 | 左に対象 i・j の $z_i, z_j$（丸）。$z_i$ → $x_{i1},\dots,x_{id}$（四角、d 列）。$z_i, z_j$ → $y_{ij}$（四角）。 |
| ラベル | 矢印に $\eta_{il}^X=f_l^\top z_i$、$\eta_{ij}^Y=w_0+w\,z_i^\top z_j$。分布名は従来版「Gaussian」「Bernoulli」、拡張版「ExpFam_X」「ExpFam_Y」（同じ図の 2 バージョンにしてもよい） |
| annotation | 「$z_i\in\mathbb R^K$（連続）、$z_i\sim\mathcal N(0,I_K)$」 |
| source | `CLAUDE.md` §1、`RESEARCH_MASTER.md` §2・§4、equation inventory E1〜E3 |
| caution | $z_i$ をクラス・クラスタ ID として描かない（離散のアイコンや色分けしたグループを使わない）。X に切片を描かない。$w$ を行列として描かない |

## Figure 2 — family-selection flow

| 項目 | 内容 |
|---|---|
| 配置 | §3.4 |
| purpose | support gate で候補を絞り、0/1 列だけ score で Bernoulli か Poisson を選び、最後に割り当てを固定して fresh refit した結果を報告する流れを示す |
| 種類 | 縦のフローチャート |
| 構成 | (1) 観測 X, Y → (2) support gate（3 分岐: 0/1 のみ → {Bernoulli, Poisson} を score／2 以上を含む非負整数 → Poisson 固定／その他 → Gaussian 固定）→ (3) 探索 MCEM のループ枠「E-step で Z サンプル（L = 5）→ 0/1 列で両候補の loading を別々に最適化 → score 比較 → 分布を更新」（ループ回数「8 反復」）→ (4) 割り当てを固定 → (5) fresh refit（8 反復）→ (6) 報告 |
| ラベル | gate に「尤度を使わない」、score に「完全な対数確率（$-\log x!$、$-\frac12\log 2\pi$ を含む）」、(6) に「報告するのはこの値」 |
| annotation | 「score で決めるのは 0/1 列だけ」 |
| source | `expfam/src/experimental/family_selection.py`（`support_gate`, `column_log_likelihood`, `select_from_records`, `run_family_exploration`, `run_hybrid_family_selection`）、equation inventory E4〜E6 |
| caution | 3 分布すべてを score で比較しているように描かない。ループが収束まで回るように描かない（探索 8 反復・refit 1 回）。探索中の値を報告値に見せない |

## Figure 3 — K = 1..5 joint selection pipeline

| 項目 | 内容 |
|---|---|
| 配置 | §5.1 |
| purpose | 候補の K ごとに独立に分布選択と refit を行い、$C_Q(K)$ の最小で K を選ぶことを示す |
| 種類 | 並列レーンの概念図 |
| 構成 | 観測 X, Y から 5 本の並列レーン（K = 1, …, 5）。各レーン: gate → family selection → freeze → fresh refit → $C_Q(K)$。右端で 5 つの $C_Q(K)$ を集めて $\hat K=\arg\min_K C_Q(K)$ |
| ラベル | レーンの間に「持ち越しなし（warm start なし）」 |
| annotation（任意） | 3 dataset の選ばれた K: 2, 3, 3（K_true = 3）。付けるなら「1 条件・3 dataset」も併記 |
| source | `expfam/src/experimental/run_joint_family_k_selection.py`、`expfam/results/joint_family_k_selection/gate75b_20260927/joint_selection.csv`、equation inventory E7 |
| caution | レーン間に矢印を引かない。2 つの開始点を独立な dataset として描かない。「6 回中 4 回成功」のような確率的な表現を付けない |

## Figure 4 — C_Q vs C_Lap comparison

| 項目 | 内容 |
|---|---|
| 配置 | §7.4 |
| purpose | 2 つの基準を「当てはまり + Z に関する項 + θ の罰則」で並べ、違いが (1) Z に関する項が一定かデータ依存か、(2) 当てはまりの評価点、にあることを示す |
| 種類 | 概念的な積み上げ棒 2 本（**数値なし**） |
| 構成 | 左 C_Q: 下から $D_K=-2(Q_X+Q_Y)$（「事後サンプルの MC 平均」）／ $P_Z=nK(1+\log 2\pi)$（斜線、「一定」）／ $P_\theta=p_K\log n$。右 C_Lap: $-2[\ell_X(\hat Z)+\ell_Y(\hat Z)]$（「joint mode」）／ $\lVert\hat Z\rVert^2+\log\lvert H\rvert$（点線、「データで決まる」）／ $d_K\log N$ |
| ラベル | 左上「Z を積分しない（Q 型・ICL 型）」、右上「Z を Laplace 近似で積分」 |
| annotation | 「$P_Z$: 1 次元あたり 212.84（n = 75）」「どちらも厳密な周辺尤度ではない」 |
| source | equation inventory E8〜E11、`eval_utils.calc_bic_exp`、`laplace_k_criterion.py` |
| caution | 棒の高さに実際の数値を使わない（概念図）。どちらかを正解と読める色にしない。C_Q を「BIC」と書かない |

## Figure 5 — K_true vs exact count

| 項目 | 内容 |
|---|---|
| 配置 | §10.3 |
| purpose | 信号を揃えた設計で、K_true が大きいほど exact が減り、C_Q で顕著なことを示す |
| 種類 | 点と線（2 系列） |
| 横軸 | K_true（1, 2, 3, 4）。離散の 4 点 |
| 縦軸 | 「真の K を選んだ dataset 数（各 10 中）」、範囲 0〜10 |
| 系列 | C_Lap: 10, 10, 10, 8。C_Q: 10, 9, 7, 1。各点に数値ラベル（「8/10」など） |
| annotation | 「各点 10 datasets」「K_true = 3 は baseline 20 dataset のうち rep01..rep10」「揃えた量: loading エネルギー 0.5、$w^2K=3$、平均 edge 確率 0.3314」 |
| 補助パネル（任意） | 選ばれた K の分布の積み上げ棒（K_true × 基準）。C_Lap: 10/0/0/0/0, 0/10/0/0/0, 0/0/10/0/0, 0/0/2/8/0。C_Q: 10/0/0/0/0, 1/9/0/0/0, 1/2/7/0/0, 1/5/3/1/0 |
| source | `expfam/results/matched_k_true_sensitivity/phase9x_20260928/combined/confusion_Lap.csv`, `confusion_Q.csv` |
| caution | 縦軸を「回復率」「確率」「%」にしない。誤差棒を描かない。K_true = 5 以上へ線を延ばさない。K_true = 3 の点を baseline の 19/20・14/20 と混ぜない |

## Figure 6 — K3→K4 threshold balance

| 項目 | 内容 |
|---|---|
| 配置 | §11.3 |
| purpose | K_true = 4 の 10 dataset で、C_Q は一定の閾値 251.70 に fit gain が届かない dataset が多く（届いたのは 1/10）、C_Lap は dataset ごとに閾値（体積の増分 + 38.86）が変わり 8/10 で fit gain が上回ったことを示す |
| 種類 | 2 パネルの点図（左 C_Q、右 C_Lap）。縦軸のスケールは揃えてよいが、別パネルにする |
| 横軸 | dataset（rep01..rep10） |
| 縦軸 | −2 × 対数尤度の単位 |
| 左パネル（C_Q） | 点 = fit gain_Q（204.3, 204.9, 158.9, 162.0, 243.7, 151.7, 165.7, 158.4, 318.0, 163.4）。水平線 = 251.70（「212.84 + 38.86、一定」） |
| 右パネル（C_Lap） | 点 = fit gain_Lap（258.5, 268.6, 220.0, 243.9, 312.9, 208.2, 245.5, 251.0, 377.1, 276.3）。各 dataset の短い横棒 = 体積の増分 + 38.86（体積の増分: 204.6, 208.6, 182.2, 193.3, 228.0, 178.9, 198.4, 203.1, 251.7, 207.0） |
| 読み方 | 点が線・棒より上 → K = 4 が K = 3 に勝つ |
| annotation | 「上回った数: C_Q 1/10、C_Lap 8/10」「新しい推定なし（保存済みの値の分解）」「fit gain の評価点: C_Q は事後サンプル平均、C_Lap は joint mode」 |
| source | `expfam/results/k34_boundary_decomposition/phase9y_20260929/ktrue4_rows.csv`（列 `fit_gain_Q_34`, `fit_gain_Lap_34`, `volume_increment_34`, `margin_Q`, `margin_Lap`）、equation inventory E12・E13 |
| caution | 「P_Z が原因」と読める矢印・注記を付けない。左右の fit gain を同じ量として 1 本の軸で重ねない（定義が違う）。C_Q が K = 1・2 を選んだ 6 dataset がこの図で説明されるように見せない。図の数値は CSV から読む（上の値は丸めた確認用） |
