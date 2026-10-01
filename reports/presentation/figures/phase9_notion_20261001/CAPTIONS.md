# Figure captions — Notion 発表用ページ（2026-10-01）

対象ページ: `reports/presentation/phase9_progress_notion_presentation_20261001.md`
仕様: `reports/presentation/phase9_progress_notion_figure_spec_20261001.md`
（Figure 4 は作成していない。発表版では比較表で代える。）

---

## Figure 1 — 共通潜在構造から属性と関係を生成

**Caption**: 各対象 i の K 次元の連続潜在ベクトル $z_i\sim\mathcal N(0,I_K)$ から、属性 $x_{il}$（自然パラメータ $f_l^\top z_i$）と、対象の組の関係 $y_{ij}$（自然パラメータ $w_0+w\,z_i^\top z_j$）がともに生成される。

- **Source**: `CLAUDE.md` §1、`RESEARCH_MASTER.md` §2・§4（概念図。データは使っていない）
- **Interpretation boundary**: $z_i$ はクラス・クラスタ ID ではない。X 側に切片はなく、$w_0,\ w$ はスカラー。

## Figure 2 — 属性分布族の選択の流れ

**Caption**: support gate は尤度を使わずに、観測値の形から各属性列の候補を決める。score で比較するのは 0/1 列の Bernoulli と Poisson だけで、同じ E-step のサンプルの上で候補ごとに loading を最適化して比べる。探索 MCEM の後に割り当てを固定し、fresh refit の結果を報告する。

- **Source**: `expfam/src/experimental/family_selection.py`（`support_gate`, `column_log_likelihood`, `select_from_records`, `run_hybrid_family_selection`）（概念図）
- **Interpretation boundary**: 3 つの分布族をすべて score で比較しているわけではない。分布族の自動選択一般の成功を示す図ではない。

## Figure 3 — 分布族 + K の同時選択

**Caption**: 候補の K = 1〜5 のそれぞれで、分布族の選択から fresh refit までを最初から実行し、既存の Q 型基準 $C_Q(K)$ が最小の K を選ぶ。K の間で warm start や分布族の割り当ての持ち越しはしない。右下は 1 条件・3 dataset での実行例（選ばれた K は 2, 3, 3、真の K = 3）。

- **Source**: `expfam/src/experimental/run_joint_family_k_selection.py`、実行例は `expfam/results/joint_family_k_selection/gate75b_20260927/joint_selection.csv`（start_B の行）
- **Interpretation boundary**: 実行例は 3 dataset の記述で、K 選択の成功率を示すものではない。$C_Q$ は Schwarz BIC ではない（Q-based complete-data / ICL-type）。

## Figure 5 — 真の潜在次元を変えたときの正解数

**Caption**: 信号の平均的な強さ（平均 X loading エネルギー 0.5、$w^2K=3$、母集団の平均 edge 確率 0.3314）を揃えた固定した人工データ設計で、各 K_true について真の K を選んだ dataset 数。各条件 10 dataset。K_true = 3 は baseline 20 dataset のうち rep01..rep10。

- **Source**: `expfam/results/matched_k_true_sensitivity/phase9x_20260928/combined/confusion_Lap.csv`, `confusion_Q.csv`
- **Interpretation boundary**: 一般的な回復確率を示すものではない。C_Lap が一般に優れることも示さない。K_true = 3 の点（10/10・7/10）は baseline 20 dataset 版（19/20・14/20）とは別の数。

## Figure 6 — K = 3 と K = 4 の境界

**Caption**: 真の K = 4 の 10 dataset について、保存済みの値から K = 4 が K = 3 に勝つ条件を図示した。点が閾値より上なら K = 4。C_Q（左）の閾値は P_Z の増分 212.84 とパラメータ罰則 38.86 の和 251.70 で一定、C_Lap（右）の閾値は dataset ごとの体積の増分 + 38.86。閾値を上回ったのは C_Q 1/10、C_Lap 8/10。

- **Source**: `expfam/results/k34_boundary_decomposition/phase9y_20260929/ktrue4_rows.csv`
- **Interpretation boundary**: 算術的な分解であり、過小選択の原因（「P_Z が原因」など）を示すものではない。左右の当てはまりの改善は定義が違う（左: 事後サンプル平均、右: 同時最頻値）ので直接比べない。C_Q が K = 1・2 を選んだ dataset はこの図の比較だけでは説明されない。
