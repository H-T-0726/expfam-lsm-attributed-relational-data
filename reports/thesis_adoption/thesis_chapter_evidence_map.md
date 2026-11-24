# 修士論文の章構成案と証拠の対応

- 作成日：2026-11-24。branch `docs/thesis-adoption-20261124`（base `d383cb5`）。新しい実験はしていない。
- **章構成は決定しない。** 依頼された第一候補（7 章構成）を、既存の証拠に照らして評価した案である。
- Claim ID は `claim_adoption_matrix.md`、系列は `implementation_lineage_adoption.md` を参照。
- 既存の構成案（`reports/thesis/thesis_detailed_outline_20260905.md`、`thesis_storyline_20260905.md`）は 2026-09-05 時点の記録で、AF・M1・CORA-SH1 より前のもの。本資料はそれらを書き換えない。

## 1. 第一候補（7 章）の評価

| 章 | 内容 | 使う証拠（Claim ID） | 系列 | 区分 | 評価 |
|---|---|---|---|---|---|
| 1 序論 | 属性付き関係データ、先行研究の固定分布、研究目的（挙動・条件・限界の評価） | `RESEARCH_MASTER.md` §1 | — | — | 目的文と整合。「有効性を示す」と書かない |
| 2 関連研究 | Mikawa et al. 2024、潜在空間モデル、GLM・指数型分布族、モデル選択（BIC・ICL・Laplace） | TH-3（原論文の 1/2） | A | — | 混合モデル・潜在変数モデルの分布選択の先行研究は未調査（AF を章にするなら必要） |
| 3 指数型分布族への拡張と推定法 | 生成モデル、MCEM＋Laplace、精度行列・勾配の一般式、1/2 の扱い、分散関数、C_Q の位置づけ、識別可能性（population） | TH-1〜TH-6 | A 理論、実装は C | **A** | **最も堅い章。** TH-5 は前提（canonical、generic rank、Gaussian-Y）を明記し、Bernoulli-Y は未解決と書く |
| 4 人工データによる評価 | 先行研究の再現、n 依存、d 依存、誤指定、K 選択（記述）。拡張モデルでの誤指定と Y 情報量（任意） | SY-1〜SY-6（主）、MS-1・MS-2・KS-2（条件付き） | **C**（主）、E（条件付き） | B・C | 主結果は fixed（C）で書ける。M1/E1/#168 は**列ごとの family の拡張モデル**なので、入れるなら「拡張設定での追加分析」として節を分け、C の数値と同じ表に置かない |
| 5 実データによる評価 | Cora strict held-out（CORA-SH1）、旧 Cora・Wine・MovieLens の経緯と限界 | CO-1〜CO-3（条件付き）、RD-1〜RD-3（補助） | E-scalar、C | C・D | **主結果は CORA-SH1 のみが strict held-out**。ただし lineage E の昇格（Human Gate）が前提。旧 Cora は transductive 設計（test 辺を 0 として学習）なので、経緯または付録にとどめる |
| 6 分布選択の発展的検討と限界 | 列ごとの family、自動選択（AF1〜AF8-v3）、Q の偏り、事前登録確認の未達成、prior 依存 | SY-8、AF-1〜AF-4、TH-9、TH-10 | E（拡張）、PR #129 | D | すべて正式モデルの範囲外。否定的結果を正確に書く必要がある（§2） |
| 7 考察と結論 | 主張の境界、切片なし仕様の影響、未解決（Bernoulli-Y の識別、一致性、MCEM 収束）、今後の課題 | TH-7、TH-8、MS-3、CO-2、RD-6 | — | E | CO-2（中心化で逆転）と RD-6（raw-count Poisson の悪化）は「X の平均・尺度の扱い」という共通の論点として書ける。**ただし原因の確定はできない** |

**評価（JUDGMENT）**：7 章構成は研究目的（挙動・条件・限界の評価）と既存の証拠に合っている。ただし次の 3 点の調整が要る。

1. **第 4 章の主結果は fixed 系列（C）で書く。** 誤指定と Y 情報量の交互作用（M1/E1/#168）は拡張モデルの結果なので、第 4 章の末尾の節か第 6 章に分ける。
2. **第 5 章は lineage E の扱いで中身が変わる。** E-scalar を「strict held-out 評価のための実装」として認めないなら、第 5 章は「旧 Cora・Wine・MovieLens（いずれも fixed、in-sample または transductive）＋限界」になり、strict な予測の主張はできない。
3. **第 6 章の題は「発展的検討と限界」とし、成功の章に見せない。**

## 2. AF を独立章にする場合と付録にする場合

| 観点 | 独立章（第 6 章） | 付録＋第 7 章で要約 |
|---|---|---|
| 先生の質問（「分布が未知なら？」）への応答 | 正面から答えられる | 1 段落の要約になり、弱く見えるおそれ |
| 研究目的との整合 | 「限界の評価」として整合 | 整合（主題がより明確） |
| 正式モデルとの関係 | **範囲外の拡張**であることを章の冒頭で明記する必要 | 範囲外であることが構成上明確 |
| 必要な追加作業 | 関連研究の調査（潜在変数モデルの分布選択）、否定的結果の書き方 | ほぼなし |
| 過大主張の危険 | 中（書き方次第で「失敗の羅列」か「成功」に見える） | 低 |
| 分量 | 大きい（AF1〜AF8-v3 は 8 段階） | 小さい |
| 事前登録と未達成の記録の価値 | 手続きの信頼性を示す材料として活かせる | 活かしにくい |

**推奨（JUDGMENT）**：AF1・AF2（{0,1} 列と近い分布での Q 選択）、AF6（事前登録確認の未達成）、AF8-v3（prior 依存、NO-GO）の 4 点に絞った**短い独立章**。AF3〜AF5・AF7 の詳細は付録に回す。研究者と指導教員の判断が要る。

## 3. 本文に必要な数式・図・表（候補）

| 種類 | 内容 | 根拠 | 章 |
|---|---|---|---|
| 数式 | 生成モデル（z、y、x）、θ | `AGENTS.md` §1 | 3 |
| 数式 | 完全データ対数尤度と Q 関数、MCEM の流れ | `RESEARCH_MASTER.md` §5 | 3 |
| 数式 | E-step の勾配と精度行列 A_i（V=A''/φ、extra 1/2 なし）と導出の要点 | §6、`docs/math_notes/half_factor_math_explanation.md` | 3 |
| 数式 | M-step（Gaussian-X の閉形式 F・Σ、非 Gaussian の F、w0・w） | コード（`reproduction/src/model.py`、`model_dual_expfam.py`） | 3 |
| 数式 | C_Q = −2Q_strict + p̂ ln n と P_Z、Schwarz BIC との違い | §12.6・§17.4 | 3 |
| 数式 | 識別可能性 P1・P2（前提つき）、P7（否定的結果） | §17.3 | 3 または付録 |
| 表 | 先行研究と本研究の対応（family、E-step、K 選択） | 総括 `advisor_meeting_brief.md` §2 | 1・3 |
| 表 | 1/2 の 5 系統 | §6.1 | 3 |
| 表 | 先行研究再現（k_est=3・4 の RMSE(Z) 差） | SY-2 | 4 |
| 図 | n 依存（fixed、3 シナリオ） | SY-3（`fixed_official/exp2/`）。**既存の提出用図 fig1a は B 系列なので流用しない** | 4 |
| 表または図 | 誤指定の倍率（fixed、両側／Y のみ／X のみを分ける） | SY-5 | 4 |
| 表 | K 選択（fixed、30/30、限定文つき） | SY-6 | 4 |
| 図 | d 依存（シナリオ依存） | SY-4 | 4 または付録 |
| 表 | CORA-SH1 の条件（split、mask、arm、予算、未収束の明記） | CO-1 | 5 |
| 図 | CORA-SH1 primary Δ（split × seed） | `analysis/fig_primary_delta_K6.png`（集計スクリプトで再生成可能） | 5 |
| 図 | CORA-SH1 secondary（K=3、GXc、YO との差） | `analysis/fig_secondary_deltas.png` | 5 |
| 表 | 推定量ごとの Δ（mc_predictive、mean5、last） | `analysis/contrast_summary.csv` | 5 |
| 表 | AF の 4 段階（正解したか／比較手法より良いか／新データで再現したか／事前登録を達成したか） | 総括 `evidence_matrix.md` §B0 | 6 |
| 表 | 主張の境界（書けること／書けないこと） | Ledger §14 以降 | 7 |

## 4. 系列を混ぜないための配置ルール（本文作成時）

- 第 4 章の表・図は fixed（C）だけで作る。B 系列（学会予稿）の数値は付録の対応表で別に示す（条件対応表は存在しないので、最悪比を並べるだけにしない）。
- 第 5 章の CORA-SH1（E-scalar＋Newton-F）と旧 Cora（C）を同じ表に置かない。
- 第 6 章は列ごとの family を前提にした拡張モデル（E per-column、PR #129）であることを章の冒頭で明記する。
- K 選択は、C_Q（ICL 型）と held-out 予測スコア（Phase 7e/8b）と C_Lap を区別して書く（KI-019）。
