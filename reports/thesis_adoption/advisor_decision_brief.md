# 指導教員と決めること：本文採用の範囲（判断資料）

- 作成日：2026-11-24。branch `docs/thesis-adoption-20261124`（base `d383cb5`）。新しい実験はしていない。
- **この資料は判断材料であり、決定ではない。** 採用と研究方針は研究者と指導教員が決める。
- 詳細：`claim_adoption_matrix.md`（主張ごとの判定）、`implementation_lineage_adoption.md`（実装系列）、`thesis_chapter_evidence_map.md`（章と証拠）。
- 第一候補の方向性：「指数型分布族への拡張を本論の中心とし、分布誤指定・実データの挙動・自動分布選択の可能性と限界を追加分析として扱う」（**承認済みではない**）。

## 1. 正式な提案手法に含める範囲（案）

- **含める**：全 X 列で共通の 1 family（Gaussian / Bernoulli / Poisson）、Y も 3 family、X 切片なし、スカラー w0・w。推定は MCEM＋Laplace。E-step は採用式（V=A''/φ、extra 1/2 なし）。
- **実装の主系列**：fixed（`model_dual_expfam_fixed.py`、lineage C）。採用式と一致する。
- **含めない**：列ごとの family（per-column）、自動分布選択（AF）、COM-Poisson（PR #129）、NB。いずれも正式モデルの範囲外の拡張。
- **判断が要る**：lineage E の scalar 版（`DualExpFamLSMConsistent`）。数式は正式モデルと同一で、数値は C より採用式に忠実（clip なし）。strict held-out（pair mask）はこの系列にしかない。

## 2. 本文の主要な実験結果（案、区分 B）

| 結果 | 系列 | 数値（一次 CSV で再確認） |
|---|---|---|
| 実装レベルの一般化 | B（C も継承） | `test_dual_expfam.py` 5 テスト |
| 先行研究条件での再現実装との一致 | B | RMSE(Z) 差 0.00054（k_est=3）、0.00026（k_est=4）。**採用式の検証ではない** |
| n 依存 | C | n=50→300 で A −49.3%、B −41.2%、C −58.6% |
| 誤指定による悪化 | C | 最悪 A 4.34×、B 9.04×、C 40.37×（最大の 2 つは X・Y 両側の誤指定） |

## 3. 限定付きで本文採用する候補（区分 C）

- K 選択（fixed、C_Q が 30/30 で K=3）。「Schwarz BIC ではない」「一般の K 回復ではない」を必ず付ける。
- d 依存（シナリオ依存：A は改善、B は U 字、C は平坦）。
- 拡張モデルでの誤指定と Y 情報量（M1・E1・PR #168）。**列ごとの family（G3/B6/P3）の結果**であり、正式モデルの結果ではない。
- clean true-K n-sweep（E-scalar）。
- **新 Cora strict held-out（CORA-SH1）**。E-scalar の昇格が前提。

## 4. 探索的検証・付録の候補（区分 D）

- 学会予稿の B 系列の数値（予稿との対応として）。
- per-column（complementary blocks、latent coverage）。
- K 選択の held-out pilot（Phase 7e/8b）、C_Lap（Phase 9）、d-sweep・R1/R2。
- Wine、旧 Cora、MovieLens（in-sample と user-disjoint）、raw-count Poisson の悪化。
- AF1〜AF8-v3（独立章にするなら短く 4 点に絞る）。

## 5. 採用できない結果・主張（区分 F と禁止表現）

- AI 生成レポート（KI-007）、Phase 8b Attempt 1、生成元不明の図（KI-004）を根拠にすること。
- **旧 Cora を「strict held-out で予測できた」と書くこと**。今回の一次確認で、test 辺を Y_train=0 として学習し、全グラフの次数で subset を選び、NaN 時に seed を変えて retry する設計だった（`expfam/src/run_fixed_real_cora_heldout_link_prediction.py` L.150–155、L.223–238、L.262–267）。`KNOWN_ISSUES.md` の「今すぐ主張してよいこと」と書き方が合わないので、扱いを決める必要がある。
- 「Schwarz BIC で K を選んだ」「一致性」「Bernoulli-Y で K を識別できる」「per-column / 自動選択が一般に優れる」「分布族そのものの優越（Cora）」「MovieLens で有効性を確認」「0.5 除去の効果（41.45×→40.37×）」。

## 6. 今回の Cora 実験（CORA-SH1）の位置づけ

- **何をしたか**：full Cora（2708 ノード、上位 50 語）。全ペアの 20% を test にする dyad split を 5 つ作り、test を mask で学習から完全に除外した（Y=0 として学習しない）。BX / GX raw / Y-only（K=6 primary、K=3 sensitivity）と GXc（K=6 secondary）を、3 seed × 5 split で比較した。全 105 fit が成功。protocol は実行前に凍結している（`74f874b`）。
- **結果**（生 CSV から再計算して確認）：
  - primary Δ（BX − GX raw、K=6、mc_predictive log score）= +1.06×10⁻⁴ nats/pair（13/15 対、4/5 split）。K=3 でも +1.14×10⁻⁴。
  - **GXc（中心化 Gaussian）が BX を上回った**（−4.4×10⁻⁵、BX が良いのは 3/15）。
  - K=6 では X を使う arm が Y-only を 15/15 で上回り、K=3 ではその効果がなかった。
  - 最終サンプル 1 つの plug-in では Δ = −1.7×10⁻⁵ で向きがそろわない。
- **技術的な限定**：EM は 8 反復で未収束（BX の Q 増分が最大）。BX の F は Newton 法に変更した（GX の閉形式と最適化の質をそろえるため）。統計単位は Cora 1 ネットワーク。`fit_technical.csv` に列ずれの記録不具合（解析スクリプトで決定的に復元、生ファイルは不変）。npz の状態は git 管理外（sha256 は記録済み）。commit は local のみ（push していない）。
- **位置づけ（JUDGMENT）**：区分 C。本研究で唯一の「正式モデルと同じ観測モデル・strict held-out・事前凍結 protocol」の実データ結果。言えるのは「現行仕様・固定予算で、X family の違いによる予測の差は小さく、中心化の有無で逆転する」まで。**分布族の優越は主張しない。** 「X の平均・尺度の扱い（切片なし仕様）が結果を左右しうる」という論点は、MovieLens の raw-count Poisson（原因未解明）と同じ種類の限界として第 7 章で議論できるが、原因の確定はできない。

## 7. 本文に必要な数式・図・表

`thesis_chapter_evidence_map.md` §3 を参照。要点：
- 第 3 章：生成モデル、Q 関数と MCEM、E-step の一般式（1/2 なしの導出）、M-step、C_Q の位置づけ、識別可能性（前提つき）。
- 第 4 章：fixed の再現表・n 依存図・誤指定表（両側／片側を分ける）・K 選択表。**既存の提出用図（fig1a/1b）は B 系列なので流用しない。**
- 第 5 章：CORA-SH1 の条件表、primary 図、secondary 図、推定量ごとの Δ の表。
- 第 6 章：AF の 4 段階の表。
- 第 7 章：主張の境界の表。

## 8. 指導教員に確認すべきこと

1. **主系列を fixed（C）にしてよいか。** 学会予稿（B、0.5 あり）の数値は付録に回してよいか。
2. **lineage E の scalar 版を「strict held-out 評価用の実装」として本文で使ってよいか。** 使わない場合、第 5 章に strict な予測の結果は残らない。
3. **CORA-SH1 を本文の実データ結果として採用するか。** 採用する場合、Newton-F への変更、EM の未収束、K=6 が過去の Cora 実験に基づくことを本文に書く。
4. **M1・E1・PR #168（列ごとの family での誤指定）を本文に入れるか。** 入れるなら「拡張モデルの追加分析」として節を分けるか。PR #168 を merge するか。
5. **AF を独立章にするか、付録にするか。**
6. **旧 Cora の扱い**（transductive 設計）。`KNOWN_ISSUES.md` の該当行を forward update するか。
7. **Bernoulli-Y の K の識別（U2）を未解決のまま書いてよいか。**

## 9. 新たな実験は本当に必要か

**本文のストーリーを書くために必須の新実験はない（JUDGMENT）。** 正式モデルの主結果（C）と理論は揃っている。

検討の余地があるのは 1 点だけ。

- **正式モデル（全列共通 family）での誤指定 × Y 情報量（MS-3）**：現在の Y 情報量との交互作用の証拠（M1・E1・#168）はすべて列ごとの family の拡張モデルにある。先生の質問（「間違った分布を仮定すると何が起きるか」）に正式モデルの範囲で答えたい場合だけ、fixed または E-scalar で小さく行う意味がある。fixed_official Exp4（SY-5）で「誤指定は悪化させる」までは既に答えられるので、**必要かどうかは先生の判断**。

CORA-SH1 の追加（K の変更、反復数の延長、別ネットワーク）は、結果を見た後の条件変更になるので、確認実験として新たに事前登録しない限り行わないほうがよい。

## 10. 推奨する修論の最終ストーリー（JUDGMENT）

> 先行研究の潜在構造モデルを、X 側・Y 側で独立に指数型分布族（G/B/P）へ一般化し、E-step の一般式を独立に導いた（原論文の印刷式との 1/2 の差を含む）。正式モデルの実装（fixed）で、先行研究条件での再現、n 依存、分布誤指定による悪化を人工データで示した。K の選択は、現行基準が Schwarz BIC ではないことを示したうえで、明示した条件での記述にとどめた。実データでは、strict held-out の Cora で X family の違いによる予測の差は小さく、中心化の有無で逆転した。分布をデータから選ぶ拡張（列ごとの family）では、{0,1} 列では選べるが、近い分布では誤りに偏りがあり、それを直す基準は事前登録した確認で支持されなかった。全体として、一般化の価値は「どの条件で分布の指定が効くか・効かないか」を明らかにした点にあり、性能の一般的な向上ではない。

## 最も整合的な採用案（1 つ）

| 区分 | 内容 |
|---|---|
| 正式モデル | 全列共通 family の Dual-ExpFam LSM（G/B/P）、採用式（1/2 なし） |
| 主実装 | fixed（C）。strict held-out には E-scalar を**評価用実装として限定的に**使う（要承認） |
| 第 3 章（A） | TH-1〜TH-6 |
| 第 4 章（B＋C） | SY-1〜SY-6（fixed）。拡張モデルの M1/E1/#168 は第 4 章末の独立節（C、要承認） |
| 第 5 章（C） | CORA-SH1（CO-1〜CO-3）を主、旧 Cora・Wine・MovieLens は経緯と限界として短く |
| 第 6 章（D） | AF1・AF2・AF6・AF8-v3 に絞った短い章（「発展的検討と限界」） |
| 第 7 章（E） | Bernoulli-Y の識別、一致性、MCEM 収束、X の平均・尺度の扱い、切片なし仕様 |
| 付録 | B 系列（学会予稿）の数値、per-column、K 選択の pilot 群、AF の詳細 |

この案は、研究目的（挙動・条件・限界の評価）と、各結果の系列・モデル範囲の両方に合う。採用するかどうかは研究者と指導教員が決める。
