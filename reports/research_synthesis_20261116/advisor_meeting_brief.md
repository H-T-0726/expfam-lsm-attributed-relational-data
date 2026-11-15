# 指導教員との打ち合わせ資料（2026-11-16）

詳しい根拠は、同じフォルダの `evidence_matrix.md`、`af1_af8_timeline.md`、`thesis_direction_comparison.md` にある。リンクはすべて commit 固定の GitHub permalink である。

---

## 1. 研究目的

属性付き関係データの潜在構造モデル（Mikawa et al., 2024）を指数型分布族へ一般化し、その挙動・有効な条件・限界を評価する。

```
z_i  ~ N(0, I_K)
y_ij ~ ExpFam_Y( w_0 + w z_iᵀ z_j )      i < j
x_il ~ ExpFam_X( f_lᵀ z_i )
```

- 先行研究は Y が Bernoulli、X が Gaussian に固定されている。本研究は、それぞれを Gaussian / Bernoulli / Poisson から独立に選べるようにした。
- E-step の精度行列は A_i = I + Fᵀ V_X F + w² Σ_{j≠i} V_Y(η_ij) z_j z_jᵀ、V(η) = A''(η)/φ である。

## 2. これまでの進捗

| 時期 | 内容 |
|---|---|
| 〜2026-05 | 拡張モデルの実装、人工データ 3 シナリオ、学会予稿 |
| 2026-06〜08 | 実データ（Wine、Cora、MovieLens）、per-column prototype、1/2 係数の確認 |
| 2026-09 | 識別可能性の命題、K の選択基準（C_Q と C_Lap）の比較 |
| 2026-10-13 | 先生の質問「間違った分布を仮定すると何が起きるか」を受けて、誤指定の影響を整理し、M1 を実施 |
| 2026-11-04〜10 | E1・E2a（Z の点推定での再評価）、誤指定の拡張（PR #168） |
| 2026-11-09〜16 | 分布の自動選択（AF1〜AF8-v3）。AF8-v3 は NO-GO と判断して終了 |

## 3. 得られた成果

**本文に書ける結果（ALLOWED）**：
- X と Y を独立に指定できる実装。
- 先行研究と同じ条件での一致（5 試行）。
- n を増やすと RMSE(Z) が下がること。
- 誤指定による悪化（fixed 系列。大きな倍率は主に Y 側の誤指定による）。

**理論（population、条件つき）**：
- Poisson-X で FFᵀ を復元できる。
- Gaussian-Y で K が決まり、族は入れ子ではない。
- C_Q は Schwarz BIC ではない。

**誤指定と Y の情報量**（lineage E。本文に使うかは未決定）：
- 真の Bernoulli 属性を Poisson として fit すると、Z の推定誤差は dense で +0.020（15/20）、sparse で +0.066（17/20）増えた。sparse の悪化は別の seed 系列でも再現した。
  - [M1 報告](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reports/advisor_feedback/misspecification_m1_bernoulli_to_poisson_20261013.md)
  - [E1 報告](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reports/research_improvement/e1_z_point_estimator_m1_20261104.md)
- 誤指定の種類によって、Y の情報量との関係の向きが変わる（[PR #168](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/pull/168)）。

| 誤指定 | Y を全部使う | Y を 10% に減らす |
|---|---|---|
| B→P（Bernoulli を Poisson として fit） | +0.012 | +0.082 |
| B→G（Bernoulli を Gaussian として fit） | +0.271 | +0.146 |

**分布の自動選択（AF。すべて prototype）**：
- **自明なケース**（0/1 の列で、Bernoulli か Poisson か）：解析的に Bernoulli が必ず勝つ。そのため自動選択は全列正解し、誤指定の損失をすべて回避した（AF1）。
- **自明でないケース**（Poisson か、過小分散の CMP(ν=2) か）：
  - Q スコアは CMP2→Poisson の向きに誤る。原因は Z の誤差の大きさではなく、Z をどの分布の下で推定したかだった（AF2、AF3）。
  - Laplace 型の基準では、誤りの向きが Poisson→CMP2 に入れ替わった。事前登録した新しい 40 データでの確認では、主要評価項目を達成しなかった（AF6：+0.225 [−0.075, 0.525]）。
  - loading を積分した基準は、数値的には安定して計算できた。しかし改善は判定境界の移動で、prior の尺度に強く依存した（AF8-v3、NO-GO）。

| 手法（主 prior、480 列） | 真 Poisson の正解 | 真 CMP2 の正解 | balanced |
|---|---|---|---|
| Q | 234/240 | 188/240 | 0.879 |
| AF6 Laplace | 200/240 | 228/240 | 0.892 |
| E2（積分） | 223/240 | 216/240 | 0.915 |

E2 − AF6 Laplace の balanced の差は +0.023 [−0.004, +0.050]。

## 4. うまくいかなかった点（正直に）

- 自明でない分布の組では、自動選択の改善を確認できなかった。事前登録の確認は未達成で、最後の探索も NO-GO だった。
- 「拡張して性能が上がった」と言える証拠はない。示せているのは、誤指定の悪影響と、限定的な補完効果である。
- 実データ（MovieLens）での効果は、split 間の変動に比べて小さい。
- Bernoulli-Y での K の識別は未解決。
- 分布の自動選択も per-column も、prototype（列ごとの family）を前提にしている。標準モデルは全列で 1 つの family。

## 5. 今後の方向性（3 案。詳細は `thesis_direction_comparison.md`）

| 案 | 内容 | 実現可能性 | 主な懸念 |
|---|---|---|---|
| A | 拡張を中心に据え、分布は既知とする | 高 | 「分布が未知のとき」に答えていない |
| **B** | 拡張を中心とし、分布自動選択の難しさと限界を追加分析にする | 高 | 否定的な結果の書き方 |
| C | Z の推定と分布選択の相互依存を中心課題に再構成する | 低 | 理論も確認もこれから。主題が変わる |

**私の提案（判断。決定ではない）**：案 B。

- 目的文の「限界の評価」と一致する。
- 先生の質問には 2 段階で答えられる。
  - 誤指定すると何が起きるか（M1・E1・#168）。
  - 分布をデータから選べるか（AF）。
- 新しい実験は不要。

## 6. 先生に相談したいこと

1. **方向性**：案 A・B・C のどれが修士研究として適切か。
2. **prototype の扱い**：lineage E の結果（M1、E1、#168、per-column、AF）を本文に使ってよいか。付録や予備的検討にとどめるべきか。
3. **否定的な結果の価値**：AF の「自明でないケースでは改善しなかった」という結果を、1 章として書く価値があるか。限界の節に圧縮すべきか。
4. **分布既知の前提**：本論で分布を既知とすることを、先生の質問への応答として十分と考えるか。
5. **COM-Poisson**：未 merge の PR #129 の generic model（CMP を含む）を論文で扱ってよいか。
6. **残りの期間の優先順位**：新しい実験はせずに執筆に移るか。追加で検証すべき点（例：誤指定の別条件、Bernoulli-Y の K）があるか。
