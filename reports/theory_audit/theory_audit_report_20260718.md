# Dual-ExpFam LSM 理論監査報告書

作成日: 2026-07-18
ブランチ: `research/story-diagnostics`（HEAD `3fe24b6`、dirty worktree、既存ファイル無変更）
監査者: Claude (Fable 5)、`docs/theory_audit/CLAUDE_FABLE_5_THEORY_AUDIT_MASTER_PROMPT.md` に基づく読み取り専用監査
（本監査の実施時点では同ファイルはリポジトリ root に未追跡で置かれていた。
2026-08-17 の統合コミットで `docs/theory_audit/` へ移動して追跡下に入れた。内容は無変更）
位置づけ: **本報告はリポジトリ変更を伴わない理論監査であり、新規作成は本報告3ファイルのみ。**

証拠ラベル凡例:
[CONFIRMED_IN_REPOSITORY]=リポジトリ内のコード・CSV・文書で直接確認 /
[DERIVED]=本監査で導出（導出過程を明記）/ [SUPPORTED_BY_PRIMARY_SOURCE]=一次文献で確認 /
[EMPIRICALLY_OBSERVED]=既存実験結果と整合 / [PLAUSIBLE]=もっともらしいが未証明 /
[UNRESOLVED]=未解決 / [CONTRADICTED]=矛盾を検出 / [OUT_OF_SCOPE]=本監査の範囲外

関連ファイル:
- 数式・コード対応表: `math_code_correspondence_20260718.md`
- 修正・実験計画: `fix_and_experiment_plan_20260718.md`

---

## 1. エグゼクティブサマリー

1. **fixed 系列（`DualExpFamLSMFixed` 以下）の E-step 勾配・precision は、宣言された
   生成モデルの per-node 条件付き対数事後の勾配・負ヘッセ行列と一致する**
   [DERIVED]+[CONFIRMED_IN_REPOSITORY]（per-column 監査 31/31 PASS は clip 非作動域で
   これを数値的に裏付ける）。「1/2 不要」の導出は正しい（§4.1）。
2. **旧系列（0.5 あり）は「間違った Newton 方向」ではなく「Y 側尤度を 1/2 乗に
   温度緩和（tempering）した別の事後分布に対する正確な Laplace 近似」として特徴づけられる**
   [DERIVED]。ただしこれは真の事後に対しては mode も分散も一般にずれるため、
   root CLAUDE.md の「Newton 方向が正しいとは断定できない」という限定表現は
   引き続き妥当（§4.2）。
3. **新発見: 先行研究 MATLAB `calcEtaNewton.m` の `calcGrad` は Y 側残差項に w を
   掛けておらず、`calcAi` の対角除去項にも変換済み変数を再変換する不整合がある**
   [CONFIRMED_IN_REPOSITORY]。「MATLAB に 1/2 がない」という事実は変わらないが、
   MATLAB を単独のゴールドスタンダードとして引用することには限定が必要（§4.3）。
4. **現行 BIC は周辺尤度 BIC ではない。** `Q_strict` は EM の Q 関数（完全データ対数尤度の
   近似事後期待値）の MC 近似であり、周辺尤度 ln p(X,Y|θ) とはエントロピー項
   H(q) の分だけ体系的にずれる。このずれは k と n に依存して増大し得るため、
   現行基準は Schwarz BIC ではなく **ICL 型（完全データ型）基準**として解釈すべき
   [DERIVED]。疎な Y で k=1 が選ばれる現象（KI-011）はこの機構で定性的に説明できる
   [PLAUSIBLE]（§7–§9）。
5. **識別可能性: 観測分布を不変にする変換群は（一般の位置で）直交群 O(k) に一致**し、
   実装の自由度カウント `kd − k(k−1)/2` はこの群次元とちょうど整合する [DERIVED]。
   Procrustes 評価はこの O(k) 不定性を正しく処理している（§6）。
6. **異種 family の混在自体は BIC を無効にしない。** 無効化する要因は
   (a) Q を周辺尤度の代わりに使っていること、(b) 台・支配測度の異なる尤度同士の比較
   （family 間比較のみ無効）、(c) 潜在変数の増加と特異性、(d) ペア間依存と標本数の曖昧さ、
   に分解される（§8）。
7. **修論スコープ推奨: 「family 事前指定 + k 選択」を核とし、k 選択は held-out 予測を
   主基準・現行基準は ICL 型と明示して併記。per-column は prototype（将来課題）に格下げ、
   sparse-Y での X 統合の有効性（y_sparsity_stress、trials=10）を応用側の貢献として
   位置づける**のが最も誠実かつ実現可能（§17）。

---

## 2. 完全な同時分布 [DERIVED]（コードから確定した形）

観測: 属性行列 X = (x_il) ∈ 各列の台の直積、関係行列 Y = (y_ij)（対称、対角なし）。
潜在: Z = (z_1,…,z_n)^T ∈ R^{n×k}。パラメータ:

θ = { F ∈ R^{d×k}, w_0^Y ∈ R, w^Y ∈ R } ∪ { σ_l² : c(l)=gaussian } ∪ { σ_y² : family_y=gaussian }

観測ペア集合 O ⊆ {(i,j): i<j}（masked 版; 非 masked では O = 全 i<j）に対し

```
p(X, Y_O, Z | θ) = ∏_{i=1}^n N(z_i; 0, I_k)
                 · ∏_{i=1}^n ∏_{l=1}^d p_{c(l)}( x_il | η_il^X = f_l^T z_i, φ_l )
                 · ∏_{(i,j)∈O} p_Y( y_ij | η_ij^Y = w_0^Y + w^Y z_i^T z_j, φ_Y )
```

- X 側に切片なし（η_il = f_l^T z_i のみ）[CONFIRMED_IN_REPOSITORY: 全系列共通]
- 条件付き独立性: Z を与えたとき x_il は全て独立、y_ij は全て独立、X ⊥ Y | Z。
- 欠測ペア (i,j) ∉ O は尤度に一切寄与しない（§12 の ignorability 条件を前提とする）。
- 対角 y_ii は常に除外（mask 対角強制 False / 非 mask 版は j≠i 和で除外）。

**注意（生成器との差、§5.2）**: 人工データ生成では Z 自体が列ごとに z-score され、
Gaussian-X は生成後に列 z-score される。返却される「真値」Z, F, σ は生成後データの
厳密な生成パラメータとは O(n^{-1/2}) のオーダーでずれる。

---

## 3. family 別仕様表 [DERIVED]（実装と照合済み）

1変量指数型分布族 p(x|η,φ) = h(x,φ) exp{ (η T(x) − A(η)) / φ } として:

| 項目 | Bernoulli | Poisson | Gaussian（分散付き） |
|---|---|---|---|
| 標本空間 | {0,1}（計数測度） | {0,1,2,…}（計数測度） | R（Lebesgue 測度） |
| h(x,φ) | 1 | 1/x! | (2πφ)^{-1/2} exp(−x²/2φ) |
| T(x) | x | x | x |
| 自然パラメータ空間 | η ∈ R | η ∈ R | η ∈ R（η = μ） |
| A(η) | log(1+e^η) | e^η | η²/2 |
| A′(η) = E[T] | σ(η) | e^η | η |
| A″(η) | σ(η)(1−σ(η)) | e^η | 1 |
| 分散 | A″ | A″ | φ = σ² |
| E-step 重み A″/φ | σ(1−σ) | e^η | 1/σ² |
| 切片 | なし | なし | なし |
| 実装上の数値ガード | η clip ±500（尤度）、A″ 下限 1e-8 | **η clip [−20,10]**、A″ 下限 1e-8 | σ² 下限 1e-6〜1e-8 |
| 欠測時の寄与 | 0（mask） | 0（mask） | 0（mask） |

- Gaussian は canonical link = identity なので mean link と一致
  [CONFIRMED_IN_REPOSITORY: per_column_math_audit_20260711.md と同じ結論]。
- **正規化定数の実装規約**（BIC の絶対値に影響）: X 側 Gaussian は ln2π 込み
  （`model_dual_expfam.py` L.318-323）、Y 側 Gaussian は `calc_log_likelihood_Y` では
  ln2π 省略で、strict Q の補正は `eval_utils.calc_Q_dual_strict_exp` L.227-228 でのみ行われる。
  旧 `utils_expfam.calc_Q_dual_strict`（L.355-379）は **Gaussian-Y の ln2π 補正を行わない**
  [CONFIRMED_IN_REPOSITORY]。同一 family 内の k 選択には影響しない（定数）が、
  family_y 間で BIC を比較した場合は不整合となる（§8.3）。

---

## 4. 成立している部分（E-step の数理）

### 4.1 「1/2 不要」の導出は正しい [DERIVED]

Q_Y = Σ_{i<j} ℓ(y_ij, η_ij)、η_ij = w_0 + w z_i^T z_j（y, η ともに対称）とする。
固定した i について、i を含むペアは (i,j) [j>i] と (j,i) [j<i] で各1回ずつ現れ、
∂η_ij/∂z_i = w z_j より

∂Q_Y/∂z_i = w Σ_{j≠i} [T(y_ij) − A′(η_ij)]/φ_Y · z_j （1/2 なし）

−∂²Q_Y/∂z_i∂z_i^T = w² Σ_{j≠i} A″(η_ij)/φ_Y · z_j z_j^T （1/2 なし）

(1/2)Σ_{i≠j} と書いた場合も (i,j)・(j,i) 両側の寄与が合算され同じ式になる。
root CLAUDE.md・`model_dual_expfam_fixed.py` docstring の導出と一致
[CONFIRMED_IN_REPOSITORY]。fixed 系列（L.77, L.113）はこの式を正しく実装している。

### 4.2 旧系列（0.5 あり）の正確な特徴づけ [DERIVED]

旧系列（`model_expfam.py` L.109/L.135、`model_dual_expfam.py` L.159/L.200、
基底 `reproduction/src/model.py` L.283/L.353）は、勾配と precision の**両方の Y 項に
同じ 0.5** を掛けている。したがって旧 E-step は、目的関数

log p̃(z_i | ·) = log p(z_i) + log p(x_i|z_i) + **(1/2)**·log p(y_{i,·}|z_i, Z_{−i})

に対する**正確な** Newton 法・Laplace 近似である（Y 側尤度の 1/2 乗温度緩和）。含意:

- 温度緩和された目的に対しては Newton 方向・mode・曲率とも自己整合的。
- **真の**（緩和なしの）条件付き事後に対しては、mode は prior + X 側に寄った点にずれ、
  精度行列は I + F^T V_X F + (1/2)w²ΣA″zz^T ⪯ 正しい precision となるため、
  **Y 側が支配的な方向では事後分散が最大約2倍まで過大になり得る**（X/prior 支配方向では
  ほぼ差なし）。「一律2倍」とは言えない（root CLAUDE.md の限定表現と整合）。
- M-step（calc_w0/calc_w の /(2Lφ)）は両系列同一で、こちらの 1/2 は
  「全 i≠j 和 = 2×(i<j 和)」を直す**正しい**係数 [DERIVED]。したがって旧系列の
  バイアスは E-step サンプル経由でのみ入る。
- 以上は per-node 条件付き分布の話であり、**どちらの系列も Z 全体の同時事後を
  厳密に近似しているわけではない**（§13）。「旧版でも全体として正しい」とは
  断定できない、という従来の限定は維持する。

### 4.3 MATLAB 原実装に関する新発見 [CONFIRMED_IN_REPOSITORY]

`Mato Lab Program/calcEtaNewton.m` を精読した結果:

1. `calcAi`（L.56-63）の Y 項に 1/2 はない — root CLAUDE.md の主張どおり。
2. **`calcGrad`（L.43-49）の Y 側残差項 `YSWZ = (Y(ind,:) − S) * Z − …` には
   w が掛かっていない**（コメントアウトされた旧版 L.30-41 も同様）。連鎖律では
   ∂η_ij/∂z_i = w z_j なので w が必要であり、Python 実装（w を掛ける）の方が
   導出と整合する。w≈1 でない限り MATLAB の勾配は正しい勾配と一致しない。
3. `calcAi` L.61 の自己ペア除去は、既に `sig = w²·s(1−s)` へ変換済みの行列に対して
   `sig(ind,ind)*(1−sig(ind,ind))·w²` を掛けており、除去すべき量
   `w²·s_ii(1−s_ii)·z_i z_i^T` と一致しない（二重変換）。
4. サンプリング（L.23）は最終 Newton 反復**前**の invAi を再利用している
   （Python は最終点で A_i を再計算しており、この点は Python の方が丁寧）。

**含意**: 「MATLAB calcAi に 1/2 がない」ことは事実として維持されるが、同じファイルの
勾配側に別の不整合があるため、**MATLAB を「正しさの根拠」として単独で引用するのは危険**。
「1/2 不要」の主張は §4.1 の独立な導出で支えるべきであり、
`docs/teacher/half_factor_teacher_reply.md`・`docs/math_notes/half_factor_literature_code_check.md`
に本件の限定を追記することを推奨する（P0-6）。なお先行研究論文 PDF は読込不可のため、
印刷された式（Python docstring は「Eq.(23) に (1/2)」と記す）との突合は [UNRESOLVED]。

### 4.4 その他の成立事項

- masked 版の観測ペア制限（E-step / calc_w0 / calc_w / calc_sigma_y / llY）は
  数式どおりで、held-out 値の書換え不変性テストで漏洩なしを確認済み
  [CONFIRMED_IN_REPOSITORY: `test_experimental_models.py` test_masked_ignores_heldout_pairs]。
- per-column の列和構造・全列同一 family での既存モデル厳密一致（差 0.0）
  [CONFIRMED_IN_REPOSITORY: 31/31 PASS、ただし clip 非作動域・E-step と尤度のみ]。
- Procrustes（`utils_expfam.py` L.38-43）は直交 Procrustes の標準解 R = UV^T
  （反射も許容）で、§6 の不定性群 O(k) と正確に対応 [DERIVED]。

---

## 5. 未確認・矛盾部分

### 5.1 実装上の未検証・問題点

| # | 内容 | ラベル |
|---|---|---|
| U1 | Poisson clip [−20,10] 作動域では、実装勾配（clip 後 A′ を使用）は clip 後尤度（x·η_c − e^{η_c}）の勾配と**一致しない**（clip 域では後者の η 微分は 0、前者は非 0）。clip 非作動域では厳密一致（監査済み）。clip 発動率は記録されていない。 | [DERIVED] / 発動頻度は [UNRESOLVED] |
| U2 | Adam M-step（F, w0, w）が Q を単調増加させる保証はない（有限 50 反復・固定 lr・座標順次更新）。GEM としての単調性も未検証。 | [UNRESOLVED] |
| U3 | `scale_Z()` は毎 EM 反復で全サンプルを平均二乗 1 に強制再スケールする。これは MCEM の対象分布を変更する 헤ューリスティックであり、理論的正当化はリポジトリ内にない（§13.3）。 | [DERIVED]（性質）/ 正当化 [UNRESOLVED] |
| U4 | E-step の L 個のサンプルは独立サンプルではなく、逐次チェーン（前サンプルを初期値に再度 Newton+1 回サンプリング、ノード順次更新で更新済み z_j を使用）。近似分布の性質は未解析。 | [DERIVED]（構造）/ 性質 [UNRESOLVED] |
| U5 | `Z_est = Z_samples[:,:,-1]`（最後の 1 サンプル）を点推定として RMSE・予測に使用（`run_em_dual`、`em_runner.predict_mu_y`）。事後平均でなくサンプル 1 個なので、報告 RMSE にはサンプリングノイズが加算されている。 | [CONFIRMED_IN_REPOSITORY] |
| U6 | `em_runner.py` L.183-184 の `except Exception: pass` により Q/BIC 計算失敗が無警告で NaN になる。 | [CONFIRMED_IN_REPOSITORY] |
| U7 | 旧 `calc_Q_dual_strict` は Gaussian-Y の ln2π 補正なし（eval_utils 版は補正あり）。旧経路での family_y 間 BIC 比較は不整合。 | [CONFIRMED_IN_REPOSITORY] |
| U8 | 基底 `initialize_params` の `sigma` 初期値は単位行列、`w` 初期値 3×randn だが、各 runner の informed init が上書きするため実効初期値は runner 依存。複数初期値からの再スタートは未実施（1 fit = 1 初期値）。 | [CONFIRMED_IN_REPOSITORY] |

### 5.2 生成器と「真値」の整合性 [DERIVED]

`generate_dual_data`（`data_generator_expfam.py` L.223-363）:

- Z ~ N(0,I) 生成後に**列 z-score**（L.283）。よって「真の Z」は N(0,I) の厳密な
  サンプルではなく、標本平均 0・標本分散 1 に条件付けられたもの。モデル内の
  prior N(0,I) との差は O(n^{-1/2})。
- F は行ノルムを √(1−uniq)（既定 √0.9 ≈ 0.949）に正規化（L.287-290）。`var_f` は
  方向にのみ影響。
- **Gaussian-X**: X = ZF^T + noise 生成後に列 z-score（L.298）。設計上
  Var(x_l) ≈ ‖f_l‖² + uniq = 0.9 + 0.1 = 1 なので z-score はほぼ恒等変換だが、
  厳密には返却される F・sigma は z-score 後の X の生成真値と O(n^{-1/2}) ずれる。
  RMSE(F) ≈ 0.035–0.04（表2）の一部はこの生成側ずれを含む可能性がある
  [PLAUSIBLE、定量化は将来実験]。
- Bernoulli/Poisson-X・Y 側は正規化なしで生成過程とモデルが一致（Poisson は生成時にも
  η clip [−20,10]）。
- **誤指定実験の位置づけ**: 台の異なる family を強制する条件
  （例: 生カウントに Bernoulli スコア、z-score 連続値に Poisson リンク）は
  正しい確率モデルではなく quasi-likelihood 的目的の最適化である
  [CONFIRMED_IN_REPOSITORY: per_column_math_audit_20260711.md L.94-97 の認識と一致]。
  RMSE(Z) 比較は「作業推定量の比較」として意味を持つが、これらの条件の
  「尤度」「BIC」は確率モデルとしては無効（§8.3）。

### 5.3 文書の時系列矛盾 [CONTRADICTED]（いずれも stale 文書、実装の誤りではない）

| 文書 | 記載 | 矛盾する事実 |
|---|---|---|
| `reports/claims_and_evidence.md` L.21（2026-05-31） | 「実データへの適用は未実施」 | 実データ実験フェーズ（2026-06-17〜07-07、main マージ済み）が存在 |
| `RESEARCH_MASTER.md` §11 | 「Wine実データで有効性が確認された（未評価）」を「まだ言ってはいけない」に記載 | 同文書 §8b・KI-006 更新（fixed 版 Wine 評価は実施済み、位置づけの限定付き） |
| `KNOWN_ISSUES.md` KI-012 | 「現在のモデル API は pair mask に対応しておらず」 | experimental `DualExpFamLSMMasked`（2026-07-10 コミット 16d456c 以降）が対応済み。KI-012 は「本体 API（fixed 版）は未対応・experimental で対応」と更新すべき |
| root `CLAUDE.md` 残タスク | 「MovieLens pair mask 対応（strict held-out）」が未完扱い | per-column フェーズの MovieLens pilot は strict held-out（mask）で実施済み（experimental 系列に限る） |

---

## 6. 識別可能性 [DERIVED]

**主張**: 一般の位置の θ（rank(F)=k、w≠0）に対し、観測分布 p(X,Y|θ) を不変に保つ
パラメータ変換は F → F R^T（R ∈ O(k)）にちょうど一致する。

導出: 潜在変数を z → R z と変換すると、
(i) prior N(0,I_k) は R R^T = I のとき（かつそのときに限り）不変、
(ii) X 側 η = F z は F → F R^T で不変、
(iii) Y 側 z_i^T z_j は R^T R = I で不変。
一般可逆行列 T では prior の共分散が T T^T となり N(0,I) と一致しないため、
prior を固定する本モデルでは群は O(k) に制限される。スケール変換
(z, w, F) → (cz, w/c², F/c) は**尤度のみ**を不変にするが prior が破るため、
厳密な不定性ではなく**弱識別（尾根）方向**として残る（§13.3 の scale_Z と関係）。

- 群 O(k) の次元は k(k−1)/2。よって F の実効自由度 kd − k(k−1)/2
  [CONFIRMED_IN_REPOSITORY: `calc_bic_dual` L.399 と一致]。w_0, w, σ 系は
  一般の位置で識別可能。
- **特異部分集合**: rank(F) < k、w = 0、F = 0 等では固定部分群が大きくなり
  自由度カウントが変わる（過剰指定 k > k* では真値が特異点に乗る）→ §8.2。
- Procrustes 後 RMSE(Z) は O(k) 不定性を正しく除去する（SVD 解は反射込み）。
  ただし (a) スケール尾根方向の弱識別は Procrustes では扱わない
  （生成器・scale_Z の双方が二乗平均 1 を強制しているため実験上は整合）、
  (b) Gram 行列 ZZ^T・リンク確率・held-out 予測は変換自由な評価として併用価値が高い
  [DERIVED]。real-data 側では既に held-out AP 等を併用しており方向性は正しい。

---

## 7. 現在の BIC の厳密な定義 [CONFIRMED_IN_REPOSITORY + DERIVED]

実装（`eval_utils.calc_Q_dual_strict_exp` L.186-229、`calc_bic_exp` L.232-256、
旧経路 `utils_expfam.calc_Q_dual_strict` L.355-379、`calc_bic_dual` L.386-404）:

```
BIC_impl = −2 · Q̂ + p̂ · ln(n)

Q̂ = (1/L) Σ_{l=1}^{L} [ ln p(Z^{(l)}) + ln p(X | Z^{(l)}, θ̂) + ln p(Y_O | Z^{(l)}, θ̂) ]
p̂ = kd − k(k−1)/2 + (Gaussian-X 列数) + 1{family_y=gaussian} (+ 1{NB r 推定})
    （w0, w, Z は数えない。n = オブジェクト数）
```

Z^{(l)} は §5.1-U4 の逐次チェーンサンプル（scale_Z 適用後）。

**分類**: Q̂ は「近似事後 q のもとでの完全データ対数尤度の期待値」= EM の Q 関数の
MC 近似である。周辺尤度でも条件付き尤度でも観測データ尤度でもない。
標準 EM 分解 ln p(X,Y|θ) = Q(θ) + H(q) + KL(q ‖ p(Z|X,Y,θ)) より

BIC_impl = [標準BIC相当: −2 ln p(X,Y|θ̂) + p̂ ln n] + 2·H(q) + 2·KL(…)

**H(q) は k・n・事後の濃度に依存する**: q を per-node Laplace ガウスとすれば
H(q) = Σ_i [ (k/2)ln(2πe) + (1/2)ln det A_i^{-1} ]。事後が拡散的
（A_i ≈ I、情報の乏しい疎 Y・弱 X）なら H ≈ 1.42·n·k > 0 となり、
**k を 1 増やすごとに約 2.84n の追加ペナルティ**が乗る。n=280（Cora）では
これは約 795/k で、パラメータ側ペナルティ d·ln n ≈ 282/k を大きく上回る。
事後が濃い（A_i 大）場合は ln det A_i^{-1} が大きく負になり相殺・逆転する。

**含意** [DERIVED（機構）+ EMPIRICALLY_OBSERVED（整合）+ PLAUSIBLE（因果断定は不可）]:
- 密で情報の多いデータ（人工 3 シナリオ、Wine 密度 0.34）では BIC_impl が
  k*=3 を選べた一方、疎な Cora（密度 0.011）で k=1 を選んだ KI-011 の現象は、
  「疎 → 事後拡散 → H(q) 正で大 → 完全データ型基準の過大ペナルティ」という機構と
  定性的に整合する。ただし実行時の H(q) を計測していないため因果は未確定。
  **H(q) は A_i から低コストで計算可能**であり、検証実験を P1 に提案する。
- したがって「BIC がペナルティ過大」(KI-011) の第一容疑は ln(n) の选択よりも
  **Q と周辺尤度の乖離（エントロピー欠落）**である。

**KI-010（num_params）の部分的解決** [DERIVED]:
- kd − k(k−1)/2 は §6 の O(k) 群と整合し、因子分析の慣例とも一致。
- w0, w（2 個）は k に依存しない定数なので **k 選択には影響しない**。family 間比較でも
  両モデルに共通なら影響しない。「数えるべきか」は基準の定義次第であり、
  数えても k 選択順位は不変。
- Z を数えないのは「Z は積分すべき潜在変数」という立場では正しいが、
  現行 Q̂ は Z を積分しておらず（サンプル代入）、立場が混在している。
  これが上記エントロピー問題の別表現である。
- 標本数 n の選択: F の各行 f_l は X の n 個の観測で識別されるため、F・σ_l に
  対する ln(n) は オーダーとして妥当。w0, w は O(n²) ペアで識別されるが数えて
  いないため実害なし。Y 側モデルの複雑度が k に依存して増えないため、
  「Y の情報量に対する k のペナルティ」は F 経由でしか入らない点は
  設計上の非対称として認識しておく。
- 先行研究（Mikawa et al. 2024）の規約との一致は、論文 PDF が読込不可のため
  コード注釈（`calc_bic` L.103「matches Mikawa et al. 2024 formula」）以上の
  確認ができない [UNRESOLVED]。

---

## 8. 通常 BIC を使える条件・使えない可能性の原因分解

### 8.1 Schwarz BIC の成立条件 [SUPPORTED_BY_PRIMARY_SOURCE]

Schwarz (1978), "Estimating the Dimension of a Model", Ann. Statist. 6(2):461-464,
DOI: 10.1214/aos/1176344136。BIC は (i) **観測データの（周辺）尤度**の最大値、
(ii) 正則モデル（Fisher 情報非退化・内点真値）、(iii) 十分大きな iid 型標本、の下で
ln 周辺尤度 = ln p̂ − (p/2)ln n + O_p(1) という Laplace 展開に基づく。

### 8.2 本モデルでの逸脱要因の分解 [DERIVED]

| 要因 | 内容 | k 選択への影響 |
|---|---|---|
| (A) Q ≠ 周辺尤度 | §7。エントロピー欠落が k・n 依存で入る | **大**（疎データで過大ペナルティの疑い） |
| (B) 潜在変数の増加 | z_i が n に比例して増える（incidental parameters）。周辺化すれば消えるが、現行は周辺化していない | (A) と同根 |
| (C) 特異性 | k > k* では真値が rank 欠損 F の特異点に乗り、正則 Laplace 展開が破綻。正しいペナルティは学習係数 λ·ln n（λ ≤ p/2）で、**標準 BIC は過剰指定側を過大ペナルティ化する傾向** | 中（k* 上側の比較に影響）[SUPPORTED_BY_PRIMARY_SOURCE: Watanabe 2013 WBIC, JMLR 14:867-897; Drton & Plummer 2017 sBIC, JRSS-B 79(2):323-380, DOI:10.1111/rssb.12187] |
| (D) ペア間依存・標本数の曖昧さ | Y の n(n−1)/2 ペアは z 共有で従属。X は nd 観測。ln(n) の「n」の選択に複数の自然な候補 | 中（w0,w 未計上のため現状は限定的） |
| (E) 誤指定 | overdispersion（MovieLens var/mean≈10）等では BIC の前提（真値近傍での二次展開）が KL 最適点の意味でしか成立しない | 中 |
| (F) 異種 family 混在それ自体 | 各列・各ペアが正しい確率モデルなら対数尤度の和は well-defined であり、**混在自体は障害ではない** | なし |
| (G) family 間比較 | 計数測度と Lebesgue 測度の尤度は支配測度が異なり、**値の直接比較は無意味**。台違反の quasi-likelihood は確率モデルですらない | family 選択に BIC を使う場合のみ致命的 [DERIVED]（per_column_final_summary の「all_bernoulli の BIC が最小に見える」観察 [EMPIRICALLY_OBSERVED] の理論的説明） |

**結論**: 「異種 family だから BIC 不可」は誤り。**同一 family 割当・同一データの
k 選択**に限れば、(A)(C) を直せば BIC 系基準は原理的に使える。family 自体の選択には
(G) により、同じ台上の分布同士（Poisson vs NB 等）以外は BIC を使うべきでない。

### 8.3 家族間比較の可否早見 [DERIVED]

- 可: Poisson vs NB（同じ計数測度、定数込み尤度で）。Bernoulli vs（同じ {0,1} 上の）別モデル。
- 不可: Bernoulli/Poisson（離散）vs Gaussian（連続）の尤度値比較。
  台違反 quasi 条件（{0,1} 外への Bernoulli 等）の「BIC」。
- 実務上の代替: family 選択は held-out 予測（同一予測対象・同一評価量）で行う。

---

## 9. 代替モデル選択法の比較

| 手法 | 何を近似/評価するか | 前提 | 利点 | 欠点 | 実装難度 | 修論での実現性 | 推奨 |
|---|---|---|---|---|---|---|---|
| **ELBO 補正 BIC**（−2(Q̂+H(q)) + p̂ ln n; variational BIC 型） | 周辺尤度の下界 | Laplace q の妥当性 | **A_i を既に計算しており H(q) が殆ど無コスト**。(A)(B) を直接是正 | 下界の緩みは k 依存で残る | 低 | ◎ | **P1 で最優先** |
| ICL 型（現行基準の再解釈） | 完全データ尤度基準 | — | 実装済み。クラスタ分離を好む性質は既知 [SUPPORTED_BY_PRIMARY_SOURCE: Biernacki-Celeux-Govaert 2000, IEEE TPAMI 22:719-725, DOI:10.1109/34.865189] | 「BIC」と呼ぶと誤解を生む | 0 | ◎（呼称と解釈の修正のみ） | ◎ |
| held-out 予測（CV / pair split） | 予測リスク | 分割の独立性・MCAR | 特異性・依存・誤指定に頑健。masked 実装済み | 計算コスト×分割数。X 側 held-out は未実装 | 中 | ◎（実データで既に主評価） | ◎ |
| WBIC | 特異モデルの自由エネルギー | 逆温度 1/ln n の事後サンプリング | 特異性 (C) に理論対応 | MCMC 必須で現行 Laplace 枠外。実装重い | 高 | △ | 将来課題 |
| sBIC | 特異 BIC | 学習係数 λ の知識 | 理論的に正しい罰則 | 本モデルの λ は未知（文献未発見 [UNRESOLVED]） | 高 | × | 将来課題 |
| 周辺尤度の直接近似（bridge/IS） | ln p(X,Y) | サンプラ品質 | 定義が明確 | 高分散・高コスト | 高 | △ | 将来課題 |
| PPC / calibration | 適合の絶対評価 | — | overdispersion 等の診断に最適 | k 選択には間接的 | 低〜中 | ◎ | ○（診断として） |

---

## 10. 「真のモデル」の扱い [DERIVED + CONFIRMED_IN_REPOSITORY]

- 本研究は **family をデータ型から分析者が事前指定する研究**である
  （原稿 3 章「分析者が柔軟に指定できる」、per-column 設計も同様。family 自体の
  データ駆動選択は per_column_final_summary #12-5 で将来課題と明記）。
  したがって「真のモデル」問題は主に k* に関するもの。
- 人工データ: k* は生成者が設定でき、候補集合 {1..6} に真値が含まれる。
  BIC_impl の成功（3 シナリオ×10 試行で k*=3）は [EMPIRICALLY_OBSERVED] だが、
  §7 の理由により「BIC の理論的正当化の証拠」ではなく「この設定での経験的成功」。
- Wine: 「真のクラス数 3」と「真の潜在次元 3」は同一視できない。3 クラスの
  同クラス関係行列は k=2（単体配置）でも分離表現が可能であり、k=3 一致は
  示唆的だが「正解を当てた」と主張すべきでない [DERIVED]。
- Cora: BIC 最小 k=1 / AP 最大 k=6 / NMI・ARI 最大 k=3 は**異なる損失関数の
  最適値**であり、単一の「真の k」が観測不能な実データでは矛盾ではない。
  「目的別に k を選ぶ」という現行の整理は適切 [CONFIRMED_IN_REPOSITORY]。
- 候補集合に真値がない場合（実データは常にそう）: KL 最適（擬真値への収束）、
  予測最適（リスク最小）、説明最適（解釈性）を分けて主張する。実データの主張は
  予測最適（held-out）に限定するのが安全。

---

## 11. 漸近理論の整理 [DERIVED]（すべて将来課題の分類であり、証明は行っていない）

| 設定 | 内容 | 本研究での状態 |
|---|---|---|
| R1: n→∞、d,k 固定 | X は nd 観測、Y は Θ(n²) ペア。z 固定次元なら η_ij=O(1) → **dense ネットワーク**（平均次数 Θ(n)） | RMSE(Z) 減少を n=50→300 で経験的観測のみ [EMPIRICALLY_OBSERVED]。一致性・レートの証明なし |
| R2: n,d→∞ | F のパラメータ数も増加 | 未検討 [OUT_OF_SCOPE] |
| R3: n→∞、sparse Y（ρ_n→0） | 現行モデルは w0 固定では表現不可。w0_n = O(ln ρ_n) のドリフトが必要 | モデル定義に含まれず [UNRESOLVED]。Cora（ρ=0.011）は固定 n の有限標本問題として扱っている |
| R4: 観測率 π_n のペア部分観測 | masked 尤度。π_n→0 で Y 情報消失 | y_sparsity_stress（π=1→0.1、n=80 固定）で経験的にのみ検証 [EMPIRICALLY_OBSERVED] |

性質別の分類:
- 大域パラメータ（F, w0, w, σ）の一致性: **仮定整理から将来課題**。潜在変数を周辺化した
  尤度の M 推定として定式化すれば R1 で標準論法が見込めるが、Laplace-MCEM 推定量に
  ついての保証は別問題。
- 個々の z_i: n→∞ でも各 z_i の情報は Y 側 Θ(n)・X 側 Θ(d) で、Y が dense なら
  z_i も一致推定可能と**予想**される [PLAUSIBLE]。
- k 選択の一致性: §7-8 の理由により現行基準では主張不可。**「k 一致性は将来課題」
  と明示するのが誠実** [DERIVED]。
- Hoff-Raftery-Handcock 2002（JASA 97(460):1090-1098, DOI:10.1198/016214502388618906）は
  latent space model の推定枠組みの一次文献だが、そこでも k 選択一致性の理論は
  与えられていない（本監査で全文精査はしていない）[SUPPORTED_BY_PRIMARY_SOURCE は
  枠組みの存在まで。詳細は UNRESOLVED]。

---

## 12. sparse / missing / PU の区別と現在地 [DERIVED]

| # | 状況 | 尤度の正しい扱い | 現行実装 |
|---|---|---|---|
| 1 | 完全観測で 0 が多い（疎ネットワーク） | 全ペアを尤度に入れる（0 は観測） | fixed 版・Cora がこれ |
| 2 | 一部ペアのみ観測（既知の観測集合） | 観測ペアのみ尤度（ignorable なら valid） | masked 版 `train_mask` がこれ |
| 3 | MCAR/MAR/MNAR | MCAR/MAR+distinctness なら #2 で valid。MNAR は欠測機構のモデル化が必要 | ランダム pair split は MCAR を構成 [DERIVED]。MNAR 未対応 |
| 4 | negative sampling | 重み補正が必要 | 未実装（評価側の neg_ratio は評価専用） |
| 5 | positive-unlabeled | PU 尤度（0 = 未観測∨真の0 の混合） | 未対応 |
| 6 | 1 のみ観測・0/未観測不可分 | #5 の特殊形 | 未対応 |

- 現行 `train_mask=False` は「未知（尤度から除外）」であり「観測された 0」と明確に
  区別されている [CONFIRMED_IN_REPOSITORY]。文書上は KI-012 等の旧記述が
  experimental 実装前の状態を指すため更新が必要（§5.3）。
- リンク予測評価: pair split は対称化されており対角除外も一貫。同一ノードが
  train/test 双方のペアに現れる transductive 設定であることは明示すべき
  （cold-start ノード予測は評価していない）。
- MovieLens（Poisson 主結果）は全ペア正（density_pos=1.0）で、「疎」問題ではなく
  「0 が存在しない」問題。sparse Y の議論と混同しないこと。

---

## 13. Laplace・MCEM 監査（要点）

### 13.1 一致している点

- 勾配・precision の符号規約（−ln f の勾配を返し z ← z − αA⁻¹∇(−ln f)）は
  全系列で一貫 [CONFIRMED_IN_REPOSITORY]。
- A_i は Term1=I により正定値が保証され、対称化 + 1e-6 正則化も追加
  （`reproduction/src/model.py` L.423-426）。
- 最終点で A_i を再計算してからサンプリング（MATLAB より丁寧、§4.3-4）。

### 13.2 反対査読で残る問題

1. **per-node 近似**: q(Z) = Π_i q_i(z_i) 相当の近似で z_i 間の事後依存を無視。
   ノード順次更新（更新済み z_j を使用）で Gibbs 的だが、詳細釣合いの保証はない。
2. **mode 未収束のまま Laplace**: Newton は max_iter=10、α=0.5（expfam 系）/
   0.01（reproduction 既定）。α=0.01 の場合 10 反復では mode に到達しないのが普通で、
   「mode でない点まわりの Laplace」になっている [DERIVED]。
3. **L=5・EM 8 反復**: Q̂・BIC の MC 分散は未定量化。チェーンサンプル（U4）のため
   有効サンプルサイズは L より小さい可能性が高い [PLAUSIBLE]。
4. **scale_Z**（§13.3）と **Adam M-step**（U2）により、GEM としての単調性も
   保証されない。
5. NaN ガードは「前反復の Z を複製して続行」であり、失敗を隠して統計を歪める
   可能性がある（発生時は nan_occurred で記録される点は良い）。

### 13.3 scale_Z の位置づけ [DERIVED]

`scale_Z`（`reproduction/src/model.py` L.468-504）は全サンプルの平均二乗を 1 に強制する。
§6 のスケール尾根（尤度不変・prior のみが抑制）方向のドリフトを抑える実務的装置と
解釈できるが、(i) 事後からのサンプルを変形するため MCEM の対象分布を変え、
(ii) w・F の推定値がスケールを吸収する形で共変し、(iii) Q の評価もスケール後の
サンプルで行われる。真の Z も生成時に z-score されているため**評価は自己整合的**だが、
「prior 分散 1 で識別する」という原稿の記述と「サンプルを毎回強制正規化する」実装は
同じではない。除去アブレーション（P1）を推奨。

---

## 14. 数式・コード対応表

別ファイル `math_code_correspondence_20260718.md` 参照（全行番号は本監査で実物と突合済み）。

---

## 15. 先生コメントへの回答案（解釈・回答・確認質問）

> 各コメントの原文が手元にないため、KNOWN_ISSUES・研究文書から再構成した論点に
> 対する回答案である。先生の意図と異なる場合は確認質問を優先する。

### 15.1 「真の次元と BIC」（KI-010/011 系）

- 可能な解釈: (a) BIC の定義・自由度は正しいか、(b) 実データで機能しない理由、
  (c) 異種 family で BIC は意味を持つか。
- 回答案: 現行基準は周辺尤度 BIC ではなく完全データ（ICL 型）基準である（§7）。
  自由度 kd − k(k−1)/2 は識別群 O(k) と整合（§6）。疎データでの k=1 選択は
  エントロピー欠落機構で定性的に説明できる（§7）。異種 family 混在自体は障害でなく、
  family **間**比較にのみ使えない（§8）。
- 未回答部分: H(q) 計測による機構の定量検証（P1 実験）。
- 確認質問: 「BIC の一致性を理論的に示すこと」と「実データで妥当な k を選ぶ実務基準」の
  どちらを優先すべきか。

### 15.2 「生成モデルの整合性」

- 回答案: 同時分布は §2 の形で完全に書ける。正規化定数・台・切片・clip の扱いは
  §3・§5 のとおりで、Poisson clip 域と誤指定条件のみ確率モデルから逸脱する
  （それぞれ数値ガード・quasi-loss として明示する）。
- 確認質問: X 側切片（η_il = μ_l + f_l^T z_i）の導入は先行研究からの逸脱になるが、
  実データ適用上は必要と判明した（story diagnostics）。修論で導入してよいか。

### 15.3 「1次元の真のモデル」

- 可能な解釈: k*=1 の最小例で推定・選択が何を意味するかを説明せよ。
- 回答案: k=1 では O(1)={±1} なので不定性は符号のみ。z_i スカラーで
  η_ij = w0 + w z_i z_j。この最小例で Q̂ と周辺尤度の差・H(q) を数値で示すのが
  最も教育的（P1 実験に含める）。
- 実データでの k*=1 の操作的定義は存在しない（§10）。

### 15.4 「実験だけでなく理屈を」

- 回答案: 本監査の §4（E-step の厳密性）、§6（識別可能性）、§7-8（基準の分類）、
  §12（欠測の ignorability）は証明可能な範囲。k 選択・推定量の一致性は
  仮定整理＋将来課題として明示（§11）。
- 「実験で成功」と「証明」を混同しない記述方針を全文書に適用する。

### 15.5 「sparse Y と欠損予測」

- 回答案: §12 の 6 分類で整理。現行は #1（完全観測疎）と #2（MCAR 部分観測）のみ
  正当に扱える。y_sparsity_stress は #2 の設定で「Y 情報が乏しいほど正しい X 統合が
  効く」ことを trials=10 で示した（1 生成設定のみ）[EMPIRICALLY_OBSERVED]。
  PU・MNAR は未対応と明示。
- 確認質問: 実データで想定する欠測は #2（観測ペア既知）と #5（PU）のどちらに近いか。

### 15.6 「データを無限に増やしたら」

- 回答案: §11 の R1〜R4 を区別して回答する。「n→∞・dense Y」では大域パラメータの
  一致性が標準論法で見込める（未証明）、「sparse Y」は現行モデルの範囲外、
  「k 選択の一致性」は現行基準では主張できない。

---

## 16. 修士研究の現実的到達点（スコープ比較）

| 案 | 内容 | 評価 |
|---|---|---|
| S1: per-column 完全理論化 | 切片・スケール・family 選択・一致性まで | 実現性低。pilot で前提条件（切片等）の未解決が判明済み |
| S2: 生成モデル/実装監査まで | 本報告の内容を成果とする | 誠実だが手法貢献が薄い |
| S3: **family 事前指定 + k 選択の理論整理** | 現行基準の ICL 型としての再解釈 + ELBO 補正 + held-out 主基準。k 一致性は将来課題と明示 | **推奨核**。低リスクで理論的貢献が立つ |
| S4: per-column を将来課題に格下げ | prototype・診断結果（切片必要性の特定）を「課題の同定」として記載 | S3 と両立。既存レポートの記述方針と一致 |
| S5: sparse/missing 予測を応用貢献 | masked 尤度の正当化（§12）+ y_sparsity_stress の拡張 | S3 に載せる形で推奨。単一設定依存の解消が必要 |

**推奨: S3 + S4 + S5。**スコープ縮小（S1 の放棄）が最も誠実である。
学会予稿の主張（人工データ・旧系列）は変更せず、修論では fixed/masked 系列 +
再定義した選択基準で一貫させる。

---

## 17. 未解決事項（優先順）

1. H(q) 計測による BIC 機構検証（§7）— 実験未実施。
2. 先行研究論文の印刷式（Eq.22/23 の 1/2、BIC 規約）との突合 — PDF 読込不可 [UNRESOLVED]。
3. MATLAB calcGrad の w 欠落が先行研究の公表数値に与えた影響 — 判定不能 [UNRESOLVED]。
4. Adam M-step・MCEM 全体の単調性/収束性 — 未検証。
5. scale_Z 除去時の挙動 — 未検証。
6. Poisson clip 発動率の実測 — 未記録。
7. 本モデルの学習係数（sBIC 用）— 文献未発見 [UNRESOLVED]。
8. Gaussian-X 生成後 z-score の RMSE(F) への寄与の定量化 — 未実施。

## 18. 先生への確認質問（まとめ）

1. k 選択の目標は「理論的一致性」か「実データでの実務基準」か（§15.1）。
2. X 側切片の導入（先行研究からの逸脱）を修論で行ってよいか（§15.2）。
3. 実データの欠測想定は「観測ペア既知」か「PU」か（§15.5）。
4. 現行基準を「BIC」と呼び続けるか「完全データ情報量基準（ICL 型）」と改称するか。
5. 先行研究の原論文 PDF（`paper/`、読込不可）の別入手は可能か（式・BIC 規約の突合のため）。

---

## 19. 承認依頼

ここまでは読み取り専用監査であり、リポジトリへの変更は本報告 3 ファイル
（`reports/theory_audit/` 配下の新規作成）のみである。コード・既存文書・CSV・図は
一切変更していない。**修正フェーズへ進める P0/P1 項目の承認**
（`fix_and_experiment_plan_20260718.md` 参照）をお願いしたい。
