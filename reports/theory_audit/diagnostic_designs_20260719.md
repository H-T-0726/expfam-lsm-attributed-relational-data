# 診断設計と前回監査の訂正（2026-07-19）

対象: 前回理論監査（`theory_audit_report_20260718.md` ほか2ファイル）の再検証結果、
および承認済み診断・設計項目（P1-1 / P1-3 / P1-4 / P1-8）の設計。
**前回監査の3ファイル自体は履歴保全のため変更せず、訂正はすべて本ファイルに記載する。**

---

## 1. 前回監査の訂正（必読）

### 訂正1: k=1 でも joint 周辺化は n 次元積分である [DERIVED]

`fix_and_experiment_plan_20260718.md` 実験計画2 の
「k*=1 で Q̂ vs 周辺尤度（k=1 なら数値積分で厳密計算可能）」は**誤り**。

k=1 は各 z_i がスカラーであることを意味するに過ぎず、周辺尤度は

p(X, Y | θ) = ∫_{R^n} p(X, Y, Z | θ) dZ

であり、Y 側の項 p(y_ij | w0 + w z_i z_j) が z_i と z_j を結合するため
被積分関数は z_1,…,z_n の積で**因子化しない**。積分次元は n である
（X 側のみなら因子化するが、Y 側がある限り不可）。

厳密・準厳密な周辺尤度比較を行う場合の正しい選択肢:
- n = 2〜6 程度の極小 joint モデルでの多次元 quadrature
- 1 ノード条件付き積分（z_{−i} を固定した p(x_i, y_{i,·} | z_{−i}) の 1 次元積分）
  — これは周辺尤度そのものではないことを明記して使う
- joint Laplace 近似（Z 全体 nk 次元のヘッセ行列による）
- 高精度重要度サンプリング（提案分布の品質評価付き）
- 極小設定での複数手法の突合

### 訂正2: 「MATLAB calcGrad の w 欠落は新発見」ではない [CONTRADICTED]

前回報告 §4.3 は calcGrad の w 欠落・calcAi 対角除去の不整合を「新発見」としたが、
`docs/math_notes/half_factor_literature_code_check.md`（2026-05-08）の
§2-A（L.79-81）・§2-B（L.101-102）に**両方とも既に記録されていた**。
事実認定自体（w 欠落・対角除去の二重変換）は正しく維持されるが、
先行文書の存在を見落とした点を訂正する。
なお同文書 L.245-246 と `half_factor_teacher_reply.md` L.31 には
「ニュートン更新の方向は正しい」「サンプリング分散が 2 倍に膨らむ」という
root CLAUDE.md（再発防止表現）より強い断定が残っていたため、
P0-6 の日付付き注記で限定した（両文書の追記参照）。

### 訂正3: 名称の遡及的制限

前回報告 §9 は「ELBO 補正 BIC」を最有力修正案と呼んだが、現段階では
以下が未確定であるため（§2 参照）、この名称と `calc_elbo_bic` の正式実装は保留する。

- 現行 E-step のサンプルが単一の明示的 q(Z) からの iid サンプルであること
- Q を計算した分布と H を計算する分布の一致
- scale_Z 後のサンプル分布の扱い

使用してよい暫定名称:
`nodewise Gaussian entropy diagnostic` / `block-diagonal Laplace entropy diagnostic` /
`entropy-based diagnostic candidate`。
禁止: `ELBO-BIC` / `marginal BIC` / `corrected BIC` / `true BIC`。

### 訂正4: 基準の呼称と Cora k=1 の原因

- 現行基準（−2Q̂ + p̂ ln n）は **Schwarz BIC ではない**（Q̂ が周辺尤度でないため）。
  また標準 ICL そのものとも断定せず、
  「**Qベース完全データ型基準（Q-based complete-data criterion / ICL-type）**」と呼ぶ。
- Cora で k=1 が選ばれた原因を「エントロピー欠落」と**断定しない**。
  前回報告 §7 の機構は [PLAUSIBLE]（定性的整合のみ、H(q) 未計測）であり、
  検証は §2 の診断設計を経てから行う。

### 訂正5: 「理論的 BIC ではない」≠「無価値」

現行基準は完全データ型のモデル選択指標として経験的に機能する可能性があり
（人工 3 シナリオ・Wine での k*=3 選択 [EMPIRICALLY_OBSERVED]）、
「Schwarz BIC でない」ことを「使えない」と読み替えてはならない。

---

## 2. P1-1: エントロピー診断の設計（実装は今回見送り）

### 2.1 現行 E-step が暗黙に定義している確率過程 [CONFIRMED_IN_REPOSITORY]

`run_em_experimental`（`em_runner.py`）/ `run_em_dual`（`utils_expfam.py`）は
各 EM 反復で l = 1..L について:

1. 現在の Z 行列（前サンプル）を初期値に、ノード i = 1..n を**順次** Newton 更新
   （max_iter=10、α=0.5、更新済み z_j を後続ノードが参照）
2. 各ノードで最終点の A_i を再計算し z_i ~ N(z_i^{mode候補}, A_i^{-1}) を**その場で**サンプル
   （`reproduction/src/model.py` calc_eta_newton）
3. l 番目のサンプル Z^{(l)} として保存し、次の l の初期値にする
4. L 個収集後に scale_Z で全サンプルを平均二乗 1 に強制再スケール

これは「systematic-scan の近似 Gibbs に似た逐次過程」だが、
(i) 提案が Laplace ガウスで受理判定がない、(ii) mode 未収束点を中心に使う、
(iii) scale_Z の変形が入る、ため**標準的な MCMC の定常性・詳細釣合いは主張できない**
[DERIVED]。したがって Z^{(1..L)} を単一の q(Z) からの iid サンプルとは呼べない。

### 2.2 q(Z) = Π_i q_i(z_i) と置ける条件

積形式の明示的 q が定義できるのは、少なくとも:
- 各 q_i = N(m_i, A_i^{-1}) の (m_i, A_i) が**固定された** Z_{−i}・θ で評価され、
- サンプリングが全ノード同時（更新済み z_j を参照しない）で、
- scale_Z を適用しない
場合である。現行実装はこの 3 条件をいずれも満たさない。
よって Σ_i H(q_i) は「ある固定時点の per-node Laplace 近似のエントロピー和」であり、
**Q̂ を計算したサンプル分布のエントロピーとは同定できない** [DERIVED]。

### 2.3 Σ_i H(q_i) が何の診断になるか

H_i = (k/2)ln(2πe) − (1/2)ln det A_i。Σ_i H_i は
「最終 E-step 時点の per-node 事後不確かさの総量」の指標であり:
- k・データ情報量への依存の観察（疎 Y で大きくなるか）
- −2Q̂ 基準と −2(Q̂+ΣH_i) の選択 k の感度比較（**診断としてのみ**）
に使える。周辺尤度推定量・ELBO 補正としての地位は未確立。

### 2.4 実装する場合の条件（今回は見送り）

- 関数名は `nodewise_gaussian_entropy(precisions)` 等に限定。
- モデル選択へ接続せず、結果 dict の暫定診断値のみ。
- docstring に必ず次を記載:

```text
This quantity is not established as an ELBO correction or a marginal
likelihood estimator for the current sequential Laplace-MC procedure.
```

### 2.5 検証実験の設計（承認後）

極小 n（2〜6）・k=1〜2 の joint quadrature で ln p(X,Y|θ) を厳密計算し、
Q̂・Q̂+ΣH_i・joint Laplace の 3 者と比較する（訂正1 の方法論に従う）。

---

## 3. P1-3: scale_Z アブレーション設計（実験なし）

- 入出力: `scale_Z(Z_samples)` は (n,k,L) の全要素平均二乗を 1 に強制する
  一様スケーリング（`reproduction/src/model.py` L.468-504）。
- 理論的位置づけ: 変換 (Z, w, F) → (cZ, w/c², F/c) は**尤度を不変**にするが
  prior N(0,I) がこれを破る（弱識別の「尾根」）。scale_Z は prior の 2 次モーメントを
  サンプルへ強制する実務的装置であり、事後サンプルを変形するため
  尤度原理・MCEM の対象分布からは逸脱する [DERIVED]。
- 非破壊切替案: `run_em_experimental` に既定 True の `apply_scale_z=True` を追加
  （既定で従来挙動、False で off）。既存関数・既存結果は不変。
- on/off 比較で記録すべき指標: RMSE_Z（Procrustes 後）、w·mean(‖z‖²) の反復推移
  （スケールドリフト）、‖F‖、Q̂ 推移、held-out Y ll、nan 発生率。
- 軽量テスト（実装時）: off でも小設定で NaN なく走ること、on で比較対象の
  決定的な数値結果が既定動作と一致すること。

## 4. P1-4: Z 点推定比較の設計（実験なし）

比較対象: (a) 最終 1 サンプル（現行 `Z_est`）、(b) L サンプル平均、
(c) 最終 Newton mode（サンプリング直前の点）、(d) 最終 EM 反復の全サンプル平均。
注意: (b)(d) は O(k) 回転がサンプル間で揃っている前提が必要（現行は同一チェーン内
なので実用上は近いが、平均前に基準サンプルへの Procrustes 整列を行う設計とする）。
記録指標: Procrustes 後 RMSE(Z)、Gram 行列誤差 ‖ẐẐ^T − ZZ^T‖_F/n²、
リンク確率/強度の RMSE、held-out log loss・AUC・AP、サンプル間分散。
実装位置: 新規診断関数（モデル・runner の既定挙動は不変）。

## 5. P1-8: MC サンプル依存性の診断設計（実験なし）

- 現行サンプルは §2.1 の逐次過程であり、定常チェーンの保証がないため
  「effective sample size (ESS)」という語は使わず
  `sample dependence diagnostic` / `lag correlation diagnostic` と呼ぶ。
- 設計: (i) L を 5→10→20 に増やしたときの Q̂・BIC・選択 k の変動、
  (ii) サンプル列 Z^{(l)} の要素ごと lag-1 相関（回転整列後）、
  (iii) チェーン継続 vs 独立初期化（seed 分離）での Q̂ 分散比較。
- いずれも小規模設定（n≤80）で設計し、本実験は承認後。

---

## 6. 本日実装した診断の意味と限界

| 実装 | 意味 | 限界 |
|---|---|---|
| `clip_diag`（em_runner 新キー、`compute_clip_diagnostic=True` のopt-in） | **最終推定値**における Poisson 自然パラメータの clip 域該当率。既定Falseでは計算せずNone | EM 反復中の発動率ではない。発動率 0 でも反復中に発動した可能性は残る（反復中計測は将来課題） |
| `mstep_q_history` | 同一 Z_samples・同一 mask での M-step 前後の strict Q 差（M-step 単調性診断） | E-step を跨ぐ Q 変化は診断対象外（サンプルが変わるため）。GEM 全体の単調性は依然未検証 |
| `failure_reason` / `q_bic_failed` | Q/BIC 計算失敗の可視化 | 失敗原因の分類はメッセージ文字列のみ |
| `validate_family_support` ほか | family と台の整合の事前検査（opt-in） | 分布の妥当性（過分散等）は検査しない。台の整合のみ |

### 6.1 後方互換性確認の範囲（2026-07-19再検証）

比較対象は `Z_est`, `Z_samples`, `F`, `sigma`, `w0`, `w`, `var_z`,
`sigma_y_est`, `Q_strict`, `bic`, `num_params`, `nan_occurred`, `nan_count` の
**比較可能な数値キー13件**である。masked構成13件、per-column構成13件の
合計26比較を行う。`model`は可変内部状態を持つオブジェクト、`runtime_s`は壁時計時間
なのでbit比較対象外であり、全キー同一とは呼ばない。

変更前snapshotは削除済みで、Codexはbefore/afterを独立再現できなかった。現在の
テストで再現できるのは、変更後コード内で診断off/on時に上記13数値結果が同一で
あることまでである。診断有効時の`runtime_s`には追加計算時間が含まれ得る。
