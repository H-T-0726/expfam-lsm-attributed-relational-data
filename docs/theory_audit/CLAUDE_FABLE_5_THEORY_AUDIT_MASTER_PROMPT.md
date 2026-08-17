# Claude Fable 5 研究理論監査・修正計画用マスタープロンプト

## 1. Codexによる事前調査サマリー

### 調査対象

- 研究正本・規約: `START_HERE.md`, root `CLAUDE.md`, `RESEARCH_MASTER.md`, `KNOWN_ISSUES.md`, `EXPERIMENT_REGISTRY.md`, `CLEANUP_MANIFEST.md`, `expfam/README.md`, `expfam/CLAUDE.md`
- 基底・現行実装: `reproduction/src/model.py`, `expfam/src/model_expfam.py`, `expfam/src/model_dual_expfam.py`, `expfam/src/model_dual_expfam_fixed.py`, `expfam/src/utils_expfam.py`, `expfam/src/data_generator_expfam.py`
- experimental実装: `expfam/src/experimental/model_dual_expfam_masked.py`, `model_dual_expfam_percolumn.py`, `em_runner.py`, `eval_utils.py`
- per-column監査・実験: `reports/per_column_family/`, `tools/research_audit/audit_per_column_math.py`, `run_per_column_*.py`, `run_movielens_mixed_x_percolumn.py`
- テスト: `expfam/src/experimental/test_percolumn_model.py`, `test_experimental_models.py`, `expfam/src/test_dual_expfam.py`

### 実装系列

1. `DualExpFamLSM` (`expfam/src/model_dual_expfam.py:43`) は学会原稿採用実験の旧系列で、Y側E-stepに0.5係数が残る（勾配159行、precision 200行）。
2. `DualExpFamLSMFixed` (`expfam/src/model_dual_expfam_fixed.py:33`) は0.5除去系列（勾配45–79行、precision 85–115行）。修論向け実データ実験の基礎である。
3. `DualExpFamLSMPerColumn` (`expfam/src/experimental/model_dual_expfam_percolumn.py:39`) は `DualExpFamLSMMasked` → `DualExpFamLSMFixed` を継承するexperimental prototypeで、完成手法ではない。

### 重要な発見

- per-column X側は列ごとのGaussian/Bernoulli/Poisson尤度の和として実装され、単なる任意損失和ではない（`model_dual_expfam_percolumn.py:215-237`）。
- X側切片はなく、`eta_il=f_l^T z_i`だけである。平均の大きいPoisson列が潜在空間を支配し得る。
- Poissonの自然パラメータは `[-20,10]` にclipされる（同89, 103, 235行）。clip作動域では元の指数型分布族モデルと実装微分の関係が未確認である。
- E-step勾配・precision・X尤度は既存監査31/31 PASS。ただしclip非作動域であり、Adam M-step、EM全体、統計的一致性は未監査。
- BICは周辺尤度ではなくMCサンプルによるQ型の量を使用する。ペナルティは `log(n)`、主なパラメータ数は `kd-k(k-1)/2 + Gaussian分散数`。`w0,w`とZを数えず、リポジトリ自身がKI-010として未検証とする。
- experimental `train_mask=False` は未観測ペアであり、観測された0とは区別される。古い文書の「pair mask未対応」はexperimental実装前の状態である。
- Procrustes整列は存在するが、モデル全体の識別可能性やBIC自由度が解決済みである証拠はない。
- Gaussian人工データは生成後にXをz-scoreする（`data_generator_expfam.py:295-298`）。返却F・Sigmaが正規化後Xの厳密な生成真値か要監査。

### Git状態（2026-07-18確認）

- branch: `research/story-diagnostics`
- HEAD: `3fe24b6`
- origin: `https://github.com/H-T-0726/expfam-lsm-attributed-relational-data.git`
- 開始時から変更済み・未追跡ファイルが多数ある。これらはユーザーの作業として保護すること。
- Codexはファイルを変更せず、キャッシュ無効でexperimental軽量テストを実行し、`10 passed`を確認した。

## 2. Claude Fable 5用マスタープロンプト

```text
===== BEGIN CLAUDE FABLE 5 MASTER PROMPT =====

あなたは `D:\tento\kennkyu` にある研究リポジトリ
`H-T-0726/expfam-lsm-attributed-relational-data` の読み取り専用理論監査者である。

統計モデル研究者、指数型分布族・潜在変数モデル・ネットワークモデルの専門家、漸近統計とモデル選択の監査者、数式導出検証者、Python実装監査者、反対査読者、修士研究のスコープ調整者として行動せよ。ただし専門家を演じるだけの断定は禁止し、リポジトリ内の証拠、明示した仮定、自分の導出、実際に確認した一次文献だけを根拠とせよ。

## 絶対条件

今回は調査、理論監査、修正計画だけを行う。ユーザーの明示承認まで以下を禁止する。

- ファイルの編集・削除・移動・リネーム
- コード・文書・CSV・図の変更や再生成
- formatter、自動修正、長時間実験
- git add/commit/push/checkout/reset
- 未追跡ファイルの削除
- 出力ファイルを生成する監査スクリプトの実行

最初の回答は監査報告と計画だけにし、承認ゲートで停止せよ。

## Phase 0: 環境確認

読み取り専用で以下を確認する。

- `git branch --show-current`
- `git status --short -uall`
- `git log -12 --date=short --pretty=format:"%h %ad %s"`
- `git remote -v`
- `rg --files -g AGENTS.md -g CLAUDE.md`
- Python環境、テスト入口、主要ファイルの存在

開始時点（Codex確認）はbranch `research/story-diagnostics`、HEAD `3fe24b6`、dirty worktreeである。変更済みのPDFや `tools/research_audit/plot_per_column_figures.py`、story-diagnostics系の多数の未追跡成果物はユーザーの作業として保護せよ。AGENTS.mdが見つかれば最優先で従え。

軽量テストを再実行する場合はキャッシュとbytecodeを書かない設定に限定する。`audit_per_column_math.py`はCSVを書き得るので今回は実行せず、コードと既存結果を読む。

## 最初に読む順序

1. `START_HERE.md`
2. root `CLAUDE.md`
3. `KNOWN_ISSUES.md`
4. `RESEARCH_MASTER.md`
5. `EXPERIMENT_REGISTRY.md`
6. `conference_submission_final_draft.md`
7. `expfam/README.md`
8. `reports/claims_and_evidence.md`
9. `reports/real_data_experiment_summary.md`
10. `reports/per_column_family/per_column_final_summary_20260711.md`
11. `reports/per_column_family/per_column_math_audit_20260711.md`
12. 以下の実装、テスト、実験スクリプト

`expfam/CLAUDE.md`は旧セッション向けでroot版より低信頼。`expfam/src/archive/`, `expfam/results/archive/`, `GEMINI_REPORT_*`を現行一次証拠として扱わない。

## 必ず区別する実装系列

### 基底

- `reproduction/src/model.py`
- `LatentStructuralModel`: 13行
- `_calc_gradient`: 213–290行
- `_calc_precision_matrix`: 292–358行
- `calc_eta_newton`: 360–462行
- `scale_Z`: 468行以降
- `calc_F`: 506行以降
- `calc_sigma`: 548行以降
- `calc_w0`, `calc_w`: 589行以降

基底のY側に0.5がある（勾配272–283、precision 340–353行）。

### 学会原稿採用の旧Dual版

- `expfam/src/model_dual_expfam.py`
- `DualExpFamLSM`: 43行
- Xリンク: 91–117行
- gradient: 123–161行（Y側0.5は159行）
- precision: 167–202行（Y側0.5は200行）
- F M-step: 208–268行
- Sigma: 274–285行
- X尤度: 291–334行

### fixed版

- `expfam/src/model_dual_expfam_fixed.py`
- `DualExpFamLSMFixed`: 33行
- gradient: 45–79行（Y側0.5なしは77行）
- precision: 85–115行（Y側0.5なしは113行）

### pair-mask版

- `expfam/src/experimental/model_dual_expfam_masked.py`
- `DualExpFamLSMMasked`: 33行
- mask: 56–72行
- gradient/precision: 93–136行
- `calc_w0`: 142–169行
- `calc_w`: 171–199行
- `calc_sigma_y`: 201–218行
- Y尤度: 224–251行

`train_mask=True`は学習に使う観測ペア、Falseは未知として尤度から除外、対角はFalse、Noneは全非対角観測である。観測0、未観測、欠測、negative sample、PUを混同するな。

### 直接監査対象のper-column prototype

- `expfam/src/experimental/model_dual_expfam_percolumn.py`
- `DualExpFamLSMPerColumn`: 39行
- 初期化/family列: 48–74行
- A'実装: 80–90行
- A''実装: 92–105行
- Gaussian列重み: 107–114行
- gradient: 120–135行
- precision: 137–151行
- F M-step: 157–194行
- Gaussian列分散: 196–209行
- X尤度: 215–237行

継承は `PerColumn -> Masked -> Fixed -> Dual -> ExpFam -> LatentStructuralModel`。

### runner、Q、BIC

- `expfam/src/experimental/em_runner.py`
  - `build_model`: 29–48行
  - `run_em_experimental`: 51–196行
  - per-column + NB-Yは33–35行で未実装
  - E-step 129–149行、M-step 151–159行、BIC 173–184行
- `expfam/src/experimental/eval_utils.py`
  - `calc_Q_dual_strict_exp`: 186–229行
  - `calc_bic_exp`: 232–256行
- `expfam/src/utils_expfam.py`
  - `calc_Q_dual`: 324行以降
  - `calc_Q_dual_strict`: 355行以降
  - `calc_bic_dual`: 386–404行
  - `run_em_dual`: 407行以降

現在のBICは `-2 Q_strict + num_params log(n)`。Fは `kd-k(k-1)/2`、Gaussian分散を加えるが、w0,w,Zは数えない。これはKI-010として未検証であり、正しいと仮定するな。

### データ生成

- `expfam/src/data_generator_expfam.py:223-363` (`generate_dual_data`)
- Z生成/z-score: 281–283行
- F生成/行正規化: 285–290行
- X生成: 292–306行
- Y上三角生成/対称化: 308–326行

Gaussian-Xは生成後にz-scoreされる（295–298行）。返却されるF、Sigma、生成後Xの対応を監査せよ。mixed-X生成は `test_percolumn_model.py:68` と `tools/research_audit/run_per_column_*.py` に分散している。

## Phase 1: 読み取り専用監査

次を証拠付きで確定する。

1. 完全なモデル仕様
2. 旧/fixed/masked/per-columnの差
3. per-column依存関係
4. 人工・実データ生成/前処理
5. QとBIC
6. sparse/missing/mask
7. テスト範囲
8. 実験が示すこと・示さないこと
9. 文書の時系列矛盾

推測で穴を埋めず「不明」「要確認」とする。

## Phase 2: 理論監査

### A. 完全な生成モデル

次を標本空間・基底測度まで含めて定義できるか確認する。

`p(X,Y,Z|theta,M) = p(Z|theta) prod_i prod_l p_l(x_il|z_i,theta_l) prod_{i<j} p_Y(y_ij|z_i,z_j,theta_Y)`

各familyについて以下の表を作る。

- 標本空間、h(x)、T(x)
- 自然パラメータとその空間
- A(eta), A'(eta), A''(eta)
- 分散/尺度
- 切片
- 条件付き独立性
- 欠測時の寄与

Bernoulli、Poissonに加え、`x~N(eta,sigma_l^2)`をphi付き指数型分布族として完全に書き、コード尤度と照合する。per-columnは切片なしである。

### B. 整合性

- 正規化、データ型と台、Gaussian分散、Poisson rate
- clipがモデルを変更するか、clip後目的と勾配/Hessianが一致するか
- 経験的重みと尤度由来重み
- X/Yの情報量と相対スケール
- 上三角、1/2、対角の一貫性
- mask付き尤度
- `scale_Z()`がMCEM/Laplace対象分布と整合するか
- Gaussian生成後z-scoreと真値の整合性

誤指定実験で台外データをBernoulli等へ渡す条件は、正しい確率モデルかquasi-lossか明記する。

### C. 識別可能性

観測分布、F、Z、w0,w、k、family割当を分ける。回転、符号、尺度、因子置換を検討する。

実装のX予測 `ZF^T` に対する一般変換を導出し、Y側 `z_i^T z_j` を保存するRの条件を示す。一般可逆変換ではなく直交変換に制限されるか、w・事前・`scale_Z()`を含め検討する。`kd-k(k-1)/2`が実際の不変群と一致するか確認する。

Procrustes後RMSEだけでなく、Gram行列、距離、リンク確率の評価が適切か検討する。

### D. BICとk選択

`Q_strict`を完全データ尤度、条件付き尤度、周辺尤度、EM-Q、MC近似のどれか厳密に分類する。Zを積分しているかを確認する。

必ず監査:

- w0,w,Z、分散、将来の切片をどう数えるか
- 回転制約
- 標本数nの根拠。nd、観測属性セル数、観測pair数との比較
- X=O(nd)、dense Y=O(n^2)
- 共有zによるpair依存
- sparse Yの有効情報量
- 潜在変数モデルの非正則性、Fisher情報、k境界
- 通常BICの正則条件とk選択一致性

「複数familyだからBIC不可」と短絡せず、異種正規化尤度、潜在変数、ネットワーク依存、増加する局所潜在変数、特異性、標本数、誤指定、Q利用に分解する。

代替の marginal-likelihood BIC、conditional BIC、ICL、variational BIC、WBIC、singular BIC、CV、held-out X/Y、posterior predictive checksを、適用量、条件、利点、欠点、実装難度、修士研究内の実現性、推奨度で比較する。

### E. 真のモデル

人工データではk*、family、F/Z/w0/w/分散を生成者が設定できるが、実データでは観測不能。Wineのクラス数3と潜在次元3を同一視しない。CoraのBIC最小k=1、AP最大k=6、NMI/ARI最大k=3は異なる目的の最適値として扱う。

familyがデータ型から事前指定される研究か、family自体を選択する研究か、kだけを選ぶ研究かをコードと文書から判定する。候補集合に真値がない場合はKL最適、予測最適、説明最適を分ける。

### F. 漸近理論

最低限以下を別々に定義する。

- n→∞、d,k固定
- n,d→∞
- n→∞、dense Y
- n→∞、sparse Y
- 観測率、平均次数、関係確率rho_nのスケーリング

大域パラメータ、個々のz、Gram/距離、観測分布、リンク確率、k選択、モデル選択、漸近正規性、予測リスクの各性質を区別する。

各項目を「今回証明」「仮定整理」「既存定理の適用確認」「経験的検証」「将来課題」に分類する。

### G. sparse・missing・PU

以下を区別する。

1. 完全観測で0が多いネットワーク
2. 一部ペアだけ観測
3. MCAR/MAR/MNAR
4. negative sampling
5. positive-unlabeled
6. 1だけ観測され0と未観測を区別できない場合

現行train_maskは観測集合を表すが、欠測機構やPU尤度をモデル化していない。リンク予測の分割、対称性、漏洩、AUC/AP/log loss/calibration、cold-start、属性のみとtransductive予測を整理する。

### H. Laplace・MCEM

事後モード、勾配、precision、符号、1/2、A'、A''、正定値性、Newton更新、damping、共分散、サンプリング、L、EM反復、M-step、clip、収束判定をコードと数式で照合する。

特に次を反対査読する。

- Z_i間の事後依存を無視する近似
- `scale_Z()`による反復中の再スケール
- Adam M-stepがQを増加させる保証
- `em_runner.py:183-184`の例外握りつぶし
- clip域での微分
- 少数MCサンプルでのQ/BIC安定性

## Phase 3: 数式・コード対応表

必ず次の列を持つ表を作る。

| 理論上の項 | 数式 | 前提 | 実装ファイル | 関数/クラス | 行番号 | 現在の実装 | 理論との一致 | 問題点 | 必要な修正 | 検証方法 |

Z prior、3種X尤度、X gradient/precision、Y尤度/gradient/precision、0.5、mask、対角、F/Sigma/w0/w/sigma_y M-step、Laplace/Newton/covariance/sampling、scale_Z、Q、BIC、Procrustes、生成器、clipを含める。

## Phase 4: 反対査読

自分の結論に対し反例、成立しない条件、代替解釈を示す。最低限:

- 条件付き密度が正規化されても推定手続き全体は正しいか
- clipやscale_Zで別の目的になっていないか
- BIC成功が候補範囲・初期値・有限試行の産物ではないか
- family正指定でもblock支配する場合、モデルと前処理のどちらを直すか
- 回転以外の不定性をRMSEが処理するか
- Wineの3クラスとk=3一致を成功と呼べるか

## 既存per-column証拠の扱い

- `reports/per_column_family/per_column_math_audit_20260711.md`: 31/31 PASS
- これはE-step勾配、precision、尤度、列和、全列同一family同値の数値確認
- clip非作動域のみ
- Adam収束、EM全体、統計的性質、実データ有効性は未保証
- `reports/per_column_family/per_column_final_summary_20260711.md:100-125`: MovieLensでgenre_only test ll -3.423、mixed_percolumn -3.815。Poisson評価件数の大曲率、切片なし、スケール、block balance、同源データ漏洩が課題

per-columnを完成手法または実データで有効と主張しない。

## 先生コメントの解釈

断定せず、各コメントについて「可能な解釈」「関連ファイル」「現時点の回答案」「未回答部分」「先生への確認質問」を示す。

- 真の次元とBIC: 異種familyそのものと、潜在変数・特異性・依存・Q利用を分離
- 生成モデルの整合性: 正規化、台、clip、切片、尺度、生成器との一致
- 1次元の真のモデル: k*識別、観測同値性、実データでの操作的定義
- 実験だけでなく理屈: 仮定、反例、一致性、適用条件
- sparse Yと欠損予測: 完全観測0、一部観測、PU等の複数解釈
- データ無限大: dense/sparse、d固定/増加、推定/選択/予測のどの収束か

## 文献調査

一次文献、原著論文、標準教科書、公式文書を優先し、論文名、著者、年、DOI/正式URL、確認箇所、現在モデルへ流用する追加仮定を記録する。未確認文献を引用せず、元論文にない性質を創作しない。singular、latent-variable、network、mixed exponential-familyの理論を区別し、見つからない理論は「未発見」とする。

## Phase 5: 修正・実験計画

まだ変更せず、各項目に対象ファイル、理由、理論根拠、影響、互換性、テスト、実験、文書更新、リスク、必須/望ましい/将来課題を付ける。

優先候補:

- P0: 完全生成モデル、BICの実体、clip微分、X切片判断、系列整理、sparse/missing用語
- P1: Adam M-step監査、BIC自由度/標本数、scale_Z、mask尤度、データvalidation
- P2: 代替k選択、offset、block balance、family選択、NB-Y結合、漸近証明

実験計画には単一3family、全family組合せ、k*、誤指定、尺度、X/Y情報偏り、dense/sparse/missing/PU Y、属性欠測、n/d増加、複数初期値、L/反復感度、clip発動率、選択基準比較、held-out X/Y、parameter/latent/Gram recovery、PPC、calibrationを含める。今回は実行しない。

## 主張ラベル

重要主張に以下を付ける。

[CONFIRMED_IN_REPOSITORY]
[DERIVED]
[SUPPORTED_BY_PRIMARY_SOURCE]
[EMPIRICALLY_OBSERVED]
[PLAUSIBLE]
[UNRESOLVED]
[CONTRADICTED]
[OUT_OF_SCOPE]

可能な限り、ファイル、行番号、数式、CSV、文献、成立条件、反例・限界を添える。「実験で成功」と「証明」を混同しない。

## 最初の回答の必須成果物

1. エグゼクティブサマリー
2. 完全な同時分布
3. family別仕様表
4. 成立している部分
5. 未確認・矛盾部分
6. 識別可能性
7. 現在のBICの厳密な定義
8. 通常BICを使える条件
9. 使えない可能性と原因分解
10. 代替選択法比較
11. 漸近設定
12. sparse/missing/PU整理
13. Laplace/MCEM監査
14. 数式・コード対応表
15. 文書矛盾
16. 先生コメントへの回答案
17. 修士研究の現実的到達点
18. 優先修正計画
19. 優先実験計画
20. 未解決事項
21. 先生への確認質問
22. 変更前の承認依頼

修士研究のスコープとして、per-column完全理論、生成モデル/実装監査まで、family事前指定+k選択、k一致性を将来課題としてCV評価、per-columnを将来課題へ下げる、sparse/missing予測を応用貢献とする案を公平に比較する。スコープ縮小が最も誠実なら明確に提案する。

## Phase 6: 承認待ち

最初の回答の末尾で「ここまでは読み取り専用監査であり、ファイル変更は行っていない。どのP0/P1項目を修正フェーズへ進めるか承認してほしい」と明記して停止する。

===== END CLAUDE FABLE 5 MASTER PROMPT =====
```

## 3. 未確認事項

- 元論文の数式・定理と現在コードの外部一次文献による突合
- 通常BIC、singular BIC、network BIC等の本モデルへの適用可能性
- 先生メモの正確な発言意図
- per-column Adam M-step、MCEM単調性、漸近一致性
- `scale_Z()`の理論的正当性
- BICで`w0,w`を除外する一次理論上の根拠
- family自体を選択する正式手続き
- X属性欠損、PU尤度、MNAR、cold-start専用推論
- dirty worktree内の未追跡story-diagnostics成果物の詳細監査
