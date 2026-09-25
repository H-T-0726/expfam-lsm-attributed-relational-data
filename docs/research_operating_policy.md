# Research Operating Policy — 研究を主役にする詳細規約

> この文書は root `CLAUDE.md` の **Research-first operating policy** の詳細版。
> 実験設計、gate 設計、severity 判断、結果解釈、multi-stage 実行を行うときに読む。
> root `CLAUDE.md` と矛盾する場合は root `CLAUDE.md` を優先する。

---

### Research-first operating policy（研究を主役にする）

この repository の主目的は**研究上の問いに、再現可能で過剰主張のない証拠で答えること**である。
コード品質・CI・gate・audit・最適化精度はそのための手段であり、
**green check を増やすこと、数値的完全性そのもの、gate を通すことを研究目的にしない。**

優先順位は原則として次の順とする。

1. **scientific validity** — 数式・目的関数・データ生成・比較設計が研究問いに対応している
2. **evidence integrity** — 一次 artifact / provenance / lineage が追跡できる
3. **reproducibility** — seed・設定・コード版・環境が記録され、同じ条件を再現できる
4. **engineering quality** — test / CI / refactor / usability

4 が 1〜3 を支える範囲では厳密に行うが、4 のために研究を止め続けない。

#### 研究タスクの mode を最初に決める

作業開始時に、そのタスクの主 mode を次から 1 つ明記する。

- **THEORY / AUDIT** — 数式・仮定・実装対応・既存証拠の確認。新しい科学的結果は作らない
- **IMPLEMENTATION** — 承認済み数式・設計をコード化する。新しい研究結論は出さない
- **EXPLORATORY** — 仮説生成・失敗条件探索・設計感度の把握。結果を見て次を考えてよいが post-hoc と明記する
- **CONFIRMATORY** — 事前に固定した問い・条件・主要指標に対する検証。結果を見て protocol を変えない
- **REPRODUCTION** — 既存結果・先行研究・既存 artifact の再現。元条件との差を混ぜない
- **CHARACTERIZATION** — 実データや条件差の性質を記述する。因果・一般性を勝手に主張しない

複数 mode が必要なら phase を分ける。探索で見つけた条件を、そのまま confirmatory evidence として扱わない。

#### 実験前に固定するものは「研究解釈に必要な最小限」にする

実験前に、少なくとも次を明記する。

- **Research question / claim** — 何を知りたいか、何を言えるようにしたいか
- **Primary estimand / primary metric** — 最重要の観測量
- **Comparison** — 何と何を比較するか。比較しないならその旨
- **Fixed conditions / varying conditions** — 何を固定し何を変えるか
- **Replicates / seeds / data split policy** — 必要な場合のみ
- **Technical integrity conditions** — run が証拠として有効かを判定する条件
- **Artifacts** — 後から検証するために何を残すか

ただし、**研究解釈に影響しない incidental な defaults・tuple の完全一致・内部実装順序まで frozen spec にしない。**
freeze するのは科学的比較・再現性・一次証拠に必要な contract に限る。

#### evidence label — 事実・観測・解釈を混ぜない

重要な主張は、少なくとも頭の中では次を区別する。文書で混同しそうな場合は明示ラベルを使う。

- **FACT** — 実コード・一次 artifact・原論文などで直接確認できる事実
- **DERIVED** — FACT と明示した仮定から数式的に導いたもの
- **OBSERVED** — 特定条件の実験で観測した結果
- **INTERPRETATION** — OBSERVED を説明する解釈。別説明の可能性がある
- **HYPOTHESIS** — まだ検証していない説明・予想
- **DECISION** — Human が研究 protocol として採用した判断

`OBSERVED -> 一般的に正しい`、`INTERPRETATION -> 原因`、`DECISION -> 理論的正当性` のような昇格を自動で行わない。

#### scientific outcome と technical validity を分離する

実験結果には 2 つの別軸がある。

**A. Technical validity**
- run / artifact が壊れていないか
- protocol を守ったか
- hidden retry / seed rescue / data leakage がないか
- primary metric が正しく計算されたか

**B. Scientific outcome**
- 仮説を支持したか
- 改善したか
- 差がなかったか
- 条件依存・負の結果・不安定性が観測されたか

**科学的に望ましくない結果は FAIL ではない。**
technical validity が保たれていれば、悪化・差なし・不安定・条件依存も研究結果として保存し、解釈する。
PASS/FAIL は主として technical integrity に使い、期待した科学結果が出たかどうかのラベルにしない。

#### severity — BLOCKER / WARNING / DIAGNOSTIC

新しい問題を見つけたら研究上の影響で分類する。

- **BLOCKER**
  - 数式・objective・data generation・mask・evaluation の誤りで結果が無効または解釈不能
  - nonfinite / artifact 破損 / provenance 欠落で一次証拠を信用できない
  - frozen scientific protocol 違反
  - hidden retry / replacement / seed rescue / data leakage
  - primary な選択・主要結論を実質的に変えうることが具体的に示された欠陥
- **WARNING**
  - 結果は利用できるが限定が必要な数値的不安定性・solver warning・感度
  - primary 結論を変えることが示されていない軽微な閾値未達
  - 一部条件での不安定性や limitation
- **DIAGNOSTIC**
  - provenance、性能、debug、将来改善のための観測
  - scientific progression の可否には直接使わない

**測定できる問題だから BLOCKER にする、という判断は禁止。**
WARNING / DIAGNOSTIC は記録し、Approved Task の範囲内では原則として研究を継続する。
Human が frozen spec で明示的に stop condition とした場合だけ、その spec を優先する。

#### 新しい gate を置く前の 4 問

新しい gate / blocker / 数値閾値を追加する前に、必ず次を答える。

1. **どの研究主張または primary estimand を守る gate か**
2. **gate がないと、どのように結果が無効・解釈不能になるか**
3. **WARNING として記録するだけではなぜ不十分か**
4. **閾値の根拠は何か**（理論、既存規約、独立 reference、実質的な結果差）

答えられない場合、その項目は原則 gate ではなく WARNING / DIAGNOSTIC とする。
「念のため」「厳しいほど安全」「測れるから」という理由だけで progression gate を作らない。

#### 数値最適化・収束判定

- solver の `success=False`、precision-loss status、単一 tolerance の未達だけで自動的に BLOCKER にしない
- convergence を progression gate にするなら、**objective gap / selected model / primary metric / conclusion への実質的影響**を示す
- solver の内部 flag より、研究で使う objective と analytic / independent diagnostic を優先する
- 極端に厳しい tolerance を「安全のため」に設定しない。必要精度を研究主張から逆算する
- 結果を見た後に閾値を動かして同じ run を PASS にしない
- 新しい threshold / solver rule は future-facing に preregister し、過去 artifact の判定を遡及変更しない
- 数値差が十分小さく、選択・主要結論・主要指標が安定している場合は、完全 stationarity の不足を limitation として扱えるかを先に検討する

optimizer 自体が研究対象でない限り、proof-level の最適化精度を研究進行の既定条件にしない。

#### negative result / null result を消さない

- 改善しなかった run、悪化した条件、start 依存、失敗条件も一次 evidence として保存する
- 良い seed・良い split・良い replicate だけを選ばない
- retry / redraw / seed replacement が必要なら、その理由と rule を**結果を見る前**に決める
- bug により invalid になった run は削除せず invalid と記録し、新 version / 新 run と分ける
- 「通るまで条件を変える」「良い結果が出るまで experiment を追加する」をしない

#### exploratory と confirmatory を分ける

EXPLORATORY では、結果を見て仮説・条件・可視化を追加してよい。
ただしその発見は exploratory として保存し、重要な主張に使うなら将来の別条件・別 run で確認する。

CONFIRMATORY では、主要条件・主要 metric・比較・stop rule を事前に固定する。
confirmatory run の結果を見て threshold / seed / comparator / metric を変更して同じ主張を救わない。

#### 比較実験の fairness

比較する場合は、研究問いに必要な範囲で条件を揃える。

- 同じ data / split / mask / seed pairing を使うべき比較では揃える
- optimizer budget や initialization を揃えること自体が不公平になる場合は、その理由を明示する
- candidate ごとの正しい parameterization / support / base measure を尊重する
- train/test の情報漏洩、test set を使った tuning、結果を見た split 選択をしない
- best run だけでなく replicate 間のばらつき・勝率・分布を必要に応じて報告する

#### synthetic / real-data の役割を分ける

- **synthetic**: 真値が分かる利点を使い、recovery・misspecification・条件差を調べる
- **real data**: 真の潜在構造や真 family が不明な場合、それらを「回収した」とは言わない
- real data では held-out prediction、再現性、characterization、既知の外部 criterion など、観測可能な評価に限定する
- synthetic で成功しただけで real-data effectiveness や一般性を主張しない

#### claim ladder — 証拠より大きな主張をしない

主張は概ね次の順で強くなる。必要 evidence も順に増える。

1. **IMPLEMENTED** — 実装した
2. **FEASIBLE** — 限定条件で意図どおり動いた
3. **STABLE / ROBUST** — seeds / starts /条件を跨いで安定した
4. **IMPROVES / OUTPERFORMS** — 適切な comparator に対して改善した
5. **GENERAL** — 条件・データ型を跨いで一般化できる
6. **REAL-DATA EFFECTIVE** — 実データ上の外部評価で有効性を示した

下位 evidence から上位 claim を飛ばさない。
smoke 1 回で robustness を、synthetic だけで real-data effectiveness を、prototype だけで manuscript method を主張しない。

#### testing philosophy — 研究 contract を test する

test は**科学的・公開 contract と再現性に必要な invariant**を固定する。

- 数式・objective・support・shape・seed policy・artifact schema・public API の invariant を優先する
- incidental な private implementation、defaults tuple の全要素、内部順序を必要以上に exact-match しない
- opt-in behavior の test は、その opt-in invariant を直接検査する
- refactor で contract が不変なのに brittle test だけ壊れた場合は、研究挙動を戻すのではなく test を contract に合わせて修正する
- regression test で数値を pin する場合は、何の研究上の不変条件を守る pin かコメントする

CI の green は「研究が正しい」の証明ではなく、定義済み contract を満たしたことの確認である。

#### gate proliferation を避ける

- 前の gate が通らなかったこと自体を理由に新しい sub-gate を作らない
- concrete な scientific defect がない限り、「validation の validation」の連鎖を作らない
- 同じ論点で validation-only stage が繰り返されたら、追加 stage より先に
  **何が研究上分かったか / 残る不確実性 / その不確実性がどの claim を妨げるか / limitation として進める選択肢**
  を Human に要約する
- 同じ論点でさらに validation-only stage を追加する場合は、研究上の便益を Human Gate で明示する
- 最低限の implementation correctness が確認できたら、原則として
  **scientific experiment → result interpretation → limitation / next question** に戻る

#### feasibility pilot の既定姿勢

feasibility pilot の目的は、完全な数値解析証明ではなく通常は次を確認することである。

- machinery が意図したデータフローで動く
- hidden repair / seed rescue なしに実行できる
- primary な選択・推定結果と provenance を記録できる
- start / seed / condition に対する安定性と失敗条件を観測できる
- limitation を一次 evidence 付きで説明できる

これを満たす結果は WARNING を含んでいても研究上有用でありうる。
**「完全 PASS でなければ研究失敗」と解釈しない。**

#### 実験を追加する前の stop / value check

新しい実験・ablation・solver 検証を追加する前に次を確認する。

- この追加実験で、**どの未解決な研究問い**に答えられるか
- 結果 A / B のどちらが出ても、研究上の次の判断が変わるか
- 既存 evidence だけで limitation として十分説明できないか
- 修論の主要 claim に必要か、それとも engineering curiosity か

どの結果でも研究判断が変わらないなら、その実験の優先度は低い。

#### advisor / thesis-ready checkpoint

主要 phase・pilot・full experiment の終了時は、コード差分より先に次の 6 点をまとめる。

1. **研究問い**
2. **何をしたか**
3. **一次結果**
4. **何が言えるか**
5. **何はまだ言えないか / limitation**
6. **次に Human が決めること**

この 6 点が説明できないまま validation stage を増やさない。
先生へ相談できる evidence が揃った時点では、追加実装より先に研究上の相談材料を作る。

#### unattended / multi-stage execution

Human が複数 stage の順序・条件・stop rule を**結果を見る前に**承認した場合、
agent はその bounded scope 内で自動進行してよい。
各 micro-step ごとに Human Gate を要求しない。

ただし、次に進む条件が結果を見て初めて決まる場合、研究 scope / model / metric / protocol を変える場合、
または BLOCKER が発生した場合は停止する。
WARNING / DIAGNOSTIC だけなら、事前承認済み scope の範囲で原則継続できる。

#### frozen spec / historical evidence

- 過去 artifact・過去判定・過去 protocol は遡及改変しない
- 新しい rule を採用しても、旧 run は旧 rule の historical evidence として保持する
- strict gate を緩和・再定義する場合は Human Gate で future-facing に承認する
- exploratory finding を後から「事前仮説だった」ことにしない

この原則は研究を甘くするためではなく、**厳密さを研究上意味のある場所に集中させるため**に使う。