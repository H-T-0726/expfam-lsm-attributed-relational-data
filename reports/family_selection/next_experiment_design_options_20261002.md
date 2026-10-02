# 次の family-selection 実験の設計候補（実行しない）（2026-10-02）

- 位置づけ: **DESIGN ONLY**。**どの候補も実行しない。**Human が `x_intercept_research_decision_memo_20261002.md` の
  Option A / B / C を決めた後に、別の Approved Task として scope・seed・出力先を固定してから走らせる候補である。
- 新しい EM 0、dataset 0、seed run 0、コード変更 0。本文中の数値はすべて決定論的な数値積分（乱数なし）。
- 前提: `family_selection_hard_case_audit_20261002.md`（以下「監査」）。特に監査 §5：
  **0/1 列では、同じパラメータ数の尤度 score は真の family によらず必ず Bernoulli を選ぶ**（intercept の有無によらない）。

---

## 0. 設計の原則

「できるから実験する」ではなく、**どの claim を守る／強めるために必要か**で選ぶ。
監査 §5 により、現行 score の「0/1 列での Bernoulli vs Poisson の正解率」は**実験をしなくても決まっている**（真 Bernoulli → 常に正解、真 Poisson → 常に不正解）。
したがって **selector の正解率を測る実験は、どの Option でも新しい情報を生まない**。
実験に意味があるのは、(i) gate が Poisson を検出できる確率（解析的に計算できる）を超える何か、
すなわち **Bernoulli で代用したときの下流への影響**、または (ii) score 自体を変える方法論の検証だけである。

---

## 1. 候補一覧

| ID | 前提 Option | 何を問うか | 守る / 強める claim | 推奨 |
|---|---|---|---|---|
| **D0** | A / B / C | 実験しない | 現行 claim は監査の理論で十分に限定できる | **Human 決定までは D0** |
| D1 | A / C | support 検出確率の解析曲線（実験ではない） | 「family 割り当ては support で決まる」を図で示す | 修論に載せるなら有用（計算のみ） |
| D2 | A | no-intercept のまま小さい n で 0/1 の真 Poisson を作る | — | **非推奨**（§3） |
| D3 | B | Bernoulli 代用の下流コスト（Z 回復・K 選択・予測） | 「support gate による割り当ては下流でどれだけ効くか」 | B を選ぶなら最優先 |
| D4 | B / C（方法変更） | Poisson を選びうる score（事前オッズ・予測評価）の理論 | selector を「判別」できるものにする | 理論が先。selector 変更は Human Gate |
| D5 | B | 依頼にある「公平な Bernoulli vs Poisson benchmark」 | — | **非推奨**（§5：結果は事前に決まる） |

---

## 2. D1：support 検出確率の解析曲線（Option A / C、計算のみ）

- 量: `P(gate が Poisson と判定) = 1 − p(b, s)^n`、`p(b, s) = E_{η~N(b,s²)}[e^{−λ}(1+λ)]`。
- no-intercept（b = 0）: n = 75 で ≥ 1 − 1.01×10⁻¹⁰（監査 §4）。
- 図: 横軸 n（1–200）、縦軸 `P(全観測 ∈ {0,1})`、曲線は s² ∈ {0, 0.5, 1}。b = 0 では n ≈ 10 で 0.05 を切る。
- 必要な run: 0（Gauss–Hermite 求積のみ）。
- claim: 「現行モデルでは score が判断する列は事実上すべて真 Bernoulli であり、family 割り当ては support で決まる」。

## 3. D2：no-intercept のまま hard case を作る（非推奨、理由）

- **NUMERICAL** 最良の場合（f_l = 0）でも `P(全 n 観測 ∈ {0,1}) ≥ 0.05` には **n ≤ 9**、≥ 0.5 には **n ≤ 2** が必要。
  s² = 0.5 では n ≤ 7 / n ≤ 1。n = 75 の既存条件では 1 列あたり ≤ 1.0×10⁻¹⁰。
- n ≤ 9 では Y（関係データ）も K 選択もほぼ意味を持たず、既存 Phase 9 の条件と比較できない。
- 0/1 の列だけを選び出す（棄却サンプリング）と canonical draw ではなくなり、seed rescue 禁止（generator M3）にも反する。
- **そもそも作れても結果は決まっている**（監査 §5：必ず Bernoulli）。
- **結論: 事実上不可能かつ無情報。設計しない。**

## 4. D3：Bernoulli 代用の下流コスト benchmark（Option B 採用時のみ）

**前提**: X 列 intercept を持つ新 lineage が Human に承認・実装・検証済みであること（b = 0 固定で既存 lineage と一致する回帰テストを含む）。

**research question**: 低 rate の真 Poisson 列が有限標本で 0/1 にしか見えず support gate + score によって Bernoulli と割り当てられたとき、
oracle の Poisson 割り当てと比べて (a) Z の回復、(b) K の選択（C_Lap / C_Q）、(c) X の held-out 予測（2 以上が出たときの support 違反を含む）
がどれだけ変わるか。

**条件（案、NUMERICAL は intercept ありの理論値、s² = ‖f‖² = 0.5、n = 75）**:

| 真 Poisson の平均 m | b_P | P(X=1) | P(全 75 観測 ∈ {0,1}) | Poisson の E[A''] | P(X=1) を揃えた Bernoulli の b_B | Bernoulli の E[A''] | margin 下界 2nE[g] |
|---|---|---|---|---|---|---|---|
| 0.1 | −2.553 | 0.0855 | 0.592 | 0.100 | −2.570 | 0.0749 | 1.06 |
| 0.3 | −1.454 | 0.194 | 0.027 | 0.300 | −1.566 | 0.145 | 7.73 |

- m = 0.1 は「0/1 にしか見えないことが多い」側、m = 0.3 は「たいてい 2 以上が出て gate が Poisson と判定する」側の対照。
- **比較の腕**: 同じ dataset に対し (i) support gate + score の割り当て（0/1 列は Bernoulli）、(ii) oracle の真 family 割り当て、の 2 つの fit。
  dataset を共有するので対応のある比較になる（Phase 9U の成分分離の考え方を流用）。
- **固定**: n = 75, d = 12, K_true = 3, Y = Bernoulli（w0, w は Phase 9K と同じ）、X = G3 / B6 / P3（P3 の b_P を上表で設定、B6 は b = 0 のまま）、
  L = 5、探索・refit 8/8、候補 K = 1..5、start_B のみ。
- **最小 replicate**: 2 rate × 10 dataset = 20 dataset、各 dataset で 2 腕 × K 5 = 10 fit（計 200 fit）。
  10 dataset は Phase 9X の 1 セルと同じ規模で、「傾向の記述」には足りるが確率の推定ではない。
- **seed**: 新しい seed family を事前に固定（例 `1301000 + r`）。retry / replacement / seed rescue 0。
- **注意（交絡）**: b_P < 0 は同時に X 側の情報（E[A''] = m）を小さくする（no-intercept の Poisson 列は s² = 0.5 で E[A''] = 1.284）。
  K 選択の差は「family の誤割り当て」と「X 信号の減少」の両方を含むので、腕 (i)(ii) を**同じ dataset で比べる**ことが本質（データ側の信号は共通、違いは割り当てだけ）。
- **測らないこと**: selector の正解率（監査 §5 で決まっている）。

## 5. D5：依頼にある「公平な Bernoulli vs Poisson benchmark」（設計のみ、非推奨）

依頼の条件（真 Bernoulli・真 Poisson、同程度の binary rate、同程度の signal strength）は intercept があれば作れる:

- binary rate を揃える: 真 Poisson の P(X=1) に Bernoulli の `b_B` を合わせる（上表）。
- signal strength を揃える: 1 列が z にもたらす情報は `E[A''(η)] · ‖f‖²` に比例するので、
  family ごとに ‖f‖² を `E[A''_P] ‖f_P‖² = E[A''_B] ‖f_B‖²` となるよう調整する（b と ‖f‖ を同時に解く 2 次元の較正）。
- 最小構成: 2 rate × 10 dataset、各列 family を 1:1。

**しかし結果は実行前に決まっている**: 0/1 になった列は真の family によらず Bernoulli、2 以上が出た列は Poisson（gate）。
したがって「正解率」は `P(真 Bernoulli) · 1 + P(真 Poisson) · (1 − p^n)` で解析的に計算でき、run は新しい情報を生まない。
観測できるのは margin の分布だけで、それは選択を変えない。**この設計は推奨しない**。同じ資源を使うなら D3。

## 6. D4：Poisson を選びうる score（方法論、理論が先）

0/1 データで Poisson が勝つには、尤度以外の情報が必要（監査 §5.3）:

- family の事前オッズ（例: 「count として収集された変数」という外部情報）。
- 2 以上を含みうる held-out データでの予測評価（Bernoulli は 2 以上に確率 0 を与えるので、support の外挿リスクを測れる）。
- 実データでの変数の意味（測定の仕組み）に基づく人手指定。

いずれも **selector の目的関数の変更**であり、今回は禁止事項・Human Gate。まず理論メモ（どの情報を使えば何が識別できるか）を作るのが先で、実験はその後。

---

## 7. まとめ

| Option | 推奨する次の実験 |
|---|---|
| A（維持＋限定） | **なし**（D1 の計算図は任意） |
| B（intercept 追加） | 実装・検証の後に **D3**（D5 は非推奨） |
| C（prototype 扱い） | **なし**（D1 の計算図は任意） |

**現時点の recommended next experiment: NONE YET**（Human が Option を決めるまで）。
