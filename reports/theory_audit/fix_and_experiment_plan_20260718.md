# 修正・実験計画（理論監査 2026-07-18）

**本計画は提案のみであり、いかなる変更・実験も未実施。ユーザー承認後に着手する。**
根拠はすべて `theory_audit_report_20260718.md`（以下「報告」）の該当節を参照。

方針（既存規約の継続）:
- 既存クラス・既存 CSV・提出済み原稿の数値は変更しない（新クラス/新関数/新文書の追加で対応）。
- 実験は必ずブランチ上で、seed 分離・runinfo 記録・trials≥3（本実験は ≥5〜10）。
- 誤指定・変換条件は CSV/レポートに変換方法を明記する。

---

## P0（必須: 修論の主張の正しさに直結）

### P0-1. モデル選択基準の再定義・文書化 【報告 §7-9】
- 対象: `expfam/src/experimental/eval_utils.py`（追加関数）、`RESEARCH_MASTER.md`、`KNOWN_ISSUES.md`（KI-010/011 更新）
- 内容: (a) 現行基準を「完全データ型（ICL 型）基準」と改称・文書化。
  (b) ELBO 補正版 `calc_elbo_bic = −2(Q̂ + Ĥ(q)) + p̂ ln n` を**新関数として追加**
  （Ĥ(q) = Σ_i [k/2·ln(2πe) + ½ ln det A_i^{-1}]、A_i は E-step 最終値を再利用）。
  既存 `calc_bic_dual`/`calc_bic_exp` は変更しない。
- 理論根拠: ln p(X,Y) = Q + H + KL（報告 §7）。
- 影響/互換性: 既存 CSV の BIC 列は不変。新列 `elbo_bic` を併記。
- テスト: 人工 k*=3 設定で新旧基準の選択 k 比較、H(q) の符号・大きさの単体テスト。
- リスク: 低（追加のみ）。

### P0-2. Poisson clip の扱いの確定 【報告 §5.1-U1】
- 対象: 文書（`expfam/README.md` or 新規 math note）+ 実験スクリプトへの clip 発動率カウンタ追加（モデル本体は変更しない案を推奨）
- 内容: clip 域では「実装勾配 ≠ 実装尤度の勾配」であることを明文化し、
  実験で clip 発動率を記録。発動率が無視できない条件では結果に注記。
- 理論根拠: 報告 §5.1-U1 [DERIVED]。
- リスク: 低。モデル挙動は不変。

### P0-3. X 側切片（列 offset）の設計判断 【報告 §15.2、story diagnostics 結論】
- 対象: 設計文書（新規）→ 承認後 `DualExpFamLSMPerColumn` 派生 or 新クラス
- 内容: η_il = μ_l + f_l^T z_i の導入設計（M-step 解析解/Adam、BIC 自由度 +d、
  先行研究 eq(2)（バイアス 0）からの逸脱の明示、先生への確認質問 #2 とセット）。
- 理論根拠: raw count Poisson の悪化が切片なしに帰着（story diagnostics、
  attribute diagnosis trials4）[EMPIRICALLY_OBSERVED]。
- リスク: 中（モデル拡張）。**先生の承認を前提**。

### P0-4. 文書の時系列矛盾の解消 【報告 §5.3】
- 対象: `reports/claims_and_evidence.md`（L.21 注記）、`RESEARCH_MASTER.md` §11、
  `KNOWN_ISSUES.md` KI-012、root `CLAUDE.md` 残タスク
- 内容: stale 記述に「(2026-07-18 更新)」付きで現状注記を追加（履歴は消さない）。
- リスク: 低。

### P0-5. 完全生成モデル・用語の正本文書化 【報告 §2-3, §12】
- 対象: 新規 `docs/math_notes/generative_model_spec_20260718.md`（案）
- 内容: 同時分布・family 仕様表・欠測の ignorability 条件・
  「観測 0 / 未観測 / 欠測 / negative sampling / PU」の用語定義・
  生成器の z-score 規約を1文書に確定。
- リスク: 低（新規文書のみ）。

### P0-6. MATLAB 根拠の限定注記 【報告 §4.3】
- 対象: `docs/teacher/half_factor_teacher_reply.md`、`docs/math_notes/half_factor_literature_code_check.md`（追記）
- 内容: calcGrad の w 欠落・calcAi 対角除去の二重変換を記載し、
  「1/2 不要」の根拠を独立導出（§4.1）主体に組み替える。
  **先生へ返答を送る前に反映すべき**（返答が MATLAB を根拠に引く場合）。
- リスク: 低。ただし返答文面の再確認が必要。

---

## P1（望ましい: 推定・評価の信頼性向上）

| # | 項目 | 対象 | 内容 | 検証 |
|---|---|---|---|---|
| P1-1 | H(q) 計測実験 | 新規実験スクリプト | Cora/人工疎 Y で k ごとの Q̂・H(q)・選択 k を記録し、KI-011 の機構仮説を検証 | 報告 §7 の予測（疎→H 正大）と照合 |
| P1-2 | Adam M-step / EM 単調性計測 | runner に Q ロギング追加（挙動不変） | 各 M-step 前後の Q̂ 変化・violation 率を記録 | violation 率レポート |
| P1-3 | scale_Z アブレーション | 実験スクリプト | on/off で RMSE_Z・w·‖Z‖² ドリフト・収束を比較（人工 3 シナリオ、trials≥5） | ドリフト有無の確認 |
| P1-4 | Z 点推定の改善比較 | 実験スクリプト | 最終サンプル vs サンプル平均 vs mode の RMSE_Z / held-out ll 比較 | 差の定量化 |
| P1-5 | em_runner 例外の可視化 | `em_runner.py` L.173-184 | except 節で warning ログ + 失敗理由を結果 dict に記録（挙動互換） | 単体テスト |
| P1-6 | データ validation | 新規 util | family と台の整合チェック（誤指定実験では明示フラグで bypass） | 単体テスト |
| P1-7 | 旧 strict Q の定数不整合 | `utils_expfam.py` に注記 or eval_utils 版へ誘導 | Gaussian-Y ln2π（報告 §5.1-U7）。既存 CSV は再計算しない | 文書照合 |
| P1-8 | MC サンプル感度 | 実験 | L∈{5,10,20}・独立再スタート数の感度、Q̂/BIC の MC 分散推定 | 分散レポート |

## P2（将来課題）

- 代替 k 選択の総合比較（ELBO-BIC / ICL 解釈 / held-out X・Y / PPC / WBIC は文献検討のみ）
- X 側 offset 実装後の per-column 再評価（MovieLens、リーク対策で属性を train-only 化）
- ブロック重み・列数不均衡の理論的定式化（勝手に導入しない方針を維持）
- family 選択手続き（同一台内のみ BIC 可、他は held-out。報告 §8.3）
- NB2-Y と per-column の結合（NotImplementedError の解消）
- 漸近理論: R1（n→∞, dense Y, d 固定）での大域パラメータ一致性の仮定整理 → 証明試行
- sparse regime（w0_n ドリフト）・PU 尤度・MNAR・cold-start
- Gaussian-X 生成後 z-score の真値ずれ定量化（正規化係数を返す拡張）

---

## 実験計画（承認後に実施、今回は未実行）

1. **基準比較実験**（P0-1/P1-1 連動）: 人工 3 シナリオ + 疎 Y sweep（y_obs_rate 1.0→0.1）
   × k∈{1..6} × trials 10 で、BIC_impl / ELBO-BIC / held-out Y ll / RMSE_Z の選択 k を比較。
   clip 発動率・H(q)・nan_occurred を全 fit で記録。
2. **k*=1 最小例**（先生説明用、報告 §15.3）: n=100, k*=1 で Q̂ vs 周辺尤度
   （k=1 なら数値積分で厳密計算可能）を直接比較し、H(q) 欠落の影響を図示。
3. **単一 family 3 種 × 全 9 組合せの回帰確認**: fixed 系列で現行結果の再現性
   （複数初期値 ≥3、L 感度含む）。
4. **誤指定・スケール実験**: 変換方法（raw/log/zscore）を明記した count 属性比較の
   trials≥10 拡張（story diagnostics の追試）。
5. **held-out X**（属性欠損予測）: X 要素 mask の評価系を新設（transductive 明示）。
6. **PPC / calibration**: MovieLens Poisson の Pearson 残差・PIT、Cora の calibration。

いずれもブランチ `research/theory-audit-fixes`（仮）で実施し、EXPERIMENT_REGISTRY.md に
行追加、runinfo CSV を保存する。

---

## 承認依頼

P0-1〜P0-6 / P1-1〜P1-8 のうち、修正フェーズへ進める項目の承認をお願いしたい。
特に **P0-6（先生返答前の MATLAB 注記）** は残タスク「先生への返答を送る」より
先に行うべきである。P0-3（X 切片）は先生への確認質問 #2 の回答待ちを推奨する。
