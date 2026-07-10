# pair mask / strict held-out 実装報告（Phase 2）

作成日: 2026-07-08
ブランチ: `research/overdispersion-z-ablation`

## 1. 背景（confirmed）

既存 API は欠測ペアマスクを持たない（KI-012、
`run_fixed_real_movielens_heldout_count.py` L.5-31 に明文化）。
このため従来の MovieLens 評価は「全ペアで学習 → 評価だけ train/test に分割」
という **masked evaluation** であり、strict held-out ではなかった。
held-out ペアを Y=0 に置く代替案は、Poisson/Bernoulli では
「カウント 0 / 非リンクの観測」と区別できず学習を汚染する。

## 2. 実装（confirmed、本フェーズで作成）

`expfam/src/experimental/model_dual_expfam_masked.py` の
`DualExpFamLSMMasked`（`DualExpFamLSMFixed` のサブクラス。
**既存ファイルは一切変更していない**）。

観測ペア集合 O（対称 bool 行列 `train_mask`、対角強制 False）に対し:

| メソッド | マスクの入り方 |
|---|---|
| `_calc_gradient` Term3 | スコア残差 × mask 行（j∉O_i の寄与ゼロ） |
| `_calc_precision_matrix` Term3 | A''(η) × mask 行 |
| `calc_w0` / `calc_w` | 残差行列 × mask（Adam 勾配から test ペア除外） |
| `calc_sigma_y`（Gaussian Y） | 観測上三角ペアのみで MLE |
| `calc_log_likelihood_Y` | ln p × mask（Q 監視・BIC も観測ペアのみ） |

KI-012 が挙げた必要変更 4 箇所（calc_w0/calc_w/_calc_gradient/
_calc_precision_matrix）+ sigma_y + 尤度の計 6 箇所で閉じることを確認。
E-step 本体（`calc_eta_newton`）は `_calc_gradient`/`_calc_precision_matrix`
のみを呼ぶため（`reproduction/src/model.py` L.411-460）、変更不要。

## 3. 正しさの保証（confirmed、`test_experimental_models.py` 全 PASS）

1. **後方互換**: `train_mask=None` のとき `DualExpFamLSMFixed` と
   勾配・精度行列・`calc_w0`・Y 尤度が**数値的に一致**（3 family すべて）。
2. **リーク無し**: held-out ペアの Y 値を 999 に書き換えても、
   勾配・`calc_w0`・`calc_w`・Y 尤度が**bit 単位で不変**。
   → test ペア情報が学習に漏れないことの直接検証。
3. **分割の正当性**: train/test マスクが非対角要素の分割
   （overlap なし・union 完全・対称）。

## 4. 制限・注意（inference 含む）

- **MCAR 分割のみ**: `make_pair_split` は一様ランダム分割。
  観測バイアス（人気ペアほど観測されやすい等）を伴う欠測（MNAR）は
  尤度がこのままでは正しくない — 将来課題。
- **予測は plug-in**: held-out 対数尤度は最終点推定 (Ẑ, ŵ0, ŵ) による
  plug-in であり、事後予測分布の積分ではない（family 間比較は同条件なので公平）。
- **BIC の n**: 観測ペア数が減ることのペナルティ項への影響は未整理
  （KI-010/KI-011 と同じ論点系。本フェーズでは held-out 指標を主、BIC は補助）。
- **孤立ノード**: マスクにより実質的に Y 情報を失うノードが生じ得る
  （test_ratio=0.2、n=100 では各ノード平均 79 観測ペアが残るため実害なし）。

## 5. 評価結果（strict held-out が可能になったことの実証）

`tools/overdispersion/run_movielens_strict_heldout.py`（36 fits, 5.3 分）で
strict held-out 評価を実施済み。主要数値は
`reports/overdispersion/movielens_overdispersion_diagnostics_20260708.md` §5
参照。従来 masked evaluation の楽観（te_ll で +0.09〜+0.18/pair）を定量化した。

## 6. 次に必要な作業

- Cora held-out link prediction（`cora_heldout_link_prediction`, neg_ratio=5
  で学習は全エッジ使用）の strict 版への置き換え
- MNAR / 観測プロセスモデルの検討（推薦データでは本質的）
- `run_em_dual`（安定版）への mask 正式統合は、experimental 版の実績を
  積んでから修論後半で判断（既存 API 破壊を避けるため当面 experimental に隔離）
