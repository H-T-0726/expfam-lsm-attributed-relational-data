# 実行記録

- 作成日：2026-11-25。
- branch `experiment/k-selection-exp2-generalization-20261125`。worktree `D:/tento/kennkyu-kexp2`。基点 `origin/main` = `1b5030b2417a314927fb3a60d4d9016307f28d3b`（fetch 後に再確認）。
- **本実験：未実行。** protocol は未凍結（`preregistered_protocol.md` §0 の D1〜D6 が未決定）。protocol の SHA-256 もまだない。
- 環境：Python は `D:/tento/kennkyu/.venv`（Windows 11、12 論理 CPU）。

## 1. 実行したもの（すべて技術的な確認。K 選択の結果を含まない）

| # | 内容 | script | 出力 | MCEM | 基準の計算 |
|---|---|---|---|---|---|
| 1 | 生成器診断（4 系列 × 生成器 2〜4 種 × K_TRUE 5 × 5 draw = 150 draw） | `tools/research_audit/diag_k_exp2_generators.py` | `expfam/results/k_selection/k_exp2_generalization_20261125/generator_diagnostics/` | 0 | なし |
| 2 | fixed 系列での true_params の漏洩確認（4 family 組 × 4 variant、n=40, d=6, K=2, L=2, 3 反復） | `tools/research_audit/check_true_params_leak_fixed.py` | `.../true_params_leak_check/` | 16 fit（コード経路テスト） | なし（Q_strict の一致比較だけ） |
| 3 | 実行時間の計測（canonical matched Series O、K_TRUE=9 の 1 データ、候補 K = 1・10、lineage E と fixed、L=10、10 反復） | `tools/research_audit/timing_k_exp2_smoke.py` | `.../timing_smoke/timing_smoke.json` | 4 fit | **なし**（時間・有限性だけを記録） |

## 2. 結果（技術的なもの）

- 生成器：`mathematical_validity.md` を参照。canonical matched の Series B は 25/25 draw で生成器が停止した（Poisson-Y の分散発散）。
- true_params：4 variant すべてで Q_strict がビット単位で一致した（漏洩なし）。
- 実行時間（1 プロセス、BLAS のスレッド数は既定）：

  | 系列 | K=1 | K=10 |
  |---|---|---|
  | E | 93.2 s | 98.2 s |
  | fixed | 79.9 s | 89.8 s |

  4 fit とも有限で終了した。

## 3. 計算予算の見積もり（概算）

1 fit を約 95 s とする。8 worker で並列にしたとき、1 worker あたりの速度が同じと仮定した概算である（BLAS スレッドの競合で遅くなりうる）。

| 構成 | fit 数 | CPU 時間 | 8 並列の wall 時間 |
|---|---|---|---|
| primary（4 系列、R=10、T=2） | 4,000 | 約 106 h | 約 13〜16 h |
| ＋ PT 生成器の Series O（D5 b） | 5,000 | 約 132 h | 約 17〜20 h |
| ＋ 信号規則の感度分析（D3 c） | 9,000 | 約 238 h | 約 30〜36 h |
| C_Lap の評価 | 同じ fit 上（追加の MCEM なし） | fit あたり数秒〜数十秒（K=10 で nK = 1500 の Hessian） | — |

予算を減らす場合の影響：
- T=2 → 1：初期値の影響を測れなくなる（研究問いの 9 が答えられない）。fit 数は半分になる。
- R=10 → 5：K_TRUE ごとの正解率の精度が下がる（10 データで 1 件 = 10%）。
- L や反復数を減らす：元論文の条件（10・10）との対応が崩れる。

## 4. 本実験を始めるときの手順（判断の後）

1. D1〜D6 を決める。
2. runner を実装して test する。
   - 生成器の unit test
   - true_params を使わない初期化
   - 候補 K の間でデータが一致すること
   - K_TRUE = 1・9、候補 K = 1・10 の境界
   - Q の再計算
   - 保存と再読込
   - seed の再現性
   - fail_fast
3. `protocol.json` を凍結して commit し、SHA-256 をここに記録する。
4. 実行し、事前に決めた解析を行う。
