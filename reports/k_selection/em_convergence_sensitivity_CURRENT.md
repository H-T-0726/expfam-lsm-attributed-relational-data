# EM 収束感度実験 — 現行の科学的設計（2026-09-06 時点）

この文書は **現在有効な設計だけ**を1か所にまとめたものである。
先生への報告・今後の研究判断・実験実行はこの文書を見れば足りる。

- 機械可読な仕様の正本: `reports/k_selection/em_convergence_sensitivity_20260905_protocol.DRAFT.json`
- 設計の検討経緯（歴史記録、現行値ではない）: `em_convergence_sensitivity_design_20260905.md`

---

## 1. 研究目的

同じ `L=5` の stochastic MCEM を **8 反復から 40 反復まで継続**したとき、
**選択される K がどう変化するか**を見る診断実験。

これは production 基盤を作るための実験ではなく、
「潜在次元 K の選択」「MCEM の反復数」「次に何を研究すべきか」を
先生と相談するための材料を作るための実験である。

## 2. 背景

先行する clean true-K 実験（`clean_true_k_asymptotics_20260904`）で、
`K_TRUE=5` に対して under-selection が観測された。

| n | exact 選択（8 replicates 中） |
|---|---|
| 50 | 2 / 8 |
| 75 | 0 / 8 |
| 100 | 4 / 8 |
| 150 | 8 / 8 |

この実験は `num_iter=8` 固定で走っている。
高い K ほど推定すべきパラメータが多いため、
**8 反復では高い K が十分推定できておらず、それが under-selection に寄与した可能性**がある。
本実験はその可能性を確認する。

## 3. 実験条件

| 項目 | 値 |
|---|---|
| K_TRUE | 5 |
| n | 50, 75, 100, 150 |
| replicates | 8 |
| candidate K | 1..7 |
| starts | 2 |
| L | 5 |
| max_iter | 40 |
| checkpoints | 8, 12, 16, 24, 32, 40 |
| family_x | Poisson |
| family_y | Bernoulli |
| d | 15 |
| test_ratio | 0.20 |

fit 数 448、checkpoint 行 2,688、iteration 行 17,920。
runtime 見積もりは約 6.1 時間（archive からの線形外挿であり、**上界ではない**）。

## 4. 固定するもの

旧 8 反復実験と**同一**にする:

- data（generator `canonical-clean-v1`、data_seed）
- split / mask（split_seed、archive の mask hash と一致すること）
- initialization（model_seed）
- family（X = Poisson、Y = Bernoulli）
- 数値実装系列（`numerics_mode = consistent`）

**変えるのは反復数だけ**である。

## 5. Primary

**checkpoint 8 と checkpoint 40 の selected K。**

各 (n, replicate) について、candidate K ごとに 2 starts の S1 を単純平均し、
最大の K を選ぶ（tie は tolerance 1e-12 で小さい K に倒す）。
checkpoint 8 と 40 の間で replicate 内 paired に比較する。

## 6. Secondary

**candidate K ごとの S1(40) − S1(8)。**

```
delta(n, r, K) = 0.5 * Σ_{start∈{1,2}} [ S1(n,r,K,start,40) − S1(n,r,K,start,8) ]
```

replicate レベル 224 行、要約 28 行。**記述のみ**で、検定は行わない
（starts は replicate ではない）。

副次的に S2 / S3 版、start 間の不一致、勝ち幅も見る。

## 7. 最低限の再現確認（replication gate）

**本実験で最も重要な検査。**

gate cell（`n = min(n_grid) = 50`、replicate 1、7 candidate K × 2 starts = 14 fit）を
iteration 8 まで走らせ、archive `clean_true_k_asymptotics_20260904` の対応する fit と
**厳密一致**することを確認する。比較対象は
`data_seed` / `split_seed` / `model_seed` / `num_params` /
`heldout_mean_log_score` / `q_strict` / `s2_q_based` / `s3_plugin_conditional` と
train/test mask hash。**tolerance は定義しない。**

一致しなければ、40 反復の run は旧 8 反復 run の継続ではないので、比較そのものが無効になる。

production では同じ比較を **全 448 fit** について checkpoint callback の中で行う。
prefix が archive と違う fit は iteration 9 に進まない。

## 8. 解釈上の注意（結果を見る前に確定）

- **pure optimization effect の証明ではない。** 追加反復では新しい Monte Carlo draw も
  増えるため、「最適化が進んだ効果」と「MC 変動が増えた効果」はこの設計では分離できない。
- **convergence の証明ではない。** 反復ごとの変化量は記述的な trace であって、
  収束判定にも停止規則にも選択基準にも使っていない。
- **consistency の主張ではない。** 有限標本での挙動を見るだけである。
- 「8 反復が under-selection の原因だった」とは書かない。
- 「選択基準に問題はなかった」とは書かない。

## 9. 実行前チェック

1. protocol 一致 — runner contract の全フィールドが protocol 本体と一致すること
   （EM 開始前に自動で確認され、不一致なら exit）
2. **replication gate が PASS**
3. NaN / retry が 0 件
4. 出力先 `expfam/results/k_selection/em_convergence_sensitivity_20260905` が存在しないこと、
   staging `...20260905.partial` の残骸が無いこと

実行:

```
python tools/research_audit/run_em_convergence_sensitivity.py --plan
python tools/research_audit/run_em_convergence_sensitivity.py --replication-gate --allow-em
python tools/research_audit/run_em_convergence_sensitivity.py --full --allow-em --confirm-em-convergence-sweep
python tools/research_audit/audit_em_convergence_sensitivity.py
```

途中で落ちた場合は `.partial` ディレクトリだけが残り、最終成果物は作られない。
残骸は人間が確認して削除する。**部分的な fit は分析対象にしない。**

## 10. 結果が出た後、先生に確認したい論点

1. **MCEM の反復数をどこまで議論すべきか。**
   反復数を固定した比較でよいのか、収束を定義して止めるべきなのか。
2. **K 選択基準そのものを次に検討すべきか。**
   現行の primary は heldout S1。基準を変えるべきか、複数基準の一致/不一致を報告すべきか。
3. **理論的一致性まで必要か。**
   選択の consistency を理論的に議論する必要があるのか、実験的整理で足りるのか。
4. **有限標本の挙動整理で修論として十分か。**
   n を増やすと exact 選択が増える、という有限標本の記述で研究の主張として成立するか。
