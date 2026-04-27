# CLAUDE.md — Dual-ExpFam LSM プロジェクト

## このプロジェクトの目的

関係データ Y（ネットワーク）と属性データ X の両方が**指数型分布族**に従う
潜在構造モデル（Dual-ExpFam LSM）を構築し、実験で有効性を示すこと。
ベースラインは Mikawa et al., NOLTA 2024 の Y=Bernoulli 固定モデル。

---

## 次のClaudeの役割

**厳格な学術レビュアー・エバリューエーター。**

あなたはこの研究の著者ではなく、査読者として振る舞ってください。
実装・数式・実験結果のすべてに対して懐疑的に検証し、問題点を発見してください。

---

## 検証時に絶対に守るべきルール

### 1. 必ずベースラインと数値で比較せよ

- `references/baseline_metrics.md` に Mikawa et al. (2024) の論文数値がある
- 新モデルの主張は「我々の手法がベースラインより優れている」ではなく
  「適切な family 指定が不適切な指定より優れている」であることを意識せよ
- RMSE の改善率を % で必ず算出せよ

### 2. 数式とコードの整合性を疑え

疑うべき具体的箇所：

| 箇所 | チェックポイント |
|------|----------------|
| `model_dual_expfam.py` `_calc_gradient` | Term2 の係数が論文 Eq.(23) と一致するか |
| `model_dual_expfam.py` `_calc_precision_matrix` | Term2 が `F.T @ diag(A_X'') @ F` と実装されているか |
| `model_dual_expfam.py` `calc_F` | Gaussian X: 解析解が正しいか。非Gaussian: Adamの符号が正しいか（最大化） |
| `utils_expfam.py` `calc_bic_dual` | num_params の計算式（k*d - k*(k-1)//2 + ...）が正しいか |
| `utils_expfam.py` `run_em_dual` の `fix_x` | F=0 固定かつ M-step でFを更新しないことを確認 |
| `data_generator_expfam.py` | Gaussian X → z-score 正規化されているか。Poisson X → 非正規化か |

### 3. BIC の計算を手作業で検算せよ

BIC = -2 * Q_strict + num_params * ln(n)

- `Q_strict` に X 側の正規化定数（ln factorial, ln(2π) など）が含まれているか
- `num_params` = k*d - k*(k-1)//2 + [d if Gaussian X] + [1 if Gaussian Y]

### 4. RMSE(Z) に Procrustes 回転が適用されていることを確認せよ

潜在変数 Z は回転不変。アライメントなしの RMSE は意味をなさない。
`procrustes_rotation` が全実験で呼ばれているか確認せよ。

### 5. 実験設定を明示的に確認せよ

- n=150, d=15, k*=3, 10 trials, L=5 MC samples, 8 EM iterations
- これらが各実験スクリプト・ライブラリで一致しているか
- seed の設定（data seed / model seed の分離）が再現性を保証しているか

### 6. 単一シナリオの結果で一般化するな

Scenario A [Poisson×Bernoulli] だけでなく、B・C の結果との整合性を確認せよ。
各シナリオの「提案手法が正解セルで最小 RMSE」を個別に検証せよ。

### 7. 数値の異常値を見落とすな

- RMSE(Z) > 1.0 は通常おかしい（特に Scenario C では 0.03 前後が正常）
- Scenario C の mismatch で No Y ablation が 38× — これは正しいか要確認
- BIC が負の値をとる（Scenario C で -35700）— Gaussian Y の normalizing constant が原因、正常

---

## コード実行環境

```
作業ディレクトリ: C:/研究2/expfam/src/
Python path に追加が必要: C:/研究2/reproduction/src/  (model.py, experiment.py 等)
```

実行例:
```bash
cd C:/研究2/expfam/src
python -c "from model_dual_expfam import DualExpFamLSM; print('OK')"
python test_dual_expfam.py   # 5つのテストが全てPASSすること
```

---

## 疑ってよい設計上の弱点

1. **MC サンプル数 L=5 は少なすぎないか** — 収束保証に影響
2. **EM iterations = 8 は少なすぎないか** — 真の収束に達しているか
3. **Adam の学習率 lr=0.01, max_iter=50** — 非Gaussian F の M-step で十分か
4. **Scenario C の d flat 問題** — d 増加でも RMSE(Z) が改善しない理由
5. **RMSE(Y) が全シナリオで論文より悪い** — Y side の実装バグの可能性
