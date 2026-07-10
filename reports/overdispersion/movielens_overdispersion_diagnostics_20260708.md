# MovieLens 過分散診断レポート（Phase 1 + strict held-out 検証）

作成日: 2026-07-08
ブランチ: `research/overdispersion-z-ablation`
スクリプト: `tools/overdispersion/diagnose_movielens_overdispersion.py`,
`tools/overdispersion/run_movielens_strict_heldout.py`
結果: `expfam/results/overdispersion/`
図: `figures/overdispersion/`

## 1. 中心的発見（要約）

**「MovieLens Y は var/mean ≈ 9.9 の過分散なので Poisson 仮定が怪しい」
（KI-012）は、周辺（marginal）統計と条件付き（conditional）統計の混同だった。**

- 周辺 var/mean = 9.89（confirmed、KI-012 の記載と一致）
- しかし潜在構造 μ̂_ij = exp(ŵ0 + ŵ ẑ_i^T ẑ_j) をフィットした後の
  **条件付き Pearson 過分散は k=3 で 1.14、k=5 で 0.76（in-sample）**、
  **strict held-out の test ペアでも k=3 で ≈1.34、k=5 で ≈0.86–0.96**
- plug-in PPC でも Poisson 複製が周辺 var/mean=9.79（観測 9.89、p=0.15）を再現

つまり周辺過分散の大部分は「μ_ij がペアごとに大きく異なる」という
**潜在的異質性**で説明され、これはまさに潜在構造モデルが表現する構造である。
Poisson 誤指定の被害は、このデータ（movie-projection 共評価カウント）では
**当初想定（KI-012）より小さい**。これは negative result ではなく、
「周辺診断だけで family を選ぶ/棄却すると誤る」という
分布族選択研究の中心的教訓を与える（修論で使える主張、§5）。

## 2. 周辺統計（confirmed、`movielens_overdispersion_diagnostics.csv`）

| Y 定義 | mean | var/mean | zero率 | max | 歪度 |
|---|---:|---:|---:|---:|---:|
| co-rating count（`movielens_Y_count.npy`, n=100, 4950 pairs） | 45.22 | 9.89 | 0.000 | 144 | 0.83 |
| co-like count（ml-100k.zip から再構築, rating≥4） | 14.75 | 8.85 | 0.007 | — | — |
| binary_t10 | density 0.99 | — | 0.009 | — | — |
| binary_t20 | density 0.91 | — | 0.089 | — | — |

- co-like の mean=14.75 は `movielens_colike_clean` の文書値 14.75 と一致
  （再構築手順の整合を confirmed）。co-like も周辺過分散は同程度（8.85）。
- binary_t10/t20 は density が 0.9 超で正例過多（link prediction 用途には
  t80 相当の高い閾値が必要 — 既存 `movielens_bernoulli_t80_pilot` と整合）。

## 3. 条件付き診断（confirmed）

Poisson モデル（fixed 系列 `DualExpFamLSMMasked`、full mask、X=Bernoulli genre）
をフィットした後の Pearson 残差過分散 (1/N)Σ(y−μ̂)²/μ̂:

| 条件 | Pearson 過分散 | NB r̂（moment） |
|---|---:|---:|
| in-sample, k=3 | 1.135 | 182 |
| in-sample, k=5 | 0.762 | 実質 ∞（過分散なし） |
| strict held-out test, k=3（6 fits 平均） | 1.34 | train r̂ ≈ 167–204 |
| strict held-out test, k=5（安定 5 fits） | ≈0.86–0.96 | 実質 ∞ |

- k=5 in-sample の 0.76 (<1) は **in-sample 過適合による過小分散**の兆候
  （inference）。strict held-out では 1 弱〜1.34 に戻る。
- 残差過分散は k に依存する: k を増やすほど潜在構造が周辺分散を吸収する。
  「family の妥当性」は「k の選択」と切り離せない（分布族選択と次元選択の
  交絡 — 修論の主題と直結）。

## 4. PPC（`movielens_ppc_summary.csv`、plug-in・k=5）

| 統計量 | 観測 | Poisson 複製平均 | p 値 (rep≥obs) |
|---|---:|---:|---:|
| var/mean | 9.89 | 9.79 | 0.15 |
| max | 144 | 143.6 | 0.43 |
| q99 | 102 | 104.2 | 0.99 |
| zero率 | 0.000 | 0.000 | 1.00 |

どの統計量でも Poisson 複製は棄却されない。
注: μ̂ plug-in（事後積分なし）の近似 PPC であり、in-sample μ̂ を使うため
保守的（モデルに有利）方向のバイアスがある（inference）。

## 5. strict held-out での Poisson vs NB（`movielens_strict_heldout_agg.csv`）

36 fits（k∈{3,5} × 3 splits × 2 seeds × 3 条件、test 20% ペア完全除外）:

| 条件 | k | test mean_ll | test RMSE | test Pearson | test 過分散 |
|---|---|---:|---:|---:|---:|
| poisson_strict | 3 | −3.459 | 7.63 | 0.934 | 1.34 |
| nb_strict | 3 | **−3.438** | 7.69 | 0.932 | 1.34 |
| poisson_full（リーク参照） | 3 | −3.372 | 7.09 | 0.943 | 1.17 |
| poisson_strict | 5 | −3.755※ | 14.58※ | 0.844※ | 1.75※ |
| nb_strict | 5 | **−3.338** | 6.79 | 0.948 | 1.04 |
| poisson_full（リーク参照） | 5 | −3.172 | 5.73 | 0.963 | 0.77 |

※ poisson_strict k=5 は 6 fits 中 1 fit（split=2, seed=0）が発散気味
（te_ll=−5.51, te_rmse=48.2）で平均が汚染されている。残り 5 fits は
te_ll ≈ −3.20〜−3.25, te_rmse ≈ 6.1〜6.5 で nb_strict と同水準（confirmed、
`movielens_strict_heldout_summary.csv` 参照）。

**読み取り:**
1. **Q2（NB は改善するか）**: te_ll は k=3 で +0.02/pair、k=5 では
   安定 fits 同士で同水準。**平均予測の改善はごく小さい**。ただし
   Poisson が不安定化した 1 条件で NB は安定（te_ll −3.57 vs −5.51）—
   NB の Fisher 重み μr/(μ+r) は大 μ で頭打ちになり Newton が安定するため
   （inference）。「性能向上」より「頑健性の保険」として効く。
2. **Q1（in-sample の楽観）**: full 学習は strict 比で te_ll +0.09〜+0.18/pair、
   te_rmse −0.5〜−0.9 の楽観。従来の masked evaluation
   （`movielens_heldout_count`, Pearson≈0.96）は**この楽観を含む**。
   strict でも Pearson ≈ 0.93–0.95 であり、結論の方向は変わらないが
   数値は下方修正が必要（confirmed）。
3. r̂（train 残差）は k=3 で 170–200、k=5 で実質 ∞ — 「必要な過分散対応の
   大きさ」も k に依存する。

## 6. 結論と修論で使える主張

- **使える主張 A（診断方法論）**: 周辺 var/mean は潜在構造モデルの family
  診断として不適切。条件付き Pearson 過分散・PPC・held-out 過分散を使うべき。
  MovieLens はその実例（周辺 9.89 → 条件付き ≈1）。
- **使える主張 B（strict held-out 基盤）**: pair mask により strict held-out
  が可能になり（リーク無しをテストで保証）、従来評価の楽観を定量化した。
- **使える主張 C（NB の位置づけ）**: 本データでは NB の予測改善は小さいが、
  推定安定性を改善する。過分散が強い場合の挙動は人工データ実験
  （`poisson_misspecification_*`）で別途定量化。
- **まだ言えないこと**: co-rating 投影以外の Y（user-node、二部グラフ、
  purely sparse counts で zero 過剰のあるデータ）での同様の結論。
  zero率 0.000 の本データは NB が効きにくい（zero-inflation がない）
  タイプである点に注意（inference）。
