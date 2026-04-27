# Exp 4 Report: Distribution Mismatch 3x3

n=150, d=15, k=3, L=5, iter=8, trials=10

**KEY FINDING**: Diagonal (correct family) yields drastically lower RMSE than off-diagonal (mismatched family). This proves why ExpFam generalisation is necessary.

## RMSE(Z) mean ± std (rows=true data, cols=model)

| True \ Model | Bernoulli | Poisson | Gaussian |
|---|---|---|---|
| **Bernoulli** | **0.1784±0.0095** | 0.2147±0.0188 | 0.1927±0.0133 |
| **Poisson** | 1.1968±0.1482 | **0.1976±0.0604** | 0.5851±0.2861 |
| **Gaussian** | 0.5422±0.1135 | 0.2765±0.0123 | **0.0289±0.0014** |

## RMSE(F) mean ± std (rows=true data, cols=model)

| True \ Model | Bernoulli | Poisson | Gaussian |
|---|---|---|---|
| **Bernoulli** | **0.0288±0.0028** | 0.0375±0.0077 | 0.0295±0.0047 |
| **Poisson** | 3789512.1517±8674818.1780 | **0.0704±0.0647** | 0.2459±0.2219 |
| **Gaussian** | 3172.2037±10031.1494 | 0.0393±0.0056 | **0.0267±0.0026** |

## RMSE(Y) mean ± std (rows=true data, cols=model)

| True \ Model | Bernoulli | Poisson | Gaussian |
|---|---|---|---|
| **Bernoulli** | **0.0866±0.0033** | 647.1999±139.3760 | 2.6884±0.0097 |
| **Poisson** | 0.5314±0.0013 | **0.8106±0.4171** | 2.5293±0.9483 |
| **Gaussian** | 0.4742±0.0042 | 2.2919±0.9124 | **0.0353±0.0017** |

## Mismatch Penalty (RMSE(Z) ratio: mismatch / correct)

| True data | Correct RMSE(Z) | Best mismatch RMSE(Z) | Ratio |
|---|---|---|---|
| Bernoulli | 0.1784 | 0.1927 | **1.1x** |
| Poisson | 0.1976 | 0.5851 | **3.0x** |
| Gaussian | 0.0289 | 0.2765 | **9.6x** |

実行時間: 1492.8s
