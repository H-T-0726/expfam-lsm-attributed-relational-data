# Exp 5 Report: Ablation Study (X+Y vs X-only)

n=150, d=15, k=3, L=5, iter=8, trials=10

**Ablation**: w=0 fixed → Z learned from X only (no Y relational signal).
**Proposed**: w freely estimated → Z learned jointly from X and Y.

## RMSE(Z): Proposed vs Ablation

| Family | Proposed (mean±std) | Ablation w=0 (mean±std) | Gain (%) |
|--------|---------------------|------------------------|----------|
| Bernoulli | **0.1764**±0.0079 | 0.2485±0.0309 | **29.0%** |
| Poisson | **0.1875**±0.0469 | 0.2485±0.0309 | **24.5%** |
| Gaussian | **0.0286**±0.0009 | 0.2485±0.0309 | **88.5%** |

## RMSE(F): Proposed vs Ablation

| Family | Proposed (mean±std) | Ablation w=0 (mean±std) | Gain (%) |
|--------|---------------------|------------------------|----------|
| Bernoulli | **0.0320**±0.0085 | 0.0727±0.0363 | **56.0%** |
| Poisson | **0.0683**±0.0611 | 0.0727±0.0363 | **6.2%** |
| Gaussian | **0.0295**±0.0080 | 0.0727±0.0363 | **59.5%** |

## RMSE(Y): Proposed vs Ablation

| Family | Proposed (mean±std) | Ablation w=0 (mean±std) | Gain (%) |
|--------|---------------------|------------------------|----------|
| Bernoulli | **0.0860**±0.0033 | 0.3145±0.0044 | **72.7%** |
| Poisson | **0.7294**±0.2623 | 3.2413±1.3059 | **77.5%** |
| Gaussian | **0.0348**±0.0012 | 0.8608±0.0070 | **96.0%** |

## RMSE(X): Proposed vs Ablation

| Family | Proposed (mean±std) | Ablation w=0 (mean±std) | Gain (%) |
|--------|---------------------|------------------------|----------|
| Bernoulli | **0.3085**±0.0095 | 0.3125±0.0089 | **1.3%** |
| Poisson | **0.3090**±0.0094 | 0.3125±0.0089 | **1.1%** |
| Gaussian | **0.3124**±0.0092 | 0.3125±0.0089 | **0.0%** |

実行時間: 967.6s
