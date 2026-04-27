# Comparison Summary: Baseline (Mikawa 2024) vs Dual-ExpFam LSM

Generated: 2026-04-24  |  Elapsed: 37.4 min

Settings: N_TRIALS=5, L=5, NUM_ITER=8, n=150, d=15, k*=3

## 1. Audit Summary

| Item | Baseline | Dual-ExpFam |
|------|----------|------------|
| X distribution | Gaussian only | gaussian/bernoulli/poisson |
| Y distribution | Bernoulli only | gaussian/bernoulli/poisson |
| L (MC samples) | 10 (original) → **5 (matched)** | 5 |
| EM iterations | 10–30 (original) → **8 (matched)** | 8 |
| Trials | 3 (original) → **5 (matched)** | 10 |
| BIC formula | −2logL + ((k+1)d − k(k−1)/2) ln n | Same (Gauss/Bern) |

## 2. Comparability Classification

| Scenario | Baseline Y-valid? | X-valid? | Type |
|----------|-------------------|---------|------|
| Control: Gauss-X × Bern-Y | ✅ Yes | ✅ Yes | **Main comparison** |
| Scenario A: Pois-X × Bern-Y | ✅ Yes | ❌ Misspecified | Auxiliary |
| Scenario B: Gauss-X × Pois-Y | ❌ Non-binary Y | ✅ Yes | **Not applicable** |
| Scenario C: Bern-X × Gauss-Y | ❌ Continuous Y | ❌ | **Not applicable** |

## 3. Main Comparison (Control: Gauss-X × Bern-Y, k=k*=3)

**Condition**: same data, same L=5, same num_iter=8, same seeds.
Both models are correctly specified for this scenario.

| Metric | Baseline | Dual-ExpFam (Gauss,Bern) | Δ |
|--------|----------|--------------------------|---|
| rmse_Z | 0.1793±0.0144 | 0.1799±0.0155 | +0.0005 |
| rmse_F | 0.0397±0.0114 | 0.0351±0.0062 | -0.0046 |
| rmse_Y | 0.0881±0.0082 | 0.0877±0.0075 | -0.0004 |
| rmse_X | 0.3069±0.0056 | 0.3074±0.0049 | +0.0006 |
| w0_err | 0.0599±0.0162 | 0.0567±0.0151 | -0.0032 |
| w_err | 0.0828±0.0397 | 0.0794±0.0385 | -0.0034 |

**BIC-based k* identification (Control):**

| Model | BIC-optimal k | Correct? |
|-------|--------------|---------|
| baseline | k=3 | ✅ |
| dual_expfam | k=3 | ✅ |

## 4. Auxiliary Comparison (Scenario A: Pois-X × Bern-Y, k=k*=3)

**⚠️ Caution**: Baseline X is misspecified here (Gaussian assumption on Poisson data).
This result shows X-mismatch robustness, NOT fundamental model superiority.

| Metric | Baseline (X misspecified) | Dual-ExpFam (correct) | Δ |
|--------|--------------------------|----------------------|---|
| rmse_Z | 0.6818±0.0710 | 0.2798±0.0158 | -0.4020 |
| rmse_F | 1.2904±0.2025 | 0.0695±0.0066 | -1.2209 |
| rmse_Y | 0.2066±0.0052 | 0.1250±0.0081 | -0.0816 |
| rmse_X | 1.6967±0.1118 | 1.2100±0.0208 | -0.4867 |
| w0_err | 0.1125±0.0626 | 0.0419±0.0087 | -0.0705 |
| w_err | 0.4498±0.0816 | 0.1481±0.0292 | -0.3017 |

## 5. Not Applicable Cases

The following scenarios cannot be run with Baseline:

- **B_Gauss_Pois**: Y=Poisson is non-binary; Baseline requires Y∈{0,1}
- **C_Bern_Gauss**: Y=Gaussian is continuous; Baseline requires Y∈{0,1}

## 6. Conclusion

### What can be claimed from this experiment

**Control scenario (fair comparison):**
- Both models share the same distributional assumption (Gauss-X × Bern-Y).
- Any performance difference reflects algorithmic/implementation differences,
  NOT distributional advantage.
- DualExpFamLSM(gaussian, bernoulli) should be numerically equivalent to Baseline
  as both implement the same generative model; differences, if any, come from
  minor implementation choices (BIC formula, sigma update, scaling).

**Scenario A (Auxiliary — X misspecified for Baseline):**
- When true X is Poisson, the Baseline (Gaussian X assumption) is misspecified.
- Dual-ExpFam with correct family_x='poisson' is expected to outperform.
- This advantage is **distributional** (correct vs wrong X model), not algorithmic.

**Cannot claim from this experiment:**
- Scenarios B and C cannot be evaluated for Baseline due to non-binary Y.
- Dual-ExpFam's advantage in Scenarios B and C cannot be compared against Baseline.
- Any advantage in these scenarios must be attributed to family generality alone.

### Separation of advantages

| Advantage type | Evidence | Scenario |
|---------------|---------|---------|
| Algorithmic/estimation strength | Main comparison (Control) | Gauss-X × Bern-Y |
| Distributional (X-mismatch) | Auxiliary comparison | Pois-X × Bern-Y |
| Distributional (Y-mismatch) | Not comparable (Y non-binary) | B, C |

## 7. Unresolved Issues

- **Baseline Y-side**: Scenarios B (Pois-Y) and C (Gauss-Y) cannot be run
  with Baseline without Y transformation. A binarisation (y>0 → 1) would
  be possible but is not valid for main results.
- **Trial count mismatch**: Baseline original used 3 trials, Dual-ExpFam used 10.
  This comparison uses 5 (compromise). Results may have higher variance than
  the original paper's 3-trial setup.
- **EM iteration budget**: Original Baseline used 10–30 iterations; here 8 is used
  for fairness. Baseline may not fully converge in 8 iterations.
