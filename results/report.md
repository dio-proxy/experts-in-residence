## Trained models, no inference-time rerouting

| run | perplexity | vs flat | vs own λ=0 | miss @4 | miss @8 | miss @16 | expert switch | cluster switch | coverage @8 |
|---|---|---|---|---|---|---|---|---|---|
| dense | 65.85 | +22.9% | +0.0% | – | – | – | – | – | – |
| flat | 53.59 | +0.0% | +0.0% | 0.856 | 0.739 | 0.538 | 0.952 | – | 0.436 |
| flat-l0.03 | 54.36 | +1.4% | +1.4% | 0.573 | 0.379 | 0.152 | 0.809 | – | 0.716 |
| flat-l0.1 | 67.17 | +25.4% | +25.4% | 0.108 | 0.051 | 0.041 | 0.428 | – | 0.976 |
| flat-l0.3 | 71.83 | +34.0% | +34.0% | 0.036 | 0.023 | 0.005 | 0.198 | – | 0.998 |
| flat-seed1 | 53.65 | +0.1% | +0.1% | 0.853 | 0.734 | 0.534 | 0.953 | – | 0.438 |
| tree | 58.26 | +8.7% | +0.0% | 0.886 | 0.780 | 0.591 | 0.962 | 0.838 | 0.404 |
| tree-l0.03 | 60.04 | +12.0% | +3.1% | 0.666 | 0.434 | 0.191 | 0.893 | 0.400 | 0.678 |
| tree-l0.1 | 61.03 | +13.9% | +4.7% | 0.460 | 0.092 | 0.060 | 0.822 | 0.055 | 0.835 |
| tree-l0.3 | 67.63 | +26.2% | +16.1% | 0.430 | 0.061 | 0.051 | 0.807 | 0.030 | 0.868 |
| tree-slow | 62.20 | +16.1% | +6.8% | 0.441 | 0.064 | 0.057 | 0.821 | 0.019 | 0.857 |

Seed-to-seed perplexity gap (flat, λ=0): 0.13%

## Decision (cache of 8 experts per layer; selected on even streams, reported on odd)

Gates of 1%, 2% and 5% are pre-registered. Wider gates were added after seeing the results and are exploratory.

| gate | family | selected configuration | perplexity vs flat | miss rate | tree / flat |
|---|---|---|---|---|---|
| ≤1% | flat | flat λ=0 | +0.00% | 0.7426 | – |
| ≤1% | tree | none passes | – | – | – |
| ≤2% | flat | flat λ=0.03 | +1.75% | 0.3849 | – |
| ≤2% | tree | none passes | – | – | – |
| ≤5% | flat | flat λ=0.03 | +1.75% | 0.3849 | – |
| ≤5% | tree | none passes | – | – | – |
| ≤10% (exploratory) | flat | flat λ=0.03, Δ=1 | +10.89% | 0.1185 | |
| ≤10% (exploratory) | tree | tree λ=0 | +8.23% | 0.7825 | 6.60 |
| ≤15% (exploratory) | flat | flat λ=0.03, Δ=1 | +10.89% | 0.1185 | |
| ≤15% (exploratory) | tree | tree λ=0.1, Δ=1 | +16.53% | 0.0625 | 0.53 |
| ≤20% (exploratory) | flat | flat λ=0.03, Δ=1 | +10.89% | 0.1185 | |
| ≤20% (exploratory) | tree | tree λ=0.1, Δ=1, m=2 | +17.23% | 0.0503 | 0.42 |
| ≤25% (exploratory) | flat | flat λ=0.1 | +26.03% | 0.0518 | |
| ≤25% (exploratory) | tree | tree λ=0.1, Δ=2, m=2 | +24.70% | 0.0349 | 0.67 |
| ≤30% (exploratory) | flat | flat λ=0.1, Δ=2 | +30.67% | 0.0332 | |
| ≤30% (exploratory) | tree | tree λ=0.1, Δ=2, m=2 | +24.70% | 0.0349 | 1.05 |

**H1 at the pre-registered 1% gate: not supported.** No tree configuration is within the gate.

## Follow-up 1 (exploratory): sticky post-hoc expert groups on the flat models

Group restriction alone (no hysteresis, no cache bonus) against the flat model it is applied to:

| flat model | perplexity flat | grouped | change | miss @8 flat | grouped |
|---|---|---|---|---|---|
| λ=0 | 53.59 | 57.74 | +7.7% | 0.739 | 0.709 |
| λ=0.03 | 54.36 | 58.29 | +7.2% | 0.379 | 0.348 |
| λ=0.1 | 67.17 | 68.27 | +1.6% | 0.051 | 0.052 |
| λ=0.3 | 71.83 | 72.05 | +0.3% | 0.023 | 0.021 |

| gate | family | selected configuration | perplexity vs flat | miss rate | grouped / flat |
|---|---|---|---|---|---|
| ≤1% | flat | flat λ=0 | +0.00% | 0.7426 | – |
| ≤1% | grouped | none passes | – | – | – |
| ≤2% | flat | flat λ=0.03 | +1.75% | 0.3849 | – |
| ≤2% | grouped | none passes | – | – | – |
| ≤5% | flat | flat λ=0.03 | +1.75% | 0.3849 | – |
| ≤5% | grouped | none passes | – | – | – |
| ≤10% | flat | flat λ=0.03, Δ=1 | +10.89% | 0.1185 | |
| ≤10% | grouped | grouped λ=0.03 | +9.25% | 0.3518 | 2.97 |
| ≤15% | flat | flat λ=0.03, Δ=1 | +10.89% | 0.1185 | |
| ≤15% | grouped | grouped λ=0.03, Δ=1 | +12.73% | 0.2019 | 1.70 |
| ≤20% | flat | flat λ=0.03, Δ=1 | +10.89% | 0.1185 | |
| ≤20% | grouped | grouped λ=0.03, Δ=1 | +12.73% | 0.2019 | 1.70 |

**Follow-up 1 at the 1% gate: not supported.** No grouped configuration is within the gate.
