# Relative-price paired experiment

Five environments, five matched seeds, 500 products, 104 training weeks and 26 test weeks. Original formal results and report are unchanged.

Only the predictive price feature changes: log price minus the product mean log price calculated on the training period. Demand targets, other inputs, training budgets, candidate prices and profit scoring remain unchanged. Four slope estimators are rerun on unchanged data and checked against the original records.

| Environment | Model | Original regret % | Relative-price regret % | Improvement pp |
|---|---|---:|---:|---:|
| higher_elasticity | HGB | 7.057 | 1.764 | 5.294 |
| higher_elasticity | TransformerSmall | 7.444 | 0.640 | 6.803 |
| higher_elasticity | TransformerLarge | 7.444 | 0.506 | 6.937 |
| hidden_confounding | HGB | 11.073 | 7.323 | 3.750 |
| hidden_confounding | TransformerSmall | 12.637 | 6.353 | 6.284 |
| hidden_confounding | TransformerLarge | 12.688 | 6.379 | 6.309 |
| weak_instrument | HGB | 7.016 | 1.859 | 5.157 |
| weak_instrument | TransformerSmall | 7.462 | 0.704 | 6.758 |
| weak_instrument | TransformerLarge | 7.462 | 0.492 | 6.970 |
| exclusion_violated | HGB | 6.966 | 1.948 | 5.018 |
| exclusion_violated | TransformerSmall | 7.455 | 0.663 | 6.792 |
| exclusion_violated | TransformerLarge | 7.455 | 0.510 | 6.946 |
| baseline | HGB | 0.059 | 0.417 | -0.359 |
| baseline | TransformerSmall | 0.000 | 0.000 | 0.000 |
| baseline | TransformerLarge | 0.000 | 0.000 | 0.000 |

Positive improvement means lower regret. Overall elastic-world intervals average worlds within each seed before computing uncertainty; there are five independent seed blocks, not twenty independent observations. This is a representation sensitivity experiment, not a fully controlled architecture comparison.