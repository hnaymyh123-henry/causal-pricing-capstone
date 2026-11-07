# Code

Run every script from the repository root. `experiment.py` contains the simulator,
the seven methods, the price choice and the scoring; every other script imports it
or reads saved results.

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `experiment.py` | Simulates the five worlds, fits the seven methods (original absolute-price input), chooses prices and scores Oracle regret. `--quick` runs a small demo. | — | `--output` folder, relative to `code/`: `seed_results.csv`, `summary.csv`, `heatmap.png` |
| `relative_price/run_experiment.py` | Main comparison: predictive models refitted with training-relative log price; slope estimators rerun and checked against the archived records | `results/benchmark/matched_seed_results.csv` | `results/relative_price_reproduced/` (default) |
| `relative_price/verify_results.py` | Re-scores saved actions and forecasts against simulator-known demand; checks the simulator SHA-256 | a relative-price results folder | console (plus `verification.json` for reruns) |
| `relative_price/analyze_results.py` | Per-world, elastic-world and group summaries of a complete run | `paired_results.csv` | `final_comparison.csv`, `elastic_world_aggregate.csv`, `group_comparison.json`, `RESULTS.md` |
| `relative_price/plot_figures.py` | Report Figures 2 and 3 | `seed_results.csv` | `figures/` |
| `relative_price/probe_linear.py`, `probe_transformer.py` | Seed-0 diagnostics of the price representation (not part of the main results) | `experiment.py` | `results/relative_price/diagnostics/` |
| `compare_results.py` | Row-by-row comparison of a rerun with the saved results | two `seed_results.csv` files | console |
| `build_matched_results.py` | Matched-seed (0–4) tables from the 30-seed benchmark records | `results/benchmark/seed_results.csv` | `results/benchmark/matched_*.csv` |
| `audit_statistics.py` | Integrity, coverage and uncertainty checks of the benchmark records | `results/benchmark/` | console |
| `plot_results.py` | Figures for the original-input benchmark | `results/benchmark/` | `results/benchmark/*.png` |
| `diagnose_fixed_policies.py` | Regret of three fixed price policies (report Table 9) | `experiment.py`, `results/benchmark/` | `results/diagnostics/fixed_policies.json` |
| `diagnose_price_anchor.py` | Seed-0 OLS check of an added training-only price-anchor feature | `experiment.py` | `results/diagnostics/price_anchor.json` |
| `diagnose_design.py` | 30-seed checks of the simulated price and demand margins, weak-instrument first-stage F, and Oracle price shares | `experiment.py` | console |
| `calibrate_demand_anchor.py` | Grid search for the demand-anchor mean and SD that match the target demand mean, median and coefficient of variation | `experiment.py` | console |
| `m5_stats.py`, `forecasting_utils.py` | Load the M5 files and build the CA_3/FOODS weekly panel | `data/m5/*.csv` | — |
| `calibrate_effect_sizes.py` | M5 price and demand scales behind the simulator parameters | `data/m5/*.csv` | console (JSON) |

Notes:

- `experiment.py` resolves `--output` relative to its own folder, so use
  `--output ../results/<name>` to write under `results/`. Its default (`results`)
  writes to `code/results/`.
- `diagnose_fixed_policies.py`, `diagnose_price_anchor.py` and the two probes
  overwrite their saved diagnostic files when rerun.
