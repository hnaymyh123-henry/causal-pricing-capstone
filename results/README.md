# Results

Saved outputs of the reported runs. The rerun commands in the main README write to
separate `*_reproduced/` folders, so these files stay as the reference.

## relative_price/ — main comparison (report Section 6)

175 records: 7 methods × 5 worlds × 5 matched seeds. The three predictive models use
training-relative log price; the four slope estimators are rerun on unchanged data.

| File | Content |
|---|---|
| `seed_results.csv` | One row per world, seed and method: Oracle regret, share of +15% price choices, estimated price effect, fit time |
| `paired_results.csv`, `summary.csv` | Each record paired with the original-input benchmark record of the same world, seed and method |
| `final_comparison.csv`, `elastic_world_aggregate.csv`, `group_comparison.json`, `RESULTS.md` | Summaries written by `analyze_results.py` |
| `figures/` | Report Figures 2 and 3, written by `plot_figures.py` |
| `predictions/` | 175 saved action vectors; the 75 predictive records also hold the seven-price log-demand forecasts |
| `training_means/` | Training-period mean log price of every product, per world and seed |
| `diagnostics/` | Seed-0 probes that preceded the relative-price run (`linear_probe.json`, `transformer_probe_results.json`, `diagnosis_summary.json`) |
| `manifest.json` | Run record: scope, settings and SHA-256 of the protected inputs. Keys use the original project layout (`Causal_Pricing_Final/…`); the report documents listed there are not part of this repository |
| `verification.json` | Acceptance check of the run (re-scoring of all saved decisions) |

## benchmark/ — original-input benchmark (report Tables 7 and 10, Section 5.2)

800 records from `experiment.py` in five worlds: IV, DML, naive OLS, DirectLiNGAM and
HGB for seeds 0–29, and both Transformers for seeds 0–4.

| File | Content |
|---|---|
| `seed_results.csv`, `summary.csv`, `heatmap.png` | Written by `experiment.py` |
| `matched_seed_results.csv`, `matched_summary.csv`, `matched_wmape_summary.csv` | Seeds 0–4 only, written by `build_matched_results.py` |
| `report_heatmap.png`, `report_predictive_comparison.png` | Original-input figures from `plot_results.py`; the final report uses the relative-price figures instead |

## diagnostics/

| File | Content |
|---|---|
| `fixed_policies.json` | Regret of three fixed price policies (report Table 9) |
| `price_anchor.json` | Seed-0 OLS check of an added price-anchor feature |

## Notes

- Changes made before publication, so that reruns in the pinned environment
  reproduce every saved file byte for byte:
  - `relative_price/manifest.json`: absolute local paths in the keys of
    `protected_hashes` were replaced with project-relative paths; hashes unchanged.
  - `diagnostics/fixed_policies.json`: regenerated with the pinned packages. The
    archived copy had been written with a newer NumPy and differed only by
    floating-point rounding (at most 7 × 10⁻¹⁵ relative); all booleans identical.
  - `benchmark/heatmap.png`, `report_heatmap.png`, `report_predictive_comparison.png`:
    regenerated; pixel-identical to the archived images (only the embedded Matplotlib
    version changed).
  - `diagnostics/price_anchor.json`: regenerated with the published script, which
    records repository-relative paths; all numbers identical.
  - `relative_price/figures/`: new, written by `plot_figures.py`.
- Each run record was written when the run finished. Statements in them such as
  "formal report unchanged" describe that moment.
