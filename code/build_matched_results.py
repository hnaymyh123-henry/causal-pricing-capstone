"""Build the common-seed (seeds 0-4) benchmark files from the formal per-seed results.

Run with Python containing pandas: python code/build_matched_results.py
Reads results/benchmark/seed_results.csv and writes matched_seed_results.csv,
matched_summary.csv and matched_wmape_summary.csv to the same folder. No models are trained.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/benchmark'
WORLDS = ['baseline', 'higher_elasticity', 'weak_instrument', 'hidden_confounding', 'exclusion_violated']
METHODS = ['IV', 'DML', 'Naive', 'DirectLiNGAM', 'HGB', 'TransformerSmall', 'TransformerLarge']

raw = pd.read_csv(RESULTS / 'seed_results.csv', float_precision='round_trip')
assert len(raw) == 800 and not raw.duplicated(['world', 'seed', 'method']).any()
matched = raw[raw.seed.between(0, 4)].copy()
assert len(matched) == 175 and (matched.groupby(['world', 'method']).size() == 5).all()
matched.to_csv(RESULTS / 'matched_seed_results.csv', index=False)

# Mean regret (%) per world and method over the five shared seeds.
grid = matched.pivot(index=['world', 'seed'], columns='method', values='oracle_regret') * 100
grid = grid.reindex(pd.MultiIndex.from_product([WORLDS, range(5)], names=['world', 'seed']))
grid.groupby(level='world').mean().reindex(WORLDS)[METHODS].to_csv(RESULTS / 'matched_summary.csv')
wmape = matched.groupby(['world', 'method']).observed_wmape_pct.agg(['mean', 'count'])
wmape.to_csv(RESULTS / 'matched_wmape_summary.csv')
print(f'Wrote {len(matched)} matched records and two common-seed summaries to {RESULTS}')
