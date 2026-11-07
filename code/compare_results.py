"""Compare a rerun's seed_results.csv with the reference copy, row by row.

Rows are matched on (world, seed, method). For every numeric column found in both
files (run time excluded), the script prints the largest absolute difference per
method and lists any row that differs by more than the tolerance. The exit code is
0 when every compared value matches.

Run from the repository root, for example:
    python code/compare_results.py results/relative_price/seed_results.csv results/relative_price_reproduced/seed_results.csv
    python code/compare_results.py results/benchmark/seed_results.csv results/benchmark_reproduced/seed_results.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd

KEYS = ['world', 'seed', 'method']
IGNORED = {'seconds'}  # wall-clock time is expected to differ

parser = argparse.ArgumentParser(description='Compare rerun results with reference results.')
parser.add_argument('reference', help='reference seed_results.csv')
parser.add_argument('rerun', help='seed_results.csv written by the rerun')
parser.add_argument('--tolerance', type=float, default=1e-9,
                    help='largest absolute difference treated as a match (default 1e-9)')
args = parser.parse_args()

reference = pd.read_csv(args.reference)
rerun = pd.read_csv(args.rerun)
merged = rerun.merge(reference, on=KEYS, how='left', suffixes=('_rerun', '_ref'),
                     validate='one_to_one', indicator=True)
unmatched = merged[merged['_merge'] == 'left_only']
merged = merged[merged['_merge'] == 'both']
columns = [c for c in rerun.columns
           if c not in KEYS and c not in IGNORED and c in reference.columns
           and pd.api.types.is_numeric_dtype(rerun[c]) and pd.api.types.is_numeric_dtype(reference[c])]

# Absolute difference per value; two missing values (e.g. no beta for a predictive model) count as equal.
diff = pd.DataFrame({c: (merged[f'{c}_rerun'] - merged[f'{c}_ref']).abs() for c in columns})
both_missing = pd.DataFrame({c: merged[f'{c}_rerun'].isna() & merged[f'{c}_ref'].isna() for c in columns})
diff = diff.mask(both_missing, 0.0).fillna(np.inf)

print(f'Compared {len(merged)} rerun rows with the reference on: {", ".join(columns)}')
print('\nLargest absolute difference per method:')
print(diff.groupby(merged['method']).max().to_string(float_format=lambda v: f'{v:.3g}'))

bad = diff.gt(args.tolerance).any(axis=1)
if len(unmatched):
    print(f'\n{len(unmatched)} rerun rows have no reference row:')
    print(unmatched[KEYS].to_string(index=False))
if bad.any():
    print(f'\n{int(bad.sum())} rows differ by more than {args.tolerance:g}:')
    shown = merged.loc[bad, KEYS + [f'{c}_{s}' for c in ('oracle_regret',) if c in columns for s in ('ref', 'rerun')]]
    print(shown.to_string(index=False))
if not bad.any() and not len(unmatched):
    print(f'\nAll {len(merged)} rows match the reference within {args.tolerance:g}.')
sys.exit(1 if bad.any() or len(unmatched) else 0)
