"""Read-only, single-seed OLS feature-information diagnostic.

This is not a rerun of HGB, either Transformer, or the formal benchmark.
It does not modify experiment.py or the saved results. It compares the original
four predictive features with one added training-only per-item log-price mean.
No held-out demand enters the added feature or coefficient estimation.
"""
import ast
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'code' / 'experiment.py'
DEST = ROOT / 'results/diagnostics/price_anchor.json'
DEST.parent.mkdir(parents=True, exist_ok=True)
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'))
KEEP = {'World', 'experimental_worlds', 'unit_variance_laplace',
        'generate_panel', 'split_and_add_item_mean'}
SELECTED = ast.Module(body=[n for n in TREE.body
                          if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                          and n.name in KEEP], type_ignores=[])
exec(compile(SELECTED, str(SOURCE), 'exec'), globals())

MULT = np.array([.85, .90, .95, 1., 1.05, 1.10, 1.15])
OUTPUT = {
    'diagnostic_only': True,
    'formal_results_modified': False,
    'seed': 0,
    'method': 'numpy.linalg.lstsq OLS with an intercept and log_units target',
    'generator': 'AST-extracted current World/generate_panel/split_and_add_item_mean',
    'added_feature': 'training-only item mean(log_price), mapped unchanged to test',
    'decision_rule': 'argmax[(candidate_price - unit_cost) * multiplier ** fitted_price_slope]',
    'reason_decision_rule_is_equivalent':
        'All non-price regressors are held fixed across candidates, so their positive demand factor cancels.',
    'evaluation': 'Same held-out demand, seven prices, and simulator regret as experiment.py',
    'limitations': [
        'One seed and two worlds; no replication uncertainty.',
        'An OLS diagnostic is not a measured result for retrained HGB/Transformer models.',
        'The original item_log_mean follows the existing full-training aggregation procedure.',
        'Adding a stable price control cannot remove the hidden time-varying demand confounder.'
    ],
    'source_path': SOURCE.relative_to(ROOT).as_posix(),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'script_path': Path(__file__).resolve().relative_to(ROOT).as_posix(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'analytic_clean_pooled_slope_approximation': -2.62*.15**2/(.57**2+.15**2),
    'worlds': [],
}
for world_name in ['higher_elasticity', 'hidden_confounding']:
    w = experimental_worlds()[world_name]
    train, test = split_and_add_item_mean(generate_panel(w, 0))
    price_means = train.groupby('item_id')['log_price'].mean()
    train['item_log_price_mean'] = train.item_id.map(price_means)
    test['item_log_price_mean'] = test.item_id.map(price_means)
    true_demand = test.simulator_demand.to_numpy()[:, None]*MULT[None, :]**w.beta
    candidate_prices = test.price.to_numpy()[:, None]*MULT[None, :]
    costs = test.unit_cost.to_numpy()[:, None]
    evaluated_profit = (candidate_prices-costs)*true_demand
    oracle_total = np.max(evaluated_profit, axis=1).sum()
    entry = {
        'world': world_name, 'true_beta': w.beta,
        'train_rows': len(train), 'test_rows': len(test),
        'oracle_option_counts': np.bincount(np.argmax(evaluated_profit, axis=1), minlength=7).tolist(),
        'pipelines': [],
    }
    for label, features in [
        ('original_four_features', ['log_price', 'season', 'cost_shock', 'item_log_mean']),
        ('add_train_only_item_price_mean', ['log_price', 'season', 'cost_shock', 'item_log_mean', 'item_log_price_mean']),
    ]:
        x = np.column_stack([np.ones(len(train)), train[features].to_numpy()])
        coef = np.linalg.lstsq(x, train.log_units.to_numpy(), rcond=None)[0]
        beta = float(coef[1])
        chosen = np.argmax((candidate_prices-costs)*MULT[None, :]**beta, axis=1)
        realized_total = evaluated_profit[np.arange(len(test)), chosen].sum()
        entry['pipelines'].append({
            'name': label, 'features': features, 'price_slope': beta,
            'coefficients': dict(zip(['intercept', *features], coef.tolist())),
            'regret_pct': float(100*(oracle_total-realized_total)/oracle_total),
            'upper_15_share_pct': float(100*np.mean(chosen == 6)),
            'option_counts': np.bincount(chosen, minlength=7).tolist(),
        })
    OUTPUT['worlds'].append(entry)
DEST.write_text(json.dumps(OUTPUT, indent=2), encoding='utf-8')
print(json.dumps(OUTPUT, indent=2))
