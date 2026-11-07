"""No-training diagnostic of fixed grid policies, kept outside formal results."""
import ast, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
source = root / 'code' / 'experiment.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
keep = {'World','experimental_worlds','unit_variance_laplace','generate_panel','split_and_add_item_mean'}
selected = ast.Module(body=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in keep],type_ignores=[])
exec(compile(selected,str(source),'exec'),globals())
raw = pd.read_csv(root / 'results/benchmark/seed_results.csv')
multipliers = np.array([.85,.90,.95,1.,1.05,1.10,1.15])
records = []
for world_name, world in experimental_worlds().items():
    for seed in range(5):
        _, test = split_and_add_item_mean(generate_panel(world,seed))
        true_demand = test.simulator_demand.to_numpy()[:,None]*multipliers[None,:]**world.beta
        price = test.price.to_numpy()[:,None]*multipliers[None,:]
        realized_profit = (price-test.unit_cost.to_numpy()[:,None])*true_demand
        oracle = realized_profit.max(axis=1).sum()
        values = {'world':world_name,'seed':seed}
        for label,index in [('fixed_minus15',0),('unchanged_price',3),('fixed_plus15',6)]:
            values[label+'_regret_pct'] = float(100*(oracle-realized_profit[:,index].sum())/oracle)
        for name in ['TransformerSmall','TransformerLarge']:
            formal = raw[(raw.world==world_name)&(raw.seed==seed)&(raw.method==name)].oracle_regret.iloc[0]*100
            values[name+'_profit_matches_fixed_plus15'] = bool(abs(formal-values['fixed_plus15_regret_pct']) < 1e-10)
        records.append(values)
frame = pd.DataFrame(records)
print(frame.groupby('world')[[c for c in frame if c.endswith('_regret_pct')]].mean().to_string())
print(frame.groupby('world')[[c for c in frame if c.endswith('_profit_matches_fixed_plus15')]].sum().to_string())
(root/'results/diagnostics/fixed_policies.json').write_text(json.dumps({'diagnostic_only':True,'note':'Profit equality does not independently establish identical per-row actions. No trained models were rerun.','records':records},indent=2),encoding='utf-8')
