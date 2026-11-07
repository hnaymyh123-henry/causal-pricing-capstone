from pathlib import Path
import ast, dataclasses, json
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[2]
code=(root/'code/experiment.py').read_text(encoding='utf-8')
names={'World','experimental_worlds','unit_variance_laplace','generate_panel','split_and_add_item_mean'}
nodes=[n for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
g={'np':np,'pd':pd,'dataclass':dataclasses.dataclass}
exec(compile(ast.Module(nodes,type_ignores=[]),'<unchanged simulator functions>','exec'),g)
rows=[]
for name,world in g['experimental_worlds']().items():
 for seed in range(5):
  train,test=g['split_and_add_item_mean'](g['generate_panel'](world,seed))
  features=['log_price','season','cost_shock','item_log_mean']
  x=np.column_stack([np.ones(len(train)),train[features].to_numpy()]); y=train.log_units.to_numpy()
  coef=np.linalg.lstsq(x,y,rcond=None)[0]
  # Diagnostic only: expose product-relative price to the same simple predictor.
  # No changes to the formal experiment or result files.
  means=train.groupby('item_id').log_price.mean()
  rel=train.log_price-train.item_id.map(means)
  xr=x.copy();xr[:,1]=rel
  cofr=np.linalg.lstsq(xr,y,rcond=None)[0]
  rows.append({'world':name,'seed':seed,'true_beta':world.beta,'same_features_price_slope':float(coef[1]),'relative_price_diagnostic_slope':float(cofr[1])})
p=root/'results/relative_price/diagnostics'
p.mkdir(parents=True,exist_ok=True)
(p/'linear_probe.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).groupby('world')[['true_beta','same_features_price_slope','relative_price_diagnostic_slope']].mean().to_string())
