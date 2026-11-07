"""Summarise a complete relative-price run (5 worlds x 5 seeds x 7 methods).

Reads paired_results.csv and writes, in the same folder:
  final_comparison.csv       per world and method: original vs relative-price regret,
                             paired improvement and its 95% t interval half-width
  elastic_world_aggregate.csv  the same per method, averaged over the four elastic worlds
  group_comparison.json      causal group (IV, DML, DirectLiNGAM) vs predictive group
                             (HGB, Tabular Small, Tabular Large)
  RESULTS.md                 a short Markdown table of the predictive-model changes

Run from the repository root:
    python code/relative_price/analyze_results.py
    python code/relative_price/analyze_results.py --results-dir results/relative_price_reproduced
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description='Summarise a complete relative-price run.')
parser.add_argument('--results-dir', default='results/relative_price',
                    help='results folder, relative to the repository root')
P = ROOT / parser.parse_args().results_dir

df=pd.read_csv(P/'paired_results.csv')
pred=['HGB','TransformerSmall','TransformerLarge']
assert len(df)==175 and not df.duplicated(['world','seed','method']).any()
assert df.groupby(['world','method']).size().eq(5).all()

# Per world and method: mean regret before/after and a paired 95% t interval over five seeds.
rows=[]
for (world,method),g in df.groupby(['world','method'],sort=False):
    d=g.improvement_pp
    rows.append(dict(world=world,method=method,n=5,old_regret_pct=100*g.oracle_regret_old.mean(),new_regret_pct=100*g.oracle_regret_new.mean(),improvement_pp=d.mean(),improvement_ci_halfwidth=t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5),upper_share_pct=100*g.upper_share.mean(),median_slope_mean=g.median_price_slope.mean()))
summary=pd.DataFrame(rows)
summary.to_csv(P/'final_comparison.csv',index=False)

# Four elastic worlds: average worlds within each seed first, so the five seeds are the independent blocks.
elastic=df[df.world!='baseline']
aggregate=[]
for method,g in elastic.groupby('method',sort=False):
    seeds=g.groupby('seed')[['oracle_regret_old','oracle_regret_new','improvement_pp']].mean()
    d=seeds.improvement_pp
    aggregate.append(dict(method=method,old_regret_pct=100*seeds.oracle_regret_old.mean(),new_regret_pct=100*seeds.oracle_regret_new.mean(),improvement_pp=d.mean(),ci_halfwidth=t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5),independent_seed_blocks=len(seeds)))
pd.DataFrame(aggregate).to_csv(P/'elastic_world_aggregate.csv',index=False)

# Group comparison: equal-weight mean of each group's methods in every world-seed cell.
groups={}
for name,member in [('causal',['IV','DML','DirectLiNGAM']),('predictive',pred)]:
    groups[name]=elastic[elastic.method.isin(member)].groupby(['seed','world'])[['oracle_regret_old','oracle_regret_new']].mean()
gap=100*(groups['predictive']-groups['causal'])
seedgap=gap.groupby('seed').mean()
comparison=dict(causal_members=['IV','DML','DirectLiNGAM'],predictive_members=pred,causal_mean_pct=100*groups['causal'].oracle_regret_new.mean(),predictive_original_mean_pct=100*groups['predictive'].oracle_regret_old.mean(),predictive_relative_mean_pct=100*groups['predictive'].oracle_regret_new.mean(),new_gap_pp=seedgap.oracle_regret_new.mean(),new_gap_ci_halfwidth=t.ppf(.975,4)*seedgap.oracle_regret_new.std(ddof=1)/np.sqrt(5),causal_lower_cells=int((gap.oracle_regret_new>0).sum()),total_world_seed_cells=20,independent_seed_blocks=5)
(P/'group_comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')

# The slope estimators use unchanged data, so their regrets must equal the archived ones.
causal=df[~df.method.isin(pred)]
assert np.allclose(causal.oracle_regret_new,causal.oracle_regret_old,atol=1e-9)

lines=['# Relative-price paired experiment','', 'Five environments, five matched seeds, 500 products, 104 training weeks and 26 test weeks. Original formal results and report are unchanged.','', 'Only the predictive price feature changes: log price minus the product mean log price calculated on the training period. Demand targets, other inputs, training budgets, candidate prices and profit scoring remain unchanged. Four slope estimators are rerun on unchanged data and checked against the original records.','', '| Environment | Model | Original regret % | Relative-price regret % | Improvement pp |','|---|---|---:|---:|---:|']
for r in rows:
    if r['method'] in pred: lines.append(f"| {r['world']} | {r['method']} | {r['old_regret_pct']:.3f} | {r['new_regret_pct']:.3f} | {r['improvement_pp']:.3f} |")
lines+=['','Positive improvement means lower regret. Overall elastic-world intervals average worlds within each seed before computing uncertainty; there are five independent seed blocks, not twenty independent observations. This is a representation sensitivity experiment, not a fully controlled architecture comparison.']
(P/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print(summary[summary.method.isin(pred)].to_string(index=False))
