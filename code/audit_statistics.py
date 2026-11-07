from pathlib import Path
import pandas as pd
import numpy as np

root = Path(__file__).resolve().parents[1]
raw = pd.read_csv(root / 'results/benchmark/seed_results.csv')
summary = pd.read_csv(root / 'results/benchmark/summary.csv')
raw['regret_pct'] = 100*raw.oracle_regret
causal = ['IV', 'DML', 'DirectLiNGAM']
pred = ['HGB', 'TransformerSmall', 'TransformerLarge']
matched = raw[(raw.seed < 5) & (raw.world != 'baseline')]
grid = matched.pivot(index=['world','seed'], columns='method', values='regret_pct')
grid['causal'] = grid[causal].mean(axis=1)
grid['pred'] = grid[pred].mean(axis=1)
grid['difference'] = grid['pred']-grid['causal']
def describe(x):
    x = np.asarray(x, dtype=float)
    return {'n':len(x),'mean':float(x.mean()),'sd':float(x.std(ddof=1)),
      't_ci95_halfwidth':float({5:2.776445105197799,25:2.06389856162802,30:2.045229642132703}[len(x)]*x.std(ddof=1)/np.sqrt(len(x))),
      'min':float(x.min()),'median':float(np.median(x)),'max':float(x.max())}
print('INTEGRITY', len(raw), len(summary), 'duplicate keys', raw.duplicated(['world','seed','method']).sum())
print('COVERAGE',raw.groupby(['world','method']).seed.agg(['count','min','max']).to_string())
print('MATCHED MEANS',grid[['causal','pred','difference']].mean().to_dict(), 'wins',int((grid.difference>0).sum()))
print('WORLD GROUPS',grid.groupby(level='world')[['causal','pred','difference']].mean().to_string())
print('SEED CLUSTERS',grid.groupby(level='seed')[['causal','pred','difference']].mean().to_string())
print('CLUSTERED DIFFERENCE CI',describe(grid.groupby(level='seed').difference.mean()))
print('INCLUDING NAIVE',float(grid[causal+['Naive']].mean(axis=1).mean()),'ALL 5 WORLDS',
      raw[(raw.seed<5)&raw.method.isin(causal)].regret_pct.mean(),raw[(raw.seed<5)&raw.method.isin(pred)].regret_pct.mean())
print('30 SEED GROUP MEANS',raw[(raw.world!='baseline')&raw.method.isin(causal)].regret_pct.mean(),raw[(raw.world!='baseline')&(raw.method=='HGB')].regret_pct.mean())
weak = raw[(raw.world=='weak_instrument')&(raw.method=='IV')].copy()
print('WEAK IV 30', describe(weak.regret_pct), 'BETA',describe(weak.beta_hat),'F',describe(weak.iv_first_stage_f))
print('WEAK IV first5',describe(weak[weak.seed<5].regret_pct),'remaining25',describe(weak[weak.seed>=5].regret_pct))
print('WEAK IV WORST',weak.nlargest(6,'regret_pct')[['seed','beta_hat','iv_first_stage_f','regret_pct']].to_string(index=False))
print('WEAK F UNDER 10',int((weak.iv_first_stage_f<10).sum()), 'WEAK REGRET OVER HGB',int((weak.set_index('seed').regret_pct > raw[(raw.world=='weak_instrument')&(raw.method=='HGB')].set_index('seed').regret_pct).sum()))
computed = raw.groupby(['world','method']).regret_pct.apply(lambda x:pd.Series(describe(x))).unstack()
merged = summary.merge(computed,left_on=['world','method'],right_index=True)
print('SUMMARY MAX ERRORS',float((merged.oracle_regret_pct-merged['mean']).abs().max()),float((merged.monte_carlo_ci95_half_width_pct-merged.t_ci95_halfwidth).abs().max()))
print('BETA ERROR',summary[summary.method.isin(causal+['Naive'])][['world','method','beta_hat_mean','absolute_elasticity_error_mean']].to_string(index=False))
print('WMAPE COVERAGE',raw.groupby(['world','method']).observed_wmape_pct.agg(['count','mean']).to_string())
print('ORACLE BOUNDARIES',summary[summary.method=='IV'][['world','oracle_lower_boundary_share','oracle_upper_boundary_share']].to_string(index=False))
