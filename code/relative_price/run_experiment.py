"""Relative-price paired experiment: the main comparison in the report (Section 6).

For every world and seed this script
  * refits the four slope estimators (IV, DML, Naive OLS, DirectLiNGAM) on the
    unchanged simulated panel and asserts that their estimates and regrets equal
    the archived benchmark records (results/benchmark/matched_seed_results.csv);
  * refits the three predictive models (HGB, Tabular Transformer Small and Large)
    with one change: the log-price feature becomes log price minus the product's
    mean log price over the training weeks.
Each method then picks one of seven candidate prices for every product-week, and
the choice is scored with simulator-known demand (Oracle regret).

Run from the repository root:
    python code/relative_price/run_experiment.py                      # 5 worlds x 5 seeds
    python code/relative_price/run_experiment.py --worlds higher_elasticity --seeds 0

Results go to results/relative_price_reproduced/ by default, so the accepted
results in results/relative_price/ are never overwritten.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import experiment as e  # noqa: E402  simulator, estimators, models and scoring

# Order used for the accepted run: the clean elastic reference first, then the rest.
ALL_WORLDS = ['higher_elasticity', 'hidden_confounding', 'weak_instrument',
              'exclusion_violated', 'baseline']
MATCHED_SEEDS = [0, 1, 2, 3, 4]  # seeds with archived benchmark records to pair with
REFERENCE_DIR = ROOT / 'results/relative_price'

parser = argparse.ArgumentParser(description='Run the relative-price paired experiment.')
parser.add_argument('--output', default='results/relative_price_reproduced',
                    help='output folder, relative to the repository root')
parser.add_argument('--worlds', default=','.join(ALL_WORLDS),
                    help='comma-separated worlds (default: all five)')
parser.add_argument('--seeds', default=','.join(map(str, MATCHED_SEEDS)),
                    help='comma-separated seeds from 0-4 (default: all five)')
args = parser.parse_args()
worlds = [name.strip() for name in args.worlds.split(',') if name.strip()]
seeds = [int(seed) for seed in args.seeds.split(',') if seed.strip()]
if set(worlds) - set(ALL_WORLDS):
    raise SystemExit(f'Unknown worlds {sorted(set(worlds) - set(ALL_WORLDS))}; choose from {ALL_WORLDS}')
if set(seeds) - set(MATCHED_SEEDS):
    raise SystemExit('Seeds must be in 0-4: only these have archived benchmark records to pair with.')

OUT = (ROOT / args.output).resolve()
if OUT == REFERENCE_DIR.resolve():
    raise SystemExit('Refusing to overwrite the accepted results in results/relative_price; '
                     'choose another --output.')
OUT.mkdir(parents=True, exist_ok=True)
(OUT/'predictions').mkdir(exist_ok=True)
(OUT/'training_means').mkdir(exist_ok=True)

SOURCE = ROOT / 'code/experiment.py'
OLD = ROOT / 'results/benchmark/matched_seed_results.csv'
# Inputs that must not change during the run. Keys are repository-relative paths,
# so the manifest reads the same on every machine.
protected = [SOURCE, OLD]
hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
old = pd.read_csv(OLD)
methods = ['HGB', 'TransformerSmall', 'TransformerLarge']
records = []


def save_summary():
    """Rewrite the raw, paired and summary CSVs after every record, so an
    interrupted run keeps everything finished so far."""
    df = pd.DataFrame(records)
    df.to_csv(OUT/'seed_results.csv', index=False)
    # Pair each new record with the archived original-input record of the same world, seed and method.
    paired = df.merge(old[['world','seed','method','oracle_regret']], on=['world','seed','method'], suffixes=('_new','_old'), validate='one_to_one')
    paired['improvement_pp'] = 100*(paired.oracle_regret_old-paired.oracle_regret_new)
    paired.to_csv(OUT/'paired_results.csv',index=False)
    summary = paired.groupby(['world','method'],sort=False).agg(n=('seed','count'),original_regret=('oracle_regret_old','mean'),relative_regret=('oracle_regret_new','mean'),improvement_pp=('improvement_pp','mean'),upper_share=('upper_share','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)


def record(world_name,seed,method,demand,truth,prices,costs,beta=np.nan,seconds=0,logs=None):
    """Pick the candidate price that maximises profit under the method's `demand`,
    score that choice with the simulator's `truth`, and save the actions (plus the
    seven-price log-demand forecasts for predictive models)."""
    fp=e.profit(prices,demand,costs)
    action=fp.argmax(1)
    true_profit=e.profit(prices,truth,costs)
    total=e.select_price_and_evaluate(demand,prices,costs,truth)
    oracle=true_profit.max(1).sum()
    assert np.isclose(total,true_profit[np.arange(len(action)),action].sum())
    result=dict(world=world_name,seed=seed,method=method,variant='relative_price' if method in methods else 'unchanged_causal',oracle_regret=(oracle-total)/oracle,upper_share=float(np.mean(action==6)),beta_hat=beta,seconds=seconds,oracle_profit=float(oracle))
    if logs is not None:
        # Implied price slope of the forecasts between the lowest and highest candidate price.
        slope=(logs[:,-1]-logs[:,0])/np.log(e.PRICE_MULTIPLIERS[-1]/e.PRICE_MULTIPLIERS[0])
        result['median_price_slope']=float(np.median(slope))
    else:
        result['median_price_slope']=beta
    records.append(result)
    save_summary()
    np.savez_compressed(OUT/'predictions'/f'{world_name}_{seed}_{method}.npz',actions=action,log_forecasts=logs if logs is not None else np.array([]))
    print(json.dumps(result),flush=True)


manifest=dict(status='running',scope=f'{len(worlds)} world(s) x seeds {seeds}; three predictive models with training-only item-relative log price; four unchanged slope estimators rerun',worlds=worlds,seeds=seeds,protected_hashes=hashes,epochs=12,n_sku=500,n_week=130,price_definition='log_price minus item training-period mean log_price')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
try:
    for world_name in worlds:
        for seed in seeds:
            world=e.experimental_worlds()[world_name]
            # 500 products x 130 weeks: the first 104 weeks train, the last 26 weeks test.
            train,test=e.split_and_add_item_mean(e.generate_panel(world,seed,500,130))
            assert len(train)==52000 and len(test)==13000 and train.week.max()<test.week.min()
            # Relative price: subtract each product's training-period mean log price (no test data used).
            means=train.groupby('item_id').log_price.mean()
            reltrain,reltest=train.copy(),test.copy()
            for frame in (reltrain,reltest):
                frame['log_price']=frame.log_price-frame.item_id.map(means)
            assert reltrain.groupby('item_id').log_price.mean().abs().max()<1e-12
            assert np.array_equal(reltest.price.to_numpy(),test.price.to_numpy())
            # Candidate transformation must equal log(actual candidate price) minus frozen training mean.
            for m in e.PRICE_MULTIPLIERS:
                assert np.allclose(reltest.log_price+np.log(m),np.log(test.price*m)-test.item_id.map(means))
            means.to_csv(OUT/'training_means'/f'{world_name}_{seed}_training_price_means.csv')
            prices=test.price.to_numpy()[:,None]*e.PRICE_MULTIPLIERS
            costs=test.unit_cost.to_numpy()[:,None]
            truth=e.simulator_demand_at_candidate_prices(test,world,np.log(e.PRICE_MULTIPLIERS)[None,:])
            # Slope estimators use the unchanged data and must reproduce the archived records exactly.
            beta,first_stage=e.estimate_betas(train)
            indices=e.method_demand_indices_at_candidate_prices({},test,beta)
            for method,b in beta.items():
                reference=old[(old.world==world_name)&(old.seed==seed)&(old.method==method)].iloc[0]
                assert np.isclose(b,reference.beta_hat,rtol=1e-7,atol=1e-8),(world_name,seed,method,b,reference.beta_hat)
                record(world_name,seed,method,indices[method],truth,prices,costs,beta=b)
                assert np.isclose(records[-1]['oracle_regret'],reference.oracle_regret,atol=1e-9)
            # Predictive models are refitted on the relative-price features.
            for method in methods:
                start=time.perf_counter()
                if method=='HGB': model=e.fit_hgb_model(reltrain,seed)
                else: model=e.fit_transformer_model(reltrain,seed,32 if method=='TransformerSmall' else 96,2 if method=='TransformerSmall' else 3)
                logs=e.predictor_log_demand_at_candidate_prices(model,reltest)
                assert np.isfinite(logs).all()
                record(world_name,seed,method,np.exp(logs),truth,prices,costs,seconds=time.perf_counter()-start,logs=logs)
    assert len(records)==7*len(worlds)*len(seeds)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    manifest.update(status='complete',record_count=len(records),protected_files_unchanged=True)
except BaseException as exc:
    manifest.update(status='failed',error=repr(exc),record_count=len(records))
    raise
finally:
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
