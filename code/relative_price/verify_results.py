"""Re-score saved relative-price results without retraining any model.

The simulator in code/experiment.py regenerates the test data of every world and
seed. The saved price actions (and, for the predictive models, the saved
seven-price log-demand forecasts) are re-scored with simulator-known demand, and
the recomputed Oracle regret must match seed_results.csv to within 1e-12.

The script also checks that code/experiment.py is byte-identical to the simulator
recorded in the run manifest (SHA-256), and that the archived benchmark rows used
for the pairing are unchanged.

Run from the repository root:
    python code/relative_price/verify_results.py                      # accepted results
    python code/relative_price/verify_results.py --results-dir results/relative_price_reproduced

For a rerun, the summary is also saved as verification.json in that folder; the
accepted record results/relative_price/verification.json is left untouched.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import experiment as e  # noqa: E402

REFERENCE_DIR = ROOT / 'results/relative_price'
PREDICTIVE = ['HGB', 'TransformerSmall', 'TransformerLarge']
# Manifest keys may use the original project layout (for example
# 'Causal_Pricing_Final/experiment.py'), so inputs are matched by file name.
CURRENT_LOCATION = {
    'experiment.py': ROOT / 'code/experiment.py',
    'matched_seed_results.csv': ROOT / 'results/benchmark/matched_seed_results.csv',
}

parser = argparse.ArgumentParser(description='Re-score saved relative-price results.')
parser.add_argument('--results-dir', default='results/relative_price',
                    help='results folder, relative to the repository root')
args = parser.parse_args()
P = (ROOT / args.results_dir).resolve()

manifest = json.loads((P/'manifest.json').read_text(encoding='utf-8'))
assert manifest['status'] == 'complete'
df = pd.read_csv(P/'seed_results.csv')
assert len(df) == manifest['record_count'] and df.groupby(['world','seed']).size().eq(7).all()

# 1. Re-score every saved decision against simulator-known demand.
max_error = 0.
forecast_matrices = 0
for (world_name,seed),g in df.groupby(['world','seed']):
    world=e.experimental_worlds()[world_name]
    train,test=e.split_and_add_item_mean(e.generate_panel(world,int(seed),500,130))
    prices=test.price.to_numpy()[:,None]*e.PRICE_MULTIPLIERS
    costs=test.unit_cost.to_numpy()[:,None]
    truth=e.simulator_demand_at_candidate_prices(test,world,np.log(e.PRICE_MULTIPLIERS)[None,:])
    trueprofit=(prices-costs)*truth
    oracle=trueprofit.max(axis=1).sum()
    for r in g.itertuples():
        with np.load(P/'predictions'/f'{world_name}_{seed}_{r.method}.npz') as saved:
            actions=saved['actions']
            assert actions.shape==(13000,) and np.all((actions>=0)&(actions<=6))
            logs=saved['log_forecasts']
            if logs.size:
                # Predictive models: the saved actions must follow from the saved forecasts.
                assert logs.shape==(13000,7) and np.isfinite(logs).all()
                assert np.array_equal(actions,((prices-costs)*np.exp(logs)).argmax(axis=1))
                forecast_matrices += 1
            realized=trueprofit[np.arange(13000),actions].sum()
            regret=1-realized/oracle
            error=abs(regret-r.oracle_regret)
            assert error<1e-12 and abs(np.mean(actions==6)-r.upper_share)<1e-12
            max_error=max(error,max_error)

# 2. Provenance: the simulator and the paired benchmark rows must be the ones the run used.
checked = []
revised_inputs = []
not_in_repository = []
for path, digest in manifest['protected_hashes'].items():
    name = path.replace('\\', '/').rsplit('/', 1)[-1]
    protected = CURRENT_LOCATION.get(name)
    if protected is None:
        # Report drafts hashed in the original project folder are not part of this repository.
        not_in_repository.append(name)
        continue
    if name == 'matched_seed_results.csv' and hashlib.sha256(protected.read_bytes()).hexdigest() != digest:
        # The benchmark file was later trimmed to the seven reported methods; the rows used here must be unchanged.
        used = pd.read_csv(P/'paired_results.csv').merge(pd.read_csv(protected), on=['world','seed','method'], validate='one_to_one')
        assert len(used) == len(df) and np.allclose(used.oracle_regret_old, used.oracle_regret, rtol=0, atol=1e-15), path
        revised_inputs.append(name)
        continue
    assert hashlib.sha256(protected.read_bytes()).hexdigest() == digest, path
    checked.append(name)
assert 'experiment.py' in checked, 'the manifest does not record the simulator hash'

qa = dict(status='passed', records=len(df),
          predictive_fits=int(df.method.isin(PREDICTIVE).sum()),
          slope_estimator_results=int((~df.method.isin(PREDICTIVE)).sum()),
          world_seed_combinations=int(df.groupby(['world','seed']).ngroups),
          prediction_matrices_verified=forecast_matrices, saved_actions_verified=len(df),
          max_regret_recomputation_error=max_error,
          simulator_sha256_matches_manifest=True,
          protected_inputs_revised_after_run_with_used_rows_unchanged=revised_inputs,
          hashed_documents_not_in_repository=not_in_repository)
if P != REFERENCE_DIR.resolve():
    (P/'verification.json').write_text(json.dumps(qa, indent=2))
print(json.dumps(qa, indent=2))
