"""Redraw report Figures 2 and 3 from the relative-price records.

Figure 2: mean Oracle regret (%) of the seven methods in the five worlds (heatmap).
Figure 3: mean Oracle regret (%) in the clean elastic and hidden-confounding
          worlds for six methods (grouped bars).

Every value is the mean of `oracle_regret` over the five matched seeds in
seed_results.csv; the predictive models use the training-relative log price.
The plotting script originally used for the report was not kept, so this script
redraws the same values in the same layout. It also prints the value table.

Run from the repository root:
    python code/relative_price/plot_figures.py
    python code/relative_price/plot_figures.py --results-dir results/relative_price_reproduced
Figures are written to <results-dir>/figures/.
"""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use('Agg')  # write image files only; no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# Column order and labels used in the report.
WORLDS = {
    'baseline': 'Inelastic',
    'higher_elasticity': 'Clean\nelastic',
    'weak_instrument': 'Weak\ninstrument',
    'hidden_confounding': 'Hidden\nconfounding',
    'exclusion_violated': 'Invalid\ninstrument',
}
METHODS = {
    'IV': 'IV',
    'DML': 'DML',
    'Naive': 'Naive OLS',
    'DirectLiNGAM': 'DirectLiNGAM',
    'HGB': 'HGB',
    'TransformerSmall': 'Tabular Small',
    'TransformerLarge': 'Tabular Large',
}

parser = argparse.ArgumentParser(description='Redraw report Figures 2 and 3.')
parser.add_argument('--results-dir', default='results/relative_price',
                    help='results folder, relative to the repository root')
results_dir = ROOT / parser.parse_args().results_dir
out_dir = results_dir / 'figures'
out_dir.mkdir(exist_ok=True)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

records = pd.read_csv(results_dir / 'seed_results.csv')
worlds = [w for w in WORLDS if w in set(records.world)]  # a partial rerun may cover fewer worlds
# Mean regret in percent: one row per method, one column per world.
regret = (100 * records.groupby(['method', 'world']).oracle_regret.mean()).unstack()
regret = regret.loc[list(METHODS), worlds]
seed_counts = records.groupby(['method', 'world']).seed.nunique()
print(f'Seeds per cell: {seed_counts.min()}-{seed_counts.max()}')
print(regret.rename(index=METHODS, columns=lambda w: WORLDS[w].replace('\n', ' ')).round(2).to_string())

# Figure 2: heatmap of all methods and worlds.
fig, ax = plt.subplots(figsize=(8.0, 4.1))
image = ax.imshow(regret.to_numpy(), cmap='Blues', vmin=0, vmax=8, aspect='auto')
for i in range(regret.shape[0]):
    for j in range(regret.shape[1]):
        value = regret.iat[i, j]
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=10,
                color='white' if value > 4 else '#1f2d3d')  # light text on dark cells
ax.set_xticks(range(len(worlds)), [WORLDS[w] for w in worlds])
ax.set_yticks(range(len(METHODS)), list(METHODS.values()))
fig.colorbar(image, ax=ax, label='Oracle regret (%)')
fig.tight_layout()
fig.savefig(out_dir / 'figure2_regret_heatmap.png', dpi=220, facecolor='white')
plt.close(fig)
print(f'Wrote {out_dir / "figure2_regret_heatmap.png"}')

# Figure 3: clean elastic vs hidden confounding, without Naive OLS.
if not {'higher_elasticity', 'hidden_confounding'} <= set(worlds):
    print('Figure 3 skipped: it needs the higher_elasticity and hidden_confounding worlds.')
    raise SystemExit(0)
bar_methods = [m for m in METHODS if m != 'Naive']
clean = regret.loc[bar_methods, 'higher_elasticity'].to_numpy()
hidden = regret.loc[bar_methods, 'hidden_confounding'].to_numpy()
x = np.arange(len(bar_methods))
width = 0.36
fig, ax = plt.subplots(figsize=(8.6, 4.1))
for offset, values, label, color in [(-width / 2, clean, 'Clean elastic', '#a5b4c1'),
                                     (width / 2, hidden, 'Hidden confounding', '#2a6f9b')]:
    bars = ax.bar(x + offset, values, width, label=label, color=color)
    ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=2, fontsize=8)
ax.set_xticks(x, [METHODS[m] for m in bar_methods])
ax.set_ylabel('Mean Oracle regret (%)')
ax.set_ylim(0, 1.2 * max(clean.max(), hidden.max()))
ax.yaxis.grid(True, color='#e6e6e6')
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper left', ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(out_dir / 'figure3_clean_vs_hidden.png', dpi=240, facecolor='white')
plt.close(fig)
print(f'Wrote {out_dir / "figure3_clean_vs_hidden.png"}')
