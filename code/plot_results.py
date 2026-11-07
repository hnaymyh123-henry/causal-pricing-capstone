"""Standalone analytical figures from the independently accepted CSV snapshot."""
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/benchmark'
raw = pd.read_csv(RESULTS / 'matched_seed_results.csv', float_precision='round_trip')
worlds = ['baseline', 'higher_elasticity', 'weak_instrument', 'hidden_confounding', 'exclusion_violated']
labels = ['Inelastic reference', 'Elastic reference', 'Weak instrument', 'Hidden confounding', 'Exclusion violated']
methods = ['IV', 'DML', 'Naive', 'DirectLiNGAM', 'HGB', 'TransformerSmall', 'TransformerLarge']
short = ['IV', 'DML', 'Naive', 'LiNGAM', 'HGB', 'T-Small', 'T-Large']
grid = (raw.groupby(['world', 'method']).oracle_regret.mean().unstack() * 100).reindex(worlds)[methods]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(6.65, 2.85))
img = ax.imshow(grid, cmap='Blues', vmin=0, vmax=13, aspect='auto')
ax.set_xticks(range(len(methods)), short, fontsize=9)
ax.set_yticks(range(5), labels, fontsize=9)
ax.tick_params(length=0)
for row in range(5):
    for col in range(len(methods)):
        v = grid.iloc[row, col]
        ax.text(col, row, f'{v:.2f}', ha='center', va='center', fontsize=9,
                color='white' if v > 7.5 else '#17212c')
for edge in ax.spines.values():
    edge.set_visible(False)
bar = fig.colorbar(img, ax=ax, fraction=.033, pad=.025, ticks=[0, 4, 8, 12])
bar.ax.set_ylabel('Regret (%)', fontsize=9)
bar.ax.tick_params(labelsize=8, length=2)
fig.tight_layout(pad=.7)
fig.savefig(RESULTS / 'report_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.65, 3.1))
predictors = ['HGB', 'TransformerSmall', 'TransformerLarge']
names = ['HGB', 'T-Small', 'T-Large']
colors = ['#697b8c', '#aac0d4', '#3b719d']
y = np.arange(4)
height = .22
for j, (method, name, color) in enumerate(zip(predictors, names, colors)):
    sub = raw[(raw.method == method) & (raw.world != 'baseline')]
    mean = (sub.groupby('world').oracle_regret.mean() * 100).reindex(worlds[1:])
    se = (sub.groupby('world').oracle_regret.std(ddof=1) * 100 / np.sqrt(5)).reindex(worlds[1:])
    ax.barh(y + (j - 1) * height, mean, height=height, label=name, color=color,
            xerr=2.7764451051977987 * se, error_kw={'elinewidth': .65, 'capsize': 2, 'ecolor': '#17212c'})
ax.set_yticks(y, labels[1:], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 14.4)
ax.set_xlabel('Mean Oracle regret (%)', fontsize=10)
ax.grid(axis='x', color='#dddddd', linewidth=.55)
ax.set_axisbelow(True)
ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(.49, 1.115), frameon=False,
          fontsize=9, columnspacing=1, handlelength=1.1)
fig.tight_layout(pad=.9)
fig.savefig(RESULTS / 'report_predictive_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
available = pd.read_csv(RESULTS / 'summary.csv', float_precision='round_trip')
full = available.pivot(index='world', columns='method', values='oracle_regret_pct').reindex(worlds)[methods]
counts = available.pivot(index='world', columns='method', values='n_seeds').reindex(worlds)[methods]
fig, ax = plt.subplots(figsize=(6.65, 3.15))
img = ax.imshow(full, cmap='Blues', vmin=0, vmax=13, aspect='auto')
ax.set_xticks(range(len(methods)), short, fontsize=9)
ax.set_yticks(range(5), labels, fontsize=9)
ax.tick_params(length=0)
for row in range(5):
    for col in range(len(methods)):
        v = full.iloc[row, col]
        ax.text(col, row, f'{v:.2f}\nn={int(counts.iloc[row,col])}', ha='center', va='center', fontsize=8,
                color='white' if v > 7.5 else '#17212c')
for edge in ax.spines.values():
    edge.set_visible(False)
bar = fig.colorbar(img, ax=ax, fraction=.033, pad=.025)
bar.ax.set_ylabel('Regret (%)', fontsize=9)
bar.ax.tick_params(labelsize=8)
ax.set_title('All available seeds  differing coverage shown in each cell', fontsize=9, pad=8)
fig.tight_layout(pad=.7)
fig.savefig(RESULTS / 'heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved two matched figures and the full-coverage heatmap')
