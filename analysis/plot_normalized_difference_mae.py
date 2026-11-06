"""Compare source-balanced normalized A-B MAE on the same valid pairs."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
with (folder / 'benchmark_46_land_pairs_raw.csv').open(encoding='utf-8-sig', newline='') as stream:
    rows = list(csv.DictReader(stream))

matched = [row for row in rows if row['calculated_difference_valid'] == 'True'
           and row['direct_difference_valid'] == 'True']
if not matched:
    raise ValueError('No pairs have valid difference predictions from both methods.')

scores = []
for method in ('calculated', 'direct'):
    sources = defaultdict(list)
    for row in matched:
        sources[row['source_group']].append(float(row[f'{method}_difference_error_normalized']))
    scores.append(mean(mean(values) for values in sources.values()))

fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(['Extract + calculate', 'Direct answer'], scores,
              color=['tab:blue', 'tab:orange'])
ax.bar_label(bars, labels=[f'{score:.4f}' for score in scores], padding=3)
ax.set_ylabel('Normalized difference MAE')
ax.set_title(f'Difference error ({len(matched)} matched pairs)')
ax.set_ylim(0, max(scores) * 1.2)
fig.tight_layout()
for extension in ('png', 'svg'):
    fig.savefig(folder / f'normalized_difference_mae.{extension}', dpi=200)
plt.close(fig)

for label, score in zip(('Extract + calculate', 'Direct answer'), scores):
    print(f'{label}: {score:.4f}')
