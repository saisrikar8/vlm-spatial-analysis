"""Plot ordering accuracy for the 46 land-coordinate benchmark pairs."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
with (folder / 'benchmark_46_land_pairs_raw.csv').open(encoding='utf-8-sig', newline='') as stream:
    rows = list(csv.DictReader(stream))

labels = ['Extract + calculate', 'Direct answer', 'Always false']
counts = [
    sum(row['calculated_order_correct'] == 'True' for row in rows),
    sum(row['direct_order_correct'] == 'True' for row in rows),
    sum(row['reference_A_greater_B'] == 'False' for row in rows),
]
accuracy = [100 * count / len(rows) for count in counts]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(labels, accuracy, color=['tab:blue', 'tab:orange', 'tab:gray'])
ax.bar_label(bars, labels=[f'{value:.1f}%' for value in accuracy], padding=3)
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 100)
ax.set_title('Ordering accuracy (46 pairs)')
fig.tight_layout()
for extension in ('png', 'svg'):
    fig.savefig(folder / f'real_map_ordering_accuracy.{extension}', dpi=200)
plt.close(fig)
