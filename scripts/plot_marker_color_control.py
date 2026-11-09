#!/usr/bin/env python3
"""Plot completed marker-color results from their saved summary."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'analysis/hpc-marker-color-control-v1/marker-color-control-40947249'


def main():
    result = json.loads((RUN / 'summary.json').read_text())['test']
    n = result['n']
    counts = [result['red_correct'], result['magenta_correct']]
    accuracy = [100 * count / n for count in counts]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(['Red marker', 'Magenta marker'], accuracy,
                  color=['#d9534f', '#b64cbb'], width=0.55)
    ax.bar_label(bars, labels=[f'{value:.1f}%' for value in accuracy], padding=4)
    ax.set(ylabel='Color-matching accuracy (%)', ylim=(0, 100),
           title=f'Marker color comparison ({n} points)')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    for suffix in ['png', 'svg']:
        path = RUN.parent / f'marker_color_accuracy.{suffix}'
        fig.savefig(path, dpi=180)
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
