#!/usr/bin/env python3
"""Plot selections of the same blue distractor before/after recoloring the ring."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'analysis/hpc-marker-available-color-v1'


def main():
    s=json.loads((OUTPUT/'marker-available-color-40949906/summary.json').read_text())['test']
    values=[s['red_marker_option_selected'],s['blue_marker_option_selected']]
    fig,ax=plt.subplots(figsize=(5,4))
    bars=ax.bar(['Red outline','Blue outline'],values,color=['#c44e52','#4c72b0'],width=.55)
    ax.bar_label(bars,padding=4)
    ax.set(title='Selections of the designated blue swatch',ylabel='Number of selections',ylim=(0,s['n']))
    ax.set_yticks(range(0,s['n']+1,4));ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for suffix in ['png','svg']:fig.savefig(OUTPUT/f'blue_swatch_selections.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
