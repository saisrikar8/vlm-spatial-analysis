#!/usr/bin/env python3
"""Plot accuracy from the verified processor-resolution experiment summary."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'analysis/hpc-resolution-color-matching-v1'


def main():
    s=json.loads((OUTPUT/'resolution-color-matching-41029953/summary.json').read_text())['test']
    values=[100*s['conditions'][c]['accuracy'] for c in ['baseline','high']]
    fig,ax=plt.subplots(figsize=(5,4))
    bars=ax.bar(['Baseline\n616 visual tokens','Higher resolution\n2,610 visual tokens'],values,color=['#4c72b0','#dd8452'],width=.55)
    ax.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=4)
    ax.set(title=f'Full-map color matching ({s["n"]} points)',ylabel='Accuracy (%)',ylim=(0,100))
    ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(OUTPUT/f'resolution_accuracy.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
