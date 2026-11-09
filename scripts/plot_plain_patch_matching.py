#!/usr/bin/env python3
"""Plot the three conditions from the completed common-prompt run."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'analysis/hpc-plain-patch-matching-v1'


def main():
    s=json.loads((OUTPUT/'plain-patch-matching-40967775/summary.json').read_text())['test']
    values=[100*s['conditions'][c]['accuracy'] for c in ['full','crop','patch']]
    fig,ax=plt.subplots(figsize=(6,4))
    bars=ax.bar(['Full map','Enlarged crop','Plain patch'],values,color=['#4c72b0','#dd8452','#55a868'],width=.6)
    ax.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=4)
    ax.set(title=f'Color matching ({s["n"]} points)',ylabel='Accuracy (%)',ylim=(0,100))
    ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    for suffix in ['png','svg']:fig.savefig(OUTPUT/f'plain_patch_accuracy.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
