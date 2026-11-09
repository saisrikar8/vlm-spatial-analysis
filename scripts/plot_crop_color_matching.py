#!/usr/bin/env python3
"""Plot full-map vs enlarged-crop accuracy from the completed run summary."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'analysis/hpc-crop-color-matching-v1'


def main():
    s=json.loads((OUTPUT/'crop-color-matching-40957950/summary.json').read_text())['test']
    values=[100*s['full_correct']/s['n'],100*s['crop_correct']/s['n']]
    fig,ax=plt.subplots(figsize=(5,4))
    bars=ax.bar(['Full map','Enlarged crop'],values,color=['#4c72b0','#dd8452'],width=.55)
    ax.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=4)
    ax.set(title=f'Color matching ({s["n"]} points)',ylabel='Accuracy (%)',ylim=(0,100))
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for extension in ['png','svg']:fig.savefig(OUTPUT/f'crop_color_accuracy.{extension}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
