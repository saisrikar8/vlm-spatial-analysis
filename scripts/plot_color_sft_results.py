#!/usr/bin/env python3
"""Grouped accuracy bars for the source-disjoint color-reading LoRA pilot."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'analysis/hpc-color-sft-v1'


def main():
    summary=json.loads((OUTPUT/'color-sft-41030911/summary.json').read_text())
    fig,ax=plt.subplots(figsize=(6,4))
    for offset,backend,label,color in [(-.18,'base','Base model','#4c72b0'),(.18,'adapted','LoRA adapter','#dd8452')]:
        values=[100*summary['conditions'][c][backend]['accuracy'] for c in ['full','crop']]
        bars=ax.bar([offset,1+offset],values,width=.36,label=label,color=color)
        ax.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=4)
    ax.set(xticks=[0,1],xticklabels=['Full map','Enlarged crop'],ylim=(0,100),ylabel='Accuracy (%)',title='Color-reading LoRA pilot (48 test points)')
    ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,loc='upper left');fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(OUTPUT/f'color_sft_accuracy.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
