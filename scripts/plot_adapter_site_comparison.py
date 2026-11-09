#!/usr/bin/env python3
"""Plot the verified adapter-site comparison from its saved aggregate."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'analysis/hpc-adapter-site-comparison-v1'


def main():
    data=json.loads((OUTPUT/'adapter-sites-41037688/comparison.json').read_text())['results']
    specs=[('Base',None,'#888888'),('Language','language','#4c72b0'),('Vision','vision','#dd8452'),('Joint','joint','#55a868')]
    fig,ax=plt.subplots(figsize=(7,4))
    for i,(label,site,color) in enumerate(specs):
        source=data[site or 'language']['conditions'];backend='adapted' if site else 'base'
        values=[100*source[c][backend]['accuracy'] for c in ['full','crop']]
        offset=(i-1.5)*.2;bars=ax.bar([offset,1+offset],values,width=.19,label=label,color=color)
        ax.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=3,fontsize=8)
    ax.set(xticks=[0,1],xticklabels=['Full map','Enlarged crop'],ylim=(0,105),ylabel='Accuracy (%)',title='Adapter-site comparison (48 test points)')
    ax.set_yticks(range(0,101,20));ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,ncol=4,loc='upper left',fontsize=9);fig.tight_layout()
    for suffix in ['png','svg']:fig.savefig(OUTPUT/f'adapter_site_accuracy.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
