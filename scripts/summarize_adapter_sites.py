#!/usr/bin/env python3
"""Collect the three completed adapter-site pilots into one comparison table."""
import argparse
import json
from pathlib import Path
from color_matching import dump,table

SITES=('language','vision','joint')


def summarize(root):
    meta={};summaries={};baseline={};rows=[];identities=set()
    for site in SITES:
        run=root/f'full-{site}';m=json.loads((run/'run.json').read_text());s=json.loads((run/'summary.json').read_text())
        if m['status']!='finished' or m['adapter_site']!=site or m['smoke']:raise ValueError('Incomplete or incorrect site run')
        identities.add(json.dumps(dict(dataset=m['dataset_sha256'],hp=m['hyperparameters'],generation=m['generation'],prompt=m['prompt'],revision=m['revision']),sort_keys=True))
        meta[site]=m;summaries[site]=s
        raw=[json.loads(line) for line in (run/'responses.jsonl').read_text().splitlines()]
        baseline[site]={r['id']:r['raw_response'].strip() for r in raw if r['backend']=='base'}
        for condition in ['full','crop']:
            for backend in ['base','adapted']:
                d=s['conditions'][condition][backend]
                rows.append(dict(adapter_site=site,condition=condition,backend=backend,n=d['n'],correct=d['correct'],valid=d['valid'],accuracy=d['accuracy'],
                    trainable_parameters=m['trainable_parameters'],peak_allocated_bytes=m['peak_allocated_bytes'],training_seconds=m['training_seconds']))
    if len(identities)!=1:raise ValueError('Mixed dataset/protocol/settings')
    if len({tuple(sorted(b.keys())) for b in baseline.values()})!=1:raise ValueError('Different baseline membership')
    disagreements=sum(len({baseline[s][eid] for s in SITES})>1 for eid in baseline['language'])
    table(root/'comparison.csv',rows)
    dump(root/'comparison.json',dict(status='finished',sites=list(SITES),baseline_disagreements=disagreements,
        results=summaries,parameter_counts={s:meta[s]['trainable_parameters_by_site'] for s in SITES},
        notes=['All original weights frozen; compare adapter sites, not full unfreezing.',
            'Rank/epochs/data/settings equal; trainable parameter counts and compute are not matched.',
            'Same three previously inspected held-out groups; exploratory comparison, not independent confirmation.',
            'Vision uses fused qkv plus output projection; language uses q/v, so target operators differ.']))
    print(json.dumps(dict(baseline_disagreements=disagreements,rows=rows)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();summarize(a.root)
