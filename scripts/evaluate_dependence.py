#!/usr/bin/env python3
"""Paired dependence diagnostic; one source, not one generation, is the unit."""
import csv
import getopt
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
from experiment_contract import validate_dataset


def crps(samples, truth):
    if not samples:raise ValueError('Empty predictive distribution')
    ordered=sorted(samples);n=len(ordered)
    if not math.isfinite(truth) or not all(math.isfinite(x) for x in ordered):raise ValueError('Nonfinite CRPS input')
    # Exact empirical CRPS without a quadratic pairwise-distance allocation.
    return math.fsum(abs(x-truth) for x in ordered)/n-math.fsum((2*i-n+1)*x for i,x in enumerate(ordered))/(n*n)



def summarize(dataset, predictions, output, minimum=5):
    from evaluate_random_pairs import evaluate
    output=Path(output)
    if minimum<2:raise ValueError('At least two readings required')
    # Mandatory full input/protocol/row validation, retained alongside the diagnostic.
    evaluate(dataset,predictions,output)
    manifest,truths=validate_dataset(dataset)
    entries={e['id']:e for e in manifest['examples']}
    files=sorted(Path(predictions).glob('*/predictions.csv'))
    groups={}
    for file in files:
        for row in csv.DictReader(file.open()):
            if row['experiment']=='verbose':groups.setdefault(row['example_id'],[]).append(row)
    rows=[]
    for eid,e in entries.items():
        pairs=[]
        for row in groups.get(eid,[]):
            try:a,b=float(row['predicted_a']),float(row['predicted_b'])
            except (ValueError,TypeError):continue
            if math.isfinite(a) and math.isfinite(b):pairs.append((a,b))
        row=dict(example_id=eid,source_group=e['source_group'],condition=e.get('condition','unspecified'),
            attempted_readings=len(groups.get(eid,[])),valid_readings=len(pairs),status='scored' if len(pairs)>=minimum else 'insufficient',
            intact_crps=None,independent_crps=None,delta=None,oracle_joint_crps=None,oracle_independent_crps=None,reference_expected_intact_crps=None,reference_expected_independent_crps=None)
        if len(pairs)>=minimum:
            scale=e.get('display_scale')
            if scale is None or scale<=0:raise ValueError('Frozen positive display_scale required')
            truth=float(truths[eid]['true_difference_a_minus_b'])
            joint=[a-b for a,b in pairs];independent=[a-b for a,_ in pairs for _,b in pairs]
            row.update(intact_crps=crps(joint,truth)/scale,independent_crps=crps(independent,truth)/scale)
            row['delta']=row['intact_crps']-row['independent_crps']
            ref=json.loads((Path(dataset)/e['reference']).read_text())
            feasible=ref.get('feasible_readings')
            if feasible:
                probabilities=ref['probabilities']
                if any(abs(p-1/len(feasible))>1e-12 for p in probabilities):raise ValueError('This oracle scorer supports equiprobable readings only')
                row['oracle_joint_crps']=crps([a-b for a,b in feasible],truth)/scale
                oracle_joint=[a-b for a,b in feasible]
                row['oracle_independent_crps']=crps([a-b for a,_ in feasible for _,b in feasible],truth)/scale
                row['reference_expected_intact_crps']=statistics.mean(crps(joint,t) for t in oracle_joint)/scale
                row['reference_expected_independent_crps']=statistics.mean(crps(independent,t) for t in oracle_joint)/scale
        rows.append(row)
    with (output/'dependence_pairs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for condition in dict.fromkeys(r['condition'] for r in rows):
        source={}
        for row in rows:
            if row['condition']==condition and row['status']=='scored':source.setdefault(row['source_group'],[]).append(row['delta'])
        summary.append(dict(condition=condition,scored_sources=len(source),expected_pairs=sum(r['condition']==condition for r in rows),scored_pairs=sum(r['condition']==condition and r['status']=='scored' for r in rows),mean_source_delta=statistics.mean(statistics.mean(v) for v in source.values()) if source else None))
    (output/'dependence_summary.json').write_text(json.dumps(dict(minimum_readings=minimum,primary_query='A-B',summary=summary,
        notes=['Negative delta favors intact pairing.','Conditional on complete finite readings; failures remain in completion counts.','No calibrated coverage or significance claim. Related conditions share source groups.']),indent=2)+'\n')
    return rows


if __name__=='__main__':
    opts,rest=getopt.gnu_getopt(sys.argv[1:],'',['dataset=','predictions=','output=','minimum='])
    a=dict(opts)
    if rest:raise ValueError('Unexpected arguments')
    summarize(a['--dataset'],a['--predictions'],a['--output'],int(a.get('--minimum',5)))
