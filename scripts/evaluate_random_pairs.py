#!/usr/bin/env python3
"""Validated descriptive and matched-pair scores; no pair-independence p-values."""
import csv
import getopt
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from experiment_contract import validate_dataset, protocol, inside
from run_vlm import parse_reading, parse_direct


def number(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (ValueError,TypeError): return None


def boolean(value):
    return {'true':1.,'false':0.,'1':1.,'0':0.}.get(str(value).strip().lower())


def write(path,rows,fields):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def binomial_tail(k,n,p):
    # Stable exact binomial tail, with no SciPy dependency.
    if k == 0: return 1.
    logs=[math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)+j*math.log(p)+(n-j)*math.log1p(-p)
          for j in range(k,n+1)]
    top=max(logs)
    return min(1., math.exp(top)*sum(math.exp(v-top) for v in logs))


def holm(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);out=[None]*len(values);previous=0.
    for rank,i in enumerate(order):
        previous=max(previous,min(1.,(len(values)-rank)*values[i]));out[i]=previous
    return out


def evaluate(dataset,predictions,output,null_accuracy=.5,test_trial=0,map_id=None,legacy=False):
    dataset,predictions,output=Path(dataset),Path(predictions),Path(output)
    if output.exists(): raise ValueError('Output exists; use a new directory.')
    if not 0<null_accuracy<1 or test_trial<0: raise ValueError('Null accuracy must be between 0 and 1; test trial must be nonnegative.')
    manifest,truths=validate_dataset(dataset,legacy=legacy)
    if map_id is not None:
        truths={k:v for k,v in truths.items() if v['map_id']==map_id}
        if not truths: raise ValueError('Unknown map ID')
    files=sorted(predictions.glob('*/predictions.csv')) if predictions.is_dir() else [predictions]
    if not files:raise ValueError('No predictions found.')
    records=[];protocols=[];expected_experiments=set();quality=[]
    for file in files:
        meta=file.parent/'run.json'
        if not meta.exists():raise ValueError('Missing run provenance: '+str(meta))
        if meta.exists():
            m=json.loads(meta.read_text())
            if m.get('manifest_sha256')!=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest():
                raise ValueError('Dataset manifest mismatch')
            if not legacy and (not m.get('code_sha256') or m.get('protocol_sha256')!=protocol(m)):
                raise ValueError('Missing or invalid protocol fingerprint')
            if not legacy and any(not m.get(k) for k in ('model','resolved_revision','prompts','packages','code_sha256')):
                raise ValueError('Incomplete model/prompt/environment provenance')
            mode=m['options']['experiment']
            if mode not in ('both','verbose','non-verbose') or m['options']['repeats']<1 or test_trial>=m['options']['repeats']:
                raise ValueError('Invalid run mode/repeats or selected test trial')
            if m['map_id'] not in {t['map_id'] for t in truths.values()}:raise ValueError('Run map outside declared scope')
            if not legacy:
                requested=set(manifest.get('queries',['difference','ratio','ordering'])) if mode!='verbose' else set()
                if mode!='non-verbose':requested.add('verbose')
                if set(m['prompts'])!=requested or any(not isinstance(v,str) or not v for v in m['prompts'].values()):
                    raise ValueError('Prompts do not match declared experiment')
            expected_experiments.update(('verbose','non-verbose') if mode=='both' else (mode,))
            protocols.append(dict(fingerprint=protocol(m),resolved_revision=m.get('resolved_revision'),model=m.get('model'),
                settings={k:v for k,v in m['options'].items() if k not in ('map-index','output','dry-run','dataset','resume')}))
        with file.open() as f:
            reader=csv.DictReader(f)
            required={'example_id','experiment','trial','predicted_difference_a_minus_b','predicted_ratio_a_over_b','predicted_a_greater_b'}
            if not required.issubset(reader.fieldnames or []):raise ValueError(f'Required columns: {sorted(required)}')
            file_rows=list(reader)
            allowed={'verbose','non-verbose'} if mode=='both' else {mode}
            for row in file_rows:
                if row['experiment'] not in allowed or not 0<=int(row['trial'])<m['options']['repeats']:
                    raise ValueError('Row violates run methods/repeats')
                if row['example_id'] not in truths or truths[row['example_id']]['map_id']!=m['map_id']:
                    raise ValueError('Row violates run map scope')
            if m.get('format_version',0)>=2:
                raw_cache={}
                queries=manifest.get('queries',['difference','ratio','ordering'])
                for row in file_rows:
                    eid=row['example_id'];trial=int(row['trial'])
                    keys=['verbose'] if row['experiment']=='verbose' else queries
                    answers={}
                    for query in keys:
                        cachekey=(eid,query)
                        if cachekey not in raw_cache:
                            rawpath=inside(file.parent,f'raw/{eid}/{query}.jsonl')
                            if not rawpath.exists():raise ValueError('Missing raw response provenance')
                            samples=[json.loads(line) for line in rawpath.read_text().splitlines()]
                            if 'seed' in m['options']:
                                global_index=next(i for i,e in enumerate(manifest['examples']) if e['id']==eid)
                                base=m['options']['seed']+(global_index*4+('verbose','difference','ratio','ordering').index(query))*m['options']['repeats']
                                if any(s.get('seed')!=base+i for i,s in enumerate(samples)):raise ValueError('Raw sample seed mismatch')
                            if any(s['sample']!=i for i,s in enumerate(samples)) or len(samples)>m['options']['repeats']:
                                raise ValueError('Invalid raw sample identities')
                            raw_cache[cachekey]=samples
                        samples=raw_cache[cachekey]
                        if trial>=len(samples):raise ValueError('Prediction has no raw response')
                        raw=samples[trial]
                        value,error=parse_reading(raw['raw_response'],['A','B']) if query=='verbose' else parse_direct(raw['raw_response'],query)
                        valid=value is not None and error is None and all(v is not None for v in value.values()) and raw.get('complete') is True and not raw.get('hit_token_limit')
                        answers[query]=value if valid else None
                        compliant=error is None if query=='verbose' else (parse_direct(raw['raw_response'],query,allow_scalar=False)[1] is None)
                        quality.append(dict(example_id=eid,experiment=row['experiment'],query=query,trial=trial,format_compliant=compliant,usable=valid,truncated=bool(raw.get('hit_token_limit'))))
                    if row['experiment']=='verbose':
                        pair=answers['verbose'];a,b=(pair['A'],pair['B']) if pair else (None,None)
                        expected=dict(difference=a-b if pair else None,ratio=a/b if pair and abs(b)>=manifest['ratio_min_abs'] else None,ordering=a>b if pair else None)
                        for key,value in [('predicted_a',a),('predicted_b',b)]:
                            if number(row.get(key))!=value:raise ValueError('Prediction/raw extracted-value mismatch')
                    else:expected={q:answers[q]['answer'] if answers[q] else None for q in queries}
                    for query,suffix,convert in [('difference','difference_a_minus_b',number),('ratio','ratio_a_over_b',number),('ordering','a_greater_b',boolean)]:
                        if query not in queries:continue
                        value=expected[query]
                        if isinstance(value,float) and not math.isfinite(value):value=None
                        if convert(row['predicted_'+suffix])!=value:raise ValueError('Prediction/raw response mismatch')
            records.extend(file_rows)
    if protocols and any(p!=protocols[0] for p in protocols):raise ValueError('Cannot aggregate different inference protocols.')
    experiments=sorted({r['experiment'] for r in records} | expected_experiments)
    if not experiments or set(experiments)-{'verbose','non-verbose'}:raise ValueError('Unknown or empty experiment selection.')
    seen=set();rows=[];groups={}
    for r in records:
        trial=int(r['trial']);key=(r['example_id'],r['experiment'],trial)
        if trial<0 or key in seen:raise ValueError('Duplicate or negative trial ID.')
        seen.add(key)
        if r['example_id'] not in truths:raise ValueError('Unknown/out-of-scope example ID: '+r['example_id'])
        t=truths[r['example_id']]
        for query,suffix,convert in [('difference','difference_a_minus_b',number),('ratio','ratio_a_over_b',number),('ordering','a_greater_b',boolean)]:
            truth=convert(t['true_'+suffix]);prediction=convert(r['predicted_'+suffix])
            valid=truth is not None and prediction is not None
            error=abs(prediction-truth) if valid else None
            if error is not None and not math.isfinite(error):valid=False;error=None
            row=dict(example_id=r['example_id'],map_id=t['map_id'],experiment=r['experiment'],trial=trial,
                query=query,truth=truth,prediction=prediction,valid=valid,absolute_error=error)
            if query not in manifest.get('queries',['difference','ratio','ordering']):continue
            rows.append(row);groups.setdefault((r['example_id'],r['experiment'],query),[]).append(row)
    summaries=[]
    for experiment in experiments:
        for mid in dict.fromkeys(t['map_id'] for t in truths.values()):
            expected=[identifier for identifier,t in truths.items() if t['map_id']==mid]
            for query in manifest.get('queries',['difference','ratio','ordering']):
                errors=[];valid_trials=total_trials=0;test_rows=[]
                for identifier in expected:
                    trials=groups.get((identifier,experiment,query),[]);total_trials+=len(trials)
                    valid=[r for r in trials if r['valid']];valid_trials+=len(valid)
                    if valid:errors.append(statistics.mean(r['absolute_error'] for r in valid))
                    test_rows.extend(r for r in trials if r['trial']==test_trial)
                k=sum(r['valid'] and r['prediction']==r['truth'] for r in test_rows)
                complete=len(test_rows)==len(expected)
                ordering=query=='ordering'
                summaries.append(dict(map_id=mid,experiment=experiment,query=query,expected_pairs=len(expected),
                    scored_pairs=len(errors),valid_trials=valid_trials,invalid_trials=total_trials-valid_trials,
                    mean_absolute_error=statistics.mean(errors) if errors and not ordering else None,
                    test_trial=test_trial if ordering else None,test_pairs=len(test_rows) if ordering else None,
                    test_correct=k if ordering else None,accuracy=k/len(test_rows) if ordering and test_rows else None,
                    reference_true_count=sum(boolean(truths[i]['true_a_greater_b']) for i in expected) if ordering else None,
                    null_accuracy=null_accuracy if ordering else None,
                    binomial_p_value=None,
                    holm_p_value=None,test_status=('complete' if complete else 'incomplete') if ordering else 'not_applicable'))
    for experiment in experiments:
        family=[r for r in summaries if r['experiment']==experiment and r['query']=='ordering']
        if all(r['binomial_p_value'] is not None for r in family):
            for row,p in zip(family,holm([r['binomial_p_value'] for r in family])):row['holm_p_value']=p
    matched=[]
    for mid in dict.fromkeys(t['map_id'] for t in truths.values()):
        for query in [q for q in manifest.get('queries',['difference','ratio','ordering']) if q!='ordering']:
            pairs=[]
            for identifier,t in truths.items():
                if t['map_id']!=mid:continue
                left={r['trial']:r for r in groups.get((identifier,'verbose',query),[])}
                right={r['trial']:r for r in groups.get((identifier,'non-verbose',query),[])}
                shared=[(left[i]['absolute_error'],right[i]['absolute_error']) for i in left.keys() & right.keys() if left[i]['valid'] and right[i]['valid']]
                if shared:pairs.append((statistics.mean(a for a,b in shared),statistics.mean(b for a,b in shared)))
            matched.append(dict(map_id=mid,query=query,shared_valid_pairs=len(pairs),extract_mae=statistics.mean(a for a,b in pairs) if pairs else None,direct_mae=statistics.mean(b for a,b in pairs) if pairs else None,paired_delta=statistics.mean(a-b for a,b in pairs) if pairs else None))
    output.mkdir(parents=True)
    if quality:write(output/'response_quality.csv',quality,list(quality[0]))
    write(output/'matched_comparisons.csv',matched,list(matched[0]) if matched else ['map_id','query','shared_valid_pairs','extract_mae','direct_mae','paired_delta'])
    write(output/'trial_errors.csv',rows,['example_id','map_id','experiment','trial','query','truth','prediction','valid','absolute_error'])
    write(output/'per_map_metrics.csv',summaries,list(summaries[0]))
    (output/'evaluation.json').write_text(json.dumps(dict(null_accuracy=null_accuracy,test_trial=test_trial,
        legacy_unverified=legacy,protocol_fingerprints=protocols,
        ground_truth_sha256=hashlib.sha256((dataset/'ground_truth.csv').read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest(),
        prediction_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        notes=['Numerical MAE averages within each pair, then across valid pairs.',
               'No binomial inference: pairs share images/cells. One prespecified trial per pair for ordering. Invalid attempted answers count as incorrect.',
               'P-values are disabled because sampled pairs share figures and can share cells.',
               'Binomial inference assumes independent trials under the null; shared maps/cells can violate this.',
               'No numerical-error p-value and no significance boolean are produced.']),indent=2)+'\n')
    print(f'Saved metrics and errors to {output}')


if __name__=='__main__':
    try:
        opts,rest=getopt.gnu_getopt(sys.argv[1:],'',['dataset=','predictions=','output=','null-accuracy=','test-trial=','map-id=','legacy-unverified'])
        args=dict(opts)
        if rest or '--predictions' not in args or '--output' not in args:raise ValueError('Required: --predictions CSV_OR_RUN_ROOT --output NEW_DIR')
        evaluate(args.get('--dataset','experiments/controlled-v1'),args['--predictions'],args['--output'],
            float(args.get('--null-accuracy',.5)),int(args.get('--test-trial',0)),args.get('--map-id'),'--legacy-unverified' in args)
    except (ValueError,OSError,KeyError,getopt.GetoptError) as exc:sys.exit(f'Error: {exc}')
