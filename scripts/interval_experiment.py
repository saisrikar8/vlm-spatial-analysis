#!/usr/bin/env python3
"""Direct feasible-bound questions: no sample pairing, recombination, or permutation."""
import csv
from datetime import datetime, timezone
import getopt
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
from experiment_contract import validate_dataset, sha, inside, protocol
from run_vlm import MODELS, parse_reading, write_json
from run_pilot import REVISION

QUERIES = ('a', 'b', 'difference')
TOLERANCE = 1e-6
GENERATION = dict(do_sample=False, num_beams=1, repetition_penalty=1., max_new_tokens=128, use_cache=True)


def prompts():
    prefix=('Image 1 is the clean figure with its legend and rules. Image 2 is the same figure with A/B pointer marks. '
            'A and B are at the centers of the gaps in those marks. Use the colors, legend, and all written rules. ')
    suffix=(' Give the tightest closed numerical interval containing EVERY value consistent with the figure and rules. '
            'This is a feasible range, not a confidence interval. If the answer is uniquely determined, lower and upper must be equal. '
            'Return only {"lower": number, "upper": number}. If unreadable, return {"lower": null, "upper": null}. Do not explain.')
    return {q:prefix+question+suffix for q,question in dict(a='What is the true value at A?',
        b='What is the true value at B?',difference='What is the true value at A minus the true value at B?').items()}


def parse_interval(raw):
    value,error=parse_reading(raw,['lower','upper'])
    if error:return None,error
    lo,hi=value['lower'],value['upper']
    if lo is None or hi is None:
        return (None,'unreadable') if lo is None and hi is None else (None,'Both endpoints must be numbers or both null')
    if lo>hi:return None,'Reversed endpoints'
    return value,None


def bounds(reference, query):
    readings=reference['feasible_readings']
    if not readings:raise ValueError('Missing feasible readings')
    values=[a if query=='a' else b if query=='b' else a-b for a,b in readings]
    return min(values),max(values)


def score(raw, reference, query):
    lo,hi=bounds(reference,query)
    truth=reference['ground_truth'][{'a':'true_a','b':'true_b','difference':'true_difference_a_minus_b'}[query]]
    value,error=parse_interval(raw['raw_response'])
    valid=value is not None and not raw.get('hit_token_limit',False)
    row=dict(query=query,reference_lower=lo,reference_upper=hi,reference_width=hi-lo,
        predicted_lower=None,predicted_upper=None,predicted_width=None,valid=valid,
        error='truncated' if raw.get('hit_token_limit',False) else error or '',
        contains_truth=False,contains_feasible_range=False,exact_bounds=False,
        endpoint_mae=None,width_absolute_error=None)
    if valid:
        lower,upper=value['lower'],value['upper']
        row.update(predicted_lower=lower,predicted_upper=upper,predicted_width=upper-lower,
            contains_truth=lower-TOLERANCE<=truth<=upper+TOLERANCE,
            contains_feasible_range=lower<=lo+TOLERANCE and upper>=hi-TOLERANCE,
            exact_bounds=abs(lower-lo)<=TOLERANCE and abs(upper-hi)<=TOLERANCE,
            endpoint_mae=(abs(lower-lo)+abs(upper-hi))/2,width_absolute_error=abs((upper-lower)-(hi-lo)))
        if not all(math.isfinite(row[k]) for k in ('predicted_width','endpoint_mae','width_absolute_error')):
            row.update(valid=False,error='arithmetic overflow',contains_truth=False,contains_feasible_range=False,exact_bounds=False,
                       predicted_width=None,endpoint_mae=None,width_absolute_error=None)
    return row


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def evaluate(dataset,runs,output):
    dataset,runs,output=Path(dataset),Path(runs),Path(output)
    if output.exists():raise ValueError('Evaluation output exists')
    manifest,_=validate_dataset(dataset)
    if manifest.get('experiment')!='controlled_offset_diagnostic':raise ValueError('Expected controlled source figures')
    entries={e['id']:e for e in manifest['examples']}
    records={};fingerprints=set();raw_hashes={}
    for folder in sorted(runs.iterdir()):
        if not folder.is_dir():continue
        meta_path=folder/'run.json'
        if not meta_path.exists():raise ValueError('Missing run metadata')
        meta=json.loads(meta_path.read_text())
        if meta['source_manifest_sha256']!=sha(dataset/'manifest.json') or meta['prompts']!=prompts() or meta['generation']!=GENERATION:
            raise ValueError('Dataset/prompt/generation mismatch')
        if meta.get('protocol_sha256')!=protocol(meta):raise ValueError('Protocol fingerprint mismatch')
        if meta['experiment']!='direct_feasible_intervals_v1' or meta['queries']!=list(QUERIES):raise ValueError('Wrong interval protocol')
        fingerprints.add(meta['protocol_sha256'])
        eid=meta['example_id']
        if eid not in entries or entries[eid]['map_id']!=folder.name or eid in records:raise ValueError('Wrong or duplicate source identity')
        path=inside(folder,'responses.jsonl')
        raw_hashes[str(path)]=sha(path)
        raw=[json.loads(line) for line in path.read_text().splitlines()]
        queries=[r['query'] for r in raw]
        if len(queries)!=len(set(queries)) or set(queries)-set(QUERIES):raise ValueError('Duplicate or unknown query')
        global_index=list(entries).index(eid)
        for r in raw:
            if r['seed']!=meta['options']['seed']+global_index*3+QUERIES.index(r['query']):raise ValueError('Seed identity mismatch')
        if meta['status']=='finished' and set(queries)!=set(QUERIES):raise ValueError('Finished run missing questions')
        records[eid]={r['query']:r for r in raw}
    if not records:raise ValueError('No responses')
    if len(fingerprints)!=1:raise ValueError('Mixed inference protocols')
    rows=[];patterns=[]
    for eid,e in entries.items():
        reference=json.loads(inside(dataset,e['reference']).read_text());example_rows=[]
        for q in QUERIES:
            raw=records.get(eid,{}).get(q)
            row=score(raw or dict(raw_response='{}'),reference,q)
            if raw is None:row['error']='missing'
            row=dict(example_id=eid,source_group=e['source_group'],condition=e['condition'],attempted=raw is not None,**row)
            rows.append(row);example_rows.append(row)
        patterns.append(dict(example_id=eid,source_group=e['source_group'],condition=e['condition'],
            all_attempted=all(r['attempted'] for r in example_rows),all_valid=all(r['valid'] for r in example_rows),
            all_bounds_exact=all(r['exact_bounds'] for r in example_rows),
            all_feasible_ranges_contained=all(r['contains_feasible_range'] for r in example_rows)))
    summaries=[]
    for condition in dict.fromkeys(e['condition'] for e in entries.values()):
        for q in QUERIES:
            subset=[r for r in rows if r['condition']==condition and r['query']==q]
            # Each source currently contributes one example per condition. Preserve source weighting if extended.
            sources={r['source_group'] for r in subset}
            def mean(key,valid_only=False):
                means=[]
                for source in sources:
                    values=[r[key] for r in subset if r['source_group']==source and (not valid_only or r['valid'])]
                    if values:means.append(statistics.mean(values))
                return statistics.mean(means) if means else None
            summaries.append(dict(condition=condition,query=q,expected=len(subset),attempted=sum(r['attempted'] for r in subset),
                valid=sum(r['valid'] for r in subset),sources=len(sources),reference_width=mean('reference_width'),
                predicted_width_valid_only=mean('predicted_width',True),endpoint_mae_valid_only=mean('endpoint_mae',True),
                feasible_range_containment=mean('contains_feasible_range'),exact_bounds_rate=mean('exact_bounds'),
                truth_containment=mean('contains_truth')))
    output.mkdir(parents=True)
    write_csv(output/'interval_answers.csv',rows);write_csv(output/'interval_summary.csv',summaries);write_csv(output/'source_patterns.csv',patterns)
    write_json(output/'evaluation.json',dict(experiment='direct_feasible_intervals_v1',expected_answers=len(rows),
        attempted_answers=sum(r['attempted'] for r in rows),valid_answers=sum(r['valid'] for r in rows),
        all_exact_examples=sum(r['all_bounds_exact'] for r in patterns),source_manifest_sha256=sha(dataset/'manifest.json'),
        protocol_fingerprints=sorted(fingerprints),raw_sha256=raw_hashes,tolerance=TOLERANCE,
        notes=['No permutation, recombination, or repeated-sample comparison.',
               'Reference is the tightest closed interval containing all specified feasible readings; gaps inside the interval are allowed.',
               'Missing/invalid answers count as unsuccessful in exactness/containment rates. Width/error means condition on valid answers.',
               'One deterministic answer per query in fresh conversations. Six source groups; related conditions are paired.',
               'Feasible bounds are not calibrated confidence intervals. Prior experiment informed protocol choice: development analysis only.']))
    return summaries


def run(dataset,index,output,seed=4242,dry_run=False):
    dataset=Path(dataset).resolve();manifest,_=validate_dataset(dataset)
    if manifest.get('experiment')!='controlled_offset_diagnostic':raise ValueError('Expected controlled source figures')
    if not 0<=index<len(manifest['examples']) or seed<0:raise ValueError('Invalid index/seed')
    e=manifest['examples'][index];folder=Path(output)/e['map_id']
    if folder.exists():raise ValueError('Output exists; choose a new run')
    options=dict(dataset=str(dataset),**{'map-index':index},output=str(output),seed=seed,model='qwen7b',revision=REVISION,dtype='bfloat16',max_pixels=1003520)
    meta=dict(experiment='direct_feasible_intervals_v1',status='planned',example_id=e['id'],queries=list(QUERIES),
        source_manifest_sha256=sha(dataset/'manifest.json'),options=options,prompts=prompts(),generation=GENERATION,
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in ('interval_experiment.py','run_vlm.py','experiment_contract.py')},
        model=MODELS['qwen7b'],calls=3)
    if dry_run:print(json.dumps(meta,indent=2));return
    import torch
    from PIL import Image
    from huggingface_hub import model_info
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('CUDA/bfloat16 GPU required')
    revision=model_info(MODELS['qwen7b'],revision=REVISION).sha
    meta.update(status='loading',resolved_revision=revision,started_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,slurm_job_id=os.getenv('SLURM_JOB_ID'),
        packages={name:importlib.metadata.version(name) for name in ('torch','transformers','pillow','accelerate')})
    folder.mkdir(parents=True);write_json(folder/'run.json',meta)
    try:
        processor=AutoProcessor.from_pretrained(MODELS['qwen7b'],revision=revision,min_pixels=3136,max_pixels=1003520)
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODELS['qwen7b'],revision=revision,
            torch_dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
        meta.update(status='running',model_generation_config=model.generation_config.to_dict())
        meta['protocol_sha256']=protocol(meta);write_json(folder/'run.json',meta)
        images=[]
        for key in ('clean_image','image'):
            with Image.open(inside(dataset,e[key])) as im:images.append(im.convert('RGB'))
        with (folder/'responses.jsonl').open('x') as stream:
            for q in QUERIES:
                trial_seed=seed+index*3+QUERIES.index(q);set_seed(trial_seed)
                messages=[{'role':'user','content':[{'type':'image'},{'type':'image'},{'type':'text','text':prompts()[q]}]}]
                rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                inputs=processor(text=[rendered],images=images,padding=True,return_tensors='pt').to(model.device)
                with torch.inference_mode():tokens=model.generate(**inputs,**GENERATION)
                generated=tokens[:,inputs['input_ids'].shape[1]:]
                raw=processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                hit=generated.shape[1]>=GENERATION['max_new_tokens'] and int(generated[0,-1]) not in eos
                stream.write(json.dumps(dict(query=q,seed=trial_seed,raw_response=raw,hit_token_limit=hit,
                    output_tokens=generated.shape[1],rendered_prompt=rendered),allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                print(q,raw,flush=True)
        meta.update(status='finished',finished_utc=datetime.now(timezone.utc).isoformat());write_json(folder/'run.json',meta)
    except BaseException as error:
        meta.update(status='failed',error=str(error));write_json(folder/'run.json',meta);raise


if __name__=='__main__':
    try:
        opts,rest=getopt.gnu_getopt(sys.argv[1:],'',['dataset=','index=','output=','seed=','dry-run','evaluate='])
        a=dict(opts)
        if rest or '--output' not in a:raise ValueError('Required: --output NEW_DIR; --index N or --evaluate RUN_ROOT')
        dataset=a.get('--dataset','experiments/controlled-v1')
        if '--evaluate' in a:evaluate(dataset,a['--evaluate'],a['--output'])
        else:run(dataset,int(a['--index']),a['--output'],int(a.get('--seed',4242)),'--dry-run' in a)
    except (ValueError,OSError,KeyError,ImportError,getopt.GetoptError) as error:sys.exit('Error: '+str(error))
