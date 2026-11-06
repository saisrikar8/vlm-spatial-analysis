#!/usr/bin/env python3
"""Separate displayed-color reading from text-only feasible-interval reasoning."""
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
from experiment_contract import sha, inside, safe_id, protocol, validate_dataset
from interval_experiment import QUERIES, GENERATION, score, write_csv
from run_vlm import MODELS, parse_direct, write_json
from run_pilot import REVISION

CONDITIONS=('clear','shared_offset','independent_offset')
TASKS=('color_a','color_b')+tuple(c+'_'+q for c in CONDITIONS for q in QUERIES)


def prepare(source,output):
    from PIL import Image
    source,output=Path(source),Path(output)
    if output.exists():raise ValueError('Dataset output exists')
    manifest,_=validate_dataset(source)
    groups={}
    for e in manifest['examples']:groups.setdefault(e['source_group'],{})[e['condition']]=e
    output.mkdir(parents=True);(output/'images').mkdir();(output/'references').mkdir()
    entries=[]
    for i,(group,conditions) in enumerate(groups.items()):
        if set(conditions)!=set(CONDITIONS):raise ValueError('Source lacks a condition')
        refs={c:json.loads(inside(source,e['reference']).read_text()) for c,e in conditions.items()}
        shared=refs['shared_offset']['feasible_readings'];a=min(x for x,y in shared)-10;b=min(y for x,y in shared)-10
        if max(x for x,y in shared)-20!=a or max(y for x,y in shared)-20!=b:raise ValueError('Unexpected shared-offset construction')
        independent=refs['independent_offset']['feasible_readings']
        if min(x for x,y in independent)-10!=a or min(y for x,y in independent)-10!=b:raise ValueError('Mismatched displayed values across conditions')
        identifier=f'source-{i:03d}'
        e=conditions['clear'];paths={}
        for key in ('clean_image','image'):
            relative=f'images/{identifier}-{key}.png'
            with Image.open(inside(source,e[key])) as image:
                if image.size!=(750,475):raise ValueError('Unexpected source layout')
                # Retain tiles, labels and legend; remove all offset instructions at y>=395.
                image.crop((0,0,750,385)).save(output/relative)
            paths[key]=relative
        with Image.open(output/paths['clean_image']) as image:
            for point,value in zip(refs['clear']['points'],(a,b)):
                x,y=point['pixel'];rgb=image.getpixel((x,y))
                if rgb!=(int(255*value/8),80,int(255*(8-value)/8)):raise ValueError('Displayed pixel/reference mismatch')
        refpath=f'references/{identifier}.json'
        write_json(output/refpath,dict(displayed_a=a,displayed_b=b,interval_references=refs,
            source_examples={c:conditions[c]['id'] for c in CONDITIONS}))
        entries.append(dict(id=identifier,source_group=group,reference=refpath,reference_sha256=sha(output/refpath),
            **paths,clean_image_sha256=sha(output/paths['clean_image']),image_sha256=sha(output/paths['image'])))
    m=dict(experiment='component_diagnostics_v1',example_count=len(entries),tasks=list(TASKS),source_manifest_sha256=sha(source/'manifest.json'),
        builder_sha256=sha(Path(__file__)),selection='Same six development sources; color figures cropped above offset rules; exact displayed values supplied only to text tasks.',examples=entries)
    write_json(output/'manifest.json',m)
    return m


def load_dataset(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('experiment')!='component_diagnostics_v1' or manifest.get('tasks')!=list(TASKS):raise ValueError('Wrong diagnostic protocol')
    entries=manifest['examples']
    if manifest['example_count']!=len(entries) or len({e['id'] for e in entries})!=len(entries):raise ValueError('Invalid source membership')
    if len({e['source_group'] for e in entries})!=len(entries):raise ValueError('Each diagnostic task must represent a distinct source group')
    for e in entries:
        safe_id(e['id'])
        for key in ('reference','clean_image','image'):
            if sha(inside(root,e[key]))!=e[key+'_sha256']:raise ValueError('Diagnostic input checksum mismatch')
    return manifest


def prompts(ref):
    out={}
    for label in ('a','b'):
        out['color_'+label]=('Image 1 is the clean figure and its color legend. Image 2 shows the same figure with A/B pointer marks. '
            f'Read the displayed color-coded number v at {label.upper()}, at the center of the gap in its pointer mark, using the legend. '
            'Return only {"answer": number}, or {"answer": null} if unreadable. Return the displayed legend number only. Do not explain.')
    a,b=ref['displayed_a'],ref['displayed_b'];clear=ref['interval_references']['clear']['ground_truth']
    rules=dict(clear=f'The offsets are known: b_A = {clear["true_a"]-a} and b_B = {clear["true_b"]-b}.',
        shared_offset='There is one shared offset: b_A = b_B = b, where b can be 10 or 20.',
        independent_offset='The offsets are separate: b_A can be 10 or 20 and b_B can independently be 10 or 20. Every combination is possible.')
    for c in CONDITIONS:
        prefix=f'The displayed numerical inputs are v_A = {a} and v_B = {b}. The true values are A = v_A + b_A and B = v_B + b_B. '+rules[c]+' '
        for q,question in dict(a='What is the true value A?',b='What is the true value B?',difference='What is A minus B?').items():
            out[c+'_'+q]=prefix+question+(' Give the tightest closed numerical interval containing EVERY value consistent with these inputs and rules. '
                'This is a feasible range, not a confidence interval. If the answer is uniquely determined, lower and upper must be equal. '
                'Return only {"lower": number, "upper": number}. Do not explain.')
    return out


def task_scores(raw,ref):
    task=raw['task']
    if task.startswith('color_'):
        label=task[-1];value,error=parse_direct(raw['raw_response'],'difference')
        pred=value['answer'] if value else None;valid=pred is not None and not raw.get('hit_token_limit')
        truth=ref['displayed_'+label]
        return dict(diagnostic='color',condition='color_only',query=label,valid=bool(valid),error=error or ('null' if pred is None else ''),
            reference_value=truth,predicted_value=pred,absolute_error=abs(pred-truth) if valid else None,exact=bool(valid and abs(pred-truth)<=1e-6))
    c,q=next((c,task[len(c)+1:]) for c in CONDITIONS if task.startswith(c+'_'))
    row=score(raw,ref['interval_references'][c],q)
    return dict(diagnostic='text_intervals',condition=c,**row)


def fingerprint(meta):
    return protocol(dict(meta,prompts={'template':'component_diagnostics_v1'},model_generation_config=dict(meta.get('model_generation_config',{}),diagnostic_generation=GENERATION)))


def evaluate(dataset,runs,output,baseline=None):
    dataset,runs,output=Path(dataset),Path(runs),Path(output)
    if output.exists():raise ValueError('Evaluation output exists')
    manifest=load_dataset(dataset);entries={e['id']:e for e in manifest['examples']};records={};fingerprints=set();hashes={}
    for folder in sorted(runs.iterdir()):
        if not folder.is_dir():continue
        meta=json.loads((folder/'run.json').read_text());eid=meta['example_id']
        if eid not in entries or folder.name!=eid or eid in records:raise ValueError('Unexpected or duplicate source')
        ref=json.loads(inside(dataset,entries[eid]['reference']).read_text())
        if meta['dataset_sha256']!=sha(dataset/'manifest.json') or meta['prompts']!=prompts(ref) or meta['generation']!=GENERATION:raise ValueError('Diagnostic input/prompt/generation mismatch')
        if meta['experiment']!='component_diagnostics_v1' or meta['tasks']!=list(TASKS):raise ValueError('Wrong diagnostic')
        if meta.get('protocol_sha256')!=fingerprint(meta):raise ValueError('Protocol fingerprint mismatch')
        fingerprints.add(meta['protocol_sha256'])
        path=inside(folder,'responses.jsonl');hashes[str(path)]=sha(path)
        raw=[json.loads(line) for line in path.read_text().splitlines()]
        tasks=[r['task'] for r in raw]
        if len(set(tasks))!=len(tasks) or set(tasks)-set(TASKS):raise ValueError('Duplicate or unknown task')
        index=list(entries).index(eid)
        for r in raw:
            if r.get('input_modality')!=('image' if r['task'].startswith('color_') else 'text'):raise ValueError('Wrong input modality')
            if r['seed']!=meta['options']['seed']+index*len(TASKS)+TASKS.index(r['task']):raise ValueError('Seed mismatch')
        if meta['status']=='finished' and set(tasks)!=set(TASKS):raise ValueError('Finished run missing tasks')
        records[eid]={r['task']:r for r in raw}
    if not records or len(fingerprints)!=1:raise ValueError('No runs or mixed protocols')
    color=[];text=[]
    for eid,e in entries.items():
        ref=json.loads(inside(dataset,e['reference']).read_text())
        for task in TASKS:
            raw=records.get(eid,{}).get(task)
            row=task_scores(raw or dict(task=task,raw_response='{}'),ref)
            row=dict(example_id=eid,source_group=e['source_group'],attempted=raw is not None,**row)
            if raw is None:row['error']='missing'
            (color if row['diagnostic']=='color' else text).append(row)
    summaries=[]
    for diagnostic,rows,groups,error_key in [('color',color,[('color_only',q) for q in ('a','b')],'absolute_error'),
        ('text_intervals',text,[(c,q) for c in CONDITIONS for q in QUERIES],'endpoint_mae')]:
        for condition,q in groups:
            subset=[r for r in rows if r['condition']==condition and r['query']==q];valid=[r for r in subset if r['valid']]
            summaries.append(dict(diagnostic=diagnostic,condition=condition,query=q,expected=len(subset),attempted=sum(r['attempted'] for r in subset),
                valid=len(valid),mean_numerical_error=statistics.mean(r[error_key] for r in valid) if valid else None,
                exact_count=sum(r['exact'] if diagnostic=='color' else r['exact_bounds'] for r in subset),
                exact_rate=statistics.mean(r['exact'] if diagnostic=='color' else r['exact_bounds'] for r in subset),
                reference_width=statistics.mean(r['reference_width'] for r in subset) if diagnostic!='color' else None,
                mean_predicted_width=statistics.mean(r['predicted_width'] for r in valid) if valid and diagnostic!='color' else None))
    comparisons=[]
    if baseline:
        with Path(baseline).open() as stream:previous=list(csv.DictReader(stream))
        expected_ids={refid for e in entries.values() for refid in json.loads(inside(dataset,e['reference']).read_text())['source_examples'].values()}
        keys=[(r['example_id'],r['query']) for r in previous]
        if len(keys)!=len(set(keys)) or set(keys)!={(eid,q) for eid in expected_ids for q in QUERIES}:raise ValueError('Baseline scope mismatch')
        previous={(r['example_id'],r['query']):r for r in previous}
        for row in text:
            ref=json.loads(inside(dataset,entries[row['example_id']]['reference']).read_text())
            old=previous[(ref['source_examples'][row['condition']],row['query'])]
            if float(old['reference_lower'])!=row['reference_lower'] or float(old['reference_upper'])!=row['reference_upper']:raise ValueError('Baseline reference mismatch')
            shared=row['valid'] and old['valid']=='True'
            comparisons.append(dict(source_group=row['source_group'],condition=row['condition'],query=row['query'],both_valid=shared,
                original_image_error=float(old['endpoint_mae']) if shared else None,text_error=row['endpoint_mae'] if shared else None,
                text_minus_image_error=row['endpoint_mae']-float(old['endpoint_mae']) if shared else None))
    output.mkdir(parents=True);write_csv(output/'color_answers.csv',color);write_csv(output/'text_answers.csv',text);write_csv(output/'diagnostic_summary.csv',summaries)
    if comparisons:write_csv(output/'matched_image_text.csv',comparisons)
    write_json(output/'evaluation.json',dict(experiment='component_diagnostics_v1',expected_answers=len(entries)*len(TASKS),
        attempted=sum(r['attempted'] for r in color+text),valid=sum(r['valid'] for r in color+text),
        dataset_sha256=sha(dataset/'manifest.json'),protocol_fingerprints=sorted(fingerprints),raw_sha256=hashes,
        baseline_sha256=sha(baseline) if baseline else None,
        notes=['Six development sources; one answer per task, greedy decoding. No prompt selection or repeated outputs.',
               'Color images retain legend and markers but omit all offset rules. Text interval tasks receive exact displayed values and rules, with no images.',
               'Missing/invalid answers count as unsuccessful in exactness rates; mean errors condition on valid answers.',
               'Baseline is the previous completed image interval run on identical source examples. Changing modality also changes input presentation; this is diagnostic, not a causal proof.',
               'No permutation, recombination, calibration, or significance claim.']))
    return summaries


def run(dataset,index,output,seed=4343,dry_run=False):
    dataset=Path(dataset).resolve();manifest=load_dataset(dataset)
    if not 0<=index<len(manifest['examples']) or seed<0:raise ValueError('Invalid index/seed')
    e=manifest['examples'][index];ref=json.loads(inside(dataset,e['reference']).read_text());tasks=prompts(ref);folder=Path(output)/e['id']
    if folder.exists():raise ValueError('Output exists; choose a new run')
    options=dict(dataset=str(dataset),**{'map-index':index},output=str(output),seed=seed,model='qwen7b',revision=REVISION,dtype='bfloat16',max_pixels=1003520)
    meta=dict(experiment='component_diagnostics_v1',status='planned',example_id=e['id'],tasks=list(TASKS),dataset_sha256=sha(dataset/'manifest.json'),
        prompts=tasks,generation=GENERATION,options=options,model=MODELS['qwen7b'],calls=len(TASKS),
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in ('component_diagnostics.py','interval_experiment.py','experiment_contract.py','run_vlm.py')})
    if dry_run:print(json.dumps(meta,indent=2));return
    import torch
    from PIL import Image
    from huggingface_hub import model_info
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('CUDA/bfloat16 required')
    revision=model_info(MODELS['qwen7b'],revision=REVISION).sha
    meta.update(status='loading',resolved_revision=revision,python=sys.version,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
        started_utc=datetime.now(timezone.utc).isoformat(),slurm_job_id=os.getenv('SLURM_JOB_ID'),
        packages={name:importlib.metadata.version(name) for name in ('torch','transformers','pillow','accelerate')})
    folder.mkdir(parents=True);write_json(folder/'run.json',meta)
    try:
        processor=AutoProcessor.from_pretrained(MODELS['qwen7b'],revision=revision,min_pixels=3136,max_pixels=1003520)
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODELS['qwen7b'],revision=revision,torch_dtype=torch.bfloat16,
            device_map={'':'cuda'},attn_implementation='sdpa').eval()
        meta.update(status='running',model_generation_config=model.generation_config.to_dict());meta['protocol_sha256']=fingerprint(meta);write_json(folder/'run.json',meta)
        images=[]
        for key in ('clean_image','image'):
            with Image.open(inside(dataset,e[key])) as im:images.append(im.convert('RGB'))
        with (folder/'responses.jsonl').open('x') as stream:
            for task in TASKS:
                trial_seed=seed+index*len(TASKS)+TASKS.index(task);set_seed(trial_seed)
                visual=task.startswith('color_')
                content=([{'type':'image'},{'type':'image'}] if visual else [])+[{'type':'text','text':tasks[task]}]
                rendered=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
                kwargs=dict(text=[rendered],padding=True,return_tensors='pt')
                if visual:kwargs['images']=images
                inputs=processor(**kwargs).to(model.device)
                with torch.inference_mode():tokens=model.generate(**inputs,**GENERATION)
                generated=tokens[:,inputs['input_ids'].shape[1]:];raw=processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                hit=generated.shape[1]>=GENERATION['max_new_tokens'] and int(generated[0,-1]) not in eos
                stream.write(json.dumps(dict(task=task,seed=trial_seed,raw_response=raw,hit_token_limit=hit,output_tokens=generated.shape[1],
                    input_modality='image' if visual else 'text',rendered_prompt=rendered),allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                print(task,raw,flush=True)
        meta.update(status='finished',finished_utc=datetime.now(timezone.utc).isoformat());write_json(folder/'run.json',meta)
    except BaseException as error:
        meta.update(status='failed',error=str(error));write_json(folder/'run.json',meta);raise


if __name__=='__main__':
    try:
        opts,rest=getopt.gnu_getopt(sys.argv[1:],'',['prepare=','dataset=','index=','output=','seed=','dry-run','evaluate=','baseline='])
        a=dict(opts)
        if rest or '--output' not in a:raise ValueError('Required --output NEW_DIR')
        dataset=a.get('--dataset','experiments/component-diagnostics-v1')
        if '--prepare' in a:prepare(a['--prepare'],a['--output'])
        elif '--evaluate' in a:evaluate(dataset,a['--evaluate'],a['--output'],a.get('--baseline'))
        else:run(dataset,int(a['--index']),a['--output'],int(a.get('--seed',4343)),'--dry-run' in a)
    except (ValueError,OSError,KeyError,ImportError,getopt.GetoptError) as error:sys.exit('Error: '+str(error))
