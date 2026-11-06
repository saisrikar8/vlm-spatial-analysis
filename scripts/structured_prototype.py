#!/usr/bin/env python3
"""Prototype: extract values plus named offset domains, then execute fixed algebra."""
import getopt
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from datetime import datetime,timezone
from experiment_contract import validate_dataset,sha,inside,protocol
from interval_experiment import QUERIES,score,write_csv
from component_diagnostics import load_dataset,prompts as component_prompts
from run_vlm import MODELS,write_json
from run_pilot import REVISION

MODES=('image','exact_text')
CONDITIONS=('clear','shared_offset','independent_offset')
TASKS=tuple(c+'_'+m for c in CONDITIONS for m in MODES)
GENERATION=dict(do_sample=False,num_beams=1,repetition_penalty=1.,max_new_tokens=256,use_cache=True)
SCHEMA=('Extract one reusable numerical representation, without calculating any final answers. '
 'Return only a JSON object with exactly two keys: "values" and "offsets". '
 '"values" has exactly keys "A" and "B". Each has exactly "base" (the displayed numerical value) and "offset" (an offset variable name). '
 '"offsets" maps each used variable name to a list of every permitted numerical offset value. '
 'Reuse the SAME variable name when the rules specify one shared offset. Use DIFFERENT names for separate offsets. '
 'A known offset has a one-element list. Do not include unused variables. '
 'A value means base plus its named offset. Distinct named offsets vary independently. '
 'Example of the JSON shape, with placeholders to replace: '
 '{"values":{"A":{"base":number,"offset":"name1"},"B":{"base":number,"offset":"name2"}},'
 '"offsets":{"name1":[numbers],"name2":[numbers]}}. '
 'If a shared name is used, include it only once in offsets. If unreadable, return null. Do not explain.')


def parse_representation(raw):
    text=raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text=re.sub(r'^```(?:json)?\s*','',text,count=1)[:-3].strip()
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('Duplicate key')
            result[k]=v
        return result
    def number(v):return type(v) in (int,float) and math.isfinite(v)
    try:
        value=json.loads(text,object_pairs_hook=unique)
        if value is None:return None,'unreadable'
        if not isinstance(value,dict) or set(value)!={'values','offsets'}:raise ValueError('Expected values and offsets only')
        vals,offsets=value['values'],value['offsets']
        if not isinstance(vals,dict) or set(vals)!={'A','B'} or not isinstance(offsets,dict) or not 1<=len(offsets)<=2:raise ValueError('Invalid values/offset domains')
        for name,domain in offsets.items():
            if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}',name):raise ValueError('Invalid variable name')
            if not isinstance(domain,list) or not 1<=len(domain)<=8 or not all(number(v) for v in domain) or len(set(domain))!=len(domain):raise ValueError('Invalid finite domain')
        used=set()
        for label,entry in vals.items():
            if not isinstance(entry,dict) or set(entry)!={'base','offset'} or not number(entry['base']):raise ValueError('Invalid base')
            if not isinstance(entry['offset'],str) or entry['offset'] not in offsets:raise ValueError('Undefined offset')
            used.add(entry['offset'])
        if used!=set(offsets):raise ValueError('Unused variables')
        return value,None
    except (ValueError,TypeError,OverflowError) as error:return None,str(error)


def execute(rep):
    a,b=rep['values']['A'],rep['values']['B'];da=rep['offsets'][a['offset']];db=rep['offsets'][b['offset']]
    result=dict(a=(a['base']+min(da),a['base']+max(da)),b=(b['base']+min(db),b['base']+max(db)))
    delta=a['base']-b['base']
    result['difference']=(delta,delta) if a['offset']==b['offset'] else (delta+min(da)-max(db),delta+max(da)-min(db))
    if not all(math.isfinite(v) for pair in result.values() for v in pair):raise ValueError('Executor overflow')
    return result


def dataset(source,components):
    source,components=Path(source),Path(components);original,_=validate_dataset(source);comp=load_dataset(components)
    originals={(e['source_group'],e['condition']):e for e in original['examples']}
    if len(originals)!=len(original['examples']):raise ValueError('Duplicate source/condition')
    for e in comp['examples']:
        ref=json.loads(inside(components,e['reference']).read_text())
        for c in CONDITIONS:
            original_ref=json.loads(inside(source,originals[(e['source_group'],c)]['reference']).read_text())
            if original_ref!=ref['interval_references'][c]:raise ValueError('Source reference mismatch')
    if len(originals)!=len(comp['examples'])*3:raise ValueError('Unexpected source scope')
    return comp,originals


def prompts(ref):
    out={};given=component_prompts(ref)
    for c in CONDITIONS:
        out[c+'_image']=('Image 1 is the clean figure with its legend and numerical rules. Image 2 is the same figure with A/B pointer marks. '
            'A/B are at the centers of the gaps in those marks. Read the displayed colors and the written rules. '+SCHEMA)
        prefix=given[c+'_a'].split('What is the true value A?')[0]
        out[c+'_exact_text']=prefix+SCHEMA
    return out


def fingerprint(meta):
    return protocol(dict(meta,prompts={'schema':SCHEMA,'tasks':list(TASKS)},model_generation_config=dict(meta.get('model_generation_config',{}),prototype_generation=GENERATION)))


def score_representation(raw,ref,condition):
    rep,error=parse_representation(raw['raw_response']);valid=rep is not None and not raw.get('hit_token_limit')
    intervals={}
    if valid:
        try:intervals=execute(rep)
        except ValueError as exc:valid=False;error=str(exc)
    if raw.get('hit_token_limit'):error='truncated'
    reference=ref['interval_references'][condition]
    expected_domains={label:[reference['ground_truth']['true_'+label.lower()]-ref['displayed_'+label.lower()]] if condition=='clear' else [10,20] for label in ('A','B')}
    diagnostics=dict(valid=bool(valid),error=error or '',base_a=None,base_b=None,base_mae=None,domains_correct=False,shared_predicted=None,sharing_correct=None)
    if valid:
        a,b=rep['values']['A'],rep['values']['B']
        diagnostics.update(base_a=a['base'],base_b=b['base'],base_mae=abs(a['base']-ref['displayed_a'])/2+abs(b['base']-ref['displayed_b'])/2,
            domains_correct=all(sorted(rep['offsets'][rep['values'][label]['offset']])==sorted(expected_domains[label]) for label in ('A','B')),
            shared_predicted=a['offset']==b['offset'],sharing_correct=(a['offset']==b['offset'])==(condition=='shared_offset') if condition!='clear' else None)
    rows=[]
    for q in QUERIES:
        interval=intervals.get(q)
        answer=dict(raw_response=json.dumps(dict(lower=interval[0],upper=interval[1]))) if valid else dict(raw_response='{}')
        row=score(answer,reference,q)
        if not valid:row['error']=error or 'invalid representation'
        rows.append(row)
    return diagnostics,rows


def baseline_rows(path,entries,components):
    import csv
    with Path(path).open() as stream:rows=list(csv.DictReader(stream))
    expected=set()
    refs={}
    for e in entries:
        ref=json.loads(inside(components,e['reference']).read_text());refs[e['id']]=ref
        for c,eid in ref['source_examples'].items():
            for q in QUERIES:expected.add((eid,q))
    keys=[(r['example_id'],r['query']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Direct baseline membership mismatch')
    return {k:r for k,r in zip(keys,rows)}


def evaluate(source,components,runs,output,image_baseline=None,text_baseline=None):
    source,components,runs,output=map(Path,(source,components,runs,output))
    if output.exists():raise ValueError('Output exists')
    comp,_=dataset(source,components);entries={e['id']:e for e in comp['examples']};records={};protocols=set();hashes={}
    for folder in sorted(runs.iterdir()):
        if not folder.is_dir():continue
        meta=json.loads((folder/'run.json').read_text());eid=meta['example_id']
        if eid not in entries or eid!=folder.name or eid in records:raise ValueError('Unexpected/duplicate source')
        ref=json.loads(inside(components,entries[eid]['reference']).read_text())
        if meta['source_sha256']!=sha(source/'manifest.json') or meta['components_sha256']!=sha(components/'manifest.json'):raise ValueError('Dataset mismatch')
        if meta['prompts']!=prompts(ref) or meta['generation']!=GENERATION or meta.get('tasks')!=list(TASKS) or meta['experiment']!='structured_offsets_v1':raise ValueError('Prompt/protocol mismatch')
        if meta['protocol_sha256']!=fingerprint(meta):raise ValueError('Fingerprint mismatch')
        protocols.add(meta['protocol_sha256']);path=inside(folder,'responses.jsonl');hashes[str(path)]=sha(path)
        raw=[json.loads(line) for line in path.read_text().splitlines()];tasks=[r['task'] for r in raw]
        if len(set(tasks))!=len(tasks) or set(tasks)-set(TASKS):raise ValueError('Duplicate/unknown task')
        index=list(entries).index(eid)
        for r in raw:
            if r['seed']!=meta['options']['seed']+index*len(TASKS)+TASKS.index(r['task']):raise ValueError('Seed mismatch')
            if r['input_modality']!=('image' if r['task'].endswith('_image') else 'text'):raise ValueError('Modality mismatch')
        if meta['status']=='finished' and set(tasks)!=set(TASKS):raise ValueError('Finished run missing tasks')
        records[eid]={r['task']:r for r in raw}
    if not records or len(protocols)!=1:raise ValueError('No runs or mixed protocols')
    representations=[];answers=[]
    for eid,e in entries.items():
        ref=json.loads(inside(components,e['reference']).read_text())
        for c in CONDITIONS:
            for mode in MODES:
                raw=records.get(eid,{}).get(c+'_'+mode)
                diag,rows=score_representation(raw or dict(raw_response='{}'),ref,c)
                if raw is None:diag['error']='missing'
                common=dict(example_id=eid,source_group=e['source_group'],condition=c,mode=mode,attempted=raw is not None)
                representations.append(dict(**common,**diag))
                for row in rows:answers.append(dict(**common,**row))
    comparisons=[]
    for mode,path in [('image',image_baseline),('exact_text',text_baseline)]:
        if not path:continue
        if mode=='image':direct=baseline_rows(path,comp['examples'],components)
        else:
            import csv
            with Path(path).open() as stream:rows=list(csv.DictReader(stream))
            keys=[(r['source_group'],r['condition'],r['query']) for r in rows]
            expected={(e['source_group'],c,q) for e in comp['examples'] for c in CONDITIONS for q in QUERIES}
            if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Text baseline membership mismatch')
            direct=dict(zip(keys,rows))
        for row in answers:
            if row['mode']!=mode:continue
            ref=json.loads(inside(components,entries[row['example_id']]['reference']).read_text())
            key=(ref['source_examples'][row['condition']],row['query']) if mode=='image' else (row['source_group'],row['condition'],row['query'])
            old=direct[key]
            if float(old['reference_lower'])!=row['reference_lower'] or float(old['reference_upper'])!=row['reference_upper']:raise ValueError('Baseline bounds mismatch')
            both=row['valid'] and old['valid']=='True'
            comparisons.append(dict(source_group=row['source_group'],mode=mode,condition=row['condition'],query=row['query'],both_valid=both,
                direct_error=float(old['endpoint_mae']) if both else None,structured_error=row['endpoint_mae'] if both else None,
                structured_minus_direct=row['endpoint_mae']-float(old['endpoint_mae']) if both else None))
    summaries=[]
    for mode,c,q in itertools.product(MODES,CONDITIONS,QUERIES):
        rows=[r for r in answers if (r['mode'],r['condition'],r['query'])==(mode,c,q)];valid=[r for r in rows if r['valid']]
        shared=[r for r in comparisons if (r['mode'],r['condition'],r['query'])==(mode,c,q) and r['both_valid']]
        summaries.append(dict(mode=mode,condition=c,query=q,expected=len(rows),attempted=sum(r['attempted'] for r in rows),valid=len(valid),
            exact_count=sum(r['exact_bounds'] for r in rows),mean_endpoint_mae_valid_only=statistics.mean(r['endpoint_mae'] for r in valid) if valid else None,
            matched_count=len(shared),matched_mean_delta=statistics.mean(r['structured_minus_direct'] for r in shared) if shared else None))
    output.mkdir(parents=True);write_csv(output/'representations.csv',representations);write_csv(output/'answers.csv',answers);write_csv(output/'summary.csv',summaries)
    if comparisons:write_csv(output/'matched_direct.csv',comparisons)
    write_json(output/'evaluation.json',dict(experiment='structured_offsets_v1',expected_representations=len(representations),
        attempted=sum(r['attempted'] for r in representations),valid=sum(r['valid'] for r in representations),
        protocol_fingerprints=sorted(protocols),source_sha256=sha(source/'manifest.json'),components_sha256=sha(components/'manifest.json'),raw_sha256=hashes,
        image_baseline_sha256=sha(image_baseline) if image_baseline else None,text_baseline_sha256=sha(text_baseline) if text_baseline else None,
        notes=['Prompted extraction prototype plus fixed calculator; no weight changes or training.',
               'Variable names express dependence. No permutation or recombination of model outputs.',
               'Six previously inspected development sources. Exact-text control supplies displayed values and rules without images.',
               'Missing/invalid representations count as failed exact answers; conditional errors and matched-comparison counts are explicit.',
               'This is an established extraction/execution approach with a restricted shared-offset schema, not a novelty or calibration claim.']))
    return summaries


def run(source,components,index,output,seed=4646,dry_run=False):
    source,components=Path(source).resolve(),Path(components).resolve();comp,originals=dataset(source,components)
    if not 0<=index<len(comp['examples']) or seed<0:raise ValueError('Invalid index/seed')
    e=comp['examples'][index];ref=json.loads(inside(components,e['reference']).read_text());tasks=prompts(ref);folder=Path(output)/e['id']
    if folder.exists():raise ValueError('Output exists')
    options=dict(dataset=str(source),components=str(components),**{'map-index':index},output=str(output),seed=seed,revision=REVISION)
    meta=dict(experiment='structured_offsets_v1',status='planned',example_id=e['id'],tasks=list(TASKS),source_sha256=sha(source/'manifest.json'),
        components_sha256=sha(components/'manifest.json'),options=options,prompts=tasks,generation=GENERATION,model=MODELS['qwen7b'],calls=len(TASKS),
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in ('structured_prototype.py','interval_experiment.py','component_diagnostics.py','experiment_contract.py','run_vlm.py')})
    if dry_run:print(json.dumps(meta,indent=2));return
    import torch
    from PIL import Image
    from huggingface_hub import model_info
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('CUDA/bfloat16 required')
    revision=model_info(MODELS['qwen7b'],revision=REVISION).sha
    meta.update(status='loading',resolved_revision=revision,started_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,gpu=torch.cuda.get_device_name(0),
        cuda=torch.version.cuda,slurm_job_id=os.getenv('SLURM_JOB_ID'),packages={n:importlib.metadata.version(n) for n in ('torch','transformers','pillow','accelerate')})
    folder.mkdir(parents=True);write_json(folder/'run.json',meta)
    try:
        processor=AutoProcessor.from_pretrained(MODELS['qwen7b'],revision=revision,min_pixels=3136,max_pixels=1003520)
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODELS['qwen7b'],revision=revision,torch_dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
        meta.update(status='running',model_generation_config=model.generation_config.to_dict());meta['protocol_sha256']=fingerprint(meta);write_json(folder/'run.json',meta)
        with (folder/'responses.jsonl').open('x') as stream:
            for task in TASKS:
                c,mode=next((c,task[len(c)+1:]) for c in CONDITIONS if task.startswith(c+'_'))
                visual=mode=='image';trial_seed=seed+index*len(TASKS)+TASKS.index(task);set_seed(trial_seed)
                content=([{'type':'image'},{'type':'image'}] if visual else [])+[{'type':'text','text':tasks[task]}]
                rendered=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
                kwargs=dict(text=[rendered],padding=True,return_tensors='pt')
                if visual:
                    images=[]
                    for key in ('clean_image','image'):
                        with Image.open(inside(source,originals[(e['source_group'],c)][key])) as im:images.append(im.convert('RGB'))
                    kwargs['images']=images
                inputs=processor(**kwargs).to(model.device)
                with torch.inference_mode():tokens=model.generate(**inputs,**GENERATION)
                generated=tokens[:,inputs['input_ids'].shape[1]:];raw=processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                hit=generated.shape[1]>=GENERATION['max_new_tokens'] and int(generated[0,-1]) not in eos
                stream.write(json.dumps(dict(task=task,seed=trial_seed,input_modality='image' if visual else 'text',raw_response=raw,
                    hit_token_limit=hit,output_tokens=generated.shape[1],rendered_prompt=rendered),allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                print(task,raw,flush=True)
        meta.update(status='finished',finished_utc=datetime.now(timezone.utc).isoformat());write_json(folder/'run.json',meta)
    except BaseException as error:
        meta.update(status='failed',error=str(error));write_json(folder/'run.json',meta);raise


if __name__=='__main__':
    try:
        opts,rest=getopt.gnu_getopt(sys.argv[1:],'',['source=','components=','index=','output=','seed=','dry-run','evaluate=','image-baseline=','text-baseline='])
        a=dict(opts)
        if rest or '--output' not in a:raise ValueError('Required --output NEW_DIR')
        source=a.get('--source','experiments/controlled-v1');components=a.get('--components','experiments/component-diagnostics-v1')
        if '--evaluate' in a:evaluate(source,components,a['--evaluate'],a['--output'],a.get('--image-baseline'),a.get('--text-baseline'))
        else:run(source,components,int(a['--index']),a['--output'],int(a.get('--seed',4646)),'--dry-run' in a)
    except (ValueError,OSError,KeyError,ImportError,getopt.GetoptError) as error:sys.exit('Error: '+str(error))
