#!/usr/bin/env python3
"""Frozen, text-only arithmetic/interval/rule checks with inference provenance."""
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
from experiment_contract import sha
from interval_experiment import GENERATION, parse_interval, write_csv
from run_vlm import MODELS, parse_direct, write_json
from run_pilot import REVISION

CASES=((14,12),(18,24),(17,12),(24,22),(18,21),(21,25),(0,0),(5,5),(12,14),(1,9),
       (9,1),(-3,2),(3,-2),(-3,-2),(100,1),(101,99),(7,0),(0,7),(28,11),(11,28))
VERSIONS=('plain','interval','offset')


def prompts(a,b):
    suffix=(' Give the tightest closed numerical interval containing the answer. '
            'Both endpoints must equal the answer when it is uniquely determined. '
            'Return only {"lower": number, "upper": number}. Do not explain.')
    return dict(plain=f'What is ({a}) minus ({b})? Return only one number. Do not explain.',
        interval=f'What is ({a}) minus ({b})?'+suffix,
        offset=f'A = ({a-10}) + 10. B = ({b-10}) + 10. What is A minus B?'+suffix)


def plan():
    return dict(experiment='arithmetic_sanity_v1',model=MODELS['qwen7b'],revision=REVISION,generation=GENERATION,
        cases=[dict(id=f'case-{i:02d}',a=a,b=b,answer=a-b,prompts=prompts(a,b)) for i,(a,b) in enumerate(CASES)],
        scored_calls=len(CASES)*len(VERSIONS),extra_control_calls=1,
        control='Repeat the first plain prompt after all scored calls; compare rendered input IDs and generated token IDs.',
        metrics='Exact numerical answer and absolute error. Interval error is mean absolute endpoint error to [a-b,a-b].',
        scope='Fixed diagnostic examples, including zeros, ties, signs and two-/three-digit operands; no uncertainty or generalization claim.')


def score(raw):
    i=raw['case_index'];version=raw['version'];a,b=CASES[i];truth=a-b
    if version=='plain':
        value,error=parse_direct(raw['raw_response'],'difference');answer=value['answer'] if value else None
        lo=hi=answer
    else:
        value,error=parse_interval(raw['raw_response']);lo=value['lower'] if value else None;hi=value['upper'] if value else None
    valid=lo is not None and hi is not None and not raw.get('hit_token_limit',False)
    err=(abs(lo-truth)+abs(hi-truth))/2 if valid else None
    if err is not None and not math.isfinite(err):valid=False;err=None;error='arithmetic overflow'
    return dict(case_index=i,a=a,b=b,version=version,truth=truth,valid=valid,error='truncated' if raw.get('hit_token_limit') else error or '',
        predicted_lower=lo,predicted_upper=hi,numerical_error=err,exact=bool(valid and abs(lo-truth)<=1e-6 and abs(hi-truth)<=1e-6))


def evaluate(run,output):
    run,output=Path(run),Path(output)
    if output.exists():raise ValueError('Output exists')
    metadata=json.loads((run/'run.json').read_text());saved=json.loads((run/'plan.json').read_text())
    if saved!=plan() or metadata['plan_sha256']!=sha(run/'plan.json'):raise ValueError('Plan mismatch')
    if metadata.get('generation')!=GENERATION:raise ValueError('Generation configuration mismatch')
    if metadata['model']!=MODELS['qwen7b'] or metadata['resolved_revision']!=REVISION:raise ValueError('Checkpoint mismatch')
    records=[json.loads(line) for line in (run/'responses.jsonl').read_text().splitlines()]
    seen=set();rows=[];control=[]
    for raw in records:
        i,version=raw['case_index'],raw['version']
        if not 0<=i<len(CASES) or version not in VERSIONS:raise ValueError('Unknown test case/version')
        if raw['prompt']!=prompts(*CASES[i])[version]:raise ValueError('Prompt mismatch')
        if raw.get('control'):
            if i!=0 or version!='plain':raise ValueError('Unexpected repeat control')
            control.append(raw);continue
        if (i,version) in seen:raise ValueError('Duplicate test answer')
        seen.add((i,version));rows.append(score(raw))
    for i,(a,b) in enumerate(CASES):
        for version in VERSIONS:
            if (i,version) not in seen:
                row=score(dict(case_index=i,version=version,raw_response='{}'));row['error']='missing';rows.append(row)
    rows.sort(key=lambda r:(r['case_index'],VERSIONS.index(r['version'])))
    if metadata['status']=='finished' and (len(seen)!=60 or len(control)!=1):raise ValueError('Finished run missing calls')
    summaries=[]
    for version in VERSIONS:
        subset=[r for r in rows if r['version']==version];valid=[r for r in subset if r['valid']]
        summaries.append(dict(version=version,expected=len(subset),attempted=sum((r['case_index'],version) in seen for r in subset),valid=len(valid),
            correct=sum(r['exact'] for r in subset),accuracy=statistics.mean(r['exact'] for r in subset),
            mean_numerical_error_valid_only=statistics.mean(r['numerical_error'] for r in valid) if valid else None))
    if len(control)>1:raise ValueError('Duplicate repeat control')
    first=next((r for r in records if r['case_index']==0 and r['version']=='plain' and not r.get('control')),None)
    repeated=control[0] if len(control)==1 else None
    check=dict(present=first is not None and repeated is not None,
        identical_input_ids=first['input_ids']==repeated['input_ids'] if first and repeated else None,
        identical_generated_ids=first['generated_ids']==repeated['generated_ids'] if first and repeated else None,
        first_answer=first['raw_response'] if first else None,repeated_answer=repeated['raw_response'] if repeated else None)
    output.mkdir(parents=True);write_csv(output/'answers.csv',rows);write_csv(output/'summary.csv',summaries)
    write_json(output/'evaluation.json',dict(experiment='arithmetic_sanity_v1',scored_attempts=len(seen),expected_scored_answers=60,
        valid=sum(r['valid'] for r in rows),summary=summaries,repeat_control=check,plan_sha256=sha(run/'plan.json'),
        raw_sha256=sha(run/'responses.jsonl'),loader_checks=metadata.get('loader_checks'),
        notes=['No images, uncertainty sampling, permutation or prompt selection.',
               'Missing/invalid answers count as incorrect; numerical MAE conditions on valid answers.',
               'One fixed model/configuration, greedy decoding, fresh user-only conversations. The repeated prompt checks order/state sensitivity.',
               'A repeated-answer difference can reflect runtime/state/numerical effects; it does not identify a cause by itself.']))
    return summaries


def run(output,dry_run=False):
    output=Path(output)
    if output.exists():raise ValueError('Run output exists')
    if dry_run:print(json.dumps(plan(),indent=2));return
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
    from huggingface_hub import model_info,try_to_load_from_cache
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('CUDA/bfloat16 required')
    revision=model_info(MODELS['qwen7b'],revision=REVISION).sha
    if revision!=REVISION:raise ValueError('Requested checkpoint resolved unexpectedly')
    output.mkdir(parents=True);write_json(output/'plan.json',plan())
    metadata=dict(experiment='arithmetic_sanity_v1',status='loading',model=MODELS['qwen7b'],resolved_revision=revision,
        plan_sha256=sha(output/'plan.json'),generation=GENERATION,python=sys.version,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
        started_utc=datetime.now(timezone.utc).isoformat(),slurm_job_id=os.getenv('SLURM_JOB_ID'),
        packages={name:importlib.metadata.version(name) for name in ('torch','transformers','pillow','accelerate','huggingface-hub')},
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in ('arithmetic_sanity.py','interval_experiment.py','run_vlm.py','experiment_contract.py')})
    write_json(output/'run.json',metadata)
    try:
        processor=AutoProcessor.from_pretrained(MODELS['qwen7b'],revision=revision,min_pixels=3136,max_pixels=1003520)
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODELS['qwen7b'],revision=revision,torch_dtype=torch.bfloat16,
            device_map={'':'cuda'},attn_implementation='sdpa').eval()
        tokenizer=processor.tokenizer
        commit=getattr(model.config,'_commit_hash',None)
        if commit is not None and commit!=revision:raise ValueError('Loaded model config has wrong commit')
        if model.config.model_type!='qwen2_5_vl':raise ValueError('Unexpected model architecture')
        files={}
        for name in ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'):
            path=try_to_load_from_cache(MODELS['qwen7b'],name,revision=revision)
            if isinstance(path,str) and Path(path).is_file():files[name]=sha(path)
        checks=dict(model_class=type(model).__name__,model_type=model.config.model_type,config_commit=commit,
            config_name=getattr(model.config,'_name_or_path',None),architectures=getattr(model.config,'architectures',None),
            tokenizer_class=type(tokenizer).__name__,tokenizer_name=tokenizer.name_or_path,
            tokenizer_requested_revision=revision,tokenizer_commit=tokenizer.init_kwargs.get('_commit_hash'),
            parameter_count=sum(p.numel() for p in model.parameters()),parameter_device=str(next(model.parameters()).device),
            attention_implementation=getattr(model.config,'_attn_implementation',None),
            tokenizer_length=len(tokenizer),embedding_rows=model.get_input_embeddings().num_embeddings,
            special_token_ids=dict(eos=tokenizer.eos_token_id,pad=tokenizer.pad_token_id,bos=tokenizer.bos_token_id),
            checkpoint_asset_sha256=files,actual_model_dtype=str(next(model.parameters()).dtype),model_training=model.training)
        metadata.update(status='running',loader_checks=checks,model_generation_config=model.generation_config.to_dict());write_json(output/'run.json',metadata)
        order=[(i,v,False) for i in range(len(CASES)) for v in VERSIONS]+[(0,'plain',True)]
        with (output/'responses.jsonl').open('x') as stream:
            for call,(i,version,control) in enumerate(order):
                prompt=prompts(*CASES[i])[version];set_seed(4545)
                rendered=processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':prompt}]}],tokenize=False,add_generation_prompt=True)
                inputs=processor(text=[rendered],padding=True,return_tensors='pt').to(model.device)
                ids=inputs['input_ids'][0].tolist()
                if max(ids)>=checks['embedding_rows'] or min(ids)<0:raise ValueError('Input tokens exceed embedding vocabulary')
                encoded=tokenizer.encode(rendered,add_special_tokens=False)
                if ids!=encoded:raise ValueError('Processor and direct tokenizer disagree on input IDs')
                with torch.inference_mode():tokens=model.generate(**inputs,**GENERATION)
                generated=tokens[:,inputs['input_ids'].shape[1]:];generated_ids=generated[0].tolist()
                raw=processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                hit=len(generated_ids)>=GENERATION['max_new_tokens'] and generated_ids[-1] not in eos
                record=dict(call_index=call,case_index=i,version=version,control=control,prompt=prompt,rendered_prompt=rendered,
                    input_ids=ids,decoded_input=tokenizer.decode(ids,skip_special_tokens=False),generated_ids=generated_ids,raw_response=raw,
                    hit_token_limit=hit,processor_tokenizer_agree=True)
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                print(call,i,version,'control' if control else '',raw,flush=True)
        metadata.update(status='finished',finished_utc=datetime.now(timezone.utc).isoformat());write_json(output/'run.json',metadata)
    except BaseException as error:
        metadata.update(status='failed',error=str(error));write_json(output/'run.json',metadata);raise


if __name__=='__main__':
    try:
        opts,rest=getopt.gnu_getopt(sys.argv[1:],'',['output=','dry-run','evaluate='])
        a=dict(opts)
        if rest or '--output' not in a:raise ValueError('Required --output NEW_DIR')
        if '--evaluate' in a:
            print(json.dumps(evaluate(a['--evaluate'],a['--output']),indent=2))
        else:run(a['--output'],'--dry-run' in a)
    except (ValueError,OSError,KeyError,ImportError,getopt.GetoptError) as error:sys.exit('Error: '+str(error))
