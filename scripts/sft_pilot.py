#!/usr/bin/env python3
"""One fixed LoRA epoch, assistant-only labels, matched frozen test inference."""
from contextlib import nullcontext
from datetime import datetime,timezone
import getopt
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
from experiment_contract import sha,inside
from prepare_sft import validate,PROMPT
from structured_prototype import score_representation
from interval_experiment import write_csv
from run_vlm import MODELS,write_json
from run_pilot import REVISION

HP=dict(seed=20261102,epochs=1,learning_rate=1e-4,weight_decay=0.,microbatch=1,gradient_accumulation=4,
        lora_r=8,lora_alpha=16,lora_dropout=0.,max_pixels=327680,max_sequence=2048,max_grad_norm=1.,peft_version='0.17.1')
GENERATION=dict(do_sample=False,num_beams=1,repetition_penalty=1.,max_new_tokens=256,use_cache=True)


def mask_targets(full,prefix):
    if full[:len(prefix)]!=prefix:raise ValueError('Assistant boundary does not match user prefix')
    if len(full)<=len(prefix):raise ValueError('No assistant tokens')
    return [-100]*len(prefix)+full[len(prefix):]


def choose_targets(names):
    targets=[n for n in names if n.endswith(('.self_attn.q_proj','.self_attn.v_proj')) and 'visual' not in n]
    if not targets:raise ValueError('No text-attention adapter targets')
    return targets


def reference(root,e):
    ref=json.loads(inside(root,e['reference']).read_text())
    return dict(displayed_a=ref['displayed_a'],displayed_b=ref['displayed_b'],interval_references={e['condition']:ref})


def messages():
    return [{'role':'user','content':[{'type':'image'},{'type':'image'},{'type':'text','text':PROMPT}]}]


def encode(root,e,processor,device,supervised=False):
    import torch
    from PIL import Image
    images=[]
    for key in ('clean_image','image'):
        with Image.open(inside(root,e[key])) as im:images.append(im.convert('RGB'))
    prefix=processor.apply_chat_template(messages(),tokenize=False,add_generation_prompt=True)
    if supervised:
        target=json.dumps(e['target'],separators=(',',':'))
        full=processor.apply_chat_template(messages()+[{'role':'assistant','content':[{'type':'text','text':target}]}],tokenize=False,add_generation_prompt=False)
        batch=processor(text=[full],images=images,padding=True,return_tensors='pt')
        prefix_batch=processor(text=[prefix],images=images,padding=True,return_tensors='pt')
        labels=mask_targets(batch['input_ids'][0].tolist(),prefix_batch['input_ids'][0].tolist())
        batch['labels']=torch.tensor([labels],dtype=batch['input_ids'].dtype)
    else:batch=processor(text=[prefix],images=images,padding=True,return_tensors='pt')
    if batch['input_ids'].shape[1]>HP['max_sequence']:raise ValueError('Sequence exceeds frozen budget; no truncation allowed')
    return batch.to(device),prefix


def fingerprint_adapter(model):
    return sum(float(p.detach().float().norm().cpu()) for n,p in model.named_parameters() if 'lora_B' in n)


def infer(root,entries,model,processor,backend,stream):
    import torch
    model.eval();model.config.use_cache=True
    context=model.disable_adapter() if backend=='base' else nullcontext()
    with context,torch.inference_mode():
        for e in entries:
            inputs,rendered=encode(root,e,processor,model.device)
            out=model.generate(**inputs,**GENERATION);generated=out[:,inputs['input_ids'].shape[1]:]
            raw=processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
            hit=generated.shape[1]>=GENERATION['max_new_tokens'] and int(generated[0,-1]) not in eos
            stream.write(json.dumps(dict(example_id=e['id'],backend=backend,raw_response=raw,hit_token_limit=hit,
                output_tokens=generated.shape[1],rendered_prompt=rendered),allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            print('inference',backend,e['id'],flush=True)


def dev_loss(root,entries,model,processor):
    import torch
    model.eval();model.config.use_cache=False;total=0.;tokens=0
    with torch.inference_mode():
        for e in entries:
            if e['split']!='dev':raise ValueError('Development loss must use dev only')
            batch,_=encode(root,e,processor,model.device,True);n=int((batch['labels']!=-100).sum())
            loss=float(model(**batch).loss)
            if not math.isfinite(loss):raise ValueError('Nonfinite dev loss')
            total+=loss*n;tokens+=n
    return total/tokens


def run(dataset,output,smoke=False):
    dataset,output=Path(dataset).resolve(),Path(output)
    if output.exists():raise ValueError('Run output exists')
    manifest=validate(dataset)
    import torch
    from peft import LoraConfig,get_peft_model
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
    from huggingface_hub import model_info
    if importlib.metadata.version('peft')!=HP['peft_version']:raise ValueError('Unexpected PEFT version')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('CUDA/bfloat16 required')
    set_seed(HP['seed']);revision=model_info(MODELS['qwen7b'],revision=REVISION).sha
    if revision!=REVISION:raise ValueError('Revision mismatch')
    output.mkdir(parents=True)
    meta=dict(experiment='structured_sft_v1',status='loading',smoke=smoke,dataset_sha256=sha(dataset/'manifest.json'),hyperparameters=HP,generation=GENERATION,
        model=MODELS['qwen7b'],revision=revision,started_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
        slurm_job_id=os.getenv('SLURM_JOB_ID'),packages={n:importlib.metadata.version(n) for n in ('torch','transformers','peft','pillow','accelerate','huggingface-hub')},
        code_sha256={n:sha(Path(__file__).with_name(n)) for n in ('sft_pilot.py','prepare_sft.py','structured_prototype.py','interval_experiment.py')})
    write_json(output/'run.json',meta)
    try:
        processor=AutoProcessor.from_pretrained(MODELS['qwen7b'],revision=revision,min_pixels=3136,max_pixels=HP['max_pixels'])
        base=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODELS['qwen7b'],revision=revision,torch_dtype=torch.bfloat16,
            device_map={'':'cuda'},attn_implementation='sdpa')
        targets=choose_targets(n for n,m in base.named_modules())
        model=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',revision=revision,r=HP['lora_r'],lora_alpha=HP['lora_alpha'],
            lora_dropout=HP['lora_dropout'],target_modules=targets,bias='none'))
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not trainable or any('lora_' not in n or 'visual' in n for n,p in trainable):raise ValueError('Unexpected trainable base/vision weights')
        meta.update(status='running',adapter_targets=targets,trainable_parameters=sum(p.numel() for n,p in trainable),trainable_names=[n for n,p in trainable])
        write_json(output/'run.json',meta)
        train=[e for e in manifest['entries'] if e['split']=='train'];dev=[e for e in manifest['entries'] if e['split']=='dev']
        evaluation=[e for e in manifest['entries'] if e['split']==('dev' if smoke else 'test')]
        if smoke:train=train[:2];dev=dev[:3];evaluation=evaluation[:3]
        random.Random(HP['seed']).shuffle(train)
        # Test examples are used only for generation, never gradient updates or checkpoint selection.
        with (output/'responses.jsonl').open('x') as responses:
            infer(dataset,evaluation,model,processor,'base',responses)
            before=dev_loss(dataset,dev,model,processor)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
            model.train();base.model.visual.eval();model.config.use_cache=False
            optimizer=torch.optim.AdamW([p for n,p in trainable],lr=HP['learning_rate'],weight_decay=HP['weight_decay'])
            optimizer.zero_grad(set_to_none=True);initial_b=fingerprint_adapter(model);updates=0
            torch.cuda.reset_peak_memory_stats();started=time.monotonic();examples_seen=[]
            with (output/'training.jsonl').open('x') as log:
                for i,e in enumerate(train):
                    if e['split']!='train':raise ValueError('Training attempted to use non-train source')
                    batch,_=encode(dataset,e,processor,model.device,True)
                    active=int((batch['labels']!=-100).sum());loss=model(**batch).loss
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    group_start=i//HP['gradient_accumulation']*HP['gradient_accumulation'];group_size=min(HP['gradient_accumulation'],len(train)-group_start)
                    (loss/group_size).backward();examples_seen.append(e['id'])
                    end=(i+1)%HP['gradient_accumulation']==0 or i+1==len(train)
                    if end:
                        norm=torch.nn.utils.clip_grad_norm_([p for n,p in trainable],HP['max_grad_norm'],error_if_nonfinite=True)
                        optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                    record=dict(microstep=i+1,example_id=e['id'],source_group=e['source_group'],loss=float(loss.detach()),assistant_tokens=active,
                        sequence_tokens=batch['input_ids'].shape[1],optimizer_updates=updates,seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated())
                    log.write(json.dumps(record)+'\n');log.flush();os.fsync(log.fileno())
                    print('train',i+1,'/',len(train),'loss',record['loss'],'updates',updates,flush=True)
                    del batch,loss
            after_b=fingerprint_adapter(model)
            if after_b<=initial_b:raise ValueError('No adapter update detected')
            model.gradient_checkpointing_disable();model.eval();after=dev_loss(dataset,dev,model,processor)
            adapter=output/'adapter';model.save_pretrained(adapter,safe_serialization=True)
            write_json(output/'training_scope.json',dict(training_ids=examples_seen,dev_ids=[e['id'] for e in dev],
                evaluation_ids=[e['id'] for e in evaluation],optimizer_updates=updates,epochs=1,checkpoint_selection='Final epoch only; no dev/test model selection'))
            model.load_adapter(str(adapter),adapter_name='saved',is_trainable=False);model.set_adapter('saved')
            infer(dataset,evaluation,model,processor,'adapted',responses)
        meta.update(status='finished',finished_utc=datetime.now(timezone.utc).isoformat(),dev_loss_before=before,dev_loss_after=after,
            lora_b_norm_before=initial_b,lora_b_norm_after=after_b,optimizer_updates=updates,training_examples=len(train),
            evaluated_adapter='saved',evaluated_split='dev' if smoke else 'test',peak_gpu_bytes=torch.cuda.max_memory_allocated(),training_seconds=time.monotonic()-started,
            adapter_sha256={p.name:sha(p) for p in adapter.iterdir() if p.is_file()})
        write_json(output/'run.json',meta)
    except BaseException as error:
        meta.update(status='failed',error=str(error));write_json(output/'run.json',meta);raise


def evaluate(dataset,run,output):
    dataset,run,output=map(Path,(dataset,run,output))
    if output.exists():raise ValueError('Analysis exists')
    manifest=validate(dataset);meta=json.loads((run/'run.json').read_text())
    if meta['dataset_sha256']!=sha(dataset/'manifest.json') or meta['hyperparameters']!=HP or meta['generation']!=GENERATION:raise ValueError('Protocol/data mismatch')
    scope=json.loads((run/'training_scope.json').read_text());byid={e['id']:e for e in manifest['entries']}
    if any(byid[eid]['split']!='train' for eid in scope['training_ids']):raise ValueError('Training/test leakage')
    expected=[byid[eid] for eid in scope['evaluation_ids']]
    if any(e['split']!=meta['evaluated_split'] for e in expected):raise ValueError('Wrong evaluation split')
    if not meta['smoke'] and {e['id'] for e in expected}!={e['id'] for e in manifest['entries'] if e['split']=='test'}:raise ValueError('Missing held-out test examples')
    records={}
    for line in (run/'responses.jsonl').read_text().splitlines():
        raw=json.loads(line);key=(raw['example_id'],raw['backend'])
        if key in records or key[0] not in {e['id'] for e in expected} or key[1] not in ('base','adapted'):raise ValueError('Duplicate/out-of-scope response')
        records[key]=raw
    answers=[];representations=[]
    for e in expected:
        ref=reference(dataset,e)
        for backend in ('base','adapted'):
            raw=records.get((e['id'],backend));diag,rows=score_representation(raw or dict(raw_response='{}'),ref,e['condition'])
            common=dict(example_id=e['id'],source_group=e['source_group'],condition=e['condition'],backend=backend,attempted=raw is not None)
            representations.append(dict(**common,**diag))
            for row in rows:answers.append(dict(**common,**row))
    summaries=[]
    for backend in ('base','adapted'):
        for condition in ('clear','shared_offset','independent_offset'):
            reps=[r for r in representations if r['backend']==backend and r['condition']==condition];vr=[r for r in reps if r['valid']]
            for q in ('a','b','difference'):
                rows=[r for r in answers if (r['backend'],r['condition'],r['query'])==(backend,condition,q)];valid=[r for r in rows if r['valid']]
                summaries.append(dict(backend=backend,condition=condition,query=q,expected=len(rows),valid=len(valid),exact=sum(r['exact_bounds'] for r in rows),
                    endpoint_mae_valid_only=statistics.mean(r['endpoint_mae'] for r in valid) if valid else None,
                    base_mae_valid_only=statistics.mean(r['base_mae'] for r in vr) if vr else None,domains_correct=sum(r['domains_correct'] for r in reps),
                    sharing_correct=sum(r['sharing_correct'] is True for r in reps) if condition!='clear' else None))
    matched=[]
    lookup={(r['example_id'],r['backend'],r['query']):r for r in answers}
    for e in expected:
        for q in ('a','b','difference'):
            before,after=[lookup[e['id'],b,q] for b in ('base','adapted')];both=before['valid'] and after['valid']
            matched.append(dict(example_id=e['id'],source_group=e['source_group'],condition=e['condition'],query=q,both_valid=both,
                base_error=before['endpoint_mae'] if both else None,adapted_error=after['endpoint_mae'] if both else None,
                adapted_minus_base=after['endpoint_mae']-before['endpoint_mae'] if both else None))
    output.mkdir(parents=True);write_csv(output/'answers.csv',answers);write_csv(output/'representations.csv',representations);write_csv(output/'summary.csv',summaries);write_csv(output/'matched.csv',matched)
    write_json(output/'evaluation.json',dict(experiment='structured_sft_v1',smoke=meta['smoke'],split=meta['evaluated_split'],source_groups=len({e['source_group'] for e in expected}),
        examples=len(expected),dataset_sha256=sha(dataset/'manifest.json'),responses_sha256=sha(run/'responses.jsonl'),adapter_sha256=meta['adapter_sha256'],
        notes=['One fixed final-epoch adapter versus disabled-adapter base on identical test examples, prompt, processor pixel budget and calculator.',
               'Training sources and test sources are disjoint; all related conditions remain grouped. No test-based checkpoint selection.',
               'Readable synthetic layout with explicit numerical rules. Held-out fields in the same renderer, not natural-image or new-rule generalization.',
               'Exact counts retain invalid responses; numerical means and paired deltas condition on valid outputs. No significance or calibration claim.']))
    return summaries


if __name__=='__main__':
    try:
        opts,rest=getopt.gnu_getopt(sys.argv[1:],'',['dataset=','output=','smoke','evaluate='])
        a=dict(opts)
        if rest or '--output' not in a:raise ValueError('Required --output NEW_DIR')
        dataset=a.get('--dataset','experiments/structured-sft-v1')
        if '--evaluate' in a:print(json.dumps(evaluate(dataset,a['--evaluate'],a['--output']),indent=2))
        else:run(dataset,a['--output'],'--smoke' in a)
    except (ValueError,OSError,KeyError,ImportError,getopt.GetoptError) as error:sys.exit('Error: '+str(error))
