#!/usr/bin/env python3
"""Controlled language/vision/joint LoRA comparison on the fixed color task."""
from contextlib import nullcontext
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
from color_matching import ROOT,MODEL,REVISION,sha,dump,table
from prepare_color_sft import validate,PROMPT

HP=dict(seed=20261108,epochs=1,learning_rate=1e-4,weight_decay=0.,gradient_accumulation=4,
        lora_r=8,lora_alpha=16,lora_dropout=0.,min_pixels=3136,max_pixels=1003520,max_sequence=2048,max_grad_norm=1.,peft_version='0.17.1')
GENERATION=dict(do_sample=False,num_beams=1,repetition_penalty=1.,max_new_tokens=16,use_cache=True)


def mask_targets(full,prefix):
    if full[:len(prefix)]!=prefix or len(full)<=len(prefix):raise ValueError('Assistant boundary mismatch')
    return [-100]*len(prefix)+full[len(prefix):]


def choose_targets(names,site):
    if site not in ('language','vision','joint'):raise ValueError('Unknown adapter site')
    names=list(names)
    language=[n for n in names if n.endswith(('.self_attn.q_proj','.self_attn.v_proj')) and 'visual' not in n]
    vision=[n for n in names if '.visual.blocks.' in n and n.endswith(('.attn.qkv','.attn.proj'))]
    if not language or not vision:raise ValueError('Expected language and vision attention modules')
    return language if site=='language' else vision if site=='vision' else language+vision


def branch_norms(model,gradient=False):
    values={'language':0.,'vision':0.}
    for name,p in model.named_parameters():
        if 'lora_' not in name:continue
        if not gradient and 'lora_B' not in name:continue
        value=p.grad if gradient else p.detach()
        if value is not None:values['vision' if 'visual' in name else 'language']+=float(value.detach().float().square().sum())
    return {key:value**.5 for key,value in values.items()}


def encode(root,e,processor,device,supervised=False):
    import torch
    from PIL import Image
    with Image.open(root/e['image']) as im:image=im.convert('RGB')
    messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text=PROMPT)])]
    prefix=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    if supervised:
        rendered=processor.apply_chat_template(messages+[dict(role='assistant',content=[dict(type='text',text=e['answer'])])],tokenize=False,add_generation_prompt=False)
        batch=processor(text=[rendered],images=[image],return_tensors='pt')
        pre=processor(text=[prefix],images=[image],return_tensors='pt')
        labels=mask_targets(batch['input_ids'][0].tolist(),pre['input_ids'][0].tolist())
        batch['labels']=torch.tensor([labels],dtype=batch['input_ids'].dtype)
    else:batch=processor(text=[prefix],images=[image],return_tensors='pt')
    if batch['input_ids'].shape[1]>HP['max_sequence']:raise ValueError('Frozen sequence budget exceeded; no truncation')
    return batch.to(device)


def infer(root,entries,model,processor,backend,stream):
    import torch
    model.eval();model.config.use_cache=True
    with (model.disable_adapter() if backend=='base' else nullcontext()):
        for e in entries:
            batch=encode(root,e,processor,model.device);torch.cuda.synchronize();started=time.monotonic()
            with torch.inference_mode():generated=model.generate(**batch,**GENERATION)
            torch.cuda.synchronize();raw=processor.batch_decode(generated[:,batch.input_ids.shape[1]:],skip_special_tokens=True)[0]
            record=dict(id=e['id'],backend=backend,raw_response=raw,seconds=time.monotonic()-started,
                visual_tokens=int(batch['image_grid_thw'].prod().item())//4)
            stream.write(json.dumps(record)+'\n');stream.flush();print(backend,e['id'],raw,flush=True)
            del batch,generated


def dev_loss(root,entries,model,processor):
    import torch
    model.eval();total=0.;tokens=0
    with torch.no_grad():
        for e in entries:
            batch=encode(root,e,processor,model.device,True);n=int((batch['labels']!=-100).sum())
            loss=model(**batch).loss
            if not torch.isfinite(loss):raise ValueError('Nonfinite development loss')
            total+=float(loss)*n;tokens+=n;del batch,loss
    return total/tokens


def adapter_norm(model):return sum(float(p.detach().float().square().sum()) for n,p in model.named_parameters() if 'lora_B' in n)**.5


def run(dataset,out,smoke=False,smoke_gate=None,site="language"):
    import torch
    from peft import LoraConfig,get_peft_model
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
    root=Path(dataset).resolve();m=validate(root)
    if out.exists():raise ValueError('Choose new output path')
    if importlib.metadata.version('peft')!=HP['peft_version']:raise ValueError('Unexpected PEFT version')
    if not smoke:
        if smoke_gate is None:raise ValueError('Completed smoke gate required')
        gate=json.loads((smoke_gate/'run.json').read_text())
        if gate.get('status')!='finished' or not gate.get('smoke') or gate['dataset_sha256']!=sha(root/'manifest.json') or gate['code_sha256']['adapter_site_sft.py']!=sha(__file__) or gate.get('adapter_site')!=site or gate['hyperparameters']!=HP:raise ValueError('Smoke gate mismatch')
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();set_seed(HP['seed']);out.mkdir(parents=True)
    meta=dict(experiment='adapter_site_sft_v1',adapter_site=site,status='loading',smoke=smoke,dataset_sha256=sha(root/'manifest.json'),hyperparameters=HP,generation=GENERATION,
        model=MODEL,revision=REVISION,prompt=PROMPT,started_utc=datetime.now(timezone.utc).isoformat(),gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
        job_id=os.getenv('SLURM_JOB_ID'),python=sys.version,packages={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','pillow','accelerate']},
        code_sha256={n:sha(Path(__file__).with_name(n)) for n in ['adapter_site_sft.py','prepare_color_sft.py','color_matching.py','plain_patch_matching.py']},
        smoke_gate=str(smoke_gate) if smoke_gate else None)
    dump(out/'run.json',meta)
    try:
        processor=AutoProcessor.from_pretrained(MODEL,revision=REVISION,min_pixels=HP['min_pixels'],max_pixels=HP['max_pixels'])
        base=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa')
        targets=choose_targets((n for n,_ in base.named_modules()),site)
        model=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',revision=REVISION,r=HP['lora_r'],lora_alpha=HP['lora_alpha'],lora_dropout=HP['lora_dropout'],target_modules=targets,bias='none'))
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not trainable or any('lora_' not in n for n,p in trainable):raise ValueError('Unexpected trainable base weights')
        active={'language','vision'} if site=='joint' else {site}
        actual={'vision' if 'visual' in n else 'language' for n,p in trainable}
        if actual!=active:raise ValueError('Wrong trainable subsystem')
        train=[e for e in m['entries'] if e['split']=='train'];dev=[e for e in m['entries'] if e['split']=='dev'];test=[e for e in m['entries'] if e['split']=='test']
        if smoke:train=train[:2];dev=dev[:2];test=dev
        random.Random(HP['seed']).shuffle(train)
        dump(out/'training_scope.json',dict(training_ids=[e['id'] for e in train],dev_ids=[e['id'] for e in dev],evaluation_ids=[e['id'] for e in test],evaluated_split='dev' if smoke else 'test',checkpoint_selection='Final epoch only; no dev/test checkpoint selection'))
        meta.update(status='running',trainable_parameters_by_site={b:sum(p.numel() for n,p in trainable if ('vision' if 'visual' in n else 'language')==b) for b in ['language','vision']},trainable_parameters=sum(p.numel() for n,p in trainable),adapter_targets=targets,trainable_names=[n for n,p in trainable]);dump(out/'run.json',meta)
        with (out/'responses.jsonl').open('x') as responses:
            infer(root,test,model,processor,'base',responses)
            before=dev_loss(root,dev,model,processor)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
            model.train()
            if site=='language':base.model.visual.eval()
            model.config.use_cache=False
            optimizer=torch.optim.AdamW([p for n,p in trainable],lr=HP['learning_rate'],weight_decay=HP['weight_decay']);optimizer.zero_grad(set_to_none=True)
            initial=adapter_norm(model);initial_branches=branch_norms(model);first_gradients=None;updates=0;torch.cuda.reset_peak_memory_stats();started=time.monotonic()
            with (out/'training.jsonl').open('x') as stream:
                for i,e in enumerate(train):
                    if e['split']!='train':raise ValueError('Non-training example in optimizer')
                    batch=encode(root,e,processor,model.device,True);loss=model(**batch).loss
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    group_start=i//HP['gradient_accumulation']*HP['gradient_accumulation'];size=min(HP['gradient_accumulation'],len(train)-group_start)
                    (loss/size).backward()
                    if first_gradients is None:
                        first_gradients=branch_norms(model,gradient=True)
                        if any(first_gradients[b]<=0 for b in active):raise ValueError('Missing gradient in intended adapter subsystem')
                    if (i+1)%HP['gradient_accumulation']==0 or i+1==len(train):
                        torch.nn.utils.clip_grad_norm_([p for n,p in trainable],HP['max_grad_norm'],error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                    record=dict(microstep=i+1,id=e['id'],source_group=e['source_group'],loss=float(loss.detach()),assistant_tokens=int((batch['labels']!=-100).sum()),sequence_tokens=int(batch['input_ids'].shape[1]),updates=updates,elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    stream.write(json.dumps(record)+'\n');stream.flush();print('train',i+1,'/',len(train),record['loss'],flush=True);del batch,loss
            training_elapsed=time.monotonic()-started
            after_norm=adapter_norm(model)
            if after_norm<=initial:raise ValueError('No adapter update detected')
            after_branches=branch_norms(model)
            if any(after_branches[b]<=initial_branches[b] for b in active):raise ValueError('No LoRA-B update in intended subsystem')
            model.gradient_checkpointing_disable();model.eval();model.config.use_cache=True;after=dev_loss(root,dev,model,processor)
            model.save_pretrained(out/'adapter',safe_serialization=True);del optimizer
            model.load_adapter(str(out/'adapter'),adapter_name='saved',is_trainable=False);model.set_adapter('saved')
            infer(root,test,model,processor,'adapted',responses)
        meta.update(status='finished',dev_loss_before=before,dev_loss_after=after,lora_b_norm_before=initial,lora_b_norm_after=after_norm,
            initial_adapter_norms=initial_branches,final_adapter_norms=after_branches,first_gradient_norms=first_gradients,training_examples=len(train),optimizer_updates=updates,training_seconds=training_elapsed,training_plus_final_evaluation_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            evaluated_adapter='saved',adapter_sha256={p.name:sha(p) for p in (out/'adapter').iterdir() if p.is_file()})
        dump(out/'run.json',meta);evaluate(root,out)
    except BaseException as error:
        meta.update(status='failed',error=str(error));dump(out/'run.json',meta);raise


def evaluate(dataset,out):
    m=validate(dataset);meta=json.loads((out/'run.json').read_text());scope=json.loads((out/'training_scope.json').read_text());byid={e['id']:e for e in m['entries']}
    if meta['dataset_sha256']!=sha(dataset/'manifest.json'):raise ValueError('Dataset mismatch')
    if any(byid[eid]['split']!='train' for eid in scope['training_ids']):raise ValueError('Training split violation')
    if set(scope['training_ids'])&set(scope['evaluation_ids']):raise ValueError('Training/evaluation overlap')
    if scope['evaluated_split']!=('dev' if meta['smoke'] else 'test'):raise ValueError('Wrong evaluation split')
    raw=[json.loads(s) for s in (out/'responses.jsonl').read_text().splitlines()];records={(r['id'],r['backend']):r for r in raw}
    if len(records)!=len(raw):raise ValueError('Duplicate responses')
    expected={(eid,b) for eid in scope['evaluation_ids'] for b in ['base','adapted']}
    if set(records)!=expected:raise ValueError('Incomplete/unknown evaluation responses')
    rows=[]
    for eid in scope['evaluation_ids']:
        e=byid[eid]
        if e['split']!=scope['evaluated_split']:raise ValueError('Unexpected evaluation example')
        for backend in ['base','adapted']:
            r=records[(eid,backend)];p=r['raw_response'].strip();valid=re.fullmatch('[1-6]',p) is not None
            rows.append(dict(id=eid,point_id=e['point_id'],source_group=e['source_group'],condition=e['condition'],backend=backend,
                reference=e['answer'],prediction=p,valid=valid,correct=valid and p==e['answer'],raw_response=r['raw_response']))
    table(out/'answers.csv',rows)
    paired=[]
    for eid in scope['evaluation_ids']:
        group={r['backend']:r for r in rows if r['id']==eid};b,a=group['base'],group['adapted']
        paired.append(dict(id=eid,condition=b['condition'],source_group=b['source_group'],reference=b['reference'],base_prediction=b['prediction'],adapted_prediction=a['prediction'],base_correct=b['correct'],adapted_correct=a['correct'],base_valid=b['valid'],adapted_valid=a['valid']))
    table(out/'paired_answers.csv',paired)
    summary=dict(adapter_site=meta['adapter_site'],evaluated_split=scope['evaluated_split'],source_groups=len({r['source_group'] for r in rows}),conditions={})
    for condition in ['full','crop']:
        summary['conditions'][condition]={}
        for backend in ['base','adapted']:
            rs=[r for r in rows if r['condition']==condition and r['backend']==backend]
            summary['conditions'][condition][backend]=dict(n=len(rs),correct=sum(r['correct'] for r in rs),valid=sum(r['valid'] for r in rs),accuracy=statistics.mean(r['correct'] for r in rs),
                per_source={g:dict(n=len(gs:=[r for r in rs if r['source_group']==g]),correct=sum(r['correct'] for r in gs)) for g in sorted({r['source_group'] for r in rs})})
    for condition in ['full','crop']:
        ps=[r for r in paired if r['condition']==condition]
        summary['conditions'][condition]['paired']=dict(wrong_to_correct=sum(not r['base_correct'] and r['adapted_correct'] for r in ps),correct_to_wrong=sum(r['base_correct'] and not r['adapted_correct'] for r in ps))
    summary['notes']=['One fixed-epoch seed; only three held-out source groups in full pilot, not independent per-point evidence.',
        'Same images/prompts/resolution/generation for base and saved adapter. All original weights frozen; LoRA site recorded separately.',
        'Test source groups excluded from training and recent diagnostics, but drawn from archived ILAMB images.',
        'Six-choice color reading only; not numerical legend reading or geographic evaluation.']
    dump(out/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=ROOT/'experiments/color-sft-v1');p.add_argument('--output',type=Path,required=True);p.add_argument('--site',choices=['language','vision','joint'],required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-gate',type=Path);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    if a.evaluate:evaluate(a.dataset,a.output)
    else:run(a.dataset,a.output,a.smoke,a.smoke_gate,a.site)
