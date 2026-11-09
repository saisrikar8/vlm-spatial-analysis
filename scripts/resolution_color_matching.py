#!/usr/bin/env python3
"""Paired processor-resolution intervention on identical full-map images."""
import argparse
import json
from pathlib import Path
import re
import statistics
from color_matching import ROOT,MODEL,REVISION,sha,dump,table
from plain_patch_matching import PROMPT

SETTINGS={'baseline':dict(min_pixels=3136,max_pixels=1003520),
          'high':dict(min_pixels=2007040,max_pixels=2007040)}
GENERATION=dict(do_sample=False,max_new_tokens=16)


def prepare(source,out):
    import shutil
    if out.exists():raise ValueError('Choose a new dataset directory')
    m=json.loads((source/'manifest.json').read_text());examples=[e for e in m['examples'] if e['condition']=='full']
    if len(examples)!=40:raise ValueError('Expected 40 full-map examples')
    for e in examples:
        if sha(source/e['image'])!=e['image_sha256']:raise ValueError('Source image changed')
    out.mkdir(parents=True);(out/'images').mkdir();entries=[]
    for index,e in enumerate(examples):
        image=f"images/{e['point_id']}.png";shutil.copyfile(source/e['image'],out/image)
        order=['baseline','high'] if index%2==0 else ['high','baseline']
        for condition in order:
            entries.append(dict(id=e['point_id']+'-'+condition,point_id=e['point_id'],condition=condition,
                split=e['split'],map_id=e['map_id'],source_group=e['source_group'],answer=e['answer'],
                image=image,image_sha256=sha(out/image)))
    dump(out/'manifest.json',dict(experiment='resolution_color_matching_v1',source_manifest_sha256=sha(source/'manifest.json'),
        prompt=PROMPT,model=MODEL,revision=REVISION,generation=GENERATION,seed=20261106,settings=SETTINGS,
        primary='32 reused test points: baseline vs high processor-resolution accuracy, missing/invalid/OOM incorrect; costs descriptive.',
        design='Exact same full-map files and prompt. Force larger processed size using min_pixels, not only a higher cap. Weights unchanged.',examples=entries))
    table(out/'answer_key.csv',entries);print('Prepared 40 frozen images and 80 paired calls.')


def input_record(processor,image,prompt):
    text=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=prompt)])],tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[text],images=[image],return_tensors='pt')
    grid=inputs['image_grid_thw'][0].tolist();ip=processor.image_processor
    patch=ip.patch_size;merge=ip.merge_size
    expected=grid[0]*grid[1]*grid[2]//(merge*merge)
    image_id=processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    actual=int((inputs['input_ids']==image_id).sum().item())
    if actual!=expected:raise ValueError('Visual token accounting mismatch')
    info=dict(source_width=image.width,source_height=image.height,processed_height=grid[1]*patch,
        processed_width=grid[2]*patch,image_grid_thw=grid,patch_size=patch,merge_size=merge,
        visual_tokens=actual,total_input_tokens=int(inputs['input_ids'].shape[1]))
    return inputs,info


def evaluate(dataset,out):
    m=json.loads((dataset/'manifest.json').read_text());meta=json.loads((out/'run.json').read_text())
    if meta['manifest_sha256']!=sha(dataset/'manifest.json'):raise ValueError('Manifest mismatch')
    raw=[json.loads(s) for s in (out/'responses.jsonl').read_text().splitlines()];byid={r['id']:r for r in raw}
    if len(raw)!=len(byid) or set(byid)-{e['id'] for e in m['examples']}:raise ValueError('Duplicate/unknown responses')
    rows=[]
    for e in m['examples']:
        r=byid.get(e['id'],{});text=r.get('raw_response','');p=text.strip();valid=r.get('status')=='ok' and re.fullmatch('[1-6]',p) is not None
        row=dict(id=e['id'],point_id=e['point_id'],condition=e['condition'],split=e['split'],map_id=e['map_id'],reference=e['answer'],
            prediction=p,valid=valid,correct=valid and p==e['answer'],status=r.get('status','missing'),raw_response=text)
        for k in ['processed_width','processed_height','visual_tokens','preprocess_seconds','generation_seconds','peak_allocated_bytes','peak_reserved_bytes']:row[k]=r.get(k)
        rows.append(row)
    table(out/'answers.csv',rows);pairs=[]
    for pid in dict.fromkeys(e['point_id'] for e in m['examples']):
        group={r['condition']:r for r in rows if r['point_id']==pid};b,h=group['baseline'],group['high']
        pairs.append(dict(point_id=pid,split=b['split'],map_id=b['map_id'],reference=b['reference'],
            baseline_prediction=b['prediction'],high_prediction=h['prediction'],baseline_correct=b['correct'],high_correct=h['correct'],
            baseline_valid=b['valid'],high_valid=h['valid'],changed_answer=b['prediction']!=h['prediction']))
    table(out/'paired_answers.csv',pairs);summary={}
    for split in ['development','test']:
        ps=[r for r in pairs if r['split']==split];summary[split]=dict(n=len(ps),conditions={})
        for c in SETTINGS:
            rs=[r for r in rows if r['split']==split and r['condition']==c];ok=[r for r in rs if r['status']=='ok']
            d=dict(correct=sum(r['correct'] for r in rs),valid=sum(r['valid'] for r in rs),accuracy=sum(r['correct'] for r in rs)/len(rs),oom=sum(r['status']=='oom' for r in rs))
            for key in ['visual_tokens','preprocess_seconds','generation_seconds','peak_allocated_bytes','peak_reserved_bytes']:
                values=[r[key] for r in ok if r[key] is not None];d['mean_'+key]=statistics.mean(values) if values else None
            d['processed_sizes']=sorted({f"{r['processed_width']}x{r['processed_height']}" for r in ok})
            d['per_map']={mid:dict(n=len(mrs:=[r for r in rs if r['map_id']==mid]),correct=sum(r['correct'] for r in mrs)) for mid in sorted({r['map_id'] for r in rs})}
            summary[split]['conditions'][c]=d
        summary[split].update(wrong_to_correct=sum(not r['baseline_correct'] and r['high_correct'] for r in ps),correct_to_wrong=sum(r['baseline_correct'] and not r['high_correct'] for r in ps),changed_answers=sum(r['changed_answer'] for r in ps))
    summary['notes']=['Exploratory reused examples; no significance, fresh-test or training claim.',
        'Only processor resolution settings change; source images, choices, prompt, weights and generation stay fixed.',
        'Higher resolution upsamples existing pixels; cannot recover missing native detail or uniquely isolate attention.',
        'Costs include first-call effects and condition order; timing is descriptive. Peak GPU allocation includes model and inputs.',
        'Missing, malformed and OOM outputs count incorrect; no adaptive resolution fallback.']
    dump(out/'summary.json',summary);print(json.dumps(summary))


def run(dataset,out,inspect_only=False):
    print("Starting resolution run: importing inference stack",flush=True)
    import os,sys,time,importlib.metadata
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
    m=json.loads((dataset/'manifest.json').read_text())
    if out.exists():raise ValueError('Choose a new output directory')
    if m['experiment']!='resolution_color_matching_v1' or m['settings']!=SETTINGS or m['prompt']!=PROMPT or m['generation']!=GENERATION or m['revision']!=REVISION:raise ValueError('Protocol changed')
    for e in m['examples']:
        if sha(dataset/e['image'])!=e['image_sha256']:raise ValueError('Image changed')
    out.mkdir(parents=True)
    meta=dict(status='inspecting',manifest_sha256=sha(dataset/'manifest.json'),script_sha256=sha(__file__),
        helper_sha256=sha(Path(__file__).with_name('color_matching.py')),prompt_module_sha256=sha(Path(__file__).with_name('plain_patch_matching.py')),
        model=MODEL,revision=REVISION,prompt=PROMPT,settings=SETTINGS,generation=GENERATION,seed=m['seed'],job_id=os.getenv('SLURM_JOB_ID'),
        python=sys.version,packages={p:importlib.metadata.version(p) for p in ['torch','transformers','pillow','accelerate']})
    dump(out/'run.json',meta);set_seed(m['seed'])
    print('Loading cached processors',flush=True)
    processors={c:AutoProcessor.from_pretrained(MODEL,revision=REVISION,**opts) for c,opts in SETTINGS.items()}
    inspection=[]
    for e in m['examples']:
        with Image.open(dataset/e['image']) as im:image=im.convert('RGB')
        inputs,info=input_record(processors[e['condition']],image,PROMPT)
        inspection.append(dict(id=e['id'],point_id=e['point_id'],condition=e['condition'],**info));del inputs
        print('Processor inspection',e['id'],info['processed_width'],info['processed_height'],info['visual_tokens'],flush=True)
    for pid in {e['point_id'] for e in inspection}:
        group={e['condition']:e for e in inspection if e['point_id']==pid}
        if group['high']['visual_tokens']<3*group['baseline']['visual_tokens']:raise ValueError('Resolution manipulation failed')
    dump(out/'processor_inspection.json',inspection)
    meta['processors']={c:dict(image_processor_class=type(p.image_processor).__name__,configuration=p.image_processor.to_dict()) for c,p in processors.items()}
    if inspect_only:
        meta['status']='inspected_no_inference';dump(out/'run.json',meta);print(json.dumps(inspection[:2]));return
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    meta.update(status='loading',gpu=torch.cuda.get_device_name(0));dump(out/'run.json',meta)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
    meta['status']='running';dump(out/'run.json',meta)
    with (out/'responses.jsonl').open('x') as stream:
        for e in m['examples']:
            inputs=generated=None
            with Image.open(dataset/e['image']) as im:image=im.convert('RGB')
            torch.cuda.synchronize();start=time.perf_counter()
            inputs,info=input_record(processors[e['condition']],image,PROMPT)
            record=dict(id=e['id'],condition=e['condition'],**info,preprocess_seconds=time.perf_counter()-start)
            try:
                inputs=inputs.to(model.device);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                with torch.inference_mode():generated=model.generate(**inputs,**GENERATION)
                torch.cuda.synchronize();record.update(status='ok',generation_seconds=time.perf_counter()-start,
                    raw_response=processors[e['condition']].batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0],
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            except torch.cuda.OutOfMemoryError as error:
                record.update(status='oom',raw_response='',error=str(error),generation_seconds=None,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            finally:
                del inputs,generated;torch.cuda.empty_cache()
            stream.write(json.dumps(record)+'\n');stream.flush();print(e['id'],record['status'],record.get('raw_response'),flush=True)
    meta['status']='finished';dump(out/'run.json',meta);evaluate(dataset,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','inspect','run','evaluate']);p.add_argument('--dataset',type=Path,default=ROOT/'experiments/resolution-color-matching-v1');p.add_argument('--source',type=Path,default=ROOT/'experiments/plain-patch-matching-v1');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.mode=='prepare':prepare(a.source,a.dataset)
    elif a.output is None:p.error('--output required')
    elif a.mode=='evaluate':evaluate(a.dataset,a.output)
    else:run(a.dataset,a.output,inspect_only=a.mode=='inspect')
