#!/usr/bin/env python3
"""Matched-point color diagnostic; no geographic or numerical ground truth used."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import re

ROOT=Path(__file__).resolve().parents[1]
MODEL='Qwen/Qwen2.5-VL-7B-Instruct'
REVISION='cc594898137f460bfe9f0759e9844b3ce807cfb5'
PROMPT=('Which numbered color swatch at the bottom best matches the underlying map color '
        'at the CENTER of the red circular marker? Match the map color, not the red or white '
        'marker outline or the gray grid lines. Return only one digit from 1 to 6. '
        'Do not read numerical map values or report the grid cell.')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def table(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def prepare(source,out):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    m=json.loads((source/'manifest.json').read_text())
    if out.exists():raise ValueError('Choose new dataset path')
    rng=random.Random(20261106)
    labels=[str(i%6+1) for i in range(len(m['examples']))];rng.shuffle(labels)
    plans=[]
    for e,label in zip(m['examples'],labels):
        clean=ROOT/e['source_image'];assert sha(clean)==e['source_sha256']
        arr=np.asarray(Image.open(clean).convert('RGB'))
        rgb=arr[e['y'],e['x']].astype(float)
        # Deterministic farthest-color choices drawn from the same map panel.
        pool=np.unique(arr[:375].reshape(-1,3),axis=0).astype(float)
        pool=pool[np.ptp(pool,axis=1)>8]
        selected=[rgb]
        for _ in range(5):
            dist=np.min(np.stack([np.linalg.norm(pool-c,axis=1) for c in selected]),axis=0)
            j=int(np.argmax(dist))
            if dist[j]<35:raise ValueError('Insufficient distinct colors')
            selected.append(pool[j])
        distractors=selected[1:];rng.shuffle(distractors)
        choices=distractors[:];choices.insert(int(label)-1,rgb)
        choices=[[int(v) for v in c] for c in choices]
        original=source/e['image'];assert sha(original)==e['image_sha256']
        with Image.open(original) as im:
            base=im.convert('RGB');assert base.getpixel((e['x']+45,e['y']+45))==tuple(rgb.astype(int))
        plans.append(dict(id=e['id'],split=e['split'],map_id=e['map_id'],source_group=e['source_group'],
            grid_reference=e['answer'],x=e['x'],y=e['y'],reference_rgb=rgb.astype(int).tolist(),
            swatches=choices,answer=label,source_image=e['image'],source_sha256=e['image_sha256']))
    out.mkdir(parents=True);(out/'images').mkdir()
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',22)
    for e in plans:
        with Image.open(source/e['source_image']) as im:
            canvas=Image.new('RGB',(795,625),'white');canvas.paste(im,(0,0))
        d=ImageDraw.Draw(canvas)
        for i,rgb in enumerate(e['swatches']):
            left=35+i*128
            d.rectangle((left,535,left+85,585),fill=tuple(rgb),outline='black',width=1)
            d.text((left+42,604),str(i+1),font=font,fill='black',anchor='mm')
        e['image']=f"images/{e['id']}.png";canvas.save(out/e['image']);e['image_sha256']=sha(out/e['image'])
    dump(out/'manifest.json',dict(experiment='color_matching_v1',seed=20261106,source_manifest_sha256=sha(source/'manifest.json'),
        prompt=PROMPT,model=MODEL,revision=REVISION,generation=dict(do_sample=False,max_new_tokens=16),
        swatch_rule='Exact original center RGB plus five deterministic farthest RGB colors from same map panel; randomized positions independent of model outputs.',
        scope='Six-option color matching at existing markers; same grid map pixels plus appended swatches. Not numerical reading or geographic validation.',examples=plans))
    table(out/'answer_key.csv',plans)
    (out/'preview.html').write_text('<html><body>'+''.join(f'<p>{e["id"]} ({e["split"]})</p><img src="{e["image"]}">' for e in plans)+'</body></html>')
    print('Prepared',len(plans),'examples')

def evaluate(dataset,out,grid):
    m=json.loads((dataset/'manifest.json').read_text());meta=json.loads((out/'run.json').read_text())
    if meta['manifest_sha256']!=sha(dataset/'manifest.json'):raise ValueError('Manifest mismatch')
    raw=[json.loads(x) for x in (out/'responses.jsonl').read_text().splitlines()]
    byid={r['id']:r for r in raw}
    if len(byid)!=len(raw) or set(byid)-{e['id'] for e in m['examples']}:raise ValueError('Duplicate/unknown responses')
    gr=[json.loads(x) for x in (grid/'responses.jsonl').read_text().splitlines()];g={r['id']:r for r in gr}
    gm=json.loads((ROOT/'experiments/grid-localization-v1/manifest.json').read_text())
    # Same source manifest and exact original grid responses, not selected by outcomes.
    if m['source_manifest_sha256']!=sha(ROOT/'experiments/grid-localization-v1/manifest.json'):raise ValueError('Source mismatch')
    if json.loads((grid/'run.json').read_text())['manifest_sha256']!=m['source_manifest_sha256']:raise ValueError('Grid run mismatch')
    rows=[]
    for e in m['examples']:
        r=byid.get(e['id']);prediction=r['raw_response'].strip() if r else ''
        valid=re.fullmatch('[1-6]',prediction) is not None
        if g[e['id']]['reference']!=e['grid_reference']:raise ValueError('Grid reference mismatch')
        correct=valid and prediction==e['answer']
        rows.append(dict(id=e['id'],split=e['split'],map_id=e['map_id'],grid_reference=e['grid_reference'],
            grid_prediction=g[e['id']]['prediction'],grid_correct=g[e['id']]['correct'],
            color_reference=e['answer'],color_prediction=prediction,color_valid=valid,color_correct=correct,
            raw_response=r['raw_response'] if r else '',reference_rgb=json.dumps(e['reference_rgb']),
            predicted_rgb=json.dumps(e['swatches'][int(prediction)-1]) if valid else '',
            category=('grid_correct' if g[e['id']]['correct'] else 'grid_wrong')+'_'+('color_correct' if correct else 'color_wrong')))
    table(out/'matched_answers.csv',rows)
    summary={}
    for split in ['development','test']:
        subset=[r for r in rows if r['split']==split];counts=Counter(r['category'] for r in subset)
        summary[split]=dict(n=len(subset),color_correct=sum(r['color_correct'] for r in subset),
            color_accuracy=sum(r['color_correct'] for r in subset)/len(subset),valid=sum(r['color_valid'] for r in subset),
            cross_table={f'grid_{g}_color_{c}':counts[f'grid_{g}_color_{c}'] for g in ['correct','wrong'] for c in ['correct','wrong']},
            per_map={mid:dict(n=len(rs:=[r for r in subset if r['map_id']==mid]),color_correct=sum(r['color_correct'] for r in rs)) for mid in sorted({r['map_id'] for r in subset})})
    summary['notes']=['Six choices: uniform random expectation 1/6. No numerical-reading or geographic claim.',
        'Color response requires marker finding and swatch matching; cross-task association is not a causal separation.',
        'Choices intentionally distinct; success does not establish fine color discrimination. Missing/invalid count incorrect.',
        'Same map panel retained but canvas extended for choices; this changes presentation. Existing sources are exploratory reuse.']
    dump(out/'summary.json',summary);print(json.dumps(summary))

def run(dataset,out):
    import os,sys,importlib.metadata
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
    m=json.loads((dataset/'manifest.json').read_text())
    if out.exists():raise ValueError('Choose new output path')
    if m['experiment']!='color_matching_v1' or m['prompt']!=PROMPT or m['revision']!=REVISION:raise ValueError('Protocol mismatch')
    for e in m['examples']:
        if sha(dataset/e['image'])!=e['image_sha256']:raise ValueError('Image changed')
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    out.mkdir(parents=True)
    meta=dict(status='loading',manifest_sha256=sha(dataset/'manifest.json'),script_sha256=sha(__file__),model=MODEL,revision=REVISION,
        prompt=PROMPT,generation=m['generation'],max_pixels=1003520,seed=20261106,job_id=os.getenv('SLURM_JOB_ID'),
        gpu=torch.cuda.get_device_name(0),python=sys.version,packages={k:importlib.metadata.version(k) for k in ['torch','transformers','pillow','accelerate']})
    dump(out/'run.json',meta);set_seed(20261106)
    processor=AutoProcessor.from_pretrained(MODEL,revision=REVISION,min_pixels=3136,max_pixels=1003520)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
    meta['status']='running';dump(out/'run.json',meta)
    with (out/'responses.jsonl').open('x') as f:
        for e in m['examples']:
            with Image.open(dataset/e['image']) as im:image=im.convert('RGB')
            text=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=PROMPT)])],tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[text],images=[image],return_tensors='pt').to(model.device)
            with torch.inference_mode():generated=model.generate(**inputs,**m['generation'])
            raw=processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
            f.write(json.dumps(dict(id=e['id'],raw_response=raw))+'\n');f.flush();print(e['id'],raw,flush=True)
    meta['status']='finished';dump(out/'run.json',meta)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','evaluate']);p.add_argument('--dataset',type=Path,default=ROOT/'experiments/color-matching-v1');p.add_argument('--source',type=Path,default=ROOT/'experiments/grid-localization-v1');p.add_argument('--output',type=Path);p.add_argument('--grid-run',type=Path,default=ROOT/'analysis/hpc-grid-localization-v1/grid-localization-40939306');a=p.parse_args()
    if a.mode=='prepare':prepare(a.source,a.dataset)
    elif a.output is None:p.error('--output required')
    elif a.mode=='run':run(a.dataset,a.output)
    else:evaluate(a.dataset,a.output,a.grid_run)
