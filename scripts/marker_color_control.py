#!/usr/bin/env python3
"""Paired red/magenta marker-color intervention with a common neutral prompt."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
from color_matching import ROOT,MODEL,REVISION,sha,dump,table

PROMPT=('Which numbered color swatch at the bottom best matches the underlying map color '
        'at the CENTER of the circular marker? Match the map color, not the marker outline '
        'or the gray grid lines. Return only one digit from 1 to 6. '
        'Do not read numerical map values or report the grid cell.')
COLORS={'red':[255,0,0],'magenta':[255,0,255]}

def prepare(source,out):
    import numpy as np
    from PIL import Image,ImageDraw
    if out.exists():raise ValueError('Choose a new dataset directory')
    m=json.loads((source/'manifest.json').read_text());plans=[]
    # Alternate condition order across points, independent of model answers.
    for index,e in enumerate(m['examples']):
        p=source/e['image']
        if sha(p)!=e['image_sha256']:raise ValueError('Source image changed')
        im=Image.open(p).convert('RGB');a=np.asarray(im)
        mask=Image.new('L',im.size,0);d=ImageDraw.Draw(mask);x,y=e['x']+45,e['y']+45
        d.ellipse((x-7,y-7,x+7,y+7),outline=255,width=3)
        mask=np.asarray(mask)>0
        if not (a[mask]==[255,0,0]).all():raise ValueError('Marker mask mismatch')
        if im.getpixel((x,y))!=tuple(e['reference_rgb']):raise ValueError('Center mismatch')
        order=['red','magenta'] if index%2==0 else ['magenta','red']
        for condition in order:
            plans.append(dict(id=e['id']+'-'+condition,point_id=e['id'],condition=condition,
                split=e['split'],map_id=e['map_id'],source_group=e['source_group'],x=e['x'],y=e['y'],
                answer=e['answer'],swatches=e['swatches'],reference_rgb=e['reference_rgb'],
                source_image=e['image'],source_sha256=e['image_sha256'],marker_rgb=COLORS[condition]))
    out.mkdir(parents=True);(out/'images').mkdir()
    for e in plans:
        with Image.open(source/e['source_image']) as original:
            im=original.convert('RGB');a=np.array(im)
        maskim=Image.new('L',im.size,0);x,y=e['x']+45,e['y']+45
        ImageDraw.Draw(maskim).ellipse((x-7,y-7,x+7,y+7),outline=255,width=3)
        mask=np.asarray(maskim)>0;b=a.copy();b[mask]=e['marker_rgb']
        assert (b[~mask]==a[~mask]).all()
        assert b[y,x].tolist()==e['reference_rgb']
        e['changed_pixels']=int(np.any(a!=b,axis=2).sum())
        e['image']=f"images/{e['id']}.png";Image.fromarray(b).save(out/e['image']);e['image_sha256']=sha(out/e['image'])
    dump(out/'manifest.json',dict(experiment='marker_color_control_v1',source_manifest_sha256=sha(source/'manifest.json'),
        prompt=PROMPT,model=MODEL,revision=REVISION,generation=dict(do_sample=False,max_new_tokens=16),seed=20261106,
        conditions=COLORS,design='Same neutral prompt for both colors; same points, choices, positions, center pixels and model. Alternate condition order by point.',
        primary='All 32 reused test points, paired red vs magenta exact color-choice accuracy; missing/invalid incorrect. Development separate; exploratory follow-up.',examples=plans))
    table(out/'answer_key.csv',plans)
    (out/'preview.html').write_text('<html><body>'+''.join(f'<p>{e["id"]}</p><img src="{e["image"]}">' for e in plans)+'</body></html>')
    print('Prepared',len(plans),'images; only ring pixels changed')

def evaluate(dataset,out):
    m=json.loads((dataset/'manifest.json').read_text());meta=json.loads((out/'run.json').read_text())
    if meta['manifest_sha256']!=sha(dataset/'manifest.json'):raise ValueError('Manifest mismatch')
    raw=[json.loads(s) for s in (out/'responses.jsonl').read_text().splitlines()];byid={r['id']:r for r in raw}
    if len(byid)!=len(raw) or set(byid)-{e['id'] for e in m['examples']}:raise ValueError('Unknown/duplicate responses')
    rows=[]
    for e in m['examples']:
        r=byid.get(e['id']);p=r['raw_response'].strip() if r else '';valid=re.fullmatch('[1-6]',p) is not None
        rgb=e['swatches'][int(p)-1] if valid else None
        red_like=rgb is not None and rgb[0]>=180 and rgb[1]<=60 and rgb[2]<=60
        rows.append(dict(id=e['id'],point_id=e['point_id'],condition=e['condition'],split=e['split'],map_id=e['map_id'],
            reference=e['answer'],prediction=p,valid=valid,correct=valid and p==e['answer'],
            selected_rgb=json.dumps(rgb),selected_red_like=red_like,raw_response=r['raw_response'] if r else ''))
    table(out/'answers.csv',rows);pairs=[]
    ids=list(dict.fromkeys(e['point_id'] for e in m['examples']))
    for point_id in ids:
        r,mg=([x for x in rows if x['point_id']==point_id and x['condition']==c][0] for c in ['red','magenta'])
        pairs.append(dict(point_id=point_id,split=r['split'],map_id=r['map_id'],reference=r['reference'],
            red_prediction=r['prediction'],magenta_prediction=mg['prediction'],red_correct=r['correct'],magenta_correct=mg['correct'],
            changed_answer=r['prediction']!=mg['prediction'],red_selected_red=r['selected_red_like'],magenta_selected_red=mg['selected_red_like']))
    table(out/'paired_answers.csv',pairs);summary={}
    for split in ['development','test']:
        ps=[r for r in pairs if r['split']==split];summary[split]=dict(n=len(ps),
            red_correct=sum(r['red_correct'] for r in ps),magenta_correct=sum(r['magenta_correct'] for r in ps),
            red_accuracy=sum(r['red_correct'] for r in ps)/len(ps),magenta_accuracy=sum(r['magenta_correct'] for r in ps)/len(ps),
            wrong_to_correct=sum(not r['red_correct'] and r['magenta_correct'] for r in ps),
            correct_to_wrong=sum(r['red_correct'] and not r['magenta_correct'] for r in ps),
            changed_answers=sum(r['changed_answer'] for r in ps),red_selected_red=sum(r['red_selected_red'] for r in ps),
            magenta_selected_red=sum(r['magenta_selected_red'] for r in ps),
            per_map={mid:dict(n=len(rs:=[r for r in ps if r['map_id']==mid]),red_correct=sum(r['red_correct'] for r in rs),magenta_correct=sum(r['magenta_correct'] for r in rs)) for mid in sorted({r['map_id'] for r in ps})})
    summary['notes']=['Exploratory reused points, no significance claim. Common neutral prompt differs from original red-naming prompt.',
        'Only colored ring pixels change; white border, center, swatches and positions unchanged.',
        'Improvement would show sensitivity to marker presentation, not uniquely prove copying the outline color.',
        'Magenta may have no matching swatch; distinguishability and marker visibility also change.']
    dump(out/'summary.json',summary);print(json.dumps(summary))

def run(dataset,out):
    import os,sys,importlib.metadata
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
    m=json.loads((dataset/'manifest.json').read_text())
    if out.exists():raise ValueError('Choose a new run directory')
    if m['experiment']!='marker_color_control_v1' or m['prompt']!=PROMPT or m['revision']!=REVISION:raise ValueError('Protocol changed')
    for e in m['examples']:
        if sha(dataset/e['image'])!=e['image_sha256']:raise ValueError('Image changed')
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    out.mkdir(parents=True)
    meta=dict(status='loading',manifest_sha256=sha(dataset/'manifest.json'),script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('color_matching.py')),
        model=MODEL,revision=REVISION,prompt=PROMPT,generation=m['generation'],max_pixels=1003520,seed=m['seed'],
        job_id=os.getenv('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(0),python=sys.version,
        packages={p:importlib.metadata.version(p) for p in ['torch','transformers','pillow','accelerate']})
    dump(out/'run.json',meta);set_seed(m['seed'])
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
    meta['status']='finished';dump(out/'run.json',meta);evaluate(dataset,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','evaluate']);p.add_argument('--dataset',type=Path,default=ROOT/'experiments/marker-color-control-v1');p.add_argument('--source',type=Path,default=ROOT/'experiments/color-matching-v1');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.mode=='prepare':prepare(a.source,a.dataset)
    elif a.output is None:p.error('--output required')
    elif a.mode=='run':run(a.dataset,a.output)
    else:evaluate(a.dataset,a.output)
