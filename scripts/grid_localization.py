#!/usr/bin/env python3
"""Prepare, run, and score a coarse marker-localization diagnostic."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
PROMPT = ('The map has columns A through F from left to right and rows 1 through 4 '
          'from top to bottom. Which grid cell contains the CENTER of the red circular '
          'marker? Return only the cell label, for example C2. Do not read numerical '
          'map values or geographic coordinates.')
REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(out):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    if out.exists():
        raise ValueError('Choose a new output directory')
    records = json.loads((ROOT/'data/ilamb-gridded-v2/alignment.json').read_text())['records']
    chosen=[]; groups=set()
    for rec in records:
        if rec['source_group'] not in groups:
            chosen.append(rec); groups.add(rec['source_group'])
        if len(chosen)==10: break
    if len(chosen)!=10: raise ValueError('Need ten distinct source groups')
    rng=random.Random(20261105)
    plans=[]
    for mi,rec in enumerate(chosen):
        source=ROOT/'experiments/random-pairs/clean'/f"map-{rec['index']:03d}-{rec['id']}.png"
        image=Image.open(source).convert('RGB')
        if image.size!=(750,475): raise ValueError('Unexpected image layout')
        arr=np.asarray(image)
        # Grid covers the known map panel, not its legend. No geographic claim.
        cells=[]
        for row in range(4):
            for col in range(6):
                candidates=[]
                for y in range(int(row*375/4)+18,int((row+1)*375/4)-18,4):
                    for x in range(col*125+18,(col+1)*125-18,4):
                        patch=arr[y-2:y+3,x-2:x+3].astype(float)
                        if (np.ptp(patch,axis=2)>8).all() and patch.std(axis=(0,1)).max()<18:
                            candidates.append((x,y))
                if candidates: cells.append((row,col,candidates))
        if len(cells)<4: raise ValueError('Insufficient visible colored cells')
        for pi,(row,col,candidates) in enumerate(rng.sample(cells,4)):
            x,y=rng.choice(candidates)
            plans.append(dict(id=f'map{mi:02d}-point{pi}',source_image=str(source.relative_to(ROOT)),
                source_sha256=sha(source),source_group=rec['source_group'],map_id=rec['id'],
                split='development' if mi<2 else 'test',x=x,y=y,answer=f'{chr(65+col)}{row+1}'))
    out.mkdir(parents=True); (out/'images').mkdir()
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
    for e in plans:
        source=Image.open(ROOT/e['source_image']).convert('RGB')
        canvas=Image.new('RGB',(795,520),'white');canvas.paste(source,(45,45))
        draw=ImageDraw.Draw(canvas)
        for col in range(7):
            x=45+col*125;draw.line((x,45,x,420),fill='#555555',width=1)
        for row in range(5):
            y=45+round(row*375/4);draw.line((45,y,795,y),fill='#555555',width=1)
        for col in range(6):draw.text((45+col*125+62,23),chr(65+col),font=font,fill='black',anchor='mm')
        for row in range(4):draw.text((23,45+(row+.5)*375/4),str(row+1),font=font,fill='black',anchor='mm')
        x,y=e['x']+45,e['y']+45
        draw.ellipse((x-8,y-8,x+8,y+8),outline='white',width=4)
        draw.ellipse((x-7,y-7,x+7,y+7),outline='red',width=3)
        e['image']=f"images/{e['id']}.png";canvas.save(out/e['image']);e['image_sha256']=sha(out/e['image'])
    manifest=dict(experiment='grid_localization_v1',seed=20261105,prompt=PROMPT,model=MODEL,revision=REVISION,
        generation=dict(do_sample=False,max_new_tokens=16),grid=dict(columns=6,rows=4,source_box=[0,0,750,375]),
        scope='Coarse marker localization only. Colored-pixel screen is not land or geographic-registration validation.',
        scoring='Exact cell accuracy; missing/malformed responses count incorrect. Report test by map and overall; development separately.',
        examples=plans)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (out/'answer_key.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(plans[0]));w.writeheader();w.writerows(plans)
    (out/'preview.html').write_text('<html><body><h1>Grid localization examples</h1>'+''.join(
        f'<div><p>{e["id"]} — {e["split"]}</p><img width="795" src="{e["image"]}"></div>' for e in plans)+'</body></html>')
    print(f'Prepared {len(plans)} examples in {out}')

def run(dataset,out):
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
    manifest=json.loads((dataset/'manifest.json').read_text())
    if out.exists():raise ValueError('Choose a new run directory')
    for e in manifest['examples']:
        if sha(dataset/e['image'])!=e['image_sha256']:raise ValueError('Image changed')
    if manifest['prompt']!=PROMPT or manifest['revision']!=REVISION:raise ValueError('Protocol changed')
    out.mkdir(parents=True)
    meta=dict(status='loading',manifest_sha256=sha(dataset/'manifest.json'),script_sha256=sha(__file__),
              model=MODEL,revision=REVISION,prompt=PROMPT,generation=manifest['generation'],max_pixels=1003520)
    (out/'run.json').write_text(json.dumps(meta,indent=2))
    set_seed(20261105)
    processor=AutoProcessor.from_pretrained(MODEL,revision=REVISION,min_pixels=3136,max_pixels=1003520)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
    rows=[]
    with (out/'responses.jsonl').open('x') as f:
        for e in manifest['examples']:
            with Image.open(dataset/e['image']) as im:image=im.convert('RGB')
            text=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=PROMPT)])],tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[text],images=[image],return_tensors='pt').to(model.device)
            with torch.inference_mode():generated=model.generate(**inputs,**manifest['generation'])
            raw=processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
            prediction=raw.strip();valid=re.fullmatch('[A-F][1-4]',prediction) is not None
            row=dict(id=e['id'],split=e['split'],map_id=e['map_id'],reference=e['answer'],prediction=prediction,valid=valid,correct=valid and prediction==e['answer'],raw_response=raw)
            f.write(json.dumps(row)+'\n');f.flush();rows.append(row)
    with (out/'answers.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={s:dict(n=len(r:=[x for x in rows if x['split']==s]),correct=sum(x['correct'] for x in r),accuracy=sum(x['correct'] for x in r)/len(r)) for s in ['development','test']}
    from collections import Counter
    test_rows=[r for r in rows if r['split']=='test']
    counts=Counter(r['reference'] for r in test_rows)
    summary['test']['label_counts']=dict(sorted(counts.items()))
    summary['test']['majority_label_baseline']=max(counts.values())/len(test_rows)
    summary['test']['uniform_random_baseline']=1/24
    summary['test']['per_map']={mid:dict(n=len(rs:=[r for r in test_rows if r['map_id']==mid]),correct=sum(r['correct'] for r in rs),accuracy=sum(r['correct'] for r in rs)/len(rs)) for mid in sorted({r['map_id'] for r in test_rows})}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    meta['status']='finished';(out/'run.json').write_text(json.dumps(meta,indent=2));print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--dataset',type=Path,default=ROOT/'experiments/grid-localization-v1');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.dataset)
    else:
        if a.output is None:p.error('--output required')
        run(a.dataset,a.output)
