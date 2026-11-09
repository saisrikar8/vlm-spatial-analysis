#!/usr/bin/env python3
"""Source-disjoint color-reading LoRA dataset, with full/crop paired supervision."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
from color_matching import ROOT,MODEL,REVISION,sha,dump,table
from plain_patch_matching import PROMPT

SEED=20261108


def validate(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    if m['experiment']!='color_sft_v1' or m['prompt']!=PROMPT:raise ValueError('Wrong protocol')
    seen=set();groups={};hashes={};points={}
    for e in m['entries']:
        if e['id'] in seen:raise ValueError('Duplicate example')
        seen.add(e['id'])
        if groups.setdefault(e['source_group'],e['split'])!=e['split']:raise ValueError('Source leakage')
        if hashes.setdefault(e['source_sha256'],e['split'])!=e['split']:raise ValueError('Image leakage')
        if e['source_group'] in m['excluded_diagnostic_groups']:raise ValueError('Old diagnostic group reused')
        if e['answer'] not in '123456' or len(e['answer'])!=1:raise ValueError('Bad label')
        if e['swatches'][int(e['answer'])-1]!=e['reference_rgb']:raise ValueError('Wrong color target')
        if sha(root/e['image'])!=e['image_sha256']:raise ValueError('Image changed')
        if sha(root/e['source_image'])!=e['source_sha256']:raise ValueError('Source changed')
        points.setdefault(e['point_id'],[]).append(e)
    for es in points.values():
        if len(es)!=2 or {e['condition'] for e in es}!={'full','crop'} or len({e['answer'] for e in es})!=1:raise ValueError('Bad pair')
    expected={'train':288,'dev':32,'test':96}
    if dict(Counter(e['split'] for e in m['entries']))!=expected:raise ValueError('Split size changed')
    if dict(Counter(groups.values()))!={'train':6,'dev':2,'test':3}:raise ValueError('Source split changed')
    return m


def prepare(output):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    out=Path(output)
    if out.exists():raise ValueError('Choose a new dataset path')
    alignment=ROOT/'data/ilamb-gridded-v2/alignment.json';records=json.loads(alignment.read_text())['records']
    old=ROOT/'experiments/grid-localization-v1/manifest.json';excluded={e['source_group'] for e in json.loads(old.read_text())['examples']}
    groups={}
    for r in records:
        if r['source_group'] not in excluded:groups.setdefault(r['source_group'],[]).append(r)
    keys=sorted(groups);rng=random.Random(SEED);rng.shuffle(keys)
    if len(keys)!=11:raise ValueError('Expected 11 remaining source groups')
    font_path='/System/Library/Fonts/Supplemental/Arial.ttf'
    try:font=ImageFont.truetype(font_path,22)
    except OSError:font=ImageFont.truetype('DejaVuSans.ttf',22)
    plans=[];source_records=[]
    for index,group in enumerate(keys):
        split='train' if index<6 else 'dev' if index<8 else 'test';count={'train':24,'dev':8,'test':16}[split]
        r=sorted(groups[group],key=lambda x:x['id'])[0]
        source=ROOT/'experiments/random-pairs/clean'/f"map-{r['index']:03d}-{r['id']}.png"
        image=Image.open(source).convert('RGB')
        if image.size!=(750,475):raise ValueError('Unexpected layout')
        a=np.asarray(image);candidates=[]
        for y in range(18,357,4):
            for x in range(18,732,4):
                if min(x%125,125-x%125)<18 or min(y%(375/4),375/4-y%(375/4))<18:continue
                p=a[y-2:y+3,x-2:x+3].astype(float)
                if (np.ptp(p,axis=2)>8).all() and p.std(axis=(0,1)).max()<18:candidates.append((x,y))
        rng.shuffle(candidates);selected=[]
        for x,y in candidates:
            if all((x-u)**2+(y-v)**2>=24**2 for u,v in selected):selected.append((x,y))
            if len(selected)==count:break
        if len(selected)!=count:raise ValueError('Not enough separated colored points')
        pool=np.unique(a[:375].reshape(-1,3),axis=0).astype(float);pool=pool[np.ptp(pool,axis=1)>8]
        labels=[str(i%6+1) for i in range(count)];rng.shuffle(labels)
        srcpath=f'sources/source-{index:02d}.png';source_records.append(dict(path=srcpath,local_path=str(source),sha256=sha(source),source_group=group,split=split,map_id=r['id']))
        for pi,((x,y),label) in enumerate(zip(selected,labels)):
            rgb=a[y,x].astype(float);colors=[rgb]
            for _ in range(5):
                distances=np.min(np.stack([np.linalg.norm(pool-c,axis=1) for c in colors]),axis=0);j=int(np.argmax(distances))
                if distances[j]<35:raise ValueError('Insufficient color separation')
                colors.append(pool[j])
            choices=colors[1:];rng.shuffle(choices);choices.insert(int(label)-1,rgb);choices=[[int(v) for v in c] for c in choices]
            plans.append(dict(point_id=f'source{index:02d}-point{pi:02d}',split=split,map_id=r['id'],source_group=group,
                source_image=srcpath,source_sha256=sha(source),x=x,y=y,answer=label,swatches=choices,reference_rgb=rgb.astype(int).tolist()))
    out.mkdir(parents=True);(out/'images').mkdir();(out/'sources').mkdir()
    for s in source_records:shutil.copyfile(s['local_path'],out/s['path'])
    entries=[]
    for p in plans:
        source=Image.open(out/p['source_image']).convert('RGB');full=Image.new('RGB',(795,625),'white');full.paste(source,(45,45));d=ImageDraw.Draw(full)
        for col in range(7):d.line((45+col*125,45,45+col*125,420),fill='#555555',width=1)
        for row in range(5):d.line((45,45+round(row*375/4),795,45+round(row*375/4)),fill='#555555',width=1)
        for col in range(6):d.text((45+col*125+62,23),chr(65+col),font=font,fill='black',anchor='mm')
        for row in range(4):d.text((23,45+(row+.5)*375/4),str(row+1),font=font,fill='black',anchor='mm')
        x,y=p['x']+45,p['y']+45;d.ellipse((x-8,y-8,x+8,y+8),outline='white',width=4);d.ellipse((x-7,y-7,x+7,y+7),outline='red',width=3)
        for i,rgb in enumerate(p['swatches']):
            left=35+i*128;d.rectangle((left,535,left+85,585),fill=tuple(rgb),outline='black',width=1);d.text((left+42,604),str(i+1),font=font,fill='black',anchor='mm')
        crop=full.copy();crop.paste('white',(0,0,795,520));crop.paste(full.crop((x-32,y-32,x+33,y+33)).resize((325,325),Image.Resampling.NEAREST),(235,90))
        assert full.getpixel((x,y))==crop.getpixel((397,252))==tuple(p['reference_rgb'])
        assert full.crop((0,520,795,625)).tobytes()==crop.crop((0,520,795,625)).tobytes()
        for condition,im in [('full',full),('crop',crop)]:
            e=dict(p,id=p['point_id']+'-'+condition,condition=condition,image=f"images/{p['point_id']}-{condition}.png")
            im.save(out/e['image']);e['image_sha256']=sha(out/e['image']);entries.append(e)
    manifest=dict(experiment='color_sft_v1',seed=SEED,prompt=PROMPT,model=MODEL,revision=REVISION,excluded_diagnostic_groups=sorted(excluded),
        alignment_sha256=sha(alignment),diagnostic_manifest_sha256=sha(old),builder_sha256=sha(__file__),font=font_path,
        selection='Unused by recent diagnostic series, from existing archive; archived original benchmark may include these sources. All colored-pixel truth from source images, no geographic claim.',
        split_counts=dict(Counter(e['split'] for e in entries)),source_records=source_records,entries=entries)
    dump(out/'manifest.json',manifest);table(out/'answer_key.csv',entries)
    preview=[next(e for e in entries if e['split']==split and e['condition']==c) for split in ['train','dev','test'] for c in ['full','crop']]
    (out/'preview.html').write_text('<html><body>'+''.join(f'<p>{e["id"]} {e["split"]}</p><img src="{e["image"]}">' for e in preview)+'</body></html>')
    validate(out);print(json.dumps(manifest['split_counts']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/color-sft-v1');p.add_argument('--validate',type=Path);a=p.parse_args()
    if a.validate:print(validate(a.validate)['split_counts'])
    else:prepare(a.output)
