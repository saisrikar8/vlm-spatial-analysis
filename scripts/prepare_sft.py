#!/usr/bin/env python3
"""Verified synthetic SFT targets, grouped splits and readable numerical figures."""
import getopt
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
from experiment_contract import sha,inside
from structured_prototype import SCHEMA,parse_representation,execute
from interval_experiment import bounds
from run_vlm import write_json

SPLITS={'train':64,'dev':8,'test':16}
CONDITIONS=('clear','shared_offset','independent_offset')
PROMPT=('Image 1 is the clean figure with its legend and numerical rules. Image 2 is the same figure with A/B pointer marks. '
        'A/B are at the centers of the gaps in those marks. Read the displayed colors and the written rules. '+SCHEMA)


def color(v):return (int(255*v/8),80,int(255*(8-v)/8))


def canonical(a,b,condition,oa,ob):
    if condition=='shared_offset':return dict(values={'A':dict(base=a,offset='s'),'B':dict(base=b,offset='s')},offsets={'s':[10,20]})
    return dict(values={'A':dict(base=a,offset='s_A'),'B':dict(base=b,offset='s_B')},
        offsets={'s_A':[oa],'s_B':[ob]} if condition=='clear' else {'s_A':[10,20],'s_B':[10,20]})


def prepare(output,counts=None,seed=20261102):
    from PIL import Image,ImageDraw,ImageFont
    output=Path(output);counts=counts or SPLITS
    if output.exists():raise ValueError('Output exists')
    if set(counts)!=set(SPLITS) or any(type(v)!=int or v<1 for v in counts.values()):raise ValueError('Positive source counts required for train/dev/test')
    fontfile=Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    font=ImageFont.truetype(str(fontfile),20) if fontfile.exists() else ImageFont.load_default(size=20)
    large=ImageFont.truetype(str(fontfile),24) if fontfile.exists() else ImageFont.load_default(size=24)
    rng=random.Random(seed);seen=set();excluded=[]
    # Exclude previously inspected underlying fields, not only their A/B selectors.
    old=Path(__file__).resolve().parents[1]/'experiments/controlled-v1/manifest.json'
    if old.exists():
        m=json.loads(old.read_text())
        for e in m['examples']:
            with Image.open(old.parent/e['clean_image']) as im:
                inverse={color(v):v for v in range(1,9)}
                field=[inverse[im.getpixel((55+(i%4)*160+72,65+(i//4)*85+35))] for i in range(12)]
            digest=hashlib.sha256(json.dumps(field,separators=(',',':')).encode()).hexdigest();seen.add(digest);excluded.append(digest)
    output.mkdir(parents=True);(output/'images').mkdir();(output/'references').mkdir();entries=[]
    for split,count in counts.items():
        for j in range(count):
            while True:
                field=[rng.randint(1,8) for _ in range(12)]
                digest=hashlib.sha256(json.dumps(field,separators=(',',':')).encode()).hexdigest()
                if digest not in seen:break
            seen.add(digest);cells=rng.sample(range(12),2);a,b=[field[i] for i in cells];oa,ob=rng.choice([10,20]),rng.choice([10,20])
            group=f'{split}-{j:03d}-{digest[:12]}'
            for condition in CONDITIONS:
                identifier=group+'-'+condition
                im=Image.new('RGB',(640,512),'white');draw=ImageDraw.Draw(im)
                draw.text((18,10),'Colors show the displayed value v',font=font,fill='black')
                for i,v in enumerate(field):
                    x=24+(i%4)*148;y=48+(i//4)*78;draw.rectangle((x,y,x+135,y+65),fill=color(v))
                for v in range(1,9):
                    x=20+(v-1)*75;draw.rectangle((x,290,x+45,310),fill=color(v));draw.text((x+14,316),str(v),font=font,fill='black')
                draw.text((18,355),'True value = v + offset',font=font,fill='black')
                rules=([f'Known offsets: A = {oa}; B = {ob}.'] if condition=='clear' else
                       ['ONE offset s is shared by A and B.','Allowed s: {10, 20}.'] if condition=='shared_offset' else
                       ['A and B have separate offsets.','Each offset is in {10, 20}.'])
                for line,text in enumerate(rules):draw.text((18,391+line*32),text,font=font,fill='black')
                clean=f'images/{identifier}-clean.png';marked=f'images/{identifier}-marked.png';im.save(output/clean)
                points=[]
                for label,cell in zip(('A','B'),cells):
                    x=24+(cell%4)*148+67;y=48+(cell//4)*78+32
                    points.append(dict(label=label,index=cell,pixel=[x,y],displayed=field[cell]))
                    for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        draw.line((x+4*dx,y+4*dy,x+11*dx,y+11*dy),fill='white',width=4)
                        draw.line((x+4*dx,y+4*dy,x+11*dx,y+11*dy),fill='black',width=2)
                    draw.text((x+14,y-16),label,font=large,fill='black',stroke_width=1,stroke_fill='white')
                im.save(output/marked)
                target=canonical(a,b,condition,oa,ob)
                if condition=='clear':feasible=[(a+oa,b+ob)]
                elif condition=='shared_offset':feasible=[(a+s,b+s) for s in (10,20)]
                else:feasible=list(itertools.product((a+10,a+20),(b+10,b+20)))
                reference=dict(id=identifier,condition=condition,field=field,field_sha256=digest,points=points,displayed_a=a,displayed_b=b,
                    target=target,feasible_readings=feasible,ground_truth=dict(true_a=feasible[0][0],true_b=feasible[0][1],true_difference_a_minus_b=feasible[0][0]-feasible[0][1]))
                refpath=f'references/{identifier}.json';write_json(output/refpath,reference)
                entries.append(dict(id=identifier,source_group=group,field_sha256=digest,split=split,condition=condition,clean_image=clean,image=marked,reference=refpath,
                    clean_image_sha256=sha(output/clean),image_sha256=sha(output/marked),reference_sha256=sha(output/refpath),target=target))
    manifest=dict(experiment='structured_sft_v1',seed=seed,source_counts=counts,prompt=PROMPT,entries=entries,
        excluded_previous_field_sha256=sorted(set(excluded)),builder_sha256=sha(Path(__file__)),font_sha256=sha(fontfile) if fontfile.exists() else 'Pillow-default',
        notes=['Known renderer coordinates; source-level splits; all three conditions remain with the same field.',
               'New readable layout and explicit A/B offsets in clear condition; matched unchanged-model baseline required.',
               'Finite additive-offset synthetic task only. No natural-ambiguity, OOD, calibration or general VLM claim.'])
    write_json(output/'manifest.json',manifest);validate(output)
    (output/'preview.html').write_text('<!doctype html><title>Verified SFT split previews</title>'+''.join(f'<h2>{e["id"]}</h2><img src="{e["image"]}">' for split in counts for e in [x for x in entries if x['split']==split][:3]))
    return manifest


def validate(root):
    from PIL import Image
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    if m['experiment']!='structured_sft_v1' or m['prompt']!=PROMPT:raise ValueError('Wrong SFT protocol')
    ids=set();groups={};fields={};forbidden=set(m['excluded_previous_field_sha256'])
    for e in m['entries']:
        if e['id'] in ids or e['split'] not in SPLITS:raise ValueError('Duplicate/unknown split')
        ids.add(e['id'])
        for key in ('image','clean_image','reference'):
            if sha(inside(root,e[key]))!=e[key+'_sha256']:raise ValueError('Input digest mismatch')
        ref=json.loads(inside(root,e['reference']).read_text())
        if ref['id']!=e['id'] or ref['condition']!=e['condition']:raise ValueError('Reference identity mismatch')
        if [p['label'] for p in ref['points']]!=['A','B']:raise ValueError('Point labels mismatch')
        digest=hashlib.sha256(json.dumps(ref['field'],separators=(',',':')).encode()).hexdigest()
        if digest!=e['field_sha256'] or digest in forbidden:raise ValueError('Invalid or previously inspected field')
        if digest in fields and fields[digest]!=e['source_group']:raise ValueError('Field duplicated across source groups')
        fields[digest]=e['source_group']
        group=groups.setdefault(e['source_group'],dict(split=e['split'],field=digest,conditions=set()))
        if group['split']!=e['split'] or group['field']!=digest or e['condition'] in group['conditions']:raise ValueError('Source leakage or duplicate condition')
        group['conditions'].add(e['condition'])
        target,error=parse_representation(json.dumps(e['target']))
        if error or target!=ref['target']:raise ValueError('Invalid target')
        calculated=execute(target)
        if any(calculated[q]!=bounds(ref,q) for q in ('a','b','difference')):raise ValueError('Oracle target does not match feasible bounds')
        for key in ('clean_image','image'):
            with Image.open(inside(root,e[key])) as image:
                if image.size!=(640,512):raise ValueError('Invalid rendered dimensions')
                for point in ref['points']:
                    if image.getpixel(tuple(point['pixel']))!=color(point['displayed']):raise ValueError('Target pixel was hidden or misregistered')
    if any(g['conditions']!=set(CONDITIONS) for g in groups.values()):raise ValueError('Incomplete source condition family')
    for split,count in m['source_counts'].items():
        if sum(g['split']==split for g in groups.values())!=count:raise ValueError('Split source count mismatch')
    return m


if __name__=='__main__':
    opts,rest=getopt.gnu_getopt(sys.argv[1:],'',['output=','validate='])
    a=dict(opts)
    if rest:raise ValueError('Unexpected arguments')
    if '--validate' in a:print('Verified examples:',len(validate(a['--validate'])['entries']))
    else:print('Prepared examples:',len(prepare(a.get('--output','experiments/structured-sft-v1'))['entries']))
