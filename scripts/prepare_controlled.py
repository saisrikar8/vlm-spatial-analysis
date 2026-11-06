#!/usr/bin/env python3
"""Render a small, explicitly specified uncertainty diagnostic, not natural ambiguity."""
import csv
import getopt
import itertools
import json
from pathlib import Path
import random
import sys
from PIL import Image, ImageDraw
from experiment_contract import sha


def prepare(output, sources=6, seed=42):
    output=Path(output)
    if output.exists():raise ValueError('Output exists')
    if sources<1 or seed<0:raise ValueError('Invalid sources/seed')
    output.mkdir(parents=True);(output/'points').mkdir();(output/'clean').mkdir()
    rng=random.Random(seed);entries=[];truths=[]
    for source in range(sources):
        # Related conditions share a source group; never split them across train/test.
        base=[rng.randint(1,8) for _ in range(12)]
        cells=rng.sample(range(12),2)
        for condition in ('clear','shared_offset','independent_offset'):
            mid=f'source-{source:03d}-{condition}';eid=mid+'-pair-000'
            offsets=[rng.choice([10,20]) for _ in range(12)]
            if condition=='shared_offset':offsets=[offsets[0]]*12
            values=[x+b for x,b in zip(base,offsets)]
            image=Image.new('RGB',(750,475),'white');draw=ImageDraw.Draw(image)
            draw.text((25,15),'Controlled numerical map: colors encode the displayed integer v',fill='black')
            for i,v in enumerate(base):
                x=55+(i%4)*160;y=65+(i//4)*85
                draw.rectangle((x,y,x+145,y+70),fill=(int(255*v/8),80,int(255*(8-v)/8)))
            for v in range(1,9):
                x=45+(v-1)*85
                draw.rectangle((x,330,x+65,350),fill=(int(255*v/8),80,int(255*(8-v)/8)))
                draw.text((x+20,355),str(v),fill='black')
            if condition=='clear':
                rules=['True value = v + the cell offset. Offsets by row:',
                       ' / '.join(', '.join(map(str,offsets[i:i+4])) for i in range(0,12,4))]
            elif condition=='shared_offset':
                rules=['True value = v + b. ONE offset b is shared by ALL cells.',
                       'b is 10 or 20, each with probability 1/2, independently of the displayed colors.']
            else:
                rules=['True value = v + b_cell. Each cell has its OWN independent offset.',
                       'Each offset is 10 or 20 with probability 1/2, independently of the displayed colors.']
            for i,line in enumerate(rules):draw.text((20,395+i*20),line,fill='black')
            clean=f'clean/{mid}.png';image.save(output/clean)
            points=[]
            for label,cell in zip(('A','B'),cells):
                x=55+(cell%4)*160+72;y=65+(cell//4)*85+35
                points.append(dict(label=label,index=[cell//4,cell%4],flat_index=cell,value=values[cell],pixel=[x,y],units='controlled units'))
                for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    draw.line((x+4*dx,y+4*dy,x+12*dx,y+12*dy),fill='white',width=4)
                    draw.line((x+4*dx,y+4*dy,x+12*dx,y+12*dy),fill='black',width=2)
                draw.text((x+16,y-15),label,fill='black',stroke_width=1,stroke_fill='white')
            a,b=[p['value'] for p in points]
            truth=dict(true_a=a,true_b=b,true_difference_a_minus_b=a-b,true_ratio_a_over_b=a/b,true_a_greater_b=a>b,ratio_valid=True)
            candidates=[(a,b)] if condition=='clear' else [(base[cells[0]]+o,base[cells[1]]+o) for o in (10,20)] if condition=='shared_offset' else list(itertools.product([base[cells[0]]+o for o in (10,20)],[base[cells[1]]+o for o in (10,20)]))
            marked=f'points/{eid}.png';reference=f'points/{eid}_reference.json';image.save(output/marked)
            (output/reference).write_text(json.dumps(dict(id=eid,points=points,ground_truth=truth,condition=condition,
                feasible_readings=candidates,probabilities=[1/len(candidates)]*len(candidates),registration='Exact tile centers from this renderer'),indent=2)+'\n')
            entries.append(dict(id=eid,map_id=mid,source_group=f'controlled-source-{source:03d}',condition=condition,
                image=marked,clean_image=clean,reference=reference,image_sha256=sha(output/marked),clean_image_sha256=sha(output/clean),reference_sha256=sha(output/reference),display_scale=17.))
            truths.append(dict(example_id=eid,map_id=mid,**truth))
    with (output/'ground_truth.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(truths[0]));writer.writeheader();writer.writerows(truths)
    manifest=dict(format_version=3,experiment='controlled_offset_diagnostic',relationship_direction='A_relative_to_B',
        queries=['difference','ordering'],primary_query='difference',seed=seed,example_count=len(entries),map_count=len(entries),
        ratio_min_abs=1e-8,ground_truth_sha256=sha(output/'ground_truth.csv'),acceptance=dict(status='validated',
        evidence='Known renderer tile coordinates; explicit generative offset rules; exact colors and declared prior.'),
        builder_sha256=sha(Path(__file__)),limitations=['Explicit rule-following diagnostic, not evidence of naturally occurring visual ambiguity.','Development dataset; fresh source groups required for a confirmatory study.'],examples=entries)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'preview.html').write_text('<!doctype html><title>Controlled development inputs</title>'+''.join(f'<h2>{e["id"]}</h2><img src="{e["image"]}">' for e in entries))
    return manifest


if __name__=='__main__':
    opts,rest=getopt.gnu_getopt(sys.argv[1:],'',['output=','sources=','seed='])
    a=dict(opts)
    if rest:raise ValueError('Unexpected arguments')
    m=prepare(a.get('--output','experiments/controlled-v1'),int(a.get('--sources',6)),int(a.get('--seed',42)))
    print('Rendered',m['example_count'],'examples')
