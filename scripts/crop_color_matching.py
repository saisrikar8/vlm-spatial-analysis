#!/usr/bin/env python3
"""Paired full-map vs enlarged-region color matching on frozen points."""
import argparse
import json
from pathlib import Path
import re
from color_matching import ROOT,MODEL,REVISION,sha,dump,table
from marker_color_control import PROMPT


def prepare(source,out):
    from PIL import Image
    if out.exists():raise ValueError('Choose a new dataset directory')
    m=json.loads((source/'manifest.json').read_text());plans=[]
    out.mkdir(parents=True);(out/'images').mkdir()
    for index,e in enumerate(m['examples']):
        path=source/e['image']
        if sha(path)!=e['image_sha256']:raise ValueError('Source image changed')
        with Image.open(path) as im:original=im.convert('RGB')
        if original.size!=(795,625):raise ValueError('Unexpected canvas')
        x,y=e['x']+45,e['y']+45
        box=[x-32,y-32,x+33,y+33]
        if box[0]<0 or box[1]<0 or box[2]>795 or box[3]>520:raise ValueError('Crop outside map panel')
        region=original.crop(box).resize((325,325),Image.Resampling.NEAREST)
        crop=original.copy();crop.paste('white',(0,0,795,520));crop.paste(region,(235,90))
        assert crop.crop((0,520,795,625)).tobytes()==original.crop((0,520,795,625)).tobytes()
        assert original.getpixel((x,y))==crop.getpixel((397,252))==tuple(e['reference_rgb'])
        order=['full','crop'] if index%2==0 else ['crop','full']
        for condition in order:
            row=dict(id=e['id']+'-'+condition,point_id=e['id'],condition=condition,split=e['split'],map_id=e['map_id'],
                source_group=e['source_group'],answer=e['answer'],swatches=e['swatches'],reference_rgb=e['reference_rgb'],
                source_image=e['image'],source_sha256=e['image_sha256'],crop_box=box,
                image=f"images/{e['id']}-{condition}.png")
            (original if condition=='full' else crop).save(out/row['image']);row['image_sha256']=sha(out/row['image']);plans.append(row)
    dump(out/'manifest.json',dict(experiment='crop_color_matching_v1',source_manifest_sha256=sha(source/'manifest.json'),
        prompt=PROMPT,model=MODEL,revision=REVISION,generation=dict(do_sample=False,max_new_tokens=16),seed=20261106,
        design='Paired full map vs 65x65 marked region resized 5x nearest-neighbor. Same 795x625 canvas and six unchanged swatches. Alternate task order.',
        primary='All 32 reused test points: full vs crop color-choice accuracy; invalid/missing incorrect. Development separate.',examples=plans))
    table(out/'answer_key.csv',plans)
    (out/'preview.html').write_text('<html><body>'+''.join(f'<p>{e["id"]}</p><img src="{e["image"]}">' for e in plans)+'</body></html>')
    print('Prepared',len(plans),'images; choices and center RGB verified unchanged.')


def evaluate(dataset,out):
    m=json.loads((dataset/'manifest.json').read_text());meta=json.loads((out/'run.json').read_text())
    if meta['manifest_sha256']!=sha(dataset/'manifest.json'):raise ValueError('Manifest mismatch')
    raw=[json.loads(s) for s in (out/'responses.jsonl').read_text().splitlines()];byid={r['id']:r for r in raw}
    if len(byid)!=len(raw) or set(byid)-{e['id'] for e in m['examples']}:raise ValueError('Unknown/duplicate responses')
    rows=[]
    for e in m['examples']:
        rawtext=byid.get(e['id'],{}).get('raw_response','');prediction=rawtext.strip();valid=re.fullmatch('[1-6]',prediction) is not None
        rows.append(dict(id=e['id'],point_id=e['point_id'],condition=e['condition'],split=e['split'],map_id=e['map_id'],
            reference=e['answer'],prediction=prediction,valid=valid,correct=valid and prediction==e['answer'],raw_response=rawtext))
    table(out/'answers.csv',rows);pairs=[]
    for point_id in dict.fromkeys(e['point_id'] for e in m['examples']):
        f,c=([r for r in rows if r['point_id']==point_id and r['condition']==condition][0] for condition in ['full','crop'])
        pairs.append(dict(point_id=point_id,split=f['split'],map_id=f['map_id'],reference=f['reference'],
            full_prediction=f['prediction'],crop_prediction=c['prediction'],full_valid=f['valid'],crop_valid=c['valid'],
            full_correct=f['correct'],crop_correct=c['correct'],changed_answer=f['prediction']!=c['prediction']))
    table(out/'paired_answers.csv',pairs);summary={}
    for split in ['development','test']:
        ps=[r for r in pairs if r['split']==split]
        summary[split]=dict(n=len(ps),full_correct=sum(r['full_correct'] for r in ps),crop_correct=sum(r['crop_correct'] for r in ps),
            full_valid=sum(r['full_valid'] for r in ps),crop_valid=sum(r['crop_valid'] for r in ps),
            full_accuracy=sum(r['full_correct'] for r in ps)/len(ps),crop_accuracy=sum(r['crop_correct'] for r in ps)/len(ps),
            wrong_to_correct=sum(not r['full_correct'] and r['crop_correct'] for r in ps),
            correct_to_wrong=sum(r['full_correct'] and not r['crop_correct'] for r in ps),changed_answers=sum(r['changed_answer'] for r in ps),
            per_map={mid:dict(n=len(rs:=[r for r in ps if r['map_id']==mid]),full_correct=sum(r['full_correct'] for r in rs),crop_correct=sum(r['crop_correct'] for r in rs)) for mid in sorted({r['map_id'] for r in ps})})
    summary['notes']=['Exploratory reused points, no significance claim. Same prompt/model/canvas/choices.',
        'Crop enlarges the marked region and removes surrounding context; not a pure scale-only intervention.',
        'Red outline and white border also enlarge. Nearest-neighbor scaling preserves RGB without added detail.',
        'This is six-choice color matching, not numerical legend reading or geographic localization.']
    dump(out/'summary.json',summary);print(json.dumps(summary))


def run(dataset,out):
    import os,sys,importlib.metadata
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
    m=json.loads((dataset/'manifest.json').read_text())
    if out.exists():raise ValueError('Choose a new run directory')
    if m['experiment']!='crop_color_matching_v1' or m['prompt']!=PROMPT or m['revision']!=REVISION:raise ValueError('Protocol changed')
    for e in m['examples']:
        if sha(dataset/e['image'])!=e['image_sha256']:raise ValueError('Image changed')
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    out.mkdir(parents=True)
    meta=dict(status='loading',manifest_sha256=sha(dataset/'manifest.json'),script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('color_matching.py')),prompt_module_sha256=sha(Path(__file__).with_name('marker_color_control.py')),
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
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','evaluate']);p.add_argument('--dataset',type=Path,default=ROOT/'experiments/crop-color-matching-v1');p.add_argument('--source',type=Path,default=ROOT/'experiments/color-matching-v1');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.mode=='prepare':prepare(a.source,a.dataset)
    elif a.output is None:p.error('--output required')
    elif a.mode=='run':run(a.dataset,a.output)
    else:evaluate(a.dataset,a.output)
