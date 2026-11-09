#!/usr/bin/env python3
"""Three-way full/crop/plain-patch diagnostic with common choices and prompt."""
import argparse
import json
from pathlib import Path
import re
from color_matching import ROOT,MODEL,REVISION,sha,dump,table

PROMPT=('Which numbered color swatch at the bottom best matches the target color in the upper '
        'part of the image? If a circular marker is present, the target is the underlying map '
        'color at its CENTER, not the marker outline or grid lines. Otherwise, the target is '
        'the large plain colored square. Return only one digit from 1 to 6. '
        'Do not read numerical map values or report the grid cell.')
CONDITIONS=('full','crop','patch')


def prepare(source,out):
    from PIL import Image
    if out.exists():raise ValueError('Choose a new dataset directory')
    m=json.loads((source/'manifest.json').read_text());groups={}
    for e in m['examples']:groups.setdefault(e['point_id'],{})[e['condition']]=e
    if len(groups)!=40 or any(set(g)!={'full','crop'} for g in groups.values()):raise ValueError('Unexpected input scope')
    plans=[];out.mkdir(parents=True);(out/'images').mkdir()
    for index,(point_id,g) in enumerate(groups.items()):
        originals={}
        for condition,e in g.items():
            p=source/e['image']
            if sha(p)!=e['image_sha256']:raise ValueError('Source image changed')
            with Image.open(p) as im:originals[condition]=im.convert('RGB')
        e=g['full'];patch=originals['full'].copy();patch.paste('white',(0,0,795,520));patch.paste(tuple(e['reference_rgb']),(235,90,560,415))
        originals['patch']=patch
        for im in originals.values():
            assert im.size==(795,625)
            assert im.crop((0,520,795,625)).tobytes()==originals['full'].crop((0,520,795,625)).tobytes()
        assert patch.getpixel((397,252))==tuple(e['reference_rgb'])
        order=CONDITIONS[index%3:]+CONDITIONS[:index%3]
        for condition in order:
            row=dict(id=point_id+'-'+condition,point_id=point_id,condition=condition,split=e['split'],map_id=e['map_id'],
                source_group=e['source_group'],answer=e['answer'],swatches=e['swatches'],reference_rgb=e['reference_rgb'],
                image=f'images/{point_id}-{condition}.png')
            originals[condition].save(out/row['image']);row['image_sha256']=sha(out/row['image']);plans.append(row)
    dump(out/'manifest.json',dict(experiment='plain_patch_matching_v1',source_manifest_sha256=sha(source/'manifest.json'),
        prompt=PROMPT,model=MODEL,revision=REVISION,generation=dict(do_sample=False,max_new_tokens=16),seed=20261106,
        conditions=list(CONDITIONS),design='Same 40 points, choices, canvas; rerun all three conditions with common conditional target prompt. Rotating condition order.',
        primary='32 reused test points: plain-patch color accuracy and paired changes versus crop/full. Invalid/missing incorrect; development separate.',examples=plans))
    table(out/'answer_key.csv',plans)
    (out/'preview.html').write_text('<html><body>'+''.join(f'<p>{e["id"]}</p><img src="{e["image"]}">' for e in plans)+'</body></html>')
    print('Prepared',len(plans),'images with identical swatch panels.')


def evaluate(dataset,out):
    m=json.loads((dataset/'manifest.json').read_text());meta=json.loads((out/'run.json').read_text())
    if meta['manifest_sha256']!=sha(dataset/'manifest.json'):raise ValueError('Manifest mismatch')
    raw=[json.loads(s) for s in (out/'responses.jsonl').read_text().splitlines()];byid={r['id']:r for r in raw}
    if len(byid)!=len(raw) or set(byid)-{e['id'] for e in m['examples']}:raise ValueError('Unknown/duplicate responses')
    rows=[]
    for e in m['examples']:
        rawtext=byid.get(e['id'],{}).get('raw_response','');p=rawtext.strip();valid=re.fullmatch('[1-6]',p) is not None
        rows.append(dict(id=e['id'],point_id=e['point_id'],condition=e['condition'],split=e['split'],map_id=e['map_id'],
            reference=e['answer'],prediction=p,valid=valid,correct=valid and p==e['answer'],raw_response=rawtext))
    table(out/'answers.csv',rows);pairs=[]
    for point_id in dict.fromkeys(e['point_id'] for e in m['examples']):
        group={r['condition']:r for r in rows if r['point_id']==point_id};e=group['full']
        row=dict(point_id=point_id,split=e['split'],map_id=e['map_id'],reference=e['reference'])
        for condition in CONDITIONS:
            for key in ['prediction','valid','correct']:row[condition+'_'+key]=group[condition][key]
        pairs.append(row)
    table(out/'paired_answers.csv',pairs);summary={}
    for split in ['development','test']:
        ps=[r for r in pairs if r['split']==split];summary[split]=dict(n=len(ps),conditions={})
        for c in CONDITIONS:
            summary[split]['conditions'][c]=dict(correct=sum(r[c+'_correct'] for r in ps),valid=sum(r[c+'_valid'] for r in ps),accuracy=sum(r[c+'_correct'] for r in ps)/len(ps),
                per_map={mid:dict(n=len(rs:=[r for r in ps if r['map_id']==mid]),correct=sum(r[c+'_correct'] for r in rs)) for mid in sorted({r['map_id'] for r in ps})})
        summary[split]['patch_comparisons']={c:dict(wrong_to_correct=sum(not r[c+'_correct'] and r['patch_correct'] for r in ps),correct_to_wrong=sum(r[c+'_correct'] and not r['patch_correct'] for r in ps),changed_answers=sum(r[c+'_prediction']!=r['patch_prediction'] for r in ps)) for c in ['full','crop']}
    summary['notes']=['Exploratory reused inputs, no significance or fresh-test claim.',
        'All conditions rerun under the same new conditional-target prompt. Do not conflate baseline scores with earlier prompt runs.',
        'Plain patch removes marker, context and local variability while enlarging target; it does not isolate a single internal mechanism.',
        'Patch accuracy includes visual color matching and numbered-option association; no numerical legend or geographic task.']
    dump(out/'summary.json',summary);print(json.dumps(summary))


def run(dataset,out):
    import os,sys,importlib.metadata
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
    m=json.loads((dataset/'manifest.json').read_text())
    if out.exists():raise ValueError('Choose a new run directory')
    if m['experiment']!='plain_patch_matching_v1' or m['prompt']!=PROMPT or m['revision']!=REVISION:raise ValueError('Protocol changed')
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
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','evaluate']);p.add_argument('--dataset',type=Path,default=ROOT/'experiments/plain-patch-matching-v1');p.add_argument('--source',type=Path,default=ROOT/'experiments/crop-color-matching-v1');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.mode=='prepare':prepare(a.source,a.dataset)
    elif a.output is None:p.error('--output required')
    elif a.mode=='run':run(a.dataset,a.output)
    else:evaluate(a.dataset,a.output)
