#!/usr/bin/env python3
"""Run one map's verbose/non-verbose experiments and score against NetCDF truth."""
import csv
import getopt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
from run_vlm import MODELS, collect, parse_reading, parse_direct, write_json
from run_pilot import REVISION
from evaluate_random_pairs import evaluate
from experiment_contract import validate_dataset, inside, safe_id, code_hashes, protocol

HELP='''python scripts/run_random_pairs.py --map-index 0 --output runs/random-check
  --dataset experiments/controlled-v1
  --experiment both|verbose|non-verbose (default both)
  --presentation clean-marked|marked-only (default clean-marked)
  --repeats 10 --seed 42 --model qwen7b [--revision COMMIT] [--resume]
  --temperature 0.7 --top-p 0.9 --max-new-tokens 128 --max-pixels 1003520
  --dtype bfloat16|float16|float32 --null-accuracy 0.5 --test-trial 0 --dry-run
Sample size (pairs per map) is set when generating the dataset, not here.
'''


def options(argv):
    opts,rest=getopt.gnu_getopt(argv,'',['dataset=','map-index=','output=','experiment=','presentation=',
        'repeats=','seed=','model=','revision=','temperature=','top-p=','max-new-tokens=','max-pixels=',
        'dtype=','null-accuracy=','test-trial=','dry-run','resume','help'])
    a={k[2:]:v for k,v in opts}
    if 'help' in a:return a
    if rest or 'map-index' not in a or 'output' not in a:raise ValueError(HELP)
    for k,v in dict(dataset='experiments/controlled-v1',experiment='both',presentation='clean-marked',model='qwen7b',dtype='bfloat16').items():a.setdefault(k,v)
    for k,v in {'map-index':0,'repeats':10,'seed':42,'max-new-tokens':128,'max-pixels':1003520,'test-trial':0}.items():a[k]=int(a.get(k,v))
    for k,v in {'temperature':.7,'top-p':.9,'null-accuracy':.5}.items():a[k]=float(a.get(k,v))
    if a['model'] not in MODELS or a['experiment'] not in ('both','verbose','non-verbose') or a['presentation'] not in ('clean-marked','marked-only'):raise ValueError('Invalid model, experiment, or presentation')
    a.setdefault('revision',REVISION if a['model']=='qwen7b' else 'main')
    if a['dtype'] not in ('bfloat16','float16','float32') or a['repeats']<1 or not 0<=a['test-trial']<a['repeats'] or a['seed']<0:raise ValueError('Invalid dtype, repeats, seed, or test trial')
    if not math.isfinite(a['temperature']) or a['temperature']<=0 or not 0<a['top-p']<=1 or not 0<a['null-accuracy']<1 or a['max-new-tokens']<1 or a['max-pixels']<3136:raise ValueError('Invalid sampling settings')
    return a


def prompts(presentation):
    prefix=('Image 1 is the clean map and legend. Image 2 is the same map with black-and-white A/B pointer marks. '
            'Locate the centers of the gaps in the pointer marks in image 2 and read the corresponding colors in image 1. '
            if presentation=='clean-marked' else
            'Read the map and legend at the centers of the gaps in the black-and-white pointer marks labeled A and B. ')
    return dict(verbose=prefix+'Return only {"A": number, "B": number}, in the legend units. Use null for an unreadable value. Do not calculate relationships or explain.',
        difference=prefix+'What is A minus B in the legend units? Return only {"answer": number}, or null if unreadable. Do not return individual A/B values or explain.',
        ratio=prefix+'What is A divided by B? Return only {"answer": number}, or null if unreadable or undefined. Do not return individual A/B values or explain.',
        ordering=prefix+'Is A strictly greater than B? Equality means false. Return only {"answer": true} or {"answer": false}, or {"answer": null} if unreadable. Do not return individual A/B values or explain.')


def planned(a):
    root=Path(a['dataset']).resolve();manifest,_=validate_dataset(root)
    if manifest.get('relationship_direction')!='A_relative_to_B':raise ValueError('Expected random A-relative-to-B dataset.')
    maps=list(dict.fromkeys(e['map_id'] for e in manifest['examples']))
    if not 0<=a['map-index']<len(maps):raise ValueError(f'Map index must be between 0 and {len(maps)-1}')
    entries=[(i,e) for i,e in enumerate(manifest['examples']) if e['map_id']==maps[a['map-index']]]
    for _,e in entries:
        for path_key,hash_key in [('image','image_sha256'),('clean_image','clean_image_sha256'),('reference','reference_sha256')]:
            if hashlib.sha256((root/e[path_key]).read_bytes()).hexdigest()!=e[hash_key]:raise ValueError(f'Input hash mismatch: {e[path_key]}')
    if a['seed']+4*a['repeats']*len(manifest['examples'])>=2**32:raise ValueError('Seed range exceeds 32 bits')
    return root,manifest,entries,maps[a['map-index']]


def decoded(path,query):
    parser=(lambda raw:parse_reading(raw,['A','B'])) if query=='verbose' else (lambda raw:parse_direct(raw,query))
    out=[]
    for line in path.read_text().splitlines():
        r=json.loads(line);value,error=parser(r['raw_response'])
        out.append(value if r.get('complete') and not r.get('hit_token_limit') and not error else None)
    return out


def prediction_rows(entry,answers,repeats,threshold):
    rows=[]
    for experiment in ('verbose','non-verbose'):
        if (experiment=='verbose' and 'verbose' not in answers) or (experiment=='non-verbose' and 'ordering' not in answers):continue
        for trial in range(repeats):
            a=b=d=r=o=None
            if experiment=='verbose':
                pair=answers['verbose'][trial]
                if pair is not None:
                    a,b=pair['A'],pair['B']
                    if a is not None and b is not None:
                        d=a-b;r=a/b if abs(b)>=threshold else None;o=a>b
            else:
                d,r,o=[answers[q][trial]['answer'] if q in answers and answers[q][trial] else None for q in ('difference','ratio','ordering')]
            if d is not None and not math.isfinite(d):d=None
            if r is not None and not math.isfinite(r):r=None
            rows.append(dict(example_id=entry['id'],experiment=experiment,trial=trial,predicted_a=a,predicted_b=b,
                predicted_difference_a_minus_b=d,predicted_ratio_a_over_b=r,predicted_a_greater_b=o))
    return rows


def collect_resumable(model,processor,inputs,args,rawdir,torch,set_seed,filename,parser):
    path=rawdir/filename
    records=[]
    if path.exists():
        lines=path.read_text().splitlines()
        for i,line in enumerate(lines):
            try:record=json.loads(line)
            except ValueError:
                if i!=len(lines)-1:raise ValueError('Corrupt interior raw response')
                break
            if record['sample']!=i or record['seed']!=args['seed']+i or i>=args['samples']:
                raise ValueError('Raw sample identity mismatch')
            records.append(record)
    if len(records)==args['samples']:return
    # Repair only a trailing interrupted JSON line, then flush each new generation.
    temporary=path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(r)+'\n' for r in records))
    temporary.replace(path)
    collect(model,processor,inputs,args,rawdir,torch,set_seed,filename=filename,parser=parser,
            start_index=len(records),append=True)



def main(argv=None):
    a=options(sys.argv[1:] if argv is None else argv)
    if 'help' in a:print(HELP);return
    root,manifest,entries,map_id=planned(a)
    tasks=prompts(a['presentation']);keys=['verbose'] if a['experiment']=='verbose' else ['difference','ratio','ordering']
    if a['experiment']=='both':keys=['verbose','difference','ratio','ordering']
    keys=[q for q in keys if q=='verbose' or q in manifest.get('queries',['difference','ratio','ordering'])]
    output=Path(a['output'])/map_id
    previous=None
    if output.exists():
        if 'resume' not in a:raise ValueError('Run exists; choose a new output root or --resume.')
        previous=json.loads((output/'run.json').read_text())
    metadata=dict(method_labels={'verbose':'extract_then_calculate','non-verbose':'direct_answer'},code_sha256=code_hashes(),format_version=2,status='planned',map_id=map_id,options=a,model=MODELS[a['model']],
        manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prompts={k:tasks[k] for k in keys},examples=[e for _,e in entries],calls=len(entries)*len(keys)*a['repeats'])
    if 'dry-run' in a:print(json.dumps(metadata,indent=2));return
    import torch
    from PIL import Image
    from huggingface_hub import model_info
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable; use a GPU allocation.')
    if a['dtype']=='bfloat16' and not torch.cuda.is_bf16_supported():raise ValueError('Use --dtype float16 for this GPU.')
    revision=model_info(MODELS[a['model']],revision=a['revision']).sha
    metadata.update(status='loading',resolved_revision=revision,started_utc=datetime.now(timezone.utc).isoformat(),
        gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,slurm_job_id=os.getenv('SLURM_JOB_ID'),
        python=sys.version,packages={name:importlib.metadata.version(name) for name in ('torch','transformers','pillow','accelerate')})
    if previous is not None:
        if previous.get('manifest_sha256')!=metadata['manifest_sha256'] or previous.get('code_sha256')!=metadata['code_sha256'] or previous.get('prompts')!=metadata['prompts']:
            raise ValueError('Resume dataset/code/prompt mismatch')
    output.mkdir(parents=True,exist_ok=previous is not None)
    try:
        processor=AutoProcessor.from_pretrained(MODELS[a['model']],revision=revision,min_pixels=3136,max_pixels=a['max-pixels'])
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODELS[a['model']],revision=revision,
            torch_dtype=getattr(torch,a['dtype']),device_map={'':'cuda'},attn_implementation='sdpa').eval()
        metadata.update(status='running',model_generation_config=model.generation_config.to_dict(),rendered_prompts={})
        metadata['protocol_sha256']=protocol(metadata)
        if previous is not None and previous.get('protocol_sha256')!=metadata['protocol_sha256']:
            raise ValueError('Resume protocol mismatch')
        write_json(output/'run.json',metadata)
        with (output/'predictions.csv').open('w' if previous is not None else 'x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['example_id','experiment','trial','predicted_a','predicted_b',
                'predicted_difference_a_minus_b','predicted_ratio_a_over_b','predicted_a_greater_b']);writer.writeheader()
            for global_index,e in entries:
                paths=[e['clean_image'],e['image']] if a['presentation']=='clean-marked' else [e['image']]
                images=[]
                for path in paths:
                    with Image.open(root/path) as im:images.append(im.convert('RGB'))
                rawdir=output/'raw'/e['id'];rawdir.mkdir(parents=True,exist_ok=previous is not None)
                answers={}
                for query in keys:
                    messages=[{'role':'user','content':[{'type':'image'} for _ in images]+[{'type':'text','text':tasks[query]}]}]
                    rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                    inputs=processor(text=[rendered],images=images,padding=True,return_tensors='pt').to(model.device)
                    metadata['rendered_prompts'][query]=rendered;write_json(output/'run.json',metadata)
                    offset=('verbose','difference','ratio','ordering').index(query)
                    args=dict(a,samples=a['repeats'],labels=['A','B'],seed=a['seed']+(global_index*4+offset)*a['repeats'])
                    parser=(lambda raw:parse_reading(raw,['A','B'])) if query=='verbose' else (lambda raw,q=query:parse_direct(raw,q))
                    print(e['id'],query,flush=True)
                    collect_resumable(model,processor,inputs,args,rawdir,torch,set_seed,query+'.jsonl',parser)
                    answers[query]=decoded(rawdir/(query+'.jsonl'),query)
                writer.writerows(prediction_rows(e,answers,a['repeats'],manifest['ratio_min_abs']))
                stream.flush();os.fsync(stream.fileno())
        metadata.update(status='finished',finished_utc=datetime.now(timezone.utc).isoformat(),peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
        write_json(output/'run.json',metadata)
        evaluation=output/'evaluation'
        if evaluation.exists():
            index=1
            while (output/f'evaluation-resume-{index}').exists():index+=1
            evaluation=output/f'evaluation-resume-{index}'
        evaluate(root,output/'predictions.csv',evaluation,a['null-accuracy'],a['test-trial'],map_id)
        print(f'CSV ready: {evaluation / "per_map_metrics.csv"}',flush=True)
    except BaseException as error:
        if previous is not None and metadata.get('protocol_sha256')!=previous.get('protocol_sha256'):raise
        metadata.update(status='failed',error=str(error));write_json(output/'run.json',metadata);raise


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,ImportError,getopt.GetoptError) as exc:sys.exit(f'Error: {exc}')
