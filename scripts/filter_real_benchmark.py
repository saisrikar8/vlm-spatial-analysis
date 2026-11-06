#!/usr/bin/env python3
"""Post-hoc land/display-support screening of archived real-map benchmark results."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import importlib.metadata
import numpy as np
from PIL import Image,ImageDraw
from global_land_mask import globe
from experiment_contract import validate_dataset,sha,inside
from run_vlm import parse_reading,parse_direct
from evaluate_random_pairs import number,boolean

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/real-map-filtered-v1'
CONFIG=dict(patch_radius=2,minimum_rgb_channel_spread=8,maximum_patch_channel_std=18,
            land_mask='global-land-mask==1.0.0',rule='Keep pair only if BOTH reference coordinates are land and BOTH clean-image 5x5 patches have colored, stable displayed support.')


def write(path,rows,fields=None):
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def inspect_point(image,point):
    reasons=[];lat=point.get('latitude');lon=point.get('longitude');x,y=point['pixel']
    finite=all(isinstance(v,(int,float)) and math.isfinite(v) for v in (lat,lon,x,y,point.get('value')))
    is_land=False;rgb=None;spread=None;patch_std=None
    if not finite:reasons.append('nonfinite_reference_or_location')
    else:
        lon=(lon+180)%360-180
        if not -90<=lat<=90:reasons.append('invalid_latitude')
        else:
            is_land=bool(globe.is_land(lat,lon))
            if not is_land:reasons.append('ocean_coordinate')
        ix,iy=round(x),round(y);radius=CONFIG['patch_radius'];h,w=image.shape[:2]
        if not (radius<=ix<w-radius and radius<=iy<h-radius):reasons.append('outside_image_support')
        else:
            patch=image[iy-radius:iy+radius+1,ix-radius:ix+radius+1].astype(float)
            rgb=list(map(int,image[iy,ix]));spread=float(np.ptp(patch,axis=2).min());patch_std=float(patch.std(axis=(0,1)).max())
            if spread<=CONFIG['minimum_rgb_channel_spread']:reasons.append('background_mask_or_grayscale_patch')
            if patch_std>=CONFIG['maximum_patch_channel_std']:reasons.append('unstable_edge_or_mixed_patch')
    return dict(label=point['label'],latitude=lat,longitude=lon,pixel_x=x,pixel_y=y,reference_value=point.get('value'),
        geographic_land=is_land,center_rgb=json.dumps(rgb),minimum_patch_channel_spread=spread,maximum_patch_channel_std=patch_std,
        keep=not reasons,exclusion_reasons=';'.join(reasons))


def raw_predictions(folder,e,repeats,threshold):
    answers={}
    for q in ('verbose','difference','ratio','ordering'):
        records=[json.loads(line) for line in (folder/'raw'/e['id']/(q+'.jsonl')).read_text().splitlines()]
        if len(records)!=repeats or [r['sample'] for r in records]!=list(range(repeats)):raise ValueError('Raw trial identity mismatch')
        values=[]
        for r in records:
            value,error=parse_reading(r['raw_response'],['A','B']) if q=='verbose' else parse_direct(r['raw_response'],q)
            values.append(value if value is not None and not error and not r.get('hit_token_limit') and all(v is not None for v in value.values()) else None)
        answers[q]=values
    rows=[]
    for i in range(repeats):
        pair=answers['verbose'][i];a,b=(pair['A'],pair['B']) if pair else (None,None)
        for method in ('verbose','non-verbose'):
            if method=='verbose':d=a-b if pair else None;r=a/b if pair and abs(b)>=threshold else None;o=a>b if pair else None
            else:d,r,o=[answers[q][i]['answer'] if answers[q][i] else None for q in ('difference','ratio','ordering')]
            rows.append(dict(example_id=e['id'],experiment=method,trial=i,predicted_difference_a_minus_b=d,
                predicted_ratio_a_over_b=r,predicted_a_greater_b=o))
    return rows


def score_run(run,entries,truths,scales,parse_mode,output):
    output.mkdir(parents=True);kept={e['id']:e for e in entries};records=[];available=set();input_hashes={};repeats=None;protocol=None
    for path in sorted(run.glob('*/predictions.csv')):
        meta=json.loads(path.with_name('run.json').read_text())
        if meta['manifest_sha256']!=sha(ROOT/'experiments/random-pairs/manifest.json'):raise ValueError('Archived dataset identity mismatch')
        signature=json.dumps(dict(model=meta['model'],revision=meta['resolved_revision'],prompts=meta['prompts'],script_sha256=meta['script_sha256'],
            options={k:v for k,v in meta['options'].items() if k not in ('map-index','output','dataset')}),sort_keys=True)
        if protocol is not None and protocol!=signature:raise ValueError('Mixed archived protocols')
        protocol=signature;repeats=meta['options']['repeats'];input_hashes[str(path.relative_to(ROOT))]=sha(path)
        available.add(meta['map_id'])
        relevant=[e for e in entries if e['map_id']==meta['map_id']]
        if parse_mode=='archived_strict':
            with path.open() as stream:records.extend(r for r in csv.DictReader(stream) if r['example_id'] in kept)
        else:
            for e in relevant:records.extend(raw_predictions(path.parent,e,repeats,1e-8))
    if repeats is None:raise ValueError('No archived results')
    scope=[e for e in entries if e['map_id'] in available];expected={e['id'] for e in scope}
    rows=[];seen=set()
    for r in records:
        key=(r['example_id'],r['experiment'],int(r['trial']))
        if key in seen or r['example_id'] not in expected or not 0<=key[2]<repeats:raise ValueError('Duplicate/out-of-scope predictions')
        seen.add(key);t=truths[r['example_id']];e=kept[r['example_id']]
        for query,suffix,convert in [('difference','difference_a_minus_b',number),('ratio','ratio_a_over_b',number),('ordering','a_greater_b',boolean)]:
            prediction=convert(r['predicted_'+suffix]);truth=convert(t['true_'+suffix]);valid=prediction is not None and truth is not None
            error=abs(prediction-truth) if valid else None
            if error is not None and not math.isfinite(error):valid=False;error=None
            rows.append(dict(example_id=e['id'],map_id=e['map_id'],source_group=e['source_group'],units=e['units'],method=r['experiment'],trial=key[2],query=query,
                truth=truth,prediction=prediction,valid=valid,absolute_error=error,normalized_error=error/scales[e['map_id']] if valid and query=='difference' and scales[e['map_id']]>0 else None))
    expected_keys={(e['id'],method,t) for e in scope for method in ('verbose','non-verbose') for t in range(repeats)}
    if seen!=expected_keys:raise ValueError('Missing saved answers inside available-map scope')
    write(output/'trial_errors.csv',rows,['example_id','map_id','source_group','units','method','trial','query','truth','prediction','valid','absolute_error','normalized_error'])
    write(output/'filtered_predictions.csv',records,list(records[0]) if records else ['example_id','experiment','trial','predicted_difference_a_minus_b','predicted_ratio_a_over_b','predicted_a_greater_b'])
    maps=[];summary=[];paired=[]
    for method in ('verbose','non-verbose'):
        for q in ('difference','ratio','ordering'):
            rr=[r for r in rows if r['method']==method and r['query']==q];valid=[r for r in rr if r['valid']]
            trial0=[r for r in rr if r['trial']==0];correct=sum(r['valid'] and r['prediction']==r['truth'] for r in trial0)
            # Average within pair, then within source group, then equally over source groups.
            metric='normalized_error' if q=='difference' else 'absolute_error';perpair={}
            for e in scope:
                values=[r[metric] for r in rr if r['example_id']==e['id'] and r['valid'] and r[metric] is not None]
                if values:perpair[e['id']]=statistics.mean(values)
            source_values={}
            for eid,val in perpair.items():source_values.setdefault(kept[eid]['source_group'],[]).append(val)
            pooled=statistics.mean(statistics.mean(v) for v in source_values.values()) if source_values else None
            summary.append(dict(run=run.name,parsing=parse_mode,method=method,query=q,retained_pairs_in_available_maps=len(scope),retained_maps=len({e['map_id'] for e in scope}),
                source_groups=len({e['source_group'] for e in scope}),expected_trials=len(scope)*repeats,valid_trials=len(valid),invalid_or_undefined_trials=len(rr)-len(valid),
                ordering_trial0_correct=correct if q=='ordering' else None,ordering_trial0_total=len(trial0) if q=='ordering' else None,
                ordering_trial0_accuracy=correct/len(trial0) if trial0 and q=='ordering' else None,
                always_false_trial0_accuracy=sum(boolean(truths[e['id']]['true_a_greater_b'])==0 for e in scope)/len(scope) if scope and q=='ordering' else None,
                scored_pairs=len(perpair) if q!='ordering' else None,
                source_balanced_mean_error=pooled if q!='ordering' else None,
                error_units='native_P95_minus_P05_normalized' if q=='difference' else 'dimensionless' if q=='ratio' else 'not_applicable'))
            for mid in dict.fromkeys(e['map_id'] for e in scope):
                mr=[r for r in rr if r['map_id']==mid];pv=[]
                for eid in {r['example_id'] for r in mr}:
                    vals=[r['absolute_error'] for r in mr if r['example_id']==eid and r['valid']]
                    if vals:pv.append(statistics.mean(vals))
                maps.append(dict(map_id=mid,method=method,query=q,expected_pairs=len({r['example_id'] for r in mr}),valid_trials=sum(r['valid'] for r in mr),
                    mean_absolute_error=statistics.mean(pv) if pv else None,units=mr[0]['units'] if q=='difference' else 'dimensionless'))
    lookup={(r['example_id'],r['method'],r['trial'],r['query']):r for r in rows}
    for e in scope:
        for q in ('difference','ratio'):
            shared=[]
            for trial in range(repeats):
                a,b=[lookup[e['id'],m,trial,q] for m in ('verbose','non-verbose')]
                if a['valid'] and b['valid']:shared.append((a['absolute_error'],b['absolute_error']))
            paired.append(dict(example_id=e['id'],map_id=e['map_id'],source_group=e['source_group'],query=q,matched_valid_trials=len(shared),
                extracted_error=statistics.mean(a for a,b in shared) if shared else None,direct_error=statistics.mean(b for a,b in shared) if shared else None))
    write(output/'summary.csv',summary);write(output/'per_map_metrics.csv',maps,['map_id','method','query','expected_pairs','valid_trials','mean_absolute_error','units']);write(output/'matched_pairs.csv',paired,['example_id','map_id','source_group','query','matched_valid_trials','extracted_error','direct_error'])
    (output/'provenance.json').write_text(json.dumps(dict(run=str(run),parsing=parse_mode,repeats=repeats,available_maps=sorted(available),retained_pairs=len(scope),protocol=json.loads(protocol),prediction_sha256=input_hashes,
        note='Saved responses only. Same input-only pair filter for both methods. No model failures removed. Protocols and repeat counts are not pooled. No p-values. Numeric-content parsing is an explicitly post-hoc sensitivity analysis accepting bare finite JSON scalars.'),indent=2)+'\n')
    return summary


def main():
    if OUT.exists():raise ValueError('Output exists; preserve previous filtered analysis')
    dataset=ROOT/'experiments/random-pairs';manifest,truths=validate_dataset(dataset,legacy=True)
    data=ROOT/'data/ilamb-gridded-v2'
    for name,digest in manifest['source_sha256'].items():
        if sha(data/name)!=digest:raise ValueError('Export source checksum mismatch')
    numerical=np.load(data/'numerical.npy',mmap_mode='r');alignment=json.loads((data/'alignment.json').read_text())['records']
    entries=[];points=[];decisions=[];scales={}
    for e in manifest['examples']:
        with Image.open(inside(dataset,e['clean_image'])) as im:image=np.asarray(im.convert('RGB'))
        ref=json.loads(inside(dataset,e['reference']).read_text());checked=[inspect_point(image,p) for p in ref['points']]
        for p in checked:points.append(dict(example_id=e['id'],map_id=e['map_id'],**p))
        keep=len(checked)==2 and all(p['keep'] for p in checked)
        reasons=';'.join(p['label']+':'+p['exclusion_reasons'] for p in checked if not p['keep'])
        decisions.append(dict(example_id=e['id'],map_id=e['map_id'],source_group=e['source_group'],keep=keep,reasons=reasons))
        if keep:entries.append(e)
        if e['map_id'] not in scales:
            record=alignment[e['sample']];values=numerical[e['sample'],:record['length']];finite=values[np.isfinite(values)]
            scales[e['map_id']]=float(np.percentile(finite,95)-np.percentile(finite,5))
    OUT.mkdir(parents=True)
    # Save decisions before opening any model predictions.
    write(OUT/'point_audit.csv',points);write(OUT/'pair_decisions.csv',decisions)
    kept_ids={e['id'] for e in entries}
    write(OUT/'filtered_ground_truth.csv',[truths[e['id']] for e in entries],list(next(iter(truths.values()))))
    info=dict(experiment='posthoc_land_visible_real_map_benchmark',original_pairs=len(manifest['examples']),retained_pairs=len(entries),excluded_pairs=len(decisions)-len(entries),
        retained_maps=len({e['map_id'] for e in entries}),retained_source_groups=len({e['source_group'] for e in entries}),filter=CONFIG,
        package_version=importlib.metadata.version('global-land-mask'),source_manifest_sha256=sha(dataset/'manifest.json'),
        kept_example_ids=sorted(kept_ids),note='Land and visible-support screening only. Does not certify pixel/native-cell registration or make this a fresh held-out benchmark.')
    (OUT/'filter_manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    # Original, unchanged marked images for every retained pair.
    for start in range(0,len(entries),8):
        block=entries[start:start+8];sheet=Image.new('RGB',(1000,360*math.ceil(len(block)/2)),'white');draw=ImageDraw.Draw(sheet)
        for i,e in enumerate(block):
            with Image.open(dataset/e['image']) as im:thumb=im.convert('RGB').resize((500,317))
            x=(i%2)*500;y=(i//2)*360;sheet.paste(thumb,(x,y+25));draw.text((x+5,y+5),e['id'],fill='black')
        sheet.save(OUT/f'retained-contact-{start//8+1:02d}.jpg')
    (OUT/'preview.html').write_text('<!doctype html><meta charset="utf-8"><title>Retained real-map pairs</title><h1>Land and displayed-support filtered pairs</h1>'+''.join(f'<h3>{e["id"]}</h3><img width="750" src="../../experiments/random-pairs/{e["image"]}">' for e in entries))
    all_summaries=[]
    for name in ('random-40147869','random-40182581'):
        for parsing in ('archived_strict','numeric_content'):
            all_summaries.extend(score_run(ROOT/'runs3'/name,entries,truths,scales,parsing,OUT/name/parsing))
    write(OUT/'all_summaries.csv',all_summaries)
    print(json.dumps(info,indent=2))
    for r in all_summaries:
        if r['query']=='ordering':print(r['run'],r['parsing'],r['method'],r['ordering_trial0_correct'],r['ordering_trial0_total'],r['ordering_trial0_accuracy'])


if __name__=='__main__':main()
