#!/usr/bin/env python3
"""Generate reproducible random point-pair examples from aligned gridded maps."""
import csv
import getopt
import hashlib
import html
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from get_ilamb_data import coordinate_grid, map_transform, write_json
from prepare_pilot import BOUNDS
from experiment_contract import inside, safe_id


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_pairs(valid, pixels, count, rng, min_distance):
    if count < 1 or count > len(valid)*(len(valid)-1)//2:
        raise ValueError('Not enough distinct valid cell pairs for the requested sample size.')
    pairs, seen = [], set()
    for _ in range(max(10000, count*1000)):
        a, b = rng.choice(len(valid), size=2, replace=False)
        key = tuple(sorted((int(valid[a]), int(valid[b]))))
        if key in seen or np.linalg.norm(pixels[a]-pixels[b]) < min_distance:
            continue
        seen.add(key)
        pairs.append((int(a),int(b)))
        if len(pairs) == count:
            return pairs
    raise ValueError('Could not sample enough separated pairs; reduce sample size or minimum separation.')


def prepare(data, output, sample_size=10, seed=42, ratio_min_abs=1e-8, min_distance=24., bounds=BOUNDS, registration=None):
    data, output = Path(data), Path(output)
    if output.exists():
        raise ValueError('Output exists. Choose a new directory to preserve the previous dataset.')
    if sample_size < 1 or seed < 0 or not np.isfinite(ratio_min_abs) or ratio_min_abs <= 0:
        raise ValueError('Sample size must be positive, seed nonnegative, and ratio threshold finite and positive.')
    if not np.isfinite(min_distance) or min_distance < 0:
        raise ValueError('Minimum separation must be finite and nonnegative.')
    if registration is None:
        raise ValueError('Validated per-source registration required; use controlled data until source rendering metadata is verified.')
    registration=json.loads(Path(registration).read_text())
    records = json.loads((data/'alignment.json').read_text())['records']
    if not records:
        raise ValueError('No aligned maps.')
    images = np.load(data/'images.npy', mmap_mode='r')
    values = np.load(data/'numerical.npy', mmap_mode='r')
    if registration.get('alignment_sha256')!=sha(data/'alignment.json'):
        raise ValueError('Registration does not match source alignment')
    plans = []
    # Validate and select everything before creating output, avoiding half-built datasets.
    for row, record in enumerate(records):
        if len(record['shape']) != 2:
            raise ValueError(f'Row {row} is not gridded. Use --gridded-only data.')
        if record['image_shape'] != [475,750,3] or record['map_box'] != [0,0,750,375]:
            raise ValueError(f'Row {row}: unsupported image layout; verify projection/bounds first.')
        review=registration['records'].get(record['id'],{})
        if review.get('status')!='validated' or not review.get('evidence'):
            raise ValueError('Unvalidated source registration: '+record['id'])
        source_bounds=review['bounds']
        x0,y0,x1,y1=record['map_box']
        if not (x0<=source_bounds[0]<source_bounds[2]<=x1 and y0<=source_bounds[1]<source_bounds[3]<=y1):
            raise ValueError('Registered bounds outside source map')
        forward,_=map_transform(source_bounds,'robinson')
        safe_id(record['id'])
        array = values[row,:record['length']].reshape(record['shape'])
        lat, lon = coordinate_grid(record)
        valid = np.flatnonzero(np.isfinite(array.ravel()) & np.isfinite(lat.ravel()) &
                              np.isfinite(lon.ravel()) & (np.abs(lat.ravel()) <= 90))
        x,y = forward(lat.ravel()[valid],lon.ravel()[valid])
        pixels = np.column_stack((x,y))
        keep = np.isfinite(pixels).all(axis=1)
        # Require a fully colored local patch, away from background and boundaries.
        for j,(px,py) in enumerate(pixels):
            if not keep[j]:continue
            ix,iy=round(px),round(py)
            patch=images[row,iy-2:iy+3,ix-2:ix+3].astype(float) if ix>=2 and iy>=2 else np.empty((0,))
            keep[j]=patch.shape==(5,5,3) and np.all(np.ptp(patch,axis=2)>8) and np.max(patch.std(axis=(0,1)))<18
        valid,pixels = valid[keep],pixels[keep]
        rng = np.random.default_rng(np.random.SeedSequence([seed,row]))
        selected = select_pairs(valid,pixels,sample_size,rng,min_distance)
        plans.append((record,array,lat,lon,valid,pixels,selected,source_bounds))
    output.mkdir(parents=True); (output/'points').mkdir(); (output/'clean').mkdir()
    font = ImageFont.load_default(size=16)
    entries, truth_rows, cards = [], [], []
    for row,(record,array,lat,lon,valid,pixels,selected,source_bounds) in enumerate(plans):
        map_id = f'map-{row:03d}-{record["id"]}'
        clean_path = f'clean/{map_id}.png'
        source = Image.fromarray(np.asarray(images[row,:475,:750]))
        source.save(output/clean_path)
        clean_hash = sha(output/clean_path)
        for pair_index,(ai,bi) in enumerate(selected):
            identifier = f'{map_id}-pair-{pair_index:03d}'
            points = []
            marked = source.copy(); draw = ImageDraw.Draw(marked)
            for label, pos in zip(('A','B'),(ai,bi)):
                index = int(valid[pos]); x,y = map(float,pixels[pos])
                points.append(dict(label=label,index=list(map(int,np.unravel_index(index,array.shape))),
                    flat_index=index,latitude=float(lat.flat[index]),longitude=float(lon.flat[index]),
                    value=float(array.flat[index]),units=record['units'],pixel=[x,y]))
                # Four monochrome pointer arms leave a 4px central gap and target color intact.
                segments=[(x-10,y,x-4,y),(x+4,y,x+10,y),(x,y-10,x,y-4),(x,y+4,x,y+10)]
                for segment in segments: draw.line(segment,fill='white',width=4)
                for segment in segments: draw.line(segment,fill='black',width=2)
                draw.text((min(x+12,725),max(y-20,2)),label,font=font,fill='black',stroke_width=2,stroke_fill='white')
            a,b=[p['value'] for p in points]
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                difference = float(np.float64(a)-np.float64(b))
                ratio = float(np.float64(a)/np.float64(b)) if abs(b)>=ratio_min_abs else float('nan')
            truth = dict(true_a=a,true_b=b,true_difference_a_minus_b=difference if np.isfinite(difference) else None,
                true_ratio_a_over_b=ratio if np.isfinite(ratio) else None,true_a_greater_b=a>b,
                ratio_valid=bool(np.isfinite(ratio)))
            image_path=f'points/{identifier}.png'; reference_path=f'points/{identifier}_reference.json'
            marked.save(output/image_path)
            write_json(output/reference_path,dict(id=identifier,map_id=map_id,sample=row,
                image_path=image_path,clean_image_path=clean_path,points=points,ground_truth=truth,
                map_bounds=list(source_bounds),projection='robinson',ratio_min_abs=ratio_min_abs,
                registration='Validated source metadata plus displayed-support filter'))
            entry=dict(id=identifier,map_id=map_id,sample=row,pair_index=pair_index,source_id=record['id'],
                dataset=record['dataset'],climate_model=record['model'],variable=record['variable'],units=record['units'],
                source_group=record.get('source_group'),display_scale=float(registration['records'][record['id']]['display_scale']),image=image_path,clean_image=clean_path,
                reference=reference_path,image_sha256=sha(output/image_path),clean_image_sha256=clean_hash,
                reference_sha256=sha(output/reference_path))
            entries.append(entry)
            truth_rows.append(dict(example_id=identifier,map_id=map_id,pair_index=pair_index,units=record['units'],**truth))
            cards.append(f'<article><h3>{html.escape(identifier)}</h3><img src="{image_path}" width="750">'
                         f'<p>A: {a:.6g}; B: {b:.6g} {html.escape(record["units"])}</p></article>')
    with (output/'ground_truth.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(truth_rows[0]));writer.writeheader();writer.writerows(truth_rows)
    write_json(output/'manifest.json',dict(format_version=3,ground_truth_sha256=sha(output/'ground_truth.csv'),acceptance=dict(status='validated',evidence=registration),experiment='random_point_pairs',
        relationship_direction='A_relative_to_B',sample_size=sample_size,seed=seed,map_count=len(records),
        example_count=len(entries),ratio_min_abs=ratio_min_abs,min_distance_pixels=min_distance,
        source_sha256={name:sha(data/name) for name in ('alignment.json','images.npy','numerical.npy')},
        selection='Uniform cell-pair draws, rejecting duplicate unordered pairs and pairs below pixel separation. '
                  'A/B order is random; cells may recur across pairs; no value-gap balancing. Not area-weighted.',
        limitations=['Registration acceptance depends on supplied per-source evidence; pixel filtering alone does not establish alignment.',
                     'Invalid near-zero-B ratios are retained as unscorable; difference/ordering remain usable.',
                     'This dataset does not specify model-repeat count.'],examples=entries))
    (output/'preview.html').write_text('<!doctype html><meta charset="utf-8"><title>Random point pairs</title>'
        '<style>body{font:16px sans-serif}article{margin:30px}img{max-width:100%}</style>'+''.join(cards))
    print(f'Created {len(entries)} pairs: {sample_size} per map across {len(records)} maps. Output: {output}')


def main():
    opts,rest=getopt.gnu_getopt(sys.argv[1:],'',['data=','output=','sample-size=','seed=',
        'ratio-min-abs=','min-distance=','map-bounds=','registration=','help'])
    args=dict(opts)
    if '--help' in args:
        print('python scripts/prepare_random_pairs.py --data data/ilamb-gridded-v2 '
              '--output experiments/random-pairs --sample-size 10 --seed 42\n'
              'Optional: --ratio-min-abs 1e-8 --min-distance 24 --map-bounds L,T,R,B')
        return
    if rest: raise ValueError('Unexpected arguments')
    prepare(args.get('--data','data/ilamb-gridded-v2'),args.get('--output','experiments/random-pairs'),
        int(args.get('--sample-size',10)),int(args.get('--seed',42)),float(args.get('--ratio-min-abs',1e-8)),
        float(args.get('--min-distance',24)),list(map(float,args['--map-bounds'].split(','))) if '--map-bounds' in args else BOUNDS,args.get('--registration'))


if __name__=='__main__':
    try: main()
    except (ValueError,OSError,KeyError,getopt.GetoptError) as exc: sys.exit(f'Error: {exc}')
