#!/usr/bin/env python3
"""Prepare reproducible development examples from existing aligned ILAMB maps."""
import getopt
import hashlib
import html
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from get_ilamb_data import coordinate_grid, map_transform, write_json

# Reconstruct the default 7.5 x 3.75-inch Robinson subplot geometry.
BOUNDS = [99.717685, 45., 669.032315, 333.75]
SAMPLES = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]


def prepare(data_dir, destination, seed=42):
    if destination.exists():
        raise ValueError('Destination exists; use a new directory to preserve selected examples.')
    records = json.loads((data_dir/'alignment.json').read_text())['records']
    images = np.load(data_dir/'images.npy', mmap_mode='r')
    values = np.load(data_dir/'numerical.npy', mmap_mode='r')
    rng = np.random.default_rng(seed)
    forward, _ = map_transform(BOUNDS, 'robinson')
    selected, groups = [], set()
    font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 17) if Path('/System/Library/Fonts/Helvetica.ttc').exists() else ImageFont.load_default(size=17)
    for sample in SAMPLES:
        record = records[sample]
        if record['source_group'] in groups:
            raise ValueError('Source groups must be distinct')
        if tuple(record['image_shape']) != (475,750,3):
            raise ValueError('This prepared protocol requires the inspected 750x375 map layout.')
        groups.add(record['source_group'])
        array=values[sample,:record['length']].reshape(record['shape'])
        lat,lon=coordinate_grid(record)
        finite=array[np.isfinite(array)]
        lo,hi=np.quantile(finite,[.05,.95]);scale=float(hi-lo)
        if scale<=0:raise ValueError('Map has no useful numerical variation')
        valid=np.flatnonzero(np.isfinite(array.ravel()) & (np.abs(lat.ravel())<65) &
                            (np.abs(lon.ravel())<155) & (np.abs(array.ravel())>=.05*scale))
        rng.shuffle(valid)
        candidates=[]
        source=images[sample]
        for index in valid[:12000]:
            x,y=forward(float(lat.flat[index]),float(lon.flat[index])); ix,iy=round(x),round(y)
            patch=source[iy-1:iy+2,ix-1:ix+2].astype(float)
            # Exclude blank/masked background and visually unstable grid boundaries.
            if patch.shape!=(3,3,3) or np.ptp(patch[1,1])<18:continue
            if array.ndim==2 and np.max(patch.std(axis=(0,1)))>18:continue
            candidates.append((int(index),float(x),float(y)))
            if len(candidates)>=800:break
        used=set()
        for kind in ['similar','different']:
            chosen=None
            for ai,a in enumerate(candidates):
                for b in candidates[ai+1:]:
                    if a[0] in used or b[0] in used:continue
                    if np.hypot(a[1]-b[1],a[2]-b[2])<100:continue
                    delta=abs(float(array.flat[a[0]]-array.flat[b[0]]))/scale
                    if (.01<=delta<=.05 if kind=='similar' else .4<=delta<=.9):
                        chosen=[a,b];break
                if chosen:break
            if not chosen:raise ValueError(f'No {kind} pair for sample {sample}')
            # Balance ordering within each pair type across maps.
            chosen.sort(key=lambda p: array.flat[p[0]], reverse=(len(selected)//2)%2==1)
            used.update(p[0] for p in chosen)
            points=[]
            for label,(index,x,y) in zip(['A','B'],chosen):
                points.append(dict(label=label,index=[int(v) for v in np.unravel_index(index,array.shape)],
                    flat_index=index,latitude=float(lat.flat[index]),longitude=float(lon.flat[index]),
                    value=float(array.flat[index]),units=record['units'],pixel=[x,y]))
            identifier=f'map-{sample:02d}-{kind}'
            image=Image.fromarray(source);draw=ImageDraw.Draw(image)
            for point in points:
                x,y=point['pixel'];draw.ellipse((x-6,y-6,x+6,y+6),outline='red',width=2)
                draw.text((x+9,y-17),point['label'],font=font,fill='black',stroke_width=2,stroke_fill='white')
            selected.append(dict(id=identifier,sample=sample,pair_type=kind,source_group=record['source_group'],
                dataset=record['dataset'],model=record['model'],variable=record['variable'],units=record['units'],
                scale_p95_minus_p05=scale,ratio_min_abs=.05*scale,
                image=f'points/{identifier}.png',reference=f'points/{identifier}_reference.json',
                points=points,rendered=image))
    destination.mkdir(parents=True);(destination/'points').mkdir()
    entries=[];cards=[]
    for entry in selected:
        image=entry.pop('rendered');points=entry.pop('points');image.save(destination/entry['image'])
        entry['image_sha256']=hashlib.sha256((destination/entry['image']).read_bytes()).hexdigest()
        write_json(destination/entry['reference'],dict(id=entry['id'],sample=entry['sample'],
            image_path=entry['image'],variable=entry['variable'],projection='Robinson centered at 0',
            map_bounds=BOUNDS,registration='Reconstructed subplot; approximate, visually review before scientific claims',points=points))
        entry['reference_sha256']=hashlib.sha256((destination/entry['reference']).read_bytes()).hexdigest()
        cards.append(f'<article><h2>{entry["id"]}: {html.escape(entry["variable"])}</h2>'
                     f'<img src="{entry["image"]}" width="750"><p>'+
                     html.escape(' | '.join(f'{p["label"]}: {p["value"]:.5g} {p["units"]}, index {p["index"]}' for p in points))+'</p></article>')
        entries.append(entry)
    write_json(destination/'manifest.json',dict(format_version=1,seed=seed,split='development',
        selection='Fixed source samples; seeded candidate order; similar gap 1-5%, different gap 40-90% of source P95-P05; '
                  '100px separation; no point reuse within map; balanced order; nonzero denominators; no model outputs used.',
        limitations=['Mixed variables: never pool raw dimensional errors.',
                     'Distinct source URLs do not ensure independent climate models or observations.',
                     'Approximate Robinson registration; this is a development set, not held-out evaluation.',
                     'Map 0 overlaps the previous pilot source.'],examples=entries))
    (destination/'preview.html').write_text('<!doctype html><meta charset="utf-8"><title>Pilot examples</title>'
        '<style>body{font:16px sans-serif;margin:30px}article{margin-bottom:40px}img{max-width:100%}</style>'+''.join(cards))
    for page in range(2):
        sheet=Image.new('RGB',(1500,5*510),'white');draw=ImageDraw.Draw(sheet)
        for j,entry in enumerate(entries[page*10:(page+1)*10]):
            x=(j%2)*750;y=(j//2)*510
            sheet.paste(Image.open(destination/entry['image']),(x,y+30));draw.text((x+10,y+5),entry['id'],font=font,fill='black')
        sheet.save(destination/f'preview-{page+1}.jpg')
    print(f'Prepared {len(entries)} examples from {len(groups)} source groups in {destination}')


if __name__=='__main__':
    try:
        pairs,extra=getopt.gnu_getopt(sys.argv[1:],'',['data=','output=','seed='])
        if extra:raise ValueError('Unexpected arguments')
        args=dict(pairs)
        prepare(Path(args.get('--data','data/ilamb_maps')),Path(args.get('--output','experiments/pilot')),int(args.get('--seed',42)))
    except (ValueError,OSError,KeyError) as e:sys.exit(f'Error: {e}')
