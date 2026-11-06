"""One pair per row, original real-map responses and auditable derived scores."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/real-map-land-only-v1/benchmark_46_land_pairs_raw.csv'


def read(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))


def num(value):
    return float(value) if value not in ('',None) else None


def boolean(value):
    return None if value in ('',None) else str(value).lower() in ('true','1','1.0')


def main():
    if OUT.exists():raise ValueError('Export exists; do not overwrite')
    dataset=ROOT/'experiments/random-pairs';run=ROOT/'runs3/random-40147869'
    screen=json.loads((ROOT/'analysis/real-map-land-only-v1/filter_manifest.json').read_text());kept=set(screen['kept_ids'])
    manifest=json.loads((dataset/'manifest.json').read_text())
    assert hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()==screen['source_manifest_sha256']
    truths={r['example_id']:r for r in read(dataset/'ground_truth.csv')}
    audit={(r['example_id'],r['label']):r for r in read(ROOT/'analysis/real-map-filtered-v1/point_audit.csv')}
    long=read(ROOT/'analysis/real-map-land-only-v1/random-40147869/numeric_content/trial_errors.csv')
    errors={(r['example_id'],r['method'],r['query']):r for r in long}
    assert len(long)==len(errors)==46*2*3 and all(r['trial']=='0' for r in long)
    rows=[]
    for global_index,e in enumerate(manifest['examples']):
        if e['id'] not in kept:continue
        eid=e['id'];folder=run/e['map_id'];rawdir=folder/'raw'/eid
        meta=json.loads((folder/'run.json').read_text());assert meta['options']['repeats']==1
        assert meta['manifest_sha256']==screen['source_manifest_sha256']
        raw={}
        for offset,q in enumerate(('verbose','difference','ratio','ordering')):
            records=[json.loads(line) for line in (rawdir/(q+'.jsonl')).read_text().splitlines()]
            assert len(records)==1 and records[0]['sample']==0
            assert records[0]['seed']==meta['options']['seed']+global_index*4+offset
            raw[q]=records[0]
        original={r['experiment']:r for r in read(folder/'predictions.csv') if r['example_id']==eid}
        assert set(original)=={'verbose','non-verbose'}
        t=truths[eid];ref=json.loads((dataset/e['reference']).read_text());a,b=ref['points']
        assert a['label']=='A' and b['label']=='B'
        aa,bb=audit[eid,'A'],audit[eid,'B']
        assert aa['geographic_land']==bb['geographic_land']=='True'
        ex={q:errors[eid,'verbose',q] for q in ('difference','ratio','ordering')}
        direct={q:errors[eid,'non-verbose',q] for q in ('difference','ratio','ordering')}
        partial=raw['verbose'].get('reading') or {}
        truth_order=boolean(t['true_a_greater_b']);extract_order=boolean(ex['ordering']['prediction']);direct_order=boolean(direct['ordering']['prediction'])
        row=dict(pair_id=eid,map_id=e['map_id'],units=e['units'],
            reference_A=num(t['true_a']),reference_B=num(t['true_b']),extracted_A=partial.get('A'),extracted_B=partial.get('B'),
            reference_A_minus_B=num(t['true_difference_a_minus_b']),calculated_A_minus_B=num(ex['difference']['prediction']),direct_A_minus_B=num(direct['difference']['prediction']),
            calculated_difference_error_native_units=num(ex['difference']['absolute_error']),direct_difference_error_native_units=num(direct['difference']['absolute_error']),
            reference_A_greater_B=truth_order,calculated_A_greater_B=extract_order,direct_A_greater_B=direct_order,
            calculated_order_correct=ex['ordering']['valid']=='True' and extract_order==truth_order,
            direct_order_correct=direct['ordering']['valid']=='True' and direct_order==truth_order,
            raw_extraction_response=raw['verbose']['raw_response'],raw_direct_difference_response=raw['difference']['raw_response'],
            raw_direct_ratio_response=raw['ratio']['raw_response'],raw_direct_order_response=raw['ordering']['raw_response'],
            reference_A_divided_by_B=num(t['true_ratio_a_over_b']),calculated_A_divided_by_B=num(ex['ratio']['prediction']),direct_A_divided_by_B=num(direct['ratio']['prediction']),
            calculated_ratio_absolute_error=num(ex['ratio']['absolute_error']),direct_ratio_absolute_error=num(direct['ratio']['absolute_error']),
            calculated_difference_valid=ex['difference']['valid']=='True',direct_difference_valid=direct['difference']['valid']=='True',
            calculated_ratio_valid=ex['ratio']['valid']=='True',direct_ratio_valid=direct['ratio']['valid']=='True',
            calculated_order_valid=ex['ordering']['valid']=='True',direct_order_valid=direct['ordering']['valid']=='True',
            calculated_difference_error_normalized=num(ex['difference']['normalized_error']),direct_difference_error_normalized=num(direct['difference']['normalized_error']),
            direct_difference_archived_strict=num(original['non-verbose']['predicted_difference_a_minus_b']),
            direct_ratio_archived_strict=num(original['non-verbose']['predicted_ratio_a_over_b']),
            A_latitude=a['latitude'],A_longitude=a['longitude'],A_pixel_x=a['pixel'][0],A_pixel_y=a['pixel'][1],
            B_latitude=b['latitude'],B_longitude=b['longitude'],B_pixel_x=b['pixel'][0],B_pixel_y=b['pixel'][1],
            A_geographic_land=True,B_geographic_land=True,A_display_support_pass=aa['keep']=='True',B_display_support_pass=bb['keep']=='True',
            both_display_support_pass=aa['keep']==bb['keep']=='True',A_display_support_flags=aa['exclusion_reasons'],B_display_support_flags=bb['exclusion_reasons'],
            filter_scope='both_reference_coordinates_land_only',geographic_registration_independently_validated=False,
            numerical_parsing='numeric_content; bare finite JSON scalars accepted; archived_strict values also included',
            normalization='difference divided by native-map finite P95-P05; do not pool raw errors across units',
            source_dataset=e['dataset'],climate_model=e['climate_model'],variable=e['variable'],source_group=e['source_group'],
            model=meta['model'],model_revision=meta['resolved_revision'],source_run=run.name,trial=0,
            generation_temperature=meta['options']['temperature'],generation_top_p=meta['options']['top-p'],
            marked_image_path=str((dataset/e['image']).resolve()),clean_image_path=str((dataset/e['clean_image']).resolve()),
            reference_json_path=str((dataset/e['reference']).resolve()),original_predictions_path=str((folder/'predictions.csv').resolve()),
            run_metadata_path=str((folder/'run.json').resolve()),original_raw_directory=str(rawdir.resolve()),
            extraction_prompt=meta['prompts']['verbose'],direct_difference_prompt=meta['prompts']['difference'],direct_ratio_prompt=meta['prompts']['ratio'],direct_order_prompt=meta['prompts']['ordering'],
            extraction_record_json=json.dumps(raw['verbose'],ensure_ascii=False),direct_difference_record_json=json.dumps(raw['difference'],ensure_ascii=False),
            direct_ratio_record_json=json.dumps(raw['ratio'],ensure_ascii=False),direct_order_record_json=json.dumps(raw['ordering'],ensure_ascii=False))
        rows.append(row)
    assert len(rows)==len(kept)==46 and {r['pair_id'] for r in rows}==kept
    assert sum(r['calculated_order_correct'] for r in rows)==25 and sum(r['direct_order_correct'] for r in rows)==21
    assert sum(r['calculated_difference_valid'] and r['direct_difference_valid'] for r in rows)==38
    assert sum(r['both_display_support_pass'] for r in rows)==12
    with OUT.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    reopened=read(OUT)
    assert len(reopened)==46
    for before,after in zip(rows,reopened):
        assert before['pair_id']==after['pair_id']
        for field in ('raw_extraction_response','raw_direct_difference_response','raw_direct_ratio_response','raw_direct_order_response'):
            assert before[field]==after[field],field
        for field in ('extraction_record_json','direct_difference_record_json','direct_ratio_record_json','direct_order_record_json'):
            assert json.loads(before[field])==json.loads(after[field]),field
    print(OUT);print(f'{len(rows)} pairs; {len(rows[0])} columns; all raw response text and records round-trip exactly.')


if __name__=='__main__':main()
