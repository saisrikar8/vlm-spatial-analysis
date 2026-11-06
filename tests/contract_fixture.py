import csv
import json
from experiment_contract import sha,code_hashes,protocol


def bind_fixture(root, prediction, repeats=2, mode='both'):
    with (root/'ground_truth.csv').open() as stream:rows=list(csv.DictReader(stream))
    entries=[]
    (root/'image').write_bytes(b'fixture')
    for r in rows:
        truth={k:(r[k].lower()=='true' if k=='true_a_greater_b' else float(r[k])) for k in r if k.startswith('true_')}
        name=r['example_id']+'.json';(root/name).write_text(json.dumps(dict(ground_truth=truth)))
        entries.append(dict(id=r['example_id'],map_id=r['map_id'],image='image',clean_image='image',reference=name,image_sha256=sha(root/'image'),clean_image_sha256=sha(root/'image'),reference_sha256=sha(root/name)))
    m=dict(relationship_direction='A_relative_to_B',examples=entries,example_count=len(entries),map_count=len({r['map_id'] for r in rows}),acceptance=dict(status='validated',evidence='fixture'),ground_truth_sha256=sha(root/'ground_truth.csv'))
    (root/'manifest.json').write_text(json.dumps(m))
    meta=dict(model='fixture-model',resolved_revision='fixture-revision',packages={'fixture':'1'},prompts={q:'fixture prompt' for q in (['verbose','difference','ratio','ordering'] if mode=='both' else ['verbose'] if mode=='verbose' else ['difference','ratio','ordering'])},map_id=rows[0]['map_id'],manifest_sha256=sha(root/'manifest.json'),options=dict(experiment=mode,repeats=repeats),code_sha256=code_hashes())
    meta['protocol_sha256']=protocol(meta)
    prediction.with_name('run.json').write_text(json.dumps(meta))
