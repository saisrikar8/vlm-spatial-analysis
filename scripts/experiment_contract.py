"""Dataset and protocol acceptance checks shared by generation and evaluation."""
import csv
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('Unsafe identifier')
    return value


def inside(root, relative):
    root = Path(root).resolve()
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Dataset paths must be relative and contained')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Dataset path escapes root')
    return resolved


def validate_dataset(root, legacy=False):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    entries = manifest.get('examples', [])
    if not legacy:
        queries=manifest.get('queries',['difference','ratio','ordering'])
        if not queries or len(queries)!=len(set(queries)) or set(queries)-{'difference','ratio','ordering'}:
            raise ValueError('Invalid declared queries')
    ids = [safe_id(e['id']) for e in entries]
    if not entries or len(ids) != len(set(ids)):
        raise ValueError('Dataset requires unique manifest example IDs')
    if manifest.get('relationship_direction') != 'A_relative_to_B':
        raise ValueError('Expected A-relative-to-B dataset')
    if not legacy and manifest.get('acceptance', {}).get('status') != 'validated':
        raise ValueError('Dataset lacks validated registration/display support; archive analysis requires explicit legacy mode')
    truth_path = root/'ground_truth.csv'
    if not legacy and manifest.get('ground_truth_sha256') != sha(truth_path):
        raise ValueError('Ground truth checksum mismatch')
    with truth_path.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len({r['example_id'] for r in rows}) or {r['example_id'] for r in rows} != set(ids):
        raise ValueError('Truth membership must exactly match manifest')
    truths = {r['example_id']: r for r in rows}
    required={'example_id','map_id','true_difference_a_minus_b','true_ratio_a_over_b','true_a_greater_b'}
    if any(not required.issubset(r) for r in rows):raise ValueError('Truth schema missing required columns')
    if any(r['true_a_greater_b'].lower() not in ('true','false') for r in rows):raise ValueError('Truth ordering must be boolean')
    if not legacy and not manifest.get('acceptance',{}).get('evidence'):
        raise ValueError('Acceptance evidence required')
    if manifest.get('example_count') != len(entries) or manifest.get('map_count') != len({e['map_id'] for e in entries}):
        raise ValueError('Manifest counts mismatch')
    for e in entries:
        safe_id(e['map_id'])
        if truths[e['id']]['map_id'] != e['map_id']:
            raise ValueError('Truth map assignment mismatch')
        for key in ('image', 'clean_image', 'reference'):
            if sha(inside(root,e[key])) != e[key+'_sha256']:
                raise ValueError('Input checksum mismatch: '+e[key])
        ref = json.loads(inside(root,e['reference']).read_text())
        for key, value in ref['ground_truth'].items():
            actual = truths[e['id']].get(key)
            if isinstance(value, bool):
                good = actual.lower() == str(value).lower()
            elif value is None:
                good = actual == ''
            else:
                good = float(actual) == value
            if not good:
                raise ValueError('CSV/reference truth mismatch')
    return manifest, truths


def protocol(metadata):
    settings = {k:v for k,v in metadata['options'].items()
                if k not in ('map-index','output','dataset','dry-run','resume')}
    value = dict(model=metadata.get('model'), revision=metadata.get('resolved_revision'),
        settings=settings, prompts=metadata.get('prompts'), packages=metadata.get('packages'),
        code=metadata.get('code_sha256'), generation=metadata.get('model_generation_config'),python=metadata.get('python'),cuda=metadata.get('cuda'),gpu=metadata.get('gpu'))
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def code_hashes():
    base = Path(__file__).parent
    return {name:sha(base/name) for name in ('run_random_pairs.py','run_vlm.py','experiment_contract.py','evaluate_random_pairs.py')}
