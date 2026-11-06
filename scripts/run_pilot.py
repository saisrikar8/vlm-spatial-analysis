#!/usr/bin/env python3
"""Run one frozen pilot example, then automatically evaluate it."""
import getopt
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from experiment_contract import inside, safe_id

REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'


def commands(manifest_path, index, output, samples):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('relationship_direction', 'B_relative_to_A') != 'B_relative_to_A':
        raise ValueError('This runner uses B-A/B/A/B>A and red-circle prompts. Do not use it with the new A-relative-to-B pointer dataset.')
    if samples < 1 or not 0 <= index < len(manifest['examples']):
        raise ValueError('Samples must be positive and index must identify a manifest example.')
    entry = manifest['examples'][index]
    image = inside(manifest_path.parent,entry['image'])
    reference = inside(manifest_path.parent,entry['reference'])
    if hashlib.sha256(image.read_bytes()).hexdigest() != entry['image_sha256']:
        raise ValueError('Image changed since the manifest was prepared.')
    if hashlib.sha256(reference.read_bytes()).hexdigest() != entry['reference_sha256']:
        raise ValueError('Reference changed since the manifest was prepared.')
    run = Path(output).resolve() / safe_id(entry['id'])
    if run.exists():
        raise ValueError(f'Run already exists: {run}. Choose a new output root for retries.')
    script = str(Path(__file__).with_name('run_vlm.py'))
    inference = [sys.executable, '-u', script, '--image', str(image), '--output', str(run),
                 '--samples', str(samples), '--model', 'qwen7b', '--revision', REVISION,
                 '--seed', str(42 + index * 100000), '--direct']
    evaluation = [sys.executable, '-u', script, '--evaluate', str(run),
                  '--reference', str(reference), '--ratio-min-abs', str(entry['ratio_min_abs'])]
    return entry, run, inference, evaluation


def main():
    options, rest = getopt.gnu_getopt(sys.argv[1:], '',
        ['manifest=', 'index=', 'output=', 'samples=', 'dry-run','legacy-unverified'])
    args = dict(options)
    if rest or '--index' not in args or '--output' not in args:
        raise ValueError('Usage: run_pilot.py --index N --output DIR [--samples 30] [--dry-run]')
    if '--dry-run' not in args and '--legacy-unverified' not in args:
        raise ValueError('Archived unvalidated pilot: use current controlled workflow. Explicit legacy reproduction requires --legacy-unverified.')
    samples = int(args.get('--samples', 30))
    manifest_path = Path(args.get('--manifest', 'experiments/pilot/manifest.json'))
    entry, run, inference, evaluation = commands(manifest_path, int(args['--index']), args['--output'], samples)
    print(f"Example: {entry['id']}; {4*samples} model calls", flush=True)
    if '--dry-run' in args:
        print(shlex.join(inference)); print(shlex.join(evaluation))
        return
    run.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(inference, check=True)
    subprocess.run(evaluation, check=True)
    if not all((run / name).stat().st_size for name in ('evaluation.csv', 'distributions.csv')):
        raise ValueError('Evaluation did not produce a CSV.')
    (run / 'pilot_entry.json').write_text(json.dumps(dict(experiment='relationship_distributions', example=entry, samples=samples,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest()), indent=2)+'\n')
    print(f'CSV ready: {run / "evaluation.csv"}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError, getopt.GetoptError) as exc:
        sys.exit(f'Error: {exc}')
