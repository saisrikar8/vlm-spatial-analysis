#!/usr/bin/env python3
"""Collect and compare direct and calculated relationship distributions."""
import csv
import getopt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
from datetime import datetime, timezone

HELP = """Usage:
  python scripts/run_vlm.py --image IMAGE --output NEW_DIR --samples 30 --direct
  python scripts/run_vlm.py --evaluate RUN --reference JSON [--ratio-min-abs 1e-8]
Inference collects paired A/B readings; --direct also collects B-A, B/A, B>A answers.
Evaluation exports two methods (intact/calculated and direct), plus distributions.csv.
No cross-trial recombination is performed. --dry-run validates inference inputs only.
"""

MODELS = {
    'qwen7b': 'Qwen/Qwen2.5-VL-7B-Instruct',
    'qwen3b': 'Qwen/Qwen2.5-VL-3B-Instruct',
}


def options(argv):
    pairs, extra = getopt.gnu_getopt(argv, '', [
        'image=', 'output=', 'model=', 'revision=', 'samples=', 'labels=', 'seed=',
        'temperature=', 'top-p=', 'max-new-tokens=', 'max-pixels=', 'device=',
        'dtype=', 'dry-run', 'help', 'direct', 'evaluate=', 'reference=', 'ratio-min-abs='])
    if extra:
        raise ValueError(f'Unexpected arguments: {extra}')
    args = dict((k[2:], v if v else True) for k, v in pairs)
    if 'help' in args:
        return args
    if 'evaluate' in args:
        if 'reference' not in args:
            raise ValueError('--evaluate requires --reference FILE')
        if set(args) - {'evaluate', 'reference', 'ratio-min-abs'}:
            raise ValueError('Evaluation is a separate mode; omit inference options.')
        args['ratio-min-abs'] = float(args.get('ratio-min-abs', 1e-8))
        return args
    if 'reference' in args or 'ratio-min-abs' in args:
        raise ValueError('--reference and --ratio-min-abs require --evaluate')
    for name in ('image', 'output'):
        if not isinstance(args.get(name), str):
            raise ValueError(f'--{name} is required')
    for name, default in [('samples', 30), ('seed', 42), ('max-new-tokens', 128),
                          ('max-pixels', 1280 * 28 * 28)]:
        args[name] = int(args.get(name, default))
    for name, default in [('temperature', 0.7), ('top-p', 0.9)]:
        args[name] = float(args.get(name, default))
    if 'direct' in args and len(str(args.get('labels', 'A,B')).split(',')) != 2:
        raise ValueError('--direct requires exactly two labels')
    args.setdefault('model', 'qwen7b')
    args.setdefault('revision', 'main')
    args.setdefault('device', 'cuda')
    args.setdefault('dtype', 'bfloat16')
    args['labels'] = str(args.get('labels', 'A,B')).split(',')
    if args['model'] not in MODELS:
        raise ValueError(f'--model must be one of {list(MODELS)}')
    if args['device'] not in ('cuda', 'cpu') or args['dtype'] not in ('bfloat16', 'float16', 'float32'):
        raise ValueError('Invalid device or dtype')
    if args['device'] == 'cpu' and args['dtype'] != 'float32':
        raise ValueError('Use --dtype float32 with --device cpu')
    if args['samples'] < 1 or args['max-new-tokens'] < 1 or args['max-pixels'] < 4 * 28 * 28:
        raise ValueError('Samples/tokens must be positive; max-pixels must be at least 3136')
    if not 0 <= args['seed'] <= 2**32 - 4 * args['samples']:
        raise ValueError('Seed range exceeds supported 32-bit seeds')
    if not math.isfinite(args['temperature']) or args['temperature'] <= 0 or not 0 < args['top-p'] <= 1:
        raise ValueError('Temperature must be finite and positive; top-p must be in (0,1]')
    if len(set(args['labels'])) != len(args['labels']) or any(
            not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', label) for label in args['labels']):
        raise ValueError('Labels must be unique identifiers such as A,B')
    return args


def make_prompt(labels):
    schema = json.dumps(dict.fromkeys(labels, None))
    return ('Read the scientific map and its color legend. Estimate the numerical value '
            'at the center of each red circle labeled ' + ', '.join(labels) + '. '
            'Return all values together as one complete reading, in the units printed '
            'on the legend. Read the underlying map colors, not the red annotations. '
            'Return only a JSON object with exactly these keys: ' + schema + '. '
            'Replace each null with a JSON number, or leave it null if you cannot read '
            'that location. Do not calculate ratios, differences, or comparisons.')


def parse_reading(raw, labels):
    text = raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text, count=1)
        text = text[:-3].strip()

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate key: {key}')
            result[key] = value
        return result

    try:
        result = json.loads(text, object_pairs_hook=unique)
        if not isinstance(result, dict) or set(result) != set(labels):
            raise ValueError('Response must contain exactly the requested labels')
        for value in result.values():
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError('Values must be finite numbers or null')
        return result, None
    except (ValueError, OverflowError) as error:
        return None, str(error)


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def collect(model, processor, inputs, args, output, torch, set_seed,
            filename='readings.jsonl', parser=None, start_index=0, append=False):
    generation = dict(do_sample=True, temperature=args['temperature'], top_p=args['top-p'],
                      top_k=0, num_beams=1, repetition_penalty=1.0,
                      max_new_tokens=args['max-new-tokens'], use_cache=True)
    complete = 0
    with (output / filename).open('a' if append else 'x') as stream:
        for index in range(start_index,args['samples']):
            seed = args['seed'] + index
            set_seed(seed)
            started = time.monotonic()
            with torch.inference_mode():
                tokens = model.generate(**inputs, **generation)
            generated = tokens[:, inputs['input_ids'].shape[1]:]
            raw = processor.batch_decode(generated, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]
            reading, error = parser(raw) if parser else parse_reading(raw, args['labels'])
            count = generated.shape[1]
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            hit_limit = count >= args['max-new-tokens'] and int(generated[0, -1]) not in eos
            valid = reading is not None and all(value is not None for value in reading.values()) and not hit_limit
            format_compliant=(parse_direct(raw,'ordering',allow_scalar=False)[1] is None if filename=='ordering.jsonl' else parse_reading(raw,['answer'])[1] is None if filename in ('difference.jsonl','ratio.jsonl') else error is None)
            record = dict(format_compliant=format_compliant,sample=index, seed=seed, raw_response=raw, reading=reading,
                          parse_error=error, complete=valid, output_tokens=count,
                          hit_token_limit=hit_limit, seconds=time.monotonic() - started)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            complete += int(valid)
            print(f'{index + 1}/{args["samples"]}: {"complete" if valid else "incomplete"}', flush=True)
    return complete


def direct_prompt(labels, query):
    a, b = labels
    question = {'difference': f'What is {b} minus {a}, in the legend units?',
                'ratio': f'What is {b} divided by {a}?',
                'ordering': f'Is {b} greater than {a}?'}[query]
    kind = 'a JSON boolean (true or false)' if query == 'ordering' else 'a finite JSON number'
    return (f'Read the scientific map and its legend. {a} and {b} refer to the centers '
            f'of the red circles with those labels. {question} '
            f'Return only {{"answer": ...}} with {kind}, or null if unreadable or undefined. '
            'Do not return individual location values.')


def parse_direct(raw, query, allow_scalar=True):
    if query != 'ordering':
        # Accept only a single JSON scalar or the requested object, never prose.
        try:
            scalar = json.loads(raw.strip())
            if allow_scalar and (scalar is None or (type(scalar) in (int, float) and math.isfinite(scalar))):
                return {'answer': scalar}, None
        except (ValueError, OverflowError):
            pass
        return parse_reading(raw, ['answer'])
    # Reuse strict JSON handling, converting only JSON booleans for validation.
    normalized = re.sub(r'\btrue\b', '1', re.sub(r'\bfalse\b', '0', raw))
    reading, error = parse_reading(normalized, ['answer'])
    if error:
        return None, error
    if reading['answer'] is None:
        return reading, None
    if not re.search(r':\s*(true|false)\s*}', raw):
        return None, 'Ordering answer must be a JSON boolean or null'
    return {'answer': bool(reading['answer'])}, None


def evaluate_run(run, reference_path, ratio_min_abs=1e-8):
    import numpy as np
    run, reference_path = Path(run), Path(reference_path)
    metadata = json.loads((run / 'run.json').read_text())
    reference = json.loads(reference_path.read_text())
    reference_image = reference_path.parent.parent / reference['image_path']
    if hashlib.sha256(reference_image.read_bytes()).hexdigest() != metadata['image_sha256']:
        raise ValueError('Reference image does not match the image used for this run.')
    if not math.isfinite(ratio_min_abs) or ratio_min_abs <= 0:
        raise ValueError('--ratio-min-abs must be finite and positive (in the reference units).')
    labels = metadata['options']['labels']
    if len(labels) != 2:
        raise ValueError('Evaluation requires exactly two labels; their order defines A then B.')
    points = {point['label']: point for point in reference['points']}
    if len(points) != len(reference['points']) or any(label not in points for label in labels):
        raise ValueError('Reference labels are missing or duplicated.')
    a_ref, b_ref = [points[label]['value'] for label in labels]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (a_ref, b_ref)):
        raise ValueError('Reference values must be finite numbers.')
    units = points[labels[0]]['units']
    if units != points[labels[1]]['units']:
        raise ValueError('Both values must use the same units.')
    pairs, sample_ids, excluded = [], [], []
    lines = (run / 'readings.jsonl').read_text().splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as error:
            raise ValueError(f'Malformed readings.jsonl line {line_number}; file may be incomplete.') from error
        reading, error = parse_reading(record.get('raw_response', ''), labels)
        if (error or reading is None or any(v is None for v in reading.values())
                or record.get('hit_token_limit') or record.get('complete') is not True):
            excluded.append(dict(line=line_number, sample=record.get('sample'),
                                 reason=error or 'Incomplete, null, or truncated response'))
            continue
        if record['sample'] in sample_ids:
            raise ValueError('Duplicate sample IDs in readings.jsonl.')
        pairs.append([reading[label] for label in labels])
        sample_ids.append(record['sample'])
    n = len(pairs)
    if n == 0:
        raise ValueError('No complete paired readings available for evaluation.')
    pairs = np.asarray(pairs, dtype=float)
    conditions = {'intact': (pairs[:, 0], pairs[:, 1])}
    truth = dict(difference=b_ref-a_ref,
                 ratio=b_ref/a_ref if abs(a_ref) >= ratio_min_abs else None,
                 ordering=bool(b_ref > a_ref))
    summaries, arrays = {}, {'sample_ids': np.asarray(sample_ids), 'readings': pairs}
    for name, (a, b) in conditions.items():
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            ratio = np.divide(b, a, out=np.full(a.shape, np.nan), where=np.abs(a) >= ratio_min_abs)
            difference = b-a
        summaries[name] = dict(pair_count=len(a))
        for query, values in [('difference', difference), ('ratio', ratio), ('ordering', b > a)]:
            arrays[f'{name}_{query}'] = values
            finite = np.isfinite(values)
            valid = values[finite]
            result = dict(valid_count=int(finite.sum()), excluded_count=int((~finite).sum()),
                          reference=truth[query])
            if valid.size and query == 'ordering':
                probability = float(valid.mean())
                result.update(probability_b_greater_a=probability,
                              brier_score=(probability-float(truth[query]))**2,
                              sample_accuracy=float((valid == truth[query]).mean()))
            elif valid.size:
                result.update(mean=float(valid.mean()), median=float(np.median(valid)), std=float(valid.std()),
                              q05=float(np.quantile(valid, .05)), q95=float(np.quantile(valid, .95)))
                if truth[query] is not None:
                    result.update(mean_absolute_error=float(np.abs(valid-truth[query]).mean()),
                                  absolute_error_of_mean=abs(float(valid.mean())-truth[query]))
            summaries[name][query] = result
    notes = ['Direct relationship distributions versus relationships calculated within each paired reading.',
             'No cross-trial recombination is performed.',
             'One-image descriptive comparison; no calibration or statistical significance claim.',
             'Exclusions can cause selection bias; inspect valid counts for each method and query.',
             'q05/q95 describe output spread, not calibrated confidence intervals.',
             'Calculated ratios exclude small denominators; direct answers do not expose a denominator.']
    if n < 2:
        notes.append('Only one reading: smoke test only; too few trials to assess a distribution.')
    if np.unique(pairs, axis=0).shape[0] == 1:
        notes.append('All complete readings are identical; the observed calculated distribution is a point mass, not proof of certainty.')
    result = dict(format_version=2, experiment='relationship_distributions', labels=dict(A=labels[0], B=labels[1]), units=units,
                  reference_values=dict(A=a_ref, B=b_ref), ratio_min_abs=ratio_min_abs,
                  accepted_count=n, excluded=excluded, total_responses=n+len(excluded),
                  unique_pairs=int(np.unique(pairs, axis=0).shape[0]), run_status=metadata.get('status'),
                  source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                                 [('run', run/'run.json'), ('readings', run/'readings.jsonl'),
                                  ('reference', reference_path)]},
                  conditions=summaries, notes=notes)
    direct_rows = []
    direct_summary = {}
    for query in truth:
        path = run / f'direct_{query}.jsonl'
        if not path.exists():
            continue
        valid_answers, rejected = [], []
        seen = set()
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            if entry['sample'] in seen:
                raise ValueError('Duplicate direct sample IDs')
            seen.add(entry['sample'])
            parsed, error = parse_direct(entry['raw_response'], query)
            valid = (parsed is not None and parsed['answer'] is not None
                     and entry.get('complete') is True and not entry.get('hit_token_limit'))
            answer = parsed['answer'] if valid else None
            direct_rows.append(dict(query=query, sample=entry['sample'], answer=answer,
                                    valid=valid, error=error or ('' if valid else 'Null/incomplete/truncated'),
                                    raw_response=entry['raw_response']))
            if valid:
                valid_answers.append(answer)
            else:
                rejected.append(entry['sample'])
        metric = dict(reference=truth[query], valid_count=len(valid_answers),
                      excluded_count=len(rejected), excluded_samples=rejected)
        if valid_answers:
            data = np.asarray(valid_answers)
            metric['mean'] = float(data.mean())
            if query == 'ordering':
                metric['probability_b_greater_a'] = float(data.mean())
                metric['brier_score'] = (float(data.mean())-float(truth[query]))**2
                metric['sample_accuracy'] = float((data == truth[query]).mean())
            else:
                metric.update(median=float(np.median(data)), std=float(data.std()), q05=float(np.quantile(data,.05)),
                              q95=float(np.quantile(data,.95)))
                if truth[query] is not None:
                    metric['mean_absolute_error'] = float(np.abs(data-truth[query]).mean())
                    metric['absolute_error_of_mean'] = abs(float(data.mean())-truth[query])
        direct_summary[query] = metric
        arrays[f'direct_{query}'] = np.asarray(valid_answers)
        result['source_sha256'][f'direct_{query}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    if direct_summary:
        summaries['direct'] = direct_summary
        notes.append('Direct queries use separate image-only prompts, not extracted A/B values. '
                     'Missing direct answers are excluded per query; compare response counts.')
    result['direct_queries_present'] = list(direct_summary)
    # Reject nonfinite derived summaries before replacing prior evaluation artifacts.
    json.dumps(result, allow_nan=False)
    temporary = run / 'calculations.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(run / 'calculations.npz')
    csv_path = run / 'evaluation.csv.tmp'
    fields = ['condition', 'a_sample', 'b_sample', 'a_label', 'b_label', 'units',
              'true_a', 'true_b', 'predicted_a', 'predicted_b',
              'true_difference_b_minus_a', 'predicted_difference_b_minus_a',
              'true_ratio_b_over_a', 'predicted_ratio_b_over_a',
              'true_b_greater_a', 'predicted_b_greater_a',
              'true_ratio_valid', 'predicted_ratio_valid', 'query', 'direct_sample',
              'valid', 'error', 'raw_response']
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for condition, (a, b) in conditions.items():
            for i in range(len(a)):
                ai, bi = i, i
                difference = arrays[f'{condition}_difference'][i]
                ratio = arrays[f'{condition}_ratio'][i]
                writer.writerow(dict(
                    condition=condition, a_sample=sample_ids[ai], b_sample=sample_ids[bi],
                    a_label=labels[0], b_label=labels[1], units=units,
                    true_a=a_ref, true_b=b_ref, predicted_a=float(a[i]), predicted_b=float(b[i]),
                    true_difference_b_minus_a=truth['difference'],
                    predicted_difference_b_minus_a=float(difference) if np.isfinite(difference) else '',
                    true_ratio_b_over_a=truth['ratio'],
                    predicted_ratio_b_over_a=float(ratio) if np.isfinite(ratio) else '',
                    true_b_greater_a=truth['ordering'], predicted_b_greater_a=bool(b[i] > a[i]),
                    true_ratio_valid=truth['ratio'] is not None,
                    predicted_ratio_valid=bool(np.isfinite(ratio))))
        for row in direct_rows:
            query = row['query']
            column = {'difference': 'predicted_difference_b_minus_a',
                      'ratio': 'predicted_ratio_b_over_a',
                      'ordering': 'predicted_b_greater_a'}[query]
            writer.writerow(dict(condition='direct', direct_sample=row['sample'], query=query,
                a_label=labels[0], b_label=labels[1], units=units, true_a=a_ref, true_b=b_ref,
                true_difference_b_minus_a=truth['difference'], true_ratio_b_over_a=truth['ratio'],
                true_b_greater_a=truth['ordering'], true_ratio_valid=truth['ratio'] is not None,
                predicted_ratio_valid=row['valid'] if query == 'ratio' else '',
                valid=row['valid'], error=row['error'], raw_response=row['raw_response'],
                **{column: row['answer']}))
    csv_path.replace(run / 'evaluation.csv')
    # Empirical probability masses, preserving each method's own valid sample count.
    distribution_path = run / 'distributions.csv.tmp'
    with distribution_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['condition', 'query', 'value', 'count',
            'probability', 'valid_count', 'excluded_count', 'reference'])
        writer.writeheader()
        for condition, metrics in summaries.items():
            for query in truth:
                if query not in metrics:
                    continue
                values = arrays[f'{condition}_{query}']
                values = values[np.isfinite(values)]
                unique, counts = np.unique(values, return_counts=True)
                excluded_count = metrics[query]['excluded_count'] + (len(excluded) if condition == 'intact' else 0)
                if not len(values):
                    writer.writerow(dict(condition=condition, query=query, valid_count=0,
                        excluded_count=excluded_count, reference=truth[query]))
                for value, count in zip(unique, counts):
                    writer.writerow(dict(condition=condition, query=query, value=float(value),
                        count=int(count), probability=float(count/len(values)), valid_count=len(values),
                        excluded_count=excluded_count, reference=truth[query]))
    distribution_path.replace(run / 'distributions.csv')
    write_json(run / 'evaluation.json', result)
    print(f'Evaluated {n} complete readings; excluded {len(excluded)} responses.')
    print('Query       Calculated score   Direct score')
    for query in truth:
        metric = 'brier_score' if query == 'ordering' else 'mean_absolute_error'
        scores = [summaries.get(c, {}).get(query, {}).get(metric) for c in ('intact', 'direct')]
        formatted = ['n/a' if v is None else f'{v:.6g}' for v in scores]
        print(f'{query:11} {formatted[0]:18} {formatted[1]}')
    print('Scores: mean absolute error for numerical answers; Brier score for ordering. Lower is better.')
    print(f'Saved {run / "evaluation.json"} plus evaluation.csv, distributions.csv and calculations.npz')
    return 0


def main(argv=None):
    args = options(sys.argv[1:] if argv is None else argv)
    if 'help' in args:
        print(HELP)
        return 0
    if 'evaluate' in args:
        return evaluate_run(args['evaluate'], args['reference'], args['ratio-min-abs'])
    from PIL import Image, ImageOps
    path = Path(args['image']).resolve()
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    prompt = make_prompt(args['labels'])
    metadata = dict(model=MODELS[args['model']], requested_revision=args['revision'],
                    image=str(path), image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    image_size=list(image.size), prompt=prompt, options=args,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    sampling_note='Empirical model outputs, not calibrated posterior samples.')
    if 'direct' in args:
        metadata['direct_prompts'] = {q: direct_prompt(args['labels'], q)
                                      for q in ('difference', 'ratio', 'ordering')}
    if 'dry-run' in args:
        print(json.dumps(metadata, indent=2))
        return 0
    output = Path(args['output'])
    if output.exists():
        raise ValueError('Output directory already exists; choose a new run directory.')
    import torch
    import transformers
    from huggingface_hub import model_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed
    if args['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable. Run inside a Slurm GPU allocation.')
    if args['device'] == 'cuda' and args['dtype'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support bfloat16; use --dtype float16.')
    revision = model_info(metadata['model'], revision=args['revision']).sha
    metadata.update(resolved_revision=revision, started_utc=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(),
                    packages={name: importlib.metadata.version(name) for name in
                              ('torch', 'transformers', 'accelerate', 'huggingface-hub', 'pillow')},
                    gpu=torch.cuda.get_device_name(0) if args['device'] == 'cuda' else None,
                    torch_cuda=torch.version.cuda, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                    status='loading')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'run.json', metadata)
    try:
        set_seed(args['seed'])
        processor = AutoProcessor.from_pretrained(metadata['model'], revision=revision,
                                                 min_pixels=4 * 28 * 28, max_pixels=args['max-pixels'])
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            metadata['model'], revision=revision, torch_dtype=getattr(torch, args['dtype']),
            device_map={'': args['device']}, attn_implementation='sdpa').eval()
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors='pt').to(model.device)
        metadata.update(status='running', rendered_prompt=text,
                        input_tokens=inputs['input_ids'].shape[1],
                        image_grid_thw=inputs['image_grid_thw'].tolist(),
                        model_generation_config=model.generation_config.to_dict(),
                        generation_overrides=dict(do_sample=True, top_k=0, num_beams=1,
                            repetition_penalty=1.0, use_cache=True, temperature=args['temperature'],
                            top_p=args['top-p'], max_new_tokens=args['max-new-tokens']))
        write_json(output / 'run.json', metadata)
        complete = collect(model, processor, inputs, args, output, torch, set_seed)
        if 'direct' in args:
            metadata['direct_complete'] = {}
            metadata['direct_rendered_prompts'] = {}
            for offset, (query, direct_text) in enumerate(metadata['direct_prompts'].items(), 1):
                messages = [{'role': 'user', 'content': [
                    {'type': 'image'}, {'type': 'text', 'text': direct_text}]}]
                rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                direct_inputs = processor(text=[rendered], images=[image], padding=True,
                                          return_tensors='pt').to(model.device)
                metadata['direct_rendered_prompts'][query] = rendered
                write_json(output / 'run.json', metadata)
                direct_args = dict(args, seed=args['seed'] + offset * args['samples'])
                print(f'Direct query: {query}', flush=True)
                metadata['direct_complete'][query] = collect(
                    model, processor, direct_inputs, direct_args, output, torch, set_seed,
                    filename=f'direct_{query}.jsonl', parser=lambda raw, q=query: parse_direct(raw, q))
        metadata.update(status='finished', complete_readings=complete,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        if args['device'] == 'cuda':
            metadata['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated()
        write_json(output / 'run.json', metadata)
        print(f'Saved {args["samples"]} responses ({complete} complete) to {output}')
        return 0
    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        error=str(error))
        write_json(output / 'run.json', metadata)
        raise


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, ImportError, getopt.GetoptError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(2)
