#!/usr/bin/env python3
"""Summarize complete pilot runs with equal source-group weights."""
import csv
import getopt
import hashlib
import json
from pathlib import Path
import statistics
import sys


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def summarize(manifest_path, runs):
    manifest_path, runs = Path(manifest_path), Path(runs)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text())
    rows, status = [], []
    for example in manifest['examples']:
        directory = runs / example['id']
        record = dict(example_id=example['id'], status='missing', accepted_count=0)
        status.append(record)
        if not (directory/'evaluation.json').exists():
            continue
        if not (directory/'pilot_entry.json').exists():
            record['status'] = 'incomplete_runner'; continue
        provenance = json.loads((directory/'pilot_entry.json').read_text())
        if provenance['manifest_sha256'] != manifest_hash or provenance['example'] != example:
            raise ValueError(f"Manifest mismatch for {example['id']}")
        result = json.loads((directory/'evaluation.json').read_text())
        record['accepted_count'] = result['accepted_count']
        expected = provenance['samples']
        direct = result['conditions'].get('direct', {})
        if (result['run_status'] != 'finished' or result['total_responses'] != expected or
            any(q not in direct or direct[q]['valid_count'] + direct[q]['excluded_count'] != expected
                for q in ('difference', 'ratio', 'ordering'))):
            record['status'] = 'incomplete'; continue
        record['status'] = 'smoke_only' if expected < 2 else 'complete'
        if expected < 2:
            continue
        for condition in ('intact', 'direct'):
            for query in ('difference', 'ratio', 'ordering'):
                metric = result['conditions'][condition][query]
                key = 'brier_score' if query == 'ordering' else 'mean_absolute_error'
                value = metric.get(key)
                score = value / example['scale_p95_minus_p05'] if value is not None and query == 'difference' else value
                rows.append(dict(example_id=example['id'], source_group=example['source_group'],
                    variable=example['variable'], units=example['units'], pair_type=example['pair_type'],
                    condition=condition, query=query, valid_count=metric['valid_count'],
                    excluded_count=metric['excluded_count'] + (len(result['excluded']) if condition == 'intact' else 0),
                    predicted_mean=metric.get('mean', metric.get('probability_b_greater_a')),
                    predicted_median=metric.get('median'), std=metric.get('std'),
                    q05=metric.get('q05'), q95=metric.get('q95'),
                    absolute_error_of_mean=metric.get('absolute_error_of_mean'), raw_error=value,
                    score=score, score_type='normalized_MAE' if query=='difference' else key))
    # Include a group only when BOTH selected pairs have scores for ALL conditions.
    # This gives each comparison identical map coverage and reports omissions explicitly.
    groups = {}
    for row in rows:
        groups.setdefault((row['source_group'], row['query']), []).append(row)
    macro = []
    for query in ('difference', 'ratio', 'ordering'):
        eligible = [group for (source, q), group in groups.items() if q == query and
                    len(group) == 4 and all(row['score'] is not None for row in group)]
        for condition in ('intact', 'direct'):
            scores = [statistics.mean(row['score'] for row in group if row['condition'] == condition)
                      for group in eligible]
            macro.append(dict(query=query, condition=condition, source_groups=len(scores),
                score_type='normalized_MAE' if query=='difference' else ('brier_score' if query=='ordering' else 'mean_absolute_error'),
                macro_score=statistics.mean(scores) if scores else None))
    runs.mkdir(parents=True, exist_ok=True)
    fields = ['example_id','source_group','variable','units','pair_type','condition','query',
              'valid_count','excluded_count','predicted_mean','predicted_median','std','q05','q95',
              'absolute_error_of_mean','raw_error','score','score_type']
    write_csv(runs/'per_example_metrics.csv', rows, fields)
    write_csv(runs/'macro_metrics.csv', macro, ['query','condition','source_groups','score_type','macro_score'])
    write_csv(runs/'completion.csv', status, ['example_id','status','accepted_count'])
    print(f"{sum(r['status']=='complete' for r in status)}/{len(status)} complete examples. Summaries: {runs}")


if __name__ == '__main__':
    try:
        options, rest = getopt.gnu_getopt(sys.argv[1:], '', ['manifest=', 'runs='])
        args = dict(options)
        if rest or '--runs' not in args:
            raise ValueError('Usage: summarize_pilot.py --runs DIR [--manifest experiments/pilot/manifest.json]')
        summarize(args.get('--manifest','experiments/pilot/manifest.json'), args['--runs'])
    except (OSError, ValueError, KeyError, getopt.GetoptError) as exc:
        sys.exit(f'Error: {exc}')
