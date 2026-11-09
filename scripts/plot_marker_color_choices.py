#!/usr/bin/env python3
"""Count marker-colored swatch choices directly from raw test responses."""
import csv
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'analysis/hpc-marker-color-control-v1'


def matches(rgb, color):
    # Fixed coarse color definitions; exact marker RGB never appears as a choice.
    red, green, blue = rgb
    return red >= 180 and green <= 60 and (blue <= 60 if color == 'red' else blue >= 180)


def main():
    experiments = [
        ('Original red', 'color-matching-v1', 'hpc-color-matching-v1/color-matching-40944345', 'red', None),
        ('Red (neutral prompt)', 'marker-color-control-v1', 'hpc-marker-color-control-v1/marker-color-control-40947249', 'red', 'red'),
        ('Magenta (neutral prompt)', 'marker-color-control-v1', 'hpc-marker-color-control-v1/marker-color-control-40947249', 'magenta', 'magenta'),
    ]
    rows = []
    for label, dataset, run, color, condition in experiments:
        manifest = json.loads((ROOT / 'experiments' / dataset / 'manifest.json').read_text())
        raw = [json.loads(line) for line in (ROOT / 'analysis' / run / 'responses.jsonl').read_text().splitlines()]
        responses = {r['id']: r['raw_response'].strip() for r in raw}
        if len(responses) != len(raw):
            raise ValueError('Duplicate responses')
        examples = [e for e in manifest['examples'] if e['split'] == 'test'
                    and (condition is None or e['condition'] == condition)]
        selected = available = valid = 0
        for example in examples:
            available += any(matches(c, color) for c in example['swatches'])
            prediction = responses.get(example['id'], '')
            if re.fullmatch('[1-6]', prediction):
                valid += 1
                selected += matches(example['swatches'][int(prediction) - 1], color)
        rows.append(dict(test=label,marker_color=color,test_points=len(examples),valid=valid,
                         marker_color_option_available=available,marker_color_selected=selected))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'marker_color_choices.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['Original red', 'Red\n(neutral prompt)', 'Magenta\n(neutral prompt)'],
                  [r['marker_color_selected'] for r in rows], color=['#c44e52', '#c44e52', '#b64cbb'], width=.6)
    ax.bar_label(bars, padding=4)
    ax.set(ylabel='Marker-colored choices (count)', title='Choices matching the marker color', ylim=(0, 32))
    ax.set_yticks(range(0, 33, 4))
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for extension in ['png','svg']:
        fig.savefig(OUTPUT / f'marker_color_choices.{extension}', dpi=180)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
