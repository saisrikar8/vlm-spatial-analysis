"""Relationship distributions use actual trials, never cross-trial pairs."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from run_vlm import evaluate_run


def fixture(root):
    points = root/'points'; points.mkdir()
    image = points/'map.png'; image.write_bytes(b'fixture-image')
    reference = points/'reference.json'
    reference.write_text(json.dumps(dict(image_path='points/map.png', points=[
        dict(label='A', value=1., units='units'), dict(label='B', value=2., units='units')])))
    run = root/'run'; run.mkdir()
    (run/'run.json').write_text(json.dumps(dict(status='finished',
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(), options=dict(labels=['A','B']))))
    def save(name, answers):
        with (run/name).open('w') as stream:
            for i, answer in enumerate(answers):
                stream.write(json.dumps(dict(sample=i, complete=True, hit_token_limit=False,
                    raw_response=answer))+'\n')
    save('readings.jsonl', ['{"A":1,"B":2}', '{"A":2,"B":4}', '{"A":0,"B":3}', 'bad'])
    save('direct_difference.jsonl', ['{"answer":1}', '{"answer":3}', '{"answer":null}', 'bad'])
    save('direct_ratio.jsonl', ['{"answer":2}', '{"answer":2}', '{"answer":null}', 'bad'])
    save('direct_ordering.jsonl', ['{"answer":true}', '{"answer":false}', '{"answer":null}', 'bad'])
    return run, reference


class DistributionTests(unittest.TestCase):
    def test_probabilities_and_exclusions(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            run, reference = fixture(Path(tmp))
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_run(run, reference, .1)
            result = json.loads((run/'evaluation.json').read_text())
            self.assertEqual(result['format_version'], 2)
            self.assertEqual(set(result['conditions']), {'intact','direct'})
            rows = list(csv.DictReader(io.StringIO((run/'evaluation.csv').read_text())))
            self.assertEqual(len(rows), 15)  # Three paired trials + twelve direct attempts.
            self.assertTrue(all(r['a_sample']==r['b_sample'] for r in rows if r['condition']=='intact'))
            distributions = list(csv.DictReader(io.StringIO((run/'distributions.csv').read_text())))
            for condition in ('intact','direct'):
                for query in ('ratio','difference','ordering'):
                    subset = [r for r in distributions if r['condition']==condition and r['query']==query]
                    self.assertAlmostEqual(sum(float(r['probability']) for r in subset),1.)
            for condition in ('intact','direct'):
                ratio = [r for r in distributions if r['condition']==condition and r['query']=='ratio']
                self.assertEqual(len(ratio),1)
                self.assertEqual(float(ratio[0]['value']),2.)
                self.assertEqual(int(ratio[0]['count']),2)
                self.assertEqual(int(ratio[0]['excluded_count']),2)
            self.assertEqual(result['conditions']['direct']['ordering']['brier_score'],.25)
            with np.load(run/'calculations.npz') as arrays:
                self.assertFalse(any('independent' in name for name in arrays.files))
                np.testing.assert_equal(arrays['intact_ratio'], [2.,2.,np.nan])

    def test_missing_and_all_invalid_direct(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, reference = fixture(Path(tmp))
            (run/'direct_ratio.jsonl').unlink()
            (run/'direct_difference.jsonl').write_text(json.dumps(dict(sample=0,
                raw_response='{"answer":null}', complete=True))+'\n')
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_run(run, reference)
            rows = list(csv.DictReader(io.StringIO((run/'distributions.csv').read_text())))
            self.assertFalse(any(r['condition']=='direct' and r['query']=='ratio' for r in rows))
            row = next(r for r in rows if r['condition']=='direct' and r['query']=='difference')
            self.assertEqual(row['valid_count'],'0')
            self.assertEqual(row['excluded_count'],'1')
            self.assertEqual(row['probability'],'')


if __name__ == '__main__':
    unittest.main()
