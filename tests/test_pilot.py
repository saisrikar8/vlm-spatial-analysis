"""CPU-only checks for pilot integrity, evaluation, and aggregation."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_pilot
from run_vlm import evaluate_run
from summarize_pilot import summarize


class PilotTests(unittest.TestCase):
    def test_all_inputs_and_ground_truth(self):
        import netCDF4
        manifest = json.loads((ROOT/'experiments/pilot/manifest.json').read_text())
        records = json.loads((ROOT/'data/ilamb_maps/alignment.json').read_text())['records']
        self.assertEqual(len(manifest['examples']), 20)
        ordering = []
        with tempfile.TemporaryDirectory() as tmp:
            for index, entry in enumerate(manifest['examples']):
                _, _, command, _ = run_pilot.commands(ROOT/'experiments/pilot/manifest.json', index, tmp, 30)
                self.assertNotIn('--reference', command)
                ref = json.loads((ROOT/'experiments/pilot'/entry['reference']).read_text())
                source = records[entry['sample']]
                with netCDF4.Dataset(ROOT/'data/ilamb_maps'/source['netcdf_path']) as nc:
                    array = nc[source['variable']]
                    for point in ref['points']:
                        self.assertEqual(float(array[tuple(point['index'])]), point['value'])
                ordering.append(ref['points'][1]['value'] > ref['points'][0]['value'])
        self.assertEqual(sum(ordering), 10)

    def test_runner_and_summary(self):
        manifest_path = ROOT/'experiments/pilot/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        real_subprocess = run_pilot.subprocess.run
        def fake_inference(command, check):
            if '--evaluate' in command:
                return real_subprocess(command, check=check, stdout=-1)
            output = Path(command[command.index('--output')+1]); output.mkdir()
            image = Path(command[command.index('--image')+1])
            entry = next(e for e in manifest['examples'] if Path(e['image']).name == image.name)
            ref = json.loads((ROOT/'experiments/pilot'/entry['reference']).read_text())
            a,b = [p['value'] for p in ref['points']]
            (output/'run.json').write_text(json.dumps(dict(image_sha256=entry['image_sha256'],
                status='finished', options=dict(labels=['A','B']))))
            def save(name, answer):
                (output/name).write_text(''.join(json.dumps(dict(sample=i, complete=True,
                    hit_token_limit=False, raw_response=json.dumps(answer)))+'\n' for i in range(2)))
            save('readings.jsonl', dict(A=a,B=b))
            for query, answer in [('difference',b-a),('ratio',b/a),('ordering',b>a)]:
                save(f'direct_{query}.jsonl', dict(answer=answer))
        with tempfile.TemporaryDirectory() as tmp:
            for index in (0,1):
                with patch.object(sys, 'argv', ['run_pilot.py','--manifest',str(manifest_path),
                    '--index',str(index),'--output',tmp,'--samples','2','--legacy-unverified']), \
                    patch.object(run_pilot.subprocess,'run',side_effect=fake_inference), \
                    contextlib.redirect_stdout(io.StringIO()):
                    run_pilot.main()
            summarize(manifest_path, tmp)
            rows = list(csv.DictReader(io.StringIO((Path(tmp)/'macro_metrics.csv').read_text())))
            self.assertEqual(len(rows),6)
            self.assertTrue(all(r['source_groups']=='1' and float(r['macro_score'])==0 for r in rows))
            statuses = list(csv.DictReader(io.StringIO((Path(tmp)/'completion.csv').read_text())))
            self.assertEqual(sum(r['status']=='complete' for r in statuses),2)
            # A missing member excludes the map from every condition's aggregate.
            (Path(tmp)/manifest['examples'][1]['id']/'pilot_entry.json').unlink()
            summarize(manifest_path,tmp)
            rows = list(csv.DictReader(io.StringIO((Path(tmp)/'macro_metrics.csv').read_text())))
            self.assertTrue(all(r['source_groups']=='0' for r in rows))
            with self.assertRaisesRegex(ValueError,'already exists'):
                run_pilot.commands(manifest_path,0,tmp,30)


if __name__ == '__main__':
    unittest.main()
