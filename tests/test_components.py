import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from component_diagnostics import prepare,load_dataset,prompts,task_scores,evaluate,fingerprint,TASKS,GENERATION
from prepare_controlled import prepare as prepare_source
from interval_experiment import bounds
from experiment_contract import sha


class ComponentTests(unittest.TestCase):
    def fixture(self,root,with_runs=False):
        original=root/'original';prepare_source(original,sources=2)
        data=root/'data';manifest=prepare(original,data)
        if not with_runs:return data,manifest
        runs=root/'runs';runs.mkdir()
        for i,e in enumerate(manifest['examples']):
            ref=json.loads((data/e['reference']).read_text());folder=runs/e['id'];folder.mkdir()
            meta=dict(experiment='component_diagnostics_v1',status='finished',example_id=e['id'],tasks=list(TASKS),dataset_sha256=sha(data/'manifest.json'),
                prompts=prompts(ref),generation=GENERATION,options=dict(seed=4343,**{'map-index':i}),model='fixture',resolved_revision='fixed',
                packages={'fixture':'1'},code_sha256={'fixture':'fixed'})
            meta['protocol_sha256']=fingerprint(meta);(folder/'run.json').write_text(json.dumps(meta))
            rows=[]
            for task in TASKS:
                if task.startswith('color_'):answer=dict(answer=ref['displayed_'+task[-1]])
                else:
                    c,q=next((c,task[len(c)+1:]) for c in ref['interval_references'] if task.startswith(c+'_'))
                    lo,hi=bounds(ref['interval_references'][c],q);answer=dict(lower=lo,upper=hi)
                rows.append(dict(task=task,seed=4343+i*len(TASKS)+TASKS.index(task),raw_response=json.dumps(answer),hit_token_limit=False,
                    input_modality='image' if task.startswith('color_') else 'text'))
            (folder/'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return data,manifest,runs

    def test_crops_remove_rules_and_preserve_pixels(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            data,m=self.fixture(Path(tmp));load_dataset(data)
            for e in m['examples']:
                with Image.open(data/e['image']) as image:self.assertEqual(image.size,(750,385))
                ref=json.loads((data/e['reference']).read_text());p=prompts(ref)
                self.assertEqual(len(p),11)
                self.assertIn('independently',p['independent_offset_difference'])
                self.assertIn(str(ref['displayed_a']),p['shared_offset_a'])
                self.assertNotIn('offset',p['color_a'])

    def test_color_scalar_null_and_truncation(self):
        ref={'displayed_a':4}
        self.assertEqual(task_scores(dict(task='color_a',raw_response='3'),ref)['absolute_error'],1)
        self.assertTrue(task_scores(dict(task='color_a',raw_response='{"answer":4}'),ref)['exact'])
        self.assertFalse(task_scores(dict(task='color_a',raw_response='null'),ref)['valid'])
        self.assertFalse(task_scores(dict(task='color_a',raw_response='4',hit_token_limit=True),ref)['valid'])

    def test_oracle_complete_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,m,runs=self.fixture(root,True)
            rows=evaluate(data,runs,root/'evaluation')
            self.assertEqual(len(rows),11);self.assertTrue(all(r['exact_rate']==1 and r['mean_numerical_error']==0 for r in rows))
            ev=json.loads((root/'evaluation/evaluation.json').read_text());self.assertEqual(ev['expected_answers'],22)

    def test_missing_source_keeps_scope(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,m,runs=self.fixture(root,True);shutil.rmtree(runs/m['examples'][0]['id'])
            rows=evaluate(data,runs,root/'evaluation')
            self.assertTrue(all(r['expected']==2 and r['attempted']==1 and r['exact_rate']==.5 for r in rows))

    def test_matched_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,m,runs=self.fixture(root,True);baseline=root/'baseline.csv';rows=[]
            for e in m['examples']:
                ref=json.loads((data/e['reference']).read_text())
                for c,sourceid in ref['source_examples'].items():
                    for q in ('a','b','difference'):
                        lo,hi=bounds(ref['interval_references'][c],q)
                        rows.append(dict(example_id=sourceid,query=q,reference_lower=lo,reference_upper=hi,valid=True,endpoint_mae=1))
            with baseline.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            evaluate(data,runs,root/'evaluation',baseline)
            with (root/'evaluation/matched_image_text.csv').open() as stream:matched=list(csv.DictReader(stream))
            self.assertEqual(len(matched),18);self.assertTrue(all(float(r['text_minus_image_error'])==-1 for r in matched))

    def test_changed_prompt_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,m,runs=self.fixture(root,True)
            p=runs/m['examples'][0]['id']/'run.json';meta=json.loads(p.read_text());meta['prompts']['color_a']='changed';p.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'mismatch'):evaluate(data,runs,root/'evaluation')

if __name__=='__main__':unittest.main()
