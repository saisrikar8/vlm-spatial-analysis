import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from interval_experiment import parse_interval,bounds,score,evaluate,prompts,GENERATION,QUERIES
from prepare_controlled import prepare
from experiment_contract import sha,protocol


class IntervalTests(unittest.TestCase):
    def test_parser(self):
        self.assertEqual(parse_interval('{"lower":2,"upper":2}'),({'lower':2,'upper':2},None))
        for raw in ['{"lower":3,"upper":2}','{"lower":true,"upper":4}','{"lower":0,"upper":Infinity}',
                    '{"lower":1,"lower":2,"upper":3}','[1,2]','{"lower":null,"upper":2}']:
            self.assertIsNotNone(parse_interval(raw)[1])
        self.assertEqual(parse_interval('{"lower":null,"upper":null}'),(None,'unreadable'))

    def test_shared_and_independent_bounds(self):
        ref=dict(feasible_readings=[[10,15],[20,25]],ground_truth=dict(true_a=10,true_b=15,true_difference_a_minus_b=-5))
        self.assertEqual(bounds(ref,'a'),(10,20));self.assertEqual(bounds(ref,'b'),(15,25));self.assertEqual(bounds(ref,'difference'),(-5,-5))
        ref['feasible_readings']=[[10,15],[10,25],[20,15],[20,25]]
        self.assertEqual(bounds(ref,'difference'),(-15,5))

    def test_truth_containment_is_not_feasible_containment(self):
        ref=dict(feasible_readings=[[10,15],[20,25]],ground_truth=dict(true_a=10,true_b=15,true_difference_a_minus_b=-5))
        row=score(dict(raw_response='{"lower":10,"upper":10}'),ref,'a')
        self.assertTrue(row['contains_truth']);self.assertFalse(row['contains_feasible_range']);self.assertFalse(row['exact_bounds'])
        row=score(dict(raw_response='{"lower":0,"upper":30}'),ref,'a')
        self.assertTrue(row['contains_feasible_range']);self.assertFalse(row['exact_bounds']);self.assertEqual(row['width_absolute_error'],20)
        self.assertFalse(score(dict(raw_response='{"lower":10,"upper":20}',hit_token_limit=True),ref,'a')['valid'])

    def fixture(self,root,missing=False):
        data=root/'data';manifest=prepare(data,sources=2);runs=root/'runs';runs.mkdir()
        for i,e in enumerate(manifest['examples']):
            folder=runs/e['map_id'];folder.mkdir()
            ref=json.loads((data/e['reference']).read_text())
            meta=dict(experiment='direct_feasible_intervals_v1',status='finished',example_id=e['id'],queries=list(QUERIES),
                source_manifest_sha256=sha(data/'manifest.json'),prompts=prompts(),generation=GENERATION,model='fixture',resolved_revision='fixed',
                options=dict(seed=4242,**{'map-index':i}),code_sha256={'fixture':'fixed'},packages={'fixture':'1'})
            meta['protocol_sha256']=protocol(meta);(folder/'run.json').write_text(json.dumps(meta))
            rows=[]
            for q in QUERIES:
                lo,hi=bounds(ref,q)
                rows.append(dict(query=q,seed=4242+i*3+QUERIES.index(q),raw_response=json.dumps(dict(lower=lo,upper=hi)),hit_token_limit=False))
            (folder/'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return data,runs,manifest

    def test_oracle_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,runs,m=self.fixture(root)
            summary=evaluate(data,runs,root/'evaluation')
            self.assertEqual(len(summary),9)
            self.assertTrue(all(r['exact_bounds_rate']==1 for r in summary))
            shared={r['query']:r for r in summary if r['condition']=='shared_offset'}
            self.assertEqual(shared['a']['reference_width'],10);self.assertEqual(shared['b']['reference_width'],10)
            self.assertEqual(shared['difference']['reference_width'],0)
            independent=next(r for r in summary if r['condition']=='independent_offset' and r['query']=='difference')
            self.assertEqual(independent['reference_width'],20)

    def test_missing_answers_do_not_shrink_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,runs,m=self.fixture(root)
            folder=runs/m['examples'][0]['map_id']
            meta=json.loads((folder/'run.json').read_text());meta['status']='failed';(folder/'run.json').write_text(json.dumps(meta))
            lines=(folder/'responses.jsonl').read_text().splitlines();(folder/'responses.jsonl').write_text(lines[0]+'\n')
            summary=evaluate(data,runs,root/'evaluation')
            row=next(r for r in summary if r['condition']=='clear' and r['query']=='difference')
            self.assertEqual(row['expected'],2);self.assertEqual(row['attempted'],1);self.assertEqual(row['exact_bounds_rate'],.5)

    def test_protocol_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,runs,m=self.fixture(root)
            path=runs/m['examples'][0]['map_id']/'run.json';meta=json.loads(path.read_text());meta['prompts']['a']='changed';path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'mismatch'):evaluate(data,runs,root/'evaluation')

if __name__=='__main__':unittest.main()
