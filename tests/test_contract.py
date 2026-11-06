import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment_contract import validate_dataset, inside, protocol, sha, code_hashes
from prepare_controlled import prepare
from run_vlm import parse_direct
from evaluate_random_pairs import evaluate
from evaluate_dependence import crps
from run_random_pairs import options, planned, collect_resumable


class ContractTests(unittest.TestCase):
    def test_scalar_parser_and_rejections(self):
        for raw in ['3.5','{"answer":3.5}']:
            self.assertEqual(parse_direct(raw,'difference'),({'answer':3.5},None))
        for raw in ['true','NaN','Infinity','3 then 4','{"answer":1,"answer":2}']:
            self.assertIsNotNone(parse_direct(raw,'difference')[1])
        self.assertIsNotNone(parse_direct('3','difference',allow_scalar=False)[1])
        self.assertIsNotNone(parse_direct('1','ordering')[1])

    def test_dataset_tampering_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';prepare(root,sources=1)
            m,t=validate_dataset(root);self.assertEqual(len(t),3)
            truth=root/'ground_truth.csv';original=truth.read_bytes()
            truth.write_bytes(original+b'\n')
            with self.assertRaisesRegex(ValueError,'checksum'):validate_dataset(root)
            truth.write_bytes(original)
            m['examples'][0]['id']=m['examples'][1]['id']
            (root/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'unique'):validate_dataset(root)
            for bad in ['../escape','/tmp/escape']:
                with self.assertRaises(ValueError):inside(root,bad)

    def test_old_dataset_blocked(self):
        with self.assertRaisesRegex(ValueError,'registration'):
            planned(options(['--dataset',str(Path(__file__).resolve().parents[1]/'experiments/random-pairs'),'--map-index','0','--output','unused']))

    def test_oracle_shared_and_independent(self):
        joint=[-5,-5];independent=[-5,-15,5,-5]
        self.assertEqual(crps(joint,-5),0)
        self.assertEqual(crps(independent,-5),1.25)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';m=prepare(root,sources=2)
            for e in m['examples']:
                r=json.loads((root/e['reference']).read_text());pairs=r['feasible_readings']
                a=[x-y for x,y in pairs];b=[x-y for x,_ in pairs for _,y in pairs]
                if e['condition']=='shared_offset':self.assertEqual(len(set(a)),1);self.assertGreater(len(set(b)),1)
                if e['condition']=='independent_offset':
                    self.assertEqual(sorted(a*len(pairs)),sorted(b))

    def test_provenance_rows_and_missing_files(self):
        from contract_fixture import bind_fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'ground_truth.csv').write_text('example_id,map_id,true_difference_a_minus_b,true_ratio_a_over_b,true_a_greater_b\na,m,1,2,True\nb,m,2,3,True\n')
            pred=root/'predictions.csv';header='example_id,experiment,trial,predicted_difference_a_minus_b,predicted_ratio_a_over_b,predicted_a_greater_b\n'
            pred.write_text(header+'a,verbose,0,1,2,true\n')
            bind_fixture(root,pred,repeats=1)
            evaluate(root,pred,root/'valid')
            rows=list(csv.DictReader((root/'valid/per_map_metrics.csv').open()))
            self.assertEqual(rows[2]['expected_pairs'],'2');self.assertEqual(rows[2]['test_status'],'incomplete')
            for bad in ['a,verbose,9,1,2,true\n','a,wrong,0,1,2,true\n']:
                pred.write_text(header+bad)
                with self.assertRaises(ValueError):evaluate(root,pred,root/'invalid')
            pred.write_text(header+'a,verbose,0,1,2,true\n')
            meta=root/'run.json';m=json.loads(meta.read_text());m['prompts']={'changed':'yes'};meta.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'fingerprint'):evaluate(root,pred,root/'invalid')
            meta.unlink()
            with self.assertRaisesRegex(ValueError,'provenance'):evaluate(root,pred,root/'invalid')

    def test_resume_uses_only_missing_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'verbose.jsonl'
            path.write_text(json.dumps(dict(sample=0,seed=42,raw_response='{}'))+'\n')
            def fake(*args,**kw):
                settings=args[3];folder=args[4]
                with (folder/kw['filename']).open('a') as stream:stream.write(''.join(json.dumps(dict(sample=i,seed=settings['seed']+i,raw_response='{}'))+'\n' for i in range(kw.get('start_index',0),settings['samples'])))
            with patch('run_random_pairs.collect',side_effect=fake) as call:
                collect_resumable(None,None,None,{'seed':42,'samples':3},root,None,None,'verbose.jsonl',None)
                self.assertEqual(call.call_args.kwargs['start_index'],1)
                self.assertEqual(call.call_args.args[3]['seed'],42)
                collect_resumable(None,None,None,{'seed':42,'samples':3},root,None,None,'verbose.jsonl',None)
                self.assertEqual(call.call_count,1)
            records=[json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual([r['sample'] for r in records],[0,1,2])
            with self.assertRaises(ValueError):collect_resumable(None,None,None,{'seed':43,'samples':3},root,None,None,'verbose.jsonl',None)


class IntegrationTests(unittest.TestCase):
    def test_dependence_oracle_end_to_end(self):
        from evaluate_dependence import summarize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';m=prepare(data,sources=2);runs=root/'runs'
            for e in m['examples']:
                folder=runs/e['map_id'];folder.mkdir(parents=True)
                ref=json.loads((data/e['reference']).read_text());pairs=ref['feasible_readings']
                rows=[]
                for i in range(4):
                    a,b=pairs[i%len(pairs)]
                    rows.append(dict(example_id=e['id'],experiment='verbose',trial=i,predicted_a=a,predicted_b=b,predicted_difference_a_minus_b=a-b,predicted_ratio_a_over_b=a/b,predicted_a_greater_b=a>b))
                with (folder/'predictions.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                meta=dict(model='oracle-fixture',resolved_revision='fixture',prompts={'verbose':'fixture prompt'},packages={'fixture':'1'},map_id=e['map_id'],manifest_sha256=sha(data/'manifest.json'),code_sha256=code_hashes(),options=dict(experiment='verbose',repeats=4))
                meta['protocol_sha256']=protocol(meta);(folder/'run.json').write_text(json.dumps(meta))
            rows=summarize(data,runs,root/'analysis',minimum=2)
            for r in rows:
                if r['condition']=='shared_offset':self.assertLess(r['delta'],0)
                else:self.assertAlmostEqual(r['delta'],0)
            summary=json.loads((root/'analysis/dependence_summary.json').read_text())
            self.assertTrue(all(r['scored_sources']==2 for r in summary['summary']))

    def test_download_size_gate(self):
        from get_ilamb_data import Client
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,size):return b'x'*size
        with patch('get_ilamb_data.urlopen',return_value=Response()):
            with self.assertRaisesRegex(ValueError,'size limit'):Client(0).get('https://example.test',limit=8)

if __name__=='__main__':unittest.main()
