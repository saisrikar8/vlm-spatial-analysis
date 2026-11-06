import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from structured_prototype import parse_representation,execute,score_representation,dataset,prompts,fingerprint,evaluate,TASKS,CONDITIONS,GENERATION
from prepare_controlled import prepare
from component_diagnostics import prepare as prepare_components
from experiment_contract import sha


def representation(a=4,b=2,shared=True):
    return dict(values={'A':dict(base=a,offset='s'),'B':dict(base=b,offset='s' if shared else 't')},
                offsets={'s':[10,20]} if shared else {'s':[10,20],'t':[10,20]})


class StructuredTests(unittest.TestCase):
    def test_symbolic_sharing_not_equal_domain_values(self):
        shared=execute(representation());separate=execute(representation(shared=False))
        self.assertEqual(shared,dict(a=(14,24),b=(12,22),difference=(2,2)))
        self.assertEqual(separate,dict(a=(14,24),b=(12,22),difference=(-8,12)))

    def test_parser_rejects_unsafe_or_ambiguous_data(self):
        valid=representation();self.assertEqual(parse_representation(json.dumps(valid)),(valid,None))
        self.assertEqual(parse_representation('null'),(None,'unreadable'))
        cases=[]
        value=representation();value['values']['A']['base']=True;cases.append(value)
        value=representation();value['values']['A']['offset']='missing';cases.append(value)
        value=representation();value['offsets']['unused']=[1];cases.append(value)
        value=representation();value['offsets']['s']=[10,10];cases.append(value)
        value=representation();value['offsets']['s']=[float('inf')];cases.append(value)
        value=representation();value['offsets']['s']=[];cases.append(value)
        for value in cases:self.assertIsNotNone(parse_representation(json.dumps(value))[1])
        self.assertIsNotNone(parse_representation('{"values":{},"values":{},"offsets":{}}')[1])
        with self.assertRaises(ValueError):execute(dict(values={'A':dict(base=1e308,offset='s'),'B':dict(base=1e308,offset='s')},offsets={'s':[1e308]}))

    def test_analytic_executor_matches_declared_domains(self):
        for shared in (True,False):
            rep=representation(a=-3,b=7,shared=shared);rep['offsets']['s']=[-5,0,20]
            calculated=execute(rep);names=list(rep['offsets']);pairs=[]
            for values in itertools.product(*(rep['offsets'][n] for n in names)):
                assignment=dict(zip(names,values));a=rep['values']['A'];b=rep['values']['B']
                pairs.append((a['base']+assignment[a['offset']],b['base']+assignment[b['offset']]))
            for q,values in [('a',[a for a,b in pairs]),('b',[b for a,b in pairs]),('difference',[a-b for a,b in pairs])]:
                self.assertEqual(calculated[q],(min(values),max(values)))

    def fixture(self,root):
        source=root/'source';components=root/'components';prepare(source,sources=2);prepare_components(source,components)
        comp,_=dataset(source,components);runs=root/'runs';runs.mkdir()
        for i,e in enumerate(comp['examples']):
            ref=json.loads((components/e['reference']).read_text());folder=runs/e['id'];folder.mkdir()
            meta=dict(experiment='structured_offsets_v1',status='finished',example_id=e['id'],tasks=list(TASKS),source_sha256=sha(source/'manifest.json'),
                components_sha256=sha(components/'manifest.json'),prompts=prompts(ref),generation=GENERATION,options=dict(seed=4646,**{'map-index':i}),
                model='fixture',resolved_revision='fixed',packages={'fixture':'1'},code_sha256={'fixture':'fixed'})
            meta['protocol_sha256']=fingerprint(meta);(folder/'run.json').write_text(json.dumps(meta))
            rows=[]
            for task in TASKS:
                c,mode=next((c,task[len(c)+1:]) for c in CONDITIONS if task.startswith(c+'_'))
                rep=representation(ref['displayed_a'],ref['displayed_b'],c=='shared_offset')
                if c=='clear':
                    truth=ref['interval_references'][c]['ground_truth']
                    rep['offsets']={'s':[truth['true_a']-ref['displayed_a']],'t':[truth['true_b']-ref['displayed_b']]}
                rows.append(dict(task=task,seed=4646+i*len(TASKS)+TASKS.index(task),input_modality='image' if mode=='image' else 'text',
                    raw_response=json.dumps(rep),hit_token_limit=False))
            (folder/'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return source,components,runs,comp

    def test_complete_oracle_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,components,runs,_=self.fixture(root);rows=evaluate(source,components,runs,root/'analysis')
            self.assertEqual(len(rows),18);self.assertTrue(all(r['exact_count']==2 and r['mean_endpoint_mae_valid_only']==0 for r in rows))
            meta=json.loads((root/'analysis/evaluation.json').read_text());self.assertEqual(meta['valid'],12)

    def test_direct_comparison_matches_exact_scope(self):
        import csv
        from interval_experiment import bounds
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,components,runs,comp=self.fixture(root)
            image=[];text=[]
            for e in comp['examples']:
                ref=json.loads((components/e['reference']).read_text())
                for c in CONDITIONS:
                    for q in ('a','b','difference'):
                        lo,hi=bounds(ref['interval_references'][c],q)
                        image.append(dict(example_id=ref['source_examples'][c],query=q,reference_lower=lo,reference_upper=hi,valid=True,endpoint_mae=2))
                        text.append(dict(source_group=e['source_group'],condition=c,query=q,reference_lower=lo,reference_upper=hi,valid=True,endpoint_mae=1))
            for name,rows in [('image.csv',image),('text.csv',text)]:
                with (root/name).open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            summary=evaluate(source,components,runs,root/'analysis',root/'image.csv',root/'text.csv')
            self.assertTrue(all(r['matched_count']==2 for r in summary))
            self.assertTrue(all(r['matched_mean_delta']==(-2 if r['mode']=='image' else -1) for r in summary))
            with (root/'image.csv').open('a') as stream:stream.write('unexpected,a,0,0,True,0\n')
            with self.assertRaisesRegex(ValueError,'membership'):evaluate(source,components,runs,root/'bad',root/'image.csv')

    def test_protocol_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,components,runs,comp=self.fixture(root)
            path=runs/comp['examples'][0]['id']/'run.json';meta=json.loads(path.read_text())
            meta['prompts']['clear_image']='changed';path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'Prompt'):evaluate(source,components,runs,root/'bad')

    def test_missing_source_and_bad_sharing_remain_visible(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,components,runs,comp=self.fixture(root);shutil.rmtree(runs/comp['examples'][0]['id'])
            rows=evaluate(source,components,runs,root/'analysis');self.assertTrue(all(r['expected']==2 and r['exact_count']==1 for r in rows))
            ref=json.loads((components/comp['examples'][1]['reference']).read_text())
            rep=representation(ref['displayed_a'],ref['displayed_b'],shared=False)
            diag,answers=score_representation(dict(raw_response=json.dumps(rep)),ref,'shared_offset')
            self.assertFalse(diag['sharing_correct']);self.assertTrue(diag['domains_correct']);self.assertEqual(diag['base_mae'],0)
            self.assertFalse(next(r for r in answers if r['query']=='difference')['exact_bounds'])

if __name__=='__main__':unittest.main()
