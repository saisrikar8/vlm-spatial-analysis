import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch,MagicMock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from contract_fixture import bind_fixture
from evaluate_random_pairs import binomial_tail,holm,evaluate
from run_random_pairs import main,options,prediction_rows,prompts

class ExperimentTests(unittest.TestCase):
    def test_math_and_direction(self):
        self.assertAlmostEqual(binomial_tail(9,10,.5),11/1024)
        self.assertAlmostEqual(binomial_tail(10,10,.5),1/1024)
        self.assertEqual(holm([.01,.04,.03]),[.03,.06,.06])
        answers={'verbose':[{'A':6.,'B':2.},{'A':1.,'B':0.}]}
        rows=prediction_rows({'id':'x'},answers,2,1e-8)
        self.assertEqual(rows[0]['predicted_difference_a_minus_b'],4.)
        self.assertEqual(rows[0]['predicted_ratio_a_over_b'],3.)
        self.assertTrue(rows[0]['predicted_a_greater_b'])
        self.assertIsNone(rows[1]['predicted_ratio_a_over_b'])
        self.assertIn('A divided by B',prompts('clean-marked')['ratio'])
        self.assertNotIn('Image 1',prompts('marked-only')['verbose'])
        with self.assertRaises(ValueError):options(['--map-index','0','--output','x','--repeats','0'])

    def test_no_pseudoreplication_or_incomplete_pvalue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'manifest.json').write_text(json.dumps(dict(relationship_direction='A_relative_to_B')))
            (root/'ground_truth.csv').write_text('example_id,map_id,true_difference_a_minus_b,true_ratio_a_over_b,true_a_greater_b\na,m,1,2,True\nb,m,-1,0.5,False\n')
            header='example_id,experiment,trial,predicted_difference_a_minus_b,predicted_ratio_a_over_b,predicted_a_greater_b\n'
            (root/'p.csv').write_text(header+'a,verbose,0,1,2,true\na,verbose,1,1,2,true\n')
            bind_fixture(root,root/'p.csv')
            evaluate(root,root/'p.csv',root/'partial')
            r=next(r for r in csv.DictReader(io.StringIO((root/'partial/per_map_metrics.csv').read_text())) if r['experiment']=='verbose' and r['query']=='ordering')
            self.assertEqual(r['binomial_p_value'],'')
            with (root/'p.csv').open('a') as f:f.write('b,verbose,0,-1,0.5,\nb,verbose,1,-1,0.5,false\n')
            evaluate(root,root/'p.csv',root/'full')
            r=next(r for r in csv.DictReader(io.StringIO((root/'full/per_map_metrics.csv').read_text())) if r['experiment']=='verbose' and r['query']=='ordering')
            self.assertEqual(r['test_pairs'],'2');self.assertEqual(r['test_correct'],'1')
            self.assertEqual(r['binomial_p_value'],'')
            self.assertNotIn('significant',r)

    def test_mocked_two_image_runner(self):
        processor=MagicMock();processor.apply_chat_template.return_value='rendered'
        processor.return_value.to.return_value={}
        model=MagicMock();model.device='cuda';model.generation_config.to_dict.return_value={}
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True,is_bf16_supported=lambda:True,
            get_device_name=lambda i:'fake',max_memory_allocated=lambda:0),version=types.SimpleNamespace(cuda='fake'),bfloat16='bf16')
        transformers=types.SimpleNamespace(AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a,**k:processor),
            Qwen2_5_VLForConditionalGeneration=types.SimpleNamespace(from_pretrained=lambda *a,**k:types.SimpleNamespace(eval=lambda:model)),set_seed=lambda n:None)
        hub=types.SimpleNamespace(model_info=lambda *a,**k:types.SimpleNamespace(sha='pinned'))
        def fake_collect(model,processor,inputs,args,output,torch,set_seed,filename,parser,**kw):
            query=filename.split('.')[0]
            answer={'A':6,'B':2} if query=='verbose' else {'answer':{'difference':4,'ratio':3,'ordering':True}[query]}
            with (output/filename).open('w') as f:
                for i in range(kw.get('start_index',0),args['samples']):f.write(json.dumps(dict(sample=i,seed=args['seed']+i,raw_response=json.dumps(answer),complete=True,hit_token_limit=False))+'\n')
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'torch':torch,'transformers':transformers,'huggingface_hub':hub}), \
             patch('run_random_pairs.collect',side_effect=fake_collect),patch('run_random_pairs.importlib.metadata.version',return_value='fake'),contextlib.redirect_stdout(io.StringIO()):
            from prepare_controlled import prepare
            dataset=Path(tmp)/'data';prepare(dataset,sources=1)
            output=Path(tmp)/'run'
            main(['--map-index','0','--dataset',str(dataset),'--output',str(output),'--repeats','1'])
            directory=next(output.iterdir())
            with (directory/'predictions.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2)
            self.assertEqual({r['experiment'] for r in rows},{'verbose','non-verbose'})
            self.assertEqual(len(processor.call_args.kwargs['images']),2)
            self.assertEqual(len(processor.apply_chat_template.call_args.args[0]),1)
            self.assertTrue((directory/'evaluation/per_map_metrics.csv').exists())
            self.assertEqual(json.loads((directory/'run.json').read_text())['status'],'finished')
            with patch('run_random_pairs.collect',side_effect=AssertionError('Resume regenerated existing samples')):
                main(['--map-index','0','--dataset',str(dataset),'--output',str(output),'--repeats','1','--resume'])
            self.assertTrue((directory/'evaluation-resume-1/per_map_metrics.csv').exists())

if __name__=='__main__':unittest.main()
