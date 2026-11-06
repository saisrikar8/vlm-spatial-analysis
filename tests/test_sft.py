import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_sft import prepare,validate
from sft_pilot import mask_targets,choose_targets,evaluate,HP,GENERATION
from experiment_contract import sha


class SFTTests(unittest.TestCase):
    def test_assistant_only_mask_and_no_truncation(self):
        self.assertEqual(mask_targets([1,2,3,4],[1,2]),[-100,-100,3,4])
        with self.assertRaises(ValueError):mask_targets([1,3,4],[1,2])
        with self.assertRaises(ValueError):mask_targets([1,2],[1,2])

    def test_lora_excludes_vision_and_requires_targets(self):
        names=['model.language_model.layers.0.self_attn.q_proj','model.language_model.layers.0.self_attn.v_proj',
               'model.visual.layers.0.self_attn.q_proj','model.lm_head']
        self.assertEqual(choose_targets(names),names[:2])
        with self.assertRaises(ValueError):choose_targets(names[2:])

    def test_verified_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';m=prepare(root,dict(train=2,dev=1,test=1));validate(root)
            self.assertEqual(len(m['entries']),12)
            fields={s:{e['field_sha256'] for e in m['entries'] if e['split']==s} for s in ('train','dev','test')}
            self.assertFalse(fields['train']&fields['dev']);self.assertFalse(fields['train']&fields['test']);self.assertFalse(fields['dev']&fields['test'])
            p=root/'manifest.json';m['entries'][0]['split']='test';p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'leakage'):validate(root)

    def test_oracle_baseline_adapter_scoring_and_leakage_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';m=prepare(data,dict(train=1,dev=1,test=1));run=root/'run';run.mkdir()
            expected=[e for e in m['entries'] if e['split']=='test'];train=[e['id'] for e in m['entries'] if e['split']=='train']
            meta=dict(dataset_sha256=sha(data/'manifest.json'),hyperparameters=HP,generation=GENERATION,smoke=False,evaluated_split='test',adapter_sha256={'fixture':'fixture'})
            (run/'run.json').write_text(json.dumps(meta));scope=dict(training_ids=train,evaluation_ids=[e['id'] for e in expected]);(run/'training_scope.json').write_text(json.dumps(scope))
            records=[dict(example_id=e['id'],backend=b,raw_response=json.dumps(e['target']),hit_token_limit=False) for e in expected for b in ('base','adapted')]
            (run/'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            rows=evaluate(data,run,root/'analysis');self.assertTrue(all(r['exact']==1 and r['endpoint_mae_valid_only']==0 for r in rows))
            scope['training_ids'].append(expected[0]['id']);(run/'training_scope.json').write_text(json.dumps(scope))
            with self.assertRaisesRegex(ValueError,'leakage'):evaluate(data,run,root/'invalid')

if __name__=='__main__':unittest.main()
