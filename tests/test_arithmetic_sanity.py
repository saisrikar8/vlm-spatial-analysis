import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from arithmetic_sanity import CASES,VERSIONS,GENERATION,REVISION,MODELS,plan,prompts,score,evaluate
from experiment_contract import sha


class ArithmeticSanityTests(unittest.TestCase):
    def fixture(self,root,omit=None):
        run=root/'run';run.mkdir();(run/'plan.json').write_text(json.dumps(plan()))
        meta=dict(status='finished' if omit is None else 'failed',generation=GENERATION,model=MODELS['qwen7b'],resolved_revision=REVISION,
            plan_sha256=sha(run/'plan.json'))
        (run/'run.json').write_text(json.dumps(meta));records=[]
        for i,(a,b) in enumerate(CASES):
            for v in VERSIONS:
                if omit==(i,v):continue
                raw=str(a-b) if v=='plain' else json.dumps(dict(lower=a-b,upper=a-b))
                records.append(dict(case_index=i,version=v,prompt=prompts(a,b)[v],raw_response=raw,hit_token_limit=False,
                    input_ids=[i,VERSIONS.index(v)],generated_ids=[a-b]))
        repeat=dict(records[0],control=True);records.append(repeat)
        (run/'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        return run

    def test_frozen_case_scope(self):
        self.assertEqual(len(CASES),20);self.assertEqual(len(set(CASES)),20)
        self.assertTrue(any(a-b<0 for a,b in CASES));self.assertTrue(any(a-b==0 for a,b in CASES))
        self.assertEqual(plan()['scored_calls'],60);self.assertEqual(plan()['extra_control_calls'],1)

    def test_scalar_and_interval_errors(self):
        plain=score(dict(case_index=0,version='plain',raw_response='3'))
        interval=score(dict(case_index=0,version='interval',raw_response='{"lower":1,"upper":3}'))
        self.assertEqual(plain['numerical_error'],1);self.assertEqual(interval['numerical_error'],1)
        self.assertFalse(interval['exact'])
        self.assertFalse(score(dict(case_index=0,version='plain',raw_response='2',hit_token_limit=True))['valid'])

    def test_oracle_and_repeat_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=self.fixture(root);rows=evaluate(run,root/'evaluation')
            self.assertTrue(all(r['correct']==20 and r['mean_numerical_error_valid_only']==0 for r in rows))
            report=json.loads((root/'evaluation/evaluation.json').read_text())
            self.assertTrue(report['repeat_control']['identical_input_ids']);self.assertTrue(report['repeat_control']['identical_generated_ids'])

    def test_missing_answer_preserves_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=self.fixture(root,(19,'offset'));rows=evaluate(run,root/'evaluation')
            row=next(r for r in rows if r['version']=='offset')
            self.assertEqual(row['expected'],20);self.assertEqual(row['attempted'],19);self.assertEqual(row['accuracy'],.95)

    def test_prompt_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=self.fixture(root);p=run/'responses.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
            rows[0]['prompt']='changed';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'Prompt'):evaluate(run,root/'evaluation')

if __name__=='__main__':unittest.main()
