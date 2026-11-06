import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_random_pairs import select_pairs
from contract_fixture import bind_fixture
from evaluate_random_pairs import evaluate

class RandomPairsTests(unittest.TestCase):
    def test_selection(self):
        valid=np.arange(20); pixels=np.column_stack((valid*30,valid*0))
        a=select_pairs(valid,pixels,10,np.random.default_rng(42),24)
        b=select_pairs(valid,pixels,10,np.random.default_rng(42),24)
        self.assertEqual(a,b)
        self.assertEqual(len({tuple(sorted(p)) for p in a}),10)
        self.assertTrue(all(i!=j for i,j in a))
        with self.assertRaises(ValueError):select_pairs(valid[:1],pixels[:1],10,np.random.default_rng(42),24)

    def test_metrics_and_hidden_ordering_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'manifest.json').write_text(json.dumps(dict(relationship_direction='A_relative_to_B')))
            (root/'ground_truth.csv').write_text('example_id,map_id,true_difference_a_minus_b,true_ratio_a_over_b,true_a_greater_b\na,m,2,2,True\nb,m,-2,0.5,False\n')
            # Every ordering is wrong, but the true/false proportions agree.
            (root/'pred.csv').write_text('example_id,experiment,trial,predicted_difference_a_minus_b,predicted_ratio_a_over_b,predicted_a_greater_b\na,non-verbose,0,3,3,false\nb,non-verbose,0,-4,1.5,true\n')
            bind_fixture(root,root/'pred.csv',repeats=1,mode='non-verbose')
            evaluate(root,root/'pred.csv',root/'scored')
            rows=list(csv.DictReader(io.StringIO((root/'scored/per_map_metrics.csv').read_text())))
            self.assertEqual(float(rows[0]['mean_absolute_error']),1.5)
            self.assertEqual(float(rows[1]['mean_absolute_error']),1.)
            self.assertEqual(rows[2]['binomial_p_value'],'')
            self.assertEqual(float(rows[2]['accuracy']),0.)

if __name__=='__main__':unittest.main()
