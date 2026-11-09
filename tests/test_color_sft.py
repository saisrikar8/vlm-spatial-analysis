import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from color_sft_pilot import mask_targets,choose_targets,evaluate


class ColorSFTTests(unittest.TestCase):
    def test_assistant_only_labels_and_boundary(self):
        self.assertEqual(mask_targets([10,11,12,13],[10,11]),[-100,-100,12,13])
        with self.assertRaises(ValueError):mask_targets([10,11],[10,12])
        with self.assertRaises(ValueError):mask_targets([10,11],[10,11])

    def test_visual_weights_excluded(self):
        names=['model.language_model.layers.0.self_attn.q_proj','model.language_model.layers.0.self_attn.v_proj','model.visual.blocks.0.self_attn.q_proj','model.language_model.layers.0.self_attn.k_proj']
        self.assertEqual(choose_targets(names),names[:2])

    def test_evaluator_rejects_training_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'manifest.json').write_text('{}')
            import hashlib
            digest=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
            (root/'run.json').write_text(json.dumps(dict(dataset_sha256=digest,smoke=False)))
            (root/'training_scope.json').write_text(json.dumps(dict(training_ids=['heldout'],evaluation_ids=['heldout'],evaluated_split='test')))
            with patch('color_sft_pilot.validate',return_value=dict(entries=[dict(id='heldout',split='test')])):
                with self.assertRaisesRegex(ValueError,'Training split violation'):evaluate(root,root)


if __name__=='__main__':unittest.main()
