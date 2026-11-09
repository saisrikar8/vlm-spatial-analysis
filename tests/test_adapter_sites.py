from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from adapter_site_sft import choose_targets,mask_targets


class AdapterSiteTests(unittest.TestCase):
    def setUp(self):
        self.lang=['model.language_model.layers.0.self_attn.q_proj','model.language_model.layers.0.self_attn.v_proj']
        self.vision=['model.visual.blocks.0.attn.qkv','model.visual.blocks.0.attn.proj']
        self.names=self.lang+self.vision+['model.visual.merger.mlp.0','model.visual.patch_embed.proj','model.language_model.layers.0.self_attn.k_proj']
    def test_language_only(self):self.assertEqual(choose_targets(self.names,'language'),self.lang)
    def test_vision_only(self):self.assertEqual(choose_targets(self.names,'vision'),self.vision)
    def test_joint_is_disjoint_union(self):self.assertEqual(choose_targets(self.names,'joint'),self.lang+self.vision)
    def test_missing_vision_fails(self):
        with self.assertRaises(ValueError):choose_targets(self.lang,'vision')
    def test_invalid_site_fails(self):
        with self.assertRaises(ValueError):choose_targets(self.names,'all-weights')
    def test_mask_does_not_supervise_image_or_prompt(self):self.assertEqual(mask_targets([1,2,3,4],[1,2]),[-100,-100,3,4])


if __name__=='__main__':unittest.main()
