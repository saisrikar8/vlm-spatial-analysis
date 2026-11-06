import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from filter_real_benchmark import inspect_point


class RealFilterTests(unittest.TestCase):
    def point(self):return dict(label='A',latitude=40.,longitude=10.,pixel=[5.,5.],value=3.)
    def image(self):return np.full((11,11,3),(30,100,200),dtype=np.uint8)
    def test_land_with_colored_support_kept(self):
        with patch('filter_real_benchmark.globe.is_land',return_value=True):self.assertTrue(inspect_point(self.image(),self.point())['keep'])
    def test_ocean_rejected_even_when_colored(self):
        with patch('filter_real_benchmark.globe.is_land',return_value=False):
            r=inspect_point(self.image(),self.point());self.assertFalse(r['keep']);self.assertIn('ocean_coordinate',r['exclusion_reasons'])
    def test_land_gray_or_off_image_rejected(self):
        with patch('filter_real_benchmark.globe.is_land',return_value=True):
            self.assertFalse(inspect_point(np.full((11,11,3),191,dtype=np.uint8),self.point())['keep'])
            p=self.point();p['pixel']=[100,100];self.assertFalse(inspect_point(self.image(),p)['keep'])
    def test_nonfinite_reference_rejected(self):
        p=self.point();p['value']=float('nan');self.assertFalse(inspect_point(self.image(),p)['keep'])

if __name__=='__main__':unittest.main()
