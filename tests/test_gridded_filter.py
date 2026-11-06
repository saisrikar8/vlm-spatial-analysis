import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import get_ilamb_data as ilamb
from netCDF4 import Dataset


def make_nc(path, kind):
    with Dataset(path, 'w') as nc:
        g = nc.createGroup('MeanState')
        for dim in ('y','x','site','time'):
            g.createDimension(dim, 3)
        dims = {'station':('site',), 'time_site':('time','site')}.get(kind, ('y','x'))
        g.createVariable('timeint_of_test', 'f8', dims)[:] = 1
        if kind in ('station','time_site'):
            latdims = londims = ('site',)
        elif kind == 'curvilinear':
            latdims = londims = ('y','x')
        else:
            latdims, londims = ('y',), ('x',)
        g.createVariable('lat', 'f8', latdims)[:] = 1
        g.createVariable('lon', 'f8', londims)[:] = 2


class GridFilterTests(unittest.TestCase):
    def test_grid_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ('station','time_site','rectilinear','curvilinear'):
                path=Path(tmp)/(kind+'.nc'); make_nc(path,kind)
                self.assertEqual(ilamb.inspect_grid(path,'timeint')['eligible'],
                                 kind in ('rectilinear','curvilinear'))
            with self.assertRaisesRegex(ValueError,'Expected one'):
                ilamb.inspect_grid(path,'bias')

    def test_bounds_are_not_map_variables(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'grid.nc'
            make_nc(path, 'rectilinear')
            with Dataset(path, 'a') as nc:
                group = nc.groups['MeanState']
                group.createDimension('nv', 2)
                for name in ('timeint_of_test_bnds', 'timeint_of_test_bounds', 'timeint_of_envelope'):
                    group.createVariable(name, 'f8', ('y','x','nv'))[:] = 99
                group.variables['timeint_of_test'].bounds = 'timeint_of_envelope'
            self.assertTrue(ilamb.inspect_grid(path, 'timeint')['eligible'])
            Image.new('RGB', (10,10)).save(root/'map.png')
            Image.new('RGB', (10,2)).save(root/'legend.png')
            (root/'manifest.json').write_text(json.dumps(dict(figures=[dict(
                id='test', region='global', plot='timeint', dataset='test', model='test',
                image_path='map.png', legend_path='legend.png', netcdf_path='grid.nc') ])))
            ilamb.align_arrays(root)
            record = json.loads((root/'alignment.json').read_text())['records'][0]
            self.assertEqual(record['variable'], 'MeanState/timeint_of_test')
            np.testing.assert_equal(np.load(root/'numerical.npy')[0], np.ones(9))
            with Dataset(path, 'a') as nc:
                nc.groups['MeanState'].createVariable('timeint_of_other', 'f8', ('y','x'))
            with self.assertRaisesRegex(ValueError, 'Expected one'):
                ilamb.inspect_grid(path, 'timeint')
            with self.assertRaisesRegex(ValueError, 'Expected one'):
                ilamb.align_arrays(root)

    def test_count_applies_after_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);catalog=root/'catalog.json';output=root/'output'
            common=dict(category='c',subcategory='s',dataset='d',model='m',region='global',plot='timeint',
                        page_url='https://example.org/page',legend_url='https://example.org/legend.png')
            catalog.write_text(json.dumps(dict(base_url='https://example.org/',datasets=[common],
                figures=[dict(common,image_url=f'https://example.org/{i}.png') for i in range(3)])))
            def download(client, record, out, pages):
                record['netcdf_path']=record['id']+'.nc'
                make_nc(out/record['netcdf_path'],'station' if record['image_url'].endswith('0.png') else 'rectilinear')
            with patch.object(ilamb,'download_netcdf',side_effect=download), \
                 patch.object(ilamb.Client,'png',return_value='hash') as png, \
                 contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                status=ilamb.main(['--catalog',str(catalog),'--output',str(output),'--count','2','--gridded-only'])
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(status,0)
            self.assertEqual(len(manifest['figures']),2)
            self.assertEqual(len(manifest['exclusions']),1)
            self.assertEqual(png.call_count,4)
            self.assertTrue(all(r['grid']['eligible'] for r in manifest['figures']))


if __name__=='__main__':
    unittest.main()
