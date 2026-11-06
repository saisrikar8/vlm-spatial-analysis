import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plot_vlm import main,plot_map_metrics


def fixture(path):
    rows=[]
    for method in ('verbose','non-verbose'):
        for query in ('difference','ratio','ordering'):
            rows.append(dict(map_id='map-000',experiment=method,query=query,expected_pairs=10,scored_pairs=10,
                mean_absolute_error=(1.2 if method=='verbose' else 2.3) if query!='ordering' else '',
                accuracy=.9 if query=='ordering' else '',test_correct=9,test_pairs=10,
                binomial_p_value=.0107421875 if query=='ordering' else '',
                holm_p_value='',test_status='complete'))
    rows[1]['mean_absolute_error']='';rows[1]['scored_pairs']=0
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows


class PlotMetricsTests(unittest.TestCase):
    def test_cli_render_and_exact_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'per_map_metrics.csv';fixture(source)
            main(['--csv',str(source),'--output',str(root/'plots')])
            for name in ('map-000-metrics.png','per_map_metrics.pdf','plotted_metrics.csv'):
                self.assertGreater((root/'plots'/name).stat().st_size,0)
            self.assertEqual((root/'plots/plotted_metrics.csv').read_text(),source.read_text())

    def test_invalid_pvalue_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'metrics.csv';fixture(source)
            source.write_text(source.read_text().replace('0.0107421875','1.2'))
            with self.assertRaisesRegex(ValueError,'Invalid summary value'):
                plot_map_metrics(source,root/'plots')
            self.assertFalse((root/'plots').exists())

if __name__=='__main__':unittest.main()
