import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dinoexp.evidence_query_metrics import summarize_updates, describe_scores


def pair(full,skip,first=False):
    return dict(kind='update',eligible=not first,first_write=first,full_correct=full,skip_correct=skip,
        margin_delta=.02,null_probability_delta=0.,vehicle_uid='A',view_group='side',quality='clean',
        variant='E_memory_repeat',seed=7)


class QueryMetricsTests(unittest.TestCase):
    def test_pair_rates(self):
        s=summarize_updates([pair(True,False),pair(True,False),pair(False,True),pair(True,True)])['overall']
        for key,value in dict(help_rate=.5,harm_rate=.25,net_gain=.25,conditional_harm=.5).items():
            self.assertEqual(s[key]['value'],value)
        self.assertEqual(s['conditional_harm']['denominator'],2)

    def test_first_write_excluded(self):
        s=summarize_updates([pair(True,False,True),pair(False,True)])['overall']
        self.assertEqual(s['help_rate']['value'],0)
        self.assertEqual(s['harm_rate']['value'],1)
        self.assertEqual(s['first_write_pairs'],1)

    def test_ranking_vs_null(self):
        d=describe_scores([2.,1.,3.],['A','B'],['A','B'],'A')
        self.assertFalse(d['correct']); self.assertTrue(d['ranking_correct'])
        self.assertTrue(d['false_reject'])
        row=dict(kind='query',query_group=1,**d)
        s=summarize_updates([row])['overall']
        self.assertEqual(s['query_accuracy']['value'],0)
        self.assertEqual(s['ranking_accuracy']['value'],1)
        self.assertEqual(s['null_false_reject_rate']['value'],1)
        self.assertIsNone(s['null_false_accept_rate']['value'])

    def test_no_competitor_margin(self):
        d=describe_scores([2.,-1e4,0.],['A','B'],['A'],'A')
        self.assertIsNone(d['margin']); self.assertIsNone(d['ranking_correct'])

    def test_empty_and_repeat_denominators(self):
        s=summarize_updates([])['overall']
        self.assertIsNone(s['help_rate']['value'])
        rows=[dict(kind='repeat',full_choice='A',repeat_choice='B',full_correct=True,repetitions=1),
              dict(kind='repeat',full_choice='B',repeat_choice='B',full_correct=False,repetitions=1)]
        s=summarize_updates(rows)['overall']
        self.assertEqual(s['repeat_flip_rate']['value'],.5)
        self.assertEqual(s['repeat_flip_correct']['value'],1)

    def test_probability_is_not_identification_gain(self):
        row=pair(False,False); row['margin_delta']=.5
        s=summarize_updates([row])['overall']
        self.assertEqual(s['net_gain']['value'],0)
        self.assertEqual(s['margin_delta']['value'],.5)

    def test_quality_is_separated_by_method(self):
        a=pair(True,False); a['variant']='A_last'; a['quality']='blur'
        b=pair(False,True); b['quality']='blur'
        s=summarize_updates([a,b])['by_method_seed_strata']
        self.assertEqual(s['A_last']['7']['quality']['blur']['net_gain']['value'],1)
        self.assertEqual(s['E_memory_repeat']['7']['quality']['blur']['net_gain']['value'],-1)

    def test_report_from_saved_rows(self):
        from dinoexp.region_ab_data import write_json
        from tests.test_evidence_query_data import fixture
        from dinoexp.evidence_query_data import build_manifest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data,source,cfg=fixture(root); m=build_manifest(source,data,cfg)
            out=root/'output'; out.mkdir(); write_json(out/'manifest.json',m)
            ev=out/'evaluation'/'smoke'/'probe'; ev.mkdir(parents=True)
            ep=next(e for e in m['episodes'] if e['split']=='probe')
            q=ep['steps'][0]['queries'][0]
            row=dict(kind='query',episode_id=ep['id'],step=0,query_id=q['sample_id'],variant='A_last',seed=7,
                query_group=0,**describe_scores([2.,1.,0.],ep['identities'],ep['identities'],q['label']))
            write_json(ev/'records.json',dict(status='complete',synthetic=True,rows=[row,pair(True,False)]))
            cmd=[sys.executable,'run_evidence_query.py','report','--output',str(out)]
            r=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertTrue((out/'report'/'index.html').exists())
            self.assertTrue((out/'report'/'report.md').exists())
            self.assertTrue(list((out/'report'/'assets').glob('*.jpg')))
            code='import sys; import dinoexp.evidence_query_report; assert "torch" not in sys.modules'
            self.assertEqual(subprocess.run([sys.executable,'-c',code]).returncode,0)


if __name__=='__main__': unittest.main()
