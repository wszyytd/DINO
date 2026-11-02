from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from dinoexp.dense_collection import make_capture_plan, export_manifest,collect_dense
from dinoexp.local_pairs import sha256


class DenseCollectionTests(unittest.TestCase):
    def test_pilot_covers_distinct_asset_pairs_and_never_mutates_source_configs(self):
        base=json.loads(Path('configs/two_candidates.json').read_text())
        cases=[]
        for split,pairs in [('train',[('a','b'),('b','c'),('c','a')]),('val',[('d','e')]),('test',[('f','g')])]:
            for j,pair in enumerate(pairs):
                for location in range(2):
                    for ref in ('A','B'):
                        cfg=deepcopy(base);cfg['target_identity']=ref;cfg['anchor_spawn_index']=j+location*10+{'train':0,'val':100,'test':200}[split]
                        for c,m in zip(cfg['candidates'],pair,strict=True):c['blueprint']='vehicle.'+m
                        cfg['views']=[dict(name='overview_raised' if split=='test' else 'overview',azimuth=90,elevation=80,distance=70)]
                        group=f'{split}_{j}_{location}'
                        cases.append(dict(name=group+'_ref'+ref,split=split,capture_group=group,config=cfg))
        source=dict(status='complete',plan=dict(cases=cases));before=deepcopy(source)
        result=make_capture_plan(source,['overview'],max_train_groups=3)
        self.assertEqual(len([c for c in result if c['split']=='train']),6)
        pairs={tuple(sorted(c['blueprint'] for c in item['config']['candidates']))
               for item in result if item['split']=='train'}
        self.assertEqual(len(pairs),3)
        self.assertEqual(source,before)
        self.assertTrue(all(len(c['config']['views'])==1 for c in result))

    def test_incomplete_source_and_unverified_child_cannot_be_exported(self):
        with self.assertRaisesRegex(ValueError,'complete'):
            make_capture_plan(dict(status='incomplete'),['overview'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'train'/'case').mkdir(parents=True)
            (root/'train/case/run.json').write_text(json.dumps(dict(status='incomplete')))
            with self.assertRaisesRegex(ValueError,'complete|capture'):
                export_manifest(root,[dict(name='case',split='train')])

    def test_full_collection_exports_pixels_and_resume_never_recaptures_verified_children(self):
        base=json.loads(Path('configs/two_candidates.json').read_text());cases=[]
        for split,i in [('train',0),('val',1),('test',2)]:
            for target in ('A','B'):
                cfg=deepcopy(base);cfg.update(target_identity=target,anchor_spawn_index=i)
                cfg['views']=[dict(name='overview',azimuth=90,elevation=80,distance=70)]
                for j,c in enumerate(cfg['candidates']):c['blueprint']=f'vehicle.asset{i*2+j}'
                cases.append(dict(name=split+'_ref'+target,split=split,capture_group=split,config=cfg))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            (source/'collection.json').write_text(json.dumps(dict(status='complete',plan=dict(cases=cases))))
            output=root/'dense';calls=[]
            def renderer(command,check):
                cfg=json.loads(Path(command[command.index('--config')+1]).read_text())
                dest=Path(command[command.index('--output')+1]);dest.mkdir(parents=True);calls.append(cfg['dense_case'])
                color=len(calls)*25
                Image.new('RGB',(28,28),(color,5,10)).save(dest/'reference.png')
                mask=np.zeros((28,28),np.uint8);mask[:14,:14]=255;Image.fromarray(mask).save(dest/'mask.png')
                rows=[]
                for absent in (False,True):
                    name='null.png' if absent else 'image.png'
                    Image.new('RGB',(28,28),(color,30+absent,50)).save(dest/name)
                    rows.append(dict(id=cfg['dense_case']+str(absent),split=cfg['dense_split'],group=cfg['dense_group'],
                        reference='reference.png',reference_sha256=sha256(dest/'reference.png'),image=name,image_sha256=sha256(dest/name),
                        positive_masks=[] if absent else ['mask.png'],negative_masks=[],mask_sha256={'mask.png':sha256(dest/'mask.png')},
                        presence='absent' if absent else 'present',supervision='simulation',mask_source='carla_instance_verified'))
                (dest/'run.json').write_text(json.dumps(dict(status='complete',config=cfg,cleanup_errors=[],
                    instance_id_method='remove_restore_probe',associations=[dict(identity='A'),dict(identity='B')],samples=rows)))
            first=collect_dense(source,output,['overview'],runner=renderer)
            self.assertEqual(len(calls),6)
            manifest=json.loads((output/'dense_manifest.json').read_text())
            self.assertEqual(len(manifest['samples']),12)
            before=(output/'train/train_refA/run.json').read_bytes()
            resumed=collect_dense(None,output,resume=True,runner=renderer)
            self.assertEqual(resumed['remaining'],0)
            self.assertEqual(len(calls),6)
            self.assertEqual(before,(output/'train/train_refA/run.json').read_bytes())
