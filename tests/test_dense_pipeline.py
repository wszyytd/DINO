import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from PIL import Image
import torch
from dinoexp.dense_cache import DenseFeatureCache
from dinoexp.dense_training import train_dense, evaluate_dense
from dinoexp.local_pairs import sha256


def fixture(root):
    rng=np.random.default_rng(51)
    config=dict(model='fixture',image_size=28,layers=[3,6,9,12],input_dim=8,
                grid_size=2,tile_size=512,rows=3,cols=5,backbone_sha256='0'*64,dtype='float32')
    cache=root/'cache';cache.mkdir()
    images={};samples=[]
    for i,split in enumerate(('train','val','test')):
        for role in ('ref','image'):
            path=root/f'{split}_{role}.png';Image.new('RGB',(28,28),(30+i*30,80,40+(role=='image')*30)).save(path)
        mask=np.zeros((28,28),np.uint8);mask[:14,:14]=255
        Image.fromarray(mask).save(root/f'{split}_mask.png')
        row=dict(id=split,split=split,group=split,reference=f'{split}_ref.png',image=f'{split}_image.png',
                 reference_sha256=sha256(root/f'{split}_ref.png'),image_sha256=sha256(root/f'{split}_image.png'),
                 positive_masks=[f'{split}_mask.png'],negative_masks=[],presence='present',
                 supervision='simulation',mask_source='carla_instance_verified',
                 mask_sha256={f'{split}_mask.png':sha256(root/f'{split}_mask.png')})
        samples.append(row)
        for role in ('reference','image'):
            key=role+':'+row[role+'_sha256'];filename=hashlib.sha256(key.encode()).hexdigest()+'.npz'
            item=dict(tokens=rng.normal(size=(4,4,8)).astype(np.float32),valid=np.ones(4,bool))
            if role=='image':
                item.update(tokens=item['tokens'][None].repeat(16,axis=0),valid=np.ones((16,4),bool),
                    geometry=np.tile([0,0,1,1,1,1],(16,1)).astype(np.float32),
                    indices=np.arange(4).reshape(1,4,1).repeat(16,axis=0).repeat(4,axis=2),
                    weights=np.tile([1,0,0,0],(16,4,1)).astype(np.float32),priors=np.ones(16,np.float32),
                    view_weights=np.full(16,1/16,np.float32))
            np.savez_compressed(cache/filename,**item)
            images[key]=dict(file=filename,sha256=sha256(cache/filename))
    manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(format_version=1,task='reference_guided_dense_search',samples=samples)))
    (cache/'cache.json').write_text(json.dumps(dict(status='complete',format_version=1,
        kind='dense_reference_features',config=config,manifest_sha256=sha256(manifest),images=images)))
    return manifest,cache


class DensePipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_tiny_spatial_training_checkpoint_reload_and_original_image_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,cache=fixture(root)
            args=SimpleNamespace(manifest=manifest,data_root=root,cache=cache,output=root/'train',
                stage='simulation',checkpoint=None,device='cpu',epochs=3,lr=.002,batch_size=1,
                dim=16,heads=2,threads=2,seed=7,gamma=2.,null_weight=1.)
            run=train_dense(args)
            self.assertEqual(run['status'],'complete')
            checkpoint=torch.load(root/'train/best.pt',weights_only=True)
            self.assertEqual(checkpoint['kind'],'dense_reference_search_v1')
            result=evaluate_dense(SimpleNamespace(manifest=manifest,data_root=root,cache=cache,
                checkpoint=root/'train/best.pt',output=root/'report',split='test',device='cpu',threads=2))
            self.assertEqual(result['samples'],1)
            self.assertEqual(result['status'],'complete')
            self.assertTrue((root/'report/index.html').is_file())
            self.assertTrue((root/'report/assets/0_heatmap.png').is_file())

    def test_modified_cached_features_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest,root=fixture(Path(tmp))
            cache=DenseFeatureCache(root)
            key=next(iter(cache.images));path=root/cache.images[key]['file']
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash'):
                cache.get(key)
