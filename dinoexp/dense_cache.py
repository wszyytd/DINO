"""Frozen full-image/tile DINOv2 features with streaming and SHA provenance."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .core import checked_path
from .dense_data import read_dense_manifest
from .dense_views import make_views, prepare_reference
from .local_features import resolve_device
from .local_pairs import sha256
from .local_paths import resolve_backbone_paths


def save_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)


class DenseFeatureCache:
    def __init__(self,root):
        self.root=Path(root)
        meta=json.loads((self.root/'cache.json').read_text(encoding='utf-8'))
        if (meta.get('status')!='complete' or meta.get('kind')!='dense_reference_features'
                or meta.get('format_version')!=1):
            raise ValueError('need complete dense feature cache; crop caches are incompatible')
        self.config,self.images=meta['config'],meta['images']
        self.manifest_sha256=meta['manifest_sha256'];self.memory=OrderedDict()

    def get(self,key):
        if key in self.memory:
            value=self.memory.pop(key);self.memory[key]=value;return value
        if key not in self.images:
            raise ValueError('image not found in dense cache')
        entry=self.images[key];path=checked_path(self.root,entry['file'])
        if sha256(path)!=entry['sha256']:
            raise ValueError('dense feature hash mismatch')
        with np.load(path,allow_pickle=False) as archive:
            value={k:archive[k].copy() for k in archive.files}
        side=self.config['image_size']//14
        shape=(len(self.config['layers']),side**2,self.config['input_dim'])
        image=key.startswith('image:');count=1+self.config['rows']*self.config['cols']
        if value['tokens'].shape!=((count,*shape) if image else shape):
            raise ValueError('dense cached token shape mismatch')
        if value['valid'].shape!=((count,side**2) if image else (side**2,)):
            raise ValueError('dense validity shape mismatch')
        if not value['valid'].any(-1).all() or not np.isfinite(value['tokens']).all():
            raise ValueError('invalid dense cached content')
        if image:
            expected={'geometry':(count,6),'indices':(count,side**2,4),
                      'weights':(count,side**2,4),'priors':(count,),'view_weights':(count,)}
            for field,want in expected.items():
                if value[field].shape!=want or not np.isfinite(value[field]).all():
                    raise ValueError('invalid dense geometry '+field)
            if (value['indices'].min()<0 or value['indices'].max()>=self.config['grid_size']**2
                    or (value['weights']<0).any() or (value['priors']<=0).any()
                    or (value['view_weights']<=0).any()):
                raise ValueError('dense mapping outside source grid')
        self.memory[key]=value
        while len(self.memory)>2:
            self.memory.popitem(last=False)
        return value


def extract_dense_features(manifest_path,output,*,data_root=None,model_repo=None,checkpoint=None,
                           model='dinov2_vitb14',image_size=518,grid_size=32,tile_size=512,
                           batch_size=1,device='auto',dtype='float32'):
    import torch
    manifest=read_dense_manifest(manifest_path,data_root)
    if dtype not in ('float32','float16') or batch_size<1:
        raise ValueError('valid cache dtype and positive batch size required')
    # Geometry/path/weight failures occur before output creation or backbone allocation.
    make_views(Image.new('RGB',(2,2)),image_size,tile_size,grid_size=grid_size)
    repo,weight=resolve_backbone_paths(model_repo,checkpoint,model,Path(__file__).resolve().parents[1])
    cfg=dict(model=model,image_size=image_size,layers=[3,6,9,12],input_dim=(768 if model=='dinov2_vitb14' else 384),
             grid_size=grid_size,tile_size=tile_size,rows=3,cols=5,backbone_sha256=sha256(weight),dtype=dtype,
             preprocessing='native_letterbox_imagenet_v1',geometry_version=1)
    output=Path(output)
    marker=output/'cache.json'
    if output.exists():
        meta=json.loads(marker.read_text(encoding='utf-8'))
        if meta.get('config')!=cfg or meta.get('manifest_sha256')!=manifest['_manifest_sha256']:
            raise ValueError('existing dense cache belongs to different data/backbone/geometry')
    else:
        output.mkdir(parents=True)
        meta=dict(status='incomplete',format_version=1,kind='dense_reference_features',config=cfg,
                  manifest_sha256=manifest['_manifest_sha256'],images={})
        save_json(marker,meta)
    selected={}
    for sample in manifest['samples']:
        for role in ('reference','image'):
            selected[role+':'+sample[role+'_sha256']]=sample[role]
    pending=[]
    for key,source in selected.items():
        if key in meta['images']:
            entry=meta['images'][key]
            if sha256(checked_path(output,entry['file']))!=entry['sha256']:
                raise ValueError('existing cached feature hash mismatch; preserve and inspect cache')
        else:
            pending.append((key,source))
    if not pending:
        meta['status']='complete';save_json(marker,meta);return meta
    dev=resolve_device(device,torch)
    backbone=torch.hub.load(str(repo),model,source='local',pretrained=False)
    backbone.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    backbone.requires_grad_(False).to(dev).eval()
    for key,source in pending:
        with Image.open(checked_path(manifest['_root'],source)) as im:
            if key.startswith('reference:'):
                prepared,valid=prepare_reference(im,image_size)
                pictures=[prepared];extra=dict(valid=valid)
            else:
                view=make_views(im,image_size,tile_size,grid_size=grid_size)
                pictures=view.pop('images');view.pop('grid_size');extra=view
        features=[]
        with torch.inference_mode():
            for start in range(0,len(pictures),batch_size):
                a=np.stack([np.asarray(p,np.float32)/255 for p in pictures[start:start+batch_size]])
                a=(a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
                x=torch.from_numpy(a.transpose(0,3,1,2).copy()).to(dev)
                layers=backbone.get_intermediate_layers(x,n=[2,5,8,11],norm=True)
                features.append(torch.stack(layers,1).cpu().numpy().astype(dtype))
        tokens=np.concatenate(features)
        if key.startswith('reference:'):
            tokens=tokens[0]
        filename=hashlib.sha256(key.encode()).hexdigest()+'.npz'
        temporary=output/(filename+'.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,tokens=tokens,**extra)
        temporary.replace(output/filename)
        meta['images'][key]=dict(file=filename,source=source,sha256=sha256(output/filename))
        save_json(marker,meta)
        print(f'Dense frozen features: {len(meta["images"])}/{len(selected)} sources',flush=True)
    meta['status']='complete';save_json(marker,meta)
    return meta
