"""Auditable query episodes and feature IO. Deliberately has no torch dependency."""
import copy
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .core import checked_path
from .region_ab_data import digest_json, read_json, sha

DEFAULT_CONFIG = dict(seed=7, lengths=[4, 8], stride=4, group_mode='query_all',
    split_mode='smoke_shared_identity', current_queries=2, history_queries=1,
    perturbations=dict(clean=.70, blur=.15, cover=.15), repeat_probability=.25,
    history_weight=1., repeat_weight=.1, heads=4, slots=8, batch_size=4,
    lr=1e-4, weight_decay=.01, clip=1., seeds=[7,17,27], smoke_updates=200,
    full_updates=2300)


def rng_for(seed, *parts):
    return np.random.default_rng(int(digest_json([seed, *parts])[:16], 16))


def pixel_hash(image):
    return hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()


def render_sample(root, sample, perturbation):
    path = checked_path(Path(root), sample['source_rgb'])
    if sha(path) != sample['source_rgb_sha256']:
        raise ValueError('source image hash mismatch')
    with Image.open(path) as im:
        crop = im.convert('RGB').crop(sample['box'])
    if pixel_hash(crop) != sample['crop_sha256']:
        raise ValueError('source crop hash mismatch')
    kind = perturbation['kind']
    if kind == 'clean':
        return crop
    if kind == 'blur':
        original = crop.size
        crop.thumbnail((256,256), Image.Resampling.LANCZOS)
        return crop.filter(ImageFilter.GaussianBlur(perturbation['sigma'])).resize(original, Image.Resampling.BICUBIC)
    if kind == 'cover':
        a = np.array(crop)
        h,w = a.shape[:2]; fraction = float(perturbation['fraction'])
        if not 0 < fraction < 1:
            raise ValueError('invalid cover fraction')
        ch,cw = max(1,round(h*fraction**.5)), max(1,round(w*fraction**.5))
        gen=np.random.default_rng(perturbation['seed'])
        y,x=int(gen.integers(h-ch+1)),int(gen.integers(w-cw+1))
        if 'box_xywh' in perturbation and perturbation['box_xywh']!=[x,y,cw,ch]:
            raise ValueError('cover position does not match seed and crop')
        a[y:y+ch,x:x+cw]=127
        return Image.fromarray(a)
    raise ValueError('unknown RGB perturbation')


def context_digest(episode):
    e=copy.deepcopy(episode)
    e.pop('context_sha256',None)
    for s in e['steps']:
        s.pop('queries',None)
    return digest_json(e)


def _feature_key(role, sid, perturbation):
    return digest_json([role,sid,perturbation])


def build_manifest(source_dir, data_root, config):
    source_dir,data_root=Path(source_dir).resolve(),Path(data_root).resolve()
    cfg=copy.deepcopy(DEFAULT_CONFIG); cfg.update(config)
    if cfg['group_mode'] not in ('query_all','grouped'):
        raise ValueError('unknown group_mode')
    if cfg['split_mode'] not in ('smoke_shared_identity','identity_disjoint'):
        raise ValueError('unknown split_mode')
    if cfg['current_queries'] != 2 or cfg['history_queries'] != 1:
        raise ValueError('v1 protocol uses two current/general and one history query')
    if not cfg['lengths'] or min(cfg['lengths'])<1 or cfg['stride']<1:
        raise ValueError('invalid window length/stride')
    probabilities=[cfg['perturbations'][k] for k in ('clean','blur','cover')]
    if min(probabilities)<0 or not np.isclose(sum(probabilities),1):
        raise ValueError('perturbation probabilities must sum to one')
    plan=read_json(source_dir/'plan.json')
    records=read_json(source_dir/'observations.json'); routes=read_json(source_dir/'routes.json')
    if plan['status']!='complete':
        raise ValueError('source plan incomplete')
    for name in ('observations.json','routes.json'):
        if sha(source_dir/name)!=plan['artifacts'][name]:
            raise ValueError('source artifact hash mismatch: '+name)
    groups=read_json(Path(cfg['view_group_manifest'])) if cfg.get('view_group_manifest') else {}
    if cfg['group_mode']=='grouped' and not groups:
        raise ValueError('grouped requires external view group annotations')
    samples,lookup,record_map={}, {}, {r['id']:r for r in records}
    images={}; image_hashes={}
    for r in records:
        if r['image'] not in images:
            path=checked_path(data_root,r['image'])
            if sha(path)!=r['image_sha256']:
                raise ValueError('source image hash mismatch')
            with Image.open(path) as im:
                images[r['image']]=im.convert('RGB')
            image_hashes[r['image']]=r['image_sha256']
        if image_hashes[r['image']]!=r['image_sha256']:
            raise ValueError('conflicting source image hashes')
        for o in r['observations']:
            if not o.get('write_allowed',False):
                continue
            crop=images[r['image']].crop(o['box'])
            sid=digest_json([r['split'],r['scene'],r['image_sha256'],o['identity'],o['box']])
            annotation=groups.get(o['key'],{})
            if cfg['group_mode']=='grouped' and (not annotation.get('source') or annotation.get('group') is None):
                raise ValueError('missing trustworthy view group annotation: '+o['key'])
            sample=dict(source_rgb=r['image'],source_rgb_sha256=r['image_sha256'],
                parent_source_sha256=r.get('parent_source_sha256',r['image_sha256']),
                crop_sha256=pixel_hash(crop),box=o['box'],frame_id=r['frame'],
                scene_id=r['scene'],split=r['split'],vehicle_uid=o['identity'],
                view_group=annotation.get('group'),view_group_source=annotation.get('source'),
                source_observation_key=o['key'])
            if sid in samples and any(samples[sid][f]!=sample[f] for f in ('vehicle_uid','view_group','crop_sha256')):
                raise ValueError('ambiguous source observation metadata')
            samples[sid]=sample; lookup[(r['id'],o['key'])]=sid
    episodes,missing=[],[]
    for route in routes:
        for length in cfg['lengths']:
            for start in range(0,len(route['records'])-length+1,cfg['stride']):
                rid=f"{route['id']}/L{length}/s{start}"
                gen=rng_for(cfg['seed'],'context',rid); qgen=rng_for(cfg['seed'],'query',rid)
                identities=list(route['identities']); gen.shuffle(identities)
                chosen=[record_map[x] for x in route['records'][start:start+length]]
                source_ref=route['reference_sha256']
                if any(r['split']!=route['split'] or r['reference_sha256']!=source_ref for r in chosen):
                    raise ValueError('route split / fixed source reference mismatch')
                blocked={r['image_sha256'] for r in chosen}
                blocked.update(r.get('parent_source_sha256',r['image_sha256']) for r in chosen)
                context_samples=[lookup[(r['id'],o['key'])] for r in chosen for o in r['observations'] if o.get('write_allowed',False)]
                crop_hashes={samples[s]['crop_sha256'] for s in context_samples}
                pool=[sid for sid,s in samples.items() if s['split']==route['split'] and s['scene_id']==route['scene']
                    and s['source_rgb_sha256'] not in blocked and s['parent_source_sha256'] not in blocked
                    and s['crop_sha256'] not in crop_hashes]
                if not pool:
                    missing.append(dict(episode=rid,reason='no independent query source')); continue
                seen=set(); written=set(); past_groups={u:set() for u in identities}; steps=[]
                for r in chosen:
                    obs=[]; old_seen=set(seen); current_groups={}
                    for o in r['observations']:
                        if not o.get('write_allowed',False): continue
                        sid=lookup[(r['id'],o['key'])]; s=samples[sid]; uid=s['vehicle_uid']
                        if uid not in identities: raise ValueError('observation not in candidates')
                        dedup=(uid,s['source_rgb_sha256'],s['crop_sha256'])
                        write=dedup not in written; written.add(dedup)
                        kind=str(gen.choice(['clean','blur','cover'],p=probabilities))
                        perturb=dict(kind=kind,seed=int(gen.integers(2**31)))
                        if kind=='blur': perturb['sigma']=int(gen.choice([1,2]))
                        if kind=='cover':
                            perturb['fraction']=float(gen.choice([.15,.30]))
                            w,h=s['box'][2]-s['box'][0],s['box'][3]-s['box'][1]
                            ch,cw=max(1,round(h*perturb['fraction']**.5)),max(1,round(w*perturb['fraction']**.5))
                            cover_rng=np.random.default_rng(perturb['seed'])
                            y,x=int(cover_rng.integers(h-ch+1)),int(cover_rng.integers(w-cw+1))
                            perturb['box_xywh']=[x,y,cw,ch]
                        obs.append(dict(sample_id=sid,vehicle_uid=uid,write=write,perturbation=perturb,
                            feature_key=_feature_key('observations',sid,perturb)))
                        if write:
                            seen.add(uid); current_groups[uid]=s['view_group']
                    repeated=[o['vehicle_uid'] for o in obs if o['write'] and o['vehicle_uid'] in old_seen]
                    repeat=int(identities.index(str(gen.choice(repeated)))) if repeated and gen.random()<cfg['repeat_probability'] else -1
                    def select(candidates):
                        if not candidates: return None
                        uids=sorted({samples[s]['vehicle_uid'] for s in candidates})
                        uid=str(qgen.choice(uids)); return str(qgen.choice([s for s in candidates if samples[s]['vehicle_uid']==uid]))
                    queries=[]; used=set()
                    focus=[s for s in pool if samples[s]['vehicle_uid'] in current_groups and
                        (cfg['group_mode']=='query_all' or samples[s]['view_group']==current_groups[samples[s]['vehicle_uid']])]
                    for j in range(2):
                        choices=focus if j==0 and focus else pool
                        sid=select([s for s in choices if s not in used])
                        if sid is None: continue
                        used.add(sid); uid=samples[sid]['vehicle_uid']
                        queries.append(dict(sample_id=sid,feature_key=_feature_key('queries',sid,{'kind':'clean'}),
                                            label=uid if uid in seen else None,group=0))
                    if cfg['group_mode']=='grouped':
                        candidates=[sid for sid in pool if sid not in used and samples[sid]['vehicle_uid'] in current_groups
                            and samples[sid]['view_group'] in past_groups[samples[sid]['vehicle_uid']]
                            and samples[sid]['view_group']!=current_groups[samples[sid]['vehicle_uid']]]
                        sid=select(candidates)
                        if sid is not None:
                            queries.append(dict(sample_id=sid,feature_key=_feature_key('queries',sid,{'kind':'clean'}),
                                label=samples[sid]['vehicle_uid'],group=1))
                    for uid,g in current_groups.items():
                        if g is not None: past_groups[uid].add(g)
                    steps.append(dict(frame_id=r['frame'],source_rgb=r['image'],source_rgb_sha256=r['image_sha256'],
                        parent_source_sha256=r.get('parent_source_sha256',r['image_sha256']),
                        observations=obs,queries=queries,repeat_candidate=repeat,
                        physical_visible=r.get('physical_visible')))
                ep=dict(id=rid,split=route['split'],scene_id=route['scene'],identities=identities,
                    source_reference_id=source_ref,steps=steps,actions=route.get('actions',[])[start:start+length-1])
                ep['context_sha256']=context_digest(ep); episodes.append(ep)
    if not episodes: raise ValueError('no eligible episodes / independent queries')
    splits={}
    for s in samples.values(): splits.setdefault(s['split'],set()).add(s['vehicle_uid'])
    shared={f'{a}|{b}':sorted(splits[a]&splits[b]) for a in splits for b in splits if a<b and splits[a]&splits[b]}
    if shared and cfg['split_mode']=='identity_disjoint': raise ValueError('identity overlap across outer splits')
    m=dict(schema_version=1,status='complete',config=cfg,samples=samples,episodes=episodes,
        source_run=str(source_dir),data_root=str(data_root),source_hashes={n:sha(source_dir/n) for n in ('plan.json','observations.json','routes.json')},
        encoder_source={k:plan.get(k) for k in ('teacher_sha256','backbone_sha256','teacher_feature_config','backbone_source','code')},
        split_audit=dict(shared_identities=shared,identity_mode=cfg['split_mode'],probe_used_for_tuning=False),
        coverage=dict(missing=missing,history_queries=sum(q['group']==1 for e in episodes for s in e['steps'] for q in s['queries'])))
    if cfg['group_mode']=='grouped' and not m['coverage']['history_queries']:
        raise ValueError('grouped data have no eligible historical queries')
    m['coverage'].update(validate_manifest(m))
    m['coverage']['by_split']={}
    for split in sorted(splits):
        eps=[e for e in episodes if e['split']==split]
        qs=[q for e in eps for st in e['steps'] for q in st['queries']]
        m['coverage']['by_split'][split]=dict(episodes=len(eps),queries=len(qs),
            history_queries=sum(q['group']==1 for q in qs),null_queries=sum(q['label'] is None for q in qs),
            writes=sum(o['write'] for e in eps for st in e['steps'] for o in st['observations']))
    if cfg['group_mode']=='grouped' and not m['coverage']['by_split'].get('train',{}).get('history_queries'):
        raise ValueError('grouped training split has no history queries')
    return m


def training_weights(episodes):
    counts=[0,0]
    for ep in episodes:
        for st in ep['steps']:
            for q in st['queries']: counts[int(q['label'] is None)]+=1
    if not sum(counts): raise ValueError('training episodes have no queries')
    # Normalize over actual training queries, NOT over the two class names.
    weights=np.array([1/n if n else 0 for n in counts],np.float64)
    weights/=np.dot(weights,counts)/sum(counts)
    return weights.astype(np.float32),counts


def validate_manifest(m):
    if m['schema_version']!=1 or m['status']!='complete': raise ValueError('incomplete manifest')
    samples=m['samples']; split_sources={}; split_crops={}; split_ids={}
    counts=dict(episodes=len(m['episodes']),queries=0,null_queries=0,writes=0)
    for ep in m['episodes']:
        ids=ep['identities']
        if len(ids)!=len(set(ids)) or not ids: raise ValueError('duplicate/empty candidates')
        if context_digest(ep)!=ep['context_sha256']: raise ValueError('context hash mismatch')
        blocked={s[k] for s in ep['steps'] for k in ('source_rgb_sha256','parent_source_sha256')}
        crops={samples[o['sample_id']]['crop_sha256'] for s in ep['steps'] for o in s['observations']}
        seen=set(); past={i:set() for i in ids}; dedup=set()
        for st in ep['steps']:
            old_seen=set(seen); current={}
            for o in st['observations']:
                s=samples[o['sample_id']]; uid=s['vehicle_uid']
                if s['split']!=ep['split'] or uid not in ids or o['vehicle_uid']!=uid or s['source_rgb_sha256']!=st['source_rgb_sha256']:
                    raise ValueError('observation source/split mismatch')
                if o['feature_key']!=_feature_key('observations',o['sample_id'],o['perturbation']):
                    raise ValueError('observation feature key mismatch')
                key=(uid,s['source_rgb_sha256'],s['crop_sha256'])
                if o['write']!=(key not in dedup): raise ValueError('write dedup mismatch')
                dedup.add(key)
                if o['write']:
                    seen.add(uid); current[uid]=s['view_group']; counts['writes']+=1
            rc=st['repeat_candidate']
            if rc!=-1 and (not 0<=rc<len(ids) or ids[rc] not in old_seen or ids[rc] not in current):
                raise ValueError('invalid repeat candidate')
            for q in st['queries']:
                s=samples[q['sample_id']]; uid=s['vehicle_uid']
                if s['split']!=ep['split'] or s['scene_id']!=ep['scene_id']: raise ValueError('query split mismatch')
                if s['source_rgb_sha256'] in blocked or s['parent_source_sha256'] in blocked or s['crop_sha256'] in crops:
                    raise ValueError('query source leakage')
                if q['label']!=(uid if uid in seen else None): raise ValueError('query label mismatch')
                if q['feature_key']!=_feature_key('queries',q['sample_id'],{'kind':'clean'}): raise ValueError('query feature key mismatch')
                if q['group'] not in (0,1): raise ValueError('unknown query group')
                if q['group']==1 and (m['config']['group_mode']!='grouped' or uid not in current or s['view_group'] not in past[uid] or s['view_group']==current[uid]):
                    raise ValueError('invalid history query group')
                counts['queries']+=1; counts['null_queries']+=q['label'] is None
            for uid,g in current.items():
                if g is not None: past[uid].add(g)
        split_sources.setdefault(ep['split'],set()).update(blocked)
        used=[samples[item['sample_id']] for st in ep['steps'] for item in st['observations']+st['queries']]
        split_sources[ep['split']].update(s[key] for s in used for key in ('source_rgb_sha256','parent_source_sha256'))
        split_crops.setdefault(ep['split'],set()).update(s['crop_sha256'] for s in used)
        split_ids.setdefault(ep['split'],set()).update(s['vehicle_uid'] for s in used)
    for a in split_sources:
        for b in split_sources:
            if a<b and split_sources[a]&split_sources[b]: raise ValueError('outer split source leakage')
            if a<b and split_crops[a]&split_crops[b]: raise ValueError('outer split crop leakage')
            if a<b and m['config']['split_mode']=='identity_disjoint' and split_ids[a]&split_ids[b]:
                raise ValueError('outer split identity leakage')
    counts['independent_source_images']=len({s['source_rgb_sha256'] for s in samples.values()})
    counts['vehicles']=len({s['vehicle_uid'] for s in samples.values()})
    counts['view_groups']=len({s['view_group'] for s in samples.values() if s['view_group'] is not None})
    return counts


def load_old_observation(entry,root):
    """Only for verified region_sequence_features_v1: old array name is observation."""
    path=checked_path(Path(root),entry['file'])
    if sha(path)!=entry['sha256']: raise ValueError('old cache hash mismatch')
    with np.load(path,allow_pickle=False) as a:
        x,v=a['observation'].copy(),a['valid'].copy()
    if x.shape!=(16,256) or v.shape!=(16,) or v.dtype!=bool or not v.any() or not np.isfinite(x).all():
        raise ValueError('invalid old observation cache')
    return x.astype(np.float32),v


def verify_code_sources(root,records):
    for name,expected in records.items():
        data=checked_path(Path(root),name).read_bytes()
        digests={hashlib.sha256(data).hexdigest(),hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()}
        if expected not in digests: raise ValueError('encoder/source code mismatch: '+name)


def load_feature(item, root, role, encoder_signature, sample_id=None):
    if item['role']!=role: raise ValueError('cache role mismatch')
    if item['encoder_signature']!=encoder_signature: raise ValueError('cache encoder mismatch')
    if sample_id is not None and item['sample_id']!=sample_id: raise ValueError('cache source mismatch')
    path=checked_path(Path(root),item['file'])
    if sha(path)!=item['sha256']: raise ValueError('cache hash mismatch')
    with np.load(path,allow_pickle=False) as a:
        tokens,valid=a['tokens'].copy(),a['valid'].copy()
    n=16 if role=='observations' else 64
    if tokens.shape!=(n,256) or valid.shape!=(n,) or valid.dtype!=bool or not valid.any() or not np.isfinite(tokens).all():
        raise ValueError('invalid feature shape/validity')
    return tokens.astype(np.float32),valid


def code_hashes():
    """LF-normalized code hashes are stable across Windows/Linux checkouts."""
    root=Path(__file__).resolve().parents[1]
    files=list((root/'dinoexp').glob('evidence_query_*.py'))
    files += [root/p for p in ('dinoexp/evidence_memory.py','dinoexp/region_sequence_training.py',
        'dinoexp/region_ab_prepare.py','dinoexp/dense_model.py','dinoexp/dense_views.py',
        'dinoexp/local_attention.py','dinoexp/region_geometry.py','dinoexp/core.py',
        'dinoexp/local_paths.py','dinoexp/dense_cache.py','dinoexp/region_ab_data.py','run_evidence_query.py')]
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in files if p.exists()}


def load_bundle(output):
    output=Path(output); m=read_json(output/'manifest.json'); p=read_json(output/'prepared.json')
    validate_manifest(m)
    if p['status']!='complete' or p['manifest_digest']!=digest_json(m) or p['manifest_sha256']!=sha(output/'manifest.json'):
        raise ValueError('prepared cache / manifest mismatch')
    if p['code_hashes']!=code_hashes(): raise ValueError('code changed since feature preparation; use a new run')
    for e in m['episodes']:
        for st in e['steps']:
            for role,items in (('observations',st['observations']),('queries',st['queries'])):
                for item in items:
                    s=m['samples'][item['sample_id']]; entry=p[role][item['feature_key']]
                    if entry['sample_digest']!=digest_json(s) or entry['role']!=role:
                        raise ValueError('cache source metadata mismatch')
                    expected=item.get('perturbation',{'kind':'clean'})
                    if entry['perturbation']!=expected: raise ValueError('cache preprocessing mismatch')
                    if entry['cache_key']!=digest_json([p['encoder_signature'],item['feature_key']]):
                        raise ValueError('cache encoder namespace mismatch')
    return m,p


def pack_query_batch(episodes, prepared, feature_root):
    if prepared['status']!='complete': raise ValueError('incomplete feature cache')
    b=len(episodes); t=max(len(e['steps']) for e in episodes); k=max(len(e['identities']) for e in episodes); j=3
    a=dict(observation=np.zeros((b,t,k,16,256),np.float32),observation_valid=np.zeros((b,t,k,16),bool),
        step_valid=np.zeros((b,t),bool),query=np.zeros((b,t,j,64,256),np.float32),query_valid=np.zeros((b,t,j,64),bool),
        query_label=np.full((b,t,j),k,np.int64),query_active=np.zeros((b,t,j),bool),
        query_group=np.zeros((b,t,j),np.int8),repeat_candidate=np.full((b,t),-1,np.int64))
    for bi,ep in enumerate(episodes):
        for ti,st in enumerate(ep['steps']):
            a['step_valid'][bi,ti]=True; a['repeat_candidate'][bi,ti]=st['repeat_candidate']
            for o in st['observations']:
                if not o['write']: continue
                ki=ep['identities'].index(o['vehicle_uid'])
                x,v=load_feature(prepared['observations'][o['feature_key']],feature_root,'observations',prepared['encoder_signature'],o['sample_id'])
                a['observation'][bi,ti,ki]=x; a['observation_valid'][bi,ti,ki]=v
            if len(st['queries'])>j: raise ValueError('query capacity exceeded')
            for ji,q in enumerate(st['queries']):
                x,v=load_feature(prepared['queries'][q['feature_key']],feature_root,'queries',prepared['encoder_signature'],q['sample_id'])
                a['query'][bi,ti,ji]=x; a['query_valid'][bi,ti,ji]=v
                a['query_active'][bi,ti,ji]=True; a['query_group'][bi,ti,ji]=q['group']
                a['query_label'][bi,ti,ji]=k if q['label'] is None else ep['identities'].index(q['label'])
    return a
