"""Read-only planning, verified resume and portable dense-manifest export."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json
import re
import subprocess
import sys
from .core import checked_path
from .dense_cache import save_json
from .dense_data import read_dense_manifest
from .dense_capture import validate_dense_config
from .local_pairs import sha256


def make_capture_plan(source,views,max_train_groups=None):
    if source.get('status')!='complete':
        raise ValueError('source collection must be complete; preserve the earlier interrupted data')
    if not views or len(views)!=len(set(views)) or (max_train_groups is not None and max_train_groups<1):
        raise ValueError('need distinct view names and positive pilot group limit')
    groups=defaultdict(list)
    for case in source['plan']['cases']:
        groups[(case['split'],case['capture_group'])].append(case)
    selected=set()
    for split in ('train','val','test'):
        candidates=[(key,items) for key,items in groups.items() if key[0]==split]
        if not candidates:raise ValueError('source has no '+split+' groups')
        limit=max_train_groups if split=='train' else len(candidates)
        # Cover asset pairings before adding repeated backgrounds of the same pair.
        buckets=defaultdict(list)
        for key,items in candidates:
            assets=tuple(sorted(c['blueprint'] for c in items[0]['config']['candidates']))
            buckets[assets].append((key,items))
        count=0
        while buckets and (limit is None or count<limit):
            for assets in list(buckets):
                if limit is not None and count>=limit:break
                key,items=buckets[assets].pop(0);selected.add(key);count+=1
                if not buckets[assets]:del buckets[assets]
    cases=[];models={s:set() for s in ('train','val','test')};anchors={s:set() for s in models}
    for key,items in groups.items():
        if key not in selected:continue
        if len(items)!=2 or {c['config']['target_identity'] for c in items}!={'A','B'}:
            raise ValueError('dense collection requires paired A/B references')
        for case in items:
            case=deepcopy(case);cfg=case['config']
            if cfg.get('hide_environment_objects'):raise ValueError('dense capture does not support environment hiding')
            cfg['views']=[next((deepcopy(v) for v in cfg['views'] if v['name'].removesuffix('_raised')==name),None) for name in views]
            if any(v is None for v in cfg['views']):raise ValueError('requested dense view missing in saved config')
            cfg.update(dense_case=case['name'],dense_split=case['split'],dense_group=case['capture_group'])
            validate_dense_config(cfg)
            if not re.fullmatch(r'[A-Za-z0-9_-]+',case['name']):raise ValueError('unsafe dense case name')
            models[case['split']].update(c['blueprint'] for c in cfg['candidates'])
            anchors[case['split']].add(cfg['anchor_spawn_index']);cases.append(case)
    for a,b in (('train','val'),('train','test'),('val','test')):
        if models[a]&models[b] or anchors[a]&anchors[b]:raise ValueError('source assets/anchors overlap across splits')
    return cases


def child_samples(root,case):
    directory=checked_path(root,case['split']+'/'+case['name'])
    run=json.loads((directory/'run.json').read_text(encoding='utf-8'))
    if run.get('status')!='complete' or run.get('cleanup_errors'):
        raise ValueError('dense child capture is not complete')
    if 'config' in case and run.get('config')!=case['config']:
        raise ValueError('completed dense child config differs from saved plan')
    if run.get('instance_id_method')!='remove_restore_probe' or len(run.get('associations',[]))!=2:
        raise ValueError('dense capture has no verified opaque instance association')
    samples=run.get('samples',[])
    if not samples or not any(r['presence']=='absent' for r in samples):
        raise ValueError('dense capture needs present/confirmed Null samples')
    if 'config' in case and len(samples)!=len(case['config']['views'])+1:
        raise ValueError('dense child has incomplete query views')
    portable=[];prefix=directory.relative_to(Path(root).resolve()).as_posix()+'/'
    for sample in samples:
        sample=deepcopy(sample)
        for key in ('reference','image'):
            checked_path(directory,sample[key]);sample[key]=prefix+sample[key]
        for key in ('positive_masks','negative_masks'):
            sample[key]=[prefix+p for p in sample.get(key,[])]
        sample['mask_sha256']={prefix+k:v for k,v in sample['mask_sha256'].items()}
        if sample.get('unknown_mask'):sample['unknown_mask']=prefix+sample['unknown_mask']
        portable.append(sample)
    return portable


def export_manifest(root,cases):
    root=Path(root).resolve();samples=[]
    for case in cases:samples+=child_samples(root,case)
    value=dict(format_version=1,task='reference_guided_dense_search',samples=samples,
        note='Visible masks verified by owned-actor remove/restore probes. Limited pilot coverage is not broad generalization.')
    temporary=root/'dense_manifest.pending.json';save_json(temporary,value)
    read_dense_manifest(temporary)
    temporary.replace(root/'dense_manifest.json')
    return value


def collect_dense(source,output,views=('overview','descend'),max_train_groups=None,*,resume=False,
                  dry_run=False,runner=subprocess.run,take_tick=False):
    output=Path(output).resolve()
    if resume:
        marker=json.loads((output/'collection.json').read_text(encoding='utf-8'));cases=marker['cases']
    else:
        if output.exists():raise ValueError('dense output exists; use --resume or a new directory')
        original=json.loads((Path(source)/'collection.json').read_text(encoding='utf-8-sig'))
        cases=make_capture_plan(original,list(views),max_train_groups)
        marker=dict(status='incomplete',source_collection_sha256=sha256(Path(source)/'collection.json'),
                    cases=cases,completed_cases=[],purpose='Small verified-mask pilot' if max_train_groups else 'Dense capture')
    done=[];pending=[]
    for case in cases:
        directory=checked_path(output,case['split']+'/'+case['name'])
        if (directory/'run.json').is_file():
            try:
                samples=child_samples(output,case)
                # Verify image/mask content before accepting a child, without rewriting it.
                for row in samples:
                    for role in ('reference','image'):
                        if sha256(checked_path(output,row[role]))!=row[role+'_sha256']:
                            raise ValueError('dense resume image hash mismatch')
                    for path,expected in row['mask_sha256'].items():
                        if sha256(checked_path(output,path))!=expected:raise ValueError('dense resume mask hash mismatch')
                done.append(case['name']);continue
            except (ValueError,OSError,KeyError):
                if case['name'] in marker.get('completed_cases',[]):raise
        pending.append(case)
    summary=dict(cases=len(cases),completed=len(done),remaining=len(pending),
                 expected_queries=sum(len(c['config']['views'])+1 for c in cases),
                 references=len(cases),views=list(views),mask_source='synchronized RGB+instance, remove/restore verified')
    if dry_run:return summary
    output.mkdir(parents=True,exist_ok=True);(output/'_configs').mkdir(exist_ok=True)
    marker['completed_cases']=done;marker['status']='incomplete';save_json(output/'collection.json',marker)
    try:
        for case in pending:
            destination=checked_path(output,case['split']+'/'+case['name'])
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                archive=checked_path(output,'_failed');archive.mkdir(exist_ok=True)
                stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
                target=checked_path(output,'_failed/'+case['split']+'_'+case['name']+'_'+stamp)
                destination.rename(target)
            config=checked_path(output,'_configs/'+case['name']+'.json');save_json(config,case['config'])
            command=[sys.executable,'-u',str(Path(__file__).resolve().parents[1]/'capture_dense_scene.py'),
                     '--config',str(config),'--output',str(destination)]
            if take_tick:command.append('--take-tick-ownership')
            print(f'Dense capture {len(done)+1}/{len(cases)}: {case["name"]}',flush=True)
            runner(command,check=True);child_samples(output,case)
            done.append(case['name']);marker['completed_cases']=done;save_json(output/'collection.json',marker)
        export_manifest(output,cases);marker['status']='complete';marker.pop('error',None)
    except BaseException as error:
        marker['error']=str(error);raise
    finally:
        save_json(output/'collection.json',marker)
    return summary
