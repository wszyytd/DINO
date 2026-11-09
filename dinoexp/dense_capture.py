"""Synchronized pixel masks identified by remove/restore probes, never actor.id."""
from contextlib import contextmanager
import html
import math
from pathlib import Path
import queue
import re
import time
import numpy as np
from PIL import Image
from capture import as_transform,find_blueprint
from .core import image_quality,pose_matches,pose_values
from .scene import candidate_poses,capture_view_pose
from .dense_cache import save_json
from .local_pairs import sha256


def grid_views(grid):
    size = grid.get('size')
    if type(size) is not int or size < 1:
        raise ValueError('grid size must be a positive integer')
    for field in ('spacing_m', 'height_m', 'oblique_pitch_deg'):
        value = grid.get(field)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('invalid grid ' + field)
    if grid['spacing_m'] <= 0 or grid['height_m'] <= 0:
        raise ValueError('grid spacing and height must be positive')
    if not -90 < grid['oblique_pitch_deg'] < 0:
        raise ValueError('oblique pitch must be between -90 and 0 degrees')

    middle = (size - 1) / 2
    views = []
    for row in range(size):
        for col in range(size):
            for direction, yaw in (('down', 0), ('front', 0), ('right', 90),
                                   ('back', 180), ('left', 270)):
                views.append(dict(
                    name=f'r{row:02d}_c{col:02d}_{direction}', grid_cell=[row, col],
                    direction=direction, relative_yaw=yaw,
                    pitch=-90 if direction == 'down' else grid['oblique_pitch_deg'],
                    offset_forward_m=(middle - row) * grid['spacing_m'],
                    offset_right_m=(col - middle) * grid['spacing_m'],
                    height_m=grid['height_m'],
                ))
    # Use the most central overhead frame to identify both vehicles first.
    calibration = min(views, key=lambda view: (
        view['direction'] != 'down',
        view['offset_forward_m'] ** 2 + view['offset_right_m'] ** 2,
    ))
    views.remove(calibration)
    return [calibration, *views]


def grid_camera_pose(center, anchor, view):
    angle = math.radians(anchor[4])
    forward, right = view['offset_forward_m'], view['offset_right_m']
    return (
        center[0] + forward * math.cos(angle) - right * math.sin(angle),
        center[1] + forward * math.sin(angle) + right * math.cos(angle),
        anchor[2] + view['height_m'], view['pitch'],
        (anchor[4] + view['relative_yaw'] + 180) % 360 - 180, 0.0,
    )


def validate_dense_config(cfg):
    from .scene import validate_scene
    validate_scene(cfg)
    if 'grid_capture' in cfg:
        cfg = dict(cfg, views=grid_views(cfg['grid_capture']))
    for field in ('width','height','warmup_ticks','settle_ticks'):
        if type(cfg.get(field)) is not int or cfg[field]<1:
            raise ValueError('dense '+field+' must be a positive integer')
    for field in ('timeout','fixed_delta','fov'):
        if not isinstance(cfg.get(field),(int,float)) or not math.isfinite(cfg[field]) or cfg[field]<=0:
            raise ValueError('invalid dense '+field)
    if cfg['fov']>=180:raise ValueError('dense fov must be below180')
    views=cfg.get('views',[])
    names=[v.get('name','') for v in views]
    if (not names or len(names)!=len(set(names)) or
            any(not re.fullmatch(r'[A-Za-z0-9_-]+',name) for name in names) or names[0]+'_absent' in names):
        raise ValueError('dense view names must be distinct safe filenames without Null collision')
    orbit_views = [] if 'grid_capture' in cfg else views
    for view in [cfg['reference'],*orbit_views]:
        for field in ('azimuth','elevation','distance'):
            if not isinstance(view.get(field),(int,float)) or not math.isfinite(view[field]):
                raise ValueError('invalid dense view '+field)
        if not 0<=view['elevation']<=90 or view['distance']<=0:
            raise ValueError('invalid dense view elevation/distance')
    if cfg.get('hide_environment_objects'):
        raise ValueError('dense capture rejects environment hiding without known previous state')
    if set(cfg.get('camera_attributes',{}))&{'image_size_x','image_size_y','fov','sensor_tick'}:
        raise ValueError('camera attributes cannot override geometry/timing')
    return cfg


def decode_instance_ids(rgb):
    rgb=np.asarray(rgb)
    if rgb.ndim!=3 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8:
        raise ValueError('raw instance image needs uint8 RGB channels')
    return rgb[:,:,1].astype(np.uint16)+(rgb[:,:,2].astype(np.uint16)<<8)


def discover_instance_ids(before,hidden,restored,min_pixels=4):
    if before.shape!=hidden.shape or before.shape!=restored.shape:
        raise ValueError('instance probe frame dimensions differ')
    ids=[]
    for value in np.unique(before):
        region=before==value;count=int(region.sum())
        if value==0 or count<min_pixels:
            continue
        if int((hidden==value).sum())<=count*.01:
            if float((restored[region]==value).mean())<.99:
                raise ValueError('instance probe did not restore original object pixels')
            ids.append(int(value))
    if not ids:
        raise ValueError('instance removal probe found no unambiguous disappearing IDs')
    return ids


def occlusion_probe_pose(actor_pose,camera_pose):
    pose=list(actor_pose)
    # All supported query cameras look horizontally/down. Move above/behind them,
    # never below the map where Unreal's KillZ may destroy the actor before restore.
    pose[2]=max(actor_pose[2],camera_pose[2])+1000
    return pose


@contextmanager
def managed_capture(world,owned,run):
    initial,weather=world.get_settings(),world.get_weather()
    try:
        yield
    finally:
        for actor in reversed(owned):
            if actor.type_id.startswith('sensor.'):
                try:
                    actor.stop()
                except Exception as error:
                    run['cleanup_errors'].append(f'actor {actor.id} stop: {error}')
            try:
                if not actor.destroy():
                    raise RuntimeError('destroy returned false')
            except Exception as error:
                run['cleanup_errors'].append(f'actor {actor.id}: {error}')
        for name,restore in [('weather',lambda:world.set_weather(weather)),
                             ('settings',lambda:world.apply_settings(initial))]:
            try:
                restore()
            except Exception as error:
                run['cleanup_errors'].append(f'{name}: {error}')


def _receive(q):
    def receive(image):
        try:
            q.put_nowait(image)
        except queue.Full:
            try:
                q.get_nowait();q.put_nowait(image)
            except (queue.Empty,queue.Full):
                pass
    return receive


def next_pair(world,queues,desired,cfg):
    barrier=None
    for _ in range(cfg['warmup_ticks']):
        barrier=world.tick(cfg['timeout'])
    pending=[{},{}];deadline=time.monotonic()+cfg['timeout']
    while time.monotonic()<deadline:
        world.tick(min(cfg['timeout'],max(.1,deadline-time.monotonic())))
        for i,q in enumerate(queues):
            while True:
                try:
                    frame=q.get_nowait()
                except queue.Empty:
                    break
                if frame.frame>=barrier and pose_matches(pose_values(frame.transform),desired):
                    pending[i][frame.frame]=frame
            if len(pending[i])>64:
                pending[i]={k:pending[i][k] for k in sorted(pending[i])[-64:]}
        common=set(pending[0])&set(pending[1])
        if common:
            key=max(common)
            pair=[p[key] for p in pending]
            if (pair[0].width,pair[0].height)!=(pair[1].width,pair[1].height):
                raise RuntimeError('RGB/instance camera dimensions differ')
            return pair
        time.sleep(.002)
    raise TimeoutError('no synchronized RGB/instance frames at requested pose')


def _rgb(image):
    a=np.frombuffer(image.raw_data,np.uint8).reshape(image.height,image.width,4)
    return a[:,:,:3][:,:,::-1].copy()


def write_preview(output,run):
    cards=[];(output/'preview').mkdir(exist_ok=True)
    for i,sample in enumerate(run['samples']):
        with Image.open(output/sample['image']) as im:source=im.convert('RGB')
        for paths,color in ((sample['positive_masks'],(0,220,40)),(sample['negative_masks'],(40,110,255))):
            for path in paths:
                with Image.open(output/path) as im:alpha=im.convert('L').point(lambda x:round(x*.4))
                source=Image.composite(Image.new('RGB',source.size,color),source,alpha)
        source.thumbnail((1200,900));source.save(output/'preview'/f'{i}.png')
        cards.append(f'<h2>{html.escape(sample["id"])} — {sample["presence"]}</h2><img src="preview/{i}.png">')
    note = ('每个网格点有四向斜视和一张俯视。absent 表示当前图中看不到目标。'
            if 'grid_capture' in run['config'] else 'Null帧已移除自建目标。')
    (output/'preview.html').write_text('<!doctype html><meta charset="utf-8"><title>可见实例掩码采集</title>'
        '<style>body{font:17px sans-serif;margin:30px auto;max-width:1200px}img{max-width:100%}</style>'
        '<h1>参考与可见实例掩码</h1><p>绿色为参考目标，蓝色为干扰车辆。'+note+'</p>'
        '<img src="reference.png">'+''.join(cards),encoding='utf-8')


def capture_dense_scene(carla,world,versions,cfg,output,take_tick=False):
    cfg = validate_dense_config(cfg)
    grid_mode = 'grid_capture' in cfg
    output=Path(output)
    if cfg.get('hide_environment_objects'):
        raise ValueError('dense capture rejects environment hiding without known previous state')
    if world.get_settings().synchronous_mode and not take_tick:
        raise RuntimeError('exclusive tick ownership required for synchronous world')
    if len(world.get_actors().filter('vehicle.*')):
        raise RuntimeError('dense capture needs empty scene; existing vehicles will not be removed')
    if output.exists():
        raise ValueError('output exists; resume through collector to preserve previous attempt')
    points=world.get_map().get_spawn_points();index=cfg['anchor_spawn_index']
    if not 0<=index<len(points):
        raise ValueError('dense anchor unavailable')
    library=world.get_blueprint_library()
    for name in ('sensor.camera.rgb','sensor.camera.instance_segmentation'):
        find_blueprint(library,name)
    output.mkdir(parents=True);(output/'images').mkdir();(output/'masks').mkdir()
    owned=[];actors=[]
    run=dict(status='incomplete',config=cfg,map=world.get_map().name,versions=versions,
             samples=[],cleanup_errors=[],instance_id_method='remove_restore_probe',associations=[])
    try:
        with managed_capture(world,owned,run):
            settings=world.get_settings();settings.synchronous_mode=True
            settings.fixed_delta_seconds=cfg['fixed_delta'];settings.no_rendering_mode=False
            settings.substepping=True;settings.max_substep_delta_time=.01
            settings.max_substeps=max(10,math.ceil(cfg['fixed_delta']/.01));world.apply_settings(settings)
            weather=world.get_weather()
            for k,v in cfg['weather'].items():setattr(weather,k,v)
            world.set_weather(weather)
            anchor=pose_values(points[index])
            poses=candidate_poses(anchor,cfg['separation'],
                                  [c.get('yaw_offset',0) for c in cfg['candidates']])
            for candidate,pose in zip(cfg['candidates'],poses,strict=True):
                bp=find_blueprint(library,candidate['blueprint']);bp.set_attribute('color',candidate['color'])
                if bp.has_attribute('role_name'):bp.set_attribute('role_name','dino_dense_owned')
                actor=world.try_spawn_actor(bp,as_transform(carla,pose))
                if actor is None:raise RuntimeError('cannot spawn dense candidate '+candidate['identity'])
                owned.append(actor);actors.append((candidate,actor))
                actor.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
            previous=[a.get_transform() for _,a in actors]
            for _ in range(cfg['settle_ticks']):
                previous=[a.get_transform() for _,a in actors];world.tick(cfg['timeout'])
            geometry=[]
            for (candidate,actor),old in zip(actors,previous,strict=True):
                tf=actor.get_transform();velocity=actor.get_velocity()
                if not pose_matches(pose_values(tf),pose_values(old)) or math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2)>.1:
                    raise RuntimeError('dense candidate not stationary')
                actor.set_simulate_physics(False)
                center=np.mean([[p.x,p.y,p.z] for p in actor.bounding_box.get_world_vertices(tf)],axis=0)
                geometry.append((candidate,actor,tf,center))
            center=np.mean([g[3] for g in geometry],axis=0)
            target=next(g for g in geometry if g[0]['identity']==cfg['target_identity'])
            sensors=[];queues=[]
            for name in ('sensor.camera.rgb','sensor.camera.instance_segmentation'):
                bp=find_blueprint(library,name)
                attrs=dict(image_size_x=cfg['width'],image_size_y=cfg['height'],fov=cfg['fov'],sensor_tick=0)
                if name.endswith('.rgb'):
                    if set(attrs)&set(cfg['camera_attributes']):raise ValueError('camera attributes cannot override geometry/timing')
                    attrs.update(cfg['camera_attributes'])
                else:
                    attrs.update({k:v for k,v in cfg['camera_attributes'].items() if k.startswith('lens_')})
                for k,v in attrs.items():
                    if not bp.has_attribute(k):raise ValueError('unsupported dense sensor attribute '+k)
                    bp.set_attribute(k,str(v))
                sensor=world.spawn_actor(bp,carla.Transform(carla.Location(z=100)))
                owned.append(sensor);sensors.append(sensor)
                q=queue.Queue(maxsize=64);queues.append(q);sensor.listen(_receive(q))
            def photograph(view,is_reference=False):
                if grid_mode and not is_reference:
                    pose=grid_camera_pose(center,anchor,view)
                else:
                    pose=capture_view_pose(target[3] if is_reference else center,anchor[4],
                        target[2].rotation.yaw,view,is_reference=is_reference,
                        query_yaw_mode=cfg.get('query_yaw_mode','anchor'))
                for sensor in sensors:sensor.set_transform(as_transform(carla,pose))
                rgb,instance=next_pair(world,queues,pose,cfg)
                picture=Image.fromarray(_rgb(rgb))
                if image_quality(picture)['near_black']:raise RuntimeError('near-black dense RGB frame')
                return picture,decode_instance_ids(_rgb(instance)),rgb
            # Camera IDs are opaque engine keys. Calibrate by interventions on owned actors.
            calibration=cfg['views'][0]
            _,before,frame=photograph(calibration)
            mapped={}
            for candidate,actor,tf,_ in geometry:
                hidden=occlusion_probe_pose(pose_values(tf),pose_values(frame.transform))
                try:
                    actor.set_transform(as_transform(carla,hidden))
                    _,removed,hidden_frame=photograph(calibration)
                finally:
                    actor.set_transform(tf)
                _,restored,restored_frame=photograph(calibration)
                ids=discover_instance_ids(before,removed,restored)
                if any(set(ids)&set(other) for other in mapped.values()):
                    raise RuntimeError('ambiguous overlapping segmentation instance IDs')
                if int(np.isin(before,ids).sum())<64:
                    raise RuntimeError('too few visible pixels for dense instance association')
                mapped[candidate['identity']]=ids
                run['associations'].append(dict(identity=candidate['identity'],python_actor_id=actor.id,
                    opaque_segmentation_ids=ids,base_frame=frame.frame,hidden_frame=hidden_frame.frame,
                    restored_frame=restored_frame.frame))
            reference,ids,ref_frame=photograph(cfg['reference'],True)
            mask=np.isin(ids,mapped[cfg['target_identity']])
            yy,xx=np.where(mask)
            if len(xx)<64:raise RuntimeError('reference target not visibly segmented')
            dx=max(2,round((xx.max()-xx.min()+1)*.04));dy=max(2,round((yy.max()-yy.min()+1)*.04))
            bounds=(max(0,int(xx.min())-dx),max(0,int(yy.min())-dy),
                    min(reference.width,int(xx.max())+dx+1),min(reference.height,int(yy.max())+dy+1))
            reference.crop(bounds).save(output/'reference.png')
            reference_hash=sha256(output/'reference.png')
            def save_sample(view,picture,instances,rgb_frame,absent=False):
                name=view['name']+('_absent' if absent else '')
                image_path='images/'+name+'.png';picture.save(output/image_path)
                masks={}
                for identity,keys in mapped.items():
                    visible=np.isin(instances,keys)
                    if absent and identity==cfg['target_identity'] and visible.any():
                        raise RuntimeError('stale target pixels after owned target removal')
                    if visible.any():
                        path=f'masks/{name}_{identity}.png'
                        Image.fromarray(visible.astype(np.uint8)*255).save(output/path);masks[identity]=path
                if not grid_mode and not absent and cfg['target_identity'] not in masks:
                    raise RuntimeError('present query target has no visible mask; inspect geometry/occlusion')
                positive=[masks[cfg['target_identity']]] if cfg['target_identity'] in masks else []
                negative=[p for k,p in masks.items() if k!=cfg['target_identity']]
                sample=dict(id=cfg.get('dense_case',output.name)+':'+name,
                    split=cfg.get('dense_split','train'),group=cfg.get('dense_group',output.name),
                    reference='reference.png',reference_sha256=reference_hash,image=image_path,
                    image_sha256=sha256(output/image_path),positive_masks=positive,negative_masks=negative,
                    mask_sha256={p:sha256(output/p) for p in masks.values()},presence='present' if positive else 'absent',
                    supervision='simulation',mask_source='carla_instance_verified',
                    camera_pose=list(pose_values(rgb_frame.transform)),fov=cfg['fov'],frame=rgb_frame.frame)
                if grid_mode:
                    sample.update(grid_cell=view['grid_cell'], direction=view['direction'], visible_masks=masks)
                run['samples'].append(sample)
                save_json(output/'run.json',run)
                print(f'dense {name}: {sample["presence"]}, {len(masks)} vehicles visible',flush=True)
            for view in cfg['views']:
                picture,instances,rgb_frame=photograph(view)
                if any(not pose_matches(pose_values(a.get_transform()),pose_values(tf)) for _,a,tf,_ in geometry):
                    raise RuntimeError('owned candidate moved during dense capture')
                save_sample(view,picture,instances,rgb_frame)
            if not grid_mode:
                # Existing orbit dataset uses an intervention to produce one Null sample.
                if not target[1].destroy():raise RuntimeError('cannot remove owned target for Null capture')
                owned.remove(target[1])
                picture,instances,rgb_frame=photograph(cfg['views'][0])
                save_sample(cfg['views'][0],picture,instances,rgb_frame,True)
        if run['cleanup_errors']:raise RuntimeError('dense cleanup failed: '+str(run['cleanup_errors']))
        run['status']='complete'
    except BaseException as error:
        run['error']=str(error);raise
    finally:
        save_json(output/'run.json',run)
        if (output/'reference.png').is_file():write_preview(output,run)
    return run
