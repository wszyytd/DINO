"""Fixed-height inspection collection; no claim of identity-disjoint training splits."""

import copy
import math

from .scene import validate_scene


def build_plan(base, models, anchors, height=8., radius=18., angle_step=15):
    if (len(models) != 2 or len(set(models)) != 2 or not anchors
            or len({a['index'] for a in anchors}) != len(anchors)):
        raise ValueError('need two distinct actual vehicle assets and distinct anchors')
    if (not all(math.isfinite(x) and x > 0 for x in (height, radius))
            or type(angle_step) is not int or angle_step < 1 or 360 % angle_step):
        raise ValueError('positive finite geometry and an angle step dividing 360 required')
    elevation = math.degrees(math.atan2(height, radius))
    distance = math.hypot(height, radius)
    pairs = [(models, ('235,235,235', '235,235,235'), 'same_color_different_models'),
             ([models[0]]*2, ('235,235,235', '210,210,210'), 'same_model_close_colors'),
             ([models[1]]*2, ('180,180,180', '205,205,205'), 'same_model_close_colors')]
    cases = []
    for pair_index, (assets, colors, kind) in enumerate(pairs):
        for loc, anchor in enumerate(anchors):
            group = f'pair{pair_index}_location{loc}'
            for target in ('A', 'B'):
                cfg = copy.deepcopy(base)
                cfg.pop('hide_environment_objects', None)
                cfg.update(width=2048, height=1536, anchor_spawn_index=anchor['index'],
                           fixed_camera_height_m=height, query_yaw_mode='anchor', target_identity=target)
                cfg['candidates'] = [dict(identity=identity, blueprint=asset, color=color, yaw_offset=0)
                                     for identity, asset, color in zip(('A', 'B'), assets, colors, strict=True)]
                if loc % 2:
                    cfg['candidates'].reverse()
                cfg['weather'] = dict(sun_altitude_angle=60, sun_azimuth_angle=30,
                                      cloudiness=10 if loc % 2 == 0 else 65, precipitation=0,
                                      wetness=0, fog_density=0)
                cfg['reference'] = dict(azimuth=45, elevation=elevation, distance=distance)
                cfg['views'] = [dict(name=f'orbit_{angle:03d}', azimuth=angle,
                                     elevation=elevation, distance=distance)
                                for angle in range(0, 360, angle_step)]
                validate_scene(cfg)
                cases.append(dict(name=f'{group}_ref{target}', group=group, pair_kind=kind,
                                  anchor_z=anchor['pose'][2], config=cfg))
    return dict(format_version=1, purpose='visual inspection; not a held-out evaluation collection',
                fixed_height_above_anchor_m=height, horizontal_radius_m=radius, angle_step=angle_step,
                expected_crops=len(cases)*(1+2*(360//angle_step)),
                expected_full_images=len(cases)*(1+360//angle_step), cases=cases,
                note='A/B reference tasks repeat a physical configuration; not independent scenes. '
                     'Same-model pairs differ only in color; human inspection must assess difficulty.')
