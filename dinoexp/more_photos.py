"""Controlled capture plan with held-out blueprint assets and spatially separated anchors."""

import copy
import math
import random

from .scene import validate_scene


VIEWS = [
    ('overview', 90, 80, 70), ('descend', 90, 55, 40),
    ('side', 90, 25, 26), ('front_side', 45, 20, 26), ('rear_side', 135, 20, 26),
    ('opposite_side', 270, 25, 26), ('opposite_front', 315, 20, 26),
    ('opposite_rear', 225, 20, 26), ('front_high', 45, 45, 35), ('rear_high', 135, 45, 35),
]


def choose_models(available, explicit=None):
    available = set(available)
    if explicit is not None:
        selected = list(explicit)
        if len(selected) != 8 or len(set(selected)) != 8 or set(selected) - available:
            raise ValueError('need exactly eight distinct available four-wheel colored vehicle assets')
        return selected
    preferred = ['vehicle.dodge.charger', 'vehicle.lincoln.mkz', 'vehicle.audi.a2',
                 'vehicle.audi.tt', 'vehicle.tesla.model3', 'vehicle.bmw.grandtourer',
                 'vehicle.mercedes.coupe', 'vehicle.nissan.patrol']
    # These IDs are preferences only, never invented replacements for unavailable assets.
    eligible = [m for m in available if not any(word in m.lower() for word in
                ('cop', 'police', 'ambulance', 'firetruck', 'truck', 'bus', 'sprinter'))]
    selected = [m for m in preferred if m in eligible]
    selected += sorted(set(eligible) - set(selected))
    if len(selected) < 8:
        raise ValueError(f'need eight eligible assets, found {len(selected)}: {selected}. '
                         'Inspect the actual CARLA inventory; no photos were collected.')
    return selected[:8]


def select_anchors(points, count=9, minimum_distance=80):
    if count < 1 or minimum_distance <= 0 or len(points) < count:
        raise ValueError('need enough spawn points and a positive spacing constraint')
    remaining = sorted(points, key=lambda p: p['index'])
    selected = [remaining.pop(0)]
    def distance(point):
        return min(math.dist(point['pose'][:2], other['pose'][:2]) for other in selected)
    while len(selected) < count:
        point = max(remaining, key=distance)
        if distance(point) < minimum_distance:
            raise ValueError(f'cannot find {count} anchors at least {minimum_distance}m apart; '
                             'use another map or explicitly adjust --min-anchor-distance')
        selected.append(point)
        remaining.remove(point)
    return selected


def build_plan(base, models, anchors, seed=11):
    if len(models) != 8 or len(set(models)) != 8 or len(anchors) != 9:
        raise ValueError('capture plan requires eight distinct models and nine anchors')
    if len({a['index'] for a in anchors}) != 9:
        raise ValueError('anchors must be distinct')
    models, anchors = list(models), list(anchors)
    rng = random.Random(seed)
    rng.shuffle(models)
    rng.shuffle(anchors)
    splits = {'train': (models[:4], anchors[:3]),
              'val': (models[4:6], anchors[3:6]), 'test': (models[6:], anchors[6:])}
    cases = []
    for split, (assets, locations) in splits.items():
        for pair in range(0, len(assets), 2):
            for loc, anchor in enumerate(locations):
                for lighting in range(2):
                    group = f'{split}_pair{pair // 2}_location{loc}_light{lighting}'
                    for target in ('A', 'B'):
                        cfg = copy.deepcopy(base)
                        cfg.pop('hide_environment_objects', None)
                        cfg.update(width=2048, height=1536, anchor_spawn_index=anchor['index'],
                                   query_yaw_mode='anchor',
                                   target_identity=target, weather_name=('noon', 'low_sun')[lighting])
                        cfg['weather'] = dict(sun_altitude_angle=(65, 18)[lighting],
                                              sun_azimuth_angle=(0, 120)[lighting],
                                              cloudiness=(10, 65)[lighting], precipitation=0,
                                              wetness=0, fog_density=0)
                        color = ('255,255,255', '180,180,180')[loc % 2]
                        cfg['candidates'] = [dict(identity=identity, blueprint=blueprint, color=color,
                            yaw_offset=(180 if i == (loc + lighting) % 2 else 0))
                            for i, (identity, blueprint) in enumerate(
                                zip(('A', 'B'), assets[pair:pair + 2], strict=True))]
                        if (loc + lighting) % 2:
                            cfg['candidates'].reverse()
                        cfg['reference'] = dict(azimuth=(90 if target == 'A' else 270),
                                                elevation=20, distance=12)
                        cfg['views'] = [dict(name=n, azimuth=a, elevation=e, distance=d)
                                        for n, a, e, d in VIEWS]
                        validate_scene(cfg)
                        cases.append(dict(name=f'{group}_ref{target}', split=split,
                                          capture_group=group, config=cfg))
    return dict(format_version=1, seed=seed, cases=cases,
                split_models={s: a for s, (a, _) in splits.items()},
                split_anchors={s: [p['index'] for p in a] for s, (_, a) in splits.items()},
                expected_query_crops=len(cases) * len(VIEWS) * 2,
                expected_reference_crops=len(cases),
                note='Held-out blueprint assets and anchors within one map, not cross-map testing. '
                     'GT boxes and instance identity do not certify visibility or discriminability.')
