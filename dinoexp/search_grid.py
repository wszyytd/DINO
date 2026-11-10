"""World-aligned observation graph and annotation-filtered offline routes."""

from collections import deque
import math
import random

HEADINGS = list(range(0, 360, 45))
MOVES = {'row_plus': (1, 0), 'row_minus': (-1, 0),
         'col_plus': (0, 1), 'col_minus': (0, -1)}
TEMPLATES = ('discover', 'compare', 'reappear', 'repeat_then_useful')
ACTION_IDS = {name: index for index, name in enumerate(
    (*MOVES, 'yaw_left', 'yaw_right', 'hold_capture', 'stop'))}


def build_graph(size, spacing, center, pitch, *, move_speed, turn_speed, hold_s, excluded=()):
    if (type(size) is not int or size < 3 or size % 2 != 1 or
        any(not math.isfinite(v) or v <= 0 for v in (spacing, move_speed, turn_speed, hold_s))):
        raise ValueError('grid requires odd size >= 3 and positive finite spacing and costs')
    if len(center) != 3 or not all(map(math.isfinite, (*center, pitch))) or not -90 < pitch < 0:
        raise ValueError('invalid grid center or fixed pitch')
    ids = {(r, c): f'r{r:02d}c{c:02d}' for r in range(size) for c in range(size)}
    omitted = {tuple(point) for point in excluded}
    if omitted - ids.keys() or len(omitted) == len(ids):
        raise ValueError('invalid excluded grid coordinates')
    points, states = {}, {}
    half = size // 2
    for (row, col), point in ids.items():
        xyz = [center[0] + (row-half)*spacing, center[1] + (col-half)*spacing, center[2]]
        neighbors = {name: ids[(row+dr, col+dc)] for name, (dr, dc) in MOVES.items()
                     if (row+dr, col+dc) in ids and (row+dr, col+dc) not in omitted}
        allowed = (row, col) not in omitted
        points[point] = dict(row=row, col=col, xyz=xyz, capture_eligible=allowed,
                             capture_status='pending' if allowed else 'excluded_by_config',
                             neighbors=neighbors if allowed else {})
        if not allowed:
            continue
        for direction, yaw in enumerate(HEADINGS):
            state_id = f'{point}_d{direction:02d}'
            links = [(name, f'{dest}_d{direction:02d}', spacing, 0, spacing/move_speed)
                     for name, dest in neighbors.items()]
            links.extend((name, f'{point}_d{(direction + delta) % 8:02d}', 0, 45*delta, 45/turn_speed)
                         for name, delta in (('yaw_left', -1), ('yaw_right', 1)))
            links.append(('hold_capture', state_id, 0, 0, hold_s))
            states[state_id] = dict(point_id=point, direction_id=direction, yaw_deg=yaw,
                                    pose=[*xyz, pitch, yaw, 0], edges={
                name: dict(to=dest, distance_m=distance, yaw_delta_deg=delta, cost_s=time,
                           traversability='unknown', collision_check='unknown')
                for name, dest, distance, delta, time in links
            })
    return dict(mode='discrete_view_graph', points=points, states=states)


def action_spec(grid):
    return dict(version='world-grid-v1', mode='discrete_view_graph', action_ids=ACTION_IDS,
                coordinates='CARLA +X is row+, +Y is col+; absolute yaw 0=+X, 90=+Y, clockwise from above',
                yaw_headings_deg=HEADINGS, move_distance_m=grid['spacing_m'],
                yaw_step_deg=45, costs='nominal configured speed and dwell; not acquisition or flight time',
                move_speed_m_s=grid['move_speed'], turn_speed_deg_s=grid['turn_speed'],
                hold_s=grid['hold_s'], hold_source='cached_rgb', stop_success=None,
                flight_feasibility=None, recommendation_labels=None)


def different_view(states, a, b, target_xy):
    first, second = states[a]['pose'], states[b]['pose']
    if first[:2] == second[:2]:
        return False
    angles = [math.degrees(math.atan2(p[1]-target_xy[1], p[0]-target_xy[0])) for p in (first, second)]
    return abs((angles[1] - angles[0] + 180) % 360 - 180) >= 25


def shortest_path(states, start, goal, rng):
    """Return actions along a shortest graph walk, with seeded tie breaking."""
    queue = deque([start])
    parents = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            actions = []
            while parents[current] is not None:
                current, action = parents[current]
                actions.append(action)
            return actions[::-1]
        edges = list(states[current]['edges'].items())
        rng.shuffle(edges)
        for action, edge in edges:
            dest = edge['to']
            if dest not in parents:
                parents[dest] = current, action
                queue.append(dest)
    return None


def walk(states, start, actions):
    path = [start]
    for name in actions:
        path.append(states[path[-1]]['edges'][name]['to'])
    return path


def route_events(graph, frames, route, target_xy):
    states = graph['states']
    path, actions = route['states'], route['actions']
    objects = [frames[s]['objects'] for s in path]
    target = [row[route['target_uid']]['quality'] for row in objects]
    seen = [q['visibility_present'] for q in target]
    usable = [q['recognition_quality'] == 'usable' for q in target]
    kind = route['template']
    if kind == 'discover':
        return not seen[0] and any(usable[1:]) and any(a in MOVES for a in actions)
    if kind == 'compare':
        return (all(q['quality']['visibility_present'] for q in objects[0].values())
                and any(ok and different_view(states, path[0], sid, target_xy)
                        for sid, ok in zip(path, usable, strict=True)))
    if kind == 'reappear':
        return (usable[0] and any(not seen[i] and not seen[i+1] and
                any(usable[j] and different_view(states, path[0], path[j], target_xy)
                    for j in range(i+2, len(path))) for i in range(1, len(path)-1)))
    if kind == 'repeat_then_useful':
        return (actions[:2] == ['hold_capture', 'hold_capture'] and
                any(ok and different_view(states, path[0], sid, target_xy)
                    for sid, ok in zip(path, usable, strict=True)))
    raise ValueError('unknown route template')


def derive_routes(graph, frames, objects, tasks, seed):
    states = graph['states']
    rng = random.Random(seed)
    routes, gaps = [], []
    used_paths = set()
    for template, target in tasks:
        other, = objects.keys() - {target}
        target_xy = objects[target]['pose'][:2]
        usable, absent, both, distractor = [], [], [], []
        for sid, row in frames.items():
            a, b = (row['objects'][uid]['quality'] for uid in (target, other))
            if a['recognition_quality'] == 'usable':
                usable.append(sid)
            if not a['visibility_present']:
                absent.append(sid)
                if b['visibility_present']:
                    distractor.append(sid)
            elif b['visibility_present']:
                both.append(sid)
        absent_set = set(absent)
        missing_edges = [(sid, name, edge['to']) for sid in absent
                         for name, edge in states[sid]['edges'].items()
                         if name != 'hold_capture' and edge['to'] in absent_set]
        starts = {'discover': absent, 'compare': both, 'reappear': usable,
                  'repeat_then_useful': usable}[template]
        best = None
        for _ in range(128 if starts and usable else 0):
            start = rng.choice(distractor if template == 'discover' and distractor and rng.random() < .8
                               else starts)
            goals = usable if template == 'discover' else [
                sid for sid in usable if different_view(states, start, sid, target_xy)]
            if not goals:
                continue
            goal = rng.choice(goals)
            if template == 'reappear':
                if not missing_edges:
                    break
                a, name, b = rng.choice(missing_edges)
                first = shortest_path(states, start, a, rng)
                last = shortest_path(states, b, goal, rng)
                if first is None or last is None:
                    continue
                actions = first + [name] + last
            else:
                actions = shortest_path(states, start, goal, rng)
                if actions is None:
                    continue
                if template == 'repeat_then_useful':
                    actions = ['hold_capture', 'hold_capture'] + actions
            if len(actions) >= 32:
                continue
            path = walk(states, start, actions)
            while len(path) < 16:
                edges = states[path[-1]]['edges']
                moves = [n for n in MOVES if n in edges and edges[n]['to'] not in path]
                fresh = [n for n, edge in edges.items() if edge['to'] not in path]
                choices = moves or fresh or [n for n in edges if n != 'hold_capture']
                action = rng.choice(choices)
                actions.append(action)
                path.append(edges[action]['to'])
            route = dict(template=template, target_uid=target, states=path, actions=actions)
            if tuple(path) in used_paths or not route_events(graph, frames, route, target_xy):
                continue
            # Favor useful, diverse actual source views and movement over extra turns.
            score = (len({states[s]['point_id'] for s in path}),
                     len({s for s in path if s in usable}),
                     sum(a in MOVES for a in actions), -len(path))
            if best is None or score > best[0]:
                best = score, route
        if best is None:
            gaps.append(dict(template=template, target_uid=target, reason='no qualifying connected route found'))
        else:
            route = best[1]
            used_paths.add(tuple(route['states']))
            routes.append(route)
    return routes, gaps


def route_rows(graph, frames, route, reference):
    labels, inputs = [], []
    elapsed = 0
    target = route['target_uid']
    for step, sid in enumerate(route['states']):
        frame = frames[sid]
        name = route['actions'][step-1] if step else None
        action = None
        if name is not None:
            before = route['states'][step-1]
            edge = graph['states'][before]['edges'][name]
            if edge['to'] != sid:
                raise ValueError('route contains a non-adjacent transition')
            elapsed += edge['cost_s']
            action = dict(id=ACTION_IDS[name], name=name, status='offline_transition',
                          image_source='cached_repeat' if name == 'hold_capture' else 'grid_capture',
                          **edge)
        labels.append(dict(step_index=step, state_id=sid, time_s=elapsed,
                           source_sensor_frame=frame['sensor_frame'], action_from_previous=action,
                           image_path=frame['image_path'],
                           visibility_present=frame['objects'][target]['quality']['visibility_present'],
                           quality=frame['objects'][target]['quality']['recognition_quality']))
        inputs.append(dict(reference=reference, rgb=frame['image_path'], time_s=elapsed,
                           previous_action_id=ACTION_IDS[name] if name else None, legal_action_mask=None))
    return labels, inputs
