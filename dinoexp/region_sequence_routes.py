"""Target-independent graph patrols. Every intermediate camera state is observed."""
import math


# Scan neighboring headings before moving. No object labels, reference or scores here.
ACTION_ORDER = ('yaw_right', 'yaw_left', 'row_plus', 'col_plus', 'row_minus', 'col_minus')


def _edges(graph, state, allowed):
    return [(name, edge) for name in ACTION_ORDER
            if (edge := graph['states'][state]['edges'].get(name)) is not None
            and edge['to'] in allowed and edge['to'] != state]


def _action(graph, start, end):
    names = [name for name, edge in graph['states'][start]['edges'].items()
             if edge['to'] == end and name != 'stop']
    if not names:
        raise ValueError('patrol needs a reverse graph edge')
    return sorted(names)[0]


def validate_route(route, graph, allowed_states):
    states, actions = route['states'], route['actions']
    if len(states) < 2 or len(actions) + 1 != len(states):
        raise ValueError('route needs one observed state per action plus the initial state')
    if not set(states) <= set(allowed_states):
        raise ValueError('route crosses the allowed capture/split')
    costs = []
    for a, action, b in zip(states[:-1], actions, states[1:], strict=True):
        edge = graph['states'][a]['edges'].get(action)
        if edge is None or edge['to'] != b or action == 'stop':
            raise ValueError('discontinuous route edge')
        cost = edge['cost_s']
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError('invalid nominal route cost')
        costs.append(cost)
    if 'costs_s' in route and route['costs_s'] != costs:
        raise ValueError('stored route costs differ from graph')
    if 'poses' in route and route['poses'] != [graph['states'][s]['pose'] for s in states]:
        raise ValueError('stored route poses differ from graph')
    return True


def build_routes(scene, split, graph, frames, *, length=24, stride=8):
    """Fixed DFS patrol + reverse, windowed without skipping any transitions.

    Reciprocal captured edges only. Isolated states cannot form a moving sequence
    and are listed by the caller as uncovered. No target-based route selection.
    These are nominal camera-graph walks, not certified collision-free flights.
    """
    if any(type(n) is not int or n < 1 for n in (length, stride)) or length < 2:
        raise ValueError('route length >= 2 and positive stride required')
    allowed = {f['state'] for f in frames if f['scene'] == scene and f['split'] == split}
    if not allowed <= graph['states'].keys():
        raise ValueError('frame absent from camera graph')
    for start in allowed:
        for _, edge in _edges(graph, start, allowed):
            _action(graph, edge['to'], start)
    remaining, routes, paths = set(allowed), [], set()
    component = 0
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        walk, stack = [start], [(start, iter(_edges(graph, start, allowed)))]
        while stack:
            state, choices = stack[-1]
            for _, edge in choices:
                end = edge['to']
                if end not in remaining:
                    continue
                remaining.remove(end)
                walk.append(end)
                stack.append((end, iter(_edges(graph, end, allowed))))
                break
            else:
                stack.pop()
                if stack:
                    walk.append(stack[-1][0])
        if len(walk) == 1:
            continue
        # Closed DFS walk: remove the duplicate endpoint before cycling.
        for direction, cycle in (('forward', walk[:-1]), ('reverse', list(reversed(walk))[:-1])):
            for offset in range(0, len(cycle), stride):
                states = [cycle[(offset + i) % len(cycle)] for i in range(length)]
                key = tuple(states)
                if key in paths:
                    continue
                paths.add(key)
                actions = [_action(graph, a, b) for a, b in zip(states[:-1], states[1:], strict=True)]
                route = dict(id=f'{scene}/{split}/patrol{component:02d}_{direction}_{offset:04d}',
                             scene=scene, split=split, states=states, actions=actions,
                             direction=direction, component=component,
                             poses=[graph['states'][s]['pose'] for s in states],
                             costs_s=[graph['states'][a]['edges'][name]['cost_s']
                                      for a, name in zip(states[:-1], actions, strict=True)],
                             policy='fixed heading-scan/neighbor patrol; no target labels used')
                validate_route(route, graph, allowed)
                routes.append(route)
        component += 1
    return routes
