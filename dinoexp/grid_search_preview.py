"""Static visual QA for the original frames and generated offline sequences."""

import html
import json

from PIL import Image

from .core import checked_path, prepare_image
from .grid_search_validation import read_json
from .search_episodes import json_rows


def page(title, content):
    return ('<!doctype html><meta charset="utf-8"><title>' + html.escape(title) + '</title>'
            '<style>body{font:16px sans-serif;margin:24px}.images{display:grid;'
            'grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}img{width:100%;}'
            'article{border-top:1px solid #aaa;margin:16px 0}td{padding:8px}</style>'
            '<h1>' + html.escape(title) + '</h1>' + content)


def images(paths):
    return '<div class="images">' + ''.join(
        f'<a href="{html.escape(path, quote=True)}"><img loading="lazy" alt="{html.escape(label)}" '
        f'src="{html.escape(path, quote=True)}">{html.escape(label)}</a>' for path, label in paths) + '</div>'


def publish_preview(root, manifest, result):
    preview = root / 'preview'
    preview.mkdir()
    index = []
    for case in manifest['cases']:
        sid = case['scene_id']
        directory = root / case['path']
        scene = read_json(directory / 'scene.json')
        graph = read_json(directory / 'graph.json')
        overview = ['<p><a href="index.html">Dataset</a></p>']
        for uid, ref in scene['references'].items():
            overview.append(f'<h2>{html.escape(uid)} independent reference</h2>')
            overview.append(images([('../' + ref[key], label) for key, label in
                                    [('full', 'Raw reference'), ('object', 'Reference crop'),
                                     ('mask', 'Reference mask')]]))
        route_index = read_json(directory / 'episodes' / 'index.json')
        overview.append('<h2>Ordered offline sequences</h2><ul>')
        for path in route_index['paths']:
            route = read_json(checked_path(root, path) / 'episode.json')
            overview.append(f'<li><a href="{route["episode_id"]}.html">'
                            f'{route["episode_id"]} {route["template"]} {html.escape(route["target_uid"])}</a></li>')
        overview.append('</ul><h2>Grid points, row = +X and column = +Y</h2><table>')
        size = manifest['config']['grid']['size']
        for row in range(size):
            overview.append('<tr>')
            for col in range(size):
                point = f'r{row:02d}c{col:02d}'
                label = (f'<a href="#{point}">{point}</a>' if graph['points'][point]['capture_eligible']
                         else f'{point} excluded')
                overview.append(f'<td>{label}</td>')
            overview.append('</tr>')
        overview.append('</table><p>Raw RGB, audit overlay and full-frame 518 px preprocessing preview.</p>')
        frame_cards = {}
        for row in json_rows(directory / 'frames.jsonl'):
            state = row['state_id']
            stem = f'{sid}_{state}'
            with Image.open(checked_path(root, row['image_path'])) as image:
                rgb = image.convert('RGB')
            overlay = rgb.copy()
            for color, label in zip(((0, 225, 85), (40, 130, 255)), row['objects'].values(), strict=True):
                if label['mask']:
                    with Image.open(checked_path(root, label['mask']['path'])) as mask:
                        overlay = Image.composite(Image.new('RGB', rgb.size, color), overlay,
                                                  mask.point(lambda value: round(value * .40)))
            overlay.thumbnail((640, 480))
            overlay.save(preview / f'{stem}_overlay.jpg', quality=91)
            prepare_image(rgb, 518).save(preview / f'{stem}_518.png')
            details = ', '.join(f'{uid}: {q["quality"]["recognition_quality"] if q["quality"]["visibility_present"] else "absent"}, '
                                f'{q["quality"]["visible_pixels"]} px'
                                for uid, q in row['objects'].items())
            frame_cards[state] = (f'<p>{html.escape(details)} | source frame {row["sensor_frame"]} '
                                  f'at {row["sensor_timestamp"]:.3f} s</p>' +
                                  images([('../' + row['image_path'], 'Raw RGB'),
                                          (f'{stem}_overlay.jpg', 'Overlay A green / B blue'),
                                          (f'{stem}_518.png', '518 px full frame')]))
            if row['direction_id'] == 0:
                point = row['point_id']
                overview.append(f'<h2 id="{point}">{point}</h2>')
            yaw = graph['states'][state]['yaw_deg']
            overview.append(f'<article><h3>{state} yaw {yaw}°</h3>{frame_cards[state]}</article>')
        (preview / f'{sid}.html').write_text(page(sid, ''.join(overview)), encoding='utf-8')
        index.append(f'<li><a href="{sid}.html">{sid} ({case["split"]})</a></li>')
        for path in route_index['paths']:
            route = read_json(checked_path(root, path) / 'episode.json')
            ref = scene['references'][route['target_uid']]
            cards = [f'<p><a href="{sid}.html">Back to source grid</a></p>',
                     images([('../' + ref['full'], 'Independent reference'),
                             ('../' + ref['object'], 'Reference crop'), ('../' + ref['mask'], 'Mask')])]
            for row in json_rows(checked_path(root, path) / 'frames.jsonl'):
                state = row['state_id']
                action = row['action_from_previous']
                name = action['name'] if action else 'start'
                if name == 'hold_capture':
                    name += ' (cached repeat)'
                cards.append(f'<article><h3>Step {row["step_index"]} {state} | {name} | '
                             f'offline {row["time_s"]:.2f} s | target {row["quality"]}</h3>'
                             f'{frame_cards[state]}</article>')
            (preview / f'{route["episode_id"]}.html').write_text(
                page(route['episode_id'], ''.join(cards)), encoding='utf-8')
    (preview / 'index.html').write_text(
        page('CARLA search grids', '<p>Each observation is a separate synchronized camera capture.</p>'
             '<ul>' + ''.join(index) + '</ul>'), encoding='utf-8')
    (root / 'validation_report.md').write_text(
        '# Grid capture validation\n\nStructural and pixel checks passed. Visual review is pending.\n\n'
        'Inspect the [grid and sequence previews](preview/index.html). A usable mask is a quality '
        'screen; recorded offline transitions have unknown flight feasibility.\n\n'
        '```json\n' + json.dumps(result, ensure_ascii=False, indent=2) + '\n```\n', encoding='utf-8')
