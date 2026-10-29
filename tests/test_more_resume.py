import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from collect_match_photos import capture_plan, main
from dinoexp.more_photos import build_plan, select_anchors
from dinoexp.more_resume import check_resume_environment, prepare_resume


class ResumePhotoTests(unittest.TestCase):
    def setUp(self):
        base = json.loads((Path(__file__).resolve().parents[1]
                           / 'configs/two_candidates.json').read_text())
        points = [dict(index=i, pose=[i * 200, 0, 0, 0, 0, 0]) for i in range(9)]
        self.plan = build_plan(base, [f'vehicle.test.car{i}' for i in range(7)],
                               select_anchors(points))
        self.plan['map'] = 'TestTown'
        self.counter = 0

    def scene(self, root, case):
        path = root / case['split'] / case['name']
        path.mkdir(parents=True)
        (path / 'run.json').write_text(json.dumps(dict(status='complete',
            config=case['config'], map='TestTown', cleanup_errors=[])))
        records = [('reference', case['config']['target_identity'], 'reference', 'reference')]
        records += [(f'{c["identity"]}_{v["name"]}', c['identity'], 'query', v['name'])
                    for v in case['config']['views'] for c in case['config']['candidates']]
        rows = []
        for name, identity, role, view in records:
            self.counter += 1
            crop = name + '.png'
            Image.new('RGB', (28, 14), (80 + self.counter % 120,
                      100 + self.counter // 120, 120)).save(path / crop)
            rows.append(dict(sample_id=name, identity=identity, role=role, view=view,
                             status='captured', crop=crop))
        with (path / 'metadata.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def interrupted(self, root):
        root.mkdir()
        (root / '_configs').mkdir()
        for split in ('train', 'val', 'test'):
            (root / split).mkdir()
            cases = self.plan['cases'][:2] if split == 'train' else []
            (root / split / 'suite.json').write_text(json.dumps(dict(status='incomplete',
                conditions=[dict(name=c['name'], status='complete', config=c['config'],
                                 capture_group=c['capture_group']) for c in cases])))
        for case in self.plan['cases'][:2]:
            self.scene(root, case)
        failed = root / 'train' / self.plan['cases'][2]['name']
        failed.mkdir()
        (failed / 'run.json').write_text(json.dumps(dict(status='incomplete',
            error='std::exception', cleanup_errors=['settings failed'])))
        (root / 'collection.json').write_text(json.dumps(dict(status='incomplete',
            completed_scenes=2, plan=self.plan, error='old failure')))
        return failed

    def runner(self, root, calls):
        def run(command, check):
            destination = Path(command[command.index('--output') + 1])
            name = destination.name
            calls.append(name)
            case = next(c for c in self.plan['cases'] if c['name'] == name)
            self.scene(root, case)
        return run

    def test_small_pilot_reuses_successes_only_adds_balanced_val_test_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp, patch('builtins.print'):
            root = Path(tmp) / 'capture'
            failed = self.interrupted(root)
            successful = root / 'train' / self.plan['cases'][0]['name'] / 'metadata.csv'
            before = successful.read_bytes(), successful.stat().st_mtime_ns
            calls = []
            capture_plan(None, root, runner=self.runner(root, calls), resume=True, pilot=True)
            saved = json.loads((root / 'collection.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['status'], 'complete')
            self.assertEqual(saved['scope'], 'reduced_pilot')
            self.assertEqual(saved['completed_scenes'], 6)
            self.assertEqual(len(saved['original_plan']['cases']), 60)
            self.assertEqual([name.split('_')[0] for name in calls], ['val', 'val', 'test', 'test'])
            self.assertEqual(before, (successful.read_bytes(), successful.stat().st_mtime_ns))
            self.assertTrue(failed.is_dir())
            self.assertEqual(json.loads((failed / 'run.json').read_text(encoding='utf-8'))['status'], 'incomplete')
            self.assertEqual(saved['plan']['expected_reference_crops'], 6)
            for split in ('train', 'val', 'test'):
                manifest = json.loads((root / f'pairs_{split}.json').read_text(encoding='utf-8'))
                self.assertEqual(len(manifest['pairs']), 40)
                self.assertEqual(sum(r['label'] for r in manifest['pairs']), 20)
            self.assertEqual(len(saved['plan']['split_anchors']['val']), 1)

    def test_resume_after_another_failure_archives_partial_scene_and_skips_completed(self):
        with tempfile.TemporaryDirectory() as tmp, patch('builtins.print'):
            root = Path(tmp) / 'capture'
            self.interrupted(root)
            val = next(c for c in self.plan['cases'] if c['split'] == 'val')
            def fail(command, check):
                destination = Path(command[command.index('--output') + 1])
                destination.mkdir()
                (destination / 'run.json').write_text(json.dumps(dict(status='incomplete')))
                raise subprocess.CalledProcessError(2, command)
            with self.assertRaises(subprocess.CalledProcessError):
                capture_plan(None, root, runner=fail, resume=True, pilot=True)
            capture_plan(None, root, runner=self.runner(root, []), resume=True, pilot=True)
            archives = list((root / '_failed').iterdir())
            self.assertEqual(len(archives), 1)
            self.assertIn(val['name'], archives[0].name)
            self.assertTrue((archives[0] / 'run.json').is_file())
            self.assertEqual(json.loads((root / 'collection.json').read_text(encoding='utf-8'))['status'], 'complete')

    def test_resume_dry_run_is_offline_and_does_not_change_existing_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'capture'
            self.interrupted(root)
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            output = io.StringIO()
            with patch('sys.argv', ['collect_match_photos.py', '--resume', str(root),
                                   '--pilot', '--dry-run']), redirect_stdout(output):
                main()
            after = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            summary = json.loads(output.getvalue())
            self.assertEqual(summary['remaining_scenes'], 4)
            self.assertEqual(summary['completed_scenes'], 2)
            self.assertEqual(summary['scope'], 'reduced_pilot')

    def test_resume_rejects_changed_complete_scene_config_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'capture'
            self.interrupted(root)
            path = root / 'train' / self.plan['cases'][0]['name'] / 'run.json'
            run = json.loads(path.read_text())
            run['config']['target_identity'] = 'B'
            path.write_text(json.dumps(run))
            before = (root / 'collection.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'config'):
                capture_plan(None, root, resume=True, pilot=True)
            self.assertEqual(before, (root / 'collection.json').read_bytes())

    def test_resume_recovers_completed_scene_before_controller_checkpoint_and_keeps_its_partner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'capture'
            self.interrupted(root)
            val = next(c for c in self.plan['cases'] if c['split'] == 'val')
            self.scene(root, val)
            saved, completed, remaining = prepare_resume(root, pilot=True)
            self.assertEqual(len(completed), 3)
            self.assertNotIn(val['name'], [c['name'] for c in remaining])
            self.assertEqual([c['config']['target_identity'] for c in remaining
                              if c['split'] == 'val'], ['B'])
            self.assertEqual(len(saved['plan']['cases']), 6)

    def test_resume_environment_requires_original_map_layout_and_assets(self):
        self.plan['anchor_selection'] = dict(selected_spawn_points=[
            dict(index=0, pose=[0, 0, 0, 0, 0, 0])])
        models = [model for split in self.plan['split_models'].values() for model in split]
        valid = dict(map='TestTown', spawn_points=[dict(index=0, pose=[0, 0, 0, 0, 0, 0])])
        check_resume_environment(self.plan, valid, models)
        for catalog, eligible in (
                (dict(valid, map='OtherTown'), models),
                (dict(valid, spawn_points=[dict(index=0, pose=[3, 0, 0, 0, 0, 0])]), models),
                (valid, [])):
            with self.subTest(catalog=catalog, eligible=eligible), self.assertRaises(ValueError):
                check_resume_environment(self.plan, catalog, eligible)

    def test_malformed_excluded_partial_train_marker_is_left_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'capture'
            failed = self.interrupted(root)
            marker = failed / 'run.json'
            marker.write_text('{"config":', encoding='utf-8')
            _, completed, remaining = prepare_resume(root, pilot=True)
            self.assertEqual(len(completed), 2)
            self.assertEqual(len(remaining), 4)
            self.assertEqual(marker.read_text(encoding='utf-8'), '{"config":')

    def test_malformed_pending_val_marker_is_archived_and_recaptured(self):
        with tempfile.TemporaryDirectory() as tmp, patch('builtins.print'):
            root = Path(tmp) / 'capture'
            self.interrupted(root)
            case = next(c for c in self.plan['cases'] if c['split'] == 'val')
            partial = root / 'val' / case['name']
            partial.mkdir()
            (partial / 'run.json').write_text('{"config":', encoding='utf-8')
            capture_plan(None, root, runner=self.runner(root, []), resume=True, pilot=True)
            archives = list((root / '_failed').iterdir())
            self.assertEqual(len(archives), 1)
            self.assertEqual((archives[0] / 'run.json').read_text(encoding='utf-8'), '{"config":')
            self.assertEqual(json.loads((root / 'collection.json').read_text(encoding='utf-8'))['status'], 'complete')

    def test_malformed_previously_completed_marker_still_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'capture'
            self.interrupted(root)
            marker = root / 'train' / self.plan['cases'][0]['name'] / 'run.json'
            marker.write_text('{"config":', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'previously completed'):
                prepare_resume(root, pilot=True)

    def test_test_relocation_preserves_completed_train_val_and_persists_both_test_configs(self):
        with tempfile.TemporaryDirectory() as tmp, patch('builtins.print'):
            root = Path(tmp) / 'capture'
            self.interrupted(root)
            val_cases = [c for c in self.plan['cases'] if c['split'] == 'val'][:2]
            for case in val_cases:
                self.scene(root, case)
            before = {p: p.read_bytes() for s in ('train', 'val')
                      for p in (root / s).rglob('*.png')}
            anchor = self.plan['split_anchors']['test'][1]
            def capture(command, check):
                name = Path(command[command.index('--output') + 1]).name
                cfg = json.loads(Path(command[command.index('--config') + 1]).read_text(encoding='utf-8'))
                case = dict(next(c for c in self.plan['cases'] if c['name'] == name), config=cfg)
                self.scene(root, case)
            capture_plan(None, root, runner=capture, resume=True, pilot=True,
                         test_placement=dict(anchor_spawn_index=anchor, separation=12, attempts=[]))
            saved, completed, remaining = prepare_resume(root, pilot=True)
            self.assertEqual(saved['status'], 'complete')
            self.assertEqual(len(completed), 6)
            self.assertEqual(remaining, [])
            self.assertEqual(len(saved['placement_revisions']), 1)
            self.assertTrue(all(c['config']['anchor_spawn_index'] == anchor
                                and c['config']['separation'] == 12
                                for c in saved['plan']['cases'] if c['split'] == 'test'))
            self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_camera_adjustment_recaptures_pending_test_pair_with_consistent_persisted_views(self):
        with tempfile.TemporaryDirectory() as tmp, patch('builtins.print'):
            root = Path(tmp) / 'capture'
            self.interrupted(root)
            for case in [c for c in self.plan['cases'] if c['split'] == 'val'][:2]:
                self.scene(root, case)
            first_test = next(c for c in self.plan['cases'] if c['split'] == 'test')
            partial = root / 'test' / first_test['name']
            partial.mkdir()
            (partial / 'run.json').write_text(json.dumps(dict(status='incomplete',
                config=first_test['config'], error='near_black_image')))
            before = {p: p.read_bytes() for s in ('train', 'val')
                      for p in (root / s).rglob('*.png')}
            def capture(command, check):
                name = Path(command[command.index('--output') + 1]).name
                cfg = json.loads(Path(command[command.index('--config') + 1]).read_text(encoding='utf-8'))
                self.scene(root, dict(next(c for c in self.plan['cases'] if c['name'] == name), config=cfg))
            capture_plan(None, root, runner=capture, resume=True, pilot=True, test_camera_radius=10)
            saved, completed, remaining = prepare_resume(root, pilot=True)
            self.assertEqual(saved['status'], 'complete')
            self.assertEqual(len(completed), 6)
            self.assertEqual(remaining, [])
            views = [c['config']['views'] for c in saved['plan']['cases'] if c['split'] == 'test']
            self.assertEqual(views[0], views[1])
            self.assertTrue(all(v['name'].endswith('_raised') for v in views[0]))
            self.assertEqual(len(saved['camera_revisions']), 1)
            self.assertEqual(len(list((root / '_failed').iterdir())), 1)
            self.assertEqual(before, {p: p.read_bytes() for p in before})


if __name__ == '__main__':
    unittest.main()
