from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_probe import make_plan, run_plan


class ProbeRunnerTests(unittest.TestCase):
    def test_stage_failure_stops_later_stages(self):
        plan = [{'stage': 'capture', 'command': ['capture']},
                {'stage': 'segment', 'command': ['segment']},
                {'stage': 'evidence', 'command': ['evidence']}]
        executed = []

        def fail(command, **kwargs):
            executed.append(command[0])
            raise subprocess.CalledProcessError(2, command)

        with self.assertRaisesRegex(RuntimeError, 'capture'):
            run_plan(plan, executor=fail)
        self.assertEqual(executed, ['capture'])

    def test_stage_order_and_conda_environment_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / 'parts.pt'
            weights.write_bytes(b'fixture-not-a-model')
            plan = make_plan(root, 'probe04', weights=weights)
            self.assertEqual([p['stage'] for p in plan], ['capture', 'segment', 'evidence'])
            self.assertEqual(plan[0]['command'][:5],
                             ['conda', 'run', '--no-capture-output', '-n', 'carla10'])
            self.assertEqual(plan[1]['command'][4], 'dino-probe')
            self.assertIn(str(root / 'data/probe04'), plan[1]['command'])
            executed = []
            run_plan(plan, executor=lambda cmd, **kw: executed.append(cmd))
            self.assertEqual(executed, [p['command'] for p in plan])

    def test_output_collision_is_detected_before_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / 'parts.pt'
            weights.write_bytes(b'fixture')
            (root / 'out/probe04-evidence').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'exists'):
                make_plan(root, 'probe04', weights=weights)
            self.assertFalse((root / 'data').exists())
            with self.assertRaisesRegex(ValueError, 'name'):
                make_plan(root, '../outside', weights=weights)

    def test_dry_run_does_not_need_conda_carla_weights_or_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = subprocess.run([sys.executable, 'run_probe.py', '--root', str(root),
                                     '--name', 'dry_probe', '--dry-run'],
                                    check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('2048', result.stdout)
            self.assertEqual(list(root.iterdir()), [])


class SuitePreflightTests(unittest.TestCase):
    def test_refused_server_does_not_leave_capture_directory(self):
        import capture_suite
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'capture'
            with patch.object(sys, 'argv', ['capture_suite.py', '--output', str(output)]), \
                    patch.dict(sys.modules, {'carla': object()}), \
                    patch('capture.connect', side_effect=RuntimeError('Connection refused')), \
                    patch('capture_suite.subprocess.run',
                          side_effect=subprocess.CalledProcessError(2, ['capture'])):
                with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                    capture_suite.main()
            self.assertFalse(output.exists())

    def test_connection_check_only_does_not_capture_or_create_output(self):
        import capture_suite
        from types import SimpleNamespace
        world = SimpleNamespace(get_map=lambda: SimpleNamespace(name='TestTown'))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'capture'
            with patch.object(sys, 'argv', ['capture_suite.py', '--check-connection',
                                           '--output', str(output)]), \
                    patch.dict(sys.modules, {'carla': object()}), \
                    patch('capture.connect', return_value=(None, world, ['0.10.0', '0.10.0'])), \
                    patch('capture_suite.subprocess.run') as capture:
                capture_suite.main()
            capture.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
