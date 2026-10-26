from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

import run_local_probe


class RunLocalProbeTests(unittest.TestCase):
    def test_failure_stops_before_later_stages(self):
        runner = Mock(side_effect=subprocess.CalledProcessError(2, ['cache']))
        with self.assertRaises(subprocess.CalledProcessError):
            run_local_probe.execute_plan([['cache'], ['train'], ['evaluate']], runner)
        self.assertEqual(runner.call_count, 1)

    def test_dry_run_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'new'
            repo = Path(tmp) / 'repo'
            repo.mkdir()
            (repo / 'hubconf.py').write_text('# fixture')
            backbone = Path(tmp) / 'model.pth'
            backbone.write_bytes(b'fixture')
            data = Path(tmp) / 'pilot'
            data.mkdir()
            args = run_local_probe.parse_args(['--data', str(Path(tmp) / 'pilot'),
                '--output', str(output), '--model-repo', str(repo), '--backbone', str(backbone),
                '--dry-run'])
            plan = run_local_probe.make_plan(args)
            self.assertEqual(len(plan), 4)
            self.assertIn('--allow-train-only', plan[2])
            self.assertFalse(output.exists())

    def test_missing_backbone_paths_fail_before_pair_output_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'new'
            args = run_local_probe.parse_args(['--data', str(Path(tmp)),
                '--output', str(output), '--model-repo', str(Path(tmp) / 'missing'),
                '--backbone', str(Path(tmp) / 'missing.pth')])
            with self.assertRaisesRegex(ValueError, 'hubconf.py'):
                run_local_probe.make_plan(args)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
