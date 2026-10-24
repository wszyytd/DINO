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
            args = run_local_probe.parse_args(['--data', str(Path(tmp) / 'pilot'),
                '--output', str(output), '--model-repo', 'repo', '--backbone', 'model.pth',
                '--dry-run'])
            plan = run_local_probe.make_plan(args)
            self.assertEqual(len(plan), 4)
            self.assertIn('--allow-train-only', plan[2])
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
