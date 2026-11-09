import json
from pathlib import Path
import subprocess
import sys

import pytest

from dinoexp import dense_capture


def test_grid_views_share_each_position_and_height():
    cfg = json.loads(Path('configs/two_candidates.json').read_text())
    cfg['grid_capture'] = {'size': 3, 'spacing_m': 8, 'height_m': 8, 'oblique_pitch_deg': -45}
    cfg = dense_capture.validate_dense_config(cfg)

    assert len(cfg['views']) == 45
    assert cfg['views'][0]['name'] == 'r01_c01_down'
    center = (10, 20, 2)
    anchor = (10, 20, 1, 0, 90, 0)
    views = [view for view in cfg['views'] if view['grid_cell'] == [0, 0]]
    poses = [dense_capture.grid_camera_pose(center, anchor, view) for view in views]

    assert len(poses) == 5
    assert all(pose[:3] == pytest.approx((18, 28, 9)) for pose in poses)
    assert sorted(pose[3] for pose in poses) == [-90, -45, -45, -45, -45]
    assert {pose[4] % 360 for pose in poses if pose[3] == -45} == {0, 90, 180, 270}


def test_grid_cli_plans_45_images_without_synthetic_absent_frame(tmp_path):
    output = tmp_path / 'capture'
    result = subprocess.run(
        [sys.executable, 'capture_dense_scene.py', '--config', 'configs/grid_pilot.json',
         '--output', str(output), '--dry-run'], capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout)
    assert preview['query_frames'] == 45
    assert preview['null_frames'] == 0
    assert not output.exists()
