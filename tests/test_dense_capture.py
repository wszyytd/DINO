from contextlib import contextmanager
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
import numpy as np
from dinoexp.dense_capture import decode_instance_ids, discover_instance_ids, managed_capture, capture_dense_scene


class DenseCaptureTests(unittest.TestCase):
    def test_instance_ids_use_g_low_b_high_and_are_not_python_actor_ids(self):
        rgb=np.array([[[14,2,1],[15,255,3]]],np.uint8)
        self.assertEqual(decode_instance_ids(rgb).tolist(),[[258,1023]])

    def test_remove_restore_probe_identifies_only_disappearing_reappearing_instance(self):
        original=np.array([[0,258,258],[500,500,0]],np.uint16)
        hidden=np.array([[0,0,0],[500,500,0]],np.uint16)
        self.assertEqual(discover_instance_ids(original,hidden,original,min_pixels=2),[258])
        with self.assertRaisesRegex(ValueError,'instance|probe'):
            discover_instance_ids(original,original,original,min_pixels=2)
        with self.assertRaisesRegex(ValueError,'restore|instance|probe'):
            discover_instance_ids(original,hidden,hidden,min_pixels=2)

    def test_capture_exception_cleans_only_owned_resources_and_restores_world(self):
        settings=SimpleNamespace(synchronous_mode=False)
        world=SimpleNamespace(settings=copy.copy(settings),weather='initial',
                              get_settings=lambda:copy.copy(settings),get_weather=lambda:'initial')
        world.apply_settings=lambda value:setattr(world,'settings',value)
        world.set_weather=lambda value:setattr(world,'weather',value)
        calls=[]
        actor=SimpleNamespace(id=12,type_id='sensor.camera.rgb',stop=lambda:calls.append('stop'),
                              destroy=lambda:calls.append('destroy') or True)
        run={'cleanup_errors':[]}
        with self.assertRaisesRegex(RuntimeError,'photography failed'):
            with managed_capture(world,[actor],run):
                world.settings.synchronous_mode=True;world.weather='changed'
                raise RuntimeError('photography failed')
        self.assertEqual(calls,['stop','destroy'])
        self.assertFalse(world.settings.synchronous_mode)
        self.assertEqual(world.weather,'initial')
        self.assertEqual(run['cleanup_errors'],[])

    def test_unsafe_view_filename_is_rejected_before_world_access_or_output_creation(self):
        cfg=json.loads(Path('configs/two_candidates.json').read_text())
        cfg['views'][0]['name']='../escape'
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'capture'
            with self.assertRaisesRegex(ValueError,'view'):
                capture_dense_scene(None,None,[],cfg,output)
            self.assertFalse(output.exists())

    def test_invalid_sensor_warmup_is_rejected_before_world_access(self):
        cfg=json.loads(Path('configs/two_candidates.json').read_text());cfg['warmup_ticks']=0
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'warmup'):
                capture_dense_scene(None,None,[],cfg,Path(tmp)/'capture')

    def test_single_scene_cli_can_preview_two_views_without_carla_or_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'untouched'
            result=subprocess.run([sys.executable,'capture_dense_scene.py','--config','configs/two_candidates.json',
                '--output',str(target),'--views','overview','descend','--dry-run'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            preview=json.loads(result.stdout)
            self.assertEqual(preview['query_frames'],2)
            self.assertFalse(preview['carla_connected'])
            self.assertFalse(target.exists())

    def test_intervention_moves_actor_above_camera_instead_of_unreal_kill_z(self):
        from dinoexp.dense_capture import occlusion_probe_pose
        original=[10,20,.1,0,90,0]
        result=occlusion_probe_pose(original,[10,30,70,-80,90,0])
        self.assertEqual(result[:2],[10,20])
        self.assertGreater(result[2],70)
        self.assertEqual(original,[10,20,.1,0,90,0])

    def test_sensor_destroy_is_attempted_even_when_stop_raises(self):
        world=SimpleNamespace(get_settings=lambda:'settings',get_weather=lambda:'weather',
            apply_settings=lambda value:None,set_weather=lambda value:None)
        calls=[]
        def stop():raise RuntimeError('listener already failed')
        actor=SimpleNamespace(id=9,type_id='sensor.camera.rgb',stop=stop,
                              destroy=lambda:calls.append('destroy') or True)
        vehicle=SimpleNamespace(id=8,type_id='vehicle.test',destroy=lambda:calls.append('vehicle') or True)
        run={'cleanup_errors':[]}
        with managed_capture(world,[vehicle,actor],run):pass
        self.assertEqual(calls,['destroy','vehicle'])
        self.assertTrue(run['cleanup_errors'])
