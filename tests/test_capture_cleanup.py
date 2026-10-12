"""Exercise exception cleanup at the CARLA boundary without a simulator."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from capture import capture


class Blueprint:
    id = "vehicle.test"

    def has_attribute(self, name):
        return name != "number_of_wheels"

    def set_attribute(self, name, value):
        pass


class FailingWorld:
    def __init__(self):
        self.settings = SimpleNamespace(
            synchronous_mode=False, fixed_delta_seconds=None, no_rendering_mode=True
        )
        self.weather = SimpleNamespace(sun_altitude_angle=25)

    def get_settings(self):
        return copy.copy(self.settings)

    def apply_settings(self, settings):
        self.settings = settings

    def get_weather(self):
        return copy.copy(self.weather)

    def set_weather(self, weather):
        self.weather = weather

    def get_actors(self):
        return SimpleNamespace(filter=lambda pattern: [])

    def get_map(self):
        return SimpleNamespace(name="TestTown", get_spawn_points=lambda: [object()])

    def get_blueprint_library(self):
        return SimpleNamespace(find=lambda name: Blueprint())

    def spawn_actor(self, blueprint, transform):
        raise RuntimeError("sensor spawn failed")


class CaptureCleanupTests(unittest.TestCase):
    def test_native_lookup_error_identifies_the_blueprint(self):
        cfg = json.loads(Path("configs/experiment.json").read_text())
        world = FailingWorld()

        def failed_find(name):
            raise RuntimeError("std::exception")

        world.get_blueprint_library = lambda: SimpleNamespace(find=failed_find)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "capture"
            with self.assertRaisesRegex(RuntimeError, "vehicle.lincoln.mkz"):
                capture(None, world, ["0.10.0", "0.10.0"], cfg, output, False)
            self.assertFalse(output.exists())

    def test_sensor_failure_restores_world_and_records_incomplete_run(self):
        cfg = json.loads(Path("configs/experiment.json").read_text())
        world = FailingWorld()
        carla = SimpleNamespace(Location=lambda **kw: kw, Transform=lambda *args: args)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "capture"
            with self.assertRaisesRegex(RuntimeError, "sensor spawn failed"):
                capture(carla, world, ["0.10.0", "0.10.0"], cfg, output, False)
            self.assertFalse(world.settings.synchronous_mode)
            self.assertIsNone(world.settings.fixed_delta_seconds)
            self.assertTrue(world.settings.no_rendering_mode)
            self.assertEqual(world.weather.sun_altitude_angle, 25)
            run = json.loads((output / "run.json").read_text())
            self.assertEqual(run["status"], "incomplete")
            self.assertEqual(run["cleanup_errors"], [])


if __name__ == "__main__":
    unittest.main()
