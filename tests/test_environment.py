from types import SimpleNamespace
import copy
import json
from pathlib import Path
import tempfile
import unittest

from dinoexp.environment import select_nearby_objects
from capture_scene import capture_scene


class EnvironmentTests(unittest.TestCase):
    def test_hidden_poles_are_restored_if_vehicle_spawn_fails(self):
        cfg = json.loads(Path("configs/two_candidates_no_poles.json").read_text())
        anchor = SimpleNamespace(
            location=SimpleNamespace(x=10, y=20, z=1),
            rotation=SimpleNamespace(pitch=0, yaw=0, roll=0),
        )
        pole = SimpleNamespace(id=17, name="pole", transform=anchor)
        state = SimpleNamespace(settings=SimpleNamespace(synchronous_mode=False), visible=True)

        def visibility(ids, enable):
            state.visible = enable

        world = SimpleNamespace(
            get_settings=lambda: copy.copy(state.settings),
            apply_settings=lambda settings: setattr(state, "settings", settings),
            get_actors=lambda: SimpleNamespace(filter=lambda pattern: []),
            get_map=lambda: SimpleNamespace(name="TestTown", get_spawn_points=lambda: [anchor]),
            get_blueprint_library=lambda: SimpleNamespace(
                find=lambda name: SimpleNamespace(
                    id=name, has_attribute=lambda name: True, set_attribute=lambda name, value: None
                )
            ),
            get_weather=lambda: SimpleNamespace(),
            set_weather=lambda weather: None,
            get_environment_objects=lambda tag: [pole] if tag == "Poles" else [],
            enable_environment_objects=visibility,
            try_spawn_actor=lambda bp, transform: None,
        )
        carla = SimpleNamespace(
            CityObjectLabel=SimpleNamespace(Poles="Poles", TrafficLight="TrafficLight"),
            Location=lambda **kw: SimpleNamespace(**kw),
            Rotation=lambda **kw: SimpleNamespace(**kw),
            Transform=lambda p, r: SimpleNamespace(location=p, rotation=r),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "capture"
            with self.assertRaisesRegex(RuntimeError, "cannot spawn"):
                capture_scene(carla, world, ["0.10.0", "0.10.0"], cfg, root, False)
            run = json.loads((root / "run.json").read_text())
            self.assertEqual([r["id"] for r in run["hidden_environment_objects"]], [17])
            self.assertTrue(state.visible)
            self.assertFalse(state.settings.synchronous_mode)
            self.assertEqual(run["cleanup_errors"], [])

    def test_only_nearby_ids_are_selected_even_for_tall_poles(self):
        def obj(identifier, x, y, z):
            return SimpleNamespace(
                id=identifier,
                name="pole",
                transform=SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=z)),
            )

        result = select_nearby_objects([obj(1, 3, 4, 20), obj(2, 8, 0, 0)], (0, 0, 1), 5)
        self.assertEqual([r["id"] for r in result], [1])
        self.assertEqual(result[0]["distance_xy_m"], 5)

    def test_invalid_radius_is_not_allowed_to_hide_entire_map(self):
        with self.assertRaises(ValueError):
            select_nearby_objects([], (0, 0, 0), -1)


if __name__ == "__main__":
    unittest.main()
