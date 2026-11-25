"""Exercise the server launcher with capture and training isolated from CARLA and CUDA."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize(("scenario", "expected"), [
    ("separate", "captured in carla10"),
    ("combined", "captured in current"),
    ("override", "captured in custom"),
    ("cached", "audited in current"),
    ("offline", "prepared in current"),
])
def test_capture_environment_and_return_to_training(tmp_path, scenario, expected):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("server launcher requires Bash")
    shutil.copy(Path(__file__).resolve().parents[1] / "run_experiment8.sh", tmp_path)
    if scenario == "cached":
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "plan.json").touch()
    env = dict(os.environ, SCENARIO=scenario, CLEAR_MEMORY_DATA="data")
    env.pop("CARLA_PYTHON", None)
    env.pop("MEMORY_PLAN", None)
    if scenario == "override":
        env["CARLA_PYTHON"] = "custom-python"
    harness = r'''
python() {
    if [[ $1 == -c ]]; then [[ $SCENARIO == combined ]]; return; fi
    case "$*" in
        *"prepare_experiment8_offline.py "*) echo "prepared in current";;
        *"capture_clear_memory.py capture "*)
            if [[ $SCENARIO != combined ]]; then echo "No module named 'carla'" >&2; return 1; fi
            echo "captured in current";;
        *"capture_clear_memory.py audit "*) echo "audited in current";;
        *"run_memory_replay.py train "*)
            mode=foreground
            if [[ $SCENARIO == offline ]]; then mode=bbox; fi
            [[ " $* " == *" --crop-mode $mode "* ]] || return 2
            echo "trained in current"; exit 0;;
        "-m unittest "*) return 0;;
        *) return 2;;
    esac
}
conda() {
    [[ $* == "run --no-capture-output -n carla10 python -u capture_clear_memory.py capture "* ]] || return 2
    echo "captured in carla10"
}
custom-python() {
    [[ $* == "-u capture_clear_memory.py capture "* ]] || return 2
    echo "captured in custom"
}
if [[ $SCENARIO == offline ]]; then set -- --offline 1 7; fi
source ./run_experiment8.sh
'''
    result = subprocess.run([bash, "-c", harness, "./run_experiment8.sh"], cwd=tmp_path,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert [line for line in result.stdout.splitlines()
            if line.startswith(("captured in ", "audited in ", "prepared in ", "trained in "))] == [
                expected, "trained in current"]
