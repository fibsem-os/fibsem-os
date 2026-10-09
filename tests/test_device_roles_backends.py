"""Every backend fills the roles its configuration binds (``roles:`` on an entry).

The Demo's binding is tested in ``tests/test_scan_generator.py``; here a Tescan, over its
fake SharkSEM connection, and a Thermo, over the fake AutoScript SDK, bind the electron
beam's scanner to the Demo's simulated scan generator. The Odemis case is in
``tests/test_odemis_devices.py``, beside its fake client.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import fibsem
import fibsem.config as cfg
from fibsem import utils
from fibsem.devices.entries import RoleBindingError
from fibsem.structures import BeamType, DeviceEntry, ImageSettings
from tests.fixtures.tescan_sdk import connect

SCAN_GENERATOR = {"name": "scan_generator", "type": "scan_generator", "driver": "demo"}
FIXTURES = Path(__file__).parent / "fixtures"


def _tescan(monkeypatch, roles, *entries):
    system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    system.other_devices = [DeviceEntry.from_dict(e) for e in entries]
    system.device_roles = {"electron": roles}
    microscope, _ = connect(monkeypatch, system)
    return microscope


def test_a_tescan_beam_images_through_the_bound_scanner(monkeypatch):
    microscope = _tescan(monkeypatch, {"scanner": "scan_generator"}, SCAN_GENERATOR)

    scanner = microscope.devices["scan_generator"]
    assert microscope.beams[BeamType.ELECTRON].scanner is scanner
    assert "scanner" not in microscope.beams[BeamType.ION].roles

    settings = ImageSettings(
        resolution=(64, 48), dwell_time=1e-7, hfw=80e-6, beam_type=BeamType.ELECTRON
    )
    image = microscope.acquire_image(settings)
    assert scanner.sim_frames == 1
    assert image.data.shape == (48, 64)


def test_a_tescan_binding_to_no_device_fails_connect(monkeypatch):
    with pytest.raises(RoleBindingError, match="no device is named 'nothing'"):
        _tescan(monkeypatch, {"scanner": "nothing"})


THERMO = """
import json, sys
sys.path.insert(0, sys.argv[1])
import autoscript_beam_parity as B
from fibsem.structures import BeamType, DeviceEntry, ImageSettings

microscope = B.routed(False)
microscope._build_stage()
microscope.system.other_devices = [DeviceEntry.from_dict(json.loads(sys.argv[2]))]
microscope.system.device_roles = {"electron": {"scanner": "scan_generator"}}
microscope._build_parts()
microscope._bind_device_roles()
scanner = microscope.devices["scan_generator"]
settings = ImageSettings(
    resolution=(64, 48), dwell_time=1e-7, hfw=80e-6, beam_type=BeamType.ELECTRON
)
image = microscope.acquire_image(settings)
print(json.dumps({
    "bound": microscope.beams[BeamType.ELECTRON].scanner is scanner,
    "ion": list(microscope.beams[BeamType.ION].roles),
    "frames": scanner.sim_frames,
    "shape": list(image.data.shape),
}))
"""


def test_a_thermo_beam_images_through_the_bound_scanner():
    # In its own process: the fake SDK replaces the AutoScript modules for good.
    result = subprocess.run(
        [sys.executable, "-c", THERMO, str(FIXTURES), json.dumps(SCAN_GENERATOR)],
        env=dict(os.environ, FIBSEM_SIM_NO_DELAY="1"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    facts = json.loads(result.stdout.strip().splitlines()[-1])
    assert facts == {"bound": True, "ion": [], "frames": 1, "shape": [48, 64]}


BUILDS = (
    "_build_beams",
    "_build_stage",
    "_build_parts",
    "_build_devices",
    "build_device_entries",
)


@pytest.mark.parametrize("backend", ["autoscript", "tescan", "odemis"])
def test_roles_are_bound_after_every_build_step(backend):
    """A binding can name any device, so it is made once the connect step that
    calls it has built them all."""
    source = (
        Path(fibsem.__file__).parent / "drivers" / backend / "microscope.py"
    ).read_text(encoding="utf-8")

    def calls(node, names):
        return [
            call.lineno
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and getattr(call.func, "attr", getattr(call.func, "id", None)) in names
        ]

    steps = [
        function
        for function in ast.walk(ast.parse(source))
        if isinstance(function, ast.FunctionDef)
        and calls(function, ("_bind_device_roles",))
    ]
    assert len(steps) == 1
    (bind,) = calls(steps[0], ("_bind_device_roles",))
    assert calls(steps[0], BUILDS)
    assert all(line < bind for line in calls(steps[0], BUILDS))
