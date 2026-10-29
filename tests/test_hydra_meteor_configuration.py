"""The Hydra + METEOR example (deployment B): its FM entry, and its geometry driven
on the simulator with the FM served locally instead of from the METEOR PC."""

import copy
import os

import pytest
import yaml

import fibsem.config as cfg
from fibsem import utils
from fibsem.codec.upgrades.configuration import upgrade
from fibsem.structures.system import FM_DRIVER_REMOTE, MicroscopeSettings

HYDRA_METEOR = os.path.join(
    cfg.SHIPPED_CONFIG_PATH, "tfs-hydra-meteor-configuration.yaml"
)


def _load() -> dict:
    with open(HYDRA_METEOR) as f:
        return yaml.safe_load(f)


def _fm_entry(data: dict) -> dict:
    return next(d for d in data["hardware"]["devices"] if d["name"] == "fm")


def test_the_fm_is_served_from_the_meteor_pc():
    system = MicroscopeSettings.from_dict(upgrade(_load())).system
    assert system.info.manufacturer == "ThermoFisher"
    assert system.fm.enabled is True
    assert system.fm.driver == FM_DRIVER_REMOTE
    assert (system.fm.address, system.fm.port) == ("127.0.0.1", 8765)


def test_the_fm_sits_at_calib_dx_dy_and_images_in_the_fib_pose():
    fm = MicroscopeSettings.from_dict(upgrade(_load())).system.stage.devices["FM"]
    assert (fm.origin.x, fm.origin.y) == (50.6252e-3, 4.9832e-3)
    assert fm.available_orientations == ["FIB"]
    # The window must cover y too, or the FM is lost off-centre in y.
    assert fm.range.x and fm.range.y


@pytest.fixture
def microscope(tmp_path):
    """The example on the simulator: Demo beams and a local FM in place of
    AutoScript and the remote one, with the nominal origins saved as calibrated."""
    data = copy.deepcopy(_load())
    data["info"]["manufacturer"] = "Demo"
    entry = _fm_entry(data)
    for key in ("driver", "address", "port"):
        del entry[key]
    origin = entry["origin"]
    data["calibration"]["devices"] = {
        "fibsem": {"origin": {"x": 0.0, "y": 0.0}},
        "fm": {"origin": dict(origin)},
        "holder": None,
    }
    data["sim"] = {"is_compustage": False, "has_fm": True}
    path = tmp_path / "hydra-meteor-sim.yaml"
    path.write_text(yaml.safe_dump(data))
    microscope, _ = utils.setup_session(config_path=str(path))
    microscope.move_to_orientation("SEM")
    return microscope


def test_the_fib_pose_is_the_tfs3_fm_posture(microscope):
    fib = microscope.get_orientation("FIB")
    assert fib.r == pytest.approx(3.14159265, abs=1e-6)
    assert fib.t == pytest.approx(0.29670597, abs=1e-6)  # 17 degrees


def test_sem_to_fm_and_back(microscope):
    start = microscope.get_stage_position()
    movement = microscope.stage_movement

    movement.move_to_device("FM")
    at_fm = microscope.get_stage_position()
    assert at_fm.x - start.x == pytest.approx(50.6252e-3, abs=1e-9)
    assert at_fm.y - start.y == pytest.approx(4.9832e-3, abs=1e-9)
    assert movement.current_device() == "FM"

    # Off centre in y, still at the FM.
    microscope.fm.objective.retract()
    microscope.move_stage_relative(type(at_fm)(y=2.0e-3))
    assert movement.current_device() == "FM"

    movement.move_to_device("FIBSEM")
    assert movement.current_device() == "FIBSEM"
