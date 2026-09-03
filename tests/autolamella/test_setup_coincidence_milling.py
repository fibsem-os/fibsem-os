"""SetupCoincidenceMillingTask: the per-site setup step for a queued coincidence
mill (FIB-909). Headless here -- the viewer hand-off is FIB-911 -- so what is
under test is the microscope choreography and the record it leaves.
"""

import os

import pytest
import yaml

import fibsem.config as fconfig
from fibsem import utils
from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.workflows.tasks import get_tasks
from fibsem.applications.autolamella.workflows.tasks.acquire_fluorescence import (
    AcquireFluorescenceImageConfig,
)
from fibsem.applications.autolamella.workflows.tasks.setup_coincidence_milling import (
    COINCIDENCE_SETUP_REFERENCE_FILENAME,
    SetupCoincidenceMillingTask,
    SetupCoincidenceMillingTaskConfig,
)
from fibsem.fm.structures import ChannelSettings
from fibsem.structures import FibsemRectangle, Point

SIM_ARCTIS_CONFIG_PATH = os.path.join(
    fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml"
)


@pytest.fixture
def microscope(tmp_path):
    """Arctis sim with a fluorescence module; sample scene off (noise frames)."""
    with open(SIM_ARCTIS_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config.setdefault("sim", {}).setdefault("sample", {})["enabled"] = False
    path = tmp_path / "sim-arctis-configuration.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=str(path))
    assert microscope.fm is not None
    yield microscope
    microscope.disconnect()


def _lamella(microscope, tmp_path, fm_objective: float = 2.45e-3) -> Lamella:
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    fm_pose = microscope.get_microscope_state()
    fm_pose.objective_position = fm_objective
    lamella.fluorescence_pose = fm_pose
    lamella.task_config["Acquire Fluorescence Image"] = AcquireFluorescenceImageConfig(
        task_name="Acquire Fluorescence Image",
        channel_settings=[
            ChannelSettings(name="Reflection Channel", excitation_wavelength=550),
            ChannelSettings(
                name="Red Channel",
                excitation_wavelength=550,
                emission_wavelength="Fluorescence",
            ),
        ],
    )
    return lamella


def _task(microscope, lamella, **kwargs) -> SetupCoincidenceMillingTask:
    config = SetupCoincidenceMillingTaskConfig(
        task_name="Setup Coincidence Milling", **kwargs
    )
    lamella.task_config[config.task_name] = config
    return SetupCoincidenceMillingTask(
        microscope=microscope, config=config, lamella=lamella
    )


# ---------------------------------------------------------------------------
# registration + serialisation
# ---------------------------------------------------------------------------


def test_registered_as_a_builtin_task():
    assert get_tasks()["SETUP_COINCIDENCE_MILLING"] is SetupCoincidenceMillingTask


def test_config_roundtrip_keeps_the_per_site_record():
    config = SetupCoincidenceMillingTaskConfig(
        task_name="Setup Coincidence Milling",
        channel_name="Red Channel",
        intensity_drop_fraction=0.3,
        objective_position=2.418e-3,
        fm_roi=FibsemRectangle(0.44, 0.38, 0.16, 0.24),
        pattern_offset=Point(1.2e-6, -0.4e-6),
    )
    data = yaml.safe_load(yaml.safe_dump(config.to_dict()))
    loaded = SetupCoincidenceMillingTaskConfig.from_dict(data)
    assert loaded.objective_position == pytest.approx(2.418e-3)
    assert loaded.fm_roi == FibsemRectangle(0.44, 0.38, 0.16, 0.24)
    assert loaded.pattern_offset.x == pytest.approx(1.2e-6)
    assert loaded.pattern_offset.y == pytest.approx(-0.4e-6)
    assert loaded.intensity_drop_fraction == pytest.approx(0.3)
    assert loaded.is_set_up
    # the non-scalar fields are not in the generic parameters block
    assert "fm_roi" not in data["parameters"]
    assert "pattern_offset" not in data["parameters"]


def test_unset_config_roundtrip():
    loaded = SetupCoincidenceMillingTaskConfig.from_dict(
        SetupCoincidenceMillingTaskConfig().to_dict()
    )
    assert loaded.objective_position is None
    assert loaded.fm_roi is None
    assert loaded.pattern_offset == Point(0.0, 0.0)
    assert not loaded.is_set_up


def test_development_protocol_loads_the_task():
    import fibsem
    from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol

    path = os.path.join(
        os.path.dirname(fibsem.__file__),
        "applications",
        "autolamella",
        "protocol",
        "development",
        "task-protocol-coincidence.yaml",
    )
    protocol = AutoLamellaTaskProtocol.load(path)
    config = protocol.task_config["Setup Coincidence Milling"]
    assert isinstance(config, SetupCoincidenceMillingTaskConfig)
    assert config.channel_name == "Red Channel"


# ---------------------------------------------------------------------------
# headless run
# ---------------------------------------------------------------------------


def test_headless_run_records_the_objective_and_the_reference(microscope, tmp_path):
    lamella = _lamella(microscope, tmp_path, fm_objective=2.45e-3)
    task = _task(microscope, lamella)

    task.run()

    config = lamella.task_config["Setup Coincidence Milling"]
    # first guess is the FM pose's objective height; the objective was moved there
    assert config.objective_position == pytest.approx(2.45e-3)
    assert config.is_set_up
    # the boxes' frame is on disk and recorded as an output
    assert os.path.exists(task.setup_reference_path)
    assert os.path.basename(task.setup_reference_path) == (
        COINCIDENCE_SETUP_REFERENCE_FILENAME
    )
    recorded = lamella.task_state.outputs.get("other_fib", [])
    assert COINCIDENCE_SETUP_REFERENCE_FILENAME in recorded
    # (the retract on exit is not observable on the simulator: its objective
    # state is derived from the position, so the base retract guard sees a
    # focused objective as already retracted)
    # the stage was left at the milling pose
    assert microscope.get_stage_position().t == pytest.approx(
        lamella.milling_pose.stage_position.t
    )


def test_a_stored_objective_position_wins_over_the_fm_pose(microscope, tmp_path):
    lamella = _lamella(microscope, tmp_path, fm_objective=2.45e-3)
    task = _task(microscope, lamella, objective_position=2.418e-3)

    task.run()

    assert lamella.task_config[
        "Setup Coincidence Milling"
    ].objective_position == pytest.approx(2.418e-3)


def test_no_objective_position_anywhere_records_where_insertion_put_it(
    microscope, tmp_path, caplog
):
    """Nothing stored and no FM pose height: the task still completes, recording
    the objective's post-insertion position and saying so. A supervised run gives
    the operator the chance to correct it; a headless one has nothing better."""
    import logging

    lamella = _lamella(microscope, tmp_path)
    lamella.fluorescence_pose.objective_position = None
    task = _task(microscope, lamella)

    with caplog.at_level(logging.WARNING):
        task.run()

    config = lamella.task_config["Setup Coincidence Milling"]
    assert config.is_set_up
    assert "no objective position known" in caplog.text


def test_requires_a_milling_pose(microscope, tmp_path):
    lamella = _lamella(microscope, tmp_path)
    lamella.poses.pop("MILLING")
    task = _task(microscope, lamella)

    with pytest.raises(ValueError, match="Milling pose"):
        task.run()

    assert "milling pose" in lamella.task_state.status_message.lower()
    assert not lamella.task_config["Setup Coincidence Milling"].is_set_up
