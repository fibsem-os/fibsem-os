"""MillCoincidentTask: a queued coincidence mill driven by the per-site setup
(FIB-910). Headless on the Arctis sim: setup runs first, then the mill, and what
is under test is the choreography, the record, and the ways the loop ends.
"""

import logging
import os
import threading
from copy import deepcopy

import pytest
import yaml

import fibsem.config as fconfig
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.workflows.tasks import get_tasks
from fibsem.applications.autolamella.workflows.tasks.acquire_fluorescence import (
    AcquireFluorescenceImageConfig,
)
from fibsem.applications.autolamella.workflows.tasks.mill_coincident import (
    MILL_COINCIDENT_KEY,
    MillCoincidentTask,
    MillCoincidentTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.setup_coincidence_milling import (
    SetupCoincidenceMillingTask,
    SetupCoincidenceMillingTaskConfig,
)
from fibsem.fm.structures import ChannelSettings
from fibsem.milling.strategy.coincidence import CoincidenceMillingStrategy
from fibsem.structures import FibsemRectangle, Point

SIM_ARCTIS_CONFIG_PATH = os.path.join(
    fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml"
)
SETUP_NAME = "Setup Coincidence Milling"


class _Manager:
    """The one thing a task reads off its manager: the abort token."""

    def __init__(self):
        self.abort_token = threading.Event()


@pytest.fixture
def microscope(tmp_path):
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


def _lamella(microscope, tmp_path) -> Lamella:
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    fm_pose = microscope.get_microscope_state()
    fm_pose.objective_position = 2.45e-3
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


def _run_setup(microscope, lamella, **kwargs) -> SetupCoincidenceMillingTaskConfig:
    config = SetupCoincidenceMillingTaskConfig(task_name=SETUP_NAME, **kwargs)
    lamella.task_config[SETUP_NAME] = config
    SetupCoincidenceMillingTask(
        microscope=microscope, config=config, lamella=lamella
    ).run()
    return config


def _mill_task(
    microscope, lamella, timeout: int = 30, manager=None
) -> MillCoincidentTask:
    config = MillCoincidentTaskConfig(task_name="Coincidence Milling")
    lamella.task_config[config.task_name] = config
    milling = config.milling[MILL_COINCIDENT_KEY]
    for stage in milling.stages:
        strategy = stage.strategy
        assert isinstance(strategy, CoincidenceMillingStrategy)
        strategy.config.timeout = timeout
        strategy.config.warmup_duration = 1.0
        strategy.config.save_fm_images = False
        strategy.config.acquire_fib_image = False
    return MillCoincidentTask(
        microscope=microscope, config=config, lamella=lamella, task_manager=manager
    )


def test_registered_as_a_builtin_task():
    assert get_tasks()["MILL_COINCIDENT"] is MillCoincidentTask


def test_development_protocol_loads_both_tasks():
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
    config = protocol.task_config["Coincidence Milling"]
    assert isinstance(config, MillCoincidentTaskConfig)
    assert config.setup_task == SETUP_NAME
    assert config.monitoring_channel.name == "Monitoring"
    assert config.monitoring_channel.exposure_time == 0.1
    assert isinstance(
        config.milling[MILL_COINCIDENT_KEY].stages[0].strategy,
        CoincidenceMillingStrategy,
    )


def test_monitoring_channel_roundtrips_and_is_not_a_parameter():
    from fibsem.fm.structures import ChannelSettings

    config = MillCoincidentTaskConfig(
        task_name="Coincidence Milling",
        monitoring_channel=ChannelSettings(
            name="Green", excitation_wavelength=488, power=0.05, exposure_time=0.05
        ),
    )
    data = yaml.safe_load(yaml.safe_dump(config.to_dict()))
    assert "monitoring_channel" not in data["parameters"]
    loaded = MillCoincidentTaskConfig.from_dict(data)
    assert loaded.monitoring_channel.name == "Green"
    assert loaded.monitoring_channel.excitation_wavelength == 488
    assert loaded.monitoring_channel.exposure_time == pytest.approx(0.05)


def test_the_mill_sets_the_monitoring_channel_on_the_fm(microscope, tmp_path):
    from fibsem.fm.structures import ChannelSettings

    lamella = _lamella(microscope, tmp_path)
    _run_setup(microscope, lamella)
    task = _mill_task(microscope, lamella, timeout=3)
    task.config.monitoring_channel = ChannelSettings(
        name="Green", excitation_wavelength=488, power=0.05, exposure_time=0.05
    )
    # the final z-stack would set the fluorescence task's channels afterwards;
    # skip it so the channel left on the FM is the one the mill monitored on
    task.config.acquire_fluorescence_images = False

    task.run()

    assert microscope.fm.channel_name == "Green"
    assert microscope.fm.filter_set.excitation_wavelength == 488


def test_fails_before_moving_without_a_setup_record(microscope, tmp_path):
    lamella = _lamella(microscope, tmp_path)
    task = _mill_task(microscope, lamella)
    pose_before = microscope.get_stage_position()

    with pytest.raises(ValueError, match="Setup Coincidence Milling"):
        task.run()

    assert microscope.get_stage_position() == pose_before


def test_fails_when_setup_has_not_run(microscope, tmp_path):
    lamella = _lamella(microscope, tmp_path)
    lamella.task_config[SETUP_NAME] = SetupCoincidenceMillingTaskConfig(
        task_name=SETUP_NAME
    )
    task = _mill_task(microscope, lamella)

    with pytest.raises(ValueError, match="no objective position"):
        task.run()


def test_headless_mill_runs_from_the_setup_record_to_timeout(
    microscope, tmp_path, caplog
):
    """Setup then mill, no UI. The setup's boxes land on every stage, the strategy
    is unsupervised, and with noise frames the loop ends at the (short) timeout."""
    lamella = _lamella(microscope, tmp_path)
    setup = _run_setup(
        microscope,
        lamella,
        intensity_drop_fraction=0.3,
        fm_roi=FibsemRectangle(0.4, 0.4, 0.2, 0.2),
        pattern_offset=Point(1.0e-6, -0.5e-6),
    )
    assert setup.objective_position == pytest.approx(2.45e-3)
    task = _mill_task(microscope, lamella, timeout=3)

    with caplog.at_level(logging.INFO):
        task.run()

    milling = lamella.task_config["Coincidence Milling"].milling[MILL_COINCIDENT_KEY]
    for stage in milling.enabled_stages:
        assert stage.pattern.point.x == pytest.approx(1.0e-6)
        assert stage.pattern.point.y == pytest.approx(-0.5e-6)
        strategy = stage.strategy
        assert isinstance(strategy, CoincidenceMillingStrategy)
        assert strategy.config.bbox == FibsemRectangle(0.4, 0.4, 0.2, 0.2)
        assert strategy.config.intensity_drop_fraction == pytest.approx(0.3)
        assert strategy.config.supervised is False  # headless = unsupervised
        assert strategy.end_reason == "timeout"
    assert "Coincidence milling ended" in caplog.text
    assert "timeout" in caplog.text
    # and the record says so, not just the log
    assert lamella.task_history[-1].status_message.startswith("Finished · ")
    assert "timeout" in lamella.task_history[-1].status_message
    # aligned to the setup reference, not the generic one
    assert "no coincidence setup reference" not in caplog.text
    # the FM is quiet on the way out
    assert not microscope.fm.is_acquiring
    # final images recorded
    assert lamella.task_state.outputs.get("fluorescence")
    assert lamella.task_state.outputs.get("final_fib")


def test_abort_token_stops_the_mill(microscope, tmp_path):
    lamella = _lamella(microscope, tmp_path)
    _run_setup(microscope, lamella)
    manager = _Manager()
    task = _mill_task(microscope, lamella, timeout=600, manager=manager)

    from fibsem.milling.progress import MillingProgressStatus

    def _stop_on_first_tick(progress):
        if progress.status is MillingProgressStatus.STAGE_UPDATE:
            manager.abort_token.set()

    microscope.milling_progress_signal.connect(_stop_on_first_tick)
    try:
        task.run()
    except Exception:
        pass  # the base run re-raises a cancellation after recording it
    finally:
        microscope.milling_progress_signal.disconnect(_stop_on_first_tick)

    milling = lamella.task_config["Coincidence Milling"].milling[MILL_COINCIDENT_KEY]
    strategy = milling.enabled_stages[0].strategy
    assert isinstance(strategy, CoincidenceMillingStrategy)
    assert strategy.end_reason == "stopped"
    # a second stage the stop cancelled before it began reads as not run
    second = deepcopy(milling.enabled_stages[0])
    second.strategy.end_reason = None
    milling.stages.append(second)
    task._record_end_reason(milling)
    assert task.finished_message.endswith("not run")


class _Responder:
    """A stand-in for the GUI: answers every request at once, as told."""

    def __init__(self, answer=None):
        self.answer = answer
        self.requests = []

    def submit(self, request, future):
        self.requests.append(request)
        future.set_result(self.answer(request) if callable(self.answer) else None)


class _Signal:
    """The Qt signals the task path emits into, without Qt (this directory
    runs on the thin CI build)."""

    def emit(self, *args):
        pass


class _ParentUI:
    """The attributes the task path reads off the main window.

    The experiment and its protocol are the real objects: a stub protocol
    answering get_supervision is what silently broke when the protocol moved
    to one attention per task.
    """

    def __init__(self, responder, attention: Attention, tmp_path, task_name):
        self.ui_responder = responder
        self._workflow_stop_event = threading.Event()
        self.workflow_status_signal = _Signal()
        self.step_update_signal = _Signal()
        self.update_experiment_signal = _Signal()
        self.experiment = Experiment(path=str(tmp_path), name="mill-coincident-ui")
        self.experiment.task_protocol = AutoLamellaTaskProtocol(
            workflow_config=AutoLamellaWorkflowConfig(
                tasks=[AutoLamellaTaskDescription(name=task_name, attention=attention)]
            )
        )


def test_supervised_asks_the_viewer_to_run_and_records_its_answer(microscope, tmp_path):
    from fibsem.applications.autolamella.workflows.interaction import (
        RunCoincidenceMilling,
    )

    lamella = _lamella(microscope, tmp_path)
    _run_setup(microscope, lamella, pattern_offset=Point(1.0e-6, -0.5e-6))

    def answer(request):
        if isinstance(request, RunCoincidenceMilling):
            ran = deepcopy(request.milling_config)
            for stage in ran.enabled_stages:
                stage.strategy.end_reason = "drop"
            return ran
        return None

    responder = _Responder(answer)
    task = _mill_task(microscope, lamella, timeout=3)
    task.parent_ui = _ParentUI(
        responder, Attention.supervised, tmp_path, task.task_name
    )
    task.run()

    asked = [r for r in responder.requests if isinstance(r, RunCoincidenceMilling)]
    assert len(asked) == 1
    assert asked[0].lamella is lamella
    assert asked[0].fib_image is not None  # the aligned frame the boxes sit on
    assert asked[0].monitoring_channel.name == task.config.monitoring_channel.name
    milling = lamella.task_config["Coincidence Milling"].milling[MILL_COINCIDENT_KEY]
    assert all(s.strategy.end_reason == "drop" for s in milling.enabled_stages)
    assert "drop" in lamella.task_history[-1].status_message


def test_automated_with_a_ui_tells_the_viewer_to_watch_and_release(
    microscope, tmp_path
):
    from fibsem.applications.autolamella.workflows.interaction import (
        ReleaseCoincidenceMilling,
        WatchCoincidenceMilling,
    )

    lamella = _lamella(microscope, tmp_path)
    _run_setup(microscope, lamella)
    responder = _Responder()
    task = _mill_task(microscope, lamella, timeout=3)
    task.parent_ui = _ParentUI(responder, Attention.automated, tmp_path, task.task_name)
    task.run()

    kinds = [type(r) for r in responder.requests]
    assert kinds.index(WatchCoincidenceMilling) < kinds.index(ReleaseCoincidenceMilling)
    watch = next(
        r for r in responder.requests if isinstance(r, WatchCoincidenceMilling)
    )
    # the viewer's Stop would have stopped this very run
    assert callable(watch.stop)
    milling = lamella.task_config["Coincidence Milling"].milling[MILL_COINCIDENT_KEY]
    assert milling.enabled_stages[0].strategy.end_reason == "timeout"
