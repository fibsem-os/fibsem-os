"""`RemoteMicroscope`: a `FibsemMicroscope` whose devices and services are an App
Server's, over the Demo. The facade's calls run on the server.
"""

import threading
import time

import pytest

from fibsem.cancellation import OperationCancelledError
from fibsem.client.remote_microscope import RemoteMicroscope
from fibsem.drivers.remote.devices import RemoteBeam, RemoteStage
from fibsem.drivers.remote.services import RemoteService
from fibsem.server import AuthConfig, build_server
from fibsem.services.milling import Milling
from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import (
    BeamType,
    FibsemMillingSettings,
    FibsemRectangleSettings,
    FibsemStagePosition,
    ImageSettings,
    MillingState,
)
from tests.fixtures.app_server import serving

TOKEN = "test-token"


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


@pytest.fixture(scope="module")
def auth():
    return AuthConfig.generate(level="drive", token=TOKEN)


@pytest.fixture(scope="module")
def url(microscope, auth):
    with serving(build_server(microscope, auth=auth)) as url:
        yield url


@pytest.fixture(scope="module")
def remote(url):
    remote = RemoteMicroscope.connect(url, TOKEN)
    yield remote
    remote.disconnect()


@pytest.fixture(autouse=True)
def at_drive(auth):
    yield
    auth.set_level("drive")


def test_it_has_the_servers_devices_and_services(remote, microscope):
    assert set(remote.devices) == set(microscope.devices)
    assert set(remote.services) == set(microscope.services)
    assert isinstance(remote.beams[BeamType.ELECTRON], RemoteBeam)
    assert isinstance(remote.stage, RemoteStage)
    assert isinstance(remote.milling, RemoteService)
    assert remote.system.info.model == "DemoMicroscope"
    # a service keeps its own signals and the signatures of what it serves
    assert remote.milling.patterns_changed is not Milling.patterns_changed
    assert "draw" in remote.milling.commands and "run" in remote.milling.commands
    assert remote.milling.commands["run"].long


def test_the_facade_reads_and_acquires_on_the_server(remote, microscope):
    assert remote.get_stage_position() == microscope.get_stage_position()
    image = remote.acquire_image(
        ImageSettings(beam_type=BeamType.ELECTRON, resolution=(64, 48))
    )
    assert isinstance(image, FibsemImage) and image.data.shape == (48, 64)
    state = remote.get_microscope_state()
    assert state.stage_position == microscope.get_stage_position()


def test_a_move_moves_the_servers_stage(remote, microscope):
    start = microscope.get_stage_position()
    try:
        moved = remote.move_stage_relative(FibsemStagePosition(x=10e-6))
        assert moved.x == pytest.approx(start.x + 10e-6)
        assert microscope.get_stage_position().x == pytest.approx(start.x + 10e-6)
    finally:
        microscope.move_stage_absolute(start)


def test_a_query_takes_its_arguments_positionally(remote, microscope):
    position = microscope.get_stage_position()
    assert remote.stage_movement.current_milling_angle() == pytest.approx(
        microscope.stage_movement.current_milling_angle()
    )
    assert remote.stage_movement.stage_orientation(position) == (
        microscope.stage_movement.stage_orientation(position)
    )


def _square():
    return FibsemRectangleSettings(
        width=5e-6, height=5e-6, depth=1e-6, centre_x=0, centre_y=0
    )


def test_a_mill_runs_on_the_server_and_its_signals_cross(remote):
    drawn = []
    remote.milling.patterns_changed.connect(drawn.append)
    try:
        remote.setup_milling(FibsemMillingSettings())
        remote.draw_patterns([_square()])
        remote.run_milling()  # waits on the server's run
        remote.finish_milling()
        assert remote.milling.state.get_value() is MillingState.IDLE
        # drawn, then cleared at the end of the run
        assert wait_for(lambda: len(drawn) >= 2)
        assert len(drawn[0]) == 1 and drawn[-1] == []
    finally:
        remote.milling.patterns_changed.disconnect(drawn.append)


def test_a_set_stop_event_stops_the_mill(remote):
    stop = threading.Event()
    stop.set()
    remote.setup_milling(FibsemMillingSettings())
    remote.draw_patterns([_square()])
    try:
        with pytest.raises(OperationCancelledError):
            remote.run_milling(stop_event=stop)
    finally:
        remote.finish_milling()


def test_a_parameter_change_on_the_server_reaches_the_remote(remote, microscope):
    hfw = remote.beams[BeamType.ELECTRON].hfw
    microscope.set_field_of_view(165e-6, BeamType.ELECTRON)
    assert wait_for(lambda: hfw.cached == pytest.approx(165e-6))


def test_a_refused_command_raises(remote, auth):
    auth.set_level("watch")
    with pytest.raises(Exception, match="403"):
        remote.move_stage_relative(FibsemStagePosition(x=1e-6))


def test_disconnect_closes_its_connection(url):
    remote = RemoteMicroscope.connect(url, TOKEN)
    remote.disconnect()
    assert remote._client is None


def test_a_stop_event_set_while_milling_stops_the_servers_run(
    remote, microscope, monkeypatch
):
    monkeypatch.setenv("FIBSEM_SIM_NO_DELAY", "0")  # a mill that takes a while
    stop = threading.Event()
    remote.setup_milling(FibsemMillingSettings())
    remote.draw_patterns([_square()])
    threading.Timer(0.3, stop.set).start()
    started = time.monotonic()
    try:
        with pytest.raises(OperationCancelledError):
            remote.run_milling(stop_event=stop)
        assert time.monotonic() - started < 5
        assert microscope.milling.runs.latest.state == "cancelled"
    finally:
        remote.finish_milling()


def test_back_to_back_commands_never_find_the_last_one_in_control(remote):
    """Control is given up before a command's answer is sent."""
    for _ in range(200):
        remote.milling.clear()


def test_what_workflows_read_off_the_services_is_served(remote, microscope):
    """The calls the app and its tasks make on a service, beyond its commands:
    each answers as the server's would."""
    position = microscope.get_stage_position()
    assert remote.orientations == microscope.orientations
    assert remote.stage_movement.supports_vertical_move(BeamType.ION) == (
        microscope.stage_movement.supports_vertical_move(BeamType.ION)
    )
    assert remote.stage_movement.fm_calibration_needed() == (
        microscope.stage_movement.fm_calibration_needed()
    )
    assert remote.sample_handling.grid_inventory() == (
        microscope.sample_handling.grid_inventory()
    )
    assert remote.sample_handling.grid_at_position(position) == (
        microscope.sample_handling.grid_at_position(position)
    )
    assert remote.milling.current_used() == microscope.milling.current_used()
    remote.stop_milling()  # stop_pattern, with nothing milling: nothing to do


def test_what_touches_no_hardware_runs_here(remote, microscope):
    stage = microscope.milling.estimate_stage  # a classmethod: no server call
    assert remote.milling.estimate_stage.__func__ is stage.__func__
    shown = []
    remote.imaging.image_shown.connect(shown.append)
    image = FibsemImage.generate_blank_image(resolution=(32, 24))
    remote.imaging.show(image)  # to this app's views, not the server's
    assert shown == [image]


def test_reference_images_come_back_as_images(remote):
    images = remote.imaging.reference_images(
        ImageSettings(resolution=(64, 48), save=False), ion=False
    )
    assert images[1] is None
    assert isinstance(images[0], FibsemImage) and images[0].data.shape == (48, 64)
    assert images[0].metadata is not None


def test_a_saved_image_is_saved_on_this_computer(remote, tmp_path):
    """A task's save path is this computer's: the server acquires, this side
    saves and records, and the server is never sent the path (it would refuse
    one outside its experiment)."""
    from fibsem import acquire

    records = []
    remote.session.record_signal.connect(lambda kind, payload: records.append(payload))
    settings = ImageSettings(
        resolution=(64, 48), save=True, path=str(tmp_path), filename="ref"
    )
    try:
        image = acquire.acquire_image(remote, settings)
        pair = remote.imaging.reference_images(settings)
        sets = remote.imaging.reference_image_sets(settings, hfws=[80e-6, 150e-6])
        agent = remote.imaging.acquire(BeamType.ION, path=str(tmp_path), filename="a")
    finally:
        remote.session.record_signal.disconnect()
    assert image.filepath == str(tmp_path / "ref_eb.tif")
    assert [i.filepath for i in pair] == [
        str(tmp_path / "ref_eb.tif"),
        str(tmp_path / "ref_ib.tif"),
    ]
    assert len(sets) == 2 and sets[1][1].filepath.endswith("ref_image_res_02_ib.tif")
    assert agent.filepath == str(tmp_path / "a_ib.tif")
    assert {p.name for p in tmp_path.iterdir()} == {
        "ref_eb.tif",
        "ref_ib.tif",
        "ref_image_res_01_eb.tif",
        "ref_image_res_01_ib.tif",
        "ref_image_res_02_eb.tif",
        "ref_image_res_02_ib.tif",
        "a_ib.tif",
    }
    assert [r["path"] for r in records][:1] == [str(tmp_path / "ref_eb.tif")]


def test_an_image_unsaved_says_so(remote):
    image = remote.acquire_image(ImageSettings(resolution=(64, 48)))
    assert image.filepath is None


def test_images_carry_this_computers_experiment(remote):
    from fibsem.structures.system import FibsemExperimentRef

    before = remote.session.experiment
    remote.session.experiment = FibsemExperimentRef(name="here")
    try:
        image = remote.acquire_image(ImageSettings(resolution=(64, 48)))
    finally:
        remote.session.experiment = before
    assert image.metadata.experiment.name == "here"


def _task_config(name="Rough Milling"):
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.milling.tasks import FibsemMillingTaskConfig

    config = FibsemMillingTaskConfig.from_stages(
        [FibsemMillingStage(name="Rough")], name=name
    )
    config.alignment.enabled = True
    config.acquisition.acquire_sem = config.acquisition.acquire_fib = True
    return config


def test_a_milling_task_runs_on_the_server_and_its_folder_comes_here(
    remote, tmp_path, monkeypatch
):
    """A workflow's milling task through a remote microscope: the server runs it
    (its run_task), and the folder it saved, alignment images and all, comes back
    under this computer's experiment; the server keeps none of it."""
    import fibsem.config as fcfg
    from fibsem.milling.progress import MillingProgressStatus
    from fibsem.milling.tasks import run_milling_task

    server_data = tmp_path / "support-pc"
    monkeypatch.setattr(fcfg, "DATA_CC_PATH", str(server_data))
    here = tmp_path / "remote-pc" / "lamella-01"
    task = run_milling_task(remote, _task_config(), output_dir=str(here))

    assert task.error is None
    assert task.outcome is MillingProgressStatus.TASK_FINISHED
    folder = here / "Milling" / "Rough-Milling"
    assert task.ended()["path"] == str(folder)
    assert task.ended()["stages"][0]["status"] == "completed"
    saved = {p.relative_to(folder).parts[0] for p in folder.rglob("*") if p.is_file()}
    assert "Alignment" in saved  # the alignment diagnostics came back too
    assert any(name.endswith("_ib.tif") for name in saved)
    assert task.images and all(isinstance(i, FibsemImage) for i in task.images)
    assert not any(p.is_file() for p in server_data.rglob("*"))


def test_a_stopped_remote_task_ends_cancelled(remote, tmp_path, monkeypatch):
    """A workflow's stop reaches the task on the server, whenever it comes: here
    while the stage is about to mill."""
    import fibsem.config as fcfg
    from fibsem.milling.progress import MillingProgressStatus
    from fibsem.milling.strategy.standard import StandardMillingStrategy
    from fibsem.milling.tasks import run_milling_task

    monkeypatch.setattr(fcfg, "DATA_CC_PATH", str(tmp_path / "support-pc"))
    stop = threading.Event()
    strategy_run = StandardMillingStrategy.run

    def held(self, *args, **kwargs):  # on the server: mills once the stop is in
        stop.set()
        time.sleep(1.0)  # the client's stop reaches the server
        return strategy_run(self, *args, **kwargs)

    monkeypatch.setattr(StandardMillingStrategy, "run", held)
    config = _task_config()
    config.alignment.enabled = False
    task = run_milling_task(remote, config, stop_event=stop, output_dir=str(tmp_path))
    assert task.outcome is MillingProgressStatus.TASK_CANCELLED
    assert task.error is None
