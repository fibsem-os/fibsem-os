"""The grid screening report's data, read off a screened experiment (FIB-1057).

Three grids on the Arctis simulator, the shape the issue asks the report to be
tested on: one screened in full (SEM, FIB, FM) with lamellae placed on it, one
whose FIB overview failed after its SEM, and one that did not load. The overview
runs are real -- the images on disk carry the metadata the report reads -- and
the failures are the entries the task manager writes.
"""

import os
from types import SimpleNamespace

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Experiment,
    GridRecord,
    Lamella,
    Verdict,
)
from fibsem.applications.autolamella.tools.grid_report import (
    LAMELLA_BOX_WIDTH,
    collect_grid_report,
    read_beam_metadata,
)
from fibsem.applications.autolamella.workflows.tasks.grid import (
    BeamOverviewGridTaskConfig,
    FluorescenceOverviewGridTaskConfig,
    run_grid_task,
)
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    LOAD_ENTRY_NAME,
)
from fibsem.fm.structures import ChannelSettings, OverviewParameters
from fibsem.microscopes._stage import SampleGrid
from fibsem.structures import (
    BeamType,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    OverviewAcquisitionSettings,
)


def _settings(beam=BeamType.ELECTRON) -> OverviewAcquisitionSettings:
    return OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=(128, 128), hfw=200e-6, beam_type=beam),
        nrows=2,
        ncols=2,
    )


def _entry(name, status, message="", outputs=None) -> AutoLamellaTaskState:
    state = AutoLamellaTaskState(
        name=name, status=status, status_message=message, outputs=outputs or {}
    )
    state.end_timestamp = state.start_timestamp + 1
    return state


def _lamella(name, grid, milling=None, fluorescence=None) -> Lamella:
    lamella = Lamella(path="/nowhere", number=0, petname=name, grid_id=grid.id)
    if milling is not None:
        lamella.milling_pose = MicroscopeState(stage_position=milling)
    if fluorescence is not None:
        lamella.fluorescence_pose = MicroscopeState(stage_position=fluorescence)
    return lamella


@pytest.fixture(scope="module")
def screened(tmp_path_factory):
    """A screened three-grid experiment. Module-scoped: the runs are the cost."""
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
    )
    tmp_path = tmp_path_factory.mktemp("report")
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.grid_protocol.add(
        BeamOverviewGridTaskConfig(task_name="overview_sem", settings=_settings())
    )
    exp.grid_protocol.add(
        BeamOverviewGridTaskConfig(
            task_name="overview_fib",
            orientation="FIB",
            settings=_settings(BeamType.ION),
        )
    )
    exp.grid_protocol.add(
        FluorescenceOverviewGridTaskConfig(
            task_name="overview_fm",
            channels=[
                ChannelSettings(name="GFP", color="green"),
                ChannelSettings(name="mCherry", color="red"),
            ],
            overview=OverviewParameters(rows=1, cols=2),
        )
    )
    slot = microscope._stage.holder.slots["Slot-01"]

    # grid-aspen: loaded, screened in full, two lamellae, judged good
    aspen = exp.add_grid(GridRecord(name="grid-aspen"))
    aspen.quality.verdict = Verdict.GOOD
    aspen.quality.author = "human:operator"
    aspen.description = "Even ice, cells on the east half."
    aspen.task_history.append(
        _entry(LOAD_ENTRY_NAME, AutoLamellaTaskStatus.Completed, "Loaded into Slot-01.")
    )
    slot.loaded_grid = SampleGrid(name="grid-aspen")
    for task in ("overview_sem", "overview_fib", "overview_fm"):
        run_grid_task(microscope, task, exp, aspen)

    sem_meta, _ = read_beam_metadata(
        os.path.join(
            exp.grid_path(aspen), aspen.task_history[1].outputs["overview_sem"][0]
        )
    )
    sem_centre = sem_meta.microscope_state.stage_position
    fm_stage = microscope.get_stage_position()  # the runs end at the FM
    east = FibsemStagePosition(
        x=sem_centre.x + 50e-6,
        y=sem_centre.y,
        z=sem_centre.z,
        r=sem_centre.r,
        t=sem_centre.t,
    )
    exp.positions.append(
        _lamella(
            "lamella-01",
            aspen,
            milling=east,
            fluorescence=FibsemStagePosition(
                x=fm_stage.x + 20e-6,
                y=fm_stage.y,
                z=fm_stage.z,
                r=fm_stage.r,
                t=fm_stage.t,
            ),
        )
    )
    exp.positions.append(_lamella("lamella-02", aspen, milling=sem_centre))

    # grid-birch: loaded, SEM taken, FIB failed, FM never ran; judged poor
    birch = exp.add_grid(GridRecord(name="grid-birch"))
    birch.quality.verdict = Verdict.FAILED
    birch.description = "Cracked film, thick ice."
    birch.task_history.append(
        _entry(LOAD_ENTRY_NAME, AutoLamellaTaskStatus.Completed, "Loaded into Slot-01.")
    )
    slot.loaded_grid = SampleGrid(name="grid-birch")
    run_grid_task(microscope, "overview_sem", exp, birch)
    birch.task_history.append(
        _entry(
            "overview_fib",
            AutoLamellaTaskStatus.Failed,
            "Grid 'grid-birch' is not in a holder slot. Load it first.",
        )
    )

    # grid-cedar: never loaded
    cedar = exp.add_grid(GridRecord(name="grid-cedar"))
    cedar.task_history.append(
        _entry(
            LOAD_ENTRY_NAME,
            AutoLamellaTaskStatus.Failed,
            "grid-cedar could not be gripped from Slot-03.",
        )
    )
    exp.save(save_protocol=True)
    return Experiment.load(tmp_path / "exp" / "experiment.yaml")


@pytest.fixture(scope="module")
def report(screened):
    return collect_grid_report(screened)


class TestCover:
    def test_one_section_per_grid_in_card_order(self, report):
        assert [s.name for s in report.sections] == [
            "grid-aspen",
            "grid-birch",
            "grid-cedar",
        ]

    def test_the_verdict_is_the_operators(self, report):
        aspen, birch, cedar = report.sections
        assert aspen.quality.verdict is Verdict.GOOD and aspen.recommended
        assert aspen.quality.author == "human:operator"
        assert aspen.description == "Even ice, cells on the east half."
        assert birch.quality.verdict is Verdict.FAILED and not birch.recommended
        assert cedar.quality.verdict is Verdict.UNASSESSED and not cedar.recommended

    def test_whether_each_grid_loaded(self, report):
        aspen, birch, cedar = report.sections
        assert aspen.loaded is True
        assert cedar.loaded is False
        assert "could not be gripped" in cedar.load.status_message
        assert cedar.overviews == []

    def test_the_slot_comes_from_the_inventory_or_is_unknown(self, screened):
        assert collect_grid_report(screened).sections[0].slot is None
        rows = [SimpleNamespace(name="grid-aspen", slot_name="Slot-01")]
        assert (
            collect_grid_report(screened, inventory=rows).sections[0].slot == "Slot-01"
        )

    def test_the_microscope_is_read_off_an_overview(self, report):
        assert report.microscope

    def test_screened_spans_every_entry(self, report):
        first, last = report.screened
        assert first <= last
        assert report.experiment_name == "exp"
        assert report.protocol == ["overview_sem", "overview_fib", "overview_fm"]


class TestOverviews:
    def test_every_run_in_history_order(self, report):
        overviews = report.sections[0].overviews
        assert [o.task_name for o in overviews] == [
            "overview_sem",
            "overview_fib",
            "overview_fm",
        ]
        assert [o.modality for o in overviews] == ["SEM", "FIB", "FM"]
        assert all(os.path.isfile(o.path) for o in overviews)

    def test_a_beam_caption_has_what_it_needs(self, report):
        sem = report.sections[0].overviews[0]
        assert sem.tiles == (2, 2)
        assert sem.shape is not None and sem.pixel_size > 0
        # the field of view is the mosaic's, not one tile's
        assert sem.fov[0] == pytest.approx(sem.shape[1] * sem.pixel_size)
        assert sem.fov[0] > 200e-6
        assert sem.pose.startswith("r=")

    def test_the_fm_caption_names_its_channels(self, report):
        fm = report.sections[0].overviews[2]
        assert fm.tiles == (1, 2)
        assert fm.channels == ["GFP", "mCherry"]
        assert fm.pixel_size > 0 and fm.fov[0] > 0

    def test_a_failed_run_is_a_row_with_no_image(self, report):
        birch = report.sections[1]
        assert [o.task_name for o in birch.overviews] == [
            "overview_sem",
            "overview_fib",
        ]
        failed = birch.overviews[1]
        assert failed.status is AutoLamellaTaskStatus.Failed
        assert failed.path is None
        assert failed.modality == "FIB"  # from the protocol, since it recorded nothing
        assert "Load it first" in failed.status_message

    def test_a_deleted_image_is_still_a_row(self, screened):
        birch = screened.get_grid_by_name("grid-birch")
        sem = birch.task_history[1]
        path = os.path.join(screened.grid_path(birch), sem.outputs["overview_sem"][0])
        os.rename(path, path + ".gone")
        try:
            entry = collect_grid_report(screened).sections[1].overviews[0]
            assert entry.task_name == "overview_sem" and entry.path is None
            assert entry.status is AutoLamellaTaskStatus.Completed
        finally:
            os.rename(path + ".gone", path)


class TestMarks:
    def test_lamellae_land_where_their_milling_pose_puts_them(self, report):
        sem = report.sections[0].overviews[0]
        by_name = {m.name: m for m in sem.marks}
        assert set(by_name) == {"lamella-01", "lamella-02"}
        h, w = sem.shape
        centre = by_name["lamella-02"]
        assert centre.x == pytest.approx(w / 2, abs=0.5)
        assert centre.y == pytest.approx(h / 2, abs=0.5)
        east = by_name["lamella-01"]
        assert east.x - centre.x == pytest.approx(50e-6 / sem.pixel_size, rel=0.05)
        assert east.width == pytest.approx(LAMELLA_BOX_WIDTH / sem.pixel_size)
        assert sem.unmarked == []

    def test_the_fib_view_places_the_same_lamellae(self, report):
        fib = report.sections[0].overviews[1]
        assert {m.name for m in fib.marks} == {"lamella-01", "lamella-02"}

    def test_the_fm_view_marks_only_lamellae_with_a_fluorescence_pose(self, report):
        fm = report.sections[0].overviews[2]
        assert [m.name for m in fm.marks] == ["lamella-01"]
        assert fm.unmarked == ["lamella-02"]
        h, w = fm.shape
        (mark,) = fm.marks
        assert 0 <= mark.x <= w and 0 <= mark.y <= h

    def test_a_row_with_no_image_names_every_lamella_as_unmarked(self, report):
        assert report.sections[0].lamellae == ["lamella-01", "lamella-02"]
        failed = report.sections[1].overviews[1]
        assert failed.marks == [] and failed.unmarked == []


class TestWithoutAProtocol:
    def test_an_experiment_loaded_without_its_protocol_still_reports(self, screened):
        bare = Experiment.load(os.path.join(screened.path, "experiment.yaml"))
        bare.task_protocol = None
        report = collect_grid_report(bare)
        assert report.protocol == []
        aspen, birch, _ = report.sections
        # what the history recorded is still read; only the protocol's facts go
        assert [o.role for o in aspen.overviews] == [
            "overview_sem",
            "overview_fib",
            "overview_fm",
        ]
        assert aspen.overviews[0].tiles is None
        # a failed run recorded nothing, so without the protocol it has no role
        assert [o.task_name for o in birch.overviews] == ["overview_sem"]
        assert report.outcomes["grid-aspen"] == {}


class TestOutcomes:
    def test_latest_status_and_run_count_per_task(self, report):
        aspen = report.outcomes["grid-aspen"]
        assert {k: v.status for k, v in aspen.items()} == {
            "overview_sem": AutoLamellaTaskStatus.Completed,
            "overview_fib": AutoLamellaTaskStatus.Completed,
            "overview_fm": AutoLamellaTaskStatus.Completed,
        }
        birch = report.outcomes["grid-birch"]
        assert birch["overview_fib"].status is AutoLamellaTaskStatus.Failed
        assert birch["overview_fm"].status is None and birch["overview_fm"].runs == 0
        cedar = report.outcomes["grid-cedar"]
        assert all(v.status is None for v in cedar.values())
