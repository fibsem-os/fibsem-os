"""What fibsem writes today, pinned, so switching a class to `fibsem.codec` shows
every change to its saved shape.

Each case is a structure as it is saved: one built here, or a file in an older shape
read and saved again (configurations, image metadata). Its saved form is compared
with `tests/golden/<case>.json`. A switch that changes a saved shape on purpose
updates the golden file in the same pull request, so the change is in the diff.

Write the golden files again with::

    FIBSEM_UPDATE_GOLDEN=1 python -m pytest tests/test_saved_shapes.py
"""

import json
import math
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from fibsem import structures
from fibsem.autofunctions.autofocus import (
    AutoFocusSettings,
    FocusMethod,
    FocusSweepPass,
)
from fibsem.codec.upgrades.configuration import upgrade
from fibsem.fm.structures import (
    CameraSettings,
    ChannelSettings,
    EmissionFilter,
    FluorescenceChannelMetadata,
    FluorescenceConfiguration,
    FluorescenceImage,
    FluorescenceImageMetadata,
    ObjectiveStartPosition,
    OverviewParameters,
    ZParameters,
    ZStackOrder,
)
from fibsem.structures import (
    AutoFocusMode,
    BeamSettings,
    BeamType,
    CameraImageTransform,
    FibsemDetectorSettings,
    FibsemImageMetadata,
    FibsemManipulatorPosition,
    FibsemRectangle,
    FibsemStagePosition,
    FibsemUser,
    FocusStackSettings,
    GridSlot,
    ImageSettings,
    ManipulatorSystemSettings,
    MicroscopeSettings,
    MicroscopeState,
    MillingAlignment,
    OverviewAcquisitionSettings,
    Point,
    ReferenceImageParameters,
    SampleGrid,
    SampleHolder,
    SessionInfo,
    SlotCalibration,
    SystemInfo,
    TileOrderStrategy,
)

TESTS = Path(__file__).parent
GOLDEN = TESTS / "golden"
FIXTURES = TESTS / "fixtures"
CONFIGURATIONS = TESTS.parent / "fibsem" / "config"
UPDATE = os.environ.get("FIBSEM_UPDATE_GOLDEN") == "1"


@pytest.fixture(autouse=True)
def _pinned_revision(monkeypatch):
    # A file with no revision is given the running one; pin it.
    monkeypatch.setattr(structures, "get_revision", lambda: "pinned-revision")


def _plain(data):
    """``data`` as it reads back from JSON: tuples are lists."""
    return json.loads(json.dumps(data))


def _posix(data):
    """``data`` with its paths written with forward slashes, as on any OS."""
    return json.loads(json.dumps(data).replace("\\\\", "/"))


def _check(case: str, data) -> None:
    path = GOLDEN / f"{case}.json"
    if UPDATE:
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        return
    assert _plain(data) == json.loads(path.read_text())


RECTANGLE = FibsemRectangle(left=0.1, top=0.2, width=0.3, height=0.4)
IMAGE = ImageSettings(
    resolution=(768, 512),
    dwell_time=2e-6,
    hfw=80e-6,
    autocontrast=True,
    beam_type=BeamType.ION,
    save=True,
    filename="reference",
    path="/data/experiment",
    reduced_area=RECTANGLE,
    line_integration=2,
    scan_interlacing=None,
    frame_integration=4,
    drift_correction=True,
)
STAGE = FibsemStagePosition(
    name="lamella",
    x=1e-3,
    y=-2e-3,
    z=3e-3,
    r=math.radians(30),
    t=math.radians(52),
    coordinate_system="RAW",
)
ELECTRON = BeamSettings(
    beam_type=BeamType.ELECTRON,
    working_distance=4e-3,
    beam_current=50e-12,
    voltage=2000.0,
    hfw=150e-6,
    resolution=(1536, 1024),
    dwell_time=1e-6,
    stigmation=Point(0.1, -0.1),
    shift=Point(1e-6, 2e-6),
    scan_rotation=math.pi,
    preset="30 keV; UHR imaging",
)
DETECTOR = FibsemDetectorSettings(type="ETD", mode="SE", brightness=0.4, contrast=0.6)

# Structures saved on their own or inside others, each built from non-defaults.
BUILT = {
    "point": Point(1e-6, -2e-6),
    "stage-position": STAGE,
    "manipulator-position": FibsemManipulatorPosition(
        x=1e-4, y=2e-4, z=-3e-4, r=0.0, t=0.0, coordinate_system="STAGE"
    ),
    "rectangle": RECTANGLE,
    "image-settings": IMAGE,
    "focus-stack-settings": FocusStackSettings(enabled=True, n_steps=5),
    "overview-acquisition-settings": OverviewAcquisitionSettings(
        nrows=3, ncols=4, overlap=0.1
    ),
    "beam-settings": ELECTRON,
    "detector-settings": DETECTOR,
    "microscope-state": MicroscopeState(
        timestamp=1760000000.0,
        stage_position=STAGE,
        electron_beam=ELECTRON,
        ion_beam=BeamSettings(
            beam_type=BeamType.ION,
            beam_current=1e-9,
            stigmation=Point(),
            shift=Point(),
        ),
        electron_detector=DETECTOR,
        ion_detector=FibsemDetectorSettings(),
        objective_position=1.5e-3,
    ),
}


@pytest.mark.parametrize("case", sorted(BUILT))
def test_a_built_structure_saves_as_pinned_and_reads_back(case):
    value = BUILT[case]
    saved = value.to_dict()
    _check(case, saved)
    assert _plain(type(value).from_dict(_plain(saved)).to_dict()) == _plain(saved)


CONFIGURATION_FILES = {
    f"configuration-{folder}-{path.stem}": path
    for folder, root in (
        ("current", CONFIGURATIONS),
        ("v0", FIXTURES / "configuration_v0"),
        ("v1", FIXTURES / "configuration_v1"),
    )
    for path in sorted(root.glob("*.yaml"))
}


@pytest.mark.parametrize("case", sorted(CONFIGURATION_FILES))
def test_a_configuration_file_saves_again_as_pinned(case):
    data = upgrade(yaml.safe_load(CONFIGURATION_FILES[case].read_text()))
    _check(case, MicroscopeSettings.from_dict(data).to_dict())


METADATA_FILES = {
    f"image-metadata-{path.stem}": path
    for path in sorted((FIXTURES / "metadata").glob("*.json"))
}


@pytest.mark.parametrize("case", sorted(METADATA_FILES))
def test_image_metadata_saves_again_as_pinned(case):
    data = json.loads(METADATA_FILES[case].read_text())
    _check(case, FibsemImageMetadata.from_dict(data).to_dict())


# ---- fluorescence and autofocus ----------------------------------------------

FM_CHANNEL = ChannelSettings(
    name="GFP",
    excitation_wavelength=488,
    emission_filter=EmissionFilter("GFP", bands=[(500.0, 550.0)]),
    power=0.2,
    exposure_time=0.1,
    color="green",
    gain=0.5,
)
FM_DUAL_BAND = EmissionFilter("DAPI/GFP", bands=[(500.0, 550.0), (420.0, 460.0)])
FOCUS = AutoFocusSettings(
    method=FocusMethod.LAPLACIAN,
    passes=[
        FocusSweepPass(search_range=40e-6, step_size=4e-6),
        FocusSweepPass(search_range=8e-6, step_size=1e-6, enabled=False),
    ],
    probe_resolution=(1536, 1024),
    probe_dwell_time=1e-6,
    reduced_area=RECTANGLE,
    use_autocontrast=False,
    channel_name="GFP",
)
FM_OVERVIEW = OverviewParameters(
    rows=2,
    cols=3,
    overlap=0.2,
    use_zstack=True,
    autofocus_mode=AutoFocusMode.EACH_ROW,
    tile_order=TileOrderStrategy.SERPENTINE,
    tile_mask=[[True, False, True], [True, True, False]],
    objective_start=ObjectiveStartPosition.FOCUS,
    register_tiles=True,
)
FM_CAMERA = CameraSettings(
    gain=0.3, offset=0.1, binning=2, transform=CameraImageTransform.FLIP_X
)
FM_Z = ZParameters(zmin=-5e-6, zmax=5e-6, zstep=0.5e-6, order=ZStackOrder.Z_LEVEL)

BUILT_FM = {
    "emission-filter": FM_DUAL_BAND,
    "channel-settings": FM_CHANNEL,
    "z-parameters": FM_Z,
    "overview-parameters": FM_OVERVIEW,
    "camera-settings": FM_CAMERA,
    "autofocus-settings": FOCUS,
    "fluorescence-configuration": FluorescenceConfiguration(
        channel_settings=[FM_CHANNEL, ChannelSettings()],
        z_parameters=FM_Z,
        overview_parameters=FM_OVERVIEW,
        autofocus_settings=FOCUS,
        camera_settings=FM_CAMERA,
        focus_position=8.3e-3,
        limit_position=8.6e-3,
        default_orientation="FM",
    ),
    "fluorescence-image-metadata": FluorescenceImageMetadata(
        acquisition_datetime=datetime.fromisoformat("2026-10-19T03:00:00+11:00"),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        resolution=(512, 256),
        channels=[
            FluorescenceChannelMetadata(
                name="GFP",
                excitation_wavelength=488,
                power=0.2,
                exposure_time=0.1,
                gain=None,
                offset=0.0,
                emission_filter=FM_DUAL_BAND,
                binning=2,
                objective_position=8.4e-3,
                objective_magnification=100.0,
                objective_numerical_aperture=0.9,
                color="green",
            )
        ],
        z_positions=[8.3e-3, 8.4e-3, 8.5e-3],
        stage_position=STAGE,
        filename="fm-image",
        description="an overview tile",
        system_info={"name": "sim"},
    ),
}


@pytest.mark.parametrize("case", sorted(BUILT_FM))
def test_a_built_fm_structure_saves_as_pinned_and_reads_back(case):
    value = BUILT_FM[case]
    saved = value.to_dict()
    _check(case, saved)
    assert _plain(type(value).from_dict(_plain(saved)).to_dict()) == _plain(saved)


# Shapes older builds wrote, read and saved again.
OLD_FM = {
    "old-channel-emission-wavelength-number": (
        ChannelSettings,
        {
            "name": "Red",
            "excitation_wavelength": 550,
            "emission_wavelength": 610,
            "power": 0.1,
            "exposure_time": 0.2,
            "color": "red",
            "gain": 0.0,
        },
    ),
    "old-channel-emission-wavelength-text": (
        ChannelSettings,
        {
            "name": "Fluo",
            "excitation_wavelength": 550,
            "emission_wavelength": "Fluorescence",
            "power": 0.1,
            "exposure_time": 0.2,
        },
    ),
    "old-channel-emission-wavelength-none": (
        ChannelSettings,
        {"name": "Refl", "excitation_wavelength": 550, "emission_wavelength": None},
    ),
    "old-autofocus-coarse-fine": (
        AutoFocusSettings,
        {
            "coarse_range": 40e-6,
            "coarse_step": 4e-6,
            "coarse_enabled": True,
            "fine_range": 8e-6,
            "fine_step": 1e-6,
            "fine_enabled": False,
            "method": "laplacian",
            "channel_name": "Reflection Channel",
        },
    ),
    "old-autofocus-n-steps": (
        AutoFocusSettings,
        {"method": "sobel", "passes": [{"n_steps": 10, "step_size": 2e-6}]},
    ),
    "old-autofocus-no-passes": (AutoFocusSettings, {"passes": []}),
    "old-camera-rotate-180": (CameraSettings, {"transform": "rotate-180"}),
    "old-camera-rotate-90": (CameraSettings, {"gain": 0.2, "transform": "rotate-90"}),
    "old-overview-int-autofocus-mode": (
        OverviewParameters,
        {"rows": 2, "cols": 2, "autofocus_mode": 2},
    ),
    "old-overview-numpy-mask": (
        OverviewParameters,
        {"rows": 1, "cols": 2, "tile_mask": [[np.bool_(True), np.bool_(False)]]},
    ),
    "old-fm-configuration-no-camera": (
        FluorescenceConfiguration,
        {
            "channel_settings": [{"name": "Refl", "emission_wavelength": None}],
            "z_parameters": {},
            "overview_parameters": {},
            "camera_settings": None,
            "autofocus_settings": {},
        },
    ),
}


@pytest.mark.parametrize("case", sorted(OLD_FM))
def test_an_old_fm_shape_saves_again_as_pinned(case):
    cls, data = OLD_FM[case]
    _check(case, cls.from_dict(data).to_dict())


def test_the_default_fm_configuration_saves_again_as_pinned():
    data = yaml.safe_load(
        (CONFIGURATIONS / "fm" / "fm-configuration-default.yaml").read_text()
    )
    _check(
        "fm-configuration-default", FluorescenceConfiguration.from_dict(data).to_dict()
    )


FM_IMAGES = sorted((FIXTURES / "fm_legacy").glob("*.ome.tiff"))


@pytest.mark.parametrize("path", FM_IMAGES, ids=[p.name for p in FM_IMAGES])
def test_an_old_fm_image_metadata_saves_again_as_pinned(path):
    image = FluorescenceImage.load(str(path))
    _check(f"fm-image-metadata-{path.name.split('.')[0]}", image.metadata.to_dict())


def test_a_numpy_tile_mask_saves_as_plain_bools():
    mask = np.array([[True, False], [False, True]])
    overview = OverviewParameters(rows=2, cols=2, tile_mask=[list(row) for row in mask])

    saved = overview.to_dict()

    assert saved["tile_mask"] == [[True, False], [False, True]]
    assert all(type(v) is bool for row in saved["tile_mask"] for v in row)
    yaml.safe_dump(saved)


# ---- holders, sessions, alignment and the rest of fibsem.structures ----------

GRID = SampleGrid(name="grid-01", description="cryo", radius=1.4e-3)
CALIBRATION = SlotCalibration(
    orientation="SEM",
    pre_tilt=35.0,
    rotation_reference=10.0,
    captured_at="2026-10-01T10:00:00+10:00",
    fibsem_version="0.6.0",
)
SLOT = GridSlot(
    name="Slot-01",
    index=0,
    position=FibsemStagePosition(name="Slot-01", x=1e-3, y=2e-3, z=0.0, r=0.0, t=0.0),
    loaded_grid=GRID,
    calibration=CALIBRATION,
)
BUILT_REST = {
    "sample-grid": GRID,
    "slot-calibration": CALIBRATION,
    "grid-slot": SLOT,
    "sample-holder": SampleHolder(
        pre_tilt=35.0,
        name="Shuttle",
        description="two slots",
        capacity=2,
        slots={"Slot-01": SLOT, "Slot-02": GridSlot(name="Slot-02", index=1)},
    ),
    "session-info": SessionInfo(
        recorded_at=datetime.fromisoformat("2026-10-01T10:00:00+10:00"),
        system=SystemInfo(
            name="Arctis",
            manufacturer="Thermo",
            fibsem_version="0.6.0",
            fibsem_revision="v0.6.0-1-gabcdef0",
        ),
        user=FibsemUser(name="operator", email="op@example.com"),
        plugins={"fibsem-plugin-example": "0.1.0"},
        utc_offset="+10:00",
        zone_name="AEST",
    ),
    "milling-alignment": MillingAlignment(
        enabled=False, rect=RECTANGLE, steps=2, imaging=IMAGE, full_frame_below=1e-10
    ),
    "reference-image-parameters": ReferenceImageParameters(
        imaging=IMAGE, field_of_view1=80e-6, acquire_fib=False, acquire_image2=False
    ),
    "manipulator-system-settings": ManipulatorSystemSettings(enabled=False),
}


@pytest.mark.parametrize("case", sorted(BUILT_REST))
def test_a_built_rest_structure_saves_as_pinned_and_reads_back(case):
    value = BUILT_REST[case]
    saved = value.to_dict()
    _check(case, saved)
    assert _plain(type(value).from_dict(_plain(saved)).to_dict()) == _plain(saved)


OLD_REST = {
    "old-overview-use-focus-stack": (
        OverviewAcquisitionSettings,
        {"use_focus_stack": True},
    ),
    "old-overview-mode-in-autofocus": (
        OverviewAcquisitionSettings,
        {"autofocus_settings": {"mode": "EACH_ROW"}},
    ),
    "old-overview-no-autocontrast-mode": (
        OverviewAcquisitionSettings,
        {"image_settings": {"autocontrast": True}, "tile_order": "serpentine"},
    ),
    "old-overview-empty": (OverviewAcquisitionSettings, {}),
    "old-alignment-empty": (MillingAlignment, {}),
    "old-slot-calibration-blank-time": (
        SlotCalibration,
        {
            "orientation": "SEM",
            "pre_tilt": 35,
            "rotation_reference": 0,
            "captured_at": "",
            "fibsem_version": "built-in",
        },
    ),
    "old-slot-calibration-naive-time": (
        SlotCalibration,
        {
            "orientation": "FIB",
            "pre_tilt": 35.0,
            "rotation_reference": 0.0,
            "captured_at": "2026-09-01T10:00:00",
            "fibsem_version": "0.5.3",
        },
    ),
    "old-holder-no-pre-tilt": (
        SampleHolder,
        {
            "name": "Old Shuttle",
            "capacity": 2,
            "slots": {
                "Slot-01": {
                    "name": "Slot-01",
                    "index": 0,
                    "position": {
                        "name": "Slot-01",
                        "x": 0.001,
                        "y": 0.0,
                        "z": 0.0,
                        "r": 0.0,
                        "t": 0.0,
                        "coordinate_system": "RAW",
                    },
                    "loaded_grid": None,
                }
            },
        },
    ),
    "old-session-posix": (
        SessionInfo,
        {"recorded_at": 1760000000.0, "system": None, "user": None},
    ),
}


@pytest.mark.parametrize("case", sorted(OLD_REST))
def test_an_old_rest_shape_saves_again_as_pinned(case):
    cls, data = OLD_REST[case]
    _check(case, cls.from_dict(data).to_dict())


# ---- milling stages, patterns, strategies and monitors ------------------------


def _stage(**overrides):
    from fibsem.milling.base import FibsemMillingStage, get_strategy_config
    from fibsem.milling.monitors.fm_intensity import FmIntensityConfig
    from fibsem.milling.monitors.sem_peek import SemPeekConfig
    from fibsem.milling.patterning import get_pattern
    from fibsem.structures import FibsemMillingSettings

    fields = dict(
        name="Rough Mill",
        num=1,
        milling=FibsemMillingSettings(
            milling_current=2e-9, hfw=80e-6, milling_channel=BeamType.ION
        ),
        pattern=get_pattern("Trench", {"width": 12e-6, "depth": 2e-6}),
        strategy=get_strategy_config(
            "CoincidenceMilling",
            {"config": {"timeout": 120, "intensity_drop_fraction": 0.4}},
        ),
        alignment=MillingAlignment(enabled=True, rect=RECTANGLE),
        imaging=IMAGE,
        monitors=[FmIntensityConfig(drop_fraction=0.3), SemPeekConfig()],
    )
    fields.update(overrides)
    return FibsemMillingStage(**fields)


def _patterns():
    from fibsem.milling.patterning import BUILTIN_PATTERNS
    from fibsem.milling.patterning.patterns.polygon import PolygonPattern

    # Built-in only: a test may register its own pattern in the same process.
    patterns = {name: cls() for name, cls in BUILTIN_PATTERNS.items()}
    # Not registered, so read back by its class.
    patterns["PolygonPattern"] = PolygonPattern(
        vertices=np.array([[0.0, 0.0], [1e-6, 0.0], [0.0, 1e-6]])
    )
    return patterns


def test_a_milling_stage_saves_as_pinned_and_reads_back():
    from fibsem.milling.base import FibsemMillingStage

    stage = _stage()
    saved = stage.to_dict()
    _check("milling-stage", saved)
    _check("milling-stage-short", stage.to_dict(short=True))
    again = FibsemMillingStage.from_dict(_plain(saved))
    assert again == stage
    assert _plain(again.to_dict()) == _plain(saved)


@pytest.mark.parametrize("name", sorted(_patterns()))
def test_every_pattern_saves_as_pinned_and_reads_back(name):
    from fibsem.milling.patterning.base import BasePattern

    pattern = _patterns()[name]
    saved = pattern.to_dict()
    _check(f"milling-pattern-{name}", saved)
    reader = type(pattern) if name == "PolygonPattern" else BasePattern
    assert _plain(reader.from_dict(_plain(saved)).to_dict()) == _plain(saved)


OLD_STAGES = {
    # No alignment block: alignment off, as a saved alignment without keys reads.
    "old-stage-no-alignment": {
        "name": "Mill",
        "milling": {"milling_current": 1e-9, "hfw": 1e-4},
        "pattern": {"name": "rectangle", "width": 1e-5},
    },
    # A legacy `strategy: {}` is Standard.
    "old-stage-empty-strategy": {
        "name": "Mill",
        "milling": {},
        "pattern": {"name": "Rectangle"},
        "strategy": {},
    },
    # A strategy and a monitor not installed here: Standard, and no monitor.
    "old-stage-unknown-plugins": {
        "name": "Mill",
        "milling": {},
        "pattern": {"name": "Rectangle"},
        "strategy": {"name": "NotInstalled", "config": {"passes": 3}},
        "monitors": [
            {"name": "NotInstalled", "config": {}},
            {"name": "SEM Peek", "config": {"hfw": 1e-4}},
        ],
    },
}


@pytest.mark.parametrize("case", sorted(OLD_STAGES))
def test_an_old_stage_shape_saves_again_as_pinned(case):
    from fibsem.milling.base import FibsemMillingStage

    _check(case, FibsemMillingStage.from_dict(OLD_STAGES[case]).to_dict())


# ---- task protocols and experiments --------------------------------------------

PROTOCOLS = TESTS.parent / "fibsem" / "applications" / "autolamella" / "protocol"
PROTOCOL_FILES = {
    f"protocol-{folder}-{path.stem}": path
    for folder, paths in (
        (
            "shipped",
            [
                PROTOCOLS / "task-protocol.yaml",
                PROTOCOLS / "task-protocol-waffle.yaml",
                PROTOCOLS / "development" / "task-protocol-coincidence.yaml",
            ],
        ),
        ("v0.5.2", sorted((FIXTURES / "protocol_v0_5_2").glob("*.yaml"))),
    )
    for path in paths
}


def _task_protocol(path=PROTOCOLS / "task-protocol.yaml"):
    from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol

    protocol = AutoLamellaTaskProtocol.from_dict(yaml.safe_load(path.read_text()))
    protocol.id = "pinned-protocol-id"
    protocol.grid_tasks.id = "pinned-grid-protocol-id"
    return protocol


@pytest.mark.parametrize("case", sorted(PROTOCOL_FILES))
def test_a_task_protocol_file_saves_again_as_pinned_and_reads_back(case):
    from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol

    saved = _task_protocol(PROTOCOL_FILES[case]).to_dict()
    _check(case, saved)
    again = AutoLamellaTaskProtocol.from_dict(_plain(saved)).to_dict()
    assert _plain(again) == _plain(saved)


PRE_TASK_PROTOCOLS = {
    "protocol-legacy-on-grid": PROTOCOLS / "legacy" / "protocol-on-grid.yaml",
    "protocol-legacy-trench": PROTOCOLS / "legacy" / "protocol-trench.yaml",
    "protocol-legacy-waffle": PROTOCOLS / "legacy" / "protocol-waffle.yaml",
    "protocol-legacy-flat-on-grid": FIXTURES
    / "protocol_pre_task"
    / "protocol-flat-on-grid.yaml",
}


@pytest.mark.parametrize("case", sorted(PRE_TASK_PROTOCOLS))
def test_a_pre_task_protocol_converts_as_pinned(case):
    _check(case, _task_protocol(PRE_TASK_PROTOCOLS[case]).to_dict())


def _experiment():
    from fibsem.applications.autolamella.structures import Experiment, Lamella

    experiment = Experiment(path=Path("/data"), name="pinned-experiment")
    experiment.id = "pinned-experiment-id"
    experiment.created_at = datetime.fromisoformat("2026-10-20T09:00:00+10:00")
    experiment.task_protocol = _task_protocol()
    for number in (1, 2):
        name = f"Lamella-{number:02d}"
        lamella = Lamella(
            path=Path("/data/pinned-experiment") / name,
            number=number,
            petname=name,
            id=f"lamella-{number}",
            task_config=deepcopy(experiment.task_protocol.task_config),
        )
        lamella.task_state.task_id = f"task-{number}"
        lamella.task_state.start_timestamp = experiment.created_at
        experiment.positions.append(lamella)
    return experiment


def test_an_experiment_with_lamellae_saves_as_pinned_and_reads_back():
    from fibsem.applications.autolamella.structures import Experiment

    saved = _experiment().to_dict()
    _check("experiment-with-lamellae", _posix(saved))
    again = Experiment.from_dict(_plain(saved)).to_dict()
    assert _posix(again) == _posix(saved)


def _grid_protocol():
    from fibsem.workflows.grid.protocol import GridTaskProtocol

    data = yaml.safe_load((PROTOCOLS / "task-protocol.yaml").read_text())
    protocol = GridTaskProtocol.from_dict(data["grid_tasks"])
    protocol.id = "pinned-grid-protocol-id"
    return protocol


def _volume_protocol():
    from fibsem.applications.volume.protocol import VolumeProtocol

    protocol = VolumeProtocol.default()
    protocol.id = "pinned-volume-protocol-id"
    return protocol


@pytest.mark.parametrize(
    "case, build",
    [("grid-protocol", _grid_protocol), ("volume-protocol", _volume_protocol)],
)
def test_an_engine_protocol_saves_as_pinned_and_reads_back(case, build):
    protocol = build()
    saved = protocol.to_dict()
    _check(case, saved)
    assert _plain(type(protocol).from_dict(_plain(saved)).to_dict()) == _plain(saved)


def _unversioned(document):
    document = dict(document)
    assert len(document.pop("version")) == 16
    return document


def test_the_app_servers_protocol_documents_are_as_pinned():
    from fibsem.applications.autolamella.server import AutoLamellaAppContext
    from tests.run_state import stub_session

    session = stub_session()
    session.experiment = _experiment()
    context = AutoLamellaAppContext(session)
    _check("app-protocol", _posix(context.protocol()))
    # These versions hash the configs' paths, which differ by OS.
    _check(
        "app-protocol-task-config",
        _posix(_unversioned(context.protocol_task_config("Mill Fiducial"))),
    )
    _check(
        "app-item-task-config",
        _posix(_unversioned(context.item_task_config("Lamella-01", "Mill Fiducial"))),
    )
    _check("app-grid-task-config", _posix(context.grid_task_config("SEM Overview")))


# ---- the shapes FORMAT_VERSION 2 wrote -----------------------------------------
# Saved by the build before task configs went flat, and read by this one into the
# shapes pinned above.

V2 = FIXTURES / "saved_shapes_v2"


def _v2(name):
    return json.loads((V2 / f"{name}.json").read_text())


def test_a_v2_protocol_reads_as_the_current_one():
    from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol

    saved = AutoLamellaTaskProtocol.from_dict(_v2("protocol-shipped-task-protocol"))
    assert _plain(saved.to_dict()) == _plain(
        _task_protocol(PROTOCOLS / "task-protocol.yaml").to_dict()
    )


def test_a_v2_experiment_reads_as_the_current_one():
    from fibsem.applications.autolamella.structures import Experiment

    loaded = Experiment.from_dict(_v2("experiment-with-lamellae"))
    assert _posix(loaded.to_dict()) == _posix(_experiment().to_dict())


@pytest.mark.parametrize(
    "case, build",
    [("grid-protocol", _grid_protocol), ("volume-protocol", _volume_protocol)],
)
def test_a_v2_engine_protocol_reads_as_the_current_one(case, build):
    protocol = build()
    loaded = type(protocol).from_dict(_v2(case))
    assert _plain(loaded.to_dict()) == _plain(protocol.to_dict())


def test_each_upgrade_step_moves_one_shape():
    from fibsem.codec.upgrades import protocol as protocol_upgrade

    assert protocol_upgrade.task_config(
        {"task_name": "a", "parameters": {"depth": 1, "task_name": "b"}}
    ) == {"depth": 1, "task_name": "a"}, "a key already at the top level wins"
    assert protocol_upgrade.run_plan({"tasks": ["a", None, {"name": "b"}]}) == {
        "tasks": [{"name": "a"}, {"name": "", "required": False}, {"name": "b"}]
    }
    assert protocol_upgrade.task_protocol(
        {"_id": "x", "tasks": {"a": {"parameters": {"n": 1}}}, "order": ["a"]}
    ) == {
        "id": "x",
        "task_config": {"a": {"n": 1}},
        "workflow_config": {"tasks": [{"name": "a"}]},
    }
    old = {
        "tasks": {"a": {}, "b": {"attention": "automated"}, "c": {"requires": ["a"]}},
        "workflow": {
            "tasks": [
                {"name": "a", "supervise": True, "requires": []},
                {"name": "b", "attention": "supervised", "requires": ["a"]},
                {"name": "c"},
                {"name": "ghost", "attention": "supervised"},
            ]
        },
    }
    new = protocol_upgrade.autolamella_protocol(old)
    assert new["task_config"] == {"a": {}, "b": {}, "c": {}}
    assert new["workflow_config"]["tasks"] == [
        {"name": "a", "attention": "supervised", "requires": []},
        {"name": "b", "attention": "supervised", "requires": ["a"]},
        {"name": "c", "requires": ["a"]},
        {"name": "ghost", "attention": "supervised"},
    ], "the plan entry's own value wins; a config's moves onto its entry"
    current = deepcopy(new)
    assert protocol_upgrade.autolamella_protocol(current) == new, (
        "the current shape is kept"
    )


def test_an_upgrade_leaves_what_it_is_given_alone():
    from fibsem.codec.upgrades import protocol as protocol_upgrade

    for name, step in (
        ("protocol-shipped-task-protocol", protocol_upgrade.autolamella_protocol),
        ("experiment-with-lamellae", protocol_upgrade.experiment),
        ("grid-protocol", protocol_upgrade.task_protocol),
    ):
        data = _v2(name)
        before = deepcopy(data)
        step(data)
        assert data == before, name


# ---- correlation -----------------------------------------------------------------


def _correlation_points():
    from fibsem.correlation.structures import (
        Coordinate,
        CorrelationInputData,
        PointProvenance,
        PointStatus,
        PointType,
        PointXYZ,
    )

    def at(x, y, z, kind, status="", provenance=""):
        return Coordinate(
            PointXYZ(x, y, z),
            kind,
            status=status,
            provenance=provenance,
            suggested=True,
        )

    return CorrelationInputData(
        fib_coordinates=[
            at(
                10.0, 20.0, 0.0, PointType.FIB, PointStatus.PLACED, PointProvenance.USER
            ),
            at(30.5, 40.25, 0.0, PointType.FIB, PointStatus.FITTED),
        ],
        fm_coordinates=[
            at(1.0, 2.0, 3.0, PointType.FM, PointStatus.PREDICTED, "projected"),
            at(4.0, 5.0, 6.5, PointType.FM),
        ],
        poi_coordinates=[at(7.0, 8.0, 9.0, PointType.POI)],
        fm_surface_coordinate=at(0.5, 0.5, 2.0, PointType.SURFACE_FM),
        ri_pre_correction_factor=1.5,
        ri_pre_correction_params={"tilt_deg": 15.0, "source": "computed"},
        stored_fib_image_shape=(512, 768),
        stored_fib_image_pixel_size=5e-9,
        stored_fm_pixel_size_z=2e-7,
        stored_fib_image_filename="ref_fib.tif",
        stored_fm_image_filename="fm_stack.ome.tiff",
    )


def _correlation_result():
    from fibsem.correlation.structures import (
        CorrelationPointOfInterest,
        CorrelationResult,
        PointXYZ,
    )

    poi = CorrelationPointOfInterest(
        image_px=Point(100.0, 200.0),
        px=Point(-284.0, 56.0),
        px_m=Point(-1.42e-6, 2.8e-7),
    )
    return CorrelationResult(
        poi=[poi],
        scale=1.25,
        rotation_eulers=[10.0, 20.0, 30.0],
        rotation_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation=[1.0, 2.0, 3.0],
        translation_custom=[4.0, 5.0, 6.0],
        rms_error=0.5,
        mean_absolute_error=[0.25, 0.75],
        delta_2d=[Point(0.1, -0.2)],
        reprojected_3d=[PointXYZ(1.0, 2.0, 0.0)],
        input_data=_correlation_points(),
        refractive_index_correction_factor=1.5,
        refractive_index_correction_mode="pre",
        updated_at=datetime.fromisoformat("2026-10-20T10:30:00+00:00"),
        fm_z_scale=2.5,
        seed={"source": "metadata"},
        branch_check={"ratio": 3.0},
        diagnostics={"rms_um": 0.1},
        placement_offset=[0.5, -0.5],
    )


def _fit_evidence():
    from fibsem.correlation.verdict import FitEvidence, PairDiagnostic

    return FitEvidence(
        rms_um=0.2,
        pairs=[PairDiagnostic(0, 0.1, 0.3), PairDiagnostic(2, 0.2, 0.4)],
        mirror_ratio=4.0,
        depth_span_um=2.5,
        scale_ratio=1.01,
        n_pairs=2,
        n_accepted=1,
        poi_jackknife_um=0.05,
        worst=1,
        suggested_z=12.0,
        rotation=np.eye(3),
    )


def _correlation_config():
    from fibsem.correlation.config import CorrelationConfig, FitSettings, RISettings

    return CorrelationConfig(
        fit=FitSettings(fib_method="Gaussian", fm_poi_channel="Cy5"),
        ri=RISettings(na=0.8, mode="post"),
        auto_rerun=True,
        delete_fiducial="pair",
    )


def _correlation_built():
    from fibsem.correlation.config import CorrelationConfig
    from fibsem.correlation.structures import CorrelationResult, CorrelationState
    from fibsem.correlation.verdict import FitEvidence

    return {
        "correlation-state": (
            CorrelationState,
            CorrelationState(
                input_data=_correlation_points(), result=_correlation_result()
            ),
        ),
        "correlation-state-no-result": (
            CorrelationState,
            CorrelationState(input_data=_correlation_points()),
        ),
        "correlation-result-empty": (
            CorrelationResult,
            CorrelationResult(
                updated_at=datetime.fromisoformat("2026-10-20T10:30:00+00:00")
            ),
        ),
        "correlation-fit-evidence": (FitEvidence, _fit_evidence()),
        "correlation-config": (CorrelationConfig, _correlation_config()),
        "correlation-config-default": (CorrelationConfig, CorrelationConfig()),
    }


@pytest.mark.parametrize("case", sorted(_correlation_built()))
def test_a_correlation_structure_saves_as_pinned_and_reads_back(case):
    cls, built = _correlation_built()[case]
    saved = built.to_dict()
    _check(case, saved)
    assert _plain(cls.from_dict(_plain(saved)).to_dict()) == _plain(saved)


def test_a_correlation_config_section_absent_reads_as_the_defaults():
    from fibsem.correlation.config import CorrelationConfig, FitSettings, RISettings

    assert CorrelationConfig.from_dict(None) == CorrelationConfig()
    assert CorrelationConfig.from_dict({"fit": None, "ri": None}) == CorrelationConfig()
    assert FitSettings.from_dict(None) == FitSettings()
    assert RISettings.from_dict(None) == RISettings()
