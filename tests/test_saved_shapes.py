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
from pathlib import Path

import pytest
import yaml

from fibsem import structures
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemDetectorSettings,
    FibsemImageMetadata,
    FibsemManipulatorPosition,
    FibsemRectangle,
    FibsemStagePosition,
    FocusStackSettings,
    ImageSettings,
    MicroscopeSettings,
    MicroscopeState,
    OverviewAcquisitionSettings,
    Point,
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
    data = yaml.safe_load(CONFIGURATION_FILES[case].read_text())
    _check(case, MicroscopeSettings.from_dict(data).to_dict())


METADATA_FILES = {
    f"image-metadata-{path.stem}": path
    for path in sorted((FIXTURES / "metadata").glob("*.json"))
}


@pytest.mark.parametrize("case", sorted(METADATA_FILES))
def test_image_metadata_saves_again_as_pinned(case):
    data = json.loads(METADATA_FILES[case].read_text())
    _check(case, FibsemImageMetadata.from_dict(data).to_dict())
