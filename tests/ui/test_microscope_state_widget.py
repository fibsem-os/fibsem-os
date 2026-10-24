"""The widget that renders a `MicroscopeState`.

Built from real `MicroscopeState` objects rather than stand-ins: the record is a plain
dataclass, so there is nothing to fake, and a stub would let a field be renamed under
the widget without a test noticing.

`qapp` is the shared offscreen QApplication from `tests/conftest.py`; this project has
no pytest-qt, so there is no `qtbot` and signals are checked by connecting to them.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtWidgets import QLabel

from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemDetectorSettings,
    FibsemStagePosition,
    MicroscopeState,
)
from fibsem.ui.widgets.microscope_state_widget import MicroscopeStateWidget
from fibsem.utils import NOT_AVAILABLE


def _state(x=0.0, y=0.0, z=0.0, r=0.0, t=0.2094, ion_current=2e-11) -> MicroscopeState:
    """A realistic state: the Demo microscope's MILLING pose, 12 degrees of tilt."""
    return MicroscopeState(
        timestamp=1788000000.0,
        stage_position=FibsemStagePosition(
            x=x, y=y, z=z, r=r, t=t, coordinate_system="RAW"
        ),
        electron_beam=BeamSettings(
            beam_type=BeamType.ELECTRON,
            voltage=2000.0,
            beam_current=1e-10,
            hfw=150e-6,
            working_distance=4e-3,
            resolution=[1536, 1024],
            dwell_time=1e-6,
            scan_rotation=0.0,
        ),
        ion_beam=BeamSettings(
            beam_type=BeamType.ION,
            voltage=30000.0,
            beam_current=ion_current,
            hfw=150e-6,
            working_distance=16.5e-3,
            resolution=[1536, 1024],
            dwell_time=1e-6,
            scan_rotation=0.0,
        ),
        electron_detector=FibsemDetectorSettings(type="ETD", mode="SecondaryElectrons"),
        ion_detector=FibsemDetectorSettings(type="ETD", mode="SecondaryElectrons"),
    )


def _grid_text(grid) -> dict:
    """{row label: [value, ...]} for a `_ValueGrid`."""
    layout = grid._layout
    out = {}
    for row in range(layout.rowCount()):
        cells = []
        for column in range(layout.columnCount()):
            item = layout.itemAtPosition(row, column)
            cells.append(item.widget().text() if item is not None else None)
        if cells[0]:
            out[cells[0]] = [c for c in cells[1:] if c is not None]
    return out


# ---------------------------------------------------------------------------
# Rendering one state
# ---------------------------------------------------------------------------


def test_it_renders_the_stage_position(qapp):
    widget = MicroscopeStateWidget()
    widget.set_state(_state(x=42e-6, y=-13e-6), title="MILLING")

    rows = _grid_text(widget.grid_stage)
    assert rows["X"] == ["42.00 µm"]
    assert rows["Y"] == ["-13.00 µm"]
    assert rows["T"] == ["12.0°"]
    assert widget.label_title.text() == "MILLING"


def test_the_beams_are_rendered_too(qapp):
    """The point of the widget: these were recorded at every pose and shown nowhere."""
    widget = MicroscopeStateWidget()
    widget.set_state(_state())

    ion = _grid_text(widget.grid_ion)
    assert ion["Voltage"] == ["30.00 kV"]
    assert ion["Current"] == ["20 pA"]
    assert ion["Resolution"] == ["1536 x 1024"]
    assert ion["Detector"] == ["ETD / SE"]


def test_a_collapsed_beam_still_reports_its_headline(qapp):
    """So shutting a section loses the detail, not the fact that the FIB is at 30 kV.

    The headline is its own right-aligned label in the panel header rather than part
    of the title, so it lands on the same edge as the values it summarises.
    """
    widget = MicroscopeStateWidget()
    widget.set_state(_state())

    assert widget.panel_ion._btn_collapse.isChecked() is False
    assert "30.00 kV" in widget.headline_ion.text()
    assert "20 pA" in widget.headline_ion.text()
    # The title stays the section's name; the numbers are not part of it.
    assert widget.panel_ion._title_label.text() == "FIB"
    assert "font-weight: normal" in widget.headline_ion.styleSheet()


def test_the_beams_are_named_the_way_the_canvases_name_them(qapp):
    """SEM and FIB, not Electron and Ion.

    The enum members are ELECTRON and ION, but every surface an operator reads says
    SEM and FIB -- the quad-view panel titles, the coincidence tabs, the stage
    orientations. A readout beside those canvases should agree with them.
    """
    widget = MicroscopeStateWidget()
    assert widget.panel_electron._title_label.text() == "SEM"
    assert widget.panel_ion._title_label.text() == "FIB"

    widget.set_state(_state())
    assert widget.panel_electron._title_label.text() == "SEM"
    assert widget.panel_ion._title_label.text() == "FIB"

    titles = {label.text() for label in widget.findChildren(QLabel) if label.text()}
    assert not any(t.startswith(("Electron", "Ion")) for t in titles)


def test_the_beams_start_collapsed_and_the_stage_does_not_collapse(qapp):
    """Collapsed keeps the popup short: opening both takes it from 277 to 635 pixels,
    dropped over a row that may already be near the bottom of a panel. The headline on
    each title carries what most glances are after.

    The stage never collapses: a pose that will not say where it is has no reason to
    be on screen.
    """
    widget = MicroscopeStateWidget()

    assert widget.panel_stage._collapsible is False
    assert widget.panel_electron._btn_collapse.isChecked() is False
    assert widget.panel_ion._btn_collapse.isChecked() is False
    assert widget.panel_electron._collapsible is True


def test_a_missing_value_renders_as_a_dash(qapp):
    """`MicroscopeState` is `Optional` throughout, and "not reported" has to stay
    distinguishable from "reported as zero"."""
    widget = MicroscopeStateWidget()
    state = _state()
    state.ion_beam.beam_current = None
    widget.set_state(state)

    assert _grid_text(widget.grid_ion)["Current"] == [NOT_AVAILABLE]


def test_clearing_empties_every_section(qapp):
    widget = MicroscopeStateWidget()
    widget.set_state(_state())
    widget.clear()

    assert _grid_text(widget.grid_stage) == {}
    assert widget.headline_ion.text() == "", "a cleared panel reports no values"
    assert widget.label_timestamp.text() == ""


# ---------------------------------------------------------------------------
# What it refuses to do
# ---------------------------------------------------------------------------


def test_refresh_emits_and_does_nothing_else(qapp):
    """The host owns what a refresh costs, because reading the instrument is a device
    call. Same contract as `stage_position_widget`."""
    widget = MicroscopeStateWidget(show_refresh=True)
    widget.set_state(_state())

    fired = []
    widget.refresh_requested.connect(lambda: fired.append(True))
    widget.button_refresh.click()
    assert fired == [True]

    # The record on screen is untouched: the widget did not go and fetch a new one.
    assert _grid_text(widget.grid_stage)["X"] == ["0 nm"]


def test_a_saved_pose_offers_no_refresh(qapp):
    widget = MicroscopeStateWidget(show_refresh=False)
    assert widget.button_refresh.isHidden() is True


def test_it_holds_no_microscope(qapp):
    """The architectural constraint, asserted rather than trusted to review.

    A widget that can reach a microscope will eventually be made to poll one on a
    paint or a selection change, which is the thing this repository keeps off UI
    event paths.
    """
    widget = MicroscopeStateWidget()
    widget.set_state(_state())

    from fibsem.microscope import FibsemMicroscope

    for name, value in vars(widget).items():
        assert not isinstance(value, FibsemMicroscope), name


def test_there_is_no_fluorescence_section(qapp):
    """Deliberate, and worth pinning so it is not "fixed" by adding an empty one.

    `MicroscopeState` carries only `objective_position` for the FM -- no channel, no
    filter -- and `get_microscope_state` does not populate even that; AutoLamellaUI
    patches it in at the call site. A section that can never fill reads as an
    instrument fault rather than a gap in the record.
    """
    widget = MicroscopeStateWidget()

    titles = {label.text() for label in widget.findChildren(QLabel) if label.text()}
    assert not any("luoresc" in title for title in titles)


# ---------------------------------------------------------------------------
# A stage with no rotation axis
# ---------------------------------------------------------------------------


def test_a_compustage_still_shows_a_rotation_row(qapp):
    """It reads 0.0 degrees, and that is the intended answer.

    A compustage has no rotation axis -- AutoScript's `CompustagePosition` carries
    x, y, z, a -- so `stage_position_from_autoscript` writes `r=0.0` as a literal and
    the record cannot say "this stage does not rotate". Hiding the row would need the
    host to pass a capability the record does not carry, and a saved pose has no
    microscope to ask. One honest zero is the cheaper answer.
    """
    widget = MicroscopeStateWidget()
    widget.set_state(_state(r=0.0))

    rows = _grid_text(widget.grid_stage)
    assert list(rows) == ["X", "Y", "Z", "R", "T"]
    assert rows["R"] == ["0.0°"]
