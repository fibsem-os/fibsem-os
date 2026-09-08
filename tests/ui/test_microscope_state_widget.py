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
from fibsem.ui.tokens import (
    CURRENT_POSITION_COLOUR,
    SAVED_POSITION_COLOUR,
    TEXT_MUTED_COLOR,
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


def _colour_of(grid, label: str, column: int) -> str:
    layout = grid._layout
    for row in range(layout.rowCount()):
        first = layout.itemAtPosition(row, 0)
        if first is not None and first.widget().text() == label:
            return layout.itemAtPosition(row, column).widget().styleSheet()
    raise AssertionError(f"no row {label!r}")


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
    """So shutting a section loses the detail, not the fact that the FIB is at 30 kV."""
    widget = MicroscopeStateWidget()
    widget.set_state(_state())

    assert "30.00 kV" in widget.panel_ion._title_label.text()
    assert "20 pA" in widget.panel_ion._title_label.text()


def test_the_beams_start_collapsed_and_the_stage_does_not_collapse(qapp):
    """A pose row in a list cannot expand to forty lines, and a pose that will not say
    where it is has no reason to be in the list."""
    widget = MicroscopeStateWidget()

    assert widget.panel_stage._collapsible is False
    assert widget.panel_electron._btn_collapse.isChecked() is False
    assert widget.panel_ion._btn_collapse.isChecked() is False


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
    assert widget.label_timestamp.text() == ""
    assert widget.label_delta.isHidden() is True


# ---------------------------------------------------------------------------
# Comparing two states
# ---------------------------------------------------------------------------


def test_comparing_shows_both_columns(qapp):
    widget = MicroscopeStateWidget()
    widget.set_state(_state(x=42e-6), title="MILLING")
    widget.set_reference(_state(x=0.0))

    rows = _grid_text(widget.grid_stage)
    assert rows["X"] == ["0 nm", "42.00 µm"], "saved column first, then live"
    assert widget.label_mode.text() == "saved vs live"


def test_only_the_rows_that_differ_are_coloured(qapp):
    """Colouring every value would hide the two that matter.

    The colours are `tokens.py`'s own: CURRENT_POSITION_COLOUR is documented as "where
    the stage is now" and SAVED_POSITION_COLOUR as "a marked position", which is what
    the overview canvases already draw stage markers in.
    """
    widget = MicroscopeStateWidget()
    widget.set_state(_state(x=42e-6), title="MILLING")
    widget.set_reference(_state(x=0.0))

    assert SAVED_POSITION_COLOUR in _colour_of(widget.grid_stage, "X", 1)
    assert CURRENT_POSITION_COLOUR in _colour_of(widget.grid_stage, "X", 2)

    # Y did not move.
    assert TEXT_MUTED_COLOR in _colour_of(widget.grid_stage, "Y", 1)
    assert TEXT_MUTED_COLOR in _colour_of(widget.grid_stage, "Y", 2)


def test_two_identical_states_colour_nothing(qapp):
    widget = MicroscopeStateWidget()
    widget.set_state(_state())
    widget.set_reference(_state())

    for label in ("X", "Y", "Z", "R", "T"):
        assert TEXT_MUTED_COLOR in _colour_of(widget.grid_stage, label, 1), label


def test_values_are_compared_as_rendered_not_as_floats(qapp):
    """Two records holding 2.0e-11 and 2.0000001e-11 both read "20 pA".

    Marking that row as a difference would report a change the panel cannot show.
    """
    widget = MicroscopeStateWidget()
    widget.set_state(_state(ion_current=2.0000001e-11))
    widget.set_reference(_state(ion_current=2.0e-11))

    assert _grid_text(widget.grid_ion)["Current"] == ["20 pA", "20 pA"]
    assert TEXT_MUTED_COLOR in _colour_of(widget.grid_ion, "Current", 1)


def test_the_separation_is_reported_in_three_dimensions(qapp):
    """One number to act on. 3-4-5 in micrometres, so the arithmetic is checkable."""
    widget = MicroscopeStateWidget()
    widget.set_state(_state(x=3e-6, y=4e-6, z=0.0))
    widget.set_reference(_state(x=0.0, y=0.0, z=0.0))

    # `isHidden`, not `isVisible`: a child of a widget that was never shown is never
    # visible, so `isVisible() is False` would pass here whatever the code did.
    assert widget.label_delta.isHidden() is False
    assert "5.00 µm" in widget.label_delta.text()


def test_dropping_the_reference_stops_comparing(qapp):
    widget = MicroscopeStateWidget()
    widget.set_state(_state(x=42e-6))
    widget.set_reference(_state(x=0.0))
    widget.set_reference(None)

    assert _grid_text(widget.grid_stage)["X"] == ["42.00 µm"]
    assert widget.label_delta.isHidden() is True
    assert widget.label_mode.text() == ""


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
