"""The RI tab reads the input snapshot rather than copies of its fields.

FIB-645: `_RITab.set_result` exploded `input_data` into five parallel
attributes — the FIB surface, its y, the FM surface, the POI list and the armed
factor — every one of which had to be reassigned on every refresh to stay
truthful. It now holds the snapshot and derives all five.

These pin the property that removal buys: whatever `set_result` was handed last
is what the tab reports, with nothing left over from the call before. That was
true of the old code too — it is a pin, not a bug fix — and it is the invariant
a sixth field, added later and assigned in only one of two branches, would break.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationPointOfInterest,
    CorrelationResult,
    PointType,
    PointXYZ,
)
from fibsem.structures import Point


@pytest.fixture
def tab(qapp):
    from fibsem.ui.correlation.widgets.correlation_tab_widget import _RITab

    w = _RITab()
    yield w
    w.deleteLater()


def _coord(x=0.0, y=0.0, z=0.0, pt=PointType.POI) -> Coordinate:
    return Coordinate(point=PointXYZ(x=x, y=y, z=z), point_type=pt)


def _post_data() -> CorrelationInputData:
    """FIB surface at y=200, two POIs, no armed factor."""
    return CorrelationInputData(
        surface_coordinate=_coord(y=200.0, pt=PointType.SURFACE),
        poi_coordinates=[_coord(x=1.0, y=2.0, z=30.0), _coord(x=3.0, y=4.0, z=40.0)],
    )


def _pre_data() -> CorrelationInputData:
    """FM surface at z=10, one POI, factor armed."""
    return CorrelationInputData(
        fm_surface_coordinate=_coord(z=10.0, pt=PointType.SURFACE_FM),
        poi_coordinates=[_coord(x=9.0, y=9.0, z=90.0)],
        ri_pre_correction_factor=1.5,
    )


def _views(tab):
    """Everything the tab derives from the snapshot, in one tuple."""
    return (
        tab._surface_coordinate,
        tab._surface_y,
        tab._fm_surface_coordinate,
        tab._input_poi,
        tab._input_pre_factor,
    )


def test_every_view_comes_from_the_snapshot(tab):
    data = _pre_data()
    tab.set_result(None, input_data=data)

    assert tab._surface_coordinate is None
    assert tab._surface_y is None
    assert tab._fm_surface_coordinate is data.fm_surface_coordinate
    assert tab._input_poi == data.poi_coordinates
    assert tab._input_pre_factor == 1.5


def test_the_surface_datum_is_derived_not_stored(tab):
    """`_surface_y` is the post-mode depth datum. Its only source is the surface
    coordinate, so it cannot be right while the coordinate is wrong."""
    data = _post_data()
    tab.set_result(None, input_data=data)

    assert tab._surface_coordinate is data.surface_coordinate
    assert tab._surface_y == 200.0


def test_no_snapshot_means_no_views(tab):
    tab.set_result(None, input_data=_pre_data())
    tab.set_result(None, input_data=None)

    assert _views(tab) == (None, None, None, [], None)
    assert tab.mode is None


def test_a_new_snapshot_replaces_every_view(tab):
    """The failure the parallel fields invited: one of them not reassigned, so
    the tab keeps describing the previous inputs in one respect only."""
    tab.set_result(None, input_data=_pre_data())
    post = _post_data()

    tab.set_result(None, input_data=post)

    assert tab._fm_surface_coordinate is None       # the FM surface is gone
    assert tab._surface_coordinate is post.surface_coordinate
    assert tab._surface_y == 200.0
    assert tab._input_poi == post.poi_coordinates
    assert tab._input_pre_factor is None            # the armed factor went with it


def test_the_mode_follows_the_snapshot(tab):
    """Which correction the tab offers is read off the surface points, so it has
    to move with them — this is the user-visible half of the previous test."""
    tab.set_result(None, input_data=_post_data())
    assert tab.mode == "post"

    tab.set_result(None, input_data=_pre_data())
    assert tab.mode == "pre"

    tab.set_result(None, input_data=CorrelationInputData())
    assert tab.mode is None


def test_the_preview_table_follows_the_armed_factor(tab):
    """The end the views feed: the pre-mode table previews the correction a run
    would apply, so it must be built from the snapshot's POIs and factor."""
    tab.set_result(None, input_data=_pre_data())

    assert tab._table.rowCount() == 1
    # 10 + (90 - 10) * 1.5 = 130
    assert tab._table.item(0, 3).text() == "130.00"


def test_the_tab_reads_the_live_inputs_not_the_results_own_snapshot(tab):
    """A result carries the inputs it was computed from; the tab is handed the
    inputs as they are now. Conflating them is what makes a stale result look
    current (FIB-315), so the tab must follow the argument, not the result."""
    stale = _post_data()
    result = CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=100.0, y=300.0))],
        input_data=stale,
    )
    live = _pre_data()

    tab.set_result(result, input_data=live)

    assert tab.mode == "pre"
    assert tab._surface_coordinate is None
    assert tab._input_poi == live.poi_coordinates
    assert result.input_data is stale  # untouched
