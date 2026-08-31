"""The workflow's POI marker is drawn like every other POI marker (FIB-582).

The same concept is drawn in three places -- the workflow's Select Position step, the
lamella protocol editor, and the default-config preview -- and the workflow one had
drifted: left to `PointsSpec`'s defaults it came out at size 18 with a 2.0 edge, a
visibly fatter cross than the thin one the other two draw, and with no `legend_label` it
was the only one missing from the canvas legend.

Driven by calling `QtResponder._pick_poi` unbound against a stub, so the spec it
builds is pinned without a window or a running question: the marker drawing moved
there with the PickPOI conversion, and the image now arrives in the request rather
than being read off the host.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import types

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.qt_responder import QtResponder
from fibsem.applications.autolamella.workflows.interaction import PickPOI
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
    ImageSettings,
    MicroscopeState,
    Point,
)

# What the protocol editor asks for (autolamella_lamella_protocol_editor.py), and what
# the config preview's PointOverlay uses for its edge. Restated rather than imported:
# both are inline literals at their call sites, so this is a statement of the intended
# house style for a POI, not a live comparison.
POI_SIZE = 14
POI_EDGE_WIDTH = 1.2
POI_LEGEND_LABEL = "Point of Interest"


class _Controller:
    """Records what the method asks the quad view to draw."""

    def __init__(self):
        self.overlays = {}
        self.armed = None
        self.fib_canvas = types.SimpleNamespace(set_hint=lambda hint: None)

    def set_overlay(self, beam, spec):
        self.overlays[(beam, spec.id)] = spec

    def arm_overlay(self, beam, overlay_id, label=None, icon=None):
        self.armed = (beam, overlay_id)


def _image(shape=(64, 64), pixel_size=1e-8):
    """The image the request carries — what the marker's coordinates mean against."""
    image = FibsemImage(
        data=np.zeros(shape, dtype=np.uint8),
        metadata=FibsemImageMetadata(
            image_settings=ImageSettings(
                resolution=(shape[1], shape[0]), beam_type=BeamType.ION
            ),
            pixel_size=Point(pixel_size, pixel_size),
            microscope_state=MicroscopeState(),
        ),
    )
    return image


def _poi_spec(initial_poi=None):
    controller = _Controller()
    stub = types.SimpleNamespace(
        _view_controller=lambda: controller,
        _park_question=lambda *a, **k: None,
    )
    QtResponder._pick_poi(
        stub, PickPOI(image=_image(), initial=initial_poi), future=None
    )
    return controller.overlays[(BeamType.ION, "poi")]


def test_the_poi_marker_is_the_thin_cross_the_other_editors_draw():
    spec = _poi_spec()

    assert spec.marker == "+"
    assert spec.size == POI_SIZE
    assert spec.edge_width == POI_EDGE_WIDTH


def test_the_poi_marker_names_itself_in_the_legend():
    """Without this the workflow's canvas showed a magenta cross and no key to it."""
    assert _poi_spec().legend_label == POI_LEGEND_LABEL


def test_the_poi_marker_is_magenta_selected_or_not():
    """It is a single dragged point, so a distinct selected colour only makes it flicker."""
    spec = _poi_spec()

    assert spec.color == "magenta"
    assert spec.selected_color == "magenta"


def test_the_marker_stays_move_only():
    """Formatting only -- the POI is one point the user drags, never adds to or deletes."""
    spec = _poi_spec()

    assert spec.add_on_right_click is False
    assert spec.removable is False


def test_the_marker_starts_at_the_image_centre_when_there_is_no_poi_yet():
    assert _poi_spec().points == [(32.0, 32.0)]
