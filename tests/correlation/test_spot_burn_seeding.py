"""Seeding correlation FIB fiducials from spot-burn coordinates (FIB-259).

Spot burns are stored normalised (0-1, top-left origin, no y-flip) to the FIB
image they were placed on. The correlation widget converts them to absolute FIB
pixels (the frame the canvas and fits use); the protocol editor picks the burns
off the lamella and only for a first correlation. Headless PyQt5, offscreen.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import types

import pytest

pytest.importorskip("PyQt5")

from fibsem.correlation.structures import PointType
from fibsem.structures import FibsemImage, Point


@pytest.fixture(autouse=True)
def _no_lut_download(monkeypatch):
    """Never hit the network for the zeta LUT during widget construction."""
    import fibsem.correlation.ui.widgets.refractive_index_widget as riw

    monkeypatch.setattr(riw, "_ensure_lut", lambda: None)


def _widget():
    from fibsem.correlation.ui.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    return CorrelationTabWidget()


def _fib_image():
    # resolution=(W, H) -> data.shape == (H, W)
    return FibsemImage.generate_blank_image(resolution=(300, 200), hfw=100e-6)


# ---------------------------------------------------------------------------
# widget: normalised -> pixel conversion
# ---------------------------------------------------------------------------


def test_seed_converts_normalised_to_pixels(qapp):
    w = _widget()
    img = _fib_image()
    w.set_fib_image(img)
    h, wd = img.data.shape[:2]  # (200, 300)

    w.seed_fib_fiducials_from_spot_burns([Point(0.25, 0.5), Point(1.0, 0.0)])

    fib = w.data.fib_coordinates
    assert len(fib) == 2
    assert fib[0].point.x == pytest.approx(0.25 * wd)  # 75.0
    assert fib[0].point.y == pytest.approx(0.5 * h)    # 100.0
    assert fib[0].point.z == 0.0
    assert fib[0].point_type is PointType.FIB
    assert fib[0].fitted is False                       # a seed, not an accepted fit
    assert fib[1].point.x == pytest.approx(1.0 * wd)
    assert fib[1].point.y == pytest.approx(0.0)


def test_seed_only_touches_fib_side(qapp):
    w = _widget()
    w.set_fib_image(_fib_image())
    w.seed_fib_fiducials_from_spot_burns([Point(0.1, 0.1), Point(0.2, 0.2)])
    d = w.data
    assert len(d.fib_coordinates) == 2
    assert d.fm_coordinates == []
    assert d.poi_coordinates == []


def test_seed_noop_without_fib_image(qapp):
    w = _widget()  # no set_fib_image
    w.seed_fib_fiducials_from_spot_burns([Point(0.5, 0.5)])
    assert w.data.fib_coordinates == []


def test_seed_noop_with_empty_coordinates(qapp):
    w = _widget()
    w.set_fib_image(_fib_image())
    w.seed_fib_fiducials_from_spot_burns([])
    assert w.data.fib_coordinates == []


# ---------------------------------------------------------------------------
# protocol editor: locate the spot-burn config on a lamella
# ---------------------------------------------------------------------------


def test_spot_burn_coordinates_matches_by_type():
    from fibsem.applications.autolamella.workflows.tasks.spot_burn import (
        SpotBurnFiducialTaskConfig,
    )
    from fibsem.ui.widgets.autolamella_lamella_protocol_editor import (
        AutoLamellaProtocolEditorWidget,
    )

    burns = [Point(0.1, 0.2), Point(0.3, 0.4)]
    spot = SpotBurnFiducialTaskConfig(task_name="spot", coordinates=burns)
    # a non-spot config in the mix (any object that isn't the spot-burn type)
    lamella = types.SimpleNamespace(task_config={"other": object(), "burn": spot})

    assert AutoLamellaProtocolEditorWidget._spot_burn_coordinates(lamella) == burns


def test_spot_burn_coordinates_empty_when_absent():
    from fibsem.ui.widgets.autolamella_lamella_protocol_editor import (
        AutoLamellaProtocolEditorWidget,
    )

    lamella = types.SimpleNamespace(task_config={"other": object()})
    assert AutoLamellaProtocolEditorWidget._spot_burn_coordinates(lamella) == []
