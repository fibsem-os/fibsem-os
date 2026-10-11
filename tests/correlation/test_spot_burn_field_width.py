"""Spot burns seed only onto a FIB image at the field width they were placed on
(FIB-1233).

The coordinates are fractions of the image the burns were placed on, so they
land right on any image at the same field width, whatever its resolution, and
1.5x out from the centre on a 150 um image of a 100 um pattern. Seeding now
refuses, and says why, when the two widths differ.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/correlation/test_spot_burn_field_width.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import types

import pytest

pytest.importorskip("PyQt5")

from fibsem.structures import FibsemImage, Point

_BURNS = [Point(0.25, 0.5), Point(0.75, 0.25)]


@pytest.fixture
def widget(qapp, monkeypatch):
    import fibsem.ui.correlation.widgets.refractive_index_widget as riw
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    monkeypatch.setattr(riw, "_ensure_lut", lambda: None)
    w = CorrelationTabWidget()
    yield w
    w.close()


@pytest.fixture
def toasts(monkeypatch):
    from fibsem.ui import notification_service

    events = []
    monkeypatch.setattr(
        notification_service,
        "show_toast",
        lambda msg, notification_type="info": events.append((msg, notification_type)),
    )
    return events


def _image(hfw: float, resolution=(300, 200)) -> FibsemImage:
    return FibsemImage.generate_blank_image(resolution=resolution, hfw=hfw)


def test_burns_seed_onto_an_image_at_the_width_they_were_placed_on(widget, toasts):
    widget.set_fib_image(_image(100e-6, resolution=(600, 400)))  # any resolution
    widget.add_lamella_setup(spot_burns=_BURNS, spot_burn_field_width=100e-6)

    widget.seed_fib_fiducials_from_spot_burns(_BURNS)

    fib = widget.data.fib_coordinates
    assert [(c.point.x, c.point.y) for c in fib] == [(150.0, 200.0), (450.0, 100.0)]
    assert toasts == []


def test_burns_are_not_seeded_onto_an_image_at_another_width(widget, toasts):
    widget.set_fib_image(_image(150e-6))
    widget.add_lamella_setup(spot_burns=_BURNS, spot_burn_field_width=100e-6)

    widget.seed_fib_fiducials_from_spot_burns(_BURNS)

    assert widget.data.fib_coordinates == []
    msg = widget._lbl_status.text()
    assert "100 µm" in msg and "150 µm" in msg
    assert toasts and toasts[-1] == (msg, "warning")


def test_an_unknown_width_seeds_as_before(widget, toasts):
    """No field width from the protocol (the standalone launcher): no check."""
    widget.set_fib_image(_image(150e-6))
    widget.add_lamella_setup(spot_burns=_BURNS)

    widget.seed_fib_fiducials_from_spot_burns(_BURNS)

    assert len(widget.data.fib_coordinates) == 2
    assert toasts == []


def test_the_protocol_editor_reads_the_width_of_the_burns_reference_image():
    from fibsem.applications.autolamella.ui.autolamella_lamella_protocol_editor import (
        AutoLamellaProtocolEditorWidget,
    )
    from fibsem.applications.autolamella.workflows.tasks.spot_burn import (
        SpotBurnFiducialTaskConfig,
    )

    config = SpotBurnFiducialTaskConfig()
    config.reference_imaging.field_of_view1 = 100e-6
    lamella = types.SimpleNamespace(task_config={"Spot Burn Fiducial": config})
    width = AutoLamellaProtocolEditorWidget._spot_burn_field_width(lamella)
    assert width == pytest.approx(100e-6)

    no_burns = types.SimpleNamespace(task_config={})
    assert AutoLamellaProtocolEditorWidget._spot_burn_field_width(no_burns) is None
