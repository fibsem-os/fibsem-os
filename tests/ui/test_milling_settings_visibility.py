"""The milling form shows the recipe fields the microscope's milling service mills with.

The form used to choose by a manufacturer tag on each field, a proxy for a question
only the driver can answer, and on a JEOL instrument it once hid every tagged field
at once (FIB-975). The service now answers it: `supported_settings` lists the fields
the driver's setup reads, with their choices.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem import utils  # noqa: E402
from fibsem.structures import FibsemMillingSettings  # noqa: E402
from fibsem.ui.widgets.milling_settings_widget import (  # noqa: E402
    FibsemMillingSettingsWidget,
)


@pytest.fixture(scope="module")
def microscope():
    scope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return scope


def _widget(microscope):
    return FibsemMillingSettingsWidget(
        microscope=microscope, settings=FibsemMillingSettings()
    )


def _shown(widget):
    """The fields the form shows, advanced rows included."""
    widget.set_advanced_visible(True)
    return {row.field for row in widget._rows if not row.label.isHidden()}


def test_the_milling_service_says_which_fields_show(qapp, microscope):
    widget = _widget(microscope)
    supported = microscope.milling.supported_settings()
    rows = {row.field for row in widget._rows}
    assert _shown(widget) == set(supported) & rows
    # the Demo mills like ThermoFisher, so none of Tescan's fields
    assert not {"preset", "spot_size", "rate", "dwell_time", "spacing"} & _shown(widget)

    # the application file's choices are the service's
    (row,) = [r for r in widget._rows if r.field == "application_file"]
    combo = row.control.widget
    items = [combo.itemText(i) for i in range(combo.count())]
    assert items == list(supported["application_file"].choices)


def test_without_a_milling_service_every_field_shows(qapp, microscope, monkeypatch):
    monkeypatch.setattr(microscope, "milling", None)
    widget = _widget(microscope)
    assert _shown(widget) == {row.field for row in widget._rows}
