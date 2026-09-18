"""The milling form never hides every manufacturer-tagged field at once.

Eight of the form's fields are tagged for one manufacturer or the other, and the
rule that reads the tag used to be `row.mfr == manufacturer` with no else. On a
JEOL instrument neither string matched, so all eight disappeared together and the
form came up with no milling current, no preset, and nothing to mill by (FIB-975).

The tag is a proxy for a question only the driver can answer, and FIB-1011 replaces
it. Until then the fallback is the fix: an instrument the tags know about hides the
other one's fields, and an instrument they say nothing about hides none of them.
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

THERMO_FIELDS = {"milling_current", "milling_voltage", "application_file"}
TESCAN_FIELDS = {"preset", "spot_size", "rate", "dwell_time", "spacing"}


@pytest.fixture(scope="module")
def microscope():
    scope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return scope


def _shown(widget, manufacturer):
    """The fields the form shows for *manufacturer*, advanced rows included."""
    widget.set_advanced_visible(True)
    widget.set_manufacturer(manufacturer)
    return {row.field for row in widget._rows if not row.label.isHidden()}


@pytest.fixture
def widget(qapp, microscope):
    return FibsemMillingSettingsWidget(
        microscope=microscope, settings=FibsemMillingSettings()
    )


def test_an_unknown_manufacturer_keeps_every_field(widget):
    """The JEOL case. Neither tag matches, so neither tag hides anything."""
    shown = _shown(widget, "JEOL")

    assert THERMO_FIELDS <= shown
    assert TESCAN_FIELDS <= shown


def test_a_known_manufacturer_still_hides_the_other_ones(widget):
    thermo = _shown(widget, "ThermoFisher")
    assert THERMO_FIELDS <= thermo
    assert not (TESCAN_FIELDS & thermo)

    tescan = _shown(widget, "Tescan")
    assert TESCAN_FIELDS <= tescan
    assert not (THERMO_FIELDS & tescan)


def test_the_manufacturer_is_normalised_before_it_is_compared(widget):
    """ "TESCAN" and "Thermo" are the same instruments as their canonical spellings.

    Without normalisation an alias now falls into the unknown case and shows every
    field, which is safe but still wrong -- the tags exist to hide the five fields a
    TESCAN column has no use for.
    """
    assert _shown(widget, "TESCAN") == _shown(widget, "Tescan")
    assert _shown(widget, "Thermo Fisher Scientific") == _shown(widget, "ThermoFisher")
