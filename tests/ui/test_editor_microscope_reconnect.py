"""The protocol editors must follow the microscope across a reconnect.

Both editors are built once and survive disconnect, unlike the main tabs, which
`update_microscope_ui` destroys and rebuilds on every connect. So every widget
under them keeps whatever microscope it was constructed with unless something
tells it otherwise -- and reconnecting is exactly when that matters, because the
new microscope may be a different make, or may have a fluorescence detector where
the previous one had none.

Reported as FIB-525: connect to a non-FM system, disconnect, reconnect to an
FM-enabled one, and the channel editor's excitation/emission dropdowns stay
empty for the life of the process.

Two kinds of staleness, and they need different assertions:

* a widget that **holds** the old microscope (or its `fm`). An identity walk over
  the tree finds these, including in widgets added long after this file.
* a widget that holds only **values derived** from it -- `ChannelSettingsWidget`
  keeps the two wavelength lists and no `fm` at all, and `MillingStageListWidget`
  takes the available currents as constructor arguments. An identity walk is
  blind to these, so they are asserted against what the new microscope reports.
"""

import logging
import os
import pathlib
import tempfile

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtWidgets import QWidget  # noqa: E402

import fibsem.config as fconfig  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.acquire_fluorescence import (  # noqa: E402
    AcquireFluorescenceImageConfig,
)
from fibsem.structures import BeamType  # noqa: E402
from fibsem.ui.widgets.autolamella_lamella_protocol_editor import (  # noqa: E402
    AutoLamellaProtocolEditorWidget,
)
from fibsem.ui.widgets.autolamella_task_config_editor import (  # noqa: E402
    AutoLamellaProtocolTaskConfigEditor,
)

EDITORS = [AutoLamellaProtocolTaskConfigEditor, AutoLamellaProtocolEditorWidget]
EDITOR_IDS = ["task_config_editor", "lamella_protocol_editor"]

# Resolved from the package, not the cwd: the suite's _isolate_cwd fixture
# chdirs each test into a tmp_path, so a relative path finds nothing.
SIM_ARCTIS = os.path.join(fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml")


def _microscope_without_fm():
    """A system with no fluorescence detector, as in the reported sequence."""
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    microscope.fm = None
    return microscope


def _microscope_with_fm():
    microscope, _ = utils.setup_session(
        manufacturer="Demo", ip_address="localhost", config_path=SIM_ARCTIS
    )
    assert microscope.fm is not None, "the arctis sim is the FM-capable fixture"
    return microscope


def _experiment():
    experiment = Experiment(path=pathlib.Path(tempfile.mkdtemp()), name="reconnect")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.task_protocol.task_config = {
        "Acquire Fluorescence Image": AcquireFluorescenceImageConfig(
            task_name="Acquire Fluorescence Image"
        )
    }
    return experiment


def _build(editor_cls, microscope):
    """An editor on *microscope*, plus the host it reads the microscope from."""
    host = QWidget()
    host.experiment, host.microscope = _experiment(), microscope
    editor = editor_cls(parent=host)
    editor._host = host  # keep the parent alive for the test's duration
    return editor, host


def _reconnect(editor, host, microscope):
    host.microscope = microscope
    editor._on_microscope_connected()


def _references_to(editor, *targets):
    """Every `Widget.attribute` in the tree holding one of *targets*."""
    found = set()
    for widget in [editor] + editor.findChildren(QWidget):
        for attr, value in list(vars(widget).items()):
            if any(value is target for target in targets if target is not None):
                found.add(f"{type(widget).__name__}.{attr}")
    return found


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.mark.parametrize("editor_cls", EDITORS, ids=EDITOR_IDS)
def test_no_widget_keeps_the_old_microscope_after_reconnect(qapp, editor_cls):
    """The general guard: nothing in the tree still points at the old microscope.

    Deliberately not a list of widget names. The bug this replaces was an
    incomplete list -- the reconnect handler updated exactly one widget, and the
    other ten were found only by walking the tree. A future widget that captures
    a microscope fails here without anyone remembering to add it.
    """
    old = _microscope_without_fm()
    editor, host = _build(editor_cls, old)

    new = _microscope_with_fm()
    _reconnect(editor, host, new)

    assert _references_to(editor, old, old.fm) == set()
    # ...and the tree really does hold the new one, so the assertion above is not
    # passing merely because every reference was dropped.
    assert _references_to(editor, new, new.fm)


@pytest.mark.parametrize("editor_cls", EDITORS, ids=EDITOR_IDS)
def test_channel_editor_offers_the_new_microscope_filters(qapp, editor_cls):
    """The reported symptom: empty excitation/emission dropdowns after reconnect.

    Asserted on the derived lists rather than on a reference, because the widget
    holding them keeps no `fm` -- the identity walk above cannot see this.
    """
    old = _microscope_without_fm()
    editor, host = _build(editor_cls, old)
    channels = editor.fluorescence_acquisition_task_config_widget.channelSettingsWidget
    detail = channels._settings_widget

    assert detail.excitation_combo.count() == 0, "no FM, so nothing to offer"

    new = _microscope_with_fm()
    _reconnect(editor, host, new)

    expected = list(new.fm.filter_set.available_excitation_wavelengths)
    assert expected, "the fixture must offer some excitation wavelengths"
    offered = [
        detail.excitation_combo.itemData(i) for i in range(detail.excitation_combo.count())
    ]
    assert offered == expected
    assert channels._list._excitation_items == expected
    assert channels._list._emission_items == list(
        new.fm.filter_set.available_emission_wavelengths
    )


@pytest.mark.parametrize("editor_cls", EDITORS, ids=EDITOR_IDS)
def test_milling_stages_offer_the_new_microscope_currents(qapp, editor_cls):
    """The same fault on the milling side, which the one-line reconnect missed.

    `MillingTaskViewerWidget.microscope` was reassigned on reconnect, but it hands
    the microscope to its children at construction, so they kept the old one --
    and the available currents are taken as constructor arguments, one level
    further down again.
    """
    old = _microscope_without_fm()
    editor, host = _build(editor_cls, old)

    new = _microscope_with_fm()
    _reconnect(editor, host, new)

    stages = editor.milling_task_editor.config_widget.milling_stages_widget
    assert stages.microscope is new
    assert stages._list._current_values == new.get_available_values_cached(
        "current", BeamType.ION
    )
    assert stages._list._preset_values == new.get_available_values_cached(
        "preset", BeamType.ION
    )


@pytest.mark.parametrize("editor_cls", EDITORS, ids=EDITOR_IDS)
def test_reconnect_leaves_the_configured_channels_alone(qapp, editor_cls):
    """Swapping microscopes changes the choices offered, not the configuration.

    The channel rows are rebuilt to pick up the new wavelength lists; the channel
    objects they display must survive that, or a reconnect would quietly edit the
    protocol.
    """
    old = _microscope_without_fm()
    editor, host = _build(editor_cls, old)
    channels = editor.fluorescence_acquisition_task_config_widget.channelSettingsWidget
    before = [(ch.name, ch.excitation_wavelength) for ch in channels.channel_settings]

    _reconnect(editor, host, _microscope_with_fm())

    after = [(ch.name, ch.excitation_wavelength) for ch in channels.channel_settings]
    assert after == before


@pytest.mark.parametrize("editor_cls", EDITORS, ids=EDITOR_IDS)
def test_a_half_built_editor_still_reconnects_what_it_has(qapp, editor_cls):
    """A widget missing from a partial build is skipped, not fatal.

    The reconnect branch is entered on `milling_task_editor` existing, but
    `_create_widgets` assigns that first and is not wrapped in a try/except -- so a
    constructor raising midway leaves the branch reachable with later widgets
    absent. The fluorescence widget is exactly the one with previous form here
    (5777bb35, "prevent crash when opening fluorescence config widget with no
    FM"), and an AttributeError raised from this handler would mask whatever
    actually failed during the build.
    """
    old = _microscope_without_fm()
    editor, host = _build(editor_cls, old)
    # the state a build that raised after the milling editor would leave behind
    del editor.fluorescence_acquisition_task_config_widget

    new = _microscope_with_fm()
    _reconnect(editor, host, new)  # must not raise

    assert editor.milling_task_editor.microscope is new


def test_milling_progress_slot_stays_on_the_main_thread():
    """`_on_milling_progress` must keep its @ensure_main_thread decorator.

    It is the slot for `milling_progress_signal`, which the milling worker emits
    off the main thread, so it touches widgets from a background thread without
    it -- the failure mode this codebase has crashed on before (PR #168, the
    Windows vispy/gloo access violation, and the PyQt5 abort-on-slot-exception
    work).

    Worth asserting rather than trusting: adding a method immediately above this
    one lands between the decorator and the function unless the insertion is
    careful, which silently moves the decorator onto the new method. That is
    exactly how it was lost while FIB-525 was being written, and nothing else in
    the suite would have noticed -- the decorator changes which thread the slot
    runs on, not whether it runs.
    """
    from fibsem.ui.widgets.milling_widget import FibsemMillingWidget2

    slot = FibsemMillingWidget2._on_milling_progress
    assert hasattr(slot, "__wrapped__"), (
        "_on_milling_progress is not wrapped -- @ensure_main_thread has been lost "
        "or displaced onto a neighbouring method"
    )
    assert slot.__wrapped__.__name__ == "_on_milling_progress"

    # ...and the reconnect setter must NOT have taken it: it is called from the
    # connected_signal handler, already on the main thread, and wrapping it would
    # defer the propagation rather than doing it inline.
    assert not hasattr(FibsemMillingWidget2.set_microscope, "__wrapped__")
