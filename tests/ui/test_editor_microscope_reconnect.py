"""The protocol editors' fluorescence editor must follow the microscope on reconnect.

Both editors are built once and survive disconnect, unlike the main tabs, which
`update_microscope_ui` destroys and rebuilds on every connect. So every widget
under them keeps whatever microscope it was constructed with unless something
tells it otherwise -- and reconnecting is exactly when that matters, because the
new microscope may have a fluorescence detector where the previous one had none.

Reported as FIB-525: connect to a non-FM system, disconnect, reconnect to an
FM-enabled one, and the channel editor's excitation/emission dropdowns stay
empty for the life of the process.

Two kinds of staleness, needing different assertions:

* a widget that **holds** the old microscope (or its `fm`) -- an identity walk
  over the tree finds these, including in widgets added long after this file.
* a widget holding only values **derived** from it. `ChannelSettingsWidget` keeps
  the two wavelength lists and no `fm` at all, so an identity walk is blind to
  it; those are asserted against what the new microscope reports.

The fixture deliberately configures **real channels**. An earlier version of
these tests used the config default, which is an empty channel list, so the row
rebuild -- the mechanism that actually fixes the reported symptom -- was never
exercised and the "channels survive" assertion compared two empty lists.
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
from fibsem.applications.autolamella.ui.autolamella_lamella_protocol_editor import (  # noqa: E402
    AutoLamellaProtocolEditorWidget,
)
from fibsem.applications.autolamella.ui.autolamella_task_config_editor import (  # noqa: E402
    AutoLamellaProtocolTaskConfigEditor,
)
from fibsem.applications.autolamella.workflows.tasks.acquire_fluorescence import (  # noqa: E402
    AcquireFluorescenceImageConfig,
)
from fibsem.fm.structures import ChannelSettings  # noqa: E402

EDITORS = [AutoLamellaProtocolTaskConfigEditor, AutoLamellaProtocolEditorWidget]
EDITOR_IDS = ["task_config_editor", "lamella_protocol_editor"]

# Resolved from the package, not the cwd: the suite's _isolate_cwd fixture chdirs
# each test into a tmp_path, so a relative path finds nothing.
SIM_ARCTIS = os.path.join(fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml")


def _channels():
    return [
        ChannelSettings(
            name="Reflection",
            excitation_wavelength=550,
            emission_wavelength=None,
            power=0.02,
            exposure_time=0.005,
        ),
        ChannelSettings(
            name="Red",
            excitation_wavelength=550,
            emission_wavelength="FLUORESCENCE",
            power=0.3,
            exposure_time=0.1,
            color="red",
        ),
    ]


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
            task_name="Acquire Fluorescence Image", channel_settings=_channels()
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


def _channel_list(editor):
    return editor.fluorescence_acquisition_task_config_widget.channelSettingsWidget


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
def test_the_fluorescence_subtree_keeps_no_reference_to_the_old_microscope(
    qapp, editor_cls
):
    """Nothing under the fluorescence widget still points at the old microscope.

    Deliberately not a list of widget names: the bug being fixed was an incomplete
    list -- the reconnect handler updated one widget and the rest were found only
    by walking the tree. A future widget capturing an `fm` fails here without
    anyone remembering to add it.

    Scoped to the fluorescence subtree because the milling subtree has the same
    fault and is fixed separately (FIB-530); widening this to the whole editor is
    part of that change.
    """
    old = _microscope_without_fm()
    editor, host = _build(editor_cls, old)
    fluorescence = editor.fluorescence_acquisition_task_config_widget

    new = _microscope_with_fm()
    _reconnect(editor, host, new)

    assert _references_to(fluorescence, old, old.fm) == set()
    # ...and it really does hold the new one, so the assertion above is not passing
    # merely because every reference was dropped.
    assert _references_to(fluorescence, new, new.fm)


@pytest.mark.parametrize("editor_cls", EDITORS, ids=EDITOR_IDS)
def test_channel_editor_offers_the_new_microscope_filters(qapp, editor_cls):
    """The reported symptom: empty dropdowns after reconnecting to an FM system.

    Asserted on the derived lists as well as the widgets, because the widget
    holding them keeps no `fm` -- the identity walk above cannot see it.
    """
    old = _microscope_without_fm()
    editor, host = _build(editor_cls, old)
    channels = _channel_list(editor)
    detail = channels._settings_widget

    assert detail.excitation_combo.count() == 0, "no FM, so nothing to offer"

    new = _microscope_with_fm()
    _reconnect(editor, host, new)

    expected = list(new.fm.filter_set.available_excitation_wavelengths)
    assert expected, "the fixture must offer some excitation wavelengths"
    assert [
        detail.excitation_combo.itemData(i) for i in range(detail.excitation_combo.count())
    ] == expected
    assert channels._list._excitation_items == expected
    assert channels._list._emission_items == list(
        new.fm.filter_set.available_emission_wavelengths
    )


# --- widget-level: the row rebuild itself ------------------------------------
# Tested on the widget directly rather than through an editor, because the two
# editors populate it at different times: the task-config editor passes the
# selected task's config in, while the lamella editor constructs it with
# `config=None` and fills it when a lamella/task is chosen. Driving each editor's
# population lifecycle would test that lifecycle, not the rebuild.


def _fluorescence_widget(microscope):
    from fibsem.applications.autolamella.ui.autolamella_fluorescence_acquisition_task_config_widget import (
        AutoLamellaFluorescenceAcquisitionTaskConfigWidget,
    )

    config = AcquireFluorescenceImageConfig(
        task_name="Acquire Fluorescence Image", channel_settings=_channels()
    )
    widget = AutoLamellaFluorescenceAcquisitionTaskConfigWidget(
        microscope=microscope, config=config
    )
    assert widget.channelSettingsWidget._list._list.count() == len(_channels())
    return widget


def test_every_channel_row_offers_the_new_filters(qapp):
    """Each row's own combos, not just the detail pane and the cached lists.

    The rows take their choices as constructor arguments, so they are rebuilt
    rather than updated -- and with the empty-channel fixture this file used to
    have, that rebuild ran over zero rows and proved nothing.
    """
    widget = _fluorescence_widget(_microscope_without_fm())
    rows = widget.channelSettingsWidget._list

    new = _microscope_with_fm()
    widget.set_microscope(new)

    expected = list(new.fm.filter_set.available_excitation_wavelengths)
    assert rows._list.count() == len(_channels())
    for i in range(rows._list.count()):
        combo = rows._row(i).excitation_combo
        assert [combo.itemData(j) for j in range(combo.count())] == expected


def test_reconnect_keeps_the_very_same_channel_objects(qapp):
    """Swapping microscopes changes the choices offered, not the configuration.

    Identity, not equality: the rows are rebuilt, and the channel objects they
    display are the protocol's own. Replacing them with copies would leave the
    editor writing to objects the experiment no longer holds.
    """
    widget = _fluorescence_widget(_microscope_without_fm())
    rows = widget.channelSettingsWidget._list
    before = list(rows.channel_settings)

    widget.set_microscope(_microscope_with_fm())

    after = list(rows.channel_settings)
    assert [id(ch) for ch in after] == [id(ch) for ch in before]
    assert [ch.name for ch in after] == [ch.name for ch in before]
    assert [ch.exposure_time for ch in after] == [ch.exposure_time for ch in before]


@pytest.mark.parametrize("editor_cls", EDITORS, ids=EDITOR_IDS)
def test_reconnect_does_not_edit_or_save_the_protocol(qapp, editor_cls, monkeypatch):
    """Reconnecting must not look like the operator changed something.

    The rebuild runs through the same setter a real edit uses, so it could
    plausibly emit the change signals the editors treat as "protocol is dirty,
    write it to disk". Swapping a microscope is not an edit, and an experiment
    that rewrites itself on reconnect would be a nasty way to find that out.
    """
    old = _microscope_without_fm()
    editor, host = _build(editor_cls, old)

    events = []
    fluorescence = editor.fluorescence_acquisition_task_config_widget
    fluorescence.settings_changed.connect(lambda *_: events.append("settings_changed"))
    fluorescence.channel_settings_changed.connect(
        lambda *_: events.append("channel_settings_changed")
    )
    saves = []
    monkeypatch.setattr(editor, "_save_experiment", lambda *a, **k: saves.append(1))

    _reconnect(editor, host, _microscope_with_fm())
    qapp.processEvents()

    assert events == []
    assert saves == []


@pytest.mark.parametrize("editor_cls", EDITORS, ids=EDITOR_IDS)
def test_a_half_built_editor_still_reconnects_what_it_has(qapp, editor_cls):
    """A widget missing from a partial build is skipped, not fatal.

    The reconnect branch is entered on `milling_task_editor` existing, but
    `_create_widgets` assigns that first and is not wrapped in a try/except -- so a
    constructor raising midway leaves the branch reachable with later widgets
    absent. The fluorescence widget is exactly the one with previous form here
    (5777bb35, "prevent crash when opening fluorescence config widget with no
    FM"), and an AttributeError from this handler would mask the real failure.
    """
    old = _microscope_without_fm()
    editor, host = _build(editor_cls, old)
    del editor.fluorescence_acquisition_task_config_widget

    _reconnect(editor, host, _microscope_with_fm())  # must not raise
