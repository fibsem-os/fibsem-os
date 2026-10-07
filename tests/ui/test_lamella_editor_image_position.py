"""The lamella editor says when its FIB image was taken somewhere else (FIB-1170).

Images are acquired with the Demo microscope at two stage positions and saved into the
lamella's folder, so each carries the metadata a real acquisition writes. The lamella's
milling pose is set to the second position; the editor's default pick (newest by name)
is the image taken at the first.

    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_lamella_editor_image_position.py -q
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import tifffile as tff
from psygnal.containers import EventedDict

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QWidget

from fibsem import config as fibsem_cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.autolamella_lamella_protocol_editor import (
    AutoLamellaProtocolEditorWidget,
)
from fibsem.config import UserPreferences
from fibsem.structures import BeamType, MicroscopeState
from fibsem.ui.widgets.preferences_dialog import PreferencesDialog

_app = QApplication.instance() or QApplication(sys.argv)

# Sorted by name, the setup images come last, so they are the editor's default pick.
OLD_IB = "ref_Setup Lamella Position_final_res_01_ib.tif"  # where the lamella was
OLD_EB = "ref_Setup Lamella Position_final_res_01_eb.tif"
NEW_IB = "ref_Rough Milling_final_res_01_ib.tif"  # where it is now
NEW_EB = "ref_Rough Milling_final_res_01_eb.tif"
MOVE_M = 50e-6


class _Host(QWidget):
    """The two attributes the editor reads off its parent."""

    def __init__(self, microscope, experiment):
        super().__init__()
        self.microscope = microscope
        self.experiment = experiment


@pytest.fixture
def preferences(tmp_path, monkeypatch):
    """A preferences file of the test's own, never the user's."""
    monkeypatch.setattr(
        fibsem_cfg, "USER_PREFERENCES_PATH", str(tmp_path / "prefs.yaml")
    )

    def _set(switch: bool) -> None:
        prefs = UserPreferences()
        prefs.display.show_reference_image_at_current_position = switch
        fibsem_cfg.save_user_preferences(prefs)

    return _set


@pytest.fixture
def scene(tmp_path):
    """A lamella with images at an old position and its pose at a new one."""
    microscope, settings = utils.setup_session(manufacturer="Demo")
    settings.image.resolution = [384, 256]
    settings.image.hfw = 150e-6

    exp = Experiment(path=str(tmp_path / "exp"), name="fib1170")
    exp.task_protocol = AutoLamellaTaskProtocol()
    os.makedirs(exp.path, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    lamella = exp.positions[0]

    def acquire(filenames):
        for beam, name in zip((BeamType.ION, BeamType.ELECTRON), filenames):
            image_settings = deepcopy(settings.image)
            image_settings.beam_type = beam
            image = microscope.acquire_image(image_settings)
            image.save(os.path.join(lamella.path, name))

    old = microscope.get_stage_position()
    acquire((OLD_IB, OLD_EB))
    new = deepcopy(old)
    new.x += MOVE_M
    microscope.move_stage_absolute(new)
    lamella.milling_pose = microscope.get_microscope_state()

    def editor():
        host = _Host(microscope, exp)
        widget = AutoLamellaProtocolEditorWidget(parent=host)
        widget._host = host  # keep it alive
        return widget

    return {
        "lamella": lamella,
        "acquire_new": lambda: acquire((NEW_IB, NEW_EB)),
        "editor": editor,
    }


def _fib(editor):
    return editor.combobox_fib_filenames.currentData()


def _shown(editor):
    return (
        editor.combobox_fib_filenames.currentData(),
        editor.combobox_sem_filenames.currentData(),
    )


def _hint(editor):
    return editor.view_controller.get_canvas(BeamType.ION)._hint_text


def test_a_moved_lamella_is_named_and_the_current_image_offered(scene, preferences):
    preferences(False)
    scene["acquire_new"]()
    editor = scene["editor"]()

    assert _fib(editor) == OLD_IB
    notice = editor.position_notice
    assert not notice.isHidden()
    assert "has moved since this image was taken: ~50 µm." in notice.label.text()
    assert not notice.switch_button.isHidden()
    assert _hint(editor) == "Lamella moved since this image: ~50 µm"

    notice.switch_button.click()

    assert _shown(editor) == (NEW_IB, NEW_EB)
    assert notice.isHidden()
    assert _hint(editor) is None
    assert editor.image.filepath.endswith(NEW_IB)


def test_no_image_at_the_current_position_is_said(scene, preferences):
    preferences(True)  # nothing to switch to, preference or not
    editor = scene["editor"]()

    assert _fib(editor) == OLD_IB
    notice = editor.position_notice
    assert not notice.isHidden()
    assert "No image has been taken at its current position." in notice.label.text()
    assert notice.switch_button.isHidden()


def test_the_preference_switches_a_default_but_not_a_pick(scene, preferences):
    preferences(True)
    scene["acquire_new"]()
    editor = scene["editor"]()

    assert _shown(editor) == (NEW_IB, NEW_EB)
    notice = editor.position_notice
    assert not notice.isHidden()
    # no tasks on this lamella, so the picker labels it by its stem
    assert notice.label.text() == (
        "Showing Rough Milling final res 01, the image taken at the lamella's "
        "current position."
    )
    assert _hint(editor) is None

    # picked by hand: kept, and described
    editor.combobox_fib_filenames.setCurrentIndex(
        editor.combobox_fib_filenames.findData(OLD_IB)
    )
    assert _fib(editor) == OLD_IB
    assert "has moved since this image was taken: ~50 µm" in notice.label.text()


def test_the_preference_is_off_by_default(scene, preferences):
    assert UserPreferences().display.show_reference_image_at_current_position is False
    scene["acquire_new"]()
    editor = scene["editor"]()  # no preferences file at all
    assert _fib(editor) == OLD_IB


def test_an_image_with_no_position_is_unknown_not_moved(scene, preferences):
    preferences(True)
    scene["acquire_new"]()
    # sorts after everything else, so it is the default
    tff.imwrite(
        os.path.join(scene["lamella"].path, "ref_Spot Burn_ib.tif"),
        np.zeros((256, 384), dtype=np.uint8),
    )
    editor = scene["editor"]()

    assert _fib(editor) == "ref_Spot Burn_ib.tif"  # not switched: not "moved"
    notice = editor.position_notice
    assert notice.label.text() == "This image has no stage position recorded."
    assert not notice.switch_button.isHidden()
    assert _hint(editor) is None


def test_a_lamella_with_no_position_shows_nothing(scene, preferences):
    preferences(False)
    scene["lamella"].milling_pose = MicroscopeState()
    editor = scene["editor"]()
    assert editor.position_notice.isHidden()
    assert _hint(editor) is None


def test_the_preference_round_trips_through_the_dialog():
    prefs = UserPreferences()
    prefs.display.show_reference_image_at_current_position = True
    dialog = PreferencesDialog(prefs)
    assert dialog.get_preferences().display.show_reference_image_at_current_position
    dialog._chk_current_image.setChecked(False)
    assert not (
        dialog.get_preferences().display.show_reference_image_at_current_position
    )
