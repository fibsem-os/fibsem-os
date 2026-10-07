"""The lamella editor says when its FIB image was taken somewhere else (FIB-1170).

Images are acquired with the Demo microscope at two stage positions and saved into the
lamella's folder, so each carries the metadata a real acquisition writes. The lamella's
milling pose is set to the second position; the editor's filename default (newest by
name) is the image taken at the first.

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

from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.autolamella_lamella_protocol_editor import (
    AutoLamellaProtocolEditorWidget,
)
from fibsem.structures import BeamType, FibsemImage, MicroscopeState

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
        host = _Host(microscope, exp)  # selects the first lamella
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


def test_the_default_gives_way_to_the_image_at_the_current_position(scene):
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
    assert notice.switch_button.isHidden()
    assert _hint(editor) is None


def test_a_default_taken_where_the_lamella_is_is_kept(scene):
    scene["acquire_new"]()
    old = FibsemImage.load(os.path.join(scene["lamella"].path, OLD_IB))
    scene["lamella"].milling_pose = old.metadata.microscope_state

    editor = scene["editor"]()

    assert _fib(editor) == OLD_IB
    assert editor.position_notice.isHidden()
    assert _hint(editor) is None


def test_a_pick_is_described_not_replaced_and_the_current_image_offered(scene):
    scene["acquire_new"]()
    editor = scene["editor"]()

    picker = editor.combobox_fib_filenames
    picker.setCurrentIndex(picker.findData(OLD_IB))  # by hand

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


def test_no_image_at_the_current_position_is_said(scene):
    editor = scene["editor"]()

    assert _fib(editor) == OLD_IB  # nothing to give way to
    notice = editor.position_notice
    assert not notice.isHidden()
    assert "No image has been taken at its current position." in notice.label.text()
    assert notice.switch_button.isHidden()
    assert _hint(editor) == "Lamella moved since this image: ~50 µm"


def test_an_image_with_no_position_is_unknown_not_moved(scene):
    scene["acquire_new"]()
    bare = "ref_Spot Burn_ib.tif"  # sorts last, so it is the filename default
    tff.imwrite(
        os.path.join(scene["lamella"].path, bare),
        np.zeros((256, 384), dtype=np.uint8),
    )
    editor = scene["editor"]()
    # an image known to be at the current position is preferred to an unknown one
    assert _fib(editor) == NEW_IB

    picker = editor.combobox_fib_filenames
    picker.setCurrentIndex(picker.findData(bare))

    notice = editor.position_notice
    assert notice.label.text() == "This image has no stage position recorded."
    assert not notice.switch_button.isHidden()
    assert _hint(editor) is None  # not "moved"


def test_a_lamella_with_no_position_shows_nothing(scene):
    scene["lamella"].milling_pose = MicroscopeState()
    scene["acquire_new"]()
    editor = scene["editor"]()
    assert _fib(editor) == OLD_IB  # nothing to compare with, so nothing switched
    assert editor.position_notice.isHidden()
    assert _hint(editor) is None
