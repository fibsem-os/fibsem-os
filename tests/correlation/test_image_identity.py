"""Correlation must name an image by the file it loaded (FIB-509).

Six places asked "which file is this?" by reading the acquisition *request* --
``image_settings.filename`` for FIB, ``metadata.filename`` for FM. That is the
name the acquiring process asked for, and it is not the file:

* it is a **stem**, because ``FibsemImage.save`` appends the suffix itself, so
  matching it against a real path never succeeds;
* it is absent entirely from a plain TIFF, which loads with ``metadata = None``;
* it still says the old name after a rename or a copy.

``filepath`` is set by ``load()`` and ``save()`` on both image classes, whatever
the metadata says, and is the answer to a filesystem question.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from fibsem.correlation.structures import CorrelationInputData
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.structures import FibsemImage, ImageSettings

# What a lamella folder actually holds, and what the acquirer asked to call it.
# Note the request has no extension -- that is the whole defect.
FIB_PATH = "/lam/01-grand-dodo/ref_Mill Fiducial_start.tif"
FIB_REQUESTED_NAME = "ref_Mill Fiducial_start"
FM_PATH = "/lam/01-grand-dodo/fm_stack.ome.tiff"


@pytest.fixture(autouse=True)
def _no_lut_download(monkeypatch):
    import fibsem.correlation.ui.widgets.refractive_index_widget as riw

    monkeypatch.setattr(riw, "_ensure_lut", lambda: None)


def _fib_image(filepath=FIB_PATH, requested_name=FIB_REQUESTED_NAME):
    """A FIB image as it arrives from disk: loaded from one name, requesting another."""
    image = FibsemImage.generate_blank_image(resolution=(64, 48), hfw=100e-6)
    image.metadata.image_settings = ImageSettings(
        resolution=(64, 48), hfw=100e-6, filename=requested_name, path="/lam/01-grand-dodo"
    )
    image.filepath = filepath
    return image


def _fm_image(filepath=FM_PATH, requested_name="fm_stack_as_acquired"):
    metadata = FluorescenceImageMetadata(
        acquisition_date="2026-08-05T00:00:00",
        pixel_size_x=1e-7,
        pixel_size_y=1e-7,
        resolution=(16, 16),
        channels=[
            FluorescenceChannelMetadata(
                name="GFP",
                color="#00FF00",
                excitation_wavelength=488,
                emission_wavelength=509,
                power=1.0,
                exposure_time=0.1,
                gain=1.0,
                offset=0.0,
            )
        ],
        filename=requested_name,
    )
    return FluorescenceImage(
        data=np.zeros((1, 2, 16, 16), dtype=np.uint16),
        metadata=metadata,
        filepath=filepath,
    )


def _images_tab():
    from fibsem.correlation.ui.widgets.correlation_tab_widget import _ImagesTab

    return _ImagesTab()


def _tab_widget():
    from fibsem.correlation.ui.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    return CorrelationTabWidget()


# ---------------------------------------------------------------------------
# The picker match, which is what actually broke
# ---------------------------------------------------------------------------


def test_the_requested_name_cannot_match_a_real_file(qapp):
    """Why reading the request was never going to work.

    ``select_file`` compares basenames. The request has no extension, the file on
    disk does, so the comparison is ``'x' == 'x.tif'`` for every image ever
    acquired through the normal path.
    """
    from fibsem.correlation.ui.widgets.correlation_tab_widget import _ImagePicker

    picker = _ImagePicker("TIFF (*.tif *.tiff);;All Files (*)")
    picker.set_options([FIB_PATH])

    assert picker.select_file(FIB_REQUESTED_NAME) == ""
    assert picker.select_file(FIB_PATH) == FIB_PATH


def test_a_preloaded_fib_image_selects_its_file(qapp):
    """A lamella's own image, handed straight to the tab.

    The match here was dead code: it could not succeed, so the only reason the
    protocol editor's dialog ends up on the right file is that
    ``add_lamella_setup`` runs a moment later and re-selects it. A caller that
    pre-loads without that -- ``CorrelationTabDialog.set_fib_image`` on its own --
    got nothing.
    """
    tab = _images_tab()
    tab.set_image_options("fib", [FIB_PATH])

    tab.set_fib_image(_fib_image())

    assert tab._fib_loaded_path == FIB_PATH
    assert tab._fib_picker.current_path() == FIB_PATH


def test_a_preloaded_fm_image_selects_its_file(qapp):
    tab = _images_tab()
    tab.set_image_options("fm", [FM_PATH])

    tab.set_fm_image(_fm_image())

    assert tab._fm_loaded_path == FM_PATH
    assert tab._fm_picker.current_path() == FM_PATH


def test_an_image_from_outside_the_list_leaves_the_picker_alone(qapp):
    """``select_file`` returning "" must not be turned into a selection.

    Showing a file the picker cannot load is worse than showing the previous one.
    """
    tab = _images_tab()
    tab.set_image_options("fib", [FIB_PATH])
    tab.set_fib_image(_fib_image())

    tab.set_fib_image(_fib_image(filepath="/elsewhere/unrelated_ib.tif"))

    assert tab._fib_loaded_path == FIB_PATH


# ---------------------------------------------------------------------------
# The header labels
# ---------------------------------------------------------------------------


def test_the_header_and_the_picker_agree(qapp):
    """The visible symptom in the flow people actually use.

    ``_open_correlation_dialog`` calls ``set_fib_image`` and *then*
    ``add_lamella_setup``, which re-selects the file in the picker. The header
    label is never revisited, so it went on showing the acquirer's stem beside a
    picker showing the real file — one image, two names, adjacent controls.
    """
    widget = _tab_widget()

    widget.set_fib_image(_fib_image())  # what the dialog does first
    widget._images_tab.set_image_options("fib", [FIB_PATH], FIB_PATH)  # ...then this

    assert widget._fib_name_label.text() == os.path.basename(
        widget._images_tab._fib_picker.current_path()
    )


def test_the_fib_header_names_the_file_not_the_request(qapp):
    widget = _tab_widget()

    widget._update_fib_name_label(_fib_image())

    assert widget._fib_name_label.text() == "ref_Mill Fiducial_start.tif"
    assert widget._fib_name_label.toolTip() == FIB_PATH


def test_the_fm_header_names_the_file_not_the_request(qapp):
    from fibsem.correlation.ui.widgets.fm_image_display_widget import (
        FMImageDisplayWidget,
    )

    display = FMImageDisplayWidget()
    display.set_fm_image(_fm_image())

    assert display._name_label.text() == "fm_stack.ome.tiff"
    assert display._name_label.toolTip() == FM_PATH


def test_an_unsaved_image_is_named_nothing_rather_than_wrongly(qapp):
    """An image that was never written has no file, and saying otherwise sends a
    reader looking for one that does not exist."""
    widget = _tab_widget()

    widget._update_fib_name_label(_fib_image(filepath=None))

    assert widget._fib_name_label.text() == ""
    assert widget._fib_name_label.isVisible() is False


# ---------------------------------------------------------------------------
# The record field, which rides into every saved correlation
# ---------------------------------------------------------------------------


def test_the_record_names_the_file_it_correlated(qapp):
    data = CorrelationInputData(fib_image=_fib_image(), fm_image=_fm_image())

    assert data.fib_image_filename == "ref_Mill Fiducial_start.tif"
    assert data.fm_image_filename == "fm_stack.ome.tiff"
    assert data.to_dict()["fib_image_filename"] == "ref_Mill Fiducial_start.tif"
    assert data.to_dict()["fm_image_filename"] == "fm_stack.ome.tiff"


def test_a_plain_tiff_is_still_named(qapp):
    """The case the old code could not answer at all.

    ``FibsemImage.load`` sets ``metadata = None`` for a TIFF written by other
    software but sets ``filepath`` regardless, so the widget knows exactly which
    file it holds -- and used to record ``null`` for it (FIB-316's chains stop
    the crash; they cannot produce a name).
    """
    image = FibsemImage.generate_blank_image(resolution=(64, 48), hfw=100e-6)
    image.metadata = None
    image.filepath = "/elsewhere/from_another_program.tif"

    data = CorrelationInputData(fib_image=image)

    assert data.to_dict()["fib_image_filename"] == "from_another_program.tif"


# ---------------------------------------------------------------------------
# Derived volumes, which naming from `filepath` would otherwise strand
# ---------------------------------------------------------------------------


def test_an_interpolated_volume_is_still_that_files_volume(qapp):
    """Interpolate is a button in the Images tab, so this is a normal path.

    Naming an image from ``filepath`` means a derived image that drops it gets no
    name at all -- worse than before, since the metadata copy used to carry the
    acquisition name forward. The resampled volume is still the source file's.
    """
    from fibsem.correlation.util import interpolate_fm_volume

    source = _fm_image()
    source.metadata.pixel_size_z = 5e-7

    interpolated = interpolate_fm_volume(
        source, target_z_size_m=2.5e-7, method="linear"
    )

    assert interpolated.filepath == FM_PATH
    assert CorrelationInputData(fm_image=interpolated).fm_image_filename == (
        "fm_stack.ome.tiff"
    )


def test_a_projection_and_a_focus_stack_keep_it_too(qapp):
    """The other two derived forms, which have the same shape."""
    source = _fm_image()
    source.metadata.pixel_size_z = 5e-7

    assert source.max_intensity_projection().filepath == FM_PATH
    assert source.focus_stack().filepath == FM_PATH


def test_no_image_and_no_file_stay_none(qapp):
    """``to_dict`` is on the only persistence path, so neither may raise (FIB-316)."""
    assert CorrelationInputData().fib_image_filename is None
    assert CorrelationInputData().fm_image_filename is None
    assert CorrelationInputData(fib_image=_fib_image(filepath=None)).to_dict()[
        "fib_image_filename"
    ] is None
