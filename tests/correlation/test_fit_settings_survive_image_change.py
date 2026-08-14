"""Fit settings must survive a new FM volume.

FIB-636: interpolating the z-stack reverted all five fit controls to the config's
values. `set_fm_image` rebuilds the channel combos — which clears them — and then
re-applied the whole config to repair the damage, overwriting the user's choices
along the way. The method combos were never rebuilt, so they were collateral.

Not cosmetic: the config is read back when the dialog is accepted and written to
`protocol.correlation` for every lamella, so a reset before close discarded the
user's fit settings into the saved experiment.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("napari")

from fibsem.correlation.config import CorrelationConfig, FitSettings
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.ui.correlation.widgets.correlation_tab_widget import CorrelationTabWidget

_SIZE = 32


def _fm_image(channel_names, nz: int = 5) -> FluorescenceImage:
    """A small (C, Z, Y, X) stack carrying the given channel names."""
    metadata = FluorescenceImageMetadata(
        acquisition_date="2026-08-14T00:00:00",
        pixel_size_x=1e-7,
        pixel_size_y=1e-7,
        pixel_size_z=5e-7,
        resolution=(_SIZE, _SIZE),
        channels=[
            FluorescenceChannelMetadata(
                name=name, color="#00ff00", excitation_wavelength=488,
                emission_wavelength=509, power=1.0, exposure_time=0.1,
                gain=1.0, offset=0.0,
            )
            for name in channel_names
        ],
        filename="stack",
    )
    data = np.zeros((len(channel_names), nz, _SIZE, _SIZE), dtype=np.uint16)
    return FluorescenceImage(data=data, metadata=metadata)


@pytest.fixture
def widget(qapp):
    w = CorrelationTabWidget()
    yield w
    w.deleteLater()


def _set_all_five(widget, *, fib, fid_method, poi_method, fid_ch, poi_ch):
    cl = widget._coords_tab
    cl._fib_method_combo.setCurrentText(fib)
    cl._fm_fid_method_combo.setCurrentText(fid_method)
    cl._fm_poi_method_combo.setCurrentText(poi_method)
    cl._fm_fid_ch_combo.setCurrentText(fid_ch)
    cl._fm_poi_ch_combo.setCurrentText(poi_ch)


def _read_all_five(widget):
    cl = widget._coords_tab
    return (
        cl._fib_method_combo.currentText(),
        cl._fm_fid_method_combo.currentText(),
        cl._fm_poi_method_combo.currentText(),
        cl._fm_fid_ch_combo.currentText(),
        cl._fm_poi_ch_combo.currentText(),
    )


def test_a_new_volume_with_the_same_channels_changes_nothing(widget):
    """The interpolation case, exactly: same channels, refined z sampling.

    Every one of the five controls must read back what the user set.
    """
    widget.set_fm_image(_fm_image(["GFP", "mCherry"]))
    _set_all_five(
        widget, fib="Gaussian", fid_method="Hole", poi_method="None",
        fid_ch="mCherry", poi_ch="GFP",
    )
    before = _read_all_five(widget)

    widget.set_fm_image(_fm_image(["GFP", "mCherry"]))  # e.g. after interpolation

    assert _read_all_five(widget) == before


def test_the_method_combos_are_never_touched_by_an_image_change(widget):
    """Methods do not depend on the image at all — they are populated from
    _FIT_METHODS at construction — so even a volume with entirely different
    channels must leave them alone."""
    widget.set_fm_image(_fm_image(["GFP", "mCherry"]))
    _set_all_five(
        widget, fib="Gaussian", fid_method="Hole", poi_method="None",
        fid_ch="GFP", poi_ch="mCherry",
    )

    widget.set_fm_image(_fm_image(["DAPI", "Cy5", "TxRed"]))

    fib, fid_method, poi_method, _, _ = _read_all_five(widget)
    assert (fib, fid_method, poi_method) == ("Gaussian", "Hole", "None")


def test_a_channel_the_new_image_lacks_falls_back_rather_than_clearing(widget):
    """A selection that cannot be honoured lands on the new image's first
    channel. The combo must never end up empty — an empty channel is not a
    valid fit input, and nothing downstream checks for one."""
    widget.set_fm_image(_fm_image(["GFP", "mCherry"]))
    widget._coords_tab.set_channel_selection("mCherry", "mCherry")

    widget.set_fm_image(_fm_image(["DAPI", "Cy5"]))

    fid, poi = widget._coords_tab.channel_selection()
    assert fid in ("DAPI", "Cy5") and poi in ("DAPI", "Cy5")


def test_the_config_still_seeds_the_channels_on_first_load(widget):
    """The behaviour the old re-apply existed for, and which must not regress.

    `set_correlation_config` runs before any FM image, when the channel combos
    are empty and selecting by name cannot land. The first `set_fm_image` is
    where the config's choice has to take effect.
    """
    widget.set_correlation_config(
        CorrelationConfig(
            fit=FitSettings(fm_fiducial_channel="mCherry", fm_poi_channel="GFP")
        )
    )

    widget.set_fm_image(_fm_image(["GFP", "mCherry"]))

    assert widget._coords_tab.channel_selection() == ("mCherry", "GFP")


def test_the_users_choice_beats_the_config_on_a_later_image(widget):
    """Once the user has chosen, the config is history — it seeds, it does not
    keep asserting itself. This is the actual FIB-636 defect."""
    widget.set_correlation_config(
        CorrelationConfig(
            fit=FitSettings(fm_fiducial_channel="mCherry", fm_poi_channel="GFP")
        )
    )
    widget.set_fm_image(_fm_image(["GFP", "mCherry"]))
    widget._coords_tab.set_channel_selection("GFP", "mCherry")  # user swaps them

    widget.set_fm_image(_fm_image(["GFP", "mCherry"]))

    assert widget._coords_tab.channel_selection() == ("GFP", "mCherry")


def test_what_is_read_back_for_the_protocol_reflects_the_user(widget):
    """The consequence that makes this more than a display bug: the config read
    back on accept is persisted to protocol.correlation for every lamella."""
    widget.set_fm_image(_fm_image(["GFP", "mCherry"]))
    _set_all_five(
        widget, fib="Gaussian", fid_method="Hole", poi_method="None",
        fid_ch="mCherry", poi_ch="GFP",
    )

    widget.set_fm_image(_fm_image(["GFP", "mCherry"]))
    fit = widget.correlation_config.fit

    assert fit.fib_method == "Gaussian"
    assert fit.fm_fiducial_method == "Hole"
    assert fit.fm_poi_method == "None"
    assert fit.fm_fiducial_channel == "mCherry"
    assert fit.fm_poi_channel == "GFP"
