"""The tiled runner's orientation gate, on the mounting it was never run on.

`FMTiledAcquisitionRunner._setup` shipped with an inlined `["SEM", "FM"]` list, and
the whole tiled suite runs on the compustage simulator, where that list happens to
have an answer. At the FM on an offset mount the classifier reports the pose the
sample was carried out in -- "FIB", measured -- so the runner refused at the one
place it is meant to run, and nothing noticed.

This file is the pin from FIB-856. The gate here is interim (an inline mounting
split); when the gates move onto `get_device_imaging_state` (FIB-839) the gate
changes and these tests keep it honest.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.fm.acquisition import acquire_tileset
from fibsem.fm.structures import (
    AutoFocusMode,
    ChannelSettings,
    OverviewParameters,
    ZParameters,
)

IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")

CHANNEL = ChannelSettings(
    name="DAPI",
    excitation_wavelength=358,
    emission_wavelength=461,
    power=0.1,
    exposure_time=0.001,
)


def _overview(rows: int = 2, cols: int = 2, use_zstack: bool = False):
    return OverviewParameters(
        rows=rows,
        cols=cols,
        overlap=0.1,
        use_zstack=use_zstack,
        autofocus_mode=AutoFocusMode.NONE,
    )


def _at_the_fm():
    """An offset system parked at the FM: FIB pose, then the 48.8 mm traverse."""
    microscope, _ = utils.setup_session(config_path=IFLM_CONFIG)
    microscope.move_to_orientation("FIB")
    microscope.move_to_microscope("FM")
    return microscope


def test_an_offset_system_can_acquire_an_overview_at_the_fm():
    """The headline: this raised "Stage is not in SEM, or FM orientation FIB"."""
    microscope = _at_the_fm()

    tiles = acquire_tileset(
        microscope=microscope,
        channel_settings=CHANNEL,
        overview_parameters=_overview(),
    )

    assert len(tiles) == 2 and len(tiles[0]) == 2
    assert all(tile is not None for row in tiles for tile in row)


def test_the_stage_comes_back_to_where_the_overview_started():
    """The runner restores its starting position -- which is at the FM, not the beams."""
    microscope = _at_the_fm()
    start_x = microscope.get_stage_position().x

    acquire_tileset(
        microscope=microscope,
        channel_settings=CHANNEL,
        overview_parameters=_overview(rows=1, cols=1),
    )

    assert microscope.get_stage_position().x == pytest.approx(start_x)
    assert microscope.get_current_device() == "FM"


def test_zstacked_overviews_work_at_the_fm_too():
    microscope = _at_the_fm()

    tiles = acquire_tileset(
        microscope=microscope,
        channel_settings=CHANNEL,
        overview_parameters=_overview(rows=1, cols=1, use_zstack=True),
        zparams=ZParameters(zmin=-1e-6, zmax=1e-6, zstep=1e-6),
    )

    assert tiles[0][0] is not None


def test_an_offset_system_still_refuses_at_milling():
    """The gate is narrowed, not removed: MILLING is refused on both mountings."""
    microscope, _ = utils.setup_session(config_path=IFLM_CONFIG)
    microscope.move_to_orientation("MILLING")

    with pytest.raises(ValueError, match="MILLING"):
        acquire_tileset(
            microscope=microscope,
            channel_settings=CHANNEL,
            overview_parameters=_overview(),
        )


def test_a_compustage_keeps_exactly_the_list_it_had():
    """["SEM", "FM"] on a compustage, unchanged.

    The tempting one-liner -- swap the inlined list for
    `is_acquisition_orientation()` -- would have *newly refused* SEM tilesets on
    every Arctis, because that predicate's list is `["FM"]` there. The refusal at
    FIB below is the old behaviour too, kept.
    """
    microscope, _ = utils.setup_session(config_path=ARCTIS_CONFIG)
    microscope.fm.objective.insert()

    microscope.move_to_orientation("SEM")
    tiles = acquire_tileset(
        microscope=microscope,
        channel_settings=CHANNEL,
        overview_parameters=_overview(rows=1, cols=1),
    )
    assert tiles[0][0] is not None

    microscope.move_to_orientation("FIB")
    with pytest.raises(ValueError, match="FIB"):
        acquire_tileset(
            microscope=microscope,
            channel_settings=CHANNEL,
            overview_parameters=_overview(),
        )
